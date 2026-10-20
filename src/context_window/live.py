"""Raw Responses streaming, explicit timeouts and observed provider accounting."""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from typing import Any, cast

from openai import AsyncOpenAI

from .models import CompiledContext, canonical
from .modern import NativeWindow, RequestOptions, Usage
from .telemetry import Telemetry


class ProviderError(Exception):
    """Sanitized provider/configuration failure."""


@dataclass(frozen=True)
class Rates:
    input_per_million: float
    output_per_million: float
    cached_per_million: float
    label: str
    cache_write_per_million: float | None = None

    def __post_init__(self):
        if (
            any(
                not math.isfinite(rate) or rate < 0
                for rate in (
                    self.input_per_million,
                    self.output_per_million,
                    self.cached_per_million,
                )
            )
            or not self.label.strip()
            or (
                self.cache_write_per_million is not None
                and (
                    not math.isfinite(self.cache_write_per_million)
                    or self.cache_write_per_million < 0
                )
            )
        ):
            raise ValueError("Rates must be finite, nonnegative and explicitly labeled")

    def estimate(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        cache_write_tokens: int | None = 0,
    ) -> float | None:
        if any(
            type(value) is not int or value < 0
            for value in (input_tokens, output_tokens, cached_tokens)
        ):
            raise ValueError("Invalid usage counts")
        if cache_write_tokens is None:
            return None
        if (
            type(cache_write_tokens) is not int
            or cache_write_tokens < 0
            or cached_tokens + cache_write_tokens > input_tokens
        ):
            raise ValueError("Invalid cache accounting")
        if cache_write_tokens and self.cache_write_per_million is None:
            return None
        return (
            (input_tokens - cached_tokens - cache_write_tokens) * self.input_per_million
            + output_tokens * self.output_per_million
            + cached_tokens * self.cached_per_million
            + cache_write_tokens * (self.cache_write_per_million or 0)
        ) / 1_000_000


@dataclass(frozen=True)
class LiveSample:
    text: str
    ttft_ms: float | None
    duration_ms: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    estimated_cost_usd: float | None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    preflight_tokens: int | None = None
    preflight_ms: float = 0


def observed_usage(usage: Any) -> Usage:
    incoming, outgoing = usage.input_tokens, usage.output_tokens
    details = getattr(usage, "input_tokens_details", None)
    cached = getattr(details, "cached_tokens", None)
    written = getattr(details, "cache_write_tokens", None)
    reasoning = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", None)
    if any(type(value) is not int or value < 0 for value in (incoming, outgoing, cached)):
        raise ValueError("Invalid usage counts")
    if any(
        value is not None and (type(value) is not int or value < 0)
        for value in (written, reasoning)
    ):
        raise ValueError("Invalid detailed usage")
    cached = cast(int, cached)
    if cached + (written or 0) > incoming or (reasoning is not None and reasoning > outgoing):
        raise ValueError("Inconsistent usage counts")
    return {
        "input_tokens": incoming,
        "output_tokens": outgoing,
        "cached_tokens": cached,
        "cache_write_tokens": written,
        "reasoning_tokens": reasoning,
    }


class OpenAIStream:
    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        telemetry: Telemetry | None = None,
        rates: Rates | None = None,
        timeout_s: float = 60,
        options: RequestOptions | None = None,
    ):
        if (
            not isinstance(model, str)
            or not model.strip()
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("Model and timeout must be valid")
        self.model, self.rates, self.timeout_s = model, rates, timeout_s
        self.options = options or RequestOptions()
        self.telemetry = telemetry or Telemetry()
        self._owns_client = client is None
        try:
            self._client: Any = (
                AsyncOpenAI(max_retries=0) if client is None else client.with_options(max_retries=0)
            )
        except Exception:
            raise ProviderError("Invalid OpenAI client configuration") from None

    async def generate(self, context: CompiledContext, *, max_output_tokens: int) -> LiveSample:
        request = self.options.prepare(context.items, context.tools)
        return await self._stream(
            request,
            capacity=context.capacity,
            prefix=context.prefix_sha256,
            application_tokens=context.token_count,
            max_output_tokens=max_output_tokens,
        )

    async def generate_native(
        self, window: NativeWindow, query: str, *, capacity: int, max_output_tokens: int
    ) -> LiveSample:
        if window.model != self.model:
            raise ValueError("Opaque compaction state must stay with its original model")
        if self.options.cache_mode == "explicit":
            raise ValueError("Do not rewrite a native window to insert explicit breakpoints")
        request = self.options.prepare(window.continuation(query), json.loads(window.tools_json))
        return await self._stream(
            request,
            capacity=capacity,
            prefix=window.prefix_sha256,
            application_tokens=None,
            max_output_tokens=max_output_tokens,
            force_preflight=True,
        )

    async def _count(self, request: dict[str, Any], capacity: int) -> int:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Provider input capacity must be a positive integer")
        params = {
            key: value
            for key, value in request.items()
            if key in {"input", "tools", "tool_choice", "reasoning", "text"}
        }
        with self.telemetry.span("llm.token_count", {"gen_ai.request.model": self.model}) as span:
            result = await self._client.responses.input_tokens.count(model=self.model, **params)
            count = result.input_tokens
            if type(count) is not int or count < 0:
                raise ProviderError("Invalid provider input-token count")
            span.set_attribute("gen_ai.usage.input_tokens", count)
            if count > capacity:
                raise ProviderError("Provider input exceeds the reserved capacity")
            return count

    async def compact(self, context: CompiledContext) -> NativeWindow:
        # The compact endpoint has no tools/reasoning/text-format parameters.
        request = {
            key: value
            for key, value in self.options.prepare(context.items, []).items()
            if key in {"input", "prompt_cache_options"}
        }
        with self.telemetry.span(
            "context.native_compact", {"gen_ai.request.model": self.model}
        ) as span:
            try:
                async with asyncio.timeout(self.timeout_s):
                    count = await self._count({"input": request["input"]}, context.capacity)
                    result = await self._client.responses.compact(model=self.model, **request)
                    usage = observed_usage(result.usage)
                    # exclude_unset preserves exactly provider-supplied fields, including unknown ones.
                    items = [
                        item.model_dump(mode="json", exclude_unset=True, warnings=False)
                        if hasattr(item, "model_dump")
                        else item
                        for item in result.output
                    ]
                    estimate = (
                        self.rates.estimate(
                            usage["input_tokens"],
                            usage["output_tokens"],
                            usage["cached_tokens"],
                            usage["cache_write_tokens"],
                        )
                        if self.rates
                        else None
                    )
                    window = NativeWindow(
                        self.model,
                        canonical(items),
                        context.tools_json,
                        context.prefix_sha256,
                        count,
                        usage,
                        estimate,
                    )
                    for key, value in usage.items():
                        if isinstance(value, int):
                            span.set_attribute("gen_ai.usage." + key, value)
                    return window
            except asyncio.CancelledError:
                raise
            except Exception:
                raise ProviderError(
                    "Native compaction failed or exceeded the reserved capacity"
                ) from None

    async def _stream(
        self,
        request: dict[str, Any],
        *,
        capacity: int,
        prefix: str,
        application_tokens: int | None,
        max_output_tokens: int,
        force_preflight: bool = False,
    ) -> LiveSample:
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("Output token reservation must be a positive integer")
        started = time.perf_counter()
        first = None
        fragments = []
        output_bytes = 0
        completed = None
        preflight_tokens = None
        preflight_ms = 0.0
        with self.telemetry.span(
            "llm.call",
            {
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": self.model,
                "gen_ai.request.max_tokens": max_output_tokens,
                "gen_ai.request.retry_count": 0,
                "context.cache_mode": self.options.cache_mode,
                **(
                    {"gen_ai.request.reasoning_effort": self.options.reasoning_effort}
                    if self.options.reasoning_effort is not None
                    else {}
                ),
                "context.prefix_sha256": prefix,
                **(
                    {"context.application_tokens": application_tokens}
                    if application_tokens is not None
                    else {}
                ),
            },
        ) as span:
            try:
                async with asyncio.timeout(self.timeout_s):
                    if self.options.preflight or force_preflight:
                        count_started = time.perf_counter()
                        preflight_tokens = await self._count(request, capacity)
                        preflight_ms = (time.perf_counter() - count_started) * 1000
                    generation_started = time.perf_counter()
                    stream = await self._client.responses.create(
                        model=self.model,
                        **request,
                        stream=True,
                        store=False,
                        max_output_tokens=max_output_tokens,
                        prompt_cache_key="cw-" + prefix[:32],
                        truncation="disabled",
                    )
                    try:
                        async for event in stream:
                            if event.type == "response.output_text.delta":
                                if completed is not None:
                                    raise ProviderError("Text arrived after terminal completion")
                                if not isinstance(event.delta, str):
                                    raise ProviderError("Invalid streaming text")
                                if event.delta:
                                    output_bytes += len(event.delta.encode("utf-8"))
                                    if output_bytes > 1_048_576:
                                        raise ProviderError("Stream output exceeds the byte limit")
                                    if first is None:
                                        first = (time.perf_counter() - generation_started) * 1000
                                    fragments.append(event.delta)
                            elif event.type == "response.completed":
                                if completed is not None:
                                    raise ProviderError("Duplicate terminal completion")
                                completed = event.response
                            elif event.type in {"response.failed", "response.incomplete", "error"}:
                                raise ProviderError("Provider stream did not complete")
                    finally:
                        await stream.close()
                if completed is None or completed.status != "completed":
                    raise ProviderError("Provider stream ended without completion")
                usage = observed_usage(completed.usage)
                incoming, outgoing, cached = (
                    usage["input_tokens"],
                    usage["output_tokens"],
                    usage["cached_tokens"],
                )
                output = "".join(fragments)
                output.encode("utf-8")
                if not output.strip():
                    raise ProviderError("Provider returned no text")
            except asyncio.CancelledError:
                span.set_attribute("error.type", "cancelled")
                raise
            except Exception:
                span.set_attribute("error.type", "provider_failure")
                raise ProviderError(
                    "Provider request failed, timed out, or returned an invalid stream"
                ) from None
            estimate = (
                self.rates.estimate(incoming, outgoing, cached, usage["cache_write_tokens"])
                if self.rates
                else None
            )
            span.set_attribute("gen_ai.usage.input_tokens", incoming)
            span.set_attribute("gen_ai.usage.output_tokens", outgoing)
            span.set_attribute("gen_ai.usage.cache_read.input_tokens", cached)
            if usage["cache_write_tokens"] is not None:
                span.set_attribute(
                    "gen_ai.usage.cache_write.input_tokens", usage["cache_write_tokens"]
                )
            if usage["reasoning_tokens"] is not None:
                span.set_attribute("gen_ai.usage.reasoning_tokens", usage["reasoning_tokens"])
            if first is not None:
                span.set_attribute("gen_ai.ttft_ms", first)
            if estimate is not None:
                span.set_attribute("gen_ai.estimated_cost_usd", estimate)
            return LiveSample(
                output,
                first,
                (time.perf_counter() - started) * 1000,
                incoming,
                outgoing,
                cached,
                estimate,
                usage["cache_write_tokens"],
                usage["reasoning_tokens"],
                preflight_tokens,
                preflight_ms,
            )

    async def aclose(self):
        if self._owns_client:
            await self._client.close()
