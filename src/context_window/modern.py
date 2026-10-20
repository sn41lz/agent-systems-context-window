"""Explicit current Responses controls and lossless handling of opaque provider state."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, TypedDict

from jsonschema import Draft202012Validator

from .models import canonical, text


class Usage(TypedDict):
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cache_write_tokens: int | None
    reasoning_tokens: int | None


@dataclass(frozen=True)
class RequestOptions:
    # default works with older models too; the other modes require provider support.
    cache_mode: str = "default"
    preflight: bool = False
    reasoning_effort: str | None = None
    answer_schema: dict[str, Any] | None = None

    def __post_init__(self):
        if self.cache_mode not in {"default", "implicit", "explicit", "off"}:
            raise ValueError("Invalid cache mode")
        if type(self.preflight) is not bool:
            raise ValueError("Preflight must be a boolean")
        if self.reasoning_effort not in {
            None,
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("Invalid reasoning effort")
        if self.answer_schema is not None:
            Draft202012Validator.check_schema(self.answer_schema)

    def prepare(self, items: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "input": copy.deepcopy(items),
            "tools": copy.deepcopy(tools),
            "tool_choice": "none",
        }
        if self.cache_mode != "default":
            result["prompt_cache_options"] = {
                "mode": "implicit" if self.cache_mode == "implicit" else "explicit",
                "ttl": "30m",
            }
        if self.cache_mode == "explicit":
            if not result["input"]:
                raise ValueError("An explicit prefix breakpoint requires a developer message")
            first = result["input"][0]
            if first.get("role") != "developer" or not isinstance(first.get("content"), str):
                raise ValueError(
                    "An explicit prefix breakpoint requires a leading text developer message"
                )
            first["content"] = [
                {
                    "type": "input_text",
                    "text": first["content"],
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ]
        if self.reasoning_effort is not None:
            result["reasoning"] = {"effort": self.reasoning_effort}
        if self.answer_schema is not None:
            result["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "current_constraints",
                    "strict": True,
                    "schema": copy.deepcopy(self.answer_schema),
                }
            }
        return result


@dataclass(frozen=True)
class NativeWindow:
    """Canonical compact output, not a human-readable summary or a portable model state."""

    model: str
    items_json: str
    tools_json: str
    prefix_sha256: str
    source_provider_tokens: int
    usage: Usage
    estimated_cost_usd: float | None

    def __post_init__(self):
        text(self.model, maximum=256)
        if not self.model.strip():
            raise ValueError("Native state requires an explicit model")
        text(self.items_json, maximum=4_194_304)
        text(self.tools_json)
        if not isinstance(json.loads(self.tools_json), list):
            raise ValueError("Invalid native tool catalogue")
        value = json.loads(self.items_json)
        if (
            not isinstance(value, list)
            or not 0 < len(value) <= 10_000
            or not all(isinstance(item, dict) for item in value)
        ):
            raise ValueError("Invalid native compaction window")
        if not any(
            item.get("type") == "compaction"
            and isinstance(item.get("encrypted_content"), str)
            and item["encrypted_content"]
            for item in value
        ):
            raise ValueError("Native compaction returned no opaque compaction item")
        # Ensure finite JSON and UTF-8 without selecting, reordering or pruning items.
        canonical(value).encode("utf-8")

    @property
    def items(self) -> list[dict[str, Any]]:
        return json.loads(self.items_json)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.items_json.encode()).hexdigest()

    def continuation(self, query: str) -> list[dict[str, Any]]:
        text(query)
        return [*self.items, {"role": "user", "content": query}]
