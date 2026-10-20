"""Current SDK wire contracts without credentials, remote traffic or fake latency claims."""

import asyncio
import json
from types import SimpleNamespace as N

import httpx2 as httpx
import pytest
from openai import AsyncOpenAI

from context_window.engine import ContextEngine
from context_window.live import OpenAIStream, ProviderError, Rates, observed_usage
from context_window.models import canonical
from context_window.modern import NativeWindow, RequestOptions
from context_window.quality import ANSWER_SCHEMA, grade_answer
from context_window.scenarios import TOOLS, conversation
from context_window.telemetry import Telemetry


def usage(*, written=20, reasoning=3):
    return {
        "input_tokens": 100,
        "output_tokens": 10,
        "total_tokens": 110,
        "input_tokens_details": {"cached_tokens": 40, "cache_write_tokens": written},
        "output_tokens_details": {"reasoning_tokens": reasoning},
    }


def sse():
    response = {
        "id": "resp_wire",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "chosen-model",
        "output": [],
        "usage": usage(),
    }
    events = [
        {"type": "response.output_text.delta", "delta": "answer", "sequence_number": 1},
        {"type": "response.completed", "response": response, "sequence_number": 2},
    ]
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join("data: " + json.dumps(event) + "\n\n" for event in events),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["default", "implicit", "explicit", "off"])
async def test_current_options_and_counting_survive_actual_sdk_serialization(mode):
    requests = []
    context = ContextEngine(tools=TOOLS).compile("query")
    original = context.items

    async def transport(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path.endswith("/input_tokens"):
            assert "prompt_cache_options" not in body
            return httpx.Response(
                200, json={"object": "response.input_tokens", "input_tokens": 100}
            )
        return sse()

    client = AsyncOpenAI(
        api_key="local-dummy",
        base_url="https://example.invalid",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    try:
        options = RequestOptions(mode, True, "low", ANSWER_SCHEMA)
        result = await OpenAIStream(
            "chosen-model", client=client, options=options, rates=Rates(2, 8, 1, "test only", 2.5)
        ).generate(context, max_output_tokens=512)
    finally:
        await client.close()
    assert len(requests) == 2
    body = requests[-1][1]
    assert body["text"]["format"]["schema"] == ANSWER_SCHEMA
    assert body["reasoning"] == {"effort": "low"} and body["truncation"] == "disabled"
    assert context.items == original
    assert ("prompt_cache_options" in body) == (mode != "default")
    if mode == "explicit":
        assert body["input"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    else:
        assert isinstance(body["input"][0]["content"], str)
    if mode == "off":
        assert body["prompt_cache_options"]["mode"] == "explicit"
        assert "prompt_cache_breakpoint" not in canonical(body)
    assert result.cache_write_tokens == 20 and result.reasoning_tokens == 3
    assert result.preflight_tokens == 100 and result.preflight_ms > 0
    assert result.estimated_cost_usd == pytest.approx(0.00025)


@pytest.mark.asyncio
async def test_native_complete_output_is_forwarded_without_pruning_or_reordering():
    # Includes retained messages, opaque state, reasoning and a future provider field.
    output = [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "retained", "annotations": []}],
        },
        {
            "type": "compaction",
            "id": "cmp_1",
            "encrypted_content": "opaque-provider-state",
            "future_field": {"keep": True},
        },
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque-reasoning"},
        {"role": "user", "content": "retained original user"},
    ]
    requests = []

    async def transport(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(
                200, json={"object": "response.input_tokens", "input_tokens": 100}
            )
        if request.url.path.endswith("/compact"):
            assert set(body) == {"input", "model"}
            return httpx.Response(
                200,
                json={
                    "id": "cmp_response",
                    "object": "response.compaction",
                    "created_at": 0,
                    "output": output,
                    "usage": usage(),
                },
            )
        return sse()

    client = AsyncOpenAI(
        api_key="local-dummy",
        base_url="https://example.invalid",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    with Telemetry() as telemetry:
        try:
            provider = OpenAIStream("chosen-model", client=client, telemetry=telemetry)
            context = ContextEngine(tools=TOOLS).compile("handoff query")
            window = await provider.compact(context)
            assert window.items == output
            window.items.clear()
            assert window.items == output
            result = await provider.generate_native(
                window, "continue", capacity=3456, max_output_tokens=512
            )
        finally:
            await client.close()
    assert result.preflight_tokens == 100
    assert [path for path, _ in requests] == [
        "/responses/input_tokens",
        "/responses/compact",
        "/responses/input_tokens",
        "/responses",
    ]
    for _, body in requests[2:]:
        assert body["input"] == [*output, {"role": "user", "content": "continue"}]
        assert body["tools"] == context.tools
    assert window.source_provider_tokens == 100 and len(window.digest) == 64
    traces = canonical(telemetry.events)
    assert "opaque-provider-state" not in traces and "retained original user" not in traces


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [999_999, -1, True])
async def test_preflight_fails_before_generation_for_overflow_or_invalid_count(count):
    calls = []

    async def counter(**kwargs):
        calls.append(kwargs)
        return N(input_tokens=count)

    client = N(with_options=lambda **kwargs: N(responses=N(input_tokens=N(count=counter))))
    provider = OpenAIStream("model", client=client, options=RequestOptions(preflight=True))
    with pytest.raises(ProviderError):
        await provider.generate(ContextEngine().compile("q"), max_output_tokens=10)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_native_error_timeout_and_cancellation_preserve_boundaries():
    async def blocked(**kwargs):
        await asyncio.sleep(1)

    for cancel in (False, True):
        client = N(with_options=lambda **kwargs: N(responses=N(input_tokens=N(count=blocked))))
        provider = OpenAIStream("model", client=client, timeout_s=10 if cancel else 0.001)
        task = asyncio.create_task(provider.compact(ContextEngine().compile("q")))
        if cancel:
            await asyncio.sleep(0.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(ProviderError):
                await task


def test_native_rejects_nonopaque_windows_and_invalid_shape():
    for value in (
        [],
        {},
        [{"type": "compaction", "encrypted_content": ""}],
        [{"role": "user", "content": "summary"}],
    ):
        with pytest.raises(ValueError):
            NativeWindow("model", canonical(value), "[]", "hash", 10, {}, None)


@pytest.mark.asyncio
async def test_native_rejects_cross_model_state_and_inserting_breakpoints():
    window = NativeWindow(
        "original",
        canonical([{"type": "compaction", "encrypted_content": "x"}]),
        "[]",
        "hash",
        10,
        {},
        None,
    )
    client = N(with_options=lambda **kwargs: N())
    for model, options in (("other", RequestOptions()), ("original", RequestOptions("explicit"))):
        with pytest.raises(ValueError):
            await OpenAIStream(model, client=client, options=options).generate_native(
                window, "q", capacity=300, max_output_tokens=10
            )


def test_write_cost_is_disjoint_and_unknown_fields_never_become_zero():
    rates = Rates(2, 8, 1, "test", 2.5)
    assert rates.estimate(100, 10, 40, 20) == pytest.approx(0.00025)
    assert rates.estimate(100, 10, 40, None) is None
    assert Rates(2, 8, 1, "test").estimate(100, 10, 40, 20) is None
    unknown = observed_usage(
        N(input_tokens=100, output_tokens=10, input_tokens_details=N(cached_tokens=40))
    )
    assert unknown["cache_write_tokens"] is None and unknown["reasoning_tokens"] is None
    for args in ((100, 10, 40, 61), (100, 10, 40, -1), (True, 10, 0, 0)):
        with pytest.raises(ValueError):
            rates.estimate(*args)
    with pytest.raises(ValueError):
        Rates(1, 1, 1, "label", float("nan"))
    for written, reasoning in ((61, 0), (0, 11), (-1, 0)):
        with pytest.raises(ValueError):
            observed_usage(
                N(
                    input_tokens=100,
                    output_tokens=10,
                    input_tokens_details=N(cached_tokens=40, cache_write_tokens=written),
                    output_tokens_details=N(reasoning_tokens=reasoning),
                )
            )


@pytest.mark.parametrize(
    "kwargs", [{"cache_mode": "bad"}, {"preflight": 1}, {"reasoning_effort": "bad"}]
)
def test_invalid_request_options(kwargs):
    with pytest.raises(ValueError):
        RequestOptions(**kwargs)


def test_explicit_mode_requires_a_stable_leading_policy():
    for items in ([], [{"role": "user", "content": "q"}]):
        with pytest.raises(ValueError):
            RequestOptions("explicit").prepare(items, [])


def test_answer_grading_checks_revisions_sources_duplicates_and_invalid_json():
    engine = ContextEngine()
    for turn, updates in conversation(120):
        engine.ingest(turn, updates)
        engine.compile("q")
    constraints = [
        {"key": fact.key, "value": fact.value, "source_turn": fact.source_turn}
        for fact in engine.facts
    ]
    answer = {"constraints": constraints, "missing": []}
    assert grade_answer(canonical(answer), engine.facts)["exact_match"]
    budget = next(item for item in constraints if item["key"] == "budget")
    budget["value"] = "EUR 48000"
    score = grade_answer(canonical(answer), engine.facts)
    assert not score["exact_match"] and score["recall"] == pytest.approx(5 / 6)
    budget["value"] = "EUR 42000"
    budget["source_turn"] = "turn-0001"
    assert not grade_answer(canonical(answer), engine.facts)["exact_match"]
    budget["source_turn"] = "turn-0047"
    constraints.append(dict(budget))
    score = grade_answer(canonical(answer), engine.facts)
    assert score["duplicate_keys"] == 1 and score["correct"] == 5
    for invalid in ('{"constraints":[],"constraints":[],"missing":[]}', "[]", "garbage"):
        assert not grade_answer(invalid, engine.facts)["schema_valid"]
