"""Stream semantics and actual SDK wire parsing, with zero provider requests."""

import asyncio
import json
from types import SimpleNamespace as N

import httpx2 as httpx
import pytest
from openai import AsyncOpenAI, AuthenticationError

from context_window.engine import ContextEngine
from context_window.live import OpenAIStream, ProviderError, Rates
from context_window.telemetry import Telemetry


def done(*, incoming=100, outgoing=20, cached=40, status="completed"):
    return N(
        type="response.completed",
        response=N(
            status=status,
            usage=N(
                input_tokens=incoming,
                output_tokens=outgoing,
                input_tokens_details=N(cached_tokens=cached, cache_write_tokens=0),
                output_tokens_details=N(reasoning_tokens=0),
            ),
        ),
    )


class FakeStream:
    def __init__(self, events, delay=0):
        self.events = events
        self.delay = delay
        self.closed = False

    async def __aiter__(self):
        for event in self.events:
            if self.delay:
                await asyncio.sleep(self.delay)
            if isinstance(event, Exception):
                raise event
            yield event

    async def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, stream=None, error=None):
        self.stream = stream
        self.error = error
        self.requests = []
        self.responses = self
        self.options = []
        self.closed = False

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return self.stream

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_first_nonempty_delta_and_real_usage_are_recorded_without_hidden_retries():
    stream = FakeStream(
        [
            N(type="response.created"),
            N(type="response.output_text.delta", delta=""),
            N(type="response.output_text.delta", delta="hello "),
            N(type="response.output_text.delta", delta="world"),
            done(),
        ],
        delay=0.001,
    )
    client = FakeClient(stream)
    with Telemetry() as telemetry:
        provider = OpenAIStream(
            "chosen-model",
            client=client,
            telemetry=telemetry,
            rates=Rates(2, 8, 1, "caller-test-rates"),
        )
        sample = await provider.generate(
            ContextEngine().compile("private query"), max_output_tokens=512
        )
        await provider.aclose()
    assert sample.text == "hello world" and 0 < sample.ttft_ms < sample.duration_ms
    assert sample.input_tokens == 100 and sample.output_tokens == 20 and sample.cached_tokens == 40
    assert sample.estimated_cost_usd == pytest.approx(0.00032)
    assert client.options == [{"max_retries": 0}] and len(client.requests) == 1
    assert client.requests[0]["stream"] and client.requests[0]["store"] is False
    assert client.requests[0]["tool_choice"] == "none"
    assert stream.closed and not client.closed
    span = telemetry.events[0]
    assert span["attributes"]["gen_ai.ttft_ms"] == sample.ttft_ms
    assert "private query" not in json.dumps(telemetry.events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        [],
        [N(type="response.output_text.delta", delta="partial")],
        [N(type="response.failed")],
        [N(type="response.incomplete")],
        [N(type="error")],
        [N(type="response.output_text.delta", delta="hi"), done(status="incomplete")],
        [N(type="response.output_text.delta", delta="hi"), done(cached=101)],
        [N(type="response.output_text.delta", delta="hi"), done(incoming=-1)],
        [N(type="response.output_text.delta", delta=123), done()],
        [N(type="response.output_text.delta", delta=chr(0xD800)), done()],
        [done()],
    ],
)
async def test_partial_invalid_or_empty_stream_never_becomes_success(events):
    stream = FakeStream(events)
    with Telemetry() as telemetry:
        provider = OpenAIStream("model", client=FakeClient(stream), telemetry=telemetry)
        with pytest.raises(ProviderError):
            await provider.generate(ContextEngine().compile("query"), max_output_tokens=512)
    assert stream.closed
    assert telemetry.events[0]["status"] == "ERROR"


@pytest.mark.asyncio
async def test_timeout_and_cancellation_close_stream_and_propagate_correctly():
    for cancel in (False, True):
        stream = FakeStream([N(type="response.output_text.delta", delta="late"), done()], delay=1)
        provider = OpenAIStream(
            "model", client=FakeClient(stream), timeout_s=0.01 if not cancel else 20
        )
        task = asyncio.create_task(
            provider.generate(ContextEngine().compile("q"), max_output_tokens=10)
        )
        if cancel:
            await asyncio.sleep(0.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(ProviderError):
                await task
        assert stream.closed


@pytest.mark.asyncio
async def test_api_exception_details_are_sanitized():
    request = httpx.Request("POST", "https://example.invalid")
    error = AuthenticationError(
        "credential-private",
        response=httpx.Response(401, request=request),
        body={"error": "secret"},
    )
    provider = OpenAIStream("model", client=FakeClient(error=error))
    with pytest.raises(ProviderError) as exc:
        await provider.generate(ContextEngine().compile("q"), max_output_tokens=10)
    assert "secret" not in str(exc.value) and "credential-private" not in str(exc.value)


def test_malformed_sdk_endpoint_is_sanitized(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-local-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid:credential-private")
    with pytest.raises(ProviderError) as exc:
        OpenAIStream("model")
    assert "credential-private" not in str(exc.value)


@pytest.mark.asyncio
async def test_actual_sdk_parses_sse_events_through_mock_transport():
    payload = {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "test-model",
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "wire success", "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 5,
            "total_tokens": 105,
            "input_tokens_details": {"cached_tokens": 32},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    events = [
        {
            "type": "response.output_text.delta",
            "item_id": "msg_test",
            "output_index": 0,
            "content_index": 0,
            "delta": "wire success",
            "sequence_number": 1,
        },
        {"type": "response.completed", "response": payload, "sequence_number": 2},
    ]

    async def transport(request):
        body = json.loads(request.content)
        assert body["stream"] is True and body["model"] == "test-model"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join("data: " + json.dumps(event) + "\n\n" for event in events),
        )

    client = AsyncOpenAI(
        api_key="dummy",
        base_url="https://example.invalid",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    try:
        sample = await OpenAIStream("test-model", client=client).generate(
            ContextEngine().compile("query"), max_output_tokens=32
        )
        assert (
            sample.text == "wire success"
            and sample.cached_tokens == 32
            and sample.estimated_cost_usd is None
        )
    finally:
        await client.close()


@pytest.mark.parametrize("rates", [(-1, 1, 1, "bad"), (1, float("nan"), 1, "bad"), (1, 1, 1, "")])
def test_invalid_price_assumptions_rejected(rates):
    with pytest.raises(ValueError):
        Rates(*rates)
