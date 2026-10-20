import pytest

from context_window.models import Budget, FactUpdate, ToolExchange, Turn, strict_json
from context_window.tokenizer import TokenCounter


@pytest.mark.parametrize(
    "value",
    [
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":1e999}',
        '{"x":1,"x":2}',
        '{"x":"\\ud800"}',
        "[" * 60 + "0" + "]" * 60,
    ],
)
def test_invalid_json_is_rejected(value):
    with pytest.raises(ValueError):
        strict_json(value)


@pytest.mark.parametrize(
    "content",
    [
        "hello",
        "A very long sentence " * 30,
        "中文🙂é " * 30,
        "<|endoftext|> is user data, not a special token",
    ],
)
def test_local_tokenization_and_unicode_safe_clipping(content, monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("Tokenizer attempted networking")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    counter = TokenCounter()
    assert counter.count(content) > 0
    for limit in (0, 1, 2, 3, 10, 100):
        clipped = counter.clip(content, limit)
        assert content.startswith(clipped)
        assert counter.count(clipped) <= limit
        clipped.encode("utf-8")
    assert counter.clip(content, counter.count(content)) == content


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context_tokens": 0},
        {"output_reserve": True},
        {"framing_reserve": -1},
        {"trigger_ratio": float("nan")},
        {"trigger_ratio": 0.1},
        {"context_tokens": 600},
    ],
)
def test_budget_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        Budget(**kwargs)


@pytest.mark.parametrize(
    "creator",
    [
        lambda: Turn("invalid id", "a", "b"),
        lambda: Turn("id", chr(0xD800), "a"),
        lambda: Turn("id", "a", "b", []),
        lambda: FactUpdate("Bad Key", "value"),
        lambda: FactUpdate("key", "value", -1),
        lambda: ToolExchange("id", "read", "[]", "{}"),
        lambda: ToolExchange("bad id", "read", "{}", "{}"),
    ],
)
def test_invalid_protocol_records_fail_at_construction(creator):
    with pytest.raises(ValueError):
        creator()


def test_duplicate_tool_ids_and_large_payloads_rejected():
    tool = ToolExchange("id", "read", "{}", "{}")
    with pytest.raises(ValueError):
        Turn("id", "a", "b", (tool, tool))
    with pytest.raises(ValueError):
        Turn("id", "a" * 300_000, "b")
    with pytest.raises(ValueError):
        FactUpdate("key", "a" * 2049)
    assert strict_json('{"x":"valid"}') == {"x": "valid"}
