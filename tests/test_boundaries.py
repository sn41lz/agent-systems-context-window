from pathlib import Path

import pytest

import context_window.tokenizer as tokenizer
from context_window.engine import ContextEngine
from context_window.models import FactUpdate, Turn


def test_tokenizer_integrity_failure_is_detected_before_encoding(tmp_path, monkeypatch):
    original = Path(tokenizer.__file__).with_name("data")
    data = tmp_path / "data"
    data.mkdir()
    (data / "encoding.json").write_text((original / "encoding.json").read_text())
    (data / "o200k_base.tiktoken.gz").write_bytes(b"corrupt")
    monkeypatch.setattr(tokenizer, "__file__", str(tmp_path / "tokenizer.py"))
    tokenizer._encoding.cache_clear()
    try:
        with pytest.raises(ValueError, match="integrity"):
            tokenizer.TokenCounter()
    finally:
        tokenizer._encoding.cache_clear()


def test_tokenizer_rejects_unbounded_source_and_invalid_clip():
    counter = tokenizer.TokenCounter()
    with pytest.raises(ValueError):
        counter.clip("value", -1)
    with pytest.raises(ValueError):
        counter.count("x" * 8_388_609)


def test_fact_count_and_raw_buffer_are_bounded_transactionally():
    engine = ContextEngine()
    with pytest.raises(ValueError, match="fact limit"):
        engine.ingest(Turn("many", "a", "b"), tuple(FactUpdate(f"key{i}", "v") for i in range(65)))
    assert not engine.facts
    engine.ingest(Turn("many", "a", "b"), (FactUpdate("one", "v"),))
    for index in range(9):
        engine.ingest(Turn(f"large{index}", "x" * 220_000, "b"))
    with pytest.raises(ValueError, match="buffer"):
        engine.ingest(Turn("overflow", "x" * 220_000, "b"))
    # A rejected ingestion has not consumed its ID.
    engine.ingest(Turn("overflow", "small", "small"))
