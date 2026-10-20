"""A pinned, local tokenizer: no network access or writable cache at runtime."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import tiktoken

from .models import canonical, text


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    directory = Path(__file__).with_name("data")
    spec = json.loads((directory / "encoding.json").read_text())
    compressed = (directory / "o200k_base.tiktoken.gz").read_bytes()
    if hashlib.sha256(compressed).hexdigest() != spec["compressed_sha256"]:
        raise ValueError("Tokenizer asset failed its integrity check")
    data = gzip.decompress(compressed)
    if hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError("Tokenizer vocabulary failed its integrity check")
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in data.splitlines())
    }
    return tiktoken.Encoding(
        name=spec["name"],
        pat_str=spec["pat_str"],
        mergeable_ranks=ranks,
        special_tokens=spec["special_tokens"],
    )


class TokenCounter:
    """Exact o200k token count of canonical application JSON, not provider framing."""

    name = "o200k_base"

    def __init__(self):
        self.encoding = _encoding()

    def tokens(self, value: str) -> list[int]:
        text(value, maximum=8_388_608)
        return self.encoding.encode(value, disallowed_special=())

    def count(self, value: str) -> int:
        return len(self.tokens(value))

    def request(self, items: list[dict], tools: list[dict]) -> int:
        return self.count(canonical({"input": items, "tools": tools}))

    def clip(self, value: str, limit: int) -> str:
        if type(limit) is not int or limit < 0:
            raise ValueError("Clip limit must be a nonnegative integer")
        tokens = self.tokens(value)
        if len(tokens) <= limit:
            return value
        candidate = self.encoding.decode_bytes(tokens[:limit]).decode("utf-8", errors="ignore")
        while self.count(candidate) > limit:
            candidate = candidate[:-1]
        return candidate
