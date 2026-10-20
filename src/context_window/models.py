"""Immutable protocol records and explicit task-state contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
KEY = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )


def text(value: str, *, maximum: int = 262_144) -> None:
    if not isinstance(value, str):
        raise ValueError("Content must be text")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError:
        raise ValueError("Content must be valid Unicode") from None
    if size > maximum:
        raise ValueError("Content exceeds its byte limit")


def strict_json(value: str) -> Any:
    text(value, maximum=65_536)

    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("Duplicate JSON keys are forbidden")
            result[key] = item
        return result

    def reject(_value):
        raise ValueError("JSON numbers must be finite")

    try:
        result = json.loads(value, object_pairs_hook=pairs, parse_constant=reject)
        pending = [(result, 0)]
        while pending:
            item, depth = pending.pop()
            if depth > 48:
                raise ValueError("JSON nesting limit exceeded")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
            if isinstance(item, str):
                text(item)
            elif isinstance(item, dict):
                pending.extend((child, depth + 1) for pair in item.items() for child in pair)
            elif isinstance(item, list):
                pending.extend((child, depth + 1) for child in item)
        return result
    except (UnicodeError, RecursionError):
        raise ValueError("Invalid JSON encoding or nesting") from None


@dataclass(frozen=True)
class ToolExchange:
    call_id: str
    name: str
    arguments: str
    output: str

    def __post_init__(self):
        if not isinstance(self.call_id, str) or not ID.fullmatch(self.call_id):
            raise ValueError("Invalid tool call ID")
        if not isinstance(self.name, str) or not KEY.fullmatch(self.name):
            raise ValueError("Invalid tool name")
        if not isinstance(strict_json(self.arguments), dict):
            raise ValueError("Tool arguments must be a JSON object")
        strict_json(self.output)


@dataclass(frozen=True)
class Turn:
    turn_id: str
    user: str
    assistant: str
    tools: tuple[ToolExchange, ...] = ()

    def __post_init__(self):
        if not isinstance(self.turn_id, str) or not ID.fullmatch(self.turn_id):
            raise ValueError("Invalid turn ID")
        text(self.user)
        text(self.assistant)
        if (
            not isinstance(self.tools, tuple)
            or len(self.tools) > 32
            or any(not isinstance(item, ToolExchange) for item in self.tools)
        ):
            raise ValueError("Tool exchanges must be a bounded tuple")
        if len({item.call_id for item in self.tools}) != len(self.tools):
            raise ValueError("Duplicate tool call IDs")
        text(canonical(asdict(self)))

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical(asdict(self)).encode()).hexdigest()

    def items(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = [{"role": "user", "content": self.user}]
        result.extend(
            {
                "type": "function_call",
                "call_id": tool.call_id,
                "name": tool.name,
                "arguments": canonical(strict_json(tool.arguments)),
            }
            for tool in self.tools
        )
        result.extend(
            {
                "type": "function_call_output",
                "call_id": tool.call_id,
                "output": canonical(strict_json(tool.output)),
            }
            for tool in self.tools
        )
        if self.assistant:
            result.append({"role": "assistant", "content": self.assistant})
        return result


@dataclass(frozen=True)
class FactUpdate:
    key: str
    value: str
    expected_revision: int | None = None

    def __post_init__(self):
        if not isinstance(self.key, str) or not KEY.fullmatch(self.key):
            raise ValueError("Invalid fact key")
        text(self.value, maximum=2048)
        if self.expected_revision is not None and (
            type(self.expected_revision) is not int or self.expected_revision < 0
        ):
            raise ValueError("Expected revision must be a nonnegative integer")


@dataclass(frozen=True)
class TaskFact:
    key: str
    value: str
    revision: int
    source_turn: str
    source_sha256: str


@dataclass(frozen=True)
class Budget:
    context_tokens: int = 4096
    output_reserve: int = 512
    framing_reserve: int = 128
    min_recent_turns: int = 2
    chunk_size: int = 4
    excerpt_tokens: int = 48
    digest_tokens: int = 16
    trigger_ratio: float = 0.85
    max_turns: int = 10_000

    def __post_init__(self):
        for name in (
            "context_tokens",
            "output_reserve",
            "framing_reserve",
            "min_recent_turns",
            "chunk_size",
            "excerpt_tokens",
            "digest_tokens",
            "max_turns",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.trigger_ratio) or not 0.5 <= self.trigger_ratio <= 1:
            raise ValueError("trigger_ratio must be finite and in [0.5, 1]")
        if self.input_capacity <= 0:
            raise ValueError("Output and framing reservations exhaust the context")

    @property
    def input_capacity(self) -> int:
        return self.context_tokens - self.output_reserve - self.framing_reserve


class BudgetExceeded(ValueError):
    def __init__(self, required: int, capacity: int):
        self.required = required
        self.capacity = capacity
        super().__init__(
            f"Protected context requires {required} application tokens; capacity is {capacity}"
        )


class RevisionConflict(ValueError):
    """The caller's task-state revision is stale; no part of ingestion commits."""


@dataclass(frozen=True)
class CompiledContext:
    items_json: str
    tools_json: str
    token_count: int
    capacity: int
    prefix_sha256: str
    section_tokens: dict[str, int]
    retained_turns: tuple[str, ...]
    archive_chunks: int
    compacted_turns: int
    omitted_turns: int
    events: tuple[dict[str, Any], ...] = ()

    @property
    def items(self) -> list[dict[str, Any]]:
        return json.loads(self.items_json)

    @property
    def tools(self) -> list[dict[str, Any]]:
        return json.loads(self.tools_json)

    def request(self) -> dict[str, Any]:
        return {"input": self.items, "tools": self.tools}

    def metadata(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"items_json", "tools_json"}
        }
