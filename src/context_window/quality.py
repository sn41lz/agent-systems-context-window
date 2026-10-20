"""Deterministic checks against scenario ground truth, independent of an LLM judge."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from .models import TaskFact, strict_json

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "constraints": {
            "type": "array",
            "maxItems": 64,
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "source_turn": {"type": "string"},
                },
                "required": ["key", "value", "source_turn"],
                "additionalProperties": False,
            },
        },
        "missing": {"type": "array", "maxItems": 64, "items": {"type": "string"}},
    },
    "required": ["constraints", "missing"],
    "additionalProperties": False,
}


def grade_answer(answer: str, facts: tuple[TaskFact, ...]) -> dict[str, Any]:
    """Exact key/value/source checks; wording equivalence is deliberately not inferred."""
    expected = {fact.key: asdict(fact) for fact in facts}
    try:
        parsed = strict_json(answer)
        Draft202012Validator(ANSWER_SCHEMA).validate(parsed)
    except (ValueError, ValidationError):
        return {"schema_valid": False, "exact_match": False, "precision": 0.0, "recall": 0.0}
    correct = set()
    seen = set()
    duplicates = set()
    incorrect = []
    for item in parsed["constraints"]:
        key = item["key"]
        if key in seen:
            duplicates.add(key)
        seen.add(key)
        if key in expected and all(
            item[field] == expected[key][field] for field in ("value", "source_turn")
        ):
            correct.add(key)
        else:
            incorrect.append(key)
    # A duplicate contradictory claim cannot earn credit for that key.
    correct -= duplicates
    count = len(parsed["constraints"])
    return {
        "schema_valid": True,
        "exact_match": len(correct) == len(expected)
        and count == len(expected)
        and not parsed["missing"],
        "precision": len(correct) / count if count else float(not expected),
        "recall": len(correct) / len(expected) if expected else 1.0,
        "correct": len(correct),
        "expected": len(expected),
        "incorrect_claims": len(incorrect),
        "duplicate_keys": len(duplicates),
        "missing_keys": sorted(expected.keys() - correct),
        "declared_missing": len(parsed["missing"]),
    }
