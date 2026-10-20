"""Matched workload comparisons; offline metrics never masquerade as inference."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import platform
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .engine import ContextEngine
from .models import Budget, CompiledContext, TaskFact
from .scenarios import QUERY, TOOLS, conversation
from .telemetry import Telemetry
from .tokenizer import TokenCounter


def percentile(values: list[float], q: float) -> float:
    if not values or not 0 <= q <= 1:
        raise ValueError("Percentiles require samples and q in [0,1]")
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)


def available(context: CompiledContext, facts: tuple[TaskFact, ...]) -> int:
    # Independent check against actual rendered content, not internal retained IDs.
    content = "\n".join(item.get("content", item.get("output", "")) for item in context.items)
    return sum(fact.value in content for fact in facts)


def run_benchmark(
    *,
    turns: int = 120,
    runs: int = 3,
    budget: Budget | None = None,
    telemetry: Telemetry | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, CompiledContext]]:
    if type(runs) is not int or not 1 <= runs <= 20:
        raise ValueError("Benchmark runs must be between 1 and 20")
    history = conversation(turns)
    budget = budget or Budget()
    counter = TokenCounter()
    counter.count("Warm tokenizer before measuring context assembly")
    own_telemetry = telemetry is None
    telemetry = telemetry or Telemetry()
    rows: list[dict[str, Any]] = []
    final: dict[str, CompiledContext] = {}
    final_facts: tuple[TaskFact, ...] = ()
    try:
        for repetition in range(runs):
            strategies = ["full", "tail", "tiered"]
            strategies = strategies[repetition % 3 :] + strategies[: repetition % 3]
            for strategy in strategies:
                engine = ContextEngine(
                    budget=budget, tools=TOOLS, counter=counter, telemetry=telemetry
                )
                for index, (turn, updates) in enumerate(history, 1):
                    engine.ingest(turn, updates)
                    started = time.perf_counter()
                    context = engine.compile(QUERY, strategy=strategy)
                    duration = (time.perf_counter() - started) * 1000
                    rows.append(
                        {
                            "repetition": repetition,
                            "strategy": strategy,
                            "turn": index,
                            "application_input_tokens": context.token_count,
                            "capacity": context.capacity,
                            "fits_budget": context.token_count <= context.capacity,
                            "assembly_ms": duration,
                            "known_constraints": len(engine.facts),
                            "available_constraints": available(context, engine.facts),
                            "prefix_sha256": context.prefix_sha256,
                            "recent_turns": len(context.retained_turns),
                            "archive_chunks": context.archive_chunks,
                            "compacted_turns": context.compacted_turns,
                            "omitted_turns": context.omitted_turns,
                            "compaction_events": list(context.events),
                            "ttft_ms": None,
                            "provider_cached_tokens": None,
                        }
                    )
                final[strategy] = context
                final_facts = engine.facts
    finally:
        if own_telemetry:
            telemetry.close()
    groups = {}
    for strategy in ("full", "tail", "tiered"):
        samples = [row for row in rows if row["strategy"] == strategy]
        last = [row for row in samples if row["turn"] == turns]
        buckets = []
        for lower in range(1, turns + 1, 40):
            bucket = [row for row in samples if lower <= row["turn"] < lower + 40]
            buckets.append(
                {
                    "turns": [lower, min(turns, lower + 39)],
                    "assembly_p50_ms": percentile([row["assembly_ms"] for row in bucket], 0.5),
                    "assembly_p95_ms": percentile([row["assembly_ms"] for row in bucket], 0.95),
                    "prompt_p95_tokens": percentile(
                        [row["application_input_tokens"] for row in bucket], 0.95
                    ),
                    "ttft_p95_ms": None,
                }
            )
        groups[strategy] = {
            "samples": len(samples),
            "last_prompt_tokens": last[0]["application_input_tokens"],
            "peak_prompt_tokens": max(row["application_input_tokens"] for row in samples),
            "budget_violations": sum(not row["fits_budget"] for row in samples),
            "final_constraint_availability": last[0]["available_constraints"]
            / last[0]["known_constraints"],
            "assembly_p50_ms": percentile([row["assembly_ms"] for row in samples], 0.5),
            "assembly_p95_ms": percentile([row["assembly_ms"] for row in samples], 0.95),
            "prefix_versions": len({row["prefix_sha256"] for row in samples}),
            "compaction_events": sum(len(row["compaction_events"]) for row in samples),
            "buckets": buckets,
        }
    root = Path(__file__).parent
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "workload": "Synthetic launch review; explicit caller-supplied constraints, two revisions and complete historical tool exchanges",
        "turns": turns,
        "runs": runs,
        "strategies": {
            "full": "Counterfactual full history; may exceed budget; no pinned-state injection",
            "tail": "Budgeted whole-turn tail eviction; no pinned-state injection",
            "tiered": "Pinned task state + extract/digest/omit tiers + protected recent turns",
        },
        "budget": asdict(budget),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tiktoken": importlib.metadata.version("tiktoken"),
        },
        "token_accounting": "Exact o200k_base count of canonical application request JSON; provider framing may differ",
        "tokenizer_sha256": hashlib.sha256(
            gzip.decompress((root / "data/o200k_base.tiktoken.gz").read_bytes())
        ).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("engine.py", "models.py", "tokenizer.py", "scenarios.py", "benchmark.py")
        },
        "method": {
            "order": "Rotate strategy order each repetition",
            "percentile": "linear interpolation at (n-1)*q",
            "timed": "compile only, including OTel exporter overhead; tokenizer warm before timing",
            "quality_metric": "Exact latest-constraint value presence in rendered content; not model answer correctness",
        },
        "actual_model_api_calls": 0,
        "actual_model_api_cost_usd": 0,
        "ttft": "Not measured offline; use the explicit live streaming path",
        "provider_cache_hits": "Not measured offline; prefix fingerprints are application layout evidence only",
        "facts": [asdict(fact) for fact in final_facts],
        "groups": groups,
        "tiered_invariants_passed": all(
            row["fits_budget"] and row["available_constraints"] == row["known_constraints"]
            for row in rows
            if row["strategy"] == "tiered"
        ),
    }
    return summary, rows, final
