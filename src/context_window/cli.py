"""Offline by default; billable streaming probes require the explicit live path."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from .artifacts import write_json, write_report, write_text
from .benchmark import percentile, run_benchmark
from .engine import DEFAULT_INSTRUCTIONS, ContextEngine
from .live import OpenAIStream, ProviderError, Rates
from .models import Budget, FactUpdate, ToolExchange, Turn, strict_json, text
from .modern import RequestOptions
from .quality import ANSWER_SCHEMA, grade_answer
from .scenarios import QUERY, TOOLS, conversation
from .telemetry import Telemetry


def parser():
    root = argparse.ArgumentParser(description="An explicit, cache-aware context window engine")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("demo", "benchmark"):
        command = sub.add_parser(
            name, help="Offline tokenization and policy comparison; no model calls"
        )
        command.add_argument("--turns", type=int, default=120)
        command.add_argument("--runs", type=int, default=1 if name == "demo" else 3)
        command.add_argument("--context-tokens", type=int, default=4096)
        command.add_argument("--output", type=Path, default=Path("artifacts") / name)
    compile_cmd = sub.add_parser(
        "compile", help="Compile your own bounded JSONL transcript locally"
    )
    compile_cmd.add_argument("--transcript", type=Path, required=True)
    compile_cmd.add_argument("--tools-file", type=Path)
    compile_cmd.add_argument("--query", required=True)
    compile_cmd.add_argument("--strategy", choices=("tail", "tiered"), default="tiered")
    compile_cmd.add_argument("--context-tokens", type=int, default=4096)
    compile_cmd.add_argument("--output-tokens", type=int, default=512)
    compile_cmd.add_argument("--output", type=Path, default=Path("artifacts/compiled"))
    for name in ("live", "native"):
        live = sub.add_parser(
            name,
            help="Billable replay probes"
            if name == "live"
            else "Billable native compaction and lossless continuation",
        )
        live.add_argument("--model", required=True)
        if name == "live":
            live.add_argument("--strategy", choices=("tail", "tiered"), default="tiered")
        live.add_argument("--turns", type=int, default=120)
        live.add_argument("--context-tokens", type=int, default=4096 if name == "live" else 32768)
        live.add_argument("--output-tokens", type=int, default=512 if name == "live" else 2048)
        live.add_argument("--output", type=Path, default=Path("artifacts") / name)
        live.add_argument("--include-content", action="store_true")
        live.add_argument(
            "--cache-mode", choices=("default", "implicit", "explicit", "off"), default="default"
        )
        live.add_argument("--preflight", action="store_true")
        live.add_argument(
            "--reasoning-effort",
            choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        )
        live.add_argument("--instructions-file", type=Path)
        live.add_argument("--grade-answers", action="store_true")
        live.add_argument("--input-rate", type=float)
        live.add_argument("--output-rate", type=float)
        live.add_argument("--cached-rate", type=float)
        live.add_argument("--cache-write-rate", type=float)
        live.add_argument("--price-label")
    return root


def compile_transcript(args):
    budget = Budget(context_tokens=args.context_tokens, output_reserve=args.output_tokens)
    count = 0
    size = 0
    tools = ()
    if args.tools_file:
        with args.tools_file.open("rb") as source:
            try:
                catalog = strict_json(source.read(65_537).decode("utf-8"))
                if not isinstance(catalog, list):
                    raise ValueError
                tools = tuple(catalog)
            except ValueError:
                raise ValueError("Tools file must contain a bounded UTF-8 JSON array") from None
    with Telemetry(args.output) as telemetry:
        engine = ContextEngine(budget=budget, tools=tools, telemetry=telemetry)
        with args.transcript.open("rb") as source:
            while line := source.readline(65_538):
                size += len(line)
                if len(line) > 65_536 or size > 16_777_216 or count >= 1000:
                    raise ValueError("Transcript limit: 64 KiB per line, 16 MiB total, 1000 turns")
                try:
                    value = strict_json(line.decode("utf-8"))
                    if not isinstance(value, dict) or set(value) - {
                        "turn_id",
                        "user",
                        "assistant",
                        "tools",
                        "facts",
                    }:
                        raise ValueError
                    exchanges = tuple(ToolExchange(**item) for item in value.get("tools", []))
                    updates = tuple(FactUpdate(**item) for item in value.get("facts", []))
                    turn = Turn(
                        value["turn_id"], value["user"], value.get("assistant", ""), exchanges
                    )
                except (TypeError, KeyError, ValueError):
                    raise ValueError(f"Invalid transcript record at line {count + 1}") from None
                engine.ingest(turn, updates)
                context = engine.compile(args.query, strategy=args.strategy)
                count += 1
        if not count:
            raise ValueError("Transcript must contain at least one turn")
    write_json(args.output / "request.json", context.request())
    write_json(args.output / "metadata.json", {**context.metadata(), "turns": count})
    print(
        f"{count} turns · {context.token_count}/{context.capacity} application tokens · {args.output / 'request.json'}"
    )
    return 0


def offline(args):
    budget = Budget(context_tokens=args.context_tokens)
    with Telemetry(args.output, service_name="context-window") as telemetry:
        summary, rows, final = run_benchmark(
            turns=args.turns, runs=args.runs, budget=budget, telemetry=telemetry
        )
    write_json(args.output / "summary.json", summary)
    write_text(
        args.output / "samples.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    for strategy, context in final.items():
        write_json(args.output / f"{strategy}-context.json", context.request())
    write_report(args.output, summary, rows)
    print(f"{args.turns} turns × {args.runs} repetition(s) per strategy")
    for strategy, group in summary["groups"].items():
        print(
            f"{strategy:6} · final {group['last_prompt_tokens']:6,} tokens · {group['budget_violations']} over-budget samples · latest constraints {group['final_constraint_availability']:.0%} · assembly p95 {group['assembly_p95_ms']:.2f} ms"
        )
    print(f"Report: {args.output / 'report.html'}")
    print(
        "Real local tokenization/timings; synthetic conversation; no model quality, cache-hit or TTFT measurement."
    )
    return 0 if summary["tiered_invariants_passed"] else 1


async def live(args):
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Set OPENAI_API_KEY for the explicit live path. The demo needs no key.", file=sys.stderr
        )
        return 2
    config = (args.input_rate, args.output_rate, args.cached_rate, args.price_label)
    if any(value is not None for value in config) and not all(
        value is not None for value in config
    ):
        raise ValueError("Supply all three rates plus --price-label, or omit pricing entirely")
    if args.cache_write_rate is not None and not all(value is not None for value in config):
        raise ValueError("Cache-write pricing also requires the other rates and a label")
    rates = (
        Rates(*config, cache_write_per_million=args.cache_write_rate)
        if all(value is not None for value in config)
        else None
    )
    budget = Budget(context_tokens=args.context_tokens, output_reserve=args.output_tokens)
    history = conversation(args.turns)
    instructions = DEFAULT_INSTRUCTIONS
    if args.instructions_file:
        # Bound the read before decoding. Do not echo file contents on errors.
        with args.instructions_file.open("rb") as source:
            extra = source.read(65_537)
        try:
            instructions += "\n\n" + extra.decode("utf-8")
            text(instructions, maximum=65_536)
        except (ValueError, UnicodeError):
            raise ValueError(
                "Instructions must be valid UTF-8 and fit the 64 KiB policy limit"
            ) from None
    options = RequestOptions(
        args.cache_mode,
        args.preflight,
        args.reasoning_effort,
        ANSWER_SCHEMA if args.grade_answers else None,
    )
    native_mode = args.command == "native"
    if native_mode and options.cache_mode == "explicit":
        raise ValueError(
            "Native continuation preserves provider output; use default, implicit or off caching"
        )
    rows = []
    status = "completed"
    native_record = None
    with Telemetry(args.output) as telemetry:
        engine = ContextEngine(
            budget=budget, tools=TOOLS, instructions=instructions, telemetry=telemetry
        )
        provider = OpenAIStream(args.model, telemetry=telemetry, rates=rates, options=options)
        try:
            for index, (turn, updates) in enumerate(history, 1):
                engine.ingest(turn, updates)
                if native_mode and index != len(history):
                    continue
                context = engine.compile(QUERY, strategy="full" if native_mode else args.strategy)
                try:
                    if native_mode:
                        window = await provider.compact(context)
                        native_record = {
                            "window_sha256": window.digest,
                            "output_items": len(window.items),
                            "source_provider_tokens": window.source_provider_tokens,
                            "usage": window.usage,
                            "estimated_cost_usd": window.estimated_cost_usd,
                        }
                        if args.include_content:
                            write_json(args.output / "native-window.json", window.items)
                        sample = await provider.generate_native(
                            window,
                            QUERY,
                            capacity=budget.input_capacity,
                            max_output_tokens=budget.output_reserve,
                        )
                    else:
                        sample = await provider.generate(
                            context, max_output_tokens=budget.output_reserve
                        )
                except ProviderError:
                    status = "provider_error"
                    break
                record = {
                    **asdict(sample),
                    "turn": index,
                    "application_input_tokens": context.token_count,
                    "prefix_sha256": context.prefix_sha256,
                }
                if args.grade_answers:
                    record["quality"] = grade_answer(sample.text, engine.facts)
                if not args.include_content:
                    record.pop("text")
                rows.append(record)
        finally:
            await provider.aclose()
    buckets = []
    for lower in range(1, args.turns + 1, 40):
        samples = [row for row in rows if lower <= row["turn"] < lower + 40]
        if samples:
            ttft = [row["ttft_ms"] for row in samples if row["ttft_ms"] is not None]
            buckets.append(
                {
                    "turn_range": [lower, min(args.turns, lower + 39)],
                    "samples": len(samples),
                    "ttft_p50_ms": percentile(ttft, 0.5) if ttft else None,
                    "ttft_p95_ms": percentile(ttft, 0.95) if ttft else None,
                    "latency_p95_ms": percentile([row["duration_ms"] for row in samples], 0.95),
                }
            )
    summary = {
        "status": status,
        "model": args.model,
        "strategy": "native" if native_mode else args.strategy,
        "requested_turns": args.turns,
        "completed_calls": len(rows),
        "failed_generation_probes": int(
            status == "provider_error" and (not native_mode or native_record is not None)
        ),
        "failed_compactions": int(
            status == "provider_error" and native_mode and native_record is None
        ),
        "sdk_retries": 0,
        "native_compaction": native_record,
        "data_source": "fixed synthetic transcript, probed independently at every turn",
        "measurement": "TTFT starts at generation request, excluding optional count preflight; duration includes preflight. Usage from completed provider responses.",
        "cache_hit_tokens": sum(row["cached_tokens"] for row in rows),
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "cache_write_tokens": sum(row["cache_write_tokens"] for row in rows)
        if rows and all(row["cache_write_tokens"] is not None for row in rows)
        else None,
        "reasoning_tokens": sum(row["reasoning_tokens"] for row in rows)
        if rows and all(row["reasoning_tokens"] is not None for row in rows)
        else None,
        "latency_p50_ms": percentile([row["duration_ms"] for row in rows], 0.5) if rows else None,
        "latency_p95_ms": percentile([row["duration_ms"] for row in rows], 0.95) if rows else None,
        "estimated_usd_per_1000_completed_calls": sum(row["estimated_cost_usd"] for row in rows)
        / len(rows)
        * 1000
        if rates and rows and all(row["estimated_cost_usd"] is not None for row in rows)
        else None,
        "pricing": asdict(rates) if rates else None,
        "cost_limitations": "No hard spend cap. Per-1000 estimate covers completed generation probes only; compaction reported separately. Unknown cache writes make estimates null. Failed attempts, token-count requests and non-token fees excluded.",
        "budget": asdict(budget),
        "buckets": buckets,
        "answer_quality": "Exact key/value/source grading, not general answer quality"
        if args.grade_answers
        else "Not scored",
        "exact_answer_rate": sum(row["quality"]["exact_match"] for row in rows) / len(rows)
        if args.grade_answers and rows
        else None,
        "request_options": {
            "cache_mode": options.cache_mode,
            "preflight": options.preflight or native_mode,
            "reasoning_effort": options.reasoning_effort,
        },
        "content_included": args.include_content,
    }
    write_json(args.output / "summary.json", summary)
    write_text(
        args.output / "samples.jsonl",
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
    )
    print(
        f"{status} · {len(rows)} completed streaming probes · summary: {args.output / 'summary.json'}"
    )
    return 0 if status == "completed" else 1


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "compile":
            return compile_transcript(args)
        return asyncio.run(live(args)) if args.command in {"live", "native"} else offline(args)
    except ProviderError:
        print(
            "Provider configuration or request failed. Check model, credentials and endpoint settings.",
            file=sys.stderr,
        )
        return 2
    except (ValueError, OSError) as exc:
        print(f"Configuration/output error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
