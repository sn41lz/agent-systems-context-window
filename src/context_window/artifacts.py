"""Private local evidence and a self-contained, script-free review report."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any


def write_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(value)


def write_json(path: Path, value: Any):
    write_text(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def chart(rows: list[dict], capacity: int, *, bounded=False) -> str:
    colors = {"full": "#f0ae72", "tail": "#93a5ff", "tiered": "#5be0bb"}
    chosen = [
        row for row in rows if row["repetition"] == 0 and (not bounded or row["strategy"] != "full")
    ]
    highest = max(capacity, max(row["application_input_tokens"] for row in chosen)) * 1.12
    turns = max(row["turn"] for row in chosen)

    def x(turn):
        return 50 + (turn - 1) / max(1, turns - 1) * 680

    def y(tokens):
        return 230 - tokens / highest * 190

    elements = [
        f'<line x1="50" y1="{y(capacity):.2f}" x2="730" y2="{y(capacity):.2f}" stroke="#f8cd65" stroke-dasharray="5 5"/><text x="52" y="{y(capacity) - 8:.2f}" fill="#f8cd65" font-size="12">budget {capacity:,}</text>'
    ]
    for strategy, color in colors.items():
        data = [row for row in chosen if row["strategy"] == strategy]
        if data:
            points = " ".join(
                f"{x(row['turn']):.2f},{y(row['application_input_tokens']):.2f}" for row in data
            )
            elements.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{points}"/>'
            )
    for number in (1, turns // 2, turns):
        elements.append(
            f'<text x="{x(number):.2f}" y="255" fill="#b3bed0" font-size="12">{number}</text>'
        )
    elements.append(
        '<text x="50" y="22" fill="#b3bed0" font-size="12">Application input tokens</text><text x="660" y="277" fill="#b3bed0" font-size="12">Turn number</text>'
    )
    title = "Bounded strategies" if bounded else "Full history versus bounded context"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 290" role="img" aria-label="{title}" style="width:100%;height:auto"><rect width="780" height="290" rx="12" fill="#111c2b"/><g font-family="system-ui,sans-serif">'
        + "".join(elements)
        + "</g></svg>"
    )


def write_report(output: Path, summary: dict, rows: list[dict]):
    groups = summary["groups"]
    table = "".join(
        f"<tr><td>{e(name)}</td><td>{g['last_prompt_tokens']:,}</td><td>{g['budget_violations']}</td><td>{g['final_constraint_availability']:.0%}</td><td>{g['assembly_p95_ms']:.2f} ms</td></tr>"
        for name, g in groups.items()
    )
    facts = "".join(
        f"<tr><td>{e(fact['key'])}</td><td>{e(fact['value'])}</td><td>r{fact['revision']} · {e(fact['source_turn'])}</td></tr>"
        for fact in summary["facts"]
    )
    events = [
        event
        for row in rows
        if row["strategy"] == "tiered" and row["repetition"] == 0
        for event in row["compaction_events"]
    ]
    event_html = "".join(
        f"<details><summary>{e(event['action'])} · {e(event['first_turn'])} → {e(event['last_turn'])}</summary><pre>{e(json.dumps(event, indent=2))}</pre></details>"
        for event in events
    )
    event_html = (
        f'<details><summary>Inspect {len(events)} recorded decisions</summary><div class="event-log">'
        + event_html
        + "</div></details>"
    )
    css = """*{box-sizing:border-box}body{margin:0;background:#0a1019;color:#e6eef8;font:15px/1.65 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:52px 32px}a{color:#5be0bb}h1{font-size:clamp(32px,5vw,58px);letter-spacing:-.055em;line-height:1.08;margin:24px 0}h2{font-size:21px;letter-spacing:-.02em}.label{color:#5be0bb;font-size:11px;font-weight:800;letter-spacing:.18em}.muted{color:#a4b3c7}.notice{border-left:3px solid #f8cd65;background:#18202c;padding:18px 22px;margin:24px 0;border-radius:8px}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:28px 0}.card,.panel{min-width:0;background:#111c2b;border:1px solid #263349;border-radius:14px;padding:22px}.value{font-size:30px;font-weight:750;letter-spacing:-.04em;color:#5be0bb}.columns{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}.panel{margin:18px 0}.scroll{overflow-x:auto;max-width:100%}table{width:100%;border-collapse:collapse;text-align:left}th,td{padding:13px 12px;border-bottom:1px solid #28344a}th{color:#a4b3c7;font-size:11px;text-transform:uppercase;letter-spacing:.08em}td{font-size:13px}pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#bfccdf;font-size:12px}.event-log{max-height:440px;overflow-y:auto;padding:0 10px}details{border-top:1px solid #28344a;padding:12px 0}summary{cursor:pointer;font-size:13px}.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:12px}.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px}footer{margin-top:26px;font-size:12px;color:#a4b3c7}@media(max-width:850px){.cards{grid-template-columns:1fr 1fr}.columns,.grid{grid-template-columns:1fr}main{padding:30px 18px}}@media(max-width:460px){.card,.panel{padding:16px}.cards{gap:8px}.value{font-size:25px}.label{font-size:9px}.card{overflow-wrap:anywhere}th,td{padding:10px 8px}h1{font-size:34px}}"""
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none';style-src 'unsafe-inline';img-src data:;base-uri 'none';form-action 'none'"><title>Context Window · Continuity Lab</title><style>{css}</style></head><body><main><div class="label">CONTEXT WINDOW / CONTINUITY LAB</div><h1>{summary["turns"]} turns.<br>One bounded context.</h1><p class="muted">A measured comparison of growing history, whole-turn eviction, and an explicit compaction policy.</p><div class="notice"><strong>Offline evidence, precisely scoped.</strong> Real local tokenization and assembly timings; synthetic conversation data. No model inference, provider cache hits, or TTFT measurements. Constraint availability measures presence, not model answer quality.</div><div class="cards"><div class="card"><div class="label">TIERED INPUT / FINAL</div><div class="value">{groups["tiered"]["last_prompt_tokens"]:,}</div><span class="muted">of {summary["budget"]["context_tokens"] - summary["budget"]["output_reserve"] - summary["budget"]["framing_reserve"]:,} application tokens</span></div><div class="card"><div class="label">LATEST CONSTRAINTS</div><div class="value">{groups["tiered"]["final_constraint_availability"]:.0%}</div><span class="muted">available in final context</span></div><div class="card"><div class="label">BUDGET VIOLATIONS</div><div class="value">{groups["tiered"]["budget_violations"]}</div><span class="muted">tiered strategy</span></div><div class="card"><div class="label">PREFIX VERSIONS</div><div class="value">{groups["tiered"]["prefix_versions"]}</div><span class="muted">stable policy and tool catalogue</span></div></div><div class="legend"><span style="color:#f0ae72">● full history</span><span style="color:#93a5ff">● tail eviction</span><span style="color:#5be0bb">● tiered compaction</span></div><div class="columns"><section class="panel"><h2>The growth problem</h2>{chart(rows, rows[0]["capacity"])}</section><section class="panel"><h2>Inside the budget</h2>{chart(rows, rows[0]["capacity"], bounded=True)}</section></div><section class="panel"><h2>Matched workload, different policies</h2><div class="scroll"><table><tr><th>Policy</th><th>Final tokens</th><th>Over budget</th><th>Constraints available</th><th>Assembly p95</th></tr>{table}</table></div><p class="muted">{summary["runs"]} repetition(s) × {summary["turns"]} turns per policy. Full history is a counterfactual baseline when it exceeds the configured budget. Timings include tracing overhead.</p></section><section class="panel"><h2>Current task state survives historical revisions</h2><div class="scroll"><table><tr><th>Constraint</th><th>Current value</th><th>Provenance</th></tr>{facts}</table></div><p class="muted">Facts are explicitly supplied by the caller. Extraction from natural language is not inferred or benchmarked.</p></section><div class="grid"><section class="panel"><h2>Compaction decisions</h2><p class="muted">Extract → digest → omit. These are lossy historical views; pinned task state is protected.</p>{event_html}</section><section class="panel"><h2>Reproduce and inspect</h2><pre>make demo\nmake test\nmake benchmark RUNS=3</pre><p><a href="summary.json">Summary and methodology</a> · <a href="samples.jsonl">Every measured sample</a> · <a href="traces.jsonl">OTel spans</a></p><h2>Provider measurements</h2><p class="muted">Actual offline API calls: 0. Actual offline model cost: $0. Use the explicit live streaming command for observed TTFT, provider token usage and cached tokens. A stable fingerprint does not establish a provider cache hit.</p><h2>Token budget boundary</h2><p class="muted">Counts use the shipped o200k tokenizer over canonical application JSON. Provider framing/tokenization can differ. Output and framing reserves are explicit; oversized protected content fails rather than silently disappearing.</p></section></div><footer>Generated from recorded synthetic-workload evidence. No scripts or external resources. Raw source is discarded by compaction; keep a separate source log if forensic replay is required.</footer></main></body></html>"""
    write_text(output / "report.html", document)
    write_text(output / "tokens.svg", chart(rows, rows[0]["capacity"]))
