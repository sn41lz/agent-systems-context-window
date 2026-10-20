import json
import stat

import pytest

from context_window.artifacts import write_json, write_report
from context_window.benchmark import percentile, run_benchmark
from context_window.cli import main


def test_demo_artifacts_match_measured_workload_and_provenance(tmp_path):
    assert main(["demo", "--turns", "24", "--output", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    rows = [json.loads(line) for line in (tmp_path / "samples.jsonl").read_text().splitlines()]
    assert len(rows) == 72 and summary["actual_model_api_calls"] == 0
    assert summary["tiered_invariants_passed"] and summary["actual_model_api_cost_usd"] == 0
    assert all(row["ttft_ms"] is None and row["provider_cached_tokens"] is None for row in rows)
    for strategy in ("full", "tail", "tiered"):
        assert (tmp_path / f"{strategy}-context.json").exists()
    assert "No model inference" in (tmp_path / "report.html").read_text()
    assert (tmp_path / "tokens.svg").exists()
    traces = (tmp_path / "traces.jsonl").read_text()
    assert "EUR 48000" not in traces and "Review work item" not in traces


def test_report_escapes_payloads_and_is_self_contained(tmp_path):
    summary, rows, _ = run_benchmark(turns=2, runs=1)
    summary["facts"][0]["value"] = '<script>alert("secret")</script>'
    write_report(tmp_path, summary, rows)
    report = (tmp_path / "report.html").read_text()
    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    assert "default-src 'none'" in report
    assert 'src="https://' not in report


def test_private_permissions_apply_even_to_preexisting_files(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("old")
    path.chmod(0o644)
    write_json(path, {"safe": True})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_missing_live_key_stops_before_any_artifact_or_client(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["live", "--model", "model", "--output", str(tmp_path)]) == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "args",
    [["demo", "--turns", "0"], ["benchmark", "--runs", "0"], ["demo", "--context-tokens", "500"]],
)
def test_invalid_cli_configuration(args, tmp_path):
    assert main([*args, "--output", str(tmp_path)]) == 2


def test_percentile_math_and_invalid_samples():
    assert percentile([10, 20, 30], 0.95) == 29
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile([1], 2)
