import json

import pytest

import context_window.cli as cli
from context_window.live import LiveSample, ProviderError


class FakeProvider:
    instances = []
    fail_at = None

    def __init__(self, model, *, telemetry=None, rates=None, options=None):
        self.model = model
        self.rates = rates
        self.calls = 0
        self.closed = False
        self.__class__.instances.append(self)

    async def generate(self, context, *, max_output_tokens):
        self.calls += 1
        if self.calls == self.fail_at:
            raise ProviderError("sanitized failure")
        assert context.token_count <= context.capacity
        estimate = self.rates.estimate(100, 10, 20) if self.rates else None
        return LiveSample("SENSITIVE MODEL OUTPUT", 2.0, 8.0, 100, 10, 20, estimate)

    async def aclose(self):
        self.closed = True


@pytest.fixture
def fake(monkeypatch):
    FakeProvider.instances = []
    FakeProvider.fail_at = None
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-never-used")
    monkeypatch.setattr(cli, "OpenAIStream", FakeProvider)
    return FakeProvider


@pytest.mark.parametrize("include", [False, True])
def test_live_records_metrics_and_only_persists_content_on_opt_in(fake, tmp_path, include):
    args = ["live", "--model", "my-model", "--turns", "3", "--output", str(tmp_path)]
    if include:
        args.append("--include-content")
    assert cli.main(args) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    samples = (tmp_path / "samples.jsonl").read_text()
    assert summary["model"] == "my-model" and summary["completed_calls"] == 3
    assert summary["buckets"][0]["ttft_p95_ms"] == 2
    assert summary["cache_hit_tokens"] == 60 and summary["input_tokens"] == 300
    assert summary["estimated_usd_per_1000_completed_calls"] is None
    assert ("SENSITIVE MODEL OUTPUT" in samples) == include
    assert "SENSITIVE MODEL OUTPUT" not in (tmp_path / "traces.jsonl").read_text()
    assert fake.instances[0].closed and fake.instances[0].calls == 3


def test_partial_provider_failure_preserves_completed_measurements(fake, tmp_path):
    fake.fail_at = 2
    assert cli.main(["live", "--model", "model", "--turns", "3", "--output", str(tmp_path)]) == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "provider_error" and summary["completed_calls"] == 1
    assert fake.instances[0].closed


def test_explicit_prices_produce_labeled_estimates(fake, tmp_path):
    assert (
        cli.main(
            [
                "live",
                "--model",
                "model",
                "--turns",
                "2",
                "--input-rate",
                "2",
                "--output-rate",
                "8",
                "--cached-rate",
                "1",
                "--price-label",
                "test rates",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["pricing"]["label"] == "test rates"
    assert summary["estimated_usd_per_1000_completed_calls"] == pytest.approx(0.26)


@pytest.mark.parametrize(
    "flag", ["--input-rate", "--output-rate", "--cached-rate", "--price-label"]
)
def test_partial_pricing_config_stops_before_client_construction(fake, tmp_path, flag):
    value = "test" if flag == "--price-label" else "1"
    assert (
        cli.main(
            ["live", "--model", "model", "--turns", "1", flag, value, "--output", str(tmp_path)]
        )
        == 2
    )
    assert not fake.instances


def test_bad_provider_config_is_reported_without_secret(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid:private-config")
    assert cli.main(["live", "--model", "model", "--turns", "1", "--output", str(tmp_path)]) == 2
    assert "private-config" not in capsys.readouterr().err
