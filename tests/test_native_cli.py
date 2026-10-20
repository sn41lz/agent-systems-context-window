import json

import pytest

import context_window.cli as cli
from context_window.live import LiveSample, ProviderError
from context_window.models import canonical
from context_window.modern import NativeWindow


class NativeProvider:
    instances = []
    fail = False

    def __init__(self, model, **kwargs):
        self.model, self.options = model, kwargs["options"]
        self.closed = False
        self.__class__.instances.append(self)

    async def compact(self, context):
        return NativeWindow(
            self.model,
            canonical([{"type": "compaction", "encrypted_content": "SENSITIVE STATE"}]),
            context.tools_json,
            context.prefix_sha256,
            200,
            {
                "input_tokens": 200,
                "output_tokens": 40,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
            },
            None,
        )

    async def generate_native(self, window, query, **kwargs):
        if self.fail:
            raise ProviderError("sanitized")
        return LiveSample(
            '{"constraints":[],"missing":["UNKNOWN PRIVATE TEXT"]}', 2, 8, 100, 20, 0, None
        )

    async def aclose(self):
        self.closed = True


@pytest.fixture
def native(monkeypatch):
    NativeProvider.instances = []
    NativeProvider.fail = False
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr(cli, "OpenAIStream", NativeProvider)
    return NativeProvider


@pytest.mark.parametrize("content", [False, True])
def test_native_cli_persists_metrics_and_scores_with_content_opt_in(native, tmp_path, content):
    args = [
        "native",
        "--model",
        "model",
        "--turns",
        "6",
        "--grade-answers",
        "--output",
        str(tmp_path),
    ]
    if content:
        args.append("--include-content")
    assert cli.main(args) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["native_compaction"]["source_provider_tokens"] == 200
    assert summary["completed_calls"] == 1 and summary["exact_answer_rate"] == 0
    assert summary["strategy"] == "native" and summary["request_options"]["preflight"]
    assert (tmp_path / "native-window.json").exists() == content
    assert ("UNKNOWN PRIVATE TEXT" in (tmp_path / "samples.jsonl").read_text()) == content
    assert native.instances[-1].closed


def test_native_second_stage_failure_preserves_compaction_accounting(native, tmp_path):
    native.fail = True
    assert cli.main(["native", "--model", "model", "--turns", "1", "--output", str(tmp_path)]) == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "provider_error" and summary["completed_calls"] == 0
    assert summary["native_compaction"]["usage"]["output_tokens"] == 40
    assert native.instances[-1].closed


def test_native_rejects_explicit_breakpoints_before_client_construction(native, tmp_path):
    assert (
        cli.main(
            ["native", "--model", "model", "--cache-mode", "explicit", "--output", str(tmp_path)]
        )
        == 2
    )
    assert not native.instances


def test_policy_file_and_output_reservation_are_explicit(native, tmp_path):
    policy = tmp_path / "policy.txt"
    policy.write_text("Review these actual operating requirements.")
    assert (
        cli.main(
            [
                "native",
                "--model",
                "model",
                "--turns",
                "1",
                "--output-tokens",
                "1024",
                "--reasoning-effort",
                "low",
                "--instructions-file",
                str(policy),
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert native.instances[-1].options.reasoning_effort == "low"
    policy.write_bytes(b"\xff")
    assert (
        cli.main(
            [
                "native",
                "--model",
                "model",
                "--instructions-file",
                str(policy),
                "--output",
                str(tmp_path),
            ]
        )
        == 2
    )


def test_cache_write_pricing_requires_other_rates(native, tmp_path):
    assert (
        cli.main(
            ["native", "--model", "model", "--cache-write-rate", "1", "--output", str(tmp_path)]
        )
        == 2
    )
    assert not native.instances
