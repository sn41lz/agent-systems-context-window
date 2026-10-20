import json

import pytest

from context_window.cli import main
from context_window.models import canonical


def test_own_transcript_compiles_revised_facts_and_complete_tool_cycle(tmp_path):
    transcript = tmp_path / "input.jsonl"
    first = {
        "turn_id": "change",
        "user": "Keep all records in Europe",
        "assistant": "Acknowledged",
        "facts": [{"key": "region", "value": "Europe", "expected_revision": 0}],
    }
    turns = [first] + [
        {
            "turn_id": f"read{i}",
            "user": "Check work item " * 80,
            "assistant": "Pending owner review " * 80,
        }
        for i in range(20)
    ]
    turns.append(
        {
            "turn_id": "new",
            "user": "Correction: France only.",
            "assistant": "Recorded.",
            "facts": [{"key": "region", "value": "France", "expected_revision": 1}],
            "tools": [
                {
                    "call_id": "read1",
                    "name": "read_status",
                    "arguments": "{}",
                    "output": '{"state":"pending"}',
                }
            ],
        }
    )
    transcript.write_text("".join(canonical(row) + "\n" for row in turns))
    output = tmp_path / "compiled"
    assert (
        main(
            [
                "compile",
                "--transcript",
                str(transcript),
                "--query",
                "Where may data reside?",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    request = json.loads((output / "request.json").read_text())
    state = json.loads(request["input"][1]["content"])["task_state"]["region"]
    assert state["value"] == "France" and state["revision"] == 2 and state["source_turn"] == "new"
    assert sum(item.get("type") == "function_call" for item in request["input"]) == 1
    assert sum(item.get("type") == "function_call_output" for item in request["input"]) == 1
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["token_count"] <= metadata["capacity"]
    assert "France" not in (output / "traces.jsonl").read_text()


@pytest.mark.parametrize(
    "value",
    [
        "",
        "[]\n",
        '{"unknown":"SECRET"}\n',
        '{"turn_id":"a","user":true}\n',
        '{"turn_id":"a","user":"q","tools":[1]}\n',
        '{"turn_id":"a","user":"q","facts":false}\n',
        "x" * 65_537,
    ],
)
def test_invalid_or_overlarge_transcript_stops_with_sanitized_record_error(tmp_path, value, capsys):
    transcript = tmp_path / "input.jsonl"
    transcript.write_text(value)
    assert (
        main(
            [
                "compile",
                "--transcript",
                str(transcript),
                "--query",
                "q",
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    assert "SECRET" not in capsys.readouterr().err


def test_transcript_tool_catalog_is_validated(tmp_path):
    transcript = tmp_path / "input.jsonl"
    transcript.write_text('{"turn_id":"a","user":"q"}\n')
    catalog = tmp_path / "tools.json"
    for value in ("{}", '[{"type":"function","name":"read","parameters":{"type":"WRONG"}}]'):
        catalog.write_text(value)
        assert (
            main(
                [
                    "compile",
                    "--transcript",
                    str(transcript),
                    "--tools-file",
                    str(catalog),
                    "--query",
                    "q",
                    "--output",
                    str(tmp_path / "out"),
                ]
            )
            == 2
        )
