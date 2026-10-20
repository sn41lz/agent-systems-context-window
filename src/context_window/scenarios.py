"""A deterministic long-running launch review with changing task constraints."""

from __future__ import annotations

from .models import FactUpdate, ToolExchange, Turn, canonical

TOOLS = (
    {
        "type": "function",
        "name": "inspect_readiness",
        "description": "Read a synthetic readiness snapshot; historical results are not current observations.",
        "parameters": {
            "type": "object",
            "properties": {"region": {"type": "string"}},
            "required": ["region"],
            "additionalProperties": False,
        },
        "strict": False,
    },
)
QUERY = "List the current launch constraints with their source turn IDs. If evidence is missing, say so; do not invent an answer."


def conversation(turns: int = 120) -> list[tuple[Turn, tuple[FactUpdate, ...]]]:
    if type(turns) is not int or not 1 <= turns <= 1000:
        raise ValueError("Scenario length must be between 1 and 1000 turns")
    updates = {
        1: ("budget", "EUR 48000"),
        2: ("region", "eu-west-1"),
        3: ("deadline", "2026-11-30"),
        4: ("residency", "Customer data must remain in the EU"),
        5: ("rollback", "Rollback must complete within 90 seconds"),
        6: ("approval", "A human release manager must approve production changes"),
        47: ("budget", "EUR 42000"),
        82: ("deadline", "2026-12-07"),
    }
    result: list[tuple[Turn, tuple[FactUpdate, ...]]] = []
    for number in range(1, turns + 1):
        turn_id = f"turn-{number:04d}"
        change: tuple[FactUpdate, ...] = ()
        if number in updates:
            key, value = updates[number]
            user = "Record this explicit task constraint: " + canonical(
                {"key": key, "value": value}
            )
            answer = f"Recorded the caller-supplied {key} constraint; this is synthetic task data."
            change = (FactUpdate(key, value),)
        else:
            subsystem = ("checkout", "inventory", "notifications", "support", "analytics")[
                number % 5
            ]
            user = (
                f"Review work item {number} for {subsystem}. Compare the canary readiness notes, "
                "operational dependencies, documentation completeness, test ownership, and open "
                "verification tasks. Keep observations separate from launch constraints. "
                "This is supporting discussion, not permission to change production. "
            ) * 2
            answer = (
                f"Work item {number}: evidence is synthetic. The readiness checklist for {subsystem} "
                "still needs owner review, a rollback rehearsal, capacity verification, and a "
                "recorded decision. No production change was made. Existing task constraints "
                "remain in force; this discussion does not revise them. "
            ) * 2
        tools: tuple[ToolExchange, ...] = ()
        if number % 9 == 0:
            tools = (
                ToolExchange(
                    f"readiness-{number}",
                    "inspect_readiness",
                    canonical({"region": "eu-west-1"}),
                    canonical(
                        {"synthetic": True, "work_item": number, "status": "pending-owner-review"}
                    ),
                ),
            )
        user = f"Source turn: {turn_id}\n{user}"
        result.append((Turn(turn_id, user, answer, tools), change))
    return result
