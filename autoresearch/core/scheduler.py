from __future__ import annotations

from typing import Any


def ordered_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # For now keep deterministic order by config declaration.
    # Later we can upgrade to gain-per-time priority scheduling.
    return [t for t in tasks if bool(t.get("enabled", True))]

