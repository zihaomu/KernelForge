from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import dump_json, load_json, now_iso


def init_run_state(run_dir: Path, *, run_id: str, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    state = {
        "run_id": run_id,
        "status": "running",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "tasks": [
            {
                "op_id": t["op_id"],
                "status": "pending",
                "summary": {},
            }
            for t in tasks
        ],
    }
    dump_json(run_dir / "state.json", state)
    return state


def load_state(run_dir: Path) -> dict[str, Any]:
    return load_json(run_dir / "state.json")


def save_state(run_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    dump_json(run_dir / "state.json", state)


def update_task_state(
    *,
    run_dir: Path,
    state: dict[str, Any],
    op_id: str,
    status: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    for task in state.get("tasks", []):
        if task.get("op_id") == op_id:
            task["status"] = status
            task["summary"] = summary
            break
    save_state(run_dir, state)
    return state

