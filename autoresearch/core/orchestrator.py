from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .dashboards import build_portfolio_dashboard
from .scheduler import ordered_tasks
from .state_store import init_run_state, load_state, save_state, update_task_state
from .utils import dump_json, ensure_dir, load_yaml, now_run_id


def _registry_map(registry_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for op in registry_cfg.get("ops", []):
        out[str(op["op_id"])] = op
    return out


def run_portfolio(
    *,
    repo_root: Path,
    global_config_path: Path,
    portfolio_path: Path | None = None,
) -> dict[str, Any]:
    global_cfg = load_yaml(global_config_path)
    paths_cfg = global_cfg["paths"]
    exec_cfg = global_cfg["execution"]
    platforms_cfg = global_cfg["platform_configs"]

    registry = load_yaml(repo_root / paths_cfg["ops_registry"])
    reg_map = _registry_map(registry)
    portfolio_cfg_path = (
        portfolio_path if portfolio_path is not None else (repo_root / paths_cfg["default_portfolio"])
    )
    portfolio = load_yaml(portfolio_cfg_path)
    tasks = ordered_tasks(list(portfolio.get("tasks", [])))

    run_id = now_run_id()
    workspace_dir = repo_root / paths_cfg["workspace_dir"]
    run_dir = workspace_dir / "runs" / run_id
    ensure_dir(run_dir)

    state = init_run_state(run_dir, run_id=run_id, tasks=tasks)
    task_summaries: list[dict[str, Any]] = []

    for task in tasks:
        op_id = str(task["op_id"])
        if op_id not in reg_map:
            summary = {
                "op_id": op_id,
                "status": "error",
                "error": "op_id_not_found_in_registry",
                "iterations": 0,
                "best_score": 0.0,
                "best_candidate_signature": "",
                "task_run_dir": str(run_dir / op_id),
            }
            task_summaries.append(summary)
            state = update_task_state(run_dir=run_dir, state=state, op_id=op_id, status="error", summary=summary)
            if bool(exec_cfg.get("stop_on_error", False)):
                state["status"] = "failed"
                save_state(run_dir, state)
                break
            continue

        reg = reg_map[op_id]
        if not bool(reg.get("enabled", True)):
            summary = {
                "op_id": op_id,
                "status": "skipped",
                "reason": "disabled_in_registry",
                "iterations": 0,
                "best_score": 0.0,
                "best_candidate_signature": "",
                "task_run_dir": str(run_dir / op_id),
            }
            task_summaries.append(summary)
            state = update_task_state(run_dir=run_dir, state=state, op_id=op_id, status="skipped", summary=summary)
            continue

        module_name = str(reg["pack_module"])
        pack_cfg_path = repo_root / str(reg["pack_config"])
        platform = str(reg["platform"])
        platform_cfg_path = repo_root / str(platforms_cfg[platform])
        task_run_dir = run_dir / op_id
        ensure_dir(task_run_dir)

        try:
            mod = importlib.import_module(module_name)
            run_fn = getattr(mod, "run_task")
            summary = run_fn(
                repo_root=repo_root,
                task=task,
                registry_entry=reg,
                global_cfg=global_cfg,
                pack_cfg_path=pack_cfg_path,
                platform_cfg_path=platform_cfg_path,
                run_dir=task_run_dir,
            )
            summary["op_id"] = op_id
            summary["task_run_dir"] = str(task_run_dir)
            task_summaries.append(summary)
            state = update_task_state(
                run_dir=run_dir,
                state=state,
                op_id=op_id,
                status=str(summary.get("status", "completed")),
                summary=summary,
            )
        except Exception as exc:  # noqa: BLE001
            summary = {
                "op_id": op_id,
                "status": "error",
                "error": str(exc),
                "iterations": 0,
                "best_score": 0.0,
                "best_candidate_signature": "",
                "task_run_dir": str(task_run_dir),
            }
            task_summaries.append(summary)
            state = update_task_state(run_dir=run_dir, state=state, op_id=op_id, status="error", summary=summary)
            if bool(exec_cfg.get("stop_on_error", False)):
                state["status"] = "failed"
                save_state(run_dir, state)
                break

    if state.get("status") != "failed":
        state["status"] = "completed"
        save_state(run_dir, state)

    dashboard = build_portfolio_dashboard(run_dir=run_dir, task_summaries=task_summaries)
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "state": str(run_dir / "state.json"),
        "portfolio": str(portfolio_cfg_path),
        "task_summaries": task_summaries,
        **dashboard,
    }
    dump_json(run_dir / "run_summary.json", summary)
    return summary


def load_run_status(*, run_dir: Path) -> dict[str, Any]:
    state = load_state(run_dir)
    return {
        "run_dir": str(run_dir),
        "status": state.get("status", "unknown"),
        "tasks": state.get("tasks", []),
    }

