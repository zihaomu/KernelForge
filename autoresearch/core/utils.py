from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # noqa: BLE001
    yaml = None


def repo_root_from_file(current_file: str) -> Path:
    return Path(current_file).resolve().parents[2]


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def now_run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if body:
        body += "\n"
    path.write_text(body, encoding="utf-8")


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if token in ("null", "None", "~"):
        return None
    if token in ("true", "True"):
        return True
    if token in ("false", "False"):
        return False
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(x.strip()) for x in inner.split(",")]
    try:
        if "." in token:
            return float(token)
        return int(token)
    except ValueError:
        return token


def _mini_yaml_load(text: str) -> dict[str, Any]:
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, stripped))

    def parse_block(idx: int, indent: int) -> tuple[Any, int]:
        if idx >= len(lines):
            return {}, idx
        is_list = lines[idx][1].startswith("- ")
        if is_list:
            out = []
            while idx < len(lines):
                i, content = lines[idx]
                if i != indent or not content.startswith("- "):
                    break
                payload = content[2:].strip()
                idx += 1
                if not payload:
                    nested, idx = parse_block(idx, indent + 2)
                    out.append(nested)
                    continue
                if ":" in payload and not payload.startswith(("http://", "https://")):
                    key, value = payload.split(":", 1)
                    item: dict[str, Any] = {}
                    key = key.strip()
                    value = value.strip()
                    if value:
                        item[key] = _parse_scalar(value)
                    else:
                        nested, idx = parse_block(idx, indent + 2)
                        item[key] = nested
                    while idx < len(lines):
                        j, row = lines[idx]
                        if j < indent + 2 or row.startswith("- "):
                            break
                        if ":" not in row:
                            idx += 1
                            continue
                        k2, v2 = row.split(":", 1)
                        k2 = k2.strip()
                        v2 = v2.strip()
                        idx += 1
                        if v2:
                            item[k2] = _parse_scalar(v2)
                        else:
                            nested2, idx = parse_block(idx, indent + 4)
                            item[k2] = nested2
                    out.append(item)
                else:
                    out.append(_parse_scalar(payload))
            return out, idx

        out_dict: dict[str, Any] = {}
        while idx < len(lines):
            i, content = lines[idx]
            if i < indent:
                break
            if i > indent:
                idx += 1
                continue
            if ":" not in content:
                idx += 1
                continue
            key, value = content.split(":", 1)
            key = key.strip()
            value = value.strip()
            idx += 1
            if value:
                out_dict[key] = _parse_scalar(value)
            else:
                nested, idx = parse_block(idx, indent + 2)
                out_dict[key] = nested
        return out_dict, idx

    parsed, _ = parse_block(0, 0)
    return parsed if isinstance(parsed, dict) else {"_": parsed}


def load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        try:
            loaded = yaml.safe_load(text)
            return loaded or {}
        except Exception:  # noqa: BLE001
            return _mini_yaml_load(text)
    return _mini_yaml_load(text)

