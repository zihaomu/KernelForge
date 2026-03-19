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


def now_ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        try:
            return yaml.safe_load(text) or {}
        except Exception:  # noqa: BLE001
            # Some triage manifests contain unquoted wildcard tokens like
            # `**/CMakeLists.txt`, which PyYAML interprets as alias syntax.
            # Fall back to our permissive parser instead of failing hard.
            return _mini_yaml_load(text)
    return _mini_yaml_load(text)


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
    # Minimal YAML loader for this project:
    # supports nested dict/list with spaces indentation and simple scalar values.
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, stripped))

    def looks_like_mapping(token: str) -> bool:
        t = token.strip()
        lowered = t.lower()
        if lowered.startswith(("http://", "https://", '"http://', '"https://', "'http://", "'https://")):
            return False
        if t.startswith("{") and t.endswith("}"):
            return False
        if ":" not in t:
            return False
        left = t.split(":", 1)[0].strip()
        if not left:
            return False
        # URL-like tokens are scalars in our configs.
        if "/" in left:
            return False
        return True

    def parse_block(idx: int, indent: int) -> tuple[Any, int]:
        if idx >= len(lines):
            return {}, idx
        is_list = lines[idx][1].startswith("- ")
        if is_list:
            out_list = []
            while idx < len(lines):
                ind, content = lines[idx]
                if ind != indent or not content.startswith("- "):
                    break
                item_payload = content[2:].strip()
                idx += 1
                if not item_payload:
                    item, idx = parse_block(idx, indent + 2)
                    out_list.append(item)
                    continue
                if looks_like_mapping(item_payload):
                    key, value = item_payload.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    item_dict: dict[str, Any] = {}
                    if value:
                        item_dict[key] = _parse_scalar(value)
                    else:
                        nested, idx = parse_block(idx, indent + 2)
                        item_dict[key] = nested
                    while idx < len(lines):
                        ind2, cont2 = lines[idx]
                        if ind2 < indent + 2:
                            break
                        if ind2 == indent and cont2.startswith("- "):
                            break
                        if ind2 != indent + 2 or cont2.startswith("- "):
                            break
                        if ":" not in cont2:
                            idx += 1
                            continue
                        k2, v2 = cont2.split(":", 1)
                        k2 = k2.strip()
                        v2 = v2.strip()
                        idx += 1
                        if v2:
                            item_dict[k2] = _parse_scalar(v2)
                        else:
                            nested2, idx = parse_block(idx, indent + 4)
                            item_dict[k2] = nested2
                    out_list.append(item_dict)
                else:
                    out_list.append(_parse_scalar(item_payload))
            return out_list, idx

        out_dict: dict[str, Any] = {}
        while idx < len(lines):
            ind, content = lines[idx]
            if ind < indent:
                break
            if ind > indent:
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
    if isinstance(parsed, dict):
        return parsed
    return {"_": parsed}


def dump_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(repo_root: Path, path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")
