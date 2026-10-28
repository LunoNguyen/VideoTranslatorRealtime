"""Config loader: config.default.json <- config.local.json (deep merge)."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_FILE = "config.default.json"
LOCAL_FILE = "config.local.json"
LOOPBACK = "127.0.0.1"


class ConfigError(ValueError):
    """A config file exists but cannot be parsed (message is user-facing Vietnamese)."""


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(
            f"Tệp cấu hình {path} bị lỗi JSON ở dòng {e.lineno}, cột {e.colno} ({e.msg}). "
            "Hãy sửa hoặc xoá tệp này rồi chạy lại."
        ) from e
    except UnicodeDecodeError as e:
        raise ConfigError(f"Tệp cấu hình {path} không phải UTF-8. Hãy sửa hoặc xoá tệp này rồi chạy lại.") from e
    if not isinstance(data, dict):
        raise ConfigError(f"Tệp cấu hình {path} phải là một đối tượng JSON {{...}}. Hãy sửa hoặc xoá tệp này.")
    return data


def deep_merge(base: dict, patch: dict) -> dict:
    """Return a new dict: dicts merge recursively, other values are replaced."""
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _diff(value: dict, default: dict) -> dict:
    """Keep only keys of `value` that differ from `default` (recursively)."""
    out: dict[str, Any] = {}
    for k, v in value.items():
        d = default.get(k)
        if isinstance(v, dict) and isinstance(d, dict):
            sub = _diff(v, d)
            if sub:
                out[k] = sub
        elif k not in default or v != d:
            out[k] = v
    return out


def load_config(root: Path) -> dict:
    """Load merged config; server.host is always forced to loopback."""
    root = Path(root)
    cfg = deep_merge(_read(root / DEFAULT_FILE), _read(root / LOCAL_FILE))
    cfg.setdefault("server", {})["host"] = LOOPBACK
    return cfg


def save_local(root: Path, patch: dict) -> None:
    """Merge patch into local config and write only non-default keys."""
    root = Path(root)
    default = _read(root / DEFAULT_FILE)
    merged = deep_merge(_read(root / LOCAL_FILE), patch)
    tmp = root / (LOCAL_FILE + ".tmp")
    tmp.write_text(json.dumps(_diff(merged, default), indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        os.replace(tmp, root / LOCAL_FILE)  # atomic: a crash never leaves a half-written file
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
