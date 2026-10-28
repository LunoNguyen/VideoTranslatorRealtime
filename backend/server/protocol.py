"""WebSocket message helpers: server->UI builders and UI->server validation."""
from __future__ import annotations

import json
import re
from typing import Any

SRC_LANGS = {"auto", "en", "ja"}
TGT_LANGS = {"vi"}
_MODEL_RE = re.compile(r"^[\w.\-]{1,64}$")

HINT_MODEL_MISSING = "Chạy scripts/download_models.py"
HINT_CUDA_OOM = "Chọn model nhỏ hơn hoặc device=cpu"


class BadMessage(ValueError):
    """Client message is malformed or unknown (message text is shown to the UI)."""


def status(level: str, code: str, message: str) -> dict:
    return {"type": "status", "level": level, "code": code, "message": message}


def config_msg(cfg: dict) -> dict:
    return {"type": "config", "config": cfg}


def apps_msg(apps: list[dict]) -> dict:
    return {"type": "apps", "apps": apps}


def subtitle_msg(fields: dict) -> dict:
    return {"type": "subtitle", **fields}


def parse(raw: str) -> tuple[str, dict]:
    """Parse and validate a UI message; returns (type, normalized payload)."""
    try:
        msg = json.loads(raw)
    except (ValueError, TypeError) as e:
        raise BadMessage("Tin nhắn không phải JSON hợp lệ") from e
    if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
        raise BadMessage("Thiếu trường 'type'")
    kind = msg["type"]
    if kind in ("list_apps", "status"):
        return kind, {}
    if kind == "set_allowed":
        exe, allowed = msg.get("exe"), msg.get("allowed")
        if not isinstance(exe, str) or not exe.strip() or not isinstance(allowed, bool):
            raise BadMessage("set_allowed cần 'exe' (chuỗi) và 'allowed' (bool)")
        return kind, {"exe": exe.strip().lower(), "allowed": allowed}
    if kind == "set_langs":
        src, tgt = msg.get("src"), msg.get("tgt")
        if src not in SRC_LANGS or tgt not in TGT_LANGS:
            raise BadMessage("Ngôn ngữ không được hỗ trợ")
        return kind, {"src": src, "tgt": tgt}
    if kind == "set_tts":
        if not isinstance(msg.get("enabled"), bool):
            raise BadMessage("set_tts cần 'enabled' (bool)")
        return kind, {"enabled": msg["enabled"]}
    if kind in ("set_streaming", "set_speaker_colors"):
        if not isinstance(msg.get("enabled"), bool):
            raise BadMessage(f"{kind} cần 'enabled' (bool)")
        return kind, {"enabled": msg["enabled"]}
    if kind == "set_volume":
        v = msg.get("value")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
            raise BadMessage("set_volume cần 'value' trong khoảng 0..1")
        return kind, {"value": float(v)}
    if kind == "set_models":
        m = msg.get("asr_model")
        if not isinstance(m, str) or not _MODEL_RE.match(m):
            raise BadMessage("set_models cần 'asr_model' hợp lệ")
        return kind, {"asr_model": m}
    raise BadMessage(f"Loại tin nhắn không hỗ trợ: {kind[:40]}")


def clean(obj: Any) -> Any:
    """JSON-safe deep copy (used for config echo)."""
    return json.loads(json.dumps(obj))
