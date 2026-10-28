import json
from pathlib import Path

from backend.config import load_config, save_local

DEFAULT = {
    "server": {"host": "127.0.0.1", "port": 8765},
    "tts": {"enabled": False, "volume": 1.0},
}


def make_root(tmp_path: Path, local: dict | None = None) -> Path:
    (tmp_path / "config.default.json").write_text(json.dumps(DEFAULT), encoding="utf-8")
    if local is not None:
        (tmp_path / "config.local.json").write_text(json.dumps(local), encoding="utf-8")
    return tmp_path


def test_local_overrides_default(tmp_path):
    root = make_root(tmp_path, {"tts": {"enabled": True}})
    cfg = load_config(root)
    assert cfg["tts"]["enabled"] is True
    assert cfg["tts"]["volume"] == 1.0
    assert cfg["server"]["port"] == 8765


def test_missing_local_returns_default(tmp_path):
    cfg = load_config(make_root(tmp_path))
    assert cfg == DEFAULT


def test_save_local_roundtrip(tmp_path):
    root = make_root(tmp_path)
    save_local(root, {"tts": {"enabled": True, "volume": 1.0}})
    on_disk = json.loads((root / "config.local.json").read_text(encoding="utf-8"))
    assert on_disk == {"tts": {"enabled": True}}  # only non-default keys
    assert load_config(root)["tts"]["enabled"] is True
    save_local(root, {"server": {"port": 9000}})  # merges with existing local
    assert load_config(root)["tts"]["enabled"] is True
    assert load_config(root)["server"]["port"] == 9000


def test_host_forced_loopback(tmp_path):
    root = make_root(tmp_path, {"server": {"host": "0.0.0.0"}})
    assert load_config(root)["server"]["host"] == "127.0.0.1"


def test_malformed_local_raises_clear_error(tmp_path):
    import pytest

    from backend.config import ConfigError

    root = make_root(tmp_path)
    (root / "config.local.json").write_text('{\n  "tts": {\n    "enabled": tru\n  }\n}', encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        load_config(root)
    msg = str(e.value)
    assert "config.local.json" in msg and "dòng 3" in msg and "cột" in msg


def test_save_local_is_atomic(tmp_path, monkeypatch):
    import os

    import pytest

    root = make_root(tmp_path, {"tts": {"enabled": True}})
    original = (root / "config.local.json").read_text(encoding="utf-8")
    calls = []

    def boom(src, dst):
        calls.append((Path(src).name, Path(dst).name))
        raise OSError("disk full")

    monkeypatch.setattr("backend.config.os.replace", boom)
    with pytest.raises(OSError):
        save_local(root, {"server": {"port": 9000}})
    assert calls and calls[0][1] == "config.local.json" and calls[0][0] != "config.local.json"
    assert (root / "config.local.json").read_text(encoding="utf-8") == original
    monkeypatch.undo()
    save_local(root, {"server": {"port": 9000}})
    assert load_config(root)["server"]["port"] == 9000
    assert sorted(p.name for p in root.iterdir()) == ["config.default.json", "config.local.json"]
