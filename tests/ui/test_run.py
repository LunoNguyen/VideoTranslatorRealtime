"""Launcher helpers."""
from __future__ import annotations

import socket
import tempfile

import run


def test_port_in_use_detected():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    try:
        assert run.port_open("127.0.0.1", s.getsockname()[1])
    finally:
        port = s.getsockname()[1]
        s.close()
    assert not run.port_open("127.0.0.1", port)


def test_port_busy_message_vietnamese():
    assert "8765" in run.port_busy_message(8765) and "bị chiếm" in run.port_busy_message(8765)


def test_tail_reads_only_last_bytes():
    with tempfile.TemporaryFile() as f:
        f.write(b"x" * 1_000_000 + b"\nlast line")
        out = run._tail(f)
    assert out.endswith("last line") and len(out) <= run.TAIL_BYTES


def test_malformed_config_shows_error_and_exits_nonzero(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    shutil.copy(Path(run.__file__).with_name("config.default.json"), tmp_path / "config.default.json")
    (tmp_path / "config.local.json").write_text("{ not json", encoding="utf-8")
    shown: list[str] = []
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "show_error", shown.append)
    monkeypatch.setattr(run.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no backend")))
    assert run.main() not in (0, None)
    assert len(shown) == 1 and "config.local.json" in shown[0] and "dòng 1" in shown[0]
