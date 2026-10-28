"""WSClient message parsing (no network)."""
from __future__ import annotations

import json

from ui.ws_client import WSClient


def test_ws_client_parses_messages(qtbot):
    c = WSClient("ws://127.0.0.1:1/ws", autoconnect=False)
    got: dict[str, list] = {k: [] for k in ("subtitle", "apps", "status", "config")}
    c.subtitle.connect(lambda d: got["subtitle"].append(d))
    c.apps.connect(lambda d: got["apps"].append(d))
    c.status.connect(lambda d: got["status"].append(d))
    c.config.connect(lambda d: got["config"].append(d))

    c.handle_text(json.dumps({"type": "subtitle", "vi_text": "xin chào", "src_text": "hello"}))
    c.handle_text(json.dumps({"type": "apps", "apps": [{"exe": "a.exe"}]}))
    c.handle_text(json.dumps({"type": "status", "level": "info", "code": "models_ready", "message": "ok"}))
    c.handle_text(json.dumps({"type": "config", "config": {"overlay": {"hold_seconds": 4.0}}}))
    c.handle_text("not json")
    c.handle_text(json.dumps({"type": "weird"}))

    assert got["subtitle"][0]["vi_text"] == "xin chào"
    assert got["apps"] == [[{"exe": "a.exe"}]]
    assert got["status"][0]["code"] == "models_ready"
    assert got["config"][0]["overlay"]["hold_seconds"] == 4.0


def test_send_when_disconnected_is_noop(qtbot):
    c = WSClient("ws://127.0.0.1:1/ws", autoconnect=False)
    assert c.send("list_apps") is False


def test_qt_client_connects_to_real_server(qtbot, tmp_path):
    """QWebSocket sends no http(s) Origin, so the backend's Origin guard lets it in."""
    import shutil
    import socket
    import threading
    from pathlib import Path

    import uvicorn

    from backend.config import load_config
    from backend.server.app import Models, Services, create_app

    repo = Path(__file__).resolve().parents[2]
    shutil.copy(repo / "config.default.json", tmp_path / "config.default.json")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    sv = Services(tmp_path, lambda: [], lambda pid: None,
                  lambda cfg, cb: Models(None, None, None, lambda: None), own_pid=1, poll_interval=0.05)
    server = uvicorn.Server(uvicorn.Config(create_app(load_config(tmp_path), sv), "127.0.0.1", port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    c = WSClient(f"ws://127.0.0.1:{port}/ws", autoconnect=False)
    got: list[dict] = []
    c.config.connect(got.append)
    try:
        with qtbot.waitSignal(c.connected, timeout=10000):
            c.start()
        qtbot.waitUntil(lambda: bool(got), timeout=5000)
    finally:
        c.stop()
        server.should_exit = True
        t.join(10)
    assert got[0]["server"]["port"] == 8765
