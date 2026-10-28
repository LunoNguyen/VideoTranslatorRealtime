import json
import shutil
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.__main__ import main
from backend.asr.whisper import CudaOutOfMemoryError, Transcript
from backend.capture.sessions import AppSession
from backend.config import load_config
from backend.errors import ModelMissingError
from backend.server.app import Models, Services, create_app, own_root_pid
import asyncio
from backend.vad.segmenter import Segment

REPO = Path(__file__).resolve().parents[2]


class FakeSource:
    sample_rate, channels = 16000, 1

    def __init__(self, pid):
        self.pid, self.error, self.stopped = pid, None, False

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def read(self, timeout):
        time.sleep(0.005)
        return np.full((320, 1), 0.1, np.float32)


class OneSeg:
    def __init__(self):
        self.done = False

    def push(self, frame):
        if self.done:
            return []
        self.done = True
        return [Segment(np.ones(16000, np.float32), 0.0, 1.0)]

    def flush(self):
        return []


class FakeASR:
    def transcribe(self, audio, lang):
        return Transcript("hello", "en", 0.0)


class FakeMT:
    def translate(self, text, src, tgt="vi"):
        return "xin chào"


def good_models(cfg, on_tts_error):
    return Models(FakeASR(), FakeMT(), None, OneSeg)


@pytest.fixture
def root(tmp_path):
    shutil.copy(REPO / "config.default.json", tmp_path / "config.default.json")
    return tmp_path


def make_client(root, sessions=None, load=good_models, source_factory=FakeSource, allowed=(), retry_s=10.0, proc_table=None, own_pid=1):
    if allowed:
        (root / "config.local.json").write_text(json.dumps({"audio": {"allowed_apps": list(allowed)}}))
    sv = Services(
        root=root,
        list_sessions=lambda: list(sessions or []),
        source_factory=source_factory,
        load_models=load,
        own_pid=own_pid,
        poll_interval=0.05,
        retry_s=retry_s,
        proc_table=proc_table or (lambda: {}),
    )
    return TestClient(create_app(load_config(root), sv))


def recv_until(ws, pred, limit=60):
    for _ in range(limit):
        msg = ws.receive_json()
        if pred(msg):
            return msg
    raise AssertionError("expected message not received")


def test_set_allowed_persists_and_broadcasts_apps(root):
    sessions = [AppSession("chrome.exe", 100, "Chrome", True)]
    with make_client(root, sessions) as c, c.websocket_connect("/ws") as ws:
        ws.send_json({"type": "set_allowed", "exe": "Chrome.exe", "allowed": True})
        msg = recv_until(ws, lambda m: m["type"] == "apps" and m["apps"] and m["apps"][0]["allowed"])
        assert msg["apps"][0] == {"exe": "chrome.exe", "display_name": "Chrome", "active": True, "allowed": True}
    assert json.loads((root / "config.local.json").read_text())["audio"]["allowed_apps"] == ["chrome.exe"]


def test_subtitle_forwarded_to_client(root):
    sessions = [AppSession("chrome.exe", 100, "Chrome", True)]
    with make_client(root, sessions, allowed=["chrome.exe"]) as c, c.websocket_connect("/ws") as ws:
        sub = recv_until(ws, lambda m: m["type"] == "subtitle")
        assert (sub["src_text"], sub["vi_text"], sub["src_lang"]) == ("hello", "xin chào", "en")


def test_unknown_message_returns_error_status(root):
    with make_client(root) as c, c.websocket_connect("/ws") as ws:
        for bad in ['{"type": "nope"}', "not json", "[1]", '{"type": "set_volume", "value": "x"}']:
            ws.send_text(bad)
            msg = recv_until(ws, lambda m: m["type"] == "status" and m["code"] == "bad_message")
            assert msg["level"] == "error"
        ws.send_json({"type": "status"})  # connection still alive
        recv_until(ws, lambda m: m["type"] == "config")


@pytest.mark.parametrize(
    "exc,code,hint",
    [
        (ModelMissingError("small"), "model_missing", "Chạy scripts/download_models.py"),
        (CudaOutOfMemoryError("oom"), "cuda_oom", "Chọn model nhỏ hơn hoặc device=cpu"),
    ],
)
def test_model_missing_reports_status(root, exc, code, hint):
    def load(cfg, cb):
        raise exc

    with make_client(root, load=load) as c, c.websocket_connect("/ws") as ws:
        msg = recv_until(ws, lambda m: m["type"] == "status" and m["code"] == code)
        assert msg["level"] == "error" and hint in msg["message"]
        ws.send_json({"type": "set_langs", "src": "en", "tgt": "vi"})  # server still serves
        assert recv_until(ws, lambda m: m["type"] == "config")["config"]["langs"]["src"] == "en"


def test_failed_source_reports_app_closed(root):
    class Dying(FakeSource):
        def start(self):
            self.error = RuntimeError("gone")

    sessions = [AppSession("chrome.exe", 3_999_999, "Chrome", True)]  # pid that cannot exist
    with make_client(root, sessions, allowed=["chrome.exe"], source_factory=Dying) as c, c.websocket_connect("/ws") as ws:
        msg = recv_until(ws, lambda m: m["type"] == "status" and m["code"] == "app_closed")
        assert msg["level"] == "warn" and "chrome.exe" in msg["message"]


def test_port_in_use_reports_error(root, capsys):
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen()
    port = holder.getsockname()[1]
    (root / "config.local.json").write_text(json.dumps({"server": {"port": port}}))
    try:
        with pytest.raises(SystemExit) as e:
            main(root)
    finally:
        holder.close()
    assert e.value.code not in (0, None)
    assert str(port) in capsys.readouterr().err


def test_own_root_pid_finds_topmost_same_exe():
    table = {
        1: ("explorer.exe", 0),
        10: ("python.exe", 1),  # venv launcher
        11: ("python.exe", 10),  # real interpreter (us)
        12: ("chrome.exe", 11),
    }
    assert own_root_pid(11, table) == 10
    assert own_root_pid(10, table) == 10
    assert own_root_pid(12, table) == 12
    assert own_root_pid(99, table) == 99


def test_binary_frame_does_not_crash_connection(root):
    with make_client(root) as c, c.websocket_connect("/ws") as ws:
        ws.send_bytes(b"ab")
        recv_until(ws, lambda m: m["type"] == "status" and m["code"] == "bad_message")
        ws.send_json({"type": "status"})
        recv_until(ws, lambda m: m["type"] == "config")


def test_failed_start_is_retried_with_backoff(root):
    attempts: list[float] = []
    second = threading.Event()

    class Flaky(FakeSource):
        def start(self):
            attempts.append(time.monotonic())
            if len(attempts) == 1:
                raise RuntimeError("busy")
            second.set()

    sessions = [AppSession("chrome.exe", 100, "Chrome", True)]
    with make_client(root, sessions, allowed=["chrome.exe"], source_factory=Flaky, retry_s=0.3) as c,             c.websocket_connect("/ws") as ws:
        assert second.wait(5)
    assert attempts[1] - attempts[0] >= 0.3
    assert len(attempts) == 2


def test_died_source_with_live_process_is_restarted(root):
    import os

    starts: list[int] = []
    restarted = threading.Event()

    class Dies(FakeSource):
        def start(self):
            starts.append(1)
            if len(starts) == 1:
                self.error = RuntimeError("died")  # reaped on the next tick
            else:
                restarted.set()

    sessions = [AppSession("chrome.exe", os.getpid(), "Chrome", True)]  # a pid that is alive
    with make_client(root, sessions, allowed=["chrome.exe"], source_factory=Dies, retry_s=0.2) as c,             c.websocket_connect("/ws") as ws:
        assert restarted.wait(5)


def test_tts_error_from_worker_thread_reaches_client_and_task_is_released(root):
    with make_client(root) as c, c.websocket_connect("/ws") as ws:
        hub = c.app.state.hub
        recv_until(ws, lambda m: m["type"] == "status" and m["code"] == "models_ready")
        threading.Thread(target=hub.on_tts_error, args=(RuntimeError("no voice"),)).start()
        msg = recv_until(ws, lambda m: m["type"] == "status" and m["code"] == "tts_error")
        assert "no voice" in msg["message"]
        for _ in range(100):
            if not hub._bg:
                break
            time.sleep(0.02)
        assert not hub._bg


def test_app_closed_via_tracker_stop_path(root):
    sessions = [AppSession("chrome.exe", 100, "Chrome", True)]
    with make_client(root, sessions, allowed=["chrome.exe"]) as c, c.websocket_connect("/ws") as ws:
        recv_until(ws, lambda m: m["type"] == "apps" and m["apps"] and m["apps"][0]["active"])
        time.sleep(0.2)  # let the source start
        sessions.clear()  # app vanished from the session list
        msg = recv_until(ws, lambda m: m["type"] == "status" and m["code"] == "app_closed")
        assert msg["level"] == "warn" and "chrome.exe" in msg["message"]


def test_dead_client_does_not_block_live_one(root):
    class Dead:
        async def send_json(self, _):
            raise RuntimeError("gone")

    with make_client(root) as c, c.websocket_connect("/ws") as ws:
        hub = c.app.state.hub
        dead = Dead()
        hub.clients.add(dead)
        ws.send_json({"type": "set_langs", "src": "ja", "tgt": "vi"})
        recv_until(ws, lambda m: m["type"] == "config" and m["config"]["langs"]["src"] == "ja")
        assert dead not in hub.clients


def test_proc_table_snapshot_taken_off_the_event_loop(root):
    threads: list[str] = []
    seen = threading.Event()

    def table():
        try:
            asyncio.get_running_loop()
            threads.append("loop")  # called on the event loop thread
        except RuntimeError:
            threads.append("worker")
        seen.set()
        return {}

    sessions = [AppSession("chrome.exe", 100, "Chrome", True)]
    with make_client(root, sessions, allowed=["chrome.exe"], proc_table=table) as c, c.websocket_connect("/ws"):
        assert seen.wait(5)
        assert set(threads) == {"worker"}


@pytest.mark.parametrize("origin", ["http://evil.example", "https://evil.example", "http://127.0.0.1:8765", "null", "file://"])
def test_browser_origin_rejected(root, origin):
    from starlette.websockets import WebSocketDisconnect as WSDisc

    with make_client(root) as c:
        with pytest.raises(WSDisc) as e, c.websocket_connect("/ws", headers={"origin": origin}) as ws:
            ws.receive_json()
        assert e.value.code == 1008
        assert not c.app.state.hub.clients


def test_no_origin_accepted(root):
    with make_client(root) as c, c.websocket_connect("/ws") as ws:
        recv_until(ws, lambda m: m["type"] == "config")


def drain(ws) -> list[dict]:
    """Return every message received so far; a bad message acts as an end-of-stream marker."""
    ws.send_text("marker")
    msgs = []
    while not msgs or msgs[-1].get("code") != "bad_message":
        msgs.append(ws.receive_json())
    return msgs


def test_self_tree_ancestor_warns_once(root):
    table = {1: ("explorer.exe", 0), 5: ("windowsterminal.exe", 1), 10: ("python.exe", 5)}
    started: list[int] = []

    class Src(FakeSource):
        def start(self):
            started.append(self.pid)

    sessions = [AppSession("windowsterminal.exe", 5, "Terminal", True)]
    c = make_client(root, sessions, allowed=["windowsterminal.exe"], source_factory=Src,
                    proc_table=lambda: table, own_pid=10)
    hub, sent = c.app.state.hub, []
    orig = hub.status

    async def record(level, code, message, ws=None):
        sent.append((level, code, message, ws))
        await orig(level, code, message, ws)

    hub.status = record
    with c:
        time.sleep(0.3)  # several ticks
        warns = [m for m in sent if m[1] == "self_tree"]
        assert len(warns) == 1 and warns[0][0] == "warn" and "windowsterminal.exe" in warns[0][2]
        with c.websocket_connect("/ws") as ws:  # a late client still learns about it
            msgs = drain(ws)
            assert any(m.get("code") == "self_tree" and "windowsterminal.exe" in m["message"] for m in msgs)
    assert 5 not in started


def test_models_ready_sent_before_warnings_and_warnings_resent(root):
    def load(cfg, cb):
        return Models(FakeASR(), FakeMT(), None, OneSeg, [("cuda_unavailable", "Không dùng được CUDA, đã chuyển sang CPU")])

    with make_client(root, load=load) as c:
        hub = c.app.state.hub
        with c.websocket_connect("/ws") as ws:
            for _ in range(100):  # wait for the load to finish
                if hub.models_ready:
                    break
                time.sleep(0.02)
        with c.websocket_connect("/ws") as ws:  # late client gets ready + the stored warning, in order
            codes = [m.get("code") for m in drain(ws) if m["type"] == "status"]
    assert "models_ready" in codes and "cuda_unavailable" in codes
    assert codes.index("models_ready") < codes.index("cuda_unavailable")


def test_live_broadcast_orders_models_ready_before_warning(root):
    def load(cfg, cb):
        return Models(FakeASR(), FakeMT(), None, OneSeg, [("cuda_oom", "Hết VRAM, đã chuyển sang CPU")])

    c = make_client(root, load=load)
    hub, sent = c.app.state.hub, []
    orig = hub.status

    async def record(level, code, message, ws=None):
        sent.append(code)
        await orig(level, code, message, ws)

    hub.status = record
    with c:
        for _ in range(100):
            if "cuda_oom" in sent:
                break
            time.sleep(0.02)
    assert sent.index("models_ready") < sent.index("cuda_oom")


def test_apps_payload_merges_rows_by_exe(root):
    sessions = [AppSession("msedgewebview2.exe", 100, "WebView", False),
                AppSession("msedgewebview2.exe", 200, "WebView", True)]
    hub = make_client(root).app.state.hub
    assert hub.apps_payload(sessions) == [
        {"exe": "msedgewebview2.exe", "display_name": "WebView", "active": True, "allowed": False}]


@pytest.mark.parametrize("enabled", [False, True])
def test_file_logging_opt_in(tmp_path, enabled):
    import logging

    from backend.__main__ import setup_logging

    root_logger = logging.getLogger()
    before = list(root_logger.handlers)
    try:
        setup_logging({"logging": {"enabled": enabled}}, tmp_path)
        logging.getLogger("backend.test").warning("probe")
        added = [h for h in root_logger.handlers if h not in before]
        assert (tmp_path / "logs").exists() is enabled
        assert any(isinstance(h, logging.FileHandler) for h in added) is enabled
        if enabled:
            for h in added:
                h.flush()
            assert "probe" in (tmp_path / "logs" / "backend.log").read_text(encoding="utf-8")
    finally:
        for h in [h for h in root_logger.handlers if h not in before]:
            root_logger.removeHandler(h)
            h.close()
