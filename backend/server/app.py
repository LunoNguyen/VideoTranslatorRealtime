"""FastAPI app: WebSocket hub wiring capture -> pipeline -> UI."""
from __future__ import annotations

import asyncio
import copy
import logging
import os
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from backend.asr.whisper import CudaOutOfMemoryError
from backend.capture.mixer import Mixer
from backend.capture.sessions import AppSession, PermissionTracker, live_proc_table
from backend.capture.source import LoopbackUnsupportedError
from backend.config import ConfigError, deep_merge, save_local
from backend.errors import ModelMissingError
from backend.pipeline.runner import Pipeline, Subtitle
from backend.server import protocol as P

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Models:
    """Loaded engines; `warnings` are (code, message) notes shown as warn status."""

    asr: Any
    mt: Any
    tts: Any  # TTSPlayer-like (say/set_enabled/set_volume/close) or None
    make_segmenter: Callable[[], Any]
    warnings: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Services:
    """Injectable dependencies (tests pass fakes)."""

    root: Path
    list_sessions: Callable[[], list[AppSession]]
    source_factory: Callable[[int], Any]
    load_models: Callable[[dict, Callable[[Exception], None]], Models]
    own_pid: int
    poll_interval: float = 2.0
    retry_s: float = 10.0  # min delay before retrying a failed source start
    proc_table: Callable[[], dict] = live_proc_table  # process snapshot, read off the event loop


def own_root_pid(pid: int | None = None, table: dict[int, tuple[str, int]] | None = None) -> int:
    """Topmost ancestor with the same exe name (venv python.exe launcher spawns a same-named child).

    `table` maps pid -> (lower-case exe name, parent pid); defaults to the live process table.
    """
    pid = os.getpid() if pid is None else pid
    if table is None:
        table = {}
        for p in psutil.process_iter(["name", "ppid"]):
            table[p.pid] = ((p.info["name"] or "").lower(), p.info["ppid"] or 0)
    if pid not in table:
        return pid
    name, cur, seen = table[pid][0], pid, {pid}
    while True:
        ppid = table[cur][1]
        if ppid in seen or ppid not in table or table[ppid][0] != name:
            return cur
        cur = ppid
        seen.add(cur)


def _self_tree_text(exe: str) -> str:
    return f"Không thể nghe {exe} vì nó là tiến trình cha của ứng dụng này"


def _pid_alive(pid: int) -> bool:
    try:
        return psutil.pid_exists(pid)
    except (OverflowError, ValueError, OSError):
        return False


def default_services(root: Path = ROOT) -> Services:
    def sessions() -> list[AppSession]:
        try:
            import comtypes

            comtypes.CoInitialize()  # pycaw needs COM in the worker thread
        except Exception:  # noqa: BLE001
            pass
        from backend.capture.sessions import list_sessions

        return list_sessions()

    def source_factory(pid: int):
        from backend.capture.source import ProcessLoopbackSource

        return ProcessLoopbackSource(pid)

    def load_models(cfg: dict, on_tts_error: Callable[[Exception], None]) -> Models:
        from backend.asr.whisper import WhisperASR
        from backend.gpu import register_cuda_dlls
        from backend.mt.nllb import NLLBTranslator
        from backend.tts.piper_tts import PiperTTS
        from backend.tts.player import TTSPlayer
        from backend.vad.segmenter import Segmenter

        register_cuda_dlls()
        a, warnings = cfg["asr"], []
        asr = WhisperASR(a["model"], a["device"], a["compute_type"], root / "models" / "asr")
        asr.load()
        if asr.fallback_reason == "cuda_oom":
            warnings.append(("cuda_oom", "Hết VRAM, đã chuyển sang CPU"))
        elif asr.fallback_reason:
            warnings.append(("cuda_unavailable", "Không dùng được CUDA, đã chuyển sang CPU"))
        mt = NLLBTranslator(root / cfg["mt"]["model_dir"], a["device"], cfg["mt"]["context_sentences"])
        mt.load()
        t = cfg["tts"]
        tts = TTSPlayer(PiperTTS(root / t["voice"]), on_error=on_tts_error)
        tts.set_enabled(t["enabled"])
        tts.set_volume(t["volume"])
        return Models(asr, mt, tts, Segmenter, warnings)

    return Services(root, sessions, source_factory, load_models, own_root_pid())


class Hub:
    """Server state shared by all WebSocket clients."""

    def __init__(self, config: dict, services: Services) -> None:
        self.cfg = copy.deepcopy(config)
        self.sv = services
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.models: Models | None = None
        self.model_status: dict | None = None  # error status while models are unavailable
        self.models_ready = False
        self.model_warnings: list[tuple[str, str]] = []  # warn notes of the loaded models (resent to new clients)
        self.pipeline: Pipeline | None = None
        self.pipe_task: asyncio.Task | None = None
        self.mixer = Mixer()
        self.tracker = PermissionTracker(set(self.cfg["audio"]["allowed_apps"]), services.own_pid)
        self.sources: dict[int, Any] = {}
        self.pid_exe: dict[int, str] = {}
        self.last_apps: list[dict] | None = None
        self.tick_lock = asyncio.Lock()
        self.restart_lock = asyncio.Lock()
        self.tasks: list[asyncio.Task] = []
        self._bg: set[asyncio.Task] = set()  # fire-and-forget tasks (kept referenced until done)
        self.retry_after: dict[int, float] = {}
        self.self_tree_warned: set[str] = set()  # exes already reported as our ancestors

    # -- messaging ----------------------------------------------------------
    async def broadcast(self, msg: dict) -> None:
        for ws in list(self.clients):
            await self.send(ws, msg)

    async def send(self, ws: WebSocket, msg: dict) -> None:
        try:
            await ws.send_json(msg)
        except Exception:  # noqa: BLE001 - dead client
            self.clients.discard(ws)

    async def status(self, level: str, code: str, message: str, ws: WebSocket | None = None) -> None:
        msg = P.status(level, code, message)
        await (self.send(ws, msg) if ws else self.broadcast(msg))

    def threadsafe_status(self, level: str, code: str, message: str) -> None:
        loop = self.loop
        if loop is None:
            return
        with suppress(RuntimeError):  # loop closed
            loop.call_soon_threadsafe(self._spawn_status, level, code, message)

    def _spawn_status(self, level: str, code: str, message: str) -> None:
        task = asyncio.get_running_loop().create_task(self.status(level, code, message))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    def on_tts_error(self, e: Exception) -> None:
        self.threadsafe_status("error", "tts_error", f"Lỗi đọc giọng: {e}")

    # -- config -------------------------------------------------------------
    async def patch_config(self, patch: dict) -> None:
        self.cfg = deep_merge(self.cfg, patch)
        try:
            await asyncio.to_thread(save_local, self.sv.root, patch)
        except (OSError, ConfigError) as e:
            await self.status("error", "config_save_failed", f"Không lưu được cấu hình: {e}")
        await self.broadcast(P.config_msg(P.clean(self.cfg)))

    # -- apps / capture -------------------------------------------------------
    def apps_payload(self, sessions: list[AppSession]) -> list[dict]:
        allowed = self.tracker.allowed
        rows: dict[str, dict] = {}  # one row per exe, even when it has several process trees
        for s in sessions:
            row = rows.get(s.exe)
            if row is None:
                rows[s.exe] = {"exe": s.exe, "display_name": s.display_name, "active": s.active,
                               "allowed": s.exe in allowed}
            else:
                row["active"] = row["active"] or s.active
        apps = list(rows.values())
        seen = set(rows)
        for exe in sorted(allowed - seen):  # allowed but not currently in a session
            apps.append({"exe": exe, "display_name": exe.removesuffix(".exe"), "active": False, "allowed": True})
        return apps

    async def tick(self, force_send: bool = False, ws: WebSocket | None = None) -> None:
        """One poll: refresh the app list, start/stop sources, report dead sources."""
        async with self.tick_lock:
            try:
                sessions = await asyncio.to_thread(self.sv.list_sessions)
            except Exception as e:  # noqa: BLE001
                log.exception("list_sessions failed")
                await self.status("error", "sessions_failed", f"Không đọc được danh sách ứng dụng: {e}")
                return
            for s in sessions:
                self.pid_exe[s.pid] = s.exe
            live_exes = {s.exe for s in sessions}
            await self._reap_failed()
            table = await asyncio.to_thread(self.sv.proc_table)
            to_start, to_stop = self.tracker.update(sessions, table)
            for exe in sorted(self.tracker.self_tree_exes - self.self_tree_warned):
                await self.status("warn", "self_tree", _self_tree_text(exe))
            self.self_tree_warned = set(self.tracker.self_tree_exes)
            for pid in to_stop:
                src = self.mixer.remove(pid) or self.sources.get(pid)
                self.sources.pop(pid, None)
                if src is not None:
                    await asyncio.to_thread(self._safe_stop, src)
                    exe = self.pid_exe.get(pid, str(pid))
                    if exe not in live_exes:
                        await self.status("warn", "app_closed", f"Ứng dụng {exe} đã đóng")
            for pid in to_start:
                if time.monotonic() < self.retry_after.get(pid, 0.0):
                    self.tracker.forget(pid)  # still backing off: ask again on a later tick
                    continue
                await self._start_source(pid)
            apps = self.apps_payload(sessions)
            if force_send or apps != self.last_apps:
                self.last_apps = apps
                await (self.send(ws, P.apps_msg(apps)) if ws else self.broadcast(P.apps_msg(apps)))

    @staticmethod
    def _safe_stop(src: Any) -> None:
        try:
            src.stop()
        except Exception:  # noqa: BLE001
            log.exception("source stop failed")

    def _backoff(self, pid: int) -> None:
        """Let the tracker re-offer `pid` for start, but not before retry_s has passed."""
        self.retry_after[pid] = time.monotonic() + self.sv.retry_s
        self.tracker.forget(pid)

    async def _start_source(self, pid: int) -> None:
        exe = self.pid_exe.get(pid, str(pid))
        try:
            src = self.sv.source_factory(pid)
            await asyncio.to_thread(src.start)
        except LoopbackUnsupportedError as e:
            self._backoff(pid)
            await self.status("error", "loopback_unsupported", f"Windows không hỗ trợ thu âm theo ứng dụng: {e}")
            return
        except Exception as e:  # noqa: BLE001
            log.exception("source start failed")
            self._backoff(pid)
            await self.status("error", "capture_failed", f"Không thu được âm thanh của {exe}: {e}")
            return
        self.sources[pid] = src
        self.mixer.add(src)

    async def _reap_failed(self) -> None:
        """Remove sources whose capture died and tell the UI why."""
        for pid in self.mixer.failed_pids():
            src = self.mixer.remove(pid)
            self.sources.pop(pid, None)
            if src is None:
                continue
            err = getattr(src, "error", None)
            await asyncio.to_thread(self._safe_stop, src)
            exe = self.pid_exe.get(pid, str(pid))
            self._backoff(pid)
            if not _pid_alive(pid):
                await self.status("warn", "app_closed", f"Ứng dụng {exe} đã đóng")
            else:
                await self.status("error", "capture_failed", f"Thu âm {exe} bị lỗi: {err}")

    async def apps_loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("apps tick failed")
            await asyncio.sleep(self.sv.poll_interval)

    # -- models / pipeline ----------------------------------------------------
    async def load_models(self) -> None:
        """(Re)load models in a worker thread and start a fresh pipeline."""
        async with self.restart_lock:
            await self._stop_engine()
            self.models_ready = False
            self.model_status = None
            self.model_warnings = []
            try:
                models = await asyncio.to_thread(self.sv.load_models, copy.deepcopy(self.cfg), self.on_tts_error)
            except ModelMissingError as e:
                log.error("model missing: %s", e)
                st = P.status("error", "model_missing", f"Thiếu model: {e}. {P.HINT_MODEL_MISSING}")
            except CudaOutOfMemoryError as e:
                st = P.status("error", "cuda_oom", f"Hết VRAM: {e}. {P.HINT_CUDA_OOM}")
            except Exception as e:  # noqa: BLE001
                log.exception("model load failed")
                if "out of memory" in str(e).lower():
                    st = P.status("error", "cuda_oom", f"Hết VRAM: {e}. {P.HINT_CUDA_OOM}")
                else:
                    st = P.status("error", "model_load_failed", f"Không tải được model: {e}")
            else:
                self.models, self.models_ready = models, True
                self.model_warnings = list(models.warnings)
                self._start_pipeline(models)
                await self.status("info", "models_ready", "Model đã sẵn sàng")
                for code, msg in self.model_warnings:  # after models_ready so the UI keeps showing them
                    await self.status("warn", code, msg)
                return
            self.model_status = st
            await self.broadcast(st)

    def _start_pipeline(self, models: Models) -> None:
        pl = Pipeline(
            self.mixer.frames(), models.make_segmenter(), models.asr, models.mt, models.tts,
            self._on_subtitle, self._on_pipeline_status,
        )
        pl.set_src_lang(self.cfg["langs"]["src"])
        pl.set_tts_enabled(self.cfg["tts"]["enabled"])
        self.pipeline = pl
        self.pipe_task = asyncio.create_task(self._run_pipeline(pl))

    async def _run_pipeline(self, pl: Pipeline) -> None:
        try:
            await pl.run()
        except asyncio.CancelledError:
            pl.stop()
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("pipeline crashed")
            await self.status("error", "pipeline_error", f"Pipeline lỗi: {e}")

    async def _on_subtitle(self, sub: Subtitle) -> None:
        await self.broadcast(P.subtitle_msg(asdict(sub)))

    async def _on_pipeline_status(self, kind: str, msg: str) -> None:
        if msg == "dropped_segment":
            await self.status(kind, "dropped_segment", "Bỏ qua một đoạn do xử lý chưa kịp")
        else:
            await self.status(kind, "pipeline_error", msg)

    async def _stop_engine(self) -> None:
        """Stop pipeline, sources and TTS; leaves a fresh mixer + tracker for the next start."""
        pl, task, models = self.pipeline, self.pipe_task, self.models
        self.pipeline = self.pipe_task = self.models = None
        if pl is not None:  # the mixer was consumed: replace it (and the sources feeding it)
            pl.stop()
            self.mixer.close()
            if task is not None:
                with suppress(Exception, asyncio.CancelledError):
                    await asyncio.wait_for(task, 10)
            async with self.tick_lock:
                for src in list(self.sources.values()):
                    await asyncio.to_thread(self._safe_stop, src)
                self.sources.clear()
                self.mixer = Mixer()
                self.tracker = PermissionTracker(self.tracker.allowed, self.sv.own_pid)
        if models is not None and models.tts is not None:
            with suppress(Exception):
                await asyncio.to_thread(models.tts.close)

    async def shutdown(self) -> None:
        for t in self.tasks:
            t.cancel()
        for t in self.tasks:
            with suppress(BaseException):
                await t
        async with self.restart_lock:
            await self._stop_engine()
            self.mixer.close()
            for src in list(self.sources.values()):
                self._safe_stop(src)
            self.sources.clear()

    # -- client handling ----------------------------------------------------
    async def handle(self, ws: WebSocket, kind: str, d: dict) -> None:
        if kind == "list_apps":
            await self.tick(force_send=True, ws=ws)
        elif kind == "status":
            await self.send_state(ws)
        elif kind == "set_allowed":
            self.tracker.set_allowed(d["exe"], d["allowed"])
            await self.patch_config({"audio": {"allowed_apps": sorted(self.tracker.allowed)}})
            await self.tick(force_send=True)
        elif kind == "set_langs":
            await self.patch_config({"langs": {"src": d["src"], "tgt": d["tgt"]}})
            if self.pipeline:
                self.pipeline.set_src_lang(d["src"])
        elif kind == "set_tts":
            await self.patch_config({"tts": {"enabled": d["enabled"]}})
            if self.pipeline:
                self.pipeline.set_tts_enabled(d["enabled"])
        elif kind == "set_volume":
            await self.patch_config({"tts": {"volume": d["value"]}})
            if self.models and self.models.tts is not None:
                self.models.tts.set_volume(d["value"])
        elif kind == "set_models":
            changed = d["asr_model"] != self.cfg["asr"]["model"]
            await self.patch_config({"asr": {"model": d["asr_model"]}})
            if changed:
                self.tasks = [t for t in self.tasks if not t.done()]
                self.tasks.append(asyncio.create_task(self.load_models()))

    async def send_state(self, ws: WebSocket) -> None:
        await self.send(ws, P.config_msg(P.clean(self.cfg)))
        if self.model_status:
            await self.send(ws, self.model_status)
        elif self.models_ready:
            await self.status("info", "models_ready", "Model đã sẵn sàng", ws)
            for code, msg in self.model_warnings:
                await self.status("warn", code, msg, ws)
        else:
            await self.status("info", "models_loading", "Đang tải model...", ws)
        for exe in sorted(self.self_tree_warned):
            await self.status("warn", "self_tree", _self_tree_text(exe), ws)
        if self.last_apps is not None:
            await self.send(ws, P.apps_msg(self.last_apps))


def create_app(config: dict, services: Services) -> FastAPI:
    hub = Hub(config, services)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        hub.loop = asyncio.get_running_loop()
        hub.tasks = [
            asyncio.create_task(hub.load_models()),
            asyncio.create_task(hub.apps_loop()),
        ]
        try:
            yield
        finally:
            await hub.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.state.hub = hub

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        # browsers always send an Origin (http(s) or "null" for file:// / sandboxed frames);
        # our Qt client sends none -> block any request that carries one
        if "origin" in ws.headers:
            await ws.close(code=1008)
            return
        await ws.accept()
        hub.clients.add(ws)
        try:
            await hub.send_state(ws)
            while True:
                frame = await ws.receive()
                if frame["type"] == "websocket.disconnect":
                    break
                try:
                    if frame.get("text") is None:
                        raise P.BadMessage("Chỉ hỗ trợ tin nhắn văn bản JSON")
                    kind, data = P.parse(frame["text"])
                    await hub.handle(ws, kind, data)
                except P.BadMessage as e:
                    await hub.status("error", "bad_message", str(e), ws)
                except WebSocketDisconnect:
                    raise
                except Exception as e:  # noqa: BLE001 - never drop the connection
                    log.exception("handler failed")
                    await hub.status("error", "internal_error", f"Lỗi xử lý: {e}", ws)
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(ws)

    return app
