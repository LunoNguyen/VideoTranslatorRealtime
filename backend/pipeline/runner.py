"""Async pipeline: capture+VAD thread -> bounded segment queue -> ASR/MT (threads) -> subtitle/TTS."""
from __future__ import annotations

import asyncio
import collections
import logging
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

import numpy as np

from backend.asr.whisper import is_hallucination

log = logging.getLogger(__name__)


@dataclass
class Subtitle:
    id: int
    src_lang: str
    src_text: str
    vi_text: str
    t_start: float
    t_end: float


class Pipeline:
    def __init__(
        self,
        frames: Iterable[np.ndarray],
        segmenter: Any,
        asr: Any,
        mt: Any,
        tts: Any,
        on_subtitle: Callable[[Subtitle], Awaitable[None]],
        on_status: Callable[[str, str], Awaitable[None]],
        seg_queue_max: int = 4,
    ) -> None:
        self._frames = frames
        self._seg = segmenter
        self._asr = asr
        self._mt = mt
        self._tts = tts
        self._on_subtitle = on_subtitle
        self._on_status = on_status
        self._max = max(1, seg_queue_max)
        self._src_lang = "auto"
        self._tts_enabled = True
        self._stop = threading.Event()
        self._queue: collections.deque = collections.deque()
        self._wake: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._capture_done = False
        self._next_id = 1

    def set_src_lang(self, lang: str) -> None:
        self._src_lang = lang

    def set_tts_enabled(self, enabled: bool) -> None:
        """Gate TTS calls (and forward to the player if it supports it)."""
        self._tts_enabled = bool(enabled)
        setter = getattr(self._tts, "set_enabled", None)
        if setter is not None:
            setter(self._tts_enabled)

    def stop(self) -> None:
        self._stop.set()
        if self._wake is not None and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._wake.set)
            except RuntimeError:  # loop already closed
                pass

    async def _status(self, kind: str, msg: str) -> None:
        """Report status; a failing callback must never break the pipeline."""
        try:
            await self._on_status(kind, msg)
        except Exception:  # noqa: BLE001
            log.exception("on_status failed")

    # -- capture thread -------------------------------------------------
    async def _enqueue(self, seg: Any) -> None:
        if len(self._queue) >= self._max:
            self._queue.popleft()
            await self._status("warn", "dropped_segment")
        self._queue.append(seg)
        self._wake.set()

    def _submit(self, seg: Any) -> None:
        asyncio.run_coroutine_threadsafe(self._enqueue(seg), self._loop).result()

    def _capture(self) -> None:
        try:
            for frame in self._frames:
                if self._stop.is_set():
                    return
                for seg in self._seg.push(frame):
                    self._submit(seg)
            if not self._stop.is_set():
                for seg in self._seg.flush():
                    self._submit(seg)
        except Exception as e:  # noqa: BLE001
            log.exception("capture failed")
            try:
                asyncio.run_coroutine_threadsafe(
                    self._status("error", f"capture: {e}"), self._loop
                ).result()
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._capture_done = True
            try:
                self._loop.call_soon_threadsafe(self._wake.set)
            except RuntimeError:
                pass

    # -- consumer ---------------------------------------------------------
    async def _process(self, seg: Any) -> None:
        try:
            tr = await asyncio.to_thread(self._asr.transcribe, seg.audio, self._src_lang)
            if tr is None or is_hallucination(tr):
                return
            lang = tr.lang if self._src_lang == "auto" else self._src_lang
            vi = await asyncio.to_thread(self._mt.translate, tr.text, lang)
        except Exception as e:  # noqa: BLE001
            log.exception("stage failed")
            await self._status("error", str(e))
            return
        sub = Subtitle(self._next_id, lang, tr.text, vi, seg.t_start, seg.t_end)
        self._next_id += 1
        try:
            await self._on_subtitle(sub)
        except Exception as e:  # noqa: BLE001
            log.exception("on_subtitle failed")
            await self._status("error", f"subtitle: {e}")
        if self._tts is not None and self._tts_enabled:
            try:
                self._tts.say(vi)
            except Exception as e:  # noqa: BLE001
                log.exception("tts failed")
                await self._status("error", f"tts: {e}")

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        self._queue.clear()
        self._capture_done = False
        thread = threading.Thread(target=self._capture, name="capture-vad", daemon=True)
        thread.start()
        try:
            while not self._stop.is_set():
                if self._queue:
                    await self._process(self._queue.popleft())
                    continue
                if self._capture_done:
                    break
                self._wake.clear()
                if self._queue or self._capture_done or self._stop.is_set():
                    continue
                await self._wake.wait()
        finally:
            self._stop.set()
            await asyncio.to_thread(thread.join, 5.0)
