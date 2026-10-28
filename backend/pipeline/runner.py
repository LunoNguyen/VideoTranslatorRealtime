"""Async pipeline: capture+VAD thread -> segment queue + latest-partial slot -> ASR/MT (threads) -> subtitle/TTS.

Two lanes:
- Segments (finished sentences) go into a bounded queue (drop oldest) and always win.
- Partials (sentence still being spoken) go into a one-slot "latest partial" that a newer
  partial overwrites. They are shown as provisional subtitles (final=False), translated with
  ``commit=False`` and never spoken.

Speaker vectors are kept in RAM only (inside the clusterer) and never logged.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

import numpy as np

from backend.asr.whisper import is_hallucination
from backend.speaker.cluster import SpeakerClusterer
from backend.vad.segmenter import Partial

log = logging.getLogger(__name__)

SR = 16000
PARTIAL_MIN_EVERY_S = 0.7  # base / floor of the adaptive partial interval
PARTIAL_MAX_EVERY_S = 2.0  # cap of the adaptive partial interval
LATE_LIMIT = 3  # consecutive late partials before streaming is turned off
PROVISIONAL_SPK_S = 1.5  # audio used for the provisional speaker of a sentence
EMBED_MAX_S = 4.0  # longest audio handed to the speaker embedder (middle part is used)


@dataclass
class Subtitle:
    id: int
    src_lang: str
    src_text: str
    vi_text: str
    t_start: float
    t_end: float
    final: bool = True
    speaker: int | None = None
    rev: int = 0


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
        speaker: Any | None = None,
        speaker_cfg: dict | None = None,
        clock: Callable[[], float] = time.monotonic,
        partial_every_s: float = PARTIAL_MIN_EVERY_S,
    ) -> None:
        self._frames = frames
        self._seg = segmenter
        self._asr = asr
        self._mt = mt
        self._tts = tts
        self._on_subtitle = on_subtitle
        self._on_status = on_status
        self._max = max(1, seg_queue_max)
        self._clock = clock
        self._src_lang = "auto"
        self._tts_enabled = True
        self._stop = threading.Event()
        self._queue: collections.deque = collections.deque()
        self._partial: Partial | None = None  # one-slot latest partial
        self._wake: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._capture_done = False
        self._next_id = 1
        # streaming lane
        self._streaming = True
        self._base = float(partial_every_s)  # floor of the adaptive interval
        self._interval = self._base
        self._late = 0
        self._final_max = 0  # highest seg_id finalized so far (seg_ids are monotonic)
        self._rev: dict[int, int] = {}  # sends so far per seg_id
        self._last_text: dict[int, str] = {}  # last partial src_text per seg_id
        self._prov_spk: dict[int, int | None] = {}  # provisional speaker per seg_id
        # speaker tagging
        cfg = speaker_cfg or {}
        self._speaker = speaker
        self._speaker_enabled = speaker is not None
        self._clusterer = SpeakerClusterer(
            threshold=cfg.get("threshold", 0.6), max_speakers=cfg.get("max_speakers", 6)
        )

    # -- controls -----------------------------------------------------------
    def set_src_lang(self, lang: str) -> None:
        self._src_lang = lang

    def set_tts_enabled(self, enabled: bool) -> None:
        """Gate TTS calls (and forward to the player if it supports it)."""
        self._tts_enabled = bool(enabled)
        setter = getattr(self._tts, "set_enabled", None)
        if setter is not None:
            setter(self._tts_enabled)

    def _push_interval(self, s: float | None) -> None:
        setter = getattr(self._seg, "set_partial_every", None)
        if setter is not None:
            setter(s)

    def set_streaming(self, enabled: bool) -> None:
        """Turn the provisional (partial) lane on/off; off also drops a pending partial."""
        self._streaming = bool(enabled)
        if self._streaming:
            self._late = 0
            self._interval = self._base
            self._push_interval(self._interval)
        else:
            self._partial = None
            self._push_interval(None)

    def set_speaker_enabled(self, enabled: bool) -> None:
        self._speaker_enabled = bool(enabled) and self._speaker is not None

    def reset_speakers(self) -> None:
        """Forget all voices. Call from the event loop (the clusterer is not thread-safe)."""
        self._clusterer.reset()
        self._prov_spk.clear()

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
    async def _enqueue(self, items: list) -> None:
        """Route one push()'s output atomically: Segments -> queue, Partials -> slot.

        Routing has no await, so the consumer cannot interleave; drop statuses go out after.
        """
        dropped = 0
        for item in items:
            if isinstance(item, Partial):
                if self._streaming:
                    self._partial = item  # overwrite an unprocessed older partial
                continue
            if len(self._queue) >= self._max:
                self._queue.popleft()
                dropped += 1
            self._queue.append(item)
        self._wake.set()
        for _ in range(dropped):
            await self._status("warn", "dropped_segment")

    def _submit(self, items: list) -> None:
        if items:
            asyncio.run_coroutine_threadsafe(self._enqueue(items), self._loop).result()

    def _capture(self) -> None:
        try:
            for frame in self._frames:
                if self._stop.is_set():
                    return
                self._submit(list(self._seg.push(frame)))
            if not self._stop.is_set():
                self._submit(list(self._seg.flush()))
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

    # -- speaker --------------------------------------------------------
    async def _embed(self, audio: np.ndarray) -> np.ndarray | None:
        """Embed off-loop; on failure disable speaker tagging for the session."""
        if not self._speaker_enabled:
            return None
        n = int(EMBED_MAX_S * SR)
        if len(audio) > n:  # keep the middle: edges are most likely silence/other voices
            a = (len(audio) - n) // 2
            audio = audio[a : a + n]
        try:
            return await asyncio.to_thread(self._speaker.embed, audio)
        except Exception:  # noqa: BLE001
            log.exception("speaker embed failed")
            self._speaker_enabled = False
            await self._status("warn", "speaker_unavailable")
            return None

    async def _provisional_speaker(self, p: Partial) -> int | None:
        """Computed once per seg_id from the first 1.5 s; uses match (never creates/updates clusters)."""
        if not self._speaker_enabled:
            return None
        if p.seg_id not in self._prov_spk:
            if self._segment_waiting(p.seg_id):  # late already: don't spend an embed
                return None
            vec = await self._embed(p.audio[: int(PROVISIONAL_SPK_S * SR)])
            if vec is None:
                return None
            self._prov_spk[p.seg_id] = self._clusterer.match(vec, self._clock())
        return self._prov_spk[p.seg_id]

    # -- consumer ---------------------------------------------------------
    def _next_rev(self, sid: int) -> int:
        rev = self._rev.get(sid, 0)
        self._rev[sid] = rev + 1
        return rev

    def _finalized(self, sid: int) -> None:
        if sid > self._final_max:
            self._final_max = sid
        for d in (self._rev, self._last_text, self._prov_spk):
            for k in [k for k in d if k <= self._final_max]:
                del d[k]

    def _segment_waiting(self, sid: int) -> bool:
        return any(getattr(s, "seg_id", 0) == sid for s in self._queue)

    async def _emit(self, sub: Subtitle) -> None:
        try:
            await self._on_subtitle(sub)
        except Exception as e:  # noqa: BLE001
            log.exception("on_subtitle failed")
            if sub.final:  # partial errors are only logged
                await self._status("error", f"subtitle: {e}")

    async def _process(self, seg: Any) -> None:
        sid = getattr(seg, "seg_id", 0)
        rev = self._rev.get(sid, 0) if sid > 0 else 0
        if sid > 0:
            self._finalized(sid)  # later partials of this sentence are dropped
        embed_task = (
            asyncio.create_task(self._embed(seg.audio)) if self._speaker_enabled else None
        )
        try:
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
            speaker = None
            if embed_task is not None:
                vec = await embed_task  # _embed handles its own failures
                embed_task = None
                if vec is not None and self._speaker_enabled:
                    speaker = self._clusterer.assign(vec, seg.t_end - seg.t_start, self._clock())
        finally:
            if embed_task is not None:  # early return / cancellation: don't leak the task
                embed_task.cancel()
                await asyncio.gather(embed_task, return_exceptions=True)
        if sid > 0:
            sub_id = sid
        else:
            sub_id = self._next_id
            self._next_id += 1
        await self._emit(
            Subtitle(sub_id, lang, tr.text, vi, seg.t_start, seg.t_end,
                     final=True, speaker=speaker, rev=rev)
        )
        if self._tts is not None and self._tts_enabled:
            try:
                self._tts.say(vi)
            except Exception as e:  # noqa: BLE001
                log.exception("tts failed")
                await self._status("error", f"tts: {e}")

    def _adapt(self, took: float) -> None:
        if took > self._interval:
            self._interval = min(max(PARTIAL_MAX_EVERY_S, self._base), took * 1.2)
        else:
            self._interval = max(self._base, self._interval * 0.9)
        if self._streaming:
            self._push_interval(self._interval)

    async def _on_late(self) -> None:
        self._late += 1
        if self._late >= LATE_LIMIT and self._streaming:
            self.set_streaming(False)
            await self._status("warn", "streaming_slow")

    async def _process_partial(self, p: Partial) -> None:
        t0 = self._clock()
        ok, sub = await self._partial_stages(p)
        self._adapt(self._clock() - t0)
        if not ok or not self._streaming:  # error/hallucination, or streaming turned off in flight
            return
        if self._segment_waiting(p.seg_id):  # its final overtook it: late, not sent
            await self._on_late()
            return
        if sub is not None:  # None = unchanged text, nothing to re-send
            self._late = 0  # only an on-time *sent* partial resets the late count
            self._last_text[p.seg_id] = sub.src_text
            sub.rev = self._next_rev(p.seg_id)
            await self._emit(sub)

    async def _partial_stages(self, p: Partial) -> tuple[bool, Subtitle | None]:
        """ASR -> MT(commit=False) -> provisional speaker. Returns (ok, subtitle-or-None)."""
        sid = p.seg_id
        try:
            tr = await asyncio.to_thread(self._asr.transcribe, p.audio, self._src_lang)
            if tr is None or is_hallucination(tr):
                return False, None
            if self._segment_waiting(sid) or self._last_text.get(sid) == tr.text:
                return True, None  # late already, or unchanged: skip MT
            lang = tr.lang if self._src_lang == "auto" else self._src_lang
            vi = await asyncio.to_thread(self._mt.translate, tr.text, lang, commit=False)
        except Exception:  # noqa: BLE001
            log.exception("partial failed")
            return False, None
        speaker = await self._provisional_speaker(p)
        return True, Subtitle(sid, lang, tr.text, vi, p.t_start, p.t_end, final=False, speaker=speaker)

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        self._queue.clear()
        self._partial = None
        self._capture_done = False
        thread = threading.Thread(target=self._capture, name="capture-vad", daemon=True)
        thread.start()
        try:
            while not self._stop.is_set():
                if self._queue:
                    await self._process(self._queue.popleft())
                    continue
                if self._partial is not None:
                    p, self._partial = self._partial, None
                    if p.seg_id <= self._final_max or not self._streaming:
                        continue  # sentence already finalized: drop silently
                    await self._process_partial(p)
                    continue
                if self._capture_done:
                    break
                self._wake.clear()
                if self._queue or self._partial is not None or self._capture_done or self._stop.is_set():
                    continue
                await self._wake.wait()
        finally:
            self._stop.set()
            await asyncio.to_thread(thread.join, 5.0)
