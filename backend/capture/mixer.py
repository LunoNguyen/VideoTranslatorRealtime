"""Mix several AudioSources into 16 kHz mono float32 frames of 20 ms.

Each source is downmixed to mono and resampled with its own streaming
``soxr.ResampleStream`` (filter state kept per source). Frames are summed
across sources and clipped to [-1, 1].

Pacing: ``frames()`` yields only when audio has arrived; process-loopback
sources deliver (possibly silent) samples continuously, so output follows
real time. With no source attached it idles (~20 ms sleeps). ``close()``
flushes resamplers, yields the remaining full frames and ends the iterator.

Alignment: a frame is emitted when every *live* source (data within
``LIVE_S``) has a full frame buffered, or when any source has buffered
``MAX_LAG_FRAMES`` (a stalled peer then contributes what it has, zero-padded).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import soxr

from backend.capture.source import AudioSource

READ_TIMEOUT_S = 0.005
IDLE_SLEEP_S = 0.02
LIVE_S = 0.1
MAX_LAG_FRAMES = 10


@dataclass
class _Track:
    source: AudioSource
    resampler: soxr.ResampleStream
    pending: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    last_data: float = 0.0

    def push(self, samples: np.ndarray, last: bool = False) -> None:
        out = self.resampler.resample_chunk(samples, last=last)
        if len(out):
            self.pending = np.concatenate([self.pending, out.astype(np.float32, copy=False)])

    def take(self, n: int) -> np.ndarray:
        chunk, self.pending = self.pending[:n], self.pending[n:]
        if len(chunk) < n:
            chunk = np.pad(chunk, (0, n - len(chunk)))
        return chunk


def _to_mono(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return np.ascontiguousarray(data, dtype=np.float32)


class Mixer:
    """Thread-safe: add/remove/close may be called while frames() iterates."""

    def __init__(self, out_rate: int = 16000, frame_ms: int = 20) -> None:
        self.out_rate = out_rate
        self.frame_len = out_rate * frame_ms // 1000
        self._tracks: dict[int, _Track] = {}
        self._lock = threading.Lock()
        self._closed = threading.Event()

    def add(self, source: AudioSource) -> None:
        """Attach a source (already started by the caller); replaces same pid."""
        rs = soxr.ResampleStream(source.sample_rate, self.out_rate, 1, dtype="float32")
        with self._lock:
            self._tracks[source.pid] = _Track(source, rs)

    def remove(self, pid: int) -> AudioSource | None:
        """Detach and drop buffered audio; returns the source (caller stops it)."""
        with self._lock:
            track = self._tracks.pop(pid, None)
        return track.source if track else None

    def pids(self) -> list[int]:
        with self._lock:
            return list(self._tracks)

    def failed_pids(self) -> list[int]:
        """Pids whose source reports a non-None ``error`` (capture died)."""
        with self._lock:
            return [
                pid for pid, t in self._tracks.items()
                if getattr(t.source, "error", None) is not None
            ]

    def close(self) -> None:
        """End frames() after flushing what is buffered."""
        self._closed.set()

    def frames(self) -> Iterator[np.ndarray]:
        n = self.frame_len
        while not self._closed.is_set():
            with self._lock:
                tracks = list(self._tracks.items())
            if not tracks:
                time.sleep(IDLE_SLEEP_S)
                continue
            got = False
            for _, track in tracks:
                data = track.source.read(READ_TIMEOUT_S)
                if data is not None and len(data):
                    track.push(_to_mono(data))
                    track.last_data = time.monotonic()
                    got = True
            while (frame := self._next_frame(force=False)) is not None:
                yield frame
            if not got:
                time.sleep(READ_TIMEOUT_S)
        with self._lock:
            for track in self._tracks.values():
                track.push(np.zeros(0, np.float32), last=True)
        while (frame := self._next_frame(force=True)) is not None:
            yield frame

    def _next_frame(self, force: bool) -> np.ndarray | None:
        n = self.frame_len
        with self._lock:
            tracks = list(self._tracks.values())
            if not tracks:
                return None
            sizes = [len(t.pending) for t in tracks]
            if max(sizes) < (1 if force else n):
                return None
            now = time.monotonic()
            live_ready = all(len(t.pending) >= n for t in tracks if now - t.last_data < LIVE_S)
            if not (force or live_ready or max(sizes) >= n * MAX_LAG_FRAMES):
                return None
            mix = np.zeros(n, np.float32)
            for t in tracks:
                mix += t.take(n)
        return np.clip(mix, -1.0, 1.0)
