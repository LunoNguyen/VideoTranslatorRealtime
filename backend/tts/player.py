"""Threaded TTS player that drops stale sentences."""
from __future__ import annotations

import logging
import threading
from typing import Callable

import numpy as np

from backend.errors import ModelMissingError
from backend.tts.piper_tts import TTSEngine

log = logging.getLogger(__name__)

CHUNK_SECONDS = 0.1


class StreamOutput:
    """One persistent sounddevice OutputStream, (re)created lazily when the sample rate changes."""

    def __init__(self) -> None:
        self._stream = None
        self._rate = 0

    def __call__(self, samples: np.ndarray, sample_rate: int) -> None:
        import sounddevice as sd

        if self._stream is None or self._rate != sample_rate:
            self.close()
            self._stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
            self._rate = sample_rate
        if self._stream.stopped:
            self._stream.start()
        self._stream.write(np.ascontiguousarray(samples, dtype=np.float32).reshape(-1, 1))

    def abort(self) -> None:
        """Drop buffered audio now (the next write restarts the stream)."""
        if self._stream is not None and not self._stream.stopped:
            self._stream.abort()

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()


class TTSPlayer:
    """say() replaces pending text and interrupts current playback."""

    def __init__(
        self,
        engine: TTSEngine,
        output: Callable[[np.ndarray, int], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._engine = engine
        self._output = output or StreamOutput()
        self._on_error = on_error
        self._missing_reported = False  # ModelMissingError is reported once per voice
        self._cond = threading.Condition()
        self._pending: str | None = None
        self._gen = 0
        self._enabled = True
        self._volume = 1.0
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="tts-player", daemon=True)
        self._thread.start()

    def say(self, text: str) -> None:
        if not text.strip():
            return
        with self._cond:
            if not self._enabled or self._closed:
                return
            self._pending = text
            self._gen += 1
            self._cond.notify_all()

    def set_enabled(self, enabled: bool) -> None:
        with self._cond:
            self._enabled = bool(enabled)
            if not enabled:
                self._pending = None
                self._gen += 1
            self._cond.notify_all()

    def set_volume(self, volume: float) -> None:
        with self._cond:
            self._volume = min(1.0, max(0.0, float(volume)))

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._pending = None
            self._gen += 1
            self._cond.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=3)
        close = getattr(self._output, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # noqa: BLE001
                log.exception("TTS output close failed")

    def _stale(self, gen: int) -> bool:
        with self._cond:
            return self._closed or not self._enabled or self._gen != gen

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                text, gen = self._pending, self._gen
                self._pending = None
            try:
                self._play(text, gen)
            except Exception as e:  # keep the thread alive on TTS/audio errors
                if isinstance(e, ModelMissingError):
                    if self._missing_reported:
                        continue
                    self._missing_reported = True
                log.exception("TTS failed")
                if self._on_error is not None:
                    try:
                        self._on_error(e)
                    except Exception:  # noqa: BLE001
                        log.exception("TTS on_error failed")

    def _play(self, text: str, gen: int) -> None:
        samples, rate = self._engine.synthesize(text)
        step = max(1, int(rate * CHUNK_SECONDS))
        for start in range(0, len(samples), step):
            if self._stale(gen):
                abort = getattr(self._output, "abort", None)
                if abort is not None:
                    abort()
                return
            with self._cond:
                volume = self._volume
            self._output((samples[start:start + step] * volume).astype(np.float32), rate)
