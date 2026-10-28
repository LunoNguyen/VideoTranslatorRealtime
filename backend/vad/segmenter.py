"""Silence-based sentence segmenter on 16 kHz mono float32 frames."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

SR = 16000
WINDOW = 512  # Silero window at 16 kHz (32 ms)
CONTEXT = 64
PREROLL = 1600  # ~100 ms kept before first speech window
TRAIL = 1024  # 64 ms of trailing silence kept (<= 100 ms)


@dataclass
class Segment:
    audio: np.ndarray
    t_start: float
    t_end: float
    seg_id: int = 0


@dataclass
class Partial:
    """Snapshot of the sentence still being spoken (provisional)."""

    seg_id: int
    audio: np.ndarray
    t_start: float
    t_end: float


PARTIAL_MIN_S = 1.0  # speech needed in a sentence before its first partial
PARTIAL_MAX_S = 8.0  # cap on partial audio length


class SileroSpeechProb:
    """Stateful Silero VAD (ONNX bundled with faster-whisper); window of 512 samples -> prob."""

    def __init__(self) -> None:
        import onnxruntime
        from faster_whisper.vad import get_assets_path
        import os

        path = os.path.join(get_assets_path(), "silero_vad_v6.onnx")
        if not os.path.exists(path):
            cands = [f for f in os.listdir(get_assets_path()) if f.endswith(".onnx")]
            if not cands:
                raise FileNotFoundError("Silero VAD onnx not found in faster_whisper assets")
            path = os.path.join(get_assets_path(), sorted(cands)[-1])
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._sess = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._ctx = np.zeros((1, CONTEXT), dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        x = np.concatenate([self._ctx, window.astype(np.float32)[None, :]], axis=1)
        out, self._h, self._c = self._sess.run(
            None, {"input": x, "h": self._h, "c": self._c}
        )
        self._ctx = x[:, -CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])


class Segmenter:
    """Feeds 20 ms frames; emits Segments when silence > silence_ms or at max_s."""

    def __init__(
        self,
        speech_prob: Callable[[np.ndarray], float] | None = None,
        silence_ms: int = 500,
        max_s: float = 8.0,
        threshold: float = 0.5,
        partial_every_s: float | None = None,
    ) -> None:
        self._prob = speech_prob
        self.silence_ms = silence_ms
        self.max_samples = int(round(max_s * SR))
        self.threshold = threshold
        self._audio = np.zeros(0, dtype=np.float32)
        self._off = 0  # absolute sample index of _audio[0]
        self._done = 0  # absolute index up to which windows were evaluated
        self._in_seg = False
        self._seg_start = 0
        self._last_speech_end = 0
        self._silence_ms = 0.0
        self.partial_every_s = partial_every_s
        self._seg_id = 0
        self._last_partial_end: int | None = None

    def set_partial_every(self, s: float | None) -> None:
        self.partial_every_s = s

    def _speech_prob(self, window: np.ndarray) -> float:
        if self._prob is None:
            self._prob = SileroSpeechProb()
        return self._prob(window)

    def _slice(self, a: int, b: int) -> np.ndarray:
        return self._audio[a - self._off : b - self._off].copy()

    def _trim(self, keep_from: int) -> None:
        keep_from = max(self._off, keep_from)
        self._audio = self._audio[keep_from - self._off :]
        self._off = keep_from

    def _emit(self, a: int, b: int) -> Segment:
        return Segment(self._slice(a, b), a / SR, b / SR, self._seg_id)

    def _maybe_partial(self, e: int, out: list) -> None:
        every = self.partial_every_s
        if every is None or not self._in_seg or self._silence_ms > 0:
            return  # partials only while speaking, never during trailing silence
        if e - self._seg_start < int(PARTIAL_MIN_S * SR):
            return
        if self._last_partial_end is not None and e - self._last_partial_end < every * SR:
            return
        a = max(self._seg_start, e - int(PARTIAL_MAX_S * SR))
        out.append(Partial(self._seg_id, self._slice(a, e), a / SR, e / SR))
        self._last_partial_end = e

    def push(self, frame: np.ndarray) -> list[Segment | Partial]:
        self._audio = np.concatenate([self._audio, frame.astype(np.float32).ravel()])
        out: list[Segment | Partial] = []
        total = self._off + len(self._audio)
        while total - self._done >= WINDOW:
            s = self._done
            e = s + WINDOW
            speech = self._speech_prob(self._slice(s, e)) >= self.threshold
            self._done = e
            self._step(s, e, speech, out)
        return out

    def _step(self, s: int, e: int, speech: bool, out: list) -> None:
        if not self._in_seg:
            if speech:
                self._in_seg = True
                self._seg_id += 1
                self._last_partial_end = None
                self._seg_start = max(self._off, s - PREROLL)
                self._last_speech_end = e
                self._silence_ms = 0.0
            else:
                self._trim(e - PREROLL)
                return
        elif speech:
            self._last_speech_end = e
            self._silence_ms = 0.0
        else:
            self._silence_ms += WINDOW * 1000 / SR
            if self._silence_ms > self.silence_ms:
                end = min(self._last_speech_end + TRAIL, e)
                out.append(self._emit(self._seg_start, end))
                self._in_seg = False
                self._trim(e - PREROLL)
                return
        if e - self._seg_start >= self.max_samples:
            cut = self._seg_start + self.max_samples
            out.append(self._emit(self._seg_start, cut))
            self._seg_start = cut
            self._trim(cut)
            if self._last_speech_end <= cut:
                self._in_seg = False
            else:
                self._seg_id += 1
                self._last_partial_end = None
            return
        self._maybe_partial(e, out)

    def flush(self) -> list[Segment | Partial]:
        out: list[Segment | Partial] = []
        if self._in_seg:
            total = self._off + len(self._audio)
            end = min(self._last_speech_end + TRAIL, total)
            if end > self._seg_start:
                out.append(self._emit(self._seg_start, end))
        self._in_seg = False
        self._trim(self._off + len(self._audio))
        self._done = self._off
        return out
