"""faster-whisper wrapper with hallucination filtering."""
from __future__ import annotations

import logging
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model

from backend.errors import ModelMissingError  # re-exported
from backend.gpu import register_cuda_dlls

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MIN_SAMPLES = int(0.3 * SAMPLE_RATE)
SUPPORTED_LANGS = {"en", "ja"}
NO_SPEECH_THRESHOLD = 0.6
LOGPROB_THRESHOLD = -1.0

HALLUCINATIONS = {
    "Thank you for watching.",
    "Thanks for watching!",
    "Thank you for watching!",
    "Thanks for watching.",
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございました。",
    "字幕視聴ありがとうございました",
    "字幕視聴ありがとうございました。",
}


class CudaOutOfMemoryError(RuntimeError):
    """CUDA ran out of memory while loading the model on an explicit cuda device."""


_TRAIL = " .!?。！？"
_NORMALIZED = {h.casefold().strip(_TRAIL) for h in HALLUCINATIONS}


@dataclass
class Transcript:
    text: str
    lang: str
    no_speech_prob: float


def _only_punct(text: str) -> bool:
    return all(unicodedata.category(c)[0] in "PSZC" for c in text)


def _bad_text(text: str) -> bool:
    """Empty, punctuation-only or a known hallucination phrase."""
    text = text.strip()
    return not text or _only_punct(text) or text.casefold().strip(_TRAIL) in _NORMALIZED


def drop_segment(text: str, no_speech_prob: float, avg_logprob: float) -> bool:
    """Drop a segment only if it is silent *and* low-confidence, or its text is junk."""
    if no_speech_prob > NO_SPEECH_THRESHOLD and avg_logprob < LOGPROB_THRESHOLD:
        return True
    return _bad_text(text)


def is_hallucination(t: Transcript) -> bool:
    """True for empty/punctuation-only text or a known phrase (silence is judged per segment)."""
    return _bad_text(t.text)


class WhisperASR:
    def __init__(self, model: str, device: str, compute_type: str,
                 model_root: Path = Path("models/asr")) -> None:
        self.model_name, self.device, self.compute_type = model, device, compute_type
        self.model_root = Path(model_root)
        self.device_used: str | None = None
        self._model: WhisperModel | None = None
        self.fallback_reason: str | None = None  # "cuda_oom" | "cuda_unavailable" | None
        self._lock = threading.Lock()

    def _resolve(self) -> str:
        """Return a local model path, downloading into model_root if needed (once)."""
        local = self.model_root / self.model_name
        if local.is_dir():
            return str(local)
        try:
            return download_model(self.model_name, cache_dir=str(self.model_root))
        except Exception as e:
            raise ModelMissingError(f"Whisper model '{self.model_name}' unavailable: {e}") from e

    @staticmethod
    def _build(path: str, device: str, compute_type: str) -> WhisperModel:
        register_cuda_dlls()
        model = WhisperModel(path, device=device, compute_type=compute_type)
        # Construction succeeds even when cuBLAS is missing; a warm-up run surfaces it.
        segs, _ = model.transcribe(np.zeros(SAMPLE_RATE, np.float32), beam_size=1, vad_filter=False)
        list(segs)
        return model

    def load(self) -> None:
        """Load the model now (idempotent, thread-safe).

        Raises ModelMissingError if the model cannot be found/downloaded. In auto mode any
        CUDA failure (including OOM) falls back to CPU int8 and sets fallback_reason; with
        an explicit device="cuda", OOM raises CudaOutOfMemoryError and other errors propagate.
        """
        with self._lock:
            if self._model is not None:
                return
            path = self._resolve()
            if self.device == "auto":
                try:
                    self._model, self.device_used = self._build(path, "cuda", self.compute_type), "cuda"
                except Exception as e:
                    oom = "out of memory" in str(e).lower()
                    self.fallback_reason = "cuda_oom" if oom else "cuda_unavailable"
                    log.warning("CUDA load failed (%s: %s); falling back to CPU, compute_type %s -> int8",
                                type(e).__name__, e, self.compute_type)
                    self._model, self.device_used = self._build(path, "cpu", "int8"), "cpu"
            else:
                try:
                    self._model = self._build(path, self.device, self.compute_type)
                except Exception as e:
                    if self.device == "cuda" and "out of memory" in str(e).lower():
                        raise CudaOutOfMemoryError(str(e)) from e
                    raise
                self.device_used = self.device
            log.info("Whisper %s loaded on %s", self.model_name, self.device_used)

    def transcribe(self, audio: np.ndarray, src_lang: str) -> Transcript | None:
        if len(audio) < MIN_SAMPLES:
            return None
        self.load()
        auto = src_lang == "auto"
        segments, info = self._model.transcribe(
            audio, language=None if auto else src_lang, vad_filter=False,
            beam_size=1, condition_on_previous_text=False)
        segs = list(segments)
        lang = info.language if auto else src_lang
        if lang not in SUPPORTED_LANGS:
            return None
        kept = [s for s in segs if not drop_segment(s.text, s.no_speech_prob, s.avg_logprob)]
        if not kept:
            return None
        text = ("" if lang == "ja" else " ").join(s.text.strip() for s in kept).strip()
        t = Transcript(text, lang, float(max(s.no_speech_prob for s in kept)))
        return None if is_hallucination(t) else t
