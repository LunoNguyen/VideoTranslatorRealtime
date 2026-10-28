"""Piper TTS engine (CPU)."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from backend.errors import ModelMissingError


class TTSEngine(Protocol):
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (mono float32 samples in [-1, 1], sample rate)."""
        ...


class PiperTTS:
    """Piper voice wrapper; loads lazily from an .onnx file (+ .onnx.json)."""

    def __init__(self, voice_path: Path) -> None:
        self.voice_path = Path(voice_path)
        self._voice = None

    def _load(self):
        if self._voice is None:
            config = Path(str(self.voice_path) + ".json")
            if not self.voice_path.is_file() or not config.is_file():
                raise ModelMissingError(
                    f"Thiếu giọng đọc Piper: {self.voice_path} (và {config.name}). "
                    "Hãy tải model TTS trước khi bật đọc tiếng Việt."
                )
            from piper import PiperVoice

            self._voice = PiperVoice.load(self.voice_path, config_path=config, use_cuda=False)
        return self._voice

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        voice = self._load()
        chunks = list(voice.synthesize(text))
        if not chunks:
            return np.zeros(0, dtype=np.float32), int(voice.config.sample_rate)
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        return audio, int(chunks[0].sample_rate)
