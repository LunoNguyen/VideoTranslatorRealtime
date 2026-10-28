"""Online speaker clusterer: stable ids (0..max-1) for voice vectors. RAM only."""
from __future__ import annotations

import numpy as np


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class SpeakerClusterer:
    def __init__(self, threshold: float = 0.6, max_speakers: int = 6,
                 min_new_s: float = 1.0, reset_after_s: float = 300.0,
                 alpha: float = 0.2):
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.min_new_s = min_new_s
        self.reset_after_s = reset_after_s
        self.alpha = alpha
        self._centroids: list[np.ndarray] = []
        self._last_seen: float | None = None

    @property
    def count(self) -> int:
        return len(self._centroids)

    def reset(self) -> None:
        self._centroids = []
        self._last_seen = None

    def _touch(self, now: float) -> None:
        if self._last_seen is not None and now - self._last_seen > self.reset_after_s:
            self.reset()
        self._last_seen = now

    def _nearest(self, v: np.ndarray) -> tuple[int, float] | None:
        if not self._centroids:
            return None
        sims = np.stack(self._centroids) @ v
        i = int(np.argmax(sims))
        return i, float(sims[i])

    def assign(self, vec: np.ndarray, dur_s: float, now: float) -> int | None:
        self._touch(now)
        v = _norm(vec)
        best = self._nearest(v)
        if best is not None and best[1] >= self.threshold:
            i = best[0]
            self._centroids[i] = _norm((1 - self.alpha) * self._centroids[i] + self.alpha * v)
            return i
        if dur_s >= self.min_new_s and self.count < self.max_speakers:
            self._centroids.append(v)
            return self.count - 1
        return best[0] if best is not None else None

    def match(self, vec: np.ndarray, now: float) -> int | None:
        self._touch(now)
        best = self._nearest(_norm(vec))
        if best is not None and best[1] >= self.threshold:
            return best[0]
        return None
