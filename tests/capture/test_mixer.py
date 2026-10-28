import threading

import numpy as np

from backend.capture.mixer import Mixer


class FakeSource:
    """Serves a preloaded (n, channels) float32 buffer in fixed-size chunks."""

    def __init__(self, pid: int, data: np.ndarray, sample_rate: int, chunk_ms: int = 10):
        self.pid = pid
        self.sample_rate = sample_rate
        self.data = data.reshape(len(data), -1).astype(np.float32)
        self.channels = self.data.shape[1]
        self.chunk = sample_rate * chunk_ms // 1000
        self.pos = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @property
    def exhausted(self) -> bool:
        return self.pos >= len(self.data)

    def read(self, timeout: float) -> np.ndarray | None:
        if self.exhausted:
            return None
        out = self.data[self.pos : self.pos + self.chunk]
        self.pos += self.chunk
        return out


def collect_until_exhausted(mixer: Mixer, sources: list[FakeSource]) -> list[np.ndarray]:
    """Iterate frames() in a thread; close the mixer once all sources ran dry."""
    frames: list[np.ndarray] = []

    def run() -> None:
        for f in mixer.frames():
            frames.append(f)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    while not all(s.exhausted for s in sources):
        t.join(0.01)
    t.join(0.1)
    mixer.close()
    t.join(5)
    assert not t.is_alive()
    return frames


def test_resample_48k_stereo_to_16k_mono():
    sr = 48000
    t = np.arange(sr) / sr
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    src = FakeSource(1, np.stack([tone, tone], axis=1), sr)
    mixer = Mixer()
    mixer.add(src)
    frames = collect_until_exhausted(mixer, [src])
    assert all(f.shape == (320,) and f.dtype == np.float32 for f in frames)
    out = np.concatenate(frames)
    assert abs(len(out) - 16000) <= 320
    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = np.argmax(spectrum) * 16000 / len(out)
    assert abs(peak_hz - 440) <= 5


def test_two_sources_summed_and_clipped():
    sr = 16000
    a = FakeSource(1, np.full(sr, 0.3), sr)
    b = FakeSource(2, np.full(sr, 0.2), sr)
    mixer = Mixer()
    mixer.add(a)
    mixer.add(b)
    out = np.concatenate(collect_until_exhausted(mixer, [a, b]))
    mid = out[2000:14000]
    assert np.allclose(mid, 0.5, atol=1e-3)

    c = FakeSource(3, np.full(sr, 0.7), sr)
    d = FakeSource(4, np.full(sr, 0.7, dtype=np.float32), sr)
    mixer = Mixer()
    mixer.add(c)
    mixer.add(d)
    out = np.concatenate(collect_until_exhausted(mixer, [c, d]))
    assert out.max() <= 1.0 and out.min() >= -1.0
    assert np.allclose(out[2000:14000], 1.0)


def test_remove_source_stops_its_audio():
    sr = 16000
    a = FakeSource(1, np.full(sr * 2, 0.3), sr)
    b = FakeSource(2, np.full(sr * 2, 0.2), sr)
    mixer = Mixer()
    mixer.add(a)
    mixer.add(b)
    before: list[np.ndarray] = []
    after: list[np.ndarray] = []
    for frame in mixer.frames():
        if len(before) < 20:
            before.append(frame)
            if len(before) == 20:
                assert mixer.remove(2) is b
        else:
            after.append(frame)
            if len(after) == 20:
                break
    mixer.close()
    assert np.allclose(before[10], 0.5, atol=1e-3)
    assert np.allclose(np.concatenate(after[5:]), 0.3, atol=1e-3)


def test_failed_pids_reports_sources_with_error():
    sr = 16000
    ok = FakeSource(1, np.zeros(sr), sr)  # no `error` attribute at all
    bad = FakeSource(2, np.zeros(sr), sr)
    bad.error = None
    mixer = Mixer()
    mixer.add(ok)
    mixer.add(bad)
    assert mixer.failed_pids() == []
    bad.error = RuntimeError("device invalidated")
    assert mixer.failed_pids() == [2]
