import threading
from pathlib import Path

import numpy as np
import pytest

from backend.errors import ModelMissingError
from backend.tts.piper_tts import PiperTTS
from backend.tts.player import TTSPlayer

RATE = 1000  # 100 ms chunk = 100 samples


class FakeEngine:
    """Sentence 'A' -> 5 chunks of 0.1, 'B' -> 0.2, 'C' -> 0.3."""

    LEVELS = {"A": 0.1, "B": 0.2, "C": 0.3}

    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text):
        self.calls.append(text)
        return np.full(500, self.LEVELS.get(text, 0.5), dtype=np.float32), RATE


def test_new_sentence_drops_pending():
    got: list[float] = []
    first_chunk = threading.Event()
    gate = threading.Event()
    last_c = threading.Event()
    n_c = 0

    def output(samples, rate):
        nonlocal n_c
        got.append(round(float(samples[0]), 3))
        if len(got) == 1:
            first_chunk.set()
            gate.wait(5)  # hold A mid-playback
        if round(float(samples[0]), 3) == 0.3:
            n_c += 1
            if n_c == 5:
                last_c.set()

    p = TTSPlayer(FakeEngine(), output)
    try:
        p.say("A")
        assert first_chunk.wait(5)
        p.say("B")
        p.say("C")
        gate.set()
        assert last_c.wait(5)
    finally:
        p.close()
    assert 0.2 not in got
    assert got[0] == 0.1 and got.count(0.1) < 5  # A interrupted
    assert got.count(0.3) == 5


def test_disabled_plays_nothing():
    engine = FakeEngine()
    got = []
    p = TTSPlayer(engine, lambda s, r: got.append(s))
    p.set_enabled(False)
    p.say("A")
    p.close()
    assert got == [] and engine.calls == []


def test_volume_scales_samples():
    got = []
    done = threading.Event()

    def output(s, r):
        got.append(s)
        if len(got) == 5:
            done.set()

    p = TTSPlayer(FakeEngine(), output)
    p.set_volume(0.5)
    try:
        p.say("A")
        assert done.wait(5)
    finally:
        p.close()
    assert all(np.allclose(s, 0.05) for s in got)


def test_close_joins_thread():
    p = TTSPlayer(FakeEngine(), lambda s, r: None)
    p.close()
    assert not p._thread.is_alive()


def test_missing_voice_raises(tmp_path):
    with pytest.raises(ModelMissingError):
        PiperTTS(tmp_path / "nope.onnx").synthesize("xin chào")


@pytest.mark.manual
def test_real_piper_vietnamese():
    voice = Path("models/tts/vi_VN.onnx")
    audio, rate = PiperTTS(voice).synthesize("Xin chào, đây là bản dịch thời gian thực.")
    assert rate > 8000 and len(audio) > rate // 2


def test_on_error_called_when_synthesis_fails():
    class BadEngine:
        def synthesize(self, text):
            raise RuntimeError("no voice")

    errors: list[Exception] = []
    got = threading.Event()

    def on_error(e):
        errors.append(e)
        got.set()

    player = TTSPlayer(BadEngine(), output=lambda s, r: None, on_error=on_error)
    try:
        player.say("A")
        assert got.wait(5)
        assert str(errors[0]) == "no voice"
        player.say("B")  # thread must survive
    finally:
        player.close()


class FakeStream:
    def __init__(self, log, samplerate, channels, dtype):
        self.log, self.samplerate, self.stopped, self.closed = log, samplerate, True, False
        assert (channels, dtype) == (1, "float32")
        log.append(("new", samplerate))

    def start(self):
        self.stopped = False
        self.log.append(("start", self.samplerate))

    def write(self, data):
        assert not self.stopped and not self.closed
        assert data.dtype == np.float32
        self.log.append(("write", self.samplerate, len(data)))

    def abort(self):
        self.stopped = True
        self.log.append(("abort", self.samplerate))

    def stop(self):
        self.stopped = True
        self.log.append(("stop", self.samplerate))

    def close(self):
        self.closed = True
        self.log.append(("close", self.samplerate))


@pytest.fixture
def fake_sd(monkeypatch):
    import sys
    from types import SimpleNamespace

    log: list[tuple] = []
    mod = SimpleNamespace(OutputStream=lambda samplerate, channels, dtype: FakeStream(log, samplerate, channels, dtype))
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    return log


def _wait(pred, timeout=5.0):
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_default_output_uses_one_persistent_stream(fake_sd):
    class Rates:
        def __init__(self):
            self.rate = RATE

        def synthesize(self, text):
            return np.full(250, 0.1, np.float32), self.rate

    eng = Rates()
    p = TTSPlayer(eng)
    try:
        p.say("A")
        assert _wait(lambda: sum(e[0] == "write" for e in fake_sd) == 3)
        p.say("B")
        assert _wait(lambda: sum(e[0] == "write" for e in fake_sd) == 6)
        assert [e for e in fake_sd if e[0] == "new"] == [("new", RATE)]  # reused across sentences
        assert [e[2] for e in fake_sd if e[0] == "write"] == [100, 100, 50] * 2  # 100 ms chunks
        eng.rate = 2000
        p.say("C")
        assert _wait(lambda: ("new", 2000) in fake_sd)
        assert ("close", RATE) in fake_sd  # old-rate stream replaced
    finally:
        p.close()
    assert fake_sd[-2:] == [("stop", 2000), ("close", 2000)]


def test_default_output_aborts_on_interrupt(fake_sd):
    gate = threading.Event()

    class Slow:
        def synthesize(self, text):
            if text == "B":
                return np.full(100, 0.2, np.float32), RATE
            return np.full(500, 0.1, np.float32), RATE

    p = TTSPlayer(Slow())
    orig = FakeStream.write

    def write(self, data):
        orig(self, data)
        if float(data.flat[0]) == np.float32(0.1):
            gate.wait(5)  # hold A mid-sentence

    FakeStream.write = write
    try:
        p.say("A")
        assert _wait(lambda: any(e[0] == "write" for e in fake_sd))
        p.say("B")
        gate.set()
        assert _wait(lambda: ("abort", RATE) in fake_sd and fake_sd[-1][0] == "write")
    finally:
        FakeStream.write = orig
        p.close()
    i = fake_sd.index(("abort", RATE))
    assert ("start", RATE) in fake_sd[i:]  # restarted for B
    assert sum(e[0] == "new" for e in fake_sd) == 1


def test_missing_voice_reported_once():
    class Missing:
        def synthesize(self, text):
            raise ModelMissingError("no voice file")

    errors: list[Exception] = []
    p = TTSPlayer(Missing(), output=lambda s, r: None, on_error=errors.append)
    try:
        for t in ("A", "B", "C"):
            p.say(t)
            assert _wait(lambda: p._pending is None)
        import time

        time.sleep(0.1)
    finally:
        p.close()
    assert len(errors) == 1 and isinstance(errors[0], ModelMissingError)
