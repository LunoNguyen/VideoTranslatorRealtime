"""Manual: plays an audible 440 Hz tone for a few seconds.

Run: python -m pytest tests/capture/test_source_manual.py -m manual -s
"""

import subprocess
import sys
import time

import numpy as np
import pytest

from backend.capture.mixer import Mixer
from backend.capture.source import ProcessLoopbackSource

pytestmark = pytest.mark.manual

TONE = (
    "import numpy as np, sounddevice as sd; sr=48000; t=np.arange(sr*{dur})/sr;"
    "sd.play((0.2*np.sin(2*np.pi*440*t)).astype(np.float32), sr, blocking=True)"
)


def capture_rms(pid: int, seconds: float = 2.0) -> tuple[float, int]:
    src = ProcessLoopbackSource(pid)
    src.start()
    mixer = Mixer()
    mixer.add(src)
    frames = []
    try:
        for frame in mixer.frames():
            frames.append(frame)
            if len(frames) * 0.02 >= seconds:
                break
    finally:
        mixer.close()
        src.stop()
    out = np.concatenate(frames)
    return float(np.sqrt(np.mean(out**2))), len(frames)


@pytest.fixture
def tone_proc():
    p = subprocess.Popen([sys.executable, "-c", TONE.format(dur=6)])
    time.sleep(2.0)  # let numpy/sounddevice import and the stream open
    yield p
    p.kill()


@pytest.fixture
def silent_proc():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    yield p
    p.kill()


def test_captures_playing_process(tone_proc):
    rms, n = capture_rms(tone_proc.pid)
    print(f"tone pid={tone_proc.pid} frames={n} rms={rms:.4f}")
    assert rms > 0.01


def test_other_process_audio_not_leaked(tone_proc, silent_proc):
    rms, n = capture_rms(silent_proc.pid)
    print(f"silent pid={silent_proc.pid} frames={n} rms={rms:.6f} (tone playing elsewhere)")
    assert n > 0
    assert rms < 0.001
