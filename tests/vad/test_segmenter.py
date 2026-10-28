import os

import numpy as np
import pytest

from backend.vad.segmenter import Segmenter, SileroSpeechProb

F = 320


def fake(w):
    return 1.0 if float(np.sqrt(np.mean(w**2))) > 0.05 else 0.0


def tone(ms):
    n = ms * 16
    return (0.3 * np.sin(np.arange(n) * 0.1)).astype(np.float32)


def quiet(ms):
    return np.zeros(ms * 16, dtype=np.float32)


def run(parts, flush=False):
    seg = Segmenter(speech_prob=fake)
    x = np.concatenate(parts)
    out = []
    for i in range(0, len(x) - F + 1, F):
        out += seg.push(x[i : i + F])
    if flush:
        out += seg.flush()
    return out


def test_cut_after_500ms_silence():
    out = run([tone(1000), quiet(600)])
    assert len(out) == 1
    assert abs((out[0].t_end - out[0].t_start) - 1.0) <= 0.1


def test_no_cut_on_400ms_pause():
    out = run([tone(1000), quiet(400), tone(1000), quiet(600)])
    assert len(out) == 1


def test_hard_cut_at_8s():
    out = run([tone(10000)], flush=True)
    assert len(out) == 2
    assert abs((out[0].t_end - out[0].t_start) - 8.0) <= 0.02
    assert len(out[0].audio) == 128000


def test_pure_silence_yields_nothing():
    seg = Segmenter(speech_prob=fake)
    out = run([quiet(3000)], flush=True)
    assert out == []


def test_flush_returns_in_progress():
    out = run([tone(700)], flush=True)
    assert len(out) == 1


def test_silero_zeros_is_not_speech():
    try:
        p = SileroSpeechProb()
    except Exception as e:  # onnx unavailable
        pytest.skip(str(e))
    prob = 0.0
    for i in range(0, 16000 - 511, 512):
        prob = p(np.zeros(512, dtype=np.float32))
    assert prob < 0.5
