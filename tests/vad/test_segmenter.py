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


# ---- partial snapshots ----
from backend.vad.segmenter import Partial, Segment  # noqa: E402


def run_p(parts, flush=False, **kw):
    seg = Segmenter(speech_prob=fake, **kw)
    x = np.concatenate(parts)
    out = []
    for i in range(0, len(x) - F + 1, F):
        out += seg.push(x[i : i + F])
    if flush:
        out += seg.flush()
    return out


def test_no_partials_by_default():
    out = run_p([tone(3000), quiet(600)])
    assert all(isinstance(o, Segment) for o in out)


def test_partials_every_0_7s_after_1s():
    out = run_p([tone(3000)], partial_every_s=0.7)
    parts = [o for o in out if isinstance(o, Partial)]
    ends = [p.t_end for p in parts]
    assert len(ends) >= 3
    for got, want in zip(ends[:3], [1.0, 1.7, 2.4]):
        assert abs(got - want) <= 0.033


def test_partial_and_segment_share_seg_id():
    out = run_p([tone(1500), quiet(600)], partial_every_s=0.7)
    parts = [o for o in out if isinstance(o, Partial)]
    segs = [o for o in out if isinstance(o, Segment)]
    assert parts and len(segs) == 1
    assert {p.seg_id for p in parts} == {1}
    assert segs[0].seg_id == 1


def test_new_sentence_increments_seg_id_and_resets_cadence():
    out = run_p([tone(1500), quiet(600), tone(1500), quiet(600)], partial_every_s=0.7)
    segs = [o for o in out if isinstance(o, Segment)]
    assert [s.seg_id for s in segs] == [1, 2]
    p2 = [o for o in out if isinstance(o, Partial) and o.seg_id == 2]
    assert p2 and p2[0].t_end - segs[1].t_start >= 1.0 - 0.032


def test_partial_audio_capped_8s():
    out = run_p([tone(12000)], max_s=20, partial_every_s=0.7)
    parts = [o for o in out if isinstance(o, Partial)]
    assert parts
    assert len(parts[-1].audio) <= 8 * 16000
    assert len(parts[-1].audio) == 8 * 16000


def test_max_cut_starts_new_seg_id():
    out = run_p([tone(10000)], max_s=8, partial_every_s=0.7)
    segs = [o for o in out if isinstance(o, Segment)]
    assert segs[0].seg_id == 1
    later = [o for o in out if isinstance(o, Partial) and o.t_end > segs[0].t_end]
    assert later and all(p.seg_id == 2 for p in later)


def test_set_partial_every_none_stops():
    seg = Segmenter(speech_prob=fake, partial_every_s=0.7)
    x = tone(3000)
    out = []
    for i in range(0, len(x) - F + 1, F):
        out += seg.push(x[i : i + F])
    assert any(isinstance(o, Partial) for o in out)
    seg.set_partial_every(None)
    x = tone(3000)
    out = []
    for i in range(0, len(x) - F + 1, F):
        out += seg.push(x[i : i + F])
    assert not any(isinstance(o, Partial) for o in out)


def test_no_partial_during_trailing_silence_short_utterance():
    out = run_p([tone(750), quiet(600)], partial_every_s=0.7)
    assert not any(isinstance(o, Partial) for o in out)
    assert len([o for o in out if isinstance(o, Segment)]) == 1


def test_no_partial_after_last_speech_window():
    out = run_p([tone(2300), quiet(600)], partial_every_s=0.7)
    segs = [o for o in out if isinstance(o, Segment)]
    parts = [o for o in out if isinstance(o, Partial)]
    assert parts and len(segs) == 1
    assert all(p.t_end <= 2.3 + 0.033 for p in parts)
