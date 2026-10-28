import numpy as np

from backend.speaker.cluster import SpeakerClusterer


def unit(*idx_weights):
    v = np.zeros(256, dtype=np.float32)
    for i, w in idx_weights:
        v[i] = w
    return v / np.linalg.norm(v)


def E(i):
    return unit((i, 1.0))


def test_first_long_creates_0():
    c = SpeakerClusterer()
    assert c.assign(E(0), 2.0, now=0.0) == 0
    assert c.count == 1


def test_same_voice_same_id():
    c = SpeakerClusterer()
    a = c.assign(E(0), 2.0, 0.0)
    b = c.assign(unit((0, 1.0), (1, 0.1)), 2.0, 1.0)
    assert a == b == 0
    assert c.count == 1


def test_new_voice_new_id():
    c = SpeakerClusterer()
    assert c.assign(E(0), 2.0, 0.0) == 0
    assert c.assign(E(1), 2.0, 1.0) == 1
    assert c.count == 2


def test_short_first_returns_none():
    c = SpeakerClusterer()
    assert c.assign(E(0), 0.5, 0.0) is None
    assert c.count == 0


def test_short_unknown_goes_to_nearest():
    c = SpeakerClusterer()
    c.assign(E(0), 2.0, 0.0)
    c.assign(E(1), 2.0, 1.0)
    # closer to speaker 1 but below threshold
    v = unit((1, 0.5), (2, 1.0))
    assert c.assign(v, 0.5, 2.0) == 1
    assert c.count == 2


def test_max_speakers_caps_at_6():
    c = SpeakerClusterer()
    ids = [c.assign(E(i), 2.0, float(i)) for i in range(7)]
    assert ids[:6] == [0, 1, 2, 3, 4, 5]
    assert ids[6] in range(6)
    assert c.count == 6


def test_idle_reset_after_300s():
    c = SpeakerClusterer()
    assert c.assign(E(0), 2.0, 0.0) == 0
    assert c.assign(E(1), 2.0, 1.0) == 1
    assert c.assign(E(1), 2.0, 302.0) == 0
    assert c.count == 1


def test_match_does_not_create_or_update():
    c = SpeakerClusterer()
    c.assign(E(0), 2.0, 0.0)
    before = c._centroids[0].copy()
    assert c.match(unit((0, 1.0), (1, 0.3)), 1.0) == 0
    assert c.match(E(5), 1.0) is None
    assert c.count == 1
    np.testing.assert_array_equal(c._centroids[0], before)


def test_centroid_ema():
    c = SpeakerClusterer()
    c0 = E(0)
    c.assign(c0, 2.0, 0.0)
    v2 = unit((0, 1.0), (1, 0.5))
    assert c.assign(v2, 2.0, 1.0) == 0
    exp = 0.8 * c0 + 0.2 * v2
    exp = exp / np.linalg.norm(exp)
    np.testing.assert_allclose(c._centroids[0], exp, atol=1e-6)
