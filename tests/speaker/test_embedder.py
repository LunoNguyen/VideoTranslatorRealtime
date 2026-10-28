from pathlib import Path

import numpy as np
import pytest

from backend.errors import ModelMissingError
from backend.speaker.embedder import SpeakerEmbedder, fbank

FIX = Path(__file__).resolve().parents[1] / "fixtures"
MODEL = Path(__file__).resolve().parents[2] / "models" / "speaker" / "embedder.onnx"


def _noise(seconds=1.0):
    return np.random.default_rng(0).standard_normal(int(16000 * seconds)).astype(np.float32) * 0.1


def test_fbank_shape():
    assert fbank(_noise()).shape == (98, 80)


def test_fbank_is_mean_normalized():
    assert abs(fbank(_noise()).mean(axis=0)).max() < 1e-4


def test_missing_model_raises(tmp_path):
    with pytest.raises(ModelMissingError):
        SpeakerEmbedder(tmp_path / "x.onnx").load()


class FakeSession:
    def run(self, _outputs, feeds):
        assert feeds["feats"].shape[0] == 1 and feeds["feats"].shape[2] == 80
        out = np.zeros((1, 256), dtype=np.float32)
        out[0, 0], out[0, 1] = 3.0, 4.0
        return [out]


def test_embed_with_fake_session():
    emb = SpeakerEmbedder(Path("unused.onnx"), session=FakeSession())
    v = emb.embed(_noise(0.2))  # shorter than 0.5 s -> padded
    assert v.shape == (256,)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)


def _read(name):
    import wave
    with wave.open(str(FIX / name), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


@pytest.mark.gpu
def test_real_model_separates_voices():
    if not MODEL.is_file():
        pytest.skip("speaker model missing")
    emb = SpeakerEmbedder(MODEL)
    emb.load()
    d1, d2, z1 = (emb.embed(_read(n)) for n in ("spk_david_1.wav", "spk_david_2.wav", "spk_zira_1.wav"))
    same, diff = float(d1 @ d2), float(z1 @ d1)
    print("same", same, "diff", diff)
    assert same > diff + 0.15
