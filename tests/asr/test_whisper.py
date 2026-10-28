from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.asr.whisper import (CudaOutOfMemoryError, ModelMissingError, Transcript,
                                 WhisperASR, is_hallucination)

FIX = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.parametrize("text,nsp,expected", [
    ("Thank you for watching.", 0.1, True),
    ("Thanks for watching!", 0.1, True),
    ("ご視聴ありがとうございました", 0.1, True),
    ("字幕視聴ありがとうございました", 0.1, True),
    ("", 0.1, True),
    ("  ...  ", 0.1, True),
    ("。、!?", 0.1, True),
    ("Hello there", 0.7, False),  # no_speech_prob is judged per segment (with avg_logprob)
    ("Hello there", 0.6, False),
    ("Hello there, how are you?", 0.1, False),
    ("こんにちは", 0.1, False),
])
def test_filters_hallucinations_and_no_speech(text, nsp, expected):
    assert is_hallucination(Transcript(text, "en", nsp)) is expected


class FakeModel:
    def __init__(self, lang, texts, nsp=(0.1,), logprob=None):
        self.lang, self.texts, self.nsp = lang, texts, nsp
        self.logprob = logprob or (-0.2,) * len(texts)
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        segs = [SimpleNamespace(text=t, no_speech_prob=p, avg_logprob=lp)
                for t, p, lp in zip(self.texts, self.nsp, self.logprob)]
        return iter(segs), SimpleNamespace(language=self.lang)


def _asr(model):
    asr = WhisperASR("small", "cpu", "int8")
    asr._model = model
    return asr


AUDIO = np.zeros(16000, dtype=np.float32)


def test_lang_outside_en_ja_dropped():
    assert _asr(FakeModel("fr", ["Bonjour"])).transcribe(AUDIO, "auto") is None


def test_auto_uses_detected_lang_and_params():
    m = FakeModel("ja", ["こんにちは", "世界"], (0.1, 0.3))
    t = _asr(m).transcribe(AUDIO, "auto")
    assert t == Transcript("こんにちは世界", "ja", 0.3)
    assert m.kwargs["language"] is None
    assert m.kwargs["vad_filter"] is False
    assert m.kwargs["beam_size"] == 1
    assert m.kwargs["condition_on_previous_text"] is False


def test_en_join_with_space_and_fixed_lang():
    m = FakeModel("en", [" Hello", " world"], (0.1, 0.2))
    t = _asr(m).transcribe(AUDIO, "en")
    assert t.text == "Hello world" and t.lang == "en" and m.kwargs["language"] == "en"


def test_short_audio_returns_none():
    m = FakeModel("en", ["Hi"])
    assert _asr(m).transcribe(np.zeros(4000, dtype=np.float32), "en") is None
    assert m.kwargs is None


def test_music_like_segment_with_good_logprob_kept():
    m = FakeModel("en", [" Hello over the music"], (0.8,), (-0.3,))
    t = _asr(m).transcribe(AUDIO, "en")
    assert t is not None and t.text == "Hello over the music"


def test_no_speech_segment_with_bad_logprob_dropped():
    m = FakeModel("en", [" Hello", " uh mm"], (0.1, 0.9), (-0.2, -1.5))
    assert _asr(m).transcribe(AUDIO, "en").text == "Hello"
    assert _asr(FakeModel("en", [" uh mm"], (0.9,), (-1.5,))).transcribe(AUDIO, "en") is None


def test_trailing_hallucination_stripped_from_real_speech():
    m = FakeModel("en", [" We should go now.", " Thanks for watching."], (0.1, 0.2))
    assert _asr(m).transcribe(AUDIO, "en").text == "We should go now."
    m = FakeModel("ja", ["行きましょう", "ご視聴ありがとうございました", "。"], (0.1, 0.1, 0.1))
    assert _asr(m).transcribe(AUDIO, "ja").text == "行きましょう"


def test_hallucination_returns_none():
    assert _asr(FakeModel("en", ["Thanks for watching!"])).transcribe(AUDIO, "en") is None


class _Model:
    def __init__(self, fail=None, lazy=False):
        self.fail, self.lazy = fail, lazy

    def transcribe(self, audio, **kw):
        if self.fail and not self.lazy:
            raise RuntimeError(self.fail)

        def gen():
            raise RuntimeError(self.fail)
            yield
        return (gen() if self.fail else iter([])), None


def _patch(monkeypatch, tmp_path, cuda_fail=None, cpu_fail=None, lazy=False, builds=None):
    builds = builds if builds is not None else []
    (tmp_path / "small").mkdir(exist_ok=True)

    def fake_wm(path, device, compute_type):
        builds.append((path, device, compute_type))
        return _Model(cuda_fail if device == "cuda" else cpu_fail, lazy)

    monkeypatch.setattr("backend.asr.whisper.WhisperModel", fake_wm)
    return builds


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("msg,reason", [
    ("Library cublas64_12.dll is not found", "cuda_unavailable"),
    ("CUDA failed with error out of memory", "cuda_oom"),
])
def test_auto_falls_back_on_warmup_failure(monkeypatch, tmp_path, lazy, msg, reason):
    builds = _patch(monkeypatch, tmp_path, cuda_fail=msg, lazy=lazy)
    asr = WhisperASR("small", "auto", "float16", model_root=tmp_path)
    asr.load()
    assert [b[1] for b in builds] == ["cuda", "cpu"]
    assert builds[1][2] == "int8" and builds[1][0] == str(tmp_path / "small")
    assert asr.device_used == "cpu" and asr.fallback_reason == reason


def test_auto_cuda_ok_no_fallback(monkeypatch, tmp_path):
    builds = _patch(monkeypatch, tmp_path)
    asr = WhisperASR("small", "auto", "float16", model_root=tmp_path)
    asr.load()
    asr.load()
    assert len(builds) == 1 and asr.device_used == "cuda" and asr.fallback_reason is None


def test_auto_both_builds_fail_raises(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, cuda_fail="cublas missing", cpu_fail="cpu broken")
    with pytest.raises(RuntimeError, match="cpu broken"):
        WhisperASR("small", "auto", "int8", model_root=tmp_path).load()


def test_explicit_cuda_oom_raises(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, cuda_fail="CUDA out of memory")
    with pytest.raises(CudaOutOfMemoryError):
        WhisperASR("small", "cuda", "int8", model_root=tmp_path).load()


def test_model_missing_not_retried_on_cpu(monkeypatch, tmp_path):
    builds = _patch(monkeypatch, tmp_path)
    (tmp_path / "small").rmdir()

    def boom(name, cache_dir=None):
        raise OSError("no network")

    monkeypatch.setattr("backend.asr.whisper.download_model", boom)
    with pytest.raises(ModelMissingError):
        WhisperASR("small", "auto", "int8", model_root=tmp_path).load()
    assert builds == []


def test_download_path_used_when_not_local(monkeypatch, tmp_path):
    builds = _patch(monkeypatch, tmp_path)
    (tmp_path / "small").rmdir()
    monkeypatch.setattr("backend.asr.whisper.download_model", lambda n, cache_dir=None: "/cache/x")
    WhisperASR("small", "cpu", "int8", model_root=tmp_path).load()
    assert builds[0][0] == "/cache/x"


def test_load_prefers_local_dir(monkeypatch, tmp_path):
    builds = _patch(monkeypatch, tmp_path)
    WhisperASR("small", "cpu", "int8", model_root=tmp_path).load()
    assert builds[0][0] == str(tmp_path / "small")


@pytest.mark.parametrize("text", ["thanks for watching", "THANK YOU FOR WATCHING!!", " ご視聴ありがとうございました。 "])
def test_hallucination_normalized(text):
    assert is_hallucination(Transcript(text, "en", 0.0))


def _load_wav(path: Path) -> np.ndarray:
    import wave
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


@pytest.mark.gpu
@pytest.mark.parametrize("fname,needle,lang", [
    ("en_hello.wav", "hello", "en"),
    ("ja_konnichiwa.wav", "こんにちは", "ja"),
])
def test_transcribe_fixtures(fname, needle, lang):
    path = FIX / fname
    if not path.exists():
        pytest.skip(f"missing fixture {fname}")
    asr = WhisperASR("small", "auto", "int8")
    t = asr.transcribe(_load_wav(path), "auto")
    assert t is not None and t.lang == lang
    assert needle in t.text.lower()
