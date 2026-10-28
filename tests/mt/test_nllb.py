from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.errors import ModelMissingError
from backend.mt.nllb import NLLB_CODES, NLLBTranslator, build_request

MODEL_DIR = Path("models/mt")


def test_build_request_no_context():
    src, prefix = build_request(["▁Hello"], [], "eng_Latn", "vie_Latn")
    assert src[0] == "eng_Latn" and src[-1] == "</s>" and src == ["eng_Latn", "▁Hello", "</s>"]
    assert prefix == ["vie_Latn"]


def test_build_request_with_context():
    ctx = [(["▁Hi"], ["▁Chào"])]
    src, prefix = build_request(["▁Bye"], ctx, "eng_Latn", "vie_Latn")
    assert src == ["eng_Latn", "▁Hi", "▁Bye", "</s>"]
    assert prefix == ["vie_Latn", "▁Chào"]


class FakeCT2:
    def __init__(self, outputs=None):
        self.calls, self.outputs = [], outputs

    def translate_batch(self, sources, target_prefix, beam_size, max_decoding_length):
        self.calls.append((sources[0], target_prefix[0]))
        n = len(self.calls)
        out = self.outputs[n - 1] if self.outputs else [f"▁out{n}", "</s>"]
        return [SimpleNamespace(hypotheses=[list(target_prefix[0]) + out])]


def make(n=1, outputs=None):
    t = NLLBTranslator(MODEL_DIR, "cpu", context_sentences=n)
    t._tr = FakeCT2(outputs)
    t._tokenize = lambda text: [f"▁{w}" for w in text.split()]
    t._detok = lambda toks: " ".join(x.lstrip("▁") for x in toks)
    return t


def test_context_limited_to_n():
    t = make(1)
    t.translate("one", "en")
    t.translate("two", "en")
    out3 = t.translate("three", "en")
    assert out3 == "out3" and "out2" not in out3
    src3, prefix3 = t._tr.calls[2]
    assert src3 == ["eng_Latn", "▁two", "▁three", "</s>"]
    assert prefix3 == ["vie_Latn", "▁out2"]


def test_empty_output_not_pushed_and_context_per_lang():
    t = make(1, outputs=[["</s>"], ["▁x", "</s>"], ["▁y", "</s>"]])
    assert t.translate("one", "en") == ""
    assert len(t._ctx["en"]) == 0
    t.translate("two", "ja")
    t.translate("three", "en")
    assert t._tr.calls[2][1] == ["vie_Latn"]  # en context empty; ja context not shared
    t.reset_context()
    assert not any(t._ctx.values())


def test_translate_commit_false_keeps_context():
    t = make(1)
    t.translate("A", "en")
    t.translate("B", "en", commit=False)
    t.translate("C", "en")
    src2, _ = t._tr.calls[1]
    assert src2 == ["eng_Latn", "▁A", "▁B", "</s>"]  # B still sees A as context
    src3, prefix3 = t._tr.calls[2]
    assert src3 == ["eng_Latn", "▁A", "▁C", "</s>"]  # C carries A, not B
    assert prefix3 == ["vie_Latn", "▁out1"]


def test_missing_model_dir_raises(tmp_path):
    t = NLLBTranslator(tmp_path / "nope", "cpu")
    with pytest.raises(ModelMissingError):
        t.translate("hi", "en")


def test_unsupported_lang_raises():
    with pytest.raises(ValueError, match="fr"):
        make().translate("hi", "fr")


def test_failed_load_does_not_keep_broken_translator(monkeypatch, tmp_path):
    import ctranslate2
    (tmp_path / "model.bin").write_bytes(b"x")

    class Boom:
        def __init__(self, *a, **k): pass
        def translate_batch(self, *a, **k): raise RuntimeError("warm fail")

    monkeypatch.setattr(ctranslate2, "Translator", Boom)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: object())
    t = NLLBTranslator(tmp_path, "cpu")
    for _ in range(2):
        with pytest.raises(RuntimeError):
            t.load()
        assert t._tr is None


@pytest.mark.gpu
def test_translate_real_consecutive():
    t = NLLBTranslator(MODEL_DIR, "auto", context_sentences=1)
    first = t.translate("Hello, how are you?", "en")
    second = t.translate("I am going to the market today.", "en")
    assert first.strip() and second.strip() and first not in second


@pytest.mark.gpu
def test_translate_real():
    t = NLLBTranslator(MODEL_DIR, "auto")
    out = t.translate("Hello, how are you?", "en")
    assert out.strip() and "Hello" not in out
    assert t.translate("こんにちは、元気ですか?", "ja").strip()
