import ast
import inspect
import os
import re
from pathlib import Path

import nbformat
import pytest

from training import cleaning

ROOT = Path(__file__).resolve().parents[2]
NB_PATH = ROOT / "training" / "finetune_whisper_ja_colab.ipynb"
HELPERS = "# CELL: helpers\n"


@pytest.fixture(scope="module")
def nb():
    nb = nbformat.read(str(NB_PATH), as_version=4)
    nbformat.validate(nb)
    return nb


def code_cells(nb):
    return [c.source for c in nb.cells if c.cell_type == "code"]


@pytest.fixture(scope="module")
def helpers(nb):
    cells = [s for s in code_cells(nb) if s.startswith(HELPERS)]
    assert len(cells) == 1
    ns: dict = {}
    exec(compile(cells[0], "helpers", "exec"), ns)
    return ns


def export_name(nb) -> str:
    m = re.search(r'^EXPORT_NAME = "([^"]+)"', "\n".join(code_cells(nb)), re.M)
    assert m
    return m.group(1)


def test_code_cells_parse(nb):
    for src in code_cells(nb):
        body = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%")))
        ast.parse(body)


def test_outputs_cleared_and_no_token(nb):
    for c in nb.cells:
        if c.cell_type == "code":
            assert c.outputs == [] and c.execution_count is None
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", "\n".join(c.source for c in nb.cells))


def test_brief_values_present(nb):
    src = "\n".join(code_cells(nb))
    for needle in ('BASE_MODEL = "openai/whisper-small"', "MAX_STEPS = 2000", "LR = 1e-5", "FP16 = True",
                   "SAMPLE_RATE = 16000", "gradient_checkpointing=True", "save_total_limit=1",
                   "output_dir=LOCAL_CKPT", "callbacks=[DriveSync()]", "max_steps=MAX_STEPS",
                   "should_sync(state.global_step, state.max_steps", "DRIVE_SYNC_MAX = 2",
                   'language="ja", task="transcribe"', "Seq2SeqTrainer(", "corpus_cer(",
                   '"ct2-transformers-converter"', '"--quantization", "int8"',
                   '"--copy_files", "tokenizer.json", "preprocessor_config.json"',
                   'ZIP_NAME = "whisper-ja-ct2-int8.zip"', "login(", "files.download(",
                   '"common_voice_ja": True', '"reazonspeech": True', "auth_check("):
        assert needle in src, needle


def test_export_name_passes_backend_regex_and_resolves(nb, tmp_path):
    from backend.asr.whisper import WhisperASR
    from backend.server.protocol import _MODEL_RE
    name = export_name(nb)
    assert _MODEL_RE.match(name)
    (tmp_path / name).mkdir()
    assert WhisperASR(name, "cpu", "int8", model_root=tmp_path)._resolve() == str(tmp_path / name)


def test_embedded_should_sync_matches_module(nb, helpers):
    cell = next(s for s in code_cells(nb) if s.startswith(HELPERS))
    assert inspect.getsource(cleaning.should_sync).strip() in cell
    h = 3600.0
    assert helpers["should_sync"](500, 2000, None, 0.0, 30)
    assert not helpers["should_sync"](2000, 2000, 0.0, 5 * h, 30)  # never at the final step
    assert not helpers["should_sync"](1500, 2000, 0.0, h, 30, n_synced=2, max_syncs=2)


def test_norm_cer_strips_punct_space_and_width(helpers):
    norm = helpers["norm_cer"]
    assert norm("こんにちは、世界！ ") == norm("こんにちは世界") == "こんにちは世界"
    assert norm("ＡＢＣ。「テスト」") == "abcテスト"
    assert norm(" 。、！ ") == ""


def test_corpus_cer(helpers):
    cer = helpers["corpus_cer"]
    assert cer(["こんにちは"], ["こんにちは。"]) == 0.0
    assert cer(["こんにちわ"], ["こんにちは"]) == pytest.approx(0.2)
    assert cer(["", "abcd"], ["ab", "abcd"]) == pytest.approx(2 / 6)  # corpus level: sum edits / sum ref chars
    assert cer(["x"], ["、"]) == 0.0  # empty references are skipped


def test_asr_text_ok(helpers):
    ok = helpers["asr_text_ok"]
    assert ok("今日はいい天気ですね。")
    assert not ok("［バーミヤンズオンエア獲得も疑惑浮上］")  # on-screen caption, not speech
    assert not ok("。。")


def test_zip_has_model_files_at_root(nb):
    """Zip is made from inside the model folder: unzip into models/asr/<EXPORT_NAME>/ gives model.bin there."""
    cell = next(s for s in code_cells(nb) if s.startswith("# CELL: zip"))
    calls = [ln for ln in cell.splitlines() if "make_archive(" in ln]
    assert len(calls) == 1 and "root_dir=OUT_DIR)" in calls[0] and "base_dir" not in calls[0]
    md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert f"vào `models/asr/{export_name(nb)}/`" in md


def test_reazon_access_before_hash_and_narrow_except(nb):
    src = "\n".join(code_cells(nb))
    assert src.index("USE = {") < src.index("DATA_CFG = dict(datasets=USE,")
    fn = src[src.index("def reazon_access"):src.index("USE = {")]
    assert "except (GatedRepoError, RepositoryNotFoundError)" in fn and "except Exception" not in fn


@pytest.mark.gpu
def test_finetuned_model_loads_in_app(nb):
    """After unzipping into models/asr/, the exported model loads and transcribes with WhisperASR unchanged."""
    name = os.environ.get("WHISPER_JA_MODEL", export_name(nb))
    if not (ROOT / "models" / "asr" / name / "model.bin").is_file():
        pytest.skip(f"models/asr/{name} not installed")
    fix = ROOT / "tests" / "fixtures" / "ja_konnichiwa.wav"
    if not fix.exists():
        pytest.skip("missing fixture tests/fixtures/ja_konnichiwa.wav (16 kHz mono Japanese 'konnichiwa')")
    import wave
    import numpy as np
    from backend.asr.whisper import WhisperASR
    asr = WhisperASR(name, "auto", "int8", model_root=ROOT / "models" / "asr")
    with wave.open(str(fix)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    t = asr.transcribe(audio, "ja")
    assert t is not None and "こんにちは" in t.text
