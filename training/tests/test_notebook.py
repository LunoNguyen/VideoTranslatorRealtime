import ast
from pathlib import Path

import nbformat
import pytest

TRAINING = Path(__file__).resolve().parents[1]
NB_PATH = TRAINING / "finetune_nllb_colab.ipynb"
WRITEFILE = "%%writefile cleaning.py\n"


@pytest.fixture(scope="module")
def nb():
    nb = nbformat.read(str(NB_PATH), as_version=4)
    nbformat.validate(nb)
    return nb


def code_cells(nb):
    return [c.source for c in nb.cells if c.cell_type == "code"]


def test_code_cells_parse(nb):
    for src in code_cells(nb):
        body = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%")))
        ast.parse(body)


def test_embedded_cleaning_matches_module(nb):
    embedded = [s for s in code_cells(nb) if s.startswith(WRITEFILE)]
    assert len(embedded) == 1
    assert embedded[0][len(WRITEFILE):].strip() == (TRAINING / "cleaning.py").read_text(encoding="utf-8").strip()


def test_notebook_uses_cleaning_and_brief_values(nb):
    src = "\n".join(code_cells(nb))
    assert ("from cleaning import (clean_pair, dedup_norm, dedup_pairs, lang_ok, leak_keys, make_context_pairs, "
            "should_sync,\n                      split_segments, touches_test)") in src
    for needle in ("SAVE_STEPS = 2000", "save_total_limit=1", "output_dir=LOCAL_CKPT", "callbacks=[DriveSync()]",
                   "DRIVE_SYNC_MIN = 60", "should_sync(state.global_step, state.max_steps",
                   "half().save_pretrained(FINAL_DIR)", "load_as_float16=True",
'"phomt_en_vi": False', "EPOCHS = 2", "LR = 5e-5", "BATCH = 8", "GRAD_ACCUM = 4", "FP16 = True",
                   "LABSE_MIN_SIM = 0.7", "Seq2SeqTrainer(", 'quantization="int8"', 'EXPORT_NAME = "nllb-vi-ct2-int8"',
                   "files.download(", '"en": "eng_Latn", "ja": "jpn_Jpan", "vi": "vie_Latn"'):
        assert needle in src, needle
