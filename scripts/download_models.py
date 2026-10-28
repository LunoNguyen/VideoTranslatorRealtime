"""Download base models into models/{asr,mt,tts}. Idempotent.

Usage: python scripts/download_models.py [--preset {4gb,6gb}] [--only {asr,mt,tts,speaker}]

Sources and licenses (verified on Hugging Face):
- ASR: Systran/faster-whisper-<small|medium> via faster_whisper.download_model (CTranslate2 format,
  stored as a plain dir models/asr/<name>). Whisper weights: MIT.
- MT: pre-converted CTranslate2 int8 NLLB-200-distilled-600M from HF repo
  JustFrederik/nllb-200-distilled-600M-ct2-int8 (model.bin + sentencepiece.bpe.model + tokenizer.json
  etc.; license cc-by-nc-4.0, inherited from facebook/nllb-200-distilled-600M, i.e. NON-COMMERCIAL).
  Path taken: pre-converted download, no local conversion (so torch is not required). To convert
  yourself instead: pip install -e .[convert] and run
  ct2-transformers-converter --model facebook/nllb-200-distilled-600M --quantization int8
  --output_dir models/mt --copy_files tokenizer.json sentencepiece.bpe.model special_tokens_map.json tokenizer_config.json
- TTS: Piper voice vi_VN-vais1000-medium from HF repo rhasspy/piper-voices (vi/vi_VN/vais1000/medium/);
  MODEL_CARD: dataset VAIS-1000, license CC BY 4.0; finetuned from the en_US lessac medium voice.
- Speaker: Wespeaker/wespeaker-voxceleb-resnet34-LM voxceleb_resnet34_LM.onnx (~26.5 MB, CC-BY-4.0),
  saved as models/speaker/embedder.onnx. Input feats [B,T,80], output embs [B,256].
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PRESETS = {"4gb": "small", "6gb": "medium"}

NLLB_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
NLLB_REVISION = "302d78f00e6fdb50a1064059df7c392b735e9d05"
NLLB_FILES = ["model.bin", "config.json", "shared_vocabulary.txt", "sentencepiece.bpe.model",
              "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json"]

PIPER_REPO = "rhasspy/piper-voices"
PIPER_DIR = "vi/vi_VN/vais1000/medium"
PIPER_NAME = "vi_VN-vais1000-medium"


def download_asr(name: str) -> None:
    dest = MODELS / "asr" / name
    if (dest / "model.bin").is_file():
        print(f"[asr] {name}: already present ({dest})")
        return
    from faster_whisper.utils import download_model
    print(f"[asr] Whisper {name} (CTranslate2) - license: MIT (OpenAI Whisper weights)")
    dest.mkdir(parents=True, exist_ok=True)
    download_model(name, output_dir=str(dest))
    print(f"[asr] saved to {dest}")


def download_mt() -> None:
    dest = MODELS / "mt"
    if (dest / "model.bin").is_file() and (dest / "sentencepiece.bpe.model").is_file():
        print(f"[mt] NLLB: already present ({dest})")
        return
    from huggingface_hub import snapshot_download
    print("[mt] NLLB-200-distilled-600M CT2 int8 - license: cc-by-nc-4.0 (non-commercial use only)")
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(NLLB_REPO, revision=NLLB_REVISION, local_dir=str(dest), allow_patterns=NLLB_FILES)
    shutil.rmtree(dest / ".cache", ignore_errors=True)
    print(f"[mt] saved to {dest}")


def download_tts() -> None:
    dest = MODELS / "tts"
    onnx, cfg = dest / "vi_VN.onnx", dest / "vi_VN.onnx.json"
    if onnx.is_file() and cfg.is_file():
        print(f"[tts] Piper vi_VN: already present ({onnx})")
        return
    from huggingface_hub import hf_hub_download
    dest.mkdir(parents=True, exist_ok=True)
    card = Path(hf_hub_download(PIPER_REPO, f"{PIPER_DIR}/MODEL_CARD")).read_text(encoding="utf-8")
    lic = next((ln.strip() for ln in card.splitlines() if "License" in ln), "see MODEL_CARD")
    print(f"[tts] Piper {PIPER_NAME} - {lic}")
    # Copy via temp file + rename so a partial file never counts as present; config last.
    for fname, target in ((f"{PIPER_NAME}.onnx", onnx), (f"{PIPER_NAME}.onnx.json", cfg)):
        src = hf_hub_download(PIPER_REPO, f"{PIPER_DIR}/{fname}")
        tmp = target.with_name(target.name + ".part")
        shutil.copyfile(src, tmp)
        tmp.replace(target)
    print(f"[tts] saved to {dest}")


def download_speaker() -> None:
    target = MODELS / "speaker" / "embedder.onnx"
    if target.is_file():
        print(f"[speaker] WeSpeaker ResNet34-LM: already present ({target})")
        return
    from huggingface_hub import hf_hub_download
    print("[speaker] WeSpeaker ResNet34-LM ONNX - license: CC-BY-4.0")
    target.parent.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(repo_id="Wespeaker/wespeaker-voxceleb-resnet34-LM",
                          filename="voxceleb_resnet34_LM.onnx")
    tmp = target.with_name(target.name + ".part")
    shutil.copyfile(src, tmp)
    tmp.replace(target)
    print(f"[speaker] saved to {target}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", choices=sorted(PRESETS), default="4gb")
    ap.add_argument("--only", choices=["asr", "mt", "tts", "speaker"])
    args = ap.parse_args(argv)
    if args.only in (None, "asr"):
        download_asr(PRESETS[args.preset])
    if args.only in (None, "mt"):
        download_mt()
    if args.only in (None, "tts"):
        download_tts()
    if args.only in (None, "speaker"):
        download_speaker()
    return 0


if __name__ == "__main__":
    sys.exit(main())
