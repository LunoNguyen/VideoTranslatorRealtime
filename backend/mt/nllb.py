"""NLLB-200-distilled-600M (CTranslate2 int8) translator with sentence context."""
from __future__ import annotations

import logging
import threading
from collections import deque
from pathlib import Path

from backend.errors import ModelMissingError
from backend.gpu import register_cuda_dlls

log = logging.getLogger(__name__)

NLLB_CODES = {"en": "eng_Latn", "ja": "jpn_Jpan", "vi": "vie_Latn"}
EOS = "</s>"

Ctx = list[tuple[list[str], list[str]]]


def build_request(src_tokens: list[str], ctx: Ctx, src_code: str, tgt_code: str) -> tuple[list[str], list[str]]:
    """Return (source_tokens, target_prefix); the model continues after the prefix with the translation."""
    source = [src_code]
    prefix = [tgt_code]
    for s, t in ctx:
        source += s
        prefix += t
    return source + src_tokens + [EOS], prefix


class NLLBTranslator:
    def __init__(self, model_dir: Path, device: str, context_sentences: int = 1) -> None:
        self.model_dir, self.device = Path(model_dir), device
        self.context_sentences = context_sentences
        self.device_used: str | None = None
        self.fallback_reason: str | None = None
        self._tr = None
        self._tok = None
        self._ctx: dict[str, deque] = {}
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load tokenizer + CT2 model (idempotent). Raises ModelMissingError if model_dir is absent."""
        with self._lock:
            if self._tr is not None:
                return
            if not (self.model_dir / "model.bin").is_file():
                raise ModelMissingError(f"NLLB model not found in {self.model_dir}")
            register_cuda_dlls()
            import ctranslate2
            from transformers import AutoTokenizer
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()  # silences the spurious "Mistral regex" warning
            tok = AutoTokenizer.from_pretrained(str(self.model_dir))
            path = str(self.model_dir)

            def build(device: str):
                tr = ctranslate2.Translator(path, device=device, compute_type="int8")
                self._warm(tr)  # CT2 can construct on CUDA yet fail at first run (missing cuBLAS)
                return tr

            if self.device == "auto":
                try:
                    tr, used = build("cuda"), "cuda"
                except Exception as e:
                    self.fallback_reason = "cuda_unavailable"
                    log.warning("NLLB CUDA failed (%s: %s); falling back to CPU", type(e).__name__, e)
                    tr, used = build("cpu"), "cpu"
            else:
                tr, used = build(self.device), self.device
            self._tok, self._tr, self.device_used = tok, tr, used
            log.info("NLLB loaded on %s", used)

    @staticmethod
    def _warm(tr) -> None:
        tr.translate_batch([["eng_Latn", "▁Hi", EOS]], target_prefix=[["vie_Latn"]],
                                 beam_size=1, max_decoding_length=4)

    def _tokenize(self, text: str) -> list[str]:
        return self._tok.convert_ids_to_tokens(self._tok.encode(text, add_special_tokens=False))

    def _detok(self, tokens: list[str]) -> str:
        return self._tok.convert_tokens_to_string(tokens)

    def reset_context(self) -> None:
        self._ctx.clear()

    def translate(self, text: str, src_lang: str, tgt_lang: str = "vi") -> str:
        for lang in (src_lang, tgt_lang):
            if lang not in NLLB_CODES:
                raise ValueError(f"Unsupported language '{lang}'; supported: {sorted(NLLB_CODES)}")
        text = text.strip()
        if not text:
            return ""
        self.load()
        with self._lock:
            ctx = self._ctx.setdefault(src_lang, deque(maxlen=max(self.context_sentences, 1)))
            history = list(ctx)[-self.context_sentences:] if self.context_sentences > 0 else []
            src_tokens = self._tokenize(text)
            source, prefix = build_request(src_tokens, history, NLLB_CODES[src_lang], NLLB_CODES[tgt_lang])
            res = self._tr.translate_batch([source], target_prefix=[prefix], beam_size=2, max_decoding_length=256)
            hyp = [t for t in list(res[0].hypotheses[0])[len(prefix):] if t != EOS]
            out = self._detok(hyp).strip() if hyp else ""
            if out and self.context_sentences > 0:
                ctx.append((src_tokens, hyp))
            return out
