"""Parallel-corpus cleaning helpers for the NLLB fine-tune notebook (stdlib only, standalone)."""
from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Hashable, Iterable

Pair = tuple[str, str]

MAX_CHARS = 200
MAX_RATIO = 3.0
CJK_WEIGHT = 3  # one CJK/kana char carries roughly as much content as 3 Latin chars

_TAG = re.compile(r"<[^>]*>")
_ASS = re.compile(r"\{[^}]*\}")  # subtitle override tags such as {\an8}
_MUSIC = re.compile(r"[♪♫♬♩]")
_WS = re.compile(r"\s+")
_VI_CHARS = set(
    "ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
)


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF
            or 0xF900 <= o <= 0xFAFF or 0xFF66 <= o <= 0xFF9F)  # kana + CJK ideographs; Hangul excluded


def _is_latin(ch: str) -> bool:
    o = ord(ch)
    return ch.isalpha() and (o < 0x250 or 0x1E00 <= o <= 0x1EFF)


def clean_text(s: str) -> str:
    """Unescape entities, drop HTML/subtitle tags and music symbols, normalise whitespace (NFC)."""
    s = html.unescape(html.unescape(s))
    s = _ASS.sub(" ", _TAG.sub(" ", s))
    s = _MUSIC.sub(" ", s)
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch in "\t\n")
    return _WS.sub(" ", unicodedata.normalize("NFC", s)).strip()


def weighted_len(s: str) -> int:
    """Character length where CJK/kana count CJK_WEIGHT (ja has no spaces and denser script)."""
    return sum(CJK_WEIGHT if _is_cjk(ch) else 1 for ch in s)


def clean_pair(src: str, tgt: str) -> Pair | None:
    """Clean one pair; None if empty, mojibake, longer than MAX_CHARS or length ratio outside [1/3, 3]."""
    if "�" in src + tgt or any(unicodedata.category(c) == "Cs" for c in src + tgt):
        return None
    src, tgt = clean_text(src), clean_text(tgt)
    if not src or not tgt or len(src) > MAX_CHARS or len(tgt) > MAX_CHARS:
        return None
    ratio = weighted_len(tgt) / weighted_len(src)
    if ratio > MAX_RATIO or ratio < 1 / MAX_RATIO:
        return None
    return src, tgt


def lang_ok(text: str, lang: str) -> bool:
    """Script-based language check for 'en', 'ja', 'vi' (no model download needed)."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    cjk = sum(_is_cjk(ch) for ch in letters)
    latin = sum(_is_latin(ch) for ch in letters)
    if lang == "ja":
        return cjk / len(letters) >= 0.5
    if cjk or latin / len(letters) < 0.8:
        return False
    if lang == "vi" and len(text.split()) >= 4:
        return any(ch in _VI_CHARS for ch in text.lower())  # long vi lines virtually always carry diacritics
    return lang in ("en", "vi")


def make_context_pairs(rows: list[Pair], n: int = 1, sep: str = " ") -> list[Pair]:
    """For consecutive rows of ONE document, prepend up to n previous rows: "prev cur" -> "prev_vi cur_vi"."""
    out = []
    for i, (s, t) in enumerate(rows):
        ctx = rows[max(0, i - n):i] if n > 0 else []
        out.append((sep.join([c[0] for c in ctx] + [s]), sep.join([c[1] for c in ctx] + [t])))
    return out


def split_segments(items: Iterable[tuple[Hashable, Pair | None]]) -> list[list[Pair]]:
    """Group (doc_id, pair-or-None) into runs of consecutive kept pairs within the same document."""
    segs: list[list[Pair]] = []
    cur: list[Pair] = []
    last_doc: object = object()
    for doc, pair in items:
        if pair is None or doc != last_doc:
            if cur:
                segs.append(cur)
            cur = []
        if pair is not None:
            cur.append(pair)
        last_doc = doc
    if cur:
        segs.append(cur)
    return segs


def norm_key(text: str) -> str:
    """Leak-detection key: NFKC, lowercase, without punctuation, symbols and whitespace."""
    s = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in s if not ch.isspace() and unicodedata.category(ch)[0] not in "PS")


def leak_keys(pairs: Iterable[Pair]) -> tuple[set[str], set[str]]:
    """(source keys, target keys) of the test pairs; empty keys are left out."""
    src, tgt = set(), set()
    for s, t in pairs:
        src.add(norm_key(s))
        tgt.add(norm_key(t))
    return src - {""}, tgt - {""}


def touches_test(rows: Iterable[Pair], keys: tuple[set[str], set[str]]) -> bool:
    """True if any row's source or target matches a test sentence (normalised)."""
    ks, kt = keys
    return any(norm_key(s) in ks or norm_key(t) in kt for s, t in rows)


def dedup_norm(pairs: Iterable[Pair]) -> list[Pair]:
    """Keep the first pair per normalised source (for test sets); pairs with an empty key are dropped."""
    seen: set[str] = set()
    out = []
    for s, t in pairs:
        k = norm_key(s)
        if k and k not in seen:
            seen.add(k)
            out.append((s, t))
    return out


def should_sync(step: int, max_steps: int, last_sync_ts: float | None, now: float, min_minutes: float,
                n_synced: int = 0, max_syncs: int | None = None) -> bool:
    """Whether a checkpoint saved at `step` is copied to Drive.

    Never at/after the last step (the final model is saved right after); the first save always syncs;
    later ones only after min_minutes since the previous sync and while fewer than max_syncs were made.
    """
    if max_steps and step >= max_steps:
        return False
    if max_syncs is not None and n_synced >= max_syncs:
        return False
    return last_sync_ts is None or now - last_sync_ts >= min_minutes * 60


def dedup_pairs(pairs: Iterable[Pair]) -> list[Pair]:
    """Drop exact duplicate pairs, keeping the first occurrence and order."""
    seen: set[Pair] = set()
    out = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out
