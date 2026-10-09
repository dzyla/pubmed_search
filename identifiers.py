"""
Identifier-like tokens (gene/protein symbols, variants, compound codes, trial
and accession ids) for exact-term matching. Shared by the index builder
(update_database/build_aux_indexes.py) and the search backend, so both
tokenize identically.

Semantic embeddings blur such tokens ("TMEM175", "rs429358", "BMS-986165",
"NCT04368728" are rarely recovered by meaning alone), while ordinary words are
handled well — so only identifier-like tokens are indexed.
"""
import hashlib
import re

import numpy as np

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9]")
MAX_LEN = 40


def _is_identifier(word: str, allow_caps: bool) -> bool:
    if len(word) > MAX_LEN or word.isdigit():
        return False
    has_digit = any(c.isdigit() for c in word)
    has_alpha = any(c.isalpha() for c in word)
    if has_digit and has_alpha:                  # TMEM175, rs429358, G12C, BMS-986165, NCT04368728
        return not re.fullmatch(r"\d+(st|nd|rd|th|s)", word)   # 1st, 2nd, 1990s
    upper = sum(c.isupper() for c in word)
    # Capitalised symbols without digits: KRAS, EGFR, mTORC, SARS-CoV.
    return allow_caps and has_alpha and upper >= 2 and 3 <= len(word) <= 12


def normalize(word: str) -> str:
    return word.replace("-", "").lower()


def identifier_tokens(text: str) -> set:
    """Normalized identifier tokens in text (lowercase, hyphens removed)."""
    text = str(text or "")
    # In ALL-CAPS text (common in old PubMed titles) every word looks like a
    # symbol; only letter+digit tokens are trusted there. Text counts as
    # all-caps when it has no ordinary lowercase word.
    allow_caps = re.search(r"\b[a-z]{3,}\b", text) is not None
    return {normalize(w) for w in _TOKEN.findall(text) if _is_identifier(w, allow_caps)}


def token_hash(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")


def hash_tokens(tokens) -> np.ndarray:
    return np.fromiter((token_hash(t) for t in tokens), dtype=np.uint64)
