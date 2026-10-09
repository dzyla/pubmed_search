"""
Cheap English-language check for abstracts (no dependencies): the share of
very common English function words. English abstracts score ~0.25-0.40;
Portuguese, Spanish, Indonesian, ... score near 0. The embedding model is
English-only, so non-English records would only add noise to the index.
"""
import re

_WORD = re.compile(r"[a-z]+")
_EN = frozenset("""the of and in to is was for with that were on by as are from this be at which an
these we or have has not been it our their between than its also into both after during can may more
such there but other using used however while when where who each our they them been within""".split())


def english_score(text: str) -> float:
    words = _WORD.findall(str(text or "").lower())
    return sum(w in _EN for w in words) / len(words) if len(words) >= 8 else 1.0   # too short to judge


def is_english(text: str, threshold: float = 0.05) -> bool:
    return english_score(text) >= threshold
