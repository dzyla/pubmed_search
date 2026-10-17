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
    tokens = set()
    for word in _TOKEN.findall(text):
        # "TMEM175-mediated" must also match "TMEM175": index the whole token
        # and each hyphen-separated part that is an identifier on its own.
        for cand in ([word] + word.split("-") if "-" in word else [word]):
            if cand and _is_identifier(cand, allow_caps):
                tokens.add(normalize(cand))
    return tokens


def token_hash(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")


def hash_tokens(tokens) -> np.ndarray:
    return np.fromiter((token_hash(t) for t in tokens), dtype=np.uint64)


# ---------------------------------------------------------------------------
# Paper identifiers (DOI, PMID, arXiv id): a reader's library -> indexed rows
# ---------------------------------------------------------------------------
_DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>{}|\\^`]+", re.IGNORECASE)
_PMID_LABELED = re.compile(r"\bpmid\s*[-:=]?\s*[{\"']?\s*(\d{1,9})\b", re.IGNORECASE)
_PUBMED_URL = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{1,9})")
_ARXIV = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|\barxiv:\s*|10\.48550/arxiv\.)"
                    r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?", re.IGNORECASE)
_BARE_PMID_LINE = re.compile(r"^\s*(\d{5,9})\s*[,;]?\s*$", re.MULTILINE)


def normalize_doi(doi) -> str:
    """Lowercase DOI without resolver prefix or trailing punctuation; '' if not a DOI."""
    text = str(doi or "").strip()
    text = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", text, flags=re.IGNORECASE)
    text = text.rstrip(".,;:)]}").lower()
    return text if text.startswith("10.") and "/" in text else ""


def id_key(kind: str, value) -> str:
    """Canonical lookup key: 'doi:10.1/x', 'pmid:123', 'arxiv:2401.01234'; '' if invalid."""
    value = str(value or "").strip()
    if kind == "doi":
        value = normalize_doi(value)
    elif kind == "pmid":
        value = value.split(".")[0]
        value = str(int(value)) if value.isdigit() and int(value) > 0 else ""
    elif kind == "arxiv":
        value = re.sub(r"v\d+$", "", value.lower())
    return f"{kind}:{value}" if value else ""


def parse_paper_ids(text: str, limit: int = 500) -> list:
    """
    DOIs, PMIDs and arXiv ids found in free text: a BibTeX, RIS or PubMed (.nbib)
    export, a CSV, or a pasted list. Returns unique keys (see id_key) in order of
    appearance. Bare numbers count as PMIDs only alone on a line, so years and page
    numbers in a reference list are not mistaken for PMIDs (and need 5+ digits, not a year).
    """
    text = str(text or "")
    found = []                       # (position, key)
    for m in _ARXIV.finditer(text):
        found.append((m.start(), id_key("arxiv", m.group(1))))
    for m in _DOI.finditer(text):
        if not m.group(0).lower().startswith("10.48550/"):
            found.append((m.start(), id_key("doi", m.group(0))))
    for rx in (_PMID_LABELED, _PUBMED_URL, _BARE_PMID_LINE):
        for m in rx.finditer(text):
            found.append((m.start(), id_key("pmid", m.group(1))))
    keys = [k for _, k in sorted(found) if k]
    return list(dict.fromkeys(keys))[:limit]


def row_id_keys(source: str, doi, pmid=None) -> list:
    """Lookup keys of one indexed row (arXiv rows store their abs URL as 'doi')."""
    keys = []
    doi = str(doi or "")
    if source == "arXiv":
        m = _ARXIV.search(doi)
        if m:
            keys += [id_key("arxiv", m.group(1)), id_key("doi", f"10.48550/arxiv.{m.group(1)}")]
    else:
        keys.append(id_key("doi", doi))
    keys.append(id_key("pmid", pmid))
    return [k for k in keys if k]
