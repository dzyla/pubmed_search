"""
Per-result links and badges, shared by the Streamlit UI and the REST API.
Pure string logic on a result row (dict or pandas Series) — no network.
"""
import re

_PREPRINT_SERVERS = {"BioRxiv": "biorxiv", "MedRxiv": "medrxiv"}

# PubMed publication types worth surfacing, in display order.
_TYPE_BADGES = [
    ("Retracted Publication", "Retracted"),
    ("Retraction Notice", "Retraction notice"),
    ("Published Erratum", "Erratum"),
    ("Meta-Analysis", "Meta-analysis"),
    ("Systematic Review", "Systematic review"),
    ("Review", "Review"),
    ("Randomized Controlled Trial", "RCT"),
    ("Clinical Trial", "Clinical trial"),
    ("Case Reports", "Case report"),
    ("Preprint", "Preprint"),
]


def _clean(value) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in ("", "none", "nan", "na") else text


def _publication_types(row) -> set:
    return {t.strip() for t in _clean(row.get("pub_type")).split(";") if t.strip()}


def is_retracted(row) -> bool:
    return ("Retracted Publication" in _publication_types(row)
            or _clean(row.get("title")).upper().startswith("RETRACTED"))


def build_links(row) -> list:
    """Returns [(label, url), …]; the first entry is the primary link (or [] if none)."""
    links = []
    doi = _clean(row.get("doi"))
    source = _clean(row.get("source"))

    if "arxiv.org" in doi:
        links.append(("arXiv", doi))
        links.append(("PDF", doi.replace("/abs/", "/pdf/")))
    elif doi:
        links.append(("DOI", f"https://doi.org/{doi}"))

    pmid = _clean(row.get("pmid"))
    if pmid.isdigit():
        links.append(("PubMed", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"))

    server = _PREPRINT_SERVERS.get(source)
    if server and doi.startswith("10.1101/"):
        version = _clean(row.get("version"))
        version = version if version.isdigit() else "1"
        links.append(("PDF", f"https://www.{server}.org/content/{doi}v{version}.full.pdf"))

    published = _clean(row.get("published_doi"))
    if published:
        links.append(("Published version", f"https://doi.org/{published}"))

    preprint = _clean(row.get("preprint_doi"))
    if preprint:
        links.append(("Preprint", f"https://doi.org/{preprint}"))

    return links


def primary_link(row):
    links = build_links(row)
    return links[0][1] if links else None


def badges(row) -> list:
    """Short labels for notable publication types; 'Retracted' always comes first."""
    types = _publication_types(row)
    labels = [label for name, label in _TYPE_BADGES if name in types]
    if is_retracted(row) and "Retracted" not in labels:
        labels.insert(0, "Retracted")
    if _clean(row.get("source")) in _PREPRINT_SERVERS:
        labels.append("Published" if _clean(row.get("published_doi")) else "Preprint")
    if _clean(row.get("preprint_doi")):
        labels.append("Has preprint")
    # "Review" is redundant next to "Systematic review"
    if "Systematic review" in labels and "Review" in labels:
        labels.remove("Review")
    return list(dict.fromkeys(labels))


def year_of(row):
    match = re.search(r"\d{4}", _clean(row.get("date")))
    return int(match.group(0)) if match else None
