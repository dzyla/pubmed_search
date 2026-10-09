"""
Per-result links and badges, shared by the Streamlit UI and the REST API.
Pure string logic on a result row (dict or pandas Series) — no network.
"""
import re

_PREPRINT_SERVERS = {"BioRxiv": "biorxiv", "MedRxiv": "medrxiv"}
_PREPRINT_DOI_PREFIXES = ("10.1101/", "10.64898/")
# Trial statuses worth a label (others, e.g. "Unknown", are left out).
_TRIAL_STATUS_LABELS = {"Recruiting", "Not yet recruiting", "Active, not recruiting",
                        "Enrolling by invitation", "Completed", "Terminated", "Withdrawn", "Suspended"}

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

    grant = _clean(row.get("grant_id"))
    if grant:
        appl = _clean(row.get("appl_id")).split(".")[0]
        url = (f"https://reporter.nih.gov/project-details/{appl}" if appl.isdigit()
               else f"https://reporter.nih.gov/search/results?query={grant}")
        return [("NIH RePORTER", url)]

    nct = _clean(row.get("nct_id"))
    if nct:
        links.append(("ClinicalTrials.gov", f"https://clinicaltrials.gov/study/{nct}"))
        for pmid in [p.strip() for p in _clean(row.get("pmids")).split(";") if p.strip().isdigit()][:2]:
            links.append(("Publication", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"))
        return links

    if "arxiv.org" in doi:
        links.append(("arXiv", doi))
        links.append(("PDF", doi.replace("/abs/", "/pdf/")))
    elif doi:
        links.append(("DOI", f"https://doi.org/{doi}"))

    pmcid = _clean(row.get("pmcid"))
    if pmcid.startswith("PMC"):
        links.append(("Free full text", f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"))

    pmid = _clean(row.get("pmid"))
    if pmid.isdigit():
        links.append(("PubMed", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"))

    server = _PREPRINT_SERVERS.get(source)
    if server and doi.startswith(_PREPRINT_DOI_PREFIXES):
        version = _clean(row.get("version"))
        version = version if version.isdigit() else "1"
        links.append(("PDF", f"https://www.{server}.org/content/{doi}v{version}.full.pdf"))

    published = _clean(row.get("published_doi"))
    if published:
        links.append(("Published version", f"https://doi.org/{published}"))

    published_pmid = _clean(row.get("published_pmid")).split(".")[0]
    if published_pmid.isdigit() and not any(label == "PubMed" for label, _ in links):
        links.append(("PubMed", f"https://pubmed.ncbi.nlm.nih.gov/{published_pmid}/"))

    preprint = _clean(row.get("preprint_doi"))
    if preprint:
        links.append(("Preprint", f"https://doi.org/{preprint}"))

    return links


def primary_link(row):
    links = build_links(row)
    return links[0][1] if links else None


def badges(row) -> list:
    """Short labels for notable publication types; 'Retracted' always comes first."""
    if _clean(row.get("grant_id")):
        return [x for x in (_clean(row.get("activity_code")), _clean(row.get("ic"))) if x]
    if _clean(row.get("nct_id")):
        labels = [p for p in _clean(row.get("trial_phase")).split("/") if p]
        status = _clean(row.get("trial_status"))
        if status in _TRIAL_STATUS_LABELS:
            labels.append(status)
        if str(row.get("has_results")).lower() == "true":
            labels.append("Has results")
        return labels
    types = _publication_types(row)
    labels = [label for name, label in _TYPE_BADGES if name in types]
    if is_retracted(row) and "Retracted" not in labels:
        labels.insert(0, "Retracted")
    if _clean(row.get("source")) in (*_PREPRINT_SERVERS, "Preprints"):
        labels.append("Published" if _clean(row.get("published_doi")) or _clean(row.get("published_pmid"))
                      else "Preprint")
    if _clean(row.get("preprint_doi")):
        labels.append("Has preprint")
    if _clean(row.get("pmcid")).startswith("PMC"):
        labels.append("Free full text")
    # "Review" is redundant next to "Systematic review"
    if "Systematic review" in labels and "Review" in labels:
        labels.remove("Review")
    return list(dict.fromkeys(labels))


def year_of(row):
    match = re.search(r"\d{4}", _clean(row.get("date")))
    return int(match.group(0)) if match else None
