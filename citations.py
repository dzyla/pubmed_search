"""
Formatted citations for search results ("Copy citation"; style chosen in Settings).

The sources write authors differently, so names are first normalized to
(family, given) pairs:
    PubMed          "Claudia Maria Trombetta; Otfried Kistner"      given first, ';'
    bioRxiv/medRxiv "Matrajt, L.; Halloran, M. E."                  family first, ';'
    arXiv           "Yidan Pan and Michael W. Deem", "A B, C D"     given first, ',' / 'and'
    Preprints       "Zhang P, Yang X"                               family + initials, ','
    Trials/Grants   sponsor or PI name
Volume, issue and pages are not in the index, so citations end at the journal (and DOI).
"""
import re

import paper_links
import ui_components

STYLES = ["APA", "Vancouver", "MLA", "Chicago", "Harvard", "Nature", "BibTeX"]
DEFAULT_STYLE = "APA"

_PREPRINT_SERVERS = {"biorxiv": "bioRxiv", "medrxiv": "medRxiv"}
_ORG_WORDS = re.compile(r"\b(university|hospital|institute|inc|ltd|llc|foundation|center|centre|"
                        r"council|ministry|company|corporation|group|consortium|path|nih|who)\b", re.I)


def _clean(value) -> str:
    s = str(value or "").strip()
    return "" if s.lower() in ("nan", "none", "n/a", "") else s


def parse_authors(authors) -> list:
    """[(family, given)]; an organization comes back as (name, "")."""
    text = _clean(authors).replace("[", "").replace("]", "").replace("'", "")
    if not text:
        return []
    if ";" in text:
        parts = text.split(";")
    elif re.search(r"\band\b", text):
        parts = re.split(r",\s*|\s+and\s+", text)
    else:
        parts = text.split(",")
        # "Matrajt, L." without ';' (a single family-first author)
        if len(parts) == 2 and re.fullmatch(r"\s*(?:[A-Z]\.?\s*-?)+", parts[1]):
            parts = [text]
    out = []
    for p in (x.strip() for x in parts):
        if not p:
            continue
        if _ORG_WORDS.search(p) and len(p.split()) != 2:
            out.append((p, ""))
        elif "," in p:                                     # "Family, Given"
            family, given = (x.strip() for x in p.split(",", 1))
            out.append((family, given))
        else:
            words = p.split()
            if len(words) >= 2 and re.fullmatch(r"[A-Z]{1,3}", words[-1]):   # "Zhang P", "Smith JA"
                out.append((" ".join(words[:-1]), " ".join(words[-1])))
            elif len(words) == 1:
                out.append((words[0], ""))
            else:                                          # "Given Middle Family"
                out.append((words[-1], " ".join(words[:-1])))
    return out


def _initials(given: str, dots: bool = True, space: bool = True) -> str:
    """'Claudia Maria' -> 'C. M.'; 'J. A.' / 'JA' -> 'J. A.'; 'Jean-Pierre' -> 'J.-P.' (dots/space optional)."""
    if not given:
        return ""
    words = list(given) if re.fullmatch(r"[A-Z]{1,3}", given) else [w for w in re.split(r"[\s.]+", given) if w]
    dot = "." if dots else ""
    out = ["-".join(part[0].upper() + dot for part in w.split("-") if part) for w in words]
    return (" " if space else "").join(out)


def _year(row) -> str:
    return str(paper_links.year_of(row) or "n.d.")


def _venue(row) -> str:
    source = str(row.get("source") or "")
    journal = _clean(row.get("journal"))
    if source in ("BioRxiv", "MedRxiv"):
        return _PREPRINT_SERVERS.get(journal.lower(), journal or source)
    if source == "arXiv":
        doi = _clean(row.get("doi"))
        m = re.search(r"arxiv\.org/abs/(\S+)", doi)
        return f"arXiv:{m.group(1)}" if m else "arXiv"
    if source == "ClinicalTrials":
        return f"ClinicalTrials.gov, {row.get('registry_id') or row.get('nct_id') or ''}".rstrip(", ")
    if source == "Grants":
        return f"NIH grant {row.get('registry_id') or row.get('grant_id') or ''}".strip()
    return journal


def _doi_url(row) -> str:
    doi = paper_links.linkable_doi(row.get("doi"))
    if doi:
        return f"https://doi.org/{doi}"
    url = row.get("url")
    return url if isinstance(url, str) and url.startswith("http") else (paper_links.primary_link(row) or "")


def _title(row) -> str:
    return _clean(row.get("title")).rstrip(".") or "Untitled"


def _is_preprint(row) -> bool:
    return row.get("source") in ("BioRxiv", "MedRxiv", "arXiv", "Preprints")


def _join(names: list, last_sep: str, sep: str = ", ") -> str:
    if len(names) <= 1:
        return "".join(names)
    return sep.join(names[:-1]) + last_sep + names[-1]


def format_citation(row, style: str = DEFAULT_STYLE) -> str:
    row = dict(row)
    authors = parse_authors(row.get("authors"))
    title, year, venue, link = _title(row), _year(row), _venue(row), _doi_url(row)
    pre = " [Preprint]" if _is_preprint(row) and style in ("APA", "Vancouver", "Harvard") else ""

    if style == "BibTeX":
        return ui_components.generate_bibtex(__import__("pandas").DataFrame([row])).strip()

    if style == "APA":
        names = [f"{f}, {_initials(g)}".rstrip(", ") for f, g in authors]
        if len(names) > 20:
            names = names[:19] + ["…", names[-1]]
            a = ", ".join(names[:-1]) + " " + names[-1]
        else:
            a = _join(names, ", & " if len(names) > 2 else " & ")
        return " ".join(x for x in (f"{a} ({year}).", f"{title}{pre}.", f"{venue}." if venue else "", link) if x)

    if style == "Vancouver":
        names = [f"{f} {_initials(g, dots=False, space=False)}".strip() for f, g in authors]
        a = ", ".join(names[:6]) + (", et al." if len(names) > 6 else ".") if names else ""
        doi = paper_links.linkable_doi(row.get("doi"))
        tail = f" doi:{doi}" if doi else (f" Available from: {link}" if link else "")
        return " ".join(x for x in (a, f"{title}{pre}.", f"{venue}. {year}." if venue else f"{year}.") if x) + tail

    if style == "MLA":
        if not authors:
            a = ""
        elif len(authors) == 1:
            a = ", ".join(x for x in authors[0] if x) + "."
        elif len(authors) == 2:
            a = f"{authors[0][0]}, {authors[0][1]}, and {authors[1][1]} {authors[1][0]}.".replace(" ,", ",")
        else:
            a = f"{authors[0][0]}, {authors[0][1]}, et al.".replace(", ,", ",")
        return " ".join(x for x in (a, f"“{title}.”", f"{venue}," if venue else "", f"{year},", link) if x).rstrip(",") + "."

    if style == "Chicago":
        if not authors:
            a = ""
        else:
            first = ", ".join(x for x in authors[0] if x)
            rest = [f"{g} {f}".strip() for f, g in authors[1:10]]
            a = _join([first] + rest, ", and " if len(authors) > 2 else " and ") + \
                (", et al." if len(authors) > 10 else ".")
        return " ".join(x for x in (a, f"{year}.", f"“{title}.”", f"{venue}." if venue else "", link) if x)

    if style == "Harvard":
        names = [f"{f}, {_initials(g, space=False)}".rstrip(", ") for f, g in authors]
        a = (names[0] + " et al.") if len(names) > 3 else _join(names, " and ")
        return " ".join(x for x in (f"{a} ({year})", f"‘{title}’{pre},", f"{venue}." if venue else "",
                                    f"Available at: {link}." if link else "") if x)

    if style == "Nature":
        names = [f"{f}, {_initials(g)}".rstrip(", ") for f, g in authors]
        a = (names[0] + " et al.") if len(names) > 5 else _join(names, " & ")
        return " ".join(x for x in (a, f"{title}.", f"{venue}" if venue else "", f"({year}).", link) if x)

    raise ValueError(f"unknown citation style {style!r}")
