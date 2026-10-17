"""Citation formatting (citations.py) across the sources' author formats."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import citations  # noqa: E402

PUBMED = {"source": "PubMed", "title": "Influenza vaccine effectiveness.", "date": "2022-05-01",
          "authors": "Claudia Maria Trombetta; Otfried Kistner; Emanuele Montomoli",
          "journal": "Vaccines", "doi": "10.3390/vaccines10050714"}
BIORXIV = {"source": "BioRxiv", "title": "Vaccine timing", "date": "2018-09-28",
           "authors": "Matrajt, L.; Halloran, M. E.; de Vries-Idema, J. J.", "journal": "biorxiv",
           "doi": "10.1101/424275"}


@pytest.mark.parametrize("authors, expected", [
    ("Claudia Maria Trombetta; Otfried Kistner", [("Trombetta", "Claudia Maria"), ("Kistner", "Otfried")]),
    ("Matrajt, L.; Halloran, M. E.", [("Matrajt", "L."), ("Halloran", "M. E.")]),
    ("Yidan Pan and Michael W. Deem", [("Pan", "Yidan"), ("Deem", "Michael W.")]),
    ("Baby Jerald, T. R. Gopalakrishnan Nair", [("Jerald", "Baby"), ("Nair", "T. R. Gopalakrishnan")]),
    ("Zhang P, Yang X, Chen B", [("Zhang", "P"), ("Yang", "X"), ("Chen", "B")]),
    ("Matrajt, L.", [("Matrajt", "L.")]),
    ("National Taiwan University Hospital", [("National Taiwan University Hospital", "")]),
    ("N/A", []), (None, []),
])
def test_parse_authors(authors, expected):
    assert citations.parse_authors(authors) == expected


def test_initials():
    assert citations._initials("Claudia Maria") == "C. M."
    assert citations._initials("M. E.") == "M. E."
    assert citations._initials("JA", dots=False, space=False) == "JA"
    assert citations._initials("Jean-Pierre") == "J.-P."


def test_apa_and_vancouver():
    assert citations.format_citation(PUBMED, "APA") == (
        "Trombetta, C. M., Kistner, O., & Montomoli, E. (2022). Influenza vaccine effectiveness. Vaccines. "
        "https://doi.org/10.3390/vaccines10050714")
    assert citations.format_citation(PUBMED, "Vancouver") == (
        "Trombetta CM, Kistner O, Montomoli E. Influenza vaccine effectiveness. Vaccines. 2022. "
        "doi:10.3390/vaccines10050714")
    apa = citations.format_citation(BIORXIV, "APA")
    assert apa.startswith("Matrajt, L., Halloran, M. E., & de Vries-Idema, J. J. (2018). Vaccine timing [Preprint]. bioRxiv.")


@pytest.mark.parametrize("style", citations.STYLES)
def test_every_style_handles_every_source(style):
    rows = [PUBMED, BIORXIV,
            {"source": "arXiv", "title": "T", "authors": "Yidan Pan and Michael W. Deem", "date": "2016-10-14",
             "doi": "https://arxiv.org/abs/1610.03915", "journal": "q-bio.PE"},
            {"source": "ClinicalTrials", "title": "Trial", "authors": "PATH", "date": "2009-05-06",
             "registry_id": "NCT00000001", "journal": "ClinicalTrials.gov", "doi": "N/A"},
            {"source": "OpenAlex", "title": "No authors", "authors": "N/A", "date": None, "doi": None}]
    for row in rows:
        text = citations.format_citation(row, style)
        assert text and "nan" not in text.lower().split() and "*" not in text
    assert "arXiv:1610.03915" in citations.format_citation(rows[2], style) or style == "BibTeX"
