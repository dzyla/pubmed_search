import pandas as pd

import ui_components


def _results():
    return pd.DataFrame([
        {"rank": 1, "title": "A paper", "authors": "J R Elkinton; R B Singer", "journal": "J Clin Invest",
         "date": "1955-11-01", "source": "PubMed", "doi": "10.1172/JCI103221", "pmid": "13271551",
         "abstract": "Text.", "score": 0.81, "citations": 12},
        {"rank": 2, "title": "A preprint", "authors": "Edward Flach;Santiago Schnell;", "journal": "biorxiv",
         "date": "2013-11-16", "source": "BioRxiv", "doi": "10.1101/000547", "pmid": None,
         "abstract": "More.", "score": 0.80, "citations": 0},
    ])


def test_ris_records():
    ris = ui_components.generate_ris(_results())
    records = ris.strip().split("\n\n")
    assert len(records) == 2
    assert records[0].startswith("TY  - JOUR") and records[1].startswith("TY  - UNPB")
    assert "AU  - J R Elkinton\nAU  - R B Singer" in records[0]
    assert "AU  - Santiago Schnell\nPY" in records[1].replace("\nJO  - biorxiv", "")
    assert "DO  - 10.1172/JCI103221" in records[0]
    assert all(r.rstrip().endswith("ER  -") for r in records)


def test_csv_has_links():
    csv = ui_components.results_csv(_results())
    header = csv.splitlines()[0]
    assert header.startswith("rank,title,authors")
    assert "https://doi.org/10.1172/JCI103221" in csv
