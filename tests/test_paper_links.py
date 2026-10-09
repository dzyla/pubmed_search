import paper_links as pl


def test_pubmed_without_doi_still_gets_a_pubmed_link():
    row = {"source": "PubMed", "doi": "", "pmid": "13271551"}
    assert pl.build_links(row) == [("PubMed", "https://pubmed.ncbi.nlm.nih.gov/13271551/")]


def test_pubmed_with_doi():
    row = {"source": "PubMed", "doi": "10.1172/JCI103221", "pmid": "13271551"}
    assert [label for label, _ in pl.build_links(row)] == ["DOI", "PubMed"]
    assert pl.primary_link(row) == "https://doi.org/10.1172/JCI103221"


def test_biorxiv_preprint_pdf_and_published_version():
    row = {"source": "BioRxiv", "doi": "10.1101/000547", "version": "2",
           "published_doi": "10.1038/xyz", "pmid": None}
    links = dict(pl.build_links(row))
    assert links["PDF"] == "https://www.biorxiv.org/content/10.1101/000547v2.full.pdf"
    assert links["Published version"] == "https://doi.org/10.1038/xyz"
    assert pl.badges(row) == ["Published"]


def test_arxiv_links():
    row = {"source": "arXiv", "doi": "https://arxiv.org/abs/2606.04390"}
    assert pl.build_links(row) == [("arXiv", "https://arxiv.org/abs/2606.04390"),
                                   ("PDF", "https://arxiv.org/pdf/2606.04390")]


def test_badges_from_publication_types():
    row = {"source": "PubMed", "pub_type": "Journal Article; Systematic Review; Review; Meta-Analysis"}
    assert pl.badges(row) == ["Meta-analysis", "Systematic review"]
    assert pl.badges({"source": "PubMed", "pub_type": "Journal Article; Retracted Publication"}) == ["Retracted"]
    assert pl.badges({"source": "PubMed", "title": "RETRACTED: Something"}) == ["Retracted"]
    assert pl.badges({"source": "MedRxiv", "published_doi": "NA"}) == ["Preprint"]


def test_nan_values_are_ignored():
    row = {"source": "PubMed", "doi": float("nan"), "pmid": float("nan"), "pub_type": float("nan")}
    assert pl.build_links(row) == []
    assert pl.badges(row) == []
    assert pl.year_of({"date": "2021-05-01"}) == 2021
