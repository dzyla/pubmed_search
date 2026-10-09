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


def test_grant_links_and_labels():
    row = {"source": "Grants", "grant_id": "UM1HL172720", "appl_id": "11251654", "ic": "HL",
           "activity_code": "UM1", "doi": ""}
    assert pl.build_links(row) == [("NIH RePORTER", "https://reporter.nih.gov/project-details/11251654")]
    assert pl.badges(row) == ["UM1", "HL"]


def test_other_preprints_and_new_biorxiv_prefix():
    row = {"source": "Preprints", "doi": "10.21203/rs.3.rs-123/v1", "published_pmid": "41234567",
           "published_doi": ""}
    assert ("PubMed", "https://pubmed.ncbi.nlm.nih.gov/41234567/") in pl.build_links(row)
    assert pl.badges(row) == ["Published"]
    new = {"source": "BioRxiv", "doi": "10.64898/2026.01.02.123456", "version": "1"}
    assert dict(pl.build_links(new))["PDF"].startswith("https://www.biorxiv.org/content/10.64898/")


def test_free_full_text_link_and_label():
    row = {"source": "PubMed", "doi": "10.1/x", "pmid": "123", "pmcid": "PMC456"}
    assert pl.build_links(row)[:3] == [("DOI", "https://doi.org/10.1/x"),
                                       ("Free full text", "https://pmc.ncbi.nlm.nih.gov/articles/PMC456/"),
                                       ("PubMed", "https://pubmed.ncbi.nlm.nih.gov/123/")]
    assert "Free full text" in pl.badges(row)
