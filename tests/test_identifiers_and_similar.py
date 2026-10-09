"""Identifier tokens, the aux-index builder, exact-term search, superseded rows and find-similar."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "update_database"))
import build_aux_indexes as bai  # noqa: E402
import search_logic  # noqa: E402
from conftest import EMB_BYTES  # noqa: E402
from identifiers import identifier_tokens  # noqa: E402


def test_identifier_tokens():
    toks = identifier_tokens("TMEM175 and rs429358 with BMS-986165 in NCT04368728; KRAS G12C; IL-6 vs mTORC1")
    assert {"tmem175", "rs429358", "bms986165", "nct04368728", "kras", "g12c", "il6", "mtorc1"} <= toks
    assert not identifier_tokens("the results were significant in 1990s and 2nd trial 2019")
    # ALL-CAPS titles: plain words are not symbols, letter+digit tokens still are
    assert identifier_tokens("EFFECT OF P53 ON THE LIVER") == {"p53"}
    assert {"tmem175", "tmem175mediated"} <= identifier_tokens("a TMEM175-mediated current")


@pytest.fixture
def pubmed_like(tmp_path):
    """3 PubMed-like files with a planted identifier and a revised PMID."""
    rng = np.random.default_rng(1)
    base = tmp_path / "base"
    emb, data = base / "pm_embed", base / "pm_data"
    emb.mkdir(parents=True)
    data.mkdir()
    for f in range(3):
        codes = rng.integers(0, 256, (100, EMB_BYTES), dtype=np.uint8)
        if f == 1:
            # The planted paper is related to the (all-zero) test query, but not enough
            # to reach the top on semantics alone: 175 of 384 bits set -> score 0.544.
            bits = np.zeros(EMB_BYTES * 8, dtype=np.uint8)
            bits[:175] = 1
            codes[7] = np.packbits(bits)
        np.save(emb / f"pubmed26n{f:04d}.npy", codes)
        abstracts = ["y" * 100] * 100
        pmids = [str(1000 * f + i) for i in range(100)]
        if f == 1:
            abstracts[7] = "We characterise TMEM175, a lysosomal potassium channel. " + "z" * 60
        if f == 2:
            pmids[9] = "4"                       # newer version of PMID 4 (first seen in file 0, row 4)
        pd.DataFrame({"title": [f"Paper {f}-{i}" for i in range(100)], "abstract": abstracts,
                      "doi": [f"10.1/p.{f}.{i}" for i in range(100)], "date": ["2021-01-01"] * 100,
                      "pmid": pmids}).to_parquet(data / f"pubmed26n{f:04d}.parquet")
    config = {"embeddings_directory": str(emb), "npy_files_pattern": "*.npy", "chunk_dir": str(tmp_path / "chunks") + "/",
              "metadata_path": str(tmp_path / "meta.json"), "data_folder": str(data), "chunk_size_bytes": 48 * 120,
              "source_name": "PubMed", "aux_index_root": str(base / "aux_index")}
    return base, config


def test_builder_exact_terms_and_superseded(pubmed_like, monkeypatch):
    base, config = pubmed_like
    monkeypatch.setitem(bai.SOURCES, "PubMed", ("pm_embed", "pm_data", None, ["title", "abstract"]))
    bai.build_source("PubMed", str(base), workers=2)

    searcher = search_logic.get_or_create_searcher(config)
    assert searcher.superseded_count == 1
    g_old = 0 * 100 + 4
    assert searcher.is_superseded(np.array([g_old, g_old + 1])).tolist() == [True, False]

    # A query vector unrelated to the planted paper: semantic ranking alone would miss it.
    query = np.zeros((1, EMB_BYTES), dtype=np.uint8)
    plain = search_logic._run_search(query, [config], 5, None, None, False, None)
    assert not plain["title"].eq("Paper 1-7").any()
    hybrid = search_logic._run_search(query, [config], 5, None, None, False, None, query_text="TMEM175 function")
    top = hybrid.iloc[0]
    assert top["title"] == "Paper 1-7" and list(top["matched_terms"]) == ["tmem175"]

    # The superseded copy (file 0, row 4) never comes back, even with a huge result list.
    everything = search_logic._run_search(query, [config], 300, None, None, False, None)
    assert "Paper 0-4" not in set(everything["title"]) and "Paper 2-9" in set(everything["title"])


def test_similar_search_excludes_examples(pubmed_like):
    _, config = pubmed_like
    config = dict(config, aux_index_root=None)
    df, seeds, packed = search_logic.similar_search(["PubMed:10", "PubMed:120"], [config], [config], 8)
    assert len(df) == 8 and packed.shape == (1, EMB_BYTES)
    assert not set(df["corpus_id"]) & {10, 120}
    assert [s["title"] for s in seeds] == ["Paper 0-10", "Paper 1-20"]
    with pytest.raises(search_logic.UnknownReference):
        search_logic.similar_search(["PubMed:999999"], [config], [config], 5)
    with pytest.raises(search_logic.UnknownReference):
        search_logic.similar_search(["Scopus:1"], [config], [config], 5)


def test_pmc_lookup(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "update_database"))
    import pmc_links_update
    from aux_index import PmcIndex

    csv = tmp_path / "PMC-ids.csv"
    pd.DataFrame({"PMCID": ["PMC10", "PMC20", "PMC30", "PMC40"], "PMID": ["300", "100", "", "200"],
                  "Release Date": ["live", "live", "live", "2999-01-01"]}).to_csv(csv, index=False)
    meta = pmc_links_update.build(str(csv), str(tmp_path / "aux" / "PMC"))
    assert meta["articles"] == 2                      # no PMID / still embargoed are left out
    idx = PmcIndex(str(tmp_path / "aux"))
    assert idx.refresh() and idx.lookup(["100", 300, "200", None, "x"]) == ["PMC20", "PMC10", None, None, None]
