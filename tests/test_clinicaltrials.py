"""ClinicalTrials.gov: record conversion, in-place updates, links/labels, search integration."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "update_database"))
import clinicaltrials_update as ct  # noqa: E402
import paper_links  # noqa: E402
import search_logic  # noqa: E402
from conftest import EMB_BYTES  # noqa: E402

STUDY = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT00711035", "briefTitle": "Matched CTL for virus infection",
                                 "acronym": "CHALLAH"},
        "statusModule": {"overallStatus": "ACTIVE_NOT_RECRUITING", "startDateStruct": {"date": "2008-11"},
                         "studyFirstPostDateStruct": {"date": "2008-07-08"},
                         "lastUpdatePostDateStruct": {"date": "2016-04-26"}},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Baylor College of Medicine"}},
        "descriptionModule": {"briefSummary": "This trial\n evaluates   CTLs."},
        "conditionsModule": {"conditions": ["Adenovirus Infection", "EBV Infection"]},
        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE1", "PHASE2"]},
        "armsInterventionsModule": {"interventions": [{"name": "CTLs"}]},
        "referencesModule": {"references": [{"pmid": "23610374"}, {"citation": "no pmid"}]},
    },
    "hasResults": True,
}


def test_study_to_row():
    row = ct.study_to_row(STUDY)
    assert row["nct_id"] == "NCT00711035"
    assert row["title"] == "Matched CTL for virus infection (CHALLAH)"
    assert row["abstract"] == "This trial evaluates CTLs."
    assert row["date"] == "2008-07-08" and row["start_date"] == "2008-11-01"
    assert row["trial_status"] == "Active, not recruiting"
    assert row["trial_phase"] == "Phase 1/Phase 2"
    assert row["study_type"] == "Interventional"
    assert row["pmids"] == "23610374" and row["has_results"] is True
    assert set(row) == set(ct.COLUMNS)


def test_trial_links_and_labels():
    row = {**ct.study_to_row(STUDY), "source": "ClinicalTrials"}
    assert paper_links.build_links(row) == [
        ("ClinicalTrials.gov", "https://clinicaltrials.gov/study/NCT00711035"),
        ("Publication", "https://pubmed.ncbi.nlm.nih.gov/23610374/"),
    ]
    assert paper_links.badges(row) == ["Phase 1", "Phase 2", "Active, not recruiting", "Has results"]


def _fake_embed(df):
    return np.zeros((len(df), EMB_BYTES), dtype=np.uint8) + 7


def test_incremental_update_rewrites_metadata_and_reembeds_only_changed_text(tmp_path, monkeypatch):
    monkeypatch.setattr(ct, "embed", _fake_embed)
    df_dir, emb_dir = tmp_path / "df", tmp_path / "emb"
    df_dir.mkdir()
    emb_dir.mkdir()
    rows = [dict(ct.study_to_row(STUDY), nct_id=f"NCT{i:08d}") for i in range(3)]
    ct.write_chunk(pd.DataFrame(rows), str(df_dir), str(emb_dir), "ctgov_1_chunk_0")
    np.save(emb_dir / "ctgov_1_chunk_0.npy", np.zeros((3, EMB_BYTES), dtype=np.uint8))

    index = ct.existing_index(str(df_dir))
    status_only = dict(rows[0], trial_status="Completed")
    new_text = dict(rows[2], abstract="Rewritten summary.")
    n = ct.apply_updates({r["nct_id"]: (index[r["nct_id"]], r) for r in (status_only, new_text)},
                         str(df_dir), str(emb_dir))

    df = pd.read_parquet(df_dir / "ctgov_1_chunk_0.parquet")
    bits = np.load(emb_dir / "ctgov_1_chunk_0.npy")
    assert n == 1
    assert df.loc[0, "trial_status"] == "Completed" and df.loc[2, "abstract"] == "Rewritten summary."
    assert (bits[0] == 0).all() and (bits[2] == 7).all()     # only the changed text re-embedded
    assert [p for p in os.listdir(df_dir) if "tmp" in p] == []


def test_trials_are_searchable_and_deduped_by_nct(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    monkeypatch.setattr(ct, "embed", lambda df: rng.integers(0, 256, (len(df), EMB_BYTES), dtype=np.uint8))
    df_dir, emb_dir = tmp_path / "clinicaltrials_df", tmp_path / "clinicaltrials_embed"
    df_dir.mkdir()
    emb_dir.mkdir()
    rows = [dict(ct.study_to_row(STUDY), nct_id=f"NCT{i:08d}", title=f"Trial {i}",
                 abstract="x" * 100, date=f"20{10 + i % 10}-01-01") for i in range(200)]
    ct.write_chunk(pd.DataFrame(rows), str(df_dir), str(emb_dir), "ctgov_1_chunk_0")
    config = {"embeddings_directory": str(emb_dir), "npy_files_pattern": "*.npy",
              "chunk_dir": str(tmp_path / "chunks") + "/", "metadata_path": str(tmp_path / "meta.json"),
              "data_folder": str(df_dir), "chunk_size_bytes": 1 << 20}

    query = np.zeros((1, EMB_BYTES), dtype=np.uint8)
    df = search_logic._run_search(query, [{}, {}, {}, {}, config], 10, "2015-01-01", None, True, None)

    assert len(df) == 10
    assert (df["source"] == "ClinicalTrials").all()
    assert df["nct_id"].str.startswith("NCT").all() and df["nct_id"].is_unique
    assert (pd.to_datetime(df["date"]) >= "2015-01-01").all()
    assert df["trial_phase"].eq("Phase 1/Phase 2").all()
