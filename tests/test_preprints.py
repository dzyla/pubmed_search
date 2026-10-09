"""Preprints (Europe PMC + Crossref): record conversion, licence rule, version and
cross-feed dedup, in-place updates. Fixtures are trimmed live API responses; no network."""
import copy
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "update_database"))
import preprints_update as pp  # noqa: E402
from conftest import EMB_BYTES  # noqa: E402

with open(os.path.join(HERE, "fixtures", "preprints_sample.json")) as f:
    SAMPLE = json.load(f)
RS_V2, PSY, CHEM_EPMC = SAMPLE["europepmc"]
CHEM_V2, OSF_V1, NO_LICENCE = SAMPLE["crossref"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def _blocked(*a, **k):
        raise AssertionError("network access in tests")
    monkeypatch.setattr(pp.requests, "get", _blocked)
    monkeypatch.setattr(pp.requests, "post", _blocked)
    monkeypatch.setattr(pp.time, "sleep", lambda s: None)


def _fake_embed(df):
    return np.zeros((len(df), EMB_BYTES), dtype=np.uint8) + 7


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def test_europepmc_multi_version_record():
    row = pp.epmc_record_to_row(RS_V2, "cc by (server terms)")
    assert set(row) == set(pp.COLUMNS)
    assert row["record_id"] == "PPR1129403"          # v1's PPR id identifies the work
    assert row["doi"] == "10.21203/rs.3.rs-8044369/v2"
    assert row["version"] == "2"
    assert row["date"] == "2025-12-02"               # first version's posting date
    assert row["journal"] == "Research Square"
    assert row["license"] == "cc by (server terms)"
    assert row["authors"].startswith("Xiong L, Wang M") and not row["authors"].endswith(".")
    assert row["source_feed"] == "europepmc"


def test_europepmc_published_version_and_doi_version():
    row = pp.epmc_record_to_row(PSY, "cc by")
    assert row["published_pmid"] == "42154474"
    assert row["published_doi"] == ""                # resolved later, in batches
    assert row["version"] == "2"                     # no versionList: taken from '..._v2'
    assert row["journal"] == "PsyArXiv"
    chem = pp.epmc_record_to_row(CHEM_EPMC, "cc by-nc-nd")
    assert chem["published_pmid"] == "34596386" and chem["record_id"] == "PPR356990"


def test_crossref_chemrxiv_v2():
    row = pp.crossref_item_to_row(CHEM_V2)
    assert set(row) == set(pp.COLUMNS)
    assert row["record_id"] == "10.26434/chemrxiv.5318188"
    assert row["version"] == "2"
    assert row["license"] == "cc by-nc-nd"
    assert row["published_doi"] == "10.1021/acscentsci.7b00488"
    assert row["journal"] == "ChemRxiv"
    assert row["date"] == "2017-08-19"
    assert row["authors"].startswith("Roach J, Sasano Y")
    assert "<" not in row["abstract"] and "jats" not in row["abstract"]
    assert row["abstract"].startswith("Salvinorin A (SalA) is a plant metabolite")


def test_crossref_osf_record_and_licence_rule():
    row = pp.crossref_item_to_row(OSF_V1)
    assert row["record_id"] == "10.31235/osf.io/xs7qr" and row["version"] == "1"
    assert row["journal"] == "SocArXiv" and row["license"] == "cc by"
    assert row["published_doi"] == "10.1177/2378023117733903"
    assert pp.crossref_item_to_row(NO_LICENCE) is None          # no CC licence -> dropped
    terms_only = dict(CHEM_V2, license=[{"URL": "https://chemrxiv.org/engage/chemrxiv/legal-information"}])
    assert pp.crossref_item_to_row(terms_only) is None


@pytest.mark.parametrize("url, label", [
    ("https://creativecommons.org/licenses/by/4.0/legalcode", "cc by"),
    ("http://creativecommons.org/licenses/by-nc-nd/4.0/", "cc by-nc-nd"),
    ("https://creativecommons.org/publicdomain/zero/1.0/legalcode", "cc0"),
    ("http://opensource.org/licenses/AFL-3.0", ""),
    ("https://escholarship.org", ""),
])
def test_licence_label(url, label):
    assert pp._license_label(url) == label


def test_clean_text():
    jats = "<jats:title>Abstract</jats:title><jats:p>CO<jats:sub>2</jats:sub> rises</jats:p><jats:p>p < 0.05.</jats:p>"
    assert pp.clean_text(jats) == "CO2 rises p < 0.05."
    escaped = "&lt;div&gt;First&lt;br&gt;second&lt;/div&gt; &amp; more"
    assert pp.clean_text(escaped) == "First second & more"
    assert pp.clean_text("Abstract: Text  here\n") == "Text here"


@pytest.mark.parametrize("doi, base, version", [
    ("10.26434/chemrxiv.14778147.v1", "10.26434/chemrxiv.14778147", 1),
    ("10.26434/chemrxiv-2023-abc12-v3", "10.26434/chemrxiv-2023-abc12", 3),
    ("10.21203/rs.3.rs-8044369/v2", "10.21203/rs.3.rs-8044369", 2),
    ("10.31235/osf.io/XS7QR_v1", "10.31235/osf.io/xs7qr", 1),
    ("10.1002/essoar.10509894.2", "10.1002/essoar.10509894", 2),
    ("10.31223/x5w163", "10.31223/x5w163", 1),
])
def test_doi_base(doi, base, version):
    assert pp.doi_base(doi) == (base, version)


def test_server_for_doi():
    assert pp.server_for_doi("10.31234/osf.io/abc_v1", "PsyArXiv") == "PsyArXiv"
    assert pp.server_for_doi("10.31219/osf.io/abc", "Open Science Framework") == "OSF Preprints"
    assert pp.server_for_doi("10.31219/osf.io/abc", "Law Archive") == "LawArXiv"
    assert pp.server_for_doi("10.31223/x5w163", "Earth Sciences") == "EarthArXiv"   # subject, not server
    assert pp.server_for_doi("10.22541/essoar.1234.5/v1") == "ESS Open Archive"
    assert pp.server_for_doi("10.22541/au.1234/v1") == "Authorea"


def test_europepmc_queries_exclude_indexed_servers_and_use_update_date():
    queries = pp.epmc_queries("2026-10-01")
    assert len(queries) == len(pp.EPMC_LICENSES) + 1
    for q, _ in queries:
        for server in ("bioRxiv", "medRxiv", "arXiv", "SSRN"):
            assert f'NOT PUBLISHER:"{server}"' in q
        assert q.endswith("AND UPDATE_DATE:[2026-10-01 TO *]")
    q, lic = queries[-1]
    assert lic == pp.SERVER_TERMS_LICENSE and "NOT LICENSE:*" in q and 'PUBLISHER:"Research Square"' in q
    assert 'PUBLISHER:"Authorea' not in q             # Authorea's default licence is "no reuse"
    assert "UPDATE_DATE" not in pp.epmc_queries()[0][0]


# ---------------------------------------------------------------------------
# Fetch loops (fake HTTP)
# ---------------------------------------------------------------------------

def test_fetch_europepmc_follows_cursor_and_limit(monkeypatch):
    calls = []

    def fake_get(url, params, attempts=8, post=False):
        calls.append(dict(params))
        if params["cursorMark"] == "*":
            return {"hitCount": 3, "nextCursorMark": "c1",
                    "resultList": {"result": [RS_V2, dict(PSY, abstractText="short")]}}
        if params["cursorMark"] == "c1":
            return {"nextCursorMark": "c2", "resultList": {"result": [CHEM_EPMC]}}
        return {"nextCursorMark": "c2", "resultList": {"result": []}}

    monkeypatch.setattr(pp, "_get", fake_get)
    monkeypatch.setattr(pp, "EPMC_LICENSES", ("cc by",))
    monkeypatch.setattr(pp, "CC_BY_SERVERS", ("Research Square",))
    rows = list(pp.fetch_europepmc())
    # 2 queries x (RS_V2 + CHEM; PSY dropped: abstract too short)
    assert [r["record_id"] for r in rows] == ["PPR1129403", "PPR356990"] * 2
    assert [r["license"] for r in rows] == ["cc by", "cc by", pp.SERVER_TERMS_LICENSE, pp.SERVER_TERMS_LICENSE]
    assert calls[0]["pageSize"] == 1000 and calls[0]["resultType"] == "core"
    limited = list(pp.fetch_europepmc(limit=1))
    assert len(limited) == 2


def test_fetch_crossref_filters_and_essoar_doi_filter(monkeypatch):
    seen_params = []
    essoar = dict(CHEM_V2, DOI="10.22541/essoar.170000000.12345678/v1")
    authorea = dict(CHEM_V2, DOI="10.22541/au.170000000.1/v1")

    def fake_get(url, params, attempts=8, post=False):
        seen_params.append(dict(params))
        if params["cursor"] == "*":
            return {"message": {"total-results": 3, "next-cursor": "n1", "items": [essoar, authorea, NO_LICENCE]}}
        return {"message": {"next-cursor": "n2", "items": []}}

    monkeypatch.setattr(pp, "_get", fake_get)
    rows = list(pp.fetch_crossref("10.22541", since="2026-10-01"))
    assert [r["doi"] for r in rows] == ["10.22541/essoar.170000000.12345678/v1"]
    assert rows[0]["journal"] == "ESS Open Archive"
    p = seen_params[0]
    assert p["filter"] == "type:posted-content,has-abstract:true,prefix:10.22541,from-index-date:2026-10-01"
    assert p["mailto"] == pp.MAILTO and p["rows"] == 1000


def test_resolve_pmid_dois_batches(monkeypatch):
    batches = []

    def fake_get(url, params, attempts=8, post=False):
        assert post and url == pp.EPMC_POST_API
        ids = re.findall(r"EXT_ID:(\d+)", params["query"])
        batches.append(len(ids))
        return {"resultList": {"result": [{"id": i, "doi": f"10.1/{i}"} for i in ids if i != "5"]}}

    monkeypatch.setattr(pp, "_get", fake_get)
    monkeypatch.setattr(pp, "RESOLVE_BATCH", 3)
    found = pp.resolve_pmid_dois([str(i) for i in range(1, 8)] + ["1", ""])
    assert batches == [3, 3, 1]
    assert found["1"] == "10.1/1" and "5" not in found and len(found) == 6


# ---------------------------------------------------------------------------
# Merging, chunks, incremental updates
# ---------------------------------------------------------------------------

def test_merge_rows_rules():
    epmc = pp.epmc_record_to_row(CHEM_EPMC, "cc by-nc-nd")
    cr = dict(pp.crossref_item_to_row(CHEM_V2), date="2010-01-01")
    assert pp.merge_rows(epmc, cr) is None                       # Europe PMC wins
    merged = pp.merge_rows(cr, epmc)
    assert merged["source_feed"] == "europepmc" and merged["date"] == "2010-01-01"
    assert merged["published_doi"] == cr["published_doi"]        # known link kept
    v1 = dict(cr, version="1", doi="10.26434/chemrxiv.5318188.v1")
    assert pp.merge_rows(cr, v1) is None                         # older version ignored
    assert pp.merge_rows(v1, cr)["doi"].endswith(".v2")


def _setup_dirs(tmp_path):
    df_dir, emb_dir = tmp_path / "preprints_df", tmp_path / "preprints_embed"
    df_dir.mkdir()
    emb_dir.mkdir()
    return str(df_dir), str(emb_dir)


def test_collector_dedups_versions_and_feeds_then_updates_in_place(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "embed", _fake_embed)
    monkeypatch.setattr(pp, "resolve_pmid_dois", lambda pmids: {p: f"10.9999/{p}" for p in pmids})
    df_dir, emb_dir = _setup_dirs(tmp_path)

    # Run 1: Europe PMC (two versions of one work + ChemRxiv), then Crossref.
    col = pp.Collector(pp.existing_index(df_dir), df_dir, emb_dir, run_id=1)
    rs_v1 = copy.deepcopy(RS_V2)
    rs_v1.update(id="PPR1129403", doi="10.21203/rs.3.rs-8044369/v1", versionNumber=1, title="Old title")
    for rec in (rs_v1, RS_V2, CHEM_EPMC):
        col.add(pp.epmc_record_to_row(rec, "cc by"))
    col.flush()
    chem_cr_v1 = dict(CHEM_V2, DOI="10.26434/chemrxiv.14778147.v2")      # same work as CHEM_EPMC
    for item in (chem_cr_v1, CHEM_V2, OSF_V1):
        col.add(pp.crossref_item_to_row(item))
    assert col.finish() == (0, 0)

    files = sorted(os.listdir(df_dir))
    assert files == ["preprints_1_chunk_0.parquet", "preprints_1_chunk_1.parquet"]
    df = pd.concat(pd.read_parquet(os.path.join(df_dir, f)) for f in files).reset_index(drop=True)
    assert df["record_id"].is_unique and len(df) == 4
    rs = df[df.record_id == "PPR1129403"].iloc[0]
    assert rs["version"] == "2" and rs["doi"].endswith("/v2") and rs["title"] != "Old title"
    chem = df[df.record_id == "PPR356990"].iloc[0]
    assert chem["source_feed"] == "europepmc" and chem["published_doi"] == "10.9999/34596386"
    for f in files:
        assert len(pd.read_parquet(os.path.join(df_dir, f))) == len(np.load(os.path.join(emb_dir, f[:-8] + ".npy")))

    # Run 2 (incremental): RS v3 with new text; metadata-only change for OSF; older ChemRxiv crossref.
    for f in files:   # mark stored vectors so re-embedding is visible
        np.save(os.path.join(emb_dir, f[:-8] + ".npy"), np.zeros((len(pd.read_parquet(os.path.join(df_dir, f))),
                                                                   EMB_BYTES), dtype=np.uint8))
    col = pp.Collector(pp.existing_index(df_dir), df_dir, emb_dir, run_id=2)
    rs_v3 = copy.deepcopy(RS_V2)
    rs_v3.update(id="PPR2000000", doi="10.21203/rs.3.rs-8044369/v3", versionNumber=3,
                 abstractText="A rewritten abstract for the third version of this preprint.")
    rs_v3["versionList"]["version"].append({"id": "PPR2000000", "versionNumber": 3, "firstPublishDate": "2026-10-09"})
    col.add(pp.epmc_record_to_row(rs_v3, "cc by (server terms)"))
    osf_published = copy.deepcopy(OSF_V1)
    osf_published["relation"] = {"is-preprint-of": [{"id-type": "doi", "id": "10.1/new"}]}
    col.add(pp.crossref_item_to_row(osf_published))
    col.add(pp.crossref_item_to_row(dict(CHEM_V2, DOI="10.26434/chemrxiv.5318188.v1")))   # older: ignored
    assert col.finish() == (2, 1)                 # RS text change + OSF metadata
    assert col.chunks == 0

    df0 = pd.read_parquet(os.path.join(df_dir, "preprints_1_chunk_0.parquet"))
    bits0 = np.load(os.path.join(emb_dir, "preprints_1_chunk_0.npy"))
    i = int(np.flatnonzero(df0.record_id == "PPR1129403")[0])
    assert df0.at[i, "version"] == "3" and df0.at[i, "date"] == "2025-12-02"
    assert (bits0[i] == 7).all() and (np.delete(bits0, i, axis=0) == 0).all()
    df1 = pd.read_parquet(os.path.join(df_dir, "preprints_1_chunk_1.parquet"))
    bits1 = np.load(os.path.join(emb_dir, "preprints_1_chunk_1.npy"))
    osf = df1[df1.record_id == "10.31235/osf.io/xs7qr"].iloc[0]
    assert osf["published_doi"] == "10.1/new"
    assert df1[df1.record_id == "10.26434/chemrxiv.5318188"].iloc[0]["version"] == "2"
    assert (bits1 == 0).all()                                  # metadata-only: not re-embedded
    assert [p for p in os.listdir(df_dir) + os.listdir(emb_dir) if "tmp" in p] == []


def test_existing_index_maps_ids_and_doi_bases(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "embed", _fake_embed)
    df_dir, emb_dir = _setup_dirs(tmp_path)
    row = pp.crossref_item_to_row(CHEM_V2)
    path = pp.write_chunk(pd.DataFrame([row], columns=pp.COLUMNS), df_dir, emb_dir, "preprints_1_chunk_0")
    index = pp.existing_index(df_dir)
    assert index[row["record_id"]] == (path, row["record_id"])
    assert index["doi:10.26434/chemrxiv.5318188"] == (path, row["record_id"])
