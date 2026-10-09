"""NIH RePORTER grants: ExPORTER/API parsing, one row per core project, API slicing,
in-place updates. Fixtures are real ExPORTER rows / API records (trimmed); no network."""
import copy
import datetime
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "update_database"))
import nih_grants_update as ng  # noqa: E402
from conftest import EMB_BYTES  # noqa: E402

CORE, CENTER = "R01AI116059", "P01AG052350"
with open(os.path.join(FIX, "nih_api_sample.json")) as f:
    API_SAMPLE = json.load(f)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def _blocked(*a, **k):
        raise AssertionError("network access in tests")
    monkeypatch.setattr(ng.requests, "request", _blocked)
    monkeypatch.setattr(ng.time, "sleep", lambda s: None)


def _zip(name: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        with open(os.path.join(FIX, name), "rb") as f:
            zf.writestr(name.replace(".csv", "") + ".csv", f.read())
    return buf.getvalue()


def _fake_embed(df):
    return np.zeros((len(df), EMB_BYTES), dtype=np.uint8) + 7


def _apps(fy: int) -> pd.DataFrame:
    apps = ng.exporter_projects(_zip(f"nih_prj_fy{fy}.csv"))
    abstracts = ng.exporter_abstracts(_zip(f"nih_abs_fy{fy}.csv"))
    apps["abstract"] = apps["appl_id"].map(abstracts).fillna("")
    return apps


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, clean", [
    ("ABSTRACT\nHost-adapted strains", "Host-adapted strains"),
    ("OVERALL – PROJECT SUMMARY/ABSTRACT\nThis is a program", "This is a program"),
    ("¿     DESCRIPTION (provided by applicant): We study", "We study"),
    ("  [unreadable] DESCRIPTION (provided by applicant): The goal", "The goal"),
    ("PROJECT SUMMARY/ABSTRACT The goal", "The goal"),
    ("Modified Project Summary/Abstract Section Daily PrEP", "Daily PrEP"),
    ("Overall, this project aims", "Overall, this project aims"),
    ("Summary of prior work shows", "Summary of prior work shows"),
    (None, ""),
])
def test_clean_abstract(raw, clean):
    assert ng.clean_abstract(raw) == clean


def test_format_pis_and_decode_mixed():
    assert ng.format_pis("NATION, DANIEL A;TOGA, ARTHUR W (contact)") == "Daniel A Nation; Arthur W Toga"
    assert ng.format_pis("Mu A; John Smith") == "Mu A; John Smith"
    raw = "café ".encode("utf-8") + b"\x92quoted\x92"            # UTF-8 with stray CP1252
    assert ng.decode_mixed(raw) == "café ’quoted’"


def test_exporter_parsing_old_and_new_formats():
    new = _apps(2025)
    assert set(ng.APP_COLUMNS) <= set(new.columns)
    row = new[new.appl_id == "10975859"].iloc[0]
    assert row["grant_id"] == CORE and row["fiscal_year"] == 2025 and row["application_type"] == "5"
    assert row["notice_date"] == "2024-11-05" and row["start_date"] == "2014-11-01"
    assert row["pis"] == "Denise M Monack" and row["total_cost"] == 430311.0
    assert row["abstract"].startswith("Host-adapted strains of Salmonella")
    assert (new.loc[new.grant_id == CENTER, "subproject_id"] != "").sum() == 2
    old = _apps(2015)                                                  # CP1252 file, other columns
    assert old.iloc[0]["grant_id"] == CORE and old.iloc[0]["abstract"].startswith("We study Salmonella")


def test_api_record_to_app_matches_exporter():
    app = ng.api_record_to_app(API_SAMPLE["results"][0])
    assert app["grant_id"] == CORE and app["appl_id"] == "10975859" and app["fiscal_year"] == 2025
    assert app["pis"] == "Denise M Monack" and app["ic"] == "AI"
    assert app["abstract"].startswith("Host-adapted strains")
    std = ng.standardize(pd.DataFrame([app], columns=ng.APP_COLUMNS))
    assert std.iloc[0]["notice_date"] == "2024-11-05" and std.iloc[0]["end_date"] == "2025-10-31"


# ---------------------------------------------------------------------------
# One row per core project
# ---------------------------------------------------------------------------

def test_reduce_and_combine_keep_newest_parent_with_span():
    best = ng.combine(ng.reduce_apps(_apps(2025)), ng.reduce_apps(_apps(2015)))
    rows = ng.to_rows(best).set_index("grant_id")
    assert sorted(rows.index) == [CENTER, CORE]                       # sub-projects dropped
    r = rows.loc[CORE]
    assert r["appl_id"] == "10975859"                                  # type 5 beats the type-3 supplement
    assert r["fiscal_year"] == 2025 and r["date"] == "2024-11-05"
    assert r["start_date"] == "2014-11-01" and r["end_date"] == "2025-10-31"
    assert r["journal"] == "NIH RePORTER (NIAID)" and r["doi"] == ""
    assert r["abstract"].startswith("Host-adapted")                    # newest FY's text
    c = rows.loc[CENTER]
    assert c["appl_id"] == "10827875" and c["abstract"].startswith("This is a continuing")
    assert c["authors"] == "Daniel A Nation; Arthur W Toga" and c["journal"] == "NIH RePORTER (NIA)"
    assert list(ng.to_rows(best).columns) == ng.COLUMNS


def test_date_fallbacks():
    apps = _apps(2025)
    apps.loc[apps.appl_id == "10975859", ["notice_date", "budget_start"]] = ""
    apps.loc[apps.grant_id == CENTER, ["notice_date", "budget_start", "start_date"]] = ""
    rows = ng.to_rows(ng.reduce_apps(apps[apps.application_type != "3"])).set_index("grant_id")
    assert rows.loc[CORE, "date"] == "2014-11-01"                      # project start
    assert rows.loc[CENTER, "date"] == "2024-10-01"                    # start of FY2025


class _FakeDate(datetime.date):
    @classmethod
    def today(cls):
        return cls(2026, 3, 1)


def test_full_download_api_year_then_exporter(monkeypatch):
    newer = copy.deepcopy(API_SAMPLE["results"][0])
    newer.update(appl_id=12000001, fiscal_year=2026, project_num="5R01AI116059-11",
                 award_notice_date="2025-11-10T00:00:00", project_end_date="2026-10-31T00:00:00",
                 abstract_text="PROJECT SUMMARY\nRenewed aims on persistent Salmonella in macrophages.")
    calls = []

    def fake_fetch_api(criteria, from_date, to_date, limit=None):
        calls.append(criteria)
        return iter([ng.api_record_to_app(newer)])

    monkeypatch.setattr(ng, "exporter_years", lambda: [2015, 2025])
    monkeypatch.setattr(ng, "download_exporter",
                        lambda kind, fy: _zip(f"nih_{'prj' if kind == 'EXPPRJ' else 'abs'}_fy{fy}.csv"))
    monkeypatch.setattr(ng, "fetch_api", fake_fetch_api)
    monkeypatch.setattr(ng, "date", _FakeDate)                         # FY2026 is current, no file yet
    rows = ng.full_download().set_index("grant_id")
    assert calls == [{"fiscal_years": [2026]}]
    r = rows.loc[CORE]
    assert r["fiscal_year"] == 2026 and r["appl_id"] == "12000001" and r["date"] == "2025-11-10"
    assert r["abstract"] == "Renewed aims on persistent Salmonella in macrophages."
    assert r["start_date"] == "2014-11-01" and r["end_date"] == "2026-10-31"
    assert len(rows) == 2


def test_fetch_api_splits_large_ranges_and_respects_offset_cap(monkeypatch):
    # 40,000 records spread evenly over 2026-01-01..2026-01-08 (5,000 per day).
    requests_seen = []

    def fake_search(criteria, offset, limit):
        rng = criteria["date_added"]
        d0, d1 = ng.date.fromisoformat(rng["from_date"]), ng.date.fromisoformat(rng["to_date"])
        total = 5000 * max(0, (min(d1, ng.date(2026, 1, 8)) - max(d0, ng.date(2026, 1, 1))).days + 1)
        requests_seen.append((rng["from_date"], rng["to_date"], offset, limit))
        n = max(0, min(limit, total - offset))
        res = [dict(API_SAMPLE["results"][0], appl_id=f"{rng['from_date']}-{offset + i}") for i in range(n)]
        return {"meta": {"total": total}, "results": res}

    monkeypatch.setattr(ng, "_api_search", fake_search)
    apps = list(ng.fetch_api({}, "2026-01-01", "2026-01-08"))
    assert len(apps) == 40_000 and len({a["appl_id"] for a in apps}) == 40_000
    assert all(off + lim <= ng.API_MAX_RESULTS for _, _, off, lim in requests_seen)
    assert len(list(ng.fetch_api({}, "2026-01-01", "2026-01-08", limit=1234))) == 1234


# ---------------------------------------------------------------------------
# Files + incremental
# ---------------------------------------------------------------------------

def test_write_chunks_then_incremental_update(tmp_path, monkeypatch):
    monkeypatch.setattr(ng, "embed", _fake_embed)
    df_dir, emb_dir = tmp_path / "grants_df", tmp_path / "grants_embed"
    df_dir.mkdir()
    emb_dir.mkdir()
    rows = ng.to_rows(ng.combine(ng.reduce_apps(_apps(2025)), ng.reduce_apps(_apps(2015))))
    assert ng.write_chunks(rows, str(df_dir), str(emb_dir), run_id=1) == 1
    stem = "grants_1_chunk_0"
    np.save(emb_dir / f"{stem}.npy", np.zeros((2, EMB_BYTES), dtype=np.uint8))
    known = ng.existing_index(str(df_dir))
    assert set(known) == {CORE, CENTER}

    # FY2026 award with new text for CORE; older FY2015 record and a metadata-only
    # change (later end date) for the centre grant.
    new_core = copy.deepcopy(API_SAMPLE["results"][0])
    new_core.update(appl_id=12000001, fiscal_year=2026, project_num="5R01AI116059-11",
                    award_notice_date="2025-11-10T00:00:00", abstract_text="New aims for year eleven of this R01 on Salmonella persistence.")
    old_core = ng.api_record_to_app(API_SAMPLE["results"][0]) | {"fiscal_year": 2015, "appl_id": "8838665",
                                                                 "abstract": "Old text that must not win."}
    center = _apps(2025)
    center = center[(center.grant_id == CENTER) & (center.subproject_id == "")].copy()
    center["end_date"] = "2029-03-31"
    apps = pd.concat([ng.standardize(pd.DataFrame([ng.api_record_to_app(new_core), old_core],
                                                  columns=ng.APP_COLUMNS)), center], ignore_index=True)
    updated, reembedded = ng.apply_updates(ng.reduce_apps(apps), known, str(df_dir), str(emb_dir))

    assert (updated, reembedded) == (2, 1)
    df = pd.read_parquet(df_dir / f"{stem}.parquet").set_index("grant_id")
    bits = np.load(emb_dir / f"{stem}.npy")
    assert df.loc[CORE, "abstract"] == "New aims for year eleven of this R01 on Salmonella persistence." and df.loc[CORE, "fiscal_year"] == 2026
    assert df.loc[CORE, "date"] == "2025-11-10" and df.loc[CORE, "start_date"] == "2014-11-01"
    assert df.loc[CENTER, "end_date"] == "2029-03-31"
    i_core = list(df.index).index(CORE)
    assert (bits[i_core] == 7).all() and (bits[1 - i_core] == 0).all()   # only CORE re-embedded
    assert [p for p in os.listdir(df_dir) + os.listdir(emb_dir) if "tmp" in p] == []
    # Re-applying the same applications changes nothing.
    assert ng.apply_updates(ng.reduce_apps(apps), ng.existing_index(str(df_dir)),
                            str(df_dir), str(emb_dir)) == (0, 0)
