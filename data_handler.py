import os
import re
import json
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import streamlit as st
import ast
from crossref.restful import Works

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOGGER = logging.getLogger(__name__)

# =============================================================================
# Caching & API Helpers
# =============================================================================

def create_session():
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

GLOBAL_SESSION = create_session()

@st.cache_data(ttl=3600, show_spinner=False)
def get_citation_count_cached(doi_str):
    try:
        works = Works()
        paper_data = works.doi(doi_str)
        if paper_data:
            return paper_data.get("is-referenced-by-count", 0)
        return 0
    except:
        return 0

@st.cache_data(ttl=3600, show_spinner=False)
def get_link_info_cached(row_dict):
    source = str(row_dict.get("source", "")).lower()
    pmid = row_dict.get("version")
    doi_val = row_dict.get("doi")

    full_text = None
    if source == "pubmed" and pmid:
        url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
        try:
            r = GLOBAL_SESSION.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if "records" in data and "pmcid" in data["records"][0]:
                    full_text = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{data['records'][0]['pmcid']}/"
        except:
            pass
    elif "arxiv" in str(doi_val):
        full_text = doi_val

    return full_text

@st.cache_data(show_spinner=False)
def get_references(doi_str):
    works = Works()
    try:
        paper_data = works.doi(doi_str)
        references = paper_data.get("reference", [])
        formatted_refs = []
        for ref in references:
            if isinstance(ref, dict):
                if "unstructured" in ref:
                    formatted_refs.append(ref["unstructured"])
                else:
                    formatted_refs.append(", ".join(f"{k}: {v}" for k, v in ref.items()))
            else:
                formatted_refs.append(str(ref))
        return formatted_refs
    except Exception as e:
        LOGGER.error(f"Error fetching references for {doi_str}: {e}")
        return []

def get_citation_count(doi_str):
    works = Works()
    try:
        paper_data = works.doi(doi_str)
        if paper_data:
            return paper_data.get("is-referenced-by-count", 0)
        return 0
    except Exception as e:
        LOGGER.error(f"Error fetching citation count for {doi_str}: {e}")
        return 0

def get_clean_doi(doi_str):
    if 'arxiv.org' in doi_str:
        return doi_str
    try:
        import doi # Importing here to avoid circular dependencies if any, though likely fine at top
        doi_clean = doi.get_clean_doi(doi_str)
        return doi_clean
    except Exception as e:
        LOGGER.error(f"Error cleaning DOI {doi_str}: {e}")
        return doi_str

def get_full_text_link(row):
    # This function seems redundant with get_link_info_cached but is used in precalculate_full_text_links_parallel
    # We can reuse logic or keep it.
    source = row.get("source", "None").lower()
    if source == "pubmed":
        pmid = row.get("version")
        doi = row.get("doi")
        if pmid:
            url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    records = data.get("records", [])
                    if records:
                        record = records[0]
                        pmcid = record.get("pmcid")
                        if pmcid:
                            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
            except Exception as e:
                pass
        return None
    else:
        doi = row.get("doi")
        if doi:
            if "arxiv.org" in doi:
                return doi
            else:
                return f"https://doi.org/{doi}"
        return None

def precalculate_full_text_links_parallel(df):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(get_full_text_link, [row for _, row in df.iterrows()]))
    df["full_text_link"] = results
    return df

# =============================================================================
# Data Reformatting
# =============================================================================

def reformat_biorxiv_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "server" in df.columns:
        df.rename(columns={"server": "journal"}, inplace=True)
    required_cols = ["doi", "title", "authors", "date", "version", "type", "journal", "abstract", "score", "embedding"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    df = df[required_cols]
    return df

def reformat_arxiv_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return pd.DataFrame(columns=["doi", "title", "authors", "date", "version", "type", "journal", "abstract", "score", "embedding"])

    if "doi" not in df.columns and "id" in df.columns:
        df["doi"] = df["id"].apply(lambda x: f"https://arxiv.org/abs/{x}")
    else:
        df["doi"] = df["doi"].fillna("").astype(str).str.strip()
        df.loc[df["doi"] == "", "doi"] = df["id"].apply(lambda x: f"https://arxiv.org/abs/{x}")
    df["date"] = pd.to_datetime(df["update_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["journal"] = df["journal-ref"] if "journal-ref" in df.columns else None
    def get_version_count(versions):
        try:
            if pd.isnull(versions):
                return None
            if isinstance(versions, list):
                return len(versions)
            versions_list = ast.literal_eval(versions)
            return len(versions_list)
        except Exception:
            return None
    if "versions" in df.columns:
        df["version"] = df["versions"].apply(get_version_count)
    else:
        df["version"] = None
    df["type"] = "preprint"
    for col in ["score", "embedding"]:
        if col not in df.columns:
            df[col] = None
    required_cols = ["doi", "title", "authors", "date", "version", "type", "journal", "abstract", "score", "embedding"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    df = df[required_cols]
    return df

def report_dates_from_metadata(metadata_file: str) -> dict:
    def extract_dates(source_stem: str) -> list:
        return re.findall(r"(\d{6})", source_stem)

    def format_date(date_str: str) -> str:
        if len(date_str) != 6:
            return date_str
        return f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:]}"

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    result = {"second_date_from_second_part": None, "second_date_from_last_part": None}

    if not metadata.get("chunks"):
        return result
    chunk = metadata["chunks"][0]
    parts = chunk.get("parts", [])

    parts = sorted(parts, key=lambda x: x.get("source_stem", ""))

    if parts and isinstance(parts[-1], dict):
        source_stem_last = parts[-1].get("source_stem", "")
        dates_last = extract_dates(source_stem_last)
        if len(dates_last) >= 2:
            result["second_date_from_last_part"] = format_date(dates_last[1])
        else:
            pass # warning
    else:
        pass # warning
    return result

def load_configs_and_db_sizes(config_yaml_path="config_mss.yaml"):
    import yaml
    with open(config_yaml_path, "r") as f:
        config_data = yaml.safe_load(f)
    pubmed_config = config_data.get("pubmed_config", {})
    biorxiv_config = config_data.get("biorxiv_config", {})
    medrxiv_config = config_data.get("medrxiv_config", {})
    arxiv_config = config_data.get("arxiv_config", {})
    configs = [pubmed_config, biorxiv_config, medrxiv_config, arxiv_config]
    def load_db_size(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata.get("total_rows", "N/A")
        except Exception:
            return 0
    pubmed_db_size = load_db_size(pubmed_config.get("metadata_path", ""))
    biorxiv_db_size = load_db_size(biorxiv_config.get("metadata_path", ""))
    medrxiv_db_size = load_db_size(medrxiv_config.get("metadata_path", ""))
    arxiv_db_size = load_db_size(arxiv_config.get("metadata_path", ""))
    return {
        "pubmed_config": pubmed_config,
        "biorxiv_config": biorxiv_config,
        "medrxiv_config": medrxiv_config,
        "arxiv_config": arxiv_config,
        "configs": configs,
        "pubmed_db_size": pubmed_db_size,
        "biorxiv_db_size": biorxiv_db_size,
        "medrxiv_db_size": medrxiv_db_size,
        "arxiv_db_size": arxiv_db_size
    }
