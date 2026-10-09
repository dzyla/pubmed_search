import json
import logging

import streamlit as st
import yaml

LOGGER = logging.getLogger(__name__)

# Order matters: search_logic._SOURCE_NAMES zips over this list.
SOURCE_KEYS = ("pubmed_config", "biorxiv_config", "medrxiv_config", "arxiv_config")


def read_source_configs(config_yaml_path: str) -> dict:
    """
    Reads config_mss.yaml and returns {source_key: config_dict} for all four
    sources (missing stanzas become {}). Raises on a missing or invalid file.
    No Streamlit dependency — shared by the UI and the REST API.
    """
    with open(config_yaml_path) as f:
        data = yaml.safe_load(f) or {}
    return {key: data.get(key) or {} for key in SOURCE_KEYS}


def _db_size(metadata_path: str) -> int:
    try:
        with open(metadata_path) as f:
            return json.load(f).get("total_rows", 0)
    except Exception:
        return 0


@st.cache_data(ttl=3600)
def load_configs_and_db_sizes(config_yaml_path="./config_mss.yaml"):
    """
    Loads the configuration YAML and calculates database sizes from metadata files.
    """
    try:
        source_configs = read_source_configs(config_yaml_path)
    except FileNotFoundError:
        LOGGER.error(f"Config file not found at {config_yaml_path}")
        source_configs = {key: {} for key in SOURCE_KEYS}

    result = dict(source_configs)
    result["configs"] = [source_configs[key] for key in SOURCE_KEYS]
    for key in SOURCE_KEYS:
        size_key = key.replace("_config", "_db_size")
        result[size_key] = _db_size(source_configs[key].get("metadata_path", ""))
    return result
