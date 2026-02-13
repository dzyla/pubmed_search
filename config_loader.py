import yaml
import json
import logging
import streamlit as st

LOGGER = logging.getLogger(__name__)

@st.cache_data
def load_configs_and_db_sizes(config_yaml_path="config_mss.yaml"):
    """
    Loads the configuration YAML and calculates database sizes from metadata files.
    """
    try:
        with open(config_yaml_path, "r") as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        LOGGER.error(f"Config file not found at {config_yaml_path}")
        return {}

    pubmed_config = config_data.get("pubmed_config", {})
    biorxiv_config = config_data.get("biorxiv_config", {})
    medrxiv_config = config_data.get("medrxiv_config", {})
    arxiv_config = config_data.get("arxiv_config", {})
    
    configs = [pubmed_config, biorxiv_config, medrxiv_config, arxiv_config]
    
    def load_db_size(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata.get("total_rows", 0)
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