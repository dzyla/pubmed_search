import yaml

# Order matters: search_logic._SOURCE_NAMES zips over this list.
SOURCE_KEYS = ("pubmed_config", "biorxiv_config", "medrxiv_config", "arxiv_config", "clinicaltrials_config")


def read_source_configs(config_yaml_path: str) -> dict:
    """
    Reads config_mss.yaml and returns {source_key: config_dict} for all four
    sources (missing stanzas become {}). Raises on a missing or invalid file.
    No Streamlit dependency — shared by the UI and the REST API.
    """
    with open(config_yaml_path) as f:
        data = yaml.safe_load(f) or {}
    return {key: data.get(key) or {} for key in SOURCE_KEYS}
