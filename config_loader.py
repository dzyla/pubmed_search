import yaml

# Order matters: search_logic._SOURCE_NAMES zips over this list.
SOURCE_KEYS = ("pubmed_config", "biorxiv_config", "medrxiv_config", "arxiv_config", "clinicaltrials_config",
               "preprints_config", "grants_config", "openalex_config")


SOURCE_NAMES = ("PubMed", "BioRxiv", "MedRxiv", "arXiv", "ClinicalTrials", "Preprints", "Grants", "OpenAlex")
# Searched when the caller does not choose sources. Grants are opt-in: funding
# records should not appear among papers unless asked for.
DEFAULT_SOURCES = ("PubMed", "BioRxiv", "MedRxiv", "arXiv", "ClinicalTrials", "Preprints", "OpenAlex")


def read_source_configs(config_yaml_path: str) -> dict:
    """
    Reads config_mss.yaml and returns {source_key: config_dict} for every
    source (missing stanzas become {}). Each non-empty config also gets its
    display name ("source_name") and the shared "aux_index_root" (top-level
    key in the YAML, optional). Raises on a missing or invalid file.
    """
    with open(config_yaml_path) as f:
        data = yaml.safe_load(f) or {}
    configs = {}
    for key, name in zip(SOURCE_KEYS, SOURCE_NAMES):
        cfg = dict(data.get(key) or {})
        if cfg:
            cfg.setdefault("source_name", name)
            if data.get("aux_index_root"):
                cfg.setdefault("aux_index_root", data["aux_index_root"])
        configs[key] = cfg
    return configs
