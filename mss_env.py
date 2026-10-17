"""
Machine-specific settings (data locations, contact address) for the ingestion
scripts and tools. Nothing machine-specific is hard-coded in the repository.

Each setting is read from the environment first, then from a KEY=VALUE file
(default ~/.config/mss/paths.env, or the file named by MSS_PATHS_FILE), then
falls back to a neutral default. The file exists so scheduled jobs (which do
not load shell start-up files) pick the settings up without wrappers.

    MSS_DATA_ROOT        root of the data tree (embeddings, parquet, aux_index)
                         default: <repo>/data
    MSS_MAP_WORK         working folder for the map build (map/)
                         default: <MSS_DATA_ROOT>/map_work
    MSS_PIPELINE_CONFIG  config read by the bioRxiv/medRxiv and PubMed download
                         scripts; default: <repo>/config_mss.yaml
    MSS_CONTACT_EMAIL    contact address sent to public APIs (polite pools);
                         default: none (no address is sent)
"""
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_URL = "https://manuscript-search.org"

_file_values = None


def _read_file() -> dict:
    global _file_values
    if _file_values is None:
        _file_values = {}
        path = os.path.expanduser(os.environ.get("MSS_PATHS_FILE", "~/.config/mss/paths.env"))
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    if key.startswith("export "):
                        key = key[len("export "):].strip()
                    _file_values[key] = value.strip().strip('"').strip("'")
        except OSError:
            pass
    return _file_values


def setting(key: str, default: str = "") -> str:
    """Environment variable, else the settings file, else the default."""
    value = os.environ.get(key)
    if value:
        return value
    return _read_file().get(key) or default


def data_root() -> str:
    return setting("MSS_DATA_ROOT", os.path.join(REPO_ROOT, "data"))


def data_path(*parts: str) -> str:
    """A path inside the data root. A trailing "" keeps the trailing slash."""
    return os.path.join(data_root(), *parts)


def map_work() -> str:
    return setting("MSS_MAP_WORK", data_path("map_work"))


def pipeline_config() -> str:
    return setting("MSS_PIPELINE_CONFIG", os.path.join(REPO_ROOT, "config_mss.yaml"))


def contact_email() -> str:
    return setting("MSS_CONTACT_EMAIL", "")


def user_agent() -> str:
    """User-Agent for public APIs; includes the contact address only when one is configured."""
    email = contact_email()
    return f"ManuscriptSearch/2.0 ({PROJECT_URL}; mailto:{email})" if email else f"ManuscriptSearch/2.0 ({PROJECT_URL})"
