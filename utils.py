import time
import logging
from contextlib import contextmanager
import doi
import json
import glob
LOGGER = logging.getLogger(__name__)

@contextmanager
def log_time(task_name: str, status_placeholder=None):
    start = time.perf_counter()
    msg = f"Starting {task_name}..."
    LOGGER.info(msg)
    if status_placeholder:
        status_placeholder.info(msg)
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        completion_message = f"Completed {task_name} in {duration:.2f} seconds."
        LOGGER.info(completion_message)
        if status_placeholder:
            status_placeholder.info(completion_message)


def get_clean_doi(doi_str):
    if not isinstance(doi_str, str):
        return ""
    if 'arxiv.org' in doi_str:
        return doi_str
    try:
        doi_clean = doi.get_clean_doi(doi_str)
        return doi_clean
    except Exception:
        return doi_str


def report_dates_from_metadata(metadata_dict: dict) -> str:
    folder = metadata_dict.get("embeddings_directory", "")
    json_files = glob.glob(f"{folder}/*.json")
    if not json_files:
        logging.warning(f"No JSON files found in {folder}")
        return "N/A"
    with open(json_files[0], "r") as f:
        return json.load(f).get("last_fetch_date", "N/A")
