#!/usr/bin/env python3
"""
Unified Update Pipeline for bioRxiv and medRxiv

This script performs the following steps for each source:
  1. Determines the last update date by checking both the combined database file
     and the update (Parquet) directory.
  2. Fetches new data (in JSON blocks) from the appropriate API.
  3. Processes the JSON files into Parquet files.
  4. Checks for unprocessed Parquet files (i.e. missing embeddings) and generates embeddings.
  5. Cleans up the JSON update folder and uploads updated directories to Dropbox.

Paths and API parameters are read from a configuration file (config_mss.yaml).
"""

import os
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
import requests
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import dropbox
import streamlit as st

###############################
# Load Configuration from YAML
###############################
CONFIG_FILE = "config_mss.yaml"
try:
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    print("Error loading configuration file:", e)
    config = {}

# Extract configurations for bioRxiv and medRxiv.
biorxiv_config = config.get("biorxiv_config", {})
medrxiv_config = config.get("medrxiv_config", {})

###############################
# Retry Decorator for API calls
###############################
def retry_on_exception(exception, retries=5, delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for _ in range(retries):
                try:
                    return func(*args, **kwargs)
                except exception as e:
                    last_exception = e
                    print(f"Retrying due to: {str(e)}")
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

###############################
# API Fetch Functions (shared by both sources)
###############################
@retry_on_exception(requests.exceptions.ConnectionError)
def fetch_and_save_data_block(endpoint, server, block_start, block_end, save_directory, format='json', api_type="biorxiv"):
    """
    Fetch one block of data from the API and save it as a JSON file.

    For bioRxiv:
      URL format: https://api.biorxiv.org/{endpoint}/{server}/{block_interval}/{cursor}/{format}
    For medRxiv:
      URL format: https://api.medrxiv.org/details/{server}/{block_interval}/{cursor}/{format}
    """
    if api_type == "biorxiv":
        base_url = f"https://api.biorxiv.org/{endpoint}/{server}/"
    elif api_type == "medrxiv":
        base_url = f"https://api.medrxiv.org/details/{server}/"
    else:
        raise ValueError("Unsupported api_type")

    block_interval = f"{block_start.strftime('%Y-%m-%d')}/{block_end.strftime('%Y-%m-%d')}"
    block_data = []
    cursor = 0
    continue_fetching = True

    while continue_fetching:
        url = f"{base_url}{block_interval}/{cursor}/{format}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch data for block {block_interval} at cursor {cursor}. HTTP Status: {response.status_code}")
            break

        data = response.json()
        fetched_papers = len(data.get('collection', []))
        if fetched_papers > 0:
            block_data.extend(data['collection'])
            cursor += fetched_papers
            print(f"Fetched {fetched_papers} papers for block {block_interval}. Total fetched: {cursor}.")
        else:
            continue_fetching = False

    if block_data:
        save_data_block(block_data, block_start, block_end, endpoint, save_directory)

def save_data_block(block_data, start_date, end_date, endpoint, save_directory):
    """
    Save a block of API data to a JSON file.
    """
    os.makedirs(save_directory, exist_ok=True)
    start_yymmdd = start_date.strftime("%y%m%d")
    end_yymmdd = end_date.strftime("%y%m%d")
    filename = f"{save_directory}/{endpoint}_data_{start_yymmdd}_{end_yymmdd}.json"
    with open(filename, 'w') as file:
        json.dump(block_data, file, indent=4)
    print(f"Saved data block to {filename}")

def fetch_data(endpoint, server, interval, save_directory, format='json', api_type="biorxiv"):
    """
    Fetch new data in monthly blocks from the API.
    """
    os.makedirs(save_directory, exist_ok=True)
    start_date, end_date = [datetime.strptime(date, "%Y-%m-%d") for date in interval.split('/')]
    current_date = start_date
    tasks = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        while current_date <= end_date:
            block_start = current_date
            block_end = min(current_date + relativedelta(months=1) - relativedelta(days=1), end_date)
            tasks.append(executor.submit(fetch_and_save_data_block,
                                         endpoint, server, block_start, block_end,
                                         save_directory, format, api_type))
            current_date += relativedelta(months=1)
        for future in as_completed(tasks):
            future.result()

###############################
# JSON and Parquet Processing Functions
###############################
def load_json_to_dataframe(json_file):
    """Load JSON data from a file into a pandas DataFrame."""
    with open(json_file, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def save_dataframe(df, save_path):
    """Save a DataFrame in Parquet format."""
    df.to_parquet(save_path)

def process_json_files(directory, save_directory):
    """
    Convert JSON files in a directory to Parquet files.
    If a corresponding embedding file exists (from a previous run), it is removed.
    """
    os.makedirs(save_directory, exist_ok=True)
    json_files = list(Path(directory).glob('*.json'))
    print(f"Found {len(json_files)} JSON file(s) in {directory}: {json_files}")
    
    for json_file in json_files:
        df = load_json_to_dataframe(json_file)
        parquet_filename = f"{json_file.stem}.parquet"
        save_path = os.path.join(save_directory, parquet_filename)
        # Remove an existing embedding (if any) for an update.
        npy_file_path = save_path.replace('db_update', 'embed_update').replace('parquet', 'npy')
        if os.path.exists(npy_file_path):
            os.remove(npy_file_path)
            print(f"Removed embedding file {npy_file_path} due to the dataframe update")
        save_dataframe(df, save_path)
        print(f"Processed and saved {json_file.name} to {parquet_filename}")

def load_unprocessed_parquets(db_update_directory, embed_update_directory):
    """
    Return a list of Parquet files in db_update_directory that do not have a corresponding
    .npy embedding file in embed_update_directory.
    """
    db_update_directory = Path(db_update_directory)
    embed_update_directory = Path(embed_update_directory)
    parquet_files = list(db_update_directory.glob('*.parquet'))
    npy_files = {f.stem for f in embed_update_directory.glob('*.npy')}
    unprocessed_files = []
    for parquet_file in parquet_files:
        if parquet_file.stem not in npy_files:
            unprocessed_files.append(parquet_file)
            print(f"Loaded unprocessed Parquet file: {parquet_file.name}")
        else:
            print(f"Skipping processed Parquet file: {parquet_file.name}")
    return unprocessed_files

###############################
# Dropbox Upload Functions
###############################
def connect_to_dropbox():
    dropbox_APP_KEY = st.secrets["dropbox_APP_KEY"]
    dropbox_APP_SECRET = st.secrets["dropbox_APP_SECRET"]
    dropbox_REFRESH_TOKEN = st.secrets["dropbox_REFRESH_TOKEN"]
    dbx = dropbox.Dropbox(
        app_key=dropbox_APP_KEY,
        app_secret=dropbox_APP_SECRET,
        oauth2_refresh_token=dropbox_REFRESH_TOKEN
    )
    return dbx

def upload_file(file_path, dropbox_file_path, dbx):
    try:
        dropbox_file_path = dropbox_file_path.replace('\\', '/')
        try:
            metadata = dbx.files_get_metadata(dropbox_file_path)
            dropbox_mod_time = metadata.server_modified
            local_mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if dropbox_mod_time >= local_mod_time:
                print(f"Skipped {dropbox_file_path}, Dropbox version is up-to-date.")
                return
        except dropbox.exceptions.ApiError as e:
            if not (hasattr(e, 'error') and e.error.is_path() and e.error.get_path().is_not_found()):
                raise e
        with file_path.open('rb') as f:
            dbx.files_upload(f.read(), dropbox_file_path, mode=dropbox.files.WriteMode.overwrite)
            print(f"Uploaded {dropbox_file_path}")
    except Exception as e:
        print(f"Failed to upload {dropbox_file_path}: {str(e)}")

def upload_path(local_path, dropbox_path):
    dbx = connect_to_dropbox()
    local_path = Path(local_path)
    if local_path.is_file():
        relative_path = local_path.name
        dropbox_file_path = os.path.join(dropbox_path, relative_path).replace('\\', '/').replace('//', '/')
        upload_file(local_path, dropbox_file_path, dbx)
    elif local_path.is_dir():
        for local_file in local_path.rglob('*'):
            if local_file.is_file():
                relative_path = local_file.relative_to(local_path.parent)
                dropbox_file_path = os.path.join(dropbox_path, str(relative_path)).replace('\\', '/').replace('//', '/')
                upload_file(local_file, dropbox_file_path, dbx)
    else:
        print("The provided path does not exist.")

###############################
# Utility: Get Latest Date from Combined File and Update Directory
###############################
def get_latest_date(combined_data_file, update_parquet_directory):
    latest_date = None

    # Check the combined file
    if os.path.exists(combined_data_file):
        try:
            df_combined = pd.read_parquet(combined_data_file)
            if 'date' in df_combined.columns and not df_combined.empty:
                latest_date = df_combined['date'].max()
        except Exception as e:
            print(f"Error reading date from {combined_data_file}: {e}")

    # Check update (Parquet) directory for additional data
    update_files = list(Path(update_parquet_directory).glob('*.parquet'))
    if update_files:
        dates = []
        for file in update_files:
            try:
                df = pd.read_parquet(file)
                if 'date' in df.columns and not df.empty:
                    dates.append(df['date'].max())
            except Exception as e:
                print(f"Error reading date from {file}: {e}")
        if dates:
            update_latest_date = max(dates)
            if latest_date is None or update_latest_date > latest_date:
                latest_date = update_latest_date

    if latest_date is None:
        latest_date = "1990-01-01"
    return latest_date

###############################
# Unified Pipeline Function
###############################
def run_pipeline(source, combined_data_file, update_json_directory,
                 update_parquet_directory, update_embed_directory,
                 endpoint, server, api_type):
    """
    Run the complete update pipeline for a given source.
    
    Parameters:
      source: A string identifier ("biorxiv" or "medrxiv")
      combined_data_file: Path to the combined database file.
      update_json_directory: Directory where new JSON blocks will be saved.
      update_parquet_directory: Directory where JSON files are converted to Parquet.
      update_embed_directory: Directory where new embeddings will be saved.
      endpoint, server: API parameters.
      api_type: Either "biorxiv" or "medrxiv" (to choose the URL format).
    """
    print(f"\n===== Running update pipeline for {source} =====")
    # Determine the most recent date by checking both the combined file and the update directory.
    start_date = get_latest_date(combined_data_file, update_parquet_directory)
    print(f"Latest date found for {source}: {start_date}")
    last_date = datetime.today().strftime('%Y-%m-%d')
    interval = f"{start_date}/{last_date}"
    print(f"Using interval for {source}: {interval}")

    # Step 1. Fetch new JSON data from the API.
    fetch_data(endpoint, server, interval, update_json_directory, format='json', api_type=api_type)

    # Step 2. Convert the JSON files into Parquet files.
    process_json_files(update_json_directory, update_parquet_directory)

    # Step 3. Generate embeddings for unprocessed Parquet files.
    unprocessed_files = load_unprocessed_parquets(update_parquet_directory, update_embed_directory)
    if unprocessed_files:
        for file in unprocessed_files:
            df_update = pd.read_parquet(file)
            query = df_update['abstract'].tolist()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
            model.to(device)
            query_embedding = model.encode(query, normalize_embeddings=True, precision='ubinary', show_progress_bar=True)
            file_stem = os.path.basename(file).split('.')[0]
            os.makedirs(update_embed_directory, exist_ok=True)
            embeddings_path = os.path.join(update_embed_directory, f"{file_stem}.npy")
            np.save(embeddings_path, query_embedding)
            print(f"Saved embeddings {embeddings_path}")

        # Clean up the JSON update directory.
        shutil.rmtree(update_json_directory)
        print(f"Directory '{update_json_directory}' and its contents have been removed.")

        # Upload the updated Parquet and embedding directories to Dropbox.
        for path in [update_parquet_directory, update_embed_directory]:
            upload_path(path, '/')
    else:
        print(f"Nothing to do for {source}.")

###############################
# Main Execution
###############################
if __name__ == "__main__":
    # --- Run pipeline for bioRxiv using configuration ---
    run_pipeline(
        source="biorxiv",
        combined_data_file=biorxiv_config.get("combined_data_file", "combined_biorxiv_data.parquet"),
        update_json_directory=biorxiv_config.get("db_update_json_directory", "db_update_json"),
        update_parquet_directory=biorxiv_config.get("data_folder", "db_update"),
        update_embed_directory=biorxiv_config.get("embeddings_directory", "embed_update"),
        endpoint=biorxiv_config.get("endpoint", "details"),
        server=biorxiv_config.get("server", "biorxiv"),
        api_type="biorxiv"
    )

    # --- Run pipeline for medRxiv using configuration ---
    run_pipeline(
        source="medrxiv",
        combined_data_file=medrxiv_config.get("combined_data_file", "database_2024-08-02.parquet"),
        update_json_directory=medrxiv_config.get("db_update_json_directory", "db_update_json_med"),
        update_parquet_directory=medrxiv_config.get("data_folder", "db_update_med"),
        update_embed_directory=medrxiv_config.get("embeddings_directory", "embed_update_med"),
        endpoint=medrxiv_config.get("endpoint", "details"),
        server=medrxiv_config.get("server", "medrxiv"),
        api_type="medrxiv"
    )
