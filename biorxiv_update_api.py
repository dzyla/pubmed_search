import os
import json
import time
import shutil
import yaml
import requests
import numpy as np
import pandas as pd
import dropbox
import streamlit as st
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI
import uvicorn
import threading
import asyncio

# Create FastAPI app.
app = FastAPI()

@app.get("/status")
def status():
    return {"status": "running"}

# Global variable to hold the server instance.
global_server = None

def run_server():
    global global_server
    print("Starting FastAPI status server on port 8001...")
    config = uvicorn.Config(app, host="0.0.0.0", port=8001, log_level="info")
    server = uvicorn.Server(config)
    global_server = server
    # Create a new event loop for this thread.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())
    print("FastAPI status server has stopped.")

# Start the server in a separate thread.
server_thread = threading.Thread(target=run_server)
server_thread.start()

# --------------------- Load YAML Configuration ---------------------
print("Loading YAML configuration...")
with open("/root/pubmed_search/config_mss.yaml", "r") as f:
    config = yaml.safe_load(f)
print("YAML configuration loaded.")

# Extract configuration for bioRxiv and medRxiv.
biorxiv_conf = config.get("biorxiv_config", {})
medrxiv_conf = config.get("medrxiv_config", {})

# Use the config values.
biorxiv_data_folder = biorxiv_conf.get("data_folder", "biorxiv_df")
biorxiv_embeddings_dir = biorxiv_conf.get("embeddings_directory", "biorxiv_embed")

medrxiv_data_folder = medrxiv_conf.get("data_folder", "medrxiv_df")
medrxiv_embeddings_dir = medrxiv_conf.get("embeddings_directory", "medarxiv_embed")

# Temporary directories for JSON files (not defined in the config)
temp_json_biorxiv = "db_update_json_temp_biorxiv"
temp_json_medrxiv = "db_update_json_temp_medrxiv"

# --------------------- Retry Decorator ---------------------
def retry_on_exception(exception, retries=5, delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(retries):
                try:
                    print(f"Attempt {i+1} for function {func.__name__}...")
                    return func(*args, **kwargs)
                except exception as e:
                    last_exception = e
                    print(f"Retrying due to: {str(e)}")
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

# --------------------- Data Fetching Functions ---------------------
def fetch_and_save_data_block_biorxiv(endpoint, server, block_start, block_end, save_directory, format='json'):
    base_url = f"https://api.biorxiv.org/{endpoint}/{server}/"
    block_interval = f"{block_start.strftime('%Y-%m-%d')}/{block_end.strftime('%Y-%m-%d')}"
    block_data = []
    cursor = 0
    continue_fetching = True

    print(f"Starting fetch for bioRxiv block {block_interval}...")
    while continue_fetching:
        url = f"{base_url}{block_interval}/{cursor}/{format}"
        print(f"Requesting URL: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch data for block {block_interval} at cursor {cursor}. HTTP Status: {response.status_code}")
            break

        data = response.json()
        fetched_papers = len(data['collection'])
        if fetched_papers > 0:
            block_data.extend(data['collection'])
            cursor += fetched_papers
            print(f"Fetched {fetched_papers} papers for block {block_interval}. Total fetched: {cursor}.")
        else:
            continue_fetching = False

    if block_data:
        save_data_block(block_data, block_start, block_end, endpoint, save_directory)

@retry_on_exception(requests.exceptions.ConnectionError)
def fetch_and_save_data_block_medrxiv(endpoint, server, block_start, block_end, save_directory, format='json'):
    base_url = f"https://api.medrxiv.org/details/{server}/"
    block_interval = f"{block_start.strftime('%Y-%m-%d')}/{block_end.strftime('%Y-%m-%d')}"
    block_data = []
    cursor = 0
    continue_fetching = True

    print(f"Starting fetch for medRxiv block {block_interval}...")
    while continue_fetching:
        url = f"{base_url}{block_interval}/{cursor}/{format}"
        print(f"Requesting URL: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch data for block {block_interval} at cursor {cursor}. HTTP Status: {response.status_code}")
            break

        data = response.json()
        fetched_papers = len(data['collection'])
        if fetched_papers > 0:
            block_data.extend(data['collection'])
            cursor += fetched_papers
            print(f"Fetched {fetched_papers} papers for block {block_interval}. Total fetched: {cursor}.")
        else:
            continue_fetching = False

    if block_data:
        save_data_block(block_data, block_start, block_end, endpoint, save_directory)

def fetch_data_biorxiv(endpoint, server, interval, save_directory, format='json'):
    os.makedirs(save_directory, exist_ok=True)
    start_date, end_date = [datetime.strptime(date, "%Y-%m-%d") for date in interval.split('/')]
    current_date = start_date
    tasks = []
    print("Starting concurrent fetching for bioRxiv data...")
    with ThreadPoolExecutor(max_workers=12) as executor:
        while current_date <= end_date:
            block_start = current_date
            block_end = min(current_date + relativedelta(months=1) - relativedelta(days=1), end_date)
            print(f"Scheduling fetch for block: {block_start.strftime('%Y-%m-%d')} to {block_end.strftime('%Y-%m-%d')}")
            tasks.append(executor.submit(fetch_and_save_data_block_biorxiv, endpoint, server, block_start, block_end, save_directory, format))
            current_date += relativedelta(months=1)
        for future in as_completed(tasks):
            future.result()
    print("Completed fetching bioRxiv data.")

def fetch_data_medrxiv(endpoint, server, interval, save_directory, format='json'):
    os.makedirs(save_directory, exist_ok=True)
    start_date, end_date = [datetime.strptime(date, "%Y-%m-%d") for date in interval.split('/')]
    current_date = start_date
    tasks = []
    print("Starting concurrent fetching for medRxiv data...")
    with ThreadPoolExecutor(max_workers=12) as executor:
        while current_date <= end_date:
            block_start = current_date
            block_end = min(current_date + relativedelta(months=1) - relativedelta(days=1), end_date)
            print(f"Scheduling fetch for block: {block_start.strftime('%Y-%m-%d')} to {block_end.strftime('%Y-%m-%d')}")
            tasks.append(executor.submit(fetch_and_save_data_block_medrxiv, endpoint, server, block_start, block_end, save_directory, format))
            current_date += relativedelta(months=1)
        for future in as_completed(tasks):
            future.result()
    print("Completed fetching medRxiv data.")

# --------------------- File Saving and Processing Functions ---------------------
def save_data_block(block_data, start_date, end_date, endpoint, save_directory):
    start_yymmdd = start_date.strftime("%y%m%d")
    end_yymmdd = end_date.strftime("%y%m%d")
    filename = f"{save_directory}/{endpoint}_data_{start_yymmdd}_{end_yymmdd}.json"
    with open(filename, 'w') as file:
        json.dump(block_data, file, indent=4)
    print(f"Saved data block to {filename}")

def load_json_to_dataframe(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def save_dataframe(df, save_path):
    df.to_parquet(save_path)
    print(f"Saved DataFrame to {save_path}")

def process_json_files(json_directory, output_directory):
    """
    Process all JSON files in json_directory.
    Convert them to Parquet and save in output_directory.
    """
    os.makedirs(output_directory, exist_ok=True)
    json_files = list(Path(json_directory).glob("*.json"))
    print(f"Found JSON files: {json_files}")
    for json_file in json_files:
        df = load_json_to_dataframe(json_file)
        parquet_filename = f"{json_file.stem}.parquet"
        save_path = os.path.join(output_directory, parquet_filename)
        # If an old embedding exists for this file, remove it.
        npy_file_path = save_path.replace("df", os.path.basename(os.path.normpath(output_directory)).replace("df", "embed")).replace("parquet", "npy")
        if os.path.exists(npy_file_path):
            os.remove(npy_file_path)
            print(f"Removed embedding file {npy_file_path} due to dataframe update")
        save_dataframe(df, save_path)
        print(f"Processed and saved {json_file.name} to {parquet_filename}")

def load_unprocessed_parquets(data_folder, embeddings_directory):
    """
    List all Parquet files in data_folder that do not have a corresponding .npy file
    in embeddings_directory.
    """
    data_folder = Path(data_folder)
    embeddings_directory = Path(embeddings_directory)
    parquet_files = list(data_folder.glob("*.parquet"))
    npy_files = {f.stem for f in embeddings_directory.glob("*.npy")}
    unprocessed_files = []
    for parquet_file in parquet_files:
        if parquet_file.stem not in npy_files:
            unprocessed_files.append(parquet_file)
            print(f"Loaded unprocessed Parquet file: {parquet_file.name}")
        else:
            print(f"Skipping processed Parquet file: {parquet_file.name}")
    return unprocessed_files

# --------------------- Dropbox Upload Functions ---------------------
def connect_to_dropbox():
    dropbox_APP_KEY = st.secrets["dropbox_APP_KEY"]
    dropbox_APP_SECRET = st.secrets["dropbox_APP_SECRET"]
    dropbox_REFRESH_TOKEN = st.secrets["dropbox_REFRESH_TOKEN"]
    dbx = dropbox.Dropbox(
        app_key=dropbox_APP_KEY,
        app_secret=dropbox_APP_SECRET,
        oauth2_refresh_token=dropbox_REFRESH_TOKEN
    )
    print("Connected to Dropbox.")
    return dbx

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
                print(f"No existing file on Dropbox, proceeding with upload: {dropbox_file_path}")
        with file_path.open('rb') as f:
            dbx.files_upload(f.read(), dropbox_file_path, mode=dropbox.files.WriteMode.overwrite)
            print(f"Uploaded {dropbox_file_path}")
    except Exception as e:
        print(f"Failed to upload {dropbox_file_path}: {str(e)}")

# --------------------- REST API Embedding Functions ---------------------
def get_embedding_from_api(text):
    """
    Call the REST API to encode a single text.
    The API is expected to be running at http://localhost:8000/encode.
    """
    url = "http://localhost:8000/encode"
    payload = {"text": text, "normalize": True, "precision": "ubinary"}
    print(f"Requesting embedding for text: {text[:30]}...")  # Show first 30 characters
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        emb = result.get("embedding", None)
        if emb is not None and isinstance(emb, list) and len(emb) > 0:
            print("Received embedding successfully.")
            # The API returns a list containing one embedding vector.
            return emb[0]
        else:
            raise Exception("Unexpected embedding format from API.")
    else:
        raise Exception(f"API call failed with status code: {response.status_code}")

def get_embeddings_from_api(text_list):
    """
    For each text in the provided list, call the REST API to obtain its embedding.
    Return a 2D numpy array of shape (n_texts, 128) with dtype uint8.
    """
    embeddings = []
    print("Getting embeddings for a list of texts...")
    for text in text_list:
        emb = get_embedding_from_api(text)
        embeddings.append(emb)
    np_embeddings = np.array(embeddings, dtype=np.uint8)
    if np_embeddings.ndim == 1:
        np_embeddings = np.expand_dims(np_embeddings, axis=0)
    print("Completed obtaining embeddings.")
    return np_embeddings

# --------------------- Last Checking Date Function ---------------------
def get_last_checking_date(data_folder):
    """
    Scan all Parquet files in data_folder and return the maximum date found
    in the "date" column as a string in "YYYY-MM-DD" format.
    If no files are found, return a default start date.
    """
    data_folder = Path(data_folder)
    parquet_files = list(data_folder.glob("*.parquet"))
    max_date = None
    print(f"Scanning {len(parquet_files)} parquet files for last checking date...")
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            if not df.empty and "date" in df.columns:
                file_max = pd.to_datetime(df["date"]).max()
                if max_date is None or file_max > max_date:
                    max_date = file_max
        except Exception as e:
            print(f"Error reading {file}: {e}")
    if max_date is None:
        print("No parquet files found; using default start date.")
        return "1990-01-01"
    else:
        last_date_str = max_date.strftime("%Y-%m-%d")
        print(f"Last checking date determined as: {last_date_str}")
        return last_date_str

# --------------------- Main Processing ---------------------
if __name__ == "__main__":
    
    update_in_progress = True
    
    # ------------------ Process bioRxiv Data ------------------
    print("Processing bioRxiv data...")
    start_date_bio = get_last_checking_date(biorxiv_data_folder)
    last_date = datetime.today().strftime('%Y-%m-%d')
    interval_bio = f"{start_date_bio}/{last_date}"
    print(f"Using interval for bioRxiv: {interval_bio}")

    # Fetch new JSON data into a temporary folder.
    os.makedirs(temp_json_biorxiv, exist_ok=True)
    print("Fetching bioRxiv JSON data...")
    fetch_data_biorxiv(endpoint="details", server="biorxiv", interval=interval_bio, save_directory=temp_json_biorxiv)

    # Process JSON files into Parquet files in the data_folder.
    print("Processing JSON files into Parquet format for bioRxiv...")
    process_json_files(temp_json_biorxiv, biorxiv_data_folder)

    # Check for any Parquet file in data_folder that does not have a corresponding embedding.
    unprocessed_files_bio = load_unprocessed_parquets(biorxiv_data_folder, biorxiv_embeddings_dir)
    if unprocessed_files_bio:
        print("Generating embeddings for bioRxiv data...")
        for file in unprocessed_files_bio:
            df = pd.read_parquet(file)
            query = df['abstract'].tolist()
            query_embedding = get_embeddings_from_api(query)
            file_basename = file.stem
            embedding_path = os.path.join(biorxiv_embeddings_dir, f"{file_basename}.npy")
            np.save(embedding_path, query_embedding)
            print(f"Saved embeddings to {embedding_path}")
        # Remove the temporary JSON directory.
        shutil.rmtree(temp_json_biorxiv)
        print(f"Removed temporary JSON directory {temp_json_biorxiv}")
        # Upload the updated data and embeddings folders to Dropbox.
        for path in [biorxiv_data_folder, biorxiv_embeddings_dir]:
            print(f"Uploading folder {path} to Dropbox...")
            upload_path(path, "//")
    else:
        print("Nothing to do for bioRxiv data.")

    # ------------------ Process medRxiv Data ------------------
    print("Processing medRxiv data...")
    start_date_med = get_last_checking_date(medrxiv_data_folder)
    interval_med = f"{start_date_med}/{last_date}"
    print(f"Using interval for medRxiv: {interval_med}")

    os.makedirs(temp_json_medrxiv, exist_ok=True)
    print("Fetching medRxiv JSON data...")
    fetch_data_medrxiv(endpoint="details", server="medrxiv", interval=interval_med, save_directory=temp_json_medrxiv)
    print("Processing JSON files into Parquet format for medRxiv...")
    process_json_files(temp_json_medrxiv, medrxiv_data_folder)

    unprocessed_files_med = load_unprocessed_parquets(medrxiv_data_folder, medrxiv_embeddings_dir)
    if unprocessed_files_med:
        print("Generating embeddings for medRxiv data...")
        for file in unprocessed_files_med:
            df = pd.read_parquet(file)
            query = df['abstract'].tolist()
            query_embedding = get_embeddings_from_api(query)
            file_basename = file.stem
            os.makedirs(medrxiv_embeddings_dir, exist_ok=True)
            embedding_path = os.path.join(medrxiv_embeddings_dir, f"{file_basename}.npy")
            np.save(embedding_path, query_embedding)
            print(f"Saved embeddings to {embedding_path}")
        shutil.rmtree(temp_json_medrxiv)
        print(f"Removed temporary JSON directory {temp_json_medrxiv}")
        for path in [medrxiv_data_folder, medrxiv_embeddings_dir]:
            print(f"Uploading folder {path} to Dropbox...")
            upload_path(path, "/")
    else:
        print("Nothing to do for medRxiv data.")

    # Signal the server to shut down.
    if global_server is not None:
        print("Signaling FastAPI status server to shut down...")
        global_server.should_exit = True

    # Wait for the server thread to finish.
    server_thread.join()
    print("Server has shut down.")
    quit()
