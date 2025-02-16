"""
Updated PubMed Pipeline (API Version):
Downloads, extracts, and processes PubMed XML files into Parquet files,
and then generates embeddings by sending requests to an external API.
The abstract extraction has been updated to combine all AbstractText fields,
and the publication date extraction now considers PubDate, ArticleDate,
DateCompleted, and DateRevised.
Configuration values (such as directories) are loaded from config_mss.yaml.
"""

import os
import re
import requests
import gzip
import shutil
import yaml
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import xml.etree.ElementTree as ET
from html import unescape
from datetime import datetime
import numpy as np
import glob
import unidecode
import multiprocessing

###############################
# Load Configuration from YAML
###############################
CONFIG_FILE = "/root/pubmed_search/config_mss.yaml"
try:
    with open(CONFIG_FILE, "r") as stream:
        CONFIG = yaml.safe_load(stream)
except Exception as e:
    print("Error loading configuration file:", e)
    exit(1)

# Get pubmed_config section (if missing, fall back to default values)
PUBMED_CONFIG = CONFIG.get("pubmed_config", {})

# Directories: use the YAML config if available, else use defaults.
DEST_XML_FOLDER   = PUBMED_CONFIG.get("xml_directory", "pubmed25_update_xmls")  # XML folder
DEST_DF_FOLDER    = PUBMED_CONFIG.get("data_folder", "pubmed25_update_df")
DEST_EMBED_FOLDER = PUBMED_CONFIG.get("embeddings_directory", "pubmed25_embed_update")
# (The original GPU lock folder is no longer used in the API version.)
GPU_LOCK_FOLDER   = "gpu_locks"  

###############################
# Utility Functions and Global Helpers
###############################
def ensure_directories():
    """
    Ensure that all required directories exist.
    """
    os.makedirs(DEST_XML_FOLDER, exist_ok=True)
    os.makedirs(DEST_DF_FOLDER, exist_ok=True)
    os.makedirs(DEST_EMBED_FOLDER, exist_ok=True)
    os.makedirs(GPU_LOCK_FOLDER, exist_ok=True)

def robust_basename(filepath: str) -> str:
    """
    Return the file name stem (without extension) robustly.
    """
    return os.path.splitext(os.path.basename(filepath))[0]

def check_existing_files():
    """
    Check for the existence of any XML, Parquet, or embeddings (NumPy) files.
    Return True if at least one file exists.
    """
    xml_exists = os.path.exists(DEST_XML_FOLDER) and any(fname.endswith('.xml') for fname in os.listdir(DEST_XML_FOLDER))
    df_exists = os.path.exists(DEST_DF_FOLDER) and any(fname.endswith('.parquet') for fname in os.listdir(DEST_DF_FOLDER))
    embed_exists = os.path.exists(DEST_EMBED_FOLDER) and any(fname.endswith('.npy') for fname in os.listdir(DEST_EMBED_FOLDER))
    return xml_exists or df_exists or embed_exists

###############################
# Part 1: Download and Extract PubMed XML Files
###############################
def download_and_extract(file_name, base_url, destination_folder):
    """
    Download the gzipped XML file from the given URL if no file with the same stem exists.
    Checks for an existing XML, Parquet, or npy file with the same stem.
    After download, extracts the gzipped file into an XML file in the destination folder.
    """
    base_stem = file_name[:-3]  # Remove '.gz'
    xml_file_name = os.path.join(destination_folder, base_stem)
    # Determine corresponding parquet and embeddings file names:
    parquet_file = os.path.join(DEST_DF_FOLDER, robust_basename(base_stem[:-4]) + ".parquet")
    npy_file = os.path.join(DEST_EMBED_FOLDER, robust_basename(base_stem[:-4]) + ".npy")
    
    # Skip download if any file exists.
    if os.path.exists(xml_file_name) or os.path.exists(parquet_file) or os.path.exists(npy_file):
        return

    full_url = urljoin(base_url, file_name)
    print(f"Downloading {file_name}...")
    response = requests.get(full_url, stream=True)
    response.raise_for_status()
    
    with open(file_name, 'wb') as f_out:
        shutil.copyfileobj(response.raw, f_out)
    
    with gzip.open(file_name, 'rb') as f_in, open(xml_file_name, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    
    os.remove(file_name)

def get_file_list(url):
    """
    Retrieve the list of files ending with .xml.gz from the given URL.
    """
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    return [link.get('href') for link in soup.find_all('a', href=re.compile(r'.*\.xml\.gz$'))]

def download_all_xmls():
    """
    Download XML files from the server if not already present locally.
    """
    updated_files_url = 'https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/'
    base_url = 'https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/'
    
    base_files = get_file_list(base_url)
    updated_files_list = get_file_list(updated_files_url)
    all_files = base_files + updated_files_list

    print(f"Found {len(all_files)} files to check. Base: {len(base_files)}, Update: {len(updated_files_list)}")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for file in all_files:
            url_to_use = base_url if file in base_files else updated_files_url
            futures.append(executor.submit(download_and_extract, file, url_to_use, DEST_XML_FOLDER))
        for future in as_completed(futures):
            future.result()
    print("All files have been checked and downloaded if needed.")

###############################
# Part 2: Process XML Files to Extract Article Metadata
###############################
def format_text(text):
    """
    Clean and format text by removing HTML entities, tags, and non-printable characters.
    """
    text = str(text)
    text = unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\x20-\x7E]', '', text)
    return ' '.join(text.split())

def parse_date(year, month, day):
    """
    Parse date components and return a formatted date string.
    Default month and day are '01' if missing.
    """
    month_mapping = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                     'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                     'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    month = month_mapping.get(month, '01')
    day = '01' if day is None or not day.strip().isdigit() else day.strip().zfill(2)
    return datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").strftime("%Y-%m-%d")

def get_custom_publication_date(article):
    """
    Extract the publication date from an article element.
    In order of preference:
      1. JournalIssue PubDate (standard)
      2. ArticleDate (with DateType="Electronic")
      3. DateCompleted
      4. DateRevised
      5. Fallback to PubMedPubDate statuses.
    """
    # 1. Try standard PubDate inside JournalIssue
    pub_date = article.find('.//Journal/JournalIssue/PubDate')
    if pub_date is not None:
        year = pub_date.findtext('Year')
        month = pub_date.findtext('Month')
        day = pub_date.findtext('Day')
        if year and year.strip():
            try:
                return parse_date(year, month, day)
            except Exception:
                pass

    # 2. Try ArticleDate with DateType Electronic
    article_date = article.find('.//ArticleDate[@DateType="Electronic"]')
    if article_date is not None:
        year = article_date.findtext('Year')
        month = article_date.findtext('Month')
        day = article_date.findtext('Day')
        if year and year.strip():
            try:
                return parse_date(year, month, day)
            except Exception:
                pass

    # 3. Try DateCompleted
    date_completed = article.find('.//DateCompleted')
    if date_completed is not None:
        year = date_completed.findtext('Year')
        month = date_completed.findtext('Month')
        day = date_completed.findtext('Day')
        if year and year.strip():
            try:
                return parse_date(year, month, day)
            except Exception:
                pass

    # 4. Try DateRevised
    date_revised = article.find('.//DateRevised')
    if date_revised is not None:
        year = date_revised.findtext('Year')
        month = date_revised.findtext('Month')
        day = date_revised.findtext('Day')
        if year and year.strip():
            try:
                return parse_date(year, month, day)
            except Exception:
                pass

    # 5. Fallback: use PubMedPubDate statuses
    pub_status_priority = ['accepted', 'epublish', 'aheadofprint', 'ppublish',
                           'ecollection', 'revised', 'received']
    pubmed_dates = article.findall('.//PubmedData/History/PubMedPubDate')
    dates_by_status = {status: None for status in pub_status_priority}
    for pub_date in pubmed_dates:
        pub_status = pub_date.get('PubStatus')
        if pub_status in pub_status_priority:
            year = pub_date.findtext('Year')
            month = pub_date.findtext('Month')
            day = pub_date.findtext('Day')
            if year and year.strip():
                try:
                    complete_date = parse_date(year, month, day)
                    dates_by_status[pub_status] = complete_date
                except Exception:
                    continue
    for status in pub_status_priority:
        if dates_by_status[status]:
            return dates_by_status[status]
    return None

def process_xml_file(filename):
    """
    Process an XML file and extract article metadata into a DataFrame.
    The abstract is constructed by concatenating all AbstractText elements.
    If an AbstractText element has a 'Label', it is prefixed to its text.
    """
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
        articles_data = []
        for article in root.findall('.//PubmedArticle'):
            doi = format_text(article.findtext(".//ArticleId[@IdType='doi']"))
            title = format_text(article.findtext('.//ArticleTitle'))
            
            # Extract authors individually.
            authors = []
            for auth in article.findall('.//AuthorList/Author'):
                name_parts = []
                for tag in ['ForeName', 'MiddleName', 'LastName']:
                    part = auth.findtext(tag)
                    if part:
                        name_parts.append(format_text(part))
                if name_parts:
                    authors.append(" ".join(name_parts))
            
            date = get_custom_publication_date(article)
            if not date:
                continue  # Skip if no valid date found
            
            version = article.findtext('.//PMID[@Version]')
            type_ = format_text(article.findtext('.//PublicationType'))
            journal = article.findtext('.//Journal/Title')
            
            # Combine all AbstractText elements.
            abstract_parts = []
            for a in article.findall('.//Abstract/AbstractText'):
                txt = a.text
                if txt:
                    txt = txt.strip()
                    label = a.attrib.get('Label', '').strip()
                    if label:
                        abstract_parts.append(f"{label}: {txt}")
                    else:
                        abstract_parts.append(txt)
            abstract = "\n".join(abstract_parts).strip()
            
            articles_data.append({
                'doi': doi,
                'title': title,
                'authors': '; '.join(authors),
                'date': date,
                'version': version,
                'type': type_,
                'journal': journal,
                'abstract': abstract
            })
        articles_df = pd.DataFrame(articles_data)
        articles_df.drop_duplicates(['doi', 'title', 'abstract'], inplace=True)
        return articles_df
    except ET.ParseError:
        print(f"Error parsing file: {filename}")
        return pd.DataFrame()

def process_and_save(file, processed_directory):
    """
    Process a single XML file and save the resulting DataFrame as a Parquet file.
    Skips processing if the Parquet file already exists.
    """
    file_stem = robust_basename(file)
    parquet_file = os.path.join(processed_directory, file_stem + ".parquet")
    if os.path.exists(parquet_file):
        return
    print(f"Processing {file}...")
    df = process_xml_file(file)
    if df is not None and not df.empty:
        df.to_parquet(parquet_file, index=False)
        print(f"Saved in {parquet_file}")
    else:
        print(f"File {file} resulted in an empty DataFrame.")

def parallel_process_xml_files(input_path, processed_directory, n_jobs=24):
    """
    Process XML files in parallel from the input directory and save them as Parquet files.
    """
    if not os.path.exists(processed_directory):
        os.makedirs(processed_directory)
    xml_files = []
    if os.path.isdir(input_path):
        xml_files = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith('.xml')]
    elif os.path.isfile(input_path) and input_path.endswith('.xml'):
        xml_files = [input_path]
    else:
        print("Invalid input path. It must be either an XML file or a directory containing XML files.")
        return
    print(f"Found {len(xml_files)} XML file(s) to process.")
    Parallel(n_jobs=n_jobs)(delayed(process_and_save)(file, processed_directory) for file in tqdm(xml_files))

###############################
# Part 4: Generate Embeddings via API
###############################
def get_embedding(text, normalize=True, precision="ubinary"):
    """
    Sends a POST request to the embedding API and returns the embedding as a 1D NumPy array (shape: (128,))
    with uint8 precision.
    """
    payload = {
        "text": text,
        "normalize": normalize,
        "precision": precision
    }
    try:
        # Adjust the URL if your server runs elsewhere.
        response = requests.post("http://localhost:8000/encode", json=payload)
        response.raise_for_status()
        data = response.json()
        # Convert the returned list to a NumPy array with uint8 precision.
        embedding = np.array(data["embedding"], dtype=np.uint8)
        # Remove any extra dimensions (if shape is (1, 128), squeeze it to (128,))
        embedding = np.squeeze(embedding)
        return embedding
    except Exception as e:
        print(f"Error obtaining embedding for text starting with '{text[:30]}': {e}")
        # Return a zero vector of size 128 if there's an error (adjust if necessary)
        return np.zeros(128, dtype=np.uint8)

def process_embeddings_api(file):
    """
    For a given Parquet file, construct queries by joining the title and abstract,
    then request embeddings via the API for each query and save the results.
    """
    file_stem = robust_basename(file)
    embeddings_path = os.path.join(DEST_EMBED_FOLDER, f"{file_stem}.npy")
    if os.path.exists(embeddings_path):
        return
    df = pd.read_parquet(file)
    queries = [
        f"{(title or '').strip()} {(abstract or '').strip()}".strip()
        for title, abstract in zip(df['title'], df['abstract'])
    ]
    print(f"Processing embeddings for {file} with {len(queries)} entries using API")
    embeddings = []
    for query in tqdm(queries, desc=f"Embedding {file_stem}"):
        emb = get_embedding(query, normalize=True, precision="ubinary")
        embeddings.append(emb)
    embeddings_array = np.array(embeddings, dtype=np.uint8)
    np.save(embeddings_path, embeddings_array)
    print(f"Saved embeddings at {embeddings_path}")

def main_api_embeddings(input_directory):
    """
    Process embeddings for all Parquet files sequentially by sending API requests.
    """
    file_paths = sorted(glob.glob(os.path.join(input_directory, '*.parquet')))
    if not file_paths:
        print("No parquet files found in the directory.")
        return
    os.makedirs(DEST_EMBED_FOLDER, exist_ok=True)
    for file in tqdm(file_paths, desc="API embeddings processing"):
        process_embeddings_api(file)

###############################
# Main Entry Point
###############################
def main():
    """
    Execute the complete pipeline:
      1. Download and extract XML files.
      2. Process XML files into Parquet DataFrames.
      3. Generate embeddings by sending requests to the embedding API.
    """
    ensure_directories()
    
    # Step 1: Download XML files if not already present.
    download_all_xmls()
    
    # Step 2: Process XML files into Parquet DataFrames.
    parallel_process_xml_files(DEST_XML_FOLDER, DEST_DF_FOLDER)
    
    # Step 3: Generate embeddings via the API.
    print("Starting API embeddings processing...")
    main_api_embeddings(DEST_DF_FOLDER)

if __name__ == "__main__":
    # In this API version, we use sequential embedding processing.
    main()
    print("Pipeline completed successfully.")
