#!/usr/bin/env python3
"""
PubMed XML Pipeline:
1. Downloads updated/baseline XML files from NCBI FTP.
2. Extracts and processes them into optimized Parquet files.
3. Configuration is loaded from 'config_mss.yaml'.

Note: Embedding generation has been removed and moved to a separate script.
"""

import os
import re
import sys
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
import threading
import asyncio
from fastapi import FastAPI
import uvicorn

# Global flag to indicate update is in progress.
update_in_progress = True
MAX_WORKERS = 4  # Adjust based on CPU cores

# Create FastAPI app for status monitoring
app = FastAPI()

@app.get("/status")
def status():
    if update_in_progress:
        return {"status": "update running"}
    else:
        return {"status": "update completed"}

# Global variable to hold the server instance.
global_server = None

def run_server():
    global global_server
    print("Starting FastAPI status server on port 8001...")
    # Log level warning to keep console clean for pipeline output
    config = uvicorn.Config(app, host="0.0.0.0", port=8001, log_level="warning")
    server = uvicorn.Server(config)
    global_server = server
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())
    print("FastAPI status server has stopped.")

# Start the server in a separate thread.
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

#########################################
# Configuration
#########################################
CONFIG_FILE = "/home/dzyla/pubmed_search/snowflake_code/config_mss.yaml"

print(f"Loading configuration from {CONFIG_FILE}...")

try:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as stream:
            CONFIG = yaml.safe_load(stream)
    else:
        print(f"Warning: Config file {CONFIG_FILE} not found. Using defaults.")
        CONFIG = {}
except Exception as e:
    print(f"Critical Error: Failed to load configuration file: {e}")
    sys.exit(1)

# Get pubmed_config section
PUBMED_CONFIG = CONFIG.get("pubmed_config", {})

# Directories from YAML
DEST_XML_FOLDER = PUBMED_CONFIG.get("xml_directory", "pubmed25_update_xmls")
DEST_DF_FOLDER  = PUBMED_CONFIG.get("data_folder", "pubmed25_update_df")

print(f"  - XML Download Directory: {DEST_XML_FOLDER}")
print(f"  - Parquet Output Directory: {DEST_DF_FOLDER}")

###############################
# Utility Functions
###############################
def ensure_directories():
    """Ensure that all required directories exist."""
    try:
        os.makedirs(DEST_XML_FOLDER, exist_ok=True)
        os.makedirs(DEST_DF_FOLDER, exist_ok=True)
    except OSError as e:
        print(f"Critical Error: Could not create directories: {e}")
        sys.exit(1)

def robust_basename(filepath: str) -> str:
    """Return the file name stem (without extension) robustly."""
    return os.path.splitext(os.path.basename(filepath))[0]

###############################
# Part 1: Download and Extract PubMed XML Files
###############################
def download_and_extract(file_name, base_url, destination_folder):
    """
    Download the gzipped XML file if it doesn't exist.
    Checks for existing XML or existing Parquet to avoid redundant work.
    """
    base_stem = file_name[:-3]  # Remove '.gz'
    xml_file_path = os.path.join(destination_folder, base_stem)
    
    # Check if we already have the XML
    if os.path.exists(xml_file_path):
        return

    # Check if we already have the processed parquet (skip download if so)
    parquet_file = os.path.join(DEST_DF_FOLDER, robust_basename(base_stem[:-4]) + ".parquet")
    if os.path.exists(parquet_file):
        return

    full_url = urljoin(base_url, file_name)
    local_gz_path = os.path.join(destination_folder, file_name)

    # print(f"Downloading {file_name}...") 

    try:
        with requests.get(full_url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(local_gz_path, 'wb') as f_out:
                shutil.copyfileobj(response.raw, f_out)
        
        # Extract immediately to save space if needed, or just standard flow
        with gzip.open(local_gz_path, 'rb') as f_in, open(xml_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        
        os.remove(local_gz_path) # Remove .gz after extraction

    except Exception as e:
        print(f"Error downloading {file_name}: {e}")
        # Do not exit; allow other threads to continue

def get_file_list(url):
    """Retrieve the list of files ending with .xml.gz from the given URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return [link.get('href') for link in soup.find_all('a', href=re.compile(r'.*\.xml\.gz$'))]
    except Exception as e:
        print(f"Error fetching file list from {url}: {e}")
        return []

def download_all_xmls(limit=None):
    """Download XML files from the server if not already present locally."""
    print("\n--- Step 1: Downloading XML Files ---")
    updated_files_url = 'https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/'
    base_url = 'https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/'
    
    base_files = get_file_list(base_url)
    updated_files_list = get_file_list(updated_files_url)
    
    if not base_files and not updated_files_list:
        print("Critical: No files found on FTP servers.")
        sys.exit(1)

    all_files = base_files + updated_files_list
    print(f"Found {len(all_files)} total files (Base: {len(base_files)}, Updates: {len(updated_files_list)})")
    
    if limit is not None:
        all_files = all_files[:limit]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for file in all_files:
            url_to_use = base_url if file in base_files else updated_files_url
            futures.append(executor.submit(download_and_extract, file, url_to_use, DEST_XML_FOLDER))
        
        # Wait for all downloads to complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            future.result()
            
    print("Download phase completed.")

###############################
# Part 2: Process XML Files to Extract Article Metadata
###############################
def format_text(text):
    """Clean and format text by removing HTML entities, tags, and non-printable characters."""
    if not text:
        return ""
    text = str(text)
    text = unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\x20-\x7E]', '', text)
    return ' '.join(text.split())

def parse_date(year, month, day):
    """Parse date components and return a formatted date string (YYYY-MM-DD)."""
    month_mapping = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                     'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                     'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    month = month_mapping.get(month, '01')
    day = '01' if day is None or not str(day).strip().isdigit() else str(day).strip().zfill(2)
    try:
        return datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        # Fallback if date is invalid (e.g., Feb 30), return just year-01-01
        return f"{year}-01-01"

def get_custom_publication_date(article):
    """Extract the publication date with hierarchical fallback logic."""
    best_year = None
    
    def is_valid(y):
        return y and y.strip() and y.strip().isdigit()

    # 1. JournalIssue PubDate
    pub_date = article.find('.//Journal/JournalIssue/PubDate')
    if pub_date is not None:
        year = pub_date.findtext('Year')
        if is_valid(year):
            best_year = year.strip()
            month = pub_date.findtext('Month')
            day = pub_date.findtext('Day')
            try: return parse_date(year, month, day)
            except: pass

    # 2. ArticleDate (Electronic)
    article_date = article.find('.//ArticleDate[@DateType="Electronic"]')
    if article_date is not None:
        year = article_date.findtext('Year')
        if is_valid(year):
            best_year = best_year or year.strip()
            month = article_date.findtext('Month')
            day = article_date.findtext('Day')
            try: return parse_date(year, month, day)
            except: pass

    # 3. DateCompleted
    date_completed = article.find('.//DateCompleted')
    if date_completed is not None:
        year = date_completed.findtext('Year')
        if is_valid(year):
            best_year = best_year or year.strip()
            month = date_completed.findtext('Month')
            day = date_completed.findtext('Day')
            try: return parse_date(year, month, day)
            except: pass

    # 4. History Statuses
    pub_status_priority = ['accepted', 'epublish', 'aheadofprint', 'ppublish']
    for status in pub_status_priority:
        hdate = article.find(f'.//PubmedData/History/PubMedPubDate[@PubStatus="{status}"]')
        if hdate is not None:
            year = hdate.findtext('Year')
            if is_valid(year):
                best_year = best_year or year.strip()
                month = hdate.findtext('Month')
                day = hdate.findtext('Day')
                try: return parse_date(year, month, day)
                except: pass

    # 5. MedlineDate
    medline_date = article.find('.//MedlineDate')
    if medline_date is not None and medline_date.text:
        year_match = re.search(r'\b(19|20)\d{2}\b', medline_date.text)
        if year_match:
            best_year = best_year or year_match.group(0)

    return best_year if best_year else ""

def efficient_process_xml_file(filename):
    """
    Memory-efficient processing of PubMed XML file using iterparse.
    """
    STANDARD_COLUMNS = [
        'doi', 'title', 'authors', 'date', 'version', 'type', 'journal', 'abstract',
        'pmid', 'mesh_terms', 'keywords', 'chemicals'
    ]

    articles_data = []
    
    try:
        context = ET.iterparse(filename, events=('end',))
        
        for event, elem in context:
            if elem.tag.endswith('PubmedArticle'):
                try:
                    article_data = {col: "" for col in STANDARD_COLUMNS}
                    
                    # Identifiers
                    pmid_elem = elem.find('.//PMID')
                    if pmid_elem is not None and pmid_elem.text:
                        article_data['pmid'] = pmid_elem.text.strip()
                        if 'Version' in pmid_elem.attrib:
                            article_data['version'] = pmid_elem.attrib['Version']
                            
                    for aid in elem.findall('.//ArticleId'):
                        if aid.get('IdType') == 'doi' and aid.text:
                            article_data['doi'] = format_text(aid.text)
                            break

                    # Title
                    title_elem = elem.find('.//ArticleTitle')
                    if title_elem is not None:
                        title_parts = [t.strip() for t in title_elem.itertext() if t and t.strip()]
                        article_data['title'] = format_text(' '.join(title_parts))

                    # Metadata
                    article_data['date'] = get_custom_publication_date(elem)
                    
                    ptype = elem.find('.//PublicationType')
                    if ptype is not None and ptype.text:
                        article_data['type'] = format_text(ptype.text)

                    journal = elem.find('.//Journal/Title') or elem.find('.//MedlineTA')
                    if journal is not None and journal.text:
                        article_data['journal'] = format_text(journal.text)

                    # Authors
                    authors = []
                    for auth in elem.findall('.//AuthorList/Author'):
                        parts = []
                        for tag in ['ForeName', 'MiddleName', 'LastName']:
                            part = auth.find(tag)
                            if part is not None and part.text:
                                parts.append(format_text(part.text))
                        if parts:
                            authors.append(" ".join(parts))
                    article_data['authors'] = '; '.join(authors)

                    # Abstract
                    abstract_parts = []
                    for abs_node in elem.findall('.//Abstract/AbstractText'):
                        # Capture full text including children (like <i> tags)
                        full_text = "".join(abs_node.itertext())
                        label = abs_node.get('Label', '').strip()
                        clean_text = format_text(full_text)
                        if label:
                            abstract_parts.append(f"{label}: {clean_text}")
                        else:
                            abstract_parts.append(clean_text)
                    article_data['abstract'] = "\n".join(abstract_parts).strip()

                    # Lists
                    mesh = []
                    for m in elem.findall('.//MeshHeading'):
                        d = m.find('DescriptorName')
                        if d is not None and d.text:
                            t = d.text.strip()
                            q = [x.text.strip() for x in m.findall('QualifierName') if x.text]
                            if q: t += f" [{', '.join(q)}]"
                            mesh.append(t)
                    article_data['mesh_terms'] = '; '.join(mesh)

                    kws = [k.text.strip() for k in elem.findall('.//Keyword') if k.text]
                    article_data['keywords'] = '; '.join(kws)

                    chems = [c.text.strip() for c in elem.findall('.//Chemical/NameOfSubstance') if c.text]
                    article_data['chemicals'] = '; '.join(chems)

                    articles_data.append(article_data)

                except Exception as inner_e:
                    # Log error but don't stop parsing the file
                    pass

                elem.clear() # Clear memory
        
        # Clear root
        if context.root is not None:
            context.root.clear()

    except Exception as e:
        print(f"Error parsing XML structure in {filename}: {e}")
        return pd.DataFrame() # Return empty on fatal error

    df = pd.DataFrame(articles_data)
    if not df.empty:
        df.drop_duplicates(['doi', 'title', 'abstract'], inplace=True)

    return df

def save_optimized_parquet(df, parquet_file, row_group_size=10000):
    """Save DataFrame to an optimized Parquet file."""
    os.makedirs(os.path.dirname(parquet_file), exist_ok=True)
    
    compression_dict = {
        'abstract': 'ZSTD', 'title': 'ZSTD', 'authors': 'ZSTD',
        'mesh_terms': 'ZSTD', 'keywords': 'ZSTD', 'chemicals': 'ZSTD',
        'journal': 'SNAPPY', 'type': 'SNAPPY', 'date': 'SNAPPY',
        'version': 'SNAPPY', 'doi': 'NONE', 'pmid': 'NONE'
    }
    
    # Filter dictionary to only columns present
    comp_settings = {k: v for k, v in compression_dict.items() if k in df.columns}

    if 'date' in df.columns and not df.empty:
        df = df.sort_values('date')
        
    try:
        df.to_parquet(
            parquet_file,
            index=False,
            engine='pyarrow',
            compression=comp_settings,
            row_group_size=row_group_size,
            use_dictionary=True,
            data_page_size=8*1024*1024,
            write_statistics=True,
            version='2.6'
        )
        print(f"Saved optimized parquet: {parquet_file}")
    except Exception as e:
        print(f"Error saving parquet {parquet_file}: {e}")

def process_and_save(file_path):
    """Worker function to process one XML file."""
    file_stem = robust_basename(file_path)
    parquet_file = os.path.join(DEST_DF_FOLDER, file_stem + ".parquet")
    
    if os.path.exists(parquet_file):
        return

    df = efficient_process_xml_file(file_path)
    if df is not None and not df.empty:
        save_optimized_parquet(df, parquet_file)
    else:
        print(f"Warning: No valid data extracted from {file_stem}")

def parallel_process_xml_files():
    """Process XML files in parallel."""
    print("\n--- Step 2: Processing XML to Parquet ---")
    
    if not os.path.exists(DEST_XML_FOLDER):
        print(f"Error: XML directory {DEST_XML_FOLDER} does not exist.")
        return

    xml_files = [os.path.join(DEST_XML_FOLDER, f) for f in os.listdir(DEST_XML_FOLDER) if f.endswith('.xml')]
    
    if not xml_files:
        print("No XML files found to process.")
        return

    print(f"Processing {len(xml_files)} XML files with {MAX_WORKERS} workers...")
    
    Parallel(n_jobs=MAX_WORKERS)(
        delayed(process_and_save)(file) for file in tqdm(xml_files, desc="Processing")
    )

###############################
# Main Entry Point
###############################
def main():
    global update_in_progress
    try:
        ensure_directories()
        download_all_xmls()
        parallel_process_xml_files()
        print("\n--- Pipeline Completed Successfully ---")
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
    finally:
        update_in_progress = False
        if global_server is not None:
            global_server.should_exit = True
        # Join thread to ensure clean exit
        server_thread.join(timeout=2)
        print("Server shutdown complete.")

if __name__ == "__main__":
    main()