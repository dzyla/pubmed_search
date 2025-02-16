import os
import re
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import torch
import pyarrow as pa
import pyarrow.parquet as pq
import streamlit as st
import time
import plotly.express as px
import requests
import gc
import streamlit.components.v1 as components
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.manifold import MDS
import yaml
import google.genai as genai

from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import semantic_search_faiss
from umap_pytorch import load_pumap

# Import Crossref Works for citation lookup
from crossref.restful import Works

###############################################################################
# Global settings and helper function to call the REST API for model encoding
###############################################################################

# Set the URL of your model server REST API
MODEL_SERVER_URL = "http://localhost:8000/encode"


def get_query_embedding(query, normalize=True, precision="ubinary"):
    """
    Get the query embedding from the REST API model server.
    The API expects a JSON payload with keys "text", "normalize", and "precision".
    It returns the embedding as a list.
    """
    payload = {"text": query, "normalize": normalize, "precision": precision}
    response = requests.post(MODEL_SERVER_URL, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["embedding"]
    else:
        raise Exception(
            f"Failed to get query embedding from model server: {response.status_code} {response.text}"
        )


# --- Crossref Helper Functions ---


@st.cache_data(show_spinner=False)
def get_citation_count(doi):
    """
    Retrieve the number of times a manuscript has been cited using its DOI.
    """
    works = Works()
    try:
        paper_data = works.doi(doi)
        return paper_data.get("is-referenced-by-count", 0)  # type: ignore
    except Exception as e:
        print(f"An error occurred while fetching citation count for {doi}: {e}")
        return 0


@st.cache_data(show_spinner=False)
def get_references(doi):
    """
    Retrieve and format the list of references cited by a manuscript.
    """
    works = Works()
    try:
        paper_data = works.doi(doi)
        references = paper_data.get("reference", [])  # type: ignore
        formatted_refs = []
        for ref in references:
            if isinstance(ref, dict):
                if "unstructured" in ref:
                    formatted_refs.append(ref["unstructured"])
                else:
                    formatted_refs.append(
                        ", ".join(f"{k}: {v}" for k, v in ref.items())
                    )
            else:
                formatted_refs.append(str(ref))
        return formatted_refs
    except Exception as e:
        print(f"An error occurred while fetching references for {doi}: {e}")
        return []


###############################################################################
# PAGE CONFIGURATION
###############################################################################

st.set_page_config(
    page_title="MSS",
    page_icon="📜",
)

###############################################################################
# PUBMED & BIORXIV COMMON FUNCTIONS (CHUNKED EMBEDDINGS)
###############################################################################


def build_sorted_intervals_from_metadata(metadata: dict) -> list:
    """
    Build a sorted list of global intervals for binary search.
    Each interval is represented by a dictionary with keys:
      - "global_start": global starting index (inclusive)
      - "global_end": global ending index (inclusive)
      - "source_stem": base name of the npy file
      - "source_local_start": starting index within the source file
    """
    STATUS.info("Building sorted intervals from metadata...")
    intervals = []
    for chunk in metadata["chunks"]:
        for part in chunk["parts"]:
            part_global_start = chunk["global_start"] + part["chunk_local_start"]
            part_global_end = chunk["global_start"] + part["chunk_local_end"] - 1
            intervals.append(
                {
                    "global_start": part_global_start,
                    "global_end": part_global_end,
                    "source_stem": part["source_stem"],
                    "source_local_start": part["source_local_start"],
                }
            )
    intervals.sort(key=lambda x: x["global_start"])
    STATUS.info("Completed building and sorting intervals.")
    return intervals


def find_file_for_index_in_metadata(global_idx: int, intervals: list) -> dict:
    """
    Use binary search to find which npy file (and its local index) corresponds to a global index.
    """
    lo = 0
    hi = len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        interval = intervals[mid]
        if interval["global_start"] <= global_idx <= interval["global_end"]:
            offset = global_idx - interval["global_start"]
            local_idx = interval["source_local_start"] + offset
            return {"source_stem": interval["source_stem"], "local_idx": local_idx}
        elif global_idx < interval["global_start"]:
            hi = mid - 1
        else:
            lo = mid + 1
    return None  # type: ignore


def load_data_for_indices(
    global_indices: list, metadata: dict, folder: str
) -> pd.DataFrame:
    """
    Load only the necessary rows from Parquet files given global indices.
    """
    STATUS.info("Loading data for given global indices from Parquet files...")
    intervals = build_sorted_intervals_from_metadata(metadata)
    file_indices = {}
    for idx in global_indices:
        location = find_file_for_index_in_metadata(idx, intervals)
        if location is not None:
            file_name = location["source_stem"]
            file_indices.setdefault(file_name, []).append(location["local_idx"])
        else:
            print(f"Global index {idx} not found in metadata intervals.")
    results = []
    for file_name, local_indices in file_indices.items():
        parquet_path = os.path.join(folder, f"{file_name}.parquet")
        print(f"Processing file: {parquet_path} with indices: {local_indices}")
        if not os.path.exists(parquet_path):
            print(
                f"Warning: Parquet file not found for source {file_name} at {parquet_path}"
            )
            continue
        table = pq.read_table(parquet_path)
        local_indices.sort()
        pa_indices = pa.array(local_indices, type=pa.int64())
        subset_table = table.take(pa_indices)
        df_subset = subset_table.to_pandas()
        results.append(df_subset)
    if not results:
        STATUS.info("No data loaded. Returning empty DataFrame.")
        return pd.DataFrame()
    df_combined = pd.concat(results, ignore_index=True)
    STATUS.info("Completed loading data for all global indices.")
    return df_combined


class ChunkedEmbeddings:
    """
    A wrapper for a list of memory-mapped embedding chunks.
    """

    def __init__(
        self, chunks: list, chunk_boundaries: list, total_rows: int, embedding_dim: int
    ):
        STATUS.info("Initializing ChunkedEmbeddings object...")
        self.chunks = chunks
        self.chunk_boundaries = (
            chunk_boundaries  # List of tuples (global_start, global_end) per chunk.
        )
        self.total_rows = total_rows
        self.embedding_dim = embedding_dim
        print(
            f"ChunkedEmbeddings initialized with {len(chunks)} chunks, total_rows={total_rows}, embedding_dim={embedding_dim}"
        )

    @property
    def shape(self):
        return (self.total_rows, self.embedding_dim)

    def close(self):
        """
        Close the memory-mapped files to release RAM.
        """
        for idx, chunk in enumerate(self.chunks):
            if hasattr(chunk, "_mmap"):
                try:
                    chunk._mmap.close()
                    print(f"Closed memmap for chunk {idx}")
                except Exception as e:
                    print(f"Error closing memmap for chunk {idx}: {e}")


def create_chunked_embeddings_memmap(
    embeddings_directory: str,
    npy_files_pattern: str,
    chunk_dir: str,
    metadata_path: str,
    chunk_size_bytes: int = 1 << 30,
):
    """
    Create (or load) chunked memory-mapped embeddings.
    If metadata and chunk files exist, they are loaded; otherwise, the chunks are created.
    Additionally, this function now checks whether the npy files referenced in the metadata
    match the current files in the embeddings directory. If a mismatch is detected, the old
    chunks are removed and fresh ones are generated.
    """
    STATUS.info("Starting creation or loading of chunked memory-mapped embeddings...")
    recreate = False
    if os.path.exists(metadata_path):
        print(f"Metadata file found at {metadata_path}. Loading metadata...")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        # Check that all chunk files are present.
        for chunk_info in metadata["chunks"]:
            chunk_file = chunk_info["chunk_file"]
            chunk_path = os.path.join(chunk_dir, chunk_file)
            if not os.path.exists(chunk_path):
                print(
                    f"Chunk file {chunk_file} not found in {chunk_dir}. Will recreate chunks."
                )
                recreate = True
                break
        # If all chunk files exist, then check that the npy files in the metadata match the current directory.
        if not recreate:
            metadata_npy_files = set()
            for chunk_info in metadata["chunks"]:
                for part in chunk_info["parts"]:
                    metadata_npy_files.add(part["source_stem"])
            current_npy_files = set(
                f.stem for f in Path(embeddings_directory).glob(npy_files_pattern)
            )
            if metadata_npy_files != current_npy_files:
                print(
                    "Mismatch in npy files between metadata and current directory. Will recreate chunks."
                )
                recreate = True
        # If nothing requires recreation, then load the existing chunks.
        if not recreate:
            STATUS.info(
                "All chunk files and npy file mappings are valid. Loading memory-mapped chunks from disk..."
            )
            chunks = []
            chunk_boundaries = []
            for chunk_info in metadata["chunks"]:
                chunk_file = chunk_info["chunk_file"]
                actual_rows = chunk_info["actual_rows"]
                embedding_dim = metadata["embedding_dim"]
                memmap_array = np.memmap(
                    os.path.join(chunk_dir, chunk_file),
                    dtype=np.uint8,
                    mode="r",
                    shape=(actual_rows, embedding_dim),
                )
                chunks.append(memmap_array)
                chunk_boundaries.append(
                    (chunk_info["global_start"], chunk_info["global_end"])
                )
                print(
                    f"Loaded chunk from {chunk_file}: rows={actual_rows}, boundaries=({chunk_info['global_start']}, {chunk_info['global_end']})"
                )
            total_rows = metadata["total_rows"]
            return ChunkedEmbeddings(
                chunks, chunk_boundaries, total_rows, metadata["embedding_dim"]
            ), metadata
        else:
            STATUS.info(
                "Recreation flag triggered. Removing old metadata and chunk files."
            )
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            for file in Path(chunk_dir).glob("chunk_*.npy"):
                os.remove(file)
    else:
        print(
            f"No metadata file found at {metadata_path}. Will create new chunked embeddings."
        )
    os.makedirs(chunk_dir, exist_ok=True)
    STATUS.info(f"Chunk directory ensured at {chunk_dir}.")

    # Get the list of npy files.
    npy_files = sorted(list(Path(embeddings_directory).glob(npy_files_pattern)))
    # If no files are found, raise an error.
    if not npy_files:
        raise FileNotFoundError(
            f"No npy files found in {embeddings_directory} with pattern {npy_files_pattern}"
        )

    STATUS.info(
        f"Found {len(npy_files)} npy files matching pattern {npy_files_pattern}."
    )
    total_rows = 0
    embedding_dim = None
    file_infos = []
    for fname in npy_files:
        print(f"Processing file: {fname}")
        arr = np.load(fname, mmap_mode="r")
        try:
            rows, dims = arr.shape
        except ValueError:
            try:
                rows, _, dims = arr.shape
            except ValueError:
                raise ValueError(f"Invalid shape for file {fname}: {arr.shape}")
        print(f"File {fname} has shape: ({rows}, {dims})")
        if embedding_dim is None:
            embedding_dim = dims
            print(f"Setting embedding dimension to {dims}")
        elif dims != embedding_dim:
            raise ValueError(f"Embedding dimension mismatch in {fname}")
        file_infos.append(
            {"file_stem": fname.stem, "rows": rows, "file_path": str(fname)}
        )
        total_rows += rows
        del arr
    STATUS.info(f"Total rows: {total_rows}")
    rows_per_chunk = chunk_size_bytes // embedding_dim  # type: ignore
    STATUS.info(f"Maximum rows per chunk calculated as: {rows_per_chunk}")
    # Group npy files into chunks without splitting any file.
    chunk_groups = []
    current_chunk_files = []
    current_chunk_rows = 0
    STATUS.info("Grouping npy files into chunk groups...")
    for file_info in file_infos:
        file_rows = file_info["rows"]
        if file_rows > rows_per_chunk:
            if current_chunk_files:
                chunk_groups.append(current_chunk_files)
                current_chunk_files = []
                current_chunk_rows = 0
            chunk_groups.append([file_info])
            print(
                f"File {file_info['file_stem']} is too large and placed in its own chunk."
            )
        else:
            if current_chunk_rows + file_rows > rows_per_chunk and current_chunk_files:
                chunk_groups.append(current_chunk_files)
                current_chunk_files = []
                current_chunk_rows = 0
            current_chunk_files.append(file_info)
            current_chunk_rows += file_rows
    if current_chunk_files:
        chunk_groups.append(current_chunk_files)
        STATUS.info("Added final chunk group.")
    # Create memmap chunks from groups.
    chunks = []
    chunk_boundaries = []
    chunks_metadata = []
    global_index = 0
    chunk_idx = 0
    STATUS.info("Creating memory-mapped chunks from grouped files...")
    for group in chunk_groups:
        group_total_rows = sum(file_info["rows"] for file_info in group)
        chunk_file = f"chunk_{chunk_idx}.npy"
        chunk_path = os.path.join(chunk_dir, chunk_file)
        print(
            f"Creating chunk {chunk_idx}: file={chunk_file}, total_rows={group_total_rows}"
        )
        memmap_array = np.memmap(
            chunk_path,
            dtype=np.uint8,
            mode="w+",
            shape=(group_total_rows, embedding_dim), # type: ignore
        )  # type: ignore
        chunks.append(memmap_array)
        chunk_global_start = global_index
        chunk_global_end = global_index + group_total_rows
        chunk_boundaries.append((chunk_global_start, chunk_global_end))
        parts = []
        offset = 0
        for file_info in group:
            print(
                f"Copying data from file {file_info['file_stem']} into chunk {chunk_idx}"
            )
            arr = np.load(file_info["file_path"], mmap_mode="r")
            file_rows = file_info["rows"]
            memmap_array[offset : offset + file_rows] = arr[:file_rows]
            parts.append(
                {
                    "source_stem": file_info["file_stem"],
                    "parquet_file": file_info["file_stem"] + ".parquet",
                    "chunk_local_start": offset,
                    "chunk_local_end": offset + file_rows,
                    "source_local_start": 0,
                    "source_local_end": file_rows,
                }
            )
            print(
                f"Copied {file_rows} rows from {file_info['file_stem']} into chunk at offset {offset}"
            )
            offset += file_rows
            del arr
        memmap_array.flush()
        chunks_metadata.append(
            {
                "chunk_file": chunk_file,
                "global_start": chunk_global_start,
                "global_end": chunk_global_end,
                "actual_rows": group_total_rows,
                "parts": parts,
            }
        )
        print(
            f"Finished creating chunk {chunk_idx} with boundaries ({chunk_global_start}, {chunk_global_end})"
        )
        global_index += group_total_rows
        chunk_idx += 1
    metadata = {
        "total_rows": total_rows,
        "embedding_dim": embedding_dim,
        "chunk_size_bytes": chunk_size_bytes,
        "rows_per_chunk": rows_per_chunk,
        "chunks": chunks_metadata,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    STATUS.info(
        "Metadata saved successfully and chunked embeddings creation completed."
    )
    return ChunkedEmbeddings(
        chunks, chunk_boundaries, total_rows, embedding_dim # type: ignore
    ), metadata  # type: ignore


def perform_semantic_search_chunks(
    query_embedding: str,
    model,  # model not used
    metadata: dict,
    chunk_dir: str,
    folder: str,
    precision: str = "ubinary",
    top_k: int = 50,
    corpus_index=None,
):
    """
    Execute semantic search on chunked embeddings.
    For each chunk, the corresponding memmap is loaded and searched via FAISS.
    Local indices are adjusted to global indices.
    For each hit, the embedding vector is also recorded so that the final DataFrame
    will include an "embedding" column.
    """
    all_hits = []
    for chunk_info in metadata["chunks"]:
        chunk_file = chunk_info["chunk_file"]
        actual_rows = chunk_info["actual_rows"]
        global_start = chunk_info["global_start"]
        chunk_path = os.path.join(chunk_dir, chunk_file)
        print(
            f"Processing chunk {chunk_file}: global indices {global_start} to {chunk_info['global_end']}, rows={actual_rows}"
        )
        try:
            chunk_data = np.memmap(
                chunk_path,
                dtype=np.uint8,
                mode="r",
                shape=(actual_rows, metadata["embedding_dim"]),
            )
        except Exception as e:
            print(f"Error: Failed to load chunk {chunk_file}: {e}")
            continue
        chunk_results, _, _ = semantic_search_faiss(
            query_embedding,  # type: ignore
            corpus_index=corpus_index,
            corpus_embeddings=chunk_data if corpus_index is None else None,
            corpus_precision=precision,  # type: ignore
            top_k=top_k,
            calibration_embeddings=None,
            rescore=False,
            rescore_multiplier=4,
            exact=True,
            output_index=True,
        )
        if len(chunk_results) > 0:
            hits = chunk_results[0]
            for hit in hits:
                global_idx = global_start + hit["corpus_id"]
                # Retrieve and copy the embedding vector from the chunk.
                embedding_vec = np.array(chunk_data[hit["corpus_id"]])  # type: ignore
                all_hits.append(
                    {
                        "corpus_id": global_idx,
                        "score": hit["score"],
                        "embedding": embedding_vec,
                    }
                )
        del chunk_data
    if not all_hits:
        STATUS.info("Warning: No search results found across all chunks.")
        return [], pd.DataFrame()

    all_hits.sort(key=lambda x: x["score"])
    top_hits = all_hits[:top_k]
    print(f"Total merged results: {len(all_hits)}; Top {top_k} selected.")
    global_indices = [hit["corpus_id"] for hit in top_hits]
    data = load_data_for_indices(global_indices, metadata, folder=folder)
    # Add score and embedding columns to the data.
    top_hits_df = pd.DataFrame(top_hits)
    data["score"] = top_hits_df["score"].values
    data["embedding"] = top_hits_df["embedding"].values
    print("Score and embedding columns added to data.")
    print("Final search results:")
    return top_hits, data


def report_dates_from_metadata(metadata_file: str) -> dict:
    """
    Loads metadata from the specified JSON file and extracts:
      - The second date from the second part's source_stem.
      - The second date from the last part's source_stem (most recent update).

    The dates are assumed to be in six-digit YYMMDD format within the source_stem.
    The function converts them to YYYY-MM-DD format (by prefixing the year with "20")
    and prints the results.
    """

    def extract_dates(source_stem: str) -> list:
        return re.findall(r"(\d{6})", source_stem)

    def format_date(date_str: str) -> str:
        if len(date_str) != 6:
            return date_str
        return f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:]}"

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    result = {"second_date_from_second_part": None, "second_date_from_last_part": None}

    if not metadata.get("chunks"):
        print("No chunks found in metadata.")
        return result

    chunk = metadata["chunks"][0]
    parts = chunk.get("parts", [])

    if len(parts) >= 2 and isinstance(parts[1], dict):
        source_stem = parts[1].get("source_stem", "")
        dates = extract_dates(source_stem)
        if len(dates) >= 2:
            result["second_date_from_second_part"] = format_date(dates[1])  # type: ignore
        else:
            print("Not enough date strings found in the second part's source_stem.")
    else:
        print("The metadata does not contain a valid second part.")

    if parts and isinstance(parts[-1], dict):
        source_stem_last = parts[-1].get("source_stem", "")
        dates_last = extract_dates(source_stem_last)
        if len(dates_last) >= 2:
            result["second_date_from_last_part"] = format_date(dates_last[1])  # type: ignore
        else:
            print("Not enough date strings found in the last part's source_stem.")
    else:
        print("The metadata does not contain a valid last part.")

    return result


###############################################################################
# BIORXIV POST-PROCESSING
###############################################################################
def reformat_biorxiv_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reformat a BioRxiv DataFrame so its columns match the PubMed DataFrame.
    Renames "server" to "journal" and ensures all desired columns exist.
    Also ensures the "embedding" column is present.
    """
    df = df.copy()
    if "server" in df.columns:
        df.rename(columns={"server": "journal"}, inplace=True)
    required_cols = [
        "doi",
        "title",
        "authors",
        "date",
        "version",
        "type",
        "journal",
        "abstract",
        "score",
        "embedding",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    df = df[required_cols]
    return df


###############################################################################
# COMMON: LOAD SENTENCE TRANSFORMER MODEL (Not used in this REST API approach)
###############################################################################
def load_model() -> SentenceTransformer:
    """
    Load the SentenceTransformer model and send it to the appropriate device.
    (This function is retained for reference; the REST API method is now used.)
    """
    print("Loading SentenceTransformer model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    model.to(device)
    print("Model loaded and moved to the appropriate device.")
    return model


###############################################################################
# ADVANCED RELATIONSHIP PLOT FUNCTION (EMBEDDING NETWORK)
###############################################################################
def plot_embedding_network_advanced(
    query: str, query_embedding, final_results: pd.DataFrame, metric: str = "hamming"
) -> go.Figure:
    """
    Create a 2D plot where the distances between the query and found abstracts
    reflect pairwise similarities derived from the binary embeddings.
    The marker size reflects the number of citations.
    """
    if "embedding" not in final_results.columns:
        st.error("Embeddings not found in final results.")
        return go.Figure()

    # Ensure citations column exists; compute if necessary.
    if "citations" not in final_results.columns:
        final_results["citations"] = final_results["doi"].apply(
            lambda doi: get_citation_count(doi) if pd.notnull(doi) else 0
        )

    # Compute marker sizes using a logarithmic transformation.
    marker_sizes = (
        np.log1p(final_results["citations"]) * 5
    )  # Adjust the factor as needed.
    # Set query marker size as a constant.
    query_marker_size = 15

    paper_embeddings = np.stack(final_results["embedding"].values)  # type: ignore
    combined_embeddings = np.vstack([query_embedding, paper_embeddings])

    distances = pairwise_distances(combined_embeddings, metric=metric)

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords = mds.fit_transform(distances)

    labels = ["Query"] + final_results["title"].tolist()
    types = ["Query"] + ["Paper"] * len(final_results)
    # Create marker sizes array (first element for query, rest from citations).
    marker_size_arr = np.concatenate(([query_marker_size], marker_sizes.values))  # type: ignore
    # For hover data, include citations (for query set as None).
    citations_arr = np.concatenate(([0], final_results["citations"].values))  # type: ignore

    min_size = 3  # minimum marker size
    factor = 5
    # Ensure citations_arr is numerical (e.g., an array of citation counts)
    marker_size_arr = np.log1p(citations_arr.astype(float)) * factor + min_size

    df_plot = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "title": labels,
            "type": types,
            "marker_size": marker_size_arr,
            "citations": citations_arr,  # actual number of citations for hover info
        }
    )

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="type",
        size="marker_size",
        size_max=30,
        hover_data={"title": True, "citations": True, "x": False, "y": False},
        title="2D Relationship Plot (Marker size ~ citations)",
    )

    fig.update_traces(
        marker=dict(line=dict(width=1, color="DarkSlateGrey")),
        hovertemplate="<b>%{customdata[0]}</b><br>Citations: %{customdata[1]}<extra></extra>",
    )

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        title_font=dict(size=18, color="black"),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    # Optionally, draw dotted lines from the query point to each paper.
    for i in range(1, len(df_plot)):
        fig.add_shape(
            type="line",
            x0=df_plot.iloc[0]["x"],
            y0=df_plot.iloc[0]["y"],
            x1=df_plot.iloc[i]["x"],
            y1=df_plot.iloc[i]["y"],
            line=dict(color="gray", width=1, dash="dot"),
        )

    return fig


###############################################################################
# COMBINED SEARCH FUNCTION (Using REST API for Query Embedding)
###############################################################################
def combined_search(
    query: str, configs: list, top_show: int = 10, precision: str = "ubinary"
):
    """
    Execute a semantic search on PubMed, BioRxiv, and MedRxiv using the REST API
    to encode the query. Ensures that the query embedding is a 2D NumPy array with
    dtype np.uint8.
    """
    message_holder = STATUS
    message_holder.info("Encoding query via REST API...")
    # Get query embedding from the REST API.
    query_embedding = get_query_embedding(query, normalize=True, precision=precision)
    query_embedding = np.array(query_embedding, dtype=np.uint8)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding[np.newaxis, :]
    st.session_state["query_embedding"] = query_embedding

    # --- PubMed ---
    message_holder.info("Loading PubMed chunked embeddings metadata...")
    pubmed_chunk_obj, pubmed_metadata = create_chunked_embeddings_memmap(
        embeddings_directory=pubmed_config["embeddings_directory"],
        npy_files_pattern=pubmed_config["npy_files_pattern"],
        chunk_dir=pubmed_config["chunk_dir"],
        metadata_path=pubmed_config["metadata_path"],
        chunk_size_bytes=pubmed_config.get("chunk_size_bytes", 1 << 30),
    )

    # --- BioRxiv ---
    message_holder.info("Loading BioRxiv chunked embeddings metadata...")
    biorxiv_chunk_obj, biorxiv_metadata = create_chunked_embeddings_memmap(
        embeddings_directory=biorxiv_config["embeddings_directory"],
        npy_files_pattern=biorxiv_config["npy_files_pattern"],
        chunk_dir=biorxiv_config["chunk_dir"],
        metadata_path=biorxiv_config["metadata_path"],
        chunk_size_bytes=biorxiv_config.get("chunk_size_bytes", 1 << 30),
    )

    # --- MedRxiv ---
    message_holder.info("Loading MedRxiv chunked embeddings metadata...")
    medrxiv_chunk_obj, medrxiv_metadata = create_chunked_embeddings_memmap(
        embeddings_directory=medrxiv_config["embeddings_directory"],
        npy_files_pattern=medrxiv_config["npy_files_pattern"],
        chunk_dir=medrxiv_config["chunk_dir"],
        metadata_path=medrxiv_config["metadata_path"],
        chunk_size_bytes=medrxiv_config.get("chunk_size_bytes", 1 << 30),
    )

    top_k = top_show * 3  # extra results to remove duplicates

    STATUS.info("Performing PubMed semantic search...")
    pubmed_results, pubmed_df = perform_semantic_search_chunks(
        query_embedding,  # type: ignore
        None,
        pubmed_metadata,
        chunk_dir=pubmed_config["chunk_dir"],
        folder=pubmed_config["data_folder"],
        precision=precision,
        top_k=top_k,
    )
    pubmed_df["source"] = "PubMed"

    STATUS.info("Performing BioRxiv semantic search...")
    biorxiv_results, biorxiv_df = perform_semantic_search_chunks(
        query_embedding,  # type: ignore
        None,
        biorxiv_metadata,
        chunk_dir=biorxiv_config["chunk_dir"],
        folder=biorxiv_config["data_folder"],
        precision=precision,
        top_k=top_k,
    )

    STATUS.info("Performing MedRxiv semantic search...")
    medrxiv_results, medrxiv_df = perform_semantic_search_chunks(
        query_embedding,  # type: ignore
        None,
        medrxiv_metadata,
        chunk_dir=medrxiv_config["chunk_dir"],
        folder=medrxiv_config["data_folder"],
        precision=precision,
        top_k=top_k,
    )

    print("#######################")
    biorxiv_df = reformat_biorxiv_df(biorxiv_df)
    biorxiv_df["source"] = "BioRxiv"

    medrxiv_df = reformat_biorxiv_df(medrxiv_df)
    medrxiv_df["source"] = "MedRxiv"

    combined_df = pd.concat([pubmed_df, biorxiv_df, medrxiv_df], ignore_index=True)
    raw_min = combined_df["score"].min()
    raw_max = combined_df["score"].max()
    combined_df["quality"] = abs(combined_df["score"] - raw_max) + raw_min
    combined_df.sort_values(by="score", inplace=True)

    if combined_df["doi"].notnull().all():
        combined_df = combined_df.drop_duplicates(subset=["doi"], keep="first")
    else:
        combined_df = combined_df.drop_duplicates(subset=["title"], keep="first")

    final_results = combined_df.head(top_show).reset_index(drop=True)
    message_holder.empty()
    print("Combined search results:")
    print(final_results)

    # --- Cleanup heavy objects to free RAM ---
    pubmed_chunk_obj.close()
    biorxiv_chunk_obj.close()
    medrxiv_chunk_obj.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_results


###############################################################################
# STYLE AND LOGO FUNCTIONS
###############################################################################
def define_style():
    st.markdown(
        """
        <style>
            .stExpander > .stButton > button {
                width: 100%;
                border: none;
                background-color: #f0f2f6;
                color: #333;
                text-align: left;
                padding: 15px;
                font-size: 18px;
                border-radius: 10px;
                margin-top: 5px;
            }
            .stExpander > .stExpanderContent {
                padding-left: 10px;
                padding-top: 10px;
            }
            a {
                color: #FF4B4B;
                text-decoration: none;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def logo(db_update_date, db_size_bio, db_size_pubmed, db_size_med):
    biorxiv_logo = (
        "https://www.biorxiv.org/sites/default/files/biorxiv_logo_homepage.png"
    )
    medarxiv_logo = (
        "https://www.medrxiv.org/sites/default/files/medRxiv_homepage_logo.png"
    )
    pubmed_logo = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/US-NLM-PubMed-Logo.svg/720px-US-NLM-PubMed-Logo.svg.png?20080121063734"

    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <div style="display: flex; justify-content: center; align-items: center; gap: 30px;">
                <img src="{pubmed_logo}" alt="PubMed logo" style="max-height: 80px; object-fit: contain;">
                <img src="{biorxiv_logo}" alt="BioRxiv logo" style="max-height: 80px; object-fit: contain;">
                <img src="{medarxiv_logo}" alt="medRxiv logo" style="max-height: 80px; object-fit: contain;">
            </div>
            <div style="text-align: center; margin-top: 10px;">
                <h3 style="color: black; margin: 0; font-weight: 400;">Manuscript Semantic Search [MSS]</h3>
                <p style="font-size: 14px; color: #555; margin: 5px 0 0 0;">
                    Last database update: {db_update_date}<br>
                    Database size: PubMed: {int(db_size_pubmed):,} entries / BioRxiv: {int(db_size_bio):,} / MedRxiv: {int(db_size_med):,}
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


###############################################################################
# AI SUMMARY FUNCTIONS
###############################################################################

LLM_prompt = """You are a research assistant tasked with summarizing a collection of abstracts extracted from a database of 39 million academic entries. Your goal is to synthesize a concise, clear, and insightful summary that captures the main themes, common findings, and noteworthy trends across the abstracts. 

Instructions:
- Read the provided abstracts carefully.
- Shortly digest each abstract's content.
- Identify and list the central topics and findings.
- Highlight any recurring patterns or shared insights.
- Keep the summary succinct (1-3 short paragraphs) without using overly technical jargon.
- Do not include any external links or references.
- Format the response in markdown, using bullet points where appropriate.

Now, review the abstracts provided below and generate your summary.
"""


def summarize_abstract(
    abstracts, instructions, api_key, model_name="gemini-2.0-flash-lite-preview-02-05"
):
    """
    Summarizes the provided abstracts using Google's Gemini Flash model via the new genai library.
    """
    from google.genai import types

    if not api_key:
        return "API key not provided. Please obtain your own API key at https://aistudio.google.com/apikey"
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    formatted_text = "\n".join(
        f"{idx + 1}. {abstract}" for idx, abstract in enumerate(abstracts)
    )
    prompt = f"{instructions}\n\n{formatted_text}"
    content_part = types.Part.from_text(text=prompt)
    config = types.GenerateContentConfig(
        temperature=1, top_p=0.95, top_k=64, max_output_tokens=8192
    )
    try:
        response = client.models.generate_content(
            model=model_name, contents=content_part, config=config
        )
        summary = response.text
    except Exception as e:
        summary = f"Google Flash model not available or usage limit exceeded: {e}"
    return summary


###############################################################################
# STREAMLIT APP CODE
###############################################################################

# Load configurations from config_mss.yaml.
with open("config_mss.yaml", "r") as f:
    config_data = yaml.safe_load(f)

pubmed_config = config_data["pubmed_config"]
biorxiv_config = config_data["biorxiv_config"]
medrxiv_config = config_data["medrxiv_config"]

configs = [pubmed_config, biorxiv_config, medrxiv_config]

# Retrieve database sizes from JSON metadata files.
try:
    with open(pubmed_config["metadata_path"], "r") as f:
        pubmed_meta = json.load(f)
    pubmed_db_size = pubmed_meta.get("total_rows", "N/A")
except Exception as e:
    pubmed_db_size = 0
try:
    with open(biorxiv_config["metadata_path"], "r") as f:
        biorxiv_meta = json.load(f)
    biorxiv_db_size = biorxiv_meta.get("total_rows", "N/A")
except Exception as e:
    biorxiv_db_size = 0
try:
    with open(medrxiv_config["metadata_path"], "r") as f:
        medrxiv_meta = json.load(f)
    medrxiv_db_size = medrxiv_meta.get("total_rows", "N/A")
except Exception as e:
    medrxiv_db_size = 0

define_style()

# Show the most recent date.
try:
    last_biorxiv_date = report_dates_from_metadata(biorxiv_config["metadata_path"]).get(
        "second_date_from_last_part", "N/A"
    )
except:
    last_biorxiv_date = "N/A"

logo(last_biorxiv_date, biorxiv_db_size, pubmed_db_size, medrxiv_db_size)

# -----------------------------------------------------------------------------
# Form and search input.
# -----------------------------------------------------------------------------

use_ai = False

with st.form("search_form"):
    query = st.text_input("Enter your search query:", max_chars=8192)
    col1, col2 = st.columns(2)
    with col1:
        num_to_show = st.number_input(
            "Number of results to show:", min_value=1, max_value=50, value=10
        )
    with col2:
        if st.session_state.get("use_ai_checkbox"):
            ai_api_provided = st.text_input(
                "Google AI API Key",
                value="",
                help="Obtain your own API key at https://aistudio.google.com/apikey",
                type="password",
            )
        else:
            ai_api_provided = None
    submitted = st.form_submit_button("Search 🔍")

STATUS = st.empty()
col1, col2 = st.columns(2)
use_ai = col1.checkbox("Use AI generated summary?", key="use_ai_checkbox")

if submitted and query:
    with st.spinner("Searching..."):
        search_start_time = datetime.now()
        final_results = combined_search(
            query, configs, top_show=num_to_show, precision="ubinary"
        )
        total_time = datetime.now() - search_start_time
        st.markdown(
            f"<h6 style='text-align: center; color: #7882af;'>Search completed in {total_time.total_seconds():.2f} seconds</h6>",
            unsafe_allow_html=True,
        )
        st.session_state["final_results"] = final_results
        st.session_state["search_query"] = query
        st.session_state["num_to_show"] = num_to_show
        st.session_state["use_ai"] = st.session_state.get("use_ai_checkbox", False)
        st.session_state["ai_api_provided"] = ai_api_provided
else:
    final_results = st.session_state.get("final_results", pd.DataFrame())

if not final_results.empty:
    # --- Sorting segmented control ---
    sort_option = col2.segmented_control(
        "Sort results by:",
        options=["Relevance", "Publication Date", "Citations"],
        key="sort_option",
    )

    sorted_results = st.session_state["final_results"].copy()
    if sort_option == "Publication Date":
        sorted_results["date_parsed"] = pd.to_datetime(
            sorted_results["date"], errors="coerce"
        )
        sorted_results = sorted_results.sort_values(
            by="date_parsed", ascending=False
        ).reset_index(drop=True)
        sorted_results.drop(columns=["date_parsed"], inplace=True)
    elif sort_option == "Citations":
        sorted_results["citations"] = sorted_results["doi"].apply(
            lambda doi: get_citation_count(doi) if pd.notnull(doi) else 0
        )
        sorted_results = sorted_results.sort_values(
            by="citations", ascending=False
        ).reset_index(drop=True)
    else:
        sorted_results = sorted_results.sort_values(
            by="score", ascending=True
        ).reset_index(drop=True)

    abstracts_for_summary = []
    for idx, row in sorted_results.iterrows():
        # Compute citation count for display.
        citations = get_citation_count(row["doi"]) if pd.notnull(row["doi"]) else "N/A"
        # Modify the expander title to include score and citation information.
        expander_title = f"{idx + 1}. {row['title']}\n\n _(Score: {row['quality']:.2f} | Citations: {citations})_"
        doi_link = f"https://doi.org/{row['doi']}" if pd.notnull(row["doi"]) else "#"
        with st.expander(expander_title):
            col_a, col_b, col_c = st.columns(3)
            col_a.markdown(f"**Relative Score:** {row['quality']:.2f}")
            col_b.markdown(f"**Source:** {row['source']}")
            col_c.markdown(f"**Citations:** {citations}")
            st.markdown(f"**Authors:** {row['authors']}")
            col_d, col_e = st.columns(2)
            col_d.markdown(f"**Date:** {row['date']}")
            col_e.markdown(f"**Journal/Server:** {row.get('journal', 'N/A')}")
            abstracts_for_summary.append(row["abstract"])
            st.markdown(f"**Abstract:**\n{row['abstract']}")
            st.markdown(f"**[Full Text Read]({doi_link})** 🔗")

            # Checkbox for showing references.
            # show_refs = st.checkbox("Show References", key=f"show_refs_{idx}")
            # if show_refs:
            #     if pd.notnull(row['doi']):
            #         refs = get_references(row['doi'])
            #         if refs:
            #             for ref in refs:
            #                 st.markdown(f"- {ref}")
            #         else:
            #             st.markdown("No references available.")
            #     else:
            #         st.markdown("DOI not available.")

    # --- Create plots in tabs ---
    try:
        sorted_results["Date"] = pd.to_datetime(sorted_results["date"])
        if "citations" not in sorted_results.columns:
            sorted_results["citations"] = sorted_results["doi"].apply(
                lambda doi: get_citation_count(doi) if pd.notnull(doi) else 0
            )
        plot_data = {
            "Date": sorted_results["Date"],
            "Title": sorted_results["title"],
            "Relative Score": sorted_results["quality"],
            "DOI": sorted_results["doi"],
            "Source": sorted_results["source"],
            "citations": sorted_results["citations"],
        }
        plot_df = pd.DataFrame(plot_data)
        # Use np.log1p and scale plus add a minimum size so that even zero citations produce a visible dot.
        plot_df["marker_size"] = np.log1p(plot_df["citations"]) * 5 + 3

        fig_scatter = px.scatter(
            plot_df,
            x="Date",
            y="Relative Score",
            size="marker_size",
            hover_data={
                "Title": True,
                "DOI": True,
                "citations": True,
                "marker_size": False,
            },
            color="Source",
            title="Publication Dates and Relative Score (Marker size ~ citations)",
        )
        fig_scatter.update_layout(legend=dict(title="Source"))
    except Exception as e:
        st.error(f"Error in plotting Score vs Year: {str(e)}")

    # if "query_embedding" in st.session_state:
    #     fig_network = plot_embedding_network_advanced(st.session_state["search_query"],
    #                                                   st.session_state["query_embedding"],
    #                                                   sorted_results)

    tabs = st.tabs(["Score vs Year", "Abstract Map"])
    with tabs[0]:
        st.plotly_chart(fig_scatter, use_container_width=True)
    # with tabs[1]:
    #     st.plotly_chart(fig_network, use_container_width=True)
    with tabs[1]:
        # Load the saved 2D histogram data.
        try:
            hist_data = np.load("hist2d.npz")
            hist = hist_data["hist"]
            xedges = hist_data["xedges"]
            yedges = hist_data["yedges"]
        except Exception as e:
            st.error(f"Error loading histogram data: {e}")

        # Determine the axis ranges from the bin edges.
        x_min, x_max = float(xedges[0]), float(xedges[-1])
        y_min, y_max = float(yedges[0]), float(yedges[-1])

        # Create a heatmap trace using the histogram.
        heatmap = go.Heatmap(
            z=np.sqrt(
                hist.T
            ),  # Transpose so that the axes match the coordinate system.
            x=xedges,
            y=yedges,
            colorscale="Blues",
            reversescale=False,
            showscale=False,
            opacity=1,
        )

        # Instead of using stored 2D embeddings, recalc the 2D coordinates from raw embeddings.
        # It is assumed that final_results has a column "embedding" containing each abstract’s raw 128-dimensional vector.
        if "embedding" in final_results.columns:
            raw_embeddings = np.stack(final_results["embedding"].values)  # type: ignore # shape: (n, 128)
            # Convert raw embeddings to a torch.Tensor.
            raw_embeddings_tensor = torch.from_numpy(raw_embeddings).float()

            pumap = load_pumap("param_umap_model.h5")
            # Use the pre-trained PUMAP model to compute new 2D locations.
            new_2d = pumap.transform(raw_embeddings_tensor)  # shape: (n, 2)
            scatter_x = new_2d[:, 0]
            scatter_y = new_2d[:, 1]
        else:
            st.error(
                "final_results does not contain an 'embedding' column with high-dimensional embeddings."
            )
            scatter_x, scatter_y = [], []

        # Create a scatter trace for the found abstracts.
        scatter = go.Scatter(
            x=scatter_x,
            y=scatter_y,
            mode="markers",
            marker=dict(
                color="#2a7aba",
                size=12,
                line=dict(width=1, color="black"),
                opacity=1,
            ),
            text=final_results.apply(
                lambda row: (
                    f"<b>{row['title']}</b><br>"
                    f"Date: {row['date']}<br>"
                    f"Authors: {row['authors']}<br>"
                    f"Score: {row['score']:.2f}"
                ),
                axis=1,
            ),
            hoverinfo="text",
            name="Found Abstracts",
        )

        # Create the final Plotly figure combining the heatmap and scatter.
        fig_abstract_map = go.Figure(data=[heatmap, scatter])
        fig_abstract_map.update_layout(
            title="Abstract Map with 2D Embeddings",
            xaxis=dict(
                title="Component 1",
                range=[x_min, x_max],
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                title="Component 2",
                range=[y_min, y_max],
                showgrid=False,
                zeroline=False,
            ),
            plot_bgcolor="white",
            # change hight and width of the plot
            height=600,
            width=800,
        )

        st.plotly_chart(fig_abstract_map, use_container_width=True)

    if st.session_state.get("use_ai") and abstracts_for_summary:
        with st.spinner("Generating AI summary..."):
            ai_gen_start = time.time()
            if st.session_state.get("ai_api_provided"):
                st.markdown("**AI Summary of abstracts:**")
                summary_text = summarize_abstract(
                    abstracts_for_summary[:9],
                    LLM_prompt,
                    st.session_state["ai_api_provided"],
                )
                st.markdown(summary_text)
            total_ai_time = time.time() - ai_gen_start
            st.markdown(f"**Time to generate summary:** {total_ai_time:.2f} seconds")
st.markdown(
    """
    <div style='text-align: center;'>
        <b>Developed by <a href="https://www.dzyla.com/" target="_blank">Dawid Zyla</a></b>
        <br>
        <a href="https://github.com/dzyla/pubmed_search" target="_blank">Source code on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("---")
cost_monthly = 50

c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 10px;">
            <h3 style="margin-bottom: 5px; font-weight: normal;">Support MSS Server</h3>
            <p style="font-size: 14px; color: #555;">
                The monthly server cost is approximately ${cost_monthly}.
                Any donation helps me keep the service running.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    donation_collected = 0.0
    target_amount = 50
    progress_fraction = donation_collected / target_amount
    st.progress(progress_fraction)
    st.markdown(
        f"""
        <p style="text-align: center; font-size: 14px; color: #555; margin-bottom: 10px;">
            ${donation_collected:.2f} raised of ${target_amount:.2f}
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="text-align: center; margin-top: 5px;">
            <a href="https://www.buymeacoffee.com/dzyla" target="_blank" 
               style="
                    background-color: #3679ae;
                    color: #ffffff;
                    padding: 10px 20px;
                    border-radius: 5px;
                    text-decoration: none;
                    font-family: Lato, sans-serif;
                    font-size: 16px;
                    box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
                    transition: background-color 0.2s ease;
               "
               onMouseOver="this.style.backgroundColor='#26557b'"
               onMouseOut="this.style.backgroundColor='#26557b'">
                Buy me a coffee
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
