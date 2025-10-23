#!/usr/bin/env python3
"""
Prewarm chunked memmap indices for MSS.

This script scans configured embedding .npy files, builds chunked memmaps
and writes a metadata JSON compatible with pbmss_app.py expectations.

Run:
  python scripts/prewarm_chunks.py --config config_mss.yaml [--source pubmed|biorxiv|medrxiv|arxiv] [--force]
"""
import argparse
import concurrent.futures
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np

try:
    import yaml  # type: ignore
except Exception as e:
    raise SystemExit("pyyaml is required for prewarm; please install it (pip install pyyaml)")


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
LOGGER = logging.getLogger("prewarm")


def copy_file_into_chunk(file_info: Dict[str, Any], memmap_array: np.memmap, offset: int) -> Dict[str, Any]:
    LOGGER.info(f"Copying data from file {file_info['file_stem']} into chunk at offset {offset}")
    arr = np.load(file_info["file_path"], mmap_mode="r")
    file_rows = file_info["rows"]
    memmap_array[offset: offset + file_rows] = arr[:file_rows]
    LOGGER.info(f"Copied {file_rows} rows from {file_info['file_stem']} into chunk at offset {offset}")
    return {
        "source_stem": file_info["file_stem"],
        "parquet_file": file_info["file_stem"] + ".parquet",
        "chunk_local_start": offset,
        "chunk_local_end": offset + file_rows,
        "source_local_start": 0,
        "source_local_end": file_rows,
    }


def create_chunked_embeddings_memmap(
    embeddings_directory: str,
    npy_files_pattern: str,
    chunk_dir: str,
    metadata_path: str,
    chunk_size_bytes: int = 1 << 30,
    force: bool = False,
) -> Dict[str, Any]:
    """Create or load chunked memmaps and return metadata dict.

    Matches pbmss_app.py's metadata schema and dtype assumptions (uint8 embeddings).
    """
    recreate = force
    if os.path.exists(metadata_path) and not force:
        LOGGER.info(f"Metadata file found at {metadata_path}. Validating chunks...")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        for chunk_info in metadata.get("chunks", []):
            chunk_file = chunk_info["chunk_file"]
            chunk_path = os.path.join(chunk_dir, chunk_file)
            if not os.path.exists(chunk_path):
                LOGGER.warning(f"Missing chunk file {chunk_file}; will recreate chunks.")
                recreate = True
                break
        if not recreate:
            metadata_npy_files = set()
            for chunk_info in metadata["chunks"]:
                for part in chunk_info["parts"]:
                    metadata_npy_files.add(part["source_stem"])
            current_npy_files = set(f.stem for f in Path(embeddings_directory).glob(npy_files_pattern))
            if metadata_npy_files != current_npy_files:
                LOGGER.warning("Mismatch in npy files between metadata and current directory; will recreate chunks.")
                recreate = True
        if not recreate:
            LOGGER.info("Chunk files and mappings are valid; nothing to do.")
            return metadata
        else:
            LOGGER.info("Recreating chunks: removing old metadata and chunk files...")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            for file in Path(chunk_dir).glob("chunk_*.npy"):
                try:
                    os.remove(file)
                except Exception:
                    pass
    else:
        if force:
            LOGGER.info("Force enabled; will rebuild chunks from scratch.")
        else:
            LOGGER.info(f"No metadata found at {metadata_path}; will create new chunked embeddings.")

    os.makedirs(chunk_dir, exist_ok=True)
    npy_files = sorted(list(Path(embeddings_directory).glob(npy_files_pattern)))
    if not npy_files:
        raise FileNotFoundError(f"No npy files found in {embeddings_directory} with pattern {npy_files_pattern}")

    total_rows = 0
    embedding_dim = None
    file_infos = []

    # Inspect files to determine shape and total rows
    for fname in npy_files:
        arr = np.load(fname, mmap_mode="r")
        try:
            rows, dims = arr.shape
        except ValueError:
            try:
                rows, _, dims = arr.shape
            except ValueError:
                raise ValueError(f"Invalid shape for file {fname}: {arr.shape}")
        if embedding_dim is None:
            embedding_dim = dims
        elif dims != embedding_dim:
            raise ValueError(f"Embedding dimension mismatch in {fname}")
        file_infos.append({"file_stem": fname.stem, "rows": rows, "file_path": str(fname)})
        total_rows += rows
        del arr

    # rows_per_chunk assumes uint8 embeddings (1 byte per dim)
    rows_per_chunk = chunk_size_bytes // embedding_dim  # type: ignore[arg-type]

    # Group files into chunks without splitting individual files
    chunk_groups = []
    current_chunk_files = []
    current_chunk_rows = 0
    for file_info in file_infos:
        file_rows = file_info["rows"]
        if file_rows > rows_per_chunk:
            if current_chunk_files:
                chunk_groups.append(current_chunk_files)
                current_chunk_files = []
                current_chunk_rows = 0
            chunk_groups.append([file_info])
        else:
            if current_chunk_rows + file_rows > rows_per_chunk and current_chunk_files:
                chunk_groups.append(current_chunk_files)
                current_chunk_files = []
                current_chunk_rows = 0
            current_chunk_files.append(file_info)
            current_chunk_rows += file_rows
    if current_chunk_files:
        chunk_groups.append(current_chunk_files)

    chunks_metadata = []
    global_index = 0

    for chunk_idx, group in enumerate(chunk_groups):
        group_total_rows = sum(file_info["rows"] for file_info in group)
        chunk_file = f"chunk_{chunk_idx}.npy"
        chunk_path = os.path.join(chunk_dir, chunk_file)
        memmap_array = np.memmap(chunk_path, dtype=np.uint8, mode="w+", shape=(group_total_rows, embedding_dim))  # type: ignore[arg-type]
        # Offsets per source file within the chunk
        offsets = []
        current_offset = 0
        for file_info in group:
            offsets.append(current_offset)
            current_offset += file_info["rows"]
        parts = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(copy_file_into_chunk, file_info, memmap_array, off) for file_info, off in zip(group, offsets)]
            for future in concurrent.futures.as_completed(futures):
                parts.append(future.result())
        memmap_array.flush()
        chunks_metadata.append({
            "chunk_file": chunk_file,
            "global_start": global_index,
            "global_end": global_index + group_total_rows,
            "actual_rows": group_total_rows,
            "parts": parts,
        })
        global_index += group_total_rows

    metadata = {
        "total_rows": total_rows,
        "embedding_dim": embedding_dim,
        "chunk_size_bytes": int(chunk_size_bytes),
        "rows_per_chunk": int(rows_per_chunk),
        "chunks": chunks_metadata,
    }
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    LOGGER.info(f"Wrote metadata to {metadata_path}")
    return metadata


def prewarm_from_config(config_path: str, source: str | None, force: bool = False, chunk_size_bytes: int | None = None) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # The config file uses top-level keys for each source in pbmss_app
    sources = []
    if source:
        sources = [source]
    else:
        for key in ["pubmed_config", "biorxiv_config", "medrxiv_config", "arxiv_config"]:
            if key in cfg:
                sources.append(key.replace("_config", ""))

    for src in sources:
        key = f"{src}_config"
        if key not in cfg:
            LOGGER.warning(f"No config found for {src}; skipping")
            continue
        c = cfg[key]
        LOGGER.info(f"Prewarming {src} chunks...")
        create_chunked_embeddings_memmap(
            embeddings_directory=c["embeddings_directory"],
            npy_files_pattern=c.get("npy_files_pattern", "*.npy"),
            chunk_dir=c["chunk_dir"],
            metadata_path=c["metadata_path"],
            chunk_size_bytes=chunk_size_bytes or c.get("chunk_size_bytes", 1 << 30),
            force=force,
        )


def main():
    parser = argparse.ArgumentParser(description="Prewarm MSS chunked memmaps")
    parser.add_argument("--config", default="config_mss.yaml", help="Path to MSS YAML config")
    parser.add_argument("--source", choices=["pubmed", "biorxiv", "medrxiv", "arxiv"], help="Limit to a single source", default=None)
    parser.add_argument("--force", action="store_true", help="Force rebuild of chunks/metadata")
    parser.add_argument("--chunk-size-bytes", type=int, default=None, help="Override chunk size in bytes")
    args = parser.parse_args()

    prewarm_from_config(args.config, args.source, force=args.force, chunk_size_bytes=args.chunk_size_bytes)


if __name__ == "__main__":
    main()
