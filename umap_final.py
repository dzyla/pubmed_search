import os
import json
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import umap
from pathlib import Path
from tqdm import tqdm
import plotly.graph_objects as go
from glob import glob

# Import your existing custom modules
import config_loader 
import data_handler

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("SemanticMap")

CONFIG_PATH = "config_mss.yaml"
ASSETS_DIR = "./visualization_assets"
SAMPLE_SIZE = 100_000 
BATCH_SIZE = 4096
EPOCHS = 500
LR = 0.001
GRID_RES = 600
ABSTRACT_MIN_CHAR = 75 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def unpack_bits_batch(packed_batch, target_float_type=np.float32):
    unpacked = np.unpackbits(packed_batch, axis=1)
    return unpacked.astype(target_float_type)

def get_valid_indices_for_chunk(chunk_meta, data_folder, combined_file=None):
    """
    Returns 0-BASED INDICES strictly for the provided chunk file.
    Assumes 1-to-1 mapping: Row 0 of chunk text file == Row 0 of chunk embedding file.
    """
    # 1. Identify the specific text file for this chunk
    filename = (
        chunk_meta.get("filename") or 
        chunk_meta.get("json_file") or 
        chunk_meta.get("file")
    )
    
    file_path = None
    is_combined_fallback = False

    # Priority: Specific chunk file > Global combined file
    if filename:
        candidate = os.path.join(data_folder, filename)
        if os.path.exists(candidate):
            file_path = candidate
    
    # Only fall back to combined if no specific file exists
    if not file_path and combined_file and os.path.exists(combined_file):
        file_path = combined_file
        is_combined_fallback = True
    
    if not file_path:
        return []

    valid_indices = []
    chunk_rows = chunk_meta.get("actual_rows", 0)

    try:
        ext = os.path.splitext(file_path)[1].lower()

        # --- CASE A: Parquet ---
        if ext == ".parquet":
            # If it's a specific chunk file, read it from 0
            if not is_combined_fallback:
                # Read columns needed for filter
                df = pd.read_parquet(file_path, columns=["abstract", "title"])
                
                # STRICT FILTER
                mask = (
                    df["title"].notna() & (df["title"] != "") & 
                    df["abstract"].notna() & 
                    (df["abstract"].astype(str).str.len() > ABSTRACT_MIN_CHAR)
                )
                valid_indices = np.where(mask)[0]

            else:
                # Fallback to global file (Only if individual files missing)
                # This uses the global_start offset
                start = chunk_meta.get("global_start", 0)
                # Optimization: Try to read only the row group or slice
                # Warning: This is slow if file is huge
                df = pd.read_parquet(file_path, columns=["abstract", "title"])
                # Bounds check
                if start < len(df):
                    end = min(start + chunk_rows, len(df))
                    df_slice = df.iloc[start:end]
                    
                    mask = (
                        df_slice["title"].notna() & (df_slice["title"] != "") & 
                        df_slice["abstract"].notna() & 
                        (df_slice["abstract"].astype(str).str.len() > ABSTRACT_MIN_CHAR)
                    )
                    valid_indices = np.where(mask)[0]

        # --- CASE B: JSONL ---
        else:
            # JSONL is typically 1 file per chunk
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= chunk_rows: break 
                    try:
                        row = json.loads(line)
                        abstract = row.get('abstract', '')
                        title = row.get('title', '')
                        
                        if title and abstract and len(str(abstract)) > ABSTRACT_MIN_CHAR:
                            valid_indices.append(i)
                    except:
                        continue
            valid_indices = np.array(valid_indices, dtype=np.int64)

    except Exception as e:
        LOGGER.error(f"Filter failed for {file_path}: {e}")
        return []

    # Final Safety: Clamp indices to be within the chunk size
    if len(valid_indices) > 0:
        valid_indices = valid_indices[valid_indices < chunk_rows]

    return valid_indices

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
class ParametricUMAP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128]):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Linear(in_d, h_d))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_d))
            layers.append(nn.Dropout(0.1))
            in_d = h_d
        layers.append(nn.Linear(in_d, 2))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

# -----------------------------------------------------------------------------
# STEP 1: TRAINING
# -----------------------------------------------------------------------------
def run_training(config):
    LOGGER.info("Step 1: Training - Identifying valid data...")
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    all_chunks_registry = [] 
    total_valid_rows = 0
    
    db_keys = ["pubmed_config", "biorxiv_config", "medrxiv_config", "arxiv_config"]
    
    for key in db_keys:
        if key not in config: continue
        cfg = config[key]
        if not os.path.exists(cfg["metadata_path"]): continue
        
        with open(cfg["metadata_path"], 'r') as f: meta = json.load(f)
        
        for chunk in tqdm(meta.get("chunks", []), desc=f"Scanning {key}"):
            c_path = os.path.join(cfg["chunk_dir"], chunk["chunk_file"])
            if not os.path.exists(c_path): continue
            
            valid_idxs = get_valid_indices_for_chunk(
                chunk, cfg["data_folder"], cfg.get("combined_data_file")
            )
            
            if len(valid_idxs) > 0:
                all_chunks_registry.append({
                    "path": c_path,
                    "indices": valid_idxs,
                    "dim": meta["embedding_dim"],
                    "file_size": os.path.getsize(c_path)
                })
                total_valid_rows += len(valid_idxs)

    LOGGER.info(f"Total VALID rows identified: {total_valid_rows:,}")
    if total_valid_rows == 0:
        LOGGER.error("No valid data found! Check paths.")
        return

    # Random Sampling
    collected_data = []
    rows_needed = SAMPLE_SIZE
    
    np.random.shuffle(all_chunks_registry)
    
    pbar = tqdm(total=SAMPLE_SIZE, desc="Collecting Vectors")
    
    for item in all_chunks_registry:
        if rows_needed <= 0: break
        
        n_valid = len(item["indices"])
        prob = SAMPLE_SIZE / max(total_valid_rows, 1)
        n_take = np.random.binomial(n_valid, prob)
        
        if n_take == 0: continue
        
        # Sample from VALID indices only
        chosen = np.random.choice(item["indices"], size=min(n_take, n_valid), replace=False)
        
        try:
            file_rows = item["file_size"] // item["dim"]
            chosen = chosen[chosen < file_rows] # Safety
            
            if len(chosen) == 0: continue

            data = np.memmap(item["path"], dtype=np.uint8, mode='r', 
                             shape=(file_rows, item["dim"]))
            
            subset = data[chosen]
            subset_float = unpack_bits_batch(subset)
            collected_data.append(subset_float)
            
            rows_needed -= len(subset_float)
            pbar.update(len(subset_float))
            del data
            
        except Exception as e:
            LOGGER.error(f"Read error {item['path']}: {e}")

    pbar.close()
    
    if not collected_data: raise ValueError("Collected data is empty.")
    
    X_train = np.concatenate(collected_data, axis=0)
    if len(X_train) > SAMPLE_SIZE: X_train = X_train[:SAMPLE_SIZE]
    
    LOGGER.info(f"Final Training Set: {X_train.shape}")

    # UMAP & Model
    LOGGER.info("Fitting Reference UMAP...")
    reducer = umap.UMAP(n_components=2, metric='cosine', n_neighbors=150, min_dist=0.1, verbose=True, init='pca', n_jobs=-1)
    y_train = reducer.fit_transform(X_train)
    
    LOGGER.info("Training Neural Network...")
    model = ParametricUMAP(input_dim=X_train.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 50 == 0:
            LOGGER.info(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), os.path.join(ASSETS_DIR, "param_umap_model.pth"))
    LOGGER.info("Model Saved.")

# -----------------------------------------------------------------------------
# STEP 2: PROJECTION
# -----------------------------------------------------------------------------
def run_projection(config):
    LOGGER.info("Step 2: Projection - STRICT filtering enabled...")
    model_path = os.path.join(ASSETS_DIR, "param_umap_model.pth")
    if not os.path.exists(model_path): return

    dummy_key = [k for k in config.keys() if "config" in k][0]
    with open(config[dummy_key]["metadata_path"]) as f:
        dim = json.load(f)["embedding_dim"] * 8
        
    model = ParametricUMAP(input_dim=dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    db_keys = ["pubmed_config", "biorxiv_config", "medrxiv_config", "arxiv_config"]

    for key in db_keys:
        if key not in config: continue
        LOGGER.info(f"Projecting {key}...")
        
        cfg = config[key]
        with open(cfg["metadata_path"], 'r') as f: meta = json.load(f)
        
        source_coords = []
        source_indices = []
        
        total_kept = 0
        
        for chunk in tqdm(meta.get("chunks", [])):
            c_path = os.path.join(cfg["chunk_dir"], chunk["chunk_file"])
            if not os.path.exists(c_path): continue
            
            # --- STRICT FILTER ---
            # Returns indices relative to the chunk (0..N)
            valid_indices = get_valid_indices_for_chunk(
                chunk, cfg["data_folder"], cfg.get("combined_data_file")
            )
            
            file_rows = os.path.getsize(c_path) // meta["embedding_dim"]
            
            # Safety Clamp
            if len(valid_indices) > 0:
                valid_indices = valid_indices[valid_indices < file_rows]
            
            if len(valid_indices) == 0: continue
            
            total_kept += len(valid_indices)
            
            try:
                data = np.memmap(c_path, dtype=np.uint8, mode='r', 
                                 shape=(file_rows, meta["embedding_dim"]))
                
                valid_indices = np.sort(valid_indices)
                
                chunk_proj = []
                for i in range(0, len(valid_indices), BATCH_SIZE):
                    batch_idxs = valid_indices[i : i + BATCH_SIZE]
                    # This guarantees we ONLY project valid rows
                    batch_raw = data[batch_idxs] 
                    batch_float = unpack_bits_batch(batch_raw)
                    
                    with torch.no_grad():
                        t_in = torch.from_numpy(batch_float).float().to(DEVICE)
                        t_out = model(t_in).cpu().numpy()
                        chunk_proj.append(t_out)
                
                if chunk_proj:
                    source_coords.append(np.concatenate(chunk_proj, axis=0))
                    # Store Global ID so we can look up tooltip text later
                    source_indices.append(chunk["global_start"] + valid_indices)
                    
                del data
            except Exception as e:
                LOGGER.error(f"Error processing chunk {c_path}: {e}")

        LOGGER.info(f"{key}: Total valid vectors projected: {total_kept}")

        if source_coords:
            final_coords = np.concatenate(source_coords, axis=0)
            final_indices = np.concatenate(source_indices, axis=0)
            np.save(os.path.join(ASSETS_DIR, f"projection_{key}.npy"), final_coords)
            np.save(os.path.join(ASSETS_DIR, f"indices_{key}.npy"), final_indices)

# -----------------------------------------------------------------------------
# STEP 3: VISUALIZATION (WITH BUTTONS)
# -----------------------------------------------------------------------------
def run_generation():
    LOGGER.info("Step 3: HTML Generation with Layer Controls...")
    
    LABEL_MAP = {"pubmed_config": "PubMed", "biorxiv_config": "BioRxiv", "medrxiv_config": "MedRxiv", "arxiv_config": "arXiv"}
    COLORSCALES = {"All": "Inferno", "pubmed_config": "Blues_r", "biorxiv_config": "Reds_r", "medrxiv_config": "Greens_r", "arxiv_config": "Oranges_r"}
    
    files = glob(os.path.join(ASSETS_DIR, "projection_*.npy"))
    if not files: return
    
    data_map = {}
    index_map = {}
    all_points_list = []
    
    for fpath in files:
        key = os.path.basename(fpath).replace("projection_", "").replace(".npy", "")
        coords = np.load(fpath)
        indices = np.load(os.path.join(ASSETS_DIR, f"indices_{key}.npy"))
        
        data_map[key] = coords
        index_map[key] = indices
        all_points_list.append(coords)
        
    stack = np.vstack(all_points_list)
    x_min, x_max = stack[:,0].min(), stack[:,0].max()
    y_min, y_max = stack[:,1].min(), stack[:,1].max()
    pad = (x_max - x_min) * 0.05
    x_edges = np.linspace(x_min-pad, x_max+pad, GRID_RES+1)
    y_edges = np.linspace(y_min-pad, y_max+pad, GRID_RES+1)
    x_c = (x_edges[:-1] + x_edges[1:]) / 2
    y_c = (y_edges[:-1] + y_edges[1:]) / 2

    # --- Tooltip Generation ---
    LOGGER.info("Mapping pixels to text...")
    pixel_requests = {} 
    for key, coords in data_map.items():
        original_ids = index_map[key]
        xb = np.digitize(coords[:,0], x_edges) - 1
        yb = np.digitize(coords[:,1], y_edges) - 1
        valid_mask = (xb >= 0) & (xb < GRID_RES) & (yb >= 0) & (yb < GRID_RES)
        valid_idxs = np.where(valid_mask)[0]
        
        # Last writer wins (simulates newest)
        for i in valid_idxs:
            pixel_requests[(xb[i], yb[i])] = {
                "source": key, 
                "corpus_id": original_ids[i] 
            }

    text_grid = np.full((GRID_RES, GRID_RES), "", dtype=object)
    requests_by_source = {}
    for pos, info in pixel_requests.items():
        requests_by_source.setdefault(info["source"], []).append({
            "pos": pos, "corpus_id": info["corpus_id"]
        })
        
    config_data = config_loader.load_configs_and_db_sizes()
    
    LOGGER.info("Fetching metadata...")
    for source, items in requests_by_source.items():
        cfg = config_data.get(source)
        if not cfg: continue
        hits = [{"corpus_id": x["corpus_id"], "score": 0.0} for x in items]
        
        try:
            with open(cfg["metadata_path"], 'r') as f: meta = json.load(f)
            df = data_handler.fetch_specific_rows(hits, meta, cfg["data_folder"], cfg.get("combined_data_file"))
            if 'score' in df.columns: df = df.drop(columns=['score'])
            records = df.to_dict('records')
            
            for i, row in enumerate(records):
                if i >= len(items): break
                gx, gy = items[i]["pos"]
                title = str(row.get('title', 'No Title'))
                if len(title) > 80: title = title[:80] + "..."
                date = str(row.get('date', '') or row.get('year', ''))[:4]
                text_grid[gx, gy] = f"<b>{title}</b><br>({date}) {LABEL_MAP.get(source, source)}"
        except Exception as e:
            LOGGER.error(f"Metadata error for {source}: {e}")

    # --- Plotting ---
    LOGGER.info("Building Plotly HTML...")
    fig = go.Figure()
    
    # Trace 0: All Papers (Base)
    z_all, _, _ = np.histogram2d(stack[:,0], stack[:,1], bins=[x_edges, y_edges])
    z_all_log = np.log1p(z_all).T
    z_all_log[z_all_log == 0] = np.nan
    
    fig.add_trace(go.Heatmap(
        z=z_all_log, x=x_c, y=y_c, 
        colorscale=COLORSCALES["All"], showscale=False, 
        name="All Papers", visible=True, opacity=0.7, hoverinfo='skip'
    ))

    # Add Source Traces
    # We keep track of trace indices to build buttons
    trace_indices = {"All": 0}
    current_idx = 1
    
    sources_present = list(data_map.keys())
    
    for key in sources_present:
        coords = data_map[key]
        z_s, _, _ = np.histogram2d(coords[:,0], coords[:,1], bins=[x_edges, y_edges])
        z_s_log = np.log1p(z_s).T
        z_s_log[z_s_log == 0] = np.nan
        
        label = LABEL_MAP.get(key, key)
        fig.add_trace(go.Heatmap(
            z=z_s_log, x=x_c, y=y_c, 
            colorscale=COLORSCALES.get(key, "Viridis"), 
            showscale=False, 
            name=label, 
            visible=True,
            opacity=0.8, 
            hoverinfo='skip'
        ))
        trace_indices[label] = current_idx
        current_idx += 1

    # Final Trace: Tooltips (Always visible but transparent)
    fig.add_trace(go.Heatmap(
        z=z_all_log, x=x_c, y=y_c, 
        customdata=text_grid.T, 
        hovertemplate="%{customdata}<extra></extra>", 
        name="Tooltips", 
        opacity=0, 
        showscale=False,
        visible=True
    ))
    tooltip_idx = current_idx

    # --- Create Buttons ---
    # Logic: 
    # "All" button -> Shows All Trace (0) + Tooltips (Last)
    # "Source X" button -> Shows Source X Trace + Tooltips
    
    buttons = []
    
    # 1. "Show All" Button
    # Visibility: All(True), Sources(True), Tooltip(True) -> Composite View
    visibility_composite = [True] * (tooltip_idx + 1)
    buttons.append(dict(
        label="Show Composite",
        method="update",
        args=[{"visible": visibility_composite}]
    ))
    
    # 2. Individual Source Buttons
    for label, idx in trace_indices.items():
        if label == "All": continue # Skip base layer in specific toggles
        
        # Visibility: Turn OFF everything, Turn ON this source + Tooltips
        vis = [False] * (tooltip_idx + 1)
        vis[idx] = True # The source layer
        vis[tooltip_idx] = True # The tooltip layer
        
        buttons.append(dict(
            label=f"Show {label}",
            method="update",
            args=[{"visible": vis}]
        ))

    fig.update_layout(
        title="Global Semantic Map", 
        plot_bgcolor="black", 
        paper_bgcolor="black", 
        font=dict(color="white"), 
        width=1000, height=900, 
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.1, y=1.1,
            showactive=True,
            buttons=buttons
        )]
    )
    
    fig.write_html("embedding_labeled_map.html", include_plotlyjs='cdn')
    LOGGER.info("Done. Saved embedding_labeled_map.html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "project", "viz", "all"], default="all")
    args = parser.parse_args()
    config = load_config()

    if args.mode in ["train", "all"]:
        run_training(config)
    if args.mode in ["project", "all"]:
        run_projection(config)
    if args.mode in ["viz", "all"]:
        run_generation()