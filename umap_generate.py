import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import logging
import yaml
from glob import glob
from tqdm import tqdm
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("MapGen")

# Suppress Streamlit warnings for standalone execution
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching.storage.memory_cache_storage_manager").setLevel(logging.ERROR)

# Import your existing modules for data fetching
import config_loader
import data_handler

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
ASSETS_DIR = "./visualization_assets"
OUTPUT_FILE = "embedding_labeled_map.html"
DATA_FILE = "map_data.npz" # File to save processed grids
GRID_RES = 600  # 600x600 resolution as requested

LABEL_MAP = {
    "pubmed_config": "PubMed",
    "biorxiv_config": "BioRxiv",
    "medrxiv_config": "MedRxiv",
    "arxiv_config": "arXiv"
}

# Optimized for Dark Mode (Glowing effect)
# We use _r (reversed) scales because standard Plotly sequential scales go Light -> Dark.
# On a black background, we want Low Density = Dark (invisible), High Density = Bright/Light.
# So we reverse them: Low=Dark Color, High=White/Light Color.
COLORSCALES = {
    "All": "Inferno",       # Natural Dark -> Bright (Black-Red-Yellow)
    "pubmed_config": "Blues_r",   # Deep Blue -> White
    "biorxiv_config": "Reds_r",   # Deep Red -> White
    "medrxiv_config": "Greens_r", # Deep Green -> White
    "arxiv_config": "Oranges_r"   # Deep Orange -> White
}

# Legend Marker Colors (High intensity match)
LEGEND_COLORS = {
    "All": "#fcba03", # Yellow-Orange from Inferno
    "pubmed_config": "#00ffff", # Cyan
    "biorxiv_config": "#ff4444", # Bright Red
    "medrxiv_config": "#55ff55", # Bright Green
    "arxiv_config": "#ffaa00" # Bright Orange
}

def load_projections_and_bounds(assets_dir):
    """
    Loads all .npy projection files and calculates global coordinate bounds.
    """
    data_map = {}
    files = glob(os.path.join(assets_dir, "projection_*.npy"))
    
    if not files:
        LOGGER.error("No projection files found.")
        return None, None, None

    all_points = []
    
    for fpath in files:
        fname = os.path.basename(fpath)
        key = fname.replace("projection_", "").replace(".npy", "")
        
        try:
            data = np.load(fpath)
            data_map[key] = data
            all_points.append(data)
            LOGGER.info(f"Loaded {len(data):,} points for {key}")
        except Exception as e:
            LOGGER.error(f"Failed to load {fname}: {e}")

    if not all_points:
        return None, None, None

    # Calculate global bounds
    stack = np.vstack(all_points)
    x_min, x_max = stack[:, 0].min(), stack[:, 0].max()
    y_min, y_max = stack[:, 1].min(), stack[:, 1].max()
    
    # 5% Padding
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    
    # Define bin edges for the grid
    x_edges = np.linspace(x_min - x_pad, x_max + x_pad, GRID_RES + 1)
    y_edges = np.linspace(y_min - y_pad, y_max + y_pad, GRID_RES + 1)
    
    return data_map, x_edges, y_edges

def create_representative_grid(data_map, x_edges, y_edges):
    """
    Maps every pixel in the 600x600 grid to a specific (source, index) 
    of a paper residing in that pixel.
    """
    LOGGER.info("Mapping points to grid pixels...")
    pixel_map = {}
    config_data = config_loader.load_configs_and_db_sizes()
    configs = config_data["configs"]
    
    source_offsets = {} 
    
    for i, (key, coords) in enumerate(data_map.items()):
        cfg = config_data.get(key)
        if not cfg: continue
        
        meta_path = cfg.get("metadata_path")
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        chunk_intervals = []
        current_offset = 0
        for chunk in meta["chunks"]:
            rows = chunk["actual_rows"]
            g_start = chunk["global_start"]
            chunk_intervals.append( (current_offset, current_offset + rows, g_start) )
            current_offset += rows
            
        source_offsets[key] = chunk_intervals

        # Digitize coordinates to find bin indices
        x_bins = np.digitize(coords[:, 0], x_edges) - 1
        y_bins = np.digitize(coords[:, 1], y_edges) - 1
        
        valid = (x_bins >= 0) & (x_bins < GRID_RES) & (y_bins >= 0) & (y_bins < GRID_RES)
        valid_indices = np.where(valid)[0]
        
        # Populate pixel map (first comer wins strategy for simplicity)
        for idx in valid_indices:
            xi, yi = x_bins[idx], y_bins[idx]
            grid_key = (xi, yi)
            if grid_key not in pixel_map:
                pixel_map[grid_key] = (key, idx)

    return pixel_map, source_offsets

def fetch_grid_metadata(pixel_map, source_offsets):
    """
    Fetches actual titles for the representatives.
    """
    LOGGER.info(f"Fetching metadata for {len(pixel_map)} representative papers...")
    
    batch_requests = []
    
    # Reconstruct Global ID from (source, local_npy_index)
    for (xi, yi), (source, local_idx) in pixel_map.items():
        intervals = source_offsets.get(source, [])
        global_id = -1
        
        for start, end, g_base in intervals:
            if start <= local_idx < end:
                chunk_offset = local_idx - start
                global_id = g_base + chunk_offset
                break
        
        if global_id != -1:
            batch_requests.append({
                "source": source,
                "corpus_id": global_id,
                "grid_pos": (xi, yi)
            })

    config_data = config_loader.load_configs_and_db_sizes()
    text_grid = np.full((GRID_RES, GRID_RES), "", dtype=object)
    
    grouped = {}
    for item in batch_requests:
        grouped.setdefault(item['source'], []).append(item)
        
    for source, items in grouped.items():
        cfg = config_data.get(source)
        if not cfg: continue
        
        with open(cfg["metadata_path"], 'r') as f:
            meta = json.load(f)
            
        hits = [{'corpus_id': x['corpus_id'], 'score': 0} for x in items]
        
        try:
            df = data_handler.fetch_specific_rows(
                hits, 
                meta, 
                cfg["data_folder"], 
                cfg.get("combined_data_file")
            )
            
            for i, row in enumerate(df.to_dict('records')):
                if i >= len(items): break
                
                grid_item = items[i]
                gx, gy = grid_item['grid_pos']
                
                title = row.get('title') or "No Title"
                title = title[:100] + "..." if len(title) > 100 else title
                
                date = str(row.get('date') or row.get('year') or "")[:4]
                if not date or date == "nan": date = "n.d."
                
                source_label = LABEL_MAP.get(source, source)
                
                # HTML tooltip format
                tooltip = f"<b>{title}</b><br>({date}) {source_label}"
                text_grid[gx, gy] = tooltip
                
        except Exception as e:
            LOGGER.error(f"Error fetching metadata for {source}: {e}")

    return text_grid

def generate_html_and_save(data_map, x_edges, y_edges, text_grid):
    LOGGER.info("Generating density grids and Plotly HTML...")
    fig = go.Figure()
    
    # Centers for Plotly Heatmap
    x_c = (x_edges[:-1] + x_edges[1:]) / 2
    y_c = (y_edges[:-1] + y_edges[1:]) / 2
    
    # Data container for saving
    saved_data = {
        "text_grid": text_grid,
        "x_centers": x_c,
        "y_centers": y_c,
        "x_edges": x_edges,
        "y_edges": y_edges
    }
    
    # --- 1. Global Density (Bottom Layer) ---
    all_coords = np.vstack(list(data_map.values()))
    z_all, _, _ = np.histogram2d(all_coords[:,0], all_coords[:,1], bins=[x_edges, y_edges])
    z_all_log = np.log1p(z_all).T
    
    # Save raw density
    saved_data["z_All"] = z_all_log
    
    # Mask zeros for transparency in plot
    z_plot_all = z_all_log.copy()
    z_plot_all[z_plot_all == 0] = np.nan
    
    # Background Trace
    fig.add_trace(go.Heatmap(
        z=z_plot_all,
        x=x_c, y=y_c,
        colorscale=COLORSCALES["All"],
        showscale=False,
        name="All Papers",
        visible=True,
        opacity=0.7, # Higher opacity for dark mode visibility
        hoverinfo='skip',
        legendgroup="All"
    ))
    
    # Dummy Scatter for "All" Legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', 
                           marker=dict(size=10, color=LEGEND_COLORS["All"]), 
                           name="All Papers", showlegend=True, legendgroup="All"))

    # --- 2. Source Layers (Overlays) ---
    for source, coords in data_map.items():
        z_s, _, _ = np.histogram2d(coords[:,0], coords[:,1], bins=[x_edges, y_edges])
        z_s_log = np.log1p(z_s).T
        
        # Save raw density
        saved_data[f"z_{source}"] = z_s_log
        
        # Mask zeros
        z_plot_s = z_s_log.copy()
        z_plot_s[z_plot_s == 0] = np.nan
        
        label = LABEL_MAP.get(source, source)
        color_hex = LEGEND_COLORS.get(source, "white")
        
        fig.add_trace(go.Heatmap(
            z=z_plot_s,
            x=x_c, y=y_c,
            colorscale=COLORSCALES.get(source, "Viridis"),
            showscale=False,
            name=label,
            visible=True,
            opacity=0.8,
            hoverinfo='skip',
            legendgroup=source
        ))
        
        # Legend Item
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers', 
            marker=dict(size=10, color=color_hex), 
            name=label, showlegend=True, legendgroup=source
        ))

    # --- 3. Interaction Layer (Top, Invisible) ---
    # Create valid mask from ALL data
    z_interact = z_all_log.copy()
    z_interact[z_interact == 0] = np.nan
    
    fig.add_trace(go.Heatmap(
        z=z_interact, 
        x=x_c, y=y_c,
        customdata=text_grid.T, # Transpose to match Z
        hovertemplate="%{customdata}<extra></extra>",
        showscale=False,
        name="Tooltips",
        opacity=0, # Completely invisible
        hoverinfo="text", # Triggers based on this trace presence
        showlegend=False
    ))

    # --- 4. Dark Mode Layout ---
    fig.update_layout(
        title=dict(text="Global Semantic Map (Hover for details)", font=dict(color="white", size=18)),
        plot_bgcolor="black",
        paper_bgcolor="black",
        width=1000, height=900,
        margin=dict(t=50, b=0, l=0, r=0),
        xaxis=dict(showticklabels=False, zeroline=False, visible=False),
        yaxis=dict(showticklabels=False, zeroline=False, visible=False),
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01, 
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white")
        )
    )
    
    # Save Data
    save_path = os.path.join(ASSETS_DIR, DATA_FILE)
    np.savez_compressed(save_path, **saved_data)
    LOGGER.info(f"Saved processed grid data to {save_path}")
    
    # Save HTML
    fig.write_html(OUTPUT_FILE, include_plotlyjs='cdn')
    LOGGER.info(f"Saved HTML to {OUTPUT_FILE}")

def main():
    if not os.path.exists(ASSETS_DIR): return
    
    # 1. Load Data
    data_map, x_e, y_e = load_projections_and_bounds(ASSETS_DIR)
    if not data_map: return
    
    # 2. Map Pixels to Papers
    pixel_map, offsets = create_representative_grid(data_map, x_e, y_e)
    
    # 3. Fetch Metadata
    text_grid = fetch_grid_metadata(pixel_map, offsets)
    
    # 4. Generate Plot & Save
    generate_html_and_save(data_map, x_e, y_e, text_grid)

if __name__ == "__main__":
    main()