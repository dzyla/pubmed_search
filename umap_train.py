import os
import json
import yaml
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import umap
from pathlib import Path
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("Trainer")

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG_PATH = "config_mss.yaml"
SAMPLE_SIZE = 800_000  # Number of points to sample for training the map
BATCH_SIZE = 4096
EPOCHS = 500
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./visualization_assets"

LOGGER.info(f"Using device: {DEVICE}")

# -----------------------------------------------------------------------------
# PYTORCH MODEL
# -----------------------------------------------------------------------------
class ParametricUMAP(nn.Module):
    """
    A lightweight feed-forward network to approximate UMAP projection.
    Maps High-Dim Binary (unpacked to float) -> 2D.
    """
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
            
        layers.append(nn.Linear(in_d, 2)) # Output 2D
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

# -----------------------------------------------------------------------------
# DATA UTILITIES
# -----------------------------------------------------------------------------
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def unpack_bits_batch(packed_batch, target_float_type=np.float32):
    """
    Converts packed uint8 batch (N, D_bytes) -> float32 batch (N, D_bits)
    """
    # unpackbits creates uint8 0/1. We cast to float.
    # axis=1 ensures we unpack bits along the feature dimension
    unpacked = np.unpackbits(packed_batch, axis=1)
    return unpacked.astype(target_float_type)

def collect_training_sample(configs, sample_size):
    """
    Reservoir samples vectors from all configured databases.
    """
    LOGGER.info("Scanning chunks for training sample...")
    
    # 1. Identify all chunks
    all_chunks = []
    
    db_keys = ["pubmed_config", "biorxiv_config", "medrxiv_config", "arxiv_config"]
    
    for key in db_keys:
        if key not in configs: continue
        
        cfg = configs[key]
        meta_path = cfg["metadata_path"]
        chunk_dir = cfg["chunk_dir"]
        
        if not os.path.exists(meta_path):
            LOGGER.warning(f"Metadata not found for {key}")
            continue
            
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        embedding_dim_bytes = meta.get("embedding_dim") # Packed dim
        if not embedding_dim_bytes: continue
        
        # We need unpacked dim for model initialization
        embedding_dim_bits = embedding_dim_bytes * 8
        
        for chunk in meta.get("chunks", []):
            chunk_path = os.path.join(chunk_dir, chunk["chunk_file"])
            if os.path.exists(chunk_path):
                all_chunks.append({
                    "path": chunk_path,
                    "rows": chunk.get("actual_rows", 0),
                    "dim_bytes": embedding_dim_bytes,
                    "source": key
                })

    if not all_chunks:
        raise ValueError("No chunks found!")

    total_rows = sum(c["rows"] for c in all_chunks)
    LOGGER.info(f"Total available rows: {total_rows:,}. Target sample: {sample_size:,}")
    
    # 2. Reservoir Sampling
    # We want a uniform sample across the entire dataset
    # Probability of picking a row = sample_size / total_rows
    
    sampled_data = []
    rows_collected = 0
    
    # If total rows < sample size, take everything
    take_all = total_rows <= sample_size
    
    pbar = tqdm(total=len(all_chunks), desc="Sampling Chunks")
    for chunk in all_chunks:
        try:
            # mmap chunk
            data = np.memmap(chunk["path"], dtype=np.uint8, mode='r', 
                           shape=(chunk["rows"], chunk["dim_bytes"]))
            
            if take_all:
                indices = np.arange(chunk["rows"])
            else:
                # Calculate how many to take from this chunk proportional to its size?
                # Or just random selection?
                # Simple approach: Randomly select N indices based on ratio
                n_select = int((chunk["rows"] / total_rows) * sample_size)
                # Ensure at least 1 if rows > 0
                if n_select == 0 and chunk["rows"] > 0: n_select = 1
                
                indices = np.random.choice(chunk["rows"], size=n_select, replace=False)
            
            # Extract and copy to memory
            subset_packed = data[indices]
            
            # Unpack immediately to float32 to verify validity
            subset_float = unpack_bits_batch(subset_packed)
            
            sampled_data.append(subset_float)
            rows_collected += len(subset_float)
            
            del data
            
        except Exception as e:
            LOGGER.error(f"Error reading chunk {chunk['path']}: {e}")
        pbar.update(1)
    pbar.close()
    
    if not sampled_data:
        raise ValueError("Failed to collect any data.")
        
    full_sample = np.concatenate(sampled_data, axis=0)
    
    # Shuffle
    np.random.shuffle(full_sample)
    
    # Trim to exact sample size if slightly over
    if len(full_sample) > sample_size:
        full_sample = full_sample[:sample_size]
        
    LOGGER.info(f"Final training sample shape: {full_sample.shape}")
    return full_sample

# -----------------------------------------------------------------------------
# MAIN WORKFLOW
# -----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = load_config()
    
    # 1. Collect Data
    X_train = collect_training_sample(config, SAMPLE_SIZE)
    input_dim = X_train.shape[1]
    
    # 2. Learn UMAP Manifold (CPU)
    LOGGER.info("Fitting Reference UMAP (this may take a while)...")
    # Using 'hamming' metric or 'euclidean' on bits. 
    # Since X_train is float 0.0/1.0, euclidean distance squared is Hamming distance.
    reducer = umap.UMAP(
        n_components=2,
        metric='cosine', 
        n_neighbors=250,
        min_dist=0.01,
        verbose=True,
        init='pca',
        n_jobs=-1
    )
    y_train = reducer.fit_transform(X_train)
    
    # 3. Train Parametric Proxy (PyTorch)
    LOGGER.info("Training Parametric Proxy Model...")
    
    model = ParametricUMAP(input_dim=input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss() # Simple regression to UMAP coords
    
    # Prepare DataLoader
    dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    
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
        
        if (epoch+1) % 10 == 0:
            LOGGER.info(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.6f}")
            
    # Save Model
    model_path = os.path.join(OUTPUT_DIR, "param_umap_model.pth")
    torch.save(model.state_dict(), model_path)
    LOGGER.info(f"Model saved to {model_path}")
    
    # 4. Generate Global Maps (Batched Inference)
    LOGGER.info("Generating Global Projections...")
    model.eval()
    
    global_coords = []
    
    db_keys = ["pubmed_config", "biorxiv_config", "medrxiv_config", "arxiv_config"]
    
    for key in db_keys:
        if key not in config: continue
        cfg = config[key]
        
        chunk_dir = cfg["chunk_dir"]
        meta_path = cfg["metadata_path"]
        
        if not os.path.exists(meta_path): continue
        with open(meta_path, 'r') as f: meta = json.load(f)
        
        source_coords = []
        LOGGER.info(f"Projecting {key}...")
        
        for chunk in tqdm(meta.get("chunks", [])):
            c_path = os.path.join(chunk_dir, chunk["chunk_file"])
            if not os.path.exists(c_path): continue
            
            rows = chunk["actual_rows"]
            dim = meta["embedding_dim"]
            
            # Load
            data = np.memmap(c_path, dtype=np.uint8, mode='r', shape=(rows, dim))
            
            # Batch process this chunk
            chunk_2d = []
            for i in range(0, rows, BATCH_SIZE):
                batch_packed = data[i : i + BATCH_SIZE]
                batch_float = unpack_bits_batch(batch_packed)
                
                with torch.no_grad():
                    t_in = torch.from_numpy(batch_float).float().to(DEVICE)
                    t_out = model(t_in).cpu().numpy()
                    chunk_2d.append(t_out)
            
            if chunk_2d:
                chunk_coords = np.concatenate(chunk_2d, axis=0)
                source_coords.append(chunk_coords)
            
            del data
            
        if source_coords:
            full_source = np.concatenate(source_coords, axis=0)
            # Save individual map
            np.save(os.path.join(OUTPUT_DIR, f"projection_{key}.npy"), full_source)
            global_coords.append(full_source)
            LOGGER.info(f"Saved projection for {key}: {full_source.shape}")

    # 5. Generate Heatmap
    if global_coords:
        LOGGER.info("Generating Global Heatmap...")
        all_points = np.concatenate(global_coords, axis=0)
        
        # 2D Histogram
        heatmap, xedges, yedges = np.histogram2d(
            all_points[:, 0], all_points[:, 1], bins=500
        )
        
        # Save npz
        np.savez(
            os.path.join(OUTPUT_DIR, "hist2d.npz"), 
            hist=heatmap, 
            xedges=xedges, 
            yedges=yedges
        )
        LOGGER.info(f"Heatmap saved. Total points mapped: {len(all_points):,}")

if __name__ == "__main__":
    main()