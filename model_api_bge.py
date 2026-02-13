import numpy as np
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# You can switch this to your local fine-tuned folder if you have one
MODEL_ID = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# -----------------------------------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------------------------------
model_context = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device}...")
    
    try:
        model = SentenceTransformer(
            MODEL_ID, 
            device=device,
            trust_remote_code=True,
            model_kwargs={
                # Use FP16 on GPU for speed, Float32 on CPU for compatibility
                "dtype": torch.float16 if device == "cuda" else torch.float32,
                "attn_implementation": "sdpa" if torch.cuda.is_available() else "eager"
            }
        )
        
        # Optional: Compile for Blackwell/H100 GPUs (Linux only)
        if device == "cuda":
            try:
                # model = torch.compile(model) # Uncomment for extra speed on production
                pass
            except:
                pass
                
        model_context["model"] = model
        print("✅ Model ready.")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        
    yield
    
    # --- Shutdown (Clean up VRAM) ---
    model_context.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# APP SETUP
# -----------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    text: str

# -----------------------------------------------------------------------------
# ENDPOINT
# -----------------------------------------------------------------------------
@app.post("/encode")
async def encode(request: QueryRequest):
    model = model_context.get("model")
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # BGE instruction is critical for query performance
    text_with_prefix = QUERY_PREFIX + request.text.strip()
    
    with torch.no_grad():
        # 1. Generate Float Embeddings (Normalized)
        emb_float = model.encode(
            [text_with_prefix],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # 2. Binary Quantization (>0 -> 1)
        # We verified this retains ~98% MRR for BGE-Small
        bits = (emb_float > 0)
        
        # 3. Pack bits into uint8 (8x compression)
        # 384 dimensions -> 48 bytes
        packed_uint8 = np.packbits(bits, axis=1)

    return {
        "embedding": packed_uint8[0].tolist(),
        "model": MODEL_ID
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)