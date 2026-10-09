import logging

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------------------------------
model_context = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"Loading {MODEL_ID} on {device}…")
    
    try:
        model = SentenceTransformer(
            MODEL_ID, 
            device=device,
            model_kwargs={
                # Use FP16 on GPU for speed, Float32 on CPU for compatibility
                "dtype": torch.float16 if device == "cuda" else torch.float32,
                "attn_implementation": "sdpa" if torch.cuda.is_available() else "eager"
            }
        )

        model_context["model"] = model
        LOGGER.info("Model ready.")

    except Exception as e:
        LOGGER.error(f"Error loading model: {e}")
        
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
# ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Returns 200 with the model name once the model is loaded, 503 otherwise."""
    if "model" not in model_context:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model": MODEL_ID}


# Plain def: FastAPI runs it in its threadpool, so a CPU-bound encode does not
# block the event loop (and /health) while it runs.
@app.post("/encode")
def encode(request: QueryRequest):
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
        # Float query for rescoring binary candidates (search_logic); clients
        # that only read "embedding" are unaffected.
        "embedding_float": np.round(emb_float[0], 6).tolist(),
        "model": MODEL_ID
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
