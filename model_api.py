from fastapi import FastAPI
from pydantic import BaseModel
import torch
from sentence_transformers import SentenceTransformer
import uvicorn

# Create the FastAPI app.
app = FastAPI()

# Define the request schema.
class QueryRequest(BaseModel):
    text: str
    normalize: bool = True
    precision: str = "ubinary"

# Global variable to hold the model.
model = None

# On startup, load the model once.
@app.on_event("startup")
async def startup_event():
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    model.to(device)
    print("Model loaded and moved to device:", device)

# Define the endpoint to encode a query.
@app.post("/encode")
async def encode(request: QueryRequest):
    global model
    with torch.no_grad():
        embedding = model.encode(
            [request.text],
            normalize_embeddings=request.normalize,
            precision=request.precision
        )
    # Convert the NumPy array to a list so it can be serialized as JSON.
    return {"embedding": embedding.tolist()}

# Run the app with Uvicorn if this file is executed directly.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
