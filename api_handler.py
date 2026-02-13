import requests
import numpy as np
import streamlit as st
import logging

LOGGER = logging.getLogger(__name__)

# Default URL, can be overridden if needed
MODEL_SERVER_URL = "http://localhost:8000/encode"

def get_query_embedding_packed(query, server_url=MODEL_SERVER_URL):
    """
    Sends query to remote server.
    The server handles:
      1. Adding 'query: ' prefix
      2. Encoding to float
      3. Quantizing to binary
      4. Packing into uint8
    
    This function simply retrieves the packed uint8 list and converts it to a numpy array.
    """
    # The server expects just {"text": "..."} as defined in QueryRequest pydantic model
    payload = {"text": query}
    
    try:
        response = requests.post(server_url, json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            # Server returns a list of integers (0-255) representing packed bytes
            # e.g., [255, 128, 0, 42, ...]
            packed_list = data.get("embedding", [])
            
            if not packed_list:
                st.error("Received empty embedding from server.")
                return None
                
            # Convert list directly to numpy array of uint8
            packed_embedding = np.array(packed_list, dtype=np.uint8)
            
            # Ensure 2D array (1, D_bytes) for FAISS
            # FAISS expects shape (n_queries, n_bytes)
            if packed_embedding.ndim == 1:
                packed_embedding = packed_embedding[np.newaxis, :]
            
            return packed_embedding
        else:
            st.error(f"Model API returned error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Model API not available. Please ensure that the model server is running.")
        return None
    except Exception as e:
        st.error(f"An error occurred while obtaining the query embedding: {e}")
        LOGGER.error(f"API Error: {e}")
        return None

def check_update_status():
    try:
        response = requests.get("http://localhost:8001/status", timeout=1)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "update running":
                return True
    except Exception:
        return None
    return None