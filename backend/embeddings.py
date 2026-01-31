# backend/embeddings.py

from sentence_transformers import SentenceTransformer
from backend.kb_loader import load_kb_sections

# Free, lightweight, CPU-friendly model
MODEL_NAME = "all-MiniLM-L6-v2"

def build_kb_embeddings():
    """
    Loads KB chunks and converts them into embeddings.
    Returns (chunks, embeddings)
    """
    chunks = load_kb_sections()

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks)

    return chunks, embeddings
