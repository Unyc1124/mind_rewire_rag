# backend/retriever.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.kb_loader import load_kb_sections

MODEL_NAME = "all-MiniLM-L6-v2"

class KBRetriever:
    def __init__(self):
        # Load KB
        self.chunks = load_kb_sections()

        # Load embedding model
        self.model = SentenceTransformer(MODEL_NAME)

        # Build embeddings
        self.embeddings = self.model.encode(self.chunks)

        # Build FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings))

    def search(self, query: str, top_k: int = 3):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.chunks[idx])

        return results
