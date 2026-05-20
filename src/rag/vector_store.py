"""Numpy-based vector store for manual chunk retrieval."""

import pickle
from pathlib import Path
import numpy as np
from loguru import logger
from src.rag.manual_parser import ManualChunk


class VectorStore:
    """Numpy-based vector store for semantic search."""

    def __init__(self, store_dir):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = self.store_dir / "embeddings.npy"
        self.meta_path = self.store_dir / "metadata.pkl"
        self.chunks = []
        self.embeddings = None

    def build(self, embeddings, chunks):
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Mismatch: {} chunks vs {} embeddings".format(
                    len(chunks), len(embeddings)
                )
            )
        self.chunks = chunks
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = (embeddings / norms).astype(np.float32)
        logger.info(
            "Built vector store: {} vectors, dim={}".format(
                len(chunks), embeddings.shape[1]
            )
        )

    def save(self):
        if self.embeddings is None:
            raise RuntimeError('No embeddings to save. Call build() first.')
        np.save(str(self.embeddings_path), self.embeddings)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info("Saved vector store to {}".format(self.store_dir))

    def load(self):
        if not self.embeddings_path.exists() or not self.meta_path.exists():
            return False
        self.embeddings = np.load(str(self.embeddings_path))
        with open(self.meta_path, "rb") as f:
            self.chunks = pickle.load(f)
        logger.info(
            "Loaded vector store: {} vectors".format(
                self.embeddings.shape[0]
            )
        )
        return True

    def search(self, query_embedding, top_k=10):
        if self.embeddings is None:
            raise RuntimeError("Index not loaded.")
        q = query_embedding.astype(np.float32).reshape(1, -1)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        scores = (self.embeddings @ q.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self.chunks[idx], float(scores[idx]))
            for idx in top_indices
        ]




