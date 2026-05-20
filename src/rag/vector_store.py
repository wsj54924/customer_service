"""FAISS vector store for manual chunk retrieval."""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from loguru import logger

from src.rag.manual_parser import ManualChunk


class VectorStore:
    """FAISS-based vector store for semantic search over manual chunks."""

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / "faiss.index"
        self.meta_path = self.store_dir / "metadata.pkl"

        self.index: faiss.Index | None = None
        self.chunks: list[ManualChunk] = []
        self.embeddings: np.ndarray | None = None

    def build(self, embeddings: np.ndarray, chunks: list[ManualChunk]):
        """Build FAISS index from embeddings and chunks."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)

        # Build index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine sim after normalization
        self.index.add(self.embeddings)

        logger.info(f"Built FAISS index: {len(chunks)} vectors, dim={dim}")

    def save(self):
        """Save index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("No index to save. Call build() first.")

        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info(f"Saved index to {self.index_path}")

    def load(self) -> bool:
        """Load index and metadata from disk. Returns True if successful."""
        if not self.index_path.exists() or not self.meta_path.exists():
            return False

        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "rb") as f:
            self.chunks = pickle.load(f)

        logger.info(f"Loaded index: {self.index.ntotal} vectors, {len(self.chunks)} chunks")
        return True

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[tuple[ManualChunk, float]]:
        """Search for similar chunks.

        Returns list of (chunk, similarity_score) tuples sorted by score descending.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() or build() first.")

        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.chunks[idx], float(score)))

        return results
