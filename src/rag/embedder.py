"""Text embedding using sentence-transformers."""

import numpy as np
from pathlib import Path
from loguru import logger

from src.config import settings
from src.rag.manual_parser import ManualChunk


class Embedder:
    """Text embedding wrapper using sentence-transformers."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded")
        return self._model

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of texts into vectors."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.array(embeddings)

    def embed_chunks(self, chunks: list[ManualChunk], batch_size: int = 32) -> np.ndarray:
        """Embed manual chunks."""
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts, batch_size)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.model.encode([query], normalize_embeddings=True)[0]
