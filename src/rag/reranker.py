"""Reranker for improving retrieval precision."""

from loguru import logger
from src.config import settings
from src.rag.manual_parser import ManualChunk


class Reranker:
    """Cross-encoder reranker for search results."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.reranker_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name, max_length=512)
            logger.info("Reranker model loaded")
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[tuple[ManualChunk, float]],
        top_k: int | None = None,
    ) -> list[tuple[ManualChunk, float]]:
        """Rerank chunks by relevance to query.

        Args:
            query: The user's question
            chunks: List of (chunk, initial_score) tuples from vector search
            top_k: Number of top results to return

        Returns:
            Reranked list of (chunk, relevance_score) tuples
        """
        if not chunks:
            return []

        top_k = top_k or settings.top_k_rerank

        # Prepare pairs for cross-encoder
        pairs = [(query, chunk.text) for chunk, _ in chunks]

        # Score with cross-encoder
        scores = self.model.predict(pairs)

        # Sort by reranker score
        scored = list(zip([c for c, _ in chunks], scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [(chunk, float(score)) for chunk, score in scored[:top_k]]
