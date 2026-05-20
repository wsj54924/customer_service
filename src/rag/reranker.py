"""Optional reranker - uses vector search order when no model configured."""

from loguru import logger
from src.config import settings
from src.rag.manual_parser import ManualChunk


class Reranker:
    """Optional reranker. If no model configured, returns top-k by vector score."""

    def __init__(self, model_name=None):
        self.model_name = model_name or settings.reranker_model

    def rerank(self, query, chunks, top_k=None):
        if not chunks:
            return []
        top_k = top_k or settings.top_k_rerank

        # No reranker model configured: just return top-k by vector score
        if not self.model_name:
            logger.info("No reranker configured, using vector search order")
            return [(c, s) for c, s in chunks[:top_k]]

        # Try Ollama-based reranking
        try:
            return self._ollama_rerank(query, chunks, top_k)
        except Exception as e:
            logger.warning(
                "Rerank failed: {}, falling back to vector search order".format(e)
            )
            return [(c, s) for c, s in chunks[:top_k]]

    def _ollama_rerank(self, query, chunks, top_k):
        """Use Ollama to score relevance of each chunk to the query."""
        import httpx

        base_url = settings.ollama_base_url.rstrip("/")
        scores = []
        for chunk, orig_score in chunks:
            prompt = (
                "Rate the relevance of the following document to the query "
                "on a scale from 0.0 to 1.0. Reply with ONLY a number.\n\n"
                "Query: {}\nDocument: {}\nRelevance:".format(
                    query, chunk.text[:500]
                )
            )
            resp = httpx.post(
                "{}/api/generate".format(base_url),
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 10},
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            text = resp.json()["response"].strip()
            try:
                score = float(text.split()[0])
            except (ValueError, IndexError):
                score = orig_score
            scores.append((chunk, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]