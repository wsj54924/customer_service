"""Reranker using DashScope API."""

from loguru import logger
from src.config import settings
from src.rag.manual_parser import ManualChunk


class Reranker:
    """Reranker using DashScope TextReRank api."""

    def __init__(self, model_name=None):
        self.model_name = model_name or settings.reranker_model

    def rerank(self, query, chunks, top_k=None):
        if not chunks:
            return []
        top_k = top_k or settings.top_k_rerank
        try:
            import dashscope
            dashscope.api_key = settings.dashscope_api_key
            from dashscope import TextReRank
            docs = [chunk.text for chunk, _ in chunks]
            resp = TextReRank.call(
                model=self.model_name,
                query=query,
                documents=docs,
                top_n=top_k,
                return_documents=False,
            )
            if resp.status_code != 200:
                logger.warning("Rerank API error: {}, using original order".format(resp.code))
                return [(c, s) for c, s in chunks[:top_k]]
            results = []
            for item in resp.output["results"]:
                idx = item["index"]
                score = item["relevance_score"]
                results.append((chunks[idx][0], float(score)))
            return results
        except Exception as e:
            logger.warning("Rerank failed: {}, using original order".format(e))
            return [(c, s) for c, s in chunks[:top_k]]

