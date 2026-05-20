"""Text embedding using DashScope API."""

import time
import numpy as np
from loguru import logger
from src.config import settings
from src.rag.manual_parser import ManualChunk


class Embedder:
    """Embedding wrapper using DashScope text-embedding API."""

    def __init__(self, model_name=None):
        self.model_name = model_name or settings.embedding_model

    def _call_api(self, texts):
        import dashscope
        dashscope.api_key = settings.dashscope_api_key
        from dashscope import TextEmbedding
        resp = TextEmbedding.call(model=self.model_name, input=texts)
        if resp.status_code != 200:
            raise RuntimeError(
                "Embedding API error: {} - {}".format(resp.code, resp.message)
            )
        sorted_embs = sorted(resp.output, key=lambda x: x["text_index"])
        return [item["embedding"] for item in sorted_embs]

    def embed_texts(self, texts, batch_size=25):
        all_embeddings = []
        total = (len(texts) - 1) // batch_size + 1
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            idx = i // batch_size + 1
            logger.info("Embedding batch {}/{} ({} texts)".format(idx, total, len(batch)))
            embs = self._call_api(batch)
            all_embeddings.extend(embs)
            if i + batch_size < len(texts):
                time.sleep(0.5)
        return np.array(all_embeddings)

    def embed_chunks(self, chunks, batch_size=25):
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts, batch_size)

    def embed_query(self, query):
        embs = self._call_api([query])
        return np.array(embs[0])


