"""Text embedding using Ollama (local) or DashScope (cloud)."""

import time
import numpy as np
from loguru import logger
from src.config import settings
from src.rag.manual_parser import ManualChunk


class Embedder:
    """Embedding with auto-detection: Ollama local or DashScope cloud."""

    def __init__(self, model_name=None):
        self.model_name = model_name or settings.embedding_model
        self._dashscope_model = self.model_name
        if self._dashscope_model and ':' in self._dashscope_model:
            self._dashscope_model = self._dashscope_model.split(':', 1)[0]
        self._backend = None  # resolved on first call

    def _detect_backend(self):
        """Detect available backend: prefer Ollama, fallback to DashScope."""
        if self._backend:
            return self._backend
        # Try Ollama first
        try:
            import httpx
            base_url = settings.ollama_base_url.rstrip("/")
            resp = httpx.get("{}/api/tags".format(base_url), timeout=3.0)
            if resp.status_code == 200:
                self._backend = "ollama"
                logger.info("Using Ollama backend at {}".format(base_url))
                return self._backend
        except Exception:
            pass
        # Fallback to DashScope
        if settings.dashscope_api_key:
            self._backend = "dashscope"
            logger.info("Using DashScope cloud backend")
            return self._backend
        raise RuntimeError(
            "No embedding backend available. "
            "Start Ollama locally or set DASHSCOPE_API_KEY."
        )

    def _call_ollama(self, texts):
        import httpx
        base_url = settings.ollama_base_url.rstrip("/")
        resp = httpx.post(
            "{}/api/embed".format(base_url),
            json={"model": self.model_name, "input": texts},
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def _call_dashscope(self, texts):
        import dashscope
        dashscope.api_key = settings.dashscope_api_key
        from dashscope import TextEmbedding
        resp = TextEmbedding.call(model=self._dashscope_model, input=texts)
        if resp.status_code != 200:
            raise RuntimeError(
                "DashScope error: {} - {}".format(resp.code, resp.message)
            )
        sorted_embs = sorted(resp.output, key=lambda x: x["text_index"])
        return [item["embedding"] for item in sorted_embs]

    def _call_api(self, texts):
        backend = self._detect_backend()
        if backend == "ollama":
            return self._call_ollama(texts)
        return self._call_dashscope(texts)

    def embed_texts(self, texts, batch_size=25):
        all_embeddings = []
        total = (len(texts) - 1) // batch_size + 1
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            idx = i // batch_size + 1
            logger.info(
                "Embedding batch {}/{} ({} texts)".format(idx, total, len(batch))
            )
            embs = self._call_api(batch)
            all_embeddings.extend(embs)
            if i + batch_size < len(texts):
                time.sleep(0.2)
        return np.array(all_embeddings)

    def embed_chunks(self, chunks, batch_size=25):
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts, batch_size)

    def embed_query(self, query):
        embs = self._call_api([query])
        return np.array(embs[0])