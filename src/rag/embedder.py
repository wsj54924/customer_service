"""Text embedding using Ollama (local), HuggingFace (cloud), or DashScope (cloud)."""

import time
import numpy as np
from loguru import logger
from src.config import settings
from src.rag.manual_parser import ManualChunk


class Embedder:
    """Embedding with auto-detection: Ollama -> HuggingFace -> DashScope."""

    def __init__(self, model_name=None):
        self.model_name = model_name or settings.embedding_model
        self._dashscope_model = settings.dashscope_embedding_model
        self._hf_model = "Qwen/Qwen3-Embedding-4B"
        self._backend = None  # resolved on first call

    def _detect_backend(self):
        """Detect available backend: prefer Ollama, then HuggingFace, then DashScope."""
        logger.info(
            "Embedder init: model={} dashscope_model={} ollama_url={}".format(
                self.model_name, self._dashscope_model, settings.ollama_base_url
            )
        )
        if self._backend:
            return self._backend
        # Try Ollama first
        try:
            import httpx
            base_url = settings.ollama_base_url.rstrip("/")
            resp = httpx.get("{}/api/tags".format(base_url), timeout=3.0)
            if resp.status_code == 200:
                self._backend = "ollama"
                logger.info(
                    "Using Ollama backend at {} with model={}".format(
                        base_url, self.model_name
                    )
                )
                return self._backend
        except Exception as e:
            logger.warning(
                "Ollama not available at {}: {}. "
                "Note: Vercel cannot reach localhost; "
                "set OLLAMA_BASE_URL to a public endpoint "
                "or use HuggingFace/DashScope.".format(
                    settings.ollama_base_url, e
                )
            )
        # Fallback to HuggingFace (free, same model as local Ollama, 2560-dim)
        if settings.hf_api_token:
            self._backend = "huggingface"
            logger.info(
                "Using HuggingFace backend with model={}".format(self._hf_model)
            )
            return self._backend
        # Fallback to DashScope
        if settings.dashscope_api_key:
            self._backend = "dashscope"
            logger.warning(
                "Using DashScope cloud backend with model={}. "
                "WARNING: vector store was built with local model '{}'; "
                "if dimensions differ, queries will fail. "
                "Rebuild index with DashScope or use a matching model.".format(
                    self._dashscope_model, self.model_name
                )
            )
            return self._backend
        raise RuntimeError(
            "No embedding backend available. "
            "Set OLLAMA_BASE_URL, HF_API_TOKEN, or DASHSCOPE_API_KEY."
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

    def _call_huggingface(self, texts):
        """Call HuggingFace Inference API with Qwen3-Embedding-4B (2560-dim)."""
        import httpx
        api_url = "https://router.huggingface.co/hf-inference/models/{}".format(
            self._hf_model
        )
        headers = {"Authorization": "Bearer {}".format(settings.hf_api_token)}
        all_embeddings = []
        for text in texts:
            resp = httpx.post(
                api_url, headers=headers, json={"inputs": text}, timeout=30.0
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    "HuggingFace embedding error: {} - {} (model={})".format(
                        resp.status_code, resp.text[:300], self._hf_model
                    )
                )
            emb = resp.json()
            # HF may return flat [dim1, dim2, ...] or nested [[dim1, dim2, ...]]
            if isinstance(emb, list) and emb and isinstance(emb[0], list):
                emb = emb[0]
            all_embeddings.append(emb)
        return all_embeddings

    def _call_dashscope(self, texts):
        import json as _json
        import dashscope
        dashscope.api_key = settings.dashscope_api_key
        from dashscope import TextEmbedding
        resp = TextEmbedding.call(model=self._dashscope_model, input=texts)
        if resp.status_code != 200:
            raise RuntimeError(
                "DashScope embedding error: {} - {} (model={})".format(
                    resp.code, resp.message, self._dashscope_model
                )
            )
        output = resp.output
        # Handle JSON string responses
        if isinstance(output, str):
            try:
                output = _json.loads(output)
            except _json.JSONDecodeError:
                pass
        if (
            isinstance(output, list)
            and output
            and isinstance(output[0], dict)
            and "embedding" in output[0]
        ):
            sorted_embs = sorted(output, key=lambda x: x.get("text_index", 0))
            return [item["embedding"] for item in sorted_embs]
        if isinstance(output, list) and output and isinstance(output[0], list):
            return output
        raise RuntimeError(
            "DashScope embedding response unexpected (model={}, type={}): {}".format(
                self._dashscope_model, type(output).__name__, repr(output)[:500]
            )
        )

    def _call_api(self, texts):
        backend = self._detect_backend()
        if backend == "ollama":
            return self._call_ollama(texts)
        if backend == "huggingface":
            return self._call_huggingface(texts)
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
