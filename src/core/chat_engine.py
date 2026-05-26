"""Chat engine: orchestrates RAG retrieval and LLM generation."""

import re
from loguru import logger

from src.config import settings
from src.core.llm_client import LLMClient
from src.core.prompts import SYSTEM_PROMPT, SERVICE_PROMPT
from src.rag.manual_parser import ManualChunk
from src.rag.vector_store import VectorStore
from src.rag.embedder import Embedder
from src.rag.reranker import Reranker


class ChatEngine:
    """Core chat engine that handles question answering with RAG."""

    def __init__(self):
        self.llm = LLMClient()
        self.embedder = Embedder()
        self.reranker = None  # Lazy load
        self.vector_store = VectorStore(settings.vector_store_dir)

        # Session memory for multi-turn
        self._sessions: dict[str, list[dict]] = {}

    def initialize(self):
        """Load vector store and models. Must be called before first use."""
        if not self.vector_store.load():
            raise RuntimeError(
                "Vector store not found. Run 'python -m scripts.build_index' first."
            )
        logger.info("Chat engine initialized")

    def answer(
        self,
        question: str,
        images: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict:
        """Answer a customer question.

        Args:
            question: User's question text
            images: Optional list of base64-encoded images
            session_id: Optional session ID for multi-turn

        Returns:
            Dict with 'answer', 'session_id', 'retrieved_chunks', 'image_ids'
        """
        # 1. Detect question type
        question_type = self._classify_question(question)
        logger.info(f"Question type: {question_type}")

        # 2. Retrieve relevant chunks (only for product/mixed questions)
        if question_type in ("product", "mixed", "general"):
            retrieved_chunks = self._retrieve(question)
            context, image_ids = self._build_context(retrieved_chunks)
            system_prompt = SYSTEM_PROMPT
        else:
            # Service questions: skip RAG, use service prompt directly
            retrieved_chunks = []
            context = ""
            image_ids = []
            system_prompt = SERVICE_PROMPT

        # 3. Handle multi-turn if session exists
        if session_id and session_id in self._sessions:
            history = self._sessions[session_id]
            answer_text = self._answer_with_history(
                question, context, history, images, system_prompt
            )
        elif images:
            answer_text = self.llm.chat_with_images(
                question=question,
                image_base64_list=images,
                context=context,
                system_prompt=system_prompt,
            )
        else:
            answer_text = self.llm.chat_text_only(
                question=question,
                context=context,
                system_prompt=system_prompt,
            )

        # 5. Extract image IDs from answer if present
        answer_text, answer_image_ids = self._extract_answer_images(answer_text)

        # Merge retrieved images with answer images
        final_image_ids = answer_image_ids or image_ids

        # 6. Update session memory
        if session_id:
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            self._sessions[session_id].append({"role": "user", "content": question})
            self._sessions[session_id].append({"role": "assistant", "content": answer_text})
            # Keep last 10 turns
            if len(self._sessions[session_id]) > 20:
                self._sessions[session_id] = self._sessions[session_id][-20:]

        # 7. Build chunks metadata for frontend
        chunks_meta = [
            {
                "source": chunk.manual_name,
                "score": round(score, 3),
                "text": chunk.text[:200],
            }
            for chunk, score in retrieved_chunks[:5]
        ]

        return {
            "answer": answer_text,
            "session_id": session_id or "new_session",
            "image_ids": final_image_ids,
            "question_type": question_type,
            "retrieved_count": len(retrieved_chunks),
            "chunks": chunks_meta,
        }

    def _classify_question(self, question: str) -> str:
        """Classify question into product/service/other."""
        product_keywords = [
            "钻", "表带", "健身", "冰箱", "烤箱", "键盘", "相机", "洗碗", "空调",
            "鼠标", "耳机", "摩托艇", "水泵", "清洁", "温控", "吹风", "空气净化",
            "座椅", "椅子", "电视", "摄像", "牙刷", "割草", "洗衣机", "充电",
            "电池", "指示灯", "how", "what", "use", "step", "button", "setting",
            "Manual", "manual",
        ]
        service_keywords = [
            "退货", "换货", "退款", "发票", "物流", "快递", "运费", "投诉",
            "保修", "维修", "售后", "客服", "发货", "收货", "7天无理由",
            "假货", "破损", "缺件", "补发", "运费",
        ]

        q = question.lower()
        is_product = any(kw.lower() in q for kw in product_keywords)
        is_service = any(kw in q for kw in service_keywords)

        if is_product and not is_service:
            return "product"
        elif is_service and not is_product:
            return "service"
        elif is_product and is_service:
            return "mixed"
        return "general"

    def _retrieve(self, question: str) -> list[tuple[ManualChunk, float]]:
        """Retrieve relevant chunks for a question."""
        # Embed query
        query_embedding = self.embedder.embed_query(question)

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=settings.top_k_retrieval)

        # Rerank if available
        if self.reranker is None:
            try:
                self.reranker = Reranker()
            except Exception as e:
                logger.warning(f"Could not load reranker: {e}")

        if self.reranker is not None:
            results = self.reranker.rerank(question, results, top_k=settings.top_k_rerank)

        return results

    def _build_context(
        self, chunks: list[tuple[ManualChunk, float]]
    ) -> tuple[str, list[str]]:
        """Build context string from retrieved chunks and collect image IDs."""
        context_parts = []
        all_image_ids = []

        for chunk, score in chunks:
            context_parts.append(
                f"[Source: {chunk.manual_name}, Relevance: {score:.3f}]\n{chunk.text}"
            )
            all_image_ids.extend(chunk.image_ids)

        context = "\n\n---\n\n".join(context_parts)
        # Deduplicate images while preserving order, cap at 10
        seen = set()
        unique_images = []
        for img in all_image_ids:
            if img not in seen:
                seen.add(img)
                unique_images.append(img)

        return context, unique_images[:10]

    def _answer_with_history(
        self,
        question: str,
        context: str,
        history: list[dict],
        images: list[str] | None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> str:
        """Answer with conversation history."""
        history_text = ""
        for msg in history[-6:]:  # Last 3 turns
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

        full_prompt = f"Conversation history:\n{history_text}\n\nCurrent question: {question}"

        if images:
            return self.llm.chat_with_images(
                question=full_prompt,
                image_base64_list=images,
                context=context,
                system_prompt=system_prompt,
            )
        else:
            return self.llm.chat_text_only(
                question=full_prompt,
                context=context,
                system_prompt=system_prompt,
            )

    def _extract_answer_images(self, answer: str) -> tuple[str, list[str]]:
        """Extract image IDs from answer text if present.

        Expected format at end of answer: ["img1", "img2"]
        """
        # Look for JSON array at the end
        match = re.search(r'\[("[^"]+"(?:,\s*"[^"]+")*)\]\s*$', answer)
        if match:
            try:
                import json
                image_ids = json.loads(match.group(0))[:10]
                clean_answer = answer[:match.start()].strip()
                return clean_answer, image_ids
            except json.JSONDecodeError:
                pass

        return answer, []
