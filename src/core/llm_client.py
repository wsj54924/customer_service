"""LLM client for chat and vision APIs."""

import base64
from pathlib import Path
from loguru import logger
from src.config import settings


class LLMClient:
    """Unified LLM client supporting OpenAI-compatible and DashScope APIs."""

    def __init__(self):
        self._client = None
        self._provider = None

    @staticmethod
    def _is_valid_key(key: str) -> bool:
        return bool(key) and key.strip() not in ("", "sk-xxx") and len(key) > 10

    def _get_client(self):
        if self._client is not None:
            return self._client, self._provider

        from openai import OpenAI

        # Try OpenAI-compatible first (covers MiMo, OpenAI, any compatible API)
        if self._is_valid_key(settings.openai_api_key):
            self._client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
            self._provider = "openai-compatible"
            logger.info(f"Using OpenAI-compatible API: {settings.openai_base_url}")
        # Fallback to DashScope
        elif self._is_valid_key(settings.dashscope_api_key):
            self._client = OpenAI(
                api_key=settings.dashscope_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            self._provider = "dashscope"
            logger.info("Using DashScope (Qwen) API")
        else:
            raise RuntimeError(
                "No valid LLM API key found. "
                "Set OPENAI_API_KEY or DASHSCOPE_API_KEY in .env file."
            )

        return self._client, self._provider

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name override
            temperature: Sampling temperature
            max_tokens: Max tokens in response

        Returns:
            The assistant's reply text
        """
        client, provider = self._get_client()
        model = model or settings.chat_model

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    def chat_with_images(
        self,
        question: str,
        image_base64_list: list[str],
        context: str = "",
        system_prompt: str | None = None,
    ) -> str:
        """Chat with text and images (multimodal).

        Args:
            question: User's question text
            image_base64_list: List of base64-encoded images
            context: RAG-retrieved context
            system_prompt: System prompt override

        Returns:
            The assistant's reply text
        """
        client, provider = self._get_client()
        model = settings.vision_model

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build user message with text and images
        user_content = []

        if context:
            user_content.append({
                "type": "text",
                "text": f"以下是客服知识库中的相关内容，请基于这些信息回答用户问题：\n\n{context}",
            })

        user_content.append({
            "type": "text",
            "text": f"用户问题：{question}",
        })

        # Add images
        for img_b64 in image_base64_list:
            if not img_b64.startswith("data:"):
                img_b64 = f"data:image/png;base64,{img_b64}"
            user_content.append({
                "type": "image_url",
                "image_url": {"url": img_b64},
            })

        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
        )

        return response.choices[0].message.content

    def chat_text_only(
        self,
        question: str,
        context: str = "",
        system_prompt: str | None = None,
    ) -> str:
        """Text-only chat for non-multimodal queries."""
        client, provider = self._get_client()
        model = settings.chat_model

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_text = ""
        if context:
            user_text += f"以下是客服知识库中的相关内容，请基于这些信息回答用户问题：\n\n{context}\n\n"
        user_text += f"用户问题：{question}"

        messages.append({"role": "user", "content": user_text})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
        )

        return response.choices[0].message.content
