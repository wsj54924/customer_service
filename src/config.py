"""Application configuration."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    # Paths
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "鎵嬪唽"
    image_dir: Path = BASE_DIR / "鎵嬪唽" / "鎻掑浘"
    vector_store_dir: Path = BASE_DIR / "data" / "vector_store"
    chunks_dir: Path = BASE_DIR / "data" / "chunks"
    processed_dir: Path = BASE_DIR / "data" / "processed"

    # LLM API (OpenAI-compatible, e.g. MiMo, OpenAI, etc.)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Ollama (local embedding)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # DashScope (optional, for cloud embedding/reranking)
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")

    # Models
    chat_model: str = os.getenv("CHAT_MODEL", "qwen-plus")
    vision_model: str = os.getenv("VISION_MODEL", "qwen-vl-max")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    embedding_model_fallback: str = "text-embedding-v3"
    reranker_model: str = os.getenv("RERANKER_MODEL", "")

    # Auth
    kafu_api_token: str = os.getenv("KAFU_API_TOKEN", "sk_customer_20260518")

    # RAG settings
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 10
    top_k_rerank: int = 5

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()
