"""FastAPI routes for the customer service chatbot."""

import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel, Field

from src.config import settings
from src.core.chat_engine import ChatEngine

router = APIRouter()

# Singleton chat engine - initialized on startup
_engine: ChatEngine | None = None


def get_engine() -> ChatEngine:
    global _engine
    if _engine is None:
        _engine = ChatEngine()
        _engine.initialize()
    return _engine


def verify_token(authorization: str = Header(...)):
    """Verify Bearer token."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    token = authorization[7:]
    if token != settings.kafu_api_token:
        raise HTTPException(status_code=401, detail="Invalid token")


# --- Request / Response Models ---

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")
    images: list[str] = Field(default=[], description="Base64 encoded images")
    session_id: Optional[str] = Field(default=None, description="Session ID for multi-turn")
    stream: bool = Field(default=False, description="Stream response (not implemented)")


class ChatResponseData(BaseModel):
    answer: str
    session_id: str
    timestamp: int
    images: list[dict] = []
    chunks: list[dict] = []
    type: str = "general"


class ChatResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: ChatResponseData


# --- Endpoints ---

@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_token)])
async def chat(request: ChatRequest):
    """Main chat endpoint for customer service agent."""
    engine = get_engine()
    start_time = time.time()

    try:
        result = engine.answer(
            question=request.question,
            images=request.images if request.images else None,
            session_id=request.session_id,
        )

        session_id = result.get("session_id", str(uuid.uuid4()))

        # Build images list for frontend
        images = [{"id": img_id} for img_id in result.get("image_ids", [])]

        return ChatResponse(
            code=0,
            msg="success",
            data=ChatResponseData(
                answer=result["answer"],
                session_id=session_id,
                timestamp=int(time.time()),
                images=images,
                chunks=result.get("chunks", []),
                type=result.get("question_type", "general"),
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": int(time.time())}
