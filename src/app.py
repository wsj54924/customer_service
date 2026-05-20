"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from src.api.routes import router, get_engine

# Project root: 2 levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize chat engine on startup."""
    logger.info("Initializing chat engine...")
    try:
        engine = get_engine()
        logger.info("Chat engine ready")
    except Exception as e:
        logger.warning(f"Chat engine init deferred: {e}")
        logger.info("Will initialize on first request")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="多模态客服智能体 API",
    description="具有多模态能力的客服智能体 - 竞赛作品",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router, prefix="/api/v1")

# Static: serve product images — try both .png and .jpg
IMAGES_DIR = PROJECT_ROOT / "手册" / "插图"


@app.get("/images/{image_id}")
async def get_image(image_id: str):
    """Serve product images, auto-detecting .png / .jpg extension."""
    # Strip extension if provided
    stem = Path(image_id).stem
    for ext in (".png", ".jpg", ".jpeg"):
        path = IMAGES_DIR / f"{stem}{ext}"
        if path.is_file():
            media = "image/jpeg" if ext == ".jpg" else "image/png"
            return FileResponse(str(path), media_type=media)
    raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

# Static: serve frontend design files
FRONTEND_DIR = PROJECT_ROOT / "106f55bd-f914-447f-89c9-1215d77c8e12"
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the chat frontend."""
    index = FRONTEND_DIR / "smartservice-ai-chatbot.html"
    if index.is_file():
        return FileResponse(str(index), media_type="text/html")
    return {
        "service": "多模态客服智能体",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/v1/chat",
            "health": "/api/v1/health",
        },
    }


@app.get("/health")
async def health_root():
    return {"status": "ok"}
