#!/usr/bin/env python3
"""Build the RAG vector index from manual files.

Usage:
    python -m scripts.build_index
"""

import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from loguru import logger
from src.config import settings
from src.rag.manual_parser import ManualParser
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore


def main():
    logger.info("Starting index build...")

    # 1. Parse all manuals
    parser = ManualParser(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    parsed_manuals = parser.parse_all(settings.data_dir)

    # Collect all chunks
    all_chunks = []
    for manual in parsed_manuals:
        all_chunks.extend(manual.chunks)

    logger.info(f"Total chunks from all manuals: {len(all_chunks)}")

    if not all_chunks:
        logger.error("No chunks generated. Check manual files.")
        return

    # Save chunks metadata for inspection
    chunks_meta = []
    for chunk in all_chunks:
        chunks_meta.append({
            "chunk_id": chunk.chunk_id,
            "manual_name": chunk.manual_name,
            "text_preview": chunk.text[:100],
            "text_len": len(chunk.text),
            "image_ids": chunk.image_ids,
        })

    meta_path = settings.chunks_dir / "chunks_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks_meta, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved chunks metadata to {meta_path}")

    # 2. Generate embeddings
    embedder = Embedder()
    embeddings = embedder.embed_chunks(all_chunks)
    logger.info(f"Generated embeddings: shape={embeddings.shape}")

    # 3. Build and save FAISS index
    store = VectorStore(settings.vector_store_dir)
    store.build(embeddings, all_chunks)
    store.save()

    logger.info("Index build complete!")
    logger.info(f"  Chunks: {len(all_chunks)}")
    logger.info(f"  Embedding dim: {embeddings.shape[1]}")
    logger.info(f"  Store location: {settings.vector_store_dir}")


if __name__ == "__main__":
    main()
