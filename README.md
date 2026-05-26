# AI Customer Service Agent

A multimodal RAG-powered chatbot that answers customer questions using product manuals. Supports text and image input, multi-turn dialogue, and retrieval-augmented generation.

> Architecture is language-agnostic — swap in any product manual corpus and it works the same way.

## Tech Stack

| Module | Technology |
|--------|-----------|
| Web Framework | FastAPI |
| Embedding | BAAI/bge-large-zh-v1.5 |
| Vector Search | FAISS IndexFlatIP |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | OpenAI-compatible API (GPT-4o, Claude, Qwen, etc.) |
| Frontend | Single-page HTML/CSS/JS chat interface |

## Architecture

```
User Question → FastAPI → Question Classifier → RAG Pipeline
                                          │
                  ┌───────────────────────┘
                  ▼
          FAISS Vector Store → BGE Reranker → Top-5 Chunks
                  │
                  ▼
          LLM (text or multimodal) → Answer + Image References
```

## Results

| Metric | Value |
|--------|-------|
| Total Questions | 400 |
| Success Rate | 100% |
| Questions with Images | 55.5% |
| Avg. Answer Length | 522 chars |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Build Vector Index

```bash
python -m scripts.build_index
```

### 4. Start the API Server

```bash
python -m src.main
```

### 5. Test

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk_customer_20260518" \
  -d '{"question": "What do the DCB107 indicator lights mean?"}'
```

### 6. Batch Answer Generation

```bash
python -m scripts.batch_answer --input question_public.csv --output submission.csv
```

## API

### POST /api/v1/chat

```json
// Request
{
  "question": "What do the DCB107 indicator lights mean?",
  "images": [],
  "session_id": "optional-session-id"
}

// Response
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "The DCB107 indicator lights indicate: <PIC> ...",
    "session_id": "session-uuid",
    "timestamp": 1716000000
  }
}
```

## Project Structure

```
customer_service/
├── src/
│   ├── api/              # FastAPI routes (POST /api/v1/chat)
│   ├── core/             # ChatEngine, LLMClient, Prompts
│   ├── rag/              # ManualParser, Embedder, VectorStore, Reranker
│   ├── config.py         # pydantic-settings configuration
│   ├── app.py            # FastAPI application entry
│   └── main.py           # uvicorn launcher
├── scripts/
│   ├── build_index.py    # Build vector index
│   ├── batch_answer.py   # Batch answer generation
│   ├── fix_empty.py      # Fix empty answers
│   └── analyze_dataset.py
├── docs/
│   └── ARCHITECTURE.md   # Architecture design
├── data/
│   ├── vector_store/     # FAISS index
│   └── chunks/           # Text chunk metadata
├── requirements.txt
└── vercel.json           # Vercel deployment config
```

## Key Features

- **RAG Knowledge Base**: Vector search over 21 product manuals with FAISS + BGE Reranker for grounded, accurate answers
- **Multimodal Understanding**: Accepts text and image input — recognizes product fault photos and matches them with repair guides
- **Multi-turn Dialogue**: Maintains conversation context for follow-ups, clarifications, and topic switches
- **Hallucination Prevention**: Only answers based on retrieved knowledge base content — never fabricates information
