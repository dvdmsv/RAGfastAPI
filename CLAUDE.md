# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A local RAG (Retrieval-Augmented Generation) system built with FastAPI (backend) and Streamlit (frontend). All inference runs locally via **LM Studio** — no external AI APIs are used. Documents are indexed in a local Qdrant vector database using HuggingFace sentence-transformer embeddings.

## Running the Project

**Prerequisite:** LM Studio must be running with a model loaded on port 1234.

### Option A — Docker (standard Linux/macOS/Windows)
```bash
docker compose up --build
docker compose down
docker compose logs -f
```

### Option B — Docker + WSL (Windows LM Studio + WSL2)
```bash
./scripts/docker-compose-wsl.sh up --build
./scripts/docker-compose-wsl.sh down
```

### Option C — Native Python (no Docker)
```bash
bash start_local.sh          # auto-installs deps, starts both services
pkill -f uvicorn && pkill -f streamlit   # stop
tail -f backend_local.log    # view logs
```

**Ports:**
- Backend (FastAPI): `http://localhost:8000`
- Frontend (Streamlit): `http://localhost:8501`
- LM Studio: `http://localhost:1234` (native) or `http://host.docker.internal:1234` (Docker)

**Configuration** is read from `.venv` (not `.env`) — this file sets `OPENAI_API_BASE`, `LLM_MODEL`, `EMBEDDING_MODEL`, `QDRANT_PATH`, etc.

## Architecture

```
frontend.py (Streamlit)
    └── HTTP → main.py (FastAPI)
                    └── app/services/rag.py (RAG logic)
                              ├── Qdrant (./data/qdrant_db/)  — vector store
                              ├── HuggingFace embeddings       — local, no GPU required
                              └── LM Studio (port 1234)        — OpenAI-compatible LLM
```

### Backend (`main.py`)
FastAPI app exposing REST endpoints. Upload and indexing handlers are intentionally **non-async** (blocking) for thread safety with the Qdrant client. Chat uses `StreamingResponse`.

| Endpoint | Purpose |
|----------|---------|
| `POST /api/upload` | Ingest and index a document |
| `GET /api/documents` | List indexed documents |
| `DELETE /api/documents/{filename}` | Remove a document |
| `POST /api/chat` | Streaming RAG chat response |
| `POST /api/indexar` | Manual re-indexing trigger |
| `GET /health` | Health check |

### Frontend (`frontend.py`)
Streamlit app. Chat sessions are UUID-keyed and stored in Streamlit session state (in-memory only — `data/conversations.json` exists but is not used for persistence). The sidebar manages threads, document upload, and advanced settings (system prompt, temperature).

### RAG Service (`app/services/rag.py`)
Core logic:
- **Per-session chat engines**: Each UUID session gets its own LlamaIndex `CondensePlusContextChatEngine` instance, recreated if system prompt or temperature changes.
- **Hybrid search**: Dense (embedding) + sparse (BM25) vectors via Qdrant.
- **Time detection**: Queries about current time/date are answered directly, bypassing RAG.
- **Reasoning model support**: `<think>...</think>` tags from models like DeepSeek R1 are intercepted and formatted before streaming to the frontend.
- **Citations**: Source filenames, page numbers, and relevance scores are appended to streamed responses.

## Key Implementation Notes

- The embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) is downloaded on first run and cached by HuggingFace.
- Qdrant data persists in `./data/qdrant_db/`; uploaded documents in `./data/docs/`.
- CORS is set to allow all origins (`*`) — intentional for local dev.
- Python 3.12 is required (specified in Dockerfile).
- There are no automated tests in this repository.
