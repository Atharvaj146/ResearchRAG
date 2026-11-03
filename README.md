# ResearchRAG 🔬

An **Agentic Multi-Document Research Assistant** powered by a LangGraph agent, hybrid search (vector + BM25), cross-encoder reranking, and a React frontend.

---

## Architecture

```
PDF upload → PyMuPDF (text extraction) → chunk → embed (sentence-transformers)
         → Qdrant (vector store)
                ↓
User question → LangGraph Agent
                  ├── decide (conversational vs retrieval vs summary)
                  ├── rewrite query (using chat history)
                  ├── hybrid search (vector + BM25, RRF fusion)
                  ├── cross-encoder rerank
                  └── generate answer (OpenRouter LLM)
```

## Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, Vite |
| Backend | FastAPI, Python 3.11 |
| Agent | LangGraph |
| Embeddings | `sentence-transformers` (local) |
| Vector DB | Qdrant (cloud or local) |
| Keyword search | BM25 (`rank-bm25`) |
| Reranker | Cross-encoder (`sentence-transformers`) |
| LLM | OpenRouter (`qwen/qwen3-8b-instruct:free`) |
| PDF parsing | PyMuPDF (`fitz`) |
| Containerisation | Docker + Docker Compose |

---

## Quickstart (Docker Compose)

### Prerequisites
- Docker Desktop
- A running Qdrant instance (cloud or `docker run -p 6333:6333 qdrant/qdrant`)
- An [OpenRouter](https://openrouter.ai/) API key

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

### 2. Build and run

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |

---

## Local Development

### Backend

```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

# Copy and fill in the env file
cp .env.example .env

uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Environment Variables

| Variable | Where | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | root `.env` / `backend/.env` | OpenRouter API key |
| `QDRANT_URL` | `backend/.env` | Qdrant instance URL |
| `QDRANT_API_KEY` | `backend/.env` | Qdrant API key (cloud only) |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload a PDF document |
| `POST` | `/ask` | Ask a question (with optional chat history) |

---

## Features

- **Multi-document support** — upload and query across multiple PDFs in a session
- **Hybrid search** — combines dense vector search (cosine) with sparse BM25 for better recall
- **Cross-encoder reranking** — re-scores retrieved chunks for higher precision
- **Agentic loop** — LangGraph agent decides whether to retrieve, summarize, or answer directly
- **Query rewriting** — resolves pronouns/references using conversation history
- **Conversation memory** — full chat history passed from the frontend
- **Docker-ready** — production Dockerfiles for both services

---

## License

MIT
