# Marrow — RAG Voice Agent

> *"When did you last write something that sounded exactly like you?"*

Marrow is a RAG (Retrieval-Augmented Generation) agent that reads real writing samples, builds a voice fingerprint, and lets you generate anything in that exact voice.

---

## Architecture

```
data/personas/*.txt          ← Raw writing samples (texts, emails, journal entries)
        ↓
   ChromaDB                  ← Chunked + embedded via SentenceTransformers
        ↓
   Retrieval                 ← Semantic search for relevant chunks
        ↓
   Claude (claude-sonnet)    ← Voice Map analysis + generation + contrast
        ↓
   FastAPI                   ← REST API
        ↓
   Browser UI                ← Warm, human frontend
```

---

## Quick Start

### 1. Clone / navigate to the project
```bash
cd marrow
```

### 2. Set your Anthropic API key
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Run with one command
```bash
bash run.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Ingest persona documents into ChromaDB
- Start the server on http://localhost:8000

---

## Manual Setup (if you prefer)

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Run
python3 main.py
```

---

## Adding Your Own Personas

Create a `.txt` file in `data/personas/` with this format:

```
NAME: Your Name
META: Age, Role, Context
TAGS: Tag1, Tag2, Tag3

[Source label]
Your actual writing here...

[Another source]
More writing...
```

Restart the server. Your persona will be automatically ingested.

To force re-ingestion of all personas:
```python
# In Python
from app.rag_agent import ingest_personas
ingest_personas(force=True)
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Frontend UI |
| GET | `/api/personas` | List all personas |
| POST | `/api/voice-map/persona` | Build Voice Map from a persona |
| POST | `/api/voice-map/own` | Build Voice Map from pasted text |
| POST | `/api/generate` | Generate content in a voice |
| POST | `/api/contrast` | Score text against a Voice Map |
| GET | `/api/health` | Health check |

---

## Project Structure

```
marrow/
├── main.py                  ← Entry point
├── run.sh                   ← One-command setup + run
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── server.py            ← FastAPI routes
│   └── rag_agent.py         ← RAG pipeline (ingest/retrieve/analyse/generate)
├── data/
│   ├── personas/            ← .txt writing sample files
│   │   ├── nadia.txt
│   │   ├── darius.txt
│   │   ├── priya.txt
│   │   ├── theo.txt
│   │   ├── simone.txt
│   │   └── felix.txt
│   └── chroma_db/           ← Auto-created vector store
├── templates/
│   └── index.html           ← Full UI
└── static/                  ← Static assets (if needed)
```

---

## How the RAG Pipeline Works

1. **Ingest** — Each `.txt` file is chunked (300 chars, 60 overlap), embedded via `all-MiniLM-L6-v2`, stored in ChromaDB
2. **Retrieve** — When building a Voice Map, all chunks are retrieved. When generating, the most semantically relevant chunks to the request are retrieved
3. **Analyse** — Claude receives the retrieved chunks and extracts a structured Voice Map JSON
4. **Generate** — Claude receives the Voice Map as system context + relevant writing examples, then writes in that voice
5. **Contrast** — Claude scores pasted text against the Voice Map sentence by sentence

---

## Requirements

- Python 3.10+
- Anthropic API key
- ~500MB disk (for SentenceTransformer model, downloaded once)
- Internet connection (for Claude API calls)
