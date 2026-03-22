"""
Marrow — FastAPI Server
"""

import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.rag_agent import (
    ingest_personas,
    build_voice_map,
    build_voice_map_from_text,
    generate_content,
    contrast_text,
)

# ── In-memory session store (voice maps + persona metadata) ──
# In production replace with Redis or a DB
sessions: dict[str, dict] = {}
personas_meta: dict[str, dict] = {}


# ── Lifespan: ingest personas on startup ────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global personas_meta
    print("\n🌱 Marrow is starting up...")
    print("   Ingesting persona documents into ChromaDB...")
    personas_meta = ingest_personas(force=False)
    print(f"   ✓ {len(personas_meta)} personas ready.\n")
    yield
    print("👋 Marrow shutting down.")


app = FastAPI(title="Marrow", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ════════════════════════════════
# PYDANTIC MODELS
# ════════════════════════════════

class PersonaRequest(BaseModel):
    persona_id: str

class OwnTextRequest(BaseModel):
    text: str

class GenerateRequest(BaseModel):
    session_id: str
    request: str

class ContrastRequest(BaseModel):
    session_id: str
    text: str


# ════════════════════════════════
# ROUTES
# ════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/personas")
async def list_personas():
    """Return all available personas for the UI."""
    result = []
    for pid, p in personas_meta.items():
        # Show a snippet of the raw text as preview
        raw = p.get("raw_text", "")
        lines = [l.strip() for l in raw.split("\n") if l.strip() and not l.startswith("[")]
        preview = lines[0] if lines else ""
        result.append({
            "id": pid,
            "name": p["name"],
            "meta": p["meta"],
            "tags": p["tags"],
            "preview": preview[:160],
        })
    return result


@app.post("/api/voice-map/persona")
async def voice_map_persona(body: PersonaRequest):
    """
    RAG pipeline: retrieve all chunks for persona → Claude analysis → Voice Map.
    Creates a session and returns session_id + voice map.
    """
    pid = body.persona_id
    if pid not in personas_meta:
        raise HTTPException(status_code=404, detail=f"Persona '{pid}' not found.")

    p = personas_meta[pid]
    try:
        voice_map = build_voice_map(pid, p["name"], p["meta"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "persona_id": pid,
        "persona_name": p["name"],
        "persona_meta": p["meta"],
        "voice_map": voice_map,
    }

    return {
        "session_id": session_id,
        "persona_name": p["name"],
        "persona_meta": p["meta"],
        "voice_map": voice_map,
    }


@app.post("/api/voice-map/own")
async def voice_map_own(body: OwnTextRequest):
    """
    Build a voice map from the user's own pasted text.
    Chunks + embeds into a temp ChromaDB collection, then analyses.
    """
    if len(body.text.strip()) < 80:
        raise HTTPException(status_code=400, detail="Please provide at least a few sentences.")

    session_id = str(uuid.uuid4())
    try:
        voice_map = build_voice_map_from_text(body.text, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sessions[session_id] = {
        "persona_id": f"own_{session_id}",
        "persona_name": "Your Voice",
        "persona_meta": "Personal analysis",
        "voice_map": voice_map,
    }

    return {
        "session_id": session_id,
        "persona_name": "Your Voice",
        "persona_meta": "Personal analysis",
        "voice_map": voice_map,
    }


@app.post("/api/generate")
async def generate(body: GenerateRequest):
    """
    RAG-augmented generation:
    Retrieves relevant writing chunks, then generates content in the persona's voice.
    """
    session = sessions.get(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Build a Voice Map first.")

    try:
        result = generate_content(
            persona_id=session["persona_id"],
            persona_name=session["persona_name"],
            voice_map=session["voice_map"],
            request=body.request,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


@app.post("/api/contrast")
async def contrast(body: ContrastRequest):
    """Score pasted text against the active Voice Map."""
    session = sessions.get(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Build a Voice Map first.")

    try:
        results = contrast_text(
            voice_map=session["voice_map"],
            text_to_analyse=body.text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"results": results}


@app.get("/api/health")
async def health():
    return {"status": "ok", "personas_loaded": len(personas_meta)}
