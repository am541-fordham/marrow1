"""
Marrow — RAG Voice Agent (Anthropic API)
"""
 
import os
import json
import re
from pathlib import Path
 
import anthropic
import chromadb
from chromadb.utils import embedding_functions
 
# ── CONFIG ──────────────────────────────────────────────────────
PERSONAS_DIR  = Path("data/personas")
CHROMA_PATH   = "./data/chroma_db"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 60
MODEL         = "claude-haiku-4-5-20251001"   # fast + cheap for demo
 
# ── CLIENTS ─────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
 
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)
 
 
# ════════════════════════════════════════════════════════════════
# LLM HELPER
# ════════════════════════════════════════════════════════════════
 
def call_llm(system: str, user: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=1200,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text.strip()
 
 
# ════════════════════════════════════════════════════════════════
# 1. INGESTION
# ════════════════════════════════════════════════════════════════
 
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks
 
 
def parse_persona_file(filepath):
    content = filepath.read_text(encoding="utf-8")
    name, meta, tags, body_lines = "", "", [], []
    for line in content.strip().split("\n"):
        if line.startswith("NAME:"):
            name = line.replace("NAME:", "").strip()
        elif line.startswith("META:"):
            meta = line.replace("META:", "").strip()
        elif line.startswith("TAGS:"):
            tags = [t.strip() for t in line.replace("TAGS:", "").split(",")]
        else:
            body_lines.append(line)
    return {"name": name, "meta": meta, "tags": tags, "raw_text": "\n".join(body_lines).strip()}
 
 
def ingest_personas(force=False):
    personas = {}
    for filepath in sorted(PERSONAS_DIR.glob("*.txt")):
        persona_id = filepath.stem
        persona    = parse_persona_file(filepath)
        personas[persona_id] = persona
 
        cname    = f"persona_{persona_id}"
        existing = [c.name for c in chroma_client.list_collections()]
 
        if cname in existing and not force:
            print(f"  v {persona['name']} already ingested, skipping.")
            continue
        if cname in existing:
            chroma_client.delete_collection(cname)
 
        col    = chroma_client.create_collection(name=cname, embedding_function=embed_fn)
        chunks = chunk_text(persona["raw_text"])
        col.add(
            documents=chunks,
            ids=[f"{persona_id}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"persona_id": persona_id, "chunk_index": i} for i in range(len(chunks))]
        )
        print(f"  v Ingested {persona['name']}: {len(chunks)} chunks")
    return personas
 
 
# ════════════════════════════════════════════════════════════════
# 2. RETRIEVAL
# ════════════════════════════════════════════════════════════════
 
def retrieve_chunks(persona_id, query, n_results=8):
    try:
        col     = chroma_client.get_collection(name=f"persona_{persona_id}", embedding_function=embed_fn)
        results = col.query(query_texts=[query], n_results=min(n_results, col.count()))
        return results["documents"][0] if results["documents"] else []
    except Exception:
        return []
 
 
def get_all_chunks(persona_id):
    try:
        col     = chroma_client.get_collection(name=f"persona_{persona_id}", embedding_function=embed_fn)
        results = col.get()
        return results["documents"] if results["documents"] else []
    except Exception:
        return []
 
 
# ════════════════════════════════════════════════════════════════
# 3. VOICE MAP
# ════════════════════════════════════════════════════════════════
 
VOICE_MAP_SYSTEM = """You are a voice analyst. Analyze writing samples and extract a precise fingerprint of HOW this person communicates — not WHAT they say.
 
Return ONLY a valid JSON object. No markdown, no preamble, nothing outside the braces.
 
Keys:
{
  "thinking":        "how they structure arguments (2-3 sentences)",
  "writing":         "sentence length, rhythm, paragraph style (2-3 sentences)",
  "humor":           "when and how humor appears (1-2 sentences)",
  "certainty":       "how they express confidence (1-2 sentences)",
  "doubt":           "how they express uncertainty (1-2 sentences)",
  "metaphors":       "what images they naturally reach for (1-2 sentences)",
  "opening":         "how they typically begin thoughts (1-2 sentences)",
  "closing":         "how they end thoughts (1-2 sentences)",
  "emotional":       "their emotional register (2 sentences)",
  "distinctiveness": "the single most unique quality of their voice (2 sentences)",
  "identity_prompt": "3-sentence system prompt capturing this voice"
}"""
 
 
def _parse_json_obj(raw):
    raw = re.sub(r"```json|```", "", raw).strip()
    m   = re.search(r'\{.*\}', raw, re.DOTALL)
    return json.loads(m.group(0) if m else raw)
 
 
def _parse_json_arr(raw):
    raw = re.sub(r"```json|```", "", raw).strip()
    m   = re.search(r'\[.*\]', raw, re.DOTALL)
    return json.loads(m.group(0) if m else raw)
 
 
def build_voice_map(persona_id, persona_name, persona_meta):
    chunks = get_all_chunks(persona_id)
    if not chunks:
        raise ValueError(f"No chunks found for: {persona_id}")
    samples = "\n\n---\n\n".join(chunks)
    raw     = call_llm(
        system=VOICE_MAP_SYSTEM,
        user=f"Analyze these writing samples from {persona_name} ({persona_meta}).\nReturn only the JSON.\n\n{samples}"
    )
    return _parse_json_obj(raw)
 
 
# ════════════════════════════════════════════════════════════════
# 4. GENERATION
# ════════════════════════════════════════════════════════════════
 
def generate_content(persona_id, persona_name, voice_map, request):
    m        = voice_map
    relevant = retrieve_chunks(persona_id, request, n_results=5)
    examples = "\n\n".join(relevant)
 
    system = f"""You are a writing agent for {persona_name}. Write EXACTLY as they would — not cleaner, not safer.
 
Voice fingerprint:
- Thinking:      {m.get('thinking','')}
- Writing:       {m.get('writing','')}
- Humor:         {m.get('humor','')}
- Certainty:     {m.get('certainty','')}
- Doubt:         {m.get('doubt','')}
- Metaphors:     {m.get('metaphors','')}
- Opens:         {m.get('opening','')}
- Closes:        {m.get('closing','')}
- Emotional:     {m.get('emotional','')}
- Distinct:      {m.get('distinctiveness','')}
 
{m.get('identity_prompt','')}
 
Write the piece. Then write ---REASONING--- on its own line. Then 3-4 bullets on which voice traits you used."""
 
    raw   = call_llm(system=system, user=f"Task: {request}\n\nStyle reference:\n{examples}\n\nWrite in their voice.")
    parts = raw.split("---REASONING---")
    piece = parts[0].strip()
 
    reasoning = []
    if len(parts) > 1:
        for line in parts[1].strip().split("\n"):
            line = re.sub(r"^[-•·*\d.]\s*", "", line).strip()
            if line:
                reasoning.append(line)
 
    return {"content": piece, "reasoning": reasoning}
 
 
# ════════════════════════════════════════════════════════════════
# 5. CONTRAST
# ════════════════════════════════════════════════════════════════
 
CONTRAST_SYSTEM = """You are a voice contrast analyst.
Score each sentence of submitted text against a voice fingerprint.
 
Return ONLY a valid JSON array. Nothing before or after the brackets. No markdown.
 
Each object must use this exact format with double quotes only:
{"sentence": "text here", "signal": "green", "note": "explanation here"}
 
signal values: green (sounds like them), amber (softened), red (disappeared)
Keep notes under 12 words. No apostrophes inside strings."""
 
 
def contrast_text(voice_map, text_to_analyse):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_to_analyse) if s.strip()]
    if not sentences:
        sentences = [text_to_analyse.strip()]
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
 
    raw = call_llm(
        system=CONTRAST_SYSTEM,
        user="Fingerprint:\n" + json.dumps(voice_map) + "\n\nSentences:\n" + numbered + "\n\nReturn only the JSON array."
    )
 
    raw = re.sub(r"```json|```", "", raw).strip()
 
    # Try 1: direct array parse
    m = re.search(r'\[.*\]', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
 
    # Try 2: fix quotes
    fixed = raw.replace("'", '"')
    m2 = re.search(r'\[.*\]', fixed, re.DOTALL)
    if m2:
        try:
            return json.loads(m2.group(0))
        except Exception:
            pass
 
    # Fallback: never break the UI
    return [{"sentence": s, "signal": "amber", "note": "Could not analyse."} for s in sentences]
# ════════════════════════════════════════════════════════════════
# 6. OWN TEXT
# ════════════════════════════════════════════════════════════════
 
def ingest_own_text(text, session_id):
    cname    = f"own_{session_id}"
    existing = [c.name for c in chroma_client.list_collections()]
    if cname in existing:
        chroma_client.delete_collection(cname)
    col    = chroma_client.create_collection(name=cname, embedding_function=embed_fn)
    chunks = chunk_text(text)
    col.add(
        documents=chunks,
        ids=[f"{session_id}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"session_id": session_id, "chunk_index": i} for i in range(len(chunks))]
    )
 
 
def build_voice_map_from_text(text, session_id):
    ingest_own_text(text, session_id)
    samples = "\n\n---\n\n".join(chunk_text(text))
    raw     = call_llm(
        system=VOICE_MAP_SYSTEM,
        user=f"Analyze these writing samples and build a voice fingerprint. Return only the JSON.\n\n{samples}"
    )
    return _parse_json_obj(raw)
