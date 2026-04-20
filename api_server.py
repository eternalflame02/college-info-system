"""
Small FastAPI wrapper exposing the chatbot as a simple HTTP API.

Endpoints:
- POST /chat  -> { message: str } -> ChatResponse as JSON
- GET  /stats -> simple knowledge-base metrics (chunks, faculty)

This module intentionally performs lazy imports of the heavy `chatbot`
resources to keep module import fast; warmup is executed on startup
when `CHAT_WARMUP_ON_STARTUP` != '0'.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import os
import asyncio
import dataclasses
import logging
import config
from pathlib import Path

app = FastAPI(title="MBCET CSE Assistant API")
logger = logging.getLogger(__name__)

# Serve the static frontend from the backend for a single-origin integration.
app.mount("/frontend", StaticFiles(directory=str(Path(__file__).parent / "frontend")), name="frontend")

def _get_allowed_origins() -> list[str]:
    """Resolve CORS allowlist from env with safe localhost/public defaults."""
    defaults = ["http://127.0.0.1:8000", "http://localhost:8000"]

    public_base = str(getattr(config, "PUBLIC_BASE_URL", "") or "").strip()
    if public_base:
        defaults.append(public_base.rstrip("/"))

    ngrok_domain = str(getattr(config, "NGROK_DOMAIN", "") or "").strip()
    if ngrok_domain:
        defaults.append(f"https://{ngrok_domain}")

    raw = os.getenv("CORS_ALLOW_ORIGINS", ",".join(defaults))
    origins = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]

    if not origins:
        origins = defaults

    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for origin in origins:
        if origin not in seen:
            ordered.append(origin)
            seen.add(origin)
    return ordered


allowed_origins = _get_allowed_origins()

# CORS defaults to localhost only; override with CORS_ALLOW_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.on_event("startup")
async def startup_event():
    """Optionally warmup heavy chatbot resources on server start."""
    warm = os.getenv("CHAT_WARMUP_ON_STARTUP", "1")
    if warm != "0":
        try:
            # lazy import to avoid heavy modules on simple imports
            import chatbot as _chatbot

            # run warmup in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _chatbot.warmup)
        except Exception as exc:
            # best-effort warmup; log to console and continue
            print("[api_server] chatbot.warmup() failed:", exc)


@app.get("/", response_class=HTMLResponse)
def root():
    html_path = Path(__file__).parent / "frontend" / "cse_department.html"
    if not html_path.exists():
        return {"status": "error", "message": "Frontend file not found."}
    return html_path.read_text(encoding="utf-8")


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    try:
        # lazy import to reduce initial import cost
        import chatbot as _chatbot

        response = _chatbot.answer_question(req.message)

        # dataclasses -> primitives
        resp_dict = dataclasses.asdict(response)

        # use jsonable_encoder to handle any numpy or non-standard types
        return jsonable_encoder(resp_dict)
    except Exception as exc:
        logger.exception("/chat request failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stats")
def stats():
    """Return simple KB metrics derived from local data files when available."""
    import json
    chunks_count = None
    faculty_count = None

    try:
        if config.CHUNKS_FILE.exists():
            with open(config.CHUNKS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    chunks_count = len(data)
    except Exception:
        chunks_count = None

    try:
        if config.FACULTY_FILE.exists():
            with open(config.FACULTY_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    faculty_count = len(data)
    except Exception:
        faculty_count = None

    return {
        "chunks": chunks_count,
        "faculty": faculty_count,
        "collection": str(config.CHROMADB_COLLECTION),
    }
