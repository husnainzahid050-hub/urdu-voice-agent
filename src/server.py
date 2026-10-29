"""Browser front end: a web app with a mic button, chat bubbles and spoken replies.

Run:  python -m src.server        (or double-click start.bat)
Locally it opens http://127.0.0.1:7860 by itself. With APP_MODE=cloud (see
docs/DEPLOY.md) the same app serves strangers through a public link: every visitor gets
their own conversation, and rate limits protect the owner's free Gemini quota.

API (all JSON unless noted). Requests may send `X-Session-Id` (one per browser tab);
without it the client IP is the session.
    GET  /api/status             models loaded? which voice?
    POST /api/reset              new chat: forget this visitor's conversation
    POST /api/turn    (form)     audio -> NDJSON stream: heard, reply sentences, audio
    POST /api/chat               {"text": ...} -> NDJSON stream (typed message)
    POST /api/listen  (form)     audio file from the browser -> what the user said
    POST /api/reply              {"text": ...} -> agent's reply + audio (non-streaming)
    POST /api/persona            {"gender": "female"|"male"} -> switch voice
    GET  /healthz                liveness for hosting platforms

Streaming events (one JSON object per line), in order:
    {"type":"heard","text","stt_seconds","speech_seconds"}      /api/turn only
    {"type":"not_heard","text","audio","mime"}                  nothing understood; then done
    {"type":"reply_text","index","text"}                        one sentence at a time
    {"type":"audio","index","audio","mime"}                     base64 audio in index order;
                                                                 an index may be skipped
    {"type":"done","reply","goodbye","voice_backend","timings"}
    {"type":"error","message"}
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
import webbrowser
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import audio_io
from src.config import ROOT_DIR, Settings, WebSettings, setup_logging
from src.errors import RateLimitError, STTError, STTQuotaError, VoiceAgentError
from src.limits import RateLimiter, client_ip
from src.pipeline import RATE, VOICES, ChatSession, Conversation, Heard, Reply, TurnTimings
from src.tts import Speech

logger = logging.getLogger(__name__)

WEB = WebSettings.from_env()

# Local by default. A host like Render sets PORT; then listen on all interfaces and
# don't try to open a browser (HEADLESS=1 does the same locally).
PORT = int(os.getenv("PORT", "7860"))
HOST = os.getenv("HOST", "0.0.0.0" if "PORT" in os.environ else "127.0.0.1")
HEADLESS = os.getenv("HEADLESS", "") == "1" or "PORT" in os.environ or WEB.cloud
WEB_DIR = ROOT_DIR / "web"

# Abuse and accident limits (a public link must not be able to burn the free quota).
MAX_UPLOAD_BYTES = 3 * 1024 * 1024
MAX_BODY_BYTES = MAX_UPLOAD_BYTES + 64 * 1024  # multipart overhead
MAX_AUDIO_SECONDS = 30
MAX_TEXT_CHARS = 500
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
# What browsers send for a recording (MediaRecorder webm/ogg/mp4, or a WAV built in JS).
AUDIO_TYPES = {
    "",
    "application/octet-stream",
    "audio/aac",
    "audio/mp4",
    "audio/mpeg",
    "audio/ogg",
    "audio/vnd.wave",
    "audio/wav",
    "audio/wave",
    "audio/webm",
    "audio/x-m4a",
    "audio/x-wav",
    "video/webm",
}
GENERIC_ERROR = "Something went wrong. Please try again."
# Shown when Google's free tier is used up for the day (the page shows it in Urdu too).
QUOTA_MESSAGE = "The free daily limit of this demo is reached. Please try again tomorrow."

# The page loads its fonts from Google Fonts and the browser VAD (onnxruntime-web WASM
# + Silero model) from jsDelivr; audio is played from data: URLs.
CSP = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' https://cdn.jsdelivr.net",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com",
        "connect-src 'self' data: blob: https://cdn.jsdelivr.net",
        "worker-src 'self' blob: https://cdn.jsdelivr.net",
        "media-src 'self' data: blob:",
        "img-src 'self' data:",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ]
)

limiter = RateLimiter(WEB.turns_per_minute, WEB.turns_per_day, WEB.daily_turn_limit)
_state: dict[str, Any] = {"conversation": None, "error": None, "loading": "Starting..."}


def _load_models() -> None:
    """Load all models in the background so the page can show progress."""

    def status(message: str) -> None:
        _state["loading"] = message
        logger.info(message)

    try:
        settings = Settings.from_env()
        conversation = Conversation(settings, status, WEB.session_ttl_seconds, WEB.max_sessions)
    except VoiceAgentError as exc:
        _state["error"] = str(exc)
        logger.error("%s", exc)
        return
    # Warm Gemini, STT and the voice so the first turn is as fast as later ones.
    status("Warming up...")
    conversation.warm_up(timeout=8.0)
    _state["conversation"] = conversation
    _state["loading"] = None
    logger.info("Ready (%s mode) on port %d", settings.app_mode, PORT)


@asynccontextmanager
async def lifespan(_app: FastAPI):  # noqa: ANN201
    threading.Thread(target=_load_models, daemon=True).start()
    if not HEADLESS:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    yield


# No interactive API docs on a public server (smaller attack surface).
_docs = {} if not WEB.cloud else {"docs_url": None, "redoc_url": None, "openapi_url": None}
app = FastAPI(title="Urdu Voice Agent", lifespan=lifespan, **_docs)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
if WEB.allowed_origins:  # only when the page is hosted elsewhere (e.g. Vercel)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(WEB.allowed_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Session-Id"],
        max_age=600,
    )


@app.middleware("http")
async def protect(request: Request, call_next):  # noqa: ANN001, ANN201
    """Reject oversized bodies early and add security headers to every response."""
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > MAX_BODY_BYTES:
        return JSONResponse({"detail": "Request too large."}, status_code=413)
    response = await call_next(request)
    if WEB.security_headers:
        headers = response.headers
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("Referrer-Policy", "no-referrer")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Permissions-Policy", "microphone=(self), camera=(), geolocation=()")
        headers.setdefault("Content-Security-Policy", CSP)
        if WEB.cloud:  # the public host serves HTTPS only
            headers.setdefault("Strict-Transport-Security", "max-age=31536000")
    return response


@app.exception_handler(Exception)
async def unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
    """Never send stack traces or internal paths to the browser."""
    logger.exception("Unhandled error", exc_info=exc)
    return JSONResponse({"detail": GENERIC_ERROR}, status_code=500)


# --- who is asking ---------------------------------------------------------------------


@dataclass(frozen=True)
class Client:
    """The caller: IP (for rate limits) and session key (for the conversation)."""

    ip: str
    session_key: str


def client_info(request: Request, x_session_id: str | None = Header(default=None)) -> Client:
    peer = request.client.host if request.client else None
    ip = client_ip(peer, request.headers.get("x-forwarded-for"), WEB.trust_proxy)
    sid = x_session_id if x_session_id and SESSION_ID.match(x_session_id) else "-"
    # The IP is part of the key, so a guessed session id cannot join someone else's chat.
    return Client(ip, f"{ip}|{sid}")


def limited(client: Client = Depends(client_info)) -> Client:
    """Count one Gemini-costing turn for this client; 429 when over the limit."""
    message = limiter.check(client.ip)
    if message:
        raise HTTPException(429, message)
    return client


def _conversation() -> Conversation:
    conversation = _state["conversation"]
    if conversation is None:
        raise HTTPException(503, _state["error"] or _state["loading"] or "Loading...")
    return conversation


def _session(conversation: Conversation, client: Client) -> ChatSession:
    return conversation.session(client.session_key)


# --- payloads ----------------------------------------------------------------------------


def _audio_payload(speech: Speech | None) -> dict[str, Any]:
    """Speech as a base64 WAV the browser can play directly."""
    if speech is None:
        return {"audio": None}
    buffer = io.BytesIO()
    sf.write(buffer, speech.audio, speech.sample_rate, format="WAV", subtype="PCM_16")
    return {
        "audio": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "voice_backend": speech.backend,
    }


def _reply_payload(reply: Reply) -> dict[str, Any]:
    t = reply.timings
    return {
        "reply": reply.text,
        "goodbye": reply.goodbye,
        "timings": {
            "stt": round(t.stt, 2),
            "llm": round(t.llm, 2),
            "tts": round(t.tts, 2),
            "total": round(t.response, 2),
        },
        **_audio_payload(reply.speech),
    }


def _read_upload(audio: UploadFile) -> bytes:
    content_type = (audio.content_type or "").split(";")[0].strip().lower()
    if content_type not in AUDIO_TYPES:
        raise HTTPException(415, "Unsupported recording type.")
    data = audio.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Recording too large (max {MAX_UPLOAD_BYTES // 1_000_000} MB).")
    return data


def _check_text(text: str) -> str:
    text = text.strip()
    if not text:
        raise HTTPException(400, "Empty message.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(413, f"Message too long (max {MAX_TEXT_CHARS} characters).")
    return text


def _decode(data: bytes) -> np.ndarray | None:
    """Whatever the browser recorded (webm/ogg/wav) as 16 kHz mono float32.

    Recordings longer than MAX_AUDIO_SECONDS are cut, never sent whole to STT.
    """
    try:
        return audio_io.decode_audio_bytes(data, RATE, max_seconds=MAX_AUDIO_SECONDS)
    except ValueError as exc:  # corrupt, empty or not audio
        logger.warning("Could not decode the recording: %s", exc)
        return None


def _ndjson(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


def _timings_json(t: TurnTimings) -> dict[str, float]:
    return {
        "stt": round(t.stt, 2),
        "llm_first": round(t.llm_first, 2),
        "llm": round(t.llm, 2),
        "tts_first": round(t.tts_first, 2),
        "first_audio": round(t.first_audio, 2),
        "total": round(t.total, 2),
    }


def _reply_events(
    conversation: Conversation,
    text: str,
    heard: Heard | None,
    cancel: threading.Event,
    session: ChatSession | None = None,
) -> Iterator[bytes]:
    """Conversation.respond_stream as NDJSON lines (audio base64-encoded)."""
    for event in conversation.respond_stream(text, heard, cancel, session=session):
        if event["type"] == "audio":
            data = base64.b64encode(event.pop("data")).decode("ascii")
            yield _ndjson({**event, "audio": data})
        elif event["type"] == "done":
            yield _ndjson({**event, "timings": _timings_json(event["timings"])})
        else:
            yield _ndjson(event)


def _not_heard_events(
    conversation: Conversation, heard: Heard, session: ChatSession | None = None
) -> Iterator[bytes]:
    session = session or conversation.session()
    line = session.lines["not_heard"]
    audio, mime = None, "audio/wav"
    encoded = conversation.fixed_line_audio(session.gender, "not_heard")
    if encoded is not None:
        audio, mime = base64.b64encode(encoded.data).decode("ascii"), encoded.mime
    yield _ndjson({"type": "not_heard", "text": line, "audio": audio, "mime": mime})
    timings = TurnTimings(speech_seconds=heard.speech_seconds, stt=heard.stt_seconds)
    timings.total = heard.stt_seconds
    done = {"type": "done", "reply": "", "goodbye": False, "voice_backend": ""}
    yield _ndjson({**done, "timings": _timings_json(timings)})


def _public_message(exc: Exception) -> str:
    """Our own errors are written for users; anything else stays in the server log
    (locally it is shown too, to make debugging easy)."""
    if isinstance(exc, (STTQuotaError, RateLimitError)):
        return QUOTA_MESSAGE
    if isinstance(exc, VoiceAgentError) or not WEB.cloud:
        return str(exc) or exc.__class__.__name__
    return GENERIC_ERROR


def _guard(events: Iterator[bytes]) -> Iterator[bytes]:
    """Turn unexpected failures into an error event instead of a broken stream."""
    try:
        yield from events
    except VoiceAgentError as exc:  # expected (quota, offline...): one line, no traceback
        logger.warning("Turn failed: %s", exc)
        yield _ndjson({"type": "error", "message": _public_message(exc)})
    except Exception as exc:
        logger.exception("Streaming turn failed")
        yield _ndjson({"type": "error", "message": _public_message(exc)})


def _stream(events: Iterator[bytes], cancel: threading.Event) -> StreamingResponse:
    """NDJSON response. If the browser hangs up (barge-in, New chat), `cancel` is set,
    which stops Gemini and the voice for that turn instead of finishing it unheard."""
    iterator = _guard(events)
    done = object()

    async def body() -> AsyncIterator[bytes]:
        try:
            while (chunk := await run_in_threadpool(next, iterator, done)) is not done:
                yield chunk
        finally:
            cancel.set()

    return StreamingResponse(
        body(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class TextIn(BaseModel):
    text: str = Field(max_length=4 * MAX_TEXT_CHARS)
    speech_seconds: float = 0.0
    stt_seconds: float = 0.0


class PersonaIn(BaseModel):
    gender: str = Field(max_length=16)


# --- endpoints ---------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Liveness for hosting platforms: up even while models are still loading."""
    return {"ok": True, "ready": _state["conversation"] is not None}


def _status(client: Client) -> dict[str, Any]:
    conversation = _state["conversation"]
    session = conversation.peek_session(client.session_key) if conversation else None
    gender = session.gender if session else (conversation.gender if conversation else None)
    voice = VOICES.get(gender) if session else None
    return {
        "ready": conversation is not None,
        "loading": _state["loading"],
        "error": _state["error"],
        "gender": gender,
        "voice": voice or (conversation.settings.tts_voice if conversation else None),
        "stt_backend": conversation.settings.stt_backend if conversation else None,
        "stt_model": conversation.settings.stt_model if conversation else None,
        "llm_model": conversation.settings.gemini_model if conversation else None,
    }


@app.get("/api/status")
def status(client: Client = Depends(client_info)) -> dict[str, Any]:
    return _status(client)


@app.post("/api/reset")
def reset(client: Client = Depends(client_info)) -> dict[str, bool]:
    """New chat: forget this visitor's conversation (no greeting is spoken)."""
    conversation = _conversation()
    conversation.reset(_session(conversation, client))
    return {"ok": True}


@app.post("/api/turn")
def turn(audio: UploadFile = File(...), client: Client = Depends(limited)) -> StreamingResponse:
    """One spoken turn, streamed: what was heard, then the reply sentence by sentence."""
    conversation = _conversation()
    data = _read_upload(audio)  # read now: the upload is closed once streaming starts
    session = _session(conversation, client)
    cancel = threading.Event()

    def events() -> Iterator[bytes]:
        samples = _decode(data)
        heard = conversation.hear(samples) if samples is not None else Heard("", 0.0, 0.0)
        yield _ndjson(
            {
                "type": "heard",
                "text": heard.text,
                "stt_seconds": round(heard.stt_seconds, 2),
                "speech_seconds": round(heard.speech_seconds, 2),
            }
        )
        if not heard.text:
            yield from _not_heard_events(conversation, heard, session)
            return
        if cancel.is_set():
            return
        yield from _reply_events(conversation, heard.text, heard, cancel, session)

    return _stream(events(), cancel)


@app.post("/api/chat")
def chat(body: TextIn, client: Client = Depends(limited)) -> StreamingResponse:
    """A typed message, streamed like /api/turn (no "heard" event)."""
    conversation = _conversation()
    text = _check_text(body.text)
    session = _session(conversation, client)
    cancel = threading.Event()
    return _stream(_reply_events(conversation, text, None, cancel, session), cancel)


@app.post("/api/listen")
def listen(audio: UploadFile = File(...), client: Client = Depends(limited)) -> dict[str, Any]:
    """Decode whatever the browser recorded (webm/ogg/wav) and transcribe it."""
    conversation = _conversation()
    session = _session(conversation, client)
    samples = _decode(_read_upload(audio))
    if samples is None:
        return {"text": "", "reason": "decode_failed"}
    try:
        heard: Heard = conversation.hear(samples)
    except STTQuotaError as exc:
        raise HTTPException(429, QUOTA_MESSAGE) from exc
    except STTError as exc:
        raise HTTPException(503, _public_message(exc)) from exc
    return {
        "text": heard.text,
        "speech_seconds": round(heard.speech_seconds, 2),
        "stt_seconds": round(heard.stt_seconds, 2),
        "reason": "" if heard.text else "no_speech",
        "not_heard": session.lines["not_heard"],
    }


@app.post("/api/reply")
def reply(body: TextIn, client: Client = Depends(limited)) -> dict[str, Any]:
    conversation = _conversation()
    text = _check_text(body.text)
    heard = Heard(text, body.speech_seconds, body.stt_seconds)
    return _reply_payload(conversation.respond(text, heard, _session(conversation, client)))


@app.post("/api/persona")
def persona(body: PersonaIn, client: Client = Depends(client_info)) -> dict[str, Any]:
    """Switch this visitor's voice and persona (Uzma / Asad); starts a new chat."""
    conversation = _conversation()
    if body.gender not in VOICES:
        raise HTTPException(400, f"gender must be one of {list(VOICES)}")
    conversation.set_gender(body.gender, _session(conversation, client))
    return _status(client)


def main() -> None:
    import uvicorn

    settings = Settings.from_env()
    setup_logging(settings.log_level)
    if WEB.cloud:
        print(f"\n  Urdu Voice Agent (cloud mode) listening on {HOST}:{PORT}\n")
    else:
        print(f"\n  Urdu Voice Agent is starting: http://127.0.0.1:{PORT}")
        print("  Keep this window open while you use it. Close it to stop.\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
