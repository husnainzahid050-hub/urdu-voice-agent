"""Browser front end: a local web app with a mic button, chat bubbles and spoken replies.

Run:  python -m src.server        (or double-click start.bat)
Then open http://127.0.0.1:7860 (it opens by itself). Everything runs on this PC;
the browser only talks to this local server.

API (all JSON unless noted):
    GET  /api/status             models loaded? which voice?
    POST /api/greeting           new conversation; returns the greeting + audio
    POST /api/listen  (form)     audio file from the browser -> what the user said
    POST /api/reply              {"text": ...} -> agent's reply + audio
    POST /api/persona            {"gender": "female"|"male"} -> switch voice
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import webbrowser
from contextlib import asynccontextmanager
from typing import Any

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import ROOT_DIR, Settings, setup_logging
from src.errors import VoiceAgentError
from src.pipeline import RATE, VOICES, Conversation, Heard, Reply
from src.tts import Speech

logger = logging.getLogger(__name__)

HOST, PORT = "127.0.0.1", 7860
WEB_DIR = ROOT_DIR / "web"

_state: dict[str, Any] = {"conversation": None, "error": None, "loading": "Starting..."}


def _load_models() -> None:
    """Load all models in the background so the page can show progress."""
    settings = Settings.from_env()

    def status(message: str) -> None:
        _state["loading"] = message
        logger.info(message)

    try:
        _state["conversation"] = Conversation(settings, status)
        _state["loading"] = None
        logger.info("Ready on http://%s:%d", HOST, PORT)
    except VoiceAgentError as exc:
        _state["error"] = str(exc)
        logger.error("%s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):  # noqa: ANN201
    threading.Thread(target=_load_models, daemon=True).start()
    threading.Timer(1.0, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    yield


app = FastAPI(title="Urdu Voice Agent", lifespan=lifespan)


def _conversation() -> Conversation:
    conversation = _state["conversation"]
    if conversation is None:
        raise HTTPException(503, _state["error"] or _state["loading"] or "Loading...")
    return conversation


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


class TextIn(BaseModel):
    text: str
    speech_seconds: float = 0.0
    stt_seconds: float = 0.0


class PersonaIn(BaseModel):
    gender: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
def status() -> dict[str, Any]:
    conversation = _state["conversation"]
    return {
        "ready": conversation is not None,
        "loading": _state["loading"],
        "error": _state["error"],
        "gender": conversation.gender if conversation else None,
        "voice": conversation.settings.tts_voice if conversation else None,
        "stt_model": conversation.settings.stt_model if conversation else None,
        "llm_model": conversation.settings.gemini_model if conversation else None,
    }


@app.post("/api/greeting")
def greeting() -> dict[str, Any]:
    conversation = _conversation()
    conversation.reset()
    text = conversation.lines["greeting"]
    return {"reply": text, **_audio_payload(conversation.speak(text))}


@app.post("/api/listen")
def listen(audio: UploadFile = File(...)) -> dict[str, Any]:
    """Decode whatever the browser recorded (webm/ogg/wav) and transcribe it."""
    conversation = _conversation()
    from faster_whisper.audio import decode_audio  # PyAV ships with faster-whisper

    try:
        samples = decode_audio(io.BytesIO(audio.file.read()), sampling_rate=RATE)
    except Exception as exc:  # corrupt or empty recording
        logger.warning("Could not decode the recording: %s", exc)
        return {"text": "", "reason": "decode_failed"}
    heard: Heard = conversation.hear(samples)
    return {
        "text": heard.text,
        "speech_seconds": round(heard.speech_seconds, 2),
        "stt_seconds": round(heard.stt_seconds, 2),
        "reason": "" if heard.text else "no_speech",
        "not_heard": conversation.lines["not_heard"],
    }


@app.post("/api/reply")
def reply(body: TextIn) -> dict[str, Any]:
    conversation = _conversation()
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Empty message.")
    heard = Heard(text, body.speech_seconds, body.stt_seconds)
    return _reply_payload(conversation.respond(text, heard))


@app.post("/api/speak")
def speak(body: TextIn) -> dict[str, Any]:
    """Speak a fixed line (e.g. "I didn't hear you") without asking Gemini."""
    return _audio_payload(_conversation().speak(body.text))


@app.post("/api/persona")
def persona(body: PersonaIn) -> dict[str, Any]:
    conversation = _conversation()
    if body.gender not in VOICES:
        raise HTTPException(400, f"gender must be one of {list(VOICES)}")
    conversation.set_gender(body.gender)
    return status()


def main() -> None:
    import uvicorn

    settings = Settings.from_env()
    setup_logging(settings.log_level)
    print(f"\n  Urdu Voice Agent is starting: http://{HOST}:{PORT}")
    print("  Keep this window open while you use it. Close it to stop.\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
