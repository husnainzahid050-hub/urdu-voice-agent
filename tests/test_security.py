"""Public-demo protection: sessions, rate limits, headers, error bodies, cloud imports."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src import server
from src.config import ROOT_DIR
from src.limits import MSG_GLOBAL, MSG_PER_DAY, MSG_PER_MINUTE, RateLimiter, client_ip
from src.pipeline import ChatSession, Conversation, Heard, SessionStore
from src.tts import EncodedSpeech

# --- sessions ----------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _HistoryLLM:
    """Remembers what it was told, like a Gemini chat history."""

    def __init__(self) -> None:
        self.history: list[str] = []

    def reply_stream(self, text: str) -> Iterator[str]:
        self.history.append(text)
        yield f"آپ نے {len(self.history)} بار پوچھا۔"

    def reset(self) -> None:
        self.history.clear()


def _store(clock: _Clock, ttl: float = 60, cap: int = 3) -> SessionStore:
    return SessionStore(lambda: ChatSession(_HistoryLLM(), "female"), ttl, cap, clock)


def test_sessions_are_isolated() -> None:
    store = _store(_Clock())
    a, b = store.get("a"), store.get("b")
    assert a is not b and a.llm is not b.llm
    assert store.get("a") is a


def test_sessions_expire_after_idle_ttl() -> None:
    clock = _Clock()
    store = _store(clock, ttl=60)
    first = store.get("a")
    clock.now += 61
    assert store.peek("a") is None
    assert store.get("a") is not first


def test_session_cap_evicts_least_recently_used() -> None:
    store = _store(_Clock(), cap=2)
    a = store.get("a")
    store.get("b")
    store.get("a")  # a is now the most recent
    store.get("c")  # evicts b
    assert len(store) == 2 and store.peek("b") is None and store.peek("a") is a


# --- rate limits ---------------------------------------------------------------------------


def test_per_minute_bucket_refills() -> None:
    clock = _Clock()
    limiter = RateLimiter(per_minute=2, per_day=0, global_per_day=0, clock=clock)
    assert limiter.check("ip") is None and limiter.check("ip") is None
    assert limiter.check("ip") == MSG_PER_MINUTE
    assert limiter.check("other") is None  # per client
    clock.now += 30  # one token back (2 per 60 s)
    assert limiter.check("ip") is None


def test_per_day_and_global_caps_reset_next_day() -> None:
    clock = _Clock()
    limiter = RateLimiter(per_minute=0, per_day=2, global_per_day=3, clock=clock)
    assert [limiter.check("a"), limiter.check("a")] == [None, None]
    assert limiter.check("a") == MSG_PER_DAY
    assert limiter.check("b") is None
    assert limiter.check("c") == MSG_GLOBAL
    clock.now += 86_400
    assert limiter.check("a") is None and limiter.check("c") is None


def test_forwarded_for_only_trusted_behind_a_proxy() -> None:
    assert client_ip("10.0.0.1", "1.2.3.4", trust_proxy=0) == "10.0.0.1"
    # The proxy appends the real client last; earlier entries can be forged.
    assert client_ip("10.0.0.1", "6.6.6.6, 1.2.3.4", trust_proxy=1) == "1.2.3.4"
    assert client_ip("10.0.0.1", "not-an-ip", trust_proxy=1) == "10.0.0.1"


def test_client_ip_behind_vercel_and_render() -> None:
    # Vercel sets the visitor's IP; Render appends Vercel's edge address.
    assert client_ip("10.0.0.1", "6.6.6.6, 1.2.3.4, 76.76.21.9", trust_proxy=2) == "1.2.3.4"
    # Fewer hops than configured: take the leftmost rather than failing.
    assert client_ip("10.0.0.1", "1.2.3.4", trust_proxy=2) == "1.2.3.4"


# --- endpoints -----------------------------------------------------------------------------


class _TTS:
    def synthesize_encoded(self, text: str) -> EncodedSpeech:
        return EncodedSpeech(b"mp3", "audio/mpeg", "edge", 0.0)


def _multi_session_conversation() -> Conversation:
    conv = Conversation.__new__(Conversation)
    conv.settings = SimpleNamespace(
        agent_gender="female", tts_voice="v", stt_backend="gemini", stt_model="m", gemini_model="g"
    )
    conv._lock = threading.Lock()
    conv.llm = _HistoryLLM()
    conv.tts = _TTS()
    conv._latency_log = False
    conv._sessions = _store(_Clock(), ttl=600, cap=50)
    conv.hear = lambda samples: Heard("سلام", 1.0, 0.5)  # type: ignore[method-assign]
    return conv


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, Conversation]]:
    conv = _multi_session_conversation()
    monkeypatch.setattr(server, "_load_models", lambda: None)
    monkeypatch.setattr(server.webbrowser, "open", lambda *a, **k: None)
    monkeypatch.setitem(server._state, "conversation", conv)
    monkeypatch.setattr(server, "_decode", lambda data: [0.0] * 16000)
    with TestClient(server.app) as c:
        yield c, conv


def _reply(response) -> str:  # noqa: ANN001
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    return events[-1]["reply"]


def test_each_tab_has_its_own_conversation(app_client) -> None:  # noqa: ANN001
    c, _ = app_client
    tab1, tab2 = {"X-Session-Id": "tab-one-123"}, {"X-Session-Id": "tab-two-456"}
    assert _reply(c.post("/api/chat", json={"text": "ا"}, headers=tab1)) == "آپ نے 1 بار پوچھا۔"
    assert _reply(c.post("/api/chat", json={"text": "ب"}, headers=tab1)) == "آپ نے 2 بار پوچھا۔"
    assert _reply(c.post("/api/chat", json={"text": "ج"}, headers=tab2)) == "آپ نے 1 بار پوچھا۔"
    c.post("/api/reset", headers=tab1)  # resets only tab 1
    assert _reply(c.post("/api/chat", json={"text": "د"}, headers=tab1)) == "آپ نے 1 بار پوچھا۔"
    assert _reply(c.post("/api/chat", json={"text": "ہ"}, headers=tab2)) == "آپ نے 2 بار پوچھا۔"


def test_persona_is_per_session(app_client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    c, _ = app_client
    from src import pipeline

    monkeypatch.setattr(
        pipeline.GeminiChat, "from_settings", lambda *a, **k: _HistoryLLM(), raising=True
    )
    tab1, tab2 = {"X-Session-Id": "tab-one-123"}, {"X-Session-Id": "tab-two-456"}
    c.post("/api/chat", json={"text": "ا"}, headers=tab2)  # tab 2 exists, default persona
    assert c.post("/api/persona", json={"gender": "male"}, headers=tab1).json()["gender"] == "male"
    assert c.get("/api/status", headers=tab1).json()["gender"] == "male"
    assert c.get("/api/status", headers=tab2).json()["gender"] == "female"


def test_rate_limit_returns_friendly_429(app_client, monkeypatch) -> None:  # noqa: ANN001
    c, _ = app_client
    monkeypatch.setattr(server, "limiter", RateLimiter(per_minute=1, per_day=0, global_per_day=0))
    assert c.post("/api/chat", json={"text": "سلام"}).status_code == 200
    r = c.post("/api/chat", json={"text": "سلام"})
    assert r.status_code == 429 and r.json() == {"detail": MSG_PER_MINUTE}
    assert c.get("/api/status").status_code == 200  # status is never limited


def test_security_headers(app_client) -> None:  # noqa: ANN001
    c, _ = app_client
    h = c.get("/").headers
    assert h["x-content-type-options"] == "nosniff"
    assert h["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in h["content-security-policy"]
    assert "wasm-unsafe-eval" in h["content-security-policy"]
    assert "microphone=(self)" in h["permissions-policy"]


def test_upload_type_and_size_checks(app_client) -> None:  # noqa: ANN001
    c, _ = app_client
    r = c.post("/api/turn", files={"audio": ("x.exe", b"MZ", "application/x-msdownload")})
    assert r.status_code == 415
    ok = c.post("/api/turn", files={"audio": ("t.wav", b"RIFF", "audio/wav")})
    assert ok.status_code == 200
    huge = c.post(
        "/api/chat",
        content=b"{" + b" " * (server.MAX_BODY_BYTES + 1) + b"}",
        headers={"Content-Type": "application/json"},
    )
    assert huge.status_code == 413


def test_cloud_errors_hide_internal_details(app_client, monkeypatch) -> None:  # noqa: ANN001
    c, conv = app_client
    monkeypatch.setattr(server, "WEB", dataclasses.replace(server.WEB, app_mode="cloud"))

    def boom(samples):  # noqa: ANN001, ANN202
        raise RuntimeError(r"C:\secret\path broke")

    conv.hear = boom  # type: ignore[method-assign]
    r = c.post("/api/turn", files={"audio": ("t.webm", b"x", "audio/webm")})
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    assert events == [{"type": "error", "message": server.GENERIC_ERROR}]


def test_cloud_build_does_not_need_torch_or_audio_devices() -> None:
    """APP_MODE=cloud must import without torch, Whisper, Silero or PortAudio."""
    code = (
        "import sys, builtins\n"
        "real = builtins.__import__\n"
        "def guard(name, *a, **k):\n"
        "    if name.split('.')[0] in {'torch','transformers','faster_whisper','silero_vad',"
        "'sounddevice','ctranslate2'}:\n"
        "        raise ImportError('blocked in cloud: ' + name)\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = guard\n"
        "import src.server\n"
        "from src.config import Settings\n"
        "s = Settings.from_env()\n"
        "assert s.cloud and s.stt_backend == 'gemini' and not s.server_vad\n"
        "assert s.tts_fallback == 'none' and not s.latency_log and not s.log_transcripts\n"
        "print('ok')\n"
    )
    mode_vars = ("STT_BACKEND", "STT_FALLBACK", "SERVER_VAD", "TTS_FALLBACK", "LATENCY_LOG")
    env = {k: v for k, v in os.environ.items() if k not in mode_vars + ("LOG_TRANSCRIPTS",)}
    env.update(APP_MODE="cloud", ENV_FILE=str(ROOT_DIR / "tests" / "no-such.env"))
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT_DIR, env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")


def test_quota_error_becomes_friendly_message(app_client) -> None:  # noqa: ANN001
    from src.errors import STTQuotaError

    c, conv = app_client

    def quota(samples):  # noqa: ANN001, ANN202
        raise STTQuotaError("429 RESOURCE_EXHAUSTED for gemini-3.5-transcribe")

    conv.hear = quota  # type: ignore[method-assign]
    r = c.post("/api/turn", files={"audio": ("t.webm", b"x", "audio/webm")})
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    assert events == [{"type": "error", "message": server.QUOTA_MESSAGE}]
    r = c.post("/api/listen", files={"audio": ("t.webm", b"x", "audio/webm")})
    assert r.status_code == 429 and r.json()["detail"] == server.QUOTA_MESSAGE


def test_stt_without_usable_fallback_reports_the_quota_error() -> None:
    from src.errors import ModelLoadError, STTError, STTQuotaError
    from src.stt import BaseSTT, FallbackSTT

    class Exhausted(BaseSTT):
        def transcribe(self, audio, sample_rate=16000):  # noqa: ANN001, ANN201
            raise STTQuotaError("daily quota used up")

    def no_whisper() -> BaseSTT:
        raise ModelLoadError("Local Whisper is not installed")

    stt = FallbackSTT(Exhausted(), no_whisper)
    with pytest.raises(STTQuotaError):
        stt.transcribe([0.0] * 16000)
    with pytest.raises(STTError):  # still paused: fails fast without calling Gemini again
        stt.transcribe([0.0] * 16000)
