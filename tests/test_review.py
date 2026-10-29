"""Regression tests for issues found in the end-to-end review."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import llm, server
from src.errors import STTError, TTSError
from src.pipeline import Conversation
from src.stt import BaseSTT, FallbackSTT, Transcript
from src.tts import EdgeTTS, EncodedSpeech

# --- edge-tts: a stalled request is raced, not waited on --------------------------------


def _edge(plan: list[str]) -> tuple[EdgeTTS, dict[str, int]]:
    """EdgeTTS whose requests follow `plan`: "stall", "fail", "empty" or "ok"."""
    tts = EdgeTTS(hedge_after=0.1, timeout=2.0)
    seen = {"calls": 0, "cancelled": 0}

    async def fake_fetch(text: str) -> bytes:
        kind = plan[min(seen["calls"], len(plan) - 1)]
        seen["calls"] += 1
        if kind == "stall":
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                seen["cancelled"] += 1
                raise
        if kind == "fail":
            raise ConnectionError("dropped")
        return b"" if kind == "empty" else b"mp3"

    tts._fetch_mp3 = fake_fetch  # type: ignore[method-assign]
    return tts, seen


def test_stalled_voice_request_is_raced_and_cancelled() -> None:
    tts, seen = _edge(["stall", "ok"])
    started = time.perf_counter()
    assert tts._fetch_mp3_sync("سلام") == b"mp3"
    assert time.perf_counter() - started < 1.5  # did not wait for the 2 s timeout
    time.sleep(0.1)  # let the loop process the cancellation
    assert seen == {"calls": 2, "cancelled": 1}


def test_failed_voice_request_is_retried_at_once() -> None:
    tts, seen = _edge(["fail", "ok"])
    assert tts._fetch_mp3_sync("سلام") == b"mp3"
    assert seen["calls"] == 2


def test_voice_gives_up_after_two_attempts() -> None:
    tts, seen = _edge(["empty", "fail"])
    with pytest.raises(TTSError):
        tts._fetch_mp3_sync("سلام")
    assert seen["calls"] == 2


# --- Gemini: an exhausted model is skipped by every new conversation --------------------


def test_new_chats_skip_a_model_another_chat_found_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "_unavailable", {})
    client = SimpleNamespace(chats=SimpleNamespace(create=lambda **kw: SimpleNamespace(**kw)))
    fallbacks = ["lite-b", "lite-c"]
    assert llm.GeminiChat("key", "main", fallback_models=fallbacks, client=client).model == "main"
    llm._mark_unavailable("main")
    chat = llm.GeminiChat("key", "main", fallback_models=fallbacks, client=client)
    assert chat.model == "lite-b" and chat._fallback_models == ["lite-c"]
    llm._mark_unavailable("lite-b")
    llm._mark_unavailable("lite-c")  # all exhausted: keep the last one to try
    assert llm.GeminiChat("key", "main", fallback_models=fallbacks, client=client).model == "lite-c"


def test_unavailable_models_come_back_after_the_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_unavailable", {})
    llm._mark_unavailable("main", seconds=-1)
    assert not llm._is_unavailable("main")


def test_prompt_does_not_assume_the_users_gender() -> None:
    prompt = llm.build_system_prompt("female")
    assert "do not know the user's gender" in prompt


# --- STT: fallback preloaded, and listening time is what the user waited ----------------


class _Slow(BaseSTT):
    name = "slow"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        time.sleep(0.2)
        return Transcript("سلام", "ur", 1.0, 0.0)


def test_fallback_can_be_preloaded_during_warm_up() -> None:
    built: list[str] = []

    class Primary(BaseSTT):
        def warm_up(self) -> None:
            pass

        def transcribe(self, audio, sample_rate=16000):  # noqa: ANN001, ANN201
            raise STTError("quota")

    def make() -> BaseSTT:
        built.append("whisper")
        return _Slow()

    FallbackSTT(Primary(), make).warm_up()
    assert built == []  # default: lazy
    FallbackSTT(Primary(), make, preload=True).warm_up()
    assert built == ["whisper"]


def test_heard_seconds_include_everything_the_user_waited_for() -> None:
    conv = Conversation.__new__(Conversation)
    conv._lock = conv._stt_lock = threading.Lock()
    conv.vad = None
    conv.stt = _Slow()
    heard = conv.hear(np.zeros(16000, np.float32))
    assert heard.text == "سلام" and heard.stt_seconds >= 0.2


# --- fixed lines are voiced once ---------------------------------------------------------


def test_not_heard_line_audio_is_cached() -> None:
    calls: list[str] = []

    class Voice:
        def synthesize_encoded(self, text: str) -> EncodedSpeech:
            calls.append(text)
            return EncodedSpeech(b"mp3", "audio/mpeg", "edge", 0.0)

    conv = Conversation.__new__(Conversation)
    conv.tts = Voice()
    first = conv.fixed_line_audio("female", "not_heard")
    second = conv.fixed_line_audio("female", "not_heard")
    assert first is second and len(calls) == 1


# --- endpoints removed: no greeting, no free-form text-to-speech ------------------------


def test_greeting_and_speak_endpoints_are_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_load_models", lambda: None)
    monkeypatch.setattr(server.webbrowser, "open", lambda *a, **k: None)
    with TestClient(server.app) as c:
        for path in ("/api/greeting", "/api/speak"):
            assert c.post(path, json={"text": "سلام"}).status_code in (404, 405)
