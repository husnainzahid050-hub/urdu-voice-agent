"""Unit tests that need no network, mic or downloaded models."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from google.genai import errors as genai_errors

from src import llm
from src.audio_io import resample
from src.config import Settings
from src.errors import ConfigError, LLMError, RateLimitError, TTSError
from src.pipeline import is_goodbye
from src.stt import URDU_PROMPT, _looks_hallucinated
from src.tts import BaseTTS, FallbackTTS, Speech


def _api_error(code: int) -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"message": f"boom {code}", "status": "X"}})


class _FakeChat:
    def __init__(self, failures: list[int]) -> None:
        self.failures = failures
        self.calls = 0

    def send_message(self, _text: str) -> SimpleNamespace:
        self.calls += 1
        if self.failures:
            raise _api_error(self.failures.pop(0))
        return SimpleNamespace(text="جی ہاں۔", candidates=[])


def _chat_with(fake: _FakeChat) -> llm.GeminiChat:
    chat = llm.GeminiChat.__new__(llm.GeminiChat)
    chat.model = "test-model"
    chat._chat = fake
    return chat


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)


def test_llm_retries_when_busy_then_succeeds() -> None:
    fake = _FakeChat([503, 503])
    assert _chat_with(fake).reply("سلام").text == "جی ہاں۔"
    assert fake.calls == 3


def test_llm_gives_up_with_readable_rate_limit_error() -> None:
    fake = _FakeChat([429, 429, 429])
    with pytest.raises(RateLimitError, match="429"):
        _chat_with(fake).reply("سلام")
    assert fake.calls == 3


def test_llm_does_not_retry_client_errors() -> None:
    fake = _FakeChat([404])
    with pytest.raises(LLMError, match="not available"):
        _chat_with(fake).reply("سلام")
    assert fake.calls == 1


def test_persona_gender_in_prompt() -> None:
    assert "سکتی" in llm.build_system_prompt("female")
    assert "سکتا" in llm.build_system_prompt("male")


def test_goodbye_detection() -> None:
    assert is_goodbye("ٹھیک ہے، اللہ حافظ")
    assert not is_goodbye("میرا آرڈر کب آئے گا؟")


def test_hallucination_guard() -> None:
    assert _looks_hallucinated("سبسکرائب کریں", 0.1, -0.2)
    assert _looks_hallucinated("کچھ", 0.9, -1.5)
    assert not _looks_hallucinated("میرا آرڈر کب آئے گا", 0.1, -0.3)


def test_prompt_echo_is_dropped() -> None:
    assert _looks_hallucinated(f"{URDU_PROMPT[:-1]}", 0.1, -0.2)
    assert _looks_hallucinated("اردو میں بات کرتے ہیں", 0.1, -0.2)
    assert not _looks_hallucinated("جی میرا آرڈر ٹھیک ہے؟", 0.1, -0.2)


class _Broken(BaseTTS):
    name = "broken"

    def synthesize(self, text: str) -> Speech:
        raise TTSError("offline")


class _Silent(BaseTTS):
    name = "silent"

    def synthesize(self, text: str) -> Speech:
        return Speech(np.zeros(10, dtype=np.float32), 16000, 0.0, self.name)


def test_tts_falls_back_when_online_voice_fails() -> None:
    tts = FallbackTTS(_Broken(), _Silent)
    assert tts.synthesize("سلام").backend == "silent"


def test_resample_length() -> None:
    audio = np.ones(24000, dtype=np.float32)
    assert len(resample(audio, 24000, 16000)) == 16000


def test_invalid_gender_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_GENDER", "robot")
    with pytest.raises(ConfigError, match="AGENT_GENDER"):
        Settings.from_env()


def test_missing_key_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(ConfigError, match="aistudio"):
        Settings.from_env().require_gemini()
