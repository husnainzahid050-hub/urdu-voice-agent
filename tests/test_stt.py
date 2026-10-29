"""STT tests that need no network, mic or downloaded models."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.benchmark_stt import edit_distance, error_rates, normalize_for_scoring
from src.errors import STTError, STTQuotaError
from src.stt import BaseSTT, FallbackSTT, GeminiSTT, Transcript


class _Broken(BaseSTT):
    name = "broken"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        raise STTError("offline")


class _Echo(BaseSTT):
    name = "echo"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        return Transcript("جی ہاں", "ur", 1.0, 0.1, "echo")


def test_fallback_is_built_lazily_and_used_on_failure() -> None:
    built = []

    def make() -> BaseSTT:
        built.append(1)
        return _Echo()

    stt = FallbackSTT(_Echo(), make)
    assert stt.transcribe(np.zeros(16000, np.float32)).backend == "echo"
    assert not built  # Whisper is never loaded while Gemini works

    stt = FallbackSTT(_Broken(), make)
    assert stt.transcribe(np.zeros(16000, np.float32)).text == "جی ہاں"
    stt.transcribe(np.zeros(16000, np.float32))
    assert built == [1]  # built once, reused


def _gemini_with(parts: list[SimpleNamespace]) -> GeminiSTT:
    stt = GeminiSTT.__new__(GeminiSTT)
    stt.model, stt.language = "gemini-3.5-transcribe", "ur"
    response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])
    stt._request = lambda _flac: response  # type: ignore[method-assign]
    return stt


def test_gemini_reads_audio_transcription_part() -> None:
    part = SimpleNamespace(audio_transcription=SimpleNamespace(text="ميرا آرڈر"), text=None)
    result = _gemini_with([part]).transcribe(np.zeros(16000, np.float32))
    assert result.text == "میرا آرڈر"  # Arabic yeh normalised to Urdu
    assert result.backend == "gemini:gemini-3.5-transcribe"


def test_gemini_devanagari_answer_triggers_fallback() -> None:
    part = SimpleNamespace(audio_transcription=SimpleNamespace(text="हेलो हेलो"), text=None)
    with pytest.raises(STTError, match="Devanagari"):
        _gemini_with([part]).transcribe(np.zeros(16000, np.float32))


def test_gemini_empty_response_is_no_speech() -> None:
    assert _gemini_with([]).transcribe(np.zeros(16000, np.float32)).text == ""


def test_scoring_normalisation() -> None:
    assert normalize_for_scoring("آرڈر نمبر 12، ٹھیک ہے؟") == "آرڈر نمبر بارہ ٹھیک ہے"
    assert edit_distance("abc", "abd") == 1
    assert edit_distance(["a", "b"], ["a", "b"]) == 0
    char_err, chars, word_err, words = error_rates("میرا آرڈر", "میرا آڈر")
    assert (char_err, word_err, words) == (1, 1, 2)
    assert chars == 8


def test_implausibly_long_transcript_is_dropped() -> None:
    long_text = "میرا آرڈر " * 40  # ~320 letters for one second of audio
    part = SimpleNamespace(audio_transcription=SimpleNamespace(text=long_text), text=None)
    assert _gemini_with([part]).transcribe(np.zeros(16000, np.float32)).text == ""
    ok = SimpleNamespace(audio_transcription=SimpleNamespace(text="میرا آرڈر"), text=None)
    assert _gemini_with([ok]).transcribe(np.zeros(16000, np.float32)).text == "میرا آرڈر"


class _Quota(BaseSTT):
    name = "quota"

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        self.calls += 1
        raise STTQuotaError("429")


def test_quota_error_pauses_primary_then_retries_later() -> None:
    now = [0.0]
    primary = _Quota()
    stt = FallbackSTT(primary, _Echo, clock=lambda: now[0])
    audio = np.zeros(16000, np.float32)
    assert stt.transcribe(audio).backend == "echo"
    assert stt.transcribe(audio).backend == "echo"
    assert primary.calls == 1  # second turn went straight to Whisper
    now[0] += FallbackSTT.QUOTA_PAUSE_SECONDS + 1
    stt.transcribe(audio)
    assert primary.calls == 2  # tried Gemini again after the pause


def test_warm_up_failure_loads_fallback_early() -> None:
    built = []

    class _Offline(_Echo):
        def warm_up(self) -> None:
            raise STTError("offline")

    def make() -> BaseSTT:
        built.append(1)
        return _Echo()

    FallbackSTT(_Offline(), make).warm_up()
    assert built == [1]


def test_quota_details_parse_minute_and_day_limits() -> None:
    from src.stt import _quota_info

    def err(quota_id: str, delay: str) -> SimpleNamespace:
        return SimpleNamespace(
            details={
                "error": {
                    "details": [
                        {"violations": [{"quotaId": quota_id, "quotaValue": "20"}]},
                        {"retryDelay": delay},
                    ]
                }
            }
        )

    text, wait = _quota_info(err("GenerateRequestsPerMinutePerProject", "37s"))
    assert wait == 37.0 and "= 20" in text
    _, wait = _quota_info(err("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "20s"))
    assert wait == 3600.0
    assert _quota_info(Exception("no details")) == ("", None)
