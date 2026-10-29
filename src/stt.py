"""Speech-to-text. `BaseSTT` is the common interface.

Backends (STT_BACKEND):
- "gemini": Gemini audio transcription (default model gemini-3.5-transcribe with the
  Urdu language code). Online, ~1.5-3 s per turn and the most accurate on Urdu.
- "whisper": local faster-whisper (CTranslate2). Offline, but slow on a laptop CPU.
- `FallbackSTT` uses Gemini and switches to local Whisper when Gemini is unreachable,
  so the agent keeps working offline (Whisper is only loaded when first needed).

`STT_MODEL` (Whisper) can be a size ("small", "medium", "large-v3"), a Hugging Face repo
with a CTranslate2 model, or a local folder. Fine-tuned Transformers checkpoints must be
converted to CTranslate2 first (`ct2-transformers-converter`).
"""

from __future__ import annotations

import importlib.util
import io
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from src.config import MODELS_DIR, Settings
from src.errors import ConfigError, ModelLoadError, STTError, STTQuotaError
from src.text_utils import normalize_urdu

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Transcript:
    """Result of one transcription."""

    text: str
    language: str
    audio_seconds: float
    elapsed_seconds: float
    backend: str = ""


class BaseSTT(ABC):
    """Common interface for every speech-to-text engine."""

    name: str = "base"

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        """Transcribe mono float32 audio."""


# --- guards shared by all backends ---------------------------------------------------

# Whisper continues the style of its prompt: a short everyday Urdu line nudges it to
# Urdu script (not Hindi/Devanagari or Roman). Keep it short and statement-like: on noise
# Whisper sometimes repeats the prompt back, and a question gets answered by the LLM.
URDU_PROMPT = "جی ٹھیک ہے، اردو میں بات کرتے ہیں۔"

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _strip(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def _echoes_prompt(text: str, prompt: str = URDU_PROMPT) -> bool:
    """True if `text` is just (part of) the prompt repeated back."""
    core = _strip(text)
    return bool(core) and len(core) >= 4 and core in _strip(prompt)


# Phrases speech models are known to invent on noise or silence (subtitle training data).
_HALLUCINATIONS = (
    "ترجمہ",
    "سبسکرائب",
    "subscribe",
    "شکریہ دیکھنے",
    "amara.org",
    "transcribed by",
)


def _looks_hallucinated(text: str, no_speech_prob: float, avg_logprob: float) -> bool:
    """True for segments that are most likely not real speech."""
    lowered = text.strip().lower()
    if not lowered:
        return True
    if any(phrase in lowered for phrase in _HALLUCINATIONS) or _echoes_prompt(lowered):
        return True
    # Whisper itself thinks there was no speech and is unsure of the words.
    return no_speech_prob > 0.6 and avg_logprob < -1.0


# People speak roughly 10-15 Urdu letters per second; far more means invented text.
_MAX_CHARS_PER_SECOND = 25


def _too_long(text: str, audio_seconds: float) -> bool:
    """True if `text` has more letters than anyone could say in `audio_seconds`."""
    return len(_strip(text)) > _MAX_CHARS_PER_SECOND * max(audio_seconds, 1.0)


# --- Gemini (online) -----------------------------------------------------------------

GEMINI_STT_INSTRUCTION = """\
You are a speech-to-text engine. Transcribe the audio exactly as spoken.
- The speaker talks Urdu (Pakistan). Write ONLY in Urdu script (Arabic/Nastaliq letters).
- Never translate, never answer or comment, never use Roman Urdu or Devanagari.
- English words the speaker uses (order, delivery, mobile) stay, written in Urdu script.
- Write numbers the way they were spoken.
- Output only the transcript. If there is no speech, output nothing.
"""

# Busy (503/500) usually clears within seconds. 429 is never retried: on the free tier it
# almost always means a per-minute or per-day quota, and waiting would stall the turn.
_RETRY_CODES = {500, 503}
# One quick retry only: a voice turn can't wait long, the Whisper fallback is better.
_RETRY_DELAYS_SECONDS = (1.0,)


def _quota_info(exc: Exception) -> tuple[str, float | None]:
    """(", quota X = N, retry in Ys", seconds to wait) from a 429's error details.

    Free-tier limits are per minute (short wait, the server says how long) or per day
    (wait for hours); the quota id says which.
    """
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return "", None
    found: list[str] = []
    retry_after: float | None = None
    daily = False
    for item in details.get("error", {}).get("details", []):
        for violation in item.get("violations", []):
            quota_id = str(violation.get("quotaId", ""))
            daily = daily or "PerDay" in quota_id
            found.append(f"quota {quota_id} = {violation.get('quotaValue')}")
        delay = str(item.get("retryDelay", ""))
        if delay.endswith("s"):
            try:
                retry_after = float(delay[:-1])
                found.append(f"retry in {delay}")
            except ValueError:
                pass
    if daily:
        retry_after = max(retry_after or 0.0, 3600.0)
    return "".join(f", {piece}" for piece in found), retry_after


class GeminiSTT(BaseSTT):
    """Urdu transcription with Gemini.

    `gemini-*-transcribe` models are dedicated speech models: they take the language
    through `audio_transcription_config` and return an `audio_transcription` part
    (without `language_codes=["ur"]` they write Urdu speech in Devanagari). Other
    Gemini models get a strict transcription instruction instead.
    """

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-transcribe",
        language: str = "ur",
        timeout_seconds: float = 10.0,  # the API rejects deadlines under 10 s
    ) -> None:
        if not api_key:
            raise ConfigError("GEMINI_API_KEY is not set; it is needed for STT_BACKEND=gemini.")
        from google import genai
        from google.genai import errors as genai_errors
        from google.genai import types

        self._types = types
        self._errors = genai_errors
        self.model = model
        self.language = language
        self._dedicated = "transcribe" in model
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000))
        )
        if self._dedicated:
            self._config = types.GenerateContentConfig(
                temperature=0,
                audio_transcription_config=types.AudioTranscriptionConfig(
                    language_codes=[language]
                ),
            )
        else:
            self._config = types.GenerateContentConfig(
                temperature=0,
                system_instruction=GEMINI_STT_INSTRUCTION,
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
            )

    @classmethod
    def from_settings(cls, settings: Settings) -> GeminiSTT:
        return cls(settings.gemini_api_key, settings.gemini_stt_model, settings.stt_language)

    def warm_up(self) -> None:
        """Open the connection and check key + model with a metadata call (no quota used)."""
        try:
            self._client.models.get(model=self.model)
        except self._errors.APIError as exc:
            raise STTError(f"Gemini STT model {self.model!r} is not usable: {exc}") from exc
        except Exception as exc:  # offline
            raise STTError(f"Gemini STT ({self.model}) unreachable: {exc!r}") from exc

    def _request(self, flac: bytes):  # noqa: ANN202 (SDK response type)
        parts = [self._types.Part.from_bytes(data=flac, mime_type="audio/flac")]
        if not self._dedicated:
            parts.append(self._types.Part.from_text(text="Transcribe this audio."))
        for attempt, delay in enumerate((*_RETRY_DELAYS_SECONDS, None), start=1):
            try:
                return self._client.models.generate_content(
                    model=self.model, contents=parts, config=self._config
                )
            except self._errors.APIError as exc:
                code = getattr(exc, "code", None)
                if code == 429:
                    info, retry_after = _quota_info(exc)
                    error = STTQuotaError(
                        f"Gemini STT ({self.model}) free quota reached (HTTP 429{info})."
                    )
                    error.retry_after = retry_after  # type: ignore[attr-defined]
                    raise error from exc
                if delay is None or code not in _RETRY_CODES:
                    raise STTError(f"Gemini STT ({self.model}) failed: HTTP {code} {exc}") from exc
                logger.warning("Gemini STT HTTP %s (try %d), retry in %.1fs", code, attempt, delay)
                time.sleep(delay)
            except Exception as exc:  # network down, DNS, timeout
                raise STTError(f"Gemini STT ({self.model}) unreachable: {exc!r}") from exc
        raise AssertionError("unreachable")

    def _text_of(self, response) -> str:  # noqa: ANN001
        if not response.candidates or not response.candidates[0].content:
            return ""
        pieces = []
        for part in response.candidates[0].content.parts or []:
            if getattr(part, "audio_transcription", None) is not None:
                pieces.append(part.audio_transcription.text or "")
            elif part.text and not getattr(part, "thought", False):
                pieces.append(part.text)
        return " ".join(p.strip() for p in pieces if p and p.strip())

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        started = time.perf_counter()
        buffer = io.BytesIO()
        sf.write(buffer, audio.astype(np.float32), sample_rate, format="FLAC")
        text = normalize_urdu(self._text_of(self._request(buffer.getvalue()))).strip()
        if _DEVANAGARI.search(text):
            raise STTError(f"Gemini STT ({self.model}) answered in Devanagari: {text!r}")
        seconds = len(audio) / sample_rate
        if text and (_looks_hallucinated(text, 0.0, 0.0) or _too_long(text, seconds)):
            logger.info("Dropped implausible transcript: %r", text)
            text = ""
        return Transcript(
            text=text,
            language=self.language,
            audio_seconds=len(audio) / sample_rate,
            elapsed_seconds=time.perf_counter() - started,
            backend=f"gemini:{self.model}",
        )


# --- Whisper (local) -----------------------------------------------------------------


def resolve_device(device: str, compute_type: str) -> tuple[str, str]:
    """Turn "auto" into cuda/float16 when an NVIDIA GPU is usable, else cpu/int8."""
    if device == "auto":
        import ctranslate2

        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


class FasterWhisperSTT(BaseSTT):
    """Local Whisper via faster-whisper (CTranslate2)."""

    name = "faster-whisper"

    def __init__(
        self,
        model: str = "medium",
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "ur",
        beam_size: int = 5,
        cpu_threads: int = 0,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # the cloud build has no local models
            raise ModelLoadError(
                "Local Whisper is not installed (requirements-cloud.txt has no local "
                "models). Install requirements.txt or use STT_BACKEND=gemini."
            ) from exc

        self.model_name = model
        self.language = language
        self.beam_size = beam_size
        self.device, self.compute_type = resolve_device(device, compute_type)
        logger.info(
            "Loading Whisper %r on %s (%s); first run downloads it to %s",
            model,
            self.device,
            self.compute_type,
            MODELS_DIR,
        )
        started = time.perf_counter()
        try:
            self._model = WhisperModel(
                model,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=cpu_threads,
                download_root=str(MODELS_DIR / "whisper"),
            )
        except Exception as exc:  # download, disk or format problems all land here
            raise ModelLoadError(
                f"Could not load Whisper model {model!r}: {exc}. Check your internet "
                "connection for the first download, free disk space, and that the "
                "model is a CTranslate2 model."
            ) from exc
        logger.info("Whisper loaded in %.1f s", time.perf_counter() - started)

    @classmethod
    def from_settings(cls, settings: Settings) -> FasterWhisperSTT:
        return cls(
            model=settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            language=settings.stt_language,
            beam_size=settings.stt_beam_size,
            cpu_threads=settings.stt_cpu_threads,
        )

    def warm_up(self) -> None:
        """Decode half a second of silence so the first real turn is not the slow one."""
        segments, _ = self._model.transcribe(
            np.zeros(8000, np.float32), language=self.language or None, beam_size=1
        )
        list(segments)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        if sample_rate != 16000:
            raise ValueError(f"Whisper expects 16000 Hz audio, got {sample_rate}.")
        started = time.perf_counter()
        segments, info = self._model.transcribe(
            audio.astype(np.float32),
            language=self.language or None,
            beam_size=self.beam_size,
            vad_filter=False,  # silence is handled by src.vad
            condition_on_previous_text=False,
            initial_prompt=URDU_PROMPT if self.language == "ur" else None,
            without_timestamps=True,  # fewer tokens to decode; timestamps are unused
            # One greedy/beam pass only. Temperature fallback re-decodes "failed" audio
            # several times; on noise that took 52 s for 1.4 s of sound.
            temperature=0.0,
        )
        # `segments` is a lazy generator: the actual decoding happens here.
        kept = []
        for segment in segments:
            if _looks_hallucinated(segment.text, segment.no_speech_prob, segment.avg_logprob):
                logger.info("Dropped likely hallucination: %r", segment.text.strip())
                continue
            kept.append(segment.text.strip())
        text = normalize_urdu(" ".join(kept)).strip()
        return Transcript(
            text=text,
            language=info.language,
            audio_seconds=len(audio) / sample_rate,
            elapsed_seconds=time.perf_counter() - started,
            backend=f"whisper:{self.model_name}",
        )


# --- fallback ------------------------------------------------------------------------


class FallbackSTT(BaseSTT):
    """Use `primary`; on STTError log a warning and use `fallback` (built on first use).

    After a failure the primary is skipped for a while, so every following turn does not
    pay for a doomed request first: after a 429 for as long as the server asks (a minute
    for per-minute limits, an hour or more for daily ones; QUOTA_PAUSE_SECONDS if it does
    not say), ERROR_PAUSE_SECONDS after other failures (offline, timeouts, wrong script).
    """

    QUOTA_PAUSE_SECONDS = 120.0
    ERROR_PAUSE_SECONDS = 60.0

    def __init__(
        self,
        primary: BaseSTT,
        make_fallback: Callable[[], BaseSTT],
        clock: Callable[[], float] = time.monotonic,
        preload: bool = False,
    ) -> None:
        """`preload`: build (and warm) the fallback during warm_up instead of at the first
        failure, so the turn that hits an exhausted quota doesn't also pay the model load."""
        self.primary = primary
        self._make_fallback = make_fallback
        self._preload = preload
        self._fallback: BaseSTT | None = None
        self._lock = threading.Lock()
        self._clock = clock
        self._skip_primary_until = 0.0
        self.name = f"{primary.name} (with fallback)"

    def _get_fallback(self) -> BaseSTT:
        with self._lock:
            if self._fallback is None:
                self._fallback = self._make_fallback()
            return self._fallback

    def warm_up(self) -> None:
        """Check the primary without spending quota; if it is unusable, load Whisper now."""
        warm = getattr(self.primary, "warm_up", None)
        try:
            if warm is not None:
                warm()
        except STTError as exc:
            logger.warning("%s Loading the fallback now.", exc)
            self._skip_primary_until = self._clock() + self.ERROR_PAUSE_SECONDS
            self._get_fallback()
        if self._preload:
            fallback = self._get_fallback()
            warm_fallback = getattr(fallback, "warm_up", None)
            if warm_fallback is not None:
                warm_fallback()

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        if self._clock() >= self._skip_primary_until:
            try:
                return self.primary.transcribe(audio, sample_rate)
            except STTQuotaError as exc:
                pause = getattr(exc, "retry_after", None) or self.QUOTA_PAUSE_SECONDS
                logger.warning("%s Using the fallback for %.0f s.", exc, pause)
                self._skip_primary_until = self._clock() + pause
                self._primary_error: STTError = exc
            except STTError as exc:
                pause = self.ERROR_PAUSE_SECONDS
                logger.warning("%s Using the fallback for %.0f s.", exc, pause)
                self._skip_primary_until = self._clock() + pause
                self._primary_error = exc
        try:
            fallback = self._get_fallback()
        except ModelLoadError as exc:
            # No usable fallback: report why the primary failed (e.g. the daily quota),
            # which is what the user can act on.
            raise getattr(self, "_primary_error", STTError(str(exc))) from exc
        return fallback.transcribe(audio, sample_rate)


def create_stt(settings: Settings) -> BaseSTT:
    """Build the STT backend named in STT_BACKEND ("gemini" or "whisper").

    GEMINI_STT_MODEL may list several models ("a,b"): each one falls back to the next
    (free-tier quotas are per model), and the last to local Whisper if STT_FALLBACK=whisper.
    """
    backend = settings.stt_backend
    if backend == "whisper":
        return FasterWhisperSTT.from_settings(settings)
    if backend != "gemini":
        raise ConfigError(f"STT_BACKEND={backend!r} is not supported. Use 'gemini' or 'whisper'.")

    models = [m.strip() for m in settings.gemini_stt_model.split(",") if m.strip()]
    if not models:
        raise ConfigError("GEMINI_STT_MODEL is empty; e.g. gemini-3.5-transcribe.")

    def whisper() -> BaseSTT:
        return FasterWhisperSTT.from_settings(settings)

    use_whisper = settings.stt_fallback
    if use_whisper and importlib.util.find_spec("faster_whisper") is None:
        logger.warning("STT_FALLBACK=whisper, but faster-whisper is not installed; no fallback.")
        use_whisper = False

    # On the laptop, load Whisper at startup (in the background warm-up): the free Gemini
    # quota runs out daily, and loading it at that moment would stall a turn by seconds.
    preload = use_whisper and getattr(settings, "app_mode", "local") == "local"

    def chain(index: int) -> BaseSTT:
        primary = GeminiSTT(settings.gemini_api_key, models[index], settings.stt_language)
        if index + 1 < len(models):
            return FallbackSTT(primary, lambda: chain(index + 1), preload=preload)
        return FallbackSTT(primary, whisper, preload=preload) if use_whisper else primary

    return chain(0)
