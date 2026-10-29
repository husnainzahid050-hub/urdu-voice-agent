"""All settings, loaded from environment variables (.env) with sensible defaults.

Every setting can be overridden in `.env`; see `.env.example` for the full list.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.errors import ConfigError

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "output"

# ENV_FILE points at another file (tests use a missing one to ignore the owner's .env).
# On a host like Render there is no .env: the platform's environment variables are used.
load_dotenv(os.getenv("ENV_FILE") or ROOT_DIR / ".env")

# Windows without Developer Mode has no symlinks; the HF cache still works, so skip the warning.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# huggingface_hub resets its log level on import; this also hides the HF_TOKEN nag
# (all models used here are public).
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
# The Xet downloader stalled at 0 bytes on a slow connection; plain HTTP downloads fine.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Silero VAD and MMS still use torch.jit; harmless deprecation noise on torch 2.15.
warnings.filterwarnings("ignore", message=r".*torch\.jit\.\w+.*deprecated", category=FutureWarning)


def _env(name: str, default: str = "") -> str:
    """The variable's value; an unset or empty variable (`NAME=` in .env) means `default`."""
    return os.getenv(name, "").strip() or default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a whole number, got {raw!r}.") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}.") from exc


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = _env(name, default).lower()
    if value not in allowed:
        raise ConfigError(f"{name} must be one of {', '.join(allowed)}; got {value!r}.")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def app_mode() -> str:
    """APP_MODE: "local" (laptop, all models) or "cloud" (public server, online models only)."""
    return "cloud" if _env("APP_MODE", "local").lower() == "cloud" else "local"


@dataclass(frozen=True)
class WebSettings:
    """Web server protection settings (sessions, rate limits, headers).

    Defaults are relaxed for "local" and strict for "cloud" (a public link must not be
    able to burn the owner's free Gemini quota). 0 disables a limit.
    """

    app_mode: str
    trust_proxy: bool  # read the client IP from X-Forwarded-For (only behind a proxy)
    turns_per_minute: int  # per client IP
    turns_per_day: int  # per client IP
    daily_turn_limit: int  # all clients together
    allowed_origins: tuple[str, ...]  # CORS; empty = same origin only
    session_ttl_seconds: int
    max_sessions: int
    security_headers: bool

    @property
    def cloud(self) -> bool:
        return self.app_mode == "cloud"

    @classmethod
    def from_env(cls) -> WebSettings:
        cloud = app_mode() == "cloud"
        return cls(
            app_mode="cloud" if cloud else "local",
            trust_proxy=_env_bool("TRUST_PROXY", False),
            turns_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", 8 if cloud else 60),
            turns_per_day=_env_int("RATE_LIMIT_PER_DAY", 30 if cloud else 0),
            # Google's free tier allows ~20-25 requests/day per model (each turn = STT + chat),
            # so the honest public default is ~100 turns/day across the fallback models.
            daily_turn_limit=_env_int("DAILY_TURN_LIMIT", 100 if cloud else 0),
            allowed_origins=tuple(
                o.strip().rstrip("/") for o in _env("ALLOWED_ORIGINS").split(",") if o.strip()
            ),
            session_ttl_seconds=_env_int("SESSION_TTL_SECONDS", 1800),
            max_sessions=_env_int("MAX_SESSIONS", 200),
            security_headers=_env_bool("SECURITY_HEADERS", True),
        )


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Create with `Settings.from_env()`."""

    # LLM (Gemini free tier)
    gemini_api_key: str
    gemini_model: str
    gemini_temperature: float
    gemini_max_output_tokens: int
    gemini_thinking_level: str | None
    gemini_thinking_budget: int | None
    gemini_fallback_models: tuple[str, ...]

    # Speech-to-text
    stt_backend: str
    stt_model: str
    stt_device: str
    stt_compute_type: str
    stt_language: str
    stt_beam_size: int
    stt_cpu_threads: int
    gemini_stt_model: str
    stt_fallback: bool

    # Text-to-speech
    tts_backend: str
    tts_model: str
    tts_voice: str
    tts_rate: str

    # Agent persona
    agent_gender: str

    # Audio
    sample_rate: int
    record_seconds: float
    max_turn_seconds: float
    end_silence_ms: int
    start_timeout_seconds: float

    log_level: str

    # Deployment (APP_MODE=cloud: no torch/Whisper/MMS/Silero, nothing logged to disk).
    app_mode: str = "local"
    server_vad: bool = True  # Silero on the server; the browser already does VAD
    tts_fallback: str = "mms"  # mms | none
    latency_log: bool = True  # logs/latency.csv
    log_transcripts: bool = True  # what users said, in logs

    @property
    def cloud(self) -> bool:
        return self.app_mode == "cloud"

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the environment and `.env` file."""
        thinking = _env("GEMINI_THINKING_BUDGET")
        cloud = app_mode() == "cloud"
        return cls(
            app_mode="cloud" if cloud else "local",
            server_vad=_env_bool("SERVER_VAD", not cloud),
            tts_fallback=_choice("TTS_FALLBACK", "none" if cloud else "mms", ("mms", "none")),
            latency_log=_env_bool("LATENCY_LOG", not cloud),
            log_transcripts=_env_bool("LOG_TRANSCRIPTS", not cloud),
            gemini_api_key=_env("GEMINI_API_KEY"),
            gemini_model=_env("GEMINI_MODEL"),
            gemini_temperature=_env_float("GEMINI_TEMPERATURE", 0.6),
            gemini_max_output_tokens=_env_int("GEMINI_MAX_OUTPUT_TOKENS", 400),
            gemini_thinking_level=_env("GEMINI_THINKING_LEVEL").upper() or None,
            gemini_thinking_budget=_env_int("GEMINI_THINKING_BUDGET", 0) if thinking else None,
            gemini_fallback_models=tuple(
                m.strip()
                for m in _env(
                    "GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3.5-flash"
                ).split(",")
                if m.strip() and m.strip().lower() != "none"  # "none" = no fallback
            ),
            stt_backend=_env("STT_BACKEND", "gemini").lower(),  # local falls back to Whisper
            stt_model=_env("STT_MODEL", "small"),
            stt_device=_env("STT_DEVICE", "auto"),
            stt_compute_type=_env("STT_COMPUTE_TYPE", "auto"),
            stt_language=_env("STT_LANGUAGE", "ur"),
            stt_beam_size=_env_int("STT_BEAM_SIZE", 5),
            stt_cpu_threads=_env_int("STT_CPU_THREADS", 0),
            gemini_stt_model=_env(
                "GEMINI_STT_MODEL", "gemini-3.5-transcribe,gemini-3.1-flash-lite"
            ),
            stt_fallback=_env("STT_FALLBACK", "none" if cloud else "whisper").lower() == "whisper",
            tts_backend=_env("TTS_BACKEND", "edge").lower(),
            tts_model=_env("TTS_MODEL", "facebook/mms-tts-urd-script_arabic"),
            tts_voice=_env("TTS_VOICE", "ur-PK-UzmaNeural"),
            tts_rate=_env("TTS_RATE", "+0%"),
            agent_gender=_choice("AGENT_GENDER", "female", ("female", "male")),
            sample_rate=_env_int("SAMPLE_RATE", 16000),
            record_seconds=_env_float("RECORD_SECONDS", 5.0),
            max_turn_seconds=_env_float("MAX_TURN_SECONDS", 20.0),
            end_silence_ms=_env_int("END_SILENCE_MS", 900),
            start_timeout_seconds=_env_float("START_TIMEOUT_SECONDS", 10.0),
            log_level=_env("LOG_LEVEL", "INFO").upper(),
        )

    def require_gemini(self) -> None:
        """Raise a readable ConfigError if the Gemini settings are incomplete."""
        if not self.gemini_api_key:
            raise ConfigError(
                "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
                "and put it in .env (copy .env.example to .env first)."
            )
        if not self.gemini_model:
            raise ConfigError(
                "GEMINI_MODEL is not set. Run `python -m scripts.list_gemini_models` "
                "and put one of the suggested model names in .env."
            )


def setup_logging(level: str = "INFO") -> None:
    """Configure console logging once for the whole app."""
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Third-party libraries are noisy at INFO.
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "faster_whisper", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
