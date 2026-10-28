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

load_dotenv(ROOT_DIR / ".env")

# Windows without Developer Mode has no symlinks; the HF cache still works, so skip the warning.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# huggingface_hub resets its log level on import; this also hides the HF_TOKEN nag
# (all models used here are public).
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
# Silero VAD and MMS still call torch.jit.script; harmless deprecation noise on torch 2.15.
warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*", category=FutureWarning)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


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

    # Speech-to-text (faster-whisper)
    stt_model: str
    stt_device: str
    stt_compute_type: str
    stt_language: str
    stt_beam_size: int

    # Text-to-speech
    tts_backend: str
    tts_model: str

    # Audio
    sample_rate: int
    record_seconds: float

    log_level: str

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the environment and `.env` file."""
        thinking = _env("GEMINI_THINKING_BUDGET")
        return cls(
            gemini_api_key=_env("GEMINI_API_KEY"),
            gemini_model=_env("GEMINI_MODEL"),
            gemini_temperature=_env_float("GEMINI_TEMPERATURE", 0.6),
            gemini_max_output_tokens=_env_int("GEMINI_MAX_OUTPUT_TOKENS", 400),
            gemini_thinking_level=_env("GEMINI_THINKING_LEVEL").upper() or None,
            gemini_thinking_budget=_env_int("GEMINI_THINKING_BUDGET", 0) if thinking else None,
            stt_model=_env("STT_MODEL", "medium"),
            stt_device=_env("STT_DEVICE", "auto"),
            stt_compute_type=_env("STT_COMPUTE_TYPE", "auto"),
            stt_language=_env("STT_LANGUAGE", "ur"),
            stt_beam_size=_env_int("STT_BEAM_SIZE", 5),
            tts_backend=_env("TTS_BACKEND", "mms"),
            tts_model=_env("TTS_MODEL", "facebook/mms-tts-urd-script_arabic"),
            sample_rate=_env_int("SAMPLE_RATE", 16000),
            record_seconds=_env_float("RECORD_SECONDS", 5.0),
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
