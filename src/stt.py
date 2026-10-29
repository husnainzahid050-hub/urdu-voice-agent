"""Speech-to-text. `BaseSTT` is the common interface; `FasterWhisperSTT` runs locally.

`STT_MODEL` can be a Whisper size ("small", "medium", "large-v3"), a Hugging Face
repo that already holds a CTranslate2 model, or a local folder with one. Fine-tuned
Transformers checkpoints (e.g. ihanif/whisper-medium-urdu) must be converted to
CTranslate2 first; that is part of the STT benchmark in a later phase.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.config import MODELS_DIR, Settings
from src.errors import ModelLoadError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Transcript:
    """Result of one transcription."""

    text: str
    language: str
    audio_seconds: float
    elapsed_seconds: float


class BaseSTT(ABC):
    """Common interface for every speech-to-text engine."""

    name: str = "base"

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        """Transcribe mono float32 audio."""


# Whisper continues the style of its prompt: a short everyday Urdu line nudges it to
# Urdu script (not Hindi/Devanagari or Roman). Keep it short and statement-like: on noise
# Whisper sometimes repeats the prompt back, and a question gets answered by the LLM.
URDU_PROMPT = "جی ٹھیک ہے، اردو میں بات کرتے ہیں۔"


def _strip(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def _echoes_prompt(text: str, prompt: str = URDU_PROMPT) -> bool:
    """True if `text` is just (part of) the prompt repeated back."""
    core = _strip(text)
    return bool(core) and len(core) >= 4 and core in _strip(prompt)


# Phrases Whisper is known to invent on noise or silence (from subtitle training data).
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
    ) -> None:
        from faster_whisper import WhisperModel

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
        )

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
        text = " ".join(kept).strip()
        return Transcript(
            text=text,
            language=info.language,
            audio_seconds=len(audio) / sample_rate,
            elapsed_seconds=time.perf_counter() - started,
        )
