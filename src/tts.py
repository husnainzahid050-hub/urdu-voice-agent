"""Text-to-speech. `BaseTTS` is the common interface; `MMSTTS` runs locally.

Planned backends behind the same interface: Piper (local Urdu voice) and
edge-tts (online fallback only).
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.config import MODELS_DIR, Settings
from src.errors import ConfigError, ModelLoadError
from src.text_utils import clean_for_tts, split_sentences

logger = logging.getLogger(__name__)

SENTENCE_PAUSE_SECONDS = 0.25


@dataclass(frozen=True)
class Speech:
    """Synthesised audio and how long it took."""

    audio: np.ndarray
    sample_rate: int
    elapsed_seconds: float

    @property
    def audio_seconds(self) -> float:
        return len(self.audio) / self.sample_rate


class BaseTTS(ABC):
    """Common interface for every text-to-speech engine."""

    name: str = "base"
    sample_rate: int = 16000

    @abstractmethod
    def _synthesize_sentence(self, sentence: str) -> np.ndarray:
        """Return float32 mono audio at `self.sample_rate` for one clean sentence."""

    def synthesize(self, text: str) -> Speech:
        """Clean `text`, speak it sentence by sentence and join with short pauses."""
        started = time.perf_counter()
        sentences = split_sentences(clean_for_tts(text))
        if not sentences:
            raise ValueError("Nothing left to speak after cleaning the text.")
        pause = np.zeros(int(self.sample_rate * SENTENCE_PAUSE_SECONDS), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for sentence in sentences:
            logger.debug("TTS sentence: %s", sentence)
            chunks.extend([self._synthesize_sentence(sentence), pause])
        audio = np.concatenate(chunks[:-1])
        return Speech(audio, self.sample_rate, time.perf_counter() - started)


class MMSTTS(BaseTTS):
    """Meta MMS VITS voice for Urdu (Arabic script), via Hugging Face Transformers."""

    name = "mms"

    def __init__(self, model: str = "facebook/mms-tts-urd-script_arabic", seed: int = 555) -> None:
        import torch
        from transformers import VitsModel, VitsTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        self.model_name = model
        self.seed = seed
        self._torch = torch
        cache = str(MODELS_DIR / "tts")
        logger.info("Loading TTS %r; first run downloads it to %s", model, cache)
        started = time.perf_counter()
        try:
            self._tokenizer = VitsTokenizer.from_pretrained(model, cache_dir=cache)
            self._model = VitsModel.from_pretrained(model, cache_dir=cache).eval()
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load TTS model {model!r}: {exc}. Check your internet "
                "connection for the first download and free disk space."
            ) from exc
        self.sample_rate = int(self._model.config.sampling_rate)
        logger.info("TTS loaded in %.1f s", time.perf_counter() - started)

    def _synthesize_sentence(self, sentence: str) -> np.ndarray:
        inputs = self._tokenizer(sentence, return_tensors="pt")
        if inputs["input_ids"].shape[-1] == 0:
            logger.warning("TTS skipped a sentence with no speakable characters: %r", sentence)
            return np.zeros(0, dtype=np.float32)
        # VITS samples noise; a fixed seed makes the same text sound the same every time.
        self._torch.manual_seed(self.seed)
        with self._torch.inference_mode():
            waveform = self._model(**inputs).waveform
        return waveform.squeeze().cpu().numpy().astype(np.float32)


def create_tts(settings: Settings) -> BaseTTS:
    """Build the TTS backend named in TTS_BACKEND."""
    if settings.tts_backend == "mms":
        return MMSTTS(settings.tts_model)
    raise ConfigError(f"TTS_BACKEND={settings.tts_backend!r} is not available yet. Supported: mms.")
