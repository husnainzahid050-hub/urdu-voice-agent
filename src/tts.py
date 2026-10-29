"""Text-to-speech. `BaseTTS` is the common interface.

Backends:
- `EdgeTTS`: Microsoft neural Urdu voices (ur-PK-UzmaNeural / ur-PK-AsadNeural) through
  the free edge-tts service. Natural pronunciation; needs internet.
- `MMSTTS`: Meta MMS VITS voice, runs fully locally; robotic but works offline.
- `FallbackTTS`: tries one backend and falls back to another if it fails.

Planned behind the same interface: a Piper Urdu voice.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from src.config import MODELS_DIR, Settings
from src.errors import ConfigError, ModelLoadError, TTSError
from src.text_utils import clean_for_tts, split_sentences

logger = logging.getLogger(__name__)

SENTENCE_PAUSE_SECONDS = 0.25


@dataclass(frozen=True)
class Speech:
    """Synthesised audio and how long it took."""

    audio: np.ndarray
    sample_rate: int
    elapsed_seconds: float
    backend: str = ""

    @property
    def audio_seconds(self) -> float:
        return len(self.audio) / self.sample_rate


class BaseTTS(ABC):
    """Common interface for every text-to-speech engine."""

    name: str = "base"
    sample_rate: int = 16000

    @abstractmethod
    def synthesize(self, text: str) -> Speech:
        """Clean `text` and return it as speech."""


class SentenceTTS(BaseTTS):
    """Base for engines that work best one sentence at a time (e.g. VITS)."""

    @abstractmethod
    def _synthesize_sentence(self, sentence: str) -> np.ndarray:
        """Return float32 mono audio at `self.sample_rate` for one clean sentence."""

    def synthesize(self, text: str) -> Speech:
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
        return Speech(audio, self.sample_rate, time.perf_counter() - started, self.name)


class EdgeTTS(BaseTTS):
    """Microsoft neural Urdu voice via edge-tts (free, online)."""

    name = "edge"

    def __init__(
        self, voice: str = "ur-PK-UzmaNeural", rate: str = "+0%", timeout: float = 15.0
    ) -> None:
        import edge_tts  # imported here so the offline backends work without it

        self._edge_tts = edge_tts
        self.voice = voice
        self.rate = rate
        self.timeout = timeout

    async def _fetch_mp3(self, text: str) -> bytes:
        communicate = self._edge_tts.Communicate(text, self.voice, rate=self.rate)
        mp3 = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3.extend(chunk["data"])
        return bytes(mp3)

    def synthesize(self, text: str) -> Speech:
        started = time.perf_counter()
        clean = clean_for_tts(text)
        if not clean:
            raise ValueError("Nothing left to speak after cleaning the text.")
        try:
            mp3 = asyncio.run(asyncio.wait_for(self._fetch_mp3(clean), self.timeout))
        except Exception as exc:  # network down, DNS, service change, timeout
            raise TTSError(f"Online Urdu voice {self.voice!r} failed: {exc!r}") from exc
        if not mp3:
            raise TTSError(f"Online Urdu voice {self.voice!r} returned no audio.")
        audio, rate = sf.read(io.BytesIO(mp3), dtype="float32", always_2d=True)
        self.sample_rate = int(rate)
        return Speech(
            audio.mean(axis=1), self.sample_rate, time.perf_counter() - started, self.name
        )


class MMSTTS(SentenceTTS):
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


class FallbackTTS(BaseTTS):
    """Use `primary`; if it fails, log a warning and use `fallback` (built on first use)."""

    def __init__(self, primary: BaseTTS, make_fallback: Callable[[], BaseTTS]) -> None:
        self.primary = primary
        self._make_fallback = make_fallback
        self._fallback: BaseTTS | None = None
        self.name = f"{primary.name} (falls back to offline voice)"

    def synthesize(self, text: str) -> Speech:
        try:
            return self.primary.synthesize(text)
        except TTSError as exc:
            logger.warning("%s Using the offline voice instead.", exc)
            if self._fallback is None:
                self._fallback = self._make_fallback()
            return self._fallback.synthesize(text)


def create_tts(settings: Settings) -> BaseTTS:
    """Build the TTS backend named in TTS_BACKEND."""
    backend = settings.tts_backend
    if backend == "mms":
        return MMSTTS(settings.tts_model)
    if backend == "edge":
        return FallbackTTS(
            EdgeTTS(settings.tts_voice, settings.tts_rate),
            lambda: MMSTTS(settings.tts_model),
        )
    raise ConfigError(f"TTS_BACKEND={backend!r} is not supported. Use 'edge' or 'mms'.")
