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
import concurrent.futures
import io
import logging
import threading
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


@dataclass(frozen=True)
class EncodedSpeech:
    """Speech as a ready-to-play file (e.g. for a browser), and how long it took."""

    data: bytes
    mime: str
    backend: str
    elapsed_seconds: float


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """16-bit PCM WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class BaseTTS(ABC):
    """Common interface for every text-to-speech engine."""

    name: str = "base"
    sample_rate: int = 16000

    @abstractmethod
    def synthesize(self, text: str) -> Speech:
        """Clean `text` and return it as speech."""

    def synthesize_encoded(self, text: str) -> EncodedSpeech:
        """Clean `text` and return it as an audio file. Default: WAV of `synthesize`.

        Safe to call from several threads at once (callers synthesise sentences in
        parallel); engines that are not thread-safe must serialise internally.
        """
        speech = self.synthesize(text)
        data = encode_wav(speech.audio, speech.sample_rate)
        return EncodedSpeech(data, "audio/wav", speech.backend, speech.elapsed_seconds)

    def with_voice(self, voice: str) -> BaseTTS:
        """The same engine speaking with `voice` (shares connections and loaded models).
        Engines with a single voice return themselves."""
        return self


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
    """Microsoft neural Urdu voice via edge-tts (free, online).

    Requests run on one long-lived asyncio loop in a background thread: reusing the
    loop is faster than `asyncio.run` per call (no new loop, warm DNS/TLS), and several
    sentences can be fetched in parallel from different threads.
    """

    name = "edge"

    def __init__(
        self,
        voice: str = "ur-PK-UzmaNeural",
        rate: str = "+0%",
        timeout: float = 8.0,
        hedge_after: float = 4.0,
    ) -> None:
        """`timeout` is per attempt. One sentence normally takes 1.5-4 s, but on slow
        lines a request now and then stalls for 15 s+ or returns nothing; so after
        `hedge_after` seconds (or at once if the first attempt fails) a second request
        is started and whichever finishes first wins."""
        import edge_tts  # imported here so the offline backends work without it

        self._edge_tts = edge_tts
        self.voice = voice
        self.rate = rate
        self.timeout = timeout
        self.hedge_after = hedge_after
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_lock = threading.Lock()

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                threading.Thread(
                    target=self._loop.run_forever, name="edge-tts-loop", daemon=True
                ).start()
            return self._loop

    def with_voice(self, voice: str) -> EdgeTTS:
        if voice == self.voice:
            return self
        other = EdgeTTS(voice, self.rate, self.timeout, self.hedge_after)
        other._loop, other._loop_lock = self._get_loop(), self._loop_lock  # one shared loop
        return other

    def _fetch_mp3_sync(self, clean: str) -> bytes:
        """Fetch MP3 for already-cleaned text with at most two attempts; raises TTSError.

        The second attempt is a hedge, not a retry after a full timeout: it starts as soon
        as the first one fails or is slower than `hedge_after`. Unfinished attempts are
        cancelled, so stalled requests never pile up on the shared event loop.
        """
        loop = self._get_loop()

        def attempt() -> concurrent.futures.Future[bytes]:
            return asyncio.run_coroutine_threadsafe(self._fetch_mp3(clean), loop)

        pending = [attempt()]
        hedged = False
        deadline = time.monotonic() + self.hedge_after + self.timeout
        error: Exception = TTSError("timed out")
        try:
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                wait = remaining if hedged else min(remaining, self.hedge_after)
                done, _ = concurrent.futures.wait(
                    pending, timeout=wait, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    pending.remove(future)
                    try:
                        mp3 = future.result()
                    except Exception as exc:  # network down, DNS, service change
                        error = exc
                        continue
                    if mp3:
                        return mp3
                    error = TTSError("no audio received")
                if not hedged:  # slow or failed: race a second request
                    logger.info("Online voice slow or failed (%r); racing a 2nd request", error)
                    pending.append(attempt())
                    hedged = True
        finally:
            for future in pending:
                future.cancel()
        raise TTSError(f"Online Urdu voice {self.voice!r} failed: {error!r}") from error

    def synthesize_encoded(self, text: str) -> EncodedSpeech:
        """MP3 straight from the service: no decoding, smaller than WAV."""
        started = time.perf_counter()
        clean = clean_for_tts(text)
        if not clean:
            raise ValueError("Nothing left to speak after cleaning the text.")
        mp3 = self._fetch_mp3_sync(clean)
        return EncodedSpeech(mp3, "audio/mpeg", self.name, time.perf_counter() - started)

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
        mp3 = self._fetch_mp3_sync(clean)
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
        self._infer_lock = threading.Lock()  # one VITS forward pass at a time
        logger.info("TTS loaded in %.1f s", time.perf_counter() - started)

    def synthesize(self, text: str) -> Speech:
        with self._infer_lock:
            return super().synthesize(text)

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

    def __init__(
        self,
        primary: BaseTTS,
        make_fallback: Callable[[], BaseTTS],
        _shared: dict[str, object] | None = None,
    ) -> None:
        self.primary = primary
        self._make_fallback = make_fallback
        # One fallback model for every voice (see with_voice): {"tts": ..., "lock": ...}
        self._shared = _shared or {"tts": None, "lock": threading.Lock()}
        self.name = f"{primary.name} (falls back to offline voice)"

    @property
    def _fallback(self) -> BaseTTS | None:
        return self._shared["tts"]  # type: ignore[return-value]

    def preload_fallback(self) -> None:
        """Load the fallback now (e.g. at startup, in the background): loading it at the
        moment the online voice fails took ~30 s and stalled that reply."""
        with self._shared["lock"]:  # type: ignore[attr-defined]
            if self._shared["tts"] is None:
                self._shared["tts"] = self._make_fallback()

    def with_voice(self, voice: str) -> FallbackTTS:
        primary = self.primary.with_voice(voice)
        if primary is self.primary:
            return self
        return FallbackTTS(primary, self._make_fallback, self._shared)

    def _get_fallback(self, exc: Exception) -> BaseTTS:
        logger.warning("%s Using the offline voice instead.", exc)
        self.preload_fallback()  # parallel sentences must not load the model twice
        assert self._fallback is not None
        return self._fallback

    def synthesize(self, text: str) -> Speech:
        try:
            return self.primary.synthesize(text)
        except TTSError as exc:
            return self._get_fallback(exc).synthesize(text)

    def synthesize_encoded(self, text: str) -> EncodedSpeech:
        try:
            return self.primary.synthesize_encoded(text)
        except TTSError as exc:
            return self._get_fallback(exc).synthesize_encoded(text)


def create_tts(settings: Settings) -> BaseTTS:
    """Build the TTS backend named in TTS_BACKEND."""
    backend = settings.tts_backend
    if backend == "mms":
        return MMSTTS(settings.tts_model)
    if backend == "edge":
        edge = EdgeTTS(settings.tts_voice, settings.tts_rate)
        if getattr(settings, "tts_fallback", "mms") == "none":  # cloud: no torch/MMS
            return edge
        return FallbackTTS(edge, lambda: MMSTTS(settings.tts_model))
    raise ConfigError(f"TTS_BACKEND={backend!r} is not supported. Use 'edge' or 'mms'.")
