"""Voice activity detection with Silero VAD (runs locally, model ships with the package).

Two uses: offline (`speech_segments`, `trim`) to check and trim a finished recording,
and streaming (`chunk_probability`) so the mic stops recording when the user stops talking.

torch and silero-vad are imported only when a `SileroVAD` is created, so the cloud build
(which has neither: the browser does VAD there) can still import this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_RATES = (8000, 16000)
# Silero's streaming model takes exactly this many samples per call.
CHUNK_SAMPLES = {16000: 512, 8000: 256}


@dataclass(frozen=True)
class SpeechSegment:
    """A span of speech, in samples."""

    start: int
    end: int


class SileroVAD:
    """Thin wrapper around Silero VAD."""

    def __init__(self, threshold: float = 0.5, min_silence_ms: int = 300) -> None:
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        self._torch = torch
        self._get_speech_timestamps = get_speech_timestamps
        self._model = load_silero_vad()

    def speech_segments(self, audio: np.ndarray, sample_rate: int) -> list[SpeechSegment]:
        """Return the speech spans found in `audio`."""
        if sample_rate not in SUPPORTED_RATES:
            raise ValueError(f"Silero VAD needs 8000 or 16000 Hz audio, got {sample_rate}.")
        stamps = self._get_speech_timestamps(
            self._torch.from_numpy(audio.astype(np.float32)),
            self._model,
            sampling_rate=sample_rate,
            threshold=self.threshold,
            min_silence_duration_ms=self.min_silence_ms,
        )
        return [SpeechSegment(int(s["start"]), int(s["end"])) for s in stamps]

    def reset(self) -> None:
        """Clear the streaming state; call before each new utterance."""
        self._model.reset_states()

    def chunk_probability(self, chunk: np.ndarray, sample_rate: int = 16000) -> float:
        """Speech probability (0..1) of one chunk of `CHUNK_SAMPLES[sample_rate]` samples."""
        torch = self._torch
        with torch.inference_mode():
            return float(self._model(torch.from_numpy(chunk.astype(np.float32)), sample_rate))

    def has_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
        """True if any speech is detected."""
        return bool(self.speech_segments(audio, sample_rate))

    def trim(self, audio: np.ndarray, sample_rate: int, pad_ms: int = 200) -> np.ndarray:
        """Cut silence before the first and after the last speech segment."""
        segments = self.speech_segments(audio, sample_rate)
        if not segments:
            return audio
        pad = int(sample_rate * pad_ms / 1000)
        start = max(0, segments[0].start - pad)
        end = min(len(audio), segments[-1].end + pad)
        return audio[start:end]
