"""Voice activity detection with Silero VAD (runs locally, model ships with the package).

Phase 1 uses it to check that a recording actually contains speech and to trim
leading/trailing silence before STT. Phase 2 will use it for hands-free turn-taking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

logger = logging.getLogger(__name__)

SUPPORTED_RATES = (8000, 16000)


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
        self._model = load_silero_vad()

    def speech_segments(self, audio: np.ndarray, sample_rate: int) -> list[SpeechSegment]:
        """Return the speech spans found in `audio`."""
        if sample_rate not in SUPPORTED_RATES:
            raise ValueError(f"Silero VAD needs 8000 or 16000 Hz audio, got {sample_rate}.")
        stamps = get_speech_timestamps(
            torch.from_numpy(audio.astype(np.float32)),
            self._model,
            sampling_rate=sample_rate,
            threshold=self.threshold,
            min_silence_duration_ms=self.min_silence_ms,
        )
        return [SpeechSegment(int(s["start"]), int(s["end"])) for s in stamps]

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
