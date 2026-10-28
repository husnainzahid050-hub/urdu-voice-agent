"""Microphone recording, speaker playback and WAV files.

Audio is always mono float32 in [-1, 1] as a 1-D numpy array.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from src.errors import AudioDeviceError

logger = logging.getLogger(__name__)


def _check_device(kind: str) -> None:
    """Raise AudioDeviceError if there is no default input/output device."""
    try:
        sd.query_devices(kind=kind)
    except (sd.PortAudioError, ValueError) as exc:
        what = "microphone" if kind == "input" else "speaker"
        raise AudioDeviceError(
            f"No {what} found. Plug one in or set it as the default device in Windows "
            f"Sound settings. ({exc})"
        ) from exc


def default_device_name(kind: str = "input") -> str:
    """Name of the default input or output device."""
    _check_device(kind)
    return str(sd.query_devices(kind=kind)["name"])


def record(seconds: float, sample_rate: int = 16000) -> np.ndarray:
    """Record `seconds` of mono audio from the default microphone (blocking)."""
    _check_device("input")
    frames = int(seconds * sample_rate)
    logger.debug("Recording %.1f s at %d Hz", seconds, sample_rate)
    try:
        audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
    except sd.PortAudioError as exc:
        raise AudioDeviceError(f"Recording from the microphone failed: {exc}") from exc
    return audio.reshape(-1)


def play(audio: np.ndarray, sample_rate: int) -> None:
    """Play audio on the default speaker and wait until it finishes."""
    _check_device("output")
    try:
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
    except sd.PortAudioError as exc:
        raise AudioDeviceError(f"Playback failed: {exc}") from exc


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write audio as 16-bit PCM WAV, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    return path


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV/FLAC file as mono float32 audio and its sample rate."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(axis=1), int(sample_rate)


def peak_level(audio: np.ndarray) -> float:
    """Peak absolute amplitude (0 = silence, 1 = full scale)."""
    return float(np.max(np.abs(audio))) if audio.size else 0.0
