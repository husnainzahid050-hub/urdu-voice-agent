"""Microphone recording, speaker playback and WAV files.

Audio is always mono float32 in [-1, 1] as a 1-D numpy array.
"""

from __future__ import annotations

import collections
import logging
import queue
import time
from collections.abc import Callable
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


def record_utterance(
    speech_probability: Callable[[np.ndarray], float],
    sample_rate: int = 16000,
    chunk_samples: int = 512,
    end_silence_ms: int = 900,
    max_seconds: float = 20.0,
    start_timeout_seconds: float = 10.0,
    on_speech_start: Callable[[], None] | None = None,
    min_speech_ms: int = 300,
) -> np.ndarray | None:
    """Record one spoken turn: wait for speech, stop after `end_silence_ms` of silence.

    `speech_probability` scores one chunk (e.g. `SileroVAD.chunk_probability`).
    Returns None if nobody spoke within `start_timeout_seconds`, or if the "speech"
    was shorter than `min_speech_ms` (a cough, a click, a door): sending noise to
    Whisper makes it invent text and can take a minute.
    """
    _check_device("input")
    chunk_ms = 1000 * chunk_samples / sample_rate
    start_threshold, stop_threshold = 0.5, 0.35  # hysteresis avoids cutting soft syllables
    start_chunks = max(1, round(200 / chunk_ms))  # ~200 ms of voice confirms a start
    preroll: collections.deque[np.ndarray] = collections.deque(maxlen=round(400 / chunk_ms))
    blocks: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata: np.ndarray, _frames: int, _time: object, status: object) -> None:
        if status:
            logger.debug("Mic status: %s", status)
        blocks.put(indata[:, 0].copy())

    voiced: list[np.ndarray] = []
    started = False
    loud_run = 0
    silence_ms = 0.0
    speech_ms = 0.0
    began = time.monotonic()
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            callback=callback,
        ):
            while True:
                chunk = blocks.get(timeout=2.0)
                prob = speech_probability(chunk)
                if not started:
                    preroll.append(chunk)
                    loud_run = loud_run + 1 if prob >= start_threshold else 0
                    if loud_run >= start_chunks:
                        started = True
                        speech_ms = loud_run * chunk_ms
                        voiced.extend(preroll)  # keep the first syllable
                        if on_speech_start:
                            on_speech_start()
                    elif time.monotonic() - began > start_timeout_seconds:
                        return None
                    continue
                voiced.append(chunk)
                if prob >= start_threshold:
                    speech_ms += chunk_ms
                silence_ms = silence_ms + chunk_ms if prob < stop_threshold else 0.0
                if silence_ms >= end_silence_ms:
                    break
                if len(voiced) * chunk_ms / 1000 >= max_seconds:
                    logger.info("Turn reached MAX_TURN_SECONDS (%.0f s); stopping.", max_seconds)
                    break
    except queue.Empty as exc:
        raise AudioDeviceError("The microphone stopped sending audio.") from exc
    except sd.PortAudioError as exc:
        raise AudioDeviceError(f"Recording from the microphone failed: {exc}") from exc
    if speech_ms < min_speech_ms:
        logger.info("Ignored %.0f ms of sound (too short to be speech).", speech_ms)
        return None
    return np.concatenate(voiced)


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


def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Linear-interpolation resampling; good enough for speech going into Whisper."""
    if from_rate == to_rate or audio.size == 0:
        return audio
    n_out = int(round(len(audio) * to_rate / from_rate))
    positions = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


def peak_level(audio: np.ndarray) -> float:
    """Peak absolute amplitude (0 = silence, 1 = full scale)."""
    return float(np.max(np.abs(audio))) if audio.size else 0.0
