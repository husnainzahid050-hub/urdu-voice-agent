"""Voice pipeline: audio -> VAD -> STT -> LLM -> TTS -> audio.

- `Conversation` is the core: it loads the models once and turns speech into a spoken
  reply. Both front ends use it.
- `VoiceAgent` is the console front end (laptop mic and speaker, hands-free loop).
- `src.server` is the browser front end (mic button, chat bubbles).
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from src import audio_io
from src.config import ROOT_DIR, Settings
from src.errors import LLMError, TTSError
from src.llm import GeminiChat
from src.stt import BaseSTT, create_stt
from src.tts import BaseTTS, Speech, create_tts
from src.vad import CHUNK_SAMPLES, SileroVAD

logger = logging.getLogger(__name__)

LATENCY_LOG = ROOT_DIR / "logs" / "latency.csv"
RATE = 16000
MIN_SPEECH_SECONDS = 0.3
LATENCY_COLUMNS = ["time", "speech_s", "stt_s", "llm_s", "tts_s", "response_s", "tts", "heard"]

# Saying any of these ends the conversation.
GOODBYE_WORDS = ("خدا حافظ", "اللہ حافظ", "بند کرو", "بند کریں", "بس کرو", "الوداع")

# Voices and the grammatical gender that goes with each.
VOICES = {"female": "ur-PK-UzmaNeural", "male": "ur-PK-AsadNeural"}

# Fixed lines the agent says itself, per grammatical gender.
LINES = {
    "female": {
        "greeting": (
            "السلام علیکم! میں آپ کی اردو اسسٹنٹ ہوں۔ بتائیں، میں آپ کی کیا مدد کر سکتی ہوں؟"
        ),
        "not_heard": "معاف کیجیے، آواز صاف نہیں آئی۔ ذرا دوبارہ بولیں۔",
        "llm_failed": "معاف کیجیے، ابھی جواب نہیں دے سکی۔ تھوڑی دیر بعد دوبارہ کوشش کریں۔",
        "goodbye": "ٹھیک ہے، اللہ حافظ! آپ سے بات کر کے اچھا لگا۔",
    },
    "male": {
        "greeting": (
            "السلام علیکم! میں آپ کا اردو اسسٹنٹ ہوں۔ بتائیں، میں آپ کی کیا مدد کر سکتا ہوں؟"
        ),
        "not_heard": "معاف کیجیے، آواز صاف نہیں آئی۔ ذرا دوبارہ بولیں۔",
        "llm_failed": "معاف کیجیے، ابھی جواب نہیں دے سکا۔ تھوڑی دیر بعد دوبارہ کوشش کریں۔",
        "goodbye": "ٹھیک ہے، اللہ حافظ! آپ سے بات کر کے اچھا لگا۔",
    },
}


@dataclass
class TurnTimings:
    """Seconds spent in each stage of one turn."""

    speech_seconds: float = 0.0
    stt: float = 0.0
    llm: float = 0.0
    tts: float = 0.0
    tts_backend: str = ""

    @property
    def response(self) -> float:
        """From the end of the user's speech to the start of the reply audio."""
        return self.stt + self.llm + self.tts


@dataclass
class Heard:
    """Result of listening to one turn. `text` is empty if no speech was found."""

    text: str
    speech_seconds: float
    stt_seconds: float


@dataclass
class Reply:
    """The agent's answer to one turn."""

    text: str
    speech: Speech | None
    timings: TurnTimings
    goodbye: bool = False


def is_goodbye(text: str) -> bool:
    return any(word in text for word in GOODBYE_WORDS)


class Conversation:
    """Loads STT, VAD, LLM and TTS once; turns audio into a spoken reply.

    Thread-safe: one turn at a time (the models are not built for parallel calls).
    """

    def __init__(self, settings: Settings, status: Callable[[str], None] = logger.info) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        status("Connecting to Gemini...")
        self.llm = GeminiChat.from_settings(settings)
        status(f"Loading speech recognition ({settings.stt_backend})...")
        self.stt: BaseSTT = create_stt(settings)
        self.vad = SileroVAD()
        status("Loading voice...")
        self.tts: BaseTTS = create_tts(settings)

    @property
    def gender(self) -> str:
        return self.settings.agent_gender

    @property
    def lines(self) -> dict[str, str]:
        return LINES[self.gender]

    def set_gender(self, gender: str) -> None:
        """Switch persona and matching voice; starts a new conversation."""
        if gender not in VOICES:
            raise ValueError(f"gender must be one of {list(VOICES)}")
        with self._lock:
            self.settings = dataclasses.replace(
                self.settings, agent_gender=gender, tts_voice=VOICES[gender]
            )
            self.tts = create_tts(self.settings)
            self.llm = GeminiChat.from_settings(self.settings)
            logger.info("Persona switched to %s (%s)", gender, VOICES[gender])

    def reset(self) -> None:
        """Forget the conversation so far."""
        with self._lock:
            self.llm.reset()

    def speak(self, text: str) -> Speech | None:
        """Synthesise `text`; None if no voice is available (the text is still shown)."""
        try:
            return self.tts.synthesize(text)
        except (TTSError, ValueError) as exc:
            logger.error("Could not speak the reply: %s", exc)
            return None

    def hear(self, audio: np.ndarray) -> Heard:
        """Transcribe one turn of 16 kHz audio, ignoring silence and noise."""
        with self._lock:
            speech_seconds = sum(s.end - s.start for s in self.vad.speech_segments(audio, RATE))
            speech_seconds /= RATE
            if speech_seconds < MIN_SPEECH_SECONDS:
                return Heard("", speech_seconds, 0.0)
            transcript = self.stt.transcribe(self.vad.trim(audio, RATE), RATE)
            return Heard(transcript.text, speech_seconds, transcript.elapsed_seconds)

    def respond(self, text: str, heard: Heard | None = None) -> Reply:
        """Answer `text` with Gemini and speak the answer."""
        timings = TurnTimings()
        if heard:
            timings.speech_seconds, timings.stt = heard.speech_seconds, heard.stt_seconds
        if is_goodbye(text):
            reply_text, goodbye = self.lines["goodbye"], True
        else:
            goodbye = False
            with self._lock:
                try:
                    answer = self.llm.reply(text)
                    reply_text, timings.llm = answer.text, answer.elapsed_seconds
                except LLMError as exc:
                    logger.error("%s", exc)
                    reply_text = self.lines["llm_failed"]
        speech = self.speak(reply_text)
        if speech:
            timings.tts, timings.tts_backend = speech.elapsed_seconds, speech.backend
        log_latency(timings, text)
        return Reply(reply_text, speech, timings, goodbye)


# --- console front end ---------------------------------------------------------


class AgentUI(Protocol):
    """What the console loop tells the person using it."""

    def status(self, message: str) -> None: ...
    def user_said(self, text: str) -> None: ...
    def agent_said(self, text: str) -> None: ...
    def timings(self, turn: TurnTimings) -> None: ...


class VoiceAgent:
    """Hands-free spoken conversation through the laptop mic and speaker."""

    def __init__(self, settings: Settings, ui: AgentUI) -> None:
        self.settings = settings
        self.ui = ui
        ui.status(f"Microphone: {audio_io.default_device_name('input')}")
        ui.status(f"Speaker:    {audio_io.default_device_name('output')}")
        self.conversation = Conversation(settings, ui.status)

    def say(self, text: str) -> None:
        self.ui.agent_said(text)
        speech = self.conversation.speak(text)
        if speech:
            audio_io.play(speech.audio, speech.sample_rate)

    def listen(self) -> np.ndarray | None:
        """Record one spoken turn from the mic. None if nobody spoke."""
        vad = self.conversation.vad
        vad.reset()
        self.ui.status("Listening... (speak now)")
        return audio_io.record_utterance(
            lambda chunk: vad.chunk_probability(chunk, RATE),
            sample_rate=RATE,
            chunk_samples=CHUNK_SAMPLES[RATE],
            end_silence_ms=self.settings.end_silence_ms,
            max_seconds=self.settings.max_turn_seconds,
            start_timeout_seconds=self.settings.start_timeout_seconds,
            on_speech_start=lambda: self.ui.status("Hearing you..."),
        )

    def run(self, push_to_talk: bool = False, wait_for_enter: Callable[[], None] = input) -> None:
        """Talk until the user says goodbye or presses Ctrl+C."""
        lines = self.conversation.lines
        self.say(lines["greeting"])
        while True:
            if push_to_talk:
                self.ui.status("Press Enter and speak.")
                wait_for_enter()
            audio = self.listen()
            if audio is None:
                continue
            self.ui.status("Understanding...")
            heard = self.conversation.hear(audio)
            if not heard.text:
                self.say(lines["not_heard"])
                continue
            self.ui.user_said(heard.text)
            self.ui.status("Thinking...")
            reply = self.conversation.respond(heard.text, heard)
            self.ui.agent_said(reply.text)
            if reply.speech:
                audio_io.play(reply.speech.audio, reply.speech.sample_rate)
            self.ui.timings(reply.timings)
            if reply.goodbye:
                return


def log_latency(t: TurnTimings, heard: str, path: Path = LATENCY_LOG) -> None:
    """Append one row per turn to logs/latency.csv (gitignored) for later benchmarks."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if new:
                writer.writerow(LATENCY_COLUMNS)
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    f"{t.speech_seconds:.2f}",
                    f"{t.stt:.2f}",
                    f"{t.llm:.2f}",
                    f"{t.tts:.2f}",
                    f"{t.response:.2f}",
                    t.tts_backend,
                    heard,
                ]
            )
    except OSError as exc:
        logger.warning("Could not write %s: %s", path, exc)
