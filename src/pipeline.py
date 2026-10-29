"""Voice loop: mic -> VAD -> STT -> LLM -> TTS -> speaker.

`VoiceAgent.run()` keeps a spoken conversation going: it listens until the user stops
talking, transcribes, asks Gemini, speaks the reply and listens again. Playback is
blocking, so the mic never records the agent's own voice.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from src import audio_io
from src.config import ROOT_DIR, Settings
from src.errors import LLMError, TTSError
from src.llm import GeminiChat
from src.stt import FasterWhisperSTT
from src.tts import create_tts
from src.vad import CHUNK_SAMPLES, SileroVAD

logger = logging.getLogger(__name__)

LATENCY_LOG = ROOT_DIR / "logs" / "latency.csv"
RATE = 16000
LATENCY_COLUMNS = [
    "time",
    "speech_s",
    "stt_s",
    "llm_s",
    "tts_s",
    "response_s",
    "tts",
    "heard",
    "reply",
]

# Saying any of these ends the conversation.
GOODBYE_WORDS = ("خدا حافظ", "اللہ حافظ", "بند کرو", "بند کریں", "بس کرو", "الوداع")

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


class AgentUI(Protocol):
    """What the loop tells the person using it (console, later Gradio)."""

    def status(self, message: str) -> None: ...
    def user_said(self, text: str) -> None: ...
    def agent_said(self, text: str) -> None: ...
    def timings(self, turn: TurnTimings) -> None: ...


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


def is_goodbye(text: str) -> bool:
    return any(word in text for word in GOODBYE_WORDS)


class VoiceAgent:
    """Spoken Urdu conversation with Gemini."""

    def __init__(self, settings: Settings, ui: AgentUI) -> None:
        self.settings = settings
        self.ui = ui
        self.lines = LINES[settings.agent_gender]
        self._last_backend = ""
        ui.status("Checking microphone and speaker...")
        ui.status(f"Microphone: {audio_io.default_device_name('input')}")
        ui.status(f"Speaker:    {audio_io.default_device_name('output')}")
        ui.status("Connecting to Gemini...")
        self.llm = GeminiChat.from_settings(settings)
        ui.status(f"Loading speech recognition (Whisper {settings.stt_model})...")
        self.stt = FasterWhisperSTT.from_settings(settings)
        self.vad = SileroVAD()
        ui.status("Loading voice...")
        self.tts = create_tts(settings)

    # --- one stage at a time -------------------------------------------------

    def say(self, text: str) -> float:
        """Speak `text`; returns synthesis time. Falls back to printing if TTS fails."""
        self.ui.agent_said(text)
        try:
            speech = self.tts.synthesize(text)
        except (TTSError, ValueError) as exc:
            logger.error("Could not speak the reply: %s", exc)
            return 0.0
        self._last_backend = speech.backend
        audio_io.play(speech.audio, speech.sample_rate)
        return speech.elapsed_seconds

    def listen(self) -> tuple[str, TurnTimings] | None:
        """Wait for one spoken turn and transcribe it. None if nobody spoke."""
        self.vad.reset()
        self.ui.status("Listening... (speak now)")
        audio = audio_io.record_utterance(
            lambda chunk: self.vad.chunk_probability(chunk, RATE),
            sample_rate=RATE,
            chunk_samples=CHUNK_SAMPLES[RATE],
            end_silence_ms=self.settings.end_silence_ms,
            max_seconds=self.settings.max_turn_seconds,
            start_timeout_seconds=self.settings.start_timeout_seconds,
            on_speech_start=lambda: self.ui.status("Hearing you..."),
        )
        if audio is None:
            return None
        timings = TurnTimings(speech_seconds=len(audio) / RATE)
        self.ui.status("Understanding...")
        transcript = self.stt.transcribe(self.vad.trim(audio, RATE), RATE)
        timings.stt = transcript.elapsed_seconds
        return transcript.text, timings

    # --- the loop -------------------------------------------------------------

    def run(self, push_to_talk: bool = False, wait_for_enter: Callable[[], None] = input) -> None:
        """Talk until the user says goodbye or presses Ctrl+C."""
        self.say(self.lines["greeting"])
        while True:
            if push_to_talk:
                self.ui.status("Press Enter and speak.")
                wait_for_enter()
            heard = self.listen()
            if heard is None:
                if not push_to_talk:
                    self.ui.status("(no speech; still listening, Ctrl+C to quit)")
                continue
            text, timings = heard
            if not text:
                self.say(self.lines["not_heard"])
                continue
            self.ui.user_said(text)
            if is_goodbye(text):
                self.say(self.lines["goodbye"])
                return

            self.ui.status("Thinking...")
            try:
                reply = self.llm.reply(text)
                timings.llm = reply.elapsed_seconds
                reply_text = reply.text
            except LLMError as exc:
                logger.error("%s", exc)
                reply_text = self.lines["llm_failed"]
            timings.tts = self.say(reply_text)
            timings.tts_backend = self._last_backend
            self.ui.timings(timings)
            _log_latency(timings, text, reply_text)


def _log_latency(t: TurnTimings, heard: str, reply: str, path: Path = LATENCY_LOG) -> None:
    """Append one row per turn to logs/latency.csv (gitignored) for later benchmarks."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if new:
                writer.writerow(
                    [
                        "time",
                        "speech_s",
                        "stt_s",
                        "llm_s",
                        "tts_s",
                        "response_s",
                        "tts",
                        "heard",
                        "reply",
                    ]
                )
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
                    reply,
                ]
            )
    except OSError as exc:
        logger.warning("Could not write %s: %s", path, exc)
