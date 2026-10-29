"""Voice pipeline: audio -> VAD -> STT -> LLM -> TTS -> audio.

- `Conversation` is the core: it loads the models once and turns speech into a spoken
  reply. Both front ends use it.
- `VoiceAgent` is the console front end (laptop mic and speaker, hands-free loop).
- `src.server` is the browser front end (mic button, chat bubbles). Every visitor gets a
  `ChatSession` (own Gemini history and persona); the models are shared.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src import audio_io
from src.config import ROOT_DIR, Settings
from src.errors import LLMError, TTSError
from src.llm import GeminiChat
from src.stt import BaseSTT, create_stt
from src.tts import BaseTTS, EncodedSpeech, Speech, create_tts
from src.vad import CHUNK_SAMPLES, SileroVAD

logger = logging.getLogger(__name__)

LATENCY_LOG = ROOT_DIR / "logs" / "latency.csv"
RATE = 16000
MIN_SPEECH_SECONDS = 0.3
LATENCY_COLUMNS = [
    "time",
    "speech_s",
    "stt_s",
    "llm_first_s",
    "llm_s",
    "tts_first_s",
    "first_audio_s",
    "total_s",
    "tts",
    "heard",
]
LOCAL_SESSION = "local"  # the console app and anyone without a session id
# Sentences are voiced in parallel. Two workers stay ahead of playback; more made the
# first sentence slower on a slow line (requests compete) and caused timeouts.
TTS_WORKERS = 2

# Saying any of these ends the conversation.
GOODBYE_WORDS = ("خدا حافظ", "اللہ حافظ", "بند کرو", "بند کریں", "بس کرو", "الوداع")

# Voices and the grammatical gender that goes with each.
VOICES = {"female": "ur-PK-UzmaNeural", "male": "ur-PK-AsadNeural"}

# Fixed lines the agent says itself, per grammatical gender.
LINES = {
    "female": {
        "not_heard": "معاف کیجیے، آواز صاف نہیں آئی۔ ذرا دوبارہ بولیں۔",
        "llm_failed": "معاف کیجیے، ابھی جواب نہیں دے سکی۔ تھوڑی دیر بعد دوبارہ کوشش کریں۔",
        "goodbye": "ٹھیک ہے، اللہ حافظ! آپ سے بات کر کے اچھا لگا۔",
    },
    "male": {
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
    # Streaming turns only (0 for the non-streaming path):
    llm_first: float = 0.0  # Gemini call -> first text chunk
    tts_first: float = 0.0  # first sentence complete -> its audio ready
    first_audio: float = 0.0  # end of user speech -> first reply audio ready
    total: float = 0.0  # end of user speech -> last reply audio ready

    @property
    def response(self) -> float:
        """From the end of the user's speech to the start of the reply audio."""
        if self.first_audio:
            return self.first_audio
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


_END_MARKS = "۔؟!?\n"
_COMMAS = "،,"


class SentenceStreamer:
    """Cuts streamed text into speakable sentences as soon as each one is complete.

    A sentence ends at ۔ ؟ ! ? or a newline, or at "." unless it sits between digits
    (3.5). A long run without an end mark is cut at a comma once it has
    `max_words` words, so the voice can start before the sentence ends; the first
    cut uses the lower `first_max_words` to get the first audio out sooner.
    """

    def __init__(self, max_words: int = 12, first_max_words: int = 6) -> None:
        self.max_words = max_words
        self.first_max_words = first_max_words
        self._buffer = ""
        self._emitted = 0

    def feed(self, chunk: str) -> list[str]:
        """Add streamed text; return the sentences completed by it."""
        self._buffer += chunk
        out: list[str] = []
        while (cut := self._find_cut()) is not None:
            sentence, self._buffer = self._buffer[:cut].strip(), self._buffer[cut:]
            if sentence.strip(_END_MARKS + _COMMAS + ". "):
                out.append(sentence)
                self._emitted += 1
        return out

    def flush(self) -> list[str]:
        """Return whatever is left at the end of the stream."""
        rest, self._buffer = self._buffer.strip(), ""
        return [rest] if rest.strip(_END_MARKS + _COMMAS + ". ") else []

    def _find_cut(self) -> int | None:
        text = self._buffer
        for i, ch in enumerate(text):
            if ch in _END_MARKS:
                return i + 1
            if ch == ".":
                if i + 1 == len(text):
                    return None  # might be a decimal point; wait for the next chunk
                if not (i > 0 and text[i - 1].isdigit() and text[i + 1].isdigit()):
                    return i + 1
        limit = self.first_max_words if self._emitted == 0 else self.max_words
        words = 0
        for i, ch in enumerate(text):
            if ch.isspace() and i > 0 and not text[i - 1].isspace():
                words += 1
            if ch in _COMMAS and words + 1 >= limit:
                return i + 1
        return None


@dataclass
class ChatSession:
    """One visitor's conversation: Gemini history and persona. Models are shared."""

    llm: GeminiChat
    gender: str
    lock: threading.Lock = field(default_factory=threading.Lock)  # one turn at a time
    last_used: float = field(default_factory=time.monotonic)

    @property
    def lines(self) -> dict[str, str]:
        return LINES[self.gender]


class SessionStore:
    """In-memory sessions with an idle timeout and a size cap (least recently used goes).

    Nothing is persisted: a server restart starts every conversation afresh.
    """

    def __init__(
        self,
        factory: Callable[[], ChatSession],
        ttl_seconds: float = 1800,
        max_sessions: int = 200,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._factory = factory
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max(1, max_sessions)
        self._clock = clock
        self._sessions: OrderedDict[str, ChatSession] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._sessions)

    def _evict(self, now: float) -> None:
        if self.ttl_seconds > 0:
            expired = [k for k, v in self._sessions.items() if now - v.last_used > self.ttl_seconds]
            for key in expired:
                del self._sessions[key]
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

    def get(self, key: str) -> ChatSession:
        """The session for `key`, created on first use."""
        with self._lock:
            now = self._clock()
            self._evict(now)
            session = self._sessions.get(key)
            if session is None:
                session = self._factory()
                self._sessions[key] = session
            self._sessions.move_to_end(key)
            session.last_used = now
            self._evict(now)
            return session

    def peek(self, key: str) -> ChatSession | None:
        """The session for `key` if it exists (does not create or refresh it)."""
        with self._lock:
            self._evict(self._clock())
            return self._sessions.get(key)


class Conversation:
    """Loads STT, VAD, LLM and TTS once; turns audio into a spoken reply.

    Thread-safe. Speech recognition runs one turn at a time; each session's Gemini
    calls are serialised by that session's lock, so visitors don't wait for each other.
    """

    def __init__(
        self,
        settings: Settings,
        status: Callable[[str], None] = logger.info,
        session_ttl_seconds: float = 1800,
        max_sessions: int = 200,
    ) -> None:
        self.settings = settings
        self._lock = threading.Lock()  # the local (console) session's Gemini calls
        self._stt_lock = threading.Lock()
        self._voices_lock = threading.Lock()
        self._latency_log = getattr(settings, "latency_log", True)
        self._log_text = getattr(settings, "log_transcripts", True)
        status("Connecting to Gemini...")
        self.llm = GeminiChat.from_settings(settings)
        status(f"Loading speech recognition ({settings.stt_backend})...")
        self.stt: BaseSTT = create_stt(settings)
        # The browser already detects speech; Silero on the server is optional (cloud: off).
        self.vad: SileroVAD | None = SileroVAD() if getattr(settings, "server_vad", True) else None
        status("Loading voice...")
        self.tts: BaseTTS = create_tts(settings)
        self._voices: dict[str, BaseTTS] = {settings.tts_voice: self.tts}
        default_gender = settings.agent_gender
        self._sessions = SessionStore(
            lambda: ChatSession(
                GeminiChat.from_settings(settings, default_gender, client=self.llm.client),
                default_gender,
            ),
            session_ttl_seconds,
            max_sessions,
        )

    # --- sessions ----------------------------------------------------------------

    def session(self, key: str | None = None) -> ChatSession:
        """The visitor's session; None or LOCAL_SESSION is the local/console session."""
        store = getattr(self, "_sessions", None)
        if key is None or key == LOCAL_SESSION or store is None:
            return ChatSession(self.llm, self.gender, self._lock)
        return store.get(key)

    def peek_session(self, key: str | None) -> ChatSession | None:
        """An existing visitor session, without creating one."""
        store = getattr(self, "_sessions", None)
        if key is None or key == LOCAL_SESSION or store is None:
            return None
        return store.peek(key)

    def tts_for(self, gender: str) -> BaseTTS:
        """The voice for `gender` (Uzma or Asad); built once, then shared."""
        voice = VOICES.get(gender)
        voices = getattr(self, "_voices", None)
        if voices is None or voice is None or not hasattr(self.tts, "with_voice"):
            return self.tts
        with self._voices_lock:
            if voice not in voices:
                voices[voice] = self.tts.with_voice(voice)
            return voices[voice]

    def _log(self, timings: TurnTimings, text: str) -> None:
        if getattr(self, "_latency_log", True):
            log_latency(timings, text if getattr(self, "_log_text", True) else "")

    def _redact(self, text: str) -> str:
        """What users said is only logged when LOG_TRANSCRIPTS is on (off in cloud)."""
        return text if getattr(self, "_log_text", True) else f"<{len(text)} chars>"

    @property
    def gender(self) -> str:
        return self.settings.agent_gender

    @property
    def lines(self) -> dict[str, str]:
        return LINES[self.gender]

    def set_gender(self, gender: str, session: ChatSession | None = None) -> None:
        """Switch persona and matching voice; starts a new conversation.

        With a visitor `session` only that visitor changes; without one, the local
        (console) default changes.
        """
        if gender not in VOICES:
            raise ValueError(f"gender must be one of {list(VOICES)}")
        client = getattr(self.llm, "client", None)
        if session is not None:
            with session.lock:
                session.llm = GeminiChat.from_settings(self.settings, gender, client=client)
                session.gender = gender
            return
        with self._lock:
            voice = self.tts_for(gender)
            self.settings = dataclasses.replace(
                self.settings, agent_gender=gender, tts_voice=VOICES[gender]
            )
            self.tts = voice
            self.llm = GeminiChat.from_settings(self.settings, client=client)
            logger.info("Persona switched to %s (%s)", gender, VOICES[gender])

    def reset(self, session: ChatSession | None = None) -> None:
        """Forget the conversation so far (one visitor's, or the local one)."""
        s = session or self.session()
        with s.lock:
            s.llm.reset()

    def warm_up(self, timeout: float = 10.0) -> dict[str, float | str]:
        """Make the first real turn as fast as later ones.

        In parallel: open Gemini's HTTPS connection (free metadata call), run STT once
        on a short tone (loads Whisper kernels / opens the Gemini STT connection), and
        fetch one tiny sentence from the voice. Returns seconds per part or the error.
        Never raises; waits at most `timeout` seconds.
        """
        tone = (0.05 * np.sin(np.linspace(0, 2 * np.pi * 220, RATE))).astype(np.float32)
        parts: dict[str, Callable[[], object]] = {
            "llm": self.llm.warm_up,
            # A backend's own warm_up() wins: transcribing a tone with an online STT
            # would spend one of the ~20 free daily requests.
            "stt": getattr(self.stt, "warm_up", None) or (lambda: self.stt.transcribe(tone, RATE)),
            "tts": lambda: self.tts.synthesize_encoded("جی۔"),
        }
        preload = getattr(self.tts, "preload_fallback", None)
        if preload is not None:  # offline voice ready before the online one ever fails
            parts["tts_fallback"] = preload
        results: dict[str, float | str] = {}

        def run(name: str, fn: Callable[[], object]) -> None:
            started = time.perf_counter()
            try:
                fn()
                results[name] = round(time.perf_counter() - started, 2)
            except Exception as exc:  # warm-up is best effort
                results[name] = f"skipped: {exc}"

        threads = [threading.Thread(target=run, args=item, daemon=True) for item in parts.items()]
        for t in threads:
            t.start()
        deadline = time.monotonic() + timeout
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))
        logger.info("Warm-up: %s", results)
        return results

    def fixed_line_audio(self, gender: str, key: str) -> EncodedSpeech | None:
        """Audio for one of the agent's fixed LINES, synthesised once per voice and cached
        (e.g. "I didn't hear you" would otherwise cost a TTS round trip every time)."""
        cache: dict[tuple[str, str], EncodedSpeech] = self.__dict__.setdefault("_line_cache", {})
        if (gender, key) not in cache:
            try:
                cache[(gender, key)] = self.tts_for(gender).synthesize_encoded(LINES[gender][key])
            except (TTSError, ValueError) as exc:
                logger.error("Could not speak the %s line: %s", key, exc)
                return None
        return cache[(gender, key)]

    def speak(self, text: str, gender: str | None = None) -> Speech | None:
        """Synthesise `text`; None if no voice is available (the text is still shown)."""
        tts = self.tts_for(gender) if gender else self.tts
        try:
            return tts.synthesize(text)
        except (TTSError, ValueError) as exc:
            logger.error("Could not speak the reply: %s", exc)
            return None

    def hear(self, audio: np.ndarray) -> Heard:
        """Transcribe one turn of 16 kHz audio, ignoring silence and noise.

        Without a server VAD (cloud) the browser has already cut the turn to speech;
        STT itself returns "" for silence.
        """
        vad = getattr(self, "vad", None)
        with getattr(self, "_stt_lock", self._lock):
            if vad is not None:
                speech_seconds = sum(s.end - s.start for s in vad.speech_segments(audio, RATE))
                speech_seconds /= RATE
            else:
                speech_seconds = len(audio) / RATE
            if speech_seconds < MIN_SPEECH_SECONDS:
                return Heard("", speech_seconds, 0.0)
            clip = vad.trim(audio, RATE) if vad is not None else audio
            # Wall time, not the backend's own figure: it includes a failed Gemini call
            # and the switch to the fallback, which the user also waits for.
            started = time.perf_counter()
            transcript = self.stt.transcribe(clip, RATE)
            return Heard(transcript.text, speech_seconds, time.perf_counter() - started)

    def respond(
        self, text: str, heard: Heard | None = None, session: ChatSession | None = None
    ) -> Reply:
        """Answer `text` with Gemini and speak the answer."""
        s = session or self.session()
        timings = TurnTimings()
        if heard:
            timings.speech_seconds, timings.stt = heard.speech_seconds, heard.stt_seconds
        if is_goodbye(text):
            reply_text, goodbye = s.lines["goodbye"], True
        else:
            goodbye = False
            with s.lock:
                try:
                    answer = s.llm.reply(text)
                    reply_text, timings.llm = answer.text, answer.elapsed_seconds
                except LLMError as exc:
                    logger.error("%s", exc)
                    reply_text = s.lines["llm_failed"]
        speech = self.speak(reply_text, s.gender)
        if speech:
            timings.tts, timings.tts_backend = speech.elapsed_seconds, speech.backend
        self._log(timings, text)
        return Reply(reply_text, speech, timings, goodbye)

    def respond_stream(
        self,
        text: str,
        heard: Heard | None = None,
        cancel: threading.Event | None = None,
        session: ChatSession | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Answer `text`, yielding events while Gemini is still writing.

        Gemini's reply is cut into sentences as it streams; each sentence is voiced
        in a worker thread as soon as it is complete, so the first sentence can play
        while the rest is still being written. Events (dicts), in order:
            {"type": "reply_text", "index", "text"}
            {"type": "audio", "index", "data": bytes, "mime"}   (index order; a
                sentence with nothing speakable gets no audio event)
            {"type": "done", "reply", "goodbye", "voice_backend", "timings": TurnTimings}
        Setting `cancel` (or closing the generator) stops Gemini and the voice early.
        """
        cancel = cancel or threading.Event()
        s = session or self.session()
        tts = self.tts_for(s.gender)
        started = time.perf_counter()
        timings = TurnTimings()
        if heard:
            timings.speech_seconds, timings.stt = heard.speech_seconds, heard.stt_seconds
        goodbye = is_goodbye(text)
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        jobs: queue.Queue[tuple[int, Future[EncodedSpeech | None], float] | None] = queue.Queue()
        sentences: list[str] = []
        backends: list[str] = []
        pool = ThreadPoolExecutor(TTS_WORKERS, thread_name_prefix="tts")

        def voice(sentence: str) -> EncodedSpeech | None:
            if cancel.is_set():
                return None
            try:
                return tts.synthesize_encoded(sentence)
            except (TTSError, ValueError) as exc:
                logger.error("Could not speak %r: %s", self._redact(sentence), exc)
                return None

        def emit(sentence: str) -> None:
            index = len(sentences)
            sentences.append(sentence)
            events.put({"type": "reply_text", "index": index, "text": sentence})
            jobs.put((index, pool.submit(voice, sentence), time.perf_counter()))

        def emit_all(line: str) -> None:
            streamer = SentenceStreamer()
            for sentence in [*streamer.feed(line), *streamer.flush()]:
                emit(sentence)

        def produce() -> None:
            try:
                if goodbye:
                    emit_all(s.lines["goodbye"])
                    return
                streamer = SentenceStreamer()
                failed = False
                with s.lock:  # this visitor's Gemini calls are serialised; voicing is not
                    t0 = time.perf_counter()
                    try:
                        for chunk in s.llm.reply_stream(text):
                            if not timings.llm_first:
                                timings.llm_first = time.perf_counter() - t0
                            if cancel.is_set():
                                break
                            for sentence in streamer.feed(chunk):
                                emit(sentence)
                    except LLMError as exc:
                        logger.error("%s", exc)
                        failed = True
                    timings.llm = time.perf_counter() - t0
                for sentence in streamer.flush():
                    emit(sentence)
                if failed and not sentences:
                    emit_all(s.lines["llm_failed"])
            except Exception:  # never leave the consumer waiting forever
                logger.exception("Reply stream failed")
                if not sentences:
                    emit_all(s.lines["llm_failed"])
            finally:
                jobs.put(None)

        def deliver() -> None:
            try:
                while (job := jobs.get()) is not None:
                    index, future, submitted = job
                    try:
                        encoded = future.result()
                    except CancelledError:  # the turn was aborted and the pool shut down
                        continue
                    if encoded is None or cancel.is_set():
                        continue
                    if not timings.first_audio:
                        now = time.perf_counter()
                        timings.tts_first = now - submitted
                        timings.first_audio = timings.stt + (now - started)
                    backends.append(encoded.backend)
                    timings.tts += encoded.elapsed_seconds
                    events.put(
                        {
                            "type": "audio",
                            "index": index,
                            "data": encoded.data,
                            "mime": encoded.mime,
                        }
                    )
            finally:
                timings.total = timings.stt + (time.perf_counter() - started)
                events.put(None)

        threading.Thread(target=produce, name="reply-producer", daemon=True).start()
        threading.Thread(target=deliver, name="reply-audio", daemon=True).start()
        try:
            while (event := events.get()) is not None:
                yield event
            timings.tts_backend = "mms" if "mms" in backends else (backends[0] if backends else "")
            reply_text = " ".join(sentences)
            self._log(timings, text)
            yield {
                "type": "done",
                "reply": reply_text,
                "goodbye": goodbye,
                "voice_backend": timings.tts_backend,
                "timings": timings,
            }
        finally:
            cancel.set()
            pool.shutdown(wait=False, cancel_futures=True)


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
        self.conversation.warm_up()
        while True:  # no spoken greeting: the user speaks first
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
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                header = fh.readline().strip()
            if header != ",".join(LATENCY_COLUMNS):  # old layout: keep it, start a new file
                path.rename(path.with_name(f"latency_{datetime.now():%Y%m%d_%H%M%S}.csv"))
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
                    f"{t.llm_first:.2f}",
                    f"{t.llm:.2f}",
                    f"{t.tts_first:.2f}",
                    f"{t.response:.2f}",
                    f"{t.total or t.response:.2f}",
                    t.tts_backend,
                    heard,
                ]
            )
    except OSError as exc:
        logger.warning("Could not write %s: %s", path, exc)
