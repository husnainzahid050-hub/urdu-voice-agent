"""Streaming reply pipeline and NDJSON endpoints, with fake LLM/TTS (no network, no models)."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src import pipeline, server
from src.errors import LLMError, TTSError
from src.pipeline import Conversation, Heard, SentenceStreamer
from src.tts import EncodedSpeech

# --- incremental sentence splitter ---------------------------------------------------


def _feed_all(chunks: list[str], **kw: int) -> list[str]:
    s = SentenceStreamer(**kw)
    out: list[str] = []
    for c in chunks:
        out += s.feed(c)
    return out + s.flush()


def test_splits_on_urdu_marks_across_chunks() -> None:
    chunks = ["جی ہاں، آپ کا آر", "ڈر تیار ہے۔ کیا آپ", " گھر پر ہیں؟ شکریہ"]
    assert _feed_all(chunks) == ["جی ہاں، آپ کا آرڈر تیار ہے۔", "کیا آپ گھر پر ہیں؟", "شکریہ"]


def test_sentence_is_emitted_as_soon_as_it_ends() -> None:
    s = SentenceStreamer()
    assert s.feed("پہلا جملہ۔") == ["پہلا جملہ۔"]
    assert s.feed(" دوسرا") == []
    assert s.flush() == ["دوسرا"]


def test_decimal_point_is_not_a_sentence_end() -> None:
    s = SentenceStreamer()
    assert s.feed("قیمت 3.") == []  # could be a decimal: wait
    assert s.feed("5 روپے ہے. اگلا") == ["قیمت 3.5 روپے ہے."]
    assert s.flush() == ["اگلا"]


def test_long_run_is_cut_at_a_comma() -> None:
    first = "ایک دو تین چار پانچ چھ، سات آٹھ"
    s = SentenceStreamer(max_words=12, first_max_words=6)
    assert s.feed(first) == ["ایک دو تین چار پانچ چھ،"]
    words = " ".join(["لفظ"] * 12)
    assert s.feed(f" {words}، باقی") == ["سات آٹھ " + words + "،"]


def test_short_clause_is_not_cut_at_a_comma() -> None:
    assert SentenceStreamer().feed("جی ہاں، ٹھیک") == []


def test_newline_ends_a_sentence_and_empty_bits_are_dropped() -> None:
    assert _feed_all(["پہلا\n\n۔ دوسرا۔"]) == ["پہلا", "دوسرا۔"]


# --- respond_stream with fakes ---------------------------------------------------------


class FakeLLM:
    def __init__(self, chunks: list[str] | None = None, error_after: int | None = None):
        self.chunks = chunks or []
        self.error_after = error_after
        self.calls = 0

    def reply_stream(self, text: str) -> Iterator[str]:
        self.calls += 1
        for i, chunk in enumerate(self.chunks):
            if self.error_after is not None and i == self.error_after:
                raise LLMError("Gemini is busy")
            time.sleep(0.01)
            yield chunk
        if self.error_after is not None and self.error_after >= len(self.chunks):
            raise LLMError("Gemini is busy")

    def reset(self) -> None:
        pass


class FakeTTS:
    """Voices each sentence; the first one slowly, to prove ordering is kept."""

    def __init__(self, fail: set[str] | None = None):
        self.fail = fail or set()
        self.spoken: list[str] = []
        self._lock = threading.Lock()

    def synthesize_encoded(self, text: str) -> EncodedSpeech:
        with self._lock:
            first = not self.spoken
            self.spoken.append(text)
        time.sleep(0.05 if first else 0.0)
        if text in self.fail:
            raise TTSError("offline")
        return EncodedSpeech(text.encode("utf-8"), "audio/mpeg", "edge", 0.01)


def _conversation(llm: FakeLLM, tts: FakeTTS) -> Conversation:
    conv = Conversation.__new__(Conversation)
    conv.settings = SimpleNamespace(agent_gender="female")
    conv._lock = threading.Lock()
    conv.llm = llm
    conv.tts = tts
    return conv


@pytest.fixture(autouse=True)
def _no_latency_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "log_latency", lambda *a, **k: None)


def test_events_come_in_order_with_audio_per_sentence() -> None:
    llm = FakeLLM(["جی ہاں۔ آپ کا", " آرڈر تیار ہے۔ شکریہ۔"])
    events = list(_conversation(llm, FakeTTS()).respond_stream("سلام", Heard("سلام", 1.0, 2.0)))
    kinds = [(e["type"], e.get("index")) for e in events]
    texts = [e["text"] for e in events if e["type"] == "reply_text"]
    assert texts == ["جی ہاں۔", "آپ کا آرڈر تیار ہے۔", "شکریہ۔"]
    audio_order = [i for t, i in kinds if t == "audio"]
    assert audio_order == [0, 1, 2]
    for i in range(3):  # each sentence's text arrives before its audio
        assert kinds.index(("reply_text", i)) < kinds.index(("audio", i))
    done = events[-1]
    assert done["type"] == "done" and done["reply"] == " ".join(texts)
    assert not done["goodbye"] and done["voice_backend"] == "edge"
    t = done["timings"]
    assert t.stt == 2.0 and t.first_audio >= 2.0 and t.total >= t.first_audio
    assert t.llm_first > 0 and t.llm >= t.llm_first


def test_goodbye_skips_gemini() -> None:
    llm = FakeLLM(["should not be used۔"])
    events = list(_conversation(llm, FakeTTS()).respond_stream("ٹھیک ہے، اللہ حافظ"))
    assert llm.calls == 0
    assert events[-1]["goodbye"] is True
    assert "اللہ حافظ" in events[-1]["reply"]


def test_gemini_failure_speaks_the_apology() -> None:
    events = list(_conversation(FakeLLM([], error_after=0), FakeTTS()).respond_stream("سلام"))
    assert "معاف کیجیے" in events[-1]["reply"]
    assert any(e["type"] == "audio" for e in events)


def test_gemini_failure_mid_reply_keeps_what_was_said() -> None:
    llm = FakeLLM(["پہلا جملہ۔ ادھورا"], error_after=1)
    events = list(_conversation(llm, FakeTTS()).respond_stream("سلام"))
    assert events[-1]["reply"] == "پہلا جملہ۔ ادھورا"


def test_sentence_without_voice_is_skipped_not_fatal() -> None:
    tts = FakeTTS(fail={"دوسرا۔"})
    events = list(_conversation(FakeLLM(["پہلا۔ دوسرا۔ تیسرا۔"]), tts).respond_stream("سلام"))
    assert [e["index"] for e in events if e["type"] == "audio"] == [0, 2]
    assert events[-1]["type"] == "done"


# --- NDJSON endpoints ------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, Conversation]]:
    conv = _conversation(FakeLLM(["جی ہاں۔ ٹھیک ہے۔"]), FakeTTS())
    conv.hear = lambda samples: Heard("میرا آرڈر", 1.2, 0.8)  # type: ignore[method-assign]
    monkeypatch.setattr(server, "_load_models", lambda: None)
    monkeypatch.setattr(server.webbrowser, "open", lambda *a, **k: None)
    monkeypatch.setitem(server._state, "conversation", conv)
    monkeypatch.setattr(server, "_decode", lambda data: [0.0] * 16000)
    with TestClient(server.app) as c:
        yield c, conv


def _lines(response) -> list[dict]:  # noqa: ANN001
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_turn_streams_heard_then_reply(client) -> None:  # noqa: ANN001
    c, _ = client
    r = c.post("/api/turn", files={"audio": ("t.webm", b"x", "audio/webm")})
    assert r.headers["content-type"].startswith("application/x-ndjson")
    events = _lines(r)
    assert events[0] == {
        "type": "heard",
        "text": "میرا آرڈر",
        "stt_seconds": 0.8,
        "speech_seconds": 1.2,
    }
    kinds = [(e["type"], e.get("index")) for e in events[1:]]
    assert sorted(kinds[:-1], key=str) == sorted(
        [("reply_text", 0), ("audio", 0), ("reply_text", 1), ("audio", 1)], key=str
    )
    assert kinds[0] == ("reply_text", 0) and kinds[-1] == ("done", None)
    audio = next(e for e in events if e["type"] == "audio")
    assert audio["mime"] == "audio/mpeg" and audio["index"] == 0 and isinstance(audio["audio"], str)
    assert set(events[-1]["timings"]) == {
        "stt",
        "llm_first",
        "llm",
        "tts_first",
        "first_audio",
        "total",
    }


def test_turn_with_nothing_heard(client) -> None:  # noqa: ANN001
    c, conv = client
    conv.hear = lambda samples: Heard("", 0.1, 0.0)  # type: ignore[method-assign]
    events = _lines(c.post("/api/turn", files={"audio": ("t.webm", b"x", "audio/webm")}))
    assert [e["type"] for e in events] == ["heard", "not_heard", "done"]
    assert events[1]["audio"] and events[1]["mime"] == "audio/mpeg"


def test_chat_streams_without_heard(client) -> None:  # noqa: ANN001
    c, _ = client
    events = _lines(c.post("/api/chat", json={"text": "سلام"}))
    assert events[0]["type"] == "reply_text" and events[-1]["type"] == "done"
    assert c.post("/api/chat", json={"text": "  "}).status_code == 400


def test_unexpected_failure_becomes_error_event(client) -> None:  # noqa: ANN001
    c, conv = client

    def boom(samples):  # noqa: ANN001, ANN202
        raise RuntimeError("mic data broken")

    conv.hear = boom  # type: ignore[method-assign]
    events = _lines(c.post("/api/turn", files={"audio": ("t.webm", b"x", "audio/webm")}))
    assert events == [{"type": "error", "message": "mic data broken"}]


def test_status_reports_stt_backend(client) -> None:  # noqa: ANN001
    c, conv = client
    conv.settings = SimpleNamespace(
        agent_gender="female", tts_voice="v", stt_backend="whisper", stt_model="m", gemini_model="g"
    )
    body = c.get("/api/status").json()
    assert body["ready"] and body["stt_backend"] == "whisper"


def test_reset_endpoint(client) -> None:  # noqa: ANN001
    c, conv = client
    calls = []
    conv.llm.reset = lambda: calls.append(1)  # type: ignore[method-assign]
    assert c.post("/api/reset").json() == {"ok": True}
    assert calls == [1]


def test_warm_up_never_raises() -> None:
    class Broken:
        def warm_up(self) -> None:
            raise RuntimeError("offline")

        def transcribe(self, audio, rate):  # noqa: ANN001, ANN201
            raise RuntimeError("offline")

        def synthesize_encoded(self, text):  # noqa: ANN001, ANN201
            raise RuntimeError("offline")

    conv = _conversation(Broken(), Broken())  # type: ignore[arg-type]
    conv.stt = Broken()
    results = conv.warm_up(timeout=2.0)
    assert set(results) == {"llm", "stt", "tts"}
    assert all(str(v).startswith("skipped") for v in results.values())


def test_prompt_asks_for_direct_answers_and_no_greeting() -> None:
    from src.llm import build_system_prompt

    prompt = build_system_prompt("female")
    assert "FIRST sentence" in prompt and "Do not greet" in prompt
    assert "speech recognition" in prompt


def test_console_loop_does_not_greet(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = pipeline.VoiceAgent.__new__(pipeline.VoiceAgent)
    conv = _conversation(FakeLLM(["جی۔"]), FakeTTS())
    conv.warm_up = lambda timeout=10.0: {}  # type: ignore[method-assign]
    conv.hear = lambda audio: Heard("اللہ حافظ", 1.0, 0.5)  # type: ignore[method-assign]
    conv.respond = lambda text, heard=None: pipeline.Reply(  # type: ignore[method-assign]
        "اللہ حافظ", None, pipeline.TurnTimings(), goodbye=True
    )
    agent.conversation, agent.settings = conv, SimpleNamespace()
    said: list[str] = []
    agent.ui = SimpleNamespace(
        status=lambda m: None,
        user_said=lambda t: None,
        agent_said=said.append,
        timings=lambda t: None,
    )
    agent.listen = lambda: [0.0]  # type: ignore[method-assign]
    agent.run()
    assert said == ["اللہ حافظ"]  # only the goodbye reply, no greeting first


# --- quota fallback, cancellation, limits ---------------------------------------------


def _gemini_error(code: int, message: str):  # noqa: ANN202
    from google.genai import errors as genai_errors

    return genai_errors.APIError(code, {"error": {"message": message, "status": "X"}})


class _StreamChat:
    def __init__(self, plan: dict[str, list]) -> None:
        self.plan, self.models = plan, []

    def make(self, model: str):  # noqa: ANN201
        chat = SimpleNamespace(get_history=lambda: ["history"])

        def send_message_stream(text):  # noqa: ANN001, ANN202
            self.models.append(model)
            step = self.plan[model].pop(0)
            if isinstance(step, Exception):
                raise step
            for piece in step:
                yield SimpleNamespace(text=piece, candidates=[])

        chat.send_message_stream = send_message_stream
        return chat


def _gemini_with(plan: dict[str, list], fallbacks: list[str], monkeypatch):  # noqa: ANN001, ANN202
    from src import llm

    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    sleeps: list[float] = []
    fake = _StreamChat(plan)
    chat = llm.GeminiChat.__new__(llm.GeminiChat)
    chat.model, chat._fallback_models, chat._config = "main", list(fallbacks), None
    chat._client = SimpleNamespace(
        chats=SimpleNamespace(create=lambda model, config, history=None: fake.make(model))
    )
    chat._chat = fake.make("main")
    return chat, fake, sleeps


def test_daily_quota_switches_model_without_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    daily = _gemini_error(429, "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    chat, fake, sleeps = _gemini_with({"main": [daily], "lite": [["جی۔"]]}, ["lite"], monkeypatch)
    assert "".join(chat.reply_stream("سلام")) == "جی۔"
    assert fake.models == ["main", "lite"] and chat.model == "lite" and sleeps == []


def test_short_rate_limit_is_retried_on_same_model(monkeypatch: pytest.MonkeyPatch) -> None:
    burst = _gemini_error(429, "Resource exhausted, retry shortly")
    chat, fake, sleeps = _gemini_with({"main": [burst, ["ٹھیک۔"]]}, ["lite"], monkeypatch)
    assert "".join(chat.reply_stream("سلام")) == "ٹھیک۔"
    assert fake.models == ["main", "main"] and len(sleeps) == 1


def test_daily_quota_without_fallback_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.errors import RateLimitError

    daily = _gemini_error(429, "requests per day limit exceeded")
    chat, fake, sleeps = _gemini_with({"main": [daily]}, [], monkeypatch)
    with pytest.raises(RateLimitError):
        list(chat.reply_stream("سلام"))
    assert sleeps == []


def test_cancel_stops_gemini_and_voice() -> None:
    llm = FakeLLM(["ایک۔", " دو۔", " تین۔", " چار۔"])
    tts = FakeTTS()
    cancel = threading.Event()
    stream = _conversation(llm, tts).respond_stream("سلام", cancel=cancel)
    first = next(stream)
    assert first["type"] == "reply_text"
    cancel.set()
    rest = list(stream)
    spoken_texts = [e["text"] for e in rest if e["type"] == "reply_text"]
    assert len(spoken_texts) < 3  # stopped early instead of finishing the reply


def test_limits_and_healthz(client) -> None:  # noqa: ANN001
    c, _ = client
    assert c.get("/healthz").json() == {"ok": True, "ready": True}
    assert c.post("/api/chat", json={"text": "ا" * 501}).status_code == 413
    big = b"x" * (3 * 1024 * 1024 + 1)
    assert c.post("/api/turn", files={"audio": ("t.webm", big, "audio/webm")}).status_code == 413
