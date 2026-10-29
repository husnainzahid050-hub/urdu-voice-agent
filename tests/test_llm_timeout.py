"""A stalled Gemini request must switch to the fallback model, not hang or give up."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import llm
from src.errors import LLMError


class _StallingChat:
    def __init__(self, stall: bool) -> None:
        self.stall = stall

    def send_message_stream(self, _text: str):
        if self.stall:
            raise TimeoutError("read timed out")
        yield SimpleNamespace(text="جی ہاں۔", candidates=[])

    def get_history(self) -> list:
        return []


class _Chats:
    def create(self, model: str, config: object, history: list) -> _StallingChat:
        return _StallingChat(stall=False)


def _chat(fallbacks: list[str]) -> llm.GeminiChat:
    chat = llm.GeminiChat.__new__(llm.GeminiChat)
    chat.model = "stalls"
    chat._fallback_models = list(fallbacks)
    chat._config = None
    chat._client = SimpleNamespace(chats=_Chats())
    chat._chat = _StallingChat(stall=True)
    return chat


@pytest.fixture(autouse=True)
def _clear_unavailable() -> None:
    llm._unavailable.clear()
    yield
    llm._unavailable.clear()


def test_stall_switches_to_fallback_model() -> None:
    chat = _chat(["backup"])
    assert "".join(chat.reply_stream("سلام")) == "جی ہاں۔"
    assert chat.model == "backup"
    assert llm._is_unavailable("stalls")


def test_stall_without_fallback_raises_readable_error() -> None:
    chat = _chat([])
    with pytest.raises(LLMError, match="did not answer in time"):
        list(chat.reply_stream("سلام"))
