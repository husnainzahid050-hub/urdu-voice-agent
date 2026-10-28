"""Gemini chat client (google-genai SDK, free tier) with conversation history."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.config import Settings
from src.errors import ConfigError, LLMError, RateLimitError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
آپ ایک مددگار اردو وائس اسسٹنٹ ہیں۔ آپ کا ہر جواب بول کر سنایا جائے گا۔

Rules (follow strictly):
- Reply ONLY in Urdu, written in Urdu (Nastaliq/Arabic) script. Never use Roman Urdu.
- Do not use English words unless the user explicitly asks for English.
- Speak naturally, like a person on a phone call: short, simple sentences.
- Keep replies to two or three sentences unless the user asks for more detail.
- No markdown, no lists, no bullet points, no headings, no emojis, no symbols.
- End every sentence with the Urdu full stop ۔ or the Urdu question mark ؟
- Write numbers as Urdu words, not digits.
"""


@dataclass(frozen=True)
class LLMReply:
    """One model reply and how long it took."""

    text: str
    elapsed_seconds: float


class GeminiChat:
    """A multi-turn conversation with Gemini. History lives in the SDK chat session."""

    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.6,
        max_output_tokens: int = 300,
        thinking_budget: int | None = None,
    ) -> None:
        if not api_key:
            raise ConfigError("GEMINI_API_KEY is not set (see .env.example).")
        self.model = model
        self._client = genai.Client(api_key=api_key)
        thinking = (
            types.ThinkingConfig(thinking_budget=thinking_budget)
            if thinking_budget is not None
            else None
        )
        self._config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=thinking,
        )
        self.reset()

    @classmethod
    def from_settings(cls, settings: Settings) -> GeminiChat:
        settings.require_gemini()
        return cls(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_output_tokens,
            thinking_budget=settings.gemini_thinking_budget,
        )

    def reset(self) -> None:
        """Forget the conversation so far."""
        self._chat = self._client.chats.create(model=self.model, config=self._config)

    @property
    def turns(self) -> int:
        """Number of user + model messages in the history."""
        return len(self._chat.get_history())

    def reply(self, user_text: str) -> LLMReply:
        """Send one user message and return the model's Urdu reply."""
        started = time.perf_counter()
        try:
            response = self._chat.send_message(user_text)
        except genai_errors.APIError as exc:
            raise _readable_error(exc, self.model) from exc
        text = (response.text or "").strip()
        if not text:
            reason = response.candidates[0].finish_reason if response.candidates else "unknown"
            raise LLMError(f"Gemini returned an empty reply (finish reason: {reason}).")
        return LLMReply(text=text, elapsed_seconds=time.perf_counter() - started)


def _readable_error(exc: genai_errors.APIError, model: str) -> LLMError:
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None) or str(exc)
    if code == 429:
        return RateLimitError(
            "Gemini free-tier limit reached (HTTP 429). Wait a minute and try again, or "
            "switch GEMINI_MODEL to a Flash-Lite model, which has higher free limits. "
            f"Details: {message}"
        )
    if code in (401, 403) or (code == 400 and "API key" in message):
        return LLMError(f"Gemini rejected the API key (HTTP {code}). Check GEMINI_API_KEY.")
    if code == 404:
        return LLMError(
            f"Model {model!r} is not available to your key (HTTP 404). "
            "Run `python -m scripts.list_gemini_models` and update GEMINI_MODEL."
        )
    return LLMError(f"Gemini API error (HTTP {code}): {message}")


def list_models(api_key: str) -> list[types.Model]:
    """All models visible to this API key."""
    if not api_key:
        raise ConfigError("GEMINI_API_KEY is not set (see .env.example).")
    client = genai.Client(api_key=api_key)
    try:
        return list(client.models.list())
    except genai_errors.APIError as exc:
        raise _readable_error(exc, "-") from exc
