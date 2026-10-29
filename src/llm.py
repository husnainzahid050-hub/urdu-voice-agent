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

_PERSONA = {
    "female": (
        "You are a woman. Always use feminine verb forms for yourself, for example "
        "میں بتا سکتی ہوں، میں کرتی ہوں، میں سمجھ گئی۔"
    ),
    "male": (
        "You are a man. Always use masculine verb forms for yourself, for example "
        "میں بتا سکتا ہوں، میں کرتا ہوں، میں سمجھ گیا۔"
    ),
}

_PROMPT_TEMPLATE = """\
آپ ایک مددگار اردو وائس اسسٹنٹ ہیں۔ آپ کا ہر جواب بول کر سنایا جائے گا۔

Rules (follow strictly):
- {persona}
- Reply ONLY in Urdu script. Never use Roman Urdu or Latin letters.
- Use simple, everyday Pakistani Urdu, the way people talk at home or on the phone.
  Prefer common words over formal or bookish ones: say مدد not معاونت، بتائیں not مطلع کریں،
  ٹھیک ہے not بخوبی، شروع not آغاز، کام not امور۔
- Everyday English words that Pakistanis use (آرڈر، ڈیلیوری، موبائل، فون، ایڈریس) are fine,
  but always write them in Urdu script.
- Be warm and polite: use آپ, never تم or تو.
- Keep it short: one to three sentences, at most about forty words, unless the user
  asks for more detail. Ask at most one question at a time.
- No markdown, lists, headings, emojis, symbols or diacritics (zer, zabar, pesh).
- End every sentence with ۔ or ؟
- Write numbers as Urdu words, not digits (پانچ سو روپے، دو دن).
- If you did not understand the user, politely ask them to say it again.
"""


def build_system_prompt(gender: str = "female") -> str:
    """System prompt with the persona's grammatical gender filled in."""
    return _PROMPT_TEMPLATE.format(persona=_PERSONA.get(gender, _PERSONA["female"]))


SYSTEM_PROMPT = build_system_prompt()

# Busy (503/500) and short rate limits (429) are usually gone after a few seconds.
_RETRY_CODES = {429, 500, 503}
_RETRY_DELAYS_SECONDS = (1.5, 4.0)


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
        max_output_tokens: int = 400,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
    ) -> None:
        """`thinking_level` (MINIMAL/LOW/MEDIUM/HIGH) is for Gemini 3+ models,
        `thinking_budget` (tokens, 0 = off) for Gemini 2.5. Thinking tokens count
        against `max_output_tokens`, so heavy thinking can cut replies short."""
        if not api_key:
            raise ConfigError("GEMINI_API_KEY is not set (see .env.example).")
        self.model = model
        self._client = genai.Client(api_key=api_key)
        thinking = None
        if thinking_level or thinking_budget is not None:
            thinking = types.ThinkingConfig(
                thinking_level=thinking_level, thinking_budget=thinking_budget
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
            system_prompt=build_system_prompt(settings.agent_gender),
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_output_tokens,
            thinking_level=settings.gemini_thinking_level,
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
        response = self._send_with_retry(user_text)
        text = (response.text or "").strip()
        reason = response.candidates[0].finish_reason if response.candidates else None
        if not text:
            raise LLMError(f"Gemini returned an empty reply (finish reason: {reason}).")
        if reason == types.FinishReason.MAX_TOKENS:
            logger.warning(
                "Reply was cut off at GEMINI_MAX_OUTPUT_TOKENS; raise it or lower "
                "GEMINI_THINKING_LEVEL."
            )
        return LLMReply(text=text, elapsed_seconds=time.perf_counter() - started)

    def _send_with_retry(self, user_text: str) -> types.GenerateContentResponse:
        """Send a message, retrying a couple of times when Gemini is busy."""
        for attempt, delay in enumerate((*_RETRY_DELAYS_SECONDS, None), start=1):
            try:
                return self._chat.send_message(user_text)
            except genai_errors.APIError as exc:
                if delay is None or getattr(exc, "code", None) not in _RETRY_CODES:
                    raise _readable_error(exc, self.model) from exc
                logger.warning(
                    "Gemini HTTP %s (attempt %d), retrying in %.1f s", exc.code, attempt, delay
                )
                time.sleep(delay)
        raise AssertionError("unreachable")


def _readable_error(exc: genai_errors.APIError, model: str) -> LLMError:
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None) or str(exc)
    if code == 429:
        return RateLimitError(
            "Gemini free-tier limit reached (HTTP 429). Wait a minute and try again, or "
            "switch GEMINI_MODEL to another Flash model (each has its own free quota). "
            f"Details: {message}"
        )
    if code in (500, 503):
        return LLMError(
            f"Gemini model {model!r} is busy or temporarily unavailable (HTTP {code}). "
            "Try again in a moment, or switch GEMINI_MODEL to another Flash model."
        )
    if code == 400 and "invalid argument" in message.lower():
        return LLMError(
            f"Gemini rejected the request (HTTP 400). If you set GEMINI_THINKING_LEVEL or "
            f"GEMINI_THINKING_BUDGET, check that {model!r} supports it: Gemini 3+ uses "
            "the level, Gemini 2.5 the budget."
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
