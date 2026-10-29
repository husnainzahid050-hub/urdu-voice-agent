"""Gemini chat client (google-genai SDK, free tier) with conversation history."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
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
- You do not know the user's gender. When talking about the user, use the respectful
  plural forms that suit everyone (آپ کر سکتے ہیں، آپ جائیں گے، آپ نے بتایا), never feminine
  forms such as کر سکتی ہیں for the user, unless the user tells you their gender.
- Answer the actual question in your FIRST sentence. Do not open with filler such as
  جی بالکل، میں آپ کی مدد کر سکتی ہوں، اچھا سوال ہے، or by repeating the question.
  Do not greet unless the user greets you, and never introduce yourself unless asked.
- Make the first sentence short (about eight words or fewer): it is spoken first,
  while you are still writing the rest.
- Keep it short: one to three sentences, at most about forty words, unless the user
  asks for more detail.
- Ask a question back only when you truly cannot answer without it, and then only one.
- The user's words come from speech recognition and may contain small mistakes or
  odd spellings. Understand what they most likely meant and answer that. Only if the
  meaning is really unclear, ask them briefly to say it again.
- No markdown, lists, headings, emojis, symbols or diacritics (zer, zabar, pesh).
- End every sentence with ۔ or ؟
- Write numbers as Urdu words, not digits (پانچ سو روپے، دو دن).
"""


def build_system_prompt(gender: str = "female") -> str:
    """System prompt with the persona's grammatical gender filled in."""
    return _PROMPT_TEMPLATE.format(persona=_PERSONA.get(gender, _PERSONA["female"]))


SYSTEM_PROMPT = build_system_prompt()

# Busy (503/500) and short rate limits (429) are usually gone after a few seconds.
_RETRY_CODES = {429, 500, 503}
_RETRY_DELAYS_SECONDS = (1.5, 4.0)

# Models whose daily free quota is used up (or that are gone), shared by every
# conversation: a new visitor's chat starts on the first model still usable instead of
# spending a doomed request on the exhausted one. Values are time.monotonic() deadlines.
_UNAVAILABLE_SECONDS = 3600.0
_unavailable: dict[str, float] = {}
_unavailable_lock = threading.Lock()


def _mark_unavailable(model: str, seconds: float = _UNAVAILABLE_SECONDS) -> None:
    with _unavailable_lock:
        _unavailable[model] = time.monotonic() + seconds


def _is_unavailable(model: str) -> bool:
    with _unavailable_lock:
        until = _unavailable.get(model)
        if until is not None and time.monotonic() >= until:
            del _unavailable[model]
            return False
        return until is not None


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
        fallback_models: list[str] | None = None,
        client: genai.Client | None = None,
    ) -> None:
        """`thinking_level` (MINIMAL/LOW/MEDIUM/HIGH) is for Gemini 3+ models,
        `thinking_budget` (tokens, 0 = off) for Gemini 2.5. Thinking tokens count
        against `max_output_tokens`, so heavy thinking can cut replies short.
        `fallback_models` are tried in order when `model`'s daily free quota is used up;
        they must accept the same thinking setting. Pass `client` to share one HTTP client
        (and its connection pool) between many conversations."""
        if not api_key:
            raise ConfigError("GEMINI_API_KEY is not set (see .env.example).")
        candidates = [model] + [m for m in (fallback_models or []) if m and m != model]
        # Skip models another conversation already found exhausted (keep the last one,
        # so there is always something to try).
        while len(candidates) > 1 and _is_unavailable(candidates[0]):
            candidates.pop(0)
        self.model = candidates[0]
        self._fallback_models = candidates[1:]
        self._client = client or genai.Client(api_key=api_key)
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

    @property
    def client(self) -> genai.Client:
        return self._client

    @classmethod
    def from_settings(
        cls, settings: Settings, gender: str | None = None, client: genai.Client | None = None
    ) -> GeminiChat:
        """`gender` overrides settings.agent_gender (per-visitor persona)."""
        settings.require_gemini()
        return cls(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            system_prompt=build_system_prompt(gender or settings.agent_gender),
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_output_tokens,
            thinking_level=settings.gemini_thinking_level,
            thinking_budget=settings.gemini_thinking_budget,
            fallback_models=list(getattr(settings, "gemini_fallback_models", ())),
            client=client,
        )

    def reset(self) -> None:
        """Forget the conversation so far."""
        self._chat = self._client.chats.create(model=self.model, config=self._config)

    def warm_up(self) -> None:
        """Open the HTTPS connection with a free metadata call (no generation quota used)."""
        try:
            self._client.models.get(model=self.model)
        except genai_errors.APIError as exc:
            raise _readable_error(exc, self.model) from exc

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

    def reply_stream(self, user_text: str) -> Iterator[str]:
        """Send one user message and yield the reply as it is generated.

        Retries (like `reply`) only while nothing has been yielded yet; an error after
        the first chunk is raised as LLMError so the caller can keep what it has.
        """
        attempt = 0
        while True:
            attempt += 1
            yielded = False
            reason = None
            try:
                for chunk in self._chat.send_message_stream(user_text):
                    if chunk.candidates:
                        reason = chunk.candidates[0].finish_reason or reason
                    text = chunk.text or ""
                    if text:
                        yielded = True
                        yield text
            except genai_errors.APIError as exc:
                if yielded:
                    raise _readable_error(exc, self.model) from exc
                self._recover(exc, attempt)
                continue
            if not yielded:
                raise LLMError(f"Gemini returned an empty reply (finish reason: {reason}).")
            if reason == types.FinishReason.MAX_TOKENS:
                logger.warning("Reply was cut off at GEMINI_MAX_OUTPUT_TOKENS.")
            return

    def _send_with_retry(self, user_text: str) -> types.GenerateContentResponse:
        """Send a message, retrying when Gemini is busy and switching model on quota."""
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._chat.send_message(user_text)
            except genai_errors.APIError as exc:
                self._recover(exc, attempt)

    def _recover(self, exc: genai_errors.APIError, attempt: int) -> None:
        """Decide what to do after a failed call; returns to retry, raises to give up.

        - Daily free-tier quota used up (429 "per day") or model gone (404): switch to
          the next GEMINI_FALLBACK_MODELS entry at once (each model has its own quota),
          keeping the conversation history. Waiting would not help: the reset is hours away.
        - Busy (500/503) or a short 429: wait and retry a couple of times.
        """
        code = getattr(exc, "code", None)
        if (code == 404 or _is_daily_quota(exc)) and self._switch_model(exc):
            return
        retries = _RETRY_DELAYS_SECONDS
        if code in _RETRY_CODES and not _is_daily_quota(exc) and attempt <= len(retries):
            delay = retries[attempt - 1]
            logger.warning("Gemini HTTP %s (attempt %d), retrying in %.1f s", code, attempt, delay)
            time.sleep(delay)
            return
        raise _readable_error(exc, self.model) from exc

    def _switch_model(self, exc: genai_errors.APIError) -> bool:
        _mark_unavailable(self.model)
        if not getattr(self, "_fallback_models", None):
            return False
        old, self.model = self.model, self._fallback_models.pop(0)
        history = self._chat.get_history()
        self._chat = self._client.chats.create(
            model=self.model, config=self._config, history=history
        )
        logger.warning(
            "Gemini %s unavailable (HTTP %s: %s). Switched to %s.",
            old,
            getattr(exc, "code", None),
            "daily free quota used up" if _is_daily_quota(exc) else "not found",
            self.model,
        )
        return True


def _is_daily_quota(exc: genai_errors.APIError) -> bool:
    """429 because the per-day free-tier request limit is used up (not a short burst)."""
    text = f"{getattr(exc, 'message', '')} {exc}"
    return getattr(exc, "code", None) == 429 and ("PerDay" in text or "per day" in text.lower())


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
