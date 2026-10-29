"""Exceptions with messages that are safe to show to the user as-is."""


class VoiceAgentError(Exception):
    """Base class for all errors raised by this project."""


class ConfigError(VoiceAgentError):
    """A required setting is missing or invalid (for example GEMINI_API_KEY)."""


class AudioDeviceError(VoiceAgentError):
    """No usable microphone or speaker was found, or the device failed."""


class ModelLoadError(VoiceAgentError):
    """A local model (STT or TTS) could not be downloaded or loaded."""


class STTError(VoiceAgentError):
    """Speech recognition failed (for example the online transcriber is unreachable)."""


class STTQuotaError(STTError):
    """The online transcriber's free quota is used up (HTTP 429 RESOURCE_EXHAUSTED)."""


class TTSError(VoiceAgentError):
    """Speech synthesis failed (for example the online voice is unreachable)."""


class LLMError(VoiceAgentError):
    """The Gemini API returned an error."""


class RateLimitError(LLMError):
    """Gemini returned HTTP 429 (free-tier quota or rate limit reached)."""
