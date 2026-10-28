"""List the Gemini models your API key can use and suggest a Flash / Flash-Lite one.

Usage:  python -m scripts.list_gemini_models [--all]
"""

from __future__ import annotations

import argparse
import re
import sys

from src.config import Settings, setup_logging
from src.errors import VoiceAgentError
from src.llm import list_models


def _version(name: str) -> tuple[float, ...]:
    match = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return (float(match.group(1)),) if match else (0.0,)


def _is_stable_flash(name: str) -> bool:
    lowered = name.lower()
    return "flash" in lowered and not any(
        tag in lowered for tag in ("preview", "exp", "tts", "image", "live", "audio")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="also show non-chat models")
    args = parser.parse_args()

    settings = Settings.from_env()
    setup_logging(settings.log_level)
    try:
        models = list_models(settings.gemini_api_key)
    except VoiceAgentError as exc:
        print(f"ERROR: {exc}")
        return 1

    chat_models = [m for m in models if "generateContent" in (m.supported_actions or [])]
    shown = models if args.all else chat_models
    print(f"{len(shown)} models ({'all' if args.all else 'support generateContent'}):\n")
    for m in sorted(shown, key=lambda m: m.name or ""):
        name = (m.name or "").removeprefix("models/")
        print(f"  {name:<45} in={m.input_token_limit} out={m.output_token_limit}")

    names = [(m.name or "").removeprefix("models/") for m in chat_models]
    flash = sorted((n for n in names if _is_stable_flash(n)), key=_version, reverse=True)
    lite = [n for n in flash if "lite" in n]
    full = [n for n in flash if "lite" not in n]
    print("\nSuggested for a voice agent (put one in .env as GEMINI_MODEL):")
    if lite:
        print(f"  Flash-Lite (fastest, highest free limits): {lite[0]}")
    if full:
        print(f"  Flash (smarter, lower free limits):        {full[0]}")
    if not flash:
        print("  No stable Flash model found; pick a Flash model from the list above.")
    print(f"\nCurrent GEMINI_MODEL: {settings.gemini_model or '(not set)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
