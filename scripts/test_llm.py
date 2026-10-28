"""Send an Urdu sentence to Gemini and print the Urdu reply and its latency.

Usage:
    python -m scripts.test_llm
    python -m scripts.test_llm "آج موسم کیسا ہے؟"
"""

from __future__ import annotations

import argparse
import sys

from src.config import Settings, setup_logging
from src.errors import VoiceAgentError
from src.llm import GeminiChat
from src.text_utils import clean_for_tts

DEFAULT_PROMPT = "السلام علیکم، آپ کیسے ہیں؟ مجھے کشمیری چائے بنانے کا آسان طریقہ بتائیں۔"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default=DEFAULT_PROMPT, help="Urdu message to send")
    args = parser.parse_args()

    settings = Settings.from_env()
    setup_logging(settings.log_level)
    try:
        chat = GeminiChat.from_settings(settings)
        print(f"Model: {chat.model}")
        print(f"You:   {args.text}")
        reply = chat.reply(args.text)
    except VoiceAgentError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"\nReply:\n  {reply.text}")
    print(f"\nCleaned for TTS:\n  {clean_for_tts(reply.text)}")
    print(f"\nLatency: {reply.elapsed_seconds:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
