"""Talk to the Urdu voice agent from the console.

Usage:
    python -m scripts.talk                 # hands-free: just speak, it answers
    python -m scripts.talk --push-to-talk  # press Enter before each turn
Say "اللہ حافظ" or press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import sys

from src.config import Settings, setup_logging
from src.errors import VoiceAgentError
from src.pipeline import TurnTimings, VoiceAgent


class ConsoleUI:
    """Prints the conversation and per-stage timings."""

    def status(self, message: str) -> None:
        print(f"  … {message}", flush=True)

    def user_said(self, text: str) -> None:
        print(f"\n🧑 آپ:  {text}", flush=True)

    def agent_said(self, text: str) -> None:
        print(f"🤖 ایجنٹ: {text}", flush=True)

    def timings(self, t: TurnTimings) -> None:
        print(
            f"  ⏱  you spoke {t.speech_seconds:.1f}s | STT {t.stt:.1f}s | Gemini {t.llm:.1f}s"
            f" | voice {t.tts:.1f}s ({t.tts_backend}) | reply after {t.response:.1f}s\n",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push-to-talk", action="store_true", help="press Enter before speaking")
    args = parser.parse_args()

    settings = Settings.from_env()
    setup_logging(settings.log_level)
    print("\n=== Urdu Voice Agent ===  (say اللہ حافظ or press Ctrl+C to stop)\n")
    try:
        agent = VoiceAgent(settings, ConsoleUI())
        agent.run(push_to_talk=args.push_to_talk)
    except VoiceAgentError as exc:
        print(f"\nERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nStopped. اللہ حافظ!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
