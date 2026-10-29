"""Speak an Urdu sentence: write a WAV, play it and print the timings.

Usage:
    python -m scripts.test_tts
    python -m scripts.test_tts "آپ کا آرڈر کل تک پہنچ جائے گا۔"
    python -m scripts.test_tts --no-play
"""

from __future__ import annotations

import argparse
import sys
import time

from src import audio_io
from src.config import OUTPUT_DIR, Settings, setup_logging
from src.errors import VoiceAgentError
from src.text_utils import clean_for_tts
from src.tts import create_tts

DEFAULT_TEXT = (
    "السلام علیکم! میں آپ کی اردو اسسٹنٹ ہوں۔ آپ کا آرڈر کل تک پہنچ جائے گا۔ "
    "کیا آپ اس وقت گھر پر ہوں گے؟"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT, help="Urdu text to speak")
    parser.add_argument("--no-play", action="store_true", help="only write the WAV file")
    args = parser.parse_args()

    settings = Settings.from_env()
    setup_logging(settings.log_level)
    try:
        load_started = time.perf_counter()
        tts = create_tts(settings)
        load_seconds = time.perf_counter() - load_started

        print(f"Text:    {args.text}")
        print(f"Cleaned: {clean_for_tts(args.text)}")
        speech = tts.synthesize(args.text)
        path = audio_io.save_wav(OUTPUT_DIR / "tts_test.wav", speech.audio, speech.sample_rate)
        print(f"Saved:   {path}")
        if not args.no_play:
            print("Playing...")
            audio_io.play(speech.audio, speech.sample_rate)
    except VoiceAgentError as exc:
        print(f"ERROR: {exc}")
        return 1

    used = settings.tts_voice if speech.backend == "edge" else settings.tts_model
    print(f"\nBackend:          {speech.backend} ({used})")
    print(f"Model load:       {load_seconds:.2f} s")
    print(f"Synthesis time:   {speech.elapsed_seconds:.2f} s")
    print(f"Audio length:     {speech.audio_seconds:.2f} s")
    print(f"Real-time factor: {speech.elapsed_seconds / max(speech.audio_seconds, 1e-6):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
