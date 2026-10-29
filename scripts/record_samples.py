"""Record your own voice reading the benchmark sentences, for a real accuracy test.

Each sentence from benchmarks/sentences.tsv is shown; read it aloud and pause.
Recording stops by itself when you go quiet. Files go to benchmarks/audio/real/
(gitignored), then run:  python -m scripts.benchmark_stt --sets real

Usage:
    python -m scripts.record_samples            # all sentences
    python -m scripts.record_samples --only s03 s07
    python -m scripts.record_samples --redo     # overwrite existing recordings
"""

from __future__ import annotations

import argparse
import sys

from scripts.benchmark_stt import AUDIO_DIR, RATE, read_sentences
from src import audio_io
from src.config import setup_logging
from src.errors import VoiceAgentError
from src.vad import CHUNK_SAMPLES, SileroVAD


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", help="sentence ids to record, e.g. s03 s07")
    parser.add_argument("--redo", action="store_true", help="re-record sentences already saved")
    args = parser.parse_args()

    setup_logging("WARNING")
    out_dir = AUDIO_DIR / "real"
    vad = SileroVAD()
    sentences = [(sid, text) for sid, text in read_sentences() if not args.only or sid in args.only]
    print(f"Microphone: {audio_io.default_device_name('input')}")
    print("Read each sentence in your normal voice, then pause. Ctrl+C to stop.\n")
    try:
        for number, (sid, text) in enumerate(sentences, start=1):
            path = out_dir / f"{sid}_me.wav"
            if path.exists() and not args.redo:
                print(f"[{number}/{len(sentences)}] {sid} already recorded (use --redo)")
                continue
            while True:
                print(f"[{number}/{len(sentences)}] {sid}:  {text}")
                input("   Press Enter, then read it... ")
                vad.reset()
                audio = audio_io.record_utterance(
                    lambda chunk: vad.chunk_probability(chunk, RATE),
                    sample_rate=RATE,
                    chunk_samples=CHUNK_SAMPLES[RATE],
                    end_silence_ms=1200,
                    max_seconds=20,
                    start_timeout_seconds=8,
                    on_speech_start=lambda: print("   recording..."),
                )
                if audio is None:
                    print("   Didn't hear you, let's try again.")
                    continue
                audio_io.save_wav(path, audio, RATE)
                print(f"   saved {path.name} ({len(audio) / RATE:.1f} s)\n")
                break
    except VoiceAgentError as exc:
        print(f"ERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
    print("Now run:  python -m scripts.benchmark_stt --sets real")
    return 0


if __name__ == "__main__":
    sys.exit(main())
