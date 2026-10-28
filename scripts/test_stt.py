"""Record from the mic (or read a WAV) and print the Urdu transcript and timings.

Usage:
    python -m scripts.test_stt                      # record 5 s from the mic
    python -m scripts.test_stt --seconds 8
    python -m scripts.test_stt --file output/tts_test.wav
    python -m scripts.test_stt --model large-v3     # override STT_MODEL
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src import audio_io
from src.config import OUTPUT_DIR, Settings, setup_logging
from src.errors import VoiceAgentError
from src.stt import FasterWhisperSTT
from src.vad import SileroVAD


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, help="recording length (default RECORD_SECONDS)")
    parser.add_argument("--file", type=Path, help="transcribe this audio file instead of the mic")
    parser.add_argument("--model", help="Whisper model to use instead of STT_MODEL")
    args = parser.parse_args()

    settings = Settings.from_env()
    setup_logging(settings.log_level)
    rate = 16000

    try:
        load_started = time.perf_counter()
        stt = (
            FasterWhisperSTT.from_settings(settings)
            if not args.model
            else FasterWhisperSTT(
                model=args.model,
                device=settings.stt_device,
                compute_type=settings.stt_compute_type,
                language=settings.stt_language,
                beam_size=settings.stt_beam_size,
            )
        )
        vad = SileroVAD()
        load_seconds = time.perf_counter() - load_started

        if args.file:
            audio, file_rate = audio_io.load_wav(args.file)
            if file_rate != rate:
                print(f"ERROR: {args.file} is {file_rate} Hz; this test needs 16000 Hz audio.")
                return 1
            print(f"File: {args.file}")
        else:
            seconds = args.seconds or settings.record_seconds
            print(f"Microphone: {audio_io.default_device_name('input')}")
            input(f"Press Enter, then speak Urdu for {seconds:.0f} seconds...")
            print("Recording...")
            audio = audio_io.record(seconds, rate)
            saved = audio_io.save_wav(OUTPUT_DIR / "stt_last_recording.wav", audio, rate)
            print(f"Done. Saved to {saved}")

        if audio_io.peak_level(audio) < 0.01:
            print("WARNING: the recording is almost silent. Check the mic and its volume.")
        vad_started = time.perf_counter()
        if not vad.has_speech(audio, rate):
            print("No speech detected (Silero VAD). Try again and speak closer to the mic.")
            return 1
        speech = vad.trim(audio, rate)
        vad_seconds = time.perf_counter() - vad_started

        result = stt.transcribe(speech, rate)
    except VoiceAgentError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("\nTranscript:")
    print(f"  {result.text or '(empty)'}")
    print(f"\nModel:            {stt.model_name} on {stt.device} ({stt.compute_type})")
    print(f"Detected lang:    {result.language}")
    print(f"Model load:       {load_seconds:.2f} s")
    print(f"VAD + trim:       {vad_seconds:.2f} s")
    print(f"Speech length:    {result.audio_seconds:.2f} s")
    print(f"Transcribe time:  {result.elapsed_seconds:.2f} s")
    print(f"Real-time factor: {result.elapsed_seconds / max(result.audio_seconds, 1e-6):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
