"""Compare speech-to-text backends on Urdu: accuracy (CER/WER) and speed.

Test sets (audio is gitignored; references live in benchmarks/sentences.tsv):
    synthetic-clean  benchmarks/sentences.tsv spoken by edge-tts (Uzma + Asad), 16 kHz
    synthetic-noisy  the same with background noise added (laptop-mic-like, ~15 dB SNR)
    real             benchmarks/audio/real/<id>.wav recorded with scripts.record_samples,
                     plus output/stt_last_recording.wav (approximate reference)
Synthetic voices are clearer than real people, so they flatter every model: use the
real set to decide, the synthetic sets to compare.

Usage:
    python -m scripts.benchmark_stt --make            # synthesise the test audio once
    python -m scripts.benchmark_stt                   # run the default candidates
    python -m scripts.benchmark_stt --candidates gemini:gemini-3.5-transcribe whisper:small:1
    python -m scripts.benchmark_stt --sets real
Candidate syntax: gemini:<model>  or  whisper:<size>:<beam>
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import re
import sys
import time
import unicodedata
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from src import audio_io
from src.config import OUTPUT_DIR, ROOT_DIR, Settings, setup_logging
from src.errors import VoiceAgentError
from src.stt import BaseSTT, FasterWhisperSTT, GeminiSTT
from src.text_utils import normalize_urdu, numbers_to_words

BENCH_DIR = ROOT_DIR / "benchmarks"
SENTENCES = BENCH_DIR / "sentences.tsv"
AUDIO_DIR = BENCH_DIR / "audio"
RESULTS = BENCH_DIR / "results.csv"
RATE = 16000
VOICES = ("ur-PK-UzmaNeural", "ur-PK-AsadNeural")
REAL_EXTRA = (
    OUTPUT_DIR / "stt_last_recording.wav",
    "ہلو ہلو کیا تم سمجھ سکتے ہو ہلو ہلو ہلو مائک ٹیسٹنگ ون ٹو تھری ہلو ہلو",
)
DEFAULT_CANDIDATES = (
    "gemini:gemini-3.5-transcribe",
    "gemini:gemini-3.5-flash",
    "whisper:small:1",
    "whisper:small:5",
    "whisper:medium:1",
    "whisper:medium:5",
)


# --- scoring ---------------------------------------------------------------------------


def normalize_for_scoring(text: str) -> str:
    """Same normalisation on both sides: Urdu letter forms, digits as words,
    no punctuation or diacritics (zer/zabar/pesh; letters like آ stay), single spaces."""
    text = numbers_to_words(normalize_urdu(text))
    text = "".join(
        ch for ch in unicodedata.normalize("NFC", text) if unicodedata.category(ch) != "Mn"
    )
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def edit_distance(a: list[str] | str, b: list[str] | str) -> int:
    """Levenshtein distance between two sequences (characters or words)."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def error_rates(reference: str, hypothesis: str) -> tuple[int, int, int, int]:
    """(char errors, chars, word errors, words) after normalisation."""
    ref, hyp = normalize_for_scoring(reference), normalize_for_scoring(hypothesis)
    ref_chars, hyp_chars = ref.replace(" ", ""), hyp.replace(" ", "")
    return (
        edit_distance(ref_chars, hyp_chars),
        max(len(ref_chars), 1),
        edit_distance(ref.split(), hyp.split()),
        max(len(ref.split()), 1),
    )


# --- test sets ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    name: str
    audio: np.ndarray
    reference: str


def read_sentences() -> list[tuple[str, str]]:
    with SENTENCES.open(encoding="utf-8") as fh:
        return [(row["id"], row["text"]) for row in csv.DictReader(fh, delimiter="\t")]


async def _edge_mp3(text: str, voice: str) -> bytes:
    import edge_tts

    mp3 = bytearray()
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            mp3.extend(chunk["data"])
    return bytes(mp3)


def add_noise(audio: np.ndarray, snr_db: float = 15.0, seed: int = 0) -> np.ndarray:
    """Add pink-ish background noise at the given signal-to-noise ratio."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(len(audio)).astype(np.float32)
    pink = np.convolve(white, np.ones(8) / 8, mode="same")  # crude low-pass: room hum
    signal_power = float(np.mean(audio**2)) or 1e-8
    noise_power = float(np.mean(pink**2)) or 1e-8
    scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    return np.clip(audio + scale * pink, -1, 1).astype(np.float32)


def _synthesise(text: str, voice: str, attempts: int = 3) -> bytes:
    """edge-tts with a timeout: a dropped request otherwise hangs forever."""
    for attempt in range(1, attempts + 1):
        try:
            mp3 = asyncio.run(asyncio.wait_for(_edge_mp3(text, voice), 30))
            if mp3:
                return mp3
        except Exception as exc:  # timeout, NoAudioReceived, network
            print(f"    retry {attempt}: {exc!r}")
    raise RuntimeError(f"could not synthesise {text!r} with {voice}")


def make_synthetic() -> None:
    """Synthesise every sentence in both voices, clean and noisy, as 16 kHz WAV."""
    for voice in VOICES:
        tag = voice.split("-")[2].replace("Neural", "").lower()
        for sid, text in read_sentences():
            clean_path = AUDIO_DIR / "synthetic-clean" / f"{sid}_{tag}.wav"
            noisy_path = AUDIO_DIR / "synthetic-noisy" / f"{sid}_{tag}.wav"
            if clean_path.exists() and noisy_path.exists():
                continue
            mp3 = _synthesise(text, voice)
            audio, rate = sf.read(io.BytesIO(mp3), dtype="float32", always_2d=True)
            clean = audio_io.resample(audio.mean(axis=1), int(rate), RATE)
            pad = np.zeros(RATE // 4, np.float32)
            clean = np.concatenate([pad, clean, pad])
            audio_io.save_wav(clean_path, clean, RATE)
            noisy = add_noise(clean, seed=zlib.crc32(f"{sid}_{tag}".encode()))
            audio_io.save_wav(noisy_path, noisy, RATE)
            print(f"  made {sid}_{tag}", flush=True)


def load_set(name: str) -> list[Sample]:
    references = dict(read_sentences())
    samples = []
    for path in sorted((AUDIO_DIR / name).glob("*.wav")):
        sid = path.stem.split("_")[0]
        if sid in references:
            audio, rate = audio_io.load_wav(path)
            samples.append(Sample(path.stem, audio_io.resample(audio, rate, RATE), references[sid]))
    if name == "real" and REAL_EXTRA[0].exists():
        audio, rate = audio_io.load_wav(REAL_EXTRA[0])
        samples.append(Sample("mic_test", audio_io.resample(audio, rate, RATE), REAL_EXTRA[1]))
    return samples


# --- candidates --------------------------------------------------------------------------


def build(candidate: str, settings: Settings) -> BaseSTT:
    kind, _, rest = candidate.partition(":")
    if kind == "gemini":
        return GeminiSTT(settings.gemini_api_key, rest or settings.gemini_stt_model)
    if kind == "whisper":
        size, _, beam = rest.partition(":")
        return FasterWhisperSTT(
            model=size or "medium",
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            beam_size=int(beam or 5),
            cpu_threads=settings.stt_cpu_threads,
        )
    raise SystemExit(f"Unknown candidate {candidate!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--make", action="store_true", help="synthesise the test audio first")
    parser.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--sets", nargs="+", default=["synthetic-clean", "synthetic-noisy", "real"])
    parser.add_argument("--limit", type=int, default=0, help="max clips per set (0 = all)")
    parser.add_argument(
        "--whisper-limit", type=int, default=8, help="max clips per set for Whisper (slow on CPU)"
    )
    parser.add_argument("--show", action="store_true", help="print every transcript")
    parser.add_argument(
        "--gemini-delay", type=float, default=4.0, help="seconds between Gemini calls (free RPM)"
    )
    parser.add_argument("--out", type=Path, default=RESULTS, help="CSV file for the results")
    args = parser.parse_args()

    settings = Settings.from_env()
    setup_logging("WARNING")
    if args.make:
        make_synthetic()

    sets = {name: load_set(name) for name in args.sets}
    for name, samples in sets.items():
        print(f"{name}: {len(samples)} clips")

    rows = []
    for candidate in args.candidates:
        try:
            stt = build(candidate, settings)
        except VoiceAgentError as exc:
            print(f"SKIP {candidate}: {exc}")
            continue
        limit = args.whisper_limit if candidate.startswith("whisper") else args.limit
        for set_name, samples in sets.items():
            chosen = samples[:limit] if limit else samples
            if not chosen:
                continue
            totals = np.zeros(4)
            seconds = audio_seconds = 0.0
            failures = 0
            for sample in chosen:
                try:
                    result = stt.transcribe(sample.audio, RATE)
                except VoiceAgentError as exc:
                    failures += 1
                    print(f"  {candidate} {sample.name}: FAILED {exc}")
                    continue
                totals += error_rates(sample.reference, result.text)
                seconds += result.elapsed_seconds
                audio_seconds += result.audio_seconds
                if args.show:
                    print(f"  {sample.name} [{result.elapsed_seconds:.1f}s] {result.text}")
                if candidate.startswith("gemini"):
                    time.sleep(args.gemini_delay)  # stay under free-tier requests-per-minute
            done = len(chosen) - failures
            if not done:
                continue
            row = {
                "set": set_name,
                "candidate": candidate,
                "clips": done,
                "failed": failures,
                "cer": totals[0] / totals[1],
                "wer": totals[2] / totals[3],
                "avg_s": seconds / done,
                "rtf": seconds / max(audio_seconds, 1e-6),
            }
            rows.append(row)
            print(
                f"{set_name:16} {candidate:32} n={done:<3} CER {row['cer']:6.1%}  "
                f"WER {row['wer']:6.1%}  {row['avg_s']:5.2f}s/clip  RTF {row['rtf']:.2f}"
            )

    if rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
