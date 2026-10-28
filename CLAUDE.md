# CLAUDE.md — urdu-voice-agent

Local voice agent for Windows: the user speaks Urdu, the agent answers in spoken Urdu.
Portfolio project (GitHub + LinkedIn): code must be clean, modular, documented, easy to run.
Current phase: **Phase 1**.

## Hard constraints
1. **Free only.** Open-source tools, plus the Gemini API on its FREE tier (`GEMINI_API_KEY` in `.env`).
2. **STT and TTS run locally** on Windows, Python 3.11.
3. **No secrets in git.** `.env`, `.venv/`, `models/`, `output/`, `*.wav`, `logs/` are gitignored.
4. **No hard-coded Gemini model names.** `GEMINI_MODEL` comes from `.env`; pick it with
   `python -m scripts.list_gemini_models`.
5. Model downloads go to `models/` (Whisper in `models/whisper`, TTS in `models/tts`).

## Architecture
```
Mic -> Silero VAD -> STT (faster-whisper) -> LLM (Gemini Flash, google-genai) -> text cleaning -> TTS (MMS) -> Speaker
```
- STT interface: `stt.BaseSTT.transcribe(audio, sr) -> Transcript`. Implemented: `FasterWhisperSTT`.
  Benchmark later: large-v3, ihanif/whisper-medium-urdu, sharjeel103/whisper-tiny-urdu
  (Transformers checkpoints need CTranslate2 conversion: `ct2-transformers-converter`).
- LLM: `llm.GeminiChat` (SDK chat session keeps history). System prompt: Urdu script only,
  short spoken sentences, no markdown/emojis/English, numbers as words.
- TTS interface: `tts.BaseTTS.synthesize(text) -> Speech` (cleans text, speaks sentence by
  sentence). Implemented: `MMSTTS` (facebook/mms-tts-urd-script_arabic, VitsModel).
  Planned behind the same interface: Piper Urdu voice, edge-tts (online fallback only).
- `text_utils.clean_for_tts` always runs before TTS (TTS breaks on symbols).

## Phases
1. Each component alone (`scripts/test_stt.py`, `test_llm.py`, `test_tts.py`), each prints latency.
2. Push-to-talk loop (Enter, speak, hear reply), then hands-free with VAD.
3. Lower latency: stream Gemini output, speak sentence by sentence; barge-in; conversation
   memory; per-stage latency logging.
4. Gradio UI + use case: Urdu customer-support / order-confirmation agent for a food brand.
5. README with architecture diagram, latency table, STT/TTS comparison, setup guide; demo video.

Build only the current phase.

## Conventions
- Python 3.11, type hints, docstrings, `logging` (bare `print` only in `scripts/`).
- Audio is mono float32 numpy in [-1, 1]; STT and VAD use 16 kHz.
- Errors: raise subclasses of `errors.VoiceAgentError` with messages a user can act on
  (missing key, no mic, model download failure, Gemini 429). Scripts catch them and print.
- All settings in `config.Settings.from_env()`; add new ones to `.env.example` too.
- Scripts run from the repo root as modules: `python -m scripts.test_tts`.
- Small conventional commits: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`.

## Commands (PowerShell, repo root)
```powershell
.venv\Scripts\activate
python -m scripts.list_gemini_models
python -m scripts.test_tts
python -m scripts.test_llm
python -m scripts.test_stt [--seconds 8] [--file x.wav] [--model large-v3]
```

## Current status (update at the end of every session)
- 2026-10-28: **Phase 1 done** (all three components work alone; live-mic STT and
  listening to TTS still to be checked by the owner).
  Hardware: i7-8550U (4C/8T), 16 GB RAM, Intel UHD 620, no NVIDIA GPU -> torch 2.15 CPU
  build, Whisper on cpu/int8. Python 3.11.9 installed via winget (py -3.11).
- Measured latency (warm, this laptop):
  | Stage | Setting | Time |
  |---|---|---|
  | TTS | MMS urd-script_arabic | 2.9-4.1 s for 5.4-7.9 s of audio (RTF 0.5-0.75); load ~3 s |
  | LLM | gemini-3.5-flash, thinking MINIMAL | 1.9-4.3 s (typ. 2.5 s) |
  | STT | Whisper medium int8, beam 5 | real mic: 12.8 s for 16.6 s of speech (RTF 0.77); TTS voice: ~14 s for 5.5 s (RTF ~2.5); load 8-17 s |
  Live mic test (owner, 2026-10-28): "ہلو ہلو کیا تم سمجھ سکتے ہو ... میک ٹیسٹنگ 1 2 3" transcribed
  correctly. STT is still the bottleneck (a short reply would wait ~4-6 s on STT alone). cpu_threads 4 vs 8 and beam 1 vs 5 barely change it (beam 1 is worse).
- Gemini choice: gemini-3.5-flash-lite gave good Urdu but 10-14 s per reply on the free tier;
  gemini-3.8-flash returned 503 (busy). Gemini 3 models reject thinking_budget (HTTP 400):
  use GEMINI_THINKING_LEVEL. Without MINIMAL, thinking tokens ate the 300-token limit and
  cut replies off (default raised to 400).
- MMS vocab (58 chars) has no Urdu punctuation (pauses come from sentence splitting), no
  Urdu digits (mapped to 0-9), no ۃ / ۓ (mapped in text_utils). Numbers: prompt asks Gemini
  to write them as words.
- Downloads: HF Xet stalled at 0 bytes on the owner's ~0.65 MB/s line; HF_HUB_DISABLE_XET=1
  is set in config.py. Whisper medium = 1.46 GB (~50 min first download).
- Round trip TTS -> STT on "کل تک پہنچ جائے گا" came back "کرتک پہنسے آئے گا" (synthetic
  voice; judge accuracy on real speech).

### Next (Phase 2 prep)
1. STT speed: benchmark `small` and a CTranslate2-converted Urdu fine-tune
   (sharjeel103/whisper-tiny-urdu, ihanif/whisper-medium-urdu) on real recordings,
   accuracy vs time; medium is too slow for conversation on this CPU.
2. Push-to-talk loop in pipeline.py, then hands-free with SileroVAD.
