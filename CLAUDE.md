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
- 2026-10-28: Phase 1 in progress. Hardware: i7-8550U (4C/8T), 16 GB RAM, Intel UHD 620,
  no NVIDIA GPU -> torch CPU build, Whisper on cpu/int8, default STT_MODEL=medium.
