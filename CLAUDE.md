# CLAUDE.md — urdu-voice-agent

Voice agent: the user speaks Urdu in the browser (or the laptop mic), the agent answers in
spoken Urdu. Portfolio project (GitHub + LinkedIn, public demo link later): it must look
polished, respond fast, and be easy for others to run. Owner writes Hindi/Urdu; reply in English.

## Constraints
1. **Free only**: open-source tools + Gemini API free tier (`GEMINI_API_KEY` in `.env`).
   Owner later accepted online services for speed/quality: Gemini audio transcription (local
   Whisper stays as offline fallback) and edge-tts neural voices (offline MMS fallback).
2. **No secrets in git**: `.env`, `.env.*` (except example), `models/`, `output/`, `logs/`,
   audio files, `samples/private/`, `benchmarks/audio/` are gitignored. Secrets audit of all
   history was clean on 2026-10-29.
3. **No hard-coded model guesses**: models come from `.env`; `scripts/list_gemini_models.py`
   lists what the key can use.
4. Model downloads go to `models/` (plain HTTP: `HF_HUB_DISABLE_XET=1`, Xet stalled here).
5. Never deploy or push without the owner. No Co-Authored-By / AI attribution in commits.

## Architecture
```
Browser: mic -> Silero VAD (vad-web, energy fallback) -> POST /api/turn (WAV)
Server:  STT (Gemini transcribe -> Whisper small) -> Gemini Flash-Lite stream
         -> SentenceStreamer -> 2 TTS workers (edge-tts -> MMS) -> NDJSON events
Browser: heard / reply_text / audio (in order) / done -> gapless Web Audio playback
```
- NDJSON events: `heard`, `not_heard`, `reply_text{index}`, `audio{index,mime}`,
  `done{reply,goodbye,voice_backend,timings}`, `error{message}`. `/api/chat` = typed text.
  Old endpoints kept for fallback: `/api/listen`, `/api/reply`, `/api/speak`, `/api/greeting`.
- Sessions: one Gemini chat + persona per (IP, `X-Session-Id`), 30 min TTL, cap 200 (LRU).
- Only the Gemini call holds a lock; aborting the request cancels Gemini and TTS.

## File map
- `src/server.py` FastAPI app, endpoints, sessions, security headers, cloud/local mode.
- `src/pipeline.py` `Conversation` core (hear, respond, respond_stream, warm_up),
  `SentenceStreamer`, console `VoiceAgent`, latency CSV.
- `src/stt.py` `GeminiSTT`, `FasterWhisperSTT`, `FallbackSTT`, `create_stt`, hallucination guards.
- `src/llm.py` `GeminiChat` (prompt, reply/reply_stream, retries, quota model fallback).
- `src/tts.py` `EdgeTTS`, `MMSTTS`, `FallbackTTS`, `synthesize_encoded` (MP3 to the browser).
- `src/text_utils.py` normalization, numbers to Urdu words, sentence splitting.
- `src/limits.py` per-IP token bucket, per-IP daily and global daily caps.
- `src/vad.py`, `src/audio_io.py`, `src/config.py`, `src/errors.py`.
- `web/index.html` single-file UI (orb canvas, streaming playback, settings drawer).
- `scripts/` talk (console), test_tts/test_llm/test_stt, list_gemini_models,
  benchmark_stt, record_samples. `docs/DEPLOY.md`, `docs/research.md`.

## Key decisions (with measurements, laptop i7-8550U, no GPU, ~0.65 MB/s)
- **Chat model `gemini-3.1-flash-lite`** (~1.2 s, good direct Urdu). `gemini-3.5-flash` is
  20 requests/day on the free tier (confirmed from 429 details); 3.5-flash-lite was ~5-12 s.
  Gemini 3 models reject `thinking_budget` (use `GEMINI_THINKING_LEVEL=MINIMAL`); 3.7/3.8
  flash and flash-latest reject MINIMAL (HTTP 400), so not used as fallbacks.
- **STT `gemini-3.5-transcribe`** (1.5-2.6 s; needs `language_codes=["ur"]` or it writes
  Devanagari; 25 requests/day, separate from chat), fallback Whisper `small` beam 5
  (3-4 s, CER ~11 % synthetic). Whisper `medium` was 11-14 s per turn: too slow. Whisper pads
  to 30 s windows, so short turns cost nearly as much as long ones.
- **Streaming**: time to first audio for typed messages went from 8.9-16.7 s to 2.4-5.8 s.
- **edge-tts Uzma** replaced robotic MMS as the main voice (owner: pronunciation). MMS vocab
  has no Urdu punctuation/digits/ۃ/ۓ (handled in text_utils).
- **No greeting** (owner): the app starts listening on the first tap.
- **Noise**: turns with < 0.3 s of speech are ignored; no Whisper temperature fallback (noise
  once took 52 s); prompt-echo and known-hallucination guards.
- **Deploy on Render** (Docker, `render.yaml`), not Vercel: the backend is a long-running
  streaming Python process. Cloud mode uses `requirements-cloud.txt` (~200 MB, no torch).

## Conventions
- Python 3.11, type hints, docstrings, `logging` (bare `print` only in `scripts/`).
- Audio: mono float32 numpy in [-1, 1]; STT/VAD at 16 kHz.
- Errors subclass `errors.VoiceAgentError` with messages a user can act on.
- Settings only via `config.Settings.from_env()`; add new ones to `.env.example`.
- Small conventional commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`).

## Commands (PowerShell, repo root)
```powershell
start.bat                                   # web app on http://127.0.0.1:7860
tools.bat                                   # console loop + component tests menu
.venv\Scripts\python.exe -m src.server      # same as start.bat
.venv\Scripts\python.exe -m pytest -q       # 83 tests
.venv\Scripts\python.exe -m ruff check src scripts tests
python -m scripts.benchmark_stt --sets real # STT benchmark (see --help)
python -m scripts.record_samples            # record real-voice benchmark clips
```

## Current status (2026-10-29)
- Web app with streaming replies, browser VAD, per-visitor sessions, rate limits, cloud mode,
  Docker/Render files, README, MIT license. 83 tests pass, ruff clean.
- Owner `.env`: GEMINI_MODEL=gemini-3.1-flash-lite, fallbacks 3.5-flash-lite,3.5-flash,
  STT_BACKEND=gemini, GEMINI_STT_MODEL=gemini-3.5-transcribe, STT_FALLBACK=whisper,
  STT_MODEL=small, TTS edge Uzma.
- Not yet verified: Docker build, Render deploy, browser VAD with a real mic in a browser.

### Next
1. Owner: record a demo GIF (`docs/demo.gif`), push to GitHub, deploy on Render (docs/DEPLOY.md),
   put the live URL in README.
2. Real-voice STT benchmark (`record_samples` then `benchmark_stt --sets real`).
3. One-call audio -> reply (halves quota and latency); local Piper Urdu voice.
