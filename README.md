# Urdu Voice Agent

A voice assistant that runs on your own Windows laptop: you speak Urdu into the
microphone and it answers in spoken Urdu. Speech recognition and speech synthesis
run locally; only the language model is a cloud call (Gemini, free tier).

> Status: **Phase 1**. Each component works on its own and reports its latency.
> The full voice loop is coming in Phase 2.

## Architecture

```
Mic -> Silero VAD -> faster-whisper (STT) -> Gemini Flash (LLM) -> MMS-TTS Urdu -> Speaker
        local           local                  free API              local
```

| Stage | Tool | Runs |
|---|---|---|
| Voice activity | [Silero VAD](https://github.com/snakers4/silero-vad) | local |
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | local |
| Language model | Gemini Flash / Flash-Lite via [google-genai](https://pypi.org/project/google-genai/) | Google API (free tier) |
| Text-to-speech | [facebook/mms-tts-urd-script_arabic](https://huggingface.co/facebook/mms-tts-urd-script_arabic) | local |

## Requirements

- Windows 10/11, Python 3.11, a microphone and speakers
- About 5 GB of free disk space for models
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
- An NVIDIA GPU is optional (CPU works, just more slowly)

## Setup

```powershell
git clone <this repo> urdu-voice-agent
cd urdu-voice-agent
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt   # installs the CPU build of torch; see the file for GPU
copy .env.example .env      # then put your GEMINI_API_KEY in .env
python -m scripts.list_gemini_models   # pick a model, set GEMINI_MODEL in .env
```

Models download into `models/` the first time they're used.

## Test each component

```powershell
python -m scripts.test_tts     # Urdu text -> WAV -> speaker
python -m scripts.test_llm     # Urdu question -> Gemini -> Urdu reply
python -m scripts.test_stt     # 5 s from the mic -> Urdu transcript
```

Every script prints how long each step took.

## Project layout

```
src/
  config.py      settings from .env + defaults
  errors.py      readable exceptions
  audio_io.py    mic recording, playback, WAV files
  vad.py         Silero VAD
  stt.py         STT interface + faster-whisper
  llm.py         Gemini chat, system prompt, error handling
  tts.py         TTS interface + MMS
  text_utils.py  Urdu text cleaning and sentence splitting
  pipeline.py    voice loop (Phase 2)
scripts/         list_gemini_models, test_stt, test_llm, test_tts
```

## Roadmap

1. Each component working on its own, with latency numbers. **(current)**
2. Push-to-talk loop, then hands-free with VAD.
3. Streaming replies spoken sentence by sentence, barge-in, memory, latency logging.
4. Gradio UI and a real use case: an Urdu order-confirmation agent for a food brand.
5. Benchmarks (latency table, STT/TTS comparison), architecture diagram, demo video.
