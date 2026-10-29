# How strong voice agents are built (research notes, October 2026)

Background for the design choices in this project. Sources are listed at the end of each section.

## 1. Latency targets

| Measure | Value | Source |
|---|---|---|
| Human gap between turns | ~200 ms median, across 10 languages | Stivers et al. 2009, via Picovoice |
| Industry voice-to-voice target | ~800 ms (hard to hit consistently with today's LLMs) | Latent Space, Pipecat talks |
| Delay where experience clearly degrades | > ~4 s; conversational fillers or visible "thinking" feedback reduce the perceived delay | arXiv 2507.22352 |
| Default silence before a turn ends | 0.5 s minimum, 3.0 s maximum | LiveKit Agents defaults |
| Default minimum interruption length | 0.5 s of speech | LiveKit Agents defaults |

Realistic target here: about 2–2.5 s from the end of speech to the first audio. That assumes free-tier Gemini for transcription and the reply, Microsoft neural voices through edge-tts, a laptop CPU, and a slow connection.

- https://picovoice.ai/guide/voice-agents/voice-ux-latency-turn-taking/
- https://latent.space/p/realtime-api
- https://ai.engineer/talks/pipecat-cloud-enterprise-voice-agents-built-on-open-source
- https://arxiv.org/html/2507.22352v1
- https://docs.livekit.io/agents/logic/turns/tuning/

## 2. Architecture patterns (Pipecat, LiveKit Agents)

- **Streaming everything.** Pipeline frameworks treat audio and text as streams of small frames. The LLM streams tokens, and TTS starts on the first complete sentence instead of the full reply. *Adopted:* Gemini streams the reply, sentences are spoken as soon as each is complete, and the first sentence is kept short.
- **Overlap the stages.** Start the next stage before the previous one has fully finished. LiveKit's "preemptive generation" starts the LLM as soon as a final transcript arrives (on by default). Preemptive TTS is off by default because cancellations waste compute. *Adopted:* the TTS of sentence n runs while Gemini writes sentence n+1.
- **Turn detection.** Silence-only VAD (voice activity detection) is the simple baseline. LiveKit and Pipecat add a turn-detection model: Pipecat's open Smart Turn v3 runs in about 12 ms on CPU and covers 23 languages, including Hindi but not Urdu. *Adopted:* Silero VAD with a tuned silence window, since no Urdu turn model exists.
- **Interruptions (barge-in).** On user speech: stop playback, flush the audio buffers, cancel the queued TTS and the LLM generation, and keep the interrupted turn in the history. LiveKit's "adaptive interruption handling" filters out backchannels ("mm-hmm"); the simple version is a minimum length of 0.5 s. *Adopted:* tap to interrupt, and optionally interrupting by voice.
- **Feedback during waits.** Show a visible "thinking" state immediately. Give a short, low-commitment cue for slow turns, tied to real system state, and never talk over the user.

- https://docs.livekit.io/agents/logic/turns/tuning/
- https://livekit.com/blog/adaptive-interruption-handling
- https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/
- https://github.com/pipecat-ai/smart-turn
- https://www.coval.dev/blog/voice-ai-latency
- https://dev.to/kenimo49/your-voice-agent-is-slow-here-are-5-tricks-to-hide-it-3pcb

## 3. VAD in the browser

- `@ricky0123/vad-web` runs Silero VAD in the browser through onnxruntime-web. It returns `onSpeechEnd(Float32Array @ 16 kHz)`, which we upload as a WAV.
- The current docs show vad-web 0.0.31 with onnxruntime-web 1.22.0, loaded via `ort.wasm.min.js` and then `bundle.min.js`. `baseAssetPath` and `onnxWASMBasePath` must be set to the matching jsDelivr `dist/` URLs.
- The package ships three models: `silero_vad_legacy.onnx`, `silero_vad_v5.onnx` and `silero_vad_v6.onnx`.
- Pin onnxruntime-web to the exact version vad-web declares. Earlier caret ranges pulled in incompatible versions.

- https://docs.vad.ricky0123.com/user-guide/browser/
- https://docs.vad.ricky0123.com/user-guide/api/
- https://www.npmjs.com/package/@ricky0123/vad-web

## 4. Running a public demo safely

- The API key stays on the server; the browser only talks to our backend.
- Rate limiting: limit turns per IP, honouring the first hop of `X-Forwarded-For` behind a proxy such as Render. On a single instance an in-memory store is enough; slowapi is the usual FastAPI library.
- Limit input size: maximum upload bytes, maximum audio seconds and maximum text length, checked before any model call.
- Give each visitor (each browser tab) their own chat history, with a time-to-live (TTL).
- Free hosts have about 512 MB of RAM, so the cloud configuration must not load torch, Whisper or MMS; those models stay local-only.

- https://pythonandchill.substack.com/p/how-to-protect-your-api-from-abusive
- https://pypi.org/project/fastapi-user-limiter

## 5. Portfolio presentation (patterns seen in open-source voice assistants)

- Lead the README with a short demo GIF or video, then a one-paragraph pitch.
- Include an architecture diagram (mic → VAD → STT → LLM → TTS → speaker, marking local and cloud parts).
- Include a **measured** latency table per stage, with the method and the hardware. Headline numbers without the method aren't trustworthy.
- Include an STT/TTS comparison table with accuracy (CER/WER, character and word error rates) and speed, labelling synthetic and real audio separately.
- Examples: Vocalis (architecture and latency breakdown in the README), RealtimeVoiceChat (modular, claims ~500 ms), Hugging Face speech-to-speech (a modular open-source pipeline), faster-local-voice-ai (under 1 s on a CPU laptop with Vosk, Ollama and Piper).
- Caveat: the original Piper TTS repository was archived on 2025-10-06, and development continues in a fork.

- https://github.com/Lex-au/Vocalis
- https://www.scriptbyai.com/realtime-voice-chat
- https://www.voiceaispace.com/knowledge/repo/speech-to-speech
- https://awesome.ecosyste.ms/projects/github.com%2Fm15-ai%2Ffaster-local-voice-ai
- https://www.kunalganglani.com/blog/local-ai-voice-assistant-whisper-piper-ollama

## 6. How this project compares

| Area | Typical strong agent | This project |
|---|---|---|
| Transport | WebRTC (LiveKit/Daily) | HTTP upload + NDJSON stream (simpler, good enough for one user per tab) |
| Turn detection | VAD + turn-detection model | Silero VAD in the browser; no Urdu turn model exists |
| STT | Streaming STT (Deepgram etc.) | Gemini transcription per turn; local Whisper offline fallback |
| LLM | Streaming, short first sentence | Gemini 3.5 Flash streaming, first sentence ≤ ~8 words |
| TTS | Streaming TTS from the first tokens | Microsoft neural Urdu voice per sentence; local MMS fallback |
| Interruption | Adaptive, voice-triggered | Tap to interrupt; voice-triggered optional (echo risk without headphones) |
| Language | Mostly English | Urdu-first: Nastaliq UI, everyday-Urdu prompt, Urdu number words for TTS |
