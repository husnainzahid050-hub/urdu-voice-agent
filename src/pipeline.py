"""Voice loop: mic -> VAD -> STT -> LLM -> TTS -> speaker.

Phase 2 (not built yet): push-to-talk loop, then hands-free with VAD.
Phase 3: streaming replies, barge-in, per-stage latency logging.
"""
