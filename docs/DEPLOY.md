# Deploying the Urdu Voice Agent (public link)

The same code runs in two modes:

| | `APP_MODE=local` (laptop, `start.bat`) | `APP_MODE=cloud` (public server) |
|---|---|---|
| Listening (STT) | Gemini, falls back to local Whisper offline | Gemini only |
| Voice (TTS) | edge-tts Uzma/Asad, falls back to offline MMS | edge-tts only |
| Voice detection | browser + Silero on the server | browser only |
| Install | `requirements.txt` (torch, ~2 GB) | `requirements-cloud.txt` (~200 MB, no torch) |
| Conversations | one per browser tab | one per browser tab, 30 min idle timeout |
| Limits | none | 8 turns/min and 30/day per visitor, 100/day in total |
| Logs on disk | `logs/latency.csv` | nothing; transcripts are not logged |

## Recommended: one free Render web service

Render runs the Docker image from this repo. The page and the API come from the same
address, so no CORS setup is needed, and Render provides the HTTPS that browsers require
before they allow the microphone.

### 1. Put the code on GitHub

1. Check that no secret is about to be published: `git status` must not list `.env`
   (it is in `.gitignore`), and `git log --all -- .env` must print nothing.
2. On github.com: **New repository** → name `urdu-voice-agent` → Public → **Create**
   (no README, it already exists).
3. In PowerShell, from the project folder:
   ```powershell
   git remote add origin https://github.com/<your-user>/urdu-voice-agent.git
   git push -u origin main
   ```

### 2. Create the service on Render

1. Sign in at https://dashboard.render.com with your GitHub account.
2. **New** → **Blueprint** → choose the `urdu-voice-agent` repository. Render reads
   `render.yaml` and shows one web service, `urdu-voice-agent` (plan: Free).
3. It asks for `GEMINI_API_KEY`: paste your key from https://aistudio.google.com/apikey.
   It is stored by Render as a secret, never in git.
4. **Apply**. The first build takes about 5 minutes. When the log shows
   `Urdu Voice Agent (cloud mode) listening`, open the address Render shows, e.g.
   `https://urdu-voice-agent.onrender.com`.
5. Open it, tap the orb, allow the microphone, and speak Urdu.

Every push to `main` redeploys automatically (`autoDeploy: true` in `render.yaml`).

### 3. Good to know

- **Cold start:** free Render services sleep after ~15 minutes without visitors. The
  next visit wakes it in about a minute (the page shows "loading" meanwhile). Open the
  link yourself a minute before you show it to someone.
- **Quota is the real limit.** Google's free tier allows roughly 20-25 requests per day
  per model, and each spoken turn uses two (listening + answer). The app spreads load
  over fallback models (`GEMINI_FALLBACK_MODELS`, `GEMINI_STT_MODEL`) and stops at
  `DAILY_TURN_LIMIT` turns per day with a friendly message ("today's free demo limit
  is reached"). A paid Gemini key raises the ceiling; then raise `DAILY_TURN_LIMIT`.
- **Abuse protection** (all configurable in the Render dashboard → Environment):
  `RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_PER_DAY` (per visitor IP), `DAILY_TURN_LIMIT`
  (everyone). Uploads are capped at 3 MB / 30 s of audio, typed messages at 500
  characters. `TRUST_PROXY` is the number of proxies in front of the app, so the limits
  use the visitor's real IP: `2` when visitors come through the Vercel front end (the
  default in `render.yaml`), `1` if they open the Render address directly. Never set it on
  a server without a proxy in front.
- **Privacy:** audio is processed in memory and never stored; transcripts are not
  logged in cloud mode; conversations live in memory only and are gone after 30 minutes
  idle or a restart. Audio and text are sent to Google (Gemini) and Microsoft (edge-tts)
  to produce the reply.
- **Security headers:** a Content-Security-Policy that only allows this site, Google
  Fonts and jsDelivr (browser voice detection), no framing, HSTS, no API docs pages.

## Why not Vercel for the backend?

Vercel runs serverless functions: each request may start a fresh instance, there is no
long-lived process to keep conversations and rate limits in memory, long streaming
responses hit execution-time limits, and the Python bundle size is limited. This app
streams a reply sentence by sentence and keeps per-visitor state, so it needs a normal
long-running server, which Render's free web service is.

## Optional: a Vercel link in front of Render

`vercel.json` publishes only the page (`web/`) on Vercel and proxies `/api/*` and
`/healthz` to the Render service, so the browser sees one site: no CORS, and the Gemini
key stays on Render.

1. Deploy on Render first (above) and copy its address, e.g.
   `https://urdu-voice-agent.onrender.com`. If Render gave the service a different name,
   replace the two `destination` URLs in `vercel.json`, commit and push.
2. On https://vercel.com: **Add New → Project → Import** the `urdu-voice-agent` GitHub repo.
   Leave every setting as detected (`vercel.json` sets "no framework" and serves `web/`),
   no environment variables are needed, click **Deploy**.
3. Open the `https://<project>.vercel.app` link and try it. Keep `TRUST_PROXY=2` on Render.

Note: Render's free service sleeps after ~15 minutes idle; the first visit after that
waits about a minute while it wakes up (the page shows "loading" meanwhile).

## Alternative: Hugging Face Spaces (Docker)

A free CPU Space (2 vCPU, 16 GB RAM) also runs this Dockerfile and sleeps less
aggressively. Create a Space with the **Docker** SDK, push this repo to it, add
`GEMINI_API_KEY` under Settings → Secrets, and set `PORT=7860` plus the variables from
`render.yaml` under Settings → Variables (Spaces route traffic to port 7860).

## Run the cloud build locally (optional)

```powershell
docker build -t urdu-voice-agent .
docker run --rm -p 8000:8000 -e GEMINI_API_KEY=<your key> urdu-voice-agent
# open http://localhost:8000 (the mic works on localhost without HTTPS)
```

Or without Docker: `$env:APP_MODE="cloud"; python -m src.server` uses the cloud
settings with your current venv.
