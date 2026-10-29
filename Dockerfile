# Public cloud build of the Urdu Voice Agent (APP_MODE=cloud).
# Online models only (Gemini for listening and thinking, edge-tts for the voice), so it
# fits a 512 MB free instance. Local laptop use needs requirements.txt instead.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_MODE=cloud \
    HEADLESS=1 \
    PORT=8000

WORKDIR /app

COPY requirements-cloud.txt .
RUN pip install -r requirements-cloud.txt

COPY src ./src
COPY web ./web

# Never run as root.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz', timeout=4)"

CMD ["python", "-m", "src.server"]
