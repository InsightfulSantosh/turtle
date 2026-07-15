FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/cache

WORKDIR /app
COPY embedding-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --requirement requirements.txt
COPY embedding-service/app.py ./app.py

RUN useradd --create-home --uid 10001 appuser && mkdir -p /models/cache && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/healthz', timeout=2)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "1"]
