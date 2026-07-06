# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11.15-slim AS builder

WORKDIR /build

# install only what's needed to compile wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ── Stage 2: lean production image ────────────────────────────────────────────
FROM python:3.11.15-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PORT=5000

WORKDIR /app

# install wheels from builder — no compiler needed
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* \
    && rm -rf /wheels

# copy only the files the server needs at runtime
COPY app.py          ./app.py
COPY model_utils.py  ./model_utils.py
COPY templates/      ./templates/
COPY model/weights.npy ./model/weights.npy
COPY model/vocab.npy   ./model/vocab.npy

# non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

EXPOSE 5000

# gunicorn: 2 workers × (2 × cpu + 1) threads is a solid baseline for I/O bound apps
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "60", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
