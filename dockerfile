# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# RAG app image.
#
# Serves the Flask app via gunicorn. The LOCAL (offline) model weights are baked
# in at build time so the running container needs NO network to HuggingFace --
# it loads everything from the local cache (HF_HUB_OFFLINE=1). The BEDROCK
# backend needs no baked models; it calls AWS at runtime using the task/instance
# IAM role.
#
# NOTE: `docker build` downloads the HF models, so build where the Hub is
# reachable (CI / AWS CodeBuild / a machine with internet). The *runtime* is
# fully offline.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Where the baked model weights live; HF loads from here offline at runtime.
ENV HF_HOME=/models \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Minimal system libs. PyMuPDF / onnxruntime / faiss / torch ship manylinux
# wheels, so we mainly need libgomp (OpenMP runtime used by faiss & onnxruntime).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (the default wheel bundles CUDA and is ~5x bigger).
# Pinning the CPU index keeps the image small; the app auto-detects CPU anyway.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the offline model weights into the image (needs network at build time).
COPY scripts/download_models.py scripts/download_models.py
RUN python scripts/download_models.py

# Application code + the committed OCR text cache (so OCR never runs at startup).
COPY . .

# Lock the runtime to offline model loading -- no calls to the HuggingFace Hub.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOST=0.0.0.0

EXPOSE 8080

# Single worker so the memory-heavy pipeline is built once; threads serve
# concurrent requests. app.py starts the one-time build at import.
CMD ["sh", "-c", "gunicorn --workers 1 --threads 8 --timeout 120 --bind 0.0.0.0:${PORT} app:app"]