# ============================================================================
# Multi-stage Dockerfile for Product Taxonomy Classifier
# ============================================================================
# Stage 1: Builder - installs dependencies
# Stage 2: Runtime - minimal image with only what's needed
#
# Build:  docker build -t product-classifier:latest .
# Run:    docker run -p 8000:8000 product-classifier:latest
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# LightGBM needs libgomp for OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY configs/ configs/
COPY src/ src/
COPY models/ models/

# Environment
ENV PYTHONPATH=/app
ENV MODEL_PATH=models/classifier_pipeline.joblib
ENV PYTHONUNBUFFERED=1

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

# Uvicorn with production settings
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--timeout-keep-alive", "30"]
