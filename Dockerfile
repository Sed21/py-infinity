# Unified Docker Image with ENGINE selection
# Build: docker build --build-arg ENGINE=slim -t infinity:slim .
#        docker build --build-arg ENGINE=full -t infinity:full .

ARG ENGINE=slim

FROM python:3.13-slim-bookworm AS builder

ARG ENGINE
ENV ENGINE=${ENGINE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install uv for fast package management
RUN pip install uv

# Copy package files
COPY pyproject.toml README.md ./
COPY infinity_emb ./infinity_emb

# Install dependencies based on ENGINE
# slim: ONNX-only (~300MB) - no torch
# full: torch + optimum (~1.1GB) - maximum compatibility
RUN if [ "$ENGINE" = "full" ]; then \
        echo "Installing FULL dependencies (torch + optimum)..." && \
        uv pip install --system -e ".[full]" \
            --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
        echo "Installing SLIM dependencies (ONNX-only)..." && \
        uv pip install --system -e ".[slim]"; \
    fi

# Cleanup
RUN find /usr/local/lib/python*/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python*/site-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python*/site-packages -type f -name "*.pyc" -delete 2>/dev/null || true \
    && rm -rf /root/.cache 2>/dev/null || true

# Runtime stage
FROM python:3.13-slim-bookworm AS runtime

ARG ENGINE
ENV ENGINE=${ENGINE} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    INFINITY_ENGINE=${ENGINE}

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin/infinity_emb /usr/local/bin/

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/.cache/huggingface \
    && chown -R appuser:appuser /app

USER appuser
WORKDIR /app

EXPOSE 7997

# Set default engine based on build arg
# slim builds use optimum engine, full builds can use torch
ENV INFINITY_ENGINE=${ENGINE:-optimum}

ENTRYPOINT ["infinity_emb"]
CMD ["v2", "--port", "7997"]

# Labels for identification
LABEL org.opencontainers.image.title="infinity-emb" \
      org.opencontainers.image.description="Text embedding server - ENGINE=${ENGINE}" \
      org.opencontainers.image.source="https://github.com/Sed21/py-infinity"
