# ---- Stage 1: Builder ----
FROM python:3.11-slim-bookworm AS builder
WORKDIR /usr/src/app

RUN set -eux; \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    ARCH="$(uname -m)"; \
    if [ "$ARCH" = "aarch64" ]; then \
        pip install --no-cache-dir torch==2.3.0 --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cpu; \
    fi && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN . /opt/venv/bin/activate && pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.11-slim-bookworm
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./static ./static
COPY ./app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
