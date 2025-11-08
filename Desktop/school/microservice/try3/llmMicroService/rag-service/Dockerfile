FROM python:3.11-slim-bookworm AS builder
WORKDIR /usr/src/app
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y build-essential curl && \
    python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

FROM python:3.11-slim-bookworm
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY ./app.py .
ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]