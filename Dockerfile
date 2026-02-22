# Dockerfile for LISF Agent — Lean API-key mode (no Node.js)
# Build: docker build -t lisf-agent .
# Run:   docker run -p 8080:8080 --env-file .env lisf-agent

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py index.html ./
COPY docs/lisf_md/ docs/lisf_md/

ENV PORT=8080

EXPOSE ${PORT}

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1
