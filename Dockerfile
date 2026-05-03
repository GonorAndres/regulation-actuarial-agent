# Dockerfile for ActuarialClaude — API-key mode (no Node.js, no OAuth)
# Build: docker build -t actuarial-regulation-agent .
# Run:   docker run -p 8080:8080 -e ANTHROPIC_API_KEY=sk-... actuarial-regulation-agent

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py index.html cusf.html favicon.svg build_index.py generic_keywords.txt ./
COPY docs/lisf_md/ docs/lisf_md/
COPY docs/cusf_md/ docs/cusf_md/
COPY docs/articles/ docs/articles/
COPY subagents_outputs/lisf_faq.json subagents_outputs/lisf_faq.json
COPY subagents_outputs/titulo_answers.json subagents_outputs/titulo_answers.json
COPY subagents_outputs/casos_practicos.json subagents_outputs/casos_practicos.json

# Build search index from per-article files
RUN python build_index.py

ENV PORT=8080

EXPOSE ${PORT}

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
