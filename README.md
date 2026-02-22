# LISF Agent — Consultor de la Ley de Instituciones de Seguros y Fianzas

A personal Claude-powered agent that answers questions about Mexico's LISF regulation,
built with the Claude Agent SDK and served through a FastAPI backend with a chat frontend.

## Architecture

```
Browser (any device)
    │
    ▼
FastAPI server (GCP VM, port 8000)
    │
    ├── /api/chat  →  Claude Agent SDK  →  Anthropic (via OAuth token)
    │                      │
    │                      └── Reads LISF PDF from /docs/LISF.pdf
    │
    └── /  →  Serves frontend (static HTML)
```

## Setup on GCP VM

### 1. Clone or create the project
```bash
mkdir -p ~/lisf-agent/docs
cd ~/lisf-agent
```

### 2. Download the LISF PDF
```bash
curl -o docs/LISF.pdf "https://www.diputados.gob.mx/LeyesBiblio/pdf/LISF.pdf"
```

### 3. Get your OAuth token
```bash
claude setup-token
# Copy the token output
```

### 4. Create .env file
```bash
cp .env.example .env
# Edit .env and paste your token
nano .env
```

### 5. Install dependencies
```bash
pip install fastapi uvicorn python-dotenv claude-agent-sdk --break-system-packages
```

### 6. Run the server
```bash
cd ~/lisf-agent
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 7. Access it
- From VM directly: `http://localhost:8000`
- From your PC via SSH tunnel: `gcloud compute ssh YOUR_VM -- -L 8000:localhost:8000`
  Then open `http://localhost:8000` in your browser

## Files
```
lisf-agent/
├── app.py              # FastAPI backend + Agent SDK logic
├── index.html          # Chat frontend
├── .env.example        # Template for secrets
├── .env                # Your actual secrets (gitignored)
├── .gitignore
├── docs/
│   └── LISF.pdf        # The law document
└── README.md
```
