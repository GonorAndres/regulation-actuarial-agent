#!/bin/bash
# setup.sh — One-shot setup for LISF Agent on GCP VM
set -e

echo "=== LISF Agent Setup ==="

# 1. Choose mode
echo ""
echo "Select authentication mode:"
echo "  [1] API Key (lean, recommended for small VMs / public deployment)"
echo "  [2] OAuth  (requires Node.js + Claude CLI)"
echo ""
read -p "Enter 1 or 2 [1]: " MODE_CHOICE
MODE_CHOICE=${MODE_CHOICE:-1}

# 2. Install Python dependencies
echo ""
echo "[1/5] Installing Python packages..."
pip install -r requirements.txt --break-system-packages -q

if [ "$MODE_CHOICE" = "2" ]; then
    echo "    Installing OAuth dependencies..."
    pip install -r requirements-oauth.txt --break-system-packages -q
fi

# 3. Download LISF PDF
echo "[2/5] Downloading LISF from diputados.gob.mx..."
mkdir -p docs
if [ ! -f docs/LISF.pdf ]; then
    curl -sL -o docs/LISF.pdf "https://www.diputados.gob.mx/LeyesBiblio/pdf/LISF.pdf"
    echo "    Downloaded LISF.pdf ($(du -h docs/LISF.pdf | cut -f1))"
else
    echo "    LISF.pdf already exists"
fi

# 4. Convert PDF to markdown
echo "[3/5] Converting PDF to markdown chapters..."
if [ -d docs/lisf_md ] && [ "$(ls -A docs/lisf_md/*.md 2>/dev/null)" ]; then
    echo "    Markdown files already exist in docs/lisf_md/"
else
    echo "    Installing pymupdf4llm..."
    pip install pymupdf4llm --break-system-packages -q
    echo "    Running conversion..."
    python3 convert_pdf.py
    echo "    Conversion complete: $(ls docs/lisf_md/*.md 2>/dev/null | wc -l) files generated"
fi

# 5. Setup .env if not exists
echo "[4/5] Checking environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    if [ "$MODE_CHOICE" = "1" ]; then
        echo "    Created .env — add your ANTHROPIC_API_KEY!"
    else
        echo "    Created .env — OAuth mode (uses Claude CLI credentials)"
    fi
else
    echo "    .env exists"
fi

# 6. Verify
echo "[5/5] Verification..."
python3 -c "from fastapi import FastAPI; print('    fastapi OK')" 2>/dev/null || echo "    fastapi import failed"
python3 -c "import anthropic; print('    anthropic SDK OK')" 2>/dev/null || echo "    anthropic import failed"

if [ "$MODE_CHOICE" = "2" ]; then
    python3 -c "from claude_code_sdk import query; print('    claude-code-sdk OK')" 2>/dev/null || echo "    claude-code-sdk import failed"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
if [ "$MODE_CHOICE" = "1" ]; then
    echo "  1. Add your API key: echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env"
    echo "  2. Start the server:  uvicorn app:app --host 0.0.0.0 --port 8000"
else
    echo "  1. Authenticate:  claude login"
    echo "  2. Start the server:  uvicorn app:app --host 0.0.0.0 --port 8000"
fi
echo "  3. Open in browser:   http://localhost:8000"
echo ""
echo "If accessing from your PC, use SSH tunnel:"
echo "  gcloud compute ssh YOUR_VM -- -L 8000:localhost:8000"
echo "  Then open http://localhost:8000"
