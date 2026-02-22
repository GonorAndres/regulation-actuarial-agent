"""
LISF Agent -- FastAPI backend
Dual-mode: API key (anthropic SDK) or OAuth (claude-code-sdk).
Streams responses via Server-Sent Events (SSE).
"""
import asyncio
import json
import logging
import os
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lisf-agent")

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

load_dotenv()

# Write OAuth credentials from env var to disk (Cloud Run secret mount workaround)
_creds_env = os.environ.get("CLAUDE_CREDENTIALS_JSON")
if _creds_env:
    _creds_dir = Path.home() / ".claude"
    _creds_dir.mkdir(parents=True, exist_ok=True)
    (_creds_dir / ".credentials.json").write_text(_creds_env)
    logger.info("Wrote OAuth credentials to %s", _creds_dir / ".credentials.json")

app = FastAPI(title="LISF Agent")

# --- Configuration -----------------------------------------------------------

PROJECT_DIR = Path(__file__).parent.resolve()
DOCS_DIR = PROJECT_DIR / "docs"
LISF_MD_DIR = DOCS_DIR / "lisf_md"

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
FAQ_PATH = PROJECT_DIR / "subagents_outputs" / "lisf_faq.json"

# --- FAQ Cache ---------------------------------------------------------------

def _normalize(text: str) -> str:
    """Normalize text for fuzzy FAQ matching: lowercase, strip accents/punctuation."""
    import unicodedata
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) not in ("Mn",))  # strip accents
    text = re.sub(r"[^\w\s]", "", text)  # strip punctuation
    return " ".join(text.split())

def _load_faq() -> list[dict]:
    """Load FAQ entries from JSON file."""
    if FAQ_PATH.exists():
        try:
            return json.loads(FAQ_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

FAQ_ENTRIES = _load_faq()
# Pre-compute normalized questions for matching
FAQ_NORMALIZED = [(_normalize(e["q"]), i) for i, e in enumerate(FAQ_ENTRIES)]

def _match_faq(user_message: str) -> str | None:
    """Check if user message matches a FAQ question. Returns answer or None."""
    norm = _normalize(user_message)
    if not norm:
        return None
    # Exact normalized match
    for nq, idx in FAQ_NORMALIZED:
        if norm == nq:
            return FAQ_ENTRIES[idx]["a"]
    # Substring match: if the user's message contains the FAQ question or vice versa
    for nq, idx in FAQ_NORMALIZED:
        # User typed something very close to the FAQ question
        words_user = set(norm.split())
        words_faq = set(nq.split())
        if not words_faq:
            continue
        overlap = len(words_user & words_faq) / len(words_faq)
        if overlap >= 0.75 and len(words_user) <= len(words_faq) + 3:
            return FAQ_ENTRIES[idx]["a"]
    return None

# Detect mode at import time
API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
USE_API_KEY_MODE = bool(API_KEY)

SYSTEM_PROMPT = """Eres un consultor experto en regulacion mexicana de seguros y fianzas.
Tu fuente principal es la Ley de Instituciones de Seguros y de Fianzas (LISF).

## Herramientas disponibles
Tienes 3 herramientas para consultar la LISF:
1. **list_lisf_files** — Muestra el indice de archivos con los articulos que contiene cada uno. Usala primero para orientarte.
2. **search_lisf** — Busca texto en toda la LISF. Usa terminos cortos y especificos. Si no encuentras resultados, prueba con sinonimos o sin acentos.
3. **read_lisf_file** — Lee un archivo completo. Usala despues de identificar el archivo relevante con list o search.

## Estrategia de busqueda
- SIEMPRE consulta la ley antes de responder. No confies en tu memoria.
- Primero busca con search_lisf usando palabras clave. Si necesitas contexto completo, lee el archivo con read_lisf_file.
- Para preguntas sobre un articulo especifico (ej: "Art. 201"), busca "ARTICULO 201" directamente.
- Si la busqueda no da resultados, intenta con variantes: sin acentos, sinonimos, o busca en el indice.

## Formato de respuesta
- Responde en espanol a menos que el usuario escriba en otro idioma.
- Escribe en PROSA, como un consultor explicando a un colega. Parrafos fluidos y naturales.
- NO uses listas con viñetas ni bullet points para explicar. Solo usa listas numeradas cuando
  cites fracciones o incisos textuales de la ley que ya vienen en formato de lista.
- Estructura: primero la respuesta directa en un parrafo, luego el fundamento legal.
- Cita articulos en linea dentro del texto: "segun el Articulo 201, fraccion I..." o "(Art. 201-I)".
- Cuando cites texto literal de la ley, usa comillas para distinguirlo de tu explicacion.
- Se conciso. No repitas el texto completo del articulo — resume y cita la referencia.

## Limites
- Si no encuentras la respuesta en la LISF, dilo honestamente.
- Si el usuario pregunta algo fuera del alcance de la LISF, puedes responder
  con tu conocimiento general pero aclara que no proviene de la ley.
"""

# System prompt for OAuth mode (points at markdown files if available, else PDF)
SYSTEM_PROMPT_OAUTH = """Eres un consultor experto en regulacion mexicana de seguros y fianzas.
Tu fuente principal es la Ley de Instituciones de Seguros y de Fianzas (LISF).

{source_instructions}

## Estrategia de busqueda
- SIEMPRE consulta la ley antes de responder. No confies en tu memoria.
- Consulta docs/lisf_md/00_indice.md para orientarte sobre que archivo contiene que articulos.
- Usa Grep para buscar terminos especificos en docs/lisf_md/.
- Usa Read para leer el archivo completo cuando necesites contexto.

## Formato de respuesta
- Responde en espanol a menos que el usuario escriba en otro idioma.
- Escribe en PROSA, como un consultor explicando a un colega. Parrafos fluidos y naturales.
- NO uses listas con viñetas ni bullet points. Solo usa listas numeradas cuando
  cites fracciones o incisos textuales de la ley que ya vienen en formato de lista.
- Cita articulos en linea dentro del texto: "segun el Articulo 201, fraccion I...".
- Se conciso. Resume y cita la referencia, no repitas texto completo.

## Limites
- Si no encuentras la respuesta en la LISF, dilo honestamente.
- Si el usuario pregunta algo fuera del alcance de la LISF, puedes responder
  con tu conocimiento general pero aclara que no proviene de la ley.
"""


# =============================================================================
# API KEY MODE — anthropic SDK with custom tools
# =============================================================================

if USE_API_KEY_MODE:
    from anthropic import AsyncAnthropic

    _client = AsyncAnthropic(api_key=API_KEY)

    # Tool definitions for the messages API
    TOOLS = [
        {
            "name": "list_lisf_files",
            "description": (
                "Lista todos los archivos markdown de la LISF disponibles "
                "con su indice de articulos. Usa esto primero para saber "
                "que archivos existen y que articulos contiene cada uno."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "search_lisf",
            "description": (
                "Busca texto en todos los archivos markdown de la LISF. "
                "Devuelve las primeras 50 lineas que coincidan con la busqueda, "
                "junto con el nombre del archivo. Usa busquedas cortas y especificas."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Texto a buscar (case-insensitive)",
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "read_lisf_file",
            "description": (
                "Lee el contenido completo de un archivo markdown especifico "
                "de la LISF. Usa list_lisf_files primero para ver los nombres "
                "de archivos disponibles."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Nombre del archivo (e.g. '01_titulo_primero.md')",
                    }
                },
                "required": ["filename"],
            },
        },
    ]

    def _get_source_dir():
        """Return the directory containing LISF source files."""
        if LISF_MD_DIR.is_dir() and any(LISF_MD_DIR.glob("*.md")):
            return LISF_MD_DIR
        return None

    def _execute_tool(name: str, input_data: dict) -> str:
        """Execute a tool and return its result as a string."""
        source_dir = _get_source_dir()

        if source_dir is None:
            return "Error: No se encontraron archivos markdown de la LISF en docs/lisf_md/. Ejecuta convert_pdf.py primero."

        if name == "list_lisf_files":
            index_file = source_dir / "00_indice.md"
            if index_file.exists():
                return index_file.read_text(encoding="utf-8")[:30000]
            # Fallback: list all .md files
            files = sorted(source_dir.glob("*.md"))
            return "\n".join(f.name for f in files)

        elif name == "search_lisf":
            query = input_data.get("query", "").strip()
            if not query:
                return "Error: query vacia"
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            results = []
            for md_file in sorted(source_dir.glob("*.md")):
                try:
                    for i, line in enumerate(md_file.read_text(encoding="utf-8").splitlines(), 1):
                        if pattern.search(line):
                            results.append(f"[{md_file.name}:{i}] {line.strip()}")
                            if len(results) >= 50:
                                break
                except Exception:
                    continue
                if len(results) >= 50:
                    break
            if not results:
                return f"No se encontraron resultados para: {query}"
            return "\n".join(results)

        elif name == "read_lisf_file":
            filename = input_data.get("filename", "").strip()
            if not filename:
                return "Error: filename vacio"
            # Sanitize
            safe_name = Path(filename).name
            target = source_dir / safe_name
            if not target.exists():
                available = [f.name for f in sorted(source_dir.glob("*.md"))]
                return f"Archivo no encontrado: {safe_name}. Disponibles: {', '.join(available)}"
            content = target.read_text(encoding="utf-8")
            if len(content) > 30000:
                content = content[:30000] + "\n\n... [TRUNCADO - archivo muy largo] ..."
            return content

        return f"Herramienta desconocida: {name}"

    async def _api_key_stream(user_message: str):
        """Stream a response using the Anthropic API with tool use."""
        messages = [{"role": "user", "content": user_message}]

        max_iterations = 15
        for _ in range(max_iterations):
            # Stream the response
            collected_content = []
            stop_reason = None

            async with _client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "text":
                            pass  # text will come via deltas
                        elif event.content_block.type == "tool_use":
                            # Notify frontend about tool use
                            yield json.dumps({
                                "type": "tool_use",
                                "tool": event.content_block.name,
                                "input": "",
                            })
                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield json.dumps({
                                "type": "text",
                                "content": event.delta.text,
                            })

                # Get the final message to check for tool use
                final_message = await stream.get_final_message()
                stop_reason = final_message.stop_reason
                collected_content = final_message.content

            # If model wants to use tools, execute them and continue
            if stop_reason == "tool_use":
                # Add assistant message with all content blocks
                messages.append({
                    "role": "assistant",
                    "content": [block.model_dump() for block in collected_content],
                })

                # Execute each tool call and collect results
                tool_results = []
                for block in collected_content:
                    if block.type == "tool_use":
                        result = _execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})
            else:
                # Model is done (end_turn or max_tokens)
                break

        yield json.dumps({"type": "done", "subtype": stop_reason or "end_turn"})


# =============================================================================
# OAUTH MODE — claude-code-sdk (conditional import)
# =============================================================================

_oauth_available = False

if not USE_API_KEY_MODE:
    try:
        # Clear CLAUDECODE env to avoid "cannot be launched inside another
        # Claude Code session" errors.
        os.environ.pop("CLAUDECODE", None)

        from claude_code_sdk import (
            query as claude_query,
            ClaudeCodeOptions,
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ToolUseBlock,
            ToolResultBlock,
        )

        # Monkey-patch: SDK v0.0.25 raises on unknown message types
        import claude_code_sdk._internal.client as _sdk_client
        _original_parse = _sdk_client.parse_message

        def _safe_parse(data):
            try:
                return _original_parse(data)
            except Exception:
                return SystemMessage(subtype=data.get("type", "unknown"), data=data)

        _sdk_client.parse_message = _safe_parse
        _oauth_available = True
    except ImportError:
        pass

if _oauth_available:
    def _get_oauth_system_prompt():
        """Build OAuth system prompt based on available sources."""
        if LISF_MD_DIR.is_dir() and any(LISF_MD_DIR.glob("*.md")):
            instructions = (
                "Los archivos markdown de la LISF se encuentran en: docs/lisf_md/\n"
                "Usa Grep para buscar articulos y Read para leer archivos especificos.\n"
                "Consulta docs/lisf_md/00_indice.md para ver el indice de archivos."
            )
        else:
            instructions = (
                "El documento PDF de la LISF se encuentra en: docs/LISF.pdf\n"
                "Puedes leerlo usando las herramientas disponibles (Read, Bash)."
            )
        return SYSTEM_PROMPT_OAUTH.format(source_instructions=instructions)

    AGENT_OPTIONS = ClaudeCodeOptions(
        system_prompt=_get_oauth_system_prompt(),
        model=CLAUDE_MODEL,
        allowed_tools=["Read", "Bash", "Glob", "Grep"],
        permission_mode="bypassPermissions",
        cwd=str(PROJECT_DIR),
        max_turns=15,
    )

    async def _oauth_stream(user_message: str):
        """Stream a response using claude-code-sdk."""
        async for message in claude_query(
            prompt=user_message,
            options=AGENT_OPTIONS,
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text:
                        yield json.dumps({
                            "type": "text",
                            "content": block.text,
                        })
                    elif isinstance(block, ToolUseBlock):
                        yield json.dumps({
                            "type": "tool_use",
                            "tool": block.name,
                            "input": str(block.input)[:200],
                        })
            elif isinstance(message, ResultMessage):
                subtype = getattr(message, "subtype", "end_turn")
                if subtype == "error_during_execution":
                    logger.error("Claude SDK error_during_execution: %s", vars(message) if hasattr(message, '__dict__') else str(message))
                yield json.dumps({
                    "type": "done",
                    "subtype": subtype,
                })


# =============================================================================
# Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the chat frontend."""
    html_path = PROJECT_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/chat")
async def chat(request: Request):
    """
    Receive a user message, stream Claude's response via SSE.
    Request body: {"message": "Que dice el articulo 201?"}
    Response: text/event-stream with JSON chunks
    """
    body = await request.json()
    user_message = body.get("message", "").strip()

    if not user_message:
        return {"error": "No message provided"}

    # Check FAQ cache first for instant response
    faq_answer = _match_faq(user_message)

    async def event_stream():
        try:
            if faq_answer is not None:
                # Instant FAQ response — no API call needed
                yield f"data: {json.dumps({'type': 'text', 'content': faq_answer})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'subtype': 'faq_cache'})}\n\n"
                return

            if USE_API_KEY_MODE:
                gen = _api_key_stream(user_message)
            elif _oauth_available:
                gen = _oauth_stream(user_message)
            else:
                yield f"data: {json.dumps({'type': 'error', 'content': 'No auth configured. Set ANTHROPIC_API_KEY or install claude-code-sdk.'})}\n\n"
                return

            async for chunk in gen:
                yield f"data: {chunk}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/faq")
async def faq():
    """Return FAQ questions for the frontend."""
    return [{"q": e["q"]} for e in FAQ_ENTRIES]


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    lisf_pdf = (DOCS_DIR / "LISF.pdf").exists()
    lisf_md = LISF_MD_DIR.is_dir() and any(LISF_MD_DIR.glob("*.md"))

    if USE_API_KEY_MODE:
        mode = "api_key"
        auth_configured = True
    elif _oauth_available:
        mode = "oauth"
        import shutil
        auth_configured = shutil.which("claude") is not None
    else:
        mode = "none"
        auth_configured = False

    return {
        "status": "ok",
        "mode": mode,
        "auth_configured": auth_configured,
        "lisf_document": lisf_pdf,
        "lisf_markdown": lisf_md,
    }


# --- Run with: uvicorn app:app --host 0.0.0.0 --port 8000 ---
