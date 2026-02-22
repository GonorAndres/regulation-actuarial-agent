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
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

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

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
MODEL_MAP = {
    "rapido": "claude-haiku-4-5-20251001",
    "detallado": CLAUDE_MODEL,
}
FAQ_PATH = PROJECT_DIR / "subagents_outputs" / "lisf_faq.json"

# --- Auth -------------------------------------------------------------------

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() in ("true", "1", "yes")
ACCESS_CODE = os.getenv("ACCESS_CODE", "")
ALLOWED_EMAILS = {
    e.strip().lower()
    for e in os.getenv("ALLOWED_EMAILS", "").split(",")
    if e.strip()
}

def _verify_auth(request: Request) -> dict | None:
    """Verify request auth via Google token or access code. Returns claims or None."""
    if not AUTH_ENABLED:
        return {"email": "anonymous"}

    auth = request.headers.get("Authorization", "")

    # Check access code (X-Access-Code header)
    code = request.headers.get("X-Access-Code", "")
    if ACCESS_CODE and code == ACCESS_CODE:
        return {"email": "guest", "auth_method": "access_code"}

    # Check Google token
    if GOOGLE_CLIENT_ID and auth.startswith("Bearer "):
        token = auth[7:]
        try:
            from google.oauth2 import id_token
            from google.auth.transport import requests as google_requests
            claims = id_token.verify_oauth2_token(
                token, google_requests.Request(), GOOGLE_CLIENT_ID
            )
            email = claims.get("email", "").lower()
            if ALLOWED_EMAILS and email not in ALLOWED_EMAILS:
                logger.warning("Rejected email: %s", email)
                return None
            claims["auth_method"] = "google"
            return claims
        except Exception as e:
            logger.warning("Token verification failed: %s", e)
            return None

    # No valid auth method and auth is enabled
    if not GOOGLE_CLIENT_ID and not ACCESS_CODE:
        return {"email": "anonymous"}  # auth enabled but nothing configured
    return None

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

# --- Article Index -----------------------------------------------------------

_ARTICLE_PATTERN = re.compile(r"\*\*ARTÍCULO\s+(\d+)")

def _build_article_index() -> dict[int, str]:
    """Scan LISF markdown files and build article_number -> filename mapping."""
    index: dict[int, str] = {}
    if not LISF_MD_DIR.is_dir():
        return index
    for md_file in sorted(LISF_MD_DIR.glob("*.md")):
        if md_file.name in ("00_indice.md", "full_lisf.md"):
            continue
        try:
            for line in md_file.read_text(encoding="utf-8").splitlines():
                m = _ARTICLE_PATTERN.search(line)
                if m:
                    index[int(m.group(1))] = md_file.name
        except Exception:
            continue
    return index

ARTICLE_INDEX = _build_article_index()
logger.info("Article index built: %d articles mapped", len(ARTICLE_INDEX))

_ARTICLE_REF = re.compile(r"(?:art[ií]culo|art\.?)\s*(\d+)", re.IGNORECASE)

def _article_hints(user_message: str) -> str:
    """Detect article references in user message and return file hints."""
    matches = _ARTICLE_REF.findall(user_message)
    if not matches:
        return ""
    hints = []
    seen = set()
    for num_str in matches:
        num = int(num_str)
        if num in seen:
            continue
        seen.add(num)
        filename = ARTICLE_INDEX.get(num)
        if filename:
            hints.append(
                f"[interno] Articulo {num} -> archivo {filename}. No menciones este archivo al usuario."
            )
    return "\n".join(hints)

# Detect mode at import time
API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
USE_API_KEY_MODE = bool(API_KEY)

_LISF_CONTEXT = """## Datos clave de la LISF (usa esta informacion para responder sin buscar)
- **Nombre completo:** Ley de Instituciones de Seguros y de Fianzas (LISF)
- **Publicacion original:** 4 de abril de 2013 en el DOF
- **Total de articulos:** 510 (Arts. 1-510), mas articulos transitorios
- **Estructura:** 13 Titulos, cada uno con capitulos. Los Articulos Transitorios (originales y de todas las reformas) se ubican al final de la ley.
- **Reformas publicadas en el DOF:**
  1. 10 de enero de 2014 — Materia financiera (Arts. 49, 50, 51, 80, 369, 372)
  2. 22 de junio de 2018 — Inclusion de personas con discapacidad (Art. 27)
  3. 11 de mayo de 2022 — Paridad de genero (Art. 368)
  4. 24 de enero de 2024 — Procedimiento administrativo (Arts. 334, 335, 364, 388, 478)
  5. 14 de noviembre de 2025 — Homologacion con Codigo Nacional de Procedimientos Civiles y Familiares (Arts. 193, 280, 281, 479)
- **Articulos mas recientemente reformados:** 193, 280, 281 y 479 (reforma del 14 nov 2025)
- **Regulador:** Comision Nacional de Seguros y Fianzas (CNSF)
- **Alcance:** Regula la organizacion, operacion y funcionamiento de Instituciones de Seguros, Instituciones de Fianzas y Sociedades Mutualistas de Seguros

## Estructura de Titulos
| Titulo | Tema | Articulos aprox. |
|--------|------|-----------------|
| I | Disposiciones preliminares | 1-18 |
| II | Organizacion (autorizaciones, capital, gobierno corporativo) | 19-38 |
| III | Intermediarios (agentes, reaseguradoras extranjeras) | 39-89 |
| IV | Operacion (contratos, notas tecnicas, coaseguro, reaseguro) | 90-117 |
| V | Reservas tecnicas, inversiones, capital minimo, solvencia | 118-273 |
| VI | Contabilidad, actuarios, auditoria | 214-293 |
| VII | Vigilancia, medidas correctivas, intervencion | 294-319 |
| VIII | Revocacion, liquidacion, quiebra | 274-319 |
| IX | Otras instituciones (reaseguradoras, oficinas de representacion) | 320-369 |
| X | Grupos financieros, filiales | 370-392 |
| XI | CNSF (facultades, sanciones) | 393-443 |
| XII | Procedimientos administrativos y penales | 444-485 |
| XIII | Disposiciones finales | 486-510 |
"""

SYSTEM_PROMPT = """**Aviso:** Esta herramienta es una referencia rapida de estudio. No sustituye la lectura completa de la LISF ni constituye asesoria legal.

Eres un tutor amigable que ayuda a entender la Ley de Instituciones de Seguros y de Fianzas (LISF) de forma clara y accesible.

""" + _LISF_CONTEXT + """
## Herramientas disponibles
Tienes 3 herramientas para consultar la LISF:
1. **search_lisf** -- Busca texto en toda la LISF. Usa terminos cortos y especificos.
2. **read_lisf_file** -- Lee un archivo completo. Si el usuario pregunta por un articulo especifico y te indican el archivo, lee directamente ese archivo.
3. **list_lisf_files** -- Muestra el indice de archivos. Usala solo si necesitas orientarte.

## Estrategia de respuesta
- Si puedes responder con los Datos clave de arriba, hazlo INMEDIATAMENTE sin usar herramientas. Usa herramientas solo cuando necesites el texto exacto de un articulo.
- Si la pregunta es ambigua o demasiado general, pide una aclaracion al usuario en vez de buscar a ciegas. Ejemplo: "articulos mas nuevos" puede significar los de numeracion mas alta, los mas recientemente reformados, o los transitorios. Pregunta que quiere decir.
- Responde de forma progresiva: empieza con una explicacion breve y clara, luego consulta la fuente para citar el fundamento exacto.
- SIEMPRE consulta la ley antes de dar la cita exacta. No confies en tu memoria para los detalles.
- Si el mensaje incluye una pista de archivo, lee ese archivo directamente sin buscar primero.

## Formato de respuesta
- Responde en espanol a menos que el usuario escriba en otro idioma.
- Usa un tono didactico y accesible, como un tutor explicando a un estudiante.
- Escribe en prosa clara. Solo usa listas numeradas cuando cites fracciones o incisos textuales de la ley.
- Cita articulos en linea: "segun el Articulo 201, fraccion I..." o "(Art. 201-I)".
- Se conciso. Resume y cita la referencia.
- Siempre indica donde profundizar: "Para mas detalle, consulta el Articulo X, fracciones Y-Z del Titulo N."
- NO uses emojis en tus respuestas. Nunca.
- NUNCA menciones nombres de archivos, rutas internas, ni herramientas (Read, Grep, docs/lisf_md/, etc.) en tu respuesta. El usuario no sabe que existen. Cuando quieras referir al usuario a una fuente, usa el nombre del Titulo y Capitulo (ej: "Titulo Quinto, Capitulo Tercero") o la URL publica: https://www.diputados.gob.mx/LeyesBiblio/pdf/LISF.pdf

## Limites
- Si no encuentras la respuesta en la LISF, dilo honestamente.
- Si el usuario pregunta algo fuera del alcance de la LISF, puedes responder
  con tu conocimiento general pero aclara que no proviene de la ley.
"""

# System prompt for OAuth mode (points at markdown files if available, else PDF)
SYSTEM_PROMPT_OAUTH = """**Aviso:** Esta herramienta es una referencia rapida de estudio. No sustituye la lectura completa de la LISF ni constituye asesoria legal.

Eres un tutor amigable que ayuda a entender la Ley de Instituciones de Seguros y de Fianzas (LISF) de forma clara y accesible.

""" + _LISF_CONTEXT + """
{source_instructions}

## Estrategia de respuesta
- Si puedes responder con los Datos clave de arriba, hazlo INMEDIATAMENTE sin usar herramientas. Usa Read/Grep solo cuando necesites el texto exacto de un articulo.
- Si la pregunta es ambigua o demasiado general, pide una aclaracion al usuario en vez de buscar a ciegas. Ejemplo: "articulos mas nuevos" puede significar los de numeracion mas alta, los mas recientemente reformados, o los transitorios. Pregunta que quiere decir.
- Responde de forma progresiva: empieza con una explicacion breve y clara, luego consulta la fuente para citar el fundamento exacto.
- SIEMPRE consulta la ley antes de dar la cita exacta. No confies en tu memoria para los detalles.
- Si el mensaje incluye una pista de archivo, lee ese archivo directamente sin buscar primero.
- Usa Grep solo cuando necesites buscar un tema general.

## Formato de respuesta
- Responde en espanol a menos que el usuario escriba en otro idioma.
- Usa un tono didactico y accesible, como un tutor explicando a un estudiante.
- Escribe en prosa clara. Solo usa listas numeradas cuando cites fracciones o incisos textuales de la ley.
- Cita articulos en linea: "segun el Articulo 201, fraccion I..." o "(Art. 201-I)".
- Se conciso. Resume y cita la referencia.
- Siempre indica donde profundizar: "Para mas detalle, consulta el Articulo X, fracciones Y-Z del Titulo N."
- NO uses emojis en tus respuestas. Nunca.
- NUNCA menciones nombres de archivos, rutas internas, ni herramientas (Read, Grep, docs/lisf_md/, etc.) en tu respuesta. El usuario no sabe que existen. Cuando quieras referir al usuario a una fuente, usa el nombre del Titulo y Capitulo (ej: "Titulo Quinto, Capitulo Tercero") o la URL publica: https://www.diputados.gob.mx/LeyesBiblio/pdf/LISF.pdf

## Limites
- Si no encuentras la respuesta en la LISF, dilo honestamente.
- Si el usuario pregunta algo fuera del alcance de la LISF, puedes responder
  con tu conocimiento general pero aclara que no proviene de la ley.
"""

HAIKU_SUFFIX = (
    "\n\n## Modo rapido\n"
    "Estas en modo rapido. Se mas conciso: "
    "da la respuesta directa con la cita del articulo. "
    "Omite explicaciones largas y ve al grano."
)


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
            return "Error: Los archivos de la LISF no estan disponibles en este momento."

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

    async def _api_key_stream(user_message: str, history: list | None = None, *, model: str = CLAUDE_MODEL, rapido: bool = False):
        """Stream a response using the Anthropic API with tool use."""
        hints = _article_hints(user_message)
        prompt = f"{user_message}\n\n[Sistema: {hints}]" if hints else user_message

        # Build messages from conversation history
        messages = []
        if history:
            # Include prior turns (exclude the current message which is last in history)
            for turn in history[:-1]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})

        system = SYSTEM_PROMPT + (HAIKU_SUFFIX if rapido else "")

        max_iterations = 5
        for _ in range(max_iterations):
            # Stream the response
            collected_content = []
            stop_reason = None

            async with _client.messages.stream(
                model=model,
                max_tokens=4096,
                system=system,
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
                "## Fuentes internas (NO mencionar al usuario)\n"
                "Los archivos de la LISF estan en docs/lisf_md/. "
                "Usa Grep para buscar y Read para leer. "
                "El indice esta en docs/lisf_md/00_indice.md.\n"
                "NUNCA menciones estos paths, nombres de archivo, ni herramientas en tu respuesta."
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
        allowed_tools=["Read", "Grep"],
        permission_mode="bypassPermissions",
        cwd=str(PROJECT_DIR),
        max_turns=5,
    )

    async def _oauth_stream(user_message: str, history: list | None = None, *, model: str = CLAUDE_MODEL, rapido: bool = False):
        """Stream a response using claude-code-sdk."""
        hints = _article_hints(user_message)
        prompt = f"{user_message}\n\n[Sistema: {hints}]" if hints else user_message

        # Include conversation history as context in the prompt
        if history and len(history) > 1:
            context_lines = []
            for turn in history[:-1]:
                role = "Usuario" if turn.get("role") == "user" else "Asistente"
                content = turn.get("content", "")
                if content:
                    context_lines.append(f"{role}: {content}")
            if context_lines:
                context = "\n\n".join(context_lines)
                prompt = f"[Historial de conversacion previa]\n{context}\n\n[Mensaje actual del usuario]\n{prompt}"

        system_prompt = _get_oauth_system_prompt() + (HAIKU_SUFFIX if rapido else "")
        # OAuth mode: always use CLAUDE_MODEL (credentials may not authorize other models)
        # Rapido differentiation is handled via prompt suffix only
        request_options = ClaudeCodeOptions(
            system_prompt=system_prompt,
            model=CLAUDE_MODEL,
            allowed_tools=["Read", "Grep"],
            permission_mode="bypassPermissions",
            cwd=str(PROJECT_DIR),
            max_turns=3 if rapido else 5,
        )

        async for message in claude_query(
            prompt=prompt,
            options=request_options,
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


@app.get("/api/auth-config")
async def auth_config():
    """Return auth configuration for the frontend."""
    return {
        "google_client_id": GOOGLE_CLIENT_ID if AUTH_ENABLED else "",
        "auth_required": AUTH_ENABLED and bool(GOOGLE_CLIENT_ID or ACCESS_CODE),
        "has_google": AUTH_ENABLED and bool(GOOGLE_CLIENT_ID),
        "has_access_code": AUTH_ENABLED and bool(ACCESS_CODE),
    }


@app.post("/api/verify-code")
async def verify_code(request: Request):
    """Verify an access code. Returns success or 401."""
    body = await request.json()
    code = body.get("code", "").strip()
    if ACCESS_CODE and code == ACCESS_CODE:
        return {"valid": True}
    return JSONResponse(status_code=401, content={"valid": False, "error": "Codigo de acceso invalido."})


@app.post("/api/chat")
async def chat(request: Request):
    """
    Receive a user message, stream Claude's response via SSE.
    Request body: {"message": "Que dice el articulo 201?"}
    Response: text/event-stream with JSON chunks
    """
    # Auth check
    claims = _verify_auth(request)
    if claims is None:
        return JSONResponse(
            status_code=401,
            content={"error": "No autorizado. Inicia sesion o ingresa el codigo de acceso."},
        )

    body = await request.json()
    user_message = body.get("message", "").strip()
    history = body.get("history", [])
    mode = body.get("mode", "detallado")
    model = MODEL_MAP.get(mode, MODEL_MAP["detallado"])
    is_rapido = (mode == "rapido")

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
                gen = _api_key_stream(user_message, history, model=model, rapido=is_rapido)
            elif _oauth_available:
                gen = _oauth_stream(user_message, history, model=model, rapido=is_rapido)
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


@app.get("/api/indice")
async def indice():
    """Return the LISF index content."""
    index_file = LISF_MD_DIR / "00_indice.md"
    if index_file.exists():
        return {"content": index_file.read_text(encoding="utf-8")}
    return {"content": "Indice no disponible."}


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
