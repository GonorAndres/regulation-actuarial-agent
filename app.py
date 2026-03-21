"""
ActuarialClaude -- FastAPI backend
Mexican Insurance Regulation Agent (LISF + CUSF).
Three modes: Vertex AI (production), API Key (fallback), Agent SDK (dev).
Streams responses via Server-Sent Events (SSE).
"""
import json
import logging
import os
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("actuarial-claude")

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

load_dotenv()

app = FastAPI(title="ActuarialClaude")

# =============================================================================
# Configuration
# =============================================================================

PROJECT_DIR = Path(__file__).parent.resolve()
DOCS_DIR = PROJECT_DIR / "docs"
LISF_MD_DIR = DOCS_DIR / "lisf_md"
CUSF_MD_DIR = DOCS_DIR / "cusf_md"

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
MODEL_MAP = {
    "rapido": "claude-haiku-4-5-20251001",
    "intuitivo": CLAUDE_MODEL,
    "detallado": CLAUDE_MODEL,
}
FAQ_PATH = PROJECT_DIR / "subagents_outputs" / "lisf_faq.json"
CASOS_PATH = PROJECT_DIR / "subagents_outputs" / "casos_practicos.json"

# Auth
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() in ("true", "1", "yes")
ACCESS_CODE = os.getenv("ACCESS_CODE", "")
SONNET_PASSWORD = os.getenv("SONNET_PASSWORD", "")

# Rate limiting
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# =============================================================================
# Mode Detection & Client Init
# =============================================================================

API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
VERTEX_PROJECT = os.getenv("VERTEX_PROJECT_ID", "") or os.getenv("GOOGLE_CLOUD_PROJECT", "")
VERTEX_REGION = os.getenv("VERTEX_REGION", "us-east5")

_client = None
MODE = "none"

if API_KEY:
    from anthropic import AsyncAnthropic
    _client = AsyncAnthropic(api_key=API_KEY)
    MODE = "api_key"
    logger.info("Mode: API Key")
elif VERTEX_PROJECT:
    from anthropic import AsyncAnthropicVertex
    _client = AsyncAnthropicVertex(region=VERTEX_REGION, project_id=VERTEX_PROJECT)
    MODE = "vertex"
    logger.info("Mode: Vertex AI (project=%s, region=%s)", VERTEX_PROJECT, VERTEX_REGION)

# Agent SDK (dev mode) -- conditional import
_agent_sdk_available = False
if not _client:
    try:
        os.environ.pop("CLAUDECODE", None)
        from claude_agent_sdk import (
            query as claude_query,
            ClaudeAgentOptions,
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ToolUseBlock,
        )
        _agent_sdk_available = True
        MODE = "agent_sdk"
        logger.info("Mode: Agent SDK (dev)")
    except ImportError:
        try:
            os.environ.pop("CLAUDECODE", None)
            from claude_code_sdk import (
                query as claude_query,
                ClaudeCodeOptions as ClaudeAgentOptions,
                AssistantMessage,
                ResultMessage,
                SystemMessage,
                TextBlock,
                ToolUseBlock,
            )
            import claude_code_sdk._internal.client as _sdk_client
            _original_parse = _sdk_client.parse_message
            def _safe_parse(data):
                try:
                    return _original_parse(data)
                except Exception:
                    return SystemMessage(subtype=data.get("type", "unknown"), data=data)
            _sdk_client.parse_message = _safe_parse
            _agent_sdk_available = True
            MODE = "agent_sdk"
            logger.info("Mode: Agent SDK (dev, legacy claude-code-sdk)")
        except ImportError:
            logger.warning("No API key, no Vertex project, no Agent SDK. Backend will not respond.")

# =============================================================================
# Auth & Rate Limiting
# =============================================================================

def _verify_auth(request: Request) -> dict | None:
    """Verify access code. Returns claims dict or None."""
    if not AUTH_ENABLED:
        return {"email": "anonymous"}
    code = request.headers.get("X-Access-Code", "")
    if ACCESS_CODE and code == ACCESS_CODE:
        return {"email": "guest", "auth_method": "access_code"}
    if not ACCESS_CODE:
        return {"email": "anonymous"}
    return None

_rate_store: dict[str, list[float]] = defaultdict(list)

def _check_rate_limit(request: Request) -> bool:
    """Return True if request is within rate limit."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
    now = time.time()
    _rate_store[ip] = [t for t in _rate_store[ip] if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_store[ip]) >= RATE_LIMIT_REQUESTS:
        return False
    _rate_store[ip].append(now)
    # Periodic cleanup: remove stale IPs every ~100 requests
    if sum(len(v) for v in _rate_store.values()) > 200:
        cutoff = now - RATE_LIMIT_WINDOW * 2
        stale = [k for k, v in _rate_store.items() if all(t < cutoff for t in v)]
        for k in stale:
            del _rate_store[k]
    return True

# =============================================================================
# FAQ Cache
# =============================================================================

def _normalize(text: str) -> str:
    """Normalize text for fuzzy FAQ matching."""
    import unicodedata
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) not in ("Mn",))
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

def _load_faq() -> list[dict]:
    if FAQ_PATH.exists():
        try:
            return json.loads(FAQ_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

FAQ_ENTRIES = _load_faq()
FAQ_NORMALIZED = [(_normalize(e["q"]), i) for i, e in enumerate(FAQ_ENTRIES)]

def _match_faq(user_message: str) -> str | None:
    norm = _normalize(user_message)
    if not norm:
        return None
    for nq, idx in FAQ_NORMALIZED:
        if norm == nq:
            return FAQ_ENTRIES[idx]["a"]
    for nq, idx in FAQ_NORMALIZED:
        words_user = set(norm.split())
        words_faq = set(nq.split())
        if not words_faq:
            continue
        overlap = len(words_user & words_faq) / len(words_faq)
        if overlap >= 0.75 and len(words_user) <= len(words_faq) + 3:
            return FAQ_ENTRIES[idx]["a"]
    return None

# =============================================================================
# SQLite Search Database
# =============================================================================

DB_PATH = PROJECT_DIR / "docs" / "regulation.db"
_ARTICLE_REF = re.compile(r"(?:art[ií]culo|art\.?)\s*(\d+)", re.IGNORECASE)
_DISP_REF = re.compile(r"(?:disposici[oó]n|disp\.?)\s*(\d+\.\d+\.\d+)", re.IGNORECASE)

def _open_db() -> sqlite3.Connection | None:
    if not DB_PATH.exists():
        logger.warning("Database not found: %s. Run build_index.py first.", DB_PATH)
        return None
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

_db = _open_db()
if _db:
    _db_count = _db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    logger.info("SQLite database: %d articles/dispositions indexed", _db_count)
else:
    _db_count = 0
    logger.warning("No SQLite database -- search will not work")

def _search_db(query: str, law: str = "both", limit: int = 10) -> list[dict]:
    """Full-text search using FTS5. Returns matching articles with text."""
    if not _db:
        return []
    try:
        law_clause = "" if law == "both" else f"AND a.law = '{law}'"
        rows = _db.execute(f"""
            SELECT a.law, a.number, a.title, a.filename, a.text
            FROM articles_fts f
            JOIN articles a ON a.id = f.rowid
            WHERE articles_fts MATCH ?
            {law_clause}
            ORDER BY rank
            LIMIT ?
        """, (query, limit)).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("FTS5 search error for '%s': %s", query, e)
        return []

def _get_article_db(law: str, number: str) -> dict | None:
    """Get a specific article/disposition by number."""
    if not _db:
        return None
    row = _db.execute(
        "SELECT law, number, title, filename, text FROM articles WHERE law = ? AND number = ?",
        (law, number),
    ).fetchone()
    return dict(row) if row else None

def _get_cross_refs_db(law: str, number: str) -> list[dict]:
    """Find articles that mention this article number."""
    if not _db:
        return []
    pattern = f"%artículo {number}%" if law == "lisf" else f"%disposición {number}%"
    rows = _db.execute(
        "SELECT law, number, title, filename FROM articles WHERE text LIKE ? AND NOT (law = ? AND number = ?) LIMIT 20",
        (pattern, law, number),
    ).fetchall()
    return [dict(r) for r in rows]

def _article_to_doc_block(article: dict) -> dict:
    """Convert a database article row to a document block for Citations API."""
    law_label = article["law"].upper()
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": article["text"]},
        "title": f"{law_label} - Art. {article['number']} ({article['title']})",
        "citations": {"enabled": True},
    }

def _select_documents(user_message: str) -> list[dict]:
    """Select relevant documents: by specific article refs OR by FTS5 search."""
    docs = []

    # First: check for specific article/disposition references
    for num_str in _ARTICLE_REF.findall(user_message):
        art = _get_article_db("lisf", str(int(num_str)))
        if art:
            docs.append(_article_to_doc_block(art))
    for disp_str in _DISP_REF.findall(user_message):
        art = _get_article_db("cusf", disp_str)
        if art:
            docs.append(_article_to_doc_block(art))

    # If no specific refs found, do a FTS5 search
    if not docs:
        results = _search_db(user_message, limit=5)
        for r in results[:3]:
            docs.append(_article_to_doc_block(r))

    # Add cache_control to last document
    if docs:
        docs[-1] = {**docs[-1], "cache_control": {"type": "ephemeral"}}
    return docs

# =============================================================================
# Regulation Index (compact, always in context)
# =============================================================================

def _load_index_text(source_dir: Path, label: str) -> str:
    """Load the regulation index as compact text for the system prompt."""
    index_file = source_dir / "00_indice.md"
    if index_file.exists():
        return f"## Indice {label}\n{index_file.read_text(encoding='utf-8')[:8000]}\n"
    return ""

LISF_INDEX_TEXT = _load_index_text(LISF_MD_DIR, "LISF")
CUSF_INDEX_TEXT = _load_index_text(CUSF_MD_DIR, "CUSF")

# =============================================================================
# System Prompt
# =============================================================================

_LISF_CONTEXT = """## Datos clave de la LISF
- **Nombre completo:** Ley de Instituciones de Seguros y de Fianzas (LISF)
- **Publicación original:** 4 de abril de 2013 en el DOF
- **Total de artículos:** 510 (Arts. 1-510), más artículos transitorios
- **Estructura:** 13 Títulos, cada uno con capítulos
- **Reformas publicadas en el DOF:**
  1. 10 de enero de 2014 -- Materia financiera (Arts. 49, 50, 51, 80, 369, 372)
  2. 22 de junio de 2018 -- Inclusión de personas con discapacidad (Art. 27)
  3. 11 de mayo de 2022 -- Paridad de género (Art. 368)
  4. 24 de enero de 2024 -- Procedimiento administrativo (Arts. 334, 335, 364, 388, 478)
  5. 14 de noviembre de 2025 -- Homologación con Código Nacional (Arts. 193, 280, 281, 479)
- **Regulador:** Comisión Nacional de Seguros y Fianzas (CNSF)

## Estructura de Títulos LISF
| Título | Tema | Artículos aprox. |
|--------|------|-----------------|
| I | Disposiciones preliminares | 1-18 |
| II | Organización | 19-38 |
| III | Intermediarios | 39-89 |
| IV | Operación | 90-117 |
| V | Reservas, inversiones, solvencia | 118-273 |
| VI | Contabilidad, actuarios, auditoría | 214-293 |
| VII | Vigilancia, medidas correctivas | 294-319 |
| VIII | Revocación, liquidación, quiebra | 274-319 |
| IX | Otras instituciones | 320-369 |
| X | Grupos financieros, filiales | 370-392 |
| XI | CNSF (facultades, sanciones) | 393-443 |
| XII | Procedimientos administrativos y penales | 444-485 |
| XIII | Disposiciones finales | 486-510 |
"""

_CUSF_CONTEXT = """## Datos clave de la CUSF
- **Nombre completo:** Circular Única de Seguros y Fianzas (CUSF)
- **Emitida por:** Comisión Nacional de Seguros y Fianzas (CNSF)
- **Naturaleza:** Normativa secundaria que reglamenta la LISF
- **Estructura:** Disposiciones organizadas por materia (reservas, inversiones, gobierno corporativo, etc.)
- **Relación con LISF:** La CUSF detalla y operacionaliza lo que la LISF establece en términos generales
"""

SYSTEM_PROMPT = """**Aviso legal:** Esta herramienta es una referencia de estudio y consulta. No sustituye la lectura completa de la normativa ni constituye asesoría legal o actuarial.

Eres ActuarialClaude, un consultor especializado en regulación aseguradora mexicana. Tu audiencia son actuarios, abogados corporativos y profesionales del sector asegurador y afianzador en México.

## Tu conocimiento
Dominas dos cuerpos normativos fundamentales:
1. **LISF** -- Ley de Instituciones de Seguros y de Fianzas (ley federal, 510 artículos)
2. **CUSF** -- Circular Única de Seguros y Fianzas (normativa secundaria emitida por la CNSF)

La LISF establece el marco general; la CUSF lo reglamenta con disposiciones operativas detalladas.

""" + _LISF_CONTEXT + _CUSF_CONTEXT + """
## Herramientas disponibles
Tienes herramientas para consultar la normativa:
- **search_regulation** -- Busca texto en la LISF, CUSF o ambas
- **read_chapter** -- Lee un capítulo completo (se incorpora con citas verificables)
- **find_related_articles** -- Encuentra artículos que referencian uno dado
- **correlate_lisf_cusf** -- Relaciona artículos de la LISF con disposiciones de la CUSF

## Estrategia de respuesta
- Responde como un colega experto: directo, preciso, sin condescendencia.
- Si puedes responder con los datos clave de arriba, hazlo de inmediato sin usar herramientas.
- Cuando la pregunta sea ambigua, pregunta para clarificar. Un actuario prefiere que le preguntes a recibir una respuesta genérica.
- Para cuestiones simples (definiciones, artículos específicos), responde de inmediato con la cita.
- Para cuestiones complejas (interacción LISF-CUSF, interpretación, cálculo de reservas), estructura tu respuesta: contexto -> fundamento legal -> análisis -> conclusión práctica.
- SIEMPRE consulta la ley antes de citar texto exacto. No confíes en tu memoria para los detalles.
- Si el mensaje incluye pistas de archivo, lee ese archivo directamente.
- Cuando un artículo de la LISF tiene reglamentación en la CUSF, mencionalo proactivamente.

## Formato
- Responde en español a menos que el usuario escriba en otro idioma.
- Escribe en prosa clara. Solo usa listas numeradas cuando cites fracciones o incisos textuales de la ley.
- SIEMPRE especifica de cuál ley proviene la cita: "Art. 201 de la LISF" o "Disposición 5.1.1 de la CUSF". Nunca cites solo "Art. 201" sin indicar si es LISF o CUSF.
- Sé conciso. Resume y cita la referencia con su ley de origen.
- Indica dónde profundizar: "Para más detalle, consulta el Artículo X de la LISF, Título N" o "Capítulo Y de la CUSF".
- NO uses emojis. Nunca.
- NUNCA menciones nombres de archivos, rutas internas, ni herramientas en tu respuesta. El usuario no sabe que existen.
- Para referencias externas usa el nombre del Título y Capítulo, o las URLs públicas: diputados.gob.mx o cnsf.gob.mx.

## Límites
- Si no encuentras la respuesta en la LISF ni la CUSF, dilo honestamente.
- Para temas fuera del alcance de la regulación, puedes responder con conocimiento general pero aclara que no proviene de la normativa.
"""

HAIKU_SUFFIX = (
    "\n\n## Modo rápido -- OBLIGATORIO\n"
    "MÁXIMO 2-3 ORACIONES. Sin listas, sin subtítulos, sin estructura.\n"
    "Formato: respuesta directa + cita (Art. X de la LISF / Disp. Y de la CUSF).\n"
    "Si el usuario necesita más detalle, pregúntale: \"¿Quieres que profundice?\"\n"
    "NUNCA des más de 4 líneas. Es una búsqueda rápida, no un análisis."
)

INTUITIVO_SUFFIX = """

## Modo intuitivo
Estás en modo intuitivo. Tu objetivo es que el usuario ENTIENDA la regulación, no solo la conozca.

Estructura tu respuesta así:
1. **Caso práctico:** Empieza con un escenario real y concreto (una empresa, un actuario, un siniestro). Hazlo vivido y específico.
2. **Qué dice la ley:** Cita el fundamento legal que aplica al caso, con número de artículo/disposición.
3. **Por qué importa:** Explica las consecuencias prácticas -- qué pasa si no se cumple, qué problemas resuelve.
4. **Dónde profundizar:** Indica los artículos o capítulos relacionados para estudio completo.

Usa analogías accesibles. Evita lenguaje excesivamente jurídico. El tono es de un colega senior explicándole a un actuario junior en una capacitación.
"""

# =============================================================================
# Casos Practicos Cache (for Intuitivo mode)
# =============================================================================

def _load_casos() -> list[dict]:
    if CASOS_PATH.exists():
        try:
            return json.loads(CASOS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

CASOS_ENTRIES = _load_casos()
CASOS_NORMALIZED = [(_normalize(e["q"]), i) for i, e in enumerate(CASOS_ENTRIES)]

def _match_caso(user_message: str) -> str | None:
    """Match user message against pre-built casos practicos."""
    norm = _normalize(user_message)
    if not norm:
        return None
    for nq, idx in CASOS_NORMALIZED:
        if norm == nq:
            return CASOS_ENTRIES[idx]["a"]
    for nq, idx in CASOS_NORMALIZED:
        words_user = set(norm.split())
        words_faq = set(nq.split())
        if not words_faq:
            continue
        overlap = len(words_user & words_faq) / len(words_faq)
        if overlap >= 0.75 and len(words_user) <= len(words_faq) + 3:
            return CASOS_ENTRIES[idx]["a"]
    return None

# =============================================================================
# Tool Definitions
# =============================================================================

TOOLS = [
    {
        "name": "search_regulation",
        "description": (
            "Busca texto en los archivos de la LISF, CUSF o ambas. "
            "Devuelve las primeras 50 líneas que coincidan. "
            "Usa búsquedas cortas y específicas."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Texto a buscar (case-insensitive)"},
                "law": {
                    "type": "string",
                    "enum": ["lisf", "cusf", "both"],
                    "description": "En cuál normativa buscar. Default: both",
                    "default": "both",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_chapter",
        "description": (
            "Lee el contenido completo de un capítulo específico de la LISF o CUSF. "
            "El capítulo se incorpora al contexto con citas habilitadas."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Nombre del archivo (e.g. '01_título_primero_capítulo_único.md')"},
                "law": {"type": "string", "enum": ["lisf", "cusf"], "description": "Ley a consultar. Default: lisf", "default": "lisf"},
            },
            "required": ["filename"],
        },
    },
    {
        "name": "find_related_articles",
        "description": (
            "Dado un número de artículo, encuentra todos los artículos que lo referencian "
            "dentro de la misma ley. Útil para entender el contexto y alcance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "article_number": {"type": "integer", "description": "Número del artículo (1-510)"},
                "law": {"type": "string", "enum": ["lisf", "cusf"], "description": "En cuál ley buscar. Default: lisf", "default": "lisf"},
            },
            "required": ["article_number"],
        },
    },
    {
        "name": "correlate_lisf_cusf",
        "description": (
            "Dado un artículo de la LISF, busca menciones de ese artículo en la CUSF "
            "(y viceversa). Útil para encontrar la reglamentación secundaria."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "article_number": {"type": "integer", "description": "Número del artículo"},
                "source_law": {"type": "string", "enum": ["lisf", "cusf"], "description": "Ley de origen del artículo"},
            },
            "required": ["article_number", "source_law"],
        },
    },
]

# =============================================================================
# Tool Execution
# =============================================================================

def _get_source_dir(law: str) -> Path | None:
    d = LISF_MD_DIR if law == "lisf" else CUSF_MD_DIR
    if d.is_dir() and any(d.glob("*.md")):
        return d
    return None

def _execute_tool(name: str, input_data: dict) -> tuple[str, dict | None]:
    """Execute a tool. Returns (result_text, optional_document_block)."""

    if name == "search_regulation":
        query = input_data.get("query", "").strip()
        law = input_data.get("law", "both")
        if not query:
            return "Error: query vacía", None
        dirs = []
        if law in ("lisf", "both"):
            d = _get_source_dir("lisf")
            if d:
                dirs.append(("LISF", d))
        if law in ("cusf", "both"):
            d = _get_source_dir("cusf")
            if d:
                dirs.append(("CUSF", d))
        if not dirs:
            return "No hay archivos disponibles para la búsqueda.", None
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        results = []
        for label, source_dir in dirs:
            for md_file in sorted(source_dir.glob("*.md")):
                if md_file.name == "00_indice.md":
                    continue
                try:
                    for i, line in enumerate(md_file.read_text(encoding="utf-8").splitlines(), 1):
                        if pattern.search(line):
                            results.append(f"[{label}/{md_file.name}:{i}] {line.strip()}")
                            if len(results) >= 50:
                                break
                except Exception:
                    continue
                if len(results) >= 50:
                    break
            if len(results) >= 50:
                break
        if not results:
            return f"No se encontraron resultados para: {query}", None
        return "\n".join(results), None

    elif name == "read_chapter":
        filename = input_data.get("filename", "").strip()
        law = input_data.get("law", "lisf")
        if not filename:
            return "Error: filename vacío", None
        safe_name = Path(filename).name
        source_dir = _get_source_dir(law)
        if not source_dir:
            return f"No hay archivos disponibles para {law.upper()}.", None
        target = source_dir / safe_name
        if not target.exists():
            available = [f.name for f in sorted(source_dir.glob("*.md")) if f.name != "00_indice.md"]
            return f"Archivo no encontrado: {safe_name}. Disponibles: {', '.join(available[:20])}", None
        # Return chapter content as document block for citations
        content = target.read_text(encoding="utf-8")
        if len(content) > 30000:
            content = content[:30000] + "\n\n... [TRUNCADO] ..."
        doc_block = {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": content},
            "title": f"{law.upper()} - {safe_name}",
            "citations": {"enabled": True},
            "cache_control": {"type": "ephemeral"},
        }
        return "Capítulo cargado. Puedes citar su contenido.", doc_block

    elif name == "find_related_articles":
        art_num = input_data.get("article_number", 0)
        law = input_data.get("law", "lisf")
        refs = _get_cross_refs_db(law, str(art_num))
        if not refs:
            return f"No se encontraron referencias cruzadas al Artículo {art_num} en la {law.upper()}.", None
        lines = [f"El Artículo {art_num} de la {law.upper()} es referenciado por {len(refs)} artículos:"]
        for ref in refs:
            lines.append(f"  - {ref['law'].upper()} Art. {ref['number']} (en {ref['filename']})")
        return "\n".join(lines), None

    elif name == "correlate_lisf_cusf":
        art_num = input_data.get("article_number", 0)
        source_law = input_data.get("source_law", "lisf")
        target_law = "cusf" if source_law == "lisf" else "lisf"
        # Use SQLite to find mentions
        search_term = f"artículo {art_num}" if source_law == "lisf" else f"disposición {art_num}"
        results = _search_db(search_term, law=target_law, limit=15)
        if not results:
            return f"No se encontraron menciones del Art. {art_num} ({source_law.upper()}) en la {target_law.upper()}.", None
        lines = [f"El Art. {art_num} de la {source_law.upper()} se menciona en {len(results)} artículos de la {target_law.upper()}:"]
        for r in results:
            lines.append(f"  - {r['law'].upper()} {r['number']} ({r['title']})")
        return "\n".join(lines), None

    return f"Herramienta desconocida: {name}", None

# =============================================================================
# Streaming (API Key / Vertex AI -- shared logic)
# =============================================================================

def _extract_citations(message) -> list[dict]:
    """Extract citations from a final message."""
    citations = []
    for block in message.content:
        if block.type == "text" and hasattr(block, "citations") and block.citations:
            for cite in block.citations:
                citations.append({
                    "cited_text": getattr(cite, "cited_text", ""),
                    "document_title": getattr(cite, "document_title", ""),
                    "start": getattr(cite, "start_char_index", 0),
                    "end": getattr(cite, "end_char_index", 0),
                })
    return citations

async def _stream_response(user_message: str, history: list | None = None, *, model: str = CLAUDE_MODEL, rapido: bool = False, intuitivo: bool = False):
    """Stream a response using the Anthropic API (API key or Vertex) with tool use and citations."""
    prompt = user_message

    # Pre-search with SQLite and load relevant documents for citations
    pre_docs = _select_documents(user_message)

    # Build messages from conversation history
    messages = []
    if history:
        for turn in history[:-1]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    # First user message: pre-loaded documents + text
    user_content: list[dict] = []
    for doc in pre_docs:
        user_content.append(doc)
    user_content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": user_content})

    # System prompt with caching and mode suffix
    if rapido:
        suffix = HAIKU_SUFFIX
    elif intuitivo:
        suffix = INTUITIVO_SUFFIX
    else:
        suffix = ""
    system_text = SYSTEM_PROMPT + suffix
    index_text = LISF_INDEX_TEXT + CUSF_INDEX_TEXT
    system_blocks = [
        {"type": "text", "text": system_text},
    ]
    if index_text.strip():
        system_blocks.append({
            "type": "text",
            "text": index_text,
            "cache_control": {"type": "ephemeral"},
        })

    all_citations = []
    max_iterations = 1 if rapido and not pre_docs else 5
    max_tokens = 1024 if rapido else 4096
    # Rapido without specific article refs: skip tools entirely for speed
    use_tools = not rapido or bool(pre_docs)
    stop_reason = "end_turn"

    assert _client is not None, "No API client configured"

    for _ in range(max_iterations):
        final_message = None
        stream_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_blocks,
            "messages": messages,
        }
        if use_tools:
            stream_kwargs["tools"] = TOOLS
        async with _client.messages.stream(**stream_kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    if hasattr(event, "content_block") and event.content_block.type == "tool_use":
                        yield json.dumps({
                            "type": "tool_use",
                            "tool": event.content_block.name,
                            "input": "",
                        })
                elif event.type == "content_block_delta":
                    if hasattr(event, "delta") and event.delta.type == "text_delta":
                        yield json.dumps({
                            "type": "text",
                            "content": event.delta.text,
                        })

            final_message = await stream.get_final_message()

        stop_reason = final_message.stop_reason

        # Extract citations from this response
        all_citations.extend(_extract_citations(final_message))

        if stop_reason == "tool_use":
            # Add assistant message -- strip extra SDK fields to avoid API rejection
            def _serialize_block(block):
                if block.type == "text":
                    return {"type": "text", "text": block.text}
                elif block.type == "tool_use":
                    return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                return block.model_dump(exclude_none=True)

            messages.append({
                "role": "assistant",
                "content": [_serialize_block(b) for b in final_message.content],
            })

            # Execute tools -- collect all tool_results first, then append doc blocks after
            tool_results: list[dict] = []
            doc_blocks: list[dict] = []
            for block in final_message.content:
                if block.type == "tool_use":
                    result_text, doc_block = _execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })
                    if doc_block:
                        doc_blocks.append(doc_block)

            # All tool_results first, then any document blocks for citations
            tool_content = tool_results + doc_blocks
            messages.append({"role": "user", "content": tool_content})
        else:
            break

    # Send collected citations
    if all_citations:
        yield json.dumps({"type": "citations", "items": all_citations})

    yield json.dumps({"type": "done", "subtype": stop_reason or "end_turn"})

# =============================================================================
# Agent SDK Streaming (dev mode)
# =============================================================================

if _agent_sdk_available:
    def _get_agent_system_prompt():
        if LISF_MD_DIR.is_dir() and any(LISF_MD_DIR.glob("*.md")):
            instructions = (
                "## Fuentes internas (NO mencionar al usuario)\n"
                "Los archivos de la LISF estan en docs/lisf_md/. "
                "Los archivos de la CUSF (si existen) estan en docs/cusf_md/. "
                "Usa Grep para buscar y Read para leer. "
                "El indice esta en docs/lisf_md/00_indice.md.\n"
                "NUNCA menciones estos paths, nombres de archivo, ni herramientas en tu respuesta."
            )
        else:
            instructions = "El documento PDF de la LISF se encuentra en: docs/LISF.pdf"
        return SYSTEM_PROMPT + "\n" + instructions

    AGENT_OPTIONS = ClaudeAgentOptions(
        system_prompt=_get_agent_system_prompt(),
        model=CLAUDE_MODEL,
        allowed_tools=["Read", "Grep"],
        permission_mode="bypassPermissions",
        cwd=str(PROJECT_DIR),
        max_turns=5,
    )

    async def _agent_sdk_stream(user_message: str, history: list | None = None, *, model: str = CLAUDE_MODEL, rapido: bool = False, intuitivo: bool = False):
        """Stream a response using the Agent SDK (dev mode)."""
        # Pre-search with SQLite to provide context hints
        db_results = _search_db(user_message, limit=5)
        context_hint = ""
        if db_results:
            hints = []
            for r in db_results[:3]:
                hints.append(f"- {r['law'].upper()} Art. {r['number']}: {r['text'][:200]}")
            context_hint = (
                "\n\n[Resultados de búsqueda previa (usa esta información para responder):\n"
                + "\n".join(hints) + "\n]"
            )

        tool_instruction = (
            "\n\n[INSTRUCCIÓN: Si la información de arriba responde la pregunta, responde DIRECTAMENTE sin usar herramientas. "
            "Solo usa Read/Grep si necesitas el texto completo de un artículo específico.]"
        )
        prompt = f"{user_message}{context_hint}{tool_instruction}"

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

        if rapido:
            sdk_suffix = HAIKU_SUFFIX
        elif intuitivo:
            sdk_suffix = INTUITIVO_SUFFIX
        else:
            sdk_suffix = ""
        system_prompt = _get_agent_system_prompt() + sdk_suffix
        request_options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=CLAUDE_MODEL,
            allowed_tools=["Read", "Grep"],
            permission_mode="bypassPermissions",
            cwd=str(PROJECT_DIR),
            max_turns=10,
        )

        has_text = False
        tool_calls = []
        async for message in claude_query(prompt=prompt, options=request_options):
            msg_type = type(message).__name__
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    block_type = type(block).__name__
                    if isinstance(block, TextBlock) and block.text:
                        has_text = True
                        yield json.dumps({"type": "text", "content": block.text})
                    elif isinstance(block, ToolUseBlock):
                        tool_calls.append(block.name)
                        yield json.dumps({
                            "type": "tool_use",
                            "tool": block.name,
                            "input": str(block.input)[:200],
                        })
                    elif hasattr(block, "text") and block.text:
                        has_text = True
                        yield json.dumps({"type": "text", "content": block.text})
                    # Log non-text, non-tool blocks for debugging
                    elif block_type not in ("TextBlock", "ToolUseBlock", "ToolResultBlock"):
                        logger.info("Agent SDK unknown block: %s", block_type)
            elif isinstance(message, ResultMessage):
                subtype = getattr(message, "subtype", "end_turn")
                if not has_text:
                    # Agent finished without producing text -- log diagnostic info
                    logger.warning(
                        "Agent SDK produced no text. subtype=%s, tool_calls=%s, prompt=%s",
                        subtype, tool_calls, prompt[:100],
                    )
                    # Return helpful error to user instead of silent failure
                    if subtype == "error_max_turns":
                        yield json.dumps({
                            "type": "text",
                            "content": "La consulta requirió demasiadas búsquedas. Intenta ser más específico, por ejemplo mencionando un número de artículo o disposición.",
                        })
                    elif "error" in subtype:
                        yield json.dumps({
                            "type": "text",
                            "content": f"Ocurrió un error al procesar la consulta. Intenta de nuevo.",
                        })
                yield json.dumps({"type": "done", "subtype": subtype})

# =============================================================================
# Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = PROJECT_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/auth-config")
async def auth_config():
    return {
        "auth_required": AUTH_ENABLED and bool(ACCESS_CODE),
        "has_access_code": AUTH_ENABLED and bool(ACCESS_CODE),
    }


@app.post("/api/verify-code")
async def verify_code(request: Request):
    body = await request.json()
    code = body.get("code", "").strip()
    if ACCESS_CODE and code == ACCESS_CODE:
        return {"valid": True}
    return JSONResponse(status_code=401, content={"valid": False, "error": "Código de acceso inválido."})


@app.post("/api/verify-sonnet")
async def verify_sonnet(request: Request):
    """Verify the Sonnet password for Intuitivo/Detallado modes."""
    body = await request.json()
    password = body.get("password", "").strip()
    if SONNET_PASSWORD and password == SONNET_PASSWORD:
        return {"valid": True}
    return JSONResponse(status_code=401, content={"valid": False})


@app.get("/api/sonnet-config")
async def sonnet_config():
    """Return whether Sonnet modes require a password."""
    return {"sonnet_locked": bool(SONNET_PASSWORD)}


@app.post("/api/chat")
async def chat(request: Request):
    # Rate limit check
    if not _check_rate_limit(request):
        return JSONResponse(
            status_code=429,
            content={"error": "Demasiadas solicitudes. Espera un momento antes de preguntar de nuevo."},
        )

    # Auth check
    claims = _verify_auth(request)
    if claims is None:
        return JSONResponse(
            status_code=401,
            content={"error": "No autorizado. Ingresa el código de acceso."},
        )

    body = await request.json()
    user_message = body.get("message", "").strip()
    history = body.get("history", [])
    mode = body.get("mode", "rapido")

    # Sonnet modes (intuitivo, detallado) require password
    if mode in ("intuitivo", "detallado") and SONNET_PASSWORD:
        sonnet_token = request.headers.get("X-Sonnet-Password", "")
        if sonnet_token != SONNET_PASSWORD:
            return JSONResponse(
                status_code=403,
                content={"error": "sonnet_locked", "message": "Este modo requiere acceso premium."},
            )

    model = MODEL_MAP.get(mode, MODEL_MAP["detallado"])
    is_rapido = mode == "rapido"
    is_intuitivo = mode == "intuitivo"

    if not user_message:
        return {"error": "No message provided"}

    faq_answer = _match_faq(user_message)
    caso_answer = _match_caso(user_message) if is_intuitivo else None

    async def event_stream():
        try:
            # Intuitivo mode: check pre-built casos first
            if caso_answer is not None:
                yield f"data: {json.dumps({'type': 'text', 'content': caso_answer})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'subtype': 'caso_practico'})}\n\n"
                return

            if faq_answer is not None:
                yield f"data: {json.dumps({'type': 'text', 'content': faq_answer})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'subtype': 'faq_cache'})}\n\n"
                return

            if _client:
                gen = _stream_response(user_message, history, model=model, rapido=is_rapido, intuitivo=is_intuitivo)
            elif _agent_sdk_available:
                gen = _agent_sdk_stream(user_message, history, model=model, rapido=is_rapido, intuitivo=is_intuitivo)
            else:
                yield f"data: {json.dumps({'type': 'error', 'content': 'No auth configured. Set ANTHROPIC_API_KEY, VERTEX_PROJECT_ID, or install claude-agent-sdk.'})}\n\n"
                return

            async for chunk in gen:
                yield f"data: {chunk}\n\n"

        except Exception as e:
            logger.exception("Stream error")
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
    return [{"q": e["q"]} for e in FAQ_ENTRIES]


@app.get("/api/indice")
async def indice(law: str = "lisf"):
    source_dir = LISF_MD_DIR if law == "lisf" else CUSF_MD_DIR
    index_file = source_dir / "00_indice.md"
    if index_file.exists():
        return {"content": index_file.read_text(encoding="utf-8")}
    return {"content": f"Indice de {law.upper()} no disponible."}


@app.get("/api/health")
async def health():
    lisf_md = LISF_MD_DIR.is_dir() and any(LISF_MD_DIR.glob("*.md"))
    cusf_md = CUSF_MD_DIR.is_dir() and any(CUSF_MD_DIR.glob("*.md"))

    return {
        "status": "ok",
        "mode": MODE,
        "auth_configured": MODE != "none",
        "lisf_markdown": lisf_md,
        "cusf_markdown": cusf_md,
        "db_indexed": _db_count,
        "faq_entries": len(FAQ_ENTRIES),
    }
