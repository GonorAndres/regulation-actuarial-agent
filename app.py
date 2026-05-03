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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

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
TITULO_ANSWERS_PATH = PROJECT_DIR / "subagents_outputs" / "titulo_answers.json"

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

def _load_titulo_answers() -> list[dict]:
    if TITULO_ANSWERS_PATH.exists():
        try:
            return json.loads(TITULO_ANSWERS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

TITULO_ENTRIES = _load_titulo_answers()
TITULO_NORMALIZED = [(_normalize(e["q"]), i) for i, e in enumerate(TITULO_ENTRIES)]
if TITULO_ENTRIES:
    logger.info("Titulo quick answers: %d entries loaded", len(TITULO_ENTRIES))

def _match_faq(user_message: str) -> str | None:
    norm = _normalize(user_message)
    if not norm:
        return None
    # Check FAQ entries
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
    # Check titulo quick answers
    for nq, idx in TITULO_NORMALIZED:
        if norm == nq:
            return TITULO_ENTRIES[idx]["a"]
    for nq, idx in TITULO_NORMALIZED:
        words_user = set(norm.split())
        words_titulo = set(nq.split())
        if not words_titulo:
            continue
        overlap = len(words_user & words_titulo) / len(words_titulo)
        if overlap >= 0.75 and len(words_user) <= len(words_titulo) + 3:
            return TITULO_ENTRIES[idx]["a"]
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
        params: list = [query]
        law_clause = ""
        if law != "both":
            law_clause = "AND a.law = ?"
            params.append(law)
        params.append(limit)
        rows = _db.execute(f"""
            SELECT a.law, a.number, a.title, a.filename, a.text
            FROM articles_fts f
            JOIN articles a ON a.id = f.rowid
            WHERE articles_fts MATCH ?
            {law_clause}
            ORDER BY bm25(articles_fts, 0.0, 5.0, 1.0, 10.0, 8.0)
            LIMIT ?
        """, params).fetchall()
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
    """Find cross-referenced articles using the cross_refs index table."""
    if not _db:
        return []
    try:
        rows = _db.execute("""
            SELECT cr.to_law AS law, cr.to_number AS number, a.title, a.filename
            FROM cross_refs cr
            JOIN articles a ON a.law = cr.to_law AND a.number = cr.to_number
            WHERE cr.from_law = ? AND cr.from_number = ?
            UNION
            SELECT cr.from_law AS law, cr.from_number AS number, a.title, a.filename
            FROM cross_refs cr
            JOIN articles a ON a.law = cr.from_law AND a.number = cr.from_number
            WHERE cr.to_law = ? AND cr.to_number = ?
            LIMIT 30
        """, (law, number, law, number)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []

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
    """Select relevant documents: by specific article refs OR by FTS5 search.
    When specific articles are found, also includes top cross-references."""
    docs = []
    seen = set()  # (law, number) to avoid duplicates

    # First: check for specific article/disposition references
    for num_str in _ARTICLE_REF.findall(user_message):
        num = str(int(num_str))
        art = _get_article_db("lisf", num)
        if art and ("lisf", num) not in seen:
            docs.append(_article_to_doc_block(art))
            seen.add(("lisf", num))
    for disp_str in _DISP_REF.findall(user_message):
        art = _get_article_db("cusf", disp_str)
        if art and ("cusf", disp_str) not in seen:
            docs.append(_article_to_doc_block(art))
            seen.add(("cusf", disp_str))

    # Auto-include top cross-references for matched articles (max 3 extra)
    if docs and len(docs) <= 3:
        xref_docs = []
        for law, num in list(seen):
            xrefs = _get_cross_refs_db(law, num)
            for xref in xrefs[:2]:
                key = (xref["law"], xref["number"])
                if key not in seen:
                    xart = _get_article_db(xref["law"], xref["number"])
                    if xart:
                        xref_docs.append(_article_to_doc_block(xart))
                        seen.add(key)
                if len(xref_docs) >= 3:
                    break
            if len(xref_docs) >= 3:
                break
        docs.extend(xref_docs)

    # If no specific refs found, do a FTS5 search
    if not docs:
        results = _search_db(user_message, limit=8)
        for r in results[:5]:
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
        results = _search_db(query, law=law, limit=15)
        if not results:
            return f"No se encontraron resultados para: {query}", None
        lines = []
        for r in results:
            snippet = r["text"][:300].replace("\n", " ")
            lines.append(f"[{r['law'].upper()} Art. {r['number']}] ({r['title']}) {snippet}")
        return "\n".join(lines), None

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
        # Use cross_refs table for fast indexed lookup
        if _db:
            try:
                results = _db.execute("""
                    SELECT DISTINCT cr.to_law AS law, cr.to_number AS number, a.title
                    FROM cross_refs cr
                    JOIN articles a ON a.law = cr.to_law AND a.number = cr.to_number
                    WHERE cr.from_law = ? AND cr.from_number = ? AND cr.to_law = ?
                    UNION
                    SELECT DISTINCT cr.from_law AS law, cr.from_number AS number, a.title
                    FROM cross_refs cr
                    JOIN articles a ON a.law = cr.from_law AND a.number = cr.from_number
                    WHERE cr.to_law = ? AND cr.to_number = ? AND cr.from_law = ?
                    LIMIT 20
                """, (source_law, str(art_num), target_law,
                      source_law, str(art_num), target_law)).fetchall()
                results = [dict(r) for r in results]
            except Exception:
                results = []
        else:
            results = []
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


@app.get("/favicon.svg")
async def favicon():
    favicon_path = PROJECT_DIR / "favicon.svg"
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type="image/svg+xml")
    return JSONResponse(status_code=404, content={"error": "not found"})


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


_LISF_INDICE = """# Índice de la LISF

**Ley de Instituciones de Seguros y de Fianzas**
510 artículos organizados en 13 Títulos

| Título | Tema | Artículos |
|--------|------|-----------|
| **I** | Disposiciones preliminares | Arts. 1 - 18 |
| **II** | De las operaciones de seguros y fianzas | Arts. 19 - 40 |
| **III** | De la organización de las instituciones | Arts. 41 - 89 |
| **IV** | De los intermediarios, agentes y ajustadores | Arts. 90 - 117 |
| **V** | Reservas técnicas, inversiones y solvencia | Arts. 118 - 275 |
| **VI** | De los procedimientos de seguros y fianzas | Arts. 276 - 293 |
| **VII** | De las operaciones prohibidas | Arts. 294 - 295 |
| **VIII** | Contabilidad, estados financieros y auditoría | Arts. 296 - 319 |
| **IX** | Planes de regularización e intervención | Arts. 320 - 335 |
| **X** | De las sociedades mutualistas de seguros | Arts. 336 - 365 |
| **XI** | De la CNSF: facultades e inspección | Arts. 366 - 392 |
| **XII** | Liquidación administrativa y concurso mercantil | Arts. 393 - 458 |
| **XIII** | Sanciones, infracciones y delitos | Arts. 459 - 510 |

Publicada en el DOF el 4 de abril de 2013. Última reforma: 14 de noviembre de 2025.
"""

_CUSF_INDICE = """# Índice de la CUSF

**Circular Única de Seguros y Fianzas**
Normativa secundaria que reglamenta la LISF, emitida por la CNSF.
Más de 1,700 disposiciones organizadas en 41 Títulos.

| Título | Tema | Disposiciones |
|--------|------|---------------|
| **1** | Disposiciones preliminares | 1.1.1 - 1.1.3 |
| **2** | Autorizaciones y modificación de estatutos | 2.1.1 - 2.3.7 |
| **3** | Gobierno corporativo | 3.1.1 - 3.11.11 |
| **4** | Productos de seguros y fianzas | 4.1.1 - 4.12.3 |
| **5** | Reservas técnicas | 5.1.1 - 5.20.6 |
| **6** | Requerimiento de capital de solvencia (RCS) | 6.1.1 - 6.10.7 |
| **7** | Fondos propios admisibles y prueba de solvencia | 7.1.1 - 7.4.3 |
| **8** | Régimen de inversiones | 8.1.1 - 8.23.7 |
| **9** | Reaseguro y reafianzamiento | 9.1.1 - 9.7.12 |
| **10** | Obligaciones subordinadas y títulos de crédito | 10.1.1 - 10.6.4 |
| **11** | Garantías de recuperación para fianzas | 11.1.1 - 11.7.4 |
| **12** | Contratación de servicios con terceros | 12.1.1 - 12.3.5 |
| **13** | Operación física y días inhábiles | 13.1.1 - 13.4.1 |
| **14** | Seguros de pensiones | 14.1.1 - 14.6.4 |
| **15** | Seguros de salud | 15.1.1 - 15.9.10 |
| **16** | Seguros de crédito y caución | 16.1.1 - 16.3.1 |
| **17** | Seguros de crédito a la vivienda | 17.1.1 - 17.3.5 |
| **18** | Seguros de garantía financiera | 18.1.1 - 18.3.13 |
| **19** | Fianzas especializadas | 19.1.1 - 19.2.3 |
| **20** | Fondos especiales de seguros y pensiones | 20.1.1 - 20.3.2 |
| **21** | Operaciones análogas y conexas | 21.1.1 - 21.1.5 |
| **22** | Contabilidad | 22.1.1 - 22.7.7 |
| **23** | Auditores externos y actuarios independientes | 23.1.1 - 23.3.1 |
| **24** | Publicación de estados financieros | 24.1.1 - 24.4.4 |
| **25** | Estados financieros de grupos financieros | 25.1.1 - 25.2.1 |
| **26** | Sistema estadístico del sector | 26.1.1 - 26.3.1 |
| **27** | Prevención de lavado de dinero | 27.1.1 - 27.2.1 |
| **28** | Planes de regularización y autocorrección | 28.1.1 - 28.3.5 |
| **29** | Liquidación administrativa y convencional | 29.1.1 - 29.4.6 |
| **30** | Registro de auditores y actuarios | 30.1.1 - 30.6.11 |
| **31** | Acreditación de actuarios | 31.1.1 - 31.2.21 |
| **32** | Agentes de seguros y fianzas | 32.1.1 - 32.13.2 |
| **33** | Personas morales intermediarias | 33.1.1 - 33.5.2 |
| **34** | Reaseguradoras extranjeras | 34.1.1 - 34.4.21 |
| **35** | Intermediarios de reaseguro | 35.1.1 - 35.5.6 |
| **36** | Ajustadores de seguros y fianzas | 36.1.1 - 36.2.3 |
| **37** | Organizaciones aseguradoras y afianzadoras | 37.1.1 - 37.3.4 |
| **38** | Reportes regulatorios (RR-1 a RR-13) | 38.1.1 - 38.1.14 |
| **39** | Entrega electrónica de información | 39.1.1 - 39.6.3 |
| **40** | Fondos de aseguramiento agropecuario | 40.1.1 - 40.3.2 |
| **41** | Modelos novedosos (fintech sandbox) | 41.1.1 - 41.6.2 |

Incluye además disposiciones transitorias con más de 100 artículos transitorios.
"""

_CUSF_TITULO_NAMES: dict[str, str] = {
    "1": "Disposiciones preliminares", "2": "Autorizaciones y modificacion de estatutos",
    "3": "Gobierno corporativo", "4": "Productos de seguros y fianzas",
    "5": "Reservas tecnicas", "6": "Requerimiento de capital de solvencia (RCS)",
    "7": "Fondos propios admisibles y prueba de solvencia", "8": "Regimen de inversiones",
    "9": "Reaseguro y reafianzamiento", "10": "Obligaciones subordinadas y titulos de credito",
    "11": "Garantias de recuperacion para fianzas", "12": "Contratacion de servicios con terceros",
    "13": "Operacion fisica y dias inhabiles", "14": "Seguros de pensiones",
    "15": "Seguros de salud", "16": "Seguros de credito y caucion",
    "17": "Seguros de credito a la vivienda", "18": "Seguros de garantia financiera",
    "19": "Fianzas especializadas", "20": "Fondos especiales de seguros y pensiones",
    "21": "Operaciones analogas y conexas", "22": "Contabilidad",
    "23": "Auditores externos y actuarios independientes", "24": "Publicacion de estados financieros",
    "25": "Estados financieros de grupos financieros", "26": "Sistema estadistico del sector",
    "27": "Prevencion de lavado de dinero", "28": "Planes de regularizacion y autocorreccion",
    "29": "Liquidacion administrativa y convencional", "30": "Registro de auditores y actuarios",
    "31": "Acreditacion de actuarios", "32": "Agentes de seguros y fianzas",
    "33": "Personas morales intermediarias", "34": "Reaseguradoras extranjeras",
    "35": "Intermediarios de reaseguro", "36": "Ajustadores de seguros y fianzas",
    "37": "Organizaciones aseguradoras y afianzadoras", "38": "Reportes regulatorios (RR-1 a RR-13)",
    "39": "Entrega electronica de informacion", "40": "Fondos de aseguramiento agropecuario",
    "41": "Modelos novedosos (fintech sandbox)",
}

_LISF_TITULO_NAMES: dict[str, str] = {
    "I": "Disposiciones preliminares", "II": "De las operaciones de seguros y fianzas",
    "III": "De la organizacion de las instituciones",
    "IV": "De los intermediarios, agentes y ajustadores",
    "V": "Reservas tecnicas, inversiones y solvencia",
    "VI": "De los procedimientos de seguros y fianzas",
    "VII": "De las operaciones prohibidas",
    "VIII": "Contabilidad, estados financieros y auditoria",
    "IX": "Planes de regularizacion e intervencion",
    "X": "De las sociedades mutualistas de seguros",
    "XI": "De la CNSF: facultades e inspeccion",
    "XII": "Liquidacion administrativa y concurso mercantil",
    "XIII": "Sanciones, infracciones y delitos",
}

_TITULO_WORD_TO_ROMAN = {
    "PRIMERO": "I", "SEGUNDO": "II", "TERCERO": "III", "CUARTO": "IV",
    "QUINTO": "V", "SEXTO": "VI", "SÉPTIMO": "VII", "SEPTIMO": "VII",
    "OCTAVO": "VIII", "NOVENO": "IX", "DÉCIMO": "X", "DECIMO": "X",
}
_CAP_WORD_TO_NUM = {
    "ÚNICO": "1", "UNICO": "1", "PRIMERO": "1", "SEGUNDO": "2", "TERCERO": "3",
    "CUARTO": "4", "QUINTO": "5", "SEXTO": "6", "SÉPTIMO": "7", "SEPTIMO": "7",
    "OCTAVO": "8", "NOVENO": "9", "DÉCIMO": "10", "DECIMO": "10",
    "DÉCIMO PRIMERO": "11", "DECIMO PRIMERO": "11",
}

def _parse_lisf_title(title: str) -> tuple[str, str]:
    """Parse 'TÍTULO QUINTO - CAPÍTULO SEGUNDO' into ('V', '2')."""
    parts = title.split(" - ")
    t_part = parts[0].replace("TÍTULO ", "").strip() if len(parts) >= 1 else ""
    c_part = parts[1].replace("CAPÍTULO ", "").replace("CAPITULO ", "").strip() if len(parts) >= 2 else ""

    # Handle compound titles like "DÉCIMO PRIMERO"
    t_roman = _TITULO_WORD_TO_ROMAN.get(t_part, "")
    if not t_roman and " " in t_part:
        base = t_part.split()[0]
        suffix = " ".join(t_part.split()[1:])
        base_val = _TITULO_WORD_TO_ROMAN.get(base, "")
        suffix_val = _TITULO_WORD_TO_ROMAN.get(suffix, "")
        if base_val == "X" and suffix_val:
            roman_map = {"I": "XI", "II": "XII", "III": "XIII"}
            t_roman = roman_map.get(suffix_val, base_val)

    c_num = _CAP_WORD_TO_NUM.get(c_part, "")
    if not c_num and " " in c_part:
        c_num = _CAP_WORD_TO_NUM.get(c_part, "1")

    return t_roman or t_part, c_num or "1"

_lisf_tree_cache: dict | None = None

def _build_lisf_tree() -> dict:
    global _lisf_tree_cache
    if _lisf_tree_cache:
        return _lisf_tree_cache
    if not _db:
        return {"titulos": [], "transitorios": []}

    rows = _db.execute(
        "SELECT number, title FROM articles WHERE law='lisf' ORDER BY rowid"
    ).fetchall()

    titulos_map: dict[str, dict] = {}
    transitorios = []
    roman_order = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII"]

    for r in rows:
        num, title = r["number"], r["title"]
        if num.startswith("trans"):
            transitorios.append({"numero": num, "titulo": title})
            continue
        t_roman, c_num = _parse_lisf_title(title)
        if not t_roman:
            continue

        if t_roman not in titulos_map:
            titulos_map[t_roman] = {
                "numero": t_roman,
                "nombre": _LISF_TITULO_NAMES.get(t_roman, f"Titulo {t_roman}"),
                "capitulos": {},
            }
        cap_key = f"{t_roman}.{c_num}"
        caps = titulos_map[t_roman]["capitulos"]
        if cap_key not in caps:
            caps[cap_key] = {"numero": cap_key, "disposiciones": []}
        caps[cap_key]["disposiciones"].append({"numero": num, "titulo": title})

    titulos = []
    for t_roman in roman_order:
        if t_roman not in titulos_map:
            continue
        t = titulos_map[t_roman]
        sorted_caps = sorted(t["capitulos"].values(), key=lambda c: int(c["numero"].split(".")[-1]))
        for cap in sorted_caps:
            cap["disposiciones"].sort(key=lambda d: int(d["numero"]) if d["numero"].isdigit() else 0)
        t["capitulos"] = sorted_caps
        titulos.append(t)

    _lisf_tree_cache = {"titulos": titulos, "transitorios": transitorios}
    return _lisf_tree_cache

_cusf_tree_cache: dict | None = None

def _build_cusf_tree() -> dict:
    global _cusf_tree_cache
    if _cusf_tree_cache:
        return _cusf_tree_cache
    if not _db:
        return {"titulos": [], "transitorios": []}

    rows = _db.execute(
        "SELECT number, title FROM articles WHERE law='cusf' ORDER BY rowid"
    ).fetchall()

    titulos_map: dict[str, dict] = {}
    transitorios = []

    for r in rows:
        num, title = r["number"], r["title"]
        if num.startswith("trans"):
            transitorios.append({"numero": num, "titulo": title})
            continue
        parts = num.split(".")
        if len(parts) < 3:
            continue
        t_num = parts[0]
        cap_num = f"{parts[0]}.{parts[1]}"

        if t_num not in titulos_map:
            titulos_map[t_num] = {
                "numero": t_num,
                "nombre": _CUSF_TITULO_NAMES.get(t_num, f"Titulo {t_num}"),
                "capitulos": {},
            }
        caps = titulos_map[t_num]["capitulos"]
        if cap_num not in caps:
            caps[cap_num] = {"numero": cap_num, "disposiciones": []}
        caps[cap_num]["disposiciones"].append({"numero": num, "titulo": title})

    def _sort_key(n: str) -> list[int]:
        try:
            return [int(x) for x in n.split(".")]
        except ValueError:
            return [0]

    titulos = []
    for t_num in sorted(titulos_map, key=lambda x: _sort_key(x)):
        t = titulos_map[t_num]
        sorted_caps = []
        for c_num in sorted(t["capitulos"], key=lambda x: _sort_key(x)):
            cap = t["capitulos"][c_num]
            cap["disposiciones"].sort(key=lambda d: _sort_key(d["numero"]))
            sorted_caps.append(cap)
        t["capitulos"] = sorted_caps
        titulos.append(t)

    _cusf_tree_cache = {"titulos": titulos, "transitorios": transitorios}
    return _cusf_tree_cache


@app.get("/explorer", response_class=HTMLResponse)
@app.get("/cusf", response_class=HTMLResponse)
@app.get("/lisf", response_class=HTMLResponse)
async def serve_explorer():
    html_path = PROJECT_DIR / "cusf.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/explorer/tree/{law}")
async def explorer_tree(law: str):
    if law == "lisf":
        return _build_lisf_tree()
    return _build_cusf_tree()


@app.get("/api/explorer/article/{law}/{number:path}")
async def explorer_article(law: str, number: str):
    if law not in ("lisf", "cusf"):
        return JSONResponse(status_code=400, content={"error": "law must be lisf or cusf"})
    article = _get_article_db(law, number)
    if not article:
        return JSONResponse(status_code=404, content={"error": f"Articulo {number} no encontrado"})
    cross_refs = _get_cross_refs_db(law, number)
    return {
        "law": law,
        "number": article["number"],
        "title": article["title"],
        "text": article["text"],
        "cross_refs": cross_refs,
    }


@app.get("/api/indice")
async def indice(law: str = "lisf"):
    if law == "cusf":
        return {"content": _CUSF_INDICE}
    return {"content": _LISF_INDICE}


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
