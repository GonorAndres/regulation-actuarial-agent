"""Retrieval module for LISF/CUSF regulatory database.

STANDALONE MODULE -- This is the HF Space retrieval system.
It is NOT shared with the Cloud Run app (app.py at repo root).
Changes here do NOT affect Cloud Run. The two systems use
intentionally different retrieval strategies:
  - HF Space: single-pass pre-computed RAG (this module)
  - Cloud Run: iterative tool-calling RAG with Claude

Dependencies: Python stdlib only (sqlite3, json, re, unicodedata).
No Anthropic, no HuggingFace imports.
"""

import json
import re
import sqlite3
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex patterns for detecting article/disposition references in user messages
# ---------------------------------------------------------------------------
_ARTICLE_REF = re.compile(r"(?:art[ií]culo|art\.?)\s*(\d+)", re.IGNORECASE)
_DISP_REF = re.compile(r"(?:disposici[oó]n|disp\.?)\s*(\d+\.\d+\.\d+)", re.IGNORECASE)
_LAW_MENTION = re.compile(r"\b(lisf|cusf)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def open_db(db_path: str | Path) -> sqlite3.Connection | None:
    """Open the regulation SQLite database."""
    db_path = Path(db_path)
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


_STOP_WORDS = {
    "que", "es", "son", "la", "el", "los", "las", "de", "del", "en", "un",
    "una", "por", "para", "con", "se", "su", "al", "lo", "como", "mas",
    "pero", "sus", "le", "ya", "este", "si", "no", "muy", "sin", "sobre",
    "tambien", "me", "hasta", "hay", "donde", "quien", "desde", "nos",
    "durante", "uno", "ni", "otros", "ese", "eso", "ante", "ellos", "cual",
    "fueron", "ser", "tiene", "era", "entre", "asi", "cuando", "todo",
    "esta", "fue", "puede", "todos", "estas", "esto", "cuales", "dice",
    "segun", "tiene", "tienen", "cuantos", "cuantas",
}


def _sanitize_fts_query(query: str) -> str:
    """Clean a natural language query for FTS5 MATCH.

    Strips punctuation, stop words, and joins remaining terms with OR
    for broader matching.
    """
    text = unicodedata.normalize("NFD", query.lower())
    text = "".join(c for c in text if unicodedata.category(c) not in ("Mn",))
    tokens = re.findall(r"\w+", text)
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    if not tokens:
        return ""
    return " OR ".join(tokens)


def _detect_law(user_message: str) -> str:
    """Detect if the user is asking about a specific law (LISF or CUSF).

    Returns 'lisf', 'cusf', or 'both'.
    """
    mentions = [m.lower() for m in _LAW_MENTION.findall(user_message)]
    if mentions:
        unique = set(mentions)
        if len(unique) == 1:
            return unique.pop()
    return "both"


def search_db(conn: sqlite3.Connection, query: str, law: str = "both", limit: int = 10) -> list[dict]:
    """Full-text search using FTS5 with BM25 ranking."""
    if not conn:
        return []
    sanitized = _sanitize_fts_query(query)
    if not sanitized:
        return []
    try:
        law_clause = "" if law == "both" else f"AND a.law = '{law}'"
        rows = conn.execute(f"""
            SELECT a.law, a.number, a.title, a.filename, a.text,
                   a.context_summary
            FROM articles_fts f
            JOIN articles a ON a.id = f.rowid
            WHERE articles_fts MATCH ?
            {law_clause}
            ORDER BY bm25(articles_fts, 0.0, 5.0, 1.0, 10.0, 8.0)
            LIMIT ?
        """, (sanitized, limit)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_article(conn: sqlite3.Connection, law: str, number: str) -> dict | None:
    """Get a specific article/disposition by law and number."""
    if not conn:
        return None
    row = conn.execute(
        "SELECT law, number, title, filename, text, context_summary "
        "FROM articles WHERE law = ? AND number = ?",
        (law, number),
    ).fetchone()
    return dict(row) if row else None


def get_cross_refs(conn: sqlite3.Connection, law: str, number: str) -> list[dict]:
    """Find cross-referenced articles in both directions."""
    if not conn:
        return []
    try:
        rows = conn.execute("""
            SELECT cr.to_law AS law, cr.to_number AS number, a.title
            FROM cross_refs cr
            JOIN articles a ON a.law = cr.to_law AND a.number = cr.to_number
            WHERE cr.from_law = ? AND cr.from_number = ?
            UNION
            SELECT cr.from_law AS law, cr.from_number AS number, a.title
            FROM cross_refs cr
            JOIN articles a ON a.law = cr.from_law AND a.number = cr.from_number
            WHERE cr.to_law = ? AND cr.to_number = ?
            LIMIT 30
        """, (law, number, law, number)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _format_article(article: dict, max_chars: int, use_summary: bool = False) -> str:
    """Format a single article for injection into the prompt.

    If use_summary=True and context_summary exists, uses summary + truncated text
    to fit more articles within the budget.
    """
    law_label = article["law"].upper()
    header = f"--- {law_label} Art. {article['number']} - {article.get('title', '')} ---"

    summary = article.get("context_summary") or ""
    text = article.get("text", "")

    if use_summary and summary:
        body = f"Resumen: {summary}\n\n"
        remaining = max_chars - len(body)
        if remaining > 200 and text:
            body += text[:remaining] + "\n... [texto truncado]"
        return f"{header}\n{body}"

    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [texto truncado]"
    return f"{header}\n{text}"


# ---------------------------------------------------------------------------
# Structural / metadata query handler
# ---------------------------------------------------------------------------

_TITULO_REF = re.compile(
    r"t[ií]tulo\s+(\d+)\s+(?:de\s+la\s+)?(lisf|cusf)",
    re.IGNORECASE,
)
_TITULO_REF_NOLAW = re.compile(r"t[ií]tulo\s+(\d+)", re.IGNORECASE)
_CAPITULO_REF = re.compile(
    r"cap[ií]tulo\s+(\d+\.\d+)\s+(?:de\s+la\s+)?(lisf|cusf)",
    re.IGNORECASE,
)
_FULL_LAW_STRUCTURE = re.compile(
    r"estructura\s+(?:de\s+la\s+)?(lisf|cusf)", re.IGNORECASE
)

# LISF uses Spanish ordinals in title field; CUSF uses numbers
_NUM_TO_ORDINAL = {
    "1": "PRIMERO", "2": "SEGUNDO", "3": "TERCERO", "4": "CUARTO",
    "5": "QUINTO", "6": "SEXTO", "7": "SÉPTIMO", "8": "OCTAVO",
    "9": "NOVENO", "10": "DÉCIMO", "11": "DÉCIMO PRIMERO",
    "12": "DÉCIMO SEGUNDO", "13": "DÉCIMO TERCERO",
}


def _natural_sort_key(title: str) -> tuple:
    """Sort titles naturally: TÍTULO 3 - CAPÍTULO 3.2 before 3.10."""
    nums = re.findall(r"\d+", title)
    return tuple(int(n) for n in nums) if nums else (0,)


def _titulo_pattern(law: str, titulo_num: str) -> str:
    """Build SQL LIKE pattern for a titulo, handling LISF ordinals vs CUSF numbers."""
    if law == "lisf" and titulo_num in _NUM_TO_ORDINAL:
        return f"TÍTULO {_NUM_TO_ORDINAL[titulo_num]} -%"
    return f"TÍTULO {titulo_num} -%"


def _structural_context(conn: sqlite3.Connection, user_message: str) -> str:
    """Handle structural/metadata questions about titulos, chapters, and full law structure.

    Detects questions like:
    - 'Cuantos articulos tiene el titulo 3 de la CUSF?'
    - 'Estructura de la LISF'
    - 'Capitulos del titulo 5 de la LISF'
    Returns DB metadata summaries instead of FTS5 text search.
    """
    parts = []

    # Check for full law structure overview ("estructura de la LISF")
    full_match = _FULL_LAW_STRUCTURE.search(user_message)
    if full_match:
        law = full_match.group(1).lower()
        rows = conn.execute(
            "SELECT title, COUNT(*) as cnt FROM articles "
            "WHERE law = ? GROUP BY title ORDER BY title",
            (law,),
        ).fetchall()
        if rows:
            law_label = law.upper()
            total = sum(r[1] for r in rows)
            # Group by titulo
            titulos: dict[str, list] = {}
            for r in rows:
                titulo = r[0].split(" - ")[0] if " - " in r[0] else r[0]
                titulos.setdefault(titulo, []).append((r[0], r[1]))
            summary = f"--- Estructura completa de la {law_label} ---\n"
            summary += f"Total: {total} articulos/disposiciones en {len(titulos)} titulos\n\n"
            for titulo in sorted(titulos.keys(), key=_natural_sort_key):
                chapters = titulos[titulo]
                titulo_total = sum(c[1] for c in chapters)
                summary += f"**{titulo}** ({titulo_total} articulos, {len(chapters)} capitulos)\n"
                for ch_title, ch_count in sorted(chapters, key=lambda x: _natural_sort_key(x[0])):
                    cap_name = ch_title.split(" - ")[-1] if " - " in ch_title else ch_title
                    summary += f"  - {cap_name}: {ch_count}\n"
                summary += "\n"
            parts.append(summary)
        return "\n\n".join(parts)

    # Check for titulo + law references
    for match in _TITULO_REF.finditer(user_message):
        titulo_num = match.group(1)
        law = match.group(2).lower()
        pattern = _titulo_pattern(law, titulo_num)
        rows = conn.execute(
            "SELECT title, COUNT(*) as cnt FROM articles "
            "WHERE law = ? AND title LIKE ? GROUP BY title ORDER BY title",
            (law, pattern),
        ).fetchall()
        if rows:
            law_label = law.upper()
            total = sum(r[1] for r in rows)
            sorted_rows = sorted(rows, key=lambda r: _natural_sort_key(r[0]))
            summary = f"--- Estructura del Titulo {titulo_num} de la {law_label} ---\n"
            summary += f"Total: {total} disposiciones en {len(rows)} capitulos\n\n"
            for r in sorted_rows:
                summary += f"- {r[0]}: {r[1]} disposiciones\n"

            sample = conn.execute(
                "SELECT number, text FROM articles "
                "WHERE law = ? AND title LIKE ? ORDER BY rowid LIMIT 3",
                (law, pattern),
            ).fetchall()
            if sample:
                summary += "\nPrimeras disposiciones:\n"
                for s in sample:
                    text_preview = s[1][:200] if s[1] else ""
                    summary += f"- {law_label} Disp. {s[0]}: {text_preview}...\n"
            parts.append(summary)

    # If no explicit law, try both
    if not parts:
        for match in _TITULO_REF_NOLAW.finditer(user_message):
            titulo_num = match.group(1)
            for law in ["lisf", "cusf"]:
                pattern = _titulo_pattern(law, titulo_num)
                rows = conn.execute(
                    "SELECT title, COUNT(*) as cnt FROM articles "
                    "WHERE law = ? AND title LIKE ? GROUP BY title ORDER BY title",
                    (law, pattern),
                ).fetchall()
                if rows:
                    law_label = law.upper()
                    total = sum(r[1] for r in rows)
                    sorted_rows = sorted(rows, key=lambda r: _natural_sort_key(r[0]))
                    summary = f"--- Estructura del Titulo {titulo_num} de la {law_label} ---\n"
                    summary += f"Total: {total} articulos/disposiciones en {len(sorted_rows)} capitulos\n\n"
                    for r in sorted_rows:
                        summary += f"- {r[0]}: {r[1]} disposiciones\n"
                    parts.append(summary)

    # Check for capitulo references
    for match in _CAPITULO_REF.finditer(user_message):
        cap_num = match.group(1)
        law = match.group(2).lower()
        cap_pattern = f"%CAPÍTULO {cap_num}"
        rows = conn.execute(
            "SELECT number, title, text FROM articles "
            "WHERE law = ? AND title LIKE ? ORDER BY rowid LIMIT 10",
            (law, cap_pattern),
        ).fetchall()
        if rows:
            law_label = law.upper()
            summary = f"--- Capitulo {cap_num} de la {law_label} ({len(rows)} disposiciones) ---\n\n"
            for r in rows:
                text_preview = r[2][:300] if r[2] else ""
                summary += f"- {law_label} Disp. {r[0]}: {text_preview}...\n\n"
            parts.append(summary)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main context selection
# ---------------------------------------------------------------------------

def select_context(conn: sqlite3.Connection, user_message: str, max_chars: int = 12000) -> str:
    """Select relevant articles and format as context string.

    Strategy:
    0. Check for structural/metadata questions about titulos/chapters
    1. Regex-detect explicit article references -> direct lookup (full text)
    2. Auto-include top cross-references for matched articles
    3. Fallback to FTS5 search (law-aware) if no explicit refs found
    4. Smart budgeting: explicit refs get full text, FTS results get summaries
    """
    if not conn:
        return ""

    # 0. Structural queries (titulo/chapter metadata)
    structural = _structural_context(conn, user_message)
    if structural:
        return structural[:max_chars]

    articles = []
    seen: set[tuple[str, str]] = set()
    explicit_count = 0  # Track how many are direct refs vs FTS

    # 1. Check for explicit article/disposition references
    for num_str in _ARTICLE_REF.findall(user_message):
        num = str(int(num_str))
        art = get_article(conn, "lisf", num)
        if art and ("lisf", num) not in seen:
            articles.append(art)
            seen.add(("lisf", num))
            explicit_count += 1

    for disp_str in _DISP_REF.findall(user_message):
        art = get_article(conn, "cusf", disp_str)
        if art and ("cusf", disp_str) not in seen:
            articles.append(art)
            seen.add(("cusf", disp_str))
            explicit_count += 1

    # 2. Auto-include cross-references (max 3 extra)
    if articles and len(articles) <= 3:
        xref_articles = []
        for law, num in list(seen):
            xrefs = get_cross_refs(conn, law, num)
            for xref in xrefs[:2]:
                key = (xref["law"], xref["number"])
                if key not in seen:
                    xart = get_article(conn, xref["law"], xref["number"])
                    if xart:
                        xref_articles.append(xart)
                        seen.add(key)
                if len(xref_articles) >= 3:
                    break
            if len(xref_articles) >= 3:
                break
        articles.extend(xref_articles)

    # 3. Fallback: law-aware FTS5 search
    if not articles:
        detected_law = _detect_law(user_message)
        results = search_db(conn, user_message, law=detected_law, limit=5)
        articles = results[:5]

    if not articles:
        return ""

    # 4. Smart budgeting: explicit refs get more space, FTS/xrefs get summaries
    parts = []
    if explicit_count > 0:
        # Explicit refs: 70% of budget, rest: 30%
        explicit_budget = int(max_chars * 0.7) // max(explicit_count, 1)
        other_budget = int(max_chars * 0.3) // max(len(articles) - explicit_count, 1)
        for i, art in enumerate(articles):
            if i < explicit_count:
                parts.append(_format_article(art, explicit_budget, use_summary=False))
            else:
                parts.append(_format_article(art, other_budget, use_summary=True))
    else:
        # All FTS results: equal budget with summaries
        per_article = max_chars // max(len(articles), 1)
        parts = [_format_article(a, per_article, use_summary=True) for a in articles]

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# FAQ / cache matching
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Normalize text for fuzzy matching (strip accents, punctuation, lowercase)."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) not in ("Mn",))
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def load_json_cache(path: str | Path) -> list[dict]:
    """Load a JSON cache file (FAQ, titulo answers, casos practicos)."""
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def build_normalized_index(entries: list[dict]) -> list[tuple[str, int]]:
    """Build normalized question index for fuzzy matching."""
    return [(normalize(e["q"]), i) for i, e in enumerate(entries)]


def match_cache(entries: list[dict], normalized_index: list[tuple[str, int]], user_message: str) -> str | None:
    """Match user message against a cache (FAQ, titulo, casos).

    Returns the cached answer if match found, None otherwise.
    Uses exact match first, then 75% word overlap threshold.
    """
    norm = normalize(user_message)
    if not norm:
        return None

    # Exact match
    for nq, idx in normalized_index:
        if norm == nq:
            return entries[idx]["a"]

    # Fuzzy match: 75% word overlap
    words_user = set(norm.split())
    for nq, idx in normalized_index:
        words_cache = set(nq.split())
        if not words_cache:
            continue
        overlap = len(words_user & words_cache) / len(words_cache)
        if overlap >= 0.75 and len(words_user) <= len(words_cache) + 3:
            return entries[idx]["a"]

    return None
