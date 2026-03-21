#!/usr/bin/env python3
"""
convert_pdf.py -- Convert LISF.pdf to markdown chapters.

One-time script. Reads docs/LISF.pdf, converts to markdown via pymupdf4llm,
splits by TITULO/CAPITULO boundaries, outputs to docs/lisf_md/.

Usage:
    pip install pymupdf4llm
    python convert_pdf.py
"""
import re
import sys
from pathlib import Path

try:
    import pymupdf4llm
except ImportError:
    print("ERROR: pymupdf4llm not installed. Run: pip install pymupdf4llm")
    sys.exit(1)

PDF_PATH = Path(__file__).parent / "docs" / "LISF.pdf"
OUTPUT_DIR = Path(__file__).parent / "docs" / "lisf_md"

# Spanish ordinals for matching
ORDINALS = (
    "PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|"
    "S[EÉ]PTIMO|OCTAVO|NOVENO|D[EÉ]CIMO|"
    "UND[EÉ]CIMO|DUO?D[EÉ]CIMO|"
    "DÉCIMO\\s+PRIMERO|DÉCIMO\\s+SEGUNDO|DÉCIMO\\s+TERCERO|"
    "DÉCIMO\\s+CUARTO|DÉCIMO\\s+QUINTO|DÉCIMO\\s+SEXTO|"
    "DÉCIMO\\s+SÉPTIMO|DÉCIMO\\s+OCTAVO|"
    "PRIMERA|SEGUNDA|TERCERA|CUARTA|QUINTA|SEXTA|"
    "S[EÉ]PTIMA|OCTAVA|NOVENA|D[EÉ]CIMA"
)

# Regex patterns for structural elements
# The PDF converts to bold markdown: **TITULO PRIMERO**, **CAPITULO UNICO**, etc.
TITULO_RE = re.compile(
    rf"^\*\*T[IÍ]TULO\s+({ORDINALS})\*\*",
    re.IGNORECASE | re.MULTILINE,
)
CAPITULO_RE = re.compile(
    rf"^\*\*CAP[IÍ]TULO\s+({ORDINALS}|[ÚU]NICO)\*\*",
    re.IGNORECASE | re.MULTILINE,
)
ARTICULO_RE = re.compile(
    r"\*?\*?ART[IÍ]CULO\s+(\d+)",
    re.IGNORECASE,
)
# Transitorios section
TRANSITORIOS_RE = re.compile(
    r"^\*\*ART[IÍ]CULOS?\s+TRANSITORIOS?\*\*",
    re.IGNORECASE | re.MULTILINE,
)


def clean_markdown(text: str) -> str:
    """
    Remove PDF extraction artifacts from the raw markdown:
    - Repeated page headers/footers (CAMARA DE DIPUTADOS... / LEY DE INSTITUCIONES...)
    - Embedded page numbers (e.g. "1 de 262")
    - Ellipsis artifacts (middle-dot sequences)
    - Orphaned bold punctuation (e.g. **.**)
    - Excessive consecutive blank lines (collapse to max 2)
    """
    lines = text.split("\n")
    cleaned = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # --- Skip repeated page header block ---
        # Pattern: "**CAMARA DE DIPUTADOS DEL H.** **CONGRESO DE LA UNION**"
        #          "Secretaria General"
        #          "Secretaria de Servicios Parlamentarios"
        if re.match(r"^\*\*C[AÁ]MARA DE DIPUTADOS", stripped, re.IGNORECASE):
            # Skip this line and the next few lines that are part of the header
            j = i + 1
            while j < len(lines) and j < i + 4:
                s = lines[j].strip()
                if s.startswith("Secretaría") or s.startswith("Secretaria") or s == "":
                    j += 1
                else:
                    break
            i = j
            continue

        # --- Skip repeated page footer block ---
        # Pattern: "**LEY DE INSTITUCIONES DE SEGUROS Y DE FIANZAS**"
        #          ""
        #          "_Ultima Reforma DOF ..."
        if re.match(r"^\*\*LEY DE INSTITUCIONES DE SEGUROS Y DE FIANZAS\*\*", stripped, re.IGNORECASE):
            j = i + 1
            while j < len(lines) and j < i + 4:
                s = lines[j].strip()
                if s == "" or re.match(r"^_[UÚ]ltima\s+Reforma", s, re.IGNORECASE):
                    j += 1
                else:
                    break
            i = j
            continue

        # --- Skip standalone page numbers: "N de 262" ---
        if re.match(r"^\d{1,3}\s+de\s+262$", stripped):
            i += 1
            continue

        # --- Remove ellipsis artifacts (middle-dot sequences) ---
        if re.match(r"^[.·…]+$", stripped.replace("…", "...").replace("·", ".")):
            # Line is only dots/ellipsis chars
            if len(stripped) >= 3:
                i += 1
                continue

        # --- Fix orphaned bold punctuation like **.**  -> . ---
        line = re.sub(r"\*\*\s*\.\s*\*\*", ".", line)
        # Also fix **"** -> "
        line = re.sub(r'\*\*\s*"\s*\*\*', '"', line)

        cleaned.append(line)
        i += 1

    # --- Collapse 3+ consecutive blank lines to 2 ---
    result = []
    blank_count = 0
    for line in cleaned:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)

    # --- Rejoin sentences broken by page-break gaps ---
    # When a non-empty line does NOT end with sentence-terminating punctuation
    # and is followed by exactly 2 blank lines then a continuation line,
    # collapse the blank lines to rejoin the sentence.
    result = _rejoin_broken_sentences(result)

    return "\n".join(result)


# Punctuation that signals end of paragraph / logical break
_SENTENCE_ENDINGS = re.compile(r"[.;:)\]]$|_$|\*\*$")
# Lines that are structural headers (new article, fraction, section) -- never merge into prev
_STRUCTURAL_LINE = re.compile(
    r"^\*\*(ART[IÍ]CULO|SECCI[OÓ]N|CAP[IÍ]TULO|T[IÍ]TULO|TRANSITORI|DISPOSICI)",
    re.IGNORECASE,
)


def _rejoin_broken_sentences(lines: list[str]) -> list[str]:
    """
    Find double-blank-line gaps that sit in the middle of a sentence
    (previous line doesn't end with sentence punctuation, next line is
    a lowercase continuation) and collapse them so the text flows.
    """
    result = []
    i = 0
    n = len(lines)
    while i < n:
        # Check for pattern: non-blank, blank, blank, non-blank
        if (
            i + 3 < n
            and lines[i].strip() != ""
            and lines[i + 1].strip() == ""
            and lines[i + 2].strip() == ""
            and lines[i + 3].strip() != ""
        ):
            prev_stripped = lines[i].strip()
            next_stripped = lines[i + 3].strip()

            # Is this a mid-sentence break?
            prev_ends_sentence = bool(_SENTENCE_ENDINGS.search(prev_stripped))
            next_is_structural = bool(_STRUCTURAL_LINE.match(next_stripped))

            if not prev_ends_sentence and not next_is_structural:
                # Mid-sentence gap: emit the current line, skip blanks, continue
                result.append(lines[i])
                i += 3  # skip the 2 blank lines, next iteration picks up continuation
                continue

        result.append(lines[i])
        i += 1

    return result


def slugify(text: str) -> str:
    """Convert text to a filename-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[áà]", "a", text)
    text = re.sub(r"[éè]", "e", text)
    text = re.sub(r"[íì]", "i", text)
    text = re.sub(r"[óò]", "o", text)
    text = re.sub(r"[úù]", "u", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text


def find_articles(text: str) -> list[int]:
    """Find all article numbers mentioned in a text block."""
    return sorted(set(int(m) for m in ARTICULO_RE.findall(text)))


# Patterns for extracting subtitle and section names from content
_SUBTITLE_RE = re.compile(
    r"^\*\*(DE\s+(?:LOS|LAS|LA|EL)\s+.+?)\*\*",
    re.IGNORECASE | re.MULTILINE,
)
_SECCION_NAME_RE = re.compile(
    r"^\*\*SECCI[OÓ]N\s+[IVX]+\*\*\s*\n\*\*(.+?)\*\*",
    re.IGNORECASE | re.MULTILINE,
)


def generate_header(title: str, content: str, articles: list[int]) -> str:
    """
    Generate a keyword/metadata header block to prepend to each chapter file.
    This helps Claude quickly identify what the file contains.
    """
    lines = []
    lines.append("---")

    # Title
    clean_title = title.replace("**", "").strip()
    lines.append(f"titulo: {clean_title}")

    # Subtitle (first "DE LOS/LAS/LA/EL..." bold line)
    subtitles = _SUBTITLE_RE.findall(content)
    if subtitles:
        lines.append(f"tema: {subtitles[0].strip()}")

    # Sections within the chapter
    section_names = _SECCION_NAME_RE.findall(content)
    if section_names:
        secs = [s.strip() for s in section_names]
        lines.append(f"secciones: {'; '.join(secs)}")

    # Article range
    if articles:
        lines.append(f"articulos: {articles[0]}-{articles[-1]}")
        lines.append(f"total_articulos: {len(articles)}")

    # Extract key legal concepts as keywords
    keywords = _extract_keywords(content)
    if keywords:
        lines.append(f"palabras_clave: {', '.join(keywords)}")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _extract_keywords(text: str) -> list[str]:
    """Extract frequently referenced legal concepts from the text."""
    # Key legal terms to look for
    term_patterns = [
        ("reservas tecnicas", r"reservas?\s+t[eé]cnicas?"),
        ("prima", r"\bprimas?\b"),
        ("reaseguro", r"\breaseguros?\b"),
        ("reafianzamiento", r"\breafianzamientos?\b"),
        ("nota tecnica", r"notas?\s+t[eé]cnicas?"),
        ("capital minimo", r"capital\s+m[ií]nimo"),
        ("solvencia", r"\bsolvencia\b"),
        ("fianza", r"\bfianzas?\b"),
        ("asegurado", r"\basegurados?\b"),
        ("beneficiario", r"\bbeneficiarios?\b"),
        ("poliza", r"\bp[oó]lizas?\b"),
        ("coaseguro", r"\bcoaseguros?\b"),
        ("reaseguradoras", r"\breaseguradoras?\b"),
        ("CNSF", r"\bComisi[oó]n\b"),
        ("liquidacion", r"\bliquidaci[oó]n\b"),
        ("revocacion", r"\brevocaci[oó]n\b"),
        ("infracciones", r"\binfracciones?\b"),
        ("delitos", r"\bdelitos?\b"),
        ("sanciones", r"\bsanciones?\b"),
        ("multas", r"\bmultas?\b"),
        ("capital contable", r"capital\s+contable"),
        ("fondos propios", r"fondos\s+propios"),
        ("requerimiento de capital", r"requerimiento\s+de\s+capital"),
        ("inversion", r"\binversi[oó]n\b"),
        ("filiales", r"\bfiliales?\b"),
        ("gobierno corporativo", r"gobierno\s+corporativo"),
        ("contralor normativo", r"contralor\s+normativo"),
        ("auditoria", r"\bauditor[ií]a\b"),
        ("actuario", r"\bactuarios?\b"),
        ("ramos de seguro", r"ramos?\s+de\s+seguros?"),
        ("operaciones de seguro", r"operaciones?\s+de\s+seguros?"),
        ("estados financieros", r"estados?\s+financieros?"),
        ("contabilidad", r"\bcontabilidad\b"),
        ("transitorios", r"\btransitorios?\b"),
    ]

    found = []
    text_lower = text.lower()
    for label, pattern in term_patterns:
        if len(re.findall(pattern, text_lower)) >= 3:
            found.append(label)
        if len(found) >= 10:
            break

    return found


def split_into_sections(full_md: str) -> list[dict]:
    """
    Split the full markdown into sections by TITULO and CAPITULO boundaries.
    Returns a list of dicts: {title, content, articles}
    """
    # Find all structural markers (titulo, capitulo, transitorios) with positions
    markers = []

    for pattern, kind in [
        (TITULO_RE, "titulo"),
        (CAPITULO_RE, "capitulo"),
        (TRANSITORIOS_RE, "transitorios"),
    ]:
        for m in pattern.finditer(full_md):
            markers.append((m.start(), kind, m.group(0).strip()))

    # Deduplicate by position (plain and heading versions may overlap)
    seen_positions = set()
    unique_markers = []
    for pos, kind, text in sorted(markers):
        # Check if there's already a marker within 10 chars
        if any(abs(pos - sp) < 10 for sp in seen_positions):
            continue
        seen_positions.add(pos)
        unique_markers.append((pos, kind, text))

    markers = unique_markers

    if not markers:
        print("WARNING: No structural markers found. Outputting as single file.")
        return [{"title": "LISF Completa", "content": full_md, "articles": find_articles(full_md)}]

    # Split content by markers
    sections = []
    current_titulo = ""
    current_titulo_num = 0

    for i, (pos, kind, heading) in enumerate(markers):
        # Get content from this marker to the next
        end_pos = markers[i + 1][0] if i + 1 < len(markers) else len(full_md)
        content = full_md[pos:end_pos].strip()

        if kind == "titulo":
            current_titulo_num += 1
            current_titulo = heading
            # Check if this titulo has capitulos inside
            next_marker_kind = markers[i + 1][1] if i + 1 < len(markers) else None
            if next_marker_kind == "capitulo":
                # Titulo header only -- content before first capitulo
                sections.append({
                    "title": heading,
                    "titulo_num": current_titulo_num,
                    "kind": "titulo_header",
                    "content": content,
                    "articles": find_articles(content),
                })
            else:
                # Titulo without chapters (entire title is one section)
                sections.append({
                    "title": heading,
                    "titulo_num": current_titulo_num,
                    "kind": "titulo",
                    "content": content,
                    "articles": find_articles(content),
                })
        elif kind == "capitulo":
            sections.append({
                "title": f"{current_titulo} - {heading}",
                "titulo_num": current_titulo_num,
                "kind": "capitulo",
                "content": content,
                "articles": find_articles(content),
            })
        elif kind == "transitorios":
            sections.append({
                "title": heading,
                "titulo_num": 99,
                "kind": "transitorios",
                "content": content,
                "articles": find_articles(content),
            })

    return sections


def merge_small_sections(sections: list[dict], min_chars: int = 500) -> list[dict]:
    """Merge very small sections into the previous one."""
    if not sections:
        return sections

    merged = [sections[0]]
    for sec in sections[1:]:
        if (
            sec["kind"] == "titulo_header"
            and len(sec["content"]) < min_chars
            and not sec["articles"]
        ):
            # Tiny titulo header with no articles -- prepend to next section
            # We'll handle this by keeping it but it won't get its own file
            merged.append(sec)
        else:
            merged.append(sec)

    return merged


def main():
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        print("Run setup.sh first or download LISF.pdf to docs/")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Converting {PDF_PATH} to markdown...")
    full_md = pymupdf4llm.to_markdown(str(PDF_PATH))
    print(f"  Full markdown: {len(full_md):,} characters")

    # Clean up PDF artifacts
    print("Cleaning PDF artifacts (headers, footers, page numbers, blank lines)...")
    full_md = clean_markdown(full_md)
    print(f"  After cleanup: {len(full_md):,} characters")

    # Save full markdown for reference
    full_path = OUTPUT_DIR / "full_lisf.md"
    full_path.write_text(full_md, encoding="utf-8")
    print(f"  Saved full markdown to {full_path}")

    # Split into sections
    print("Splitting into sections...")
    sections = split_into_sections(full_md)
    sections = merge_small_sections(sections)
    print(f"  Found {len(sections)} sections")

    # Write individual files
    index_lines = ["# Indice de Archivos LISF\n"]
    index_lines.append("| Archivo | Titulo/Capitulo | Articulos |")
    index_lines.append("|---------|----------------|-----------|")

    file_num = 0
    files_written = []

    for sec in sections:
        # Skip tiny titulo headers that have no articles
        if sec["kind"] == "titulo_header" and not sec["articles"] and len(sec["content"]) < 500:
            continue

        file_num += 1
        slug = slugify(sec["title"])[:60]
        filename = f"{file_num:02d}_{slug}.md"
        filepath = OUTPUT_DIR / filename

        header = generate_header(sec["title"], sec["content"], sec["articles"])
        filepath.write_text(header + sec["content"], encoding="utf-8")
        files_written.append(filename)

        # Article range for index
        arts = sec["articles"]
        if arts:
            if len(arts) == 1:
                art_range = f"Art. {arts[0]}"
            else:
                art_range = f"Arts. {arts[0]}-{arts[-1]}"
        else:
            art_range = "(sin articulos)"

        index_lines.append(f"| {filename} | {sec['title']} | {art_range} |")
        print(f"  {filename}: {sec['title']} ({len(sec['content']):,} chars, {art_range})")

    # Write index
    index_path = OUTPUT_DIR / "00_indice.md"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"\nIndex written to {index_path}")
    print(f"Total files: {len(files_written)} + index + full_lisf.md")


if __name__ == "__main__":
    main()
