#!/usr/bin/env python3
"""
convert_cusf.py -- Convert CUSF DOCX to markdown chapters.

Reads docs/CUSF_gob_2024.docx, splits by CAPITULO boundaries,
outputs to docs/cusf_md/ with YAML frontmatter and disposition markers.

Usage:
    pip install python-docx
    python convert_cusf.py
"""
import re
import sys
from pathlib import Path

try:
    from docx import Document
except ImportError:
    print("ERROR: python-docx not installed. Run: pip install python-docx")
    sys.exit(1)

DOCX_PATH = Path(__file__).parent / "docs" / "CUSF_gob_2024.docx"
OUTPUT_DIR = Path(__file__).parent / "docs" / "cusf_md"

# Content starts after the table of contents (paragraph ~305)
CONTENT_START = 305
# Transitorias start around paragraph 11949
TRANSITORIAS_MARKER = "TRANSITORIAS"

# Regex patterns
TITULO_RE = re.compile(r"^TÍTULO\s+(\d+)\.?\s*$", re.IGNORECASE)
CAPITULO_RE = re.compile(r"^CAPÍTULO\s+(\d+\.\d+)\.?\s*$", re.IGNORECASE)
DISP_RE = re.compile(r"^(\d+\.\d+\.\d+)")
KEYWORD_CANDIDATES = re.compile(
    r"(reservas?\s+técnicas?|prima|siniestro|reaseguro|reafianzamiento|"
    r"capital\s+mínimo|solvencia|inversión|gobierno\s+corporativo|"
    r"auditoría|actuario|agente|fianza|seguro|coaseguro|"
    r"nota\s+técnica|póliza|contrato|liquidación|revocación|"
    r"sanción|multa|comisión|intermediario|beneficiario|"
    r"riesgo|obligación|patrimonio|tarifa|dictamen|"
    r"estados?\s+financieros?|contabilidad|información\s+financiera|"
    r"autorización|registro|fondos?|cartera|margen\s+de\s+solvencia)",
    re.IGNORECASE,
)


def parse_docx(docx_path: Path):
    """Parse the DOCX and return structured chapter data."""
    doc = Document(str(docx_path))
    paragraphs = doc.paragraphs

    # Phase 1: Find all chapter and title boundaries after content start
    markers = []  # (para_idx, type, value, name)
    current_titulo = ""
    current_titulo_name = ""

    for i in range(CONTENT_START, len(paragraphs)):
        text = paragraphs[i].text.strip()
        if not text:
            continue

        # Title marker
        m = TITULO_RE.match(text)
        if m:
            current_titulo = m.group(1)
            # Next non-empty paragraph has the title name
            for j in range(i + 1, min(i + 3, len(paragraphs))):
                ntext = paragraphs[j].text.strip()
                if ntext and not CAPITULO_RE.match(ntext):
                    current_titulo_name = ntext
                    break
            continue

        # Chapter marker
        m = CAPITULO_RE.match(text)
        if m:
            cap_num = m.group(1)
            # Next non-empty paragraph has the chapter name
            cap_name = ""
            for j in range(i + 1, min(i + 3, len(paragraphs))):
                ntext = paragraphs[j].text.strip()
                if ntext and not DISP_RE.match(ntext):
                    cap_name = ntext
                    break
            markers.append({
                "para_idx": i,
                "cap_num": cap_num,
                "cap_name": cap_name,
                "titulo_num": current_titulo,
                "titulo_name": current_titulo_name,
            })
            continue

        # Transitorias marker
        if paragraphs[i].style.name == "ANOTACION" and "TRANSITORIA" in text.upper():
            markers.append({
                "para_idx": i,
                "cap_num": "TRANS",
                "cap_name": "TRANSITORIAS",
                "titulo_num": "T",
                "titulo_name": "TRANSITORIAS",
            })
            # Only take the first TRANSITORIAS block
            break

    # Phase 2: Extract content for each chapter
    chapters = []
    for idx, marker in enumerate(markers):
        start = marker["para_idx"]
        # End is start of next chapter, or end of doc
        if idx + 1 < len(markers):
            end = markers[idx + 1]["para_idx"]
        else:
            end = len(paragraphs)

        # Collect paragraphs
        lines = []
        for i in range(start, end):
            text = paragraphs[i].text.strip()
            if not text:
                continue

            # Format disposition numbers as bold markers (like LISF articles)
            m = DISP_RE.match(text)
            if m:
                disp_num = m.group(1)
                # Make disposition number bold for indexing
                text = f"**DISPOSICIÓN {disp_num}.-** {text[len(disp_num):]}"
                # Clean up tab/whitespace after number
                text = re.sub(r"\.-\s*\.\s*", ".- ", text)
                text = re.sub(r"\.-\s+", ".- ", text)

            lines.append(text)

        # Extract disposition numbers for frontmatter
        disps = []
        for line in lines:
            m = re.search(r"\*\*DISPOSICIÓN\s+(\d+\.\d+\.\d+)", line)
            if m:
                disps.append(m.group(1))

        # Extract keywords
        full_text = "\n".join(lines)
        kw_matches = KEYWORD_CANDIDATES.findall(full_text.lower())
        keywords = sorted(set(k.strip() for k in kw_matches))

        chapters.append({
            **marker,
            "content_lines": lines,
            "dispositions": disps,
            "keywords": keywords,
        })

    return chapters


def slugify(text: str, max_len: int = 50) -> str:
    """Create a filesystem-safe slug from text."""
    import unicodedata

    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) not in ("Mn",))
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "_", text).strip("_")
    return text[:max_len].rstrip("_")


def write_chapters(chapters: list[dict], output_dir: Path):
    """Write chapter markdown files and index."""
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    file_num = 1

    for ch in chapters:
        cap_num = ch["cap_num"]
        cap_name = ch["cap_name"]
        titulo_num = ch["titulo_num"]
        titulo_name = ch["titulo_name"]
        disps = ch["dispositions"]
        keywords = ch["keywords"]

        # Build filename
        if cap_num == "TRANS":
            filename = f"{file_num:03d}_transitorias.md"
            titulo_line = "TRANSITORIAS"
        else:
            cap_slug = slugify(cap_name) if cap_name else f"capitulo_{cap_num}"
            filename = f"{file_num:03d}_cap_{cap_num}_{cap_slug}.md"
            titulo_line = f"TÍTULO {titulo_num} - CAPÍTULO {cap_num}"

        # Build frontmatter
        disp_range = f"{disps[0]} - {disps[-1]}" if disps else "N/A"
        fm_lines = [
            "---",
            f"titulo: {titulo_line}",
        ]
        if cap_name and cap_num != "TRANS":
            fm_lines.append(f"tema: {cap_name}")
        if titulo_name and cap_num != "TRANS":
            fm_lines.append(f"titulo_nombre: {titulo_name}")
        fm_lines.append(f"disposiciones: {disp_range}")
        fm_lines.append(f"total_disposiciones: {len(disps)}")
        if keywords:
            fm_lines.append(f"palabras_clave: {', '.join(keywords[:15])}")
        fm_lines.append("---")

        # Build content
        content = "\n".join(fm_lines) + "\n\n"
        content += "\n\n".join(ch["content_lines"])
        content += "\n"

        # Write file
        filepath = output_dir / filename
        filepath.write_text(content, encoding="utf-8")

        # Index entry
        index_rows.append(
            f"| {filename} | **TÍTULO {titulo_num}** - **CAPÍTULO {cap_num}** | Disps. {disp_range} |"
        )

        file_num += 1

    # Write index
    index_content = "# Indice de Archivos CUSF\n\n"
    index_content += "| Archivo | Titulo/Capitulo | Disposiciones |\n"
    index_content += "|---------|----------------|---------------|\n"
    index_content += "\n".join(index_rows) + "\n"

    (output_dir / "00_indice.md").write_text(index_content, encoding="utf-8")

    return file_num - 1


def main():
    if not DOCX_PATH.exists():
        print(f"ERROR: DOCX not found at {DOCX_PATH}")
        print("Download from: https://www.gob.mx (search CUSF compulsada)")
        sys.exit(1)

    print(f"Parsing {DOCX_PATH}...")
    chapters = parse_docx(DOCX_PATH)
    print(f"Found {len(chapters)} chapters")

    # Stats
    total_disps = sum(len(ch["dispositions"]) for ch in chapters)
    total_lines = sum(len(ch["content_lines"]) for ch in chapters)
    print(f"Total dispositions: {total_disps}")
    print(f"Total content lines: {total_lines}")

    print(f"\nWriting to {OUTPUT_DIR}/...")
    count = write_chapters(chapters, OUTPUT_DIR)
    print(f"Wrote {count} chapter files + index")

    # Summary
    print("\n=== Summary ===")
    for ch in chapters[:5]:
        n = ch["cap_num"]
        d = len(ch["dispositions"])
        name = ch["cap_name"][:60]
        print(f"  Cap {n}: {d} dispositions - {name}")
    print("  ...")
    for ch in chapters[-3:]:
        n = ch["cap_num"]
        d = len(ch["dispositions"])
        name = ch["cap_name"][:60]
        print(f"  Cap {n}: {d} dispositions - {name}")


if __name__ == "__main__":
    main()
