#!/usr/bin/env python3
"""
split_chapters.py -- Split chapter-level markdown files into per-article files.

Reads from docs/lisf_md/ and docs/cusf_md/, splits by article/disposition markers,
writes per-article files to docs/articles/ with unified frontmatter, per-article
keywords, cross-references, and basic summaries.

Usage:
    python split_chapters.py
"""
import re
import unicodedata
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
LISF_MD_DIR = PROJECT_DIR / "docs" / "lisf_md"
CUSF_MD_DIR = PROJECT_DIR / "docs" / "cusf_md"
ARTICLES_DIR = PROJECT_DIR / "docs" / "articles"

# Splitting patterns
LISF_SPLIT = re.compile(r"(\*\*ARTÍCULO\s+(\d+)\.?-\*\*)")
CUSF_SPLIT = re.compile(r"(\*\*DISPOSICIÓN\s+(\d+\.\d+\.\d+)\.?-\*\*)")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)^---\s*\n", re.MULTILINE | re.DOTALL)

# Transitorio ordinal patterns
LISF_TRANS_SPLIT = re.compile(
    r"(\*\*(?:PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[EÉ]PTIMO|OCTAVO|NOVENO|"
    r"D[EÉ]CIMO|UND[EÉ]CIMO|DUOD[EÉ]CIMO|D[EÉ]CIMO\s+\w+)\.?-\*\*)",
    re.IGNORECASE,
)
CUSF_TRANS_SPLIT = re.compile(
    r"((?:PRIMERA|SEGUNDA|TERCERA|CUARTA|QUINTA|SEXTA|S[EÉ]PTIMA|OCTAVA|NOVENA|"
    r"D[EÉ]CIMA|UND[EÉ]CIMA|DUOD[EÉ]CIMA|D[EÉ]CIMA\s+\w+)\.-)",
    re.IGNORECASE,
)

# Cross-reference extraction patterns
REF_LISF_ART = re.compile(r"art[ií]culo\s+(\d+)", re.IGNORECASE)
REF_CUSF_DISP = re.compile(r"disposici[oó]n\s+(\d+\.\d+\.\d+)", re.IGNORECASE)
REF_ART_DE_LISF = re.compile(r"art[ií]culo\s+(\d+)\s+de\s+la\s+LISF", re.IGNORECASE)
REF_DISP_DE_CUSF = re.compile(r"disposici[oó]n\s+(\d+\.\d+\.\d+)\s+de\s+la\s+CUSF", re.IGNORECASE)

# Expanded keyword extraction -- merged from convert_pdf.py and convert_cusf.py + actuarial terms
KEYWORD_PATTERNS = [
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
    ("comision", r"\bComisi[oó]n\b"),
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
    ("siniestro", r"\bsiniestros?\b"),
    ("tarifa", r"\btarifas?\b"),
    ("dictamen", r"\bdictam[eé]n\b"),
    ("autorizacion", r"\bautorizaci[oó]n\b"),
    ("registro", r"\bregistros?\b"),
    ("fondos", r"\bfondos?\b"),
    ("cartera", r"\bcarteras?\b"),
    ("margen de solvencia", r"margen\s+de\s+solvencia"),
    ("intermediario", r"\bintermediarios?\b"),
    ("contrato", r"\bcontratos?\b"),
    ("obligacion", r"\bobligaci[oó]n\b"),
    ("patrimonio", r"\bpatrimonios?\b"),
    ("agente", r"\bagentes?\b"),
    # Actuarial/technical terms
    ("RCS", r"\bRCS\b"),
    ("BEL", r"\bBEL\b"),
    ("IBNR", r"\bIBNR\b"),
    ("reserva de riesgos en curso", r"reserva[s]?\s+de\s+riesgos?\s+en\s+curso"),
    ("reserva de obligaciones pendientes", r"reserva[s]?\s+de\s+obligaciones?\s+pendientes?"),
    ("reserva de contingencia", r"reserva[s]?\s+de\s+contingencia"),
    ("reserva matematica", r"reserva[s]?\s+matem[aá]ticas?"),
    ("seguros de vida", r"seguros?\s+de\s+vida"),
    ("seguros de danos", r"seguros?\s+de\s+da[nñ]os"),
    ("accidentes y enfermedades", r"accidentes\s+y\s+enfermedades"),
    ("pensiones", r"\bpensiones?\b"),
    ("sociedad mutualista", r"sociedades?\s+mutualistas?"),
    ("grupo financiero", r"grupos?\s+financieros?"),
    ("informacion financiera", r"informaci[oó]n\s+financiera"),
    ("base de inversion", r"base\s+de\s+inversi[oó]n"),
    ("modelo interno", r"modelos?\s+internos?"),
    ("prueba de solvencia", r"prueba[s]?\s+de\s+solvencia"),
    ("administracion de riesgos", r"administraci[oó]n\s+(?:integral\s+)?de\s+riesgos?"),
    ("control interno", r"control\s+interno"),
    ("funcion actuarial", r"funci[oó]n\s+actuarial"),
]


def normalize_text(text: str) -> str:
    """Strip accents for matching."""
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def extract_keywords(text: str) -> list[str]:
    """Extract per-article keywords using expanded glossary. Threshold: 1+ mentions."""
    found = []
    text_lower = text.lower()
    for label, pattern in KEYWORD_PATTERNS:
        if re.search(pattern, text_lower):
            found.append(label)
    return sorted(set(found))


def extract_first_sentence(text: str) -> str:
    """Extract first meaningful sentence as interim resumen."""
    # Skip the article marker line
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        # Skip blank lines, marker lines, roman numerals
        if not line or line.startswith("**") and line.endswith("**"):
            continue
        # Strip bold markers from start
        clean = re.sub(r"^\*\*.*?\*\*\s*", "", line).strip()
        if len(clean) > 20:
            # Take up to first period or semicolon
            m = re.search(r"[.;]", clean)
            if m:
                return clean[: m.end()].strip()
            return clean[:250].strip()
    return ""


def extract_cross_refs(text: str, own_law: str, own_number: str) -> tuple[list[str], list[str]]:
    """Extract internal and cross-law references from article text."""
    refs_internas = set()
    refs_cruzadas = set()

    if own_law == "lisf":
        # Internal LISF refs
        for num in REF_LISF_ART.findall(text):
            if num != own_number:
                refs_internas.add(num)
        # Cross-refs to CUSF
        for disp in REF_DISP_DE_CUSF.findall(text):
            refs_cruzadas.add(f"cusf:{disp}")
        for disp in REF_CUSF_DISP.findall(text):
            refs_cruzadas.add(f"cusf:{disp}")
    else:
        # Internal CUSF refs
        for disp in REF_CUSF_DISP.findall(text):
            if disp != own_number:
                refs_internas.add(disp)
        # Cross-refs to LISF
        for num in REF_ART_DE_LISF.findall(text):
            refs_cruzadas.add(f"lisf:{num}")

    return sorted(refs_internas), sorted(refs_cruzadas)


def parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter fields, joining multi-line values."""
    fm = {}
    m = FRONTMATTER_RE.match(content)
    if m:
        current_key = None
        for line in m.group(1).splitlines():
            if ":" in line and not line.startswith(" "):
                key, _, val = line.partition(":")
                current_key = key.strip()
                fm[current_key] = val.strip()
            elif current_key and line.strip():
                # Continuation of previous value (multi-line)
                fm[current_key] += " " + line.strip()
    return fm


def write_article_file(filepath: Path, law: str, number: str, title: str,
                       chapter: str, tema: str, text: str, keywords: list[str],
                       resumen: str, refs_internas: list[str], refs_cruzadas: list[str]):
    """Write a single per-article markdown file with unified frontmatter."""
    kw_str = ", ".join(keywords) if keywords else ""
    refs_int_str = ", ".join(refs_internas) if refs_internas else ""
    refs_cruz_str = ", ".join(refs_cruzadas) if refs_cruzadas else ""

    fm = f"""---
ley: {law}
numero: "{number}"
titulo: "{title}"
capitulo: "{chapter}"
tema: "{tema}"
palabras_clave: {kw_str}
resumen: "{resumen}"
refs_internas: [{refs_int_str}]
refs_cruzadas: [{refs_cruz_str}]
---
"""
    filepath.write_text(fm + "\n" + text + "\n", encoding="utf-8")


# Pattern to detect trailing section headers at end of article text
_TRAILING_HEADER = re.compile(
    r"(?:\n\s*\n|\n)"
    r"(?:\*?\s*Modificad[ao]\s+DOF\s+[\d\-]+\s*\*?\s*\n\s*)?"
    r"(TÍTULO\s+\d+\.?\s*\n.*?)$",
    re.IGNORECASE | re.DOTALL,
)
_TRAILING_CAP = re.compile(
    r"(?:\n\s*\n|\n)"
    r"(CAPÍTULO\s+[\d.]+\.?\s*\n.*?)$",
    re.IGNORECASE | re.DOTALL,
)


def strip_trailing_headers(text: str) -> str:
    """Remove trailing TITULO/CAPITULO headers that belong to the next section."""
    # Strip trailing TITULO (may include subtitle on next line)
    text = _TRAILING_HEADER.sub("", text).rstrip()
    # Strip trailing CAPITULO
    text = _TRAILING_CAP.sub("", text).rstrip()
    return text


def split_articles(content: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    """Split content into (number, text) pairs by article/disposition markers."""
    markers = [(m.start(), m.group(2)) for m in pattern.finditer(content)]
    if not markers:
        return []
    articles = []
    for i, (pos, number) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(content)
        text = content[pos:end].strip()
        # Strip trailing section headers from the last article in each chapter
        if i == len(markers) - 1:
            text = strip_trailing_headers(text)
        articles.append((number, text))
    return articles


def split_transitorios(content: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    """Split transitorio content by ordinal markers."""
    markers = [(m.start(), m.group(0)) for m in pattern.finditer(content)]
    if not markers:
        return []
    articles = []
    for i, (pos, label) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(content)
        text = content[pos:end].strip()
        if i == len(markers) - 1:
            text = strip_trailing_headers(text)
        num = f"trans_{i + 1:02d}"
        articles.append((num, text))
    return articles


def process_lisf():
    """Process all LISF chapter files into per-article files."""
    count = 0
    for md_file in sorted(LISF_MD_DIR.glob("*.md")):
        if md_file.name in ("00_indice.md", "full_lisf.md"):
            continue

        content = md_file.read_text(encoding="utf-8")
        fm = parse_frontmatter(content)
        title = fm.get("titulo", md_file.stem)
        tema = fm.get("tema", title)
        # Extract chapter number from filename (e.g., "13" from "13_titulo_quinto_...")
        chapter = md_file.name.split("_")[0]
        body = FRONTMATTER_RE.sub("", content).strip()

        # Check for transitorios file
        if "transitori" in md_file.name.lower():
            articles = split_transitorios(body, LISF_TRANS_SPLIT)
            if not articles:
                # Fallback: store as single file
                articles = [("trans_01", body)]
            for number, text in articles:
                keywords = extract_keywords(text)
                resumen = extract_first_sentence(text)
                refs_int, refs_cruz = extract_cross_refs(text, "lisf", number)
                filepath = ARTICLES_DIR / f"lisf_{number}.md"
                write_article_file(filepath, "lisf", number, title, chapter,
                                   tema, text, keywords, resumen, refs_int, refs_cruz)
                count += 1
        else:
            articles = split_articles(content, LISF_SPLIT)
            if not articles:
                # File with no article markers -- store as single chunk
                if body and len(body) > 100:
                    number = fm.get("articulos", chapter)
                    keywords = extract_keywords(body)
                    resumen = extract_first_sentence(body)
                    refs_int, refs_cruz = extract_cross_refs(body, "lisf", str(number))
                    filepath = ARTICLES_DIR / f"lisf_{number}.md"
                    write_article_file(filepath, "lisf", str(number), title, chapter,
                                       tema, body, keywords, resumen, refs_int, refs_cruz)
                    count += 1
            else:
                for number, text in articles:
                    keywords = extract_keywords(text)
                    resumen = extract_first_sentence(text)
                    refs_int, refs_cruz = extract_cross_refs(text, "lisf", number)
                    filepath = ARTICLES_DIR / f"lisf_{number.zfill(3)}.md"
                    write_article_file(filepath, "lisf", number, title, chapter,
                                       tema, text, keywords, resumen, refs_int, refs_cruz)
                    count += 1

    return count


def process_cusf():
    """Process all CUSF chapter files into per-article files."""
    count = 0
    for md_file in sorted(CUSF_MD_DIR.glob("*.md")):
        if md_file.name in ("00_indice.md", "full_cusf.md"):
            continue

        content = md_file.read_text(encoding="utf-8")
        fm = parse_frontmatter(content)
        title = fm.get("titulo", md_file.stem)
        tema = fm.get("tema", fm.get("titulo_nombre", title))
        chapter = md_file.name.split("_")[0]
        body = FRONTMATTER_RE.sub("", content).strip()

        # Check for transitorias file
        if "transitoria" in md_file.name.lower():
            articles = split_transitorios(body, CUSF_TRANS_SPLIT)
            if not articles:
                articles = [("trans_01", body)]
            for number, text in articles:
                keywords = extract_keywords(text)
                resumen = extract_first_sentence(text)
                refs_int, refs_cruz = extract_cross_refs(text, "cusf", number)
                filepath = ARTICLES_DIR / f"cusf_{number}.md"
                write_article_file(filepath, "cusf", number, title, chapter,
                                   tema, text, keywords, resumen, refs_int, refs_cruz)
                count += 1
        else:
            articles = split_articles(content, CUSF_SPLIT)
            if not articles:
                if body and len(body) > 100:
                    number = fm.get("disposiciones", chapter)
                    keywords = extract_keywords(body)
                    resumen = extract_first_sentence(body)
                    refs_int, refs_cruz = extract_cross_refs(body, "cusf", str(number))
                    filepath = ARTICLES_DIR / f"cusf_{number}.md"
                    write_article_file(filepath, "cusf", str(number), title, chapter,
                                       tema, body, keywords, resumen, refs_int, refs_cruz)
                    count += 1
            else:
                for number, text in articles:
                    keywords = extract_keywords(text)
                    resumen = extract_first_sentence(text)
                    refs_int, refs_cruz = extract_cross_refs(text, "cusf", number)
                    filepath = ARTICLES_DIR / f"cusf_{number}.md"
                    write_article_file(filepath, "cusf", number, title, chapter,
                                       tema, text, keywords, resumen, refs_int, refs_cruz)
                    count += 1

    return count


def main():
    print("Splitting chapter files into per-article files...")
    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing LISF ({LISF_MD_DIR})...")
    lisf_count = process_lisf()
    print(f"  Created {lisf_count} LISF article files")

    print(f"\nProcessing CUSF ({CUSF_MD_DIR})...")
    cusf_count = process_cusf()
    print(f"  Created {cusf_count} CUSF disposition files")

    total = lisf_count + cusf_count
    print(f"\nTotal: {total} per-article files in {ARTICLES_DIR}")

    # Quick stats
    files = list(ARTICLES_DIR.glob("*.md"))
    lisf_files = [f for f in files if f.name.startswith("lisf_")]
    cusf_files = [f for f in files if f.name.startswith("cusf_")]
    print(f"  LISF files: {len(lisf_files)}")
    print(f"  CUSF files: {len(cusf_files)}")

    # Sample a file to verify format
    sample = sorted(files)[0]
    print(f"\nSample file ({sample.name}):")
    text = sample.read_text(encoding="utf-8")
    # Show first 15 lines
    for i, line in enumerate(text.splitlines()[:15], 1):
        print(f"  {i}: {line}")

    print("\nDone!")


if __name__ == "__main__":
    main()
