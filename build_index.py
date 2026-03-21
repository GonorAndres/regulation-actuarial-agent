#!/usr/bin/env python3
"""
build_index.py -- Build SQLite FTS5 database from LISF + CUSF markdown files.

Parses all chapter files, splits by article/disposition markers,
and creates a full-text search index for fast retrieval.

Usage:
    python build_index.py

Output: docs/regulation.db
"""
import re
import sqlite3
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
LISF_MD_DIR = PROJECT_DIR / "docs" / "lisf_md"
CUSF_MD_DIR = PROJECT_DIR / "docs" / "cusf_md"
DB_PATH = PROJECT_DIR / "docs" / "regulation.db"

# Patterns for splitting articles/dispositions
LISF_SPLIT = re.compile(r"(\*\*ARTÍCULO\s+(\d+)\.?-\*\*)")
CUSF_SPLIT = re.compile(r"(\*\*DISPOSICIÓN\s+(\d+\.\d+\.\d+)\.?-\*\*)")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)^---\s*\n", re.MULTILINE | re.DOTALL)


def parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter fields."""
    fm = {}
    m = FRONTMATTER_RE.match(content)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                fm[key.strip()] = val.strip()
    return fm


def split_articles(content: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    """Split content into (number, text) pairs by article/disposition markers."""
    # Find all marker positions
    markers = [(m.start(), m.group(2)) for m in pattern.finditer(content)]
    if not markers:
        return []

    articles = []
    for i, (pos, number) in enumerate(markers):
        # Text runs from this marker to the next (or end of file)
        end = markers[i + 1][0] if i + 1 < len(markers) else len(content)
        text = content[pos:end].strip()
        articles.append((number, text))
    return articles


def process_directory(source_dir: Path, law: str, pattern: re.Pattern) -> list[dict]:
    """Process all markdown files in a directory."""
    rows = []
    if not source_dir.is_dir():
        print(f"  Directory not found: {source_dir}")
        return rows

    for md_file in sorted(source_dir.glob("*.md")):
        if md_file.name in ("00_indice.md", "full_lisf.md", "full_cusf.md"):
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
            fm = parse_frontmatter(content)
            title = fm.get("titulo", md_file.stem)
            keywords = fm.get("palabras_clave", "")

            articles = split_articles(content, pattern)
            if not articles:
                # File has no article markers -- store as single chunk
                # Strip frontmatter
                body = FRONTMATTER_RE.sub("", content).strip()
                if body and len(body) > 100:
                    rows.append({
                        "law": law,
                        "number": fm.get("disposiciones", md_file.stem),
                        "title": title,
                        "filename": md_file.name,
                        "text": body[:30000],
                        "keywords": keywords,
                    })
            else:
                for number, text in articles:
                    rows.append({
                        "law": law,
                        "number": number,
                        "title": title,
                        "filename": md_file.name,
                        "text": text[:30000],
                        "keywords": keywords,
                    })
        except Exception as e:
            print(f"  Error processing {md_file.name}: {e}")
            continue

    return rows


def build_database(rows: list[dict], db_path: Path):
    """Create SQLite database with FTS5 index."""
    db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(str(db_path))

    conn.execute("""
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            law TEXT NOT NULL,
            number TEXT NOT NULL,
            title TEXT,
            filename TEXT NOT NULL,
            text TEXT NOT NULL,
            keywords TEXT
        )
    """)

    conn.execute("""
        CREATE VIRTUAL TABLE articles_fts USING fts5(
            number, title, text, keywords,
            content='articles',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        )
    """)

    # Insert all rows
    for row in rows:
        conn.execute(
            "INSERT INTO articles (law, number, title, filename, text, keywords) VALUES (?, ?, ?, ?, ?, ?)",
            (row["law"], row["number"], row["title"], row["filename"], row["text"], row["keywords"]),
        )

    # Populate FTS index
    conn.execute("""
        INSERT INTO articles_fts (rowid, number, title, text, keywords)
        SELECT id, number, title, text, keywords FROM articles
    """)

    # Create indexes for common lookups
    conn.execute("CREATE INDEX idx_law_number ON articles (law, number)")
    conn.execute("CREATE INDEX idx_filename ON articles (filename)")

    conn.commit()
    conn.close()


def main():
    print("Building regulation search index...")

    print(f"\nProcessing LISF ({LISF_MD_DIR})...")
    lisf_rows = process_directory(LISF_MD_DIR, "lisf", LISF_SPLIT)
    print(f"  Found {len(lisf_rows)} LISF articles")

    print(f"\nProcessing CUSF ({CUSF_MD_DIR})...")
    cusf_rows = process_directory(CUSF_MD_DIR, "cusf", CUSF_SPLIT)
    print(f"  Found {len(cusf_rows)} CUSF dispositions")

    all_rows = lisf_rows + cusf_rows
    print(f"\nTotal: {len(all_rows)} entries")

    print(f"\nBuilding database at {DB_PATH}...")
    build_database(all_rows, DB_PATH)

    # Verify
    conn = sqlite3.connect(str(DB_PATH))
    count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM articles_fts").fetchone()[0]

    # Test a search
    results = conn.execute("""
        SELECT a.law, a.number, snippet(articles_fts, 2, '>>>', '<<<', '...', 30) as snippet
        FROM articles_fts f
        JOIN articles a ON a.id = f.rowid
        WHERE articles_fts MATCH 'nota tecnica'
        ORDER BY rank
        LIMIT 5
    """).fetchall()

    print(f"\nDatabase built: {count} articles, {fts_count} FTS entries")
    print(f"File size: {DB_PATH.stat().st_size / 1024 / 1024:.1f} MB")

    print(f"\nTest search: 'nota tecnica'")
    for law, number, snippet in results:
        print(f"  [{law.upper()}] {number}: {snippet[:80]}")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
