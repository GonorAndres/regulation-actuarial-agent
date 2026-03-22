#!/usr/bin/env python3
"""
merge_keywords.py -- Merge Opus-validated keyword data back into per-article .md files.

Reads all Opus-validated JSON files from subagents_outputs/keywords/opus_validated/,
updates frontmatter in docs/articles/*.md, then rebuilds the database.

Usage:
    python merge_keywords.py
"""
import json
import os
import re
import unicodedata
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
ARTICLES_DIR = PROJECT_DIR / "docs" / "articles"
OPUS_DIR = PROJECT_DIR / "subagents_outputs" / "keywords" / "opus_validated"
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)^---\s*\n", re.MULTILINE | re.DOTALL)


def normalize_keyword(kw: str) -> str:
    """Normalize a keyword: strip accents, lowercase, strip whitespace."""
    kw = unicodedata.normalize("NFD", kw.lower().strip())
    kw = "".join(c for c in kw if unicodedata.category(c) != "Mn")
    return kw


def load_opus_data() -> dict:
    """Load all Opus-validated data, indexed by filename."""
    all_data = {}
    for opus_file in sorted(OPUS_DIR.glob("*.json")):
        try:
            raw = json.load(open(opus_file, encoding="utf-8"))
            articles = raw.get("articles", [])
            for art in articles:
                fname = art.get("file", "")
                if fname:
                    all_data[fname] = art
        except Exception as e:
            print(f"  Error loading {opus_file.name}: {e}")
    return all_data


def update_article_file(filepath: Path, opus_entry: dict) -> bool:
    """Update a per-article .md file with Opus-validated metadata."""
    content = filepath.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(content)
    if not m:
        return False

    # Parse existing frontmatter
    fm_text = m.group(1)
    fm = {}
    for line in fm_text.splitlines():
        if ":" in line and not line.startswith(" "):
            key, _, val = line.partition(":")
            fm[key.strip()] = val.strip()

    # Get body (everything after frontmatter)
    body = content[m.end():]

    # Update with Opus data
    kw_list = opus_entry.get("palabras_clave", [])
    kw_normalized = [normalize_keyword(k) for k in kw_list if k.strip()]
    kw_str = ", ".join(kw_normalized)

    resumen = opus_entry.get("resumen", "").strip()
    # Escape quotes in resumen for YAML
    resumen = resumen.replace('"', '\\"')

    categoria = opus_entry.get("categoria", "").strip()

    refs_add = opus_entry.get("refs_adicionales", [])

    # Merge refs_adicionales with existing refs
    existing_int = fm.get("refs_internas", "").strip("[]").strip()
    existing_cruz = fm.get("refs_cruzadas", "").strip("[]").strip()

    # Parse existing refs
    existing_int_list = [r.strip() for r in existing_int.split(",") if r.strip()] if existing_int else []
    existing_cruz_list = [r.strip() for r in existing_cruz.split(",") if r.strip()] if existing_cruz else []

    # Merge new refs
    for ref in refs_add:
        ref = ref.strip()
        if ":" in ref:
            if ref not in existing_cruz_list:
                existing_cruz_list.append(ref)
        else:
            if ref not in existing_int_list:
                existing_int_list.append(ref)

    refs_int_str = ", ".join(existing_int_list)
    refs_cruz_str = ", ".join(existing_cruz_list)

    # Rebuild frontmatter
    new_fm = f"""---
ley: {fm.get('ley', '')}
numero: {fm.get('numero', '')}
titulo: {fm.get('titulo', '')}
capitulo: {fm.get('capitulo', '')}
tema: {fm.get('tema', '')}
categoria: {categoria}
palabras_clave: {kw_str}
resumen: "{resumen}"
refs_internas: [{refs_int_str}]
refs_cruzadas: [{refs_cruz_str}]
---
"""
    filepath.write_text(new_fm + body, encoding="utf-8")
    return True


def main():
    print("Loading Opus-validated data...")
    opus_data = load_opus_data()
    print(f"  Loaded metadata for {len(opus_data)} articles")

    print(f"\nUpdating article files in {ARTICLES_DIR}...")
    updated = 0
    skipped = 0
    not_found = 0

    article_files = sorted(ARTICLES_DIR.glob("*.md"))
    for filepath in article_files:
        fname = filepath.name
        if fname in opus_data:
            if update_article_file(filepath, opus_data[fname]):
                updated += 1
            else:
                skipped += 1
        else:
            # No Opus data for this file -- keep as-is
            not_found += 1

    print(f"  Updated: {updated}")
    print(f"  Skipped (no frontmatter): {skipped}")
    print(f"  No Opus data: {not_found}")

    # Stats
    print(f"\nVerifying...")
    empty_kw = 0
    empty_res = 0
    over_15 = 0
    for filepath in article_files:
        content = filepath.read_text(encoding="utf-8")
        for line in content.splitlines():
            if line.startswith("palabras_clave:"):
                kw = line.split(":", 1)[1].strip()
                if not kw:
                    empty_kw += 1
                elif len(kw.split(",")) > 15:
                    over_15 += 1
            if line.startswith("resumen:"):
                res = line.split(":", 1)[1].strip().strip('"')
                if not res or res == ".":
                    empty_res += 1

    print(f"  Empty keywords: {empty_kw}")
    print(f"  Empty resumenes: {empty_res}")
    print(f"  Over 15 keywords: {over_15}")

    print("\nDone! Now run: python build_index.py")


if __name__ == "__main__":
    main()
