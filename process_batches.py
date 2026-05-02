#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path

# Load batch assignments
with open('/home/andtega349/lisf-agent/subagents_outputs/keywords/batch_assignments.json', 'r') as f:
    batch_data = json.load(f)

articles_dir = Path('/home/andtega349/lisf-agent/docs/articles')

def guess_category(content, filename):
    """Determine category based on content keywords."""
    content_lower = content.lower()

    # Define category keywords
    categories = {
        'disposiciones_preliminares': ['definiciones', 'objeto', 'ambito', 'aplicacion', 'principios'],
        'organizacion': ['sociedad', 'constitucion', 'organos', 'consejo', 'asamblea', 'administrador', 'director'],
        'intermediarios': ['agente', 'intermediario', 'broker', 'comisionista', 'registrado'],
        'operacion': ['operaciones', 'contratacion', 'poliza', 'vigencia', 'pago', 'prima', 'reclamos'],
        'reservas': ['reserva', 'tecnica', 'insuficiencia', 'constitucion', 'fondo'],
        'inversiones': ['inversion', 'cartera', 'activos', 'valores', 'deuda', 'inmueble'],
        'solvencia': ['solvencia', 'capital', 'patrimonio', 'minimo', 'requerimiento'],
        'contabilidad': ['contabilidad', 'estados', 'financieros', 'balance', 'resultado', 'registro'],
        'vigilancia': ['vigilancia', 'inspeccion', 'comisario', 'auditor', 'revision', 'supervision'],
        'liquidacion': ['liquidacion', 'insolvencia', 'quiebra', 'disolucion', 'extincion'],
        'grupos_financieros': ['grupo', 'financiero', 'integrante', 'consolidado', 'coordinacion'],
        'sanciones': ['sancion', 'multa', 'penalidad', 'infraccion', 'incumplimiento'],
        'procedimientos': ['procedimiento', 'tramite', 'solicitud', 'resolucion', 'recurso'],
        'gobierno_corporativo': ['gobierno', 'corporativo', 'junta', 'consejero', 'ejecutivo'],
        'productos': ['poliza', 'seguro', 'fianza', 'cobertura', 'rama', 'ramo'],
        'reaseguro': ['reaseguro', 'retrocesion', 'cedente', 'asegurador'],
        'fianzas': ['fianza', 'fiador', 'obligacion', 'garantia'],
        'pensiones': ['pension', 'jubilacion', 'retiro', 'invalidez', 'aportacion'],
        'salud': ['salud', 'medico', 'hospitalario', 'farmaceutico'],
        'danos': ['danos', 'incendio', 'terremoto', 'robo', 'responsabilidad'],
        'transitorios': ['transitorio', 'vigencia', 'derogacion', 'abrogacion'],
        'informacion_financiera': ['informacion', 'reporte', 'divulgacion', 'transparencia'],
        'agentes': ['agente', 'comision', 'registro'],
    }

    scores = {}
    for cat, keywords in categories.items():
        score = sum(content_lower.count(kw) for kw in keywords)
        if score > 0:
            scores[cat] = score

    return max(scores, key=scores.get) if scores else 'otros'

def extract_article_refs(content):
    """Extract article references from content."""
    refs = []

    # Pattern: "articulo XXX", "art. XXX"
    single_refs = re.findall(r'articulo\s+(\d+)', content, re.IGNORECASE)
    single_refs += re.findall(r'art\.?\s+(\d+)', content, re.IGNORECASE)

    for ref in set(single_refs):
        refs.append(f"lisf:{ref}")

    # Pattern: "articulos X a Y"
    range_refs = re.findall(r'articulos?\s+(\d+)\s+a\s+(\d+)', content, re.IGNORECASE)
    for start, end in range_refs:
        refs.append(f"lisf:{start}..{end}")

    # CUSF references
    cusf_refs = re.findall(r'cusf\s*[:\s]+\s*([\d\.]+)', content, re.IGNORECASE)
    for ref in set(cusf_refs):
        refs.append(f"cusf:{ref}")

    return list(set(refs))

def extract_keywords(content, filename):
    """Extract insurance-specific keywords from content."""
    keywords = set()

    # Insurance-specific terms
    insurance_terms = [
        'seguro', 'poliza', 'prima', 'siniestro', 'indemnizacion', 'cobertura',
        'asegurado', 'asegurador', 'tomador', 'beneficiario', 'intermediario',
        'agente', 'broker', 'comisionista', 'fianza', 'reaseguro',
        'rama', 'ramo', 'vigencia', 'renovacion', 'cancelacion', 'rescate',
        'reserva', 'tecnica', 'insuficiencia', 'capital', 'patrimonio',
        'solvencia', 'liquidez', 'siniestralidad',
        'danos', 'incendio', 'robo', 'responsabilidad', 'vida', 'salud',
        'invalidez', 'incapacidad', 'muerte', 'pension', 'jubilacion',
        'retiro', 'aportacion', 'fondo', 'inversion', 'activos',
        'contabilidad', 'estados financieros', 'balance', 'resultado',
        'inspeccion', 'vigilancia', 'comisario', 'auditor', 'supervision',
        'liquidacion', 'insolvencia', 'quiebra', 'disolucion', 'extincion',
        'sancion', 'multa', 'penalidad', 'infraccion', 'incumplimiento',
        'procedimiento', 'tramite', 'solicitud', 'recurso', 'apelacion',
        'gobierno corporativo', 'junta directiva', 'consejo', 'asamblea',
        'grupo financiero', 'consolidado', 'integrante', 'coordinacion',
    ]

    content_lower = content.lower()
    for term in insurance_terms:
        if term in content_lower:
            # Remove accents
            clean_term = term.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
            keywords.add(clean_term)

    # Extract acronyms (3+ capital letters)
    acronyms = re.findall(r'\b([A-Z]{3,})\b', content)
    for acr in set(acronyms):
        keywords.add(acr.lower())

    # Extract key phrases around important words
    for pattern in ['operacion', 'procedimiento', 'requisito', 'obligacion', 'derecho', 'facultad']:
        if pattern in content_lower:
            keywords.add(pattern)

    return sorted(list(keywords))

def extract_resumen(content):
    """Extract a meaningful 1-2 sentence summary."""
    # Remove YAML frontmatter
    if content.startswith('---'):
        parts = content.split('---')
        if len(parts) >= 3:
            content = parts[2].strip()

    # Clean up markdown headers and formatting
    lines = content.split('\n')
    for line in lines:
        stripped = line.strip()
        # Look for non-empty, non-header lines with reasonable length
        if stripped and not stripped.startswith('#') and not stripped.startswith('-') and len(stripped) > 20:
            # This is our summary candidate
            # Limit to first 300 chars or full sentence if shorter
            if len(stripped) > 300:
                # Find sentence boundary
                sent_end = stripped.find('.', 100)
                if sent_end > 0:
                    resumen = stripped[:sent_end+1]
                else:
                    resumen = stripped[:300] + '.'
            else:
                if not stripped.endswith('.'):
                    resumen = stripped + '.'
                else:
                    resumen = stripped
            return resumen

    # Fallback
    return content[:200].replace('\n', ' ').strip()

def process_file(filepath):
    """Process a single article file and extract metadata."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Limit to first 5000 chars if file is too long
        if len(content) > 10000:
            content_to_analyze = content[:5000]
        else:
            content_to_analyze = content

        # Extract metadata
        palabras_clave = extract_keywords(content_to_analyze, filepath.name)
        categoria = guess_category(content_to_analyze, filepath.name)
        refs_adicionales = extract_article_refs(content_to_analyze)
        resumen = extract_resumen(content_to_analyze)

        return {
            'file': filepath.name,
            'palabras_clave': palabras_clave,
            'resumen': resumen,
            'categoria': categoria,
            'refs_adicionales': refs_adicionales
        }
    except Exception as e:
        print(f"Error processing {filepath.name}: {e}")
        return None

# Process batches 0-4
for batch_id in range(5):
    batch = batch_data['batches'][batch_id]
    results = []

    print(f"Processing batch {batch_id}...")

    for filename in batch['files']:
        filepath = articles_dir / filename
        if filepath.exists():
            result = process_file(filepath)
            if result:
                results.append(result)
        else:
            print(f"  NOT FOUND: {filename}")

    # Write batch output
    output_file = f'/home/andtega349/lisf-agent/subagents_outputs/keywords/batch_{batch_id:03d}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Batch {batch_id}: wrote {len(results)} records to {output_file}")

print("Done!")
