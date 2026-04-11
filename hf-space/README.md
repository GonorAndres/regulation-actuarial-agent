---
title: LISF Agent Demo
emoji: "\u2696\uFE0F"
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
license: mit
short_description: Consulta la Ley de Instituciones de Seguros y de Fianzas
---

# LISF Agent Demo

Demo gratuito para consultar la **Ley de Instituciones de Seguros y de Fianzas (LISF)** y la **Circular Unica de Seguros y Fianzas (CUSF)** -- regulacion aseguradora mexicana.

## Caracteristicas

- Busqueda por texto completo (FTS5) en 510 articulos de la LISF y disposiciones de la CUSF
- Grafo de referencias cruzadas entre articulos
- Cache de preguntas frecuentes con respuestas pre-computadas
- Modelo abierto (Qwen2.5-72B-Instruct) para generacion de respuestas

## Version completa

La version completa con Claude esta disponible en [Cloud Run](https://actuarial-regulation-agent-451451662791.us-central1.run.app). Ofrece respuestas mas precisas, mayor contexto, y citas verificables.

## Tecnologia

- **Backend:** SQLite FTS5 con ranking BM25
- **Modelo:** Qwen2.5-72B-Instruct via HF Inference API
- **Frontend:** Gradio
