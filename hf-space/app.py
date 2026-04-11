"""LISF Agent Demo -- Hugging Face Spaces version.

STANDALONE APP -- This is the HF Space frontend/backend.
It is NOT coupled to the Cloud Run app (app.py at repo root).
Changes here do NOT affect Cloud Run, and vice versa.

Architecture: Gradio ChatInterface + Qwen2.5-72B (free HF Inference API)
+ single-pass pre-computed RAG from retrieval.py (SQLite FTS5).
"""

import logging
import os
from pathlib import Path

import gradio as gr
from huggingface_hub import InferenceClient

from retrieval import (
    open_db,
    select_context,
    load_json_cache,
    build_normalized_index,
    match_cache,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "regulation.db"
CLOUD_RUN_URL = "https://actuarial-regulation-agent-451451662791.us-central1.run.app"
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

# ---------------------------------------------------------------------------
# Load database and caches
# ---------------------------------------------------------------------------

db = open_db(DB_PATH)
if db:
    count = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    logger.info("Database loaded: %d articles/dispositions", count)
else:
    count = 0
    logger.warning("No database found at %s", DB_PATH)

faq_entries = load_json_cache(DATA_DIR / "lisf_faq.json")
faq_index = build_normalized_index(faq_entries)

titulo_entries = load_json_cache(DATA_DIR / "titulo_answers.json")
titulo_index = build_normalized_index(titulo_entries)

casos_entries = load_json_cache(DATA_DIR / "casos_practicos.json")
casos_index = build_normalized_index(casos_entries)

_hf_token = os.environ.get("HF_TOKEN")
client = InferenceClient(token=_hf_token)
logger.info("HF token: %s | DB: %d articles | FAQ: %d | Titulo: %d | Casos: %d",
            bool(_hf_token), count, len(faq_entries), len(titulo_entries), len(casos_entries))

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
**Aviso legal:** Esta herramienta es una referencia de estudio y consulta. \
No sustituye la lectura completa de la normativa ni constituye asesoria legal o actuarial.

Eres un consultor especializado en regulacion aseguradora mexicana. \
Tu audiencia son actuarios, abogados corporativos y profesionales del sector asegurador.

## Tu conocimiento
Dominas dos cuerpos normativos:
1. **LISF** -- Ley de Instituciones de Seguros y de Fianzas (ley federal, 510 articulos en 13 titulos)
2. **CUSF** -- Circular Unica de Seguros y Fianzas (normativa secundaria emitida por la CNSF)
La LISF establece el marco general; la CUSF lo reglamenta con disposiciones operativas.

## Como responder
- SIEMPRE responde en espanol.
- Se recibes articulos o datos estructurales en el contexto, basa tu respuesta en ellos. \
Si el contexto incluye estructura o metadata de la ley (titulos, capitulos, conteos), \
usala directamente para responder.
- Si no hay contexto relevante, indica que no encontraste informacion especifica \
y ofrece orientacion general con tu conocimiento de la regulacion.
- SIEMPRE especifica la ley de origen al citar: "Art. X de la LISF" o "Disposicion Y de la CUSF". \
Nunca cites solo "Art. X" sin indicar la ley.
- Para cuestiones simples, responde directo con la cita.
- Para cuestiones complejas: contexto -> fundamento legal -> analisis -> conclusion practica.
- Cuando un articulo de la LISF tiene reglamentacion en la CUSF, mencionalo.
- NO uses emojis. Nunca.
- NUNCA menciones archivos, bases de datos, rutas internas, ni herramientas.
- Si no conoces la respuesta, dilo honestamente.
- Indica donde profundizar: "Para mas detalle, consulta el Articulo X de la LISF, Titulo N".\
"""

# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------


def _check_caches(message: str) -> str | None:
    for entries, index in [
        (faq_entries, faq_index),
        (titulo_entries, titulo_index),
        (casos_entries, casos_index),
    ]:
        answer = match_cache(entries, index, message)
        if answer:
            return answer
    return None


def respond(message: str, history: list[dict]) -> str:
    if not message or not message.strip():
        return ""

    cached = _check_caches(message)
    if cached:
        return cached

    context = select_context(db, message, max_chars=12000) if db else ""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-6:]:
        messages.append(turn)

    if context:
        augmented = (
            f"[Articulos relevantes encontrados en la base de datos]\n"
            f"{context}\n\n"
            f"[Pregunta del usuario]\n{message}"
        )
    else:
        augmented = message
    messages.append({"role": "user", "content": augmented})

    try:
        response = client.chat_completion(
            model=MODEL_ID,
            messages=messages,
            max_tokens=1500,
            temperature=0.3,
        )
        return response.choices[0].message.content or "No se pudo generar una respuesta."
    except Exception as e:
        logger.error("Inference error (%s): %s", type(e).__name__, e, exc_info=True)
        return (
            f"Error al consultar el modelo: {type(e).__name__}. "
            "Puedes intentar de nuevo o visitar la version completa en "
            f"{CLOUD_RUN_URL}"
        )


# ---------------------------------------------------------------------------
# Custom CSS -- retro 90s Windows theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* Root variables */
:root {
    --bg-retro: #C0C0C0;
    --titlebar: #000080;
    --surface: #DFDFDF;
    --border-retro: #000000;
    --shadow-sm: 2px 2px 0 #000;
    --shadow: 4px 4px 0 #000;
}

/* Global */
body, .gradio-container {
    background: var(--bg-retro) !important;
    font-family: system-ui, 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
.gradio-container {
    max-width: 960px !important;
}

/* Remove all border radius */
.gradio-container *, .gradio-container *::before, .gradio-container *::after {
    border-radius: 0 !important;
}

/* Title area */
#component-0 {
    background: var(--titlebar) !important;
    padding: 8px 16px !important;
    border: 3px solid var(--border-retro) !important;
    border-bottom: none !important;
    box-shadow: var(--shadow) !important;
}
#component-0 h3, #component-0 p, #component-0 a, #component-0 strong {
    color: #FFFFFF !important;
}
#component-0 a {
    text-decoration: underline !important;
}

/* Chatbot */
.chatbot {
    border: 3px solid var(--border-retro) !important;
    background: #FFFFFF !important;
    box-shadow: var(--shadow) !important;
}

/* User messages */
.user .message-bubble-border {
    background: #E8E8E8 !important;
    border: 2px solid var(--border-retro) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Bot messages */
.bot .message-bubble-border {
    background: #F5F5F5 !important;
    border: 2px solid var(--border-retro) !important;
    box-shadow: var(--shadow) !important;
}
.bot .message-bubble-border .md strong {
    color: var(--titlebar) !important;
}
.bot .message-bubble-border .md h1,
.bot .message-bubble-border .md h2,
.bot .message-bubble-border .md h3 {
    color: var(--titlebar) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* Input */
.input-container textarea {
    border: 2px solid var(--border-retro) !important;
    background: #FFFFFF !important;
    box-shadow: inset 1px 1px 2px #808080 !important;
    font-size: 13px !important;
}

/* Buttons */
button.primary, button.secondary {
    background: var(--surface) !important;
    border: 2px solid var(--border-retro) !important;
    color: #000000 !important;
    font-weight: 700 !important;
    box-shadow: var(--shadow-sm) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
button.primary:hover, button.secondary:hover {
    background: #EEEEEE !important;
}
button.primary:active, button.secondary:active {
    border-style: inset !important;
    box-shadow: none !important;
}

/* Example buttons */
.examples button {
    background: var(--surface) !important;
    border: 2px solid var(--border-retro) !important;
    box-shadow: var(--shadow-sm) !important;
    font-size: 12px !important;
    color: #000000 !important;
}
.examples button:hover {
    background: var(--titlebar) !important;
    color: #FFFFFF !important;
}

/* Status bar at bottom */
#status-bar {
    background: var(--surface) !important;
    border: 3px solid var(--border-retro) !important;
    border-top: none !important;
    padding: 4px 12px !important;
    font-size: 11px !important;
    color: #4A4A4A !important;
}

/* Hide default footer */
footer { display: none !important; }
"""

# ---------------------------------------------------------------------------
# Gradio UI -- ChatInterface (reliable on HF Spaces) + custom theme
# ---------------------------------------------------------------------------

DESCRIPTION = f"""\
Consulta la **LISF** (510 articulos) y la **CUSF** (normativa secundaria de la CNSF).
Modelo: Qwen2.5-72B | Base de datos: {count} articulos indexados | \
Preguntas frecuentes: {len(faq_entries) + len(titulo_entries) + len(casos_entries)} respuestas pre-computadas.

**[Version completa con Claude]({CLOUD_RUN_URL})** -- \
respuestas mas precisas, mayor contexto, y citas verificables.

*Aviso: Referencia de estudio. No sustituye asesoria legal o actuarial.*
"""

EXAMPLES = [
    "Cuantos articulos tiene la LISF?",
    "Que son las reservas tecnicas segun la LISF?",
    "Que dice el articulo 216 de la LISF?",
    "Quien supervisa a las instituciones de seguros en Mexico?",
    "Que tipos de operaciones de seguros existen?",
    "Cuales son los requisitos para operar como institucion de seguros?",
    "Que es la Base de Inversion en la LISF?",
    "Que sanciones existen por violaciones a la LISF?",
    "Que son las Sociedades Mutualistas de Seguros?",
    "Cuando se publico la LISF?",
]

demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="ActuarialClaude Demo -- Regulacion Aseguradora Mexicana",
    description=DESCRIPTION,
    examples=EXAMPLES,
    textbox=gr.Textbox(
        placeholder="Escribe tu consulta sobre la LISF o CUSF...",
        lines=1,
    ),
    css=CUSTOM_CSS,
)


if __name__ == "__main__":
    demo.launch()
