"""Página de validação humana (``assets/templates/human-validation.html``) com uma configuração.

O template guarda a configuração num objeto ``VALIDATION`` (campos, pílulas, vereditos, motivos e
de onde vêm itens e notas); ``render`` troca esse objeto pelo ``config`` dado e esvazia os itens de
exemplo. No modo servidor, a página faz ``GET items``, ``GET notes`` e ``POST notes`` com
``{"id": ..., "record": {...}}``; o registro é
``{id, verdict, reasons[], flagged[], note, reviewer, at, ...meta e copy}``.
"""
from __future__ import annotations

import re

from reports import embed_json, template

CONFIG = re.compile(r"var VALIDATION = \{.*?\n\};", re.S)
ITEMS = re.compile(r'(<script type="application/json" id="items">)(.*?)(</script>)', re.S)

# Registros antigos do eval.sft_review (antes do template) -> formato do template.
OLD_KEYS = {"veredito": "verdict", "motivos": "reasons", "nota": "note"}


def render(config: dict, items: list | None = None) -> str:
    html, n = CONFIG.subn(lambda m: "var VALIDATION = " + embed_json(config) + ";", template("human-validation.html"),
                          count=1)
    if n != 1:
        raise ValueError("objeto VALIDATION não encontrado no template")
    return ITEMS.sub(lambda m: f"{m.group(1)}\n{embed_json(items or [])}\n{m.group(3)}", html, count=1)


def normalize(record: dict) -> dict:
    """Aceita registros no formato antigo (``veredito``/``motivos``/``nota``)."""
    out = {OLD_KEYS.get(k, k): v for k, v in record.items()}
    out.setdefault("reasons", [])
    out.setdefault("flagged", [])
    return out
