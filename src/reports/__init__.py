"""Páginas no visual do Seidelbots: benchmark e validação humana.

O relatório técnico é gerado pela ferramenta do blog (``seidelbots-blog/design/tools/render_report.py``).

Os templates ficam em ``assets/templates/`` (cópia de ``seidelbots-blog/design/templates/``).
Cada página é um HTML único, sem dependências além das fontes do Google.
"""
import json
import re
import subprocess
from pathlib import Path

from codelm.languages import ASSETS_DIR, PROJECT_ROOT

TEMPLATES_DIR = ASSETS_DIR / "templates"
DOCS_DIR = PROJECT_ROOT / "docs"


def template(name: str) -> str:
    return (TEMPLATES_DIR / name).read_text(encoding="utf-8")


def git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def repo_info() -> dict:
    """``repo`` (owner/nome no GitHub), ``commit`` curto e se a árvore tinha mudanças."""
    url = git("config", "--get", "remote.origin.url") or ""
    match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
    return {"repo": match.group(1) if match else None, "commit": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def embed_json(data) -> str:
    """JSON seguro dentro de ``<script>``: ``</`` fecharia a tag antes da hora."""
    return json.dumps(data, ensure_ascii=False, indent=1).replace("</", "<\\/")


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "pagina"
