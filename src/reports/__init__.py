"""Páginas no visual do Seidelbots: benchmark e validação humana.

O relatório técnico é gerado pela ferramenta do blog (``seidelbots-blog/design/tools/render_report.py``).

Os templates não ficam no codelm: são lidos do repositório do blog, em
``seidelbots-blog/design/templates/`` ao lado deste repositório, ou em ``$SEIDELBOTS_TEMPLATES``.
Mudanças de template se fazem lá.
Cada página é um HTML único, sem dependências além das fontes do Google.
"""
import json
import re
import subprocess
from pathlib import Path

import os

from codelm.languages import PROJECT_ROOT

TEMPLATES_DIR = Path(os.environ.get("SEIDELBOTS_TEMPLATES")
                     or PROJECT_ROOT.parent / "seidelbots-blog" / "design" / "templates")
DOCS_DIR = PROJECT_ROOT / "docs"


def template(name: str) -> str:
    path = TEMPLATES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"template {path} não existe: clone o seidelbots-blog ao lado do codelm "
                                "ou aponte SEIDELBOTS_TEMPLATES para a pasta design/templates dele")
    return path.read_text(encoding="utf-8")


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
