"""Configuração central das linguagens suportadas pelo modelo."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"


@dataclass(frozen=True)
class Language:
    name: str
    # Gramáticas do tree-sitter-language-pack usadas para extrair terminais.
    grammars: tuple[str, ...]
    # Pasta da linguagem no dataset de origem (data/<stack_dir>/...).
    stack_dir: str
    extensions: tuple[str, ...]
    # Dataset de origem, chave de ``dataprep.download.SOURCES``.
    source: str = "the-stack-smol-xl"


LANGUAGES: dict[str, Language] = {
    lang.name: lang
    for lang in [
        Language("python", ("python",), "python", (".py",)),
        Language("javascript", ("javascript",), "javascript", (".js", ".mjs", ".cjs", ".jsx")),
        Language("typescript", ("typescript", "tsx"), "typescript", (".ts", ".tsx")),
        Language("java", ("java",), "java", (".java",)),
        Language("c", ("c",), "c", (".c", ".h")),
        Language("rust", ("rust",), "rust", (".rs",)),
        Language("html", ("html",), "html", (".html", ".htm")),
        Language("css", ("css",), "css", (".css",)),
        # JSON e YAML não estão no the-stack-smol-xl; o the-stack-dedup é gated
        # (aceitar os termos no Hugging Face e fazer ``hf auth login``).
        Language("json", ("json",), "json", (".json",), source="the-stack-dedup"),
        Language("yaml", ("yaml",), "yaml", (".yaml", ".yml"), source="the-stack-dedup"),
    ]
}

# Idiomas naturais priorizados no tokenizador (códigos ISO 639-1). Outros
# idiomas funcionam (o tokenizador é byte-level), mas gastam mais tokens.
NATURAL_LANGUAGES: dict[str, str] = {
    "en": "inglês",
    "es": "espanhol",
    "pt": "português",
    "fr": "francês",
    "de": "alemão",
    "it": "italiano",
}
