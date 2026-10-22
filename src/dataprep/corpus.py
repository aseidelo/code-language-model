"""Leitura dos textos preparados por ``dataprep.download`` e ``dataprep.natural_text``."""

from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from codelm.languages import DATA_DIR

PROCESSED_DIR = DATA_DIR / "processed"
NATURAL_DIR = PROCESSED_DIR / "natural"


def _iter_parquet(path: Path, max_chars: int | None, hint: str) -> Iterator[str]:
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe; rode antes: {hint}")
    total = 0
    for text in pq.read_table(path, columns=["content"])["content"].to_pylist():
        if max_chars is not None and total >= max_chars:
            return
        total += len(text)
        yield text


def iter_texts(lang: str, split: str = "train", max_chars: int | None = None) -> Iterator[str]:
    """Arquivos de código de uma linguagem, parando ao atingir ``max_chars``."""
    path = PROCESSED_DIR / lang / f"{split}.parquet"
    yield from _iter_parquet(path, max_chars, f"uv run python -m dataprep.download {lang}")


def iter_natural_texts(lang: str, split: str = "train", max_chars: int | None = None) -> Iterator[str]:
    """Documentos de texto natural de um idioma (sem código)."""
    path = NATURAL_DIR / lang / f"{split}.parquet"
    yield from _iter_parquet(path, max_chars, f"uv run python -m dataprep.natural_text {lang}")
