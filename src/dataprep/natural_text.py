"""Baixa texto natural (sem código) dos idiomas priorizados.

Fontes: FineWeb-Edu (inglês) e FineWeb-2 (demais idiomas), textos da web já
filtrados por qualidade e deduplicados. Os arquivos têm vários GB, mas são
divididos em blocos de 1.000 documentos: lemos só os blocos necessários do
primeiro arquivo, espalhados ao longo dele, sem baixá-lo inteiro.

Salva ``data/code-language-model/processed/natural/<idioma>/{train,valid}.parquet``.

Uso:
    uv run python -m dataprep.natural_text            # todos os idiomas
    uv run python -m dataprep.natural_text pt es      # apenas alguns
"""

import argparse

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from codelm.languages import NATURAL_LANGUAGES
from dataprep.corpus import NATURAL_DIR

SOURCES = {
    "en": "datasets/HuggingFaceFW/fineweb-edu/sample/10BT/000_00000.parquet",
    "es": "datasets/HuggingFaceFW/fineweb-2/data/spa_Latn/train/000_00000.parquet",
    "pt": "datasets/HuggingFaceFW/fineweb-2/data/por_Latn/train/000_00000.parquet",
    "fr": "datasets/HuggingFaceFW/fineweb-2/data/fra_Latn/train/000_00000.parquet",
    "de": "datasets/HuggingFaceFW/fineweb-2/data/deu_Latn/train/000_00000.parquet",
    "it": "datasets/HuggingFaceFW/fineweb-2/data/ita_Latn/train/000_00000.parquet",
}

TRAIN_CHARS = 30_000_000
VALID_CHARS = 1_000_000


def open_source(path: str) -> tuple[pq.ParquetFile, list[int]]:
    """Arquivo parquet remoto e a ordem de leitura dos seus blocos de 1.000 documentos.

    O primeiro bloco da ordem é sempre o da validação.
    """
    parquet = pq.ParquetFile(HfFileSystem().open(path))
    n_groups = parquet.metadata.num_row_groups
    # Blocos espalhados pelo arquivo (mais variedade que só os primeiros).
    return parquet, [i * 97 % n_groups for i in range(n_groups)]


def download(lang: str) -> dict[str, int]:
    parquet, order = open_source(SOURCES[lang])

    splits: dict[str, list[dict]] = {"valid": [], "train": []}
    limits = {"valid": VALID_CHARS, "train": TRAIN_CHARS}
    chars = {"valid": 0, "train": 0}
    for group in order:
        split = "valid" if chars["valid"] < VALID_CHARS else "train"
        if chars[split] >= limits[split]:
            break
        for text in parquet.read_row_group(group, columns=["text"])["text"].to_pylist():
            splits[split].append({"content": text, "lang": lang})
            chars[split] += len(text)

    out_dir = NATURAL_DIR / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        pq.write_table(pa.Table.from_pylist(rows), out_dir / f"{split}.parquet")
    return {f"docs {s}": len(r) for s, r in splits.items()} | {f"chars {s}": c for s, c in chars.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("languages", nargs="*", help=f"padrão: todos ({', '.join(NATURAL_LANGUAGES)})")
    args = parser.parse_args()
    if unknown := set(args.languages) - set(NATURAL_LANGUAGES):
        parser.error(f"idiomas desconhecidos: {', '.join(sorted(unknown))}")

    for lang in args.languages or NATURAL_LANGUAGES:
        print(f"== {lang} ({NATURAL_LANGUAGES[lang]})")
        for key, value in download(lang).items():
            print(f"  {key:<12} {value:>12,}")


if __name__ == "__main__":
    main()
