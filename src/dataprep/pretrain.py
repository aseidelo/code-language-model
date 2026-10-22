"""Corpus de pré-treino: ~1 bilhão de tokens no formato do modelo.

Subcomandos (rodar nesta ordem):

``download``
    Código: para cada linguagem, baixa shards do bigcode/the-stack-dedup até a
    meta de tokens dela (``CODE_MIX``), aplica os mesmos filtros e a mesma
    padronização do corpus pequeno (``dataprep.download.clean_file``) e a mesma
    divisão treino/validação por repositório (um repositório da validação nunca
    cai no treino). Salva ``data/the-stack-dedup/filtered/<lang>/{train,valid}-NNN.parquet``.

    Texto natural: ``NATURAL_TOKENS_PER_LANG`` tokens de cada idioma, para o
    modelo entender os pedidos dos usuários. Inglês do FineWeb-Edu (o mesmo
    arquivo do corpus pequeno) e os outros idiomas do FineWeb2-HQ (os ~10% de
    maior qualidade do FineWeb-2). A validação sai do primeiro bloco da ordem de
    leitura (``dataprep.natural_text.open_source``) e o treino dos seguintes.
    Salva ``data/fineweb-edu/filtered/en/`` e ``data/fineweb2-hq/filtered/<idioma>/``.

``pack``
    Tokeniza cada documento no formato de pré-treino e embaralha a ordem:

    * código: ``<|lang:x|>`` + quebra de linha + ``<|code|>`` + código (com
      indentação relativa) + ``<|endoftext|>`` (``chat_template.pretraining_segments``);
    * texto natural (``dataprep.natural_text``): texto + ``<|endoftext|>``.

    Grava ``data/code-language-model/pretrain/{train,valid}.bin`` (tokens uint16 em sequência; o
    treino corta em blocos de ``CONTEXT_LENGTH``), ``{split}.docs.npy`` (início
    de cada documento), ``{split}.langs.npy`` (linguagem de cada documento) e
    ``meta.json``.

``check``
    Valida o resultado: tamanhos, proporções, formato de cada documento,
    ida e volta (decodificar e codificar de novo dá os mesmos tokens) e a
    gramática do Python nos documentos de Python.

Uso:
    uv run python -m dataprep.pretrain download [linguagens e/ou natural:<idioma>...]
    uv run python -m dataprep.pretrain pack
    uv run python -m dataprep.pretrain check
"""

import argparse
import json
import random
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from codelm.chat_template import EOS, decode_response, pretraining_segments
from codelm.languages import DATA_DIR, LANGUAGES, NATURAL_LANGUAGES, PROJECT_DATA_DIR
from codelm.model import CONTEXT_LENGTH
from codelm.tokenizer import encode_code, encode_segments, load_tokenizer
from dataprep import natural_text
from dataprep.corpus import iter_natural_texts, iter_texts
from dataprep.download import RAW_FIELDS, SOURCES, clean_file

REPO = SOURCES["the-stack-dedup"]
CODE_DIR = DATA_DIR / "the-stack-dedup" / "filtered"
PRETRAIN_DIR = PROJECT_DATA_DIR / "pretrain"

CODE_TOKENS = 950_000_000  # + ~120M de texto natural (NATURAL_TOKENS_PER_LANG)
# Fração dos tokens de código por linguagem: mais peso para as de programação.
CODE_MIX = {
    "python": 0.23,
    "javascript": 0.12,
    "typescript": 0.12,
    "java": 0.12,
    "c": 0.10,
    "rust": 0.10,
    "html": 0.06,
    "css": 0.05,
    "json": 0.05,
    "yaml": 0.05,
}
ROWS_PER_PART = 20_000

# Texto natural: ~2% do treino por idioma (~11% no total).
NATURAL_TOKENS_PER_LANG = 20_000_000
NATURAL_VALID_DOCS = 200
NATURAL_DATASETS = {lang: "fineweb-edu" if lang == "en" else "fineweb2-hq" for lang in NATURAL_LANGUAGES}
FINEWEB2_HQ_CODES = {"es": "spa_Latn", "pt": "por_Latn", "fr": "fra_Latn", "de": "deu_Latn", "it": "ita_Latn"}
SEED = 0
# Ids dos documentos de texto natural em ``{split}.langs.npy`` vêm depois das linguagens de código.
DOC_KINDS = [*LANGUAGES, *(f"natural:{lang}" for lang in NATURAL_LANGUAGES)]


# --- download ---------------------------------------------------------------

def chars_per_token(lang: str) -> float:
    """Caracteres por token da linguagem, medidos na validação do corpus pequeno."""
    tokenizer = load_tokenizer()
    texts = list(iter_texts(lang, split="valid"))
    return sum(map(len, texts)) / sum(len(encode_code(tokenizer, t, lang)) for t in texts)


def download_lang(lang: str) -> dict:
    out_dir = CODE_DIR / lang
    done_file = out_dir / "done.json"
    if done_file.exists():
        return json.loads(done_file.read_text())

    target_chars = int(CODE_MIX[lang] * CODE_TOKENS * chars_per_token(lang))
    shards = sorted(
        f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith(f"data/{lang}/")
    )
    random.Random(SEED).shuffle(shards)  # shards de partes diferentes do dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    stats, seen = Counter(), set()
    buffers: dict[str, list[dict]] = {"train": [], "valid": []}
    parts = Counter()

    def flush(split: str) -> None:
        if buffers[split]:
            pq.write_table(pa.Table.from_pylist(buffers[split]), out_dir / f"{split}-{parts[split]:03d}.parquet")
            parts[split] += 1
            buffers[split] = []

    with tempfile.TemporaryDirectory(dir=CODE_DIR) as tmp:  # shards baixados, apagados depois de lidos
        for shard in shards:
            if stats["chars train"] >= target_chars:
                break
            local = hf_hub_download(REPO, shard, repo_type="dataset", local_dir=tmp)
            for row in pq.read_table(local, columns=list(RAW_FIELDS)).to_pylist():
                if not (file := clean_file(row, lang, seen, stats)):
                    continue
                split = file.pop("split")
                buffers[split].append(file)
                stats[split] += 1
                stats[f"chars {split}"] += len(file["content"])
                if len(buffers[split]) >= ROWS_PER_PART:
                    flush(split)
                if stats["chars train"] >= target_chars:
                    break
            Path(local).unlink()
            stats["shards"] += 1
    flush("train")
    flush("valid")

    result = {"lang": lang, "target_chars": target_chars, **stats}
    done_file.write_text(json.dumps(result, indent=1))
    return result


def natural_dir(lang: str) -> Path:
    return DATA_DIR / NATURAL_DATASETS[lang] / "filtered" / lang


def natural_source(lang: str) -> str:
    if lang == "en":
        return natural_text.SOURCES["en"]
    return f"datasets/epfml/FineWeb2-HQ/{FINEWEB2_HQ_CODES[lang]}/000_00000.parquet"


def download_natural(lang: str) -> dict:
    out_dir = natural_dir(lang)
    done_file = out_dir / "done.json"
    if done_file.exists():
        return json.loads(done_file.read_text())

    tokenizer = load_tokenizer()
    sample = list(iter_natural_texts(lang, split="valid"))
    chars_per_token = sum(map(len, sample)) / sum(len(e.ids) for e in tokenizer.encode_batch(sample))
    target_chars = int(NATURAL_TOKENS_PER_LANG * chars_per_token)

    parquet, order = natural_text.open_source(natural_source(lang))
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = parquet.read_row_group(order[0], columns=["text"])["text"].to_pylist()[:NATURAL_VALID_DOCS]
    pq.write_table(pa.Table.from_pylist([{"content": t} for t in texts]), out_dir / "valid-000.parquet")
    stats = Counter({"valid": len(texts), "chars valid": sum(map(len, texts))})

    rows, part = [], 0
    for group in order[1:]:
        if stats["chars train"] >= target_chars:
            break
        for text in parquet.read_row_group(group, columns=["text"])["text"].to_pylist():
            rows.append({"content": text})
            stats["train"] += 1
            stats["chars train"] += len(text)
        stats["blocos"] += 1
        if len(rows) >= ROWS_PER_PART:
            pq.write_table(pa.Table.from_pylist(rows), out_dir / f"train-{part:03d}.parquet")
            rows, part = [], part + 1
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), out_dir / f"train-{part:03d}.parquet")

    result = {**stats, "lang": f"natural:{lang}", "target_chars": target_chars}
    done_file.write_text(json.dumps(result, indent=1))
    return result


def _download_any(kind: str) -> dict:
    return download_natural(kind.removeprefix("natural:")) if kind.startswith("natural:") else download_lang(kind)


def download(kinds: list[str]) -> None:
    print(f"{'tipo':<12}{'treino':>10}{'valid':>8}{'chars treino':>15}{'meta':>15}")
    with ProcessPoolExecutor(max_workers=4) as pool:
        for s in pool.map(_download_any, kinds):
            print(
                f"{s['lang']:<12}{s.get('train', 0):>10,}{s.get('valid', 0):>8,}"
                f"{s.get('chars train', 0):>15,}{s['target_chars']:>15,}"
            )


# --- pack -------------------------------------------------------------------

def _tokenize_file(task: tuple[str, str, str]) -> tuple[str, str, int, np.ndarray, np.ndarray]:
    """(split, tipo de documento, id do tipo, tokens concatenados, tamanho de cada documento)."""
    path, split, kind = task
    tokenizer = load_tokenizer()
    eos = tokenizer.token_to_id(EOS)
    docs = []
    for text in pq.read_table(path, columns=["content"])["content"].to_pylist():
        if kind.startswith("natural:"):
            ids = [*tokenizer.encode(text, add_special_tokens=False).ids, eos]
        else:
            ids = encode_segments(tokenizer, pretraining_segments(kind, text))
        docs.append(np.array(ids, dtype=np.uint16))
    lengths = np.array([len(d) for d in docs], dtype=np.int64)
    tokens = np.concatenate(docs) if docs else np.zeros(0, dtype=np.uint16)
    return split, kind, DOC_KINDS.index(kind), tokens, lengths


def pack() -> None:
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for lang in LANGUAGES:
        for path in sorted((CODE_DIR / lang).glob("*.parquet")):
            tasks.append((str(path), path.name.split("-")[0], lang))
    for lang in NATURAL_LANGUAGES:
        for path in sorted(natural_dir(lang).glob("*.parquet")):
            tasks.append((str(path), path.name.split("-")[0], f"natural:{lang}"))

    pieces: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {"train": [], "valid": []}
    with ProcessPoolExecutor() as pool:
        for split, kind, kind_id, tokens, lengths in pool.map(_tokenize_file, tasks):
            pieces[split].append((kind_id, tokens, lengths))
            print(f"  {kind:<12} {split:<5} {len(lengths):>8,} docs {len(tokens):>13,} tokens")

    meta = {"context_length": CONTEXT_LENGTH, "doc_kinds": DOC_KINDS, "seed": SEED, "splits": {}}
    for split, parts in pieces.items():
        # Todos os documentos (tipo, parte, início, tamanho), em ordem embaralhada.
        docs = [
            (kind_id, part, int(start), int(length))
            for part, (kind_id, _, lengths) in enumerate(parts)
            for start, length in zip(np.cumsum(lengths) - lengths, lengths)
        ]
        random.Random(SEED).shuffle(docs)
        total = sum(length for *_, length in docs)
        out = np.lib.format.open_memmap(PRETRAIN_DIR / f"{split}.tmp.npy", mode="w+", dtype=np.uint16, shape=(total,))
        starts = np.empty(len(docs), dtype=np.int64)
        kinds = np.empty(len(docs), dtype=np.uint8)
        position = 0
        for i, (kind_id, part, start, length) in enumerate(docs):
            out[position : position + length] = parts[part][1][start : start + length]
            starts[i], kinds[i] = position, kind_id
            position += length
        out.flush()
        del out
        # .bin: tokens crus (o laço de treino abre com np.memmap, sem cabeçalho).
        np.load(PRETRAIN_DIR / f"{split}.tmp.npy", mmap_mode="r").tofile(PRETRAIN_DIR / f"{split}.bin")
        (PRETRAIN_DIR / f"{split}.tmp.npy").unlink()
        np.save(PRETRAIN_DIR / f"{split}.docs.npy", starts)
        np.save(PRETRAIN_DIR / f"{split}.langs.npy", kinds)

        per_kind = Counter()
        for kind_id, _, _, length in docs:
            per_kind[DOC_KINDS[kind_id]] += length
        meta["splits"][split] = {
            "tokens": int(total),
            "docs": len(docs),
            "sequences": int(total // CONTEXT_LENGTH),
            "tokens_per_kind": dict(per_kind),
        }
        print(f"{split}: {total:,} tokens, {len(docs):,} documentos, {total // CONTEXT_LENGTH:,} sequências de {CONTEXT_LENGTH}")
    (PRETRAIN_DIR / "meta.json").write_text(json.dumps(meta, indent=1))


# --- check ------------------------------------------------------------------

def load_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens = np.memmap(PRETRAIN_DIR / f"{split}.bin", dtype=np.uint16, mode="r")
    return tokens, np.load(PRETRAIN_DIR / f"{split}.docs.npy"), np.load(PRETRAIN_DIR / f"{split}.langs.npy")


def check(samples: int = 2000) -> None:
    from codelm.grammar import GrammarConstraint

    tokenizer = load_tokenizer()
    eos = tokenizer.token_to_id(EOS)
    meta = json.loads((PRETRAIN_DIR / "meta.json").read_text())
    rng = np.random.default_rng(SEED)

    for split, info in meta["splits"].items():
        tokens, starts, kinds = load_split(split)
        print(f"\n== {split}: {info['tokens']:,} tokens, {info['docs']:,} documentos, {info['sequences']:,} sequências")
        assert len(tokens) == info["tokens"], "tamanho do .bin diferente do meta.json"
        assert tokens.max() < tokenizer.get_vocab_size(), "token fora do vocabulário"
        print(f"   {'tipo':<12}{'tokens':>14}{'%':>7}{'docs':>10}{'tokens/doc':>12}")
        ends = np.append(starts[1:], len(tokens))
        for kind_id, kind in enumerate(meta["doc_kinds"]):
            mask = kinds == kind_id
            n = int((ends[mask] - starts[mask]).sum())
            if mask.any():
                print(f"   {kind:<12}{n:>14,}{n / info['tokens']:>7.1%}{mask.sum():>10,}{n / mask.sum():>12,.0f}")

        problems = Counter()
        python_checked = python_ok = 0
        for i in rng.choice(len(starts), size=min(samples, len(starts)), replace=False):
            doc = tokens[starts[i] : ends[i]].tolist()
            kind = meta["doc_kinds"][kinds[i]]
            if doc[-1] != eos or eos in doc[:-1]:
                problems["documento sem <|endoftext|> só no fim"] += 1
                continue
            if kind.startswith("natural:"):
                continue
            response = decode_response(tokenizer, doc)
            if response.lang != kind or response.kind != "code":
                problems["cabeçalho errado"] += 1
            elif encode_segments(tokenizer, pretraining_segments(kind, response.content)) != doc:
                problems["ida e volta diferente"] += 1
            if kind == "python" and python_checked < 300:
                python_checked += 1
                code_start = doc.index(tokenizer.token_to_id("<|code|>")) + 1
                python_ok += GrammarConstraint("python").validate(doc[code_start:]) == len(doc) - code_start
        print(f"   {min(samples, len(starts))} documentos sorteados: {dict(problems) or 'nenhum problema'}")
        if python_checked:
            print(f"   gramática do Python aceita {python_ok}/{python_checked} documentos de Python")

    tokens, starts, _ = load_split("train")
    first = tokenizer.decode(tokens[:CONTEXT_LENGTH].tolist(), skip_special_tokens=False)
    print(f"\nInício da 1ª sequência de treino ({CONTEXT_LENGTH} tokens):\n{first[:700]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download")
    dl.add_argument("kinds", nargs="*", help="linguagens e/ou natural:<idioma> (padrão: todos)")
    sub.add_parser("pack")
    sub.add_parser("check")
    args = parser.parse_args()

    if args.command == "download":
        if unknown := set(args.kinds) - set(DOC_KINDS):
            parser.error(f"tipos desconhecidos: {', '.join(sorted(unknown))} (válidos: {', '.join(DOC_KINDS)})")
        download(args.kinds or DOC_KINDS)
    elif args.command == "pack":
        pack()
    else:
        check()


if __name__ == "__main__":
    main()
