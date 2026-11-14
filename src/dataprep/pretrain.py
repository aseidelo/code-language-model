"""Corpus de pré-treino v2: ~11,5 bilhões de tokens no formato do modelo (ver ``docs/dados-v2.md``).

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
    Tokeniza cada documento no formato de pré-treino (tokenizador v2) e embaralha a ordem:

    * código: ``<|lang:x|>`` + quebra de linha + ``<|code|>`` + código (com
      indentação relativa) + ``<|endoftext|>`` (``chat_template.pretraining_segments``);
    * texto natural (``dataprep.natural_text``): texto + ``<|endoftext|>``.

    Em duas fases, sem carregar o corpus na memória: cada parquet vira uma parte
    tokenizada em ``model/pretrain-v2/parts/`` (em paralelo, retomável); depois,
    por split, os documentos são embaralhados e copiados para ``{split}.bin``
    (tokens uint16 em sequência; o treino corta em blocos do contexto),
    ``{split}.docs.npy`` (início de cada documento), ``{split}.langs.npy`` (tipo
    de cada documento), ``{split}.bytes.npy`` (bytes UTF-8 do texto de cada
    documento, para bits por byte) e ``meta.json`` (com o hash do tokenizador).

    Splits: ``train``; ``valid`` (``VALID_DOCS_PER_KIND`` documentos por tipo da
    validação v2); ``valid-v1`` (a validação do corpus v1, retokenizada, para
    comparar modelos de versões diferentes).

``check``
    Valida o resultado: tamanhos, proporções, formato de cada documento,
    ida e volta (decodificar e codificar de novo dá os mesmos tokens) e a
    gramática do Python nos documentos de Python.

Uso:
    uv run python -m dataprep.pretrain download [linguagens e/ou natural:<idioma>...]
    uv run python -m dataprep.pretrain pack
    uv run python -m dataprep.pretrain check                  # model/pretrain-v2
    uv run python -m dataprep.pretrain check --data pretrain  # o corpus v1
"""

import argparse
import json
import os
import random
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from codelm.chat_template import EOS, decode_response, pretraining_segments
from codelm.languages import DATA_DIR, LANGUAGES, NATURAL_LANGUAGES, PROJECT_MODEL_DIR
from codelm.model import CONTEXT_LENGTHS
from codelm.tokenizer import (LEGACY_TOKENIZER_PATH, TOKENIZER_PATH, encode_code, encode_segments, load_tokenizer,
                              tokenizer_sha256, use_tokenizer)
from dataprep import natural_text, stack_edu
from dataprep.corpus import iter_natural_texts, iter_texts
from dataprep.download import RAW_FIELDS, SOURCES, clean_file

REPO = SOURCES["the-stack-dedup"]
# Corpus v1 (the-stack-dedup), que o SFT ainda lê (dataprep.sft_candidates, sft_reverse).
CODE_DIR = DATA_DIR / "the-stack-dedup" / "filtered"
PRETRAIN_V1_DIR = PROJECT_MODEL_DIR / "pretrain"
PRETRAIN_DIR = PROJECT_MODEL_DIR / "pretrain-v2"
PARTS = "parts"  # partes tokenizadas, uma por parquet (ver ``pack``)
PACK_WORKERS = 12
# A validação do v2 é grande (~2% do corpus, separada por repositório): o pack
# guarda uma amostra de documentos por tipo.
VALID_DOCS_PER_KIND = 2000

# Corpus v2 (docs/dados-v2.md): 10B tokens. Fração do total por tipo de documento.
TOTAL_TOKENS = 10_000_000_000
MIX = {
    "python": 0.35, "javascript": 0.08, "typescript": 0.08, "java": 0.08, "c": 0.07, "rust": 0.06,
    "bash": 0.03, "html": 0.04, "css": 0.03, "json": 0.03, "yaml": 0.04,
    "natural:en": 0.09, "natural:es": 0.004, "natural:pt": 0.004, "natural:fr": 0.004,
    "natural:de": 0.004, "natural:it": 0.004,
}
# Baixa um pouco mais que a meta: os filtros do pack e a validação comem uma parte.
DOWNLOAD_MARGIN = 1.05
# Código de qualidade (Stack-Edu) onde existe; HTML, CSS, JSON e YAML do the-stack-dedup.
STACK_EDU_LANGS = set(stack_edu.CONFIGS)
ROWS_PER_PART = 20_000

NATURAL_VALID_DOCS = 200
NATURAL_DATASETS = {lang: "fineweb-edu" if lang == "en" else "fineweb2-hq" for lang in NATURAL_LANGUAGES}
FINEWEB2_HQ_CODES = {"es": "spa_Latn", "pt": "por_Latn", "fr": "fra_Latn", "de": "deu_Latn", "it": "ita_Latn"}
FINEWEB_EDU_PREFIX = "sample/10BT/"
SEED = 0
# Ids dos documentos de texto natural em ``{split}.langs.npy`` vêm depois das linguagens de código.
DOC_KINDS = [*LANGUAGES, *(f"natural:{lang}" for lang in NATURAL_LANGUAGES)]
assert set(MIX) == set(DOC_KINDS) and abs(sum(MIX.values()) - 1) < 1e-9


# --- download ---------------------------------------------------------------

def chars_per_token(lang: str) -> float:
    """Caracteres por token da linguagem, medidos na validação do corpus pequeno."""
    tokenizer = load_tokenizer()
    texts = list(iter_texts(lang, split="valid"))
    return sum(map(len, texts)) / sum(len(encode_code(tokenizer, t, lang)) for t in texts)


def target_chars(kind: str, chars_per_token: float) -> int:
    return int(MIX[kind] * TOTAL_TOKENS * chars_per_token * DOWNLOAD_MARGIN)


def code_dir(lang: str) -> Path:
    """Pasta do corpus v2 da linguagem."""
    if lang in STACK_EDU_LANGS:
        return stack_edu.FILTERED_DIR / lang
    return DATA_DIR / "the-stack-dedup" / "filtered-v2" / lang


def download_lang(lang: str) -> dict:
    target = target_chars(lang, chars_per_token(lang))
    if lang in STACK_EDU_LANGS:
        return stack_edu.download(lang, target, seed=SEED)

    out_dir = code_dir(lang)
    done_file = out_dir / "done.json"
    if done_file.exists():
        return json.loads(done_file.read_text())

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

    with tempfile.TemporaryDirectory(dir=out_dir.parent) as tmp:  # shards baixados, apagados depois de lidos
        for shard in shards:
            if stats["chars train"] >= target:
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
                if stats["chars train"] >= target:
                    break
            Path(local).unlink()
            stats["shards"] += 1
    flush("train")
    flush("valid")

    result = {"lang": lang, "target_chars": target, **stats}
    done_file.write_text(json.dumps(result, indent=1))
    return result


def natural_dir(lang: str) -> Path:
    """Pasta do texto natural v2 do idioma."""
    return DATA_DIR / NATURAL_DATASETS[lang] / "filtered-v2" / lang


def natural_sources(lang: str) -> list[str]:
    """Arquivos remotos do idioma, na ordem de leitura (a validação sai do primeiro)."""
    if lang == "en":
        files = HfApi().list_repo_files("HuggingFaceFW/fineweb-edu", repo_type="dataset")
        return [f"datasets/HuggingFaceFW/fineweb-edu/{f}" for f in sorted(files)
                if f.startswith(FINEWEB_EDU_PREFIX) and f.endswith(".parquet")]
    return [f"datasets/epfml/FineWeb2-HQ/{FINEWEB2_HQ_CODES[lang]}/000_00000.parquet"]


def download_natural(lang: str) -> dict:
    out_dir = natural_dir(lang)
    done_file = out_dir / "done.json"
    if done_file.exists():
        return json.loads(done_file.read_text())

    tokenizer = load_tokenizer()
    sample = list(iter_natural_texts(lang, split="valid"))
    ratio = sum(map(len, sample)) / sum(len(e.ids) for e in tokenizer.encode_batch(sample))
    target = target_chars(f"natural:{lang}", ratio)

    out_dir.mkdir(parents=True, exist_ok=True)
    stats, rows, part = Counter(), [], 0
    for source_index, source in enumerate(natural_sources(lang)):
        if stats["chars train"] >= target:
            break
        parquet, order = natural_text.open_source(source)
        if source_index == 0:  # validação: o primeiro bloco do primeiro arquivo, como no v1
            texts = parquet.read_row_group(order[0], columns=["text"])["text"].to_pylist()[:NATURAL_VALID_DOCS]
            pq.write_table(pa.Table.from_pylist([{"content": t} for t in texts]), out_dir / "valid-000.parquet")
            stats.update({"valid": len(texts), "chars valid": sum(map(len, texts))})
            order = order[1:]
        for group in order:
            if stats["chars train"] >= target:
                break
            for text in parquet.read_row_group(group, columns=["text"])["text"].to_pylist():
                rows.append({"content": text})
                stats["train"] += 1
                stats["chars train"] += len(text)
            stats["blocos"] += 1
            if len(rows) >= ROWS_PER_PART:
                pq.write_table(pa.Table.from_pylist(rows), out_dir / f"train-{part:03d}.parquet")
                rows, part = [], part + 1
        stats["arquivos"] += 1
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), out_dir / f"train-{part:03d}.parquet")

    result = {**stats, "lang": f"natural:{lang}", "target_chars": target}
    done_file.write_text(json.dumps(result, indent=1))
    return result


def _download_any(kind: str) -> dict:
    return download_natural(kind.removeprefix("natural:")) if kind.startswith("natural:") else download_lang(kind)


def download(kinds: list[str]) -> None:
    print(f"{'tipo':<12}{'treino':>10}{'valid':>8}{'chars treino':>15}{'meta':>15}")
    with ProcessPoolExecutor(max_workers=8) as pool:
        for s in pool.map(_download_any, kinds):
            print(
                f"{s['lang']:<12}{s.get('train', 0):>10,}{s.get('valid', 0):>8,}"
                f"{s.get('chars train', 0):>15,}{s['target_chars']:>15,}"
            )


# --- pack -------------------------------------------------------------------

def pack_sources() -> list[tuple[str, str, Path]]:
    """(split, tipo de documento, parquet) de tudo que entra no corpus empacotado.

    ``train`` e ``valid`` vêm do corpus v2. ``valid-v1`` é a validação do corpus
    v1, retokenizada: um conjunto fixo para comparar modelos de versões
    diferentes em bits por byte.
    """
    sources = []
    for kind in DOC_KINDS:
        natural = kind.startswith("natural:")
        lang = kind.removeprefix("natural:")
        for path in sorted((natural_dir(lang) if natural else code_dir(kind)).glob("*.parquet")):
            sources.append((path.name.split("-")[0], kind, path))
        v1 = DATA_DIR / NATURAL_DATASETS[lang] / "filtered" / lang if natural else CODE_DIR / kind
        for path in sorted(v1.glob("valid-*.parquet")):
            sources.append(("valid-v1", kind, path))
    return sources


def part_name(split: str, kind: str, path: Path) -> str:
    return f"{split}.{kind.replace(':', '-')}.{path.stem}"


def tokenize_part(task: tuple[str, str, str, str]) -> tuple[str, int, int]:
    """Tokeniza um parquet para ``parts/<nome>.tokens.npy`` e ``parts/<nome>.docs.npy``.

    ``docs`` tem, por documento, o número de tokens e os bytes UTF-8 do texto
    original (para bits por byte). Ele é gravado por último: se existe, a parte
    está completa (``pack`` retoma pulando as partes prontas).
    """
    # Uma thread por processo: sem isso, cada um dos PACK_WORKERS abre um pool do
    # tokenizers (Rust) e a máquina fica com carga ~80 e ~6x mais lenta.
    os.environ["RAYON_NUM_THREADS"] = "1"
    path, kind, name, tokenizer_path = task
    parts = PRETRAIN_DIR / PARTS
    tokenizer = load_tokenizer(tokenizer_path)
    eos = tokenizer.token_to_id(EOS)
    docs, sizes = [], []
    for text in pq.read_table(path, columns=["content"])["content"].to_pylist():
        if kind.startswith("natural:"):
            ids = [*tokenizer.encode(text, add_special_tokens=False).ids, eos]
        else:
            ids = encode_segments(tokenizer, pretraining_segments(kind, text))
        docs.append(np.array(ids, dtype=np.uint16))
        sizes.append((len(ids), len(text.encode("utf-8"))))
    tokens = np.concatenate(docs) if docs else np.zeros(0, dtype=np.uint16)
    np.save(parts / f"{name}.tokens.npy", tokens)
    np.save(parts / f"{name}.docs.tmp.npy", np.array(sizes, dtype=np.int64).reshape(-1, 2))
    (parts / f"{name}.docs.tmp.npy").rename(parts / f"{name}.docs.npy")
    return name, len(sizes), len(tokens)


def pack() -> None:
    """Empacota o corpus v2 em ``PRETRAIN_DIR`` sem carregar tudo na memória.

    1. Cada parquet vira uma parte tokenizada em ``parts/`` (em paralelo, retomável).
    2. Por split, os documentos são embaralhados e copiados das partes (abertas
       por memmap) para ``{split}.bin``, em blocos.
    """
    tokenizer_path = TOKENIZER_PATH
    parts = PRETRAIN_DIR / PARTS
    parts.mkdir(parents=True, exist_ok=True)
    sources = [(split, kind, path, part_name(split, kind, path)) for split, kind, path in pack_sources()]
    todo = [(str(path), kind, name, str(tokenizer_path)) for _, kind, path, name in sources
            if not (parts / f"{name}.docs.npy").exists()]
    print(f"{len(sources)} partes, {len(sources) - len(todo)} já tokenizadas")
    start, done_tokens = time.time(), 0
    with ProcessPoolExecutor(PACK_WORKERS) as pool:
        for i, (name, n_docs, n_tokens) in enumerate(pool.map(tokenize_part, todo, chunksize=1)):
            done_tokens += n_tokens
            if i % 20 == 0 or i == len(todo) - 1:
                minutes = (time.time() - start) / 60
                print(f"  {i + 1}/{len(todo)} {name}: {n_docs:,} docs | {done_tokens / 1e9:.2f} B tokens "
                      f"em {minutes:.1f} min", flush=True)

    meta = {"version": 2, "context_length": max(CONTEXT_LENGTHS.values()), "doc_kinds": DOC_KINDS, "seed": SEED,
            "tokenizer": str(tokenizer_path.relative_to(tokenizer_path.parents[2])),
            "tokenizer_sha256": tokenizer_sha256(tokenizer_path),
            "valid_docs_per_kind": VALID_DOCS_PER_KIND, "splits": {}}
    rng = np.random.default_rng(SEED)
    for split in dict.fromkeys(s for s, *_ in sources):
        chosen = [(kind, name) for s, kind, _, name in sources if s == split]
        tokens = [np.load(parts / f"{name}.tokens.npy", mmap_mode="r") for _, name in chosen]
        sizes = [np.load(parts / f"{name}.docs.npy") for _, name in chosen]
        # Por documento: parte, início na parte, tokens, bytes e tipo.
        doc_part = np.concatenate([np.full(len(s), i, dtype=np.int32) for i, s in enumerate(sizes)])
        doc_start = np.concatenate([np.cumsum(s[:, 0]) - s[:, 0] for s in sizes])
        doc_len = np.concatenate([s[:, 0] for s in sizes])
        doc_bytes = np.concatenate([s[:, 1] for s in sizes])
        doc_kind = np.concatenate([np.full(len(s), DOC_KINDS.index(k), dtype=np.uint8)
                                   for (k, _), s in zip(chosen, sizes)])
        keep = np.arange(len(doc_len))
        if split == "valid":  # a validação v2 tem ~2% do corpus; basta uma amostra por tipo
            keep = np.concatenate([
                rng.choice(idx, VALID_DOCS_PER_KIND, replace=False) if len(idx) > VALID_DOCS_PER_KIND else idx
                for idx in (np.flatnonzero(doc_kind == k) for k in range(len(DOC_KINDS)))
            ])
        order = keep[rng.permutation(len(keep))]

        tmp = PRETRAIN_DIR / f"{split}.bin.tmp"
        with tmp.open("wb") as out:
            buffer, buffered = [], 0
            for i in order:
                p, s, n = doc_part[i], doc_start[i], doc_len[i]
                buffer.append(tokens[p][s : s + n])
                buffered += n
                if buffered >= 1 << 26:
                    out.write(np.concatenate(buffer).tobytes())
                    buffer, buffered = [], 0
            if buffer:
                out.write(np.concatenate(buffer).tobytes())
        tmp.rename(PRETRAIN_DIR / f"{split}.bin")
        lengths = doc_len[order]
        np.save(PRETRAIN_DIR / f"{split}.docs.npy", np.cumsum(lengths) - lengths)
        np.save(PRETRAIN_DIR / f"{split}.langs.npy", doc_kind[order])
        np.save(PRETRAIN_DIR / f"{split}.bytes.npy", doc_bytes[order])

        kinds = doc_kind[order]
        total = int(lengths.sum())
        meta["splits"][split] = {
            "tokens": total,
            "docs": len(order),
            "bytes": int(doc_bytes[order].sum()),
            "sequences": total // meta["context_length"],
            "tokens_per_kind": {DOC_KINDS[k]: int(lengths[kinds == k].sum()) for k in np.unique(kinds)},
            "bytes_per_kind": {DOC_KINDS[k]: int(doc_bytes[order][kinds == k].sum()) for k in np.unique(kinds)},
        }
        print(f"{split}: {total:,} tokens, {len(order):,} documentos", flush=True)
    (PRETRAIN_DIR / "meta.json").write_text(json.dumps(meta, indent=1))


# --- check ------------------------------------------------------------------

def load_split(split: str, data_dir: Path = PRETRAIN_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tokens, início de cada documento e tipo de cada documento de um split."""
    tokens = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
    return tokens, np.load(data_dir / f"{split}.docs.npy"), np.load(data_dir / f"{split}.langs.npy")


def load_meta(data_dir: Path = PRETRAIN_DIR) -> dict:
    return json.loads((data_dir / "meta.json").read_text())


def data_tokenizer_sha256(meta: dict) -> str:
    """Hash do tokenizador dos dados; o corpus v1 (sem hash no meta.json) é do tokenizador v1."""
    return meta.get("tokenizer_sha256") or tokenizer_sha256(LEGACY_TOKENIZER_PATH)


def data_tokenizer_path(meta: dict) -> Path:
    """O tokenizador (v1 ou v2) com que os dados foram empacotados."""
    sha = data_tokenizer_sha256(meta)
    for path in (TOKENIZER_PATH, LEGACY_TOKENIZER_PATH):
        if tokenizer_sha256(path) == sha:
            return path
    raise ValueError(f"nenhum tokenizador conhecido tem o hash dos dados ({sha[:12]})")


def check(data_dir: Path = PRETRAIN_DIR, samples: int = 2000) -> None:
    from codelm.grammar import GrammarConstraint

    meta = load_meta(data_dir)
    tokenizer_path = use_tokenizer(data_tokenizer_path(meta))
    tokenizer = load_tokenizer()
    print(f"dados em {data_dir} | tokenizador {tokenizer_path}")
    eos = tokenizer.token_to_id(EOS)
    rng = np.random.default_rng(SEED)

    for split, info in meta["splits"].items():
        tokens, starts, kinds = load_split(split, data_dir)
        print(f"\n== {split}: {info['tokens']:,} tokens, {info['docs']:,} documentos, {info['sequences']:,} sequências")
        assert len(tokens) == info["tokens"], "tamanho do .bin diferente do meta.json"
        assert len(starts) == len(kinds) == info["docs"], "número de documentos diferente do meta.json"
        assert tokens.max() < tokenizer.get_vocab_size(), "token fora do vocabulário"
        doc_bytes = np.load(data_dir / f"{split}.bytes.npy") if (data_dir / f"{split}.bytes.npy").exists() else None
        print(f"   {'tipo':<12}{'tokens':>15}{'%':>7}{'docs':>11}{'tokens/doc':>12}{'bytes/token':>13}")
        ends = np.append(starts[1:], len(tokens))
        for kind_id, kind in enumerate(meta["doc_kinds"]):
            mask = kinds == kind_id
            n = int((ends[mask] - starts[mask]).sum())
            if mask.any():
                per_byte = f"{doc_bytes[mask].sum() / n:>13.2f}" if doc_bytes is not None else ""
                print(f"   {kind:<12}{n:>15,}{n / info['tokens']:>7.1%}{mask.sum():>11,}{n / mask.sum():>12,.0f}{per_byte}")

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

    tokens, starts, _ = load_split("train", data_dir)
    first = tokenizer.decode(tokens[:2048].tolist(), skip_special_tokens=False)
    print(f"\nInício da 1ª sequência de treino:\n{first[:700]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download")
    dl.add_argument("kinds", nargs="*", help="linguagens e/ou natural:<idioma> (padrão: todos)")
    sub.add_parser("pack")
    ck = sub.add_parser("check")
    ck.add_argument("--data", default=PRETRAIN_DIR.name, help="pasta em model/ (pretrain-v2 ou pretrain, o v1)")
    args = parser.parse_args()

    if args.command == "download":
        if unknown := set(args.kinds) - set(DOC_KINDS):
            parser.error(f"tipos desconhecidos: {', '.join(sorted(unknown))} (válidos: {', '.join(DOC_KINDS)})")
        download(args.kinds or DOC_KINDS)
    elif args.command == "pack":
        pack()
    else:
        check(PROJECT_MODEL_DIR / args.data)


if __name__ == "__main__":
    main()
