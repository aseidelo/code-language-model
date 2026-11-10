"""Empacota o banco de SFT para o treino (ver docs/sft.md).

Lê todos os parquets de ``sft/bank/``, aplica a receita da mistura e o limite de
contexto, separa a validação por repositório e tokeniza com
``chat_template.encode_example``. Salva em ``sft/pack/`` um ``.npz`` por split:

* ``tokens`` e ``mask`` concatenados (``mask`` = 1 no que o modelo gera);
* ``starts``: início de cada exemplo (com um último valor = fim);
* ``task``, ``lang``, ``nl`` e ``level``: etiqueta de cada exemplo, como índices
  das listas em ``meta.json``.

Uso:
    uv run python -m dataprep.sft_pack pack [--medio 0.2] [--dificil 0] [--tasks T1] [--langs python] [--name pack-t1]
    uv run python -m dataprep.sft_pack check
"""

import argparse
import hashlib
import json
import random
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from codelm.chat_template import encode_example, render, decode_response
from codelm.model import CONTEXT_LENGTH
from codelm.tokenizer import decode_segments, load_tokenizer
from dataprep.sft_mix import BANK_DIR, SFT_DIR

PACK_DIR = SFT_DIR / "pack"
LABELS = ("task", "lang", "nl", "level")
VALID_PERCENT = 2  # dos repositórios
SEED = 0


def is_valid(repo: str) -> bool:
    return int(hashlib.sha1(repo.encode()).hexdigest(), 16) % 100 < VALID_PERCENT


def load_bank(fractions: dict[str, float]) -> list[dict]:
    rng = random.Random(SEED)
    rows = []
    for path in sorted(BANK_DIR.glob("*.parquet")):
        rows += pq.read_table(path).to_pylist()
    return [r for r in rows if r["n_tokens"] <= CONTEXT_LENGTH and rng.random() < fractions[r["level"]]]


def pack_split(tokenizer, rows: list[dict], names: dict[str, list[str]], path) -> None:
    tokens, mask, starts = [], [], [0]
    for row in rows:
        ids, generated = encode_example(tokenizer, row["prompt"], row["response"], row["lang"], row["kind"],
                                        code=row["code"], lang_in_prompt=row["lang_in_prompt"])
        tokens += ids
        mask += generated
        starts.append(len(tokens))
    labels = {k: np.array([names[k].index(r[k]) for r in rows], dtype=np.uint8) for k in LABELS}
    np.savez(path, tokens=np.array(tokens, dtype=np.uint16), mask=np.array(mask, dtype=np.uint8),
             starts=np.array(starts, dtype=np.int64), **labels)


def pack(fractions: dict[str, float], tasks: list[str] | None = None, langs: list[str] | None = None,
         directory=PACK_DIR) -> None:
    rows = [r for r in load_bank(fractions) if (tasks is None or r["task"] in tasks)
            and (langs is None or r["lang"] in langs)]
    random.Random(SEED).shuffle(rows)
    names = {k: sorted({r[k] for r in rows}) for k in LABELS}
    splits = {"valid": [r for r in rows if is_valid(r["repo"])], "train": [r for r in rows if not is_valid(r["repo"])]}
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer()
    for name, group in splits.items():
        pack_split(tokenizer, group, names, directory / f"{name}.npz")
        total = sum(r["n_tokens"] for r in group)
        print(f"{name}: {len(group):,} exemplos, {total / 1e6:.1f}M tokens", flush=True)
        for key in ("task", "level"):
            print(f"  {key}: " + ", ".join(f"{k} {v:,}" for k, v in sorted(Counter(r[key] for r in group).items())))
    (directory / "meta.json").write_text(json.dumps({"labels": names, "fractions": fractions}, indent=1))


def check() -> None:
    tokenizer = load_tokenizer()
    meta = json.loads((PACK_DIR / "meta.json").read_text())
    data = np.load(PACK_DIR / "valid.npz")
    for task in range(len(meta["labels"]["task"])):
        i = int(np.flatnonzero(data["task"] == task)[0])
        ids = data["tokens"][data["starts"][i] : data["starts"][i + 1]].astype(int).tolist()
        mask = data["mask"][data["starts"][i] : data["starts"][i + 1]]
        first = int(mask.argmax())
        assert mask[first:].all() and not mask[:first].any(), "a máscara deve ser um único trecho final"
        given = render(decode_segments(tokenizer, ids[:first]))
        generated = render(decode_segments(tokenizer, ids[first:]))
        print(f"--- {meta['labels']['task'][task]} ({len(ids)} tokens, {len(ids) - first} na perda)")
        print(given[:300].rstrip(), "\n[...]\n>>> GERADO:", generated[:240])
        print(">>> FIM:", repr(generated[-40:]))
        if tokenizer.id_to_token(ids[first]).startswith("<|lang:"):
            print(">>> decode_response:", decode_response(tokenizer, ids[first:]).kind)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    pack_cmd = commands.add_parser("pack")
    pack_cmd.add_argument("--medio", type=float, default=0.2, help="fração do nível médio")
    pack_cmd.add_argument("--dificil", type=float, default=0.0, help="fração do nível difícil")
    pack_cmd.add_argument("--tasks", nargs="+", help="só estas tarefas (padrão: todas)")
    pack_cmd.add_argument("--langs", nargs="+", help="só estas linguagens (padrão: todas)")
    pack_cmd.add_argument("--name", default=PACK_DIR.name, help="pasta do pacote em sft/")
    commands.add_parser("check")
    args = parser.parse_args()
    if args.command == "pack":
        pack({"facil": 1.0, "medio": args.medio, "dificil": args.dificil}, args.tasks, args.langs, SFT_DIR / args.name)
    else:
        check()


if __name__ == "__main__":
    main()
