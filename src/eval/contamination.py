"""Contaminação dos benchmarks (HumanEval, MBPP) num corpus, por n-gramas compartilhados.

O texto vira palavras e símbolos (``\\w+`` ou um caractere de pontuação), sem espaços
nem indentação. Cada problema (enunciado + solução canônica) vira o conjunto dos seus
n-gramas. Para cada documento do corpus, conta quantos n-gramas de cada problema
aparecem nele. A medida é a **cobertura**: a maior fração dos n-gramas do problema
encontrada num único documento. Cobertura alta significa que o problema (ou boa parte
dele) está copiado naquele documento.

    uv run python -m eval.contamination pretrain [--kinds python] [--n 13]

Saída: ``data/codelm_sft/eval/contaminacao/<nome>.json``, com a cobertura de cada
problema, separada em enunciado e solução, e os documentos com mais cobertura.
"""

import argparse
import multiprocessing
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq
from datasets import load_dataset

from dataprep.pretrain import pack_sources
from eval.sft_samples import EVAL_DIR

OUT_DIR = EVAL_DIR / "contaminacao"
WORD = re.compile(r"\w+|[^\w\s]")
TOP_DOCS = 3


def words(text: str) -> list[str]:
    return WORD.findall(text)


def ngrams(text: str, n: int) -> set[int]:
    w = words(text)
    return {hash(tuple(w[i:i + n])) for i in range(len(w) - n + 1)}


def problems() -> list[dict]:
    """Enunciado e solução canônica de cada problema do HumanEval e do MBPP (todos os splits)."""
    result = []
    for row in load_dataset("openai/openai_humaneval", split="test"):
        result.append({"id": row["task_id"], "prompt": row["prompt"], "solution": row["canonical_solution"]})
    for split in ("train", "validation", "test", "prompt"):
        for row in load_dataset("google-research-datasets/mbpp", "full", split=split):
            prompt = row["text"] + "\n" + "\n".join(row["test_list"])
            result.append({"id": f"mbpp-{row['task_id']}", "prompt": prompt, "solution": row["code"]})
    return result


_INDEX: dict[int, list[tuple[int, int]]] = {}  # n-grama -> [(problema, parte)]; parte 0 enunciado, 1 solução


def _init(index):
    global _INDEX
    _INDEX = index


def scan_file(task: tuple[str, str, int]) -> list[tuple[int, int, int, int, str]]:
    """(problema, n-gramas do enunciado, n-gramas da solução, linha, repo/path) dos documentos com algum acerto."""
    path, kind, n = task
    columns = [c for c in ("content", "text", "repo", "path") if c in pq.read_schema(path).names]
    table = pq.read_table(path, columns=columns).to_pylist()
    hits = []
    for row_number, row in enumerate(table):
        text = row.get("content") or row.get("text") or ""
        w = words(text)
        found: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        seen = set()
        for i in range(len(w) - n + 1):
            h = hash(tuple(w[i:i + n]))
            if h in _INDEX and h not in seen:
                seen.add(h)
                for problem, part in _INDEX[h]:
                    found[problem][part] += 1
        where = f"{row.get('repo', '')}/{row.get('path', '')}" if "repo" in row else f"{path}#{row_number}"
        hits += [(p, c[0], c[1], row_number, where) for p, c in found.items()]
    return hits


def scan(name: str, files: list[tuple[str, str]], n: int) -> None:
    probs = problems()
    sizes = []
    index: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, p in enumerate(probs):
        parts = (ngrams(p["prompt"], n), ngrams(p["solution"], n))
        sizes.append([len(parts[0]), len(parts[1])])
        for part, grams in enumerate(parts):
            for h in grams:
                index[h].append((i, part))
    print(f"{len(probs)} problemas, {len(index):,} n-gramas; {len(files)} arquivos", flush=True)

    best = [[] for _ in probs]  # (cobertura total, enunciado, solução, onde)
    start = time.time()
    # fork: os processos herdam o índice e a semente do hash() de str (que é aleatória por processo)
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("fork"), initializer=_init,
                             initargs=(dict(index),)) as pool:
        tasks = [(str(path), kind, n) for path, kind in files]
        for done, hits in enumerate(pool.map(scan_file, tasks), 1):
            for problem, a, b, _, where in hits:
                total = (a + b) / max(1, sum(sizes[problem]))
                cov = (round(total, 3), round(a / max(1, sizes[problem][0]), 3), round(b / max(1, sizes[problem][1]), 3),
                       where)
                best[problem] = sorted(best[problem] + [cov], reverse=True)[:TOP_DOCS]
            if done % 20 == 0:
                print(f"  {done}/{len(files)} arquivos, {time.time() - start:.0f}s", flush=True)

    report = {"n": n, "arquivos": len(files), "problemas": {}}
    for p, size, top in zip(probs, sizes, best):
        report["problemas"][p["id"]] = {
            "ngramas": size,
            "cobertura": top[0][0] if top else 0.0,
            "cobertura_enunciado": max((t[1] for t in top), default=0.0),
            "cobertura_solucao": max((t[2] for t in top), default=0.0),
            "documentos": [{"cobertura": t[0], "enunciado": t[1], "solucao": t[2], "onde": t[3]} for t in top],
        }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    summarize(report)
    print(f"gravado em {path}")


def summarize(report: dict) -> None:
    for suite in ("HumanEval", "mbpp"):
        rows = [v for k, v in report["problemas"].items() if k.startswith(suite)]
        print(f"{suite}: {len(rows)} problemas")
        for limit in (0.2, 0.5, 0.8):
            print(f"  cobertura >= {limit:.0%}: total {sum(r['cobertura'] >= limit for r in rows)}, "
                  f"enunciado {sum(r['cobertura_enunciado'] >= limit for r in rows)}, "
                  f"solução {sum(r['cobertura_solucao'] >= limit for r in rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    pretrain = commands.add_parser("pretrain", help="split de treino do corpus de pré-treino v2")
    pretrain.add_argument("--kinds", nargs="+", default=["python"])
    pretrain.add_argument("--n", type=int, default=13)
    args = parser.parse_args()
    files = [(path, kind) for split, kind, path in pack_sources() if split == "train" and kind in args.kinds]
    scan(f"pretrain-{'-'.join(k.replace(':', '_') for k in args.kinds)}-n{args.n}", files, args.n)


if __name__ == "__main__":
    main()
