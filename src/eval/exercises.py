"""Acerto nos exercícios verificáveis: o sinal disponível para o GRPO, por conceito.

Só com a validação, de modelos de exercício nunca vistos no treino. Mede:

* acerto em T=0, a medida principal;
* acerto com amostragem (T=0,7) e a fração de grupos mistos, com ao menos um
  sucesso e um fracasso, com 8 e 16 amostras: grupo uniforme não ensina nada ao GRPO.

Uso (o motor gguf precisa de ``uv run --with llama-cpp-python``):
    uv run python -m eval.exercises --checkpoint base-sft-v1/step-06400 [--per-concept 24] [--samples 16]
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq

from codelm.chat_template import decode_response, encode_prompt
from codelm.tokenizer import load_tokenizer, use_tokenizer_of
from dataprep.sft_mix import SFT_DIR
from eval.generation import ENGINES, load_engine, load_generator
from eval.sandbox import tests_passed
from eval.sft_samples import EVAL_DIR

GROUPS = (8, 16)


def load_exercises(concept: str | None = None, per_concept: int | None = None, seed: int = 0,
                   split: str = "valid") -> list[dict]:
    rows = pq.read_table(SFT_DIR / "exercises" / f"{split}.parquet").to_pylist()
    rng = np.random.default_rng(seed)
    rows = [rows[i] for i in rng.permutation(len(rows)) if concept in (None, rows[i]["concept"])]
    if per_concept is None:
        return rows
    taken: dict[str, list[dict]] = {}
    for row in rows:
        group = taken.setdefault(row["concept"], [])
        if len(group) < per_concept:
            group.append(row)
    return [row for group in taken.values() for row in group]


def answer(tokenizer, run, row: dict) -> str | None:
    lang = "python" if row["lang_in_prompt"] else None
    try:
        return decode_response(tokenizer, run(encode_prompt(tokenizer, row["prompt"], lang))).content
    except ValueError:  # resposta incompleta ou recusada pela gramática
        return None


def correct(code: str | None, row: dict) -> bool:
    return bool(code) and tests_passed(code, row["tests"]) == len(row["tests"])


def solved(tokenizer, run, row: dict) -> tuple[str | None, bool]:
    code = answer(tokenizer, run, row)
    return code, correct(code, row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--engine", choices=ENGINES, default="gguf")
    parser.add_argument("--per-concept", type=int, default=24)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    use_tokenizer_of(args.checkpoint)
    tokenizer = load_tokenizer()
    rows = load_exercises(per_concept=args.per_concept)
    engine = load_engine(args.engine, args.checkpoint)
    greedy = load_generator(engine, tokenizer, args.max_tokens, temperature=0.0)
    sampled = load_generator(engine, tokenizer, args.max_tokens, temperature=0.7)

    results = []
    with ThreadPoolExecutor(8) as pool:
        for n, row in enumerate(rows, 1):
            codes = [answer(tokenizer, sampled, row) for _ in range(args.samples)]
            results.append({"id": row["id"], "concept": row["concept"], "template": row["template"],
                            "greedy": solved(tokenizer, greedy, row)[1],
                            "samples": list(pool.map(correct, codes, [row] * len(codes)))})
            if n % 50 == 0:
                print(f"{n}/{len(rows)}", flush=True)

    path = EVAL_DIR / "exercises" / f"{args.checkpoint.replace('/', '-')}-{args.engine}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results))

    def line(name: str, group: list[dict]) -> None:
        mixed = [100 * np.mean([any(r["samples"][:k]) and not all(r["samples"][:k]) for r in group]) for k in GROUPS]
        print(f"{name:<16}{len(group):>5}{100 * np.mean([r['greedy'] for r in group]):>8.1f}%"
              f"{100 * np.mean([r['samples'] for r in group]):>8.1f}%{mixed[0]:>9.1f}%{mixed[1]:>9.1f}%")

    print(f"{'conceito':<16}{'n':>5}{'T=0':>9}{'T=0,7':>9}{'mistos/8':>10}{'mistos/16':>10}")
    for concept in sorted({r["concept"] for r in results}):
        line(concept, [r for r in results if r["concept"] == concept])
    line("gerador", [r for r in results if not r["concept"].startswith("oc-")])
    line("opencoder", [r for r in results if r["concept"].startswith("oc-")])


if __name__ == "__main__":
    main()
