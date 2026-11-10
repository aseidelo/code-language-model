"""Semelhança entre as respostas do benchmark e as soluções oficiais (MBPP e HumanEval).

Lê os ``.jsonl`` de ``eval.benchmark run``:

* ROUGE-1 e ROUGE-2: F1 de tokens de código e de pares de tokens em comum (léxico);
* cosseno entre embeddings do UniXcoder (mais perto de semântica);
* "base": a mesma medida com respostas de outros problemas, o que qualquer código
  Python já compartilha. O sinal é a diferença entre a medida e a base.

No HumanEval, a resposta repete a assinatura e a docstring do enunciado: comparamos
só o corpo, depois da docstring. Resposta vazia vale 0.

Uso:
    uv run python -m eval.similarity small-v1/step-04227 small-sft-v1/step-04601
"""

import argparse
import json
import re
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from eval.benchmark import variant_path
from eval.sft_samples import EVAL_DIR

EMBEDDER = "microsoft/unixcoder-base"
TOKEN = re.compile(r"\w+|[^\w\s]")
BOOTSTRAP = 2000


def references() -> dict[str, str]:
    result = {f"mbpp-{r['task_id']}": r["code"]
              for r in load_dataset("google-research-datasets/mbpp", "full", split="test")}
    return result | {r["task_id"]: r["canonical_solution"]
                     for r in load_dataset("openai/openai_humaneval", split="test")}


def answer_text(row: dict) -> str:
    code = row["code"] or ""
    if row["suite"] == "humaneval":
        parts = re.split(r'"""|\'\'\'', code, maxsplit=2)
        return parts[2] if len(parts) == 3 else code
    return code


def rouge(answer: str, reference: str, n: int) -> float:
    def grams(text: str) -> Counter:
        tokens = TOKEN.findall(text)
        return Counter(zip(*(tokens[i:] for i in range(n))))

    a, b = grams(answer), grams(reference)
    common = sum((a & b).values())
    return 2 * common / (a.total() + b.total()) if common else 0.0


@torch.no_grad()
def embed(tokenizer, model, texts: list[str]) -> np.ndarray:
    """Vetores normalizados; texto vazio vira vetor nulo (cosseno 0)."""
    vectors = []
    for i in range(0, len(texts), 32):
        batch = tokenizer(texts[i : i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt")
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"][..., None]
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        vectors.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
    result = np.concatenate(vectors)
    result[[not t.strip() for t in texts]] = 0
    return result


def interval(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    means = values[rng.integers(len(values), size=(BOOTSTRAP, len(values)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", help="o primeiro é a base das diferenças")
    args = parser.parse_args()

    gold = references()
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDER)
    model = AutoModel.from_pretrained(EMBEDDER).eval()
    measures = {}
    for checkpoint in args.checkpoints:
        rows = [json.loads(l) for l in variant_path(checkpoint, "gguf", True).read_text().splitlines()]
        rows = [r for r in rows if r["id"] in gold]
        for suite in ("mbpp", "humaneval"):
            group = [r for r in rows if r["suite"] == suite]
            answers = [answer_text(r) for r in group]
            refs = [gold[r["id"]] for r in group]
            shuffled = answers[1:] + answers[:1]
            ref_vectors = embed(tokenizer, model, refs)
            answer_vectors = embed(tokenizer, model, answers)
            measures[checkpoint, suite] = {
                "ROUGE-1": np.array([rouge(a, r, 1) for a, r in zip(answers, refs)]),
                "ROUGE-2": np.array([rouge(a, r, 2) for a, r in zip(answers, refs)]),
                "cosseno": (answer_vectors * ref_vectors).sum(1),
                "ROUGE-1 base": np.array([rouge(a, r, 1) for a, r in zip(shuffled, refs)]),
                "ROUGE-2 base": np.array([rouge(a, r, 2) for a, r in zip(shuffled, refs)]),
                "cosseno base": (np.roll(answer_vectors, -1, 0) * ref_vectors).sum(1),
            }
        print(f"{checkpoint}: pronto", flush=True)

    means = {f"{c} {s}": {k: float(v.mean()) for k, v in m.items()} for (c, s), m in measures.items()}
    (EVAL_DIR / "benchmark" / "similarity.json").write_text(json.dumps(means, indent=1))

    first = args.checkpoints[0]
    for suite in ("mbpp", "humaneval"):
        print(f"\n{suite}")
        for checkpoint in args.checkpoints:
            current = measures[checkpoint, suite]
            for name in ("ROUGE-1", "ROUGE-2", "cosseno"):
                low, high = interval(current[name] - current[f"{name} base"])
                line = (f"  {checkpoint:<36}{name:<9}{current[name].mean():.3f}  base {current[f'{name} base'].mean():.3f}"
                        f"  sinal {current[name].mean() - current[f'{name} base'].mean():+.3f} [{low:+.3f}, {high:+.3f}]")
                if checkpoint != first:
                    low, high = interval(current[name] - measures[first, suite][name])
                    line += f"  vs {first.split('/')[0]}: {current[name].mean() - measures[first, suite][name].mean():+.3f} [{low:+.3f}, {high:+.3f}]"
                print(line)


if __name__ == "__main__":
    main()
