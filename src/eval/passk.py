"""pass@k no HumanEval (e no MBPP, no modo chat), por temperatura (ver docs/etapas.md).

Dois modos:

* ``completar``: como no paper do Codex, para o modelo do pré-treino. O prompt é o
  enunciado do HumanEval (assinatura e docstring) no formato do pré-treino
  (``<|lang:python|>`` + ``<|code|>``), sem fechar os blocos, e o modelo continua o
  corpo. A continuação para no ``<|endoftext|>`` ou nas sequências de parada do paper.
* ``chat``: o formato do SFT, com a máscara de gramática (os mesmos prompts de
  ``eval.benchmark``).

O pass@k usa o estimador sem viés do paper: ``n`` amostras por problema, ``c`` certas,
``pass@k = 1 − C(n−c, k) / C(n, k)``. O plano diz quantas amostras por temperatura:
``0.2:20,0.8:100`` dá pass@1 e pass@10 a T=0,2 e pass@100 a T=0,8, como no paper.

Uso:
    uv run python -m eval.passk --checkpoint base-v1/step-04227 --mode completar
    uv run python -m eval.passk --checkpoint base-ex-v3/step-00762 --mode chat \\
        --plan 0.2:20,0.4:20,0.6:20,0.8:20,1.0:20
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from datasets import load_dataset

from codelm.chat_template import CODE, EOS, decode_response, pretraining_segments
from codelm.indentation import DEDENT, decode_indentation
from codelm.model import PAD
from codelm.tokenizer import decode_segments, encode_segments, load_tokenizer, use_tokenizer_of
from eval.benchmark import humaneval, mbpp
from eval.sandbox import passes, syntax_ok
from eval.sft_samples import EVAL_DIR
from train.grpo import load_model, sample_batch
from train.pretrain import CHECKPOINT_DIR
from train.smoke_test import pick_device, synchronize

STOP = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")  # as do paper do Codex
KS = (1, 5, 8, 10, 100)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimador sem viés (Chen et al., 2021), na forma numericamente estável."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


# --- modo completar -----------------------------------------------------------

def completion_problems(tokenizer) -> list[dict]:
    problems = []
    for row in load_dataset("openai/openai_humaneval", split="test"):
        segments = pretraining_segments("python", row["prompt"])
        while segments[-1][1] in (EOS, DEDENT):  # o corpo continua dentro da função
            segments.pop()
        body = segments[[s[1] for s in segments].index(CODE) + 1 :]
        problems.append({"suite": "humaneval", "id": row["task_id"], "ids": encode_segments(tokenizer, segments),
                         "body": body, "prefix": decode_indentation(body, "python"), "preamble": "",
                         "tests": f"{row['test']}\ncheck({row['entry_point']})"})
    return problems


def complete_code(tokenizer, problem: dict, generated: list[int]) -> str | None:
    """Prefixo + continuação, cortada no ``<|endoftext|>`` ou na primeira sequência de parada."""
    eos = tokenizer.token_to_id(EOS)
    if eos in generated:
        generated = generated[: generated.index(eos)]
    full = decode_indentation(problem["body"] + decode_segments(tokenizer, generated), "python")
    if not full.startswith(problem["prefix"]):
        return None
    completion = full[len(problem["prefix"]) :]
    cut = min([i for i in (completion.find(s) for s in STOP) if i >= 0], default=len(completion))
    return problem["prefix"] + completion[:cut]


@torch.no_grad()
def sample_free(model, tokenizer, prompts: list[list[int]], device, temperature: float, max_tokens: int,
                done) -> list[list[int]]:
    """Amostragem sem gramática, com cache e preenchimento à esquerda; ``done(i, gerados)``
    diz quando uma sequência já passou de uma sequência de parada (conferido a cada 16 tokens)."""
    pad, eos = tokenizer.token_to_id(PAD), tokenizer.token_to_id(EOS)
    longest = max(len(p) for p in prompts)
    x = torch.tensor([[pad] * (longest - len(p)) + p for p in prompts], device=device)
    attention = torch.tensor([[0] * (longest - len(p)) + [1] * len(p) for p in prompts], device=device)
    positions = (attention.cumsum(1) - 1).clamp(min=0)
    generated: list[list[int]] = [[] for _ in prompts]
    finished = [False] * len(prompts)
    cache = None
    for step in range(max_tokens):
        with torch.autocast(device.type, dtype=torch.bfloat16):
            output = model(input_ids=x, attention_mask=attention, position_ids=positions, past_key_values=cache,
                           use_cache=True)
        cache = output.past_key_values
        logits = output.logits[:, -1].float()
        if temperature > 0:
            tokens = torch.multinomial((logits / temperature).softmax(-1), 1)[:, 0].tolist()
        else:
            tokens = logits.argmax(-1).tolist()
        for i, token in enumerate(tokens):
            if finished[i]:
                tokens[i] = pad
                continue
            generated[i].append(token)
            finished[i] = token == eos
        if step % 16 == 15:
            for i in range(len(prompts)):
                finished[i] = finished[i] or done(i, generated[i])
        if all(finished):
            break
        x = torch.tensor(tokens, device=device)[:, None]
        attention = torch.cat([attention, torch.ones_like(x)], 1)
        positions = positions[:, -1:] + 1
    synchronize(device)
    return generated


# --- os dois modos ------------------------------------------------------------

def generate(model, tokenizer, problems: list[dict], mode: str, device, temperature: float, samples: int,
             max_tokens: int, batch: int) -> list[list[str | None]]:
    """``samples`` códigos por problema, em lotes que encolhem com o comprimento (atenção eager)."""
    jobs = sorted([(i, j) for i in range(len(problems)) for j in range(samples)], key=lambda t: len(problems[t[0]]["ids"]))
    codes: list[list[str | None]] = [[None] * samples for _ in problems]
    first, start = 0, time.perf_counter()
    while first < len(jobs):
        longest = len(problems[jobs[min(first + batch, len(jobs)) - 1][0]]["ids"]) + max_tokens
        size = max(1, min(batch, int(batch * (768 / longest) ** 2)))
        chunk, first = jobs[first : first + size], first + size
        prompts = [problems[i]["ids"] for i, _ in chunk]
        if mode == "completar":
            outputs =sample_free(model, tokenizer, prompts, device, temperature, max_tokens, stop_check(tokenizer, problems, chunk))
            for (i, j), ids in zip(chunk, outputs):
                codes[i][j] = complete_code(tokenizer, problems[i], ids)
        else:
            rollouts, _ = sample_batch(model, tokenizer, prompts, device, temperature, max_tokens)
            for (i, j), rollout in zip(chunk, rollouts):
                if rollout.constraint.done:
                    try:
                        codes[i][j] = decode_response(tokenizer, rollout.response()).content
                    except ValueError:
                        pass
        print(f"  {min(first, len(jobs)):,}/{len(jobs):,} | lote {size} | {time.perf_counter() - start:.0f} s", flush=True)
    return codes


def stop_check(tokenizer, problems: list[dict], chunk: list[tuple[int, int]]):
    """Para a sequência quando a continuação já tem uma sequência de parada."""
    def done(n: int, ids: list[int]) -> bool:
        problem = problems[chunk[n][0]]
        full = decode_indentation(problem["body"] + decode_segments(tokenizer, ids), "python")
        return any(s in full[len(problem["prefix"]) :] for s in STOP)
    return done


def score_all(problems: list[dict], codes: list[list[str | None]]) -> list[int]:
    """Acertos por problema; códigos repetidos rodam uma vez só."""
    unique = {(i, code) for i, group in enumerate(codes) for code in group if code is not None}
    def run(item):
        i, code = item
        return item, syntax_ok(code) and passes(problems[i]["preamble"], code, problems[i]["tests"])
    with ThreadPoolExecutor(8) as pool:
        verdict = dict(pool.map(run, unique))
    return [sum(verdict.get((i, code), False) for code in group if code is not None) for i, group in enumerate(codes)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("completar", "chat"), required=True)
    parser.add_argument("--plan", help="temperatura:amostras, separados por vírgula "
                        "(padrão: 0.2:20,0.8:100 no completar; 0.2 a 1.0 com 20 no chat)")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--batch", type=int, default=96, help="sequências por lote para 768 tokens")
    parser.add_argument("--limit", type=int, help="só os primeiros problemas de cada conjunto (teste)")
    args = parser.parse_args()
    plan = args.plan or ("0.2:20,0.8:100" if args.mode == "completar" else "0.2:20,0.4:20,0.6:20,0.8:20,1.0:20")

    use_tokenizer_of(args.checkpoint)
    tokenizer, device = load_tokenizer(), pick_device()
    if args.mode == "completar":
        problems = completion_problems(tokenizer)
    else:
        problems = humaneval(tokenizer) + mbpp(tokenizer)
    if args.limit:
        problems = [p for suite in dict.fromkeys(p["suite"] for p in problems)
                    for p in [q for q in problems if q["suite"] == suite][: args.limit]]
    model = load_model(CHECKPOINT_DIR / args.checkpoint, device).eval()
    print(f"{args.checkpoint} | modo {args.mode} | {len(problems)} problemas | plano {plan}", flush=True)

    path = EVAL_DIR / "passk" / f"{args.checkpoint.replace('/', '-')}-{args.mode}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(path.read_text()) if path.exists() and not args.limit else {}
    for item in plan.split(","):
        temperature, samples = float(item.split(":")[0]), int(item.split(":")[1])
        codes = generate(model, tokenizer, problems, args.mode, device, temperature, samples, args.max_tokens, args.batch)
        correct = score_all(problems, codes)
        entry = {"samples": samples, "correct": dict(zip((p["id"] for p in problems), correct))}
        for suite in dict.fromkeys(p["suite"] for p in problems):
            hits = [c for p, c in zip(problems, correct) if p["suite"] == suite]
            entry[suite] = {f"pass@{k}": float(np.mean([pass_at_k(samples, c, k) for c in hits])) for k in KS if k <= samples}
            print(f"T={temperature:.1f} n={samples} | {suite}: " + " | ".join(
                f"{k} {100 * v:.1f}%" for k, v in entry[suite].items()), flush=True)
        results[f"{temperature:.1f}"] = entry
        if not args.limit:
            path.write_text(json.dumps(results, indent=1))
    print(f"gravado em {path}" if not args.limit else "teste: nada gravado")


if __name__ == "__main__":
    main()
