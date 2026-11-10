"""Benchmarks do SFT: compara motores (float, QAT, Q4_0 real) e a máscara de gramática.

* ``loss``: perda na validação em cada motor e a distância entre os logits do QAT
  simulado e os do GGUF, o teste direto de que a simulação bate com o llama.cpp;
* ``run``: gera as respostas de uma variante e pontua (use ``gguf`` e ``f16``: na
  GPU, um token por vez fica dez vezes mais lento que na CPU);
* ``compare``: tabela das variantes e comparação pareada problema a problema.

Conjuntos: MBPP e HumanEval (Python, executados com os testes oficiais) e T4 da
validação (JSON, a outra linguagem com gramática). O código gerado é do modelo,
não confiável: roda com ``bwrap``, sem rede e sem acesso ao disco.

Uso (o motor gguf precisa de ``uv run --with llama-cpp-python``):
    uv run python -m eval.benchmark loss --checkpoint small-sft-v1/step-04601
    uv run python -m eval.benchmark run --checkpoint small-sft-v1/step-04601 --engine qat [--no-grammar]
    uv run python -m eval.benchmark run --checkpoint grpo-v3/step-00500 --engine batch  # em lote na GPU
    uv run python -m eval.benchmark compare --checkpoint small-sft-v1/step-04601
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from datasets import load_dataset

from codelm.chat_template import decode_response, encode_prompt
from codelm.model import CONTEXT_LENGTH
from codelm.tokenizer import load_tokenizer, use_tokenizer_of
from dataprep.sft_pack import PACK_DIR
from eval.generation import ENGINES, load_engine, load_generator
from eval.sandbox import passes, syntax_ok
from eval.sft_samples import EVAL_DIR

T4_FIELDS = {"name", "summary", "parameters", "returns", "side_effects", "complexity", "dependencies"}


def mbpp(tokenizer) -> list[dict]:
    problems = []
    for row in load_dataset("google-research-datasets/mbpp", "full", split="test"):
        tests = "\n".join(row["test_list"])
        prompt = f"{row['text']} Your code should pass these tests:\n{tests}"
        problems.append({"suite": "mbpp", "id": f"mbpp-{row['task_id']}", "preamble": row["test_setup_code"],
                         "tests": tests, "ids": encode_prompt(tokenizer, prompt, "python")})
    return problems


def humaneval(tokenizer) -> list[dict]:
    problems = []
    for row in load_dataset("openai/openai_humaneval", split="test"):
        # A resposta repete a assinatura (como em T2), mas os imports do enunciado ficam com ela.
        preamble = "\n".join(l for l in row["prompt"].split("def " + row["entry_point"])[0].splitlines()
                             if l.startswith(("import ", "from ")))
        ids = encode_prompt(tokenizer, "Implement the following function as documented.", "python",
                            code=row["prompt"])
        problems.append({"suite": "humaneval", "id": row["task_id"], "preamble": preamble,
                         "tests": f"{row['test']}\ncheck({row['entry_point']})", "ids": ids})
    return problems


def t4_valid(limit: int = 300, seed: int = 0) -> list[dict]:
    data = np.load(PACK_DIR / "valid.npz")
    names = json.loads((PACK_DIR / "meta.json").read_text())["labels"]
    chosen = np.flatnonzero(data["task"] == names["task"].index("T4"))
    problems = []
    for i in np.random.default_rng(seed).permutation(chosen)[:limit]:
        tokens = data["tokens"][data["starts"][i] : data["starts"][i + 1]].astype(int).tolist()
        first = int(data["mask"][data["starts"][i] : data["starts"][i + 1]].argmax())
        problems.append({"suite": "t4", "id": f"t4-{i}", "ids": tokens[:first]})
    return problems


def score(problem: dict, code: str | None) -> dict:
    if code is None:
        return {"valid": False, "passed": False}
    if problem["suite"] == "t4":
        try:
            value = json.loads(code)
        except ValueError:
            return {"valid": False, "passed": False}
        return {"valid": True, "passed": isinstance(value, dict) and set(value) == T4_FIELDS}
    valid = syntax_ok(code)
    return {"valid": valid, "passed": valid and passes(problem["preamble"], code, problem["tests"])}


def variant_path(checkpoint: str, engine: str, grammar: bool):
    name = f"{checkpoint.replace('/', '-')}-{engine}-{'mascara' if grammar else 'livre'}.jsonl"
    return EVAL_DIR / "benchmark" / name


def run(args) -> None:
    tokenizer = load_tokenizer()
    problems = mbpp(tokenizer) + humaneval(tokenizer) + t4_valid()
    if args.limit:
        problems = [p for p in problems[:: len(problems) // args.limit]]
    rows = run_batched(args, tokenizer, problems) if args.engine == "batch" else run_serial(args, tokenizer, problems)

    with ThreadPoolExecutor(8) as pool:
        scores = list(pool.map(score, problems, [r["code"] for r in rows]))
    path = variant_path(args.checkpoint, args.engine, not args.no_grammar)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({**r, **s}, ensure_ascii=False) + "\n" for r, s in zip(rows, scores)))
    for suite in sorted({p["suite"] for p in problems}):
        passed = [s["passed"] for p, s in zip(problems, scores) if p["suite"] == suite]
        print(f"{suite}: {sum(passed)}/{len(passed)} ({100 * np.mean(passed):.1f}%)")
    print(f"gravado em {path}")


def run_batched(args, tokenizer, problems: list[dict]) -> list[dict]:
    """T=0 em lotes na GPU, com o QAT e a máscara do GRPO: minutos em vez de ~1,5 h do gguf na CPU."""
    from train.grpo import load_model, sample_batch
    from train.pretrain import CHECKPOINT_DIR
    from train.smoke_test import pick_device

    device = pick_device()
    model = load_model(CHECKPOINT_DIR / args.checkpoint, device).eval()
    order = sorted(range(len(problems)), key=lambda i: len(problems[i]["ids"]))  # lotes de tamanho parecido
    rows: list[dict | None] = [None] * len(problems)
    start, first = time.perf_counter(), 0
    while first < len(order):
        # a atenção eager é quadrática: o lote cai com o comprimento (``batch`` vale para 768 tokens)
        longest = len(problems[order[min(first + args.batch, len(order)) - 1]]["ids"]) + args.max_tokens
        size = max(1, min(args.batch, int(args.batch * (768 / longest) ** 2)))
        chunk = order[first : first + size]
        first += size
        rollouts, _ = sample_batch(model, tokenizer, [problems[i]["ids"] for i in chunk], device, 0.0,
                                   args.max_tokens, use_grammar=not args.no_grammar)
        for i, rollout in zip(chunk, rollouts):
            code = None
            if rollout.constraint.done:
                try:
                    code = decode_response(tokenizer, rollout.response()).content
                except ValueError:
                    pass
            rows[i] = {"id": problems[i]["id"], "suite": problems[i]["suite"], "code": code,
                       "finished": rollout.constraint.done, "tokens": len(rollout.generated), "seconds": None}
        print(f"{first}/{len(order)} | lote {size} | {time.perf_counter() - start:.0f} s", flush=True)
    return rows


def run_serial(args, tokenizer, problems: list[dict]) -> list[dict]:
    eos = tokenizer.token_to_id("<|endoftext|>")
    generator = load_generator(load_engine(args.engine, args.checkpoint), tokenizer, args.max_tokens,
                               use_grammar=not args.no_grammar, temperature=0.0)
    rows, start = [], time.perf_counter()
    for n, problem in enumerate(problems, 1):
        begin, code = time.perf_counter(), None
        try:
            generated = generator(problem["ids"])
            finished = generated[-1] == eos
            try:
                code = decode_response(tokenizer, generated).content
            except ValueError:
                pass
        except ValueError:  # a gramática não aceitou nenhum token
            generated, finished = [], False
        rows.append({"id": problem["id"], "suite": problem["suite"], "code": code, "finished": finished,
                     "tokens": len(generated), "seconds": time.perf_counter() - begin})
        if n % 25 == 0:
            print(f"{n}/{len(problems)} | {sum(r['tokens'] for r in rows) / (time.perf_counter() - start):.1f} tokens/s",
                  flush=True)
    return rows


def compare(args) -> None:
    variants = {}
    for engine in ENGINES:
        for grammar in (True, False):
            path = variant_path(args.checkpoint, engine, grammar)
            if path.exists():
                rows = [json.loads(l) for l in path.read_text().splitlines()]
                variants[f"{engine} {'máscara' if grammar else 'livre'}"] = {r["id"]: r for r in rows}

    print(f"{'variante':<16}{'conjunto':<11}{'n':>5}{'válido':>9}{'correto':>9}{'terminou':>10}{'tokens/s':>10}")
    for name, rows in variants.items():
        for suite in ("mbpp", "humaneval", "t4"):
            group = [r for r in rows.values() if r["suite"] == suite]
            if group:
                rate = lambda key: 100 * sum(r[key] for r in group) / len(group)
                speed = sum(r["tokens"] for r in group) / sum(r["seconds"] for r in group)
                print(f"{name:<16}{suite:<11}{len(group):>5}{rate('valid'):>8.1f}%{rate('passed'):>8.1f}%"
                      f"{rate('finished'):>9.1f}%{speed:>10.1f}")

    print("\npares (só a primeira acerta / só a segunda acerta / código idêntico):")
    pairs = [("f16 máscara", "gguf máscara"), ("f16 livre", "gguf livre"), ("f16 máscara", "f16 livre"),
             ("gguf máscara", "gguf livre")]
    for first, second in pairs:
        if first in variants and second in variants:
            a, b = variants[first], variants[second]
            shared = [i for i in a if i in b]
            only_a = sum(a[i]["passed"] and not b[i]["passed"] for i in shared)
            only_b = sum(b[i]["passed"] and not a[i]["passed"] for i in shared)
            same = sum(a[i]["code"] == b[i]["code"] for i in shared)
            print(f"  {first} x {second}: {only_a} / {only_b} / {same} de {len(shared)}")


def loss(args) -> None:
    data = np.load(PACK_DIR / "valid.npz")
    chosen = np.random.default_rng(0).permutation(len(data["starts"]) - 1)[: args.limit]
    examples = []
    for i in chosen:
        tokens = data["tokens"][data["starts"][i] : data["starts"][i + 1]].astype(np.int64)
        mask = data["mask"][data["starts"][i] : data["starts"][i + 1]] > 0
        if len(tokens) <= CONTEXT_LENGTH:
            examples.append((tokens, mask))

    outputs = {}
    for kind in args.engines.split(","):
        engine, start, total, count, picked = load_engine(kind, args.checkpoint), time.perf_counter(), 0.0, 0, []
        for tokens, mask in examples:
            engine.reset()
            logits = engine.feed(tokens[:-1].tolist()).astype(np.float64)
            logits -= logits.max(1, keepdims=True)
            log_probs = logits - np.log(np.exp(logits).sum(1, keepdims=True))
            generated = mask[1:]
            total -= log_probs[np.arange(len(logits))[generated], tokens[1:][generated]].sum()
            count += int(generated.sum())
            picked.append(log_probs[generated][:: 16].astype(np.float32))
        outputs[kind] = picked
        print(f"{kind}: perda {total / count:.4f} em {count:,} tokens ({time.perf_counter() - start:.0f} s)", flush=True)

    keys = list(outputs)
    for i, first in enumerate(keys):
        for second in keys[i + 1 :]:
            p, q = np.concatenate(outputs[first]), np.concatenate(outputs[second])
            kl = float((np.exp(p) * (p - q)).sum(1).mean())
            agree = float((p.argmax(1) == q.argmax(1)).mean())
            print(f"{first} x {second}: KL médio {kl:.5f} nats | mesmo token mais provável em {100 * agree:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("loss", "run", "compare"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        if name == "loss":
            command.add_argument("--engines", default="float,qat,gguf")
            command.add_argument("--limit", type=int, default=300)
        if name == "run":
            command.add_argument("--engine", choices=ENGINES + ("batch",), default="qat")
            command.add_argument("--batch", type=int, default=64, help="sequências por lote no motor batch")
            command.add_argument("--no-grammar", action="store_true")
            command.add_argument("--max-tokens", type=int, default=512)
            command.add_argument("--limit", type=int, help="usa só cerca desse número de problemas (teste)")
    args = parser.parse_args()
    use_tokenizer_of(args.checkpoint)
    {"loss": loss, "run": run, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
