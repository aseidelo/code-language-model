"""Perda das soluções oficiais do HumanEval e do MBPP, dado o enunciado (modelos do pré-treino).

Com modelos pequenos, o pass@k fica perto de zero e não distingue nada: é binário
e o modelo está longe de acertar. Esta medida é contínua: o modelo recebe o
enunciado e a solução oficial, no formato do pré-treino (como o modo
``completar`` do ``eval.passk``), e mede-se a perda média só nos tokens da
solução. Perda menor quer dizer que o modelo "espera" mais a resposta certa.

A medida principal é em **bits por byte**: a perda total nos tokens da solução,
em bits, dividida pelos bytes do texto da solução oficial. Assim modelos com
tokenizadores diferentes (v1 e v2, ver ``codelm.tokenizer``) são comparáveis: o
texto é o mesmo e o denominador também. Cada checkpoint usa o próprio
tokenizador. A perda média por token sai junto, só como referência.
Não garante mais acertos; é uma medida aproximada para quando o pass@k ainda
não funciona.

* HumanEval: enunciado = ``prompt`` (assinatura e docstring); solução =
  ``canonical_solution`` (o corpo da função).
* MBPP (``full``, ``test``): enunciado = descrição e testes num docstring de
  módulo; solução = ``code``.

A tokenização do programa inteiro tem de começar exatamente pelos tokens do
enunciado (com a indentação relativa, nem sempre acontece). Os problemas em que
isso falha ficam de fora, e também os maiores que ``max_tokens`` (o contexto da
Llama v1 é 2048; acima dele a perda explode). Os dois casos são contados.

Para comparar modelos, sai a diferença pareada problema a problema, com o erro
padrão (cada problema é uma observação), nos problemas válidos para todos os
tokenizadores envolvidos.

Uso:
    uv run python -m eval.solution_loss ab-llama-small/step-03814 ab-gemma3-small/step-03814
"""

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from codelm.chat_template import EOS, pretraining_segments
from codelm.indentation import DEDENT
from codelm.quantization import enable_qat
from codelm.tokenizer import checkpoint_tokenizer_path, encode_segments, load_tokenizer
from train.pretrain import CHECKPOINT_DIR
from train.smoke_test import pick_device


def prompt_ids(tokenizer, prompt: str) -> list[int]:
    """Enunciado no formato do pré-treino, sem fechar os blocos (o código continua)."""
    segments = pretraining_segments("python", prompt)
    while segments[-1][1] in (EOS, DEDENT):
        segments.pop()
    return encode_segments(tokenizer, segments)


def problems(tokenizer, max_tokens: int = 2048) -> tuple[list[dict], dict[str, int]]:
    rows = []
    for row in load_dataset("openai/openai_humaneval", split="test"):
        rows.append(("humaneval", row["task_id"], row["prompt"], row["canonical_solution"]))
    for row in load_dataset("google-research-datasets/mbpp", "full", split="test"):
        tests = "\n".join(row["test_list"])
        rows.append(("mbpp", f"mbpp-{row['task_id']}", f'"""\n{row["text"]}\n{tests}\n"""\n', row["code"]))

    result, skipped = [], {}
    for suite, problem_id, prompt, solution in rows:
        prefix = prompt_ids(tokenizer, prompt)
        try:
            full = encode_segments(tokenizer, pretraining_segments("python", prompt + solution))
        except Exception:
            full = []
        if full[: len(prefix)] != prefix or len(full) <= len(prefix):
            skipped[f"{suite} prefixo"] = skipped.get(f"{suite} prefixo", 0) + 1
            continue
        if len(full) > max_tokens:
            skipped[f"{suite} longo"] = skipped.get(f"{suite} longo", 0) + 1
            continue
        result.append({"suite": suite, "id": problem_id, "ids": full, "start": len(prefix),
                       "bytes": len(solution.encode("utf-8"))})
    return result, skipped


@torch.no_grad()
def losses(model, items: list[dict], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Por problema: bits por byte da solução e perda média por token (nats)."""
    bpb, per_token = [], []
    for item in items:
        x = torch.tensor([item["ids"]], device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(input_ids=x).logits[0, :-1].float()
        token_loss = F.cross_entropy(logits, x[0, 1:], reduction="none")[item["start"] - 1 :]
        bpb.append(token_loss.sum().item() / math.log(2) / item["bytes"])
        per_token.append(token_loss.mean().item())
    return np.array(bpb), np.array(per_token)


def load(checkpoint: str, device: torch.device, qat: bool):
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_DIR / checkpoint, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    if qat:
        enable_qat(model.model.layers)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", help="pastas em checkpoints/ (ex.: ab-llama-small/step-03814)")
    parser.add_argument("--no-qat", action="store_true", help="avalia sem a simulação Q4_0/Q8_0 do QAT")
    args = parser.parse_args()

    device = pick_device()
    # Problemas tokenizados com o tokenizador de cada checkpoint (uma vez por tokenizador).
    paths = {ck: checkpoint_tokenizer_path(ck) for ck in args.checkpoints}
    by_path = {path: problems(load_tokenizer(path)) for path in set(paths.values())}
    common = set.intersection(*({i["id"] for i in items} for items, _ in by_path.values()))
    for path, (items, skipped) in by_path.items():
        print(f"tokenizador {path.relative_to(path.parents[2])}: {len(items)} problemas | fora: {skipped}")
    order = sorted(common)
    reference = {i["id"]: i for i in next(iter(by_path.values()))[0]}
    suites = sorted({reference[k]["suite"] for k in order})
    suite_of = np.array([reference[k]["suite"] for k in order])
    print(f"comparados: {len(order)} problemas, válidos para todos os tokenizadores | QAT: {not args.no_qat}")

    results = {}
    for checkpoint in args.checkpoints:
        items = {i["id"]: i for i in by_path[paths[checkpoint]][0]}
        bpb, per_token = losses(load(checkpoint, device, not args.no_qat), [items[k] for k in order], device)
        results[checkpoint] = bpb
        means = " | ".join(f"{s} {bpb[suite_of == s].mean():.4f}" for s in suites)
        print(f"{checkpoint}: bits por byte: {means} | total {bpb.mean():.4f} "
              f"(perda por token {per_token.mean():.4f})", flush=True)

    names = list(results)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            print(f"\n{second} − {first}, em bits por byte (negativo = {second} melhor), média ± erro padrão:")
            for suite in [*suites, "total"]:
                mask = np.ones(len(order), bool) if suite == "total" else suite_of == suite
                diff = results[second][mask] - results[first][mask]
                se = diff.std(ddof=1) / np.sqrt(len(diff))
                print(f"  {suite:10s} {diff.mean():+.4f} ± {se:.4f}  ({diff.mean() / se:+.1f} erros padrão; "
                      f"{second} melhor em {100 * (diff < 0).mean():.0f}% dos problemas)")


if __name__ == "__main__":
    main()
