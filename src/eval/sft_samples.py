"""Gera respostas para exemplos da validação do SFT, para leitura.

Sorteia ``--per-task`` exemplos de cada tarefa e grava pedido, referência e
resposta em ``sft/eval/<run>-<passo>.md``.

Uso:
    uv run python -m eval.sft_samples --checkpoint small-sft-v1/step-04601
"""

import argparse
import json
import time

import numpy as np

from codelm.chat_template import decode_response, render
from codelm.tokenizer import decode_segments, load_tokenizer, use_tokenizer_of
from dataprep.sft_mix import SFT_DIR
from dataprep.sft_pack import PACK_DIR
from eval.generation import ENGINES, load_engine, load_generator

EVAL_DIR = SFT_DIR / "eval"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="caminho em checkpoints/, como run/step-00100")
    parser.add_argument("--per-task", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engine", choices=ENGINES, default="qat")
    args = parser.parse_args()

    use_tokenizer_of(args.checkpoint)
    tokenizer = load_tokenizer()
    engine = load_engine(args.engine, args.checkpoint)
    run = load_generator(engine, tokenizer, args.max_tokens, temperature=args.temperature)
    data = np.load(PACK_DIR / "valid.npz")
    names = json.loads((PACK_DIR / "meta.json").read_text())["labels"]
    response_id = tokenizer.token_to_id("<|response|>")
    rng = np.random.default_rng(args.seed)

    sections = []
    for task, name in enumerate(names["task"]):
        for i in rng.choice(np.flatnonzero(data["task"] == task), args.per_task, replace=False):
            ids = data["tokens"][data["starts"][i] : data["starts"][i + 1]].astype(int).tolist()
            first = int(data["mask"][data["starts"][i] : data["starts"][i + 1]].argmax())
            start = time.perf_counter()
            try:
                generated = run(ids[:first])
                try:
                    answer = decode_response(tokenizer, generated).content
                except ValueError:
                    answer = render(decode_segments(tokenizer, generated))
                state = "fim" if generated[-1] == tokenizer.token_to_id("<|endoftext|>") else "cortada"
            except ValueError as error:  # a gramática recusou o token escolhido
                answer, generated, state = str(error), [], "erro"
            speed = len(generated) / (time.perf_counter() - start)
            reference = render(decode_segments(tokenizer, ids[ids.index(response_id) + 1 :]))
            lang, nl = names["lang"][data["lang"][i]], names["nl"][data["nl"][i]]
            print(f"{name} {lang} {nl}: {len(generated)} tokens, {state}, {speed:.0f} tokens/s", flush=True)
            sections.append(f"## {name} | {lang} | {nl} | {state}\n\n### Entrada\n\n````\n"
                            f"{render(decode_segments(tokenizer, ids[:first]))}\n````\n\n### Referência\n\n````\n"
                            f"{reference}\n````\n\n### Gerado\n\n````\n{answer}\n````\n")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"{args.checkpoint.replace('/', '-')}.md"
    path.write_text("\n".join(sections))
    print(f"gravado em {path}")


if __name__ == "__main__":
    main()
