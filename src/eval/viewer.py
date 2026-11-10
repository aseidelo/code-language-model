"""Visualizador no terminal, um exemplo por tela.

* ``mbpp``: lê o ``.jsonl`` de ``eval.benchmark run`` e mostra o enunciado, os
  testes, a solução oficial e a resposta do modelo;
* ``valid``: mostra a validação do SFT (T1 a T5) em ordem aleatória, com o pedido,
  o código de entrada (T2 a T5), a referência e a resposta do modelo, gerada quando o exemplo abre.

* ``exercises``: mostra a validação dos exercícios, com o pedido, a solução, os
  testes e a resposta do modelo, gerada e executada quando o exemplo abre.

Teclas: ← → troca de exemplo, ↑ ↓ rola a tela, q sai.

Uso:
    uv run python -m eval.viewer mbpp base-sft-v1/step-06400 [--engine gguf] [--no-grammar] [--start 11]
    uv run --with llama-cpp-python python -m eval.viewer valid base-sft-v1/step-06400 [--task T1] [--lang python]
        [--nl en] [--level facil] [--engine gguf] [--no-grammar] [--seed 0] [--start 11]
    uv run --with llama-cpp-python python -m eval.viewer exercises base-sft-v1/step-06400 [--concept laco] [--start 11]
"""

import argparse
import curses
import json
import locale
import textwrap

import numpy as np
from datasets import load_dataset

from codelm.chat_template import CODE, CODE_END, RESPONSE, decode_response, render
from codelm.tokenizer import decode_code, decode_segments, load_tokenizer, use_tokenizer_of
from dataprep.sft_pack import LABELS, PACK_DIR
from eval.benchmark import ENGINES, variant_path
from eval.exercises import load_exercises, solved
from eval.generation import load_engine, load_generator

SECTION, RIGHT, WRONG = 1, 2, 3


def flag(name: str, value: bool) -> str:
    return f"{name}: {'sim' if value else 'não'}"


def load_mbpp(args) -> list[dict]:
    official = {f"mbpp-{r['task_id']}": r for r in load_dataset("google-research-datasets/mbpp", "full", split="test")}
    rows = [json.loads(l) for l in variant_path(args.checkpoint, args.engine, not args.no_grammar).read_text().splitlines()]
    items = []
    for row in rows:
        if row["suite"] == "mbpp":
            problem = official[row["id"]]
            items.append({
                "header": f"{row['id']} | {flag('correto', row['passed'])} | {flag('válido', row['valid'])} | "
                          f"{flag('terminou', row['finished'])} | {row['tokens']} tokens",
                "color": RIGHT if row["passed"] else WRONG,
                "sections": [("PERGUNTA", problem["text"]), ("TESTES", "\n".join(problem["test_list"])),
                             ("ESPERADO", problem["code"]),
                             ("MODELO", row["code"] if row["code"] is not None else "(sem resposta decodificável)")],
            })
    return items


def load_valid(args) -> list[dict]:
    data = np.load(PACK_DIR / "valid.npz")
    tokens, mask, starts = data["tokens"], data["mask"], data["starts"]
    names = json.loads((PACK_DIR / "meta.json").read_text())["labels"]
    labels = {k: np.array(names[k])[data[k]] for k in LABELS}
    chosen = np.ones(len(starts) - 1, dtype=bool)
    for key in LABELS:
        if getattr(args, key):
            chosen &= labels[key] == getattr(args, key)
    if not chosen.any():
        return []

    tokenizer = load_tokenizer()
    run = load_generator(load_engine(args.engine, args.checkpoint), tokenizer, args.max_tokens,
                         use_grammar=not args.no_grammar, temperature=0.0)
    eos, response_id = tokenizer.token_to_id("<|endoftext|>"), tokenizer.token_to_id(RESPONSE)
    code_id, code_end_id = tokenizer.token_to_id(CODE), tokenizer.token_to_id(CODE_END)

    def item(i: int) -> dict:
        ids = tokens[starts[i] : starts[i + 1]].astype(int).tolist()
        first = int(mask[starts[i] : starts[i + 1]].argmax())
        response_at = ids.index(response_id)
        code_at = ids.index(code_id) if code_id in ids[:response_at] else None
        sections = [("PEDIDO", tokenizer.decode(ids[1 : code_at or response_at]).strip())]
        if code_at:
            code = ids[code_at + 1 : ids.index(code_end_id)]
            sections.append(("CÓDIGO", decode_code(tokenizer, code, labels["lang"][i])))
        sections.append(("ESPERADO", decode_response(tokenizer, ids[response_at + 1 :]).content))

        def fill() -> None:
            try:
                generated = run(ids[:first])
                try:
                    answer = decode_response(tokenizer, generated).content
                except ValueError:
                    answer = render(decode_segments(tokenizer, generated))
                state = "terminou" if generated[-1] == eos else "cortada"
            except ValueError as error:  # a gramática recusou o token escolhido
                answer, generated, state = str(error), [], "erro"
            current["sections"][-1] = ("MODELO", answer)
            current["header"] += f" | {state} | {len(generated)} tokens"
            del current["fill"]

        current = {
            "header": " | ".join(labels[k][i] for k in LABELS),
            "color": 0,
            "sections": [*sections, ("MODELO", "(gerando...)")],
            "fill": fill,
        }
        return current

    return [item(int(i)) for i in np.random.default_rng(args.seed).permutation(np.flatnonzero(chosen))]


def load_exercise_items(args) -> list[dict]:
    rows = load_exercises(args.concept, seed=args.seed)
    if not rows:
        return []
    tokenizer = load_tokenizer()
    run = load_generator(load_engine(args.engine, args.checkpoint), tokenizer, args.max_tokens,
                         use_grammar=not args.no_grammar, temperature=0.0)

    def item(row: dict) -> dict:
        def fill() -> None:
            code, passed = solved(tokenizer, run, row)
            current["sections"][-1] = ("MODELO", code if code is not None else "(sem resposta decodificável)")
            current["header"] += f" | {flag('correto', passed)}"
            current["color"] = RIGHT if passed else WRONG
            del current["fill"]

        current = {"header": f"{row['concept']} | {row['template']} | {row['nl']}", "color": 0,
                   "sections": [("PEDIDO", row["prompt"]), ("TESTES", "\n".join(row["tests"])),
                                ("ESPERADO", row["response"]), ("MODELO", "(gerando...)")],
                   "fill": fill}
        return current

    return [item(row) for row in rows]


def page(item: dict, width: int) -> list[tuple[str, int]]:
    """Linhas da tela, cada uma com seu par de cores."""
    lines = []
    for title, body in item["sections"]:
        lines.append((title, SECTION))
        for line in body.replace("\0", "").splitlines() or [""]:
            lines.extend((part, 0) for part in textwrap.wrap(line, width, drop_whitespace=False, replace_whitespace=False) or [""])
        lines.append(("", 0))
    return lines


def view(screen, items: list[dict], current: int, title: str) -> None:
    curses.curs_set(0)
    curses.use_default_colors()
    curses.init_pair(SECTION, curses.COLOR_CYAN, -1)
    curses.init_pair(RIGHT, curses.COLOR_GREEN, -1)
    curses.init_pair(WRONG, curses.COLOR_RED, -1)
    scroll = 0
    while True:
        item = items[current]
        height, width = screen.getmaxyx()
        lines = page(item, max(width - 1, 20))
        room = max(height - 2, 1)
        scroll = min(scroll, max(len(lines) - room, 0))

        screen.erase()
        header = f"{title} | {current + 1:,}/{len(items):,} | {item['header']}"
        screen.addnstr(0, 0, header, width - 1, curses.A_BOLD | curses.color_pair(item["color"]))
        for row, (line, color) in enumerate(lines[scroll : scroll + room], 1):
            screen.addnstr(row, 0, line, width - 1, curses.color_pair(color) | (curses.A_BOLD if color else 0))
        last = min(scroll + room, len(lines))
        screen.addnstr(height - 1, 0, f"← → exemplo | ↑ ↓ rolar | q sair | linhas {scroll + 1}-{last} de {len(lines)}",
                       width - 1, curses.A_DIM)
        screen.refresh()

        if "fill" in item:
            item["fill"]()
            continue

        key = screen.getch()
        if key in (ord("q"), 27):
            return
        if key == curses.KEY_RIGHT:
            current, scroll = min(current + 1, len(items) - 1), 0
        elif key == curses.KEY_LEFT:
            current, scroll = max(current - 1, 0), 0
        elif key == curses.KEY_DOWN:
            scroll += 1
        elif key == curses.KEY_UP:
            scroll = max(scroll - 1, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    mbpp = commands.add_parser("mbpp")
    mbpp.add_argument("checkpoint", help="como base-sft-v1/step-06400")
    mbpp.add_argument("--engine", choices=ENGINES, default="gguf")
    mbpp.add_argument("--no-grammar", action="store_true")
    valid = commands.add_parser("valid")
    valid.add_argument("checkpoint", help="como base-sft-v1/step-06400")
    valid.add_argument("--engine", choices=ENGINES, default="gguf")
    valid.add_argument("--no-grammar", action="store_true")
    valid.add_argument("--max-tokens", type=int, default=512)
    valid.add_argument("--seed", type=int, default=0)
    for name in LABELS:
        valid.add_argument(f"--{name}")
    exercises = commands.add_parser("exercises")
    exercises.add_argument("checkpoint", help="como base-sft-v1/step-06400")
    exercises.add_argument("--engine", choices=ENGINES, default="gguf")
    exercises.add_argument("--no-grammar", action="store_true")
    exercises.add_argument("--max-tokens", type=int, default=256)
    exercises.add_argument("--seed", type=int, default=0)
    exercises.add_argument("--concept")
    for command in (mbpp, valid, exercises):
        command.add_argument("--start", type=int, default=1, help="posição do primeiro exemplo mostrado")
    args = parser.parse_args()
    use_tokenizer_of(args.checkpoint)

    items = {"mbpp": load_mbpp, "valid": load_valid, "exercises": load_exercise_items}[args.command](args)
    if not items:
        parser.error("nenhum exemplo com esses filtros")
    locale.setlocale(locale.LC_ALL, "")
    curses.wrapper(view, items, min(max(args.start, 1), len(items)) - 1, args.checkpoint)


if __name__ == "__main__":
    main()
