"""Gera os exercícios dos modelos, junta os do OpenCoder, valida cada um e monta o pacote do SFT.

Três partes, sem problema em comum entre elas (ver docs/grpo.md):

* ``train``: o SFT;
* ``rl``: o GRPO, que precisa de problemas que o SFT não viu (senão o modelo os decora
  e os grupos ficam uniformes);
* ``valid``: modelos de exercício (ou funções, no OpenCoder) nunca vistos pelos dois.

No gerador, a parte vem do modelo (``HELD_OUT``, ``RL_TEMPLATES``); no OpenCoder, do
nome da função. Os exercícios ficam em ``sft/exercises/`` com as colunas do banco, para
entrarem nele quando os dados finais forem gerados, mais ``concept``, ``template``,
``name`` e ``tests``. No pacote do SFT, ``level`` é o conceito, para o relatório por nível.

``--exclude-seen`` tira do GRPO o que um modelo já viu num SFT anterior (o piloto parte
do ``base-ex-v3``, treinado antes desta separação).

Uso:
    uv run python -m dataprep.exercises.build [--per-template 300] [--opencoder 30000] [--seed 0]
        [--exclude-seen data/codelm_sft/exercises-v3/train.parquet]
"""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from random import Random

import pyarrow as pa
import pyarrow.parquet as pq

from codelm.chat_template import encode_example
from codelm.tokenizer import load_tokenizer
from dataprep.exercises import Exercise, opencoder, valid_exercise
from dataprep.exercises.templates import TEMPLATES, Function, Module
from dataprep.sft_mix import SFT_DIR
from dataprep.sft_pack import LABELS, pack_split

EXERCISES_DIR = SFT_DIR / "exercises"
PACK_NAME = "pack-exercicios"
# Fixos, para a validação não mudar entre versões. Modelos gêmeos ficam juntos: se um
# deles fosse para o treino, a validação mediria semelhança, não generalização.
HELD_OUT = {
    "atribuicao-assign_int", "atribuicao-assign_two",
    "double", "doubled",
    "maximum", "minimum", "max_of_three",
    "sum_list", "product_list",
    "evens", "positives",
    "count_vowels", "count_char", "char_count",
    "average", "absolute", "remove_spaces",
}
MAX_TESTS = 5
MAX_TOKENS = 512
ATTEMPTS = 20  # por exercício pedido: modelos pequenos esgotam as combinações

FUNCTION_OPENERS = {
    "en": ("Write a {py}function {sig} that {s}.", "Create a {py}function named {name} that takes {params} and {s}.",
           "Define a {py}function {sig}. It {s}.", "I need a {py}function called {name}, with parameters {params}, that {s}."),
    "pt": ("Escreva uma função {py}{sig} que {s}.", "Crie uma função {py}chamada {name} que recebe {params} e {s}.",
           "Defina uma função {py}{sig}. Ela {s}.", "Preciso de uma função {py}chamada {name}, com parâmetros {params}, que {s}."),
}
PYTHON = {"en": "Python ", "pt": "em Python "}
MODULE_OPENERS = {"en": "Write Python code for this: {s}", "pt": "Escreva um código em Python para isto: {s}"}
EXAMPLES = {"en": "Examples:", "pt": "Exemplos:"}
TESTS = {"en": "Your code should pass these tests:", "pt": "Seu código deve passar nestes testes:"}


def function_exercise(template: Function, r: Random, nl: str) -> Exercise:
    name, params = r.choice(template.names), r.choice(template.params)
    cases = [*template.edges, *(template.inputs(r) for _ in range(MAX_TESTS))]
    cases = list({repr(c): c for c in cases}.values())[:MAX_TESTS]
    calls = [(f"{name}({', '.join(map(repr, c))})", template.reference(*c)) for c in cases]
    tests = [f"assert {call} == {expected!r}" for call, expected in calls]
    given = r.random() < 0.5  # a linguagem vem no cabeçalho; senão, o pedido cita Python
    prompt = r.choice(FUNCTION_OPENERS[nl]).format(py="" if given else PYTHON[nl], name=name, params=", ".join(params),
                                                   sig=f"{name}({', '.join(params)})",
                                                   s=template.statement(nl, params))
    style = r.choice(("nome", "exemplos", "testes"))
    if style == "exemplos":
        prompt += f"\n{EXAMPLES[nl]}\n" + "\n".join(f"{call} == {expected!r}" for call, expected in calls[:3])
    elif style == "testes":
        prompt += f"\n{TESTS[nl]}\n" + "\n".join(tests[:3])
    return Exercise("", template.id, template.concept, nl, prompt, template.solution(name, params), tests, "gerador",
                    name, given)


def module_exercise(template: Module, r: Random, nl: str) -> Exercise:
    en, pt, solution, expected = template.build(r)
    given = r.random() < 0.5
    statement = en if nl == "en" else pt
    prompt = statement if given else MODULE_OPENERS[nl].format(s=statement)
    return Exercise("", template.id, template.concept, nl, prompt, solution,
                    [f"assert {name} == {value!r}" for name, value in expected.items()], "gerador", None, given)


def generate(template: Function | Module, count: int, r: Random) -> list[Exercise]:
    build = function_exercise if isinstance(template, Function) else module_exercise
    seen = {}
    for _ in range(count * ATTEMPTS):
        if len(seen) == count:
            break
        exercise = build(template, r, r.choice(("en", "pt")))
        key = exercise.prompt + "\n".join(exercise.tests)
        if key not in seen:
            exercise.id = f"ex-{template.id}-{hashlib.sha1(key.encode()).hexdigest()[:16]}"
            seen[key] = exercise
    return list(seen.values())


RL_TEMPLATES: set[str] = set()  # modelos do gerador reservados ao GRPO


def split_of(exercise: Exercise) -> str:
    if exercise.source == opencoder.SOURCE:
        return opencoder.split_of(exercise.name)
    return "valid" if exercise.template in HELD_OUT else "rl" if exercise.template in RL_TEMPLATES else "train"


def bank_row(tokenizer, exercise: Exercise) -> dict:
    ids, _ = encode_example(tokenizer, exercise.prompt, exercise.solution, "python", "code",
                            lang_in_prompt=exercise.lang_in_prompt)
    return {"id": exercise.id, "task": "T1", "level": "facil", "lang": "python", "nl": exercise.nl,
            "source": exercise.source, "repo": f"exercicios/{exercise.template}", "path": exercise.template,
            "licenses": ["MIT"], "prompt": exercise.prompt, "code": None, "response": exercise.solution,
            "kind": "code", "lang_in_prompt": exercise.lang_in_prompt, "n_tokens": len(ids),
            "concept": exercise.concept, "template": exercise.template, "name": exercise.name, "tests": exercise.tests}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-template", type=int, default=300)
    parser.add_argument("--opencoder", type=int, default=30000, help="limite de exercícios do OpenCoder no SFT")
    parser.add_argument("--exclude-seen", help="exercícios já vistos num SFT anterior, que saem do GRPO")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    r = Random(args.seed)
    exercises = [e for template in TEMPLATES for e in generate(template, args.per_template, r)]
    taken = Counter()
    for exercise in opencoder.load(r):
        part = split_of(exercise)
        if part != "train" or taken[part] < args.opencoder:
            taken[part] += 1
            exercises.append(exercise)
    with ThreadPoolExecutor(8) as pool:
        verdicts = list(pool.map(valid_exercise, exercises))
    rejected = Counter(e.source for e, ok in zip(exercises, verdicts) if not ok)
    print(f"{len(exercises):,} exercícios, {sum(verdicts):,} válidos; rejeitados: {dict(rejected)}", flush=True)

    tokenizer = load_tokenizer()
    rows = [bank_row(tokenizer, e) for e, ok in zip(exercises, verdicts) if ok]
    rows = [x for x in rows if x["n_tokens"] <= MAX_TOKENS]
    r.shuffle(rows)
    parts = {e.id: split_of(e) for e in exercises}
    splits = {part: [x for x in rows if parts[x["id"]] == part] for part in ("train", "valid", "rl")}
    if args.exclude_seen:
        seen = pq.read_table(args.exclude_seen, columns=["name", "response"]).to_pylist()
        names, codes = {x["name"] for x in seen if x["name"]}, {opencoder.squeeze(x["response"]) for x in seen}
        is_seen = [x["name"] in names or opencoder.squeeze(x["response"]) in codes for x in splits["rl"]]
        splits["rl-vistos"] = [x for x, s in zip(splits["rl"], is_seen) if s]  # o GRPO do piloto usa os dois
        splits["rl"] = [x for x, s in zip(splits["rl"], is_seen) if not s]
        print(f"rl: {len(splits['rl-vistos']):,} exercícios já vistos foram para rl-vistos", flush=True)
    for first, second in (("train", "valid"), ("train", "rl"), ("rl", "valid")):
        assert not {x["template"] for x in splits[first]} & {x["template"] for x in splits[second]}, (first, second)

    EXERCISES_DIR.mkdir(parents=True, exist_ok=True)
    directory = SFT_DIR / PACK_NAME
    directory.mkdir(parents=True, exist_ok=True)
    packed = {name: [{**x, "level": x["concept"]} for x in splits[name]] for name in ("train", "valid")}
    names = {k: sorted({x[k] for group in packed.values() for x in group}) for k in LABELS}
    for name, group in splits.items():
        pq.write_table(pa.Table.from_pylist(group), EXERCISES_DIR / f"{name}.parquet")
        if name in packed:
            pack_split(tokenizer, packed[name], names, directory / f"{name}.npz")
        print(f"{name}: {len(group):,} exemplos, {sum(x['n_tokens'] for x in group) / 1e6:.2f}M tokens, "
              f"{dict(sorted(Counter(x['concept'] for x in group).items()))}", flush=True)
    (directory / "meta.json").write_text(json.dumps({"labels": names, "held_out": sorted(HELD_OUT)}, indent=1))


if __name__ == "__main__":
    main()
