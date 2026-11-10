"""Exercícios do OpenCoder educational (MIT, pedidos fáceis com testes).

O conceito sai da AST da solução, com o prefixo ``oc-`` para não misturar com o gerador.
Parte desses dados repete o MBPP e o HumanEval: o que coincide com os benchmarks sai,
para que continuem medindo problemas nunca vistos.
"""

import ast
import hashlib
import re
from random import Random

from datasets import load_dataset

from dataprep.exercises import Exercise
from eval.sandbox import syntax_ok

REPO, CONFIG = "OpenCoder-LLM/opc-sft-stage2", "educational_instruct"
SOURCE = "opencoder"
MAX_CHARS = 1000
ALLOWED_IMPORTS = {"math", "collections", "itertools", "functools", "re", "string", "heapq", "bisect", "typing"}
FORBIDDEN = re.compile(r"\b(input|open|print|exit|eval|exec|random|time)\s*\(")
VALID_PERCENT = 5  # das funções
RL_PERCENT = 25  # das funções, reservadas ao GRPO: o SFT não as vê


def squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def anonymous(test: str, name: str) -> str:
    """Teste sem o nome da função, para reconhecer um problema renomeado."""
    return squeeze(re.sub(rf"\b{re.escape(name)}\b", "f", test))


def benchmark_keys() -> dict[str, set[str]]:
    keys = {"name": set(), "code": set(), "text": set(), "test": set()}
    for row in load_dataset("google-research-datasets/mbpp", "full", split="test"):
        functions = re.findall(r"def\s+(\w+)", row["code"])
        used = [f for f in functions if f in row["test_list"][0]]
        name = (used or functions or [""])[-1]
        keys["name"].add(name)
        keys["code"].add(squeeze(row["code"]))
        keys["text"].add(squeeze(row["text"]))
        keys["test"] |= {anonymous(t, name) for t in row["test_list"]}
    for row in load_dataset("openai/openai_humaneval", split="test"):
        keys["name"].add(row["entry_point"])
        keys["code"].add(squeeze(row["prompt"] + row["canonical_solution"]))
    return keys


def in_benchmark(keys: dict[str, set[str]], row: dict) -> bool:
    return (row["entry_point"] in keys["name"] or squeeze(row["code"]) in keys["code"]
            or squeeze(row["instruction"]) in keys["text"]
            or any(anonymous(t, row["entry_point"]) in keys["test"] for t in row["testcase"]))


def self_contained(code: str) -> bool:
    if not syntax_ok(code) or FORBIDDEN.search(code):
        return False
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] not in ALLOWED_IMPORTS for a in node.names):
            return False
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
            return False
    return True


def concept(code: str) -> str:
    nodes = list(ast.walk(ast.parse(code)))
    functions = sum(isinstance(n, (ast.FunctionDef, ast.Lambda)) for n in nodes)
    if functions > 1:
        return "oc-auxiliar"
    if any(isinstance(n, (ast.Dict, ast.DictComp)) for n in nodes):
        return "oc-dicionario"
    if any(isinstance(n, (ast.For, ast.While)) for n in nodes):
        return "oc-laco"
    if any(isinstance(n, (ast.ListComp, ast.GeneratorExp, ast.SetComp)) for n in nodes):
        return "oc-lista"
    if any(isinstance(n, ast.If) for n in nodes):
        return "oc-condicional"
    return "oc-expressao"


def split_of(name: str) -> str:
    """Pelo nome da função: o mesmo problema reescrito cai sempre na mesma parte."""
    bucket = int(hashlib.sha1(name.encode()).hexdigest(), 16) % 100
    return "valid" if bucket < VALID_PERCENT else "rl" if bucket < VALID_PERCENT + RL_PERCENT else "train"


def load(r: Random) -> list[Exercise]:
    keys = benchmark_keys()
    rows = list(load_dataset(REPO, CONFIG, split="train"))
    r.shuffle(rows)
    excluded = 0
    exercises, seen = [], set()
    for row in rows:
        code, name, tests = row["code"], row["entry_point"], row["testcase"]
        if name and tests and in_benchmark(keys, row):
            excluded += 1
            continue
        if (not name or not tests or len(code) > MAX_CHARS or re.search(r"\bdef\s+\w+\(self\b", code)
                or code in seen or not self_contained(code)):
            continue
        seen.add(code)
        prompt = f"{row['instruction'].strip()} The function must be named {name}."
        style = r.choice(("nome", "exemplos", "testes"))
        if style == "exemplos":
            prompt += "\nExamples:\n" + "\n".join(t.removeprefix("assert ").strip() for t in tests[:3])
        elif style == "testes":
            prompt += "\nYour code should pass these tests:\n" + "\n".join(tests[:3])
        exercises.append(Exercise(f"ex-oc-{row['seq_id']}", f"opencoder-{name}", concept(code), "en", prompt, code,
                                  list(tests), SOURCE, name))
    print(f"opencoder: {excluded:,} exemplos coincidem com os benchmarks e saíram", flush=True)
    return exercises
