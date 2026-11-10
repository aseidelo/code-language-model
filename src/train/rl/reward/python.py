"""Adaptador Python da recompensa (ver docs/grpo.md).

Uma execução no sandbox por resposta: carrega o código, roda cada teste e conta, com
``sys.monitoring``, as instruções de bytecode da resposta executadas nos testes (as do
próprio teste ficam de fora). A contagem é determinística, ao contrário do tempo;
chamadas de funções prontas contam como uma. Tempo esgotado conta como não compilar:
a execução é uma só, e não sobra o resultado dos outros testes.
"""

import ast
import io
import json
import re
import threading
import tokenize

from pyflakes.checker import Checker

from eval.sandbox import execute, syntax_ok
from train.rl.reward import Scores

LINT_LIMIT = 3  # avisos do pyflakes que zeram o item
COMMENTS = (0.05, 0.30)  # faixa da proporção de linhas com comentário
MARKER = "__recompensa__"
# "sem usar max", nos idiomas do SFT; o exercício também pode trazer ``forbidden``.
FORBIDDEN = re.compile(r"(?:without using|sem usar|sin usar|sans utiliser|senza usare|ohne)\s+`?(\w+)`?")

HARNESS = """
import json, sys
__tests = {tests}
__result = {{"loaded": False, "passed": [], "instructions": 0, "error": None}}
__namespace = {{"__name__": "__main__"}}
try:
    exec(compile({code}, "<resposta>", "exec"), __namespace)
    __result["loaded"] = True
except BaseException as error:
    __result["error"] = type(error).__name__
if __result["loaded"]:
    __monitoring = sys.monitoring
    __tool = __monitoring.PROFILER_ID
    __monitoring.use_tool_id(__tool, "recompensa")
    __count = [0]
    def __instruction(code, offset):
        if code.co_filename == "<resposta>":
            __count[0] += 1
    __monitoring.register_callback(__tool, __monitoring.events.INSTRUCTION, __instruction)
    for __test in __tests:
        __compiled = compile(__test, "<teste>", "exec")
        __monitoring.set_events(__tool, __monitoring.events.INSTRUCTION)
        try:
            exec(__compiled, __namespace)
            __passed = True
        except BaseException:
            __passed = False
        __monitoring.set_events(__tool, 0)
        __result["passed"].append(__passed)
    __result["instructions"] = __count[0]
print("{marker}" + json.dumps(__result))
"""


def run(code: str, tests: list[str]) -> dict | None:
    """Carregou, quais testes passaram e quantas instruções; None se estourou o tempo ou travou."""
    done = execute(HARNESS.format(tests=repr(tests), code=repr(code), marker=MARKER))
    if done is None:
        return None
    line = next((l for l in done[1].decode(errors="replace").splitlines()[::-1] if l.startswith(MARKER)), None)
    return json.loads(line[len(MARKER):]) if line else None


def forbidden_names(exercise: dict) -> set[str]:
    return set(exercise.get("forbidden") or FORBIDDEN.findall(exercise.get("prompt", "")))


def uses(tree: ast.AST, names: set[str]) -> bool:
    return any(isinstance(n, ast.Name) and n.id in names or isinstance(n, ast.Attribute) and n.attr in names
               for n in ast.walk(tree))


def readability(code: str, tree: ast.AST, name: str | None) -> tuple[float, dict]:
    """Média dos itens que se aplicam: tipos, docstring, linter e comentários."""
    items = {}
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if functions:
        slots = typed = 0
        for function in functions:
            params = [a for a in function.args.posonlyargs + function.args.args + function.args.kwonlyargs
                      if a.arg not in ("self", "cls")]
            slots += len(params) + 1
            typed += sum(a.annotation is not None for a in params) + (function.returns is not None)
        items["tipos"] = typed / slots
    target = next((f for f in functions if f.name == name), None) if name else None
    if target is not None:
        items["docstring"] = float(ast.get_docstring(target) is not None)
    items["linter"] = max(0.0, 1 - len(Checker(tree).messages) / LINT_LIMIT)
    lines = [l for l in code.splitlines() if l.strip()]
    commented = {t.start[0] for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type == tokenize.COMMENT}
    ratio = len(commented) / len(lines) if lines else 0.0
    items["comentarios"] = float(COMMENTS[0] <= ratio <= COMMENTS[1])
    return sum(items.values()) / len(items), {**items, "proporcao_comentarios": ratio}


class PythonReward:
    def __init__(self) -> None:
        self.best: dict[str, int] = {}  # menor contagem de instruções correta por exercício
        self.solutions: dict[str, int | None] = {}  # contagem da solução do exercício
        self.lock = threading.Lock()

    def reference(self, exercise: dict) -> int | None:
        key = exercise["id"]
        if key not in self.solutions:
            result = run(exercise["response"], list(exercise["tests"]))
            solved = result and result["loaded"] and all(result["passed"])
            self.solutions[key] = result["instructions"] if solved else None
        candidates = [v for v in (self.solutions[key], self.best.get(key)) if v]
        return min(candidates) if candidates else None

    def score(self, code: str | None, exercise: dict) -> Scores:
        tests = list(exercise["tests"])
        if not code or not syntax_ok(code):
            return Scores(-1.0, 0.0, 0.0, {"motivo": "incompleto ou sintaxe inválida"})
        result = run(code, tests)
        if result is None or not result["loaded"]:
            reason = "tempo esgotado" if result is None else f"erro ao carregar: {result['error']}"
            return Scores(-1.0, 0.0, 0.0, {"motivo": reason})

        tree = ast.parse(code)
        legibility, items = readability(code, tree, exercise.get("name"))
        # resposta errada não ganha por ser legível: a readability fica só nos detalhes
        details = {"testes": f"{sum(result['passed'])}/{len(tests)}", "instrucoes": result["instructions"],
                   "legibilidade_bruta": legibility, **items}
        banned = forbidden_names(exercise)
        if banned and uses(tree, banned):
            return Scores(0.0, 0.0, 0.0, {**details, "motivo": f"usa nome proibido: {', '.join(sorted(banned))}"})
        if sum(result["passed"]) < len(tests):
            return Scores(0.0, 0.0, 0.0, details)

        instructions = max(result["instructions"], 1)
        with self.lock:
            reference = self.reference(exercise)
            self.best[exercise["id"]] = min(self.best.get(exercise["id"], instructions), instructions)
        efficiency = min(1.0, reference / instructions) if reference else 1.0
        return Scores(1.0, legibility, efficiency, {**details, "referencia": reference})

    def state(self) -> dict:
        return {"best": self.best, "solutions": self.solutions}

    def load(self, state: dict) -> None:
        self.best.update(state["best"])
        self.solutions.update(state["solutions"])
