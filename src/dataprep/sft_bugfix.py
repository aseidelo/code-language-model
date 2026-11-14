"""T6, corrigir bug: exemplos procedurais a partir de soluções com testes (ver docs/sft-v2.md).

Uma mutação introduz um bug numa solução Python que passa nos testes. O mutante
só entra se reprovar pelo menos um teste. A mutação troca um trecho do texto
(posição vinda dos tokens ou da AST): o resto do código fica idêntico, e a
correção é a menor diferença possível.

Todos os mutantes de uma solução rodam numa única execução do sandbox, cada um
com limite de tempo próprio. Mutantes que estouram o tempo (laço infinito)
são descartados.

    uv run python -m dataprep.sft_bugfix <fonte> [<fonte> ...] [--limit N]

``<fonte>``: uma fonte convertida por ``dataprep.sft_convert`` com testes
(``exercicios-v1``, ``kodcode``, ``open-code-instruct``, ``opc-stage2-educational``,
``acecode-87k``). Só o split ``train``. A saída vai para ``fontes/t6-<fonte>/``, uma
parte por parte da fonte (retomável).

Os testes são trechos executados um a um depois de ``test_setup`` + solução: um
``assert`` solto ou uma função de teste do pytest seguida da chamada. Na variante (c),
o esperado e o obtido vêm do ``assert`` que falhou, avaliado com as variáveis locais
da função de teste.
"""

import argparse
import ast
import hashlib
import io
import json
import re
import subprocess
import time
import tokenize
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

import pyarrow as pa
import pyarrow.parquet as pq

from dataprep.sft_mix import SFT_DIR
from eval.sandbox import MEMORY_LIMIT_KB, SANDBOX, syntax_ok

FONTES_DIR = SFT_DIR / "v2" / "fontes"
CANDIDATES_PER_SOLUTION = 24  # mutantes testados por solução
BUGS_PER_SOLUTION = 3  # mutantes que viram exemplos
MUTANT_TIME_LIMIT = 1.0  # segundos por mutante
RUN_TIME_LIMIT = 60  # segundos por solução (todos os mutantes)
SOLUTIONS_PER_RUN = 16  # soluções por execução do sandbox (a inicialização do bwrap pesa)
MARK = "__mutantes__"

# --- mutações -------------------------------------------------------------------

TOKEN_SWAPS = {
    "comparador": {"<": ["<=", ">"], "<=": ["<", ">="], ">": [">=", "<"], ">=": [">", "<="],
                   "==": ["!="], "!=": ["=="]},
    "aritmetico": {"+": ["-"], "-": ["+"], "//": ["/", "%"], "/": ["//"], "%": ["//"], "*": ["+"],
                   "+=": ["-="], "-=": ["+="], "*=": ["+="]},
    "logico": {"and": ["or"], "or": ["and"], "True": ["False"], "False": ["True"], "is": ["is not"],
               "in": ["not in"]},
}


@dataclass
class Mutant:
    operator: str
    code: str


@dataclass
class Edit:
    """Troca do texto entre dois offsets (em caracteres) do código."""
    start: int
    end: int
    text: str
    operator: str


@dataclass
class Source:
    code: str
    lines: list[str] = field(init=False)
    starts: list[int] = field(init=False)

    def __post_init__(self):
        self.lines = self.code.splitlines(keepends=True)
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))

    def char(self, lineno: int, col: int) -> int:
        """Offset de (linha a partir de 1, coluna em caracteres)."""
        return self.starts[lineno - 1] + col

    def node_char(self, lineno: int, col_bytes: int) -> int:
        """Offset de uma posição da AST (coluna em bytes UTF-8)."""
        line = self.lines[lineno - 1]
        return self.starts[lineno - 1] + len(line.encode()[:col_bytes].decode("utf-8", "replace"))

    def span(self, node: ast.AST) -> tuple[int, int]:
        return self.node_char(node.lineno, node.col_offset), self.node_char(node.end_lineno, node.end_col_offset)

    def text(self, start: int, end: int) -> str:
        return self.code[start:end]

    def apply(self, edit: Edit) -> str:
        return self.code[:edit.start] + edit.text + self.code[edit.end:]


def token_edits(src: Source) -> list[Edit]:
    edits = []
    tokens = tokenize.generate_tokens(io.StringIO(src.code).readline)
    try:
        for tok in tokens:
            if tok.type not in (tokenize.OP, tokenize.NAME):
                continue
            for operator, swaps in TOKEN_SWAPS.items():
                for new in swaps.get(tok.string, ()):
                    edits.append(Edit(src.char(*tok.start), src.char(*tok.end), new, operator))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return edits


def nudge(src: Source, node: ast.AST, operator: str) -> list[Edit]:
    """±1 numa expressão: literal inteiro muda de valor, outra expressão ganha ``+ 1``/``- 1``."""
    start, end = src.span(node)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return [Edit(start, end, str(node.value + d), operator) for d in (1, -1) if node.value + d >= 0]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return []  # -1: o literal dentro já é mutado
    return [Edit(end, end, f" {sign} 1", operator) for sign in "+-"]


def local_names(function: ast.AST) -> list[str]:
    names = {a.arg for a in ast.walk(function.args) if isinstance(a, ast.arg)} if hasattr(function, "args") else set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return sorted(names - {"self", "cls"})


def ast_edits(src: Source, tree: ast.AST) -> list[Edit]:
    edits = []
    for node in ast.walk(tree):
        # off-by-one no limite de range()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range" and node.args:
            stop = node.args[0] if len(node.args) == 1 else node.args[1]
            edits += nudge(src, stop, "off_by_one")
        # índices e fatias
        elif isinstance(node, ast.Subscript):
            index = node.slice
            if isinstance(index, ast.Slice):
                edits += [e for part in (index.lower, index.upper) if part is not None
                          for e in nudge(src, part, "fatia")]
            elif isinstance(index, (ast.Constant, ast.Name, ast.BinOp)):
                edits += nudge(src, index, "off_by_one")
        # constantes inteiras soltas (fora dos casos acima, que já se repetem sem problema)
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            edits += [Edit(*src.span(node), str(node.value + d), "constante") for d in (1, -1) if node.value + d >= 0]
        # condição negada
        elif isinstance(node, (ast.If, ast.While, ast.IfExp)):
            start, end = src.span(node.test)
            edits.append(Edit(start, end, f"not ({src.text(start, end)})", "condicao"))
    # variável trocada por outra do mesmo escopo; linha apagada
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = local_names(function)
        for node in ast.walk(function):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in names:
                edits += [Edit(*src.span(node), other, "variavel") for other in names if other != node.id]
            for body in (getattr(node, "body", None), getattr(node, "orelse", None)):
                if not isinstance(body, list) or len(body) < 2:
                    continue
                for stmt in body:
                    if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.Expr)) and not (
                            isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)):
                        start = src.char(stmt.lineno, 0)
                        end = src.char(stmt.end_lineno + 1, 0) if stmt.end_lineno < len(src.lines) else len(src.code)
                        if src.node_char(stmt.lineno, stmt.col_offset) == start + len(
                                src.lines[stmt.lineno - 1]) - len(src.lines[stmt.lineno - 1].lstrip()):
                            edits.append(Edit(start, end, "", "linha_apagada"))
    return edits


def mutants(code: str) -> list[Mutant]:
    """Todos os mutantes distintos e sintaticamente válidos da solução."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    src = Source(code)
    seen, result = {code}, []
    for edit in token_edits(src) + ast_edits(src, tree):
        new = src.apply(edit)
        if new not in seen and syntax_ok(new):
            seen.add(new)
            result.append(Mutant(edit.operator, new))
    return result


# --- execução -------------------------------------------------------------------

RUNNER = r"""
import ast, json, signal, sys
jobs = json.loads(sys.stdin.read())

class _Tempo(BaseException):
    pass

def _alarme(*_):
    raise _Tempo()

signal.signal(signal.SIGALRM, _alarme)

def _comparar(node, globais, locais):
    if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1 \
            and isinstance(node.test.ops[0], ast.Eq):
        obtido = eval(compile(ast.Expression(node.test.left), "<teste>", "eval"), globais, locais)
        esperado = eval(compile(ast.Expression(node.test.comparators[0]), "<teste>", "eval"), globais, locais)
        return {"obtido": repr(obtido)[:300], "esperado": repr(esperado)[:300]}
    return {}

def _detalhe(teste, ns, erro):
    # O assert que falhou: o próprio teste, ou a linha dentro de uma função de teste (estilo pytest).
    try:
        tb, frame = erro.__traceback__, None
        while tb is not None:
            if tb.tb_frame.f_code.co_filename == "<teste>":
                frame = tb.tb_frame
            tb = tb.tb_next
        if frame is None:
            return {}
        if frame.f_code.co_name == "<module>":
            return _comparar(ast.parse(teste).body[0], ns, None)
        linha = teste.splitlines()[frame.f_lineno - 1].strip()
        return {"linha": linha, **_comparar(ast.parse(linha).body[0], frame.f_globals, frame.f_locals)}
    except BaseException:
        return {}

def _avaliar(codigo, testes, preambulo, limite):
    ns = {"__name__": "__solucao__"}
    try:
        signal.setitimer(signal.ITIMER_REAL, limite)
        exec(compile(preambulo + "\n" + codigo, "<solucao>", "exec"), ns)
        for teste in testes:
            try:
                exec(compile(teste, "<teste>", "exec"), ns)
            except _Tempo:
                raise
            except BaseException as e:
                return {"status": "falha", "teste": teste, "erro": f"{type(e).__name__}: {e}"[:300],
                        **_detalhe(teste, ns, e)}
        return {"status": "passa"}
    except _Tempo:
        return {"status": "tempo"}
    except BaseException as e:
        return {"status": "falha", "teste": None, "erro": f"{type(e).__name__}: {e}"[:300]}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

saida = []
for job in jobs:
    resultados = [_avaliar(job["codes"][0], job["tests"], job["preamble"], job["limit"])]
    if resultados[0]["status"] == "passa":  # se a original falha, os mutantes não interessam
        resultados += [_avaliar(c, job["tests"], job["preamble"], job["limit"]) for c in job["codes"][1:]]
    saida.append(resultados)
sys.stdout.flush()
print("\n__mutantes__" + json.dumps(saida))
"""


def run_jobs(jobs: list[dict]) -> list[list[dict] | None]:
    """Roda vários conjuntos ``{codes, tests, preamble}`` numa só execução do sandbox.

    Para cada conjunto, o status de cada código (``passa``, ``falha`` com o teste e o
    erro, ``tempo``); se a original (o primeiro código) falha, só ela. Se a execução cai
    (um mutante que mata o processo, memória), refaz um conjunto por vez.
    """
    payload = json.dumps([{**job, "limit": MUTANT_TIME_LIMIT} for job in jobs])
    limits = f"ulimit -v {MEMORY_LIMIT_KB}; exec /usr/bin/python3 -I -c \"$0\""
    try:
        done = subprocess.run([*SANDBOX, "/usr/bin/bash", "-c", limits, RUNNER], input=payload.encode(),
                              capture_output=True, timeout=RUN_TIME_LIMIT * len(jobs))
        marker = done.stdout.rfind(MARK.encode())
        results = json.loads(done.stdout[marker + len(MARK):].decode("utf-8", "replace")) if marker >= 0 else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        results = None
    if results is not None and len(results) == len(jobs):
        return results
    if len(jobs) == 1:
        return [None]
    return [run_jobs([job])[0] for job in jobs]


def run_mutants(codes: list[str], tests: list[str], preamble: str = "") -> list[dict] | None:
    """Um conjunto só (para testes manuais)."""
    return run_jobs([{"codes": codes, "tests": tests, "preamble": preamble}])[0]


# --- pedidos ----------------------------------------------------------------------

# Paráfrases fixas por idioma. {descricao}, {teste} e {detalhe} são preenchidos.
INSTRUCTIONS = {
    "en": ["Fix the bug in this code.", "This function has a bug. Fix it.", "Find and fix the bug.",
           "The code below doesn't work correctly. Fix it.", "There is a mistake in this code. Correct it.",
           "Debug this function and return the corrected version.", "Fix this code so it works as intended."],
    "pt": ["Corrija o bug neste código.", "Esta função tem um bug. Corrija.", "Encontre e corrija o erro.",
           "O código abaixo não funciona direito. Conserte.", "Há um erro neste código. Corrija-o.",
           "Depure esta função e devolva a versão corrigida.", "Corrija este código para que funcione como deveria."],
    "es": ["Corrige el error en este código.", "Esta función tiene un bug. Corrígelo.", "Encuentra y corrige el error.",
           "El código de abajo no funciona bien. Arréglalo.", "Hay un fallo en este código. Corrígelo.",
           "Depura esta función y devuelve la versión corregida.", "Corrige este código para que funcione como debe."],
    "fr": ["Corrige le bug dans ce code.", "Cette fonction a un bug. Corrige-le.", "Trouve et corrige l'erreur.",
           "Le code ci-dessous ne fonctionne pas correctement. Répare-le.", "Il y a une erreur dans ce code. Corrige-la.",
           "Débogue cette fonction et renvoie la version corrigée.",
           "Corrige ce code pour qu'il fonctionne comme prévu."],
    "de": ["Behebe den Fehler in diesem Code.", "Diese Funktion hat einen Bug. Behebe ihn.",
           "Finde und behebe den Fehler.", "Der folgende Code funktioniert nicht richtig. Repariere ihn.",
           "In diesem Code steckt ein Fehler. Korrigiere ihn.",
           "Debugge diese Funktion und gib die korrigierte Version zurück.",
           "Korrigiere diesen Code, damit er wie vorgesehen funktioniert."],
    "it": ["Correggi il bug in questo codice.", "Questa funzione ha un bug. Correggilo.", "Trova e correggi l'errore.",
           "Il codice qui sotto non funziona bene. Sistemalo.", "C'è un errore in questo codice. Correggilo.",
           "Esegui il debug di questa funzione e restituisci la versione corretta.",
           "Correggi questo codice in modo che funzioni come previsto."],
}
WITH_DESCRIPTION = {
    "en": ["{instrucao}\n\nWhat the code should do:\n{descricao}", "The code should do this:\n{descricao}\n\n{instrucao}",
           "{instrucao} It is supposed to solve this task:\n{descricao}"],
    "pt": ["{instrucao}\n\nO que o código deveria fazer:\n{descricao}", "O código deveria fazer isto:\n{descricao}\n\n{instrucao}",
           "{instrucao} Ele deveria resolver esta tarefa:\n{descricao}"],
    "es": ["{instrucao}\n\nLo que debería hacer el código:\n{descricao}",
           "El código debería hacer esto:\n{descricao}\n\n{instrucao}",
           "{instrucao} Debería resolver esta tarea:\n{descricao}"],
    "fr": ["{instrucao}\n\nCe que le code devrait faire :\n{descricao}",
           "Le code devrait faire ceci :\n{descricao}\n\n{instrucao}",
           "{instrucao} Il est censé résoudre cette tâche :\n{descricao}"],
    "de": ["{instrucao}\n\nWas der Code tun soll:\n{descricao}", "Der Code soll Folgendes tun:\n{descricao}\n\n{instrucao}",
           "{instrucao} Er soll diese Aufgabe lösen:\n{descricao}"],
    "it": ["{instrucao}\n\nCosa dovrebbe fare il codice:\n{descricao}",
           "Il codice dovrebbe fare questo:\n{descricao}\n\n{instrucao}",
           "{instrucao} Dovrebbe risolvere questo compito:\n{descricao}"],
}
WITH_TEST = {
    "en": ["{instrucao}\n\nThis test fails:\n{teste}\n{detalhe}", "This test is failing:\n{teste}\n{detalhe}\n\n{instrucao}"],
    "pt": ["{instrucao}\n\nEste teste falha:\n{teste}\n{detalhe}", "Este teste está falhando:\n{teste}\n{detalhe}\n\n{instrucao}"],
    "es": ["{instrucao}\n\nEsta prueba falla:\n{teste}\n{detalhe}", "Esta prueba está fallando:\n{teste}\n{detalhe}\n\n{instrucao}"],
    "fr": ["{instrucao}\n\nCe test échoue :\n{teste}\n{detalhe}", "Ce test ne passe pas :\n{teste}\n{detalhe}\n\n{instrucao}"],
    "de": ["{instrucao}\n\nDieser Test schlägt fehl:\n{teste}\n{detalhe}",
           "Dieser Test schlägt fehl:\n{teste}\n{detalhe}\n\n{instrucao}"],
    "it": ["{instrucao}\n\nQuesto test fallisce:\n{teste}\n{detalhe}",
           "Questo test non passa:\n{teste}\n{detalhe}\n\n{instrucao}"],
}
EXPECTED_GOT = {"en": "Expected {esperado}, got {obtido}.", "pt": "Esperado {esperado}, obtido {obtido}.",
                "es": "Se esperaba {esperado}, se obtuvo {obtido}.", "fr": "Attendu {esperado}, obtenu {obtido}.",
                "de": "Erwartet {esperado}, erhalten {obtido}.", "it": "Atteso {esperado}, ottenuto {obtido}."}
ERROR = {"en": "Error: {erro}", "pt": "Erro: {erro}", "es": "Error: {erro}", "fr": "Erreur : {erro}",
         "de": "Fehler: {erro}", "it": "Errore: {erro}"}
NLS = tuple(INSTRUCTIONS)


def shown_test(test: str) -> str:
    """O teste como aparece no pedido: sem a linha que chama a função de teste (estilo pytest)."""
    lines = test.strip().splitlines()
    if len(lines) > 1 and re.fullmatch(r"test\w*\(\)", lines[-1].strip()):
        lines = lines[:-1]
    return "\n".join(lines)


def build_prompt(r: Random, variant: str, nl: str, description: str | None, failure: dict) -> str:
    instruction = r.choice(INSTRUCTIONS[nl])
    if variant == "a":
        return instruction
    if variant == "b":
        return r.choice(WITH_DESCRIPTION[nl]).format(instrucao=instruction, descricao=description.strip())
    if "obtido" in failure:
        detail = EXPECTED_GOT[nl].format(esperado=failure["esperado"], obtido=failure["obtido"])
    else:
        detail = ERROR[nl].format(erro=failure["erro"])
    return r.choice(WITH_TEST[nl]).format(instrucao=instruction, teste=shown_test(failure["teste"]), detalhe=detail)


# --- montagem ----------------------------------------------------------------------

@dataclass
class Solution:
    id: str
    code: str
    tests: list[str]
    description: str | None  # enunciado (para a variante b)
    description_nl: str | None
    preamble: str = ""
    licenca: str = ""


def candidates(solution: Solution, seed: int) -> list[Mutant]:
    """Mutantes a testar: um de cada operador por vez (mais variedade), até ``CANDIDATES_PER_SOLUTION``."""
    r = Random(f"{seed}-{solution.id}")
    found = mutants(solution.code)
    r.shuffle(found)
    by_operator: dict[str, list[Mutant]] = {}
    for m in found:
        by_operator.setdefault(m.operator, []).append(m)
    ordered = []
    while any(by_operator.values()):
        for ms in by_operator.values():
            if ms:
                ordered.append(ms.pop())
    return ordered[:CANDIDATES_PER_SOLUTION]


def examples_from(solution: Solution, ordered: list[Mutant], results: list[dict] | None, fonte: str,
                  seed: int) -> tuple[list[dict], Counter]:
    """Até ``BUGS_PER_SOLUTION`` exemplos de T6 da solução, e contagens do que aconteceu."""
    r = Random(f"{seed}-{solution.id}-exemplos")
    stats: Counter = Counter()
    if not ordered:
        stats["sem_mutantes"] += 1
        return [], stats
    if results is None:
        stats["execucao_caiu"] += 1
        return [], stats
    if results[0]["status"] != "passa":
        stats["original_falha"] += 1
        return [], stats
    examples, used = [], set()
    operators = {m.operator for m in ordered}
    for mutant, result in zip(ordered, results[1:]):
        stats[f"{mutant.operator}:{result['status']}"] += 1
        if result["status"] != "falha" or len(examples) >= BUGS_PER_SOLUTION:
            continue
        if mutant.operator in used and len(used) < len(operators):
            continue  # ainda há operadores não usados entre os testados
        used.add(mutant.operator)
        variants = ["a"] + (["b"] if solution.description else []) + (["c"] if result.get("teste") else [])
        variant = r.choice(variants)
        nl = solution.description_nl if variant == "b" and solution.description_nl in NLS else r.choice(NLS)
        prompt = build_prompt(r, variant, nl, solution.description, result)
        digest = hashlib.sha1(f"{solution.id}|{mutant.code}".encode()).hexdigest()[:16]
        examples.append({
            "id": f"T6-python-{digest}", "task": "T6", "level": "facil", "lang": "python", "nl": nl,
            "source": fonte, "repo": None, "path": None, "licenses": None,
            "prompt": prompt, "code": mutant.code, "response": solution.code, "kind": "code",
            "lang_in_prompt": True, "n_tokens": example_tokens(prompt, mutant.code, solution.code), "fonte": f"t6-{fonte}", "licenca": solution.licenca,
            "tests": solution.tests, "test_setup": solution.preamble or None, "mutacao": mutant.operator,
            "variante": variant, "origem": solution.id,
        })
    stats["solucoes_com_exemplo"] += bool(examples)
    stats["exemplos"] += len(examples)
    return examples, stats


def example_tokens(prompt: str, code: str, response: str) -> int:
    from dataprep.sft_convert import estimate_tokens
    return estimate_tokens(prompt, None) + estimate_tokens(code, "python") + estimate_tokens(response, "python")


def process_batch(task: tuple[list[Solution], str, int]) -> tuple[list[dict], Counter]:
    """Um lote de soluções: gera os mutantes, roda tudo numa execução do sandbox e monta os exemplos."""
    solutions, fonte, seed = task
    ordered = [candidates(s, seed) for s in solutions]
    pending = [i for i, o in enumerate(ordered) if o]
    results: list[list[dict] | None] = [None] * len(solutions)
    if pending:
        jobs = [{"codes": [solutions[i].code] + [m.code for m in ordered[i]], "tests": solutions[i].tests,
                 "preamble": solutions[i].preamble} for i in pending]
        for i, result in zip(pending, run_jobs(jobs)):
            results[i] = result
    rows, stats = [], Counter()
    for solution, o, result in zip(solutions, ordered, results):
        new, s = examples_from(solution, o, result, fonte, seed)
        rows += new
        stats.update(s)
    return rows, stats


def fonte_solutions(path: Path) -> list[Solution]:
    """Soluções Python com testes de uma parte convertida (só o split ``train``)."""
    columns = ["id", "lang", "nl", "prompt", "response", "tests", "test_setup", "licenca", "split"]
    rows = pq.read_table(path, columns=columns).to_pylist()
    return [Solution(id=r["id"], code=r["response"], tests=list(r["tests"]), description=r["prompt"],
                     description_nl=r["nl"], preamble=r["test_setup"] or "", licenca=r["licenca"] or "")
            for r in rows if r["lang"] == "python" and r["response"] and r["tests"] and r["split"] == "train"
            and syntax_ok(r["response"])]


T6_SCHEMA = pa.schema([
    ("id", pa.string()), ("task", pa.string()), ("level", pa.string()), ("lang", pa.string()), ("nl", pa.string()),
    ("source", pa.string()), ("repo", pa.string()), ("path", pa.string()), ("licenses", pa.string()),
    ("prompt", pa.string()), ("code", pa.string()), ("response", pa.string()), ("kind", pa.string()),
    ("lang_in_prompt", pa.bool_()), ("n_tokens", pa.int64()), ("fonte", pa.string()), ("licenca", pa.string()),
    ("tests", pa.list_(pa.string())), ("test_setup", pa.string()), ("mutacao", pa.string()),
    ("variante", pa.string()), ("origem", pa.string()),
])


def build(fonte: str, limit: int | None, seed: int = 0, workers: int = 14, max_parts: int | None = None) -> None:
    parts = sorted((FONTES_DIR / fonte).glob("part-*.parquet"))[:max_parts]
    if not parts:
        raise SystemExit(f"{fonte}: nenhuma parte convertida em {FONTES_DIR / fonte}")
    out_dir = FONTES_DIR / (f"t6-{fonte}-teste" if limit else f"t6-{fonte}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "execucao.json"
    stats = Counter(json.loads(stats_path.read_text())) if stats_path.exists() else Counter()
    start, done_solutions = time.time(), 0
    with ProcessPoolExecutor(workers) as pool:
        for part in parts:
            out = out_dir / part.name
            if out.exists():
                continue
            solutions = fonte_solutions(part)[:limit]
            batches = [(solutions[i:i + SOLUTIONS_PER_RUN], fonte, seed)
                       for i in range(0, len(solutions), SOLUTIONS_PER_RUN)]
            rows = []
            for new, s in pool.map(process_batch, batches):
                rows += new
                stats.update(s)
            tmp = out.with_suffix(".tmp")
            pq.write_table(pa.Table.from_pylist(rows, schema=T6_SCHEMA), tmp, compression="zstd")
            tmp.rename(out)
            stats_path.write_text(json.dumps(stats))
            done_solutions += len(solutions)
            elapsed = time.time() - start
            print(f"  {part.name}: {len(solutions):,} soluções, {len(rows):,} exemplos | "
                  f"{done_solutions / max(elapsed, 1):.0f} soluções/s", flush=True)
            if limit:
                break
    save_meta(fonte, out_dir, stats)


def save_meta(fonte: str, out_dir: Path, stats: Counter) -> None:
    table = pq.read_table(sorted(out_dir.glob("part-*.parquet")), columns=["mutacao", "variante", "nl"]) \
        if any(out_dir.glob("part-*.parquet")) else None
    rows = table.to_pylist() if table is not None else []
    meta = {"fonte": f"t6-{fonte}", "exemplos": len(rows),
            "por_mutacao": dict(Counter(r["mutacao"] for r in rows).most_common()),
            "por_variante": dict(Counter(r["variante"] for r in rows)),
            "por_idioma": dict(Counter(r["nl"] for r in rows)),
            "execucao": {k: v for k, v in stats.items() if ":" not in k},
            "mutantes": {k: v for k, v in sorted(stats.items()) if ":" in k},
            "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps({k: meta[k] for k in ("fonte", "exemplos", "por_mutacao", "por_variante", "execucao")},
                     indent=1, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fontes", nargs="+", help="fontes convertidas com testes (ex.: kodcode exercicios-v1)")
    parser.add_argument("--limit", type=int, help="só as primeiras soluções da primeira parte (teste)")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--max-partes", type=int, help="só as N primeiras partes de cada fonte")
    parser.add_argument("--preencher-tokens", action="store_true", help="só preenche n_tokens das partes já geradas")
    args = parser.parse_args()
    for fonte in args.fontes:
        if args.preencher_tokens:
            fill_tokens(fonte)
        else:
            build(fonte, args.limit, workers=args.workers, max_parts=args.max_partes)


def fill_tokens(fonte: str) -> None:
    """Partes geradas antes de o T6 estimar tokens ficaram com n_tokens nulo."""
    for part in sorted((FONTES_DIR / f"t6-{fonte}").glob("part-*.parquet")):
        table = pq.read_table(part)
        if table.column("n_tokens").null_count == 0:
            continue
        rows = table.to_pylist()
        for row in rows:
            row["n_tokens"] = example_tokens(row["prompt"], row["code"], row["response"])
        tmp = part.with_suffix(".tmp")
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), tmp, compression="zstd")
        tmp.rename(part)
    print(f"t6-{fonte}: n_tokens preenchido")


if __name__ == "__main__":
    main()
