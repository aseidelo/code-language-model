"""Conversão das fontes do SFT v2 para o schema do banco (ver docs/sft-v2.md).

Lê ``data/codelm_sft/v2/raw/<fonte>/`` (``dataprep.sft_sources``) e as funções do
corpus v2 (``v2/units/``, de ``dataprep.sft_candidates corpus --v2``) e grava
``data/codelm_sft/v2/fontes/<fonte>/part-*.parquet`` e um ``meta.json`` com as
contagens por tarefa × linguagem × idioma × nível. Nada é descartado por qualidade:
linhas cuja resposta não está numa das nossas linguagens ficam com ``lang`` =
``outra:<rótulo>``, e a mistura decide o que usa.

Colunas além das do banco:

* ``fonte``, ``licenca``;
* ``response_raw``: a resposta original (com a prosa em volta do código);
* ``tests`` (lista de trechos executáveis) e ``test_setup`` (código que roda antes),
  nas fontes com testes, no formato do executor do T6 (``dataprep.sft_bugfix``);
* ``split``: ``train``, ou ``rl``/``valid`` nos exercícios v1.

``n_tokens`` é estimado pelos caracteres por token do corpus v2; o pack calcula o exato.

    uv run python -m dataprep.sft_convert [fonte ...]
    uv run python -m dataprep.sft_convert --list
"""

import argparse
import ast
import hashlib
import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from codelm.languages import ASSETS_DIR
from dataprep.sft_candidates import extract_functions, level, load_chars_per_token_v2
from dataprep.sft_docstrings import BAD_DOC, MIN_TOKENS, TEST_FUNCTION, build_example, doc_prose, prose_words, \
    render_doc
from dataprep.sft_mix import SFT_DIR
from dataprep.sft_sources import BY_NAME, RAW_DIR

V2_DIR = SFT_DIR / "v2"
FONTES_DIR = V2_DIR / "fontes"
UNITS_DIR = V2_DIR / "units"
INSTRUCTIONS = ASSETS_DIR / "sft" / "instructions.json"
OUR_LANGS = ("python", "javascript", "typescript", "java", "c", "rust", "html", "css", "json", "yaml", "bash")
NLS = ("en", "pt", "es", "fr", "de", "it")
SEED = 0

# --- linguagem do código --------------------------------------------------------

ALIASES = {
    "python": "python", "py": "python", "python3": "python", "python2": "python",
    "javascript": "javascript", "js": "javascript", "jsx": "javascript", "node": "javascript", "nodejs": "javascript",
    "typescript": "typescript", "ts": "typescript", "tsx": "typescript",
    "java": "java", "c": "c", "h": "c", "ansi c": "c",
    "rust": "rust", "rs": "rust",
    "html": "html", "htm": "html", "xhtml": "html",
    "css": "css", "json": "json", "yaml": "yaml", "yml": "yaml",
    "bash": "bash", "sh": "bash", "shell": "bash", "zsh": "bash", "unix shell": "bash", "shellscript": "bash",
    "console": "bash",
}
FENCE = re.compile(r"```[ \t]*([\w+#.-]*)[^\n]*\n(.*?)```", re.DOTALL)


def canonical_lang(tag: str | None) -> str | None:
    if not tag:
        return None
    tag = tag.strip().lower()
    return ALIASES.get(tag, f"outra:{tag}")


def guess_lang(code: str) -> str | None:
    stripped = code.strip()
    if not stripped:
        return None
    if stripped.startswith(("<!DOCTYPE", "<html", "<div", "<")):
        return "html"
    if stripped[0] in "[{":
        try:
            json.loads(stripped)
            return "json"
        except ValueError:
            pass
    if stripped.startswith("#!") and ("bash" in stripped[:30] or "/sh" in stripped[:30]):
        return "bash"
    try:
        compile(stripped, "<código>", "exec", dont_inherit=True)
        return "python"
    except (SyntaxError, ValueError):
        return None


def main_code(response: str, declared: str | None = None) -> tuple[str | None, str]:
    """(linguagem, código) do maior bloco de código da resposta; sem cerca, a resposta inteira."""
    blocks = FENCE.findall(response or "")
    if blocks:
        tag, code = max(blocks, key=lambda b: len(b[1]))
        lang = canonical_lang(tag) or canonical_lang(declared) or guess_lang(code)
        return lang, code.rstrip("\n")
    lang = canonical_lang(declared) or guess_lang(response or "")
    return lang, (response or "").strip("\n")


LANG_NAMES = {
    "python": r"\bpython\b", "javascript": r"\b(java ?script|node\.?js|js)\b", "typescript": r"\b(type ?script|ts)\b",
    "java": r"\bjava\b(?! ?script)", "c": r"(\bC\b(?![+#]))|\bANSI C\b", "rust": r"\brust\b", "html": r"\bhtml\b",
    "css": r"\bcss\b", "json": r"\bjson\b", "yaml": r"\b(yaml|yml)\b", "bash": r"\b(bash|shell|zsh|sh)\b",
}


def lang_in_prompt(prompt: str, lang: str) -> bool:
    pattern = LANG_NAMES.get(lang)
    if pattern is None:
        return False
    return re.search(pattern, prompt, 0 if lang == "c" else re.I) is not None


# --- idioma ----------------------------------------------------------------------

STOPWORDS = {
    "en": "the a an of to and is in that for with this it be are as on by from or not which",
    "pt": "o os as um uma de do da dos das que não para com é em no na se por ao mais como",
    "es": "el la los las un una de del que no para con es en se por al más como lo",
    "fr": "le la les un une des de du que ne pas pour avec est en dans sur au aux qui",
    "de": "der die das ein eine und ist nicht mit zu den dem des von für auf im sich",
    "it": "il lo la gli le un una di del della che non per con è in nel sul al più come",
}
STOPWORDS = {nl: set(words.split()) for nl, words in STOPWORDS.items()}
INLINE_CODE = re.compile(r"`[^`\n]*`")


SCRIPTS = (("zh", r"[\u4e00-\u9fff]"), ("ja", r"[\u3040-\u30ff]"), ("ko", r"[\uac00-\ud7af]"),
           ("ru", r"[\u0400-\u04ff]"))


def script_nl(text: str) -> str | None:
    """Idioma pela escrita, quando ela não é latina (o japonês também usa ideogramas chineses)."""
    letters = len(re.findall(r"[^\W\d_]", text)) or 1
    found = {nl: len(re.findall(pattern, text)) for nl, pattern in SCRIPTS}
    if found["ja"] >= 0.1 * letters:
        return "ja"
    nl = max(found, key=found.get)
    return nl if found[nl] >= 0.2 * letters else None


def stopword_nl(text: str) -> str | None:
    """Idioma pelas palavras funcionais, quando a contagem é clara."""
    letters = re.findall(r"[^\W\d_]", text)
    if not letters or sum(ch < "\u0250" for ch in letters) < 0.8 * len(letters):
        return None  # escrita não latina (chinês, russo...): decide o script_nl
    # Linhas com cara de código (import, chaves, ponto e vírgula) não contam: "import os as" não é português.
    prose = "\n".join(line for line in text.splitlines()
                      if not re.search(r"[{};=<>]|^\s*(import|from|package|def|class|public|#include)\b", line))
    words = re.findall(r"[^\W\d_]+", prose.lower())
    counts = {nl: sum(w in stop for w in words) for nl, stop in STOPWORDS.items()}
    best = max(counts, key=counts.get)
    second = max(v for nl, v in counts.items() if nl != best)
    return best if counts[best] >= 3 and counts[best] >= 2 * second else None


def detect_nls(texts: list[str]) -> list[str]:
    """Idioma de cada texto: um dos 6, ou ``outra:<código>``. Só a prosa (sem blocos nem trechos de código).

    Escrita não latina decide pelo alfabeto (chinês, japonês, coreano, russo). Em texto
    latino, as palavras funcionais; sem contagem clara (texto curto, código colado sem
    cerca, só termos técnicos), inglês. Detectores estatísticos erram muito nesses casos.
    """
    result = []
    for text in texts:
        prose = INLINE_CODE.sub(" ", FENCE.sub(" ", text or ""))[:2000]
        code = script_nl(prose) or stopword_nl(prose) or "en"
        result.append(code if code in NLS else f"outra:{code}")
    return result


# --- linhas ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def ratios() -> dict[str, float]:
    return load_chars_per_token_v2()


def estimate_tokens(text: str, lang: str | None) -> int:
    r = ratios()
    return round(len(text or "") / r.get(lang or "", r["natural"]))


def row(fonte: str, licenca: str, uid: str, task: str, lang: str | None, nl: str, prompt: str, response: str | None,
        response_raw: str | None = None, code: str | None = None, repo: str | None = None, path: str | None = None,
        licenses: str | None = None, tests: list[str] | None = None, test_setup: str | None = None,
        split: str = "train", source: str | None = None) -> dict:
    lang = lang or "outra:?"
    if licenses is not None and not isinstance(licenses, str):
        licenses = json.dumps(list(licenses))
    code_tokens = estimate_tokens(response if task in ("T1", "T2", "T5", "T6") else code, lang)
    digest = hashlib.sha1(f"{fonte}|{uid}|{task}".encode()).hexdigest()[:16]
    return {
        "id": f"{task}-{fonte}-{digest}", "task": task, "level": level(code_tokens), "lang": lang, "nl": nl,
        "source": source or fonte, "repo": repo, "path": path, "licenses": licenses, "prompt": prompt, "code": code,
        "response": response, "kind": "code", "lang_in_prompt": bool(lang_in_prompt(prompt, lang)),
        "n_tokens": estimate_tokens(prompt, None) + estimate_tokens(code, lang) + code_tokens,
        "fonte": fonte, "licenca": licenca, "response_raw": response_raw, "tests": tests, "test_setup": test_setup,
        "split": split,
    }


SCHEMA = pa.schema([
    ("id", pa.string()), ("task", pa.string()), ("level", pa.string()), ("lang", pa.string()), ("nl", pa.string()),
    ("source", pa.string()), ("repo", pa.string()), ("path", pa.string()), ("licenses", pa.string()),
    ("prompt", pa.string()), ("code", pa.string()), ("response", pa.string()), ("kind", pa.string()),
    ("lang_in_prompt", pa.bool_()), ("n_tokens", pa.int64()), ("fonte", pa.string()), ("licenca", pa.string()),
    ("response_raw", pa.string()), ("tests", pa.list_(pa.string())), ("test_setup", pa.string()),
    ("split", pa.string()),
])


def literal_list(value) -> list:
    """Lista guardada como JSON ou como repr do Python."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []


# --- testes no estilo pytest (KodCode) --------------------------------------------

PYTEST_SHIM = '''
class _Raises:
    def __init__(self, expected, match=None):
        self.expected = expected
    def __enter__(self):
        return self
    def __exit__(self, kind, value, tb):
        if kind is None:
            raise AssertionError("DID NOT RAISE")
        return issubclass(kind, self.expected)
class _Approx:
    def __init__(self, value, rel=1e-6, abs=1e-12):
        self.value, self.rel, self.abs = value, rel, abs
    def __eq__(self, other):
        if isinstance(self.value, (list, tuple)):
            return len(other) == len(self.value) and all(_Approx(a, self.rel, self.abs) == b for a, b in zip(self.value, other))
        return abs(other - self.value) <= max(self.rel * abs(self.value), self.abs)
    def __repr__(self):
        return f"approx({self.value!r})"
class _Mark:
    def __getattr__(self, name):
        return lambda *a, **k: (lambda f: f)
class pytest:
    raises = _Raises
    approx = _Approx
    mark = _Mark()
    @staticmethod
    def fixture(*a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)
'''


def pytest_tests(test_file: str) -> tuple[str, list[str]] | None:
    """(setup, testes) de um arquivo pytest: cada teste vira ``def ...`` + a chamada.

    O ``from solution import`` sai (a solução roda antes), o ``import pytest`` vira um
    substituto mínimo. Testes com parâmetros (fixtures, parametrize) ficam de fora.
    """
    try:
        tree = ast.parse(test_file)
    except SyntaxError:
        return None
    lines = test_file.splitlines(keepends=True)
    setup, tests = [], []
    for node in tree.body:
        text = "".join(lines[node.lineno - 1 - len(getattr(node, "decorator_list", [])): node.end_lineno])
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", None) or ""
            names = [a.name for a in node.names]
            if module.split(".")[0] == "solution" or "solution" in names or names == ["pytest"]:
                continue
            setup.append(text)
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            if node.args.args or node.decorator_list:
                continue
            tests.append(f"{ast.get_source_segment(test_file, node)}\n{node.name}()")
        else:
            setup.append(text)
    if not tests:
        return None
    return PYTEST_SHIM + "".join(setup), tests


# --- fontes -------------------------------------------------------------------------

def t1_generic(prompt_col, response_col, lang_col=None, fixed_lang=None, nbsp=False):
    """T1 a partir de pedido e resposta (o código é o maior bloco da resposta).

    ``nbsp``: troca espaços não quebráveis por espaços (o Rosetta Code vem do HTML com eles
    nas linhas em branco e na indentação, o que quebra o parser).
    """
    def convert(fonte, licenca, rows):
        out = []
        for r in rows:
            prompt, raw = r.get(prompt_col) or "", r.get(response_col) or ""
            if nbsp:
                raw = re.sub(r"(?m)^[ \t]+$", "", raw.replace("\xa0", " "))
            declared = fixed_lang or (r.get(lang_col) if lang_col else None)
            lang, code = main_code(raw, declared)
            if not prompt.strip() or not code.strip():
                continue
            uid = r.get("id") or r.get("seq_id") or hashlib.sha1((prompt + raw).encode()).hexdigest()
            out.append(row(fonte, licenca, str(uid), "T1", lang, "?", prompt, code, response_raw=raw))
        return out
    return convert


def kodcode(fonte, licenca, rows):
    out = []
    for r in rows:
        converted = pytest_tests(r["test"] or "")
        setup, tests = converted if converted else (None, None)
        out.append(row(fonte, licenca, r["question_id"], "T1", "python", "?", r["question"], r["solution"],
                       response_raw=r["solution"], tests=tests, test_setup=setup, source=f"kodcode:{r['subset']}"))
    return out


def open_code_instruct(fonte, licenca, rows):
    out = []
    for r in rows:
        lang, code = main_code(r["output"], "python" if r["domain"] == "generic" else None)
        status = literal_list(r["tests_execution_status"])
        tests = [t.strip() for t, s in zip(literal_list(r["unit_tests"]), status) if s == "pass" and t.strip()]
        out.append(row(fonte, licenca, r["id"], "T1", lang, "?", r["input"], code, response_raw=r["output"],
                       tests=tests or None, source=f"open-code-instruct:{r['domain']}"))
    return out


def opc_educational(fonte, licenca, rows):
    out = []
    for r in rows:
        tests = [t for t in literal_list(r["testcase"]) if t.strip()]
        out.append(row(fonte, licenca, r["seq_id"], "T1", "python", "?", r["instruction"], r["code"],
                       response_raw=r["output"], tests=tests or None))
    return out


def acecode(fonte, licenca, rows):
    """AceCode: pedidos com testes. A solução é uma inferência com todos os testes passando (no 87K)."""
    out = []
    for r in rows:
        tests = [t for t in literal_list(r.get("test_cases") or r.get("tests")) if t.strip()]
        solution = None
        for inference in literal_list(r.get("inferences")) if r.get("inferences") is not None else []:
            if float(inference.get("pass_rate") or 0) >= 1.0:
                solution = inference["completion"].strip("\n")
                break
        out.append(row(fonte, licenca, r["id"], "T1", "python", "?", r["question"], solution, tests=tests or None,
                       source=f"{fonte}:{r.get('source')}"))
    return out


def bash_pairs(prompt_col, response_col):
    def convert(fonte, licenca, rows):
        return [row(fonte, licenca, f"{i}", "T1", "bash", "en", r[prompt_col], r[response_col].strip())
                for i, r in enumerate(rows) if (r.get(prompt_col) or "").strip() and (r.get(response_col) or "").strip()]
    return convert


def bash_instruct(fonte, licenca, rows):
    out = []
    for i, r in enumerate(rows):
        messages = literal_list(r["messages"])
        user = next((m["content"] for m in messages if m["role"] == "user"), None)
        answer = next((m["content"] for m in messages if m["role"] == "assistant"), None)
        if user and answer:
            out.append(row(fonte, licenca, f"{i}", "T1", "bash", "en", user, answer.strip(), source=f"{fonte}:{r.get('category')}"))
    return out


TLDR_DIRS = {"pages": "en", "pages.pt_BR": "pt", "pages.pt_PT": "pt", "pages.es": "es", "pages.fr": "fr",
             "pages.de": "de", "pages.it": "it"}
TLDR_SKIP = {"windows", "cisco-ios", "dos"}
TLDR_EXAMPLE = re.compile(r"^- (.+?)\n\n`(.+?)`$", re.M)


def tldr_command(command: str) -> str:
    command = re.sub(r"\{\{\[([^|\]]+)\|[^\]]*\]\}\}", r"\1", command)  # {{[-f|--force]}} -> -f
    return re.sub(r"\{\{(.*?)\}\}", r"\1", command)


def tldr_pages(fonte, licenca, paths):
    out = []
    for path in paths:
        path = Path(path)
        nl = TLDR_DIRS[path.parent.parent.name]
        title = re.search(r"^# (.+)$", path.read_text(), re.M)
        text = path.read_text()
        for description, command in TLDR_EXAMPLE.findall(text):
            name = title.group(1).strip() if title else path.stem
            prompt = f"{name}: {description.rstrip(':')}"
            out.append(row(fonte, licenca, f"{path.parent.parent.name}/{path.parent.name}/{path.stem}/{description}",
                           "T1", "bash", nl, prompt, tldr_command(command), path=str(path.relative_to(path.parents[2])),
                           source=f"tldr:{path.parent.name}"))
    return out


def code_search_net(fonte, licenca, rows, instructions, rng):
    """T2 e T5 das funções do CodeSearchNet: a doc volta para o código e passa pelo mesmo extrator do corpus."""
    out = []
    for r in rows:
        lang = r["language"]
        code = r["func_code_string"]
        if lang != "python":
            code = f"{render_doc(r['func_documentation_string'], lang, '/**')}\n{code}"
        out += units_to_rows(fonte, licenca, lang, extract_functions(code, lang)[:1],
                             r["repository_name"], r["func_path_in_repository"], None, instructions, rng)
    return out


def units_to_rows(fonte, licenca, lang, units, repo, path, licenses, instructions, rng, nls=None):
    out = []
    for i, unit in enumerate(units):
        if not unit["doc"] or not unit["nodoc"] or TEST_FUNCTION.search(unit["signature"] + "\n" + unit["doc_raw"]):
            continue
        if BAD_DOC.search(unit["doc"]) or prose_words(unit["doc"]) < 4:
            continue
        nl = nls[i] if nls else "?"
        for task in ("T2", "T5"):
            example = build_example({**unit, "lang": lang, "task": task, "nl": nl if nl in NLS else "en", "id": "",
                                     "level": "", "repo": repo, "path": path, "licenses": licenses},
                                    None, instructions, rng)
            if example is None:
                continue
            new = row(fonte, licenca, f"{repo}/{path}/{unit.get('digest') or i}", task, lang, nl,
                      example["prompt"], example["response"], code=example["code"], repo=repo, path=path,
                      licenses=licenses)
            new["lang_in_prompt"] = example["lang_in_prompt"]
            out.append(new)
    return out


# Fontes do raw com conversor de linhas: nome -> função(fonte, licença, linhas).
ROW_CONVERTERS = {
    "opc-stage1-infinity": t1_generic("instruction", "output"),
    "opc-stage1-largescale": t1_generic("instruction", "output"),
    "opc-stage1-realuser": t1_generic("instruction", "output"),
    "opc-stage2-package": t1_generic("instruction", "output"),
    "opc-stage2-educational": opc_educational,
    "magicoder-oss": t1_generic("problem", "solution", lang_col="lang"),
    "magicoder-evol": t1_generic("instruction", "response"),
    "self-oss-instruct": t1_generic("instruction", "response", fixed_lang="python"),
    "code-alpaca": t1_generic("instruction", "output"),
    "codefeedback": t1_generic("query", "answer", lang_col="lang"),
    "glaive-v3": t1_generic("question", "answer"),
    "mceval-instruct": t1_generic("instruction", "output", lang_col="language"),
    "rosetta-code": t1_generic("task_description", "code", lang_col="language_name", nbsp=True),
    "websight": t1_generic("llm_generated_idea", "text", fixed_lang="html"),
    "kodcode": kodcode,
    "open-code-instruct": open_code_instruct,
    "acecode-87k": acecode,
    "acecode-v2": acecode,
    "bash-instruct-iii": bash_instruct,
    "bash-command-6k": bash_pairs("prompt", "completion"),
    "bash-commands": bash_pairs("prompt", "response"),
    "nl2bash": bash_pairs("nl", "bash"),
    "nl2bash-custom": bash_pairs("nl_command", "bash_code"),
}


# --- execução ------------------------------------------------------------------------

def fill_nls(rows: list[dict]) -> None:
    """Idioma dos pedidos ainda sem idioma (``?``)."""
    pending = [r for r in rows if r["nl"] == "?"]
    for r, nl in zip(pending, detect_nls([r["prompt"] for r in pending])):
        r["nl"] = nl


def write_part(rows: list[dict], out: Path) -> int:
    tmp = out.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), tmp, compression="zstd")
    tmp.rename(out)
    return len(rows)


def convert_raw_file(task: tuple[str, str, str, str]) -> tuple[str, Counter]:
    name, licenca, path, out = task
    if Path(out).exists():
        return out, Counter()
    rows = pq.read_table(path).to_pylist()
    converted = ROW_CONVERTERS[name](name, licenca, rows)
    fill_nls(converted)
    write_part(converted, Path(out))
    return out, Counter((r["task"], r["lang"], r["nl"], r["level"]) for r in converted)


def convert_units_file(task: tuple[str, str, int, int, str]) -> tuple[str, Counter]:
    lang, path, start, stop, out = task
    if Path(out).exists():
        return out, Counter()
    instructions = json.loads(INSTRUCTIONS.read_text())
    rng = random.Random(f"{SEED}-{lang}-{start}")
    table = pq.ParquetFile(path)
    units = table.read_row_groups(range(start, stop), columns=[
        "repo", "path", "licenses", "digest", "code", "doc", "doc_raw", "signature", "nodoc", "doc_indent",
        "tokens"]).to_pylist()
    units = [u for u in units if u["doc"] and u["nodoc"] and u["tokens"] >= MIN_TOKENS]
    nls = detect_nls([doc_prose(u["doc"]) for u in units])
    rows = []
    for unit, nl in zip(units, nls):
        rows += units_to_rows("corpus-v2", "por repositório", lang, [unit], unit["repo"], unit["path"],
                              unit["licenses"], instructions, rng, nls=[nl])
    write_part(rows, Path(out))
    return out, Counter((r["task"], r["lang"], r["nl"], r["level"]) for r in rows)


def convert_csn_file(task: tuple[str, str]) -> tuple[str, Counter]:
    path, out = task
    if Path(out).exists():
        return out, Counter()
    instructions = json.loads(INSTRUCTIONS.read_text())
    rng = random.Random(f"{SEED}-{path}")
    rows = code_search_net("code-search-net", BY_NAME["code-search-net"]["licenca"], pq.read_table(path).to_pylist(),
                           instructions, rng)
    for r in rows:
        r["nl"] = "en"
    write_part(rows, Path(out))
    return out, Counter((r["task"], r["lang"], r["nl"], r["level"]) for r in rows)


def convert_tldr(out_dir: Path) -> Counter:
    repo = RAW_DIR / "tldr-pages" / "repo"
    paths = [str(p) for d in TLDR_DIRS for p in sorted((repo / d).glob("*/*.md")) if p.parent.name not in TLDR_SKIP]
    rows = tldr_pages("tldr-pages", BY_NAME["tldr-pages"]["licenca"], paths)
    write_part(rows, out_dir / "part-00000.parquet")
    return Counter((r["task"], r["lang"], r["nl"], r["level"]) for r in rows)


def convert_v1_exercises(out_dir: Path) -> Counter:
    """Exercícios v1 (nossos templates + OpenCoder educational) com os splits train/rl/valid."""
    rows = []
    licenses = {"opencoder": "MIT", "gerador": "nosso"}
    for split in ("train", "rl", "valid"):
        for r in pq.read_table(SFT_DIR / "exercises" / f"{split}.parquet").to_pylist():
            new = row("exercicios-v1", licenses.get(r["source"], ""), r["id"], "T1", "python", r["nl"], r["prompt"],
                      r["response"], tests=list(r["tests"]), split=split, source=r["source"])
            new["lang_in_prompt"] = bool(r["lang_in_prompt"])
            rows.append(new)
    write_part(rows, out_dir / "part-00000.parquet")
    return Counter((r["task"], r["lang"], r["nl"], r["level"]) for r in rows)


def convert_v1_bank_corpus(out_dir: Path) -> Counter:
    """T2/T5 do banco v1 sem tradução (docstrings originais; o resto do banco v1 foi gerado pelo Claude)."""
    table = pq.read_table(SFT_DIR / "bank" / "t2_t5.parquet").to_pylist()
    rows = []
    for r in table:
        if r["source"] != "corpus":
            continue
        new = row("banco-v1-corpus", "por repositório", r["id"], r["task"], r["lang"], r["nl"], r["prompt"],
                  r["response"], code=r["code"], repo=r["repo"], path=r["path"], licenses=r["licenses"])
        new["lang_in_prompt"] = bool(r["lang_in_prompt"])
        rows.append(new)
    write_part(rows, out_dir / "part-00000.parquet")
    return Counter((r["task"], r["lang"], r["nl"], r["level"]) for r in rows)


def units_tasks(out_dir: Path) -> list[tuple]:
    tasks = []
    for path in sorted(UNITS_DIR.glob("*.parquet")):
        groups = pq.ParquetFile(path).num_row_groups
        step = max(1, min(8, groups))
        for start in range(0, groups, step):
            tasks.append((path.stem, str(path), start, min(groups, start + step),
                          str(out_dir / f"part-{path.stem}-{start:05d}.parquet")))
    return tasks


SPECIAL = ["corpus-v2", "code-search-net", "tldr-pages", "exercicios-v1", "banco-v1-corpus"]
ALL = [*ROW_CONVERTERS, *SPECIAL]


def save_meta(name: str, out_dir: Path, counts: Counter, start: float) -> None:
    """Recalcula as contagens a partir das partes (as já prontas de uma execução anterior entram também)."""
    counts = Counter()
    tokens = Counter()
    with_tests = 0
    for part in sorted(out_dir.glob("part-*.parquet")):
        t = pq.read_table(part, columns=["task", "lang", "nl", "level", "n_tokens", "tests", "response"]).to_pylist()
        for r in t:
            key = f"{r['task']}|{r['lang']}|{r['nl']}|{r['level']}"
            counts[key] += 1
            tokens[key] += r["n_tokens"] or 0
            with_tests += bool(r["tests"]) and r["response"] is not None
    raw_meta = RAW_DIR / name / "meta.json"
    meta = {"fonte": name, "origem": json.loads(raw_meta.read_text()) if raw_meta.exists() else None,
            "exemplos": sum(counts.values()), "tokens_estimados": sum(tokens.values()),
            "com_testes_e_solucao": with_tests, "convertido_em": time.strftime("%Y-%m-%d %H:%M:%S"),
            "segundos": round(time.time() - start),
            "celulas": {k: {"exemplos": counts[k], "tokens": tokens[k]} for k in sorted(counts)}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"{name}: {meta['exemplos']:,} exemplos, ~{meta['tokens_estimados'] / 1e6:,.0f}M tokens, "
          f"{with_tests:,} com testes, {meta['segundos']}s", flush=True)


def convert(name: str, pool: ProcessPoolExecutor) -> None:
    out_dir = FONTES_DIR / name
    if (out_dir / "meta.json").exists():
        print(f"{name}: já convertido")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    counts = Counter()
    if name in ROW_CONVERTERS:
        raw_dir = RAW_DIR / name
        if not (raw_dir / "meta.json").exists():
            print(f"{name}: download incompleto, pulando")
            return
        licenca = BY_NAME[name]["licenca"]
        tasks = [(name, licenca, str(p), str(out_dir / f"part-{p.stem}.parquet")) for p in sorted(raw_dir.glob("*.parquet"))]
        for _, c in pool.map(convert_raw_file, tasks):
            counts.update(c)
    elif name == "corpus-v2":
        for _, c in pool.map(convert_units_file, units_tasks(out_dir)):
            counts.update(c)
    elif name == "code-search-net":
        tasks = [(str(p), str(out_dir / f"part-{p.stem}.parquet"))
                 for p in sorted((RAW_DIR / name).glob("*.parquet"))]
        for _, c in pool.map(convert_csn_file, tasks):
            counts.update(c)
    elif name == "tldr-pages":
        counts = convert_tldr(out_dir)
    elif name == "exercicios-v1":
        counts = convert_v1_exercises(out_dir)
    elif name == "banco-v1-corpus":
        counts = convert_v1_bank_corpus(out_dir)
    save_meta(name, out_dir, counts, start)


def list_sources() -> None:
    total_examples = total_tokens = 0
    for name in ALL:
        meta_path = FONTES_DIR / name / "meta.json"
        if not meta_path.exists():
            print(f"{name:<24}-")
            continue
        meta = json.loads(meta_path.read_text())
        total_examples += meta["exemplos"]
        total_tokens += meta["tokens_estimados"]
        print(f"{name:<24}{meta['exemplos']:>12,}{meta['tokens_estimados'] / 1e6:>10,.0f}M{meta['com_testes_e_solucao']:>11,}")
    print(f"{'total':<24}{total_examples:>12,}{total_tokens / 1e6:>10,.0f}M")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="fontes (padrão: todas)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.list:
        list_sources()
        return
    unknown = set(args.names) - set(ALL)
    if unknown:
        parser.error(f"fontes desconhecidas: {sorted(unknown)}; conhecidas: {ALL}")
    with ProcessPoolExecutor(args.workers) as pool:
        for name in args.names or ALL:
            convert(name, pool)


if __name__ == "__main__":
    main()
