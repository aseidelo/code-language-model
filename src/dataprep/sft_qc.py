"""Controle de qualidade do SFT v2, camada 1: verificações determinísticas (ver docs/sft-v2.md).

Para cada parte de ``v2/fontes/<fonte>/`` grava ``v2/qc/<fonte>/<parte>`` com o ``id``
do exemplo e as colunas ``qc_*``. ``True`` passou, ``False`` reprovou, nulo não se
aplica (por exemplo, sintaxe de uma linguagem que não é nossa). ``qc_falhas`` lista as
verificações reprovadas. O QC não apaga nada: a mistura decide o que usa.

O código verificado é a resposta (T1, T2, T5, T6), na linguagem do exemplo.

* ``qc_sintaxe``: ``ast`` (Python), ``json.loads``, ``yaml``, tree-sitter nas outras
  (métodos soltos de Java/JS/TS são testados também dentro de uma classe);
* ``qc_gramatica`` (Python e JSON, com ``--gramatica``): a máscara do modelo aceita a
  resposta inteira. Cara (~25 mil tokens/s por núcleo);
* ``qc_indentacao``: sem recuo ímpar nem mistura de tabs e espaços;
* ``qc_incompleto``: sem placeholder (``# your code here``, ``TODO``, corpo só com
  ``pass``/``...``), com código de verdade e cerca de Markdown fechada;
* ``qc_linguagem``: o pedido não nomeia outra linguagem que não a da resposta;
* ``qc_tamanho``: pedido com pelo menos 3 palavras e exemplo que cabe em 8k tokens;
  ``qc_cabe_2k`` só informa;
* ``qc_lixo``: sem linha gigante, blob base64, caractere de controle ou repetição
  excessiva (razão de compressão).

    uv run python -m dataprep.sft_qc [fonte ...] [--gramatica]
    uv run python -m dataprep.sft_qc --relatorio
"""

import argparse
import ast
import json
import re
import time
import warnings
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import tree_sitter_language_pack as tslp
import yaml

from codelm.indentation import has_mixed_indentation, has_odd_indentation, normalize_code
from dataprep.sft_convert import FONTES_DIR, LANG_NAMES, V2_DIR

QC_DIR = V2_DIR / "qc"
CODE_TASKS = ("T1", "T2", "T5", "T6", "T7")
TREE_SITTER = ("javascript", "typescript", "java", "c", "rust", "bash", "html", "css")
CHECKS = ("sintaxe", "gramatica", "indentacao", "incompleto", "linguagem", "tamanho", "lixo")
MAX_TOKENS = 8192
MAX_LINE = 1000

PLACEHOLDER = re.compile(
    r"your code (goes )?here|insert (your )?code|implement (this|me|the logic)|add your (code|implementation)"
    r"|\bTODO\b|\bFIXME\b|<\s*your[ _-]", re.I)
BASE64 = re.compile(r"[A-Za-z0-9+/=]{300,}")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Para a coerência de linguagem, C só conta quando aparece claramente como linguagem.
STRICT_NAMES = {**LANG_NAMES, "c": r"\bC (language|program|function|code)\b|\b(in|using|with|em|en) C\b|\bANSI C\b"}


# --- verificações --------------------------------------------------------------------

def syntax_ok(code: str, lang: str) -> bool:
    if lang == "python":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ast.parse(code)
            return True
        except (SyntaxError, ValueError):
            return False
    if lang == "json":
        try:
            json.loads(code)
            return True
        except ValueError:
            return False
    if lang == "yaml":
        try:
            list(yaml.safe_load_all(code))
            return True
        except yaml.YAMLError:
            return False
    parser = tslp.get_parser(lang)
    if not parser.parse(code.encode()).root_node.has_error:
        return True
    if lang in ("java", "javascript", "typescript"):  # método solto (T2/T5): vale dentro de uma classe
        return not parser.parse(f"class _Q {{\n{code}\n}}".encode()).root_node.has_error
    return False


def incomplete(code: str, lang: str, raw: str | None) -> bool:
    if not code.strip():
        return True
    if raw is not None and raw.count("```") % 2:
        return True
    if PLACEHOLDER.search(code):
        return True
    if lang == "python":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(code)
        except (SyntaxError, ValueError):
            return False  # já reprova na sintaxe
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    body = body[1:]  # docstring
                if not body or all(isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(
                        getattr(s, "value", None), ast.Constant) and s.value.value is Ellipsis) for s in body):
                    # Métodos abstratos e protocolos são corpo vazio de propósito.
                    if not any(isinstance(d, (ast.Name, ast.Attribute)) and "abstract" in ast.unparse(d)
                               for d in node.decorator_list):
                        return True
    return False


def language_mismatch(prompt: str, lang: str) -> bool:
    named = {l for l, pattern in STRICT_NAMES.items()
             if re.search(pattern, prompt, 0 if l == "c" else re.I)}
    if lang == "typescript":
        named.discard("javascript")  # pedidos de TS citam JS com frequência
    if lang == "javascript":
        named -= {"html", "css", "json"}  # JS que manipula HTML/JSON
    if lang in ("html", "css"):
        named -= {"javascript", "css", "html"}
    return bool(named) and lang not in named


def garbage(text: str) -> bool:
    if not text:
        return False
    if CONTROL.search(text) or BASE64.search(text):
        return True
    if max(map(len, text.splitlines()), default=0) > MAX_LINE:
        return True
    if len(text) > 2000:
        data = text.encode()
        return len(zlib.compress(data)) / len(data) < 0.08
    return False


@lru_cache(maxsize=1)
def grammar_tools():
    from codelm.chat_template import EOS
    from codelm.grammar import GrammarConstraint
    from codelm.tokenizer import encode_code, load_tokenizer
    tokenizer = load_tokenizer()
    return tokenizer, tokenizer.token_to_id(EOS), encode_code, GrammarConstraint


def grammar_ok(code: str, lang: str) -> bool:
    tokenizer, eos, encode_code, GrammarConstraint = grammar_tools()
    ids = [*encode_code(tokenizer, code, lang), eos]
    return GrammarConstraint(lang).validate(ids) == len(ids)


def check(row: dict, with_grammar: bool) -> dict:
    task, lang = row["task"], row["lang"] or ""
    # Só resposta de código passa pelas checagens de código (o negativo do T6s responde em texto).
    code = row["response"] if task in CODE_TASKS and (row.get("kind") or "code") == "code" else None
    ours = not lang.startswith("outra")
    result: dict = {"id": row["id"]}
    words = len(re.findall(r"\w+", row["prompt"] or ""))
    tokens = row["n_tokens"] or 0
    result["qc_tamanho"] = words >= 3 and tokens <= MAX_TOKENS
    result["qc_cabe_2k"] = tokens <= 2048
    result["qc_lixo"] = not garbage(code or "") and not garbage(row["prompt"] or "")
    if code is None or not ours:
        result.update(qc_sintaxe=None, qc_gramatica=None, qc_indentacao=None, qc_linguagem=None,
                      qc_incompleto=None if code is None else bool(code.strip()))
    else:
        syntax = syntax_ok(code, lang)
        result["qc_sintaxe"] = syntax
        result["qc_gramatica"] = (syntax and grammar_ok(code, lang)) if with_grammar and lang in ("python", "json") \
            else None
        normalized = normalize_code(code)
        result["qc_indentacao"] = not has_mixed_indentation(code) and not has_odd_indentation(normalized, lang)
        result["qc_incompleto"] = not incomplete(code, lang, row.get("response_raw"))
        result["qc_linguagem"] = not language_mismatch(row["prompt"] or "", lang)
    result["qc_falhas"] = [c for c in CHECKS if result.get(f"qc_{c}") is False]
    return result


QC_SCHEMA = pa.schema([("id", pa.string()), *[(f"qc_{c}", pa.bool_()) for c in CHECKS],
                       ("qc_cabe_2k", pa.bool_()), ("qc_falhas", pa.list_(pa.string()))])


def qc_part(task: tuple[str, str, bool]) -> str:
    path, out, with_grammar = task
    if Path(out).exists():
        return out
    names = pq.read_schema(path).names
    columns = [c for c in ("id", "task", "lang", "prompt", "response", "response_raw", "n_tokens", "kind") if c in names]
    rows = pq.read_table(path, columns=columns).to_pylist()
    results = [check(r, with_grammar) for r in rows]
    tmp = Path(out).with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(results, schema=QC_SCHEMA), tmp, compression="zstd")
    tmp.rename(out)
    return out


# --- relatório -------------------------------------------------------------------------

def report_fonte(fonte: str) -> dict:
    """Taxa de reprovação por verificação, por linguagem, de uma fonte."""
    qc_parts = sorted((QC_DIR / fonte).glob("part-*.parquet"))
    stats: dict[str, Counter] = defaultdict(Counter)
    for qc_path in qc_parts:
        langs = pq.read_table(FONTES_DIR / fonte / qc_path.name, columns=["lang"]).column("lang").to_pylist()
        qc = pq.read_table(qc_path).to_pylist()
        for lang, r in zip(langs, qc):
            key = lang if not lang.startswith("outra") else "outra"
            s = stats[key]
            s["exemplos"] += 1
            s["aprovados"] += not r["qc_falhas"]
            s["cabe_2k"] += bool(r["qc_cabe_2k"])
            for c in CHECKS:
                if r[f"qc_{c}"] is not None:
                    s[f"{c}:testado"] += 1
                    s[f"{c}:reprovado"] += r[f"qc_{c}"] is False
    report = {}
    for lang, s in sorted(stats.items(), key=lambda kv: -kv[1]["exemplos"]):
        report[lang] = {"exemplos": s["exemplos"], "aprovados": round(s["aprovados"] / s["exemplos"], 4),
                        "cabe_2k": round(s["cabe_2k"] / s["exemplos"], 4),
                        **{c: round(s[f"{c}:reprovado"] / s[f"{c}:testado"], 4)
                           for c in CHECKS if s[f"{c}:testado"]}}
    return report


def print_report() -> None:
    reports = {}
    for qc_dir in sorted(QC_DIR.glob("*")):
        if qc_dir.is_dir() and any(qc_dir.glob("part-*.parquet")):
            reports[qc_dir.name] = report_fonte(qc_dir.name)
            (qc_dir / "relatorio.json").write_text(json.dumps(reports[qc_dir.name], indent=1))
    header = f"{'fonte':<24}{'ling':<12}{'exemplos':>10}{'aprov':>7}" + "".join(f"{c[:6]:>8}" for c in CHECKS)
    print(header + "   (% reprovado)")
    for fonte, report in reports.items():
        for lang, r in report.items():
            if r["exemplos"] < 1000 and lang == "outra":
                continue
            cells = "".join(f"{100 * r[c]:>7.1f}%" if c in r else f"{'-':>8}" for c in CHECKS)
            print(f"{fonte:<24}{lang:<12}{r['exemplos']:>10,}{100 * r['aprovados']:>6.1f}%{cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fontes", nargs="*", help="fontes (padrão: todas as convertidas)")
    parser.add_argument("--gramatica", action="store_true", help="valida Python e JSON na máscara do modelo")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--relatorio", action="store_true")
    args = parser.parse_args()
    if args.relatorio:
        print_report()
        return
    fontes = args.fontes or sorted(p.parent.name for p in FONTES_DIR.glob("*/meta.json"))
    tasks = []
    for fonte in fontes:
        (QC_DIR / fonte).mkdir(parents=True, exist_ok=True)
        tasks += [(str(p), str(QC_DIR / fonte / p.name), args.gramatica)
                  for p in sorted((FONTES_DIR / fonte).glob("part-*.parquet"))]
    start = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(qc_part, tasks), 1):
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} partes, {time.time() - start:.0f}s", flush=True)
    print_report()


if __name__ == "__main__":
    main()
