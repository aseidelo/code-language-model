"""Pós-processamento do SFT v2: regras tiradas da revisão humana de 2026-11-21 (ver docs/memory/sft-v2.md).

Roda depois do QC (``dataprep.sft_qc``) e não muda nem apaga nada nas fontes. Para cada parte de
``v2/fontes/<fonte>/`` grava ``v2/post/<fonte>/<parte>`` com:

* ``post_falhas``: regras precisas. A mistura exclui o exemplo.
  * ``qa_vazado``: o pedido já traz a resposta (``**Answer**:``, ``答案：``), comum no opc-stage1-largescale;
  * ``sem_linguagem``: ``lang = outra:?``, resposta que não é código de nenhuma linguagem identificada
    (perguntas e respostas, prosa);
  * ``diff``: resposta em formato diff (``lang = outra:ndiff`` ou ``outra:diff``);
  * ``trivial``: comando do tldr que é só o nome do programa (``chatgpt`` -> ``chatgpt``).
* ``post_marcas``: heurísticas, só marcam até a revisão confirmar a precisão.
  * ``contexto_ausente``: o pedido manda mexer num código, array ou snippet que não veio junto, ou
    o contexto é só um link (codesandbox, jsfiddle...);
  * ``sem_codigo``: T1 sem bloco de código e com a resposta quase toda em prosa.
* ``lang_corrigido`` / ``nl_corrigido``: rótulo corrigido, ou nulo se o original está certo.
  * ``nl``: rotulado ``en``, mas o pedido (a prosa, ou a docstring no T2/T5) é chinês, japonês ou
    coreano -> ``outra:zh|ja|ko``;
  * ``lang``: o pedido cita uma linguagem que não é nossa (PHP, CUDA, C++...) e a resposta não
    parseia na linguagem do rótulo -> ``outra:<linguagem>``; ou o QC acusou ``linguagem``, o pedido
    cita só uma das nossas, a resposta não parseia no rótulo e parseia nela -> essa linguagem (nunca
    HTML/CSS/JSON/YAML, cujos parsers aceitam quase tudo).

Nas traduções (``traducao-*``), as regras de pedido usam o ``prompt_en``, e o ``nl`` não é corrigido.

    uv run python -m dataprep.sft_post [fonte ...] [--workers 4]
    uv run python -m dataprep.sft_post --relatorio
"""

import argparse
import json
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dataprep.sft_convert import FENCE, FONTES_DIR, V2_DIR
from dataprep.sft_qc import QC_DIR, STRICT_NAMES, syntax_ok

POST_DIR = V2_DIR / "post"
FALHAS = ("qa_vazado", "sem_linguagem", "diff", "trivial")
MARCAS = ("contexto_ausente", "sem_codigo")
MARKUP = ("html", "css", "json", "yaml")

QA = re.compile(r"\*\*\s*(Answer|答案|回答|解答)\s*\*\*|^\s*(Answer|答案|回答)\s*[:：]", re.M | re.I)

# Mexer num código que deveria estar no pedido. "the given string" é entrada normal de função: só
# conta quando o objeto é código ou dado colado.
REF_OBJECT = r"(code|snippet|script|program|function|query|array|table|class|method|json|html|xml|data ?set|dataframe)"
QUAL = r"(given|provided|following|above|below|existing|attached|erroneous|faulty|sample)"
# Com "the" o objeto precisa de qualificador: "modify the function" costuma falar de uma função
# descrita no próprio pedido. "the following X:" sem verbo antes costuma abrir uma especificação.
REF = re.compile(
    rf"\b(optimi[sz]e|edit|modify|update|fix|refactor|improve|debug|rewrite|convert|correct|change|complete|"
    rf"explain|analy[sz]e|review|translate|use|run|execute)\s+"
    rf"(the\s+({QUAL}\s+){{1,2}}|(this|that|my)\s+({QUAL}\s+){{0,2}}){REF_OBJECT}\b"
    rf"|\b(the|this)\s+(given|provided|above|below|attached)\s+{REF_OBJECT}\b"
    rf"|\b(given|provided)\s+(a|the)\s+(code )?snippet\b", re.I)
# O snippet citado só como inspiração ("as a starting point", "without referencing") não faz falta.
EXCUSE_AFTER = re.compile(r"^\W*(\w+\W+){0,3}as\s+(a|the)\s+(starting point|reference|template|basis|guide|inspiration)", re.I)
EXCUSE_BEFORE = re.compile(r"without\s+(referencing|using|relying on|looking at)\W*$", re.I)
INLINE_DATA = re.compile(r"[\[{][^\[\]{}\n]*,[^\[\]{}\n]*,[^\[\]{}\n]*[\]}]")


def missing_context(prompt: str) -> bool:
    """O pedido cita código ou dado que não veio junto (ver REF)."""
    if has_code(prompt):
        return False
    if SANDBOX.search(prompt):
        return True
    for m in REF.finditer(prompt):
        if EXCUSE_AFTER.search(prompt[m.end():m.end() + 60]) or EXCUSE_BEFORE.search(prompt[max(0, m.start() - 30):m.start()]):
            continue
        return True
    return False


SANDBOX = re.compile(r"https?://(www\.)?(codesandbox\.io|jsfiddle\.net|stackblitz\.com|codepen\.io|pastebin\.com|"
                     r"jsbin\.com|repl\.it|replit\.com|plnkr\.co|ideone\.com|gist\.github\.com|github\.com/\S+/blob/|"
                     r"i\.stack\.imgur\.com|imgur\.com)", re.I)

# Linguagens que não são nossas e aparecem nomeadas nos pedidos.
OTHER_LANGS = {
    "php": r"\bphp\b", "cpp": r"\bc\s*\+\+|\bcpp\b", "csharp": r"\bc\s*#|\bc ?sharp\b|\.net\b", "go": r"\bgolang\b|\bin go\b",
    "cuda": r"\bcuda\b", "kotlin": r"\bkotlin\b", "swift": r"\bswift\b", "ruby": r"\bruby\b", "perl": r"\bperl\b",
    "scala": r"\bscala\b", "sql": r"\bsql\b", "matlab": r"\bmatlab\b", "dart": r"\bdart\b", "lua": r"\blua\b",
    "haskell": r"\bhaskell\b", "powershell": r"\bpowershell\b", "vba": r"\bvba\b|\bvisual basic\b",
    "objc": r"\bobjective[- ]c\b", "julia": r"\bjulia\b", "assembly": r"\bassembly\b|\basm\b", "solidity": r"\bsolidity\b",
    "r": r"\bin R\b|\bR (code|script|programming|language)\b",
}
CASE_SENSITIVE = {"r"}

HAN = re.compile(r"[一-鿿]")
KANA = re.compile(r"[぀-ヿ]")
HANGUL = re.compile(r"[가-힯]")
LETTER = re.compile(r"[^\W\d_]")


def cjk_language(text: str) -> str | None:
    """``zh``/``ja``/``ko`` quando esses caracteres são pelo menos 10% das letras do texto."""
    letters = len(LETTER.findall(text))
    if not letters:
        return None
    han, kana, hangul = len(HAN.findall(text)), len(KANA.findall(text)), len(HANGUL.findall(text))
    if (han + kana + hangul) < max(3, 0.1 * letters):
        return None
    return "ko" if hangul > han + kana else "ja" if kana else "zh"


def has_code(prompt: str) -> bool:
    """O pedido traz código ou dado colado: bloco cercado ou várias linhas com cara de código."""
    if FENCE.search(prompt) or "```" in prompt:
        return True
    lines = [l for l in prompt.splitlines() if l.strip()]
    codey = sum(1 for l in lines if l.startswith(("    ", "\t")) or re.search(r"[{};=\[\]<>:)]\s*$", l.strip()))
    # Dado colado na linha: "organize the provided array [\"7\", \"2\", \"45\"]".
    return codey >= 3 or bool(INLINE_DATA.search(prompt))


def prose_line(line: str) -> bool:
    text = re.sub(r"<[^>]+>", " ", line).strip()  # <p>texto</p> é prosa
    if len(text.split()) < 8 or text.startswith(("#", "//", "/*", "*", "--", "<!--")):
        return False
    return sum(c.isalpha() or c.isspace() for c in text) / len(text) > 0.9


DOCSTRING = re.compile(r"(\"\"\"|''').*?\1", re.S)
LINE_COMMENT = {"python": r"#.*$", "bash": r"#.*$", "javascript": r"//.*$", "typescript": r"//.*$", "java": r"//.*$",
                "c": r"//.*$", "rust": r"//.*$"}


def mostly_prose(response: str, lang: str = "") -> bool:
    if "```" in response:
        return False
    # Docstring e comentário são prosa dentro de código bom.
    if lang == "python":
        response = DOCSTRING.sub("", response)
    if lang in LINE_COMMENT:
        response = re.sub(LINE_COMMENT[lang], "", response, flags=re.M)
    lines = [l for l in response.splitlines() if l.strip()]
    return len(lines) >= 3 and sum(map(prose_line, lines)) / len(lines) > 0.6


def named_langs(prompt: str, patterns: dict) -> set[str]:
    return {l for l, p in patterns.items() if re.search(p, prompt, 0 if l in CASE_SENSITIVE or l == "c" else re.I)}


def fixed_lang(row: dict, qc: dict | None, prompt: str) -> str | None:
    lang, code = row["lang"] or "", row.get("response") or ""
    if lang.startswith("outra") or row["task"] not in ("T1",) or (row.get("kind") or "code") != "code":
        return None
    syntax = (qc or {}).get("qc_sintaxe")
    others = named_langs(prompt, OTHER_LANGS)
    ours = named_langs(prompt, STRICT_NAMES)
    if others and syntax is False and lang not in ours:
        return "outra:" + sorted(others)[0]
    # Só quando a resposta não parseia no rótulo, e nunca para markup: os parsers de HTML/YAML/CSS
    # aceitam quase tudo (bash que mexe em "report.yaml" virava yaml, Python que gera HTML virava html).
    if (qc or {}).get("qc_linguagem") is False and syntax is False and len(ours) == 1:
        target = next(iter(ours))
        if target not in MARKUP and target != lang and code.strip() and syntax_ok(code, target):
            return target
    return None


def check(row: dict, qc: dict | None, translated: bool) -> dict:
    prompt = (row.get("prompt_en") if translated else row["prompt"]) or ""
    response = row.get("response") or ""
    lang, task = row["lang"] or "", row["task"]
    falhas, marcas = [], []
    if QA.search(prompt):
        falhas.append("qa_vazado")
    if lang == "outra:?":
        falhas.append("sem_linguagem")
    if lang in ("outra:ndiff", "outra:diff"):
        falhas.append("diff")
    if row.get("fonte") == "tldr-pages" and response.strip() == prompt.split(":", 1)[0].strip():
        falhas.append("trivial")
    if task == "T1":
        if missing_context(prompt):
            marcas.append("contexto_ausente")
        if not lang.startswith("outra") and (row.get("kind") or "code") == "code" and mostly_prose(response, lang):
            marcas.append("sem_codigo")
    nl = None
    if not translated and row["nl"] == "en":
        text = row.get("code") if task in ("T2", "T5") else FENCE.sub("", prompt)
        found = cjk_language(text or "")
        nl = f"outra:{found}" if found else None
    return {"id": row["id"], "post_falhas": falhas, "post_marcas": marcas,
            "lang_corrigido": fixed_lang(row, qc, prompt), "nl_corrigido": nl}


POST_SCHEMA = pa.schema([("id", pa.string()), ("post_falhas", pa.list_(pa.string())),
                         ("post_marcas", pa.list_(pa.string())), ("lang_corrigido", pa.string()),
                         ("nl_corrigido", pa.string())])
COLUMNS = ("id", "task", "lang", "nl", "kind", "prompt", "prompt_en", "code", "response")


def post_part(task: tuple[str, str, str, bool]) -> str:
    fonte, path, out, redo = task
    if Path(out).exists() and not redo:
        return out
    names = pq.read_schema(path).names
    rows = pq.read_table(path, columns=[c for c in COLUMNS if c in names]).to_pylist()
    qc_path = QC_DIR / fonte / Path(path).name
    qc = {}
    if qc_path.exists():
        qc = {r["id"]: r for r in pq.read_table(qc_path, columns=["id", "qc_sintaxe", "qc_linguagem"]).to_pylist()}
    translated = fonte.startswith("traducao")
    results = [check({**r, "fonte": fonte}, qc.get(r["id"]), translated) for r in rows]
    tmp = Path(out).with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(results, schema=POST_SCHEMA), tmp, compression="zstd")
    tmp.rename(out)
    return out


def report() -> None:
    total, counts = Counter(), Counter()
    for fonte_dir in sorted(POST_DIR.glob("*")):
        for part in fonte_dir.glob("part-*.parquet"):
            t = pq.read_table(part).to_pylist()
            total[fonte_dir.name] += len(t)
            for r in t:
                for f in r["post_falhas"]:
                    counts[(fonte_dir.name, f)] += 1
                for m in r["post_marcas"]:
                    counts[(fonte_dir.name, m)] += 1
                if r["lang_corrigido"]:
                    counts[(fonte_dir.name, "lang:" + r["lang_corrigido"])] += 1
                if r["nl_corrigido"]:
                    counts[(fonte_dir.name, "nl:" + r["nl_corrigido"])] += 1
    out = {f: {k: v for (ff, k), v in sorted(counts.items()) if ff == f} | {"exemplos": n} for f, n in total.items()}
    (POST_DIR / "relatorio.json").write_text(json.dumps(out, indent=1))
    keys = [*FALHAS, *MARCAS]
    print(f"{'fonte':<26}{'exemplos':>11}" + "".join(f"{k[:12]:>13}" for k in keys) + f"{'lang fix':>10}{'nl fix':>9}")
    for f, n in total.items():
        langs = sum(v for (ff, k), v in counts.items() if ff == f and k.startswith("lang:"))
        nls = sum(v for (ff, k), v in counts.items() if ff == f and k.startswith("nl:"))
        cells = "".join(f"{100 * counts[(f, k)] / n:>12.2f}%" for k in keys)
        print(f"{f:<26}{n:>11,}{cells}{langs:>10,}{nls:>9,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fontes", nargs="*", help="fontes (padrão: todas as que têm partes)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--relatorio", action="store_true")
    parser.add_argument("--refazer", action="store_true", help="recalcula as partes que já têm saída (regras mudaram)")
    args = parser.parse_args()
    if args.relatorio:
        report()
        return
    fontes = args.fontes or sorted(p.name for p in FONTES_DIR.iterdir() if p.is_dir() and any(p.glob("part-*.parquet")))
    tasks = []
    for fonte in fontes:
        (POST_DIR / fonte).mkdir(parents=True, exist_ok=True)
        tasks += [(fonte, str(p), str(POST_DIR / fonte / p.name), args.refazer) for p in sorted((FONTES_DIR / fonte).glob("part-*.parquet"))]
    start = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(post_part, tasks, chunksize=4), 1):
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} partes, {time.time() - start:.0f}s", flush=True)
    report()


if __name__ == "__main__":
    main()
