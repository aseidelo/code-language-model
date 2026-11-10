"""Levantamento dos candidatos a dados de SFT (ver docs/sft.md).

Subcomandos:

``corpus``
    Percorre o split de treino do corpus de pré-treino
    (``data/the-stack-dedup/filtered/``) com tree-sitter e extrai as unidades
    que viram exemplos:

    * funções e métodos das 6 linguagens com docstring, com a documentação
      (docstring do Python, JSDoc, Javadoc, rustdoc, comentário de bloco em C)
      separada (T2, T5);
    * todas as funções e todos os arquivos, para a instrução reversa (T1, T3, T4).

    Salva as funções em ``data/codelm_sft/units/<lang>.parquet`` e
    as contagens por linguagem e nível em ``sft/candidates/corpus.json``.

``public``
    Para cada dataset público candidato, lê uma amostra, identifica a
    linguagem e o nível da resposta e estima quantos exemplos sobram em cada
    linguagem e nível no dataset inteiro. Salva ``sft/candidates/public.json``.

Os níveis vêm do tamanho em tokens do código (estimado por caracteres por
token de cada linguagem): fácil até 256, médio até 1.024, difícil até 4.096;
acima disso, "longo".

Uso:
    uv run python -m dataprep.sft_candidates corpus [linguagens...]
    uv run python -m dataprep.sft_candidates public [--sample 3000]
"""

import argparse
import hashlib
import inspect
import json
import re
import textwrap
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import tree_sitter_language_pack as tslp
from datasets import load_dataset
from huggingface_hub import get_session

from codelm.languages import LANGUAGES
from codelm.tokenizer import load_tokenizer
from dataprep.corpus import iter_natural_texts
from dataprep.pretrain import CODE_DIR, chars_per_token
from dataprep.sft_mix import SFT_DIR

UNITS_DIR = SFT_DIR / "units"
REPORT_DIR = SFT_DIR / "candidates"

LEVELS = {"facil": 256, "medio": 1024, "dificil": 4096}
DOC_LANGS = ("python", "javascript", "typescript", "java", "c", "rust")
MIN_FUNCTION_TOKENS = 16


def level(tokens: float) -> str:
    return next((name for name, limit in LEVELS.items() if tokens <= limit), "longo")


def load_chars_per_token() -> dict[str, float]:
    """Caracteres por token de cada linguagem e do inglês (guardado em cache)."""
    path = SFT_DIR / "chars_per_token.json"
    if path.exists():
        return json.loads(path.read_text())
    ratios = {lang: chars_per_token(lang) for lang in LANGUAGES}
    tokenizer = load_tokenizer()
    texts = list(iter_natural_texts("en", split="valid"))
    ratios["natural"] = sum(map(len, texts)) / sum(len(e.ids) for e in tokenizer.encode_batch(texts))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ratios, indent=1))
    return ratios


# --- corpus -----------------------------------------------------------------

FUNCTION_NODES = {
    "python": {"function_definition"},
    "javascript": {"function_declaration", "generator_function_declaration", "method_definition"},
    "typescript": {"function_declaration", "generator_function_declaration", "method_definition"},
    "java": {"method_declaration", "constructor_declaration"},
    "c": {"function_definition"},
    "rust": {"function_item"},
}
# Funções atribuídas a variáveis: ``const f = (x) => ...``.
FUNCTION_VALUES = {"arrow_function", "function_expression", "function"}
# Nós que envolvem a função e ficam entre ela e a documentação.
WRAPPERS = {"export_statement", "decorated_definition"}
COMMENT_NODES = {"comment", "block_comment", "line_comment"}
# Documentação que não descreve a função.
BAD_DOC = re.compile(r"copyright|licen[cs]e|auto-?generated|do not edit|eslint|@ts-|prettier-ignore", re.I)


def function_anchor(node, lang: str):
    """Nó que representa a função inteira (com ``export``/decorador), ou None se não é função."""
    if node.type in FUNCTION_NODES[lang]:
        anchor = node
    elif lang in ("javascript", "typescript") and node.type in ("lexical_declaration", "variable_declaration"):
        declarators = [c for c in node.named_children if c.type == "variable_declarator"]
        value = declarators[0].child_by_field_name("value") if len(declarators) == 1 else None
        if value is None or value.type not in FUNCTION_VALUES:
            return None
        anchor = node
    else:
        return None
    while anchor.parent is not None and anchor.parent.type in WRAPPERS:
        anchor = anchor.parent
    return anchor


def preceding_doc(anchor, lang: str) -> str | None:
    """Comentário de documentação colado antes da função (linhas seguidas, sem linha em branco)."""
    comments = []
    row = anchor.start_point[0]
    node = anchor.prev_sibling
    while node is not None:
        if lang == "rust" and node.type == "attribute_item":  # #[derive(...)] entre o doc e a função
            row = node.start_point[0]
        elif node.type in COMMENT_NODES and node.end_point[0] >= row - 1:
            comments.insert(0, node)
            row = node.start_point[0]
        else:
            break
        node = node.prev_sibling
    if not comments:
        return None
    texts = [c.text.decode("utf-8", "replace").rstrip("\n") for c in comments]
    if lang in ("javascript", "typescript", "java"):
        doc = texts[-1] if texts[-1].startswith("/**") else None
    elif lang == "rust":
        doc = "\n".join(t for t in texts if t.startswith(("///", "/**"))) or None
    else:  # C: qualquer comentário colado (não há convenção única)
        doc = "\n".join(texts)
    return doc


def python_docstring(node) -> tuple[str, object] | None:
    """Docstring de uma ``function_definition`` e o nó da instrução que a contém."""
    function = node.child_by_field_name("definition") if node.type == "decorated_definition" else node
    body = function.child_by_field_name("body")
    first = body.named_children[0] if body is not None and body.named_children else None
    if first is None:
        return None
    # Conforme a versão da gramática, a string vem solta no bloco ou dentro de um expression_statement.
    string = first.named_children[0] if first.type == "expression_statement" and first.named_children else first
    if string.type == "string":
        return string.text.decode("utf-8", "replace"), first
    return None


def clean_doc(raw: str, lang: str) -> str:
    """Texto da documentação sem os marcadores de comentário."""
    if lang == "python":
        return inspect.cleandoc(re.sub(r'^[rRuUbB]*("""|\'\'\'|"|\')|("""|\'\'\'|"|\')$', "", raw.strip()))
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(/\*+|\*+/|\*+(?!/)|//[/!]?)", "", line)
        line = re.sub(r"\*+/\s*$", "", line).rstrip()
        # Tira linhas só de -----, =====; cercas ``` e fechamentos como }); fazem parte dos exemplos.
        if not re.fullmatch(r"[^\w`{}()\[\];,]*", line) or not line.strip():
            lines.append(line)
    return textwrap.dedent("\n".join(lines)).strip()


def good_doc(doc: str) -> bool:
    words = re.findall(r"[^\W\d_]{2,}", doc)
    return 15 <= len(doc) <= 2000 and len(words) >= 4 and not BAD_DOC.search(doc)


def dedent_by(text: str, indent: int) -> str:
    """Tira até ``indent`` espaços do começo de cada linha (o recuo da função no arquivo)."""
    return "\n".join(re.sub(rf"^[ \t]{{0,{indent}}}", "", line) for line in text.split("\n"))


def function_node(anchor):
    """A função dentro do nó que a representa (tira ``export``, decorador e ``const f =``)."""
    node = anchor
    while node.type in WRAPPERS:
        node = node.child_by_field_name("declaration") or node.child_by_field_name("definition")
    if node.type in ("lexical_declaration", "variable_declaration"):
        declarator = next(c for c in node.named_children if c.type == "variable_declarator")
        node = declarator.child_by_field_name("value")
    return node


def extract_functions(content: str, lang: str) -> list[dict]:
    """Funções do arquivo: código, documentação e as variações usadas em T2 e T5.

    * ``signature``: a função até o começo do corpo (com a doc, vira a entrada do T2);
    * ``nodoc``: a função sem a documentação (entrada do T5);
    * ``doc_raw``: o comentário ou a docstring como está no arquivo;
    * ``doc_indent``: recuo da docstring dentro da função (só Python).
    """
    source = content.encode()
    tree = tslp.get_parser(lang).parse(source)
    functions, seen = [], set()
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        stack.extend(reversed(node.named_children))
        anchor = function_anchor(node, lang)
        if anchor is None or anchor.id in seen or anchor.has_error:
            continue
        seen.add(anchor.id)
        body = function_node(anchor).child_by_field_name("body")
        if body is None:  # declaração sem corpo (método abstrato, protótipo)
            continue
        line_start = source.rfind(b"\n", 0, anchor.start_byte) + 1
        indent = anchor.start_byte - line_start

        def text(start: int, end: int) -> str:
            return dedent_by(source[start:end].decode("utf-8", "replace"), indent)

        code = text(line_start, anchor.end_byte)
        signature = text(line_start, body.start_byte).rstrip()
        nodoc, doc_indent = None, 0
        if lang == "python":
            found = python_docstring(anchor)
            doc_raw = found[0] if found else None
            if found:
                statement = found[1]
                doc_raw = text(statement.start_byte, statement.end_byte)
                # Sem a docstring: corta as linhas dela (se ela não é a única instrução do corpo).
                start = source.rfind(b"\n", 0, statement.start_byte) + 1
                end = source.find(b"\n", statement.end_byte)
                end = len(source) if end < 0 else end + 1
                if len(body.named_children) > 1 and not source[start:statement.start_byte].strip() \
                        and end <= anchor.end_byte and not source[statement.end_byte:end].strip():
                    nodoc = text(line_start, start) + text(end, anchor.end_byte)
                    doc_indent = statement.start_byte - start - indent
                else:  # sem corpo além da docstring: não serve para T2 nem T5
                    doc_raw = None
        else:
            doc_raw = preceding_doc(anchor, lang)
            if doc_raw is not None:
                doc_raw = dedent_by(doc_raw, indent)
                nodoc = code
        doc = clean_doc(doc_raw, lang) if doc_raw else None
        if doc is not None and not good_doc(doc):
            doc = doc_raw = nodoc = None
        functions.append({"code": code, "doc": doc, "doc_raw": doc_raw, "signature": signature,
                          "nodoc": nodoc, "doc_indent": doc_indent})
    return functions


def process_part(task: tuple[str, str]) -> tuple[str, list[dict], Counter]:
    lang, path = task
    ratios = load_chars_per_token()
    table = pq.read_table(path, columns=["content", "repo", "path", "licenses"]).to_pylist()
    stats: Counter = Counter()
    units, seen = [], set()
    for row in table:
        tokens = len(row["content"]) / ratios[lang]
        stats[f"arquivos:{level(tokens)}"] += 1
        if lang not in DOC_LANGS:
            continue
        for fn in extract_functions(row["content"], lang):
            fn_tokens = len(fn["code"]) / ratios[lang]
            digest = hashlib.sha1(re.sub(r"\s+", "", fn["code"]).encode()).hexdigest()
            if fn_tokens < MIN_FUNCTION_TOKENS or digest in seen:
                continue
            seen.add(digest)
            lvl = level(fn_tokens)
            stats[f"funcoes:{lvl}"] += 1
            if fn["doc"]:
                stats[f"com_doc:{lvl}"] += 1
            units.append({"lang": lang, "repo": row["repo"], "path": row["path"], "licenses": row["licenses"], "level": lvl,
                          "tokens": round(fn_tokens), "digest": digest, **fn})
    return lang, units, stats


def corpus(langs: list[str]) -> None:
    load_chars_per_token()  # calcula o cache antes de abrir os processos
    UNITS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(lang, str(p)) for lang in langs for p in sorted((CODE_DIR / lang).glob("train-*.parquet"))]
    stats: dict[str, Counter] = defaultdict(Counter)
    repos: dict[str, set] = defaultdict(set)
    writers: dict[str, pq.ParquetWriter] = {}
    seen: dict[str, set] = defaultdict(set)  # duplicatas entre partes
    with ProcessPoolExecutor() as pool:
        for lang, units, part_stats in pool.map(process_part, tasks):
            stats[lang].update(part_stats)
            units = [u for u in units if u["digest"] not in seen[lang]]
            seen[lang].update(u["digest"] for u in units)
            repos[lang].update(u["repo"] for u in units if u["doc"])
            if units:
                table = pa.Table.from_pylist(units)
                if lang not in writers:
                    writers[lang] = pq.ParquetWriter(UNITS_DIR / f"{lang}.parquet", table.schema)
                writers[lang].write_table(table)
            print(f"  {lang}: {sum(v for k, v in part_stats.items() if k.startswith('arquivos')):,} arquivos", flush=True)
    for writer in writers.values():
        writer.close()

    report = {lang: {**dict(sorted(stats[lang].items())), "repos_com_doc": len(repos[lang])} for lang in langs}
    path = REPORT_DIR / "corpus.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**previous, **report}, indent=1))
    print_corpus(report)


def print_corpus(report: dict) -> None:
    header = f"{'ling':<11}" + "".join(f"{f'{kind}:{lvl}':>17}" for kind in ("com_doc", "funcoes", "arquivos")
                                       for lvl in ("facil", "medio", "dificil"))
    print(header)
    for lang, stats in report.items():
        print(f"{lang:<11}" + "".join(f"{stats.get(f'{kind}:{lvl}', 0):>17,}" for kind in ("com_doc", "funcoes", "arquivos")
                                      for lvl in ("facil", "medio", "dificil")))


# --- datasets públicos ------------------------------------------------------

LANG_ALIASES = {
    "python": "python", "py": "python", "python3": "python",
    "javascript": "javascript", "js": "javascript", "jsx": "javascript", "node": "javascript",
    "typescript": "typescript", "ts": "typescript", "tsx": "typescript",
    "java": "java", "c": "c", "h": "c", "rust": "rust", "rs": "rust",
    "html": "html", "htm": "html", "css": "css", "json": "json", "yaml": "yaml", "yml": "yaml",
}
FENCE = re.compile(r"```([\w+#.-]*)[^\n]*\n(.*?)```", re.DOTALL)


def guess_lang(code: str) -> str | None:
    """Linguagem de código sem cerca de Markdown (só os casos fáceis)."""
    stripped = code.strip()
    if stripped.startswith("<"):
        return "html"
    try:
        json.loads(stripped)
        return "json"
    except ValueError:
        pass
    try:
        compile(stripped, "<código>", "exec")
        return "python"
    except (SyntaxError, ValueError):
        return None


def main_code(response: str, declared: str | None) -> tuple[str | None, str]:
    """Linguagem e texto do maior bloco de código da resposta."""
    blocks = FENCE.findall(response)
    if blocks:
        tag, code = max(blocks, key=lambda b: len(b[1]))
        lang = LANG_ALIASES.get(tag.lower()) if tag else None
        if lang is None and not tag and declared:
            lang = LANG_ALIASES.get(declared.lower())
        return lang if lang or tag else guess_lang(code), code
    if declared:
        return LANG_ALIASES.get(declared.lower()), response
    return guess_lang(response), response


# Cada fonte: como ler pedido, resposta e linguagem declarada de uma linha.
PUBLIC = [
    # T1: pedido -> código
    dict(name="magicoder-oss", repo="ise-uiuc/Magicoder-OSS-Instruct-75K", tasks="T1", license="MIT",
         prompt="problem", response="solution", lang="lang"),
    dict(name="magicoder-evol", repo="ise-uiuc/Magicoder-Evol-Instruct-110K", tasks="T1", license="Apache-2.0",
         prompt="instruction", response="response"),
    dict(name="self-oss-instruct", repo="bigcode/self-oss-instruct-sc2-exec-filter-50k", tasks="T1",
         license="ODC-BY", prompt="instruction", response="response", fixed_lang="python"),
    dict(name="code-alpaca", repo="sahil2801/CodeAlpaca-20k", tasks="T1", license="CC-BY-4.0",
         prompt="instruction", response="output"),
    *[dict(name=f"opc-stage2/{c}", repo="OpenCoder-LLM/opc-sft-stage2", config=c, tasks="T1", license="MIT",
           prompt="instruction", response="output")
      for c in ("educational_instruct", "evol_instruct", "mceval_instruct", "package_instruct")],
    *[dict(name=f"opc-stage1/{c}", repo="OpenCoder-LLM/opc-sft-stage1", config=c, tasks="T1", license="MIT",
           prompt="instruction", response="output")
      for c in ("filtered_infinity_instruct", "largescale_diverse_instruct", "realuser_instruct")],
    dict(name="codefeedback", repo="m-a-p/CodeFeedback-Filtered-Instruction", tasks="T1", license="Apache-2.0",
         prompt="query", response="answer", lang="lang"),
    dict(name="glaive-v3", repo="glaiveai/glaive-code-assistant-v3", tasks="T1", license="Apache-2.0",
         prompt="question", response="answer"),
    dict(name="mceval-instruct", repo="Multilingual-Multimodal-NLP/McEval-Instruct", tasks="T1",
         license="CC-BY-SA-4.0", prompt="instruction", response="output", lang="language"),
    dict(name="rosetta-code", repo="christopher/rosetta-code", tasks="T1", license="GFDL",
         prompt="task_description", response="code", lang="language_name"),
    dict(name="websight", repo="HuggingFaceM4/WebSight", config="v0.2", tasks="T1 (html)", license="CC-BY-4.0",
         prompt="llm_generated_idea", response="text", fixed_lang="html", columns=["llm_generated_idea", "text"]),
    # T2/T5: docstring <-> código
    *[dict(name=f"code-search-net/{c}", repo="code-search-net/code_search_net", config=c, tasks="T2, T5",
           license="várias (por repositório)", prompt="func_documentation_string", response="func_code_string",
           fixed_lang=c)
      for c in ("python", "java", "javascript")],
    # RL: pedidos com testes
    dict(name="kodcode", repo="KodCode/KodCode-V1", tasks="RL", license="CC-BY-NC-4.0",
         prompt="question", response="solution", fixed_lang="python", extra="gpt_difficulty"),
    dict(name="open-code-instruct", repo="nvidia/OpenCodeInstruct", tasks="RL", license="CC-BY-4.0",
         prompt="input", response="output", extra="domain"),
    dict(name="acecode-87k", repo="TIGER-Lab/AceCode-87K", tasks="RL", license="MIT",
         prompt="question", response=None, fixed_lang="python", extra="source"),
    dict(name="acecode-v2", repo="TIGER-Lab/AceCode-V2-122K", tasks="RL", license="Apache-2.0",
         prompt="question", response=None, fixed_lang="python", extra="source"),
]


def num_rows(repo: str, config: str | None) -> int | None:
    """Linhas do split de treino, pela API do dataset viewer."""
    response = get_session().get("https://datasets-server.huggingface.co/size", params={"dataset": repo}, timeout=60)
    if response.status_code != 200:
        return None
    splits = [s for s in response.json()["size"]["splits"]
              if s["split"] == "train" and (config is None or s["config"] == config)]
    return sum(s["num_rows"] for s in splits) or None


def analyze_public(spec: dict, sample: int, ratios: dict[str, float]) -> dict:
    kwargs = {"columns": spec["columns"]} if "columns" in spec else {}
    rows = load_dataset(spec["repo"], spec.get("config"), split="train", streaming=True, **kwargs).take(sample)
    cells: Counter = Counter()
    extra: Counter = Counter()
    n = 0
    for row in rows:
        n += 1
        if spec.get("extra"):
            extra[str(row.get(spec["extra"]))] += 1
        if spec["response"] is None:  # só pedidos (RL): linguagem fixa, sem nível
            cells[(spec["fixed_lang"], "sem resposta")] += 1
            continue
        prompt, response = row[spec["prompt"]] or "", row[spec["response"]] or ""
        lang, code = main_code(response, spec.get("fixed_lang") or row.get(spec.get("lang", ""), None))
        if lang is None:
            cells[("outra", "-")] += 1
            continue
        code_tokens = len(code) / ratios[lang]
        total = code_tokens + (len(prompt) + len(response) - len(code)) / ratios["natural"]
        cells[(lang, level(code_tokens))] += 1
        cells[(lang, "cabe_2048")] += total <= 2048
    total_rows = num_rows(spec["repo"], spec.get("config")) or n
    scale = total_rows / max(n, 1)
    by_lang: dict[str, dict] = defaultdict(dict)
    for (lang, key), count in cells.items():
        by_lang[lang][key] = round(count * scale)
    return {**{k: spec[k] for k in ("name", "repo", "tasks", "license")}, "rows": total_rows, "sample": n,
            "by_lang": dict(by_lang), "extra": {k: round(v * scale) for k, v in extra.most_common(10)}}


def public(sample: int) -> None:
    ratios = load_chars_per_token()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "public.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for spec in PUBLIC:
        if spec["name"] in results:
            continue
        print(f"  {spec['name']}...", flush=True)
        try:
            results[spec["name"]] = analyze_public(spec, sample, ratios)
        except Exception as error:  # um dataset fora do ar não para o levantamento
            print(f"    erro: {type(error).__name__}: {str(error)[:200]}")
            continue
        path.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print_public(results)


def print_public(results: dict) -> None:
    langs = list(LANGUAGES)
    print(f"{'dataset':<34}{'linhas':>10}  " + "".join(f"{lang[:6]:>8}" for lang in langs) + f"{'outra':>9}")
    for name, result in results.items():
        counts = [sum(v for k, v in result["by_lang"].get(lang, {}).items() if k in (*LEVELS, "longo", "sem resposta"))
                  for lang in langs]
        other = sum(result["by_lang"].get("outra", {}).values())
        print(f"{name:<34}{result['rows']:>10,}  " + "".join(f"{c:>8,}" for c in counts) + f"{other:>9,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    corpus_cmd = commands.add_parser("corpus")
    corpus_cmd.add_argument("langs", nargs="*", default=list(LANGUAGES))
    public_cmd = commands.add_parser("public")
    public_cmd.add_argument("--sample", type=int, default=3000)
    args = parser.parse_args()
    if args.command == "corpus":
        corpus(args.langs)
    else:
        public(args.sample)


if __name__ == "__main__":
    main()
