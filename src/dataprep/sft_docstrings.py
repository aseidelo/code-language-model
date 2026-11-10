"""Exemplos de T2 (docstring -> código) e T5 (código -> docstring) a partir do corpus.

Parte das funções documentadas extraídas por ``dataprep.sft_candidates corpus``
(``sft/units/<lang>.parquet``). Subcomandos, nesta ordem:

``select``
    Escolhe as funções de cada célula (tarefa × nível × linguagem × idioma),
    respeitando as cotas de ``dataprep.sft_mix``: no máximo ``MAX_PER_REPO``
    funções por repositório e cada função em uma única célula. Docs já escritas
    em um dos 6 idiomas preenchem primeiro as células desse idioma; as em
    inglês preenchem o resto e são traduzidas. Salva ``sft/docstrings/selected.parquet``
    e os textos a traduzir em ``sft/docstrings/translate_in.jsonl``.

``build``
    Monta os exemplos no formato do banco (``sft/bank/t2_t5.parquet``), com a
    documentação traduzida reescrita no estilo de comentário da linguagem.
    Exemplos que precisam de tradução e não têm uma utilizável ficam de fora.

``repair``
    Grava em ``sft/docstrings/translate_fix_in.jsonl`` as docs a traduzir de
    novo; a saída entra no ``build`` por ``--fixes``.

Tradução (entre ``select`` e ``build``), com ``dataprep.translate``:

    uv run python -m dataprep.translate submit data/codelm_sft/docstrings/translate_in.jsonl
    uv run python -m dataprep.translate collect <batch_id> data/codelm_sft/docstrings/translate_out.jsonl \\
        --input data/codelm_sft/docstrings/translate_in.jsonl

Uso:
    uv run python -m dataprep.sft_docstrings select
    uv run python -m dataprep.sft_docstrings build [--translations arquivo.jsonl ...]
"""

import argparse
import json
import random
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from lingua import LanguageDetectorBuilder

from codelm.chat_template import encode_example
from codelm.languages import ASSETS_DIR, NATURAL_LANGUAGES
from codelm.tokenizer import load_tokenizer
from dataprep.sft_candidates import clean_doc
from dataprep.sft_mix import BANK_COLUMNS, BANK_DIR, DOC_LANGS, MIXES, SFT_DIR, quota
from dataprep.translate import placeholder

UNITS_DIR = SFT_DIR / "units"
WORK_DIR = SFT_DIR / "docstrings"
INSTRUCTIONS = ASSETS_DIR / "sft" / "instructions.json"

TASKS = ("T2", "T5")
MAX_PER_REPO = 20
MIN_TOKENS = 24
SEED = 0
# Testes não servem: o nome e a doc dizem pouco sobre o comportamento.
TEST_FUNCTION = re.compile(r"(\bdef\s+test|\bfn\s+test_|@Test\b|\bit\s*\(|\bdescribe\s*\()")

NATIVE_CONFIDENCE = 0.5
# Docs geradas por ferramenta ou cabeçalhos de arquivo colados na primeira função.
BAD_DOC = re.compile(r"@generated|user-doc|<!--|^\s*created (by|on|with)\b|@author\b", re.I | re.M)
def doc_prose(doc: str) -> str:
    """Só a prosa da doc (sem linhas de tag como @param, :param, Args:), para detectar o idioma."""
    lines = [line for line in doc.splitlines() if not re.match(r"\s*(@|:\w+|\w+:\s*$)", line)]
    return " ".join(lines) or doc


# --- select -----------------------------------------------------------------

def load_candidates(lang: str) -> list[dict]:
    columns = ["repo", "path", "licenses", "level", "tokens", "digest", "code", "doc", "doc_raw", "signature", "nodoc", "doc_indent"]
    rows = pq.read_table(UNITS_DIR / f"{lang}.parquet", columns=columns,
                         filters=[("doc", "!=", ""), ("nodoc", "!=", "")]).to_pylist()
    return [{**r, "lang": lang} for r in rows if r["doc"] and r["nodoc"] and r["tokens"] >= MIN_TOKENS
            and not TEST_FUNCTION.search(r["signature"] + "\n" + r["doc_raw"])
            and not BAD_DOC.search(r["doc"]) and prose_words(r["doc"]) >= 4]


def detect_languages(docs: list[str]) -> list[str]:
    """Idioma de cada doc: um dos 6, ou "?" quando é outro ou incerto.

    Docs curtas são ambíguas ("insert line for each cluster" sai como alemão),
    então só contam como de outro idioma com confiança alta. As incertas não
    vão para as células em inglês (que usam a doc sem traduzir), só para as
    traduzidas: a tradução funciona a partir de qualquer idioma.
    """
    detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
    labels = []
    for values in detector.compute_language_confidence_values_in_parallel(list(map(doc_prose, docs))):
        top = values[0] if values else None
        code = top.language.iso_code_639_1.name.lower() if top else None
        if code == "en" or (code in NATURAL_LANGUAGES and top.value >= NATIVE_CONFIDENCE):
            labels.append(code)
        else:
            labels.append("?")
    return labels


def prose_words(doc: str) -> int:
    """Palavras fora de exemplos de código (linhas >>> e ...)."""
    text = "\n".join(line for line in doc.splitlines() if not line.lstrip().startswith((">>>", "...")))
    return len(re.findall(r"[^\W\d_]{2,}", text))


def select_cells(lang: str, level: str, pool: list[dict]) -> list[dict]:
    """Distribui as funções do nível entre as células de T2 e T5 dos 6 idiomas."""
    need = {(task, nl): quota(task, level, lang, nl) for task in TASKS for nl in NATURAL_LANGUAGES}
    by_nl: dict[str, list[dict]] = defaultdict(list)
    for unit in pool:
        by_nl[unit["doc_nl"]].append(unit)

    chosen = []

    def fill(units: list[dict], cells: list[tuple[str, str]]) -> None:
        for unit in units:
            open_cells = [c for c in cells if need[c] > 0]
            if not open_cells:
                return
            cell = max(open_cells, key=lambda c: need[c])  # a célula mais vazia primeiro
            need[cell] -= 1
            chosen.append({**unit, "task": cell[0], "nl": cell[1]})

    # Docs já escritas em outro idioma preenchem as células desse idioma; o inglês, o resto.
    for nl in NATURAL_LANGUAGES:
        if nl != "en":
            fill(by_nl[nl], [(task, nl) for task in TASKS])
    fill(by_nl["en"], list(need))
    fill(by_nl["?"], [c for c in need if c[1] != "en"])
    return chosen


def select() -> None:
    rng = random.Random(SEED)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    selected, report = [], []
    for lang in DOC_LANGS:
        units = load_candidates(lang)
        for unit, nl in zip(units, detect_languages([u["doc"] for u in units])):
            unit["doc_nl"] = nl
        rng.shuffle(units)
        per_repo: Counter = Counter()
        pools: dict[str, list[dict]] = defaultdict(list)
        for unit in units:
            key = (unit["repo"], unit["level"])
            if per_repo[key] < MAX_PER_REPO:
                per_repo[key] += 1
                pools[unit["level"]].append(unit)
        for level in MIXES:
            chosen = select_cells(lang, level, pools[level])
            selected += chosen
            for task in TASKS:
                got = [c for c in chosen if c["task"] == task]
                report.append((lang, level, task, quota(task, level, lang), len(got),
                               sum(c["doc_nl"] != c["nl"] for c in got)))
        print(f"  {lang}: {len(units):,} funções documentadas", flush=True)

    for unit in selected:
        unit["id"] = f"{unit['task']}-{unit['lang']}-{unit['digest'][:16]}"
    pq.write_table(pa.Table.from_pylist(selected), WORK_DIR / "selected.parquet")
    jobs = [{"id": u["id"], "text": u["doc"], "target": u["nl"], "kind": "docstring"}
            for u in selected if u["doc_nl"] != u["nl"]]
    with (WORK_DIR / "translate_in.jsonl").open("w") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")

    print(f"\n{'ling':<11}{'nível':<9}{'tarefa':<7}{'meta':>8}{'escolhidas':>12}{'a traduzir':>12}")
    for lang, level, task, target, got, translate in report:
        flag = "" if got >= 0.99 * target else f"  falta {target - got:,}"
        print(f"{lang:<11}{level:<9}{task:<7}{target:>8,}{got:>12,}{translate:>12,}{flag}")
    chars = sum(len(j["text"]) for j in jobs)
    print(f"\n{len(selected):,} exemplos; {len(jobs):,} docs a traduzir ({chars / 1e6:.1f}M caracteres) "
          f"em {WORK_DIR / 'translate_in.jsonl'}")


# --- build ------------------------------------------------------------------

def render_doc(text: str, lang: str, original: str, indent: int = 0) -> str:
    """Documentação traduzida no estilo de comentário da linguagem (e do original)."""
    lines = text.strip().splitlines()
    if lang == "python":
        text = text.strip().replace('"""', r"\"\"\"")
        if len(lines) == 1:
            return f'"""{text}"""'
        pad = " " * indent
        body = "\n".join(pad + line if line.strip() else "" for line in text.splitlines()[1:])
        return f'"""{text.splitlines()[0]}\n{body}\n{pad}"""'
    lines = [line.replace("*/", "* /") for line in lines]
    if lang == "rust" or (lang == "c" and original.startswith("//")):
        marker = "///" if lang == "rust" else "//"
        return "\n".join(f"{marker} {line}".rstrip() for line in lines)
    opener = "/**" if original.startswith("/**") or lang != "c" else "/*"
    if len(lines) == 1 and len(lines[0]) <= 80:
        return f"{opener} {lines[0]} */"
    return "\n".join([opener, *(f" * {line}".rstrip() for line in lines), " */"])


def with_doc(unit: dict, doc_block: str) -> tuple[str, str] | None:
    """(assinatura com a doc, função completa com a doc)."""
    signature, nodoc = unit["signature"], unit["nodoc"]
    if unit["lang"] != "python":
        return f"{doc_block}\n{signature}", f"{doc_block}\n{nodoc}"
    if not nodoc.startswith(signature):
        return None
    pad = " " * unit["doc_indent"]
    head = f"{signature}\n{pad}{doc_block}"
    return head, head + nodoc[len(signature):]


def build_example(unit: dict, translation: str | None, instructions: dict, rng: random.Random) -> dict | None:
    if translation is None:  # doc original, como está no arquivo
        doc_block = unit["doc_raw"]
    else:
        doc_block = render_doc(translation, unit["lang"], unit["doc_raw"], unit["doc_indent"])
    pair = with_doc(unit, doc_block)
    if pair is None:
        return None
    head, full = pair
    if unit["lang"] == "python":
        try:
            with warnings.catch_warnings():  # escapes inválidos em strings só avisam
                warnings.simplefilter("ignore")
                compile(full, "<função>", "exec")
        except SyntaxError:
            return None
    code_in = head if unit["task"] == "T2" else unit["nodoc"]
    return {
        "id": unit["id"], "task": unit["task"], "level": unit["level"], "lang": unit["lang"], "nl": unit["nl"],
        "source": "corpus" if translation is None else "corpus+traducao", "repo": unit["repo"],
        "path": unit["path"], "licenses": unit["licenses"],
        "prompt": rng.choice(instructions[unit["task"]][unit["nl"]]), "code": code_in, "response": full,
        "kind": "code", "lang_in_prompt": rng.random() < 0.5,
    }


def load_translations(paths: list[Path]) -> dict[str, str]:
    translations = {}
    for path in paths:
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                translations[row["id"]] = row["translation"]
    return translations


def usable_translation(unit: dict, translation: str | None) -> str | None:
    """A tradução, se ela partiu da doc certa e traduziu de verdade.

    Doc diferente de ``clean_doc`` veio da versão que apagava linhas de código.
    Tradução idêntica ao original é doc só de tags (``@param {T} x``), sem texto
    no idioma da célula.
    """
    if translation is None or clean_doc(unit["doc_raw"], unit["lang"]) != unit["doc"]:
        return None
    if placeholder(unit["doc"], translation) or translation.strip() == unit["doc"].strip():
        return None
    return translation


def repair(translation_paths: list[Path]) -> None:
    """Grava os textos a traduzir de novo: docs corrigidas, falhas e respostas vazias."""
    translations = load_translations(translation_paths)
    jobs = []
    for unit in pq.read_table(WORK_DIR / "selected.parquet").to_pylist():
        translation = translations.get(unit["id"])
        doc = clean_doc(unit["doc_raw"], unit["lang"])
        unchanged = translation is not None and translation.strip() == unit["doc"].strip()
        if unit["doc_nl"] != unit["nl"] and not usable_translation(unit, translation) and (doc != unit["doc"] or not unchanged):
            jobs.append({"id": unit["id"], "text": doc, "target": unit["nl"], "kind": "docstring"})
    with (WORK_DIR / "translate_fix_in.jsonl").open("w") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")
    print(f"{len(jobs):,} docs a traduzir de novo em {WORK_DIR / 'translate_fix_in.jsonl'}")


def build(translation_paths: list[Path], fix_paths: list[Path]) -> None:
    rng = random.Random(SEED)
    instructions = json.loads(INSTRUCTIONS.read_text())
    translations = load_translations(translation_paths)
    fixes = load_translations(fix_paths)
    tokenizer = load_tokenizer()
    units = pq.read_table(WORK_DIR / "selected.parquet").to_pylist()
    examples, stats = [], Counter()
    for unit in units:
        needs = unit["doc_nl"] != unit["nl"]
        translation = None
        if needs:
            translation = fixes.get(unit["id"]) or usable_translation(unit, translations.get(unit["id"]))
            if translation is None or placeholder(unit["doc"], translation):
                stats["sem tradução utilizável"] += 1
                continue
        example = build_example(unit, translation, instructions, rng)
        if example is None:
            stats["descartado (não compila ou formato)"] += 1
            continue
        ids, _ = encode_example(tokenizer, example["prompt"], example["response"], example["lang"],
                                example["kind"], code=example["code"], lang_in_prompt=example["lang_in_prompt"])
        example["n_tokens"] = len(ids)
        examples.append({column: example[column] for column in BANK_COLUMNS})
        stats["ok"] += 1
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(examples), BANK_DIR / "t2_t5.parquet")
    print(", ".join(f"{k}: {v:,}" for k, v in stats.items()))
    print(f"{len(examples):,} exemplos em {BANK_DIR / 't2_t5.parquet'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("select")
    build_cmd = commands.add_parser("build")
    build_cmd.add_argument("--translations", type=Path, nargs="*", default=[])
    build_cmd.add_argument("--fixes", type=Path, nargs="*", default=[], help="saída da tradução de translate_fix_in.jsonl")
    repair_cmd = commands.add_parser("repair")
    repair_cmd.add_argument("--translations", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    if args.command == "select":
        select()
    elif args.command == "repair":
        repair(args.translations)
    else:
        build(args.translations, args.fixes)


if __name__ == "__main__":
    main()
