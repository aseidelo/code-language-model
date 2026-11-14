"""T6 de sintaxe: estraga a sintaxe de funções do corpus v2 e pede a correção.

Desenho em docs/sft-v2.md ("T6 de sintaxe"). Resumo:

* Linguagens: C, TypeScript, Rust, JavaScript, Java e Bash. Python já tem o T6 semântico.
* Fonte: ``v2/units/<lang>.parquet``, funções fáceis e médias com 16–600 tokens, cada uma usada uma vez.
* Mutações por token do tree-sitter, preservando o resto do texto. A original precisa parsear limpa e o
  mutante precisa ter ``ERROR``/``MISSING`` (Bash: também ``bash -n``); senão a mutação é descartada.
* 1 erro em ~80% dos exemplos, 2–3 nos demais; ~5% são negativos (código certo, resposta em texto).
* Variantes de pedido nos 6 idiomas: a) só a instrução; b) com a linha do erro (Bash: mensagem do
  ``bash -n``); c) com a descrição, quando a função tem docstring.
* Resposta: o código original completo.

Saída: ``v2/fontes/t6s-corpus-v2/part-<lang>-NNNN.parquet`` + ``meta.json``.

    uv run python -m dataprep.sft_syntax [linguagens] [--por-linguagem 400000] [--workers 8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from random import Random

import pyarrow as pa
import pyarrow.parquet as pq
import tree_sitter_language_pack as tslp

from dataprep.sft_bugfix import FONTES_DIR, T6_SCHEMA
from dataprep.sft_candidates import UNITS_V2_DIR
from dataprep.sft_convert import estimate_tokens

LANGS = ("c", "typescript", "rust", "javascript", "java", "bash")
DISPLAY = {"c": "C", "typescript": "TypeScript", "rust": "Rust", "javascript": "JavaScript", "java": "Java",
           "bash": "Bash"}
FONTE = "t6s-corpus-v2"
OUT_DIR = FONTES_DIR / FONTE
PER_PART = 50_000
MIN_TOKENS, MAX_TOKENS = 16, 600
NEGATIVE_RATE = 0.05
MULTI_RATE = 0.20
DOC_SHARE = 0.25  # fração máxima de funções documentadas (as outras já viraram T2/T5)
NLS = ("en", "pt", "es", "fr", "de", "it")

SCHEMA = T6_SCHEMA.append(pa.field("tipo_bug", pa.string()))

# --- mutações ------------------------------------------------------------------------

CLOSE = {"}", ")", "]"}
OPEN = {"{", "(", "["}
QUOTES = {'"', "'", "`"}
KEYWORDS = {"return", "function", "else", "while", "for", "if", "public", "private", "static", "const", "let",
            "struct", "impl", "match", "void", "class", "switch", "case", "break", "continue", "typedef", "async",
            "await", "export", "import", "interface", "new", "throw", "try", "catch", "pub", "mut", "unsigned"}
OPERATOR_SWAPS = {"rust": {"->": ["=>"], "=>": ["->"], "::": [":"]},
                  "javascript": {"=>": ["="]}, "typescript": {"=>": ["="]}}
BASH_TOKENS = {"fi", "done", "then", "esac", ";;", "do", "]]", "in"}
SEMICOLON_LANGS = {"c", "java", "rust", "javascript", "typescript"}


def leaves(node):
    stack = [node]
    while stack:
        n = stack.pop()
        if n.child_count == 0:
            yield n
        else:
            stack.extend(reversed(n.children))


def typo(word: str, r: Random) -> str:
    i = r.randrange(len(word) - 1)
    if r.random() < 0.5:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]  # troca duas letras
    return word[:i] + word[i + 1:]  # apaga uma letra


def candidate_edits(lang: str, root, source: bytes, r: Random) -> list[tuple[str, int, int, bytes]]:
    """(operador, início, fim, substituto) sobre os bytes da original."""
    edits = []
    for leaf in leaves(root):
        kind, start, end = leaf.type, leaf.start_byte, leaf.end_byte
        if leaf.is_named and kind not in ("string_content",):
            continue
        text = source[start:end].decode(errors="replace")
        if lang == "bash":
            if kind in BASH_TOKENS:
                edits.append((f"apagar_{kind}", start, end, b""))
            elif kind in CLOSE:
                edits.append(("apagar_fechamento", start, end, b""))
            elif kind in QUOTES and leaf.parent is not None and leaf.parent.children[-1] == leaf:
                edits.append(("string_aberta", start, end, b""))
            continue
        if kind in CLOSE:
            edits.append(("apagar_fechamento", start, end, b""))
            edits.append(("fechamento_sobrando", end, end, kind.encode()))
        elif kind in OPEN:
            edits.append(("apagar_abertura", start, end, b""))
        elif kind == ";" and lang in SEMICOLON_LANGS:
            edits.append(("apagar_ponto_e_virgula", start, end, b""))
        elif kind == ",":
            edits.append(("apagar_virgula", start, end, b""))
        elif kind in QUOTES and leaf.parent is not None and leaf.parent.children[-1] == leaf:
            edits.append(("string_aberta", start, end, b""))
        elif kind in KEYWORDS and text == kind and len(kind) >= 3:
            edits.append(("palavra_chave", start, end, typo(kind, r).encode()))
        elif kind in OPERATOR_SWAPS.get(lang, {}):
            edits.append(("operador", start, end, r.choice(OPERATOR_SWAPS[lang][kind]).encode()))
    return edits


def apply(source: bytes, edits: list[tuple[str, int, int, bytes]]) -> bytes:
    for _, start, end, new in sorted(edits, key=lambda e: e[1], reverse=True):
        source = source[:start] + new + source[end:]
    return source


# --- verificação --------------------------------------------------------------------------

WRAP = {"java": ("class _Q {\n", "\n}"), "javascript": ("class _Q {\n", "\n}"), "typescript": ("class _Q {\n", "\n}")}


@lru_cache(maxsize=None)
def parser(lang: str):
    return tslp.get_parser(lang)


def parse(lang: str, code: bytes):
    """(árvore, deslocamento de linhas, deslocamento de bytes); métodos soltos vão dentro de uma classe."""
    tree = parser(lang).parse(code)
    if not tree.root_node.has_error or lang not in WRAP:
        return tree, 0, 0
    head, tail = (s.encode() for s in WRAP[lang])
    wrapped = parser(lang).parse(head + code + tail)
    return (wrapped, 1, len(head)) if not wrapped.root_node.has_error else (tree, 0, 0)


def first_error_line(node, line_shift: int) -> int | None:
    stack = [node]
    while stack:
        n = stack.pop()
        if n.is_error or n.is_missing:
            return n.start_point[0] - line_shift + 1
        if n.has_error:
            stack.extend(reversed(n.children))
    return None


def mutant_error(lang: str, code: bytes) -> tuple[bool, int | None, str | None]:
    """(tem erro?, linha do primeiro erro, mensagem do bash -n)."""
    if lang in WRAP:
        head, tail = (s.encode() for s in WRAP[lang])
        plain, wrapped = parser(lang).parse(code), parser(lang).parse(head + code + tail)
        if not (plain.root_node.has_error and wrapped.root_node.has_error):
            return False, None, None
        return True, first_error_line(wrapped.root_node, 1), None
    tree = parser(lang).parse(code)
    if not tree.root_node.has_error:
        return False, None, None
    line, message = first_error_line(tree.root_node, 0), None
    if lang == "bash":
        result = subprocess.run(["bash", "-n"], input=code, capture_output=True, timeout=5,
                                env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"})  # mensagem em inglês
        if result.returncode == 0:
            return False, None, None
        message = result.stderr.decode(errors="replace").strip().splitlines()[0].replace("bash: ", "", 1)
    return True, line, message


# --- pedidos ---------------------------------------------------------------------------------

# {linguagem} é preenchido. NEUTRAL serve para 1 ou mais erros; SINGLE só para 1.
NEUTRAL = {
    "en": ["This {linguagem} code doesn't compile. Fix it.", "Fix the syntax of this {linguagem} code.",
           "The {linguagem} code below has syntax problems. Correct it.",
           "Make this {linguagem} code syntactically valid."],
    "pt": ["Este código {linguagem} não compila. Corrija.", "Corrija a sintaxe deste código {linguagem}.",
           "O código {linguagem} abaixo tem problemas de sintaxe. Conserte.",
           "Deixe este código {linguagem} sintaticamente válido."],
    "es": ["Este código {linguagem} no compila. Corrígelo.", "Corrige la sintaxis de este código {linguagem}.",
           "El código {linguagem} de abajo tiene problemas de sintaxis. Arréglalo.",
           "Haz que este código {linguagem} sea sintácticamente válido."],
    "fr": ["Ce code {linguagem} ne compile pas. Corrige-le.", "Corrige la syntaxe de ce code {linguagem}.",
           "Le code {linguagem} ci-dessous a des problèmes de syntaxe. Répare-le.",
           "Rends ce code {linguagem} syntaxiquement valide."],
    "de": ["Dieser {linguagem}-Code kompiliert nicht. Behebe das.", "Korrigiere die Syntax dieses {linguagem}-Codes.",
           "Der folgende {linguagem}-Code hat Syntaxprobleme. Repariere ihn.",
           "Mach diesen {linguagem}-Code syntaktisch korrekt."],
    "it": ["Questo codice {linguagem} non compila. Correggilo.", "Correggi la sintassi di questo codice {linguagem}.",
           "Il codice {linguagem} qui sotto ha problemi di sintassi. Sistemalo.",
           "Rendi questo codice {linguagem} sintatticamente valido."],
}
SINGLE = {
    "en": ["Fix the syntax error in this {linguagem} code.", "There is a syntax error in this {linguagem} function. Fix it."],
    "pt": ["Corrija o erro de sintaxe neste código {linguagem}.", "Há um erro de sintaxe nesta função {linguagem}. Corrija."],
    "es": ["Corrige el error de sintaxis en este código {linguagem}.",
           "Hay un error de sintaxis en esta función {linguagem}. Corrígelo."],
    "fr": ["Corrige l'erreur de syntaxe dans ce code {linguagem}.",
           "Il y a une erreur de syntaxe dans cette fonction {linguagem}. Corrige-la."],
    "de": ["Behebe den Syntaxfehler in diesem {linguagem}-Code.",
           "In dieser {linguagem}-Funktion steckt ein Syntaxfehler. Behebe ihn."],
    "it": ["Correggi l'errore di sintassi in questo codice {linguagem}.",
           "C'è un errore di sintassi in questa funzione {linguagem}. Correggilo."],
}
MULTI = {
    "en": ["Correct the syntax errors in this {linguagem} code.", "This {linguagem} code has a few syntax errors. Fix them."],
    "pt": ["Corrija os erros de sintaxe deste código {linguagem}.", "Este código {linguagem} tem alguns erros de sintaxe. Corrija."],
    "es": ["Corrige los errores de sintaxis de este código {linguagem}.",
           "Este código {linguagem} tiene algunos errores de sintaxis. Corrígelos."],
    "fr": ["Corrige les erreurs de syntaxe de ce code {linguagem}.",
           "Ce code {linguagem} contient quelques erreurs de syntaxe. Corrige-les."],
    "de": ["Behebe die Syntaxfehler in diesem {linguagem}-Code.", "Dieser {linguagem}-Code hat mehrere Syntaxfehler. Behebe sie."],
    "it": ["Correggi gli errori di sintassi in questo codice {linguagem}.",
           "Questo codice {linguagem} ha alcuni errori di sintassi. Correggili."],
}
# Negativos usam pedidos que não afirmam que há erro.
CHECK = {
    "en": ["Does this {linguagem} code have a syntax error? If so, fix it.",
           "Check the syntax of this {linguagem} code and fix it if needed."],
    "pt": ["Este código {linguagem} tem erro de sintaxe? Se tiver, corrija.",
           "Verifique a sintaxe deste código {linguagem} e corrija se precisar."],
    "es": ["¿Este código {linguagem} tiene un error de sintaxis? Si lo tiene, corrígelo.",
           "Revisa la sintaxis de este código {linguagem} y corrígela si hace falta."],
    "fr": ["Ce code {linguagem} a-t-il une erreur de syntaxe ? Si oui, corrige-la.",
           "Vérifie la syntaxe de ce code {linguagem} et corrige-la si nécessaire."],
    "de": ["Hat dieser {linguagem}-Code einen Syntaxfehler? Falls ja, behebe ihn.",
           "Prüfe die Syntax dieses {linguagem}-Codes und korrigiere sie, falls nötig."],
    "it": ["Questo codice {linguagem} ha un errore di sintassi? Se sì, correggilo.",
           "Controlla la sintassi di questo codice {linguagem} e correggila se serve."],
}
NO_ERROR = {"en": "This code has no syntax error; it is valid {linguagem} as it is.",
            "pt": "Este código não tem erro de sintaxe; já é {linguagem} válido.",
            "es": "Este código no tiene errores de sintaxis; ya es {linguagem} válido.",
            "fr": "Ce code n'a pas d'erreur de syntaxe ; c'est déjà du {linguagem} valide.",
            "de": "Dieser Code hat keinen Syntaxfehler; er ist bereits gültiges {linguagem}.",
            "it": "Questo codice non ha errori di sintassi; è già {linguagem} valido."}
LINE_HINT = {"en": "The parser reports an error near line {linha}.", "pt": "O parser acusa erro perto da linha {linha}.",
             "es": "El parser marca un error cerca de la línea {linha}.",
             "fr": "Le parseur signale une erreur vers la ligne {linha}.",
             "de": "Der Parser meldet einen Fehler in der Nähe von Zeile {linha}.",
             "it": "Il parser segnala un errore vicino alla riga {linha}."}
BASH_HINT = {"en": "`bash -n` says: {msg}", "pt": "O `bash -n` diz: {msg}", "es": "`bash -n` dice: {msg}",
             "fr": "`bash -n` indique : {msg}", "de": "`bash -n` meldet: {msg}", "it": "`bash -n` dice: {msg}"}
DESCRIPTION = {"en": "What the function should do:\n{descricao}", "pt": "O que a função deveria fazer:\n{descricao}",
               "es": "Lo que debería hacer la función:\n{descricao}", "fr": "Ce que la fonction devrait faire :\n{descricao}",
               "de": "Was die Funktion tun soll:\n{descricao}", "it": "Cosa dovrebbe fare la funzione:\n{descricao}"}


def build_prompt(r: Random, lang: str, nl: str, variant: str, n_errors: int, line: int | None,
                 bash_msg: str | None, description: str | None) -> str:
    pool = NEUTRAL[nl] + (SINGLE[nl] if n_errors == 1 else MULTI[nl])
    parts = [r.choice(pool).format(linguagem=DISPLAY[lang])]
    if variant == "b":
        parts.append(BASH_HINT[nl].format(msg=bash_msg) if bash_msg else LINE_HINT[nl].format(linha=line))
    elif variant == "c":
        parts.append(DESCRIPTION[nl].format(descricao=description.strip()))
    return "\n\n".join(parts)


# --- montagem ---------------------------------------------------------------------------------

def make_example(unit: dict, lang: str, r: Random, stats: Counter) -> dict | None:
    code = unit["code"]
    source = code.encode()
    tree, _, _ = parse(lang, source)
    if tree.root_node.has_error:
        stats["original_invalida"] += 1
        return None
    description = unit["doc"] if unit.get("doc") and unit.get("nodoc") else None
    nl = r.choice(NLS)
    digest = unit["digest"]
    base = {"task": "T6", "level": unit["level"], "lang": lang, "nl": nl, "source": "corpus-v2",
            "repo": unit["repo"], "path": unit["path"],
            "licenses": unit["licenses"] if isinstance(unit["licenses"], (str, type(None)))
            else json.dumps(unit["licenses"]), "lang_in_prompt": True,
            "fonte": FONTE, "licenca": "por repositório", "tests": None, "test_setup": None,
            "origem": digest, "tipo_bug": "sintaxe"}
    if r.random() < NEGATIVE_RATE:
        variant = "c" if description and r.random() < 0.3 else "a"
        prompt = r.choice(CHECK[nl]).format(linguagem=DISPLAY[lang])
        if variant == "c":
            prompt += "\n\n" + DESCRIPTION[nl].format(descricao=description.strip())
        response = NO_ERROR[nl].format(linguagem=DISPLAY[lang])
        stats["negativos"] += 1
        return {**base, "id": f"T6s-{lang}-{digest[:16]}", "prompt": prompt, "code": code, "response": response,
                "kind": "description", "mutacao": "nenhuma", "variante": variant,
                "n_tokens": estimate_tokens(prompt, None) + estimate_tokens(code, lang) + estimate_tokens(response, None)}
    edits = candidate_edits(lang, tree.root_node, source, r)
    if not edits:
        stats["sem_candidatos"] += 1
        return None
    # Sorteia primeiro o operador e depois a posição: senão os delimitadores (os tokens mais comuns) dominam.
    by_op: dict[str, list] = {}
    for edit in edits:
        by_op.setdefault(edit[0], []).append(edit)
    ops = list(by_op)
    r.shuffle(ops)
    for op in ops:
        r.shuffle(by_op[op])
    ordered = [by_op[op][i] for i in range(max(map(len, by_op.values()))) for op in ops if i < len(by_op[op])]
    n_errors = 1 if r.random() >= MULTI_RATE else r.choice((2, 3))
    chosen: list = []
    for edit in ordered[:40]:
        if len(chosen) == n_errors:
            break
        if any(not (edit[2] < c[1] or edit[1] > c[2]) for c in chosen):  # não sobrepor
            continue
        trial = apply(source, chosen + [edit])
        ok, _, _ = mutant_error(lang, trial)
        if ok:
            chosen.append(edit)
        else:
            stats[f"aceito_pelo_parser:{edit[0]}"] += 1
    if not chosen:
        stats["sem_mutante"] += 1
        return None
    mutant = apply(source, chosen)
    ok, line, bash_msg = mutant_error(lang, mutant)
    if not ok:
        stats["sem_mutante"] += 1
        return None
    variants = ["a"] + (["b"] if line or bash_msg else []) + (["c"] if description else [])
    variant = r.choice(variants)
    prompt = build_prompt(r, lang, nl, variant, len(chosen), line, bash_msg, description)
    mutant_code = mutant.decode(errors="replace")
    for op, *_ in chosen:
        stats[f"mutacao:{op}"] += 1
    stats[f"erros:{len(chosen)}"] += 1
    return {**base, "id": f"T6s-{lang}-{hashlib.sha1(mutant).hexdigest()[:16]}", "prompt": prompt,
            "code": mutant_code, "response": code, "kind": "code", "mutacao": "+".join(op for op, *_ in chosen),
            "variante": variant,
            "n_tokens": estimate_tokens(prompt, None) + 2 * estimate_tokens(code, lang)}


def process_group(task: tuple[str, int, int]) -> tuple[list[dict], Counter]:
    lang, group, seed = task
    r = Random(f"{seed}-{lang}-{group}")
    stats: Counter = Counter()
    table = pq.ParquetFile(UNITS_V2_DIR / f"{lang}.parquet").read_row_group(group)
    units = [u for u in table.to_pylist()
             if u["level"] in ("facil", "medio") and MIN_TOKENS <= (u["tokens"] or 0) <= MAX_TOKENS]
    seen, out = set(), []
    documented = [u for u in units if u.get("doc") and u.get("nodoc")]
    plain = [u for u in units if not (u.get("doc") and u.get("nodoc"))]
    # Até DOC_SHARE das funções vêm das documentadas, que já aparecem nos T2/T5.
    k = min(len(documented), int(len(plain) * DOC_SHARE / (1 - DOC_SHARE)))
    units = plain + r.sample(documented, k)
    r.shuffle(units)
    for unit in units:
        if unit["digest"] in seen:
            continue
        seen.add(unit["digest"])
        stats["funcoes"] += 1
        example = make_example(unit, lang, r, stats)
        if example:
            out.append(example)
    return out, stats


def build(lang: str, target: int, workers: int, seed: int = 0) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done_parts = sorted(OUT_DIR.glob(f"part-{lang}-*.parquet"))
    if done_parts:
        print(f"{lang}: já gerado ({len(done_parts)} partes), pulando")
        return
    groups = list(range(pq.ParquetFile(UNITS_V2_DIR / f"{lang}.parquet").num_row_groups))
    Random(f"{seed}-{lang}").shuffle(groups)
    rows, stats, seen, start = [], Counter(), set(), time.time()
    with ProcessPoolExecutor(workers) as pool:
        for new, s in pool.map(process_group, [(lang, g, seed) for g in groups]):
            stats.update(s)
            for row in new:  # a mesma função pode estar em mais de um grupo
                if row["origem"] not in seen:
                    seen.add(row["origem"])
                    rows.append(row)
            print(f"  {lang}: {len(rows):,} exemplos | {time.time() - start:.0f}s", flush=True)
            if len(rows) >= target:
                break
    rows = rows[:target]
    for i in range(0, len(rows), PER_PART):
        out = OUT_DIR / f"part-{lang}-{i // PER_PART:04d}.parquet"
        tmp = out.with_suffix(".tmp")
        pq.write_table(pa.Table.from_pylist(rows[i:i + PER_PART], schema=SCHEMA), tmp, compression="zstd")
        tmp.rename(out)
    stats_path = OUT_DIR / "execucao.json"
    all_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    all_stats[lang] = dict(sorted(stats.items()))
    stats_path.write_text(json.dumps(all_stats, indent=1, ensure_ascii=False))
    print(json.dumps({"lang": lang, "exemplos": len(rows), **{k: v for k, v in stats.items() if ":" not in k}},
                     ensure_ascii=False))


def save_meta() -> None:
    cells: Counter = Counter()
    tokens: Counter = Counter()
    for part in sorted(OUT_DIR.glob("part-*.parquet")):
        for r in pq.read_table(part, columns=["lang", "nl", "variante", "n_tokens"]).to_pylist():
            key = f"T6|{r['lang']}|{r['nl']}|{r['variante']}"
            cells[key] += 1
            tokens[key] += r["n_tokens"]
    meta = {"fonte": FONTE, "exemplos": sum(cells.values()), "tokens_estimados": sum(tokens.values()),
            "celulas": {k: {"exemplos": cells[k], "tokens": tokens[k]} for k in sorted(cells)},
            "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))


def main() -> None:
    parser_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser_.add_argument("langs", nargs="*", default=list(LANGS))
    parser_.add_argument("--por-linguagem", type=int, default=400_000)
    parser_.add_argument("--workers", type=int, default=8)
    parser_.add_argument("--amostra", type=int, help="só imprime N exemplos de um grupo de cada linguagem")
    args = parser_.parse_args()
    for lang in args.langs:
        if lang not in LANGS:
            raise SystemExit(f"linguagem desconhecida: {lang} (opções: {', '.join(LANGS)})")
    if args.amostra:
        for lang in args.langs:
            rows, stats = process_group((lang, 0, 0))
            print(f"### {lang}: {len(rows)} exemplos de {stats['funcoes']} funções; {dict(stats.most_common(12))}")
            for row in Random(1).sample(rows, min(args.amostra, len(rows))):
                print(f"--- {row['mutacao']} / {row['variante']} / {row['nl']}\n{row['prompt']}\n"
                      f"[código]\n{row['code'][:600]}\n[resposta]\n{row['response'][:300]}\n")
        return
    for lang in args.langs:
        build(lang, args.por_linguagem, args.workers)
    save_meta()


if __name__ == "__main__":
    main()
