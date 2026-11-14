"""Tarefas procedurais de JSON/YAML a partir dos arquivos do The Stack (os mesmos do pré-treino v2).

Desenho em docs/sft-v2.md ("JSON/YAML procedural"). Resumo:

* Sementes: ``the-stack-dedup/filtered-v2/{json,yaml}``, documentos de 16–600 tokens que parseiam
  (YAML: um documento só, sem tags desconhecidas), objeto ou lista na raiz, sem duplicatas.
* Tarefas: converter 25%, corrigir 25% (T6), formatar 10% (só JSON), extrair 15%, editar 25% (T7).
* JSON usa cada semente em todas as tarefas (só há ~66k); o YAML completa o total (~800k).
* Converter só entra com round-trip exato; editar só entra quando o documento é reescrito idêntico
  (JSON: ``json.dumps`` com indentação 2/4; YAML: ``ruamel.yaml``, que preserva a formatação).
* Corrigir: corrupções no texto; o mutante precisa falhar no parser; 20% com 2–3 erros; 5% negativos.
* Idiomas: en 53%, pt/es/fr/de/it ~9,4% cada.

Saída: ``v2/fontes/jy-stack/part-<formato>-NNNN.parquet`` + ``meta.json``.

    uv run python -m dataprep.sft_jsonyaml sementes [--workers 8]
    uv run python -m dataprep.sft_jsonyaml gerar [--total 800000] [--workers 8]
    uv run python -m dataprep.sft_jsonyaml amostra [--n 2]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from random import Random

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from dataprep.sft_bugfix import FONTES_DIR
from dataprep.sft_convert import V2_DIR, estimate_tokens
from dataprep.sft_syntax import SCHEMA

STACK_DIR = Path(__file__).resolve().parents[2] / "data" / "the-stack-dedup" / "filtered-v2"
SEEDS_DIR = V2_DIR / "jy-sementes"
FONTE = "jy-stack"
OUT_DIR = FONTES_DIR / FONTE
FORMATS = ("json", "yaml")
DISPLAY = {"json": "JSON", "yaml": "YAML"}
PER_PART = 50_000
MIN_TOKENS, MAX_TOKENS = 16, 600
MAX_VALUE_TOKENS = 200
MAX_LINE = 1000  # o QC reprova linhas maiores
NEGATIVE_RATE = 0.05
MULTI_RATE = 0.20
NLS = ("en", "pt", "es", "fr", "de", "it")
NL_WEIGHTS = (0.53, 0.094, 0.094, 0.094, 0.094, 0.094)
TASK_WEIGHTS = {"json": {"converter": 25, "corrigir": 25, "formatar": 10, "extrair": 15, "editar": 25},
                "yaml": {"converter": 25, "corrigir": 25, "extrair": 15, "editar": 25}}
KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")

try:
    LOADER = yaml.CSafeLoader
except AttributeError:  # pragma: no cover
    LOADER = yaml.SafeLoader


# --- sementes ------------------------------------------------------------------------------

def parse(text: str, fmt: str):
    if fmt == "json":
        return json.loads(text)
    docs = list(yaml.load_all(text, Loader=LOADER))
    if len(docs) != 1:
        raise ValueError("multidocumento")
    return docs[0]


def seed_shard(task: tuple[str, str]) -> tuple[list[dict], Counter]:
    fmt, path = task
    stats: Counter = Counter()
    table = pq.read_table(path, columns=["content", "repo", "path", "licenses", "sha256"])
    out = []
    for row in table.to_pylist():
        stats["lidos"] += 1
        text = row["content"].replace("\r\n", "\n")
        tokens = estimate_tokens(text, fmt)
        if not MIN_TOKENS <= tokens <= MAX_TOKENS:
            stats["tamanho"] += 1
            continue
        if max(map(len, text.splitlines()), default=0) > MAX_LINE:
            stats["linha_longa"] += 1
            continue
        try:
            obj = parse(text, fmt)
        except Exception:
            stats["nao_parseia"] += 1
            continue
        if not isinstance(obj, (dict, list)) or not obj:
            stats["raiz_escalar_ou_vazia"] += 1
            continue
        stats["ok"] += 1
        out.append({"content": text if text.endswith("\n") else text + "\n", "repo": row["repo"],
                    "path": row["path"], "licenses": json.dumps(row["licenses"] or []),
                    "digest": row["sha256"], "tokens": tokens})
    return out, stats


def build_seeds(workers: int) -> None:
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in FORMATS:
        shards = sorted((STACK_DIR / fmt).glob("train-*.parquet"))
        rows, stats, seen = [], Counter(), set()
        with ProcessPoolExecutor(workers) as pool:
            for new, s in pool.map(seed_shard, [(fmt, str(p)) for p in shards]):
                stats.update(s)
                for row in new:
                    if row["digest"] not in seen:
                        seen.add(row["digest"])
                        rows.append(row)
        stats["duplicados"] = stats["ok"] - len(rows)
        pq.write_table(pa.Table.from_pylist(rows), SEEDS_DIR / f"{fmt}.parquet", compression="zstd",
                       row_group_size=5_000)
        print(json.dumps({"formato": fmt, "sementes": len(rows), **stats}, ensure_ascii=False), flush=True)


# --- pedidos -------------------------------------------------------------------------------------

CONVERT = {
    "en": ["Convert this {origem} to {destino}.", "Rewrite the following {origem} as {destino}.",
           "Turn this {origem} document into equivalent {destino}.", "I need this {origem} in {destino} format."],
    "pt": ["Converta este {origem} para {destino}.", "Reescreva o {origem} abaixo em {destino}.",
           "Transforme este documento {origem} em {destino} equivalente.", "Preciso deste {origem} no formato {destino}."],
    "es": ["Convierte este {origem} a {destino}.", "Reescribe el siguiente {origem} en {destino}.",
           "Transforma este documento {origem} en {destino} equivalente.", "Necesito este {origem} en formato {destino}."],
    "fr": ["Convertis ce {origem} en {destino}.", "Réécris le {origem} suivant en {destino}.",
           "Transforme ce document {origem} en {destino} équivalent.", "J'ai besoin de ce {origem} au format {destino}."],
    "de": ["Konvertiere dieses {origem} nach {destino}.", "Schreibe das folgende {origem} als {destino} um.",
           "Wandle dieses {origem}-Dokument in gleichwertiges {destino} um.", "Ich brauche dieses {origem} im {destino}-Format."],
    "it": ["Converti questo {origem} in {destino}.", "Riscrivi il seguente {origem} in {destino}.",
           "Trasforma questo documento {origem} in {destino} equivalente.", "Mi serve questo {origem} in formato {destino}."],
}
FIX = {
    "en": ["This {fmt} is invalid. Fix it.", "Fix the syntax of this {fmt}.", "The {fmt} below doesn't parse. Correct it.",
           "Make this {fmt} valid."],
    "pt": ["Este {fmt} é inválido. Corrija.", "Corrija a sintaxe deste {fmt}.", "O {fmt} abaixo não é lido pelo parser. Conserte.",
           "Deixe este {fmt} válido."],
    "es": ["Este {fmt} no es válido. Corrígelo.", "Corrige la sintaxis de este {fmt}.",
           "El {fmt} de abajo no se puede parsear. Arréglalo.", "Haz que este {fmt} sea válido."],
    "fr": ["Ce {fmt} est invalide. Corrige-le.", "Corrige la syntaxe de ce {fmt}.",
           "Le {fmt} ci-dessous ne se parse pas. Répare-le.", "Rends ce {fmt} valide."],
    "de": ["Dieses {fmt} ist ungültig. Korrigiere es.", "Korrigiere die Syntax dieses {fmt}.",
           "Das folgende {fmt} lässt sich nicht parsen. Repariere es.", "Mach dieses {fmt} gültig."],
    "it": ["Questo {fmt} non è valido. Correggilo.", "Correggi la sintassi di questo {fmt}.",
           "Il {fmt} qui sotto non si riesce a parsare. Sistemalo.", "Rendi valido questo {fmt}."],
}
CHECK = {
    "en": ["Is this {fmt} valid? Fix it if needed.", "Check the syntax of this {fmt} and correct it if necessary."],
    "pt": ["Este {fmt} é válido? Corrija se precisar.", "Verifique a sintaxe deste {fmt} e corrija se necessário."],
    "es": ["¿Este {fmt} es válido? Corrígelo si hace falta.", "Revisa la sintaxis de este {fmt} y corrígela si es necesario."],
    "fr": ["Ce {fmt} est-il valide ? Corrige-le si besoin.", "Vérifie la syntaxe de ce {fmt} et corrige-la si nécessaire."],
    "de": ["Ist dieses {fmt} gültig? Korrigiere es bei Bedarf.", "Prüfe die Syntax dieses {fmt} und korrigiere sie, falls nötig."],
    "it": ["Questo {fmt} è valido? Correggilo se serve.", "Controlla la sintassi di questo {fmt} e correggila se necessario."],
}
NO_ERROR = {"en": "This {fmt} has no syntax error; it is already valid.",
            "pt": "Este {fmt} não tem erro de sintaxe; já é válido.",
            "es": "Este {fmt} no tiene errores de sintaxis; ya es válido.",
            "fr": "Ce {fmt} n'a pas d'erreur de syntaxe ; il est déjà valide.",
            "de": "Dieses {fmt} hat keinen Syntaxfehler; es ist bereits gültig.",
            "it": "Questo {fmt} non ha errori di sintassi; è già valido."}
PARSER_HINT = {"en": "The parser says: {msg}", "pt": "O parser diz: {msg}", "es": "El parser dice: {msg}",
               "fr": "Le parseur indique : {msg}", "de": "Der Parser meldet: {msg}", "it": "Il parser dice: {msg}"}
LINE_HINT = {"en": "The error is near line {linha}.", "pt": "O erro está perto da linha {linha}.",
             "es": "El error está cerca de la línea {linha}.", "fr": "L'erreur est vers la ligne {linha}.",
             "de": "Der Fehler liegt in der Nähe von Zeile {linha}.", "it": "L'errore è vicino alla riga {linha}."}
FORMAT = {
    "indent2": {"en": "Format this JSON with 2-space indentation.", "pt": "Formate este JSON com indentação de 2 espaços.",
                "es": "Formatea este JSON con sangría de 2 espacios.", "fr": "Formate ce JSON avec une indentation de 2 espaces.",
                "de": "Formatiere dieses JSON mit 2 Leerzeichen Einrückung.", "it": "Formatta questo JSON con indentazione di 2 spazi."},
    "indent4": {"en": "Format this JSON with 4-space indentation.", "pt": "Formate este JSON com indentação de 4 espaços.",
                "es": "Formatea este JSON con sangría de 4 espacios.", "fr": "Formate ce JSON avec une indentation de 4 espaces.",
                "de": "Formatiere dieses JSON mit 4 Leerzeichen Einrückung.", "it": "Formatta questo JSON con indentazione di 4 spazi."},
    "minify": {"en": "Minify this JSON.", "pt": "Minifique este JSON.", "es": "Minifica este JSON.", "fr": "Minifie ce JSON.",
               "de": "Minifiziere dieses JSON.", "it": "Minifica questo JSON."},
    "sorted": {"en": "Sort the keys of this JSON alphabetically (2-space indentation).",
               "pt": "Ordene as chaves deste JSON em ordem alfabética (indentação de 2 espaços).",
               "es": "Ordena las claves de este JSON alfabéticamente (sangría de 2 espacios).",
               "fr": "Trie les clés de ce JSON par ordre alphabétique (indentation de 2 espaces).",
               "de": "Sortiere die Schlüssel dieses JSON alphabetisch (2 Leerzeichen Einrückung).",
               "it": "Ordina alfabeticamente le chiavi di questo JSON (indentazione di 2 spazi)."},
}
EXTRACT = {
    "en": ["What is the value of `{caminho}` in this {fmt}?", "Extract `{caminho}` from this {fmt}.",
           "Give me the value at `{caminho}`."],
    "pt": ["Qual é o valor de `{caminho}` neste {fmt}?", "Extraia `{caminho}` deste {fmt}.", "Me dê o valor em `{caminho}`."],
    "es": ["¿Cuál es el valor de `{caminho}` en este {fmt}?", "Extrae `{caminho}` de este {fmt}.",
           "Dame el valor en `{caminho}`."],
    "fr": ["Quelle est la valeur de `{caminho}` dans ce {fmt} ?", "Extrais `{caminho}` de ce {fmt}.",
           "Donne-moi la valeur à `{caminho}`."],
    "de": ["Welchen Wert hat `{caminho}` in diesem {fmt}?", "Extrahiere `{caminho}` aus diesem {fmt}.",
           "Gib mir den Wert unter `{caminho}`."],
    "it": ["Qual è il valore di `{caminho}` in questo {fmt}?", "Estrai `{caminho}` da questo {fmt}.",
           "Dammi il valore in `{caminho}`."],
}
ROOT = {"en": "the root", "pt": "a raiz", "es": "la raíz", "fr": "la racine", "de": "die Wurzel", "it": "la radice"}
EDIT = {
    "renomear": {"en": "In this {fmt}, rename the key `{caminho}` to `{nova}`.",
                 "pt": "Neste {fmt}, renomeie a chave `{caminho}` para `{nova}`.",
                 "es": "En este {fmt}, renombra la clave `{caminho}` a `{nova}`.",
                 "fr": "Dans ce {fmt}, renomme la clé `{caminho}` en `{nova}`.",
                 "de": "Benenne in diesem {fmt} den Schlüssel `{caminho}` in `{nova}` um.",
                 "it": "In questo {fmt}, rinomina la chiave `{caminho}` in `{nova}`."},
    "remover": {"en": "Remove `{caminho}` from this {fmt}.", "pt": "Remova `{caminho}` deste {fmt}.",
                "es": "Elimina `{caminho}` de este {fmt}.", "fr": "Supprime `{caminho}` de ce {fmt}.",
                "de": "Entferne `{caminho}` aus diesem {fmt}.", "it": "Rimuovi `{caminho}` da questo {fmt}."},
    "mudar_valor": {"en": "Change `{caminho}` to `{valor}` in this {fmt}.", "pt": "Mude `{caminho}` para `{valor}` neste {fmt}.",
                    "es": "Cambia `{caminho}` a `{valor}` en este {fmt}.", "fr": "Change `{caminho}` en `{valor}` dans ce {fmt}.",
                    "de": "Ändere `{caminho}` in diesem {fmt} auf `{valor}`.", "it": "Cambia `{caminho}` in `{valor}` in questo {fmt}."},
    "adicionar": {"en": "Add the key `{nova}` with value `{valor}` to {onde} of this {fmt}.",
                  "pt": "Adicione a chave `{nova}` com valor `{valor}` em {onde} deste {fmt}.",
                  "es": "Agrega la clave `{nova}` con valor `{valor}` en {onde} de este {fmt}.",
                  "fr": "Ajoute la clé `{nova}` avec la valeur `{valor}` dans {onde} de ce {fmt}.",
                  "de": "Füge den Schlüssel `{nova}` mit dem Wert `{valor}` zu {onde} dieses {fmt} hinzu.",
                  "it": "Aggiungi la chiave `{nova}` con valore `{valor}` in {onde} di questo {fmt}."},
}
NEW_KEYS = [("enabled", True), ("debug", False), ("timeout", 30), ("retries", 3), ("version", "1.0.0"),
            ("priority", 1), ("region", "us-east-1"), ("owner", "admin"), ("cache", True), ("max_items", 100)]
COMMENTS = ["// TODO: check this", "// default settings", "// see docs", "// updated", "// FIXME"]


# --- utilidades de árvore --------------------------------------------------------------------------

def path_text(path: tuple) -> str:
    out = ""
    for p in path:
        out += f"[{p}]" if isinstance(p, int) else (p if not out else f".{p}")
    return out


def nodes(obj, path=(), depth=0):
    """(caminho, valor) de todos os nós com caminho expressável (chaves simples), menos a raiz."""
    if depth > 8:
        return
    items = obj.items() if isinstance(obj, dict) else enumerate(obj) if isinstance(obj, list) else ()
    for k, v in items:
        if isinstance(k, str) and not KEY.match(k):
            continue
        if not isinstance(k, (str, int)) or isinstance(k, bool):
            continue
        p = path + (k,)
        yield p, v
        yield from nodes(v, p, depth + 1)


def get(obj, path):
    for p in path:
        obj = obj[p]
    return obj


def dump_value(value, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    text = yaml.safe_dump(value, allow_unicode=True, sort_keys=False, default_flow_style=False, width=10**6)
    return text[:-4] if text.endswith("\n...\n") else text


def literal(value, fmt: str) -> str:
    if fmt == "json" or not isinstance(value, str):
        return json.dumps(value, ensure_ascii=False) if fmt == "json" else dump_value(value, "yaml").strip()
    return value


def snake_camel(key: str) -> str | None:
    if "_" in key and key.strip("_"):
        head, *rest = key.split("_")
        new = head + "".join(w[:1].upper() + w[1:] for w in rest if w)
    elif re.search(r"[a-z][A-Z]", key):
        new = re.sub(r"(?<=[a-z0-9])([A-Z])", lambda m: "_" + m.group(1).lower(), key)
    else:
        return None
    return new if new != key and KEY.match(new) else None


def rename_key(m, old, new) -> None:
    if isinstance(m, CommentedMap):
        pos = list(m).index(old)
        value = m.pop(old)
        m.insert(pos, new, value)
    else:
        items = list(m.items())
        m.clear()
        m.update((new if k == old else k, v) for k, v in items)


def ruamel_dumper(mapping: int, sequence: int, offset: int) -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 10**6
    y.indent(mapping=mapping, sequence=sequence, offset=offset)
    return y


RUAMEL_STYLES = ((2, 2, 0), (2, 4, 2), (4, 4, 2), (4, 2, 0))


def ruamel_dump(y: YAML, data) -> str:
    buf = io.StringIO()
    y.dump(data, buf)
    return buf.getvalue()


# --- tarefas ---------------------------------------------------------------------------------------

def task_convert(text: str, obj, fmt: str, nl: str, r: Random, stats: Counter):
    if fmt == "json":
        out = dump_value(obj, "yaml")
        try:
            same = yaml.load(out, Loader=LOADER) == obj
        except yaml.YAMLError:
            same = False
        target = "yaml"
    else:
        try:
            out = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
        except (TypeError, ValueError):
            stats["converter:nao_serializavel"] += 1
            return None
        same = json.loads(out) == obj
        target = "json"
    if not same or estimate_tokens(out, target) > MAX_TOKENS * 1.5:
        stats["converter:round_trip"] += 1
        return None
    prompt = r.choice(CONVERT[nl]).format(origem=DISPLAY[fmt], destino=DISPLAY[target])
    return "T7", prompt, text, out, target, "code", f"converter:{fmt}->{target}", "a"


JSON_STYLES = {"indent2": dict(indent=2), "indent4": dict(indent=4), "minify": dict(separators=(",", ":")),
               "sorted": dict(indent=2, sort_keys=True)}


def json_style(obj, style: str) -> str:
    return json.dumps(obj, ensure_ascii=False, **JSON_STYLES[style]) + "\n"


def task_format(text: str, obj, fmt: str, nl: str, r: Random, stats: Counter):
    source = text
    if r.random() < 0.3:  # às vezes parte do minificado
        source = json_style(obj, "minify")
    current = next((s for s in JSON_STYLES if json_style(obj, s).strip() == source.strip()), None)
    options = [s for s in JSON_STYLES if s != current]
    style = r.choice(options)
    out = json_style(obj, style)
    if style == "sorted" and json_style(obj, "indent2") == out:
        style = r.choice([s for s in options if s != "sorted"])
        out = json_style(obj, style)
    if max(map(len, out.splitlines()), default=0) > MAX_LINE or max(map(len, source.splitlines())) > MAX_LINE:
        stats["formatar:linha_longa"] += 1
        return None
    return "T7", FORMAT[style][nl], source, out, "json", "code", f"formatar:{style}", "a"


def task_extract(text: str, obj, fmt: str, nl: str, r: Random, stats: Counter):
    candidates = [(p, v) for p, v in nodes(obj)
                  if estimate_tokens(dump_value(v, fmt), fmt) <= MAX_VALUE_TOKENS
                  and (fmt == "yaml" or isinstance(v, (dict, list)) or r.random() < 0.6)]
    if not candidates:
        stats["extrair:sem_caminho"] += 1
        return None
    deep = [c for c in candidates if len(c[0]) >= 2]
    path, value = r.choice(deep if deep and r.random() < 0.7 else candidates)
    prompt = r.choice(EXTRACT[nl]).format(caminho=path_text(path), fmt=DISPLAY[fmt])
    return "T7", prompt, text, dump_value(value, fmt), fmt, "code", "extrair", "a"


def json_layout(text: str, obj) -> dict | None:
    for kw in (dict(indent=2), dict(indent=4), dict(indent=2, ensure_ascii=True), dict(indent=4, ensure_ascii=True)):
        kw = {"ensure_ascii": False, **kw}
        if json.dumps(obj, **kw) == text.rstrip("\n"):
            return kw
    return None


def pick_edit(tree, plain, fmt: str, r: Random):
    """Sorteia a edição e aplica nas duas árvores (a que é reescrita e a ``plain`` de conferência)."""
    maps = [((), tree)] if isinstance(tree, dict) else []
    maps += [(p, v) for p, v in nodes(tree) if isinstance(v, dict)]
    maps = [(p, m) for p, m in maps if m]
    if not maps:
        return None
    op = r.choice(["renomear", "remover", "mudar_valor", "adicionar"])
    path, m = r.choice(maps)
    keys = [k for k in m if isinstance(k, str) and KEY.match(k)]
    pm = get(plain, path)
    if op == "renomear":
        options = [(k, snake_camel(k)) for k in keys]
        options = [(k, n) for k, n in options if n and n not in m]
        if not options:
            return None
        k, new = r.choice(options)
        rename_key(m, k, new)
        rename_key(pm, k, new)
        return op, {"caminho": path_text(path + (k,)), "nova": new}
    if op == "remover":
        if not keys or (not path and len(m) == 1):
            return None
        k = r.choice(keys)
        del m[k]
        del pm[k]
        return op, {"caminho": path_text(path + (k,))}
    if op == "mudar_valor":
        scalars = [k for k in keys if type(pm[k]) in (int, bool, str)]
        if not scalars:
            return None
        k = r.choice(scalars)
        old = pm[k]
        if type(old) is bool:
            new = not old
        elif type(old) is int:
            new = old + r.choice([-1, 1]) * r.randint(1, 100)
        else:
            pool = sorted({v for _, v in nodes(plain) if type(v) is str and v != old and 0 < len(v) <= 40
                           and "\n" not in v and "`" not in v})
            if not pool:
                return None
            new = r.choice(pool)
        current = m[k]
        m[k] = type(current)(new) if isinstance(current, str) and type(current) is not str else new
        pm[k] = new
        return op, {"caminho": path_text(path + (k,)), "valor": literal(new, fmt)}
    options = [(k, v) for k, v in NEW_KEYS if k not in m]
    if not options:
        return None
    k, v = r.choice(options)
    m[k] = v
    pm[k] = v
    return op, {"nova": k, "valor": literal(v, fmt), "caminho": path_text(path)}


def task_edit(text: str, obj, fmt: str, nl: str, r: Random, stats: Counter):
    plain = copy.deepcopy(obj)
    if fmt == "json":
        layout = json_layout(text, obj)
        if layout is None:
            stats["editar:formato_nao_canonico"] += 1
            return None
        tree = copy.deepcopy(obj)
        dump = lambda t: json.dumps(t, **layout) + "\n"  # noqa: E731
    else:
        y = None
        for style in RUAMEL_STYLES:
            candidate = ruamel_dumper(*style)
            try:
                tree = candidate.load(text)
                if ruamel_dump(candidate, tree) == text:
                    y = candidate
                    break
            except Exception:
                continue
        if y is None:
            stats["editar:formato_nao_preservado"] += 1
            return None
        dump = lambda t: ruamel_dump(y, t)  # noqa: E731
    try:  # ruamel (YAML 1.2) e PyYAML (1.1) podem discordar em chaves como `on`/`yes`
        picked = pick_edit(tree, plain, fmt, r)
    except (KeyError, IndexError, TypeError):
        picked = None
    if picked is None:
        stats["editar:sem_edicao"] += 1
        return None
    op, fields = picked
    try:
        out = dump(tree)
        ok = parse(out, fmt) == plain and out != text
    except Exception:
        ok = False
    if not ok:
        stats[f"editar:conferencia:{op}"] += 1
        return None
    if op == "adicionar":
        fields["onde"] = f"`{fields['caminho']}`" if fields["caminho"] else ROOT[nl]
    prompt = EDIT[op][nl].format(fmt=DISPLAY[fmt], **fields)
    return "T7", prompt, text, out, fmt, "code", f"editar:{op}", "a"


# --- corrigir (T6) ---------------------------------------------------------------------------------

JSON_TOKEN = re.compile(r'"(?:[^"\\\n]|\\.)*"|[{}\[\],:]')


def json_edits(text: str) -> list[tuple[str, int, int, str]]:
    tokens = [(m.group(), m.start(), m.end()) for m in JSON_TOKEN.finditer(text)]
    edits = []
    for i, (tok, s, e) in enumerate(tokens):
        nxt = tokens[i + 1][0] if i + 1 < len(tokens) else None
        if tok == ",":
            edits.append(("apagar_virgula", s, e, ""))
        elif tok == ":":
            edits.append(("apagar_dois_pontos", s, e, ""))
        elif tok in "}]":
            edits.append(("apagar_fechamento", s, e, ""))
            prev = text[:s].rstrip()
            if prev and prev[-1] not in "{[,":
                edits.append(("virgula_sobrando", len(prev), len(prev), ","))
        elif tok.startswith('"') and nxt == ":":
            inner = tok[1:-1]
            edits.append(("aspas_faltando", e - 1, e, ""))
            if "'" not in inner and "\\" not in inner:
                edits.append(("aspas_simples", s, e, f"'{inner}'"))
            if KEY.match(inner):
                edits.append(("chave_sem_aspas", s, e, inner))
    pos = 0
    for line in text.split("\n"):
        stripped = line.lstrip()
        if pos and stripped.startswith('"'):
            indent = line[:len(line) - len(stripped)]
            edits.append(("comentario", pos, pos, indent + "{c}\n"))
        pos += len(line) + 1
    return edits


YAML_KEY_LINE = re.compile(r"^(\s*(?:- )?)([A-Za-z_][\w.-]*):(\s|$)")
YAML_QUOTED = re.compile(r":\s+([\"'])[^\"'\n]*(\1)\s*$")


def yaml_edits(text: str) -> list[tuple[str, int, int, str]]:
    edits = []
    pos = 0
    for line in text.split("\n"):
        indent = len(line) - len(line.lstrip(" "))
        if line.strip() and not line.lstrip().startswith("#"):
            if indent >= 2:
                edits.append(("tab_indentacao", pos, pos + indent, "\t"))
                edits.append(("indentacao_quebrada", pos, pos + 1, ""))
            if indent and pos:
                edits.append(("indentacao_quebrada", pos, pos, " "))
            if m := YAML_KEY_LINE.match(line):
                edits.append(("apagar_dois_pontos", pos + m.end(2), pos + m.end(2) + 1, ""))
            if m := YAML_QUOTED.search(line):
                edits.append(("aspas_abertas", pos + m.start(2), pos + m.end(2), ""))
        pos += len(line) + 1
    return edits


def apply_edits(text: str, chosen: list) -> str:
    out = text
    for _, s, e, rep in sorted(chosen, key=lambda c: (c[1], c[2]), reverse=True):
        out = out[:s] + rep + out[e:]
    return out


def parse_error(text: str, fmt: str) -> tuple[str, int] | None:
    """(mensagem do parser, linha) se não parseia; None se parseia."""
    try:
        parse(text, fmt)
        return None
    except json.JSONDecodeError as e:
        return f"{e.msg}: line {e.lineno} column {e.colno}", e.lineno
    except yaml.MarkedYAMLError as e:
        mark = e.problem_mark or e.context_mark
        line = mark.line + 1 if mark else None
        problem = e.problem or e.context or "invalid YAML"
        return (f"line {line}, column {mark.column + 1}: {problem}" if mark else problem), line
    except Exception as e:  # noqa: BLE001 — YAML multidocumento, etc.
        return str(e).splitlines()[0][:200], None


def task_fix(text: str, obj, fmt: str, nl: str, r: Random, stats: Counter):
    if r.random() < NEGATIVE_RATE:
        stats["corrigir:negativo"] += 1
        return ("T6", r.choice(CHECK[nl]).format(fmt=DISPLAY[fmt]), text, NO_ERROR[nl].format(fmt=DISPLAY[fmt]),
                fmt, "description", "nenhuma", "a")
    edits = json_edits(text) if fmt == "json" else yaml_edits(text)
    if not edits:
        stats["corrigir:sem_candidatos"] += 1
        return None
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
        if edit[0] == "comentario":
            edit = (edit[0], edit[1], edit[2], edit[3].format(c=r.choice(COMMENTS)))
        if any(not (edit[2] < c[1] or edit[1] > c[2]) for c in chosen):
            continue
        if parse_error(apply_edits(text, chosen + [edit]), fmt):
            chosen.append(edit)
        else:
            stats[f"corrigir:aceito_pelo_parser:{edit[0]}"] += 1
    if not chosen:
        stats["corrigir:sem_mutante"] += 1
        return None
    mutant = apply_edits(text, chosen)
    error = parse_error(mutant, fmt)
    if not error:
        return None
    msg, line = error
    variants = ["a", "b"] + (["c"] if line else [])
    variant = r.choice(variants)
    parts = [r.choice(FIX[nl]).format(fmt=DISPLAY[fmt])]
    if variant == "b":
        parts.append(PARSER_HINT[nl].format(msg=msg))
    elif variant == "c":
        parts.append(LINE_HINT[nl].format(linha=line))
    for op, *_ in chosen:
        stats[f"corrigir:{op}"] += 1
    return "T6", "\n\n".join(parts), mutant, text, fmt, "code", "+".join(op for op, *_ in chosen), variant


TASKS = {"converter": task_convert, "corrigir": task_fix, "formatar": task_format, "extrair": task_extract,
         "editar": task_edit}


# --- montagem --------------------------------------------------------------------------------------

def make_examples(seed: dict, fmt: str, uses: int, r: Random, stats: Counter) -> list[dict]:
    text = seed["content"]
    try:
        obj = parse(text, fmt)
    except Exception:
        stats["semente_invalida"] += 1
        return []
    weights = TASK_WEIGHTS[fmt]
    names = list(weights)
    order: list[str] = []
    while len(order) < len(names):  # amostragem ponderada sem reposição
        rest = [n for n in names if n not in order]
        order.append(r.choices(rest, [weights[n] for n in rest])[0])
    out = []
    for name in order:
        if len(out) >= uses:
            break
        nl = r.choices(NLS, NL_WEIGHTS)[0]
        result = TASKS[name](text, obj, fmt, nl, r, stats)
        if result is None:
            continue
        task, prompt, code, response, lang, kind, mutation, variant = result
        stats[f"tarefa:{name}"] += 1
        digest = hashlib.sha1(f"{name}|{prompt}|{code}".encode()).hexdigest()[:16]
        out.append({
            "id": f"JY-{fmt}-{digest}", "task": task, "level": "facil" if seed["tokens"] < 200 else "medio",
            "lang": lang, "nl": nl, "source": "the-stack-dedup", "repo": seed["repo"], "path": seed["path"],
            "licenses": seed["licenses"], "prompt": prompt, "code": code, "response": response, "kind": kind,
            "lang_in_prompt": True, "fonte": FONTE, "licenca": "por repositório", "tests": None, "test_setup": None,
            "mutacao": mutation, "variante": variant, "origem": seed["digest"], "tipo_bug": name,
            "n_tokens": estimate_tokens(prompt, None) + estimate_tokens(code, fmt)
                        + estimate_tokens(response, lang if kind == "code" else None),
        })
    return out


def process_group(task: tuple[str, int, float, int]) -> tuple[list[dict], Counter]:
    fmt, group, rate, seed = task
    r = Random(f"{seed}-{fmt}-{group}")
    stats: Counter = Counter()
    out = []
    for row in pq.ParquetFile(SEEDS_DIR / f"{fmt}.parquet").read_row_group(group).to_pylist():
        uses = math.floor(rate) + (r.random() < rate - math.floor(rate))
        if uses:
            out += make_examples(row, fmt, uses, r, stats)
    return out, stats


def build(total: int, workers: int, seed: int = 0) -> None:
    """JSON primeiro, usando cada semente em todas as tarefas (há poucas); o YAML completa o total."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = OUT_DIR / "execucao.json"
    all_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    done = 0
    for fmt in FORMATS:
        parts = sorted(OUT_DIR.glob(f"part-{fmt}-*.parquet"))
        if parts:
            done += sum(pq.read_metadata(p).num_rows for p in parts)
            print(f"{fmt}: já gerado, pulando")
            continue
        seeds = pq.ParquetFile(SEEDS_DIR / f"{fmt}.parquet")
        target = total - done if fmt == "yaml" else total // 2
        # ~10% das tentativas falham (round-trip, formato não canônico): sobra margem e corta no alvo.
        rate = min(len(TASK_WEIGHTS[fmt]), target * 1.15 / seeds.metadata.num_rows)
        groups = list(range(seeds.num_row_groups))
        Random(f"{seed}-{fmt}").shuffle(groups)
        rows, stats, start = [], Counter(), time.time()
        with ProcessPoolExecutor(workers) as pool:
            for new, s in pool.map(process_group, [(fmt, g, rate, seed) for g in groups]):
                rows += new
                stats.update(s)
                print(f"  {fmt}: {len(rows):,} exemplos | {time.time() - start:.0f}s", flush=True)
        Random(seed).shuffle(rows)
        rows = rows[:target]
        done += len(rows)
        for i in range(0, len(rows), PER_PART):
            out = OUT_DIR / f"part-{fmt}-{i // PER_PART:04d}.parquet"
            tmp = out.with_suffix(".tmp")
            pq.write_table(pa.Table.from_pylist(rows[i:i + PER_PART], schema=SCHEMA), tmp, compression="zstd")
            tmp.rename(out)
        all_stats[fmt] = {"exemplos": len(rows), "taxa_por_semente": round(rate, 3), **dict(sorted(stats.items()))}
        stats_path.write_text(json.dumps(all_stats, indent=1, ensure_ascii=False))
        print(json.dumps({"formato": fmt, "exemplos": len(rows)}, ensure_ascii=False), flush=True)
    save_meta()


def save_meta() -> None:
    cells: Counter = Counter()
    tokens: Counter = Counter()
    for part in sorted(OUT_DIR.glob("part-*.parquet")):
        for r in pq.read_table(part, columns=["task", "lang", "nl", "tipo_bug", "n_tokens"]).to_pylist():
            key = f"{r['task']}|{r['lang']}|{r['nl']}|{r['tipo_bug']}"
            cells[key] += 1
            tokens[key] += r["n_tokens"]
    meta = {"fonte": FONTE, "origem": "the-stack-dedup (filtered-v2, json/yaml)", "licenca": "por repositório",
            "exemplos": sum(cells.values()), "tokens_estimados": sum(tokens.values()),
            "celulas": {k: {"exemplos": cells[k], "tokens": tokens[k]} for k in sorted(cells)},
            "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))


def sample(n: int) -> None:
    for fmt in FORMATS:
        rows = pq.ParquetFile(SEEDS_DIR / f"{fmt}.parquet").read_row_group(0).to_pylist()
        r, stats = Random(1), Counter()
        shown: Counter = Counter()
        for row in rows:
            for ex in make_examples(row, fmt, 5, r, stats):
                if shown[ex["tipo_bug"]] >= n:
                    continue
                shown[ex["tipo_bug"]] += 1
                print(f"\n===== {fmt} | {ex['task']} {ex['mutacao']} | nl={ex['nl']} | var={ex['variante']} "
                      f"| {ex['n_tokens']} tok")
                print(ex["prompt"])
                print("--- código ---")
                print(ex["code"][:600])
                print(f"--- resposta ({ex['lang']}, {ex['kind']}) ---")
                print(ex["response"][:600])
            if all(shown[t] >= n for t in TASK_WEIGHTS[fmt]):
                break
        print(f"\n{fmt} stats:", dict(stats.most_common(25)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("comando", choices=["sementes", "gerar", "amostra"])
    parser.add_argument("--total", type=int, default=800_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n", type=int, default=2)
    args = parser.parse_args()
    if args.comando == "sementes":
        build_seeds(args.workers)
    elif args.comando == "gerar":
        build(args.total, args.workers)
    else:
        sample(args.n)


if __name__ == "__main__":
    main()
