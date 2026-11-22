"""Revisão humana do SFT v2: uma página web local para marcar uma amostra (ver docs/memory/sft-v2.md).

``amostra`` sorteia poucos exemplos por célula fonte × linguagem × tarefa (as linguagens
que não são nossas viram uma célula só) e grava ``v2/revisao/amostra.json`` com as
marcações do QC. ``servir`` abre a página em http://localhost:8765, no template de validação do
Seidelbots (``reports.validation``), com os textos em inglês como o template: para cada exemplo,
approve / reject / unsure, motivos, campos marcados e uma nota livre. As marcações vão para ``v2/revisao/notas.json``
a cada clique, e a revisão pode parar e continuar.

    uv run python -m eval.sft_review amostra [--por-celula 2] [--maximo 300]
    uv run python -m eval.sft_review servir
    uv run python -m eval.sft_review resumo

``amostra-pos`` sorteia exemplos marcados pelas heurísticas do pós-processamento (``dataprep.sft_post``)
para medir a precisão delas; ``--pos`` em ``servir`` e ``resumo`` usa essa amostra:

    uv run python -m eval.sft_review amostra-pos [--por-marca 25] [--lang 10]
    uv run python -m eval.sft_review servir --pos
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pathlib import Path

import pyarrow.parquet as pq

from dataprep.sft_convert import FONTES_DIR, V2_DIR
from dataprep.sft_qc import QC_DIR
from reports import validation

REVIEW_DIR = V2_DIR / "revisao"
SAMPLE = REVIEW_DIR / "amostra.json"
NOTES = REVIEW_DIR / "notas.json"
REASONS = ["bad request", "response misses the request", "wrong code", "incomplete", "syntax",
           "wrong programming language", "wrong natural language", "junk / spam", "too trivial", "other"]
# Notas gravadas antes da página em inglês.
OLD_VERDICTS = {"ruim": "bad", "duvida": "unsure"}
OLD_REASONS = dict(zip(["pedido ruim", "resposta não atende", "código errado", "incompleto", "sintaxe",
                        "linguagem errada", "idioma errado", "lixo / spam", "trivial demais", "outro"], REASONS))
FIELDS = ["id", "task", "lang", "nl", "level", "source", "prompt", "code", "response", "response_raw", "tests"]


def sample(per_cell: int, maximum: int, seed: int = 0, only_qc: bool = False) -> None:
    r = random.Random(seed)
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for fonte_dir in sorted(FONTES_DIR.glob("*")):
        parts = sorted(fonte_dir.glob("part-*.parquet"))
        if only_qc:
            parts = [p for p in parts if (QC_DIR / fonte_dir.name / p.name).exists()]
        if not parts:
            continue
        # Duas partes sorteadas por fonte bastam para uma amostra pequena.
        for part in r.sample(parts, min(2, len(parts))):
            names = pq.read_schema(part).names
            rows = pq.read_table(part, columns=[c for c in FIELDS if c in names]).to_pylist()
            qc_path = QC_DIR / fonte_dir.name / part.name
            qc = {q["id"]: q for q in pq.read_table(qc_path).to_pylist()} if qc_path.exists() else {}
            for row in r.sample(rows, min(200, len(rows))):
                lang = row["lang"] if not row["lang"].startswith("outra") else "outra"
                cells[(fonte_dir.name, lang, row["task"])].append({**row, "fonte": fonte_dir.name,
                                                                   "qc": qc.get(row["id"])})
    chosen = [ex for key in sorted(cells) for ex in r.sample(cells[key], min(per_cell, len(cells[key])))]
    r.shuffle(chosen)
    chosen = chosen[:maximum]
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE.write_text(json.dumps(chosen, ensure_ascii=False, indent=1, default=str))
    print(f"{len(chosen)} exemplos de {len(cells)} células em {SAMPLE}")


def sample_post(per_mark: int, lang_fixes: int, seed: int = 0) -> None:
    from dataprep.sft_post import MARCAS, POST_DIR
    r = random.Random(seed)
    pools: dict[str, list[tuple[Path, str, dict]]] = defaultdict(list)
    for post_dir in sorted(POST_DIR.glob("*")):
        if post_dir.name.startswith("traducao"):
            continue
        for path in sorted(post_dir.glob("part-*.parquet")):
            qc_path = QC_DIR / post_dir.name / path.name
            # Só o que a heurística acrescenta: quem já reprova no QC ou numa regra precisa sai de qualquer jeito.
            failed = {q["id"] for q in pq.read_table(qc_path, columns=["id", "qc_falhas"]).to_pylist()
                      if q["qc_falhas"]} if qc_path.exists() else set()
            for p in pq.read_table(path).to_pylist():
                if p["post_falhas"]:
                    continue
                if p["id"] not in failed:
                    for m in p["post_marcas"]:
                        pools[m].append((path, p["id"], p))
                # A linguagem só é corrigida quando a resposta reprova na sintaxe do rótulo antigo.
                if p["lang_corrigido"]:
                    pools["lang_corrigido"].append((path, p["id"], p))
    quotas = {**{m: per_mark for m in MARCAS}, "lang_corrigido": lang_fixes}
    chosen: dict[Path, dict[str, tuple[str, dict]]] = defaultdict(dict)
    for group, n in quotas.items():
        for path, eid, p in r.sample(pools[group], min(n, len(pools[group]))):
            chosen[path][eid] = (group, p)
    out = []
    for path, wanted in chosen.items():
        fonte = path.parent.name
        names = pq.read_schema(FONTES_DIR / fonte / path.name).names
        rows = pq.read_table(FONTES_DIR / fonte / path.name, columns=[c for c in FIELDS if c in names]).to_pylist()
        qc_path = QC_DIR / fonte / path.name
        qc = {q["id"]: q for q in pq.read_table(qc_path).to_pylist()} if qc_path.exists() else {}
        for row in rows:
            if row["id"] in wanted:
                group, p = wanted[row["id"]]
                out.append({**row, "fonte": fonte, "qc": qc.get(row["id"]), "grupo": group, "post": p})
    r.shuffle(out)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(f"{len(out)} exemplos em {SAMPLE}: " + ", ".join(f"{g} {len(pools[g]):,} no total" for g in quotas))


VERDICTS = [
    {"value": "ok", "label": "Approve", "key": "a", "tone": "green", "icon": "✓", "advance": True},
    {"value": "bad", "label": "Reject", "key": "r", "tone": "amber", "icon": "✕", "advance": False},
    {"value": "unsure", "label": "Unsure", "key": "u", "tone": "violet", "icon": "?", "advance": False},
]
FIELD_VIEW = [
    {"key": "original", "label": "Original request (English)", "type": "text"},
    {"key": "prompt", "label": "Request", "type": "text"},
    {"key": "code", "label": "Code in the request", "type": "code"},
    {"key": "response", "label": "Response", "type": "auto"},
    {"key": "response_raw", "label": "Raw response", "type": "text", "collapsed": True},
    {"key": "tests", "label": "Tests", "type": "code", "collapsed": True},
]


# Referência das tarefas (docs/memory/sft.md e docs/memory/sft-v2.md): rótulo, resumo e o que checar em cada uma.
TASKS = {
    "T1": {
        "label": "description → code",
        "summary": "A request in natural language; the response is code that does what was asked.",
        "guidelines": [
            "The code does everything the request asks, in the programming language it asks for.",
            "Prose around the code is fine when it is correct and in the request's natural language.",
            "Reject when parts of the request are ignored, or the request is too vague to have a right answer.",
        ],
    },
    "T2": {
        "label": "docstring → code",
        "summary": "A function signature with its docstring (Code in the request); the response is the function "
                   "with the body implemented.",
        "guidelines": [
            "The request is a fixed instruction; it may be in another natural language than the docstring. "
            "That is on purpose.",
            "The response keeps the signature and the docstring and adds a body that does what the docstring says.",
            "Calls to other parts of the original project are fine; judge whether the body matches the docstring.",
            "Reject stubs (pass, TODO) and docstrings too vague to define the behaviour (reason: bad request).",
        ],
    },
    "T3": {
        "label": "code → description",
        "summary": "Code in the request; the response explains in natural language what it does.",
        "guidelines": [
            "The description is accurate and covers the main behaviour, with no invented details.",
            "It is written in the request's natural language.",
        ],
    },
    "T4": {
        "label": "code → structured description",
        "summary": "Code in the request; the response describes it as JSON in the fixed schema.",
        "guidelines": [
            "The JSON is valid and follows the schema.",
            "Every field is accurate for this code; nothing is invented.",
        ],
    },
    "T5": {
        "label": "code → docstring",
        "summary": "A function without documentation; the response is the same function with a docstring added.",
        "guidelines": [
            "Apart from the docstring, the code is unchanged (compare with Code in the request).",
            "The docstring uses the language's convention: docstring, JSDoc, Javadoc, rustdoc, block comment in C.",
            "The docstring comes from the original author: judge whether it describes the function, not its style.",
            "Reject leftovers that say nothing about the function, such as '(non-Javadoc) @see ...'.",
        ],
    },
    "T6": {
        "label": "fix the bug",
        "summary": "Broken code, sometimes with a hint (what it should do, the error line, a failing test); the "
                   "response is the corrected code.",
        "guidelines": [
            "Bugs are injected automatically: a logic change in Python, checked by tests; a syntax error in C, "
            "TS, Rust, JS, Java or Bash; or a parse error in JSON/YAML.",
            "The response is the full original code: the fix should be minimal and everything else identical.",
            "About 5% are negatives: the code is already correct and the response says so. Approve if it really is.",
            "Reject when the hint contradicts the code, or the bug leaves more than one reasonable fix and "
            "nothing in the request picks one.",
        ],
    },
    "T7": {
        "label": "JSON/YAML transformation",
        "summary": "A JSON or YAML document; the response converts it (JSON ↔ YAML), reformats it, edits it, or "
                   "extracts a value.",
        "guidelines": [
            "Generated and checked automatically; confirm the response does exactly what was asked: the right "
            "path, key or format.",
            "In edits and conversions, nothing else in the document changes.",
            "The documents come from real repositories: reject only when the document itself is junk.",
        ],
    },
}


def task_guide(translation: bool) -> dict:
    # Na tradução só o pedido muda: basta saber o que é cada tarefa.
    entries = {k: {key: v for key, v in t.items() if not (translation and key == "guidelines")}
               for k, t in TASKS.items()}
    return {"key": "task", "title": "Tasks", "entries": entries}


POST_GUIDELINES = [
    "Each item was flagged by a post-processing heuristic (pill 'grupo', amber 'post:' warning).",
    "Judge the example as usual: Reject if it really has the problem the flag names, Approve if the flag is wrong.",
    "For 'lang_corrigido', judge the corrected language: Approve if the new label is right.",
]


def page_config(translation: bool, post: bool = False) -> dict:
    config = _page_config(translation)
    if post:
        config["title"] = "SFT v2 post-processing check"
        config["guidelines"] = POST_GUIDELINES + config["guidelines"]
    return config


def _page_config(translation: bool) -> dict:
    return {
        "title": "Translation pilot" if translation else "SFT v2 review",
        "dataset": SAMPLE.name,
        "description": ("Read the translated request next to the original. Approve if the translation is correct and "
                        "natural, with code names left intact." if translation else
                        "Read the request and the response. Approve if the response does what was asked, in the requested "
                        "programming and natural language; Reject otherwise, with the reasons ticked."),
        "guidelines": [
            "Judge the response against the request; the code in the request and the tests are context.",
            "Amber warnings come from the automatic QC: check whether you agree with them.",
            "Unsure when deciding would take running the code or looking something up.",
            "Flag the field with the problem and explain in the note.",
        ],
        "source": {"items": "/amostra", "notes": "/notas"},
        "storageKey": f"codelm-review:{SAMPLE.stem}",
        "idKey": "id",
        "meta": ["dataset", "task", "lang", "nl", "level", "source", "grupo"],
        "itemGuide": task_guide(translation),
        "copy": ["qc_falhas"],
        "warn": "avisos",
        "fields": FIELD_VIEW,
        "verdicts": VERDICTS,
        "reasons": REASONS,
        "reasonsFor": ["bad", "unsure"],
    }


def items_for_page() -> list[dict]:
    items = json.loads(SAMPLE.read_text())
    for ex in items:
        qc = ex.get("qc") or {}
        ex["dataset"] = ex.get("fonte")
        ex["qc_falhas"] = qc.get("qc_falhas")
        ex["avisos"] = [f"qc: {f}" for f in (qc.get("qc_falhas") or [])]
        if ex.get("traducao") and ex["traducao"] != "ok":
            ex["avisos"].append(f"translation: {ex['traducao']}")
        post = ex.get("post") or {}
        ex["avisos"] += [f"post: {m}" for m in (post.get("post_falhas") or []) + (post.get("post_marcas") or [])]
        if post.get("lang_corrigido"):
            ex["avisos"].append(f"post: lang {ex.get('lang')} → {post['lang_corrigido']}")
        if post.get("nl_corrigido"):
            ex["avisos"].append(f"post: nl {ex.get('nl')} → {post['nl_corrigido']}")
    return items


def load_notes() -> dict:
    notes = json.loads(NOTES.read_text()) if NOTES.exists() else {}
    out = {}
    for k, v in notes.items():
        v = validation.normalize(v)
        v["verdict"] = OLD_VERDICTS.get(v.get("verdict"), v.get("verdict"))
        v["reasons"] = [OLD_REASONS.get(r, r) for r in v["reasons"]]
        v.setdefault("dataset", v.get("fonte"))
        out[k] = v
    return out


class Handler(BaseHTTPRequestHandler):
    translation = False
    post = False

    def _send(self, body: bytes, kind: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/amostra":
            self._send(json.dumps(items_for_page(), ensure_ascii=False, default=str).encode(), "application/json")
        elif self.path == "/notas":
            self._send(json.dumps(load_notes(), ensure_ascii=False).encode(), "application/json")
        else:
            self._send(validation.render(page_config(self.translation, self.post)).encode(), "text/html; charset=utf-8")

    def do_POST(self) -> None:
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        notes = load_notes()
        notes[data["id"]] = data["record"]
        tmp = NOTES.with_suffix(".tmp")
        tmp.write_text(json.dumps(notes, ensure_ascii=False, indent=1))
        tmp.rename(NOTES)
        self._send(b"ok", "text/plain")

    def log_message(self, *args) -> None:
        pass


def serve(port: int, translation: bool = False, post: bool = False) -> None:
    Handler.translation, Handler.post = translation, post
    if not SAMPLE.exists():
        raise SystemExit("rode antes: uv run python -m eval.sft_review amostra")
    print(f"revisão em http://localhost:{port} (Ctrl+C para parar)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def summary() -> None:
    done = [n for n in load_notes().values() if n.get("verdict")]
    by_fonte: dict[str, Counter] = defaultdict(Counter)
    reasons = Counter()
    qc_agrees = Counter()
    for n in done:
        by_fonte[n.get("dataset") or "?"][n["verdict"]] += 1
        reasons.update(n["reasons"])
        flagged = bool(n.get("qc_falhas"))
        qc_agrees[(n["verdict"], "qc reprovou" if flagged else "qc aprovou")] += 1
    print(f"{len(done)} revisados")
    for fonte, c in sorted(by_fonte.items()):
        total = sum(c.values())
        print(f"  {fonte:<24} {c['bad']:>3} ruins, {c['unsure']:>3} dúvidas de {total:>3} ({100 * c['bad'] / total:.0f}%)")
    print("motivos:", dict(reasons.most_common()))
    print("revisão × QC:", {f"{v} / {q}": n for (v, q), n in sorted(qc_agrees.items())})
    groups: dict[str, Counter] = defaultdict(Counter)
    for n in done:
        if n.get("grupo"):
            groups[n["grupo"]][n["verdict"]] += 1
    for group, c in sorted(groups.items()):
        total = sum(c.values())
        hit = c["ok"] if group == "lang_corrigido" else c["bad"]
        print(f"  {group:<18} precisão {100 * hit / total:.0f}% ({hit} de {total}; dúvidas {c['unsure']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sample_cmd = commands.add_parser("amostra")
    sample_cmd.add_argument("--por-celula", type=int, default=2)
    sample_cmd.add_argument("--maximo", type=int, default=300)
    sample_cmd.add_argument("--so-com-qc", action="store_true", help="só partes que já passaram pelo QC")
    serve_cmd = commands.add_parser("servir")
    serve_cmd.add_argument("--porta", type=int, default=8765)
    serve_cmd.add_argument("--traducao", action="store_true", help="revisa o piloto de tradução")
    serve_cmd.add_argument("--pos", action="store_true", help="revisa a amostra do pós-processamento")
    summary_cmd = commands.add_parser("resumo")
    summary_cmd.add_argument("--traducao", action="store_true")
    summary_cmd.add_argument("--pos", action="store_true")
    post_cmd = commands.add_parser("amostra-pos", help="exemplos marcados pelas heurísticas do pós-processamento")
    post_cmd.add_argument("--por-marca", type=int, default=25)
    post_cmd.add_argument("--lang", type=int, default=10, help="exemplos com a linguagem corrigida")
    args = parser.parse_args()
    global SAMPLE, NOTES
    if getattr(args, "traducao", False):
        SAMPLE, NOTES = REVIEW_DIR / "amostra-traducao.json", REVIEW_DIR / "notas-traducao.json"
    if getattr(args, "pos", False) or args.command == "amostra-pos":
        SAMPLE, NOTES = REVIEW_DIR / "amostra-pos.json", REVIEW_DIR / "notas-pos.json"
    if args.command == "amostra-pos":
        sample_post(args.por_marca, args.lang)
    elif args.command == "amostra":
        sample(args.por_celula, args.maximo, only_qc=args.so_com_qc)
    elif args.command == "servir":
        serve(args.porta, args.traducao, args.pos)
    else:
        summary()


if __name__ == "__main__":
    main()
