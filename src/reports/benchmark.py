"""Página de benchmark (``assets/templates/benchmark.html``) a partir dos resultados das avaliações.

O template lê um JSON ``run-data`` (esquema no comentário do topo do template): modelo, rodada,
benchmarks com pontuação, intervalo, linhas de base e quebras. Aqui ficam o ``render`` genérico e
os adaptadores dos formatos do projeto:

* ``from_passk``: o JSON de uma rodada do ``eval.passk``, com todos os conjuntos que ela avaliou
  (``docs/benchmarks/<nome do JSON>.html``);
* ``from_variants``: as variantes do ``eval.benchmark`` (motor × máscara), a primeira como modelo
  e as outras como linhas de base.

    uv run python -m reports.benchmark passk data/codelm_sft/eval/passk/base-v2-step-43938-completar.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

from reports import DOCS_DIR, embed_json, repo_info, slug, template

OUT_DIR = DOCS_DIR / "benchmarks"
BLOCK = re.compile(r'(<script type="application/json" id="run-data">)(.*?)(</script>)', re.S)

SUITES = {
    "humaneval": ("HumanEval", "Python"), "mbpp": ("MBPP", "Python, sanitized"),
    "javascript": ("MultiPL-E HumanEval", "JavaScript"), "typescript": ("MultiPL-E HumanEval", "TypeScript"),
    "java": ("MultiPL-E HumanEval", "Java"), "rust": ("MultiPL-E HumanEval", "Rust"),
    "bash": ("MultiPL-E HumanEval", "Bash"), "t4": ("T4 validation", "JSON description of a function"),
}
for _lang in ("javascript", "typescript", "java", "rust", "bash"):
    SUITES[f"mbpp-{_lang}"] = ("MultiPL-E MBPP", SUITES[_lang][1])

# Tamanho dos conjuntos, para resultados antigos sem acerto por problema.
SUITE_SIZES = {"humaneval": 164, "mbpp": 500, "javascript": 161, "typescript": 159, "java": 158, "rust": 156,
               "bash": 158, "mbpp-javascript": 397, "mbpp-typescript": 390, "mbpp-java": 386, "mbpp-rust": 354,
               "mbpp-bash": 382}

# Números publicados, para comparar (pass@1 em %).
# phi-1-small: Gunasekar et al., 2023, tabela 1. Codex: Chen et al., 2021, tabela 1 (pass@1 a T=0.2);
# o 85M é o do porte do nosso (113M) e o 300M, o tamanho seguinte. O Codex só tem HumanEval.
# Nome do modo na página (os arquivos de resultado usam os nomes do eval.passk).
MODES = {"completar": "completion", "chat": "chat"}

PUBLISHED = {
    "codex-85m": {"label": "Codex-85M", "humaneval": 8.22},
    "codex-300m": {"label": "Codex-300M", "humaneval": 13.17},
    "phi-1-small": {"label": "phi-1-small (350M)", "humaneval": 45.0, "mbpp": 45.0},
}


def render(data: dict, out_dir: str | Path = OUT_DIR, filename: str | None = None) -> Path:
    """Grava a página e devolve o caminho. Completa id, data, gerador e commit quando faltam."""
    data = json.loads(json.dumps(data))
    run = data.setdefault("run", {})
    now = datetime.now().astimezone()
    run.setdefault("date", now.isoformat(timespec="seconds"))
    run.setdefault("id", now.strftime("%Y%m%d-%H%M"))
    run.setdefault("generator", Path(sys.argv[0]).stem if sys.argv[0] else "reports.benchmark")
    info = repo_info()
    code = data.setdefault("code", {})
    code.setdefault("repo", info["repo"])
    code.setdefault("commit", info["commit"])
    if info["dirty"]:
        run.setdefault("notes", "The working tree had uncommitted changes when this ran.")
    html, count = BLOCK.subn(lambda m: f"{m.group(1)}\n{embed_json(data)}\n{m.group(3)}", template("benchmark.html"),
                             count=1)
    if count != 1:
        raise ValueError("bloco run-data não encontrado no template")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (filename or f"{slug(data.get('model', {}).get('name', 'modelo'))}_{run['id']}.html")
    path.write_text(html, encoding="utf-8")
    return path


def mean_ci(values: list[float]) -> list[float] | None:
    """Intervalo de 95% da média (aproximação normal sobre os problemas)."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    half = 1.96 * sd / math.sqrt(len(values))
    return [round(100 * max(0.0, mean - half), 1), round(100 * min(1.0, mean + half), 1)]


def per_problem(entry: dict, suite: str, suites: list[str]) -> list[float]:
    """Taxa de acerto de cada problema do conjunto (``correct`` / amostras)."""
    correct, n = entry.get("correct", {}), entry["samples"]
    keyed = {k.split(":", 1)[1]: v for k, v in correct.items() if k.startswith(f"{suite}:")}
    if not keyed and len(suites) == 1 and not any(":" in k for k in correct):
        keyed = correct  # arquivo antigo, de um conjunto só
    return [c / n for c in keyed.values()]


def from_passk(paths: str | Path | list, command: str | None = None, hardware: str | None = None) -> dict:
    """O documento de uma rodada do ``eval.passk``: todos os conjuntos que ela avaliou.

    Uma linha por conjunto: pass@1 na menor temperatura; as outras temperaturas e k viram quebra.
    Aceita também vários JSONs do mesmo checkpoint (quando um conjunto se repete, vale o mais novo).
    """
    paths = sorted((Path(p) for p in (paths if isinstance(paths, list) else [paths])), key=lambda p: p.stat().st_mtime)
    mode = "completar" if "-completar" in paths[-1].stem else "chat"
    checkpoint = paths[-1].stem.partition(f"-{mode}")[0]
    by_suite: dict[str, dict] = {}  # conjunto -> resultados (por temperatura) do arquivo mais novo
    for path in paths:
        results = json.loads(path.read_text())
        for t in results:
            for suite in results[t]:
                if suite not in ("samples", "correct", "n"):
                    by_suite[suite] = results
    order = [s for s in SUITES if s in by_suite] + [s for s in by_suite if s not in SUITES]
    benchmarks = []
    for suite in order:
        results = by_suite[suite]
        temps = sorted(results, key=float)
        suites = [s for s in results[temps[0]] if s not in ("samples", "correct", "n")]
        temp = next(t for t in temps if suite in results[t] and "pass@1" in results[t][suite])
        entry = results[temp]
        hits = per_problem(entry, suite, suites)
        name, sub = SUITES.get(suite, (suite, None))
        breakdown = [{"label": f"{k} · T={t}", "score": round(100 * v, 1)}
                     for t in temps if suite in results[t] for k, v in results[t][suite].items()
                     if not (t == temp and k == "pass@1")]
        bench = {"name": name if name != "MultiPL-E HumanEval" else f"HumanEval-{sub}",
                 "version": (sub if name != "MultiPL-E HumanEval" else "MultiPL-E") + f" · T={temp}",
                 "n": len(hits) or SUITE_SIZES.get(suite), "metric": "pass@1", "unit": "%",
                 "score": round(100 * entry[suite]["pass@1"], 1), "ci": mean_ci(hits),
                 "baselines": {key: ref[suite] for key, ref in PUBLISHED.items() if suite in ref}}
        if breakdown:
            bench["breakdown"] = breakdown
        if entry["samples"] > 1:
            bench["notes"] = f"{entry['samples']} samples per problem at T={temp}; unbiased pass@k estimator (Chen et al., 2021)."
        benchmarks.append(bench)
    used = {k for b in benchmarks for k in b["baselines"]}
    task = ("Code completion from a function signature and docstring (pre-training format)" if mode == "completar"
            else "Code generation from an instruction (SFT chat format, grammar mask on)")
    newest = datetime.fromtimestamp(paths[-1].stat().st_mtime)
    run = {"id": newest.strftime("%Y%m%d-%H%M"), "date": newest.astimezone().isoformat(timespec="seconds"),
           "generator": "eval.passk", "hardware": hardware, "command": command}
    model = checkpoint.split("-step-")[0]
    return {
        "title": f"{model} benchmark",
        "task": {"name": "pass@k", "description": task + ". Scored by running the official tests in a sandbox."},
        "model": {"name": model, "checkpoint": checkpoint, "mode": MODES[mode]},
        "run": {k: v for k, v in run.items() if v is not None},
        "baselines": [{"key": k, "label": PUBLISHED[k]["label"]} for k in PUBLISHED if k in used],
        "benchmarks": benchmarks,
        "environment": {"results": ", ".join(p.name for p in paths)},
    }


def passk_filename(path: str | Path) -> str:
    """O documento tem o nome do JSON da rodada: ``<checkpoint>-<modo>[-<conjuntos>].html``."""
    return Path(path).stem + ".html"


def from_variants(checkpoint: str, variants: dict[str, dict[str, dict]], primary: str | None = None) -> dict:
    """``variants``: nome -> {id: linha do ``eval.benchmark run``}. A primeira (ou ``primary``) é o modelo."""
    names = list(variants)
    primary = primary or names[0]
    others = [n for n in names if n != primary]

    def rate(rows: list[dict], key: str) -> float:
        return round(100 * sum(r[key] for r in rows) / len(rows), 1)

    benchmarks, examples = [], []
    rows = variants[primary]
    for suite in ("humaneval", "mbpp", "t4"):
        group = [r for r in rows.values() if r["suite"] == suite]
        if not group:
            continue
        name, sub = SUITES[suite]
        metric = "valid JSON" if suite == "t4" else "pass@1"
        key = "valid" if suite == "t4" else "passed"
        baselines = {}
        for other in others:
            og = [r for r in variants[other].values() if r["suite"] == suite]
            if og:
                baselines[slug(other)] = rate(og, key)
        benchmarks.append({
            "name": name, "version": sub, "n": len(group), "metric": metric, "unit": "%", "score": rate(group, key),
            "ci": mean_ci([float(r[key]) for r in group]), "baselines": baselines,
            "breakdown": [{"label": "valid output", "score": rate(group, "valid")},
                          {"label": "finished", "score": rate(group, "finished")}],
            "notes": f"{sum(r['tokens'] for r in group) / sum(r['seconds'] for r in group):.1f} tokens/s"})
        fails = [r for r in group if not r[key]][:2]
        passes = [r for r in group if r[key]][:1]
        examples += [{"benchmark": name, "id": r["id"], "status": "pass" if r[key] else "fail",
                      "input": r.get("prompt", r["id"]), "output": r.get("code") or "(no code)"} for r in fails + passes]
    return {"title": f"{checkpoint.split('/')[0]} benchmark", "task": {"name": "SFT benchmarks",
            "description": "MBPP and HumanEval run with the official tests in a sandbox; T4 checks the JSON description."},
            "model": {"name": checkpoint.split("/")[0], "checkpoint": checkpoint, "variant": primary},
            "run": {"generator": "eval.benchmark"},
            "baselines": [{"key": slug(o), "label": o} for o in others], "benchmarks": benchmarks, "examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    passk = sub.add_parser("passk", help="documento de uma rodada a partir do JSON do eval.passk")
    passk.add_argument("resultados")
    passk.add_argument("--saida", default=str(OUT_DIR))
    raw = sub.add_parser("json", help="página a partir de um JSON já no esquema run-data")
    raw.add_argument("resultados")
    raw.add_argument("--saida", default=str(OUT_DIR))
    args = parser.parse_args()
    if args.cmd == "passk":
        data = from_passk(args.resultados)
        # documento refeito depois: o commit atual pode não ser o que gerou os resultados
        data["run"]["notes"] = ("Rebuilt from saved results; the commit shown is the one checked out when the "
                                "page was built, not necessarily the one that produced the results.")
        print(render(data, args.saida, passk_filename(args.resultados)))
    else:
        print(render(json.loads(Path(args.resultados).read_text()), args.saida))


if __name__ == "__main__":
    main()
