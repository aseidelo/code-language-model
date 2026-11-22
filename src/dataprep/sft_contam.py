"""Contaminação do SFT v2 com os benchmarks (QC camada 3), desenho aprovado em 2026-11-22.

Marca, não apaga. Grava ``v2/contaminacao/<fonte>/<parte>`` com ``id``, ``contaminado``
(``<benchmark>:<id do problema>``, ou nulo, pela ``rule()``) e as coberturas que a regra usa
(melhor enunciado, melhor solução e o tamanho dela, enunciado curto igual), para recalibrar sem rodar de novo.

Referências (``references()``):

* HumanEval e MBPP (todos os splits);
* MultiPL-E HumanEval e MBPP nas nossas linguagens;
* HumanEval-XL: python, java, javascript, typescript × en, pt, es, fr, de, it, baixado do GitHub
  dos autores (o loader do HF é um script);
* McEval generation nas nossas linguagens;
* CRUXEval;
* NL2SH-ALFA (teste);
* os nossos exercícios: ``valid`` de todas as versões e ``rl``.

Medida: n-gramas de ``N`` palavras e símbolos (o mesmo tokenizador do ``eval/contamination.py``).
Cada problema tem um enunciado e uma solução. A cobertura de uma parte é a fração dos seus
n-gramas que aparece no exemplo (pedido + código de entrada + resposta, e o ``prompt_en`` nas
traduções). A marca usa só o enunciado (``rule()``, decidido em 2026-11-22): cobertura >= ``MIN_COVERAGE``
com pelo menos ``MIN_STATEMENT_NGRAMS`` n-gramas, ou total nos enunciados menores. Enunciados mais
curtos que ``N`` palavras (NL2SH, alguns MBPP) contam só quando o pedido é igual a eles, com a caixa e os
espaços normalizados. A cobertura da solução fica gravada, mas não marca: ela casava idiomas genéricos.

    uv run python -m dataprep.sft_contam [fonte ...] [--workers 8]
    uv run python -m dataprep.sft_contam --relatorio
    uv run python -m dataprep.sft_contam --reaplicar   # nova rule() sobre as coberturas gravadas
"""

import argparse
import json
import re
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from dataprep.sft_convert import FONTES_DIR, SFT_DIR, V2_DIR
from dataprep.sft_dedup import word_hashes

CONTAM_DIR = V2_DIR / "contaminacao"
BENCH_CACHE = SFT_DIR / "eval" / "benchmarks"
N = 10
MIN_COVERAGE = 0.5
MIN_NGRAMS = 10
MIN_STATEMENT_NGRAMS = 10
POW = np.array([pow(1_000_003, k, 2**64) for k in range(N)], dtype=np.uint64)

MULTIPL_E = ("js", "ts", "java", "rs", "sh")
XL_PL = ("python", "java", "javascript", "typescript")
XL_NL = ("English", "Portuguese", "Spanish", "French", "German", "Italian")
XL_URL = "https://raw.githubusercontent.com/FloatAI/humaneval-xl/main/data/{pl}/{nl}.jsonl"
MCEVAL = ("Python", "Java", "JavaScript", "TypeScript", "C", "Rust", "Shell", "HTML", "JSON")
EXERCISES = ("exercises/valid", "exercises/rl", "exercises-v1/valid", "exercises-v2/valid", "exercises-v3/valid")


def humaneval_xl() -> list[dict]:
    rows = []
    for pl in XL_PL:
        for nl in XL_NL:
            path = BENCH_CACHE / "humaneval-xl" / pl / f"{nl}.jsonl"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(XL_URL.format(pl=pl, nl=nl), path)
            rows += [{**json.loads(l), "nl": nl} for l in path.read_text().splitlines() if l.strip()]
    return rows


def references() -> list[dict]:
    """Cada problema: ``ref`` (benchmark:id), ``statement`` e ``solution`` (qualquer um pode ser vazio)."""
    refs = []
    for r in load_dataset("openai/openai_humaneval", split="test"):
        refs.append({"ref": f"humaneval:{r['task_id']}", "statement": r["prompt"], "solution": r["canonical_solution"]})
    for split in ("train", "validation", "test", "prompt"):
        for r in load_dataset("google-research-datasets/mbpp", "full", split=split):
            refs.append({"ref": f"mbpp:{r['task_id']}", "statement": r["text"], "solution": r["code"]})
    for suite in ("humaneval", "mbpp"):
        for lang in MULTIPL_E:
            for r in load_dataset("nuprl/MultiPL-E", f"{suite}-{lang}", split="test"):
                refs.append({"ref": f"multipl-e:{r['name']}", "statement": r["prompt"], "solution": ""})
    for r in humaneval_xl():
        refs.append({"ref": f"humaneval-xl:{r['task_id']}:{r['nl']}", "statement": r["description"],
                     "solution": r["canonical_solution"]})
    for r in load_dataset("Multilingual-Multimodal-NLP/McEval", "generation", split="test"):
        if r["task_id"].split("/")[0] in MCEVAL:
            refs.append({"ref": f"mceval:{r['task_id']}", "statement": r["instruction"], "solution": r["canonical_solution"]})
    for r in load_dataset("cruxeval-org/cruxeval", split="test"):
        refs.append({"ref": f"cruxeval:{r['id']}", "statement": "", "solution": r["code"]})
    for r in load_dataset("westenfelder/NL2SH-ALFA", "test", split="train"):
        refs.append({"ref": f"nl2sh-alfa:{r['nl'][:40]}", "statement": r["nl"], "solution": r["bash"]})
    for name in EXERCISES:
        for r in pq.read_table(SFT_DIR / f"{name}.parquet", columns=["id", "prompt", "response"]).to_pylist():
            refs.append({"ref": f"{name.replace('/', ':')}:{r['id']}", "statement": r["prompt"], "solution": r["response"]})
    for r in refs:  # campos nulos ou NaN viram texto vazio
        r["statement"] = r["statement"] if isinstance(r["statement"], str) else ""
        r["solution"] = r["solution"] if isinstance(r["solution"], str) else ""
    seen, unique = set(), []
    for r in refs:  # as versões dos exercícios repetem problemas
        key = (r["statement"], r["solution"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def ngrams(text: str) -> np.ndarray:
    w = word_hashes(text)
    if len(w) < N:
        return np.zeros(0, dtype=np.uint64)
    n = len(w) - N + 1
    with np.errstate(over="ignore"):
        return np.unique(sum(w[k:k + n] * POW[k] for k in range(N)))


def short_key(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


# Índice global, montado no processo pai antes do fork.
REFS: list[dict] = []
PART_SIZE: np.ndarray = np.zeros(0)       # n-gramas de cada parte (2 por problema: enunciado, solução)
KEYS: np.ndarray = np.zeros(0, np.uint64)  # n-gramas de todas as partes, ordenados e únicos
POSTINGS: dict[int, list[int]] = {}        # n-grama -> partes que o contêm
SHORT: dict[str, int] = {}                 # enunciado curto normalizado -> problema
STATEMENT_SIZE: dict[str, int] = {}        # problema -> n-gramas do enunciado


def build_index() -> None:
    global REFS, PART_SIZE, KEYS, POSTINGS, SHORT
    REFS = references()
    sizes, postings = [], defaultdict(list)
    for i, r in enumerate(REFS):
        for j, text in enumerate((r["statement"], r["solution"])):
            g = ngrams(text or "")
            part = 2 * i + j
            if j == 1 and len(g) < MIN_NGRAMS:
                g = g[:0]
            sizes.append(len(g))
            for h in g.tolist():
                postings[h].append(part)
            if j == 0 and len(g) == 0 and (text or "").strip():
                SHORT.setdefault(short_key(text), i)
    PART_SIZE = np.array(sizes)
    STATEMENT_SIZE.update({r['ref']: int(PART_SIZE[2 * i]) for i, r in enumerate(REFS)})
    POSTINGS = dict(postings)
    KEYS = np.array(sorted(POSTINGS), dtype=np.uint64)
    print(f"  {len(REFS):,} problemas, {len(KEYS):,} n-gramas, {len(SHORT):,} enunciados curtos", flush=True)


def check(text: str, prompt: str) -> dict:
    """Melhor enunciado e melhor solução cobertos pelo exemplo, separados, e o enunciado curto igual ao pedido."""
    best = {"enunciado": (None, 0.0), "solucao": (None, 0.0)}
    g = ngrams(text)
    if len(g):
        pos = np.searchsorted(KEYS, g).clip(0, len(KEYS) - 1)
        hits = g[KEYS[pos] == g]
        if len(hits):
            count = Counter(p for h in hits.tolist() for p in POSTINGS[h])
            for part, c in count.items():
                kind = "solucao" if part % 2 else "enunciado"
                cov = c / PART_SIZE[part]
                if cov > best[kind][1]:
                    best[kind] = (part // 2, cov)
    short = SHORT.get(short_key(prompt))
    return {"ref_enunciado": REFS[best["enunciado"][0]]["ref"] if best["enunciado"][0] is not None else None,
            "cob_enunciado": round(float(best["enunciado"][1]), 3),
            "ref_solucao": REFS[best["solucao"][0]]["ref"] if best["solucao"][0] is not None else None,
            "cob_solucao": round(float(best["solucao"][1]), 3),
            "ngramas_solucao": int(PART_SIZE[2 * best["solucao"][0] + 1]) if best["solucao"][0] is not None else 0,
            "ref_curto": REFS[short]["ref"] if short is not None else None}


def rule(c: dict) -> str | None:
    """Regra aprovada em 2026-11-22: só o enunciado conta.

    A solução casava idiomas genéricos (o ``is_prime`` de 6k±1 dava 79 mil hits num exercício e
    30 mil num problema do McEval), e enunciados de 4 n-gramas casavam com metade deles. Agora o
    enunciado precisa de ``MIN_STATEMENT_NGRAMS`` n-gramas (os mais curtos, cobertura total) ou
    ser igual ao pedido (``ref_curto``). As colunas da solução ficam gravadas só para consulta.
    """
    ref = c["ref_enunciado"]
    if ref and c["cob_enunciado"] >= MIN_COVERAGE and (STATEMENT_SIZE[ref] >= MIN_STATEMENT_NGRAMS
                                                     or c["cob_enunciado"] >= 0.99):
        return ref
    return c["ref_curto"]


def reapply() -> None:
    """Recalcula ``contaminado`` com a ``rule()`` atual a partir das coberturas gravadas."""
    build_index()
    paths = sorted(CONTAM_DIR.glob("*/part-*.parquet"))
    for i, path in enumerate(paths, 1):
        t = pq.read_table(path)
        rows = t.to_pylist()
        refs = pa.array([rule(r) for r in rows], pa.string())
        t = t.set_column(t.schema.get_field_index("contaminado"), "contaminado", refs)
        tmp = path.with_suffix(".tmp")
        pq.write_table(t, tmp, compression="zstd")
        tmp.rename(path)
        if i % 500 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)} partes", flush=True)


COLS = {"ref_enunciado": pa.string(), "cob_enunciado": pa.float32(), "ref_solucao": pa.string(),
        "cob_solucao": pa.float32(), "ngramas_solucao": pa.int32(), "ref_curto": pa.string()}


def contam_part(task: tuple[str, str, str]) -> str:
    fonte, path, out = task
    names = pq.read_schema(path).names
    cols = [c for c in ("id", "prompt", "prompt_en", "code", "response") if c in names]
    rows = []
    for r in pq.read_table(path, columns=cols).to_pylist():
        prompt = r.get("prompt") or ""
        text = "\n".join(x for x in (prompt, r.get("prompt_en"), r.get("code"), r.get("response")) if x)
        c = check(text, prompt)
        if c["ref_curto"] is None and r.get("prompt_en"):
            c["ref_curto"] = SHORT.get(short_key(r["prompt_en"])) is not None and REFS[SHORT[short_key(r["prompt_en"])]]["ref"] or None
        rows.append({"id": r["id"], "contaminado": rule(c), **c})
    schema = pa.schema([("id", pa.string()), ("contaminado", pa.string()), *COLS.items()])
    tmp = Path(out).with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression="zstd")
    tmp.rename(out)
    return out


def report() -> None:
    per_fonte, per_ref = Counter(), defaultdict(Counter)
    for fonte_dir in sorted(p for p in CONTAM_DIR.iterdir() if p.is_dir()):
        for part in fonte_dir.glob("part-*.parquet"):
            t = pq.read_table(part, columns=["contaminado"]).column("contaminado").to_pylist()
            per_fonte[(fonte_dir.name, "exemplos")] += len(t)
            for ref in t:
                if ref:
                    bench = ref.split(":")[0] if not ref.startswith("exercises") else ":".join(ref.split(":")[:2])
                    per_fonte[(fonte_dir.name, bench)] += 1
                    per_ref[bench][ref] += 1
    benches = sorted(per_ref)
    summary = {b: {"problemas": len(per_ref[b]), "exemplos": sum(per_ref[b].values()),
                   "mais_frequentes": per_ref[b].most_common(10)} for b in benches}
    (CONTAM_DIR / "problemas.json").write_text(json.dumps({b: dict(per_ref[b]) for b in benches}, indent=1))
    (CONTAM_DIR / "relatorio.json").write_text(json.dumps(
        {"por_benchmark": summary, "por_fonte": {f"{f}|{k}": v for (f, k), v in sorted(per_fonte.items())}}, indent=1))
    print(f"{'benchmark':<22}{'problemas':>10}{'exemplos':>10}")
    for b in benches:
        print(f"{b:<22}{summary[b]['problemas']:>10,}{summary[b]['exemplos']:>10,}")
    print()
    fontes = sorted({f for f, _ in per_fonte})
    for f in fontes:
        hits = {k: v for (ff, k), v in per_fonte.items() if ff == f and k != "exemplos"}
        if hits:
            n = per_fonte[(f, "exemplos")]
            print(f"{f:<26}{n:>11,}  " + ", ".join(f"{k} {v:,}" for k, v in sorted(hits.items(), key=lambda x: -x[1])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fontes", nargs="*")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--relatorio", action="store_true")
    parser.add_argument("--reaplicar", action="store_true", help="recalcula a marca com a rule() atual, sem refazer a busca")
    args = parser.parse_args()
    if args.relatorio:
        report()
        return
    if args.reaplicar:
        reapply()
        report()
        return
    build_index()
    fontes = args.fontes or sorted(p.name for p in FONTES_DIR.iterdir() if p.is_dir() and any(p.glob("part-*.parquet")))
    tasks = []
    for fonte in fontes:
        (CONTAM_DIR / fonte).mkdir(parents=True, exist_ok=True)
        tasks += [(fonte, str(p), str(CONTAM_DIR / fonte / p.name)) for p in sorted((FONTES_DIR / fonte).glob("part-*.parquet"))
                  if not (CONTAM_DIR / fonte / p.name).exists()]
    start = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(contam_part, tasks, chunksize=2), 1):
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} partes, {time.time() - start:.0f}s", flush=True)
    report()


if __name__ == "__main__":
    main()
