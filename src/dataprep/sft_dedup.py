"""Duplicatas do SFT v2 (QC camada 3), desenho aprovado em 2026-11-22 (ver docs/memory/sft-v2.md).

Marca, não apaga. Grava ``v2/dedup/<fonte>/<parte>`` com ``id``, ``dup_de`` (id do exemplo que fica
no grupo) e ``dup_tipo`` (``exata`` ou ``proxima``); os dois são nulos para quem fica.

* **Exata:** mesma chave ``(task, nl, lang, pedido + resposta)``, em que o pedido é ``prompt`` + ``code``, com os espaços normalizados. Vale para
  todas as tarefas.
* **Próxima:** MinHash do pedido (5-gramas de palavras e símbolos, 128 permutações; LSH com 16
  bandas de 8 linhas), confirmada com Jaccard estimado >= 0,8, dentro do mesmo ``(task, nl, lang)`` e
  entre fontes. A ``lang`` entra porque o mesmo problema em outra linguagem não é duplicata (o
  rosetta-code repete o enunciado em cada linguagem). T6 e T7 ficam só com a exata: pedidos quase iguais são o desenho dessas tarefas.
* **Quem fica:** um por grupo. Primeiro quem passa no QC e no pós-processamento, depois a
  prioridade da fonte (``TIERS``), depois o hash do id.

Passos (os dois primeiros gravam arrays em ``v2/dedup/trabalho/``, uma linha por exemplo):

    uv run python -m dataprep.sft_dedup assinaturas [--workers 12]
    uv run python -m dataprep.sft_dedup agrupar
    uv run python -m dataprep.sft_dedup --relatorio
"""

import argparse
import json
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash

from dataprep.sft_convert import FONTES_DIR, V2_DIR
from dataprep.sft_post import POST_DIR
from dataprep.sft_qc import QC_DIR

DEDUP_DIR = V2_DIR / "dedup"
WORK_DIR = DEDUP_DIR / "trabalho"

WORD = re.compile(r"\w+|[^\w\s]")
SPACES = re.compile(r"\s+")
SHINGLE = 5
PERMS = 128
BANDS, ROWS = 16, 8
THRESHOLD = 0.8
EXACT_ONLY = ("T6", "T7")

# Prioridade da fonte (menor fica). Fonte fora da lista: instruções sintéticas.
TIERS = {
    **dict.fromkeys(["kodcode", "acecode-87k", "acecode-v2", "exercicios-v1", "opc-stage2-educational",
                     "opc-stage2-package", "t6-kodcode", "t6-acecode-87k", "t6-exercicios-v1",
                     "t6-opc-stage2-educational"], 0),
    **dict.fromkeys(["corpus-v2", "t6s-corpus-v2", "code-search-net", "rosetta-code", "nl2bash", "nl2bash-custom",
                     "tldr-pages", "banco-v1-corpus", "websight", "jy-stack"], 1),
    **dict.fromkeys(["traducao-v2", "traducao-v2b"], 3),
}
SYNTHETIC = 2

rng = np.random.default_rng(20261122)
A = rng.integers(1, 2**63, PERMS, dtype=np.uint64) | np.uint64(1)  # multiplicadores ímpares
B = rng.integers(0, 2**63, PERMS, dtype=np.uint64)
POW = np.array([pow(1_000_003, k, 2**64) for k in range(SHINGLE)], dtype=np.uint64)
MIX = np.uint64(0x9E3779B97F4A7C15)


def h64(text: str) -> int:
    return xxhash.xxh64_intdigest(text.encode("utf-8", "surrogatepass"))


def parts() -> list[tuple[str, Path]]:
    return [(f.name, p) for f in sorted(FONTES_DIR.iterdir()) if f.is_dir()
            for p in sorted(f.glob("part-*.parquet"))]


def layout() -> tuple[list[tuple[str, Path]], np.ndarray]:
    """Partes e o deslocamento de cada uma nos arrays globais."""
    ps = parts()
    sizes = [pq.ParquetFile(p).metadata.num_rows for _, p in ps]
    return ps, np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


def norm(text: str | None) -> str:
    return SPACES.sub(" ", text or "").strip()


_WORDS: dict[str, int] = {}


def word_hashes(text: str) -> np.ndarray:
    out = []
    for w in WORD.findall(text):
        h = _WORDS.get(w)
        if h is None:
            h = _WORDS[w] = h64(w)
        out.append(h)
    if len(_WORDS) > 2_000_000:
        _WORDS.clear()
    return np.array(out, dtype=np.uint64)


def minhash(text: str) -> np.ndarray:
    w = word_hashes(text)
    if len(w) == 0:
        return np.full(PERMS, 0xFFFFFFFF, dtype=np.uint32)
    if len(w) < SHINGLE:
        sh = np.array([np.sum(w * POW[:len(w)])], dtype=np.uint64)
    else:
        n = len(w) - SHINGLE + 1
        sh = sum(w[k:k + n] * POW[k] for k in range(SHINGLE))
        sh = np.unique(sh)
    with np.errstate(over="ignore"):
        return ((sh[:, None] * A[None, :] + B[None, :]) >> np.uint64(32)).min(axis=0).astype(np.uint32)


def band_keys(sig: np.ndarray, scope: np.ndarray) -> np.ndarray:
    """Hash de cada banda, já com o escopo (task, nl, lang): bandas iguais em escopos diferentes não colidem."""
    words = sig.view(np.uint64).reshape(len(sig), BANDS, ROWS // 2)
    with np.errstate(over="ignore"):
        h = scope[:, None] * MIX + np.arange(BANDS, dtype=np.uint64)[None, :] * np.uint64(0xC2B2AE3D27D4EB4F)
        for k in range(ROWS // 2):
            h = (h ^ words[:, :, k]) * MIX
            h ^= h >> np.uint64(29)
    return h


def clean_flags(fonte: str, path: Path, ids: list[str]) -> np.ndarray:
    bad = set()
    qc = QC_DIR / fonte / path.name
    if qc.exists():
        t = pq.read_table(qc, columns=["id", "qc_falhas"]).to_pylist()
        bad |= {r["id"] for r in t if r["qc_falhas"]}
    post = POST_DIR / fonte / path.name
    if post.exists():
        t = pq.read_table(post, columns=["id", "post_falhas", "post_marcas"]).to_pylist()
        bad |= {r["id"] for r in t if r["post_falhas"] or r["post_marcas"]}
    return np.array([i not in bad for i in ids], dtype=bool)


def open_work(n: int, mode: str) -> dict[str, np.memmap]:
    shapes = {"sig": ((n, PERMS), np.uint32), "bands": ((n, BANDS), np.uint64), "exact": ((n,), np.uint64),
              "scope": ((n,), np.uint64), "idhash": ((n,), np.uint64), "flags": ((n, 3), np.uint8)}
    return {k: np.memmap(WORK_DIR / f"{k}.bin", dtype=dt, mode=mode, shape=shape) for k, (shape, dt) in shapes.items()}


def sign_part(task: tuple[str, str, int, int]) -> int:
    fonte, path, start, n = task
    path = Path(path)
    t = pq.read_table(path, columns=["id", "task", "nl", "lang", "prompt", "code", "response"]).to_pylist()
    for r in t:  # o pedido é o texto mais o código de entrada (T2, T5, T6, T7)
        r["input"] = (r["prompt"] or "") + ("\n" + r["code"] if r["code"] else "")
    assert len(t) == n, (path, len(t), n)
    work = open_work(N_TOTAL, "r+")
    sl = slice(start, start + n)
    scope = np.array([h64(f"{r['task']}|{r['nl']}|{r['lang']}") for r in t], dtype=np.uint64)
    exact = np.array([h64(f"{r['task']}|{r['nl']}|{r['lang']}\0{norm(r['input'])}\0{norm(r['response'])}")
                      for r in t], dtype=np.uint64)
    near = np.array([r["task"] not in EXACT_ONLY for r in t], dtype=bool)
    sig = np.stack([minhash(r["input"]) if ok else np.zeros(PERMS, np.uint32) for r, ok in zip(t, near)])
    flags = np.zeros((n, 3), dtype=np.uint8)
    flags[:, 0] = clean_flags(fonte, path, [r["id"] for r in t])
    flags[:, 1] = TIERS.get(fonte, SYNTHETIC)
    flags[:, 2] = near
    work["sig"][sl] = sig
    work["bands"][sl] = band_keys(sig, scope)
    work["exact"][sl] = exact
    work["scope"][sl] = scope
    work["idhash"][sl] = [h64(r["id"]) for r in t]
    work["flags"][sl] = flags
    for m in work.values():
        m.flush()
    return n


N_TOTAL = 0


def _init(n: int) -> None:
    global N_TOTAL
    N_TOTAL = n


def sign_all(workers: int) -> None:
    ps, offsets = layout()
    n = int(offsets[-1])
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    open_work(n, "w+")  # cria os arquivos com o tamanho final
    (WORK_DIR / "layout.json").write_text(json.dumps({"partes": [[f, str(p)] for f, p in ps],
                                                      "offsets": offsets.tolist()}))
    tasks = [(f, str(p), int(offsets[i]), int(offsets[i + 1] - offsets[i])) for i, (f, p) in enumerate(ps)]
    tasks.sort(key=lambda x: -x[3])  # as maiores primeiro, para não sobrar uma parte longa no fim
    start, done = time.time(), 0
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(n,)) as pool:
        for i, k in enumerate(pool.map(sign_part, tasks), 1):
            done += k
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} partes, {done:,}/{n:,} exemplos, {time.time() - start:.0f}s", flush=True)


def find(parent: np.ndarray, i: int) -> int:
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:
        parent[i], i = root, parent[i]
    return root


def group() -> None:
    lay = json.loads((WORK_DIR / "layout.json").read_text())
    offsets = np.array(lay["offsets"])
    n = int(offsets[-1])
    work = open_work(n, "r")
    start = time.time()

    # Exatas: rótulo = menor índice com a mesma chave (a chave já inclui task e nl).
    exact = np.asarray(work["exact"])
    order = np.argsort(exact, kind="stable")
    sorted_keys = exact[order]
    new_run = np.concatenate([[True], sorted_keys[1:] != sorted_keys[:-1]])
    run_first = order[np.maximum.accumulate(np.where(new_run, np.arange(n), 0))]
    parent = np.empty(n, dtype=np.int64)
    parent[order] = run_first
    n_exact = int(n - new_run.sum())
    print(f"  exatas: {n_exact:,} duplicatas, {time.time() - start:.0f}s", flush=True)
    del order, sorted_keys, new_run, run_first

    # Próximas: candidatos com alguma banda igual, confirmados contra o primeiro do bloco.
    near = np.asarray(work["flags"][:, 2]).astype(bool)
    sig = work["sig"]
    pairs = 0
    for b in range(BANDS):
        keys = np.asarray(work["bands"][:, b])
        idx = np.flatnonzero(near)
        idx = idx[np.argsort(keys[idx], kind="stable")]
        k = keys[idx]
        boundaries = np.flatnonzero(np.concatenate([[True], k[1:] != k[:-1], [True]]))
        sizes = np.diff(boundaries)
        for s, size in zip(boundaries[:-1][sizes > 1], sizes[sizes > 1]):
            members = np.sort(idx[s:s + size])
            rep = members[0]
            same = (sig[members[1:]] == sig[rep]).mean(axis=1) >= THRESHOLD
            ra = find(parent, int(rep))
            for m in members[1:][same]:
                rm = find(parent, int(m))
                if rm != ra:
                    parent[max(ra, rm)] = min(ra, rm)
                    ra = min(ra, rm)
                    pairs += 1
        print(f"  banda {b + 1}/{BANDS}: {pairs:,} uniões, {time.time() - start:.0f}s", flush=True)

    # Raízes de todos, vetorizado: repete parent[parent] até estabilizar.
    roots = parent.copy()
    while True:
        nxt = roots[roots]
        if np.array_equal(nxt, roots):
            break
        roots = nxt

    # Quem fica: limpo, depois a prioridade da fonte, depois o hash do id.
    flags = np.asarray(work["flags"])
    order = np.lexsort((np.asarray(work["idhash"]), flags[:, 1], 1 - flags[:, 0], roots))
    # Estrela, não cadeia: A~B~C juntava variantes distantes (23,8 mil "palíndromos" com
    # similaridade mediana de 0,54 com o que ficava). Em cada componente, o melhor fica, viram
    # duplicata dele só os com similaridade >= THRESHOLD, e o resto recomeça.
    keeper = np.arange(n, dtype=np.int64)
    r = roots[order]
    bounds = np.flatnonzero(np.concatenate([[True], r[1:] != r[:-1], [True]]))
    big = np.flatnonzero(np.diff(bounds) > 1)
    for gi, g in enumerate(big):
        rest = order[bounds[g]:bounds[g + 1]]
        rest_sig = np.asarray(sig[np.sort(rest)])[np.argsort(np.argsort(rest))]  # leitura em ordem no disco
        while len(rest) > 1:
            same = (rest_sig[1:] == rest_sig[0]).mean(axis=1) >= THRESHOLD
            keeper[rest[1:][same]] = rest[0]
            keep = np.concatenate([[False], ~same])
            rest, rest_sig = rest[keep], rest_sig[keep]
        if gi % 200_000 == 0:
            print(f"  estrelas: {gi:,}/{len(big):,} componentes, {time.time() - start:.0f}s", flush=True)
    is_dup = keeper != np.arange(n)
    exact_dup = is_dup & (exact == exact[keeper])
    print(f"  {is_dup.sum():,} duplicatas ({exact_dup.sum():,} exatas), {time.time() - start:.0f}s", flush=True)
    write_outputs(lay, offsets, keeper, is_dup, exact_dup)


def write_outputs(lay: dict, offsets: np.ndarray, keeper: np.ndarray, is_dup: np.ndarray, exact_dup: np.ndarray) -> None:
    ids = pa.chunked_array([pq.read_table(p, columns=["id"]).column("id").combine_chunks() for _, p in lay["partes"]])
    ids = ids.combine_chunks()
    schema = pa.schema([("id", pa.string()), ("dup_de", pa.string()), ("dup_tipo", pa.string())])
    stats = Counter()
    for i, (fonte, path) in enumerate(lay["partes"]):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        dup = is_dup[lo:hi]
        dup_de = ids.take(pa.array(keeper[lo:hi])).to_pylist()
        tipo = np.where(exact_dup[lo:hi], "exata", "proxima")
        table = pa.table({"id": ids.slice(lo, hi - lo),
                          "dup_de": [d if x else None for d, x in zip(dup_de, dup)],
                          "dup_tipo": [t if x else None for t, x in zip(tipo.tolist(), dup)]}, schema=schema)
        out = DEDUP_DIR / fonte / Path(path).name
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out, compression="zstd")
        stats[(fonte, "exemplos")] += hi - lo
        stats[(fonte, "exata")] += int((dup & exact_dup[lo:hi]).sum())
        stats[(fonte, "proxima")] += int((dup & ~exact_dup[lo:hi]).sum())
    report(stats)


def report(stats: Counter | None = None) -> None:
    path = DEDUP_DIR / "relatorio.json"
    if stats is None:
        stats = Counter({tuple(k.split("|")): v for k, v in json.loads(path.read_text()).items()})
    else:
        path.write_text(json.dumps({f"{f}|{k}": v for (f, k), v in sorted(stats.items())}, indent=1))
    fontes = sorted({f for f, _ in stats})
    print(f"{'fonte':<26}{'exemplos':>12}{'exata':>9}{'próxima':>9}")
    for f in fontes:
        n = stats[(f, "exemplos")]
        print(f"{f:<26}{n:>12,}{100 * stats[(f, 'exata')] / n:>8.2f}%{100 * stats[(f, 'proxima')] / n:>8.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("comando", nargs="?", choices=["assinaturas", "agrupar"])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--relatorio", action="store_true")
    args = parser.parse_args()
    if args.relatorio or not args.comando:
        report()
    elif args.comando == "assinaturas":
        sign_all(args.workers)
    else:
        group()


if __name__ == "__main__":
    main()
