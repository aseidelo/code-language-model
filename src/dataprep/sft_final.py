"""Fecha o SFT v2: mistura final, validação, pack e card do Hugging Face (ver docs/memory/sft-v2.md).

Mistura aprovada em 2026-11-22: entra **todo o dado limpo** nos 6 idiomas (en, pt, es, fr, de, it).
Limpo quer dizer sem ``qc_falhas``, ``post_falhas``, ``post_marcas``, ``dup_de`` nem ``contaminado``.
Os rótulos ``lang``/``nl`` corrigidos pelo pós-processamento substituem os originais. Há só dois pesos:

* WebSight em inglês: ``peso = 0.5`` (metade sorteada pelo id no pack);
* Bash: ``peso = 2`` (duas cópias no pack).

Comandos:

* ``montar``: grava ``data/codelm-sft-v2/`` (pasta pronta para ``hf upload``), com o texto de cada
  exemplo limpo, segredos e e-mails trocados por ``<SECRET>``/``<EMAIL>``, e a coluna ``peso``.
  ``data/train/<fonte>/<parte>.parquet`` e ``data/validation/validation.parquet``. A validação são
  ~0,07% dos **grupos**: um grupo junta o exemplo, as traduções dele e as outras tarefas da mesma
  função (T2/T5), para a mesma função não aparecer nos dois splits.
* ``pack``: tokeniza com o tokenizador v2 e o chat template, aplica os pesos e grava
  ``v2/pack/{train-NN,valid}.npz`` no formato do ``sft_pack`` (``tokens``, ``mask``, ``starts`` e
  etiquetas), em shards embaralhados. Linguagens sem token no template (``outra:*``) ficam só no
  texto.
* ``card``: escreve o ``README.md`` do dataset.

    uv run python -m dataprep.sft_final montar [--workers 12]
    uv run python -m dataprep.sft_final pack [--workers 12]
    uv run python -m dataprep.sft_final card
    uv run python -m dataprep.sft_final publicar   # só o texto e o README; retomável
"""

import argparse
import json
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash

from codelm.languages import DATA_DIR, LANGUAGES
from dataprep.publish import redact
from dataprep.sft_contam import CONTAM_DIR
from dataprep.sft_convert import FONTES_DIR, V2_DIR
from dataprep.sft_dedup import DEDUP_DIR
from dataprep.sft_post import POST_DIR
from dataprep.sft_qc import QC_DIR

OUT_DIR = DATA_DIR / "codelm-sft-v2"
PACK_DIR = V2_DIR / "pack"
NLS = ("en", "pt", "es", "fr", "de", "it")
VALID_PER_10K = 7
MAX_TOKENS = 8192
SHARDS = 24
LABELS = ("task", "lang", "nl", "level", "fonte")
TEXT = ("prompt", "code", "response", "prompt_en")
COLUMNS = {
    "id": pa.string(), "grupo": pa.string(), "fonte": pa.string(), "fonte_original": pa.string(),
    "source": pa.string(), "task": pa.string(), "level": pa.string(), "lang": pa.string(), "nl": pa.string(),
    "kind": pa.string(), "lang_in_prompt": pa.bool_(), "prompt": pa.string(), "code": pa.string(),
    "response": pa.string(), "prompt_en": pa.string(), "tests": pa.string(), "licenca": pa.string(),
    "repo": pa.string(), "path": pa.string(), "licenses": pa.string(), "n_tokens": pa.int64(), "peso": pa.float32(),
}
NL_SUFFIX = re.compile(r"-(pt|es|fr|de|it)$")


def h64(text: str) -> int:
    return xxhash.xxh64_intdigest(text.encode("utf-8", "surrogatepass"))


def group_of(row: dict) -> str:
    """Exemplo, traduções dele e as tarefas irmãs da mesma função (T2-x e T5-x) caem no mesmo grupo."""
    gid = re.sub(r"^T\d+-", "", row["id"])
    return NL_SUFFIX.sub("", gid) if row["fonte"].startswith("traducao") else gid


def weight(row: dict) -> float:
    if row["lang"] == "bash":
        return 2.0
    if row["fonte"] == "websight" and row["nl"] == "en":
        return 0.5
    return 1.0


def side(sub: Path, fonte: str, name: str, cols: list[str], n: int, ids: list[str]) -> list[dict] | None:
    path = sub / fonte / name
    if not path.exists():
        return None
    t = pq.read_table(path, columns=["id", *cols])
    assert t.num_rows == n and t.column("id").to_pylist() == ids, f"{path} não bate com a fonte"
    return t.to_pylist()


def clean_rows(task: tuple[str, str]) -> tuple[str, list[dict], Counter]:
    fonte, path = task
    path = Path(path)
    names = pq.read_schema(path).names
    rows = pq.read_table(path, columns=[c for c in COLUMNS if c in names]).to_pylist()
    ids = [r["id"] for r in rows]
    n = len(rows)
    qc = side(QC_DIR, fonte, path.name, ["qc_falhas"], n, ids)
    post = side(POST_DIR, fonte, path.name, ["post_falhas", "post_marcas", "lang_corrigido", "nl_corrigido"], n, ids)
    dup = side(DEDUP_DIR, fonte, path.name, ["dup_de"], n, ids)
    contam = side(CONTAM_DIR, fonte, path.name, ["contaminado"], n, ids)
    assert post is not None and dup is not None and contam is not None, f"falta pós, dedup ou contaminação em {path}"
    stats, out = Counter(), []
    for i, r in enumerate(rows):
        if qc and qc[i]["qc_falhas"]:
            stats["qc"] += 1
            continue
        p = post[i]
        if p["post_falhas"] or p["post_marcas"]:
            stats["pos"] += 1
            continue
        if contam[i]["contaminado"]:
            stats["contaminado"] += 1
            continue
        if dup[i]["dup_de"]:
            stats["duplicata"] += 1
            continue
        r["lang"] = p["lang_corrigido"] or r["lang"]
        r["nl"] = p["nl_corrigido"] or r["nl"]
        if r["nl"] not in NLS:
            stats["idioma"] += 1
            continue
        r["fonte"] = fonte
        for k in TEXT:
            if r.get(k):
                r[k] = redact(r[k], stats)
        if not isinstance(r.get("tests"), (str, type(None))):
            r["tests"] = json.dumps(r["tests"])
        r["grupo"] = group_of(r)
        r["peso"] = weight(r)
        r["n_tokens"] = int(r["n_tokens"]) if r.get("n_tokens") is not None else None
        out.append({k: r.get(k) for k in COLUMNS})
        stats["ok"] += 1
    return path.name, out, stats


def is_valid(grupo: str) -> bool:
    return h64("valid:" + grupo) % 10_000 < VALID_PER_10K


def build_part(task: tuple[str, str]) -> tuple[list[dict], Counter, Counter]:
    """Grava o train da parte e devolve só a validação e as contagens (o pai não segura o texto)."""
    fonte = task[0]
    name, rows, stats = clean_rows(task)
    train = [r for r in rows if not is_valid(r["grupo"])]
    valid = [r for r in rows if is_valid(r["grupo"])]
    if train:
        out = OUT_DIR / "data" / "train" / fonte / name
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(train, schema=pa.schema(list(COLUMNS.items()))), out, compression="zstd")
    per = Counter()
    for r in train:
        per[(fonte, r["task"], r["lang"], r["nl"], "train")] += 1
        per[(fonte, r["task"], r["lang"], r["nl"], "train_tokens")] += r["n_tokens"] or 0
    return valid, stats, per


def build(workers: int) -> None:
    shutil.rmtree(OUT_DIR / "data", ignore_errors=True)
    tasks = [(f.name, str(p)) for f in sorted(FONTES_DIR.iterdir()) if f.is_dir() for p in sorted(f.glob("part-*.parquet"))]
    schema = pa.schema(list(COLUMNS.items()))
    stats, per, valid, start = Counter(), Counter(), [], time.time()
    with ProcessPoolExecutor(workers) as pool:
        futures = []
        for i, task in enumerate(tasks, 1):
            futures.append(pool.submit(build_part, task))
            if len(futures) >= 2 * workers or i == len(tasks):  # janela pequena: no máximo 2 partes por worker em voo
                for f in futures:
                    v, s, c = f.result()
                    valid += v
                    stats.update(s)
                    per.update(c)
                futures = []
                if i % 240 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} partes, {stats['ok']:,} exemplos, {time.time() - start:.0f}s", flush=True)
    out = OUT_DIR / "data" / "validation" / "validation.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(valid, schema=schema), out, compression="zstd")
    for r in valid:
        per[(r["fonte"], r["task"], r["lang"], r["nl"], "validation")] += 1
        per[(r["fonte"], r["task"], r["lang"], r["nl"], "validation_tokens")] += r["n_tokens"] or 0
    summary = {"exclusoes": dict(stats), "celulas": [[*k, v] for k, v in sorted(per.items())]}
    (OUT_DIR / "stats.json").write_text(json.dumps(summary))
    print(json.dumps(dict(stats), indent=1))
    print(f"validação: {len(valid):,} exemplos")


# --- pack --------------------------------------------------------------------

def vocab() -> dict[str, list[str]]:
    cells = json.loads((OUT_DIR / "stats.json").read_text())["celulas"]
    names = {k: set() for k in LABELS}
    for fonte, task, lang, nl, *_ in cells:
        names["fonte"].add(fonte)
        names["task"].add(task)
        names["lang"].add(lang)
        names["nl"].add(nl)
    names["lang"] = set(LANGUAGES)  # o pack só codifica as linguagens com token no template
    names["level"] = {"facil", "medio", "dificil"}
    return {k: sorted(v) + ["?"] for k, v in names.items()}  # "?" para valor fora da lista


def copies(row: dict) -> list[int]:
    """Shards das cópias do exemplo, já com o peso aplicado."""
    if row["peso"] == 0.5 and h64("peso:" + row["id"]) % 2:
        return []
    keys = [row["id"]] if row["peso"] <= 1 else [row["id"], row["id"] + "#2"]
    return [h64("shard:" + k) % SHARDS for k in keys]


def encode_file(task: tuple[str, str, str]) -> Counter:
    from codelm.chat_template import encode_example
    from codelm.tokenizer import load_tokenizer

    path, split, tmp = task
    tokenizer = load_tokenizer()
    names = vocab()
    index = {k: {v: i for i, v in enumerate(names[k])} for k in names}
    # Tokens em arrays numpy (2 bytes/token): listas de int do Python custam ~36 bytes/token e, com
    # 10 workers, derrubaram a máquina por falta de memória em 2026-11-22.
    buckets: dict[int, dict[str, list]] = {}
    stats = Counter()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2000):
        for r in batch.to_pylist():
            if r["lang"] not in LANGUAGES:
                stats["sem_token_de_linguagem"] += 1
                continue
            shards = copies(r) if split == "train" else [0]
            if not shards:
                stats["peso_descartado"] += 1
                continue
            try:
                ids, mask = encode_example(tokenizer, r["prompt"] or "", r["response"] or "", r["lang"], r["kind"] or "code",
                                           code=r["code"] or None, lang_in_prompt=bool(r["lang_in_prompt"]))
            except Exception as e:  # um exemplo que o template não aceita não derruba o pack
                stats["erro_codificacao"] += 1
                if stats["erro_codificacao"] <= 3:
                    print(f"  erro em {r['id']}: {e!r}"[:300], flush=True)
                continue
            if len(ids) > MAX_TOKENS:
                stats["longo"] += 1
                continue
            ids, mask = np.array(ids, np.uint16), np.array(mask, np.uint8)
            labels = {k: index[k].get(r.get(k), index[k]["?"]) for k in LABELS}
            for s in shards:
                b = buckets.setdefault(s, {"tokens": [], "mask": [], "lens": [], **{k: [] for k in LABELS}})
                b["tokens"].append(ids)
                b["mask"].append(mask)
                b["lens"].append(len(ids))
                for k in LABELS:
                    b[k].append(labels[k])
                stats["exemplos"] += 1
                stats["tokens"] += len(ids)
                stats["tokens_na_perda"] += int(mask.sum())
    for s, b in buckets.items():
        out = Path(tmp) / f"{split}-{s:02d}" / (Path(path).parent.name + "-" + Path(path).name.replace(".parquet", ".npz"))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, tokens=np.concatenate(b["tokens"]), mask=np.concatenate(b["mask"]),
                 lens=np.array(b["lens"], np.int32), **{k: np.array(b[k], np.uint8) for k in LABELS})
    return stats


def merge_shard(task: tuple[str, str]) -> str:
    """Junta os pedaços de um shard e embaralha a ordem dos exemplos."""
    chunk_dir, out = task
    parts = [np.load(p) for p in sorted(Path(chunk_dir).glob("*.npz"))]
    lens = np.concatenate([p["lens"] for p in parts])
    tokens = np.concatenate([p["tokens"] for p in parts])
    mask = np.concatenate([p["mask"] for p in parts])
    labels = {k: np.concatenate([p[k] for p in parts]) for k in LABELS}
    starts = np.concatenate([[0], np.cumsum(lens)])
    order = np.random.default_rng(h64(out) % 2**32).permutation(len(lens))
    new_starts = np.concatenate([[0], np.cumsum(lens[order])]).astype(np.int64)
    new_tokens, new_mask = np.empty_like(tokens), np.empty_like(mask)
    for j, i in enumerate(order):  # cópia por exemplo: um índice int64 do shard inteiro passaria de 4 GB
        new_tokens[new_starts[j]:new_starts[j + 1]] = tokens[starts[i]:starts[i + 1]]
        new_mask[new_starts[j]:new_starts[j + 1]] = mask[starts[i]:starts[i + 1]]
    np.savez(out, tokens=new_tokens, mask=new_mask, starts=new_starts, **{k: v[order] for k, v in labels.items()})
    return out


def pack(workers: int) -> None:
    tmp = PACK_DIR / "tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    files = [(str(p), "train", str(tmp)) for p in sorted((OUT_DIR / "data" / "train").glob("*/*.parquet"))]
    files.append((str(OUT_DIR / "data" / "validation" / "validation.parquet"), "valid", str(tmp)))
    stats, start = {"train": Counter(), "valid": Counter()}, time.time()
    with ProcessPoolExecutor(workers) as pool:
        for i, (task, s) in enumerate(zip(files, pool.map(encode_file, files, chunksize=2)), 1):
            stats[task[1]].update(s)
            if i % 200 == 0 or i == len(files):
                print(f"  {i}/{len(files)} arquivos, {stats['train']['tokens'] / 1e9:.2f}B tokens, "
                      f"{time.time() - start:.0f}s", flush=True)
    merges = [(str(tmp / f"train-{s:02d}"), str(PACK_DIR / f"train-{s:02d}.npz")) for s in range(SHARDS)]
    merges.append((str(tmp / "valid-00"), str(PACK_DIR / "valid.npz")))
    with ProcessPoolExecutor(min(workers, 2)) as pool:
        for out in pool.map(merge_shard, merges):
            print(f"  {out}", flush=True)
    shutil.rmtree(tmp)
    meta = {"labels": vocab(), "max_tokens": MAX_TOKENS, "shards": SHARDS, "tokenizer": "v2",
            "stats": {k: dict(v) for k, v in stats.items()}}
    (PACK_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["stats"], indent=1))


HF_REPO = "seidel/codelm-sft-v2"


def publish() -> None:
    """Envia ``OUT_DIR`` ao HF (repositório privado e condicionado, já criado), uma fonte por commit.

    Retomável: pula o que já está no repositório. O pack tokenizado nunca é enviado.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    done = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    dirs = sorted(p for p in (OUT_DIR / "data" / "train").iterdir() if p.is_dir()) + [OUT_DIR / "data" / "validation"]
    for d in dirs:
        rel = d.relative_to(OUT_DIR).as_posix()
        missing = [f.relative_to(OUT_DIR).as_posix() for f in sorted(d.glob("*.parquet"))
                   if f.relative_to(OUT_DIR).as_posix() not in done]
        for i in range(0, len(missing), 400):
            chunk = missing[i:i + 400]
            start = time.time()
            api.upload_folder(repo_id=HF_REPO, repo_type="dataset", folder_path=str(OUT_DIR), allow_patterns=chunk,
                              commit_message=f"{rel} ({i + len(chunk)}/{len(missing)})")
            print(f"  {rel}: {i + len(chunk)}/{len(missing)} em {time.time() - start:.0f}s", flush=True)
    api.upload_folder(repo_id=HF_REPO, repo_type="dataset", folder_path=str(OUT_DIR),
                      allow_patterns=["README.md", "stats.json"], commit_message="README e estatísticas")
    print("upload concluído", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("comando", choices=["montar", "pack", "card", "publicar"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.comando == "montar":
        build(args.workers)
    elif args.comando == "pack":
        pack(args.workers)
    elif args.comando == "publicar":
        publish()
    else:
        from dataprep.sft_card import write_card
        write_card()


if __name__ == "__main__":
    main()
