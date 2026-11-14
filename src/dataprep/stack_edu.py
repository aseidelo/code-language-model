"""Stack-Edu: código do The Stack v2 filtrado por valor didático (HuggingFaceTB/stack-edu).

O dataset traz só metadados: um parquet por linguagem (ou alguns) com ``blob_id``,
caminho, repositório, nota do classificador e licenças. O conteúdo de cada
arquivo vem do bucket S3 público da Software Heritage, por HTTPS anônimo e sem
conta da AWS (``CONTENT_HOST``, gzip): uma requisição por arquivo, então o
download é paralelo.

Os metadados ficam em ``data/stack-edu/meta/<config>/``. Ver ``docs/dados-v2.md``.
"""

import gzip
import http.client
import json
import random
import threading
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from codelm.languages import DATA_DIR

REPO = "HuggingFaceTB/stack-edu"
META_DIR = DATA_DIR / "stack-edu" / "meta"
META_COLUMNS = ("blob_id", "repo_name", "path", "src_encoding", "length_bytes", "score", "int_score",
                "detected_licenses", "license_type")
# O Shell do Stack-Edu tem de tudo (zsh, PKGBUILD, arquivos sem extensão): do bash,
# só scripts e arquivos de configuração do bash. Shebangs de outros shells são
# rejeitados depois, em ``dataprep.download.rejection_reason``.
BASH_EXTENSIONS = (".sh", ".bash", ".bashrc", ".bash_profile")


def metadata_files(lang: str) -> list[Path]:
    """Baixa (uma vez) os parquets de metadados da linguagem."""
    config = CONFIGS[lang]
    names = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith(f"{config}/"))
    return [Path(hf_hub_download(REPO, name, repo_type="dataset", local_dir=META_DIR)) for name in names]


def accepts(lang: str, path: str) -> bool:
    """Filtro pelo caminho, antes de baixar o conteúdo."""
    if lang == "bash":
        return path.lower().endswith(BASH_EXTENSIONS)
    return True


def iter_metadata(lang: str) -> Iterator[dict]:
    for path in metadata_files(lang):
        for batch in pq.ParquetFile(path).iter_batches(columns=list(META_COLUMNS)):
            for row in batch.to_pylist():
                if accepts(lang, row["path"]):
                    yield row


CONTENT_HOST = "softwareheritage.s3.amazonaws.com"
# Uma conexão HTTPS por thread, reaproveitada entre arquivos: sem isso, cada arquivo
# paga um handshake TLS até os EUA, e o download fica em ~110 arquivos/s em vez de ~800.
_connection = threading.local()


def fetch(row: dict, retries: int = 4) -> str | None:
    """Conteúdo de um arquivo; None se não deu para baixar."""
    for attempt in range(retries):
        conn = getattr(_connection, "conn", None)
        if conn is None:
            conn = _connection.conn = http.client.HTTPSConnection(CONTENT_HOST, timeout=60)
        try:
            conn.request("GET", f"/content/{row['blob_id']}")
            response = conn.getresponse()
            data = response.read()
            if response.status == 404:
                return None
            if response.status == 200:
                return gzip.decompress(data).decode(row.get("src_encoding") or "utf-8", errors="replace")
        except (OSError, EOFError, LookupError, http.client.HTTPException):
            conn.close()
            _connection.conn = None
        time.sleep(2**attempt)
    return None


def as_stack_row(row: dict, content: str) -> dict:
    """No formato das linhas do The Stack que ``dataprep.download.clean_file`` lê."""
    name = row["path"].rsplit("/", 1)[-1]
    return {
        "content": content,
        "ext": name.rsplit(".", 1)[-1] if "." in name else "",
        "max_stars_repo_name": row["repo_name"],
        "max_stars_repo_path": row["path"],
        "max_stars_repo_licenses": list(row.get("detected_licenses") or []),
        "score": row.get("score"),
        "license_type": row.get("license_type"),
    }


def fetch_many(rows: Iterable[dict], workers: int = 256) -> Iterator[dict]:
    """Baixa os conteúdos em paralelo, na ordem de ``rows``; pula os que falharam."""
    with ThreadPoolExecutor(workers) as pool:
        for row, content in zip(rows, pool.map(fetch, rows)):
            if content is not None:
                yield as_stack_row(row, content)


def sample(lang: str, size: int, seed: int = 0) -> list[dict]:
    """Amostra aleatória de ``size`` arquivos da linguagem, com conteúdo.

    Amostragem por reservatório: guarda só os ``size`` escolhidos, não os milhões
    de linhas de metadados.
    """
    rng, chosen = random.Random(seed), []
    for i, row in enumerate(iter_metadata(lang)):
        if i < size:
            chosen.append(row)
        elif (j := rng.randrange(i + 1)) < size:
            chosen[j] = row
    return list(fetch_many(chosen))


# --- download em escala (pré-treino v2) -----------------------------------------

FILTERED_DIR = DATA_DIR / "stack-edu" / "filtered"
CHUNK_ROWS = 20_000  # linhas de metadados por parte (uma parte = um parquet de treino e um de validação)
# Conexões simultâneas por linguagem. O total importa: com ~1.800 (7 x 256) o roteador
# engasgou e as requisições ficaram presas sem erro (~300 arquivos/s no total); 256 numa
# linguagem só dão ~800/s. O Python, com a maior parte dos arquivos, fica com mais.
WORKERS = {"python": 256}
DEFAULT_WORKERS = 64
# Config do Stack-Edu de cada linguagem do modelo.
CONFIGS = {"python": "Python", "javascript": "JavaScript", "typescript": "TypeScript", "java": "Java",
           "c": "C", "rust": "Rust", "bash": "Shell"}


def shuffled_groups(lang: str, seed: int) -> list[tuple[Path, int]]:
    """(parquet, row group) em ordem aleatória: embaralha sem carregar milhões de linhas."""
    groups = [(path, i) for path in metadata_files(lang) for i in range(pq.ParquetFile(path).num_row_groups)]
    random.Random(seed).shuffle(groups)
    return groups


def iter_chunks(lang: str, seed: int) -> Iterator[list[dict]]:
    """Metadados em partes de ``CHUNK_ROWS`` linhas, na ordem aleatória e determinística de ``seed``."""
    rng, buffer = random.Random(seed), []
    for path, group in shuffled_groups(lang, seed):
        rows = [r for r in pq.ParquetFile(path).read_row_group(group, columns=list(META_COLUMNS)).to_pylist()
                if accepts(lang, r["path"])]
        rng.shuffle(rows)
        buffer += rows
        while len(buffer) >= CHUNK_ROWS:
            yield buffer[:CHUNK_ROWS]
            buffer = buffer[CHUNK_ROWS:]
    if buffer:
        yield buffer


def download(lang: str, target_chars: int, seed: int = 0, workers: int | None = None,
             shard: int = 0, shards: int = 1) -> dict:
    """Baixa e filtra arquivos da linguagem até ``target_chars`` caracteres de treino.

    Retomável: cada parte (``CHUNK_ROWS`` linhas de metadados) vira
    ``{train,valid}-NNNNN.parquet`` em ``data/stack-edu/filtered/<lang>/``, e
    ``state.json`` guarda quantas partes já foram processadas e os contadores. Ao
    retomar, as partes processadas são puladas (a ordem é determinística) e os
    hashes já aceitos são relidos, para a remoção de duplicatas continuar valendo.
    Termina com ``done.json``.

    Com ``shards > 1``, este processo fica só com as partes ``i`` em que
    ``i % shards == shard``, com estado próprio (``state-<shard>of<shards>.json``),
    e para em ``target_chars`` (a meta dele). Um único processo fica limitado pela
    CPU no Python (o filtro compila cada arquivo, e as threads de download disputam
    o interpretador); ver ``main``. As partes já feitas por ``state.json`` (sem
    divisão) são puladas.
    """
    from dataprep.download import clean_file  # evita import circular (download importa este módulo)

    workers = workers or WORKERS.get(lang, DEFAULT_WORKERS)
    out_dir = FILTERED_DIR / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if shards == 1 else f"-{shard}of{shards}"
    if (done := out_dir / f"done{suffix}.json").exists():
        return json.loads(done.read_text())
    state_file = out_dir / f"state{suffix}.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
    else:
        single = out_dir / "state.json"
        first = json.loads(single.read_text())["chunks"] if shards > 1 and single.exists() else 0
        state = {"chunks": first, "stats": {}}
    stats = Counter(state["stats"])
    seen = {h for p in out_dir.glob("*.parquet") for h in pq.read_table(p, columns=["sha256"])["sha256"].to_pylist()}

    start = time.time()
    for index, chunk in enumerate(iter_chunks(lang, seed)):
        if index < state["chunks"] or index % shards != shard:
            continue
        if stats["chars train"] >= target_chars:
            break
        splits: dict[str, list[dict]] = {"train": [], "valid": []}
        fetched = 0
        for row in fetch_many(chunk, workers):
            fetched += 1
            if file := clean_file(row, lang, seen, stats):
                file.update(score=row["score"], license_type=row["license_type"])
                splits[file.pop("split")].append(file)
        stats["baixar falhou"] += len(chunk) - fetched
        for split, files in splits.items():
            if files:
                pq.write_table(pa.Table.from_pylist(files), out_dir / f"{split}-{index:05d}.parquet")
            stats[split] += len(files)
            stats[f"chars {split}"] += sum(len(f["content"]) for f in files)
        state = {"chunks": index + 1, "stats": dict(stats)}
        state_file.write_text(json.dumps(state, indent=1))
        hours = (time.time() - start) / 3600
        print(f"{lang}{suffix}: parte {index} | {stats['chars train'] / 1e9:.2f} de {target_chars / 1e9:.2f} G caracteres "
              f"| {stats['lidos']:,} arquivos lidos | {hours:.2f} h nesta execução", flush=True)

    result = {"lang": lang, "target_chars": target_chars, **stats}
    done.write_text(json.dumps(result, indent=1))
    return result


def download_sharded(lang: str, shards: int, workers: int) -> dict:
    """Divide o resto da meta da linguagem entre ``shards`` processos e junta o resultado em ``done.json``."""
    from dataprep.pretrain import SEED, chars_per_token, target_chars

    out_dir = FILTERED_DIR / lang
    target = target_chars(lang, chars_per_token(lang))
    single = out_dir / "state.json"
    before = Counter(json.loads(single.read_text())["stats"]) if single.exists() else Counter()
    per_shard = max(0, target - before["chars train"]) // shards
    with ProcessPoolExecutor(shards) as pool:
        futures = [pool.submit(download, lang, per_shard, SEED, workers, k, shards) for k in range(shards)]
        results = [f.result() for f in futures]
    total = before
    for r in results:
        total.update({k: v for k, v in r.items() if isinstance(v, int) and k != "target_chars"})
    result = {"lang": lang, "target_chars": target, "shards": shards, **total}
    (out_dir / "done.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download do Stack-Edu de uma linguagem em vários processos.")
    parser.add_argument("lang", choices=sorted(CONFIGS))
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--workers", type=int, default=48, help="conexões por processo")
    args = parser.parse_args()
    result = download_sharded(args.lang, args.shards, args.workers)
    print(f"{args.lang}: {result['chars train'] / 1e9:.2f} de {result['target_chars'] / 1e9:.2f} G caracteres")


if __name__ == "__main__":
    main()
