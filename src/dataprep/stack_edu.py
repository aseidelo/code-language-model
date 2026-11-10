"""Stack-Edu: código do The Stack v2 filtrado por valor didático (HuggingFaceTB/stack-edu).

O dataset traz só metadados: um parquet por linguagem (ou alguns) com ``blob_id``,
caminho, repositório, nota do classificador e licenças. O conteúdo de cada
arquivo vem do bucket S3 público da Software Heritage, por HTTPS anônimo e sem
conta da AWS (``CONTENT_URL``, gzip): uma requisição por arquivo, então o
download é paralelo.

Os metadados ficam em ``data/stack-edu/meta/<config>/``. Ver ``docs/dados-v2.md``.
"""

import gzip
import random
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from codelm.languages import DATA_DIR, LANGUAGES

REPO = "HuggingFaceTB/stack-edu"
META_DIR = DATA_DIR / "stack-edu" / "meta"
CONTENT_URL = "https://softwareheritage.s3.amazonaws.com/content/{}"
META_COLUMNS = ("blob_id", "repo_name", "path", "src_encoding", "length_bytes", "score", "int_score",
                "detected_licenses", "license_type")
# O Shell do Stack-Edu tem de tudo (zsh, PKGBUILD, arquivos sem extensão): do bash,
# só scripts e arquivos de configuração do bash. Shebangs de outros shells são
# rejeitados depois, em ``dataprep.download.rejection_reason``.
BASH_EXTENSIONS = (".sh", ".bash", ".bashrc", ".bash_profile")


def metadata_files(lang: str) -> list[Path]:
    """Baixa (uma vez) os parquets de metadados da linguagem."""
    config = LANGUAGES[lang].stack_dir
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


def fetch(row: dict, retries: int = 4) -> str | None:
    """Conteúdo de um arquivo; None se não deu para baixar."""
    url = CONTENT_URL.format(row["blob_id"])
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = gzip.decompress(response.read())
            return data.decode(row.get("src_encoding") or "utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
        except (OSError, EOFError, LookupError):
            pass
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


def fetch_many(rows: Iterable[dict], workers: int = 64) -> Iterator[dict]:
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
