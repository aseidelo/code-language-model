"""Fontes públicas do SFT v2: download cru, uma pasta por fonte (ver docs/sft-v2.md).

Cada fonte vai para ``data/codelm_sft/v2/raw/<fonte>/``, sem nenhum filtro: os
parquets do Hugging Face (só as colunas úteis, lidas direto do conversor de parquet
do Hub) ou um clone do repositório. A conversão para o schema do banco é um passo
separado. Retomável: arquivos já baixados são pulados.

    uv run python -m dataprep.sft_sources download [fonte ...]
    uv run python -m dataprep.sft_sources list
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from huggingface_hub.utils import get_session

from dataprep.sft_mix import SFT_DIR

V2_DIR = SFT_DIR / "v2"
RAW_DIR = V2_DIR / "raw"
DOWNLOAD_THREADS = 4


def hf(name, repo, licenca, configs=None, splits=None, columns=None, drop=(), threads=DOWNLOAD_THREADS, files=None):
    """``files``: glob dos parquets originais do repositório, quando o conversor do Hub só tem parte."""
    return dict(name=name, kind="hf", repo=repo, licenca=licenca, configs=configs, splits=splits,
                columns=columns, drop=tuple(drop), threads=threads, files=files)


SOURCES = [
    # T1: pedido -> código
    *[hf(f"opc-stage1-{short}", "OpenCoder-LLM/opc-sft-stage1", "MIT", configs=[config])
      for short, config in (("infinity", "filtered_infinity_instruct"), ("largescale", "largescale_diverse_instruct"),
                            ("realuser", "realuser_instruct"))],
    *[hf(f"opc-stage2-{short}", "OpenCoder-LLM/opc-sft-stage2", "MIT", configs=[config])
      # evol_instruct e mceval_instruct são cópias de magicoder-evol e mceval-instruct (100% dos prompts).
      for short, config in (("educational", "educational_instruct"), ("package", "package_instruct"))],
    hf("magicoder-oss", "ise-uiuc/Magicoder-OSS-Instruct-75K", "MIT"),
    hf("magicoder-evol", "ise-uiuc/Magicoder-Evol-Instruct-110K", "Apache-2.0"),
    hf("self-oss-instruct", "bigcode/self-oss-instruct-sc2-exec-filter-50k", "ODC-BY"),
    hf("code-alpaca", "sahil2801/CodeAlpaca-20k", "CC-BY-4.0"),
    hf("codefeedback", "m-a-p/CodeFeedback-Filtered-Instruction", "Apache-2.0"),
    hf("glaive-v3", "glaiveai/glaive-code-assistant-v3", "Apache-2.0"),
    hf("mceval-instruct", "Multilingual-Multimodal-NLP/McEval-Instruct", "CC-BY-SA-4.0"),
    hf("rosetta-code", "christopher/rosetta-code", "GFDL"),
    # T1 com testes (exercícios, T6, GRPO)
    hf("kodcode", "KodCode/KodCode-V1", "CC-BY-NC-4.0", splits=["train"], drop=["trials"]),
    # O conversor do Hub só tem 1,4M de 5M (partial-train): os parquets originais.
    hf("open-code-instruct", "nvidia/OpenCodeInstruct", "CC-BY-4.0", files="data/train-*.parquet"),
    hf("acecode-87k", "TIGER-Lab/AceCode-87K", "MIT"),
    hf("acecode-v2", "TIGER-Lab/AceCode-V2-122K", "Apache-2.0"),
    # T2/T5: docstring <-> código
    hf("code-search-net", "code-search-net/code_search_net", "várias (por repositório)",
       configs=["python", "java", "javascript"]),
    # Bash
    hf("bash-instruct-iii", "Frost2o24/bash-instruct-III-55k", "MIT"),
    hf("bash-command-6k", "emirkaanozdemr/bash_command_data_6K", "Apache-2.0"),
    hf("bash-commands", "aelhalili/bash-commands-dataset", "MIT"),
    hf("nl2bash", "jiacheng-ye/nl2bash", "GPL-3.0"),
    hf("nl2bash-custom", "AnishJoshi/nl2bash-custom", "GPL-3.0 (derivado do NL2Bash)"),
    dict(name="tldr-pages", kind="git", repo="https://github.com/tldr-pages/tldr.git", licenca="CC-BY-4.0"),
    # Por último: 738 parquets de ~390 MB, quase tudo screenshots. Só as colunas de texto são lidas
    # (range requests), mas cada arquivo ainda demora; mais threads compensam a latência.
    hf("websight", "HuggingFaceM4/WebSight", "CC-BY-4.0", configs=["v0.2"], columns=["llm_generated_idea", "text"],
       threads=8),
]
BY_NAME = {s["name"]: s for s in SOURCES}


def parquet_files(spec: dict) -> list[dict]:
    """Parquets do conversor do Hub (refs/convert/parquet) para os configs e splits pedidos,
    ou os originais do repositório (``files``)."""
    if spec.get("files"):
        paths = sorted(HfFileSystem().glob(f"datasets/{spec['repo']}/{spec['files']}"))
        return [{"config": "default", "split": "train", "filename": p.rsplit("/", 1)[1], "path": p} for p in paths]
    response = get_session().get("https://datasets-server.huggingface.co/parquet",
                                 params={"dataset": spec["repo"]}, timeout=60)
    response.raise_for_status()
    files = response.json()["parquet_files"]
    if spec["configs"]:
        files = [f for f in files if f["config"] in spec["configs"]]
    if spec["splits"]:
        files = [f for f in files if f["split"] in spec["splits"]]
    if any("partial-" in f["url"] for f in files):
        raise RuntimeError(f"{spec['name']}: o conversor do Hub só tem parte do dataset; use files=")
    if not files:
        raise RuntimeError(f"{spec['name']}: nenhum parquet para {spec['configs']} {spec['splits']}")
    return files


def download_file(fs: HfFileSystem, spec: dict, file: dict, out_dir: Path) -> int:
    out = out_dir / f"{file['config']}-{file['split']}-{file['filename']}"
    if out.exists():
        return pq.read_metadata(out).num_rows
    path = file.get("path") or \
        f"datasets/{spec['repo']}@refs%2Fconvert%2Fparquet/{file['config']}/{file['split']}/{file['filename']}"
    for attempt in range(5):
        try:
            with fs.open(path, "rb", block_size=16 * 2**20) as handle:
                schema = pq.read_schema(handle)
                columns = spec["columns"] or [c for c in schema.names if c not in spec["drop"]]
                table = pq.read_table(handle, columns=columns)
            break
        except Exception as error:  # rede instável: tenta de novo
            if attempt == 4:
                raise
            print(f"    {out.name}: {type(error).__name__}, tentando de novo", flush=True)
            time.sleep(10 * (attempt + 1))
    tmp = out.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.rename(out)
    return table.num_rows


def download_hf(spec: dict, out_dir: Path) -> dict:
    files = parquet_files(spec)
    fs = HfFileSystem()
    with ThreadPoolExecutor(spec["threads"]) as pool:
        rows = list(pool.map(lambda f: download_file(fs, spec, f, out_dir), files))
    counts: dict[str, int] = {}
    for file, n in zip(files, rows):
        key = f"{file['config']}/{file['split']}"
        counts[key] = counts.get(key, 0) + n
    return {"rows": counts}


def download_git(spec: dict, out_dir: Path) -> dict:
    repo_dir = out_dir / "repo"
    if not repo_dir.exists():
        subprocess.run(["git", "clone", "--depth", "1", spec["repo"], str(repo_dir)], check=True)
    commit = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    return {"commit": commit}


def download(names: list[str]) -> None:
    for name in names:
        spec = BY_NAME[name]
        out_dir = RAW_DIR / name
        meta_path = out_dir / "meta.json"
        if meta_path.exists():
            print(f"{name}: já baixado")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{name}: baixando {spec['repo']}", flush=True)
        start = time.time()
        try:
            info = download_hf(spec, out_dir) if spec["kind"] == "hf" else download_git(spec, out_dir)
        except Exception as error:  # uma fonte fora do ar não para as outras
            print(f"{name}: erro: {type(error).__name__}: {str(error)[:300]}", flush=True)
            continue
        meta = {k: v for k, v in spec.items() if v not in (None, ())} | info | {
            "baixado_em": time.strftime("%Y-%m-%d %H:%M:%S")}
        meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        print(f"{name}: ok em {time.time() - start:.0f}s, {info}", flush=True)


def list_sources() -> None:
    for spec in SOURCES:
        meta_path = RAW_DIR / spec["name"] / "meta.json"
        status = "baixado" if meta_path.exists() else "-"
        rows = json.loads(meta_path.read_text()).get("rows", {}) if meta_path.exists() else {}
        print(f"{spec['name']:<22}{spec['licenca']:<28}{status:<9}{sum(rows.values()):>11,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    download_cmd = commands.add_parser("download")
    download_cmd.add_argument("names", nargs="*", help="fontes (padrão: todas)")
    commands.add_parser("list")
    args = parser.parse_args()
    if args.command == "download":
        unknown = set(args.names) - set(BY_NAME)
        if unknown:
            parser.error(f"fontes desconhecidas: {sorted(unknown)}; conhecidas: {list(BY_NAME)}")
        download(args.names or list(BY_NAME))
    else:
        list_sources()


if __name__ == "__main__":
    main()
