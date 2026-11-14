"""Pré-treino na nuvem: roda o ``train.pretrain`` numa GPU NVIDIA do Modal.

O treino é o mesmo do ``train.pretrain``, sem mudanças: este script só monta o
container (CUDA, as mesmas versões das bibliotecas do projeto, o ``src/`` e o
tokenizador) e um volume do Modal no lugar da pasta ``model/`` do projeto:

    volume "codelm"
    ├── pretrain-v2/           dados v2 (train.bin, valid.bin, ...), enviados com ``upload``
    ├── pretrain/              dados v1
    └── checkpoints/<rodada>/  checkpoints e metrics.jsonl, trazidos com ``download``

GPUs do Modal podem ser interrompidas pela plataforma. A função tenta de novo
sozinha e, se a rodada já tem ``latest/``, continua dele com ``--resume``.

Uso (uma vez: ``uv run modal setup`` para autenticar):
    uv run modal run src/train/modal_pretrain.py::upload               # sobe model/pretrain-v2 (~23 GB)
    uv run modal run src/train/modal_pretrain.py::upload --data pretrain   # o v1 (~2,3 GB)
    uv run modal run --detach src/train/modal_pretrain.py::main \\
        --args="--run ab-gemma3 --arch gemma3 --size small --tokens 1000000000 --micro-batch 8"
    uv run modal run src/train/modal_pretrain.py::download --run ab-gemma3   # -> model/checkpoints/ab-gemma3

``--detach`` mantém o treino rodando se o terminal fechar; o progresso aparece
no painel do Modal. A GPU sai de ``CODELM_GPU`` (padrão H100), por exemplo
``CODELM_GPU=A100-80GB uv run modal run ...``.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

REMOTE_ROOT = "/root/project"
# No container, este arquivo fica solto em /root/; o projeto está em REMOTE_ROOT.
PROJECT_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path(REMOTE_ROOT)
REMOTE_MODEL_DIR = f"{REMOTE_ROOT}/model"
VOLUME_NAME = "codelm"
GPU = os.environ.get("CODELM_GPU", "H100")
FLASH_ATTN_WHEEL = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
                    "flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")
TIMEOUT_HOURS = 24  # máximo do Modal por execução; rodadas maiores continuam com --resume

app = modal.App("codelm-pretrain")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    # Mesmas versões do uv.lock, com o torch para CUDA no lugar do xpu.
    .uv_pip_install(
        "torch==2.8.0",
        "torchao==0.13.0",
        "transformers==5.20.2",
        "tokenizers==0.23.3",
        "safetensors==0.8.0",
        "numpy==2.5.4",
        "pyarrow==26.0.0",
        "huggingface-hub==2.2.0",
    )
    # FlashAttention 2, wheel oficial pré-compilado para torch 2.8 + CUDA 12. Tem
    # janela local nativa: é o que deixa a Gemma 3 em 8k rápida (ver
    # train.attention_check). Os kernels do Hub (pacote kernels) só têm build para
    # torch >= 2.9.
    .uv_pip_install(FLASH_ATTN_WHEEL)
    .add_local_dir(PROJECT_ROOT / "src", f"{REMOTE_ROOT}/src", ignore=["**/__pycache__"])
    .add_local_dir(PROJECT_ROOT / "assets", f"{REMOTE_ROOT}/assets")
)


def run_name(args: list[str]) -> str:
    if "--run" not in args:
        raise SystemExit("faltou --run <nome> nos argumentos do treino")
    return args[args.index("--run") + 1]


@app.function(image=image, gpu=GPU, volumes={REMOTE_MODEL_DIR: volume},
              timeout=TIMEOUT_HOURS * 3600, retries=modal.Retries(max_retries=3, initial_delay=10.0))
def pretrain(args: list[str]) -> None:
    run = run_name(args)
    volume.reload()
    if Path(REMOTE_MODEL_DIR, "checkpoints", run, "latest").exists():
        print(f"rodada {run} já tem latest/: continuando com --resume")
        args = ["--run", run, "--resume"]
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    try:
        subprocess.run([sys.executable, "-m", "train.pretrain", *args], cwd=f"{REMOTE_ROOT}/src", check=True)
    finally:
        volume.commit()


@app.function(image=image, gpu=GPU, timeout=1800)
def attention_check(args: list[str]) -> None:
    subprocess.run([sys.executable, "-m", "train.attention_check", *args], cwd=f"{REMOTE_ROOT}/src", check=True)


@app.local_entrypoint()
def check_attention(args: str = "") -> None:
    """Correção e velocidade das implementações de atenção da Gemma 3 (``train.attention_check``)."""
    attention_check.remote(shlex.split(args))


@app.local_entrypoint()
def main(args: str) -> None:
    pretrain.remote(shlex.split(args))


@app.local_entrypoint()
def upload(data: str = "pretrain-v2") -> None:
    """Sobe ``model/<data>`` para o volume (substitui o que estiver lá; as partes de ``pack`` ficam de fora)."""
    source = PROJECT_ROOT / "model" / data
    files = sorted(p for p in source.iterdir() if p.is_file())
    print(f"enviando {len(files)} arquivos ({sum(p.stat().st_size for p in files) / 1e9:.2f} GB) para {VOLUME_NAME}:/{data}")
    with volume.batch_upload(force=True) as batch:
        for path in files:
            batch.put_file(path, f"/{data}/{path.name}")
    print("ok")


@app.local_entrypoint()
def download(run: str) -> None:
    """Traz ``checkpoints/<run>`` do volume para ``model/checkpoints/<run>``."""
    target = PROJECT_ROOT / "model" / "checkpoints"
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME_NAME, f"/checkpoints/{run}", str(target)], check=True)
    print(f"ok: {target / run}")
