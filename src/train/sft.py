"""Ajuste fino supervisionado sobre o pacote de ``dataprep.sft_pack``.

* Parte de um checkpoint do pré-treino (``--base``), com QAT ligado do início ao
  fim: os pesos já vieram treinados com ele.
* A perda vale só no que o modelo gera (a máscara de ``encode_example``), somada
  por token: exemplos e lotes pesam pelo que têm de resposta.
* Lotes por tamanho: cada exemplo cai no menor de ``BUCKETS`` em que cabe e cada
  lote tem ``micro_tokens`` células. Com poucas formas, o ``torch.compile`` não
  recompila, e o preenchimento à direita não precisa de máscara de atenção.
* Avaliação com a perda por tarefa, linguagem, idioma e nível.
* Checkpoints em ``checkpoints/<run>/``, como no pré-treino.

Uso:
    uv run python -m train.sft --run sft-v1
    uv run python -m train.sft --run teste --epochs 0.05     # rodada curta
    uv run python -m train.sft --run sft-v1 --resume
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from codelm.model import CONTEXT_LENGTH, PAD
from codelm.quantization import enable_qat
from codelm.tokenizer import load_tokenizer, use_tokenizer_of
from dataprep.sft_mix import SFT_DIR
from dataprep.sft_pack import LABELS
from train.pretrain import CHECKPOINT_DIR, _arg_type, learning_rate, make_optimizer, save_checkpoint
from train.smoke_test import pick_device, synchronize

BUCKETS = (256, 512, 1024, CONTEXT_LENGTH)
IGNORE = -100


@dataclass
class Config:
    base: str = "small-v1/step-04227"
    pack: str = "pack"  # pasta do pacote em sft/
    epochs: float = 2.0
    micro_tokens: int = 4096  # células por passada na GPU (2 sequências de 2048; 4 de 2048 estouram 12 GB)
    batch_tokens: int = 32_768  # células por passo do otimizador
    lr: float = 2e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 50
    decay_fraction: float = 0.2
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 200
    save_every: int = 500
    seed: int = 0
    compile: bool = True


class Examples:
    """Exemplos de um split, em lotes de ``micro_tokens`` células com forma fixa por tamanho."""

    def __init__(self, directory: Path, split: str, micro_tokens: int) -> None:
        self.directory = directory
        data = np.load(directory / f"{split}.npz")
        self.tokens, self.mask, self.starts = data["tokens"], data["mask"], data["starts"]
        self.labels = {key: data[key] for key in LABELS}
        lengths = np.diff(self.starts)
        self.buckets = {size: np.flatnonzero((lengths <= size) & (lengths > low))
                        for low, size in zip((0, *BUCKETS), BUCKETS)}
        self.micro_tokens = micro_tokens
        self.pad = load_tokenizer().token_to_id(PAD)

    def batches(self, rng: np.random.Generator | None) -> list[np.ndarray]:
        """Lotes de índices; com ``rng``, embaralhados e sem o resto de cada tamanho."""
        result = []
        for size, indices in self.buckets.items():
            rows = self.micro_tokens // size
            if rng is not None:
                indices = rng.permutation(indices)
            stop = len(indices) - len(indices) % rows if rng is not None else len(indices)
            result += [indices[i : i + rows] for i in range(0, stop, rows)]
        if rng is not None:
            rng.shuffle(result)
        return result

    def collate(self, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """(entrada, alvos): alvos deslocados, com ``IGNORE`` fora do que o modelo gera."""
        lengths = self.starts[indices + 1] - self.starts[indices]
        size = next(b for b in BUCKETS if lengths.max() <= b)
        rows = self.micro_tokens // size
        x = np.full((rows, size), self.pad, dtype=np.int64)
        y = np.full((rows, size), IGNORE, dtype=np.int64)
        for row, (i, length) in enumerate(zip(indices, lengths)):
            ids = self.tokens[self.starts[i] : self.starts[i] + length].astype(np.int64)
            x[row, : length] = ids
            y[row, : length] = np.where(self.mask[self.starts[i] : self.starts[i] + length] > 0, ids, IGNORE)
        return torch.from_numpy(x), torch.from_numpy(y[:, 1:])


class Epochs:
    """Sequência de lotes por época, determinística e retomável."""

    def __init__(self, examples: Examples, seed: int, epoch: int = 0, cursor: int = 0) -> None:
        self.examples, self.seed, self.epoch, self.cursor = examples, seed, epoch, cursor
        self.order = examples.batches(np.random.default_rng(seed + epoch))

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cursor == len(self.order):
            self.epoch, self.cursor = self.epoch + 1, 0
            self.order = self.examples.batches(np.random.default_rng(self.seed + self.epoch))
        self.cursor += 1
        return self.examples.collate(self.order[self.cursor - 1])


@torch.no_grad()
def evaluate(model, examples: Examples, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = np.zeros(len(examples.starts) - 1)
    counts = np.zeros(len(examples.starts) - 1)
    for indices in examples.batches(None):
        x, y = examples.collate(indices)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(input_ids=x).logits
        per_token = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), y.flatten(), ignore_index=IGNORE,
                                    reduction="none").view(y.shape)
        n = len(indices)
        losses[indices] = per_token[:n].sum(1).cpu().numpy()
        counts[indices] = (y[:n] != IGNORE).sum(1).cpu().numpy()
    model.train()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    result = {"valid": float(losses.sum() / counts.sum())}
    for key, labels in examples.labels.items():
        names = json.loads((examples.directory / "meta.json").read_text())["labels"][key]
        for i, name in enumerate(names):
            chosen = labels == i
            if chosen.any():
                result[f"{key} {name}"] = float(losses[chosen].sum() / counts[chosen].sum())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="nome da rodada (pasta dos checkpoints)")
    parser.add_argument("--resume", action="store_true", help="continua do checkpoint latest/ da rodada")
    for f in fields(Config):
        kind = float if f.name in ("epochs", "lr") else _arg_type(f.type)
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=kind, default=None)
    args = parser.parse_args()

    run_dir = CHECKPOINT_DIR / args.run
    state = {"step": 0, "epoch": 0, "cursor": 0}
    if args.resume:
        state = json.loads((run_dir / "latest" / "state.json").read_text())
        cfg = Config(**state["config"])
    else:
        if (run_dir / "latest").exists():
            parser.error(f"a rodada {args.run} já existe; use --resume ou outro nome")
        cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config) if getattr(args, f.name) is not None})
    run_dir.mkdir(parents=True, exist_ok=True)
    source = run_dir / "latest" if args.resume else CHECKPOINT_DIR / cfg.base
    use_tokenizer_of(source)  # o do modelo base (e o salvo nos checkpoints do SFT)

    device = pick_device()
    torch.manual_seed(cfg.seed)
    train = Examples(SFT_DIR / cfg.pack, "train", cfg.micro_tokens)
    valid = Examples(SFT_DIR / cfg.pack, "valid", cfg.micro_tokens)
    data = Epochs(train, cfg.seed, state["epoch"], state["cursor"])
    accum = cfg.batch_tokens // cfg.micro_tokens
    total_steps = int(cfg.epochs * len(data.order)) // accum

    model = AutoModelForCausalLM.from_pretrained(source).to(device)
    enable_qat(model.model.layers)
    optimizer = make_optimizer(model, cfg)
    if args.resume:
        optimizer.load_state_dict(torch.load(run_dir / "latest" / "optimizer.pt", map_location=device))
    compiled = torch.compile(model) if cfg.compile else model

    print(f"rodada {args.run} | {device} | base {source} | {len(train.starts) - 1:,} exemplos | "
          f"{len(data.order):,} lotes por época | {total_steps:,} passos de {accum} lotes")
    print(f"configuração: {asdict(cfg)}")
    metrics = (run_dir / "metrics.jsonl").open("a")

    def log(entry: dict) -> None:
        metrics.write(json.dumps(entry) + "\n")
        metrics.flush()

    def run_eval() -> None:
        result = evaluate(compiled, valid, device)
        print(f"  avaliação: valid {result['valid']:.3f} | " + " | ".join(
            f"{k.split()[1]} {v:.2f}" for k, v in result.items() if k.startswith(("task ", "level "))), flush=True)
        log({"step": state["step"], **result})

    if not args.resume:
        run_eval()
    model.train()
    start, window_cells, window_start = time.time(), 0, time.time()
    while state["step"] < total_steps:
        step = state["step"]
        lr = learning_rate(step, total_steps, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum, targets = torch.zeros((), device=device), 0
        for _ in range(accum):
            x, y = data.next()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = compiled(input_ids=x).logits
            generated = y != IGNORE  # só essas posições: a maior parte da memória do passo está nos logits
            loss = F.cross_entropy(logits[:, :-1][generated].float(), y[generated], reduction="sum")
            loss.backward()
            loss_sum += loss.detach()
            targets += int(generated.sum())
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= targets  # média por token gerado no passo inteiro
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state.update(step=step + 1, epoch=data.epoch, cursor=data.cursor)
        window_cells += accum * cfg.micro_tokens

        if state["step"] % cfg.log_every == 0 or state["step"] == total_steps:
            synchronize(device)
            loss_value = float(loss_sum) / targets
            speed = window_cells / (time.time() - window_start)
            eta = (total_steps - state["step"]) * accum * cfg.micro_tokens / speed
            print(f"passo {state['step']:>6,}/{total_steps:,} | loss {loss_value:.3f} | lr {lr:.2e} | "
                  f"grad {float(grad_norm):.2f} | {speed:,.0f} células/s | faltam {eta / 3600:.2f} h", flush=True)
            log({"step": state["step"], "loss": loss_value, "lr": lr, "grad_norm": float(grad_norm),
                 "cells_per_s": speed, "elapsed_s": time.time() - start})
            window_cells, window_start = 0, time.time()

        if state["step"] % cfg.eval_every == 0 or state["step"] == total_steps:
            run_eval()
            window_start = time.time()

        if state["step"] % cfg.save_every == 0 or state["step"] == total_steps:
            for full in (True, False):
                save_checkpoint(run_dir, model, optimizer, {**state, "config": asdict(cfg)}, full)
            window_start = time.time()

    print(f"fim: {total_steps:,} passos em {(time.time() - start) / 3600:.2f} h; "
          f"modelo final em {run_dir / f'step-{total_steps:05d}'}")


if __name__ == "__main__":
    main()
