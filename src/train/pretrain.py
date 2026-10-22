"""Pré-treino do modelo no corpus de ``dataprep.pretrain``.

* Dados: blocos de ``CONTEXT_LENGTH`` tokens do ``train.bin`` (documentos já
  embaralhados e separados por ``<|endoftext|>``), em ordem aleatória por época.
* Otimização: AdamW, bf16 (autocast), ``torch.compile``, acúmulo de gradiente
  até ``batch_tokens`` por passo, corte de gradiente.
* Agenda de learning rate WSD: aquecimento linear, patamar constante e queda
  linear até ``min_lr_ratio`` nos últimos ``decay_fraction`` dos passos.
* QAT nos últimos ``qat_fraction`` dos passos (por padrão, junto com a queda do
  learning rate): as camadas lineares dos blocos passam a simular pesos Q4_0 e
  ativações Q8_0 exatamente como o llama.cpp (``codelm.quantization``).
* Avaliação periódica na validação, com a perda por linguagem/idioma.
* Checkpoints em ``data/code-language-model/checkpoints/<run>/``: ``latest/``
  (modelo + otimizador + posição nos dados, para retomar) e ``step-NNNNN/``
  (só o modelo). Métricas em ``metrics.jsonl``.

Uso:
    uv run python -m train.pretrain --run small-v1                     # 1 época (~1,1 bilhão de tokens)
    uv run python -m train.pretrain --run teste --tokens 20000000      # rodada curta (agenda completa)
    uv run python -m train.pretrain --run small-v1 --resume            # continua de latest/
"""

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM

from codelm.languages import PROJECT_DATA_DIR
from codelm.model import CONTEXT_LENGTH, build_model
from codelm.quantization import enable_qat
from dataprep.pretrain import PRETRAIN_DIR, load_split
from train.smoke_test import pick_device, synchronize

CHECKPOINT_DIR = PROJECT_DATA_DIR / "checkpoints"


@dataclass
class Config:
    size: str = "small"
    tokens: int | None = None  # tokens de treino no total (None = 1 época)
    micro_batch: int = 4  # sequências por passada na GPU
    batch_tokens: int = 262_144  # tokens por passo do otimizador (128 sequências de 2048)
    lr: float = 2e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    decay_fraction: float = 0.2
    qat_fraction: float = 0.2  # 0 desliga o QAT
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 250
    eval_sequences: int = 256
    save_every: int = 500
    seed: int = 0
    compile: bool = True


class TrainBlocks:
    """Blocos de ``CONTEXT_LENGTH`` tokens em ordem aleatória (nova a cada época), retomáveis."""

    def __init__(self, seed: int, epoch: int = 0, cursor: int = 0) -> None:
        self.tokens, _, _ = load_split("train")
        self.n_blocks = len(self.tokens) // CONTEXT_LENGTH
        self.seed, self.epoch, self.cursor = seed, epoch, cursor
        self.order = np.random.default_rng(seed + epoch).permutation(self.n_blocks)

    def next(self, size: int) -> torch.Tensor:
        if self.cursor + size > self.n_blocks:
            self.epoch, self.cursor = self.epoch + 1, 0
            self.order = np.random.default_rng(self.seed + self.epoch).permutation(self.n_blocks)
        blocks = self.order[self.cursor : self.cursor + size]
        self.cursor += size
        batch = np.stack([self.tokens[b * CONTEXT_LENGTH : (b + 1) * CONTEXT_LENGTH] for b in blocks])
        return torch.from_numpy(batch.astype(np.int64))


class Evaluator:
    """Perda na validação, geral e por tipo de documento (linguagem ou idioma)."""

    def __init__(self, sequences: int) -> None:
        tokens, starts, kinds = load_split("valid")
        meta = json.loads((PRETRAIN_DIR / "meta.json").read_text())
        self.names = meta["doc_kinds"]
        # Tipo de documento de cada token.
        token_kinds = np.repeat(kinds, np.diff(np.append(starts, len(tokens))))
        n_blocks = len(tokens) // CONTEXT_LENGTH
        chosen = np.linspace(0, n_blocks - 1, num=min(sequences, n_blocks)).astype(int)
        self.inputs = np.stack([tokens[b * CONTEXT_LENGTH : (b + 1) * CONTEXT_LENGTH] for b in chosen]).astype(np.int64)
        self.kinds = np.stack([token_kinds[b * CONTEXT_LENGTH : (b + 1) * CONTEXT_LENGTH] for b in chosen])

    @torch.no_grad()
    def __call__(self, model, device: torch.device, micro_batch: int) -> dict[str, float]:
        model.eval()
        sums = np.zeros(len(self.names))
        counts = np.zeros(len(self.names))
        for i in range(0, len(self.inputs), micro_batch):
            x = torch.from_numpy(self.inputs[i : i + micro_batch]).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(input_ids=x).logits
            losses = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), x[:, 1:].flatten(), reduction="none")
            target_kinds = self.kinds[i : i + micro_batch, 1:].reshape(-1)
            np.add.at(sums, target_kinds, losses.cpu().numpy())
            np.add.at(counts, target_kinds, 1)
        model.train()
        result = {"valid": float(sums.sum() / counts.sum())}
        code = [i for i, n in enumerate(self.names) if not n.startswith("natural:")]
        text = [i for i, n in enumerate(self.names) if n.startswith("natural:")]
        result["valid código"] = float(sums[code].sum() / counts[code].sum())
        result["valid texto"] = float(sums[text].sum() / counts[text].sum())
        result.update({f"valid {n}": float(sums[i] / counts[i]) for i, n in enumerate(self.names) if counts[i]})
        return result


def learning_rate(step: int, total: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    decay_start = int(total * (1 - cfg.decay_fraction))
    if step < decay_start:
        return cfg.lr
    progress = (step - decay_start) / max(1, total - decay_start)
    return cfg.lr * (1 - (1 - cfg.min_lr_ratio) * progress)


def make_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    # Sem weight decay em normas e vetores; com weight decay nas matrizes (incluindo embeddings).
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)


def save_checkpoint(run_dir: Path, model, optimizer, state: dict, full: bool) -> None:
    """``latest/`` completo (retomar) ou ``step-NNNNN/`` só com o modelo."""
    name = "latest" if full else f"step-{state['step']:05d}"
    tmp = run_dir / f"{name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, safe_serialization=True)
    if full:
        torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    (tmp / "state.json").write_text(json.dumps(state, indent=1))
    shutil.rmtree(run_dir / name, ignore_errors=True)
    tmp.rename(run_dir / name)


def _arg_type(annotation):
    if annotation is bool:
        return lambda value: value.lower() in ("1", "true", "sim")
    if annotation in (int, int | None):
        return int
    return annotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="nome da rodada (pasta dos checkpoints)")
    parser.add_argument("--resume", action="store_true", help="continua do checkpoint latest/ da rodada")
    for f in fields(Config):  # cada campo da configuração vira um argumento (--micro-batch, --lr, ...)
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=_arg_type(f.type), default=None)
    args = parser.parse_args()

    run_dir = CHECKPOINT_DIR / args.run
    state = {"step": 0, "epoch": 0, "cursor": 0, "qat": False}
    if args.resume:
        state = json.loads((run_dir / "latest" / "state.json").read_text())
        cfg = Config(**state["config"])
    else:
        if (run_dir / "latest").exists():
            parser.error(f"a rodada {args.run} já existe; use --resume ou outro nome")
        cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config) if getattr(args, f.name) is not None})
    run_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    torch.manual_seed(cfg.seed)
    data = TrainBlocks(cfg.seed, state["epoch"], state["cursor"])
    evaluator = Evaluator(cfg.eval_sequences)
    tokens = cfg.tokens or data.n_blocks * CONTEXT_LENGTH
    accum = cfg.batch_tokens // (cfg.micro_batch * CONTEXT_LENGTH)
    total_steps = tokens // (accum * cfg.micro_batch * CONTEXT_LENGTH)
    qat_start = int(total_steps * (1 - cfg.qat_fraction)) if cfg.qat_fraction > 0 else total_steps + 1

    if args.resume:
        model = LlamaForCausalLM.from_pretrained(run_dir / "latest").to(device)
    else:
        model = build_model(cfg.size).to(device)
    if state["qat"]:
        enable_qat(model.model.layers)
    optimizer = make_optimizer(model, cfg)
    if args.resume:
        optimizer.load_state_dict(torch.load(run_dir / "latest" / "optimizer.pt", map_location=device))
    compiled = torch.compile(model) if cfg.compile else model

    params = sum(p.numel() for p in model.parameters())
    print(f"rodada {args.run} | {device} | modelo {cfg.size} ({params / 1e6:.1f}M) | {total_steps:,} passos de "
          f"{accum * cfg.micro_batch} x {CONTEXT_LENGTH} tokens | QAT a partir do passo {qat_start:,}")
    print(f"configuração: {asdict(cfg)}")
    metrics = (run_dir / "metrics.jsonl").open("a")

    def log(entry: dict) -> None:
        metrics.write(json.dumps(entry) + "\n")
        metrics.flush()

    def checkpoint(full: bool) -> None:
        save_checkpoint(run_dir, model, optimizer, {**state, "config": asdict(cfg)}, full)

    model.train()
    start, window_tokens, window_start = time.time(), 0, time.time()
    while state["step"] < total_steps:
        step = state["step"]
        if step == qat_start and not state["qat"]:
            count = enable_qat(model.model.layers)
            compiled = torch.compile(model) if cfg.compile else model
            state["qat"] = True
            print(f"passo {step}: QAT ligado em {count} camadas (pesos Q4_0, ativações Q8_0)")

        lr = learning_rate(step, total_steps, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = 0.0
        for _ in range(accum):
            x = data.next(cfg.micro_batch).to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                loss = compiled(input_ids=x, labels=x).loss / accum
            loss.backward()
            loss_sum += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state.update(step=step + 1, epoch=data.epoch, cursor=data.cursor)
        window_tokens += accum * cfg.micro_batch * CONTEXT_LENGTH

        if state["step"] % cfg.log_every == 0 or state["step"] == total_steps:
            synchronize(device)
            loss_value = float(loss_sum)
            speed = window_tokens / (time.time() - window_start)
            eta = (total_steps - state["step"]) * accum * cfg.micro_batch * CONTEXT_LENGTH / speed
            print(f"passo {state['step']:>6,}/{total_steps:,} | loss {loss_value:.3f} | lr {lr:.2e} | "
                  f"grad {float(grad_norm):.2f} | {speed:,.0f} tok/s | faltam {eta / 3600:.1f} h", flush=True)
            log({"step": state["step"], "loss": loss_value, "lr": lr, "grad_norm": float(grad_norm),
                 "tokens_per_s": speed, "qat": state["qat"], "elapsed_s": time.time() - start})
            window_tokens, window_start = 0, time.time()

        if state["step"] % cfg.eval_every == 0 or state["step"] == total_steps:
            result = evaluator(compiled, device, cfg.micro_batch)
            main_keys = ["valid", "valid código", "valid texto"]
            print("  avaliação: " + " | ".join(f"{k} {result[k]:.3f}" for k in main_keys))
            print("  por tipo:  " + " | ".join(f"{k.removeprefix('valid ')} {v:.2f}" for k, v in result.items() if k not in main_keys))
            log({"step": state["step"], **result})

        if state["step"] % cfg.save_every == 0 or state["step"] == total_steps:
            checkpoint(full=True)
            checkpoint(full=False)
            window_start = time.time()  # não contar o tempo de salvar na velocidade

    print(f"fim: {total_steps:,} passos em {(time.time() - start) / 3600:.2f} h; modelo final em {run_dir / f'step-{total_steps:05d}'}")


if __name__ == "__main__":
    main()
