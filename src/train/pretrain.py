"""Pré-treino do modelo no corpus de ``dataprep.pretrain``.

* Arquitetura: ``--arch llama`` (v1, contexto de 2048) ou ``--arch gemma3`` (v2,
  atenção local/global, contexto de 8192; ver ``docs/arquitetura-v2.md``).
* Dados: blocos do tamanho do contexto da arquitetura, tirados do ``train.bin``
  (documentos já embaralhados e separados por ``<|endoftext|>``), em ordem
  aleatória por época.
* Otimização: AdamW, bf16 (autocast), ``torch.compile``, acúmulo de gradiente
  até ``batch_tokens`` por passo, corte de gradiente.
* Agenda de learning rate WSD: aquecimento linear, patamar constante e queda
  linear até ``min_lr_ratio`` nos últimos ``decay_fraction`` dos passos.
* QAT nos últimos ``qat_fraction`` dos passos (por padrão, junto com a queda do
  learning rate): as camadas lineares dos blocos passam a simular pesos Q4_0 e
  ativações Q8_0 exatamente como o llama.cpp (``codelm.quantization``).
* Avaliação periódica na validação, com a perda por linguagem/idioma e pelo
  contexto disponível de cada token (``Evaluator``), em blocos de 8192 para
  qualquer arquitetura: as duas são comparáveis no mesmo conjunto.
* Checkpoints em ``data/code-language-model/checkpoints/<run>/``: ``latest/``
  (modelo + otimizador + posição nos dados, para retomar) e ``step-NNNNN/``
  (só o modelo). Todo checkpoint guarda o ``tokenizer.json`` com que foi
  treinado (ver ``codelm.tokenizer``). Métricas em ``metrics.jsonl``.

Uso:
    uv run python -m train.pretrain --run small-v1                     # 1 época (~1,1 bilhão de tokens)
    uv run python -m train.pretrain --run teste --tokens 20000000      # rodada curta (agenda completa)
    uv run python -m train.pretrain --run small-v1 --resume            # continua de latest/
    uv run python -m train.pretrain --run teste-v2 --arch gemma3 --tokens 20000000

Na nuvem (H100 no Modal), ver ``train.modal_pretrain``.
"""

import argparse
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available

from codelm.languages import PROJECT_MODEL_DIR
from codelm.model import ARCHS, CONTEXT_LENGTHS, build_model
from codelm.quantization import enable_qat
from codelm.tokenizer import CHECKPOINT_TOKENIZER, active_tokenizer_path, tokenizer_sha256, use_tokenizer, use_tokenizer_of
from dataprep.pretrain import PRETRAIN_DIR, data_tokenizer_path, data_tokenizer_sha256, load_meta, load_split
from train.smoke_test import pick_device, synchronize

CHECKPOINT_DIR = PROJECT_MODEL_DIR / "checkpoints"
# A validação usa sempre blocos deste tamanho. Um modelo de contexto menor é
# avaliado com janela deslizante (cada token vê de meio contexto a um contexto inteiro).
EVAL_CONTEXT = max(CONTEXT_LENGTHS.values())
# Faixas do contexto disponível de cada token: quantos tokens do mesmo documento
# vêm antes dele dentro do bloco. Acima de 2048, só o contexto longo enxerga.
CONTEXT_BUCKETS = ((0, 2048), (2048, 4096), (4096, EVAL_CONTEXT))


@dataclass
class Config:
    size: str = "small"
    data: str = PRETRAIN_DIR.name  # pasta dos dados em model/ (pretrain-v2; pretrain é o v1)
    arch: str = "llama"  # llama (v1, 2k) ou gemma3 (v2, local/global, 8k)
    attention: str = "auto"  # auto, sdpa ou flash_attention_2 (ver attention_implementation)
    tokens: int | None = None  # tokens de treino no total (None = 1 época)
    micro_batch: int = 4  # sequências por passada na GPU
    batch_tokens: int = 262_144  # tokens por passo do otimizador (128 x 2048 ou 32 x 8192)
    lr: float = 2e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    decay_fraction: float = 0.2
    qat_fraction: float = 0.2  # 0 desliga o QAT
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 250
    eval_sequences: int = 256  # blocos de EVAL_CONTEXT tokens da validação
    save_every: int = 500
    seed: int = 0
    compile: bool = True


class TrainBlocks:
    """Blocos de ``context`` tokens em ordem aleatória (nova a cada época), retomáveis."""

    def __init__(self, data_dir: Path, context: int, seed: int, epoch: int = 0, cursor: int = 0) -> None:
        self.tokens, _, _ = load_split("train", data_dir)
        self.context = context
        self.n_blocks = len(self.tokens) // context
        self.seed, self.epoch, self.cursor = seed, epoch, cursor
        self.order = np.random.default_rng(seed + epoch).permutation(self.n_blocks)

    def next(self, size: int) -> torch.Tensor:
        if self.cursor + size > self.n_blocks:
            self.epoch, self.cursor = self.epoch + 1, 0
            self.order = np.random.default_rng(self.seed + self.epoch).permutation(self.n_blocks)
        blocks = self.order[self.cursor : self.cursor + size]
        self.cursor += size
        batch = np.stack([self.tokens[b * self.context : (b + 1) * self.context] for b in blocks])
        return torch.from_numpy(batch.astype(np.int64))


class Evaluator:
    """Perda na validação: geral, por tipo de documento e por contexto disponível.

    Com os bytes de cada documento (dados v2), sai também a perda em bits por
    byte (``bpb ...``): cada token leva a fração média de bytes do seu documento,
    então tokenizadores diferentes ficam comparáveis no mesmo texto.

    Usa sempre os mesmos blocos de ``EVAL_CONTEXT`` tokens. Se o modelo tem
    contexto menor, cada bloco é percorrido em janelas de ``context`` tokens que
    avançam meio contexto; cada janela conta só a sua metade final (a primeira
    conta tudo). Assim todo token é avaliado uma vez, vendo entre meio e um
    contexto inteiro de tokens anteriores.
    """

    def __init__(self, data_dir: Path, sequences: int, context: int) -> None:
        tokens, starts, kinds = load_split("valid", data_dir)
        self.names = load_meta(data_dir)["doc_kinds"]
        self.context = context
        self.stride = context // 2
        if context < EVAL_CONTEXT and (EVAL_CONTEXT - context) % self.stride:
            raise ValueError(f"contexto {context} não divide o bloco de avaliação {EVAL_CONTEXT}")
        lengths = np.diff(np.append(starts, len(tokens)))
        bytes_file = data_dir / "valid.bytes.npy"
        bytes_per_token = np.load(bytes_file) / lengths if bytes_file.exists() else None
        n_blocks = len(tokens) // EVAL_CONTEXT
        chosen = np.linspace(0, n_blocks - 1, num=min(sequences, n_blocks)).astype(int)
        self.inputs = np.stack([tokens[b * EVAL_CONTEXT : (b + 1) * EVAL_CONTEXT] for b in chosen]).astype(np.int64)
        # Documento de cada token dos blocos escolhidos.
        position = np.arange(EVAL_CONTEXT)
        doc = np.stack([np.searchsorted(starts, b * EVAL_CONTEXT + position, side="right") - 1 for b in chosen])
        self.kinds = kinds[doc]
        self.bytes = bytes_per_token[doc] if bytes_per_token is not None else None
        # Contexto disponível: tokens do mesmo documento antes deste, dentro do bloco.
        self.available = np.minimum(position, chosen[:, None] * EVAL_CONTEXT + position - starts[doc])

    def token_losses(self, model, x: torch.Tensor) -> torch.Tensor:
        """Perda de cada alvo ``x[:, 1:]``, com janela deslizante se o contexto for menor que o bloco."""
        def window(w: torch.Tensor) -> torch.Tensor:
            logits = model(input_ids=w).logits
            return F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), w[:, 1:].flatten(),
                                   reduction="none").view(w.shape[0], -1)

        if self.context >= x.shape[1]:
            return window(x)
        parts = [window(x[:, : self.context])]
        for begin in range(self.stride, x.shape[1] - self.context + 1, self.stride):
            parts.append(window(x[:, begin : begin + self.context])[:, -self.stride :])
        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def __call__(self, model, device: torch.device, micro_batch: int) -> dict[str, float]:
        model.eval()
        losses = []
        for i in range(0, len(self.inputs), micro_batch):
            x = torch.from_numpy(self.inputs[i : i + micro_batch]).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                losses.append(self.token_losses(model, x).cpu().numpy())
        model.train()
        loss = np.concatenate(losses).reshape(-1)
        kinds = self.kinds[:, 1:].reshape(-1)
        available = self.available[:, 1:].reshape(-1)

        def mean(mask: np.ndarray) -> float | None:
            return float(loss[mask].mean()) if mask.any() else None

        is_text = np.isin(kinds, [i for i, n in enumerate(self.names) if n.startswith("natural:")])
        result = {"valid": float(loss.mean()), "valid código": mean(~is_text), "valid texto": mean(is_text)}
        result.update({f"valid {n}": mean(kinds == i) for i, n in enumerate(self.names)})
        if self.bytes is not None:
            token_bytes = self.bytes[:, 1:].reshape(-1)

            def bpb(mask: np.ndarray) -> float | None:
                return float(loss[mask].sum() / math.log(2) / token_bytes[mask].sum()) if mask.any() else None

            result.update({"bpb": bpb(np.ones_like(is_text)), "bpb código": bpb(~is_text), "bpb texto": bpb(is_text)})
            result.update({f"bpb {n}": bpb(kinds == i) for i, n in enumerate(self.names)})
        for lo, hi in CONTEXT_BUCKETS:
            in_bucket = (available >= lo) & (available < hi)
            result[f"valid ctx {lo // 1024}k-{hi // 1024}k"] = mean(in_bucket)
            result[f"valid código ctx {lo // 1024}k-{hi // 1024}k"] = mean(in_bucket & ~is_text)
        return {k: v for k, v in result.items() if v is not None}


def attention_implementation(cfg: Config, device: torch.device) -> str:
    """``auto``: FlashAttention 2 em CUDA (se o pacote ``flash-attn`` existe), ``sdpa`` no resto.

    Na H100 o ``sdpa`` cai no kernel "memory-efficient" da A100, que não aproveita
    a janela local: a Gemma 3 em 8k fazia ~120k tokens/s. Com o FlashAttention 2,
    que tem janela local nativa, faz ~500k (``train.attention_check``). Na B580
    (XPU) fica o ``sdpa``.
    """
    if cfg.attention != "auto":
        return cfg.attention
    return "flash_attention_2" if device.type == "cuda" and is_flash_attn_2_available() else "sdpa"


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
    shutil.copy(active_tokenizer_path(), tmp / CHECKPOINT_TOKENIZER)
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
        cfg = Config(**{"data": "pretrain", **state["config"]})  # rodadas sem "data" são do v1
        use_tokenizer_of(run_dir / "latest")
    else:
        if (run_dir / "latest").exists():
            parser.error(f"a rodada {args.run} já existe; use --resume ou outro nome")
        cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config) if getattr(args, f.name) is not None})
    data_dir = PROJECT_MODEL_DIR / cfg.data
    meta = load_meta(data_dir)
    if not args.resume:
        use_tokenizer(data_tokenizer_path(meta))  # o tokenizador com que os dados foram empacotados
    if tokenizer_sha256(active_tokenizer_path()) != data_tokenizer_sha256(meta):
        parser.error(f"os dados em {data_dir} não são do tokenizador do checkpoint ({active_tokenizer_path()})")
    run_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    torch.manual_seed(cfg.seed)
    if cfg.arch not in ARCHS:
        parser.error(f"arquitetura desconhecida: {cfg.arch} (opções: {', '.join(ARCHS)})")
    context = CONTEXT_LENGTHS[cfg.arch]
    data = TrainBlocks(data_dir, context, cfg.seed, state["epoch"], state["cursor"])
    evaluator = Evaluator(data_dir, cfg.eval_sequences, context)
    tokens = cfg.tokens or data.n_blocks * context
    accum = cfg.batch_tokens // (cfg.micro_batch * context)
    if accum < 1 or cfg.batch_tokens % (cfg.micro_batch * context):
        parser.error(f"batch_tokens={cfg.batch_tokens} não é múltiplo de micro_batch x contexto = {cfg.micro_batch * context}")
    total_steps = tokens // (accum * cfg.micro_batch * context)
    qat_start = int(total_steps * (1 - cfg.qat_fraction)) if cfg.qat_fraction > 0 else total_steps + 1

    if args.resume:
        model = AutoModelForCausalLM.from_pretrained(run_dir / "latest").to(device)
    else:
        model = build_model(cfg.size, cfg.arch).to(device)
    attention_name = attention_implementation(cfg, device)
    model.set_attn_implementation(attention_name)
    # Sem cache de K/V no treino: ele só serve para gerar, e com o FlashAttention
    # fazia o torch.compile recompilar até o limite (e voltar a rodar sem compilar).
    model.config.use_cache = False
    if state["qat"]:
        enable_qat(model.model.layers)
    optimizer = make_optimizer(model, cfg)
    if args.resume:
        optimizer.load_state_dict(torch.load(run_dir / "latest" / "optimizer.pt", map_location=device))
    compiled = torch.compile(model) if cfg.compile else model

    params = sum(p.numel() for p in model.parameters())
    print(f"rodada {args.run} | {device} | modelo {cfg.arch} {cfg.size} ({params / 1e6:.1f}M, {attention_name}) | {total_steps:,} passos de "
          f"{accum * cfg.micro_batch} x {context} tokens | QAT a partir do passo {qat_start:,}")
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
        window_tokens += accum * cfg.micro_batch * context

        if state["step"] % cfg.log_every == 0 or state["step"] == total_steps:
            synchronize(device)
            loss_value = float(loss_sum)
            speed = window_tokens / (time.time() - window_start)
            eta = (total_steps - state["step"]) * accum * cfg.micro_batch * context / speed
            print(f"passo {state['step']:>6,}/{total_steps:,} | loss {loss_value:.3f} | lr {lr:.2e} | "
                  f"grad {float(grad_norm):.2f} | {speed:,.0f} tok/s | faltam {eta / 3600:.1f} h", flush=True)
            log({"step": state["step"], "loss": loss_value, "lr": lr, "grad_norm": float(grad_norm),
                 "tokens_per_s": speed, "qat": state["qat"], "elapsed_s": time.time() - start})
            window_tokens, window_start = 0, time.time()

        if state["step"] % cfg.eval_every == 0 or state["step"] == total_steps:
            result = evaluator(compiled, device, cfg.micro_batch)
            main_keys = ["valid", "valid código", "valid texto", *(k for k in result if k.startswith("valid ctx"))]
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
