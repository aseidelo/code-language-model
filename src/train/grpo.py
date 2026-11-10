"""GRPO nos exercícios verificáveis, com currículo por conceito.

* Cada passo amostra ``groups`` exercícios, ``samples`` vezes cada, com a máscara de
  gramática por sequência. A vantagem é relativa à média do grupo: grupo uniforme não
  ensina nada.
* Recompensa pela porta ``train.rl.reward`` (ver docs/grpo.md): correctness binária,
  readability e efficiency só como desempate entre respostas corretas, com os pesos de
  ``Config``. Vantagem sem divisão pelo desvio do grupo. As referências da efficiency vão no checkpoint.
* A perda usa a mesma máscara e a mesma temperatura da amostragem, com QAT ligado, e
  pesa cada token igual (soma das log-probs dividida pelos tokens gerados no passo).
* Sem currículo: os exercícios saem de uma permutação de todo o conjunto, sem reposição.
* Avaliação em T=0 na validação, com modelos de exercício nunca vistos.
* ``throughput`` mede o laço sem atualizar os pesos.

Uso:
    uv run python -m train.grpo train --run grpo-v1 --base base-ex-v3/step-00762
    uv run python -m train.grpo train --run grpo-v1 --resume
    uv run python -m train.grpo throughput --checkpoint base-ex-v3/step-00762 [--groups 16] [--samples 8]
"""

import argparse
import json
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
from transformers import LlamaForCausalLM

from codelm.chat_template import RESPONSE, decode_response, encode_prompt
from codelm.model import PAD
from codelm.quantization import enable_qat
from codelm.sampler import ResponseConstraint
from codelm.tokenizer import load_tokenizer, use_tokenizer_of
from eval.exercises import load_exercises
from train.rl.reward import RewardModel, Weights, combine, reward_for
from train.pretrain import CHECKPOINT_DIR, _arg_type, learning_rate, make_optimizer, save_checkpoint
from train.smoke_test import pick_device, synchronize

# respostas do modelo com "\w" em string comum: o Python só avisa, e o aviso polui o log
warnings.filterwarnings("ignore", category=SyntaxWarning)

VISIBLE_TESTS = 3  # o pedido mostra no máximo isto; acima, há testes ocultos


@dataclass
class Config:
    base: str = "base-ex-v3/step-00762"
    steps: int = 500
    groups: int = 16
    samples: int = 8
    temperature: float = 0.7
    max_tokens: int = 256
    micro_tokens: int = 2048
    lr: float = 2e-5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 20
    decay_fraction: float = 0.2
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lang: str = "python"
    w_correctness: float = 1.0
    w_readability: float = 0.2
    w_efficiency: float = 0.2
    seen: bool = True  # inclui rl-vistos: os exercícios da parte rl que o SFT do piloto viu
    eval_every: int = 50
    eval_per_concept: int = 24
    save_every: int = 100
    log_every: int = 5
    seed: int = 0


class Sampler:
    """Sem reposição: percorre uma permutação do conjunto e, no fim, sorteia outra (nova época)."""

    def __init__(self, size: int, seed: int) -> None:
        self.size, self.seed, self.epoch, self.cursor = size, seed, 0, 0

    def _order(self) -> np.ndarray:
        return np.random.default_rng([self.seed, self.epoch]).permutation(self.size)

    def pick(self, rows: list[dict], count: int) -> list[dict]:
        picked = []
        while len(picked) < count:
            if self.cursor == self.size:
                self.epoch, self.cursor = self.epoch + 1, 0
                print(f"  época {self.epoch + 1}", flush=True)
            order = self._order()
            take = min(count - len(picked), self.size - self.cursor)
            picked += [rows[int(i)] for i in order[self.cursor : self.cursor + take]]
            self.cursor += take
        return picked

    def state(self) -> dict:
        return {"size": self.size, "seed": self.seed, "epoch": self.epoch, "cursor": self.cursor}

    def load(self, state: dict) -> None:
        assert state["size"] == self.size, "o conjunto de exercícios mudou desde o checkpoint"
        self.epoch, self.cursor = state["epoch"], state["cursor"]


class Rollout:
    """Uma sequência em geração: ids do prompt, gerados e as máscaras de cada token gerado."""

    def __init__(self, tokenizer, prompt: list[int], use_grammar: bool) -> None:
        self.prompt = prompt
        self.generated: list[int] = []
        self.masks: list[np.ndarray] = []
        self.constraint = ResponseConstraint(tokenizer, use_grammar)
        self.response_start = prompt.index(tokenizer.token_to_id(RESPONSE)) + 1
        for token_id in prompt[self.response_start:]:
            self.constraint.feed(token_id)
        self.failed = False

    @property
    def done(self) -> bool:
        return self.failed or self.constraint.done

    def response(self) -> list[int]:
        return self.prompt[self.response_start:] + self.generated


@torch.no_grad()
def sample_batch(model, tokenizer, prompts: list[list[int]], device, temperature: float, max_tokens: int,
                 use_grammar: bool = True) -> tuple[list[Rollout], dict[str, float]]:
    """Amostra com cache, à esquerda preenchido por pad; cada sequência tem sua máscara."""
    times = {"modelo": 0.0, "mascara": 0.0}
    vocab = tokenizer.get_vocab_size()
    pad = tokenizer.token_to_id(PAD)
    rollouts = [Rollout(tokenizer, p, use_grammar) for p in prompts]
    longest = max(len(p) for p in prompts)
    x = torch.tensor([[pad] * (longest - len(p)) + p for p in prompts], device=device)
    attention = torch.tensor([[0] * (longest - len(p)) + [1] * len(p) for p in prompts], device=device)
    positions = (attention.cumsum(1) - 1).clamp(min=0)
    cache = None
    for _ in range(max_tokens):
        start = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16):
            output = model(input_ids=x, attention_mask=attention, position_ids=positions, past_key_values=cache,
                           use_cache=True)
        cache = output.past_key_values
        logits = output.logits[:, -1].float()
        synchronize(device)
        times["modelo"] += time.perf_counter() - start

        start = time.perf_counter()
        allowed = np.ones((len(rollouts), vocab), dtype=bool)
        for i, rollout in enumerate(rollouts):
            if rollout.done:
                continue
            mask = rollout.constraint.allowed()
            if mask is not None and not mask.any():  # beco sem saída da gramática: sem nenhum token permitido
                rollout.failed = True
            elif mask is not None:
                allowed[i] = mask
        times["mascara"] += time.perf_counter() - start

        start = time.perf_counter()
        logits = logits.masked_fill(~torch.from_numpy(allowed).to(device), -torch.inf)
        if temperature > 0:
            tokens = torch.multinomial((logits / temperature).softmax(-1).nan_to_num(), 1)[:, 0].tolist()
        else:
            tokens = logits.argmax(-1).tolist()
        times["modelo"] += time.perf_counter() - start

        start = time.perf_counter()
        for i, (rollout, token) in enumerate(zip(rollouts, tokens)):
            if rollout.done:
                tokens[i] = pad
                continue
            try:
                rollout.constraint.feed(token)
            except ValueError:  # a gramática chegou a um beco sem saída
                rollout.failed = True
                continue
            rollout.generated.append(token)
            rollout.masks.append(allowed[i].copy())
        times["mascara"] += time.perf_counter() - start
        if all(r.done for r in rollouts):
            break
        x = torch.tensor(tokens, device=device)[:, None]
        attention = torch.cat([attention, torch.ones_like(x)], 1)
        positions = positions[:, -1:] + 1
    return rollouts, times


def rewards(tokenizer, reward: RewardModel, weights: Weights, rows: list[dict], rollouts: list[Rollout],
            samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompensas, acertos (correctness = 1) e as três dimensões, por grupo e amostra."""
    def evaluate(rollout: Rollout, row: dict):
        try:
            code = decode_response(tokenizer, rollout.response()).content if rollout.constraint.done else None
        except ValueError:
            code = None
        return reward.score(code, row)

    repeated = [row for row in rows for _ in range(samples)]
    with ThreadPoolExecutor(8) as pool:
        scores = list(pool.map(evaluate, rollouts, repeated))
    shape = len(rows), samples
    values = np.array([combine(s, weights) for s in scores], dtype=np.float32).reshape(shape)
    dimensions = np.array([(s.correctness, s.readability, s.efficiency) for s in scores], dtype=np.float32)
    return values, np.array([s.correctness == 1 for s in scores]).reshape(shape), dimensions


def advantages_of(values: np.ndarray) -> np.ndarray:
    """Recompensa menos a média do grupo, sem dividir pelo desvio (Dr. GRPO): diferenças
    pequenas, como as de readability, continuam pequenas. Grupo uniforme fica com zero."""
    return (values - values.mean(1, keepdims=True)).ravel()


def policy_loss(model, rollouts: list[Rollout], advantages: np.ndarray, device, pad: int, temperature: float) -> torch.Tensor:
    """-Σ vantagem × soma das log-probs da resposta, com a mesma máscara da amostragem.
    Cada token pesa igual (perda por token, como no DAPO): quem chama divide pelo total de
    tokens gerados no passo. A média por resposta diluía a punição de respostas erradas
    longas (viés de comprimento do Dr. GRPO)."""
    size = max(len(r.prompt) + len(r.generated) for r in rollouts)
    x = torch.full((len(rollouts), size), pad, device=device)
    attention = torch.zeros((len(rollouts), size), dtype=torch.long, device=device)
    for i, rollout in enumerate(rollouts):
        ids = rollout.prompt + rollout.generated
        x[i, : len(ids)] = torch.tensor(ids, device=device)
        attention[i, : len(ids)] = 1
    with torch.autocast(device.type, dtype=torch.bfloat16):
        logits = model(input_ids=x, attention_mask=attention, use_cache=False).logits.float()
    total = torch.zeros((), device=device)
    for i, rollout in enumerate(rollouts):
        if rollout.generated:
            first = len(rollout.prompt)
            step = logits[i, first - 1 : first - 1 + len(rollout.generated)]
            masks = torch.from_numpy(np.stack(rollout.masks)).to(device)
            log_probs = (step / temperature).masked_fill(~masks, -torch.inf).log_softmax(-1)
            chosen = log_probs[torch.arange(len(step)), torch.tensor(rollout.generated, device=device)]
            total = total - float(advantages[i]) * chosen.sum()
    return total


def backward_in_chunks(model, rollouts: list[Rollout], advantages: np.ndarray, device, pad: int, temperature: float,
                       micro_tokens: int) -> int:
    """Acumula o gradiente em micro-lotes de até ``micro_tokens`` células, como no SFT."""
    order = sorted(range(len(rollouts)), key=lambda i: len(rollouts[i].prompt) + len(rollouts[i].generated))
    chunks, current = [], []
    for i in order:
        size = len(rollouts[i].prompt) + len(rollouts[i].generated)
        if current and size * (len(current) + 1) > micro_tokens:
            chunks.append(current)
            current = []
        current.append(i)
    chunks.append(current)
    # dividir pelo total de tokens = dividir cada resposta pelo comprimento médio: a escala
    # fica a da média por resposta, mas cada token pesa igual
    tokens = max(1, sum(len(r.generated) for r in rollouts))
    for chunk in chunks:
        loss = policy_loss(model, [rollouts[i] for i in chunk], advantages[chunk], device, pad, temperature)
        (loss / tokens).backward()
    return len(chunks)


def load_model(source, device) -> LlamaForCausalLM:
    # Atenção "eager": o kernel sdpa da XPU acessa memória inválida em alguns lotes com
    # preenchimento à esquerda e cache, e a GPU perde o dispositivo (UR_RESULT_ERROR_DEVICE_LOST).
    model = LlamaForCausalLM.from_pretrained(source, attn_implementation="eager").to(device)
    enable_qat(model.model.layers)
    model.gradient_checkpointing_enable()
    return model


def throughput(args) -> None:
    device = pick_device() if args.device is None else torch.device(args.device)
    use_tokenizer_of(CHECKPOINT_DIR / args.checkpoint)
    tokenizer = load_tokenizer()
    model = load_model(CHECKPOINT_DIR / args.checkpoint, device)
    print(f"{device} | {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parâmetros", flush=True)
    rows = load_exercises(per_concept=args.groups)[:: 14][: args.groups]
    prompts = [encode_prompt(tokenizer, r["prompt"], "python" if r["lang_in_prompt"] else None) for r in rows]
    repeated = [p for p in prompts for _ in range(args.samples)]
    pad = tokenizer.token_to_id(PAD)

    for round_ in range(args.rounds):
        start = time.perf_counter()
        model.eval()
        rollouts, times = sample_batch(model, tokenizer, repeated, device, args.temperature, args.max_tokens)
        sampled = time.perf_counter() - start

        start = time.perf_counter()
        passed, solved, _ = rewards(tokenizer, reward_for("python"), Weights(), rows, rollouts, args.samples)
        scored = time.perf_counter() - start

        start = time.perf_counter()
        advantages = advantages_of(passed)
        model.train()
        chunks = backward_in_chunks(model, rollouts, advantages, device, pad, args.temperature, args.micro_tokens)
        synchronize(device)
        model.zero_grad()
        gradient = time.perf_counter() - start

        tokens = sum(len(r.generated) for r in rollouts)
        total = sampled + scored + gradient
        mixed = np.mean([p.min() < p.max() for p in passed])
        print(f"rodada {round_ + 1}: {tokens:,} tokens | amostragem {sampled:.1f} s ({tokens / sampled:.0f} tokens/s; "
              f"modelo {times['modelo']:.1f} s, máscara {times['mascara']:.1f} s) | execução {scored:.1f} s | "
              f"gradiente {gradient:.1f} s em {chunks} lotes | passo {total:.1f} s | acerto {100 * solved.mean():.1f}% | "
              f"mistos {100 * mixed:.0f}% | terminaram {100 * np.mean([r.constraint.done for r in rollouts]):.0f}%",
              flush=True)
    if device.type == "xpu":
        print(f"memória máxima {torch.xpu.max_memory_allocated() / 2**30:.2f} GB")


def weights_of(cfg: Config) -> Weights:
    return Weights(cfg.w_correctness, cfg.w_readability, cfg.w_efficiency)


def prompts_of(tokenizer, rows: list[dict]) -> list[list[int]]:
    return [encode_prompt(tokenizer, r["prompt"], "python" if r["lang_in_prompt"] else None) for r in rows]


def training_rows(seen: bool) -> list[dict]:
    """A parte do GRPO (e, com ``seen``, a parte dela que o SFT do piloto viu), sem os
    exemplos do pedido quando eles mostrariam todos os testes: copiar não pode dar recompensa."""
    rows = load_exercises(split="rl") + (load_exercises(split="rl-vistos") if seen else [])
    for row in rows:
        if len(row["tests"]) <= VISIBLE_TESTS:
            row["prompt"] = row["prompt"].split("\nExamples:")[0].split("\nYour code should")[0]
    return rows


def measure(model, tokenizer, reward: RewardModel, rows: list[dict], device, cfg: Config, temperature: float,
            samples: int) -> np.ndarray:
    """Acertos de ``rows``, em lotes do tamanho de um passo."""
    size = max(1, cfg.groups * cfg.samples // samples)
    results = []
    for i in range(0, len(rows), size):
        chunk = rows[i : i + size]
        prompts = [p for p in prompts_of(tokenizer, chunk) for _ in range(samples)]
        rollouts, _ = sample_batch(model, tokenizer, prompts, device, temperature, cfg.max_tokens)
        results.append(rewards(tokenizer, reward, weights_of(cfg), chunk, rollouts, samples)[1])
    return np.concatenate(results)


def train(args) -> None:
    run_dir = CHECKPOINT_DIR / args.run
    state = {"step": 0}
    if args.resume:
        state = json.loads((run_dir / "latest" / "state.json").read_text())
        cfg = Config(**state["config"])
    else:
        if (run_dir / "latest").exists():
            raise SystemExit(f"a rodada {args.run} já existe; use --resume ou outro nome")
        cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config) if getattr(args, f.name) is not None})
    run_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    torch.manual_seed(cfg.seed + state["step"])
    source = run_dir / "latest" if args.resume else CHECKPOINT_DIR / cfg.base
    use_tokenizer_of(source)
    tokenizer = load_tokenizer()
    pad = tokenizer.token_to_id(PAD)
    model = load_model(source, device)
    optimizer = make_optimizer(model, cfg)
    reward, weights = reward_for(cfg.lang), weights_of(cfg)
    if args.resume:
        optimizer.load_state_dict(torch.load(run_dir / "latest" / "optimizer.pt", map_location=device))
        reward.load(state["reward"])

    pool = training_rows(cfg.seen)
    valid = load_exercises(per_concept=cfg.eval_per_concept)
    print(f"rodada {args.run} | {device} | base {source} | {len(pool):,} exercícios | "
          f"{cfg.steps:,} passos de {cfg.groups} grupos com {cfg.samples} amostras")
    print(f"configuração: {asdict(cfg)}", flush=True)
    metrics = (run_dir / "metrics.jsonl").open("a")

    def log(entry: dict) -> None:
        metrics.write(json.dumps(entry) + "\n")
        metrics.flush()

    def run_eval() -> None:
        model.eval()
        passed = measure(model, tokenizer, reward, valid, device, cfg, 0.0, 1)[:, 0]
        scores = {}
        for concept in sorted({r["concept"] for r in valid}):
            scores[concept] = float(np.mean([p for p, r in zip(passed, valid) if r["concept"] == concept]))
        generator = float(np.mean([p for p, r in zip(passed, valid) if r["source"] == "gerador"]))
        opencoder = float(np.mean([p for p, r in zip(passed, valid) if r["source"] != "gerador"]))
        print(f"  avaliação T=0: gerador {100 * generator:.1f}% | opencoder {100 * opencoder:.1f}% | " + " | ".join(
            f"{k} {100 * v:.0f}" for k, v in scores.items()), flush=True)
        log({"step": state["step"], "gerador": generator, "opencoder": opencoder, **scores})

    sampler = Sampler(len(pool), cfg.seed)
    if args.resume:
        sampler.load(state["sampler"])
    else:
        run_eval()
    # desde o último log: grupos e grupos mistos (acerto e erro juntos) por conceito, e
    # tokens das respostas certas e erradas
    groups, mixed_groups = Counter(), Counter()
    lengths = {True: [], False: []}

    start, first_step = time.time(), state["step"]
    while state["step"] < cfg.steps:
        step = state["step"]
        lr = learning_rate(step, cfg.steps, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        rows = sampler.pick(pool, cfg.groups)
        prompts = [p for p in prompts_of(tokenizer, rows) for _ in range(cfg.samples)]
        model.eval()
        rollouts, _ = sample_batch(model, tokenizer, prompts, device, cfg.temperature, cfg.max_tokens)
        values, solved, dimensions = rewards(tokenizer, reward, weights, rows, rollouts, cfg.samples)
        model.train()
        backward_in_chunks(model, rollouts, advantages_of(values), device, pad, cfg.temperature, cfg.micro_tokens)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state["step"] = step + 1

        for row, group in zip(rows, solved):
            groups[row["concept"]] += 1
            mixed_groups[row["concept"]] += bool(group.min() < group.max())
        for rollout, ok in zip(rollouts, solved.ravel()):
            lengths[bool(ok)].append(len(rollout.generated))
        if state["step"] % cfg.log_every == 0 or state["step"] == cfg.steps:
            elapsed = time.time() - start
            mixed = float(np.mean([g.min() < g.max() for g in values]))
            mixed_passed = float(np.mean([g.min() < g.max() for g in solved]))  # só correctness = 1 contra o resto
            by_concept = {c: mixed_groups[c] / groups[c] for c in sorted(groups)}
            length_ok, length_wrong = (float(np.mean(lengths[k])) if lengths[k] else 0.0 for k in (True, False))
            remaining = (cfg.steps - state["step"]) * elapsed / (state["step"] - first_step)
            print(f"passo {state['step']:>5,}/{cfg.steps:,} | recompensa {values.mean():.3f} (C {dimensions[:, 0].mean():+.2f} R {dimensions[:, 1].mean():.2f} "
                  f"E {dimensions[:, 2].mean():.2f}) | acerto {100 * solved.mean():.0f}% | mistos {100 * mixed:.0f}% (com acerto {100 * mixed_passed:.0f}%) | "
                  f"tokens certas {length_ok:.0f} erradas {length_wrong:.0f} | lr {lr:.2e} | "
                  f"grad {float(grad_norm):.2f} | faltam {remaining / 3600:.2f} h", flush=True)
            print("  mistos com acerto por conceito: " + " | ".join(
                f"{c.removeprefix('oc-')} {100 * v:.0f}% de {groups[c]}" for c, v in by_concept.items()), flush=True)
            log({"step": state["step"], "reward": float(values.mean()), "correctness": float(dimensions[:, 0].mean()),
                 "readability": float(dimensions[:, 1].mean()), "efficiency": float(dimensions[:, 2].mean()),
                 "passed": float(solved.mean()), "mixed": mixed, "mixed_passed": mixed_passed, "lr": lr, "grad_norm": float(grad_norm),
                 "mixed_by_concept": by_concept, "groups_by_concept": dict(groups), "tokens_passed": length_ok,
                 "tokens_failed": length_wrong, "epoch": sampler.epoch, "elapsed_s": elapsed})
            groups, mixed_groups = Counter(), Counter()
            lengths = {True: [], False: []}

        if state["step"] % cfg.eval_every == 0 or state["step"] == cfg.steps:
            run_eval()
        if state["step"] % cfg.save_every == 0 or state["step"] == cfg.steps:
            for full in (True, False):
                save_checkpoint(run_dir, model, optimizer,
                                {"step": state["step"], "config": asdict(cfg), "sampler": sampler.state(),
                                 "reward": reward.state()}, full)

    print(f"fim: {cfg.steps:,} passos em {(time.time() - start) / 3600:.2f} h; "
          f"modelo final em {run_dir / f'step-{cfg.steps:05d}'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--run", required=True)
    training.add_argument("--resume", action="store_true")
    for f in fields(Config):
        kind = float if f.type is float or f.type == "float" else _arg_type(f.type)
        training.add_argument(f"--{f.name.replace('_', '-')}", type=kind, default=None)
    command = commands.add_parser("throughput")
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--device")
    command.add_argument("--groups", type=int, default=16)
    command.add_argument("--samples", type=int, default=8)
    command.add_argument("--temperature", type=float, default=0.7)
    command.add_argument("--max-tokens", type=int, default=256)
    command.add_argument("--rounds", type=int, default=3)
    command.add_argument("--micro-tokens", type=int, default=2048)
    command = commands.add_parser("evaluate", help="T=0 na validação, por conceito, em vários checkpoints")
    command.add_argument("checkpoints", nargs="+")
    command.add_argument("--per-concept", type=int, default=150)
    command = commands.add_parser("sweep", help="pass@1, pass@k e grupos mistos por temperatura")
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--temperatures", default="0.5,0.7,0.9,1.0,1.2")
    command.add_argument("--per-concept", type=int, default=50)
    command.add_argument("--samples", type=int, default=8)
    command.add_argument("--split", default="rl", help="rl (os do treino, com rl-vistos) ou valid")
    args = parser.parse_args()
    {"train": train, "throughput": throughput, "evaluate": evaluate, "sweep": sweep}[args.command](args)


def evaluate(args) -> None:
    """A avaliação que mostrou a queda do grpo-v2: 150 por conceito (2.100 exercícios), ~7 min por modelo."""
    device, cfg = pick_device(), Config()
    rows = load_exercises(per_concept=args.per_concept)
    results = {}
    for checkpoint in args.checkpoints:
        use_tokenizer_of(CHECKPOINT_DIR / checkpoint)  # cada checkpoint com o seu
        tokenizer = load_tokenizer()
        model = load_model(CHECKPOINT_DIR / checkpoint, device).eval()
        passed = measure(model, tokenizer, reward_for(cfg.lang), rows, device, cfg, 0.0, 1)[:, 0]
        concepts = {c: float(np.mean([p for p, r in zip(passed, rows) if r["concept"] == c]))
                    for c in sorted({r["concept"] for r in rows})}
        generator = float(np.mean([p for p, r in zip(passed, rows) if r["source"] == "gerador"]))
        opencoder = float(np.mean([p for p, r in zip(passed, rows) if r["source"] != "gerador"]))
        results[checkpoint] = {"total": float(passed.mean()), "gerador": generator, "opencoder": opencoder, **concepts}
        error = 100 * np.sqrt(passed.mean() * (1 - passed.mean()) / len(passed))
        print(f"{checkpoint}: total {100 * passed.mean():.1f}% (±{error:.1f}) | gerador {100 * generator:.1f}% | "
              f"opencoder {100 * opencoder:.1f}% | " + " | ".join(f"{k} {100 * v:.0f}" for k, v in concepts.items()),
              flush=True)
        del model
    path = CHECKPOINT_DIR / "avaliacoes" / f"exercicios-{args.per_concept}.json"
    path.parent.mkdir(exist_ok=True)
    saved = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**saved, **results}, indent=1))
    print(f"gravado em {path}")


def sweep(args) -> None:
    """Escolha da temperatura do GRPO: só grupos com acerto e erro dão sinal (ver docs/grpo.md)."""
    use_tokenizer_of(CHECKPOINT_DIR / args.checkpoint)
    device, tokenizer, cfg = pick_device(), load_tokenizer(), Config()
    if args.split == "rl":
        pool = training_rows(True)
        rng = np.random.default_rng(0)
        by_concept: dict[str, list[dict]] = {}
        for i in rng.permutation(len(pool)):
            group = by_concept.setdefault(pool[i]["concept"], [])
            if len(group) < args.per_concept:
                group.append(pool[i])
        rows = [r for group in by_concept.values() for r in group]
    else:
        rows = load_exercises(per_concept=args.per_concept)
    model = load_model(CHECKPOINT_DIR / args.checkpoint, device).eval()
    print(f"{args.checkpoint} | {len(rows):,} exercícios ({args.split}) × {args.samples} amostras", flush=True)
    for temperature in map(float, args.temperatures.split(",")):
        passed = measure(model, tokenizer, reward_for(cfg.lang), rows, device, cfg, temperature, args.samples)
        hits = passed.sum(1)
        print(f"T={temperature:.1f} | pass@1 {100 * passed.mean():.1f}% | pass@{args.samples} {100 * (hits > 0).mean():.1f}% | "
              f"mistos {100 * ((hits > 0) & (hits < args.samples)).mean():.1f}% | "
              f"todas erradas {100 * (hits == 0).mean():.1f}% | todas certas {100 * (hits == args.samples).mean():.1f}%",
              flush=True)


if __name__ == "__main__":
    main()
