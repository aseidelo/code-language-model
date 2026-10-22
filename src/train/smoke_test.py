"""Teste de fumaça do treino: o modelo treina na GPU, com e sem QAT?

Com pesos e dados aleatórios (não precisa de corpus), mede para um tamanho de
modelo:

* velocidade de treino em tokens/s e memória de pico, em bf16;
* o mesmo com QAT (torchao: pesos int4 em grupos de 32 + ativações int8
  dinâmicas, simulados no treino) nas camadas lineares dos blocos;
* se a conversão final para int4 roda e dá logits próximos dos do QAT.

Uso:
    uv run python -m train.smoke_test [--size small] [--batch 8] [--steps 10]
"""

import argparse
import time

import torch
from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer

from codelm.model import CONTEXT_LENGTH, build_model

GROUP_SIZE = 32  # grupos de 32 pesos por escala: o mesmo do Q4_0 do llama.cpp


def pick_device() -> torch.device:
    if torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def peak_memory_gb(device: torch.device) -> float:
    if device.type == "xpu":
        return torch.xpu.max_memory_allocated() / 2**30
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    return float("nan")


def train_speed(model, device, batch: int, steps: int) -> tuple[float, float]:
    """(tokens/s, loss final) de alguns passos de AdamW com autocast bf16."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    vocab = model.config.vocab_size
    ids = torch.randint(0, vocab, (batch, CONTEXT_LENGTH), device=device)
    for step in range(steps + 2):  # 2 passos de aquecimento
        if step == 2:
            synchronize(device)
            start = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    elapsed = time.perf_counter() - start
    return batch * CONTEXT_LENGTH * steps / elapsed, loss.item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", default="small")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--compile", action="store_true", help="usa torch.compile")
    parser.add_argument("--checkpointing", action="store_true", help="gradient checkpointing (menos memória)")
    args = parser.parse_args()

    device = pick_device()
    torch.manual_seed(0)
    model = build_model(args.size).to(device)
    if args.checkpointing:
        model.gradient_checkpointing_enable()
    params = sum(p.numel() for p in model.parameters())
    print(f"dispositivo: {device} | modelo {args.size}: {params / 1e6:.1f}M parâmetros")
    print(f"lote: {args.batch} x {CONTEXT_LENGTH} = {args.batch * CONTEXT_LENGTH:,} tokens por passo")

    trained = torch.compile(model) if args.compile else model
    speed, loss = train_speed(trained, device, args.batch, args.steps)
    print(f"bf16:       {speed:>9,.0f} tokens/s | loss {loss:.2f} | memória de pico {peak_memory_gb(device):.1f} GB")
    hours = 1e9 / speed / 3600
    print(f"            1 bilhão de tokens levaria ~{hours:.1f} h")

    # QAT só nos blocos: embeddings (amarrados à saída) ficam fora, em 8 bits no GGUF.
    quantizer = Int8DynActInt4WeightQATQuantizer(groupsize=GROUP_SIZE)
    model.model.layers = quantizer.prepare(model.model.layers)
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    trained = torch.compile(model) if args.compile else model
    speed, loss = train_speed(trained, device, args.batch, args.steps)
    print(f"bf16 + QAT: {speed:>9,.0f} tokens/s | loss {loss:.2f} | embeddings ainda amarrados: {tied}")

    # Conversão: as camadas simuladas viram int4 de verdade (feito na CPU).
    model = model.float().cpu().eval()
    ids = torch.randint(0, model.config.vocab_size, (1, 256))
    with torch.no_grad():
        fake_quant = model(input_ids=ids).logits
        model.model.layers = quantizer.convert(model.model.layers)
        real_quant = model(input_ids=ids).logits
    agreement = (fake_quant.argmax(-1) == real_quant.argmax(-1)).float().mean().item()
    error = ((fake_quant - real_quant).abs().mean() / fake_quant.abs().mean()).item()
    print(f"conversão int4: mesmo token mais provável em {agreement:.1%} das posições; erro relativo {error:.2%}")


if __name__ == "__main__":
    main()
