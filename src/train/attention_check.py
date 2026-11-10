"""Compara as implementações de atenção numa GPU CUDA: correção e velocidade.

O ``flex_attention`` e o FlashAttention só rodam em CUDA no nosso ambiente,
então este teste roda no Modal:

    uv run modal run src/train/modal_pretrain.py::check_attention
    uv run modal run src/train/modal_pretrain.py::check_attention --args="--profile"

1. Correção (Gemma 3, sequência de 8192): logits e gradiente de cada
   implementação contra ``sdpa`` (a referência), em bf16, porque o
   FlashAttention não roda em fp32. Controle: a mesma comparação com a janela
   local errada (512) tem de dar diferença grande. As implementações:
   ``flex_attention`` (a do ``transformers``) e FlashAttention 2 (pacote
   ``flash-attn``, wheel oficial para torch 2.8; os kernels do Hub só têm build
   para torch >= 2.9).
2. Velocidade: passos de treino (frente e trás, bf16, ``torch.compile``), todos
   com 64k tokens por passada, em variantes que separam o custo do comprimento
   (Llama 2k x 8k), da arquitetura (Llama x Gemma 3 em 2k) e da janela local
   (Gemma 3 8k só global x local/global). Conta as quebras de grafo do
   ``torch.compile`` e, com ``--profile``, mostra os kernels mais caros das
   variantes com FlashAttention.

Resultado de 2026-11-10 (H100, ``small``): o ``sdpa`` cai no kernel
"memory-efficient" da A100 (``fmha_cutlass..._sm80``, 89% do tempo), que não
aproveita a janela local: Gemma 3 em 8k a ~120k tokens/s. O ``flex_attention``
dá o resultado certo mas fica 4x mais lento (também com a máscara em blocos
calculada uma vez só). O FlashAttention 2 dá o mesmo resultado (diferença do
nível do bf16) a ~500k tokens/s, mais rápido que a Llama em 2k com ``sdpa``.
"""

import argparse
import copy
import time
import traceback

import torch
from torch._dynamo.utils import counters
from transformers import Gemma3ForCausalLM, LlamaForCausalLM

from codelm.model import CONTEXT_LENGTHS, build_model, model_config

FA2 = "flash_attention_2"  # pacote flash-attn (wheel oficial na imagem do Modal)
IMPLEMENTATIONS = ("sdpa", "flex_attention", FA2)

# (nome, arquitetura, comprimento, sequências por passada, atenção, todas globais)
VARIANTS = [
    ("llama 2k sdpa", "llama", 2048, 32, "sdpa", False),
    ("gemma3 8k sdpa", "gemma3", 8192, 8, "sdpa", False),
    ("llama 2k fa2", "llama", 2048, 32, FA2, False),
    ("llama 8k fa2", "llama", 8192, 8, FA2, False),
    ("gemma3 2k fa2", "gemma3", 2048, 32, FA2, False),
    ("gemma3 8k global fa2", "gemma3", 8192, 8, FA2, True),
    ("gemma3 8k fa2", "gemma3", 8192, 8, FA2, False),
]


def set_window(model, window: int) -> None:
    model.config.sliding_window = window
    for layer in model.model.layers:
        if layer.self_attn.sliding_window:
            layer.self_attn.sliding_window = window


def use(model, impl: str, length: int) -> None:
    model.set_attn_implementation(impl)
    model.config.use_cache = False  # como no treino (train.pretrain)


def run(model, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=x, labels=x)
    out.loss.backward()
    grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return out.logits.detach().float(), grad


def check_correctness(size: str, length: int) -> None:
    torch.manual_seed(0)
    base = build_model(size, "gemma3").cuda()
    x = torch.randint(2, base.config.vocab_size, (1, length), device="cuda")
    results = {}
    cases = [(impl, impl, None) for impl in IMPLEMENTATIONS] + [("controle: janela 512", "sdpa", 512)]
    for name, impl, window in cases:
        model = copy.deepcopy(base)
        if window:
            set_window(model, window)
        try:
            use(model, impl, length)
            results[name] = run(model, x)
        except Exception:  # um kernel indisponível não derruba o resto do teste
            print(f"  {name}: falhou\n{traceback.format_exc(limit=-4)}")
    ref_logits, ref_grad = results["sdpa"]
    print(f"correção ({size}, {length} tokens, bf16; referência sdpa):")
    for name in (*IMPLEMENTATIONS[1:], "controle: janela 512"):
        if name not in results:
            continue
        logits, grad = results[name]
        diff = (logits - ref_logits).abs().max().item() / ref_logits.std().item()
        grad_diff = ((grad - ref_grad).norm() / ref_grad.norm()).item()
        print(f"  {name:22s} logits: dif máx / desvio = {diff:.2e} | gradiente: dif relativa = {grad_diff:.2e}")


def make_model(arch: str, size: str, all_global: bool = False):
    config = model_config(size, arch)
    if all_global:
        config.layer_types = ["full_attention"] * config.num_hidden_layers
    return (Gemma3ForCausalLM if arch == "gemma3" else LlamaForCausalLM)(config).cuda()


def benchmark(size: str, steps: int, profile: bool) -> None:
    """Velocidade de cada variante; todas com 64k tokens por passada."""
    print(f"velocidade ({size}, 64k tokens por passada, bf16, torch.compile):")
    for name, arch, length, micro_batch, impl, all_global in VARIANTS:
        torch.manual_seed(0)
        torch._dynamo.reset()
        counters["graph_break"].clear()
        model = make_model(arch, size, all_global)
        try:
            use(model, impl, length)
        except Exception:
            print(f"  {name:24s} falhou\n{traceback.format_exc(limit=-4)}")
            continue
        compiled = torch.compile(model)
        x = torch.randint(2, model.config.vocab_size, (micro_batch, length), device="cuda")

        def step() -> None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = compiled(input_ids=x, labels=x).loss
            loss.backward()
            model.zero_grad(set_to_none=True)

        try:
            for _ in range(3):  # compilação e aquecimento
                step()
        except Exception:
            print(f"  {name:24s} falhou no treino\n{traceback.format_exc(limit=-4)}")
            continue
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
        speed = steps * micro_batch * length / (time.time() - start)
        memory = torch.cuda.max_memory_allocated() / 2**30
        breaks = sum(counters["graph_break"].values())
        print(f"  {name:24s} {speed:>10,.0f} tokens/s | memória de pico {memory:5.1f} GB | quebras de grafo {breaks}")
        if profile and arch == "gemma3" and length == 8192 and not all_global and impl == FA2:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                step()
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8, max_name_column_width=70))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", default="small")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--profile", action="store_true", help="kernels mais caros da Gemma 3 em 8k com FlashAttention")
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()
    print(torch.cuda.get_device_name(), "| torch", torch.__version__)
    if not args.skip_correctness:
        check_correctness(args.size, CONTEXT_LENGTHS["gemma3"])
    benchmark(args.size, args.steps, args.profile)


if __name__ == "__main__":
    main()
