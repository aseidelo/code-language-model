"""Quantização simulada (QAT) idêntica à do llama.cpp na CPU.

No GGUF, as camadas lineares dos blocos ficam em ``Q4_0`` e, na hora de
multiplicar, o llama.cpp quantiza as ativações em ``Q8_0``. Os dois formatos
usam blocos de 32 valores ao longo da dimensão de entrada, com uma escala por
bloco guardada em fp16 (``quantize_row_q4_0_ref``/``quantize_row_q8_0_ref`` em
``ggml/src/ggml-quants.c``):

* ``Q4_0``: ``d = (valor de maior módulo) / -8``; ``q = min(15, floor(x/d + 8.5))``;
  o peso vira ``(q - 8) * fp16(d)``;
* ``Q8_0``: ``d = max|x| / 127``; ``q = round(x/d)``; vira ``q * fp16(d)``.

``QuantizedLinear`` aplica exatamente isso no treino (pesos ``Q4_0`` vezes
ativações ``Q8_0``), com o gradiente passando direto pelo arredondamento
(straight-through estimator). Assim o QAT otimiza bit a bit o que roda na CPU.
"""

import torch
from torch import nn

BLOCK = 32


def _fp16(d: torch.Tensor) -> torch.Tensor:
    return d.to(torch.float16).to(d.dtype)


def fake_q4_0(w: torch.Tensor) -> torch.Tensor:
    """Pesos (out, in) como o llama.cpp os guarda em Q4_0 (blocos de 32 em ``in``)."""
    blocks = w.float().reshape(*w.shape[:-1], -1, BLOCK)
    index = blocks.abs().argmax(dim=-1, keepdim=True)
    d = blocks.gather(-1, index) / -8  # valor com sinal de maior módulo
    inv = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    q = torch.clamp(torch.floor(blocks * inv + 8.5), max=15) - 8
    return (q * _fp16(d)).reshape(w.shape).to(w.dtype)


def fake_q8_0(x: torch.Tensor) -> torch.Tensor:
    """Ativações (..., in) como o llama.cpp as quantiza para multiplicar (Q8_0)."""
    blocks = x.float().reshape(*x.shape[:-1], -1, BLOCK)
    d = blocks.abs().amax(dim=-1, keepdim=True) / 127
    inv = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    scaled = blocks * inv
    q = torch.sign(scaled) * torch.floor(scaled.abs() + 0.5)  # roundf do C: .5 longe do zero
    return (q * _fp16(d)).reshape(x.shape).to(x.dtype)


def straight_through(x: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """Valor quantizado no forward; gradiente como se fosse a identidade no backward."""
    return x + (quantized - x).detach()


class QuantizedLinear(nn.Linear):
    """``nn.Linear`` com pesos Q4_0 e ativações Q8_0 simulados.

    Tem os mesmos parâmetros de um ``nn.Linear``: o ``state_dict`` não muda, e
    os pesos treinados exportam direto para GGUF.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = straight_through(self.weight, fake_q4_0(self.weight))
        x = straight_through(x, fake_q8_0(x))
        return nn.functional.linear(x, weight, self.bias)


def enable_qat(module: nn.Module) -> int:
    """Troca (no lugar) todos os ``nn.Linear`` de ``module`` por ``QuantizedLinear``.

    Retorna quantas camadas foram trocadas. Use nos blocos do modelo
    (``model.model.layers``): os embeddings ficam em Q8_0 no GGUF e fora do QAT.
    """
    count = 0
    for child in module.modules():
        if type(child) is nn.Linear:
            if child.in_features % BLOCK:
                raise ValueError(f"in_features={child.in_features} não é múltiplo de {BLOCK}")
            child.__class__ = QuantizedLinear
            count += 1
    return count
