"""Arquitetura do modelo: decoder estilo Llama, dimensionado para gerar na CPU.

A arquitetura é a Llama padrão do ``transformers`` (RMSNorm, SwiGLU, RoPE,
atenção com poucas cabeças de K/V, sem bias), para exportar para o llama.cpp
(GGUF), onde o modelo roda na CPU quantizado em int4 e com a gramática do
sampler (llguidance) embutida.

Os tamanhos foram escolhidos pela inferência na CPU, não pela GPU de treino:
gerar um token lê todos os pesos da memória, então a velocidade máxima é
~banda de memória / tamanho dos pesos. Em int4 (grupos de 32), o ``small``
ocupa ~24 MB e cabe quase inteiro no cache L3 de uma CPU de desktop.

Todas as dimensões são múltiplas de 64 (cabeças de 64), compatíveis com os
grupos de 32 pesos da quantização int4 (``Q4_0`` do llama.cpp).
"""

from dataclasses import dataclass

from transformers import LlamaConfig, LlamaForCausalLM

from codelm.chat_template import EOS
from codelm.tokenizer import load_tokenizer

PAD = "<|pad|>"
CONTEXT_LENGTH = 2048
HEAD_DIM = 64
# Base alta do RoPE: deixa espaço para estender o contexto depois do treino.
ROPE_THETA = 100_000.0


@dataclass(frozen=True)
class ModelSize:
    hidden_size: int
    num_layers: int
    num_kv_heads: int
    intermediate_size: int  # SwiGLU: ~2,75x hidden_size, múltiplo de 128

    @property
    def num_heads(self) -> int:
        return self.hidden_size // HEAD_DIM


SIZES: dict[str, ModelSize] = {
    "tiny": ModelSize(hidden_size=384, num_layers=8, num_kv_heads=2, intermediate_size=1024),  # ~19M
    "small": ModelSize(hidden_size=512, num_layers=12, num_kv_heads=2, intermediate_size=1408),  # ~42M
    "base": ModelSize(hidden_size=768, num_layers=16, num_kv_heads=4, intermediate_size=2048),  # ~113M
}
DEFAULT_SIZE = "small"


def model_config(size: str = DEFAULT_SIZE) -> LlamaConfig:
    tokenizer = load_tokenizer()
    s = SIZES[size]
    return LlamaConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=s.hidden_size,
        intermediate_size=s.intermediate_size,
        num_hidden_layers=s.num_layers,
        num_attention_heads=s.num_heads,
        num_key_value_heads=s.num_kv_heads,
        head_dim=HEAD_DIM,
        hidden_act="silu",
        max_position_embeddings=CONTEXT_LENGTH,
        rope_parameters={"rope_type": "default", "rope_theta": ROPE_THETA},
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        attention_bias=False,
        mlp_bias=False,
        # Embeddings de entrada e saída compartilham a matriz (economiza vocab x hidden).
        tie_word_embeddings=True,
        bos_token_id=None,
        eos_token_id=tokenizer.token_to_id(EOS),
        pad_token_id=tokenizer.token_to_id(PAD),
    )


def build_model(size: str = DEFAULT_SIZE) -> LlamaForCausalLM:
    """Modelo novo, com pesos aleatórios."""
    return LlamaForCausalLM(model_config(size))
