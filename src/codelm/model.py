"""Arquitetura do modelo: decoder dimensionado para gerar na CPU.

Duas arquiteturas do ``transformers``, ambas exportáveis para o llama.cpp
(GGUF), onde o modelo roda na CPU quantizado em int4 e com a gramática do
sampler (llguidance) embutida:

* ``llama`` (v1): Llama padrão (RMSNorm, SwiGLU, RoPE, atenção com poucas
  cabeças de K/V, sem bias), contexto de 2048, todas as camadas globais.
* ``gemma3`` (v2, ver ``docs/arquitetura-v2.md``): Gemma 3 só texto, com
  camadas de atenção local (janela de 1024) e global alternadas, 3 locais para
  cada global, e contexto de 8192. Usa o pacote do Gemma 3 como ele é (GELU-tanh,
  QK-norm, normas antes e depois de cada bloco, embeddings escalados por √d),
  porque o grafo do llama.cpp fixa esses detalhes.

Os tamanhos foram escolhidos pela inferência na CPU, não pela GPU de treino:
gerar um token lê todos os pesos da memória, então a velocidade máxima é
~banda de memória / tamanho dos pesos. Em int4 (grupos de 32), o ``small``
ocupa ~24 MB e cabe quase inteiro no cache L3 de uma CPU de desktop.

Todas as dimensões são múltiplas de 64 (cabeças de 64), compatíveis com os
grupos de 32 pesos da quantização int4 (``Q4_0`` do llama.cpp).
"""

from dataclasses import dataclass

from transformers import Gemma3ForCausalLM, Gemma3TextConfig, LlamaConfig, LlamaForCausalLM, PreTrainedModel

from codelm.chat_template import EOS
from codelm.tokenizer import load_tokenizer

PAD = "<|pad|>"
CONTEXT_LENGTH = 2048
HEAD_DIM = 64
# Base alta do RoPE: deixa espaço para estender o contexto depois do treino.
ROPE_THETA = 100_000.0

# v2 (gemma3): atenção local/global alternada.
ARCHS = ("llama", "gemma3")
DEFAULT_ARCH = "llama"
CONTEXT_LENGTHS = {"llama": CONTEXT_LENGTH, "gemma3": 8192}
SLIDING_WINDOW = 1024
GLOBAL_EVERY = 4  # 3 camadas locais e 1 global; a última camada é sempre global
ROPE_THETA_LOCAL = 10_000.0
ROPE_THETA_GLOBAL = 1_000_000.0


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


def layer_types(num_layers: int) -> list[str]:
    """Tipo de atenção de cada camada: a cada ``GLOBAL_EVERY``, uma global (contando do fim)."""
    return ["full_attention" if (num_layers - 1 - i) % GLOBAL_EVERY == 0 else "sliding_attention"
            for i in range(num_layers)]


def gemma3_config(size: str = DEFAULT_SIZE) -> Gemma3TextConfig:
    tokenizer = load_tokenizer()
    s = SIZES[size]
    return Gemma3TextConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=s.hidden_size,
        intermediate_size=s.intermediate_size,
        num_hidden_layers=s.num_layers,
        num_attention_heads=s.num_heads,
        num_key_value_heads=s.num_kv_heads,
        head_dim=HEAD_DIM,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=CONTEXT_LENGTHS["gemma3"],
        layer_types=layer_types(s.num_layers),
        sliding_window=SLIDING_WINDOW,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": ROPE_THETA_LOCAL},
            "full_attention": {"rope_type": "default", "rope_theta": ROPE_THETA_GLOBAL},
        },
        # Escala 1/sqrt(head_dim), a mesma que o llama.cpp usa no Gemma 3.
        query_pre_attn_scalar=HEAD_DIM,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        attention_bias=False,
        tie_word_embeddings=True,
        bos_token_id=None,
        eos_token_id=tokenizer.token_to_id(EOS),
        pad_token_id=tokenizer.token_to_id(PAD),
    )


def model_config(size: str = DEFAULT_SIZE, arch: str = DEFAULT_ARCH) -> LlamaConfig | Gemma3TextConfig:
    if arch == "gemma3":
        return gemma3_config(size)
    if arch != "llama":
        raise ValueError(f"arquitetura desconhecida: {arch}")
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


def build_model(size: str = DEFAULT_SIZE, arch: str = DEFAULT_ARCH) -> PreTrainedModel:
    """Modelo novo, com pesos aleatórios."""
    config = model_config(size, arch)
    return Gemma3ForCausalLM(config) if arch == "gemma3" else LlamaForCausalLM(config)
