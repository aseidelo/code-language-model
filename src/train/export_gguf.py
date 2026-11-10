"""Exporta o modelo para GGUF (llama.cpp) e quantiza para rodar na CPU.

1. salva o modelo no formato do ``transformers`` com o nosso tokenizador;
2. converte para GGUF (f16) com o ``convert_hf_to_gguf.py`` do llama.cpp;
3. quantiza com ``llama-quantize``: blocos em ``Q4_0`` (int4 em grupos de 32,
   o mesmo esquema do QAT) e embeddings em ``Q8_0`` (ficam fora do QAT).

O llama.cpp nunca tokeniza texto para nós: os IDs vêm do nosso tokenizador
(indentação relativa, chat template) e a decodificação também é nossa. Por
isso o pré-tokenizador é registrado como "gpt-2" (mesmo esquema byte-level); o
que importa no GGUF é o vocabulário em bytes, que é exato (o llguidance usa).

Requer o llama.cpp compilado em ``third_party/llama.cpp`` (ver README).

Uso:
    uv run python -m train.export_gguf --random --size small     # modelo aleatório (teste)
    uv run python -m train.export_gguf --random --arch gemma3    # idem, arquitetura v2
                                                                 # -> data/code-language-model/export/
    uv run python -m train.export_gguf --checkpoint <pasta>       # modelo treinado
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, PreTrainedModel

from codelm.chat_template import EOS
from codelm.languages import PROJECT_MODEL_DIR, PROJECT_ROOT
from codelm.model import ARCHS, DEFAULT_ARCH, PAD, build_model
from codelm.tokenizer import active_tokenizer_path, use_tokenizer_of

LLAMA_CPP = PROJECT_ROOT / "third_party" / "llama.cpp"
EXPORT_DIR = PROJECT_MODEL_DIR / "export"
QUANTIZATIONS = ("Q4_0", "Q8_0")


def save_hf(model: PreTrainedModel, out_dir: Path) -> None:
    """Modelo + tokenizador ativo (``codelm.tokenizer.use_tokenizer_of``) no formato que o conversor lê."""
    model.save_pretrained(out_dir, safe_serialization=True)
    shutil.copy(active_tokenizer_path(), out_dir / "tokenizer.json")
    config = {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": EOS, "pad_token": PAD}
    (out_dir / "tokenizer_config.json").write_text(json.dumps(config, indent=1))


def convert_to_gguf(hf_dir: Path, out_file: Path) -> None:
    sys.path.insert(0, str(LLAMA_CPP))
    import convert_hf_to_gguf
    from conversion import base, gemma, llama

    # Nosso pré-tokenizador não está na lista do llama.cpp (e ele não vai usá-lo).
    base.TextModel.get_vocab_base_pre = lambda self, tokenizer: "gpt-2"
    # Vocabulário BPE byte-level (como o do Llama 3), sem tentar o sentencepiece antes.
    llama.LlamaModel.set_vocab = lambda self: self._set_vocab_gpt2()

    # Gemma 3: o llama.cpp assume o padrão 5:1 do Gemma original e o RoPE local
    # padrão. Gravamos o tipo de cada camada e a base do RoPE local explicitamente.
    gemma3_params = gemma.Gemma3Model.set_gguf_parameters

    def set_gemma3_params(self) -> None:
        gemma3_params(self)
        self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in self.hparams["layer_types"]])
        self.gguf_writer.add_rope_freq_base_swa(self.rope_parameters["sliding_attention"]["rope_theta"])

    gemma.Gemma3Model.set_gguf_parameters = set_gemma3_params
    sys.argv = ["convert_hf_to_gguf.py", str(hf_dir), "--outfile", str(out_file), "--outtype", "f16"]
    convert_hf_to_gguf.main()


def quantize(f16_file: Path, kind: str) -> Path:
    out_file = f16_file.with_name(f16_file.name.replace("f16", kind.lower()))
    binary = LLAMA_CPP / "build" / "bin" / "llama-quantize"
    cmd = [str(binary), "--token-embedding-type", "q8_0", str(f16_file), str(out_file), kind]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="pasta de um modelo salvo com save_pretrained")
    source.add_argument("--random", action="store_true", help="modelo com pesos aleatórios (para testar o caminho)")
    parser.add_argument("--size", default="small")
    parser.add_argument("--arch", default=DEFAULT_ARCH, choices=ARCHS, help="arquitetura do modelo aleatório")
    parser.add_argument("--name", help="nome dos arquivos (padrão: o tamanho ou a pasta)")
    args = parser.parse_args()

    if not (LLAMA_CPP / "build" / "bin" / "llama-quantize").exists():
        parser.error(f"llama.cpp não compilado em {LLAMA_CPP} (ver README)")
    suffix = "" if args.arch == DEFAULT_ARCH else f"-{args.arch}"
    name = args.name or (f"{args.size}{suffix}-aleatorio" if args.random else args.checkpoint.name)
    if not args.random:
        use_tokenizer_of(args.checkpoint.resolve())  # o do checkpoint (ou o v1, se ele for antigo)
    model = build_model(args.size, args.arch) if args.random else AutoModelForCausalLM.from_pretrained(args.checkpoint)

    hf_dir = EXPORT_DIR / name / "hf"
    hf_dir.mkdir(parents=True, exist_ok=True)
    save_hf(model, hf_dir)
    f16_file = EXPORT_DIR / name / f"{name}-f16.gguf"
    convert_to_gguf(hf_dir, f16_file)
    for kind in QUANTIZATIONS:
        out = quantize(f16_file, kind)
        print(f"{out.relative_to(PROJECT_ROOT)}: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
