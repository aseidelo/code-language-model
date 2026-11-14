"""Motores de inferência para comparar o modelo em ponto flutuante, com QAT e quantizado.

* ``float``: pesos do checkpoint, sem quantização;
* ``qat``: Q4_0/Q8_0 simulados, como no treino;
* ``gguf``: o arquivo Q4_0 no llama.cpp, na CPU (requer ``llama-cpp-python``);
* ``f16``: o GGUF sem quantização, a base de comparação do ``gguf`` na CPU.

Todos têm a mesma interface: ``reset()`` e ``feed(ids)``, que devolve os logits
dos ids novos. ``load_generator`` monta sobre ela uma função ``ids do prompt ->
ids da resposta``, com parada no limite de contexto.
"""

from collections.abc import Callable

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from codelm.languages import PROJECT_MODEL_DIR
from codelm.model import CONTEXT_LENGTH
from codelm.quantization import enable_qat
from codelm.sampler import generate
from train.pretrain import CHECKPOINT_DIR
from train.smoke_test import pick_device

ENGINES = ("float", "qat", "gguf", "f16")


class TorchEngine:
    def __init__(self, checkpoint: str, qat: bool) -> None:
        self.device = pick_device()
        self.model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_DIR / checkpoint).to(self.device).eval()
        if qat:
            enable_qat(self.model.model.layers)
        self.reset()

    def reset(self) -> None:
        self.cache = None

    @torch.no_grad()
    def feed(self, ids: list[int]) -> np.ndarray:
        x = torch.tensor([ids], device=self.device)
        output = self.model(input_ids=x, past_key_values=self.cache, use_cache=True)
        self.cache = output.past_key_values
        return output.logits[0].cpu().numpy()


class GgufEngine:
    def __init__(self, run: str, kind: str) -> None:
        from llama_cpp import Llama

        path = PROJECT_MODEL_DIR / "export" / run / f"{run}-{kind}.gguf"
        self.model = Llama(str(path), n_ctx=CONTEXT_LENGTH, n_batch=CONTEXT_LENGTH, logits_all=True, verbose=False)

    def reset(self) -> None:
        self.model.reset()

    def feed(self, ids: list[int]) -> np.ndarray:
        self.model.eval(ids)
        return np.array(self.model.scores[self.model.n_tokens - len(ids) : self.model.n_tokens])


def load_engine(kind: str, checkpoint: str) -> TorchEngine | GgufEngine:
    if kind in ("gguf", "f16"):
        return GgufEngine(checkpoint.split("/")[0], "q4_0" if kind == "gguf" else "f16")
    return TorchEngine(checkpoint, qat=kind == "qat")


def load_generator(engine: TorchEngine | GgufEngine, tokenizer: Tokenizer, max_tokens: int = 512,
                   use_grammar: bool = True, **sampling) -> Callable[[list[int]], list[int]]:
    def run(prompt_ids: list[int]) -> list[int]:
        engine.reset()
        fed = 0

        def next_logits(ids: list[int]) -> np.ndarray:
            nonlocal fed
            logits = engine.feed(ids[fed:])
            fed = len(ids)
            return logits[-1]

        room = CONTEXT_LENGTH - len(prompt_ids)
        return generate(next_logits, tokenizer, prompt_ids, min(max_tokens, room), use_grammar, **sampling)

    return run
