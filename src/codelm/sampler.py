"""Geração: amostragem comum (temperatura, top-k, top-p) restrita pelo formato da resposta.

``ResponseConstraint`` acompanha a resposta token a token e diz quais tokens
são permitidos em cada ponto:

1. cabeçalho: um token ``<|lang:...|>``, depois ``<|version|>`` + texto ou a
   quebra de linha, e então o tipo (``<|code|>``, ``<|description|>``,
   ``<|json_output|>``);
2. conteúdo: se há gramática para ele (``codelm.grammar``: a da linguagem do
   cabeçalho em ``<|code|>``, a de JSON em ``<|json_output|>``), só tokens que
   a gramática aceita; senão, qualquer token (amostragem comum).

Se o usuário informou a linguagem, o cabeçalho preenchido pelo chat template
passa pelo mesmo caminho (``feed``) antes da geração.
"""

from collections.abc import Callable

import numpy as np
from tokenizers import Tokenizer

from codelm.chat_template import EOS, KIND_TOKENS, VERSION, content_lang, lang_from_token
from codelm.grammar import GrammarConstraint, available
from codelm.languages import LANGUAGES
from codelm.tokenizer import control_tokens


def sample(
    logits: np.ndarray,
    allowed: np.ndarray | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    """Escolhe um token. ``allowed`` (máscara booleana) zera os proibidos; ``temperature=0`` é greedy."""
    logits = np.asarray(logits, dtype=np.float64).copy()
    if allowed is not None:
        logits[~allowed] = -np.inf
    if temperature <= 0:
        return int(np.argmax(logits))

    logits /= temperature
    if top_k is not None and top_k < logits.size:
        cutoff = np.partition(logits, -top_k)[-top_k]
        logits[logits < cutoff] = -np.inf
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    if top_p is not None and top_p < 1.0:
        order = np.argsort(-probs)
        keep = np.cumsum(probs[order]) - probs[order] < top_p  # o primeiro token sempre fica
        probs[order[~keep]] = 0.0
        probs /= probs.sum()
    rng = rng or np.random.default_rng()
    return int(rng.choice(probs.size, p=probs))


class ResponseConstraint:
    """Máscara de cada passo da resposta: cabeçalho, depois a gramática do conteúdo (se houver)."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        vocab = tokenizer.get_vocab_size()
        self.tokenizer = tokenizer
        self.version_id = tokenizer.token_to_id(VERSION)
        self.eos_id = tokenizer.token_to_id(EOS)
        self.kind_ids = {tokenizer.token_to_id(token): kind for kind, token in KIND_TOKENS.items()}
        self.lang_ids = {tokenizer.token_to_id(f"<|lang:{lang}|>") for lang in LANGUAGES}
        self.newline_ids = {i for t, i in tokenizer.get_vocab().items() if "Ċ" in t and not t.startswith("<|")}

        def mask(ids) -> np.ndarray:
            m = np.zeros(vocab, dtype=bool)
            m[list(ids)] = True
            return m

        self._lang_mask = mask(self.lang_ids)
        self._after_lang_mask = mask([self.version_id, tokenizer.token_to_id("Ċ")])
        self._kind_mask = mask(self.kind_ids)
        self._version_mask = ~mask(tokenizer.token_to_id(t) for t in control_tokens())  # texto livre

        self.state = "lang"  # lang -> after_lang -> (version) -> kind -> body -> done
        self.lang: str | None = None
        self.kind: str | None = None
        self.grammar: GrammarConstraint | None = None

    @property
    def done(self) -> bool:
        return self.state == "done"

    def allowed(self) -> np.ndarray | None:
        """Máscara booleana dos tokens permitidos agora, ou None (sem restrição)."""
        if self.state == "lang":
            return self._lang_mask
        if self.state == "after_lang":
            return self._after_lang_mask
        if self.state == "version":
            return self._version_mask
        if self.state == "kind":
            return self._kind_mask
        if self.state == "body" and self.grammar is not None:
            return self.grammar.allowed()
        return None

    def feed(self, token_id: int) -> None:
        """Avança com um token (gerado ou preenchido pelo template)."""
        if self.state == "lang":
            self.lang = lang_from_token(self.tokenizer.id_to_token(token_id))
            self.state = "after_lang"
        elif self.state == "after_lang":
            self.state = "version" if token_id == self.version_id else "kind"
        elif self.state == "version":
            if token_id in self.newline_ids:
                self.state = "kind"
        elif self.state == "kind":
            self.kind = self.kind_ids[token_id]
            grammar_lang = content_lang(self.kind, self.lang) if self.kind != "description" else None
            self.grammar = GrammarConstraint(grammar_lang) if grammar_lang in available() else None
            self.state = "body"
        elif self.state == "body":
            if self.grammar is not None and not self.grammar.consume(token_id):
                raise ValueError(f"token {self.tokenizer.id_to_token(token_id)!r} não é aceito pela gramática")
            if token_id == self.eos_id:
                self.state = "done"


def generate(
    next_logits: Callable[[list[int]], np.ndarray],
    tokenizer: Tokenizer,
    prompt_ids: list[int],
    max_tokens: int = 512,
    **sampling,
) -> list[int]:
    """Gera a resposta a partir dos ids do prompt (``chat_template.encode_prompt``).

    ``next_logits(ids)`` devolve os logits do próximo token (o modelo).
    Devolve os ids depois de ``<|response|>``, incluindo um cabeçalho
    preenchido, prontos para ``chat_template.decode_response``.
    """
    response_start = prompt_ids.index(tokenizer.token_to_id("<|response|>")) + 1
    constraint = ResponseConstraint(tokenizer)
    for token_id in prompt_ids[response_start:]:
        constraint.feed(token_id)

    ids = list(prompt_ids)
    while not constraint.done and len(ids) - len(prompt_ids) < max_tokens:
        token_id = sample(next_logits(ids), constraint.allowed(), **sampling)
        constraint.feed(token_id)
        ids.append(token_id)
    return ids[response_start:]
