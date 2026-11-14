"""Definição do tokenizador do modelo.

O tokenizador é BPE byte-level com pré-tokenização específica para código.
Seu vocabulário tem uma parte fixa, definida aqui (tokens de controle e de
espaço) e em ``assets/tokenizer/<versão>/fixed_tokens.json`` (terminais das
gramáticas e vocabulário frequente), e uma parte aprendida do corpus (código e
texto natural) por ``train.train_tokenizer``.

Versões (``assets/tokenizer/<versão>/``):

* ``v1``: 10 linguagens, o dos modelos até o pré-treino v1, o A/B, os SFT e os
  GRPO. Congelado. Esses checkpoints não guardam o tokenizador.
* ``v2``: 11 linguagens (com bash), o do pré-treino v2 em diante.

Cada checkpoint novo guarda o ``tokenizer.json`` com que foi treinado
(``train.pretrain.save_checkpoint``). As ferramentas que abrem um checkpoint
chamam ``use_tokenizer_of(pasta)``: daí em diante ``load_tokenizer()`` devolve
o tokenizador dele, ou o v1 se ele for antigo. Sem isso, ``load_tokenizer()``
devolve a versão atual (``TOKENIZER_VERSION``), a dos dados novos.

O código nunca é tokenizado direto: antes, a indentação vira tokens
``<|indent|>``/``<|dedent|>`` (``codelm.indentation``). Por isso a sequência é
montada a partir de segmentos: tokens de controle entram como IDs e o texto
dos segmentos é sempre tokenizado como texto comum (um ``<|endoftext|>``
escrito no código vira caracteres, nunca o token especial).
"""

import hashlib
import re
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

from codelm.indentation import DEDENT, INDENT, Segment, decode_indentation, encode_indentation, normalize_code
from codelm.languages import ASSETS_DIR, LANGUAGES, PROJECT_MODEL_DIR

TOKENIZER_ROOT = ASSETS_DIR / "tokenizer"
TOKENIZER_VERSION = "v2"
# Arquivos da versão atual (o que train.fixed_tokens e train.train_tokenizer geram).
TOKENIZER_DIR = TOKENIZER_ROOT / TOKENIZER_VERSION
FIXED_TOKENS_PATH = TOKENIZER_DIR / "fixed_tokens.json"
TOKENIZER_PATH = TOKENIZER_DIR / "tokenizer.json"
# Checkpoints sem tokenizador próprio são de antes do versionamento: v1.
LEGACY_TOKENIZER_PATH = TOKENIZER_ROOT / "v1" / "tokenizer.json"
CHECKPOINT_TOKENIZER = "tokenizer.json"

_active_path = TOKENIZER_PATH


def checkpoint_tokenizer_path(checkpoint: str | Path) -> Path:
    """Tokenizador de um checkpoint: o que ele guarda, ou o v1 se ele não guarda.

    ``checkpoint`` é uma pasta, ou um nome relativo a ``model/checkpoints/``
    (``run/step-00100``, como nos argumentos das ferramentas).
    """
    path = Path(checkpoint)
    if not path.is_absolute():
        path = PROJECT_MODEL_DIR / "checkpoints" / path
    own = path / CHECKPOINT_TOKENIZER
    return own if own.exists() else LEGACY_TOKENIZER_PATH


def use_tokenizer_of(checkpoint: str | Path) -> Path:
    """Passa a usar o tokenizador do checkpoint em ``load_tokenizer()`` (e na gramática)."""
    global _active_path
    _active_path = checkpoint_tokenizer_path(checkpoint)
    return _active_path


def use_tokenizer(path: str | Path) -> Path:
    """Passa a usar o tokenizador em ``path`` em ``load_tokenizer()`` (e na gramática)."""
    global _active_path
    _active_path = Path(path)
    return _active_path


def active_tokenizer_path() -> Path:
    return _active_path


def tokenizer_sha256(path: str | Path) -> str:
    """Identidade de um tokenizador: o hash do ``tokenizer.json``."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def control_tokens() -> list[str]:
    tokens = ["<|endoftext|>", "<|pad|>"]
    tokens += [f"<|lang:{name}|>" for name in LANGUAGES]
    # Estrutura do chat template (ver ``codelm.chat_template``).
    tokens += ["<|version|>", "<|prompt|>", "<|response|>", "<|code|>", "<|/code|>"]
    tokens += ["<|description|>", "<|json_output|>"]
    # Indentação relativa (ver ``codelm.indentation``).
    tokens += [INDENT, DEDENT]
    # Fill-in-the-middle, útil para completar código no meio de um arquivo.
    tokens += ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]
    return tokens


def whitespace_tokens() -> list[str]:
    # A indentação vira <|indent|>/<|dedent|>; sobram quebras de linha e
    # espaços no meio da linha (alinhamento).
    tokens = ["\n", "\n\n", "\n\n\n"]
    tokens += [" " * n for n in range(2, 17)]
    tokens += ["\t" * n for n in range(1, 7)]
    return tokens


def all_fixed_tokens(fixed: dict) -> list[str]:
    """União de todos os tokens fixos não-controle, sem duplicatas, em ordem estável."""
    seen = dict.fromkeys(fixed["whitespace"])
    for lang in fixed["languages"].values():
        seen.update(dict.fromkeys(lang["grammar"]))
        seen.update(dict.fromkeys(lang["vocabulary"]))
    return list(seen)


def operators(fixed: dict) -> list[str]:
    """Operadores e pontuação de várias letras das gramáticas (``==``, ``->``, ``::``), do maior para o menor."""
    terminals = {t for lang in fixed["languages"].values() for t in lang["grammar"]}
    return sorted((t for t in terminals if len(t) > 1 and not re.search(r"[\w\s]", t)), key=lambda t: (-len(t), t))


def pre_tokenize_pattern(operators: list[str]) -> str:
    """Regex que corta o texto em "palavras" que o BPE nunca atravessa.

    * operador da gramática que começa com ``$`` (``$(``, ``${``), antes dos
      identificadores, que também podem começar com ``$``;
    * identificador com espaço opcional antes (``_`` e ``$`` fazem parte do nome);
    * um dígito por vez (o primeiro pode levar um espaço antes);
    * um operador da gramática inteiro, ou senão um único caractere de
      pontuação, com espaço opcional antes. Assim cada token de pontuação
      corresponde a exatamente um terminal da gramática (``):`` vira ``)`` ``:``);
    * quebras de linha e espaços (o último espaço antes de uma palavra fica com ela).
    """
    alternatives = "".join(re.escape(op) + "|" for op in operators)
    # Operadores que começam com $ (``$(``, ``${``, ``$((`` do bash; ``${`` das template
    # strings do JS) vêm antes dos identificadores, que também começam com $ (``$x``).
    dollar = "|".join(re.escape(op) for op in operators if op.startswith("$"))
    return (
        (rf" ?(?:{dollar})|" if dollar else "")
        + r" ?[\p{L}_$][\p{L}\p{N}_$]*"
        r"| ?\p{N}"
        rf"| ?(?:{alternatives}[^\s\p{{L}}\p{{N}}_$])"
        r"|\s+(?!\S)"
        r"|\s+"
    )


def build_tokenizer(fixed: dict) -> Tokenizer:
    """Tokenizador vazio (sem vocabulário) com a pré-tokenização e o decoder do modelo.

    ``fixed`` é o conteúdo de ``fixed_tokens.json``, de onde vêm os operadores.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pre_tokenize_pattern(operators(fixed))), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def load_tokenizer(path: str | Path | None = None) -> Tokenizer:
    """Tokenizador ativo (ver ``use_tokenizer_of``), ou o de ``path``."""
    path = Path(path) if path else _active_path
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe; rode antes: uv run python -m train.train_tokenizer")
    return Tokenizer.from_file(str(path))


def code_segments(code: str, lang: str | None) -> list[Segment]:
    """Código -> segmentos: texto padronizado (``normalize_code``) e indentação como tokens."""
    return encode_indentation(normalize_code(code), lang)


def encode_segments(tokenizer: Tokenizer, segments: list[Segment]) -> list[int]:
    texts = [text for is_control, text in segments if not is_control and text]
    previous = tokenizer.encode_special_tokens
    tokenizer.encode_special_tokens = True  # texto nunca vira token de controle
    try:
        encoded = iter(tokenizer.encode_batch(texts, add_special_tokens=False))
    finally:
        tokenizer.encode_special_tokens = previous

    ids = []
    for is_control, text in segments:
        if is_control:
            token_id = tokenizer.token_to_id(text)
            if token_id is None:
                raise ValueError(f"token de controle {text!r} não está no vocabulário; treine o tokenizador de novo")
            ids.append(token_id)
        elif text:
            ids += next(encoded).ids
    return ids


def decode_segments(tokenizer: Tokenizer, ids: list[int]) -> list[Segment]:
    special = {tokenizer.token_to_id(t) for t in control_tokens()}
    segments: list[Segment] = []
    run: list[int] = []
    for token_id in [*ids, None]:
        if token_id is not None and token_id not in special:
            run.append(token_id)
            continue
        if run:
            segments.append((False, tokenizer.decode(run, skip_special_tokens=False)))
            run = []
        if token_id is not None:
            segments.append((True, tokenizer.id_to_token(token_id)))
    return segments


def encode_code(tokenizer: Tokenizer, code: str, lang: str) -> list[int]:
    return encode_segments(tokenizer, code_segments(code, lang))


def decode_code(tokenizer: Tokenizer, ids: list[int], lang: str) -> str:
    """IDs de código -> texto, reconstruindo a indentação. Outros tokens de controle são ignorados."""
    segments = [s for s in decode_segments(tokenizer, ids) if not s[0] or s[1] in (INDENT, DEDENT)]
    return decode_indentation(segments, lang)
