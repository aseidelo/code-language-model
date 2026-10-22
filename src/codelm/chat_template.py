"""Chat template do modelo: como os campos viram a sequência de tokens.

A resposta sempre começa com o cabeçalho de linguagem (e versão, opcional) e
depois o tipo de conteúdo. Se o usuário informa a linguagem, o cabeçalho já
vem preenchido e o modelo continua dali; senão, o próprio modelo escolhe::

    <|prompt|>Escreva uma função que retorna os n primeiros primos.
    <|response|><|lang:python|><|version|>3.12
    <|code|>def primos(n):
    <|indent|><|indent|>...<|dedent|><|dedent|><|endoftext|>

Descrever código (o código do usuário vai entre ``<|code|>`` e ``<|/code|>``;
o cabeçalho da resposta diz a linguagem desse código)::

    <|prompt|>Descreva o que este código faz.
    <|code|>def primos(n):
    ...
    <|/code|>
    <|response|><|lang:python|>
    <|description|>Retorna os n primeiros primos...<|endoftext|>

Pré-treino (arquivo cru, sem pedido): só a parte da resposta::

    <|lang:rust|>
    <|code|>fn main() { ... }<|endoftext|>

Tipos de conteúdo, que também dizem ao sampler qual gramática usar:

* ``<|code|>``: código na linguagem do cabeçalho;
* ``<|description|>``: texto em linguagem natural;
* ``<|json_output|>``: JSON (por exemplo, uma descrição estruturada).

Código e JSON passam pela indentação relativa (``codelm.indentation``); o
texto natural, não. O código do pedido é codificado com a linguagem informada,
ou no modo genérico se ela não for informada.
"""

from dataclasses import dataclass
from typing import Literal

from tokenizers import Tokenizer

from codelm.indentation import Segment, decode_indentation
from codelm.languages import LANGUAGES
from codelm.tokenizer import code_segments, decode_segments, encode_segments

EOS = "<|endoftext|>"
VERSION = "<|version|>"
PROMPT = "<|prompt|>"
RESPONSE = "<|response|>"
CODE = "<|code|>"
CODE_END = "<|/code|>"
DESCRIPTION = "<|description|>"
JSON_OUTPUT = "<|json_output|>"

Kind = Literal["code", "description", "json"]
KIND_TOKENS: dict[str, str] = {"code": CODE, "description": DESCRIPTION, "json": JSON_OUTPUT}
# Tipos de resposta cujo conteúdo é estruturado (passa pela indentação relativa).
STRUCTURED_KINDS = {"code", "json"}


@dataclass
class Response:
    lang: str
    version: str | None
    kind: Kind
    content: str


def lang_token(lang: str) -> str:
    if lang not in LANGUAGES:
        raise ValueError(f"linguagem desconhecida: {lang!r} (disponíveis: {', '.join(LANGUAGES)})")
    return f"<|lang:{lang}|>"


def lang_from_token(token: str) -> str | None:
    if token.startswith("<|lang:") and token.endswith("|>"):
        return token[len("<|lang:") : -2]
    return None


def content_lang(kind: Kind, lang: str | None) -> str | None:
    """Linguagem do conteúdo da resposta: a do cabeçalho, ou JSON em ``<|json_output|>``."""
    return "json" if kind == "json" else lang


def header_segments(lang: str, version: str | None = None) -> list[Segment]:
    """Cabeçalho da resposta: ``<|lang:...|>``, versão opcional e quebra de linha."""
    segments = [(True, lang_token(lang))]
    if version:
        if "\n" in version:
            raise ValueError("a versão não pode conter quebra de linha")
        segments += [(True, VERSION), (False, version)]
    return segments + [(False, "\n")]


def prompt_segments(
    prompt: str, lang: str | None = None, version: str | None = None, code: str | None = None
) -> list[Segment]:
    """Tudo o que o modelo recebe: o pedido, o código opcional e o início da resposta.

    Com ``lang``, o cabeçalho da resposta já vem preenchido (e o código do
    pedido é codificado nessa linguagem); sem, o modelo escolhe a linguagem.
    """
    if version and not lang:
        raise ValueError("a versão só pode ser informada junto com a linguagem")
    segments = [(True, PROMPT), (False, prompt.rstrip() + "\n")]
    if code is not None:
        segments += [(True, CODE), *code_segments(code, lang), (True, CODE_END), (False, "\n")]
    segments.append((True, RESPONSE))
    if lang:
        segments += header_segments(lang, version)
    return segments


def body_segments(kind: Kind, content: str, lang: str) -> list[Segment]:
    """Tipo e conteúdo da resposta, até o ``<|endoftext|>``."""
    if kind not in KIND_TOKENS:
        raise ValueError(f"tipo de resposta desconhecido: {kind!r}")
    body = code_segments(content, content_lang(kind, lang)) if kind in STRUCTURED_KINDS else [(False, content)]
    return [(True, KIND_TOKENS[kind]), *body, (True, EOS)]


def pretraining_segments(lang: str, code: str, version: str | None = None) -> list[Segment]:
    return header_segments(lang, version) + body_segments("code", code, lang)


def render(segments: list[Segment]) -> str:
    """Texto com os tokens de controle à mostra, como o modelo vê (para depuração)."""
    return "".join(text for _, text in segments)


# --- Atalhos ---------------------------------------------------------------

def format_prompt(prompt: str, lang: str | None = None, version: str | None = None, code: str | None = None) -> str:
    return render(prompt_segments(prompt, lang, version, code))


def encode_prompt(
    tokenizer: Tokenizer, prompt: str, lang: str | None = None, version: str | None = None, code: str | None = None
) -> list[int]:
    return encode_segments(tokenizer, prompt_segments(prompt, lang, version, code))


def encode_example(
    tokenizer: Tokenizer,
    prompt: str,
    response: str,
    lang: str,
    kind: Kind = "code",
    version: str | None = None,
    code: str | None = None,
    lang_in_prompt: bool = False,
) -> tuple[list[int], list[int]]:
    """Exemplo de ajuste fino: (ids, máscara de perda). A perda vale 1 só no que o modelo gera.

    Com ``lang_in_prompt``, a linguagem foi informada pelo usuário: o cabeçalho
    vem preenchido e não entra na perda. Sem, o modelo aprende a escolhê-la.
    """
    given = (lang, version) if lang_in_prompt else (None, None)
    prompt_ids = encode_segments(tokenizer, prompt_segments(prompt, *given, code))
    generated = [] if lang_in_prompt else header_segments(lang, version)
    response_ids = encode_segments(tokenizer, generated + body_segments(kind, response, lang))
    return prompt_ids + response_ids, [0] * len(prompt_ids) + [1] * len(response_ids)


def encode_pretraining(tokenizer: Tokenizer, lang: str, code: str, version: str | None = None) -> list[int]:
    return encode_segments(tokenizer, pretraining_segments(lang, code, version))


def decode_response(tokenizer: Tokenizer, ids: list[int]) -> Response:
    """Tokens depois de ``<|response|>`` (incluindo um cabeçalho preenchido) -> ``Response``."""
    segments = decode_segments(tokenizer, ids)
    if not segments or not segments[0][0] or not (lang := lang_from_token(segments[0][1])):
        raise ValueError("a resposta não começa com um token de linguagem")

    i, version = 1, None
    if i < len(segments) and segments[i] == (True, VERSION):
        version, _, rest = segments[i + 1][1].partition("\n") if i + 1 < len(segments) else ("", "", "")
        i += 2
    elif i < len(segments) and segments[i] == (False, "\n"):
        i += 1
    if i >= len(segments) or segments[i][1] not in KIND_TOKENS.values():
        raise ValueError(f"o cabeçalho deve terminar com um destes tokens: {', '.join(KIND_TOKENS.values())}")
    kind = next(k for k, token in KIND_TOKENS.items() if token == segments[i][1])

    body = []
    for segment in segments[i + 1 :]:
        if segment == (True, EOS):
            break
        body.append(segment)
    if kind in STRUCTURED_KINDS:
        content = decode_indentation(body, content_lang(kind, lang))
    else:
        content = "".join(text for is_control, text in body if not is_control)
    return Response(lang, version or None, kind, content)
