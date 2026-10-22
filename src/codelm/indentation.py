"""Indentação relativa: ``<|indent|>`` (+2 espaços) e ``<|dedent|>`` (-2 espaços).

O modelo não vê os espaços do início das linhas; vê só as mudanças de nível::

    def f(x):\\n                 def f(x):\\n
        if x:\\n          ->     <|indent|><|indent|>if x:\\n
            return 1\\n          <|indent|><|indent|>return 1\\n
        return 0\\n              <|dedent|><|dedent|>return 0\\n
                                <|dedent|><|dedent|>

Regras (sem perda: ``decode(encode(texto)) == texto``):

* cada 2 espaços no início de uma linha valem um nível;
* o que sobra (um espaço ímpar, tabs, outros brancos) fica como texto literal;
* no fim, ``<|dedent|>`` até voltar ao nível 0 (os tokens sempre se equilibram);
* algumas linhas não mudam o nível e mantêm o recuo literal, como faz o
  tokenizador do próprio Python. Seus recuos não indicam blocos:

  - linhas em branco;
  - linhas que começam dentro de uma string ou comentário de várias linhas;
  - em Python: linhas de continuação (dentro de ``()``, ``[]``, ``{}`` ou
    depois de ``\\``) e linhas só de comentário.

Codificador e decodificador acompanham o texto com o mesmo lexer
(``codelm.lexer``) para decidir isso do mesmo jeito. Com linguagem
desconhecida (``lang=None``) só valem as regras gerais, sem as de strings,
comentários e continuação.

Antes de tudo, ``normalize_code`` padroniza o texto: quebras de linha ``\\n``,
tabs do início das linhas trocados por 4 espaços e uma quebra de linha no fim
(vale para os dados de treino e para o código que o usuário envia).
"""

import re

from codelm.lexer import CODE, ContextLexer, make_lexer

INDENT = "<|indent|>"
DEDENT = "<|dedent|>"
INDENT_WIDTH = 2
TAB_WIDTH = 4

# Um segmento é (é_token_de_controle, texto).
Segment = tuple[bool, str]

_LEADING_TABS = re.compile(r"^[ \t]*\t[ \t]*", re.M)


def normalize_tabs(text: str) -> str:
    """Troca cada tab da indentação (início da linha) por ``TAB_WIDTH`` espaços."""
    return _LEADING_TABS.sub(lambda m: m.group().replace("\t", " " * TAB_WIDTH), text)


def normalize_code(text: str) -> str:
    """Quebras de linha ``\\n`` (sem ``\\r``), tabs da indentação viram espaços e ``\\n`` no fim."""
    text = normalize_tabs(text.replace("\r\n", "\n").replace("\r", "\n"))
    return text if not text or text.endswith("\n") else text + "\n"


def has_mixed_indentation(text: str) -> bool:
    """Se o arquivo indenta com tabs e com espaços (em Python, converter pode mudar o significado)."""
    leading = re.findall(r"^[ \t]+", text, re.M)
    return any("\t" in ws for ws in leading) and any(" " in ws for ws in leading)


class LineTracker:
    """Acompanha o texto para saber se a próxima linha é continuação (recuo literal)."""

    def __init__(self, lang: str | None) -> None:
        self.lexer = make_lexer(lang)
        self.python = lang == "python"
        self.depth = 0  # parênteses/colchetes/chaves abertos (Python)
        self.previous = ""  # caractere anterior
        self.backslash = False  # a linha anterior terminou com \ (continuação explícita)

    def feed(self, text: str) -> None:
        for ch in text:
            in_code = self.lexer.context == CODE
            in_code &= self.lexer.feed(ch) == CODE  # o caractere é código (não abre/fecha string ou comentário)
            if self.python and in_code:
                if ch in "([{":
                    self.depth += 1
                elif ch in ")]}":
                    self.depth = max(0, self.depth - 1)
            if ch == "\n":
                self.backslash = self.previous == "\\"
            self.previous = ch

    @property
    def continuation(self) -> bool:
        """A linha que começa agora tem recuo literal?"""
        # Só strings e comentários de verdade: no HTML o "texto" entre tags é indentado normalmente.
        if isinstance(self.lexer, ContextLexer) and self.lexer.mode in ("string", "block"):
            return True
        return self.python and (self.depth > 0 or self.backslash)

    def is_comment_start(self, ch: str) -> bool:
        """Em Python, uma linha que começa com ``#`` (fora de continuação) é só comentário."""
        return self.python and ch == "#"


def _level(line: str) -> int:
    spaces = len(line) - len(line.lstrip(" "))
    return spaces // INDENT_WIDTH


def has_odd_indentation(text: str, lang: str | None) -> bool:
    """Se alguma linha que define bloco tem recuo fora da unidade de ``INDENT_WIDTH`` espaços.

    Esse recuo sobra como espaço literal depois dos tokens de indentação, e em
    Python a gramática do sampler não o aceita (``codelm/grammars/python.lark``).
    """
    tracker = LineTracker(lang)
    for i, line in enumerate(text.split("\n")):
        if i:
            tracker.feed("\n")
        stripped = line.lstrip()
        if stripped and not tracker.continuation and not tracker.is_comment_start(stripped[0]):
            spaces = len(line) - len(line.lstrip(" "))
            if spaces % INDENT_WIDTH or line[spaces].isspace():
                return True
        tracker.feed(line)
    return False


def encode_indentation(text: str, lang: str | None) -> list[Segment]:
    """Texto -> segmentos com ``<|indent|>``/``<|dedent|>`` no lugar da indentação."""
    segments: list[Segment] = []
    buffer: list[str] = []
    tracker = LineTracker(lang)
    level = 0

    def flush() -> None:
        if buffer and (chunk := "".join(buffer)):
            segments.append((False, chunk))
        buffer.clear()

    def move_to(new_level: int) -> None:
        nonlocal level
        if new_level != level:
            flush()
            token = INDENT if new_level > level else DEDENT
            segments.extend([(True, token)] * abs(new_level - level))
            level = new_level

    for i, line in enumerate(text.split("\n")):
        if i:
            buffer.append("\n")
            tracker.feed("\n")
        stripped = line.lstrip()
        if not stripped or tracker.continuation or tracker.is_comment_start(stripped[0]):
            buffer.append(line)  # recuo literal, nível não muda
        else:
            move_to(_level(line))
            buffer.append(line[level * INDENT_WIDTH:])
        tracker.feed(line)

    move_to(0)
    flush()
    return segments


class IndentationDecoder:
    """Reconstrói o texto a partir dos tokens de indentação, de forma incremental.

    Feito para a geração token a token: o recuo de uma linha só é escrito
    quando aparece o primeiro caractere que não é branco (até lá não dá para
    saber se a linha está em branco ou é só comentário).
    """

    def __init__(self, lang: str | None) -> None:
        self.tracker = LineTracker(lang)
        self.level = 0
        self.at_line_start = True
        self.pending = ""

    def indent(self) -> None:
        self.level += 1

    def dedent(self) -> None:
        self.level = max(0, self.level - 1)

    def feed(self, text: str) -> str:
        out = []
        for ch in text:
            if not self.at_line_start:
                piece = ch
                if ch == "\n":
                    self.at_line_start = True
            elif ch == "\n":  # linha em branco
                piece, self.pending = self.pending + ch, ""
            elif ch.isspace():
                self.pending += ch
                continue
            else:
                literal = self.tracker.continuation or self.tracker.is_comment_start(ch)
                indent = "" if literal else " " * (self.level * INDENT_WIDTH)
                piece, self.pending = indent + self.pending + ch, ""
                self.at_line_start = False
            self.tracker.feed(piece)
            out.append(piece)
        return "".join(out)

    def finish(self) -> str:
        rest, self.pending = self.pending, ""
        self.tracker.feed(rest)
        return rest


def decode_indentation(segments: list[Segment], lang: str | None) -> str:
    decoder = IndentationDecoder(lang)
    out = []
    for is_control, text in segments:
        if is_control and text == INDENT:
            decoder.indent()
        elif is_control and text == DEDENT:
            decoder.dedent()
        else:
            out.append(decoder.feed(text))
    return "".join(out) + decoder.finish()
