"""Lexer de contexto: diz se o texto está em código ou em trecho livre.

Trecho livre é string, comentário ou texto (conteúdo de tags HTML). A
indentação relativa (``codelm.indentation``) usa isso para não trocar por
tokens o recuo de linhas que começam dentro de uma string ou comentário.

O lexer anda um caractere por vez e nunca olha à frente, então funciona igual
sobre um arquivo completo (codificando) e sobre o texto que está sendo gerado
(decodificando token a token). Ele não precisa acertar 100% da semântica de
cada linguagem, só ser consistente entre os dois lados; em caso de dúvida,
fica em código.
"""

import re
from dataclasses import dataclass

CODE = 0
FREE = 1

@dataclass(frozen=True)
class LexerSpec:
    line_comments: tuple[str, ...] = ()
    block_comments: tuple[tuple[str, str], ...] = ()
    # Delimitadores de string de um caractere, com escape por ``\\``.
    quotes: tuple[str, ...] = ()
    # Delimitadores que podem atravessar linhas (os outros fecham no fim da linha).
    multiline_quotes: tuple[str, ...] = ()
    triple_quotes: bool = False  # Python e text blocks do Java: \"\"\" e '''
    rust_raw_strings: bool = False  # r"...", r#"..."#
    # YAML: '#' só abre comentário depois de espaço ou no início da linha, e
    # aspas só abrem string no começo de um valor (não em ``don't``).
    yaml_rules: bool = False


SPECS: dict[str, LexerSpec] = {
    "python": LexerSpec(line_comments=("#",), quotes=('"', "'"), triple_quotes=True),
    "javascript": LexerSpec(
        line_comments=("//",), block_comments=(("/*", "*/"),), quotes=('"', "'", "`"), multiline_quotes=("`",)
    ),
    "typescript": LexerSpec(
        line_comments=("//",), block_comments=(("/*", "*/"),), quotes=('"', "'", "`"), multiline_quotes=("`",)
    ),
    "java": LexerSpec(line_comments=("//",), block_comments=(("/*", "*/"),), quotes=('"', "'"), triple_quotes=True),
    "c": LexerSpec(line_comments=("//",), block_comments=(("/*", "*/"),), quotes=('"', "'")),
    # Em Rust ' também abre lifetimes ('a), então só " abre string.
    "rust": LexerSpec(line_comments=("//",), block_comments=(("/*", "*/"),), quotes=('"',), rust_raw_strings=True),
    "css": LexerSpec(block_comments=(("/*", "*/"),), quotes=('"', "'")),
    "json": LexerSpec(quotes=('"',)),
    "yaml": LexerSpec(line_comments=("#",), quotes=('"', "'"), yaml_rules=True),
}


class ContextLexer:
    """Lexer genérico para linguagens de sintaxe "C-like" e afins."""

    def __init__(self, spec: LexerSpec) -> None:
        self.spec = spec
        self.recent = ""  # últimos caracteres vistos (para operadores de vários caracteres)
        self.mode: str | None = None  # None (código), "line", "block", "string"
        self.close = ""  # delimitador que fecha o modo atual
        self.since = 0  # caracteres desde que o modo começou
        self.escaped = False
        self.escapes = True
        self.multiline = False
        self.empty_string_closed = ""  # aspa de uma string vazia que acabou de fechar ("")

    @property
    def context(self) -> int:
        return CODE if self.mode is None else FREE

    def _enter(self, mode: str, close: str, escapes: bool = True, multiline: bool = True) -> None:
        self.mode, self.close, self.since = mode, close, 0
        self.escaped, self.escapes, self.multiline = False, escapes, multiline

    def _value_start(self) -> bool:
        """YAML: o caractere atual começa um valor (início da linha ou depois de ``:``, ``-``, ``[`` ...)."""
        before = self.recent[:-1].rstrip(" \t")
        return not before or before[-1] in "\n:-[{,?"

    def feed(self, text: str) -> int:
        for ch in text:
            self.recent = (self.recent + ch)[-32:]
            if self.mode is None:
                self._feed_code(ch)
            else:
                self._feed_free(ch)
        return self.context

    def _feed_code(self, ch: str) -> None:
        spec = self.spec
        empty_quote, self.empty_string_closed = self.empty_string_closed, ""

        for opener, closer in spec.block_comments:
            if self.recent.endswith(opener):
                self._enter("block", closer)
                return
        for marker in spec.line_comments:
            if self.recent.endswith(marker):
                before = self.recent[: -len(marker)]
                if spec.yaml_rules and before and before[-1] not in " \t\n":
                    continue
                self._enter("line", "\n")
                return
        if ch not in spec.quotes:
            return
        if spec.yaml_rules and not self._value_start():
            return
        if spec.triple_quotes and ch in "\"'" and empty_quote == ch:
            self._enter("string", ch * 3)  # "" seguido de " é o início de \"\"\"
            return
        if spec.rust_raw_strings and ch == '"' and (raw := re.search(r"(?:^|\W)b?r(#*)\"$", self.recent)):
            self._enter("string", '"' + raw.group(1), escapes=False)
            return
        self._enter("string", ch, multiline=ch in spec.multiline_quotes or spec.yaml_rules)

    def _feed_free(self, ch: str) -> None:
        self.since += 1
        if self.mode == "line":
            if ch == "\n":
                self.mode = None
            return
        if self.mode == "string":
            if self.escaped:
                self.escaped = False
                return
            if ch == "\\" and self.escapes:
                self.escaped = True
                return
            if ch == "\n" and not self.multiline and len(self.close) == 1:
                self.mode = None  # string não fechada na linha: volta para código
                return
        if self.since >= len(self.close) and self.recent.endswith(self.close):
            if self.mode == "string" and self.since == 1:
                self.empty_string_closed = ch
            self.mode = None


class HtmlLexer:
    """HTML: o interior das tags é código; texto, comentários, valores de atributos
    e o conteúdo de ``<script>``/``<style>`` são trechos livres."""

    RAW_TAGS = ("script", "style")

    def __init__(self) -> None:
        self.mode = "text"  # text, tag, attr, comment, raw
        self.tag = ""
        self.close = ""
        self.recent = ""

    @property
    def context(self) -> int:
        return CODE if self.mode == "tag" else FREE

    def feed(self, text: str) -> int:
        for ch in text:
            self.recent = (self.recent + ch)[-16:]
            if self.mode == "text":
                if ch == "<":
                    self.mode, self.tag = "tag", ch
            elif self.mode == "tag":
                self.tag += ch
                if self.tag == "<!--":
                    self.mode = "comment"
                elif ch in "\"'":
                    self.mode, self.close = "attr", ch
                elif ch == ">":
                    name = re.match(r"<(\w*)", self.tag).group(1).lower()
                    self.mode = "raw" if name in self.RAW_TAGS else "text"
                    self.close = f"</{name}"
            elif self.mode == "attr":
                if ch == self.close:
                    self.mode = "tag"
            elif self.mode == "comment":
                if self.recent.endswith("-->"):
                    self.mode = "text"
            elif self.mode == "raw" and self.recent.lower().endswith(self.close):
                self.mode, self.tag = "tag", self.close
        return self.context


def make_lexer(lang: str | None) -> ContextLexer | HtmlLexer:
    """Lexer da linguagem; ``None`` (linguagem desconhecida) nunca sai de código."""
    if lang == "html":
        return HtmlLexer()
    return ContextLexer(SPECS.get(lang, LexerSpec()))
