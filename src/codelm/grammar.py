"""Máscara guiada por gramática: só permite tokens que continuam um programa válido.

As gramáticas ficam em ``codelm/grammars/<linguagem>.lark`` (sintaxe Lark do
llguidance) e são escritas sobre a representação do modelo: a indentação chega
como ``<|indent|>``/``<|dedent|>`` e o fim como ``<|endoftext|>``. O llguidance
(Earley + lexer incremental, em Rust) acompanha o estado do parser e calcula a
máscara sobre o vocabulário inteiro a cada passo.

Uso na geração (a partir do primeiro token depois de ``<|code|>`` ou
``<|json_output|>``)::

    constraint = GrammarConstraint("python")
    for _ in range(max_tokens):
        logits = model(...)
        token_id = sample(constrain_logits(logits, constraint.allowed()))
        if not constraint.consume(token_id) or token_id == eos_id:
            break
"""

import re
from functools import cache
from pathlib import Path

import llguidance as llg
import numpy as np

from codelm.tokenizer import TOKENIZER_PATH

GRAMMAR_DIR = Path(__file__).parent / "grammars"
EOS = "<|endoftext|>"

# Marcadores das seções de uma gramática com espaço explícito (ver python.lark):
# regras de comando, regras de expressão (duplicadas para dentro de
# parênteses) e terminais.
BRACKET_START = "// ---------------------------------------------------------------- BRACKET"
BRACKET_END = "// ---------------------------------------------------------------- TERMINAIS"
_RULE = re.compile(r"^([?!]?)([a-z_][a-z0-9_]*)\s*:(.*)$")
_TERMINAL = re.compile(r"^_?[A-Z][A-Z0-9_]*\s*:.*$", re.M)
# Terminais que recebem espaço opcional depois (além dos literais "...").
_SPACED_TERMINALS = ("NAME", "NUMBER", "STRING", "DOTS")
_SYMBOL = re.compile(r'("(?:\\.|[^"\\])*")|\b([A-Za-z_][A-Za-z0-9_]*)\b')
_KEYWORD = re.compile(r'^"[A-Za-z_]\w*"$')


def available() -> list[str]:
    return sorted(p.stem for p in GRAMMAR_DIR.glob("*.lark"))


def _parse_rules(section: str) -> list[tuple[str, str, str]]:
    """(prefixo ?/!, nome, corpo) de cada regra; linhas que começam com | continuam a anterior."""
    rules: list[tuple[str, str, str]] = []
    for line in section.splitlines():
        if match := _RULE.match(line):
            rules.append((match.group(1), match.group(2), match.group(3)))
        elif line.strip().startswith("|") and rules:
            prefix, name, body = rules[-1]
            rules[-1] = (prefix, name, body + " " + line.strip())
    return rules


def _with_spaces(rules: list[tuple[str, str, str]], space: str, rename: set[str] = frozenset()) -> str:
    """Regras com ``space`` depois de cada terminal; as de ``rename`` ganham o sufixo ``_b``."""

    def transform(match: re.Match) -> str:
        literal, word = match.group(1), match.group(2)
        if literal and _KEYWORD.match(literal):
            # O lexer do llguidance só considera o que a gramática espera: onde só
            # cabe "in", leria "initi" como "in" + "iti". A alternativa sem saída
            # NAME _NEVER faz o lexer preferir o nome mais longo, como o Python.
            return f"(({literal} | NAME _NEVER) {space})"
        if literal:
            return f"({literal} {space})"
        if word in rename:
            return word + "_b"
        if word in _SPACED_TERMINALS:
            return f"({word} {space})"
        return word

    suffix = "_b" if rename else ""
    return "\n".join(f"{p}{name}{suffix}: {_SYMBOL.sub(transform, body)}" for p, name, body in rules)


@cache
def grammar_text(lang: str) -> str:
    """Texto final da gramática.

    Nas gramáticas com seção BRACKET: espaço opcional depois de cada terminal
    (``_WS?``) e cópias ``_b`` das expressões para dentro de parênteses, onde
    também valem quebras de linha (``_nl``).
    """
    text = (GRAMMAR_DIR / f"{lang}.lark").read_text()
    if BRACKET_START not in text:
        return text
    statements, rest = text.split(BRACKET_START, 1)
    expressions, terminals = rest.split(BRACKET_END, 1)
    statement_rules, expression_rules = _parse_rules(statements), _parse_rules(expressions)
    names = {name for _, name, _ in expression_rules}
    return "\n".join([
        _with_spaces(statement_rules, "_WS?"),
        _with_spaces(expression_rules, "_WS?"),
        _with_spaces(expression_rules, "_nl", rename=names),
        *_TERMINAL.findall(statements + expressions),  # terminais definidos no meio das regras
        terminals,
    ])


@cache
def _ll_tokenizer() -> llg.LLTokenizer:
    from codelm.tokenizer import load_tokenizer

    eos = load_tokenizer().token_to_id(EOS)
    return llg.LLTokenizer(str(TOKENIZER_PATH), eos_token=eos)


@cache
def _compiled(lang: str) -> str:
    grammar = llg.grammar_from("lark", grammar_text(lang))
    error = llg.LLMatcher.validate_grammar(grammar, _ll_tokenizer())
    if error:
        raise ValueError(f"gramática {lang} inválida: {error}")
    return grammar


class GrammarConstraint:
    """Estado da gramática de uma geração."""

    def __init__(self, lang: str) -> None:
        self.tokenizer = _ll_tokenizer()
        self.matcher = llg.LLMatcher(self.tokenizer, _compiled(lang), log_level=0)

    def allowed(self) -> np.ndarray:
        """Máscara booleana dos tokens que a gramática permite agora."""
        bits = np.frombuffer(self.matcher.compute_bitmask(), dtype=np.uint8)
        return np.unpackbits(bits, bitorder="little")[: self.tokenizer.vocab_size].astype(bool)

    def consume(self, token_id: int) -> bool:
        """Avança com o token escolhido; False se ele não era permitido (a gramática para)."""
        return self.matcher.consume_token(token_id)

    def validate(self, ids: list[int]) -> int:
        """Quantos tokens do começo de ``ids`` a gramática aceita (sem avançar o estado)."""
        return self.matcher.validate_tokens(ids)

    @property
    def error(self) -> str:
        return self.matcher.get_error()
