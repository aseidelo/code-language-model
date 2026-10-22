"""Extrai o vocabulário fixo (pré-definido) do tokenizador.

O vocabulário fixo tem três camadas:

* ``control``: tokens especiais do formato de prompt (linguagem, tarefa, FIM...).
* ``whitespace``: quebras de linha + indentação e sequências de espaços/tabs.
* ``languages``: por linguagem, os terminais literais da gramática tree-sitter
  (palavras-chave, operadores, pontuação), sempre incluídos, e um vocabulário
  complementar (tipos/funções embutidos, tags e propriedades do MDN para
  HTML/CSS). O complementar é só uma lista de candidatos: entra apenas o que
  aparece pelo menos ``MIN_VOCAB_COUNT`` vezes no corpus de treino da linguagem.

Requer os dados preparados por ``dataprep.download``.

Uso:
    uv run python -m train.fixed_tokens
"""

import builtins
import json
import keyword
import re
from collections import Counter

import tree_sitter_language_pack as tslp

from codelm.languages import LANGUAGES
from codelm.tokenizer import FIXED_TOKENS_PATH, all_fixed_tokens, control_tokens, whitespace_tokens
from dataprep.corpus import iter_texts
from dataprep.mdn import BCD_VERSION, bcd_keys, load_bcd

MIN_VOCAB_COUNT = 100
# Limite de texto lido por linguagem ao contar frequências.
COUNT_MAX_CHARS = 40_000_000
# Em HTML/CSS o hífen faz parte dos nomes (``grid-template-columns``, ``data-id``).
WORD_PATTERN = re.compile(r"[A-Za-z_$][\w$]*")
HYPHEN_WORD_PATTERN = re.compile(r"@?[A-Za-z_][\w-]*")
HYPHENATED_NAMES = {"html", "css"}

# Terminais de gramática que não valem um token: extensões específicas do MSVC
# no tree-sitter-c e formas de Python 2.
GRAMMAR_EXCLUDE = {
    "c": {
        "__based", "__cdecl", "__clrcall", "__declspec", "__except", "__fastcall",
        "__finally", "__forceinline", "__leave", "__stdcall", "__thiscall", "__try",
        "__unaligned", "__vectorcall", "_unaligned", "_alignof",
    },
    "python": {"<>", "exec"},
}

# Identificadores muito frequentes que não são terminais da gramática
# (tipos primitivos, embutidos, biblioteca padrão mais comum).
SUPPLEMENTS: dict[str, list[str]] = {
    "python": ["self", "cls", "__init__", "__name__", "__main__", "__file__", "__dict__"],
    "javascript": [
        "this", "null", "undefined", "true", "false", "console", "log", "error",
        "document", "window", "require", "module", "exports", "Promise", "JSON",
        "Object", "Array", "String", "Number", "Math", "Error", "Map", "Set",
        "length", "prototype", "addEventListener", "querySelector", "fetch",
    ],
    "typescript": [
        "this", "null", "undefined", "true", "false", "console", "Promise", "Record",
        "Partial", "Readonly", "Pick", "Omit", "Array", "Map", "Set", "Error",
        "bigint",
    ],
    "java": [
        "this", "super", "null", "true", "false", "void", "boolean", "String",
        "Object", "System", "out", "println", "Integer", "Long", "Double", "Boolean",
        "List", "ArrayList", "Map", "HashMap", "Set", "HashSet", "Override",
        "Exception", "RuntimeException", "main", "args", "var",
    ],
    "c": [
        "int", "char", "float", "double", "void", "bool", "size_t", "ssize_t",
        "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t",
        "uint32_t", "uint64_t", "FILE", "printf", "fprintf", "sprintf", "snprintf",
        "malloc", "calloc", "realloc", "free", "memcpy", "memset", "strlen",
        "strcmp", "stdio", "stdlib", "string", "main", "argc", "argv", "true", "false",
    ],
    "rust": [
        "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128",
        "usize", "f32", "f64", "bool", "char", "str", "String", "Vec", "Option",
        "Some", "None", "Result", "Ok", "Err", "Box", "Rc", "Arc", "self", "Self",
        "crate", "super", "println", "format", "vec", "panic", "assert_eq",
        "unwrap", "clone", "derive", "Debug", "Clone", "Default", "new",
    ],
    "html": ["html", "DOCTYPE", "charset", "utf-8", "viewport", "stylesheet"],
    "json": ["true", "false", "null"],
    "yaml": ["true", "false", "null", "yes", "no", "on", "off"],
    "css": [
        "px", "em", "rem", "vh", "vw", "fr", "deg", "ms", "important",
        "inherit", "initial", "unset", "auto", "none", "transparent",
        "hover", "focus", "active", "before", "after", "root", "var",
    ],
}


def grammar_literals(grammar: str) -> set[str]:
    """Terminais literais (nós anônimos visíveis) de uma gramática tree-sitter."""
    lang = tslp.get_language(grammar)
    return {
        lang.node_kind_for_id(i)
        for i in range(lang.node_kind_count)
        if lang.node_kind_is_visible(i) and not lang.node_kind_is_named(i)
    }


def is_valid_literal(token: str) -> bool:
    # Terminais com espaço interno (``is not``, ``static get``) viram dois tokens.
    return bool(token) and not any(ch.isspace() for ch in token)






def html_vocabulary(bcd: dict) -> set[str]:
    html = bcd["html"]
    tokens = set(bcd_keys(html["global_attributes"]))
    for element, node in html["elements"].items():
        tokens.add(element)
        tokens.update(bcd_keys(node))  # atributos específicos do elemento
    return tokens


def css_vocabulary(bcd: dict) -> set[str]:
    css = bcd["css"]
    tokens = {f"@{rule}" for rule in css["at-rules"]}
    tokens.update(bcd_keys(css["selectors"]))
    tokens.update(bcd_keys(css["types"]))
    for prop, node in css["properties"].items():
        tokens.add(prop)
        tokens.update(bcd_keys(node))  # valores aceitos pela propriedade
    return tokens


def python_vocabulary() -> set[str]:
    public_builtins = {name for name in dir(builtins) if not name.startswith("_")}
    return set(keyword.kwlist) | set(keyword.softkwlist) | public_builtins


def word_counts(lang: str) -> Counter:
    pattern = HYPHEN_WORD_PATTERN if lang in HYPHENATED_NAMES else WORD_PATTERN
    counts = Counter()
    for text in iter_texts(lang, max_chars=COUNT_MAX_CHARS):
        counts.update(pattern.findall(text))
    return counts






def build_fixed_tokens() -> dict:
    bcd = load_bcd()
    languages = {}
    for name, lang in LANGUAGES.items():
        grammar = set().union(*(grammar_literals(g) for g in lang.grammars))
        grammar = {t for t in grammar if is_valid_literal(t)} - GRAMMAR_EXCLUDE.get(name, set())

        extra = set(SUPPLEMENTS.get(name, []))
        if name == "python":
            extra |= python_vocabulary()
        elif name == "html":
            extra |= html_vocabulary(bcd)
        elif name == "css":
            extra |= css_vocabulary(bcd)
        extra -= grammar

        counts = word_counts(name)
        kept = {t for t in extra if counts[t] >= MIN_VOCAB_COUNT}
        languages[name] = {
            "grammar": sorted(grammar),
            "vocabulary": sorted(kept, key=lambda t: (-counts[t], t)),
            "vocabulary_candidates": len(extra),
        }

    return {
        "sources": {
            "tree-sitter-language-pack": tslp.__version__ if hasattr(tslp, "__version__") else None,
            "browser-compat-data": BCD_VERSION,
            "python-keywords": f"{keyword.__name__} (Python do ambiente)",
            "min_vocab_count": MIN_VOCAB_COUNT,
        },
        "control": control_tokens(),
        "whitespace": whitespace_tokens(),
        "languages": languages,
    }




def main() -> None:
    fixed = build_fixed_tokens()
    FIXED_TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXED_TOKENS_PATH.write_text(json.dumps(fixed, indent=1, ensure_ascii=False) + "\n")

    print(f"Salvo em {FIXED_TOKENS_PATH}")
    print(f"  controle:   {len(fixed['control'])}")
    print(f"  whitespace: {len(fixed['whitespace'])}")
    for name, lang in fixed["languages"].items():
        print(
            f"  {name:<11} gramática={len(lang['grammar']):>4}  "
            f"vocabulário={len(lang['vocabulary']):>4} de {lang['vocabulary_candidates']:>4} candidatos"
        )
    print(f"  total único (sem controle): {len(all_fixed_tokens(fixed))}")


if __name__ == "__main__":
    main()
