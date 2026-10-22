"""Mede as gramáticas do sampler (``codelm.grammar``) em código real.

Para cada linguagem com gramática, passa arquivos válidos (checados com o
parser oficial: ``ast`` para Python, ``json`` para JSON) pela máscara, token a
token, como se o modelo os estivesse gerando:

* aceitos: % dos arquivos que a gramática deixa gerar até o fim. O que falta
  é código válido que a máscara bloquearia (deve ser ~0);
* velocidade de validação, em tokens por segundo;
* exemplos de onde a gramática parou, para corrigi-la.

Uso:
    uv run python -m train.evaluate_grammars [--split train] [--limit 1000]
"""

import argparse
import ast
import json
import time
import warnings

from codelm.chat_template import EOS
from codelm.grammar import GrammarConstraint, available
from codelm.tokenizer import encode_code, load_tokenizer
from dataprep.corpus import iter_texts


def is_valid(lang: str, text: str) -> bool:
    try:
        if lang == "python":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ast.parse(text)
        elif lang == "json":
            json.loads(text)
    except (SyntaxError, ValueError):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    eos = tokenizer.token_to_id(EOS)
    for lang in available():
        accepted = files = invalid = tokens = 0
        failures = []
        start = time.perf_counter()
        for text in list(iter_texts(lang, split=args.split))[: args.limit]:
            if not is_valid(lang, text):
                invalid += 1
                continue
            ids = [*encode_code(tokenizer, text, lang), eos]
            valid_prefix = GrammarConstraint(lang).validate(ids)
            files += 1
            tokens += len(ids)
            accepted += valid_prefix == len(ids)
            if valid_prefix < len(ids):
                before = tokenizer.decode(ids[max(0, valid_prefix - 12) : valid_prefix], skip_special_tokens=False)
                after = tokenizer.decode(ids[valid_prefix : valid_prefix + 3], skip_special_tokens=False)
                failures.append((before[-60:], after))
        speed = tokens / (time.perf_counter() - start)
        print(f"{lang}: {accepted}/{files} aceitos ({accepted / files:.1%}); {invalid} inválidos ignorados; {speed:,.0f} tokens/s")
        for before, after in failures[: args.examples]:
            print(f"   parou em {before!r} -> {after!r}")


if __name__ == "__main__":
    main()
