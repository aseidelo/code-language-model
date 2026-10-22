"""Mede a qualidade do tokenizador na validação de cada linguagem e idioma.

Métricas:
* caracteres por token (maior = comprime melhor = contexto rende mais). No
  nosso tokenizador o código passa antes pela indentação relativa, como o
  modelo vai ver;
* ida e volta no código: ``decode_code(encode_code(texto)) == texto``;
* comparação com tokenizadores de referência (GPT-2 e StarCoder2).

Uso:
    uv run python -m train.evaluate_tokenizer
"""

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from codelm.languages import LANGUAGES, NATURAL_LANGUAGES
from codelm.tokenizer import decode_code, encode_code, load_tokenizer
from dataprep.corpus import iter_natural_texts, iter_texts

REFERENCES = {
    "gpt2 (50k)": "openai-community/gpt2",
    "starcoder2 (49k)": "bigcode/starcoder2-3b",
}
NATURAL_VALID_DOCS = 300


def load_references() -> dict[str, Tokenizer]:
    tokenizers = {}
    for name, repo in REFERENCES.items():
        try:
            tokenizers[name] = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        except Exception as error:  # repositório indisponível/gated: segue sem ele
            print(f"Aviso: referência {name} ignorada ({error.__class__.__name__})")
    return tokenizers


def reference_chars_per_token(tokenizer: Tokenizer, texts: list[str]) -> float:
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    return sum(map(len, texts)) / sum(len(e.ids) for e in encodings)


def print_table(title: str, rows: dict[str, list[float]], names: list[str], extra: dict[str, str]) -> None:
    print(f"\n{title:<11} " + " ".join(f"{n:>17}" for n in names) + ("   ida e volta" if extra else ""))
    for key, ratios in rows.items():
        print(f"{key:<11} " + " ".join(f"{r:>17.2f}" for r in ratios) + (f"   {extra[key]}" if extra else ""))


def main() -> None:
    ours = load_tokenizer()
    references = load_references()
    names = [f"codelm ({ours.get_vocab_size() // 1000}k)", *references]

    code_rows, roundtrip = {}, {}
    for lang in LANGUAGES:
        texts = list(iter_texts(lang, split="valid"))
        encoded = [encode_code(ours, t, lang) for t in texts]
        ok = sum(decode_code(ours, ids, lang) == t for ids, t in zip(encoded, texts))
        ours_ratio = sum(map(len, texts)) / sum(map(len, encoded))
        code_rows[lang] = [ours_ratio, *(reference_chars_per_token(r, texts) for r in references.values())]
        roundtrip[lang] = f"{ok}/{len(texts)} ok"
    print_table("código", code_rows, names, roundtrip)

    natural_rows = {}
    for lang in NATURAL_LANGUAGES:
        texts = list(iter_natural_texts(lang, split="valid"))[:NATURAL_VALID_DOCS]
        natural_rows[lang] = [reference_chars_per_token(t, texts) for t in [ours, *references.values()]]
    print_table("idioma", natural_rows, names, {})
    print("\n(caracteres por token: maior é melhor)")


if __name__ == "__main__":
    main()
