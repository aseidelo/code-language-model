"""Treina o tokenizador BPE byte-level e força os tokens fixos no vocabulário.

Etapas:
1. pré-tokenização específica para código (identificadores inteiros, dígitos
   isolados, um operador da gramática por token);
2. treino BPE no corpus: código (com a indentação já trocada por
   ``<|indent|>``/``<|dedent|>``), com o mesmo volume por linguagem, e cerca de
   30% de texto natural nos idiomas priorizados;
3. para cada token fixo (e sua versão com espaço antes, ``return`` e
   `` return``) que o BPE não aprendeu, acrescenta no FIM da lista de merges as
   junções que o formam. Merges novos no fim só se aplicam depois de todos os
   aprendidos, então não mudam a tokenização do resto do texto;
4. verifica que todo token fixo virou exatamente um token.

A pré-tokenização e os tokens de controle são definidos em ``codelm.tokenizer``.
Requer ``dataprep.download`` e ``train.fixed_tokens``.

Uso:
    uv run python -m train.train_tokenizer [--vocab-size 16384]
"""

import argparse
import json
from collections.abc import Iterator

from tokenizers import Tokenizer, pre_tokenizers, trainers

from codelm.languages import LANGUAGES, NATURAL_LANGUAGES
from codelm.tokenizer import FIXED_TOKENS_PATH, TOKENIZER_PATH, all_fixed_tokens, build_tokenizer, code_segments
from dataprep.corpus import iter_natural_texts, iter_texts

# Mesmo volume por linguagem para nenhuma dominar o vocabulário.
TRAIN_MAX_CHARS_PER_LANG = 40_000_000
# ~370M de caracteres de código + 6 x 28M de texto natural = ~30% de texto.
TRAIN_MAX_CHARS_PER_NATURAL_LANG = 28_000_000


def training_corpus() -> Iterator[list[str]]:
    """Um lote por documento. No código, cada trecho entre tokens de indentação é um texto separado."""
    for lang in LANGUAGES:
        for text in iter_texts(lang, max_chars=TRAIN_MAX_CHARS_PER_LANG):
            yield [t for is_control, t in code_segments(text, lang) if not is_control and t]
    for lang in NATURAL_LANGUAGES:
        for text in iter_natural_texts(lang, max_chars=TRAIN_MAX_CHARS_PER_NATURAL_LANG):
            yield [text]


def bpe_encode(piece: str, ranks: dict[tuple[str, str], int]) -> list[str]:
    """BPE de uma palavra já pré-tokenizada: junta sempre o par de menor rank."""
    parts = list(piece)
    while len(parts) > 1:
        pairs = [(ranks.get((a, b)), i) for i, (a, b) in enumerate(zip(parts, parts[1:]))]
        rank, i = min((p for p in pairs if p[0] is not None), default=(None, None))
        if rank is None:
            break
        parts[i : i + 2] = [parts[i] + parts[i + 1]]
    return parts


def fixed_pieces(tokenizer: Tokenizer, fixed: list[str]) -> list[str]:
    """Palavras pré-tokenizadas (em bytes) que devem virar um único token.

    Um token fixo que a pré-tokenização quebra (``@media`` -> ``@`` ``media``)
    é garantido parte por parte.
    """
    pieces = {}
    for token in fixed:
        variants = [token] if token[0].isspace() else [token, " " + token]
        for variant in variants:
            for piece, _ in tokenizer.pre_tokenizer.pre_tokenize_str(variant):
                pieces[piece] = None
    return list(pieces)


def load_merges(state: dict) -> list[tuple[str, str]]:
    # Versões antigas do ``tokenizers`` salvam merges como "a b"; as novas, como ["a", "b"].
    return [tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in state["model"]["merges"]]


def force_tokens(tokenizer: Tokenizer, pieces: list[str]) -> tuple[Tokenizer, int]:
    """Acrescenta merges no fim para que cada peça vire um token. Retorna o número de tokens novos."""
    state = json.loads(tokenizer.to_str())
    vocab: dict[str, int] = state["model"]["vocab"]
    merges = load_merges(state)
    ranks = {pair: i for i, pair in enumerate(merges)}

    added = 0
    for piece in pieces:
        parts = bpe_encode(piece, ranks)
        while len(parts) > 1:
            pair = (parts[0], parts[1])
            merged = parts[0] + parts[1]
            if pair not in ranks:
                ranks[pair] = len(merges)
                merges.append(pair)
            if merged not in vocab:
                vocab[merged] = len(vocab)
                added += 1
            parts = [merged, *parts[2:]]

    state["model"]["merges"] = [list(m) for m in merges]
    return Tokenizer.from_str(json.dumps(state)), added


def check_fixed(tokenizer: Tokenizer, pieces: list[str]) -> list[str]:
    """Peças que o tokenizador real NÃO codifica como um único token (deve ser vazio)."""
    return [p for p in pieces if len(tokenizer.model.tokenize(p)) != 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab-size", type=int, default=16_384, help="tamanho alvo do vocabulário final")
    args = parser.parse_args()

    fixed = json.loads(FIXED_TOKENS_PATH.read_text())
    pieces = fixed_pieces(build_tokenizer(fixed), all_fixed_tokens(fixed))

    # Não dá para saber antes quantos tokens fixos o BPE vai aprender sozinho:
    # treina, força, e se passou do alvo treina de novo com menos tokens.
    bpe_size = args.vocab_size
    for round_ in range(1, 5):
        print(f"Rodada {round_}: treinando BPE com {bpe_size:,} tokens")
        tokenizer = build_tokenizer(fixed)
        trainer = trainers.BpeTrainer(
            vocab_size=bpe_size,
            min_frequency=2,
            special_tokens=fixed["control"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tokenizer.train_from_iterator(training_corpus(), trainer=trainer)
        learned = sum(piece in tokenizer.get_vocab() for piece in pieces)
        tokenizer, added = force_tokens(tokenizer, pieces)
        overshoot = tokenizer.get_vocab_size() - args.vocab_size
        if overshoot <= 0:
            break
        bpe_size -= overshoot

    missing = check_fixed(tokenizer, pieces)
    if missing:
        raise RuntimeError(f"{len(missing)} tokens fixos não viraram um único token: {missing[:20]}")

    tokenizer.save(str(TOKENIZER_PATH))
    print(f"Salvo em {TOKENIZER_PATH}")
    print(f"  vocabulário final:      {tokenizer.get_vocab_size():,}")
    print(f"  aprendidos pelo BPE:    {tokenizer.get_vocab_size() - added:,}")
    print(f"  peças fixas:            {len(pieces):,} ({learned:,} o BPE aprendeu sozinho)")
    print(f"  tokens novos forçados:  {added:,}")


if __name__ == "__main__":
    main()
