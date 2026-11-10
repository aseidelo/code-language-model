"""Tradução do texto natural dos dados de SFT com o Claude Haiku (ver docs/sft.md).

Traduz só texto em linguagem natural (pedidos, descrições, docstrings, valores
do JSON); quem chama separa isso do código. Trechos de código citados no texto
(entre crases) são trocados por marcadores antes da tradução e restaurados
depois, para que identificadores nunca sejam traduzidos.

Dois modos:

* ``translate``: uma chamada direta, para testes e volumes pequenos;
* ``submit`` / ``collect``: Message Batches API (metade do preço, resultado em
  até 24 h), para o volume do SFT.

A chave vem de ``ANTHROPIC_API_KEY``.

Uso:
    uv run python -m dataprep.translate try pt "Return the `n` first primes."
    uv run python -m dataprep.translate submit entrada.jsonl      # {"id", "text", "target", "kind"?}
    uv run python -m dataprep.translate status <batch_id>
    uv run python -m dataprep.translate collect <batch_id> saida.jsonl
"""

import argparse
import json
import re
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from codelm.languages import NATURAL_LANGUAGES

MODEL = "claude-haiku-5-5"
# Limite da API por lote.
BATCH_SIZE = 100_000

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "pt": "Brazilian Portuguese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
}

# O que o texto é, para o tom da tradução.
KINDS = {
    "prompt": "a request a programmer writes to a coding assistant",
    "description": "a natural-language explanation of what a piece of code does",
    "docstring": "a docstring or doc comment from source code",
    "json_value": "a text field inside a structured description of code",
}

SYSTEM = """You translate text that appears in programming datasets.

Rules:
- Translate only the natural language. Keep the meaning, tone and length close to the original.
- Never translate or alter markers like ⟦0⟧, ⟦1⟧: copy each one exactly once, in the place where it fits.
- Keep identifiers, function names, file names, URLs, numbers and code-like tokens unchanged.
- Keep line breaks and formatting (lists, Markdown, docstring tags such as @param, :param, Args:, Returns:).
- If the text is already in the target language, return it unchanged."""

OUTPUT_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {"translation": {"type": "string"}},
        "required": ["translation"],
        "additionalProperties": False,
    },
}

# Blocos ``` primeiro (podem conter crases simples), depois `trechos` inline.
CODE_SPAN = re.compile(r"```.*?```|`[^`\n]+`", re.DOTALL)
MARKER = re.compile(r"⟦(\d+)⟧")


def protect(text: str) -> tuple[str, list[str]]:
    """Troca os trechos de código por marcadores ⟦i⟧."""
    spans: list[str] = []

    def mark(match: re.Match) -> str:
        spans.append(match.group(0))
        return f"⟦{len(spans) - 1}⟧"

    return CODE_SPAN.sub(mark, text), spans


def restore(text: str, spans: list[str]) -> str | None:
    """Devolve os trechos de código; ``None`` se algum marcador sumiu ou se repetiu."""
    found = [int(i) for i in MARKER.findall(text)]
    if sorted(found) != list(range(len(spans))):
        return None
    return MARKER.sub(lambda m: spans[int(m.group(1))], text)


def placeholder(source: str, translation: str) -> bool:
    """O modelo devolveu um resumo do formato em vez da tradução: vale reenviar."""
    text = translation.strip()
    truncated = text.endswith("...") and not source.rstrip().endswith(("...", "…"))
    return text.startswith('{"translation"') or truncated


def request_params(text: str, target: str, kind: str = "description") -> MessageCreateParamsNonStreaming:
    if target not in LANGUAGE_NAMES:
        raise ValueError(f"idioma desconhecido: {target!r} (disponíveis: {', '.join(NATURAL_LANGUAGES)})")
    return MessageCreateParamsNonStreaming(
        model=MODEL,
        # Folga para o raciocínio (o Haiku 5.5 pensa por padrão) e para idiomas mais longos que o inglês.
        max_tokens=4096 + 4 * len(text) // 3,
        output_config={"effort": "low", "format": OUTPUT_FORMAT},
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Translate to {LANGUAGE_NAMES[target]}. The text is {KINDS[kind]}.\n\n<text>\n{text}\n</text>",
        }],
    )


def parse_response(message: anthropic.types.Message, spans: list[str]) -> str | None:
    """Tradução já com o código restaurado; ``None`` se a resposta não serve."""
    if message.stop_reason != "end_turn":  # max_tokens, refusal, ...
        return None
    text = next((b.text for b in message.content if b.type == "text"), None)
    if text is None:
        return None
    return restore(json.loads(text)["translation"], spans)


def translate(text: str, target: str, kind: str = "description", client: anthropic.Anthropic | None = None) -> str | None:
    """Traduz um texto com uma chamada direta."""
    client = client or anthropic.Anthropic()
    protected, spans = protect(text)
    message = client.messages.create(**request_params(protected, target, kind))
    return parse_response(message, spans)


def submit(items: list[dict], client: anthropic.Anthropic | None = None) -> list[str]:
    """Envia os itens ({"id", "text", "target", "kind"?}) em lotes; devolve os ids dos lotes.

    Os trechos protegidos não ficam guardados: ``collect`` refaz a proteção a
    partir do mesmo arquivo de entrada.
    """
    client = client or anthropic.Anthropic()
    batch_ids = []
    for start in range(0, len(items), BATCH_SIZE):
        requests = [
            Request(custom_id=str(item["id"]),
                    params=request_params(protect(item["text"])[0], item["target"], item.get("kind", "description")))
            for item in items[start : start + BATCH_SIZE]
        ]
        batch_ids.append(client.messages.batches.create(requests=requests).id)
    return batch_ids


def collect(batch_id: str, items: list[dict], client: anthropic.Anthropic | None = None) -> tuple[dict[str, str], list[str]]:
    """Traduções de um lote terminado e os ids que falharam (para reenviar)."""
    client = client or anthropic.Anthropic()
    texts = {str(item["id"]): item["text"] for item in items}
    translations, failed = {}, []
    for result in client.messages.batches.results(batch_id):
        translation = None
        if result.result.type == "succeeded":
            translation = parse_response(result.result.message, protect(texts[result.custom_id])[1])
        if translation is None:
            failed.append(result.custom_id)
        else:
            translations[result.custom_id] = translation
    return translations, failed


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    try_cmd = commands.add_parser("try", help="traduz um texto na hora")
    try_cmd.add_argument("target", choices=LANGUAGE_NAMES)
    try_cmd.add_argument("text")
    try_cmd.add_argument("--kind", choices=KINDS, default="description")
    submit_cmd = commands.add_parser("submit", help="envia um JSONL para a Batches API")
    submit_cmd.add_argument("input", type=Path)
    status_cmd = commands.add_parser("status", help="andamento de um lote")
    status_cmd.add_argument("batch_id")
    collect_cmd = commands.add_parser("collect", help="baixa as traduções de um lote terminado")
    collect_cmd.add_argument("batch_id")
    collect_cmd.add_argument("output", type=Path)
    collect_cmd.add_argument("--input", type=Path, required=True, help="o mesmo JSONL enviado no submit")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    if args.command == "try":
        print(translate(args.text, args.target, args.kind, client))
    elif args.command == "submit":
        for batch_id in submit(read_jsonl(args.input), client):
            print(batch_id)
    elif args.command == "status":
        batch = client.messages.batches.retrieve(args.batch_id)
        counts = batch.request_counts
        print(f"{batch.processing_status} | ok {counts.succeeded:,} | processando {counts.processing:,} | "
              f"erro {counts.errored:,} | expirado {counts.expired:,}")
    elif args.command == "collect":
        translations, failed = collect(args.batch_id, read_jsonl(args.input), client)
        with args.output.open("w") as f:
            for item_id, translation in translations.items():
                f.write(json.dumps({"id": item_id, "translation": translation}, ensure_ascii=False) + "\n")
        print(f"{len(translations):,} traduções em {args.output}; {len(failed):,} falharam")
        if failed:
            failed_path = args.output.with_suffix(".failed.txt")
            failed_path.write_text("\n".join(failed) + "\n")
            print(f"ids para reenviar em {failed_path}")


if __name__ == "__main__":
    main()
