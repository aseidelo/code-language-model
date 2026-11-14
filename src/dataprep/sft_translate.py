"""Tradução local de pedidos (en -> pt/es/fr/de/it) com Opus-MT na XPU.

Desenho em docs/sft-v2.md ("Tradução local"). Resumo:

* Só a prosa é traduzida. Blocos de código ficam intactos. Trechos entre crases, termos do glossário
  (``assets/sft/glossario.txt``) e identificadores viram marcadores ``X0X`` e voltam depois. Os
  identificadores são os nomes com ``_``, dígitos, camelCase ou ponto, e as palavras que também
  aparecem no código do exemplo (``nums``, ``target``).
* A prosa é traduzida linha a linha (linhas longas por sentença), mantendo os prefixos de markdown.
* Rejeita quando um marcador se perde, o detector acusa outro idioma ou o tamanho foge do esperado.

Piloto: ``piloto --por-idioma 50`` sorteia exemplos em inglês de várias fontes e linguagens (cada
idioma recebe exemplos diferentes), traduz o pedido e grava ``v2/traducao/piloto.parquet`` e
``v2/revisao/amostra-traducao.json`` para a página de revisão:

    uv run python -m dataprep.sft_translate piloto
    uv run python -m eval.sft_review servir --traducao
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from random import Random

import pyarrow as pa
import pyarrow.parquet as pq

from codelm.languages import ASSETS_DIR
from dataprep.sft_bugfix import FONTES_DIR
from dataprep.sft_convert import FENCE, detect_nls

TRANSLATION_DIR = FONTES_DIR.parent / "traducao"
REVIEW_SAMPLE = FONTES_DIR.parent / "revisao" / "amostra-traducao.json"
GLOSSARY = ASSETS_DIR / "sft" / "glossario.txt"
TARGETS = ("pt", "es", "fr", "de", "it")
# tc-big onde existe; en-de só tem o modelo base.
MODELS = {"pt": ("Helsinki-NLP/opus-mt-tc-big-en-pt", ">>por<< "), "es": ("Helsinki-NLP/opus-mt-tc-big-en-es", ""),
          "fr": ("Helsinki-NLP/opus-mt-tc-big-en-fr", ""), "it": ("Helsinki-NLP/opus-mt-tc-big-en-it", ""),
          "de": ("Helsinki-NLP/opus-mt-en-de", "")}
BATCH = 48
MAX_SENTENCE_CHARS = 400

# Palavras comuns do inglês que aparecem em código mas devem ser traduzidas no texto.
COMMON = set("""a an the and or not of to in on for with by from as at is are be if else then return returns
returned value values number numbers list lists item items element elements string strings result results data
input output name names key keys count index first last length size sum max min new get set add remove
find check sort sorted file files line lines word words text type print read write open close time date
true false none null user users function functions method class object array arrays map filter call
test tests error errors message order total all any each every this that it its""".split())

PLACEHOLDER = "X{}X"
INLINE = re.compile(r"`[^`\n]+`|\"[^\"\n]{1,80}\"|(?<!\w)'[^'\n]{1,40}'(?!\w)")  # crases e literais entre aspas
IDENTIFIER = re.compile(r"\b(?:[A-Za-z_]\w*\.)+\w+(?:\(\))?|\b\w+\(\)|\b[a-z]+[A-Z]\w*\b|\b\w*_\w*\b|\b[A-Za-z]+\d+\w*\b")
SHELL_VAR = re.compile(r"\$\{[^}\n]+\}|\$\w+|\$[@#?*!-]")  # $1, $HOME, ${x}: senão vira "US $ 1"
URL = re.compile(r"https?://\S+|\S+\.(?:py|js|ts|rs|java|c|h|sh|json|yaml|yml|html|css|md|txt|csv)\b")
WORD = re.compile(r"[A-Za-z_]\w*")
MARKDOWN_PREFIX = re.compile(r"^(\s*(?:[-*+]|\d+[.)]|#{1,6}|>)\s+|\s+)")
SENTENCE = re.compile(r"(?<=[.!?:])\s+(?=[A-Z0-9`\"'(])")


@lru_cache(maxsize=1)
def glossary() -> re.Pattern:
    terms = [t.strip() for t in GLOSSARY.read_text().splitlines() if t.strip() and not t.startswith("#")]
    sensitive = sorted((t for t in terms if t != t.lower()), key=len, reverse=True)
    insensitive = sorted((t for t in terms if t == t.lower()), key=len, reverse=True)
    parts = []
    if sensitive:
        parts.append(r"(?-i:" + "|".join(re.escape(t) for t in sensitive) + r")")
    if insensitive:
        parts.append("(?i:" + "|".join(re.escape(t) + "s?" for t in insensitive) + ")")
    return re.compile(r"(?<![\w.#+])(?:" + "|".join(parts) + r")(?![\w+#])")  # (?i:) aplicado em mask


# Nome em posição de identificador no código: perto de ( [ = , . : ) (parâmetros, variáveis, chamadas).
CODE_NAME = re.compile(r"(?<=[(\[=,.])\s*([A-Za-z_]\w*)|([A-Za-z_]\w*)\s*(?=[(\[=,:)\]])")


def prompt_code(prompt: str) -> str:
    """Só os blocos de código do pedido: a prosa dele tem "`people`: an integer", que pareceria identificador."""
    return "\n".join(m.group(0) for m in FENCE.finditer(prompt or ""))


CODE_NOISE = re.compile(r"/\*.*?\*/|//[^\n]*|(?<![\w$])#[^\n]*|--[^\n]*|\"\"\".*?\"\"\"|\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*'",
                        re.S)


def code_words(*texts: str | None) -> set[str]:
    """Nomes que o código usa como identificador; no pedido, ficam sem tradução (``nums``, ``target``)."""
    words = set()
    for text in texts:
        text = CODE_NOISE.sub(" ", text or "")  # comentários e strings têm prosa, não identificadores
        for m in CODE_NAME.finditer(text):
            w = m.group(1) or m.group(2)
            if len(w) > 1 and w.lower() not in COMMON:
                words.add(w)
    return words


def mask(line: str, protected: set[str]) -> tuple[str, list[str]]:
    """Uma passada só com todos os padrões, para um marcador nunca casar com outro padrão depois."""
    slots: list[str] = []
    patterns = [INLINE.pattern, SHELL_VAR.pattern, URL.pattern, IDENTIFIER.pattern, glossary().pattern]
    if protected:
        names = "|".join(re.escape(w) for w in sorted(protected, key=len, reverse=True))
        patterns.append(rf"(?<![\w.])(?:{names})(?!\w)")
    combined = re.compile("|".join(f"(?:{p})" for p in patterns))

    def keep(m: re.Match) -> str:
        slots.append(m.group(0))
        return PLACEHOLDER.format(len(slots) - 1)

    return combined.sub(keep, line), slots


def unmask(text: str, slots: list[str]) -> str | None:
    """Recoloca os trechos; None se algum marcador sumiu ou duplicou."""
    for i, original in enumerate(slots):
        marker = PLACEHOLDER.format(i)
        if text.count(marker) != 1:
            return None
        text = text.replace(marker, original)
    return None if re.search(r"X\d+X", text) else text


def is_code_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    letters = sum(ch.isalpha() for ch in stripped)
    if line.startswith(("    ", "\t")) or letters < 0.5 * len(stripped):
        return True
    return bool(re.match(r"^(import|from|def|class|public|private|#include|fn|let|const|var|\$|>>>)\b", stripped))


def segments(text: str, protected: set[str]) -> list[dict]:
    """Quebra o texto em pedaços: os que se traduzem (prosa mascarada) e os que ficam como estão."""
    out: list[dict] = []
    last = 0
    for m in FENCE.finditer(text):
        out.extend(prose_segments(text[last:m.start()], protected))
        out.append({"keep": m.group(0)})
        last = m.end()
    out.extend(prose_segments(text[last:], protected))
    return out


def prose_segments(text: str, protected: set[str]) -> list[dict]:
    out: list[dict] = []
    for line in text.splitlines(keepends=True):
        body, newline = (line[:-1], "\n") if line.endswith("\n") else (line, "")
        if is_code_line(body):
            out.append({"keep": line})
            continue
        prefix_match = MARKDOWN_PREFIX.match(body)
        prefix = prefix_match.group(1) if prefix_match else ""
        rest = body[len(prefix):]
        out.append({"keep": prefix})
        sentences = SENTENCE.split(rest) if len(rest) > MAX_SENTENCE_CHARS else [rest]
        for j, sentence in enumerate(sentences):
            masked, slots = mask(sentence, protected)
            if not re.search(r"[A-Za-z]{2}", re.sub(r"X\d+X", "", masked)):
                out.append({"keep": sentence})
            else:
                out.append({"src": masked, "slots": slots})
            if j < len(sentences) - 1:
                out.append({"keep": " "})
        out.append({"keep": newline})
    return out


class Translator:
    def __init__(self, nl: str, device: str = "xpu"):
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        name, self.prefix = MODELS[nl]
        self.torch = torch
        self.tokenizer = MarianTokenizer.from_pretrained(name)
        # fp16 dá overflow (saídas vazias) em algumas sentenças; bf16 tem a faixa do fp32.
        self.model = MarianMTModel.from_pretrained(name).to(device, dtype=torch.bfloat16).eval()
        self.device = device

    def __call__(self, sentences: list[str], beams: int = 4) -> list[str]:
        out: list[str] = [""] * len(sentences)
        order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            batch = self.tokenizer([self.prefix + sentences[i] for i in idx], return_tensors="pt", padding=True,
                                   truncation=True, max_length=256).to(self.device)
            with self.torch.no_grad():
                generated = self.model.generate(**batch, num_beams=beams, max_new_tokens=320)
            for i, text in zip(idx, self.tokenizer.batch_decode(generated, skip_special_tokens=True)):
                out[i] = text
        return out


def translate_texts(translator: Translator, nl: str, items: list[tuple[str, set[str]]]) -> list[tuple[str | None, str]]:
    """(tradução ou None, motivo) para cada (texto, palavras protegidas)."""
    plans = [segments(text, protected) for text, protected in items]
    sources = [seg["src"] for plan in plans for seg in plan if "src" in seg]
    translated = iter(translator(sources)) if sources else iter(())
    results = []
    for (text, _), plan in zip(items, plans):
        pieces, reason = [], "ok"
        for seg in plan:
            if "keep" in seg:
                pieces.append(seg["keep"])
                continue
            restored = unmask(next(translated), seg["slots"])
            if restored is None:
                reason = "marcador perdido"
                restored = ""
            pieces.append(restored)
        result = "".join(pieces)
        if reason == "ok":
            ratio = len(result) / max(len(text), 1)
            detected = detect_nls([result])[0]
            if result.strip() == text.strip():
                reason = "nao_traduzido"
            elif not 0.6 <= ratio <= 2.0:
                reason = f"tamanho ({ratio:.2f})"
            elif detected not in (nl, "en"):
                reason = f"idioma ({detected})"
        results.append((result if reason == "ok" else None, reason if reason != "ok" else "ok", ))
    return results


# --- piloto ------------------------------------------------------------------------------

PILOT_SOURCES = ["kodcode", "open-code-instruct", "magicoder-oss", "codefeedback", "glaive-v3", "opc-stage1-largescale",
                 "bash-instruct-iii", "nl2bash-custom", "corpus-v2", "acecode-v2", "mceval-instruct", "rosetta-code"]
PILOT_LANGS = ("python", "c", "typescript", "rust", "javascript", "java", "bash")
FIELDS = ["id", "task", "lang", "nl", "level", "prompt", "code", "response", "kind", "fonte"]


def pilot_pool(seed: int) -> dict[str, list[dict]]:
    r = Random(seed)
    pool: dict[str, list[dict]] = {lang: [] for lang in PILOT_LANGS}
    for fonte in PILOT_SOURCES:
        parts = sorted((FONTES_DIR / fonte).glob("part-*.parquet"))
        for part in r.sample(parts, min(2, len(parts))):
            names = pq.read_schema(part).names
            rows = pq.read_table(part, columns=[c for c in FIELDS if c in names]).to_pylist()
            for row in r.sample(rows, min(3000, len(rows))):
                words = len((row["prompt"] or "").split())
                # T5 é só template (já existe nos 6 idiomas): não se traduz.
                if row["nl"] == "en" and row["task"] != "T5" and row["lang"] in pool and 5 <= words <= 250:
                    pool[row["lang"]].append({**row, "fonte": fonte})
    for rows in pool.values():
        r.shuffle(rows)
    return pool


def pilot(per_nl: int, seed: int = 0) -> None:
    pool = pilot_pool(seed)
    print({lang: len(rows) for lang, rows in pool.items()})
    # Cada idioma recebe exemplos diferentes, espalhados pelas linguagens e fontes.
    picked: dict[str, list[dict]] = {nl: [] for nl in TARGETS}
    cursor = {lang: 0 for lang in PILOT_LANGS}
    for nl in TARGETS:
        while len(picked[nl]) < per_nl:
            for lang in PILOT_LANGS:
                if len(picked[nl]) >= per_nl or cursor[lang] >= len(pool[lang]):
                    continue
                picked[nl].append(pool[lang][cursor[lang]])
                cursor[lang] += 1
            if all(cursor[lang] >= len(pool[lang]) for lang in PILOT_LANGS):
                break
    rows, stats = [], {}
    for nl in TARGETS:
        start = time.time()
        translator = Translator(nl)
        items = [(ex["prompt"], code_words(ex.get("code"), ex.get("response"))) for ex in picked[nl]]
        results = translate_texts(translator, nl, items)
        elapsed = time.time() - start
        chars = sum(len(text) for text, _ in items)
        reasons: dict[str, int] = {}
        for ex, (translated, reason) in zip(picked[nl], results):
            reasons[reason.split(" (")[0]] = reasons.get(reason.split(" (")[0], 0) + 1
            rows.append({**ex, "id": f"tr-{nl}-{ex['id']}", "nl": nl, "original": ex["prompt"],
                         "prompt": translated if translated else f"[REJEITADA: {reason}]", "traducao": reason})
        stats[nl] = {"exemplos": len(items), "segundos": round(elapsed, 1), "caracteres_por_s": round(chars / elapsed),
                     **reasons}
        print(nl, stats[nl], flush=True)
        del translator
    TRANSLATION_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), TRANSLATION_DIR / "piloto.parquet")
    (TRANSLATION_DIR / "piloto.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    REVIEW_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_SAMPLE.write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str))
    print(f"{len(rows)} exemplos em {REVIEW_SAMPLE}")


# --- rodada completa (CTranslate2 na CPU) ------------------------------------------------------

CT2_DIR = TRANSLATION_DIR / "ct2"
OUT_FONTE = "traducao-v2"
# Fase A (2026-11-13): 1 tradução por exemplo (com reposição: o original em inglês fica), num idioma
# sorteado pelo id. Java só em parte, porque falta pouco para o alvo.
PLAN = {"c": 1.0, "typescript": 1.0, "rust": 1.0, "javascript": 1.0, "bash": 1.0, "java": 0.6}
SKIP_FONTES = ("t6", "jy-", "traducao")
CHUNK = 4000


class CT2Translator:
    """Mesmo modelo do piloto, em int8 na CPU: 4 lotes em paralelo × 3 threads, busca gulosa."""

    def __init__(self, nl: str, inter: int = 4, intra: int = 3):
        import ctranslate2
        from transformers import MarianTokenizer
        name, self.prefix = MODELS[nl]
        path = CT2_DIR / nl
        if not (path / "model.bin").exists():
            from ctranslate2.converters import TransformersConverter
            TransformersConverter(name).convert(str(path), quantization="int8", force=True)
        self.tokenizer = MarianTokenizer.from_pretrained(name)
        self.model = ctranslate2.Translator(str(path), device="cpu", compute_type="int8",
                                            inter_threads=inter, intra_threads=intra)

    def __call__(self, sentences: list[str], beams: int = 1) -> list[str]:
        order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
        tokens = [self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(self.prefix + sentences[i], truncation=True, max_length=256)) for i in order]
        results = self.model.translate_batch(tokens, beam_size=beams, max_batch_size=64, max_decoding_length=320)
        out = [""] * len(sentences)
        for i, r in zip(order, results):
            out[i] = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(r.hypotheses[0]),
                                           skip_special_tokens=True)
        return out


def assignment(example_id: str) -> tuple[float, str]:
    """(sorteio em [0, 1), idioma de destino), estáveis pelo id."""
    h = int(hashlib.sha1(example_id.encode()).hexdigest(), 16)
    return (h % 10_000) / 10_000, TARGETS[(h // 10_000) % len(TARGETS)]


def selected(table: pa.Table, qc: pa.Table, nl: str) -> pa.Table:
    import pyarrow.compute as pc
    ok = pc.filter(qc.column("id"), pc.equal(pc.list_value_length(qc.column("qc_falhas")), 0))
    mask = pc.and_(pc.is_in(table.column("id"), ok), pc.equal(table.column("nl"), "en"))
    mask = pc.and_(mask, pc.not_equal(table.column("task"), "T5"))
    mask = pc.and_(mask, pc.is_in(table.column("lang"), pa.array(list(PLAN))))
    if "kind" in table.column_names:
        mask = pc.and_(mask, pc.equal(pc.fill_null(table.column("kind"), "code"), "code"))
    table = table.filter(mask)
    keep = []
    for i, (eid, lang) in enumerate(zip(table.column("id").to_pylist(), table.column("lang").to_pylist())):
        u, target = assignment(eid)
        if target == nl and u < PLAN[lang]:
            keep.append(i)
    return table.take(pa.array(keep, pa.int64()))


def translate_part(translator: CT2Translator, nl: str, fonte: str, part: Path, stats: Counter) -> None:
    from dataprep.sft_convert import estimate_tokens
    out = FONTES_DIR / OUT_FONTE / f"part-{fonte}-{part.stem.removeprefix('part-')}-{nl}.parquet"
    qc_path = FONTES_DIR.parent / "qc" / fonte / part.name
    if out.exists() or not qc_path.exists():
        return
    table = selected(pq.read_table(part), pq.read_table(qc_path, columns=["id", "qc_falhas"]), nl)
    if not len(table):
        return
    rows = table.to_pylist()
    kept = []
    for start in range(0, len(rows), CHUNK):
        chunk = rows[start:start + CHUNK]
        items = [(r["prompt"], code_words(prompt_code(r["prompt"]), r.get("response"), r.get("code"))) for r in chunk]
        for row, (text, reason) in zip(chunk, translate_texts(translator, nl, items)):
            stats[f"{nl}:{reason.split(' ')[0]}"] += 1
            if text is None:
                continue
            row["n_tokens"] = (row.get("n_tokens") or 0) + estimate_tokens(text, None) - estimate_tokens(row["prompt"], None)
            kept.append({**row, "id": f"{row['id']}-{nl}", "nl": nl, "prompt": text, "prompt_en": row["prompt"],
                         "fonte_original": fonte})
    if kept:
        out.parent.mkdir(parents=True, exist_ok=True)
        schema = table.schema.append(pa.field("prompt_en", pa.string())).append(pa.field("fonte_original", pa.string()))
        tmp = out.with_suffix(".tmp")
        pq.write_table(pa.Table.from_pylist(kept, schema=schema), tmp, compression="zstd")
        tmp.rename(out)


def translate_all(targets: list[str]) -> None:
    out_dir = FONTES_DIR / OUT_FONTE
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "execucao.json"
    fontes = sorted(p.name for p in FONTES_DIR.iterdir() if p.is_dir() and not p.name.startswith(SKIP_FONTES))
    for nl in targets:
        translator, start = CT2Translator(nl), time.time()
        stats: Counter = Counter(json.loads(stats_path.read_text())) if stats_path.exists() else Counter()
        for fonte in fontes:
            for part in sorted((FONTES_DIR / fonte).glob("part-*.parquet")):
                translate_part(translator, nl, fonte, part, stats)
                stats_path.write_text(json.dumps(dict(sorted(stats.items())), indent=1))
            print(f"  {nl} {fonte}: ok={stats[f'{nl}:ok']:,} | {time.time() - start:.0f}s", flush=True)
        print(f"{nl}: pronto em {time.time() - start:.0f}s", flush=True)
    save_translation_meta()


def save_translation_meta() -> None:
    out_dir = FONTES_DIR / OUT_FONTE
    cells: Counter = Counter()
    tokens: Counter = Counter()
    for part in sorted(out_dir.glob("part-*.parquet")):
        for r in pq.read_table(part, columns=["lang", "nl", "n_tokens"]).to_pylist():
            cells[f"{r['lang']}|{r['nl']}"] += 1
            tokens[f"{r['lang']}|{r['nl']}"] += r["n_tokens"] or 0
    meta = {"fonte": OUT_FONTE, "modelo": "Opus-MT (CTranslate2 int8, busca gulosa)", "plano": PLAN,
            "exemplos": sum(cells.values()), "tokens_estimados": sum(tokens.values()),
            "celulas": {k: {"exemplos": cells[k], "tokens": tokens[k]} for k in sorted(cells)},
            "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    pilot_cmd = sub.add_parser("piloto")
    pilot_cmd.add_argument("--por-idioma", type=int, default=50)
    run_cmd = sub.add_parser("traduzir", help="fase A: linguagens escassas, 1 tradução por exemplo")
    run_cmd.add_argument("idiomas", nargs="*", default=list(TARGETS))
    args = parser.parse_args()
    if args.cmd == "piloto":
        pilot(args.por_idioma)
    else:
        translate_all(args.idiomas)


if __name__ == "__main__":
    main()
