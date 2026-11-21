"""Tradução local de pedidos (en -> pt/es/fr/de/it) com Opus-MT na XPU.

Desenho em docs/memory/sft-v2.md ("Tradução local"). Resumo:

* Só a prosa é traduzida. Blocos de código ficam intactos. Trechos entre crases, termos do glossário
  (``assets/sft/glossario.txt``) e identificadores viram marcadores ``X0X`` e voltam depois. Os
  identificadores são os nomes com ``_``, dígitos, camelCase ou ponto, e as palavras que também
  aparecem no código do exemplo (``nums``, ``target``).
* A prosa é traduzida linha a linha (linhas longas por sentença), mantendo os prefixos de markdown.
* Rejeita quando um marcador se perde, o detector acusa outro idioma ou o tamanho foge do esperado.
* Pedido que é um template em inglês (T2: "Finish this function.") não passa pelo Opus-MT: vira o
  template de mesmo índice no idioma de destino, de ``assets/sft/instructions.json``.

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
INSTRUCTIONS = ASSETS_DIR / "sft" / "instructions.json"
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


@lru_cache(maxsize=1)
def instruction_templates() -> dict[str, dict[str, str]]:
    """Template em inglês -> {idioma: template de mesmo índice}, de todas as tarefas."""
    out: dict[str, dict[str, str]] = {}
    for by_nl in json.loads(INSTRUCTIONS.read_text()).values():
        for i, en in enumerate(by_nl["en"]):
            out[en] = {nl: by_nl[nl][i] for nl in TARGETS}
    return out


def translated_row(row: dict, text: str, nl: str, fonte: str) -> dict:
    from dataprep.sft_convert import estimate_tokens
    n_tokens = (row.get("n_tokens") or 0) + estimate_tokens(text, None) - estimate_tokens(row["prompt"], None)
    return {**row, "id": f"{row['id']}-{nl}", "nl": nl, "prompt": text, "prompt_en": row["prompt"],
            "fonte_original": fonte, "n_tokens": n_tokens}


def part_paths(nl: str, fonte: str, part: Path, out_fonte: str = OUT_FONTE) -> tuple[Path, Path]:
    """(saída traduzida, QC da parte de origem)."""
    out = FONTES_DIR / out_fonte / f"part-{fonte}-{part.stem.removeprefix('part-')}-{nl}.parquet"
    return out, FONTES_DIR.parent / "qc" / fonte / part.name


def write_part(out: Path, schema: pa.Schema, rows: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = schema.append(pa.field("prompt_en", pa.string())).append(pa.field("fonte_original", pa.string()))
    tmp = out.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression="zstd")
    tmp.rename(out)


def translate_part(translator: CT2Translator, nl: str, fonte: str, part: Path, stats: Counter,
                   select=None, out_fonte: str = OUT_FONTE) -> None:
    out, qc_path = part_paths(nl, fonte, part, out_fonte)
    if out.exists() or not qc_path.exists():
        return
    table = (select or selected)(pq.read_table(part), pq.read_table(qc_path, columns=["id", "qc_falhas"]), nl)
    if not len(table):
        return
    templates = instruction_templates()
    kept, to_translate = [], []
    for row in table.to_pylist():
        if row["prompt"] in templates:
            stats[f"{nl}:template"] += 1
            kept.append(translated_row(row, templates[row["prompt"]][nl], nl, fonte))
        else:
            to_translate.append(row)
    for start in range(0, len(to_translate), CHUNK):
        chunk = to_translate[start:start + CHUNK]
        items = [(r["prompt"], code_words(prompt_code(r["prompt"]), r.get("response"), r.get("code"))) for r in chunk]
        for row, (text, reason) in zip(chunk, translate_texts(translator, nl, items)):
            stats[f"{nl}:{reason.split(' ')[0]}"] += 1
            if text is not None:
                kept.append(translated_row(row, text, nl, fonte))
    if kept:
        write_part(out, table.schema, kept)


def apply_templates(targets: list[str]) -> None:
    """Refaz, nas partes já traduzidas, os pedidos que são template: troca a saída do Opus-MT pelo template
    escrito à mão e recupera os que o Opus-MT tinha perdido (rejeitados). As outras linhas ficam como estão."""
    templates = instruction_templates()
    fontes = sorted(p.name for p in FONTES_DIR.iterdir() if p.is_dir() and not p.name.startswith(SKIP_FONTES))
    stats_path = FONTES_DIR / OUT_FONTE / "templates.json"
    stats: Counter = Counter()
    for nl in targets:
        for fonte in fontes:
            for part in sorted((FONTES_DIR / fonte).glob("part-*.parquet")):
                out, qc_path = part_paths(nl, fonte, part)
                if not out.exists():  # parte ainda não traduzida: o ``traduzir`` já usa os templates
                    continue
                old = pq.read_table(out).to_pylist()
                kept = [r for r in old if r["prompt_en"] not in templates]
                table = selected(pq.read_table(part), pq.read_table(qc_path, columns=["id", "qc_falhas"]), nl)
                new = [translated_row(r, templates[r["prompt"]][nl], nl, fonte)
                       for r in table.to_pylist() if r["prompt"] in templates]
                stats[f"{nl}:antes"] += len(old) - len(kept)
                stats[f"{nl}:depois"] += len(new)
                write_part(out, table.schema, kept + new)
            print(f"  {nl} {fonte}: template {stats[f'{nl}:antes']:,} -> {stats[f'{nl}:depois']:,}", flush=True)
    previous = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    stats_path.write_text(json.dumps(dict(sorted({**previous, **stats}.items())), indent=1))
    save_translation_meta()


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


# --- fase B: Python e HTML ---------------------------------------------------------------------
# Decidido em 2026-11-21 (docs/memory/sft-v2.md, "Fase B"). Orçamento por idioma, em tokens: HTML do WebSight
# por Opus-MT; Python metade T2 (template, sem Opus-MT) e metade T1 (Opus-MT, prioridade para muitos
# tokens por caractere traduzido). T5 fica de fora: a resposta (a docstring) seguiria em inglês.
OUT_FONTE_B = "traducao-v2b"
PLAN_B_FILE = FONTES_DIR / OUT_FONTE_B / "plano.json"
BUDGET_B = {"html-t1": 36e6, "python-t2": 112e6, "python-t1": 112e6}
T1_REJECTED = 0.05  # fração rejeitada na fase A; o orçamento do T1 compensa
MIN_WORDS = 5


def prose_chars(prompt: str) -> int:
    """Caracteres que o Opus-MT traduz: o pedido sem os blocos de código."""
    return len(FENCE.sub("", prompt or "").strip())


def base_mask(table: pa.Table, qc: pa.Table):
    import pyarrow.compute as pc
    ok = pc.filter(qc.column("id"), pc.equal(pc.list_value_length(qc.column("qc_falhas")), 0))
    mask = pc.and_(pc.is_in(table.column("id"), ok), pc.equal(table.column("nl"), "en"))
    mask = pc.and_(mask, pc.is_in(table.column("task"), pa.array(["T1", "T2"])))
    mask = pc.and_(mask, pc.is_in(table.column("lang"), pa.array(["python", "html"])))
    if "kind" in table.column_names:
        mask = pc.and_(mask, pc.equal(pc.fill_null(table.column("kind"), "code"), "code"))
    return mask


def cell_b(row: dict) -> str | None:
    """Célula da fase B do exemplo (já filtrado por ``base_mask``), ou None."""
    templates = instruction_templates()
    if row["lang"] == "html":
        return "html-t1" if row["task"] == "T1" else None
    if row["task"] == "T2":
        return "python-t2" if row["prompt"] in templates else None
    if row["prompt"] in templates or len((row["prompt"] or "").split()) < MIN_WORDS:
        return None
    return "python-t1"


def t1_ratio(row: dict) -> float:
    return (row["n_tokens"] or 0) / max(prose_chars(row["prompt"]), 1)


def fontes_b() -> list[str]:
    return sorted(p.name for p in FONTES_DIR.iterdir() if p.is_dir() and not p.name.startswith(SKIP_FONTES))


def plan_b() -> dict:
    """Uma passada nas fontes: frações do HTML e do T2 e o corte de tokens/caractere do T1 que fecham o
    orçamento (cada exemplo vai a um idioma só, sorteado pelo id, então o total é 5 × o orçamento)."""
    tokens: Counter = Counter()
    t1: list[tuple[float, int, int]] = []
    for fonte in fontes_b():
        for part in sorted((FONTES_DIR / fonte).glob("part-*.parquet")):
            qc_path = FONTES_DIR.parent / "qc" / fonte / part.name
            names = pq.read_schema(part).names
            if not qc_path.exists() or "lang" not in names:
                continue
            cols = [c for c in ("id", "task", "lang", "nl", "kind", "prompt", "n_tokens") if c in names]
            table = pq.read_table(part, columns=cols)
            table = table.filter(base_mask(table, pq.read_table(qc_path, columns=["id", "qc_falhas"])))
            for row in table.to_pylist():
                cell = cell_b(row)
                if cell is None:
                    continue
                tokens[cell] += row["n_tokens"] or 0
                if cell == "python-t1":
                    t1.append((t1_ratio(row), row["n_tokens"] or 0, prose_chars(row["prompt"])))
        print(f"  {fonte}: " + " | ".join(f"{k} {v / 1e6:,.0f}M" for k, v in sorted(tokens.items())), flush=True)
    n = len(TARGETS)
    plan = {"orcamento_por_idioma": BUDGET_B, "disponivel": dict(tokens),
            "fracao": {c: min(1.0, n * BUDGET_B[c] / tokens[c]) for c in ("html-t1", "python-t2") if tokens[c]}}
    goal, total, chars, cut = n * BUDGET_B["python-t1"] / (1 - T1_REJECTED), 0, 0, 0.0
    for ratio, n_tokens, n_chars in sorted(t1, reverse=True):
        if total >= goal:
            break
        total, chars, cut = total + n_tokens, chars + n_chars, ratio
    plan["corte_t1"] = cut
    plan["t1_selecionado"] = {"tokens": total, "caracteres": chars, "horas_cpu": round(chars / 9800 / 3600, 1)}
    PLAN_B_FILE.parent.mkdir(parents=True, exist_ok=True)
    PLAN_B_FILE.write_text(json.dumps(plan, indent=1))
    return plan


def selected_b(plan: dict):
    def select(table: pa.Table, qc: pa.Table, nl: str) -> pa.Table:
        names = table.column_names
        if "lang" not in names:
            return table.slice(0, 0)
        table = table.filter(base_mask(table, qc))
        keep = []
        for i, row in enumerate(table.select([c for c in ("id", "task", "lang", "prompt", "n_tokens") if c in names])
                                .to_pylist()):
            cell = cell_b(row)
            u, target = assignment(row["id"])
            if cell is None or target != nl:
                continue
            if cell == "python-t1" and t1_ratio(row) >= plan["corte_t1"] or u < plan["fracao"].get(cell, 0.0):
                keep.append(i)
        return table.take(pa.array(keep, pa.int64()))
    return select


def translate_all_b(targets: list[str]) -> None:
    if not PLAN_B_FILE.exists():
        raise SystemExit("rode antes: uv run python -m dataprep.sft_translate planejar-b")
    plan = json.loads(PLAN_B_FILE.read_text())
    select, out_dir = selected_b(plan), FONTES_DIR / OUT_FONTE_B
    stats_path = out_dir / "execucao.json"
    for nl in targets:
        translator, start = CT2Translator(nl), time.time()
        stats: Counter = Counter(json.loads(stats_path.read_text())) if stats_path.exists() else Counter()
        for fonte in fontes_b():
            for part in sorted((FONTES_DIR / fonte).glob("part-*.parquet")):
                translate_part(translator, nl, fonte, part, stats, select, OUT_FONTE_B)
                stats_path.write_text(json.dumps(dict(sorted(stats.items())), indent=1))
            print(f"  {nl} {fonte}: ok={stats[f'{nl}:ok']:,} template={stats[f'{nl}:template']:,} | "
                  f"{time.time() - start:.0f}s", flush=True)
        print(f"{nl}: pronto em {time.time() - start:.0f}s", flush=True)
    save_translation_meta(OUT_FONTE_B, {"orcamento_por_idioma": BUDGET_B, **{k: plan[k] for k in ("fracao", "corte_t1")}})


def save_translation_meta(out_fonte: str = OUT_FONTE, plan: dict | None = None) -> None:
    out_dir = FONTES_DIR / out_fonte
    cells: Counter = Counter()
    tokens: Counter = Counter()
    for part in sorted(out_dir.glob("part-*.parquet")):
        for r in pq.read_table(part, columns=["lang", "nl", "n_tokens"]).to_pylist():
            cells[f"{r['lang']}|{r['nl']}"] += 1
            tokens[f"{r['lang']}|{r['nl']}"] += r["n_tokens"] or 0
    meta = {"fonte": out_fonte, "modelo": "Opus-MT (CTranslate2 int8, busca gulosa)", "plano": plan or PLAN,
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
    tpl_cmd = sub.add_parser("templates", help="troca, nas partes já traduzidas, o Opus-MT dos templates pelo manual")
    tpl_cmd.add_argument("idiomas", nargs="*", default=list(TARGETS))
    sub.add_parser("planejar-b", help="fase B: frações e corte do T1 que fecham o orçamento")
    run_b = sub.add_parser("traduzir-b", help="fase B: Python (T2 por template, T1 por Opus-MT) e HTML")
    run_b.add_argument("idiomas", nargs="*", default=list(TARGETS))
    args = parser.parse_args()
    if args.cmd == "planejar-b":
        print(json.dumps(plan_b(), indent=1))
    elif args.cmd == "traduzir-b":
        translate_all_b(args.idiomas)
    elif args.cmd == "piloto":
        pilot(args.por_idioma)
    elif args.cmd == "templates":
        apply_templates(args.idiomas)
    else:
        translate_all(args.idiomas)


if __name__ == "__main__":
    main()
