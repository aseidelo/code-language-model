"""Monta o corpus de pré-treino para publicar no Hugging Face.

Gera em ``data/codelm-pretrain/`` uma pasta pronta para ``hf upload``:

* ``code/<lang>/{train,valid}-NNN.parquet``: código filtrado do the-stack-dedup,
  com repositório, caminho e licenças de cada arquivo;
* ``natural/<idioma>/...``: texto natural do FineWeb-Edu e do FineWeb2-HQ;
* ``README.md``: card com licenças, atribuições, uma configuração por
  linguagem e idioma, e os termos de uso do The Stack para o acesso condicionado
  (a BigCode exige que quem redistribui peça o aceite desses termos).

Antes de gravar, o código passa por ``redact``: segredos (chaves de API, tokens,
chaves privadas) viram ``<SECRET>`` e e-mails viram ``<EMAIL>``.

Uso:
    uv run python -m dataprep.publish                  # corpus v2 -> data/codelm-pretrain-v2/
    uv run python -m dataprep.publish --subir          # envia o v2 (retomável)
    uv run python -m dataprep.publish --versao v1      # corpus v1 (histórico)
Depois, para subir (privado e com acesso condicionado):
    hf repos create <usuário>/codelm-pretrain --type dataset --private
    hf repos settings <usuário>/codelm-pretrain --type dataset --gated auto
    hf upload <usuário>/codelm-pretrain data/codelm-pretrain . --type dataset
"""

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from codelm.languages import DATA_DIR, LANGUAGES, NATURAL_LANGUAGES
from dataprep.pretrain import CODE_DIR, NATURAL_DATASETS, natural_dir

OUT_DIR = DATA_DIR / "codelm-pretrain"

# --- redação de segredos e e-mails -------------------------------------------

SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS
    re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"),  # GitHub
    re.compile(r"\bsk-(ant-|proj-)?[A-Za-z0-9_-]{20,}"),  # OpenAI, Anthropic
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),  # Google
    re.compile(r"\b[rs]k_live_[0-9A-Za-z]{24,}"),  # Stripe
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # JWT
]
PRIVATE_KEY = re.compile(r"(-----BEGIN [A-Z ]*PRIVATE KEY-----).*?(-----END [A-Z ]*PRIVATE KEY-----)", re.DOTALL)
# password = "..." e afins: troca só o valor entre aspas.
SECRET_ASSIGNMENT = re.compile(
    r"""((?:password|passwd|secret|api_?key|access_?token|auth_?token|client_?secret)["']?\s*[:=]\s*["'])"""
    r"""(?![$<{%])([^"'\s]{8,})(["'])""", re.IGNORECASE)  # ${VAR}, <placeholder> ficam
# E-mail só com domínio de topo conhecido: ``x@self.weight`` em Python não é e-mail.
TLDS = ("com|net|org|edu|gov|mil|io|info|biz|co|me|ai|dev|app|us|uk|de|fr|br|it|es|nl|ru|cn|jp|in|au|ca|ch|se|"
        "no|fi|dk|pl|cz|pt|be|at|ir|kr|tw|hk|sg|nz|za|mx|ar|cl|il|gr|hu|ro|tr|ua|eu")
EMAIL = re.compile(rf"\b[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+(?:{TLDS})\b")
EMAIL_KEEP = re.compile(r"^(git|noreply|no-reply|example|user|test|foo|name|email|someone)@|@(example|test)\.", re.I)


def redact(text: str, counts: Counter | None = None) -> str:
    counts = counts if counts is not None else Counter()

    def secret(match: re.Match) -> str:
        counts["segredos"] += 1
        return "<SECRET>"

    text = PRIVATE_KEY.sub(lambda m: (counts.update(["segredos"]), f"{m[1]}\n<SECRET>\n{m[2]}")[1], text)
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(secret, text)
    text = SECRET_ASSIGNMENT.sub(lambda m: (counts.update(["segredos"]), f"{m[1]}<SECRET>{m[3]}")[1], text)

    def email(match: re.Match) -> str:
        if EMAIL_KEEP.search(match[0]):
            return match[0]
        counts["e-mails"] += 1
        return "<EMAIL>"

    return EMAIL.sub(email, text)


def redact_columns(table: pa.Table, columns: list[str], counts: Counter) -> pa.Table:
    for name in columns:
        values = [None if v is None else redact(v, counts) for v in table[name].to_pylist()]
        table = table.set_column(table.schema.get_field_index(name), name, pa.array(values, pa.string()))
    return table


# --- cards ------------------------------------------------------------------

STACK_TERMS = """The Stack dataset is a collection of source code in over 300 programming languages. We ask that you read and acknowledge the following points before using the dataset:
1. The Stack is a collection of source code from repositories with various licenses. Any use of all or part of the code gathered in The Stack must abide by the terms of the original licenses, including attribution clauses when relevant. We facilitate this by providing provenance information for each data point.
2. The Stack is regularly updated to enact validated data removal requests. By clicking on "Access repository", you agree to update your own version of The Stack to the most recent usable version specified by the maintainers in [the following thread](https://huggingface.co/datasets/bigcode/the-stack/discussions/7). If you have questions about dataset versions and allowed uses, please also ask them in the dataset's [community discussions](https://huggingface.co/datasets/bigcode/the-stack/discussions/new). We will also notify users via email when the latest usable version changes.
3. To host, share, or otherwise provide access to The Stack dataset, you must include [these Terms of Use](https://huggingface.co/datasets/bigcode/the-stack#terms-of-use-for-the-stack) and require users to agree to it.

This dataset contains code derived from The Stack (bigcode/the-stack-dedup); the terms above apply to it."""


def card_header(configs: list[tuple[str, dict[str, str]]], tags: list[str]) -> str:
    """Cabeçalho YAML do card: licença, acesso condicionado e configurações."""
    lines = ["---", "license: other", "license_name: mixed", "language:", *(f"- {nl}" for nl in NATURAL_LANGUAGES),
             "tags:", *(f"- {tag}" for tag in tags), "extra_gated_prompt: |-",
             *(f"  {line}" for line in STACK_TERMS.splitlines()),
             "extra_gated_fields:", "  I have read the Terms of Use of The Stack and agree with them: checkbox",
             "configs:"]
    for name, files in configs:
        lines += [f"- config_name: {name}", "  data_files:"]
        for split, pattern in files.items():
            lines += [f"  - split: {split}", f"    path: {pattern}"]
    return "\n".join(lines + ["---", ""])


REDACTION_NOTE = """## Redaction

Before publication, likely secrets (API keys, access tokens, private keys, `password = "..."` style
assignments) were replaced by `<SECRET>` and e-mail addresses by `<EMAIL>`. The redaction is
pattern based and not exhaustive. The code used for training was not redacted.

## Removal requests

If your code is in this dataset and you want it removed, open a discussion in this repository
(and see the opt-out process of [The Stack](https://huggingface.co/spaces/bigcode/in-the-stack)).
"""


def card(stats: dict[str, Counter], redacted: Counter) -> str:
    configs = [(lang, {"train": f"code/{lang}/train-*.parquet", "validation": f"code/{lang}/valid-*.parquet"})
               for lang in LANGUAGES if lang in stats]
    configs += [(f"natural-{nl}", {"train": f"natural/{nl}/train-*.parquet", "validation": f"natural/{nl}/valid-*.parquet"})
                for nl in NATURAL_LANGUAGES if f"natural-{nl}" in stats]
    rows = "\n".join(f"| `{name}` | {stats[name]['train']:,} | {stats[name]['valid']:,} |" for name, _ in configs)
    return card_header(configs, ["code", "pretraining"]) + f"""# codelm pretraining corpus

Text of the pretraining corpus of [codelm](https://github.com/aseidelo/code-language-model), a small
language model trained from scratch for programming, made to run quantized on CPU.

- **Code** (10 languages): files from [bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup),
  filtered by quality, exactly deduplicated, standardized (line breaks, tabs, indentation) and split
  into train/validation **by repository**. Python files compile with Python 3. Each row keeps
  `repo`, `path` and `licenses` (permissive licenses only, as in the-stack-dedup).
- **Natural text** (en, es, pt, fr, de, it): English from
  [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and the other
  languages from [epfml/FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ), both under
  ODC-By 1.0 and subject to the [Common Crawl terms of use](https://commoncrawl.org/terms-of-use).

| config | train | validation |
|---|---|---|
{rows}

## License

Each code file keeps its original license (column `licenses`); use must follow it, including
attribution. The natural text is ODC-By 1.0. Access requires accepting the Terms of Use of The Stack.

{REDACTION_NOTE}
Redaction counts: {redacted['segredos']:,} secrets, {redacted['e-mails']:,} e-mails.
"""


# --- montagem ----------------------------------------------------------------

def stage(out: Path) -> tuple[dict[str, Counter], Counter]:
    stats: dict[str, Counter] = {}
    redacted = Counter()
    for lang in LANGUAGES:
        for path in sorted((CODE_DIR / lang).glob("*.parquet")):
            table = pq.read_table(path, columns=["content", "repo", "path", "licenses"])
            table = redact_columns(table, ["content"], redacted)
            table = table.append_column("lang", pa.array([lang] * len(table), pa.string()))
            (out / "code" / lang).mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out / "code" / lang / path.name, compression="zstd")
            stats.setdefault(lang, Counter())["valid" if path.name.startswith("valid") else "train"] += len(table)
        print(f"  {lang}: {dict(stats.get(lang, {}))}", flush=True)
    for nl in NATURAL_LANGUAGES:
        for path in sorted(natural_dir(nl).glob("*.parquet")):
            table = pq.read_table(path, columns=["content"])
            source = "HuggingFaceFW/fineweb-edu" if NATURAL_DATASETS[nl] == "fineweb-edu" else "epfml/FineWeb2-HQ"
            table = table.append_column("source", pa.array([source] * len(table), pa.string()))
            (out / "natural" / nl).mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out / "natural" / nl / path.name, compression="zstd")
            stats.setdefault(f"natural-{nl}", Counter())["valid" if path.name.startswith("valid") else "train"] += len(table)
    return stats, redacted


# --- corpus v2 ----------------------------------------------------------------

OUT_DIR_V2 = DATA_DIR / "codelm-pretrain-v2"
HF_REPO_V2 = "seidel/codelm-pretrain-v2"
STACK_V2_TERMS = """The code of the languages taken from Stack-Edu (Python, Java, JavaScript, TypeScript, C, Rust, Bash) comes from The Stack v2 (bigcode/the-stack-v2), through HuggingFaceTB/stack-edu, with the contents downloaded from the Software Heritage archive. The Terms of Use of The Stack v2 (https://huggingface.co/datasets/bigcode/the-stack-v2) apply to it, including the obligation to honor removal requests. Part of these files come from repositories without a license (column `license_type`)."""


def stage_file_v2(task: tuple[str, str, str]) -> tuple[str, str, Counter]:
    """Copia um parquet do corpus v2 para a pasta de publicação; o código passa pelo ``redact``."""
    kind, src, out = task
    stats, redacted = Counter(), Counter()
    names = pq.read_schema(src).names
    if kind.startswith("natural-"):
        nl = kind.removeprefix("natural-")
        table = pq.read_table(src, columns=["content"])
        source = "HuggingFaceFW/fineweb-edu" if NATURAL_DATASETS[nl] == "fineweb-edu" else "epfml/FineWeb2-HQ"
        table = table.append_column("source", pa.array([source] * len(table), pa.string()))
    else:
        cols = [c for c in ("content", "repo", "path", "licenses", "license_type", "score") if c in names]
        table = pq.read_table(src, columns=cols)
        table = redact_columns(table, ["content"], redacted)
        n = len(table)
        if "license_type" not in names:  # the-stack-dedup: só licenças permissivas
            table = table.append_column("license_type", pa.array(["permissive"] * n, pa.string()))
        if "score" not in names:
            table = table.append_column("score", pa.array([None] * n, pa.float64()))
        source = "HuggingFaceTB/stack-edu" if "score" in names else "bigcode/the-stack-dedup"
        table = table.append_column("lang", pa.array([kind] * n, pa.string()))
        table = table.append_column("source", pa.array([source] * n, pa.string()))
        table = table.select(["content", "lang", "source", "repo", "path", "licenses", "license_type", "score"])
        for lt, size in zip(table["license_type"].to_pylist(), pc.utf8_length(table["content"]).to_pylist()):
            stats[f"chars_{lt}"] += size or 0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd")
    split = "valid" if Path(src).name.startswith("valid") else "train"
    stats[split] += len(table)
    stats[f"chars_{split}"] += sum(x or 0 for x in pc.utf8_length(table["content"]).to_pylist())
    return kind, split, stats + redacted


def card_v2(stats: dict[str, Counter], redacted: Counter) -> str:
    from dataprep.pretrain import MIX

    kinds = [k for k in LANGUAGES if k in stats] + [f"natural-{nl}" for nl in NATURAL_LANGUAGES if f"natural-{nl}" in stats]
    configs = [(k, {"train": f"{'natural/' + k[8:] if k.startswith('natural-') else 'code/' + k}/train-*.parquet",
                    "validation": f"{'natural/' + k[8:] if k.startswith('natural-') else 'code/' + k}/valid-*.parquet"})
               for k in kinds]

    def row(k: str) -> str:
        st = stats[k]
        mix = MIX.get(k.replace("natural-", "natural:"), 0)
        code = st["chars_permissive"] + st["chars_no_license"]
        nolic = f"{100 * st['chars_no_license'] / code:.0f}%" if code else "-"
        source = ("Stack-Edu" if st["chars_no_license"] or k in ("python", "java", "javascript", "typescript", "c", "rust", "bash")
                  else "FineWeb-Edu" if k == "natural-en" else "FineWeb2-HQ" if k.startswith("natural-") else "the-stack-dedup")
        return (f"| `{k}` | {source} | {st['train']:,} | {st['valid']:,} | {(st['chars_train'] + st['chars_valid']) / 1e9:.2f} | "
                f"{100 * mix:.1f}% | {nolic} |")

    rows = "\n".join(row(k) for k in kinds)
    header = card_header(configs, ["code", "pretraining"])
    header = header.replace("license_name: mixed", "license_name: mixed").replace(
        "extra_gated_fields:", "  " + STACK_V2_TERMS.replace("\n", " ") + "\nextra_gated_fields:", 1)
    return header + f"""# codelm pretraining corpus v2

Text of the pretraining corpus v2 of [codelm](https://github.com/aseidelo/code-language-model), used to
train `base-v2` (Gemma 3 style, 113M parameters, 8k context, 10B tokens). 11 programming languages and
6 natural languages.

- **Code, 7 languages from [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)**
  (Python, Java, JavaScript, TypeScript, C, Rust, Bash): files of The Stack v2 filtered by an
  educational-value classifier (score ≥ 3, ≥ 2 for Java), contents downloaded from the Software
  Heritage archive. Bash keeps only `.sh`/`.bash` files or files with a bash/sh shebang.
- **Code, 4 languages from [bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup)**
  (HTML, CSS, JSON, YAML), with the heuristic quality filters of the v1 corpus.
- **Natural text**: English from [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
  es/pt/fr/de/it from [epfml/FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ), both ODC-By 1.0 and
  subject to the [Common Crawl terms of use](https://commoncrawl.org/terms-of-use).

All code is exactly deduplicated, standardized (line breaks, tabs, indentation) and split into
train/validation **by repository**. Each row keeps `repo`, `path`, `licenses` and `license_type`
(`permissive` or `no_license`); Stack-Edu rows also keep the classifier `score`.

| config | source | train documents | validation documents | characters (B) | share in base-v2 training | no license |
|---|---|---|---|---|---|---|
{rows}

"share in base-v2 training" is the token share of each config in the 10B tokens used to train
`base-v2`; configs with more text than their share were subsampled.

## License

There is no single license. Each code file keeps the license of its repository (column `licenses`);
use must follow it, including attribution. **Files with `license_type = no_license` come from
repositories without a license**: they are redistributed, as in The Stack v2, under its Terms of Use,
and removal requests are honored. The natural text is ODC-By 1.0. Access requires accepting the
Terms of Use of The Stack (v1 and v2).

{REDACTION_NOTE}
Redaction counts: {redacted['segredos']:,} secrets, {redacted['e-mails']:,} e-mails.
"""


def stage_v2(workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    from dataprep.pretrain import code_dir

    shutil.rmtree(OUT_DIR_V2, ignore_errors=True)
    tasks = []
    for lang in LANGUAGES:
        for path in sorted(code_dir(lang).glob("*.parquet")):
            tasks.append((lang, str(path), str(OUT_DIR_V2 / "code" / lang / path.name)))
    for nl in NATURAL_LANGUAGES:
        for path in sorted(natural_dir(nl).glob("*.parquet")):
            tasks.append((f"natural-{nl}", str(path), str(OUT_DIR_V2 / "natural" / nl / path.name)))
    stats: dict[str, Counter] = {}
    redacted = Counter()
    with ProcessPoolExecutor(workers) as pool:
        for i, (kind, split, st) in enumerate(pool.map(stage_file_v2, tasks, chunksize=1), 1):
            redacted.update({k: st.pop(k) for k in ("segredos", "e-mails") if k in st})
            stats.setdefault(kind, Counter()).update(st)
            if i % 100 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} arquivos", flush=True)
    (OUT_DIR_V2 / "stats.json").write_text(json.dumps({k: dict(v) for k, v in stats.items()} | {"redacao": dict(redacted)}))
    (OUT_DIR_V2 / "README.md").write_text(card_v2(stats, redacted))
    print(f"redação: {redacted['segredos']:,} segredos, {redacted['e-mails']:,} e-mails; pronto: {OUT_DIR_V2}")


def upload_v2() -> None:
    """Envia ``OUT_DIR_V2`` (repositório privado e condicionado), uma pasta por commit; retomável."""
    from huggingface_hub import HfApi

    api = HfApi()
    done = set(api.list_repo_files(HF_REPO_V2, repo_type="dataset"))
    for d in sorted(p for p in OUT_DIR_V2.glob("*/*") if p.is_dir()):
        rel = d.relative_to(OUT_DIR_V2).as_posix()
        missing = [f.relative_to(OUT_DIR_V2).as_posix() for f in sorted(d.glob("*.parquet"))
                   if f.relative_to(OUT_DIR_V2).as_posix() not in done]
        for i in range(0, len(missing), 300):
            chunk = missing[i:i + 300]
            api.upload_folder(repo_id=HF_REPO_V2, repo_type="dataset", folder_path=str(OUT_DIR_V2), allow_patterns=chunk,
                              commit_message=f"{rel} ({i + len(chunk)}/{len(missing)})")
            print(f"  {rel}: {i + len(chunk)}/{len(missing)}", flush=True)
    api.upload_folder(repo_id=HF_REPO_V2, repo_type="dataset", folder_path=str(OUT_DIR_V2),
                      allow_patterns=["README.md", "stats.json"], commit_message="README e estatísticas")
    print("upload concluído", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--versao", choices=["v1", "v2"], default="v2")
    parser.add_argument("--subir", action="store_true", help="só envia a pasta já montada (v2)")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.versao == "v2":
        upload_v2() if args.subir else stage_v2(args.workers)
        return
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    stats, redacted = stage(OUT_DIR)
    (OUT_DIR / "README.md").write_text(card(stats, redacted))
    print(f"redação: {redacted['segredos']:,} segredos, {redacted['e-mails']:,} e-mails; pronto para subir: {OUT_DIR}")


if __name__ == "__main__":
    main()
