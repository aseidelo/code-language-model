"""Card (README.md) do dataset SFT v2 no Hugging Face. Chamado por ``sft_final card``.

Lê ``data/codelm-sft-v2/stats.json`` (gravado por ``sft_final montar``) e escreve o README com as
tabelas de fontes, tarefas, linguagens e idiomas, o processo de geração e a licença.
"""

import json
from collections import defaultdict

from dataprep.publish import REDACTION_NOTE, STACK_TERMS
from dataprep.sft_final import NLS, OUT_DIR

LICENSE = "cc-by-nc-sa-4.0"

# fonte -> (origem, licença, como entrou)
SOURCES = {
    "corpus-v2": ("codelm pretraining corpus v2 (Stack-Edu, the-stack-dedup)", "per repository (Stack-Edu: permissive or none)",
                  "functions with their original docstring: T2 (docstring → code) and T5 (code → docstring)"),
    "banco-v1-corpus": ("codelm pretraining corpus v1 (the-stack-dedup)", "per repository (permissive)",
                        "T2/T5 from documented functions, original English docstrings"),
    "code-search-net": ("[code-search-net/code_search_net](https://huggingface.co/datasets/code-search-net/code_search_net)",
                        "per repository", "T2/T5 from documented functions"),
    "t6s-corpus-v2": ("generated from corpus-v2 functions", "per repository (Stack-Edu: permissive or none)",
                      "T6 syntax repair: tree-sitter mutations of C, TS, Rust, JS, Java and Bash functions"),
    "jy-stack": ("generated from JSON/YAML files of the-stack-dedup", "per repository (permissive)",
                 "T7/T6: convert, fix, format, extract and edit JSON/YAML, instructions from templates"),
    "open-code-instruct": ("[nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)", "CC-BY-4.0", "T1"),
    "kodcode": ("[KodCode/KodCode-V1](https://huggingface.co/datasets/KodCode/KodCode-V1)", "CC-BY-NC-4.0", "T1, solutions with tests"),
    "t6-kodcode": ("generated from KodCode", "CC-BY-NC-4.0", "T6 bug fixing, bugs verified by the tests"),
    "acecode-87k": ("[TIGER-Lab/AceCode-87K](https://huggingface.co/datasets/TIGER-Lab/AceCode-87K)", "MIT", "T1"),
    "t6-acecode-87k": ("generated from AceCode-87K", "MIT", "T6 bug fixing, bugs verified by the tests"),
    "acecode-v2": ("[TIGER-Lab/AceCode-V2-122K](https://huggingface.co/datasets/TIGER-Lab/AceCode-V2-122K)", "Apache-2.0", "T1"),
    "opc-stage1-largescale": ("[OpenCoder-LLM/opc-sft-stage1](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage1) (largescale_diverse_instruct)", "MIT", "T1"),
    "opc-stage1-infinity": ("[OpenCoder-LLM/opc-sft-stage1](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage1) (filtered_infinity_instruct)", "MIT", "T1"),
    "opc-stage1-realuser": ("[OpenCoder-LLM/opc-sft-stage1](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage1) (realuser_instruct)", "MIT", "T1"),
    "opc-stage2-educational": ("[OpenCoder-LLM/opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2) (educational_instruct)", "MIT", "T1, with tests"),
    "t6-opc-stage2-educational": ("generated from opc-sft-stage2 educational", "MIT", "T6 bug fixing, bugs verified by the tests"),
    "opc-stage2-package": ("[OpenCoder-LLM/opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2) (package_instruct)", "MIT", "T1"),
    "glaive-v3": ("[glaiveai/glaive-code-assistant-v3](https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v3)", "Apache-2.0", "T1"),
    "magicoder-evol": ("[ise-uiuc/Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K)", "Apache-2.0", "T1"),
    "magicoder-oss": ("[ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K)", "MIT", "T1"),
    "codefeedback": ("[m-a-p/CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction)", "Apache-2.0", "T1"),
    "self-oss-instruct": ("[bigcode/self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k)", "ODC-By", "T1"),
    "mceval-instruct": ("[Multilingual-Multimodal-NLP/McEval-Instruct](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/McEval-Instruct)", "CC-BY-SA-4.0", "T1"),
    "code-alpaca": ("[sahil2801/CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)", "CC-BY-4.0", "T1"),
    "websight": ("[HuggingFaceM4/WebSight](https://huggingface.co/datasets/HuggingFaceM4/WebSight) (v0.2)", "CC-BY-4.0", "T1: page description → HTML/CSS"),
    "rosetta-code": ("[christopher/rosetta-code](https://huggingface.co/datasets/christopher/rosetta-code)", "GFDL-1.2", "T1"),
    "nl2bash": ("[jiacheng-ye/nl2bash](https://huggingface.co/datasets/jiacheng-ye/nl2bash)", "GPL-3.0", "T1 (Bash)"),
    "nl2bash-custom": ("[AnishJoshi/nl2bash-custom](https://huggingface.co/datasets/AnishJoshi/nl2bash-custom)", "GPL-3.0", "T1 (Bash)"),
    "bash-instruct-iii": ("[Frost2o24/bash-instruct-III-55k](https://huggingface.co/datasets/Frost2o24/bash-instruct-III-55k)", "MIT", "T1 (Bash)"),
    "bash-command-6k": ("[emirkaanozdemr/bash_command_data_6K](https://huggingface.co/datasets/emirkaanozdemr/bash_command_data_6K)", "Apache-2.0", "T1 (Bash)"),
    "bash-commands": ("[aelhalili/bash-commands-dataset](https://huggingface.co/datasets/aelhalili/bash-commands-dataset)", "MIT", "T1 (Bash)"),
    "tldr-pages": ("[tldr-pages/tldr](https://github.com/tldr-pages/tldr)", "CC-BY-4.0", "T1 (Bash): command description → command"),
    "exercicios-v1": ("codelm exercise generator (procedural)", "MIT", "T1 with tests"),
    "t6-exercicios-v1": ("generated from the codelm exercises", "MIT", "T6 bug fixing, bugs verified by the tests"),
    "traducao-v2": ("machine translation of the prompts of other sources", "license of the original example",
                    "prompts translated to pt/es/fr/de/it (C, TS, Rust, JS, Java, Bash)"),
    "traducao-v2b": ("machine translation of the prompts of other sources", "license of the original example",
                     "prompts translated to pt/es/fr/de/it (Python, HTML)"),
}

TASKS = {
    "T1": "natural-language request → code",
    "T2": "docstring + signature → implementation",
    "T5": "code → docstring",
    "T6": "fix a bug or a syntax error → corrected code (5% of the syntax cases say there is no error)",
    "T7": "transform JSON/YAML (convert, format, extract a field, edit)",
}


def table(rows: list[tuple], header: tuple) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def write_card() -> None:
    stats = json.loads((OUT_DIR / "stats.json").read_text())
    ex, tok = defaultdict(int), defaultdict(int)
    for fonte, task, lang, nl, kind, value in stats["celulas"]:
        split = kind.replace("_tokens", "")
        bucket = tok if kind.endswith("_tokens") else ex
        for key in (("fonte", fonte), ("task", task), ("lang", lang if not lang.startswith("outra") else "other"),
                    ("nl", nl), ("all", "all")):
            bucket[(*key, split)] += value
    total_tok = tok[("all", "all", "train")] + tok[("all", "all", "validation")]

    def rows_for(kind: str, names: list[str]) -> list[tuple]:
        out = []
        for name in names:
            t = tok[(kind, name, "train")] + tok[(kind, name, "validation")]
            out.append((f"`{name}`", f"{ex[(kind, name, 'train')]:,}", f"{ex[(kind, name, 'validation')]:,}",
                        f"{t / 1e6:,.0f}M", f"{100 * t / total_tok:.1f}%"))
        return out

    head = ("", "train", "validation", "tokens", "share")
    by = lambda kind: sorted({k[1] for k in ex if k[0] == kind}, key=lambda n: -tok[(kind, n, "train")])
    sources = [(f"`{f}`", *SOURCES.get(f, ("", "", "")), f"{ex[('fonte', f, 'train')] + ex[('fonte', f, 'validation')]:,}")
               for f in by("fonte")]
    excl = stats["exclusoes"]
    header = "\n".join([
        "---", f"license: {LICENSE}", "language:", *(f"- {nl}" for nl in NLS),
        "tags:", "- code", "- sft", "- instruction-tuning", "pretty_name: codelm SFT v2",
        "size_categories:", "- 10M<n<100M", "task_categories:", "- text-generation",
        "extra_gated_prompt: |-", *(f"  {line}" for line in STACK_TERMS.splitlines()),
        "  Part of the code also comes from Stack-Edu (HuggingFaceTB/stack-edu, a subset of The Stack v2);",
        "  the Terms of Use of The Stack v2 apply to it as well.",
        "  Some examples come from sources under non-commercial (CC-BY-NC-4.0) and copyleft (CC-BY-SA-4.0,",
        "  GPL-3.0, GFDL) licenses; see the License section.",
        "extra_gated_fields:", "  I have read the Terms of Use of The Stack and agree with them: checkbox",
        "  I will use this dataset only for non-commercial purposes: checkbox",
        "configs:", "- config_name: default", "  data_files:",
        "  - split: train", "    path: data/train/*/*.parquet",
        "  - split: validation", "    path: data/validation/*.parquet",
        "---", ""])
    text = header + f"""# codelm SFT v2

Supervised fine-tuning data of [codelm](https://github.com/aseidelo/code-language-model), a small
code model trained from scratch. **{(ex[('all', 'all', 'train')] + ex[('all', 'all', 'validation')]) / 1e6:.1f}M examples,
{total_tok / 1e9:.1f}B tokens**, in 11 programming languages (plus a small share of others) and
6 natural languages (en, pt, es, fr, de, it).

We did not use an LLM to write or rewrite any example. The dataset combines open instruction
datasets (several of them were generated by LLMs; see their cards), examples derived from real code
(documented functions, JSON/YAML files), procedurally generated bug and syntax repairs, and prompts
machine translated with Opus-MT.

```python
from datasets import load_dataset
ds = load_dataset("seidel/codelm-sft-v2", split="train")          # all examples, text only
ds = ds.filter(lambda r: r["task"] == "T1" and r["lang"] == "python")
```

## Fields

| field | content |
|---|---|
| `id`, `grupo` | example id; `grupo` joins an example with its translations and its sibling tasks (T2/T5 of the same function). The validation split is chosen by group |
| `fonte`, `fonte_original`, `source` | source folder (table below); for translations, the source of the original example |
| `task` | task id (table below) |
| `lang` | programming language of the response (`outra:<name>` for languages outside the 11 main ones) |
| `nl` | natural language of the prompt |
| `level` | `facil` / `medio` / `dificil` (easy / medium / hard), a heuristic of the source |
| `prompt`, `code` | the request; `code` is the input code when the task has one (T2, T5, T6, T7) |
| `response` | the expected answer |
| `kind` | `code`, `description` or `json`: what the response is |
| `lang_in_prompt` | the prompt names the programming language |
| `prompt_en` | original English prompt (translations only) |
| `tests` | unit tests, when the source has them |
| `licenca`, `repo`, `path`, `licenses` | license of the source dataset; repository, file and repository licenses for examples from code |
| `n_tokens` | length with the codelm v2 tokenizer (16k vocabulary) |
| `peso` | sampling weight used to train codelm (see Mix) |

## Tasks

{table([(f'`{t}`', d) for t, d in TASKS.items()], ('task', 'description'))}

{table(rows_for('task', by('task')), head)}

## Programming languages

{table(rows_for('lang', by('lang')), head)}

## Natural languages

{table(rows_for('nl', by('nl')), head)}

Non-English prompts are machine translated from English (Opus-MT through CTranslate2, greedy).
The code, comments and docstrings stay in English: a translated example is bilingual on purpose.
Translated prompts that lost code markers, kept English or changed too much in length were
rejected. For T2, the translated prompt is the hand-written template with the same index in the
target language, not a machine translation.

## Sources

{table(sources, ('folder', 'origin', 'license', 'used as', 'examples'))}

## How it was built

1. **Conversion.** Each source goes to its own folder with a common schema, keeping its license
   and provenance. Prompts that answer themselves, Q&A without code and similar cases are kept at
   this point and marked later.
2. **Derived and procedural data.**
   - T2/T5 from documented functions of the codelm pretraining corpus (one task per function and
     template, original docstrings).
   - T6 syntax repair: functions of C, TypeScript, Rust, JavaScript, Java and Bash get 1 to 3
     token-level mutations (delimiters, `;`, unclosed strings, typos in keywords, swapped
     operators, Bash `fi`/`done`/...). The original must parse cleanly and the mutant must fail
     with tree-sitter (and `bash -n`). Variants: only the instruction, with the error line, or
     with the description.
   - T6 bug fixing from sources with tests (KodCode, AceCode, OpenCoder educational, codelm
     exercises): the bug must make the tests fail and the original must pass.
   - JSON/YAML: files of the-stack-dedup that parse, turned into conversion, repair, formatting,
     field extraction and editing tasks with template instructions in the 6 languages.
3. **Translation** of prompts to pt/es/fr/de/it, first for the under-represented languages,
   then Python and HTML. Each example is translated to at most one language.
4. **Quality control** (marks, never edits): the code parses (Python `ast`, tree-sitter, JSON/YAML
   parsers) and is accepted by the codelm grammar mask (Python, JSON); consistent indentation; no
   placeholders (`# your code here`, `pass`, `TODO`) or unclosed fences; the language named in the
   prompt matches the response; size (≤ 8k tokens); no binaries, base64, minified code or
   repetition.
5. **Post-processing**, with rules taken from a human review of random samples: the prompt
   already contains the answer; response with no identifiable language; diffs; trivial examples;
   the prompt refers to code, data or links that are not there; T1 whose answer is prose. Wrong
   `lang` and `nl` labels were corrected (CUDA labelled as C, PHP as HTML, Chinese labelled as
   English). The heuristic rules were validated by hand before being used.
6. **Deduplication.** Exact (same task, natural language, programming language, prompt and
   response) and near duplicates (MinHash of the prompt, Jaccard ≥ 0.8, across sources). One
   example stays per group: the one that passes the checks, then sources with tests, then
   sources from real code, then synthetic instructions, then translations.
7. **Decontamination** against HumanEval, MBPP, MultiPL-E, HumanEval-XL, McEval, CRUXEval,
   NL2SH-ALFA and the codelm evaluation exercises: an example is removed when it covers ≥ 50% of
   the 10-grams of a benchmark statement (or equals a short statement).
8. **Validation split**: ~0.07% of the groups, chosen by hash.

Removed before release ({sum(excl.get(k, 0) for k in ('qc', 'pos', 'duplicata', 'contaminado', 'idioma')):,} examples):
{excl.get('qc', 0):,} by quality control, {excl.get('pos', 0):,} by post-processing,
{excl.get('duplicata', 0):,} duplicates, {excl.get('contaminado', 0):,} contaminated and
{excl.get('idioma', 0):,} in other natural languages (zh, ja, ko, ru...).

## Mix

The dataset holds every example that passed the steps above, once. To train codelm, `peso`
sets the sampling: WebSight in English 0.5 (it is a single synthetic source and would otherwise be
11% of the data), Bash 2 (to reach ~3%), everything else 1. Python is ~63% of the tokens.

## License

The examples come from sources with different licenses, some of them incompatible with each
other. This compilation is released under **CC-BY-NC-SA-4.0**, the most restrictive combination
of the terms involved (non-commercial, share-alike, attribution). In addition, **each example keeps
the license of its source** (column `licenca`, and `licenses` for code from repositories), and
its terms apply to that example:

- KodCode and the examples derived from it are **CC-BY-NC-4.0** (non-commercial);
- McEval-Instruct is **CC-BY-SA-4.0**, nl2bash **GPL-3.0** and Rosetta Code **GFDL**: if you
  redistribute those examples, they stay under their own copyleft license;
- code from the-stack-dedup and CodeSearchNet keeps the license of its repository (permissive);
- code from Stack-Edu (`corpus-v2`, `t6s-corpus-v2` and their translations) comes from repositories
  with a permissive license **or with no license at all** (about half of those examples; `licenses`
  is then empty). It is redistributed under the Terms of Use of The Stack v2, as BigCode does, and
  removal requests are honored (see below);
- translations keep the license of the original example.

Access requires accepting the Terms of Use of The Stack, which apply to the code derived from it
(the-stack-dedup, and Stack-Edu from [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2)).

{REDACTION_NOTE.replace('The code used for training was not redacted.', 'codelm is trained on this redacted text.')}
Redaction counts: {excl.get('segredos', 0):,} secrets, {excl.get('e-mails', 0):,} e-mails.

## Limitations

- The data was checked by automatic rules and small human samples, not example by example.
  Responses from the instruction datasets were not executed unless the source had tests.
- Machine-translated prompts can be literal or slightly wrong.
- Python and English dominate; the other languages rely heavily on derived and translated data.
"""
    (OUT_DIR / "README.md").write_text(text)
    print(f"README.md: {len(text):,} caracteres")
