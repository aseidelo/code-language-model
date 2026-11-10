"""Metas do SFT: tarefas, pesos e volumes de cada mistura (ver docs/sft.md).

Cada exemplo cai numa célula (tarefa × nível × linguagem × idioma); ``quota``
diz quantos exemplos a célula precisa.
"""

from codelm.languages import DATA_DIR, LANGUAGES, NATURAL_LANGUAGES

SFT_DIR = DATA_DIR / "codelm_sft"
# Banco único: um parquet por gerador, todos com as mesmas colunas (``BANK_COLUMNS``).
BANK_DIR = SFT_DIR / "bank"

DOC_LANGS = ("python", "javascript", "typescript", "java", "c", "rust")

# Fração de cada tarefa em todas as misturas: 50% gerar código, 50% gerar descrições.
TASKS = {
    "T1": 0.30,  # descrição -> código
    "T2": 0.20,  # docstring -> código
    "T3": 0.20,  # código -> descrição
    "T4": 0.15,  # código -> JSON
    "T5": 0.15,  # código -> docstring
}
TASK_LANGS = {"T1": tuple(LANGUAGES), "T2": DOC_LANGS, "T3": tuple(LANGUAGES), "T4": tuple(LANGUAGES), "T5": DOC_LANGS}

# Exemplos por mistura (nível).
MIXES = {"facil": 240_000, "medio": 80_000, "dificil": 40_000}

# Peso das linguagens nas tarefas com as 10: mais para as de programação.
MARKUP = ("html", "css", "json", "yaml")
LANG_WEIGHTS_ALL = {lang: 0.055 if lang in MARKUP else 0.13 for lang in LANGUAGES}

# Colunas do banco (entradas de ``chat_template.encode_example`` + etiquetas).
BANK_COLUMNS = ("id", "task", "level", "lang", "nl", "source", "repo", "path", "licenses", "prompt", "code",
                "response", "kind", "lang_in_prompt", "n_tokens")


def lang_weight(task: str, lang: str) -> float:
    langs = TASK_LANGS[task]
    if lang not in langs:
        return 0.0
    return LANG_WEIGHTS_ALL[lang] if len(langs) == len(LANGUAGES) else 1 / len(langs)


def quota(task: str, level: str, lang: str, nl: str | None = None) -> int:
    """Exemplos da célula; sem ``nl``, a soma dos 6 idiomas."""
    total = MIXES[level] * TASKS[task] * lang_weight(task, lang)
    return round(total if nl is None else total / len(NATURAL_LANGUAGES))
