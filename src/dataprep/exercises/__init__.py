"""Exercícios Python autocontidos e executáveis, para o SFT e o GRPO (ver docs/sft.md)."""

from dataclasses import dataclass

from eval.sandbox import execute, syntax_ok, tests_passed

# Respostas que não fazem nada: se uma delas passa em todos os testes, os testes são fracos.
STUBS = ("None", "0", "1", "''", "[]", "{}", "True", "False", "args[0] if args else None")


@dataclass
class Exercise:
    id: str
    template: str
    concept: str
    nl: str
    prompt: str
    solution: str
    tests: list[str]
    source: str
    name: str | None = None  # função pedida; None quando a resposta é em nível de módulo
    lang_in_prompt: bool = True


def valid_exercise(exercise: Exercise) -> bool:
    if not syntax_ok(exercise.solution) or tests_passed(exercise.solution, exercise.tests) != len(exercise.tests):
        return False
    if exercise.name is None:
        return True
    tests = "\n".join(f"    {t}" for t in exercise.tests)
    stubs = "".join(f"""
def {exercise.name}(*args, **kwargs):
    return {stub}
try:
{tests}
    print('fraco')
except BaseException:
    pass
""" for stub in STUBS)
    done = execute(stubs)
    return done is not None and b"fraco" not in done[1]
