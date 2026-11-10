"""Execução de código gerado: não confiável, roda com ``bwrap``, sem rede e sem acesso ao disco."""

import ast
import re
import subprocess
import warnings

SANDBOX = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--ro-bind", "/usr", "/usr",
           "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
           "--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev", "--"]
PASSED = "__testes_ok__"  # o código pode sair com exit(0) antes dos testes
TIME_LIMIT = 10  # segundos por problema
MEMORY_LIMIT_KB = 1 << 20


def syntax_ok(code: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return True


def execute(program: str) -> tuple[int, bytes] | None:
    """(código de saída, stdout), ou ``None`` se estourou o tempo."""
    limits = f"ulimit -v {MEMORY_LIMIT_KB}; exec /usr/bin/python3 -I -c \"$0\""
    try:
        done = subprocess.run([*SANDBOX, "/usr/bin/bash", "-c", limits, program], capture_output=True,
                              timeout=TIME_LIMIT)
    except subprocess.TimeoutExpired:
        return None
    return done.returncode, done.stdout


def passes(preamble: str, code: str, tests: str) -> bool:
    done = execute(f"{preamble}\n{code}\n{tests}\nprint('{PASSED}')\n")
    return done is not None and done[0] == 0 and PASSED.encode() in done[1]


def tests_passed(code: str, tests: list[str], preamble: str = "") -> int:
    """Quantos testes passam; o código que não carrega vale 0."""
    checks = "".join(f"try:\n    {t}\n    __passed += 1\nexcept BaseException:\n    pass\n" for t in tests)
    done = execute(f"{preamble}\n{code}\n__passed = 0\n{checks}print('{PASSED}', __passed)\n")
    found = re.search(rf"{PASSED} (\d+)".encode(), done[1]) if done else None
    return int(found.group(1)) if found else 0
