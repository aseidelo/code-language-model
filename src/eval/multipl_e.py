"""MultiPL-E: HumanEval e MBPP traduzidos para outras linguagens (ver docs/benchmarks.md).

Cada problema traz o prompt (comentário de documentação + assinatura aberta), os
testes e as sequências de parada próprias da linguagem. O programa avaliado é
``prompt + continuação + testes``. Ele passa quando sai com código 0, como no
MultiPL-E.

Execução no sandbox (``bwrap``, sem rede, ``/tmp`` vazio). O programa entra pela
entrada padrão e é gravado num arquivo dentro do sandbox:

* ``js``: node;
* ``ts``: tsc (checagem de tipos, como no MultiPL-E) e depois node;
* ``java``: javac + ``java -ea`` (asserts ligados), com o javatuples no classpath;
* ``rust``: rustc (debug, com checagem de overflow) e o binário;
* ``bash``: bash.

Ferramentas sem sudo ficam em ``model/tools`` (typescript via npm, o jar do
javatuples). Java e Rust vêm do sistema e do rustup (``~/.cargo``, ``~/.rustup``).
"""

import shutil
import subprocess
from pathlib import Path

from datasets import load_dataset

from codelm.languages import PROJECT_MODEL_DIR

TOOLS_DIR = PROJECT_MODEL_DIR / "tools"
JAVATUPLES = TOOLS_DIR / "javatuples-1.2.jar"
TSC = TOOLS_DIR / "node_modules" / "typescript" / "bin" / "tsc"
HOME = Path.home()

# nossa linguagem -> sufixo do MultiPL-E
LANGS = {"javascript": "js", "typescript": "ts", "java": "java", "rust": "rs", "bash": "sh"}
TIME_LIMIT = {"javascript": 10, "typescript": 30, "java": 30, "rust": 30, "bash": 10}

SCRIPTS = {
    "javascript": "cat > /tmp/p.js && exec node /tmp/p.js",
    "typescript": f"cat > /tmp/p.ts && node {TSC} --target esnext --module commonjs --outDir /tmp/out /tmp/p.ts "
                  "&& exec node /tmp/out/p.js",
    "java": f"cat > /tmp/Problem.java && javac -nowarn -cp {JAVATUPLES} -d /tmp /tmp/Problem.java "
            f"&& exec java -ea -cp /tmp:{JAVATUPLES} Problem",
    "rust": "cat > /tmp/p.rs && rustc --edition 2021 -o /tmp/p /tmp/p.rs 2>/dev/null && exec /tmp/p",
    "bash": "cat > /tmp/p.sh && exec bash /tmp/p.sh",
}


def sandbox() -> list[str]:
    """bwrap com /usr e as ferramentas do usuário (node do nvm, rustup) só para leitura."""
    path = ":".join(str(p) for p in (shutil.which("node") and Path(shutil.which("node")).parent,
                                      HOME / ".cargo" / "bin") if p) + ":/usr/bin:/bin"
    binds = []
    for extra in (HOME / ".nvm", HOME / ".cargo", HOME / ".rustup", TOOLS_DIR):
        binds += ["--ro-bind-try", str(extra), str(extra)]
    return ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
            "--ro-bind-try", "/etc/alternatives", "/etc/alternatives", "--ro-bind-try", "/etc/java-21-openjdk",
            "/etc/java-21-openjdk", *binds, "--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev",
            "--setenv", "PATH", path, "--setenv", "HOME", str(HOME), "--setenv", "RUSTUP_HOME",
            str(HOME / ".rustup"), "--setenv", "CARGO_HOME", str(HOME / ".cargo"), "--"]


def missing_tools(lang: str) -> str | None:
    """O que falta para rodar a linguagem, ou None."""
    needs = {"javascript": ["node"], "typescript": ["node"], "java": ["javac", "java"],
             "rust": ["rustc"], "bash": ["bash"]}[lang]
    missing = [t for t in needs if not shutil.which(t) and not (HOME / ".cargo" / "bin" / t).exists()]
    if lang == "typescript" and not TSC.exists():
        missing.append(str(TSC))
    if lang == "java" and not JAVATUPLES.exists():
        missing.append(str(JAVATUPLES))
    return ", ".join(missing) or None


def run_program(lang: str, program: str) -> bool:
    try:
        done = subprocess.run([*sandbox(), "/usr/bin/bash", "-c", SCRIPTS[lang]], input=program.encode(),
                              capture_output=True, timeout=TIME_LIMIT[lang])
    except subprocess.TimeoutExpired:
        return False
    return done.returncode == 0


def problems(lang: str, suite: str = "humaneval") -> list[dict]:
    """Problemas do MultiPL-E (``suite`` humaneval ou mbpp) na linguagem."""
    rows = load_dataset("nuprl/MultiPL-E", f"{suite}-{LANGS[lang]}", split="test")
    return [{"id": row["name"], "prompt": row["prompt"], "tests": row["tests"], "stop": list(row["stop_tokens"])}
            for row in rows]
