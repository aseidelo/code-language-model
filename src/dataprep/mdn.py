"""Base de compatibilidade do MDN (``@mdn/browser-compat-data``).

Lista todas as tags e atributos HTML, propriedades/valores/seletores CSS e
APIs JavaScript, com a versão de cada navegador que os suporta.
"""

import json
import urllib.request

from codelm.languages import DATA_DIR

BCD_VERSION = "8.1.6"
BCD_URL = f"https://cdn.jsdelivr.net/npm/@mdn/browser-compat-data@{BCD_VERSION}/data.json"
BCD_PATH = DATA_DIR / "external" / f"browser-compat-data-{BCD_VERSION}.json"


def load_bcd() -> dict:
    """Carrega a base, baixando na primeira vez (~20 MB)."""
    if not BCD_PATH.exists():
        BCD_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Baixando {BCD_URL}")
        urllib.request.urlretrieve(BCD_URL, BCD_PATH)
    return json.loads(BCD_PATH.read_text())


def bcd_keys(node: dict) -> list[str]:
    """Subchaves de um nó do BCD que são nomes de features (não metadados)."""
    return [k for k in node if k != "__compat" and "_" not in k]
