# Benchmarks

O que mede cada capacidade do modelo, em quais linguagens e idiomas, e em que etapa
usar. Os resultados ficam em `data/codelm_sft/eval/`.

## Com execução de testes

| Benchmark | Linguagens (problemas) | Licença | Formato | Etapa |
|---|---|---|---|---|
| HumanEval | python 164 | MIT | completar | base e SFT |
| MBPP | python 500 (test) | CC-BY-4.0 | pedido + testes (chat) | SFT |
| **MultiPL-E** HumanEval | js 161, ts 159, java 158, rust 156, bash 158 | MIT | completar | **base** e SFT |
| **MultiPL-E** MBPP | js 397, ts 390, java 386, rust 354, bash 382 | MIT | completar | base e SFT |
| **McEval** (generation) | c 50, bash 50, ts 50, rust 53, java 53, js 50, python 50 | CC-BY-SA-4.0 | instrução → código | SFT |
| **HumanEval-XL** | 80 × python, java, js, ts × en, pt, es, fr, de, it | Apache-2.0 | enunciado em vários idiomas | SFT |
| CRUXEval | python 800 | MIT | prever a saída do código | opcional |

- **MultiPL-E** (`eval/multipl_e.py`, integrado ao `eval.passk --suites`): prompt com
  comentário de documentação e assinatura aberta, sequências de parada próprias de
  cada linguagem. O programa avaliado é `prompt + continuação + testes`, e passa se
  sair com código 0. É o padrão dos modelos de código pequenos (SantaCoder,
  StarCoder), então dá para comparar.
- **C** só tem o McEval (50 problemas). O MultiPL-E tem C++, não C.
- **McEval-Instruct** está entre as fontes do SFT e é dos mesmos autores do McEval:
  a deduplicação (`eval/contamination.py`) precisa incluir o McEval.
- **NL2Bash**: o split de teste entra no SFT (política de máximo de dados), então
  ele não serve de benchmark.

## Páginas de resultado

Cada rodada do `eval.passk` grava o JSON (`data/codelm_sft/eval/passk/`) e um relatório com o que
foi rodado, no template de benchmark do Seidelbots: `docs/benchmarks/<nome do JSON>.html`
(`reports.benchmark`). O que entra na rodada é o `--suites`: por exemplo `humaneval javascript
typescript`, ou `todos` (HumanEval e as 5 linguagens do MultiPL-E HumanEval).

Linhas de base publicadas: Codex-85M e Codex-300M no HumanEval (Chen et al., 2021, pass@1 a
T=0.2; o 85M é o do nosso porte) e phi-1-small no HumanEval e no MBPP. Para refazer o documento de
resultados antigos: `uv run python -m reports.benchmark passk <resultado.json>`. O
`eval.benchmark compare` também gera uma página, com as variantes como linhas de base.

### Execução

No sandbox (`bwrap`, sem rede). O programa entra pela entrada padrão.

| Linguagem | Como roda | Ferramenta |
|---|---|---|
| javascript | node | node 24 (nvm) |
| typescript | `tsc --target esnext` (checa tipos, como o MultiPL-E) + node | typescript 5.6 em `model/tools` |
| java | `javac` + `java -ea` com javatuples no classpath | `sudo apt install openjdk-21-jdk-headless`; jar em `model/tools` |
| rust | `rustc --edition 2021` (debug, checa overflow) | rustup em `~/.cargo` |
| bash | bash | sistema |

## Sem execução: HTML, CSS, JSON, YAML

Não há benchmark executável. O McEval tem HTML, JSON e Markdown sem testes.

- **Pré-treino:** bits por byte por linguagem no `valid-v1` (contexto 2k na máquina
  local; 8k só no Modal).
- **Depois do SFT:** probes de validade próprias. O pedido descreve um JSON/YAML/HTML/CSS
  e a resposta precisa parsear e conter os campos ou elementos pedidos. Os enunciados de
  JSON e HTML do McEval servem de ponto de partida, com as verificações escritas por nós.

## Contaminação

`eval/contamination.py` mede a sobreposição de 13-gramas entre os problemas e um corpus.
No pré-treino v2 (Python), o HumanEval tem 22 problemas com ≥50% de cobertura no
enunciado ou na solução, e o pass@k nos limpos não é menor. O MBPP tem 130 problemas
com a solução ≥80% coberta. Falta estender para o MultiPL-E (por linguagem) e para as
fontes do SFT.

## Plano

1. MultiPL-E HumanEval, greedy, no base-v2: js, ts, bash; java e rust depois de instalar.
2. pass@k com amostragem no Modal (na XPU, 164 problemas × 121 amostras levam horas).
3. Depois do SFT: McEval, HumanEval-XL e as probes de validade.
