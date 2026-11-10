# Etapas do treino: objetivos e critérios de passagem

Três etapas: pré-treino, SFT e RLVR (GRPO). Cada uma tem uma métrica em benchmark
público, um critério para passar à seguinte e uma regra para quando não passa. Os
benchmarks públicos são o número principal. As avaliações internas (`eval.exercises`,
`train.grpo evaluate`) ficam como diagnóstico por conceito.

**Meta final do projeto:** passar o phi-1-small (350M) no HumanEval e no MBPP.

## Filosofia do treinamento

O objetivo final é um modelo **multilíngue**: várias linguagens de programação e
várias línguas naturais nos pedidos. O HumanEval e o MBPP cobrem um recorte só, Python
com pedidos em inglês. Eles são uma **sonda** do resultado nesse contexto específico, não
a definição de sucesso:

- servem para comparar com outros modelos e decidir a passagem entre etapas, porque são
  os benchmarks com mais resultados publicados;
- uma decisão de treino não pode melhorar a sonda à custa do resto, por exemplo com
  dados só de Python em inglês para subir o número. As outras linguagens e línguas são
  acompanhadas como regressão (hoje, o T4 em JSON e os pedidos em português dos
  exercícios);
- no futuro, entram mais benchmarks públicos, cobrindo outras linguagens (por exemplo,
  versões multilíngues do HumanEval e do MBPP) e outras línguas naturais. Cada um entra
  nos critérios das etapas da mesma forma: métrica pass@k, referência publicada e
  margem de ruído.

## Métricas comuns

- **HumanEval** (164 problemas) e **MBPP** (500, `test` do `full`), executados com os
  testes oficiais no sandbox (`eval.benchmark`).
- **pass@k** com o estimador sem viés do paper do HumanEval (Chen et al., 2021):
  `n` amostras por problema, `c` certas, `pass@k = 1 − C(n−c, k) / C(n, k)`, na média
  dos problemas.
- **K alvo = 8**, o tamanho do grupo do GRPO: o GRPO só aprende onde ao menos uma das 8
  amostras acerta, então o pass@8 na temperatura do treino é o alcance do RLVR. Valores
  maiores de K saem também, só como referência e para comparar com a literatura.
- **Ruído:** o erro padrão do pass@1 é ~3–4 pontos no HumanEval e ~1,5 no MBPP. Toda
  comparação, contra um critério ou entre modelos, precisa passar de 2 erros padrão. O
  critério principal usa os dois benchmarks somados (664 problemas), e cada um é
  reportado separado.
- **Contaminação:** os problemas do HumanEval e do MBPP não podem aparecer nos dados de
  nenhuma etapa (`benchmark_keys()` em `dataprep/exercises/opencoder.py` no
  OpenCoder; conferir também no corpus do pré-treino).

## 1. Pré-treino

O modelo ainda não tem o formato de chat. A avaliação é a do paper do Codex: o prompt é
a assinatura com a docstring, e o modelo completa o código.

- **Métrica:** HumanEval pass@1, pass@10 e pass@100 no modo completar. O pass@1 a
  T=0,2 e o pass@100 a T=0,8, como no paper. O MBPP pede 3 exemplos no prompt nesse
  modo e fica como complementar. A perda na validação acompanha como diagnóstico.
- **Referência:** a curva do Codex por tamanho (tabela abaixo). O Codex é um modelo de
  linguagem treinado em código Python do GitHub, sem nenhum treino de instruções nem de
  chat: é o análogo do nosso pré-treino. Com 113M, o nosso modelo fica entre o Codex-85M
  e o Codex-300M.
- **Critério para passar:** estar na curva do Codex ou acima dela para o nosso número de
  parâmetros (interpolando entre 85M e 300M).

### Referências: HumanEval no modo completar (Chen et al., 2021, Tabela 1)

| Modelo | pass@1 | pass@10 | pass@100 |
|---|---|---|---|
| GPT-Neo 125M | 0,75% | 1,88% | 2,97% |
| GPT-Neo 1.3B | 4,79% | 7,47% | 16,30% |
| GPT-Neo 2.7B | 6,41% | 11,27% | 21,37% |
| GPT-J 6B | 11,62% | 15,74% | 27,74% |
| TabNine | 2,58% | 4,35% | 7,59% |
| Codex-12M | 2,00% | 3,62% | 8,58% |
| Codex-25M | 3,21% | 7,1% | 12,89% |
| Codex-42M | 5,06% | 8,8% | 15,55% |
| **Codex-85M** | **8,22%** | **12,81%** | **22,4%** |
| **Codex-300M** | **13,17%** | **20,37%** | **36,27%** |
| Codex-679M | 16,22% | 25,7% | 40,95% |
| Codex-2.5B | 21,36% | 35,42% | 59,5% |
| Codex-12B | 28,81% | 46,81% | 72,31% |

No paper, a melhor temperatura é T=0,2 para o pass@1 e T=0,8 para o pass@100.

A outra tabela do paper que tem o Codex-S traz Codex-S-12B (pass@1 32,2%, pass@10
59,5%) e Codex-D-12B (20,3% e 46,5%). Ela é da seção de geração de docstrings, com
outro protocolo de avaliação, e não se compara direto com a tabela acima. O Codex-D
gera a docstring a partir do código, uma tarefa parecida com o nosso T4.
- **Se não passar:** o problema está nos dados ou no treino do pré-treino. Não adianta
  seguir para o SFT.
- **Resultado do `base-v1/step-04227` (2026-11-08, `eval.passk --mode completar`):**
  pass@1 1,2% a T=0, 0,6% a T=0,2; pass@10 1,8%; pass@100 4,9% a T=0,8. Só resolve
  problemas triviais (`strlen`, `max_element`, `add`). Fica na altura do GPT-Neo 125M e
  abaixo do Codex-12M, longe dos ~9% esperados para 113M: **não passa**. O harness
  está validado (as 164 soluções oficiais passam pelo mesmo caminho), mas falta
  calibrar a régua com um modelo de referência.
- **Pendente:** rodar modelos de referência (CodeParrot-small, SmolLM2-135M,
  CodeGen-350M-mono) no nosso harness para validar a régua.

## 2. SFT

- **Varredura:** T ∈ {0,2; 0,4; 0,6; 0,8; 1,0}, com n = 20 amostras por problema,
  mais uma rodada a T=0. Daí saem pass@1, pass@8 e pass@10 por temperatura.
  - O paper do HumanEval serve de conferência: para K < 10, a melhor T fica entre 0,2
    e 0,4.
- **Referência para K maior:** pass@100 a T=0,8, com n = 100 (também como no paper).
- **Referência:** o Codex-S, o Codex com ajuste supervisionado em funções isoladas e
  corretas (ainda no modo completar). É o análogo do nosso SFT. No modelo de 12B, o
  pass@1 sobe de 28,8% para ~37,7%. No gráfico do paper, perto de 10^8 parâmetros
  (~100M), o Codex-S chega a **pass@1 ~20% e pass@100 ~45%** (leitura aproximada do
  gráfico), contra 8,2% e 22,4% do Codex-85M. O SFT em funções isoladas mais que dobra o
  pass@1 nesse tamanho.
- **Resultado da etapa:** o pass@1 a T=0, o pass@8 na melhor temperatura e essa
  temperatura **T\***.
- **T\* vira a temperatura do GRPO.**
- **Critério para passar:** pass@8 em T\* ≥ X% no HumanEval e ≥ Y% no MBPP.
  - **X e Y ainda não definidos:** saem depois de medir o pass@8 do `base-ex-v3`, com um
    valor para o piloto e outro para a versão final.
- **Se não passar:** conferir se o pré-treino estava na curva do Codex. Se estava, o
  problema está nos dados do SFT.

## 3. RLVR (GRPO)

O objetivo é levar o T=0 para perto do pass@8 do SFT: fazer o modelo acertar de
primeira o que antes ele acertava em alguma de 8 tentativas.

- **Métrica: captura da distância**

  ```
  captura = (T0 depois do RL − T0 do SFT) / (pass@8 do SFT em T* − T0 do SFT)
  ```

  0% quer dizer que o RL não ajudou, e 100% que o T=0 chegou ao pass@8. Acima de 100%,
  passou do alcance, o que é raro segundo a literatura (Yue et al., 2025).
- **Critério para passar:**
  - captura **≥ 50%** no HumanEval + MBPP;
  - **sem regressão** maior que 2 erros padrão em nada que o SFT já fazia: o T4 (JSON)
    do `eval.benchmark` e os conceitos de `train.grpo evaluate` (150 por conceito). Essa
    condição pega o esquecimento visto no `grpo-v2`.
- **Transferência:** o GRPO treina nos exercícios do projeto, não no HumanEval nem no
  MBPP. Então a captura medida neles também mede a transferência. Se a captura for alta
  nos exercícios internos e baixa nos benchmarks, o problema está no conjunto de RL, não
  no algoritmo.
- **Se não passar:** revisar o desenho do RL (recompensa, lr, temperatura) ou o
  conjunto de exercícios, conforme a captura interna e a externa.

## Comandos

```bash
# benchmark a T=0, em lote na GPU
uv run python -m eval.benchmark run --checkpoint <run>/<step> --engine batch
# avaliação interna, 150 por conceito, em vários checkpoints
uv run python -m train.grpo evaluate <run>/<step> [<run>/<step> ...]
# pass@k por temperatura (pré-treino no modo completar; SFT no modo chat)
uv run python -m eval.passk --checkpoint <run>/<step> --mode completar
uv run python -m eval.passk --checkpoint <run>/<step> --mode chat
```

O `train.grpo sweep` faz a varredura de T nos exercícios internos.
