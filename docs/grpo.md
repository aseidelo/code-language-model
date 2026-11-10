# GRPO

Ajuste por reforço sobre os exercícios verificáveis (ver `docs/sft.md`): o modelo
gera várias respostas para o mesmo exercício, cada uma recebe uma recompensa, e o
gradiente favorece as melhores do grupo.

- **Vantagem relativa ao grupo:** a recompensa de cada amostra menos a média do seu
  grupo, sem dividir pelo desvio, como no Dr. GRPO (Liu et al., 2025). Grupo uniforme
  não ensina nada. Sem a divisão, uma diferença pequena continua pequena, e os pesos
  das dimensões valem o que dizem (ver "Histórico" abaixo).
- **Mesma máscara:** amostragem e perda usam a máscara de gramática e a temperatura,
  com QAT ligado, como no SFT.
- **Perda por token:** a soma das log-probs de cada resposta, dividida pelo total de
  tokens gerados no passo (como no DAPO). Com a média por resposta, cada token de uma
  resposta longa pesava menos, e uma resposta errada e longa era pouco punida (o viés
  de comprimento do Dr. GRPO). A escala fica a mesma da média por resposta.
- **Sem currículo:** os exercícios saem de uma permutação de todo o conjunto, sem
  reposição (ver "Histórico: o `grpo-v2`"). Avaliação em T=0 com modelos de exercício
  nunca vistos (`eval.exercises`).

## Dados

O GRPO precisa de problemas que o SFT não viu. No piloto, o `base-ex-v3` acertou 100%
das receitas do gerador que estavam no seu SFT, com T=0,7: os grupos ficaram uniformes,
e o gradiente, zero. No OpenCoder, que também estava no SFT mas tem 28 mil problemas
diferentes, o acerto ficou entre 9% e 48%. Decorar receitas repetidas é fácil, mesmo
que um SFT maior e mais variado decore menos.

Por isso os exercícios (`dataprep.exercises.build`) têm três partes, sem problema em
comum:

| Parte | Uso | Gerador | OpenCoder |
|---|---|---|---|
| `train` | SFT | as receitas restantes | o resto das funções, até `--opencoder` |
| `rl` | GRPO | `RL_TEMPLATES` (por enquanto nenhuma) | 25% das funções |
| `valid` | avaliação | `HELD_OUT` (17 receitas, com as gêmeas juntas) | 5% das funções |

- **No OpenCoder, a parte vem do nome da função:** o mesmo problema reescrito cai sempre
  na mesma parte. No treino final do zero, o SFT único usa `train`, e o GRPO usa `rl`,
  sem nenhum ajuste.
- **No piloto,** o `base-ex-v3` foi treinado antes dessa separação, com parte do que
  agora é `rl`. O build recebe `--exclude-seen` com o treino dele, e esses exercícios
  vão para `rl-vistos.parquet`. Até o `grpo-v2`, o GRPO usava só os 5.589 não vistos,
  quase todos fora do alcance do modelo. Desde o `grpo-v3`, ele usa as duas partes
  (13.017; `--seen false` volta ao antigo): ter visto a solução uma vez no SFT não
  garante acertá-la a T=0,7. O `rl-vistos` atual saiu de `exercises-v3/train.parquet`,
  sem `count_vowels` e `sum_list`, que são nomes de receitas da validação.
- **Receitas novas do gerador** (Fase 5b: laço `for`, casting, cópias) podem ir para
  `RL_TEMPLATES` e dar ao GRPO os níveis iniciais, que hoje só existem no SFT.
- **Testes ocultos:** no GRPO, o pedido perde os exemplos quando eles mostrariam todos os
  testes, para que copiar valores não dê recompensa.
- **Grupos mistos:** o log mostra a fração de grupos com recompensas diferentes e, entre
  parênteses, a fração com acerto e erro no mesmo grupo. Como readability e efficiency
  são contínuas, quase todo grupo tem recompensas diferentes, mesmo quando as 8 amostras
  erram. A fração com acerto e erro também sai por conceito: um conceito com poucos
  grupos mistos gasta passos sem sinal. É a medida para decidir se vale separar os
  exercícios por dificuldade (`oc-laco` é 55% do conjunto e acertava 3% na sondagem do
  `grpo-v2`).
- **Comprimento:** o log mostra os tokens médios das respostas certas e das erradas, para
  ver o efeito da perda por token.

## Recompensa

A recompensa tem três dimensões, cada uma com seu peso, que é um hiperparâmetro:

```text
recompensa = w_c × correctness + w_r × readability + w_e × efficiency
```

Pesos iniciais: `w_c = 1`, `w_r = 0,2` e `w_e = 0,2`. Readability e efficiency só
desempatam respostas corretas: resposta errada não ganha nada por ser legível. Na
prática, a recompensa é −1 (não compila), 0 (compila e erra) ou
`1 + 0,2 × R + 0,2 × E` (acerta). Cada dimensão é registrada separadamente no log do
treino.

| Dimensão | Faixa | Quando vale | Como é medida |
|---|---|---|---|
| **Correctness** | −1, 0 ou 1 | sempre | 1 se passa em todos os testes; 0 se falha algum teste ou usa um nome proibido pelo enunciado ("sem usar `max`"); −1 se não compila |
| **Readability** | 0 a 1 | só se correctness = 1 | média de quatro itens: fração de parâmetros e retornos com tipo; docstring na função pedida; `1 − avisos do pyflakes / 3`, com mínimo 0; comentários em 5% a 30% das linhas de código, 1 ou 0 |
| **Efficiency** | 0 a 1 | só se correctness = 1 | `min(1, referência / instruções)`: instruções de bytecode executadas nos testes, contra a referência do exercício |

**Detalhes:**

- **Não compilar:** resposta incompleta, sintaxe inválida, erro ao carregar o código,
  como um nome inexistente fora das funções, ou tempo esgotado (laço infinito): os
  testes rodam numa só execução, e não sobra o resultado dos outros. Erro dentro da
  função só reprova o teste.
- **Nome proibido:** verificado pela AST. Sem isso, a efficiency empurraria o modelo para
  a função pronta que o enunciado proíbe.
- **Readability:** itens que não se aplicam saem da média. Exercícios de atribuição, por
  exemplo, não têm função, então não contam tipos nem docstring. Comentários são
  medidos pelas linhas com `#`, sem contar a docstring. A faixa evita que o modelo
  aprenda a encher o código de comentários.
- **Instruções:** contadas com `sys.monitoring` (Python 3.12), em vez de tempo, que
  oscila com a carga da máquina e é dominado pelo sandbox. A contagem é determinística e
  só inclui o código da resposta, sem as instruções do próprio teste. Chamadas de
  funções prontas, como `sum`, contam como uma instrução.
- **Referência da efficiency:** o menor valor entre a solução do exercício e a melhor
  resposta correta já vista nele. Esse registro é salvo no checkpoint, para não se
  perder ao retomar. Começar pela solução evita que uma resposta correta só pelos testes
  vire um alvo impossível.
- **Sem nota parcial:** `return x` passa em parte dos testes sem resolver nada. Com
  parcial, o GRPO reforçava esse tipo de atalho contra respostas que tentavam e erravam.
  Readability e efficiency ainda são calculadas em respostas erradas, para o log
  (`details`), mas valem 0 na recompensa.

### Exemplos

Exercício `absolute(x)` ("retorna o valor absoluto de x sem usar abs"), com 5 testes e
uma referência de 40 instruções. Os valores de readability supõem pyflakes sem avisos.

| Resposta | Correctness | Readability | Efficiency | Recompensa |
|---|---|---|---|---|
| Correta, com tipos, docstring e um comentário; 40 instruções | 1 | 1 | 1 | **1,40** |
| Correta, sem tipos, docstring nem comentários; 40 instruções | 1 | 0,25 | 1 | **1,25** |
| Correta, mas com um laço desnecessário; 160 instruções | 1 | 0,25 | 0,25 | **1,10** |
| `return x`: passa nos 3 testes com `x ≥ 0` | 0 | 0 | 0 | **0,00** |
| `return abs(x)`: passa em tudo, mas usa o nome proibido | 0 | 0 | 0 | **0,00** |
| Sintaxe inválida | −1 | 0 | 0 | **−1,00** |

### Histórico: o `grpo-v1`

A primeira versão dava correctness parcial (`0,5 × k / (n − 1)` para `k` de `n` testes),
readability a toda resposta que compilava e vantagem dividida pelo desvio do grupo. Em
135 passos, a readability média subiu de 0,25 para 0,67, enquanto a correctness oscilou
sem tendência e a avaliação T=0 ficou parada (OpenCoder 43,1% → 43,8%, dentro do ruído).

A causa era a divisão pelo desvio. Num grupo em que as 8 amostras erravam do mesmo jeito,
a única diferença era a readability, e a divisão ampliava essa diferença até um sinal de
tamanho normal, qualquer que fosse o peso. O parcial fazia o mesmo com atalhos. Daí as
três mudanças: correctness binária, readability só com acerto e vantagem sem divisão.

Sem a divisão, as vantagens ficam menores (num grupo com 1 acerto em 8, a da amostra
certa cai de cerca de 2,6 para 0,875), e o mesmo `lr` vira um passo menor. Se o `grad`
ficar muito abaixo de 0,1 e o T=0 não se mover até o passo 100, o próximo teste é dobrar
o `lr`.

### Histórico: o `grpo-v2`

Com as três mudanças acima, `lr` 1e-5 e currículo por conceito, o `grpo-v2` fez 500
passos. A avaliação de 24 exercícios por conceito no treino parecia estável, mas a de
150 por conceito (2.100 exercícios, rotina `measure` na XPU) mostrou uma piora de
40,8% para 35,9% (−4,9 ±1,8): `lista` +25, `string` −30, `dicionario` −27, OpenCoder
−3,6.

O currículo explica boa parte disso. A ordem saía de uma sondagem com 8 exercícios por
conceito, e o conceito atual ficava com 75% dos grupos. O treino passou os 500 passos só
em `oc-lista` e `oc-dicionario`: nunca chegou a `oc-condicional`, `oc-expressao`,
`oc-laco` e `oc-auxiliar`, que somavam 80% do conjunto. Não havia evidência de que o
currículo ajudasse, então o `grpo-v3` testa o mais simples:

| | `grpo-v2` | `grpo-v3` |
|---|---|---|
| exercícios | 5.589 (`rl`) | 13.017 (`rl` + `rl-vistos`) |
| amostragem | currículo por conceito, com reposição | uniforme, sem reposição |
| perda | média por resposta | por token |
| `lr` | 1e-5 | 2e-5 |

Com 500 passos de 16 grupos, cada exercício aparece no máximo uma vez. Se os grupos
mistos de algum conceito forem raros, o próximo passo é separar em fácil, médio e
difícil.

### Resultado do `grpo-v3`

500 passos em 5,1 h. No treino, o acerto a T=0,7 em exercícios nunca vistos subiu de
~21% para ~31%. A fração de grupos com acerto e erro caiu de 44% para 28% nos passos
301–350 e voltou a 40% depois. Medido a T=0:

| modelo | interna (2.100) | HumanEval | MBPP | T4 |
|---|---|---|---|---|
| base-ex-v3 | 40,8% (±1,1) | 12/164 | 67/500 | 67,7% |
| grpo-v2 / 500 | 35,9% | 13 | 63 | 73,3% |
| grpo-v3 / 300 | 41,7% | 10 | 67 | 74,3% |
| grpo-v3 / 400 | 39,7% | 11 | 74 | 71,3% |
| grpo-v3 / 500 | 40,4% | 11 | 69 | 73,7% |

O v3 corrigiu o esquecimento do v2, mas não melhorou o T=0: a captura ficou perto de 0%
(critério em `docs/etapas.md`). O ganho a T=0,7 foi o modelo concentrando a
distribuição no que ele já fazia a T=0. O T4 subiu nos dois GRPOs (~+6, perto de 2
erros padrão), sem explicação ainda. Conclusão: com este modelo base, o gargalo está
antes do RL (ver a comparação com o Codex em `docs/etapas.md`).

## Estrutura do código

A recompensa segue a ideia de portas e adaptadores: o treino depende só da porta, e cada
linguagem tem seu adaptador. O piloto só implementa Python.

| Arquivo | Papel |
|---|---|
| `src/train/rl/reward/__init__.py` | porta: o protocolo `RewardModel.score(código, exercício) -> Scores` (correctness, readability, efficiency e detalhes), a combinação com os pesos e `reward_for(lang)` |
| `src/train/rl/reward/python.py` | adaptador Python: sandbox, AST, pyflakes, `sys.monitoring` e o registro das referências |
| `src/train/grpo.py` | laço de treino: chama `reward_for(lang)` e a combinação, sem saber da linguagem |

Uma linguagem nova entra com outro arquivo de adaptador, sem mudar o laço de treino.

## Execução na GPU

- **Atenção `eager`:** na geração com cache e preenchimento à esquerda, o kernel `sdpa`
  da XPU acessa memória inválida em alguns lotes, e a GPU perde o dispositivo
  (`UR_RESULT_ERROR_DEVICE_LOST`, com `Engine memory CAT error` no kernel). Com prompts
  de 25 a 273 tokens, a falha acontecia sempre no passo 4. O GRPO usa `eager`, com a
  mesma vazão. O SFT e o pré-treino seguem com `sdpa`, sem cache nem preenchimento à
  esquerda.
- **Desktop na GPU integrada:** uma falha da B580 chegou a travar a interface gráfica.
  Com o monitor na saída da placa-mãe, a B580 fica só para o treino.
- **Vazão:** cerca de 24 s por passo com 16 grupos de 8 amostras, com 2,3 GB de memória
  (`train.grpo throughput`). A contagem de instruções da recompensa soma pouco.
