# SFT v2

Dados do ajuste fino do base-v2 em diante. Substitui o banco do `docs/sft.md`, que
continua valendo para as definições das tarefas (T1 a T5), do schema do T4 e dos níveis.

## Por que um v2

- O banco v1 não pode ser usado no modelo final: a instrução reversa (T1/T3/T4) e as
  traduções (T2/T5) foram geradas com o Claude, e os termos da Anthropic proíbem treinar
  modelos com essas saídas. O banco v1 serviu só para validar o pipeline.
- O v1 tinha pouco dado no formato do HumanEval: ~13M tokens de Python pedido→código,
  ~2M em inglês, mais ~8M de exercícios. O phi-1 usou ~180M tokens de exercícios.
- Faltava bash, que entrou no pré-treino v2.

## Política

1. **Máximo de dados.** Entra tudo o que pode ser usado legalmente. Nada é cortado
   na fonte. Qualidade e balanceamento (linguagem, idioma, tarefa, fonte) são
   controlados por **pesos na mistura de treino** (fração ou número de épocas por
   fonte), não por recorte.
2. **Só sai o que não pode ser usado**: saídas geradas por nós com modelos cujos
   termos proíbem treino (Claude). Licenças não comerciais (KodCode) e copyleft
   (GPL, GFDL, CC-BY-SA) são aceitas: o projeto é hobby, sem fins comerciais. Cada
   exemplo guarda a `licenca` para filtrar se isso mudar.
3. **Uma pasta por fonte**, para manter o controle de origem:
   `data/codelm_sft/v2/fontes/<fonte>/`, com parquets no schema do banco e um
   `meta.json` (origem, licença, data do download, versão do script, contagens por
   tarefa × linguagem × idioma × nível).
4. **Deduplicação marca, não apaga.** Colunas `dup_de` (duplicata exata ou próxima
   de outra fonte) e `contaminado` (sobreposição com HumanEval, MBPP, os valid e os
   problemas reservados para o GRPO). A mistura ignora o que estiver marcado.

### Schema

As colunas do banco v1 (`id`, `task`, `level`, `lang`, `nl`, `source`, `repo`, `path`,
`licenses`, `prompt`, `code`, `response`, `kind`, `lang_in_prompt`, `n_tokens`), mais:

| Coluna | Conteúdo |
|---|---|
| `fonte` | nome da pasta de origem |
| `licenca` | licença do dataset (ou do repositório, no corpus) |
| `tests` | testes executáveis, quando existem (Python) |
| `dup_de`, `contaminado` | marcas da deduplicação |

`n_tokens` usa o tokenizador v2. O texto continua cru, então trocar o tokenizador
é só refazer o pack.

## Estágio 1: sem geração própria

Tudo o que dá para montar sem usar um LLM nosso. Os números do levantamento v1 estão em
`docs/sft_candidatos.md`; o v2 refaz o levantamento com contagens completas.

| Fonte | Licença | Tarefas | Observação |
|---|---|---|---|
| banco v1, T2/T5 sem tradução | por repositório | T2, T5 | 21k exemplos, docstrings originais em inglês |
| exercícios v1 (OpenCoder + templates) | MIT / nosso | T1 | com testes |
| OpenCoder stage 2 (educational, package, evol, mceval) | MIT | T1 | educational com testes |
| OpenCoder stage 1 (infinity, largescale, realuser) | MIT | T1 | ~4M linhas, várias linguagens |
| self-oss-instruct, Magicoder OSS e Evol, CodeFeedback, CodeAlpaca | MIT / Apache / ODC-BY / CC-BY | T1 | quase tudo Python |
| Glaive Code Assistant v3 | Apache-2.0 | T1 | melhor fonte pública de Rust, C, TS |
| KodCode V1, OpenCodeInstruct, AceCode | CC-BY-NC / CC-BY / MIT / Apache | T1, T6 | Python com testes |
| WebSight v0.2 | CC-BY-4.0 | T1 | HTML |
| CodeSearchNet (python, java, javascript) | por repositório | T2, T5 | |
| corpus v2 (Stack-Edu + the-stack), tree-sitter | por repositório | T2, T5 | as 6 linguagens com docstring **e bash** |
| bash: tldr-pages (GitHub) | CC-BY-4.0 | T1, T3 | 7,6k páginas em inglês + pt_BR 1,1k, es 2,4k, fr 1,2k, de 0,9k, it 1,5k; ~5–8 exemplos (descrição → comando) por página. Pedidos **nativos nos 6 idiomas** |
| bash: bash-instruct-III-55k | MIT | T1 | 54k pedido → comando de uma linha (coreutils), sintético |
| bash: bash_command_data_6K, bash-commands-dataset | Apache-2.0 / MIT | T1 | 6,2k + 0,8k pedido → comando |
| bash: NL2Bash (original e derivados, ~10–25k) | GPL-3.0 | T1 | curado por humanos |
| bash: shell do Glaive / OpenCoder | Apache / MIT | T1 | entra junto com essas fontes |
| **T6 bug fix** (abaixo) | a da fonte da solução | T6 | procedural |

Rosetta Code (GFDL), McEval (CC-BY-SA) e NL2Bash (GPL-3.0) também entram. Licenças
copyleft obrigam a redistribuir **a base** pela mesma licença: para publicar os dados,
separar essas fontes pela coluna `licenca`. Para o modelo é zona cinzenta; se um dia
houver uso comercial, retreinar sem elas é um filtro na mistura.

Os pares do bash público são quase todos comandos de uma linha. Scripts e funções
completos vêm do corpus (funções com comentário de cabeçalho), a contar.

O estágio 1 é quase todo em inglês e não tem T3/T4 (não existem fontes públicas
boas para eles). O primeiro SFT do base-v2 sai daqui.

## T6: corrigir bug

Exemplos gerados de forma procedural, sem LLM: a partir de uma solução **com testes
que passam**, uma mutação na AST introduz um bug. O mutante só entra se **reprovar pelo
menos um teste** (garante que o bug existe e descarta mutações equivalentes) e não
estourar o tempo.

**Mutações** (operadores, sorteados):

- off-by-one em `range`, índices e limites de laço;
- troca de comparador (`<`↔`<=`, `>`↔`>=`, `==`↔`!=`);
- troca de operador aritmético (`+`↔`-`, `*`↔`/`, `//`↔`/`, `%`);
- `and`↔`or`, remover ou adicionar `not`;
- variável trocada por outra do mesmo escopo;
- constante errada (`0`↔`1`, `True`↔`False`, ±1 em literais);
- fatia errada (`[1:]`↔`[:-1]`, passo);
- linha apagada (atribuição, `append`, incremento);
- ramos de `if`/`else` invertidos;
- `return` errado (variável trocada, retorno antecipado).

**Variantes de pedido**, as três, sorteadas por exemplo:

| Variante | Pedido |
|---|---|
| (a) | instrução + código com bug |
| (b) | instrução + descrição da função + código com bug |
| (c) | instrução + código com bug + teste que falha e mensagem de erro (esperado × obtido ou traceback) |

- **Instruções**: templates fixos escritos por nós, várias paráfrases por variante, nos
  6 idiomas. Sem LLM.
- **Resposta**: o código inteiro corrigido (como no T5), para a gramática validar a saída.
- **Volume**: várias mutações por solução (operadores e posições diferentes).
- **Fontes**: as soluções com testes do estágio 1. HumanEval, MBPP e os problemas do
  `rl` e do `valid` dos exercícios ficam de fora.
- **Linguagens**: Python primeiro (tudo executável no sandbox). JS/TS (node) depois.
  Nas demais não há como garantir que o bug existe, então ficam de fora.
- O T6 tem recompensa verificável e também serve de tarefa para o GRPO.

## Estágio 2: balancear sem gastar com geração

Decidido em 2026-11-11, depois das contagens do estágio 1. **O dataset final tem 10B tokens.**

Ficam de fora a geração com LLM e a retrotradução (descrever funções sem doc), porque custariam
~US$100+ no Modal. As linguagens sub-representadas crescem por quatro caminhos sem custo:

- **WebSight** para HTML/CSS.
- **T6 de sintaxe** para C/TS/Rust/JS/Java/Bash: estraga a sintaxe de funções do corpus e pede a
  correção. É procedural e verificado pelo tree-sitter; traz código novo dessas linguagens.
- **Tarefas procedurais de JSON/YAML** a partir dos arquivos do corpus: converter, corrigir,
  formatar, extrair um campo e editar, com as instruções em template nos 6 idiomas.
- **Tradução de prompts** começando pelas linguagens sub-representadas. Um exemplo de TS traduzido
  para pt e es vira 3 exemplos de TS. Cada código aparece no máximo 2–3 vezes.

### Alvos

| Linguagem | Alvo | Original | Como completar |
|---|---|---|---|
| Python | 45% | 7,9B | subamostrar (prioridade para testes e T6) |
| Java | 9% | 0,80B | + traduções |
| C | 7% | 0,41B | traduções + T6 de sintaxe |
| TypeScript, Rust, JavaScript | 6% cada | ~0,3B cada | traduções + T6 de sintaxe |
| Bash | 3% | 0,08B | ~3 idiomas por exemplo + T6 de sintaxe |
| HTML/CSS | 6% | 0,04B | WebSight |
| JSON/YAML | 4% | 0,02B | procedural |
| Outras + só texto | 8% | 1,37B | subamostrar |

**Idiomas:** inglês 70%, e pt/es/fr/de/it com 6% cada. zh/ru/ja/ko ficam marcados, com peso zero.

### T6 de sintaxe (`dataprep/sft_syntax.py`)

Decidido em 2026-11-12:

- **Linguagens:** C, TypeScript, Rust, JavaScript, Java e Bash.
- **Fonte:** funções do corpus v2 (`v2/units/`), fáceis e médias, com 16–600 tokens.
  - Cada função é usada uma vez.
  - No máximo 25% vêm de funções documentadas, que já viraram T2/T5.
- **Mutações por token do tree-sitter:**
  - apagar abertura ou fechamento;
  - fechamento sobrando;
  - apagar `;` ou vírgula;
  - string sem fechar;
  - palavra-chave com erro de digitação;
  - operador trocado (`->`/`=>`/`::`);
  - Bash: apagar `fi`/`done`/`then`/`esac`/`;;`/`do`/`]]`.
  - O operador é sorteado antes da posição, para os delimitadores não dominarem.
- **Verificação:**
  - a original parseia limpa e o mutante tem `ERROR`/`MISSING`;
  - Java/JS/TS são testados também dentro de uma classe;
  - Bash também com `bash -n`.
  - Mutações que o parser aceita (por exemplo `;` em JS/TS, por causa do ASI) são descartadas.
- **Quantidade de erros:** 1 erro em ~80% e 2–3 em ~20%. **5% negativos**: código certo, pedido
  "tem erro? se tiver, corrija" e resposta em texto dizendo que não há erro.
- **Variantes nos 6 idiomas:**
  - a) só a instrução;
  - b) com a linha do erro, que no Bash é a mensagem do `bash -n`;
  - c) com a descrição, quando a função tem docstring.
- **Resposta:** o código original completo.
- **Volume:** 400k exemplos por linguagem, em `fontes/t6s-corpus-v2/`, com `tipo_bug="sintaxe"`.
- **Correções de 2026-11-13:**
  - o `bash -n` roda com `LC_ALL=C` (as mensagens tinham saído em português);
  - os negativos usam `kind="description"`, porque o template não tem `text`;
  - o QC só aplica as checagens de código quando `kind="code"`.

### JSON/YAML procedural (`dataprep/sft_jsonyaml.py`)

Decidido em 2026-11-13:

- **Sementes:** `the-stack-dedup/filtered-v2/{json,yaml}`, os mesmos arquivos do pré-treino.
  - Entram documentos de 16–600 tokens que parseiam, com objeto ou lista na raiz e linhas de até
    1000 caracteres.
  - No YAML, só um documento e sem tags desconhecidas.
  - Sobraram 66k JSON e 377k YAML (`v2/jy-sementes/`).
- **Tarefas:**

  | Tarefa | Código | Peso | Resposta |
  |---|---|---|---|
  | converter JSON ↔ YAML | T7 | 25% | o outro formato |
  | corrigir sintaxe | T6 | 25% | o documento inteiro corrigido |
  | formatar (só JSON) | T7 | 10% | indentação 2/4, minificado ou chaves ordenadas |
  | extrair valor | T7 | 15% | o valor no caminho pedido (`a.b[2]`) |
  | editar | T7 | 25% | o documento inteiro editado |

- **Converter:** só entra com round-trip exato. Ficam de fora datas e chaves que não são string.
- **Editar:** renomear (snake ↔ camel), remover, mudar valor ou adicionar chave.
  - Só entra quando o documento é reescrito idêntico: no JSON, `json.dumps` com indentação 2/4; no
    YAML, `ruamel.yaml`, que preserva comentários e formatação.
  - A edição é conferida contra a árvore do PyYAML.
- **Corrigir:**
  - Corrupções no texto, sorteando o tipo antes da posição.
  - JSON: apagar vírgula, `:` ou fechamento; vírgula sobrando; aspas faltando ou simples; chave sem
    aspas; comentário `//`.
  - YAML: tab na indentação, indentação quebrada, `:` faltando, aspas abertas.
  - O mutante precisa falhar no parser. 20% têm 2–3 erros e 5% são negativos.
  - Variantes: a) só a instrução; b) com a mensagem do parser (em inglês); c) com a linha do erro.
- **Idiomas:** en 53%, e pt/es/fr/de/it com ~9,4% cada.
- **Volume:** 800k exemplos.
  - O JSON usa cada semente em todas as tarefas que funcionarem, porque há poucas sementes.
  - O YAML completa o total.
  - Saída em `fontes/jy-stack/`, com `tipo_bug` igual ao nome da tarefa.
- **Template:** o código do pedido é codificado com a linguagem do cabeçalho da resposta. Na
  conversão, a entrada vai marcada com o formato de destino; isso só afeta o rastreador de
  strings e comentários.

### Tradução local

A tradução usa o **Opus-MT** (Helsinki-NLP, Marian, CC-BY 4.0), com um modelo por par en→pt/es/fr/de/it,
rodando na B580. Os textos são curtos, então o limite de 8k da XPU não se aplica. Os passos:

1. Separar a prosa em sentenças. Blocos de código, `inline code`, identificadores e termos do
   glossário viram marcadores.
2. Traduzir em lotes.
3. Restaurar os marcadores e validar. É rejeitada a tradução com marcador perdido, idioma errado
   no detector ou tamanho fora do esperado.

Os comentários dentro do código continuam em inglês. Nos T2/T5 do corpus só a docstring é traduzida,
porque os templates já existem nos 6 idiomas.

O **glossário** reúne os termos que não se traduzem, para a tradução não ficar estranha:

- tipos e estruturas: string, array, list, dict, tuple, set, int, float, bool, struct, enum;
- termos de programação: callback, closure, hash, token, thread, mutex, async/await, getter, setter;
- nomes de ferramentas e plataformas: Git, Docker, npm, pip, cargo, Node.js, React, JSON, YAML, CSV.

A lista fica em arquivo e cresce a partir da revisão do piloto.

Primeiro roda um **piloto de ~2k exemplos**, revisado na página local, para medir a qualidade e a
vazão. Se o Opus-MT ficar ruim, a alternativa é um LLM pequeno local (Qwen3-4B na B580): é
5–10× mais lento, e nesse caso o alvo dos 5 idiomas cai para 3% cada.

**Resultado (2026-11-12/13):**

- **Piloto:** 250 exemplos, ~87% ok, aprovado na revisão.
- **Vazão na B580:** ~500 caracteres/s, e um teste em lote derrubou a GPU (`DEVICE_LOST`).
- **Troca para CTranslate2 int8 na CPU**, medido em 2.000 sentenças:

  | Configuração | Caracteres/s | Sentenças válidas |
  |---|---|---|
  | 2 lotes × 4 threads, beam 4 | 2,6k | 98,8% |
  | 3 × 4, beam 2 | 5,1k | 98,4% |
  | **4 × 3, greedy** | **9,8k** | **98,2%** |

  Ficou com **4 lotes × 3 threads, greedy**: qualidade equivalente à do beam 4 na comparação
  lado a lado. Rodar vários lotes pequenos em paralelo rende mais do que 14 threads num lote só
  (CPU com núcleos P/E, sem HT).
- **Correções do mascaramento:**
  - variáveis de shell (`$1`, `$VAR`, `${x}`) viram marcador; antes viravam "US $ 1";
  - os nomes de código vêm só do código, sem comentários e strings; do prompt, só dos blocos
    cercados. A prosa do prompt ("`people`: an integer") fazia palavras comuns ficarem em inglês;
  - tradução idêntica ao original é rejeitada (`nao_traduzido`).

**Fase A** (`sft_translate traduzir`, saída em `fontes/traducao-v2/`):

- **Escopo:** linguagens escassas com reposição (o original em inglês fica), versão conservadora com
  1 tradução por exemplo.
  - C, TS, Rust, JS e Bash: 100% dos exemplos;
  - Java: 60%.
  - O idioma de destino é sorteado pelo hash do id.
- **Entram:** só exemplos em inglês, aprovados no QC, com `kind=code`, fora T5/T6/JSON-YAML.
- **Saída:** cada parte traduzida vira `part-<fonte>-<parte>-<idioma>.parquet`, com `prompt_en` e
  `fonte_original`. A rodada é retomável.
- **Fase B (pendente):** completar os idiomas com Python, HTML e outras linguagens. A ideia é
  priorizar prompt curto com código longo, que rende mais tokens por caractere traduzido.

## Ordem de trabalho

1. Levantamento completo do estágio 1 (contagens por fonte, licenças de bash).
2. Download e conversão de cada fonte para a sua pasta.
3. Gerador de T6.
4. Deduplicação e contaminação.
5. Mistura com pesos e pack com o tokenizador v2; SFT do base-v2.
6. Planejamento do estágio 2 (feito, ver acima).

Ordem atualizada do estágio 2:

1. Fechar o estágio 1. As cópias `opc-stage2-evol` e `opc-stage2-mceval` foram removidas. O T6
   do OpenCodeInstruct fica limitado a 5 partes (~400k soluções).
2. Tarefas procedurais de JSON/YAML e T6 de sintaxe (desenho antes de implementar).
3. Piloto de tradução local com o glossário, revisado na página.
4. Tradução completa, com prioridade para C/TS/Rust/JS/Bash.
5. QC e dedup em tudo, depois a mistura até os alvos e o pack.

## Estado (2026-11-11)

Estágio 1 convertido (`dataprep/sft_convert.py`): **23,8M exemplos, ~11,3B tokens** (estimados),
fora o WebSight (download em andamento) e o T6 (rodando). Ver `uv run python -m dataprep.sft_convert --list`.

| Linguagem | Tokens | % | Exemplos T1 / T2 / T5 |
|---|---|---|---|
| python | 7.646M | 67,7 | 8,36M / 3,60M / 3,60M |
| outras (C++, SQL, Go, C#...) | 1.417M | 12,5 | marcadas `outra:<linguagem>` |
| java | 799M | 7,1 | 125k / 828k / 827k |
| c | 412M | 3,6 | 46k / 377k / 376k |
| typescript | 304M | 2,7 | 15k / 447k / 446k |
| rust | 301M | 2,7 | 45k / 382k / 382k |
| javascript | 276M | 2,4 | 163k / 217k / 216k |
| bash | 78M | 0,7 | 238k / 53k / 53k |
| html, json, css, yaml | 67M | 0,6 | só T1: 69k, 28k, 17k, 9k |

Idioma: 10,6B tokens em inglês, 662M em outros (quase tudo chinês, do OpenCoder),
e só ~50M nos nossos 5 idiomas (~150k exemplos, metade T2/T5 com doc nativa).

Lacunas para o estágio 2: T1 em TypeScript, C, Rust, Java e markup; T3/T4 inteiros;
pedidos em pt/es/fr/de/it; bash além de comandos de uma linha.

- Fontes com testes e solução: OpenCodeInstruct 3,85M, KodCode 432k, OpenCoder
  educational 118k, exercícios v1 57k, AceCode-87K 36k. O AceCode-V2 tem só pedidos
  e testes (para o GRPO).
- Idioma: alfabeto (chinês, japonês, coreano, russo) e palavras funcionais; texto
  latino sem contagem clara fica em inglês. Detector estatístico (lingua) errava
  muito em pedidos com código.

## Controle de qualidade

O QC **marca, não apaga**: cada exemplo ganha colunas `qc_*` num parquet à parte
(`v2/qc/<fonte>/`, chave `id`), e a mistura decide o que usa e com que peso.

**Camada 1, determinística (tudo):**

| Coluna | Reprova quando |
|---|---|
| `qc_sintaxe` | o código não parseia: `ast` no Python, tree-sitter (`has_error`) nas outras, `json.loads`, parser de YAML |
| `qc_gramatica` | Python e JSON: a máscara de gramática do modelo rejeita a resposta (treinar numa saída que a máscara proíbe ensina algo que o modelo nunca vai gerar). **Reprovado** |
| `qc_indentacao` | indentação ímpar ou mista (`has_odd_indentation`, `has_mixed_indentation`) |
| `qc_incompleto` | placeholder (`# your code here`, `...`, `pass` sozinho, `TODO`), corpo vazio, cerca aberta, resposta sem código |
| `qc_linguagem` | o pedido nomeia uma linguagem e a resposta está em outra |
| `qc_tamanho` | pedido vazio ou de uma palavra; não cabe em 2k (marca) ou em 8k (reprova) |
| `qc_lixo` | binário, base64, minificado, linha gigante, repetição (razão de compressão) |

**Camada 2, execução:** `qc_testes_passam` nas fontes com testes (a solução original no
sandbox; o T6 já roda isso).

**Camada 3, duplicatas e contaminação:** `qc_dup_de` (exata por hash normalizado de pedido +
resposta; quase-duplicata por MinHash do pedido; entre fontes) e `qc_contaminado` (13-gramas
contra HumanEval, MBPP, MultiPL-E, McEval, HumanEval-XL, CRUXEval e os nossos valid/rl).

**Revisão humana:** amostra pequena, estratificada por fonte × linguagem × tarefa, numa
página web local (ok / ruim / motivo). Calibra os limiares e a taxa de erro de cada fonte,
que vira o peso na mistura.

**Relatório por fonte:** % reprovado em cada verificação, duplicatas, contaminação.

Fora por enquanto: nota do classificador educacional e perplexidade do base-v2. Só entram se
a revisão mostrar que as camadas acima não separam o bom do médio.

Custo medido: a gramática valida ~20–28 mil tokens/s por núcleo (~6 h para o Python todo
em 14 núcleos). Python e JSON são as únicas linguagens com gramática; nas outras, o QC
usa tree-sitter.
