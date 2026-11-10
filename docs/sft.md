# Ajuste fino supervisionado (SFT)

Decisões sobre os dados do ajuste fino do codelm: quais tarefas o modelo
aprende, quanto de cada uma e como os dados são gerados. Os números abaixo são o
ponto de partida e ficam numa configuração, fáceis de mudar.

## Por que SFT antes de RL

O pré-treino só mostrou ao modelo arquivos crus (`<|lang:...|><|code|>...`). Os
tokens de pedido e de resposta (`<|prompt|>`, `<|response|>`,
`<|description|>`, `<|json_output|>`) nunca apareceram nessa posição. Sem SFT, o
modelo não sabe responder a um pedido e quase nunca acerta, e o GRPO não aprende
nada quando todas as amostras de um grupo recebem recompensa zero. Além disso,
descrever código não tem recompensa verificável.

Ordem: **SFT** (formato, tarefas, idiomas) → medir pass@k em problemas com
testes → **GRPO** só na geração de código, com a execução como recompensa.

## Tarefas

| Tarefa | Entrada → saída | Tipo da resposta | Linguagens |
|---|---|---|---|
| **T1** descrição → código | pedido em linguagem natural → código | `code` | as 10 |
| **T2** docstring → código | assinatura + docstring → função implementada | `code` | 6 |
| **T3** código → descrição | código → texto em linguagem natural | `description` | as 10 |
| **T4** código → descrição estruturada | código → JSON no schema abaixo | `json_output` | as 10 |
| **T5** código → docstring | função sem documentação → a mesma função com docstring | `code` | 6 |

- As 6 linguagens de T2 e T5 são as que têm docstring: Python, JavaScript,
  TypeScript, Java, C e Rust (JSDoc, Javadoc, rustdoc, comentário de bloco em C).
- Em T5 a resposta é o código inteiro com a docstring inserida, não só o texto
  dela: assim o sampler valida a gramática da resposta.

### Schema da descrição estruturada (T4)

```json
{
  "name": "primos",
  "summary": "Retorna os n primeiros números primos.",
  "parameters": [{"name": "n", "type": "int", "description": "Quantidade de primos."}],
  "returns": {"type": "list[int]", "description": "Os n primeiros primos, em ordem."},
  "side_effects": [],
  "complexity": "O(n² )",
  "dependencies": []
}
```

- As **chaves** são fixas e em inglês, como os identificadores do código; os
  **valores** em texto ficam no idioma do pedido.
- `type`, `complexity` e `returns` podem ser `null` quando não se aplicam (por
  exemplo, em linguagens sem tipos declarados ou funções sem retorno).
- Em código que não é uma função (um arquivo CSS, um YAML de configuração, uma
  página HTML), `name` é o nome do componente ou arquivo e `parameters` fica vazio.

## Balanceamento

Todas as dimensões são balanceadas ao mesmo tempo: cada exemplo cai numa
**célula** (tarefa × idioma × linguagem) e cada célula tem uma cota.

**Tarefas: 50% gerar código, 50% gerar descrições.**

| Gerar código | | Gerar descrições | |
|---|---|---|---|
| T1 descrição → código | 30% | T3 código → descrição | 20% |
| T2 docstring → código | 20% | T4 código → JSON | 15% |
| | | T5 código → docstring | 15% |

**Idiomas: partes iguais.** Inglês, português, espanhol, francês, alemão e
italiano, 1/6 cada, em todas as tarefas e linguagens.

**Linguagens: ponderadas pela riqueza semântica.**

- Nas tarefas com as 10 linguagens (T1, T3, T4): Python, JavaScript,
  TypeScript, Java, C e Rust com **13% cada** (78%); HTML, CSS, JSON e YAML
  com **5,5% cada** (22%).
- Nas tarefas com 6 linguagens (T2, T5): 1/6 cada.

## Dificuldade

Os dados ficam num **banco único**, com cada exemplo etiquetado por `tarefa`,
`idioma`, `linguagem`, `nivel`, `n_tokens` e `fonte`. Fácil, médio e difícil
são **misturas** tiradas desse banco, não datasets separados: cada tamanho de
modelo usa a sua receita sem refazer os dados.

O nível é medido no próprio exemplo, não pela fonte:

| Nível | Tamanho da resposta | Conteúdo típico |
|---|---|---|
| fácil | até ~256 tokens | uma função ou um bloco simples |
| médio | 256 a 1.024 tokens | várias funções, uma classe |
| difícil | mais de 1.024 tokens | arquivo inteiro, programação competitiva |

O banco guarda texto cru, não tokenizado. O limite de contexto (hoje 2.048
tokens) é aplicado ao montar a mistura, então um modelo com contexto maior
aproveita os exemplos longos.

## Volume

| Mistura | Exemplos | Tokens por exemplo | Tokens |
|---|---|---|---|
| fácil | 240 mil | ~400 | ~100M |
| médio | 80 mil | ~800 | ~65M |
| difícil | 40 mil | ~1.800 | ~70M |
| **total** | **360 mil** | | **~235M** |

Receita do modelo small (42M): a mistura fácil inteira e ~20% da média. Modelos
maiores aumentam a parte média e difícil.

Na mistura fácil, as cotas por tarefa e as células extremas ficam assim:

| Tarefa | Total | Por idioma | Célula maior | Célula menor |
|---|---|---|---|---|
| T1 | 72 mil | 12 mil | ~1.560 | ~660 |
| T2 | 48 mil | 8 mil | ~1.330 | ~1.330 |
| T3 | 48 mil | 8 mil | ~1.040 | ~440 |
| T4 | 36 mil | 6 mil | ~780 | ~330 |
| T5 | 36 mil | 6 mil | ~1.000 | ~1.000 |

## De onde vêm os dados

A base é **código real do nosso corpus** (split de treino do the-stack, separado
por repositório como no pré-treino). A parte em linguagem natural é escrita por
um LLM auxiliar. Datasets públicos completam T1 em Python e JavaScript.

| Tarefa | Fonte |
|---|---|
| T1, T3, T4 | **instrução reversa**: a partir de um trecho real, o LLM escreve o pedido que levaria a ele, a descrição ou o JSON. Mais Magicoder OSS-Instruct, self-oss-instruct e similares em T1. |
| T2, T5 | funções com docstring extraídas do corpus com tree-sitter. |

O levantamento do que existe em cada célula está em `docs/sft_candidatos.md`.

**LLM auxiliar:** Claude Haiku 5.5 (`claude-haiku-5-5`), pela API, com chave
própria. O volume vai pela Message Batches API (metade do preço). Em T3 e T4 o
texto gerado é o alvo do treino: esses usam esforço médio, e o `build` descarta
respostas de T4 com parâmetros que não estão no código. Orçamento: poucas dezenas de
dólares (o Sonnet custaria ~20 vezes mais). Tradução em
`src/dataprep/translate.py`.

## Idiomas e tradução

- **Gerar direto no idioma alvo sempre que o texto é gerado:** na instrução
  reversa, o LLM já escreve o pedido ou a descrição em alemão, por exemplo.
  Tradução só para material que já existe em inglês (pedidos de datasets
  públicos, docstrings do corpus).
- **Um idioma por exemplo:** cada exemplo de origem é sorteado para um único
  idioma, sem 6 cópias do mesmo código.
- **O que traduz:** texto natural, comentários e docstrings que fazem parte da
  resposta, e os valores em texto do JSON.
- **O que não traduz:** identificadores, chaves do JSON e trechos de código
  citados no texto (protegidos por marcadores antes da tradução e restaurados
  depois).
- **Instruções variadas:** 30 a 50 paráfrases por tarefa e por idioma ("Descreva
  este código", "Explique o que esta função faz", ...), geradas uma vez e
  sorteadas, para o modelo não decorar uma frase fixa.
- **Controle de qualidade:** identificação de idioma (GlotLID/fastText),
  proporção de tamanho entre original e tradução, e retradução de uma amostra.

## Avaliação

- ~50 exemplos por célula, de repositórios fora do treino.
- Benchmarks externos, nunca usados no treino e deduplicados contra ele: MBPP e
  HumanEval (Python), MultiPL-E (demais linguagens) e HumanEval-XL (pedidos em
  vários idiomas).

`src/eval/sft_samples.py` gera respostas da validação para leitura e
`src/eval/benchmark.py` roda MBPP, HumanEval e T4 da validação (JSON), com e sem a
máscara de gramática, em `float`, `qat`, `gguf` (Q4_0) e `f16`. O benchmark exige o
GGUF exportado e o `llama-cpp-python`; o código gerado roda em `bwrap`, sem rede:

```bash
uv run python -m train.export_gguf --checkpoint model/checkpoints/<run>/<passo> --name <run>
uv run --with llama-cpp-python python -m eval.benchmark loss --checkpoint <run>/<passo>
uv run --with llama-cpp-python python -m eval.benchmark run --checkpoint <run>/<passo> --engine gguf [--no-grammar]
uv run --with llama-cpp-python python -m eval.benchmark compare --checkpoint <run>/<passo>
```

MultiPL-E, HumanEval-XL e a deduplicação contra o treino ainda não foram feitos.

## Licenças das fontes

Fontes com licença não comercial (KodCode, CC-BY-NC) e de compartilhamento pela
mesma licença (McEval, CC-BY-SA; Rosetta Code, GFDL) são aceitas. Cada exemplo
guarda a `fonte`, para filtrar se a licença do projeto mudar.

## Geração

### T2 e T5 (docstrings)

`src/dataprep/sft_docstrings.py`, a partir das funções documentadas do corpus:

```bash
uv run python -m dataprep.sft_candidates corpus      # extrai as funções (sft/units/)
uv run python -m dataprep.sft_docstrings select      # escolhe as funções de cada célula
uv run python -m dataprep.translate submit data/codelm_sft/docstrings/translate_in.jsonl
uv run python -m dataprep.translate collect <batch_id> data/codelm_sft/docstrings/translate_out.jsonl \
    --input data/codelm_sft/docstrings/translate_in.jsonl
uv run python -m dataprep.sft_docstrings build --translations data/codelm_sft/docstrings/translate_out.jsonl
```

- No máximo 20 funções por repositório em cada nível, e cada função em uma
  única célula. Testes, docs geradas por ferramenta e cabeçalhos de arquivo
  ficam de fora.
- Idioma da doc com o lingua: docs já em um dos 6 idiomas (com confiança alta)
  preenchem as células desse idioma sem tradução. As em inglês preenchem o
  resto, e as de idioma incerto só entram nas células traduzidas.
- A doc traduzida é reescrita no estilo do original (docstring, `/** */`,
  `///`, `//`). As frases de instrução vêm de `assets/sft/instructions.json`
  (15 por tarefa e idioma). Em metade dos exemplos a linguagem já vem informada
  no cabeçalho.

- O `build` descarta traduções idênticas ao original (docs só de tags, como
  `@param {T} x`, sem texto no idioma da célula), respostas que são um resumo do
  formato (`{"translation": "..."}`) e docs afetadas pelo bug antigo do
  `clean_doc`, que apagava linhas só de pontuação (cercas ` ``` `, `}`).
  Para recuperá-las:

```bash
uv run python -m dataprep.sft_docstrings repair --translations data/codelm_sft/docstrings/translate_out.jsonl
uv run python -m dataprep.translate submit data/codelm_sft/docstrings/translate_fix_in.jsonl
uv run python -m dataprep.translate collect <batch_id> data/codelm_sft/docstrings/translate_fix_out.jsonl \
    --input data/codelm_sft/docstrings/translate_fix_in.jsonl
uv run python -m dataprep.sft_docstrings build --translations data/codelm_sft/docstrings/translate_out.jsonl \
    --fixes data/codelm_sft/docstrings/translate_fix_out.jsonl
```

Cobertura: o fácil e o médio fecham as cotas, exceto JavaScript (faltam ~1k
por tarefa no fácil e ~600 no médio) e C fácil (~290). No difícil faltam de
300 a 1.000 por tarefa em todas as linguagens menos Python. Para completar:
CodeSearchNet (JavaScript) e docs escritas pelo Haiku para funções longas sem
documentação.

Qualidade das traduções (93,9 mil): 21% juntam linhas que o original quebrava
(fica uma linha longa na doc). A retradução de amostra ainda não foi feita.

### T1, T3 e T4 (instrução reversa)

`src/dataprep/sft_reverse.py`, a partir de funções (menos as usadas em T2/T5) e
arquivos inteiros do corpus:

```bash
uv run python -m dataprep.sft_reverse select                 # escolhe as unidades, imprime cobertura e custo
uv run python -m dataprep.sft_reverse submit --limit 500     # piloto
uv run python -m dataprep.sft_reverse collect <batch_id> data/codelm_sft/reverse/piloto.jsonl
uv run python -m dataprep.sft_reverse submit                 # tudo; imprime um id por lote
uv run python -m dataprep.sft_reverse collect <batch_id> data/codelm_sft/reverse/out.jsonl   # para cada lote
uv run python -m dataprep.sft_reverse submit --done data/codelm_sft/reverse/out.jsonl        # o que faltou
uv run python -m dataprep.sft_reverse build data/codelm_sft/reverse/out.jsonl
```

- O texto é escrito direto no idioma alvo. T3 e T4 sorteiam a frase de
  `assets/sft/instructions.json`; em T1 o pedido é o próprio texto gerado e,
  quando a linguagem não vem no cabeçalho, ele a menciona.
- As células fecham em 100%. HTML e JSON fáceis têm unidades de menos, então
  ~3,6 mil unidades de cada aparecem em duas tarefas, nunca duas vezes na mesma.
- O `select` imprime o custo estimado por nível e tarefa.
- A API limita os pedidos em processamento. Com a fila cheia, o `submit` para e
  avisa: basta esperar, rodar o `collect` e repetir com `--done`, que pula o que
  já tem resposta e reenvia o que falhou.
- Por enquanto só o corpus: os datasets públicos de T1 ficam como complemento.

## Exercícios verificáveis

Exercícios Python autocontidos, com testes executáveis, para ensinar a resolver e dar
sinal ao GRPO. Vêm de modelos parametrizados por conceito
(`dataprep/exercises/templates.py`) e do OpenCoder educational. Toda solução é executada
no sandbox, e os testes precisam reprovar respostas vazias ou constantes. A validação
tem só modelos de exercício (ou funções, no OpenCoder) nunca vistos no treino.

```bash
uv run python -m dataprep.exercises.build
uv run --with llama-cpp-python python -m eval.exercises --checkpoint base-sft-v1/step-06400
uv run --with llama-cpp-python python -m eval.viewer exercises base-sft-v1/step-06400
```

- Por enquanto, um SFT separado, só para o piloto. Na versão final, os exercícios entram
  no SFT único após o pré-treino: `sft/exercises/` tem as colunas do banco para isso.
- Em `pack-exercicios`, `level` é o conceito, para ver a perda por conceito.
- Três partes sem problema em comum: `train` (SFT), `rl` (GRPO) e `valid`. O GRPO
  precisa de problemas que o SFT não viu; detalhes em `docs/grpo.md`.
