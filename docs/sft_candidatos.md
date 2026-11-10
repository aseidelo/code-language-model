# Candidatos a dados de SFT

Levantamento do que existe para cada tarefa e linguagem de `docs/sft.md`, antes
de gerar qualquer dado. Gerado por `src/dataprep/sft_candidates.py`:

```bash
uv run python -m dataprep.sft_candidates corpus   # nosso corpus (tree-sitter)
uv run python -m dataprep.sft_candidates public   # datasets públicos (amostra)
```

Resultados brutos em `data/codelm_sft/candidates/{corpus,public}.json`.
As funções extraídas do corpus ficam em `data/codelm_sft/units/<lang>.parquet`.

**Como ler:** os níveis vêm do tamanho do código em tokens (fácil ≤ 256,
médio ≤ 1.024, difícil > 1.024), estimado pelos caracteres por token de cada
linguagem. Nos datasets públicos, os números são **estimativas**: as primeiras
3.000 linhas de cada um, escaladas para o total de linhas.

## Nosso corpus (split de treino do the-stack)

Funções e métodos únicos (sem duplicatas exatas), com e sem documentação, e
arquivos inteiros:

| Linguagem | Com doc (F / M / D) | Todas as funções (F / M / D) | Arquivos (F / M / D) |
|---|---|---|---|
| python | 146k / 58k / 5,9k | 686k / 143k / 15k | 49k / 62k / 44k |
| javascript | 14k / 3,4k / 0,5k | 168k / 36k / 6,1k | 34k / 38k / 21k |
| typescript | 27k / 4,5k / 0,4k | 285k / 31k / 2,8k | 70k / 61k / 24k |
| java | 84k / 9,5k / 1,1k | 531k / 54k / 5,1k | 26k / 41k / 24k |
| c | 16k / 8,5k / 1,8k | 108k / 35k / 6,9k | 7,7k / 9,8k / 9,8k |
| rust | 48k / 7,7k / 1,0k | 310k / 48k / 6,1k | 12k / 18k / 16k |
| html | — | — | 5,5k / 8,1k / 6,7k |
| css | — | — | 11k / 11k / 6,3k |
| json | — | — | 6,6k / 6,1k / 2,9k |
| yaml | — | — | 33k / 18k / 4,8k |

- A documentação passa por filtros: de 15 a 2.000 caracteres, pelo menos 4
  palavras, sem licença, cabeçalho gerado ou diretiva de lint.
- Cerca de 10% das docs têm caracteres fora do ASCII. Parte delas está em
  outros idiomas (chinês, por exemplo) e precisa de identificação de idioma
  antes de usar.

## Datasets públicos

Estimativa de exemplos por linguagem, no formato fácil / médio / difícil (em
milhares). "Outra" conta respostas em outras linguagens (C++, Go, SQL, ...) ou
sem código. Quase todos os exemplos cabem em 2.048 tokens (≥ 97%; Rosetta
Code, 91%).

### Pedido → código (T1)

| Dataset | Licença | Linhas | Destaques | Outra |
|---|---|---|---|---|
| Magicoder OSS-Instruct | MIT | 75k | py 33/10/0, js 2/1/0, java 2/2/0, rust 2/1/0 | 16k |
| Magicoder Evol-Instruct | Apache-2.0 | 111k | py 36/14/0, js 5/2/0, html 2/2/0 | 40k |
| self-oss-instruct (exec) | ODC-BY | 51k | py 49/0/0 | 0 |
| CodeAlpaca | CC-BY-4.0 | 20k | py 9/0/0 | 9k |
| OpenCoder stage 2: educational | MIT | 118k | py 116/2/0, **com testes** | 0 |
| OpenCoder stage 2: evol | MIT | 111k | igual ao Magicoder Evol (mesma origem) | 40k |
| OpenCoder stage 2: package | MIT | 171k | py 52/117/0 (uso de bibliotecas) | 0 |
| OpenCoder stage 2: mceval | MIT | 36k | quase tudo em outras linguagens | 27k |
| OpenCoder stage 1: infinity | MIT | 1,0M | py 482/130/5, js 9/4/0, java 7/3/0, html 4/3/0 | 363k |
| OpenCoder stage 1: largescale | MIT | 2,5M | py 429/493/5, js 11/10/0, java 7/11/1, c 7/1/0, html 5/8/2, json 10/13/0, yaml 4/0/0 | 1,5M |
| OpenCoder stage 1: realuser | MIT | 676k | py 202/83/3, js 21/17/0, java 15/14/0, html 4/9/0 | 281k |
| CodeFeedback | Apache-2.0 | 157k | py 62/33/1, js 7/8/1, java 2/6/0 | 24k |
| Glaive Code Assistant v3 | Apache-2.0 | 950k | py 110/25/1, js 24/20/1, ts 3/3/0, java 15/22/1, c 7/10/1, rust 7/13/1, html 4/6/1, css 4/2/0 | 654k |
| McEval-Instruct | CC-BY-SA-4.0 | 36k | quase tudo em outras linguagens | 29k |
| Rosetta Code | GFDL | 79k | as nossas linguagens são poucas na amostra | 75k |
| WebSight v0.2 | CC-BY-4.0 | 1,9M | html 6/1.868/47 (páginas com Tailwind, sintéticas) | 0 |

### Docstring ↔ código (T2, T5)

| Dataset | Licença | Linhas | Destaques |
|---|---|---|---|
| CodeSearchNet python | por repositório | 412k | 267/131/12 |
| CodeSearchNet java | por repositório | 454k | 365/80/8 |
| CodeSearchNet javascript | por repositório | 124k | 81/37/5 |

### Com testes, para o RL

| Dataset | Licença | Linhas | Observação |
|---|---|---|---|
| KodCode V1 | **CC-BY-NC-4.0** | 484k | py; dificuldade: 152k fácil, 127k média, 204k difícil |
| OpenCodeInstruct | CC-BY-4.0 | 5M (1,4M no split lido) | py, com taxa de acerto dos testes |
| AceCode-87K | MIT | 87k | py, só pedidos e testes |
| AceCode-V2 | Apache-2.0 | 123k | py, só pedidos e testes |
| OpenCoder educational | MIT | 118k | py, pedidos fáceis com testes |

## Cobertura frente às metas

Necessidade por linguagem em cada mistura (de `docs/sft.md`; linguagens de
programação com 13%, HTML/CSS/JSON/YAML com 5,5%):

| Tarefa | Fácil | Médio | Difícil |
|---|---|---|---|
| T1 | 9,4k (markup 4,0k) | 3,1k (1,3k) | 1,6k (0,7k) |
| T2 + T5 (funções com doc) | 14k | 4,7k | 2,3k |
| T3 + T4 (instrução reversa) | 11k (markup 4,6k) | 3,6k (1,5k) | 1,8k (0,8k) |

Resultado por célula (✅ sobra, ⚠️ apertado, ❌ falta):

| Linguagem | T1 | T2 + T5 | T3 + T4 |
|---|---|---|---|
| python | ✅ público | ✅ corpus + CodeSearchNet | ✅ corpus |
| javascript | ✅ público (Glaive, OpenCoder) | ⚠️ corpus (14k fácil) → ✅ com CodeSearchNet | ✅ corpus |
| typescript | ❌ quase nada público → corpus | ✅ fácil/médio; ❌ difícil (0,4k) | ✅ corpus |
| java | ✅ público | ✅ corpus + CodeSearchNet | ✅ corpus |
| c | ⚠️ público (Glaive, OpenCoder) | ⚠️ fácil (16k); ✅ médio/difícil | ✅ corpus |
| rust | ⚠️ público (Glaive) | ✅ fácil/médio; ⚠️ difícil (1,0k) | ✅ corpus |
| html | ✅ WebSight (médio) | — | ⚠️ fácil (5,5k arquivos) |
| css | ❌ público → corpus | — | ✅ corpus |
| json | ⚠️ público (OpenCoder largescale) | — | ⚠️ fácil (6,6k arquivos) |
| yaml | ❌ público → corpus | — | ✅ corpus |

## Conclusões

1. **O corpus cobre quase tudo.** Funções e arquivos sobram em todas as
   linguagens. A instrução reversa (T1, T3, T4) pode partir dele em qualquer
   célula, e os datasets públicos ficam como complemento.
2. **O T1 público é quase só Python.** TypeScript, CSS e YAML praticamente não
   existem; Rust, C e JSON aparecem em pouca quantidade, quase tudo no Glaive e
   no OpenCoder stage 1 (gerados por LLM, qualidade a conferir). Nessas
   linguagens, o T1 vem da instrução reversa sobre o corpus.
3. **Quase nada público é difícil.** Os datasets de instrução têm respostas
   curtas: mais de 97% cabem em 2.048 tokens e quase não há código com mais de
   1.024 tokens. O T1 difícil vem da instrução reversa sobre **arquivos
   inteiros** do corpus (há milhares por linguagem).
4. **T2/T5 difícil falta em TypeScript e Rust:** poucas funções longas têm
   documentação. Opção: o Haiku escreve a documentação de funções longas sem
   doc (há 2,8k em TypeScript e 6,1k em Rust), marcada como `fonte=sintética`.
5. **HTML e JSON fáceis estão apertados**, porque os arquivos reais costumam
   ser grandes. Opção: cortar arquivos em fragmentos (um componente HTML, um
   objeto JSON aninhado) para formar unidades fáceis.
6. **Licenças a evitar ou decidir:** KodCode (não comercial), Rosetta Code
   (GFDL) e McEval (CC-BY-SA, compartilhamento pela mesma licença). Os outros
   são MIT, Apache, CC-BY ou ODC-BY. O CodeSearchNet herda a licença de cada
   repositório, como o the-stack.
7. **Duplicatas entre fontes:** o OpenCoder evol é o Magicoder Evol, e várias
   fontes partem de sementes do the-stack. Deduplicar entre fontes e contra os
   benchmarks antes de montar as misturas.

## Limitações do levantamento

- Os públicos foram estimados pelas primeiras 3.000 linhas; datasets ordenados
  (Rosetta Code, por tarefa) ficam enviesados.
- A linguagem vem da cerca de Markdown da resposta (ou do campo do dataset);
  respostas sem cerca e que não são Python, JSON ou HTML óbvios contam como
  "outra".
- O idioma dos pedidos não foi medido: os públicos são praticamente todos em
  inglês.
