# Dados do pré-treino v2

Desenho de 2026-11-10. A arquitetura já está decidida (`docs/arquitetura-v2.md`): Gemma 3
local/global, contexto de 8k, tamanho `base` (113M), treino no Modal (H100,
FlashAttention 2).

**Decidido com o usuário:** 10B tokens; Python com 35%; inglês como língua natural
dominante, com es/pt/fr/de/it numa fatia pequena; **Bash como 11ª linguagem**; código de qualidade do Stack-Edu, baixado direto (opção A);
todas as licenças do Stack-Edu (vale a licença do dataset).

## Por que mudar

- O base-v1 viu ~1,1B tokens, cerca de 10 por parâmetro, e as curvas mostram que ele
  ainda aprendia quando o treino acabou (perda de Python caindo ~0,11 nat por dobra). O
  gargalo é dado, não tamanho do modelo.
- No teste de linguagens não vistas, ele acertou a sintaxe e errou a lógica até em
  Python.
- Gerar dados nos outros idiomas é difícil (um problema do SFT), e o texto em
  es/pt/fr/de/it ocupava 9% do treino. **Decidido: o inglês passa a dominar; os outros
  idiomas ficam numa fatia pequena**, porque o objetivo do projeto continua multilíngue
  nos pedidos (`docs/etapas.md`) e o texto do FineWeb2-HQ já está baixado.
- O tempo de GPU deixou de ser o limite. Na H100, o `base` deve fazer ~200–250k tokens/s
  (estimativa, a medir): 5B tokens em ~6 h, ~US$ 25. **O limite passa a ser a qualidade
  por token.**

## Proposta

### Orçamento: 10B tokens (decidido)

É ~88 tokens por parâmetro, bem acima dos ~20 do Chinchilla. É normal para um modelo
pequeno que vai rodar muito na inferência. Custo estimado: ~12 h e ~US$ 50 na H100, a
confirmar com um teste curto do `base`. Passa do crédito restante (~US$ 14).

### Mistura (decidida: Python 35% + Bash)

| | v1 | v2 | tokens v2 | v1 → v2 | fonte |
|---|---|---|---|---|---|
| python | 21% | 35% | 3,5B | 15× | Stack-Edu |
| javascript | 10% | 8% | 800M | 7,5× | Stack-Edu |
| typescript | 11% | 8% | 800M | 6,5× | Stack-Edu |
| java | 11% | 8% | 800M | 6,6× | Stack-Edu |
| c | 9% | 7% | 700M | 7,3× | Stack-Edu |
| rust | 9% | 6% | 600M | 6,3× | Stack-Edu |
| **bash** | — | 3% | 300M | novo | Stack-Edu (Shell) |
| html | 5% | 4% | 400M | 6,7× | the-stack-dedup |
| css | 4% | 3% | 300M | 6,3× | the-stack-dedup |
| json | 4% | 3% | 300M | 6,7× | the-stack-dedup |
| yaml | 5% | 4% | 400M | 7,1× | the-stack-dedup |
| inglês | 2% | 9% | 900M | 45× | FineWeb-Edu |
| es, pt, fr, de, it | 9% | 2% (0,4% cada) | 40M cada | 2× | FineWeb2-HQ |

- Todas as linguagens de programação crescem pelo menos 6× em tokens absolutos. Cada
  idioma dobra.
- Python sobe para 35% porque o HumanEval, o MBPP e o SFT são em Python.
- O inglês sobe para 9%. Ele alimenta docstrings, comentários e o entendimento dos
  pedidos no SFT. Os outros idiomas ficam em 2% no total (40M tokens cada, o dobro do
  v1), para manter os pedidos multilíngues.
- O peso do Bash (3%) é uma proposta minha: ~300M tokens, ~10% do Shell do Stack-Edu.

### Filtro de qualidade: Stack-Edu

O Stack-Edu (HuggingFaceTB, base do SmolLM2) é o The Stack v2 filtrado por valor
didático. Um classificador por linguagem (sobre o StarEncoder, treinado com notas do
Llama 3 70B) dá nota de 1 a 5, e ficam os arquivos com nota ≥ 3 (≥ 2 em Java). Nos
modelos pequenos, esse tipo de filtro é o que mais rende no HumanEval (phi-1, SmolLM2).

Ele cobre 7 das 11 linguagens, com folga para a mistura (TypeScript e Rust usam ~1/4 e
~1/3 do disponível):

| | Stack-Edu | proposta v2 |
|---|---|---|
| python | 21,8B | 3,5B |
| java | 42,1B | 800M |
| javascript | 11,1B | 800M |
| c | 11,1B | 700M |
| typescript | 3,0B | 800M |
| rust | 1,75B | 600M |
| shell (bash) | 3,1B | 300M |

(tokens do tokenizer do SmolLM2; os nossos devem dar números da mesma ordem.)

**HTML, CSS, JSON e YAML não estão no Stack-Edu.** Eles continuam vindo do
the-stack-dedup, com os filtros heurísticos atuais (`dataprep.download.clean_file`).

**Como obter o Stack-Edu (decidido: A).** O dataset traz só os IDs dos arquivos; o
conteúdo vem do bucket S3 `softwareheritage` (`content/{blob_id}`, gzip). **Não
precisa de conta da AWS:** em 2026-11-10 um arquivo do Shell baixou por HTTPS anônimo
(`https://softwareheritage.s3.amazonaws.com/content/{blob_id}`, resposta 200, gzip). Cada
arquivo é uma requisição: para 10B tokens são alguns milhões, com download em paralelo.
Os metadados (blob_id, caminho, nota, licença) vêm do dataset no Hub. Como a mistura usa só uma fração de cada linguagem, basta baixar
um subconjunto aleatório de IDs, sem o dataset inteiro.

**Licenças (decidido):** todas. Vale a licença do dataset (os termos do The Stack v2),
pela regra do projeto. A coluna `license_type` (`permissive` ou `no_license`) é guardada
para o caso de precisar filtrar depois.

**Bash:** o Shell do Stack-Edu inclui outras coisas. Nas 3 primeiras linhas já
apareceram um `PKGBUILD` e um arquivo sem extensão. É preciso filtrar por extensão
(`.sh`, `.bash`) ou pela linha `#!` (bash/sh). O plano do `docs/bash.md` vale aqui: lexer e
indentação testados com heredocs, `#` e continuação com `\`; sem gramática no sampler
por enquanto.

### Texto natural: FineWeb-Edu e FineWeb2-HQ

- Inglês: o v1 usou um só arquivo do FineWeb-Edu (20M tokens). Para 900M são vários
  arquivos, e `dataprep.natural_text` precisa ler mais de uma fonte.
- es, pt, fr, de, it: o FineWeb2-HQ, como no v1, com 40M tokens cada. O que já está
  baixado (~20M cada) cobre metade e precisa ser completado.

### Tokenizer: refeito (consequência do Bash)

O Bash precisa de `<|lang:bash|>`, e o vocabulário atual (16.384 tokens) não tem nenhum
reservado. Pelo `docs/bash.md`, a adição do Bash é feita recomeçando do zero: tokens
fixos (`train.fixed_tokens`), tokenizer (`train.train_tokenizer`), corpus, pré-treino e
SFT. Os dados do SFT precisam ser tokenizados de novo.

**Decidido:** o tokenizer novo é treinado com os mesmos 6 idiomas e as 11 linguagens,
com o mesmo tamanho de vocabulário (16.384). Mudanças em relação ao v1:

- o bash, com corpus de 9.790 arquivos do Shell do Stack-Edu (17,7M caracteres, contra
  até 40M nas outras linguagens: pesa ~metade no vocabulário);
- o inglês pesa mais: todo o inglês do corpus pequeno (34M caracteres) e 14M de cada
  outro idioma (no v1 eram 28M para todos). O texto natural cai de ~30% para ~20% do
  treino do tokenizer.

**Versionamento (implementado em 2026-11-10):** o tokenizer antigo ficou congelado em
`assets/tokenizer/v1/`; o novo vai para `assets/tokenizer/v2/`. Todo checkpoint novo
guarda o `tokenizer.json` com que foi treinado. As ferramentas que abrem um checkpoint
chamam `codelm.tokenizer.use_tokenizer_of`, que usa o dele ou o v1 se ele for antigo.
`load_tokenizer()` sem argumento dá o v2. `eval.solution_loss` mede em bits por byte, com
o tokenizer de cada checkpoint.

**Lexer do bash (implementado):** `#` só abre comentário no início de uma palavra;
`'...'` sem escape; `\` escapa o caractere seguinte; heredocs (`<<EOF`, `<<'EOF'`, vários
na mesma linha, dentro de `$(...)` e de crase), ignorando `<<` dentro de `((...))`;
continuação com `\` e linhas só de comentário mantêm o recuo literal. A ida e volta
funciona nos 9.790 arquivos, e 99,7% terminam em modo código (o resto é script quebrado).
A codificação das outras 10 linguagens ficou idêntica (mesmo hash em toda a validação).
Filtros em `dataprep.download`: shebang de outro interpretador e `<<-` (depende de tabs).

### Validação

- A validação nova sai das mesmas fontes do treino, separada por repositório, como hoje.
- **Com o tokenizer novo, a perda por token deixa de ser comparável** com o v1 e com o
  A/B: cada tokenizer quebra o texto num número diferente de tokens. Para comparar
  versões, a perda vira **bits por byte** (perda total em bits dividida pelos bytes do
  texto), na validação e no `eval.solution_loss`.
- **Os documentos da validação do v1 ficam como conjunto fixo**, retokenizados com o
  tokenizer novo, para comparar o v1, o A/B e o v2 em bits por byte no mesmo texto.
- A perda nas soluções do HumanEval e do MBPP (`eval.solution_loss`, em bits por byte) e
  o pass@k (`eval.passk`) continuam como métricas principais.

## Pipeline (a detalhar)

1. Tokenizer: `bash` em `codelm/languages.py`, tokens fixos e tokenizer refeitos.
2. Download e filtro: Stack-Edu (opção A) para 7 linguagens; the-stack-dedup para HTML,
   CSS, JSON e YAML; FineWeb-Edu para o inglês; FineWeb2-HQ para es/pt/fr/de/it. Roda
   na máquina local: é rede e CPU.
3. `pack`: o mesmo formato de hoje, com a mistura nova.
4. `check`: o mesmo de hoje.
5. Upload para o volume do Modal: ~20 GB de `train.bin` para 10B tokens.

Disco: o SSD tem ~390 GB livres. Os dados brutos e filtrados cabem com folga.

## Estado

**Passo 1 (bash e tokenizer): concluído em 2026-11-10.** Tokenizer v2 em
`assets/tokenizer/v2/`. Caracteres por token na validação, v1 → v2:

| | v1 | v2 | |
|---|---|---|---|
| python | 3,21 | 3,27 | +2,0% |
| javascript / typescript / java | 2,68 / 3,20 / 3,38 | 2,72 / 3,25 / 3,45 | +1,4 a 1,8% |
| c / rust | 2,60 / 2,89 | 2,65 / 2,95 | +1,9 a 2,1% |
| html / css / json / yaml | 2,76 / 2,41 / 2,17 / 3,02 | 2,77 / 2,41 / 2,19 / 3,10 | +0,2 a 2,5% |
| bash | 2,42 | 2,57 | +6,1% |
| inglês | 3,45 | 3,51 | +1,6% |
| es, pt, fr, de, it | 3,06–3,33 | 2,90–3,15 | −4,4 a −5,5% |

A ida e volta funciona em 100% da validação das 11 linguagens. As gramáticas do sampler
(Python e JSON) compilam com o v2. Uma mudança na pré-tokenização: os operadores que
começam com `$` (`$(`, `${`, `$((`) vêm antes dos identificadores e viram um token só.
Antes eles nunca podiam virar um token, porque o `$` abria um identificador.

**Pendente para os próximos passos:**

- o `pack` deve gravar o hash do tokenizer no `meta.json`, e o `train.pretrain` deve
  conferir que os dados batem com o tokenizer;
- a validação do pré-treino em bits por byte;
- o `train.sft` e o `train.grpo` carregam `LlamaForCausalLM` e precisam de
  `AutoModelForCausalLM` para a Gemma 3 (antes do SFT do v2);
- os notebooks chamam `load_tokenizer()`, que agora dá o v2; para abrir modelos antigos
  eles precisam de `use_tokenizer_of(checkpoint)`.
