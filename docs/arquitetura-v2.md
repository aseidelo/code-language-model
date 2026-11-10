# Arquitetura do v2: atenção local/global alternada com contexto de 8k

Decidido em 2026-11-09. O pré-treino v2 começa do zero, então é a hora de mudar a
arquitetura: depois dele, ela fica congelada para o SFT e o RLVR.

## Motivação

O contexto de 2048 do v1 corta a maior parte do código. Medido no `train` do base-v1:

| | mediana (tokens) | p90 | % dos tokens em docs > 2k | % em docs > 8k |
|---|---|---|---|---|
| python | 627 | 3.249 | 65% | 23% |
| c | 877 | 6.690 | 85% | 55% |
| rust | 823 | 4.381 | 72% | 29% |
| inglês | 806 | 2.554 | 50% | 18% |
| total | 486 | 2.752 | 64% | 30% |

O documento típico é curto, mas 64% dos tokens estão em documentos maiores que 2048.
O modelo vê boa parte do código sem o início do arquivo: imports, definição da
classe, funções auxiliares.

Atenção completa em 8k aumentaria o custo por token em ~55%. Com atenção local/global
alternada, o aumento é de ~7%, e o cache de K/V na inferência fica ~3,5× menor. Isso
importa porque o modelo roda na CPU via GGUF.

O ganho esperado é no uso real, como completar código num arquivo grande ou editar
com o resto do arquivo à vista. **No HumanEval e no MBPP não se espera ganho**: os
prompts têm menos de 300 tokens.

## Desenho

**Base: `Gemma3ForCausalLM` do `transformers`, só texto, no lugar da `LlamaForCausalLM`.**

- A alternância local/global já vem pronta: `layer_types`, `sliding_window` e um RoPE
  para cada tipo de camada.
- O llama.cpp suporta atenção local na arquitetura Gemma 2/3. Na arquitetura Llama, só
  como caso especial do MobileLLM. Uma arquitetura própria quebraria a exportação para
  GGUF.
- Ela traz junto QK-norm e normalização antes e depois de cada bloco, que ajudam a
  estabilidade.
- **Aceitamos o pacote do Gemma 3 como ele é**: MLP com GELU-tanh (no lugar do SwiGLU),
  embeddings escalados por √d e `query_pre_attn_scalar`. O grafo do llama.cpp fixa
  esses detalhes; qualquer desvio quebra a exportação. O número de parâmetros não muda,
  porque o GeGLU tem as mesmas três matrizes que o SwiGLU.

**Configuração, mantendo as dimensões atuais:**

| | v1 (Llama) | v2 (Gemma 3) |
|---|---|---|
| camadas do `base` | 16 globais | 12 locais + 4 globais |
| padrão | — | 3 locais : 1 global; a última camada é global |
| janela local | — | 1024 |
| contexto | 2048 | 8192, desde o início do treino |
| RoPE | θ = 100k | local θ = 10k, global θ = 1M |
| custo por token (`base`) | 0,83 GFLOP | ~0,89 GFLOP (+7%) |
| cache de K/V em 8k | 16 × 8k | 4 × 8k + 12 × 1k (~3,5× menor) |

Os outros tamanhos seguem o mesmo padrão 3:1: `tiny` com 6 locais e 2 globais, `small`
com 9 locais e 3 globais.

Ficou decidido **3:1 em vez do 5:1 do Gemma 3**: com só 16 camadas, 5:1 deixaria 2 ou 3
camadas globais, poucas demais para levar informação de longe. Também ficou decidido
**8k desde o início** em vez de 2k → 8k no fim: com a atenção local o custo é baixo, e
o treino fica mais simples (uma fase só, sem trocar o RoPE no meio).

**Em aberto:** com blocos de 8k, mais documentos dividem a mesma sequência. Hoje a
atenção atravessa a fronteira entre documentos. Pode valer usar uma máscara por
documento, mas isso fica para outro A/B, não entra neste.

## Validação, em ordem

### 1. Exportação para GGUF (antes de qualquer treino)

Com um modelo de pesos aleatórios, como o `export_gguf --random` que já existe:
Gemma 3 → GGUF → Q4_0 → inferência no llama.cpp com o nosso tokenizer.

**Passa se** os logits do llama.cpp em f16 batem com os do PyTorch numa sequência maior
que a janela local (por exemplo, 3000 tokens), para exercitar as camadas locais e as
globais. Se não passar, o resto do plano não importa.

Na mesma etapa, conferir que o QAT (int4 em grupos de 32) funciona na arquitetura nova.

**Resultado (2026-11-09): passou.**

- `codelm.model` ganhou a arquitetura `gemma3`: `build_model(size, arch)`. A `llama`
  continua o padrão. Os tamanhos ficam iguais (`small` com 42,2M, `base` com 113,3M).
- `train.export_gguf` exporta a Gemma 3 (`--arch gemma3`). O llama.cpp assume o padrão
  5:1 do Gemma original (`load_swa_pattern(ml, 6)`). Por isso gravamos no GGUF o tipo de
  cada camada (`sliding_window_pattern`) e a base do RoPE local.
- Teste com o `small` aleatório, sequência de 3000 tokens. Para a janela fazer diferença,
  os pesos foram sorteados com variância maior, o que deixa a atenção concentrada. A
  comparação usa a diferença máxima dos logits, dividida pelo desvio dos logits:

  | comparação com o PyTorch | pos. 0–512 | pos. 512–1024 | pos. 1024–3000 |
  |---|---|---|---|
  | llama.cpp f16 | 0,008 (99,6%) | 0,009 (99,8%) | 0,008 (99,8%) |
  | controle: PyTorch com janela 512 | 0 (100%) | 4,4 (55%) | 5,6 (37%) |
  | controle: PyTorch com janela 1023 / 1025 | — | — | 0,59 / 1,22 |

  Entre parênteses, a fração de posições com o mesmo argmax. O llama.cpp reproduz a
  janela exata; um erro de um token já daria 0,59.
- QAT: troca os 84 `nn.Linear` dos blocos do `small` e reproduz o GGUF Q4_0 com
  diferença de 0,27 (argmax igual em 95%), contra 1,97 (64%) sem QAT. No `base-v1`
  treinado (Llama, texto real), a mesma medida dá 0,14 (99,1%) com QAT e 1,10 (95,6%) sem.
  A sobra já existia no v1; hipótese, ainda não testada: a matriz de embeddings
  compartilhada com a saída fica em Q8_0 no GGUF e fora do QAT. Medir de novo no modelo
  treinado do A/B.

**Treino em 8k não cabe na B580.** Na XPU, o PyTorch 2.8 não tem kernel de atenção
eficiente em memória: o SDPA monta a matriz 8192 × 8192 inteira. Até a Llama em 8k
estoura os 12 GB com uma sequência só, e a Gemma 3 derruba o driver (`DEVICE_LOST`). Em
2k e 4k a Gemma 3 treina normalmente na B580. O treino em 8k fica no Modal, em CUDA.

### 2. A/B no Modal (H100)

Implementação:

- `train.pretrain --arch {llama,gemma3}` usa o contexto da arquitetura.
- A validação usa sempre blocos de 8192, e a Llama é avaliada com janela deslizante
  (cada token vê de 1024 a 2047 tokens anteriores). Além das perdas por tipo, sai a
  perda por **contexto disponível**, isto é, quantos tokens do mesmo documento vêm antes
  dele dentro do bloco: `valid ctx 0k-2k`, `2k-4k` e `4k-8k`, também só para código.
- `train.modal_pretrain` roda o `train.pretrain` numa H100, com os dados e os
  checkpoints num volume do Modal. Se a GPU for interrompida, retoma de `latest/`.

Modelo `small` (42M), ~1B tokens de cada lado, com os dados do base-v1.

- **(A)** Llama com contexto de 2k, a arquitetura atual.
- **(B)** Gemma 3 local/global com contexto de 8k.
- **(C) opcional**: Gemma 3 só com camadas globais e contexto de 2k. Separa o efeito da
  troca de arquitetura do efeito da atenção local com contexto longo. Custa ~US$ 1.

**Métricas:**

- perda de validação nos tokens das posições 0–2047 de cada documento (as duas
  arquiteturas veem o mesmo contexto);
- perda nos tokens em **posições acima de 2047** de documentos longos. O (A) é
  avaliado com janela deslizante de 2k: cada token vê os 2047 anteriores. O (B) vê até
  8k;
- HumanEval e MBPP no modo completar (`eval.passk`).

**Critério:** o (B) passa se ficar melhor na perda acima de 2047 e não ficar pior que o
(A), além do ruído, na perda até 2047 e no HumanEval + MBPP (2 erros padrão, como em
`docs/etapas.md`).

**Custo:** ~US$ 2–4 no total, nos preços do Modal para a H100 (US$ 0,0011/s).

## Fora deste experimento

Uma mudança de arquitetura por vez, cada uma com seu A/B. Ideias para os próximos:

- máscara de atenção por documento;
- retreinar o tokenizer só com inglês e código (o atual ainda tem vocabulário de es,
  pt, fr, de e it);
- tamanho do modelo maior que 113M, se a meta do phi-1-small pedir.

### Resultado do A/B (2026-11-10)

`small` (42M), 1B tokens cada, H100 no Modal. Perda de validação no mesmo conjunto
(256 blocos de 8192), passo final 3814:

| | Llama 2k | Gemma 3 8k | diferença |
|---|---|---|---|
| geral | 1,261 | 1,253 | −0,008 |
| contexto 0–2k | 1,346 | 1,357 | **+0,011** |
| contexto 2k–4k | 1,113 | 1,076 | **−0,037** |
| contexto 4k–8k | 0,993 | 0,918 | **−0,075** |
| python | 1,242 | 1,223 | −0,019 |
| json | 0,789 | 0,753 | −0,036 |

- A Gemma 3 começa pior (+0,04 nos passos 500–1000) e passa a Llama por volta do passo
  1250. Depois disso, a vantagem nas faixas acima de 2k é estável em todas as avaliações
  (−0,02 a −0,04 em 2k–4k; −0,04 a −0,08 em 4k–8k).
- Na faixa de 0–2k ela fica sempre um pouco pior (+0,010 a +0,026). Pode ser custo das
  camadas locais ou da troca de arquitetura (GELU, normas): o (C) separaria os dois.
  Não há repetição com outra semente para medir o ruído entre rodadas.
- Velocidade: Llama ~370k tokens/s, Gemma 3 ~119k tokens/s. Pela conta de operações a
  Gemma 3 deveria custar só ~9% a mais por token. Resolvido depois com o FlashAttention 2
  (ver abaixo).
- Custo: ~US$ 13 nas duas rodadas.

### Velocidade: FlashAttention 2 (2026-11-10)

Diagnóstico com `train.attention_check` na H100 (`small`, 64k tokens por passada):

- O gargalo **não** era a máscara da janela nem a arquitetura. Em 2k, a Gemma 3 é só 4%
  mais lenta que a Llama. Em 8k, até a Llama cai de 372k para 122k tokens/s.
- O `sdpa` do PyTorch 2.8 cai no kernel "memory-efficient" da A100
  (`fmha_cutlass..._sm80`, 89% do tempo), que calcula a atenção inteira e não aproveita a
  janela local.
- `flex_attention` (do `transformers` ou com a máscara em blocos calculada uma vez só): o
  resultado é certo, mas fica 4x **mais lento** que o `sdpa`.
- FlashAttention do Hub (pacote `kernels`): sem build para torch 2.8, só para 2.9 ou mais.
- **FlashAttention 2** (pacote `flash-attn` 2.8.3, wheel oficial para torch 2.8 + CUDA
  12, na imagem do Modal): mesmo resultado que o `sdpa` (diferença do nível do bf16:
  0,06 desvio nos logits, 1,4% no gradiente; o controle de janela errada dá 6,4) e janela
  local nativa (backward de uma camada local: 1,4 ms; de uma global: 4,7 ms).

| tokens/s (H100, `small`) | `sdpa` | FA2 |
|---|---|---|
| Llama 2k | 372k | 686k |
| Gemma 3 8k só global | 120k | 389k |
| Gemma 3 8k local/global | 120k | 499k |
| Gemma 3 8k local/global, treino real (`use_cache=False`) | — | **595k** (565k com QAT) |

`train.pretrain` usa o FA2 em CUDA quando o pacote existe (`--attention auto`) e o `sdpa`
na B580. O cache de K/V fica desligado no treino: com o FA2, ele fazia o `torch.compile`
recompilar até o limite.

### Perda nas soluções do HumanEval e do MBPP (2026-11-10)

`eval.solution_loss`: perda média nos tokens da solução oficial, dado o enunciado, no
formato do pré-treino, com QAT. São 659 problemas: ficam de fora 4 do HumanEval, em que a
tokenização do enunciado não é prefixo da do programa, e o `mbpp-493`, com 4258 tokens,
maior que o contexto da Llama. Erro padrão pareado por problema. Ele não inclui a variação
entre rodadas com sementes diferentes, que não foi medida.

| | Llama 2k | Gemma 3 8k | diferença |
|---|---|---|---|
| sem contexto anterior | 1,282 | 1,309 | **+0,027 ± 0,004** (Gemma 3 pior em ~63% dos problemas) |
| com 813 tokens de outros documentos antes | 1,273 | 1,295 | +0,022 ± 0,004 |

- Todos os problemas têm menos de 1024 tokens: a janela local não interfere.
- **A Gemma 3 fica pior no tipo de tarefa do benchmark**, mais do que na faixa de 0–2k da
  validação (+0,011).
- Hipótese da posição de início (em blocos de 8k a Gemma 3 vê menos documentos começando
  do zero): explica só ~20% da diferença. Com contexto antes, a Gemma 3 ganha 0,015 e a
  Llama 0,009, mas sobram +0,022.
- O resto deve vir da troca de arquitetura (GELU, normas, RoPE) ou do treino em blocos de
  8k. O experimento (C), Gemma 3 só global em 2k, separaria os dois.

### Decisão (2026-11-10)

O v2 segue com a **Gemma 3 local/global em 8k**, aceitando a pequena perda em código
curto (+0,027 na perda das soluções) pelo ganho em arquivos longos (−0,075 na faixa de
4k–8k). O experimento (C) e a segunda semente ficam para depois, se for preciso. O próximo
passo é a expansão dos dados (`docs/dados-v2.md`).
