# Adição do bash

Plano para incluir o bash como 11ª linguagem. **Decisão:** a adição será feita
recomeçando do zero (tokenizador, corpus, pré-treino e SFT), sem estender o modelo
atual.

**Estado (2026-11-10):** o bash entrou no pré-treino v2 (`docs/dados-v2.md`). Já
implementados: a entrada em `codelm/languages.py` (fonte: o Shell do Stack-Edu, não o
the-stack-dedup), o lexer e a indentação, os filtros, o corpus pequeno e os tokens fixos.
O tokenizador v2 está sendo treinado. Faltam o corpus de pré-treino e o SFT.

## Por que bash

- Comandos curtos cabem bem num modelo de 42M.
- A execução em sandbox dá uma recompensa verificável para o RL.

## O que muda ao recomeçar

| Etapa | Mudança |
|---|---|
| `codelm/languages.py` | Nova entrada `bash` (`.sh`, `.bash`), com `source="the-stack-dedup"`. Isso também cria `<|lang:bash|>`. |
| Tokens fixos e tokenizador | Refazer com `train.fixed_tokens` e `train.train_tokenizer`, com o bash no treino. Hoje o vocabulário tem 16.384 tokens e nenhum reservado. |
| Lexer e indentação | Testar a ida e volta em arquivos reais, com atenção a heredocs, `#` e continuação de linha com `\`. |
| Corpus de pré-treino | Dar um peso ao bash em `CODE_MIX` (`dataprep/pretrain.py`) e refazer `download`, `pack` e `check`. |
| Pré-treino | Refazer do zero. |
| `sft_candidates.py` | Incluir bash em `FUNCTION_NODES` e nas linguagens com funções. Apagar `chars_per_token.json`, que é um cache. |
| `sft_mix.py` | Refazer os pesos das linguagens e as cotas. |
| Sampler | Não há gramática de bash. Começar sem a trava e validar depois com `bash -n` e `shellcheck`. Uma gramática Lark pode vir depois, se valer o esforço. |

Conferir antes de começar:

- Se o the-stack-dedup tem shell e como a pasta se chama. Isso veio de memória,
  não foi confirmado.
- Se T2 e T5 valem para bash, que não tem padrão forte de docstring.

## Dados de SFT

- **T1, T3 e T4:** instrução reversa sobre scripts e funções do corpus, como nas
  outras linguagens (`dataprep/sft_reverse.py`).
- **Complemento de T1:** bases públicas de pedido → comando, com os pedidos
  traduzidos para os 6 idiomas.

Candidatas. Tamanhos tirados de páginas e cards, sem abrir os dados; qualidade,
licença e duplicatas entre elas ainda precisam ser conferidas.

| Base | Tamanho | Observação |
|---|---|---|
| [NL2Bash](https://arxiv.org/pdf/1802.08979) | ~9,3 mil pares | Descrições em inglês escritas por especialistas |
| NL2SH-ALFA (`westenfelder/NL2SH-ALFA`) | ~40 mil pares | Teste de 300 exemplos, útil como benchmark |
| [CLI-1M](https://discuss.huggingface.co/t/dataset-cli-1m-975k-nl-shell-pairs-13-languages-6-shells-apache-2-0/176022) (`carosh/cli-1m`) | ~976 mil pares | Apache-2.0, 6 shells e 13 idiomas, sem verificação independente; filtrar só bash |
| [NL2CMD](https://arxiv.org/pdf/2302.07845) | Mais de 6× as bases anteriores | Gerada automaticamente |
| [NL2Bash-PT](https://huggingface.co/datasets/guicybercode/nl2bash-pt) | ~24,7 mil pares | Em português, comunitária, derivada de NL2Bash e NL2SH-ALFA |
| [Bash Command 6K](https://huggingface.co/datasets/emirkaanozdemr/bash_command_data_6K/blob/main/README.md) | 6 mil pares | Apache-2.0 |

## Avaliação

- Teste do NL2SH-ALFA, fora do treino e deduplicado contra ele.
- Proporção de comandos que passam em `bash -n` e `shellcheck`.
- Para o RL: executar os comandos em sandbox isolada, sem rede e com sistema de
  arquivos descartável.
