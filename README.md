# code-language-model

Um modelo de linguagem pequeno, treinado do zero e especializado em programação,
feito para rodar quantizado na CPU.

# Sobre o codelm

O codelm é 100% especializado em código em todas as etapas: tokenização, formato
de prompt, geração e treino. A partir de um pedido (e, opcionalmente, da
linguagem e versão), ele gera código; a partir de um código, gera uma descrição
em linguagem natural ou estruturada (JSON).

- **Linguagens:** Python, JavaScript, TypeScript, Java, C, Rust, HTML, CSS, JSON e YAML.
- **Idiomas dos pedidos e descrições:** inglês, espanhol, português, francês, alemão e italiano.
- **Alvo de execução:** CPU, via llama.cpp (GGUF), com pesos em int4.

O caminho até aqui:

1. **Dados:** amostras do The Stack (~10 mil arquivos por linguagem), filtradas por
   qualidade, deduplicadas, padronizadas (quebras de linha, tabs, recuos) e
   divididas em treino/validação por repositório. Mais texto natural do FineWeb
   nos 6 idiomas.
2. **Tokenizador:** BPE de 16k tokens, com parte do vocabulário fixada a partir das
   gramáticas das linguagens, indentação como tokens e pontuação alinhada à
   gramática.
3. **Chat template:** o modelo responde com a linguagem/versão e o tipo da
   resposta (código, descrição ou JSON).
4. **Sampler:** geração restrita por gramática, que garante código e JSON
   sintaticamente válidos.
5. **Arquitetura:** decoder estilo Llama de 42M parâmetros, dimensionado pela
   velocidade de geração na CPU.

Próximas etapas: pré-treino, ajuste fino nas tarefas, RL com a execução do código
como recompensa e QAT (treino ciente da quantização) para o modelo final em int4.

# Pastas

```
src/
├── codelm/         definição do modelo (tudo que é usado em inferência)
│   ├── languages.py      linguagens e idiomas suportados
│   ├── tokenizer.py      tokenizador: tokens de controle, pré-tokenização, codificação
│   ├── indentation.py    indentação relativa (<|indent|>/<|dedent|>)
│   ├── lexer.py          lexer de contexto (código, string ou comentário)
│   ├── chat_template.py  formato do prompt e da resposta
│   ├── grammar.py        máscara guiada por gramática (llguidance)
│   ├── grammars/         gramáticas Lark (python.lark, json.lark)
│   ├── sampler.py        amostragem e restrições da resposta
│   └── model.py          arquitetura e tamanhos do modelo
├── dataprep/       download e preparação dos dados
└── train/          tudo que é treino ou avaliação (tokenizador, modelo, exportação)
assets/tokenizer/   tokenizador treinado e lista de tokens fixos (versionados)
notebooks/          exploração e exemplos (ex.: gramatica_sampler.ipynb)
data/               dados baixados e gerados (fora do git)
third_party/        llama.cpp compilado localmente (fora do git)
```

# Codelm

## Tokenizer

BPE byte-level de **16.384 tokens**, treinado com ~70% de código e ~30% de texto
natural. Vocabulário pequeno de propósito: menos parâmetros de embedding e mais
velocidade na CPU.

- **Vocabulário fixo:** palavras-chave e operadores extraídos automaticamente das
  gramáticas tree-sitter de cada linguagem, mais tags/propriedades do MDN (HTML e
  CSS) e nomes embutidos (filtrados por frequência no corpus). Todos garantidos
  como um único token (merges extras acrescentados no fim do BPE, sem alterar a
  tokenização do resto).
- **Pré-tokenização para código:** identificadores inteiros, um dígito por token e
  pontuação cortada por operador da gramática (`):` vira `)` `:`; `===`, `->`,
  `::` ficam inteiros). Cada token de pontuação corresponde a um terminal da
  gramática.
- **Indentação relativa:** o recuo vira `<|indent|>` (+2 espaços) e
  `<|dedent|>` (−2), só onde o nível muda; o modelo não precisa contar espaços.
  Linhas de continuação, strings de várias linhas e comentários mantêm o recuo
  literal, como no tokenizador do próprio Python. Sem perda:
  `decode(encode(código)) == código` em 100% da validação.
- **Tokens de controle:** linguagem (`<|lang:python|>`), versão, prompt,
  resposta, tipo de resposta (`<|code|>`, `<|description|>`, `<|json_output|>`)
  e fill-in-the-middle. Texto do usuário nunca vira token de controle.

Resultado: comprime código quase como o tokenizador do StarCoder2 (49k tokens)
com um terço do vocabulário, e comprime melhor que ele e que o GPT-2 nos 5
idiomas além do inglês.

Formato de uma resposta (o cabeçalho é gerado pelo modelo ou preenchido se o
usuário informar a linguagem):

```
<|prompt|>Escreva uma função que retorna os n primeiros primos.
<|response|><|lang:python|><|version|>3.12
<|code|>def primos(n):
<|indent|><|indent|>...<|dedent|><|dedent|><|endoftext|>
```

## Arquitetura

Decoder estilo Llama do `transformers` (RMSNorm, SwiGLU, RoPE, atenção com
poucas cabeças de K/V, sem bias, embeddings de entrada e saída amarrados),
escolhido para exportar para o llama.cpp, que tem os kernels de CPU mais rápidos
para int4 e suporta o llguidance.

O tamanho foi escolhido pela **inferência na CPU**: cada token gerado lê todos os
pesos da memória, então a velocidade é limitada por banda ÷ tamanho dos pesos.

| tamanho | parâmetros | GGUF Q4_0 | geração* | leitura do prompt* |
|---|---|---|---|---|
| tiny | 19M | 13 MB | ~1.480 tok/s | ~17.800 tok/s |
| **small** (padrão) | **42M** | **27 MB** | **~840 tok/s** | ~7.500 tok/s |
| base | 113M | 67 MB | ~440 tok/s | ~4.000 tok/s |

\* Core Ultra 5 245K, 6 threads, llama.cpp.

- Dimensões múltiplas de 64, compatíveis com os grupos de 32 pesos do int4 (`Q4_0`).
- **QAT** com torchao: pesos int4 em grupos de 32 e ativações int8 simulados no
  treino, nas camadas dos blocos (embeddings em 8 bits). Planejado para o fim do
  pré-treino, o ajuste fino e o RL, para que o modelo otimizado seja o mesmo que
  roda na CPU.
- Treino numa Intel Arc B580 (PyTorch `xpu`, bf16, `torch.compile`): ~44 mil
  tokens/s no small, ~1 bilhão de tokens em ~6,5 h.

## Sampler

Amostragem comum (temperatura, top-k, top-p) restrita em duas camadas:

1. **Formato da resposta:** primeiro só tokens de linguagem, depois versão ou
   quebra de linha, depois o tipo da resposta.
2. **Gramática do conteúdo:** em `<|code|>` (com gramática da linguagem) e em
   `<|json_output|>`, só são permitidos tokens que mantêm o texto sintaticamente
   válido. As gramáticas (`codelm/grammars/*.lark`) são escritas sobre a
   representação do modelo, com `<|indent|>`/`<|dedent|>` como terminais, e
   aplicadas pelo [llguidance](https://github.com/guidance-ai/llguidance) em
   ~100 µs por token. Em descrições em linguagem natural não há restrição.

Gramáticas prontas: **Python** e **JSON**, com cobertura de 100% em arquivos
válidos reais. Com a trava ligada, até um "modelo" de notas aleatórias só gera
Python válido (ver `notebooks/gramatica_sampler.ipynb`).

# Usage

Instalação:

```bash
uv sync
```

O PyTorch vem do índice `xpu` (GPUs Intel; ver `pyproject.toml`), fixado na 2.8
porque o driver LTS da Intel instalado (compute-runtime 25.18) não roda versões
mais novas.

Dados e tokenizador:

```bash
uv run python -m dataprep.download             # código (data/processed/)
uv run python -m dataprep.natural_text         # texto natural en/es/pt/fr/de/it (data/processed/natural/)
uv run python -m train.fixed_tokens            # tokens fixos (assets/tokenizer/fixed_tokens.json)
uv run python -m train.train_tokenizer         # tokenizador BPE (assets/tokenizer/tokenizer.json)
uv run python -m train.evaluate_tokenizer      # métricas de compressão
uv run python -m train.evaluate_grammars       # cobertura das gramáticas
```

Modelo e exportação:

```bash
uv run python -m train.smoke_test --batch 4 --compile    # passos de treino na GPU (bf16 e QAT)
uv run python -m train.export_gguf --random --size small # GGUF Q4_0/Q8_0 em data/export/
```

llama.cpp (exportação e execução na CPU), compilado em `third_party/`:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp third_party/llama.cpp
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build third_party/llama.cpp/build -j --target llama-bench llama-quantize llama-cli
```

No código:

```python
from codelm.tokenizer import load_tokenizer
from codelm.chat_template import encode_prompt, decode_response
from codelm.sampler import generate

tokenizer = load_tokenizer()
prompt = encode_prompt(tokenizer, "Escreva uma função de primos.", lang="python")
ids = generate(next_logits, tokenizer, prompt, temperature=0.8, top_p=0.95)  # next_logits: o modelo
response = decode_response(tokenizer, ids)  # Response(lang, version, kind, content)
```
