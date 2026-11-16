# Inferência de Complexidade Enxadrística via Redes Neurais e Busca em Árvore

## Informações Gerais

| Item | Informação |
| :--- | :--- |
| **Curso** | Engenharia de Computação |
| **Instituição** | PUCPR — Escola Politécnica |
| **Ano / Semestre** | 2026/2 |
| **Disciplina** | IOPT (Implementação e Operação de Projeto Transformador) |
| **Estudante(s)** | Lucas Franco de Mello |
| **Orientador(a)** | Prof. Dr. Júlio Cesar Nievola |
| **Coorientador(a)** | Não há |
| **Área Técnica** | Inteligência Artificial |

---

## Resumo Técnico do Software

O `tfird` é um motor de xadrez com rede neural ResNet *Dual-Head* e busca em árvore de Monte Carlo (MCTS), desenvolvido como prova de conceito para caracterizar a resposta da busca por descritores e avaliar se esses descritores fornecem evidências sobre características associadas à complexidade enxadrística.

Além do melhor lance e da avaliação em *centipawns*, o motor devolve o **perfil instrumental** da busca: visitas, *prior*, valor médio e avaliação inicial da rede de todos os lances legais em cada nó da variante principal.

O projeto não possui firmware nem interface gráfica: o motor segue o protocolo UCI.

| Módulo do relatório | Código |
| :--- | :--- |
| 1. *Pipeline* de Dados | `src/train/miner/`, `src/train/split_dataset.py` |
| 2. Rede Neural | `src/train/train.py`, `decoder/`, `export_onnx.py`, `convert_fp16.py` |
| 3. Validação da Rede Neural | `src/train/benchmark/` |
| 4. Busca MCTS Adaptada | `src/engine/src/search.cpp`, `search_metrics.cpp` |
| 5. Inferência e API | `src/engine/src/inference_*`, `uci.cpp`, `main.cpp` |
| 6. Validação do Motor | `src/validation/` |

---

## Tecnologias e Ferramentas Utilizadas

- **Linguagens de Programação:** C++23, CUDA C++20, Python 3.13
- **Frameworks e Bibliotecas:** ONNX Runtime 1.24.1 (motor), PyTorch 2.11 (treinamento), chess.hpp, python-chess, pandas, SciPy
- **Banco de Dados / Mensageria:** não se aplica; a entrada e a saída usam arquivos CSV e o protocolo UCI
- **Plataforma Alvo / Hardware:** Windows 11 (x64), GPU NVIDIA com CUDA
- **Ferramentas de Compilação e Depuração:** Visual Studio 2026 (MSVC 19.50), CMake 3.28 ou superior, CUDA Toolkit 13, cuDNN 9, CTest

---

## Estrutura do Código

```text
.
├── data/
│   ├── partidas/                  # 3 partidas humanas: decisões, posições e perfil instrumental (8.192 simulações)
│   ├── data_<orçamento>.csv       # perfil instrumental de 16 puzzles nos 5 orçamentos de busca
│   └── sample_15375.csv           # os 15.375 puzzles usados no relatório
├── src/
│   ├── engine/                    # motor --> busca MCTS, inferência e protocolo UCI
│   ├── shared/include/            # biblioteca de xadrez compartilhada
│   ├── train/                     # extração dos dados, treinamento, exportação e benchmark da rede
│   ├── validation/                # experimentos do relatório (puzzles e partidas humanas)
│   └── requirements.txt           # dependências Python
├── tests/
│   ├── integracao/                # codificação e inferência do pipeline na GPU
│   └── unitarios/                 # busca, CSV e validação de puzzles
├── CMakeLists.txt
└── README.md
```

A explicação de cada descritor está em `src/validation/descriptors.md`.

---

## Pré-requisitos e Instalação

Requisitos: Windows 11 (x64), Visual Studio 2026 com suporte a C++23, CMake 3.28 ou superior, CUDA Toolkit 13 e cuDNN 9. As DLLs do cuDNN precisam estar no `PATH` ou na pasta `bin/`; sem elas, o motor executa a inferência sem GPU. O ONNX Runtime é baixado pelo CMake.

### 1. Clonar o repositório

```bash
git clone https://github.com/CoordENGCOMPPUCPR/TCC-Engenharia-de-Computacao.git
cd TCC-Engenharia-de-Computacao/projetos/2026-2/inferencia-de-complexidade-enxadristica
```

### 2. Compilar

```bash
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release
```

Os executáveis são gerados em `bin/`.

### 3. Instalar as dependências Python

Necessárias para o treinamento e os experimentos:

```bash
python -m pip install -r src/requirements.txt
```

### 4. Executar o motor

A rede treinada não acompanha o repositório: o modelo é gerado pelo treinamento (`train.py`) e pela exportação (`export_onnx.py`, que grava `checkpoint_epoch<N>.onnx`), descritos na etapa 5. O motor procura `model.onnx` no diretório atual e na pasta do executável; para usar outro caminho, informe a opção `WeightsFile`. O motor pode ser usado em qualquer interface UCI ou direto no terminal, com `bin/tfird.exe`:

```text
uci
setoption name WeightsFile value model.onnx
isready
position fen r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
go nodes 512
quit
```

O orçamento da busca é dado por `go nodes N` (simulações) ou `go movetime T` (milissegundos). Ao fim da busca, o motor emite a avaliação, uma linha com o perfil instrumental e o melhor lance (saída abreviada):

```text
info depth 4 seldepth 10 nodes 512 nps 1718 time 298 score cp 7 pv f1c4 g8f6 d2d4 e5d4 e1g1 f6e4 f1e1 d7d5
info string instrumental_profile 512{f1c4:114:0.360919446:0.012565708719193935:-0.0177154541|e1e2:1:7.00309943e-09:-0.37158203125:-0.371582031|...};...
bestmove f1c4 ponder g8f6
```

Cada nó da variante principal é um bloco `N{move:n:P:Q:V|...}`, separado por `;`: `N` são as visitas do nó e, para cada lance legal, `n` são as visitas, `P` o *prior* da rede, `Q` o valor médio da busca e `V` a avaliação inicial da rede. O primeiro lance de cada bloco é o mais visitado; `-` marca valores ausentes.

| Opção UCI | Padrão | Descrição |
| :--- | :--- | :--- |
| `WeightsFile` | `model.onnx` | caminho do modelo ONNX |
| `MinibatchSize` | 32 | tamanho do lote de inferência |
| `GatherBatchSize` | 32 | folhas coletadas por rodada da busca |
| `CPuct` | 2.5 | constante de exploração do PUCT |
| `VirtualLoss` | 3.0 | perda virtual entre folhas pendentes |
| `MultiPV` | 1 | número de variantes informadas |
| `RamLimitMb` | 2746 | memória máxima da árvore de busca, em MB |

### 5. Reproduzir os experimentos do relatório

Os resultados completos dependem de bases externas (Lichess) e de execuções longas, que não ficam neste repositório. Cada script em `src/validation/` traz o seu uso no início do arquivo; os programas de `src/train/` mostram as opções com `--help`, exceto o `miner.exe`, que lê a base do Lichess pela entrada padrão.

| Etapa | Comandos |
| :--- | :--- |
| Treinamento da rede | `miner.exe` extrai o *dataset* da base de avaliações do Lichess; `puzzle_hasher.exe` e `filter_dataset.exe` tiram as posições dos *puzzles*, `balance_dataset.exe` balanceia e `split_dataset.py` separa treino e validação; `train.py` treina e `export_onnx.py` exporta os *checkpoints* para ONNX |
| Amostra de *puzzles* | `make_puzzle_sample.py` gera `data/sample_15375.csv` a partir da base do Lichess |
| Busca nos *puzzles* | `run_puzzles.exe` grava o perfil instrumental de cada *puzzle* em `data_<orçamento>.csv` |
| Descritores | `profile_descriptors.py`, `descriptor_correlations.py`, `pv_resolution.py` |
| Partidas humanas | `eval_games_stockfish.py`, `games_to_positions.py`, `run_positions.exe`, `game_descriptors.py`, `game_analysis.py` |
| Rede sem busca | `model_benchmark.exe` |

---

## Procedimentos de Teste e Validação (Plano de Testes)

### Testes automatizados

Depois da compilação:

```bash
ctest --test-dir build -C Release --output-on-failure
```

| Teste | Pasta | O que verifica |
| :--- | :--- | :--- |
| `tfird_search_tests` | `tests/unitarios/` | orçamento exato de simulações, determinismo com lote de coleta unitário e perfil bruto |
| `puzzle_csv_tests` | `tests/unitarios/` | gravação do perfil bruto no CSV sem alteração de N/P/Q/V e retomada da execução |
| `puzzle_validation_tests` | `tests/unitarios/` | orçamento e perfil bruto da busca no *puzzle*, sem inferência extra |
| `tfird_cuda_encoding_tests` | `tests/integracao/` | codificação em CUDA igual à da CPU; o teste é ignorado sem GPU |

A inferência pelo pipeline na GPU é comparada com a rede rodada direto pelo ONNX Runtime, por um teste que recebe o modelo como argumento e exige as DLLs do cuDNN:

```bash
bin/tfird_cuda_inference_tests.exe model.onnx
```

### Testes em caixa branca

| Componente | Procedimento |
| :--- | :--- |
| *Pipeline* de Dados | `puzzle_hasher.exe` extrai os *hashes* de Zobrist dos *puzzles* e `filter_dataset.exe` exclui essas posições do treino, para evitar *Data Leakage* |
| Rede Neural | perdas por época em `training_log.csv` (`train.py`) e `model_benchmark.exe` |
| MCTS | `tfird_search_tests` |
| API | verificação manual da sequência UCI do item 4 de [Pré-requisitos e Instalação](#pré-requisitos-e-instalação): o motor responde `uciok`, `readyok` e `bestmove` |

### Testes em caixa preta

| Componente | Procedimento |
| :--- | :--- |
| Motor | sequência UCI do item 4: a linha `info` informa `nodes 512` e a linha seguinte traz o perfil instrumental |
| Rede Neural | `model_benchmark.exe` sobre o *holdout* de *puzzles* |
| Indicadores | reconstrução dos descritores a partir do CSV exportado, sem nova execução da busca (abaixo) |

**Reconstrução dos descritores.** Os arquivos `data/data_<orçamento>.csv` guardam o perfil instrumental de 16 *puzzles* (o primeiro identificador em ordem alfabética de cada faixa de *rating*), recortado sem alteração dos resultados do relatório, nos orçamentos de 2.048 a 32.768 simulações. No PowerShell:

```powershell
python src\validation\profile_descriptors.py (Get-ChildItem data\data_*.csv).FullName
python src\validation\descriptor_correlations.py data\descritores
```

Os descritores ficam em `data/descritores/descritores_<orçamento>.csv`, com os mesmos valores do relatório para cada *puzzle*. O arquivo `correlacoes.csv` confere apenas a execução: 16 *puzzles* não reproduzem os coeficientes do relatório, calculados sobre 15.375.

**Partidas humanas.** A pasta `data/partidas/` guarda 3 das 2.816 partidas do estudo (a primeira em ordem alfabética das faixas `600_minus`, `1600_1799` e `2600_plus`), com as 262 decisões, as posições e o perfil instrumental de cada decisão com 8.192 simulações, recortados sem alteração dos resultados do relatório:

```powershell
python src\validation\game_descriptors.py data\partidas\busca_8192.csv
python src\validation\game_analysis.py data\partidas 8192
```

Os descritores ficam em `data/partidas/descritores/descritores_8192.csv` e a análise de cada decisão em `data/partidas/analise/decisoes_analise.csv`, com os mesmos valores do relatório. As correlações de 3 partidas não reproduzem as do relatório.

**Geração de um novo perfil.** Exemplo com 16 *puzzles* e 64 simulações, que exige a execução da busca:

```powershell
New-Item -ItemType Directory -Force saida
Get-Content data\sample_15375.csv | bin\run_puzzles.exe --model model.onnx --output saida\data_64.csv --sims 64 --per-bin 1
python src\validation\profile_descriptors.py saida\data_64.csv
```

A saída é `saida/descritores/descritores_64.csv`, com uma linha por *puzzle*.

---

## Histórico de Versões (Marcos CDPT / IOPT)

| Versão | Etapa | Data | Descrição |
| :--- | :--- | :--- | :--- |
| `v0.1.0` | CDPT — 1º Bim (Plano de Projeto) | ~30/04/2026 | Revisão bibliográfica e desenho da arquitetura |
| `v0.2.0` | CDPT — 2º Bim (Especificação Física) | ~26/06/2026 | *Pipeline* de dados e protótipo do treinamento da rede |
| `v0.5.0` | IOPT — 3º Bim (Protótipo) | ~07/10/2026 | Treinamento da rede em escala, busca MCTS adaptada, inferência, API UCI e validação do motor |
| `v1.0.0` | IOPT — 4º Bim (Implementação Final) | 06/11/2026 | Análise final das partidas humanas, benchmark da rede refeito, scripts de validação simplificados e código comentado |

As datas com `~` são estimativas feitas a partir do cronograma do projeto.
