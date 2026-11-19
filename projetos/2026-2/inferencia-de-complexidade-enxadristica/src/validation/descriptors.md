# Descritores do perfil instrumental

Descritores calculados por `profile_descriptors.py` a partir do perfil instrumental.

## Perfil instrumental

$$
\Phi(D) = \left[ N_t\left\{(a_{t,i},N_{t,i},P_{t,i},Q_{t,i},V_{t,i})\right\}_{i=1}^{k_t} \right]_{t=0}^{T-1}
$$

| Campo | Significado |
|---|---|
| $t$ | profundidade do nó na variante principal ($t=0$: raiz) |
| $N_t$ | visitas do nó $s_t$ |
| $k_t$ | número de filhos legais de $s_t$ |
| $a_{t,i}$ | lance do filho $i$ ($i=1$: continuação da variante principal) |
| $N_{t,i}$ | visitas do filho $i$ |
| $P_{t,i}$ | *prior* da rede para o filho $i$ |
| $Q_{t,i}$ | média dos valores retropropagados pelo filho $i$ (ausente se $N_{t,i}=0$) |
| $V_{t,i}$ | primeira avaliação da cabeça de valor no filho $i$ (ausente se não inferido) |

$Q$ e $V$ estão na perspectiva do jogador que decide em $s_t$, na escala $\tanh(\text{cp}/600)$ usada no treinamento da rede.

## Convenções

| Símbolo | Definição |
|---|---|
| $\pi_{t,i}$ | $N_{t,i} / \sum_j N_{t,j}$, fração de visitas do filho $i$ |
| $p_{t,i}$ | $P_{t,i}$ com piso $10^{-6}$, normalizado sobre os filhos legais |
| $\mathcal{T}$ | $\{t : N_t \ge N_{\min}\}$, nós elegíveis |
| $\mathcal{T}_{\mathrm{even}}$, $\mathcal{T}_{\mathrm{odd}}$ | nós elegíveis de profundidade par (solucionador) e ímpar (adversário) |
| $\mathcal{C}_t$ | $\{i : N_{t,i} \ge 1\}$, filhos visitados, os únicos com $Q$ |

## 1. Grandezas por nó

| Símbolo | Coluna | Definição | Leitura |
|---|---|---|---|
| $H_t$ | `H` | $-\sum_i \pi_{t,i}\log_2\pi_{t,i}$ | dispersão das visitas (entropia posicional) |
| $d_t$ | `d` | $\sum_i \pi_{t,i}\log_2(\pi_{t,i}/p_{t,i})$ | afastamento da busca em relação ao *prior* |
| $g_t$ | `g_nc1` | $Q_{t,1} - \max_{i \in \mathcal{C}_t,\, i>1} Q_{t,i}$ | margem de valor do lance escolhido |
| $\sigma^2_t$ | `S_nc1` | $\sum_{i \in \mathcal{C}_t} \pi_{t,i}(Q_{t,i} - \bar{Q}_t)^2 / \sum_{i \in \mathcal{C}_t} \pi_{t,i}$ | variância de valor ponderada pelas visitas |

## 2. Colunas de saída

| Coluna | Símbolo no relatório | Definição |
|---|---|---|
| `H_root` | $H_0$ | $H_t$ na raiz |
| `d_root` | $d_0$ | $d_t$ na raiz |
| `d_sum_pv` | $\Sigma\, d_t$ | $\sum_{t \in \mathcal{T}} d_t$ |
| `d_sum_even` | $\Sigma\, d_{\mathrm{even}}$ | $\sum_{t \in \mathcal{T}_{\mathrm{even}}} d_t$ |
| `d_sum_odd` | $\Sigma\, d_{\mathrm{odd}}$ | $\sum_{t \in \mathcal{T}_{\mathrm{odd}}} d_t$ |
| `g_nc1_root` | $g_0$ | $g_t$ na raiz |
| `S_nc1_root` | $\sigma^2_0$ | $\sigma^2_t$ na raiz |
| `S_nc1_sum_pv` | $\Sigma\, \sigma^2_t$ | $\sum_{t \in \mathcal{T}} \sigma^2_t$ |
| `T` | $T$ | número de nós da variante principal no perfil, sem filtro |
| `T_even`, `T_odd` | $T_{\mathrm{even}}$, $T_{\mathrm{odd}}$ | $\lvert\mathcal{T}_{\mathrm{even}}\rvert$, $\lvert\mathcal{T}_{\mathrm{odd}}\rvert$ |

Na soma, termos indefinidos são omitidos.

## Casos degenerados

| Situação | Efeito |
|---|---|
| lance legal único | $H_t = 0$, $d_t = 0$; $g_t$ indefinido |
| um único filho em $\mathcal{C}_t$ | $g_t$ indefinido; $\sigma^2_t = 0$ |
| $T = 1$ (mate em um lance) | soma ímpar vazia |

Descritor indefinido é registrado como ausente e excluído do cálculo da correlação correspondente.

## Parâmetros

| Parâmetro | Valor |
|---|---|
| $N_{\min}$ | 1, sem filtro (`--nmin`) |
| piso do *prior* | $10^{-6}$ |

## Dados de entrada

| Arquivo | Conteúdo |
|---|---|
| `data_<orçamento>.csv` (`run_puzzles`) | `puzzle_id`, `fen`, `rating`, `cp`, `instrumental_profile` |
| `busca_<orçamento>.csv` (`run_positions`) | `position_id`, `fen`, `cp`, `instrumental_profile`, nas partidas (`game_descriptors.py`) |
