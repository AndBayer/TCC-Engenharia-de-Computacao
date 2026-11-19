#!/usr/bin/env python3
"""Cruza os descritores da busca com as decisões humanas das partidas.

Uso:
    python game_analysis.py <pasta do ritmo> 8192 [16384 ...]

<pasta do ritmo> contém decisoes.csv (games_to_positions.py) e descritores/descritores_<orçamento>.csv (game_descriptors.py).

Protocolo (Seção "Estudo Exploratório com Partidas Humanas"):
  - todas as decisões de todas as partidas, sem cortes;
  - faixa da decisão pelo rating de quem joga (faixas de 200 pontos; abaixo de 800 --> "600_minus", 2.600 ou mais --> "2600_plus");
  - avaliação do Stockfish convertida pela convenção do treino da rede, v = tanh(cp / 600), na visão de quem joga; mate = +-1;
  - fase pelo critério do Lichess (scalachess Divider) --> meio-jogo a partir da primeira posição com <= 10 peças (sem reis e peões), última fileira esparsa (< 4 peças) de um dos lados ou mixedness > 150; final a partir de <= 6 peças, se houve meio-jogo;
  - tempo relativo --> posição do lance entre os lances do mesmo jogador na partida, ordenados pelo tempo gasto, como fração em (0, 1] (empates recebem a média das posições); tempo negativo (tempo extra concedido pelo adversário) fica indefinido;
  - perda --> queda de v com o lance, max(0, v antes - v depois);
  - Spearman de cada descritor com o tempo e com a perda --> geral, por faixa, por fase, por faixa de |v| e acumulado por lote.

Saídas em <pasta>/analise/:
  decisoes_analise.csv   decisões com faixa, fase, v, perda e tempo relativo
  correlacoes.csv        orçamento x recorte x descritor x grandeza (n, rho, p)
"""

import sys
from pathlib import Path

import chess
import numpy as np
import pandas as pd
from scipy import stats

ESCALA_CP = 600          # v = tanh(cp / 600), convenção do treino da rede
FAIXA, FAIXA_MIN, FAIXA_MAX = 200, 600, 2600   # faixas de rating de 200 pontos, de 600_minus a 2600_plus
FAIXAS_V = [0, 0.25, 0.5, 0.9, 1.0]            # limites das faixas de |v| antes do lance

# coluna do CSV de descritores --> nome do descritor no TCC
DESCRITORES = {
    "H_root": "H0", "d_root": "d0", "d_sum_pv": "Sigma_d", "d_sum_even": "D_even",
    "d_sum_odd": "D_odd", "T": "T", "g_nc1_root": "g0", "S_nc1_sum_pv": "Sigma_S",
}
# grandezas das decisões comparadas com cada descritor
GRANDEZAS = ("tempo_relativo", "perda_v")


# fases da partida --> tradução de scalachess/core/src/main/scala/Divider.scala

def _score(y, white, black):
    """Pontos de uma região 2x2 pelo número de peças de cada cor nela (com peões e reis); y = fileira de baixo da região (1 a 7)."""
    if white == 0:
        if black == 1:
            return 1 + y
        if black == 2:
            return 2 + (6 - y) if y < 6 else 0
        if black in (3, 4):
            return 3 + (7 - y) if y < 7 else 0
        return 0
    if white == 1:
        return {0: 1 + (8 - y), 1: 5 + abs(4 - y), 2: 4 + (7 - y), 3: 5 + (7 - y)}.get(black, 0)
    if white == 2:
        if black == 0:
            return 2 + (y - 2) if y > 2 else 0
        return {1: 4 + (y - 1), 2: 7}.get(black, 0)
    if white == 3:
        if black == 0:
            return 3 + (y - 1) if y > 1 else 0
        return 5 + (y - 1) if black == 1 else 0
    if white == 4 and black == 0:
        return 3 + (y - 1) if y > 1 else 0
    return 0


# as 49 regiões 2x2 do tabuleiro --> (máscara das 4 casas, fileira de baixo de 1 a 7)
_REGIOES = [(0x0303 << (x + 8 * y), y + 1) for y in range(7) for x in range(7)]


def _mixedness(board):
    """Quanto as peças brancas e pretas estão misturadas no tabuleiro (soma dos pontos das 49 regiões)."""
    brancas, pretas = board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK]
    return sum(_score(y, chess.popcount(brancas & r), chess.popcount(pretas & r)) for r, y in _REGIOES)


def _maiores_menores(board):
    """Peças no tabuleiro, sem contar reis e peões."""
    return chess.popcount(board.occupied & ~(board.kings | board.pawns))


def _ultima_fileira_esparsa(board):
    """Menos de 4 peças brancas na fileira 1 ou menos de 4 peças pretas na fileira 8."""
    return (chess.popcount(chess.BB_RANK_1 & board.occupied_co[chess.WHITE]) < 4
            or chess.popcount(chess.BB_RANK_8 & board.occupied_co[chess.BLACK]) < 4)


def fases(fens):
    """Fase de cada posição de uma partida, na ordem dos meios-lances."""
    boards = [chess.Board(f) for f in fens]
    # índice da primeira posição do meio-jogo e do final (None se a fase não ocorre)
    meio = next((i for i, b in enumerate(boards)
                 if _maiores_menores(b) <= 10 or _ultima_fileira_esparsa(b) or _mixedness(b) > 150), None)
    final = None
    if meio is not None:
        final = next((i for i, b in enumerate(boards) if _maiores_menores(b) <= 6), None)
        # meio-jogo que não começa antes do final é descartado
        if final is not None and meio >= final:
            meio = None
    rotulos = []
    for i in range(len(boards)):
        if final is not None and i >= final:
            rotulos.append("final")
        elif meio is not None and i >= meio:
            rotulos.append("meio-jogo")
        else:
            rotulos.append("abertura")
    return rotulos


def v_do_jogador(cp, mate, sinal):
    """Avaliação do PGN (visão das brancas) em v = tanh(cp/600), na visão de quem joga."""
    v = np.where(pd.notna(mate), np.sign(mate), np.tanh(cp / ESCALA_CP))
    return sinal * v


def faixa(elo):
    """Faixa de 200 pontos do rating; abaixo de 800 --> "600_minus", 2600 ou mais --> "2600_plus"."""
    b = np.clip((elo // FAIXA) * FAIXA, FAIXA_MIN, FAIXA_MAX).astype(int)
    return np.where(b == FAIXA_MIN, f"{FAIXA_MIN}_minus",
                    np.where(b == FAIXA_MAX, f"{FAIXA_MAX}_plus", b.astype(str)))


def preparar(decisoes):
    """Acrescenta a cada decisão v antes e depois do lance, perda, faixas, fase e tempo relativo."""
    d = decisoes.sort_values(["game_id", "ply"]).reset_index(drop=True)
    # troca a visão das brancas pela de quem joga
    sinal = np.where(d["side"] == "w", 1, -1)
    d["v_antes"] = v_do_jogador(d["sf_cp_before_white"], d["sf_mate_before_white"], sinal)
    d["v_depois"] = v_do_jogador(d["sf_cp_after_white"], d["sf_mate_after_white"], sinal)
    # no lance final sem avaliação depois, mate dado vale +1 e empate fica indefinido
    fim = d["v_depois"].isna()
    d.loc[fim, "v_depois"] = [1.0 if chess.Board(f).is_checkmate() else np.nan for f in d.loc[fim].apply(_fen_depois, axis=1)]
    d["perda_v"] = np.maximum(0, d["v_antes"] - d["v_depois"])
    d["faixa_v"] = pd.cut(d["v_antes"].abs(), FAIXAS_V, include_lowest=True).astype(str)
    d["faixa"] = faixa(d["mover_elo"])
    # fase de cada posição, partida por partida (d está ordenado por partida e lance)
    d["fase"] = [f for _, g in d.groupby("game_id", sort=False) for f in fases(g["fen"])]
    # tempo relativo --> posição do tempo gasto entre os lances do mesmo jogador na partida, em (0, 1]; tempo negativo fica indefinido
    t = d["time_spent_s"].where(d["time_spent_s"] >= 0)
    d["tempo_relativo"] = t.groupby([d["game_id"], d["side"]]).rank(pct=True, method="average")
    return d


def _fen_depois(row):
    """FEN depois do lance jogado."""
    b = chess.Board(row["fen"])
    b.push_uci(row["move_played"])
    return b.fen()


def correlacoes(df, recorte, valor):
    """Spearman de cada descritor com cada grandeza num recorte; pula com menos de 30 decisões ou com coluna constante."""
    linhas = []
    for col, nome in DESCRITORES.items():
        for g in GRANDEZAS:
            sub = df[[col, g]].dropna()
            if len(sub) < 30 or sub[col].nunique() < 2 or sub[g].nunique() < 2:
                continue
            rho, p = stats.spearmanr(sub[col], sub[g])
            linhas.append({"recorte": recorte, "valor": valor, "descritor": nome,
                           "grandeza": g, "n": len(sub), "rho": rho, "p": p})
    return linhas


def main():
    pasta = Path(sys.argv[1])
    orcamentos = [int(b) for b in sys.argv[2:]]
    saida = pasta / "analise"
    saida.mkdir(exist_ok=True)

    decisoes = preparar(pd.read_csv(pasta / "decisoes.csv"))
    decisoes.to_csv(saida / "decisoes_analise.csv", index=False)
    print(f"{len(decisoes)} decisoes, {decisoes['game_id'].nunique()} partidas")

    linhas = []
    for b in orcamentos:
        desc = pd.read_csv(pasta / "descritores" / f"descritores_{b}.csv", usecols=["position_id"] + list(DESCRITORES))
        # só entram as decisões que já têm busca nesse orçamento
        df = decisoes.merge(desc, on="position_id", how="inner")
        blocos = [correlacoes(df, "geral", "todas")]
        for recorte in ("faixa", "fase", "faixa_v"):
            blocos += [correlacoes(sub, recorte, v) for v, sub in df.groupby(recorte)]
        # acumulado por lote --> lotes 1 a k, para cada k
        if "lote" in df:
            blocos += [correlacoes(df[df["lote"] <= k], "lotes_ate", k) for k in sorted(df["lote"].unique())]
        linhas += [{"orcamento": b, **l} for bloco in blocos for l in bloco]
        print(f"{b}: {len(df)} decisoes com busca", flush=True)

    pd.DataFrame(linhas).to_csv(saida / "correlacoes.csv", index=False)
    print("->", saida)


if __name__ == "__main__":
    main()
