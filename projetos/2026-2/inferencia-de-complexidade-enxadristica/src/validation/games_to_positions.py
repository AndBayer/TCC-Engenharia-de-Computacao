#!/usr/bin/env python3
"""Prepara as decisões das partidas humanas para a busca MCTS (run_positions.exe).

Uso:
    python games_to_positions.py <pasta dos PGNs anotados pelo Stockfish> <pasta de saída> [--lote 64] [--semente 0]

Saídas:
    posicoes.csv  entrada do run_positions (position_id,fen), uma linha por decisão; a busca roda sobre a FEN como está.
    decisoes.csv  uma linha por lance jogado nas partidas, com o position_id para cruzar com a saída da busca e os dados do Stockfish.

position_id = <id da partida no Lichess>_<ply>, estável entre execuções (o run_positions retoma pelos ids já gravados).

As partidas de cada faixa são embaralhadas com semente fixa e numeradas; a partida de ordem k recebe o lote k // LOTE + 1. O posicoes.csv segue a ordem (ordem na faixa, faixa): qualquer prefixo da busca cobre as faixas por igual, e a busca pode ser interrompida ao fim de qualquer lote.

Cada decisão recebe uma busca própria, inclusive quando a posição se repete em outras partidas.
"""
import argparse
import csv
import random
from pathlib import Path

import chess
import chess.pgn


def avaliacao(node):
    """(cp na visão das brancas, mate, profundidade) do comentário [%eval]."""
    score = node.eval()
    if score is None:
        return "", "", ""
    branco = score.white()
    return ("" if branco.is_mate() else branco.score(),
            branco.mate() if branco.is_mate() else "",
            node.eval_depth() or "")


def melhor(node):
    """Melhor lance do Stockfish na posição do nó ([%best] do comentário); '' se não tiver."""
    for parte in node.comment.split("[%best ")[1:]:
        return parte.split("]")[0].strip()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgns", type=Path)
    ap.add_argument("saida", type=Path)
    ap.add_argument("--lote", type=int, default=64, help="partidas por faixa em cada lote")
    ap.add_argument("--semente", type=int, default=0)
    args = ap.parse_args()
    pgns, saida = args.pgns, args.saida
    saida.mkdir(parents=True, exist_ok=True)

    partidas = []            # (ordem na faixa, faixa, jogo)
    for arquivo in sorted(pgns.glob("*.pgn")):
        bracket = arquivo.stem.replace("bracket_", "")
        with open(arquivo, encoding="utf-8") as f:
            jogos = []
            while (jogo := chess.pgn.read_game(f)) is not None:
                jogos.append(jogo)
        # o anotador grava na ordem de conclusão, então ordena por id antes de sortear
        jogos.sort(key=lambda j: j.headers["Site"])
        # semente própria de cada faixa --> a ordem de uma faixa não depende das outras
        random.Random(f"{args.semente}-{bracket}").shuffle(jogos)
        partidas += [(k, bracket, jogo) for k, jogo in enumerate(jogos)]
    # intercala as faixas --> primeira partida de cada faixa, depois a segunda, e assim por diante
    partidas.sort(key=lambda p: (p[0], p[1]))

    posicoes = []            # (position_id, fen), uma por decisão
    decisoes = []            # linhas do decisoes.csv, uma por lance jogado
    for ordem, bracket, jogo in partidas:
        lote = ordem // args.lote + 1
        h = jogo.headers
        gid = h["Site"].rsplit("/", 1)[-1]
        ultimo_relogio = {chess.WHITE: None, chess.BLACK: None}   # relógio de cada lado depois do seu último lance
        no_atual = jogo              # nó cuja posição é a da decisão
        # no_lance = nó do lance jogado, cuja posição é a de depois do lance
        for ply, no_lance in enumerate(jogo.mainline()):
            board = no_atual.board()
            lado = board.turn
            pid = f"{gid}_{ply}"
            posicoes.append((pid, board.fen()))
            relogio = no_lance.clock()
            # tempo gasto = relógio de quem joga após o lance anterior dele menos o relógio após o lance atual (vazio sem os dois)
            gasto = (ultimo_relogio[lado] - relogio
                     if ultimo_relogio[lado] is not None and relogio is not None else "")
            # avaliação do Stockfish antes e depois do lance, na visão das brancas
            antes = avaliacao(no_atual)
            depois = avaliacao(no_lance)
            decisoes.append({
                "game_id": gid, "bracket": bracket, "lote": lote,
                "white_elo": h.get("WhiteElo"), "black_elo": h.get("BlackElo"),
                "ply": ply, "move_number": board.fullmove_number,
                "side": "w" if lado == chess.WHITE else "b",
                "mover_elo": h.get("WhiteElo") if lado == chess.WHITE else h.get("BlackElo"),
                "position_id": pid, "fen": board.fen(),
                "move_played": no_lance.move.uci(), "sf_best": melhor(no_atual),
                "sf_cp_before_white": antes[0], "sf_mate_before_white": antes[1], "sf_depth_before": antes[2],
                "sf_cp_after_white": depois[0], "sf_mate_after_white": depois[1], "sf_depth_after": depois[2],
                "clock_before_s": "" if ultimo_relogio[lado] is None else ultimo_relogio[lado],
                "time_spent_s": gasto,
            })
            if relogio is not None:
                ultimo_relogio[lado] = relogio
            no_atual = no_lance

    with open(saida / "posicoes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["position_id", "fen"])
        w.writerows(posicoes)
    with open(saida / "decisoes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(decisoes[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(decisoes)
    print(f"{len(decisoes)} decisoes, {len(posicoes)} posicoes para a busca -> {saida}")


if __name__ == "__main__":
    main()
