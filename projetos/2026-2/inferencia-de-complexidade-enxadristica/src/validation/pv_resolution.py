#!/usr/bin/env python3
"""Resolução dos puzzles pela variante principal da busca (Seção "Desempenho da Busca").

Uso:
    python pv_resolution.py <pasta data/mcts> <lichess_db_puzzle.csv>

O puzzle conta como resolvido quando a variante principal da busca no primeiro lance do resolvedor reproduz a linha completa da solução do Lichess (lances do resolvedor e respostas do adversário). Quando não reproduz, o primeiro desvio é atribuído ao lado que deveria jogar o lance.

Saídas no terminal:
    por orçamento --> % resolvidos, desvios em lance do resolvedor e do adversário
    32.768 simulações --> % resolvidos por faixa de 200 pontos de rating
"""
import sys
from pathlib import Path

import pandas as pd

BUDGETS = (2048, 4096, 8192, 16384, 32768)


def solucoes(arquivo, ids):
    """Linha da solução de cada puzzle, sem o lance forçador."""
    sol = {}
    # lê o CSV do Lichess em blocos e guarda só os puzzles da amostra
    for c in pd.read_csv(arquivo, usecols=["PuzzleId", "Moves"], chunksize=500_000):
        c = c[c.PuzzleId.isin(ids)]
        sol.update(zip(c.PuzzleId, c.Moves.str.split().str[1:]))
    return sol


def variante(perfil):
    """Lances da variante principal, com o primeiro filho (o mais visitado) de cada nó do perfil."""
    if perfil == "-":
        return []
    return [no[:-1].split("{", 1)[1].split("|")[0].split(":")[0] for no in perfil.split(";")]


def main():
    pasta, db = Path(sys.argv[1]), sys.argv[2]
    # os puzzles são os mesmos em todos os orçamentos, então os ids vêm do CSV de 2048
    ids = set(pd.read_csv(pasta / "data_2048.csv", usecols=["puzzle_id"]).puzzle_id)
    sol = solucoes(db, ids)
    for B in BUDGETS:
        d = pd.read_csv(pasta / f"data_{B}.csv", usecols=["puzzle_id", "rating", "instrumental_profile"])
        ok, lado = [], []
        for pid, perfil in zip(d.puzzle_id, d.instrumental_profile):
            s, v = sol[pid], variante(perfil)
            # k = lances iguais desde o início; desvio em k par é do resolvedor, em k ímpar do adversário
            k = 0
            while k < len(s) and k < len(v) and s[k] == v[k]:
                k += 1
            ok.append(k == len(s))
            lado.append(None if k == len(s) else ("resolvedor" if k % 2 == 0 else "adversario"))
        d["ok"], d["lado"] = ok, lado
        print(f"{B}: resolvidos {d.ok.mean() * 100:.2f}% | desvio no resolvedor {(d.lado == 'resolvedor').sum()}"
              f" | desvio no adversario {(d.lado == 'adversario').sum()}")
        # por faixa de 200 pontos, só no maior orçamento
        if B == BUDGETS[-1]:
            faixa = d.rating // 200 * 200
            t = d.groupby(faixa).ok.agg(["size", "mean"])
            t["mean"] = (t["mean"] * 100).round(2)
            print(t.rename(columns={"size": "puzzles", "mean": "resolvidos_%"}).to_string())


if __name__ == "__main__":
    main()
