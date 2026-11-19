#!/usr/bin/env python3
"""Amostra estratificada de puzzles do Lichess (Seção "Desempenho da Busca").

Uso:
    python make_puzzle_sample.py [lichess_db_puzzle.csv] [sample_15375.csv]

Amostragem de reservatório de até 1.024 puzzles por faixa de 200 pontos entre 200 e 3.399 (semente 42); a última faixa tem apenas 15 puzzles disponíveis, o que totaliza 15.375.
"""
import csv, random, sys
from collections import defaultdict

INPUT  = sys.argv[1] if len(sys.argv) > 1 else "lichess_db_puzzle.csv"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "sample_15375.csv"
RMIN, RMAX, WIDTH = 200, 3400, 200   # faixas de 200 pontos, de 200 a 3399
SEED   = 42
TARGET_PER_BIN = 1024

rng = random.Random(SEED)
lows = list(range(RMIN, RMAX, WIDTH))
nbins = len(lows)

reservoir = defaultdict(list)   # faixa --> puzzles sorteados
seen = defaultdict(int)         # faixa --> puzzles lidos
total = 0
with open(INPUT, newline="", encoding="utf-8") as f:
    r = csv.reader(f); next(r)                      # pula o cabeçalho
    for row in r:
        total += 1
        try:
            rt = int(row[3])                        # coluna Rating
        except (ValueError, IndexError):
            continue
        if rt < RMIN or rt >= RMAX:
            continue
        b = (rt - RMIN) // WIDTH                    # faixa (0 = 200-399, ..., 15 = 3200-3399)
        seen[b] += 1; res = reservoir[b]
        # amostragem de reservatório --> cada puzzle lido na faixa tem a mesma chance de ficar na amostra
        if len(res) < TARGET_PER_BIN:
            res.append(row)
        else:
            j = rng.randint(0, seen[b] - 1)
            if j < TARGET_PER_BIN:
                res[j] = row

out = []
for b in range(nbins):
    out.extend(reservoir[b])
out.sort(key=lambda x: int(x[3]))   # ordena pelo rating

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["PuzzleId", "FEN", "Moves", "Rating"])
    for row in out:
        w.writerow([row[0], row[1], row[2], row[3]])

print(f"Lidos {total:,} | escrito {len(out)} -> {OUTPUT}\n")
for b in range(nbins):
    print(f"{lows[b]:>4}-{lows[b]+WIDTH-1}: alvo {TARGET_PER_BIN:>3}  obtido {len(reservoir[b]):>3}  disponivel {seen[b]:>8,}")
