#!/usr/bin/env python3
"""Extrai os descritores do perfil instrumental das buscas nas partidas humanas.

Uso:
    python game_descriptors.py [--nmin 1] busca_2048.csv [busca_4096.csv ...]

Entrada: saída do run_positions (position_id,fen,cp,instrumental_profile).
Saída:   <pasta do csv>/descritores/descritores_<orçamento>.csv, uma linha por decisão (position_id) e uma coluna por descritor.

Os descritores são os mesmos dos puzzles, pois o cálculo vem de profile_descriptors.py.
"""

import argparse
from pathlib import Path

import pandas as pd

from profile_descriptors import describe


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nmin", type=int, default=1)
    ap.add_argument("csv", nargs="+", type=Path)
    args = ap.parse_args()

    for path in args.csv:
        # orçamento pelo nome do arquivo (busca_<orçamento>.csv)
        budget = path.stem.split("_")[-1]
        df = pd.read_csv(path, usecols=["position_id", "instrumental_profile"])
        rows = [describe(p, args.nmin) for p in df["instrumental_profile"]]
        out = pd.concat([df[["position_id"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
        out_dir = path.parent / "descritores"
        out_dir.mkdir(exist_ok=True)
        target = out_dir / f"descritores_{budget}.csv"
        out.to_csv(target, index=False)
        print(f"{path.name}: {len(out)} decisoes, {out.shape[1] - 1} descritores -> {target}", flush=True)


if __name__ == "__main__":
    main()
