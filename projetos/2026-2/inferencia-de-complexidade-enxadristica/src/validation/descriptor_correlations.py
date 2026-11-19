#!/usr/bin/env python3
"""Correlação de Spearman entre cada descritor do perfil e o rating.

Uso:
    python descriptor_correlations.py <pasta descritores>

<pasta descritores> contém descritores_<orçamento>.csv (profile_descriptors.py).
Saída na mesma pasta: correlacoes.csv, uma linha por (orçamento, descritor) com n, rho e p.
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

# orçamentos de simulações, um CSV de descritores para cada
BUDGETS = (2048, 4096, 8192, 16384, 32768)


def correlations(folder):
    """Spearman entre o rating e cada descritor, por orçamento; rho e p ficam vazios com menos de 3 puzzles ou com descritor constante."""
    rows = []
    for budget in BUDGETS:
        df = pd.read_csv(folder / f"descritores_{budget}.csv")
        for name in (c for c in df.columns if c not in ("puzzle_id", "rating")):
            sub = df[["rating", name]].dropna()
            row = {"orcamento": budget, "descritor": name, "n": len(sub)}
            if len(sub) >= 3 and sub[name].nunique() > 1:
                row["rho"], row["p"] = stats.spearmanr(sub["rating"], sub[name])
            rows.append(row)
        print(budget, "ok", flush=True)
    return pd.DataFrame(rows)


def main():
    folder = Path(sys.argv[1])
    correlations(folder).to_csv(folder / "correlacoes.csv", index=False)
    print("->", folder / "correlacoes.csv")


if __name__ == "__main__":
    main()
