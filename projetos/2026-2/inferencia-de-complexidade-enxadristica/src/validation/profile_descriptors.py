#!/usr/bin/env python3
"""Extrai do perfil instrumental os descritores usados no relatório (descriptors.md).

Uso:
    python profile_descriptors.py [--nmin 1] data_2048.csv [data_4096.csv ...]

Saída: <pasta do csv>/descritores/descritores_<orçamento>.csv, uma linha por puzzle e uma coluna por descritor.

Colunas: <grandeza>_<agregação>, com grandeza em H (entropia), d (divergência), g_nc1 (margem de valor) e S_nc1 (variância de valor), e agregação em root (nó raiz) ou sum_even, sum_odd, sum_pv (soma nos nós pares, ímpares ou em todos os nós elegíveis da variante); T, T_even e T_odd contam os nós.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

MINIMUM_PRIOR = 1e-6
# descritores que vão para o CSV, além de T, T_even e T_odd
SAIDAS = ("H_root", "d_root", "d_sum_pv", "d_sum_even", "d_sum_odd", "g_nc1_root", "S_nc1_root", "S_nc1_sum_pv")

def parse_node(item):
    """'N{move:n:P:Q:V|...}' --> (N, [(n, P, Q|None, V|None), ...])."""
    visits_text, children_text = item[:-1].split("{", 1)
    children = []
    for child in children_text.split("|"):
        _, n, p, q, v = child.split(":")
        children.append((int(n), float(p), None if q == "-" else float(q), None if v == "-" else float(v)))
    return int(visits_text), children


def node_quantities(children):
    """Grandezas por nó (Seção 1 de descriptors.md); indefinido --> nan."""
    counts = np.array([c[0] for c in children], dtype=float)
    # piso no prior para evitar log de zero
    priors = np.maximum(np.array([c[1] for c in children]), MINIMUM_PRIOR)
    priors = priors / priors.sum()
    # pi = proporção de visitas
    pi = counts / counts.sum()
    # ignora pi = 0 (0 log 0 = 0)
    nz = pi > 0

    # H = entropia, d = divergência KL, em bits
    x = {
        "H": max(-(pi[nz] * np.log2(pi[nz])).sum(), 0.0),
        "d": max((pi[nz] * np.log2(pi[nz] / priors[nz])).sum(), 0.0),
    }
    # filhos visitados com Q; o primeiro é o lance escolhido (o mais visitado)
    idx = [i for i, c in enumerate(children) if c[0] >= 1 and c[2] is not None]
    alts = [i for i in idx if i != 0]
    qs = {i: children[i][2] for i in idx}
    # margem = Q do escolhido menos o melhor Q das alternativas
    x["g_nc1"] = qs[0] - max(qs[i] for i in alts) if (0 in idx and alts) else math.nan
    # variância de Q ponderada pelas visitas
    if idx:
        w = pi[idx]
        qv = np.array([qs[i] for i in idx])
        mean = (w * qv).sum() / w.sum()
        x["S_nc1"] = float((w * (qv - mean) ** 2).sum() / w.sum())
    else:
        x["S_nc1"] = math.nan
    return x


def describe(profile, nmin):
    """Descritores de um perfil, com as grandezas da raiz e as somas ao longo da variante principal."""
    nodes = [parse_node(item) for item in profile.split(";")]
    per_node = [node_quantities(children) for _, children in nodes]
    # nós com pelo menos nmin visitas
    eligible = [t for t, (visits, _) in enumerate(nodes) if visits >= nmin]

    # t par = solucionador, t ímpar = adversário
    out = {}
    for name in per_node[0]:
        out[f"{name}_root"] = per_node[0][name]
        for label, parity in (("even", 0), ("odd", 1), ("pv", None)):
            terms = [per_node[t][name] for t in eligible
                     if (parity is None or t % 2 == parity) and not math.isnan(per_node[t][name])]
            out[f"{name}_sum_{label}"] = sum(terms) if terms else math.nan
    out = {k: out[k] for k in SAIDAS}
    # T = nós do perfil; T_even e T_odd = nós elegíveis em t par e em t ímpar
    out["T"] = len(nodes)
    out["T_even"] = sum(1 for t in eligible if t % 2 == 0)
    out["T_odd"] = sum(1 for t in eligible if t % 2 == 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nmin", type=int, default=1)
    ap.add_argument("csv", nargs="+", type=Path)
    args = ap.parse_args()

    for path in args.csv:
        # orçamento pelo nome do arquivo (data_<orçamento>.csv)
        budget = path.stem.split("_")[-1]
        df = pd.read_csv(path, usecols=["puzzle_id", "rating", "instrumental_profile"])
        rows = [describe(p, args.nmin) for p in df["instrumental_profile"]]
        out = pd.concat([df[["puzzle_id", "rating"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
        out_dir = path.parent / "descritores"
        out_dir.mkdir(exist_ok=True)
        target = out_dir / f"descritores_{budget}.csv"
        out.to_csv(target, index=False)
        print(f"{path.name}: {len(out)} puzzles, {out.shape[1] - 2} descritores -> {target}", flush=True)


if __name__ == "__main__":
    main()
