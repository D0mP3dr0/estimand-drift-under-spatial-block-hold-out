#!/usr/bin/env python3
"""Per-cell summary of the between-draw variance decomposition of the constant predictor's error.

The input record fase3/3.7_resposta_cego.json (block "F2") holds, for each of the 16
cells, a decomposition of the variance across split draws of the constant predictor's
test MAE over all nodes, Z = X * Y with X = valid-node fraction of the test partition
and Y = MAE on valid nodes, into a composition term mean(Y)^2 var(X), a conditional
term mean(X)^2 var(Y) and a remainder (interaction), each as a fraction of var(Z).
This script does not recompute anything: it counts the cells where composition exceeds
one half, lists the composition share elsewhere, its range, the cells with a share
above 1 or a negative remainder, the sum of the three median shares, and the range of
draws used per cell.

Input: <base>/fase3/3.7_resposta_cego.json. Output: <base>/fase3/3.7b_decomposicao_por_celula.json
(also printed), where <base> is the folder _v3_2026-09-25 next to the script's parent
folder (results/ in this repository). Deterministic.

Usage: python v3_3.7b_decomposicao_por_celula.py
"""
import json
from pathlib import Path
B = Path(__file__).resolve().parent.parent / "_v3_2026-09-25"
f2 = json.load(open(B / "fase3/3.7_resposta_cego.json"))["F2"]
pc = f2["por_celula"]
comp = {k: v["frac_composicao"] for k, v in pc.items()}
res = {"n_celulas": len(pc),
       "n_celulas_composicao_acima_de_meio": sum(c > 0.5 for c in comp.values()),
       "composicao_nas_demais": sorted(round(c, 3) for c in comp.values() if c <= 0.5),
       "composicao_min_max": [min(comp.values()), max(comp.values())],
       "celulas_composicao_acima_de_1": {k: round(c, 3) for k, c in comp.items() if c > 1},
       "celulas_resto_negativo": {k: round(v["frac_resto"], 3) for k, v in pc.items() if v["frac_resto"] < 0},
       "soma_das_tres_medianas": f2["mediana_frac_composicao"] + f2["mediana_frac_condicional"] + f2["mediana_frac_resto"],
       "n_sorteios_usados_min_max": [min(v["n_sorteios"] for v in pc.values()), max(v["n_sorteios"] for v in pc.values())]}
(B / "fase3/3.7b_decomposicao_por_celula.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps(res, indent=1))
