#!/usr/bin/env python3
"""Content of the 17 columns of terrain.features_raw in the 16 cells of the dataset.

For every cell (city x quadrant) the script memory-maps the reference-field tensor
transfer_dataset_<city>_v19_<quadrant>_enriched_cftudo.pt and records, per column of
terrain.features_raw, the minimum, maximum, fraction of non-zero entries and whether
the column is constant. Columns 14 to 16 are the lidar columns written by
the graph builder, which is not part of this package (has_lidar, z_lidar, confidence); the
summary reports their non-zero fractions across cells, whether column 16 is identically
zero everywhere, and the smallest non-zero fraction among columns 0 to 13.

Input: the 16 *_enriched_cftudo.pt tensors. Output: <base>/fase3/3.9_colunas_features_raw.json
(per cell and summary), where <base> is the folder _v3_2026-09-25 next to the script's
parent folder (results/ in this repository). Deterministic.

Usage: python v3_3.9_colunas_lidar.py
"""
import json, hashlib, numpy as np, torch
from pathlib import Path
D = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")
OUT = Path(__file__).resolve().parent.parent / "_v3_2026-09-25/fase3/3.9_colunas_features_raw.json"
res = {}
for cid in ("bauru", "campinas", "lins", "sorocaba"):
    for q in ("Q1", "Q2", "Q3", "Q4"):
        p = D / f"transfer_dataset_{cid}_v19_{q}_enriched_cftudo.pt"
        f = np.asarray(torch.load(p, mmap=True, weights_only=False)["terrain"].features_raw)
        cols = {}
        for c in range(f.shape[1]):
            x = f[:, c].astype(np.float64)
            cols[c] = {"min": float(x.min()), "max": float(x.max()), "frac_nao_zero": float((x != 0).mean()),
                       "constante": bool(x.min() == x.max())}
        res[f"{cid}_{q}"] = {"arquivo": p.name, "n_nos": int(f.shape[0]), "n_colunas": int(f.shape[1]), "colunas": cols}
        print(cid, q, [round(cols[c]["frac_nao_zero"], 4) for c in (14, 15, 16)], flush=True)
fr14 = [v["colunas"][14]["frac_nao_zero"] for v in res.values()]
resumo = {"n_celulas": len(res),
          "col14_frac_nao_zero_min_max": [min(fr14), max(fr14)],
          "col15_frac_nao_zero_min_max": [min(v["colunas"][15]["frac_nao_zero"] for v in res.values()), max(v["colunas"][15]["frac_nao_zero"] for v in res.values())],
          "col16_constante_zero_em_todas": all(v["colunas"][16]["constante"] and v["colunas"][16]["max"] == 0 for v in res.values()),
          "colunas_0_13_frac_nao_zero_min": min(v["colunas"][c]["frac_nao_zero"] for v in res.values() for c in range(14))}
OUT.write_text(json.dumps({"script": __file__, "resumo": resumo, "por_celula": res}, indent=1, ensure_ascii=False))
print(json.dumps(resumo, indent=1))
