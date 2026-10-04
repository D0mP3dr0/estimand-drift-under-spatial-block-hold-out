#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Number of 10 km blocks of each cell that contain at least one valid (non-sentinel) node of the received-power target.

A node is valid when it is not a sentinel in the shared loop of analysis/v3_13_laco_comum.py (target path loss
below 299 dB), and the block id is the one of that loop (g = 10 km). The coincidence of the sentinel mask with
RSSI == -110 dBm is only recorded as a check. Per cell the record gives the occupied blocks, the blocks with a valid
node, their fraction and the node counts; the occupied-block count is checked against `meta_celulas` of
fase5/R5_resumo.json (132 in every cell). Summary ranges are given over the 16 cells and over the four Q1 cells.

Reuses by import, unchanged and with a digest check, analysis/v3_13_laco_comum.py (carregar_celula). Input: the
reference-field tensors (data on request). Output: fase5/blocos_com_no_valido.json. No random numbers; CPU only.
Usage: python analysis/v3_14_blocos_com_no_valido.py
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
SCRIPT_SHA256 = hashlib.sha256(HERE.read_bytes()).hexdigest()
BASE = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
V3 = BASE / "_v3_2026-09-25"
LACO_PATH = BASE / "scripts" / "v3_13_laco_comum.py"
LACO_SHA256 = "d6e7c2c6f5f0bb0af5f12c1fe3cbe47235e85347665f171b3fd7cd41fe813b93"
R5_RESUMO = V3 / "fase5" / "R5_resumo.json"
OUT = V3 / "fase5" / "blocos_com_no_valido.json"
CELULAS = [f"{c}_{q}" for c in ("bauru", "campinas", "lins", "sorocaba") for q in ("Q1", "Q2", "Q3", "Q4")]
Q1 = ["bauru_Q1", "campinas_Q1", "lins_Q1", "sorocaba_Q1"]

if hashlib.sha256(LACO_PATH.read_bytes()).hexdigest() != LACO_SHA256:
    raise SystemExit("ABORTA: sha do laco_comum diverge")
_spec = importlib.util.spec_from_file_location("v3_13_laco_comum", LACO_PATH)
laco = importlib.util.module_from_spec(_spec)
sys.modules["v3_13_laco_comum"] = laco
_spec.loader.exec_module(laco)
orig = laco.orig


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def por_celula(chave: str) -> dict:
    t0 = time.perf_counter()
    c = laco.carregar_celula(chave)
    gid, sent, rssi = c["gid"], c["sent"], c["rssi"]
    valido = ~sent
    ocupados = np.unique(gid)
    com_valido = np.unique(gid[valido])
    n_ocup, n_val = int(ocupados.size), int(com_valido.size)
    s110 = rssi == -110.0
    return {"celula": chave, "N_blocos_ocupados": n_ocup, "N_blocos_bbox_laco": c["N_blocos_bbox"],
            "blocos_com_ao_menos_1_no_valido": n_val,
            "fracao_blocos_com_no_valido": n_val / n_ocup,
            "blocos_sem_no_valido": n_ocup - n_val,
            "n_nos": int(rssi.size), "n_nos_validos": int(valido.sum()), "n_nos_sentinela": int(sent.sum()),
            "conferencia_registro": {"sentinela_e_rssi_igual_menos110": int((sent & s110).sum()),
                                      "sentinela_e_rssi_diferente_de_menos110": int((sent & ~s110).sum()),
                                      "valido_e_rssi_igual_menos110": int((valido & s110).sum())},
            "sha256_manifest_tensor": c["sha256_manifest"], "tempo_s": time.perf_counter() - t0}


def faixa(v):
    a = np.asarray(v, dtype=float)
    return {"n": int(a.size), "min": float(a.min()), "mediana": float(np.median(a)), "max": float(a.max())}


def main():
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(por_celula, CELULAS))
    r5 = json.loads(R5_RESUMO.read_text(encoding="utf-8"))["meta_celulas"]
    conf = {r["celula"]: {"N_blocos_ocupados_aqui": r["N_blocos_ocupados"],
                          "N_blocos_ocupados_R5": r5[r["celula"]]["N_blocos_ocupados"],
                          "reproduz": r["N_blocos_ocupados"] == r5[r["celula"]]["N_blocos_ocupados"] == 132}
            for r in res}
    reproduz = all(v["reproduz"] for v in conf.values())
    por = {r["celula"]: r for r in res}
    todas = [por[c]["blocos_com_ao_menos_1_no_valido"] for c in CELULAS]
    q1 = [por[c]["blocos_com_ao_menos_1_no_valido"] for c in Q1]
    saida = {
        "id": "blocos_com_no_valido_por_celula", "pergunta": "blocos de 10 km com >= 1 no valido (nao sentinela) por celula",
        "definicao_no_valido": "~sent do laco comum (sent = terrain.y[:,0] >= 299.0), nao comparacao de valor de RSSI",
        "grid_km": orig.GRID_KM, "id_de_bloco": "gid = SpatialKFold._assign_groups(pos_km) (laco comum)",
        "conferencia_N_blocos_ocupados_vs_R5": {"fonte": str(R5_RESUMO), "fonte_sha256": sha256_file(R5_RESUMO),
                                                 "reproduz_132_nas_16": reproduz, "por_celula": conf},
        "por_celula": por,
        "faixa_16_celulas": {"blocos_com_no_valido": faixa(todas),
                             "fracao": faixa([por[c]["fracao_blocos_com_no_valido"] for c in CELULAS])},
        "faixa_4_celulas_Q1": {"celulas": Q1, "blocos_com_no_valido": faixa(q1),
                               "fracao": faixa([por[c]["fracao_blocos_com_no_valido"] for c in Q1])},
        "script": str(HERE), "script_sha256": SCRIPT_SHA256, "laco_importado": str(LACO_PATH), "laco_sha256": LACO_SHA256,
        "script_original_sha256": laco.ORIG_SHA256, "comando": " ".join(sys.argv) or str(HERE),
        "data_utc": datetime.now(timezone.utc).isoformat(), "venv": sys.executable,
        "sementes": "nao se aplica (sem sorteio)", "tempo_total_s": time.perf_counter() - t0}
    OUT.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"reproduz_132={reproduz} -> {OUT}", flush=True)
    sys.exit(0 if reproduz else 2)


if __name__ == "__main__":
    main()
