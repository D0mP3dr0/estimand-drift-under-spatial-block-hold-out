#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check R7: spread between split draws versus spread between training seeds in the existing g = 5 km runs.

No training and no GPU. Campaign A3: five training seeds (42 to 46) on block split 42. Campaign A4:
five split draws (split seeds 101 to 105) at training seed 42. Cells bauru_Q1 and bauru_Q3; models
GNN and MLP. Metric: MAE of the received power (channel 3, RSSI) over the valid test nodes (target
path loss below 299 dB). For each cell x model pair the script reports the standard deviation
(ddof = 1) over the five seeds and over the five draws, their ratio, and the mechanical count for the
threshold of the decision criterion criteria/criterio_R7_sementes_x_sorteios_existentes.json
(written before the run).

The per-run JSON files of gpu/A3/ and gpu/A4/ do not hold the received-power MAE restricted to the
valid test nodes (only the all-node test MAE and the path-loss MAE). That value is computed by the
A3/A4 aggregators from each run's prediction file (predicoes_*.npz: target[:, 3] versus pred[:, 3]
where target[:, 0] < 299). Here it is recomputed from the .npz with the same function (mae_pop of
analysis/v3_A4_agregar.py, imported unchanged, digest recorded) and compared with agregado_A3.json
and agregado_A4.json; the run JSON files are read for provenance only (seed, split seed, status,
digest of the test partition). Reference value checked: A4, Bauru Q1, split 101, GNN = 2.3909875 dB.

Output: fase5/R7_resumo.json. Deterministic.
Usage: python analysis/v3_13_R7_sementes_x_sorteios.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
SCRIPT_SHA256 = hashlib.sha256(HERE.read_bytes()).hexdigest()
BASE = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
V3 = BASE / "_v3_2026-09-25"
A3 = V3 / "gpu" / "A3"
A4 = V3 / "gpu" / "A4"
OUT = V3 / "fase5"
OUT_RESUMO = OUT / "R7_resumo.json"
CRITERIO = V3 / "criterios" / "criterio_R7_sementes_x_sorteios_existentes.json"
AGG_PATH = BASE / "scripts" / "v3_A4_agregar.py"
CELULAS = ("bauru_Q1", "bauru_Q3")
MODELOS = ("gnn", "mlp")
SEEDS_A3 = (42, 43, 44, 45, 46)
SPLITS_A4 = (101, 102, 103, 104, 105)
CAMPO = "mae_rssi_validos_db"

_spec = importlib.util.spec_from_file_location("agg_a4", AGG_PATH)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)
AGG_SHA256 = hashlib.sha256(AGG_PATH.read_bytes()).hexdigest()


def agora() -> str:
    return datetime.now(timezone.utc).isoformat()


def ler(pasta: Path):
    """Read the single run JSON and the single prediction .npz of one run folder."""
    js = sorted(pasta.glob("run_*.json"))
    zs = sorted(pasta.glob("predicoes_*.npz"))
    assert len(js) == 1 and len(zs) == 1, f"run JSON/npz ausente ou duplicado em {pasta}"
    dj = json.loads(js[0].read_text(encoding="utf-8"))
    z = np.load(zs[0])
    return dj, {k: z[k] for k in z.files}, js[0], zs[0]


def dp1(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else None


def main():
    crit = json.loads(CRITERIO.read_text(encoding="utf-8"))
    agg3 = json.loads((A3 / "agregado_A3.json").read_text(encoding="utf-8"))
    agg4 = json.loads((A4 / "agregado_A4.json").read_text(encoding="utf-8"))
    pares = {}
    linhas_ref = []
    max_dif_agregado = 0.0
    for cel in CELULAS:
        for mod in MODELOS:
            reg = {"A3_sementes": [], "A4_sorteios": []}
            for lote, pasta_base, ids, tag, ref in (
                    ("A3", A3, SEEDS_A3, "s", agg3["celulas"][cel][f"mae_{mod}_por_seed"]),
                    ("A4", A4, SPLITS_A4, "ss", agg4["celulas"][cel]["por_modelo"][mod]["mae_por_sorteio"])):
                chave_lote = "A3_sementes" if lote == "A3" else "A4_sorteios"
                for i in ids:
                    pasta = pasta_base / f"{mod}_v3_{lote.lower()}_{cel}_{tag}{i}"
                    dj, z, jpath, zpath = ler(pasta)
                    m = agg.mae_pop(z)
                    gravado = ref[str(i)][CAMPO]
                    dif = abs(m[CAMPO] - gravado)
                    max_dif_agregado = max(max_dif_agregado, dif)
                    reg[chave_lote].append({
                        "id": i, "mae_rssi_validos_db": m[CAMPO], "n_validos": m["n_validos"], "n_teste": m["n"],
                        "idx_sha256_teste_do_npz": m["idx_sha256"],
                        "agregado_gravado": gravado, "absdiff_vs_agregado": dif,
                        "run_json": str(jpath), "npz": str(zpath),
                        "run_json_seed": dj.get("seed"), "run_json_split_seed": dj.get("split_seed"),
                        "run_json_status": dj.get("status"),
                        "run_json_grid_km": (dj.get("split") or {}).get("grid_km"),
                        "run_json_buffer_km": (dj.get("split") or {}).get("buffer_km"),
                        "run_json_mae_rssi_db_todos_os_nos_teste": ((dj.get("selecao") or {}).get("test_no_melhor_ckpt") or {}).get("mae_rssi_db")})
            v3_ = [r[CAMPO] for r in reg["A3_sementes"]]
            v4_ = [r[CAMPO] for r in reg["A4_sorteios"]]
            dp_sem, dp_sor = dp1(v3_), dp1(v4_)
            pares[f"{cel}__{mod}"] = {
                "celula": cel, "modelo": mod,
                "dp_entre_5_sementes_A3": dp_sem, "dp_entre_5_sorteios_A4": dp_sor,
                "razao_sorteio_sobre_semente": (dp_sor / dp_sem) if dp_sem else None,
                "media_A3_sementes": float(np.mean(v3_)), "media_A4_sorteios": float(np.mean(v4_)),
                "A3_particao_de_teste_identica_nas_5_sementes": len({r["idx_sha256_teste_do_npz"] for r in reg["A3_sementes"]}) == 1,
                "A4_particoes_de_teste_distintas_nos_5_sorteios": len({r["idx_sha256_teste_do_npz"] for r in reg["A4_sorteios"]}) == 5,
                "corridas": reg}
    razoes = {k: v["razao_sorteio_sobre_semente"] for k, v in pares.items()}
    n_ge2 = sum(1 for r in razoes.values() if r is not None and r >= 2.0)
    saida = {
        "id": "R7_resumo", "script": str(HERE), "script_sha256": SCRIPT_SHA256,
        "agregador_reusado": str(AGG_PATH), "agregador_sha256": AGG_SHA256,
        "agregados_conferidos": {"A3": str(A3 / "agregado_A3.json"),
                                  "A3_sha256": hashlib.sha256((A3 / "agregado_A3.json").read_bytes()).hexdigest(),
                                  "A4": str(A4 / "agregado_A4.json"),
                                  "A4_sha256": hashlib.sha256((A4 / "agregado_A4.json").read_bytes()).hexdigest()},
        "sementes": {"A3_sementes_de_treino_sorteio_42": list(SEEDS_A3), "A4_split_seeds_semente_42": list(SPLITS_A4)},
        "data_utc": agora(), "comando": " ".join(sys.argv), "criterio": str(CRITERIO), "venv": sys.executable,
        "campo": f"{CAMPO} (MAE do canal 3, potencia recebida, nos nos validos do teste), recomputado do .npz de cada corrida",
        "desvio_do_enunciado": "os run JSON nao contem o MAE de potencia recebida nos validos do teste; ver docstring. "
                               "O campo existe em agregado_A3.json/agregado_A4.json (mesmo nome: mae_rssi_validos_db), derivado "
                               "do .npz; recomputado aqui e conferido.",
        "conferencia_validacao": {
            "A4_bauru_Q1_gnn_s101_alvo_enunciado": 2.3909875,
            "A4_bauru_Q1_gnn_s101_obtido": pares["bauru_Q1__gnn"]["corridas"]["A4_sorteios"][0]["mae_rssi_validos_db"],
            "arredondado_7_casas": round(pares["bauru_Q1__gnn"]["corridas"]["A4_sorteios"][0]["mae_rssi_validos_db"], 7),
            "reproduz": round(pares["bauru_Q1__gnn"]["corridas"]["A4_sorteios"][0]["mae_rssi_validos_db"], 7) == 2.3909875,
            "max_absdiff_das_40_corridas_vs_agregados": max_dif_agregado,
            "n_corridas_lidas": sum(len(p["corridas"]["A3_sementes"]) + len(p["corridas"]["A4_sorteios"]) for p in pares.values())},
        "pares": pares,
        "contagem_mecanica_do_limiar": {"razoes": razoes, "n_pares_razao_ge_2": n_ge2, "n_pares": len(razoes),
                                         "REFORCA_se_ge_3_de_4": n_ge2 >= 3, "previsao_todas_4_acima_de_2": n_ge2 == 4},
        "leitura_do_criterio_texto": crit["leitura"],
        "interpretacao": "NAO feita neste artefato (fora do escopo do pipeline); desenho nao cruzado (o cruzado e o teste G1)"}
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_RESUMO.write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"validacao": saida["conferencia_validacao"], "razoes": razoes,
                      "dp": {k: (v["dp_entre_5_sementes_A3"], v["dp_entre_5_sorteios_A4"]) for k, v in pares.items()}}, indent=1))


if __name__ == "__main__":
    main()
