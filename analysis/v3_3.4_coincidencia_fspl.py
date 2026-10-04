#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is the agreement between the real FSPL calibrated median and a simulated band edge chance?

The FSPL calibrated transmit power of Lins Q1 on the training partition (20.4026 dBm,
baselines_analiticos.train.p_tx_eff_dbm.fspl of the frozen run JSON) agrees to the
fourth decimal with the upper edge of the median-calibration band obtained from a
simulated distance distribution (20.4022 dBm: FSPL at 900 MHz, d ~ U[1, 140] km,
sentinel fraction 0.8), read from the symbolic derivation record fismat_sympy_eqR.json.
That record is only read; the script that produced it is hashed for citation and never
executed. The Friis free-space formula is reimplemented here.

Two lines of evidence, fixed in the decision criterion before the run:
  (A) Monte Carlo grid: 200 seeds x 5 distance intervals fixed a priori, N = 200,000
      nodes per simulation, sentinel fraction 0.8. Sentinel nodes get W = -110 dBm +
      FSPL(d), d ~ U[d_min, d_max]; valid nodes get W ~ Normal(43, 10) dBm (irrelevant
      to the median while the sentinel fraction exceeds 1/2). The output is the fraction
      of (seed, interval) pairs whose simulated median falls within 0.001 dB of the real
      value, an empirical probability of a chance coincidence.
  (B) Recomputation of the band edges for Lins Q1 with the REAL distances of the training
      sentinel nodes and the real training sentinel fraction (same definition as the
      band-edge test v3_3.2_banda_p7_bordas_reais.py, loaded independently here).

Inputs: the Lins Q1 reference-field tensor, its frozen run JSON, fismat_sympy_eqR.json,
criterio_3.4.json, and the sibling script v3_2.1_3.1_deriva_calibracao.py (split and
tensor loader, imported unchanged).
Output: results/fase3/3.4_coincidencia_fspl.json.
Seeds: part A uses numpy default_rng(seed * 1000 + int(10 * d_min)), seed = 0..199;
part B is deterministic (split seed 42).

Usage: python v3_3.4_coincidencia_fspl.py --out <json>
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SCRIPT_PATH = Path(__file__).resolve()
TENSOR_LINS_Q1 = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3/"
                      "transfer_dataset_lins_v19_Q1_enriched_cftudo.pt")
RUN_LINS_Q1 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
                   "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/dados/"
                   "treinos_c1/run_c0c1cf_lins_s42_Q1_g10b2.json")
FISMAT_JSON = Path("internal/fismat_sympy_eqR.json")
FISMAT_SCRIPT = Path("internal/fismat_sympy_eqR.py")
SIBLING_SCRIPT = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts/"
                      "v3_2.1_3.1_deriva_calibracao.py")

OUT_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
OUT_JSON = OUT_DIR / "fase3" / "3.4_coincidencia_fspl.json"
CRIT_PATH = OUT_DIR / "criterios" / "criterio_3.4.json"

FREQ_MHZ = 900.0
# RSSI value carried by sentinel nodes (dBm).
CP_SENTINEL_DBM = -110.0
PL_TARGET_MAX_VALID = 299.0
GRID_KM, BUFFER_KM, FRACS, SPLIT_SEED = 10.0, 2.0, (0.70, 0.15, 0.15), 42

P_TX_EFF_FSPL_REAL_DBM = 20.402631103093142  # run JSON, baselines_analiticos.train.p_tx_eff_dbm.fspl
LIMIAR_COINCIDENCIA_DB = 0.001

# Grid fixed in the decision criterion before the run: 5 distance intervals (km).
D_INTERVALOS_KM = [(1.0, 140.0), (0.5, 150.0), (1.0, 120.0), (1.0, 160.0), (5.0, 140.0)]
N_SEEDS = 200
N_TOTAL_MC = 200_000
PI_BAR_MC = 0.8
PTX_NOMINAL_VALIDO = 43.0
SIGMA_VALIDO = 10.0


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def fspl_db(d_km: np.ndarray, f_mhz: float = FREQ_MHZ) -> np.ndarray:
    """Friis free-space path loss in dB, d in km and f in MHz (distance floored at 1 m).

    Same form as free_space_path_loss in training/baselines/empirical_models.py.
    """
    d_km = np.maximum(np.asarray(d_km, dtype=np.float64), 0.001)
    return 20.0 * np.log10(d_km) + 20.0 * np.log10(f_mhz) + 32.44


def parte_A_grade_seeds_intervalos() -> dict:
    """Part (A): simulated medians over the seed x interval grid and the hit rate against the real value."""
    combos = []
    hits_vs_real = 0
    hits_vs_toy_original = 0
    toy_original = None
    for d_min, d_max in D_INTERVALOS_KM:
        # Deterministic upper edge: analytical quantile of U[d_min, d_max] at level 1/(2 pi_bar).
        beta = 1.0 / (2.0 * PI_BAR_MC)
        d_hi = d_min + (d_max - d_min) * beta
        borda_superior_teorica = CP_SENTINEL_DBM + fspl_db(d_hi)
        if (d_min, d_max) == (1.0, 140.0):
            toy_original = borda_superior_teorica
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed * 1000 + int(d_min * 10))
            n_sent = int(round(PI_BAR_MC * N_TOTAL_MC))
            n_valid = N_TOTAL_MC - n_sent
            d_sent_km = rng.uniform(d_min, d_max, n_sent)
            w_sent = CP_SENTINEL_DBM + fspl_db(d_sent_km)
            w_valid = rng.normal(PTX_NOMINAL_VALIDO, SIGMA_VALIDO, n_valid)
            w_all = np.concatenate([w_sent, w_valid])
            mediana_mc = float(np.median(w_all))
            hit_real = abs(mediana_mc - P_TX_EFF_FSPL_REAL_DBM) <= LIMIAR_COINCIDENCIA_DB
            if hit_real:
                hits_vs_real += 1
            combos.append({
                "d_min_km": d_min, "d_max_km": d_max, "seed": seed,
                "mediana_mc_dBm": mediana_mc,
                "borda_superior_teorica_dBm": borda_superior_teorica,
                "hit_vs_real_20_4026": bool(hit_real),
            })
    n_combos = len(combos)
    prob_coincidencia = hits_vs_real / n_combos
    return {
        "n_seeds": N_SEEDS, "n_intervalos": len(D_INTERVALOS_KM),
        "n_total_combinacoes": n_combos, "N_por_simulacao": N_TOTAL_MC,
        "pi_bar_usado": PI_BAR_MC,
        "intervalos_km": D_INTERVALOS_KM,
        "borda_superior_teorica_por_intervalo_dBm": {
            f"{d_min}-{d_max}": CP_SENTINEL_DBM + fspl_db(d_min + (d_max - d_min) / (2 * PI_BAR_MC))
            for d_min, d_max in D_INTERVALOS_KM
        },
        "hits_coincidencia_le_0_001dB_vs_real": hits_vs_real,
        "probabilidade_empirica_coincidencia_por_acaso": prob_coincidencia,
        "toy_original_U1_140_borda_superior_teorica_dBm": toy_original,
        "divergencia_toy_original_vs_real_dB": abs(toy_original - P_TX_EFF_FSPL_REAL_DBM),
        "amostra_combos": combos[:10],
    }


def parte_B_recomputo_dado_real() -> dict:
    """Part (B): FSPL band edges and calibrated median for Lins Q1 from real training distances."""
    if not TENSOR_LINS_Q1.exists() or not RUN_LINS_Q1.exists():
        return {"status": "nao_verificado", "motivo": "tensor ou run JSON de Lins Q1 ausente"}

    sys.path.insert(0, str(SIBLING_SCRIPT.parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("v3_2_1_3_1_b", SIBLING_SCRIPT)
    sib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sib)

    rf, modo_carga = sib.carregar_tensor(TENSOR_LINS_Q1)
    ty = torch.as_tensor(rf["terrain"].y).float().clone().numpy()
    dist_all = torch.as_tensor(rf["terrain"].dist_nearest_m).float().clone().numpy()
    pos_deg = torch.as_tensor(rf["terrain"].pos).float().clone()
    del rf
    gc.collect()

    pos_m = sib.latlon_graus_para_metros(pos_deg)
    rssi = ty[:, 3].astype(np.float64)
    pl = ty[:, 0].astype(np.float64)
    sentinela_all = pl >= PL_TARGET_MAX_VALID

    parts = sib.split_espacial_3vias(pos_m, GRID_KM, BUFFER_KM, FRACS, SPLIT_SEED)
    tr = parts["train"]
    rssi_tr, dist_tr, sent_tr = rssi[tr], dist_all[tr], sentinela_all[tr]
    pi_bar_real = float(sent_tr.mean())

    pl_pred_tr = fspl_db(dist_tr / 1000.0)  # dist_nearest_m is in metres
    w_all = rssi_tr + pl_pred_tr
    mediana_recomputada = float(np.median(w_all))

    w_sent = w_all[sent_tr]
    beta = 1.0 / (2.0 * pi_bar_real)
    alpha = 1.0 - beta
    q_plus_real_d = float(np.quantile(w_sent, beta, method="higher"))
    q_minus_real_d = float(np.quantile(w_sent, alpha, method="lower"))

    del ty, rssi, pl, dist_all, pos_deg, pos_m, sentinela_all
    gc.collect()

    return {
        "status": "ok",
        "pi_bar_real_treino_lins_Q1": pi_bar_real,
        "mediana_recomputada_dado_real_dBm": mediana_recomputada,
        "mediana_publicada_run_json_dBm": P_TX_EFF_FSPL_REAL_DBM,
        "divergencia_recomputo_vs_publicado_dB": abs(mediana_recomputada - P_TX_EFF_FSPL_REAL_DBM),
        "Q1_mais_dBm_dado_real": q_plus_real_d,
        "Q1_menos_dBm_dado_real": q_minus_real_d,
        "divergencia_Q1_mais_real_vs_valor_publicado_dB": abs(q_plus_real_d - P_TX_EFF_FSPL_REAL_DBM),
        "divergencia_Q1_mais_real_vs_toy_U1_140_pi0_8_dB": None,  # filled in main() from the simulated edge
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    fismat = json.loads(FISMAT_JSON.read_text(encoding="utf-8"))
    borda_superior_toy_fismat = fismat["C_P7"]["pi_0.8"]["borda_superior_dBm"]
    fismat_script_sha256 = hashlib.sha256(FISMAT_SCRIPT.read_bytes()).hexdigest()

    log("parte A: grade de seeds x intervalos...")
    parte_a = parte_A_grade_seeds_intervalos()
    log(f"  probabilidade empirica de coincidencia por acaso = "
        f"{parte_a['probabilidade_empirica_coincidencia_por_acaso']:.5f}")

    log("parte B: recomputo com dado real de Lins Q1...")
    parte_b = parte_B_recomputo_dado_real()
    if parte_b.get("status") == "ok":
        parte_b["divergencia_Q1_mais_real_vs_toy_U1_140_pi0_8_dB"] = abs(
            parte_b["Q1_mais_dBm_dado_real"] - borda_superior_toy_fismat)
    log(f"  parte B status={parte_b.get('status')}")

    coincidencia_se_repete_com_d_real = (
        parte_b.get("status") == "ok"
        and parte_b["divergencia_Q1_mais_real_vs_toy_U1_140_pi0_8_dB"] <= 0.05
    )
    prob_baixa = parte_a["probabilidade_empirica_coincidencia_por_acaso"] < 0.05

    if coincidencia_se_repete_com_d_real:
        veredito = (
            "A borda superior calculada com distancia REAL de Lins Q1 (Q1+ sobre sentinelas de "
            "treino, pi_bar real) reproduz o valor publicado 20,4026 dBm dentro de "
            f"{parte_b['divergencia_Q1_mais_real_vs_toy_U1_140_pi0_8_dB']:.4f} dB do toy U[1,140]km "
            "a pi_bar=0.8 -- a coincidencia SE REPETE com dado real. Explicacao (nao e dependencia "
            "misteriosa): p_tx_eff publicado JA E a mediana de W sobre o treino, e como pi_bar_real "
            "(~0.90) > 1/2, essa mediana cai dentro da banda de P7 perto do lado sentinela -- a "
            "coincidencia decorre diretamente do proprio teorema (P7), nao de uma dependencia "
            "funcional adicional a explicar. A escolha de intervalo U[1,140]km, embora arbitraria, "
            "aproxima a FSPL na regiao de interesse porque o path loss comprime (log-distancia) "
            "distribuicoes de distancia bem diferentes em quantis proximos."
        )
        gravidade = "info"
    else:
        veredito = (
            "A borda com distancia real diverge do toy/valor publicado alem do limiar -- a "
            "coincidencia 20,4026 x 20,4022 nao se sustenta com dado real; tratar como acaso de "
            "escolha do intervalo simulado e rebaixar para nota de rodape."
        )
        gravidade = "alerta"

    if prob_baixa:
        veredito += (f" A probabilidade empirica de uma coincidencia <=0.001 dB por acaso de "
                    f"escolha de seed/intervalo e BAIXA ({parte_a['probabilidade_empirica_coincidencia_por_acaso']:.4f})"
                    ", o que reforca que o valor 20,40 nao e um numero generico obtido com "
                    "qualquer configuracao, mas proximo do especifico do estudo de caso.")
    else:
        veredito += (f" A probabilidade empirica de coincidencia por acaso e "
                    f"{parte_a['probabilidade_empirica_coincidencia_por_acaso']:.4f} (nao desprezivel), "
                    "o que enfraquece a leitura de que 20,40 e um numero especial.")

    saida = {
        "pergunta": "O FSPL real (20,4026) x borda simulada U[1,140]km pi=0.8 (20,4022) e acaso ou dependencia funcional?",
        "alegacao": "Median-calibration band: chance or functional dependence behind the FSPL coincidence",
        "criterio": json.loads(CRIT_PATH.read_text(encoding="utf-8")),
        "metodo": "ver docstring do script; parte A = MC finito com N=200000, 200 seeds x 5 intervalos de d, "
                 "pi_bar=0.8 fixo; parte B = recomputo com d real de Lins Q1 (dist_nearest_m dos nos sentinela "
                 "de treino) e pi_bar_real do treino, mesma definicao do teste 3.2.",
        "insumos": {
            "fismat_sympy_eqR.json": {"caminho": str(FISMAT_JSON), "campo_lido": "C_P7.pi_0.8.borda_superior_dBm",
                                      "valor": borda_superior_toy_fismat},
            "fismat_sympy_eqR.py_sha256_apenas_para_citacao_NAO_executado": fismat_script_sha256,
            "run_c0c1cf_lins_s42_Q1_g10b2.json": {"caminho": str(RUN_LINS_Q1),
                                                  "campo": "baselines_analiticos.train.p_tx_eff_dbm.fspl",
                                                  "valor": P_TX_EFF_FSPL_REAL_DBM},
            "tensor_lins_Q1": str(TENSOR_LINS_Q1),
        },
        "parte_A_grade_seeds_intervalos": parte_a,
        "parte_B_recomputo_dado_real": parte_b,
        "veredito_vs_criterio": veredito,
        "gravidade": gravidade,
        "nao_verificado": [] if parte_b.get("status") == "ok" else ["parte_B: " + parte_b.get("motivo", "")],
        "comando_rodado": f"{sys.executable} {SCRIPT_PATH} --out {out_path}",
        "tempo_total_s": time.perf_counter() - t0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=1)
    log(f"gravado {out_path} ({time.perf_counter()-t0:.1f}s)")
    log(f"veredito: {veredito}")


if __name__ == "__main__":
    main()
