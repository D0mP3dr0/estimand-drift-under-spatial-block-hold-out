#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check R5: variance of a single split draw estimated from its own test blocks only.

Reads the partial records of the shared loop (analysis/v3_13_laco_comum.py ->
fase5/_parcial_laco/<cell>.json: per test block, the loss sum S_B and the scored-node count M_B, per
draw, predictor and population) and computes, for each of the 16 cells, each of the 60 draws, each
predictor (constant; calibrated FSPL (b)) and each population (valid nodes; all nodes):

  Err = sum_B S_B / M,  M = sum_B M_B,  k = number of test blocks with M_B > 0
  v   = (1 - k/N) * k/(k-1) * sum_B (S_B - Err*M_B)^2 / M^2      (defined for k >= 2)

and, per cell, predictor and population: Q = mean of v / between-draw sample variance (ddof = 1) of
Err; coverage = fraction of draws in which Err +/- 2 sqrt(v) contains the mean of Err over the draws;
the number of undefined draws (k < 2); and mechanical counts for the thresholds of the decision
criterion criteria/criterio_R5_variancia_um_sorteio.json (written before the run).

Readings, all computed and labelled (the primary one is flagged in the output):
  N: "N_ocupados" (primary) = number of 10 km blocks holding at least one node of the cell;
     "N_bbox" = number of cells of the bounding rectangle of the block grid; "sem_fpc" = factor
     (1 - k/N) omitted.
  Draw set: reading "A" (primary) takes the variance and mean of Err over all draws with Err defined
     (M > 0) and the mean of v and the coverage over the draws with v defined (k >= 2); reading "B"
     restricts everything to the draws with v defined.
  The joint threshold "Q in [0.5, 2] and coverage >= 0.80 in >= 12 of 16 cells" is counted both as a
     per-cell conjunction and as "coverage >= 0.80 in >= 12 cells and median Q in [0.5, 2]".
Cells where Q or the coverage is undefined count as not meeting a threshold and are counted apart.

Requires fase5/R5R6_validacao_laco.json to report that the loop reproduces the stored records.
Outputs: fase5/R5_por_sorteio.json and fase5/R5_resumo.json. CPU only, deterministic.
Usage: python analysis/v3_13_R5_variancia_um_sorteio.py --consolidar
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
SCRIPT_SHA256 = hashlib.sha256(HERE.read_bytes()).hexdigest()
LACO = HERE.parent / "v3_13_laco_comum.py"
LACO_SHA256 = hashlib.sha256(LACO.read_bytes()).hexdigest()
BASE = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
V3 = BASE / "_v3_2026-09-25"
OUT = V3 / "fase5"
PARCIAL_DIR = OUT / "_parcial_laco"
VALID = OUT / "R5R6_validacao_laco.json"
CRITERIO = V3 / "criterios" / "criterio_R5_variancia_um_sorteio.json"
F2_JSON = V3 / "fase2" / "2.1_deriva_erro_baselines_16x60rnd.json"
OUT_SORTEIO = OUT / "R5_por_sorteio.json"
OUT_RESUMO = OUT / "R5_resumo.json"
CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
QUADRANTES = ["Q1", "Q2", "Q3", "Q4"]
PREDS = ("constante", "fspl_b")
POPS = ("validos", "todos")
VARIANTES_N = ("N_ocupados", "N_bbox", "sem_fpc")
LEITURAS = ("A", "B")


def agora() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimador(S: np.ndarray, M: np.ndarray, N: float | None) -> dict:
    """One-draw estimate from the per-block sums S and counts M of the test blocks with M_B > 0.
    N = None omits the finite-population factor (1 - k/N)."""
    k = int(M.size)
    Mt = float(M.sum())
    if k == 0 or Mt <= 0:
        return {"k": k, "M": 0, "Err": None, "v": None, "definido": False}
    Err = float(S.sum() / Mt)
    if k < 2:
        return {"k": k, "M": int(Mt), "Err": Err, "v": None, "definido": False}
    fpc = 1.0 if N is None else (1.0 - k / N)
    v = fpc * k / (k - 1.0) * float(np.sum((S - Err * M) ** 2)) / (Mt * Mt)
    return {"k": k, "M": int(Mt), "Err": Err, "v": v, "definido": True}


def resumo_celula(regs: list, leitura: str) -> dict:
    """Q, coverage and counts of one cell from its per-draw estimates (one N variant), under reading A or B."""
    E = [r for r in regs if r["Err"] is not None]
    D = [r for r in regs if r["definido"]]
    ref = E if leitura == "A" else D
    err_ref = np.array([r["Err"] for r in ref], dtype=np.float64)
    out = {"n_sorteios": len(regs), "n_Err_definido": len(E), "n_v_definido": len(D),
           "n_indefinidos_k_lt_2": len(regs) - len(D),
           "n_sem_no_pontuado_k0": sum(1 for r in regs if r["k"] == 0),
           "n_k_igual_1": sum(1 for r in regs if r["k"] == 1),
           "k_mediano": float(np.median([r["k"] for r in regs])) if regs else None,
           "k_min": int(min(r["k"] for r in regs)), "k_max": int(max(r["k"] for r in regs))}
    if len(err_ref) >= 2:
        var_emp = float(np.var(err_ref, ddof=1))
        media_err = float(np.mean(err_ref))
    else:
        var_emp, media_err = None, (float(err_ref[0]) if len(err_ref) else None)
    out["var_empirica_Err_entre_sorteios"] = var_emp
    out["media_Err_entre_sorteios"] = media_err
    if D:
        vs = np.array([r["v"] for r in D])
        errs_D = np.array([r["Err"] for r in D])
        out["media_v"] = float(np.mean(vs))
        out["media_sqrt_v"] = float(np.mean(np.sqrt(vs)))
        if media_err is not None:
            cob = np.abs(errs_D - media_err) <= 2.0 * np.sqrt(vs)
            out["n_cobertos"] = int(cob.sum())
            out["cobertura"] = float(cob.mean())
        else:
            out["n_cobertos"], out["cobertura"] = None, None
    else:
        out["media_v"], out["media_sqrt_v"], out["n_cobertos"], out["cobertura"] = None, None, None, None
    out["dp_empirico_Err"] = float(np.sqrt(var_emp)) if var_emp is not None else None
    out["Q"] = (out["media_v"] / var_emp) if (var_emp not in (None, 0.0) and out["media_v"] is not None) else None
    return out


def contagens(por_celula: dict) -> dict:
    """Counts of cells meeting each threshold of the decision criterion, under both readings."""
    cels = list(por_celula)
    Q = {c: por_celula[c]["Q"] for c in cels}
    C = {c: por_celula[c]["cobertura"] for c in cels}
    em_faixa = [c for c in cels if Q[c] is not None and 0.5 <= Q[c] <= 2.0]
    cob80 = [c for c in cels if C[c] is not None and C[c] >= 0.80]
    cob60 = [c for c in cels if C[c] is not None and C[c] < 0.60]
    ambos = [c for c in em_faixa if c in cob80]
    qs = [Q[c] for c in cels if Q[c] is not None]
    q_med = float(np.median(qs)) if qs else None
    return {
        "n_celulas": len(cels),
        "n_celulas_Q_indefinido": sum(1 for c in cels if Q[c] is None),
        "n_celulas_cobertura_indefinida": sum(1 for c in cels if C[c] is None),
        "n_celulas_Q_em_0.5_2": len(em_faixa),
        "n_celulas_cobertura_ge_0.80": len(cob80),
        "n_celulas_cobertura_lt_0.60": len(cob60),
        "n_celulas_Q_em_faixa_E_cobertura_ge_0.80": len(ambos),
        "celulas_Q_em_faixa_E_cobertura_ge_0.80": ambos,
        "celulas_cobertura_lt_0.60": cob60,
        "Q_mediano_entre_celulas": q_med,
        "Q_minimo": float(min(qs)) if qs else None, "Q_maximo": float(max(qs)) if qs else None,
        "limiar_REFORCA_leitura1_conjuncao_por_celula_ge_12": len(ambos) >= 12,
        "limiar_REFORCA_leitura2_cob_ge_0.80_em_ge_12_e_Q_mediano_em_faixa":
            (len(cob80) >= 12 and q_med is not None and 0.5 <= q_med <= 2.0),
        "limiar_DELIMITA_cobertura_lt_0.60_em_ge_8": len(cob60) >= 8,
    }


def consolidar():
    val = json.loads(VALID.read_text(encoding="utf-8"))
    if not val.get("reproduz"):
        raise SystemExit("ABORTA: validacao do laco nao reproduz o parcial gravado")
    f2 = json.loads(F2_JSON.read_text(encoding="utf-8"))
    seeds = [int(s) for s in f2["nota_divergencia_seeds"]["seeds_usados_nesta_rodada"]]
    celulas = [f"{c}_{q}" for c in CIDADES for q in QUADRANTES]
    crit = json.loads(CRITERIO.read_text(encoding="utf-8"))

    por_sorteio = {}
    resumo_cel = {}  # indexed [pred][pop][N variant][reading][cell]
    meta_cel = {}
    parciais_sha = {}
    for ch in celulas:
        arq = PARCIAL_DIR / f"{ch}.json"
        r = json.loads(arq.read_text(encoding="utf-8"))
        if r.get("status") != "ok":
            raise SystemExit(f"parcial {ch} status {r.get('status')}")
        parciais_sha[ch] = hashlib.sha256(arq.read_bytes()).hexdigest()
        assert [s["split_seed"] for s in r["por_sorteio"]] == seeds
        Ns = {"N_ocupados": float(r["N_blocos_ocupados"]), "N_bbox": float(r["N_blocos_bbox"]), "sem_fpc": None}
        meta_cel[ch] = {"N_blocos_ocupados": r["N_blocos_ocupados"], "N_blocos_bbox": r["N_blocos_bbox"],
                        "n_nodes_total": r["n_nodes_total"], "sha256_manifest_tensor": r["sha256_manifest"],
                        "sha256_parcial_laco": parciais_sha[ch]}
        por_sorteio[ch] = {"N_blocos_ocupados": r["N_blocos_ocupados"], "N_blocos_bbox": r["N_blocos_bbox"],
                           "sorteios": []}
        regs = {(p, o, vn): [] for p in PREDS for o in POPS for vn in VARIANTES_N}
        for s in r["por_sorteio"]:
            linha = {"split_seed": s["split_seed"], "n_test": s["n_test"], "n_blocos_teste_com_no": s["n_blocos_teste_com_no"]}
            for p in PREDS:
                for o in POPS:
                    x = s["R5"][p][o]
                    S, M = np.array(x["S"], dtype=np.float64), np.array(x["M"], dtype=np.float64)
                    est = {vn: estimador(S, M, Ns[vn]) for vn in VARIANTES_N}
                    for vn in VARIANTES_N:
                        regs[(p, o, vn)].append(est[vn])
                    prim = est["N_ocupados"]
                    linha[f"{p}__{o}"] = {
                        "blocos": x["blocos"], "S_B": x["S"], "M_B": x["M"], "k": prim["k"], "M": prim["M"],
                        "Err": prim["Err"], "mae_orig_script_2.1": x["mae_orig"],
                        "v_N_ocupados": prim["v"], "v_N_bbox": est["N_bbox"]["v"], "v_sem_fpc": est["sem_fpc"]["v"],
                        "erro_padrao_N_ocupados": (float(np.sqrt(prim["v"])) if prim["v"] is not None else None)}
            por_sorteio[ch]["sorteios"].append(linha)
        for (p, o, vn), lst in regs.items():
            for lt in LEITURAS:
                resumo_cel.setdefault(p, {}).setdefault(o, {}).setdefault(vn, {}).setdefault(lt, {})[ch] = \
                    resumo_celula(lst, lt)

    carimbo = {"script": str(HERE), "script_sha256": SCRIPT_SHA256, "laco_comum": str(LACO),
               "laco_comum_sha256": LACO_SHA256,
               "script_original_reusado": "scripts/v3_2.1_3.1_deriva_calibracao.py",
               "script_original_sha256": val["script_original_sha256"],
               "sementes": seeds, "sementes_fonte": str(F2_JSON), "data_utc": agora(),
               "comando": " ".join(sys.argv), "criterio": str(CRITERIO), "venv": sys.executable,
               "validacao_do_laco": {"arquivo": str(VALID), "reproduz": val["reproduz"],
                                      "celulas": {k: {"reproduz": v["reproduz"],
                                                      "n_numeros_comparados_exato": v["n_numeros_comparados_exato"],
                                                      "max_abs_diferenca_mae_orig": v["max_abs_diferenca_mae_orig"],
                                                      "max_abs_diferenca_Err_soma_vs_gravado": v["max_abs_diferenca_Err_soma_vs_gravado"]}
                                                  for k, v in val["celulas"].items()},
                                      "sha256_arquivo": hashlib.sha256(VALID.read_bytes()).hexdigest()}}
    OUT_SORTEIO.write_text(json.dumps({
        **carimbo, "id": "R5_por_sorteio", "buffer_km": 2.0, "grid_km": 10.0,
        "estimador": "v = (1 - k/N) * k/(k-1) * sum_B (S_B - Err*M_B)^2 / M^2; Err = sum S_B / M",
        "campos": "por celula/sorteio e por preditor__populacao: blocos de teste com no pontuado, S_B (soma das perdas |pred-rssi|, dB), "
                  "M_B, k, M, Err, v (N = blocos ocupados [primaria]; N = retangulo envolvente; sem fpc), erro-padrao",
        "celulas": por_sorteio}, indent=1, ensure_ascii=False), encoding="utf-8")

    resumo = {}
    for p in PREDS:
        for o in POPS:
            for vn in VARIANTES_N:
                for lt in LEITURAS:
                    bloco = resumo_cel[p][o][vn][lt]
                    resumo.setdefault(f"{p}__{o}", {}).setdefault(vn, {})[f"leitura_{lt}"] = {
                        "primaria": (vn == "N_ocupados" and lt == "A"),
                        "contagens_dos_limiares": contagens(bloco),
                        "por_celula": bloco}
    saida = {**carimbo, "id": "R5_resumo", "buffer_km": 2.0, "grid_km": 10.0,
             "leitura_do_criterio_texto": crit["leitura"], "previsao_texto": crit["previsao"],
             "definicoes": {
                 "Err": "sum_B S_B / M no teste do sorteio (MAE da populacao; igual ao mae do script 2.1)",
                 "Q": "media de v sobre os sorteios com k >= 2 / variancia empirica (ddof=1) entre sorteios de Err",
                 "cobertura": "fracao dos sorteios com v definido em que |Err - media(Err)| <= 2 sqrt(v)",
                 "leitura_A": "variancia empirica e media de Err sobre todos os sorteios com Err definido (M > 0) [primaria]",
                 "leitura_B": "variancia empirica e media de Err sobre os sorteios com v definido (k >= 2)",
                 "N_ocupados": "blocos de 10 km com >= 1 no na celula [primaria]",
                 "N_bbox": "celulas do retangulo envolvente da malha de 10 km",
                 "sem_fpc": "fator (1 - k/N) omitido"},
             "meta_celulas": meta_cel,
             "resumo": resumo,
             "interpretacao": "NAO feita neste artefato (fora do escopo do pipeline)"}
    OUT_RESUMO.write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"gravado {OUT_SORTEIO} e {OUT_RESUMO}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consolidar", action="store_true", required=True)
    ap.parse_args()
    consolidar()


if __name__ == "__main__":
    main()
