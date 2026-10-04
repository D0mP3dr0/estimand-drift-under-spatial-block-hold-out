#!/usr/bin/env python3
"""Aggregator of model batch A3: GNN versus MLP parity on sentinel nodes.

Batch A3 trained a GNN and a graph-free MLP in 8 cells x 5 training seeds (80 runs)
on the same test partitions. This script aggregates the batch exactly as specified in
the analysis criterion criterio_A3_analise.json, whose SHA-256 is checked first (the
script aborts if the criterion changed). It refuses to run until every planned run has
exit code 0 in the launcher status file, unless --parcial is given, in which case the
output is a structural check marked as partial and not citable.

Order of computation imposed by the criterion: the per-cell tolerance (standard
deviation across seeds of the MLP's sentinel-node RSSI MAE; the GNN's when the MLP's is
below 0.01 dB) is computed from the MLP predictions alone and written to
tolerancias_A3.json BEFORE any GNN prediction is opened. Then, per cell and seed, the
GNN - MLP differences of the RSSI MAE on valid, sentinel and all nodes and of the
path-loss MAE on valid nodes are formed; a cell is in parity when it is complete and
|median sentinel difference| <= its tolerance, and hypothesis H1 reads "approximate
parity" when at least 6 of the 8 cells are in parity. Each run is also checked for
GradScaler skipped steps and minimum scale, the auxiliary channel-1 range rules, the
test-partition hash, and the SHA-256 of its reference field against the artifact manifest.

Inputs: results/gpu/A3/<model>_v3_a3_<city>_<quadrant>_s<seed>/ (run_*.json and
predicoes_*.npz), lote_A3_status.json, criterio_A3.json, criterio_A3_analise.json and
the artifact manifest (group tensores_cftudo).
Outputs: results/gpu/A3/tolerancias_A3.json and agregado_A3.json (agregado_A3_PARCIAL.json
with --parcial). Deterministic given the run files.

Usage: python v3_A3_agregar.py [--parcial] [--status <lote_A3_status.json>]
"""
import argparse
import glob
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

RAIZ = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
A3 = RAIZ / "gpu" / "A3"
CRIT_ANALISE = RAIZ / "criterios" / "criterio_A3_analise.json"
CRIT_LOTE = A3 / "criterio_A3.json"
MANIFEST_V4 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/manifest_mathematics_v4.jsonl")
SHA_CRIT_ANALISE_ESPERADO = "d4bf44efef665ac70ecfcb5432e97b737beaafc91f025c71aabf91094694ad63"

SENTINELA_PL = 299.0  # target[:, 0] >= 299 dB marks a sentinel node
# Per-channel output ranges (defined for reference; not used below).
CLAMP = {0: (0.0, 200.0), 1: (0.0, 50.0), 2: (0.0, 30.0), 3: (-150.0, 0.0)}
DP_MLP_MINIMO = 0.01  # dB; below this the GNN's seed standard deviation is the tolerance
# A run is flagged when more than 1 % of the steps after warm-up were skipped by the
# GradScaler or its minimum scale fell below 1; a cell with more than 20 % flagged
# runs is reported separately.
PULOS_MAX_PCT = 1.0
ESCALA_MIN = 1.0
FRAC_CELULA_SINALIZADA = 0.20


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def carregar_manifest_cftudo() -> dict:
    out = {}
    with open(MANIFEST_V4, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            e = json.loads(ln)
            if e.get("grupo") == "tensores_cftudo":
                out[Path(e.get("arquivo") or e.get("caminho") or e.get("path") or "").name] = e.get("sha256")
    return out


def run_dir(tipo, cidade, q, seed) -> Path:
    return A3 / f"{tipo}_v3_a3_{cidade}_{q}_s{seed}"


def ler_run_json(d: Path) -> dict:
    js = sorted(glob.glob(str(d / "run_*.json")))
    if not js:
        raise FileNotFoundError(f"run JSON ausente em {d}")
    with open(js[0], "r", encoding="utf-8") as f:
        return json.load(f)


def ler_npz(d: Path):
    zs = sorted(glob.glob(str(d / "predicoes_*.npz")))
    if not zs:
        raise FileNotFoundError(f".npz ausente em {d}")
    z = np.load(zs[0])
    return {k: z[k] for k in z.files}, Path(zs[0])


def mae_por_populacao(z: dict) -> dict:
    """RSSI MAE (channel 3) on valid, sentinel and all test nodes, and path-loss MAE
    (channel 0) on valid nodes, from the physical-unit predictions stored in the .npz;
    also the SHA-256 of the sorted test node indices."""
    tgt = z["target"].astype(np.float64)
    pred = z["pred"].astype(np.float64)
    sent = (tgt[:, 0] >= SENTINELA_PL)
    # Guard: the stored sentinel mask must match the definition from the target.
    if "sentinela" in z and not np.array_equal(sent, z["sentinela"].astype(bool)):
        raise RuntimeError("campo sentinela do .npz nao bate com target[:,0] >= 299")
    e3 = np.abs(tgt[:, 3] - pred[:, 3])
    e0 = np.abs(tgt[:, 0] - pred[:, 0])
    val = ~sent
    return {
        "n": int(tgt.shape[0]),
        "n_validos": int(val.sum()),
        "n_sentinela": int(sent.sum()),
        "frac_sentinela": float(sent.mean()),
        "mae_rssi_validos_db": float(e3[val].mean()) if val.any() else None,
        "mae_rssi_sentinela_db": float(e3[sent].mean()) if sent.any() else None,
        "mae_rssi_todos_db": float(e3.mean()),
        "mae_pl_validos_db": float(e0[val].mean()) if val.any() else None,
        "idx_sha256": hashlib.sha256(np.sort(z["idx_global"].astype(np.int64)).tobytes()).hexdigest(),
    }


def gradscaler_flags(dj: dict) -> dict:
    """Mixed-precision health of a run: share of skipped steps after warm-up and minimum loss scale."""
    dt = dj.get("diagnostico_treino") or {}
    n_passos = dt.get("n_passos")
    pulos_pos = dt.get("n_pulados_pos_aquecimento")
    esc_min = dt.get("escala_minima")
    pct = (100.0 * pulos_pos / n_passos) if (n_passos and pulos_pos is not None) else None
    sinal_pulos = (pct is not None and pct > PULOS_MAX_PCT)
    sinal_escala = (esc_min is not None and esc_min < ESCALA_MIN)
    return {
        "n_passos": n_passos,
        "n_pulados_pos_aquecimento": pulos_pos,
        "pulos_pos_aquecimento_pct": pct,
        "escala_minima": esc_min,
        "sinalizada_pulos": bool(sinal_pulos),
        "sinalizada_escala": bool(sinal_escala),
        "sinalizada": bool(sinal_pulos or sinal_escala),
        "n_normas_nao_finitas": dt.get("n_normas_nao_finitas"),
    }


def regras_canal1(dj: dict) -> dict:
    """Range rules of the auxiliary terrain-derived output channel 1, read from the run JSON
    (diagnostico_saida.test.canal_1), since the unclamped prediction is not in the .npz.
    R-c1: at most 1 % out of range and 99th-percentile excess <= 5; R-c2: raw MAE within
    5 % of the clamped MAE. These rules only flag; R-c3 is evaluated per GNN/MLP pair."""
    c1 = ((dj.get("diagnostico_saida") or {}).get("test") or {}).get("canal_1") or {}
    frac = c1.get("fracao_fora_da_faixa")
    p99 = c1.get("p99_excesso_dB_ou_dBm")
    mb, mc = c1.get("mae_bruto"), c1.get("mae_pos_clamp")
    return {
        "fracao_fora_da_faixa": frac,
        "p99_excesso_db": p99,
        "mae_bruto": mb,
        "mae_pos_clamp": mc,
        "R-c1_ok": (frac is not None and p99 is not None and frac <= 0.01 and p99 <= 5.0),
        "R-c2_ok": (mb is not None and mc is not None and mc > 0 and mb <= 1.05 * mc),
        "nota": "canal auxiliar derivado do terreno; só sinaliza (critério)",
    }


def proveniencia(dj: dict, manifest: dict) -> dict:
    """Reference-field SHA-256 recorded by the run versus the artifact manifest, and the test-partition hash."""
    ins = dj.get("insumos") or {}
    rf = Path(ins.get("rf_data_file") or "").name
    sha_run = ins.get("sha256_rf_data")
    sha_man = manifest.get(rf)
    return {
        "rf_data_file": rf,
        "sha256_run": sha_run,
        "sha256_manifest_v4": sha_man,
        "bate_manifest": bool(sha_run and sha_man and sha_run == sha_man),
        "bate_manifest_segundo_wrapper": ins.get("sha256_rf_data_bate_manifest_v4"),
        "idx_sha256_global_teste": ((dj.get("particoes") or {}).get("test") or {}).get("idx_sha256_global"),
    }


def mediana(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.median(xs)) if xs else None


def dp(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.stdev(xs)) if len(xs) >= 2 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parcial", action="store_true",
                    help="permite células incompletas (artefato marcado NAO citável)")
    ap.add_argument("--status", default=str(A3 / "lote_A3_status.json"))
    args = ap.parse_args()

    sha_crit = sha256(CRIT_ANALISE)
    if sha_crit != SHA_CRIT_ANALISE_ESPERADO:
        print(f"ABORTO: criterio_A3_analise.json mudou (sha {sha_crit[:12]} != {SHA_CRIT_ANALISE_ESPERADO[:12]})")
        return 3
    with open(CRIT_ANALISE, "r", encoding="utf-8") as f:
        crit = json.load(f)
    with open(CRIT_LOTE, "r", encoding="utf-8") as f:
        lote = json.load(f)
    with open(args.status, "r", encoding="utf-8") as f:
        status = json.load(f)
    ok_labels = {c["run_label"] for c in status["corridas"] if c.get("rc") == 0}
    planejadas = lote["corridas"]
    faltam = [c["run_label"] for c in planejadas if c["run_label"] not in ok_labels]
    if faltam and not args.parcial:
        print(f"ABORTO: {len(faltam)} corridas sem rc=0 no status; use --parcial só para checar estrutura")
        return 2

    celulas = sorted({(c["cidade"], c["quadrante"]) for c in planejadas})
    seeds = sorted({c["seed_treino"] for c in planejadas})
    manifest = carregar_manifest_cftudo()

    # Step 1: MLP runs only, then the per-cell tolerances, written before any GNN file is read.
    mlp = {}
    for cid, q in celulas:
        for s in seeds:
            d = run_dir("mlp", cid, q, s)
            lbl = d.name
            if lbl not in ok_labels:
                continue
            dj = ler_run_json(d)
            z, zp = ler_npz(d)
            mlp[(cid, q, s)] = {
                "run_label": lbl, "npz": str(zp),
                "mae": mae_por_populacao(z),
                "gradscaler": gradscaler_flags(dj),
                "canal1": regras_canal1(dj),
                "proveniencia": proveniencia(dj, manifest),
                "melhor_epoca": (dj.get("selecao") or {}).get("melhor_epoca"),
                "rc_status": 0,
            }
    tol = {}
    for cid, q in celulas:
        xs = [mlp[(cid, q, s)]["mae"]["mae_rssi_sentinela_db"] for s in seeds if (cid, q, s) in mlp]
        d_mlp = dp(xs)
        tol[f"{cid}_{q}"] = {
            "n_seeds_mlp": len(xs),
            "dp_mlp_sentinela_db": d_mlp,
            "fonte": "dp entre seeds do MAE_sent do MLP (criterio_A3_analise.H1.tolerancia)",
            "usa_dp_gnn": (d_mlp is not None and d_mlp < DP_MLP_MINIMO),
            "tolerancia_db": d_mlp if (d_mlp is not None and d_mlp >= DP_MLP_MINIMO) else None,
        }
    tol_path = A3 / "tolerancias_A3.json"
    with open(tol_path, "w", encoding="utf-8") as f:
        json.dump({"criterio_sha256": sha_crit, "parcial": bool(faltam),
                   "regra": crit["H1_paridade_sentinela"]["tolerancia"],
                   "tolerancias": tol}, f, indent=1, ensure_ascii=False)
    print(f"tolerâncias gravadas ANTES do contraste: {tol_path}")

    # Step 2: GNN runs.
    gnn = {}
    for cid, q in celulas:
        for s in seeds:
            d = run_dir("gnn", cid, q, s)
            lbl = d.name
            if lbl not in ok_labels:
                continue
            dj = ler_run_json(d)
            z, zp = ler_npz(d)
            cap = dj.get("capacidade") or {}
            gnn[(cid, q, s)] = {
                "run_label": lbl, "npz": str(zp),
                "mae": mae_por_populacao(z),
                "gradscaler": gradscaler_flags(dj),
                "canal1": regras_canal1(dj),
                "proveniencia": proveniencia(dj, manifest),
                "melhor_epoca": (dj.get("selecao") or {}).get("melhor_epoca"),
                "capacidade_efetiva": cap.get("efetiva"), "capacidade_nominal": cap.get("nominal"),
                "invariancia_passou": (((dj.get("diagnostico_avaliacao") or {}).get("teste_invariancia") or {}).get("passou_criterio")),
                "rc_status": 0,
            }
    # Final tolerance: fall back to the GNN's seed standard deviation where the criterion says so.
    for cid, q in celulas:
        k = f"{cid}_{q}"
        if tol[k]["usa_dp_gnn"]:
            xs = [gnn[(cid, q, s)]["mae"]["mae_rssi_sentinela_db"] for s in seeds if (cid, q, s) in gnn]
            tol[k]["dp_gnn_sentinela_db"] = dp(xs)
            tol[k]["tolerancia_db"] = dp(xs)

    # Step 3: GNN - MLP contrast paired by seed, per cell (negative = GNN better).
    pops = ("mae_rssi_validos_db", "mae_rssi_sentinela_db", "mae_rssi_todos_db", "mae_pl_validos_db")
    celulas_out = {}
    n_paridade = 0
    n_celulas_completas = 0
    for cid, q in celulas:
        k = f"{cid}_{q}"
        pares = []
        for s in seeds:
            if (cid, q, s) not in gnn or (cid, q, s) not in mlp:
                continue
            g, m = gnn[(cid, q, s)], mlp[(cid, q, s)]
            mesmo_idx = g["mae"]["idx_sha256"] == m["mae"]["idx_sha256"]
            par = {"seed": s, "mesma_particao_teste": mesmo_idx,
                   "gnn": g["mae"], "mlp": m["mae"],
                   "delta": {p: (g["mae"][p] - m["mae"][p]) if (g["mae"][p] is not None and m["mae"][p] is not None) else None for p in pops},
                   "sinalizada_gradscaler": {"gnn": g["gradscaler"]["sinalizada"], "mlp": m["gradscaler"]["sinalizada"]},
                   "melhor_epoca": {"gnn": g["melhor_epoca"], "mlp": m["melhor_epoca"]},
                   "proveniencia_ok": g["proveniencia"]["bate_manifest"] and m["proveniencia"]["bate_manifest"],
                   "R-c3_mesmo_sinal_canal1": None}
            # R-c3: the channel-1 contrast has the same sign before and after clamping.
            gb, mb = g["canal1"]["mae_bruto"], m["canal1"]["mae_bruto"]
            gc, mc = g["canal1"]["mae_pos_clamp"], m["canal1"]["mae_pos_clamp"]
            if None not in (gb, mb, gc, mc):
                par["R-c3_mesmo_sinal_canal1"] = bool(np.sign(gb - mb) == np.sign(gc - mc))
            pares.append(par)
        completa = (len(pares) == len(seeds))
        n_celulas_completas += int(completa)
        dist = {}
        for p in pops:
            ds = [pr["delta"][p] for pr in pares]
            dist[p] = {"mediana": mediana(ds), "min": min([x for x in ds if x is not None], default=None),
                       "max": max([x for x in ds if x is not None], default=None),
                       "n_seeds_gnn_melhor": sum(1 for x in ds if x is not None and x < 0),
                       "n_seeds": len([x for x in ds if x is not None]),
                       "valores_por_seed": {str(pr["seed"]): pr["delta"][p] for pr in pares}}
        med_sent = dist["mae_rssi_sentinela_db"]["mediana"]
        t = tol[k]["tolerancia_db"]
        paridade = (completa and med_sent is not None and t is not None and abs(med_sent) <= t)
        n_paridade += int(paridade)
        n_sinal = sum(1 for pr in pares if pr["sinalizada_gradscaler"]["gnn"] or pr["sinalizada_gradscaler"]["mlp"])
        celulas_out[k] = {
            "completa": completa, "n_pares": len(pares),
            "tolerancia_db": t, "mediana_delta_sentinela_db": med_sent,
            "paridade_sentinela": paridade,
            "distribuicao_gnn_menos_mlp": dist,
            "mae_gnn_por_seed": {str(s): gnn[(cid, q, s)]["mae"] for s in seeds if (cid, q, s) in gnn},
            "mae_mlp_por_seed": {str(s): mlp[(cid, q, s)]["mae"] for s in seeds if (cid, q, s) in mlp},
            "n_corridas_sinalizadas_gradscaler": n_sinal,
            "celula_reportada_a_parte": (n_sinal / max(1, 2 * len(pares))) > FRAC_CELULA_SINALIZADA,
            "todas_mesma_particao": all(pr["mesma_particao_teste"] for pr in pares),
            "todas_proveniencia_ok": all(pr["proveniencia_ok"] for pr in pares),
            "melhores_epocas": {"gnn": [pr["melhor_epoca"]["gnn"] for pr in pares], "mlp": [pr["melhor_epoca"]["mlp"] for pr in pares]},
            "R-c3_mesmo_sinal_por_seed": [pr["R-c3_mesmo_sinal_canal1"] for pr in pares],
            "canal1_gnn": {str(s): gnn[(cid, q, s)]["canal1"] for s in seeds if (cid, q, s) in gnn},
            "canal1_mlp": {str(s): mlp[(cid, q, s)]["canal1"] for s in seeds if (cid, q, s) in mlp},
            "gradscaler_gnn": {str(s): gnn[(cid, q, s)]["gradscaler"] for s in seeds if (cid, q, s) in gnn},
            "capacidade_gnn": {str(s): [gnn[(cid, q, s)]["capacidade_efetiva"], gnn[(cid, q, s)]["capacidade_nominal"]] for s in seeds if (cid, q, s) in gnn},
            "invariancia_gnn_passou": [gnn[(cid, q, s)]["invariancia_passou"] for s in seeds if (cid, q, s) in gnn],
            "pares": pares,
        }

    veredito_h1 = None
    if n_celulas_completas == len(celulas):
        veredito_h1 = "PARIDADE_APROXIMADA" if n_paridade >= 6 else "DIFERENCA_SISTEMATICA"
    out = {
        "artefato": "agregado_A3",
        "criterio_analise_sha256": sha_crit,
        "criterio_lote_sha256": sha256(CRIT_LOTE),
        "parcial_NAO_CITAVEL": bool(faltam),
        "corridas_faltantes": faltam,
        "n_celulas": len(celulas), "n_seeds": len(seeds),
        "n_gnn_lidas": len(gnn), "n_mlp_lidas": len(mlp),
        "H1": {"regra": crit["H1_paridade_sentinela"], "n_celulas_em_paridade": n_paridade,
               "n_celulas_completas": n_celulas_completas, "veredito": veredito_h1},
        "proveniencia_todas_ok": all(c["todas_proveniencia_ok"] for c in celulas_out.values()) if celulas_out else None,
        "particao_identica_em_toda_celula": all(c["todas_mesma_particao"] for c in celulas_out.values()) if celulas_out else None,
        "tolerancias": tol,
        "celulas": celulas_out,
        "nota_nao_determinismo": crit["nao_determinismo"],
        "nota_inferencia": crit["unidade_de_analise"],
    }
    out_path = A3 / ("agregado_A3_PARCIAL.json" if faltam else "agregado_A3.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"gravado: {out_path}")
    print(f"H1: {n_paridade}/{n_celulas_completas} células completas em paridade -> {veredito_h1}")
    for k, c in celulas_out.items():
        dsent = c["distribuicao_gnn_menos_mlp"]["mae_rssi_sentinela_db"]
        dval = c["distribuicao_gnn_menos_mlp"]["mae_rssi_validos_db"]
        print(f"  {k:12s} completa={c['completa']} tol={c['tolerancia_db']} med_sent={c['mediana_delta_sentinela_db']} "
              f"(GNN melhor {dsent['n_seeds_gnn_melhor']}/{dsent['n_seeds']}) med_val={dval['mediana']} "
              f"(GNN melhor {dval['n_seeds_gnn_melhor']}/{dval['n_seeds']}) prov={c['todas_proveniencia_ok']} idx={c['todas_mesma_particao']} "
              f"sinal_gs={c['n_corridas_sinalizadas_gradscaler']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
