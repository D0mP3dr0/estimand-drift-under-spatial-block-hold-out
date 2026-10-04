#!/usr/bin/env python3
"""Aggregator of model batch A4: error drift of trained models across split draws.

The analysis follows the field analise_pre_registrada of the decision criterion
criterio_A4.json, which also lists the planned runs in two arms:
  - main arm (drift with a model): Bauru Q1 to Q4 x split seeds 101 to 105 x {GNN, MLP},
    training seed 42;
  - sensitivity arm: the MLP with its earlier layer widths in Bauru Q1 and Lins Q1,
    training seeds 42 to 46 (split seed 42), paired by training seed with the
    mlp_v3_a3_* runs of batch A3, which use the current widths.

The script refuses an incomplete batch (runs without exit code 0 in the launcher status)
unless --parcial is given, in which case the output is marked partial and not citable.
As in batch A3, the per-cell tolerance (standard deviation across split draws of the
MLP's sentinel-node RSSI MAE; the GNN's when the MLP's is below 0.01 dB) is written to
tolerancias_A4.json before any GNN prediction is opened. Per cell and model it then
reports the between-draw standard deviation and mean of the test MAE on each node
population, how many draws the model beats each analytical baseline (FSPL, Hata rural,
COST-231 suburban) on valid nodes, the between-draw spread of the baselines, and the
GNN - MLP parity on sentinel nodes paired by draw; across cells, the spread of the
per-cell mean valid-node MAE; and, for the sensitivity arm, the median old-minus-new
width difference, called negligible when within the seed standard deviation of the
current widths.

Inputs: results/gpu/A4/<run_label>/ and results/gpu/A3/mlp_v3_a3_*/ (run_*.json and
predicoes_*.npz), lote_A4_status.json, criterio_A4.json and the artifact manifest.
Outputs: results/gpu/A4/tolerancias_A4.json and agregado_A4.json (agregado_A4_PARCIAL.json
with --parcial). Deterministic given the run files.

Usage: python v3_A4_agregar.py [--parcial] [--status <lote_A4_status.json>]
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
A4 = RAIZ / "gpu" / "A4"
A3 = RAIZ / "gpu" / "A3"
CRIT = RAIZ / "criterios" / "criterio_A4.json"
MANIFEST_V4 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/manifest_mathematics_v4.jsonl")
SENTINELA_PL = 299.0  # target[:, 0] >= 299 dB marks a sentinel node
DP_MLP_MINIMO = 0.01  # dB; below this the GNN's between-draw standard deviation is the tolerance
# GradScaler flags: more than 1 % skipped steps after warm-up, or minimum scale below 1.
PULOS_MAX_PCT = 1.0
ESCALA_MIN = 1.0
BASELINES = ("fspl", "hata_rural", "cost231_sub")
POPS = ("mae_rssi_validos_db", "mae_rssi_sentinela_db", "mae_rssi_todos_db", "mae_pl_validos_db")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest_cftudo() -> dict:
    out = {}
    with open(MANIFEST_V4, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                e = json.loads(ln)
                if e.get("grupo") == "tensores_cftudo":
                    out[Path(e.get("caminho", "")).name] = e.get("sha256")
    return out


def ler_run(d: Path):
    js = sorted(glob.glob(str(d / "run_*.json")))
    zs = sorted(glob.glob(str(d / "predicoes_*.npz")))
    if not js or not zs:
        raise FileNotFoundError(f"run JSON ou .npz ausente em {d}")
    with open(js[0], "r", encoding="utf-8") as f:
        dj = json.load(f)
    z = np.load(zs[0])
    return dj, {k: z[k] for k in z.files}, Path(zs[0])


def mae_pop(z: dict) -> dict:
    """RSSI MAE (channel 3) on valid, sentinel and all test nodes, path-loss MAE (channel 0)
    on valid nodes, and the SHA-256 of the sorted test node indices."""
    tgt = z["target"].astype(np.float64)
    pred = z["pred"].astype(np.float64)
    sent = tgt[:, 0] >= SENTINELA_PL
    if "sentinela" in z and not np.array_equal(sent, z["sentinela"].astype(bool)):
        raise RuntimeError("campo sentinela do .npz nao bate com target[:,0] >= 299")
    e3 = np.abs(tgt[:, 3] - pred[:, 3])
    e0 = np.abs(tgt[:, 0] - pred[:, 0])
    val = ~sent
    return {
        "n": int(tgt.shape[0]), "n_validos": int(val.sum()), "n_sentinela": int(sent.sum()),
        "mae_rssi_validos_db": float(e3[val].mean()) if val.any() else None,
        "mae_rssi_sentinela_db": float(e3[sent].mean()) if sent.any() else None,
        "mae_rssi_todos_db": float(e3.mean()),
        "mae_pl_validos_db": float(e0[val].mean()) if val.any() else None,
        "idx_sha256": hashlib.sha256(np.sort(z["idx_global"].astype(np.int64)).tobytes()).hexdigest(),
    }


def gradscaler(dj: dict) -> dict:
    dt = dj.get("diagnostico_treino") or {}
    n, p, e = dt.get("n_passos"), dt.get("n_pulados_pos_aquecimento"), dt.get("escala_minima")
    pct = (100.0 * p / n) if (n and p is not None) else None
    return {"pulos_pos_aquecimento_pct": pct, "escala_minima": e,
            "sinalizada": bool((pct is not None and pct > PULOS_MAX_PCT) or (e is not None and e < ESCALA_MIN))}


def canal1(dj: dict) -> dict:
    """Range rules of the auxiliary output channel 1 from the run JSON (R-c1: at most 1 % out of
    range and 99th-percentile excess <= 5; R-c2: raw MAE within 5 % of the clamped MAE)."""
    c1 = ((dj.get("diagnostico_saida") or {}).get("test") or {}).get("canal_1") or {}
    frac, p99, mb, mc = c1.get("fracao_fora_da_faixa"), c1.get("p99_excesso_dB_ou_dBm"), c1.get("mae_bruto"), c1.get("mae_pos_clamp")
    return {"fracao_fora_da_faixa": frac, "p99_excesso_db": p99, "mae_bruto": mb, "mae_pos_clamp": mc,
            "R-c1_ok": (frac is not None and p99 is not None and frac <= 0.01 and p99 <= 5.0),
            "R-c2_ok": (mb is not None and mc is not None and mc > 0 and mb <= 1.05 * mc)}


def proveniencia(dj: dict, man: dict) -> dict:
    """Reference-field SHA-256 recorded by the run versus the artifact manifest, and the test-partition hash."""
    ins = dj.get("insumos") or {}
    rf = Path(ins.get("rf_data_file") or "").name
    return {"rf_data_file": rf, "sha256_run": ins.get("sha256_rf_data"), "sha256_manifest_v4": man.get(rf),
            "bate_manifest": bool(ins.get("sha256_rf_data") and man.get(rf) and ins.get("sha256_rf_data") == man.get(rf)),
            "idx_sha256_global_teste": ((dj.get("particoes") or {}).get("test") or {}).get("idx_sha256_global")}


def baselines_test(dj: dict) -> dict:
    """Test MAE of each analytical baseline as recorded in the run JSON."""
    bt = ((dj.get("baselines_analiticos") or {}).get("test") or {})
    return {b: bt.get(f"baseline_{b}_mae") for b in BASELINES}


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.median(xs)) if xs else None


def dp(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.stdev(xs)) if len(xs) >= 2 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parcial", action="store_true")
    ap.add_argument("--status", default=str(A4 / "lote_A4_status.json"))
    args = ap.parse_args()

    sha_crit = sha256(CRIT)
    with open(CRIT, "r", encoding="utf-8") as f:
        crit = json.load(f)
    with open(args.status, "r", encoding="utf-8") as f:
        status = json.load(f)
    ok = {c["run_label"] for c in status["corridas"] if c.get("rc") == 0}
    plan = crit["corridas"]
    faltam = [c["run_label"] for c in plan if c["run_label"] not in ok]
    if faltam and not args.parcial:
        print(f"ABORTO: {len(faltam)} corridas sem rc=0; use --parcial só para checar estrutura")
        return 2
    man = manifest_cftudo()
    principal = [c for c in plan if c["braco"] == "deriva_com_modelo" and c["run_label"] in ok]
    sens = [c for c in plan if c["braco"] == "sensibilidade_larguras_mlp" and c["run_label"] in ok]
    celulas = sorted({(c["cidade"], c["quadrante"]) for c in principal})
    seeds_split = sorted({c["split_seed"] for c in principal})

    def carregar(c, base=A4):
        dj, z, zp = ler_run(base / c["run_label"])
        return {"run_label": c["run_label"], "npz": str(zp), "mae": mae_pop(z), "gradscaler": gradscaler(dj),
                "canal1": canal1(dj), "proveniencia": proveniencia(dj, man), "baselines_test": baselines_test(dj),
                "melhor_epoca": (dj.get("selecao") or {}).get("melhor_epoca"),
                "capacidade": (dj.get("capacidade") or {}).get("efetiva"),
                "invariancia_passou": (((dj.get("diagnostico_avaliacao") or {}).get("teste_invariancia") or {}).get("passou_criterio"))}

    # Step 1: MLP runs only, then the per-cell tolerances, written before any GNN file is read.
    mlp = {(c["cidade"], c["quadrante"], c["split_seed"]): carregar(c) for c in principal if c["tipo"] == "mlp"}
    tol = {}
    for cid, q in celulas:
        xs = [mlp[(cid, q, s)]["mae"]["mae_rssi_sentinela_db"] for s in seeds_split if (cid, q, s) in mlp]
        d = dp(xs)
        tol[f"{cid}_{q}"] = {"n": len(xs), "dp_mlp_sentinela_db": d, "usa_dp_gnn": (d is not None and d < DP_MLP_MINIMO),
                             "tolerancia_db": d if (d is not None and d >= DP_MLP_MINIMO) else None}
    with open(A4 / "tolerancias_A4.json", "w", encoding="utf-8") as f:
        json.dump({"criterio_sha256": sha_crit, "parcial": bool(faltam), "tolerancias": tol}, f, indent=1, ensure_ascii=False)
    print("tolerâncias gravadas ANTES do contraste")

    # Step 2: GNN runs; tolerance fallback to the GNN's spread where the criterion says so.
    gnn = {(c["cidade"], c["quadrante"], c["split_seed"]): carregar(c) for c in principal if c["tipo"] == "gnn"}
    for cid, q in celulas:
        k = f"{cid}_{q}"
        if tol[k]["usa_dp_gnn"]:
            xs = [gnn[(cid, q, s)]["mae"]["mae_rssi_sentinela_db"] for s in seeds_split if (cid, q, s) in gnn]
            tol[k]["dp_gnn_sentinela_db"] = dp(xs)
            tol[k]["tolerancia_db"] = dp(xs)

    # Step 3: drift with a model, model-versus-baseline inversion per draw, and parity per draw.
    cel_out = {}
    medias_por_modelo = {"gnn": [], "mlp": []}
    for cid, q in celulas:
        k = f"{cid}_{q}"
        bloco = {"n_sorteios": 0, "por_modelo": {}, "paridade": {}, "baselines_dp_entre_sorteios": {},
                 "todas_mesma_particao": True, "todas_proveniencia_ok": True}
        for mod, store in (("gnn", gnn), ("mlp", mlp)):
            runs = [store[(cid, q, s)] for s in seeds_split if (cid, q, s) in store]
            vals = {p: [r["mae"][p] for r in runs] for p in POPS}
            inv = {b: sum(1 for r in runs if r["mae"]["mae_rssi_validos_db"] is not None and r["baselines_test"][b] is not None
                          and r["mae"]["mae_rssi_validos_db"] < r["baselines_test"][b]) for b in BASELINES}
            bloco["por_modelo"][mod] = {
                "n": len(runs),
                "mae_por_sorteio": {str(s): store[(cid, q, s)]["mae"] for s in seeds_split if (cid, q, s) in store},
                "dp_entre_sorteios": {p: dp(vals[p]) for p in POPS},
                "media_entre_sorteios": {p: (float(np.mean([x for x in vals[p] if x is not None])) if any(x is not None for x in vals[p]) else None) for p in POPS},
                "inversao_vs_baselines_validos": {b: {"n_sorteios_modelo_melhor": inv[b], "n": len(runs)} for b in BASELINES},
                "gradscaler_sinalizadas": sum(1 for r in runs if r["gradscaler"]["sinalizada"]),
                "canal1": {str(s): store[(cid, q, s)]["canal1"] for s in seeds_split if (cid, q, s) in store},
                "melhores_epocas": [r["melhor_epoca"] for r in runs],
                "capacidade_efetiva": [r["capacidade"] for r in runs],
                "invariancia_passou": [r["invariancia_passou"] for r in runs],
            }
            bloco["todas_proveniencia_ok"] &= all(r["proveniencia"]["bate_manifest"] for r in runs)
            if vals["mae_rssi_validos_db"]:
                medias_por_modelo[mod].append(bloco["por_modelo"][mod]["media_entre_sorteios"]["mae_rssi_validos_db"])
        # Baselines on the same draws, read from the MLP run JSON (identical to the GNN's on a draw).
        for b in BASELINES:
            xs = [mlp[(cid, q, s)]["baselines_test"][b] for s in seeds_split if (cid, q, s) in mlp]
            bloco["baselines_dp_entre_sorteios"][b] = dp(xs)
        # GNN - MLP paired by split draw (negative = GNN better).
        pares = []
        for s in seeds_split:
            if (cid, q, s) in gnn and (cid, q, s) in mlp:
                g, m = gnn[(cid, q, s)], mlp[(cid, q, s)]
                bloco["todas_mesma_particao"] &= (g["mae"]["idx_sha256"] == m["mae"]["idx_sha256"])
                pares.append({"split_seed": s, "delta": {p: (g["mae"][p] - m["mae"][p]) if (g["mae"][p] is not None and m["mae"][p] is not None) else None for p in POPS}})
        bloco["n_sorteios"] = len(pares)
        t = tol[k]["tolerancia_db"]
        dist = {}
        for p in POPS:
            ds = [pr["delta"][p] for pr in pares]
            dist[p] = {"mediana": med(ds), "min": min([x for x in ds if x is not None], default=None),
                       "max": max([x for x in ds if x is not None], default=None),
                       "n_sorteios_gnn_melhor": sum(1 for x in ds if x is not None and x < 0), "n": len([x for x in ds if x is not None])}
        ms = dist["mae_rssi_sentinela_db"]["mediana"]
        bloco["paridade"] = {"tolerancia_db": t, "mediana_delta_sentinela_db": ms,
                             "paridade_sentinela": (len(pares) == len(seeds_split) and ms is not None and t is not None and abs(ms) <= t),
                             "distribuicao_gnn_menos_mlp": dist, "pares": pares}
        cel_out[k] = bloco

    deriva = {mod: {"dp_entre_celulas_das_medias_validos_db": dp(medias_por_modelo[mod]),
                    "dp_entre_sorteios_por_celula_validos_db": {k: c["por_modelo"].get(mod, {}).get("dp_entre_sorteios", {}).get("mae_rssi_validos_db") for k, c in cel_out.items()}}
              for mod in ("gnn", "mlp")}

    # Step 4: width sensitivity, earlier-width MLP paired by training seed with the A3 MLP.
    sens_out = {}
    for c in sens:
        cid = c["cidade"]
        antiga = carregar(c)
        novo_lbl = f"mlp_v3_a3_{cid}_Q1_s{c['seed_treino']}"
        try:
            dj, z, zp = ler_run(A3 / novo_lbl)
            nova = {"run_label": novo_lbl, "mae": mae_pop(z)}
        except FileNotFoundError:
            nova = None
        sens_out.setdefault(f"{cid}_Q1", []).append({
            "seed_treino": c["seed_treino"], "antiga": antiga["mae"], "nova": (nova["mae"] if nova else None),
            "delta_antiga_menos_nova": ({p: (antiga["mae"][p] - nova["mae"][p]) if (nova and antiga["mae"][p] is not None and nova["mae"][p] is not None) else None for p in POPS}),
            "mesma_particao": (nova is not None and antiga["mae"]["idx_sha256"] == nova["mae"]["idx_sha256"]),
            "proveniencia_ok": antiga["proveniencia"]["bate_manifest"], "gradscaler_sinalizada": antiga["gradscaler"]["sinalizada"]})
    sens_res = {}
    for k, lst in sens_out.items():
        res = {}
        for p in POPS:
            ds = [x["delta_antiga_menos_nova"][p] for x in lst]
            novas = [x["nova"][p] for x in lst if x["nova"]]
            dpn = dp(novas)
            m = med(ds)
            res[p] = {"mediana_delta_db": m, "min": min([x for x in ds if x is not None], default=None), "max": max([x for x in ds if x is not None], default=None),
                      "dp_entre_seeds_larguras_novas_db": dpn, "desprezivel": (m is not None and dpn is not None and abs(m) <= dpn), "n": len([x for x in ds if x is not None])}
        sens_res[k] = {"pares": lst, "resultado": res}

    out = {"artefato": "agregado_A4", "criterio_sha256": sha_crit, "parcial_NAO_CITAVEL": bool(faltam), "corridas_faltantes": faltam,
           "celulas": cel_out, "deriva_2_2": deriva, "tolerancias": tol, "sensibilidade_larguras": sens_res,
           "n_celulas_em_paridade": sum(1 for c in cel_out.values() if c["paridade"]["paridade_sentinela"]),
           "proveniencia_todas_ok": all(c["todas_proveniencia_ok"] for c in cel_out.values()) if cel_out else None,
           "particao_identica_gnn_mlp": all(c["todas_mesma_particao"] for c in cel_out.values()) if cel_out else None,
           "nota": crit["analise_pre_registrada"]}
    out_path = A4 / ("agregado_A4_PARCIAL.json" if faltam else "agregado_A4.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"gravado: {out_path}")
    for k, c in cel_out.items():
        g, m = c["por_modelo"].get("gnn", {}), c["por_modelo"].get("mlp", {})
        print(f"  {k:10s} n={c['n_sorteios']} dp_sorteios validos gnn={g.get('dp_entre_sorteios', {}).get('mae_rssi_validos_db')} mlp={m.get('dp_entre_sorteios', {}).get('mae_rssi_validos_db')} "
              f"inv_fspl gnn={g.get('inversao_vs_baselines_validos', {}).get('fspl')} mlp={m.get('inversao_vs_baselines_validos', {}).get('fspl')} "
              f"paridade_sent={c['paridade']['paridade_sentinela']} tol={c['paridade']['tolerancia_db']} med={c['paridade']['mediana_delta_sentinela_db']}")
    print("deriva 2.2:", json.dumps({m: deriva[m]["dp_entre_celulas_das_medias_validos_db"] for m in deriva}))
    for k, r in sens_res.items():
        print(f"  sensibilidade {k}: " + json.dumps({p: (r['resultado'][p]['mediana_delta_db'], r['resultado'][p]['desprezivel']) for p in POPS}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
