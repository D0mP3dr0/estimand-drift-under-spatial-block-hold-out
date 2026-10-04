#!/usr/bin/env python3
"""Base aggregator of campaign G1: GNN and MLP trained at block size g = 10 km, buffer b = 2 km.

Computes the metrics listed in the field `metricas` of the decision criterion
criterio_G1_modelos_g10.json, block by block of the run plan plano_G1.json:
  - blocks 1 and 3 (cells Bauru Q1 and Campinas Q1, then Bauru Q3 and Campinas Q3;
    20 split draws at training seed 42): per model, the between-draw standard deviation
    of the valid-node RSSI MAE and its ratio to the repeat-noise threshold 0.132 dB
    (flags >= 3x and < 2x), the Pearson and Spearman correlations with the constant
    predictor's valid-node MAE on the same draws, the spread on the other node
    populations, how many draws beat each analytical baseline, and the median absolute
    GNN - MLP difference on sentinel nodes against 0.117 dB;
  - block 2 (Bauru Q1 and Campinas Q1, training seeds 42, 43, 44 on 5 draws): the
    pooled between-seed standard deviation, its ratio to the between-draw spread, and a
    two-way ANOVA without replication (draw x seed).
It only computes these quantities and their arithmetic comparison with the criterion
thresholds; it does not interpret them. Each run is also checked against the reference
record fase2/_v3_2.1_3.1_parcial_16x60rnd.json (same retained test size and node counts
on the same draw), for its declared geometry, the minimum distance between partitions
and the SHA-256 of its reference field in the artifact manifest.

A block is aggregated only when complete (every planned run has its run JSON and .npz,
or is recorded as having no valid test node); otherwise it is refused with exit code 2
and nothing is written. Block 2 also requires block 1, whose seed-42 runs it reuses.
A draw with no valid test node is listed and left out of standard deviations and
correlations. The helper functions sha256, manifest_cftudo, ler_run, mae_pop, med and dp
are those of v3_A4_agregar.py. The aggregators v3_G1_agregar_v5.py, _v8.py and _v10.py
derive from this script and verify its SHA-256 before running.

Inputs: results/gpu/G1/<run_label>/ (run_*.json, predicoes_*.npz), plano_G1.json,
lote_G1_status.json, the criterion, the reference record above and the artifact manifest.
Output: results/gpu/G1/agregado_G1_bloco<N>.json. Deterministic given the run files.

Usage: python v3_G1_agregar.py [--bloco 1|2|3]
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
G1 = RAIZ / "gpu" / "G1"
CRIT = RAIZ / "criterios" / "criterio_G1_modelos_g10.json"
PLANO = G1 / "plano_G1.json"
STATUS = G1 / "lote_G1_status.json"
PARCIAL_2_1 = RAIZ / "fase2" / "_v3_2.1_3.1_parcial_16x60rnd.json"
MANIFEST_V4 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/manifest_mathematics_v4.jsonl")
SENTINELA_PL = 299.0  # target[:, 0] >= 299 dB marks a sentinel node
RUIDO_REPETICAO_DB = 0.132  # largest GNN run-to-run difference in the repeat test A2c (criterion threshold)
RUIDO_SENTINELA_DB = 0.117  # largest run-to-run difference on sentinel nodes in A2c (criterion threshold)
POPS = ("mae_rssi_validos_db", "mae_rssi_sentinela_db", "mae_rssi_todos_db", "mae_pl_validos_db")
BASELINES = ("fspl", "hata_rural", "cost231_sub")
CELULAS_POR_BLOCO = {1: ("bauru_Q1", "campinas_Q1"), 2: ("bauru_Q1", "campinas_Q1"), 3: ("bauru_Q3", "campinas_Q3")}


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


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.median(xs)) if xs else None


def dp(xs):
    """Sample standard deviation (ddof = 1), ignoring None."""
    xs = [x for x in xs if x is not None]
    return float(statistics.stdev(xs)) if len(xs) >= 2 else None


def razao(a, b):
    return (a / b) if (a is not None and b not in (None, 0)) else None


def pearson(x, y):
    if len(x) < 3:
        return None
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    """Spearman correlation as the Pearson correlation of ranks (ties not averaged)."""
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
    return pearson(rx, ry)


def anova2_sem_repeticao(M: np.ndarray) -> dict:
    """Two-way ANOVA without replication on M (a draws in rows x b seeds in columns, no empty cell).

    Variance components from expected mean squares, (MS_draw - MS_res) / b and (MS_seed - MS_res) / a,
    reported raw and truncated at 0.
    """
    a, b = M.shape
    gm = M.mean()
    ss_sor = b * float(((M.mean(axis=1) - gm) ** 2).sum())
    ss_sem = a * float(((M.mean(axis=0) - gm) ** 2).sum())
    ss_tot = float(((M - gm) ** 2).sum())
    ss_res = ss_tot - ss_sor - ss_sem
    ms_sor, ms_sem = ss_sor / (a - 1), ss_sem / (b - 1)
    ms_res = ss_res / ((a - 1) * (b - 1))
    var_sor_bruta = (ms_sor - ms_res) / b
    var_sem_bruta = (ms_sem - ms_res) / a
    return {"a_sorteios": a, "b_sementes": b, "ms_sorteio": ms_sor, "ms_semente": ms_sem, "ms_residuo": ms_res,
            "componente_sorteio_var_bruta": var_sor_bruta, "componente_semente_var_bruta": var_sem_bruta,
            "componente_sorteio_var": max(0.0, var_sor_bruta), "componente_semente_var": max(0.0, var_sem_bruta),
            "componente_residuo_var": ms_res,
            "componente_sorteio_ge_semente": bool(max(0.0, var_sor_bruta) >= max(0.0, var_sem_bruta))}


def carregar_2_1() -> dict:
    """Retained test size, node counts and constant-predictor valid-node MAE per (cell, split seed)."""
    with open(PARCIAL_2_1, "r", encoding="utf-8") as f:
        p = json.load(f)
    out = {}
    for cel, d in p["celulas"].items():
        for s in d.get("por_sorteio", []):
            if s.get("status") == "ok":
                out[(cel, int(s["split_seed"]))] = {
                    "n_test": s["n_test"], "validos": s["n_pop_teste"]["validos"], "todos": s["n_pop_teste"]["todos"],
                    "mae_constante_validos": s["mae_constante_teste"]["validos"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bloco", type=int, choices=(1, 2, 3), default=None,
                    help="default: agrega todo bloco COMPLETO; blocos incompletos sao recusados")
    args = ap.parse_args()

    sha_crit = sha256(CRIT)
    with open(PLANO, "r", encoding="utf-8") as f:
        plano = json.load(f)["corridas"]
    with open(STATUS, "r", encoding="utf-8") as f:
        status = json.load(f)
    sem_val = {c["run_label"] for c in status["corridas"] if c.get("sem_validos")}
    man = manifest_cftudo()
    ref21 = carregar_2_1()

    def completa(c) -> bool:
        if c["run_label"] in sem_val:
            return True
        d = G1 / c["run_label"]
        return bool(glob.glob(str(d / "run_*.json")) and glob.glob(str(d / "predicoes_*.npz")))

    def bloco_completo(n: int) -> bool:
        return all(completa(c) for c in plano if c["bloco"] == n)

    cache = {}

    def carregar(c) -> dict:
        """Metrics and checks of one run (cached); a run without valid nodes and without .npz has no MAE."""
        lbl = c["run_label"]
        if lbl in cache:
            return cache[lbl]
        if lbl in sem_val and not glob.glob(str(G1 / lbl / "predicoes_*.npz")):
            r = {"run_label": lbl, "sem_validos": True, "mae": None}
        else:
            dj, z, zp = ler_run(G1 / lbl)
            ins = dj.get("insumos") or {}
            rf = Path(ins.get("rf_data_file") or "").name
            geo = dj.get("split") or {}
            m = mae_pop(z)
            ref = ref21.get((f"{c['cidade']}_{c['quadrante']}", c["split_seed"]))
            cfg = dj.get("config") or {}
            r = {
                "run_label": lbl, "sem_validos": m["n_validos"] == 0, "npz": str(zp), "mae": m,
                "grid_km": cfg.get("grid_km"), "buffer_km": cfg.get("buffer_km"), "split_seed_run": dj.get("split_seed"),
                "seed_treino_run": dj.get("seed"),
                "proveniencia_ok": bool(ins.get("sha256_rf_data") and man.get(rf) == ins.get("sha256_rf_data")),
                "dist_min_entre_particoes_km": {k: v.get("dist_min_km") for k, v in ((geo.get("verificacao") or {}).get("pares") or {}).items()},
                "intersecoes": (geo.get("verificacao") or {}).get("intersecoes"),
                "n_test_retido_run": (geo.get("n_nos_apos_buffer") or {}).get("test"),
                "baselines_test": {b: ((dj.get("baselines_analiticos") or {}).get("test") or {}).get(f"baseline_{b}_mae") for b in BASELINES},
                "melhor_epoca": (dj.get("selecao") or {}).get("melhor_epoca"),
                "bate_com_2_1": (None if ref is None else {
                    "n_test_igual": geo.get("n_nos_apos_buffer", {}).get("test") == ref["n_test"],
                    "n_validos_igual": m["n_validos"] == ref["validos"],
                    "n_todos_igual": m["n"] == ref["todos"]}),
                "mae_constante_validos_2_1": None if ref is None else ref["mae_constante_validos"],
            }
        cache[lbl] = r
        return r

    def por_celula_modelo(bloco_n: int, cel: str, tipo: str, seed: int, n_sort: int):
        """First n_sort draws (by draw index) of one cell, model and training seed, keyed by split seed.
        Block 2 reads the block-1 runs."""
        runs = [c for c in plano if c["bloco"] == (1 if bloco_n == 2 else bloco_n) and c["tipo"] == tipo
                and f"{c['cidade']}_{c['quadrante']}" == cel and c["seed_treino"] == seed]
        runs = sorted(runs, key=lambda c: c["indice_sorteio"])[:n_sort]
        return {c["split_seed"]: carregar(c) for c in runs}

    def agrega_bloco_simples(n: int) -> dict:
        out = {"artefato": f"agregado_G1_bloco{n}", "criterio_sha256": sha_crit, "plano_sha256": sha256(PLANO),
               "limiares_do_criterio_db": {"ruido_repeticao_gnn": RUIDO_REPETICAO_DB, "ruido_sentinela": RUIDO_SENTINELA_DB},
               "celulas": {}}
        for cel in CELULAS_POR_BLOCO[n]:
            g = por_celula_modelo(n, cel, "gnn", 42, 20)
            m = por_celula_modelo(n, cel, "mlp", 42, 20)
            ordem = [c["split_seed"] for c in plano if c["bloco"] == n and f"{c['cidade']}_{c['quadrante']}" == cel and c["tipo"] == "gnn"]
            ss = [s for s in ordem if s in g and s in m]
            bloco = {"sorteios": ss, "sem_validos": [s for s in ss if g[s]["sem_validos"] or m[s]["sem_validos"]], "por_modelo": {}}
            for nome, store in (("gnn", g), ("mlp", m)):
                ok = [s for s in ss if not store[s]["sem_validos"]]
                v = [store[s]["mae"]["mae_rssi_validos_db"] for s in ok]
                d = dp(v)
                const = [store[s]["mae_constante_validos_2_1"] for s in ok]
                bloco["por_modelo"][nome] = {
                    "n_sorteios_com_validos": len(ok),
                    "mae_validos_por_sorteio": {str(s): store[s]["mae"]["mae_rssi_validos_db"] for s in ok},
                    "dp_entre_sorteios_validos_db": d,
                    "razao_dp_sobre_0_132": razao(d, RUIDO_REPETICAO_DB),
                    "dp_ge_3x_0_132": (None if d is None else bool(d >= 3 * RUIDO_REPETICAO_DB)),
                    "dp_lt_2x_0_132": (None if d is None else bool(d < 2 * RUIDO_REPETICAO_DB)),
                    "correlacao_com_constante_validos": {"pearson": pearson(v, const), "spearman": spearman(v, const), "n": len(ok)},
                    "dp_entre_sorteios_demais_populacoes_db": {p: dp([store[s]["mae"][p] for s in ok]) for p in POPS},
                    "inversao_vs_baselines_validos": {b: sum(1 for s in ok if store[s]["baselines_test"][b] is not None and
                                                              store[s]["mae"]["mae_rssi_validos_db"] < store[s]["baselines_test"][b])
                                                      for b in BASELINES},
                    "proveniencia_toda_ok": all(store[s]["proveniencia_ok"] for s in ok),
                    "geometria_declarada": sorted({(store[s]["grid_km"], store[s]["buffer_km"]) for s in ok}),
                    "dist_min_entre_particoes_km_min": min([x for s in ok for x in store[s]["dist_min_entre_particoes_km"].values() if x is not None], default=None),
                    "bate_com_2_1_todos": all((store[s]["bate_com_2_1"] or {}).get(k) for s in ok for k in ("n_test_igual", "n_validos_igual", "n_todos_igual")),
                    "melhores_epocas": [store[s]["melhor_epoca"] for s in ok],
                }
            par = [s for s in ss if s not in bloco["sem_validos"]]
            dif_sent = {str(s): (abs(g[s]["mae"]["mae_rssi_sentinela_db"] - m[s]["mae"]["mae_rssi_sentinela_db"])
                                 if g[s]["mae"]["mae_rssi_sentinela_db"] is not None and m[s]["mae"]["mae_rssi_sentinela_db"] is not None else None)
                        for s in par}
            md = med(list(dif_sent.values()))
            bloco["paridade_sentinela"] = {"abs_gnn_menos_mlp_por_sorteio": dif_sent, "mediana_db": md,
                                           "mediana_le_0_117": (None if md is None else bool(md <= RUIDO_SENTINELA_DB)),
                                           "mesma_particao_gnn_mlp": all(g[s]["mae"]["idx_sha256"] == m[s]["mae"]["idx_sha256"] for s in par)}
            out["celulas"][cel] = bloco
        # Arithmetic count over the two cells of this block; the criterion's rule counts cells
        # over two blocks together.
        out["contagem_mecanica"] = {
            "celulas_com_dp_ge_3x_gnn_e_mlp": [c for c, b in out["celulas"].items()
                                                 if b["por_modelo"]["gnn"]["dp_ge_3x_0_132"] and b["por_modelo"]["mlp"]["dp_ge_3x_0_132"]],
            "celulas_com_dp_lt_2x_gnn_e_mlp": [c for c, b in out["celulas"].items()
                                                 if b["por_modelo"]["gnn"]["dp_lt_2x_0_132"] and b["por_modelo"]["mlp"]["dp_lt_2x_0_132"]],
            "celulas_paridade_le_0_117": [c for c, b in out["celulas"].items() if b["paridade_sentinela"]["mediana_le_0_117"]],
            "nota": "contagem aritmetica; REFORCA/DELIMITA e leitura do rigor, nao deste script"}
        return out

    def agrega_bloco2() -> dict:
        out = {"artefato": "agregado_G1_bloco2", "criterio_sha256": sha_crit, "plano_sha256": sha256(PLANO), "celulas": {},
               "definicao_dp_entre_sementes": ("dp pooled dentro do sorteio: raiz da media, sobre os 5 sorteios, da variancia (ddof=1) "
                                              "do MAE_validos entre as 3 sementes (42, 43, 44); o criterio nao define o estimador, "
                                              "esta e a escolha deste script (declarada)."),
               "nota_componentes": "ANOVA de dois fatores sem repeticao, sorteio (5) x semente (3); componentes por quadrados medios esperados, truncados em 0 (brutos tambem reportados)"}
        for cel in CELULAS_POR_BLOCO[2]:
            ent = {"gnn": {}, "mlp": {}}
            for tipo in ("gnn", "mlp"):
                por_semente = {}
                for sd in (42, 43, 44):
                    por_semente[sd] = {}
                    cs = [c for c in plano if c["tipo"] == tipo and f"{c['cidade']}_{c['quadrante']}" == cel
                          and c["seed_treino"] == sd and c["bloco"] in ((1,) if sd == 42 else (2,))]
                    cs = sorted(cs, key=lambda c: c["indice_sorteio"])[:5]
                    for c in cs:
                        por_semente[sd][c["split_seed"]] = carregar(c)
                sorteios = [c["split_seed"] for c in plano if c["bloco"] == 2 and c["seed_treino"] == 43
                            and c["tipo"] == tipo and f"{c['cidade']}_{c['quadrante']}" == cel]
                lin = [s for s in sorteios if all(s in por_semente[sd] and not por_semente[sd][s]["sem_validos"] for sd in (42, 43, 44))]
                M = np.array([[por_semente[sd][s]["mae"]["mae_rssi_validos_db"] for sd in (42, 43, 44)] for s in lin], float)
                # Pooled between-seed SD: root of the mean, over draws, of the across-seed variance (ddof = 1).
                var_dentro = [float(np.var(M[i], ddof=1)) for i in range(len(lin))]
                dp_sem = float(np.sqrt(np.mean(var_dentro))) if lin else None
                g20 = por_celula_modelo(1, cel, tipo, 42, 20)
                dp20 = dp([g20[s]["mae"]["mae_rssi_validos_db"] for s in g20 if not g20[s]["sem_validos"]])
                dp5 = dp(list(M.mean(axis=1))) if len(lin) >= 2 else None
                ent[tipo] = {
                    "sorteios_usados": lin, "matriz_mae_validos_sorteio_x_semente_42_43_44": M.tolist(),
                    "dp_entre_sementes_pooled_db": dp_sem,
                    "dp_entre_sorteios_20_semente42_db": dp20,
                    "razao_dp_sorteios20_sobre_dp_sementes": razao(dp20, dp_sem),
                    "dp_das_medias_por_sorteio_5_db": dp5,
                    "razao_dp_sorteios5_sobre_dp_sementes": razao(dp5, dp_sem),
                    "anova_dois_fatores_sem_repeticao": (anova2_sem_repeticao(M) if len(lin) >= 3 else None),
                    "dp_entre_sementes_sobre_0_132": razao(dp_sem, RUIDO_REPETICAO_DB)}
            out["celulas"][cel] = ent
        out["contagem_mecanica"] = {"celulas_componente_sorteio_ge_semente_gnn_e_mlp": [
            c for c, e in out["celulas"].items()
            if all((e[t]["anova_dois_fatores_sem_repeticao"] or {}).get("componente_sorteio_ge_semente") for t in ("gnn", "mlp"))],
            "nota": "contagem aritmetica; leitura e do rigor"}
        return out

    alvo = [args.bloco] if args.bloco else [1, 2, 3]
    rc = 0
    for n in alvo:
        if not bloco_completo(n) or (n == 2 and not bloco_completo(1)):
            falt = [c["run_label"] for c in plano if c["bloco"] in ((1, 2) if n == 2 else (n,)) and not completa(c)]
            print(f"bloco {n}: RECUSADO (incompleto, {len(falt)} corridas sem resultado); nao grava resultado parcial")
            rc = 2
            continue
        out = agrega_bloco2() if n == 2 else agrega_bloco_simples(n)
        path = G1 / f"agregado_G1_bloco{n}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"bloco {n}: gravado {path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
