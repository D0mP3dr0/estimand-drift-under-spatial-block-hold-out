#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Error drift of a constant predictor on a continuous target without sentinel (terrain elevation).

Question: with a target defined at every node (elevation, column 0 of `features_raw`, normalised by city by the
producer as (z - elev_mean_city) / elev_std_city), does the MAE of a constant predictor still drift across split
draws of the block hold-out, and is the hold-out worth far fewer independent nodes than a simple random sample?
Same 16 cells, blocks (g = 10 km), buffer (b = 2 km; b = 0 as a secondary reading), fractions 0.70/0.15/0.15 and the
same 60 split seeds as the error-drift test of the baselines. The decision criterion and its addendum, written
before the run, are criteria/criterio_G2_deriva_alvo_continuo.json and criteria/criterio_G2_adendo1.json; their
SHA-256 are checked on start and the script aborts on a mismatch.

Per draw: Err = mean |y - c| over the retained test nodes and n = their number, with c = c_ref fixed on one
reference draw (split seed 42; median of the target over its training nodes, same shuffle rule as
v3_12_R2_referencia_desenho.py). Per cell: CV_d = SD(Err) / mean(Err) (ddof = 1) and the design effect
deff = Var(Err) / mean over draws of V_SRS(n), V_SRS = (1 - n / N_U) S2_U / n, S2_U the variance (ddof = 1) of
|y - c_ref| over the cell's nodes; n_ef = mean(n) / deff. The decision statistics are the medians over the 16
cells; `contagem_mecanica_limiares` only counts them against the thresholds of the criterion (CV_d >= 0.05 and
deff >= 10) and does not interpret them.
Variants: primary (elevation, c_ref, b = 2 km); negative control (elevation permuted among the nodes of each
cell with np.random.default_rng(20261004 + cell index); the median deff must lie in [0.5, 2]); b = 0; median
refitted at each draw; elevation restricted to valid nodes; NDVI (column 12).
Gates: P1a-P1d (the column equals `dem.x[:, 0]` of the graph tensor, city constants identical, quadrant coverage and
moments, share of imputed nodata <= 1 %), P2 (no NaN or inf), P3 (n_test per draw equal to the one stored by the
shared loop of R5/R6, `fase5/_parcial_laco`, not distributed), P4 (share of NDVI == 0 <= 1 %), P5 (negative control).

Reuses by import, unchanged and with a digest check, analysis/v3_13_laco_comum.py (seeds, cell loading, block sums) and, through it,
split_espacial_3vias of analysis/v3_2.1_3.1_deriva_calibracao.py. Inputs: the reference-field tensors and the graph
tensors (data on request), fase2/2.1_deriva_erro_baselines_16x60rnd.json. Outputs: fase5/_parcial_G2/<cell>.json,
fase5/_parcial_G2/_P1_cidades.json, fase5/G2_resumo.json (fase5/G2_abortado_v2.json if a gate aborts the run).
CPU only, no training.

Usage:
  python analysis/v3_14_G2_alvo_continuo_v2.py --executar        # gates, 16 cells (4 processes), consolidation
  python analysis/v3_14_G2_alvo_continuo_v2.py --celula bauru_Q1 # one cell (resumable)
  python analysis/v3_14_G2_alvo_continuo_v2.py --consolidar      # 16 partial records -> G2_resumo.json
  python analysis/v3_14_G2_alvo_continuo_v2.py --fumaca <dir>    # smoke test on synthetic data
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
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
CRITERIO = V3 / "criterios" / "criterio_G2_deriva_alvo_continuo.json"
CRITERIO_SHA256 = "2beeb34142828ce0575b74f80f511d7a1f328895cdf94c9eef91d1084511a8f4"
ADENDO = V3 / "criterios" / "criterio_G2_adendo1.json"
ADENDO_SHA256 = "ee56ae8ed9667d641f9305b4c226140ce3bceea5eaf4aaa4164cf0b70964f690"
OUT = V3 / "fase5"
PARCIAL_DIR = OUT / "_parcial_G2"
LACO_PARCIAL_DIR = OUT / "_parcial_laco"
OUT_RESUMO = OUT / "G2_resumo.json"
OUT_ABORTO = OUT / "G2_abortado_v2.json"
PRE_JSON_NOME = "_P1_cidades.json"
GPU_DIR = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")

CELULAS = [f"{c}_{q}" for c in ("bauru", "campinas", "lins", "sorocaba") for q in ("Q1", "Q2", "Q3", "Q4")]
SEED_REF = 42
PERM_SEED_BASE = 20261004
COL_ELEV, COL_NDVI = 0, 12
LIM_A, LIM_B = 0.05, 10.0
P1_MEDIA, P1_DP = (-1e-3, 1e-3), (0.999, 1.001)
P1D_EPS, P1D_MAX = 1e-6, 0.01
CIDADES = ("bauru", "campinas", "lins", "sorocaba")
QUADS = ("Q1", "Q2", "Q3", "Q4")
P4_MAX = 0.01
P5_FAIXA = (0.5, 2.0)
VARIANTES = ("prim", "perm", "b0", "med_sorteio", "validos", "ndvi")
SECUNDARIAS = ("b0", "med_sorteio", "validos", "ndvi")

_adendo_sha_real = hashlib.sha256(ADENDO.read_bytes()).hexdigest()
if _adendo_sha_real != ADENDO_SHA256:
    raise SystemExit(f"ABORTA: sha do adendo G2 diverge ({_adendo_sha_real})")
_laco_sha_real = hashlib.sha256(LACO_PATH.read_bytes()).hexdigest()
if _laco_sha_real != LACO_SHA256:
    raise SystemExit(f"ABORTA: sha do laco_comum diverge ({_laco_sha_real})")
_crit_sha_real = hashlib.sha256(CRITERIO.read_bytes()).hexdigest()
if _crit_sha_real != CRITERIO_SHA256:
    raise SystemExit(f"ABORTA: sha do criterio G2 diverge ({_crit_sha_real})")

_spec = importlib.util.spec_from_file_location("v3_13_laco_comum", LACO_PATH)
laco = importlib.util.module_from_spec(_spec)
sys.modules["v3_13_laco_comum"] = laco
_spec.loader.exec_module(laco)
orig = laco.orig
ORIG_SHA256 = laco.ORIG_SHA256
log = laco.log
agora = laco.agora


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def portoes_p1_p2(y: np.ndarray) -> dict:
    fin = bool(np.isfinite(y).all())
    media = float(np.mean(y)) if fin else None
    dp0 = float(np.std(y, ddof=0)) if fin else None
    p1 = bool(fin and P1_MEDIA[0] <= media <= P1_MEDIA[1] and P1_DP[0] <= dp0 <= P1_DP[1])
    return {"P1": {"media": media, "dp_ddof0": dp0, "faixa_media": list(P1_MEDIA), "faixa_dp": list(P1_DP),
                   "passou": p1},
            "P2": {"n_nao_finitos": int((~np.isfinite(y)).sum()), "passou": fin}}


def referencia_treino(gid: np.ndarray) -> np.ndarray:
    grupos = np.unique(gid)
    n_g = len(grupos)
    rng42 = np.random.RandomState(SEED_REF)
    emb42 = grupos.copy()
    rng42.shuffle(emb42)
    n_tr42 = max(1, int(round(orig.FRACS[0] * n_g)))
    n_va42b = max(1, int(round(orig.FRACS[1] * n_g)))
    if n_tr42 + n_va42b >= n_g:
        n_tr42 = max(1, n_g - 2)
        n_va42b = 1
    return np.isin(gid, emb42[:n_tr42])


def avaliar(y: np.ndarray, c: float, te: np.ndarray, gid: np.ndarray, com_blocos: bool = False) -> dict:
    n = int(te.size)
    if n == 0:
        return {"Err": None, "n": 0}
    loss = np.abs(y[te] - c)
    r = {"Err": float(np.mean(loss)), "n": n}
    if com_blocos:
        ub, inv = np.unique(gid[te], return_inverse=True)
        r["blocos"] = laco.somas_por_bloco(inv, int(ub.size), ub, loss, np.ones(n, dtype=bool))
    return r


def estatisticas(err, n, N_U: int, S2_U: float) -> dict:
    err = np.array([np.nan if e is None else e for e in err], dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    ok = np.isfinite(err) & (n > 0)
    e, nn = err[ok], n[ok]
    media = float(e.mean())
    dp = float(e.std(ddof=1))
    vaas = (1.0 - nn / N_U) * S2_U / nn
    vaas_med = float(vaas.mean())
    deff = (dp ** 2) / vaas_med
    n_med = float(nn.mean())
    return {"n_sorteios_usados": int(ok.sum()), "n_sorteios_excluidos_n0": int((~ok).sum()),
            "media_Err": media, "dp_Err_ddof1": dp, "CV_d": dp / media, "var_Err": dp ** 2,
            "N_U": int(N_U), "S2_U": float(S2_U), "V_AAS_medio": vaas_med, "deff": float(deff),
            "n_medio": n_med, "n_ef": n_med / deff}


def s2_u(y: np.ndarray, c: float, mask=None) -> float:
    v = np.abs((y if mask is None else y[mask]) - c)
    return float(np.var(v, ddof=1))


def nucleo_celula(pos_m, gid, y, ndvi, valido, seeds, idx_cel):
    rng = np.random.default_rng(PERM_SEED_BASE + idx_cel)
    y_perm = y[rng.permutation(y.size)]
    m_ref = referencia_treino(gid)
    c_ref = float(np.median(y[m_ref]))
    c_ref_perm = float(np.median(y_perm[m_ref]))
    c_ref_ndvi = float(np.median(ndvi[m_ref]))
    N = int(y.size)
    N_val = int(valido.sum())
    pop = {"prim": (N, s2_u(y, c_ref)), "perm": (N, s2_u(y_perm, c_ref_perm)),
           "b0": (N, s2_u(y, c_ref)), "med_sorteio": (N, s2_u(y, c_ref)),
           "validos": (N_val, s2_u(y, c_ref, valido)), "ndvi": (N, s2_u(ndvi, c_ref_ndvi))}
    ps = []
    for seed in seeds:
        p2 = orig.split_espacial_3vias(pos_m, orig.GRID_KM, orig.BUFFER_KM, orig.FRACS, seed)
        tr, te = p2["train"], p2["test"]
        if tr.size == 0 or te.size == 0:
            raise RuntimeError(f"particao vazia no seed {seed}")
        p0 = orig.split_espacial_3vias(pos_m, orig.GRID_KM, 0.0, orig.FRACS, seed)
        te_v = te[valido[te]]
        c_med = float(np.median(y[tr]))
        ps.append({"split_seed": int(seed), "n_train": int(tr.size), "n_test": int(te.size),
                   "n_test_b0": int(p0["test"].size),
                   "prim": avaliar(y, c_ref, te, gid, com_blocos=True),
                   "perm": avaliar(y_perm, c_ref_perm, te, gid),
                   "b0": avaliar(y, c_ref, p0["test"], gid),
                   "med_sorteio": {**avaliar(y, c_med, te, gid), "c": c_med},
                   "validos": avaliar(y, c_ref, te_v, gid),
                   "ndvi": avaliar(ndvi, c_ref_ndvi, te, gid)})
    est = {v: estatisticas([s[v]["Err"] for s in ps], [s[v]["n"] for s in ps], *pop[v]) for v in VARIANTES}
    return {"c_ref": {"prim": c_ref, "perm": c_ref_perm, "ndvi": c_ref_ndvi},
            "populacao": {v: {"N_U": pop[v][0], "S2_U": pop[v][1]} for v in VARIANTES},
            "por_sorteio": ps, "estatisticas": est}


def agregar(estat_por_cel: dict, elev_std: dict) -> dict:
    cels = list(estat_por_cel)
    cv = np.array([estat_por_cel[c]["CV_d"] for c in cels])
    de = np.array([estat_por_cel[c]["deff"] for c in cels])
    ne = np.array([estat_por_cel[c]["n_ef"] for c in cels])
    nm = np.array([estat_por_cel[c]["n_medio"] for c in cels])
    f = lambda a: {"mediana": float(np.median(a)), "media": float(a.mean()),  # noqa: E731
                   "dp_ddof1": float(a.std(ddof=1)), "min": float(a.min()), "max": float(a.max())}
    s = np.array([elev_std[c] for c in cels])
    dp_m = np.array([estat_por_cel[c]["dp_Err_ddof1"] for c in cels]) * s
    med_m = np.array([estat_por_cel[c]["media_Err"] for c in cels]) * s
    dp_entre = float(med_m.std(ddof=1))
    return {"n_celulas": len(cels), "CV_d": f(cv), "deff": f(de), "n_ef": f(ne), "n_medio": f(nm),
            "razao_entre_celulas_descritiva": {
                "dp_medio_entre_sorteios_m": float(dp_m.mean()), "dp_entre_medias_das_celulas_m": dp_entre,
                "razao": float(dp_m.mean() / dp_entre) if dp_entre else None,
                "nota": "em metros = MAE multiplicado pelo elev_std da cidade (normalization.elev_std do <cidade>_v19_<Q>_gpu.pt; "
                        "elev_mean nao entra no MAE); dp entre celulas ddof=1; descritiva, sem papel nos ramos (adendo 1)"}}


def contagem_mecanica(prim_agr: dict) -> dict:
    mcv, mdf = prim_agr["CV_d"]["mediana"], prim_agr["deff"]["mediana"]
    A, B = bool(mcv >= LIM_A), bool(mdf >= LIM_B)
    ramo = "REFORCA" if A else ("DELIMITA" if B else "INVERTE")
    return {"mediana_CV_d": mcv, "mediana_deff": mdf, "limiar_A_CV_d_maior_ou_igual": LIM_A,
            "limiar_B_deff_maior_ou_igual": LIM_B, "A_verdadeiro": A, "B_verdadeiro": B,
            "ramo_selecionado_pelas_duas_respostas": ramo,
            "regra_aplicada": "criterio.leitura: A verdadeiro -> REFORCA; A falso e B verdadeiro -> DELIMITA; "
                              "A falso e B falso -> INVERTE"}


def ler_features(tensor_path: str):
    rf, modo = orig.carregar_tensor(Path(tensor_path))
    fr = rf["terrain"].features_raw
    y = np.asarray(fr[:, COL_ELEV], dtype=np.float64).copy()
    ndvi = np.asarray(fr[:, COL_NDVI], dtype=np.float64).copy()
    del rf, fr
    return y, ndvi


def ler_gpu(chave: str) -> dict:
    cidade, quad = chave.split("_")
    p = GPU_DIR / f"{cidade}_v19_{quad}_gpu.pt"
    g = orig.carregar_tensor(p)[0]
    nm = g["normalization"]
    rm = g["raster_meta"] if "raster_meta" in g else None
    ds = None
    if rm is not None and "dem_shape" in rm:
        ds = tuple(int(v) for v in rm["dem_shape"])
    return {"arquivo": str(p), "elev_mean": float(nm["elev_mean"]), "elev_std": float(nm["elev_std"]),
            "dem_shape": ds, "quadrante": dict(g["quadrant"]) if "quadrant" in g else None,
            "x0": np.asarray(g["dem"].x[:, COL_ELEV], dtype=np.float64),
            "pos": np.ascontiguousarray(np.asarray(g["dem"].pos, dtype=np.float32))}


def portoes_cidade(cidade: str, q: dict) -> dict:
    quads = list(q)
    r = {"cidade": cidade}
    a = {f"{cidade}_{k}": bool(np.array_equal(q[k]["y"], q[k]["x0"])) for k in quads}
    r["P1a_identidade"] = {"por_celula": a, "passou": all(a.values())}
    hm = {k: float(q[k]["elev_mean"]).hex() for k in quads}
    hs = {k: float(q[k]["elev_std"]).hex() for k in quads}
    r["P1b_constantes"] = {"elev_mean": {k: q[k]["elev_mean"] for k in quads}, "elev_std": {k: q[k]["elev_std"] for k in quads},
                           "elev_mean_hex": hm, "elev_std_hex": hs,
                           "passou": len(set(hm.values())) == 1 and len(set(hs.values())) == 1}
    n_por = {k: int(q[k]["y"].size) for k in quads}
    soma = int(sum(n_por.values()))
    pos = np.concatenate([q[k]["pos"] for k in quads], axis=0)
    assert pos.dtype == np.float32
    n_unicos = int(np.unique(np.ascontiguousarray(pos).view(np.int64).ravel()).size)
    disjuntos = bool(n_unicos == pos.shape[0])
    ds = {q[k]["dem_shape"] for k in quads}
    hw = None
    if len(ds) == 1 and None not in ds:
        d0 = next(iter(ds))
        hw = int(d0[0]) * int(d0[1])
    c0_ok = bool(disjuntos and hw is not None and soma == hw)
    r["P1c0_cobertura"] = {"n_nos_por_quadrante": n_por, "soma_nos": soma, "H_vezes_W_dem_shape": hw,
                           "dem_shape": None if hw is None else list(next(iter(ds))),
                           "posicoes_unicas": n_unicos, "posicoes_total": int(pos.shape[0]),
                           "quadrantes_disjuntos": disjuntos, "avaliavel": hw is not None, "passou": c0_ok,
                           "registro_quadrantes_gpu_pt": {k: q[k].get("quadrante") for k in quads}}
    del pos
    if c0_ok:
        ns = np.array([n_por[k] for k in quads], dtype=np.float64)
        ms = np.array([float(np.mean(q[k]["y"])) for k in quads])
        m2 = np.array([float(np.sum((q[k]["y"] - ms[i]) ** 2)) for i, k in enumerate(quads)])
        mu = float((ns * ms).sum() / ns.sum())
        var = float((m2.sum() + (ns * (ms - mu) ** 2).sum()) / ns.sum())
        dp = float(np.sqrt(var))
        ok = bool(P1_MEDIA[0] <= mu <= P1_MEDIA[1] and P1_DP[0] <= dp <= P1_DP[1])
        r["P1c_momentos"] = {"aplicavel": True, "media_ponderada": mu, "dp_agregado_ddof0": dp,
                             "faixa_media": list(P1_MEDIA), "faixa_dp": list(P1_DP), "passou": ok}
    else:
        r["P1c_momentos"] = {"aplicavel": False, "nota": "P1c0 nao passou ou nao avaliavel; portao apoiado em P1a e P1b",
                             "passou": None}
    nz = {k: int((np.abs(q[k]["y"]) < P1D_EPS).sum()) for k in quads}
    frac = float(sum(nz.values()) / soma)
    r["P1d_nodata"] = {"n_abs_lt_1e-6_por_quadrante": nz, "fracao_cidade": frac, "limite": P1D_MAX,
                       "passou": bool(frac <= P1D_MAX)}
    r["P2_nan"] = {f"{cidade}_{k}": int((~np.isfinite(q[k]["y"])).sum()) for k in quads}
    r["abortar"] = bool(not r["P1a_identidade"]["passou"] or not r["P1b_constantes"]["passou"]
                        or (r["P1c_momentos"]["aplicavel"] and not r["P1c_momentos"]["passou"])
                        or not r["P1d_nodata"]["passou"] or any(v > 0 for v in r["P2_nan"].values()))
    return r


def precheck_cidade(cidade: str) -> dict:
    q = {}
    for quad in QUADS:
        chave = f"{cidade}_{quad}"
        y, _ = ler_features(str(orig.TENSOR_DIR / f"transfer_dataset_{cidade}_v19_{quad}_enriched_cftudo.pt"))
        g = ler_gpu(chave)
        q[quad] = {"y": y, "x0": g["x0"], "elev_mean": g["elev_mean"], "elev_std": g["elev_std"], "pos": g["pos"],
                   "dem_shape": g["dem_shape"], "quadrante": g["quadrante"]}
    return portoes_cidade(cidade, q)


def rodar_celula(chave: str) -> dict:
    t0, ru0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    idx = CELULAS.index(chave)
    seeds = laco.carregar_seeds()
    c = laco.carregar_celula(chave)
    y, ndvi = ler_features(c["tensor_path"])
    assert y.size == c["n_total"] == ndvi.size, "features_raw desalinhado do tensor da celula"
    g = ler_gpu(chave)
    p1a = bool(np.array_equal(y, g["x0"]))
    p2 = portoes_p1_p2(y)["P2"]
    rec = {"celula": chave, "indice_da_celula": idx, "sha256_manifest_tensor": c["sha256_manifest"],
           "n_nodes_total": c["n_total"], "N_blocos_ocupados": c["N_blocos_ocupados"],
           "portoes_P1a_P2": {"P1a_identidade": p1a, "P2": p2}}
    if not (p1a and p2["passou"]):
        rec["status"] = "abortado_P1a_P2"
        return rec
    rec["P4_ndvi"] = {"fracao_zero_exato": float(np.mean(ndvi == 0.0)), "n_zero_exato": int((ndvi == 0.0).sum())}
    rec["elev_std_m"] = g["elev_std"]
    rec["elev_mean_m"] = g["elev_mean"]
    del g
    rec.update(nucleo_celula(c["pos_m"], c["gid"], y, ndvi, ~c["sent"], seeds, idx))
    rec["n_nos_validos"] = int((~c["sent"]).sum())
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    rec["status"] = "ok"
    rec["tempo_relogio_s"] = time.perf_counter() - t0
    rec["tempo_cpu_s"] = (ru1.ru_utime + ru1.ru_stime) - (ru0.ru_utime + ru0.ru_stime)
    return rec


def rodar_uma(chave: str) -> dict:
    PARCIAL_DIR.mkdir(parents=True, exist_ok=True)
    arq = PARCIAL_DIR / f"{chave}.json"
    if arq.exists():
        d = json.loads(arq.read_text(encoding="utf-8"))
        if d.get("status") == "ok" and d.get("script_sha256") == SCRIPT_SHA256:
            log(f"{chave}: parcial ok, pulando")
            return d
    log(f"=== {chave}: G2 ===")
    try:
        rec = rodar_celula(chave)
    except Exception as e:  # noqa: BLE001
        rec = {"celula": chave, "status": "erro_excecao", "erro": repr(e), "traceback": traceback.format_exc()}
    rec["script_sha256"] = SCRIPT_SHA256
    rec["criterio_sha256"] = CRITERIO_SHA256
    rec["data_utc"] = agora()
    arq.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"{chave}: status={rec['status']} -> {arq}")
    return rec


def p3_contra_laco(rec: dict, laco_rec: dict) -> dict:
    ref = laco_rec["por_sorteio"]
    mine = rec["por_sorteio"]
    iguais_seed = [a["split_seed"] for a in mine] == [b["split_seed"] for b in ref]
    difs = [(a["split_seed"], a["n_test"], b["n_test"]) for a, b in zip(mine, ref) if a["n_test"] != b["n_test"]]
    return {"n_sorteios": len(mine), "mesmos_split_seed_na_mesma_ordem": bool(iguais_seed),
            "n_divergentes": len(difs), "divergentes": difs[:10]}


def montar_resumo(parciais: dict, laco_parciais: dict, comando: str, fontes_extra: dict, destino: Path,
                  seeds: list, pre: dict) -> dict:
    for ch in CELULAS:
        if parciais[ch].get("status") != "ok":
            raise SystemExit(f"ABORTA: parcial {ch} status={parciais[ch].get('status')}")
    est = {v: {} for v in VARIANTES}
    for ch in CELULAS:
        p = parciais[ch]
        for v in VARIANTES:
            e = estatisticas([s[v]["Err"] for s in p["por_sorteio"]], [s[v]["n"] for s in p["por_sorteio"]],
                             p["populacao"][v]["N_U"], p["populacao"][v]["S2_U"])
            assert e == p["estatisticas"][v], f"estatisticas do parcial {ch}/{v} nao reproduzem dos valores por sorteio"
            est[v][ch] = e
    elev_std = {ch: parciais[ch]["elev_std_m"] for ch in CELULAS}

    p2 = {ch: parciais[ch]["portoes_P1a_P2"]["P2"] for ch in CELULAS}
    p3c = {ch: p3_contra_laco(parciais[ch], laco_parciais[ch]) for ch in CELULAS}
    p3_ok = all(v["n_divergentes"] == 0 and v["mesmos_split_seed_na_mesma_ordem"] and v["n_sorteios"] == 60
                for v in p3c.values())
    p4f = {ch: parciais[ch]["P4_ndvi"]["fracao_zero_exato"] for ch in CELULAS}
    p4_max = max(p4f.values())
    p4_ok = bool(p4_max <= P4_MAX)
    mediana_deff_perm = float(np.median([est["perm"][ch]["deff"] for ch in CELULAS]))
    p5_ok = bool(P5_FAIXA[0] <= mediana_deff_perm <= P5_FAIXA[1])
    portoes = {
        "P1a_identidade": {"valor_medido": {ch: pre[ch.split("_")[0]]["P1a_identidade"]["por_celula"][ch] for ch in CELULAS},
                           "criterio": "features_raw[:,0] == dem.x[:,0] do _gpu.pt, igualdade exata, 16 celulas",
                           "passou": all(pre[c]["P1a_identidade"]["passou"] for c in CIDADES)
                           and all(parciais[ch]["portoes_P1a_P2"]["P1a_identidade"] for ch in CELULAS)},
        "P1b_constantes": {"valor_medido": {c: pre[c]["P1b_constantes"] for c in CIDADES},
                           "criterio": "elev_mean e elev_std iguais bit a bit nos 4 _gpu.pt de cada cidade",
                           "passou": all(pre[c]["P1b_constantes"]["passou"] for c in CIDADES)},
        "P1c0_cobertura": {"valor_medido": {c: pre[c]["P1c0_cobertura"] for c in CIDADES},
                           "criterio": "quadrantes disjuntos e soma dos nos == H*W de raster_meta['dem_shape']; registrado",
                           "passou": all(pre[c]["P1c0_cobertura"]["passou"] for c in CIDADES)},
        "P1c_momentos": {"valor_medido": {c: pre[c]["P1c_momentos"] for c in CIDADES},
                         "criterio": "so se P1c0 passar: media ponderada em [-1e-3,1e-3], dp agregado ddof0 em [0,999;1,001]; "
                                     "senao 'nao aplicavel'",
                         "passou": (None if not all(pre[c]["P1c_momentos"]["aplicavel"] for c in CIDADES)
                                    else all(pre[c]["P1c_momentos"]["passou"] for c in CIDADES)),
                         "aplicavel_por_cidade": {c: pre[c]["P1c_momentos"]["aplicavel"] for c in CIDADES}},
        "P1d_nodata": {"valor_medido": {c: {"fracao_cidade": pre[c]["P1d_nodata"]["fracao_cidade"],
                                            "n_abs_lt_1e-6_por_quadrante": pre[c]["P1d_nodata"]["n_abs_lt_1e-6_por_quadrante"]}
                                        for c in CIDADES},
                       "criterio": "fracao de nos com |features_raw[:,0]| < 1e-6 por cidade <= 1 %; senao aborta",
                       "passou": all(pre[c]["P1d_nodata"]["passou"] for c in CIDADES)},
        "P2_nan": {"valor_medido": {ch: p2[ch]["n_nao_finitos"] for ch in CELULAS}, "criterio": "zero NaN/inf",
                   "passou": all(v["passou"] for v in p2.values())},
        "P3_particao": {"valor_medido": {"n_sorteios_total": sum(v["n_sorteios"] for v in p3c.values()),
                                         "n_divergentes_total": sum(v["n_divergentes"] for v in p3c.values()),
                                         "por_celula": p3c},
                        "criterio": "n_test por sorteio == fase5/_parcial_laco/<cel>.json nos 960 sorteios",
                        "passou": p3_ok},
        "P4_ndvi": {"valor_medido": {"fracao_max_entre_celulas": p4_max, "por_celula": p4f},
                    "criterio": "fracao de nos com features_raw[:,12]==0 exato <= 1 %; senao NDVI so registro",
                    "passou": p4_ok},
        "P5_controle_negativo": {"valor_medido": {"mediana_deff_alvo_permutado": mediana_deff_perm,
                                                  "deff_por_celula": {ch: est["perm"][ch]["deff"] for ch in CELULAS}},
                                 "criterio": "mediana(deff) do alvo permutado em [0,5; 2]; fora disso a rodada e invalida",
                                 "passou": p5_ok},
    }
    p1_ok = (portoes["P1a_identidade"]["passou"] and portoes["P1b_constantes"]["passou"]
             and portoes["P1c_momentos"]["passou"] is not False and portoes["P1d_nodata"]["passou"])
    if not (p1_ok and portoes["P2_nan"]["passou"] and p3_ok):
        raise SystemExit("ABORTA: P1/P2/P3 falharam na consolidacao: " + json.dumps(
            {k: v["passou"] for k, v in portoes.items()}))
    rodada_valida = p5_ok

    prim = agregar(est["prim"], elev_std)
    por_cel = lambda v: {ch: {k: est[v][ch][k] for k in ("media_Err", "dp_Err_ddof1", "CV_d", "deff", "n_ef",  # noqa: E731
                                                         "n_medio", "n_sorteios_usados", "n_sorteios_excluidos_n0",
                                                         "N_U", "S2_U", "V_AAS_medio")} for ch in CELULAS}
    rot = "secundario/descritivo"
    res = {
        "id": "G2_deriva_alvo_continuo_sem_sentinela_com_adendo1", "criterio": str(CRITERIO), "criterio_sha256": CRITERIO_SHA256,
        "criterio_fixado_em": "2026-10-04T10:08:04-03:00",
        "adendo1": str(ADENDO), "adendo1_sha256": ADENDO_SHA256, "adendo1_fixado_em": "2026-10-04T10:23:11-03:00",
        "script_v1_abortado": {"script": str(HERE.with_name("v3_14_G2_alvo_continuo.py")),
                               "aborto": str(OUT / "G2_abortado.json")},
        "script": str(HERE), "script_sha256": SCRIPT_SHA256,
        "laco_importado": str(LACO_PATH), "laco_sha256": LACO_SHA256,
        "script_original_importado": str(laco.ORIG_PATH), "script_original_sha256": ORIG_SHA256,
        "comando_consolidacao": comando, "data_utc": agora(), "venv": sys.executable,
        "sementes": {"split_seed_60": seeds, "split_seed_fonte": str(laco.F2_JSON) + " :: nota_divergencia_seeds.seeds_usados_nesta_rodada",
                     "seed_referencia_c_ref": SEED_REF,
                     "permutacao_P5": f"np.random.default_rng({PERM_SEED_BASE} + indice_da_celula), indice 0-based em CELULAS",
                     "ordem_celulas": CELULAS},
        "colunas": {"elevacao": COL_ELEV, "ndvi": COL_NDVI,
                    "produtor": "GNN_RF_V2/data_raw/generate_graph_v19.py:470 (dem_x[:,0]), :481-482 (colunas 8..13: b02,b03,b04,b08,ndvi,ndwi); "
                                "GNN_RF_V2/01_data/prepare_transfer_dataset_v19.py:140,306 (features_raw = x_dem)"},
        "desenho": {"grid_km": orig.GRID_KM, "buffer_primario_km": orig.BUFFER_KM, "buffer_secundario_km": 0.0,
                    "fracs": list(orig.FRACS), "n_celulas": 16, "n_sorteios": 60},
        "fontes": {"parciais_G2_sha256": {ch: sha256_file(PARCIAL_DIR / f"{ch}.json") for ch in CELULAS},
                   "parciais_laco_sha256": {ch: sha256_file(LACO_PARCIAL_DIR / f"{ch}.json") for ch in CELULAS},
                   "tensores_sha256_manifest": {ch: parciais[ch]["sha256_manifest_tensor"] for ch in CELULAS},
                   "P1_cidades_sha256": sha256_file(PARCIAL_DIR / PRE_JSON_NOME),
                   "parciais_script_sha256_unico": sorted({parciais[ch]["script_sha256"] for ch in CELULAS}),
                   **fontes_extra},
        "tempo": {"cpu_s_soma_celulas": float(sum(parciais[ch]["tempo_cpu_s"] for ch in CELULAS)),
                  "relogio_s_soma_celulas": float(sum(parciais[ch]["tempo_relogio_s"] for ch in CELULAS))},
        "portoes": portoes, "rodada_valida": rodada_valida,
        "primario": {"definicao": "elevacao (features_raw[:,0]; normalizada por cidade pelo produtor, (z - elev_mean_cidade)/elev_std_cidade), constante FIXO c_ref, todos os nos, b = 2 km",
                     "CV_d_redacao": "adimensional, invariante a qualquer transformacao afim do alvo dentro da celula, inclusive a normalizacao por cidade do produtor",
                     "c_ref_por_celula": {ch: parciais[ch]["c_ref"]["prim"] for ch in CELULAS},
                     "por_celula": por_cel("prim"), "agregado_16_celulas": prim},
        "controle_negativo_P5": {"definicao": "elevacao permutada entre os nos da celula, mesmo calculo",
                                 "por_celula": por_cel("perm"), "agregado_16_celulas": agregar(est["perm"], elev_std)},
        "secundarios_descritivos": {
            "rotulo": rot, "nota": "nao mudam o ramo (criterio.leitura)",
            "b0": {"rotulo": rot, "definicao": "mesmo preditor e populacao, b = 0 (sem aparo)",
                   "por_celula": por_cel("b0"), "agregado_16_celulas": agregar(est["b0"], elev_std)},
            "mediana_por_sorteio": {"rotulo": rot, "definicao": "c = mediana de Y no treino de cada sorteio; S2_U de |Y-c_ref|",
                                    "por_celula": por_cel("med_sorteio"),
                                    "agregado_16_celulas": agregar(est["med_sorteio"], elev_std)},
            "elevacao_nos_validos_RSSI": {"rotulo": rot, "definicao": "teste restrito a Z_i = 0 (~sent do laco); N_U = nos validos; "
                                          "sorteios com 0 nos validos no teste excluidos (contados por celula)",
                                          "por_celula": por_cel("validos"),
                                          "agregado_16_celulas": agregar(est["validos"], elev_std)},
            "ndvi": {"rotulo": rot, "definicao": "NDVI (features_raw[:,12]), c = mediana do NDVI no treino de referencia, b = 2 km",
                     "sob_P4": {"passou": p4_ok, "na_leitura": p4_ok},
                     "por_celula": por_cel("ndvi"), "agregado_16_celulas": agregar(est["ndvi"], elev_std)},
        },
    }
    if rodada_valida:
        res["contagem_mecanica_limiares"] = contagem_mecanica(prim)
    else:
        res["contagem_mecanica_limiares"] = None
        res["aviso"] = "P5 fora da faixa: rodada invalida, sem leitura (criterio)"
    destino.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


def consolidar(comando: str) -> dict:
    parciais = {ch: json.loads((PARCIAL_DIR / f"{ch}.json").read_text(encoding="utf-8")) for ch in CELULAS}
    laco_p = {ch: json.loads((LACO_PARCIAL_DIR / f"{ch}.json").read_text(encoding="utf-8")) for ch in CELULAS}
    seeds = laco.carregar_seeds()
    pre = json.loads((PARCIAL_DIR / PRE_JSON_NOME).read_text(encoding="utf-8"))["cidades"]
    fontes_extra = {"seeds_fonte_sha256": sha256_file(laco.F2_JSON)}
    res = montar_resumo(parciais, laco_p, comando, fontes_extra, OUT_RESUMO, seeds, pre)
    log(f"consolidado -> {OUT_RESUMO}; rodada_valida={res['rodada_valida']}")
    return res


def gravar_aborto(motivo: str, detalhe: dict, comando: str) -> None:
    OUT_ABORTO.write_text(json.dumps({
        "id": "G2_abortado_v2", "motivo": motivo, "criterio": str(CRITERIO), "criterio_sha256": CRITERIO_SHA256,
        "adendo1": str(ADENDO), "adendo1_sha256": ADENDO_SHA256,
        "script": str(HERE), "script_sha256": SCRIPT_SHA256, "comando": comando, "data_utc": agora(),
        "venv": sys.executable, "laco_sha256": LACO_SHA256, "script_original_sha256": ORIG_SHA256,
        "detalhe": detalhe, "resultado_calculado": False}, indent=2, ensure_ascii=False), encoding="utf-8")


def executar(comando: str) -> None:
    t0 = time.perf_counter()
    log("pre-checagem P1a-P1d e P2 (4 cidades x 4 quadrantes)")
    with ProcessPoolExecutor(max_workers=4) as ex:
        pre = list(ex.map(precheck_cidade, CIDADES))
    PARCIAL_DIR.mkdir(parents=True, exist_ok=True)
    (PARCIAL_DIR / PRE_JSON_NOME).write_text(json.dumps({
        "id": "G2_P1_por_cidade", "adendo1_sha256": ADENDO_SHA256, "script_sha256": SCRIPT_SHA256, "data_utc": agora(),
        "cidades": {r["cidade"]: r for r in pre}}, indent=2, ensure_ascii=False), encoding="utf-8")
    falhas = [r["cidade"] for r in pre if r["abortar"]]
    if falhas:
        gravar_aborto("portao P1a/P1b/P1c/P1d ou P2 falhou; nada foi calculado alem da propria pre-checagem",
                      {"cidades_que_falharam": falhas, "por_cidade": {r["cidade"]: r for r in pre}}, comando)
        log(f"ABORTA em P1/P2: cidades {falhas} -> {OUT_ABORTO}")
        sys.exit(3)
    log("P1a-P1d e P2 ok; rodando as celulas (4 processos)")
    with ProcessPoolExecutor(max_workers=4) as ex:
        recs = list(ex.map(rodar_uma, CELULAS))
    ruim = [r["celula"] for r in recs if r.get("status") != "ok"]
    if ruim:
        gravar_aborto("celula(s) sem status ok", {"celulas": ruim}, comando)
        sys.exit(4)
    consolidar(comando)
    log(f"tempo total de relogio: {time.perf_counter() - t0:.1f} s")


def fumaca(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    global PARCIAL_DIR, LACO_PARCIAL_DIR  # noqa: PLW0603
    PARCIAL_DIR, LACO_PARCIAL_DIR = dest / "_parcial_G2", dest / "_parcial_laco"
    PARCIAL_DIR.mkdir(exist_ok=True)
    LACO_PARCIAL_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    seeds = [int(s) for s in rng.choice(np.arange(1, 10 ** 6), size=60, replace=False)]
    def cidade_sint(alterar=None):
        zz = rng.standard_normal((200, 200)) * 80 + 400
        zm, zs = float(zz.mean()), float(zz.std())
        zn = ((zz - zm) / zs).astype(np.float32)
        q = {}
        for k, (r0, c0) in zip(QUADS, [(0, 0), (0, 100), (100, 0), (100, 100)]):
            rr, cc = np.meshgrid(np.arange(r0, r0 + 100), np.arange(c0, c0 + 100), indexing="ij")
            y = zn[r0:r0 + 100, c0:c0 + 100].ravel().astype(np.float64)
            q[k] = {"y": y, "x0": y.copy(), "elev_mean": zm, "elev_std": zs, "dem_shape": (200, 200), "quadrante": None,
                    "pos": np.stack([cc.ravel() * 1e-4, rr.ravel() * 1e-4], axis=1).astype(np.float32)}
        if alterar:
            alterar(q)
        return q
    ok = portoes_cidade("sint", cidade_sint())
    assert not ok["abortar"] and ok["P1c0_cobertura"]["passou"] and ok["P1c_momentos"]["passou"], ok
    assert portoes_cidade("sint", cidade_sint(lambda q: q["Q2"].__setitem__("y", q["Q2"]["y"] + 1e-3)))["P1a_identidade"]["passou"] is False
    assert portoes_cidade("sint", cidade_sint(lambda q: q["Q3"].__setitem__("elev_std", q["Q3"]["elev_std"] * (1 + 1e-12))))["P1b_constantes"]["passou"] is False
    r = portoes_cidade("sint", cidade_sint(lambda q: [d.__setitem__("dem_shape", (200, 201)) for d in q.values()]))
    assert r["P1c0_cobertura"]["passou"] is False and r["P1c_momentos"]["aplicavel"] is False and not r["abortar"]
    r = portoes_cidade("sint", cidade_sint(lambda q: q["Q4"].__setitem__("pos", q["Q1"]["pos"].copy())))
    assert r["P1c0_cobertura"]["quadrantes_disjuntos"] is False
    def nodata(q):
        q["Q1"]["y"][:5000] = 0.0; q["Q1"]["x0"][:5000] = 0.0
    assert portoes_cidade("sint", cidade_sint(nodata))["P1d_nodata"]["passou"] is False
    def deslocar(q):
        for d in q.values():
            d["y"] = d["y"] + 0.01; d["x0"] = d["x0"] + 0.01
    r = portoes_cidade("sint", cidade_sint(deslocar))
    assert r["P1c_momentos"]["passou"] is False and r["abortar"]
    assert portoes_cidade("sint", cidade_sint(lambda q: q["Q1"].__setitem__("y", np.r_[q["Q1"]["y"][:-1], np.nan])))["P2_nan"]["sint_Q1"] == 1
    pre = {c: portoes_cidade(c, cidade_sint()) for c in CIDADES}
    (PARCIAL_DIR / PRE_JSON_NOME).write_text(json.dumps({"cidades": pre}), encoding="utf-8")
    log("fumaca: portoes de cidade conferidos (passa e falha onde deve)")
    parciais, laco_p = {}, {}
    for i, ch in enumerate(CELULAS):
        n = 100000
        pos_m = rng.uniform(0, 150000.0, size=(n, 2))
        gid = orig.SpatialKFold(n_splits=3, buffer_km=orig.BUFFER_KM, grid_size_km=orig.GRID_KM,
                                random_state=0)._assign_groups(pos_m / 1000.0)
        y = rng.standard_normal(n)
        y = (y - y.mean()) / y.std()
        ndvi = rng.uniform(-0.5, 0.9, n)
        valido = rng.uniform(size=n) < 0.1
        rec = {"celula": ch, "indice_da_celula": i, "sha256_manifest_tensor": "sintetico", "n_nodes_total": n,
               "N_blocos_ocupados": int(np.unique(gid).size), "portoes_P1a_P2": {"P1a_identidade": True, "P2": portoes_p1_p2(y)["P2"]},
               "P4_ndvi": {"fracao_zero_exato": float(np.mean(ndvi == 0.0)), "n_zero_exato": 0},
               "elev_std_m": 100.0 + 10 * i, "elev_mean_m": 500.0, "n_nos_validos": int(valido.sum())}
        rec.update(nucleo_celula(pos_m, gid, y, ndvi, valido, seeds, i))
        rec["tempo_cpu_s"], rec["tempo_relogio_s"], rec["status"] = 0.0, 0.0, "ok"
        rec["script_sha256"] = SCRIPT_SHA256
        (PARCIAL_DIR / f"{ch}.json").write_text(json.dumps(rec), encoding="utf-8")
        laco_p[ch] = {"por_sorteio": [{"split_seed": s["split_seed"], "n_test": s["n_test"]} for s in rec["por_sorteio"]]}
        (LACO_PARCIAL_DIR / f"{ch}.json").write_text(json.dumps(laco_p[ch]), encoding="utf-8")
        parciais[ch] = rec
    log("fumaca: 16 celulas sinteticas calculadas")
    assert p3_contra_laco(parciais[CELULAS[0]], laco_p[CELULAS[0]])["n_divergentes"] == 0
    alt = json.loads(json.dumps(laco_p[CELULAS[0]]))
    alt["por_sorteio"][3]["n_test"] += 1
    assert p3_contra_laco(parciais[CELULAS[0]], alt)["n_divergentes"] == 1
    res = montar_resumo(parciais, laco_p, "fumaca", {"nota": "sintetico"}, dest / "G2_resumo_fumaca.json", seeds, pre)
    print(json.dumps({"rodada_valida": res["rodada_valida"],
                      "mediana_deff_perm_sintetico": res["portoes"]["P5_controle_negativo"]["valor_medido"]["mediana_deff_alvo_permutado"],
                      "contagem": res["contagem_mecanica_limiares"]}, indent=1))
    print("FUMACA OK")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--executar", action="store_true")
    g.add_argument("--celula", default="")
    g.add_argument("--consolidar", action="store_true")
    g.add_argument("--fumaca", default="")
    a = ap.parse_args()
    comando = " ".join([sys.executable] + sys.argv)
    if a.executar:
        executar(comando)
    elif a.consolidar:
        consolidar(comando)
    elif a.fumaca:
        fumaca(Path(a.fumaca))
    else:
        assert a.celula in CELULAS
        rodar_uma(a.celula)


if __name__ == "__main__":
    main()
