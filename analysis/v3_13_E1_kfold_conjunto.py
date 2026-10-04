#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test E1: block K-fold cross-validation scored jointly, without training (CPU only).

For each of the 16 city x quadrant cells and each of the 60 split draws, the occupied 10 km blocks
are permuted exactly as in the three-way spatial split (RandomState(split_seed).shuffle of the sorted
block ids) and cut into K = 6 equal folds (22 blocks per fold in the study design). Each fold is held out in turn; the other five folds
are the training set (no trimming). With b = 0 every node of the held-out fold is scored; with
b = 2 km a node of the held-out fold is dropped when its nearest node outside the fold is closer than
b (same trimming rule as the hold-out; coordinates pos_m / 1000, in km).
Predictors, as in the error-drift test: constant = median RSSI of the whole training set (sentinels
included); calibrated FSPL (b) = offset median(RSSI + FSPL(dist)) over the valid training nodes.
Err_conjunto = (sum of |pred - rssi| over the scored nodes of the six folds) / (number scored), for
each population (valid nodes; all nodes) and each buffer. The results are compared with the
between-draw spread of the 92/20/20 block hold-out.

Fast trimming: since folds are unions of blocks, "a node outside fold a lies closer than b to node i"
is equivalent to "some block of another fold has a node closer than b to i". For each node, the list
of such blocks (at most 8) is computed once per cell with one cKDTree per block; per draw the trimming
is then a vectorised fold comparison. The validation mode checks it against a direct cKDTree query.

Reuses, by import and with digest check (aborts on mismatch), analysis/v3_13_laco_comum.py and,
through it, analysis/v3_2.1_3.1_deriva_calibracao.py. The decision criterion is
criteria/criterio_E1_kfold_conjunto.json, also checked by digest.

Usage:
  python analysis/v3_13_E1_kfold_conjunto.py --validar      # checks -> fase5/E1_validacao.json
  python analysis/v3_13_E1_kfold_conjunto.py --celula bauru_Q1   # 60 draws -> fase5/_parcial_E1/<cell>.json
  python analysis/v3_13_E1_kfold_conjunto.py --consolidar   # 16 partials -> E1_por_sorteio.json, E1_resumo.json
The validation mode checks (1) the b = 0 constant control (identical Err over draws), (2) exact
reproduction of the 92/20/20 hold-out of the error-drift test (bauru_Q1 and lins_Q1, 5 draws),
(3) fast trimming equal to cKDTree, (4) the exact-sum control. Deterministic; seeds are the 60
split seeds read by carregar_seeds().
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
SCRIPT_SHA256 = hashlib.sha256(HERE.read_bytes()).hexdigest()
BASE = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
V3 = BASE / "_v3_2026-09-25"
LACO_PATH = BASE / "scripts" / "v3_13_laco_comum.py"
LACO_SHA256 = "d6e7c2c6f5f0bb0af5f12c1fe3cbe47235e85347665f171b3fd7cd41fe813b93"
CRITERIO = V3 / "criterios" / "criterio_E1_kfold_conjunto.json"
CRITERIO_SHA256 = "c47a43d2b07d0e90c0138362bb25a4673984c0e8a3016d66b9882f7e2f81cbcc"
OUT = V3 / "fase5"
PARCIAL_DIR = OUT / "_parcial_E1"
OUT_VALID = OUT / "E1_validacao.json"
OUT_SORTEIO = OUT / "E1_por_sorteio.json"
OUT_RESUMO = OUT / "E1_resumo.json"
R1_B0_SORTEIO = V3 / "fase4" / "R1_b0_por_sorteio.json"
R1_B0_RESUMO = V3 / "fase4" / "R1_b0_resumo.json"
R2_RESUMO = V3 / "fase4" / "R2_resumo.json"

K_FOLDS = 6
BUFFERS_KM = (0.0, 2.0)
SEED_REF_MU = 42  # split seed of the single reference calibration for mu_U (same as the design-reference script)
PREDS = ("constante", "fspl_b")
POPS = ("validos", "todos")
CELULAS = [f"{c}_{q}" for c in ("bauru", "campinas", "lins", "sorocaba") for q in ("Q1", "Q2", "Q3", "Q4")]

_laco_sha_real = hashlib.sha256(LACO_PATH.read_bytes()).hexdigest()
if _laco_sha_real != LACO_SHA256:
    raise SystemExit(f"ABORTA: sha do laco_comum diverge ({_laco_sha_real})")
_crit_sha_real = hashlib.sha256(CRITERIO.read_bytes()).hexdigest()
if _crit_sha_real != CRITERIO_SHA256:
    raise SystemExit(f"ABORTA: sha do criterio E1 diverge ({_crit_sha_real})")

_spec = importlib.util.spec_from_file_location("v3_13_laco_comum", LACO_PATH)
laco = importlib.util.module_from_spec(_spec)
sys.modules["v3_13_laco_comum"] = laco
_spec.loader.exec_module(laco)  # its main() is guarded by __name__; it checks the digest of the module it imports
orig = laco.orig
ORIG_SHA256 = laco.ORIG_SHA256
log = laco.log
agora = laco.agora


def permutacao_blocos(grupos: np.ndarray, seed: int) -> np.ndarray:
    """Block permutation identical to the three-way split: RandomState(seed).shuffle of the sorted ids."""
    rng = np.random.RandomState(seed)
    emb = grupos.copy()
    rng.shuffle(emb)
    return emb


def folds_de_blocos(grupos: np.ndarray, seed: int, K: int = K_FOLDS) -> np.ndarray:
    """Fold index (int8) of each block, aligned with the sorted `grupos`; K equal consecutive cuts."""
    n_g = grupos.size
    assert n_g % K == 0, f"n_g={n_g} nao divisivel por K={K}"
    m = n_g // K
    emb = permutacao_blocos(grupos, seed)
    fold_do_bloco = np.empty(n_g, dtype=np.int8)
    pos = np.searchsorted(grupos, emb)  # index in `grupos` of each permuted block
    for a in range(K):
        fold_do_bloco[pos[a * m:(a + 1) * m]] = a
    return fold_do_bloco


def preparar_celula_e1(chave: str) -> dict:
    """Load a cell and add sorted block ids, node-to-block index, km coordinates and FSPL(dist)."""
    c = laco.carregar_celula(chave)
    grupos = np.unique(c["gid"])
    c["grupos"] = grupos
    c["node_bi"] = np.searchsorted(grupos, c["gid"]).astype(np.int16)
    c["pos_km"] = c["pos_m"] / 1000.0
    fspl = orig.MODELOS["fspl"]
    with np.errstate(all="ignore"):
        c["plf"] = fspl(c["dist"])  # elementwise, so subsetting later gives the same values
    return c


def precomputar_vizinhanca(pos_km: np.ndarray, node_bi: np.ndarray, nb: int, b_km: float) -> dict:
    """For each node, the other blocks that hold a node at distance d < b_km (Euclidean, km, as
    returned by cKDTree.query, the same distance as in the hold-out trimming). Block pairs whose
    bounding boxes are at least b_km apart are skipped; at most 8 neighbouring blocks per node."""
    from scipy.spatial import cKDTree
    kw = dict(compact_nodes=False, balanced_tree=False)
    order = np.argsort(node_bi, kind="stable")
    cnt_b = np.bincount(node_bi, minlength=nb)
    starts = np.concatenate([[0], np.cumsum(cnt_b)])
    idx_b = [order[starts[i]:starts[i + 1]] for i in range(nb)]
    pts_b = [pos_km[ix] for ix in idx_b]
    lo = np.array([p.min(axis=0) for p in pts_b])
    hi = np.array([p.max(axis=0) for p in pts_b])
    trees = {}
    cap = 8
    cand_idx, cand_lst = [], []
    n_pares = 0
    for i in range(nb):
        pts = pts_b[i]
        lst = np.full((pts.shape[0], cap), -1, dtype=np.int16)
        cnt = np.zeros(pts.shape[0], dtype=np.int16)
        for j in range(nb):
            if j == i:
                continue
            gap = np.maximum(0.0, np.maximum(lo[j] - hi[i], lo[i] - hi[j]))
            if float(np.hypot(gap[0], gap[1])) >= b_km:
                continue
            n_pares += 1
            if j not in trees:
                trees[j] = cKDTree(pts_b[j], **kw)
            sel = np.flatnonzero(np.all((pts >= lo[j] - b_km) & (pts <= hi[j] + b_km), axis=1))
            if sel.size == 0:
                continue
            d, _ = trees[j].query(pts[sel], k=1, distance_upper_bound=b_km, workers=1)
            hit = sel[d < b_km]
            if hit.size:
                assert cnt[hit].max() < cap, "mais de 8 blocos vizinhos num no"
                lst[hit, cnt[hit]] = j
                cnt[hit] += 1
        has = cnt > 0
        if has.any():
            cand_idx.append(idx_b[i][has])
            cand_lst.append(lst[has])
    cand_idx = np.concatenate(cand_idx) if cand_idx else np.empty(0, dtype=np.int64)
    cand_lst = np.concatenate(cand_lst) if cand_lst else np.empty((0, cap), dtype=np.int16)
    mmax = int((cand_lst >= 0).sum(axis=1).max()) if cand_lst.shape[0] else 0
    return {"cand_idx": cand_idx, "cand_lst": np.ascontiguousarray(cand_lst[:, :max(mmax, 1)]),
            "n_cand": int(cand_idx.size), "max_blocos_vizinhos_por_no": mmax, "n_pares_blocos": n_pares}


def keep_mask_rapido(viz: dict, fold_do_bloco: np.ndarray, node_fold: np.ndarray, n: int) -> np.ndarray:
    """True for the nodes that remain scored with b = 2 km, for all folds at once."""
    ci, cl = viz["cand_idx"], viz["cand_lst"]
    nf = node_fold[ci]
    removed = np.zeros(ci.size, dtype=bool)
    for m in range(cl.shape[1]):
        L = cl[:, m]
        has = L >= 0
        removed |= has & (fold_do_bloco[np.where(has, L, 0)] != nf)
    keep = np.ones(n, dtype=bool)
    keep[ci[removed]] = False
    return keep


def keep_fold_kdtree(pos_km: np.ndarray, node_fold: np.ndarray, a: int, b_km: float, workers: int = 4) -> np.ndarray:
    """Slow reference: trimming of fold a by a cKDTree over the nodes outside it (hold-out rule)."""
    from scipy.spatial import cKDTree
    m_in = node_fold == a
    tree = cKDTree(pos_km[~m_in], compact_nodes=False, balanced_tree=False)
    d, _ = tree.query(pos_km[m_in], k=1, workers=workers)
    keep = np.zeros(pos_km.shape[0], dtype=bool)
    idx = np.flatnonzero(m_in)
    keep[idx[~(d < b_km)]] = True
    return keep


def sorteio_kfold(c: dict, viz: dict, seed: int, controle_fsum: bool = True) -> dict:
    """One split draw: per-fold calibration and losses, pooled error Err_conjunto per predictor,
    population and buffer, scored-node counts, and (optionally) the exact-sum control."""
    rssi, sent, plf = c["rssi"], c["sent"], c["plf"]
    n = rssi.shape[0]
    fb = folds_de_blocos(c["grupos"], seed)
    node_fold = fb[c["node_bi"]]
    keep2 = keep_mask_rapido(viz, fb, node_fold, n)
    N_val_total = int((~sent).sum())

    S = {(p, pop, b): [] for p in PREDS for pop in POPS for b in BUFFERS_KM}
    M = {(pop, b): [] for pop in POPS for b in BUFFERS_KM}
    consts, offsets = [], []
    fsum_ctrl = {pop: [] for pop in POPS}
    for a in range(K_FOLDS):
        in_a = node_fold == a
        tr = ~in_a
        constante = float(np.median(rssi[tr]))
        trv = tr & ~sent
        p_tx = float(np.median(rssi[trv] + plf[trv])) if trv.any() else None
        consts.append(constante)
        offsets.append(p_tx)
        idx = np.flatnonzero(in_a)
        r = rssi[idx]
        v = ~sent[idx]
        k2 = keep2[idx]
        loss = {"constante": np.abs(constante - r)}
        if p_tx is not None:
            loss["fspl_b"] = np.abs((p_tx - plf[idx]) - r)
        masks = {("validos", 0.0): v, ("todos", 0.0): np.ones_like(v), ("validos", 2.0): v & k2, ("todos", 2.0): k2}
        for (pop, b), mk in masks.items():
            M[(pop, b)].append(int(mk.sum()))
            for p, lv in loss.items():
                S[(p, pop, b)].append(float(lv[mk].sum()))
        if controle_fsum:
            for pop in POPS:
                sel = v if pop == "validos" else np.ones_like(v)
                fsum_ctrl[pop].append(math.fsum(loss["constante"][sel]))
    res = {"split_seed": seed, "status": "ok", "constante_por_fold": consts, "offset_fspl_b_por_fold": offsets,
           "n_pontuados_por_fold": {f"{pop}_b{int(b)}": M[(pop, b)] for pop in POPS for b in BUFFERS_KM},
           "n_fold_nos": [int((node_fold == a).sum()) for a in range(K_FOLDS)]}
    err, soma = {}, {}
    for p in PREDS:
        for pop in POPS:
            for b in BUFFERS_KM:
                tot_n = sum(M[(pop, b)])
                s = math.fsum(S[(p, pop, b)])
                err[f"{p}|{pop}|b{int(b)}"] = (s / tot_n) if tot_n > 0 else None
                soma[f"{p}|{pop}|b{int(b)}"] = {"S_por_fold": S[(p, pop, b)], "soma": s, "n": tot_n}
    res["Err_conjunto"] = err
    res["soma_por_fold"] = soma
    nv0, nv2 = sum(M[("validos", 0.0)]), sum(M[("validos", 2.0)])
    nt0, nt2 = sum(M[("todos", 0.0)]), sum(M[("todos", 2.0)])
    res["pontuados"] = {"b0": {"todos": nt0, "validos": nv0, "fracao_todos": nt0 / n,
                               "fracao_validos": nv0 / N_val_total},
                        "b2": {"todos": nt2, "validos": nv2, "fracao_todos": nt2 / n,
                               "fracao_validos": nv2 / N_val_total}}
    if controle_fsum:
        # exact-sum control: constant predictor at b = 0, per-fold np.sum totals versus math.fsum
        ctrl = {}
        for pop in POPS:
            s_np = soma[f"constante|{pop}|b0"]["soma"]
            s_fs = math.fsum(fsum_ctrl[pop])
            ctrl[pop] = {"soma_np_por_fold_fsum_total": s_np, "soma_fsum": s_fs, "absdiff": abs(s_np - s_fs),
                         "Err_fsum": s_fs / sum(M[(pop, 0.0)])}
        res["controle_b0_constante_fsum"] = ctrl
    return res


def mu_U_ref(c: dict) -> dict:
    """Domain mean mu_U of the node error e_i, with the recipe of the design-reference script: one
    calibration on the split of seed 42 (training = 92 blocks, no trimming), e_i over the whole domain."""
    rssi, sent, plf = c["rssi"], c["sent"], c["plf"]
    tr = holdout_mascaras_blocos(c, SEED_REF_MU)["m_tr"]
    constante = float(np.median(rssi[tr]))
    trv = tr & ~sent
    p_tx = float(np.median(rssi[trv] + plf[trv]))
    e = {"constante": np.abs(rssi - constante), "fspl_b": np.abs(rssi - (p_tx - plf))}
    out = {"seed_ref": SEED_REF_MU, "constante_ref": constante, "p_tx_eff_ref": p_tx}
    for p in PREDS:
        out[p] = {"validos": float(e[p][~sent].mean()), "todos": float(e[p].mean())}
    return out


def holdout_mascaras_blocos(c: dict, seed: int) -> dict:
    """Block roles of the 92/20/20 split (as in the three-way split), without trimming."""
    grupos = c["grupos"]
    n_g = grupos.size
    emb = permutacao_blocos(grupos, seed)
    n_tr = max(1, int(round(orig.FRACS[0] * n_g)))
    n_va = max(1, int(round(orig.FRACS[1] * n_g)))
    if n_tr + n_va >= n_g:
        n_tr = max(1, n_g - 2)
        n_va = 1
    g_tr, g_va, g_te = emb[:n_tr], emb[n_tr:n_tr + n_va], emb[n_tr + n_va:]
    gid = c["gid"]
    return {"m_tr": np.isin(gid, g_tr), "m_va": np.isin(gid, g_va), "m_te": np.isin(gid, g_te),
            "k": (int(g_tr.size), int(g_va.size), int(g_te.size))}


def holdout_proprio(c: dict, seed: int, b_km: float = 2.0) -> dict:
    """92/20/20 hold-out rebuilt with this script's own loop (blocks, permutation, cKDTree trimming in
    the original order: validation against training, then test against training plus surviving
    validation) and the MAE of the error-drift test; used to check exact reproduction."""
    from scipy.spatial import cKDTree
    kw = dict(compact_nodes=False, balanced_tree=False)
    mk = holdout_mascaras_blocos(c, seed)
    m_tr, m_va, m_te = mk["m_tr"].copy(), mk["m_va"].copy(), mk["m_te"].copy()
    pk = c["pos_km"]
    d, _ = cKDTree(pk[m_tr], **kw).query(pk[m_va], k=1, workers=4)
    iv = np.flatnonzero(m_va)
    m_va[iv[d < b_km]] = False
    d, _ = cKDTree(pk[m_tr | m_va], **kw).query(pk[m_te], k=1, workers=4)
    it = np.flatnonzero(m_te)
    m_te[it[d < b_km]] = False
    rssi, sent, plf = c["rssi"], c["sent"], c["plf"]
    tr_i, te_i = np.flatnonzero(m_tr), np.flatnonzero(m_te)
    constante = float(np.median(rssi[tr_i]))
    trv = tr_i[~sent[tr_i]]
    p_tx = float(np.median(rssi[trv] + plf[trv]))
    rte, ste = rssi[te_i], sent[te_i]
    pred_f = p_tx - plf[te_i]
    out = {"split_seed": seed, "tr": tr_i, "te": te_i, "k": mk["k"], "constante": constante,
           "n_test": int(te_i.size), "n_validos": int((~ste).sum()), "n_todos": int(te_i.size)}
    out["mae_constante"] = {"validos": orig.mae(constante, rte[~ste]), "todos": orig.mae(constante, rte)}
    out["mae_fspl_b"] = {"validos": orig.mae(pred_f[~ste], rte[~ste]), "todos": orig.mae(pred_f, rte)}
    return out


def rodar_celula(chave: str, seeds: list) -> dict:
    """All draws of one cell, plus the b = 0 constant control (Err must not vary across draws)."""
    t0 = time.perf_counter()
    c = preparar_celula_e1(chave)
    nb = int(c["grupos"].size)
    rec = {"celula": chave, "sha256_manifest": c["sha256_manifest"], "modo_carga": c["modo_carga"],
           "n_nodes_total": c["n_total"], "n_validos_total": int((~c["sent"]).sum()),
           "N_blocos_ocupados": nb, "K": K_FOLDS, "blocos_por_fold": nb // K_FOLDS,
           "tempo_load_s": time.perf_counter() - t0}
    t1 = time.perf_counter()
    viz = precomputar_vizinhanca(c["pos_km"], c["node_bi"], nb, 2.0)
    rec["vizinhanca"] = {k: viz[k] for k in ("n_cand", "max_blocos_vizinhos_por_no", "n_pares_blocos")}
    rec["tempo_vizinhanca_s"] = time.perf_counter() - t1
    log(f"{chave}: carga {rec['tempo_load_s']:.1f}s, vizinhanca {rec['tempo_vizinhanca_s']:.1f}s, n_cand={viz['n_cand']:,}")
    rec["mu_U_ref_seed42"] = mu_U_ref(c)
    rec["por_sorteio"] = []
    for i, s in enumerate(seeds):
        rec["por_sorteio"].append(sorteio_kfold(c, viz, s))
        if (i + 1) % 10 == 0:
            log(f"{chave}: {i + 1}/{len(seeds)} sorteios")
    ctrl = {}
    cs = [x for s in rec["por_sorteio"] for x in s["constante_por_fold"]]
    ctrl["constante_por_fold_min"], ctrl["constante_por_fold_max"] = float(min(cs)), float(max(cs))
    ctrl["constante_unica"] = bool(min(cs) == max(cs))
    ctrl["constantes_distintas"] = sorted(set(float(x) for x in cs))
    for pop in POPS:
        vals = [s["Err_conjunto"][f"constante|{pop}|b0"] for s in rec["por_sorteio"]]
        ctrl[pop] = {"n_valores_distintos": len(set(vals)), "max_menos_min": float(max(vals) - min(vals)),
                     "valor_primeiro": vals[0], "todos_exatamente_iguais": bool(len(set(vals)) == 1)}
        fs = [s["controle_b0_constante_fsum"][pop]["Err_fsum"] for s in rec["por_sorteio"]]
        ctrl[pop]["fsum_n_valores_distintos"] = len(set(fs))
        ctrl[pop]["fsum_max_menos_min"] = float(max(fs) - min(fs))
        ctrl[pop]["max_absdiff_soma_np_vs_fsum"] = float(max(s["controle_b0_constante_fsum"][pop]["absdiff"]
                                                             for s in rec["por_sorteio"]))
    rec["controle_b0_constante"] = ctrl
    rec["status"] = "ok"
    rec["tempo_total_s"] = time.perf_counter() - t0
    return rec


def rodar_uma(chave: str):
    """Run one cell and write its partial record; skipped if a partial with status ok exists."""
    PARCIAL_DIR.mkdir(parents=True, exist_ok=True)
    arq = PARCIAL_DIR / f"{chave}.json"
    if arq.exists() and json.loads(arq.read_text(encoding="utf-8")).get("status") == "ok":
        log(f"{chave}: parcial ok, pulando")
        return
    seeds = laco.carregar_seeds()
    log(f"=== {chave}: {len(seeds)} sorteios ===")
    try:
        rec = rodar_celula(chave, seeds)
    except Exception as e:  # noqa: BLE001
        rec = {"celula": chave, "status": "erro_excecao", "erro": repr(e), "traceback": traceback.format_exc()}
    rec["script_sha256"] = SCRIPT_SHA256
    rec["laco_sha256"] = LACO_SHA256
    rec["data_utc"] = agora()
    rec["comando"] = " ".join(sys.argv)
    arq.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"{chave}: status={rec['status']} -> {arq}")


def validar():
    """Validation mode on bauru_Q1 and lins_Q1; exits with status 2 if any check fails."""
    seeds = laco.carregar_seeds()
    seeds5 = seeds[:5]
    p = json.loads(laco.PARCIAL_B2.read_text(encoding="utf-8"))
    saida = {"id": "E1_validacao", "sementes_5": seeds5, "referencia": str(laco.PARCIAL_B2),
             "referencia_sha256": orig.sha256_file(laco.PARCIAL_B2), "celulas": {}}
    ok_global = True
    for chave in ("bauru_Q1", "lins_Q1"):
        log(f"VALIDACAO {chave}")
        c = preparar_celula_e1(chave)
        nb = int(c["grupos"].size)
        rec = {}
        # (0) fold structure: six folds of 22 blocks
        fb = folds_de_blocos(c["grupos"], seeds5[0])
        rec["folds_22_blocos_cada"] = bool(np.bincount(fb, minlength=K_FOLDS).tolist() == [nb // K_FOLDS] * K_FOLDS)
        # (2) exact reproduction of the 92/20/20 hold-out records
        ref = p["celulas"][chave]["por_sorteio"][:5]
        linhas, divergentes, n_cmp = [], [], 0
        for seed, r in zip(seeds5, ref):
            h = holdout_proprio(c, seed)
            assert h["split_seed"] == r["split_seed"]
            lin = {"split_seed": seed, "k_blocos_tr_va_te": h["k"], "n_test_proprio": h["n_test"], "n_test_2.1": r["n_test"],
                   "n_train_proprio": int(h["tr"].size), "n_train_2.1": r["n_train"]}
            ok_n = (h["n_test"] == r["n_test"] and int(h["tr"].size) == r["n_train"]
                    and h["n_validos"] == r["n_pop_teste"]["validos"] and h["n_todos"] == r["n_pop_teste"]["todos"])
            gravado = {("constante", "validos"): r["mae_constante_teste"]["validos"],
                       ("constante", "todos"): r["mae_constante_teste"]["todos"],
                       ("fspl_b", "validos"): r["mae_modelo_b_validos_teste"]["fspl"]["validos"],
                       ("fspl_b", "todos"): r["mae_modelo_b_validos_teste"]["fspl"]["todos"]}
            for (pred, pop), g in gravado.items():
                mine = h[f"mae_{pred}"][pop]
                n_cmp += 1
                lin[f"{pred}_{pop}"] = {"proprio": mine, "gravado_2.1": g, "igual_exato": bool(mine == g)}
                if mine != g:
                    divergentes.append((seed, pred, pop, mine, g))
            lin["contagens_iguais"] = bool(ok_n)
            if not ok_n:
                divergentes.append((seed, "contagens", None, None, None))
            # indices identical to orig.split_espacial_3vias (first two draws only, for cost)
            if seed in seeds5[:2]:
                parts = orig.split_espacial_3vias(c["pos_m"], orig.GRID_KM, orig.BUFFER_KM, orig.FRACS, seed)
                lin["indices_treino_e_teste_identicos_a_split_espacial_3vias"] = bool(
                    np.array_equal(parts["train"], h["tr"]) and np.array_equal(parts["test"], h["te"]))
                if not lin["indices_treino_e_teste_identicos_a_split_espacial_3vias"]:
                    divergentes.append((seed, "indices", None, None, None))
            linhas.append(lin)
            log(f"  {chave} hold-out seed {seed}: ok_n={ok_n}")
        rec["holdout_92_20_20_reproduz_2.1"] = {"reproduz": bool(not divergentes), "n_numeros_comparados_exato": n_cmp,
                                                 "divergentes": divergentes[:20], "linhas": linhas}
        # (3) fast trimming equal to cKDTree: 2 draws x 6 folds
        viz = precomputar_vizinhanca(c["pos_km"], c["node_bi"], nb, 2.0)
        eq = []
        for seed in seeds5[:2]:
            fb = folds_de_blocos(c["grupos"], seed)
            nf = fb[c["node_bi"]]
            keep2 = keep_mask_rapido(viz, fb, nf, c["rssi"].shape[0])
            for a in range(K_FOLDS):
                kref = keep_fold_kdtree(c["pos_km"], nf, a, 2.0)
                kfast = keep2 & (nf == a)
                n_dif = int((kref != kfast).sum())
                eq.append({"seed": seed, "fold": a, "n_pontuados_kdtree": int(kref.sum()),
                           "n_pontuados_rapido": int(kfast.sum()), "n_nos_diferentes": n_dif})
                log(f"  {chave} aparo seed {seed} fold {a}: kdtree={int(kref.sum())} rapido={int(kfast.sum())} dif={n_dif}")
        rec["aparo_rapido_igual_kdtree"] = {"igual": bool(all(e["n_nos_diferentes"] == 0 for e in eq)), "folds": eq}
        # (1) b = 0 constant control over 5 draws of the K-fold loop
        res5 = [sorteio_kfold(c, viz, s) for s in seeds5]
        ctrl = {"constante_por_fold_distintas": sorted(set(float(x) for s in res5 for x in s["constante_por_fold"]))}
        for pop in POPS:
            vals = [s["Err_conjunto"][f"constante|{pop}|b0"] for s in res5]
            fs = [s["controle_b0_constante_fsum"][pop]["Err_fsum"] for s in res5]
            ctrl[pop] = {"valores": vals, "todos_exatamente_iguais": bool(len(set(vals)) == 1),
                         "fsum_valores_iguais": bool(len(set(fs)) == 1), "igual_ao_fsum": bool(vals == fs)}
        ctrl["fracao_pontuada_b2_por_sorteio"] = [s["pontuados"]["b2"]["fracao_todos"] for s in res5]
        rec["controle_b0_constante_5_sorteios"] = ctrl
        reproduz = (rec["folds_22_blocos_cada"] and rec["holdout_92_20_20_reproduz_2.1"]["reproduz"]
                    and rec["aparo_rapido_igual_kdtree"]["igual"]
                    and all(ctrl[pop]["todos_exatamente_iguais"] for pop in POPS))
        rec["passou"] = bool(reproduz)
        ok_global &= bool(reproduz)
        saida["celulas"][chave] = rec
        del c
    saida["passou"] = bool(ok_global)
    saida.update({"script_sha256": SCRIPT_SHA256, "laco_sha256": LACO_SHA256, "script_original_sha256": ORIG_SHA256,
                  "criterio_sha256": CRITERIO_SHA256, "comando": " ".join(sys.argv), "data_utc": agora(),
                  "venv": sys.executable})
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_VALID.write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"validacao passou={ok_global} -> {OUT_VALID}")
    sys.exit(0 if ok_global else 2)


def dp1(v):
    """Standard deviation with ddof = 1 (np.std), None values dropped; returns exactly 0.0 when all
    values are identical, because np.std of identical values can return ~1e-14 from rounding of the mean."""
    v = [x for x in v if x is not None]
    if len(v) <= 1:
        return None
    if min(v) == max(v):
        return 0.0
    return float(np.std(v, ddof=1))


def serie_holdout(p21: dict, r1: dict, cel: str, pred: str, pop: str, b: float) -> list:
    """Per-draw hold-out MAE series (b = 2: error-drift partial record; b = 0: b = 0 drift record), None dropped."""
    if b == 2.0:
        out = []
        for s in p21["celulas"][cel]["por_sorteio"]:
            if s.get("status") != "ok":
                continue
            out.append(s["mae_constante_teste"][pop] if pred == "constante"
                       else s["mae_modelo_b_validos_teste"]["fspl"][pop])
        return [x for x in out if x is not None]
    out = []
    for s in r1["celulas"][cel]["sorteios"]:
        if s.get("status") != "ok":
            continue
        out.append(s["mae_constante"][pop] if pred == "constante" else s["mae_fspl_b_validos"][pop])
    return [x for x in out if x is not None]


def consolidar():
    """Merge the 16 partial records and write the per-draw and summary records."""
    parc = {}
    for cel in CELULAS:
        f = PARCIAL_DIR / f"{cel}.json"
        assert f.exists(), f"parcial ausente: {f}"
        r = json.loads(f.read_text(encoding="utf-8"))
        assert r["status"] == "ok", f"{cel}: {r['status']}"
        assert r["script_sha256"] == SCRIPT_SHA256, f"{cel}: parcial gerado por outro sha de script"
        parc[cel] = r
    seeds = laco.carregar_seeds()
    for cel, r in parc.items():
        assert [s["split_seed"] for s in r["por_sorteio"]] == seeds, f"{cel}: sementes divergem"
    f2 = json.loads(laco.F2_JSON.read_text(encoding="utf-8"))
    T_dp_sorteio = f2["resumo"]["dp_medio_entre_sorteios_validos_dB_simples"]
    T_dp_entre = f2["resumo"]["dp_entre_celulas_das_medias_validos_dB"]
    p21 = json.loads(laco.PARCIAL_B2.read_text(encoding="utf-8"))
    r1 = json.loads(R1_B0_SORTEIO.read_text(encoding="utf-8"))
    r1res = json.loads(R1_B0_RESUMO.read_text(encoding="utf-8"))
    r2 = json.loads(R2_RESUMO.read_text(encoding="utf-8"))
    valid = json.loads(OUT_VALID.read_text(encoding="utf-8")) if OUT_VALID.exists() else None

    # spread over draws per cell, predictor, population and buffer, K-fold versus hold-out
    tab = {}
    ho = {}
    for pred in PREDS:
        for pop in POPS:
            for b in BUFFERS_KM:
                chave = f"{pred}|{pop}|b{int(b)}"
                medias, dps, por_cel, ho_dp, ho_med = {}, {}, {}, {}, {}
                for cel in CELULAS:
                    v = [s["Err_conjunto"][chave] for s in parc[cel]["por_sorteio"]]
                    medias[cel] = float(np.mean(v))
                    dps[cel] = dp1(v)
                    hs = serie_holdout(p21, r1, cel, pred, pop, b)
                    ho_dp[cel] = dp1(hs)
                    ho_med[cel] = float(np.mean(hs))
                    por_cel[cel] = {"n_sorteios": len(v), "media": medias[cel], "dp_ddof1": dps[cel],
                                    "min": float(min(v)), "max": float(max(v)),
                                    "holdout_dp_ddof1": ho_dp[cel], "holdout_media": ho_med[cel],
                                    "holdout_n_sorteios": len(hs),
                                    "razao_dp_kfold_sobre_dp_holdout": (dps[cel] / ho_dp[cel]) if ho_dp[cel] else None}
                dp_entre = float(np.std(list(medias.values()), ddof=1))
                dp_medio = float(np.mean(list(dps.values())))
                ho_dp_medio = float(np.mean(list(ho_dp.values())))
                ho_dp_entre = float(np.std(list(ho_med.values()), ddof=1))
                for cel in CELULAS:
                    por_cel[cel]["razao_dp_sobre_dp_entre_celulas_kfold"] = (dps[cel] / dp_entre) if dp_entre else None
                    por_cel[cel]["razao_dp_sobre_dp_entre_celulas_holdout_2.1"] = (
                        dps[cel] / T_dp_entre if (b == 2.0 and pred == "constante" and pop == "validos") else
                        (dps[cel] / ho_dp_entre))
                tab[chave] = {"por_celula": por_cel, "dp_sorteios_medio_16": dp_medio,
                              "dp_entre_celulas_das_medias_16": dp_entre,
                              "razao_dp_sorteios_medio_sobre_dp_entre_celulas": (dp_medio / dp_entre) if dp_entre else None,
                              "holdout": {"dp_sorteios_medio_16": ho_dp_medio, "dp_entre_celulas_das_medias_16": ho_dp_entre,
                                          "razao": ho_dp_medio / ho_dp_entre if ho_dp_entre else None},
                              "razao_dp_medio_kfold_sobre_dp_medio_holdout": dp_medio / ho_dp_medio if ho_dp_medio else None}
                ho[chave] = {"dp_medio": ho_dp_medio, "dp_entre": ho_dp_entre}

    # consistency of the recomputed hold-out references with the stored records
    conf = {
        "T.dp_sorteio_do_artefato_fase2": T_dp_sorteio,
        "T.dp_entre_medias_do_artefato_fase2": T_dp_entre,
        "holdout_b2_constante_validos_recomputado_do_parcial": ho["constante|validos|b2"]["dp_medio"],
        "absdiff_vs_T.dp_sorteio": abs(ho["constante|validos|b2"]["dp_medio"] - T_dp_sorteio),
        "holdout_b2_constante_validos_dp_entre_recomputado": ho["constante|validos|b2"]["dp_entre"],
        "absdiff_vs_T.dp_entre_medias": abs(ho["constante|validos|b2"]["dp_entre"] - T_dp_entre),
        "U_b0_dp_medio_do_artefato_R1": r1res["medias_16_celulas"]["dp_mae_constante_validos_b0"],
        "holdout_b0_constante_validos_recomputado": ho["constante|validos|b0"]["dp_medio"],
        "absdiff_vs_U_b0": abs(ho["constante|validos|b0"]["dp_medio"] - r1res["medias_16_celulas"]["dp_mae_constante_validos_b0"]),
    }

    # bias of the mean pooled error relative to the domain mean mu_U
    vies = {}
    for pred in PREDS:
        for pop in POPS:
            for b in BUFFERS_KM:
                chave = f"{pred}|{pop}|b{int(b)}"
                d = {}
                for cel in CELULAS:
                    mu = parc[cel]["mu_U_ref_seed42"][pred][pop]
                    vals = [s["Err_conjunto"][chave] for s in parc[cel]["por_sorteio"]]
                    e = {"mu_U_ref_seed42": mu, "media_Err_conjunto": float(np.mean(vals)),
                         "vies_vs_mu_U_ref_seed42": float(np.mean(vals)) - mu}
                    if b == 2.0:
                        v0 = [s["Err_conjunto"][f"{pred}|{pop}|b0"] for s in parc[cel]["por_sorteio"]]
                        dif = [x - y for x, y in zip(vals, v0)]
                        e["vies_b2_menos_b0_mesmo_sorteio_media"] = float(np.mean(dif))
                        e["vies_b2_menos_b0_mesmo_sorteio_dp"] = dp1(dif)
                    d[cel] = e
                vies[chave] = {"por_celula": d,
                               "media_abs_vies_vs_mu_U_16": float(np.mean([abs(x["vies_vs_mu_U_ref_seed42"]) for x in d.values()])),
                               "media_vies_vs_mu_U_16": float(np.mean([x["vies_vs_mu_U_ref_seed42"] for x in d.values()]))}
    # mu_U checked against the design-reference record (it holds only the four Q1 cells)
    conf_mu = {}
    for cel, r in r2["resumo_por_celula"].items():
        conf_mu[cel] = {f"{p}|{pop}": {"proprio": parc[cel]["mu_U_ref_seed42"][p][pop], "R2": r[p2][pop]["mu_U"],
                                       "absdiff": abs(parc[cel]["mu_U_ref_seed42"][p][pop] - r[p2][pop]["mu_U"])}
                        for p, p2 in (("constante", "constante"), ("fspl_b", "fspl_calibrado_b")) for pop in POPS}

    # scored fraction per cell and buffer
    pont = {}
    for cel in CELULAS:
        pont[cel] = {}
        for bb in ("b0", "b2"):
            ft = [s["pontuados"][bb]["fracao_todos"] for s in parc[cel]["por_sorteio"]]
            fv = [s["pontuados"][bb]["fracao_validos"] for s in parc[cel]["por_sorteio"]]
            nv = [s["pontuados"][bb]["validos"] for s in parc[cel]["por_sorteio"]]
            pont[cel][bb] = {"fracao_todos_media": float(np.mean(ft)), "fracao_todos_dp": dp1(ft),
                             "fracao_todos_min": float(min(ft)), "fracao_todos_max": float(max(ft)),
                             "fracao_validos_media": float(np.mean(fv)), "fracao_validos_dp": dp1(fv),
                             "n_validos_pontuados_media": float(np.mean(nv)), "n_validos_pontuados_min": int(min(nv)),
                             "n_validos_pontuados_max": int(max(nv)), "n_validos_total": parc[cel]["n_validos_total"]}
    ctrl_b0 = {cel: parc[cel]["controle_b0_constante"] for cel in CELULAS}

    # mechanical counts for the thresholds of the decision criterion, under both readings of each denominator
    L = {}
    kb0 = tab["constante|validos|b0"]
    L["b0_constante_validos"] = {
        "dp_por_celula": {cel: kb0["por_celula"][cel]["dp_ddof1"] for cel in CELULAS},
        "n_celulas_dp_exatamente_zero": int(sum(1 for cel in CELULAS if kb0["por_celula"][cel]["dp_ddof1"] == 0.0)),
        "max_dp": float(max(kb0["por_celula"][cel]["dp_ddof1"] for cel in CELULAS)),
        "todas_constantes_unicas_(-110)": bool(all(ctrl_b0[cel]["constante_unica"] for cel in CELULAS)),
    }
    for pred, pop in (("constante", "validos"), ("constante", "todos"), ("fspl_b", "validos"), ("fspl_b", "todos")):
        kb = tab[f"{pred}|{pop}|b2"]
        dp_c = {cel: kb["por_celula"][cel]["dp_ddof1"] for cel in CELULAS}
        dp_ho_c = {cel: kb["por_celula"][cel]["holdout_dp_ddof1"] for cel in CELULAS}
        den_k = kb["dp_entre_celulas_das_medias_16"]
        den_h = kb["holdout"]["dp_entre_celulas_das_medias_16"]
        T_ref = kb["holdout"]["dp_sorteios_medio_16"]
        n_lt1_k = int(sum(1 for cel in CELULAS if dp_c[cel] / den_k < 1))
        n_lt1_h = int(sum(1 for cel in CELULAS if dp_c[cel] / den_h < 1))
        dp_m = kb["dp_sorteios_medio_16"]
        n_ge_half_cel = int(sum(1 for cel in CELULAS if dp_c[cel] >= 0.5 * dp_ho_c[cel]))
        n_ge_half_T = int(sum(1 for cel in CELULAS if dp_c[cel] >= 0.5 * T_ref))
        L[f"b2_{pred}_{pop}"] = {
            "regra_do_criterio_aplica_se": bool(pred == "constante" and pop == "validos"),
            "dp_sorteios_medio_16_kfold": dp_m,
            "dp_medio_holdout_16": T_ref,
            "dp_entre_celulas_kfold": den_k, "dp_entre_celulas_holdout": den_h,
            "REFORCA_(i)_n_celulas_razao_lt_1_denominador_dp_entre_celulas_do_proprio_kfold": n_lt1_k,
            "REFORCA_(i)_n_celulas_razao_lt_1_denominador_dp_entre_celulas_do_holdout_2.1": n_lt1_h,
            "REFORCA_(i)_ge_12_com_denominador_kfold": bool(n_lt1_k >= 12),
            "REFORCA_(i)_ge_12_com_denominador_holdout": bool(n_lt1_h >= 12),
            "REFORCA_(ii)_limiar_1_4_do_holdout_dB": 0.25 * T_ref,
            "REFORCA_(ii)_dp_medio_lt_1_4_do_holdout": bool(dp_m < 0.25 * T_ref),
            "REFORCA_(ii)_dp_medio_lt_2.0_dB_literal": bool(dp_m < 2.0),
            "DELIMITA_n_celulas_dp_ge_1_2_do_holdout_da_propria_celula": n_ge_half_cel,
            "DELIMITA_n_celulas_dp_ge_1_2_do_dp_medio_holdout": n_ge_half_T,
            "DELIMITA_limiar_1_2_do_dp_medio_holdout_dB": 0.5 * T_ref,
            "DELIMITA_ge_8_celulas_(leitura_celula)": bool(n_ge_half_cel >= 8),
            "DELIMITA_ge_8_celulas_(leitura_dp_medio)": bool(n_ge_half_T >= 8),
            "razao_dp_medio_kfold_sobre_holdout": dp_m / T_ref,
            "razao_dp_kfold_sobre_dp_entre_celulas_kfold_(media_16/entre)": kb["razao_dp_sorteios_medio_sobre_dp_entre_celulas"],
        }
    ambiguidades = [
        "dp_entre_celulas no criterio 'razao dp_sorteios/dp_entre_celulas < 1 em >= 12 das 16 celulas': implementado com duas leituras do denominador (dp entre as 16 medias do proprio K-fold conjunto; dp entre as 16 medias do hold-out do 2.1 = T.dp_entre_medias = 2,18 dB). Contagens separadas.",
        "Limiar '< 1/4 do hold-out (< 2,0 dB)': 0,25 x T.dp_sorteio = 2,004 dB; o texto escreve 2,0. Reportados ambos (campos REFORCA_(ii)).",
        "'dp entre sorteios >= 1/2 do hold-out em >= 8 celulas': hold-out lido (a) por celula (dp do hold-out da propria celula, b = 2, constante/validos) e (b) como a media das 16 celulas (T.dp_sorteio). Contagens separadas.",
        "mu_U (media do dominio por celula): R2_resumo.json so traz as 4 celulas Q1; mu_U das 16 celulas recomputado do tensor com a MESMA receita do R2 (calibracao unica da particao de seed 42: treino 92 blocos sem aparo; e_i sobre o dominio inteiro; conferido contra R2 nas 4 Q1). Como o preditor do K-fold varia por fold, relatado tambem o vies b = 2 menos b = 0 no mesmo sorteio (b = 0 pontua todo o dominio uma vez com o preditor de cada fold).",
        "Hold-out de referencia para populacoes/preditores alem de constante/validos/b=2: recomputado dos parciais (b = 2: fase2/_v3_2.1_3.1_parcial_16x60rnd.json; b = 0: fase4/R1_b0_por_sorteio.json), sorteios sem no valido (valor None) excluidos, mesma convencao do 2.1/R1.",
        "dp (ddof = 1) por celula: np.std; quando os 60 valores sao identicos o dp e fixado em 0.0 exato (np.std de valores identicos devolveu ate 1,07e-14 por arredondamento da media na 1a passada, com sha de script anterior; essa passada foi descartada e tudo foi reexecutado com o sha final).",
        "Err_conjunto soma as perdas por fold com np.sum (pairwise) e totaliza com math.fsum; para o constante em b = 0 a soma por fold e conferida contra math.fsum (campo controle_b0_constante_fsum).",
        "FSPL calibrado (b): relatado ao lado, sem regra de decisao (o criterio nao a define).",
    ]
    resumo = {
        "id": "E1_kfold_blocos_pontuado_em_conjunto", "criterio": str(CRITERIO), "criterio_sha256": CRITERIO_SHA256,
        "script": str(HERE), "script_sha256": SCRIPT_SHA256, "laco_importado": str(LACO_PATH), "laco_sha256": LACO_SHA256,
        "script_original_importado": str(laco.ORIG_PATH), "script_original_sha256": ORIG_SHA256,
        "comando_consolidacao": " ".join(sys.argv), "data_utc": agora(), "venv": sys.executable,
        "sementes": seeds, "sementes_fonte": str(laco.F2_JSON) + " :: nota_divergencia_seeds.seeds_usados_nesta_rodada",
        "K": K_FOLDS, "blocos_por_fold": 22, "buffers_km": list(BUFFERS_KM),
        "n_celulas": len(CELULAS), "n_sorteios": len(seeds),
        "parciais_script_sha256_unico": sorted(set(r["script_sha256"] for r in parc.values())),
        "referencias_holdout": conf,
        "validacao": ({"arquivo": str(OUT_VALID), "passou": valid["passou"], "script_sha256": valid["script_sha256"],
                       "igual_ao_sha_atual": valid["script_sha256"] == SCRIPT_SHA256} if valid else None),
        "controle_b0_constante_por_celula": ctrl_b0,
        "tabela": tab, "vies": vies, "conferencia_mu_U_vs_R2_resumo": conf_mu,
        "fracao_pontuada_por_celula": pont, "limiares_do_criterio_contagem_mecanica": L,
        "ambiguidades_e_leituras": ambiguidades,
    }
    OUT_RESUMO.write_text(json.dumps(resumo, indent=1, ensure_ascii=False), encoding="utf-8")
    por_sorteio = {"id": "E1_por_sorteio", "script_sha256": SCRIPT_SHA256, "sementes": seeds, "K": K_FOLDS,
                   "campos": "Err_conjunto[<pred>|<pop>|b<0|2>] por sorteio; pontuados; constante/offset por fold; "
                             "soma_por_fold (S por fold) e n_pontuados_por_fold",
                   "celulas": {cel: {"sha256_manifest": r["sha256_manifest"], "n_nodes_total": r["n_nodes_total"],
                                     "n_validos_total": r["n_validos_total"], "vizinhanca": r["vizinhanca"],
                                     "mu_U_ref_seed42": r["mu_U_ref_seed42"], "sorteios": r["por_sorteio"]}
                               for cel, r in parc.items()}}
    OUT_SORTEIO.write_text(json.dumps(por_sorteio, indent=0, ensure_ascii=False), encoding="utf-8")
    log(f"consolidado -> {OUT_RESUMO} e {OUT_SORTEIO}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validar", action="store_true")
    g.add_argument("--celula", default="")
    g.add_argument("--consolidar", action="store_true")
    a = ap.parse_args()
    if a.validar:
        validar()
    elif a.consolidar:
        consolidar()
    else:
        rodar_uma(a.celula)


if __name__ == "__main__":
    main()
