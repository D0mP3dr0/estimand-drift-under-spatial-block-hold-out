#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared engine of checks R3 (design term) and R4 (closed-form inclusion probability).

For each of the four Q1 cells (g = 10 km, b = 2 km, N = 132 blocks, k_te = 20 test
blocks, k_tr = 92 training blocks) it computes, on CPU and without training:

1. The partition, imported unchanged from v3_1.6_2.3_estimando_formal.py (split_uma_vez:
   seeded shuffle of the blocks, 70/15/15, validation trimmed against training, test
   trimmed against training and retained validation, exact cKDTree distances).
2. The edge class m(i) of the inclusion proposition: the number of OTHER blocks holding
   at least one node at distance strictly below b from node i (Euclidean, km; same strict
   test dd < B as the partition), exact per node; and condition (iii) per node: for every
   neighbouring block, with j the node of that block nearest to i, every block with a node
   closer than b to j belongs to i's own block or to i's neighbour set. An alternative
   geometric class m_geom counts the existing 8-neighbour blocks whose face or corner lies
   closer than b to the node; it is used only as a sensitivity reading.
3. The per-node absolute errors e_i of the constant predictor and of FSPL with
   calibration (b), calibrated once on split seed 42 as in the imported module.
4. Over the 200 split draws of fase1/1.8_referencia_nodal_200seeds.json: per-node test
   retention counts, retained counts per (block, class), and per (predictor, population)
   the sum S_s of e_i and the count M_s of retained test nodes. The test retention of every
   draw is asserted equal to the stored retention of record 1.8 (to 1e-12).

Output (per cell, resumable): fase4/_v3_12_R3R4_intermediarios/<city>_Q1.npz and .json,
read by v3_12_R3_termo_desenho.py and v3_12_R4_p_fechado_vs_frequencia.py.

Usage: python v3_12_R3R4_motor.py   (run before the R3 and R4 scripts)
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCRIPTS = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts")
RAIZ = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
S16_PATH = SCRIPTS / "v3_1.6_2.3_estimando_formal.py"
F18 = RAIZ / "fase1" / "1.8_referencia_nodal_200seeds.json"
INTERM = RAIZ / "fase4" / "_v3_12_R3R4_intermediarios"

_spec = importlib.util.spec_from_file_location("s16", S16_PATH)
s16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s16)

G, B = s16.G, s16.B
CIDADES = s16.CIDADES
QUAD = s16.QUAD
SLOTS = 8   # maximum number of neighbouring blocks per node (asserted)
NCG_N = 10  # nodal class code: min(m, 4) * 2 + (condition (iii) fails)
NCG_G = 5   # geometric class code: min(m_geom, 4)
POPS = ("todos", "validos")
PREDS = ("constante", "fspl_calibrado_b")


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def seeds_200():
    """The 200 split seeds of record 1.8, asserted equal to RandomState(20260926).randint(1e6, size=200)."""
    d18 =json.load(open(F18))
    seeds = d18["seeds"]
    assert len(seeds) == 200
    assert seeds == np.random.RandomState(20260926).randint(10**6, size=200).tolist()
    return seeds, d18


def carregar_grade(cidade: str):
    d = json.load(open(s16.TREINOS_DIR / f"run_c0c1cf_{cidade}_s42_{QUAD}_g10b2.json"))
    geo = d["geometria"]
    lon, lat = s16.build_synthetic_grid(geo["lon_min_deg"], geo["lon_max_deg"],
                                         geo["lat_min_deg"], geo["lat_max_deg"])
    pos_km = s16.latlon_graus_para_metros(lon, lat) / 1000.0
    del lon, lat
    gid = s16.assign_groups(pos_km, G)
    grupos = np.unique(gid)
    return pos_km, gid, grupos


def calibrar_e(cidade: str, pos_km, gid, grupos):
    """Per-node absolute errors e_i, calibrated once on split seed SEED_REF.

    Literal copy of the calibration inside processar_celula of v3_1.6_2.3_estimando_formal.py
    (without its redundant split call): constant = median training RSSI; FSPL (b) effective
    transmit power = median over valid training nodes; applied to every node of the domain.
    """
    n_g = len(grupos)
    rssi, pl, sentinela, dist, n_total, tensor_path = s16.carregar_alvo_dominio(cidade, QUAD)
    assert n_total == pos_km.shape[0]
    rng42 = np.random.RandomState(s16.SEED_REF)
    emb42 = grupos.copy(); rng42.shuffle(emb42)
    n_tr42 = max(1, int(round(s16.FRACS[0] * n_g))); n_va42b = max(1, int(round(s16.FRACS[1] * n_g)))
    if n_tr42 + n_va42b >= n_g:
        n_tr42 = max(1, n_g - 2); n_va42b = 1
    g_tr42 = emb42[:n_tr42]
    m_tr_42 = np.isin(gid, g_tr42)
    rssi_tr42 = rssi[m_tr_42]
    dist_tr42 = dist[m_tr_42]
    sent_tr42 = sentinela[m_tr_42]
    constante_ref = float(np.median(rssi_tr42))
    validos_tr42 = ~sent_tr42
    pl_pred_tr42_v = s16.free_space_path_loss(dist_tr42[validos_tr42], s16.FREQ_MHZ)
    p_tx_eff_ref = float(np.median(rssi_tr42[validos_tr42] + pl_pred_tr42_v))
    pl_pred_dom = s16.free_space_path_loss(dist, s16.FREQ_MHZ)
    rssi_fspl_dom = p_tx_eff_ref - pl_pred_dom
    e_constante = np.abs(rssi - constante_ref)
    e_fspl = np.abs(rssi - rssi_fspl_dom)
    return dict(constante=e_constante, fspl_calibrado_b=e_fspl), sentinela, constante_ref, p_tx_eff_ref, tensor_path


def classes_nodais(pos_km, gid, grupos):
    """Per node: own block index, nodal class m, condition (iii) failure flag, m_geom, and max m.

    slots[i, s] holds the s-th neighbouring block of node i and near[i, s] the node of that
    block nearest to i (only neighbours closer than b are recorded).
    """
    n = pos_km.shape[0]
    n_g = len(grupos)
    own = np.searchsorted(grupos, gid).astype(np.int16)
    max_y = int((pos_km[:, 1] / G).astype(int).max()) + 1
    max_x = int((pos_km[:, 0] / G).astype(int).max()) + 1
    # Block id = column * max_y + row on the g-km grid.
    gxb = (grupos // max_y).astype(np.int64)
    gyb = (grupos % max_y).astype(np.int64)
    idx_bloco = {(int(a), int(b_)): k for k, (a, b_) in enumerate(zip(gxb, gyb))}
    order = np.argsort(own, kind="stable")
    starts = np.searchsorted(own[order], np.arange(n_g + 1))
    slots = -np.ones((n, SLOTS), dtype=np.int16)
    near = -np.ones((n, SLOTS), dtype=np.int32)
    cnt = np.zeros(n, dtype=np.int8)
    eps = 1e-6
    for k in range(n_g):
        nodes_k = order[starts[k]:starts[k + 1]]
        x0, y0 = gxb[k] * G, gyb[k] * G
        pk = pos_km[nodes_k]
        edge = np.minimum(np.minimum(pk[:, 0] - x0, x0 + G - pk[:, 0]), np.minimum(pk[:, 1] - y0, y0 + G - pk[:, 1]))
        band = nodes_k[edge < B + eps]  # nodes of block k within b of its border: the only ones that can be within b of another block
        if band.size == 0:
            continue
        tree = cKDTree(pos_km[band], compact_nodes=False, balanced_tree=False)
        cands = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                kk = idx_bloco.get((int(gxb[k]) + dx, int(gyb[k]) + dy))
                if kk is not None:
                    cands.append(order[starts[kk]:starts[kk + 1]])
        if not cands:
            continue
        cand = np.concatenate(cands)
        pc = pos_km[cand]
        box = (pc[:, 0] > x0 - B - eps) & (pc[:, 0] < x0 + G + B + eps) & (pc[:, 1] > y0 - B - eps) & (pc[:, 1] < y0 + G + B + eps)
        cand = cand[box]
        if cand.size == 0:
            continue
        dd, ii = tree.query(pos_km[cand], k=1, distance_upper_bound=B * (1 + 1e-6))
        ok = dd < B  # same strict test as the partition
        hit = cand[ok]
        if hit.size == 0:
            continue
        kslot = cnt[hit].astype(np.int64)
        assert kslot.max() < SLOTS, "no com mais de SLOTS blocos vizinhos"
        slots[hit, kslot] = k
        near[hit, kslot] = band[ii[ok]]
        cnt[hit] += 1
        del tree, cand, dd, ii
    m = cnt.astype(np.int8)
    # Condition (iii): every block within b of the nearest node j of each neighbour must be
    # i's own block or one of i's neighbours; otherwise the node is flagged.
    falha = np.zeros(n, dtype=bool)
    com = np.flatnonzero(m >= 1)
    for s in range(SLOTS):
        idx = com[m[com] > s]
        if idx.size == 0:
            break
        j = near[idx, s]
        ok_all = np.ones(idx.size, dtype=bool)
        own_i = own[idx]
        sl_i = slots[idx]
        for t in range(SLOTS):
            w = slots[j, t]
            vazio = (w == -1)
            permitido = vazio | (w == own_i) | (w[:, None] == sl_i).any(axis=1)
            ok_all &= permitido
        falha[idx] |= ~ok_all
        del idx, j, ok_all, own_i, sl_i
    # m_geom: existing 8-neighbour blocks whose face (distance to the side) or corner
    # (distance to the vertex) is closer than b to the node.
    grupos_set = grupos
    gx = np.floor(pos_km[:, 0] / G).astype(np.int64)
    gy = np.floor(pos_km[:, 1] / G).astype(np.int64)
    dx_ = pos_km[:, 0] - gx * G
    dy_ = pos_km[:, 1] - gy * G

    def existe_ok(i, j):
        okk = (i >= 0) & (i < max_x) & (j >= 0) & (j < max_y)
        return okk & np.isin(np.where(okk, i * max_y + j, -1), grupos_set)

    cand_g = [(-1, 0, dx_), (1, 0, G - dx_), (0, -1, dy_), (0, 1, G - dy_),
              (-1, -1, np.hypot(dx_, dy_)), (-1, 1, np.hypot(dx_, G - dy_)),
              (1, -1, np.hypot(G - dx_, dy_)), (1, 1, np.hypot(G - dx_, G - dy_))]
    mg = np.zeros(n, np.int8)
    for sx, sy, dist in cand_g:
        mg += ((dist < B) & existe_ok(gx + sx, gy + sy)).astype(np.int8)
    return own, m, falha, mg, int(cnt.max())


def processar_celula(cidade: str) -> str:
    """Compute classes, e_i and the per-draw tensors of one cell and write its .npz/.json (skipped if present).

    Array shapes: R_n (draws, blocks, NCG_N) and R_g (draws, blocks, NCG_G) retained test
    nodes per block and class; te_block (draws, blocks) True for test blocks; n_bc_n / n_bc_g
    (blocks, classes) node counts; S__<pred>|<pop> and M_<pop> (draws,) sums and counts.
    """
    t0 = time.time()
    saida = INTERM / f"{cidade}_{QUAD}.npz"
    if saida.exists():
        log(f"{cidade}: intermediario ja existe, pulando")
        return str(saida)
    seeds, d18 = seeds_200()
    ret18 = np.array(d18["por_celula"][f"{cidade}_{QUAD}"]["retencoes_te"])
    pos_km, gid, grupos = carregar_grade(cidade)
    n = pos_km.shape[0]
    n_g = len(grupos)
    own, m, falha, mg, m_max = classes_nodais(pos_km, gid, grupos)
    log(f"{cidade}: classes prontas (m_max={m_max}) em {time.time()-t0:.0f}s; "
        f"frac m={[round(float((m==j).mean()),4) for j in range(5)]}")
    cg_n = (np.minimum(m, 4).astype(np.int16) * 2 + falha.astype(np.int16))
    cg_g = np.minimum(mg, 4).astype(np.int16)
    own32 = own.astype(np.int64)
    n_bc_n = np.bincount(own32 * NCG_N + cg_n, minlength=n_g * NCG_N).reshape(n_g, NCG_N)
    n_bc_g = np.bincount(own32 * NCG_G + cg_g, minlength=n_g * NCG_G).reshape(n_g, NCG_G)
    e_por_pred, sentinela, constante_ref, p_tx_eff_ref, tensor_path = calibrar_e(cidade, pos_km, gid, grupos)
    mu = {}
    for pred in PREDS:
        mu[f"{pred}|todos"] = float(e_por_pred[pred].mean())
        mu[f"{pred}|validos"] = float(e_por_pred[pred][~sentinela].mean())
    log(f"{cidade}: e_i calibrado; mu={ {k: round(v,4) for k,v in mu.items()} }")
    cnt_ret = np.zeros(n, dtype=np.int16)
    R_n = np.zeros((len(seeds), n_g, NCG_N), dtype=np.int32)
    R_g = np.zeros((len(seeds), n_g, NCG_G), dtype=np.int32)
    te_block = np.zeros((len(seeds), n_g), dtype=bool)
    n_te0 = np.zeros(len(seeds), dtype=np.int64)
    n_te = np.zeros(len(seeds), dtype=np.int64)
    S = {f"{p}|{u}": np.zeros(len(seeds)) for p in PREDS for u in POPS}
    M = {u: np.zeros(len(seeds), dtype=np.int64) for u in POPS}
    n_tr = max(1, int(round(s16.FRACS[0] * n_g))); n_va = max(1, int(round(s16.FRACS[1] * n_g)))
    if n_tr + n_va >= n_g:
        n_tr = max(1, n_g - 2); n_va = 1
    kte_ref = ktr_ref = None
    for si, s in enumerate(seeds):
        m_te0, m_va0, m_te, m_va, kte, ktr = s16.split_uma_vez(pos_km, gid, grupos, n_g, s)
        kte_ref, ktr_ref = kte, ktr
        # Recreate the block shuffle of split_uma_vez to mark the test blocks (asserted against m_te0).
        emb = grupos.copy(); np.random.RandomState(s).shuffle(emb)
        te_idx_blocks = np.searchsorted(grupos, emb[n_tr + n_va:])
        te_block[si, te_idx_blocks] = True
        assert np.array_equal(te_block[si][own], m_te0)
        n_te0[si] = int(m_te0.sum()); n_te[si] = int(m_te.sum())
        assert abs(n_te[si] / n_te0[si] - ret18[si]) < 1e-12, f"{cidade} seed {s}: retencao difere do 1.8"
        cnt_ret += m_te.astype(np.int16)
        idx = np.flatnonzero(m_te)
        R_n[si] = np.bincount(own32[idx] * NCG_N + cg_n[idx], minlength=n_g * NCG_N).reshape(n_g, NCG_N)
        R_g[si] = np.bincount(own32[idx] * NCG_G + cg_g[idx], minlength=n_g * NCG_G).reshape(n_g, NCG_G)
        sent_idx = sentinela[idx]
        M["todos"][si] = idx.size
        M["validos"][si] = int((~sent_idx).sum())
        for pred in PREDS:
            ei = e_por_pred[pred][idx]
            S[f"{pred}|todos"][si] = float(ei.sum())
            S[f"{pred}|validos"][si] = float(ei[~sent_idx].sum())
        del m_te0, m_va0, m_te, m_va, idx, sent_idx
        if si % 20 == 0:
            log(f"{cidade}: sorteio {si+1}/{len(seeds)} t={time.time()-t0:.0f}s")
    INTERM.mkdir(parents=True, exist_ok=True)
    meta = dict(cidade=cidade, celula=f"{cidade}_{QUAD}", n_nos=int(n), n_blocos=int(n_g),
                k_te=int(kte_ref), k_tr=int(ktr_ref), seeds=seeds, m_max_observado=m_max,
                constante_ref_dB=constante_ref, p_tx_eff_ref_dB=p_tx_eff_ref, tensor_path=tensor_path,
                mu_dB=mu, n_nos_por_m=[int((m == j).sum()) for j in range(m_max + 1)],
                n_falha_iii_por_m=[int(((m == j) & falha).sum()) for j in range(m_max + 1)],
                n_nos_por_m_geom=[int((mg == j).sum()) for j in range(int(mg.max()) + 1)],
                n_sentinela=int(sentinela.sum()),
                sha256_motor=sha256_file(Path(__file__)), sha256_s16_importado=sha256_file(S16_PATH),
                sha256_1_8=sha256_file(F18), tempo_s=time.time() - t0,
                timestamp_utc=datetime.now(timezone.utc).isoformat())
    arrs = dict(m=m, falha=falha, mg=mg, cnt_ret=cnt_ret, R_n=R_n, R_g=R_g, te_block=te_block,
                n_bc_n=n_bc_n, n_bc_g=n_bc_g, n_te0=n_te0, n_te=n_te,
                M_todos=M["todos"], M_validos=M["validos"], **{f"S__{k}": v for k, v in S.items()})
    tmp = INTERM / f"{cidade}_{QUAD}.tmp.npz"
    np.savez_compressed(tmp, **arrs)  # written to a temporary name and renamed last, so a present .npz is complete
    (INTERM / f"{cidade}_{QUAD}.json").write_text(json.dumps(meta, indent=1))
    tmp.rename(saida)
    log(f"{cidade}: gravado {saida} em {time.time()-t0:.0f}s")
    gc.collect()
    return str(saida)


def carregar_intermediario(cidade: str):
    z = np.load(INTERM / f"{cidade}_{QUAD}.npz")
    meta = json.load(open(INTERM / f"{cidade}_{QUAD}.json"))
    return {k: z[k] for k in z.files}, meta


def main():
    from multiprocessing import Pool
    faltam = [c for c in CIDADES if not (INTERM / f"{c}_{QUAD}.npz").exists()]
    INTERM.mkdir(parents=True, exist_ok=True)
    if faltam:
        with Pool(min(4, len(faltam))) as p:
            for r in p.imap_unordered(processar_celula, faltam):
                log(f"parcial gravado: {r}")


if __name__ == "__main__":
    main()
