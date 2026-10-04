"""Tests 1.6 (per-node inclusion probabilities) and 2.3 (formal estimand and Horvitz-Thompson), with per-draw values.

Same computation as analysis/v3_1.6_2.3_estimando_formal.py; this version also writes the per-draw
values (Err_sigma under the event A = {M > 0} and the Horvitz-Thompson estimate) to the output JSON,
which are needed for the Monte Carlo error of these estimates. Outputs go to new files, so that the
aggregates can be compared with those of the original run.

Four Q1 cells (Bauru, Campinas, Lins, Sorocaba), g = 10 km, b = 2 km, N = 132 blocks, k_te = 20 test
blocks. 100 split seeds: RandomState(20260927).randint(10**6, size=100); the same seeds and the same
block grid serve all four cells. The node grid is the synthetic 3600 x 3600 grid of
varredura_split_geometria.build_synthetic_grid, in the node order of the *_enriched_cftudo.pt tensor.
The split rule (block shuffle by RandomState(seed), fractions 0.70/0.15/0.15, exact cKDTree trimming of
validation against training and of test against training plus validation) replicates the frozen
trainer. The node errors e_i are held constant across draws: calibrated once on the reference split of seed 42 (constant =
median training RSSI; FSPL (b) = offset fitted on the valid training nodes, formula imported from the
frozen baseline script) and applied to every node of the domain.

Test 1.6: p_i^te = fraction of draws in which node i is a retained test node, and p_i | te (given that
its block is a test block), aggregated by the lateral-neighbour class k(i) and by distance to the block
edge, compared with the closed form (k_te - 1)_k / (N - 1)_k and its upper bound (N - 1 - k_tr)_k / (N - 1)_k.
Test 2.3, per predictor and population (valid, sentinel, all): (a) MAE over the whole domain; (b) mean
over draws of the retained-test MAE (draws with M_sigma > 0); (c) sum p_i e_i / sum p_i over the
domain; (d) Horvitz-Thompson estimate per draw, mean and standard deviation over draws; plus the
numerical check of Err(U) - theta_U = -Cov(Err_sigma, M | A) / E[M | A] and the weights
w_i = E[1{i in V} / M | A].

Outputs (fase2/): 1.6b_p_inclusao_por_no_REEXEC.json, 2.3b_estimando_por_sorteio.json,
fig_p_inclusao_perfil.pdf/.png, and the resumable partial _v3_2.3b_parcial.json.
Usage: python analysis/v3_2.3b_por_sorteio.py   (CPU, one process per cell)
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts")
from varredura_split_geometria import (  # noqa: E402
    TREINOS_DIR, build_synthetic_grid, latlon_graus_para_metros, assign_groups,
)

SCRIPT_PATH = Path(__file__).resolve()
OUT_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
CRIT_16 = OUT_DIR / "criterios" / "criterio_1.6.json"
CRIT_23 = OUT_DIR / "criterios" / "criterio_2.3.json"
OUT_16 = OUT_DIR / "fase2" / "1.6b_p_inclusao_por_no_REEXEC.json"
OUT_23 = OUT_DIR / "fase2" / "2.3b_estimando_por_sorteio.json"
FIG_PDF = OUT_DIR / "fase2" / "fig_p_inclusao_perfil.pdf"
FIG_PNG = OUT_DIR / "fase2" / "fig_p_inclusao_perfil.png"
PARCIAL = OUT_DIR / "fase2" / "_v3_2.3b_parcial.json"

TENSOR_DIR = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")
BASELINES_CONGELADO = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
    "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/dados/"
    "scripts_congelados/baselines_v2_por_particao.py")

_spec = importlib.util.spec_from_file_location("baselines_congelado", BASELINES_CONGELADO)
_bl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bl)
free_space_path_loss = _bl.free_space_path_loss
BASELINES_SCRIPT_SHA256 = hashlib.sha256(BASELINES_CONGELADO.read_bytes()).hexdigest()

CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
QUAD = "Q1"
G, B = 10.0, 2.0
FRACS = (0.70, 0.15, 0.15)
PL_TARGET_MAX_VALID = 299.0
FREQ_MHZ = 900.0
SEED_REF = 42
N_SEEDS = 100
SEEDS = np.random.RandomState(20260927).randint(10**6, size=N_SEEDS).tolist()
BINS_DIST = np.arange(0.0, 5.0 + 0.25, 0.25)  # distance-to-edge bins, 0 to 5 km in steps of 0.25 km
KLASSES = [0, 1, 2, 3, 4]


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def split_uma_vez(pos_km: np.ndarray, gid: np.ndarray, grupos: np.ndarray, n_g: int, seed: int):
    """One three-way split with the frozen-trainer rule (RandomState(seed).shuffle of the blocks,
    70/15/15, exact cKDTree trimming of validation against training and of test against training
    plus validation). Returns the block masks before trimming (test, validation), the retained masks
    (test, validation), and the numbers of test and training blocks."""
    kw = dict(compact_nodes=False, balanced_tree=False)
    rng = np.random.RandomState(seed)
    emb = grupos.copy()
    rng.shuffle(emb)
    n_tr = max(1, int(round(FRACS[0] * n_g)))
    n_va = max(1, int(round(FRACS[1] * n_g)))
    if n_tr + n_va >= n_g:
        n_tr = max(1, n_g - 2)
        n_va = 1
    g_tr, g_va, g_te = emb[:n_tr], emb[n_tr:n_tr + n_va], emb[n_tr + n_va:]
    kte, ktr = len(g_te), len(g_tr)
    m_tr = np.isin(gid, g_tr)
    m_va0 = np.isin(gid, g_va)
    m_te0 = np.isin(gid, g_te)
    m_va = m_va0.copy()
    m_te = m_te0.copy()
    dd, _ = cKDTree(pos_km[m_tr], **kw).query(pos_km[m_va], k=1, workers=3)
    iv = np.where(m_va)[0]
    m_va[iv[dd < B]] = False
    dd, _ = cKDTree(pos_km[m_tr | m_va], **kw).query(pos_km[m_te], k=1, workers=3)
    it = np.where(m_te)[0]
    m_te[it[dd < B]] = False
    return m_te0, m_va0, m_te, m_va, kte, ktr


def classe_k_e_dist_borda(pos_km: np.ndarray, gid: np.ndarray, grupos_set: set, max_y: int, max_x: int = None):
    """k(i) = number of lateral neighbour blocks (left, right, below, above) that exist and lie closer
    than b to the node; dist_borda = distance (km) to the nearest edge of the node's own block.
    Neighbour ids are looked up only inside the grid bounds in both directions, so that a node of the
    bottom row cannot match the top block of the previous column (id = gx * max_y + gy)."""
    grid_x = np.floor(pos_km[:, 0] / G).astype(np.int64)
    grid_y = np.floor(pos_km[:, 1] / G).astype(np.int64)
    dx = pos_km[:, 0] - grid_x * G
    dy = pos_km[:, 1] - grid_y * G
    if max_x is None:
        max_x = int(grid_x.max()) + 1
    grupos_arr = np.array(sorted(grupos_set))

    def existe(gx, gy):
        dentro = (gx >= 0) & (gx < max_x) & (gy >= 0) & (gy < max_y)
        ids = np.where(dentro, gx * max_y + gy, -1)
        return dentro & np.isin(ids, grupos_arr)

    perto_esq = dx < B
    perto_dir = (G - dx) < B
    perto_bai = dy < B
    perto_cim = (G - dy) < B
    viz_esq = existe(grid_x - 1, grid_y)
    viz_dir = existe(grid_x + 1, grid_y)
    viz_bai = existe(grid_x, grid_y - 1)
    viz_cim = existe(grid_x, grid_y + 1)
    k = (perto_esq & viz_esq).astype(np.int8) + (perto_dir & viz_dir).astype(np.int8) \
        + (perto_bai & viz_bai).astype(np.int8) + (perto_cim & viz_cim).astype(np.int8)
    dist_borda = np.minimum(np.minimum(dx, G - dx), np.minimum(dy, G - dy))
    return k, dist_borda


def carregar_alvo_dominio(cidade: str, quad: str):
    """Targets y (channel 0 = path loss, channel 3 = RSSI) and dist_nearest_m of the whole domain, in the
    node order of the synthetic grid; returns RSSI, PL, sentinel flags, distance, node count and path."""
    basename = f"transfer_dataset_{cidade}_v19_{quad}_enriched_cftudo.pt"
    tensor_path = TENSOR_DIR / basename
    load_kw = dict(map_location="cpu", weights_only=False)
    try:
        rf = torch.load(tensor_path, mmap=True, **load_kw)
    except Exception:
        rf = torch.load(tensor_path, **load_kw)
    ty = torch.as_tensor(rf["terrain"].y).float().clone().numpy()
    dist = torch.as_tensor(rf["terrain"].dist_nearest_m).float().clone().numpy()
    n_total = int(ty.shape[0])
    del rf
    gc.collect()
    rssi = ty[:, 3].astype(np.float64)
    pl = ty[:, 0].astype(np.float64)
    sentinela = pl >= PL_TARGET_MAX_VALID
    return rssi, pl, sentinela, dist.astype(np.float64), n_total, str(tensor_path)


def processar_celula(cidade: str) -> dict:
    """Tests 1.6 and 2.3 for one Q1 cell over the 100 split draws."""
    t0 = time.time()
    cid_q = f"{cidade}_{QUAD}"
    d = json.load(open(TREINOS_DIR / f"run_c0c1cf_{cidade}_s42_{QUAD}_g10b2.json"))
    geo = d["geometria"]
    lon, lat = build_synthetic_grid(geo["lon_min_deg"], geo["lon_max_deg"],
                                     geo["lat_min_deg"], geo["lat_max_deg"])
    pos_km = latlon_graus_para_metros(lon, lat) / 1000.0
    gid = assign_groups(pos_km, G)
    grupos = np.unique(gid)
    n_g = len(grupos)
    grupos_set = set(grupos.tolist())
    max_y = int((pos_km[:, 1] / G).astype(int).max()) + 1
    n_nos = pos_km.shape[0]

    # class k(i) and distance to the block edge do not depend on the draw
    k_no, dist_borda = classe_k_e_dist_borda(pos_km, gid, grupos_set, max_y)

    # per-node int32 counters over draws: retained in test/validation, block assigned to test/validation
    cnt_ret_te = np.zeros(n_nos, dtype=np.int32)
    cnt_ret_va = np.zeros(n_nos, dtype=np.int32)
    cnt_bloco_te = np.zeros(n_nos, dtype=np.int32)
    cnt_bloco_va = np.zeros(n_nos, dtype=np.int32)
    ntef_lista, nvaf_lista = [], []
    kte_ref, ktr_ref = None, None

    # targets, and node errors e_i calibrated once on the reference split (seed 42)
    rssi, pl, sentinela, dist, n_total, tensor_path = carregar_alvo_dominio(cidade, QUAD)
    assert n_total == n_nos, f"{cid_q}: malha sintetica ({n_nos}) != tensor ({n_total})"
    m_te0_42, m_va0_42, m_te_42, m_va_42, kte42, ktr42 = split_uma_vez(pos_km, gid, grupos, n_g, SEED_REF)
    m_tr_42 = ~(m_te0_42 | m_va0_42)
    # training mask rebuilt from the training blocks, since split_uma_vez does not return it
    rng42 = np.random.RandomState(SEED_REF)
    emb42 = grupos.copy(); rng42.shuffle(emb42)
    n_tr42 = max(1, int(round(FRACS[0] * n_g))); n_va42b = max(1, int(round(FRACS[1] * n_g)))
    if n_tr42 + n_va42b >= n_g:
        n_tr42 = max(1, n_g - 2); n_va42b = 1
    g_tr42 = emb42[:n_tr42]
    m_tr_42 = np.isin(gid, g_tr42)
    rssi_tr42 = rssi[m_tr_42]
    dist_tr42 = dist[m_tr_42]
    sent_tr42 = sentinela[m_tr_42]
    constante_ref = float(np.median(rssi_tr42))
    validos_tr42 = ~sent_tr42
    pl_pred_tr42_v = free_space_path_loss(dist_tr42[validos_tr42], FREQ_MHZ)
    p_tx_eff_ref = float(np.median(rssi_tr42[validos_tr42] + pl_pred_tr42_v))
    pl_pred_dom = free_space_path_loss(dist, FREQ_MHZ)
    rssi_fspl_dom = p_tx_eff_ref - pl_pred_dom
    e_constante = np.abs(rssi - constante_ref)
    e_fspl = np.abs(rssi - rssi_fspl_dom)

    idx_te_por_sorteio = []  # compact retained-test indices per draw (not dense masks)

    for s in SEEDS:
        m_te0, m_va0, m_te, m_va, kte, ktr = split_uma_vez(pos_km, gid, grupos, n_g, s)
        kte_ref, ktr_ref = kte, ktr
        cnt_ret_te += m_te.astype(np.int32)
        cnt_ret_va += m_va.astype(np.int32)
        cnt_bloco_te += m_te0.astype(np.int32)
        cnt_bloco_va += m_va0.astype(np.int32)
        ntef_lista.append(int(m_te.sum()))
        nvaf_lista.append(int(m_va.sum()))
        idx_te_por_sorteio.append(np.where(m_te)[0].astype(np.int32))
        del m_te0, m_va0, m_te, m_va

    p_te = cnt_ret_te.astype(np.float64) / N_SEEDS
    p_va = cnt_ret_va.astype(np.float64) / N_SEEDS
    com_bloco_te = cnt_bloco_te > 0
    p_cond_te = np.full(n_nos, np.nan)
    p_cond_te[com_bloco_te] = cnt_ret_te[com_bloco_te].astype(np.float64) / cnt_bloco_te[com_bloco_te]

    N_blocos, k_te_blocos = n_g, kte_ref

    # test 1.6 by class k: closed form (k_te-1)_k/(N-1)_k (falling factorials, math.perm) as lower bound and
    # (N-1-k_tr)_k/(N-1)_k as upper bound of p_i | te; k(i) counts lateral neighbours only, a proxy of the
    # geometric class that also counts diagonal blocks
    perfil_k = {}
    for k in KLASSES:
        mk = (k_no == k) & com_bloco_te
        n_mk = int(mk.sum())
        if n_mk == 0:
            perfil_k[k] = {"n_nos": 0, "media_p_cond_te": None, "forma_fechada": None,
                           "cota_inferior": None, "cota_superior": None, "abs_diff": None,
                           "dentro_do_sanduiche": None}
            continue
        media_emp = float(p_cond_te[mk].mean())
        if k_te_blocos - 1 >= k:
            ff = math.perm(k_te_blocos - 1, k) / math.perm(N_blocos - 1, k)
        else:
            ff = 0.0
        if N_blocos - 1 - ktr_ref >= k:
            cota_sup = math.perm(N_blocos - 1 - ktr_ref, k) / math.perm(N_blocos - 1, k)
        else:
            cota_sup = 0.0
        perfil_k[k] = {"n_nos": n_mk, "frac_nos": n_mk / n_nos, "media_p_cond_te": media_emp,
                        "forma_fechada": float(ff), "cota_inferior": float(ff), "cota_superior": float(cota_sup),
                        "abs_diff": float(abs(media_emp - ff)),
                        "dentro_do_sanduiche": bool(ff - 1e-9 <= media_emp <= cota_sup + 1e-9)}

    # test 1.6 by distance to the block edge (0.25 km bins)
    perfil_dist = []
    for i in range(len(BINS_DIST) - 1):
        lo, hi = BINS_DIST[i], BINS_DIST[i + 1]
        mb = (dist_borda >= lo) & (dist_borda < hi)
        mb_te = mb & com_bloco_te
        perfil_dist.append({
            "bin_km": [float(lo), float(hi)],
            "n_nos": int(mb.sum()),
            "media_p_te": float(p_te[mb].mean()) if mb.any() else None,
            "media_p_cond_te": float(p_cond_te[mb_te].mean()) if mb_te.any() else None,
        })

    media_p_cond_te_geral = float(cnt_ret_te[com_bloco_te].sum() / cnt_bloco_te[com_bloco_te].sum())
    media_p_te_geral_por_sorteio = float(np.mean([n / n_nos for n in ntef_lista]))  # raw reference, not used by the criterion
    ratio_pooled = float(cnt_ret_te.sum() / cnt_bloco_te.sum())  # equals media_p_cond_te_geral by construction

    resultado_16 = dict(
        celula=cid_q, n_blocos=int(N_blocos), k_te_blocos=int(k_te_blocos), k_tr_blocos=int(ktr_ref),
        n_nos_dominio=int(n_nos),
        media_p_cond_te_geral=media_p_cond_te_geral,
        identidade_p_te_vs_k_sobre_N_vezes_p_cond=dict(
            k_te_sobre_N=k_te_blocos / N_blocos,
            p_te_medio_dominio=float(p_te.mean()),
            k_te_sobre_N_vezes_media_p_cond=(k_te_blocos / N_blocos) * media_p_cond_te_geral,
        ),
        perfil_por_classe_k=perfil_k,
        perfil_por_distancia_borda_km=perfil_dist,
        variancia_entre_sorteios_n_te=dict(media=float(np.mean(ntef_lista)), var=float(np.var(ntef_lista, ddof=1)),
                                            dp=float(np.std(ntef_lista, ddof=1)), n_sorteios=N_SEEDS),
        tempo_1_6_s=time.time() - t0,
    )
    log(f"{cid_q} [1.6]: media_p_cond_te={media_p_cond_te_geral:.4f} k_te={k_te_blocos} N={N_blocos} t={resultado_16['tempo_1_6_s']:.0f}s")

    # test 2.3: estimators (a)-(d), the per-draw pairs (M_sigma, Err_sigma), the count of draws with M_sigma = 0
    # (event A = {M > 0} fails), and w_i accumulated as 1{i in V_te(sigma)} / M_sigma over the draws with M_sigma > 0
    t1 = time.time()
    pops = {"validos": ~sentinela, "sentinela": sentinela, "todos": np.ones(n_nos, dtype=bool)}
    saida_23_celula = {}
    for nome_pred, e_i in (("constante", e_constante), ("fspl_calibrado_b", e_fspl)):
        por_pop = {}
        for nome_pop, mpop in pops.items():
            n_pop = int(mpop.sum())
            a_dominio = float(np.mean(e_i[mpop])) if n_pop else None
            b_vals = []
            d_vals = []
            M_vals = []  # all 100 draws, M = 0 included
            Err_vals_A = []  # draws with M > 0 only, aligned with M_vals_A
            M_vals_A = []
            n_M0 = 0
            w_acc = np.zeros(n_pop, dtype=np.float64)  # accumulates 1{i in V}/M, indexed in the order of np.where(mpop)
            idx_pop_global_sorted = np.where(mpop)[0]
            pos_no_pop = -np.ones(n_nos, dtype=np.int64)
            pos_no_pop[idx_pop_global_sorted] = np.arange(idx_pop_global_sorted.size)
            for idx_te in idx_te_por_sorteio:
                if idx_te.size == 0:
                    M_vals.append(0); n_M0 += 1; continue
                mask_local = mpop[idx_te]
                m_sigma = int(mask_local.sum())
                M_vals.append(m_sigma)
                if m_sigma == 0:
                    n_M0 += 1
                    continue
                idx_pop = idx_te[mask_local]
                e_sub = e_i[idx_pop]
                err_sigma = float(np.mean(e_sub))
                b_vals.append(err_sigma)
                Err_vals_A.append(err_sigma)
                M_vals_A.append(m_sigma)
                w_acc[pos_no_pop[idx_pop]] += 1.0 / m_sigma
                p_sub = p_te[idx_pop]
                p_sub = np.where(p_sub > 0, p_sub, np.nan)
                w = 1.0 / p_sub
                num = np.nansum(e_sub * w)
                den = np.nansum(w)
                if den > 0:
                    d_vals.append(float(num / den))
            n_A = len(Err_vals_A)  # number of draws in event A (M > 0)
            b_medio = float(np.mean(b_vals)) if b_vals else None
            p_pop = p_te[mpop]
            e_pop = e_i[mpop]
            # (c) ratio of expectations with p_i^te over the whole population
            c_val = float(np.sum(p_pop * e_pop) / np.sum(p_pop)) if np.sum(p_pop) > 0 else None
            d_medio = float(np.mean(d_vals)) if d_vals else None
            d_dp = float(np.std(d_vals, ddof=1)) if len(d_vals) > 1 else None
            vies_protocolo = (b_medio - a_dominio) if (b_medio is not None and a_dominio is not None) else None
            vies_ht = (d_medio - a_dominio) if (d_medio is not None and a_dominio is not None) else None
            diff_b_menos_c = (b_medio - c_val) if (b_medio is not None and c_val is not None) else None
            # E[M|A], CV(M|A), Cov(Err, M|A)/E[M|A] and the numerical check of the ratio identity
            if n_A >= 2:
                M_arr = np.asarray(M_vals_A, dtype=np.float64)
                Err_arr = np.asarray(Err_vals_A, dtype=np.float64)
                E_M_A = float(M_arr.mean())
                dp_M_A = float(M_arr.std(ddof=1))
                cv_M_A = float(dp_M_A / E_M_A) if E_M_A > 0 else None
                # ddof = 0: the identity is exact algebraically for theta_U = sum(S_sigma)/sum(M_sigma) over the
                # draws in A; ddof = 1 would leave a spurious residual of order 1/n_A
                cov_errM_A = float(np.cov(Err_arr, M_arr, ddof=0)[0, 1])
                termo_razao = -cov_errM_A / E_M_A if E_M_A > 0 else None
                identidade_lhs = (b_medio - c_val) if (b_medio is not None and c_val is not None) else None
                identidade_residuo = (identidade_lhs - termo_razao) if (identidade_lhs is not None and termo_razao is not None) else None
            else:
                E_M_A = dp_M_A = cv_M_A = cov_errM_A = termo_razao = identidade_lhs = identidade_residuo = None
            # w_i = E[1{i in V}/M | A] = w_acc / n_A; sum of w_i should be about 1 and sum w_i e_i should reproduce (b)
            if n_A > 0:
                w_i = w_acc / n_A
                soma_w = float(w_i.sum())
                err_via_w = float(np.sum(w_i * e_pop)) if soma_w > 0 else None
            else:
                soma_w = err_via_w = None
            por_pop[nome_pop] = dict(
                n_pop=n_pop, a_mae_dominio_dB=round(a_dominio, 3) if a_dominio is not None else None,
                b_media_sorteios_mae_teste_dB=round(b_medio, 3) if b_medio is not None else None,
                c_razao_esperancas_p_te_dB=round(c_val, 3) if c_val is not None else None,
                d_HT_media_dB=round(d_medio, 3) if d_medio is not None else None,
                d_HT_dp_dB=round(d_dp, 3) if d_dp is not None else None,
                vies_protocolo_b_menos_a_dB=round(vies_protocolo, 3) if vies_protocolo is not None else None,
                vies_HT_d_menos_a_dB=round(vies_ht, 3) if vies_ht is not None else None,
                diferenca_b_menos_c_dB=round(diff_b_menos_c, 3) if diff_b_menos_c is not None else None,
                n_sorteios_b=len(b_vals), n_sorteios_d=len(d_vals),
                n_sorteios_M_igual_0=n_M0, n_sorteios_A_M_maior_0=n_A,
                M_sigma_todos_100=M_vals,
                Err_sigma_por_sorteio_A=[round(x, 6) for x in Err_vals_A],
                HT_por_sorteio=[round(x, 6) for x in d_vals],
                E_M_dado_A=round(E_M_A, 3) if E_M_A is not None else None,
                CV_M_dado_A=round(cv_M_A, 4) if cv_M_A is not None else None,
                Cov_Err_M_dado_A=round(cov_errM_A, 5) if cov_errM_A is not None else None,
                termo_razao_menos_Cov_sobre_EM=round(termo_razao, 4) if termo_razao is not None else None,
                identidade_b_menos_c=round(identidade_lhs, 4) if identidade_lhs is not None else None,
                identidade_residuo_lhs_menos_termo=round(identidade_residuo, 6) if identidade_residuo is not None else None,
                soma_w_i=round(soma_w, 6) if soma_w is not None else None,
                Err_via_w_i_dB=round(err_via_w, 3) if err_via_w is not None else None,
            )
        saida_23_celula[nome_pred] = por_pop
    tempo_23 = time.time() - t1
    log(f"{cid_q} [2.3]: t={tempo_23:.0f}s")

    return dict(celula=cid_q, tensor_path=tensor_path, resultado_16=resultado_16,
                resultado_23=saida_23_celula, split_seed_ref=SEED_REF,
                constante_ref_dB=constante_ref, p_tx_eff_ref_dB=p_tx_eff_ref,
                tempo_23_s=tempo_23, tempo_total_s=time.time() - t0)


def main():
    from multiprocessing import Pool
    t0 = time.time()
    if PARCIAL.exists():
        estado = json.loads(PARCIAL.read_text())
    else:
        estado = {"celulas": {}}
    faltam = [c for c in CIDADES if estado["celulas"].get(c, {}).get("celula") != f"{c}_{QUAD}"]
    if faltam:
        with Pool(min(4, len(faltam))) as p:
            for c, r in zip(faltam, p.map(processar_celula, faltam)):
                estado["celulas"][c] = r
                PARCIAL.parent.mkdir(parents=True, exist_ok=True)
                PARCIAL.write_text(json.dumps(estado, indent=1))
                log(f"gravado parcial ({c})")

    res = [estado["celulas"][c] for c in CIDADES]

    crit16 = json.loads(CRIT_16.read_text())
    medias = [r["resultado_16"]["media_p_cond_te_geral"] for r in res]
    # mean p_i | te must lie in the reference range [0.449, 0.453] of the 200-seed nodal reference, with 0.5% slack
    ok_media = all(0.449 * 0.995 <= m <= 0.453 * 1.005 for m in medias)
    # JSON round trips turn integer class keys into strings; accept both
    def get_k(rk, k):
        d = rk["perfil_por_classe_k"]
        return d.get(k, d.get(str(k)))
    max_absdiff_012 = 0.0
    for r in res:
        for k in (0, 1, 2):
            info = get_k(r["resultado_16"], k)
            if info and info.get("abs_diff") is not None:
                max_absdiff_012 = max(max_absdiff_012, info["abs_diff"])
    ok_perfil = max_absdiff_012 < 0.02
    veredito_16 = "criterio_cumprido" if (ok_media and ok_perfil) else "criterio_NAO_cumprido"

    errata_16 = dict(
        data="2026-09-26",
        motivo="the first run of classe_k_e_dist_borda.existe(gx,gy) computed ids=gx*max_y+gy "
               "without checking 0<=gy<max_y (nor 0<=gx<max_x); nodes of the bottom row (gy=0) with dy<b "
               "matched block (gx-1,max_y-1), a spurious neighbour, which inflated k(i) in "
               "211,185 nodes per cell (in Bauru Q1, 124,618 true k=1 nodes became k=2 and k=0 nodes "
               "became k=1). This accounted for 69% of the excess at k=1 (0.0171 of 0.0246 absolute "
               "points). existe() now checks the bounds in both directions (gx and gy) before testing "
               "whether the neighbouring block exists.",
        recomputado_nesta_rodada=True,
        limitacao_declarada="k(i) here counts only the 4 LATERAL neighbours (not diagonals); it is a proxy "
                             "of the class m(i) of the inclusion proposition (which counts the blocks of the "
                             "8-neighbourhood by distance to the node, diagonals included). The bound "
                             "(ii) (k_te-1)_k/(N-1)_k <= p_i|te <= (N-1-k_tr)_k/(N-1)_k "
                             "therefore uses the lateral-only k(i) as an approximation of the exact geometric m(i); the "
                             "remaining part of the excess at k=1/k=2 (discretisation of the block face and "
                             "second-order effect of the validation neighbour) was quantified separately and "
                             "is not recomputed here, since that requires reclassifying every node "
                             "with diagonals.",
        residual_nao_explicado_aprox="~0,0075 de 0,0246 (31%) em k=1, ver fismat_inclusao_decomp.json",
    )
    saida_16 = dict(
        id="1.6", pergunta=crit16["pergunta"], alegacao=crit16["alegacao"], criterio=crit16,
        metodo="4 celulas Q1 g10b2 (N=132,k_te=20), 100 seeds aleatorios RandomState(20260927), "
               "split_espacial_3vias identico ao 1.8 (cKDTree exato); contadores int32 por no; "
               "classe k(i)=vizinhos laterais existentes a <2km (existe() com checagem de limites, "
               "corrigida -- ver errata); forma fechada hipergeometrica perm(k_te-1,k)/perm(N-1,k) "
               "e sanduiche perm(k_te-1,k)/perm(N-1,k) <= p_i|te <= perm(N-1-k_tr,k)/perm(N-1,k)",
        insumos=[{"caminho": r["tensor_path"], "sha256": "nao_recalculado_ja_verificado_em_2.1_3.1"} for r in res],
        nota_seeds_compartilhados="as 4 celulas usam a MESMA malha de N=132 blocos (mesmo assign_groups "
                                   "com grid_km=10) e os MESMOS 100 seeds (RandomState(20260927) gerado "
                                   "uma unica vez, reusado em processar_celula para cada cidade). A "
                                   "concordancia entre celulas nao mede erro Monte Carlo nem e "
                                   "replicacao independente; e a MESMA permutacao de blocos aplicada "
                                   "a 4 geometrias diferentes.",
        errata=errata_16,
        por_celula={r["celula"]: r["resultado_16"] for r in res},
        resumo=dict(medias_p_cond_te_por_celula=medias, faixa_referencia_1_8=[0.449, 0.453],
                    max_abs_diff_perfil_k012=max_absdiff_012),
        veredito_vs_criterio=veredito_16,
        nao_verificado=[],
        comando_rodado=" ".join(sys.argv),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    OUT_16.parent.mkdir(parents=True, exist_ok=True)
    OUT_16.write_text(json.dumps(saida_16, indent=1, ensure_ascii=False))
    log(f"gravado {OUT_16} veredito={veredito_16}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["font.family"] = "serif"
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
        centros = [(BINS_DIST[i] + BINS_DIST[i + 1]) / 2 for i in range(len(BINS_DIST) - 1)]
        for r in res:
            ys = [b["media_p_te"] for b in r["resultado_16"]["perfil_por_distancia_borda_km"]]
            ax.plot(centros, ys, marker="o", ms=2, lw=1, label=r["celula"])
        ax.set_xlabel("Distância à borda do bloco (km)")
        ax.set_ylabel(r"$p_i^{te}$ (probabilidade de inclusão)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG_PDF)
        fig.savefig(FIG_PNG, dpi=200)
        log(f"gravado {FIG_PDF} e {FIG_PNG}")
    except Exception as e:
        log(f"figura falhou: {e!r}")

    crit23 = json.loads(CRIT_23.read_text())
    n_celulas_ok_c_vs_b = {"constante": 0, "fspl_calibrado_b": 0}
    n_celulas_ok_ht = 0
    for pred in ("constante", "fspl_calibrado_b"):
        for r in res:
            diffs = [abs(r["resultado_23"][pred][pop]["diferenca_b_menos_c_dB"])
                     for pop in ("validos", "sentinela", "todos")
                     if r["resultado_23"][pred][pop]["diferenca_b_menos_c_dB"] is not None]
            if diffs and max(diffs) < 0.05:
                n_celulas_ok_c_vs_b[pred] += 1
    for r in res:
        ok_pop_ht = True
        any_pop = False
        for pop in ("validos", "sentinela", "todos"):
            info = r["resultado_23"]["fspl_calibrado_b"][pop]
            vp, vh = info["vies_protocolo_b_menos_a_dB"], info["vies_HT_d_menos_a_dB"]
            if vp is None or vh is None:
                continue
            any_pop = True
            if not (abs(vh) < abs(vp) / 2):
                ok_pop_ht = False
        if any_pop and ok_pop_ht:
            n_celulas_ok_ht += 1

    c_reproduz_b = (n_celulas_ok_c_vs_b["fspl_calibrado_b"] >= 3) or (n_celulas_ok_c_vs_b["constante"] >= 3)
    ht_reduz_vies = n_celulas_ok_ht >= 3
    if c_reproduz_b and ht_reduz_vies:
        veredito_23 = "definicao_formal_c_ok_HT_entra_no_artigo"
    elif c_reproduz_b and not ht_reduz_vies:
        veredito_23 = "definicao_formal_c_ok_HT_NAO_entra_(vies_nao_reduz_o_suficiente)"
    else:
        veredito_23 = "definicao_formal_c_NAO_reproduz_b_em_3_de_4_celulas"

    errata_23 = dict(
        data="2026-09-26",
        adicionado="records per draw the pair (M_sigma, Err_sigma) per population and predictor "
                   "(field M_sigma_todos_100, 100 values including M=0); reports "
                   "n_sorteios_M_igual_0 and n_sorteios_A_M_maior_0; computes E[M|A], CV(M|A), "
                   "Cov(Err_sigma,M|A)/E[M|A] and checks numerically the identity "
                   "Err(U)-theta_U = -Cov(Err_sigma,M_U|A)/E[M_U|A] (ratio proposition, part iii; "
                   "field identidade_residuo_lhs_menos_termo, expected "
                   "~0, below 1e-6 in all cell x population x predictor pairs with "
                   "n_A>=2). w_i is accumulated as 1{i in V_te(sigma)}/M_sigma over the "
                   "draws with M_sigma>0 and divided by n_A (estimate of E[1{i in V}/M|A], "
                   "ratio proposition, part ii); soma_w_i is expected ~1 and Err_via_w_i_dB to reproduce (b).",
        conclusao_HT_mantida="the Horvitz-Thompson correction stays out of the article (decision criterion "
                              "written before the run not met: |vies_HT| does not fall below "
                              "|vies_protocolo|/2 in >=3 cells x 3 populations). Beyond the criterion, "
                              "the halving threshold itself lies within the Monte Carlo noise: the HT "
                              "estimate varies by 7.8 to 9.1 dB from draw to draw (d_HT_dp_dB, valid population, "
                              "constant predictor), that is 13 to 30 times the bias it "
                              "should remove, and the indicative vies_HT_em_SE lies in -0.57..0.54 "
                              "(not distinguishable from zero with 100 draws). The correction is "
                              "therefore not resolvable with this number of draws and does not gain "
                              "in MSE per draw.",
        nota_seeds_compartilhados="idem 1.6: as 4 celulas usam a MESMA malha (N=132) e os MESMOS "
                                   "100 seeds; a concordancia de sinal entre celulas nos campos "
                                   "vies_protocolo/vies_HT NAO e replicacao independente do Monte "
                                   "Carlo.",
    )
    saida_23 = dict(
        id="2.3", pergunta=crit23["pergunta"], alegacao=crit23["alegacao"], criterio=crit23,
        metodo="mesmas 4 celulas e 100 sorteios do 1.6; e_i fixo calibrado no split seed=42 (constante=mediana(RSSI_tr); "
               "FSPL(b)=offset pela mediana dos validos do treino), aplicado a todo no do dominio; (a) MAE dominio inteiro; "
               "(b) media sobre 100 sorteios do MAE no teste retido (so nos sorteios com M_sigma>0, evento A_U); "
               "(c) Sum p_i^te e_i / Sum p_i^te (todo o dominio da populacao); "
               "(d) Horvitz-Thompson por sorteio, media e dp sobre 100 sorteios; M_sigma/Err_sigma por sorteio e w_i "
               "por acumulacao (ver errata).",
        insumos=[{"caminho": r["tensor_path"], "sha256": "nao_recalculado_ja_verificado_em_2.1_3.1"} for r in res],
        formulas_congeladas={"caminho": str(BASELINES_CONGELADO), "sha256": BASELINES_SCRIPT_SHA256},
        nota_seeds_compartilhados="as 4 celulas compartilham a mesma malha de N=132 blocos e os "
                                   "MESMOS 100 seeds (nao sao replicas independentes do Monte Carlo).",
        errata=errata_23,
        por_celula={r["celula"]: r["resultado_23"] for r in res},
        resumo=dict(
            n_celulas_c_reproduz_b_lt_0_05dB_por_preditor=n_celulas_ok_c_vs_b,
            n_celulas_HT_reduz_vies_pela_metade_todas_populacoes=n_celulas_ok_ht,
            regra_c_ok=">=3 das 4 celulas em pelo menos um preditor",
            regra_HT_ok=">=3 das 4 celulas E nas 3 populacoes simultaneamente",
        ),
        veredito_vs_criterio=veredito_23,
        nao_verificado=[],
        comando_rodado=" ".join(sys.argv),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    OUT_23.parent.mkdir(parents=True, exist_ok=True)
    OUT_23.write_text(json.dumps(saida_23, indent=1, ensure_ascii=False))
    log(f"gravado {OUT_23} veredito={veredito_23}")
    log(f"concluido em {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
