#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check R2: design reference for the mean nodal error over the whole domain.

For each cell, the target is mu_U, the mean absolute error e_i of a baseline over all
nodes of the domain (constant predictor and FSPL with calibration (b); e_i as in the
estimand record 2.3). Four estimators of mu_U are compared over split draws: H, the
buffered block hold-out (b = 2 km); H0, the same blocks without buffer; A, a simple
random sample of nodes of the same size as H; and J, a Hajek-weighted mean over the H
test nodes with inclusion probabilities p_i. For each estimator the script reports
bias, Monte Carlo standard error of the bias, RMSE, variance and the design effects
var(H)/var(A) and var(H0)/var(A).

Partition, target loading and FSPL formula are imported from v3_1.6_2.3_estimando_formal.py;
the cells are the four cities in the quadrant QUAD fixed by that module. The single
calibration on split seed 42 (constant = median training RSSI; FSPL offset = median over
valid training nodes) is a literal copy of the inline block of that module.

Step A uses the 100 split draws of records 1.6/2.3 (RandomState(20260927)): p_i is the
retention frequency of each node in the test partition over those draws (record 1.6
stores only profiles), and H is validated per draw against fase2/2.3b_estimando_por_sorteio.json.
Step B uses the 200 split draws of fase1/1.8 (RandomState(20260926)); sample A of draw j
uses RandomState(20261001 + j). J_p200 is a supplementary Hajek estimator with p_i taken
from the same 200 draws. Cells run in parallel (multiprocessing), CPU only.

Inputs: fase1/1.8_referencia_nodal_200seeds.json, fase2/1.6_p_inclusao_por_no.json,
fase2/2.3b_estimando_por_sorteio.json, fase2/2.3_estimando_ht_baselines.json and the
decision criterion criterios/criterio_R2_referencia_desenho.json.
Outputs (fase4/): _R2_parcial_<city>.json per cell, R2_por_sorteio.json, R2_resumo.json.

Usage: python v3_12_R2_referencia_desenho.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
V3 = BASE / "_v3_2026-09-25"
ORIG = BASE / "scripts" / "v3_1.6_2.3_estimando_formal.py"
SCRIPT_PATH = Path(__file__).resolve()
OUT = V3 / "fase4"
SEEDS_JSON = V3 / "fase1" / "1.8_referencia_nodal_200seeds.json"
P16_JSON = V3 / "fase2" / "1.6_p_inclusao_por_no.json"
H23B_JSON = V3 / "fase2" / "2.3b_estimando_por_sorteio.json"
H23_JSON = V3 / "fase2" / "2.3_estimando_ht_baselines.json"
CRIT = V3 / "criterios" / "criterio_R2_referencia_desenho.json"

spec = importlib.util.spec_from_file_location("v3_16_23_orig", ORIG)
O = importlib.util.module_from_spec(spec)
sys.modules["v3_16_23_orig"] = O
spec.loader.exec_module(O)

CIDADES = O.CIDADES
QUAD = O.QUAD
G, B = O.G, O.B
PREDS = ("constante", "fspl_calibrado_b")
POPS = ("validos", "todos")
POPS_VALID = ("validos", "sentinela", "todos")
ESTS = ("H", "H0", "A", "J", "J_p200")  # buffered hold-out, unbuffered hold-out, random sample, Hajek (p from 100 / 200 draws)
SEED_A_BASE = 20261001  # seed of the random sample A in draw j is SEED_A_BASE + j
TOL_VALID = 1e-5  # tolerance (dB) of the per-draw validation of H against record 2.3b


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def preparar_celula(cidade):
    """Build the node grid and blocks of one cell and the per-node absolute errors e_i of both baselines."""
    d = json.load(open(O.TREINOS_DIR / f"run_c0c1cf_{cidade}_s42_{QUAD}_g10b2.json"))
    geo = d["geometria"]
    lon, lat = O.build_synthetic_grid(geo["lon_min_deg"], geo["lon_max_deg"],
                                      geo["lat_min_deg"], geo["lat_max_deg"])
    pos_km = O.latlon_graus_para_metros(lon, lat) / 1000.0
    gid = O.assign_groups(pos_km, G)
    grupos = np.unique(gid)
    n_g = len(grupos)
    n_nos = pos_km.shape[0]
    rssi, pl, sentinela, dist, n_total, tensor_path = O.carregar_alvo_dominio(cidade, QUAD)
    assert n_total == n_nos
    # Single calibration on split seed SEED_REF, copied from the imported module:
    # constant = median training RSSI; FSPL effective transmit power = median over valid training nodes.
    rng42 = np.random.RandomState(O.SEED_REF)
    emb42 = grupos.copy(); rng42.shuffle(emb42)
    n_tr42 = max(1, int(round(O.FRACS[0] * n_g))); n_va42b = max(1, int(round(O.FRACS[1] * n_g)))
    if n_tr42 + n_va42b >= n_g:
        n_tr42 = max(1, n_g - 2); n_va42b = 1
    g_tr42 = emb42[:n_tr42]
    m_tr_42 = np.isin(gid, g_tr42)
    rssi_tr42 = rssi[m_tr_42]
    dist_tr42 = dist[m_tr_42]
    sent_tr42 = sentinela[m_tr_42]
    constante_ref = float(np.median(rssi_tr42))
    validos_tr42 = ~sent_tr42
    pl_pred_tr42_v = O.free_space_path_loss(dist_tr42[validos_tr42], O.FREQ_MHZ)
    p_tx_eff_ref = float(np.median(rssi_tr42[validos_tr42] + pl_pred_tr42_v))
    pl_pred_dom = O.free_space_path_loss(dist, O.FREQ_MHZ)
    rssi_fspl_dom = p_tx_eff_ref - pl_pred_dom
    e_constante = np.abs(rssi - constante_ref)
    e_fspl = np.abs(rssi - rssi_fspl_dom)
    return dict(pos_km=pos_km, gid=gid, grupos=grupos, n_g=n_g, n_nos=n_nos, sentinela=sentinela,
                e={"constante": e_constante, "fspl_calibrado_b": e_fspl},
                constante_ref=constante_ref, p_tx_eff_ref=p_tx_eff_ref, tensor_path=tensor_path)


def medias_pop(e, idx, sent):
    """Mean of e over idx per population (valid, sentinel, all), NaN when empty; also returns the valid count."""
    s = e[idx]
    sv = sent[idx]
    out = {}
    nv = int((~sv).sum())
    out["validos"] = float(s[~sv].mean()) if nv > 0 else float("nan")
    out["sentinela"] = float(s[sv].mean()) if (len(s) - nv) > 0 else float("nan")
    out["todos"] = float(s.mean()) if len(s) > 0 else float("nan")
    return out, nv


def hajek_pop(e, idx, sent, p):
    """Hajek mean sum(e/p)/sum(1/p) over the nodes of idx with p > 0, for valid and all nodes.

    Also returns the number of nodes dropped because p = 0.
    """
    pp = p[idx]
    ok = pp > 0
    n_drop = int((~ok).sum())
    ii = idx[ok]
    w = 1.0 / pp[ok]
    s = e[ii]
    sv = sent[ii]
    out = {}
    wv = w[~sv]
    out["validos"] = float((s[~sv] * wv).sum() / wv.sum()) if wv.size else float("nan")
    out["todos"] = float((s * w).sum() / w.sum()) if w.size else float("nan")
    return out, n_drop


def processar_celula(cidade):
    t0 = time.time()
    cid_q = f"{cidade}_{QUAD}"
    S = preparar_celula(cidade)
    pos_km, gid, grupos, n_g, n_nos, sent, e = (S[k] for k in ("pos_km", "gid", "grupos", "n_g", "n_nos", "sentinela", "e"))
    mu_U = {pred: {"validos": float(e[pred][~sent].mean()), "todos": float(e[pred].mean()),
                   "sentinela": float(e[pred][sent].mean())} for pred in PREDS}
    seeds200 = json.load(open(SEEDS_JSON))["seeds"]
    seeds100 = np.random.RandomState(20260927).randint(10**6, size=100).tolist()
    assert len(seeds200) == 200

    # Step A: 100 draws of records 1.6/2.3 -> p_i as retention frequency, and H for validation.
    cnt_ret = np.zeros(n_nos, dtype=np.int32)
    cnt_blo = np.zeros(n_nos, dtype=np.int32)
    H_val = {pred: {pop: [] for pop in POPS_VALID} for pred in PREDS}
    M_val = []
    for s in seeds100:
        m_te0, m_va0, m_te, m_va, kte, ktr = O.split_uma_vez(pos_km, gid, grupos, n_g, s)
        cnt_ret += m_te.astype(np.int32)
        cnt_blo += m_te0.astype(np.int32)
        idx = np.flatnonzero(m_te)
        for pred in PREDS:
            mp, nv = medias_pop(e[pred], idx, sent)
            for pop in POPS_VALID:
                H_val[pred][pop].append(mp[pop])
        M_val.append(int((~sent[idx]).sum()))
        del m_te0, m_va0, m_te, m_va
    p16 = cnt_ret.astype(np.float64) / 100.0
    com_b = cnt_blo > 0
    p_cond_geral = float(cnt_ret[com_b].sum() / cnt_blo[com_b].sum())
    ref16 = json.load(open(P16_JSON))["por_celula"][cid_q]["media_p_cond_te_geral"]

    # Validation against record 2.3b (per draw) and record 2.3 (ratio of expectations c, rounded to 3 decimals).
    b23 = json.load(open(H23B_JSON))["por_celula"][cid_q]
    a23 = json.load(open(H23_JSON))["por_celula"][cid_q]
    conf = {"p_cond_te_geral_recalculado": p_cond_geral, "p_cond_te_geral_1.6": ref16,
            "dif_p_cond": abs(p_cond_geral - ref16), "por_pred_pop": {}}
    ok_all = abs(p_cond_geral - ref16) < 1e-12
    for pred in PREDS:
        for pop in POPS_VALID:
            ref = b23[pred][pop]
            Msig = np.asarray(ref["M_sigma_todos_100"])
            esperado = np.asarray(ref["Err_sigma_por_sorteio_A"], dtype=float)
            mine = np.asarray(H_val[pred][pop], dtype=float)
            # Draws kept for the comparison: those with at least one node of the population in the test partition.
            if pop == "validos":
                ok_s = np.asarray(M_val) > 0
            elif pop == "todos":
                ok_s = np.ones(100, bool)
            else:
                ok_s = ~np.isnan(mine)
            mine_A = mine[ok_s]
            assert (Msig > 0).sum() == len(esperado), (cid_q, pred, pop)
            same_mask = bool(np.array_equal(Msig > 0, ok_s))
            maxdif = float(np.max(np.abs(mine_A - esperado))) if same_mask and len(esperado) == len(mine_A) else float("inf")
            mp = ~sent if pop == "validos" else (sent if pop == "sentinela" else np.ones(n_nos, bool))
            c_mine = float((p16[mp] * e[pred][mp]).sum() / p16[mp].sum())
            c_dif = abs(round(c_mine, 3) - ref["c_razao_esperancas_p_te_dB"])
            conf["por_pred_pop"][f"{pred}/{pop}"] = dict(
                n_sorteios_A=int(len(esperado)), mesma_mascara_M_maior_0=same_mask,
                max_abs_dif_Err_por_sorteio=maxdif, c_recalculado=c_mine, c_2_3=ref["c_razao_esperancas_p_te_dB"],
                dif_c_arred_3casas=c_dif,
                b_media_recalc=round(float(mine_A.mean()), 3) if len(mine_A) else None,
                b_media_2_3=ref["b_media_sorteios_mae_teste_dB"])
            ok_all &= same_mask and maxdif < TOL_VALID and c_dif < 1e-9
    conf["passou"] = bool(ok_all)
    conf["tolerancia_Err_por_sorteio"] = TOL_VALID
    log(f"{cid_q}: validacao (H) vs 2.3b passou={conf['passou']} t={time.time()-t0:.0f}s")
    if not ok_all:
        out = {"celula": cid_q, "validacao_H_vs_2.3": conf, "abortado": True}
        (OUT / f"_R2_parcial_{cidade}_VALIDACAO_FALHOU.json").write_text(json.dumps(out, indent=1))
        raise RuntimeError(f"{cid_q}: validacao de (H) falhou, ver parcial")
    n_zero_p16 = int((p16 == 0).sum())

    # Step B: 200 draws of record 1.8 -> estimators H, H0, A, J; J_p200 after p from these draws is known.
    res ={pred: {pop: {est: [] for est in ESTS} for pop in POPS} for pred in PREDS}
    info = {"n_te_b2": [], "n_te_b0": [], "n_te_b2_validos": [], "n_A_validos": [],
            "n_descartados_p16_zero_no_teste": [], "n_descartados_p200_zero_no_teste": []}
    cnt200 = np.zeros(n_nos, dtype=np.int32)
    idx_te_guardados = []
    for j, s in enumerate(seeds200):
        m_te0, m_va0, m_te, m_va, kte, ktr = O.split_uma_vez(pos_km, gid, grupos, n_g, s)
        idx = np.flatnonzero(m_te).astype(np.int32)
        idx0 = np.flatnonzero(m_te0).astype(np.int32)
        cnt200 += m_te.astype(np.int32)
        idx_te_guardados.append(idx)
        del m_te0, m_va0, m_te, m_va
        rs = np.random.RandomState(SEED_A_BASE + j)
        idxA = rs.choice(n_nos, size=len(idx), replace=False)
        info["n_te_b2"].append(int(len(idx)))
        info["n_te_b0"].append(int(len(idx0)))
        info["n_te_b2_validos"].append(int((~sent[idx]).sum()))
        info["n_A_validos"].append(int((~sent[idxA]).sum()))
        for pred in PREDS:
            for est, ii in (("H", idx), ("H0", idx0), ("A", idxA)):
                mp, _ = medias_pop(e[pred], ii, sent)
                for pop in POPS:
                    res[pred][pop][est].append(mp[pop])
            jj, nd = hajek_pop(e[pred], idx, sent, p16)
            for pop in POPS:
                res[pred][pop]["J"].append(jj[pop])
            if pred == PREDS[0]:
                info["n_descartados_p16_zero_no_teste"].append(nd)
        if (j + 1) % 20 == 0:
            log(f"{cid_q}: passo B {j+1}/200 t={time.time()-t0:.0f}s")
    p200 = cnt200.astype(np.float64) / 200.0
    for idx in idx_te_guardados:
        for pred in PREDS:
            jj, nd = hajek_pop(e[pred], idx, sent, p200)
            for pop in POPS:
                res[pred][pop]["J_p200"].append(jj[pop])
            if pred == PREDS[0]:
                info["n_descartados_p200_zero_no_teste"].append(nd)

    out = dict(celula=cid_q, tensor_path=S["tensor_path"], n_nos=int(n_nos),
               constante_ref_dB=S["constante_ref"], p_tx_eff_ref_dB=S["p_tx_eff_ref"],
               mu_U=mu_U, n_nos_validos=int((~sent).sum()),
               n_nos_com_p16_zero=n_zero_p16, n_nos_com_p200_zero=int((p200 == 0).sum()),
               validacao_H_vs_2_3=conf, por_sorteio=res, info_por_sorteio=info,
               seeds_split=seeds200, tempo_s=time.time() - t0)
    (OUT / f"_R2_parcial_{cidade}.json").write_text(json.dumps(out, indent=1))
    log(f"{cid_q}: parcial gravado, t={time.time()-t0:.0f}s")
    return cidade


def stats(x, mu):
    """Bias, Monte Carlo SE of the bias, bias in SE units, RMSE, variance (ddof=1) of the estimates x about mu.

    Undefined (None/NaN) draws are dropped and counted.
    """
    a =np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    n = len(a)
    if n < 2:
        return dict(n=n)
    sd = float(a.std(ddof=1))
    return dict(n=n, media_estimativas=float(a.mean()), vies=float(a.mean() - mu), ep_mc_vies=sd / math.sqrt(n),
                vies_em_ep=float((a.mean() - mu) / (sd / math.sqrt(n))) if sd > 0 else None,
                reqm=float(math.sqrt(np.mean((a - mu) ** 2))), variancia=float(a.var(ddof=1)), dp=sd,
                n_sorteios_indefinidos=int(len(x) - n))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    faltam = [c for c in CIDADES if not (OUT / f"_R2_parcial_{c}.json").exists()]
    if faltam:
        from multiprocessing import Pool
        with Pool(len(faltam)) as p:
            p.map(processar_celula, faltam)
    cel = {c: json.load(open(OUT / f"_R2_parcial_{c}.json")) for c in CIDADES}

    resumo_c = {}
    por_sorteio = {}
    for c in CIDADES:
        r = cel[c]
        cid_q = r["celula"]
        resumo_c[cid_q] = {}
        por_sorteio[cid_q] = dict(mu_U=r["mu_U"], por_sorteio=r["por_sorteio"], info_por_sorteio=r["info_por_sorteio"])
        for pred in PREDS:
            resumo_c[cid_q][pred] = {}
            for pop in POPS:
                mu = r["mu_U"][pred][pop]
                est = {k: stats(r["por_sorteio"][pred][pop][k], mu) for k in ESTS}
                vA = est["A"].get("variancia")
                deff = {"deff_H_sobre_A": est["H"]["variancia"] / vA if vA else None,
                        "deff_H0_sobre_A": est["H0"]["variancia"] / vA if vA else None,
                        "reqm_J_sobre_reqm_H": est["J"]["reqm"] / est["H"]["reqm"],
                        "reqm_J_p200_sobre_reqm_H": est["J_p200"]["reqm"] / est["H"]["reqm"]}
                resumo_c[cid_q][pred][pop] = dict(mu_U=mu, estimadores=est, razoes=deff)

    # Counts of cells on each side of the thresholds of the decision criterion (no interpretation).
    tally = {}
    for pred in PREDS:
        for pop in POPS:
            cs = [resumo_c[f"{c}_{QUAD}"][pred][pop] for c in CIDADES]
            tally[f"{pred}/{pop}"] = dict(
                celulas_deff_H_ge_10=sum(x["razoes"]["deff_H_sobre_A"] >= 10 for x in cs),
                celulas_deff_H_lt_2=sum(x["razoes"]["deff_H_sobre_A"] < 2 for x in cs),
                celulas_abs_vies_H_le_2EP=sum(abs(x["estimadores"]["H"]["vies_em_ep"]) <= 2 for x in cs),
                celulas_abs_vies_H_gt_2EP=sum(abs(x["estimadores"]["H"]["vies_em_ep"]) > 2 for x in cs),
                celulas_reqm_J_lt_metade_H=sum(x["razoes"]["reqm_J_sobre_reqm_H"] < 0.5 for x in cs),
                celulas_reqm_J_p200_lt_metade_H=sum(x["razoes"]["reqm_J_p200_sobre_reqm_H"] < 0.5 for x in cs))

    comum = dict(
        script=str(SCRIPT_PATH), script_sha256=sha256_file(SCRIPT_PATH),
        script_original_importado=str(ORIG), script_original_sha256=sha256_file(ORIG),
        criterio=str(CRIT), criterio_sha256=sha256_file(CRIT),
        fonte_seeds_split=str(SEEDS_JSON), fonte_seeds_sha256=sha256_file(SEEDS_JSON),
        sementes=dict(split_200="fase1/1.8 'seeds' (RandomState(20260926).randint(1e6,size=200)); ver seeds_split em cada celula",
                      split_p_i_e_validacao_100="RandomState(20260927).randint(1e6,size=100) (1.6/2.3)",
                      amostra_aleatoria="RandomState(20261001 + indice_do_sorteio).choice(n_nos, size=n_te_b2, replace=False)"),
        comando=" ".join(["/trabalho/ambientes/s33_amb_virtual/.venv/bin/python", str(SCRIPT_PATH)]),
        data_utc=datetime.now(timezone.utc).isoformat(),
        desvios_do_criterio=[
            "Os 200 seeds de fase1/1.8 (RandomState(20260926)) e os 100 de fase2/2.3 (RandomState(20260927)) nao compartilham nenhum seed; "
            "nao existem '100 sorteios comuns'. A validacao de (H) foi feita rodando (H) b=2 nos 100 seeds do 2.3 contra "
            "Err_sigma_por_sorteio_A de fase2/2.3b_estimando_por_sorteio.json (por sorteio) e c de fase2/2.3.",
            "p_i por no nao esta gravado em fase2/1.6_p_inclusao_por_no.json (so perfis); recalculado como frequencia de retencao nos "
            "mesmos 100 sorteios do 1.6 e conferido contra media_p_cond_te_geral do 1.6. Nos de teste com p_i=0 nessa frequencia sao descartados no Hajek "
            "(contagem em info_por_sorteio.n_descartados_p16_zero_no_teste).",
            "Estimador suplementar J_p200 (nao pedido): Hajek com p_i = frequencia nos proprios 200 sorteios (sem p_i=0 no teste); p dependente da amostra.",
            "Nos validos, sorteios sem no valido no teste (M=0) ficam indefinidos em H/H0/J (n em estatisticas < 200; n_sorteios_indefinidos). "
            "A e sempre definido.",
            "H0 tem tamanho de teste maior que H (sem aparar); A usa o tamanho de H (teste retido com b=2), como pedido."])
    (OUT / "R2_por_sorteio.json").write_text(json.dumps(dict(**comum, celulas=por_sorteio), indent=1))
    (OUT / "R2_resumo.json").write_text(json.dumps(dict(
        **comum, validacao_H_vs_2_3={cel[c]["celula"]: cel[c]["validacao_H_vs_2_3"] for c in CIDADES},
        n_nos_p_zero={cel[c]["celula"]: dict(p16=cel[c]["n_nos_com_p16_zero"], p200=cel[c]["n_nos_com_p200_zero"]) for c in CIDADES},
        resumo_por_celula=resumo_c, contagem_mecanica_limiares=tally), indent=1))
    log(f"gravado R2_por_sorteio.json e R2_resumo.json; total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
