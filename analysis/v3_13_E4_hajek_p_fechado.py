#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test E4: Hajek reweighting of the test error with the closed-form inclusion probability p_i.

Four Q1 cells (Bauru, Campinas, Lins, Sorocaba), 200 split draws of the 92/20/20 block partition
(g = 10 km, b = 2 km; N = 132 blocks, k_te = 20 test blocks). Primary definition (decision criterion
criteria/criterio_E4_hajek_p_fechado.json, written before the run):
  p_i = (k_te / N) * c(m_i), with c(0) = 1, c(1) = (k_te - 1)/(N - 1), c(3) = ratio of falling
  factorials (k_te - 1)_3 / (N - 1)_3 (closed forms, class-level reading "T_classe" of the
  closed-form-p record), and c(2) = empirical conditional frequency of class m = 2 over the 200 draws
  (no closed form exists for it);
  J_fechado = sum(e_i / p_i) / sum(1 / p_i) over the scored nodes (retained test nodes, b = 2) of a draw.
Alternative readings, all computed and labelled: L_no_a_no (closed form only where m = 0 or condition
(iii) holds at the node, row frequency elsewhere), C_todas (lower closed-form bound in every class),
freq_classe (empirical conditional class frequency in every class), emp2_incond (as primary, but c(2)
from the class mean of the unconditional frequency cnt200/200). The plain test mean H and the
Hajek estimate with p_i = cnt200/200 (J_p200) are recomputed and compared, by exact equality, with
the design-reference records.

Reuses by import, with digest check before and after the run: v3_12_R2_referencia_desenho.py,
v3_12_R4_p_fechado_vs_frequencia.py, v3_12_R3R4_motor.py and v3_1.6_2.3_estimando_formal.py.
Inputs: fase4/R4_p_por_classe.json, fase4/R2_resumo.json, fase4/R2_por_sorteio.json.
Outputs (fase5/): E4_resumo.json, E4_por_sorteio.json, E4_validacao.json; partials in fase5/_parcial_E4/.
Seeds: the 200 split seeds equal RandomState(20260926).randint(10**6, size=200) (asserted); the
supplementary paired bootstrap uses np.random.default_rng(20261003) with B = 4000 resamples.
Usage: python analysis/v3_13_E4_hajek_p_fechado.py             (full run, four cells in parallel, CPU)
       python analysis/v3_13_E4_hajek_p_fechado.py --smoke N   (one cell, N draws, scratch output only)
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPTS = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts")
V3 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
SCRIPT_PATH = Path(__file__).resolve()
CRIT = V3 / "criterios" / "criterio_E4_hajek_p_fechado.json"
R4_JSON = V3 / "fase4" / "R4_p_por_classe.json"
R2_RESUMO = V3 / "fase4" / "R2_resumo.json"
R2_POR_SORTEIO = V3 / "fase4" / "R2_por_sorteio.json"
OUT = V3 / "fase5"
PARC = OUT / "_parcial_E4"
SEED_BOOT = 20261003  # seed of the supplementary paired bootstrap
B_BOOT = 4000


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


# digests of the reused modules, taken before they are imported
REUSO = {
    "R2": SCRIPTS / "v3_12_R2_referencia_desenho.py",
    "R4": SCRIPTS / "v3_12_R4_p_fechado_vs_frequencia.py",
    "motor_R3R4": SCRIPTS / "v3_12_R3R4_motor.py",
    "s16_importado_por_R2_e_motor": SCRIPTS / "v3_1.6_2.3_estimando_formal.py",
}
SHA_ANTES = {k: sha256_file(v) for k, v in REUSO.items()}

sys.path.insert(0, str(SCRIPTS))
import v3_12_R2_referencia_desenho as R2  # noqa: E402
import v3_12_R4_p_fechado_vs_frequencia as R4  # noqa: E402

mo = R4.mo
CIDADES = R2.CIDADES
QUAD = R2.QUAD
PREDS = R2.PREDS
POPS = R2.POPS
VARIANTES = ("T_primaria", "L_no_a_no", "C_todas", "freq_classe", "emp2_incond")
EST_NOME = {"T_primaria": "J_fechado", "L_no_a_no": "J_fechado_L_no_a_no", "C_todas": "J_fechado_C_todas",
            "freq_classe": "J_freq_classe", "emp2_incond": "J_fechado_emp2_incond"}
ESTS = ("H", "J_p200") + tuple(EST_NOME[v] for v in VARIANTES)


def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def tabelas_c(cel_r4, kte, N, cnt200, m, n_nos):
    """Class factor c(m) for each variant, from the nodal classes of R4_p_por_classe.json.

    Returns the per-class tables, the (m, condition (iii) fails) table of L_no_a_no, and a record of
    where each value came from (closed form or empirical frequency)."""
    rows = {r["classe"]: r for r in cel_r4["classes_nodais"]}
    mmax = int(m.max())
    assert mmax <= 4
    lo = {k: R4.lo_hi(k, N, kte, 92)[0] for k in range(5)}  # lower closed-form bound per class; 92 training blocks
    freq = {}
    vf_T = {}
    for k in range(5):
        r = rows.get(f"m={k}|todas")
        if r is None:
            continue
        freq[k] = r["frequencia_media_nodal"]
        vf_T[k] = r["leituras"]["T_classe"]["valor_fechado"]
    assert set(freq) == set(range(mmax + 1)), (sorted(freq), mmax)
    cT = {k: (vf_T[k] if vf_T[k] is not None else freq[k]) for k in freq}
    c_origem = {k: ("fechado(T_classe)" if vf_T[k] is not None else "frequencia_empirica_condicional_200(R4)") for k in freq}
    p200 = cnt200.astype(np.float64) / 200.0
    inc = {k: float(p200[m == k].mean()) for k in freq}
    c_emp2 = {k: (cT[k] if vf_T[k] is not None else inc[k] / (kte / N)) for k in freq}
    c_C = {k: lo[k] for k in freq}
    cL = {}
    for k in freq:
        for fa in (0, 1):
            lab = f"m={k}|(iii) {'falha' if fa else 'vale'}"
            r = rows.get(lab)
            if r is None:
                continue
            vf = r["leituras"]["L_no_a_no"]["valor_fechado"]
            cL[(k, fa)] = vf if vf is not None else r["frequencia_media_nodal"]
    tabs = {"T_primaria": cT, "C_todas": c_C, "freq_classe": dict(freq), "emp2_incond": c_emp2}
    return tabs, cL, dict(c_T=cT, c_origem=c_origem, lo=lo, freq_condicional_200=freq,
                          freq_incondicional_200_media_classe=inc, c_L=({f"{k[0]}|falha={k[1]}": v for k, v in cL.items()}),
                          vf_T_R4=vf_T)


def processar_celula(args):
    """One cell: rebuild the 200 partitions, the nodal classes and p_i of every variant, and the
    estimators H, J_p200 and the J variants per draw; writes the partial record of the cell."""
    cidade, n_sorteios, saida_dir = args
    t0 = time.time()
    cid_q = f"{cidade}_{QUAD}"
    S = R2.preparar_celula(cidade)
    pos_km, gid, grupos, n_nos, sent, e = (S[k] for k in ("pos_km", "gid", "grupos", "n_nos", "sentinela", "e"))
    n_g = len(grupos)
    seeds = json.load(open(R2.SEEDS_JSON))["seeds"]
    assert len(seeds) == 200 and seeds == np.random.RandomState(20260926).randint(10**6, size=200).tolist()
    seeds = seeds[:n_sorteios]
    mu_U = {pred: {"validos": float(e[pred][~sent].mean()), "todos": float(e[pred].mean())} for pred in PREDS}
    # nodal class m(i) and condition (iii) flag, checked against the stored intermediate arrays
    own, m, falha, mg, m_max = mo.classes_nodais(pos_km, gid, grupos)
    A, meta = mo.carregar_intermediario(cidade)
    conf_classes = dict(m_igual_intermediario=bool(np.array_equal(A["m"], m)),
                        falha_igual_intermediario=bool(np.array_equal(A["falha"], falha)))
    log(f"{cid_q}: classes recomputadas; conf={conf_classes}; t={time.time()-t0:.0f}s")
    # retained test indices (b = 2) of each draw and the retention count cnt200 per node
    idx_list = []
    cnt200 = np.zeros(n_nos, dtype=np.int32)
    info = {"n_te_b2": [], "n_te_b2_validos": []}
    kte_ref = None
    for j, s in enumerate(seeds):
        m_te0, m_va0, m_te, m_va, kte, ktr = R2.O.split_uma_vez(pos_km, gid, grupos, n_g, s)
        kte_ref = kte
        idx = np.flatnonzero(m_te).astype(np.int32)
        cnt200 += m_te.astype(np.int32)
        idx_list.append(idx)
        info["n_te_b2"].append(int(len(idx)))
        info["n_te_b2_validos"].append(int((~sent[idx]).sum()))
        del m_te0, m_va0, m_te, m_va
        if (j + 1) % 20 == 0:
            log(f"{cid_q}: particao {j+1}/{len(seeds)} t={time.time()-t0:.0f}s")
    N = n_g
    assert (N, kte_ref) == (132, 20), (N, kte_ref)
    conf_classes["cnt200_igual_cnt_ret_intermediario"] = bool(np.array_equal(cnt200.astype(np.int16), A["cnt_ret"])) if n_sorteios == 200 else None
    cel_r4 = json.load(open(R4_JSON))["por_celula"][cid_q]
    tabs, cL, c_info = tabelas_c(cel_r4, kte_ref, N, cnt200, m, n_nos)
    # p_i = (k_te / N) * c(m_i) for each variant
    fator = kte_ref / N
    P = {}
    cm = m.astype(np.int64)
    for v in ("T_primaria", "C_todas", "freq_classe", "emp2_incond"):
        tab = np.full(5, np.nan)
        for k, val in tabs[v].items():
            tab[k] = val
        P[v] = fator * tab[cm]
    tabL = np.full((5, 2), np.nan)
    for (k, fa), val in cL.items():
        tabL[k, fa] = val
    P["L_no_a_no"] = fator * tabL[cm, falha.astype(np.int64)]
    for v in VARIANTES:
        assert not np.isnan(P[v]).any(), v
        assert (P[v] > 0).all(), v
    P["p200"] = cnt200.astype(np.float64) / 200.0
    # full run only: conditional class frequencies recomputed with freq_linhas and compared with the R4 record
    conf_freq = None
    if n_sorteios == 200:
        te = A["te_block"].astype(np.float64)
        agg = np.zeros((int(m.max()) + 1, mo.NCG_N))
        for k in range(int(m.max()) + 1):
            agg[k, [2 * k, 2 * k + 1]] = 1.0
        f_rec = R4.freq_linhas(A["R_n"], te, A["n_bc_n"].astype(np.float64), agg)
        conf_freq = {f"m={k}": dict(recomputada=float(f_rec[k]), json_R4=c_info["freq_condicional_200"][k],
                                    dif_abs=abs(float(f_rec[k]) - c_info["freq_condicional_200"][k])) for k in range(len(f_rec))}
    # checks (3) mean of p_i / (k_te/N) per class, and (4) sum of p_i versus mean number of retained test nodes
    val34 = {}
    n_te_medio = float(np.mean(info["n_te_b2"]))
    n_te_val_medio = float(np.mean(info["n_te_b2_validos"]))
    for v in VARIANTES + ("p200",):
        p = P[v]
        por_classe = {}
        for k in range(int(m.max()) + 1):
            mk = m == k
            por_classe[f"m={k}"] = dict(n_nos=int(mk.sum()), media_p=float(p[mk].mean()), media_p_sobre_kte_N=float(p[mk].mean() / fator))
        val34[v] = dict(p_min=float(p.min()), p_max=float(p.max()), n_p_zero=int((p == 0).sum()), por_classe=por_classe,
                        soma_p=float(p.sum()), soma_p_validos=float(p[~sent].sum()),
                        n_te_medio_por_sorteio=n_te_medio, n_te_validos_medio_por_sorteio=n_te_val_medio,
                        razao_soma_p_sobre_n_te_medio=float(p.sum() / n_te_medio),
                        razao_soma_p_validos_sobre_n_te_validos_medio=float(p[~sent].sum() / n_te_val_medio))
    # estimators per draw
    res = {pred: {pop: {est: [] for est in ESTS} for pop in POPS} for pred in PREDS}
    for j, idx in enumerate(idx_list):
        for pred in PREDS:
            mp, _ = R2.medias_pop(e[pred], idx, sent)
            for pop in POPS:
                res[pred][pop]["H"].append(mp[pop])
            jj, _ = R2.hajek_pop(e[pred], idx, sent, P["p200"])
            for pop in POPS:
                res[pred][pop]["J_p200"].append(jj[pop])
            for v in VARIANTES:
                jj, nd = R2.hajek_pop(e[pred], idx, sent, P[v])
                assert nd == 0
                for pop in POPS:
                    res[pred][pop][EST_NOME[v]].append(jj[pop])
        if (j + 1) % 40 == 0:
            log(f"{cid_q}: estimadores {j+1}/{len(idx_list)} t={time.time()-t0:.0f}s")
    out = dict(celula=cid_q, n_sorteios=len(seeds), seeds_split=seeds, n_nos=int(n_nos), n_nos_validos=int((~sent).sum()),
               mu_U=mu_U, constante_ref_dB=S["constante_ref"], p_tx_eff_ref_dB=S["p_tx_eff_ref"],
               N=N, k_te=kte_ref, fator_kte_sobre_N=fator, c_info=c_info, validacao_classes=conf_classes,
               validacao_freq_condicional_R4=conf_freq, validacao_p=val34, por_sorteio=res, info_por_sorteio=info,
               tempo_s=time.time() - t0)
    saida = saida_dir / f"{cidade}.json"
    saida.write_text(json.dumps(out, indent=1))
    log(f"{cid_q}: parcial gravado {saida} t={time.time()-t0:.0f}s")
    return cidade


def igual_exato(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(a.shape == b.shape and np.array_equal(a, b, equal_nan=True))


def razao_boot(jv, hv, mu, rng):
    """Supplementary paired bootstrap over draws: standard error and 95% interval of the ratio
    RMSE(J)/RMSE(H), and standard errors of RMSE(J) and RMSE(H); NaN draws dropped pairwise."""
    j = np.asarray(jv, float)
    h = np.asarray(hv, float)
    ok = ~(np.isnan(j) | np.isnan(h))
    j, h = j[ok], h[ok]
    n = len(j)
    ix = rng.integers(0, n, size=(B_BOOT, n))
    rj = np.sqrt(((j[ix] - mu) ** 2).mean(1))
    rh = np.sqrt(((h[ix] - mu) ** 2).mean(1))
    rz = rj / rh
    return dict(n_pareado=int(n), ep_boot_razao=float(rz.std(ddof=1)), ic95_boot_razao=[float(np.percentile(rz, 2.5)), float(np.percentile(rz, 97.5))],
                ep_boot_reqm_J=float(rj.std(ddof=1)), ep_boot_reqm_H=float(rh.std(ddof=1)))


def main():
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])
    t0 = time.time()
    if smoke:
        saida_dir = Path("internal/scratch/E4_smoke")
        saida_dir.mkdir(parents=True, exist_ok=True)
        processar_celula(("bauru", smoke, saida_dir))
        r = json.load(open(saida_dir / "bauru.json"))
        ref = json.load(open(R2_POR_SORTEIO))["celulas"]["bauru_Q1"]["por_sorteio"]
        for pred in PREDS:
            for pop in POPS:
                for est, estR2 in (("H", "H"), ("J_p200", "J_p200")):
                    print("smoke", pred, pop, est, "== R2 (prefixo):", igual_exato(r["por_sorteio"][pred][pop][est], ref[pred][pop][estR2][:smoke]))
        for pred in PREDS:
            for pop in POPS:
                mu = r["mu_U"][pred][pop]
                print("smoke", pred, pop, "reqm H, J_fechado:", R2.stats(r["por_sorteio"][pred][pop]["H"], mu).get("reqm"),
                      R2.stats(r["por_sorteio"][pred][pop]["J_fechado"], mu).get("reqm"))
        print(json.dumps(r["validacao_p"]["T_primaria"], indent=1))
        print(json.dumps(r["c_info"], indent=1))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    PARC.mkdir(parents=True, exist_ok=True)
    faltam = [c for c in CIDADES if not (PARC / f"{c}.json").exists()]
    if faltam:
        from multiprocessing import Pool
        with Pool(len(faltam)) as p:
            p.map(processar_celula, [(c, 200, PARC) for c in faltam])
    cel = {c: json.load(open(PARC / f"{c}.json")) for c in CIDADES}
    ref_resumo = json.load(open(R2_RESUMO))
    ref_sorteio = json.load(open(R2_POR_SORTEIO))
    rng = np.random.default_rng(SEED_BOOT)

    resumo, por_sorteio, valid_1, valid_2 = {}, {}, {}, {}
    for c in CIDADES:
        r = cel[c]
        cq = r["celula"]
        resumo[cq] = {}
        por_sorteio[cq] = dict(mu_U=r["mu_U"], por_sorteio=r["por_sorteio"], info_por_sorteio=r["info_por_sorteio"])
        valid_1[cq], valid_2[cq] = {}, {}
        for pred in PREDS:
            resumo[cq][pred] = {}
            valid_1[cq][pred], valid_2[cq][pred] = {}, {}
            for pop in POPS:
                mu = r["mu_U"][pred][pop]
                est = {k: R2.stats(r["por_sorteio"][pred][pop][k], mu) for k in ESTS}
                razoes = {}
                for v in VARIANTES:
                    nome = EST_NOME[v]
                    rz = est[nome]["reqm"] / est["H"]["reqm"]
                    razoes[nome] = dict(razao_reqm_sobre_H=rz, reqm=est[nome]["reqm"], reqm_H=est["H"]["reqm"],
                                        vies=est[nome]["vies"], ep_mc_vies=est[nome]["ep_mc_vies"],
                                        vies_em_ep=est[nome]["vies_em_ep"], **razao_boot(r["por_sorteio"][pred][pop][nome], r["por_sorteio"][pred][pop]["H"], mu, rng))
                rz200 = est["J_p200"]["reqm"] / est["H"]["reqm"]
                razoes["J_p200"] = dict(razao_reqm_sobre_H=rz200, reqm=est["J_p200"]["reqm"], reqm_H=est["H"]["reqm"],
                                        vies=est["J_p200"]["vies"], ep_mc_vies=est["J_p200"]["ep_mc_vies"])
                resumo[cq][pred][pop] = dict(mu_U=mu, estimadores=est, razoes=razoes)
                # checks (1) and (2): exact equality with the design-reference summary and per-draw records
                rr = ref_resumo["resumo_por_celula"][cq][pred][pop]["estimadores"]
                rs = ref_sorteio["celulas"][cq]["por_sorteio"][pred][pop]
                for chave_v, valid, estR2 in (("H", valid_1, "H"), ("J_p200", valid_2, "J_p200")):
                    valid[cq][pred][pop] = dict(
                        reqm_recomputado=est[chave_v]["reqm"], reqm_R2=rr[estR2]["reqm"], reqm_igual=est[chave_v]["reqm"] == rr[estR2]["reqm"],
                        vies_recomputado=est[chave_v]["vies"], vies_R2=rr[estR2]["vies"], vies_igual=est[chave_v]["vies"] == rr[estR2]["vies"],
                        mu_U_igual=mu == ref_resumo["resumo_por_celula"][cq][pred][pop]["mu_U"],
                        por_sorteio_igual=igual_exato(r["por_sorteio"][pred][pop][chave_v], rs[estR2]),
                        n_sorteios_comparados=len(rs[estR2]))

    # mechanical counts for the thresholds of the decision criterion, per variant, predictor and population
    tally = {}
    for v in VARIANTES:
        nome = EST_NOME[v]
        for pred in PREDS:
            for pop in POPS:
                rz = {f"{c}_{QUAD}": resumo[f"{c}_{QUAD}"][pred][pop]["razoes"][nome]["razao_reqm_sobre_H"] for c in CIDADES}
                vals = list(rz.values())
                tally[f"{nome}/{pred}/{pop}"] = dict(
                    razao_por_celula=rz,
                    celulas_razao_lt_0_9=sum(x < 0.9 for x in vals),
                    celulas_razao_em_0_9_a_1_1=sum(0.9 <= x <= 1.1 for x in vals),
                    celulas_razao_gt_1_1=sum(x > 1.1 for x in vals),
                    celulas_razao_lt_1=sum(x < 1.0 for x in vals),
                    variante=v, primaria=(v == "T_primaria"))

    val3, val4 = {}, {}
    for c in CIDADES:
        r = cel[c]
        cq = r["celula"]
        fator = r["fator_kte_sobre_N"]
        vp = r["validacao_p"]["T_primaria"]
        vfT = r["c_info"]["vf_T_R4"]
        freq = r["c_info"]["freq_condicional_200"]
        det = {}
        ok = True
        for k, d in vp["por_classe"].items():
            kk = int(k[2:])
            alvo = vfT[str(kk)] if str(kk) in vfT else vfT[kk]
            alvo_origem = "valor_fechado(R4)"
            if alvo is None:
                alvo = freq[str(kk)] if str(kk) in freq else freq[kk]
                alvo_origem = "frequencia_media_nodal(R4) (m=2: sem forma fechada)"
            dif = abs(d["media_p_sobre_kte_N"] - alvo)
            det[k] = dict(media_c_do_p_i=d["media_p_sobre_kte_N"], alvo=alvo, alvo_origem=alvo_origem, dif_abs=dif, n_nos=d["n_nos"])
            ok &= dif < 1e-12
        val3[cq] = dict(por_classe=det, igual_a_1e12=bool(ok), p_min=vp["p_min"], n_p_zero=vp["n_p_zero"], nenhum_p_zero=vp["n_p_zero"] == 0)
        val4[cq] = {v: dict(soma_p=r["validacao_p"][v]["soma_p"], n_te_medio_por_sorteio=r["validacao_p"][v]["n_te_medio_por_sorteio"],
                            razao=r["validacao_p"][v]["razao_soma_p_sobre_n_te_medio"],
                            soma_p_validos=r["validacao_p"][v]["soma_p_validos"], n_te_validos_medio=r["validacao_p"][v]["n_te_validos_medio_por_sorteio"],
                            razao_validos=r["validacao_p"][v]["razao_soma_p_validos_sobre_n_te_validos_medio"])
                    for v in VARIANTES + ("p200",)}

    sha_depois = {k: sha256_file(v) for k, v in REUSO.items()}
    sha_ref = dict(
        R2=ref_resumo["script_sha256"], s16_importado_por_R2_e_motor=ref_resumo["script_original_sha256"])
    r4j = json.load(open(R4_JSON))
    for k, v in r4j["scripts_sha256"].items():
        if k.endswith("v3_12_R4_p_fechado_vs_frequencia.py"):
            sha_ref["R4"] = v
        elif k.endswith("v3_12_R3R4_motor.py"):
            sha_ref["motor_R3R4"] = v
    sha_conf = {k: dict(arquivo=str(REUSO[k]), sha256_antes_do_import=SHA_ANTES[k], sha256_depois=sha_depois[k],
                        sha256_gravado_na_saida_anterior=sha_ref.get(k),
                        igual_antes_depois=SHA_ANTES[k] == sha_depois[k],
                        igual_ao_gravado_na_saida_anterior=(SHA_ANTES[k] == sha_ref.get(k)) if sha_ref.get(k) else None) for k in REUSO}

    todos_1 = all(x["reqm_igual"] and x["vies_igual"] and x["por_sorteio_igual"] and x["mu_U_igual"]
                  for cq in valid_1 for pred in valid_1[cq] for x in valid_1[cq][pred].values())
    todos_2 = all(x["reqm_igual"] and x["vies_igual"] and x["por_sorteio_igual"] and x["mu_U_igual"]
                  for cq in valid_2 for pred in valid_2[cq] for x in valid_2[cq][pred].values())
    todos_3 = all(v["igual_a_1e12"] and v["nenhum_p_zero"] for v in val3.values())
    validacao = dict(
        V1_H_recomputado_igual_R2_resumo=dict(passou=bool(todos_1), detalhe=valid_1,
                                              regra="reqm e vies == (igualdade exata de float) e vetor por sorteio == R2_por_sorteio (NaN==NaN)"),
        V2_J_p200_igual_R2=dict(passou=bool(todos_2), detalhe=valid_2, regra="idem para J_p200"),
        V3_p_fechado_media_por_classe_igual_valor_fechado_R4_e_nenhum_p_zero=dict(passou=bool(todos_3), detalhe=val3,
            regra="media de p_i/(k_te/N) por classe m == valor_fechado (T_classe) do R4 em m=0,1,3 e == frequencia_media_nodal em m=2; tolerancia 1e-12; p_i>0 em todo no"),
        V4_soma_p_vs_nos_pontuados=dict(detalhe=val4,
            regra="soma de p_i sobre os nos do dominio vs media (200 sorteios) de nos de teste retidos b=2 (n_te_b2); identico para validos. Para p200 e identidade exata. Nenhum limite pass/fail fixado: razao e relatada."),
        classes_vs_intermediarios={c: cel[c]["validacao_classes"] for c in CIDADES},
        freq_condicional_R4_recomputada={c: cel[c]["validacao_freq_condicional_R4"] for c in CIDADES},
        sha256_reuso=sha_conf,
        sha256_reuso_todos_estaveis=all(x["igual_antes_depois"] for x in sha_conf.values()),
        sha256_reuso_iguais_aos_gravados=all(x["igual_ao_gravado_na_saida_anterior"] in (True, None) for x in sha_conf.values()))
    comum = dict(
        id="E4_hajek_p_fechado", criterio=str(CRIT), criterio_sha256=sha256_file(CRIT),
        protocolo="decision criterion written before the run; criterio fixado_em 2026-10-03T21:39:21-03:00",
        script=str(SCRIPT_PATH), script_sha256=sha256_file(SCRIPT_PATH),
        comando=" ".join(["/trabalho/ambientes/s33_amb_virtual/.venv/bin/python", str(SCRIPT_PATH)]),
        data_utc=datetime.now(timezone.utc).isoformat(),
        sementes=dict(split_200="fase1/1.8 'seeds' == RandomState(20260926).randint(1e6, size=200) (asserted); em cada parcial: seeds_split",
                      bootstrap_suplementar=f"np.random.default_rng({SEED_BOOT}), B={B_BOOT}, pareado sobre sorteios",
                      amostra_aleatoria_A="nao usada (A nao faz parte do E4)"),
        fontes=dict(R4_p_por_classe=str(R4_JSON), R4_sha256=sha256_file(R4_JSON), R2_resumo=str(R2_RESUMO), R2_resumo_sha256=sha256_file(R2_RESUMO),
                    R2_por_sorteio=str(R2_POR_SORTEIO), R2_por_sorteio_sha256=sha256_file(R2_POR_SORTEIO)),
        definicao_primaria="p_i=(k_te/N)*c(m_i); c(0)=1, c(1)=(k_te-1)/(N-1), c(3)=forma fechada (R4 T_classe); c(2)=frequencia empirica condicional da classe m=2 em 200 sorteios (R4); J_fechado=sum(e/p)/sum(1/p) sobre os nos pontuados do sorteio. Estimador primario no JSON: J_fechado.",
        leituras_alternativas={
            "L_no_a_no": "forma fechada so onde m=0 ou (iii) vale no no; demais: frequencia empirica da linha (m,(iii)) do R4 (estimador J_fechado_L_no_a_no)",
            "C_todas": "c(m)=lo(m) em todas as classes, inclusive m=2 (J_fechado_C_todas)",
            "freq_classe": "c(m)=frequencia empirica condicional da classe em todas as classes (J_freq_classe)",
            "emp2_incond": "como primaria, mas c(2)=media de classe de cnt200/200 dividida por k_te/N (J_fechado_emp2_incond)"},
        desvios_e_ambiguidades=[
            "O criterio diz 'forma fechada para m=0,1,3': implementado como a leitura T_classe do R4 (m=1 recebe a forma fechada tambem nos nos com condicao (iii) falha; sao poucos: ver n_nos_falha_iii_por_m no R4). A leitura L_no_a_no (no a no) e as demais estao como estimadores marcados.",
            "A forma fechada de m=3 (0,002646) difere da frequencia empirica da classe (0,001754) nas 4 celulas (R4); p_i fechado usa a forma fechada como o criterio manda, a frequencia aparece em J_freq_classe.",
            "'Nos pontuados' lido como nos de teste retidos com b=2 do sorteio (mesmo conjunto idx de H e J do R2). Em 'validos', sorteios sem no valido ficam indefinidos (NaN) em H e J (n<200, n_sorteios_indefinidos).",
            "Soma de p_i (validacao 4) e sobre todos os nos do dominio; compara-se a media de n_te_b2 e, para validos, de n_te_b2_validos. Nenhum limiar de aprovacao foi fixado no criterio: valor reportado.",
            "EP de Monte Carlo do vies = desvio padrao dos 200 estimadores / sqrt(n) (R2.stats). Bootstrap pareado da razao e suplementar (nao pedido)."],
        celulas_resumo=resumo, contagem_mecanica_limiares=tally)
    (OUT / "E4_resumo.json").write_text(json.dumps(comum, indent=1))
    (OUT / "E4_por_sorteio.json").write_text(json.dumps(dict(id="E4_hajek_p_fechado", script_sha256=comum["script_sha256"],
                                                           seeds_split={cel[c]["celula"]: cel[c]["seeds_split"] for c in CIDADES},
                                                           celulas=por_sorteio), indent=1))
    (OUT / "E4_validacao.json").write_text(json.dumps(dict(id="E4_hajek_p_fechado", script_sha256=comum["script_sha256"],
                                                          data_utc=comum["data_utc"], comando=comum["comando"],
                                                          c_info_por_celula={cel[c]["celula"]: cel[c]["c_info"] for c in CIDADES},
                                                          **validacao), indent=1))
    log(f"gravado E4_resumo.json, E4_por_sorteio.json, E4_validacao.json; total {time.time()-t0:.0f}s")
    log(f"V1={todos_1} V2={todos_2} V3={todos_3} sha_estaveis={validacao['sha256_reuso_todos_estaveis']} sha_iguais_gravados={validacao['sha256_reuso_iguais_aos_gravados']}")


if __name__ == "__main__":
    main()
