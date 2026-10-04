#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check R3: the design term of the bias corollary, without Monte Carlo over the error.

For the constant predictor and FSPL with calibration (b), in the four Q1 cells
(g = 10 km, b = 2 km), it evaluates

    T = sum_i p_i e_i / sum_i p_i - mu_U,   U in {all nodes, valid nodes},

with p_i = (k_te / N) p_{i|te} and p_{i|te} from the closed form of the inclusion
proposition: lower bound lo(m) = (k_te - 1)^(m) / (N - 1)^(m), upper bound
hi(m) = (N - 1 - k_tr)^(m) / (N - 1)^(m) (falling factorial powers), with the nodal class
m(i) from the engine. Where the proposition is exact, p_i = (k_te / N) lo(m); where it only
gives bounds, the script reports the interval [T_inf, T_sup], the exact extremes of the
ratio with each p_i in its box [lo, hi] (linear-fractional programme solved by Dinkelbach
iterations). The readings of which nodes are exact are those of the R4 script (L_no_a_no,
primary and fixed beforehand; T_classe; C_todas; U_so_m0), plus the geometric class m_geom
as sensitivity.

Comparators: the term with the 200-draw retention frequency phat_i (equal to
sum_s S_s / sum_s M_s, bootstrap SE over draws); the Monte Carlo values of record 2.3
(b = mean per-draw test MAE, which also contains the ratio term; c = ratio of expectations,
rebuilt from the per-draw M and Err of record 2.3b, bootstrap SE over the 100 draws); and the
same Monte Carlo quantities over the 200 draws of the engine. mu_U is checked against 'a' of
record 2.3 (rounded to 3 decimals). Bootstrap: 10000 replicates, numpy default_rng(20261002).

Inputs: engine intermediates fase4/_v3_12_R3R4_intermediarios/, fase2/2.3_estimando_ht_baselines.json,
fase2/2.3b_estimando_por_sorteio.json, criterios/criterio_R3_termo_desenho.json.
Outputs: fase4/R3_termo_desenho.json (source of the design-term table) and per-cell
fase4/_parcial_R3_<city>.json.

Usage: python v3_12_R3_termo_desenho.py   (after v3_12_R3R4_motor.py)
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPTS = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts")
sys.path.insert(0, str(SCRIPTS))
import v3_12_R3R4_motor as mo  # noqa: E402
from v3_12_R4_p_fechado_vs_frequencia import lo_hi, nao_exato  # noqa: E402  (same exactness readings as R4)

RAIZ = mo.RAIZ
FASE4 = RAIZ / "fase4"
CRIT = RAIZ / "criterios" / "criterio_R3_termo_desenho.json"
F23 = RAIZ / "fase2" / "2.3_estimando_ht_baselines.json"
F23B = RAIZ / "fase2" / "2.3b_estimando_por_sorteio.json"
OUT = FASE4 / "R3_termo_desenho.json"
B_BOOT = 10000
SEED_BOOT = 20261002
LEITURAS_NODAL = ("L_no_a_no", "T_classe", "C_todas", "U_so_m0")
LEITURAS_GEOM = ("T_classe", "C_todas", "U_so_m0")
LEITURA_PRIMARIA = "L_no_a_no"
SIG = mo.PREDS
POPS = mo.POPS


def theta_extremo(e, plo, phi, nonex, sentido):
    """Exact extreme of sum(p e) / sum(p) with plo <= p <= phi on the nodes 'nonex' (p = plo elsewhere).

    Dinkelbach iteration: for the maximum (sentido > 0) p = phi where e > theta, else plo;
    reversed for the minimum. Returns (theta, iterations).
    """
    th =float((plo * e).sum() / plo.sum())
    for it in range(200):
        sel = nonex & ((e > th) if sentido > 0 else (e < th))
        p = np.where(sel, phi, plo)
        th2 = float((p * e).sum() / p.sum())
        if abs(th2 - th) <= 1e-13 * max(1.0, abs(th)):
            return th2, it + 1
        th = th2
    return th, 200


def tabela_p(N, kte, ktr, mmax=4):
    """Unconditional p_i bounds by class m = 0..mmax: (k_te / N) * lo(m) and (k_te / N) * hi(m)."""
    lo =np.array([lo_hi(m, N, kte, ktr)[0] for m in range(mmax + 1)]) * kte / N
    hi = np.array([lo_hi(m, N, kte, ktr)[1] for m in range(mmax + 1)]) * kte / N
    return lo, hi


def dist_ate_intervalo(v, a, b, ep):
    """Signed distance from v to the interval [a, b] in units of ep (0 inside)."""
    if a - 1e-12 <= v <= b + 1e-12:
        return 0.0
    d = (v - a) if v < a else (v - b)
    return float(d / ep) if ep > 0 else float("inf")


def bootstrap_razao(S, M, rng, reps=B_BOOT):
    """Bootstrap replicates of the ratio of sums sum(S) / sum(M), resampling draws with replacement."""
    n = len(S)
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=reps).astype(np.float64)
    num, den = W @ S, W @ M
    return num / den


def mc_2_3(d23_cel, d23b_cel, mu_full, pred, pop, rng):
    """Monte Carlo comparators from records 2.3/2.3b for one cell, predictor and population.

    b = mean per-draw test MAE over draws with M > 0 (SE = sd / sqrt(n)); c = ratio of sums
    rebuilt from per-draw Err and M; joint bootstrap over the 100 draws for b, c and b - c.
    """
    x = d23_cel[pred][pop]
    xb = d23b_cel[pred][pop]
    assert abs(x["b_media_sorteios_mae_teste_dB"] - xb["b_media_sorteios_mae_teste_dB"]) < 1e-9
    Err = np.asarray(xb["Err_sigma_por_sorteio_A"], dtype=np.float64)
    Mall = np.asarray(xb["M_sigma_todos_100"], dtype=np.float64)
    MA = Mall[Mall > 0]
    assert Err.size == MA.size == x["n_sorteios_b"]
    S_A = Err * MA  # per-draw sum of e_i (Err_sigma = S / M)
    S_all = np.zeros_like(Mall); S_all[Mall > 0] = S_A
    b = float(Err.mean())
    ep_b = float(Err.std(ddof=1) / math.sqrt(Err.size))
    c_rec = float(S_all.sum() / Mall.sum())
    n = Mall.size
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=B_BOOT).astype(np.float64)
    isA = (Mall > 0).astype(np.float64)
    Err_full = np.zeros_like(Mall); Err_full[Mall > 0] = Err
    b_star = (W @ Err_full) / (W @ isA)
    c_star = (W @ S_all) / (W @ Mall)
    return dict(
        a_json_dB=x["a_mae_dominio_dB"], b_json_dB=x["b_media_sorteios_mae_teste_dB"],
        c_json_dB=x["c_razao_esperancas_p_te_dB"], b_menos_c_json_dB=x["diferenca_b_menos_c_dB"],
        n_sorteios_b=int(Err.size), n_sorteios_M0=int((Mall == 0).sum()),
        b_recalculado_dB=b, ep_b_analitico_dp_sobre_raiz_n=ep_b, ep_b_bootstrap=float(b_star.std(ddof=1)),
        dp_Err_sigma_entre_sorteios_dB=float(Err.std(ddof=1)),
        b_menos_mu_dB=b - mu_full, c_reconstruido_dB=c_rec,
        c_reconstruido_confere_json_a_5e_4=bool(abs(c_rec - x["c_razao_esperancas_p_te_dB"]) < 5e-4),
        c_menos_mu_dB=c_rec - mu_full, ep_c_bootstrap_100=float(c_star.std(ddof=1)),
        termo_razao_b_menos_c_dB=b - c_rec, ep_razao_bootstrap_100=float((b_star - c_star).std(ddof=1)),
        b_menos_mu_em_EP=(b - mu_full) / ep_b,
    )


def processar(cidade: str) -> dict:
    """Closed-form design-term intervals and all comparators for one cell."""
    A, meta = mo.carregar_intermediario(cidade)
    N, kte, ktr = meta["n_blocos"], meta["k_te"], meta["k_tr"]
    assert (N, kte, ktr) == (132, 20, 92)
    pos_km, gid, grupos = mo.carregar_grade(cidade)
    e_por, sentinela, _, _, _ = mo.calibrar_e(cidade, pos_km, gid, grupos)
    del pos_km, gid, grupos
    d23 = json.load(open(F23))["por_celula"][meta["celula"]]
    d23b = json.load(open(F23B))["por_celula"][meta["celula"]]
    m = A["m"].astype(np.int64)
    mg = np.minimum(A["mg"], 4).astype(np.int64)
    falha = A["falha"]
    cnt = A["cnt_ret"].astype(np.float64)
    plo_t, phi_t = tabela_p(N, kte, ktr)
    rng = np.random.default_rng(SEED_BOOT)
    n_s = int(A["te_block"].shape[0])
    res_cel = {}
    for pred in SIG:
        for pop in POPS:
            sel = np.ones(m.size, dtype=bool) if pop == "todos" else ~sentinela
            e = e_por[pred][sel]
            mu = float(e.mean())
            assert abs(mu - meta["mu_dB"][f"{pred}|{pop}"]) < 1e-9
            mu23 = d23[pred][pop]["a_mae_dominio_dB"]
            assert abs(round(mu, 3) - mu23) < 1.5e-3, (cidade, pred, pop, mu, mu23)
            mm, mgg, ff = m[sel], mg[sel], falha[sel]
            plo, phi = plo_t[mm], phi_t[mm]
            fechado = {"m_nodal": {}, "m_geom": {}}
            for defn, mv, leis in (("m_nodal", mm, LEITURAS_NODAL), ("m_geom", mgg, LEITURAS_GEOM)):
                for lei in leis:
                    if defn == "m_geom":
                        plo_g, phi_g = plo_t[mv], phi_t[mv]
                        nonex = np.array([nao_exato(lei, k, False) for k in range(5)])[mv]
                        lo_p, hi_p = plo_g, phi_g
                    else:
                        lo_p, hi_p = plo, phi
                        tab = np.array([[nao_exato(lei, k, bool(fl)) for fl in (0, 1)] for k in range(5)])
                        nonex = tab[mv, ff.astype(np.int64)]
                    th_lo, it1 = theta_extremo(e, lo_p, hi_p, nonex, -1)
                    th_hi, it2 = theta_extremo(e, lo_p, hi_p, nonex, +1)
                    th_pt = float((lo_p * e).sum() / lo_p.sum())  # every p at its lower bound
                    fechado[defn][lei] = dict(
                        n_nos_so_cotas=int(nonex.sum()), fracao_nos_so_cotas=float(nonex.mean()),
                        theta_inf=th_lo, theta_sup=th_hi, T_inf_dB=th_lo - mu, T_sup_dB=th_hi - mu,
                        largura_dB=th_hi - th_lo, ponto_se_exato_em_todos_dB=(th_pt - mu) if not nonex.any() else None,
                        T_com_todos_p_na_cota_inferior_dB=th_pt - mu, iteracoes_dinkelbach=[it1, it2])
            th_all_hi = float((phi * e).sum() / phi.sum())
            extremos = dict(T_todo_p_na_cota_inferior_nodal_dB=float((plo * e).sum() / plo.sum()) - mu,
                            T_todo_p_na_cota_superior_nodal_dB=th_all_hi - mu)
            # Term with the 200-draw retention frequency.
            phat = cnt[sel] / n_s
            th_f = float((phat * e).sum() / phat.sum())
            Sv, Mv = A[f"S__{pred}|{pop}"], A[f"M_{pop}"].astype(np.float64)
            assert abs(Sv.sum() / Mv.sum() - th_f) < 1e-8 * max(1, abs(th_f))
            star = bootstrap_razao(Sv, Mv, rng)
            freq200 = dict(theta=th_f, T_freq200_dB=th_f - mu, ep_bootstrap_200=float(star.std(ddof=1)),
                           n_sorteios=n_s)
            # Monte Carlo over the same 200 draws, without p.
            okA = Mv > 0
            ErrS = Sv[okA] / Mv[okA]
            mc200 = dict(b200_dB=float(ErrS.mean()), n_sorteios_A=int(okA.sum()),
                         ep_b200_analitico=float(ErrS.std(ddof=1) / math.sqrt(okA.sum())),
                         b200_menos_mu_dB=float(ErrS.mean()) - mu,
                         dp_Err_sigma_entre_sorteios_dB=float(ErrS.std(ddof=1)),
                         termo_razao_b200_menos_c200_dB=float(ErrS.mean()) - th_f)
            mc = mc_2_3(d23, d23b, mu, pred, pop, rng)
            # Distance of each comparator to the closed-form interval, in units of the comparator's SE.
            dist = {}
            for defn in ("m_nodal", "m_geom"):
                for lei, fz in fechado[defn].items():
                    a_, b_ = fz["T_inf_dB"], fz["T_sup_dB"]
                    dist[f"{defn}|{lei}"] = dict(
                        vs_b_menos_mu_2_3_MC=dict(valor=mc["b_menos_mu_dB"], ep=mc["ep_b_analitico_dp_sobre_raiz_n"],
                                                  dist_EP=dist_ate_intervalo(mc["b_menos_mu_dB"], a_, b_, mc["ep_b_analitico_dp_sobre_raiz_n"])),
                        vs_c_menos_mu_2_3_freq100=dict(valor=mc["c_menos_mu_dB"], ep=mc["ep_c_bootstrap_100"],
                                                       dist_EP=dist_ate_intervalo(mc["c_menos_mu_dB"], a_, b_, mc["ep_c_bootstrap_100"])),
                        vs_freq200=dict(valor=freq200["T_freq200_dB"], ep=freq200["ep_bootstrap_200"],
                                        dist_EP=dist_ate_intervalo(freq200["T_freq200_dB"], a_, b_, freq200["ep_bootstrap_200"])),
                        vs_b200_menos_mu_MC200=dict(valor=mc200["b200_menos_mu_dB"], ep=mc200["ep_b200_analitico"],
                                                    dist_EP=dist_ate_intervalo(mc200["b200_menos_mu_dB"], a_, b_, mc200["ep_b200_analitico"])),
                    )
            res_cel[f"{pred}|{pop}"] = dict(
                preditor=pred, populacao=pop, n_pop=int(sel.sum()), mu_U_dB=mu,
                p_medio_fechado_lo_nodal=float(plo.mean()), p_medio_freq200=float(phat.mean()),
                fechado=fechado, extremos_nodais=extremos, frequencia_200=freq200,
                mc_fase2_2_3=mc, mc_200_proprio=mc200, distancias_em_EP=dist)
            print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {cidade} {pred}|{pop}: "
                  f"L=[{fechado['m_nodal']['L_no_a_no']['T_inf_dB']:.4f},{fechado['m_nodal']['L_no_a_no']['T_sup_dB']:.4f}] "
                  f"freq200={freq200['T_freq200_dB']:.4f}+-{freq200['ep_bootstrap_200']:.4f} "
                  f"b-mu={mc['b_menos_mu_dB']:.3f}+-{mc['ep_b_analitico_dp_sobre_raiz_n']:.3f}", flush=True)
    return dict(celula=meta["celula"], n_nos=meta["n_nos"], k_te=kte, k_tr=ktr, N=N,
                mu_e_calibracao=dict(constante_ref_dB=meta["constante_ref_dB"], p_tx_eff_ref_dB=meta["p_tx_eff_ref_dB"]),
                resultados=res_cel)


def main():
    t0 = datetime.now(timezone.utc)
    faltam = [c for c in mo.CIDADES if not (mo.INTERM / f"{c}_{mo.QUAD}.npz").exists()]
    if faltam:
        raise SystemExit(f"intermediarios ausentes ({faltam}); rode scripts/v3_12_R3R4_motor.py antes")
    por_celula = {}
    for c in mo.CIDADES:
        r = processar(c)
        (FASE4 / f"_parcial_R3_{c}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        por_celula[r["celula"]] = r
    # Per comparator: number of cells farther than 2 and 3 SE from the closed-form interval.
    resumo = {}
    for defn, leis in (("m_nodal", LEITURAS_NODAL), ("m_geom", LEITURAS_GEOM)):
        for lei in leis:
            for pred in SIG:
                for pop in POPS:
                    k = f"{defn}|{lei}|{pred}|{pop}"
                    linha = {}
                    for comp in ("vs_b_menos_mu_2_3_MC", "vs_c_menos_mu_2_3_freq100", "vs_freq200", "vs_b200_menos_mu_MC200"):
                        ds = [por_celula[c]["resultados"][f"{pred}|{pop}"]["distancias_em_EP"][f"{defn}|{lei}"][comp]["dist_EP"]
                              for c in por_celula]
                        linha[comp] = dict(dist_EP_por_celula=dict(zip(por_celula, ds)),
                                           n_celulas_acima_de_3EP=int(sum(abs(x) > 3 for x in ds)),
                                           n_celulas_acima_de_2EP=int(sum(abs(x) > 2 for x in ds)))
                    resumo[k] = linha
    # Width of the bound interval relative to 2 SE of the Monte Carlo value of record 2.3.
    largura = {}
    for defn, leis in (("m_nodal", LEITURAS_NODAL), ("m_geom", LEITURAS_GEOM)):
        for lei in leis:
            for pred in SIG:
                for pop in POPS:
                    li = {}
                    for c in por_celula:
                        r = por_celula[c]["resultados"][f"{pred}|{pop}"]
                        w = r["fechado"][defn][lei]["largura_dB"]
                        ep2 = 2.0 * r["mc_fase2_2_3"]["ep_b_analitico_dp_sobre_raiz_n"]
                        li[c] = dict(largura_dB=w, dois_EP_b_2_3_dB=ep2, largura_sobre_2EP=w / ep2, mais_larga_que_2EP=bool(w > ep2))
                    largura[f"{defn}|{lei}|{pred}|{pop}"] = li
    saida = dict(
        id="R3_termo_desenho", roadmap="B3.2",
        criterio=dict(arquivo=str(CRIT), sha256=mo.sha256_file(CRIT)),
        pergunta=json.load(open(CRIT))["pergunta"],
        parametros=dict(N=132, k_te=20, k_tr=92, g_km=mo.G, b_km=mo.B, n_sorteios=200, seeds=mo.seeds_200()[0],
                        fonte_seeds=str(mo.F18), bootstrap_reps=B_BOOT, semente_bootstrap=SEED_BOOT,
                        p_i="(k_te/N) * p_{i|te}; p_{i|te} em [lo(m), hi(m)] (prop:inclusion (ii)); exato = lo(m) onde a leitura o declara",
                        e_i="calibrado UMA vez no split seed=42 (constante=mediana RSSI treino; FSPL(b) offset pela mediana dos validos do treino), como 2.3",
                        extremos_intervalo="extremos exatos da razao com p em caixa (Dinkelbach)"),
        leituras={
            "L_no_a_no": "exato (=lo) onde m=0 ou a condicao (iii) vale no no; demais nos so cotas (primaria, fixada a priori)",
            "T_classe": "exato em m=0,1,3; so cotas em m=2 e m>=4 (main.tex l.312)",
            "C_todas": "lo tratada como exata em toda classe m (ponto)",
            "U_so_m0": "exato so em m=0; m>=1 so cotas",
        },
        leitura_primaria=LEITURA_PRIMARIA,
        desvios_do_criterio=[
            "O criterio compara o termo de desenho com 'a diferenca (b)-(a) de fase2/2.3', que e o vies TOTAL (desenho + razao). Reportado como vs_b_menos_mu_2_3_MC (comparador do criterio, EP = dp/raiz(n_A)); tambem reportado o comparador SEM o termo de razao: (c)-(a) de 2.3 reconstruido (frequencia de 100 sorteios; EP bootstrap dos 100 sorteios) e a frequencia de 200 sorteios deste script.",
            "As seeds/sorteios do 2.3 (100, RandomState(20260927)) diferem dos 200 do 1.8 (RandomState(20260926)); as quatro celulas compartilham a mesma malha de 132 blocos e os mesmos seeds (nao sao replicas independentes de MC).",
            "mu_U usa o e_i recalculado em precisao total (conferido contra 'a' de 2.3 a 1,5e-3 dB); 'a' do 2.3 esta arredondado a 3 casas.",
        ],
        por_celula=por_celula, resumo_distancias=resumo, resumo_largura_intervalo_vs_2EP=largura,
        scripts_sha256={str(SCRIPTS / "v3_12_R3_termo_desenho.py"): mo.sha256_file(Path(__file__)),
                        str(SCRIPTS / "v3_12_R3R4_motor.py"): mo.sha256_file(SCRIPTS / "v3_12_R3R4_motor.py"),
                        str(SCRIPTS / "v3_12_R4_p_fechado_vs_frequencia.py"): mo.sha256_file(SCRIPTS / "v3_12_R4_p_fechado_vs_frequencia.py"),
                        str(mo.S16_PATH) + " (importado, nao editado)": mo.sha256_file(mo.S16_PATH),
                        str(F23): mo.sha256_file(F23), str(F23B): mo.sha256_file(F23B), str(mo.F18): mo.sha256_file(mo.F18)},
        comando="/trabalho/ambientes/s33_amb_virtual/.venv/bin/python " + " ".join(sys.argv),
        data_utc=t0.isoformat(),
    )
    OUT.write_text(json.dumps(saida, indent=1, ensure_ascii=False))
    print("gravado", OUT)


if __name__ == "__main__":
    main()
