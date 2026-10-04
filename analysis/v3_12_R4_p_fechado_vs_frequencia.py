#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check R4: closed-form inclusion probability versus observed frequency, by edge class.

For each edge class m(i), it asks whether the frequency with which a node is retained in
the test partition, CONDITIONAL on its block being a test block (p_{i|te}), over 200 split
draws matches the closed form where the inclusion proposition declares it exact and stays
within the bounds where the proposition only gives bounds. Four Q1 cells, N = 132 blocks,
k_te = 20, k_tr = 92.

Closed form: lo(m) = (k_te - 1)^(m) / (N - 1)^(m) and hi(m) = (N - 1 - k_tr)^(m) / (N - 1)^(m)
(falling factorial powers); lo is attained when condition (iii) holds for the node.
Readings of which nodes are exact, all reported: L_no_a_no (primary, fixed beforehand: exact
where m = 0 or condition (iii) holds at the node), T_classe (exact for m = 0, 1, 3; bounds for
m = 2 and m >= 4), C_todas (lo taken as exact in every class), U_so_m0 (exact only for m = 0).
Two definitions of m: nodal (blocks with a node closer than b, as in the proposition) and
geometric (faces/corners closer than b), the latter only with readings T, C, U.

Frequency of class c: nodal mean of f_i = (draws with i retained in test) / (draws with i's
block in test). Monte Carlo SE: bootstrap over the 200 draws (4000 replicates, numpy
default_rng(20261001)), recomputing the whole f_c including the per-block test counts, so
the correlation of nodes in the same block is kept. Secondary SE: spread of the per-draw
pooled ratio divided by sqrt(200); an alternative block-binomial SE under the null p0 = lo
is also reported.

Inputs: engine intermediates fase4/_v3_12_R3R4_intermediarios/ and the decision criterion
criterios/criterio_R4_p_fechado_vs_frequencia.json.
Outputs: fase4/R4_p_por_classe.json (source of the inclusion-by-class table) and per-cell
fase4/_parcial_R4_<city>.json. CPU only.

Usage: python v3_12_R4_p_fechado_vs_frequencia.py   (after v3_12_R3R4_motor.py)
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPTS = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts")
sys.path.insert(0, str(SCRIPTS))
import v3_12_R3R4_motor as mo  # noqa: E402

RAIZ = mo.RAIZ
FASE4 = RAIZ / "fase4"
CRIT = RAIZ / "criterios" / "criterio_R4_p_fechado_vs_frequencia.json"
OUT = FASE4 / "R4_p_por_classe.json"
B_BOOT = 4000
N_T0 = [0]  # running count of (replicate, block) pairs with no test draw in a bootstrap resample
SEED_BOOT = 20261001
LEITURAS = ("L_no_a_no", "T_classe", "C_todas", "U_so_m0")
LEITURA_PRIMARIA = "L_no_a_no"


def lo_hi(m: int, N: int, kte: int, ktr: int):
    """Lower and upper closed-form bounds of p_{i|te} for class m (falling factorial ratios; 0 when undefined)."""
    if m > kte - 1:
        lo = 0.0
    else:
        lo = math.perm(kte - 1, m) / math.perm(N - 1, m)
    hi = math.perm(N - 1 - ktr, m) / math.perm(N - 1, m) if m <= N - 1 - ktr else 0.0
    return lo, hi


def nao_exato(leitura: str, m: int, falha: bool) -> bool:
    """True when, under the given reading, a node of class m (with condition (iii) failure flag) has only bounds."""
    if m == 0:
        return False
    if leitura == "L_no_a_no":
        return bool(falha)
    if leitura == "T_classe":
        return m not in (1, 3)
    if leitura == "C_todas":
        return False
    if leitura == "U_so_m0":
        return True
    raise ValueError(leitura)


def freq_linhas(R, te, n_bc, agg, W=None):
    """Nodal mean conditional frequency per output row.

    R: (S, J, C) retained test nodes per draw, block, class; te: (S, J) test-block indicator;
    n_bc: (J, C) nodes per block and class; agg: (L, C) 0/1 map from class codes to output rows;
    W: (S,) draw weights (bootstrap counts) or None. Returns f with shape (L,).
    """
    if W is None:
        W = np.ones(R.shape[0])
    T = W @ te
    assert (T > 0).all()
    Rsum = np.tensordot(W, R, axes=(0, 0))
    Rl = Rsum @ agg.T
    n_l = (n_bc @ agg.T).sum(0)
    return (Rl / T[:, None]).sum(0) / n_l


def estatisticas(R, te, n_bc, agg, rng):
    """Frequency, bootstrap SE, per-draw SE, pooled frequency and block diagnostics per output row."""
    S = R.shape[0]
    f = freq_linhas(R, te, n_bc, agg)
    boot = np.empty((B_BOOT, agg.shape[0]))
    Rf = R.reshape(S, -1)
    n_l = (n_bc @ agg.T).sum(0)
    J, C = n_bc.shape
    for b0 in range(0, B_BOOT, 500):
        nb = min(500, B_BOOT - b0)
        Wb = rng.multinomial(S, np.full(S, 1.0 / S), size=nb).astype(np.float64)
        Tb = Wb @ te
        N_T0[0] += int((Tb <= 0).sum())  # block never in test in this resample: contributes 0, counted
        Tb = np.where(Tb > 0, Tb, 1.0)
        Rb = (Wb @ Rf).reshape(nb, J, C)
        Rl = Rb @ agg.T
        boot[b0:b0 + nb] = (Rl / Tb[:, :, None]).sum(1) / n_l
    ep = boot.std(axis=0, ddof=1)
    # Secondary: pooled ratio per draw.
    num = np.einsum("sjc,lc->sjl", R.astype(np.float64), agg)
    den = np.einsum("sj,jl->sjl", te, n_bc @ agg.T)
    fs = num.sum(1) / np.where(den.sum(1) > 0, den.sum(1), np.nan)
    ep_s = np.nanstd(fs, axis=0, ddof=1) / np.sqrt(S)
    pooled = num.sum((0, 1)) / den.sum((0, 1))
    # Block diagnostics: in small classes the draw bootstrap understates the SE of rare events.
    n_jl = n_bc @ agg.T
    Tj = te.sum(0)
    w = n_jl / n_jl.sum(0)[None, :]
    extras = dict(
        n_blocos=(n_jl > 0).sum(0), ensaios_bloco_sorteio=((n_jl > 0) * Tj[:, None]).sum(0),
        retidos_bloco_sorteio_com_algum_no=(num > 0).sum((0, 1)),
        soma_w2_sobre_T=((w ** 2) / Tj[:, None]).sum(0))
    return f, ep, ep_s, pooled, extras


def montar_linhas(nome_def, codigos, agg_rows, f, ep, ep_s, pooled, extras, n_codigo, n_total, N, kte, ktr, leituras):
    """Build one output record per class row with the bounds and distances under each reading.

    codigos: (m, failure flag) per class column; agg_rows: list of (label, [column indices]).
    """
    linhas = []
    for li, (rot, cols) in enumerate(agg_rows):
        n_row = int(sum(n_codigo[c] for c in cols))
        if n_row == 0:
            continue
        ms = sorted({codigos[c][0] for c in cols})
        lo_m = {m: lo_hi(m, N, kte, ktr) for m in ms}
        row = dict(classe=rot, definicao_m=nome_def, n_nos=n_row, fracao_nos=n_row / n_total,
                   m_valores=ms,
                   n_nos_cond_iii_falha=int(sum(n_codigo[c] for c in cols if codigos[c][1])),
                   frequencia_media_nodal=float(f[li]), ep_mc_bootstrap_sorteios=float(ep[li]),
                   ep_mc_por_sorteio_agregado=float(ep_s[li]), frequencia_agregada_pooled=float(pooled[li]),
                   n_blocos_com_a_classe=int(extras["n_blocos"][li]),
                   ensaios_bloco_sorteio=float(extras["ensaios_bloco_sorteio"][li]),
                   retidos_bloco_sorteio_com_algum_no=int(extras["retidos_bloco_sorteio_com_algum_no"][li]),
                   cota_inferior_por_m={str(m): lo_m[m][0] for m in ms},
                   cota_superior_por_m={str(m): lo_m[m][1] for m in ms},
                   leituras={})
        for lei in leituras:
            lo_row = sum(n_codigo[c] * lo_m[codigos[c][0]][0] for c in cols) / n_row
            hi_row = sum(n_codigo[c] * (lo_m[codigos[c][0]][1] if nao_exato(lei, *codigos[c]) else lo_m[codigos[c][0]][0])
                         for c in cols) / n_row
            n_nao_ex = int(sum(n_codigo[c] for c in cols if nao_exato(lei, *codigos[c])))
            exato_puro = n_nao_ex == 0
            epv = float(ep[li])
            fr = float(f[li])
            # Alternative block SE under the null p0 = lo_row (independent blocks, nodes of a block
            # perfectly correlated): sqrt(sum_j w_j^2 p0 (1 - p0) / T_j).
            ep_bn = float(math.sqrt(extras["soma_w2_sobre_T"][li] * lo_row * (1 - lo_row)))
            if epv > 0:
                dif_lo = (fr - lo_row) / epv
                if fr > hi_row:
                    dif_cota = (fr - hi_row) / epv
                elif fr < lo_row:
                    dif_cota = (fr - lo_row) / epv
                else:
                    dif_cota = 0.0
            else:
                dif_lo = 0.0 if abs(fr - lo_row) < 1e-12 else float("inf")
                dif_cota = 0.0 if (lo_row - 1e-12 <= fr <= hi_row + 1e-12) else float("inf")
            rl = dict(
                n_nos_so_cotas=n_nao_ex, exato_em_toda_a_classe=exato_puro,
                valor_fechado=lo_row if exato_puro else None,
                cota_inferior_classe=lo_row, cota_superior_classe=hi_row,
                dif_freq_menos_inferior_em_EP=dif_lo,
                ep_alternativo_por_blocos_binomial_nula=ep_bn,
                dif_freq_menos_inferior_em_EP_alternativo=((fr - lo_row) / ep_bn if ep_bn > 0 else (0.0 if abs(fr - lo_row) < 1e-12 else float("inf"))),
                dif_freq_ate_intervalo_em_EP=dif_cota,
                dentro_do_intervalo=bool(lo_row - 1e-12 <= fr <= hi_row + 1e-12),
                dentro_do_intervalo_ou_a_2EP=bool(dif_cota == 0.0 or abs(dif_cota) <= 2.0),
                coincide_a_2EP_do_valor_fechado=(bool(abs(dif_lo) <= 2.0) if exato_puro else None),
            )
            row["leituras"][lei] = rl
        linhas.append(row)
    return linhas


def veredito(linhas, leitura):
    """Apply the decision criterion at 2 SE under one reading and list the failing classes.

    Fully exact classes must match the closed value within 2 SE; classes with bound-only
    nodes must fall inside the interval or within 2 SE of it.
    """
    falham, avaliadas, alt_falham = [], 0, []
    for r in linhas:
        rl = r["leituras"].get(leitura)
        if rl is None:
            continue
        avaliadas += 1
        if rl["exato_em_toda_a_classe"]:
            ok = rl["coincide_a_2EP_do_valor_fechado"]
            ok_alt = abs(rl["dif_freq_menos_inferior_em_EP_alternativo"]) <= 2.0
        else:
            ok = rl["dentro_do_intervalo_ou_a_2EP"]
            ok_alt = ok
        if not ok_alt:
            alt_falham.append(r["classe"])
        if not ok:
            falham.append(dict(classe=r["classe"], freq=r["frequencia_media_nodal"], ep=r["ep_mc_bootstrap_sorteios"],
                               cota_inferior=rl["cota_inferior_classe"], cota_superior=rl["cota_superior_classe"],
                               exato=rl["exato_em_toda_a_classe"], dif_lo_EP=rl["dif_freq_menos_inferior_em_EP"],
                               dif_ate_intervalo_EP=rl["dif_freq_ate_intervalo_em_EP"], n_nos=r["n_nos"]))
    return dict(criterio_cumprido=len(falham) == 0, n_classes_avaliadas=avaliadas, classes_que_falham=falham,
                criterio_cumprido_com_EP_alternativo_por_blocos=len(alt_falham) == 0,
                classes_que_falham_com_EP_alternativo=alt_falham)


def processar(cidade: str) -> dict:
    A, meta = mo.carregar_intermediario(cidade)
    N, kte, ktr = meta["n_blocos"], meta["k_te"], meta["k_tr"]
    assert (N, kte, ktr) == (132, 20, 92)
    te = A["te_block"].astype(np.float64)
    S = te.shape[0]
    n_total = meta["n_nos"]
    rng = np.random.default_rng(SEED_BOOT)
    # Nodal classes: column code = m * 2 + (condition (iii) fails), 0..9.
    codigos_n = [(c // 2, bool(c % 2)) for c in range(mo.NCG_N)]
    n_cod_n = A["n_bc_n"].sum(0)
    rows_n = []
    for mm in range(5):
        cols_all = [mm * 2, mm * 2 + 1]
        if n_cod_n[cols_all].sum() == 0:
            continue
        rows_n.append((f"m={mm}|todas", cols_all))
        rows_n.append((f"m={mm}|(iii) vale", [mm * 2]))
        rows_n.append((f"m={mm}|(iii) falha", [mm * 2 + 1]))
    rows_n = [(rot, cols) for rot, cols in rows_n if n_cod_n[cols].sum() > 0]
    rows_n.append(("TOTAL|todos os nos", list(range(mo.NCG_N))))
    agg_n = np.zeros((len(rows_n), mo.NCG_N))
    for li, (_, cols) in enumerate(rows_n):
        agg_n[li, cols] = 1.0
    f, ep, ep_s, pooled, extras = estatisticas(A["R_n"], te, A["n_bc_n"].astype(np.float64), agg_n, rng)
    linhas_n = montar_linhas("nodal", codigos_n, rows_n, f, ep, ep_s, pooled, extras, n_cod_n, n_total, N, kte, ktr, LEITURAS)
    # Geometric classes.
    codigos_g = [(c, False) for c in range(mo.NCG_G)]
    n_cod_g = A["n_bc_g"].sum(0)
    rows_g = [(f"mg={c}", [c]) for c in range(mo.NCG_G) if n_cod_g[c] > 0]
    rows_g.append(("TOTAL|todos os nos", list(range(mo.NCG_G))))
    agg_g = np.zeros((len(rows_g), mo.NCG_G))
    for li, (_, cols) in enumerate(rows_g):
        agg_g[li, cols] = 1.0
    f, ep, ep_s, pooled, extras = estatisticas(A["R_g"], te, A["n_bc_g"].astype(np.float64), agg_g, rng)
    linhas_g = montar_linhas("geom", codigos_g, rows_g, f, ep, ep_s, pooled, extras, n_cod_g, n_total, N, kte, ktr,
                             ("T_classe", "C_todas", "U_so_m0"))
    # Each reading is judged at its own granularity (L per (iii) sub-class, the others per m).
    ver = {}
    nodal_fina = [r for r in linhas_n if r["classe"].endswith("vale") or r["classe"].endswith("falha")]
    nodal_m = [r for r in linhas_n if r["classe"].endswith("todas") and r["classe"].startswith("m=")]
    ver["nodal"] = {
        "L_no_a_no": veredito(nodal_fina, "L_no_a_no"),
        "T_classe": veredito(nodal_m, "T_classe"),
        "C_todas": veredito(nodal_m, "C_todas"),
        "U_so_m0": veredito(nodal_m, "U_so_m0"),
    }
    geo_m = [r for r in linhas_g if r["classe"].startswith("mg=")]
    ver["geom"] = {lei: veredito(geo_m, lei) for lei in ("T_classe", "C_todas", "U_so_m0")}
    cruz = {}
    m_ = A["m"]
    mg_ = A["mg"]
    for a in range(int(m_.max()) + 1):
        cruz[f"m_nodal={a}"] = {f"mg={b}": int(((m_ == a) & (mg_ == b)).sum()) for b in range(int(mg_.max()) + 1)}
    fid = dict(retencao_teste_por_sorteio_igual_1_8_a_1e12_assert_no_motor=True,
               media_retencao_nodal_te=float(np.mean(A["n_te"] / A["n_te0"])),
               n_nos=n_total, n_blocos=N, k_te=kte, k_tr=ktr, m_max_observado=meta["m_max_observado"],
               n_nos_por_m=meta["n_nos_por_m"], n_nos_falha_iii_por_m=meta["n_falha_iii_por_m"],
               frequencia_total_media_nodal_p_cond_te=[r["frequencia_media_nodal"] for r in linhas_n if r["classe"].startswith("TOTAL")][0])
    return dict(celula=meta["celula"], fidelidade=fid, classes_nodais=linhas_n, classes_geometricas=linhas_g,
                cruzamento_m_nodal_x_m_geom=cruz, veredito_por_leitura=ver,
                bootstrap_blocos_reamostra_sem_teste_acumulado=int(N_T0[0]))


def main():
    t0 = datetime.now(timezone.utc)
    faltam = [c for c in mo.CIDADES if not (mo.INTERM / f"{c}_{mo.QUAD}.npz").exists()]
    if faltam:
        raise SystemExit(f"intermediarios ausentes ({faltam}); rode scripts/v3_12_R3R4_motor.py antes")
    res = {}
    for c in mo.CIDADES:
        r = processar(c)
        (FASE4 / f"_parcial_R4_{c}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        res[r["celula"]] = r
        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {c}: parcial gravado", flush=True)
    seeds, _ = mo.seeds_200()
    saida = dict(
        id="R4_p_fechado_vs_frequencia", roadmap="B3.3",
        criterio=dict(arquivo=str(CRIT), sha256=mo.sha256_file(CRIT)),
        pergunta=json.load(open(CRIT))["pergunta"],
        parametros=dict(N=132, k_te=20, k_tr=92, g_km=mo.G, b_km=mo.B, n_sorteios=200, seeds=seeds,
                        fonte_seeds=str(mo.F18), bootstrap_reps=B_BOOT, semente_bootstrap=SEED_BOOT,
                        formula_cotas="lo=(k_te-1)^{m}/(N-1)^{m}; hi=(N-1-k_tr)^{m}/(N-1)^{m} (potencia fatorial descendente)"),
        leituras={
            "L_no_a_no": "exato (=lo) onde m=0 ou a condicao (iii) vale no no (calculada por no); demais so cotas. Primaria fixada a priori (leitura literal de prop:inclusion (iii)).",
            "T_classe": "exato em m=0,1,3; so cotas em m=2 e m>=4 (texto de main.tex l.312).",
            "C_todas": "lo tratada como valor exato em toda classe (ignora (iii)).",
            "U_so_m0": "exato so em m=0; m>=1 so cotas.",
        },
        leitura_primaria=LEITURA_PRIMARIA,
        ambiguidades=[
            "A proposicao diz que a cota inferior e ATINGIDA SE vale a hipotese (iii); nao diz que ela falha fora dela nem declara por classe o que e exato. As quatro leituras (L, T, C, U) sao implementadas e reportadas; nenhuma foi escolhida depois de ver o numero.",
            "m(i) tem duas definicoes no material: nodal (main.tex l.296: blocos com >= 1 no a < b; usada na proposicao) e geometrica (lados/cantos a < b, como o criterio e fismat_k_corrigido.py descrevem). Ambas reportadas; as diferencas sao discretizacao e blocos finos.",
            "'declarada exata' no criterio: so m=0 tem cota inferior = superior (p=1). Nas demais classes a proposicao so afirma 'atingida se (iii)'.",
        ],
        por_celula=res,
        trabalho_anterior=dict(
            fismat_k_corrigido=("geometric class m (8-neighbourhood by distance to face/vertex) only in Bauru Q1, 100+100 seeds; "
                                "here: exact nodal m in the 4 cells, condition (iii) per node, 200 seeds from 1.8, SE by bootstrap over split draws"),
            fismat_inclusao_decomp="decomposicao de m=1/m=2/m=3 de Bauru Q1 a partir do JSON do fismat_k_corrigido; nao recomputada",
            fismat_razao_cov="identidade da razao e decomposicao da retencao por classe k lateral (1.6); nao cobre p por classe m",
            p_inclusao_por_no_1_6="perfil por classe k lateral-only com errata; nao usado como comparador",
        ),
        scripts_sha256={str(SCRIPTS / "v3_12_R4_p_fechado_vs_frequencia.py"): mo.sha256_file(Path(__file__)),
                        str(SCRIPTS / "v3_12_R3R4_motor.py"): mo.sha256_file(SCRIPTS / "v3_12_R3R4_motor.py"),
                        str(mo.S16_PATH) + " (importado, nao editado)": mo.sha256_file(mo.S16_PATH)},
        comando="/trabalho/ambientes/s33_amb_virtual/.venv/bin/python " + " ".join(sys.argv),
        data_utc=t0.isoformat(),
    )
    OUT.write_text(json.dumps(saida, indent=1, ensure_ascii=False))
    print("gravado", OUT)


if __name__ == "__main__":
    main()
