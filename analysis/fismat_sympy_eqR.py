#!/usr/bin/env python3
"""Symbolic and numerical check of the retention law and of related closed forms.

Block side g, buffer width b (b <= g/2) and q = probability that a neighbouring block does
not trim the block under study. Sections, one key each in the output JSON:
  A1  exact enumeration of the 2^8 trim indicators of the 4 lateral and 4 diagonal
      neighbours of an interior block; the expected surviving area is compared with the
      numerator of R(q) = [(g-2b)^2 + 4b(g-2b)q + 4b^2(1-pi/4)q^2 + pi b^2 q^3] / g^2;
  A2  independent derivation: the coefficient of q^k is the area within b of exactly k neighbours;
  A3  raster check of the surviving-area formula for random neighbour sets;
  A4  lattice edge/corner closed forms; Lins Q1 lattice retention (mean-field, exact permutation);
  B   constant-predictor MAE as an affine function of the sentinel share and its variance;
  C   edges of the median band under free-space loss, and a sampled check of the inequality;
  D   floor and plateau of the link-budget error on a sentinel node.

Output: fismat_sympy_eqR.json next to this script (also printed). Seeds: 20260925 (A3), 7 (C).
Usage: python fismat_sympy_eqR.py
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp

OUT = {}

# ---------------------------------------------------------------- A1: exact enumeration
g, b, q = sp.symbols("g b q", positive=True)
names = ["L", "R", "Bm", "T", "LB", "LT", "RB", "RT"]  # 4 lateral + 4 diagonal neighbours
corner_adj = {"LB": ("L", "Bm"), "LT": ("L", "T"), "RB": ("R", "Bm"), "RT": ("R", "T")}
EA = 0
for bits in itertools.product([0, 1], repeat=8):
    t = dict(zip(names, bits))  # t=1: the neighbour trims (probability 1 - q)
    prob = 1
    for v in bits:
        prob *= (1 - q) if v else q
    A = (g - b * (t["L"] + t["R"])) * (g - b * (t["Bm"] + t["T"]))
    A -= sp.pi * b**2 / 4 * sum(t[k] * (1 - t[a1]) * (1 - t[a2]) for k, (a1, a2) in corner_adj.items())  # quarter disc per trimming diagonal with both adjacent laterals non-trimming
    EA += prob * A
EA = sp.expand(EA)
eqR_num = (g - 2*b)**2 + 4*b*(g - 2*b)*q + 4*b**2*(1 - sp.pi/4)*q**2 + sp.pi*b**2*q**3
app_lateral = (g - 2*b)**2 + 4*b*(g - 2*b)*q + 4*b**2*q**2
app_corner = sp.pi*b**2*q**2 - sp.pi*b**2*q**3
OUT["A1"] = {
    "E_A_enumeracao_2a8": str(sp.collect(EA, q)),
    "diferenca_vs_eqR_numerador": str(sp.simplify(EA - sp.expand(eqR_num))),
    "diferenca_vs_app_lateral_menos_app_corner": str(sp.simplify(EA - sp.expand(app_lateral - app_corner))),
    "R_q0": str(sp.factor(sp.simplify((eqR_num / g**2).subs(q, 0)))),
    "R_q1": str(sp.simplify((eqR_num / g**2).subs(q, 1))),
    "dR_dq_nao_negativo_em_[0,1]_para_b<=g/2": str(sp.factor(sp.diff(eqR_num, q))),
}

# ---------------------------------------------------------------- A2: areas within b of exactly k neighbours (interior block, b <= g/2)
a0 = (g - 2*b)**2                       # core, no neighbour within b
a1 = 4*b*(g - 2*b)                      # 4 lateral strips without the corner squares
a2 = 4*(b**2 - sp.pi*b**2/4)            # corner square outside the quarter disc: two lateral neighbours
a3 = 4*(sp.pi*b**2/4)                   # quarter disc: two lateral neighbours and the diagonal
EA2 = a0 + a1*q + a2*q**2 + a3*q**3
OUT["A2"] = {"coef_q^k_por_area_k_vizinhos": [str(a0), str(a1), str(sp.simplify(a2)), str(a3)],
             "diferenca_vs_eqR_numerador": str(sp.simplify(EA2 - eqR_num))}


# ---------------------------------------------------------------- A3: raster check of the surviving-area formula
def raster_counts(w, h, nb, bb, res):
    """Cell-centre raster of [0, w] x [0, h] at spacing res (km): returns M (n_points x 8),
    True where a point lies within bb of an existing neighbour (nb: name -> exists, default
    True), and the cell area res^2."""
    xs = (np.arange(int(round(w / res))) + 0.5) * res
    ys = (np.arange(int(round(h / res))) + 0.5) * res
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    X = X.ravel(); Y = Y.ravel()
    d = {
        "L": X, "R": w - X, "Bm": Y, "T": h - Y,
        "LB": np.hypot(X, Y), "LT": np.hypot(X, h - Y),
        "RB": np.hypot(w - X, Y), "RT": np.hypot(w - X, h - Y),
    }
    M = np.stack([(d[k] <= bb) & nb.get(k, True) for k in names], axis=1)
    return M, res * res


def area_formula(S, gg, bb):  # closed-form surviving area for the set S of trimming neighbours
    one = lambda k: 1 if k in S else 0
    A = (gg - bb*(one("L") + one("R"))) * (gg - bb*(one("Bm") + one("T")))
    ncorn = sum(1 for k, (a1_, a2_) in corner_adj.items() if k in S and a1_ not in S and a2_ not in S)
    return A - math.pi*bb**2/4*ncorn


rng = np.random.default_rng(20260925)
checks = []
for gg, bb in [(10.0, 2.0), (5.0, 2.0), (10.0, 5.0)]:
    M, dA = raster_counts(gg, gg, {}, bb, 0.005)
    for _ in range(12):
        S = [k for k in names if rng.random() < 0.5]
        mask = np.array([k in S for k in names])
        surv = (~(M[:, mask].any(axis=1))).sum() * dA if mask.any() else gg*gg
        fa = area_formula(S, gg, bb)
        checks.append(abs(surv - fa) / (gg*gg))
OUT["A3"] = {"n_configuracoes": len(checks), "max_erro_relativo_raster_vs_formula": float(max(checks)),
             "resolucao_km": 0.005}

# ---------------------------------------------------------------- A4: lattice edge (Bm, LB, RB absent) and corner (L, Bm absent) blocks, mean-field
E_edge = sp.expand(((g - 2*b) + 2*b*q) * (g - b*(1 - q)) - 2*(sp.pi*b**2/4)*(1 - q)*q**2)
E_corner = sp.expand((g - b*(1 - q))**2 - (sp.pi*b**2/4)*(1 - q)*q**2)
OUT["A4_formas_fechadas"] = {"E_A_bloco_de_borda": str(sp.collect(E_edge, q)),
                             "E_A_bloco_de_canto": str(sp.collect(E_corner, q))}


def falling(a, n):  # falling factorial (a)_n = a (a - 1) ... (a - n + 1)
    r = 1.0
    for j in range(n):
        r *= (a - j)
    return r


def lattice_retention(Wx, Wy, gg, bb, k_part, N_nontrim_count, N, full_blocks=False, res=0.01):
    """Expected retention = sum over blocks of E[surviving area] / sum of block areas.
    Mean-field: P(k given neighbours do not trim) = qmf^k, qmf = N_nontrim_count / N; exact
    under a permutation with fixed counts: (N_nontrim_count - 1)_k / (N - 1)_k, where the count
    includes the block itself (test: k_te; validation: N - k_tr). full_blocks=True uses whole
    g x g blocks, False keeps the thinner last column and row; k_part is unused."""
    m = int(math.floor(Wx / gg)) + (0 if abs(Wx/gg - round(Wx/gg)) < 1e-9 else 1)
    n = int(math.floor(Wy / gg)) + (0 if abs(Wy/gg - round(Wy/gg)) < 1e-9 else 1)
    if full_blocks:
        widths = [gg]*m; heights = [gg]*n
    else:
        widths = [gg]*(m-1) + [Wx - gg*(m-1)]
        heights = [gg]*(n-1) + [Wy - gg*(n-1)]
    qmf = N_nontrim_count / N
    cache = {}
    num_mf = num_ex = den = 0.0
    for i in range(m):
        for j in range(n):
            nb = {"L": i > 0, "R": i < m-1, "Bm": j > 0, "T": j < n-1,
                  "LB": i > 0 and j > 0, "LT": i > 0 and j < n-1,
                  "RB": i < m-1 and j > 0, "RT": i < m-1 and j < n-1}
            key = (round(widths[i], 6), round(heights[j], 6), tuple(nb[k] for k in names))
            if key not in cache:
                M, dA = raster_counts(widths[i], heights[j], nb, bb, res)
                cnt = M.sum(axis=1)
                ak = np.bincount(cnt, minlength=9) * dA
                cache[key] = ak
            ak = cache[key]
            num_mf += sum(ak[k] * qmf**k for k in range(9))
            num_ex += sum(ak[k] * falling(N_nontrim_count - 1, k) / falling(N - 1, k) for k in range(9))
            den += widths[i] * heights[j]
    return {"m_x_n": [m, n], "N_blocos": m*n, "mean_field": num_mf/den, "exato_permutacao": num_ex/den,
            "larguras_borda_km": [widths[-1], heights[-1]]}


# Domain: Lins Q1 extent in degrees; equirectangular, 111 km per degree, cos(latitude) at the centre.
lon_min, lon_max = -50.745140075683594, -49.745418548583984
lat_min, lat_max = -21.679304122924805, -20.679582595825195
s = 111.0
phic = math.radians(0.5*(lat_min + lat_max))
Wx = s*(lon_max - lon_min)*math.cos(phic)
Wy = s*(lat_max - lat_min)
A4 = {"dominio_km": [Wx, Wy], "nota": "cos(phi) no centro; o texto usa cos por no (trapezio), efeito sub-km na coluna fina"}
for gg in (10.0, 5.0):
    bb = 2.0
    N = (int(math.ceil(Wx/gg)))*(int(math.ceil(Wy/gg)))
    k_tr = max(1, round(0.70*N)); k_va = max(1, round(0.15*N)); k_te = N - k_tr - k_va  # 70/15/15 block split
    res = 0.01
    out = {"N": N, "k_tr_va_te": [k_tr, k_va, k_te]}
    for part, nontrim in (("teste", k_te), ("validacao", N - k_tr)):
        qmf = nontrim / N
        law = float((eqR_num/g**2).subs({g: gg, b: bb, q: qmf}))
        law_nom = float((eqR_num/g**2).subs({g: gg, b: bb, q: (0.15 if part == "teste" else 0.30)}))
        full = lattice_retention(Wx, Wy, gg, bb, None, nontrim, N, full_blocks=True, res=res)
        real = lattice_retention(Wx, Wy, gg, bb, None, nontrim, N, full_blocks=False, res=res)
        out[part] = {
            "q_realizado": qmf, "lei_eqR_q_realizado": law, "lei_eqR_q_nominal": law_nom,
            "reticulado_blocos_cheios_mean_field": full["mean_field"],
            "reticulado_real_mean_field": real["mean_field"],
            "reticulado_real_exato_permutacao": real["exato_permutacao"],
            "efeito_perimetro_pp_mean_field": 100*(full["mean_field"] - law),
            "efeito_blocos_finos_pp_mean_field": 100*(real["mean_field"] - full["mean_field"]),
            "efeito_acoplamento_pp_(exato-meanfield)": 100*(real["exato_permutacao"] - real["mean_field"]),
            "larguras_borda_km": real["larguras_borda_km"],
        }
    A4["g%d_b2" % int(gg)] = out
OUT["A4_reticulado_lins_Q1"] = A4

# ---------------------------------------------------------------- B: MAE affine in the sentinel share pibar
pb, D, e = sp.symbols("pibar Delta e", real=True)
mae = pb*D + (1 - pb)*e
OUT["B_P4"] = {"dMAE_dpibar": str(sp.simplify(sp.diff(mae, pb))),
               "Var_MAE": "(Delta - e)**2 * Var_sigma(pibar_te)  [afim => fator ao quadrado]"}

# ---------------------------------------------------------------- C: median band edges under free-space loss
def fspl_db(d_km, f_mhz=900.0):  # free-space path loss in dB, d in km and f in MHz
    return 20*math.log10(d_km) + 20*math.log10(f_mhz) + 32.44


cP = -110.0
C = {}
for pib in (0.8, 0.9146, 0.915):
    lo_lvl = 1 - 1/(2*pib); hi_lvl = 1/(2*pib)
    d_lo = 1 + 139*lo_lvl; d_hi = 1 + 139*hi_lvl       # quantiles of U[1, 140] km (continuous, so Q- = Q+)
    C["pi_%s" % pib] = {"nivel_inferior": lo_lvl, "nivel_superior": hi_lvl,
                        "largura_em_nivel_(1-pi)/pi": (1 - pib)/pib,
                        "borda_inferior_dBm": cP + fspl_db(d_lo), "borda_superior_dBm": cP + fspl_db(d_hi)}
# Sampled check of the band inequality with ties (rounded values) for arbitrary valid
# populations: every median convention must lie in [Q-, Q+] of the sentinel distribution.
rng2 = np.random.default_rng(7)
viol = 0; ntest = 0
for _ in range(400):
    n = int(rng2.integers(20, 400)); pib = rng2.uniform(0.5, 0.99)
    ns = max(int(round(pib*n)), n//2 + 1); nv = n - ns
    ws = cP + np.round(rng2.uniform(100, 140, ns))        # atoms (ties) in the sentinel set
    wv = rng2.normal(rng2.uniform(-100, 100), rng2.uniform(0, 30), nv)
    if rng2.random() < 0.3:
        wv = np.round(wv)
    w = np.concatenate([ws, wv]); pe = ns/n
    srt = np.sort(w)
    meds = [srt[(n-1)//2], srt[n//2], 0.5*(srt[(n-1)//2] + srt[n//2])]  # lower, upper and midpoint medians
    wsrt = np.sort(ws)
    F1 = lambda x: np.searchsorted(wsrt, x, side="right")/ns
    a_lvl = 1 - 1/(2*pe); b_lvl = 1/(2*pe)
    Qm = min(x for x in wsrt if F1(x) >= a_lvl - 1e-12)
    cand = [x for x in wsrt if F1(x) > b_lvl + 1e-12]
    Qp = min(cand) if cand else np.inf
    for mm in meds:
        ntest += 1
        if not (Qm - 1e-9 <= mm <= Qp + 1e-9):
            viol += 1
C["checagem_amostral"] = {"n_medianas_testadas": ntest, "violacoes": viol,
                          "nota": "sentinela com atomos (empates), validos arbitrarios, n par e impar"}
OUT["C_P7"] = C

# ---------------------------------------------------------------- D: link-budget floor on a sentinel node
# J(L) = |L - cL| + |(Ptx - L) - cP| for a prediction obeying P = Ptx - L, L in [0, Lmax_eff]
Ptx, cL, cP8 = 43.0, 300.0, -110.0
Lmax_dec, Pmin = 200.0, -150.0
Lmax_eff = min(Lmax_dec, Ptx - Pmin)
Ls = np.linspace(0, Lmax_eff, 193001)
J = np.abs(Ls - cL) + np.abs((Ptx - Ls) - cP8)
floor = J.min(); plat = Ls[np.isclose(J, floor, atol=1e-9)]
OUT["D_P8"] = {"L_max_decodificador_PL": Lmax_dec, "P_min_decodificador": Pmin,
               "L_max_efetivo_min(200,Ptx-Pmin)": Lmax_eff, "piso_dB": float(floor),
               "plato_dB": [float(plat.min()), float(plat.max())],
               "hipoteses": {"Lmax<=cL": Lmax_eff <= cL, "Ptx-cP<=Lmax": Ptx - cP8 <= Lmax_eff}}

dst = Path(__file__).with_suffix(".json")
dst.write_text(json.dumps(OUT, indent=2, ensure_ascii=False))
print(json.dumps(OUT, indent=2, ensure_ascii=False))
