"""Variance-component statistics for the crossed draw x seed block of campaign G1 (Campinas Q1).

For each model (GNN, MLP) the input is the matrix of valid-node MAE with one row per split draw
(10 draws) and one column per training seed (42, 43, 44). A one-way random-effects ANOVA with
draws as groups gives rho = (between-draw variance component) / (within-draw, between-seed
variance), its exact 95% interval from the F distribution, and the one-sided p-value of
H0: rho <= 1. It is computed on the full matrix, on log(MAE), and leaving one draw out at a time
(on both scales). The script also runs Cochran's test for the largest row variance and gives an
approximate 95% interval (F(19, 20), independence assumed) for the stored ratio of the 20-draw
standard deviation to the comparator standard deviation, with the rho that ratio implies. The
same ANOVA is redone on the Bauru Q1 matrices (5 draws) of the block-2 recalculation record,
next to the interval stored there.

Inputs: results/gpu/G1/agregado_G1_v10_bloco4.json and the block-2 record under
results/gpu/G1_votos_bloco2/ (absolute paths of the original run; their SHA-256 is stored in `entradas`).
Output: ffm_G1_bloco4_contas.json (path in OUT); a summary is printed.
Usage: python ffm_G1_bloco4_contas.py   Deterministic.
"""
import json, hashlib, numpy as np
from scipy import stats

R ="/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25"
AG = f"{R}/gpu/G1/agregado_G1_v10_bloco4.json"
V2 = f"{R}/gpu/G1_votos_bloco2/veredito.json"
OUT = f"{R}/fase5/internal/ffm_G1_bloco4_contas.json"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
ag = json.load(open(AG)); v2 = json.load(open(V2))
cel = ag["celulas"]["campinas_Q1"]


def anova(M):
    """One-way random-effects ANOVA on an (a groups x n replicates) matrix; rho = s2_between / s2_within."""
    M = np.asarray(M, float); a, n = M.shape
    gm = M.mean(); rm = M.mean(1)
    qme = n * ((rm - gm) ** 2).sum() / (a - 1)
    qmd = ((M - rm[:, None]) ** 2).sum() / (a * (n - 1))
    s2a = max(0.0, (qme - qmd) / n)
    F = qme / qmd; d1, d2 = a - 1, a * (n - 1)
    # Exact interval for rho: F / (1 + n*rho) ~ F(d1, d2), inverted at the 2.5% and 97.5% quantiles.
    lo = max(0.0, (F / stats.f.ppf(0.975, d1, d2) - 1) / n)
    hi = (F / stats.f.ppf(0.025, d1, d2) - 1) / n
    # p-value of H0: rho <= 1, evaluated at rho = 1, where F / (1 + n) ~ F(d1, d2).
    p_rho1 = float(stats.f.sf(F / (1 + n), d1, d2))
    return dict(a=a, gl_entre=d1, gl_dentro=d2, qm_entre=qme, qm_dentro=qmd,
                rho=s2a / qmd, ic95=[lo, hi], p_unilateral_rho_le_1=p_rho1)


def cochran(M):
    """Cochran's C test for the largest row variance, with its critical value at alpha = 0.05."""
    M = np.asarray(M, float); v = M.var(1, ddof=1)
    a, n = M.shape; C = v.max() / v.sum()
    # Critical value: 1 / (1 + (a-1) / F_{1-alpha/a}(n-1, (a-1)(n-1))).
    Fc = stats.f.ppf(1 - 0.05 / a, n - 1, (a - 1) * (n - 1))
    return dict(C=C, C_crit_005=1 / (1 + (a - 1) / Fc), linha_max=int(v.argmax()),
                var_linhas=v.tolist())


out = {"entradas": {AG: sha(AG), V2: sha(V2)}, "modelos": {}}
for mod in ("gnn", "mlp"):
    M = np.asarray(cel[mod]["matriz_mae_validos_sorteio_x_semente_42_43_44"])
    ss = cel[mod]["sorteios_usados"]
    r = {"completo": anova(M), "cochran": cochran(M),
         "log_mae": anova(np.log(M))}
    # Leave-one-draw-out: drop one row (split draw) at a time; keys are the dropped draw seeds.
    lodo = {}
    for i, s in enumerate(ss):
        lodo[str(s)] = anova(np.delete(M, i, 0))
    r["lodo"] = lodo
    lodo_log = {str(s_): anova(np.log(np.delete(M, i, 0))) for i, s_ in enumerate(ss)}
    r["lodo_log"] = lodo_log
    r["lodo_log_min"] = min((v["rho"], k) for k, v in lodo_log.items())
    r["lodo_log_ic_inferior_min"] = min((v["ic95"][0], k) for k, v in lodo_log.items())
    r["lodo_log_n_ic_inferior_gt_1"] = sum(v["ic95"][0] > 1 for v in lodo_log.values())
    r["lodo_min"] = min((v["rho"], k) for k, v in lodo.items())
    r["lodo_ic_inferior_min"] = min((v["ic95"][0], k) for k, v in lodo.items())
    r["lodo_n_ic_inferior_gt_1"] = sum(v["ic95"][0] > 1 for v in lodo.values())
    rr = cel[mod]["razao_dp_sorteios20_sobre_comparador"]
    r["razao_dp"] = rr
    r["rho_implicito_pela_razao_dp"] = rr ** 2 - 1
    # Squared SD ratio treated as F with (19, 20) degrees of freedom, independence assumed.
    f_lo, f_hi = stats.f.ppf([0.025, 0.975], 19, 20)
    r["razao_dp_ic95_aprox_indep"] = [rr / np.sqrt(f_hi), rr / np.sqrt(f_lo)]
    r["P_razao_ge_3_se_verdadeira_2_98_aprox"] = float(stats.f.sf((3 / rr) ** 2, 19, 20))
    out["modelos"][mod] = r

# Bauru Q1 (block 2, 5 draws): stored interval and the same ANOVA redone on the stored matrices.
out["bauru_Q1_bloco2_ic95_voto2"] ={m: v2["celulas"][f"bauru_{m}"]["razao_sorteio_sobre_semente_IC95"]
                                     for m in ("gnn", "mlp")}
out["bauru_Q1_bloco2_recalc"] = {m: anova(v2["celulas"][f"bauru_{m}"]["matriz"]) for m in ("gnn", "mlp")}
json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({m: {k: out["modelos"][m][k] for k in ("lodo_min", "lodo_ic_inferior_min", "lodo_log_min", "lodo_log_ic_inferior_min", "lodo_log_n_ic_inferior_gt_1",
      "lodo_n_ic_inferior_gt_1", "rho_implicito_pela_razao_dp", "razao_dp_ic95_aprox_indep",
      "P_razao_ge_3_se_verdadeira_2_98_aprox")} | {"cochran": {k: out["modelos"][m]["cochran"][k] for k in ("C", "C_crit_005", "linha_max")},
      "log": {k: out["modelos"][m]["log_mae"][k] for k in ("rho", "ic95")},
      "completo_p": out["modelos"][m]["completo"]["p_unilateral_rho_le_1"]} for m in ("gnn", "mlp")},
      indent=1, default=float))
print("bauru", json.dumps(out["bauru_Q1_bloco2_ic95_voto2"]), {m: (out["bauru_Q1_bloco2_recalc"][m]["rho"], out["bauru_Q1_bloco2_recalc"][m]["ic95"]) for m in ("gnn","mlp")})
