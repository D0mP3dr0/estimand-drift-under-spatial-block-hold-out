"""Estimand Drift under Spatial Block Hold-Out (MDPI Mathematics, revision 1): Figure 2 (panels F3/F34: valid fraction and error drift).
Declared copy of the F34 figure of analysis/v3_12_gerar_figuras.py (digest of the original as run:
b89f647f80343112124786e2dadb1b09a7421d3e3c21d7675c048f99e988f077; the published file was cleaned, see PROVENANCE.md, so its digest differs
and is recorded instead of asserted). Same input records and filter (g = 10 km, b = 2 km, 20 draws in (a); status ok and finite MAE in (b)),
same cross-check of F4 against tables_figures/tables_v3-12/T3_deriva_erro_v3-12.csv (rewritten to F4_conferencia_T3_v4.json; the figure is
not written if the check fails).
Changes against the earlier figure (same data): one whisker rule (1.5 IQR) in both panels; every draw plotted as a point over the box; exact
zeros of panel (a) marked with 'x' at y = 0 and the count n/20 beside it (counts in F34_zeros_painel_a_v4.json); axis label 'valid fraction of
the test partition (-)'; fonts 8.5 pt (ticks, legend, city) and 9 pt (labels); width 394.36 pt, height <= 380 pt; vector PDF, fonttype 42.
Output directory: env R1_OUT (default results/figures_v4_rerun under the repository root)."""
import csv, hashlib, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import offset_copy
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # repository root
SCRIPT_REL = os.path.relpath(HERE, ROOT).replace(os.sep, "/")
OUT = os.environ.get("R1_OUT", os.path.join(ROOT, "results", "figures_v4_rerun"))
ORIGINAL = os.path.join(ROOT, "analysis", "v3_12_gerar_figuras.py")
ENTRADAS = {
    "F3": os.path.join(ROOT, "results", "inputs", "fracao_valida_val_treino_20_splits.json"),
    "F4_por_sorteio": os.path.join(ROOT, "results", "fase2", "_v3_2.1_3.1_parcial_16x60rnd.json"),
    "F4_conferencia_resumo_2.1": os.path.join(ROOT, "results", "fase2", "2.1_deriva_erro_baselines_16x60rnd.json"),
    "F4_conferencia_T3_csv": os.path.join(ROOT, "tables_figures", "tables_v3-12", "T3_deriva_erro_v3-12.csv"),
}
TEXTWIDTH_PT = 394.36
W_IN = TEXTWIDTH_PT / 72.0
ALTURA_PT = 376.0   # teto 380 pt
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e6e3"
FT_TICK, FT_LABEL, FT_LEG, FT_CITY = 8.5, 9.0, 8.5, 8.5
plt.rcParams.update({
    "font.family": "serif", "font.size": FT_LABEL, "axes.labelsize": FT_LABEL,
    "xtick.labelsize": FT_TICK, "ytick.labelsize": FT_TICK, "legend.fontsize": FT_LEG,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
QS = ["Q1", "Q2", "Q3", "Q4"]
WHIS = 1.5


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def posicoes_celulas():
    pos, grupo, x = {}, {}, 0.0
    for cid in CIDADES:
        xs = []
        for q in QS:
            pos[f"{cid}_{q}"] = x
            xs.append(x)
            x += 1.0
        grupo[cid] = (min(xs), max(xs))
        x += 0.9
    return pos, grupo, x - 0.9 - 1.0


def eixo_celulas(fig, ax, pos, grupo, x_ultimo):
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([c.split("_")[1] for c in pos])
    ax.set_xlim(-0.7, x_ultimo + 0.7)
    ax.set_xlabel("cell", labelpad=15)
    base = offset_copy(ax.get_xaxis_transform(), fig=fig, y=-17, units="points")
    for cid, (a, b) in grupo.items():
        ax.plot([a - 0.3, b + 0.3], [0, 0], transform=base, color=INK2, lw=0.8, clip_on=False)
        ax.annotate(cid.capitalize(), xy=((a + b) / 2, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -19), textcoords="offset points", ha="center", va="top", fontsize=FT_CITY, color=INK)


def caixa(ax, xc, v):
    """caixa com bigodes a 1,5 IQR (numpy.percentile linear, como ax.boxplot) e mediana; sem fliers (todos os sorteios sao pontos)."""
    ax.boxplot([v], positions=[xc], widths=0.6, patch_artist=True, whis=WHIS, showfliers=False, manage_ticks=False,
               medianprops=dict(color=INK, lw=1.5, zorder=6),
               boxprops=dict(facecolor=BLUE, alpha=0.45, edgecolor=INK2, lw=0.7, zorder=4),
               whiskerprops=dict(color=INK2, lw=0.9, zorder=3), capprops=dict(color=INK2, lw=0.9, zorder=3))


def fracoes_por_celula():
    d = load(ENTRADAS["F3"])
    por_celula = {}
    for r in d["resultados"]:
        if r.get("g_km") != 10.0 or r.get("b_km") != 2.0:
            continue
        por_celula.setdefault(f"{r['cidade']}_{r['Q']}", []).append(r["frac_valida"]["test"])
    return {k: np.array(v, float) for k, v in por_celula.items()}


def contar_zeros(fr):
    """celulas com minimo == 0 exato e com minimo < 1e-4 (painel (a): fracao valida do teste, 20 sorteios por celula)."""
    linhas = {}
    for k in sorted(fr):
        v = fr[k]
        linhas[k] = dict(n_sorteios=int(v.size), minimo=float(v.min()), n_zero_exato=int((v == 0.0).sum()),
                         n_menor_1e_4=int((v < 1e-4).sum()))
    z = [k for k, l in linhas.items() if l["minimo"] == 0.0]
    m = [k for k, l in linhas.items() if l["minimo"] < 1e-4]
    return dict(
        descricao="painel (a) da Fig. 2: fracao valida da particao de teste, g = 10 km, b = 2 km, sorteios do JSON de entrada F3 (filtro igual ao original)",
        fonte=ENTRADAS["F3"], fonte_sha256=sha256(ENTRADAS["F3"]),
        n_celulas=len(linhas), n_celulas_minimo_igual_a_zero_exato=len(z), celulas_minimo_zero_exato=z,
        n_celulas_minimo_menor_1e_4=len(m), celulas_minimo_menor_1e_4=m,
        total_sorteios_zero_exato=int(sum(l["n_zero_exato"] for l in linhas.values())),
        por_celula=linhas)


def desenha_fracao(ax, pos, fr, rng):
    zer = {}
    for k, v in fr.items():
        xc = pos[k]
        caixa(ax, xc, v)
        nz = int((v == 0.0).sum())
        vv = v[v != 0.0]
        ax.scatter(xc + rng.uniform(-0.22, 0.22, vv.size), vv, s=5, color=INK2, alpha=0.5, linewidths=0, zorder=7)
        if nz:
            zer[k] = nz
            ax.plot([xc], [0.0], marker="x", ms=5.5, mew=1.3, color=INK, ls="", zorder=8, clip_on=False)
            ax.annotate(f"{nz}/{v.size}", xy=(xc, 0.0), xytext=(0, -6 if len(zer) % 2 else -17), textcoords="offset points", ha="center", va="top",
                        fontsize=FT_TICK, color=INK, zorder=9)
    ax.set_ylim(-0.13, float(max(v.max() for v in fr.values())) + 0.03)
    ax.set_yticks(np.arange(0.0, float(max(v.max() for v in fr.values())) + 0.03, 0.1))
    ax.set_ylabel("valid fraction of the\ntest partition (–)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    return zer


def maes_validos_por_celula():
    d = load(ENTRADAS["F4_por_sorteio"])
    out = {}
    for k, cel in d["celulas"].items():
        v = []
        for p in cel["por_sorteio"]:
            m = p["mae_constante_teste"]["validos"]
            if p["status"] == "ok" and m is not None and np.isfinite(m):
                v.append(float(m))
        out[k] = np.array(v)
    return out


def conferir_f4(maes):
    tab = {r["celula"]: r for r in csv.DictReader(open(ENTRADAS["F4_conferencia_T3_csv"], encoding="utf-8"))}
    resumo = load(ENTRADAS["F4_conferencia_resumo_2.1"])["resumo"]
    tol, linhas, ok_total = 1e-9, [], True
    for k in sorted(maes):
        v, t = maes[k], tab[k]
        calc = dict(n=int(v.size), mediana=float(np.median(v)), min=float(v.min()), max=float(v.max()), dp_simples=float(np.std(v, ddof=1)))
        ref = dict(n=int(t["n_sorteios"]), mediana=float(t["const_validos_mediana"]), min=float(t["min"]), max=float(t["max"]), dp_simples=float(t["dp_simples"]))
        dif = {c: abs(calc[c] - ref[c]) for c in calc}
        ok = all(dif[c] <= tol for c in dif)
        ok_total &= ok
        linhas.append(dict(celula=k, ok=ok, calculado=calc, T3_csv=ref, dif_abs_max=max(dif.values())))
    dp_medio = float(np.mean([np.std(maes[k], ddof=1) for k in maes]))
    dp_ent = float(np.std(np.array([maes[k].mean() for k in sorted(maes)]), ddof=1))
    agreg = {"dp_medio_entre_sorteios_dB": dict(calculado=dp_medio, resumo_2_1=resumo["dp_medio_entre_sorteios_validos_dB_simples"], T3_tex_arredondado=8.017),
             "dp_entre_celulas_das_medias_dB": dict(calculado=dp_ent, resumo_2_1=resumo["dp_entre_celulas_das_medias_validos_dB"], T3_tex_arredondado=2.180, ddof=1)}
    for a in agreg.values():
        a["ok"] = bool(abs(a["calculado"] - a["resumo_2_1"]) <= tol and round(a["calculado"], 3) == a["T3_tex_arredondado"])
        ok_total &= a["ok"]
    rel = dict(descricao="Conferencia dos agregados do painel (b) (mesma conta do v3_12_gerar_figuras.conferir_f4)", tolerancia_abs=tol,
               n_celulas=len(linhas), tudo_confere=bool(ok_total), por_celula=linhas, agregados=agreg)
    with open(os.path.join(OUT, "F4_conferencia_T3_v4.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, indent=1, ensure_ascii=False)
    if not ok_total:
        raise SystemExit("conferencia contra T3 FALHOU; figura nao gravada")
    return rel


def desenha_deriva(ax, maes, pos, rng):
    for k, v in maes.items():
        xc = pos[k]
        caixa(ax, xc, v)
        ax.scatter(xc + rng.uniform(-0.22, 0.22, v.size), v, s=5, color=INK2, alpha=0.5, linewidths=0, zorder=7)
        ax.scatter([xc], [v.mean()], marker="D", s=20, color=ORANGE, edgecolor=INK, linewidths=0.6, zorder=8)
    ax.set_ylim(0, 60)
    ax.set_ylabel("valid-node mean absolute error\nof the constant predictor (dB)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    os.makedirs(OUT, exist_ok=True)
    maes = maes_validos_por_celula()
    conf = conferir_f4(maes)
    fr = fracoes_por_celula()
    zeros = contar_zeros(fr)
    pos, grupo, x_ultimo = posicoes_celulas()
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(W_IN, ALTURA_PT / 72.0), sharex=True, gridspec_kw=dict(height_ratios=[0.85, 1.15]))
    rng = np.random.default_rng(42)
    zer = desenha_fracao(axa, pos, fr, rng)
    desenha_deriva(axb, maes, pos, rng)
    axa.tick_params(axis="x", labelbottom=False)
    eixo_celulas(fig, axb, pos, grupo, x_ultimo)
    for ax, rot in ((axa, "(a)"), (axb, "(b)")):
        ax.text(0.0, 1.03, rot, transform=ax.transAxes, ha="left", va="bottom", fontsize=FT_LABEL, fontweight="bold", color=INK)
    handles = [
        Line2D([], [], color=INK, lw=1.5, label="median"),
        Patch(facecolor=BLUE, alpha=0.45, edgecolor=INK2, label="interquartile range (IQR)"),
        Line2D([], [], color=INK2, lw=0.9, marker="_", ms=6, label=f"whiskers: {WHIS} IQR"),
        Line2D([], [], color=INK2, marker="o", ms=3, ls="", alpha=0.6, label="split draw"),
        Line2D([], [], color=INK, marker="x", ms=5.5, mew=1.3, ls="", label="draw with no valid test node (a)"),
        Line2D([], [], color=ORANGE, marker="D", ms=4.5, mec=INK, mew=0.6, ls="", label="long-run mean (mean over draws) (b)"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=2, columnspacing=1.6, handletextpad=0.4, labelspacing=0.3)
    fig.subplots_adjust(left=0.14, right=0.995, top=0.855, bottom=0.115, hspace=0.20)
    fig.savefig(os.path.join(OUT, "F34_fracao_e_deriva_v4.pdf"))
    fig.savefig(os.path.join(OUT, "F34_fracao_e_deriva_v4.png"), dpi=300)
    plt.close(fig)
    zeros["x_desenhados_por_celula"] = zer
    zeros["script"] = SCRIPT_REL
    zeros["script_sha256"] = sha256(HERE)
    zeros["original"] = dict(arquivo=os.path.relpath(ORIGINAL, ROOT).replace(os.sep, "/"), sha256=sha256(ORIGINAL))
    zeros["entradas_sha256"] = {k: dict(caminho=os.path.relpath(p, ROOT).replace(os.sep, "/"), sha256=sha256(p)) for k, p in ENTRADAS.items()}
    zeros["largura_pt"], zeros["altura_pt"] = TEXTWIDTH_PT, ALTURA_PT
    zeros["fontes_pt"] = dict(ticks=FT_TICK, rotulos=FT_LABEL, legenda=FT_LEG, cidade=FT_CITY)
    zeros["bigode"] = f"{WHIS} IQR nos dois paineis"
    zeros["conferencia_painel_b_tudo_confere"] = conf["tudo_confere"]
    with open(os.path.join(OUT, "F34_zeros_painel_a_v4.json"), "w", encoding="utf-8") as f:
        json.dump(zeros, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: zeros[k] for k in ("n_celulas_minimo_igual_a_zero_exato", "celulas_minimo_zero_exato", "n_celulas_minimo_menor_1e_4", "celulas_minimo_menor_1e_4", "total_sorteios_zero_exato")}, indent=1))
    print({k: (l["minimo"], l["n_zero_exato"]) for k, l in zeros["por_celula"].items()})


if __name__ == "__main__":
    main()
