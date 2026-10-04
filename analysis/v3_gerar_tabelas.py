#!/usr/bin/env python3
"""Earlier table generator: builds tables T2 to T9 from the JSON result files in ENT.

Tables (LaTeX tabular body + CSV each):
  T2_retencao_v3          retention per city: nodal reference over 200 split draws
                          (files 1.8 and, when present, 1.8b for g = 5 km), the
                          closed-form retention law, their relative error, and the
                          lattice Monte Carlo value with its perimeter / thin-border
                          decomposition (percentage points);
  T3_deriva_erro_v3       per cell, MAE (dB) of the constant predictor over 60 random
                          split draws, on valid nodes and on all nodes, plus FSPL (a);
  T4_inversao_v3          per cell, counts of draws in which calibrated FSPL beats the
                          constant predictor on valid test nodes, and in which the
                          constant beats FSPL on all test nodes, calibrations (a), (b);
  T5_banda_p7_v3          per cell and propagation model, width of the quantile band
                          and relative position of the median inside it;
  T6_paridade_A3_v3       batch A3: GNN vs MLP sentinel parity, 8 cells x 5 seeds;
  T7_deriva_modelo_A4_v3  batch A4: GNN and MLP valid-node MAE over split draws;
  T8_paridade_A4_v3       batch A4: sentinel parity under split draws;
  T9_larguras_A4_v3       batch A4: sensitivity of the MLP to encoder widths.
No number is typed by hand: every value is read from a named field or is
arithmetic on such fields. A manifest (MANIFEST_tabelas_v3.json) records the
SHA-256 of every input, every output and of this script.

This is the earlier version of the table generator; analysis/v3_12_gerar_tabelas.py
checks its digest. Deterministic (no random numbers).

Usage: python analysis/v3_gerar_tabelas.py
"""
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path

B = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
FIO_R2 = Path("internal/notes")
OUT = B / "redacao" / "tables_v3"
OUT.mkdir(parents=True, exist_ok=True)
ENT = {
    "1.8": B / "fase1" / "1.8_referencia_nodal_200seeds.json",
    "1.8b": B / "fase1" / "1.8b_referencia_nodal_200seeds_g5.json",
    "MCv2": FIO_R2 / "internal/artifact_06.json",
    "2.1": B / "fase2" / "2.1_deriva_erro_baselines_16x60rnd.json",
    "3.1": B / "fase3" / "3.1_calibracao_validos_vs_mediana_16x60rnd.json",
    "3.1p": B / "fase2" / "_v3_2.1_3.1_parcial_16x60rnd.json",
    "3.2": B / "fase3" / "3.2_banda_p7_bordas_reais.json",
    "A3": B / "gpu" / "A3" / "agregado_A3.json",
    "A4": B / "gpu" / "A4" / "agregado_A4.json",
}
CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
QS = ["Q1", "Q2", "Q3", "Q4"]
NOME = {"bauru": "Bauru", "campinas": "Campinas", "lins": "Lins", "sorocaba": "Sorocaba"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def f4(x):
    return "--" if x is None else f"{x:.4f}"


def pct(x):
    """Relative difference as a signed percentage."""
    return "--" if x is None else f"{100*x:+.1f}"


def pp(x):
    """Signed value with two decimals (used for percentage points)."""
    return "--" if x is None else f"{x:+.2f}"


import re as _re

def _quebra(txt, larg=11):
    """Wrap a header cell into a \\celula{...} of lines with at most `larg` visible characters.

    LaTeX commands, braces and dollar signs are not counted; math spans are kept whole.
    """
    if len(_re.sub(r"\\[a-zA-Z]+|[{}$]", "", txt)) <= larg:
        return txt
    toks = _re.findall(r"\$[^$]*\$|\S+", txt)
    linhas, cur = [], ""
    for tk in toks:
        vis = len(_re.sub(r"\\[a-zA-Z]+|[{}$]", "", cur + " " + tk))
        if cur and vis > larg:
            linhas.append(cur); cur = tk
        else:
            cur = (cur + " " + tk).strip()
    linhas.append(cur)
    return "\\celula{" + " \\\\ \\relax ".join(linhas) + "}"


def _cabecalho(h):
    return " & ".join(_quebra(c.strip()) for c in h.split(" & "))


def _linha(ln):
    """Split body cells of the form 'median [min, max] (k/n)' over two lines inside \\celula{...}."""
    if ln.startswith("\\midrule"):
        return ln
    out = []
    for c in ln.split(" & "):
        c2 = c.strip()
        if "[" in c2 and len(c2) > 16:
            a, b = c2.split("[", 1)
            b = "[" + b
            if ")" in b and "(" in b:
                ip = b.rindex("(")
                c2 = "\\celula{" + (a.strip() + " " + b[ip:].strip()).strip() + " \\\\ \\relax " + b[:ip].strip() + "}"
            else:
                c2 = "\\celula{" + a.strip() + " \\\\ \\relax " + b.strip() + "}"
        out.append(c2)
    return " & ".join(out)


def escrever(nome, header_tex, linhas_tex, header_csv, linhas_csv, caption_note):
    """Write <nome>.tex (tabular body; header_tex = (column spec, header row)) and <nome>.csv; return both paths."""
    tex = OUT / f"{nome}.tex"
    with open(tex, "w", encoding="utf-8") as f:
        f.write("% gerado por scripts/v3_gerar_tabelas.py -- nao editar a mao\n")
        f.write("% " + caption_note + "\n")
        f.write("\\begin{tabular}{" + header_tex[0] + "}\n\\toprule\n")
        f.write(_cabecalho(header_tex[1]) + " \\\\\n\\midrule\n")
        for ln in linhas_tex:
            f.write(_linha(ln) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    csvp = OUT / f"{nome}.csv"
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header_csv)
        for r in linhas_csv:
            w.writerow(r)
    return [tex, csvp]


def gerar_t2():
    """Retention table: nodal reference vs closed-form law vs lattice Monte Carlo, per city and block side g."""
    d18 = load(ENT["1.8"])["por_celula"]
    mc = load(ENT["MCv2"])["por_config"]
    d18b = load(ENT["1.8b"])["por_celula"] if ENT["1.8b"].exists() else None
    linhas_tex, linhas_csv = [], []
    for cfg, dref, gtxt in (("g10b2", d18, "10"), ("g5b2", d18b, "5")):
        mccfg = mc.get({"g10b2": "g10b2_controle", "g5b2": "g5b2"}[cfg], {}).get("por_cidade", {})  # configuration key in the Monte Carlo file
        for cid in CIDADES:
            m = mccfg.get(cid, {})
            mc_te = (m.get("MC_C") or {}).get("ret_te_media"); mc_va = (m.get("MC_C") or {}).get("ret_va_media")
            dec = m.get("decomposicao_pp_te") or {}
            per = dec.get("efeito_perimetro_T_para_A2"); bor = dec.get("efeito_borda_fina_A2_para_C")
            if dref is None:
                pend = "\\pendente{1.8b}"
                linhas_tex.append(f"{gtxt} & {NOME[cid]} & {pend} & {pend} & {pend} & {pend} & {pend} & {f4(mc_te) if mc_te is not None else '--'} & {pend} & {pend} & {pp(per)} / {pp(bor)}")
                linhas_csv.append([gtxt, cid] + ["pendente"] * 9 + [mc_te, "pendente", "pendente", per, bor, 0])
                continue
            cels = [dref[f"{cid}_{q}"] for q in QS if f"{cid}_{q}" in dref]
            te_m = [c["nodal_te_media"] for c in cels]; te_dp = [c["nodal_te_dp"] for c in cels]
            va_m = [c["nodal_va_media"] for c in cels]
            lei_te = cels[0]["lei_te"]; lei_va = cels[0]["lei_va"]
            e_te = [c["erro_lei_vs_nodal_te"] for c in cels]; e_va = [c["erro_lei_vs_nodal_va"] for c in cels]
            # relative deviation of the Monte Carlo retention from the mean nodal retention of the 4 cells
            mc_vs_te = (mc_te / statistics.mean(te_m) - 1.0) if (mc_te is not None and te_m) else None
            mc_vs_va = (mc_va / statistics.mean(va_m) - 1.0) if (mc_va is not None and va_m) else None
            linhas_tex.append(
                f"{gtxt} & {NOME[cid]} & {f4(statistics.mean(te_m))} $\\pm$ {f4(statistics.mean(te_dp))} & {f4(statistics.mean(va_m))} & "
                f"{f4(lei_te)} / {f4(lei_va)} & {pct(min(e_te))} to {pct(max(e_te))} & {pct(min(e_va))} to {pct(max(e_va))} & "
                f"{f4(mc_te) if mc_te is not None else '--'} & {pct(mc_vs_te)} & {pct(mc_vs_va)} & {pp(per)} / {pp(bor)}")
            linhas_csv.append([gtxt, cid, statistics.mean(te_m), statistics.mean(te_dp), statistics.mean(va_m), lei_te, lei_va,
                               min(e_te), max(e_te), min(e_va), max(e_va), mc_te, mc_vs_te, mc_vs_va, per, bor, len(cels)])
    hdr = ("llccccccccc",
           "$g$ (km) & City & Nodal test (mean $\\pm$ sd) & Nodal val. & Law test / val. & Law vs.\\ nodal, test (\\%) & Law vs.\\ nodal, val.\\ (\\%) & Lattice MC test & MC vs.\\ nodal, test (\\%) & MC vs.\\ nodal, val.\\ (\\%) & Perimeter / thin-border (pp)")
    return escrever("T2_retencao_v3", hdr, linhas_tex,
                    ["g_km", "cidade", "nodal_te_media_4cel", "nodal_te_dp_media", "nodal_va_media_4cel", "lei_te", "lei_va",
                     "erro_lei_te_min", "erro_lei_te_max", "erro_lei_va_min", "erro_lei_va_max", "mc_v2_te_media", "mc_v2_vs_nodal200_te", "mc_v2_vs_nodal200_va", "perimetro_pp", "borda_fina_pp", "n_celulas"],
                    linhas_csv,
                    "T2: nodal = 200 split draws (1.8 / 1.8b), law = retention law eq:R with q_te = k_te/N; lattice Monte Carlo v2 per city; MC vs nodal-200 = arithmetic on named fields")


def gerar_t3():
    """Drift table: spread across split draws of the constant-predictor MAE per cell, and summary rows."""
    d = load(ENT["2.1"]); pc = d["por_celula"]; r = d["resumo"]
    linhas_tex, linhas_csv = [], []
    for cid in CIDADES:
        for q in QS:
            k = f"{cid}_{q}"; c = pc.get(k)
            if not c:
                continue
            cv = c["constante_validos"]; ct = c["constante_todos"]; fa = c.get("fspl_a_validos", {})
            linhas_tex.append(f"{NOME[cid]} {q} & {c['n_sorteios_ok']} & {f3(cv['mediana'])} & {f3(cv['min'])}--{f3(cv['max'])} & {f3(cv['dp_simples'])} & {f3(cv.get('dp_ponderado_por_n_validos_teste'))} & {f3(ct['mediana'])} & {f3(ct['dp_simples'])} & {f3(fa.get('mediana'))} & {f3(fa.get('dp_simples'))}")
            linhas_csv.append([k, c['n_sorteios_ok'], cv['mediana'], cv['min'], cv['max'], cv['dp_simples'], cv.get('dp_ponderado_por_n_validos_teste'), ct['mediana'], ct['dp_simples'], fa.get('mediana'), fa.get('dp_simples')])
    linhas_tex.append("\\midrule\nAll cells: sd between draws (mean of cells) & & & & " + f3(r['dp_medio_entre_sorteios_validos_dB_simples']) + " & " + f3(r['dp_medio_entre_sorteios_validos_dB_ponderado']) + " & & & &")
    linhas_tex.append("All cells: sd between cells of the per-cell means & & & & " + f3(r['dp_entre_celulas_das_medias_validos_dB']) + " & & & & &")
    hdr = ("lccccccccc",
           "Cell & Draws & Const.\\ valid: median & min--max & sd & sd (weighted) & Const.\\ all: median & sd & FSPL (a) valid: median & sd")
    return escrever("T3_deriva_erro_v3", hdr, linhas_tex,
                    ["celula", "n_sorteios", "const_validos_mediana", "min", "max", "dp_simples", "dp_ponderado", "const_todos_mediana", "const_todos_dp", "fspl_a_validos_mediana", "fspl_a_validos_dp"],
                    linhas_csv, "T3: MAE (dB) of the constant predictor over 60 random split draws (2.1_16x60rnd)")


def gerar_t4():
    """Inversion table, from the per-draw MAE file (3.1p).

    Per draw: fa/fb count FSPL (a)/(b) MAE < constant MAE on valid test nodes; ca/cb count
    constant MAE < FSPL (a)/(b) MAE on all test nodes. (a) calibrated on the contaminated
    training median, (b) calibrated on valid training nodes only.
    """
    d = load(ENT["3.1p"]); cel = d["celulas"]
    linhas_tex, linhas_csv = [], []
    tot = dict(nv=0, na=0, fa=0, fb=0, ca=0, cb=0)
    for cid in CIDADES:
        for q in QS:
            k = f"{cid}_{q}"
            if k not in cel:
                continue
            ss = [s for s in cel[k]["por_sorteio"] if s.get("status") == "ok"]
            fa = fb = nv = ca = cb = 0; mc = []; ma = []
            for s in ss:
                c_v = s["mae_constante_teste"]["validos"]; a_v = s["mae_modelo_a_contaminado_teste"]["fspl"]["validos"]; b_v = s["mae_modelo_b_validos_teste"]["fspl"]["validos"]
                if c_v is None or a_v is None or b_v is None:
                    continue  # no valid test node in this draw (MAE undefined): excluded
                nv += 1; fa += a_v < c_v; fb += b_v < c_v; mc.append(c_v); ma.append(a_v)
                c_t = s["mae_constante_teste"]["todos"]
                ca += c_t < s["mae_modelo_a_contaminado_teste"]["fspl"]["todos"]
                cb += c_t < s["mae_modelo_b_validos_teste"]["fspl"]["todos"]
            n = len(ss)
            for kk, vv in (("nv", nv), ("na", nv), ("fa", fa), ("fb", fb), ("ca", ca), ("cb", cb)):
                tot[kk] += vv
            linhas_tex.append(f"{NOME[cid]} {q} & {nv} & {fa}/{nv} & {fb}/{nv} & {ca}/{nv} & {cb}/{nv} & {f3(statistics.median(mc))} & {f3(statistics.median(ma))}")
            linhas_csv.append([k, nv, fa, fb, n, ca, cb, statistics.median(mc), statistics.median(ma)])
    T = tot
    linhas_tex.append("\\midrule\nAll cells & " + f"{T['nv']} & {T['fa']}/{T['nv']} ({100*T['fa']/T['nv']:.1f}\\%) & {T['fb']}/{T['nv']} ({100*T['fb']/T['nv']:.1f}\\%) & {T['ca']}/{T['nv']} ({100*T['ca']/T['nv']:.1f}\\%) & {T['cb']}/{T['nv']} ({100*T['cb']/T['nv']:.1f}\\%) & &")
    linhas_csv.append(["ALL", T["nv"], T["fa"], T["fb"], T["na"], T["ca"], T["cb"], None, None])
    hdr = ("lccccccc", "Cell & Draws & Valid nodes: FSPL lower (a) & Valid nodes: FSPL lower (b) & All nodes: constant lower (a) & All nodes: constant lower (b) & Constant, valid (median) & FSPL (a), valid (median)")
    return escrever("T4_inversao_v3", hdr, linhas_tex,
                    ["celula", "n_sorteios_com_validos", "fspl_menor_validos_a", "fspl_menor_validos_b", "n_sorteios", "constante_menor_todos_a", "constante_menor_todos_b", "const_validos_mediana", "fspl_a_validos_mediana"],
                    linhas_csv, "T4 (27/09 corrigida): per draw, which of the constant predictor and calibrated FSPL has the lower MAE, on valid test nodes and on all test nodes, 60 random draws (3.1p per-draw file); (a) contaminated training median, (b) valid training nodes only")


def gerar_t5():
    """Band table: per cell and model, quantile-band width (dB) and relative position of the median, (median - Q1-)/(Q1+ - Q1-)."""
    d = load(ENT["3.2"]); pc = d["por_celula"]
    linhas_tex, linhas_csv = [], []
    for cid in CIDADES:
        for q in QS:
            k = f"{cid}_{q}"; c = pc.get(k)
            if not c:
                continue
            m = c["modelos"]
            cells = [f"{NOME[cid]} {q}", f4(c['pi_bar_real_treino'])]
            row = [k, c['pi_bar_real_treino']]
            for mod in ("fspl", "hata_rural", "cost231_sub"):
                x = m[mod]
                cells.append(f"{x['largura_banda_dB']:.2f} / {x['posicao_relativa_0_a_1']:.2f}")
                row += [x['largura_banda_dB'], x['posicao_relativa_0_a_1'], x['mediana_dentro_da_banda']]
            linhas_tex.append(" & ".join(cells))
            linhas_csv.append(row)
    hdr = ("lcccc", "Cell & $\\pi$ (training) & FSPL: width (dB) / position & Hata rural: width / position & COST-231: width / position")
    return escrever("T5_banda_p7_v3", hdr, linhas_tex,
                    ["celula", "pi_treino", "fspl_largura_db", "fspl_posicao", "fspl_dentro", "hata_largura_db", "hata_posicao", "hata_dentro", "cost_largura_db", "cost_posicao", "cost_dentro"],
                    linhas_csv, "T5: quantile band of prop:median on real sentinel nodes (3.2); position = (median - Q1-)/(Q1+ - Q1-)")


def gerar_t6():
    """Batch A3 parity table: GNN vs MLP MAE by node population, median over 5 training seeds."""
    a = load(B / "gpu" / "A3" / "agregado_A3.json")
    linhas_tex, linhas_csv = [], []
    for k, c in a["celulas"].items():
        cid, q = k.split("_")
        g = [c["mae_gnn_por_seed"][s] for s in sorted(c["mae_gnn_por_seed"])]
        m = [c["mae_mlp_por_seed"][s] for s in sorted(c["mae_mlp_por_seed"])]
        d = c["distribuicao_gnn_menos_mlp"]
        def med(lst, key):
            return statistics.median([x[key] for x in lst])
        ds, dv, dt = d["mae_rssi_sentinela_db"], d["mae_rssi_validos_db"], d["mae_rssi_todos_db"]
        linhas_tex.append(
            f"{NOME[cid]} {q} & {med(g,'mae_rssi_sentinela_db'):.3f} / {med(m,'mae_rssi_sentinela_db'):.3f} & {c['tolerancia_db']:.3f} & "
            f"{ds['mediana']:+.3f} [{ds['min']:+.3f}, {ds['max']:+.3f}] ({ds['n_seeds_gnn_melhor']}/5) & "
            f"{med(g,'mae_rssi_validos_db'):.2f} / {med(m,'mae_rssi_validos_db'):.2f} & {dv['mediana']:+.2f} [{dv['min']:+.2f}, {dv['max']:+.2f}] ({dv['n_seeds_gnn_melhor']}/5) & "
            f"{med(g,'mae_rssi_todos_db'):.3f} / {med(m,'mae_rssi_todos_db'):.3f} & {dt['mediana']:+.3f} ({dt['n_seeds_gnn_melhor']}/5) & {'yes' if c['paridade_sentinela'] else 'no'}")
        linhas_csv.append([k, med(g,'mae_rssi_sentinela_db'), med(m,'mae_rssi_sentinela_db'), c['tolerancia_db'], ds['mediana'], ds['min'], ds['max'], ds['n_seeds_gnn_melhor'],
                           med(g,'mae_rssi_validos_db'), med(m,'mae_rssi_validos_db'), dv['mediana'], dv['min'], dv['max'], dv['n_seeds_gnn_melhor'],
                           med(g,'mae_rssi_todos_db'), med(m,'mae_rssi_todos_db'), dt['mediana'], dt['n_seeds_gnn_melhor'], c['paridade_sentinela']])
    linhas_tex.append("\\midrule\nCells in parity & & & " + str(a['H1']['n_celulas_em_paridade']) + " of 8 & & & & &")
    hdr = ("lcccccccc",
           "Cell & Sentinel MAE: GNN / MLP (median over seeds) & Tolerance & GNN$-$MLP, sentinel: median [min, max] (GNN lower / 5) & Valid MAE: GNN / MLP & GNN$-$MLP, valid & All-node MAE: GNN / MLP & GNN$-$MLP, all & Parity")
    return escrever("T6_paridade_A3_v3", hdr, linhas_tex,
                    ["celula", "sent_gnn_med", "sent_mlp_med", "tolerancia", "d_sent_med", "d_sent_min", "d_sent_max", "n_gnn_melhor_sent",
                     "val_gnn_med", "val_mlp_med", "d_val_med", "d_val_min", "d_val_max", "n_gnn_melhor_val", "todos_gnn_med", "todos_mlp_med", "d_todos_med", "n_gnn_melhor_todos", "paridade"],
                    linhas_csv, "T6: A3 batch, 8 cells x 5 training seeds, MAE of received power (dB) by population; tolerance = sd across seeds of the MLP sentinel MAE (criterio_A3_analise)")


def gerar_t7():
    """Batch A4 drift table: per cell, mean and SD over split draws of GNN and MLP valid-node MAE."""
    a = load(ENT["A4"])
    linhas_tex, linhas_csv = [], []
    for k, c in a["celulas"].items():
        cid, q = k.split("_")
        g, m = c["por_modelo"]["gnn"], c["por_modelo"]["mlp"]
        bdp = c["baselines_dp_entre_sorteios"]
        dv = c["paridade"]["distribuicao_gnn_menos_mlp"]["mae_rssi_validos_db"]
        gi, mi = g["inversao_vs_baselines_validos"], m["inversao_vs_baselines_validos"]
        linhas_tex.append(
            f"{NOME[cid]} {q} & {g['media_entre_sorteios']['mae_rssi_validos_db']:.2f} ({g['dp_entre_sorteios']['mae_rssi_validos_db']:.2f}) & "
            f"{m['media_entre_sorteios']['mae_rssi_validos_db']:.2f} ({m['dp_entre_sorteios']['mae_rssi_validos_db']:.2f}) & "
            f"{bdp['fspl']:.2f} / {bdp['hata_rural']:.2f} & "
            f"{dv['mediana']:+.2f} [{dv['min']:+.2f}, {dv['max']:+.2f}] ({dv['n_sorteios_gnn_melhor']}/{dv['n']})")
        linhas_csv.append([k, g['media_entre_sorteios']['mae_rssi_validos_db'], g['dp_entre_sorteios']['mae_rssi_validos_db'],
                           m['media_entre_sorteios']['mae_rssi_validos_db'], m['dp_entre_sorteios']['mae_rssi_validos_db'],
                           bdp['fspl'], bdp['hata_rural'], bdp['cost231_sub'],
                           dv['mediana'], dv['min'], dv['max'], dv['n_sorteios_gnn_melhor'], dv['n'],
                           gi['fspl']['n_sorteios_modelo_melhor'], gi['hata_rural']['n_sorteios_modelo_melhor'],
                           mi['fspl']['n_sorteios_modelo_melhor'], mi['hata_rural']['n_sorteios_modelo_melhor']])
    d = a["deriva_2_2"]
    linhas_tex.append("\\midrule\nSD across the four cells of the per-cell means & "
                      f"{d['gnn']['dp_entre_celulas_das_medias_validos_db']:.2f} & {d['mlp']['dp_entre_celulas_das_medias_validos_db']:.2f} & & ")
    hdr = ("lcccc",
           "Cell & GNN valid MAE: mean (SD over draws) & MLP valid MAE: mean (SD over draws) & SD over draws, FSPL / Hata & GNN$-$MLP, valid: median [min, max] (GNN lower / 5)")
    return escrever("T7_deriva_modelo_A4_v3", hdr, linhas_tex,
                    ["celula", "gnn_med_val", "gnn_dp_val", "mlp_med_val", "mlp_dp_val", "dp_fspl", "dp_hata", "dp_cost231",
                     "d_val_med", "d_val_min", "d_val_max", "n_gnn_melhor_val", "n_sorteios",
                     "n_gnn_lt_fspl", "n_gnn_lt_hata", "n_mlp_lt_fspl", "n_mlp_lt_hata"],
                    linhas_csv, "T7: A4 batch, Bauru Q1-Q4 x split draws 101-105, MAE of received power on valid nodes (dB); SD over draws vs SD over cells; baseline SD over the same draws (MAE over all test nodes, frozen script); paired GNN-MLP contrast. CSV also carries per-cell inversion counts (model MAE_valid < baseline MAE over all test nodes, criterio_A4 3.3 as written).")


def gerar_t8():
    """Batch A4 parity table: sentinel and all-node MAE of GNN vs MLP under split draws."""
    a = load(ENT["A4"])
    linhas_tex, linhas_csv = [], []
    for k, c in a["celulas"].items():
        cid, q = k.split("_")
        g, m = c["por_modelo"]["gnn"], c["por_modelo"]["mlp"]
        p = c["paridade"]
        ds, dt = p["distribuicao_gnn_menos_mlp"]["mae_rssi_sentinela_db"], p["distribuicao_gnn_menos_mlp"]["mae_rssi_todos_db"]
        gs, ms = g["media_entre_sorteios"]["mae_rssi_sentinela_db"], m["media_entre_sorteios"]["mae_rssi_sentinela_db"]
        gt, mt = g["media_entre_sorteios"]["mae_rssi_todos_db"], m["media_entre_sorteios"]["mae_rssi_todos_db"]
        linhas_tex.append(
            f"{NOME[cid]} {q} & {gs:.3f} / {ms:.3f} & {p['tolerancia_db']:.3f} & "
            f"{ds['mediana']:+.3f} [{ds['min']:+.3f}, {ds['max']:+.3f}] ({ds['n_sorteios_gnn_melhor']}/{ds['n']}) & "
            f"{gt:.3f} / {mt:.3f} & {dt['mediana']:+.3f} ({dt['n_sorteios_gnn_melhor']}/{dt['n']}) & {'yes' if p['paridade_sentinela'] else 'no'}")
        linhas_csv.append([k, gs, ms, p['tolerancia_db'], ds['mediana'], ds['min'], ds['max'], ds['n_sorteios_gnn_melhor'],
                           gt, mt, dt['mediana'], dt['n_sorteios_gnn_melhor'], p['paridade_sentinela']])
    linhas_tex.append("\\midrule\nCells in parity & & & " + str(a['n_celulas_em_paridade']) + " of 4 & & &")
    hdr = ("lcccccc",
           "Cell & Sentinel MAE: GNN / MLP (mean over draws) & Tolerance & GNN$-$MLP, sentinel: median [min, max] (GNN lower / 5) & All-node MAE: GNN / MLP & GNN$-$MLP, all & Parity")
    return escrever("T8_paridade_A4_v3", hdr, linhas_tex,
                    ["celula", "sent_gnn_mean", "sent_mlp_mean", "tolerancia", "d_sent_med", "d_sent_min", "d_sent_max", "n_gnn_melhor_sent",
                     "todos_gnn_mean", "todos_mlp_mean", "d_todos_med", "n_gnn_melhor_todos", "paridade"],
                    linhas_csv, "T8: A4 batch, Bauru Q1-Q4 x split draws 101-105 (training seed fixed), sentinel-node MAE (dB); tolerance = SD over draws of the MLP sentinel MAE (rule of criterio_A3_analise applied to draws)")


def gerar_t9():
    """Batch A4 width-sensitivity table: paired MAE difference between the two MLP encoder widths."""
    a = load(ENT["A4"])
    POPN = {"mae_rssi_validos_db": "valid, received power", "mae_rssi_sentinela_db": "sentinel, received power",
            "mae_rssi_todos_db": "all nodes, received power", "mae_pl_validos_db": "valid, path loss"}
    linhas_tex, linhas_csv = [], []
    for k, c in a["sensibilidade_larguras"].items():
        cid, q = k.split("_")
        for pop, r in c["resultado"].items():
            linhas_tex.append(f"{NOME[cid]} {q} & {POPN[pop]} & {r['mediana_delta_db']:+.3f} [{r['min']:+.3f}, {r['max']:+.3f}] & "
                              f"{r['dp_entre_seeds_larguras_novas_db']:.3f} & {'yes' if r['desprezivel'] else 'no'}")
            linhas_csv.append([k, pop, r['mediana_delta_db'], r['min'], r['max'], r['dp_entre_seeds_larguras_novas_db'], r['desprezivel'], r['n']])
    hdr = ("llccc", "Cell & Population & Wider encoder $-$ matched-capacity encoder, MAE (dB): median [min, max] over 5 seeds & SD over seeds, matched-capacity encoder & Negligible")
    return escrever("T9_larguras_A4_v3", hdr, linhas_tex,
                    ["celula", "populacao", "mediana_delta", "min", "max", "dp_novas", "desprezivel", "n"],
                    linhas_csv, "T9: MLP width sensitivity, 5 training seeds paired old vs new widths on the same partition (Bauru Q1, Lins Q1); negligible if |median| <= SD over seeds of the new widths (criterio_A4)")


def main():
    saidas = []
    for fn in (gerar_t2, gerar_t3, gerar_t4, gerar_t5, gerar_t6, gerar_t7, gerar_t8, gerar_t9):
        saidas += fn()
    man = {"entradas": {k: (sha(p) if p.exists() else None) for k, p in ENT.items()},
           "saidas": {str(p.relative_to(B)): sha(p) for p in saidas},
           "script_sha256": sha(Path(__file__))}
    with open(OUT / "MANIFEST_tabelas_v3.json", "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    for p in saidas:
        print("gravado", p)


if __name__ == "__main__":
    main()
