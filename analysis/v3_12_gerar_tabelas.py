#!/usr/bin/env python3
"""Generate the tables of the article from the result records (LaTeX tabular + CSV per table).

Every value comes from named fields of the JSON records; nothing is typed by hand. Tables
(file prefixes): T2 retention by cell (records 1.8, 1.8b); T3 and T3_resumo error drift of the
constant predictor, with b = 0 summary rows (2.1_16x60rnd, fase4/R1_b0_*); T4, T4_S2 and
T4_totais inversion counts with and without buffer (partial record 3.1 at b = 2,
fase4/R1_b0_por_sorteio.json at b = 0); T5 inclusion by class (fase4/R4_p_por_classe.json,
primary reading, nodal m); T6 design reference (fase4/R2_*, cross-checked against the
independent recalculation in fase4/votos_fismat_R2_R4/); T7 and T7b model drift of campaign
A4 (gpu/A4/agregado_A4.json); T8 design term (fase4/R3_termo_desenho.json); TA1, TA1a, TA1b
hyper-parameters (fase4/B7.2_hiperparametros/b7_2_folha_hiperparametros.json); TN notation.
Table T9 is written by the separate script v3_12h_gerar_T9_G1.py.

Before writing, the script re-derives several totals from the per-draw records and asserts
them (for instance the T4 counts at b = 2 must reproduce 799/796 and 629/729 of 904 draws),
and compares the columns that did not change with the CSVs of the previous table set
(prior_version_csv, tolerance 1e-12). The TA1 texts that paraphrase code expressions are
guarded by asserts on the exact strings of the hyper-parameter sheet. The notation table TN
is built from a LaTeX block of the manuscript that is not distributed, so TN cannot be
regenerated from the repository.

Outputs (in redacao_v3-12/tables_v3-12/ under B): <table>.tex (tabular only, no caption),
<table>.csv, CONFERENCIAS_tabelas_v3-12.json (all checks) and MANIFEST_tabelas_v3-12.json
(SHA-256 of inputs, script and outputs).

Usage: python v3_12_gerar_tabelas.py
"""
import csv
import hashlib
import json
import re as _re
import statistics
import sys
from decimal import Decimal
from pathlib import Path

B = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
SCRIPT_ORIGINAL = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts/v3_gerar_tabelas.py")
SHA_ORIGINAL_DECLARADO = "0dd90a0cb35faa5ff4d2a9bdbef6427a0e6fd6bfb29592f2a326c83188be8459"
OUT = B / "redacao_v3-12" / "tables_v3-12"
F4 = B / "fase4"
ANT = B / "redacao" / "tables_v3"  # previous table set, read only to cross-check unchanged values

ENT = {
    "1.8": B / "fase1" / "1.8_referencia_nodal_200seeds.json",
    "1.8b": B / "fase1" / "1.8b_referencia_nodal_200seeds_g5.json",
    "2.1": B / "fase2" / "2.1_deriva_erro_baselines_16x60rnd.json",
    "3.1p": B / "fase2" / "_v3_2.1_3.1_parcial_16x60rnd.json",
    "A4": B / "gpu" / "A4" / "agregado_A4.json",
    "R1_resumo": F4 / "R1_b0_resumo.json",
    "R1_por_sorteio": F4 / "R1_b0_por_sorteio.json",
    "R2_resumo": F4 / "R2_resumo.json",
    "R2_por_sorteio": F4 / "R2_por_sorteio.json",
    "R3": F4 / "R3_termo_desenho.json",
    "R4": F4 / "R4_p_por_classe.json",
    "B7.2": F4 / "B7.2_hiperparametros" / "b7_2_folha_hiperparametros.json",
    "bc_fismat": F4 / "votos_fismat_R2_R4" / "bc.json",
    "veredito_fismat": F4 / "votos_fismat_R2_R4" / "veredito.json",
    "F10_notacao": B / "_propostas_2026-10-01" / "blocos_formais_v3-12.tex",
    "T2_antiga_csv": ANT / "T2_retencao_v3.csv",
    "T3_antiga_csv": ANT / "T3_deriva_erro_v3.csv",
    "T4_antiga_csv": ANT / "T4_inversao_v3.csv",
    "T7_antiga_csv": ANT / "T7_deriva_modelo_A4_v3.csv",
}
CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
QS = ["Q1", "Q2", "Q3", "Q4"]
NOME = {"bauru": "Bauru", "campinas": "Campinas", "lins": "Lins", "sorocaba": "Sorocaba"}
Q1 = [f"{c}_Q1" for c in CIDADES]
CONF = {}  # checks, written to CONFERENCIAS_tabelas_v3-12.json


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


def mn(s):
    """Typographic minus for LaTeX: '-' -> '$-$'."""
    return s.replace("-", "$-$")


def fnum(x, nd, plus=False):
    """Fixed-decimal number with typographic minus; a value that rounds to zero is printed unsigned."""
    s = f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"
    if float(s) == 0.0:
        s = f"{0.0:.{nd}f}"
    return mn(s)


def faixa(vals, nd, fator=1.0, plus=False):
    """Range 'a to b' (T2); a single value when both ends coincide after formatting."""
    a = fnum(fator * min(vals), nd, plus)
    b = fnum(fator * max(vals), nd, plus)
    return a if a == b else f"{a} to {b}"


def faixa_traco(vals, nd, fator=1.0):
    """Range 'a--b' (T5); a single value when both ends coincide after formatting."""
    a = fnum(fator * min(vals), nd)
    b = fnum(fator * max(vals), nd)
    return a if a == b else f"{a}--{b}"


def num(x):
    """Number as stored in the source record, without rounding (TA1)."""
    if isinstance(x, float) and x == int(x):
        return str(int(x))
    return repr(x) if isinstance(x, float) else str(x)


def milhar(n):
    return f"{int(n):,}".replace(",", "\\,")


def pot10(x):
    """1e-05 -> '$10^{-5}$'; only exact powers of ten are accepted."""
    e = Decimal(repr(x)).adjusted()
    assert Decimal(repr(x)) == Decimal(1).scaleb(e), x
    return f"$10^{{{e}}}$"


def _quebra(txt, larg=11):
    """Split a long header cell into stacked lines (LaTeX celula macro) of about larg visible characters."""
    if len(_re.sub(r"\\[a-zA-Z]+|[{}$]", "", txt)) <= larg:
        return txt
    toks = _re.findall(r"\$[^$]*\$|\S+", txt)
    linhas, cur = [], ""
    for tk in toks:
        vis = len(_re.sub(r"\\[a-zA-Z]+|[{}$]", "", cur + " " + tk))
        if cur and vis > larg:
            linhas.append(cur)
            cur = tk
        else:
            cur = (cur + " " + tk).strip()
    linhas.append(cur)
    return "\\celula{" + " \\\\ \\relax ".join(linhas) + "}"


def _cabecalho(h):
    return " & ".join(_quebra(c.strip()) for c in h.split(" & "))


def _linha(ln):
    """Put the '[min, max]' part of a 'median [min, max] (k/n)' cell on a second line."""
    if ln.startswith("\\midrule"):
        return ln
    out = []
    for c in ln.split(" & "):
        c2 = c.strip()
        if "[" in c2 and len(c2) > 16 and c2.split("[", 1)[0].strip():
            a, b = c2.split("[", 1)
            b = "[" + b
            if ")" in b and "(" in b:
                ip = b.rindex("(")
                c2 = "\\celula{" + (a.strip() + " " + b[ip:].strip()).strip() + " \\\\ \\relax " + b[:ip].strip() + "}"
            else:
                c2 = "\\celula{" + a.strip() + " \\\\ \\relax " + b.strip() + "}"
        out.append(c2)
    return " & ".join(out)


def escrever(nome, header_tex, linhas_tex, header_csv, linhas_csv, nota, cru=False):
    """Write <nome>.tex (tabular with a two-line comment header) and <nome>.csv; cru=True skips cell splitting."""
    tex = OUT / f"{nome}.tex"
    with open(tex, "w", encoding="utf-8") as f:
        f.write("% gerado por scripts/v3_12_gerar_tabelas.py -- nao editar a mao\n")
        f.write("% " + nota + "\n")
        f.write("\\begin{tabular}{" + header_tex[0] + "}\n\\toprule\n")
        f.write(_cabecalho(header_tex[1]) + " \\\\\n\\midrule\n")
        for ln in linhas_tex:
            f.write((ln if cru else _linha(ln)) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    csvp = OUT / f"{nome}.csv"
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header_csv)
        for r in linhas_csv:
            w.writerow(r)
    return [tex, csvp]


def csv_antigo(chave):
    with open(ENT[chave], newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def confere_antigo(nome, chave, linhas_novas, mapa, chave_linha):
    """Assert that each mapped column equals the same column of the previous CSV (floats to 1e-12 relative)."""
    antigo = {r[chave_linha]: r for r in csv_antigo(chave)}
    n = 0
    for r in linhas_novas:
        a = antigo[str(r[chave_linha])]
        for col_nova, col_antiga in mapa.items():
            vn, va = r[col_nova], a[col_antiga]
            if isinstance(vn, (int, float)) and not isinstance(vn, bool):
                assert abs(float(va) - float(vn)) <= 1e-12 * max(1.0, abs(float(va))), (nome, r[chave_linha], col_nova, vn, va)
            else:
                assert str(vn) == str(va), (nome, r[chave_linha], col_nova, vn, va)
            n += 1
    CONF[f"{nome}_contra_{ENT[chave].name}"] = {"valores_comparados": n, "passou": True}


def gerar_t2():
    d18 = load(ENT["1.8"])["por_celula"]
    d18b = load(ENT["1.8b"])["por_celula"]
    linhas_tex, linhas_csv = [], []
    for gtxt, dref in (("10", d18), ("5", d18b)):
        for cid in CIDADES:
            cels = [dref[f"{cid}_{q}"] for q in QS if f"{cid}_{q}" in dref]
            te_m = [c["nodal_te_media"] for c in cels]
            te_dp = [c["nodal_te_dp"] for c in cels]
            va_m = [c["nodal_va_media"] for c in cels]
            lei_te, lei_va = cels[0]["lei_te"], cels[0]["lei_va"]
            e_te = [c["erro_lei_vs_nodal_te"] for c in cels]
            e_va = [c["erro_lei_vs_nodal_va"] for c in cels]
            linhas_tex.append(
                f"{gtxt} & {NOME[cid]} & {f4(statistics.mean(te_m))} $\\pm$ {f4(statistics.mean(te_dp))} & {f4(statistics.mean(va_m))} & "
                f"{f4(lei_te)} / {f4(lei_va)} & {faixa(e_te, 1, 100, True)} & {faixa(e_va, 1, 100, True)}")
            linhas_csv.append([int(gtxt), cid, statistics.mean(te_m), statistics.mean(te_dp), statistics.mean(va_m), lei_te, lei_va,
                               min(e_te), max(e_te), min(e_va), max(e_va), len(cels)])
    cab = ["g_km", "cidade", "nodal_te_media_4cel", "nodal_te_dp_media", "nodal_va_media_4cel", "lei_te", "lei_va",
           "erro_lei_te_min", "erro_lei_te_max", "erro_lei_va_min", "erro_lei_va_max", "n_celulas"]
    linhas_chave = [{**dict(zip(cab, r)), "chave": f"{r[0]}|{r[1]}"} for r in linhas_csv]
    antigo = {f"{r['g_km']}|{r['cidade']}": r for r in csv_antigo("T2_antiga_csv")}
    n = 0
    for r in linhas_chave:
        a = antigo[r["chave"]]
        for col in cab[2:]:
            assert abs(float(a[col]) - float(r[col])) <= 1e-12, (r["chave"], col)
            n += 1
    CONF["T2_contra_T2_retencao_v3.csv"] = {"valores_comparados": n, "passou": True}
    hdr = ("llccccc",
           "$g$ (km) & City & Nodal test (mean $\\pm$ sd) & Nodal val. & Law test / val. & Law vs.\\ nodal, test (\\%) & Law vs.\\ nodal, val.\\ (\\%)")
    return escrever("T2_retencao_v3-12", hdr, linhas_tex, cab, linhas_csv,
                    "T2: nodal = 200 split draws (1.8 / 1.8b), law = eq:R with q_te = k_te/N; sd between draws; ranges over the four quadrants of a city; "
                    "single value where both ends coincide at one decimal; columns Lattice MC / MC vs nodal / Perimeter removed (B8.3)")


def dp_ddof1_por_celula_b0():
    """Independent recount from the b = 0 per-draw record: mean over cells of the between-draw sd (ddof=1)
    of the valid-node MAE and of the valid fraction, over draws with a valid test node."""
    p = load(ENT["R1_por_sorteio"])["celulas"]
    mae, frac = [], []
    for k, c in p.items():
        s_ok = [s for s in c["sorteios"] if s["status"] == "ok" and s["mae_constante"]["validos"] is not None]
        mae.append(statistics.stdev([s["mae_constante"]["validos"] for s in s_ok]))
        frac.append(statistics.stdev([s["fracao_valida_teste"] for s in s_ok]))
    return statistics.mean(mae), statistics.mean(frac)


def dp_fracao_b2_parcial():
    """Same recount of the valid-fraction sd at b = 2, from the partial record 3.1."""
    cel = load(ENT["3.1p"])["celulas"]
    dps = []
    for k, c in cel.items():
        fr = []
        for s in c["por_sorteio"]:
            if s.get("status") != "ok" or s["mae_constante_teste"]["validos"] is None:
                continue
            fr.append(s["n_pop_teste"]["validos"] / s["n_pop_teste"]["todos"])
        dps.append(statistics.stdev(fr))
    return statistics.mean(dps)


def gerar_t3():
    d = load(ENT["2.1"])
    pc, r = d["por_celula"], d["resumo"]
    r1 = load(ENT["R1_resumo"])["medias_16_celulas"]
    # The b = 0 footer values and the b = 2 valid-fraction sd must match the direct recount.
    mae_b0, fr_b0 = dp_ddof1_por_celula_b0()
    fr_b2 = dp_fracao_b2_parcial()
    for nome, a, b in (("dp_mae_constante_validos_b0", r1["dp_mae_constante_validos_b0"], mae_b0),
                       ("dp_fracao_valida_teste_b0", r1["dp_fracao_valida_teste_b0"], fr_b0),
                       ("dp_fracao_valida_teste_b2", r1["dp_fracao_valida_teste_b2"], fr_b2),
                       ("dp_mae_constante_validos_b2", r1["dp_mae_constante_validos_b2_artefato_2.1"], r["dp_medio_entre_sorteios_validos_dB_simples"])):
        assert abs(a - b) <= 1e-9, (nome, a, b)
        CONF[f"T3_rodape_{nome}"] = {"resumo_R1_b0_ou_2.1": a, "recontagem_direta": b, "absdiff": abs(a - b), "passou": True}
    linhas_tex, linhas_csv = [], []
    for cid in CIDADES:
        for q in QS:
            k = f"{cid}_{q}"
            c = pc.get(k)
            if not c:
                continue
            cv, ct = c["constante_validos"], c["constante_todos"]
            linhas_tex.append(f"{NOME[cid]} {q} & {c['n_sorteios_ok']} & {f3(cv['mediana'])} & {f3(cv['min'])}--{f3(cv['max'])} & {f3(cv['dp_simples'])} & {f3(ct['mediana'])} & {f3(ct['dp_simples'])}")
            linhas_csv.append([k, c["n_sorteios_ok"], cv["mediana"], cv["min"], cv["max"], cv["dp_simples"], ct["mediana"], ct["dp_simples"]])
    cab = ["celula", "n_sorteios", "const_validos_mediana", "min", "max", "dp_simples", "const_todos_mediana", "const_todos_dp"]
    confere_antigo("T3", "T3_antiga_csv", [dict(zip(cab, r)) for r in linhas_csv],
                   {c: c for c in cab[1:]}, "celula")
    rod = [
        ("All cells: sd between draws (mean of cells)", f3(r["dp_medio_entre_sorteios_validos_dB_simples"]),
         "ALL_dp_entre_sorteios_media_das_celulas", r["dp_medio_entre_sorteios_validos_dB_simples"]),
        ("All cells: sd between draws, weighted by the valid test nodes of each draw (mean of cells)", f3(r["dp_medio_entre_sorteios_validos_dB_ponderado"]),
         "ALL_dp_entre_sorteios_ponderado_media_das_celulas", r["dp_medio_entre_sorteios_validos_dB_ponderado"]),
        ("All cells: sd between cells of the per-cell means", f3(r["dp_entre_celulas_das_medias_validos_dB"]),
         "ALL_dp_entre_celulas_das_medias", r["dp_entre_celulas_das_medias_validos_dB"]),
        ("All cells, no buffer ($b=0$): sd between draws (mean of cells)", f3(r1["dp_mae_constante_validos_b0"]),
         "ALL_b0_dp_entre_sorteios_media_das_celulas", r1["dp_mae_constante_validos_b0"]),
        ("All cells: sd between draws of the valid fraction of the test set, $b=2$ / $b=0$ (mean of cells; unitless)",
         f"{f3(r1['dp_fracao_valida_teste_b2'])} / {f3(r1['dp_fracao_valida_teste_b0'])}",
         "ALL_dp_fracao_valida_teste_b2", r1["dp_fracao_valida_teste_b2"]),
    ]
    for i, (rot, txt, chave, val) in enumerate(rod):
        pre = "\\midrule\n" if i == 0 else ""
        linhas_tex.append(pre + f"\\multicolumn{{4}}{{p{{0.5\\linewidth}}}}{{\\raggedright {rot}}} & {txt} & &")
        linhas_csv.append([chave, None, None, None, None, val, None, None])
    linhas_csv.append(["ALL_dp_fracao_valida_teste_b0", None, None, None, None, r1["dp_fracao_valida_teste_b0"], None, None])
    hdr = ("lcccccc",
           "Cell & Draws & Constant, valid: median (dB) & min--max (dB) & sd (dB) & Constant, all: median (dB) & sd (dB)")
    return escrever("T3_deriva_erro_v3-12", hdr, linhas_tex, cab, linhas_csv,
                    "T3: MAE (dB) of the constant predictor over 60 random split draws (2.1_16x60rnd); footnote rows with b = 0 from fase4/R1_b0_resumo.json "
                    "(sd ddof=1 between draws with a valid test node, mean of the 16 cells); FSPL (a) columns and sd (weighted) column removed", cru=True)


def gerar_t3_resumo():
    """Two-column table (quantity; value with unit) with the summary rows of T3, same fields and formats."""
    d = load(ENT["2.1"])
    r = d["resumo"]
    r1 = load(ENT["R1_resumo"])["medias_16_celulas"]
    itens = [
        ("sd_entre_sorteios_b2_media_das_celulas", "Between-draw standard deviation of the valid-node MAE, $b=2$~km, mean over the cells (dB)",
         r["dp_medio_entre_sorteios_validos_dB_simples"], f3),
        ("sd_entre_sorteios_b2_ponderado_media_das_celulas", "Between-draw standard deviation of the valid-node MAE, $b=2$~km, weighted by the valid test nodes of each draw, mean over the cells (dB)",
         r["dp_medio_entre_sorteios_validos_dB_ponderado"], f3),
        ("sd_entre_celulas_das_medias", "Standard deviation between the cells' mean errors, $b=2$~km (dB)",
         r["dp_entre_celulas_das_medias_validos_dB"], f3),
        ("sd_entre_sorteios_b0_media_das_celulas", "Between-draw standard deviation of the valid-node MAE, no buffer ($b=0$), mean over the cells (dB)",
         r1["dp_mae_constante_validos_b0"], f3),
        ("sd_fracao_valida_b2", "Between-draw standard deviation of the valid fraction of the test set, $b=2$~km, mean over the cells (dimensionless)",
         r1["dp_fracao_valida_teste_b2"], f3),
        ("sd_fracao_valida_b0", "Between-draw standard deviation of the valid fraction of the test set, no buffer ($b=0$), mean over the cells (dimensionless)",
         r1["dp_fracao_valida_teste_b0"], f3),
    ]
    linhas_tex = [f"{rot} & {fmt(v)}" for _, rot, v, fmt in itens]
    linhas_csv = [[ch, rot.replace("$", "").replace("~", " "), v, ("dB" if "(dB)" in rot else "dimensionless")] for ch, rot, v, _ in itens]
    t3 = {row["celula"]: row for row in csv.DictReader(open(OUT / "T3_deriva_erro_v3-12.csv", encoding="utf-8"))}
    for ch, ch3 in (("sd_entre_sorteios_b2_media_das_celulas", "ALL_dp_entre_sorteios_media_das_celulas"),
                    ("sd_entre_sorteios_b2_ponderado_media_das_celulas", "ALL_dp_entre_sorteios_ponderado_media_das_celulas"),
                    ("sd_entre_celulas_das_medias", "ALL_dp_entre_celulas_das_medias"),
                    ("sd_entre_sorteios_b0_media_das_celulas", "ALL_b0_dp_entre_sorteios_media_das_celulas"),
                    ("sd_fracao_valida_b2", "ALL_dp_fracao_valida_teste_b2"),
                    ("sd_fracao_valida_b0", "ALL_dp_fracao_valida_teste_b0")):
        if ch3 in t3:
            a = [x for x in itens if x[0] == ch][0][2]
            assert abs(a - float(t3[ch3]["dp_simples"])) <= 1e-12, (ch, a, t3[ch3])
    CONF["T3_resumo_igual_as_linhas_resumo_da_T3"] = {"linhas": len(itens), "passou": True}
    return escrever("T3_resumo_v3-12", (">{\\raggedright\\arraybackslash}p{0.74\\linewidth}c", "Quantity & Value"), linhas_tex,
                    ["chave", "quantidade", "valor", "unidade"], linhas_csv,
                    "T3_resumo: linhas-resumo da T3 (2.1_16x60rnd resumo; R1_b0_resumo.json medias_16_celulas); a T3 completa por celula fica fora do pacote", cru=True)


def contar_inversao(trios):
    """Inversion counts over draws given (c_v, a_v, b_v, c_t, a_t, b_t) MAEs per draw.

    Only draws with a valid test node enter; fa/fb: FSPL (a)/(b) lower than the constant on
    valid nodes; ca/cb: constant lower than FSPL (a)/(b) on all nodes.
    """
    nv = fa = fb = ca = cb = 0
    for c_v, a_v, b_v, c_t, a_t, b_t in trios:
        if c_v is None or a_v is None or b_v is None:
            continue
        nv += 1
        fa += a_v < c_v
        fb += b_v < c_v
        ca += c_t < a_t
        cb += c_t < b_t
    return dict(nv=nv, fa=fa, fb=fb, ca=ca, cb=cb)


def trios_b2(sorteios):
    for s in sorteios:
        if s.get("status") != "ok":
            continue
        yield (s["mae_constante_teste"]["validos"], s["mae_modelo_a_contaminado_teste"]["fspl"]["validos"],
               s["mae_modelo_b_validos_teste"]["fspl"]["validos"], s["mae_constante_teste"]["todos"],
               s["mae_modelo_a_contaminado_teste"]["fspl"]["todos"], s["mae_modelo_b_validos_teste"]["fspl"]["todos"])


def trios_b0(sorteios):
    for s in sorteios:
        if s.get("status") != "ok":
            continue
        yield (s["mae_constante"]["validos"], s["mae_fspl_a_contaminado"]["validos"], s["mae_fspl_b_validos"]["validos"],
               s["mae_constante"]["todos"], s["mae_fspl_a_contaminado"]["todos"], s["mae_fspl_b_validos"]["todos"])


def gerar_t4():
    cel = load(ENT["3.1p"])["celulas"]
    # Check before writing: the recount on the b = 2 partial record must reproduce the stored totals.
    tot2 = dict(nv=0, fa=0, fb=0, ca=0, cb=0)
    por_celula = {}
    for cid in CIDADES:
        for q in QS:
            k = f"{cid}_{q}"
            if k in cel:
                x = contar_inversao(trios_b2(cel[k]["por_sorteio"]))
                por_celula[k] = x
                for kk in tot2:
                    tot2[kk] += x[kk]
    ref_csv = [r for r in csv_antigo("T4_antiga_csv") if r["celula"] == "ALL"][0]
    r1 = load(ENT["R1_resumo"])["inversoes_constante_gt_fspl"]
    ref_resumo_b2 = r1["b2"]
    esperado = dict(nv=int(ref_csv["n_sorteios_com_validos"]), fa=int(ref_csv["fspl_menor_validos_a"]), fb=int(ref_csv["fspl_menor_validos_b"]),
                    ca=int(ref_csv["constante_menor_todos_a"]), cb=int(ref_csv["constante_menor_todos_b"]))
    esperado_enunciado = dict(nv=904, fa=799, fb=796, ca=629, cb=729)
    assert tot2 == esperado == esperado_enunciado, (tot2, esperado)
    assert ref_resumo_b2["a_validos"] == tot2["fa"] and ref_resumo_b2["b_validos"] == tot2["fb"]
    assert tot2["nv"] - ref_resumo_b2["a_todos"] == tot2["ca"] and tot2["nv"] - ref_resumo_b2["b_todos"] == tot2["cb"]
    CONF["T4_recontagem_b2_antes_de_gravar"] = {
        "fonte": str(ENT["3.1p"]), "recontagem": tot2, "esperado_T4_v3-11g_csv": esperado,
        "esperado_enunciado_tarefa": "629/729 (constante menor em todos, a/b) e 799/796 (FSPL menor nos validos, a/b) de 904",
        "esperado_R1_b0_resumo_b2": {"fspl_menor_validos_a": ref_resumo_b2["a_validos"], "fspl_menor_validos_b": ref_resumo_b2["b_validos"],
                                     "constante_maior_todos_a": ref_resumo_b2["a_todos"], "constante_maior_todos_b": ref_resumo_b2["b_todos"],
                                     "n_nv": tot2["nv"]},
        "passou": True}
    # Recount at b = 0.
    p0 = load(ENT["R1_por_sorteio"])["celulas"]
    tot0 = dict(nv=0, fa=0, fb=0, ca=0, cb=0)
    for k in p0:
        x = contar_inversao(trios_b0(p0[k]["sorteios"]))
        for kk in tot0:
            tot0[kk] += x[kk]
    ref0 = r1["b0"]
    assert tot0["nv"] == load(ENT["R1_resumo"])["sorteios_com_no_valido"]["b0"]["n_sorteios_com_no_valido_total"]
    assert tot0["fa"] == ref0["a_validos"] and tot0["fb"] == ref0["b_validos"]
    assert tot0["nv"] - ref0["a_todos"] == tot0["ca"] and tot0["nv"] - ref0["b_todos"] == tot0["cb"]
    CONF["T4_recontagem_b0"] = {"fonte": str(ENT["R1_por_sorteio"]), "recontagem": tot0,
                                "contra_R1_b0_resumo.inversoes_constante_gt_fspl.b0": {
                                    "fspl_menor_validos_a": ref0["a_validos"], "fspl_menor_validos_b": ref0["b_validos"],
                                    "constante_maior_todos_a": ref0["a_todos"], "constante_maior_todos_b": ref0["b_todos"], "n_nv": tot0["nv"]},
                                "passou": True}
    linhas_tex, linhas_csv = [], []
    for cid in CIDADES:
        for q in QS:
            k = f"{cid}_{q}"
            if k not in cel:
                continue
            ss = [s for s in cel[k]["por_sorteio"] if s.get("status") == "ok"]
            x = por_celula[k]
            mc, ma = [], []
            for s in ss:
                c_v = s["mae_constante_teste"]["validos"]
                a_v = s["mae_modelo_a_contaminado_teste"]["fspl"]["validos"]
                b_v = s["mae_modelo_b_validos_teste"]["fspl"]["validos"]
                if c_v is None or a_v is None or b_v is None:
                    continue
                mc.append(c_v)
                ma.append(a_v)
            nv = x["nv"]
            linhas_tex.append(f"{NOME[cid]} {q} & {nv} & {x['fa']}/{nv} & {x['fb']}/{nv} & {x['ca']}/{nv} & {x['cb']}/{nv} & {f3(statistics.median(mc))} & {f3(statistics.median(ma))}")
            linhas_csv.append([k, nv, x["fa"], x["fb"], len(ss), x["ca"], x["cb"], statistics.median(mc), statistics.median(ma)])
    cab = ["celula", "n_sorteios_com_validos", "fspl_menor_validos_a", "fspl_menor_validos_b", "n_sorteios", "constante_menor_todos_a", "constante_menor_todos_b",
           "const_validos_mediana", "fspl_a_validos_mediana"]
    confere_antigo("T4", "T4_antiga_csv", [dict(zip(cab, r)) for r in linhas_csv], {c: c for c in cab[1:]}, "celula")

    def linha_all(rotulo, T, medianas=True):
        pc = lambda v: f"{100 * v / T['nv']:.1f}\\%"
        return (f"{rotulo} & {T['nv']} & {T['fa']}/{T['nv']} ({pc(T['fa'])}) & {T['fb']}/{T['nv']} ({pc(T['fb'])}) & "
                f"{T['ca']}/{T['nv']} ({pc(T['ca'])}) & {T['cb']}/{T['nv']} ({pc(T['cb'])})" + (" & &" if medianas else ""))
    linhas_tex.append("\\midrule\n" + linha_all("All cells", tot2))
    linhas_tex.append(linha_all("\\celula{All cells, \\\\ \\relax no buffer}", tot0))
    linhas_csv.append(["ALL", tot2["nv"], tot2["fa"], tot2["fb"], tot2["nv"], tot2["ca"], tot2["cb"], None, None])
    linhas_csv.append(["ALL_b0_sem_buffer", tot0["nv"], tot0["fa"], tot0["fb"], tot0["nv"], tot0["ca"], tot0["cb"], None, None])
    hdr = ("lccccccc", "Cell & Draws & Valid nodes: FSPL lower (a) & Valid nodes: FSPL lower (b) & All nodes: constant lower (a) & All nodes: constant lower (b) & Constant, valid (median, dB) & FSPL (a), valid (median, dB)")
    arq = escrever("T4_inversao_v3-12", hdr, linhas_tex, cab, linhas_csv,
                   "T4 (v3-11g, intact) + last row 'All cells, no buffer': same count per draw (draws with a valid test node) on fase4/R1_b0_por_sorteio.json (b = 0); "
                   "(a) contaminated training median, (b) valid training nodes only")
    # Variants with the calibration labels written out ((a) = first, (b) = second calibration); same values.
    PRIM, SEG = "first calibration", "second calibration"
    hdr_s = ("lccccccc", f"Cell & Draws & Valid nodes: FSPL lower ({PRIM}) & Valid nodes: FSPL lower ({SEG}) & All nodes: constant lower ({PRIM}) & All nodes: constant lower ({SEG}) "
                         f"& Constant, valid (median, dB) & FSPL ({PRIM}), valid (median, dB)")
    nota_cal = (f"{PRIM} = (a): contaminated training median (all training nodes); {SEG} = (b): valid training nodes only")
    arq += escrever("T4_S2_v3-12", hdr_s, linhas_tex, cab, linhas_csv,
                    "T4_S2: the T4 above with calibration labels written out, same values (supplementary Table S2); " + nota_cal)
    tex_tot = [linha_all("All cells", tot2, False), linha_all("All cells, no buffer", tot0, False)]
    csv_tot = [r[:7] for r in linhas_csv[-2:]]
    hdr_t = ("lccccc", f"Cells & Draws & Valid nodes: FSPL lower ({PRIM}) & Valid nodes: FSPL lower ({SEG}) & All nodes: constant lower ({PRIM}) & All nodes: constant lower ({SEG})")
    arq += escrever("T4_totais_v3-12", hdr_t, tex_tot, cab[:7], csv_tot,
                    "T4_totais: only the two total rows of the T4 (b = 2 km, and no buffer b = 0), count columns only; " + nota_cal, cru=True)
    CONF["T4_totais_e_S2"] = {"totais_b2": tot2, "totais_b0": tot0, "mesmos_valores_que_a_T4": True, "passou": True}
    return arq


def gerar_t5():
    """T5: inclusion by nodal class m in the four Q1 cells, primary reading; ranges over cells."""
    d = load(ENT["R4"])
    assert d["leitura_primaria"] == "L_no_a_no"
    pc = d["por_celula"]
    classes = {}
    tot = []
    for k in Q1:
        cl = {x["classe"]: x for x in pc[k]["classes_nodais"]}
        n_total = pc[k]["fidelidade"]["n_nos"]
        for m in range(4):
            x = cl[f"m={m}|todas"]
            assert x["m_valores"] == [m] and x["definicao_m"] == "nodal"
            classes.setdefault(m, []).append(x)
        t = cl["TOTAL|todos os nos"]
        assert abs(sum(cl[f"m={m}|todas"]["fracao_nos"] for m in range(4)) - 1.0) < 1e-12
        tot.append((1 - t["leituras"]["L_no_a_no"]["n_nos_so_cotas"] / n_total, t["leituras"]["L_no_a_no"]["n_nos_so_cotas"] / n_total, t["frequencia_media_nodal"]))
    linhas_tex, linhas_csv = [], []
    for m in range(4):
        xs = classes[m]
        lo = {x["cota_inferior_por_m"][str(m)] for x in xs}
        hi = {x["cota_superior_por_m"][str(m)] for x in xs}
        assert len(lo) == 1 and len(hi) == 1, (m, lo, hi)
        lo, hi = lo.pop(), hi.pop()
        so_cotas = [x["leituras"]["L_no_a_no"]["n_nos_so_cotas"] / x["n_nos"] for x in xs]
        frac = [x["fracao_nos"] for x in xs]
        freq = [x["frequencia_media_nodal"] for x in xs]
        epf = [x["ep_mc_bootstrap_sorteios"] for x in xs]
        # Display rule: closed value (attained lower bound) if under half of the class has bounds only, else [lower, upper].
        if max(so_cotas) < 0.5:
            fechado = f"{lo:.4f}" if lo != hi else f"{lo:.4f}"
            forma = "valor_fechado"
        else:
            assert min(so_cotas) >= 0.5
            fechado = f"[{lo:.4f}, {hi:.4f}]"
            forma = "cotas"
        linhas_tex.append(f"$m={m}$ & {faixa_traco(frac, 1, 100)} & {faixa_traco(freq, 4)} & {faixa_traco(epf, 4)} & {fechado} & {faixa_traco(so_cotas, 1, 100)}")
        linhas_csv.append([m, forma, lo, hi] + frac + freq + so_cotas + epf)
    fa = [t[0] for t in tot]
    sc = [t[1] for t in tot]
    linhas_tex.append("\\midrule\nClosed form available & " + faixa_traco(fa, 1, 100) + " & & & & " + faixa_traco(sc, 1, 100))
    linhas_csv.append(["closed_form_available", "", None, None] + fa + [None] * 4 + sc + [None] * 4)
    cab = ["classe_m", "forma_exibida", "cota_inferior", "cota_superior"] + [f"fracao_nos_{k}" for k in Q1] + [f"frequencia_200_{k}" for k in Q1] + [f"fracao_da_classe_so_cotas_{k}" for k in Q1] + [f"ep_mc_frequencia_{k}" for k in Q1]
    hdr = ("lccccc", "Class $m(i)$ & Nodes in class (\\%) & Inclusion frequency $p_{i\\mid\\mathrm{te}}$ over 200 draws & Monte Carlo standard error of the frequency & Closed form or bounds & Bounds only (\\% of class)")
    return escrever("T5_inclusao_por_classe_v3-12", hdr, linhas_tex, cab, linhas_csv,
                    "T5: R4_p_por_classe.json, m nodal, primary reading L_no_a_no (closed form where m=0 or condition (iii) holds at the node); ranges over the four Q1 cells; "
                    "frequency = mean over the nodes of the class of the inclusion frequency given a test block, 200 draws; value shown = closed form (lower bound, attained) if under half "
                    "of the class has bounds only, else the interval [lower, upper] of Proposition inclusion (ii); last row: share of all nodes with closed form")


PRED = {"constante": "Constant", "fspl_calibrado_b": "FSPL (second calibration)"}
POP = {"todos": "All nodes", "validos": "Valid nodes"}


def gerar_t6():
    """T6: bias and RMSE of the block hold-out H, RMSE of the random sample A and n/deff, deff = Var(H)/Var(A)."""
    res = load(ENT["R2_resumo"])["resumo_por_celula"]
    por = load(ENT["R2_por_sorteio"])["celulas"]
    bc = load(ENT["bc_fismat"])["b"]
    ver = load(ENT["veredito_fismat"])["b"]
    linhas_tex, linhas_csv = [], []
    nde = {"todos": [], "validos": []}
    for k in Q1:
        info = por[k]["info_por_sorteio"]
        n_medio = {"todos": statistics.mean(info["n_te_b2"]), "validos": statistics.mean(info["n_te_b2_validos"])}
        primeiro_cel = True
        for pred in ("constante", "fspl_calibrado_b"):
            primeiro_pred = True
            for pop in ("todos", "validos"):
                x = res[k][pred][pop]
                H, A = x["estimadores"]["H"], x["estimadores"]["A"]
                deff = H["variancia"] / A["variancia"]
                assert abs(deff - x["razoes"]["deff_H_sobre_A"]) <= 1e-9 * deff
                b = bc[f"{k}|{pred}|{pop}"]
                assert abs(deff - b["deff_calc"]) <= 1e-9 * deff and abs(H["variancia"] - b["varH"]) <= 1e-12 and abs(A["variancia"] - b["varA"]) <= 1e-15
                n = n_medio[pop]
                nde[pop].append(n / deff)
                pre = "\\midrule\n" if (primeiro_cel and k != Q1[0]) else ""
                cel = f"{NOME[k.split('_')[0]]} Q1" if primeiro_cel else ""
                prd = PRED[pred] if primeiro_pred else ""
                linhas_tex.append(f"{pre}{cel} & {prd} & {POP[pop]} & {fnum(H['vies'], 3)} ({fnum(H['ep_mc_vies'], 3)}) & {fnum(H['reqm'], 3)} & {fnum(A['reqm'], 3)} & {n / deff:.1f}")
                linhas_csv.append([k, pred, pop, H["vies"], H["ep_mc_vies"], H["reqm"], A["reqm"], H["n"], n, deff, n / deff])
                primeiro_cel = primeiro_pred = False
    # Cross-check of the n/deff ranges against the independent recalculation (mean number of scored nodes; 1 decimal).
    def faixa_txt(t):
        return tuple(float(v) for v in _re.match(r"([\d.]+)-([\d.]+)", t).groups())
    for pop, chave in (("todos", "todos_n/deff"), ("validos", "validos_n/deff")):
        lo, hi = faixa_txt(ver[chave])
        assert abs(min(nde[pop]) - lo) <= 0.05 and abs(max(nde[pop]) - hi) <= 0.05, (pop, min(nde[pop]), max(nde[pop]), lo, hi)
        CONF[f"T6_n_sobre_deff_{pop}"] = {"recomputado_min_max": [min(nde[pop]), max(nde[pop])], "veredito_fismat": ver[chave], "passou": True}
    CONF["T6_deff_contra_bc.json"] = {"celulas_pred_pop_comparadas": 16, "campos": "deff_calc, varH, varA", "passou": True,
                                      "nota": "the n_sobre_deff_nH field of bc.json uses n = number of split draws (nH = 199 or 200), not the number of nodes; the n/deff of this table uses the mean number of scored nodes "
                                              "(n_te_b2; n_te_b2_validos of R2_por_sorteio.json), which reproduces the ranges of the independent recalculation record (b): 14.9-22.0 (all nodes) and 1.1-4.8 (valid nodes)"}
    cab = ["celula", "preditor", "populacao", "vies_holdout_dB", "ep_mc_vies_dB", "reqm_holdout_dB", "reqm_amostra_aleatoria_dB", "n_sorteios_definidos_holdout",
           "n_medio_nos_pontuados", "deff_var_H_sobre_var_A", "n_sobre_deff"]
    hdr = ("lllcccc", "Cell & Predictor & Population & Block hold-out bias, dB (SE) & Block hold-out RMSE (dB) & Random-sample RMSE (dB) & Effective size $n/\\mathrm{deff}$")
    return escrever("T6_referencia_desenho_v3-12", hdr, linhas_tex, cab, linhas_csv,
                    "T6: R2_resumo.json, hold-out in blocks (g=10 km, b=2 km, estimator H), 200 draws; bias with Monte Carlo standard error over the draws; RMSE of the hold-out and of a simple random "
                    "sample of the same number of nodes (estimator A); deff = Var(H)/Var(A); n = mean number of scored nodes (R2_por_sorteio.json: n_te_b2 for all nodes, n_te_b2_validos for valid nodes); no Hajek", cru=True)


def gerar_t7():
    a = load(ENT["A4"])
    linhas_tex, linhas_csv = [], []
    for k, c in a["celulas"].items():
        cid, q = k.split("_")
        g, m = c["por_modelo"]["gnn"], c["por_modelo"]["mlp"]
        dv = c["paridade"]["distribuicao_gnn_menos_mlp"]["mae_rssi_validos_db"]
        gi, mi = g["inversao_vs_baselines_validos"], m["inversao_vs_baselines_validos"]
        contraste = (f"{dv['mediana']:+.2f} [{dv['min']:+.2f}, {dv['max']:+.2f}] ({dv['n_sorteios_gnn_melhor']}/{dv['n']})")
        linhas_tex.append(
            f"{NOME[cid]} {q} & {g['media_entre_sorteios']['mae_rssi_validos_db']:.2f} ({g['dp_entre_sorteios']['mae_rssi_validos_db']:.2f}) & "
            f"{m['media_entre_sorteios']['mae_rssi_validos_db']:.2f} ({m['dp_entre_sorteios']['mae_rssi_validos_db']:.2f}) & {mn(contraste)}")
        linhas_csv.append([k, g["media_entre_sorteios"]["mae_rssi_validos_db"], g["dp_entre_sorteios"]["mae_rssi_validos_db"],
                           m["media_entre_sorteios"]["mae_rssi_validos_db"], m["dp_entre_sorteios"]["mae_rssi_validos_db"],
                           dv["mediana"], dv["min"], dv["max"], dv["n_sorteios_gnn_melhor"], dv["n"],
                           gi["fspl"]["n_sorteios_modelo_melhor"], gi["hata_rural"]["n_sorteios_modelo_melhor"],
                           mi["fspl"]["n_sorteios_modelo_melhor"], mi["hata_rural"]["n_sorteios_modelo_melhor"]])
    cab = ["celula", "gnn_med_val", "gnn_dp_val", "mlp_med_val", "mlp_dp_val", "d_val_med", "d_val_min", "d_val_max", "n_gnn_melhor_val", "n_sorteios",
           "n_gnn_lt_fspl", "n_gnn_lt_hata", "n_mlp_lt_fspl", "n_mlp_lt_hata"]
    confere_antigo("T7", "T7_antiga_csv", [dict(zip(cab, r)) for r in linhas_csv], {c: c for c in cab[1:]}, "celula")
    d = a["deriva_2_2"]
    linhas_tex.append("\\midrule\nSD across the four cells of the per-cell means & "
                      f"{d['gnn']['dp_entre_celulas_das_medias_validos_db']:.2f} & {d['mlp']['dp_entre_celulas_das_medias_validos_db']:.2f} & ")
    hdr = ("lccc",
           "Cell & GNN valid MAE $\\downarrow$ (dB): mean (SD over draws) & MLP valid MAE $\\downarrow$ (dB): mean (SD over draws) & GNN$-$MLP, valid (dB): median [min, max] (GNN lower / 5)")
    return escrever("T7_deriva_modelo_A4_v3-12", hdr, linhas_tex, cab, linhas_csv,
                    "T7: A4 batch, Bauru Q1-Q4 x split draws 101-105, MAE of received power on valid nodes (dB); SD over draws vs SD over cells; paired GNN-MLP contrast. "
                    "Column 'SD over draws, FSPL / Hata' removed (B8.3). CSV also carries per-cell inversion counts (model MAE_valid < baseline MAE over all test nodes, criterio_A4 3.3 as written).")


def gerar_t7b():
    """T7 without the paired GNN-MLP contrast column: per-cell valid MAE (mean, sd over draws) and the sd across cells."""
    a = load(ENT["A4"])
    linhas_tex, linhas_csv = [], []
    for k, c in a["celulas"].items():
        cid, q = k.split("_")
        g, m = c["por_modelo"]["gnn"], c["por_modelo"]["mlp"]
        gm, gd = g["media_entre_sorteios"]["mae_rssi_validos_db"], g["dp_entre_sorteios"]["mae_rssi_validos_db"]
        mm, md = m["media_entre_sorteios"]["mae_rssi_validos_db"], m["dp_entre_sorteios"]["mae_rssi_validos_db"]
        linhas_tex.append(f"{NOME[cid]} {q} & {gm:.2f} ({gd:.2f}) & {mm:.2f} ({md:.2f})")
        linhas_csv.append([k, gm, gd, mm, md])
    cab = ["celula", "gnn_med_val", "gnn_dp_val", "mlp_med_val", "mlp_dp_val"]
    confere_antigo("T7b", "T7_antiga_csv", [dict(zip(cab, r)) for r in linhas_csv], {c: c for c in cab[1:]}, "celula")
    d = a["deriva_2_2"]
    sg, sm = d["gnn"]["dp_entre_celulas_das_medias_validos_db"], d["mlp"]["dp_entre_celulas_das_medias_validos_db"]
    linhas_tex.append(f"\\midrule\nSD across the four cells of the per-cell means & {sg:.2f} & {sm:.2f}")
    linhas_csv.append(["ALL_dp_entre_celulas_das_medias", None, sg, None, sm])
    hdr = ("lcc", "Cell & GNN valid MAE $\\downarrow$ (dB): mean (SD over draws) & MLP valid MAE $\\downarrow$ (dB): mean (SD over draws)")
    return escrever("T7b_deriva_modelo_A4_v3-12", hdr, linhas_tex, cab, linhas_csv,
                    "T7b: A4 batch, Bauru Q1-Q4 x split draws 101-105, MAE of received power on valid nodes (dB); SD over draws vs SD over cells; "
                    "T7 without the paired GNN-MLP contrast column (the full T7 stays in the folder, outside the package)")


def gerar_t8():
    d = load(ENT["R3"])
    assert d["leitura_primaria"] == "L_no_a_no"
    linhas_tex, linhas_csv = [], []
    for k in Q1:
        primeiro_cel = True
        for pred in ("constante", "fspl_calibrado_b"):
            primeiro_pred = True
            for pop in ("todos", "validos"):
                r = d["por_celula"][k]["resultados"][f"{pred}|{pop}"]
                assert r["preditor"] == pred and r["populacao"] == pop
                t = r["fechado"]["m_nodal"]["L_no_a_no"]
                pre = "\\midrule\n" if (primeiro_cel and k != Q1[0]) else ""
                cel = f"{NOME[k.split('_')[0]]} Q1" if primeiro_cel else ""
                prd = PRED[pred] if primeiro_pred else ""
                linhas_tex.append(f"{pre}{cel} & {prd} & {POP[pop]} & {fnum(r['mu_U_dB'], 3)} & [{fnum(t['T_inf_dB'], 3)}, {fnum(t['T_sup_dB'], 3)}]")
                linhas_csv.append([k, pred, pop, r["mu_U_dB"], t["T_inf_dB"], t["T_sup_dB"], t["fracao_nos_so_cotas"]])
                primeiro_cel = primeiro_pred = False
    cab = ["celula", "preditor", "populacao", "mu_U_dB", "T_inf_dB", "T_sup_dB", "fracao_nos_so_cotas"]
    hdr = ("lllcc", "Cell & Predictor & Population & Domain mean $\\mu_U$ (dB) & Design term $[T_{\\inf}, T_{\\sup}]$ (dB)")
    return escrever("T8_termo_desenho_v3-12", hdr, linhas_tex, cab, linhas_csv,
                    "T8: R3_termo_desenho.json, primary reading L_no_a_no, m nodal (closed form where m=0 or condition (iii) holds at the node, bounds elsewhere; extremes of the ratio over "
                    "p in the box); T = sum p_i e_i / sum p_i - mu_U; g=10 km, b=2 km, one calibration of e_i (seed 42)", cru=True)


def gerar_ta1():
    d = load(ENT["B7.2"])
    campos = {(c["grupo"], c["campo"]): c for c in d["campos"]}

    def C(g, c):
        return campos[(g, c)]

    def guarda(g, c, lado, esperado):
        v = C(g, c)[lado]
        assert v == esperado, (g, c, lado, v, esperado)
        return v

    assert d["valor_maximo_shadowing_ndvi_penalty_registrado"] == 0.0
    linhas, csvl = [], []
    grupos_ordem = []

    def grupo(nome):
        grupos_ordem.append(nome)
        linhas.append(("grupo", nome))

    def lin(rotulo, v_g, v_m, csv_g=None, csv_m=None, fonte=None, mesmo=None):
        """Add a row (LaTeX text for graph regressor and perceptron); 'same' in the perceptron column when mesmo."""
        if mesmo is None:
            mesmo = (v_g == v_m)
        assert not (v_m == "same" and mesmo is False), rotulo
        linhas.append(("linha", rotulo, v_g, "same" if mesmo else v_m))
        csvl.append([grupos_ordem[-1], rotulo, csv_g if csv_g is not None else v_g, "same" if mesmo else (csv_m if csv_m is not None else v_m), fonte or ""])

    def fx(g, c):
        return f"{g}/{c}"

    grupo("Architecture")
    L = C("arquitetura", "camadas de passagem de mensagem / camadas do encoder")
    lin("Layers", f"{L['gnn']} (message passing)", f"{L['mlp']} (encoder)", fonte=fx("arquitetura", "camadas ..."))
    nn = guarda("vizinhanca", "amostragem no treino", "gnn", {"num_neighbors": "[-1] por relacao (lista de tamanho 1 = UM salto, vizinhanca completa)", "k_antenna": -1, "k_terrain": -1, "disjoint_treino": False})
    guarda("vizinhanca", "amostragem no treino", "mlp", "lotes de nos de terreno (sem grafo)")
    lin("Neighbourhood", "one hop per relation, full neighbourhood (no sampling)", "node batches, no graph", fonte=fx("vizinhanca", "amostragem no treino"))
    guarda("arquitetura", "operador antena->terreno", "gnn", "GATv2Conv(hidden->hidden/heads, heads=4, concat, edge_dim=2, dropout, add_self_loops=False)")
    guarda("arquitetura", "operador terreno->terreno (Moore + self-loop)", "gnn", "GATv2Conv(..., add_self_loops=True)")
    guarda("arquitetura", "operador terreno->antena", "gnn", "SAGEConv(aggr='mean') -- sem gradiente (relacao nao amostrada a partir de sementes de terreno)")
    H = C("arquitetura", "cabecas de atencao")
    ed = int(_re.search(r"edge_dim=(\d+)", C("arquitetura", "operador antena->terreno")["gnn"]).group(1))
    lin("Operators",
        f"transmitter$\\to$terrain and terrain$\\to$terrain (Moore neighbourhood with self-loop): GATv2, {H['gnn']} heads concatenated, edge dimension {ed}; "
        "terrain$\\to$transmitter: SAGE with mean aggregation, no gradient",
        "none (no edges)", fonte=fx("arquitetura", "operador *"))
    guarda("arquitetura", "agregacao entre relacoes", "gnn", "soma")
    lin("Relation aggregation", "sum", "--", fonte=fx("arquitetura", "agregacao entre relacoes"))
    guarda("arquitetura", "por camada (GNN) / bloco (MLP)", "gnn", "LeakyReLU(0.2) + dropout + residual + LayerNorm")
    guarda("arquitetura", "por camada (GNN) / bloco (MLP)", "mlp", "Linear + BatchNorm1d + LeakyReLU(0.2) (+ Dropout, exceto ultimo)")
    sl = _re.search(r"LeakyReLU\(([\d.]+)\)", C("arquitetura", "por camada (GNN) / bloco (MLP)")["gnn"]).group(1)
    lin("Layer", f"LeakyReLU (slope {sl}), dropout, residual connection, LayerNorm",
        f"linear, BatchNorm, LeakyReLU (slope {sl}), dropout (not after the last block)", fonte=fx("arquitetura", "por camada (GNN) / bloco (MLP)"))
    lin("Attention heads", str(H["gnn"]), "--", fonte=fx("arquitetura", "cabecas de atencao"))
    W = C("arquitetura", "largura oculta / larguras do encoder")
    lin("Width", f"{W['gnn']} (hidden)", ", ".join(str(w) for w in W["mlp"]) + " (encoder)", fonte=fx("arquitetura", "largura oculta / larguras do encoder"))
    T = C("arquitetura", "entradas por no de terreno")
    A = C("arquitetura", "entradas por transmissor")
    assert A["mlp"] == "nao usado"
    lin("Input features per terrain node", str(T["gnn"]), str(T["mlp"]), fonte=fx("arquitetura", "entradas por no de terreno"))
    lin("Input features per transmitter", str(A["gnn"]), "not used", fonte=fx("arquitetura", "entradas por transmissor"))
    O = C("arquitetura", "saidas")
    lin("Output channels", str(O["gnn"]), str(O["mlp"]), fonte=fx("arquitetura", "saidas"))
    DR = C("arquitetura", "dropout")
    lin("Dropout", num(DR["gnn"]), num(DR["mlp"]), fonte=fx("arquitetura", "dropout"))
    PN = C("capacidade", "parametros nominais")
    PE = C("capacidade", "parametros efetivos (recebem gradiente)")
    lin("Parameters, nominal", milhar(PN["gnn"]), milhar(PN["mlp"]), csv_g=PN["gnn"], csv_m=PN["mlp"], fonte=fx("capacidade", "parametros nominais"))
    lin("Parameters receiving gradient", milhar(PE["gnn"]), milhar(PE["mlp"]), csv_g=PE["gnn"], csv_m=PE["mlp"], fonte=fx("capacidade", "parametros efetivos"))
    S = C("capacidade", "braco de sensibilidade (10 corridas, larguras nominais)")
    assert S["gnn"] is None
    lin("Width-sensitivity arm (10 runs)", "--",
        "widths " + ", ".join(str(w) for w in S["mlp"]["larguras"]) + f"; {milhar(S['mlp']['nominal'])} parameters",
        csv_m=json.dumps(S["mlp"]), fonte=fx("capacidade", "braco de sensibilidade"))

    grupo("Loss")
    texto_pred = "Huber(delta=5 dB) por canal, nos 5 canais, media sobre os nos semente (sentinelas incluidos)"
    guarda("perda", "termo de predicao", "gnn", texto_pred)
    guarda("perda", "termo de predicao", "mlp", "idem")
    delta = _re.search(r"delta=([\d.]+) dB", texto_pred).group(1)
    lin("Prediction term", f"Huber loss with threshold {delta}~dB on each of the {O['gnn']} channels, averaged over the seed nodes (sentinel nodes included)", "same", mesmo=True,
        fonte=fx("perda", "termo de predicao"))
    guarda("perda", "pesos dos canais", "gnn", "softmax(log_weights), log_weights=zeros(5) fora do otimizador => 1/5 cada")
    lin("Channel weights", f"uniform, $1/{O['gnn']}$ each (fixed)", "same", mesmo=True, fonte=fx("perda", "pesos dos canais"))
    ex_dist = guarda("perda", "penalidade gradiente de distancia (expressao)", "gnn", "mean_{(i,j): d_i<d_j} ReLU(rssi_hat_j - rssi_hat_i + 1 dBm), K=min(512, N//2) pares sorteados no lote")
    ex_var = guarda("perda", "penalidade de variancia (expressao)", "gnn", "ReLU(std(y_rssi) - std(rssi_hat))^2 no lote (alvo sem gradiente)")
    ex_sh = guarda("perda", "penalidade sombra x NDVI (expressao)", "gnn", "ReLU(0.15 - corr(PL_veg_hat, NDVI)) no lote, NDVI = coluna 12 (sem gradiente)")
    ex_fs = guarda("perda", "restricao FSPL (nao citada no texto)", "gnn", "0.05 * mean ReLU(FSPL_900MHz(d) - PL_hat_total)")
    for g in ("penalidade gradiente de distancia (expressao)", "penalidade de variancia (expressao)", "penalidade sombra x NDVI (expressao)", "restricao FSPL (nao citada no texto)"):
        assert C("perda", g)["mlp"] == "idem" and C("perda", g)["igual_nos_dois_bracos"] is True
    w_d = C("perda", "penalidade gradiente de distancia (peso)")["gnn"]
    w_v = C("perda", "penalidade de variancia (peso)")["gnn"]
    w_s = C("perda", "penalidade sombra x NDVI (peso)")["gnn"]
    for g in ("penalidade gradiente de distancia (peso)", "penalidade de variancia (peso)", "penalidade sombra x NDVI (peso)"):
        assert C("perda", g)["gnn"] == C("perda", g)["mlp"]
    w_f = float(_re.match(r"([\d.]+) \*", ex_fs).group(1))
    kpar = int(_re.search(r"K=min\((\d+), N//2\)", ex_dist).group(1))
    marg = _re.search(r"\+ ([\d.]+) dBm\)", ex_dist).group(1)
    lim = _re.search(r"ReLU\(([\d.]+) - corr", ex_sh).group(1)
    lin("Distance-gradient penalty",
        f"weight {num(w_d)}; $\\operatorname{{mean}}_{{(i,j):\\,\\mathrm{{dist}}_i<\\mathrm{{dist}}_j}}\\max\\{{0,\\;\\hat P_j-\\hat P_i+{marg}~\\mathrm{{dB}}\\}}$ over $K$ node pairs drawn in each batch, $K$ being the smaller of {kpar} and half the batch size rounded down, $\\mathrm{{dist}}$ being the distance to the nearest transmitter",
        "same", mesmo=True, csv_g=ex_dist + " | dist = distancia ate a antena mais proxima (physics_loss.py:360, docstring de dist_to_ant)", fonte=fx("perda", "penalidade gradiente de distancia"))
    lin("Variance penalty",
        f"weight {num(w_v)}; $\\max\\{{0,\\;\\operatorname{{sd}}(P)-\\operatorname{{sd}}(\\hat P)\\}}^{{2}}$ over the batch, target without gradient",
        "same", mesmo=True, csv_g=ex_var, fonte=fx("perda", "penalidade de variancia"))
    lin("Shadowing$\\times$NDVI penalty",
        f"weight {num(w_s)}; $\\max\\{{0,\\;{lim}-\\operatorname{{corr}}(\\hat L_{{\\mathrm{{veg}}}},\\mathrm{{NDVI}})\\}}$ over the batch, NDVI without gradient, $\\hat L_{{\\mathrm{{veg}}}}$ being the vegetation component of the decoded path loss; identically zero in these runs",
        "same", mesmo=True, csv_g=ex_sh + " | valor maximo registrado nas epocas das corridas = " + num(d["valor_maximo_shadowing_ndvi_penalty_registrado"]),
        fonte=fx("perda", "penalidade sombra x NDVI") + "; valor_maximo_shadowing_ndvi_penalty_registrado")
    lin("Free-space loss penalty",
        f"weight {num(w_f)}; $\\operatorname{{mean}}\\max\\{{0,\\;\\mathrm{{FSPL}}_{{900}}(\\mathrm{{dist}})-\\hat L_{{\\mathrm{{tot}}}}\\}}$ over the batch, $\\hat L_{{\\mathrm{{tot}}}}$ being the decoded total path loss",
        "same", mesmo=True, csv_g=ex_fs, fonte=fx("perda", "restricao FSPL"))

    grupo("Optimisation")
    ot = guarda("otimizacao", "otimizador", "gnn", "AdamW, weight_decay=1e-5")
    wd = float(_re.search(r"weight_decay=([\d.e-]+)", ot).group(1))
    lin("Optimiser", f"AdamW, weight decay {pot10(wd)}", "same", mesmo=C("otimizacao", "otimizador")["igual_nos_dois_bracos"], csv_g=ot, fonte=fx("otimizacao", "otimizador"))
    LR = C("otimizacao", "taxa inicial")
    lin("Initial learning rate", pot10(LR["gnn"]), "same", mesmo=(LR["gnn"] == LR["mlp"]), csv_g=LR["gnn"], fonte=fx("otimizacao", "taxa inicial"))
    AG = C("otimizacao", "agenda")
    ag = AG["gnn"]
    assert ag["classe"] == "CosineAnnealingWarmRestarts" and ag["passo"] == "por epoca" and ag["reinicio_dentro_de_8_epocas"] is False and AG["gnn"] == AG["mlp"]
    EP = C("otimizacao", "epocas")
    lin("Learning-rate schedule",
        f"cosine annealing with warm restarts, $T_0={ag['T_0_epocas']}$ epochs, $T_{{\\mathrm{{mult}}}}={ag['T_mult']}$, minimum rate {pot10(ag['eta_min'])}, stepped per epoch; "
        f"no restart falls within the {EP['gnn']} epochs", "same", mesmo=True, csv_g=json.dumps(ag), fonte=fx("otimizacao", "agenda"))
    CL = C("otimizacao", "corte de gradiente (norma)")
    lin("Gradient clipping, norm threshold", num(CL["gnn"]), "same", mesmo=(CL["gnn"] == CL["mlp"]), fonte=fx("otimizacao", "corte de gradiente (norma)"))
    guarda("otimizacao", "precisao mista", "gnn", "AMP (autocast + GradScaler) em CUDA")
    lin("Mixed precision", "automatic mixed precision (CUDA)", "same", mesmo=True, fonte=fx("otimizacao", "precisao mista"))
    BT = C("otimizacao", "lote (nos semente)")
    lin("Batch size (nodes)", num(BT["gnn"]), "same", mesmo=(BT["gnn"] == BT["mlp"]), fonte=fx("otimizacao", "lote (nos semente)"))
    lin("Epochs", num(EP["gnn"]), "same", mesmo=(EP["gnn"] == EP["mlp"]), fonte=fx("otimizacao", "epocas"))
    ES = C("otimizacao", "parada antecipada")
    assert ES["gnn"] == 0 and ES["mlp"] == 0
    lin("Early stopping", "none", "same", mesmo=True, fonte=fx("otimizacao", "parada antecipada"))
    PS = C("otimizacao", "passos de gradiente por corrida")

    def med_faixa(x):
        a, b = num(x["min"]), num(x["max"])
        return f"{num(x['mediana'])} ({a}--{b})" if a != b else f"{num(x['mediana'])}"
    lin("Gradient steps per run, median (range)", med_faixa(PS["gnn"]), med_faixa(PS["mlp"]), csv_g=json.dumps(PS["gnn"]), csv_m=json.dumps(PS["mlp"]), fonte=fx("otimizacao", "passos de gradiente por corrida"))

    grupo("Selection")
    guarda("selecao", "regra da epoca retida", "gnn", "menor mae_rssi_db em VAL")
    guarda("selecao", "regra da epoca retida", "mlp", "menor mae_rssi_db em VAL")
    lin("Epoch retained", "lowest validation MAE of received power", "same", mesmo=True, fonte=fx("selecao", "regra da epoca retida"))
    assert C("selecao", "teste usado na selecao")["gnn"] is False and C("selecao", "teste usado na selecao")["mlp"] is False
    lin("Test set used in selection", "no", "same", mesmo=True, fonte=fx("selecao", "teste usado na selecao"))
    BE = C("selecao", "melhor epoca (mediana, min, max)")
    lin("Best epoch, median (range)", med_faixa(BE["gnn"]), med_faixa(BE["mlp"]), csv_g=json.dumps(BE["gnn"]), csv_m=json.dumps(BE["mlp"]), fonte=fx("selecao", "melhor epoca"))

    grupo("Baselines")
    BF = C("baselines", "frequencia")
    lin("Frequency (MHz)", num(BF["gnn"]), "same", mesmo=(BF["gnn"] == BF["mlp"]), fonte=fx("baselines", "frequencia"))
    BH = C("baselines", "h_tx, h_rx (m) Hata rural e COST-231")
    assert BH["gnn"] == BH["mlp"]
    lin("Heights (m)", f"Hata rural and COST-231: $h_{{\\mathrm{{tx}}}}={num(BH['gnn']['h_tx'])}$, $h_{{\\mathrm{{rx}}}}={num(BH['gnn']['h_rx'])}$", "same", mesmo=True, csv_g=json.dumps(BH["gnn"]),
        fonte=fx("baselines", "h_tx, h_rx"))
    guarda("baselines", "variante COST-231", "gnn", "suburban (C_m = 0)")
    lin("COST-231 variant", "suburban ($C_m=0$)", "same", mesmo=True, fonte=fx("baselines", "variante COST-231"))
    guarda("baselines", "distancia", "gnn", "distancia ao transmissor mais proximo, piso 1 m")
    lin("Distance", "to the nearest transmitter, floored at 1~m", "same", mesmo=True, fonte=fx("baselines", "distancia"))
    guarda("baselines", "calibracao", "gnn", "offset = mediana(RSSI + PL_modelo) no treino ((a) todos os nos; (b) so validos)")
    lin("Calibration", "offset equal to the training median of received power plus modelled path loss, over (a) all training nodes, (b) valid training nodes only", "same", mesmo=True,
        fonte=fx("baselines", "calibracao"))

    grupo("Hardware and cost")
    HW = C("custo", "hardware")
    assert HW["gnn"] == HW["mlp"] and len(HW["gnn"]) == 3
    gpu, torch_v, cuda_v = HW["gnn"]
    lin("Hardware and software", f"{gpu}; PyTorch {torch_v}; CUDA {cuda_v}", "same", mesmo=True, csv_g=json.dumps(HW["gnn"]), fonte=fx("custo", "hardware"))
    TM = C("custo", "tempo por corrida (s)")
    lin("Time per run (s), median (range)", med_faixa(TM["gnn"]), med_faixa(TM["mlp"]), csv_g=json.dumps(TM["gnn"]), csv_m=json.dumps(TM["mlp"]), fonte=fx("custo", "tempo por corrida (s)"))
    VR = C("custo", "pico de VRAM (MB)")
    lin("Peak GPU memory (MB), median (range)", med_faixa(VR["gnn"]), med_faixa(VR["mlp"]), csv_g=json.dumps(VR["gnn"]), csv_m=json.dumps(VR["mlp"]), fonte=fx("custo", "pico de VRAM (MB)"))

    # The full table is too tall for one page, so two halves split between groups are also written.
    def escrever_ta1(nome, grupos_ok):
        tex = OUT / f"{nome}.tex"
        with open(tex, "w", encoding="utf-8") as f:
            f.write("% gerado por scripts/v3_12_gerar_tabelas.py -- nao editar a mao\n")
            f.write("% TA1: fase4/B7.2_hiperparametros/b7_2_folha_hiperparametros.json (60 + 60 corridas); 'same' = valor identico nos dois bracos; "
                    "so o limiar do corte de gradiente (a fracao de passos cortados nao e impressa); colunas p{} (largura do texto); grupos: " + ", ".join(grupos_ok) + "\n")
            f.write("\\begin{tabular}{" + "".join(">{\\raggedright\\arraybackslash}p{%s\\linewidth}" % w for w in ("0.22", "0.43", "0.25")) + "}\n\\toprule\n")
            f.write("Setting & Graph regressor & Perceptron \\\\\n\\midrule\n")
            atual, primeiro = None, True
            for it in linhas:
                if it[0] == "grupo":
                    atual = it[1]
                    if atual in grupos_ok:
                        f.write(("" if primeiro else "\\midrule\n") + "\\multicolumn{3}{l}{\\textit{" + it[1] + "}} \\\\\n")
                        primeiro = False
                elif atual in grupos_ok:
                    rot = it[1].replace('threshold', '\\mbox{threshold}')
                    f.write(f"{rot} & {it[2]} & {it[3]} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
        return tex
    tex = escrever_ta1("TA1_hiperparametros_v3-12", grupos_ordem)
    tex_a = escrever_ta1("TA1a_hiperparametros_arquitetura_perda_v3-12", ["Architecture", "Loss"])
    tex_b = escrever_ta1("TA1b_hiperparametros_otimizacao_custo_v3-12", ["Optimisation", "Selection", "Baselines", "Hardware and cost"])
    csvp = OUT / "TA1_hiperparametros_v3-12.csv"
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["grupo", "linha", "graph_regressor", "perceptron", "campo_da_folha_B7.2"])
        for r in csvl:
            w.writerow(r)
    CONF["TA1_guardas_texto_vs_folha_B7.2"] = {"strings_da_folha_conferidas_por_assert": True, "linhas": len(csvl), "grupos": grupos_ordem,
                                               "valor_maximo_shadowing_ndvi_penalty_registrado": d["valor_maximo_shadowing_ndvi_penalty_registrado"],
                                               "fracao_de_passos_cortados_impressa": False, "passou": True}
    return [tex, csvp, tex_a, tex_b]


def gerar_tn():
    """TN: notation table taken from the tabular of a LaTeX block of the manuscript, with asserted edits.

    Rows are removed, added, merged and relabelled at unique anchors; the run fails if the block changes.
    """
    src = ENT["F10_notacao"].read_text(encoding="utf-8")
    i0 = src.index("(F-10) Tabela de notacao")
    ini = src.index("\\begin{tabular}", i0)
    fim = src.index("\\end{tabular}", ini) + len("\\end{tabular}")
    bloco = src[ini:fim]
    assert bloco.startswith("\\begin{tabular}{lll}")
    bloco_fonte = bloco
    ELL_ANT = "$\\ell_x$, $\\ell_y$ & lattice spacings of the terrain nodes & Proposition~\\ref{prop:degree} \\\\\n"
    ELL_NOVA = "$\\ell_x$, $\\ell_y$ & lattice spacings of the terrain nodes & Section~\\ref{sec:degree} \\\\\n"
    ANCORA_D = "$d_i$, $d_i^V$, $\\partial V$, $\\Lambda(V)$ & "
    ANCORA_F = "$\\Delta_f$, $e_f$ & "
    NOVA_D = ("$d_{\\max}$, $\\bar d_V$ & maximum degree; mean degree over $V$ & "
              "Proposition~\\ref{prop:degree} \\\\\n")
    NOVA_MAE = ("$\\mathrm{MAE}$, $\\mathrm{MAE}_{\\mathrm{valid}}$ & mean absolute error; on valid nodes & "
                "Equation~\\eqref{eq:mixture} \\\\\n"
                "$\\mathrm{MAE}_{\\sigma}$ & mean absolute error on the test set of draw $\\sigma$ & "
                "Proposition~\\ref{prop:drift} \\\\\n")
    for anc in (ELL_ANT, ANCORA_D, ANCORA_F):
        assert bloco.count(anc) == 1, anc
    bloco = bloco.replace(ELL_ANT, "")
    i_d = bloco.index(ANCORA_D)
    i_d = bloco.index("\n", i_d) + 1  # right after the row of d_i
    bloco = bloco[:i_d] + NOVA_D + bloco[i_d:]
    i_f = bloco.index(ANCORA_F)  # right before the row of Delta_f, e_f
    bloco = bloco[:i_f] + NOVA_MAE + bloco[i_f:]
    ini_c = bloco.index("\\midrule\n") + len("\\midrule\n")
    fim_c = bloco.index("\\bottomrule")
    linhas_b = [ln[:-2].strip() if ln.rstrip().endswith("\\\\") else ln for ln in bloco[ini_c:fim_c].strip().split("\n")]
    linhas_b = [tuple(c.strip() for c in ln.split(" & ")) for ln in linhas_b]
    assert all(len(t) == 3 for t in linhas_b)
    n_antes = len(linhas_b)
    RETIRADA = "$\\hat L$, $\\hat P$, $P_{\\mathrm{tx}}$, $L_{\\max}$"
    assert sum(t[0] == RETIRADA for t in linhas_b) == 1
    linhas_b = [t for t in linhas_b if t[0] != RETIRADA]
    assert not any(t[0].startswith("$\\ell_x$") for t in linhas_b)
    # One new row for the neighbour blocks of B, inserted before the row of S.
    SIMB_VIZ = ("$B_{\\mathsf{w}}$, $B_{\\mathsf{e}}$, $B_{\\mathsf{s}}$, $B_{\\mathsf{n}}$, $B_{\\mathsf{ws}}$, $B_{\\mathsf{es}}$, $B_{\\mathsf{wn}}$, $B_{\\mathsf{en}}$",
                "lateral and diagonal neighbour blocks of $B$", "Equation~\\eqref{eq:area}")
    k_s = [k for k, t in enumerate(linhas_b) if t[0].startswith("$\\mathcal{S}$")]
    assert len(k_s) == 1
    linhas_b.insert(k_s[0], SIMB_VIZ)
    n_antes += 1

    def ini(pre):
        ks = [k for k, t in enumerate(linhas_b) if t[0].startswith(pre)]
        assert len(ks) == 1, pre
        return ks[0]

    def fundir(prefixos, significado=None):
        ks = [ini(pr) for pr in prefixos]
        assert ks == list(range(ks[0], ks[0] + len(ks))), prefixos
        grupo = [linhas_b[k] for k in ks]
        refs = []
        for t in grupo:
            for rf in t[2].split("; "):
                if rf not in refs:
                    refs.append(rf)
        novo = (", ".join(t[0] for t in grupo), significado or "; ".join(t[1] for t in grupo), "; ".join(refs))
        linhas_b[ks[0]:ks[-1] + 1] = [novo]
        return novo
    FUSOES = [("$D$, $V_T$, $V_A$", "$E_{TT}$, $E_{AT}$"),
              ("$\\varpi(V)$, $\\bar\\varpi$", "$\\psi(X)$"),
              ("$g$, $b$", "$\\beta(x)$, $N$"),
              ("$k_{\\mathrm{tr}}", "$\\nu_{\\mathrm{tr}}"),
              ("$R_{\\mathrm{va}}(g,b)$", "$V_{\\mathrm{te}}(\\sigma)$"),
              ("$\\mathrm{MAE}$, $\\mathrm{MAE}_{\\mathrm{valid}}$", "$\\mathrm{MAE}_{\\sigma}$")]
    SIGNIF = {5: "mean absolute error; on the valid nodes; on the test set of draw $\\sigma$"}
    fundidas = [fundir(f, SIGNIF.get(k)) for k, f in enumerate(FUSOES)]  # merge neighbouring rows without changing any symbol
    assert len(linhas_b) <= 26, len(linhas_b)
    bloco = (bloco[:ini_c] + "\n".join(" & ".join(t) + " \\\\" for t in linhas_b) + "\n" + bloco[fim_c:])
    bloco = bloco.replace("\\begin{tabular}{lll}", "\\begin{tabular}{" + "".join(">{\\raggedright\\arraybackslash}p{%s\\linewidth}" % w for w in ("0.30", "0.38", "0.22")) + "}", 1)
    assert bloco.count("$R_{\\mathrm{va}}(g,b)$, $R_{\\mathrm{te}}(g,b)$") == 1 and bloco.count("Proposition~\\ref{prop:degree}") == 2
    bloco = bloco.replace("$R_{\\mathrm{va}}(g,b)$, $R_{\\mathrm{te}}(g,b)$", "$r(g,b)$, $r_{\\mathrm{va}}(g,b)$, $r_{\\mathrm{te}}(g,b)$")
    bloco = bloco.replace("nodal retention of validation and test;", "nodal retention of a partition, of validation and of test;")
    assert bloco.count("$\\Delta_f$, $e_f$ & sentinel and valid mean absolute errors of $f$ & Proposition~\\ref{prop:drift}") == 1
    bloco = bloco.replace("$\\Delta_f$, $e_f$ & sentinel and valid mean absolute errors of $f$ & Proposition~\\ref{prop:drift}", "$\\Delta_f$, $e_f$, $\\varepsilon_{\\mathrm{c}}$ & sentinel and valid mean absolute errors of $f$; remainder of the composition identity & Proposition~\\ref{prop:drift}, Equation~\\eqref{eq:composition}")
    bloco = bloco.replace("Proposition~\\ref{prop:degree}", "Remark~\\ref{prop:degree}").replace("Proposition~\\ref{prop:deflation}", "Remark~\\ref{prop:deflation}")
    corpo = bloco.split("\\midrule\n", 1)[1].split("\\bottomrule", 1)[0]
    rows = [ln.rstrip()[:-2].strip() if ln.rstrip().endswith("\\\\") else ln for ln in corpo.strip().split("\n")]
    csv_rows = [[c.strip() for c in r.split(" & ")] for r in rows]
    assert all(len(r) == 3 for r in csv_rows)
    tex = OUT / "TN_notacao_v3-12.tex"
    with open(tex, "w", encoding="utf-8") as f:
        f.write("% gerado por scripts/v3_12_gerar_tabelas.py -- nao editar a mao\n")
        f.write("% TN: ambiente tabular do bloco F-10 de _propostas_2026-10-01/blocos_formais_v3-12.tex com tres alteracoes declaradas em v3-12b "
                "(sem a linha ell_x, ell_y; linhas novas MAE, MAE_valid, MAE_sigma e d_max, \\bar d_V); "
                "refs: sec:setting, sec:sentinel, prop:*, def:*, eq:* a definir no manuscrito); sem caption\n")
        f.write(bloco + "\n")
    csvp = OUT / "TN_notacao_v3-12.csv"
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Symbol", "Meaning", "Defined in"])
        for r in csv_rows:
            w.writerow(r)
    # The written file must contain the edited block byte for byte.
    gravado = tex.read_text(encoding="utf-8")
    assert bloco in gravado
    CONF["TN_copia_do_bloco_F-10_com_alteracoes_v3-12b"] = {
        "linhas_de_simbolos": len(csv_rows), "linhas_antes_da_versao_enxuta_v3-12d": n_antes, "linha_retirada_v3-12d": RETIRADA, "linha_acrescentada_v3-12h": "blocos vizinhos B_w..B_en (Equation~\\eqref{eq:area})", "linha_retirada_v3-12g": "$\\ell_x$, $\\ell_y$ (sem ocorrencia no texto)",
        "linhas_fundidas_v3-12d": [{"simbolos": f[0], "ref": f[2]} for f in fundidas], "limite_de_linhas": 26,
        "alteracoes": ["- linha ell_x, ell_y (v3-12g: sem ocorrencia no texto)", "+ linha d_max, \\bar d_V", "+ linhas MAE/MAE_valid e MAE_sigma", "+ linha dos blocos vizinhos B_w..B_en (v3-12h)", "R_va/R_te -> r, r_va, r_te (v3-12h, colisao com \\bar\\rho)", "Proposition -> Remark em prop:degree (v3-12h)", "+ \\varepsilon_c na linha Delta_f, e_f (v3-12i)"],
        "bloco_gravado_contem_o_bloco_alterado": True, "passou": True}
    return [tex, csvp]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert sha(SCRIPT_ORIGINAL) == SHA_ORIGINAL_DECLARADO, "o original foi alterado"
    saidas = []
    for fn in (gerar_t2, gerar_t3, gerar_t3_resumo, gerar_t4, gerar_t5, gerar_t6, gerar_t7, gerar_t7b, gerar_t8, gerar_ta1, gerar_tn):
        saidas += fn()
    conf = OUT / "CONFERENCIAS_tabelas_v3-12.json"
    with open(conf, "w", encoding="utf-8") as f:
        json.dump(CONF, f, indent=1, ensure_ascii=False)
    saidas.append(conf)
    man = {
        "pasta_saida": str(OUT),
        "tarefa": "tables of the v3-12 revision (T2, T3, T4, T5, T6, T7, T8, TA1, TN) generated by script",
        "entradas": {k: {"arquivo": str(p), "sha256": sha(p)} for k, p in ENT.items()},
        "script": {"arquivo": str(Path(__file__).resolve()), "sha256": sha(Path(__file__)),
                   "original": str(SCRIPT_ORIGINAL), "original_sha256": SHA_ORIGINAL_DECLARADO},
        "saidas": {str(p.relative_to(B)): sha(p) for p in saidas},
        "comando": "/trabalho/ambientes/s33_amb_virtual/.venv/bin/python scripts/v3_12_gerar_tabelas.py",
    }
    with open(OUT / "MANIFEST_tabelas_v3-12.json", "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    for p in saidas:
        print("gravado", p)


if __name__ == "__main__":
    sys.exit(main())
