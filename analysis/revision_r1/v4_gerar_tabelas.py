"""Estimand Drift under Spatial Block Hold-Out (MDPI Mathematics, revision 1): tables of the revised manuscript.
No generator of the earlier table set is edited: analysis/v3_12_gerar_tabelas.py and analysis/v3_13_gerar_T9_G1.py are IMPORTED as
modules (their digests as read are recorded in the output headers), with `escrever` replaced by a capture function (nothing is written
to the earlier output folders); the numbers are read from the same JSON records those generators read, and the new numbers from
results/fase6/*.json and results/fase5/*.json (stamped records).
Output directory: env R1_OUT (default tables_figures/tables_v4_rerun). Scratch directory for the LaTeX tests: env R1_SCRATCH.

Free-space-loss predictor names used in every table:
  FSPL_UMA_VEZ     = "FSPL, calibrated once"          (offset fixed once on a reference partition, split seed 42; R2/R3)
  FSPL_POR_SORTEIO = "FSPL, recalibrated per draw"    (offset recomputed on the training partition of each draw; 3.1p, R5/V1)
Which table uses which name is recorded in PRED_LINHA below.

REQUIRES (not part of this repository, see README.md, "Revision 1, MDPI Mathematics"): the path set-up of the two imported generators
(README, "Absolute paths in the scripts"), the MDPI LaTeX class folder (env MDPI_DEFINITIONS) and a pdflatex binary (env PDFLATEX) for
the width probes and the compilation test; the blocks of the manuscript that are not distributed are not read by this script.
"""
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root
F5, F6 = ROOT / "results" / "fase5", ROOT / "results" / "fase6"
OUT = Path(os.environ.get("R1_OUT", ROOT / "tables_figures" / "tables_v4_rerun"))
TABLES_V3 = ROOT / "tables_figures" / "tables_v3-12"
OG_TAB = ROOT / "analysis" / "v3_12_gerar_tabelas.py"
OG_T9 = ROOT / "analysis" / "v3_13_gerar_T9_G1.py"
# Digests of the ORIGINAL generators as run; the published files were cleaned (PROVENANCE.md), so the digest of each file as read
# is computed at import time and written to the headers instead of being asserted.
SHA_OG_TAB = "5c1c1f6f4b532b9290c1691bf80ff85eaaf578754d311da904a329f300a7734f"
SHA_OG_T9 = "9c910fcab79ffb9bcb7301f344e32ccd7c02103091f335b1a2ff1ba36c9ed048"
TEXLIVE = os.environ.get("PDFLATEX", "pdflatex")
DEFINITIONS = os.environ.get("MDPI_DEFINITIONS")  # folder "Definitions" of the MDPI class (not distributed)
SCRATCH = Path(os.environ.get("R1_SCRATCH") or tempfile.mkdtemp(prefix="v4tab_"))

FSPL_UMA_VEZ = "FSPL, calibrated once"
FSPL_POR_SORTEIO = "FSPL, recalibrated per draw"
REGISTRO_FSPL = {
    "T_V4_vies": {"nome": FSPL_UMA_VEZ, "lido_em": "results/fase6/V4_tabela_vies.json (predictor 'fspl_b' = 'fspl_calibrado_b' of results/fase4/R2_resumo.json and R3_termo_desenho.json); single calibration at split seed 42"},
    "T4_totais_v4 (Tab. 7) e T4_S2_v4 (Tab. S2)": {"nome": FSPL_POR_SORTEIO, "lido_em": "analysis/v3_12_gerar_tabelas.py (trios_b2/trios_b0 per draw, status ok, in results/fase2/_v3_2.1_3.1_parcial_16x60rnd.json) and results/fase4/R1_b0_por_sorteio.json: offset recomputed on the training partition of each draw"},
    "T_V1_cobertura_fspl (extra, optional)": {"nome": FSPL_POR_SORTEIO, "lido_em": "results/fase6/V1_resumo.json, block 'fspl_b__validos' (analysis/v3_13_laco_comum.py: offset recomputed on the valid training nodes of each draw)"},
    "T_V1_cobertura, T_V2bis, T_V3, T_E6, T9 (Tab. 8), T3_resumo (Tab. 4), T5 (Tab. 3)": {"nome": "(no free-space-loss predictor)", "lido_em": "-"},
    "T6_referencia_desenho (Tab. 5) and T10_cobertura_R5 - not regenerated in this round": {
        "nome": "T6 uses 'fspl_calibrado_b' of results/fase4/R2_resumo.json (calibrated once; label " + FSPL_UMA_VEZ + "); T10 uses R5 'fspl_b' (recalibrated per draw; label " + FSPL_POR_SORTEIO + ")",
        "lido_em": "analysis/v3_12_gerar_tabelas.py (PRED) prints 'FSPL (second calibration)' for T6: generator label to be aligned with the table text"},
}

# layouts tried in order; the first whose natural width fits LIMITE_PT (usable width = textwidth + extralength = 525.5 pt in the MDPI class) is used. No resizebox.
CFGS = [("", "6pt", "fonte e espacamento padrao"), ("\\small", "5pt", "\\small, tabcolsep 5pt"), ("\\small", "4pt", "\\small, tabcolsep 4pt"),
        ("\\footnotesize", "4pt", "\\footnotesize, tabcolsep 4pt"), ("\\footnotesize", "3pt", "\\footnotesize, tabcolsep 3pt"), ("\\scriptsize", "3pt", "\\scriptsize, tabcolsep 3pt")]
LIMITE_PT = 520.0
NENHUM = "none; this table has no free-space-loss predictor (%s)"
PRED_LINHA = {
    "T3_resumo_v4": NENHUM % "only the constant predictor's error",
    "T4_totais_v4": FSPL_POR_SORTEIO + " (offset recomputed on the training partition of each draw, first and second calibration); also the constant predictor",
    "T4_S2_v4": FSPL_POR_SORTEIO + " (offset recomputed on the training partition of each draw, first and second calibration); also the constant predictor",
    "T5_inclusao_por_classe_v4": NENHUM % "inclusion frequencies of the design",
    "T6_referencia_desenho_v4": FSPL_UMA_VEZ + " (R2_resumo.json 'fspl_calibrado_b': offset fixed once on a reference partition, split seed 42; the old generator printed 'second calibration'); also the constant predictor",
    "T7b_deriva_modelo_A4_v4": NENHUM % "GNN and MLP only",
    "T9_deriva_modelo_G1_v4": NENHUM % "GNN and MLP only",
    "T_E6_inversoes": NENHUM % "GNN and MLP only",
    "T_V1_cobertura": NENHUM % "constant predictor only",
    "T2_retencao_v4": NENHUM % "node retention of the partitions",
    "T_V1_cobertura_fspl": FSPL_POR_SORTEIO + " (V1_resumo.json 'fspl_b__validos': offset recomputed on the training partition of each draw)",
    "T_V2bis_razoes": NENHUM % "GNN and MLP only",
    "T_V3_celulas": NENHUM % "cell descriptors",
    "T_V4_vies": FSPL_UMA_VEZ + " (fase6/V4_tabela_vies.json 'fspl_b' = fase4 R2/R3 'fspl_calibrado_b': offset fixed once on a reference partition, split seed 42); also the constant predictor",
    "TA1a_v4": NENHUM % "GNN and MLP settings",
    "TA1b_v4": NENHUM % "GNN and MLP settings; the FSPL baseline is only described in the 'Baselines' block",
    "Table_S1_v4": NENHUM % "GNN and MLP settings",
    "Table_S2_v4": FSPL_POR_SORTEIO + " (see T4_S2_v4); also the constant predictor",
    "Table_S3_v4": NENHUM % "GNN and MLP only",
}

SHAS = {}   # path -> sha256 (sources read)


def sha(p):
    p = Path(p)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def reg(p):
    p = Path(p)
    SHAS[str(p)] = sha(p)
    return p


def load(p):
    with open(reg(p), encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ formatting
def mn(s):
    return s.replace("-", "$-$")


def fd(x, nd):
    s = f"{x:.{nd}f}"
    if float(s) == 0.0:
        s = f"{0.0:.{nd}f}"
    return mn(s)


def s3(x):
    """tres algarismos significativos, mantendo zeros finais (10.4; 3.61; 7.60; 132)."""
    if x == 0:
        return "0.00"
    nd = max(0, 2 - int(math.floor(math.log10(abs(x)))))
    return fd(x, nd)


def rotulo_celula(c):
    cid, q = c.split("_")
    return f"{cid.capitalize()} {q}"


def cel(txt):
    return "\\celula{" + txt + "}"


def mc(n, spec, txt):
    return f"\\multicolumn{{{n}}}{{{spec}}}{{{txt}}}"


# ------------------------------------------------------------------ assembly
class Tabela:
    def __init__(self, nome, colspec, ncol, cabecalho, corpo, notas, fontes, comentarios, original=None):
        self.nome, self.colspec, self.ncol = nome, colspec, ncol
        self.cabecalho, self.corpo, self.notas = cabecalho, corpo, notas
        self.fontes, self.comentarios, self.original = fontes, comentarios, original
        self.cfg = 0
        self.nomes_preditor = None

    def tex(self, largura_nota_pt=None, cfg=None):
        cfg = CFGS[self.cfg if cfg is None else cfg]
        L = ["% " + c for c in self.comentarios]
        L.append("% fontes (caminho: sha256):")
        for f in self.fontes:
            L.append(f"%   {f}: {SHAS.get(str(f)) or sha(f)}")
        if self.original:
            L.append("% gerador original importado sem alteracao: " + self.original)
        L.append("% generator: analysis/revision_r1/v4_gerar_tabelas.py")
        L.append("% predictor names: " + PRED_LINHA[self.nome])
        L.append("% layout: " + cfg[2])
        if cfg[0]:
            L.append("{" + cfg[0] + "\\setlength{\\tabcolsep}{" + cfg[1] + "}%")
        L.append("\\begin{tabular}{" + self.colspec + "}")
        L.append("\\toprule")
        L += self.cabecalho
        L.append("\\midrule")
        L += self.corpo
        L.append("\\bottomrule")
        if self.notas and largura_nota_pt:
            L.append("\\addlinespace[3pt]")
            for n in self.notas:
                L.append(mc(self.ncol, "@{}p{%.1fpt}@{}" % largura_nota_pt, "\\footnotesize\\raggedright " + n) + " \\\\")
        L.append("\\end{tabular}")
        if cfg[0]:
            L.append("}")
        return "\n".join(L) + "\n"


# ------------------------------------------------------------------ import the earlier generators (main is not executed)
def importar(nome, caminho, sha_original=None):
    SHAS[str(caminho)] = sha(caminho)  # digest of the file as read; sha_original is the digest of the generator as run
    spec = importlib.util.spec_from_file_location(nome, caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def capturar(og, funcao):
    """roda og.<funcao>() com `escrever` substituido por captura; devolve {nome: dict(header, linhas, nota)}."""
    cap = {}

    def esc(nome, header_tex, linhas_tex, header_csv, linhas_csv, nota, cru=False):
        cap[nome] = dict(header=header_tex, linhas=linhas_tex, csv=linhas_csv, cab_csv=header_csv, nota=nota, cru=cru)
        return []
    antigo = og.escrever
    og.escrever = esc
    try:
        getattr(og, funcao)()
    finally:
        og.escrever = antigo
    return cap


# ================================================================== Tab. 4 (T3_resumo)
def tab4(og):
    cap = capturar(og, "gerar_t3_resumo")["T3_resumo_v3-12"]
    d = load(og.ENT["2.1"])["resumo"]
    r1 = load(og.ENT["R1_resumo"])["medias_16_celulas"]
    reg(og.OUT / "T3_deriva_erro_v3-12.csv")
    linhas = [
        ("Between-draw SD of the valid-node MAE", d["dp_medio_entre_sorteios_validos_dB_simples"], r1["dp_mae_constante_validos_b0"], 2),
        ("\\hspace{1em}weighted by the valid test nodes of each draw", d["dp_medio_entre_sorteios_validos_dB_ponderado"], None, 2),
        ("SD between the cells' mean errors", d["dp_entre_celulas_das_medias_validos_dB"], None, 2),
    ]
    capv = [l.split(" & ")[1].replace("$-$", "-") for l in cap["linhas"]]
    assert [fd(linhas[0][1], 3), fd(linhas[1][1], 3), fd(linhas[2][1], 3), fd(linhas[0][2], 3), fd(r1["dp_fracao_valida_teste_b2"], 3), fd(r1["dp_fracao_valida_teste_b0"], 3)] == capv, capv
    corpo = [f"{rot} & {fd(a, nd)} & {'--' if b is None else fd(b, nd)} \\\\" for rot, a, b, nd in linhas]
    corpo.append("\\midrule")
    corpo.append(f"Between-draw SD of the valid fraction of the test partition (--) & {fd(r1['dp_fracao_valida_teste_b2'], 3)} & {fd(r1['dp_fracao_valida_teste_b0'], 3)} \\\\")
    cab = ["Quantity (dB unless stated) & $b = 2$~km & $b = 0$ \\\\"]
    notas = ["MAE: mean absolute error of the constant predictor; 60 random split draws per cell, 16 cells, $g = 10$~km. "
             "SD: standard deviation. Rows 1 and 2 and the last row are means over the cells of the per-cell between-draw SD; row 3 is the SD between the cells' mean errors. "
             "--: not computed for $b = 0$ in the source files. Dimensionless row given to three decimals."]
    return Tabela("T3_resumo_v4", "lcc", 3, cab, corpo, notas, [og.ENT["2.1"], og.ENT["R1_resumo"], og.OUT / "T3_deriva_erro_v3-12.csv", OG_TAB],
                  ["Tab. 4 (v4-1): T3_resumo refeita em 3 colunas (quantidade, b = 2 km, b = 0), duas casas nas linhas em dB (tres na fracao valida, dimensionless).",
                   "Valores: fase2/2.1_deriva_erro_baselines_16x60rnd.json (resumo) e fase4/R1_b0_resumo.json (medias_16_celulas), os mesmos campos de gerar_t3_resumo (conferidos contra o resumo original capturado).",
                   "Conferencia: os valores impressos reproduzem (3 casas) os capturados do gerador original: " + "; ".join(l.split(" & ")[1] for l in cap["linhas"])],
                  original=f"{OG_TAB.relative_to(ROOT).as_posix()} (original as run: sha256 {SHA_OG_TAB})")


# ================================================================== Tab. 3 (T5 inclusao)
def tab3(og):
    cap = capturar(og, "gerar_t5")["T5_inclusao_por_classe_v3-12"]
    corpo = []
    for ln in cap["linhas"]:
        if ln.startswith("\\midrule"):
            corpo.append("\\midrule")
            ln = ln.split("\n", 1)[1]
        cols = [c.strip() for c in ln.split("&")]
        cols = [c if c else "--" for c in cols]
        assert len(cols) == 6, cols
        corpo.append(" & ".join(cols) + " \\\\")
    cab = [" & & " + mc(2, "c", "Frequency over 200 draws") + " & & \\\\",
           "\\cmidrule(lr){3-4}",
           "Class $m(i)$ & Nodes in class (\\%) & $p_{i\\mid\\mathrm{te}}$ & MC SE & Closed form or bounds & Bounds only (\\% of class) \\\\"]
    notas = ["$p_{i\\mid\\mathrm{te}}$: inclusion frequency given a test block; MC SE: Monte Carlo standard error of the frequency. "
             "Brackets: lower and upper bounds. --: not applicable. Ranges over the four Q1 cells, $g = 10$~km, $b = 2$~km."]
    return Tabela("T5_inclusao_por_classe_v4", "lccccc", 6, cab, corpo, notas, [og.ENT["R4"], OG_TAB],
                  ["Tab. 3 (v4-1): cabecalho de 5 linhas -> 2 linhas com \\multicolumn ('MC SE' definido em nota); celulas vazias da linha-resumo -> '--' (v4.1: split por '&' em vez de ' & ', que deixava as vazias; assert de 6 campos). Valores = gerar_t5 do original (capturados, sem alteracao).",
                   "Fonte: fase4/R4_p_por_classe.json (m nodal, leitura primaria L_no_a_no)."],
                  original=f"{OG_TAB.relative_to(ROOT).as_posix()} (original as run: sha256 {SHA_OG_TAB})")


# ================================================================== Tab. 7 e S2 (T4)
def tab7_s2(og):
    cap = capturar(og, "gerar_t4")
    tot = cap["T4_totais_v3-12"]
    s2 = cap["T4_S2_v3-12"]
    PRIM, SEG = "first", "second"
    nota_cal = ("Offset: all training nodes, offset of the FSPL predictor equal to the training median over all training nodes, sentinel included; offset: valid training nodes, the same over the valid training nodes only. "
                f"{FSPL_POR_SORTEIO}: the offset is recomputed on the training partition of each draw. ")
    nota_cont = ("Counts: draws with a valid test node in which the error of the FSPL predictor is lower than that of the constant predictor (valid nodes) or the reverse (all nodes); "
                 "the offset named is that of the FSPL predictor compared.")
    # --- Tab. 7
    corpo7 = []
    for ln in tot["linhas"]:
        corpo7.append(ln.replace("\\celula{All cells, \\\\ \\relax no buffer}", "All cells, no buffer") + " \\\\")
    cab7 = [" & & " + mc(2, "c", "Valid nodes") + " & " + mc(2, "c", "All nodes") + " \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
            " & & " + mc(2, "c", f"{FSPL_POR_SORTEIO}, lower") + " & " + mc(2, "c", "Constant lower") + " \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
            "Cells & Draws & " + " & ".join([cel("Offset: all \\\\ \\relax training nodes"), cel("Offset: valid \\\\ \\relax training nodes")] * 2) + " \\\\"]
    t7 = Tabela("T4_totais_v4", "lccccc", 6, cab7, corpo7, [nota_cal + nota_cont], [og.ENT["3.1p"], og.ENT["R1_resumo"], og.ENT["R1_por_sorteio"], OG_TAB],
                ["Tab. 7 (v4-1): cabecalho de 5 linhas -> 3 linhas com \\multicolumn; definicao das calibracoes em nota de rodape da tabela; preditor nomeado 'FSPL, recalibrated per draw'.",
                 "Valores = gerar_t4 do original (contagens recontadas dos parciais e conferidas por assert no original: 904/938 sorteios, 799/796/629/729 e 844/847/691/801)."],
                original=f"{OG_TAB.relative_to(ROOT).as_posix()} (original as run: sha256 {SHA_OG_TAB})")
    # --- Tab. S2: adds the median of FSPL with the second calibration (same filter as the original) and checks the two existing medians
    celp = load(og.ENT["3.1p"])["celulas"]
    med2 = {}
    for cid in og.CIDADES:
        for q in og.QS:
            k = f"{cid}_{q}"
            if k not in celp:
                continue
            mcs, mas, mbs = [], [], []
            for s in celp[k]["por_sorteio"]:
                if s.get("status") != "ok":
                    continue
                c_v = s["mae_constante_teste"]["validos"]
                a_v = s["mae_modelo_a_contaminado_teste"]["fspl"]["validos"]
                b_v = s["mae_modelo_b_validos_teste"]["fspl"]["validos"]
                if c_v is None or a_v is None or b_v is None:
                    continue
                mcs.append(c_v); mas.append(a_v); mbs.append(b_v)
            med2[k] = (statistics.median(mcs), statistics.median(mas), statistics.median(mbs))
    # checks the medians captured from the original (constant and first-calibration FSPL) against a recount
    csv_rows = {r[0]: r for r in s2["csv"]}
    for k, (mc_, ma_, mb_) in med2.items():
        assert abs(csv_rows[k][7] - mc_) < 1e-12 and abs(csv_rows[k][8] - ma_) < 1e-12, k
    corpo = []
    for ln in s2["linhas"]:
        pre = ""
        if ln.startswith("\\midrule"):
            pre, ln = "\\midrule\n", ln.split("\n", 1)[1]
        ln = ln.replace("\\celula{All cells, \\\\ \\relax no buffer}", "All cells, no buffer")
        cols = [c.strip() for c in ln.split(" & ")]
        if rotulo_inv(cols[0]) in med2:  # linha de celula: 8 colunas do original + coluna nova = 9
            cols.append(f"{med2[rotulo_inv(cols[0])][2]:.3f}")
        else:  # linhas All cells: 6 colunas de contagem + 3 medianas ausentes ('--'); descarta o '&' residual do original
            cols = [c for c in cols if c not in ("&", "")][:6] + ["--", "--", "--"]
        assert len(cols) == 9, (cols[0], len(cols))
        corpo.append(pre + " & ".join(cols) + " \\\\")
    cabs = [" & & " + mc(2, "c", "Valid nodes") + " & " + mc(2, "c", "All nodes") + " & " + mc(3, "c", "Median valid-node MAE (dB)") + " \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}",
            " & & " + mc(2, "c", f"{FSPL_POR_SORTEIO}, lower") + " & " + mc(2, "c", "Constant lower") + " & & " + mc(2, "c", FSPL_POR_SORTEIO) + " \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){8-9}",
            "Cell & Draws & " + " & ".join([cel("Offset: all \\\\ \\relax training nodes"), cel("Offset: valid \\\\ \\relax training nodes")] * 2 + ["Constant"] + [cel("Offset: all \\\\ \\relax training nodes"), cel("Offset: valid \\\\ \\relax training nodes")]) + " \\\\"]
    s2t = Tabela("T4_S2_v4", "lcccccccc", 9, cabs, corpo, [nota_cal + nota_cont + " Medians are over the same draws as the counts."],
                 [og.ENT["3.1p"], og.ENT["R1_resumo"], og.ENT["R1_por_sorteio"], OG_TAB],
                 ["Tab. S2 (v4-1): cabecalho de 5 linhas -> 3 linhas com \\multicolumn; coluna NOVA com a mediana do MAE nos validos da FSPL com a segunda calibracao (mesmo filtro do original: sorteios ok com no valido; as duas medianas existentes foram reconferidas por assert contra o capturado).",
                  "Valores = gerar_t4 do original (capturados) + recalculo da coluna nova a partir de fase2/_v3_2.1_3.1_parcial_16x60rnd.json (mae_modelo_b_validos_teste.fspl.validos)."],
                 original=f"{OG_TAB.relative_to(ROOT).as_posix()} (original as run: sha256 {SHA_OG_TAB})")
    return t7, s2t, med2


def rotulo_inv(r):
    cid, q = r.split(" ")[0], r.split(" ")[-1]
    return f"{cid.lower()}_{q}"


# ================================================================== S1 (TA1) and S3 (T7b)
def s1(og):
    import tempfile
    tmp = SCRATCH / "ta1"
    tmp.mkdir(parents=True, exist_ok=True)
    antigo = og.OUT
    og.OUT = tmp
    try:
        og.gerar_ta1()
    finally:
        og.OUT = antigo
    reg(og.ENT["B7.2"])
    saidas = {}
    for nome, fn in (("TA1a_hiperparametros_arquitetura_perda_v3-12", "TA1a_v4"), ("TA1b_hiperparametros_otimizacao_custo_v3-12", "TA1b_v4")):
        txt = (tmp / f"{nome}.tex").read_text(encoding="utf-8")
        assert "Setting & Graph regressor & Perceptron \\\\" in txt
        txt = txt.replace("Setting & Graph regressor & Perceptron \\\\", "Setting & GNN & MLP \\\\")
        txt = txt.replace("Test set used in selection", "Test partition in selection")
        txt = txt.replace("Best epoch, median (range) &", "Best epoch, median (range) ($g = 5$~km) &")
        assert "test set" not in txt.lower()
        assert not re.search(r"Graph regressor|graph regressor|Perceptron|perceptron", txt)
        ln = txt.split("\n")
        head = ["% v4-1 (26/10/2026): TA1 regenerated by importing analysis/v3_12_gerar_tabelas.gerar_ta1 (original as run: sha256 " + SHA_OG_TAB + ", unchanged); only the header changes: Graph regressor / Perceptron -> GNN / MLP.",
                "% fonte: " + str(og.ENT["B7.2"]) + " sha256 " + SHAS[str(og.ENT["B7.2"])],
                "% predictor names: " + PRED_LINHA[fn],
                "% generator: analysis/revision_r1/v4_gerar_tabelas.py"]
        saidas[fn] = "\n".join(head + ln)
    return saidas


def tab_s3(og):
    a = load(og.ENT["A4"])
    r7 = load(F5 / "R7_resumo.json")
    e6 = load(F6 / "E6_resumo.json")
    piso = e6["limiar_ruido_repeticao_db"]
    assert piso == 0.132
    cap = capturar(og, "gerar_t7b")["T7b_deriva_modelo_A4_v3-12"]
    corpo, rzs = [], {"gnn": [], "mlp": []}
    for k, c in a["celulas"].items():
        cid, q = k.split("_")
        cols = [f"{cid.capitalize()} {q}"]
        for mod in ("gnn", "mlp"):
            m = c["por_modelo"][mod]
            mean, sd = m["media_entre_sorteios"]["mae_rssi_validos_db"], m["dp_entre_sorteios"]["mae_rssi_validos_db"]
            par = r7["pares"].get(f"{k}__{mod}")
            if par:
                assert abs(par["dp_entre_5_sorteios_A4"] - sd) < 1e-12
            rz_seed = f"{par['razao_sorteio_sobre_semente']:.1f}" if par else "--"
            cols += [f"{mean:.2f} ({sd:.2f})", f"{sd / piso:.1f}", rz_seed]
            rzs[mod].append(sd / piso)
        corpo.append(" & ".join(cols) + " \\\\")
    d = a["deriva_2_2"]
    corpo.append("\\midrule")
    corpo.append("SD across the four cells of the per-cell means & " + f"{d['gnn']['dp_entre_celulas_das_medias_validos_db']:.2f} & -- & -- & {d['mlp']['dp_entre_celulas_das_medias_validos_db']:.2f} & -- & -- \\\\")
    # cross-check against the captured original (mean (SD) of the 4 cells)
    for ln, cols in zip(cap["linhas"][:4], corpo[:4]):
        assert ln.split(" & ")[1] == cols.split(" & ")[1] and ln.split(" & ")[2] == cols.split(" & ")[4], (ln, cols)
    cab = [" & " + mc(3, "c", "GNN") + " & " + mc(3, "c", "MLP") + " \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
           "Cell & " + " & ".join([cel("Valid MAE (dB): \\\\ \\relax mean (SD)"), cel("SD / repeat \\\\ \\relax difference$^{a}$"), cel("SD / SD across \\\\ \\relax seeds$^{b}$")] * 2) + " \\\\"]
    notas = [f"Five split draws, $g = 5$~km, $b = 2$~km, training seed 42, valid-node MAE of received power; SD: between-draw standard deviation. "
             f"$^{{a}}$ Between-draw SD divided by the same-seed repeat difference ({piso:.3f}~dB, the largest difference between two training repeats with the same seed and partition, measured at 5~km). "
             f"$^{{b}}$ Between-draw SD divided by the SD across five training seeds ({r7['sementes']['A3_sementes_de_treino_sorteio_42'][0]}--{r7['sementes']['A3_sementes_de_treino_sorteio_42'][-1]}) at split seed 42, run in Q1 and Q3 only; --: not run."]
    return Tabela("T7b_deriva_modelo_A4_v4", "lcccccc", 7, cab, corpo, notas, [og.ENT["A4"], F5 / "R7_resumo.json", F6 / "E6_resumo.json", OG_TAB],
                  ["Tab. S3 (v4-1): sem o identificador de sorteios de bastidor; colunas novas 'SD / repeat difference' (SD / 0,132 dB) e 'SD / SD across seeds' por modelo; sem seta de menor-e-melhor.",
                   "Valores: fase? gpu/A4/agregado_A4.json (media e dp por celula e modelo, como gerar_t7b do original, que foi importado e reconferido), razoes de semente de fase5/R7_resumo.json (pares.<celula>__<modelo>.razao_sorteio_sobre_semente); razao /0,132 = dp / limiar_ruido_repeticao_db (fase6/E6_resumo.json)."],
                  original=f"{OG_TAB.relative_to(ROOT).as_posix()} (original as run: sha256 {SHA_OG_TAB})"), (min(rzs['gnn']), max(rzs['gnn']), min(rzs['mlp']), max(rzs['mlp']))


# ================================================================== Tab. 8 (T9 G1)
def tab8(og9):
    J = {k: load(p) for k, p in og9.ENT.items()}
    v2 = load(F6 / "V2bis_resumo.json")
    e6 = load(F6 / "E6_resumo.json")
    assert e6["limiar_ruido_repeticao_db"] == og9.PISO
    linhas = og9.construir(J)
    pc = v2["pares_cruzados"]
    corpo, rz_all, excl = [], [], {}
    for i, r in enumerate(linhas):
        if i and r["modelo"] == "gnn":
            corpo.append("\\midrule")
        rz_all.append(r["razao"])
        if r["celula"].endswith("Q1"):
            p = pc[f"{r['celula']}.{r['modelo']}"]
            assert abs(p["dp20_s42"] - r["dp"]) < 1e-9
            assert abs(p["R20"] - r["dp"] / r["pooled"]) < 0.05 * p["R20"] / 10   # R20 publicado = dp(20 sorteios) / SD pooled entre sementes
            lo, hi = p["R20_ic_F_19_2n"]
            razao = f"{fd(p['R20'], 2)} [{fd(lo, 2)}; {fd(hi, 2)}]"
        else:
            razao = "--"
        if (r["sv"] or r["nt"]) and r["modelo"] == "gnn":
            excl[r["rotulo"]] = (r["sv"], r["nt"])
        corpo.append(" & ".join([r["rotulo"] if r["modelo"] == "gnn" else "", r["mrot"], str(r["n"]), fd(r["media"], 2), fd(r["dp"], 2),
                                 "--" if r["pooled"] is None else fd(r["pooled"], 2), razao, fd(r["spearman"], 2)]) + " \\\\")
    paridade = "; ".join(f"{r['rotulo']} {r['paridade']:.3f}" for r in linhas if r["modelo"] == "gnn")
    txt_excl = "; ".join(f"{k}: {sv} without a valid test node" + (f" and {nt} whose validation partition has no antenna--terrain edge" if nt else "") for k, (sv, nt) in excl.items())
    cab = [" & & & " + mc(2, "c", "Across draws") + " & & & \\\\",
           "\\cmidrule(lr){4-5}",
           "Cell & Model & " + " & ".join([cel("Draws used \\\\ \\relax (of 20)"), "Mean MAE (dB)", "SD (dB)", cel("Pooled SD \\\\ \\relax across seeds (dB)"),
                                         cel("SD ratio, draw/seed \\\\ \\relax [95\\% interval]"), "Spearman"]) + " \\\\"]
    notas = ["MAE: valid-node mean absolute error of received power; mean and SD over the draws used, training seed 42, $g = 10$~km, $b = 2$~km, eight epochs. "
             "SD ratio, draw/seed: SD across the 20 draws divided by the pooled SD across training seeds, $R_{20}$ with its $F$-based 95\\% interval; --: not run, the crossed seed block covers only Bauru Q1 and Campinas Q1. "
             "Spearman: rank correlation with the constant predictor's valid-node MAE across draws. "
             f"Draws excluded from the 20 ({txt_excl}). "
             f"SD across draws divided by the same-seed repeat difference ({og9.PISO:.3f}~dB): {min(rz_all):.1f} to {max(rz_all):.1f} over the eight rows. "
             f"Sentinel parity, median $|\\mathrm{{GNN}}-\\mathrm{{MLP}}|$ (dB), one value per cell: {paridade}."]
    return Tabela("T9_deriva_modelo_G1_v4", "llcccccc", 8, cab, corpo, notas,
                  [og9.ENT["G1_bloco1"], og9.ENT["G1_bloco2"], og9.ENT["G1_bloco3"], og9.ENT["G1_bloco4"], F6 / "V2bis_resumo.json", F6 / "E6_resumo.json", OG_T9],
                  ["Tab. 8 (v4-1): 10 -> 8 colunas sem resizebox; cabecalho em 3 linhas (\\multicolumn); coluna NOVA 'SD ratio, draw/seed [95% interval]' = R_20 e IC F de fase6/V2bis_resumo.json (pares_cruzados.<celula>.<modelo>.R20 / R20_ic_F_19_2n; a grandeza publicada, 10.4; 3.61; 7.60; 2.98);",
                   "'Draws excluded' (2+2, 1+0) saiu da tabela e foi para a nota por extenso (sv = n_sorteios_sem_validos, nt = n_sorteios_nao_treinaveis dos agregados); razao contra 0,132 dB e paridade do sentinela foram para a nota (a razao e dp/0,132; faixa calculada dos oito dp).",
                   "Linhas (media, SD, n, Spearman, SD pooled) = og9.construir(J) do gerador v3_13_gerar_T9_G1.py importado sem alteracao; R_20 conferido contra dp/SD pooled; 'same' da paridade deixou de existir (valores por celula na nota)."],
                  original=f"{OG_T9.relative_to(ROOT).as_posix()} (original as run: sha256 {SHA_OG_T9})"), (min(rz_all), max(rz_all))


# ================================================================== T_V1 (coverage)
def tab_v1(chave="constante__validos", nome="T_V1_cobertura", nome_pred="constant predictor"):
    r = load(F6 / "V1_resumo.json")
    b5 = load(F5 / "blocos_com_no_valido.json")
    ps = load(F6 / "V1_por_sorteio.json")
    pc = r["por_celula"][chave]
    corpo, checks = [], {}
    cid_prev = None
    for k in sorted(pc, key=lambda x: (["bauru", "campinas", "lins", "sorocaba"].index(x.split("_")[0]), x)):
        c = pc[k]
        cid = k.split("_")[0]
        if cid_prev and cid != cid_prev:
            corpo.append("\\midrule")
        cid_prev = cid
        nvb = b5["por_celula"][k]["blocos_com_ao_menos_1_no_valido"]
        assert nvb == c["n_blocos_com_no_valido"]
        sem_int = c["n_Err_definido"] - c["n_v_definido"]
        # per-draw check: Err defined and k < 2
        sorteios = ps["celulas"][k]["sorteios"]
        n_alt = sum(1 for s in sorteios if s[chave]["Err"] is not None and s[chave]["k"] < 2)
        assert n_alt == sem_int, (k, n_alt, sem_int)
        cov = c["cobertura"]
        vals = {m: cov[m] for m in ("base", "t", "jkt", "wild_t")}
        melhor = min(abs(v - 0.95) for v in vals.values())
        cel_cov = []
        for m in ("base", "t", "jkt", "wild_t"):
            s = fd(vals[m], 2)
            cel_cov.append(f"\\textbf{{{s}}}" if abs(abs(vals[m] - 0.95) - melhor) < 1e-12 else s)
        kmed = f"{c['k_mediana']:.1f} ({c['k_min']}--{c['k_max']})"
        corpo.append(" & ".join([rotulo_celula(k), str(nvb), kmed, str(sem_int)] + cel_cov) + " \\\\")
        checks[k] = dict(N_blocos_validos=nvb, sorteios_sem_intervalo=sem_int, k=kmed)
    cab = [" & & & & " + mc(4, "c", "Coverage") + " \\\\",
           "\\cmidrule(lr){5-8}",
           "Cell & " + " & ".join(["$N_{\\mathrm{blocks}}^{\\mathrm{valid}}$", cel("$k$ \\\\ \\relax (median, min--max)"), cel("Draws without \\\\ \\relax interval (of 60)"),
                                   "Plug-in", "$t$", "Jackknife-$t$", "Wild-$t$"]) + " \\\\"]
    notas = [f"{nome_pred[0].upper() + nome_pred[1:]}, valid test nodes; 60 split draws per cell, $g = 10$~km, $b = 2$~km. "
             "$N_{\\mathrm{blocks}}^{\\mathrm{valid}}$: blocks, of the 132 of the cell, with at least one valid node. "
             "$k$: test blocks with a valid node in a draw. "
             "Draws without interval: draws with a defined error and $k < 2$. "
             "Coverage: fraction of the draws with an interval in which the interval contains the protocol estimand (mean of the reported error over draws); nominal level 0.95; bold: closest to nominal in the row. "
             "Plug-in: $\\mathrm{Err} \\pm 2\\sqrt{v}$ ($v$: plug-in cluster-variance estimate of one partition). "
             "$t$: $\\mathrm{Err} \\pm t_{k-1}\\sqrt{v}$. "
             "Jackknife-$t$: $\\mathrm{Err} \\pm t_{k-1}\\sqrt{v_{\\mathrm{jk}}}$, with the block jackknife variance. "
             f"Wild-$t$: $\\mathrm{{Err}} \\pm t_{{k-1}}\\,\\mathrm{{SD}}(\\mathrm{{Err}}^{{*}})$, Rademacher wild cluster bootstrap by test block ($R = {r['R_bootstrap']}$)."]
    return Tabela(nome, "lccccccc", 8, cab, corpo, notas, [F6 / "V1_resumo.json", F6 / "V1_por_sorteio.json", F5 / "blocos_com_no_valido.json"],
                  ["Tab. 6 (cobertura, v4-1), padrao booktabs, nomes 'Bauru Q1', 8 colunas. Fonte: fase6/V1_resumo.json (por_celula." + chave + ": k_mediana/k_min/k_max, cobertura.{base,t,jkt,wild_t}, n_Err_definido, n_v_definido) e fase5/blocos_com_no_valido.json (blocos_com_ao_menos_1_no_valido).",
                   "'Draws without interval' = n_Err_definido - n_v_definido (sorteios com k < 2 entre os de Err definido), recontado por asserts a partir de fase6/V1_por_sorteio.json (Err nao nulo e k < 2). Negrito = cobertura mais proxima de 0,95 (empates em negrito).",
                   "Preditor: " + chave + ("; nome de espaco livre usado: " + FSPL_POR_SORTEIO if chave.startswith("fspl") else "; sem preditor de espaco livre")]), checks


# ================================================================== T_V3 (16 cells)
def tab_v3():
    r = load(F6 / "V3_tabela_celulas.json")
    pc = r["por_celula"]
    assert all(c["n_nos"] == 12960000 and c["n_blocos_10km_ocupados"] == 132 for c in pc.values())
    corpo, cid_prev = [], None
    for k in sorted(pc, key=lambda x: (["bauru", "campinas", "lins", "sorocaba"].index(x.split("_")[0]), x)):
        c = pc[k]
        if cid_prev and c["cidade"] != cid_prev:
            corpo.append("\\midrule")
        cid_prev = c["cidade"]
        corpo.append(" & ".join([rotulo_celula(k), fd(c["n_validos"] / 1e6, 2), fd(c["fracao_sentinela"], 3), str(c["n_blocos_com_no_valido"]), str(c["n_transmissores"]),
                                 fd(c["dist_mediana_validos_km"], 2), fd(c["rssi_mediana_validos_dBm"], 1)]) + " \\\\")
    cab = ["Cell & " + " & ".join([cel("Valid nodes \\\\ \\relax (M)"), cel("Sentinel \\\\ \\relax fraction"), cel("Blocks with \\\\ \\relax a valid node"), cel("Transmitters \\\\ \\relax (city)"),
                                    cel("Median distance to nearest \\\\ \\relax transmitter (km)"), cel("Median received power at \\\\ \\relax valid nodes (dBm)")]) + " \\\\"]
    notas = ["Every cell has 12.96 M nodes and 132 blocks of 10 km. M: millions. Sentinel fraction: share of the cell's nodes that are sentinel nodes. Medians are over the valid nodes of the cell."]
    return Tabela("T_V3_celulas", "lcccccc", 7, cab, corpo, notas, [F6 / "V3_tabela_celulas.json"],
                  ["Tab. das 16 celulas (v4-1): cabecalhos em ingles, 7 colunas (N e 'Blocos ocup.' constantes -> nota; 'Sentinela (M)' derivavel -> fora; minimo do RSSI fora), fracao sentinela logo apos os nos validos.",
                   "Fonte: fase6/V3_tabela_celulas.json (por_celula.<celula>.n_validos, fracao_sentinela, n_blocos_com_no_valido, n_transmissores, dist_mediana_validos_km, rssi_mediana_validos_dBm); n_nos == 12960000 e n_blocos_10km_ocupados == 132 conferidos por assert nas 16 celulas.",
                   "Colunas de elevacao omitidas: sem nome no tensor (V3_tabela_celulas.json, colunas_vazias)."]), None


# ================================================================== T_V2bis (ratios)
def tab_v2bis():
    r = load(F6 / "V2bis_resumo.json")
    pc = r["pares_cruzados"]
    ordem = ["bauru_Q1.gnn", "bauru_Q1.mlp", "campinas_Q1.gnn", "campinas_Q1.mlp"]
    B = r["parametros"]["B"]

    def iv(lo, hi):
        return f"[{fd(lo, 2)}; {fd(hi, 2)}]"

    def lin(rot, f):
        return rot + " & " + " & ".join(f(pc[k]) for k in ordem) + " \\\\"
    corpo = [
        lin("$\\hat\\theta$", lambda p: fd(p["theta_hat"], 2)),
        lin("\\hspace{1em}$F$ 95\\% interval", lambda p: iv(*p["theta_ic_F95"])),
        "\\midrule",
        lin("$R$", lambda p: fd(p["R"], 2)),
        lin("\\hspace{1em}95\\% interval", lambda p: iv(*p["R_ic"])),
        lin("\\hspace{1em}leave-one-draw-out range", lambda p: iv(p["deixa_um_fora"]["R_min"], p["deixa_um_fora"]["R_max"])),
        "\\midrule",
        lin("$R_{20}$", lambda p: fd(p["R20"], 2)),
        lin("\\hspace{1em}$F$ 95\\% interval", lambda p: iv(*p["R20_ic_F_19_2n"])),
        lin("\\hspace{1em}BCa interval, 95\\%", lambda p: iv(*p["bootstrap_por_id"]["ic_bca"])),
        lin("\\hspace{1em}replicates discarded (of " + str(B) + ")", lambda p: f"{p['bootstrap_por_id']['replicas_descartadas']} ({100 * p['bootstrap_por_id']['fracao_descartada']:.1f}\\%)"),
    ]
    cab = [" & " + mc(2, "c", "Bauru Q1") + " & " + mc(2, "c", "Campinas Q1") + " \\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
           "Quantity & GNN & MLP & GNN & MLP \\\\"]
    nb, nc = (pc["bauru_Q1.gnn"]["n_sorteios_cruzados"], pc["campinas_Q1.gnn"]["n_sorteios_cruzados"])
    assert pc["bauru_Q1.mlp"]["n_sorteios_cruzados"] == nb and pc["campinas_Q1.mlp"]["n_sorteios_cruzados"] == nc
    nome = {5: "five", 10: "ten"}
    notas = [f"$\\hat\\theta$: ratio of the draw variance component to the seed variance component, from {nome[nb]} (Bauru Q1) or {nome[nc]} (Campinas Q1) split draws, three training seeds each.",
             "$R = \\sqrt{1 + \\hat\\theta}$: ratio of the SD across draws to the SD across seeds (interval truncated at 1 when the lower limit of $\\hat\\theta$ is negative).",
             "$R_{20}$: SD across the 20 draws at training seed 42 divided by the pooled SD across seeds.",
             f"BCa: bias-corrected and accelerated bootstrap of $R_{{20}}$, resampling the draws, $B = {B}$ replicates; replicates discarded are counted in the last row. "
             "Intervals and ranges are written $[\\mathrm{lower}; \\mathrm{upper}]$, to two decimals; the range is the minimum and maximum of $R$ when each draw is dropped in turn."]
    return Tabela("T_V2bis_razoes", "lcccc", 5, cab, corpo, notas, [F6 / "V2bis_resumo.json"],
                  ["Razoes sorteio/semente (v4-1): transposta (linhas = estimador, colunas = celula x modelo), 5 colunas, um formato de intervalo '[lo; hi]' (faixa leave-one-out no mesmo formato, rotulada 'range') e duas casas em tudo (v4.1, como o corpo); replicas descartadas com denominador B (n e % na celula, B no rotulo); nota com uma linha por simbolo (theta, R, R20, BCa).",
                   "Fonte: fase6/V2bis_resumo.json (pares_cruzados.<celula>.<modelo>: theta_hat, theta_ic_F95, R, R_ic, deixa_um_fora.{R_min,R_max}, R20, R20_ic_F_19_2n, bootstrap_por_id.{ic_bca,replicas_descartadas,fracao_descartada}; parametros.B)."]), None


# ================================================================== T_V4 (bias)
def tab_v4():
    r = load(F6 / "V4_tabela_vies.json")
    PRED = {"constante": "Constant", "fspl_b": FSPL_UMA_VEZ}
    POP = {"validos": "Valid nodes", "todos": "All nodes"}

    def f(t):
        hw = t.get("meia_largura", t.get("meia_largura_2EP"))
        return f"{fd(t['estimativa'], 2)}$\\,\\pm\\,${fd(hw, 2)}" + ("$^{*}$" if t["distinguido_de_zero"] else "")
    corpo, cel_prev, pred_prev, n_ast = [], None, None, 0
    for l in r["linhas"]:
        if cel_prev and l["celula"] != cel_prev:
            corpo.append("\\midrule")
            pred_prev = None
        T = l["termos"]
        cols = [f(T[k]) for k in ("desenho_fechado", "desenho_freq200", "vies_total_H_com_buffer", "razao_H_com_buffer", "razao_H0_sem_buffer")]
        n_ast += sum(1 for k in ("desenho_fechado", "desenho_freq200", "vies_total_H_com_buffer", "razao_H_com_buffer", "razao_H0_sem_buffer") if T[k]["distinguido_de_zero"])
        corpo.append(" & ".join([rotulo_celula(l["celula"]) if l["celula"] != cel_prev else "", PRED[l["preditor"]] if l["preditor"] != pred_prev else "", POP[l["populacao"]]] + cols) + " \\\\")
        cel_prev, pred_prev = l["celula"], l["preditor"]
    cab = [" & & & " + mc(2, "c", "Design term (dB)") + " & " + cel("Total bias \\\\ \\relax (dB)") + " & " + mc(2, "c", "Ratio term (dB)") + " \\\\",
           "\\cmidrule(lr){4-5}\\cmidrule(lr){7-8}",
           "Cell & Predictor & Nodes & Closed form & Frequency & With buffer & With buffer & No buffer \\\\"]
    notas = ["Entries: estimate $\\pm$ half-width of the interval, in dB; $^{*}$: the interval excludes zero. "
             "Closed form: midpoint $\\pm$ half-width of the bounds $[T_{\\inf}, T_{\\sup}]$; all other entries: estimate $\\pm$ 2 Monte Carlo standard errors. "
             "Frequency: design term estimated from 200 draws. Total bias: bias of the block hold-out error against the domain mean. "
             "With buffer: $b = 2$~km; no buffer: $b = 0$. Four Q1 cells, $g = 10$~km."]
    return Tabela("T_V4_vies", "lllccccc", 8, cab, corpo, notas, [F6 / "V4_tabela_vies.json"],
                  ["Vies (v4-1): estimativa +- meia-largura, asterisco quando o intervalo exclui zero (distinguido_de_zero), cabecalhos em ingles com \\multicolumn, rotulos 'Bauru Q1', predicao 'FSPL, calibrated once'.",
                   "Fonte: fase6/V4_tabela_vies.json (linhas[].termos.{desenho_fechado, desenho_freq200, vies_total_H_com_buffer, razao_H_com_buffer, razao_H0_sem_buffer}: estimativa, meia_largura | meia_largura_2EP, distinguido_de_zero).",
                   "Preditor 'fspl_b' desta tabela = fspl_calibrado_b de R2/R3 = " + FSPL_UMA_VEZ + " (single calibration at split seed 42; analysis/v3_12_R2_referencia_desenho.py)."]), n_ast


# ================================================================== T_E6 (inversions)
def tab_e6():
    r = load(F6 / "E6_resumo.json")
    piso = r["limiar_ruido_repeticao_db"]
    corpo, prev = [], None
    for k in ("bauru_Q1", "campinas_Q1", "bauru_Q3", "campinas_Q3"):
        c = r["celulas"][k]
        if prev:
            corpo.append("\\midrule")
        prev = k
        for pop, rot in (("validos", "Valid nodes"), ("todos", "All nodes")):
            x = c[pop]
            corpo.append(" & ".join([rotulo_celula(k) if pop == "validos" else "", rot, str(x["n_sorteios"]), str(x["n_d_pos_gnn_lt_mlp"]), str(x["n_d_neg_mlp_lt_gnn"]),
                                     str(x["n_abs_d_le_0_132"]), f"{fd(x['media_d'], 2)}$\\,\\pm\\,${fd(x['dp_d'], 2)}", "yes" if x["inversao"] else "no"]) + " \\\\")
    cab = [" & & & " + mc(3, "c", "Draws in which") + " & & \\\\",
           "\\cmidrule(lr){4-6}",
           "Cell & Population & " + " & ".join(["Draws", "GNN lower", "MLP lower", cel("$|d| \\le 0.132$ \\\\ \\relax dB"), cel("$d$: mean $\\pm$ SD \\\\ \\relax (dB)"), cel("Order \\\\ \\relax inverts")]) + " \\\\"]
    notas = [f"$d = \\mathrm{{MAE}}_{{\\mathrm{{MLP}}}} - \\mathrm{{MAE}}_{{\\mathrm{{GNN}}}}$ per draw, so $d > 0$ means the GNN error is lower. "
             f"GNN lower: draws with $d > 0$; MLP lower: draws with $d < 0$; $|d| \\le {piso:.3f}$~dB: within the same-seed repeat difference. "
             "Order inverts: the sign of $d$ differs from the sign of its mean in at least one draw. "
             "Valid nodes: draws with a valid test node; all nodes: all trained draws with predictions from both models. $g = 10$~km, $b = 2$~km, training seed 42."]
    return Tabela("T_E6_inversoes", "llcccccc", 8, cab, corpo, notas, [F6 / "E6_resumo.json"],
                  ["Inversoes de ordem GNN x MLP (v4-1): cabecalhos em ingles, rotulos 'Bauru Q1', populacoes 'Valid nodes'/'All nodes', \\multicolumn, booktabs.",
                   "Fonte: fase6/E6_resumo.json (celulas.<celula>.{validos,todos}.{n_sorteios, n_d_pos_gnn_lt_mlp, n_d_neg_mlp_lt_gnn, n_abs_d_le_0_132, media_d, dp_d, inversao}; limiar_ruido_repeticao_db)."]), None


# ================================================================== Tab. 5 (T6 design reference)
def tab_t6(og):
    """T6 of the earlier generator (gerar_t6, captured). Predictor 'fspl_calibrado_b' of results/fase4/R2_resumo.json = FSPL calibrated ONCE on a
    reference partition (split seed 42; analysis/v3_12_R2_referencia_desenho.py); the earlier generator labelled it 'FSPL (second calibration)'."""
    cap = capturar(og, "gerar_t6")["T6_referencia_desenho_v3-12"]
    antigo = "FSPL (second calibration)"
    n_rot = sum(ln.count(antigo) for ln in cap["linhas"])
    assert n_rot == 4, n_rot
    corpo = []
    for ln in cap["linhas"]:
        ln = ln.replace(antigo, FSPL_UMA_VEZ)
        pre = ""
        if ln.startswith("\\midrule"):
            pre, ln = "\\midrule\n", ln.split("\n", 1)[1]
        cols = [c.strip() for c in ln.split(" & ")]
        assert len(cols) == 7, cols
        corpo.append(pre + " & ".join(cols) + " \\\\")
    cab = ["Cell & Predictor & Population & " + " & ".join([cel("Block hold-out \\\\ \\relax bias (SE), dB"), cel("Block hold-out \\\\ \\relax RMSE, dB"),
           cel("Random-sample \\\\ \\relax RMSE, dB"), cel("Effective size \\\\ \\relax $n/\\mathrm{deff}$")]) + " \\\\"]
    notas = [f"Hold-out in blocks ($g = 10$~km, $b = 2$~km), 200 draws; bias with Monte Carlo standard error over the draws (in parentheses). "
             f"RMSE: root mean squared error of the estimate of the domain mean error, from the block hold-out and from a simple random sample of the same number of nodes. "
             f"deff: ratio of the hold-out variance to the random-sample variance; $n$: mean number of scored nodes. "
             f"{FSPL_UMA_VEZ}: the offset is fixed once on a reference partition (split seed 42) and not recomputed per draw."]
    return Tabela("T6_referencia_desenho_v4", "lllcccc", 7, cab, corpo, notas, [og.ENT["R2_resumo"], og.ENT["R2_por_sorteio"], og.ENT["bc_fismat"], og.ENT["veredito_fismat"], OG_TAB],
                  ["Tab. 5 antiga (v4-1): T6_referencia_desenho regerada; rotulo 'FSPL (second calibration)' do gerador antigo -> 'FSPL, calibrated once' (4 ocorrencias, so rotulo; valores = gerar_t6 capturado).",
                   "Preditor: coluna 'fspl_calibrado_b' de fase4/R2_resumo.json (resumo_por_celula.<celula>.fspl_calibrado_b.{todos,validos}); single calibration of e_i at split seed=42 (analysis/v3_12_R2_referencia_desenho.py), not recalibrated per draw.",
                   "Cabecalho de 4 linhas -> 3 linhas, unidades (dB) no cabecalho, nota de definicao no pe."],
                  original=f"{OG_TAB.relative_to(ROOT).as_posix()} (original as run: sha256 {SHA_OG_TAB})")


# ================================================================== Tab. 3 (T2 retention)
def tab_t2(og):
    """gerar_t2 do original (capturado: linhas cruas, ja conferidas por assert contra T2_retencao_v3.csv dentro do original); so cabecalho e nota mudam."""
    cap = capturar(og, "gerar_t2")["T2_retencao_v3-12"]
    corpo = []
    for ln in cap["linhas"]:
        corpo.append(og._linha(ln) + " \\\\")
    cab = ["$g$ (km) & City & " + " & ".join([cel("Node-level test retention \\\\ \\relax (mean $\\pm$ SD)"), cel("Node-level \\\\ \\relax validation retention"), cel("Law test / \\\\ \\relax validation"),
                                            cel("Law vs.\\ node-level, \\\\ \\relax test (\\%)"), cel("Law vs.\\ node-level, \\\\ \\relax validation (\\%)")]) + " \\\\"]
    notas = ["Node-level retention: fraction of the nodes of the partition that survive the trimming, mean over 200 split draws and over the four cells of a city; SD: standard deviation between draws. "
             "Law: closed-form retention of Equation~\\eqref{eq:R} with $q_{\\mathrm{te}} = k_{\\mathrm{te}}/N$. "
             "Law vs.\\ node-level: relative difference of the law from the node-level retention, as a range over the four cells of a city; a single value where both ends coincide at one decimal."]
    return Tabela("T2_retencao_v4", "llccccc", 7, cab, corpo, notas, [og.ENT["1.8"], og.ENT["1.8b"], og.ENT["T2_antiga_csv"], OG_TAB],
                  ["Tab. 3 (v4.1): T2_retencao regerada por gerar_t2 do v3_12_gerar_tabelas.py (importado sem alteracao, capturado; assert contra T2_retencao_v3.csv roda no original). Muda so o cabecalho ('Nodal val.' -> 'Node-level validation retention', '+- sd' -> '+- SD', 'quadrants' -> 'cells') e a nota de rodape.",
                   "Fontes: fase1/1.8 e 1.8b (por_celula.<celula>.{nodal_te_media, nodal_te_dp, nodal_va_media, lei_te, lei_va, erro_lei_vs_nodal_te, erro_lei_vs_nodal_va})."],
                  original=f"{OG_TAB.relative_to(ROOT).as_posix()} (original as run: sha256 {SHA_OG_TAB})")


# ================================================================== Tab. 1 (notation)
def tab_notacao():
    src = TABLES_V3 / "TN_notacao_v3-12.tex"
    txt = reg(src).read_text(encoding="utf-8")
    ln = txt.split("\n")
    k = ln.index("\\bottomrule")
    novas = [
        ("FSPL, calibrated once", "free-space loss predictor whose offset is fixed once on a reference partition (split seed 42)", "Table~\\ref{tab:designref}"),
        ("FSPL, recalibrated per draw", "free-space loss predictor whose offset is recomputed on the training partition of each draw", "Table~\\ref{tab:inversion}"),
        ("$\\mathrm{deff}$", "design effect: ratio of the hold-out variance to the random-sample variance", "Table~\\ref{tab:designref}"),
        ("$\\hat\\theta$", "ratio of the draw variance component to the seed variance component", "Section~\\ref{sec:evalpop}"),
        ("$R = \\sqrt{1+\\hat\\theta}$", "ratio of the SD across draws to the SD across seeds", "Table~\\ref{tab:ratios}"),
        ("$R_{20}$", "between-draw SD at one training seed over the pooled SD across seeds", "Table~\\ref{tab:ratios}"),
        ("$k$", "test blocks with a valid node in a draw", "Table~\\ref{tab:coverage_one_partition}"),
    ]
    ln[k:k] = [" & ".join(r) + " \\\\" for r in novas]
    out = "\n".join(ln)
    n0 = out.count("block counts per class")
    out = out.replace("block counts per class", "block counts per partition set (training, validation, test)")
    out = out.replace("scored set", "scored population").replace("on the test set of draw", "on the test partition of draw").replace("on the test set of draw $\\sigma$", "on the test partition of draw $\\sigma$")
    out = out.replace("conditional error of a draw", "realised error of a draw")
    out = out.replace("nodal retention", "node-level retention")
    assert "test set" not in out and "scored set" not in out
    head = ["% Tab. 1 (v4.1): notacao = tables_v3-12/TN_notacao_v3-12.tex (sha256 " + SHAS[str(src)] + ") + 7 linhas (FSPL x2, deff, theta, R, R20, k); 'block counts per class' -> 'per partition set'; 'test set' -> 'test partition'.",
            "% generator: analysis/revision_r1/v4_gerar_tabelas.py (tab_notacao); no number."]
    head.append("% layout: \\footnotesize (a tabela de 38 linhas nao cabia numa pagina em fonte normal: overfull vbox de 64 pt)")
    return "\n".join(head) + "\n{\\footnotesize\n" + out.rstrip("\n") + "\n}\n"


# ================================================================== compilation and natural-width probe
PREAMBULO = r"""\documentclass[mathematics,article,submit,moreauthors]{Definitions/mdpi}
\newcommand{\celula}[1]{\begin{tabular}[c]{@{}c@{}}#1\end{tabular}}
\newsavebox{\tabcaixa}
\newcommand{\tabajusta}[1]{\sbox{\tabcaixa}{#1}\ifdim\wd\tabcaixa>\linewidth\resizebox{\linewidth}{!}{\usebox{\tabcaixa}}\else\usebox{\tabcaixa}\fi}
\providecommand{\tabcorpo}[1]{\begingroup\renewenvironment{table}[1][]{}{}\renewcommand{\caption}[1]{}\renewcommand{\label}[1]{}\input{#1}\endgroup}
\firstpage{1}\makeatletter\setcounter{page}{\@firstpage}\makeatother
\pubvolume{1}\issuenum{1}\articlenumber{0}\pubyear{2026}\copyrightyear{2026}
\datereceived{ }\daterevised{ }\dateaccepted{ }\datepublished{ }
\Title{Teste de compilacao das tabelas v4-1}\Author{A $^{1}$}\AuthorNames{A}\address{$^{1}$ X}\corres{c}
\abstract{a}\keyword{k}\MSC{62}
\begin{document}
\newsavebox{\probe}
"""


def preparar_ambiente():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    d = SCRATCH / "Definitions"
    if not d.exists():
        if not DEFINITIONS:
            raise SystemExit("set MDPI_DEFINITIONS to the folder 'Definitions' of the MDPI LaTeX class (not distributed)")
        subprocess.run(["cp", "-r", DEFINITIONS, str(d)], check=True)


def medir_larguras(tabelas):
    """escolhe para cada tabela o primeiro layout de CFGS cuja largura natural (sem resizebox) cabe em LIMITE_PT; devolve {nome: largura} e fixa t.cfg."""
    preparar_ambiente()
    corpo = PREAMBULO
    for t in tabelas:
        for i in range(len(CFGS)):
            (SCRATCH / f"probe_{t.nome}_{i}.tex").write_text(t.tex(None, i), encoding="utf-8")
            corpo += "\\sbox{\\probe}{\\input{probe_%s_%d.tex}}\\typeout{WD:%s_%d=\\the\\wd\\probe}\n" % (t.nome, i, t.nome, i)
    corpo += "\\end{document}\n"
    (SCRATCH / "probe.tex").write_text(corpo, encoding="utf-8")
    subprocess.run([TEXLIVE, "-interaction=nonstopmode", "probe.tex"], cwd=SCRATCH, capture_output=True)
    log = (SCRATCH / "probe.log").read_text(errors="ignore")
    w = {m.group(1): float(m.group(2)) for m in re.finditer(r"WD:([^=\s]+)=([\d.]+)pt", log)}
    larg = {}
    for t in tabelas:
        for i in range(len(CFGS)):
            assert f"{t.nome}_{i}" in w, ("probe sem largura", t.nome, i, log[-1500:])
        ok = [i for i in range(len(CFGS)) if w[f"{t.nome}_{i}"] <= LIMITE_PT]
        t.cfg = ok[0] if ok else len(CFGS) - 1
        t.cabe = bool(ok)
        larg[t.nome] = w[f"{t.nome}_{t.cfg}"]
    return larg


def compilar_teste(nomes_tex, destino_tex, titulo="Teste de compilacao das tabelas v4-1"):
    preparar_ambiente()
    corpo = PREAMBULO.replace("Teste de compilacao das tabelas v4-1", titulo)
    for n in nomes_tex:
        corpo += "\\begin{table}[H]\\caption{(legenda do redator: %s)}\\begin{adjustwidth}{-\\extralength}{0cm}\\centering\\tabajusta{\\tabcorpo{%s.tex}}\\end{adjustwidth}\\end{table}\n" % (n.replace("_", "\\_"), n)
    corpo += "\\end{document}\n"
    p = SCRATCH / "teste_tabelas_v4.tex"
    p.write_text(corpo, encoding="utf-8")
    for n in nomes_tex:
        subprocess.run(["cp", str(OUT / f"{n}.tex"), str(SCRATCH / f"{n}.tex")], check=True)
    r = subprocess.run([TEXLIVE, "-interaction=nonstopmode", "teste_tabelas_v4.tex"], cwd=SCRATCH, capture_output=True)
    r = subprocess.run([TEXLIVE, "-interaction=nonstopmode", "teste_tabelas_v4.tex"], cwd=SCRATCH, capture_output=True)
    log = (SCRATCH / "teste_tabelas_v4.log").read_text(errors="ignore")
    erros = [l for l in log.split("\n") if l.startswith("!")]
    over = [l for l in log.split("\n") if "Overfull" in l]
    return erros, over, (SCRATCH / "teste_tabelas_v4.pdf").exists()


# ================================================================== main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    og = importar("og_tab", OG_TAB, SHA_OG_TAB)
    og9 = importar("og_t9", OG_T9, SHA_OG_T9)
    reg(OG_TAB); reg(OG_T9)
    tabs, extra = [], {}
    t, ck = tab_v1(); tabs.append(t); extra["T_V1_cobertura"] = ck
    t, ck = tab_v1("fspl_b__validos", "T_V1_cobertura_fspl", FSPL_POR_SORTEIO); tabs.append(t); extra["T_V1_cobertura_fspl"] = ck
    t, _ = tab_v3(); tabs.append(t)
    t, _ = tab_v2bis(); tabs.append(t)
    t, n_ast = tab_v4(); tabs.append(t); extra["T_V4_n_asteriscos"] = n_ast
    t, _ = tab_e6(); tabs.append(t)
    t, rz = tab8(og9); tabs.append(t); extra["Tab8_faixa_razao_0_132"] = rz
    tabs.append(tab4(og))
    tabs.append(tab3(og))
    tabs.append(tab_t6(og))
    tabs.append(tab_t2(og))
    t7, s2t, med2 = tab7_s2(og); tabs += [t7, s2t]; extra["S2_mediana_fspl_segunda_calibracao"] = {k: v[2] for k, v in med2.items()}
    t, faixas = tab_s3(og); tabs.append(t); extra["S3_faixas_razao_0_132_gnn_min_max_mlp_min_max"] = faixas
    larg = medir_larguras(tabs)
    escritos = {}
    for t in tabs:
        w = max(larg[t.nome], 0) - 1.0
        (OUT / f"{t.nome}.tex").write_text(t.tex(w), encoding="utf-8")
        escritos[t.nome] = dict(largura_natural_pt=larg[t.nome], nota_p_pt=w, layout=CFGS[t.cfg][2], cabe_em_520pt=t.cabe)
    (OUT / "TN_notacao_v4.tex").write_text(tab_notacao(), encoding="utf-8")
    ta1 = s1(og)
    for nome, txt in ta1.items():
        (OUT / f"{nome}.tex").write_text(txt + ("\n" if not txt.endswith("\n") else ""), encoding="utf-8")
    # stand-alone supplementary wrappers (article class, no resizebox): title + caption (text of the current Tables S, names GNN/MLP)
    f = extra["S3_faixas_razao_0_132_gnn_min_max_mlp_min_max"]
    legendas = {
        "Table_S1_v4": "GNN and MLP, 60 runs each at $g = 5$~km; the runs at $g = 10$~km use the same settings, with only the block side changed. ``same'' marks a value identical in the two models. "
                       "Architecture and loss (first half; the optimisation, selection, baselines and cost follow on the next page).",
        "Table_S2_v4": "Draws: split draws at $g = 10$~km, $b = 2$~km with at least one valid test node, by cell. " + FSPL_POR_SORTEIO + ": free-space loss whose offset is recomputed on the training partition of each draw, "
                       "as the training median of received power plus modelled path loss, over all training nodes (offset: all training nodes) or over the valid training nodes only (offset: valid training nodes). "
                       "For each cell, the table gives the number of draws in which FSPL, recalibrated per draw, errs less than the constant predictor on the valid test nodes (first group of columns, one per offset) and the number of draws in which the constant predictor errs less than FSPL, recalibrated per draw, over all test nodes (second group, one per offset); the last two rows total the sixteen cells, the second one with no buffer ($b = 0$). "
                       "The last group gives the median over draws of the valid-node mean absolute error, in dB, of the constant predictor and of FSPL, recalibrated per draw, with each of the two offsets.",
        "Table_S3_v4": "A trained model's valid-node MAE drifts with the split draw. Four Bauru cells, five split draws, $g = 5$~km, $b = 2$~km, training seed 42, training budget of eight epochs. "
                       "Valid-node mean absolute error of received power, in dB, mean over draws (between-draw standard deviation) for the GNN and the MLP; bottom row: standard deviation of the per-cell means. "
                       f"Over the five draws, the between-draw standard deviation of the valid-node MAE is ${f[0]:.1f}$ to ${f[1]:.1f}$ times (GNN) and ${f[2]:.1f}$ to ${f[3]:.1f}$ times (MLP) the same-seed repeat difference "
                       "(0.132~dB, the largest difference between two training repeats with the same seed and partition, measured at 5~km); the ratios to the standard deviation across training seeds are in the last columns, "
                       "available in Bauru Q1 and Bauru Q3 only.",
    }
    suplementar = {
        "Table_S1_v4": ("Table S1. Training settings of the two models", ["TA1a_v4"], True),
        "Table_S2_v4": ("Table S2. Draws in which each predictor errs less, FSPL (recalibrated per draw) or constant, by cell", ["T4_S2_v4"], False),
        "Table_S3_v4": ("Table S3. Trained models at $g = 5$~km in the four Bauru cells (five draws)", ["T7b_deriva_modelo_A4_v4"], False),
    }
    for nome, (titulo, partes, dupla) in suplementar.items():
        L = ["% v4-1: envolucro autonomo da tabela suplementar (titulo + legenda; texto das Tabelas S atuais com GNN/MLP). Compiled with pdflatex, no resizebox.",
             "% predictor names: " + PRED_LINHA[nome],
             "\\documentclass[10pt,a4paper]{article}", ("\\usepackage[a4paper,landscape,margin=1.5cm]{geometry}" if not dupla else "\\usepackage[a4paper,margin=1.5cm]{geometry}"), "\\usepackage[T1]{fontenc}", "\\usepackage[utf8]{inputenc}",
             "\\usepackage{amsmath,amssymb}", "\\usepackage{booktabs,array,graphicx}", "\\usepackage{microtype}", "\\pagestyle{empty}", "\\setlength{\\parindent}{0pt}",
             "\\renewcommand{\\arraystretch}{1.12}", "\\newcommand{\\celula}[1]{\\begin{tabular}[c]{@{}c@{}}#1\\end{tabular}}",
             "\\begin{document}", "", "{\\large\\bfseries " + titulo + "}", "", "\\medskip", "\\small", legendas[nome], "", "\\medskip", "\\centering"]
        if dupla:
            L += ["\\input{TA1a_v4.tex}", "", "\\clearpage", "{\\large\\bfseries Table S1 (continued)}", "", "\\medskip", "\\small", "\\centering", "\\input{TA1b_v4.tex}"]
        else:
            L += ["\\input{%s.tex}" % partes[0]]
        L += ["\\end{document}"]
        (OUT / f"{nome}.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(dict(escritos=escritos, extra=extra, fontes_sha256=SHAS), open(SCRATCH / "geracao.json", "w"), indent=1, ensure_ascii=False, default=str)
    print("tabelas:", sorted(escritos))
    print(json.dumps(extra, ensure_ascii=False, indent=1, default=str)[:2500])


if __name__ == "__main__":
    main()
