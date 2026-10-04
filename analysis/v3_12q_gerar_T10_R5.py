#!/usr/bin/env python3
"""Build the table of coverage of the one-partition variance estimate (check R5), sixteen cells.

For each of the 16 city x quadrant cells and for the valid test nodes, the table reports the
number of split draws (of 60) in which the one-partition variance estimate v is defined, and, for
the constant predictor and the second-calibration FSPL predictor, the ratio Q (mean of v over
draws divided by the between-draw variance of the error) and the coverage of Err +/- 2 sqrt(v).
Block size g = 10 km, buffer b = 2 km; N = number of occupied blocks; primary reading (A).

Input used to build the table (and nothing else): fase5/R5_resumo.json, path
resumo.<predictor>__validos.N_ocupados.leitura_A.por_celula.<cell>.
Outputs: T10_cobertura_R5_v3-12.csv and .tex in tables_v3-12/ (a table[H] environment with caption
and label), plus the T10 entries in MANIFEST_tabelas_v3-12.json and CONFERENCIAS_tabelas_v3-12.json.

Consistency checks (not used to build the table): every value re-read by its JSON path and compared
with the CSV and the TEX; n_v_definido == 60 - n_indefinidos_k_lt_2 in all 32 rows; Q, coverage and
n_v_definido recomputed in pure Python from fase5/R5_por_sorteio.json; aggregates (median, min and
max of Q, number of cells with coverage >= 0.80) compared with the record of the independent
recalculation stored in fase5/votos_R5/.

Run order: v3_12_gerar_tabelas.py (rewrites the manifest and the checks file in full), then
v3_12h_gerar_T9_G1.py, then this script, which only adds its own T10 entries and is idempotent.
Usage: python analysis/v3_12q_gerar_T10_R5.py   (CPU only, deterministic)
"""
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

B = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
OUT = B / "redacao_v3-12" / "tables_v3-12"
F5 = B / "fase5"
ENT = {
    "R5_resumo": F5 / "R5_resumo.json",
    "R5_por_sorteio": F5 / "R5_por_sorteio.json",
    "R5_criterio": B / "criterios" / "criterio_R5_variancia_um_sorteio.json",
    "R5_veredito": F5 / "votos_R5" / "veredito.json",
}
NOME_T = "T10_cobertura_R5_v3-12"
LABEL = "tab:coverage_one_partition"
CIDADES = [("bauru", "Bauru"), ("campinas", "Campinas"), ("lins", "Lins"), ("sorocaba", "Sorocaba")]
CELULAS = [(f"{c}_Q{q}", f"{r} Q{q}") for c, r in CIDADES for q in (1, 2, 3, 4)]
PREDS = [("constante__validos", "Constant predictor"), ("fspl_b__validos", "FSPL (second calibration)")]
CAM = "resumo.{key}.N_ocupados.leitura_A.por_celula.{cel}.{campo}"

LEGENDA = (
    "Coverage of the one-partition variance estimate in the sixteen cells, for the valid test nodes. "
    "$v$ is the plug-in cluster-variance estimate from one partition (finite-population factor; test blocks with a valid node), "
    "$Q$ is the ratio of its mean over draws to the between-draw variance of the same error, and coverage is the fraction of the draws "
    "with $v$ defined in which $\\mathrm{Err} \\pm 2\\sqrt{v}$ contains the mean of $\\mathrm{Err}$ over the draws with $\\mathrm{Err}$ defined; "
    "$g = 10$~km, $b = 2$~km."
)
HEAD_CSV = ["celula", "n_sorteios_v_definido_de_60", "constante_validos_Q", "constante_validos_cobertura",
            "fspl_b_validos_Q", "fspl_b_validos_cobertura"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def caminho(J, cam):
    d = J
    for k in cam.split("."):
        d = d[k]
    return d


def construir(R):
    """Table rows, read only from R5_resumo.json by named path."""
    linhas = []
    for cel, rot in CELULAS:
        g = lambda key, campo: caminho(R, CAM.format(key=key, cel=cel, campo=campo))
        linhas.append({"celula": cel, "rotulo": rot, "n": g(PREDS[0][0], "n_v_definido"),
                       "n_fspl": g(PREDS[1][0], "n_v_definido"),
                       "Q_c": g(PREDS[0][0], "Q"), "cob_c": g(PREDS[0][0], "cobertura"),
                       "Q_f": g(PREDS[1][0], "Q"), "cob_f": g(PREDS[1][0], "cobertura")})
    return linhas


def f2(x):
    return f"{x:.2f}"


def _quebra(txt, larg=11):
    """Wrap a header longer than `larg` visible characters into a multi-line \\celula{...} cell."""
    if len(re.sub(r"\\[a-zA-Z]+|[{}$]", "", txt)) <= larg:
        return txt
    toks = re.findall(r"\$[^$]*\$|\S+", txt)
    linhas, cur = [], ""
    for tk in toks:
        vis = len(re.sub(r"\\[a-zA-Z]+|[{}$]", "", cur + " " + tk))
        if cur and vis > larg:
            linhas.append(cur)
            cur = tk
        else:
            cur = (cur + " " + tk).strip()
    linhas.append(cur)
    return "\\celula{" + " \\\\ \\relax ".join(linhas) + "}"


def escrever(linhas):
    csvp, texp = OUT / f"{NOME_T}.csv", OUT / f"{NOME_T}.tex"
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEAD_CSV)
        for r in linhas:
            w.writerow([r["celula"], r["n"], r["Q_c"], r["cob_c"], r["Q_f"], r["cob_f"]])
    with open(texp, "w", encoding="utf-8") as f:
        f.write("% gerado por _pipeline/v3_12q_gerar_T10_R5.py -- nao editar a mao\n")
        f.write("% T10: R5, g = 10 km, b = 2 km; 16 celulas; populacao 'validos', N = blocos ocupados, leitura A (primarias); fonte "
                "fase5/R5_resumo.json (resumo.<preditor>__validos.N_ocupados.leitura_A.por_celula.<celula>: n_v_definido, Q, cobertura); "
                "ambiente table[H] com legenda e label (o \\tabcorpo do manuscrito os neutraliza)\n")
        f.write("\\begin{table}[H]\n\\caption{" + LEGENDA + "}\n\\label{" + LABEL + "}\n\\centering\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write(" & & \\multicolumn{2}{c}{" + PREDS[0][1] + "} & \\multicolumn{2}{c}{" + PREDS[1][1] + "} \\\\\n")
        f.write("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n")
        f.write(" & ".join([_quebra("Cell"), _quebra("Draws with $v$ defined (of 60)"), "$Q$", "Coverage", "$Q$", "Coverage"]) + " \\\\\n\\midrule\n")
        for i, r in enumerate(linhas):
            if i and i % 4 == 0:
                f.write("\\midrule\n")
            f.write(" & ".join([r["rotulo"], str(r["n"]), f2(r["Q_c"]), f2(r["cob_c"]), f2(r["Q_f"]), f2(r["cob_f"])]) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return [texp, csvp]


def recalcular(P, cel, key, N):
    """Independent recalculation of Q and coverage from the per-draw block sums.

    v = (1 - k/N) k/(k-1) sum_B (S_B - Err M_B)^2 / M^2, with Err = sum_B S_B / M and k the number of
    test blocks with a scored node. Mean and variance of Err use the draws with k >= 1; coverage uses
    the draws with v defined (k >= 2).
    """
    errs, vs = [], []  # vs holds (v, Err) pairs
    for s in P["celulas"][cel]["sorteios"]:
        d = s[key]
        S, M = list(d["S_B"]), list(d["M_B"])
        k = len(M)
        if k == 0:
            continue
        Mt = float(sum(M))
        E = sum(S) / Mt
        errs.append(E)
        if k >= 2:
            ss = sum((si - E * mi) ** 2 for si, mi in zip(S, M))
            vs.append(((1 - k / N) * k / (k - 1) * ss / Mt ** 2, E))
    m = sum(errs) / len(errs)
    var = statistics.variance(errs)
    mv = sum(v for v, _ in vs) / len(vs)
    cob = sum(abs(e - m) <= 2 * math.sqrt(v) for v, e in vs) / len(vs)
    n_sort = len(P["celulas"][cel]["sorteios"])
    return {"Q": mv / var, "cobertura": cob, "n_v_definido": len(vs), "n_Err_definido": len(errs), "n_sorteios": n_sort}


def conferir(linhas):
    """Cross-check the written CSV/TEX against the JSON records; return (checks dict, passed)."""
    R = load(ENT["R5_resumo"])
    P = load(ENT["R5_por_sorteio"])
    V = load(ENT["R5_veredito"])
    csv_lido = list(csv.DictReader(open(OUT / f"{NOME_T}.csv", newline="", encoding="utf-8")))
    tex = (OUT / f"{NOME_T}.tex").read_text(encoding="utf-8")
    corpo = [ln for ln in tex.split("\n") if ln.endswith("\\\\") and re.match(r"^(Bauru|Campinas|Lins|Sorocaba) Q[1-4] &", ln)]
    assert len(corpo) == 16 and len(csv_lido) == 16, (len(corpo), len(csv_lido))
    falhas, ncheck, det, ident = [], 0, [], []

    def ok(chave, a, b, tol=0.0):
        nonlocal ncheck
        ncheck += 1
        if not abs(float(a) - float(b)) <= tol:
            falhas.append((chave, a, b))

    for i, (cel, rot) in enumerate(CELULAS):
        rc, ln = csv_lido[i], [x.strip() for x in corpo[i].rstrip("\\ ").split("&")]
        assert rc["celula"] == cel and ln[0] == rot
        N = P["celulas"][cel]["N_blocos_ocupados"]
        ok(f"{cel}.N_ocupados == resumo.meta_celulas", N, caminho(R, f"meta_celulas.{cel}.N_blocos_ocupados"))
        vals = {}
        for key, _ in PREDS:
            g = lambda campo: caminho(R, CAM.format(key=key, cel=cel, campo=campo))
            n_v, n_ind, n_sort = g("n_v_definido"), g("n_indefinidos_k_lt_2"), g("n_sorteios")
            ok(f"{cel}.{key}.n_sorteios == 60", n_sort, 60)
            ok(f"{cel}.{key}.n_v_definido == 60 - n_indefinidos_k_lt_2", n_v, 60 - n_ind)
            ok(f"{cel}.{key}.n_indefinidos == n_sem_no_pontuado_k0 + n_k_igual_1", n_ind, g("n_sem_no_pontuado_k0") + g("n_k_igual_1"))
            ok(f"{cel}.{key}.n_Err_definido == 60 - n_sem_no_pontuado_k0", g("n_Err_definido"), 60 - g("n_sem_no_pontuado_k0"))
            rec = recalcular(P, cel, key, N)
            ok(f"{cel}.{key}.n_sorteios(por_sorteio)", rec["n_sorteios"], 60)
            ok(f"{cel}.{key}.n_v_definido recalculado", n_v, rec["n_v_definido"])
            ok(f"{cel}.{key}.n_Err_definido recalculado", g("n_Err_definido"), rec["n_Err_definido"])
            ok(f"{cel}.{key}.Q recalculado", g("Q"), rec["Q"], 1e-9)
            ok(f"{cel}.{key}.cobertura recalculada", g("cobertura"), rec["cobertura"], 1e-12)
            ok(f"{cel}.{key}.cobertura == n_cobertos / n_v_definido", g("cobertura"), g("n_cobertos") / n_v, 1e-12)
            vals[key] = {"n": n_v, "Q": g("Q"), "cob": g("cobertura"), "Q_recalc": rec["Q"], "cob_recalc": rec["cobertura"],
                         "n_Err_definido": g("n_Err_definido")}
        ok(f"{cel}.n_v_definido constante == fspl", vals[PREDS[0][0]]["n"], vals[PREDS[1][0]]["n"])
        c, f_ = vals[PREDS[0][0]], vals[PREDS[1][0]]
        # CSV must equal the JSON exactly; TEX must equal the JSON rounded to two decimals
        ok(f"{cel}.csv.n", rc["n_sorteios_v_definido_de_60"], c["n"])
        ok(f"{cel}.csv.Q_c", rc["constante_validos_Q"], c["Q"])
        ok(f"{cel}.csv.cob_c", rc["constante_validos_cobertura"], c["cob"])
        ok(f"{cel}.csv.Q_f", rc["fspl_b_validos_Q"], f_["Q"])
        ok(f"{cel}.csv.cob_f", rc["fspl_b_validos_cobertura"], f_["cob"])
        ok(f"{cel}.tex.n", ln[1], c["n"])
        ok(f"{cel}.tex.Q_c", ln[2], f"{c['Q']:.2f}")
        ok(f"{cel}.tex.cob_c", ln[3], f"{c['cob']:.2f}")
        ok(f"{cel}.tex.Q_f", ln[4], f"{f_['Q']:.2f}")
        ok(f"{cel}.tex.cob_f", ln[5], f"{f_['cob']:.2f}")
        det.append({"celula": cel, "n_v_definido": c["n"], "constante": {k: c[k] for k in ("Q", "cob", "Q_recalc", "cob_recalc", "n_Err_definido")},
                    "fspl_b": {k: f_[k] for k in ("Q", "cob", "Q_recalc", "cob_recalc", "n_Err_definido")}})
    agg = {}
    for key, _ in PREDS:
        Qs = [d["constante" if key.startswith("const") else "fspl_b"]["Q"] for d in det]
        cs = [d["constante" if key.startswith("const") else "fspl_b"]["cob"] for d in det]
        v = V["recalculo_16_celulas"][key]
        ok(f"{key}.Q_mediano vs veredito", statistics.median(Qs), v["Q_mediano"], 1e-12)
        ok(f"{key}.Q_min vs veredito", min(Qs), v["Q_min"], 1e-12)
        ok(f"{key}.Q_max vs veredito", max(Qs), v["Q_max"], 1e-12)
        ok(f"{key}.n_cob_ge80 vs veredito", sum(c >= 0.80 for c in cs), v["n_cob_ge80"])
        ok(f"{key}.n_cob_lt60 vs veredito", sum(c < 0.60 for c in cs), v["n_cob_lt60"])
        ok(f"{key}.count of Q within [0.5, 2] versus independent recalculation", sum(0.5 <= q <= 2 for q in Qs), v["n_Q_faixa"])
        agg[key] = {"Q_mediano": statistics.median(Qs), "Q_min": min(Qs), "Q_max": max(Qs), "n_cob_ge_0.80": sum(c >= 0.80 for c in cs),
                    "n_cob_lt_0.60": sum(c < 0.60 for c in cs), "cobertura_min": min(cs), "cobertura_max": max(cs)}
    ns = [d["n_v_definido"] for d in det]
    nerr = sorted({d[k]["n_Err_definido"] for d in det for k in ("constante", "fspl_b")})
    conf = {
        "T10_conferencia_por_caminho_json": {
            "fontes": {k: str(p) for k, p in ENT.items()}, "verificacoes": ncheck, "falhas": falhas, "passou": not falhas,
            "linhas": det},
        "T10_n_v_definido_igual_60_menos_indefinidos": {
            "regra": "n_v_definido == 60 - n_indefinidos_k_lt_2, conferida nas 32 linhas (16 celulas x constante__validos e fspl_b__validos); "
                     "n_v_definido igual nos dois preditores em cada celula (mesmos blocos de teste com no valido)",
            "passou": not any(f[0].endswith("n_v_definido == 60 - n_indefinidos_k_lt_2") or "n_v_definido constante == fspl" in f[0] for f in falhas),
            "faixa_n_v_definido": [min(ns), max(ns)]},
        "T10_agregados_vs_veredito_independente": {"agregados": agg, "veredito": V.get("veredito")},
        "T10_nota_Err_definido": {
            "nota": "A media e a variancia de Err (denominador de Q e referencia da cobertura) sao a leitura A: sobre os sorteios com Err definido "
                    "(pelo menos um no valido de teste), nao sobre todos os 60; a legenda diz 'over the sixty draws'",
            "n_Err_definido_valores_distintos": nerr},
        "T10_campos_ausentes_no_json": {
            "rotulo 'blocos com no valido' como nome de campo": "inexistente; usado n_v_definido (k >= 2 blocos de teste com no pontuado, criterio R5)",
            "cobertura e Q por preditor FSPL (b) com nome 'FSPL (b)'": "o JSON chama fspl_b; a tabela imprime 'FSPL (second calibration)', nome do manuscrito (T8)"},
    }
    return conf, not falhas


def atualizar_registros(saidas, conf):
    cpath = OUT / "CONFERENCIAS_tabelas_v3-12.json"
    C = load(cpath)
    for k in [k for k in C if k.startswith("T10_")]:
        del C[k]  # idempotent: removes only this script's own keys
    C.update(conf)
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(C, f, indent=1, ensure_ascii=False)
    mpath = OUT / "MANIFEST_tabelas_v3-12.json"
    M = load(mpath)
    for k, p in ENT.items():
        M["entradas"][k] = {"arquivo": str(p), "sha256": sha(p)}
    for p in saidas + [cpath]:
        M["saidas"][str(p.relative_to(B))] = sha(p)
    M["T10"] = {
        "tarefa": "T10_cobertura_R5_v3-12 (csv + tex): Q e cobertura do estimador de variancia de uma particao, 16 celulas, constante e FSPL (b), validos",
        "protocolo": str(B / "redacao_v3-12" / "internal/note_32.md") + " (list of planned additional checks, item 3) and "
                     + str(B / "_propostas_2026-10-01" / "internal/note_33.md") + " (item 3)",
        "script": {"arquivo": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "entradas": sorted(ENT),
        "saidas": [str(p.relative_to(B)) for p in saidas],
        "conferencias": [k for k in C if k.startswith("T10_")],
        "label": LABEL,
        "comando": "/trabalho/ambientes/s33_amb_virtual/.venv/bin/python redacao_v3-12/tables_v3-12/_pipeline/v3_12q_gerar_T10_R5.py",
        "ordem": "scripts/v3_12_gerar_tabelas.py -> _pipeline/v3_12h_gerar_T9_G1.py -> este script",
        "nota": "o gerador principal regera MANIFEST e CONFERENCIAS por inteiro e apaga estas entradas; reexecutar este script depois dele",
    }
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(M, f, indent=1, ensure_ascii=False)


def main():
    R = load(ENT["R5_resumo"])
    linhas = construir(R)
    saidas = escrever(linhas)
    conf, passou = conferir(linhas)
    atualizar_registros(saidas, conf)
    for p in saidas:
        print("gravado", p)
    c = conf["T10_conferencia_por_caminho_json"]
    print("conferencias T10:", c["verificacoes"], "verificacoes; falhas:", c["falhas"])
    return 0 if passou else 1


if __name__ == "__main__":
    sys.exit(main())
