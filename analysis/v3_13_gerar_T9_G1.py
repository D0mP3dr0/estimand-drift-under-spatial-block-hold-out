#!/usr/bin/env python3
"""Build the drift table of the trained models at the main geometry (g = 10 km blocks, b = 2 km buffer).

The table (LaTeX label tab:drift_models_g10) has eight rows: four cells (Bauru Q1, Campinas Q1,
Bauru Q3, Campinas Q3) x two models (GNN, MLP), training seed 42. For each cell and model it gives the
number of split draws used (of 20), the excluded draws (draws without a valid test node + draws marked
not trainable in the aggregate, printed as "a+b" or "0"), the mean and the standard deviation (SD, ddof = 1) of the
valid-node MAE across draws, the pooled SD across training seeds (Q1 cells only), the ratio of the SD
to 0.132 dB (largest difference between two same-seed training repeats), the Spearman correlation with
the constant-predictor valid-node MAE across draws, and the sentinel parity (median |GNN - MLP| at the
sentinel nodes, one value per cell, printed on the GNN row; the MLP row reads "same").

Inputs (aggregates of the G1 campaign, under gpu/G1/): agregado_G1_v5_bloco1.json (Q1 cells, 20 draws),
agregado_G1_v5_bloco2.json (crossed seeds 42/43/44 x 5 draws; pooled SD for Bauru Q1),
agregado_G1_v10_bloco4.json (crossed seeds x 10 draws; pooled SD for Campinas Q1) and
agregado_G1_v8_bloco3.json (Q3 cells); the independent recalculation file of each block in
gpu/G1_votos_bloco<k>/ (hashed and its verdict field copied into the checks); and the manuscript source
(path given by --manuscrito; not distributed), whose caption for this label must equal LEGENDA
character for character.
Outputs: T9_deriva_modelo_G1_v3-12.csv and .tex in tables_v3-12/; T9_v3-12t_valores.json next to this
script (per-draw values, mean, SD and source fields with SHA-256); the T9_* entries of
CONFERENCIAS_tabelas_v3-12.json and MANIFEST_tabelas_v3-12.json. The main table generator rewrites those
two registry files in full, so run this script after it; re-running is idempotent.
No value is typed by hand: every number comes from named JSON fields, and the checks re-read each value
by its JSON path, recompute SD, ratio, counts, mean, pooled SD and parity median from the per-draw values,
and read back the written CSV and TEX. Deterministic (no random numbers). Exit code 0 if all checks pass,
1 if any fails, 3 if the recalculated SDs do not match the expected printed values.

Usage: python analysis/v3_13_gerar_T9_G1.py [--manuscrito <main.tex>]
"""
import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

# Table caption; conferir() checks that it equals the caption of this label in the manuscript source,
# which is chosen with --manuscrito.
LEGENDA = r"""The standard deviation (SD) across draws of the valid-node MAE is $2.98$ to $10.4$ times the pooled SD across training seeds in Bauru Q1 and Campinas Q1, the two cells with crossed seeds; in the four trained-model cells it is $8.6$ to $30.9$ times the same-seed repeat difference ($0.132$~dB, Section~\ref{sec:dependence}); the models agree at the sentinel nodes. Main geometry ($g = 10$~km, $b = 2$~km), training budget of eight epochs, training seed $42$; pooled SD across seeds from three seeds $\times$ five draws in Bauru Q1 and three seeds $\times$ ten draws in Campinas Q1. Draws excluded: draws without a valid test node plus draws whose validation partition has no antenna--terrain edge. ``Same'' repeats the sentinel parity of the GNN row, one value per cell; --: not run, since the crossed seed block covers only Bauru Q1 and Campinas Q1."""
MAIN_PADRAO = "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/redacao_v3-12/main_v3-13d.tex"
MAIN_LEGENDA = Path(MAIN_PADRAO)

AQUI = Path(__file__).resolve().parent
B = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
OUT = B / "redacao_v3-12" / "tables_v3-12"
G1 = B / "gpu" / "G1"
ENT = {
    "G1_bloco1": G1 / "agregado_G1_v5_bloco1.json",
    "G1_bloco2": G1 / "agregado_G1_v5_bloco2.json",
    "G1_bloco4": G1 / "agregado_G1_v10_bloco4.json",
    "G1_bloco3": G1 / "agregado_G1_v8_bloco3.json",
}
VEREDITOS = {
    "G1_veredito_bloco1": B / "gpu" / "G1_votos_bloco1" / "veredito.json",
    "G1_veredito_bloco2": B / "gpu" / "G1_votos_bloco2" / "veredito.json",
    "G1_veredito_bloco4": B / "gpu" / "G1_votos_bloco4" / "veredito.json",
    "G1_veredito_bloco3": B / "gpu" / "G1_votos_bloco3" / "veredito.json",
}
NOME_T = "T9_deriva_modelo_G1_v3-12"
LABEL = "tab:drift_models_g10"
PISO = 0.132  # dB, same-seed repeat difference; used only to check the ratio read from the JSON
# (cell key, printed label, aggregate holding the cell's 20-draw results)
CELULAS = [("bauru_Q1", "Bauru Q1", "G1_bloco1"), ("campinas_Q1", "Campinas Q1", "G1_bloco1"),
           ("bauru_Q3", "Bauru Q3", "G1_bloco3"), ("campinas_Q3", "Campinas Q3", "G1_bloco3")]
MODELOS = [("gnn", "GNN"), ("mlp", "MLP")]

HEADER_TEX = ("llcccccccc",
              "Cell & Model & Draws used (of 20) & Draws excluded & Mean of valid-node MAE across draws (dB) & SD of valid-node MAE across draws (dB) & "
              "Pooled SD across seeds (dB) & Ratio of that SD to the same-seed repeat difference ($0.132\,\mathrm{dB}$) & Spearman with constant-predictor valid-node MAE across draws & Sentinel parity: median $|\\mathrm{GNN}-\\mathrm{MLP}|$ (dB)")
HEADER_CSV = ["celula", "modelo", "n_sorteios_usados", "n_sem_validos", "n_nao_treinaveis", "sorteios_excluidos_texto",
              "mae_medio_validos_db", "dp_entre_sorteios_validos_db", "dp_entre_sementes_pooled_db", "razao_dp_sobre_0_132",
              "spearman_com_constante_validos", "paridade_sentinela_mediana_db"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fnum(x, nd):
    """Format x with nd decimals for LaTeX: no negative zero, minus sign as $-$."""
    s = f"{x:.{nd}f}"
    if float(s) == 0.0:
        s = f"{0.0:.{nd}f}"
    return s.replace("-", "$-$")


def _quebra(txt, larg=11):
    """Wrap a header cell into a multi-line \\celula{...} when its visible text exceeds larg characters.

    Visible length ignores LaTeX commands, braces and $; math spans are kept whole. Same helpers
    (_quebra, _cabecalho) as in the main table generator, so headers wrap identically.
    """
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


def _cabecalho(h):
    return " & ".join(_quebra(c.strip()) for c in h.split(" & "))


def construir(J):
    """Return the eight table rows (dicts), read from named fields of the loaded aggregates J."""
    linhas = []
    for cel, rotulo, blk in CELULAS:
        c = J[blk]["celulas"][cel]
        sv, nt = c["n_sorteios_sem_validos"], c.get("n_sorteios_nao_treinaveis", 0)  # field absent in block 1: 0
        excl = f"{sv}+{nt}" if (sv or nt) else "0"
        par = c["paridade_sentinela"]["mediana_db"]
        for mod, mrot in MODELOS:
            m = c["por_modelo"][mod]
            # Pooled SD across seeds only for Q1: Campinas Q1 from block 4 (3 seeds x 10 draws),
            # Bauru Q1 from block 2 (3 seeds x 5 draws); Q3 cells have none (printed "--").
            pooled =(J["G1_bloco4"]["celulas"][cel][mod]["dp_entre_sementes_pooled_db"] if cel == "campinas_Q1" else J["G1_bloco2"]["celulas"][cel][mod]["dp_entre_sementes_pooled_db"]) if cel.endswith("Q1") else None
            linhas.append({
                "celula": cel, "rotulo": rotulo, "modelo": mod, "mrot": mrot, "n": m["n_sorteios_com_validos"],
                "sv": sv, "nt": nt, "excl": excl, "media": statistics.fmean(m["mae_validos_por_sorteio"].values()), "dp": m["dp_entre_sorteios_validos_db"], "razao": m["razao_dp_sobre_0_132"],
                "pooled": pooled, "spearman": m["correlacao_com_constante_validos"]["spearman"],
                "paridade": par if mod == "gnn" else "same"})
    return linhas


def escrever(linhas):
    """Write the CSV (full precision) and the LaTeX table (rounded); return [tex path, csv path]."""
    csvp, texp = OUT / f"{NOME_T}.csv", OUT / f"{NOME_T}.tex"
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER_CSV)
        for r in linhas:
            w.writerow([r["celula"], r["modelo"], r["n"], r["sv"], r["nt"], r["excl"], r["media"], r["dp"],
                        "" if r["pooled"] is None else r["pooled"], r["razao"], r["spearman"], r["paridade"]])
    with open(texp, "w", encoding="utf-8") as f:
        f.write("% gerado por _pipeline/v3_13_gerar_T9_G1.py -- nao editar a mao\n")
        f.write("% T9: batch G1, models trained at g = 10 km, b = 2 km; 4 cells x GNN/MLP; sources gpu/G1/agregado_G1_v5_bloco1.json, "
                "agregado_G1_v5_bloco2.json (only the column of SD across seeds, only Q1), agregado_G1_v8_bloco3.json; "
                "'Draws excluded' = sem_validos+nao_treinaveis (only where present); 'Mean valid-node MAE' = mean of mae_validos_por_sorteio (same list as the SD); 'same' = cell value, printed on the GNN row; "
                "table[H] environment with caption and label (the manuscript's \\tabcorpo wrapper neutralises them)\n")
        f.write("\\begin{table}[H]\n\\caption{" + LEGENDA + "}\n\\label{" + LABEL + "}\n\\centering\n")
        f.write("\\begin{tabular}{" + HEADER_TEX[0] + "}\n\\toprule\n" + _cabecalho(HEADER_TEX[1]) + " \\\\\n\\midrule\n")
        for i, r in enumerate(linhas):
            if i and r["modelo"] == "gnn":
                f.write("\\midrule\n")
            par = r["paridade"] if r["paridade"] == "same" else fnum(r["paridade"], 3)
            f.write(" & ".join([r["rotulo"] if r["modelo"] == "gnn" else "", r["mrot"], str(r["n"]), r["excl"], fnum(r["media"], 2), fnum(r["dp"], 2),
                                "--" if r["pooled"] is None else fnum(r["pooled"], 2), fnum(r["razao"], 1),
                                fnum(r["spearman"], 2), par]) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return [texp, csvp]


def caminho(J, blk, cam):
    """Return the value at the dotted path cam inside aggregate J[blk]."""
    d = J[blk]
    for k in cam.split("."):
        d = d[k]
    return d


def conferir(linhas):
    """Check every printed value against the JSON (re-read by path, independently of construir).

    Compares the written CSV (exact) and TEX (rounded) with the JSON fields and with values recomputed
    from the per-draw lists; returns (records for the registry, True if no check failed).
    """
    J = {k: load(p) for k, p in ENT.items()}
    csv_lido = list(csv.DictReader(open(OUT / f"{NOME_T}.csv", newline="", encoding="utf-8")))
    tex = (OUT / f"{NOME_T}.tex").read_text(encoding="utf-8")
    corpo = [ln for ln in tex.split("\n") if ln.endswith("\\\\") and not ln.startswith("\\celula") and "Cell" not in ln and "toprule" not in ln]
    corpo = [ln for ln in corpo if re.search(r"\b(GNN|MLP)\b", ln.split("&")[1] if "&" in ln else "")]
    assert len(corpo) == 8 and len(csv_lido) == 8, (len(corpo), len(csv_lido))
    conf, falhas = {}, []
    ncheck = 0

    def ok(chave, impresso, relido, tol=0.0):
        nonlocal ncheck
        ncheck += 1
        bom = abs(float(impresso) - float(relido)) <= tol
        if not bom:
            falhas.append((chave, impresso, relido))
        return bom

    detalhe = []
    for i, (cel, rotulo, blk) in enumerate([(c, r, b) for c, r, b in CELULAS for _ in MODELOS]):
        mod = MODELOS[i % 2][0]
        base = f"celulas.{cel}.por_modelo.{mod}"
        rc, ln = csv_lido[i], [x.strip() for x in corpo[i].rstrip("\\ ").split("&")]
        assert rc["celula"] == cel and rc["modelo"] == mod
        dp_json = caminho(J, blk, base + ".dp_entre_sorteios_validos_db")
        mae = list(caminho(J, blk, base + ".mae_validos_por_sorteio").values())
        n_json = caminho(J, blk, base + ".n_sorteios_com_validos")
        razao_json = caminho(J, blk, base + ".razao_dp_sobre_0_132")
        sp_json = caminho(J, blk, base + ".correlacao_com_constante_validos.spearman")
        sv_json = caminho(J, blk, f"celulas.{cel}.n_sorteios_sem_validos")
        try:
            nt_json = caminho(J, blk, f"celulas.{cel}.n_sorteios_nao_treinaveis")
        except KeyError:
            nt_json = 0  # block 1 has no not-trainable field (Q1 cells)
        par_json = caminho(J, blk, f"celulas.{cel}.paridade_sentinela.mediana_db")
        par_rec = statistics.median(caminho(J, blk, f"celulas.{cel}.paridade_sentinela.abs_gnn_menos_mlp_por_sorteio").values())
        # CSV equals JSON exactly; TEX equals JSON rounded to the printed decimals
        ok(f"{cel}.{mod}.csv.dp", rc["dp_entre_sorteios_validos_db"], dp_json)
        ok(f"{cel}.{mod}.csv.razao", rc["razao_dp_sobre_0_132"], razao_json)
        ok(f"{cel}.{mod}.csv.spearman", rc["spearman_com_constante_validos"], sp_json)
        ok(f"{cel}.{mod}.csv.n", rc["n_sorteios_usados"], n_json)
        ok(f"{cel}.{mod}.csv.sem_validos", rc["n_sem_validos"], sv_json)
        ok(f"{cel}.{mod}.csv.nao_treinaveis", rc["n_nao_treinaveis"], nt_json)
        ok(f"{cel}.{mod}.tex.n", ln[2], n_json)
        ok(f"{cel}.{mod}.tex.dp", ln[5].replace("$-$", "-"), f"{dp_json:.2f}")
        ok(f"{cel}.{mod}.tex.razao", ln[7].replace("$-$", "-"), f"{razao_json:.1f}")
        ok(f"{cel}.{mod}.tex.spearman", ln[8].replace("$-$", "-"), f"{sp_json:.2f}")
        esperado_excl = f"{sv_json}+{nt_json}" if (sv_json or nt_json) else "0"
        ncheck += 1
        if ln[3] != esperado_excl or rc["sorteios_excluidos_texto"] != esperado_excl:
            falhas.append((f"{cel}.{mod}.excluidos", ln[3], esperado_excl))
        # Mean column and recomputation from the per-draw values (SD with ddof = 1, ratio = SD / 0.132)
        media_rec = sum(mae) / len(mae)
        ok(f"{cel}.{mod}.csv.media", rc["mae_medio_validos_db"], media_rec, 1e-12)
        ok(f"{cel}.{mod}.tex.media", ln[4].replace("$-$", "-"), f"{media_rec:.2f}")
        ok(f"{cel}.{mod}.tex.n_sorteios_da_media == n", len(mae), n_json)
        ok(f"{cel}.{mod}.n == len(mae_validos_por_sorteio)", n_json, len(mae))
        ok(f"{cel}.{mod}.dp == stdev(ddof=1) dos MAE por sorteio", dp_json, statistics.stdev(mae), 1e-12)
        ok(f"{cel}.{mod}.razao == dp/0.132", razao_json, dp_json / PISO, 1e-9)
        # Draw counts: used = planned - without valid nodes (Q1); = usable (Q3)
        if blk == "G1_bloco3":
            ok(f"{cel}.{mod}.n == n_sorteios_usaveis", n_json, caminho(J, blk, f"celulas.{cel}.n_sorteios_usaveis"))
            ok(f"{cel}.{mod}.usaveis == considerados - sem_validos - nao_treinaveis",
               n_json, caminho(J, blk, f"celulas.{cel}.n_sorteios_plano_considerados") - sv_json - nt_json)
        else:
            ok(f"{cel}.{mod}.n == plano_usados - sem_validos", n_json, caminho(J, blk, f"celulas.{cel}.n_sorteios_plano_usados") - sv_json)
        if mod == "gnn":
            ok(f"{cel}.csv.paridade", rc["paridade_sentinela_mediana_db"], par_json)
            ok(f"{cel}.tex.paridade", ln[9], f"{par_json:.3f}")
            ok(f"{cel}.paridade == mediana(abs_gnn_menos_mlp_por_sorteio)", par_json, par_rec, 1e-15)
        else:
            ncheck += 2
            if ln[9] != "same" or rc["paridade_sentinela_mediana_db"] != "same":
                falhas.append((f"{cel}.mlp.paridade_same", ln[9], rc["paridade_sentinela_mediana_db"]))
        if cel.endswith("Q1"):
            fonte = "G1_bloco4" if cel == "campinas_Q1" else "G1_bloco2"
            pj = caminho(J, fonte, f"celulas.{cel}.{mod}.dp_entre_sementes_pooled_db")
            pj2 = caminho(J, fonte, f"celulas.{cel}.{mod}.decomposicao_um_fator_sementes_aninhadas_no_sorteio.dp_entre_sementes_pooled_db")
            ok(f"{cel}.{mod}.csv.pooled", rc["dp_entre_sementes_pooled_db"], pj)
            ok(f"{cel}.{mod}.tex.pooled", ln[6], f"{pj:.2f}")
            ok(f"{cel}.{mod}.pooled == decomposicao.dp_entre_sementes_pooled_db", pj, pj2)
            # The 20-draw seed-42 SD stored in block 2 must equal the block-1 SD
            ok(f"{cel}.{mod}.dp bloco1 == dp_entre_sorteios_20_semente42_db bloco2", dp_json,
               caminho(J, "G1_bloco2", f"celulas.{cel}.{mod}.dp_entre_sorteios_20_semente42_db"), 1e-12)
            # Pooled SD = sqrt(mean over draws of the variance across seeds), from the draw x seed matrix
            mat =caminho(J, fonte, f"celulas.{cel}.{mod}.matriz_mae_validos_sorteio_x_semente_42_43_44")
            pooled_rec = (sum(statistics.variance(l) for l in mat) / len(mat)) ** 0.5
            ok(f"{cel}.{mod}.pooled == sqrt(media das variancias entre sementes por sorteio)", pj, pooled_rec, 1e-12)
        else:
            ncheck += 2
            if ln[6] != "--" or rc["dp_entre_sementes_pooled_db"] != "":
                falhas.append((f"{cel}.{mod}.pooled_vazio", ln[6], rc["dp_entre_sementes_pooled_db"]))
        detalhe.append({"celula": cel, "modelo": mod, "n": n_json, "excluidos": esperado_excl, "dp_json": dp_json,
                        "dp_recalculado_stdev_ddof1": statistics.stdev(mae), "razao_json": razao_json, "spearman_json": sp_json, "media_recalculada": media_rec})
    leg_main = None
    ls = MAIN_LEGENDA.read_text(encoding="utf-8").split("\n")
    kk = [k for k, l in enumerate(ls) if l.startswith("\\label{" + LABEL + "}")]
    if len(kk) == 1 and ls[kk[0] - 1].startswith("\\caption{") and ls[kk[0] - 1].endswith("}"):
        leg_main = ls[kk[0] - 1][len("\\caption{"):-1]
    cap_gravada = [l for l in tex.split("\n") if l.startswith("\\caption{")]
    ok_leg = leg_main is not None and LEGENDA == leg_main and cap_gravada == ["\\caption{" + leg_main + "}"]
    ncheck += 1
    if not ok_leg:
        falhas.append((f"caption identical to the manuscript {MAIN_LEGENDA.name}", "diferente", "igual"))
    conf["T9_v3-13d_legenda_igual_ao_manuscrito"] = {"arquivo": str(MAIN_LEGENDA), "sha256_main": sha(MAIN_LEGENDA), "passou": ok_leg}
    # Expected printed SDs, same values and order as SD_IMPRESSO_V3_12H
    sd_pedido = ["1.25", "1.13", "3.26", "2.01", "3.86", "4.08", "2.46", "3.08"]
    sd_rec = [f"{statistics.stdev(list(caminho(J, blk, f'celulas.{cel}.por_modelo.{mod}.mae_validos_por_sorteio').values())):.2f}"
              for cel, _, blk in CELULAS for mod, _ in MODELOS]
    ok_sd = sd_rec == sd_pedido
    ncheck += 1
    if not ok_sd:
        falhas.append(("recalculated SD equals the expected printed SD", sd_rec, sd_pedido))
    conf["T9_v3-12t_sd_recalculado_reproduz_impresso"] = {"recalculado": sd_rec, "impresso_v3-12h": sd_pedido, "passou": ok_sd}
    conf["T9_conferencia_por_caminho_json"] = {"fontes": {k: str(p) for k, p in ENT.items()}, "verificacoes": ncheck, "falhas": falhas,
                                               "passou": not falhas, "linhas": detalhe}
    conf["T9_spearman"] = {"nota": "Spearman coefficient re-read from the aggregate by JSON path and checked against the CSV and TEX; not recomputed here "
                                   "(it needs the per-draw constant-predictor MAE, which is stored upstream of the aggregate and in the independent recalculation of each block)",
                           "veredito_independente": {k: load(p).get("veredito") for k, p in VEREDITOS.items()}}
    conf["T9_faixa_razao_0_132"] = {"min": min(r["razao"] for r in linhas), "max": max(r["razao"] for r in linhas),
                                    "impresso": f"{fnum(min(r['razao'] for r in linhas), 1)} a {fnum(max(r['razao'] for r in linhas), 1)}"}
    conf["T9_paridade_faixa_db"] = {"min": min(r["paridade"] for r in linhas if r["paridade"] != "same"),
                                    "max": max(r["paridade"] for r in linhas if r["paridade"] != "same")}
    conf["T9_campos_ausentes_no_json"] = {"n_sorteios_nao_treinaveis em agregado_G1_v5_bloco1.json (bauru_Q1, campinas_Q1)":
                                          "campo inexistente; tratado como 0 (lista sem_validos vazia; n_sorteios_plano_usados = 20 = n_sorteios_com_validos)",
                                          "dp_entre_sementes_pooled_db nas celulas Q3": "inexistente por desenho (bloco 2 so tem Q1); impresso '--' (csv vazio)"}
    return conf, not falhas


def atualizar_registros(saidas, conf):
    """Replace the T9_* check records and add T9 inputs, output hashes and a T9 block to the manifest."""
    cpath = OUT / "CONFERENCIAS_tabelas_v3-12.json"
    C = load(cpath)
    for k in [k for k in C if k.startswith("T9_")]:
        del C[k]  # only this script's own keys, so re-running is idempotent
    C.update(conf)
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(C, f, indent=1, ensure_ascii=False)
    mpath = OUT / "MANIFEST_tabelas_v3-12.json"
    M = load(mpath)
    for k, p in {**ENT, **VEREDITOS}.items():
        M["entradas"][k] = {"arquivo": str(p), "sha256": sha(p)}
    for p in saidas + [cpath]:
        M["saidas"][str(p.relative_to(B))] = sha(p)
    M["T9"] = {
        "tarefa": "T9_deriva_modelo_G1_v3-12 (csv + tex): modelos treinados a 10 km, 4 celulas x GNN/MLP, lote G1",
        "protocolo": str(B / "redacao_v3-12" / "internal/note_31.md") + " (specification of this table: content, sources and label)",
        "script": {"arquivo": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "entradas": sorted(ENT) + sorted(VEREDITOS),
        "saidas": [str(p.relative_to(B)) for p in saidas],
        "conferencias": [k for k in C if k.startswith("T9_")],
        "label": LABEL,
        "comando": "/trabalho/ambientes/s33_amb_virtual/.venv/bin/python redacao_v3-12/tables_v3-12/_pipeline/v3_13_gerar_T9_G1.py",
        "v3-12t": "column Mean valid-node MAE across draws (dB), placed before the SD column; per-draw values in _pipeline/T9_v3-12t_valores.json; caption and other columns unchanged",
        "nota": "o gerador principal scripts/v3_12_gerar_tabelas.py regera MANIFEST e CONFERENCIAS por inteiro e apaga estas entradas; reexecutar este script depois dele",
    }
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(M, f, indent=1, ensure_ascii=False)


# Expected printed SD across draws (dB); order: Bauru Q1 GNN/MLP, Campinas Q1, Bauru Q3, Campinas Q3.
# main() stops before writing anything if the SDs recomputed from the per-draw values differ.
SD_IMPRESSO_V3_12H = ["1.25", "1.13", "3.26", "2.01", "3.86", "4.08", "2.46", "3.08"]


def gravar_valores(J, linhas):
    """Write T9_v3-12t_valores.json: per cell x model, the per-draw valid-node MAE, mean, SD, n and source fields.

    The aggregates hold no mean field; the mean is the arithmetic mean of the same per-draw list
    the SD comes from (training seed 42).
    """
    import datetime
    itens = []
    for r in linhas:
        blk = next(b for c, _, b in CELULAS if c == r["celula"])
        campo = f"celulas.{r['celula']}.por_modelo.{r['modelo']}.mae_validos_por_sorteio"
        d = caminho(J, blk, campo)
        v = list(d.values())
        itens.append({
            "id": r["celula"] + "." + r["modelo"], "celula": r["celula"], "rotulo_celula": r["rotulo"], "modelo": r["modelo"],
            "media_mae_validos_db": statistics.fmean(v), "media_impressa": fnum(r["media"], 2).replace("$-$", "-"),
            "sd_entre_sorteios_db": statistics.stdev(v), "sd_impresso": fnum(r["dp"], 2),
            "sd_do_agregado_db": r["dp"], "n_sorteios": len(v), "n_sorteios_com_validos_do_agregado": r["n"],
            "semente_de_treino": 42, "sorteios_usados": list(d.keys()), "mae_validos_por_sorteio_db": d,
            "origem": {"arquivo": str(ENT[blk]), "sha256": sha(ENT[blk]), "campo_media": campo + " (media aritmetica dos valores; o agregador nao grava campo de media)",
                       "campo_sd": f"celulas.{r['celula']}.por_modelo.{r['modelo']}.dp_entre_sorteios_validos_db",
                       "campo_n": f"celulas.{r['celula']}.por_modelo.{r['modelo']}.n_sorteios_com_validos"}})
    fontes = {k: {"arquivo": str(p), "sha256": sha(p)} for k, p in ENT.items() if k != "G1_bloco2" and k != "G1_bloco4"}
    doc = {"artefato": "T9_v3-12t_valores", "tabela": "T9_deriva_modelo_G1_v3-12 (label tab:drift_models_g10)",
           "script": str(Path(__file__).resolve()), "script_sha256": sha(Path(__file__)),
           "comando": "/trabalho/ambientes/s33_amb_virtual/.venv/bin/python redacao_v3-12/tables_v3-12/_pipeline/v3_13_gerar_T9_G1.py",
           "data": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "fontes": fontes, "nota": "media e SD sobre os mesmos sorteios (mae_validos_por_sorteio, semente 42) da coluna 'SD across draws of valid-node MAE'; "
           "os agregados bloco2/bloco4 (so o SD entre sementes, Q1) nao entram nesta coluna",
           "valores": itens}
    pth = AQUI / "T9_v3-12t_valores.json"
    with open(pth, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    return pth


def main():
    J = {k: load(p) for k, p in ENT.items()}
    for blk in J:
        assert J[blk].get("modo_teste") is False, blk  # refuse aggregates produced in test mode
    linhas = construir(J)
    sd_rec = []
    for r in linhas:
        b = next(b for c, _, b in CELULAS if c == r["celula"])
        v = caminho(J, b, "celulas." + r["celula"] + ".por_modelo." + r["modelo"] + ".mae_validos_por_sorteio").values()
        sd_rec.append(f"{statistics.stdev(list(v)):.2f}")
    if sd_rec != SD_IMPRESSO_V3_12H:
        print("PARE: SD recalculado nao reproduz o impresso:", sd_rec, SD_IMPRESSO_V3_12H)
        return 3
    saidas = escrever(linhas)
    saidas.append(gravar_valores(J, linhas))
    conf, passou = conferir(linhas)
    atualizar_registros(saidas, conf)
    for p in saidas:
        print("gravado", p)
    print("conferencias T9:", conf["T9_conferencia_por_caminho_json"]["verificacoes"], "verificacoes;",
          "falhas:", conf["T9_conferencia_por_caminho_json"]["falhas"])
    return 0 if passou else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Gera a T9 (G1) e confere a legenda contra o manuscrito indicado.")
    ap.add_argument("--manuscrito", default=MAIN_PADRAO, help="main*.tex contra o qual LEGENDA e conferida (padrao: main_v3-13d.tex)")
    MAIN_LEGENDA = Path(ap.parse_args().manuscrito)
    sys.exit(main())
