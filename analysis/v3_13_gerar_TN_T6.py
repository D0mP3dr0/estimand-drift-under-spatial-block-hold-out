#!/usr/bin/env python3
"""Regenerate the notation table (Table 1) and the design-reference table (Table 5) with two label corrections.

The main table generator, analysis/v3_12_gerar_tabelas.py, is imported unchanged. The script runs its
gerar_tn() and gerar_t6() in a temporary folder (they must reproduce the files already stored) and
applies two corrections by counted substitution:
  (1) in the notation table, the word before each \\ref of the five labels prop:retention,
      prop:degree, prop:deflation, rem:designs and rem:refit becomes the environment (Proposition,
      Remark or Lemma) that the label has in the manuscript source, read from the \\begin{...} that
      precedes each \\label; nothing is typed by hand;
  (2) in the design-reference table, the row label "FSPL (second calibration)" becomes
      "FSPL (calibrated once)"; only the label changes, no value.
It writes TN_notacao_v3-12.tex and .csv and T6_referencia_desenho_v3-12.tex, scans every generated table
for \\ref to the five labels, and records the findings in TN_T6_v3-13_proveniencia.json. It also updates
the SHA-256 entries of the table manifest and adds the keys TN_v3-13_ambientes_dos_ref and
T6_v3-13_rotulo_FSPL to CONFERENCIAS_tabelas_v3-12.json.

Input that is not distributed: the manuscript source (MAIN). Run it after v3_12_gerar_tabelas.py, which
rewrites both tables without the corrections.
Usage: python analysis/v3_13_gerar_TN_T6.py   (CPU only, deterministic)
"""
import datetime, hashlib, importlib.util, json, re, sys, tempfile
from pathlib import Path

M = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
B = M / "_v3_2026-09-25"
R = B / "redacao_v3-12"
RAIZ = R / "tables_v3-12"
MAIN = R / "main_v3-13a.tex"
ANTIGO = M / "scripts" / "v3_12_gerar_tabelas.py"
ROTULOS = ["prop:retention", "prop:degree", "prop:deflation", "rem:designs", "rem:refit"]
T6_VELHO, T6_NOVO = "FSPL (second calibration)", "FSPL (calibrated once)"
TN, T6 = "TN_notacao_v3-12", "T6_referencia_desenho_v3-12"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ambientes():
    """Environment (Proposition, Remark, Lemma, ...) that precedes each of the five labels in the manuscript source."""
    L =MAIN.read_text(encoding="utf-8").split("\n")
    env = {}
    for lab in ROTULOS:
        ach = [i for i, l in enumerate(L) if "\\label{" + lab + "}" in l]
        assert len(ach) == 1, lab
        for j in range(ach[0], max(ach[0] - 8, 0), -1):
            m = re.search(r"\\begin\{(Proposition|Remark|Lemma|Theorem|Corollary|Definition)\}", L[j])
            if m:
                env[lab] = (m.group(1), j + 1)
                break
        assert lab in env, lab
    return env


def varre(env):
    """List every reference to the five labels in the generated tables: file, line, preceding word."""
    achados = []
    pad = re.compile(r"(\w+)?~?\\(?:eq|auto)?ref\{(" + "|".join(re.escape(x) for x in ROTULOS) + r")\}")
    arquivos = sorted(RAIZ.glob("*.tex")) + sorted(RAIZ.glob("*.csv")) + sorted((R / "suplementar").glob("*.tex"))
    for p in arquivos:
        for n, ln in enumerate(p.read_text(encoding="utf-8").split("\n"), 1):
            for m in pad.finditer(ln):
                achados.append({"arquivo": str(p.relative_to(R)), "linha": n, "palavra_antes": m.group(1), "rotulo": m.group(2), "ambiente_no_manuscrito": env[m.group(2)][0]})
    return achados


def main():
    env = ambientes()
    antes = varre(env)
    spec = importlib.util.spec_from_file_location("v3_12_gerar_tabelas_antigo", ANTIGO)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    atual = {k: (RAIZ / k).read_bytes() for k in (TN + ".tex", TN + ".csv", T6 + ".tex", T6 + ".csv")}
    with tempfile.TemporaryDirectory() as tmp:
        g.OUT = Path(tmp)
        g.gerar_tn()
        g.gerar_t6()
        novo = {k: (Path(tmp) / k).read_bytes() for k in atual}
    saida = {}
    pad = re.compile(r"(Proposition|Remark|Lemma)~\\ref\{(" + "|".join(re.escape(x) for x in ROTULOS) + r")\}")
    trocas = []
    for k in (TN + ".tex", TN + ".csv"):
        txt = novo[k].decode("utf-8")
        def troca(m):
            alvo = env[m.group(2)][0]
            if alvo != m.group(1):
                trocas.append({"arquivo": k, "rotulo": m.group(2), "de": m.group(1), "para": alvo})
            return f"{alvo}~\\ref{{{m.group(2)}}}"
        saida[k] = pad.sub(troca, txt).encode("utf-8")
    t6 = novo[T6 + ".tex"].decode("utf-8")
    n6 = t6.count(T6_VELHO)
    assert n6 == 4, n6
    saida[T6 + ".tex"] = t6.replace(T6_VELHO, T6_NOVO).encode("utf-8")
    saida[T6 + ".csv"] = novo[T6 + ".csv"]
    for k in atual:
        assert atual[k] in (novo[k], saida[k]), f"o gerador antigo nao reproduz {k}"
    assert T6_VELHO not in saida[T6 + ".tex"].decode()
    for k in (TN + ".tex", TN + ".csv"):
        a, b = novo[k].decode().split("\n"), saida[k].decode().split("\n")
        assert len(a) == len(b)
        for x, y in zip(a, b):
            if x != y:
                assert re.sub(r"(Proposition|Remark|Lemma)~", "W~", x) == re.sub(r"(Proposition|Remark|Lemma)~", "W~", y)
    for k, v in saida.items():
        (RAIZ / k).write_bytes(v)
    depois = varre(env)
    ruins = [a for a in depois if a["palavra_antes"] != a["ambiente_no_manuscrito"]]
    assert not ruins, ruins
    prov = {"artefato": "TN_T6_v3-13_proveniencia", "script": str(Path(__file__).resolve()), "script_sha256": sha(__file__),
            "comando": "/trabalho/ambientes/s33_amb_virtual/.venv/bin/python scripts/v3_13_gerar_TN_T6.py (cwd = MDPI_Mathematics)",
            "data": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "fontes": {"manuscrito_ambientes": {"arquivo": str(MAIN), "sha256": sha(MAIN)}, "gerador_antigo": {"arquivo": str(ANTIGO), "sha256": sha(ANTIGO)}},
            "ambientes_no_manuscrito": {k: {"ambiente": v[0], "linha_do_begin": v[1]} for k, v in env.items()},
            "varredura_antes_da_correcao": antes, "trocas_feitas_nesta_execucao": trocas, "varredura_depois": depois,
            "T6_rotulo": {"de": T6_VELHO, "para": T6_NOVO, "ocorrencias_no_tex": n6, "csv_alterado": False},
            "saidas": {k: sha(RAIZ / k) for k in saida}}
    with open(RAIZ / "_pipeline" / "TN_T6_v3-13_proveniencia.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=1, ensure_ascii=False)
    mp = RAIZ / "MANIFEST_tabelas_v3-12.json"
    mf = json.load(open(mp, encoding="utf-8"))
    for k in saida:
        ch = str((RAIZ / k).relative_to(B))
        assert ch in mf["saidas"], ch
        mf["saidas"][ch] = sha(RAIZ / k)
    mf["TN_T6_v3-13"] = {"script": prov["script"], "script_sha256": prov["script_sha256"], "proveniencia": str(RAIZ / "_pipeline" / "TN_T6_v3-13_proveniencia.json"),
                         "nota": "o gerador principal regera TN e T6 sem a correcao; reexecutar v3_13_gerar_TN_T6.py depois dele"}
    json.dump(mf, open(mp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    cp = RAIZ / "CONFERENCIAS_tabelas_v3-12.json"
    cf = json.load(open(cp, encoding="utf-8"))
    cf["TN_v3-13_ambientes_dos_ref"] = {"ambientes_do_manuscrito": prov["ambientes_no_manuscrito"], "achados_depois": depois, "todos_conferem": not ruins, "passou": not ruins}
    cf["T6_v3-13_rotulo_FSPL"] = {"de": T6_VELHO, "para": T6_NOVO, "linhas": n6, "valores_inalterados": True, "passou": True}
    json.dump(cf, open(cp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("trocas:", trocas)
    print("varredura depois:", depois)
    print("T6 rotulos trocados:", n6)


if __name__ == "__main__":
    sys.exit(main())
