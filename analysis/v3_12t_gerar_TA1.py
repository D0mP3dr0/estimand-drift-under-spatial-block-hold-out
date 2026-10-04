#!/usr/bin/env python3
"""Write the architecture-and-loss part of the supplementary hyper-parameter table (TA1a).

The table is the one produced by the main table generator (scripts/v3_12_gerar_tabelas.py, function
gerar_ta1) minus the single row that starts with "Width-sensitivity arm (10 runs)", which the article
does not report; the rows "Parameters, nominal" and "Parameters receiving gradient" are kept.
The main generator is imported as a module (its main() runs only under __main__) and left unmodified;
its output folder is redirected to a temporary directory and gerar_ta1() is run on the hyper-parameter
sheet it reads (its ENT["B7.2"], under fase4/B7.2_hiperparametros/).
Checks: the optimisation-and-cost part (TA1b), the full TA1 .tex and the TA1 CSV it generates must be
byte-identical to the files already in tables_v3-12/ (they are not rewritten); the existing TA1a must
equal either the generated table or the table without that row, so re-running is idempotent.
Outputs: tables_v3-12/TA1a_hiperparametros_arquitetura_perda_v3-12.tex; the provenance record
tables_v3-12/_pipeline/TA1a_v3-12t_proveniencia.json (SHA-256 of this script, of the generator, of the
sheet and of the output, and the dropped row); the TA1a hash and a TA1a_v3-12t entry in
MANIFEST_tabelas_v3-12.json. The main generator writes TA1a with the row, so run this script after it.
The PDF of the supplementary table is compiled separately with pdflatex. Deterministic (no random numbers).

Usage: python analysis/v3_12t_gerar_TA1.py
"""
import datetime
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

M = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
ANTIGO = M / "scripts" / "v3_12_gerar_tabelas.py"
RAIZ = M / "_v3_2026-09-25" / "redacao_v3-12" / "tables_v3-12"
ROTULO_RETIRADO = "Width-sensitivity arm (10 runs)"
NOMES = {"a": "TA1a_hiperparametros_arquitetura_perda_v3-12.tex", "b": "TA1b_hiperparametros_otimizacao_custo_v3-12.tex",
         "inteira": "TA1_hiperparametros_v3-12.tex", "csv": "TA1_hiperparametros_v3-12.csv"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    spec = importlib.util.spec_from_file_location("v3_12_gerar_tabelas_antigo", ANTIGO)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    sha_antigo = sha(ANTIGO)
    with tempfile.TemporaryDirectory() as tmp:
        g.OUT = Path(tmp)
        g.gerar_ta1()
        novo = {k: (Path(tmp) / v).read_bytes() for k, v in NOMES.items()}
    atual = {k: (RAIZ / v).read_bytes() for k, v in NOMES.items()}
    # Tables left unchanged must be reproduced byte for byte by the main generator
    for k in ("b", "inteira", "csv"):
        assert novo[k] == atual[k], f"the main table generator does not reproduce {NOMES[k]} (files in this folder differ from its output)"
    linhas = novo["a"].decode("utf-8").split("\n")
    alvo = [i for i, ln in enumerate(linhas) if ln.startswith(ROTULO_RETIRADO + " & ")]
    assert len(alvo) == 1, alvo
    retirada = linhas[alvo[0]]
    saida = "\n".join(linhas[:alvo[0]] + linhas[alvo[0] + 1:]).encode("utf-8")
    assert len(saida.decode("utf-8").split("\n")) == len(linhas) - 1
    assert b"Parameters, nominal" in saida and b"Parameters receiving gradient" in saida and ROTULO_RETIRADO.encode() not in saida
    # Existing TA1a: either the full generated table or, on a re-run, the table already without the row
    assert atual["a"] in (novo["a"], saida), "the main table generator does not reproduce the existing TA1a"
    destino = RAIZ / NOMES["a"]
    destino.write_bytes(saida)
    folha = g.ENT["B7.2"]
    prov = {"artefato": "TA1a_v3-12t_proveniencia", "script": str(Path(__file__).resolve()), "script_sha256": sha(Path(__file__)),
            "comando": "/trabalho/ambientes/s33_amb_virtual/.venv/bin/python scripts/v3_12t_gerar_TA1.py (cwd = MDPI_Mathematics)",
            "data": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "gerador_reutilizado": {"arquivo": str(ANTIGO), "sha256": sha_antigo, "funcao": "gerar_ta1 (nao alterado)"},
            "fontes": {"folha_B7.2": {"arquivo": str(folha), "sha256": sha(folha)}},
            "linha_retirada": retirada, "numero_da_linha_no_arquivo_antigo": alvo[0] + 1,
            "conferencias": {"TA1b_byte_a_byte_igual_ao_existente": True, "TA1_inteira_byte_a_byte_igual_ao_existente": True,
                             "TA1_csv_byte_a_byte_igual_ao_existente": True, "TA1a_gerada_igual_a_existente_antes_da_retirada": True,
                             "TA1a_nova_tem_uma_linha_a_menos": True},
            "saida": {"arquivo": str(destino), "sha256": sha(destino)},
            "nota": "TA1_hiperparametros_v3-12.tex (full table, not used by Table_S1) and the TA1 CSV keep the width-sensitivity row; "
                    "neither file is part of the supplementary table"}
    with open(RAIZ / "_pipeline" / "TA1a_v3-12t_proveniencia.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=1, ensure_ascii=False)
    # Manifest: update only the TA1a hash and add one TA1a_v3-12t key; everything else is kept
    mpath = RAIZ / "MANIFEST_tabelas_v3-12.json"
    mf = json.load(open(mpath, encoding="utf-8"))
    chave = str(destino.relative_to(M / "_v3_2026-09-25"))
    assert chave in mf["saidas"], chave
    mf["saidas"][chave] = sha(destino)
    mf["TA1a_v3-12t"] = {"script": prov["script"], "script_sha256": prov["script_sha256"], "linha_retirada": retirada,
                         "proveniencia": str(RAIZ / "_pipeline" / "TA1a_v3-12t_proveniencia.json"),
                         "nota": "o gerador principal v3_12_gerar_tabelas.py regera a TA1a COM a linha; reexecutar v3_12t_gerar_TA1.py depois dele"}
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(mf, f, indent=1, ensure_ascii=False)
    print("gravado", destino, "linha retirada:", retirada)


if __name__ == "__main__":
    sys.exit(main())
