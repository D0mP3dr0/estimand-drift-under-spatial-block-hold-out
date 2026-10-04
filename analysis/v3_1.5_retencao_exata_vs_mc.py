#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test 1.5: exact without-replacement retention versus the mean-field approximation.

In the polynomial expansion of the retained block area, the mean field weights the
area covered by k trimming-neighbour zones with q^k (q constant). Given that block B
is a test block, the exact probability that k specific neighbours are all test
blocks under a random permutation of roles is the falling-factorial ratio
(k_te - 1)_k / (N - 1)_k (for validation, N - k_tr replaces k_te). For each city and
design (g = 10 km and g = 5 km, b = 2 km) the script evaluates both forms on the real
block lattice (thin border column/row, extents and role counts from
montecarlo_retencao_r3_g5.py) with lattice_retention() of fismat_sympy_eqR.py
(in this folder), and compares them with the nodal retention
over 20 split seeds (Q1 and Q3) and with variant C of the block Monte Carlo.
Decision criterion (criterio_1.5.json): the exact form reduces the relative error
against nodal retention in every city x quadrant x design x role combination.

Inputs: fismat_sympy_eqR.py and montecarlo_retencao_r3_g5.py (imported by path), the
20-seed nodal record (NODAL_REAL_ARTEFATO), the block Monte Carlo record
(MC_BLOCOS_ARTEFATO) and the criterion (criteria/criterio_1.5.json).
Output: results/fase1/1.5_retencao_exata_vs_mc.json. Deterministic (no random draws).
Usage: python v3_1.5_retencao_exata_vs_mc.py --out <json> [--log <file>]
"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
INTERNAL_NOTES_DIR = SCRIPTS_DIR.parent / "draft_v2" / "internal/notes"
CRITERIO_PATH = SCRIPTS_DIR.parent / "_v3_2026-09-25" / "criterios" / "criterio_1.5.json"

NODAL_REAL_ARTEFATO = Path(
    "internal/artifact_01.json")
MC_BLOCOS_ARTEFATO = Path(
    "internal/artifact_02.json")

CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
CONFIGS = [("g10b2", 10.0, 2.0), ("g5b2", 5.0, 2.0)]
QS_REPRESENTATIVOS = ["Q1", "Q3"]  # Q1/Q2 and Q3/Q4 share the same geometry


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def import_module(path, name):
    """Import a module from a file path (file names here are not valid module names)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    logf = open(args.log, "a") if args.log else None

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n"); logf.flush()

    t0 = time.time()
    fismat_mod = import_module(INTERNAL_NOTES_DIR / "fismat_sympy_eqR.py", "fismat_mod")
    mcg5_mod = import_module(SCRIPTS_DIR / "montecarlo_retencao_r3_g5.py", "mcg5_mod")

    criterio = json.loads(CRITERIO_PATH.read_text())
    nodal_real_raw = json.loads(NODAL_REAL_ARTEFATO.read_text())
    mc_blocos_raw = json.loads(MC_BLOCOS_ARTEFATO.read_text())

    saida = {
        "frente": "internal/analysis",
        "teste": "1.5",
        "alegacao": "A1",
        "pergunta": criterio["pergunta"],
        "criterio": criterio,
        "insumos": {
            "script_fismat_eqR": {"arquivo": str(INTERNAL_NOTES_DIR / "fismat_sympy_eqR.py"),
                                   "sha256": sha256_file(INTERNAL_NOTES_DIR / "fismat_sympy_eqR.py")},
            "script_mc_blocos_r3g5": {"arquivo": str(SCRIPTS_DIR / "montecarlo_retencao_r3_g5.py"),
                                       "sha256": sha256_file(SCRIPTS_DIR / "montecarlo_retencao_r3_g5.py")},
            "artefato_nodal_real": {"arquivo": str(NODAL_REAL_ARTEFATO), "sha256": sha256_file(NODAL_REAL_ARTEFATO)},
            "artefato_mc_blocos": {"arquivo": str(MC_BLOCOS_ARTEFATO), "sha256": sha256_file(MC_BLOCOS_ARTEFATO)},
        },
        "por_celula": {},
        "comando_rodado": "/trabalho/ambientes/s33_amb_virtual/.venv/bin/python v3_1.5_retencao_exata_vs_mc.py --out fase1/1.5_retencao_exata_vs_mc.json",
    }
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    nodal_por_config = {}
    for r in nodal_real_raw["resumo"]:
        if r["Q"] not in QS_REPRESENTATIVOS:
            continue
        cfg = "g10b2" if r["g_km"] == 10.0 else "g5b2"
        nodal_por_config[(r["cidade"], r["Q"], cfg)] = {
            "retencao_test_media": r["retencao_test"]["media"],
            "retencao_val_media": r["retencao_val"]["media"],
            "n_seeds": r["retencao_test"]["n"],
        }
    mc_blocos_por_config = {}
    for cfg in ("g10b2", "g5b2"):
        for cidade, ent in mc_blocos_raw["por_config"][cfg]["por_cidade"].items():
            C = ent["variantes"]["C"]
            mc_blocos_por_config[(cidade, cfg)] = {"ret_te_media": C["ret_te_media"], "ret_va_media": C["ret_va_media"]}

    resultados = {}
    comparacoes_reducao = []  # one entry per (cell, design, role): does the exact form reduce the error?

    for cidade in CIDADES:
        ext = mcg5_mod.CIDADES_EXT[cidade]
        Wx, Wy = ext["ext_x_km"], ext["ext_y_km"]
        for cfg, g_km, b_km in CONFIGS:
            nx, ny, colw, rowh = mcg5_mod.grid_dims(Wx, Wy, g_km)
            N = nx * ny
            n_tr, n_va, n_te = mcg5_mod.n_blocos_por_papel(N)
            q_te = n_te / N
            q_va = 1 - n_tr / N

            # Non-trimming block counts: test blocks for test, all but training blocks for validation;
            # raster resolution 0.02 km.
            t_c = time.time()
            lr_te = fismat_mod.lattice_retention(Wx, Wy, g_km, b_km, None, n_te, N, full_blocks=False, res=0.02)
            lr_va = fismat_mod.lattice_retention(Wx, Wy, g_km, b_km, None, N - n_tr, N, full_blocks=False, res=0.02)
            dt = time.time() - t_c

            mc_blocos = mc_blocos_por_config.get((cidade, cfg))

            for q_rep in QS_REPRESENTATIVOS:
                nodal_real = nodal_por_config.get((cidade, q_rep, cfg))
                if nodal_real is None:
                    continue
                chave = f"{cidade}_{q_rep}_{cfg}"
                entry = {
                    "cidade": cidade, "Q": q_rep, "config": cfg, "g_km": g_km, "b_km": b_km,
                    "N_blocos_total": N, "n_blocos": {"train": n_tr, "val": n_va, "test": n_te},
                    "q_te_realizado": q_te, "q_va_realizado": q_va,
                    "campo_medio": {"teste": lr_te["mean_field"], "val": lr_va["mean_field"]},
                    "exato_sem_reposicao": {"teste": lr_te["exato_permutacao"], "val": lr_va["exato_permutacao"]},
                    "nodal_real_20seeds": nodal_real,
                    "mc_blocos_variante_C": mc_blocos,
                    "tempo_lattice_retention_s": dt,
                }
                for papel, campo_nodal in (("teste", "retencao_test_media"), ("val", "retencao_val_media")):
                    nodal_v = nodal_real[campo_nodal]
                    err_cm = abs(entry["campo_medio"][papel] - nodal_v) / nodal_v
                    err_ex = abs(entry["exato_sem_reposicao"][papel] - nodal_v) / nodal_v
                    entry.setdefault("erro_relativo_vs_nodal_real", {})[papel] = {
                        "campo_medio": err_cm, "exato_sem_reposicao": err_ex,
                        "exato_reduz_erro": bool(err_ex < err_cm),
                        "reducao_absoluta_pp": (err_cm - err_ex) * 100,
                    }
                    comparacoes_reducao.append({"chave": chave, "papel": papel,
                                                 "exato_reduz": bool(err_ex < err_cm),
                                                 "err_campo_medio": err_cm, "err_exato": err_ex})
                resultados[chave] = entry
                log(f"{chave}: campo_medio_te={entry['campo_medio']['teste']:.5f} exato_te={entry['exato_sem_reposicao']['teste']:.5f} "
                    f"nodal_real_te={nodal_real['retencao_test_media']:.5f} "
                    f"err_cm={entry['erro_relativo_vs_nodal_real']['teste']['campo_medio']:.4f} "
                    f"err_exato={entry['erro_relativo_vs_nodal_real']['teste']['exato_sem_reposicao']:.4f}")
                saida["por_celula"] = resultados
                Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    todas_reduzem = all(c["exato_reduz"] for c in comparacoes_reducao)
    saida["resumo"] = {
        "n_comparacoes": len(comparacoes_reducao),
        "n_exato_reduz_erro": sum(1 for c in comparacoes_reducao if c["exato_reduz"]),
        "todas_reduzem": todas_reduzem,
        "comparacoes_onde_exato_NAO_reduz": [c for c in comparacoes_reducao if not c["exato_reduz"]],
        "erro_medio_campo_medio": sum(c["err_campo_medio"] for c in comparacoes_reducao) / len(comparacoes_reducao),
        "erro_medio_exato": sum(c["err_exato"] for c in comparacoes_reducao) / len(comparacoes_reducao),
    }
    saida["veredito_vs_criterio"] = (
        "a esperanca exata sem reposicao reduz o erro relativo contra a retencao nodal real EM TODAS as "
        f"{len(comparacoes_reducao)} combinacoes (cidade x Q-representativo x config x papel) testadas -- "
        "a Tabela 2 deve reportar exato+nodal, com o campo medio declarado como aproximacao (erro residual "
        "explicito por celula)"
        if todas_reduzem else
        "a esperanca exata NAO reduziu o erro em pelo menos uma combinacao -- ver "
        "resumo.comparacoes_onde_exato_NAO_reduz; a Tabela 2 mantem o campo medio como numero central e cita "
        "a esperanca exata so como alternativa sem ganho universal"
    )
    saida["nao_verificado"] = [
        {"item": "correlacao entre blocos de teste (pares/tercetos) alem do efeito medio de amostragem sem reposicao",
         "motivo": ("lattice_retention() modela a probabilidade de CADA vizinho aparador individual ser de "
                    "teste via hipergeometrico marginal (falling factorial), nao a distribuicao conjunta exata "
                    "dos 8 vizinhos de um bloco sob o sorteio simultaneo dos n_g grupos -- aproximacao razoavel "
                    "para os n_te,N em jogo (n_te/N ~0.15-0.17), mas nao e a hipergeometrica multivariada "
                    "completa; nao quantificado nesta rodada por orcamento.")},
    ]
    saida["tempo_total_s"] = time.time() - t0
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))
    log(f"CONCLUIDO em {saida['tempo_total_s']:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
