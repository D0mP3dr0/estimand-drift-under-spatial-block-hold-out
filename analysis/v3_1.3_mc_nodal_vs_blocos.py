#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test 1.3: nodal Monte Carlo versus whole-block Monte Carlo of buffer retention.

The whole-block Monte Carlo (variant C of montecarlo_retencao_r3_g5.py, continuous
block areas) differs by a few percent from the nodal retention measured on the
real node grid over 20 split seeds (varredura_split_geometria.py). This script
adds a third estimator, a nodal Monte Carlo: it runs the exact frozen split
(split_espacial_3vias_exato, exact cKDTree distances, validation trimmed against
training and test trimmed against training plus retained validation) on the
3600 x 3600 node positions of each cell, for 50 new split seeds (1000-1049), and
compares its mean retention with the nodal retention over 20 seeds and with the
block Monte Carlo. Decision criterion (criterio_1.3.json): if the nodal Monte
Carlo matches the nodal retention within 1% relative error in every cell, the gap
is attributed to the continuous-area block abstraction, not to node granularity.
Q1 and Q3 are run per city: Q1/Q2 and Q3/Q4 share the same geometry and give
identical nodal retention.

Inputs: varredura_split_geometria.py (imported), its 20-seed nodal record
(NODAL_REAL_ARTEFATO), the block Monte Carlo record (MC_BLOCOS_ARTEFATO) and the
criterion (criteria/criterio_1.3.json; read from _v3_2026-09-25/criterios/).
Output: results/fase1/1.3_mc_nodal_vs_blocos.json, rewritten after each cell.
Usage: python v3_1.3_mc_nodal_vs_blocos.py --out <json> [--log <file>]
"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).parent
CRITERIO_PATH = (SCRIPTS_DIR.parent / "_v3_2026-09-25" / "criterios" / "criterio_1.3.json")

NODAL_REAL_ARTEFATO = Path(
    "internal/artifact_01.json")
MC_BLOCOS_ARTEFATO = Path(
    "internal/artifact_02.json")

CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
CONFIGS = [("g10b2", 10.0, 2.0), ("g5b2", 5.0, 2.0)]
QS_REPRESENTATIVOS = ["Q1", "Q3"]
N_PERMUTACOES = 50
SEEDS_MC_NODAL = list(range(1000, 1000 + N_PERMUTACOES))  # disjoint from the 20 seeds (42, 1-19) of the nodal record


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
    geo_mod = import_module(SCRIPTS_DIR / "varredura_split_geometria.py", "geo_mod")

    criterio = json.loads(CRITERIO_PATH.read_text())
    nodal_real_raw = json.loads(NODAL_REAL_ARTEFATO.read_text())
    mc_blocos_raw = json.loads(MC_BLOCOS_ARTEFATO.read_text())

    saida = {
        "frente": "internal/analysis",
        "teste": "1.3",
        "alegacao": "A1",
        "pergunta": criterio["pergunta"],
        "criterio": criterio,
        "insumos": {
            "script_geometria": {"arquivo": str(SCRIPTS_DIR / "varredura_split_geometria.py"),
                                  "sha256": sha256_file(SCRIPTS_DIR / "varredura_split_geometria.py")},
            "artefato_nodal_real": {"arquivo": str(NODAL_REAL_ARTEFATO), "sha256": sha256_file(NODAL_REAL_ARTEFATO)},
            "artefato_mc_blocos": {"arquivo": str(MC_BLOCOS_ARTEFATO), "sha256": sha256_file(MC_BLOCOS_ARTEFATO)},
        },
        "n_permutacoes_mc_nodal": N_PERMUTACOES,
        "seeds_mc_nodal": SEEDS_MC_NODAL,
        "regra_de_aparo_mc_nodal": criterio["metodo_declarado"]["regra_de_aparo_mc_nodal"],
        "regra_de_aparo_mc_blocos": criterio["metodo_declarado"]["regra_de_aparo_mc_blocos"],
        "por_celula": {},
        "comando_rodado": "/trabalho/ambientes/s33_amb_virtual/.venv/bin/python v3_1.3_mc_nodal_vs_blocos.py --out fase1/1.3_mc_nodal_vs_blocos.json",
    }
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    # Nodal retention (mean over 20 seeds) per (city, quadrant, design).
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

    # Whole-block Monte Carlo, variant C, per (city, design); read, not recomputed.
    mc_blocos_por_config = {}
    for cfg in ("g10b2", "g5b2"):
        for cidade, ent in mc_blocos_raw["por_config"][cfg]["por_cidade"].items():
            C = ent["variantes"]["C"]
            mc_blocos_por_config[(cidade, cfg)] = {
                "ret_te_media": C["ret_te_media"], "ret_va_media": C["ret_va_media"],
                "n_trials": C["n_trials"],
            }

    # Nodal Monte Carlo: retention = nodes after buffer / nodes before buffer, per role.
    geo_por_celula = geo_mod.geometria_todas_celulas()
    resultados_finais = {}
    conclusoes_por_celula = []
    for cidade in CIDADES:
      for q in QS_REPRESENTATIVOS:
        geo = geo_por_celula[(cidade, q)]
        lon, lat = geo_mod.build_synthetic_grid(geo["lon_min_deg"], geo["lon_max_deg"],
                                                 geo["lat_min_deg"], geo["lat_max_deg"])
        pos_m = geo_mod.latlon_graus_para_metros(lon, lat)
        for cfg, g_km, b_km in CONFIGS:
            t_c = time.time()
            ret_te, ret_va = [], []
            for seed in SEEDS_MC_NODAL:
                parts, group_ids, info = geo_mod.split_espacial_3vias_exato(
                    pos_m, grid_km=g_km, buffer_km=b_km, fracs=geo_mod.FRACS, split_seed=seed)
                n_va_bruto = info["n_nos_antes_do_buffer"]["val"]
                n_te_bruto = info["n_nos_antes_do_buffer"]["test"]
                n_va_apos = info["n_nos_apos_buffer"]["val"]
                n_te_apos = info["n_nos_apos_buffer"]["test"]
                ret_va.append(n_va_apos / n_va_bruto if n_va_bruto else np.nan)
                ret_te.append(n_te_apos / n_te_bruto if n_te_bruto else np.nan)
            ret_te = np.array(ret_te); ret_va = np.array(ret_va)
            mc_nodal = {
                "ret_te_media": float(np.nanmean(ret_te)), "ret_te_dp": float(np.nanstd(ret_te)),
                "ret_va_media": float(np.nanmean(ret_va)), "ret_va_dp": float(np.nanstd(ret_va)),
                "n_permutacoes": N_PERMUTACOES, "tempo_s": time.time() - t_c,
            }
            nodal_real = nodal_por_config.get((cidade, q, cfg))
            mc_blocos = mc_blocos_por_config.get((cidade, cfg))

            err_mcnodal_vs_nodalreal_te = abs(mc_nodal["ret_te_media"] - nodal_real["retencao_test_media"]) / nodal_real["retencao_test_media"]
            err_mcnodal_vs_nodalreal_va = abs(mc_nodal["ret_va_media"] - nodal_real["retencao_val_media"]) / nodal_real["retencao_val_media"]
            err_mcblocos_vs_nodalreal_te = abs(mc_blocos["ret_te_media"] - nodal_real["retencao_test_media"]) / nodal_real["retencao_test_media"]
            err_mcblocos_vs_nodalreal_va = abs(mc_blocos["ret_va_media"] - nodal_real["retencao_val_media"]) / nodal_real["retencao_val_media"]

            chave = f"{cidade}_{q}_{cfg}"
            resultados_finais[chave] = {
                "cidade": cidade, "Q": q, "config": cfg, "g_km": g_km, "b_km": b_km,
                "mc_nodal": mc_nodal,
                "nodal_real_20seeds": nodal_real,
                "mc_blocos_variante_C": mc_blocos,
                "erro_relativo_mc_nodal_vs_nodal_real": {"teste": err_mcnodal_vs_nodalreal_te, "val": err_mcnodal_vs_nodalreal_va},
                "erro_relativo_mc_blocos_vs_nodal_real": {"teste": err_mcblocos_vs_nodalreal_te, "val": err_mcblocos_vs_nodalreal_va},
                "mc_nodal_bate_criterio_1pct": {
                    "teste": bool(err_mcnodal_vs_nodalreal_te < 0.01),
                    "val": bool(err_mcnodal_vs_nodalreal_va < 0.01),
                },
            }
            conclusoes_por_celula.append(err_mcnodal_vs_nodalreal_te < 0.01 and err_mcnodal_vs_nodalreal_va < 0.01)
            log(f"{chave}: MC_nodal te={mc_nodal['ret_te_media']:.5f} (nodal_real={nodal_real['retencao_test_media']:.5f}, "
                f"err={err_mcnodal_vs_nodalreal_te:.4f}) | mc_blocos te={mc_blocos['ret_te_media']:.5f} "
                f"(err_vs_nodal_real={err_mcblocos_vs_nodalreal_te:.4f}) [{time.time()-t_c:.1f}s]")
            saida["por_celula"] = resultados_finais
            Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    todas_batem = all(conclusoes_por_celula)
    saida["resumo"] = {
        "n_celulas_testadas": len(resultados_finais),
        "todas_celulas_mc_nodal_bate_1pct_criterio": todas_batem,
        "erro_relativo_mc_nodal_vs_nodal_real_teste_max": max(
            v["erro_relativo_mc_nodal_vs_nodal_real"]["teste"] for v in resultados_finais.values()),
        "erro_relativo_mc_nodal_vs_nodal_real_val_max": max(
            v["erro_relativo_mc_nodal_vs_nodal_real"]["val"] for v in resultados_finais.values()),
        "erro_relativo_mc_blocos_vs_nodal_real_teste_range": [
            min(v["erro_relativo_mc_blocos_vs_nodal_real"]["teste"] for v in resultados_finais.values()),
            max(v["erro_relativo_mc_blocos_vs_nodal_real"]["teste"] for v in resultados_finais.values())],
        "erro_relativo_mc_blocos_vs_nodal_real_val_range": [
            min(v["erro_relativo_mc_blocos_vs_nodal_real"]["val"] for v in resultados_finais.values()),
            max(v["erro_relativo_mc_blocos_vs_nodal_real"]["val"] for v in resultados_finais.values())],
    }
    saida["veredito_vs_criterio"] = (
        "a lacuna MC_blocos-vs-nodal_real (4.3-6.0% teste; 1.2-6.2% val, com 5.0-6.2% no subgrupo g5) "
        "e' atribuida a regra de aparo/abstracao de area continua do MC de blocos, NAO a granularidade: "
        "o MC nodal (mesma regra de aparo sequencial do protocolo, sobre nos reais, N=50 permutacoes) "
        "bate a retencao nodal real (20 seeds) dentro de 1% em TODAS as 8 celulas testadas -- a frase de "
        "granularidade sai do texto"
        if todas_batem else
        "o MC nodal NAO bateu a retencao nodal real dentro de 1% em pelo menos uma celula -- a granularidade "
        "(ou outro efeito nao capturado por este MC) permanece sem explicacao medida; ver por_celula para as "
        "celulas que falharam o criterio"
    )
    saida["nao_verificado"] = [
        {"item": "Q2 e Q4 (so Q1 e Q3 testados por celula, por cidade)",
         "motivo": ("conferido no resumo do artefato nodal_real ANTES de decidir o escopo: Q1 e Q2 tem "
                    "retencao_test.media IDENTICA (ate a 12a casa) em toda celula/config, e Q3 e Q4 tambem "
                    "sao identicos entre si -- mas Q1/Q2 DIFEREM de Q3/Q4 por ~0.3-1% (geometrias distintas "
                    "por par). Rodar Q1 e Q3 cobre as DUAS geometrias distintas de cada cidade; Q2 e Q4 "
                    "repetiriam exatamente Q1 e Q3.")},
    ]
    saida["pico_ram_gb"] = None
    saida["tempo_total_s"] = time.time() - t0
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))
    log(f"CONCLUIDO em {saida['tempo_total_s']:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
