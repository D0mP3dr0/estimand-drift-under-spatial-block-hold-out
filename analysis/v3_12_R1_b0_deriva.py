#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Control run R1: error drift of the constant predictor with buffer b = 0.

Repeats the error-drift computation of v3_2.1_3.1_deriva_calibracao.py with the
buffer set to zero, so that the between-draw spread at b = 0 can be compared with
the spread at b = 2 km. The drift module is imported unchanged (its SHA-256 is
checked at start-up and the script aborts on mismatch); the only change is the
module-level BUFFER_KM, which processar_celula reads on every call. Partition
(three-way spatial split, g = 10 km, fractions 0.70/0.15/0.15), constant predictor,
FSPL calibrations (a) and (b) and per-population MAE are those of the imported
module. With b = 0 the KD-tree erosion branch does not run, so blocks are not eroded.

Seeds: the 60 split draws listed in fase2/2.1_deriva_erro_baselines_16x60rnd.json
(nota_divergencia_seeds.seeds_usados_nesta_rodada), checked against the b = 2 partial
file fase2/_v3_2.1_3.1_parcial_16x60rnd.json. Runs on CPU; tensor digests are not
recomputed, the manifest digest is copied into the records.

Inputs: the two fase2 records above, fase3/3.1_calibracao_validos_vs_mediana_16x60rnd.json
and the decision criterion criterios/criterio_R1_b0.json.
Outputs (fase4/): _parcial_R1_b0/<cell>.json, R1_b0_por_sorteio.json, R1_b0_resumo.json,
and R1_b0_validacao_b2.json.

Usage:
  python v3_12_R1_b0_deriva.py --validar         # b = 2 km, lins_Q1, first 5 draws; exact match with the partial file
  python v3_12_R1_b0_deriva.py --celula bauru_Q1  # b = 0, one cell, 60 draws (skipped if its partial is already ok)
  python v3_12_R1_b0_deriva.py --consolidar      # merge the 16 per-cell partials into the two records
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
SCRIPT_SHA256 = hashlib.sha256(HERE.read_bytes()).hexdigest()
BASE = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
ORIG_PATH = BASE / "scripts" / "v3_2.1_3.1_deriva_calibracao.py"
ORIG_SHA256 = "9b9bcf91bab51621107a3d7978964d46dd16c5ec1a76a3789842ae1bce77f3bc"
V3 = BASE / "_v3_2026-09-25"
F2_JSON = V3 / "fase2" / "2.1_deriva_erro_baselines_16x60rnd.json"
F3_JSON = V3 / "fase3" / "3.1_calibracao_validos_vs_mediana_16x60rnd.json"
PARCIAL_B2 = V3 / "fase2" / "_v3_2.1_3.1_parcial_16x60rnd.json"
CRITERIO = V3 / "criterios" / "criterio_R1_b0.json"
OUT = V3 / "fase4"
PARCIAL_DIR = OUT / "_parcial_R1_b0"
OUT_SORTEIO = OUT / "R1_b0_por_sorteio.json"
OUT_RESUMO = OUT / "R1_b0_resumo.json"
OUT_VALID = OUT / "R1_b0_validacao_b2.json"
POPS = ("validos", "sentinela", "todos")

_orig_sha_real = hashlib.sha256(ORIG_PATH.read_bytes()).hexdigest()
if _orig_sha_real != ORIG_SHA256:
    raise SystemExit(f"ABORTA: sha do script original diverge ({_orig_sha_real})")

_spec = importlib.util.spec_from_file_location("deriva_orig", ORIG_PATH)
orig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(orig)  # main() is guarded by __name__; this only defines functions and constants


def agora() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[{agora()}] {msg}", flush=True)


def carregar_seeds() -> list:
    a = json.loads(F2_JSON.read_text(encoding="utf-8"))
    seeds = a["nota_divergencia_seeds"]["seeds_usados_nesta_rodada"]
    p = json.loads(PARCIAL_B2.read_text(encoding="utf-8"))
    assert len(seeds) == 60 and len(set(seeds)) == 60, "esperados 60 sorteios distintos"
    assert seeds == p["seeds_usados"], "sementes do JSON final != parcial de b=2"
    return [int(s) for s in seeds]


def rodar_celula(chave: str, seeds: list, buffer_km: float) -> dict:
    cidade, quad = chave.split("_")
    orig.BUFFER_KM = float(buffer_km)  # the only difference from the imported drift module
    assert orig.GRID_KM == 10.0 and orig.FRACS == (0.70, 0.15, 0.15) and orig.FREQ_MHZ == 900.0
    return orig.processar_celula(cidade, quad, verificar_sha=False, seeds=seeds)


def deep_diff(a, b, path="", out=None):
    """Recursively compare two JSON-like structures.

    Returns a dict with the divergent (path, a, b) entries, the largest absolute
    numeric difference and the number of numeric leaves compared.
    """
    if out is None:
        out = {"divergentes": [], "max_abs": 0.0, "n_numeros": 0}
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out["divergentes"].append((path, "chaves", sorted(set(a) ^ set(b))))
        for k in a:
            if k in b:
                deep_diff(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["divergentes"].append((path, "len", (len(a), len(b))))
        for i, (x, y) in enumerate(zip(a, b)):
            deep_diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str):
        if a != b:
            out["divergentes"].append((path, a, b))
    else:
        out["n_numeros"] += 1
        d = abs(float(a) - float(b))
        out["max_abs"] = max(out["max_abs"], d)
        if a != b:
            out["divergentes"].append((path, a, b))
    return out


def validar():
    seeds = carregar_seeds()[:5]
    chave = "lins_Q1"
    log(f"VALIDACAO b=2 km, {chave}, sorteios {seeds}")
    rec = rodar_celula(chave, seeds, 2.0)
    p = json.loads(PARCIAL_B2.read_text(encoding="utf-8"))
    ref = p["celulas"][chave]["por_sorteio"][:5]
    novo = json.loads(json.dumps(rec["por_sorteio"]))  # round-trip through JSON so types match the stored file
    d = deep_diff(novo, ref)
    ok = (len(d["divergentes"]) == 0 and rec["status"] == "ok"
          and [s["split_seed"] for s in novo] == [s["split_seed"] for s in ref])
    saida = {
        "id": "R1_b0_validacao_b2", "celula": chave, "buffer_km": 2.0, "n_sorteios": 5,
        "sementes": seeds, "referencia": str(PARCIAL_B2),
        "referencia_sha256": orig.sha256_file(PARCIAL_B2),
        "criterio_de_aceite": "igualdade exata (==) de todos os campos por sorteio contra o parcial gravado",
        "reproduz": bool(ok), "n_numeros_comparados": d["n_numeros"],
        "max_abs_diferenca": d["max_abs"], "divergentes": d["divergentes"][:20],
        "n_test_por_sorteio": [s["n_test"] for s in novo],
        "n_validos_teste_por_sorteio": [s["n_pop_teste"]["validos"] for s in novo],
        "mae_constante_validos_por_sorteio": [s["mae_constante_teste"]["validos"] for s in novo],
        "script_sha256": SCRIPT_SHA256, "script_original_sha256": ORIG_SHA256,
        "comando": " ".join(sys.argv), "data_utc": agora(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_VALID.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"validacao reproduz={ok}, max_abs={d['max_abs']}, n_numeros={d['n_numeros']} -> {OUT_VALID}")
    sys.exit(0 if ok else 2)


def compacta_sorteio(s: dict) -> dict:
    """Keep only the per-draw fields used by the summary (failed draws are returned as is)."""
    if s.get("status") != "ok":
        return s
    n_pop = s["n_pop_teste"]
    return {
        "split_seed": s["split_seed"], "status": "ok",
        "n_train": s["n_train"], "n_teste": s["n_test"],
        "n_validos_teste": n_pop["validos"], "n_sentinela_teste": n_pop["sentinela"],
        "fracao_valida_teste": n_pop["validos"] / s["n_test"],
        "mae_constante": s["mae_constante_teste"],
        "mae_fspl_a_contaminado": s["mae_modelo_a_contaminado_teste"]["fspl"],
        "mae_fspl_b_validos": s["mae_modelo_b_validos_teste"]["fspl"],
        "inversao_constante_gt_fspl": s["inversao_constante_gt_fspl"],
    }


def rodar_uma(chave: str):
    PARCIAL_DIR.mkdir(parents=True, exist_ok=True)
    arq = PARCIAL_DIR / f"{chave}.json"
    if arq.exists() and json.loads(arq.read_text(encoding="utf-8")).get("status") == "ok":
        log(f"{chave}: parcial ok, pulando")
        return
    seeds = carregar_seeds()
    log(f"=== {chave} b=0, {len(seeds)} sorteios ===")
    t0 = time.perf_counter()
    try:
        rec = rodar_celula(chave, seeds, 0.0)
    except Exception as e:  # noqa: BLE001
        rec = {"status": "erro_excecao", "erro": repr(e), "traceback": traceback.format_exc()}
    rec["tempo_total_s"] = time.perf_counter() - t0
    rec["celula"] = chave
    rec["buffer_km"] = 0.0
    rec["por_sorteio_compacto"] = [compacta_sorteio(s) for s in rec.get("por_sorteio", [])]
    rec.pop("por_sorteio", None)
    arq.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"{chave}: status={rec['status']} em {rec['tempo_total_s']:.1f}s -> {arq}")


def dp1(v):
    v = [x for x in v if x is not None]
    return float(np.std(v, ddof=1)) if len(v) > 1 else None


def razao(num, den):
    return None if (num is None or den in (None, 0)) else float(num / den)


def stats_celula(sorteios: list) -> dict:
    """Per-cell statistics over split draws, as in the error-drift and inversion tables.

    Standard deviations use ddof=1. Draws with no valid node in the test partition are
    excluded (and counted) from both the valid-node MAE and the valid fraction; a variant
    of the valid-fraction spread over all non-empty test partitions is reported separately.
    """
    ok =[s for s in sorteios if s.get("status") == "ok"]
    com_valido = [s for s in ok if s["mae_constante"]["validos"] is not None]
    mae_v = [s["mae_constante"]["validos"] for s in com_valido]
    fr_v = [s["fracao_valida_teste"] for s in com_valido]
    fr_todos = [s["fracao_valida_teste"] for s in ok]
    inv = {}
    for cal, nome in (("a_contaminada", "a"), ("b_validos", "b")):
        for pop in ("todos", "validos"):
            inv[f"{nome}_{pop}"] = int(sum(1 for s in ok if s["inversao_constante_gt_fspl"][cal][pop]))
    return {
        "n_sorteios_ok": len(ok),
        "n_sorteios_com_no_valido": len(com_valido),
        "n_sorteios_sem_no_valido_excluidos": len(ok) - len(com_valido),
        "dp_mae_constante_validos": dp1(mae_v),
        "dp_fracao_valida_teste": dp1(fr_v),
        "dp_fracao_valida_teste_todos_os_sorteios": dp1(fr_todos),
        "media_mae_constante_validos": float(np.mean(mae_v)) if mae_v else None,
        "media_fracao_valida_teste": float(np.mean(fr_v)) if fr_v else None,
        "media_n_teste": float(np.mean([s["n_teste"] if "n_teste" in s else s["n_test"] for s in ok])) if ok else None,
        "inversoes_constante_gt_fspl": inv,
    }


def b2_compacto(chave: str, p: dict) -> list:
    return [compacta_sorteio(s) for s in p["celulas"][chave]["por_sorteio"]]


def consolidar():
    seeds = carregar_seeds()
    f2 = json.loads(F2_JSON.read_text(encoding="utf-8"))
    f3 = json.loads(F3_JSON.read_text(encoding="utf-8"))
    pb2 = json.loads(PARCIAL_B2.read_text(encoding="utf-8"))
    crit = json.loads(CRITERIO.read_text(encoding="utf-8"))
    celulas = [f"{c}_{q}" for c in orig.CIDADES for q in orig.QUADRANTES]

    b0 = {}
    for ch in celulas:
        arq = PARCIAL_DIR / f"{ch}.json"
        if not arq.exists():
            raise SystemExit(f"parcial ausente: {arq}")
        r = json.loads(arq.read_text(encoding="utf-8"))
        if r.get("status") != "ok":
            raise SystemExit(f"parcial {ch} com status {r.get('status')}")
        assert [s["split_seed"] for s in r["por_sorteio_compacto"]] == seeds
        b0[ch] = r

    data = agora()
    comando = " ".join(sys.argv)
    carimbo = {"script": str(HERE), "script_sha256": SCRIPT_SHA256,
               "script_original": str(ORIG_PATH), "script_original_sha256": ORIG_SHA256,
               "sementes": seeds, "sementes_fonte": str(F2_JSON),
               "data_utc": data, "comando_consolidacao": comando,
               "criterio": str(CRITERIO), "venv": sys.executable}

    por_sorteio = {
        **carimbo, "id": "R1_b0_por_sorteio", "buffer_km": 0.0, "grid_km": 10.0,
        "campos": "n_teste, fracao_valida_teste (=n_validos_teste/n_teste), mae_constante, "
                  "mae_fspl_a_contaminado, mae_fspl_b_validos (cada MAE por populacao: validos/sentinela/todos), "
                  "inversao_constante_gt_fspl",
        "celulas": {ch: {"tempo_total_s": b0[ch]["tempo_total_s"],
                          "n_nodes_total": b0[ch]["n_nodes_total"],
                          "sha256_tensor_manifest": b0[ch]["sha256_manifest"],
                          "sha256_tensor_recalculado": False,
                          "sorteios": b0[ch]["por_sorteio_compacto"]} for ch in celulas},
    }
    OUT_SORTEIO.write_text(json.dumps(por_sorteio, indent=2, ensure_ascii=False), encoding="utf-8")

    por_celula = {}
    for ch in celulas:
        s0 = stats_celula(b0[ch]["por_sorteio_compacto"])
        s2 = stats_celula(b2_compacto(ch, pb2))
        dp2_artefato = f2["por_celula"][ch]["constante_validos"]["dp_simples"]
        n2_artefato = f2["por_celula"][ch]["n_sorteios_ok"]
        por_celula[ch] = {
            "b0": s0,
            "b2": {**s2,
                   "dp_mae_constante_validos_artefato_2.1": dp2_artefato,
                   "n_sorteios_com_no_valido_artefato_2.1": n2_artefato,
                   "conferencia_dp_mae_recomputado_vs_artefato_absdiff":
                       abs(s2["dp_mae_constante_validos"] - dp2_artefato),
                   "origem": "dp_mae: copiado do artefato fase2/2.1 (por_celula.constante_validos.dp_simples); "
                             "dp_fracao_valida e inversoes: recomputados dos sorteios por celula do parcial "
                             "fase2/_v3_2.1_3.1_parcial_16x60rnd.json (o artefato 2.1 nao traz dp da fracao) "
                             "com a MESMA funcao usada em b=0"},
            "razao_dp_mae_b0_sobre_b2": razao(s0["dp_mae_constante_validos"], dp2_artefato),
            "razao_dp_fracao_b0_sobre_b2": razao(s0["dp_fracao_valida_teste"], s2["dp_fracao_valida_teste"]),
        }

    def media(buf, campo):
        v = [por_celula[ch][buf][campo] for ch in celulas]
        return float(np.mean(v))

    m_b0_err = media("b0", "dp_mae_constante_validos")
    m_b2_err = float(np.mean([f2["por_celula"][ch]["constante_validos"]["dp_simples"] for ch in celulas]))
    m_b2_err_recomp = media("b2", "dp_mae_constante_validos")
    m_b0_fr, m_b2_fr = media("b0", "dp_fracao_valida_teste"), media("b2", "dp_fracao_valida_teste")
    m_b0_frt, m_b2_frt = (media("b0", "dp_fracao_valida_teste_todos_os_sorteios"),
                          media("b2", "dp_fracao_valida_teste_todos_os_sorteios"))
    # Ratios of the decision criterion: mean over 16 cells of the b = 0 spread divided by the b = 2 spread.
    R_err, R_frac = razao(m_b0_err, m_b2_err), razao(m_b0_fr, m_b2_fr)

    def soma_inv(buf):
        out = {}
        for k in ("a_todos", "a_validos", "b_todos", "b_validos"):
            out[k] = int(sum(por_celula[ch][buf]["inversoes_constante_gt_fspl"][k] for ch in celulas))
        out["n_combinacoes_celula_x_sorteio"] = int(sum(por_celula[ch][buf]["n_sorteios_ok"] for ch in celulas))
        return out

    inv0, inv2 = soma_inv("b0"), soma_inv("b2")
    # Cross-check of the b = 2 inversion counts against the stored calibration record (3.1).
    r31 = f3["resumo"]
    conf_inv_b2 = {
        "artefato_3.1_a_validos": r31["constante_vence_fspl_calibracao_a_contaminada"]["validos"],
        "artefato_3.1_a_todos": r31["constante_vence_fspl_calibracao_a_contaminada"]["todos"],
        "artefato_3.1_b_validos": r31["constante_vence_fspl_calibracao_b_validos"]["validos"],
        "artefato_3.1_b_todos": r31["constante_vence_fspl_calibracao_b_validos"]["todos"],
        "artefato_3.1_n_combinacoes": r31["n_combinacoes_celula_x_sorteio"],
        "recomputado_do_parcial": inv2,
        "bate": (inv2["a_validos"] == r31["constante_vence_fspl_calibracao_a_contaminada"]["validos"]
                 and inv2["a_todos"] == r31["constante_vence_fspl_calibracao_a_contaminada"]["todos"]
                 and inv2["b_validos"] == r31["constante_vence_fspl_calibracao_b_validos"]["validos"]
                 and inv2["b_todos"] == r31["constante_vence_fspl_calibracao_b_validos"]["todos"]
                 and inv2["n_combinacoes_celula_x_sorteio"] == r31["n_combinacoes_celula_x_sorteio"]),
    }

    def n_sort_valido(buf):
        return {"n_sorteios_com_no_valido_total": int(sum(por_celula[ch][buf]["n_sorteios_com_no_valido"] for ch in celulas)),
                "n_sorteios_sem_no_valido_excluidos_total": int(sum(por_celula[ch][buf]["n_sorteios_sem_no_valido_excluidos"] for ch in celulas)),
                "de_total": int(sum(por_celula[ch][buf]["n_sorteios_ok"] for ch in celulas)),
                "por_celula_com_no_valido": {ch: por_celula[ch][buf]["n_sorteios_com_no_valido"] for ch in celulas}}

    resumo = {
        **carimbo, "id": "R1_b0_resumo", "buffer_km_b0": 0.0, "buffer_km_comparacao": 2.0,
        "estatisticas": "dp ddof=1 entre sorteios por celula; media simples das 16 celulas; sorteios sem no valido "
                        "no teste excluidos (conjunto principal, em MAE-validos e fracao valida) e contados",
        "por_celula": por_celula,
        "medias_16_celulas": {
            "dp_mae_constante_validos_b0": m_b0_err,
            "dp_mae_constante_validos_b2_artefato_2.1": m_b2_err,
            "dp_mae_constante_validos_b2_recomputado_do_parcial": m_b2_err_recomp,
            "dp_fracao_valida_teste_b0": m_b0_fr,
            "dp_fracao_valida_teste_b2": m_b2_fr,
            "dp_fracao_valida_teste_b0_todos_os_sorteios_suplementar": m_b0_frt,
            "dp_fracao_valida_teste_b2_todos_os_sorteios_suplementar": m_b2_frt,
        },
        "razoes_criterio": {
            "R_err": R_err, "R_frac": R_frac,
            "definicao": "R_err = media16(dp_b0)/media16(dp_b2) do MAE do constante nos validos (b2 = artefato 2.1); "
                         "R_frac idem para a fracao valida do teste",
            "R_frac_suplementar_todos_os_sorteios": razao(m_b0_frt, m_b2_frt),
        },
        "inversoes_constante_gt_fspl": {"b0": inv0, "b2": inv2, "conferencia_b2_vs_artefato_3.1": conf_inv_b2},
        "sorteios_com_no_valido": {"b0": n_sort_valido("b0"), "b2": n_sort_valido("b2")},
        "teto_criterio_texto": crit["leitura"],
        "interpretacao": "NAO feita neste artefato (fora do escopo do pipeline)",
    }
    OUT_RESUMO.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"gravado {OUT_SORTEIO} e {OUT_RESUMO}; R_err={R_err}, R_frac={R_frac}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validar", action="store_true")
    g.add_argument("--celula", default="")
    g.add_argument("--consolidar", action="store_true")
    a = ap.parse_args()
    if a.validar:
        validar()
    elif a.consolidar:
        consolidar()
    else:
        rodar_uma(a.celula)


if __name__ == "__main__":
    main()
