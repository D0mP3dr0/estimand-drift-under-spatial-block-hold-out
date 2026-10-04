#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Loop shared by checks R5 (variance of one split draw) and R6 (level of the constant predictor).

Reuses by import, unchanged and with digest check (aborts on mismatch), the partition and calibration
of analysis/v3_2.1_3.1_deriva_calibracao.py: split_espacial_3vias (g = 10 km, b = 2 km, fractions
0.70/0.15/0.15), the FSPL models, orig.mae, orig.carregar_tensor, orig.latlon_graus_para_metros and
orig.SpatialKFold. The tensor loading and the calibration (b) (offset p_tx_eff = median(rssi +
FSPL(dist)) over the valid training nodes) repeat the operations of orig.processar_celula, which does
not expose its arrays; the validation mode checks that the resulting MAE equals the stored record
field by field.

Blocks: the block rule of the partition is SpatialKFold._assign_groups(pos_km) with a 10 km grid
(id = grid_x * max_y + grid_y), independent of the seed. Each test node inherits its block id;
S_B = sum of the losses |pred - rssi| and M_B = number of scored nodes of the population in block B,
among the test nodes left after the buffer trimming (input of R5).
For R6, per draw: the MAE over the valid test nodes of the constants c in {-100, -110, -120, -130}
dBm, of the median of the valid training nodes (recomputed per draw) and, as a reference, of the
median of the whole training set (the constant of the error-drift test).

Usage:
  python analysis/v3_13_laco_comum.py --validar          # lins_Q1 and bauru_Q1, first 5 draws, against
                                                         # fase2/_v3_2.1_3.1_parcial_16x60rnd.json
                                                         # -> fase5/R5R6_validacao_laco.json
  python analysis/v3_13_laco_comum.py --celula bauru_Q1  # 60 draws -> fase5/_parcial_laco/<cell>.json
CPU only, no training. Seeds: the 60 split seeds stored in fase2/2.1_deriva_erro_baselines_16x60rnd.json.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
SCRIPT_SHA256 = hashlib.sha256(HERE.read_bytes()).hexdigest()
BASE = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
ORIG_PATH = BASE / "scripts" / "v3_2.1_3.1_deriva_calibracao.py"
ORIG_SHA256 = "9b9bcf91bab51621107a3d7978964d46dd16c5ec1a76a3789842ae1bce77f3bc"
V3 = BASE / "_v3_2026-09-25"
F2_JSON = V3 / "fase2" / "2.1_deriva_erro_baselines_16x60rnd.json"
PARCIAL_B2 = V3 / "fase2" / "_v3_2.1_3.1_parcial_16x60rnd.json"
OUT = V3 / "fase5"
PARCIAL_DIR = OUT / "_parcial_laco"
OUT_VALID = OUT / "R5R6_validacao_laco.json"
CONSTANTES_DBM = (-100.0, -110.0, -120.0, -130.0)
POPS = ("validos", "todos")

_orig_sha_real = hashlib.sha256(ORIG_PATH.read_bytes()).hexdigest()
if _orig_sha_real != ORIG_SHA256:
    raise SystemExit(f"ABORTA: sha do script original diverge ({_orig_sha_real})")

_spec = importlib.util.spec_from_file_location("deriva_orig", ORIG_PATH)
orig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(orig)  # its main() is guarded by __name__
assert orig.GRID_KM == 10.0 and orig.BUFFER_KM == 2.0 and orig.FRACS == (0.70, 0.15, 0.15) \
    and orig.FREQ_MHZ == 900.0, "configuracao g10b2 do original alterada"

import torch  # noqa: E402

torch.set_num_threads(1)


def agora() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[{agora()}] {msg}", flush=True)


def carregar_seeds() -> list:
    """The 60 distinct split seeds of the error-drift record, checked against its b = 2 partial record."""
    a = json.loads(F2_JSON.read_text(encoding="utf-8"))
    seeds = a["nota_divergencia_seeds"]["seeds_usados_nesta_rodada"]
    p = json.loads(PARCIAL_B2.read_text(encoding="utf-8"))
    assert len(seeds) == 60 and len(set(seeds)) == 60, "esperados 60 sorteios distintos"
    assert seeds == p["seeds_usados"], "sementes do JSON final != parcial de b=2"
    return [int(s) for s in seeds]


def carregar_celula(chave: str) -> dict:
    """Load one cell with the same reading sequence as orig.processar_celula; adds sentinel flags,
    block ids and the occupied and bounding-rectangle block counts."""
    cidade, quad = chave.split("_")
    basename = f"transfer_dataset_{cidade}_v19_{quad}_enriched_cftudo.pt"
    tensor_path = orig.TENSOR_DIR / basename
    rf, modo = orig.carregar_tensor(tensor_path)
    ty = torch.as_tensor(rf["terrain"].y).float().clone().numpy()
    dist_all = torch.as_tensor(rf["terrain"].dist_nearest_m).float().clone().numpy()
    pos_deg = torch.as_tensor(rf["terrain"].pos).float().clone()
    del rf
    pos_m = orig.latlon_graus_para_metros(pos_deg)
    rssi = ty[:, 3].astype(np.float64)
    pl = ty[:, 0].astype(np.float64)
    sent = pl >= orig.PL_TARGET_MAX_VALID
    # partition blocks: same rule as SpatialKFold._assign_groups on the 10 km grid
    skf = orig.SpatialKFold(n_splits=3, buffer_km=orig.BUFFER_KM, grid_size_km=orig.GRID_KM, random_state=0)
    pos_km = pos_m / 1000.0
    gid = skf._assign_groups(pos_km)
    gx = (pos_km[:, 0] / orig.GRID_KM).astype(int)
    gy = (pos_km[:, 1] / orig.GRID_KM).astype(int)
    n_ocupados = int(np.unique(gid).size)
    n_bbox = int((gx.max() + 1) * (gy.max() + 1))
    return {"chave": chave, "modo_carga": modo, "tensor_path": str(tensor_path),
            "sha256_manifest": orig.carregar_manifest_sha(basename),
            "n_total": int(rssi.shape[0]), "pos_m": pos_m, "rssi": rssi, "dist": dist_all,
            "sent": sent, "gid": gid, "N_blocos_ocupados": n_ocupados, "N_blocos_bbox": n_bbox}


def somas_por_bloco(inv: np.ndarray, nb: int, ub: np.ndarray, loss: np.ndarray, mask: np.ndarray) -> dict:
    """Per-block loss sum S_B and scored-node count M_B of the masked test nodes; blocks with M_B = 0 dropped."""
    M = np.bincount(inv[mask], minlength=nb)
    S = np.bincount(inv[mask], weights=loss[mask], minlength=nb)
    ok = M > 0
    return {"blocos": [int(x) for x in ub[ok]], "S": [float(x) for x in S[ok]], "M": [int(x) for x in M[ok]]}


def sorteio(c: dict, seed: int) -> dict:
    """One split draw: block sums for R5 and constant-predictor MAEs for R6."""
    parts = orig.split_espacial_3vias(c["pos_m"], orig.GRID_KM, orig.BUFFER_KM, orig.FRACS, seed)
    tr, te = parts["train"], parts["test"]
    if tr.size == 0 or te.size == 0:
        return {"split_seed": seed, "status": "erro_particao_vazia", "n_train": int(tr.size), "n_test": int(te.size)}
    rssi, dist, sent = c["rssi"], c["dist"], c["sent"]
    rssi_tr, rssi_te = rssi[tr], rssi[te]
    dist_tr, dist_te = dist[tr], dist[te]
    sent_tr, sent_te = sent[tr], sent[te]
    val_te = ~sent_te
    pop_te = {"validos": val_te, "todos": np.ones_like(sent_te, dtype=bool)}

    # predictors, with the same operations as orig.processar_celula
    constante = float(np.median(rssi_tr))
    fspl = orig.MODELOS["fspl"]
    mask_tr_valido = ~sent_tr
    if mask_tr_valido.sum() > 0:
        pl_pred_tr_v = fspl(dist_tr[mask_tr_valido])
        p_tx_b = float(np.median(rssi_tr[mask_tr_valido] + pl_pred_tr_v))
    else:
        p_tx_b = None
    loss = {"constante": np.abs(constante - rssi_te)}
    if p_tx_b is not None:
        rssi_bl_te = p_tx_b - fspl(dist_te)
        loss["fspl_b"] = np.abs(rssi_bl_te - rssi_te)

    ub, inv = np.unique(c["gid"][te], return_inverse=True)
    nb = int(ub.size)
    r5 = {}
    for pred, lv in loss.items():
        r5[pred] = {}
        for pop, m in pop_te.items():
            blk = somas_por_bloco(inv, nb, ub, lv, m)
            # MAE through orig.mae, identical to the stored record
            if pred == "constante":
                mae_o = orig.mae(constante, rssi_te[m])
            else:
                mae_o = orig.mae(rssi_bl_te[m], rssi_te[m])
            r5[pred][pop] = {**blk, "mae_orig": mae_o}

    # R6: MAE of each constant over the valid test nodes
    rv = rssi_te[val_te]
    r6 = {}
    for cval in CONSTANTES_DBM:
        r6[f"c{int(cval)}"] = {"c": cval, "mae_validos": orig.mae(cval, rv)}
    if mask_tr_valido.sum() > 0:
        cmv = float(np.median(rssi_tr[mask_tr_valido]))
        r6["mediana_validos_treino"] = {"c": cmv, "mae_validos": orig.mae(cmv, rv)}
    else:
        r6["mediana_validos_treino"] = {"c": None, "mae_validos": None}
    r6["mediana_treino_inteiro_ref"] = {"c": constante, "mae_validos": orig.mae(constante, rv)}

    return {"split_seed": seed, "status": "ok", "n_train": int(tr.size), "n_test": int(te.size),
            "n_pop_teste": {p: int(m.sum()) for p, m in pop_te.items()},
            "n_blocos_teste_com_no": nb, "constante_treino_mediana_rssi": constante,
            "offset_p_tx_eff_b_fspl": p_tx_b, "R5": r5, "R6": r6}


def rodar_celula(chave: str, seeds: list) -> dict:
    t0 = time.perf_counter()
    c = carregar_celula(chave)
    rec = {"celula": chave, "sha256_manifest": c["sha256_manifest"], "modo_carga": c["modo_carga"],
           "n_nodes_total": c["n_total"], "N_blocos_ocupados": c["N_blocos_ocupados"],
           "N_blocos_bbox": c["N_blocos_bbox"], "tempo_load_s": time.perf_counter() - t0}
    rec["por_sorteio"] = [sorteio(c, s) for s in seeds]
    rec["status"] = "ok"
    rec["tempo_total_s"] = time.perf_counter() - t0
    return rec


def deep_equal_nums(a, b, path, out):
    """Record an exact-equality comparison of two numbers (None-aware) into the accumulator `out`."""
    if a is None or b is None:
        if a != b:
            out["divergentes"].append((path, a, b))
        return
    out["n"] += 1
    d = abs(float(a) - float(b))
    out["max_abs"] = max(out["max_abs"], d)
    if a != b:
        out["divergentes"].append((path, a, b))


def validar():
    """Validation mode; exits with status 2 unless every comparison reproduces the stored record."""
    seeds = carregar_seeds()[:5]
    p = json.loads(PARCIAL_B2.read_text(encoding="utf-8"))
    saida_cel = {}
    ok_global = True
    for chave in ("lins_Q1", "bauru_Q1"):
        log(f"VALIDACAO {chave}, sorteios {seeds}")
        c = carregar_celula(chave)
        ref = p["celulas"][chave]["por_sorteio"][:5]
        ex = {"n": 0, "max_abs": 0.0, "divergentes": []}  # exact equality of the orig.mae values
        soma = {"n": 0, "max_abs": 0.0}  # Err = sum(S_B)/sum(M_B) versus the stored MAE
        tabela = []
        for seed, r in zip(seeds, ref):
            s = sorteio(c, seed)
            assert s["split_seed"] == r["split_seed"] and s["status"] == "ok"
            assert s["n_test"] == r["n_test"] and s["n_train"] == r["n_train"]
            assert s["n_pop_teste"]["validos"] == r["n_pop_teste"]["validos"]
            assert s["n_pop_teste"]["todos"] == r["n_pop_teste"]["todos"]
            gravado = {
                ("constante", "validos"): r["mae_constante_teste"]["validos"],
                ("constante", "todos"): r["mae_constante_teste"]["todos"],
                ("fspl_b", "validos"): r["mae_modelo_b_validos_teste"]["fspl"]["validos"],
                ("fspl_b", "todos"): r["mae_modelo_b_validos_teste"]["fspl"]["todos"],
            }
            lin = {"split_seed": seed}
            for (pred, pop), g in gravado.items():
                x = s["R5"][pred][pop]
                deep_equal_nums(x["mae_orig"], g, f"{chave}/{seed}/{pred}/{pop}/mae_orig", ex)
                err_soma = sum(x["S"]) / sum(x["M"])
                soma["n"] += 1
                soma["max_abs"] = max(soma["max_abs"], abs(err_soma - g))
                lin[f"{pred}_{pop}"] = {"gravado": g, "laco_mae_orig": x["mae_orig"], "laco_soma_S_sobre_M": err_soma,
                                         "k_blocos": len(x["M"]), "M": int(sum(x["M"]))}
            # R6: c = -110 must equal the stored constant MAE (the training median is -110 dBm)
            deep_equal_nums(s["R6"]["c-110"]["mae_validos"], r["mae_constante_teste"]["validos"],
                            f"{chave}/{seed}/R6_c-110", ex)
            lin["constante_treino_mediana_rssi"] = s["constante_treino_mediana_rssi"]
            lin["R6_c-110_mae_validos"] = s["R6"]["c-110"]["mae_validos"]
            tabela.append(lin)
        reproduz = (len(ex["divergentes"]) == 0 and soma["max_abs"] < 1e-9)
        ok_global &= reproduz
        saida_cel[chave] = {"reproduz": bool(reproduz), "n_numeros_comparados_exato": ex["n"],
                            "max_abs_diferenca_mae_orig": ex["max_abs"], "divergentes": ex["divergentes"][:20],
                            "max_abs_diferenca_Err_soma_vs_gravado": soma["max_abs"],
                            "tolerancia_soma": 1e-9, "linhas": tabela}
    saida = {"id": "R5R6_validacao_laco", "sementes": seeds, "referencia": str(PARCIAL_B2),
             "referencia_sha256": orig.sha256_file(PARCIAL_B2),
             "criterio_de_aceite": "MAE do constante e do FSPL (b) nos validos e em todos os nos: igualdade exata "
                                   "(==) de orig.mae contra o parcial gravado; Err = sum S_B / sum M_B dentro de 1e-9; "
                                   "R6 c=-110 igual ao MAE do constante gravado",
             "reproduz": bool(ok_global), "celulas": saida_cel,
             "script_sha256": SCRIPT_SHA256, "script_original_sha256": ORIG_SHA256,
             "comando": " ".join(sys.argv), "data_utc": agora(), "venv": sys.executable}
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_VALID.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"validacao reproduz={ok_global} -> {OUT_VALID}")
    sys.exit(0 if ok_global else 2)


def rodar_uma(chave: str):
    """Run one cell and write its partial record; skipped if a partial with status ok exists."""
    PARCIAL_DIR.mkdir(parents=True, exist_ok=True)
    arq = PARCIAL_DIR / f"{chave}.json"
    if arq.exists() and json.loads(arq.read_text(encoding="utf-8")).get("status") == "ok":
        log(f"{chave}: parcial ok, pulando")
        return
    seeds = carregar_seeds()
    log(f"=== {chave}: {len(seeds)} sorteios ===")
    try:
        rec = rodar_celula(chave, seeds)
    except Exception as e:  # noqa: BLE001
        rec = {"celula": chave, "status": "erro_excecao", "erro": repr(e), "traceback": traceback.format_exc()}
    rec["script_sha256"] = SCRIPT_SHA256
    rec["data_utc"] = agora()
    arq.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"{chave}: status={rec['status']} -> {arq}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validar", action="store_true")
    g.add_argument("--celula", default="")
    a = ap.parse_args()
    if a.validar:
        validar()
    else:
        rodar_uma(a.celula)


if __name__ == "__main__":
    main()
