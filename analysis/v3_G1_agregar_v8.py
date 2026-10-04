#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregation of batch G1: GNN and MLP trained at block side g = 10 km, buffer b = 2 km.

Batch G1 asks, in the main geometry of the study and in two cities (Bauru and
Campinas), whether the error of trained models on valid test nodes drifts across
split draws by more than training non-determinism and seed-to-seed variation, and
whether GNN and MLP stay in parity on sentinel nodes. Design and reading rules are
fixed before any run in criteria/criterio_G1_modelos_g10.json and its amendments
(criterio_G1_adendo1.json ...):
  block 1: Bauru Q1 and Campinas Q1, training seed 42, the first 20 split draws of
           the 60-draw list of the drift experiment (2.1), GNN and MLP (80 runs);
  block 2: the same cells, training seeds 43 and 44, the first 5 draws (40 runs;
           with block 1, 3 seeds x 5 draws per cell and model);
  block 3: Bauru Q3 and Campinas Q3, seed 42, the same 20 draws (80 runs).

For every run the MAE per node population (valid, sentinel, all; received power
from target/prediction channel 3, path loss from channel 0) is recomputed from the
saved test predictions (.npz). Before a run is used it is checked against the plan,
the data manifest, the per-draw file of the drift experiment, the sidecar of the
training wrapper and the configuration of batch A4; any violation aborts with exit
code 4 and nothing is written. MAE and RMSE recomputed from the .npz must match the
run JSON within 10 % and 3 % (relative); the per-run differences are recorded.

Per cell and model the output gives the SD across draws of the valid-node MAE and
its ratio to the comparator (0.132 dB, the largest difference between two GNN
repetitions; for the Q1 cells max(0.132 dB, pooled SD across seeds of block 2)),
the one-factor decomposition of block 2 (seeds nested in draws), the correlation
with the constant-predictor MAE of the same draws, and the median |GNN - MLP| on
sentinel nodes against 0.117 dB, plus a mechanical count of the cells meeting each
reading rule. The script computes; it does not interpret.

A block is aggregated only when complete; otherwise it is refused (exit code 2).
With --bloco3-truncado, block 3 may be aggregated on the contiguous prefix of
complete draws once the batch has ended, if each cell keeps >= 10 draws with
valid nodes.

Partial runs. When the trainer stops before training because a partition has no
antenna-to-terrain edge, it writes a partial run JSON and no .npz. Such a run is
accepted only after its own checks (status with rc != 0 and the matching flag, run
log naming the degenerate partition, partial run JSON with g, b, seeds and dataset
name and size as planned): flag sem_validos = no valid test node (criterion
amendment 3; the draw stays in the count and out of the SD); flag nao_treinavel =
degenerate validation or training partition (amendment 4; the whole draw is
excluded from the cell for both models, so that GNN and MLP are compared on the
same draws). The batch was resumed with a separate launcher whose provenance file
is also checked; block 3 is aggregated as complete only when the batch status is
final, and truncation is refused while a resume may still be running.

Inputs (under the root folder, default RAIZ_PADRAO): gpu/G1/plano_G1.json,
gpu/G1/lote_G1_status.json, gpu/G1/proveniencia_scripts_G1*.json, one folder per
run gpu/G1/<run_label>/ (run_*.json, predicoes_*.npz, shim_g10_*.json), the A4 runs
in gpu/A4/, the criterion files, fase2/_v3_2.1_3.1_parcial_16x60rnd.json,
fase2/2.1_deriva_erro_baselines_16x60rnd.json and manifest_mathematics_v4.jsonl.
Output: gpu/G1/agregado_G1_v8_bloco<N>.json, written atomically (this repository
uses block 3 from this script). Deterministic (no random numbers).

Usage: python analysis/v3_G1_agregar_v8.py [--bloco 1|2|3] [--bloco3-truncado] [--so-conferir] [--raiz-teste DIR]
"""
import argparse
import glob
import hashlib
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

BASE_PADRAO = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
RAIZ_PADRAO = BASE_PADRAO / "_v3_2026-09-25"
SHA_ORIGINAL_V1 = "88d444d4c480b7758ccd197f6936d5835a78c71863d99933fe8cbcfcd2419d2b"
SHA_V2 = "d94fac29dcf6c694cab9bec55c4bdd9e2701fe5baf2675cbfe0aa77a5c155824"
SHA_V3 = "5102299e6e5b0b3d25009693ad04c3dc0d9341833252b406841e2e894bae0e16"
SHA_V4 = "4263076007a28aa76c5e51fa73c35c513b34131c8019f2b5089981a57e916b14"
SHA_V5 = "50610e19f9ea1314c899b86b21eeefb569b74e64093f28936ee1c03a6aa0e465"
SHA_V6 = "6f852e2d56cfd61a5a4f687132a45b35369efcc06f72dccd1d7f77cf0bd8bb96"
SHA_V7 = "cc703246f65d6d4b4a23ded13b957b9d27eda1d359b0621e04a9eff8425d0b72"
TRECHO_LOG_DEGENERADA = "PARTICAO DEGENERADA"  # logged by the trainer when a partition has no antenna-to-terrain edge
# relative tolerances for MAE / RMSE over all test nodes, recomputed from the .npz vs the run JSON; both must hold.
# The all-node MAE is dominated by sentinel nodes with small absolute error, so its relative difference is
# unstable and its tolerance is loose; RMSE is the metric that separates GNN from MLP.
TOL_AMARRA_MAE_REL = 0.10
TOL_AMARRA_RMSE_REL = 0.03
SENTINELA_PL = 299.0  # path-loss target >= 299 marks a sentinel node (no valid target)
RUIDO_REPETICAO_DB = 0.132  # largest difference between two GNN repetitions (A2c); an absolute difference, not an SD
RUIDO_SENTINELA_DB = 0.117  # largest difference between repetitions on sentinel nodes (A2c)
FATOR_CONDICAO1 = 3.0  # condition 1: SD across draws >= 3 x comparator
FATOR_DELIMITA = 2.0  # delimiting rule: SD across draws < 2 x comparator
K_MIN_BLOCO3 = 10  # minimum draws with valid nodes per cell for a truncated block 3
N_SEMENTES_B2 = 3  # training seeds per draw in block 2 (42 from block 1, plus 43 and 44)
GRID_KM, BUFFER_KM = 10.0, 2.0
POPS = ("mae_rssi_validos_db", "mae_rssi_sentinela_db", "mae_rssi_todos_db", "mae_pl_validos_db")
BASELINES = ("fspl", "hata_rural", "cost231_sub")
CELULAS_POR_BLOCO = {1: ("bauru_Q1", "campinas_Q1"), 2: ("bauru_Q1", "campinas_Q1"), 3: ("bauru_Q3", "campinas_Q3")}
# config keys ignored when comparing a run with the A4 reference (geometry, seeds, labels, paths)
CONFIG_IGNORADAS_A4 = ("grid_km", "seed", "split_seed", "run_label", "evid_dir", "graph_dir", "graph_file",
                       "rf_data_file", "base_dir")
TOL_FLOAT = 0.0  # exact equality for g and b (10.0 and 2.0 are exactly representable)


class Aborta(Exception):
    """A check failed: exit code 4, nothing is written."""
    pass


class Incompleto(Exception):
    """Block incomplete or refused: exit code 2."""
    pass


class Caminhos:
    """Namespace for the input and output paths, filled by definir_caminhos()."""
    pass


P = Caminhos()


def definir_caminhos(raiz: Path) -> None:
    """Set every path relative to the root folder (RAIZ_PADRAO, or --raiz-teste)."""
    P.RAIZ = raiz
    P.BASE = raiz.parent
    P.G1 = raiz / "gpu" / "G1"
    P.A4 = raiz / "gpu" / "A4"
    P.CRIT = raiz / "criterios" / "criterio_G1_modelos_g10.json"
    P.ADENDO = raiz / "criterios" / "criterio_G1_adendo1.json"
    P.ADENDO2 = raiz / "criterios" / "criterio_G1_adendo2.json"
    P.ADENDO3 = raiz / "criterios" / "criterio_G1_adendo3.json"
    P.ADENDO4 = raiz / "criterios" / "criterio_G1_adendo4.json"
    P.PLANO = P.G1 / "plano_G1.json"
    P.STATUS = P.G1 / "lote_G1_status.json"
    P.PROV = P.G1 / "proveniencia_scripts_G1.json"
    P.PROV_RET = P.G1 / "proveniencia_scripts_G1_retomada.json"
    P.PARCIAL_2_1 = raiz / "fase2" / "_v3_2.1_3.1_parcial_16x60rnd.json"
    P.SORTEIOS_2_1 = raiz / "fase2" / "2.1_deriva_erro_baselines_16x60rnd.json"
    P.MANIFEST_V4 = P.BASE / "manifest_mathematics_v4.jsonl"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest_cftudo() -> dict:
    """Map tensor file name -> SHA-256 from the data manifest (group tensores_cftudo)."""
    out = {}
    with open(P.MANIFEST_V4, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                e = json.loads(ln)
                if e.get("grupo") == "tensores_cftudo":
                    out[Path(e.get("caminho", "")).name] = e.get("sha256")
    return out


def manifest_tamanhos() -> dict:
    """Map tensor file name -> size in bytes from the data manifest (group tensores_cftudo)."""
    out = {}
    with open(P.MANIFEST_V4, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                e = json.loads(ln)
                if e.get("grupo") == "tensores_cftudo":
                    out[Path(e.get("caminho", "")).name] = e.get("tamanho_bytes")
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.median(xs)) if xs else None


def dp(xs):
    """Sample standard deviation (ddof = 1) ignoring None; None with fewer than 2 values."""
    xs = [x for x in xs if x is not None]
    return float(statistics.stdev(xs)) if len(xs) >= 2 else None


def razao(a, b):
    return (a / b) if (a is not None and b not in (None, 0)) else None


def pearson(x, y):
    if len(x) < 3:
        return None
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    """Spearman correlation as the Pearson correlation of ordinal ranks (no tie averaging)."""
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
    return pearson(rx, ry)


def _finito(nome, arr):
    if not np.isfinite(arr).all():
        raise Aborta(f"valor nao finito no .npz: {nome}")


def mae_pop(z) -> dict:
    """Per-population errors of one prediction file (.npz with target, pred, idx_global [, sentinela]).

    Channel 3 = received power (RSSI, dB), channel 0 = path loss (dB); sentinel node: target[:, 0] >= SENTINELA_PL.
    Aborts on unexpected shapes, non-finite values or a stored sentinel mask that disagrees with the target.
    """
    tgt = z["target"].astype(np.float64)
    pred = z["pred"].astype(np.float64)
    if tgt.ndim != 2 or pred.shape != tgt.shape or tgt.shape[1] < 4 or tgt.shape[0] != len(z["idx_global"]):
        raise Aborta("formato do .npz inesperado (target/pred/idx_global)")
    _finito("target[:,0]", tgt[:, 0])
    _finito("target[:,3]", tgt[:, 3])
    _finito("pred[:,3]", pred[:, 3])
    sent = tgt[:, 0] >= SENTINELA_PL
    if "sentinela" in getattr(z, "files", z) and not np.array_equal(sent, z["sentinela"].astype(bool)):
        raise Aborta("campo sentinela do .npz nao bate com target[:,0] >= 299")
    val = ~sent
    if val.any():
        _finito("pred[:,0] nos validos", pred[val, 0])
    e3s = pred[:, 3] - tgt[:, 3]
    e3 = np.abs(e3s)
    e0 = np.abs(tgt[:, 0] - pred[:, 0])
    return {
        "n": int(tgt.shape[0]), "n_validos": int(val.sum()), "n_sentinela": int(sent.sum()),
        "mae_rssi_validos_db": float(e3[val].mean()) if val.any() else None,
        "mae_rssi_sentinela_db": float(e3[sent].mean()) if sent.any() else None,
        "mae_rssi_todos_db": float(e3.mean()),
        "rmse_rssi_todos_db": float(np.sqrt((e3s ** 2).mean())),
        "mae_pl_validos_db": float(e0[val].mean()) if val.any() else None,
        "idx_sha256": hash_idx(z["idx_global"]),
    }


def hash_idx(idx) -> str:
    """SHA-256 of the sorted global node indices (int64); identifies a test partition."""
    return hashlib.sha256(np.sort(np.asarray(idx).astype(np.int64)).tobytes()).hexdigest()


def carregar_2_1() -> dict:
    """(cell, split seed) -> test-node counts and constant-predictor MAE, from the per-draw file of the drift experiment."""
    with open(P.PARCIAL_2_1, "r", encoding="utf-8") as f:
        p = json.load(f)
    out = {}
    for cel, d in p["celulas"].items():
        for s in d.get("por_sorteio", []):
            if s.get("status") == "ok":
                out[(cel, int(s["split_seed"]))] = {
                    "n_test": s["n_test"], "validos": s["n_pop_teste"]["validos"], "todos": s["n_pop_teste"]["todos"],
                    "mae_constante_validos": s["mae_constante_teste"]["validos"]}
    return out


def decomposicao_um_fator(M: np.ndarray) -> dict:
    """One-factor variance decomposition with seeds nested in draws; M is n draws (rows) x k = 3 seeds (columns).

    Mean squares: MS_within = SS_within / (n (k - 1)), MS_between = SS_between / (n - 1) (keys qm_*);
    sigma2_seed = MS_within, sigma2_draw = max(0, (MS_between - MS_within) / k).
    Condition 2 holds when sigma2_draw >= sigma2_seed.
    """
    n, k = M.shape
    if k != N_SEMENTES_B2:
        raise Aborta(f"decomposicao exige {N_SEMENTES_B2} sementes por sorteio, recebeu {k}")
    if n < 2:
        return None
    gm = float(M.mean())
    mu_i = M.mean(axis=1)
    sq_entre = float(k * ((mu_i - gm) ** 2).sum())
    sq_dentro = float(((M - mu_i[:, None]) ** 2).sum())
    qm_entre = sq_entre / (n - 1)
    qm_dentro = sq_dentro / (n * (k - 1))
    bruto = (qm_entre - qm_dentro) / k
    s2_sem = qm_dentro
    s2_sor = max(0.0, bruto)
    return {"n_sorteios": int(n), "n_sementes": int(k), "qm_entre": qm_entre, "qm_dentro": qm_dentro,
            "sigma2_semente": s2_sem, "sigma2_sorteio_bruto": bruto, "sigma2_sorteio": s2_sor,
            "dp_entre_sementes_pooled_db": float(np.sqrt(qm_dentro)),
            "condicao2_sigma2_sorteio_ge_sigma2_semente": bool(s2_sor >= s2_sem)}


def avaliar_condicoes_modelo(dp_sorteios, comparador):
    """Ratio of the SD across draws to the comparator and the >= 3x, < 2x and in-between flags (None if not computable)."""
    r = razao(dp_sorteios, comparador)
    return {"dp_entre_sorteios_db": dp_sorteios, "comparador_db": comparador, "razao_dp_sobre_comparador": r,
            "dp_ge_3x_comparador": (None if r is None else bool(r >= FATOR_CONDICAO1)),
            "dp_lt_2x_comparador": (None if r is None else bool(r < FATOR_DELIMITA)),
            "dp_entre_2x_e_3x_comparador": (None if r is None else bool(FATOR_DELIMITA <= r < FATOR_CONDICAO1))}


def sinalizar_celula(por_modelo: dict) -> dict:
    """Cell flags from the GNN and MLP readings: condition 1 met by both, both below 2x, intermediate band, undetermined."""
    g, m = por_modelo["gnn"], por_modelo["mlp"]
    indet = any(x["razao_dp_sobre_comparador"] is None for x in (g, m))
    cond1 = bool(not indet and g["dp_ge_3x_comparador"] and m["dp_ge_3x_comparador"])
    delim = bool(not indet and g["dp_lt_2x_comparador"] and m["dp_lt_2x_comparador"])
    return {"condicao1_gnn_e_mlp": cond1, "ambos_lt_2x": delim,
            "faixa_intermediaria": bool(not indet and not cond1 and not delim),
            "algum_modelo_entre_2x_e_3x": bool(g["dp_entre_2x_e_3x_comparador"] or m["dp_entre_2x_e_3x_comparador"]),
            "indeterminada_dp_nao_calculavel": indet}


def contar_ramos(celulas: dict, celulas_q1: list) -> dict:
    """Mechanical count of the cells meeting each reading rule of the criterion; no interpretation.

    Threshold t: 3 of 4 cells, or 2 of 2 when only the Q1 cells enter. Rule "reforca": condition 1 in >= t cells and
    condition 2 in every Q1 cell. Rule "delimita": both models below 2x in >= t cells. Any other case is the partial reading.
    """
    ind = [c for c, d in celulas.items() if d.get("indeterminada_dp_nao_calculavel")]
    if ind:
        return {"indeterminada": True, "celulas_indeterminadas": ind, "n_celulas": len(celulas),
                "satisfaz_regra_reforca": None, "satisfaz_regra_delimita": None, "satisfaz_regra_delimita_parcial": None,
                "nota": "contagem de ramos RECUSADA: dp, razao ou condicao 2 nao calculavel em celula(s) acima"}
    n = len(celulas)
    t = 3 if n == 4 else n
    n1 = [c for c, d in celulas.items() if d["condicao1_gnn_e_mlp"]]
    nd = [c for c, d in celulas.items() if d["ambos_lt_2x"]]
    faixa = [c for c, d in celulas.items() if d["faixa_intermediaria"]]
    f23 = [c for c, d in celulas.items() if d["algum_modelo_entre_2x_e_3x"]]
    c2 = {c: celulas[c].get("condicao2_gnn_e_mlp") for c in celulas_q1}
    c2_todas = bool(all(v is True for v in c2.values())) if c2 else False
    reforca = bool(len(n1) >= t and c2_todas)
    delimita = bool(len(nd) >= t)
    c1_sem_c2 = bool(len(n1) >= t and not c2_todas)
    impedem = bool(len(n1) < t and len(nd) < t and len(faixa) > 0)
    parcial_literal = bool(c1_sem_c2 or impedem)
    residual = bool(not reforca and not delimita and not parcial_literal)
    parcial = bool(parcial_literal or residual)
    par = [c for c, d in celulas.items() if d.get("paridade_le_0_117")]
    return {
        "n_celulas": n, "limiar_celulas": t,
        "celulas_condicao1_gnn_e_mlp": n1, "celulas_ambos_lt_2x": nd, "celulas_faixa_intermediaria": faixa,
        "celulas_algum_modelo_entre_2x_e_3x": f23,
        "condicao2_por_celula_q1": c2, "condicao2_nas_celulas_q1_todas": c2_todas,
        "satisfaz_regra_reforca": reforca, "satisfaz_regra_delimita": delimita,
        "satisfaz_regra_delimita_parcial": parcial,
        "delimita_parcial_por": {"condicao1_atendida_condicao2_nao": c1_sem_c2,
                                 "celulas_entre_2x_e_3x_impedem_limiar": impedem,
                                 "residual_nao_prevista_literalmente_no_adendo": residual},
        "celulas_paridade_sentinela_le_0_117": par, "paridade_em_todas_as_celulas": bool(len(par) == n),
        "nota": "aplicacao aritmetica das regras do criterio/adendo; leitura e do rigor, nao deste script"}


def conferir_scripts_do_lote(modo_teste: bool) -> dict:
    """Check the SHA-256 of every script listed in proveniencia_scripts_G1.json, in the resume-launcher
    provenance file when present, and, outside test mode, of the first aggregator version
    (scripts/v3_G1_agregar.py, SHA_ORIGINAL_V1); abort on any mismatch.
    """
    with open(P.PROV, "r", encoding="utf-8") as f:
        prov = json.load(f)
    falhas, linhas = [], []
    for e in prov["arquivos"]:
        p = P.BASE / e["caminho"]
        if not p.exists():
            falhas.append(f"{e['caminho']}: ausente")
            continue
        obs = sha256(p)
        linhas.append({"caminho": e["caminho"], "sha256_registrado": e["sha256"], "sha256_observado": obs,
                       "igual": bool(obs == e["sha256"])})
        if obs != e["sha256"]:
            falhas.append(f"{e['caminho']}: sha256 {obs} != registrado {e['sha256']}")
    caminhos_reg = {e["caminho"] for e in prov["arquivos"]}
    for obrig in ("_v3_2026-09-25/gpu/G1/train_v3_g10.py", "_v3_2026-09-25/gpu/G1/rodar_lote_G1.py"):
        if obrig not in caminhos_reg:
            falhas.append(f"{obrig}: nao consta de proveniencia_scripts_G1.json")
    if not modo_teste:
        orig = P.BASE / "scripts" / "v3_G1_agregar.py"
        if not orig.exists() or sha256(orig) != SHA_ORIGINAL_V1:
            falhas.append("scripts/v3_G1_agregar.py (original v1) ausente ou editado (sha != cabecalho)")
    retomada = None
    if P.PROV_RET.exists():
        with open(P.PROV_RET, "r", encoding="utf-8") as f:
            provr = json.load(f)
        linhas_r = []
        for e in provr["arquivos"]:
            p = P.BASE / e["caminho"]
            if not p.exists():
                falhas.append(f"retomada: {e['caminho']}: ausente")
                continue
            obs = sha256(p)
            linhas_r.append({"caminho": e["caminho"], "sha256_registrado": e["sha256"], "sha256_observado": obs, "igual": bool(obs == e["sha256"])})
            if obs != e["sha256"]:
                falhas.append(f"retomada: {e['caminho']}: sha256 {obs} != registrado {e['sha256']}")
        if not any(str(e["caminho"]).endswith("rodar_lote_G1_retomada.py") for e in provr["arquivos"]):
            falhas.append("retomada: rodar_lote_G1_retomada.py nao consta de proveniencia_scripts_G1_retomada.json")
        retomada = {"fonte": str(P.PROV_RET), "arquivos": linhas_r}
    if falhas:
        raise Aborta("proveniencia dos scripts do lote: " + "; ".join(falhas))
    return {"fonte": str(P.PROV), "arquivos": linhas, "retomada": retomada}


def conferir_criterios() -> tuple:
    """Check the chain of criterion files: expected ids, and each amendment cites the SHA-256 of the file it amends."""
    with open(P.CRIT, "r", encoding="utf-8") as f:
        crit = json.load(f)
    with open(P.ADENDO, "r", encoding="utf-8") as f:
        adendo = json.load(f)
    if crit.get("id") != "G1_modelos_g10" or adendo.get("id") != "G1_modelos_g10_adendo1":
        raise Aborta("id do criterio/adendo inesperado")
    m = re.search(r"sha256 ([0-9a-f]{64})", adendo.get("emenda", ""))
    sc, sa = sha256(P.CRIT), sha256(P.ADENDO)
    if not m or m.group(1) != sc:
        raise Aborta(f"sha256 do criterio ({sc}) difere do citado no adendo ({m.group(1) if m else None})")
    if not P.ADENDO2.exists():
        raise Aborta("criterio_G1_adendo2.json ausente")
    with open(P.ADENDO2, "r", encoding="utf-8") as f:
        adendo2 = json.load(f)
    m2 = re.search(r"sha256 ([0-9a-f]{64})", adendo2.get("emenda", ""))
    if adendo2.get("id") != "G1_modelos_g10_adendo2" or not m2 or m2.group(1) != sa:
        raise Aborta("adendo2: id inesperado ou sha256 citado do adendo1 difere do observado")
    if not P.ADENDO3.exists():
        raise Aborta("criterio_G1_adendo3.json ausente")
    with open(P.ADENDO3, "r", encoding="utf-8") as f:
        adendo3 = json.load(f)
    m3 = re.search(r"sha256 ([0-9a-f]{64})", adendo3.get("emenda", ""))
    s2 = sha256(P.ADENDO2)
    if adendo3.get("id") != "G1_modelos_g10_adendo3" or not m3 or m3.group(1) != s2:
        raise Aborta("adendo3: id inesperado ou sha256 citado do adendo2 difere do observado")
    if not P.ADENDO4.exists():
        raise Aborta("criterio_G1_adendo4.json ausente")
    with open(P.ADENDO4, "r", encoding="utf-8") as f:
        adendo4 = json.load(f)
    m4 = re.search(r"sha256 ([0-9a-f]{64})", adendo4.get("emenda", ""))
    s3 = sha256(P.ADENDO3)
    if adendo4.get("id") != "G1_modelos_g10_adendo4" or not m4 or m4.group(1) != s3:
        raise Aborta("adendo4: id inesperado ou sha256 citado do adendo3 difere do observado")
    return sc, sa, s2, s3, sha256(P.ADENDO4)


def conferir_plano(plano_doc: dict) -> None:
    """Check plano_G1.json: g/b = 10/2; 80/40/80 runs in blocks 1/2/3 with 200 unique labels; split seeds taken
    in order from the 60-draw list of the drift experiment (first 20 / 5 / 20); labels consistent with their fields.
    """
    if plano_doc.get("grid_km") != GRID_KM or plano_doc.get("buffer_km") != BUFFER_KM:
        raise Aborta("plano_G1.json: grid_km/buffer_km != 10/2")
    with open(P.SORTEIOS_2_1, "r", encoding="utf-8") as f:
        lista = json.load(f)["nota_divergencia_seeds"]["seeds_usados_nesta_rodada"]
    esp = {1: lista[:20], 2: lista[:5], 3: lista[:20]}
    cs = plano_doc["corridas"]
    cont = {b: sum(1 for c in cs if c["bloco"] == b) for b in (1, 2, 3)}
    if cont != {1: 80, 2: 40, 3: 80} or len(cs) != 200 or len({c["run_label"] for c in cs}) != 200:
        raise Aborta(f"plano_G1.json: contagem por bloco {cont} / rotulos unicos != 80/40/80, 200")
    for c in plano_doc["corridas"]:
        if c["split_seed"] != esp[c["bloco"]][c["indice_sorteio"]]:
            raise Aborta(f"plano_G1.json: split_seed de {c['run_label']} fora da lista do 2.1")
        rot = f"g1_{c['tipo']}_{c['cidade']}_{c['quadrante']}_ss{c['split_seed']}_s{c['seed_treino']}"
        if rot != c["run_label"]:
            raise Aborta(f"plano_G1.json: run_label {c['run_label']} != {rot}")


_REF_A4 = {}


def referencia_a4(tipo: str) -> dict:
    """Reference configuration and code hashes of the A4 runs of one model type (cached).

    Aborts if a field is missing or if the A4 runs disagree outside CONFIG_IGNORADAS_A4.
    """
    if tipo in _REF_A4:
        return _REF_A4[tipo]
    fs = sorted(glob.glob(str(P.A4 / f"{tipo}_v3_a4_bauru_Q*_ss*" / "run_*.json")))
    if not fs:
        raise Aborta(f"A4: nenhuma corrida {tipo}_v3_a4_bauru_Q*_ss* encontrada em {P.A4}")
    refs = []
    for f in fs:
        with open(f, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        cfg = {k: v for k, v in (d.get("config") or {}).items() if k not in CONFIG_IGNORADAS_A4}
        mv = d.get("modelo_v3") or {}
        refs.append({"config": cfg, "artefato_tipo": d.get("artefato_tipo"), "script_sha256": d.get("script_sha256"),
                     "decoder_sha256": mv.get("decoder_sha256"), "wrapper_sha256": mv.get("wrapper_sha256")})
    r0 = refs[0]
    if any(v is None for v in (r0["artefato_tipo"], r0["script_sha256"], r0["decoder_sha256"], r0["wrapper_sha256"])) or not r0["config"]:
        raise Aborta(f"A4: referencia {tipo} com campo ausente/None (artefato_tipo, script/decoder/wrapper sha, config)")
    if any(r != refs[0] for r in refs[1:]):
        raise Aborta(f"A4: as {len(fs)} corridas {tipo} nao tem config identica fora das chaves ignoradas; referencia ambigua")
    _REF_A4[tipo] = {"n_corridas_a4": len(fs), **refs[0]}
    return _REF_A4[tipo]


def _igual(a, b) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= TOL_FLOAT


def _num(x) -> bool:
    """True for a finite int or float (bool excluded)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and bool(np.isfinite(x))


def _rel_ok(a, b, tol) -> bool:
    """|a - b| <= tol * |b|; a missing, non-numeric or non-finite value fails."""
    return _num(a) and _num(b) and b != 0 and abs(a - b) <= tol * abs(b)


def conferir_corrida(c: dict, dj: dict, idx_global, side, man: dict, ref21: dict, m) -> dict:
    """All checks for one run; raises Aborta listing every failure.

    m is mae_pop() of the run's .npz, or None for a run without .npz registered as having no valid test
    node (then only the run-JSON checks apply, plus rule C2).
    """
    lbl, tipo = c["run_label"], c["tipo"]
    falhas = []

    def ex(nome, ok, det=""):
        if not ok:
            falhas.append(f"{nome}{(' (' + str(det) + ')') if det else ''}")

    cfg, geo, spl = dj.get("config") or {}, dj.get("geometria") or {}, dj.get("split") or {}
    ins = dj.get("insumos") or {}
    ptest = (dj.get("particoes") or {}).get("test") or {}
    sel = (dj.get("selecao") or {}).get("test_no_melhor_ckpt") or {}
    # run label; dataset file name and SHA-256 against the data manifest
    ex("run_label", dj.get("run_label") == lbl and cfg.get("run_label") == lbl, dj.get("run_label"))
    esp_nome = f"transfer_dataset_{c['cidade']}_v19_{c['quadrante']}_enriched_cftudo.pt"
    ex("dataset_nome", Path(cfg.get("rf_data_file") or "").name == esp_nome
       and Path(ins.get("rf_data_file") or "").name == esp_nome, (cfg.get("rf_data_file"), ins.get("rf_data_file")))
    ex("dataset_sha256_x_manifest_v4", bool(ins.get("sha256_rf_data")) and man.get(esp_nome) == ins.get("sha256_rf_data"),
       (ins.get("sha256_rf_data"), man.get(esp_nome)))
    ex("dataset_flag_bate_manifest_v4", ins.get("sha256_rf_data_bate_manifest_v4") is True)
    # g and b recorded in the run JSON (config, geometry, split)
    gb_run = {"config.grid_km": cfg.get("grid_km"), "geometria.grid_km_usado": geo.get("grid_km_usado"),
              "split.grid_km": spl.get("grid_km")}
    bb_run = {"config.buffer_km": cfg.get("buffer_km"), "geometria.buffer_km_usado": geo.get("buffer_km_usado"),
              "split.buffer_km": spl.get("buffer_km")}
    for k, v in gb_run.items():
        ex(f"run_json_g=10[{k}]", _igual(v, GRID_KM), v)
    for k, v in bb_run.items():
        ex(f"run_json_b=2[{k}]", _igual(v, BUFFER_KM), v)
    # training seed and split seed against the plan
    ex("seed_x_plano", dj.get("seed") == c["seed_treino"] and cfg.get("seed") == c["seed_treino"],
       (dj.get("seed"), cfg.get("seed"), c["seed_treino"]))
    ex("split_seed_x_plano", dj.get("split_seed") == c["split_seed"] and cfg.get("split_seed") == c["split_seed"]
       and spl.get("split_seed") == c["split_seed"], (dj.get("split_seed"), cfg.get("split_seed"), spl.get("split_seed"), c["split_seed"]))
    ref = ref21.get((f"{c['cidade']}_{c['quadrante']}", c["split_seed"]))
    # (C1) test-node counts against the drift experiment, same cell and split seed
    ex("ref_2_1_presente", ref is not None)
    n_test_run = (spl.get("n_nos_apos_buffer") or {}).get("test")
    if ref is not None:
        ex("n_test_run_x_2_1", n_test_run == ref["n_test"], (n_test_run, ref["n_test"]))
        ex("particoes_test_n_x_2_1", ptest.get("n") == ref["n_test"], (ptest.get("n"), ref["n_test"]))
    if m is None:
        # (C2) run without .npz: the drift experiment and the run JSON both report no valid test node
        ex("C2_sem_validos_2_1_validos==0", ref is not None and ref["validos"] == 0, ref and ref["validos"])
        ex("C2_sem_validos_run_json_n_pl_alvo_valido==0", ptest.get("n_pl_alvo_valido") == 0, ptest.get("n_pl_alvo_valido"))
        h = None
        dif_rel = None
    else:
        if ref is not None:
            ex("n_validos_npz_x_2_1", m["n_validos"] == ref["validos"], (m["n_validos"], ref["validos"]))
            ex("n_todos_npz_x_2_1", m["n"] == ref["todos"], (m["n"], ref["todos"]))
        # sidecar of the g = 10 km wrapper: requested and effective g/b, retained test nodes
        if not isinstance(side, dict):
            ex("sidecar_shim_g10_presente", False)
        else:
            ex("sidecar_artefato", side.get("artefato") == "shim_g10" and side.get("run_label") == lbl
               and side.get("modelo") == tipo, (side.get("artefato"), side.get("run_label"), side.get("modelo")))
            ex("sidecar_g_pedido=10", _igual(side.get("grid_km_pedido"), GRID_KM), side.get("grid_km_pedido"))
            ex("sidecar_b_pedido=2", _igual(side.get("buffer_km_pedido"), BUFFER_KM), side.get("buffer_km_pedido"))
            ef = side.get("efetivo") or {}
            for k in ("config.grid_km", "geometria.grid_km_usado", "split.grid_km"):
                ex(f"sidecar_g_efetivo=10[{k}]", _igual(ef.get(k), GRID_KM), ef.get(k))
            for k in ("config.buffer_km", "geometria.buffer_km_usado", "split.buffer_km"):
                ex(f"sidecar_b_efetivo=2[{k}]", _igual(ef.get(k), BUFFER_KM), ef.get(k))
            ex("sidecar_g_b_efetivos_conferem", side.get("g_b_efetivos_conferem") is True)
            ex("sidecar_npz_igual_teste_retido", side.get("npz_igual_teste_retido") is True)
            ex("sidecar_n_nos_x_run_x_npz", n_test_run is not None and side.get("n_nos_npz") == len(idx_global)
               == side.get("n_nos_teste_retidos_run_json") == n_test_run,
               (side.get("n_nos_npz"), len(idx_global), side.get("n_nos_teste_retidos_run_json"), n_test_run))
        # test-partition index hash against the run JSON
        h = hash_idx(idx_global)
        ex("hash_idx_npz_x_run_json", ptest.get("idx_sha256_global") is not None and h == ptest.get("idx_sha256_global"),
           (h, ptest.get("idx_sha256_global")))
        ex("n_idx_npz_x_particoes_test", ptest.get("n") == len(idx_global), (ptest.get("n"), len(idx_global)))
        # (C3) tie the .npz to its run: metrics recomputed from the .npz vs selecao.test_no_melhor_ckpt
        ex("amarra_npz_n_nos", sel.get("n_nos") == m["n"], (sel.get("n_nos"), m["n"]))
        ex("amarra_npz_n_pl_alvo_valido", sel.get("n_pl_alvo_valido") == m["n_validos"] == ptest.get("n_pl_alvo_valido"),
           (sel.get("n_pl_alvo_valido"), m["n_validos"], ptest.get("n_pl_alvo_valido")))
        dif_rel = {"dif_rel_mae": (abs(m["mae_rssi_todos_db"] - sel["mae_rssi_db"]) / abs(sel["mae_rssi_db"]) if _num(sel.get("mae_rssi_db")) and sel.get("mae_rssi_db") != 0 else None),
                   "dif_rel_rmse": (abs(m["rmse_rssi_todos_db"] - sel["rmse_rssi_db"]) / abs(sel["rmse_rssi_db"]) if _num(sel.get("rmse_rssi_db")) and sel.get("rmse_rssi_db") != 0 else None)}
        ex("amarra_npz_mae_rssi_todos", _rel_ok(m["mae_rssi_todos_db"], sel.get("mae_rssi_db"), TOL_AMARRA_MAE_REL),
           "MAE do .npz difere do run JSON alem da tolerancia")
        ex("amarra_npz_rmse_rssi_todos", _rel_ok(m["rmse_rssi_todos_db"], sel.get("rmse_rssi_db"), TOL_AMARRA_RMSE_REL),
           "RMSE do .npz difere do run JSON alem da tolerancia")
    # configuration and code hashes identical to batch A4
    ref_a4 = referencia_a4(tipo)
    cfg_cmp = {k: v for k, v in cfg.items() if k not in CONFIG_IGNORADAS_A4}
    if cfg_cmp != ref_a4["config"]:
        dif = sorted(k for k in set(cfg_cmp) | set(ref_a4["config"]) if cfg_cmp.get(k) != ref_a4["config"].get(k) or
                     (k in cfg_cmp) != (k in ref_a4["config"]))
        ex("config_igual_A4", False, {k: (cfg_cmp.get(k), ref_a4["config"].get(k)) for k in dif})
    mv = dj.get("modelo_v3") or {}
    ex("script_sha_igual_A4", dj.get("script_sha256") == ref_a4["script_sha256"] and dj.get("artefato_tipo") == ref_a4["artefato_tipo"]
       and mv.get("decoder_sha256") == ref_a4["decoder_sha256"] and mv.get("wrapper_sha256") == ref_a4["wrapper_sha256"])
    if falhas:
        raise Aborta(f"{lbl}: " + "; ".join(falhas))
    return {"run_label": lbl, "passou": True, "com_npz": m is not None, "dataset_sha256": ins.get("sha256_rf_data"), "idx_sha256_teste": h, "amarracao": dif_rel}


def main(argv=None) -> int:
    """Command-line entry point; returns the exit code (0 ok, 2 block refused, 4 check failed)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bloco", type=int, choices=(1, 2, 3), default=None,
                    help="default: agrega todo bloco COMPLETO; blocos incompletos sao recusados (rc=2)")
    ap.add_argument("--bloco3-truncado", action="store_true",
                    help="declara que o lote terminou e o bloco 3 segue incompleto: usa os primeiros k sorteios com as 4 "
                         "corridas completas (exige k validos >= 10 por celula). Adendo, emenda 4.")
    ap.add_argument("--so-conferir", action="store_true",
                    help="roda so as conferencias do item 7 nas corridas ja completas do bloco pedido; nao calcula MAE, nao grava")
    ap.add_argument("--raiz-teste", default=None,
                    help="SO PARA TESTES: raiz alternativa (pasta _v3_*); grava modo_teste=true no JSON")
    args = ap.parse_args(argv)
    argv_efetivo = sys.argv if argv is None else [sys.argv[0]] + list(argv)
    modo_teste = args.raiz_teste is not None
    definir_caminhos(Path(args.raiz_teste) if modo_teste else RAIZ_PADRAO)
    _REF_A4.clear()

    try:
        return _executar(args, argv_efetivo, modo_teste)
    except Aborta as e:
        print(f"ABORTADO (conferencia do item 7; nada gravado): {e}", file=sys.stderr)
        return 4
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ABORTADO (entrada de conferencia ausente/ilegivel; nada gravado): {type(e).__name__}: {e}", file=sys.stderr)
        return 4


def _ler_sidecar(sd):
    """Sidecar JSON of a run, or None if absent or unreadable."""
    if not sd:
        return None
    try:
        with open(sd[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _executar(args, argv_efetivo, modo_teste) -> int:
    """Global checks, then aggregation of each requested block; one JSON per complete block."""
    sha_crit, sha_adendo, sha_adendo2, sha_adendo3, sha_adendo4 = conferir_criterios()
    prov_scripts = conferir_scripts_do_lote(modo_teste)
    with open(P.PLANO, "r", encoding="utf-8") as f:
        plano_doc = json.load(f)
    conferir_plano(plano_doc)
    plano = plano_doc["corridas"]
    with open(P.STATUS, "r", encoding="utf-8") as f:
        status = json.load(f)
    sem_val = {c["run_label"] for c in status["corridas"] if c.get("sem_validos")}
    status_por_lbl = {c["run_label"]: c for c in status["corridas"]}
    # while a resumed batch runs, the status keeps the previous batch outcome (fields retomadas and veredito_lote_anterior);
    # a run JSON without .npz may then be a run in progress: refuse (exit code 2) rather than abort
    retomada_em_curso_possivel = bool(status.get("retomadas")) and status.get("veredito_lote") == status.get("veredito_lote_anterior")
    nao_trein = {c["run_label"] for c in status["corridas"] if c.get("nao_treinavel") is True}
    if nao_trein & sem_val:
        raise Aborta(f"status marca sem_validos E nao_treinavel na mesma corrida: {sorted(nao_trein & sem_val)}")
    tam_man = manifest_tamanhos()
    man = manifest_cftudo()
    ref21 = carregar_2_1()
    sha_plano = sha256(P.PLANO)

    def artefatos(lbl):
        """Run folder and its run JSON, .npz and sidecar paths; aborts on unexpected file names (C3)."""
        d = P.G1 / lbl
        achados = []
        for pat, esp in (("run_*.json", f"run_{lbl}.json"), ("predicoes_*.npz", f"predicoes_{lbl}.npz"),
                         ("shim_g10_*.json", f"shim_g10_{lbl}.json")):
            fs = sorted(glob.glob(str(d / pat)))
            if len(fs) > 1 or (fs and Path(fs[0]).name != esp):
                raise Aborta(f"{lbl}: arquivos {pat} inesperados na pasta {[Path(x).name for x in fs]} (esperado exatamente {esp})")
            achados.append(fs)
        return (d, *achados)

    def completa(c) -> bool:
        """True when a run is ready to aggregate: complete run JSON plus .npz, or a status-flagged run without .npz."""
        lbl = c["run_label"]
        d, js, zs, sd = artefatos(lbl)
        if (lbl in sem_val or lbl in nao_trein) and not zs:
            return True
        # with the batch closed, a run JSON without .npz and without a status flag is not a run in progress (amendment 3)
        if js and not zs and lbl not in sem_val and lbl not in nao_trein and status.get("veredito_lote") is not None and not retomada_em_curso_possivel:
            raise Aborta(f"{lbl}: run JSON sem .npz e sem registro sem_validos no status, com o lote encerrado")
        if not (js and zs):
            return False
        try:
            with open(js[0], "r", encoding="utf-8") as f:
                r = json.load(f)
        except Exception:
            return False
        if (lbl in sem_val or lbl in nao_trein) and not ("modelo_v3" in r and "insumos" in r):
            raise Aborta(f"{lbl}: .npz presente mas run JSON parcial (sem modelo_v3/insumos) em corrida marcada sem_validos/nao_treinavel")
        return "modelo_v3" in r and "insumos" in r

    def indices_completos(n: int):
        """Draw index -> True when all 4 runs of that draw in block n are complete, in plan order."""
        por = {}
        for c in plano:
            if c["bloco"] == n:
                por.setdefault(c["indice_sorteio"], []).append(c)
        return {i: all(completa(c) for c in por[i]) for i in sorted(por)}

    cache = {}
    conferidas = {}
    # runs with a checked .npz in draws excluded because another run of the same draw is nao_treinavel
    excluidos_com_npz = set()

    def conferir_parcial(c, dj, zs, sd, modo) -> dict:
        """Checks for a PARTIAL run JSON (trainer stopped before training; no .npz, no sidecar).

        modo "sem_validos": no valid test node (amendment 3); modo "nao_treinavel": degenerate validation or training
        partition named in the run log and in the partition guard of the run JSON (amendment 4). Raises Aborta
        listing every failure.
        """
        lbl = c["run_label"]
        pre = "A3" if modo == "sem_validos" else "A4"
        falhas = []

        def ex(nome, ok, det=""):
            if not ok:
                falhas.append(f"{nome}{(' (' + str(det) + ')') if det else ''}")

        cl = f"{c['cidade']}_{c['quadrante']}"
        ref = ref21.get((cl, c["split_seed"]))
        e = status_por_lbl.get(lbl) or {}
        rc_ = e.get("rc")
        ex("A3b_status_rc!=0" if modo == "sem_validos" else "A4_status_rc!=0", isinstance(rc_, int) and not isinstance(rc_, bool) and rc_ != 0, rc_)
        if modo == "sem_validos":
            ex("A3a_2_1_validos==0", ref is not None and ref["validos"] == 0, ref and ref["validos"])
            ex("A3b_status_sem_validos==true", e.get("sem_validos") is True)
        else:
            ex("A4a_status_nao_treinavel==true", e.get("nao_treinavel") is True)
            ex("A4a_proveniencia_da_retomada_conferida", prov_scripts.get("retomada") is not None)
        logp = e.get("log")
        txt = None
        if isinstance(logp, str) and Path(logp).is_file():
            txt = Path(logp).read_text(encoding="utf-8", errors="replace")
        ex(f"{pre}b_log_da_corrida_lido", txt is not None, logp)
        ex(f"{pre}b_log_contem_particao_degenerada", txt is not None and TRECHO_LOG_DEGENERADA in txt)
        nomes = []
        if txt is not None:
            mm = re.findall(TRECHO_LOG_DEGENERADA + r":\s*\[([^\]]*)\]", txt)
            for grp in mm:
                nomes += re.findall(r"['\"]([A-Za-z_]+)['\"]", grp)
        if modo == "nao_treinavel":
            ex("A4b_particao_nomeada_no_log", bool(nomes), nomes)
            ex("A4b_particao_nomeada_diferente_de_test", bool(nomes) and "test" not in nomes, nomes)
            gl = (dj.get("guarda_particao_degenerada") or {}).get("particoes_sem_aresta_antena")
            ex("A4c_guarda_particoes_sem_aresta_antena_presente_e_nao_vazia", isinstance(gl, list) and len(gl) > 0, gl)
            ex("A4c_guarda_particoes_igual_as_nomeadas_no_log", isinstance(gl, list) and bool(nomes) and set(gl) == set(nomes), (gl, nomes))
            ex("A4c_guarda_particoes_sem_test", isinstance(gl, list) and "test" not in gl, gl)
        for k in ("selecao", "insumos", "modelo_v3"):
            ex(f"{pre}c_parcial_sem_{k}", k not in dj)
        ex(f"{pre}c_sem_npz_nem_sidecar", not zs and not sd, (len(zs), len(sd)))
        cfg, geo, spl = dj.get("config") or {}, dj.get("geometria") or {}, dj.get("split") or {}
        ex(f"{pre}c_run_label", dj.get("run_label") == lbl and cfg.get("run_label") == lbl, dj.get("run_label"))
        for k, v in (("config.grid_km", cfg.get("grid_km")), ("geometria.grid_km_usado", geo.get("grid_km_usado")), ("split.grid_km", spl.get("grid_km"))):
            ex(f"{pre}c_g=10[{k}]", _igual(v, GRID_KM), v)
        for k, v in (("config.buffer_km", cfg.get("buffer_km")), ("geometria.buffer_km_usado", geo.get("buffer_km_usado")), ("split.buffer_km", spl.get("buffer_km"))):
            ex(f"{pre}c_b=2[{k}]", _igual(v, BUFFER_KM), v)
        ex(f"{pre}c_seed_x_plano", dj.get("seed") == c["seed_treino"] and cfg.get("seed") == c["seed_treino"], (dj.get("seed"), cfg.get("seed")))
        ex(f"{pre}c_split_seed_x_plano", dj.get("split_seed") == c["split_seed"] and cfg.get("split_seed") == c["split_seed"]
           and spl.get("split_seed") == c["split_seed"], (dj.get("split_seed"), cfg.get("split_seed"), spl.get("split_seed")))
        esp_nome = f"transfer_dataset_{c['cidade']}_v19_{c['quadrante']}_enriched_cftudo.pt"
        dsb = dj.get("dataset") or {}
        ex(f"{pre}c_dataset_nome", Path(dsb.get("rf_data_file") or "").name == esp_nome and Path(cfg.get("rf_data_file") or "").name == esp_nome,
           (dsb.get("rf_data_file"), cfg.get("rf_data_file")))
        ex(f"{pre}c_dataset_bytes_x_manifest_v4", dsb.get("rf_data_bytes") is not None and tam_man.get(esp_nome) == dsb.get("rf_data_bytes"),
           (dsb.get("rf_data_bytes"), tam_man.get(esp_nome)))
        pt = (dj.get("particoes") or {}).get("test") or {}
        gd = dj.get("guarda_particao_degenerada") or {}
        if modo == "sem_validos":
            ex("A3c_particoes_test_n_pl_alvo_valido==0", pt.get("n_pl_alvo_valido") == 0, pt.get("n_pl_alvo_valido"))
            ex("A3c_guarda_n_pl_alvo_valido_test==0", (gd.get("n_pl_alvo_valido") or {}).get("test") == 0, (gd.get("n_pl_alvo_valido") or {}).get("test"))
            ex("A3c_guarda_test_sem_aresta_antena", "test" in (gd.get("particoes_sem_aresta_antena") or []))
        if falhas:
            raise Aborta(f"{lbl}: {modo} (adendo {'3' if modo == 'sem_validos' else '4'}): " + "; ".join(falhas))
        ntest = (spl.get("n_nos_apos_buffer") or {}).get("test")
        out = {"run_label": lbl, "passou": True, "com_npz": False, "dataset_bytes": dsb.get("rf_data_bytes"),
               "n_test_parcial_igual_2_1": bool(ref is not None and ntest == ref["n_test"]),
               "validos_2_1": None if ref is None else ref["validos"]}
        out["parcial_sem_validos_adendo3" if modo == "sem_validos" else "parcial_nao_treinavel_adendo4"] = True
        if modo == "nao_treinavel":
            out["particoes_nomeadas_no_log"] = nomes
            out["guarda_particoes_sem_aresta_antena"] = gd.get("particoes_sem_aresta_antena")
        return out

    def carregar(c) -> dict:
        """Check and load one run (cached); returns its record with the recomputed metrics."""
        lbl = c["run_label"]
        if lbl in cache:
            return cache[lbl]
        d, js, zs, sd = artefatos(lbl)
        if (lbl in sem_val or lbl in nao_trein) and not zs:
            modo = "sem_validos" if lbl in sem_val else "nao_treinavel"
            if not js:
                raise Aborta(f"{lbl}: registrada {modo} so no status; run JSON ausente (adendo {'3' if modo == 'sem_validos' else '4'})")
            with open(js[0], "r", encoding="utf-8") as f:
                dj = json.load(f)
            conferidas[lbl] = conferir_parcial(c, dj, zs, sd, modo)
            r = {"run_label": lbl, "sem_validos": modo == "sem_validos", "nao_treinavel": modo == "nao_treinavel", "sem_predicao": True, "mae": None}
        else:
            with open(js[0], "r", encoding="utf-8") as f:
                dj = json.load(f)
            side = _ler_sidecar(sd)
            z = np.load(zs[0])
            idx = z["idx_global"]
            if lbl in nao_trein:
                raise Aborta(f"{lbl}: marcada nao_treinavel no status, mas tem run JSON/.npz de corrida concluida (so o ramo parcial verificado aceita o marcador)")
            m = mae_pop(z)
            conferidas[lbl] = conferir_corrida(c, dj, idx, side, man, ref21, m)
            if lbl in sem_val and m["n_validos"] != 0:
                raise Aborta(f"{lbl}: status registra sem_validos, mas o .npz tem {m['n_validos']} nos validos")
            ins = dj.get("insumos") or {}
            rf = Path(ins.get("rf_data_file") or "").name
            geo = dj.get("split") or {}
            ref = ref21.get((f"{c['cidade']}_{c['quadrante']}", c["split_seed"]))
            cfg = dj.get("config") or {}
            r = {
                "run_label": lbl, "nao_treinavel": False, "sem_validos": m["n_validos"] == 0, "sem_predicao": False, "npz": str(zs[0]), "mae": m,
                "grid_km": cfg.get("grid_km"), "buffer_km": cfg.get("buffer_km"), "split_seed_run": dj.get("split_seed"),
                "seed_treino_run": dj.get("seed"),
                "proveniencia_ok": bool(ins.get("sha256_rf_data") and man.get(rf) == ins.get("sha256_rf_data")),
                "dist_min_entre_particoes_km": {k: v.get("dist_min_km") for k, v in ((geo.get("verificacao") or {}).get("pares") or {}).items()},
                "intersecoes": (geo.get("verificacao") or {}).get("intersecoes"),
                "n_test_retido_run": (geo.get("n_nos_apos_buffer") or {}).get("test"),
                "baselines_test": {b: ((dj.get("baselines_analiticos") or {}).get("test") or {}).get(f"baseline_{b}_mae") for b in BASELINES},
                "melhor_epoca": (dj.get("selecao") or {}).get("melhor_epoca"),
                "bate_com_2_1": {"n_test_igual": True, "n_validos_igual": True, "n_todos_igual": True,
                                 "nota": "divergencia ABORTA (C1); campos mantidos por compatibilidade"},
                "mae_constante_validos_2_1": ref["mae_constante_validos"],
            }
        cache[lbl] = r
        return r

    def runs(bloco_n: int, cel: str, tipo: str, seed: int, n_sort: int, apenas_indices=None):
        """Plan entries of one block, cell, model and training seed, by draw index: first n_sort, optionally only given indices."""
        sel = [c for c in plano if c["bloco"] == bloco_n and c["tipo"] == tipo
               and f"{c['cidade']}_{c['quadrante']}" == cel and c["seed_treino"] == seed]
        sel = sorted(sel, key=lambda c: c["indice_sorteio"])[:n_sort]
        if apenas_indices is not None:
            sel = [c for c in sel if c["indice_sorteio"] in apenas_indices]
        return sel

    def por_celula_modelo(bloco_n, cel, tipo, seed, n_sort, apenas_indices=None):
        return {c["split_seed"]: carregar(c) for c in runs(bloco_n, cel, tipo, seed, n_sort, apenas_indices)}

    def metricas_bloco_simples(n: int, indices: list) -> dict:
        """Per-cell, per-model metrics of a seed-42 block (1 or 3) over the given draw indices, plus sentinel parity."""
        cels = {}
        for cel in CELULAS_POR_BLOCO[n]:
            g = por_celula_modelo(n, cel, "gnn", 42, 20, set(indices))
            m = por_celula_modelo(n, cel, "mlp", 42, 20, set(indices))
            ordem = [c["split_seed"] for c in sorted(runs(n, cel, "gnn", 42, 20, set(indices)), key=lambda c: c["indice_sorteio"])]
            ss_plano = [s for s in ordem if s in g and s in m]
            # a draw with an untrainable run is excluded for both models
            nt = [s for s in ss_plano if g[s].get("nao_treinavel") or m[s].get("nao_treinavel")]
            ss = [s for s in ss_plano if s not in nt]
            for s_ in nt:
                for st_ in (g, m):
                    if st_[s_].get("mae") is not None:
                        excluidos_com_npz.add(st_[s_]["run_label"])
            sv = [s for s in ss if g[s]["sem_validos"] or m[s]["sem_validos"]]
            bloco = {"sorteios": ss, "n_sorteios_plano_considerados": len(ss_plano), "n_sorteios_plano_usados": len(ss),
                     "sem_validos": sv, "n_sorteios_sem_validos": len(sv),
                     "nao_treinaveis": nt, "n_sorteios_nao_treinaveis": len(nt),
                     "n_sorteios_usaveis": len(ss) - len(sv), "por_modelo": {}}
            for nome, store in (("gnn", g), ("mlp", m)):
                ok = [s for s in ss if not store[s]["sem_validos"]]
                v = [store[s]["mae"]["mae_rssi_validos_db"] for s in ok]
                d = dp(v)
                ac = avaliar_condicoes_modelo(d, RUIDO_REPETICAO_DB)
                const = [store[s]["mae_constante_validos_2_1"] for s in ok]
                bloco["por_modelo"][nome] = {
                    "n_sorteios_com_validos": len(ok),
                    "mae_validos_por_sorteio": {str(s): store[s]["mae"]["mae_rssi_validos_db"] for s in ok},
                    "dp_entre_sorteios_validos_db": d,
                    "razao_dp_sobre_0_132": razao(d, RUIDO_REPETICAO_DB),
                    "dp_ge_3x_0_132": ac["dp_ge_3x_comparador"],
                    "dp_lt_2x_0_132": ac["dp_lt_2x_comparador"],
                    "correlacao_com_constante_validos": {"pearson": pearson(v, const), "spearman": spearman(v, const), "n": len(ok)},
                    "dp_entre_sorteios_demais_populacoes_db": {p: dp([store[s]["mae"][p] for s in ok]) for p in POPS},
                    "inversao_vs_baselines_validos": {b: sum(1 for s in ok if store[s]["baselines_test"][b] is not None and
                                                              store[s]["mae"]["mae_rssi_validos_db"] < store[s]["baselines_test"][b])
                                                      for b in BASELINES},
                    "proveniencia_toda_ok": all(store[s]["proveniencia_ok"] for s in ok),
                    "geometria_declarada": sorted({(store[s]["grid_km"], store[s]["buffer_km"]) for s in ok}),
                    "dist_min_entre_particoes_km_min": min([x for s in ok for x in store[s]["dist_min_entre_particoes_km"].values() if x is not None], default=None),
                    "bate_com_2_1_todos": all((store[s]["bate_com_2_1"] or {}).get(k) for s in ok for k in ("n_test_igual", "n_validos_igual", "n_todos_igual")),
                    "melhores_epocas": [store[s]["melhor_epoca"] for s in ok],
                }
            # sentinel parity over every draw with predictions from both models, including draws without valid test nodes
            com_pred = [s for s in ss if not g[s].get("sem_predicao") and not m[s].get("sem_predicao")]
            sem_pred = [s for s in ss if s not in com_pred]
            dif_sent = {str(s): (abs(g[s]["mae"]["mae_rssi_sentinela_db"] - m[s]["mae"]["mae_rssi_sentinela_db"])
                                 if g[s]["mae"]["mae_rssi_sentinela_db"] is not None and m[s]["mae"]["mae_rssi_sentinela_db"] is not None else None)
                        for s in com_pred}
            md = med(list(dif_sent.values()))
            bloco["paridade_sentinela"] = {
                "abs_gnn_menos_mlp_por_sorteio": dif_sent, "mediana_db": md,
                "n_sorteios_na_mediana": sum(1 for v in dif_sent.values() if v is not None),
                "n_sorteios_sem_validos_incluidos_na_mediana": sum(1 for s in com_pred if s in sv and dif_sent[str(s)] is not None),
                "sorteios_sem_predicao_fora_da_mediana": sem_pred,
                "mediana_le_0_117": (None if md is None else bool(md <= RUIDO_SENTINELA_DB)),
                "mesma_particao_gnn_mlp": all(g[s]["mae"]["idx_sha256"] == m[s]["mae"]["idx_sha256"] for s in com_pred)}
            cels[cel] = bloco
        return cels

    def sinais_simples(cels: dict) -> dict:
        """Cell flags against the fixed 0.132 dB comparator (the comparator of the Q3 cells; provisional for Q1)."""
        out = {}
        for cel, b in cels.items():
            pm = {t: avaliar_condicoes_modelo(b["por_modelo"][t]["dp_entre_sorteios_validos_db"], RUIDO_REPETICAO_DB) for t in ("gnn", "mlp")}
            s = sinalizar_celula(pm)
            s["paridade_le_0_117"] = bool(b["paridade_sentinela"]["mediana_le_0_117"])
            s["por_modelo"] = pm
            out[cel] = s
        return out

    def avaliar_q1() -> dict:
        """Q1 cells from blocks 1 and 2: comparator max(0.132 dB, pooled SD across seeds) and condition 2.

        Also returns, for information only, the reading with the seed SD pooled over both Q1 cells.
        """
        i1 = indices_completos(1)
        i2 = indices_completos(2)
        if not all(i1.values()) or not all(i2.values()):
            faltam = [c["run_label"] for c in plano if c["bloco"] in (1, 2) and not completa(c)]
            raise Incompleto(f"{len(faltam)} corridas dos blocos 1/2 sem resultado")
        cels1 = metricas_bloco_simples(1, sorted(i1))
        cel_out, flags = {}, {}
        for cel in CELULAS_POR_BLOCO[2]:
            ent, pm = {"gnn": {}, "mlp": {}}, {}
            ps_all = {t_: {sd_: {c_["split_seed"]: carregar(c_) for c_ in runs(1 if sd_ == 42 else 2, cel, t_, sd_, 5)} for sd_ in (42, 43, 44)}
                      for t_ in ("gnn", "mlp")}
            # draws of this cell with an untrainable run, excluded for both models and all seeds
            nt_cel = {s_ for t_ in ps_all for sd_ in ps_all[t_] for s_, r_ in ps_all[t_][sd_].items() if r_.get("nao_treinavel")}
            for t_ in ps_all:
                for sd_ in ps_all[t_]:
                    for s_ in nt_cel:
                        if s_ in ps_all[t_][sd_] and ps_all[t_][sd_][s_].get("mae") is not None:
                            excluidos_com_npz.add(ps_all[t_][sd_][s_]["run_label"])
            for tipo in ("gnn", "mlp"):
                por_semente = ps_all[tipo]
                sorteios = [c["split_seed"] for c in runs(2, cel, tipo, 43, 5)]
                lin = [s for s in sorteios if s not in nt_cel and all(s in por_semente[sd] and not por_semente[sd][s]["sem_validos"] for sd in (42, 43, 44))]
                M = np.array([[por_semente[sd][s]["mae"]["mae_rssi_validos_db"] for sd in (42, 43, 44)] for s in lin], float)
                dec = decomposicao_um_fator(M) if len(lin) >= 2 else None
                dp_sem = None if dec is None else dec["dp_entre_sementes_pooled_db"]
                dp20 = cels1[cel]["por_modelo"][tipo]["dp_entre_sorteios_validos_db"]
                # Q1 comparator: the larger of the repetition noise and the pooled SD across seeds of this cell
                comparador = max(RUIDO_REPETICAO_DB, dp_sem) if dp_sem is not None else None
                pm[tipo] = avaliar_condicoes_modelo(dp20, comparador)
                ent[tipo] = {
                    "sorteios_usados": lin, "n_sorteios_usados": len(lin),
                    "sorteios_do_bloco_2_sem_validos_excluidos": [s for s in sorteios if s not in lin and s not in nt_cel],
                    "sorteios_do_bloco_2_nao_treinaveis_excluidos": sorted(s for s in sorteios if s in nt_cel),
                    "matriz_mae_validos_sorteio_x_semente_42_43_44": M.tolist(),
                    "decomposicao_um_fator_sementes_aninhadas_no_sorteio": dec,
                    "dp_entre_sementes_pooled_db": dp_sem,
                    "dp_entre_sementes_sobre_0_132": razao(dp_sem, RUIDO_REPETICAO_DB),
                    "comparador_db_max_0_132_dp_sementes": comparador,
                    "dp_entre_sorteios_20_semente42_db": dp20,
                    "razao_dp_sorteios20_sobre_comparador": pm[tipo]["razao_dp_sobre_comparador"],
                    "razao_dp_sorteios20_sobre_dp_sementes": razao(dp20, dp_sem),
                    "leitura_aritmetica": pm[tipo],
                    "condicao2_sigma2_sorteio_ge_sigma2_semente": (None if dec is None else dec["condicao2_sigma2_sorteio_ge_sigma2_semente"])}
            s = sinalizar_celula(pm)
            c2s = [ent[t]["condicao2_sigma2_sorteio_ge_sigma2_semente"] for t in ("gnn", "mlp")]
            s["condicao2_gnn_e_mlp"] = None if any(v is None for v in c2s) else bool(all(c2s))
            if s["condicao2_gnn_e_mlp"] is None:
                s["indeterminada_dp_nao_calculavel"] = True
            s["paridade_le_0_117"] = bool(cels1[cel]["paridade_sentinela"]["mediana_le_0_117"])
            cel_out[cel] = ent
            flags[cel] = s
        alt = {}
        for cel in CELULAS_POR_BLOCO[2]:
            pm_alt = {}
            for tipo in ("gnn", "mlp"):
                qms = [cel_out[c2][tipo]["decomposicao_um_fator_sementes_aninhadas_no_sorteio"] for c2 in CELULAS_POR_BLOCO[2]]
                if any(q is None for q in qms):
                    pm_alt[tipo] = avaliar_condicoes_modelo(cel_out[cel][tipo]["dp_entre_sorteios_20_semente42_db"], None)
                    continue
                tot = sum(q["n_sorteios"] for q in qms)
                # seed SD pooled over the two Q1 cells, weighted by their number of draws
                dp_pool = float(np.sqrt(sum(q["qm_dentro"] * q["n_sorteios"] for q in qms) / tot))
                pm_alt[tipo] = avaliar_condicoes_modelo(cel_out[cel][tipo]["dp_entre_sorteios_20_semente42_db"], max(RUIDO_REPETICAO_DB, dp_pool))
            alt[cel] = sinalizar_celula(pm_alt)
            alt[cel]["por_modelo"] = pm_alt
        return {"cels1": cels1, "bloco2": cel_out, "flags": flags, "alternativa_comparador_pooled_entre_celulas": alt}

    def montar_meta(n: int) -> dict:
        """Common metadata of an output file: script and input hashes, command, thresholds, global checks."""
        return {"artefato": f"agregado_G1_v8_bloco{n}", "versao_agregador": 8,
                "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
                "script_original_copiado": {"caminho": "scripts/v3_G1_agregar_v7.py", "sha256": SHA_V7, "avo_v6_sha256": SHA_V6, "avo_v5_sha256": SHA_V5, "avo_v4_sha256": SHA_V4, "avo_v3_sha256": SHA_V3, "avo_v2_sha256": SHA_V2, "avo_v1_sha256": SHA_ORIGINAL_V1},
                "comando": [sys.executable] + list(argv_efetivo), "comando_texto": " ".join([sys.executable] + list(argv_efetivo)),
                "data": datetime.now().astimezone().isoformat(timespec="seconds"),
                "criterio_sha256": sha_crit, "adendo_sha256": sha_adendo, "adendo2_sha256": sha_adendo2, "adendo3_sha256": sha_adendo3, "adendo4_sha256": sha_adendo4, "script_v3_sha256_copiado": SHA_V3, "script_v2_sha256_avo": SHA_V2, "plano_sha256": sha_plano,
                "status_lote_veredito": status.get("veredito_lote"),
                "modo_teste": bool(modo_teste), "raiz": str(P.RAIZ),
                "limiares_do_criterio_db": {"ruido_repeticao_gnn_0_132": RUIDO_REPETICAO_DB, "ruido_sentinela_0_117": RUIDO_SENTINELA_DB,
                                            "fator_condicao1": FATOR_CONDICAO1, "fator_delimita": FATOR_DELIMITA, "k_min_bloco3": K_MIN_BLOCO3},
                "nota_0_132": "0,132 dB e uma diferenca absoluta entre duas repeticoes, nao um dp: o limiar e conservador (adendo 2)",
                "conferencias_item7": {"scripts_do_lote": prov_scripts,
                                       "criterio_x_adendo": "sha256 do criterio igual ao citado no adendo",
                                       "plano_x_lista_2_1_e_g_b": "ok"}}

    def fechar_conferencias(out: dict, labels: list) -> None:
        """Attach the summary of the per-run checks, including the .npz vs run JSON relative differences."""
        usados = {lbl: conferidas[lbl] for lbl in labels if lbl in conferidas}
        carregadas = [lbl for lbl in labels if lbl in cache]
        todas = bool(carregadas) and all(conferidas.get(lbl, {}).get("passou") is True for lbl in carregadas)
        out["conferencias_item7"]["corridas_conferidas"] = {
            "n": len(usados),
            "itens": ["run_label", "dataset_nome/sha256/flag_manifest_v4", "g=10 e b=2 no run JSON (config, geometria, split)",
                      "g=10 e b=2 no sidecar shim_g10 (pedido, efetivo, flags, n_nos)", "seed e split_seed x plano_G1.json",
                      "hash dos indices do .npz x particoes.test.idx_sha256_global", "config x A4 (fora de grid_km, seed, split_seed, rotulos e caminhos)",
                      "script/decoder/wrapper sha x A4",
                      "n_test/n_validos/n_todos x 2.1 (C1)", "nomes unicos run_/predicoes_/shim_g10_<label> (C3)",
                      "amarracao .npz x run JSON (C3)", "sem_validos sem .npz: run JSON + 2.1 validos==0 (C2)"],
            "n_corridas_carregadas": len(carregadas),
            "amarracao_por_corrida": {lbl: v.get("amarracao") for lbl, v in usados.items() if v.get("amarracao")},
            "amarracao_dif_rel_maxima": {k: max([v["amarracao"][k] for v in usados.values() if v.get("amarracao") and v["amarracao"].get(k) is not None], default=None) for k in ("dif_rel_mae", "dif_rel_rmse")},
            "todas_passaram": todas,
            "n_com_npz": sum(1 for v in usados.values() if v.get("com_npz")),
            "n_sem_validos_sem_npz_conferidas_C2": sum(1 for v in usados.values() if not v.get("com_npz")),
            "sem_validos_parcial_adendo3": {lbl: {"dataset_bytes": v.get("dataset_bytes"), "n_test_parcial_igual_2_1": v.get("n_test_parcial_igual_2_1")} for lbl, v in usados.items() if v.get("parcial_sem_validos_adendo3")},
            "nao_treinavel_parcial_adendo4": {lbl: {k: v.get(k) for k in ("dataset_bytes", "n_test_parcial_igual_2_1", "validos_2_1", "particoes_nomeadas_no_log", "guarda_particoes_sem_aresta_antena")} for lbl, v in usados.items() if v.get("parcial_nao_treinavel_adendo4")},
            "nao_treinavel_com_npz_conferido_e_excluido": sorted(lbl for lbl in carregadas if lbl in excluidos_com_npz),
            "amarracao_npz": "MAE e RMSE de RSSI (todos os nos) do .npz x selecao.test_no_melhor_ckpt (tol. 10 % / 3 %), n_nos, n_pl_alvo_valido; nenhum artefato do lote grava sha do .npz",
            "sem_artefatos_sem_validos_registrados": [lbl for lbl in labels if cache.get(lbl, {}).get("sem_predicao")]}

    saidas, recusas = {}, {}
    alvo = [args.bloco] if args.bloco else [1, 2, 3]

    if args.so_conferir:
        feitas = 0
        for n in alvo:
            for c in [c for c in plano if c["bloco"] == n]:
                if completa(c):
                    carregar(c)
                    feitas += 1
                    print(f"conferencia ok: {c['run_label']}")
        print(f"--so-conferir: {feitas} corridas completas conferidas (nada gravado; MAE/RMSE recalculados so para a amarracao e nao impressos)")
        return 0

    for n in alvo:
        try:
            if n == 1:
                idx = indices_completos(1)
                if not all(idx.values()):
                    raise Incompleto(f"{sum(1 for c in plano if c['bloco'] == 1 and not completa(c))} corridas sem resultado")
                cels = metricas_bloco_simples(1, sorted(idx))
                out = montar_meta(1)
                out["celulas"] = cels
                out["leitura_aritmetica_contra_piso_0_132"] = sinais_simples(cels)
                out["nota"] = ("condicao 1 das celulas Q1 usa o comparador max(0,132; dp entre sementes) do bloco 2: ver "
                               "agregado_G1_v8_bloco2.json; aqui so a comparacao com o piso 0,132")
                out["contagem_sorteios_sem_validos_por_celula"] = {c: b["n_sorteios_sem_validos"] for c, b in cels.items()}
                out["contagem_sorteios_nao_treinaveis_por_celula"] = {c: b["n_sorteios_nao_treinaveis"] for c, b in cels.items()}
                out["contagem_sorteios_usaveis_por_celula"] = {c: b["n_sorteios_usaveis"] for c, b in cels.items()}
                fechar_conferencias(out, [c["run_label"] for c in plano if c["bloco"] == 1])
            elif n == 2:
                q1 = avaliar_q1()
                out = montar_meta(2)
                out["celulas"] = q1["bloco2"]
                celulas_flags = {c: q1["flags"][c] for c in q1["flags"]}
                out["leitura_aritmetica_por_celula"] = celulas_flags
                out["alternativa_comparador_pooled_entre_celulas"] = {
                    "nota": "leitura alternativa (nao a primaria) do adendo 2: dp pooled entre sementes juntando as duas celulas Q1, por modelo",
                    "celulas": q1["alternativa_comparador_pooled_entre_celulas"]}
                out["contagem_sorteios_sem_validos_por_celula"] = {c: b["n_sorteios_sem_validos"] for c, b in q1["cels1"].items()}
                out["contagem_sorteios_nao_treinaveis_por_celula"] = {c: b["n_sorteios_nao_treinaveis"] for c, b in q1["cels1"].items()}
                out["contagem_sorteios_usaveis_por_celula"] = {c: b["n_sorteios_usaveis"] for c, b in q1["cels1"].items()}
                out["paridade_sentinela_celulas_q1"] = {c: q1["cels1"][c]["paridade_sentinela"] for c in q1["cels1"]}
                out["contagem_mecanica_ramos_so_celulas_q1_2_de_2"] = contar_ramos(celulas_flags, list(celulas_flags))
                fechar_conferencias(out, [c["run_label"] for c in plano if c["bloco"] in (1, 2)])
            else:
                idx3 = indices_completos(3)
                completo = all(idx3.values())
                if completo:
                    if status.get("veredito_lote") != "concluido":
                        raise Incompleto(f"complete block 3 refused: veredito_lote = {status.get('veredito_lote')!r} (expected 'concluido' in the final status of the resumed batch)")
                    usados = sorted(idx3)
                    modo = "completo"
                else:
                    if not args.bloco3_truncado:
                        raise Incompleto(f"{sum(1 for c in plano if c['bloco'] == 3 and not completa(c))} corridas sem resultado "
                                         "(use --bloco3-truncado so se o lote terminou e o bloco segue incompleto)")
                    if status.get("veredito_lote") is None:
                        raise Incompleto("--bloco3-truncado refused: lote_G1_status.json has no veredito_lote (batch still running)")
                    if retomada_em_curso_possivel:
                        raise Incompleto("--bloco3-truncado refused: a resumed batch may still be running (retomadas present and veredito_lote == veredito_lote_anterior)")
                    usados = []
                    for i in sorted(idx3):
                        if not idx3[i]:
                            break
                        usados.append(i)
                    modo = "truncado_prefixo_contiguo"
                if not usados:
                    raise Incompleto("bloco 3: nenhum sorteio com as 4 corridas completas")
                cels3 = metricas_bloco_simples(3, usados)
                if modo != "completo":
                    ruins = {c: b["por_modelo"]["gnn"]["n_sorteios_com_validos"] for c, b in cels3.items()
                             if min(b["por_modelo"]["gnn"]["n_sorteios_com_validos"], b["por_modelo"]["mlp"]["n_sorteios_com_validos"]) < K_MIN_BLOCO3}
                    if ruins:
                        raise Incompleto(f"bloco 3 truncado (k={len(usados)}) com menos de {K_MIN_BLOCO3} sorteios com nos validos "
                                         f"em {sorted(ruins)}: bloco 3 nao entra (regra passa a 2 de 2 nas celulas Q1)")
                out = montar_meta(3)
                out["bloco3_modo"] = modo
                out["bloco3_k_sorteios_usados"] = len(usados)
                out["bloco3_indices_sorteio_usados"] = usados
                out["bloco3_bloqueados_depois_do_prefixo"] = [i for i in sorted(idx3) if i not in usados]
                out["comparador_celulas_q3"] = "0,132 dB (ancora medida a 5 km; adendo 2)"
                out["celulas"] = cels3
                sin3 = sinais_simples(cels3)
                out["leitura_aritmetica_por_celula"] = sin3
                out["contagem_sorteios_sem_validos_por_celula"] = {c: b["n_sorteios_sem_validos"] for c, b in cels3.items()}
                out["contagem_sorteios_nao_treinaveis_por_celula"] = {c: b["n_sorteios_nao_treinaveis"] for c, b in cels3.items()}
                out["contagem_sorteios_usaveis_por_celula"] = {c: b["n_sorteios_usaveis"] for c, b in cels3.items()}
                labels = [c["run_label"] for c in plano if c["bloco"] == 3 and c["indice_sorteio"] in set(usados)]
                try:
                    q1 = avaliar_q1()
                    todas = {**q1["flags"], **sin3}
                    out["contagem_mecanica_ramos_4_celulas"] = contar_ramos(todas, list(q1["flags"]))
                    labels += [c["run_label"] for c in plano if c["bloco"] in (1, 2)]
                except Incompleto as e:
                    out["contagem_mecanica_ramos_4_celulas"] = {"disponivel": False, "motivo": f"celulas Q1 indisponiveis: {e}"}
                fechar_conferencias(out, labels)
            saidas[n] = out
        except Incompleto as e:
            recusas[n] = str(e)

    for n, motivo in recusas.items():
        print(f"bloco {n}: RECUSADO ({motivo}); nao grava resultado parcial")
    for n, out in saidas.items():
        path = P.G1 / f"agregado_G1_v8_bloco{n}.json"
        # atomic write: temporary file, then rename
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        tmp.replace(path)
        print(f"bloco {n}: gravado {path}")
    return 2 if recusas else 0


if __name__ == "__main__":
    sys.exit(main())
