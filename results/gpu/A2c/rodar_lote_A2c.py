#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launcher of batch A2c: repeatability of GNN and MLP training on the canonical dataset.

Runs the 8 runs listed in the decision criterion `criterio_A2c.json` (field `corridas`): Bauru Q1
and Lins Q1, GNN and MLP, two repeats each with the same training seed and split seed (42). The
fixed per-run configuration comes from `config_fixa_por_corrida`: 8 epochs, whole graph through
memory mapping, no warm start from another run. Each run calls `train_gnn_v3.py` or
`train_mlp_v3.py` as a subprocess with the interpreter that runs this launcher, so start it from
the CUDA environment. The absolute difference between the two GNN repeats is the repeat
difference (0.132 dB) used as the noise floor of single-run comparisons.

Resume and budget: runs already finished with return code 0 are skipped. A new run is started
only if the elapsed time plus an estimate of the next run stays under ORCAMENTO_GPU_S. The
estimate is 1.3 x the longest finished run, or PROJECAO_INICIAL_S before any run has finished.
A run in progress is never interrupted.

Paths are relative to this file in the original layout: the training scripts in ../modelo_v3
(training/modelo_v3/ in this repository) and the criterion in ../../criterios/ (criteria/ here).
Outputs: one evidence folder per run in this directory, `log_<run_label>.txt`, and
`lote_A2c_status.json` (command, return code, duration and timestamps per run).
Usage: python rodar_lote_A2c.py   Exit code 0 = all runs done, 1 = a run failed, 2 = budget stop.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

A2C_DIR = Path(__file__).resolve().parent
MODELO_V3_DIR = A2C_DIR.parent / "modelo_v3"
CRITERIO_PATH = A2C_DIR.parent.parent / "criterios" / "criterio_A2c.json"
STATUS_PATH = A2C_DIR / "lote_A2c_status.json"

ORCAMENTO_GPU_S = 180 * 60  # hard time budget of the batch, seconds
PROJECAO_INICIAL_S = 20 * 60  # run-time estimate used before the first run is measured, seconds

PYTHON = sys.executable  # the training subprocesses inherit this interpreter (must be the CUDA environment)


def carregar_status() -> dict:
    if STATUS_PATH.exists():
        with open(STATUS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"iniciado_em": datetime.now(timezone.utc).isoformat(), "corridas": [],
            "orcamento_gpu_s": ORCAMENTO_GPU_S, "veredito_lote": None}


def salvar_status(status: dict) -> None:
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    tmp.replace(STATUS_PATH)  # atomic rename, so the status file is never left half-written


def main() -> int:
    with open(CRITERIO_PATH, "r", encoding="utf-8") as f:
        criterio = json.load(f)
    corridas_planejadas = criterio["corridas"]
    cfg = criterio["config_fixa_por_corrida"]

    status = carregar_status()
    ja_feitas = {c["run_label"] for c in status["corridas"] if c.get("rc") == 0}

    # Set explicitly for every run to reduce CUDA memory fragmentation, independent of the caller's shell.
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    t0_lote = time.perf_counter()
    duracoes_ok = [c["tempo_s"] for c in status["corridas"] if c.get("rc") == 0]

    for corrida in corridas_planejadas:
        run_label = corrida["run_label"]
        if run_label in ja_feitas:
            print(f"[lote_A2c] {run_label} ja concluida (rc=0) -- pulando (retomavel).", flush=True)
            continue

        decorrido = time.perf_counter() - t0_lote
        estimativa_proxima = (max(duracoes_ok) * 1.3) if duracoes_ok else PROJECAO_INICIAL_S
        if decorrido + estimativa_proxima > ORCAMENTO_GPU_S:
            print(f"[lote_A2c] ORCAMENTO: decorrido={decorrido:.0f}s + estimativa={estimativa_proxima:.0f}s "
                  f"> teto={ORCAMENTO_GPU_S}s -- NAO inicia {run_label}. Parando aqui (abortado_por_orcamento).",
                  flush=True)
            status["veredito_lote"] = "abortado_por_orcamento"
            salvar_status(status)
            return 2

        tipo = corrida["tipo"]
        cidade = corrida["cidade"]
        quadrante = corrida["quadrante"]
        script = "train_gnn_v3.py" if tipo == "gnn" else "train_mlp_v3.py"

        argv = [
            PYTHON, str(MODELO_V3_DIR / script),
            "--cidade", cidade,
            "--quadrante", quadrante,
            "--seed-treino", str(cfg["seed_treino"]),
            "--split-seed", str(cfg["split_seed"]),
            "--epochs", str(cfg["epochs"]),
            "--evid-dir", str(A2C_DIR),
            "--run-label", run_label,
            # The MLP trainer has no --hidden-dim argument; it is passed to GNN runs only.
            *(["--hidden-dim", str(cfg["hidden_dim"])] if tipo == "gnn" else []),
            "--batch-size", str(cfg["batch_size"]),
            "--lr", str(cfg["lr"]),
        ]
        if tipo == "gnn":
            argv += ["--k-antenna", str(cfg["k_antenna"]), "--k-terrain", str(cfg["k_terrain"])]
        if cfg["mmap"]:
            argv += ["--mmap"]

        print(f"[lote_A2c] iniciando {run_label}: {' '.join(argv)}", flush=True)
        iniciado_em = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        log_path = A2C_DIR / f"log_{run_label}.txt"
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(argv, stdout=logf, stderr=subprocess.STDOUT,
                                   cwd=str(MODELO_V3_DIR), env=env)
        tempo_s = time.perf_counter() - t0
        concluido_em = datetime.now(timezone.utc).isoformat()

        registro = {
            "run_label": run_label, "tipo": tipo, "cidade": cidade, "quadrante": quadrante,
            "rep": corrida["rep"], "comando": argv, "rc": proc.returncode,
            "tempo_s": tempo_s, "iniciado_em": iniciado_em, "concluido_em": concluido_em,
            "log": str(log_path),
        }
        status["corridas"] = [c for c in status["corridas"] if c["run_label"] != run_label] + [registro]
        salvar_status(status)

        if proc.returncode == 0:
            duracoes_ok.append(tempo_s)
            print(f"[lote_A2c] {run_label} OK em {tempo_s:.1f}s", flush=True)
        else:
            print(f"[lote_A2c] {run_label} FALHOU rc={proc.returncode} -- ver {log_path}", flush=True)
            status["veredito_lote"] = "abortado_por_falha_corrida"
            salvar_status(status)
            return 1

    status["veredito_lote"] = "concluido"
    status["tempo_total_lote_s"] = time.perf_counter() - t0_lote
    salvar_status(status)
    print("[lote_A2c] TODAS as 8 corridas concluidas.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
