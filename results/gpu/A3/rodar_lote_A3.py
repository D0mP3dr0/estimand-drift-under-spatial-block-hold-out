#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launcher of batch A3: GNN versus MLP parity runs over training seeds.

Runs the 80 runs listed in `criterio_A3.json` (field `corridas`, this folder): 8 cells x 5
training seeds (42-46) x {GNN, MLP}, all at split seed 42. The fixed configuration comes from
`config_fixa_por_corrida` (8 epochs, whole graph through memory mapping, no warm start). Each
run calls `train_gnn_v3.py` or `train_mlp_v3.py` as a subprocess with the interpreter that runs
this launcher (use the CUDA environment). The runs are read by the A3 aggregator
(analysis/v3_A3_agregar.py) and by the between-draw versus between-seed check.

Resume and guards: runs already finished with return code 0 are skipped. The time budget is
checked per model type: the next run is started only if the elapsed time plus 1.3 x the longest
finished run of the same type (or PROJECAO_INICIAL_S[type] before any) stays under
ORCAMENTO_GPU_S. The launcher also checks that at least VRAM_LIVRE_MINIMA_MIB of GPU memory is
free before each run. A run in progress is never interrupted.

Outputs: one evidence folder per run in this directory, `log_<run_label>.txt`, and
`lote_A3_status.json`. The training scripts are taken from ../modelo_v3 relative to this file
(training/modelo_v3/ in this repository).
Usage: python rodar_lote_A3.py   Exit codes: 0 done, 1 run failed, 2 budget stop,
3 not enough free VRAM, 4 a run returned 3 (dataset SHA-256 differs from the manifest).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

A3_DIR = Path(__file__).resolve().parent
MODELO_V3_DIR = A3_DIR.parent / "modelo_v3"
CRITERIO_PATH = A3_DIR / "criterio_A3.json"
STATUS_PATH = A3_DIR / "lote_A3_status.json"

ORCAMENTO_GPU_S = 840 * 60  # hard time budget of the batch, seconds
# Per-type run-time estimate before the first run of that type finishes (A2c measured about 835 s GNN, 104 s MLP).
PROJECAO_INICIAL_S = {"gnn": 20 * 60, "mlp": 4 * 60}
VRAM_LIVRE_MINIMA_MIB = 1024

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


def vram_livre_mib() -> float:
    """Free memory of the first GPU in MiB from nvidia-smi; 0 if it cannot be read (blocks the next run)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()[0]
        return float(out)
    except Exception as e:  # pragma: no cover - defensive guard
        print(f"[lote_A3] AVISO: nao consegui ler VRAM livre via nvidia-smi ({e}); "
              f"assumindo 0 MiB (guarda conservadora).", flush=True)
        return 0.0


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
    duracoes_ok = {"gnn": [], "mlp": []}
    for c in status["corridas"]:
        if c.get("rc") == 0:
            duracoes_ok.setdefault(c["tipo"], []).append(c["tempo_s"])

    for corrida in corridas_planejadas:
        run_label = corrida["run_label"]
        if run_label in ja_feitas:
            print(f"[lote_A3] {run_label} ja concluida (rc=0) -- pulando (retomavel).", flush=True)
            continue

        tipo = corrida["tipo"]
        cidade = corrida["cidade"]
        quadrante = corrida["quadrante"]
        seed = corrida["seed_treino"]

        decorrido = time.perf_counter() - t0_lote
        obs = duracoes_ok.get(tipo) or []
        estimativa_proxima = (max(obs) * 1.3) if obs else PROJECAO_INICIAL_S[tipo]
        if decorrido + estimativa_proxima > ORCAMENTO_GPU_S:
            print(f"[lote_A3] ORCAMENTO: decorrido={decorrido:.0f}s + estimativa({tipo})={estimativa_proxima:.0f}s "
                  f"> teto={ORCAMENTO_GPU_S}s -- NAO inicia {run_label}. Parando aqui (abortado_por_orcamento).",
                  flush=True)
            status["veredito_lote"] = "abortado_por_orcamento"
            status["tempo_total_lote_s"] = decorrido
            salvar_status(status)
            return 2

        vram_livre = vram_livre_mib()
        if vram_livre < VRAM_LIVRE_MINIMA_MIB:
            print(f"[lote_A3] VRAM livre={vram_livre:.0f} MiB < {VRAM_LIVRE_MINIMA_MIB} MiB -- "
                  f"guarda do protocolo (PLANO A: 'abortar lote se VRAM livre < 1 GB'). "
                  f"NAO inicia {run_label}. Parando aqui.", flush=True)
            status["veredito_lote"] = "abortado_por_vram"
            status["tempo_total_lote_s"] = decorrido
            salvar_status(status)
            return 3

        script = "train_gnn_v3.py" if tipo == "gnn" else "train_mlp_v3.py"
        argv = [
            PYTHON, str(MODELO_V3_DIR / script),
            "--cidade", cidade,
            "--quadrante", quadrante,
            "--seed-treino", str(seed),
            "--split-seed", str(cfg["split_seed"]),
            "--epochs", str(cfg["epochs"]),
            "--evid-dir", str(A3_DIR),
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

        print(f"[lote_A3] iniciando {run_label} (vram_livre={vram_livre:.0f} MiB): {' '.join(argv)}", flush=True)
        iniciado_em = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        log_path = A3_DIR / f"log_{run_label}.txt"
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(argv, stdout=logf, stderr=subprocess.STDOUT,
                                   cwd=str(MODELO_V3_DIR), env=env)
        tempo_s = time.perf_counter() - t0
        concluido_em = datetime.now(timezone.utc).isoformat()

        registro = {
            "run_label": run_label, "tipo": tipo, "cidade": cidade, "quadrante": quadrante,
            "seed_treino": seed, "comando": argv, "rc": proc.returncode,
            "tempo_s": tempo_s, "iniciado_em": iniciado_em, "concluido_em": concluido_em,
            "log": str(log_path),
        }
        status["corridas"] = [c for c in status["corridas"] if c["run_label"] != run_label] + [registro]
        salvar_status(status)

        if proc.returncode == 0:
            duracoes_ok.setdefault(tipo, []).append(tempo_s)
            print(f"[lote_A3] {run_label} OK em {tempo_s:.1f}s", flush=True)
        elif proc.returncode == 3:
            # Return code 3 from the trainer: the dataset SHA-256 does not match the manifest; the batch stops.
            print(f"[lote_A3] {run_label} ABORTOU rc=3 (sha256 do dataset NAO bate com manifest v4 -- "
                  f"achado de proveniencia, ver {log_path}). Parando o lote aqui.", flush=True)
            status["veredito_lote"] = "abortado_por_proveniencia"
            status["tempo_total_lote_s"] = time.perf_counter() - t0_lote
            salvar_status(status)
            return 4
        else:
            print(f"[lote_A3] {run_label} FALHOU rc={proc.returncode} -- ver {log_path}", flush=True)
            status["veredito_lote"] = "abortado_por_falha_corrida"
            status["tempo_total_lote_s"] = time.perf_counter() - t0_lote
            salvar_status(status)
            return 1

    status["veredito_lote"] = "concluido"
    status["tempo_total_lote_s"] = time.perf_counter() - t0_lote
    salvar_status(status)
    print("[lote_A3] TODAS as corridas planejadas foram concluidas.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
