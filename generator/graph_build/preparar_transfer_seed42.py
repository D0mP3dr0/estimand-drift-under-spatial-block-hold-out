"""Seeded driver of the per-quadrant dataset builder.

prepare_transfer_dataset_v19.py draws the per-node terrain coefficient with
np.random.uniform(0.02, 0.15) and sets no seed itself; the draw is consumed by the
propagation model of generate_realistic_coverage.py. This driver imports the builder
unchanged, sets its input and output directories, and calls np.random.seed(42) before
run_per_quadrant() for each quadrant, so that the reference targets are reproducible.

Inputs: the per-quadrant graph files in --graph-dir (and --struct-dir, if different).
Outputs: the per-quadrant dataset files in --output-dir and a JSON record of the run
(seed, SHA-256 of the builder, directories, status per quadrant).

Usage:
    python preparar_transfer_seed42.py --city campinas --graph-dir <dir> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

V2 = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2")
EVID = Path(__file__).resolve().parents[1]
OUT_JSON = EVID / "dados" / "alvo"
SEED_A_TERRAIN = 42

sys.path.insert(0, str(V2 / "01_data"))
sys.path.insert(0, str(V2 / "data_raw"))
sys.path.insert(0, str(V2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--struct-dir", default="")
    ap.add_argument("--quadrants", nargs="+", default=["Q1", "Q2", "Q3", "Q4"])
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    import prepare_transfer_dataset_v19 as P

    gd = Path(args.graph_dir)
    od = Path(args.output_dir)
    sd = Path(args.struct_dir) if args.struct_dir else gd
    od.mkdir(parents=True, exist_ok=True)
    P.GRAPH_DIR = gd
    P.OUTPUT_DIR = od
    P.STRUCT_DIR = sd

    src = V2 / "01_data" / "prepare_transfer_dataset_v19.py"
    print(f"cidade     : {args.city}")
    print(f"graph_dir  : {gd}")
    print(f"output_dir : {od}")
    print(f"seed A_terrain: {SEED_A_TERRAIN} (prepare_transfer_dataset_v19.py:141 "
          f"nao semeia sozinho)", flush=True)

    t0 = datetime.now(timezone.utc)
    resultados = {}
    for q in args.quadrants:
        np.random.seed(SEED_A_TERRAIN)
        print(f"\n--- {args.city} {q} (np.random.seed({SEED_A_TERRAIN}) aplicado) ---",
              flush=True)
        try:
            P.run_per_quadrant([q], skip_existing=args.skip_existing,
                               city=args.city.lower())
            resultados[q] = "ok"
        except Exception as e:
            resultados[q] = f"FALHOU: {type(e).__name__}: {e}"
            print(f"[ERRO] {q}: {e}", flush=True)

    rec = {
        "timestamp_utc": t0.isoformat(),
        "cidade": args.city,
        "seed_a_terrain": SEED_A_TERRAIN,
        "onde_a_semente_entra": ("np.random.seed(42) antes de run_per_quadrant; "
                                 "consumida por prepare_transfer_dataset_v19.py:141 "
                                 "(np.random.uniform 0.02-0.15) -> "
                                 "generate_realistic_coverage.py:109-116"),
        "ratificada_pelo_dono_em": "2026-09-03 (rota tudo-v2)",
        "script_original": str(src),
        "script_original_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "graph_dir": str(gd), "output_dir": str(od), "struct_dir": str(sd),
        "quadrantes": resultados,
        "segundos": round((datetime.now(timezone.utc) - t0).total_seconds(), 1),
    }
    OUT_JSON.mkdir(parents=True, exist_ok=True)
    dest = OUT_JSON / f"preparar_transfer_{args.city}.json"
    dest.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nregistro: {dest}")
    ruins = [q for q, v in resultados.items() if v != "ok"]
    return 1 if ruins else 0


if __name__ == "__main__":
    sys.exit(main())
