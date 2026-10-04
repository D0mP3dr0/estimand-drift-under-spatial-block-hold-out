#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shim that runs the third-campaign training wrappers at block size 10 km and buffer 2 km.

The wrappers `modelo_v3/train_gnn_v3.py` and `modelo_v3/train_mlp_v3.py` do not expose
the block geometry. This shim imports the selected wrapper unchanged and, at run time,
replaces `v3_common.carregar_modulo_congelado` with a version that returns the same
frozen trainer module with its `main` wrapped to append `--grid-km 10 --buffer-km 2`.
No file of `modelo_v3/` or of the frozen trainers (`training/frozen/` in this
repository; `FROZEN_DIR` below) is modified. Before running, the SHA-256 of the two
wrappers, `v3_common.py`, `rf_decoder_v3.py` and the two frozen trainers is compared with
`SHA_ESPERADOS`; any mismatch exits with code 6.

After a successful run it reads the run record and writes, next to it and without
editing it, `<evid-dir>/<run-label>/shim_g10_<run-label>.json` with the observed digests,
the effective g and b recorded by the trainer, and whether the number of nodes in the
prediction file equals the number of test nodes retained after the buffer. Exit code 7
if the effective g/b differ from 10/2 or the node counts differ.

Usage: python train_v3_g10.py --modelo gnn|mlp --evid-dir DIR --run-label LABEL
       [other wrapper arguments]   (do not pass --grid-km/--buffer-km)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

GRID_KM = 10.0
BUFFER_KM = 2.0

MODELO_V3_DIR = Path(__file__).resolve().parent.parent / "modelo_v3"
FROZEN_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/FIRST_RESPONSE_REVIEW_IEEE_ACESSES/"
                  "EVIDENCIA_RESUBMISSAO/scripts")
# Digests of the wrapped files; the same values appear in the run records of campaign A4.
SHA_ESPERADOS = {
    MODELO_V3_DIR / "train_gnn_v3.py": "903b1ffaa19f7d6d732e958516a9e3dd98110cc3ac0f1aed704582e6e23a914f",
    MODELO_V3_DIR / "train_mlp_v3.py": "86484b957e8507d8626a3e0ac386ce8c87fdca33f2548a814b02bee26c9085b8",
    MODELO_V3_DIR / "v3_common.py": "57ffd38d869450a63d5263b101b440c02e19b738447c7013e5e9d7e48eff0268",
    MODELO_V3_DIR / "rf_decoder_v3.py": "645f169755926b1525b7a8082da7e8dbfcb8d4bdbac1dfbf3aa5c45884fbfe0a",
    FROZEN_DIR / "train_gnn_c0_spatial.py": "6f955629cde164f2843f454e1ebf6977292fd80647ef48ecd0df31e464c42445",
    FROZEN_DIR / "train_mlp_c0_spatial.py": "4b75093f52e8b7fdec45c7f2b8c5bcd68c38093c420697927b49fa99a474b31c",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verificar_sha() -> dict:
    """Return the observed SHA-256 of every wrapped file; exit with code 6 on any mismatch."""
    obs = {str(p): sha256(p) for p in SHA_ESPERADOS}
    ruins = {str(p): (obs[str(p)], esp) for p, esp in SHA_ESPERADOS.items() if obs[str(p)] != esp}
    if ruins:
        print(f"[shim_g10] RECUSADO: sha256 divergente dos registrados: {ruins}", flush=True)
        sys.exit(6)
    return obs


def main() -> int:
    sha_obs = verificar_sha()
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--modelo", choices=("gnn", "mlp"), required=True)
    ap.add_argument("--evid-dir", required=True)
    ap.add_argument("--run-label", required=True)
    ap.add_argument("--grid-km", default=None)
    ap.add_argument("--buffer-km", default=None)
    known, resto = ap.parse_known_args()
    if known.grid_km is not None or known.buffer_km is not None:
        print("[shim_g10] RECUSADO: g e b sao fixos (10/2) neste shim; nao passe --grid-km/--buffer-km.", flush=True)
        return 6
    # Arguments for the wrapper: everything except --modelo.
    argv_wrapper = ["--evid-dir", known.evid_dir, "--run-label", known.run_label] + resto

    sys.path.insert(0, str(MODELO_V3_DIR))
    import v3_common as v3  # same module object the wrapper imports as `v3`

    orig_carregar = v3.carregar_modulo_congelado

    def carregar_g10(script_path, nome_modulo):
        """Load the frozen trainer as before, with `main` wrapped to append the fixed g and b."""
        mod = orig_carregar(script_path, nome_modulo)
        main_orig = mod.main

        def main_g10(argv=None):
            argv = list(argv or [])
            if "--grid-km" in argv or "--buffer-km" in argv:
                raise RuntimeError("wrapper ja passou --grid-km/--buffer-km; shim nao deve duplicar")
            argv += ["--grid-km", str(GRID_KM), "--buffer-km", str(BUFFER_KM)]
            print(f"[shim_g10] mod.main recebe + --grid-km {GRID_KM} --buffer-km {BUFFER_KM}", flush=True)
            return main_orig(argv)

        mod.main = main_g10
        return mod

    v3.carregar_modulo_congelado = carregar_g10

    if known.modelo == "gnn":
        import train_gnn_v3 as wrapper
    else:
        import train_mlp_v3 as wrapper
    sys.argv = [wrapper.__file__] + argv_wrapper
    rc = wrapper.main()
    if rc != 0:
        return rc

    # Post-run check (the run record is read, never edited).
    import numpy as np
    pasta = Path(known.evid_dir) / known.run_label
    run_json = pasta / f"run_{known.run_label}.json"
    npz = pasta / f"predicoes_{known.run_label}.npz"
    with open(run_json, "r", encoding="utf-8") as f:
        rec = json.load(f)
    cfg, geo = rec.get("config", {}), rec.get("geometria", {})
    n_teste_run = (rec.get("split", {}).get("n_nos_apos_buffer", {}) or {}).get("test")  # test nodes retained after the buffer
    n_npz = int(np.load(npz)["idx_global"].shape[0]) if npz.exists() else None
    ok_gb = (cfg.get("grid_km") == GRID_KM and cfg.get("buffer_km") == BUFFER_KM
             and geo.get("grid_km_usado") == GRID_KM and geo.get("buffer_km_usado") == BUFFER_KM
             and (rec.get("split", {}).get("grid_km") == GRID_KM) and (rec.get("split", {}).get("buffer_km") == BUFFER_KM))
    ok_npz = (n_npz is not None and n_npz == n_teste_run)
    side = {"artefato": "shim_g10", "modelo": known.modelo, "run_label": known.run_label,
            "grid_km_pedido": GRID_KM, "buffer_km_pedido": BUFFER_KM,
            "efetivo": {"config.grid_km": cfg.get("grid_km"), "config.buffer_km": cfg.get("buffer_km"),
                        "geometria.grid_km_usado": geo.get("grid_km_usado"), "geometria.buffer_km_usado": geo.get("buffer_km_usado"),
                        "split.grid_km": rec.get("split", {}).get("grid_km"), "split.buffer_km": rec.get("split", {}).get("buffer_km")},
            "g_b_efetivos_conferem": bool(ok_gb),
            "n_nos_teste_retidos_run_json": n_teste_run, "n_nos_npz": n_npz, "npz_igual_teste_retido": bool(ok_npz),
            "sha256_observados": sha_obs}
    with open(pasta / f"shim_g10_{known.run_label}.json", "w", encoding="utf-8") as f:
        json.dump(side, f, indent=1, ensure_ascii=False)
    print(f"[shim_g10] g/b efetivos conferem={ok_gb}; npz({n_npz}) == teste retido do run JSON({n_teste_run}): {ok_npz}", flush=True)
    return 0 if (ok_gb and ok_npz) else 7


if __name__ == "__main__":
    sys.exit(main())
