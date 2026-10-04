"""Corrected reference target for one tile, with a reproduction check of the stored target.

Reads an enriched transfer dataset (output of enrich_rf_targets.py) and writes a copy whose
terrain.y carries three corrections applied together, all in the target-generation code:

 1. Terrain term from the measured slope. prepare_transfer_dataset_v19.py feeds the
    terrain term of generate_realistic_coverage.py (A_terrain = 20 s, plus 30 (s - 0.1)
    above s = 0.1, clipped to [0, 30] dB) with a slope s drawn from uniform(0.02, 0.15).
    The draw is regenerated here exactly as graph_build/preparar_transfer_seed42.py makes
    it (np.random.seed(42) before each quadrant) and replaced by the measured slope
    (feature column 1, clipped at 0), a dimensionless gradient (1 = 45 degrees).
 2. Ruggedness column in the diffraction term. enrich_rf_targets.py reads feature column
    5 as TRI; in the base graph column 5 holds TPI, column 6 TRI and column 7 roughness.
    The corrected diffraction loss uses column 6 (times elev_std, in metres).
 3. Units of the obstacle-height sum. enrich_rf_targets.py adds the roughness term in
    normalized units to a ruggedness term in metres; the corrected version converts the
    roughness (column 7) to metres with the same per-tile elev_std.

The target can be corrected by difference, without recomputing the propagation:
  - A_terrain depends only on the terrain node, not on the antenna, so it does not change
    which antenna gives the highest received power, and it enters the total loss as an
    additive term. Column 0 (path loss) therefore gains delta and column 3 (received
    power) loses delta exactly, with delta = A_terrain(measured) - A_terrain(drawn).
  - Column 2 (diffraction loss) is a per-node channel computed from the node's own
    features; it is recomputed in full by calling enrich_rf_targets.compute_diffraction_loss
    unchanged.
Nodes whose column 0 holds the 300 dB sentinel (no antenna in range) keep columns 0 and
3, and columns 1 and 4 are not changed.

Reproduction check: before correcting, column 2 is recomputed with the inputs that
enrich_rf_targets.py uses (column 5 times elev_std, and column 6 in normalized units) and
compared bit for bit with the stored column (key reproducao_bate_com_a_publicada).

Inputs: the enriched tile (--src, or the first existing candidate in the configured
directories) and the tile's base graph, which supplies elev_std and elev_mean.
Outputs: the corrected tile (input name with the suffix _cftudo unless --dst is given;
the input file is not modified) and the record
dados/alvo/contrafactual_alvo_completo_<city>_<quadrant>.json with summary statistics of
each correction and the SHA-256 of this script.

Usage: python contrafactual_alvo_completo.py --city bauru --quad Q1 [--src F] [--dst F]

FREQ_MHZ (1800 MHz) is the frequency passed to compute_diffraction_loss, the default of
enrich_rf_targets.py.

The directory constants GD, V3 and PROJ hold the authors' paths; PROJ must contain
enrich_rf_targets.py (directly or under data_raw/) for the import in main().
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

EVID = Path(__file__).resolve().parents[1]
OUT_DIR = EVID / "dados" / "alvo"
GD = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2\graph_data")
V3 = Path(r"F:\TOPO_RF_DOWNLOAD_DRIVE\graph_data_v3")
PROJ = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2")
sys.path.insert(0, str(PROJ / "data_raw"))
sys.path.insert(0, str(PROJ))

SENTINELA = 300.0  # path loss (dB) written for nodes no antenna reaches
SEED = 42  # seed of the slope draw, as used when the tiles were built
FREQ_MHZ = 1800.0
IDX_SLOPE, IDX_TPI_COL5, IDX_TRI_COL6, IDX_ROUGH_COL7 = 1, 5, 6, 7


def a_terrain(slope: np.ndarray) -> np.ndarray:
    """Terrain term A_terrain in dB, the same formula as generate_realistic_coverage.py:107-116."""
    tf = slope * 20.0
    tf = np.where(slope > 0.1, tf + (slope - 0.1) * 30.0, tf)
    return np.clip(tf, 0, 30)


def resumo(a: np.ndarray) -> dict:
    """Summary statistics (min, median, p95, max, mean, std) of an array."""
    return {"min": float(a.min()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()),
            "media": float(a.mean()), "std": float(a.std())}


def carimbo(c: str, q: str):
    """Return (elev_std, elev_mean) in metres from the tile's base graph, or (None, None).

    The base graph is searched in V3 first (terrain maps regenerated from the corrected
    mosaic, with their own normalization), then in GD, then in the GRAPH_V19 folder.
    """
    for d in (V3, GD, Path(r"D:\ARPIA_RF\GRAPH_V19")):
        p = d / f"{c}_v19_{q}_gpu.pt"
        if p.exists():
            sd = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
            nm = getattr(sd, "normalization", None)
            out = ((float(nm["elev_std"]), float(nm["elev_mean"]))
                   if nm and "elev_std" in nm and "elev_mean" in nm else (None, None))
            del sd
            return out
    return None, None


def main() -> int:
    """Correct one tile; returns 0 on success and 1 if an input is missing."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="lins")
    ap.add_argument("--quad", default="Q1")
    ap.add_argument("--src", default="",
                    help="caminho do tile a corrigir; se omitido, procura nos "
                         "diretorios conhecidos")
    ap.add_argument("--dst", default="",
                    help="caminho de saida; se omitido, acrescenta _cftudo ao nome")
    args = ap.parse_args()
    c, q = args.city, args.quad

    import enrich_rf_targets as E

    # Source tile: --src if given, otherwise the first existing candidate among the two
    # file names (_enriched_v2.pt, _enriched.pt) in the GD and V3 directories.
    if args.src:
        src = Path(args.src)
    else:
        candidatos = [GD / f"transfer_dataset_{c}_v19_{q}_enriched_v2.pt",
                      V3 / f"transfer_dataset_{c}_v19_{q}_enriched.pt",
                      GD / f"transfer_dataset_{c}_v19_{q}_enriched.pt"]
        src = next((p for p in candidatos if p.exists()), candidatos[0])
    dst = (Path(args.dst) if args.dst
           else src.with_name(src.stem + "_cftudo" + src.suffix))
    if not src.exists():
        print(f"ABORTA: {src} ausente")
        return 1
    es, em = carimbo(c, q)
    if es is None:
        print("ABORT: elev_std/elev_mean normalization metadata missing")
        return 1

    print(f"lendo {src.name} | elev_std={es}", flush=True)
    d = torch.load(src, map_location="cpu", weights_only=False)
    t = d["terrain"]
    # features_raw: the raw node features stored by prepare_transfer_dataset_v19.py.
    fr = t.features_raw
    x = (fr.numpy() if torch.is_tensor(fr) else np.asarray(fr)).astype(np.float32)
    y = (t.y.numpy() if torch.is_tensor(t.y) else np.asarray(t.y)).astype(np.float32).copy()
    dist = t.dist_nearest_m.float()
    n = x.shape[0]
    rec = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "tile": f"{c}_{q}", "origem": str(src), "saida": str(dst),
           "elev_std": es, "correcoes": {}}

    # Correction 1: terrain term from the measured slope instead of the seeded draw; the
    # draw is regenerated with the seed used when the tile was built.
    np.random.seed(SEED)
    slope_sorteado = np.random.uniform(0.02, 0.15, n).astype(np.float32)
    slope_real = np.clip(x[:, IDX_SLOPE].astype(np.float64), 0, None)
    at_old, at_new = a_terrain(slope_sorteado.astype(np.float64)), a_terrain(slope_real)
    delta = at_new - at_old
    com_cob = y[:, 0] != SENTINELA  # nodes reached by at least one antenna
    y[com_cob, 0] = (y[com_cob, 0].astype(np.float64) + delta[com_cob]).astype(np.float32)
    if y.shape[1] > 3:
        y[com_cob, 3] = (y[com_cob, 3].astype(np.float64) - delta[com_cob]).astype(np.float32)
    rec["correcoes"]["1_A_terrain"] = {
        "sorteado": resumo(at_old), "com_slope_real": resumo(at_new),
        "delta_abs_medio_db": float(np.abs(delta[com_cob]).mean()),
        "delta_abs_max_db": float(np.abs(delta[com_cob]).max()),
        "n_nos_com_cobertura": int(com_cob.sum())}
    print(f"  1) A_terrain: {at_old.mean():.4f} -> {at_new.mean():.4f} dB "
          f"(delta medio {np.abs(delta[com_cob]).mean():.4f})", flush=True)

    # Corrections 2 and 3: TRI column (6) and roughness in metres in the diffraction term.
    col2_antes = y[:, 2].copy()
    tri_errado_m = torch.from_numpy((x[:, IDX_TPI_COL5].astype(np.float32) * es))  # as in enrich_rf_targets.py
    rough_norm = torch.from_numpy(x[:, IDX_TRI_COL6].astype(np.float32))
    col2_reproduzida = E.compute_diffraction_loss(tri_errado_m, rough_norm, dist, FREQ_MHZ).numpy()  # stored target

    tri_certo_m = torch.from_numpy((x[:, IDX_TRI_COL6].astype(np.float32) * es))
    rough_m = torch.from_numpy((x[:, IDX_ROUGH_COL7].astype(np.float32) * es))
    col2_corrigida = E.compute_diffraction_loss(tri_certo_m, rough_m, dist, FREQ_MHZ).numpy()

    rec["correcoes"]["2e3_difracao"] = {
        "col2_publicada": resumo(col2_antes.astype(np.float64)),
        "col2_reproduzida_com_o_codigo_atual": resumo(col2_reproduzida.astype(np.float64)),
        "reproducao_bate_com_a_publicada": bool(np.array_equal(col2_reproduzida, col2_antes)),
        "col2_corrigida": resumo(col2_corrigida.astype(np.float64)),
        "delta_abs_medio_db": float(np.abs(col2_corrigida - col2_antes).mean()),
        "delta_abs_max_db": float(np.abs(col2_corrigida - col2_antes).max()),
        "frac_no_piso_antes": float(np.mean(col2_antes <= 6.0206004)),  # 6.0206 dB = loss at v = 0 (zero obstacle height)
        "frac_no_piso_depois": float(np.mean(col2_corrigida <= 6.0206004))}
    y[:, 2] = col2_corrigida.astype(np.float32)
    print(f"  2+3) difracao: p50 {np.percentile(col2_antes,50):.4f} -> "
          f"{np.percentile(col2_corrigida,50):.4f} dB | no piso "
          f"{np.mean(col2_antes<=6.0206004):.2%} -> "
          f"{np.mean(col2_corrigida<=6.0206004):.2%}", flush=True)
    print(f"       (reproducao do codigo atual bate com a publicada: "
          f"{rec['correcoes']['2e3_difracao']['reproducao_bate_com_a_publicada']})",
          flush=True)

    t.y = torch.from_numpy(y)
    print(f"\ngravando {dst.name} ...", flush=True)
    torch.save(d, dst)
    rec["saida_bytes"] = dst.stat().st_size
    rec["sha256_gerador"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dest = OUT_DIR / f"contrafactual_alvo_completo_{c}_{q}.json"
    dest.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"gravado: {dst} ({dst.stat().st_size/1e9:.2f} GB)")
    print(f"registro: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
