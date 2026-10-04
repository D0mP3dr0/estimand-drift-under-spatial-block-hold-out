"""Target statistics of the corrected reference fields in the 16 cells (g = 10 km, b = 2 km).

One CPU pass over the 16 corrected target tensors (transfer_dataset_<city>_v19_<Qn>
_enriched_cftudo.pt), one tensor at a time. For each cell the script:
  * checks the tensor SHA-256 against the artifact manifest and the run record
    before loading it;
  * rebuilds the training/validation/test partition of the split-seed-42 run with the
    grid, buffer, fractions and seed stored in the run record, and aborts the cell
    unless the sorted index digests match the stored idx_sha256_global;
  * reports the sentinel fraction (target column 0 >= 299) in the whole cell and in
    each partition;
  * reports the squared Pearson correlation between valid RSSI (column 3, dBm) and
    -log10(dist_nearest_m), in the cell and in the test partition;
  * reports the MAE of the constant predictor on RSSI: median of the population
    itself (cell, test), training median applied to test, and the -110 dBm floor;
  * cross-checks the test indices, targets and sentinel flags against the stored
    per-node prediction file of the graph-free model (mlp_<city>_<Qn>_s42.npz).
The sentinel fractions of the cells come from this record.

Inputs: tensors in TENSOR_DIR, manifest MANIFEST, run records under RUNDIR,
prediction files under NPZ_DIR (absolute paths set below).
Output: OUT_RAW, rewritten after every cell; cells already finished with a matching
digest are skipped, so the run can be resumed.
Usage: python r3_alvo_pos_correcao_16celulas.py [--celula lins_Q1]
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SCRIPT_PATH = Path(__file__).resolve()

TENSOR_DIR = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")
MANIFEST = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/"
                 "manifest_mathematics_v3.jsonl")
RUNDIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
              "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/treinos")
SPATIAL_CV_BASE = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2")
OUT_RAW = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/"
                "_R2_2026-09-24/r3_alvo_pos_correcao_16celulas_RAW.json")
NPZ_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/"
               "_R2_2026-09-24/e3_predicoes")

CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
QUADRANTES = ["Q1", "Q2", "Q3", "Q4"]
PL_TARGET_MAX_VALID = 299.0  # target column 0 at or above this value marks a sentinel node
PISO_RSSI_DBM = -110.0  # RSSI floor (dBm) used as a fixed constant predictor


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_idx(idx: np.ndarray) -> str:
    """SHA-256 of the sorted int64 indices, as stored by the trainer in idx_sha256_global."""
    a = np.ascontiguousarray(np.sort(np.asarray(idx, dtype=np.int64)))
    return hashlib.sha256(a.tobytes()).hexdigest()


def sha256_self() -> str:
    return sha256_file(SCRIPT_PATH)


def corr_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; NaN when either input is constant."""
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).mean() * (b * b).mean())
    return float((a * b).mean() / d) if d > 0 else float("nan")


def latlon_graus_para_metros(pos: torch.Tensor) -> np.ndarray:
    """Local equirectangular projection used by the trainer: 111 km per degree, x scaled by cos(lat)."""
    lon = pos[:, 0].double().numpy()
    lat = pos[:, 1].double().numpy()
    lon_min, lat_min = float(lon.min()), float(lat.min())
    y = (lat - lat_min) * 111_000.0
    x = (lon - lon_min) * 111_000.0 * np.cos(np.radians(lat))
    return np.stack([x, y], axis=1)


def split_espacial_3vias(pos_m: np.ndarray, grid_km: float, buffer_km: float,
                          fracs: tuple, split_seed: int, SpatialKFold, log):
    """Three-way buffered spatial block split of the frozen trainer; returns global node
    indices per role. Val nodes within buffer_km of training are dropped, then test
    nodes within buffer_km of train U val."""
    from scipy.spatial import cKDTree

    pos_km = pos_m / 1000.0
    skf = SpatialKFold(n_splits=3, buffer_km=buffer_km,
                        grid_size_km=grid_km, random_state=split_seed)
    group_ids = skf._assign_groups(pos_km)
    grupos = np.unique(group_ids)
    rng = np.random.RandomState(split_seed)
    grupos_emb = grupos.copy()
    rng.shuffle(grupos_emb)

    n_g = len(grupos_emb)
    n_tr = max(1, int(round(fracs[0] * n_g)))
    n_va = max(1, int(round(fracs[1] * n_g)))
    if n_tr + n_va >= n_g:
        n_tr = max(1, n_g - 2)
        n_va = 1
    g_tr = grupos_emb[:n_tr]
    g_va = grupos_emb[n_tr:n_tr + n_va]
    g_te = grupos_emb[n_tr + n_va:]

    m_tr = np.isin(group_ids, g_tr)
    m_va = np.isin(group_ids, g_va)
    m_te = np.isin(group_ids, g_te)

    kw = dict(compact_nodes=False, balanced_tree=False)
    if buffer_km > 0:
        if m_tr.any() and m_va.any():
            tree_tr = cKDTree(pos_km[m_tr], **kw)
            d, _ = tree_tr.query(pos_km[m_va], k=1, workers=-1)
            idx_va = np.where(m_va)[0]
            m_va[idx_va[d < buffer_km]] = False
        m_trva = m_tr | m_va
        if m_trva.any() and m_te.any():
            tree_trva = cKDTree(pos_km[m_trva], **kw)
            d, _ = tree_trva.query(pos_km[m_te], k=1, workers=-1)
            idx_te = np.where(m_te)[0]
            m_te[idx_te[d < buffer_km]] = False

    parts = {"train": np.where(m_tr)[0].astype(np.int64),
             "val": np.where(m_va)[0].astype(np.int64),
             "test": np.where(m_te)[0].astype(np.int64)}
    return parts


def carregar_manifest_sha(basename: str) -> str:
    with open(MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("grupo") == "tensores_cftudo" and Path(r.get("caminho", "")).name == basename:
                return r["sha256"]
    raise RuntimeError(f"basename {basename} nao encontrado no manifest tensores_cftudo")


def carregar_tensor(path: Path):
    """Load a tensor file on CPU, memory-mapped when possible; returns (data, load mode)."""
    load_kw = dict(map_location="cpu", weights_only=False)
    try:
        rf = torch.load(path, mmap=True, **load_kw)
        return rf, "mmap"
    except Exception as e:
        log(f"  mmap falhou ({e!r}); tentando sem mmap")
        rf = torch.load(path, **load_kw)
        return rf, "sem_mmap"


def preditor_constante(col: np.ndarray) -> dict:
    """Constant predictor fitted in-sample: the median of the population; MAE = mean |col - median|."""
    med = float(np.median(col))
    mae = float(np.abs(col - med).mean())
    return {"constante_mediana": med, "mae_db": mae, "n": int(col.size)}


def mae_constante_fixo(col: np.ndarray, constante: float) -> dict:
    return {"constante": float(constante), "mae_db": float(np.abs(col - constante).mean()),
            "n": int(col.size)}


def medir_celula(cidade: str, quad: str) -> dict:
    """Measure one cell; returns a record whose "status" is "ok" or names the failed check."""
    basename = f"transfer_dataset_{cidade}_v19_{quad}_enriched_cftudo.pt"
    tensor_path = TENSOR_DIR / basename
    run_path = RUNDIR / f"c0c1cf_{cidade}_s42_{quad}_g10b2" / f"run_c0c1cf_{cidade}_s42_{quad}_g10b2.json"

    rec = {"cidade": cidade, "quadrante": quad, "tensor_path": str(tensor_path),
           "run_path": str(run_path), "timestamp_utc_inicio": datetime.now(timezone.utc).isoformat()}

    if not tensor_path.exists():
        rec["status"] = "erro_tensor_ausente"
        return rec
    if not run_path.exists():
        rec["status"] = "erro_run_json_ausente"
        return rec

    run_rec = json.loads(run_path.read_text(encoding="utf-8"))

    sha_manifest = carregar_manifest_sha(basename)
    sha_run_json = run_rec["dataset"]["rf_data_sha256"]
    rec["sha_manifest_vs_run_json_ok"] = (sha_manifest == sha_run_json)
    if sha_manifest != sha_run_json:
        rec["status"] = "erro_sha_manifest_diverge_de_run_json"
        rec["sha_manifest"] = sha_manifest
        rec["sha_run_json"] = sha_run_json
        return rec

    t0 = time.perf_counter()
    log(f"  sha256 de {basename} ...")
    sha_calc = sha256_file(tensor_path)
    rec["sha256_esperado"] = sha_manifest
    rec["sha256_calculado"] = sha_calc
    rec["sha256_ok"] = (sha_calc == sha_manifest)
    rec["tempo_sha_s"] = time.perf_counter() - t0
    if not rec["sha256_ok"]:
        rec["status"] = "erro_sha_tensor_divergente"
        return rec
    log(f"  sha256 OK ({rec['tempo_sha_s']:.1f}s)")

    t1 = time.perf_counter()
    rf, modo_carga = carregar_tensor(tensor_path)
    rec["modo_carga"] = modo_carga
    ty = rf["terrain"].y
    y = (torch.as_tensor(ty).float()).clone().numpy()
    dist_all = rf["terrain"].dist_nearest_m
    dist_all = torch.as_tensor(dist_all).float().clone().numpy()
    pos_deg = rf["terrain"].pos
    pos_deg = torch.as_tensor(pos_deg).float().clone()
    n_total = int(y.shape[0])
    del rf
    gc.collect()
    rec["tempo_load_s"] = time.perf_counter() - t1
    rec["n_nodes_total"] = n_total
    log(f"  carregado ({modo_carga}) em {rec['tempo_load_s']:.1f}s, n={n_total:,}")

    # Partition rebuilt with the configuration stored in the run record.
    cfg = run_rec["config"]
    grid_km = float(cfg["grid_km"])
    buffer_km = float(cfg["buffer_km"])
    fracs = tuple(float(v) for v in cfg["split_frac"].split(","))
    split_seed = int(cfg["split_seed"])

    sys.path.append(str(SPATIAL_CV_BASE / "03_training"))
    from spatial_cv import SpatialKFold

    pos_m = latlon_graus_para_metros(pos_deg)
    parts = split_espacial_3vias(pos_m, grid_km, buffer_km, fracs, split_seed, SpatialKFold, log)

    rec["particoes_sha"] = {}
    sha_ok_all = True
    for k, loc in parts.items():
        sha_calc_idx = sha256_idx(loc)
        sha_esp = run_rec["particoes"][k]["idx_sha256_global"]
        ok = (sha_calc_idx == sha_esp)
        sha_ok_all = sha_ok_all and ok
        rec["particoes_sha"][k] = {"n": int(loc.size), "recomputado": sha_calc_idx,
                                    "run_json": sha_esp, "ok": ok}
    if not sha_ok_all:
        rec["status"] = "erro_idx_sha_divergente"
        return rec
    log(f"  particoes OK: train={parts['train'].size:,} val={parts['val'].size:,} test={parts['test'].size:,}")

    sent_all = y[:, 0] >= PL_TARGET_MAX_VALID
    pi = {"celula": float(sent_all.mean())}
    for k, loc in parts.items():
        pi[k] = float(sent_all[loc].mean())
    rec["pi_frac_sentinela"] = pi

    def teto_r2(mask_pop: np.ndarray) -> dict:
        """Squared correlation of valid RSSI with -log10(distance), distance floored at 1 m."""
        val = mask_pop & (~sent_all)
        rssi_v = y[val, 3]
        d_v = dist_all[val]
        c = corr_pearson(rssi_v, -np.log10(np.maximum(d_v, 1.0)))
        return {"n_validos": int(val.sum()), "corr": c, "r2_teto": (c * c) if not np.isnan(c) else None}

    mask_celula = np.ones(n_total, dtype=bool)
    mask_teste = np.zeros(n_total, dtype=bool)
    mask_teste[parts["test"]] = True
    rec["r2_teto"] = {"celula": teto_r2(mask_celula), "teste": teto_r2(mask_teste)}

    col3 = y[:, 3].astype(np.float64)  # RSSI, dBm
    rec["mae_constante_mediana_celula_inteira"] = preditor_constante(col3)
    rec["mae_constante_mediana_teste"] = preditor_constante(col3[parts["test"]])
    med_treino = float(np.median(col3[parts["train"]]))
    rec["mae_constante_mediana_treino_aplicada_teste"] = mae_constante_fixo(col3[parts["test"]], med_treino)
    rec["mae_constante_piso_teste"] = mae_constante_fixo(col3[parts["test"]], PISO_RSSI_DBM)
    rec["mae_constante_piso_celula_inteira"] = mae_constante_fixo(col3, PISO_RSSI_DBM)

    # Cross-check against the stored per-node predictions of the graph-free model
    # (same test partition): indices, targets, sentinel flags and test MAE.
    npz_path = NPZ_DIR / f"mlp_{cidade}_{quad}_s42.npz"
    cruz = {"npz_path": str(npz_path), "existe": npz_path.exists()}
    if npz_path.exists():
        dnpz = np.load(npz_path)
        idx_npz = dnpz["idx_global"]
        target_npz = dnpz["target"]
        sent_npz = dnpz["sentinela"]
        idx_igual = bool(np.array_equal(np.sort(idx_npz), np.sort(parts["test"])))
        cruz["idx_igual"] = idx_igual
        if idx_igual:
            ordem = np.argsort(idx_npz)
            idx_npz_ord = idx_npz[ordem]
            target_npz_ord = target_npz[ordem]
            sent_npz_ord = sent_npz[ordem]
            ordem_tensor = np.argsort(parts["test"])
            idx_tensor_ord = parts["test"][ordem_tensor]
            assert np.array_equal(idx_npz_ord, idx_tensor_ord)
            target_tensor_ord = y[idx_tensor_ord]
            target_igual = bool(np.array_equal(target_npz_ord.astype(np.float32),
                                                target_tensor_ord.astype(np.float32)))
            sent_tensor_ord = sent_all[idx_tensor_ord]
            sentinela_igual = bool(np.array_equal(sent_npz_ord, sent_tensor_ord))
            cruz["target_igual"] = target_igual
            cruz["sentinela_igual"] = sentinela_igual
            col3_npz = target_npz_ord[:, 3].astype(np.float64)
            med_npz = float(np.median(col3_npz))
            mae_npz_mediana = float(np.abs(col3_npz - med_npz).mean())
            mae_npz_piso = float(np.abs(col3_npz - PISO_RSSI_DBM).mean())
            cruz["mae_teste_npz_mediana"] = mae_npz_mediana
            cruz["mae_teste_npz_piso"] = mae_npz_piso
            cruz["diff_mediana_vs_tensor"] = abs(mae_npz_mediana - rec["mae_constante_mediana_teste"]["mae_db"])
            cruz["diff_piso_vs_tensor"] = abs(mae_npz_piso - rec["mae_constante_piso_teste"]["mae_db"])
        else:
            cruz["aviso"] = "idx_global do npz difere da particao de teste recomputada"
    rec["cruzamento_npz"] = cruz

    rec["status"] = "ok"
    rec["timestamp_utc_fim"] = datetime.now(timezone.utc).isoformat()
    del y, dist_all, pos_deg
    gc.collect()
    return rec


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--celula", default="", help="ex.: lins_Q1 (fumaca de uma so celula)")
    args = ap.parse_args()

    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    if OUT_RAW.exists():
        estado = json.loads(OUT_RAW.read_text(encoding="utf-8"))
    else:
        estado = {"script_caminho": str(SCRIPT_PATH), "script_sha256": sha256_self(),
                  "timestamp_utc_inicio": datetime.now(timezone.utc).isoformat(),
                  "celulas": {}}
    estado["script_sha256"] = sha256_self()

    if args.celula:
        cidade, quad = args.celula.split("_")
        pares = [(cidade, quad)]
    else:
        pares = [(c, q) for c in CIDADES for q in QUADRANTES]

    for cidade, quad in pares:
        chave = f"{cidade}_{quad}"
        ja = estado["celulas"].get(chave)
        if ja is not None and ja.get("status") == "ok" and ja.get("sha256_ok"):
            log(f"{chave}: ja concluida (sha ok), pulando")
            continue
        log(f"=== {chave} ===")
        try:
            rec = medir_celula(cidade, quad)
        except Exception as e:
            rec = {"cidade": cidade, "quadrante": quad, "status": "erro_excecao",
                   "erro": repr(e), "traceback": traceback.format_exc()}
        estado["celulas"][chave] = rec
        estado["timestamp_utc_ultima_gravacao"] = datetime.now(timezone.utc).isoformat()
        OUT_RAW.write_text(json.dumps(estado, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"{chave}: status={rec.get('status')} (gravado em {OUT_RAW})")

    log("concluido")


if __name__ == "__main__":
    main()
