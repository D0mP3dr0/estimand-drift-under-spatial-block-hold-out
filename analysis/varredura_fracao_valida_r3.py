#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Valid-node fraction of the training, validation and test partitions over 20 split draws.

For the 16 cells (cities bauru, campinas, lins, sorocaba x quadrants Q1..Q4) with
block side g = 10 km and buffer b = 2 km, and for each of the 20 split seeds in
SEEDS_20, the script computes the fraction of valid nodes (nodes whose path-loss
target rf_targets[:, 0] is below PL_TARGET_MAX_VALID; valid fraction = 1 - pi) in
each partition after the buffer, and the differences
    delta_val = frac_val - frac_train,   delta_te = frac_test - frac_train.
It then summarises, per cell, the mean of each delta over the 20 seeds with a
bootstrap 95% interval and, per seed, k_val = number of cells with delta_val < 0
and k_te = number of cells with delta_te > 0. Because Q1/Q2 and Q3/Q4 of a city
share the same geometry (hence the same block permutation for a given seed), the
same summaries are repeated on 8 geometry units by averaging each pair.

Decision criterion, written before the run: the validation pattern is a "lei"
(law) if (i) k_val >= 12 of 16 in at least 18 of the 20 seeds and (ii) at least
12 of 16 cells have a negative mean delta_val whose bootstrap 95% interval
excludes 0; otherwise "distribuicao" (distribution). The same rule with the sign
reversed (delta_te > 0) is applied to the test partition.

Steps: (1) check the 16 graph tensors against the data manifest (size and stored
SHA-256) without rehashing them, rehashing only on mismatch; (2) extract once per
tensor, under a file lock, a boolean N_SIDE x N_SIDE valid-node mask into a small
.npz cache; (3) validation gate: for seed 42, the exact split
(split_espacial_3vias_exato) must reproduce the valid fractions recorded in the
16 run JSON files for the three partitions (48 comparisons) within relative error
1e-3, otherwise the script stops; (4) sweep 16 cells x 20 seeds with the
distance-transform buffer (split_frac_valida); (5) aggregation and criterion.

Shared constants and helpers (cities, quadrants, grid size, seeds, partition
fractions, block assignment, exact split) are imported from the companion module
varredura_split_geometria.py (ORIG_SCRIPT).

Inputs: graph tensors in TENSOR_DIR, HASHES_SHA256.txt, the data manifest
(MANIFEST_V3), and the 16 seed-42 run JSON files in TREINOS_DIR.
Output: one JSON file (--out; in this repository
results/inputs/fracao_valida_val_treino_20_splits.json), rewritten after each
step; optional plain-text log (--log).
Determinism: block permutations use numpy RandomState(split_seed); the bootstrap
uses RandomState(20260925) with 10000 resamples. CPU only (torch is used only to
read the tensors).

Usage: python analysis/varredura_fracao_valida_r3.py --out <json> [--log <log>]
"""
import argparse
import fcntl
import gc
import hashlib
import importlib.util
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ORIG_SCRIPT = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts/"
    "varredura_split_geometria.py"
)
# companion module loaded by path (not modified); its constants and helpers are reused below
_spec = importlib.util.spec_from_file_location("varredura_split_geometria_orig", ORIG_SCRIPT)
vs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vs)

CIDADES = vs.CIDADES
QS = vs.QS
N_SIDE = vs.N_SIDE
PL_TARGET_MAX_VALID = vs.PL_TARGET_MAX_VALID
SEEDS_20 = vs.SEEDS_20
FRACS = vs.FRACS
TREINOS_DIR = vs.TREINOS_DIR
assign_groups = vs.assign_groups
latlon_graus_para_metros = vs.latlon_graus_para_metros
build_synthetic_grid = vs.build_synthetic_grid
split_espacial_3vias_exato = vs.split_espacial_3vias_exato
sha256_of_file = vs.sha256_of_file
sha256_of_text = vs.sha256_of_text

TENSOR_DIR = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")
HASHES_FILE = TENSOR_DIR / "HASHES_SHA256.txt"
MANIFEST_V3 = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/manifest_mathematics_v3.jsonl"
)
LOCKFILE = Path(
    "internal/tensor28gb.lock"
)
CACHE_DIR = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_R2_2026-09-24/cache_r3"
)
FIO_DIR = Path("internal/notes")

# pairs of quadrants that share the same geometry, hence the same block permutation per seed
GEOMETRIAS = {
    "bauru_Q1Q2": ("bauru", "Q1", "Q2"),
    "bauru_Q3Q4": ("bauru", "Q3", "Q4"),
    "campinas_Q1Q2": ("campinas", "Q1", "Q2"),
    "campinas_Q3Q4": ("campinas", "Q3", "Q4"),
    "lins_Q1Q2": ("lins", "Q1", "Q2"),
    "lins_Q3Q4": ("lins", "Q3", "Q4"),
    "sorocaba_Q1Q2": ("sorocaba", "Q1", "Q2"),
    "sorocaba_Q3Q4": ("sorocaba", "Q3", "Q4"),
}


def tensor_path(cidade, q):
    return TENSOR_DIR / f"transfer_dataset_{cidade}_v19_{q}_enriched_cftudo.pt"


def carregar_manifest_tensores():
    out = {}
    with open(MANIFEST_V3) as f:
        for line in f:
            d = json.loads(line)
            if d.get("grupo") == "tensores_cftudo":
                out[d["celula"]] = d
    return out


def carregar_hashes_file():
    out = {}
    for line in HASHES_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        h, name = line.split(None, 1)
        out[name.strip()] = h
    return out


def conferir_fonte_sem_recalcular(cidade, q, manifest, hashes_file, log):
    """Check one tensor against the data manifest without rehashing it.

    Confirmed when the file size matches the manifest, the manifest hash flags are true,
    and the manifest SHA-256 equals the one in HASHES_SHA256.txt.
    """
    cel = f"{cidade}_{q}"
    p = tensor_path(cidade, q)
    st = p.stat()
    m = manifest.get(cel)
    esperado_hashfile = hashes_file.get(p.name)
    info = {
        "path": str(p),
        "tamanho_bytes_real": st.st_size,
        "mtime_real_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(st.st_mtime)),
        "em_manifest_v3": m is not None,
        "sha256_hashesfile": esperado_hashfile,
    }
    if m is not None:
        info["tamanho_bytes_manifest"] = m.get("tamanho_bytes")
        info["tamanho_confere_manifest"] = (m.get("tamanho_bytes") == st.st_size)
        info["sha_confere_E_no_manifest"] = m.get("sha_confere_E")
        info["sha_confere_run_json_no_manifest"] = m.get("sha_confere_run_json")
        info["sha256_manifest"] = m.get("sha256")
        info["sha_confere_hashesfile_vs_manifest"] = (m.get("sha256") == esperado_hashfile)
        info["veredito"] = (
            "confirmado_sem_recalculo"
            if (info["tamanho_confere_manifest"] and info["sha_confere_E_no_manifest"]
                and info["sha_confere_run_json_no_manifest"]
                and info["sha_confere_hashesfile_vs_manifest"])
            else "DIVERGENTE_precisa_recalculo"
        )
    else:
        info["veredito"] = "SEM_MANIFEST_precisa_recalculo"
    log(f"  fonte {cel}: veredito={info['veredito']} tamanho={st.st_size}")
    return info


def extrair_cache_valid2d(cidade, q, log):
    """Load one tensor (under an exclusive file lock) and cache its 2-D valid-node mask as .npz.

    Valid node: y[:, 0] < PL_TARGET_MAX_VALID, reshaped to N_SIDE x N_SIDE. An existing cache is reused.
    """
    cache_path = CACHE_DIR / f"{cidade}_{q}_valid2d.npz"
    if cache_path.exists():
        return cache_path, {"reusado_cache_existente": True}
    p = tensor_path(cidade, q)
    meta = {"reusado_cache_existente": False, "tensor_path": str(p)}
    LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCKFILE, "a+") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            log(f"  [lock] carregando {cidade} {q} ...")
            import torch
            t0 = time.time()
            d = torch.load(str(p), map_location="cpu", mmap=True, weights_only=False)
            y = d["terrain"].y.numpy()
            valid = (y[:, 0] < PL_TARGET_MAX_VALID)
            n_side_local = int(round(np.sqrt(valid.size)))
            assert n_side_local == N_SIDE, (cidade, q, valid.size, n_side_local)
            valid2d = valid.reshape(n_side_local, n_side_local)
            meta["tempo_carga_s"] = time.time() - t0
            meta["n_nos"] = int(valid.size)
            meta["frac_valido_full"] = float(valid.mean())
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, valid2d=valid2d)
            del d, y, valid, valid2d
            gc.collect()
            log(f"  [lock] {cidade} {q}: {meta['tempo_carga_s']:.1f}s, frac_full={meta['frac_valido_full']:.4f}")
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)
    return cache_path, meta


def carregar_valid2d_do_cache(cache_path):
    with np.load(cache_path) as z:
        return z["valid2d"]


def split_frac_valida(pos_m, ell_x_m, ell_y_m, group_ids, g, buffer_km, fracs,
                       split_seed, valid2d):
    """Three-way block split for one seed and valid-node fraction of each partition after the buffer.

    Blocks are shuffled with RandomState(split_seed) and assigned by `fracs`. Validation nodes
    closer than buffer_km to training are dropped; test nodes closer than buffer_km to training
    or retained validation are dropped. Distances use a Euclidean distance transform on the grid.
    """
    from scipy import ndimage

    grupos = np.unique(group_ids)
    rng = np.random.RandomState(split_seed)
    grupos_emb = grupos.copy()
    rng.shuffle(grupos_emb)

    n_g = len(grupos_emb)
    n_tr = max(1, int(round(fracs[0] * n_g)))
    n_va = max(1, int(round(fracs[1] * n_g)))
    # keep at least one block for the test partition
    if n_tr + n_va >= n_g:
        n_tr = max(1, n_g - 2)
        n_va = 1
    g_tr = grupos_emb[:n_tr]
    g_va = grupos_emb[n_tr:n_tr + n_va]
    g_te = grupos_emb[n_tr + n_va:]

    m_tr = np.isin(group_ids, g_tr)
    m_va = np.isin(group_ids, g_va)
    m_te = np.isin(group_ids, g_te)
    n_va_bruto, n_te_bruto = int(m_va.sum()), int(m_te.sum())

    tr2d = m_tr.reshape(N_SIDE, N_SIDE)
    va2d = m_va.reshape(N_SIDE, N_SIDE)
    te2d = m_te.reshape(N_SIDE, N_SIDE)
    # grid spacing in km, ordered (row = y, column = x) as the reshaped masks
    sampling = (ell_y_m / 1000.0, ell_x_m / 1000.0)

    dist_train_km = ndimage.distance_transform_edt(~tr2d, sampling=sampling)
    if buffer_km > 0:
        va_retido2d = va2d & (dist_train_km >= buffer_km)
    else:
        va_retido2d = va2d.copy()

    trva2d = tr2d | va_retido2d
    dist_trva_km = ndimage.distance_transform_edt(~trva2d, sampling=sampling)
    if buffer_km > 0:
        te_retido2d = te2d & (dist_trva_km >= buffer_km)
    else:
        te_retido2d = te2d.copy()

    n_tr_final = int(tr2d.sum())
    n_va_final = int(va_retido2d.sum())
    n_te_final = int(te_retido2d.sum())

    frac_train = float(valid2d[tr2d].mean()) if n_tr_final else None
    frac_val = float(valid2d[va_retido2d].mean()) if n_va_final else None
    frac_test = float(valid2d[te_retido2d].mean()) if n_te_final else None

    return {
        "n_nos_apos_buffer": {"train": n_tr_final, "val": n_va_final, "test": n_te_final},
        "n_nos_antes_do_buffer": {"train": int(m_tr.sum()), "val": n_va_bruto, "test": n_te_bruto},
        "frac_valida": {"train": frac_train, "val": frac_val, "test": frac_test},
    }


def bootstrap_ic95_media(vals, n_resamples=10000, seed_bootstrap=20260925):
    """Percentile bootstrap 95% interval of the mean of `vals` (None entries dropped; here the 20 seeds)."""
    arr = np.array([v for v in vals if v is not None], dtype=float)
    n = arr.size
    if n == 0:
        return {"media": None, "ic95_lo": None, "ic95_hi": None, "n": 0,
                "exclui_zero": None}
    media = float(arr.mean())
    rng = np.random.RandomState(seed_bootstrap)
    idx = rng.randint(0, n, size=(n_resamples, n))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    exclui_zero = bool((lo > 0) or (hi < 0))
    return {"media": media, "ic95_lo": float(lo), "ic95_hi": float(hi),
            "n": int(n), "n_resamples": n_resamples, "seed_bootstrap": seed_bootstrap,
            "exclui_zero": exclui_zero}


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
            logf.write(line + "\n")
            logf.flush()

    t_inicio = time.time()
    saida = {
        "frente": "internal/analysis",
        "fio": "internal/analysis",
        "entrega": "R3-a",
        "status": "em_andamento",
        "pergunta": (
            "in the 16 g10b2 cells (bauru/campinas/lins/sorocaba x Q1..Q4), across the "
            "20 split_seeds of the earlier split-geometry sweep, what is the fraction of nodes with a valid target (1-pi, "
            "pi = rf_targets[:,0] < 299.0) in the TRAINING, VALIDATION and TEST partitions "
            "after the buffer? Does the 15/16 (validation poorer than training, seed 42) "
            "hold as a law or become a distribution? And the 12/16 (test richer "
            "than training), now with the 16 cells?"
        ),
        "seeds_usadas": SEEDS_20,
    }
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    saida["script_sha256"] = sha256_of_text(Path(__file__).read_text())
    entradas_sha256 = {}
    run_paths_g10b2 = sorted(TREINOS_DIR.glob("run_c0c1cf_*_s42_*_g10b2.json"))
    assert len(run_paths_g10b2) == 16, len(run_paths_g10b2)
    for p in run_paths_g10b2:
        entradas_sha256[str(p)] = sha256_of_file(p)
    saida["entradas_sha256"] = entradas_sha256
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    log("conferindo fonte dos 16 tensores contra manifest_mathematics_v3 + HASHES_SHA256.txt (sem recalculo)...")
    manifest = carregar_manifest_tensores()
    hashes_file = carregar_hashes_file()
    fonte_tensores = {}
    for cidade in CIDADES:
        for q in QS:
            fonte_tensores[f"{cidade}_{q}"] = conferir_fonte_sem_recalcular(cidade, q, manifest, hashes_file, log)
    n_confirmados = sum(1 for v in fonte_tensores.values() if v["veredito"] == "confirmado_sem_recalculo")
    saida["fonte_tensores_16"] = {
        "n_confirmados_sem_recalculo": n_confirmados,
        "n_total": 16,
        "detalhe": fonte_tensores,
    }
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))
    if n_confirmados < 16:
        log("ALERTA: nem todos os 16 tensores confirmados sem recalculo -- recalculando sha256 dos divergentes...")
        for cel, info in fonte_tensores.items():
            if info["veredito"] != "confirmado_sem_recalculo":
                cidade, q = cel.rsplit("_", 1)
                p = tensor_path(cidade, q)
                with open(LOCKFILE, "a+") as lockf:
                    fcntl.flock(lockf, fcntl.LOCK_EX)
                    try:
                        info["sha256_recalculado"] = sha256_of_file(p)
                    finally:
                        fcntl.flock(lockf, fcntl.LOCK_UN)
                info["bate_hashesfile_apos_recalculo"] = (info["sha256_recalculado"] == info["sha256_hashesfile"])
        Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    geo_por_celula = vs.geometria_todas_celulas()

    def get_pos(cidade, q):
        geo = geo_por_celula[(cidade, q)]
        lon, lat = build_synthetic_grid(geo["lon_min_deg"], geo["lon_max_deg"],
                                         geo["lat_min_deg"], geo["lat_max_deg"])
        pos_m = latlon_graus_para_metros(lon, lat)
        # node spacing (m) along x (first grid row) and along y (first grid column)
        ell_x = float(np.median(np.abs(np.diff(pos_m[:N_SIDE, 0]))))
        ell_y = float(np.median(np.abs(np.diff(pos_m[::N_SIDE, 1]))))
        return pos_m, ell_x, ell_y

    log("extraindo cache leve (valid2d) dos 16 tensores (um por vez, sob flock)...")
    cache_paths = {}
    cache_meta = {}
    for cidade in CIDADES:
        for q in QS:
            cp, meta = extrair_cache_valid2d(cidade, q, log)
            cache_paths[(cidade, q)] = cp
            cache_meta[f"{cidade}_{q}"] = meta
    saida["cache_extracao"] = {
        "diretorio": str(CACHE_DIR),
        "detalhe": cache_meta,
    }
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))
    log(f"cache pronto. RAM atual (maxrss kb): {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")

    log("rodando portao (seed 42, split exato cKDTree, 16 celulas x 3 papeis = 48 comparacoes)...")
    portao_detalhe = []
    for cidade in CIDADES:
        for q in QS:
            run_path = TREINOS_DIR / f"run_c0c1cf_{cidade}_s42_{q}_g10b2.json"
            dj = json.load(open(run_path))
            cfgj = dj["config"]
            fracsj = tuple(float(x) for x in cfgj["split_frac"].split(","))
            pos_m, _, _ = get_pos(cidade, q)
            partsj, _, _ = split_espacial_3vias_exato(pos_m, cfgj["grid_km"], cfgj["buffer_km"],
                                                        fracsj, cfgj["split_seed"])
            valid2d = carregar_valid2d_do_cache(cache_paths[(cidade, q)])
            valid1d = valid2d.ravel()
            decl = dj["particoes"]
            for parte in ("train", "val", "test"):
                mask = partsj[parte]
                n = int(mask.sum())
                n_valid = int(valid1d[mask].sum())
                frac_repro = (n_valid / n) if n else None
                frac_decl = decl[parte]["frac_pl_alvo_valido"]
                erro_rel = (abs(frac_repro - frac_decl) / frac_decl) if (frac_repro is not None and frac_decl) else None
                portao_detalhe.append({
                    "cidade": cidade, "Q": q, "papel": parte,
                    "frac_reproduzida": frac_repro, "frac_declarada_run_json": frac_decl,
                    "erro_relativo": erro_rel,
                })
            del valid2d, valid1d
    erros_portao = [d["erro_relativo"] for d in portao_detalhe if d["erro_relativo"] is not None]
    erro_max_portao = max(erros_portao) if erros_portao else None
    # the sweep runs only if the exact split reproduces all 48 recorded fractions within 1e-3
    gate_ok = (erro_max_portao is not None) and (erro_max_portao <= 1e-3)
    saida["portao_validacao"] = {
        "descricao": "seed 42, split exato (cKDTree), reproducao de particoes.<papel>.frac_pl_alvo_valido nas 16 celulas g10b2 x 3 papeis (48 comparacoes)",
        "n_comparacoes": len(portao_detalhe),
        "erro_relativo_maximo": erro_max_portao,
        "gate_le_1e-3": gate_ok,
        "detalhe": portao_detalhe,
    }
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))
    log(f"portao: erro relativo maximo = {erro_max_portao}, gate<=1e-3: {gate_ok}")

    if not gate_ok:
        saida["status"] = "parcial_portao_falhou"
        saida["motivo_parada"] = (
            f"gate (48 comparisons, seed 42, exact split) failed: maximum relative error "
            f"{erro_max_portao} > 1e-3. Sweep NOT run (stopping rule: "
            "if the gate fails, stop and write the output with the reason)."
        )
        saida["pico_ram_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
        saida["tempo_total_s"] = time.time() - t_inicio
        Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))
        log("PORTAO FALHOU -- parando.")
        return

    saida["pi_validacao_definicao"] = {
        "definicao": "rf_targets[:,0] < 299.0 (PL_TARGET_MAX_VALID), equivalent to col0<299 -- already validated against col4 (rejected definition) in the earlier split-geometry sweep; frac_valida = 1 - pi",
        "reconfirmada_nesta_rodada": "sim, via portao_validacao acima (48/48 comparacoes)",
    }

    log("iniciando varredura (16 celulas x 20 seeds, g10b2, metodo EDT)...")
    resultados = []
    pos_cache = {}
    group_ids_cache = {}
    valid2d_cache = {}
    # block side and buffer width (km)
    g, b = 10.0, 2.0
    for cidade in CIDADES:
        for q in QS:
            pos_m, ell_x, ell_y = get_pos(cidade, q)
            pos_cache[(cidade, q)] = (pos_m, ell_x, ell_y)
            group_ids = assign_groups(pos_m / 1000.0, g)
            group_ids_cache[(cidade, q)] = group_ids
            valid2d_cache[(cidade, q)] = carregar_valid2d_do_cache(cache_paths[(cidade, q)])

    t_var = time.time()
    for cidade in CIDADES:
        for q in QS:
            pos_m, ell_x, ell_y = pos_cache[(cidade, q)]
            group_ids = group_ids_cache[(cidade, q)]
            valid2d = valid2d_cache[(cidade, q)]
            for seed in SEEDS_20:
                r = split_frac_valida(pos_m, ell_x, ell_y, group_ids, g, b, FRACS, seed, valid2d)
                frac = r["frac_valida"]
                delta_val = (frac["val"] - frac["train"]) if (frac["val"] is not None and frac["train"] is not None) else None
                delta_te = (frac["test"] - frac["train"]) if (frac["test"] is not None and frac["train"] is not None) else None
                resultados.append({
                    "cidade": cidade, "Q": q, "g_km": g, "b_km": b, "split_seed": seed,
                    "metodo": "edt",
                    "n_nos_apos_buffer": r["n_nos_apos_buffer"],
                    "frac_valida": frac,
                    "delta_val": delta_val,
                    "delta_te": delta_te,
                })
    log(f"varredura: {len(resultados)} combinacoes em {time.time()-t_var:.1f}s")
    saida["resultados"] = resultados
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    log("agregando por celula (16) e calculando IC bootstrap...")
    por_celula = defaultdict(list)
    for r in resultados:
        por_celula[(r["cidade"], r["Q"])].append(r)

    agregado_16 = {}
    for (cidade, q), rs in por_celula.items():
        rs_by_seed = {r["split_seed"]: r for r in rs}
        deltas_val = [r["delta_val"] for r in rs]
        deltas_te = [r["delta_te"] for r in rs]
        ic_val = bootstrap_ic95_media(deltas_val)
        ic_te = bootstrap_ic95_media(deltas_te)
        seed42 = rs_by_seed[42]
        sinal_val_seed42 = (seed42["delta_val"] < 0) if seed42["delta_val"] is not None else None
        sinal_te_seed42 = (seed42["delta_te"] > 0) if seed42["delta_te"] is not None else None
        n_seeds_mesmo_sinal_val = sum(
            1 for r in rs if r["delta_val"] is not None and sinal_val_seed42 is not None
            and (r["delta_val"] < 0) == sinal_val_seed42
        )
        n_seeds_mesmo_sinal_te = sum(
            1 for r in rs if r["delta_te"] is not None and sinal_te_seed42 is not None
            and (r["delta_te"] > 0) == sinal_te_seed42
        )
        agregado_16[f"{cidade}_{q}"] = {
            "cidade": cidade, "Q": q,
            "delta_val_por_seed": {str(r["split_seed"]): r["delta_val"] for r in rs},
            "delta_te_por_seed": {str(r["split_seed"]): r["delta_te"] for r in rs},
            "delta_val_ic95_bootstrap": ic_val,
            "delta_te_ic95_bootstrap": ic_te,
            "delta_val_seed42": seed42["delta_val"],
            "delta_te_seed42": seed42["delta_te"],
            "sinal_val_seed42_negativo": sinal_val_seed42,
            "sinal_te_seed42_positivo": sinal_te_seed42,
            "n_seeds_mesmo_sinal_val_que_seed42": n_seeds_mesmo_sinal_val,
            "n_seeds_mesmo_sinal_te_que_seed42": n_seeds_mesmo_sinal_te,
            "frac_seeds_mesmo_sinal_val": n_seeds_mesmo_sinal_val / len(rs),
            "frac_seeds_mesmo_sinal_te": n_seeds_mesmo_sinal_te / len(rs),
        }
    saida["agregado_por_celula_16"] = agregado_16
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    log("calculando k_val(s) e k_te(s) por seed (16 celulas)...")
    por_seed = defaultdict(list)
    for r in resultados:
        por_seed[r["split_seed"]].append(r)

    k_val_por_seed = {}
    k_te_por_seed = {}
    for seed, rs in por_seed.items():
        k_val = sum(1 for r in rs if r["delta_val"] is not None and r["delta_val"] < 0)
        k_te = sum(1 for r in rs if r["delta_te"] is not None and r["delta_te"] > 0)
        k_val_por_seed[str(seed)] = k_val
        k_te_por_seed[str(seed)] = k_te

    kv = np.array(list(k_val_por_seed.values()))
    kt = np.array(list(k_te_por_seed.values()))
    dist_k_val = {
        "min": int(kv.min()), "mediana": float(np.median(kv)), "max": int(kv.max()),
        "histograma": {str(k): int((kv == k).sum()) for k in range(17)},
        "n_seeds_com_k_ge_12": int((kv >= 12).sum()),
    }
    dist_k_te = {
        "min": int(kt.min()), "mediana": float(np.median(kt)), "max": int(kt.max()),
        "histograma": {str(k): int((kt == k).sum()) for k in range(17)},
        "n_seeds_com_k_ge_12": int((kt >= 12).sum()),
    }
    saida["k_val_por_seed"] = k_val_por_seed
    saida["k_te_por_seed"] = k_te_por_seed
    saida["distribuicao_k_val"] = dist_k_val
    saida["distribuicao_k_te"] = dist_k_te
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    log("agregando por geometria (8): Q1=Q2 e Q3=Q4 compartilham a mesma permutacao de blocos por seed...")
    # geometry name -> seed -> deltas averaged over the two quadrants of the pair
    por_geo_seed = defaultdict(dict)
    for geo_nome, (cidade, qa, qb) in GEOMETRIAS.items():
        rs_a = {r["split_seed"]: r for r in por_celula[(cidade, qa)]}
        rs_b = {r["split_seed"]: r for r in por_celula[(cidade, qb)]}
        for seed in SEEDS_20:
            dva, dvb = rs_a[seed]["delta_val"], rs_b[seed]["delta_val"]
            dta, dtb = rs_a[seed]["delta_te"], rs_b[seed]["delta_te"]
            dv = (dva + dvb) / 2 if (dva is not None and dvb is not None) else None
            dt = (dta + dtb) / 2 if (dta is not None and dtb is not None) else None
            por_geo_seed[geo_nome][seed] = {"delta_val_media_par": dv, "delta_te_media_par": dt}

    agregado_8 = {}
    for geo_nome, por_seed_geo in por_geo_seed.items():
        deltas_val = [v["delta_val_media_par"] for v in por_seed_geo.values()]
        deltas_te = [v["delta_te_media_par"] for v in por_seed_geo.values()]
        ic_val = bootstrap_ic95_media(deltas_val)
        ic_te = bootstrap_ic95_media(deltas_te)
        dv42 = por_seed_geo[42]["delta_val_media_par"]
        dt42 = por_seed_geo[42]["delta_te_media_par"]
        sinal_val_42 = (dv42 < 0) if dv42 is not None else None
        sinal_te_42 = (dt42 > 0) if dt42 is not None else None
        agregado_8[geo_nome] = {
            "membros": GEOMETRIAS[geo_nome][1:],
            "delta_val_media_par_por_seed": {str(s): v["delta_val_media_par"] for s, v in por_seed_geo.items()},
            "delta_te_media_par_por_seed": {str(s): v["delta_te_media_par"] for s, v in por_seed_geo.items()},
            "delta_val_ic95_bootstrap": ic_val,
            "delta_te_ic95_bootstrap": ic_te,
            "sinal_val_seed42_negativo": sinal_val_42,
            "sinal_te_seed42_positivo": sinal_te_42,
        }
    saida["agregado_por_geometria_8"] = agregado_8

    k_val_geo_por_seed = {}
    k_te_geo_por_seed = {}
    for seed in SEEDS_20:
        kv_geo = sum(1 for geo_nome in GEOMETRIAS
                     if por_geo_seed[geo_nome][seed]["delta_val_media_par"] is not None
                     and por_geo_seed[geo_nome][seed]["delta_val_media_par"] < 0)
        kt_geo = sum(1 for geo_nome in GEOMETRIAS
                     if por_geo_seed[geo_nome][seed]["delta_te_media_par"] is not None
                     and por_geo_seed[geo_nome][seed]["delta_te_media_par"] > 0)
        k_val_geo_por_seed[str(seed)] = kv_geo
        k_te_geo_por_seed[str(seed)] = kt_geo
    kvg = np.array(list(k_val_geo_por_seed.values()))
    ktg = np.array(list(k_te_geo_por_seed.values()))
    saida["k_val_por_seed_geometria_8"] = k_val_geo_por_seed
    saida["k_te_por_seed_geometria_8"] = k_te_geo_por_seed
    saida["distribuicao_k_val_geometria_8"] = {
        "min": int(kvg.min()), "mediana": float(np.median(kvg)), "max": int(kvg.max()),
        "histograma": {str(k): int((kvg == k).sum()) for k in range(9)},
        "n_seeds_com_k_ge_6_de_8": int((kvg >= 6).sum()),
    }
    saida["distribuicao_k_te_geometria_8"] = {
        "min": int(ktg.min()), "mediana": float(np.median(ktg)), "max": int(ktg.max()),
        "histograma": {str(k): int((ktg == k).sum()) for k in range(9)},
        "n_seeds_com_k_ge_6_de_8": int((ktg >= 6).sum()),
    }
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    log("applying the pre-registered decision criterion (lei vs distribuicao)...")
    n_celulas_delta_val_ic_negativo = sum(
        1 for v in agregado_16.values()
        if v["delta_val_ic95_bootstrap"]["media"] is not None
        and v["delta_val_ic95_bootstrap"]["media"] < 0
        and v["delta_val_ic95_bootstrap"]["exclui_zero"]
    )
    n_celulas_delta_te_ic_positivo = sum(
        1 for v in agregado_16.values()
        if v["delta_te_ic95_bootstrap"]["media"] is not None
        and v["delta_te_ic95_bootstrap"]["media"] > 0
        and v["delta_te_ic95_bootstrap"]["exclui_zero"]
    )
    n_seeds_k_val_ge_12 = dist_k_val["n_seeds_com_k_ge_12"]
    n_seeds_k_te_ge_12 = dist_k_te["n_seeds_com_k_ge_12"]

    # decision criterion (written before the run): conditions (i) and (ii) of the module docstring
    criterio_i_val = (n_seeds_k_val_ge_12 >= 18)
    criterio_ii_val = (n_celulas_delta_val_ic_negativo >= 12)
    veredito_15_16 = "lei" if (criterio_i_val and criterio_ii_val) else "distribuicao"

    criterio_i_te = (n_seeds_k_te_ge_12 >= 18)
    criterio_ii_te = (n_celulas_delta_te_ic_positivo >= 12)
    veredito_12_16 = "lei" if (criterio_i_te and criterio_ii_te) else "distribuicao"

    saida["criterios_fixados"] = {
        "definicao_literal": (
            "'lei' if (i) k_val(s) = #cells with frac_val < frac_treino >= 12/16 in "
            ">= 18 of the 20 seeds AND (ii) in >= 12/16 cells the mean over seeds of "
            "delta_val = frac_val - frac_treino is negative with a bootstrap 95% CI over "
            "the 20 seeds (10000 resamples, fixed bootstrap seed) excluding 0. "
            "Otherwise 'distribuicao'. Same criterion with the sign reversed for the test "
            "partition (delta_te = frac_teste - frac_treino > 0). Fixed BEFORE "
            "running the sweep."
        ),
        "validacao": {
            "n_seeds_com_k_val_ge_12_de_16": int(n_seeds_k_val_ge_12),
            "criterio_i_val_ge_18_de_20_seeds": criterio_i_val,
            "n_celulas_com_delta_val_ic95_negativo_excluindo_zero": int(n_celulas_delta_val_ic_negativo),
            "criterio_ii_val_ge_12_de_16_celulas": criterio_ii_val,
            "n_seeds_com_k_te_ge_12_de_16": int(n_seeds_k_te_ge_12),
            "criterio_i_te_ge_18_de_20_seeds": criterio_i_te,
            "n_celulas_com_delta_te_ic95_positivo_excluindo_zero": int(n_celulas_delta_te_ic_positivo),
            "criterio_ii_te_ge_12_de_16_celulas": criterio_ii_te,
        },
    }
    saida["veredito_15_16"] = veredito_15_16
    saida["veredito_12_16"] = veredito_12_16
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))
    log(f"criterion outcome 15/16 (validation): {veredito_15_16}; criterion outcome 12/16 (test): {veredito_12_16}")

    saida["resumo"] = {
        "n_celulas": 16, "n_seeds": 20, "config": "g10b2 (grid_km=10, buffer_km=2)",
        "delta_val_media_global": float(np.mean([r["delta_val"] for r in resultados if r["delta_val"] is not None])),
        "delta_te_media_global": float(np.mean([r["delta_te"] for r in resultados if r["delta_te"] is not None])),
        "n_celulas_delta_val_ic_negativo_de_16": int(n_celulas_delta_val_ic_negativo),
        "n_celulas_delta_te_ic_positivo_de_16": int(n_celulas_delta_te_ic_positivo),
        "k_val_distribuicao": dist_k_val,
        "k_te_distribuicao": dist_k_te,
        "aviso_dependencia": (
            "Q1=Q2 e Q3=Q4 compartilham a geometria (extensao_x_km/extensao_y_km "
            "identicas dentro de cada par -- conferido nos run JSON), logo o mesmo "
            "split_seed produz a MESMA permutacao de grupos (mesmo padrao geometrico "
            "de blocos train/val/test) nos dois membros do par; as 16 celulas NAO sao "
            "16 observacoes independentes quanto a geometria do sorteio (a fracao "
            "valida em si difere, pois o terreno/alvo PL e diferente). Ver "
            "agregado_por_geometria_8, k_val_por_seed_geometria_8, "
            "k_te_por_seed_geometria_8 para a leitura conservadora com 8 unidades."
        ),
        "veredito_15_16": veredito_15_16,
        "veredito_12_16": veredito_12_16,
    }

    saida["escopo"] = {
        "executado": [
            "portao: 16 celulas g10b2 x 3 papeis = 48 comparacoes, seed 42, split exato (cKDTree)",
            "varredura: 16 celulas x 20 seeds, g10b2, EDT method (same method and same EDT-vs-cKDTree validation as in the earlier split-geometry sweep)",
            "agregacao por celula (16) e por geometria (8)",
            "pre-registered decision criterion applied to validation (15/16) and test (12/16)",
        ],
        "fora_do_escopo": [
            "g5b2 in the 16 cells (did not fit the ~40 min time budget of this short run; g10b2 was given priority)",
            "full 6x6 (g,b) grid (outside the scope of this script; partly covered by the earlier split-geometry sweep)",
            "effective N under within-partition spatial autocorrelation: out of scope (belongs to a spatial-leakage analysis); "
            "here the sampling unit of the inference (bootstrap CI, k(s)) is the split_seed (20 observations, each one "
            "a COMPLETE partition of the node population), not the pixel -- within-partition spatial autocorrelation "
            "does not inflate N here because frac_valida is a POPULATION PARAMETER of the whole partition under that seed, "
            "not a sample mean over pixels; the real dependence of this inference is BETWEEN CELLS that share "
            "geometry (Q1=Q2, Q3=Q4), handled explicitly through agregado_por_geometria_8.",
        ],
    }
    saida["nao_verificado"] = [
        {"item": "g5b2 nas 16 celulas", "motivo": "outside the time budget of this short run; g10b2 was given priority"},
        {"item": "N efetivo sob autocorrelacao espacial intra-particao", "motivo": "outside the scope of this script (a spatial-leakage analysis); see escopo.fora_do_escopo for why N=20 seeds is the correct unit for this specific inference"},
    ]
    saida["tabela_numero_campo_comando"] = [
        {"numero": "frac_valida por (celula,seed,papel)", "campo": "resultados[i].frac_valida",
         "comando": "python varredura_fracao_valida_r3.py --out <json>"},
        {"numero": "delta_val / delta_te por (celula,seed)", "campo": "resultados[i].delta_val / delta_te", "comando": "idem"},
        {"numero": "k_val(s) / k_te(s) por seed", "campo": "k_val_por_seed / k_te_por_seed", "comando": "idem, bloco k_val(s)/k_te(s)"},
        {"numero": "IC95 bootstrap de delta_val/delta_te por celula", "campo": "agregado_por_celula_16.<celula>.delta_val_ic95_bootstrap / delta_te_ic95_bootstrap", "comando": "idem, bootstrap_ic95_media"},
        {"numero": "criterion outcome for 15/16 (validation) and 12/16 (test)", "campo": "veredito_15_16 / veredito_12_16", "comando": "idem, bloco criterios_fixados"},
        {"numero": "agregado por geometria (8)", "campo": "agregado_por_geometria_8", "comando": "idem, bloco geometria"},
        {"numero": "portao (48 comparacoes)", "campo": "portao_validacao.detalhe", "comando": "idem, bloco portao"},
    ]

    saida["status"] = "concluido_provisorio"
    saida["pico_ram_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    saida["tempo_total_s"] = time.time() - t_inicio
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))
    log(f"CONCLUIDO. tempo_total_s={saida['tempo_total_s']:.1f} pico_ram_gb={saida['pico_ram_gb']:.2f}")


if __name__ == "__main__":
    main()
