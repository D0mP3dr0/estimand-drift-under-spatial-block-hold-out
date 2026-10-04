#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities of the third-campaign trainers `train_gnn_v3.py` and `train_mlp_v3.py`.

The module does not reimplement training, splitting or the loss: every step
that exists in the frozen trainers (`training/frozen/train_gnn_c0_spatial.py`,
`training/frozen/train_mlp_c0_spatial.py`) is called by name on the loaded
module. It provides:
  - provenance: SHA-256 of the input tensors checked against the v4 artifact
    manifest (group `tensores_cftudo`, one entry per cell);
  - loading of a frozen trainer by importlib and run-time replacement of module
    attributes it resolves at call time (decoder class, MLP widths,
    NeighborLoader class), so the frozen files stay unchanged;
  - post-training reconstruction of a partition (same window, seeds and split)
    and per-node predictions in physical units, with the sentinel mask;
  - diagnostics written to the run JSON: complete-neighbourhood evaluation
    invariance, effective (gradient-receiving) parameter count, GradScaler and
    gradient-norm instrumentation, and per-channel output statistics.

Paths to adapt elsewhere: `GNN_RF_V2`, `GRAPH_DIR_DEFAULT`, `MANIFEST_V4_PATH`,
`FROZEN_SCRIPTS_DIR`.
Usage: imported by the two trainers; not run directly.
"""
from __future__ import annotations

import datetime
import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import bipartite_subgraph, subgraph

GNN_RF_V2 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2")
# GRAPH_DIR_V2_LEGADO holds the targets before the target correction and is not
# a default; GRAPH_DIR_DEFAULT holds the `*_enriched_cftudo.pt` and `*_gpu.pt`
# tensors whose digests are in the v4 manifest.
GRAPH_DIR_V2_LEGADO = GNN_RF_V2 / "graph_data"
GRAPH_DIR_DEFAULT = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")

MANIFEST_V4_PATH = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/manifest_mathematics_v4.jsonl")

FROZEN_SCRIPTS_DIR = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
    "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/scripts")
FROZEN_GNN_SCRIPT = FROZEN_SCRIPTS_DIR / "train_gnn_c0_spatial.py"
FROZEN_MLP_SCRIPT = FROZEN_SCRIPTS_DIR / "train_mlp_c0_spatial.py"

MODELO_V3_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODELO_V3_DIR))
from rf_decoder_v3 import AffineDecoderV3, sha256_do_arquivo  # noqa: E402


def sha256_arquivo(caminho: Path, bloco: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while True:
            chunk = f.read(bloco)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _carregar_entradas_manifest_v4(grupo: str = "tensores_cftudo",
                                    manifest_path: Path = MANIFEST_V4_PATH) -> list:
    """Return the entries of one group of the JSON Lines manifest."""
    entradas = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            d = json.loads(linha)
            if d.get("grupo") == grupo:
                entradas.append(d)
    return entradas


def sha256_com_reuso_manifest(path: Path, entradas_manifest: list) -> tuple:
    """Return `(sha256, method)` for a file.

    Reuses the digest stored in the manifest entry with the same resolved path
    only if the current size and mtime (UTC isoformat, to the second) equal the
    stored ones; otherwise recomputes the digest by streaming the file.
    """
    path = Path(path).resolve()
    st = path.stat()
    mtime_iso = datetime.datetime.fromtimestamp(
        st.st_mtime, tz=datetime.timezone.utc).isoformat()
    for entrada in entradas_manifest:
        cam = entrada.get("caminho", "")
        try:
            bate_caminho = Path(cam).resolve() == path
        except OSError:
            bate_caminho = False
        if bate_caminho:
            if (entrada.get("tamanho_bytes") == st.st_size
                    and entrada.get("mtime") == mtime_iso
                    and entrada.get("sha256")):
                return entrada["sha256"], "reuso_manifest_v4_mtime_e_tamanho_identicos"
            break
    return sha256_arquivo(path), "recomputado_sha256_arquivo_stream"


def verificar_proveniencia_dataset(rf_data_path: Path, graph_path: Path,
                                    manifest_path: Path = MANIFEST_V4_PATH,
                                    grupo: str = "tensores_cftudo") -> dict:
    """Hash the RF data file and the graph file and check the former against the manifest.

    Only the RF data file (`*_enriched_cftudo.pt`) has a manifest entry; the
    graph file (`*_gpu.pt`) is hashed for the record without a check. The caller
    aborts when `sha256_rf_data_bate_manifest_v4` is False.
    """
    entradas = _carregar_entradas_manifest_v4(grupo=grupo, manifest_path=manifest_path)
    sha_rf, metodo_rf = sha256_com_reuso_manifest(rf_data_path, entradas)
    sha_graph, metodo_graph = sha256_com_reuso_manifest(graph_path, entradas)

    rf_path_resolved = Path(rf_data_path).resolve()
    entrada_rf = None
    for e in entradas:
        try:
            if Path(e.get("caminho", "")).resolve() == rf_path_resolved:
                entrada_rf = e
                break
        except OSError:
            continue
    encontrado_no_manifest = entrada_rf is not None
    sha_manifest = entrada_rf.get("sha256") if entrada_rf else None
    bate = bool(encontrado_no_manifest and sha_manifest == sha_rf)
    return {
        "sha256_rf_data": sha_rf,
        "metodo_sha_rf_data": metodo_rf,
        "sha256_graph": sha_graph,
        "metodo_sha_graph": metodo_graph,
        "manifest": "v4",
        "manifest_path": str(manifest_path),
        "grupo_manifest": grupo,
        "encontrado_no_manifest_v4": encontrado_no_manifest,
        "sha256_manifest_v4": sha_manifest,
        "sha256_rf_data_bate_manifest_v4": bate,
        "nota_graph_file": ("_gpu.pt (arestas terreno-terreno) nao tem entrada propria no "
                             "manifest v4/grupo tensores_cftudo (so os 16 .pt de dado "
                             "canonico enriched_cftudo tem); sha256 gravado por "
                             "proveniencia, sem gate sobre ele."),
    }


def carregar_modulo_congelado(script_path: Path, nome_modulo: str) -> ModuleType:
    """Import a script by path under `nome_modulo`, so its `__main__` block does not run."""
    spec = importlib.util.spec_from_file_location(nome_modulo, script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nome_modulo] = mod
    spec.loader.exec_module(mod)
    return mod


def patch_decoder_gnn(base_dir: Path = GNN_RF_V2) -> None:
    """Replace `PhysicsConstrainedDecoder` by `AffineDecoderV3` in `gnn_rf_model`.

    `GNNRFModel.__init__` looks the class up as a module global at call time, so
    every model built after this call uses the affine decoder.
    """
    p = str(base_dir / "02_models")
    if p not in sys.path:
        sys.path.insert(0, p)
    import gnn_rf_model
    gnn_rf_model.PhysicsConstrainedDecoder = AffineDecoderV3
    # Also in rf_decoder, for parity with the MLP path.
    import rf_decoder
    rf_decoder.PhysicsConstrainedDecoder = AffineDecoderV3


def patch_decoder_mlp(base_dir: Path = GNN_RF_V2) -> None:
    """Replace `rf_decoder.PhysicsConstrainedDecoder`, which the frozen MLP `main()` imports at call time."""
    p =str(base_dir / "02_models")
    if p not in sys.path:
        sys.path.insert(0, p)
    import rf_decoder
    rf_decoder.PhysicsConstrainedDecoder = AffineDecoderV3


def _log_stub(msg: str) -> None:
    print(f"[v3_common] {msg}", flush=True)


def carregar_base_e_particoes(
    mod: ModuleType,
    graph_dir: Path,
    rf_data_file: str,
    graph_file: str,
    max_nodes: int,
    window_anchor: str,
    grid_km: float,
    buffer_km: float,
    split_frac: tuple,
    split_seed: int,
    smoke_geometria: bool,
    mmap: bool,
    precisa_arestas_ter_ter: bool,
    log=_log_stub,
) -> SimpleNamespace:
    """Rebuild the window, partitions and node features of a run with the frozen module's functions.

    Repeats the steps of `mod.main()` up to `parts_local` (train/val/test, local
    window indices) and `x_full` (features plus the distance column standardised
    with training statistics). `precisa_arestas_ter_ter=False` skips loading the
    terrain-terrain edges, which the MLP does not use.
    """
    ET_AT = mod.ET_AT
    ET_TT = mod.ET_TT
    PL_TARGET_MAX_VALID = mod.PL_TARGET_MAX_VALID

    sys.path.append(str(GNN_RF_V2 / "03_training"))
    from spatial_cv import SpatialKFold  # noqa: E402

    rf_path = graph_dir / rf_data_file
    struct_path = graph_dir / graph_file
    load_kw = dict(map_location="cpu", weights_only=False)
    t0 = time.perf_counter()
    rf_data = torch.load(rf_path, mmap=mmap, **load_kw)
    log(f"[+{time.perf_counter()-t0:.1f}s] rf_data carregado (mmap={mmap})")

    ty = rf_data["terrain"].y
    n_ter_total = int(ty.shape[0])

    # Precomputed distance to the nearest antenna, used only if it is not degenerate.
    dist_pre = None
    if hasattr(rf_data["terrain"], "dist_nearest_m"):
        c = rf_data["terrain"].dist_nearest_m.float()
        if c.shape[0] == n_ter_total and float(c.std()) > 1.0:
            dist_pre = c

    feats = rf_data["terrain"].features_raw
    feats = torch.from_numpy(feats) if isinstance(feats, np.ndarray) else feats
    feats = feats.float()

    pos_deg = rf_data["terrain"].pos.float()
    tgt_all = torch.as_tensor(rf_data["terrain"].y).float()
    ant_x = torch.as_tensor(rf_data["antenna"].x).float()
    n_antenna = int(ant_x.shape[0])
    ant_pos = (rf_data["antenna"].pos.float()
               if hasattr(rf_data["antenna"], "pos") else ant_x[:, :2])

    ei_at_full = rf_data[ET_AT].edge_index.long()
    ea_at_full = rf_data[ET_AT].edge_attr.float() if hasattr(rf_data[ET_AT], "edge_attr") else None

    ei_tt_full = ea_tt_full = None
    if precisa_arestas_ter_ter:
        tt_in_rf = ET_TT in rf_data.edge_types
        if tt_in_rf:
            ei_tt_full = rf_data[ET_TT].edge_index.long()
            ea_tt_full = rf_data[ET_TT].edge_attr.float() if hasattr(rf_data[ET_TT], "edge_attr") else None
        else:
            sd = torch.load(struct_path, **load_kw)
            t2t = sd["dem", "adjacent_to", "dem"]
            ei_tt_full = t2t.edge_index.long().clone()
            ea_tt_full = (t2t.edge_attr.float().clone()
                          if hasattr(t2t, "edge_attr") and t2t.edge_attr is not None else None)
            del sd, t2t
            gc.collect()

    del rf_data
    gc.collect()
    log(f"[+{time.perf_counter()-t0:.1f}s] grafo completo: {n_ter_total:,} terrain")

    pos_m_full, geo = mod.latlon_graus_para_metros(pos_deg)
    ancora = (tgt_all[:, 0] < PL_TARGET_MAX_VALID).numpy()  # valid nodes anchor the window
    if max_nodes and max_nodes > 0:
        g_idx = mod.janela_contigua(pos_m_full, max_nodes, ancora)
    else:
        g_idx = np.arange(n_ter_total)
    subamostrado = int(g_idx.size) != n_ter_total
    g_idx_t = torch.from_numpy(g_idx)

    base = HeteroData()
    base["terrain"].pos = pos_deg[g_idx_t]
    base["terrain"].rf_targets = tgt_all[g_idx_t]
    base["antenna"].x = ant_x
    base["antenna"].num_nodes = n_antenna
    x_base = feats[g_idx_t]
    if subamostrado:
        ei_at_full, ea_at_full = bipartite_subgraph(
            (torch.arange(n_antenna), g_idx_t), ei_at_full, ea_at_full,
            relabel_nodes=True, size=(n_antenna, n_ter_total))
        if precisa_arestas_ter_ter:
            ei_tt_full, ea_tt_full = subgraph(
                g_idx_t, ei_tt_full, ea_tt_full, relabel_nodes=True, num_nodes=n_ter_total)
    base[ET_AT].edge_index = ei_at_full
    if ea_at_full is not None:
        base[ET_AT].edge_attr = ea_at_full
    base[mod.ET_TA].edge_index = ei_at_full[[1, 0]]
    if precisa_arestas_ter_ter:
        base[ET_TT].edge_index = ei_tt_full
        base[ET_TT].edge_attr = (ea_tt_full if ea_tt_full is not None
                                 else torch.zeros((ei_tt_full.shape[1], 2), dtype=torch.float32))

    n_ter = int(x_base.shape[0])
    pos_m = pos_m_full[g_idx]
    dist_all = (dist_pre[g_idx_t] if dist_pre is not None else None)
    del tgt_all, pos_deg
    gc.collect()

    ext = min((pos_m[:, 0].max() - pos_m[:, 0].min()) / 1000.0,
              (pos_m[:, 1].max() - pos_m[:, 1].min()) / 1000.0)
    # Smoke runs only: shrink grid and buffer (same ratio) to get at least 6 blocks across.
    if smoke_geometria and ext / max(grid_km, 1e-9) < 6.0:
        novo = float(ext / 6.0)
        buffer_km = novo * (buffer_km / grid_km)
        grid_km = novo
        log(f"[SMOKE] geometria ajustada: grid={grid_km:.3f}km buffer={buffer_km:.3f}km")

    parts_local, split_info = mod.split_espacial_3vias(
        pos_m, grid_km, buffer_km, split_frac, split_seed, SpatialKFold, log)
    for k, v in parts_local.items():
        if len(v) == 0:
            raise RuntimeError(f"Particao '{k}' vazia apos buffer (grid={grid_km}, buffer={buffer_km})")

    # Distance column standardised with training-partition mean and std (std floored at 1 m).
    tr_loc = torch.from_numpy(parts_local["train"])
    d_mean_tr = float(dist_all[tr_loc].mean())
    d_std_tr = max(float(dist_all[tr_loc].std()), 1.0)
    x_full = torch.cat([x_base, ((dist_all - d_mean_tr) / d_std_tr).unsqueeze(1)], dim=1)
    base["terrain"].x = x_full

    return SimpleNamespace(
        base=base, g_idx=g_idx, parts_local=parts_local, split_info=split_info,
        x_full=x_full, ant_x=ant_x, n_antenna=n_antenna, dist_all=dist_all,
        pos_m=pos_m, n_ter=n_ter, n_ter_total=n_ter_total, subamostrado=subamostrado,
        grid_km=grid_km, buffer_km=buffer_km, geo=geo,
    )


def _hash_rng_state() -> dict:
    """SHA-256 of the global RNG states (torch CPU/CUDA, numpy, random).

    Compared before and after an evaluation to show that it consumes no random
    numbers; hashes keep the run JSON small.
    """
    import random as _random
    h: dict = {}
    t_cpu = torch.get_rng_state()
    h["torch_cpu"] = hashlib.sha256(t_cpu.numpy().tobytes()).hexdigest()
    if torch.cuda.is_available():
        h["torch_cuda"] = [
            hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()
            for t in torch.cuda.get_rng_state_all()]
    else:
        h["torch_cuda"] = None
    np_state = np.random.get_state()
    h["numpy"] = hashlib.sha256(
        np_state[1].tobytes() + repr(np_state[2:]).encode()).hexdigest()
    h["python_random"] = hashlib.sha256(repr(_random.getstate()).encode()).hexdigest()
    return h


def _graus_por_relacao(graph_part: HeteroData, mod: ModuleType) -> dict:
    """Degree statistics of the terrain nodes per relation in the induced partition graph.

    AT and TT count in-degree (terrain node as destination), TA counts
    out-degree (terrain node as source).
    """
    def _stats(edge_index: torch.Tensor, lado: str, n_nos: int) -> dict:
        if edge_index is None or edge_index.numel() == 0:
            z = {"min": 0, "max": 0, "media": 0.0, "soma": 0}
            return z
        idx = edge_index[1] if lado == "dst" else edge_index[0]
        deg = torch.zeros(n_nos, dtype=torch.long)
        deg.scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
        return {"min": int(deg.min()), "max": int(deg.max()),
                "media": float(deg.float().mean()), "soma": int(deg.sum())}

    n_ter = int(graph_part["terrain"].x.shape[0])
    n_ant = int(graph_part["antenna"].x.shape[0])
    ei_at = graph_part[mod.ET_AT].edge_index if mod.ET_AT in graph_part.edge_types else None
    ei_tt = graph_part[mod.ET_TT].edge_index if mod.ET_TT in graph_part.edge_types else None
    ei_ta = graph_part[mod.ET_TA].edge_index if mod.ET_TA in graph_part.edge_types else None
    return {
        "AT": _stats(ei_at, "dst", n_ter),
        "TT": _stats(ei_tt, "dst", n_ter),
        "TA": _stats(ei_ta, "src", n_ter),
        "n_terrain": n_ter, "n_antenna": n_ant,
    }


def gerar_predicoes_teste_gnn(
    mod: ModuleType, ctx: SimpleNamespace, model, device: str,
    k_antenna: int, k_terrain: int, eval_batch_size: int, seed: int,
    log=_log_stub, particao: str = "test", disjoint: bool = True,
    ordem_nos: Optional[np.ndarray] = None, medir_diagnostico: bool = False,
) -> dict:
    """Predict every node of one partition with the GNN and return node-aligned arrays.

    Builds the induced subgraph of `particao`, runs `model.eval()` through a
    NeighborLoader (`disjoint=True` isolates the sampling tree of each seed node)
    and returns `idx_global`, `target`, `pred` (clamped to physical ranges),
    `pred_afim` (before the clamp) and `sentinela`. `ordem_nos` permutes the seed
    order (invariance test only). `medir_diagnostico` adds RNG hashes, sampled
    edges per relation and degree statistics.
    """
    torch.manual_seed(seed)
    rng_antes = _hash_rng_state() if medir_diagnostico else None
    loc = torch.from_numpy(ctx.parts_local[particao])
    graph_part, _info = mod.induzir_particao(ctx.base, loc, ctx.n_antenna, log, particao)
    nn_kw = {mod.ET_AT: [k_antenna], mod.ET_TT: [k_terrain], mod.ET_TA: [k_antenna]}
    n_p = int(graph_part["terrain"].x.shape[0])
    seeds = ("terrain", None) if ordem_nos is None else (
        "terrain", torch.as_tensor(ordem_nos, dtype=torch.long))
    loader = NeighborLoader(data=graph_part, num_neighbors=nn_kw,
                            input_nodes=seeds, batch_size=eval_batch_size,
                            shuffle=False, num_workers=0, disjoint=disjoint)
    model.eval()
    P, T, I = [], [], []
    n_arestas = {mod.ET_AT: 0, mod.ET_TT: 0, mod.ET_TA: 0}
    with torch.no_grad():
        for b in loader:
            if medir_diagnostico:
                for et in n_arestas:
                    if et in b.edge_types:
                        n_arestas[et] += int(b[et].edge_index.shape[1])
            b = b.to(device)
            bs = b["terrain"].batch_size
            out = model(b)
            P.append(out["predictions"][:bs].float().detach().cpu())
            T.append(b["terrain"].rf_targets[:bs].float().detach().cpu())
            I.append(b["terrain"].n_id[:bs].cpu())
    # Results are scattered back by node index, so the seed order does not matter.
    po, to, visto = mod._scatter_por_semente(torch.cat(I), torch.cat(P), torch.cat(T), n_p)
    assert bool(visto.all()), f"algum no da particao '{particao}' nao recebeu predicao"

    idx_global = ctx.g_idx[ctx.parts_local[particao]]
    if hasattr(model.decoder, "fisico"):
        # Affine decoder: `po` is the raw affine output; fisico() applies the clamps.
        pred_fisico = model.decoder.fisico(po.to(device)).detach().cpu().numpy().astype(np.float32)
    else:
        # Original decoder: its forward already applies softplus/sigmoid/clamp.
        pred_fisico = po.numpy().astype(np.float32)
    pred_afim = po.numpy().astype(np.float32)
    target = to.numpy().astype(np.float32)
    # Sentinel: path-loss target at or above PL_TARGET_MAX_VALID marks a node without a valid target.
    sentinela = (target[:, 0] >= mod.PL_TARGET_MAX_VALID)
    saida = dict(idx_global=idx_global.astype(np.int64), target=target,
                 pred=pred_fisico, pred_afim=pred_afim, sentinela=sentinela)
    if medir_diagnostico:
        rng_depois = _hash_rng_state()
        graus = _graus_por_relacao(graph_part, mod)
        arestas_por_relacao = {
            "AT": n_arestas[mod.ET_AT], "TT": n_arestas[mod.ET_TT], "TA": n_arestas[mod.ET_TA]}
        soma_graus = {"AT": graus["AT"]["soma"], "TT": graus["TT"]["soma"], "TA": graus["TA"]["soma"]}
        saida["diagnostico"] = {
            "rng_hash_antes": rng_antes,
            "rng_hash_depois": rng_depois,
            "rng_identico": bool(rng_antes == rng_depois),
            "arestas_amostradas_por_relacao": arestas_por_relacao,
            "soma_graus_das_sementes_por_relacao": soma_graus,
            "arestas_bate_com_soma_graus": {
                k: bool(arestas_por_relacao[k] == soma_graus[k]) for k in ("AT", "TT", "TA")},
            "graus_por_relacao": graus,
            "k_antenna_usado": int(k_antenna), "k_terrain_usado": int(k_terrain),
            "disjoint_usado": bool(disjoint),
        }
    return saida


def gerar_predicoes_teste_mlp(
    mod: ModuleType, ctx: SimpleNamespace, model, device: str,
    eval_batch_size: int, seed: int, log=_log_stub, particao: str = "test",
) -> dict:
    """Graph-free counterpart of `gerar_predicoes_teste_gnn`: slices `x_full` and runs the MLP in batches."""
    torch.manual_seed(seed)
    loc = torch.from_numpy(ctx.parts_local[particao])
    x_test = ctx.x_full[loc]
    y_test = ctx.base["terrain"].rf_targets[loc]
    n_p = int(x_test.shape[0])
    model.eval()
    P, T = [], []
    with torch.no_grad():
        for i in range(0, n_p, eval_batch_size):
            xb = x_test[i:i + eval_batch_size].to(device)
            out = model(xb)
            P.append(out["predictions"].float().detach().cpu())
            T.append(y_test[i:i + eval_batch_size].float())
    po = torch.cat(P)
    to = torch.cat(T)

    idx_global = ctx.g_idx[ctx.parts_local[particao]]
    if hasattr(model.decoder, "fisico"):
        pred_fisico = model.decoder.fisico(po.to(device)).detach().cpu().numpy().astype(np.float32)
    else:
        pred_fisico = po.numpy().astype(np.float32)
    pred_afim = po.numpy().astype(np.float32)
    target = to.numpy().astype(np.float32)
    sentinela = (target[:, 0] >= mod.PL_TARGET_MAX_VALID)
    return dict(idx_global=idx_global.astype(np.int64), target=target,
                pred=pred_fisico, pred_afim=pred_afim, sentinela=sentinela)


def testar_invariancia_avaliacao_completa(
    mod: ModuleType, ctx: SimpleNamespace, model, device: str, seed: int,
    eval_batch_size_a: int, eval_batch_size_b: int,
    particao: str = "test", log=_log_stub,
) -> dict:
    """Test that complete-neighbourhood GNN predictions do not depend on batch size or seed order.

    With k = -1 on all relations and `disjoint=True`, no neighbours are drawn.
    The base evaluation (natural order, batch size A) is repeated once to
    measure the numerical floor of the GPU (CUDA reductions are not bit-exact),
    then compared with 2 permuted orders x 2 batch sizes. Pass criterion: the
    largest per-node difference is <= 1e-4 dB and <= 3 x the measured floor.
    """
    K = -1
    saida_base = gerar_predicoes_teste_gnn(
        mod=mod, ctx=ctx, model=model, device=device, k_antenna=K, k_terrain=K,
        eval_batch_size=eval_batch_size_a, seed=seed, log=log,
        particao=particao, disjoint=True, ordem_nos=None)
    n_p = int(saida_base["idx_global"].shape[0])

    saida_piso = gerar_predicoes_teste_gnn(
        mod=mod, ctx=ctx, model=model, device=device, k_antenna=K, k_terrain=K,
        eval_batch_size=eval_batch_size_a, seed=seed, log=log,
        particao=particao, disjoint=True, ordem_nos=None)

    rng1 = np.random.default_rng(seed)
    ordem1 = rng1.permutation(n_p)
    rng2 = np.random.default_rng(seed + 1)
    ordem2 = rng2.permutation(n_p)

    combinacoes = {
        "ordem1_bsA": dict(eval_batch_size=eval_batch_size_a, ordem_nos=ordem1),
        "ordem1_bsB": dict(eval_batch_size=eval_batch_size_b, ordem_nos=ordem1),
        "ordem2_bsA": dict(eval_batch_size=eval_batch_size_a, ordem_nos=ordem2),
        "ordem2_bsB": dict(eval_batch_size=eval_batch_size_b, ordem_nos=ordem2),
    }

    def _cmp(s1, s2):
        idx_bate = bool(np.array_equal(s1["idx_global"], s2["idx_global"]))
        if not idx_bate:
            return {"idx_global_identico": False, "diff_max_afim": None, "diff_max_fisico": None}
        return {
            "idx_global_identico": True,
            "diff_max_afim": float(np.abs(s1["pred_afim"] - s2["pred_afim"]).max()),
            "diff_max_fisico": float(np.abs(s1["pred"] - s2["pred"]).max()),
        }

    piso = _cmp(saida_base, saida_piso)
    piso_medido = piso["diff_max_fisico"] if piso["diff_max_fisico"] is not None else 0.0

    comparacoes = {}
    for nome, kw in combinacoes.items():
        saida_i = gerar_predicoes_teste_gnn(
            mod=mod, ctx=ctx, model=model, device=device, k_antenna=K, k_terrain=K,
            eval_batch_size=kw["eval_batch_size"], seed=seed, log=log,
            particao=particao, disjoint=True, ordem_nos=kw["ordem_nos"])
        comparacoes[nome] = _cmp(saida_base, saida_i)

    diffs_fisico = [c["diff_max_fisico"] for c in comparacoes.values()
                    if c["diff_max_fisico"] is not None]
    diff_max_geral = max(diffs_fisico) if diffs_fisico else None
    limiar_dB = 1e-4
    limiar_3x_piso = 3.0 * piso_medido
    passou = bool(
        all(c["idx_global_identico"] for c in comparacoes.values())
        and diff_max_geral is not None
        and diff_max_geral <= limiar_dB
        and diff_max_geral <= limiar_3x_piso)
    return {
        "particao": particao,
        "n_nos": n_p,
        "regra": "vizinhanca COMPLETA (k_antenna=k_terrain=-1), disjoint=True, sempre",
        "eval_batch_size_a": int(eval_batch_size_a),
        "eval_batch_size_b": int(eval_batch_size_b),
        "piso_numerico_medido_mesma_ordem_mesma_bs": piso,
        "comparacoes_ordem_x_batch_size": comparacoes,
        "diff_max_fisico_entre_todas_combinacoes": diff_max_geral,
        "limiar_absoluto_dB": limiar_dB,
        "limiar_3x_piso_medido": limiar_3x_piso,
        "passou_criterio": passou,
        "nota": ("with the COMPLETE neighbourhood no neighbours are drawn (real degree <= 9 on TT, "
                 "<= 5 on AT): the only expected source of difference is the numerical reduction "
                 "noise of the GPU (not bit-exact between calls), measured here as the 'piso' "
                 "(floor) and used as the scale reference of the criterion."),
    }


# MLP widths matched to the GNN parameter count. Default: matched to the
# effective GNN count (parameters that receive a gradient); the earlier widths
# matched the nominal count and are kept for the sensitivity arm.
MLP_LARGURAS_NOVAS = (654, 654, 600, 636, 256)
N_PARAMS_ALVO_NOMINAL_NOVO = 1_484_067          # parameter count of the MLP with the new widths
N_PARAMS_ALVO_EFETIVO_GNN = 1_484_037           # effective GNN parameter count at fan-out 1
MLP_LARGURAS_ANTIGAS = (712, 712, 720, 688, 256)
N_PARAMS_ALVO_NOMINAL_ANTIGO = 1_812_515
FONTE_LARGURAS = (
    "effective parameter count of the GNN measured by backward on a training batch, "
    "and the MLP widths solved to match that count")


def patch_larguras_mlp(mod: ModuleType, larguras: tuple, alvo_nominal: int) -> None:
    """Set the MLP widths and the parameter-parity target that the frozen `main()` reads as module globals."""
    mod.MLP_LARGURAS = tuple(larguras)
    mod.N_PARAMS_ALVO_GNN = int(alvo_nominal)


class _NeighborLoaderDisjointAuto(NeighborLoader):
    """NeighborLoader installed as `mod.NeighborLoader` in the frozen GNN trainer.

    The frozen `make_loader()` passes `shuffle=True` only for training, so
    `shuffle=False` identifies validation/test, which always get `disjoint=True`;
    training gets it only when `ativar_treino` is set.
    """
    ativar_treino = False

    def __init__(self, *args, **kwargs):
        shuffle = kwargs.get("shuffle", None)
        aplica_val_test = (shuffle is False)
        aplica_treino = (shuffle is True and _NeighborLoaderDisjointAuto.ativar_treino)
        if (aplica_val_test or aplica_treino) and "disjoint" not in kwargs:
            kwargs["disjoint"] = True
        super().__init__(*args, **kwargs)


def patch_neighborloader_disjoint(mod: ModuleType, disjoint_treino: bool = False) -> None:
    _NeighborLoaderDisjointAuto.ativar_treino = bool(disjoint_treino)
    mod.NeighborLoader = _NeighborLoaderDisjointAuto


def contar_parametros_efetivos_via_backward(model, preds: torch.Tensor) -> dict:
    """Count trainable parameters that receive a nonzero gradient from `preds.sum().backward()`.

    A parameter is without gradient if its grad is None (outside the
    computational graph) or exactly zero (e.g. an aggregation that is always
    empty); structural zeros do not depend on the loss, so `.sum()` suffices.
    """
    for p in model.parameters():
        p.grad = None
    preds.sum().backward()
    mortos: dict = {}
    n_total = 0
    for name, p in model.named_parameters():
        n = int(p.numel())
        n_total += n
        if not p.requires_grad:
            continue
        if p.grad is None:
            mortos[name] = {"numel": n, "shape": list(p.shape), "grad": "None"}
        elif int(torch.count_nonzero(p.grad)) == 0:
            mortos[name] = {"numel": n, "shape": list(p.shape), "grad": "zero"}
    n_mortos = sum(v["numel"] for v in mortos.values())
    return {
        "n_params_total": n_total,
        "n_params_sem_gradiente": n_mortos,
        "n_params_efetivos": n_total - n_mortos,
        "parametros_sem_gradiente": mortos,
    }


def medir_capacidade_efetiva_gnn(
    mod: ModuleType, ctx: SimpleNamespace, model, device: str,
    k_antenna: int, k_terrain: int, batch_size: int, seed: int, log=_log_stub,
) -> dict:
    """Effective capacity of the GNN on one real training batch, on a deep copy of the model."""
    import copy
    torch.manual_seed(seed)
    loc = torch.from_numpy(ctx.parts_local["train"])
    graph_train, _info = mod.induzir_particao(ctx.base, loc, ctx.n_antenna, log, "train")
    nn_kw = {mod.ET_AT: [k_antenna], mod.ET_TT: [k_terrain], mod.ET_TA: [k_antenna]}
    loader = NeighborLoader(data=graph_train, num_neighbors=nn_kw,
                            input_nodes=("terrain", None), batch_size=batch_size,
                            shuffle=True, num_workers=0)
    batch = next(iter(loader)).to(device)
    model_probe = copy.deepcopy(model).to(device)
    model_probe.train()
    out = model_probe(batch)
    bs = batch["terrain"].batch_size
    resultado = contar_parametros_efetivos_via_backward(model_probe, out["predictions"][:bs])
    del model_probe, batch, loader, graph_train
    return resultado


def medir_capacidade_efetiva_mlp(
    ctx: SimpleNamespace, model, device: str, batch_size: int, seed: int,
) -> dict:
    """Effective capacity of the MLP on one real training batch, on a deep copy of the model."""
    import copy
    torch.manual_seed(seed)
    loc = torch.from_numpy(ctx.parts_local["train"])
    xb = ctx.x_full[loc][:batch_size].to(device)
    model_probe = copy.deepcopy(model).to(device)
    model_probe.train()
    out = model_probe(xb)
    resultado = contar_parametros_efetivos_via_backward(model_probe, out["predictions"])
    del model_probe, xb
    return resultado


class InstrumentacaoTreino:
    """Context manager that counts training events by wrapping four library methods.

    - `AdamW.step`: weight updates actually applied (GradScaler skips the step
      when the gradient is not finite);
    - `GradScaler.update`: one call per step, loss scale before and after;
    - `clip_grad_norm_`: total gradient norm before clipping;
    - `HeteroConv.forward`: calls without `edge_attr_dict` (the encoder's
      fallback branch).
    The originals are restored on exit; `resultado()` returns the summary.
    """

    def __init__(self):
        self.n_atualizacoes_efetivas = 0
        self.escalas = []  # (before, after) loss scale per update() call
        self.normas_pre_clip = []
        self.n_except_edge_attr = 0
        self._orig = {}

    def __enter__(self):
        import torch.optim as optim
        import torch.amp as amp
        import torch.nn.utils as nnutils
        import torch_geometric.nn as pyg_nn

        self._orig["adamw_step"] = optim.AdamW.step
        self._orig["scaler_update"] = amp.GradScaler.update
        self._orig["clip"] = nnutils.clip_grad_norm_
        self._orig["heteroconv_forward"] = pyg_nn.HeteroConv.forward
        inst = self

        def step_patched(opt_self, *a, **kw):
            inst.n_atualizacoes_efetivas += 1
            return inst._orig["adamw_step"](opt_self, *a, **kw)

        def update_patched(scaler_self, *a, **kw):
            antes = float(scaler_self.get_scale())
            r = inst._orig["scaler_update"](scaler_self, *a, **kw)
            depois = float(scaler_self.get_scale())
            inst.escalas.append((antes, depois))
            return r

        def clip_patched(*a, **kw):
            norma = inst._orig["clip"](*a, **kw)
            inst.normas_pre_clip.append(float(norma))
            return norma

        def heteroconv_forward_patched(hc_self, x_dict, edge_index_dict,
                                        edge_attr_dict=None, *a, **kw):
            if edge_attr_dict is None:
                inst.n_except_edge_attr += 1
            return inst._orig["heteroconv_forward"](
                hc_self, x_dict, edge_index_dict, edge_attr_dict, *a, **kw)

        optim.AdamW.step = step_patched
        amp.GradScaler.update = update_patched
        nnutils.clip_grad_norm_ = clip_patched
        pyg_nn.HeteroConv.forward = heteroconv_forward_patched
        return self

    def __exit__(self, *exc):
        import torch.optim as optim
        import torch.amp as amp
        import torch.nn.utils as nnutils
        import torch_geometric.nn as pyg_nn
        optim.AdamW.step = self._orig["adamw_step"]
        amp.GradScaler.update = self._orig["scaler_update"]
        nnutils.clip_grad_norm_ = self._orig["clip"]
        pyg_nn.HeteroConv.forward = self._orig["heteroconv_forward"]
        return False

    def resultado(self, max_norm: float = 0.5) -> dict:
        n_passos = len(self.escalas)
        n_atualizados = self.n_atualizacoes_efetivas
        # A skipped step is one where GradScaler lowered the loss scale.
        pulou = [depois < antes for (antes, depois) in self.escalas]
        idx_primeiro_sem_pulo = next((i for i, p in enumerate(pulou) if not p), None)
        pulos_pos_aquecimento = (
            int(sum(pulou[idx_primeiro_sem_pulo:])) if idx_primeiro_sem_pulo is not None
            else int(sum(pulou)))
        idx_ultimo_pulo = max((i for i, p in enumerate(pulou) if p), default=None)
        escalas_iniciais = [a for (a, _) in self.escalas]
        escalas_finais = [d for (_, d) in self.escalas]
        normas = np.asarray(self.normas_pre_clip, dtype=np.float64)
        finitas = normas[np.isfinite(normas)] if normas.size else normas
        n_nao_finitas = int(normas.size - finitas.size)
        return {
            "n_passos": n_passos,
            "n_atualizacoes_efetivas": n_atualizados,
            "n_pulados_total": n_passos - n_atualizados,
            "n_pulados_pos_aquecimento": pulos_pos_aquecimento,
            "definicao_pos_aquecimento": (
                "pulos apos o indice do 1o passo SEM pulo (1a atualizacao bem-sucedida)"),
            "indice_primeiro_passo_sem_pulo": idx_primeiro_sem_pulo,
            "indice_ultimo_pulo": idx_ultimo_pulo,
            "escala_inicial": escalas_iniciais[0] if escalas_iniciais else None,
            "escala_final": escalas_finais[-1] if escalas_finais else None,
            "escala_minima": min(escalas_finais) if escalas_finais else None,
            "norma_gradiente_pre_clip_p50": (
                float(np.percentile(finitas, 50)) if finitas.size else None),
            "norma_gradiente_pre_clip_p99": (
                float(np.percentile(finitas, 99)) if finitas.size else None),
            "n_normas_nao_finitas": n_nao_finitas,
            "nota_normas_nao_finitas": (
                "a non-finite pre-clip norm (inf/nan) occurs on the steps that the GradScaler "
                "detects and skips; excluded from P50/P99 and counted here"),
            "fracao_passos_clipados": (
                float(np.mean(finitas > max_norm)) if finitas.size else None),
            "n_except_edge_attr_encoder": self.n_except_edge_attr,
        }


_CANAL_CLAMP = {0: (0.0, 200.0), 1: (0.0, 50.0), 2: (0.0, 30.0), 3: (-150.0, 0.0)}  # production clamps, dB / dBm


def diagnostico_saida_canais(pred_afim: np.ndarray, pred_fisico: np.ndarray,
                              target: np.ndarray,
                              sentinela: Optional[np.ndarray] = None) -> dict:
    """Per-channel output statistics (channels 0-3): out-of-range fraction, p99 excess, MAE before and after the clamp.

    The sentinel target of channel 0 (>= PL_TARGET_MAX_VALID, 299 dB) lies outside
    the clamp range by construction; when `sentinela` is given, the primary
    channel-0 numbers use valid nodes only, and the unmasked numbers are kept
    with the suffix `_todos_com_sentinela`. Channel 1 is also split by target == 0.
    """
    out = {}
    for canal, (lo, hi) in _CANAL_CLAMP.items():
        raw = pred_afim[:, canal].astype(np.float64)
        clampado = pred_fisico[:, canal].astype(np.float64)
        tgt = target[:, canal].astype(np.float64)

        def _bloco(idx=None):
            r = raw if idx is None else raw[idx]
            c = clampado if idx is None else clampado[idx]
            t = tgt if idx is None else tgt[idx]
            if r.size == 0:
                return {"n": 0, "fracao_fora_da_faixa": None, "p99_excesso_dB_ou_dBm": None,
                        "mae_bruto": None, "mae_pos_clamp": None, "peso_do_clamp_pct": None}
            fora = (r < lo) | (r > hi)
            excesso = np.maximum(lo - r, 0.0) + np.maximum(r - hi, 0.0)
            mb = float(np.mean(np.abs(r - t)))
            mc = float(np.mean(np.abs(c - t)))
            return {
                "n": int(r.size),
                "fracao_fora_da_faixa": float(np.mean(fora)),
                "p99_excesso_dB_ou_dBm": float(np.percentile(excesso, 99)),
                "mae_bruto": mb,
                "mae_pos_clamp": mc,
                "peso_do_clamp_pct": float((mb - mc) / mc * 100.0) if mc > 0 else None,
            }

        if canal == 0 and sentinela is not None:
            valido_idx = ~sentinela
            canal_out = _bloco(valido_idx)
            canal_out["frac_alvo_sentinela"] = float(np.mean(sentinela))
            canal_out["definicao_sentinela"] = "target[:,0] >= PL_TARGET_MAX_VALID (299 dB)"
            todos = _bloco(None)
            for k, v in todos.items():
                canal_out[f"{k}_todos_com_sentinela"] = v
        else:
            canal_out = _bloco(None)

        if canal == 1:
            mask0 = (tgt == 0.0)
            mask_pos = ~mask0
            canal_out["frac_alvo_exatamente_zero"] = float(np.mean(mask0))
            if mask0.any():
                canal_out["mae_bruto_alvo_zero"] = float(np.mean(np.abs(raw[mask0] - tgt[mask0])))
                canal_out["mae_pos_clamp_alvo_zero"] = float(np.mean(np.abs(clampado[mask0] - tgt[mask0])))
            if mask_pos.any():
                canal_out["mae_bruto_alvo_maior_zero"] = float(np.mean(np.abs(raw[mask_pos] - tgt[mask_pos])))
                canal_out["mae_pos_clamp_alvo_maior_zero"] = float(np.mean(np.abs(clampado[mask_pos] - tgt[mask_pos])))
        out[f"canal_{canal}"] = canal_out
    return out


def mae_rssi_por_populacao(pred_fisico: np.ndarray, target: np.ndarray,
                            sentinela: np.ndarray) -> dict:
    """MAE of RSSI (channel 3, dB) over all nodes, valid nodes and sentinel nodes."""
    err =np.abs(pred_fisico[:, 3].astype(np.float64) - target[:, 3].astype(np.float64))
    valido = ~sentinela
    out = {"todos": {"n": int(err.size), "mae_rssi_db": float(err.mean())}}
    for nome, mask in (("valido", valido), ("sentinela", sentinela)):
        if mask.any():
            out[nome] = {"n": int(mask.sum()), "mae_rssi_db": float(err[mask].mean())}
        else:
            out[nome] = {"n": 0, "mae_rssi_db": None}
    return out
