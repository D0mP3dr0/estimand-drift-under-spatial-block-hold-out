"""Structure of the terrain-terrain and antenna-terrain edges in two cells (Bauru Q1, Lins Q1).

Terrain-terrain edges come from the graph tensor `<city>_v19_Q1_gpu.pt` (PyG HeteroData, node type
'dem', edge type ('dem', 'adjacent_to', 'dem')). The script counts self-loops, exact duplicate
edges and edges whose reverse (j, i) also exists, and summarises out- and in-degrees. It then
draws 10,000 edges (numpy seed 42) and measures their coordinate offsets: the modal step in x
and y gives the grid spacing, and the share of offsets of at most one step in each direction
(the 8-cell Moore neighbourhood) separates a grid adjacency from a k-nearest-neighbour graph.
Antenna-terrain edges come from the reference-field tensor
`transfer_dataset_<city>_v19_Q1_enriched_cftudo.pt` (edge type ('antenna', 'propagates_to',
'terrain')): number of edges, fan-out per antenna and number of antennas reaching each terrain
node. The script also records whether that tensor has any terrain-terrain edge type.

Inputs: the four tensors under BASE (not distributed; available on request, digests in the
manifest). They are loaded with mmap; duplicates and symmetry use the packed key i*N + j and a
sort instead of pairwise comparisons.
Output: the JSON path given as first argument (default 3.6_arestas_terreno.json).
Usage: python 3.6_arestas_terreno_script.py [output.json]
"""
import sys, os, json, time
import numpy as np
import torch

OUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "3.6_arestas_terreno.json"
SEED = 42
rng = np.random.default_rng(SEED)

BASE = "/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3"
CIDADES = ["bauru", "lins"]

resultado = {"seed": SEED, "celulas": {}}


def carregar(path):
    t0 = time.time()
    d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    dt = time.time() - t0
    return d, dt


def analisar_terreno_terreno(d, tag):
    store = d[("dem", "adjacent_to", "dem")]
    ei = store.edge_index
    n_dem = d["dem"].num_nodes
    E = ei.shape[1]

    src_t = ei[0]
    dst_t = ei[1]

    self_loops = int((src_t == dst_t).sum().item())

    out_deg = torch.bincount(src_t, minlength=n_dem).numpy()
    in_deg = torch.bincount(dst_t, minlength=n_dem).numpy()

    def resumo(arr):
        return {
            "min": int(arr.min()),
            "mediana": float(np.median(arr)),
            "media": float(arr.mean()),
            "max": int(arr.max()),
        }

    src_np = src_t.numpy().astype(np.int64)
    dst_np = dst_t.numpy().astype(np.int64)
    # One int64 key per directed edge (i, j): i*N + j; duplicates are repeated keys.
    packed = src_np * np.int64(n_dem) + dst_np
    packed_sorted = np.sort(packed)
    n_unicos = int(np.unique(packed_sorted).shape[0])
    n_duplicatas = int(E - n_unicos)

    # Reverse key j*N + i, looked up in the sorted keys to find edges whose (j, i) exists.
    packed_rev = dst_np * np.int64(n_dem) + src_np
    idx = np.searchsorted(packed_sorted, packed_rev)
    idx_clip = np.clip(idx, 0, packed_sorted.shape[0] - 1)
    achou = packed_sorted[idx_clip] == packed_rev
    n_simetricas = int(achou.sum())
    frac_simetrica = n_simetricas / E

    amostra_idx = rng.choice(E, size=min(10_000, E), replace=False)
    pos = d["dem"].pos.numpy()
    s = src_np[amostra_idx]
    t = dst_np[amostra_idx]
    delta = pos[t] - pos[s]  # offset (dx, dy) in degrees; pos is (lon, lat)

    grid_shape = d["dem"].grid_shape if "grid_shape" in d["dem"] else None

    # Grid step = most frequent non-zero absolute offset (rounded to 1e-6 degrees), per axis.
    nao_loop = ~((delta[:, 0] == 0) & (delta[:, 1] == 0))
    delta_nz = delta[nao_loop]
    dx_abs_uniq, dx_counts = np.unique(np.round(np.abs(delta_nz[:, 0]), 6), return_counts=True)
    dy_abs_uniq, dy_counts = np.unique(np.round(np.abs(delta_nz[:, 1]), 6), return_counts=True)
    passo_dx = float(dx_abs_uniq[np.argmax(dx_counts)]) if len(dx_abs_uniq) else None
    passo_dy = float(dy_abs_uniq[np.argmax(dy_counts)]) if len(dy_abs_uniq) else None

    # An offset is on the grid if it is an integer multiple of the step within a relative tolerance of 1e-4;
    # Moore neighbours are on-grid offsets of at most one step per axis, excluding (0, 0).
    tol = 1e-4
    dx_steps = np.round(delta_nz[:, 0] / passo_dx) if passo_dx else None
    dy_steps = np.round(delta_nz[:, 1] / passo_dy) if passo_dy else None
    moore_ok = None
    if passo_dx and passo_dy:
        resid_x = np.abs(delta_nz[:, 0] - dx_steps * passo_dx)
        resid_y = np.abs(delta_nz[:, 1] - dy_steps * passo_dy)
        na_grade = (resid_x < tol * abs(passo_dx if passo_dx else 1) + 1e-9) & (
            resid_y < tol * abs(passo_dy if passo_dy else 1) + 1e-9
        )
        moore = (np.abs(dx_steps) <= 1) & (np.abs(dy_steps) <= 1) & na_grade & ~((dx_steps == 0) & (dy_steps == 0))
        moore_ok = float(moore.mean())

    dist_amostra = np.sqrt((delta[:, 0]) ** 2 + (delta[:, 1]) ** 2)
    n_zero = int(((delta[:, 0] == 0) & (delta[:, 1] == 0)).sum())

    saida = {
        "arquivo": tag["path"],
        "n_nos_terreno": int(n_dem),
        "n_arestas_terreno_terreno": int(E),
        "n_autolacos": self_loops,
        "frac_autolacos": self_loops / E,
        "n_duplicatas_exatas_i_j": n_duplicatas,
        "n_arestas_com_reverso_j_i": n_simetricas,
        "frac_simetrica": frac_simetrica,
        "grau_saida": resumo(out_deg),
        "grau_entrada": resumo(in_deg),
        "mean_degree_E_sobre_N": E / n_dem,
        "amostra_pos_delta": {
            "n_amostra": int(len(amostra_idx)),
            "n_delta_zero_autolaco": n_zero,
            "passo_grade_dx_estimado": passo_dx,
            "passo_grade_dy_estimado": passo_dy,
            "frac_vizinhanca_moore_8_mais_autolaco": moore_ok,
            "dist_min": float(dist_amostra[dist_amostra > 0].min()) if (dist_amostra > 0).any() else None,
            "dist_max": float(dist_amostra.max()),
        },
        "grid_shape": grid_shape.tolist() if hasattr(grid_shape, "tolist") else grid_shape,
    }
    return saida


def analisar_antena_terreno(d, tag):
    store = d[("antenna", "propagates_to", "terrain")]
    ei = store.edge_index
    n_antena = d["antenna"].num_nodes
    n_terreno = d["terrain"].x.shape[0]
    E = ei.shape[1]
    src_ant = ei[0].numpy().astype(np.int64)  # antenna index
    dst_terr = ei[1].numpy().astype(np.int64)  # terrain node index

    fanout = np.bincount(src_ant, minlength=n_antena)

    def resumo(arr):
        return {
            "min": int(arr.min()),
            "mediana": float(np.median(arr)),
            "media": float(arr.mean()),
            "max": int(arr.max()),
        }

    in_deg_terreno = np.bincount(dst_terr, minlength=n_terreno)

    saida = {
        "arquivo": tag["path"],
        "n_antenas": int(n_antena),
        "n_nos_terreno": int(n_terreno),
        "n_arestas_antena_terreno": int(E),
        "direcao": "antena(src) -> terreno(dst), unidirecional (nao ha terreno->antena)",
        "fanout_por_antena": resumo(fanout),
        "grau_entrada_terreno": resumo(in_deg_terreno),
    }
    return saida


for cidade in CIDADES:
    resultado["celulas"][cidade] = {}
    path_gpu = os.path.join(BASE, f"{cidade}_v19_Q1_gpu.pt")
    path_enriched = os.path.join(BASE, f"transfer_dataset_{cidade}_v19_Q1_enriched_cftudo.pt")

    d_gpu, dt1 = carregar(path_gpu)
    tt = analisar_terreno_terreno(d_gpu, {"path": path_gpu})
    tt["tempo_carga_s"] = dt1
    tt["node_types_arquivo"] = list(d_gpu.node_types)
    tt["edge_types_arquivo"] = [str(e) for e in d_gpu.edge_types]
    del d_gpu

    d_enr, dt2 = carregar(path_enriched)
    at = analisar_antena_terreno(d_enr, {"path": path_enriched})
    at["tempo_carga_s"] = dt2
    at["node_types_arquivo"] = list(d_enr.node_types)
    at["edge_types_arquivo"] = [str(e) for e in d_enr.edge_types]
    # True when no edge type of the reference-field tensor connects 'terrain' to 'terrain'
    # (the first clause, with the placeholder relation "?", is always true).
    achado_sem_terreno_terreno = ("terrain", "?", "terrain") not in [
        tuple(e) for e in d_enr.edge_types
    ] and not any("terrain" == e[0] == e[2] for e in d_enr.edge_types)
    at["achado_enriched_sem_aresta_terreno_terreno"] = achado_sem_terreno_terreno
    del d_enr

    resultado["celulas"][cidade]["terreno_terreno"] = tt
    resultado["celulas"][cidade]["antena_terreno"] = at

with open(OUT_JSON, "w") as f:
    json.dump(resultado, f, indent=2, ensure_ascii=False)

print("OK ->", OUT_JSON)
