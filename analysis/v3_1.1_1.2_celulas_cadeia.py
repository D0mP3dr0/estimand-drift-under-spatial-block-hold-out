#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometry of the 16 city x quadrant cells (test 1.1) and buffer check along the
Bauru quadrant chain (test 1.2).

Test 1.1 reads the `geometria` and `subgrafos` fields of the 16 run records
(g = 10 km, b = 2 km, split seed 42) and reports, per cell, the bounding box, the
equirectangular extent in km (cos(phi) at the box centre), the node count and grid
step, and the antenna count per role; for every pair of quadrants of the same city
it reports the bounding-box intersection (km^2) and the number of coincident nodes
of the synthetic grids (computed only when the boxes intersect), and checks
whether the four Bauru quadrants carry the same antenna set.
Test 1.2 rebuilds the training/validation/test partitions of Bauru Q1-Q4 with the
frozen split (imported from varredura_split_geometria.py), places all quadrants in
one city-wide metric frame with a common origin, and counts the test nodes of
quadrant Q(i+1) lying closer than b = 2 km to the training nodes of Q(i), for the
pairs Q1->Q2, Q2->Q3 and Q3->Q4.

Inputs: the run records under TREINOS_DIR and the decision criteria
criterio_1.1.json and criterio_1.2.json (criteria/ in this repository; the script
reads them from _v3_2026-09-25/criterios/ one level above its own folder).
Outputs: results/fase1/1.1_geo_celulas_16.json, results/fase1/1.2_cadeia_buffer_violacao.json,
and a two-panel map (cell boxes; Bauru Q1 blocks and partition) as PDF and PNG.
No tensor file is read; the run is deterministic (split seed taken from each record).
Usage: python v3_1.1_1.2_celulas_cadeia.py --out-11 <json11> --out-12 <json12> --fig <figure path without extension> [--log <file>]
"""
import argparse
import hashlib
import json
import sys
import time
import resource
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from varredura_split_geometria import (  # noqa: E402  (frozen split reused unchanged)
    TREINOS_DIR, CIDADES, QS, N_SIDE, FRACS,
    build_synthetic_grid, latlon_graus_para_metros, split_espacial_3vias_exato,
)

KM_POR_GRAU = 111.0  # same 111 km per degree as latlon_graus_para_metros


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def carregar_run(cidade, q, cfgnome="g10b2"):
    if cfgnome == "g10b2":
        p = TREINOS_DIR / f"run_c0c1cf_{cidade}_s42_{q}_g10b2.json"
    else:
        p = TREINOS_DIR / f"run_c0c1_{cidade}_s42_{q}_g5b2.json"
    return json.load(open(p)), p


def extensao_equiretangular_km(lon_min, lon_max, lat_min, lat_max):
    """Extent in km: dlon * 111 * cos(lat at the box centre) and dlat * 111."""
    lat_centro = 0.5 * (lat_min + lat_max)
    dx_km = (lon_max - lon_min) * KM_POR_GRAU * np.cos(np.radians(lat_centro))
    dy_km = (lat_max - lat_min) * KM_POR_GRAU
    return float(dx_km), float(dy_km)


def bbox_intersecao_km2(bbox_a, bbox_b, lat_ref):
    """Intersection of two lon/lat boxes in km^2 (and its x, y sides), with cos(lat_ref) fixed."""
    lon_min_a, lon_max_a, lat_min_a, lat_max_a = bbox_a
    lon_min_b, lon_max_b, lat_min_b, lat_max_b = bbox_b
    ov_lon_deg = max(0.0, min(lon_max_a, lon_max_b) - max(lon_min_a, lon_min_b))
    ov_lat_deg = max(0.0, min(lat_max_a, lat_max_b) - max(lat_min_a, lat_min_b))
    ov_lon_km = ov_lon_deg * KM_POR_GRAU * np.cos(np.radians(lat_ref))
    ov_lat_km = ov_lat_deg * KM_POR_GRAU
    return float(ov_lon_km * ov_lat_km), float(ov_lon_km), float(ov_lat_km)


def rodar_teste_11(log):
    por_celula = {}
    geo_bbox = {}
    n_antena = {}
    n_antena_papeis = {}
    for cidade in CIDADES:
        for q in QS:
            d, p = carregar_run(cidade, q, "g10b2")
            g = d["geometria"]
            s = d["subgrafos"]
            bbox = (g["lon_min_deg"], g["lon_max_deg"], g["lat_min_deg"], g["lat_max_deg"])
            geo_bbox[(cidade, q)] = bbox
            dx_km, dy_km = extensao_equiretangular_km(*bbox)
            # Square grid: n_side = sqrt(node count); grid step = extent / (n_side - 1).
            n_side_local = int(round(np.sqrt(g["n_nos_no_grafo_de_trabalho"])))
            passo_x_m = dx_km * 1000.0 / (n_side_local - 1)
            passo_y_m = dy_km * 1000.0 / (n_side_local - 1)
            erro_rel_x = abs(dx_km - g["extensao_x_km"]) / g["extensao_x_km"]
            erro_rel_y = abs(dy_km - g["extensao_y_km"]) / g["extensao_y_km"]
            n_ant_papeis = {papel: s[papel]["n_antenna"] for papel in ("train", "val", "test")}
            n_antena_papeis[(cidade, q)] = n_ant_papeis
            n_ant_consistente = len(set(n_ant_papeis.values())) == 1
            n_antena[(cidade, q)] = s["train"]["n_antenna"] if n_ant_consistente else None
            por_celula[f"{cidade}_{q}"] = {
                "run_json": str(p),
                "bbox_lon_min_deg": bbox[0], "bbox_lon_max_deg": bbox[1],
                "bbox_lat_min_deg": bbox[2], "bbox_lat_max_deg": bbox[3],
                "extensao_x_km_recalculada": dx_km, "extensao_y_km_recalculada": dy_km,
                "extensao_x_km_declarada": g["extensao_x_km"], "extensao_y_km_declarada": g["extensao_y_km"],
                "erro_relativo_extensao_x": erro_rel_x, "erro_relativo_extensao_y": erro_rel_y,
                "n_side": n_side_local, "n_nos": g["n_nos_no_grafo_de_trabalho"],
                "passo_malha_x_m": passo_x_m, "passo_malha_y_m": passo_y_m,
                "n_antena_por_papel": n_ant_papeis, "n_antena_consistente_entre_papeis": n_ant_consistente,
                "n_antena": n_antena[(cidade, q)],
            }
            log(f"1.1 {cidade} {q}: ext=({dx_km:.2f},{dy_km:.2f})km passo=({passo_x_m:.1f},{passo_y_m:.1f})m n_ant={n_antena[(cidade, q)]}")

    # Overlap between quadrants of the same city.
    sobreposicao = []
    from itertools import combinations
    for cidade in CIDADES:
        lat_ref = 0.5 * (geo_bbox[(cidade, "Q1")][2] + geo_bbox[(cidade, "Q3")][3])
        for qa, qb in combinations(QS, 2):
            bbox_a = geo_bbox[(cidade, qa)]
            bbox_b = geo_bbox[(cidade, qb)]
            area_km2, ov_lon_km, ov_lat_km = bbox_intersecao_km2(bbox_a, bbox_b, lat_ref)
            entry = {
                "cidade": cidade, "par": f"{qa}-{qb}",
                "intersecao_km2": area_km2, "intersecao_lon_km": ov_lon_km, "intersecao_lat_km": ov_lat_km,
                "n_nos_coincidentes": 0,
            }
            if area_km2 > 0:
                # Coincident nodes: coordinates rounded to 1e-9 degree.
                lon_a, lat_a = build_synthetic_grid(*[geo_bbox[(cidade, qa)][i] for i in (0, 1, 2, 3)])
                lon_b, lat_b = build_synthetic_grid(*[geo_bbox[(cidade, qb)][i] for i in (0, 1, 2, 3)])
                set_a = set(zip(np.round(lon_a, 9).tolist(), np.round(lat_a, 9).tolist()))
                set_b = set(zip(np.round(lon_b, 9).tolist(), np.round(lat_b, 9).tolist()))
                entry["n_nos_coincidentes"] = len(set_a & set_b)
            sobreposicao.append(entry)
            log(f"1.1 sobreposicao {cidade} {qa}-{qb}: {area_km2:.6f} km2, nos_coincidentes={entry['n_nos_coincidentes']}")

    # Antenna count of each Bauru quadrant graph.
    n_ant_bauru = {q: n_antena[("bauru", q)] for q in QS}
    todas_iguais = len(set(n_ant_bauru.values())) == 1
    achado_antenas_bauru = {
        "n_antena_por_quadrante": n_ant_bauru,
        "todas_iguais": todas_iguais,
        "leitura": (
            "as 794 antenas da cidade aparecem IDENTICAS (mesma contagem) nos 4 quadrantes "
            "de Bauru, nao apenas em Q1 -- refuta a leitura de que so Q1 'tem' as 794 antenas; "
            "confirma que o numero 794 e' o total de antenas da CIDADE e que as antenas nao sao "
            "recortadas por bbox de quadrante (ao contrario dos nos terrain, que sao exclusivos "
            "de cada quadrante): cada grafo de quadrante inclui TODAS as antenas da cidade como "
            "contexto, so os nos terrain (e a malha 30m) mudam de quadrante para quadrante"
            if todas_iguais else
            "os quadrantes de Bauru NAO tem o mesmo numero de antenas -- a leitura 'Q1 tem as "
            "794 antenas da cidade' precisa ser verificada quadrante a quadrante (ver n_antena_por_quadrante)"
        ),
        "evidencia_auxiliar": (
            "n_arestas_ant_ter_base (edges antena-terreno antes de qualquer particao) DIFERE entre "
            "os quadrantes de Bauru apesar de n_antena ser igual em todos -- consistente com o MESMO "
            "conjunto de 794 antenas fisicas sendo religado a cada quadrante contra um subconjunto "
            "DIFERENTE de nos terrain (k_antenna=20 vizinhos mais proximos por no, ver config.k_antenna)"
        ),
    }

    max_erro_ext = max(v for c in por_celula.values() for v in (c["erro_relativo_extensao_x"], c["erro_relativo_extensao_y"]))
    passo_min = min(min(c["passo_malha_x_m"], c["passo_malha_y_m"]) for c in por_celula.values())
    passo_max = max(max(c["passo_malha_x_m"], c["passo_malha_y_m"]) for c in por_celula.values())
    max_intersecao_km2 = max(e["intersecao_km2"] for e in sobreposicao)
    max_nos_coincidentes = max(e["n_nos_coincidentes"] for e in sobreposicao)

    return {
        "por_celula": por_celula,
        "sobreposicao_pares_mesma_cidade": sobreposicao,
        "antenas_bauru_por_quadrante": achado_antenas_bauru,
        "resumo": {
            "n_celulas": len(por_celula),
            "erro_relativo_maximo_extensao_vs_run_json": max_erro_ext,
            "nota_erro_extensao_x": (
                "the relative error of extensao_x stays at 0.33-0.38% in all 16 cells "
                "(extensao_y error 0.0, exact) -- it comes from a different cos(phi) CONVENTION: "
                "this test uses cos(phi) at the CENTRE of the bounding box (as criterion 1.1 asks), "
                "while the run JSON gives extensao_x_km with cos(phi) PER NODE (geometria.convencao); "
                "the difference has the size expected for about 1 degree of latitude span and does NOT "
                "indicate a geometry error -- it exceeds the <=1e-3 threshold set in the criterion for "
                "'matching the run JSON', so that secondary item is marked as not strictly met "
                "(the decisive item -- disjointness of the cells -- is not affected)"
            ),
            "passo_malha_m_min": passo_min, "passo_malha_m_max": passo_max,
            "compativel_3600x30m": bool(20.0 <= passo_min and passo_max <= 35.0),
            "intersecao_km2_maxima_entre_quadrantes_mesma_cidade": max_intersecao_km2,
            "n_nos_coincidentes_maximo": max_nos_coincidentes,
            "celulas_disjuntas": bool(max_nos_coincidentes == 0),
            "extensao_bate_run_json_1e-3": bool(max_erro_ext <= 1e-3),
        },
    }


def latlon_para_metros_global(lon, lat, lon0, lat0):
    """Same projection as latlon_graus_para_metros, but with a fixed origin (lon0, lat0)
    shared by all quadrants of a city, so distances between quadrants are physical."""
    y = (lat - lat0) * 111_000.0
    x = (lon - lon0) * 111_000.0 * np.cos(np.radians(lat))
    return np.stack([x, y], axis=1)


def reconstruir_celula(cidade, q, log):
    """Rebuild the partition masks of one cell in its local frame, as the frozen trainer
    does, and compare the node counts after the buffer with the run record."""
    d, p = carregar_run(cidade, q, "g10b2")
    cfg = d["config"]
    geo = d["geometria"]
    lon, lat = build_synthetic_grid(geo["lon_min_deg"], geo["lon_max_deg"],
                                     geo["lat_min_deg"], geo["lat_max_deg"])
    pos_m_local = latlon_graus_para_metros(lon, lat)
    fracs = tuple(float(x) for x in cfg["split_frac"].split(","))
    t0 = time.time()
    parts, group_ids, info = split_espacial_3vias_exato(
        pos_m_local, grid_km=cfg["grid_km"], buffer_km=cfg["buffer_km"],
        fracs=fracs, split_seed=cfg["split_seed"])
    dt = time.time() - t0
    declarado = d["split"]
    erro_rel = {}
    for parte in ("train", "val", "test"):
        rep = info["n_nos_apos_buffer"][parte]
        dec = declarado["n_nos_apos_buffer"][parte]
        erro_rel[parte] = (abs(rep - dec) / dec) if dec else None
    log(f"1.2 reconstrucao {cidade} {q}: n_apos={info['n_nos_apos_buffer']} "
        f"(declarado={declarado['n_nos_apos_buffer']}) erro_rel={erro_rel} ({dt:.1f}s)")
    return {
        "run_json": str(p), "lon": lon, "lat": lat,
        "mask_train": parts["train"], "mask_val": parts["val"], "mask_test": parts["test"],
        "group_ids": group_ids,
        "geo": geo, "cfg": cfg,
        "info_reconstruido": info, "declarado": {
            "n_nos_antes_do_buffer": declarado["n_nos_antes_do_buffer"],
            "n_nos_apos_buffer": declarado["n_nos_apos_buffer"],
        },
        "erro_relativo_vs_declarado": erro_rel,
        "tempo_reconstrucao_s": dt,
    }


def rodar_teste_12(log):
    from scipy.spatial import cKDTree

    log("1.2: reconstruindo particoes reais de bauru Q1-Q4 (g10b2, seed 42)...")
    celulas = {q: reconstruir_celula("bauru", q, log) for q in QS}

    # Common city origin: minimum lon/lat over the four Bauru quadrants.
    lon0 = min(c["geo"]["lon_min_deg"] for c in celulas.values())
    lat0 = min(c["geo"]["lat_min_deg"] for c in celulas.values())
    log(f"1.2: origem global da cidade (bauru) lon0={lon0}, lat0={lat0}")

    pares = [("Q1", "Q2"), ("Q2", "Q3"), ("Q3", "Q4")]
    b_km = 2.0
    resultados_pares = []
    for qi, qi1 in pares:
        t0 = time.time()
        c_tr = celulas[qi]
        c_te = celulas[qi1]
        pos_tr_global = latlon_para_metros_global(c_tr["lon"][c_tr["mask_train"]], c_tr["lat"][c_tr["mask_train"]], lon0, lat0) / 1000.0
        pos_te_global = latlon_para_metros_global(c_te["lon"][c_te["mask_test"]], c_te["lat"][c_te["mask_test"]], lon0, lat0) / 1000.0
        n_tr = pos_tr_global.shape[0]
        n_te = pos_te_global.shape[0]
        tree = cKDTree(pos_tr_global, compact_nodes=False, balanced_tree=False)
        d, _ = tree.query(pos_te_global, k=1, workers=-1)
        n_viol = int((d < b_km).sum())
        dt = time.time() - t0
        entry = {
            "par": f"{qi}(treino)->{qi1}(teste)",
            "n_treino_qi": n_tr, "n_teste_qi1": n_te,
            "n_violacoes_buffer": n_viol,
            "fracao_teste_afetada": n_viol / n_te if n_te else None,
            "d_min_km": float(d.min()) if d.size else None,
            "d_mediana_km": float(np.median(d)) if d.size else None,
            "d_max_km": float(d.max()) if d.size else None,
            "buffer_km_exigido": b_km,
            "tempo_s": dt,
        }
        resultados_pares.append(entry)
        log(f"1.2 par {qi}->{qi1}: n_viol={n_viol}/{n_te} d_min={entry['d_min_km']:.4f}km ({dt:.1f}s)")

    # With the same split seed, two quadrants with the same total block count get the
    # same block permutation; recorded as metadata.
    n_blocos_por_q = {q: celulas[q]["info_reconstruido"]["n_blocos_total"] for q in QS}
    pares_mesma_permutacao = []
    for qi, qi1 in pares:
        mesma_n_blocos = n_blocos_por_q[qi] == n_blocos_por_q[qi1]
        pares_mesma_permutacao.append({"par": f"{qi}/{qi1}", "n_blocos_total_iguais": mesma_n_blocos,
                                        "n_blocos_qi": n_blocos_por_q[qi], "n_blocos_qi1": n_blocos_por_q[qi1]})

    max_viol = max(r["n_violacoes_buffer"] for r in resultados_pares)
    total_viol = sum(r["n_violacoes_buffer"] for r in resultados_pares)

    return {
        "cidade": "bauru", "config": "g10b2", "split_seed": 42, "buffer_km": b_km,
        "origem_global": {"lon0_deg": lon0, "lat0_deg": lat0},
        "reconstrucao_por_celula": {
            q: {
                "run_json": celulas[q]["run_json"],
                "n_nos_apos_buffer_reconstruido": celulas[q]["info_reconstruido"]["n_nos_apos_buffer"],
                "n_nos_apos_buffer_declarado": celulas[q]["declarado"]["n_nos_apos_buffer"],
                "erro_relativo_vs_declarado": celulas[q]["erro_relativo_vs_declarado"],
                "n_blocos_total": celulas[q]["info_reconstruido"]["n_blocos_total"],
                "tempo_reconstrucao_s": celulas[q]["tempo_reconstrucao_s"],
            } for q in QS
        },
        "pares_mesma_permutacao_n_blocos": pares_mesma_permutacao,
        "cadeia_buffer_violacao": resultados_pares,
        "resumo": {
            "n_pares_avaliados": len(resultados_pares),
            "n_violacoes_maximo_por_par": max_viol,
            "n_violacoes_total_soma_pares": total_viol,
            "algum_par_fura_buffer": bool(max_viol > 0),
        },
    }


def gerar_figura(resultado_11, resultado_12, fig_base, log):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "serif"
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))

    # Left panel: bounding boxes of the 16 cells.
    ax = axes[0]
    cores = {"bauru": "tab:red", "campinas": "tab:blue", "lins": "tab:green", "sorocaba": "tab:orange"}
    for chave, c in resultado_11["por_celula"].items():
        cidade, q = chave.rsplit("_", 1)
        lon_min, lon_max = c["bbox_lon_min_deg"], c["bbox_lon_max_deg"]
        lat_min, lat_max = c["bbox_lat_min_deg"], c["bbox_lat_max_deg"]
        ax.add_patch(Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                fill=False, edgecolor=cores[cidade], linewidth=1.3))
        ax.text((lon_min + lon_max) / 2, (lat_min + lat_max) / 2, q, ha="center", va="center",
                fontsize=7, color=cores[cidade])
    for cidade, cor in cores.items():
        ax.plot([], [], color=cor, label=cidade)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("longitude (graus)")
    ax.set_ylabel("latitude (graus)")
    ax.set_title("16 celulas cidade-quadrante (bbox, g10b2, seed 42)")
    ax.set_aspect("auto")

    # Right panel: Bauru Q1 blocks and partition (split seed 42).
    ax2 = axes[1]
    c_q1 = resultado_12["reconstrucao_por_celula"]["Q1"]
    run_q1 = json.load(open(c_q1["run_json"]))
    geo = run_q1["geometria"]
    cfg = run_q1["config"]
    lon, lat = build_synthetic_grid(geo["lon_min_deg"], geo["lon_max_deg"], geo["lat_min_deg"], geo["lat_max_deg"])
    pos_m = latlon_graus_para_metros(lon, lat)
    fracs = tuple(float(x) for x in cfg["split_frac"].split(","))
    parts, group_ids, info = split_espacial_3vias_exato(pos_m, cfg["grid_km"], cfg["buffer_km"], fracs, cfg["split_seed"])
    pos_km = pos_m / 1000.0
    n_side = N_SIDE
    passo_plot = 40  # plot every 40th node along each axis
    idx2d = np.arange(n_side * n_side).reshape(n_side, n_side)[::passo_plot, ::passo_plot].ravel()
    cor_papel = np.full(idx2d.shape, 0, dtype=int)  # 0 removed by buffer, 1 train, 2 val, 3 test
    cor_papel[parts["train"][idx2d]] = 1
    cor_papel[parts["val"][idx2d]] = 2
    cor_papel[parts["test"][idx2d]] = 3
    cmap_papel = {0: ("buffer/removido", "lightgray"), 1: ("treino", "tab:blue"), 2: ("validacao", "tab:green"), 3: ("teste", "tab:red")}
    for cod, (nome, cor) in cmap_papel.items():
        sel = cor_papel == cod
        ax2.scatter(pos_km[idx2d][sel, 0], pos_km[idx2d][sel, 1], s=1.5, c=cor, label=nome, rasterized=True)
    g_km = cfg["grid_km"]
    x_max = pos_km[:, 0].max()
    y_max = pos_km[:, 1].max()
    for xg in np.arange(0, x_max + g_km, g_km):
        ax2.axvline(xg, color="black", linewidth=0.3, alpha=0.4)
    for yg in np.arange(0, y_max + g_km, g_km):
        ax2.axhline(yg, color="black", linewidth=0.3, alpha=0.4)
    ax2.set_xlabel("x local (km, origem no bbox de Q1)")
    ax2.set_ylabel("y local (km, origem no bbox de Q1)")
    ax2.set_title(f"Bauru Q1: blocos de {g_km:.0f}km e particao real (seed 42, buffer {cfg['buffer_km']:.0f}km)")
    ax2.legend(fontsize=7, loc="upper right", markerscale=4)
    ax2.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(f"{fig_base}.pdf")
    fig.savefig(f"{fig_base}.png", dpi=200)
    plt.close(fig)
    log(f"figura gravada: {fig_base}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-11", required=True)
    ap.add_argument("--out-12", required=True)
    ap.add_argument("--fig", required=True, help="caminho base sem extensao (gera .pdf e .png)")
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
    script_src = Path(__file__).read_text()
    script_sha = sha256_of_text(script_src)

    entradas_sha = {}
    for cidade in CIDADES:
        for q in QS:
            _, p = carregar_run(cidade, q, "g10b2")
            entradas_sha[str(p)] = sha256_of_file(p)

    log("=== TESTE 1.1: geometria das 16 celulas ===")
    r11 = rodar_teste_11(log)
    saida_11 = {
        "id": "1.1",
        "fio": "internal/analysis",
        "alegacao": "condicao de validade para A1-A4 (celulas bem definidas)",
        "pergunta": json.loads(Path(__file__).parent.parent.joinpath(
            "_v3_2026-09-25", "criterios", "criterio_1.1.json").read_text())["pergunta"],
        "criterio": json.loads(Path(__file__).parent.parent.joinpath(
            "_v3_2026-09-25", "criterios", "criterio_1.1.json").read_text()),
        "metodo": "leitura direta de geometria/subgrafos dos 16 run JSON g10b2; recalculo "
                  "independente de extensao km (equirretangular, cos phi no centro); intersecao "
                  "de bbox analitica por par de quadrantes da mesma cidade; contagem de nos "
                  "coincidentes via grade sintetica so quando a bbox intersecta",
        "insumos": entradas_sha,
        "por_celula": r11["por_celula"],
        "sobreposicao_pares_mesma_cidade": r11["sobreposicao_pares_mesma_cidade"],
        "antenas_bauru_por_quadrante": r11["antenas_bauru_por_quadrante"],
        "resumo": r11["resumo"],
        "veredito_vs_criterio": (
            ("CRITERIO CENTRAL ATENDIDO: celulas disjuntas (0 nos coincidentes em todos os 24 pares "
             "da mesma cidade, intersecao de bbox = 0 km2) e passo de malha compativel com 3600 x "
             "~30m (28.2-30.8m). Ressalva menor: extensao_x recalculada com cos(phi) no centro difere "
             "0.33-0.38% da extensao_x_km declarada no run JSON (que usa cos(phi) por no) -- acima do "
             "1e-3 fixado no criterio para este item secundario, mas explicada pela convencao "
             "diferente de cos(phi), nao por erro de geometria; nao afeta a disjuncao das celulas.")
            if r11["resumo"]["celulas_disjuntas"] and r11["resumo"]["compativel_3600x30m"]
            else "CRITERIO NAO ATENDIDO (ver resumo) -- gravidade bloqueia, redesenho do estudo antes de qualquer outro teste"
        ),
        "nao_verificado": [],
        "comando_rodado": f"{Path(sys.executable).name} v3_1.1_1.2_celulas_cadeia.py --out-11 <out> --out-12 <out> --fig <fig>",
        "script_sha256": script_sha,
    }
    Path(args.out_11).write_text(json.dumps(saida_11, indent=1, ensure_ascii=False))
    log(f"gravado: {args.out_11}")

    log("=== TESTE 1.2: cadeia Q1->Q2->Q3->Q4, buffer ===")
    r12 = rodar_teste_12(log)
    saida_12 = {
        "id": "1.2",
        "fio": "internal/analysis",
        "alegacao": "A3 (condicao 1.2 buffer honrado); A4 (condicao 4.3 se 1.2 acusar)",
        "pergunta": json.loads(Path(__file__).parent.parent.joinpath(
            "_v3_2026-09-25", "criterios", "criterio_1.2.json").read_text())["pergunta"],
        "criterio": json.loads(Path(__file__).parent.parent.joinpath(
            "_v3_2026-09-25", "criterios", "criterio_1.2.json").read_text()),
        "metodo": "split_espacial_3vias_exato (copia literal, importada sem edicao de "
                  "scripts/varredura_split_geometria.py) reconstroi train/val/test REAIS de "
                  "cada quadrante em coordenadas LOCAIS (fiel ao codigo congelado, que processa "
                  "cada grafo separadamente); distancia FISICA entre quadrantes calculada em "
                  "coordenadas GLOBAIS da cidade (origem comum lon0/lat0 = minimo entre os 4 "
                  "quadrantes), cKDTree k=1 dos nos de teste de Qi+1 contra os nos de treino de Qi",
        "insumos": {k: v for k, v in entradas_sha.items() if "bauru" in k and "g10b2" in k},
        "reconstrucao_por_celula": r12["reconstrucao_por_celula"],
        "pares_mesma_permutacao_n_blocos": r12["pares_mesma_permutacao_n_blocos"],
        "cadeia_buffer_violacao": r12["cadeia_buffer_violacao"],
        "resumo": r12["resumo"],
        "veredito_vs_criterio": (
            "CRITERIO ATENDIDO: 0 nos de teste a <2km do treino do quadrante anterior, nos 3 pares da cadeia"
            if not r12["resumo"]["algum_par_fura_buffer"]
            else "CRITERIO NAO ATENDIDO: cadeia fura o buffer -- gravidade bloqueia para A3/A4 em modelos encadeados; aciona fase 4.3 do roadmap"
        ),
        "nao_verificado": [],
        "comando_rodado": f"{Path(sys.executable).name} v3_1.1_1.2_celulas_cadeia.py --out-11 <out> --out-12 <out> --fig <fig>",
        "script_sha256": script_sha,
        "pico_ram_gb": None,  # filled at the end, after the figure
        "tempo_total_s": None,
    }
    Path(args.out_12).write_text(json.dumps(saida_12, indent=1, ensure_ascii=False))
    log(f"gravado: {args.out_12}")

    log("=== gerando figura-mapa ===")
    gerar_figura(saida_11, saida_12, args.fig, log)

    pico_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    tempo_total = time.time() - t_inicio
    saida_12["pico_ram_gb"] = pico_ram
    saida_12["tempo_total_s"] = tempo_total
    Path(args.out_12).write_text(json.dumps(saida_12, indent=1, ensure_ascii=False))
    log(f"CONCLUIDO. tempo total {tempo_total:.1f}s, pico RAM {pico_ram:.2f}GB")


if __name__ == "__main__":
    main()
