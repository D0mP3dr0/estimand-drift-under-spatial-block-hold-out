# -*- coding: utf-8 -*-
"""
Estimand Drift under Spatial Block Hold-Out (MDPI Mathematics, revision 1).
V3: descriptive table of the 16 cells (criteria/criterio_V3_tabela_celulas.json, fixed 2026-10-26T09:35:30-03:00).
Reuses by import analysis/v3_13_laco_comum.py::carregar_celula (digest of the file as read is recorded). Fields the loop
does not expose (n_transmissores, elevation) are read from the same .pt (laco.orig.carregar_tensor).
No number typed by hand; no interpretation. CPU only.

REQUIRES the 16 graph tensors transfer_dataset_<city>_v19_<Q>_enriched_cftudo.pt (about 28 GB each), which are NOT in this
repository (available on request; names, sizes and digests in manifest/data_manifest.csv), and the path set-up of
analysis/v3_13_laco_comum.py and analysis/v3_2.1_3.1_deriva_calibracao.py described in README.md.

Modes: --inspecionar (bauru_Q1 -> V3_inspecao_tensor.json) | --rodar (16 cells, <= 3 processes).
Output directory: env R1_OUT (default results/fase6_rerun).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
SCRIPT_SHA256 = hashlib.sha256(HERE.read_bytes()).hexdigest()
ROOT = HERE.parents[2]  # repository root
SCRIPT_REL = HERE.relative_to(ROOT).as_posix()
LACO_PATH = ROOT / "analysis" / "v3_13_laco_comum.py"
LACO_SHA256 = hashlib.sha256(LACO_PATH.read_bytes()).hexdigest()  # digest of the file as read (recorded, not asserted)
CRITERIO = ROOT / "criteria" / "criterio_V3_tabela_celulas.json"
BLOCOS_JSON = ROOT / "results" / "fase5" / "blocos_com_no_valido.json"
F6 = Path(os.environ.get("R1_OUT", ROOT / "results" / "fase6_rerun"))
OUT_INSP = F6 / "V3_inspecao_tensor.json"
OUT_JSON = F6 / "V3_tabela_celulas.json"
OUT_TEX = F6 / "T_V3_celulas.tex"
CELULAS = [f"{c}_{q}" for c in ("bauru", "campinas", "lins", "sorocaba") for q in ("Q1", "Q2", "Q3", "Q4")]

_spec = importlib.util.spec_from_file_location("v3_13_laco_comum", LACO_PATH)
laco = importlib.util.module_from_spec(_spec)
sys.modules["v3_13_laco_comum"] = laco
_spec.loader.exec_module(laco)
orig = laco.orig


def log(m):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {m}", flush=True)


def inspecionar_tensor(tensor_path) -> dict:
    rf, modo = orig.carregar_tensor(tensor_path)
    info = {"tensor_path": str(tensor_path), "modo_carga": modo, "classe": type(rf).__name__,
            "node_types": list(rf.node_types), "edge_types": [list(e) for e in rf.edge_types], "nos": {},
            "arestas": {}, "global_store_chaves": list(rf._global_store.keys())}
    for nt in rf.node_types:
        d = {}
        for k, v in rf[nt].items():
            if hasattr(v, "shape"):
                d[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
            else:
                d[k] = {"tipo": type(v).__name__, "valor": v if isinstance(v, (int, float, str)) else None}
        info["nos"][nt] = d
    for et in rf.edge_types:
        info["arestas"]["__".join(et)] = {k: list(v.shape) for k, v in rf[et].items() if hasattr(v, "shape")}
    nomes_possiveis = [k for nt in rf.node_types for k in rf[nt].keys()
                       if any(s in k.lower() for s in ("name", "feature_names", "cols", "columns", "attr_names"))]
    info["metadado_de_nomes_de_atributos_encontrado"] = nomes_possiveis
    info["coluna_elevacao"] = ("sem nome no tensor" if not nomes_possiveis else "ver metadado_de_nomes")
    ant = rf["antenna"]
    info["tipo_no_transmissor"] = "antenna"
    info["n_transmissores_x_shape0"] = int(ant.x.shape[0])
    info["n_transmissores_num_nodes"] = int(ant.num_nodes)
    info["n_transmissores_pos_shape0"] = int(ant.pos.shape[0])
    del rf
    return info


def inspecao_main():
    F6.mkdir(parents=True, exist_ok=True)
    p = orig.TENSOR_DIR / "transfer_dataset_bauru_v19_Q1_enriched_cftudo.pt"
    info = inspecionar_tensor(p)
    info.update({"celula": "bauru_Q1", "script": SCRIPT_REL, "script_sha256": SCRIPT_SHA256,
                 "data_utc": datetime.now(timezone.utc).isoformat(), "venv": "python"})
    OUT_INSP.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"inspecao -> {OUT_INSP}")


def por_celula(chave: str) -> dict:
    t0 = time.perf_counter()
    c = laco.carregar_celula(chave)
    rssi, dist, sent, gid = c["rssi"], c["dist"], c["sent"], c["gid"]
    val = ~sent
    rf, _ = orig.carregar_tensor(Path(c["tensor_path"]))
    ant = rf["antenna"]
    n_tx = int(ant.x.shape[0])
    n_tx_nn = int(ant.num_nodes)
    del rf
    rv = rssi[val]
    return {"celula": chave, "cidade": chave.split("_")[0], "quadrante": chave.split("_")[1],
            "n_nos": int(rssi.size), "n_validos": int(val.sum()), "n_sentinela": int(sent.sum()),
            "fracao_sentinela": float(sent.mean()),
            "n_blocos_10km_ocupados": int(c["N_blocos_ocupados"]),
            "n_blocos_com_no_valido": int(np.unique(gid[val]).size),
            "n_transmissores": n_tx, "n_transmissores_num_nodes": n_tx_nn,
            "dist_mediana_validos_km": float(np.median(dist[val].astype(np.float64)) / 1000.0),
            "rssi_mediana_validos_dBm": float(np.median(rv)), "rssi_min_validos_dBm": float(rv.min()),
            "elevacao_min_m": None, "elevacao_mediana_m": None, "elevacao_max_m": None,
            "elevacao_motivo_vazia": "sem nome no tensor (terrain.x 512 col e terrain.features_raw 17 col sem metadado de nomes)",
            "sha256_manifest": c["sha256_manifest"], "tensor_path": c["tensor_path"],
            "campos_usados": {"n_nos": "len(terrain.y[:,3])",
                              "sentinela": "terrain.y[:,0] >= orig.PL_TARGET_MAX_VALID (laco.sent)",
                              "rssi": "terrain.y[:,3]", "dist": "terrain.dist_nearest_m / 1000",
                              "blocos": "laco gid (SpatialKFold._assign_groups, 10 km)",
                              "n_transmissores": "antenna.x.shape[0] (= antenna.num_nodes)", "elevacao": None},
            "tempo_s": time.perf_counter() - t0}


def fmt_m(x):
    return f"{x / 1e6:,.2f}"


def tex(por):
    cab = ["Celula", "$N$ (M)", "Validos (M)", "Sentinela (M)", "Frac. sent.", "Blocos ocup.", "Blocos c/ valido",
           "Transm.", "Dist. med. (km)", "RSSI med.", "RSSI min."]
    L = ["% generated by v3_15_V3_tabela_celulas.py; elevation columns omitted: no column name in the tensor (see V3_tabela_celulas.json)",
         "\\begin{tabular}{l" + "r" * (len(cab) - 1) + "}", "\\hline", " & ".join(cab) + " \\\\", "\\hline"]
    for c in CELULAS:
        r = por[c]
        L.append(" & ".join([c.replace("_", "\\_"), fmt_m(r["n_nos"]), fmt_m(r["n_validos"]), fmt_m(r["n_sentinela"]),
                             f"{r['fracao_sentinela']:.3f}", str(r["n_blocos_10km_ocupados"]),
                             str(r["n_blocos_com_no_valido"]), str(r["n_transmissores"]),
                             f"{r['dist_mediana_validos_km']:.2f}", f"{r['rssi_mediana_validos_dBm']:.1f}",
                             f"{r['rssi_min_validos_dBm']:.1f}"]) + " \\\\")
    L += ["\\hline", "\\end{tabular}", ""]
    return "\n".join(L)


def rodar_main():
    t0 = time.perf_counter()
    F6.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(por_celula, CELULAS))
    por = {r["celula"]: r for r in res}
    ref = json.loads(BLOCOS_JSON.read_text(encoding="utf-8"))["por_celula"]
    conf, diverg = {}, []
    for c in CELULAS:
        rr, p = ref[c], por[c]
        d = {"n_validos": p["n_validos"] - rr["n_nos_validos"], "n_sentinela": p["n_sentinela"] - rr["n_nos_sentinela"],
             "n_nos": p["n_nos"] - rr["n_nos"],
             "n_blocos_com_no_valido": p["n_blocos_com_no_valido"] - rr["blocos_com_ao_menos_1_no_valido"],
             "n_blocos_ocupados": p["n_blocos_10km_ocupados"] - rr["N_blocos_ocupados"],
             "sha256_manifest_igual": p["sha256_manifest"] == rr["sha256_manifest_tensor"],
             "n_transmissores_x_vs_num_nodes": p["n_transmissores"] - p["n_transmissores_num_nodes"]}
        conf[c] = d
        if any(v != 0 for k, v in d.items() if k != "sha256_manifest_igual") or not d["sha256_manifest_igual"]:
            diverg.append(c)
    f = [por[c]["fracao_sentinela"] for c in CELULAS]
    tx = {}
    for cid in ("bauru", "campinas", "lins", "sorocaba"):
        v = [por[c]["n_transmissores"] for c in CELULAS if c.startswith(cid)]
        tx[cid] = {"min": min(v), "max": max(v), "por_quadrante": v}
    saida = {"id": "V3_tabela_celulas", "criterio": CRITERIO.relative_to(ROOT).as_posix(),
             "criterio_sha256": hashlib.sha256(CRITERIO.read_bytes()).hexdigest(),
             "conferencia_vs_fase5_blocos_com_no_valido": {
                 "fonte": BLOCOS_JSON.relative_to(ROOT).as_posix(), "fonte_sha256": hashlib.sha256(BLOCOS_JSON.read_bytes()).hexdigest(),
                 "diferencas_por_celula": conf, "celulas_divergentes": diverg, "bateu": not diverg},
             "faixa_fracao_sentinela": {"min": min(f), "max": max(f)},
             "faixa_n_validos": {"min": min(por[c]["n_validos"] for c in CELULAS),
                                 "max": max(por[c]["n_validos"] for c in CELULAS)},
             "faixa_n_transmissores_por_cidade": tx,
             "colunas_vazias": {"elevacao_min_m/mediana_m/max_m": "sem nome no tensor"},
             "por_celula": por, "script": SCRIPT_REL, "script_sha256": SCRIPT_SHA256, "laco_sha256": LACO_SHA256,
             "script_original_sha256": laco.ORIG_SHA256, "comando": " ".join(["python", SCRIPT_REL] + sys.argv[1:]), "venv": "python",
             "data_utc": datetime.now(timezone.utc).isoformat(), "sementes": "nao se aplica",
             "tempo_total_s": time.perf_counter() - t0}
    OUT_JSON.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_TEX.write_text(tex(por), encoding="utf-8")
    log(f"conferencia bateu={not diverg} divergentes={diverg} -> {OUT_JSON}")
    sys.exit(0 if not diverg else 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inspecionar", action="store_true")
    g.add_argument("--rodar", action="store_true")
    a = ap.parse_args()
    inspecao_main() if a.inspecionar else rodar_main()
