"""Monte Carlo of the area retention of validation and test blocks under the buffer.

For each city (Bauru, Campinas, Lins, Sorocaba) and each design (g = 10 km and
g = 5 km, buffer b = 2 km), the tile extent is cut into a block lattice whose last
row and column carry the remainder (int(pos_km / g) convention), blocks are
counted per role with the trainer's rule (train = round(0.70 n), val =
round(0.15 n), test = the rest), and roles are shuffled N_TRIALS_MAIN times.
Each trial trims every validation block by b along sides and isolated corners
shared with training blocks, then every test block against training blocks and
validation blocks, and records the retained area fraction of each role.
Five variants separate the geometric effects on test retention:
  A  uniform g x g blocks, test trimmed by all validation blocks, open domain;
  A2 uniform blocks, test trimmed only by validation blocks that kept area, open;
  B  real border widths, trimmed by all validation blocks, open;
  C  real border widths, trimmed only by retained validation blocks, open
     (the rule of the code);
  T  as A2 on a periodic (torus) lattice, without perimeter.
T -> A2 isolates the perimeter, A2 -> C the thin border blocks, A -> B the border
widths, and B -> C the asymmetric trimming rule. The script also evaluates the
mean-field closed form R_meanfield, checks it symbolically with SymPy, compares
variant C with the retention observed at split seed 42 where a run record exists,
and estimates the discretisation error of the continuous area against a 25 m raster.

Seeds: one seed per (design, city, variant), derived from SHA-256 of the labels
(seed_de_string), so results are identical across processes.
Output: one JSON record written to the absolute path hard-coded in main().
Usage: python montecarlo_retencao_r3_g5.py   (--hash prints the script SHA-256 and exits)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

SEED = 20260925
N_TRIALS_MAIN = 2500
RASTER_M = 25.0  # raster pixel (m) used only to check the continuous area formula

# Metric extent (km) of each city tile in x (east) and y (north).
CIDADES_EXT = {
    "bauru": {"ext_x_km": 103.37162264246432, "ext_y_km": 110.96908950805664},
    "campinas": {"ext_x_km": 102.95723100995667, "ext_y_km": 110.96908950805664},
    "lins": {"ext_x_km": 103.81934358302617, "ext_y_km": 110.96908950805664},
    "sorocaba": {"ext_x_km": 102.5211445573578, "ext_y_km": 110.96908950805664},
}

# Node retention after the buffer observed in the split-seed-42 run records
# (test and validation); only designs with a run record are listed.
OBS_TEST = {
    ("bauru", "g10b2"): 0.4632420674893326,
    ("campinas", "g10b2"): 0.46014706310804726,
    ("lins", "g10b2"): 0.46549287319735866,
    ("sorocaba", "g10b2"): 0.45963125105236574,
    ("bauru", "g5b2"): 0.1047868809738556,
}
OBS_VAL = {
    ("bauru", "g10b2"): 0.5210508853685026,
    ("campinas", "g10b2"): 0.5201393265848172,
    ("lins", "g10b2"): 0.5209684329222332,
    ("sorocaba", "g10b2"): 0.5201061447264552,
    ("bauru", "g5b2"): 0.20190782023022533,
}


def seed_de_string(*parts, base=SEED):
    """Process-independent seed from the labels (built-in hash() is salted per process)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return (base + int(h[:8], 16)) % (2**31 - 1)


def grid_dims(ext_x, ext_y, g):
    """Block lattice of an extent: full blocks of side g plus one remainder column/row."""
    nx_full = int(np.floor(ext_x / g))
    resto_x = ext_x - nx_full * g
    nx = nx_full + (1 if resto_x > 1e-9 else 0)
    colw = [g] * nx_full + ([resto_x] if resto_x > 1e-9 else [])

    ny_full = int(np.floor(ext_y / g))
    resto_y = ext_y - ny_full * g
    ny = ny_full + (1 if resto_y > 1e-9 else 0)
    rowh = [g] * ny_full + ([resto_y] if resto_y > 1e-9 else [])
    return nx, ny, np.array(colw), np.array(rowh)


def n_blocos_por_papel(n_g):
    """Block count per role as in the trainer: rounded 0.70 / 0.15 fractions, test gets the rest."""
    n_tr = max(1, int(round(0.70 * n_g)))
    n_va = max(1, int(round(0.15 * n_g)))
    if n_tr + n_va >= n_g:
        n_tr = max(1, n_g - 2)
        n_va = 1
    n_te = n_g - n_tr - n_va
    return n_tr, n_va, n_te


def area_bloco_continua(wx, wy, b, L, R, Bo, T, BL, BR, TL, TR):
    """Area of a wx x wy block left after a buffer b: a strip of width b is removed on
    each trimming side (L, R, Bo, T), and a quarter disc of radius b at each trimming
    diagonal neighbour (BL, BR, TL, TR) whose two adjacent sides do not trim."""
    rx = max(0.0, wx - b * (L + R))
    ry = max(0.0, wy - b * (Bo + T))
    rect = rx * ry
    corner_area = 0.0
    for corner, (lat1, lat2) in (
        (BL, (L, Bo)), (BR, (R, Bo)), (TL, (L, T)), (TR, (R, T))
    ):
        if corner and not lat1 and not lat2:
            corner_area += np.pi * b * b / 4.0
    area = rect - corner_area
    return float(np.clip(area, 0.0, wx * wy))


def area_bloco_raster(wx, wy, b, L, R, Bo, T, BL, BR, TL, TR, raster_m):
    """Same area as area_bloco_continua, counted on pixel centres of side raster_m (m)."""
    px = raster_m / 1000.0
    nxp = max(1, int(round(wx / px)))
    nyp = max(1, int(round(wy / px)))
    xs = (np.arange(nxp) + 0.5) * (wx / nxp)
    ys = (np.arange(nyp) + 0.5) * (wy / nyp)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    keep = np.ones_like(X, dtype=bool)
    if L:
        keep &= X >= b
    if R:
        keep &= X <= wx - b
    if Bo:
        keep &= Y >= b
    if T:
        keep &= Y <= wy - b
    corners = [
        (BL, not L and not Bo, 0.0, 0.0),
        (BR, not R and not Bo, wx, 0.0),
        (TL, not L and not T, 0.0, wy),
        (TR, not R and not T, wx, wy),
    ]
    for present, isolated, cx, cy in corners:
        if present and isolated:
            d2 = (X - cx) ** 2 + (Y - cy) ** 2
            keep &= d2 >= b * b
    area_px = (wx / nxp) * (wy / nyp)
    return float(keep.sum() * area_px)


def vizinhos(i, j, nx, ny, toro=False):
    """8-neighbourhood of block (i, j); None outside an open lattice, wrapped on a torus."""
    off = {"L": (-1, 0), "R": (1, 0), "Bo": (0, -1), "T": (0, 1),
           "BL": (-1, -1), "BR": (1, -1), "TL": (-1, 1), "TR": (1, 1)}
    out = {}
    for k, (di, dj) in off.items():
        ii, jj = i + di, j + dj
        if toro:
            out[k] = (ii % nx, jj % ny)
        else:
            out[k] = (ii, jj) if (0 <= ii < nx and 0 <= jj < ny) else None
    return out


def mc_variant(nx, ny, colw, rowh, b, n_tr, n_va, n_te, n_trials, seed,
                variant, raster_sample=0, raster_m=RASTER_M):
    """Run n_trials random role assignments for one variant (A, A2, B, C or T; see the
    module docstring) and return the mean, standard deviation and 5th/95th percentiles
    of the retained area fraction of the validation and test roles."""
    rng = np.random.RandomState(seed)
    N = nx * ny
    assert n_tr + n_va + n_te == N, (n_tr, n_va, n_te, N)
    labels_base = np.array([0] * n_tr + [1] * n_va + [2] * n_te)  # 0 train, 1 val, 2 test

    g_nom = float(np.median(colw))
    bordas_reais = variant == "B" or variant == "C"
    assimetrico = variant in ("A2", "C", "T")
    toro = variant == "T"

    if bordas_reais:
        colw_use, rowh_use = colw, rowh
    else:
        colw_use = np.full(nx, g_nom)
        rowh_use = np.full(ny, g_nom)

    ret_va_trials = np.empty(n_trials)
    ret_te_trials = np.empty(n_trials)
    disc_errs = []
    raster_done = 0

    for t in range(n_trials):
        labels = labels_base.copy()
        rng.shuffle(labels)
        L = labels.reshape(nx, ny)

        # Validation blocks are trimmed by training neighbours only.
        va_area_ret = np.full((nx, ny), np.nan)
        va_area_antes = np.full((nx, ny), np.nan)
        va_retido = np.zeros((nx, ny), dtype=bool)
        for i in range(nx):
            for j in range(ny):
                if L[i, j] != 1:
                    continue
                viz = vizinhos(i, j, nx, ny, toro=toro)
                flags = {}
                for k, pos in viz.items():
                    flags[k] = bool(pos is not None and L[pos] == 0)
                wx, wy = colw_use[i], rowh_use[j]
                A = area_bloco_continua(wx, wy, b, flags["L"], flags["R"],
                                          flags["Bo"], flags["T"], flags["BL"],
                                          flags["BR"], flags["TL"], flags["TR"])
                va_area_ret[i, j] = A
                va_area_antes[i, j] = wx * wy
                va_retido[i, j] = A > 1e-9

                if raster_sample and raster_done < raster_sample and bordas_reais and (wx < g_nom - 1e-6 or wy < g_nom - 1e-6):
                    Ar = area_bloco_raster(wx, wy, b, flags["L"], flags["R"],
                                             flags["Bo"], flags["T"], flags["BL"],
                                             flags["BR"], flags["TL"], flags["TR"], raster_m)
                    denom = max(A, Ar, 1e-12)
                    disc_errs.append(abs(A - Ar) / denom)
                    raster_done += 1

        antes_va = np.nansum(va_area_antes)
        ret_va_trials[t] = float(np.nansum(va_area_ret) / antes_va) if antes_va > 0 else np.nan

        # Test blocks are trimmed by training neighbours and by validation neighbours;
        # under the asymmetric rule only validation blocks that kept area trim.
        te_area_ret = np.full((nx, ny), np.nan)
        te_area_antes = np.full((nx, ny), np.nan)
        for i in range(nx):
            for j in range(ny):
                if L[i, j] != 2:
                    continue
                viz = vizinhos(i, j, nx, ny, toro=toro)
                flags = {}
                for k, pos in viz.items():
                    if pos is None:
                        flags[k] = False
                        continue
                    role = L[pos]
                    if role == 0:
                        flags[k] = True
                    elif role == 1:
                        flags[k] = bool(va_retido[pos]) if assimetrico else True
                    else:
                        flags[k] = False
                wx, wy = colw_use[i], rowh_use[j]
                A = area_bloco_continua(wx, wy, b, flags["L"], flags["R"],
                                          flags["Bo"], flags["T"], flags["BL"],
                                          flags["BR"], flags["TL"], flags["TR"])
                te_area_ret[i, j] = A
                te_area_antes[i, j] = wx * wy

        antes_te = np.nansum(te_area_antes)
        ret_te_trials[t] = float(np.nansum(te_area_ret) / antes_te) if antes_te > 0 else np.nan

    out = {
        "ret_va_media": float(np.nanmean(ret_va_trials)),
        "ret_va_dp": float(np.nanstd(ret_va_trials)),
        "ret_va_p5": float(np.nanpercentile(ret_va_trials, 5)),
        "ret_va_p95": float(np.nanpercentile(ret_va_trials, 95)),
        "ret_te_media": float(np.nanmean(ret_te_trials)),
        "ret_te_dp": float(np.nanstd(ret_te_trials)),
        "ret_te_p5": float(np.nanpercentile(ret_te_trials, 5)),
        "ret_te_p95": float(np.nanpercentile(ret_te_trials, 95)),
        "n_trials": n_trials,
    }
    if raster_sample:
        out["discretizacao_erro_rel_max"] = float(max(disc_errs)) if disc_errs else None
        out["discretizacao_erro_rel_medio"] = float(np.mean(disc_errs)) if disc_errs else None
        out["discretizacao_n_amostras"] = raster_done
    return out


def R_meanfield(g, b, q):
    """Mean-field retained area fraction of a g x g block whose neighbours are
    independently non-trimming with probability q (trimming with p = 1 - q)."""
    return ((g - 2 * b) ** 2 + 4 * b * (g - 2 * b) * q
             + 4 * b ** 2 * (1 - np.pi / 4) * q ** 2 + np.pi * b ** 2 * q ** 3) / g ** 2


def sympy_check_R_meanfield():
    """Symbolic check: (expected side)^2 minus the expected isolated-corner discs
    equals the numerator of R_meanfield."""
    g, b, q = sp.symbols('g b q', positive=True)
    p = 1 - q
    E_lado = g - 2 * b * p
    E_rect = sp.expand(E_lado * E_lado)
    E_corner = 4 * (sp.pi * b ** 2 / 4) * p * (1 - p) ** 2
    E_area = sp.expand(E_rect - E_corner)
    R_texto_num = ((g - 2 * b) ** 2 + 4 * b * (g - 2 * b) * q
                    + 4 * b ** 2 * (1 - sp.pi / 4) * q ** 2 + sp.pi * b ** 2 * q ** 3)
    diff = sp.simplify(sp.expand(E_area) - sp.expand(R_texto_num))
    return {"diferenca_e_zero": diff == 0, "diferenca": str(diff)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", action="store_true")
    args = ap.parse_args()
    if args.hash:
        print(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return

    out = {
        "seed": SEED,
        "raster_m": RASTER_M,
        "sympy_conferencia_R_meanfield": sympy_check_R_meanfield(),
        "validacao_contagem_blocos": {},
        "por_config": {},
    }

    # The role-count rule must reproduce the block counts of the run records:
    # 132 blocks (g = 10 km) and 483 blocks (g = 5 km, Bauru).
    val = {}
    for n_g, esperado in ((132, (92, 20, 20)), (483, (338, 72, 73))):
        got = n_blocos_por_papel(n_g)
        val[str(n_g)] = {"esperado": list(esperado), "obtido": list(got), "bate": got == esperado}
    out["validacao_contagem_blocos"] = val
    if not all(v["bate"] for v in val.values()):
        out["ALERTA"] = "regra de contagem de blocos NAO reproduziu os runs reais -- pare e registre"
        Path("internal/artifact_02.json").write_text(json.dumps(out, indent=1))
        print("ALERTA: regra de contagem nao bateu; abortando.")
        sys.exit(1)

    variantes_lista = ("A", "A2", "B", "C", "T")

    for config, g in (("g10b2", 10.0), ("g5b2", 5.0)):
        b = 2.0
        cidades = list(CIDADES_EXT.keys())
        out["por_config"][config] = {"g_km": g, "b_km": b, "por_cidade": {}}
        for cidade in cidades:
            ext = CIDADES_EXT[cidade]
            nx, ny, colw, rowh = grid_dims(ext["ext_x_km"], ext["ext_y_km"], g)
            N = nx * ny
            n_tr, n_va, n_te = n_blocos_por_papel(N)
            # Non-trimming fraction: test is trimmed by train and val, val by train only.
            q_te = n_te / N
            q_va = 1 - n_tr / N
            cm_te = R_meanfield(g, b, q_te)
            cm_va = R_meanfield(g, b, q_va)

            variantes = {}
            for variant in variantes_lista:
                raster_sample = 40 if (variant == "C" and cidade == cidades[0]) else 0
                seed_v = seed_de_string(config, cidade, variant)
                res = mc_variant(nx, ny, colw, rowh, b, n_tr, n_va, n_te,
                                  N_TRIALS_MAIN, seed_v, variant,
                                  raster_sample=raster_sample)
                variantes[variant] = res

            obs_te = OBS_TEST.get((cidade, config))
            obs_va = OBS_VAL.get((cidade, config))
            entry = {
                "nx": nx, "ny": ny, "n_blocos_total": N,
                "n_blocos": {"train": n_tr, "val": n_va, "test": n_te},
                "colw_km": colw.tolist(), "rowh_km": rowh.tolist(),
                "campo_medio_te": cm_te, "campo_medio_va": cm_va,
                "observado_te": obs_te, "observado_va": obs_va,
                "variantes": variantes,
                "efeito_borda_A_para_B_pp_te": (variantes["B"]["ret_te_media"] - variantes["A"]["ret_te_media"]) * 100,
                "efeito_assimetria_B_para_C_pp_te": (variantes["C"]["ret_te_media"] - variantes["B"]["ret_te_media"]) * 100,
                "efeito_perimetro_T_para_A2_pp_te": (variantes["A2"]["ret_te_media"] - variantes["T"]["ret_te_media"]) * 100,
                "efeito_borda_fina_A2_para_C_pp_te": (variantes["C"]["ret_te_media"] - variantes["A2"]["ret_te_media"]) * 100,
            }
            if obs_te is not None:
                entry["erro_rel_campo_medio_te"] = abs(cm_te - obs_te) / obs_te
                entry["erro_rel_MC_C_te"] = abs(variantes["C"]["ret_te_media"] - obs_te) / obs_te
                entry["obs_dentro_p5_p95_MC_C_te"] = bool(
                    variantes["C"]["ret_te_p5"] <= obs_te <= variantes["C"]["ret_te_p95"])
            if obs_va is not None:
                entry["erro_rel_campo_medio_va"] = abs(cm_va - obs_va) / obs_va
                entry["erro_rel_MC_C_va"] = abs(variantes["C"]["ret_va_media"] - obs_va) / obs_va
                entry["obs_dentro_p5_p95_MC_C_va"] = bool(
                    variantes["C"]["ret_va_p5"] <= obs_va <= variantes["C"]["ret_va_p95"])
            out["por_config"][config]["por_cidade"][cidade] = entry
            print(config, cidade, "C.te=%.5f" % variantes["C"]["ret_te_media"],
                  "campo_medio=%.5f" % cm_te)

    src = Path(__file__).read_text()
    out["script_sha256"] = hashlib.sha256(src.encode()).hexdigest()

    dest = Path("internal/artifact_02.json")
    dest.write_text(json.dumps(out, indent=1))
    print("OK ->", dest)


if __name__ == "__main__":
    main()
