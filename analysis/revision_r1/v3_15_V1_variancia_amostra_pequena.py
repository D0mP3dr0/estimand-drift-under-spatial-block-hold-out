# -*- coding: utf-8 -*-
"""
Estimand Drift under Spatial Block Hold-Out (MDPI Mathematics, revision 1).
V1: small-sample alternatives for the variance of one partition (reviewer 2, item 3).
Criterion: criteria/criterio_V1_variancia_amostra_pequena.json (fixed 2026-10-26T09:35:30-03:00,
before this script existed; the method is not altered here).

Reads results/fase5/R5_por_sorteio.json (S_B, M_B per test block; N = occupied blocks).
Per cell / predictor / population / draw (k = number of blocks with M_B > 0, k >= 2 for v defined):
  base : Err +- 2 sqrt(v),  v = (1-k/N) k/(k-1) sum (S_B - Err M_B)^2 / M^2   (reproduces R5; cross-check)
  t    : Err +- t_{0.975;k-1} sqrt(v)
  jk   : Err_(-B) = (S-S_B)/(M-M_B); v_jk = (k-1)/k sum (Err_(-B) - mean)^2; Err +- 2 sqrt(v_jk) and +- t_{k-1} sqrt(v_jk)
  wild : Rademacher cluster bootstrap, r_B = S_B - Err M_B, R = 999, Err* = Err + sum w_B r_B / M; 2.5-97.5 percentile
         and symmetric Err +- t_{k-1} sd(Err*); seed 20261026 (one Generator, order cell > draw > predictor > population).
Coverage target = mean of Err over draws with Err defined (reading A). No interpretation: numbers only.
Six intervals (base, t, jk2, jkt, wild_pct, wild_t) are those enumerated by the fixed method.

Output directory: env R1_OUT (default results/fase6_rerun under the repository root).
"""
from __future__ import annotations
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve()
SCRIPT_SHA256 = hashlib.sha256(HERE.read_bytes()).hexdigest()  # digest of this very file
ROOT = HERE.parents[2]  # repository root
F5 = ROOT / "results" / "fase5"
F6 = Path(os.environ.get("R1_OUT", ROOT / "results" / "fase6_rerun"))
R5_POR_SORTEIO = F5 / "R5_por_sorteio.json"
R5_RESUMO = F5 / "R5_resumo.json"
BLOCOS_VALIDO = F5 / "blocos_com_no_valido.json"
CRITERIO = ROOT / "criteria" / "criterio_V1_variancia_amostra_pequena.json"
SCRIPT_REL = HERE.relative_to(ROOT).as_posix()
CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
QUADRANTES = ["Q1", "Q2", "Q3", "Q4"]
PREDS = ("constante", "fspl_b")
POPS = ("validos", "todos")
METODOS = ("base", "t", "jk2", "jkt", "wild_pct", "wild_t")
ESTIMADORES_VAR = ("base", "jackknife", "wild")
R_BOOT = 999
SEMENTE = 20261026
TOL = 1e-9
LOG = []


def log(s):
    print(s)
    LOG.append(s)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return Path(p).relative_to(ROOT).as_posix()


def sorteio(S, M, N, rng):
    k = int(M.size)
    Mt = float(M.sum())
    if k == 0 or Mt <= 0:
        return {"k": k, "Err": None, "definido": False}
    Err = float(S.sum() / Mt)
    if k < 2:
        return {"k": k, "Err": Err, "definido": False}
    r = S - Err * M
    v = (1.0 - k / N) * k / (k - 1.0) * float(np.sum(r ** 2)) / (Mt * Mt)
    tq = float(stats.t.ppf(0.975, k - 1))
    sd = np.sqrt(v)
    e_out = (S.sum() - S) / (Mt - M)
    vjk = (k - 1.0) / k * float(np.sum((e_out - e_out.mean()) ** 2))
    sjk = np.sqrt(vjk)
    w = rng.choice(np.array([-1.0, 1.0]), size=(R_BOOT, k))
    estar = Err + (w @ r) / Mt
    lo, hi = np.percentile(estar, [2.5, 97.5])
    vb = float(np.var(estar, ddof=1))
    sb = np.sqrt(vb)
    return {"k": k, "Err": Err, "definido": True, "t_q": tq,
            "v_base": v, "v_jackknife": vjk, "v_wild": vb,
            "int": {"base": [Err - 2 * sd, Err + 2 * sd], "t": [Err - tq * sd, Err + tq * sd],
                    "jk2": [Err - 2 * sjk, Err + 2 * sjk], "jkt": [Err - tq * sjk, Err + tq * sjk],
                    "wild_pct": [float(lo), float(hi)], "wild_t": [Err - tq * sb, Err + tq * sb]}}


def resumo_celula(regs):
    E = [r for r in regs if r["Err"] is not None]
    D = [r for r in regs if r["definido"]]
    errs = np.array([r["Err"] for r in E])
    var_emp = float(np.var(errs, ddof=1)) if len(errs) >= 2 else None
    alvo = float(errs.mean()) if len(errs) else None
    ks = [r["k"] for r in regs]
    out = {"n_sorteios": len(regs), "n_Err_definido": len(E), "n_v_definido": len(D),
           "n_sorteios_k_lt_2": len(regs) - len(D),
           "k_mediana": float(np.median(ks)), "k_min": int(min(ks)), "k_max": int(max(ks)),
           "alvo_media_Err": alvo, "var_empirica_Err": var_emp, "cobertura": {}, "n_cobertos": {}, "Q": {}}
    for m in METODOS:
        if D and alvo is not None:
            c = np.array([(r["int"][m][0] <= alvo <= r["int"][m][1]) for r in D])
            out["cobertura"][m] = float(c.mean())
            out["n_cobertos"][m] = int(c.sum())
        else:
            out["cobertura"][m] = None
            out["n_cobertos"][m] = None
    for nome, chave in zip(ESTIMADORES_VAR, ("v_base", "v_jackknife", "v_wild")):
        if D and var_emp not in (None, 0.0):
            out["Q"][nome] = float(np.mean([r[chave] for r in D]) / var_emp)
        else:
            out["Q"][nome] = None
    return out


def main():
    F6.mkdir(parents=True, exist_ok=True)
    crit_sha = sha(CRITERIO)
    r5 = json.loads(R5_RESUMO.read_text(encoding="utf-8"))
    bv = json.loads(BLOCOS_VALIDO.read_text(encoding="utf-8"))["por_celula"]
    r5s = json.loads(R5_POR_SORTEIO.read_text(encoding="utf-8"))["celulas"]  # replaces the per-cell partial files
    celulas = [f"{c}_{q}" for c in CIDADES for q in QUADRANTES]
    rng = np.random.default_rng(SEMENTE)
    por_sorteio, res = {}, {}
    for ch in celulas:
        r = r5s[ch]
        assert len(r["sorteios"]) == 60, ch
        N = float(r["N_blocos_ocupados"])
        por_sorteio[ch] = {"N_blocos_ocupados": r["N_blocos_ocupados"], "sorteios": []}
        regs = {(p, o): [] for p in PREDS for o in POPS}
        for s in r["sorteios"]:
            linha = {"split_seed": s["split_seed"]}
            for p in PREDS:
                for o in POPS:
                    x = s[f"{p}__{o}"]
                    e = sorteio(np.array(x["S_B"], dtype=np.float64), np.array(x["M_B"], dtype=np.float64), N, rng)
                    regs[(p, o)].append(e)
                    linha[f"{p}__{o}"] = e
            por_sorteio[ch]["sorteios"].append(linha)
        for (p, o), lst in regs.items():
            c = resumo_celula(lst)
            c["N_blocos_ocupados"] = int(N)
            c["n_blocos_com_no_valido"] = bv[ch]["blocos_com_ao_menos_1_no_valido"]
            res.setdefault(f"{p}__{o}", {})[ch] = c
        log(f"celula {ch} ok")

    # ---- conferencia contra R5 (leitura A, N_ocupados)
    conf, maxd = {}, 0.0
    for po, cels in res.items():
        ref = r5["resumo"][po]["N_ocupados"]["leitura_A"]["por_celula"]
        for ch, c in cels.items():
            rr = ref[ch]
            d = {}
            for nome, a, b in (("cobertura", c["cobertura"]["base"], rr["cobertura"]),
                               ("Q", c["Q"]["base"], rr["Q"]),
                               ("k_mediano", c["k_mediana"], rr["k_mediano"]),
                               ("n_indefinidos", c["n_sorteios_k_lt_2"], rr["n_indefinidos_k_lt_2"])):
                d[nome] = 0.0 if (a is None and b is None) else (float("inf") if (a is None or b is None) else abs(a - b))
            conf.setdefault(po, {})[ch] = d
            maxd = max(maxd, max(d.values()))
    bateu = bool(maxd <= TOL)
    log(f"CONFERENCIA R5 (cobertura, Q, k mediano, n indefinidos; 64 combinacoes celula/pred/pop): max |dif| = {maxd:.3e}; bateu={bateu}")
    carimbo = {"script": SCRIPT_REL, "script_sha256": SCRIPT_SHA256, "criterio": rel(CRITERIO), "criterio_sha256": crit_sha,
               "data_utc": datetime.now(timezone.utc).isoformat(), "comando": " ".join(["python", SCRIPT_REL] + sys.argv[1:]),
               "semente_bootstrap": SEMENTE, "R_bootstrap": R_BOOT, "venv": "python",
               "R5_por_sorteio_sha256": sha(R5_POR_SORTEIO), "R5_resumo_sha256": sha(R5_RESUMO),
               "blocos_com_no_valido_sha256": sha(BLOCOS_VALIDO)}
    conferencia = {"fonte": rel(R5_RESUMO), "leitura": "A, N_ocupados", "tolerancia": TOL,
                   "max_abs_diferenca": maxd, "bateu": bateu, "por_celula": conf}
    if not bateu:
        (F6 / "V1_conferencia_R5_FALHOU.json").write_text(json.dumps({**carimbo, "conferencia_R5": conferencia}, indent=1), encoding="utf-8")
        (F6 / "V1_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
        raise SystemExit("PARA: conferencia contra R5 nao bateu")

    # ---- contagens e lacuna
    cont, lac = {}, {}
    for po, cels in res.items():
        cont[po], lac[po] = {}, {}
        for m in METODOS:
            cobs = {ch: c["cobertura"][m] for ch, c in cels.items()}
            cont[po][m] = {"n_celulas_cobertura_ge_0.90": sum(1 for v in cobs.values() if v is not None and v >= 0.90),
                           "n_celulas_cobertura_lt_0.80": sum(1 for v in cobs.values() if v is not None and v < 0.80),
                           "n_celulas_cobertura_indefinida": sum(1 for v in cobs.values() if v is None),
                           "mediana_cobertura": float(np.median([v for v in cobs.values() if v is not None]))}
            f = {}
            for ch, c in cels.items():
                cb, cm = c["cobertura"]["base"], c["cobertura"][m]
                f[ch] = None if (cb is None or cm is None or cb >= 1.0) else (cm - cb) / (1.0 - cb)
            vals = [v for v in f.values() if v is not None]
            lac[po][m] = {"por_celula": f, "mediana": float(np.median(vals)) if vals else None, "n_definidas": len(vals)}
    (F6 / "V1_por_sorteio.json").write_text(json.dumps({**carimbo, "id": "V1_por_sorteio", "celulas": por_sorteio}, indent=1), encoding="utf-8")
    (F6 / "V1_resumo.json").write_text(json.dumps({
        **carimbo, "id": "V1_resumo", "metodos": list(METODOS),
        "definicoes": {"base": "Err+-2sqrt(v) (R5)", "t": "Err+-t_{k-1}sqrt(v)", "jk2": "jackknife-bloco, +-2", "jkt": "jackknife-bloco, +-t_{k-1}",
                       "wild_pct": "wild cluster bootstrap Rademacher, percentil 2.5-97.5", "wild_t": "wild bootstrap, Err+-t_{k-1} dp(Err*)",
                       "Q": "media da variancia estimada (sorteios com k>=2) / variancia empirica de Err entre sorteios com Err definido",
                       "lacuna_fechada": "(cob_metodo - cob_base)/(1 - cob_base) por celula; indefinida se cob_base=1"},
        "conferencia_R5": conferencia, "contagens_limiares": cont, "fracao_lacuna_fechada": lac,
        "por_celula": res, "interpretacao": "NAO feita neste artefato"}, indent=1), encoding="utf-8")

    # ---- tabela LaTeX
    L = [r"\begin{tabular}{lcccccc}", r"\hline",
         r"Cell & $k$ median (min--max) & $N_{\mathrm{valid}}$ & Base & $t$ & Jackknife-$t$ & Wild-$t$ \\", r"\hline"]
    fm = lambda v: "--" if v is None else f"{v:.2f}"
    for ch in celulas:
        c = res["constante__validos"][ch]
        cb = c["cobertura"]
        L.append(f"{ch.replace('_', ' ')} & {c['k_mediana']:g} ({c['k_min']}--{c['k_max']}) & {c['n_blocos_com_no_valido']} & "
                 f"{fm(cb['base'])} & {fm(cb['t'])} & {fm(cb['jkt'])} & {fm(cb['wild_t'])} \\\\")
    L += [r"\hline", r"\end{tabular}"]
    (F6 / "T_V1_cobertura.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    for po in cont:
        for m in METODOS:
            log(f"{po} {m}: {cont[po][m]}  lacuna_mediana={lac[po][m]['mediana']}")
    (F6 / "V1_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
