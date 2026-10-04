#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launcher of block 4 of campaign G1: crossed draw x training-seed block in Campinas Q1.

Runs the 20 runs defined by the decision criterion
`criterio_G1b_campinas_bloco_cruzado.json`: cell Campinas Q1, training seeds 43 and 44,
the five split draws listed in the criterion (all among the first 20 of the 60-draw list
in `fase2/2.1_deriva_erro_baselines_16x60rnd.json`), GNN and MLP, at g = 10 km and
b = 2 km through the shim `train_v3_g10.py`. Order: by draw, then by seed, GNN before
MLP, so an interrupted batch leaves complete GNN/MLP pairs. The criterion fields and the
draws are checked by assertions when the plan is built.

Same logic as `rodar_lote_G1_retomada.py` (shim digest check, free-VRAM guard, atomic
status writes, fixed configuration, resumability, handling of draws without valid test
nodes and of dataset digest mismatches), with its own output files and a 4 h GPU-time
cap. A failed run whose log reports a degenerate partition (a split with no
antenna-to-terrain edge) other than `test` is marked not trainable, excluded from the
cell for both models and not replaced, following `criterio_G1_adendo4.json`; a
degenerate test partition stops the batch.

Outputs (next to this script, the run folders shared with the other G1 launchers):
`plano_G1_bloco4.json`, `lote_G1_bloco4_status.json`, `lote_G1_bloco4_driver.log`,
`nao_treinaveis_G1_bloco4.json`, one folder and one `log_<label>.txt` per run.
Exit codes as in `rodar_lote_G1.py`.

Usage: python rodar_lote_G1_bloco4.py   (with the CUDA environment interpreter)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

G1_DIR = Path(__file__).resolve().parent
RAIZ = G1_DIR.parent.parent
MODELO_V3_DIR = G1_DIR.parent / "modelo_v3"
CRITERIO_PATH = RAIZ / "criterios" / "criterio_G1b_campinas_bloco_cruzado.json"
SORTEIOS_PATH = RAIZ / "fase2" / "2.1_deriva_erro_baselines_16x60rnd.json"
PARCIAL_2_1_PATH = RAIZ / "fase2" / "_v3_2.1_3.1_parcial_16x60rnd.json"
MANIFEST_V4 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/manifest_mathematics_v4.jsonl")
STATUS_PATH = G1_DIR / "lote_G1_bloco4_status.json"
LOG_PATH = G1_DIR / "lote_G1_bloco4_driver.log"
PLANO_PATH = G1_DIR / "plano_G1_bloco4.json"
NAO_TREINAVEIS_PATH = G1_DIR / "nao_treinaveis_G1_bloco4.json"
CRITERIO_ADENDO4_PATH = RAIZ / "criterios" / "criterio_G1_adendo4.json"
NOTA_NAO_TREINAVEL = "excluido da celula para GNN e MLP; nao substituido"

GRID_KM = 10.0
BUFFER_KM = 2.0
CFG = {"epochs": 8, "hidden_dim": 256, "batch_size": 12288, "lr": 0.001,
       "k_antenna": -1, "k_terrain": -1, "mmap": True}
CIDADES = ("bauru", "campinas")

# Hard GPU-time cap of block 4 (planned ~3 h); restarting resets it.
ORCAMENTO_GPU_S = 4 * 3600
# Duration guess per model type before any run of that type has finished.
PROJECAO_INICIAL_S = {"gnn": 20 * 60, "mlp": 4 * 60}
VRAM_LIVRE_MINIMA_MIB = 1024

# Must be the CUDA environment interpreter; each run is launched with it.
PYTHON = sys.executable


def log(msg: str) -> None:
    linha = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(linha, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def carregar_status() -> dict:
    if STATUS_PATH.exists():
        with open(STATUS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"iniciado_em": datetime.now(timezone.utc).isoformat(), "corridas": [],
            "orcamento_gpu_s": ORCAMENTO_GPU_S, "veredito_lote": None}


def salvar_status(status: dict) -> None:
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    tmp.replace(STATUS_PATH)  # atomic rename


def particoes_degeneradas(log_path: Path) -> list[str]:
    """Partition names listed in 'PARTICAO DEGENERADA: [...]' lines of a run log (empty if none)."""
    try:
        txt = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    achadas: list[str] = []
    for m in re.finditer(r"PARTICAO DEGENERADA:\s*\[([^\]]*)\]", txt):
        for nome in re.findall(r"['\"]?([A-Za-z_]+)['\"]?", m.group(1)):
            if nome not in achadas:
                achadas.append(nome)
    return achadas


def marcar_nao_treinavel(status: dict, registro: dict, particoes: list[str], origem: str) -> None:
    """Add the not-trainable marker to a run record (no field is removed), save the status,
    and upsert the run in the not-trainable list file.
    """
    registro["nao_treinavel"] = True
    registro["nao_treinavel_particao"] = particoes
    registro["nao_treinavel_nota"] = NOTA_NAO_TREINAVEL
    registro["nao_treinavel_marcado_em"] = datetime.now(timezone.utc).isoformat()
    registro["nao_treinavel_origem"] = origem
    registro["nao_treinavel_criterio"] = str(CRITERIO_ADENDO4_PATH)
    salvar_status(status)
    try:
        cur = json.loads(NAO_TREINAVEIS_PATH.read_text(encoding="utf-8")) if NAO_TREINAVEIS_PATH.exists() else {"criterio": str(CRITERIO_ADENDO4_PATH), "nao_treinaveis": []}
    except Exception:
        cur = {"criterio": str(CRITERIO_ADENDO4_PATH), "nao_treinaveis": []}
    cur["nao_treinaveis"] = [x for x in cur["nao_treinaveis"] if x["run_label"] != registro["run_label"]] + [{
        "run_label": registro["run_label"], "tipo": registro["tipo"], "cidade": registro["cidade"],
        "quadrante": registro["quadrante"], "split_seed": registro["split_seed"],
        "seed_treino": registro["seed_treino"], "rc": registro["rc"], "particao_degenerada": particoes,
        "nota": NOTA_NAO_TREINAVEL, "origem": origem, "log": registro["log"],
        "marcado_em": registro["nao_treinavel_marcado_em"]}]
    tmp = NAO_TREINAVEIS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cur, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(NAO_TREINAVEIS_PATH)  # atomic rename


def vram_livre_mib() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15, check=True).stdout.strip().splitlines()[0]
        return float(out)
    except Exception as e:
        log(f"[lote_G1] AVISO: nao consegui ler VRAM livre via nvidia-smi ({e}); assumindo 0 MiB (guarda conservadora).")
        return 0.0


def manifest_cftudo() -> dict:
    """Map tensor file name -> SHA-256 for the `tensores_cftudo` group of the v4 manifest."""
    out = {}
    with open(MANIFEST_V4, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                e = json.loads(ln)
                if e.get("grupo") == "tensores_cftudo":
                    out[Path(e.get("caminho", "")).name] = e.get("sha256")
    return out


SHIM_PATH = G1_DIR / "train_v3_g10.py"


def guarda_shim() -> tuple[bool, str]:
    """Check that the shim exists, that every file it wraps still has the SHA-256 recorded
    in the shim, and that the shim uses the same g and b as this launcher.
    """
    if not SHIM_PATH.exists():
        return False, f"shim ausente: {SHIM_PATH}"
    sys.path.insert(0, str(G1_DIR))
    import train_v3_g10 as shim  # imports only the standard library; does not train
    ruins = {}
    for p, esp in shim.SHA_ESPERADOS.items():
        if not Path(p).exists():
            ruins[str(p)] = "ausente"
        elif shim.sha256(Path(p)) != esp:
            ruins[str(p)] = "sha256 divergente"
    if shim.GRID_KM != GRID_KM or shim.BUFFER_KM != BUFFER_KM:
        ruins["g/b"] = f"shim={shim.GRID_KM}/{shim.BUFFER_KM}"
    return (not ruins), (json.dumps(ruins) if ruins else "ok")


def gerar_plano() -> list[dict]:
    """Build the 20-run plan of block 4 from the criterion and validate it."""
    with open(CRITERIO_PATH, "r", encoding="utf-8") as f:
        crit = json.load(f)
    assert crit["id"] == "G1b_campinas_bloco_cruzado_completo"
    d = crit["desenho"]
    assert d["bloco"] == 4 and d["celula"] == "campinas_Q1" and d["n_corridas"] == 20
    assert d["sementes_treino"] == [43, 44] and d["modelos"] == ["gnn", "mlp"]
    sorteios = [int(x) for x in d["sorteios"]]
    assert len(sorteios) == 5
    with open(SORTEIOS_PATH, "r", encoding="utf-8") as f:
        lista = json.load(f)["nota_divergencia_seeds"]["seeds_usados_nesta_rodada"]
    assert len(lista) == 60 and all(s in lista[:20] for s in sorteios)
    plano: list[dict] = []
    for ss in sorteios:
        for sd in d["sementes_treino"]:
            for tipo in d["modelos"]:
                plano.append({"bloco": 4, "run_label": f"g1_{tipo}_campinas_Q1_ss{ss}_s{sd}",
                              "tipo": tipo, "cidade": "campinas", "quadrante": "Q1",
                              "seed_treino": sd, "split_seed": ss, "indice_sorteio": lista.index(ss)})
    assert len(plano) == 20 and len({c["run_label"] for c in plano}) == 20
    return plano


def validos_conhecidos() -> dict:
    """Map (cell, split_seed) -> number of valid test nodes from the 2.1 analysis; used only to
    classify runs on draws with no valid test node.
    """
    out = {}
    with open(PARCIAL_2_1_PATH, "r", encoding="utf-8") as f:
        p = json.load(f)
    for cel, d in p["celulas"].items():
        for s in d.get("por_sorteio", []):
            if s.get("status") == "ok":
                out[(cel, int(s["split_seed"]))] = s["n_pop_teste"]["validos"]
    return out


def corrida_completa(run_label: str, man: dict, cidade: str, quadrante: str) -> bool:
    """True if the run record and predictions exist and the recorded dataset digest matches the manifest."""
    d = G1_DIR / run_label
    rj, npz = d / f"run_{run_label}.json", d / f"predicoes_{run_label}.npz"
    if not (rj.exists() and npz.exists()):
        return False
    try:
        with open(rj, "r", encoding="utf-8") as f:
            r = json.load(f)
    except Exception:
        return False
    ins = r.get("insumos") or {}
    nome = f"transfer_dataset_{cidade}_v19_{quadrante}_enriched_cftudo.pt"
    return bool("modelo_v3" in r and ins.get("sha256_rf_data_bate_manifest_v4") is True
                and ins.get("sha256_rf_data") and ins.get("sha256_rf_data") == man.get(nome))


def main() -> int:
    ok_shim, motivo_shim = guarda_shim()
    if not ok_shim:
        log(f"[lote_G1] RECUSADO: guarda do shim falhou ({motivo_shim}). rc=5")
        return 5
    plano = gerar_plano()
    with open(PLANO_PATH, "w", encoding="utf-8") as f:
        json.dump({"criterio": str(CRITERIO_PATH), "grid_km": GRID_KM, "buffer_km": BUFFER_KM, "n": len(plano),
                   "corridas": plano}, f, indent=1, ensure_ascii=False)
    man = manifest_cftudo()
    vconh = validos_conhecidos()

    status = carregar_status()
    ja_sem_validos = {c["run_label"] for c in status["corridas"] if c.get("sem_validos")}
    status.setdefault("retomadas", []).append({"iniciada_em": datetime.now(timezone.utc).isoformat(),
                                               "lancador": "rodar_lote_G1_bloco4.py",
                                               "criterio": str(CRITERIO_PATH)})
    if status.get("veredito_lote") is not None:
        status["veredito_lote_anterior"] = status["veredito_lote"]
    salvar_status(status)

    # Earlier failed runs whose log shows a degenerate non-test partition are marked
    # not trainable and are not re-run (status starts empty for this block).
    for c in status["corridas"]:
        if c.get("rc") not in (None, 0) and not c.get("nao_treinavel") and not c.get("sem_validos"):
            pd = particoes_degeneradas(Path(c.get("log", "")))
            if pd and "test" not in pd:
                marcar_nao_treinavel(status, c, pd, "falha anterior a emenda; reconhecida pelo log existente")
                log(f"[lote_G1_retomada] {c['run_label']}: rc={c['rc']} com particao degenerada {pd} -- marcada nao_treinavel (registro original mantido), NAO reexecutada.")
    ja_nao_treinavel = {c["run_label"] for c in status["corridas"] if c.get("nao_treinavel")}

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"

    t0_lote = time.perf_counter()
    duracoes_ok = {"gnn": [], "mlp": []}
    for c in status["corridas"]:
        if c.get("rc") == 0:
            duracoes_ok.setdefault(c["tipo"], []).append(c["tempo_s"])

    log(f"[lote_G1_bloco4] inicio: {len(plano)} corridas planejadas, g={GRID_KM} b={BUFFER_KM}, teto={ORCAMENTO_GPU_S}s")
    for corrida in plano:
        run_label, tipo = corrida["run_label"], corrida["tipo"]
        cidade, quadrante = corrida["cidade"], corrida["quadrante"]
        seed, split_seed = corrida["seed_treino"], corrida["split_seed"]

        if run_label in ja_nao_treinavel:
            log(f"[lote_G1] {run_label} ja marcada nao_treinavel -- pulando (nao reexecuta).")
            continue
        if run_label in ja_sem_validos or corrida_completa(run_label, man, cidade, quadrante):
            log(f"[lote_G1] {run_label} ja concluida -- pulando (retomavel).")
            continue

        decorrido = time.perf_counter() - t0_lote
        obs = duracoes_ok.get(tipo) or []
        estimativa = (max(obs) * 1.3) if obs else PROJECAO_INICIAL_S[tipo]
        if decorrido + estimativa > ORCAMENTO_GPU_S:
            log(f"[lote_G1] ORCAMENTO: decorrido={decorrido:.0f}s + estimativa({tipo})={estimativa:.0f}s "
                f"> teto={ORCAMENTO_GPU_S}s -- NAO inicia {run_label}. abortado_por_orcamento.")
            status["veredito_lote"] = "abortado_por_orcamento"
            status["tempo_total_lote_s"] = decorrido
            salvar_status(status)
            return 2

        vram_livre = vram_livre_mib()
        if vram_livre < VRAM_LIVRE_MINIMA_MIB:
            log(f"[lote_G1] VRAM livre={vram_livre:.0f} MiB < {VRAM_LIVRE_MINIMA_MIB} MiB -- NAO inicia {run_label}. Parando.")
            status["veredito_lote"] = "abortado_por_vram"
            status["tempo_total_lote_s"] = decorrido
            salvar_status(status)
            return 3

        argv = [
            PYTHON, str(SHIM_PATH), "--modelo", tipo,
            "--cidade", cidade, "--quadrante", quadrante,
            "--seed-treino", str(seed), "--split-seed", str(split_seed),
            "--epochs", str(CFG["epochs"]),
            "--evid-dir", str(G1_DIR), "--run-label", run_label,
            *(["--hidden-dim", str(CFG["hidden_dim"])] if tipo == "gnn" else []),
            "--batch-size", str(CFG["batch_size"]), "--lr", str(CFG["lr"]),
        ]
        if tipo == "gnn":
            argv += ["--k-antenna", str(CFG["k_antenna"]), "--k-terrain", str(CFG["k_terrain"])]
        if CFG["mmap"]:
            argv += ["--mmap"]

        log(f"[lote_G1] iniciando {run_label} (bloco {corrida['bloco']}, vram_livre={vram_livre:.0f} MiB): {' '.join(argv)}")
        iniciado_em = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        log_path = G1_DIR / f"log_{run_label}.txt"
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(argv, stdout=logf, stderr=subprocess.STDOUT, cwd=str(MODELO_V3_DIR), env=env)
        tempo_s = time.perf_counter() - t0
        concluido_em = datetime.now(timezone.utc).isoformat()

        registro = {"run_label": run_label, "bloco": corrida["bloco"], "tipo": tipo, "cidade": cidade,
                    "quadrante": quadrante, "seed_treino": seed, "split_seed": split_seed,
                    "grid_km": GRID_KM, "buffer_km": BUFFER_KM, "comando": argv, "rc": proc.returncode,
                    "tempo_s": tempo_s, "iniciado_em": iniciado_em, "concluido_em": concluido_em, "log": str(log_path)}
        status["corridas"] = [c for c in status["corridas"] if c["run_label"] != run_label] + [registro]

        if proc.returncode == 0:
            salvar_status(status)
            duracoes_ok.setdefault(tipo, []).append(tempo_s)
            log(f"[lote_G1] {run_label} OK em {tempo_s:.1f}s")
        elif proc.returncode == 3:  # trainer exit code 3: dataset digest differs from the v4 manifest
            log(f"[lote_G1] {run_label} ABORTOU rc=3 (sha256 do dataset NAO bate com manifest v4 -- achado de proveniencia, ver {log_path}). Parando.")
            status["veredito_lote"] = "abortado_por_proveniencia"
            status["tempo_total_lote_s"] = time.perf_counter() - t0_lote
            salvar_status(status)
            return 4
        elif vconh.get((f"{cidade}_{quadrante}", split_seed)) == 0:
            registro["sem_validos"] = True
            registro["nota"] = "n_pop_teste.validos == 0 na analise 2.1; registrada como 'sem validos', nao substituida"
            salvar_status(status)
            log(f"[lote_G1] {run_label} rc={proc.returncode} em sorteio SEM VALIDOS (2.1) -- registrada como sem_validos, segue.")
        elif (lambda pd: bool(pd) and "test" not in pd)(particoes_degeneradas(log_path)):  # degenerate non-test partition
            pd = particoes_degeneradas(log_path)
            marcar_nao_treinavel(status, registro, pd, "detectada nesta retomada")
            ja_nao_treinavel.add(run_label)
            log(f"[lote_G1] {run_label} rc={proc.returncode} PARTICAO DEGENERADA {pd} -- registrada nao_treinavel, segue.")
        else:
            log(f"[lote_G1] {run_label} FALHOU rc={proc.returncode} -- ver {log_path}")
            status["veredito_lote"] = "abortado_por_falha_corrida"
            status["tempo_total_lote_s"] = time.perf_counter() - t0_lote
            salvar_status(status)
            return 1

    status["veredito_lote"] = "concluido"
    status["tempo_total_lote_s"] = time.perf_counter() - t0_lote
    salvar_status(status)
    log("[lote_G1] TODAS as corridas planejadas foram concluidas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
