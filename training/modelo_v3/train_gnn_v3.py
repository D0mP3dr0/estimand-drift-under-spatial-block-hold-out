#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Third-campaign GNN trainer: the frozen GNN trainer with the affine decoder.

Loads the frozen trainer `training/frozen/train_gnn_c0_spatial.py` by importlib,
replaces only its decoder class (`PhysicsConstrainedDecoder` -> `AffineDecoderV3`
from `rf_decoder_v3.py`, see `v3_common.patch_decoder_gnn`) and calls its
`main()`, so data loading, window, buffered spatial block split, training,
checkpoint selection on validation and the run JSON are those of the frozen
trainer. Training always starts from scratch (no `--transfer-from`). Used by the
drivers of the model campaigns (`results/gpu/A*/rodar_lote_A*.py`).

Inputs: `transfer_dataset_<city>_v19_<Q>_enriched_cftudo.pt` and
`<city>_v19_<Q>_gpu.pt` in `--graph-dir`; the SHA-256 of the RF data file is
checked against the v4 manifest (group `tensores_cftudo`) before any GPU work,
and the run aborts with return code 3 on a mismatch.
Outputs (in `--evid-dir/<run-label>/`): the frozen trainer's run JSON, extended
with the keys `modelo_v3`, `insumos`, `diagnostico_treino`, `capacidade`,
`diagnostico_saida` and `diagnostico_avaliacao`; and `predicoes_<label>.npz`
with the test-partition predictions (`idx_global`, `target` [N, 5], `pred`
[N, 5] clamped to physical ranges, `sentinela`).

Usage: train_gnn_v3.py --cidade bauru --quadrante Q2 --evid-dir OUT --run-label LABEL
       [--seed-treino 42] [--split-seed 42] [--epochs 8] [--k-antenna -1] [--k-terrain -1]
Seeds: `--seed-treino` is passed as the frozen trainer's `--seed`; `--split-seed`
fixes the split draw. Validation and test always use the complete neighbourhood.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_common as v3  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cidade", required=True)
    p.add_argument("--quadrante", required=True)
    p.add_argument("--seed-treino", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--evid-dir", type=str, required=True)
    p.add_argument("--run-label", type=str, required=True)
    p.add_argument("--max-nodes", type=int, default=0)
    p.add_argument("--mmap", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--graph-dir", type=str, default=str(v3.GRAPH_DIR_DEFAULT))
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=24576)
    p.add_argument("--k-antenna", type=int, default=-1,
                    help="Antenna neighbours sampled in training. The default -1 takes the "
                         "complete neighbourhood (the real degree is at most 5, so -1 is "
                         "equivalent to the earlier value 20). Pass 20 for the earlier value.")
    p.add_argument("--k-terrain", type=int, default=-1,
                    help="A0-ter item 2: DEFAULT do Plano A e -1 (vizinhanca completa no "
                         "TREINO tambem, +-7%% de nos; grau max=9). Passe 8 para o valor "
                         "antigo (mantido disponivel, nao removido).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-batch-size", type=int, default=0)
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--disjoint-treino", action="store_true",
                    help="A0-bis item 3: NeighborLoader(disjoint=True) tambem no "
                         "loader de TREINO (val/teste ja sao sempre disjuntos); "
                         "mede o custo de VRAM antes de virar default.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    def log(msg):
        print(f"[train_gnn_v3 +{time.perf_counter()-t0:.1f}s] {msg}", flush=True)

    rf_data_file = f"transfer_dataset_{args.cidade}_v19_{args.quadrante}_enriched_cftudo.pt"
    graph_file = f"{args.cidade}_v19_{args.quadrante}_gpu.pt"
    graph_dir = Path(args.graph_dir)
    for f in (graph_dir / rf_data_file, graph_dir / graph_file):
        if not f.exists():
            raise FileNotFoundError(f"Dado esperado nao encontrado: {f}")

    # Provenance gate: the RF data file must match the v4 manifest before any GPU work.
    prov = v3.verificar_proveniencia_dataset(graph_dir / rf_data_file, graph_dir / graph_file)
    log(f"proveniencia: sha256_rf_data={prov['sha256_rf_data'][:16]}... "
        f"metodo={prov['metodo_sha_rf_data']} "
        f"encontrado_no_manifest_v4={prov['encontrado_no_manifest_v4']} "
        f"bate_manifest_v4={prov['sha256_rf_data_bate_manifest_v4']}")
    if not prov["sha256_rf_data_bate_manifest_v4"]:
        log(f"ABORTANDO (rc=3): sha256 do rf_data_file NAO bate com o manifest v4 "
            f"(grupo tensores_cftudo). arquivo={graph_dir / rf_data_file} "
            f"sha_calculado={prov['sha256_rf_data']} sha_manifest={prov['sha256_manifest_v4']}")
        return 3

    # The decoder class must be replaced before any model is instantiated.
    v3.patch_decoder_gnn()
    import gnn_rf_model  # already imported and patched by patch_decoder_gnn()
    assert gnn_rf_model.PhysicsConstrainedDecoder is v3.AffineDecoderV3, (
        "monkeypatch do decoder GNN nao pegou -- abortando antes de gastar GPU")

    # Loads the frozen trainer as a module (defines its functions; main() is not run yet).
    mod = v3.carregar_modulo_congelado(v3.FROZEN_GNN_SCRIPT, "train_gnn_c0_spatial_v3")

    # NeighborLoader with disjoint=True on validation and test always, and on
    # training only with --disjoint-treino.
    v3.patch_neighborloader_disjoint(mod, disjoint_treino=args.disjoint_treino)

    evid_dir = Path(args.evid_dir)
    evid_dir.mkdir(parents=True, exist_ok=True)

    congelado_argv = [
        "--base-dir", str(v3.GNN_RF_V2),
        "--graph-dir", str(graph_dir),
        "--rf-data-file", rf_data_file,
        "--graph-file", graph_file,
        "--epochs", str(args.epochs),
        "--seed", str(args.seed_treino),
        "--split-seed", str(args.split_seed),
        "--evid-dir", str(evid_dir),
        "--run-label", args.run_label,
        "--hidden-dim", str(args.hidden_dim),
        "--batch-size", str(args.batch_size),
        "--k-antenna", str(args.k_antenna),
        "--k-terrain", str(args.k_terrain),
        "--lr", str(args.lr),
        # No --transfer-from: the frozen trainer's default "" means training from scratch.
    ]
    if args.max_nodes and args.max_nodes > 0:
        congelado_argv += ["--max-nodes", str(args.max_nodes)]
    if args.mmap:
        congelado_argv += ["--mmap"]
    if args.smoke:
        congelado_argv += ["--smoke"]
    if args.eval_batch_size:
        congelado_argv += ["--eval-batch-size", str(args.eval_batch_size)]
    if args.no_baselines:
        congelado_argv += ["--no-baselines"]

    log(f"chamando mod.main({congelado_argv}) sob instrumentacao (item 4 do A0-bis)")
    with v3.InstrumentacaoTreino() as instr:
        rc = mod.main(congelado_argv)
    if rc != 0:
        log(f"ATENCAO: mod.main retornou {rc} (!=0)")
        return rc

    run_json_path = evid_dir / args.run_label / f"run_{args.run_label}.json"
    with open(run_json_path, "r", encoding="utf-8") as f:
        rec = json.load(f)

    max_norm_usado = float(rec.get("config", {}).get("max_norm", 0.5))
    diagnostico_treino = instr.resultado(max_norm=max_norm_usado)
    diagnostico_treino["vram_pico_mb"] = rec.get("custo", {}).get("vram_peak_mb")
    diagnostico_treino["disjoint_treino"] = bool(args.disjoint_treino)
    diagnostico_treino["k_antenna_treino"] = int(args.k_antenna)
    diagnostico_treino["k_terrain_treino"] = int(args.k_terrain)
    # Per-epoch time and peak VRAM, as recorded by the frozen trainer (summarised only).
    epocas = rec.get("epocas", []) or []
    tempos_epoca = [e.get("elapsed_s") for e in epocas if e.get("elapsed_s") is not None]
    vrams_epoca = [e.get("vram_peak_mb") for e in epocas if e.get("vram_peak_mb") is not None]
    diagnostico_treino["tempo_por_epoca_s"] = tempos_epoca
    diagnostico_treino["tempo_medio_por_epoca_s"] = (
        float(np.mean(tempos_epoca)) if tempos_epoca else None)
    diagnostico_treino["vram_pico_mb_por_epoca"] = vrams_epoca
    diagnostico_treino["vram_pico_mb_max"] = (max(vrams_epoca) if vrams_epoca else None)
    log(f"diagnostico_treino: passos={diagnostico_treino['n_passos']} "
        f"pulados={diagnostico_treino['n_pulados_total']} "
        f"except_edge_attr={diagnostico_treino['n_except_edge_attr_encoder']}")

    decoder_sha = v3.sha256_do_arquivo(v3.MODELO_V3_DIR / "rf_decoder_v3.py")
    wrapper_sha = v3.sha256_arquivo(Path(__file__).resolve())
    rec["modelo_v3"] = {
        "decoder": "AffineDecoderV3",
        "decoder_sha256": decoder_sha,
        "wrapper_sha256": wrapper_sha,
        "escalas": {
            "path_loss_total_dB": "raw*100+100 (clamp fisico [0,200])",
            "path_loss_vegetation_dB": "raw*25+25 (clamp fisico [0,50])",
            "path_loss_terrain_dB": "raw*15+15 (clamp fisico [0,30])",
            "rssi_dBm": "raw*75-75 (clamp fisico [-150,0])",
            "coverage_prob": "sigmoid(raw) (unico canal comprimido; ver rf_decoder_v3.py)",
        },
        "sem_cadeia": True,
        "transfer_from": "",
        "nota": ("decoder substituido por monkeypatch de gnn_rf_model.PhysicsConstrainedDecoder "
                 "ANTES de model=GNNRFModel(...); nenhuma linha do congelado foi editada."),
    }
    rec["insumos"] = {
        "rf_data_file": str(graph_dir / rf_data_file),
        "graph_dir": str(graph_dir),
        "graph_file": str(graph_dir / graph_file),
        "sha256_rf_data": prov["sha256_rf_data"],
        "sha256_graph": prov["sha256_graph"],
        "manifest": prov["manifest"],
        "manifest_path": prov["manifest_path"],
        "metodo_sha_rf_data": prov["metodo_sha_rf_data"],
        "metodo_sha_graph": prov["metodo_sha_graph"],
        "sha256_rf_data_bate_manifest_v4": prov["sha256_rf_data_bate_manifest_v4"],
        "nota_graph_file": prov["nota_graph_file"],
    }
    rec["diagnostico_treino"] = diagnostico_treino
    with open(run_json_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    log(f"run JSON atualizado com modelo_v3 + diagnostico_treino: {run_json_path}")

    # Test predictions: rebuild the partition with the frozen trainer's functions
    # and the geometry, seeds and split fractions of the run above.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid_km_usado = rec.get("geometria", {}).get("grid_km_usado", 5.0)
    buffer_km_usado = rec.get("geometria", {}).get("buffer_km_usado", 2.0)
    split_frac = tuple(float(v) for v in "0.70,0.15,0.15".split(","))

    ctx = v3.carregar_base_e_particoes(
        mod=mod, graph_dir=graph_dir, rf_data_file=rf_data_file, graph_file=graph_file,
        max_nodes=args.max_nodes, window_anchor="cobertura",
        grid_km=grid_km_usado, buffer_km=buffer_km_usado, split_frac=split_frac,
        split_seed=args.split_seed, smoke_geometria=False,  # geometry already read from the run JSON
        mmap=args.mmap, precisa_arestas_ter_ter=True, log=log,
    )

    # Cross-check: hash of the rebuilt test partition against the hash in the run JSON.
    idx_global_test = ctx.g_idx[ctx.parts_local["test"]]
    hash_reconstruido = mod.sha256_idx(np.sort(idx_global_test.astype(np.int64)))
    hash_gravado = (rec.get("particoes", {}).get("test", {}) or {}).get("idx_sha256_global")
    particao_bate = (hash_gravado is None) or (hash_reconstruido == hash_gravado)
    log(f"hash particao teste reconstruida={hash_reconstruido[:16]}... "
        f"gravado={str(hash_gravado)[:16]}... bate={particao_bate}")

    from gnn_rf_model import GNNRFModel as GNNRFModel_v3
    model = GNNRFModel_v3(
        terrain_dim=int(ctx.x_full.shape[1]), antenna_dim=int(ctx.ant_x.shape[1]),
        hidden_dim=args.hidden_dim, num_layers=4, heads=4, edge_dim=2,
        output_dim=5, dropout=0.1, use_physics_constraints=True).to(device)
    ckpt_path = evid_dir / args.run_label / "checkpoints" / "checkpoint_best.pt"
    if not ckpt_path.exists():
        log(f"AVISO: checkpoint_best.pt nao encontrado em {ckpt_path}; pulando .npz")
    else:
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model_state_dict"])
        eval_bs = args.eval_batch_size or args.batch_size

        # Effective capacity: parameters that receive a gradient on one real
        # training batch (forward + backward on a copy of the model).
        cap = v3.medir_capacidade_efetiva_gnn(
            mod=mod, ctx=ctx, model=model, device=device,
            k_antenna=args.k_antenna, k_terrain=args.k_terrain,
            batch_size=args.batch_size, seed=args.seed_treino, log=log)
        capacidade = {
            "nominal": rec.get("modelo", {}).get("n_params_treinaveis"),
            "efetiva": cap["n_params_efetivos"],
            "parametros_sem_gradiente": sorted(cap["parametros_sem_gradiente"].keys()),
            "n_sem_gradiente": cap["n_params_sem_gradiente"],
            "medido_em": "1 batch real do loader de TREINO desta corrida (forward+backward, copia do modelo)",
        }
        log(f"capacidade GNN: nominal={capacidade['nominal']} efetiva={capacidade['efetiva']} "
            f"sem_gradiente={capacidade['n_sem_gradiente']}")

        # Fixed evaluation rule: validation and test use the complete neighbourhood
        # (k = -1 on the three relations) and disjoint=True, whatever k was used in training.
        K_EVAL = -1
        saida = v3.gerar_predicoes_teste_gnn(
            mod=mod, ctx=ctx, model=model, device=device,
            k_antenna=K_EVAL, k_terrain=K_EVAL,
            eval_batch_size=eval_bs, seed=args.seed_treino, log=log,
            particao="test", disjoint=True, medir_diagnostico=True)
        saida["particao_hash_bate_com_run_json"] = bool(particao_bate)
        npz_path = evid_dir / args.run_label / f"predicoes_{args.run_label}.npz"
        np.savez(npz_path, idx_global=saida["idx_global"], target=saida["target"],
                 pred=saida["pred"], sentinela=saida["sentinela"])
        log(f"predicoes .npz gravadas: {npz_path} ({saida['idx_global'].shape[0]} nos de teste)")

        # Output diagnostics per channel (affine vs clamped output) on validation and test.
        saida_val = v3.gerar_predicoes_teste_gnn(
            mod=mod, ctx=ctx, model=model, device=device,
            k_antenna=K_EVAL, k_terrain=K_EVAL,
            eval_batch_size=eval_bs, seed=args.seed_treino, log=log,
            particao="val", disjoint=True, medir_diagnostico=True)
        diagnostico_saida = {
            "val": v3.diagnostico_saida_canais(saida_val["pred_afim"], saida_val["pred"], saida_val["target"], sentinela=saida_val["sentinela"]),
            "test": v3.diagnostico_saida_canais(saida["pred_afim"], saida["pred"], saida["target"], sentinela=saida["sentinela"]),
        }

        # Invariance of the complete-neighbourhood evaluation to the evaluation
        # batch size and to the order of the seed nodes, against a numerical floor
        # measured in the same run.
        bs_alt = max(1024, eval_bs // 4) if eval_bs > 1024 else eval_bs * 2 + 1
        teste_invariancia = v3.testar_invariancia_avaliacao_completa(
            mod=mod, ctx=ctx, model=model, device=device,
            seed=args.seed_treino, eval_batch_size_a=eval_bs,
            eval_batch_size_b=bs_alt, particao="test", log=log)
        log(f"teste invariancia (vizinhanca completa): bs {eval_bs} vs {bs_alt} -- "
            f"diff_max_fisico={teste_invariancia['diff_max_fisico_entre_todas_combinacoes']} "
            f"piso={teste_invariancia['piso_numerico_medido_mesma_ordem_mesma_bs']['diff_max_fisico']} "
            f"(passou={teste_invariancia['passou_criterio']})")

        # Evidence that the evaluation draws nothing at random: RNG state unchanged,
        # sampled edges equal to the sum of degrees, degrees per relation.
        diag_val = saida_val.get("diagnostico", {})
        diag_test = saida.get("diagnostico", {})
        diagnostico_avaliacao = {
            "regra": "num_neighbors=-1 nas 3 relacoes (AT,TT,TA), disjoint=True, em val e teste",
            "rng_identico_val": diag_val.get("rng_identico"),
            "rng_identico_test": diag_test.get("rng_identico"),
            "arestas_bate_com_soma_graus_val": diag_val.get("arestas_bate_com_soma_graus"),
            "arestas_bate_com_soma_graus_test": diag_test.get("arestas_bate_com_soma_graus"),
            "arestas_amostradas_por_relacao_val": diag_val.get("arestas_amostradas_por_relacao"),
            "arestas_amostradas_por_relacao_test": diag_test.get("arestas_amostradas_por_relacao"),
            "graus_por_relacao_val": diag_val.get("graus_por_relacao"),
            "graus_por_relacao_test": diag_test.get("graus_por_relacao"),
            "teste_invariancia": teste_invariancia,
        }
        log(f"diagnostico_avaliacao: rng_identico val={diagnostico_avaliacao['rng_identico_val']} "
            f"test={diagnostico_avaliacao['rng_identico_test']} | "
            f"arestas=soma_graus val={diagnostico_avaliacao['arestas_bate_com_soma_graus_val']} "
            f"test={diagnostico_avaliacao['arestas_bate_com_soma_graus_test']}")

        rec["capacidade"] = capacidade
        rec["diagnostico_saida"] = diagnostico_saida
        rec["diagnostico_avaliacao"] = diagnostico_avaliacao
        with open(run_json_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        log(f"run JSON atualizado com capacidade + diagnostico_saida + diagnostico_avaliacao: {run_json_path}")

    log("train_gnn_v3 concluido.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
