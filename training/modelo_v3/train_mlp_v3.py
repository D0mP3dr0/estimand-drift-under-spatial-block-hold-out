#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Third-campaign MLP (graph-free control) trainer with the affine decoder.

Counterpart of `train_gnn_v3.py` for the frozen graph-free control
`training/frozen/train_mlp_c0_spatial.py`. The frozen script imports
`PhysicsConstrainedDecoder` from `rf_decoder` inside its `main()`, so the
decoder is replaced by overriding that attribute of the `rf_decoder` module
before `main()` is called (`v3_common.patch_decoder_mlp`). The MLP layer widths
are set through `v3_common.patch_larguras_mlp`: by default the widths matched to
the effective (gradient-receiving) parameter count of the GNN; with
`--larguras-antigas`, the earlier widths matched to its nominal count.
Training always starts from scratch (no `--transfer-from`).

Inputs: `transfer_dataset_<city>_v19_<Q>_enriched_cftudo.pt` in `--graph-dir`,
whose SHA-256 must match the v4 manifest (group `tensores_cftudo`), otherwise
the run aborts with return code 3. The `<city>_v19_<Q>_gpu.pt` graph is hashed
for the provenance record only; the MLP does not use terrain-terrain edges.
Outputs (in `--evid-dir/<run-label>/`): the frozen script's run JSON, extended
with `modelo_v3`, `insumos`, `diagnostico_treino`, `capacidade`,
`diagnostico_saida` and `diagnostico_avaliacao`; and `predicoes_<label>.npz`
with the test-partition predictions (`idx_global`, `target`, `pred`, `sentinela`).

Usage: train_mlp_v3.py --cidade bauru --quadrante Q2 --evid-dir OUT --run-label LABEL
       [--seed-treino 42] [--split-seed 42] [--epochs 8] [--larguras-antigas]
Seeds: `--seed-treino` is passed as the frozen script's `--seed`; `--split-seed`
fixes the split draw.
"""
from __future__ import annotations

import argparse
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
    p.add_argument("--batch-size", type=int, default=24576)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-batch-size", type=int, default=0)
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--larguras-antigas", action="store_true",
                    help="Use the earlier widths matched to the NOMINAL GNN parameter count "
                         "(712,712,720,688,256; N_PARAMS_ALVO_GNN=1,812,515) instead of the "
                         "widths matched to the EFFECTIVE GNN count (654,654,600,636,256; "
                         "N_PARAMS_ALVO_GNN=1,484,067); sensitivity arm of 10 extra runs.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    def log(msg):
        print(f"[train_mlp_v3 +{time.perf_counter()-t0:.1f}s] {msg}", flush=True)

    rf_data_file = f"transfer_dataset_{args.cidade}_v19_{args.quadrante}_enriched_cftudo.pt"
    graph_file = f"{args.cidade}_v19_{args.quadrante}_gpu.pt"
    graph_dir = Path(args.graph_dir)
    if not (graph_dir / rf_data_file).exists():
        raise FileNotFoundError(f"Dado esperado nao encontrado: {graph_dir / rf_data_file}")

    # Provenance gate on the RF data file before any GPU work; the graph file is
    # hashed only for the record (falls back to the RF data file if absent).
    graph_path_para_sha = graph_dir / graph_file
    prov = v3.verificar_proveniencia_dataset(
        graph_dir / rf_data_file,
        graph_path_para_sha if graph_path_para_sha.exists() else (graph_dir / rf_data_file))
    log(f"proveniencia: sha256_rf_data={prov['sha256_rf_data'][:16]}... "
        f"metodo={prov['metodo_sha_rf_data']} "
        f"encontrado_no_manifest_v4={prov['encontrado_no_manifest_v4']} "
        f"bate_manifest_v4={prov['sha256_rf_data_bate_manifest_v4']}")
    if not prov["sha256_rf_data_bate_manifest_v4"]:
        log(f"ABORTANDO (rc=3): sha256 do rf_data_file NAO bate com o manifest v4 "
            f"(grupo tensores_cftudo). arquivo={graph_dir / rf_data_file} "
            f"sha_calculado={prov['sha256_rf_data']} sha_manifest={prov['sha256_manifest_v4']}")
        return 3

    # Must run before mod.main(), which imports the decoder class at call time.
    v3.patch_decoder_mlp()
    import rf_decoder
    assert rf_decoder.PhysicsConstrainedDecoder is v3.AffineDecoderV3, (
        "monkeypatch do decoder MLP nao pegou -- abortando antes de gastar GPU")

    mod = v3.carregar_modulo_congelado(v3.FROZEN_MLP_SCRIPT, "train_mlp_c0_spatial_v3")

    # MLP_LARGURAS and N_PARAMS_ALVO_GNN are module globals read by main() at each call.
    if args.larguras_antigas:
        larguras_usadas = v3.MLP_LARGURAS_ANTIGAS
        alvo_nominal_usado = v3.N_PARAMS_ALVO_NOMINAL_ANTIGO
    else:
        larguras_usadas = v3.MLP_LARGURAS_NOVAS
        alvo_nominal_usado = v3.N_PARAMS_ALVO_NOMINAL_NOVO
    v3.patch_larguras_mlp(mod, larguras_usadas, alvo_nominal_usado)
    log(f"MLP_LARGURAS={larguras_usadas} N_PARAMS_ALVO_GNN={alvo_nominal_usado} "
        f"(larguras_antigas={args.larguras_antigas})")

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
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        # No --transfer-from: the frozen script's default "" means training from scratch.
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
    log(f"diagnostico_treino: passos={diagnostico_treino['n_passos']} "
        f"pulados={diagnostico_treino['n_pulados_total']}")

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
        "nota": ("decoder substituido por monkeypatch de rf_decoder.PhysicsConstrainedDecoder "
                 "ANTES de mod.main() rodar `from rf_decoder import PhysicsConstrainedDecoder`; "
                 "nenhuma linha do congelado foi editada."),
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
        "nota_graph_file": (prov["nota_graph_file"] + " MLP nao usa arestas terreno-terreno "
                             "(nao consome graph_file no treino/avaliacao); sha256_graph gravado "
                             "so por paridade de proveniencia com o wrapper GNN."),
    }
    rec["diagnostico_treino"] = diagnostico_treino
    with open(run_json_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    log(f"run JSON atualizado com modelo_v3 + diagnostico_treino: {run_json_path}")

    # Test predictions: rebuild the partition with the frozen script's functions
    # and the geometry, seeds and split fractions of the run above.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid_km_usado = rec.get("geometria", {}).get("grid_km_usado", 5.0)
    buffer_km_usado = rec.get("geometria", {}).get("buffer_km_usado", 2.0)
    split_frac = tuple(float(v) for v in "0.70,0.15,0.15".split(","))

    ctx = v3.carregar_base_e_particoes(
        mod=mod, graph_dir=graph_dir, rf_data_file=rf_data_file, graph_file=graph_file,
        max_nodes=args.max_nodes, window_anchor="cobertura",
        grid_km=grid_km_usado, buffer_km=buffer_km_usado, split_frac=split_frac,
        split_seed=args.split_seed, smoke_geometria=False,
        mmap=args.mmap, precisa_arestas_ter_ter=False, log=log,
    )

    # Cross-check: hash of the rebuilt test partition against the hash in the run JSON.
    idx_global_test = ctx.g_idx[ctx.parts_local["test"]]
    hash_reconstruido = mod.sha256_idx(np.sort(idx_global_test.astype(np.int64)))
    hash_gravado = (rec.get("particoes", {}).get("test", {}) or {}).get("idx_sha256_global")
    particao_bate = (hash_gravado is None) or (hash_reconstruido == hash_gravado)
    log(f"hash particao teste reconstruida={hash_reconstruido[:16]}... "
        f"gravado={str(hash_gravado)[:16]}... bate={particao_bate}")

    from rf_decoder import PhysicsConstrainedDecoder as DecoderPatched
    MLPRFModel = getattr(mod, "MLPRFModel")
    MLP_LARGURAS = getattr(mod, "MLP_LARGURAS")
    model = MLPRFModel(DecoderPatched, terrain_dim=int(ctx.x_full.shape[1]),
                       larguras=MLP_LARGURAS, dropout=0.1).to(device)

    ckpt_path = evid_dir / args.run_label / "checkpoints" / "checkpoint_best.pt"
    if not ckpt_path.exists():
        log(f"AVISO: checkpoint_best.pt nao encontrado em {ckpt_path}; pulando .npz")
    else:
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model_state_dict"])
        eval_bs = args.eval_batch_size or args.batch_size

        # Effective capacity: parameters that receive a gradient on one real
        # training batch (forward + backward on a copy of the model).
        cap = v3.medir_capacidade_efetiva_mlp(
            ctx=ctx, model=model, device=device, batch_size=args.batch_size,
            seed=args.seed_treino)
        capacidade = {
            "nominal": rec.get("modelo", {}).get("n_params_treinaveis"),
            "efetiva": cap["n_params_efetivos"],
            "larguras": list(MLP_LARGURAS),
            "alvo_gnn_efetivo": v3.N_PARAMS_ALVO_EFETIVO_GNN,
            "alvo_gnn_efetivo_fonte": v3.FONTE_LARGURAS,
            "larguras_antigas": bool(args.larguras_antigas),
            "parametros_sem_gradiente": sorted(cap["parametros_sem_gradiente"].keys()),
            "n_sem_gradiente": cap["n_params_sem_gradiente"],
            "medido_em": "1 batch real da particao de TREINO desta corrida (forward+backward, copia do modelo)",
        }
        log(f"capacidade MLP: nominal={capacidade['nominal']} efetiva={capacidade['efetiva']} "
            f"alvo_gnn_efetivo={capacidade['alvo_gnn_efetivo']} "
            f"delta={capacidade['efetiva'] - capacidade['alvo_gnn_efetivo']}")

        saida = v3.gerar_predicoes_teste_mlp(
            mod=mod, ctx=ctx, model=model, device=device,
            eval_batch_size=eval_bs, seed=args.seed_treino, log=log, particao="test")
        saida["particao_hash_bate_com_run_json"] = bool(particao_bate)
        npz_path = evid_dir / args.run_label / f"predicoes_{args.run_label}.npz"
        np.savez(npz_path, idx_global=saida["idx_global"], target=saida["target"],
                 pred=saida["pred"], sentinela=saida["sentinela"])
        log(f"predicoes .npz gravadas: {npz_path} ({saida['idx_global'].shape[0]} nos de teste)")

        # Output diagnostics per channel (affine vs clamped output) on validation and test.
        saida_val = v3.gerar_predicoes_teste_mlp(
            mod=mod, ctx=ctx, model=model, device=device,
            eval_batch_size=eval_bs, seed=args.seed_treino, log=log, particao="val")
        diagnostico_saida = {
            "val": v3.diagnostico_saida_canais(saida_val["pred_afim"], saida_val["pred"], saida_val["target"], sentinela=saida_val["sentinela"]),
            "test": v3.diagnostico_saida_canais(saida["pred_afim"], saida["pred"], saida["target"], sentinela=saida["sentinela"]),
        }

        rec["capacidade"] = capacidade
        rec["diagnostico_saida"] = diagnostico_saida
        rec["diagnostico_avaliacao"] = {
            "aplicavel": False,
            "motivo": "MLP nao usa NeighborLoader/grafo; nao ha vizinhanca a amostrar na avaliacao.",
        }
        with open(run_json_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        log(f"run JSON atualizado com capacidade + diagnostico_saida + diagnostico_avaliacao: {run_json_path}")

    log("train_mlp_v3 concluido.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
