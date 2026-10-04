"""Data sheet behind the hyper-parameter table of the appendix (GNN versus MLP, campaigns A3 and A4).

Every field of the table is taken from (i) the run records `run_*.json` of the 130 runs (60 GNN
and 60 MLP from A3 and A4, plus 10 MLP runs of the A4 width-sensitivity arm), and/or (ii) a
specific line of code, cited as path:line. `unico` requires that a field is identical across
all runs of an arm; `faixa` reports median, minimum and maximum when it varies. `ref` reads each
cited line and asserts that it contains the expected text, so a stale citation aborts the
script before anything is written. Auxiliary checks confirm that the prediction loss equals the
arithmetic mean of the five per-channel losses (uniform channel weights) and record the largest
logged values of the FSPL constraint and of the shadowing x NDVI penalty.

Inputs: the run folders of A3 and A4 under the GPU results directory; the trainers
(training/frozen/train_gnn_c0_spatial.py, train_mlp_c0_spatial.py), training/models/gnn_rf_encoder.py,
training/models/physics_loss.py, training/modelo_v3/v3_common.py and train_gnn_v3.py,
analysis/v3_2.1_3.1_deriva_calibracao.py, and the frozen baseline script
`baselines_v2_por_particao.py` (not distributed). All are read at the absolute paths of the
original run; the cited line numbers refer to those files and may differ in the copies here.
Outputs: b7_2_folha_hiperparametros.json (fields, source SHA-256 digests, checks) and .csv in OUT.
Usage: python b7_2_folha_hiperparametros.py
"""
import csv, glob, hashlib, json, os, statistics as st
from pathlib import Path

V3 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
OUT = V3 / "fase4" / "B7.2_hiperparametros"
EV = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO")
TG = EV / "scripts" / "train_gnn_c0_spatial.py"
TM = EV / "scripts" / "train_mlp_c0_spatial.py"
BL = EV / "dados" / "scripts_congelados" / "baselines_v2_por_particao.py"
G2 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2/02_models")
ENC, LOSS = G2 / "gnn_rf_encoder.py", G2 / "physics_loss.py"
M3 = V3 / "gpu" / "modelo_v3"
VC, TGV = M3 / "v3_common.py", M3 / "train_gnn_v3.py"
D21 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts/v3_2.1_3.1_deriva_calibracao.py")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ref(path, linha, trecho):
    """Assert that line `linha` (1-based) of `path` contains `trecho`; return the citation 'path:line'."""
    L = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    assert trecho in L[linha - 1], f"{path}:{linha} nao contem {trecho!r}: {L[linha-1]!r}"
    return f"{path}:{linha}"


def ler(pat):
    return [json.load(open(f)) for f in sorted(glob.glob(str(V3 / "gpu" / pat)))]


gnn = ler("A3/gnn_v3_a3_*/run_*.json") + ler("A4/gnn_v3_a4_*/run_*.json")
mlp = ler("A3/mlp_v3_a3_*/run_*.json") + ler("A4/mlp_v3_a4_*/run_*.json")
a4s = ler("A4/mlp_v3_a4s_*/run_*.json")
# Expected counts: A3 (40) + A4 main arm (20) per model, and 10 sensitivity-arm MLP runs.
assert (len(gnn), len(mlp), len(a4s)) == (60, 60, 10)


def unico(runs, f):
    """Value of f(run), asserted to be the same for every run of the list."""
    vals = {json.dumps(f(r), sort_keys=True) for r in runs}
    assert len(vals) == 1, (f, vals)
    return json.loads(vals.pop())


def faixa(runs, f):
    xs = [f(r) for r in runs]
    return {"mediana": st.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


C = lambda k: (lambda r: r["config"][k])
linhas = []


def add(grupo, campo, gnn_v, mlp_v, fonte, igual=None, nota=""):
    """Append one table row; `igual` (same value in both arms) is computed unless given."""
    if igual is None:
        igual = json.dumps(gnn_v, sort_keys=True) == json.dumps(mlp_v, sort_keys=True)
    linhas.append({"grupo": grupo, "campo": campo, "gnn": gnn_v, "mlp": mlp_v,
                   "igual_nos_dois_bracos": igual, "fonte": fonte, "nota": nota})


add("arquitetura", "classe", unico(gnn, lambda r: r["modelo"]["classe"]), unico(mlp, lambda r: r["modelo"]["classe"]),
    "run JSON modelo.classe", igual=False)
add("arquitetura", "operador antena->terreno", "GATv2Conv(hidden->hidden/heads, heads=4, concat, edge_dim=2, dropout, add_self_loops=False)",
    "nenhum (sem arestas)", ref(ENC, 120, "GATv2Conv("), igual=False)
add("arquitetura", "operador terreno->terreno (Moore + self-loop)", "GATv2Conv(..., add_self_loops=True)",
    "nenhum (sem arestas)", ref(ENC, 130, "GATv2Conv(") + " ; " + ref(ENC, 137, "add_self_loops=True"), igual=False)
add("arquitetura", "operador terreno->antena", "SAGEConv(aggr='mean') -- sem gradiente (relacao nao amostrada a partir de sementes de terreno)",
    "nenhum", ref(ENC, 140, "SAGEConv(") + " ; " + ref(ENC, 143, "aggr='mean'"), igual=False,
    nota="run JSON diagnostico_avaliacao.arestas_amostradas_por_relacao_*.TA = 0")
add("arquitetura", "agregacao entre relacoes", "soma", "-", ref(ENC, 145, "aggr='sum'"), igual=False)
add("arquitetura", "por camada (GNN) / bloco (MLP)", "LeakyReLU(0.2) + dropout + residual + LayerNorm",
    "Linear + BatchNorm1d + LeakyReLU(0.2) (+ Dropout, exceto ultimo)",
    ref(ENC, 217, "leaky_relu(out_feat, 0.2)") + " ; " + ref(TM, 712, "nn.BatchNorm1d(n), nn.LeakyReLU(0.2)"), igual=False)
add("arquitetura", "camadas de passagem de mensagem / camadas do encoder",
    unico(gnn, lambda r: r["modelo"]["num_layers"]), len(unico(mlp, lambda r: r["modelo"]["larguras_encoder"])),
    ref(TG, 1233, "num_layers=4, heads=4") + " ; run JSON modelo.num_layers / modelo.larguras_encoder", igual=False)
add("arquitetura", "cabecas de atencao", unico(gnn, lambda r: r["modelo"]["heads"]), None, "run JSON modelo.heads", igual=False)
add("arquitetura", "largura oculta / larguras do encoder", unico(gnn, lambda r: r["modelo"]["hidden_dim"]),
    unico(mlp, lambda r: r["modelo"]["larguras_encoder"]), "run JSON modelo.hidden_dim ; modelo.larguras_encoder ; " + ref(VC, 685, "(654, 654, 600, 636, 256)"), igual=False)
add("arquitetura", "entradas por no de terreno", unico(gnn, lambda r: r["modelo"]["terrain_dim"]), unico(mlp, lambda r: r["modelo"]["terrain_dim"]),
    "run JSON modelo.terrain_dim")
add("arquitetura", "entradas por transmissor", unico(gnn, lambda r: r["modelo"]["antenna_dim"]), "nao usado",
    "run JSON modelo.antenna_dim / modelo.antenna_dim_nao_usado", igual=False)
add("arquitetura", "saidas", unico(gnn, lambda r: r["modelo"]["output_dim"]), unico(mlp, lambda r: r["modelo"]["output_dim"]),
    "run JSON modelo.output_dim")
add("arquitetura", "decodificador", unico(gnn, lambda r: r["modelo_v3"]["escalas"]), unico(mlp, lambda r: r["modelo_v3"]["escalas"]),
    "run JSON modelo_v3.escalas (AffineDecoderV3, sha " + unico(gnn, lambda r: r["modelo_v3"]["decoder_sha256"])[:12] + ")")
add("arquitetura", "dropout", unico(gnn, lambda r: r["modelo"]["dropout"]), unico(mlp, lambda r: r["modelo"]["dropout"]),
    "run JSON modelo.dropout ; " + ref(TG, 1234, "dropout=0.1"))
add("capacidade", "parametros nominais", unico(gnn, lambda r: r["capacidade"]["nominal"]), unico(mlp, lambda r: r["capacidade"]["nominal"]),
    "run JSON capacidade.nominal", igual=False)
add("capacidade", "parametros efetivos (recebem gradiente)", unico(gnn, lambda r: r["capacidade"]["efetiva"]), unico(mlp, lambda r: r["capacidade"]["efetiva"]),
    "run JSON capacidade.efetiva")
add("capacidade", "braco de sensibilidade (10 corridas, larguras nominais)", None,
    {"larguras": unico(a4s, lambda r: r["capacidade"]["larguras"]), "nominal": unico(a4s, lambda r: r["capacidade"]["nominal"]),
     "efetiva": unico(a4s, lambda r: r["capacidade"]["efetiva"])}, "run JSON mlp_v3_a4s_* capacidade.*", igual=False)

add("vizinhanca", "amostragem no treino", {"num_neighbors": "[-1] por relacao (lista de tamanho 1 = UM salto, vizinhanca completa)",
    "k_antenna": unico(gnn, C("k_antenna")), "k_terrain": unico(gnn, C("k_terrain")),
    "disjoint_treino": unico(gnn, lambda r: r["diagnostico_treino"].get("disjoint_treino", False))},
    "lotes de nos de terreno (sem grafo)", ref(TG, 1171, "nn_kw = {ET_AT: [args.k_antenna]") + " ; run JSON config.k_*", igual=False,
    nota="4 camadas sobre subgrafo de 1 salto: campo receptivo = 8 vizinhos de Moore + ate 5 transmissores")
add("vizinhanca", "amostragem na avaliacao (val/teste)", unico(gnn, lambda r: r["diagnostico_avaliacao"]["regra"]), "nao se aplica",
    ref(VC, 478, "nn_kw = {mod.ET_AT: [k_antenna]") + " ; " + ref(VC, 484, "disjoint=disjoint") + " ; run JSON diagnostico_avaliacao.regra", igual=False)

add("perda", "termo de predicao", "Huber(delta=5 dB) por canal, nos 5 canais, media sobre os nos semente (sentinelas incluidos)",
    "idem", ref(LOSS, 99, "F.huber_loss(pred, true, delta=self.huber_delta"), igual=True)
add("perda", "pesos dos canais", "softmax(log_weights), log_weights=zeros(5) fora do otimizador => 1/5 cada", "idem",
    ref(LOSS, 53, "nn.Parameter(torch.zeros(5))") + " ; " + ref(LOSS, 69, "F.softmax(self.log_weights") + " ; " +
    ref(TG, 1245, "AdamW(model.parameters()") + " ; " + ref(TM, 1381, "AdamW(model.parameters()"), igual=True,
    nota="conferido: prediction_loss = media aritmetica das 5 perdas por canal (ver verificacao_pesos_uniformes)")
add("perda", "penalidade gradiente de distancia (peso)", unico(gnn, lambda r: r["loss"]["distance_gradient_weight"]),
    unico(mlp, lambda r: r["loss"]["distance_gradient_weight"]), "run JSON loss.distance_gradient_weight")
add("perda", "penalidade gradiente de distancia (expressao)",
    "mean_{(i,j): d_i<d_j} ReLU(rssi_hat_j - rssi_hat_i + 1 dBm), K=min(512, N//2) pares sorteados no lote", "idem",
    ref(LOSS, 237, "K = min(self.distance_gradient_n_pairs, N // 2)") + " ; " + ref(LOSS, 258, "torch.relu(rssi_j[closer] - rssi_i[closer] + margin)"), igual=True)
add("perda", "penalidade de variancia (peso)", unico(gnn, lambda r: r["loss"]["variance_weight"]), unico(mlp, lambda r: r["loss"]["variance_weight"]),
    "run JSON loss.variance_weight")
add("perda", "penalidade de variancia (expressao)", "ReLU(std(y_rssi) - std(rssi_hat))^2 no lote (alvo sem gradiente)", "idem",
    ref(LOSS, 286, "torch.relu(tgt_std - pred_std) ** 2"), igual=True)
add("perda", "penalidade sombra x NDVI (peso)", unico(gnn, lambda r: r["loss"]["shadowing_ndvi_weight"]), unico(mlp, lambda r: r["loss"]["shadowing_ndvi_weight"]),
    "run JSON loss.shadowing_ndvi_weight")
add("perda", "penalidade sombra x NDVI (expressao)", "ReLU(0.15 - corr(PL_veg_hat, NDVI)) no lote, NDVI = coluna 12 (sem gradiente)", "idem",
    ref(LOSS, 319, "torch.relu(self.shadowing_ndvi_min_corr - corr)") + " ; " + ref(TG, 212, "COL_NDVI"), igual=True)
add("perda", "restricao FSPL (nao citada no texto)", "0.05 * mean ReLU(FSPL_900MHz(d) - PL_hat_total)", "idem",
    ref(LOSS, 131, "self.fspl_weight * fspl_constraint") + " ; " + ref(LOSS, 181, "F.relu(fspl - pred_total_loss)"), igual=True,
    nota="ver valor_maximo_constraint_fspl_registrado: se 0 em todas as epocas, o termo nunca atuou")

add("otimizacao", "otimizador", "AdamW, weight_decay=1e-5", "idem", ref(TG, 1245, "weight_decay=1e-5") + " ; " + ref(TM, 1381, "weight_decay=1e-5"), igual=True)
add("otimizacao", "taxa inicial", unico(gnn, C("lr")), unico(mlp, C("lr")), "run JSON config.lr")
add("otimizacao", "agenda", {"classe": "CosineAnnealingWarmRestarts", "T_0_epocas": unico(gnn, C("cosine_t0")), "T_mult": 2, "eta_min": 1e-6,
    "passo": "por epoca", "reinicio_dentro_de_8_epocas": False},
    {"classe": "CosineAnnealingWarmRestarts", "T_0_epocas": unico(mlp, C("cosine_t0")), "T_mult": 2, "eta_min": 1e-6,
     "passo": "por epoca", "reinicio_dentro_de_8_epocas": False},
    ref(TG, 1247, "CosineAnnealingWarmRestarts(optimizer, T_0=args.cosine_t0") + " ; " + ref(TG, 1373, "scheduler.step()") + " ; " +
    ref(TM, 1383, "CosineAnnealingWarmRestarts(optimizer, T_0=args.cosine_t0"))
add("otimizacao", "corte de gradiente (norma)", unico(gnn, C("max_norm")), unico(mlp, C("max_norm")),
    "run JSON config.max_norm ; " + ref(TG, 1358, "clip_grad_norm_") + " ; " + ref(TM, 1494, "clip_grad_norm_"))
add("otimizacao", "fracao de passos cortados (min nas corridas)", min(r["diagnostico_treino"]["fracao_passos_clipados"] for r in gnn),
    min(r["diagnostico_treino"]["fracao_passos_clipados"] for r in mlp), "run JSON diagnostico_treino.fracao_passos_clipados")
add("otimizacao", "precisao mista", "AMP (autocast + GradScaler) em CUDA", "idem", ref(TG, 1279, "GradScaler(\"cuda\"") + " ; " + ref(TM, 1415, "GradScaler(\"cuda\""), igual=True)
add("otimizacao", "lote (nos semente)", unico(gnn, C("batch_size")), unico(mlp, C("batch_size")), "run JSON config.batch_size")
add("otimizacao", "epocas", unico(gnn, C("epochs")), unico(mlp, C("epochs")), "run JSON config.epochs ; custo.n_epocas_rodadas")
add("otimizacao", "parada antecipada", unico(gnn, C("early_stopping_patience")), unico(mlp, C("early_stopping_patience")), "run JSON config.early_stopping_patience (0 = desligada)")
add("otimizacao", "passos de gradiente por corrida", faixa(gnn, lambda r: r["diagnostico_treino"]["n_passos"]),
    faixa(mlp, lambda r: r["diagnostico_treino"]["n_passos"]), "run JSON diagnostico_treino.n_passos", igual=False,
    nota="faixas sobrepostas; diferem por celula/sorteio, nao por braco")
add("selecao", "regra da epoca retida", unico(gnn, lambda r: r["selecao"]["criterio"]), unico(mlp, lambda r: r["selecao"]["criterio"]),
    "run JSON selecao.criterio ; " + ref(TG, 1503, "return float(m_val[\"mae_rssi_db\"])"))
add("selecao", "teste usado na selecao", unico(gnn, lambda r: r["selecao"]["test_usado_na_selecao"]), unico(mlp, lambda r: r["selecao"]["test_usado_na_selecao"]),
    "run JSON selecao.test_usado_na_selecao ; " + ref(TG, 1502, "assert \"test\" not in m_val"))
add("selecao", "melhor epoca (mediana, min, max)", faixa(gnn, lambda r: r["selecao"]["melhor_epoca"]), faixa(mlp, lambda r: r["selecao"]["melhor_epoca"]),
    "run JSON selecao.melhor_epoca", igual=False)

seeds = lambda runs: sorted({(r["seed"], r["split_seed"]) for r in runs})
add("sementes", "pares (seed de treino, split seed)", seeds(gnn), seeds(mlp), "run JSON seed, split_seed",
    nota="A3: treino 42-46 x split 42; A4: treino 42 x split 101-105")
add("sementes", "determinismo declarado", unico(gnn, lambda r: [r["ambiente"]["cudnn_deterministic"], r["ambiente"]["cudnn_benchmark"]]),
    unico(mlp, lambda r: [r["ambiente"]["cudnn_deterministic"], r["ambiente"]["cudnn_benchmark"]]),
    "run JSON ambiente.cudnn_* ; " + ref(TG, 827, "cudnn.deterministic = True"),
    nota="GNN nao determinista mesmo assim (A2c: ate 0,132 dB validos / 0,117 dB sentinela); MLP determinista")

add("custo", "hardware", unico(gnn, lambda r: [r["ambiente"]["gpu"], r["ambiente"]["torch"], r["ambiente"]["cuda"]]),
    unico(mlp, lambda r: [r["ambiente"]["gpu"], r["ambiente"]["torch"], r["ambiente"]["cuda"]]), "run JSON ambiente.*")
add("custo", "tempo por corrida (s)", faixa(gnn, lambda r: r["custo"]["tempo_total_s"]), faixa(mlp, lambda r: r["custo"]["tempo_total_s"]),
    "run JSON custo.tempo_total_s", igual=False)
add("custo", "pico de VRAM (MB)", faixa(gnn, lambda r: r["custo"]["vram_peak_mb"]), faixa(mlp, lambda r: r["custo"]["vram_peak_mb"]),
    "run JSON custo.vram_peak_mb", igual=False)

add("baselines", "frequencia", 900.0, 900.0, ref(D21, 77, "FREQ_MHZ = 900.0") + " ; run JSON baselines_analiticos.train.freq_mhz", igual=True)
add("baselines", "h_tx, h_rx (m) Hata rural e COST-231", {"h_tx": 30.0, "h_rx": 1.5}, {"h_tx": 30.0, "h_rx": 1.5},
    ref(BL, 268, "def okumura_hata_rural") + " ; " + ref(BL, 269, "h_tx: float = 30.0, h_rx: float = 1.5") + " ; " +
    ref(BL, 277, "h_tx: float = 30.0, h_rx: float = 1.5"), igual=True,
    nota="valores default; v3_2.1_3.1_deriva_calibracao.py chama sem passar alturas")
add("baselines", "variante COST-231", "suburban (C_m = 0)", "idem",
    ref(D21, 99, "environment=\"suburban\"") + " ; " + ref(BL, 282, "C_m = 3.0 if environment == \"urban\" else 0.0"), igual=True)
add("baselines", "distancia", "distancia ao transmissor mais proximo, piso 1 m", "idem", ref(BL, 247, "np.maximum(distance_m / 1000.0, 0.001)"), igual=True)
add("baselines", "calibracao", "offset = mediana(RSSI + PL_modelo) no treino ((a) todos os nos; (b) so validos)", "idem",
    ref(D21, 298, "p_tx_eff = float(np.median(rssi_tr + pl_pred_tr))"), igual=True)

r0 = gnn[0]["epocas"][0]["loss_componentes"]
comp = [r0[f"loss_{k}"] for k in ["path_loss_total", "path_loss_vegetation", "path_loss_terrain", "rssi", "coverage"]]
verif = {"verificacao_pesos_uniformes": {"corrida": gnn[0]["run_label"], "media_5_canais": sum(comp) / 5, "prediction_loss": r0["prediction_loss"]},
         "valor_maximo_constraint_fspl_registrado": max(e["loss_componentes"].get("constraint_fspl", 0.0) for r in gnn + mlp for e in r["epocas"]),
         "valor_maximo_shadowing_ndvi_penalty_registrado": max(e["loss_componentes"].get("shadowing_ndvi_penalty", 0.0) for r in gnn + mlp for e in r["epocas"])}
assert abs(verif["verificacao_pesos_uniformes"]["media_5_canais"] - verif["verificacao_pesos_uniformes"]["prediction_loss"]) < 1e-3

out = {"script": __file__, "script_sha256": sha(__file__), "natureza": "hyper-parameter data sheet B7.2, compiled from run records and code lines",
       "n_corridas": {"gnn": len(gnn), "mlp": len(mlp), "mlp_sensibilidade": len(a4s)},
       "fontes_sha256": {str(p): sha(p) for p in [TG, TM, BL, ENC, LOSS, VC, TGV, D21]},
       "campos": linhas, **verif}
json.dump(out, open(OUT / "b7_2_folha_hiperparametros.json", "w"), indent=1, ensure_ascii=False)
with open(OUT / "b7_2_folha_hiperparametros.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["grupo", "campo", "gnn", "mlp", "igual_nos_dois_bracos", "fonte", "nota"])
    for l in linhas:
        w.writerow([l["grupo"], l["campo"], json.dumps(l["gnn"], ensure_ascii=False), json.dumps(l["mlp"], ensure_ascii=False),
                    l["igual_nos_dois_bracos"], l["fonte"], l["nota"]])
print(len(linhas), "campos;", json.dumps(verif))
