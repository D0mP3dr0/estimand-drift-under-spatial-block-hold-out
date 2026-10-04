# Estimand Drift under Spatial Block Hold-Out: reproducibility package

Reproducibility package for L. F. C. Seelig and R. M. Salles, *Estimand Drift
under Spatial Block Hold-Out*, a manuscript submitted to MDPI Mathematics and
currently under review. The study asks what a buffered spatial block hold-out
actually estimates when the test set is made of the nodes that survive the
buffer. It looks at which population the reported error covers, how that
population differs from the domain, and how the error moves from one split
draw to the next. The case study uses sixteen simulated city x quadrant cells
of a radio-frequency field reconstruction task (Bauru, Campinas, Lins and
Sorocaba; quadrants Q1 to Q4). In these cells, nodes beyond transmitter reach
carry an imputed sentinel target and the other nodes are valid. The repository
holds the following:

- the code that partitions and trims the cells at block sizes g = 10 km and
  g = 5 km;
- the computation of the per-node inclusion probabilities and the split draws;
- the numerical tests;
- the training and evaluation code of the two regressors (a heterogeneous graph
  neural network, GNN, and a graph-free multilayer perceptron, MLP);
- the per-run records of the trained models;
- the aggregation scripts;
- the scripts and files of the tables and figures.

## Layout

- `generator/`: the four modules that produced the stored reference fields.
  The article identifies them by SHA-256 digest (see "Things worth knowing
  before comparing hashes").
  - `generate_realistic_coverage.py`: the empirical propagation model, i.e.
    Okumura-Hata or COST-231 Hata plus an NDVI vegetation term and a slope
    terrain term.
  - `prepare_transfer_dataset_v19.py`: builds the per-quadrant graph dataset and
    its reference targets.
  - `enrich_rf_targets.py`: the target-enrichment step that fills target
    columns 1 and 2 (shadow margin and diffraction loss).
  - `contrafactual_alvo_completo.py`: the correction module that produced the
    stored targets.
  - `graph_build/preparar_transfer_seed42.py`: the seeded driver of the dataset
    builder. The builder draws one random terrain coefficient per node and sets
    no seed itself; the driver sets the seed (42) before each quadrant.
- `partition/`: `spatial_cv.py` (square-block assignment of the nodes and a
  buffered block k-fold) and the `config/` package it imports. The trainers take
  their block assignment from `spatial_cv.py`.
- `training/`: the training and evaluation code.
  - `frozen/train_gnn_c0_spatial.py`: the frozen GNN trainer for one cell under
    the three-way buffered block split.
  - `frozen/train_mlp_c0_spatial.py`: the graph-free MLP control trained under
    the same protocol.
  - `models/`: encoder, model, decoders and physics-informed loss.
  - `rf_diagnostic_metrics.py`: per-epoch diagnostic metrics.
  - `baselines/empirical_models.py`: free-space path loss (FSPL), Okumura-Hata
    and COST-231 Hata.
  - `modelo_v3/`: the trainers of the model campaigns. `train_gnn_v3.py` and
    `train_mlp_v3.py` load the frozen trainers unchanged and replace their
    decoder with the affine decoder of `rf_decoder_v3.py`. Shared utilities are
    in `v3_common.py`.
  - `G1/`: the shim `train_v3_g10.py`, which runs those trainers at g = 10 km
    and b = 2 km, and the campaign launchers `rodar_lote_G1.py`,
    `rodar_lote_G1_retomada.py` and `rodar_lote_G1_bloco4.py`.
- `analysis/`: the scripts of the article.
  - One script per numerical test, named after the test identifier:
    - `v3_1.*`, `v3_2.*` and `v3_3.*`: retention, inclusion probabilities,
      estimands, error drift and calibration;
    - `v3_12_R1_*` to `v3_12_R4_*`: checks R1 to R4;
    - `v3_13_*`: checks R5 to R7 and tests E1 and E4, with the loop shared by R5
      and R6 in `v3_13_laco_comum.py`;
    - `v3_14_G2_alvo_continuo_v2.py`: the continuous-target test (G2), which
      repeats the drift measurement with terrain elevation as the target;
    - `v3_14_blocos_com_no_valido.py`: the count of blocks that hold at least
      one valid node.
  - The aggregators of the model campaigns: `v3_A2c_veredito.py`,
    `v3_A3_agregar.py`, `v3_A4_agregar.py`, and `v3_G1_agregar.py` with its
    versions `v3_G1_agregar_v5.py`, `v3_G1_agregar_v8.py` and
    `v3_G1_agregar_v10.py`.
  - The table and figure generators: `v3_12_gerar_tabelas.py`,
    `v3_13_gerar_TN_T6.py`, `v3_13_gerar_T9_G1.py`, `v3_12q_gerar_T10_R5.py`,
    `v3_12t_gerar_TA1.py`,
    `v3_12_gerar_figuras.py` and `v3_12_fig1_block_erosion.py`.
  - Their earlier versions, which the later scripts check by digest or follow:
    `v3_gerar_tabelas.py`, `v3_gerar_figuras.py` and `v3_fig1_block_erosion.py`.
  - The modules the tests import:
    - `varredura_split_geometria.py`, which holds the shared split code;
    - `varredura_fracao_valida_r3.py`;
    - `montecarlo_retencao_r3_g5.py`;
    - `fismat_sympy_eqR.py`, the symbolic and numerical check of the retention
      law, imported by `v3_1.5_retencao_exata_vs_mc.py`.
  - Two cell-level checks: `blocos_efetivos_20celulas.py` and
    `r3_alvo_pos_correcao_16celulas.py`.
- `criteria/`: `criterio_*.json`, the decision criteria written before each
  run. Each one fixes the thresholds and reading rules of a test or campaign,
  and its file name carries the test identifier. The model launchers read their
  own copies, `results/gpu/A3/criterio_A3.json` and
  `results/gpu/A4/criterio_A4.json`.
- `results/`: the JSON records, plus the small scripts that sit beside some of
  them.
  - `fase1/`: geometry of the cells, buffer retention (block and nodal Monte
    Carlo, closed-form law, exact form) and the 200-draw nodal references at
    g = 10 km and g = 5 km.
  - `fase2/`: per-node inclusion probabilities, the formal estimand with
    Horvitz-Thompson weighting, and the error drift of the baselines over 60
    random split draws.
  - `fase3/`:
    - calibration of the analytical baselines;
    - edges of the median-calibration band and the FSPL coincidence test;
    - edge structure of the graphs (`3.6_arestas_terreno_script.py` and its
      output);
    - per-cell variance decomposition;
    - content of the raw feature columns.
  - `fase4/`:
    - checks R1 (b = 0 drift), R2 (design reference), R3 (design term) and R4
      (closed-form inclusion probability against frequency);
    - the hyper-parameter sheet `B7.2_hiperparametros/` with its script;
    - `recalculo_R2_R4/` (two recalculation scripts and their outputs);
    - the folders `votos_fismat_R2_R4/`, `votos_R3_R4/` and
      `votos_termo_razao_campinas_sorocaba/`.
  - `fase5/`:
    - checks R5 (variance of one split draw), R6 (level of the constant
      predictor) and R7 (split draws against training seeds);
    - tests E1 (block K-fold scored jointly) and E4 (Hajek weighting with the
      closed-form inclusion probability);
    - the continuous-target test G2: `G2_resumo.json`, the per-cell records of
      its 60 split draws in `_parcial_G2/` (with `_P1_cidades.json`, the
      checks on the elevation column), and its independent recalculation in
      `votos_G2/`;
    - the count of blocks with a valid node, `blocos_com_no_valido.json`, and
      its recalculation in `votos_blocos_validos/`;
    - `recalculo_G1/` and `votos_R5/`.
  - `gpu/`: the model campaigns. Each run has a folder with its record
    `run_<label>.json`, and most G1 runs also have the shim record
    `shim_g10_<label>.json`.
    - `A2c/`: run-to-run repeatability, with launcher, status and outcome.
    - `A3/`: GNN and MLP over five training seeds at split seed 42.
    - `A4/`: GNN and MLP over five split draws in the four Bauru cells, plus an
      MLP width-sensitivity arm whose run folders end in `_antigas`.
    - `G1/`: GNN and MLP at g = 10 km, with plans, launcher status, gate,
      script-provenance records and the four aggregates.
    - `G1_votos_bloco1/` to `G1_votos_bloco4/`: the outcome of the independent
      recalculation of each G1 block.
  - `verificacoes/`: the outputs of individual checks, each next to the script
    that produced it: `G4_G5_celulas/`, `V1_deriva/`, `V2_modelos/` and
    `V3_sentinela_geometria_props/`. The outputs of the two `analysis/`
    cell-level checks are in `blocos_efetivos_20celulas/` and
    `r3_alvo_pos_correcao_16celulas/`.
  - `inputs/`: `treinos_c1/` holds the records of the 20 frozen-trainer runs at
    split seed 42 (the 16 cells at g = 10 km and the four Bauru cells at
    g = 5 km), which the geometric tests read.
    `fracao_valida_val_treino_20_splits.json` holds the per-draw valid
    fractions read by the figure script.
- `tables_figures/`: the tables and figures of the article.
  - `tables_v3-12/`: the LaTeX tabular and CSV of each table, the checks
    written by the table scripts (`CONFERENCIAS_tabelas_v3-12.json`) and the
    values record of the trained-model table (`T9_v3-12t_valores.json`) and the
    provenance record of the label corrections in Tables 1 and 5
    (`TN_T6_v3-13_proveniencia.json`).
  - `figures_v3-12/`: PDF and PNG of the figures, the figure manifest
    (`MANIFEST_figuras_v3-12.json`) and the check of the drift panel against
    its table (`F4_conferencia_T3.json`).
  - `suplementar/`: sources and PDFs of Tables S1 to S3.
  - `prior_version_csv/`: four CSVs of the earlier table set, which
    `v3_12_gerar_tabelas.py` reads as cross-checks.
- `manifest/data_manifest.csv`: name, size and SHA-256 of every data file that
  is not distributed (see "What is not here").

Root files:

- `README.md`: this file.
- `LICENSE`: MIT licence for the code.
- `LICENSE-DATA`: CC BY 4.0 for the records, tables, figures and manifest.
- `CITATION.cff`: citation metadata.
- `environment.txt`: the frozen package list of the authors' environment.
- `SHA256SUMS`: the SHA-256 of every file in the repository.
- `verify_code.py` and `CODE_AST_SHA256`: check that the published scripts
  contain the same code as the originals.
- `verify_payload.py` and `PAYLOAD_SHA256`: check that the published JSON
  records carry the same numbers as the originals.
- `PROVENANCE.md`: maps the original SHA-256 of each changed file to its
  published SHA-256.
- `.gitattributes`: Git attribute settings of the repository.

### Names in Portuguese

The code and the records were written in Portuguese. Comments, docstrings and
free-text notes are in English, but file names, folder names, JSON keys,
command-line options and identifiers kept their original names, because code
and records refer to them. The names a reader meets most often are the
following.

- `criteria/criterio_*.json`: the decision criterion of a test, written before
  the run.
- Folders named `votos_*` (including `G1_votos_bloco*`) and `recalculo_*`: an
  independent recalculation of a result, with its script or output.
- Files named `veredito*.json`: the outcome of such a recalculation or of a
  check. `veredito_A2c.json` is the outcome of the repeatability check.
- `fase1` to `fase5`: the stages of the numerical tests.
- `sorteio`: split draw.
- `semente`: seed.
- `celula`: cell, i.e. city x quadrant, written `<city>_Q<n>`.
- `validos`: valid nodes.
- `agregado`: aggregate.
- Run labels: `s<n>` is the training seed, `ss<n>` the split seed and
  `rep<n>` a repeat.

| Key or name | Meaning |
| --- | --- |
| `celula`, `celulas` | cell(s), city x quadrant |
| `sorteio`, `por_sorteio` | split draw; per draw |
| `semente`, `seed_treino`, `split_seed` | seed; training seed; split seed |
| `validos` | valid nodes (target path loss below 299 dB) |
| `sentinela` | sentinel nodes (imputed target, path loss of 299 dB or more) |
| `todos` | all nodes |
| `constante` | the constant predictor |
| `treino`, `teste`, `val` | training, test, validation partition |
| `n_nos_antes_do_buffer`, `n_nos_apos_buffer` | node counts before and after the buffer |
| `retencao` | retention (share of a partition kept after the buffer) |
| `deriva` | drift (of the error across split draws) |
| `desenho` | design (as in design-based inference) |
| `bloco` | spatial block; in campaign G1 also a block of runs of the plan |
| `media`, `mediana`, `dp` | mean, median, standard deviation |
| `razao` | ratio |
| `cobertura` | coverage |
| `fechado` | closed form |
| `leitura` | reading: one of several ways of computing a quantity, each declared in the criterion |
| `limiar` | threshold |
| `agregado`, `resumo`, `parcial` | aggregate; summary; partial (per-cell intermediate) |
| `comando_rodado`, `comando` | the command that produced the record |
| `entradas`, `insumos`, `fontes` | inputs (with their SHA-256) |
| `conferencia`, `validacao` | consistency check; validation check |
| `proveniencia` | provenance |
| `lote`, `plano`, `portao` | batch of runs; run plan; gate check run before a batch |
| `nao_treinaveis` | runs that could not be trained (a partition with no antenna-terrain edge) |
| `retomada` | resumed batch |
| `melhor_epoca` | selected epoch |
| `nota` | free-text note |

## What is not here

- The data files. These are the reference-field tensors
  (`transfer_dataset_<city>_v19_<Q>_enriched_cftudo.pt`, 16 files, about
  450 GB in total), the graph tensors (`<city>_v19_<Q>_gpu.pt`, 16 files, about
  250 GB), the per-node test predictions of the trained runs
  (`predicoes_<label>.npz`, about 10 GB) and the model checkpoints (about 7 GB).
  They are too large for the repository and are available from the
  corresponding author on request. Each one is listed in
  `manifest/data_manifest.csv` with the columns `group` (`reference_field_tensors`,
  `graph_tensors`, `node_predictions`, `checkpoints`), `file`, `size_bytes` and
  `sha256`. A copy obtained separately can therefore be checked with `sha256sum`.
- The artifact manifest in JSON Lines format that the trainers and some scripts
  read (`manifest_mathematics_v4.jsonl`). For the 16 reference fields its
  digests are the ones in `manifest/data_manifest.csv`.
- The frozen baseline script `baselines_v2_por_particao.py`, from which
  `analysis/v3_2.1_3.1_deriva_calibracao.py` imports the path-loss formulas.
- The manuscript.
- The authors' internal working notes. Some records and scripts point to them
  under neutral names of the form `internal/...`. Those files are not part of
  the package.

## Checking a number

To check a number of the article, find its table or figure in the list below.
Each table file is written by the script named there, which reads named fields
of the listed records, so the value can be followed into the record. The
numbering follows the submitted manuscript. Table files are in
`tables_figures/tables_v3-12/`, figure files in `tables_figures/figures_v3-12/`,
and records in `results/`.

| Article | File prefix | Content | Written by (`analysis/`) | Records read |
| --- | --- | --- | --- | --- |
| Table 1 | `TN_notacao_v3-12` | notation | `v3_12_gerar_tabelas.py`, then `v3_13_gerar_TN_T6.py` | a LaTeX block of the manuscript and the environment of five labels in the manuscript source (not distributed) |
| Table 2 | `T2_retencao_v3-12` | retention after the buffer, by cell | `v3_12_gerar_tabelas.py` | `fase1/1.8_referencia_nodal_200seeds.json`, `fase1/1.8b_referencia_nodal_200seeds_g5.json` |
| Table 3 | `T5_inclusao_por_classe_v3-12` | inclusion probability by edge class | `v3_12_gerar_tabelas.py` | `fase4/R4_p_por_classe.json` |
| Table 4 | `T3_resumo_v3-12` (per-cell detail in `T3_deriva_erro_v3-12.csv`) | error drift of the constant predictor, with and without the buffer | `v3_12_gerar_tabelas.py` | `fase2/2.1_deriva_erro_baselines_16x60rnd.json`, `fase2/_v3_2.1_3.1_parcial_16x60rnd.json`, `fase4/R1_b0_resumo.json`, `fase4/R1_b0_por_sorteio.json` |
| Table 5 | `T6_referencia_desenho_v3-12` | design reference: block hold-out against simple random sampling | `v3_12_gerar_tabelas.py`, then `v3_13_gerar_TN_T6.py` (row label only) | `fase4/R2_resumo.json`, `fase4/R2_por_sorteio.json`; cross-checked against `fase4/votos_fismat_R2_R4/` |
| Table 6 | `T10_cobertura_R5_v3-12` | coverage of the one-partition variance estimate | `v3_12q_gerar_T10_R5.py` | `fase5/R5_resumo.json`; checked against `fase5/R5_por_sorteio.json` and `fase5/votos_R5/veredito.json` |
| Table 7 | `T4_totais_v3-12` | constant predictor against calibrated FSPL, totals with and without the buffer | `v3_12_gerar_tabelas.py` | `fase2/_v3_2.1_3.1_parcial_16x60rnd.json`, `fase4/R1_b0_por_sorteio.json` |
| Table 8 | `T9_deriva_modelo_G1_v3-12` | drift of the trained models at g = 10 km (campaign G1) | `v3_13_gerar_T9_G1.py` | `gpu/G1/agregado_G1_v5_bloco1.json`, `gpu/G1/agregado_G1_v5_bloco2.json`, `gpu/G1/agregado_G1_v8_bloco3.json`, `gpu/G1/agregado_G1_v10_bloco4.json`; outcomes in `gpu/G1_votos_bloco1/` to `gpu/G1_votos_bloco4/`; values in `tables_figures/tables_v3-12/T9_v3-12t_valores.json`; the caption is compared with the manuscript source (not distributed) |
| Table S1 | `TA1a_hiperparametros_arquitetura_perda_v3-12`, `TA1b_hiperparametros_otimizacao_custo_v3-12` (CSV `TA1_hiperparametros_v3-12.csv`) | training settings of the two models | `v3_12_gerar_tabelas.py`, then `v3_12t_gerar_TA1.py` for TA1a | `fase4/B7.2_hiperparametros/b7_2_folha_hiperparametros.json` |
| Table S2 | `T4_S2_v3-12` | constant predictor against calibrated FSPL, by cell | `v3_12_gerar_tabelas.py` | as Table 7 |
| Table S3 | `T7b_deriva_modelo_A4_v3-12` | trained models at g = 5 km in the four Bauru cells | `v3_12_gerar_tabelas.py` | `gpu/A4/agregado_A4.json` |
| Figure 1 | `fig1_block_erosion` | erosion of a held-out block by the buffer | `v3_12_fig1_block_erosion.py` | none (synthetic geometry) |
| Figure 2 | `F34_fracao_e_deriva` | valid fraction of the test partition and error drift, by cell | `v3_12_gerar_figuras.py` | `inputs/fracao_valida_val_treino_20_splits.json`, `fase2/_v3_2.1_3.1_parcial_16x60rnd.json`, `fase2/2.1_deriva_erro_baselines_16x60rnd.json`, `T3_deriva_erro_v3-12.csv` |

Numbers quoted in the text come from the same records, or from the records of
the tests listed in "How to reproduce each table and figure". Two sets of text
numbers have no table:

- The continuous-target test. With terrain elevation as the target and a
  constant predictor fixed on one reference draw, the median over the sixteen
  cells of the coefficient of variation of the all-node MAE across the sixty
  draws (`primario.agregado_16_celulas.CV_d`), the range of the number of
  independent nodes (`n_ef`) and the mean number of nodes scored per draw
  (`n_medio`) are in `results/fase5/G2_resumo.json`; the per-cell values are
  under `primario.por_celula`, and the permuted-elevation control that behaves
  as a simple random sample is under `controle_negativo_P5`. The per-draw
  records are in `results/fase5/_parcial_G2/`, the decision criterion and its
  addendum in `criteria/criterio_G2_deriva_alvo_continuo.json` and
  `criteria/criterio_G2_adendo1.json`, and the independent recalculation in
  `results/fase5/votos_G2/`.
- The blocks with a valid node. The number of the 132 blocks of each cell that
  hold at least one valid node, with its range over the sixteen cells and over
  the four Q1 cells, is in `results/fase5/blocos_com_no_valido.json`
  (`por_celula`, `faixa_16_celulas`, `faixa_4_celulas_Q1`); its independent
  recalculation is in `results/fase5/votos_blocos_validos/`.

The records of the numerical tests store the command that produced them
(`comando_rodado` or `comando`), the SHA-256 of the script (`script_sha256`)
and the SHA-256 of their inputs (`entradas_sha256`, `insumos`, `fontes`, or the
digest of the imported script). The aggregates of the model campaigns store
the digests of their criteria. They also check each run against the launcher
status or run plan, and the run's reference field against the artifact manifest. The outputs under
`results/verificacoes/` and the recalculation folders carry fewer provenance
fields, and the docstring of the script beside each one describes its inputs.
The table scripts check every printed value against its record and write the
result to `CONFERENCIAS_tabelas_v3-12.json`. The figure script writes the
SHA-256 of its inputs, scripts and outputs to `MANIFEST_figuras_v3-12.json`.

## Verifying the package

Run the following at the repository root. All three use only the Python
standard library.

    sha256sum -c SHA256SUMS
    python verify_payload.py
    python3.11 verify_code.py

- `sha256sum -c SHA256SUMS` confirms that each file is byte for byte the
  published one.
- `python verify_payload.py` confirms that each JSON or JSON Lines record
  listed in `PAYLOAD_SHA256` carries the same numbers as the original record.
- `python3.11 verify_code.py` confirms that each script listed in
  `CODE_AST_SHA256` contains the same executable code as the original script.

The last two tools print how many files matched and exit with status 1 if any
file does not match. The next section explains why the last two checks are
needed.

## Things worth knowing before comparing hashes

**Scripts.** Before publication, the comments and docstrings of the Python
sources were rewritten in English and internal working comments were removed.
The executable code is unchanged. String literals that cited the authors'
internal working notes were reworded, and the names of note files and of the
notes and scratch directories were replaced by neutral ones (`internal/...`).
One variable named after such a directory was renamed with the neutral prefix
`INTERNAL_`. `verify_code.py` shows that the code is otherwise identical, as follows.

1. It parses each script, drops the docstrings and replaces every free-text
   string with a fixed placeholder. A string counts as free text when it has
   four or more words, names a `.md` note, or contains one of the neutral
   `internal/...` names. An f-string whose literal text meets that test keeps
   only its interpolated expressions, in order. A variable whose name starts
   with `INTERNAL_` is replaced with a fixed name.
2. It hashes the resulting syntax tree.
3. It compares that hash with the digest of the original script, recorded in
   `CODE_AST_SHA256` before publication.

Everything else enters the digest: names, calls, operators, numbers, short
strings such as keys, labels and data paths, and the expressions inside every
f-string. Any change to executable code therefore fails the check. The layout
of the syntax tree differs between Python versions, so the check must be run
with Python 3.11, the version that computed the digests; with another version
the script prints a warning.

**Records.** The free-text notes of the JSON records were translated into
English. Names of internal note files and of the notes and scratch directories
were replaced by neutral ones (`internal/...`). The values of a few attribution
keys were replaced by neutral labels; these keys record who produced or
requested a record and when, and are listed as `ROLE_KEYS` in the script. The
numbers are untouched. `verify_payload.py` shows this, as follows.

1. For each record it computes the canonical JSON with every free-text string
   replaced by a fixed placeholder. Free text is defined as for the scripts,
   plus the values of the attribution keys.
2. It hashes that canonical JSON.
3. It compares the hash with the digest of the original record, recorded in
   `PAYLOAD_SHA256` before publication.

Numbers, booleans, nulls, keys, structure, short labels, identifiers, data
paths and SHA-256 values all enter the digest. A single changed value
therefore fails the check.

**Consequences for recorded digests.** Because the file bytes changed, the
SHA-256 of a published script differs from the `script_sha256` recorded inside
the records. That field documents the script as it was when the record was
produced. Likewise, some scripts assert the digest of another script or of a
criterion file before running (for example `analysis/v3_12_R1_b0_deriva.py`,
`analysis/v3_13_laco_comum.py`, `analysis/v3_13_E1_kfold_conjunto.py`,
`analysis/v3_A3_agregar.py`, `analysis/v3_14_G2_alvo_continuo_v2.py`,
`analysis/v3_14_blocos_com_no_valido.py`, `training/G1/train_v3_g10.py` and the
G1 launchers). Those assertions refer to the original files and stop when they
meet the published copies; to re-run such a script, replace the expected
digest with the published one.

**Line numbers.** Some records and notes cite a script by line number (for
example `prepare_transfer_dataset_v19.py:181-184`). Those numbers refer to the
original files. The four modules in `generator/` keep the cited code lines at
their original numbers; in the other scripts the rewritten comments may shift a
cited line by a few positions.

**One corrected note.** The note of `results/gpu/G1/agregado_G1_v10_bloco4.json`
that describes the check of each prediction file against its run record gave
the relative tolerance for the RMSE as 3 %. The aggregator that wrote the record
applies 5 % (constant `TOL_AMARRA_RMSE_REL` of `analysis/v3_G1_agregar_v10.py`),
and the published note and script say 5 %. No number of the record changed.

`PROVENANCE.md` maps the original SHA-256 of every changed file to its
published SHA-256. This includes the four generator modules that the article
identifies by the first eight hex digits of the SHA-256 of the files as run:

| Digest cited in the article | Module |
| --- | --- |
| `95ea0423` | `generator/generate_realistic_coverage.py` |
| `45c16d43` | `generator/prepare_transfer_dataset_v19.py` |
| `5d38012e` | `generator/enrich_rf_targets.py` |
| `ebeaf759` | `generator/contrafactual_alvo_completo.py` |

The data files were not modified. The digests in `manifest/data_manifest.csv`,
and the tensor digests stored in the records, can be compared directly with
`sha256sum` on a copy obtained from the authors.

## Environment

The authors used Python 3.11.16 with PyTorch 2.10.0+cu128 (CUDA 12.8) and
PyTorch Geometric 2.8.0.post1 (pyg-lib 0.8.0, torch-scatter 2.1.2,
torch-sparse 0.6.18). The other main libraries were NumPy 2.4.6, SciPy 1.17.0,
pandas 2.3.3, matplotlib 3.10.8, SymPy 1.14.0, scikit-learn 1.8.0 and
pyproj 3.7.2. The training campaigns ran on a single NVIDIA GPU, and the
scripts in `analysis/` run on CPU.

`environment.txt` is the complete package list of that environment, as printed
by `uv pip freeze`. It is broader than the package needs. To install it, run
`pip install -r environment.txt`; the CUDA builds of PyTorch and of the PyG
extensions must come from the matching wheel index.

## Absolute paths in the scripts

The scripts were written to run in the authors' working tree and carry its
absolute paths. They are included so that the computation behind each number
can be read line by line. To run them elsewhere, you need to obtain the data
(see "What is not here") and set the path constants.

- **`analysis/` scripts and the scripts under `results/`.** Set the constants
  that point to the project tree. Typical names are `B`, `BASE`, `BASE_PADRAO`,
  `V3`, `M`, `RAIZ`, `SCRIPTS`, `OUT`, `OUT_DIR`, `HERE`, `AQUI`, `TENSOR_DIR`,
  `RUNDIR`, `NPZ_DIR`, `MANIFEST` and `MANIFEST_V4`. Two literal prefixes also
  need setting:
  - `/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25`
    is the root that holds `fase1` to `fase5`, `gpu`, and the table and figure
    folders under `redacao_v3-12/`. In this repository it corresponds to
    `results/` and `tables_figures/`.
  - `/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts` is the
    script folder, which corresponds to `analysis/`.

  Outputs are written to those locations, not to the folders of this repository.
  Some scripts take their output path from `--out`.
- **`training/modelo_v3/v3_common.py` (and `rf_decoder_v3.py`).** Set
  `GNN_RF_V2`, the tree that holds the original module folders.
  `GRAPH_DIR_DEFAULT` is the directory with the `*_enriched_cftudo.pt` and
  `*_gpu.pt` tensors. `MANIFEST_V4_PATH` is the JSON Lines artifact manifest;
  it is not distributed, and `manifest/data_manifest.csv` carries the same
  reference-field digests in CSV form, but the loader reads only the JSON Lines
  format. `FROZEN_SCRIPTS_DIR` is the folder of the two frozen trainers, which
  is `training/frozen/` here.
- **`training/frozen/train_gnn_c0_spatial.py` and `train_mlp_c0_spatial.py`.**
  `--base-dir` must point to a tree with the original module layout. Here the
  modules are grouped by role:
  - `02_models` corresponds to `training/models/`;
  - `03_training` corresponds to `partition/spatial_cv.py` and
    `training/rf_diagnostic_metrics.py`;
  - `04_baselines` corresponds to `training/baselines/`;
  - `config` corresponds to `partition/config/`;
  - `data_raw` corresponds to `generator/`.

  To run the trainers unchanged, recreate that layout with copies or symbolic
  links.
- **Launchers in `results/gpu/A2c/`, `results/gpu/A3/` and `results/gpu/A4/`.**
  These find the campaign trainers at `../modelo_v3` relative to their own
  folder (`training/modelo_v3/` here). The G1 launchers write their plans,
  status files and run folders next to themselves.
- **Supplementary tables.** The sources of Tables S1 to S3 `\input` the table
  fragments through the relative path `../tables_v3-12/`. That path works as
  shipped, because `tables_figures/suplementar/` and
  `tables_figures/tables_v3-12/` are siblings.
- **Inputs that are not distributed.** The notation table (Table 1) is built
  from a LaTeX block of the manuscript, so it cannot be regenerated from this
  repository. Its tabular and CSV are provided. `v3_13_gerar_TN_T6.py` also
  reads the manuscript source, to take the environment (Proposition, Remark or
  Lemma) of five labels; its result is in `TN_T6_v3-13_proveniencia.json`.
  `v3_13_gerar_T9_G1.py` compares the caption it writes with the manuscript
  source named by `--manuscrito`. Outside the authors' tree, those paths have
  to be pointed at a copy of the manuscript or the comparison left out.
  `v3_14_G2_alvo_continuo_v2.py` checks, in its consolidation step, the number
  of test nodes of each draw against the per-cell records of the loop shared by
  checks R5 and R6 (`fase5/_parcial_laco/<cell>.json`), which are not
  distributed; the outcome of that check is stored in `G2_resumo.json`, under
  `portoes.P3_particao`.

## How to reproduce each table and figure

The commands assume the environment above, the data obtained on request and
the path constants set as described. `python` denotes the environment's
interpreter. The commands were taken from the `comando_rodado` / `comando`
fields of the records and from the `Usage` lines of the script docstrings.
Where a record and this table differ, the record is authoritative.

### Numerical tests (inputs of the tables)

| Test | Command | Output |
| --- | --- | --- |
| Geometry of the 16 cells; buffer check along the Bauru quadrant chain (1.1, 1.2) | `python analysis/v3_1.1_1.2_celulas_cadeia.py --out-11 <json> --out-12 <json> --fig <fig>` | `results/fase1/1.1_geo_celulas_16.json`, `results/fase1/1.2_cadeia_buffer_violacao.json` |
| Nodal Monte Carlo against block Monte Carlo (1.3) | `python analysis/v3_1.3_mc_nodal_vs_blocos.py --out <json>` | `results/fase1/1.3_mc_nodal_vs_blocos.json` |
| Retention law against nodal retention over b (1.4) | `python analysis/v3_1.4_varredura_b_lei_vs_nodal.py --out <json> --seeds 20` | `results/fase1/1.4_varredura_b_lei_vs_nodal.json` |
| Symbolic and numerical check of the retention law | `python analysis/fismat_sympy_eqR.py` (writes the JSON next to the script) | `results/fase1/fismat_sympy_eqR.json` |
| Exact retention against the mean-field form (1.5) | `python analysis/v3_1.5_retencao_exata_vs_mc.py --out <json>` | `results/fase1/1.5_retencao_exata_vs_mc.json` |
| Nodal retention reference, 200 draws, g = 10 km and g = 5 km (1.8, 1.8b) | `N_SEEDS=200 python analysis/v3_1.8_referencia_nodal_200seeds.py`; `N_SEEDS=200 python analysis/v3_1.8b_referencia_nodal_200seeds_g5.py` | `results/fase1/1.8_referencia_nodal_200seeds.json`, `results/fase1/1.8b_referencia_nodal_200seeds_g5.json` |
| Per-node inclusion probabilities and formal estimand (1.6, 2.3); per-draw values (2.3b) | `python analysis/v3_1.6_2.3_estimando_formal.py`; `python analysis/v3_2.3b_por_sorteio.py` | `results/fase2/1.6_p_inclusao_por_no.json`, `results/fase2/2.3_estimando_ht_baselines.json`, `results/fase2/2.3b_estimando_por_sorteio.json` |
| Error drift and calibration of the baselines, 60 random draws (2.1, 3.1) | `python analysis/v3_2.1_3.1_deriva_calibracao.py --seeds aleatorios:60:20260926 --sufixo-saida _16x60rnd --sem-sha` | `results/fase2/2.1_deriva_erro_baselines_16x60rnd.json`, `results/fase2/_v3_2.1_3.1_parcial_16x60rnd.json`, `results/fase3/3.1_calibracao_validos_vs_mediana_16x60rnd.json` |
| Valid fraction of each partition over 20 draws | `python analysis/varredura_fracao_valida_r3.py --out <json>` | `results/inputs/fracao_valida_val_treino_20_splits.json` |
| Edges of the median-calibration band (3.2) | `python analysis/v3_3.2_banda_p7_bordas_reais.py --out <json>` | `results/fase3/3.2_banda_p7_bordas_reais.json` |
| FSPL coincidence (3.4) | `python analysis/v3_3.4_coincidencia_fspl.py --out <json>` | `results/fase3/3.4_coincidencia_fspl.json` |
| Edge structure of the graphs (3.6) | `python results/fase3/3.6_arestas_terreno_script.py <json>` | `results/fase3/3.6_arestas_terreno.json` |
| Per-cell variance decomposition (3.7b, read from `results/fase3/3.7_resposta_cego.json`) | `python analysis/v3_3.7b_decomposicao_por_celula.py` | `results/fase3/3.7b_decomposicao_por_celula.json` |
| Raw feature columns (3.9) | `python analysis/v3_3.9_colunas_lidar.py` | `results/fase3/3.9_colunas_features_raw.json` |
| Drift at b = 0 (R1) | `python analysis/v3_12_R1_b0_deriva.py --validar`, then `--celula <cell>` for each of the 16 cells, then `--consolidar` | `results/fase4/R1_b0_por_sorteio.json`, `results/fase4/R1_b0_resumo.json` |
| Design reference (R2) | `python analysis/v3_12_R2_referencia_desenho.py` | `results/fase4/R2_por_sorteio.json`, `results/fase4/R2_resumo.json` |
| Design term (R3) and closed-form inclusion probability against frequency (R4) | `python analysis/v3_12_R3R4_motor.py`, then `python analysis/v3_12_R3_termo_desenho.py` and `python analysis/v3_12_R4_p_fechado_vs_frequencia.py` | `results/fase4/R3_termo_desenho.json`, `results/fase4/R4_p_por_classe.json` |
| Recalculations of R2 and R4 | `python results/fase4/recalculo_R2_R4/fismat_R2_termo_razao.py <json>`; `python results/fase4/recalculo_R2_R4/fismat_R4_teste_exato.py <json>` | `results/fase4/recalculo_R2_R4/fismat_R2_termo_razao.json`, `results/fase4/recalculo_R2_R4/fismat_R4_teste_exato.json` |
| Hyper-parameter sheet | `python results/fase4/B7.2_hiperparametros/b7_2_folha_hiperparametros.py` | `results/fase4/B7.2_hiperparametros/b7_2_folha_hiperparametros.json`, `results/fase4/B7.2_hiperparametros/b7_2_folha_hiperparametros.csv` |
| Variance of one split draw (R5) and level of the constant predictor (R6) | `python analysis/v3_13_laco_comum.py --validar`, then `--celula <cell>` for each of the 16 cells; `python analysis/v3_13_R5_variancia_um_sorteio.py --consolidar`; `python analysis/v3_13_R6_nivel_sentinela.py --consolidar` | `results/fase5/R5R6_validacao_laco.json`, `results/fase5/R5_por_sorteio.json`, `results/fase5/R5_resumo.json`, `results/fase5/R6_por_sorteio.json`, `results/fase5/R6_resumo.json` |
| Split draws against training seeds at g = 5 km (R7) | `python analysis/v3_13_R7_sementes_x_sorteios.py` | `results/fase5/R7_resumo.json` |
| Block K-fold scored jointly (E1) | `python analysis/v3_13_E1_kfold_conjunto.py --validar`, then `--celula <cell>` for each of the 16 cells, then `--consolidar` | `results/fase5/E1_validacao.json`, `results/fase5/E1_por_sorteio.json`, `results/fase5/E1_resumo.json` |
| Hajek weighting with the closed-form inclusion probability (E4) | `python analysis/v3_13_E4_hajek_p_fechado.py` | `results/fase5/E4_resumo.json`, `results/fase5/E4_por_sorteio.json`, `results/fase5/E4_validacao.json` |
| Continuous-target test, terrain elevation as target (G2) | `python analysis/v3_14_G2_alvo_continuo_v2.py --executar` (or `--celula <cell>` for each of the 16 cells, then `--consolidar`) | `results/fase5/_parcial_G2/<cell>.json`, `results/fase5/_parcial_G2/_P1_cidades.json`, `results/fase5/G2_resumo.json` |
| Independent recalculation of G2 | `python results/fase5/votos_G2/voto_G2_recalculo.py n1`, then `python results/fase5/votos_G2/voto_G2_recalculo.py n2 <cell>` for `bauru_Q2`, `campinas_Q1`, `lins_Q2` and `sorocaba_Q2` | `results/fase5/votos_G2/nivel1.json`, `results/fase5/votos_G2/nivel2_<cell>.json`, `results/fase5/votos_G2/veredito.json` |
| Blocks with a valid node | `python analysis/v3_14_blocos_com_no_valido.py`; recalculation: `python results/fase5/votos_blocos_validos/voto_blocos.py` | `results/fase5/blocos_com_no_valido.json`, `results/fase5/votos_blocos_validos/veredito.json` |
| Repeatability of training (A2c; GPU and tensors needed) | `python results/gpu/A2c/rodar_lote_A2c.py`; `python analysis/v3_A2c_veredito.py` | `results/gpu/A2c/lote_A2c_status.json`, run folders, `results/gpu/A2c/veredito_A2c.json` |
| Model campaigns A3 and A4 (GPU and tensors needed) | `python results/gpu/A3/rodar_lote_A3.py`; `python results/gpu/A4/rodar_lote_A4.py` (both call `training/modelo_v3/train_gnn_v3.py` and `train_mlp_v3.py`); then `python analysis/v3_A3_agregar.py` and `python analysis/v3_A4_agregar.py` | run folders, `results/gpu/A3/agregado_A3.json`, `results/gpu/A4/agregado_A4.json` |
| Model campaign G1 at g = 10 km (GPU and tensors needed) | `python training/G1/rodar_lote_G1.py`, `python training/G1/rodar_lote_G1_retomada.py`, `python training/G1/rodar_lote_G1_bloco4.py` (through the shim `training/G1/train_v3_g10.py`); then `python analysis/v3_G1_agregar_v5.py --bloco 1`, `--bloco 2`; `python analysis/v3_G1_agregar_v8.py --bloco 3`; `python analysis/v3_G1_agregar_v10.py --bloco 4` | run folders and plans in `results/gpu/G1/`, `results/gpu/G1/agregado_G1_v5_bloco1.json`, `results/gpu/G1/agregado_G1_v5_bloco2.json`, `results/gpu/G1/agregado_G1_v8_bloco3.json`, `results/gpu/G1/agregado_G1_v10_bloco4.json` |
| Variance components of the crossed draw x seed block of G1 | `python results/fase5/recalculo_G1/ffm_G1_bloco4_contas.py` | `results/fase5/recalculo_G1/ffm_G1_bloco4_contas.json` |
| Effective blocks per cell | `python analysis/blocos_efetivos_20celulas.py --runs results/inputs/treinos_c1/run_c0c1*_s42_Q*_g*b2.json --out <json> --hash` | `results/verificacoes/blocos_efetivos_20celulas/blocos_efetivos_20celulas.json` |
| Target statistics of the corrected reference fields | `python analysis/r3_alvo_pos_correcao_16celulas.py` | `results/verificacoes/r3_alvo_pos_correcao_16celulas/r3_alvo_pos_correcao_16celulas.json`, `..._RAW.json` |
| Individual checks | the script beside each output in `results/verificacoes/G4_G5_celulas/`, `V1_deriva/`, `V2_modelos/` and `V3_sentinela_geometria_props/` (see its docstring) | the `*_saida.json`, `saida_*.json` and `veredito.json` files in those folders |

### Tables

Run the table scripts in this order:

1. `python analysis/v3_12_gerar_tabelas.py` writes the `.tex`/`.csv` pairs of
   every table except T9 and T10. It reads the records listed in "Checking a
   number", re-derives several totals from the per-draw records and asserts
   them before writing. It also writes `CONFERENCIAS_tabelas_v3-12.json` and
   `MANIFEST_tabelas_v3-12.json` (inputs, script and outputs with SHA-256; this
   file is regenerated on each run and is not part of the repository).
2. `python analysis/v3_13_gerar_TN_T6.py` regenerates Tables 1 and 5 in a
   temporary folder with the generator of step 1, applies the two corrections
   (the environment word before five `\ref` of Table 1, and the row label
   `FSPL (calibrated once)` of Table 5), rewrites those files and adds its
   entries to the checks and the manifest. It reads the manuscript source.
3. `python analysis/v3_13_gerar_T9_G1.py [--manuscrito <main.tex>]` writes T9
   (Table 8) and `T9_v3-12t_valores.json`, and adds its entries to the checks
   and the manifest.
4. `python analysis/v3_12q_gerar_T10_R5.py` writes T10 (Table 6) and adds its
   entries.
5. `python analysis/v3_12t_gerar_TA1.py` regenerates the hyper-parameter tables
   with the generator of step 1. It checks that TA1b and the CSV are reproduced
   byte for byte, and writes TA1a, the architecture-and-loss part of Table S1,
   without the row of the MLP width-sensitivity arm.

Step 1 rewrites the checks and manifest files in full, so steps 2 to 5 must
follow it. Steps 2 to 5 can be repeated.

Tables S1, S2 and S3 are compiled with `pdflatex Table_S1.tex`,
`pdflatex Table_S2.tex` and `pdflatex Table_S3.tex` inside
`tables_figures/suplementar/`.

### Figures

1. `python analysis/v3_12_fig1_block_erosion.py` draws `fig1_block_erosion`
   (Figure 1) from synthetic geometry with g = 10 and b = 2 and reads no data.
2. `python analysis/v3_12_gerar_figuras.py` draws the panels F3 and F4 and
   their two-panel combination `F34_fracao_e_deriva` (Figure 2). Before
   drawing, it checks the drift aggregates against `T3_deriva_erro_v3-12.csv`
   and writes the check to `F4_conferencia_T3.json`. It then writes
   `MANIFEST_figuras_v3-12.json`.

Run the first script before the second, so that `fig1_block_erosion` enters the
figure manifest.

## Licence

The code is released under the MIT licence (`LICENSE`). The JSON and CSV
records, the decision criteria, the tables, the figures and the data manifest
are released under the Creative Commons Attribution 4.0 International licence
(`LICENSE-DATA`).

## Citation

The manuscript is under review at MDPI Mathematics. Until it is published,
please cite this repository using the metadata in `CITATION.cff`. Once the
article is published, please cite the article.
