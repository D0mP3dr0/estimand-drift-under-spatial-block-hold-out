"""Estimand Drift under Spatial Block Hold-Out (MDPI Mathematics, revision 1).
E6: GNN x MLP order inversions per draw (criteria/criterio_E6_inversoes_ordem_gnn_mlp.json).
MAE per run: mae_pop (with _finito, hash_idx and the constant SENTINELA_PL) is EXTRACTED by ast from
analysis/v3_G1_agregar_v10.py -- not rewritten; the digest of the file as read is recorded in the output.
Draws and exclusions (nao_treinavel / sem_validos) are read from results/gpu/G1/plano_G1.json and lote_G1_status.json
as in the aggregator. No training. Cross-check: MAE per run against the aggregates v5_bloco1, v5_bloco2, v8_bloco3,
v10_bloco4 (tol 1e-6); any divergence ends with rc 3.
The per-node prediction files (predicoes_<run>.npz, 150 runs) are not in the repository (available on request);
point env R1_NPZ_ROOT to a directory holding <run>/predicoes_<run>.npz (default: results/gpu/G1).
Output directory: env R1_OUT (default results/fase6_rerun)."""
import ast, hashlib, json, os, sys, datetime
import numpy as np
HERE = os.path.abspath(__file__)
BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # repository root
SCRIPT_REL = os.path.relpath(HERE, BASE).replace(os.sep, '/')
AGR = os.path.join(BASE, 'analysis', 'v3_G1_agregar_v10.py')
CRIT = os.path.join(BASE, 'criteria', 'criterio_E6_inversoes_ordem_gnn_mlp.json')
G1 = os.path.join(BASE, 'results', 'gpu', 'G1')
NPZ_ROOT = os.environ.get('R1_NPZ_ROOT', G1)
LIM, TOL = 0.132, 1e-6
CELULAS = [('bauru_Q1', 1), ('campinas_Q1', 1), ('bauru_Q3', 3), ('campinas_Q3', 3)]
OUT = os.environ.get('R1_OUT', os.path.join(BASE, 'results', 'fase6_rerun'))
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
LOG = []
def log(*a):
    s = ' '.join(str(x) for x in a); LOG.append(s); print(s); sys.stdout.flush()
def J(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)

# ---- reuse of the aggregator routine (ast extraction, the rest of the file is not executed)
sha_agr = sha(AGR)
src = open(AGR, encoding='utf-8').read()
tree = ast.parse(src)
ns = {'np': np, 'hashlib': hashlib}
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == 'Aborta':
        exec(compile(ast.Module([node], []), AGR, 'exec'), ns)
    if isinstance(node, ast.FunctionDef) and node.name in ('_finito', 'mae_pop', 'hash_idx'):
        exec(compile(ast.Module([node], []), AGR, 'exec'), ns)
    if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'SENTINELA_PL' for t in node.targets):
        exec(compile(ast.Module([node], []), AGR, 'exec'), ns)
mae_pop = ns['mae_pop']
# The aggregate v10_bloco4 records the digest of the ORIGINAL aggregator; the published file was cleaned (see PROVENANCE.md),
# so its digest differs. The digest of the file as read is recorded below instead of being asserted.

def main():
    os.makedirs(OUT, exist_ok=True)
    log('E6 inicio', datetime.datetime.now(datetime.timezone.utc).isoformat(), 'agregador_sha256', sha_agr)
    crit_sha = sha(CRIT)
    plano = J(os.path.join(G1, 'plano_G1.json'))['corridas']
    status = {c['run_label']: c for c in J(os.path.join(G1, 'lote_G1_status.json'))['corridas']}
    ref = {}  # (cel, tipo, split) -> [(fonte, valor)]
    def addref(fonte, cel, tipo, ss, v):
        ref.setdefault((cel, tipo, int(ss)), []).append((fonte, v))
    a1 = J(os.path.join(G1, 'agregado_G1_v5_bloco1.json')); a3 = J(os.path.join(G1, 'agregado_G1_v8_bloco3.json'))
    a2 = J(os.path.join(G1, 'agregado_G1_v5_bloco2.json')); a4 = J(os.path.join(G1, 'agregado_G1_v10_bloco4.json'))
    for cel, a, nm in (('bauru_Q1', a1, 'v5_bloco1'), ('campinas_Q1', a1, 'v5_bloco1'), ('bauru_Q3', a3, 'v8_bloco3'), ('campinas_Q3', a3, 'v8_bloco3')):
        for t in ('gnn', 'mlp'):
            for s, v in a['celulas'][cel]['por_modelo'][t]['mae_validos_por_sorteio'].items():
                addref(nm, cel, t, s, v)
    for cel in ('bauru_Q1', 'campinas_Q1'):
        for t in ('gnn', 'mlp'):
            b = a2['celulas'][cel][t]
            for s, row in zip(b['sorteios_usados'], b['matriz_mae_validos_sorteio_x_semente_42_43_44']):
                addref('v5_bloco2', cel, t, s, row[0])
    for t in ('gnn', 'mlp'):
        b = a4['celulas']['campinas_Q1'][t]
        for s, row in zip(b['sorteios_usados'], b['matriz_mae_validos_sorteio_x_semente_42_43_44']):
            addref('v10_bloco4', 'campinas_Q1', t, s, row[0])
    runs_sha, por, resumo_cel = {}, {}, {}
    conf, diverg = [], []
    for cel, bl in CELULAS:
        pl = [c for c in plano if c['bloco'] == bl and c['seed_treino'] == 42 and f"{c['cidade']}_{c['quadrante']}" == cel]
        idxs = sorted({c['indice_sorteio'] for c in pl})
        ssmap = {c['indice_sorteio']: c['split_seed'] for c in pl}
        linhas = []
        for i in idxs:
            ss = ssmap[i]; r = {'indice_sorteio': i, 'split_seed': ss}; mot = []
            for t in ('gnn', 'mlp'):
                lbl = f'g1_{t}_{cel}_ss{ss}_s42'; st = status.get(lbl, {}); d = os.path.join(G1, lbl)
                dn = os.path.join(NPZ_ROOT, lbl)
                r[t] = {'run_label': lbl}
                if st.get('nao_treinavel'):
                    r[t]['nao_treinavel'] = True; r[t]['particao'] = st.get('nao_treinavel_particao')
                    mot.append(f"{t}: nao_treinavel (particao sem aresta antena-terreno: {st.get('nao_treinavel_particao')})")
                    continue
                rj = os.path.join(d, f'run_{lbl}.json'); nz = os.path.join(dn, f'predicoes_{lbl}.npz')
                if not os.path.exists(nz):
                    if not st.get('sem_validos'):
                        raise SystemExit('ABORTA: sem npz e sem marca sem_validos em ' + lbl + ' (the .npz files are available on request; set R1_NPZ_ROOT)')
                    runs_sha[lbl] = {'run_json': sha(rj), 'npz': None}
                    r[t].update(sem_validos_sem_npz=True, mae_validos=None, mae_todos=None, n_validos=0)
                    mot.append(f'{t}: sem no valido no teste (corrida sem predicao; run JSON parcial)')
                    continue
                runs_sha[lbl] = {'run_json': sha(rj), 'npz': sha(nz)}
                dj = J(rj)
                z = np.load(nz); m = mae_pop(z)
                h = dj['particoes']['test'].get('idx_sha256_global')
                if h != m['idx_sha256']:
                    raise SystemExit(f'ABORTA: hash idx npz x run json em {lbl}')
                r[t].update(mae_validos=m['mae_rssi_validos_db'], mae_todos=m['mae_rssi_todos_db'], n=m['n'], n_validos=m['n_validos'])
                if m['n_validos'] == 0:
                    mot.append(f'{t}: sem no valido no teste')
                mv = m['mae_rssi_validos_db']
                if mv is not None:
                    for fonte, v in ref.get((cel, t, ss), []):
                        ok = abs(mv - v) <= TOL
                        conf.append({'celula': cel, 'run': lbl, 'fonte': fonte, 'recomputado': mv, 'agregado': v, 'absdiff': abs(mv - v), 'bateu': ok})
                        if not ok: diverg.append(lbl + ' ' + fonte)
            r['motivo_exclusao_validos'] = mot or None
            nt = any(r[t].get('nao_treinavel') for t in ('gnn', 'mlp'))
            sp = any(r[t].get('sem_validos_sem_npz') for t in ('gnn', 'mlp'))
            nt = nt or sp
            r['excluido_todos'] = nt
            if not nt:
                r['d_todos'] = r['mlp']['mae_todos'] - r['gnn']['mae_todos']
                if r['gnn']['n'] != r['mlp']['n']: raise SystemExit('ABORTA: n diferente GNN/MLP ' + str(ss))
            r['usado_validos'] = (not nt) and r['gnn']['n_validos'] > 0 and r['mlp']['n_validos'] > 0
            if r['usado_validos']:
                r['d_validos'] = r['mlp']['mae_validos'] - r['gnn']['mae_validos']
            linhas.append(r)
        por[cel] = linhas
        # coverage check: draws of the aggregator vs draws used
        refs = a1 if bl == 1 else a3
        agr_usados = sorted(int(s) for s in refs['celulas'][cel]['por_modelo']['gnn']['mae_validos_por_sorteio'])
        meus = sorted(r['split_seed'] for r in linhas if r['usado_validos'])
        if agr_usados != meus:
            diverg.append(f'{cel}: lista de sorteios validos difere do agregador {agr_usados} x {meus}')
        resumo_cel[cel] = {}
        for pop in ('validos', 'todos'):
            ds = np.array([r[f'd_{pop}'] for r in linhas if f'd_{pop}' in r], float)
            n = len(ds); mu = float(ds.mean()); sd = float(ds.std(ddof=1)) if n > 1 else None
            sgn = np.sign(mu)
            inv = bool(np.any(np.sign(ds) != sgn))
            resumo_cel[cel][pop] = {'n_sorteios': n, 'n_d_pos_gnn_lt_mlp': int((ds > 0).sum()), 'n_d_neg_mlp_lt_gnn': int((ds < 0).sum()),
                                    'n_abs_d_le_0_132': int((np.abs(ds) <= LIM).sum()), 'media_d': mu, 'dp_d': sd,
                                    'inversao': inv, 'n_sorteios_com_sinal_oposto_a_media': int((np.sign(ds) != sgn).sum()),
                                    'dp_ge_abs_media': (None if sd is None else bool(sd >= abs(mu))), 'dp_lt_metade_abs_media': (None if sd is None else bool(sd < abs(mu) / 2))}
        resumo_cel[cel]['excluidos'] = [{'split_seed': r['split_seed'], 'indice_sorteio': r['indice_sorteio'], 'motivo': r['motivo_exclusao_validos'],
                                         'excluido_tambem_de_todos': r['excluido_todos']} for r in linhas if not r['usado_validos']]
        log(cel, json.dumps({p: resumo_cel[cel][p] for p in ('validos', 'todos')}))
        log(cel, 'excluidos', json.dumps(resumo_cel[cel]['excluidos']))
    nbat = sum(c['bateu'] for c in conf)
    log(f'CONFERENCIA {nbat}/{len(conf)} bateram (tol {TOL}); maior absdiff {max(c["absdiff"] for c in conf):.3e}; divergencias: {diverg}')
    ncel = {c: sum(1 for x in conf if x['celula'] == c) for c, _ in CELULAS}
    log('comparacoes por celula', ncel)
    if diverg:
        log('ABORTA: divergencia com os agregados'); open(os.path.join(OUT, 'E6_run.log'), 'w').write('\n'.join(LOG) + '\n'); sys.exit(3)
    # mechanical count of the criterion thresholds
    cm = {'inversao_em_ao_menos_1_sorteio_em_ao_menos_1_celula_validos': any(resumo_cel[c]['validos']['inversao'] for c, _ in CELULAS),
          'dp_ge_abs_media_em_ao_menos_1_celula_validos': any(resumo_cel[c]['validos']['dp_ge_abs_media'] for c, _ in CELULAS),
          'nenhuma_inversao_e_dp_lt_metade_abs_media_em_todas_validos': (not any(resumo_cel[c]['validos']['inversao'] for c, _ in CELULAS)) and all(resumo_cel[c]['validos']['dp_lt_metade_abs_media'] for c, _ in CELULAS),
          'por_celula': {c: {'inversao': resumo_cel[c]['validos']['inversao'], 'dp_sobre_abs_media': resumo_cel[c]['validos']['dp_d'] / abs(resumo_cel[c]['validos']['media_d'])} for c, _ in CELULAS}}
    cm['ramo_reforca_mecanico'] = cm['inversao_em_ao_menos_1_sorteio_em_ao_menos_1_celula_validos'] or cm['dp_ge_abs_media_em_ao_menos_1_celula_validos']
    cm['ramo_neutro_mecanico'] = cm['nenhuma_inversao_e_dp_lt_metade_abs_media_em_todas_validos']
    log('LIMIARES', json.dumps(cm))
    script_sha = sha(os.path.abspath(__file__))
    cmd = ['python', SCRIPT_REL]
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump({'artefato': 'E6_por_sorteio', 'data_utc': now, 'por_celula': por}, open(os.path.join(OUT, 'E6_por_sorteio.json'), 'w'), indent=1)
    json.dump({'artefato': 'E6_resumo', 'data_utc': now, 'comando': cmd, 'script': SCRIPT_REL, 'script_sha256': script_sha,
               'criterio': os.path.relpath(CRIT, BASE), 'criterio_sha256': crit_sha,
               'rotina_mae_reusada': {'origem': 'analysis/v3_G1_agregar_v10.py', 'modo': 'extracao por ast de mae_pop/_finito/hash_idx/Aborta/SENTINELA_PL, sem executar o restante', 'sha256_origem': sha_agr},
               'limiar_ruido_repeticao_db': LIM, 'sha256_run_json_e_npz': runs_sha,
               'conferencia_agregados': {'tolerancia': TOL, 'n_comparacoes': len(conf), 'n_bateram': nbat, 'maior_absdiff': max(c['absdiff'] for c in conf), 'comparacoes_por_celula': ncel,
                                        'agregados_sha256': {f: sha(os.path.join(G1, f)) for f in ('agregado_G1_v5_bloco1.json', 'agregado_G1_v5_bloco2.json', 'agregado_G1_v8_bloco3.json', 'agregado_G1_v10_bloco4.json')},
                                        'detalhe': conf},
               'celulas': resumo_cel, 'contagem_mecanica_limiares': cm,
               'nota_populacoes': 'validos: sorteios com no valido nos dois modelos e treinaveis; todos: todos os sorteios treinaveis com predicao nos dois modelos (inclui sorteios sem no valido)'},
              open(os.path.join(OUT, 'E6_resumo.json'), 'w'), indent=1)
    # tex
    def fmt(x): return f'{x:.3f}'.replace('.', '{,}') if False else f'{x:.3f}'
    L = [r'\begin{tabular}{llrrrrcc}', r'\hline', r'Cell & Population & $n$ & GNN$<$MLP & MLP$<$GNN & $|d|\le 0.132$ & $\bar d \pm s_d$ (dB) & Inversion \\', r'\hline']
    for cel, _ in CELULAS:
        for pop, nome in (('validos', 'valid'), ('todos', 'all')):
            q = resumo_cel[cel][pop]
            L.append(f"{cel.replace('_', ' ')} & {nome} & {q['n_sorteios']} & {q['n_d_pos_gnn_lt_mlp']} & {q['n_d_neg_mlp_lt_gnn']} & {q['n_abs_d_le_0_132']} & ${q['media_d']:.2f} \\pm {q['dp_d']:.2f}$ & {'yes' if q['inversao'] else 'no'} \\\\")
    L += [r'\hline', r'\end{tabular}']
    open(os.path.join(OUT, 'T_E6_inversoes.tex'), 'w').write('\n'.join(L) + '\n')
    log('E6 fim ok')
    open(os.path.join(OUT, 'E6_run.log'), 'w').write('\n'.join(LOG) + '\n')
main()
