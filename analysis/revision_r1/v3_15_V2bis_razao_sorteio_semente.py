"""Estimand Drift under Spatial Block Hold-Out (MDPI Mathematics, revision 1).
V2-bis: precision of the draw/seed ratios (criteria/criterio_V2bis_razao_sorteio_semente.json). CPU, JSON only, no npz.
MS_seed: decomposicao_um_fator extracted by ast from analysis/v3_G1_agregar_v10.py (digest of the file as read is recorded).
Cross-check (0), aborting (rc 3): 4 published R, 8 SD of Table 8, 3 F-interval of theta. The reference values are READ from
the SD column of tables_figures/tables_v3-12/T9_deriva_modelo_G1_v3-12.tex and from two sentences of the manuscript kept in
analysis/revision_r1/V2bis_manuscript_values.json (the manuscript source is not part of this repository).
Output directory: env R1_OUT (default results/fase6_rerun)."""
import ast, hashlib, json, os, re, sys, datetime
import numpy as np
from scipy import stats
HERE = os.path.abspath(__file__)
BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # repository root
SCRIPT_REL = os.path.relpath(HERE, BASE).replace(os.sep, '/')
AGR = os.path.join(BASE, 'analysis', 'v3_G1_agregar_v10.py')
CRIT = os.path.join(BASE, 'criteria', 'criterio_V2bis_razao_sorteio_semente.json')
G1 = os.path.join(BASE, 'results', 'gpu', 'G1')
T9 = os.path.join(BASE, 'tables_figures', 'tables_v3-12', 'T9_deriva_modelo_G1_v3-12.tex')
MAIN = os.path.join(os.path.dirname(HERE), 'V2bis_manuscript_values.json')
OUT = os.environ.get('R1_OUT', os.path.join(BASE, 'results', 'fase6_rerun'))
rl = lambda p: os.path.relpath(p, BASE).replace(os.sep, '/')
B, SEED, LIM = 4000, 20261027, 0.132
TOL_DP, TOL_IC = 0.005, 0.05
CELULAS = ['bauru_Q1', 'campinas_Q1', 'bauru_Q3', 'campinas_Q3']
NOME = {'bauru_Q1': 'Bauru Q1', 'campinas_Q1': 'Campinas Q1', 'bauru_Q3': 'Bauru Q3', 'campinas_Q3': 'Campinas Q3'}
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
def abort(msg):
    log('ABORTA (rc 3):', msg)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, 'V2bis_run.log'), 'w').write('\n'.join(LOG) + '\n')
    sys.exit(3)

sha_agr = sha(AGR)
tree = ast.parse(open(AGR, encoding='utf-8').read())
ns = {'np': np}
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == 'Aborta':
        exec(compile(ast.Module([node], []), AGR, 'exec'), ns)
    if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'N_SEMENTES_B2' for t in node.targets):
        exec(compile(ast.Module([node], []), AGR, 'exec'), ns)
    if isinstance(node, ast.FunctionDef) and node.name == 'decomposicao_um_fator':
        exec(compile(ast.Module([node], []), AGR, 'exec'), ns)
decomp = ns['decomposicao_um_fator']
# v10_bloco4 records the digest of the ORIGINAL aggregator (the published file was cleaned; see PROVENANCE.md): recorded, not asserted.

def ms(M):
    d = decomp(np.asarray(M, float))
    return None if d is None else (d['qm_entre'], d['qm_dentro'])
def theta(M):
    a, b = ms(M); return (a - b) / (3 * b)
def Rth(t): return float(np.sqrt(1 + max(t, 0.0)))
def ic_theta(M):
    M = np.asarray(M, float); n = M.shape[0]; a, b = ms(M); F = a / b
    lo = (F / stats.f.ppf(0.975, n - 1, 2 * n) - 1) / 3
    hi = (F / stats.f.ppf(0.025, n - 1, 2 * n) - 1) / 3
    return float(lo), float(hi)
def bca(stat0, boots, jack, alpha=0.05):
    boots = np.asarray(boots); jack = np.asarray(jack)
    p = (boots < stat0).mean() + 0.5 * (boots == stat0).mean()
    z0 = stats.norm.ppf(min(max(p, 1e-9), 1 - 1e-9))
    d = jack.mean() - jack; den = 6 * (np.sum(d ** 2)) ** 1.5
    a = float(np.sum(d ** 3) / den) if den > 0 else 0.0
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        zq = stats.norm.ppf(q); adj = stats.norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))
        out.append(float(np.quantile(boots, adj)))
    return out, float(z0), a

def main():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    log('V2bis inicio', now, 'agregador_sha256', sha_agr)
    # ---- fontes
    a1 = J(os.path.join(G1, 'agregado_G1_v5_bloco1.json')); a3 = J(os.path.join(G1, 'agregado_G1_v8_bloco3.json'))
    a2 = J(os.path.join(G1, 'agregado_G1_v5_bloco2.json')); a4 = J(os.path.join(G1, 'agregado_G1_v10_bloco4.json'))
    caminhos = {}
    dados = {}
    for cel in CELULAS:
        a, nm = (a1, 'agregado_G1_v5_bloco1.json') if cel.endswith('Q1') else (a3, 'agregado_G1_v8_bloco3.json')
        for t in ('gnn', 'mlp'):
            d42 = {str(k): float(v) for k, v in a['celulas'][cel]['por_modelo'][t]['mae_validos_por_sorteio'].items()}
            e = {'s42': d42}
            caminhos[f'{cel}.{t}.s42'] = f'{nm} :: celulas.{cel}.por_modelo.{t}.mae_validos_por_sorteio'
            if cel in ('bauru_Q1', 'campinas_Q1'):
                src, nm2 = (a2, 'agregado_G1_v5_bloco2.json') if cel == 'bauru_Q1' else (a4, 'agregado_G1_v10_bloco4.json')
                blk = src['celulas'][cel][t]
                ids = [str(i) for i in blk['sorteios_usados']]
                M = np.array(blk['matriz_mae_validos_sorteio_x_semente_42_43_44'], float)
                e['ids'], e['M'] = ids, M
                caminhos[f'{cel}.{t}.matriz'] = f'{nm2} :: celulas.{cel}.{t}.{{sorteios_usados, matriz_mae_validos_sorteio_x_semente_42_43_44}}'
            dados[(cel, t)] = e
    # ---- referencias lidas do .tex
    ref_dp, cel_atual = {}, None
    for ln in open(T9, encoding='utf-8'):
        if '&' not in ln or ln.strip().startswith('\\celula'): continue
        c = [x.strip().rstrip('\\').strip() for x in ln.split('&')]
        if len(c) < 8 or c[1] not in ('GNN', 'MLP'): continue
        if c[0]: cel_atual = [k for k, v in NOME.items() if v == c[0]][0]
        ref_dp[(cel_atual, c[1].lower())] = (float(c[5]), c[5])
    mt = [x for x in J(MAIN)['sentences']]  # [ratio sentence, interval sentence]
    lnR = 0
    mR = re.search(r'\$([\d.]+)\$ \(GNN\) and \$([\d.]+)\$ \(MLP\) in Bauru Q1 and \$([\d.]+)\$ and \$([\d.]+)\$ in Campinas', mt[lnR])
    refR = {('bauru_Q1', 'gnn'): mR.group(1), ('bauru_Q1', 'mlp'): mR.group(2), ('campinas_Q1', 'gnn'): mR.group(3), ('campinas_Q1', 'mlp'): mR.group(4)}
    lnI = 1
    mI = re.findall(r'\[([\d.]+);\\,([\d.]+)\]', mt[lnI] + ' ' + mt[lnI + 1] if False else mt[lnI])
    refIC = dict(zip([('bauru_Q1', 'gnn'), ('campinas_Q1', 'gnn'), ('campinas_Q1', 'mlp')], mI))
    log('referencias lidas de', rl(T9), '(coluna SD) e', rl(MAIN), 'sentencas 1 (R) e 2 (IC F)')
    # ---- (0) conferencia
    diverg = []; conf = {'R': {}, 'DP': {}, 'ICF': {}}
    for k, s in refR.items():
        dec = len(s.split('.')[1]); tol = max(TOL_DP, 0.5 * 10 ** -dec)
        e = dados[k]; v = np.array(list(e['s42'].values())); r20 = float(v.std(ddof=1) / np.sqrt(ms(e['M'])[1]))
        ok = abs(r20 - float(s)) <= tol; conf['R'][f'{k[0]}.{k[1]}'] = {'ref': s, 'obtido': r20, 'tol': tol, 'ok': ok}
        if not ok: diverg.append(('R', k, s, r20))
    for k, (rv, rs) in ref_dp.items():
        dp = float(np.std(list(dados[k]['s42'].values()), ddof=1)); ok = abs(dp - rv) <= TOL_DP
        conf['DP'][f'{k[0]}.{k[1]}'] = {'ref': rs, 'obtido': dp, 'tol': TOL_DP, 'ok': ok}
        if not ok: diverg.append(('DP', k, rs, dp))
    for k, (lo, hi) in refIC.items():
        l, h = ic_theta(dados[k]['M']); ok = abs(l - float(lo)) <= TOL_IC and abs(h - float(hi)) <= TOL_IC
        conf['ICF'][f'{k[0]}.{k[1]}'] = {'ref': [lo, hi], 'obtido': [l, h], 'tol': TOL_IC, 'ok': ok}
        if not ok: diverg.append(('ICF', k, (lo, hi), (l, h)))
    log('conferencia (0):', json.dumps(conf, ensure_ascii=False))
    if len(ref_dp) != 8 or len(refR) != 4 or len(refIC) != 3: diverg.append(('contagem de referencias', len(ref_dp), len(refR), len(refIC)))
    if diverg: abort(f'conferencia (0) divergente: {diverg}')
    log('conferencia (0): OK (4 R, 8 DP, 3 IC F)')
    # ---- (a)-(d) pares cruzados
    pares = {}
    for cel in ('bauru_Q1', 'campinas_Q1'):
        for t in ('gnn', 'mlp'):
            e = dados[(cel, t)]; M, ids = e['M'], e['ids']; n = M.shape[0]
            qe, qd = ms(M); th = theta(M); lo, hi = ic_theta(M)
            R = Rth(th); Rlo, Rhi = Rth(lo), Rth(hi)
            s42 = e['s42']; v = np.array(list(s42.values())); dp20 = float(v.std(ddof=1)); R20 = dp20 / np.sqrt(qd)
            fl, fh = stats.f.ppf(0.975, 19, 2 * n), stats.f.ppf(0.025, 19, 2 * n)
            R20ic = [float(R20 / np.sqrt(fl)), float(R20 / np.sqrt(fh))]
            loo = []
            for i in range(n):
                Mi = np.delete(M, i, axis=0); ti = theta(Mi); loo.append({'id_fora': ids[i], 'theta': ti, 'R': Rth(ti)})
            lt = [x['theta'] for x in loo]; lr = [x['R'] for x in loo]
            mov = max(loo, key=lambda x: abs(x['theta'] - th))
            # joint bootstrap by draw id
            keys = list(s42.keys()); vals = np.array([s42[k] for k in keys]); pos = {k: i for i, k in enumerate(keys)}
            rowof = {pos[i]: r for r, i in enumerate(ids)}
            def stat(cnt):
                vv = np.repeat(vals, cnt)
                if len(vv) < 2: return None
                rows = [r for p, r in rowof.items() for _ in range(int(cnt[p]))]
                if len(rows) < 2: return None
                dq = ms(M[rows])[1]
                if dq <= 0: return None
                return float(vv.std(ddof=1) / np.sqrt(dq))
            obs = stat(np.ones(len(keys), int)); rng = np.random.default_rng(SEED)
            bt, desc = [], 0
            for _ in range(B):
                cnt = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys))
                s = stat(cnt)
                if s is None: desc += 1
                else: bt.append(s)
            jk = [stat(np.delete(np.ones(len(keys), int), j).tolist() and np.where(np.arange(len(keys)) == j, 0, 1)) for j in range(len(keys))]
            jk = [x for x in jk if x is not None]
            ci_bca, z0, acc = bca(obs, bt, jk); ci_pc = [float(np.quantile(bt, .025)), float(np.quantile(bt, .975))]
            pares[(cel, t)] = {'n_sorteios_cruzados': n, 'ms_sorteio': qe, 'ms_semente': qd, 'F': qe / qd, 'theta_hat': th, 'theta_ic_F95': [lo, hi],
                'R': R, 'R_ic': [Rlo, Rhi], 'R_ic_nota': 'R = sqrt(1+max(theta,0)); limite inferior truncado em 1 quando theta_inf < 0',
                'R20': R20, 'R20_ic_F_19_2n': R20ic, 'dp20_s42': dp20, 'n_s42': len(v),
                'deixa_um_fora': {'theta_min': min(lt), 'theta_max': max(lt), 'R_min': min(lr), 'R_max': max(lr), 'sorteio_que_mais_move': mov['id_fora'], 'detalhe': loo},
                'bootstrap_por_id': {'B': B, 'semente': SEED, 'observado_R20': obs, 'media_boot': float(np.mean(bt)), 'ic_bca': ci_bca, 'ic_percentil': ci_pc,
                    'z0': z0, 'aceleracao': acc, 'replicas_descartadas': desc, 'fracao_descartada': desc / B}}
            log(f'{cel} {t}: theta={th:.3f} IC F [{lo:.3f};{hi:.3f}] R={R:.3f} [{Rlo:.3f};{Rhi:.3f}] R20={R20:.3f} [{R20ic[0]:.2f};{R20ic[1]:.2f}] '
                f'LOO R {min(lr):.2f}-{max(lr):.2f} mov={mov["id_fora"]} BCa {ci_bca} desc={desc/B:.4f}')
    # ---- (e) 8 DP
    dps = {}
    for cel in CELULAS:
        for t in ('gnn', 'mlp'):
            v = np.array(list(dados[(cel, t)]['s42'].values())); n = len(v); s = float(v.std(ddof=1))
            c = [float(s * np.sqrt((n - 1) / stats.chi2.ppf(q, n - 1))) for q in (0.975, 0.025)]
            rng = np.random.default_rng(SEED); bt = np.array([v[rng.integers(0, n, n)].std(ddof=1) for _ in range(B)])
            jk = np.array([np.delete(v, j).std(ddof=1) for j in range(n)])
            cb, z0, acc = bca(s, bt, jk)
            dps[(cel, t)] = {'n': n, 'dp': s, 'ic_qui2_95': c, 'ic_bca_95': cb, 'ic_percentil_95': [float(np.quantile(bt, .025)), float(np.quantile(bt, .975))], 'media_boot': float(bt.mean()),
                'razao_ao_0_132': s / LIM, 'ic_qui2_sobre_0_132': [x / LIM for x in c], 'ic_bca_sobre_0_132': [x / LIM for x in cb],
                'nota': '0,132 dB e o maximo de duas repeticoes, nao um estimador com variancia; so o numerador tem IC'}
            log(f'DP {cel} {t}: n={n} {s:.3f} chi2 [{c[0]:.3f};{c[1]:.3f}] BCa [{cb[0]:.3f};{cb[1]:.3f}] /0.132 chi2 [{c[0]/LIM:.1f};{c[1]/LIM:.1f}]')
    # ---- contagens mecanicas
    ths = [p['theta_ic_F95'] for p in pares.values()]
    cont = {'n_pares': 4,
        'limite_inferior_theta_maior_que_1': sum(l > 1 for l, h in ths),
        'ic_exclui_0_mas_nao_1': sum(l > 0 and l <= 1 <= h for l, h in ths),
        'ic_inclui_0': sum(l <= 0 for l, h in ths),
        'por_par': {f'{k[0]}.{k[1]}': {'theta_inf': p['theta_ic_F95'][0], 'inf_gt_1': p['theta_ic_F95'][0] > 1, 'exclui_0_nao_1': 0 < p['theta_ic_F95'][0] <= 1 <= p['theta_ic_F95'][1], 'inclui_0': p['theta_ic_F95'][0] <= 0} for k, p in pares.items()}}
    log('contagens:', json.dumps(cont))
    # ---- saidas
    os.makedirs(OUT, exist_ok=True)
    cmd = ' '.join(['python', SCRIPT_REL] + sys.argv[1:])
    fontes = {f: sha(os.path.join(G1, f)) for f in ('agregado_G1_v5_bloco1.json', 'agregado_G1_v5_bloco2.json', 'agregado_G1_v8_bloco3.json', 'agregado_G1_v10_bloco4.json')}
    fontes.update({rl(T9): sha(T9), rl(MAIN): sha(MAIN)})
    k2 = lambda d: {f'{a}.{b}': v for (a, b), v in d.items()}
    res = {'artefato': 'V2bis_resumo', 'data_utc': now, 'comando': cmd, 'script': SCRIPT_REL, 'script_sha256': sha(HERE),
        'criterio': rl(CRIT), 'criterio_sha256': sha(CRIT), 'rotina_ms_semente_reusada': {'origem': 'analysis/v3_G1_agregar_v10.py::decomposicao_um_fator', 'modo': 'extracao por ast', 'sha256_origem': sha_agr},
        'fontes_sha256': fontes, 'caminhos_json_usados': caminhos, 'referencias_lidas_de': {'DP': rl(T9), 'R_e_ICF': rl(MAIN)},
        'conferencia_0': conf, 'parametros': {'B': B, 'semente': SEED, 'tol_dp': TOL_DP, 'tol_ic_theta': TOL_IC, 'tol_R': 'max(0,005; meia unidade da ultima casa publicada)'},
        'pares_cruzados': k2(pares), 'dp_entre_sorteios_8': k2(dps), 'contagens_limiares': cont}
    json.dump(res, open(os.path.join(OUT, 'V2bis_resumo.json'), 'w'), indent=1, ensure_ascii=False)
    f = lambda x, d=2: f'{x:.{d}f}'
    L = ['% generated by analysis/revision_r1/v3_15_V2bis_razao_sorteio_semente.py', '\\begin{tabular}{llcccccc}', '\\toprule',
         'Cell & Model & $\\hat\\theta$ [$F$ 95\\%] & $R$ [95\\%] & $R_{20}$ [$F$ 95\\%] & Leave-one-draw-out $R$ & BCa by id $R_{20}$ [95\\%] & Discarded \\\\', '\\midrule']
    for (cel, t), p in pares.items():
        b = p['bootstrap_por_id']
        L.append(f"{NOME[cel]} & {t.upper()} & {f(p['theta_hat'])} [{f(p['theta_ic_F95'][0])}; {f(p['theta_ic_F95'][1])}] & {f(p['R'])} [{f(p['R_ic'][0])}; {f(p['R_ic'][1])}] & "
                 f"{f(p['R20'])} [{f(p['R20_ic_F_19_2n'][0])}; {f(p['R20_ic_F_19_2n'][1])}] & {f(p['deixa_um_fora']['R_min'])}--{f(p['deixa_um_fora']['R_max'])} & "
                 f"[{f(b['ic_bca'][0])}; {f(b['ic_bca'][1])}] & {f(100*b['fracao_descartada'],1)}\\% \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(OUT, 'T_V2bis_razoes.tex'), 'w').write('\n'.join(L) + '\n')
    log('V2bis fim')
    open(os.path.join(OUT, 'V2bis_run.log'), 'w').write('\n'.join(LOG) + '\n')
if __name__ == '__main__':
    main()
