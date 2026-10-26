"""Estimand Drift under Spatial Block Hold-Out (MDPI Mathematics, revision 1).
V2: uncertainty of the variability comparisons (criteria/criterio_V2_incerteza_comparacoes_variabilidade.json).
Fixed method: bootstrap over draws per cell, B = 4000, seed 20261026, 95% percentile interval.
Draw indices are shared across sources (i), (iii), (iv) within a cell (paired); in each source, non-finite draws
(None: no valid node in the test partition) are dropped inside the replicate, as in the stamped artefact.
Reads results/ of this repository (R5 per-draw records replace the earlier per-cell partial files). numpy/json only.
Output directory: env R1_OUT (default results/fase6_rerun)."""
import json, hashlib, sys, datetime, os
import numpy as np
HERE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # repository root
OUT = os.environ.get('R1_OUT', os.path.join(ROOT, 'results', 'fase6_rerun'))
SCRIPT_REL = os.path.relpath(HERE, ROOT).replace(os.sep, '/')
B, SEED, TOL = 4000, 20261026, 0.05
def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), 'rb').read()).hexdigest()
def J(p):
    return json.load(open(os.path.join(ROOT, p)))
def log(*a):
    print(*a); sys.stdout.flush()
CEL = [f'{c}_Q{q}' for c in ('bauru', 'campinas', 'lins', 'sorocaba') for q in (1, 2, 3, 4)]
f2 = 'results/fase2/2.1_deriva_erro_baselines_16x60rnd.json'
fr1 = 'results/fase4/R1_b0_por_sorteio.json'; fr1r = 'results/fase4/R1_b0_resumo.json'
fe = 'results/fase5/E1_por_sorteio.json'; fer = 'results/fase5/E1_resumo.json'
fr5 = 'results/fase5/R5_por_sorteio.json'
CRIT = 'criteria/criterio_V2_incerteza_comparacoes_variabilidade.json'
D2, R1, R1r, E1, E1r, R5 = J(f2), J(fr1), J(fr1r), J(fe), J(fer), J(fr5)['celulas']
fnum = lambda v: np.nan if v is None else float(v)
X = {'b2': {}, 'b0': {}, 'kf': {}}
for c in CEL:
    p = R5[c]['sorteios']
    X['b2'][c] = np.array([fnum(s['constante__validos']['mae_orig_script_2.1']) for s in p])
    X['b0'][c] = np.array([fnum(s['mae_constante']['validos']) for s in R1['celulas'][c]['sorteios']])
    X['kf'][c] = np.array([fnum(s['Err_conjunto']['constante|validos|b2']) for s in E1['celulas'][c]['sorteios']])
    for k in X:
        assert len(X[k][c]) == 60, (k, c)
    sb2 = [s['split_seed'] for s in p]
    assert sb2 == [s['split_seed'] for s in E1['celulas'][c]['sorteios']] == [s['split_seed'] for s in R1['celulas'][c]['sorteios']], c
sd = lambda a: float(np.std(a[np.isfinite(a)], ddof=1))
mn = lambda a: float(np.mean(a[np.isfinite(a)]))
def stats(k):
    dps = np.array([sd(X[k][c]) for c in CEL]); mus = np.array([mn(X[k][c]) for c in CEL])
    return dps, mus
dp2, mu2 = stats('b2'); dp0, _ = stats('b0'); dpk, muk = stats('kf')
rep = {
 'i_media_dp_b2': float(dp2.mean()), 'ii_dp_entre_medias_b2': float(np.std(mu2, ddof=1)),
 'iii_media_dp_b0': float(dp0.mean()), 'iv_media_dp_kfold_b2': float(dpk.mean())}
rep['razao_i_ii'] = rep['i_media_dp_b2'] / rep['ii_dp_entre_medias_b2']
rep['razao_i_iv'] = rep['i_media_dp_b2'] / rep['iv_media_dp_kfold_b2']
conf = []
for i, c in enumerate(CEL):
    ref = D2['por_celula'][c]['constante_validos']['dp_simples']
    conf.append({'nome': 'dp_simples ' + c, 'recomputado': float(dp2[i]), 'artefato': ref, 'absdiff': abs(float(dp2[i]) - ref),
                 'bateu': abs(float(dp2[i]) - ref) <= TOL, 'fonte': f2 + ' :: por_celula.' + c + '.constante_validos.dp_simples'})
resD = D2['resumo']
def chk(nome, val, ref, fonte):
    conf.append({'nome': nome, 'recomputado': val, 'artefato': ref, 'absdiff': abs(val - ref), 'fonte': fonte, 'bateu': abs(val - ref) <= TOL})
chk('(i) media 16 DPs b=2', rep['i_media_dp_b2'], resD['dp_medio_entre_sorteios_validos_dB_simples'], f2 + ' :: resumo.dp_medio_entre_sorteios_validos_dB_simples')
chk('(ii) DP entre 16 medias', rep['ii_dp_entre_medias_b2'], resD['dp_entre_celulas_das_medias_validos_dB'], f2 + ' :: resumo.dp_entre_celulas_das_medias_validos_dB')
chk('(iii) media 16 DPs b=0', rep['iii_media_dp_b0'], R1r['medias_16_celulas']['dp_mae_constante_validos_b0'], fr1r + ' :: medias_16_celulas.dp_mae_constante_validos_b0')
chk('(iv) media 16 DPs K-fold', rep['iv_media_dp_kfold_b2'], E1r['limiares_do_criterio_contagem_mecanica']['b2_constante_validos']['dp_sorteios_medio_16_kfold'], fer + ' :: limiares_do_criterio_contagem_mecanica.b2_constante_validos.dp_sorteios_medio_16_kfold')
for r in conf:
    log('CONF', json.dumps(r))
if not all(r['bateu'] for r in conf):
    log('ABORTA: conferencia divergiu > 0,05 dB'); sys.exit(2)
log('REPRODUZIDO', json.dumps(rep))
rng = np.random.default_rng(SEED)
def sdm(M):
    m = np.isfinite(M); n = m.sum(1); Z = np.where(m, M, 0.0)
    mu = Z.sum(1) / n; v = (np.where(m, M - mu[:, None], 0.0) ** 2).sum(1) / (n - 1)
    return np.sqrt(v), mu
A = {k: np.stack([X[k][c] for c in CEL]) for k in X}
out = {k: [] for k in ('dp_cel_b2', 'i', 'ii', 'iii', 'iv', 'r_i_ii', 'r_i_iv', 'dp_cel_b0', 'dp_cel_kf')}
IDX = rng.integers(0, 60, size=(B, 16, 60))
for b in range(B):
    ix = IDX[b]
    r = {k: sdm(np.take_along_axis(A[k], ix, 1)) for k in A}
    out['dp_cel_b2'].append(r['b2'][0]); out['dp_cel_b0'].append(r['b0'][0]); out['dp_cel_kf'].append(r['kf'][0])
    i_, ii_, iii_, iv_ = r['b2'][0].mean(), np.std(r['b2'][1], ddof=1), r['b0'][0].mean(), r['kf'][0].mean()
    out['i'].append(i_); out['ii'].append(ii_); out['iii'].append(iii_); out['iv'].append(iv_)
    out['r_i_ii'].append(i_ / ii_); out['r_i_iv'].append(i_ / iv_)
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
res = {}
for k, nome in (('i', 'i_media_dp_b2'), ('ii', 'ii_dp_entre_medias_b2'), ('iii', 'iii_media_dp_b0'), ('iv', 'iv_media_dp_kfold_b2'), ('r_i_ii', 'razao_i_ii'), ('r_i_iv', 'razao_i_iv')):
    v = np.array(out[k]); res[nome] = {'estimativa': rep[nome], 'IC95_percentil': ci(v), 'media_boot': float(v.mean()), 'dp_boot': float(v.std(ddof=1))}
percel = {}
for j, c in enumerate(CEL):
    percel[c] = {}
    for k, key in (('b2', 'dp_cel_b2'), ('b0', 'dp_cel_b0'), ('kf', 'dp_cel_kf')):
        v = np.array(out[key])[:, j]; percel[c][k] = {'dp': sd(X[k][c]), 'IC95_percentil': ci(v), 'n_sorteios_finitos': int(np.isfinite(X[k][c]).sum())}
jk = np.array([np.std(np.delete(mu2, i), ddof=1) for i in range(16)])
n = 16; jse = float(np.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum()))
jkr = {'valores_deixa_uma_fora': {CEL[i]: float(jk[i]) for i in range(16)}, 'min': float(jk.min()), 'max': float(jk.max()),
       'pseudo_valor_media': float(n * rep['ii_dp_entre_medias_b2'] - (n - 1) * jk.mean()), 'EP_jackknife': jse,
       'IC95_normal_est_pm_1.96EP': [rep['ii_dp_entre_medias_b2'] - 1.96 * jse, rep['ii_dp_entre_medias_b2'] + 1.96 * jse]}
g = lambda k: res[k]['IC95_percentil']
cont = {
 'IC_i_dentro_7_a_9': bool(g('i_media_dp_b2')[0] >= 7 and g('i_media_dp_b2')[1] <= 9),
 'IC_ii_limite_superior_abaixo_de_4': bool(g('ii_dp_entre_medias_b2')[1] < 4),
 'IC_i_inteiramente_acima_IC_ii (inf_i > sup_ii)': bool(g('i_media_dp_b2')[0] > g('ii_dp_entre_medias_b2')[1]),
 'IC_razao_i_ii_exclui_1': bool(not (g('razao_i_ii')[0] <= 1 <= g('razao_i_ii')[1])),
 'IC_razao_i_ii_acima_de_2': bool(g('razao_i_ii')[0] > 2),
 'IC_razao_i_iv_exclui_1': bool(not (g('razao_i_iv')[0] <= 1 <= g('razao_i_iv')[1])),
 'IC_razao_i_iv_acima_de_5': bool(g('razao_i_iv')[0] > 5),
 'n_replicas_ii_ge_i': int(sum(1 for a, b in zip(out['ii'], out['i']) if a >= b)),
 'n_replicas_razao_i_ii_le_1': int(sum(1 for r in out['r_i_ii'] if r <= 1)),
 'n_replicas_razao_i_iv_le_1': int(sum(1 for r in out['r_i_iv'] if r <= 1))}
for k, v in cont.items():
    log('LIMIAR', k, v)
for k, v in res.items():
    log('IC', k, json.dumps(v))
log('JACKKNIFE', json.dumps({kk: vv for kk, vv in jkr.items() if kk != 'valores_deixa_uma_fora'}))
srcs = [f2, fr1, fr1r, fe, fer, fr5]
S = {
 'id': 'V2_resumo', 'criterio': CRIT,
 'criterio_sha256': sha(CRIT),
 'metodo': {'B': B, 'semente': SEED, 'gerador': 'numpy.random.default_rng', 'IC': 'percentil 2,5-97,5', 'dp': 'ddof=1',
            'reamostragem': 'indices de sorteio (60, com reposicao) por celula, compartilhados entre as fontes i, iii, iv; sorteios None (sem no valido) excluidos dentro da replica',
            'chave_kfold': 'results/fase5/E1_por_sorteio.json :: celulas.<cel>.sorteios[].Err_conjunto["constante|validos|b2"]'},
 'conferencias': conf, 'valores_reproduzidos': rep, 'ICs': res, 'por_celula': percel, 'jackknife_ii': jkr,
 'contagens_mecanicas_limiares': cont,
 'sha256_fontes': {p: sha(p) for p in srcs},
 'script': SCRIPT_REL, 'script_sha256': hashlib.sha256(open(HERE, 'rb').read()).hexdigest(),
 'data_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'comando': 'python ' + SCRIPT_REL}
os.makedirs(OUT, exist_ok=True)
json.dump(S, open(os.path.join(OUT, 'V2_resumo.json'), 'w'), indent=1, ensure_ascii=False)
log('GRAVADO V2_resumo.json em', OUT)
