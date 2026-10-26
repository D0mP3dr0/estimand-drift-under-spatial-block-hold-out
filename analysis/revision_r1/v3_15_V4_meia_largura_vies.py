"""Estimand Drift under Spatial Block Hold-Out (MDPI Mathematics, revision 1).
V4: half-width of the bias by term (criteria/criterio_V4_meia_largura_vies.json). READ-ONLY on stamped records;
no new draw. Interval = estimate +- 2 SE, or [T_inf, T_sup] when closed-form; distinguido_de_zero = interval excludes 0.
json only. Output directory: env R1_OUT (default results/fase6_rerun)."""
import json, hashlib, os, sys, datetime
HERE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # repository root
OUT = os.environ.get('R1_OUT', os.path.join(ROOT, 'results', 'fase6_rerun'))
SCRIPT_REL = os.path.relpath(HERE, ROOT).replace(os.sep, '/')
sha = lambda p: hashlib.sha256(open(os.path.join(ROOT, p), 'rb').read()).hexdigest()
J = lambda p: json.load(open(os.path.join(ROOT, p)))
def log(*a):
    print(*a); sys.stdout.flush()
CEL = ['bauru_Q1', 'campinas_Q1', 'lins_Q1', 'sorocaba_Q1']
F_R3 = 'results/fase4/R3_termo_desenho.json'; F_R2 = 'results/fase4/R2_resumo.json'
F_FM = 'results/fase4/recalculo_R2_R4/fismat_R2_termo_razao.json'
CRIT = 'criteria/criterio_V4_meia_largura_vies.json'
R3, R2, FM = J(F_R3), J(F_R2), J(F_FM)
PRED = {'constante': {'r3': 'constante', 'r2': 'constante', 'fm': 'constante'},
        'fspl_b': {'r3': 'fspl_calibrado_b', 'r2': 'fspl_calibrado_b', 'fm': 'fspl_calibrado_b'}}
POP = ['validos', 'todos']
def pm(est, ep, caminho, fonte):
    lo, hi = est - 2 * ep, est + 2 * ep
    return {'estimativa': est, 'EP': ep, 'meia_largura_2EP': 2 * ep, 'intervalo': [lo, hi], 'distinguido_de_zero': bool(lo > 0 or hi < 0), 'fonte': fonte, 'caminho_json': caminho}
linhas = []
for c in CEL:
    for pn, pr in PRED.items():
        for pop in POP:
            L = {'celula': c, 'preditor': pn, 'populacao': pop, 'termos': {}}
            # design term: closed-form bounds (primary reading L_no_a_no, m_nodal)
            r3 = R3['por_celula'][c]['resultados'][f"{pr['r3']}|{pop}"]
            fe = r3['fechado']['m_nodal']['L_no_a_no']
            ti, ts = fe['T_inf_dB'], fe['T_sup_dB']
            base3 = f"por_celula.{c}.resultados['{pr['r3']}|{pop}']"
            L['termos']['desenho_fechado'] = {'estimativa': (ti + ts) / 2, 'meia_largura': (ts - ti) / 2, 'T_inf': ti, 'T_sup': ts, 'intervalo': [ti, ts],
                'distinguido_de_zero': bool(ti > 0 or ts < 0), 'estimativa_e_ponto_medio_das_cotas': True,
                'fonte': F_R3, 'caminho_json': base3 + '.fechado.m_nodal.L_no_a_no.{T_inf_dB,T_sup_dB}'}
            f2 = r3['frequencia_200']
            L['termos']['desenho_freq200'] = pm(f2['T_freq200_dB'], f2['ep_bootstrap_200'], base3 + '.frequencia_200.{T_freq200_dB,ep_bootstrap_200}', F_R3)
            # total bias (R2): H with buffer b=2; H0 without buffer
            r2 = R2['resumo_por_celula'][c][pr['r2']][pop]['estimadores']
            b2 = f"resumo_por_celula.{c}.{pr['r2']}.{pop}.estimadores"
            L['termos']['vies_total_H_com_buffer'] = pm(r2['H']['vies'], r2['H']['ep_mc_vies'], b2 + '.H.{vies,ep_mc_vies}', F_R2)
            L['termos']['vies_total_H0_sem_buffer'] = pm(r2['H0']['vies'], r2['H0']['ep_mc_vies'], b2 + '.H0.{vies,ep_mc_vies}', F_R2)
            # ratio term (fismat_R2_termo_razao.json): H with buffer, H0 without buffer
            for tag, key in (('razao_H_com_buffer', 'H'), ('razao_H0_sem_buffer', 'H0')):
                k = f"{key}_{pr['fm']}_{pop}"
                if k in FM[c]:
                    e = FM[c][k]
                    t = pm(e['termo_razao'], e['ep_boot_razao'], f"{c}.{k}.{{termo_razao,ep_boot_razao}}", F_FM)
                    t['vies_no_registro_fismat'] = e['vies']; t['z_razao'] = e['z_razao']; t['n'] = e['n']
                    L['termos'][tag] = t
                else:
                    L['termos'][tag] = {'ausente_no_artefato': True, 'chave_procurada': f"{c}.{k}", 'fonte': F_FM}
            linhas.append(L)
# source of the -1.23 dB value
fm_s = FM['sorocaba_Q1']['H_constante_validos']
fonte123 = {'valor_no_parecer': -1.23, 'item_do_parecer': 'stored records',
            'caminho_citado_no_parecer': 'results/fase4/recalculo_R2_R4/fismat_R2_termo_razao.json -> sorocaba_Q1.H_constante_validos.termo_razao',
            'artefato_localizado': F_FM, 'sha256': sha(F_FM), 'caminho_json': 'sorocaba_Q1.H_constante_validos.{termo_razao,ep_boot_razao,z_razao,termo_desenho_empirico_MC}',
            'termo_razao': fm_s['termo_razao'], 'ep_boot_razao': fm_s['ep_boot_razao'], 'z_razao': fm_s['z_razao'], 'termo_desenho_empirico_MC': fm_s['termo_desenho_empirico_MC'],
            'reproduz_-1,23_a_0,01': abs(fm_s['termo_razao'] - (-1.23)) <= 0.01,
            'nota': 'generator script: results/fase4/recalculo_R2_R4/fismat_R2_termo_razao.py (recalculation of 01/10/2026); not an R2_resumo; H = with buffer b=2'}
# cross-checks against the values printed in the text
def faixa(pn, pop, campo):
    v = [next(l for l in linhas if l['celula'] == c and l['preditor'] == pn and l['populacao'] == pop)['termos']['desenho_fechado'][campo] for c in CEL]
    return min(v) if campo == 'T_inf' else max(v)
txt = {'abstract_main.tex_l83_e_l708': {}}
tx = txt['abstract_main.tex_l83_e_l708']
alvo = {('constante', 'validos'): (0.180, 0.652), ('fspl_b', 'validos'): (-0.487, 0.161), ('constante', 'todos'): (0.008, 0.070), ('fspl_b', 'todos'): (-0.087, 0.029)}
ok_all = True
tx['faixas_desenho_fechado_vs_texto_l708'] = {}
for (pn, pop), (a, b) in alvo.items():
    mi, ma = faixa(pn, pop, 'T_inf'), faixa(pn, pop, 'T_sup')
    d = max(abs(mi - a), abs(ma - b)); ok = d <= 0.01; ok_all &= ok
    tx['faixas_desenho_fechado_vs_texto_l708'][f'{pn}|{pop}'] = {'min_T_inf': mi, 'max_T_sup': ma, 'texto': [a, b], 'absdiff_max': d, 'reproduz_a_0,01': ok}
mx_todos = max(abs(x) for pn, pop in alvo if pop == 'todos' for x in (faixa(pn, pop, 'T_inf'), faixa(pn, pop, 'T_sup')))
mx_val = max(abs(x) for pn, pop in alvo if pop == 'validos' for x in (faixa(pn, pop, 'T_inf'), faixa(pn, pop, 'T_sup')))
tx['frase_1_design_abaixo_de_0,1_todos_e_0,7_validos'] = {'max_abs_cota_todos': mx_todos, 'max_abs_cota_validos': mx_val, 'reproduzida': bool(mx_todos < 0.1 and mx_val < 0.7 and ok_all)}
const_val = [l['termos']['vies_total_H_com_buffer'] for l in linhas if l['preditor'] == 'constante' and l['populacao'] == 'validos']
const_all = [l['termos']['vies_total_H_com_buffer'] for l in linhas if l['preditor'] == 'constante' and l['populacao'] == 'todos']
tx['vies_total_constante_indistinguivel_de_zero'] = {'validos_n_distinguidas': sum(t['distinguido_de_zero'] for t in const_val), 'todos_n_distinguidas': sum(t['distinguido_de_zero'] for t in const_all),
                                                     'reproduzida': not any(t['distinguido_de_zero'] for t in const_val + const_all)}
rz = [l['termos']['razao_H0_sem_buffer']['estimativa'] for l in linhas if l['preditor'] == 'fspl_b' and l['populacao'] == 'validos']
tx['frase_2_razao_fspl_sem_buffer_meio_decibel'] = {'valores_por_celula': dict(zip(CEL, rz)), 'min': min(rz), 'max': max(rz), 'texto_W': [0.46, 0.64],
    'reproduzida_a_0,01': bool(abs(min(rz) - 0.46) <= 0.01 and abs(max(rz) - 0.64) <= 0.01)}
rzH = {c: next(l for l in linhas if l['celula'] == c and l['preditor'] == 'fspl_b' and l['populacao'] == 'validos')['termos']['razao_H_com_buffer'] for c in CEL}
tx['razao_fspl_com_buffer_validos_(parecer: 0,31-0,96)'] = {c: t.get('estimativa') for c, t in rzH.items()}
vl = [l for l in linhas if l['populacao'] == 'validos']
resumo = {'n_linhas_validos': len(vl), 'por_termo_n_distinguidas_validos': {k: sum(1 for l in vl if l['termos'][k].get('distinguido_de_zero')) for k in vl[0]['termos']},
          'por_termo_n_distinguidas_todos': {k: sum(1 for l in linhas if l['populacao'] == 'todos' and l['termos'][k].get('distinguido_de_zero')) for k in vl[0]['termos']},
          'maior_meia_largura_vies_total_H_constante_validos': max(l['termos']['vies_total_H_com_buffer']['meia_largura_2EP'] for l in vl if l['preditor'] == 'constante'),
          'maior_meia_largura_vies_total_H0_constante_validos': max(l['termos']['vies_total_H0_sem_buffer']['meia_largura_2EP'] for l in vl if l['preditor'] == 'constante'),
          'maior_EP_vies_total_H_constante_validos': max(l['termos']['vies_total_H_com_buffer']['EP'] for l in vl if l['preditor'] == 'constante')}
# consistency of the fismat record with R2 (bias H)
coer = []
for l in linhas:
    t, f = l['termos']['vies_total_H_com_buffer'], l['termos']['razao_H_com_buffer']
    if 'vies_no_registro_fismat' in f:
        coer.append({'celula': l['celula'], 'preditor': l['preditor'], 'pop': l['populacao'], 'vies_R2': t['estimativa'], 'vies_fismat': f['vies_no_registro_fismat'], 'absdiff': abs(t['estimativa'] - f['vies_no_registro_fismat'])})
S = {'id': 'V4_tabela_vies', 'criterio': CRIT, 'criterio_sha256': sha(CRIT),
     'regra': 'intervalo = estimativa +- 2 EP (ou [T_inf, T_sup] se fechado); distinguido_de_zero = intervalo exclui 0; so leitura', 'leitura_primaria_desenho': 'm_nodal.L_no_a_no (R3.leitura_primaria)',
     'rotulos': {'H': 'estimador de Hajek com buffer b=2', 'H0': 'sem buffer (b=0)', 'fspl_b': 'fspl_calibrado_b nos artefatos'},
     'linhas': linhas, 'fonte_menos_1_23': fonte123, 'conferencias_texto': txt, 'resumo_contagens': resumo, 'coerencia_vies_R2_vs_fismat_H': {'max_absdiff': max(x['absdiff'] for x in coer), 'linhas': coer},
     'sha256_fontes': {F_R3: sha(F_R3), F_R2: sha(F_R2), F_FM: sha(F_FM)}, 'script': SCRIPT_REL, 'script_sha256': hashlib.sha256(open(HERE, 'rb').read()).hexdigest(),
     'data_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'comando': 'python ' + SCRIPT_REL}
os.makedirs(OUT, exist_ok=True)
json.dump(S, open(os.path.join(OUT, 'V4_tabela_vies.json'), 'w'), indent=1, ensure_ascii=False)
# LaTeX
def f(t, nd=2):
    if t.get('ausente_no_artefato'):
        return '--'
    if 'T_inf' in t:
        s = f"{t['estimativa']:.{nd}f}$\\pm${t['meia_largura']:.{nd}f}"
    else:
        s = f"{t['estimativa']:.{nd}f}$\\pm${t['meia_largura_2EP']:.{nd}f}"
    return s + ('$^{*}$' if t['distinguido_de_zero'] else '')
tex = ['% generated by analysis/revision_r1/v3_15_V4_meia_largura_vies.py from V4_tabela_vies.json (read-only)',
       '% estimativa $\\pm$ meia-largura (dB); * = intervalo exclui zero; desenho fechado = ponto medio $\\pm$ meia-largura de [T_inf,T_sup]; demais = $\\pm 2$ EP',
       '\\begin{tabular}{lllccccc}', '\\hline', 'Cell & Predictor & Nodes & Design (closed) & Design (freq.\\ 200) & Total bias, buffer & Ratio term, buffer & Ratio term, no buffer \\\\', '\\hline']
for l in linhas:
    T = l['termos']
    tex.append(' & '.join([l['celula'].replace('_', '\\_'), l['preditor'].replace('_', '\\_'), l['populacao'], f(T['desenho_fechado']), f(T['desenho_freq200']), f(T['vies_total_H_com_buffer']), f(T['razao_H_com_buffer']), f(T['razao_H0_sem_buffer'])]) + ' \\\\')
tex += ['\\hline', '\\end{tabular}']
open(os.path.join(OUT, 'T_V4_vies.tex'), 'w').write('\n'.join(tex) + '\n')
log('FONTE -1,23', json.dumps(fonte123, ensure_ascii=False))
log('CONF_TEXTO', json.dumps(txt, ensure_ascii=False))
log('RESUMO', json.dumps(resumo))
log('COERENCIA max absdiff vies R2 vs fismat H', S['coerencia_vies_R2_vs_fismat_H']['max_absdiff'])
for l in linhas:
    log('LINHA', l['celula'], l['preditor'], l['populacao'], ' | '.join(f"{k}={f(v)}" for k, v in l['termos'].items()))
log('GRAVADO V4_tabela_vies.json e T_V4_vies.tex em', OUT)
