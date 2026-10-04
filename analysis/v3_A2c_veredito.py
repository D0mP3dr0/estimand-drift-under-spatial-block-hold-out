#!/usr/bin/env python3
"""Run-to-run reproducibility check of the model campaign (repeat test A2c).

Each of four configurations (GNN and MLP, Bauru Q1 and Lins Q1) was trained twice
with identical settings (rep1, rep2). From the two run JSONs and the test-set .npz
files of each pair, the script computes the RSSI-channel test MAE on valid nodes, on
sentinel nodes and on all nodes, the absolute difference between repeats, the
selected checkpoint's test metrics and best epoch, the GradScaler diagnostics and
skipped steps, the test-partition hash and node indices, and the run time. A pair
passes when every absolute MAE difference is at most 1e-3 dB; the summary reports
the largest difference and the number of passing pairs.

Inputs: results/gpu/A2c/<model>_v3_a2c_<city>_Q1_rep<k>/ (run_*.json and *.npz) and the
decision criterion criterio_A2c.json. Output: results/gpu/A2c/veredito_A2c.json.

Usage: python v3_A2c_veredito.py
"""
import json, glob, hashlib, numpy as np
A='/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/gpu/A2c/'
crit=json.load(open('/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/criterios/criterio_A2c.json'))
def run(lbl):
    """Run JSON and test .npz (or None) of one run folder."""
    d=json.load(open(glob.glob(A+lbl+'/run_*.json')[0])); npz=glob.glob(A+lbl+'/*.npz'); return d, (np.load(npz[0]) if npz else None)
def mae_pop(z):
    """Test MAE (dB) of output channel 3 (RSSI) on valid nodes, sentinel nodes and all nodes."""
    y=z['target'][:,3]; p=z['pred'][:,3]; s=z['sentinela'].astype(bool); e=np.abs(y-p)
    return {'validos': float(e[~s].mean()) if (~s).any() else None, 'sentinela': float(e[s].mean()) if s.any() else None, 'todos': float(e.mean()), 'n_validos': int((~s).sum()), 'n': int(len(e))}
out={'criterio':crit.get('itens') or crit, 'pares':{}, 'resumo':{}}
tol=1e-3; piores=[]  # reproducibility tolerance on the absolute MAE difference (dB)
for cid in ('bauru','lins'):
    for mod in ('gnn','mlp'):
        r=[run(f'{mod}_v3_a2c_{cid}_Q1_rep{k}') for k in (1,2)]
        d1,z1=r[0]; d2,z2=r[1]
        m1,m2=mae_pop(z1),mae_pop(z2)
        delta={k: (abs(m1[k]-m2[k]) if m1[k] is not None and m2[k] is not None else None) for k in ('validos','sentinela','todos')}
        sel={k: [d1['selecao']['test_no_melhor_ckpt'].get(k), d2['selecao']['test_no_melhor_ckpt'].get(k)] for k in ('mae_rssi_db','mae_pl_db')}
        dt=[d.get('diagnostico_treino',{}) for d in (d1,d2)]
        hashes=[d.get('particoes',{}).get('test',{}).get('idx_sha256_global') for d in (d1,d2)]
        idx_eq = bool(np.array_equal(z1['idx_global'], z2['idx_global']))
        ds=[d.get('diagnostico_saida_canais',{}) for d in (d1,d2)]
        par={'mae_rep1':m1,'mae_rep2':m2,'delta_abs_db':delta,'selecao_test':sel,'melhor_epoca':[d1['selecao'].get('melhor_epoca'),d2['selecao'].get('melhor_epoca')],
             'gradscaler':[{k:v for k,v in x.items() if not isinstance(v,(list,dict))} for x in dt], 'pulos_identicos': (dt[0].get('pulos_indices')==dt[1].get('pulos_indices')) if dt[0].get('pulos_indices') is not None else None,
             'hash_particao_teste':hashes,'hash_iguais':hashes[0]==hashes[1] and hashes[0] is not None,'idx_npz_iguais':idx_eq,
             'diagnostico_saida_rep1':{k:v for k,v in ds[0].items()} if ds[0] else None,
             'tempo_s':[round(d1['custo']['tempo_total_s']),round(d2['custo']['tempo_total_s'])],
             'passa_1e-3_dB': all(v is not None and v<=tol for v in delta.values())}
        out['pares'][f'{cid}_Q1_{mod}']=par; piores.append(max(v for v in delta.values() if v is not None))
out['resumo']={'delta_abs_max_db':max(piores),'pares_que_passam_1e-3':sum(p['passa_1e-3_dB'] for p in out['pares'].values()),'n_pares':len(out['pares']),
 'veredito':'REPRODUTIBILIDADE_NAO_ATINGIDA_NAO_DETERMINISMO' if max(piores)>tol else 'REPRODUTIVEL'}
json.dump(out,open(A+'veredito_A2c.json','w'),indent=1,ensure_ascii=False)
for k,p in out['pares'].items():
    print(k,'| Δ válidos %.4f sent %.4f todos %.4f dB | sel rssi %s | ep %s | hash iguais %s | idx iguais %s | pulos idênticos %s | t %s' % (p['delta_abs_db']['validos'] or -1, p['delta_abs_db']['sentinela'] or -1, p['delta_abs_db']['todos'], [round(x,4) for x in p['selecao_test']['mae_rssi_db']], p['melhor_epoca'], p['hash_iguais'], p['idx_npz_iguais'], p['pulos_identicos'], p['tempo_s']))
print('RESUMO', json.dumps(out['resumo']))
print('gradscaler rep1 gnn bauru:', json.dumps(out['pares']['bauru_Q1_gnn']['gradscaler'][0])[:400])
print('saida canais rep1 gnn bauru:', json.dumps(out['pares']['bauru_Q1_gnn']['diagnostico_saida_rep1'])[:600])
