"""Recalculation of model errors from the saved per-node predictions of campaigns A2c, A3 and A4.

All errors are mean absolute errors on target channel 3 (RSSI, dB), computed from the
`predicoes_*.npz` file of each run (arrays `pred`, `target`, `sentinela`) and split into valid
and sentinel nodes. The constant predictor is the value -110 dBm. Keys of the output:
  checagem_s101  one A4 run recomputed next to the values stored in its aggregate (consistency);
  a_*            median over training seeds 42-46 of the GNN minus MLP sentinel MAE per A3 cell,
                 and the absolute GNN difference between the two repeat runs of A2c
                 (Bauru Q1 and Lins Q1), on valid and on sentinel nodes;
  b*             range and standard deviation of the valid-node MAE over A4 split draws 101-105
                 in the four Bauru quadrants, and their ratio to the 0.132 dB repeat difference;
  c*, d          selected epoch, fraction of runs whose selected epoch is the last one, and the
                 GNN neighbour counts (k_antenna, k_terrain) and layer count actually used;
  e_*, f_*       model MAE over constant-predictor MAE, and how many A4 draws the GNN wins,
                 next to the per-cell counts stored in the A4 aggregate.

Inputs: the run folders under the GPU results directory (`results/gpu/A2c`, `A3`, `A4` here;
the `.npz` prediction files are not distributed, available on request) and `A4/agregado_A4.json`.
Output: `saida_V2.json`, plus a short summary on stdout.
Usage: python verificar_V2.py   Deterministic.
"""
import numpy as np, json, glob, os, re, statistics as st
G='/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/gpu'
OUT="internal/notes"
CH=3  # target channel 3 = RSSI (dB)
def mae(run):
    """Valid-node, sentinel-node and constant-predictor (-110 dBm) MAE of one run, from its predictions."""
    d=glob.glob(f'{run}/predicoes_*.npz')[0]; z=np.load(d)
    e=np.abs(z['pred'][:,CH].astype(np.float64)-z['target'][:,CH].astype(np.float64)); s=z['sentinela']
    c=np.abs(-110.0-z['target'][:,CH].astype(np.float64))
    return dict(val=e[~s].mean(), sen=e[s].mean(), const_val=c[~s].mean(), n_val=int((~s).sum()), n_sen=int(s.sum()))
def sel(run):
    """Selected epoch, number of epochs trained, and the run record of one run."""
    j=json.load(open(glob.glob(f'{run}/run_*.json')[0])); ep=j['epocas']
    return j['selecao']['melhor_epoca'], len(ep), j
R={}
# The agregado_* values are copied from the A4 aggregate for this run, to confirm the recomputation.
m=mae(f'{G}/A4/gnn_v3_a4_bauru_Q1_ss101'); R['checagem_s101']=dict(val=m['val'],sen=m['sen'],agregado_val=2.3909875450833087,agregado_sen=0.12876172723538715)
cells=sorted({re.match(r'gnn_v3_a3_(.*)_s\d+',os.path.basename(p)).group(1) for p in glob.glob(f'{G}/A3/gnn_v3_a3_*_s*') if os.path.isdir(p)})
a={}
for c in cells:
    ds=[]
    for s in range(42,47):
        g=mae(f'{G}/A3/gnn_v3_a3_{c}_s{s}'); l=mae(f'{G}/A3/mlp_v3_a3_{c}_s{s}'); ds.append(g['sen']-l['sen'])
    a[c]=dict(mediana=float(np.median(ds)),deltas=[float(x) for x in ds])
R['a_mediana_delta_sentinela_por_celula']=a; R['a_max_abs_mediana']=max(abs(v['mediana']) for v in a.values())
a2={}
for c in ['bauru_Q1','lins_Q1']:
    r1=mae(f'{G}/A2c/gnn_v3_a2c_{c}_rep1'); r2=mae(f'{G}/A2c/gnn_v3_a2c_{c}_rep2')
    a2[c]=dict(sen=abs(r1['sen']-r2['sen']),val=abs(r1['val']-r2['val']))
R['a_A2c_gnn_delta_abs']=a2
b={}; 
for q in ['Q1','Q2','Q3','Q4']:
    for mod in ['gnn','mlp']:
        v=[mae(f'{G}/A4/{mod}_v3_a4_bauru_{q}_ss{s}')['val'] for s in range(101,106)]
        b[f'{mod}_{q}']=dict(amplitude=max(v)-min(v),dp_pop=float(np.std(v)),dp_amostral=float(np.std(v,ddof=1)),v=v)
R['b']=b
pisoval=max(a2[c]['val'] for c in a2); R['b_piso_val_calc']=pisoval
for mod in ['gnn','mlp']:
    for stt in ['amplitude','dp_pop','dp_amostral']:
        r=[b[f'{mod}_{q}'][stt]/0.132 for q in ['Q1','Q2','Q3','Q4']]; R[f'b_razao_{mod}_{stt}_sobre_0.132']=[min(r),max(r)]
mel={'gnn':[],'mlp':[]}; ult={'gnn':[],'mlp':[]}; nn=set(); kk=set()
runs=[(m_,p) for m_ in ['gnn','mlp'] for p in glob.glob(f'{G}/A3/{m_}_v3_a3_*')+glob.glob(f'{G}/A4/{m_}_v3_a4_*') if os.path.isdir(p)]
for m_,p in runs:
    be,ne,j=sel(p); mel[m_].append(be); ult[m_].append(be==ne)
    if m_=='gnn':
        d=j['diagnostico_treino']; kk.add((d['k_antenna_treino'],d['k_terrain_treino'])); nn.add((j['modelo']['num_layers'],j['config']['k_antenna'],j['config']['k_terrain']))
R['c']={m_:dict(n=len(mel[m_]),mediana=float(np.median(mel[m_])),frac_ultima=float(np.mean(ult[m_])),dist=np.bincount(mel[m_]).tolist()) for m_ in mel}
R['c_A3_A4_separados']={}
for ex in ['A3','A4']:
    for m_ in ['gnn','mlp']:
        v=[];u=[]
        for p in glob.glob(f'{G}/{ex}/{m_}_v3_{ex.lower()}_*'):
            if os.path.isdir(p): be,ne,_=sel(p); v.append(be); u.append(be==ne)
        R['c_A3_A4_separados'][f'{ex}_{m_}']=dict(n=len(v),mediana=float(np.median(v)),frac_ultima=float(np.mean(u)))
R['d']=dict(k_treino=[list(x) for x in kk],layers_k_cfg=[list(x) for x in nn])
e={'gnn':[],'mlp':[]}; f=[]
for q in ['Q1','Q2','Q3','Q4']:
    for s in range(101,106):
        g=mae(f'{G}/A4/gnn_v3_a4_bauru_{q}_ss{s}'); l=mae(f'{G}/A4/mlp_v3_a4_bauru_{q}_ss{s}')
        e['gnn'].append(g['val']/g['const_val']); e['mlp'].append(l['val']/l['const_val']); f.append((q,s,g['val']<l['val']))
for ex,cs in [('A3',cells)]:
    for c in cs:
        for s in range(42,47):
            g=mae(f'{G}/A3/gnn_v3_a3_{c}_s{s}'); l=mae(f'{G}/A3/mlp_v3_a3_{c}_s{s}')
            e['gnn'].append(g['val']/g['const_val']); e['mlp'].append(l['val']/l['const_val'])
for m_ in e: R[f'e_{m_}']=dict(n=len(e[m_]),mediana=float(np.median(e[m_])),min=float(min(e[m_])),max=float(max(e[m_])))
R['f_A4']=dict(n=len(f),gnn_melhor=sum(x[2] for x in f),detalhe=[x for x in f if not x[2]])
ag=json.load(open(f'{G}/A4/agregado_A4.json'))['celulas']
R['f_agregado']={k:v['paridade']['distribuicao_gnn_menos_mlp']['mae_rssi_validos_db']['n_sorteios_gnn_melhor'] for k,v in ag.items()}
json.dump(R,open(f'{OUT}/saida_V2.json','w'),indent=1,default=float)
print(json.dumps({k:v for k,v in R.items() if k not in('b','a_mediana_delta_sentinela_por_celula')},default=float)[:4000])
print({c:round(v['mediana'],4) for c,v in a.items()})
