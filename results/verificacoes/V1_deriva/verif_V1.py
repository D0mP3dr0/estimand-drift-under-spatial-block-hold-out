"""Between-draw versus between-cell spread of the constant-predictor error, and valid-fraction signs.

Part 1 reads the 16 cells x 60 random split draws record of the error-drift test and keeps, for
every draw with status "ok", the constant-predictor MAE on the valid test nodes and the number of
valid test nodes. Draws with no valid test node or a non-finite MAE are skipped. It compares the
mean within-cell standard deviation across draws with the standard deviation of the 16 cell means.
The ratio is reported plain, weighted by the number of valid test nodes, pooled, and corrected for
the sampling noise of the cell means. It also gives an unbalanced one-way ANOVA (MSB, MSW, n0, ICC)
and a 95% bootstrap interval for the ratio: 2000 resamples of the draws within each cell, with
numpy seed 1.
Part 2 (key `A2`) reads, for 16 cells x 20 split seeds, the differences of the valid-node
fraction between validation and training (`delta_val`) and between test and training
(`delta_te`). It counts the cells in which their sign changes across seeds and the per-cell
fractions of seeds with a poorer validation set and a richer test set.

Inputs: `results/fase2/_v3_2.1_3.1_parcial_16x60rnd.json` and
`results/inputs/fracao_valida_val_treino_20_splits.json` (absolute paths of the original run).
Output: `saida_V1.json`, also printed to stdout.
Usage: python verif_V1.py
"""
import json,numpy as np
B='/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/'
O=B+"internal/notes"
out={}
d=json.load(open(B+'fase2/_v3_2.1_3.1_parcial_16x60rnd.json'))
X={};W={}
tot=0;nval=0
for c,v in d['celulas'].items():
    xs=[];ws=[]
    for s in v['por_sorteio']:
        tot+=1
        if s.get('status')!='ok':continue
        n=s['n_pop_teste']['validos'];m=s['mae_constante_teste']['validos']
        if n>0 and m is not None and np.isfinite(m): xs.append(m);ws.append(n)
    X[c]=np.array(xs);W[c]=np.array(ws);nval+=len(xs)
out['sorteios_total']=tot;out['sorteios_com_no_valido']=nval
cs=sorted(X);n=np.array([len(X[c]) for c in cs]);mu=np.array([X[c].mean() for c in cs])
s=np.array([X[c].std(ddof=1) for c in cs])
out['n_por_celula']=n.tolist()
out['dp_intra_media_simples']=s.mean();out['dp_entre_medias']=mu.std(ddof=1)
out['razao']=s.mean()/mu.std(ddof=1)
def wm(x,w):return (w*x).sum()/w.sum()
def wsd(x,w):
    """Weighted standard deviation; denominator sum(w) - sum(w^2)/sum(w) (reliability weights)."""
    xb=wm(x,w);return np.sqrt((w*(x-xb)**2).sum()/(w.sum()-(w**2).sum()/w.sum()))
sw=np.array([wsd(X[c],W[c]) for c in cs]);out['dp_intra_ponderado_media']=sw.mean()
out['razao_ponderada_vs_dp_entre_medias']=sw.mean()/mu.std(ddof=1)
sp=np.sqrt(((n-1)*s**2).sum()/(n-1).sum());out['dp_intra_poolado']=sp
# Between-cell variance corrected for sampling noise: var(cell means) - mean(s^2/n).
vm=(s**2/n).mean();out['var_ruido_medias']=vm
b2=mu.var(ddof=1)-vm;out['dp_entre_corrigido']=np.sqrt(b2) if b2>0 else None
out['razao_corrigida']=s.mean()/np.sqrt(b2)
out['razao_corrigida_poolado']=sp/np.sqrt(b2)
# Unbalanced one-way ANOVA; n0 is the effective group size for unequal draws per cell.
N=n.sum();k=len(n);allx=np.concatenate([X[c] for c in cs]);gm=allx.mean()
ssb=(n*(mu-gm)**2).sum();ssw=((n-1)*s**2).sum()
msb=ssb/(k-1);msw=ssw/(N-k);n0=(N-(n**2).sum()/N)/(k-1)
vb=(msb-msw)/n0;out['anova']={'MSB':msb,'MSW':msw,'n0':n0,'dp_celula':np.sqrt(vb),'dp_sorteio_residual':np.sqrt(msw),'ICC':vb/(vb+msw)}
# Bootstrap: resample draws with replacement within each cell, 2000 times, seed 1.
rng=np.random.default_rng(1);R=[];I=[]
for _ in range(2000):
    xs=[rng.choice(X[c],len(X[c])) for c in cs]
    m=np.array([x.mean() for x in xs]);ss=np.array([x.std(ddof=1) for x in xs])
    R.append(ss.mean()/m.std(ddof=1))
out['razao_boot_IC95']=np.percentile(R,[2.5,97.5]).tolist()
# Part 2: sign of the valid-fraction differences across split seeds, per cell.
a=json.load(open("internal/artifact_04.json"))
from collections import defaultdict
dv=defaultdict(list);dt=defaultdict(list)
for r in a['resultados']:
    c=r['cidade']+'_'+r['Q'];dv[c].append(r['delta_val']);dt[c].append(r['delta_te'])
fv={c:float(np.mean(np.array(v)<0)) for c,v in dv.items()}
ft={c:float(np.mean(np.array(v)>0)) for c,v in dt.items()}
out['A2']={'n_celulas':len(dv),'n_seeds_por_celula':sorted({len(v) for v in dv.values()}),
 'sinal_muda_val':sum(1 for v in dv.values() if min(v)<0<max(v)),'sinal_muda_te':sum(1 for v in dt.values() if min(v)<0<max(v)),
 'frac_val_mais_pobre_min_max':[min(fv.values()),max(fv.values())],'frac_te_mais_rico_min_max':[min(ft.values()),max(ft.values())],
 'frac_val_mais_pobre':fv,'frac_te_mais_rico':ft}
json.dump(out,open(O+'saida_V1.json','w'),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
