"""Independent recalculation of the continuous-target drift test (G2).

Level 1 (`n1`): recomputes CV_d, deff and n_ef of the primary and negative-control variants from the per-draw
values stored in results/fase5/_parcial_G2/<cell>.json (formulas as in the criterion: V_SRS = (1 - n/N_U) S2_U / n,
variances with ddof = 1) -> nivel1.json.
Level 2 (`n2 <cell>`): for one cell, reloads the elevation from the reference-field tensor, reimplements the block
assignment (10 km grid), the 0.70/0.15/0.15 partition and the 2 km buffer trimming with a k-d tree, and recomputes
c_ref, S2_U, the 60 per-draw n and Err, and the statistics -> nivel2_<cell>.json. The permuted control uses
np.random.default_rng(20261004 + cell index). Seeds are read from fase2/2.1_deriva_erro_baselines_16x60rnd.json.
Absolute paths are those of the original run. Usage: python voto_G2_recalculo.py n1 | n2 <cell>
"""
import json, sys, numpy as np
M='/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics'; V=M+'/_v3_2026-09-25'; D=V+'/fase5/votos_G2'
CEL=[f"{c}_Q{q}" for c in ['bauru','campinas','lins','sorocaba'] for q in (1,2,3,4)]
def seeds():
    a=json.load(open(V+'/fase2/2.1_deriva_erro_baselines_16x60rnd.json'))
    return [int(s) for s in a['nota_divergencia_seeds']['seeds_usados_nesta_rodada']]
def stats(err,n,s2,NU):
    err=np.array(err,float); n=np.array(n,float)
    cv=err.std(ddof=1)/err.mean(); var=err.var(ddof=1)
    vaas=((1-n/NU)*s2/n).mean(); deff=var/vaas
    return dict(CV_d=cv,deff=deff,n_ef=n.mean()/deff,n_medio=n.mean(),media_Err=err.mean(),dp=err.std(ddof=1),V_AAS_medio=vaas)
def nivel1():
    out={}
    for k,v in (('prim','prim'),('perm','perm')):
        pass
    out={'prim':{},'perm':{}}
    for c in CEL:
        d=json.load(open(f'{V}/fase5/_parcial_G2/{c}.json'))
        for k in ('prim','perm'):
            ps=[s for s in d['por_sorteio'] if s[k]['n']>0]
            out[k][c]=stats([s[k]['Err'] for s in ps],[s[k]['n'] for s in ps],d['populacao'][k]['S2_U'],d['populacao'][k]['N_U'])
    json.dump(out,open(D+'/nivel1.json','w'),indent=1)
def nivel2(c,idx):
    import torch
    from scipy.spatial import cKDTree
    rf=torch.load(f'/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3/transfer_dataset_{c.split("_")[0]}_v19_{c.split("_")[1]}_enriched_cftudo.pt',map_location='cpu',weights_only=False,mmap=True)
    t=rf['terrain']
    y=np.asarray(t.features_raw[:,0]).astype(np.float64)
    pos=t.pos.float().clone()
    lon=pos[:,0].double().numpy(); lat=pos[:,1].double().numpy()
    x=(lon-lon.min())*111000.0*np.cos(np.radians(lat)); yy=(lat-lat.min())*111000.0
    pk=np.stack([x,yy],1)/1000.0
    gx=(pk[:,0]/10.0).astype(int); gy=(pk[:,1]/10.0).astype(int)
    gid=gx*(gy.max()+1)+gy
    grupos=np.unique(gid)
    def part(seed):
        rng=np.random.RandomState(seed); g=grupos.copy(); rng.shuffle(g)
        n=len(g); ntr=max(1,int(round(.70*n))); nva=max(1,int(round(.15*n)))
        if ntr+nva>=n: ntr=max(1,n-2); nva=1
        mtr=np.isin(gid,g[:ntr]); mva=np.isin(gid,g[ntr:ntr+nva]); mte=np.isin(gid,g[ntr+nva:])
        return mtr,mva,mte
    def te_ret(seed,buf=2.0):
        mtr,mva,mte=part(seed)
        d,_=cKDTree(pk[mtr]).query(pk[mva],k=1); iv=np.where(mva)[0]; mva[iv[d<buf]]=False
        m=mtr|mva
        d,_=cKDTree(pk[m]).query(pk[mte],k=1); it=np.where(mte)[0]; mte[it[d<buf]]=False
        return mte
    res={'celula':c,'idx':idx}
    yp=np.random.default_rng(20261004+idx).permutation(y)
    mtr42,_,_=part(42)
    for k,tg in (('prim',y),('perm',yp)):
        cref=float(np.median(tg[mtr42])); res[k]={'c_ref':cref}
        L=np.abs(tg-cref); res[k]['S2_U']=float(L.var(ddof=1)); res[k]['N_U']=int(L.size); res[k]['L']=L
    E={'prim':[],'perm':[]};N=[]
    for s in seeds():
        m=te_ret(s); N.append(int(m.sum()))
        for k in E: E[k].append(float(res[k]['L'][m].mean()))
    for k in ('prim','perm'):
        res[k].pop('L'); res[k]['Err']=E[k]; res[k]['n']=N
        res[k]['stats']=stats(E[k],N,res[k]['S2_U'],res[k]['N_U'])
    json.dump(res,open(f'{D}/nivel2_{c}.json','w'),indent=1)
if __name__=='__main__':
    if sys.argv[1]=='n1': nivel1()
    else: nivel2(sys.argv[2],CEL.index(sys.argv[2]))
