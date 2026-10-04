"""Ratio (Hartley-Ross) term of the per-draw test error in the design-reference test, four Q1 cells.

The per-draw test error is a ratio estimator: a sum over the test nodes divided by the number of
test nodes M. Its bias against the population mean mu_U therefore splits into a ratio term,
-cov(err, M) / mean(M), and an empirical design term, sum(err*M)/sum(M) - mu_U. `fecha` is the
closure residual and must be close to 0. The script reports both terms, a bootstrap standard
error and z-score of the ratio term (4000 resamples of draws, numpy seed 20261001), the
Hartley-Ross bound sd(err)*sd(M)/mean(M), and corr(err, M).
`preparar_celula` is imported unchanged from analysis/v3_12_R2_referencia_desenho.py. The script
redoes the b = 0 partition (H0: the test set is the union of the test blocks, no buffer) for the
200 split seeds of the nodal reference, to recover the per-draw M of valid nodes, which
R2_por_sorteio.json does not store. The recomputed b = 0 errors and test sizes are compared
with that record (`conferencia_*` keys). For b = 2 km (H) the stored errors and test sizes are used.

Inputs: analysis/v3_12_R2_referencia_desenho.py, results/fase1/1.8_referencia_nodal_200seeds.json,
results/fase4/R2_por_sorteio.json (absolute paths of the original run).
Output: the JSON path given as first argument (fismat_R2_termo_razao.json in this folder).
Usage: python fismat_R2_termo_razao.py fismat_R2_termo_razao.json
"""
import importlib.util, json, sys, numpy as np
S="/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts/v3_12_R2_referencia_desenho.py"
spec=importlib.util.spec_from_file_location("r2",S); R2=importlib.util.module_from_spec(spec); spec.loader.exec_module(R2)
O=R2.O
V3="/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/"
seeds=json.load(open(V3+"fase1/1.8_referencia_nodal_200seeds.json"))["seeds"]
PS=json.load(open(V3+"fase4/R2_por_sorteio.json"))["celulas"]
rng=np.random.default_rng(20261001)
def decomp(err,M,mu,nboot=4000):
    """Split bias = mean(err) - mu into the ratio term and the empirical design term, plus bootstrap SE.

    err: per-draw test error; M: per-draw number of test nodes; draws with M = 0 or NaN error are dropped.
    """
    ok=(M>0)&~np.isnan(err); err=err[ok]; M=M[ok].astype(float)
    bias=err.mean()-mu
    ratio=-np.mean((err-err.mean())*(M-M.mean()))/M.mean()
    theta_emp=(err*M).sum()/M.sum()
    design_emp=theta_emp-mu
    hr=err.std()*M.std()/M.mean()
    bs=[]
    n=err.size
    for _ in range(nboot):
        i=rng.integers(0,n,n); e_,m_=err[i],M[i]
        bs.append(-np.mean((e_-e_.mean())*(m_-m_.mean()))/m_.mean())
    bs=np.array(bs)
    return dict(n=int(n),vies=float(bias),termo_razao=float(ratio),termo_desenho_empirico_MC=float(design_emp),
                fecha=float(bias-(ratio+design_emp)),ep_boot_razao=float(bs.std(ddof=1)),
                z_razao=float(ratio/bs.std(ddof=1)),cota_HR=float(hr),corr_err_M=float(np.corrcoef(err,M)[0,1]))
out={}
for cid in ["bauru","campinas","lins","sorocaba"]:
    C=R2.preparar_celula(cid); gid,grupos,n_g,sent,e=C["gid"],C["grupos"],C["n_g"],C["sentinela"],C["e"]
    # Per-block node counts and error sums, so each draw only sums over its test blocks.
    ug,inv=np.unique(gid,return_inverse=True)
    assert np.array_equal(ug,np.sort(grupos))
    nb_all=np.bincount(inv); nb_val=np.bincount(inv,weights=(~sent).astype(float))
    sb={p:{"validos":np.bincount(inv,weights=np.where(~sent,e[p],0.0)),"todos":np.bincount(inv,weights=e[p])} for p in e}
    M0v=[];M0t=[];E0={p:{"validos":[],"todos":[]} for p in e}
    for s in seeds:
        # Same block shuffle and training/validation/test counts as the partition code (FRACS from the imported module).
        emb=grupos.copy(); np.random.RandomState(s).shuffle(emb)
        n_tr=max(1,int(round(O.FRACS[0]*n_g))); n_va=max(1,int(round(O.FRACS[1]*n_g)))
        if n_tr+n_va>=n_g: n_tr=max(1,n_g-2); n_va=1
        te=np.isin(ug,emb[n_tr+n_va:])
        mv=nb_val[te].sum(); mt=nb_all[te].sum(); M0v.append(mv); M0t.append(mt)
        for p in e:
            E0[p]["validos"].append(sb[p]["validos"][te].sum()/mv if mv>0 else np.nan)
            E0[p]["todos"].append(sb[p]["todos"][te].sum()/mt)
    M0v=np.array(M0v); M0t=np.array(M0t)
    key=f"{cid}_Q1"; ps=PS[key]; info=ps["info_por_sorteio"]
    res={"conferencia_n_te_b0_igual":bool(np.array_equal(M0t,np.array(info["n_te_b0"])))}
    for p in e:
        for pop in ("validos","todos"):
            Ej=np.array(ps["por_sorteio"][p][pop]["H0"],float); Em=np.array(E0[p][pop],float)
            m=~np.isnan(Ej)&~np.isnan(Em)
            res[f"conferencia_H0_{p}_{pop}_maxdif"]=float(np.max(np.abs(Ej[m]-Em[m])))
            mu=ps["mu_U"][p][pop]
            res[f"H0_{p}_{pop}"]=decomp(Em,M0v if pop=="validos" else M0t,mu)
            # b = 2 km (H): stored per-draw errors; M is the stored count of valid (or all) test nodes.
            EH=np.array(ps["por_sorteio"][p][pop]["H"],float)
            MH=np.array(info["n_te_b2_validos"] if pop=="validos" else info["n_te_b2"],float)
            res[f"H_{p}_{pop}"]=decomp(EH,MH,mu)
    out[key]=res; print(key,json.dumps({k:v for k,v in res.items() if k.startswith("conf")}),flush=True)
json.dump(out,open(sys.argv[1],"w"),indent=1)
for k,v in out.items():
    for kk in ("H0_fspl_calibrado_b_validos","H_fspl_calibrado_b_validos","H0_constante_validos","H_constante_validos"):
        r=v[kk]; print(k,kk,"vies=%.3f razao=%.3f desenhoMC=%.3f ep_razao=%.3f z=%.2f HR=%.3f corr=%.3f"%(r["vies"],r["termo_razao"],r["termo_desenho_empirico_MC"],r["ep_boot_razao"],r["z_razao"],r["cota_HR"],r["corr_err_M"]))
