#!/usr/bin/env python3
"""Test 1.8b: nodal retention reference over 200 random split seeds in the 16 cells at g = 5 km, b = 2 km.

Same computation as v3_1.8_referencia_nodal_200seeds.py with block side g = 5 km.
For each city x quadrant cell, the node grid is rebuilt from the bounding box of the
g = 10 km run record (used only for the geometry) and the frozen buffered split is
drawn for 200 split seeds (block shuffle with RandomState(seed), 70/15/rest;
validation trimmed against training, test trimmed against training plus retained
validation; exact cKDTree distances). The script reports, per cell and role, the mean
nodal retention (retained / pre-buffer nodes), its standard deviation across draws and
a 95% normal interval of the mean (mean +/- 1.96 SE), and the relative error of the
mean-field retention law, law / mean(nodal) - 1, with q taken from the nominal block
counts (test: k_te / N; validation: 1 - k_tr / N).
Seeds: RandomState(20260926).randint(10**6, size=N_SEEDS), N_SEEDS = 200 unless set
in the environment (the same seeds as test 1.8).
Input: the run records under TREINOS_DIR (varredura_split_geometria.py).
Output: results/fase1/1.8b_referencia_nodal_200seeds_g5.json (OUT). One process per cell (Pool(8)).
Usage: [N_SEEDS=200] python v3_1.8b_referencia_nodal_200seeds_g5.py
"""
import sys, json, time, os
sys.path.insert(0, '/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts')
import numpy as np
from multiprocessing import Pool
from scipy.spatial import cKDTree
from varredura_split_geometria import TREINOS_DIR, build_synthetic_grid, latlon_graus_para_metros, assign_groups
OUT = '/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/fase1/1.8b_referencia_nodal_200seeds_g5.json'
CID = ['bauru','campinas','lins','sorocaba']; QS = ['Q1','Q2','Q3','Q4']
N_SEEDS = int(os.environ.get('N_SEEDS', 200)); SEEDS = np.random.RandomState(20260926).randint(10**6, size=N_SEEDS).tolist()
G, B, FR = 5.0, 2.0, (0.70, 0.15, 0.15)  # block side g (km), buffer b (km), role fractions

def R_lei(g, b, q):
    """Simplified interior-block form without the corner term (multiplied by 0); not used below."""
    return ((g - 2*b*(1-q))**2 + 4*(np.pi*b**2/4)*(1-q)*q**2 * 0
            ) / g**2
def R_eqR(g, b, q):
    """Mean-field retained area fraction of an interior g x g block, by exact enumeration of
    the 2^8 configurations of its 8 neighbours, each trimming independently with probability 1 - q."""
    import itertools
    tot = 0.0
    for L,Rr,Bb,T,c1,c2,c3,c4 in itertools.product([0,1], repeat=8):  # 1 = neighbour trims
        p = 1.0
        for v in (L,Rr,Bb,T,c1,c2,c3,c4): p *= (1-q) if v else q
        w = g - b*(L+Rr); h = g - b*(Bb+T); a = w*h
        # Quarter disc of radius b removed where a diagonal neighbour trims and both adjacent sides do not.
        for diag,(l1,l2) in zip((c1,c2,c3,c4), ((L,Bb),(Rr,Bb),(L,T),(Rr,T))):
            if diag and not l1 and not l2: a -= np.pi*b**2/4
        tot += p*a
    return tot/g**2

def celula(args):
    """Nodal retention over all seeds and mean-field law for one (city, quadrant) cell."""
    cid, q = args; t0 = time.time()
    d = json.load(open(TREINOS_DIR / f"run_c0c1cf_{cid}_s42_{q}_g10b2.json")); geo = d['geometria']
    lon, lat = build_synthetic_grid(geo['lon_min_deg'], geo['lon_max_deg'], geo['lat_min_deg'], geo['lat_max_deg'])
    pos_km = latlon_graus_para_metros(lon, lat)/1000.0; gid = assign_groups(pos_km, G); grupos = np.unique(gid); n_g = len(grupos)
    kw = dict(compact_nodes=False, balanced_tree=False); ret_te, ret_va, kte, ktr = [], [], None, None
    for s in SEEDS:
        rng = np.random.RandomState(s); emb = grupos.copy(); rng.shuffle(emb)
        n_tr = max(1,int(round(FR[0]*n_g))); n_va = max(1,int(round(FR[1]*n_g)))
        if n_tr+n_va >= n_g: n_tr = max(1,n_g-2); n_va = 1
        g_tr, g_va, g_te = emb[:n_tr], emb[n_tr:n_tr+n_va], emb[n_tr+n_va:]; kte, ktr = len(g_te), len(g_tr)
        m_tr, m_va, m_te = np.isin(gid,g_tr), np.isin(gid,g_va), np.isin(gid,g_te)
        nva0, nte0 = int(m_va.sum()), int(m_te.sum())
        dd,_ = cKDTree(pos_km[m_tr], **kw).query(pos_km[m_va], k=1, workers=3); iv = np.where(m_va)[0]; m_va[iv[dd<B]] = False
        dd,_ = cKDTree(pos_km[m_tr|m_va], **kw).query(pos_km[m_te], k=1, workers=3); it = np.where(m_te)[0]; m_te[it[dd<B]] = False
        ret_te.append(m_te.sum()/nte0); ret_va.append(m_va.sum()/nva0)
    te, va = np.array(ret_te), np.array(ret_va)
    q_te, q_va = kte/n_g, 1 - ktr/n_g  # non-trimming block fraction: test blocks for test, non-training for val
    lei_te, lei_va = R_eqR(G,B,q_te), R_eqR(G,B,q_va)
    # 95% normal interval of the mean over seeds; the error interval maps its endpoints.
    def ic(a): m=a.mean(); se=a.std(ddof=1)/np.sqrt(len(a)); return [float(m-1.96*se), float(m+1.96*se)]
    r = dict(celula=f"{cid}_{q}", n_blocos=int(n_g), k_te=int(kte), k_tr=int(ktr), q_te=q_te, q_va=q_va,
             nodal_te_media=float(te.mean()), nodal_te_dp=float(te.std(ddof=1)), nodal_te_ic95=ic(te),
             nodal_va_media=float(va.mean()), nodal_va_dp=float(va.std(ddof=1)), nodal_va_ic95=ic(va),
             lei_te=float(lei_te), lei_va=float(lei_va),
             erro_lei_vs_nodal_te=float(lei_te/te.mean()-1), erro_lei_vs_nodal_te_ic95=[float(lei_te/ic(te)[1]-1), float(lei_te/ic(te)[0]-1)],
             erro_lei_vs_nodal_va=float(lei_va/va.mean()-1), erro_lei_vs_nodal_va_ic95=[float(lei_va/ic(va)[1]-1), float(lei_va/ic(va)[0]-1)],
             retencoes_te=[float(x) for x in te], tempo_s=time.time()-t0)
    print(r['celula'], 'te=%.4f ±%.4f lei=%.4f erro=%.2f%% [%.2f,%.2f] t=%.0fs' % (te.mean(), te.std(ddof=1), lei_te, 100*r['erro_lei_vs_nodal_te'], *[100*x for x in r['erro_lei_vs_nodal_te_ic95']], r['tempo_s']), flush=True)
    return r

if __name__ == '__main__':
    t0 = time.time()
    with Pool(8) as p: res = p.map(celula, [(c,q) for c in CID for q in QS])
    erros = np.array([r['erro_lei_vs_nodal_te'] for r in res])
    out = dict(id='1.8', alegacao='A1', seeds=SEEDS, n_seeds=N_SEEDS, regra='codigo congelado, cKDTree exato; q por blocos nominais',
               por_celula={r['celula']: r for r in res},
               resumo=dict(erro_lei_vs_nodal_te_min=float(erros.min()), erro_lei_vs_nodal_te_max=float(erros.max()), erro_lei_vs_nodal_te_media=float(erros.mean()),
                           nodal_te_media_16=float(np.mean([r['nodal_te_media'] for r in res])), dp_por_sorteio_media=float(np.mean([r['nodal_te_dp'] for r in res]))),
               comando_rodado='N_SEEDS=%d %s' % (N_SEEDS, os.path.abspath(__file__)), tempo_total_s=time.time()-t0)
    json.dump(out, open(OUT,'w'), indent=1); print('gravado', OUT, 'em %.0fs' % (time.time()-t0))
