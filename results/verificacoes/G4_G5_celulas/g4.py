"""Overlap check between the cells of different cities (bounding boxes and coincident nodes).

For each of the 16 reference fields (group `tensores_cftudo` of the v4 artifact manifest) the
script loads the terrain node positions and targets, records the bounding box of each cell, and
computes for every pair of cells from different cities the box intersection relative to the
smaller box and to the union. For the two city pairs that lie close to each other (Bauru-Lins and
Campinas-Sorocaba) it also counts nodes whose coordinates coincide within 1e-6 and compares their
targets. The output feeds the statement that the cells are disjoint across cities.

Inputs: `manifest_mathematics_v4.jsonl` in the working directory and the `*_cftudo.pt` reference
fields it points to (not distributed; available on request, digests in the v5 manifest).
Output: `g4_saida.json` (keys `bbox`, `tol`, `pares`, `coinc`), written to the relative path
given in the last line.
Usage: python g4.py   (run from the project root that holds the manifest). Deterministic.
"""
import torch,json,numpy as np,itertools
ms=[json.loads(l) for l in open('manifest_mathematics_v4.jsonl')]
t={m['celula']:m['caminho'] for m in ms if m['grupo']=='tensores_cftudo'}
P={};Y={};B={}
for c,p in sorted(t.items()):
    d=torch.load(p,mmap=True,weights_only=False)['terrain']
    # The conditional always takes the full target array d['y'].
    pos=d['pos'].numpy().astype(np.float64); y=d['y'][:,0].numpy() if False else d['y'].numpy()
    P[c]=pos;Y[c]=y
    B[c]=[float(pos[:,0].min()),float(pos[:,0].max()),float(pos[:,1].min()),float(pos[:,1].max())]
    print(c,B[c],flush=True)
def inter(a,b):
    """Area of the intersection of two boxes given as [xmin, xmax, ymin, ymax]."""
    w=max(0,min(a[1],b[1])-max(a[0],b[0]));h=max(0,min(a[3],b[3])-max(a[2],b[2]))
    return w*h
def area(a):return (a[1]-a[0])*(a[3]-a[2])
out={'bbox':B,'tol':1e-6,'pares':{}}
cs=sorted(P)
def keys(pos,tol):return np.round(pos/tol).astype(np.int64)
for a,b in itertools.combinations(cs,2):
    if a.split('_')[0]==b.split('_')[0]:continue
    i=inter(B[a],B[b])
    # No-op guard: every cross-city pair is recorded, overlapping or not.
    if i==0 and not (a[:3]=='bau' and b[:3]=='lin') : pass
    r={'inter_sobre_menor':i/min(area(B[a]),area(B[b])),'inter_sobre_uniao':i/(area(B[a])+area(B[b])-i)}
    out['pares'][a+'|'+b]=r
tol=1e-6
def coinc(a,b):
    """Nodes of cell a whose rounded coordinates also occur in cell b, and how their targets compare."""
    ka=keys(P[a],tol);kb=keys(P[b],tol)
    # One integer key per node: coordinates of about 50 degrees give 5e7 after rounding to 1e-6,
    # so the combined key stays near 5e14, well inside int64.
    va=ka[:,0]*10**7+ka[:,1]
    vb=kb[:,0]*10**7+kb[:,1]
    ia=np.where(np.isin(va,vb))[0]
    if len(ia)==0:return {'n':0,'frac_a':0.0}
    order=np.argsort(vb);pos_=np.searchsorted(vb[order],va[ia]);jb=order[pos_]
    ya=Y[a][ia];yb=Y[b][jb]
    eq=np.all(ya==yb,axis=1) if ya.ndim>1 else ya==yb
    return {'n':int(len(ia)),'frac_a':len(ia)/len(va),'frac_b':len(ia)/len(vb),'y_igual_frac':float(eq.mean()),'y_absdiff_media':float(np.abs(ya-yb).mean())}
out['coinc']={}
for a in cs:
    for b in cs:
        if a<b and ((a.startswith('bauru') and b.startswith('lins')) or (a.startswith('campinas') and b.startswith('sorocaba'))):
            out['coinc'][a+'|'+b]=coinc(a,b);out['coinc'][b+'|'+a]=coinc(b,a);print(a,b,out['coinc'][a+'|'+b],out['coinc'][b+'|'+a],flush=True)
json.dump(out,open("internal/g4_saida.json",'w'),indent=1)
