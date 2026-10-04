"""City-level overlap check that complements g4.py (coincident nodes between neighbouring cities).

For the two neighbouring city pairs (Bauru-Lins and Campinas-Sorocaba) the script merges the
per-cell bounding boxes recorded by g4.py into one box per city and reports the intersection
area and its fraction of the first city's box. It also reports, for each quadrant of one city,
the fraction of its nodes whose coordinates (rounded to 1e-6) occur in any quadrant of the other
city, in both directions.

Inputs: `manifest_mathematics_v4.jsonl` in the working directory, the `*_cftudo.pt` reference
fields it lists (not distributed; available on request), and `g4_saida.json` from g4.py.
Output: `g4b_saida.json`, also printed to stdout.
Usage: python g4b.py   (run after g4.py, from the project root). Deterministic.
"""
import torch,json,numpy as np
ms=[json.loads(l) for l in open('manifest_mathematics_v4.jsonl')]
t={m['celula']:m['caminho'] for m in ms if m['grupo']=='tensores_cftudo'}
def K(c):
    """Integer key per node of cell c (coordinates rounded to 1e-6, same encoding as g4.py)."""
    p=torch.load(t[c],mmap=True,weights_only=False)['terrain']['pos'].numpy().astype(np.float64)
    k=np.round(p/1e-6).astype(np.int64);return k[:,0]*10**7+k[:,1]
o={}
B=json.load(open('/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/'+"internal/g4_saida.json"))['bbox']
def ub(cid):
    """Union bounding box [xmin, xmax, ymin, ymax] of all cells of one city."""
    bs=[B[c] for c in B if c.startswith(cid)]
    return [min(b[0] for b in bs),max(b[1] for b in bs),min(b[2] for b in bs),max(b[3] for b in bs)]
def ov(a,b):
    """Intersection area of two boxes and that area as a fraction of the area of box a."""
    w=max(0,min(a[1],b[1])-max(a[0],b[0]));h=max(0,min(a[3],b[3])-max(a[2],b[2]));return w*h,w*h/((a[1]-a[0])*(a[3]-a[2]))
for A,Bc in [('bauru','lins'),('campinas','sorocaba')]:
    o['bbox_cidade_'+A+'_'+Bc]=ov(ub(A),ub(Bc)); o['bbox_cidade_'+A]=ub(A);o['bbox_cidade_'+Bc]=ub(Bc)
    other=np.unique(np.concatenate([K(f'{Bc}_Q{i}') for i in range(1,5)]))
    for i in range(1,5):
        k=K(f'{A}_Q{i}');o[f'{A}_Q{i}_em_qualquer_{Bc}']=float(np.isin(k,other).mean())
    # Reverse direction: quadrants of the second city against all nodes of the first.
    allA=np.unique(np.concatenate([K(f'{A}_Q{i}') for i in range(1,5)]))
    for i in range(1,5):
        k=K(f'{Bc}_Q{i}');o[f'{Bc}_Q{i}_em_qualquer_{A}']=float(np.isin(k,allA).mean())
print(json.dumps(o,indent=1));json.dump(o,open("internal/g4b_saida.json",'w'),indent=1)
