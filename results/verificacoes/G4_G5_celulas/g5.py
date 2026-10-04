"""Correlation of the per-seed test retentions across cells (block lattice is shared by all cells).

For each nodal-reference record (the 200-seed reference at the main block size and its g = 5 km
variant), the script stacks the test retention series of the 16 cells (one value per split
seed, field `retencoes_te`) and computes the pairwise Pearson correlation matrix. It reports the
minimum, median and maximum correlation, the minimum over pairs of cells from different cities,
the fraction of pairs with correlation of at least 0.995, and the number of blocks per cell.
Near-unit correlations indicate that every cell sees the same permutation of the block lattice.

Inputs: `fase1/1.8_referencia_nodal_200seeds.json` and `fase1/1.8b_referencia_nodal_200seeds_g5.json`
(in this repository under results/fase1/), read relative to the working directory.
Output: `g5_saida.json`, written to the relative path given in the last line.
Usage: python g5.py   (run from the folder that contains fase1/). Deterministic.
"""
import json,numpy as np
o={}
for f in ['1.8_referencia_nodal_200seeds.json','1.8b_referencia_nodal_200seeds_g5.json']:
    j=json.load(open('fase1/'+f));pc=j['por_celula'];cs=sorted(pc)
    M=np.array([pc[c]['retencoes_te'] for c in cs]);C=np.corrcoef(M)
    iu=np.triu_indices(len(cs),1);v=C[iu]
    ext=[(cs[i],cs[k]) for i,k in zip(*iu)]
    inter=[C[i,k] for i,k in zip(*iu) if cs[i].split('_')[0]!=cs[k].split('_')[0]]
    o[f]={'n_cel':len(cs),'blocos':{c:pc[c]['n_blocos'] for c in cs},'corr_min':float(v.min()),'corr_mediana':float(np.median(v)),'corr_max':float(v.max()),'inter_cidades_min':float(min(inter)),'frac_pares_ge_0.995':float((v>=0.995).mean())}
    print(f,o[f])
json.dump(o,open("internal/g5_saida.json",'w'),indent=1)
