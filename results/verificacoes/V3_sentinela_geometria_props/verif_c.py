"""Sentinel nodes versus distance to the nearest antenna, and the RSSI range of valid nodes.

For each of the 16 reference fields (group `tensores_cftudo` of the v4 artifact manifest) the
script streams the terrain nodes in chunks of 2,000,000 and cross-tabulates two conditions:
sentinel target (target channel 0 >= 299) and distance to the nearest antenna above 30 km
(`dist_nearest_m` > 30000). It also records, over the valid (non-sentinel) nodes, the minimum
RSSI (target channel 3, dBm) and how many fall below -110 dBm, as well as the smallest distance
of a sentinel node, the largest distance of a valid node, and the coordinate range of a
1-in-1000 node subsample. The output feeds the statements on the 30 km sentinel rule, the
minimum RSSI of valid nodes and the share of valid nodes below -110 dBm.

Inputs: `manifest_mathematics_v4.jsonl` and the `*_cftudo.pt` reference fields it lists (not
distributed; available on request, digests in the v5 manifest).
Output: `saida_c.json` (one record per cell), written to the path in the last line.
Usage: python verif_c.py   Deterministic.
"""
import json,torch,numpy as np
M='/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/manifest_mathematics_v4.jsonl'
out=[]
for l in open(M):
    d=json.loads(l)
    if d.get('grupo')!='tensores_cftudo': continue
    # Both branches of the conditional load the same terrain record.
    o=torch.load(d['caminho'],mmap=True,weights_only=False)['terrain'] if False else torch.load(d['caminho'],mmap=True,weights_only=False)['terrain']
    y=o.y; dn=o.dist_nearest_m; pos=o.pos
    N=y.shape[0]; C=2_000_000
    s=ns=0; tp=fp=fn=tn=0; vmin=1e9; vb=0; nv=0; dmin_s=1e9; dmax_v=0; nan=0
    for a in range(0,N,C):
        yy=y[a:a+C].numpy(); dd=dn[a:a+C].numpy()
        # Sentinel: target channel 0 at or above 299; far: more than 30 km (distances in metres).
        sent=yy[:,0]>=299; far=dd>30000
        tp+=int((sent&far).sum()); fp+=int((~sent&far).sum()); fn+=int((sent&~far).sum()); tn+=int((~sent&~far).sum())
        v=~sent; r=yy[v,3]; nv+=int(v.sum()); vb+=int((r<-110).sum()); vmin=min(vmin,float(r.min())) if r.size else vmin
        if sent.any(): dmin_s=min(dmin_s,float(dd[sent].min()))
        if v.any(): dmax_v=max(dmax_v,float(dd[v].max()))
    # `lat` is always None (unused); the coordinate range comes from every 1000th node.
    lat=pos[:,1].numpy() if False else None
    p=pos[::1000].numpy()
    out.append(dict(celula=d['celula'],N=N,sent_far=tp,notsent_far=fp,sent_notfar=fn,notsent_notfar=tn,
      frac_sent=(tp+fn)/N,valid=nv,valid_below_110=vb,frac_valid_below110=vb/max(nv,1),min_rssi_valid=vmin,
      min_dist_sent=dmin_s,max_dist_valid=dmax_v,pos0min=p.min(0).tolist(),pos0max=p.max(0).tolist()))
    print(out[-1],flush=True)
json.dump(out,open("internal/saida_c.json",'w'),indent=1)
