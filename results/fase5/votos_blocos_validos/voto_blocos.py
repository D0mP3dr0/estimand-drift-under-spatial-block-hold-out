"""Independent recalculation of the number of 10 km blocks with at least one valid node, per cell.

Loads each of the 16 reference-field tensors, rebuilds the block id from the node positions (10 km grid on
equirectangular coordinates) and counts the blocks holding a valid node under two definitions of valid: target
path loss below 299 dB, and RSSI different from -110 dBm. It compares both with the stored counts (`alg`), checks
that the two masks coincide, that the occupied blocks are 132 per cell and equal to `meta_celulas` of
fase5/R5_resumo.json, and that no position or target is NaN. Output: a JSON record in the working directory
(per cell, ranges over the 16 cells and over the Q1 cells). Absolute paths are those of the original run.
Usage: python voto_blocos.py
"""
import json, hashlib, datetime, numpy as np, torch
from pathlib import Path
V=Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
D=Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")
alg={"bauru":[27,15,24,18],"campinas":[24,22,32,24],"lins":[17,22,18,15],"sorocaba":[22,20,24,21]}
R5=json.loads((V/"fase5/R5_resumo.json").read_text())["meta_celulas"]
def blocos(pos):
    lon=pos[:,0].astype(np.float64); lat=pos[:,1].astype(np.float64)
    y=(lat-lat.min())*111000.0; x=(lon-lon.min())*111000.0*np.cos(np.radians(lat))
    gx=(x/1000/10).astype(int); gy=(y/1000/10).astype(int)
    return gx*(gy.max()+1)+gy
out={}; mism=[]
for c,v in alg.items():
  for i,q in enumerate(["Q1","Q2","Q3","Q4"]):
    p=D/f"transfer_dataset_{c}_v19_{q}_enriched_cftudo.pt"
    rf=torch.load(p,map_location="cpu",weights_only=False)
    t=rf["terrain"]; y=torch.as_tensor(t.y).float().numpy(); pos=torch.as_tensor(t.pos).float().numpy()
    g=blocos(pos); occ=np.unique(g)
    sent=y[:,0].astype(np.float64)>=299.0
    nv_a=int(np.unique(g[~sent]).size)
    nv_b=int(np.unique(g[y[:,3].astype(np.float64)!=-110.0]).size)
    nv_c=int(np.unique(g[y[:,3]!=np.float32(-110.0)]).size)
    sentinela_eq=bool(np.array_equal(sent,y[:,3].astype(np.float64)==-110.0))
    k=f"{c}_{q}"
    out[k]=dict(alegado=v[i],recalc_sentinela=nv_a,recalc_rssi_ne_m110=nv_b,
      mascaras_identicas=sentinela_eq,n_nos=int(len(y)),n_sent=int(sent.sum()),
      n_rssi_m110=int((y[:,3]==-110.0).sum()),ocupados=int(occ.size),
      ocupados_R5=R5[k]["N_blocos_ocupados"],nan_pos=int(np.isnan(pos).sum()),nan_y=int(np.isnan(y[:,[0,3]]).sum()),
      confere=(nv_a==v[i]))
    print(k,out[k],flush=True)
a=[o["recalc_sentinela"] for o in out.values()]
q1=[out[f"{c}_Q1"]["recalc_sentinela"] for c in alg]
st=lambda l:dict(min=min(l),mediana=float(np.median(l)),max=max(l),ordenado=sorted(l))
res=dict(data=str(datetime.date.today()),script=str(Path(__file__)),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 celulas=out,todas16=st(a),q1=st(q1),todas_confere=all(o["confere"] for o in out.values()),
 ocupados_132_todas=all(o["ocupados"]==132 for o in out.values()),
 ocupados_bate_R5=all(o["ocupados"]==o["ocupados_R5"] for o in out.values()),
 definicoes_iguais=all(o["recalc_sentinela"]==o["recalc_rssi_ne_m110"] for o in out.values()))
Path("veredito.json").write_text(json.dumps(res,indent=1))
print({k:v for k,v in res.items() if k!="celulas"})
