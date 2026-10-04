"""Checks of an error-mixture inequality and of the geodesic size of a nominal 10 km block.

Item (a): the inequality (1-w)*a < w*D + (1-w)*e compares two weighted mean absolute errors
(`MAE_fc` and `MAE_f`). It can be rewritten as a threshold on w with thr = (a-e)/(a-e+D), using
the identity (1-w)(a-e) - w*D = a - e - w(a-e+D), which sympy confirms (`a_sympy_identidade_zero`
must be "0"). Two numerical examples are stored. In the first, a-e+D < 0, so the direction of the
threshold rule flips: compare `fc_menor` with `regra_diz_fc_menor`. The second has a-e+D > 0.
Item (d): for latitudes between -20.68 and -24.5 degrees (the span of the study cells) a block
of 10 km x 10 km is laid out in degrees with the spherical approximation of 111 km per degree.
Its east-west and north-south geodesic lengths and its area on the WGS84 ellipsoid are then
measured with pyproj. The area is reported as a percent deviation from 100 km^2.

Inputs: none. Output: `saida_abd.json` in the working directory, also printed to stdout.
Usage: python verif_abd.py   Deterministic.
"""
import json,sympy as sp,numpy as np
from pyproj import Geod
R={}
a,e,D,w=sp.symbols('a e D w',real=True)
lhs=(1-w)*a < w*D+(1-w)*e
R['a_ex']=dict(A=1,E=10,Delta=1,w=0.5,MAE_fc=(1-.5)*1,MAE_f=.5*1+.5*10,
 fc_menor=bool((1-.5)*1<.5*1+.5*10),thr=(1-10)/(1-10+1),regra_diz_fc_menor=bool(.5>(1-10)/(1-10+1)))
R['a_ex2_positivo']=dict(a=20,e=5,Delta=20,thr=15/35)
expr=sp.simplify(((1-w)*(a-e)-w*D)-(a-e-w*(a-e+D)))
R['a_sympy_identidade_zero']=str(expr)
# `lats` (latitude ranges per cell) is kept for reference only; the loop uses the explicit list.
g=Geod(ellps='WGS84'); out=[]
lats={'lins':(-21.68,-20.68),'bauru':(-22.32,-21.32),'campinas':(-22.91,-21.91),'sorocaba':(-23.50,-22.50),'lins_Q3':(-22.68,-21.68),'sorocaba_Q3':(-24.50,-23.50)}
for lat in [-20.68,-21.18,-21.68,-22.18,-22.5,-23.0,-23.5,-24.0,-24.5]:
    lon0=-48.0; dlon=10/(111*np.cos(np.radians(lat))); dlat=10/111
    _,_,ew=g.inv(lon0,lat,lon0+dlon,lat)
    _,_,ns=g.inv(lon0,lat,lon0,lat+dlat)
    # Signed polygon area in m^2 (abs taken below); 100 km^2 is the nominal block area.
    la,_=g.polygon_area_perimeter([lon0,lon0+dlon,lon0+dlon,lon0],[lat,lat,lat+dlat,lat+dlat])
    out.append(dict(lat=lat,ew_km=ew/1e3,ns_km=ns/1e3,area_pct=(abs(la)/1e6/100-1)*100))
R['d']=out
json.dump(R,open('saida_abd.json','w'),indent=1,default=float)
print(json.dumps(R,indent=1,default=float))
