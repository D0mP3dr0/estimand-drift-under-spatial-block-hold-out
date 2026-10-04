"""Exact binomial test for one nodal class of the closed-form inclusion-probability test, per Q1 cell.

Reads the per-class record of the test that compares the closed-form test-inclusion probability
p_{i|te} with its Monte Carlo frequency. The nodal edge class m counts the neighbouring blocks
that hold a node closer than b. For the class "m=2, condition (iii) holds" the script takes the
number of block x draw trials and of trials with at least one retained node. It reports
P(X <= observed) for X ~ Binomial(trials, p0) and Binomial(trials, 2*p0), where p0 is the
closed-form lower bound for m = 2, and a z-score of the mean nodal frequency against p0. It also
reports the fraction of nodes whose p has an exact closed form (versus bounds only) and the node
fractions of the classes m = 0, m = 1 (iii) and m = 3 (iii).

Input: results/fase4/R4_p_por_classe.json (absolute path of the original run).
Output: the JSON path given as first argument (fismat_R4_teste_exato.json in this folder), also printed.
Usage: python fismat_R4_teste_exato.py fismat_R4_teste_exato.json
"""
import json, sys
from scipy import stats
V="/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/fase4/R4_p_por_classe.json"
d=json.load(open(V)); out={}
for cel,c in d["por_celula"].items():
    cl={k["classe"]:k for k in c["classes_nodais"]}
    n=c["fidelidade"]["n_nos"]
    so_cotas=cl["TOTAL|todos os nos"]["leituras"]["L_no_a_no"]["n_nos_so_cotas"]
    sub=cl["m=2|(iii) vale"]; p0=sub["cota_inferior_por_m"]["2"]
    ens=int(sub["ensaios_bloco_sorteio"]); ret=int(sub["retidos_bloco_sorteio_com_algum_no"])
    out[cel]={"frac_exata":1-so_cotas/n,"frac_so_cotas":so_cotas/n,
      "frac_m0":cl["m=0|todas"]["fracao_nos"],"frac_m1_iii":cl["m=1|(iii) vale"]["fracao_nos"],
      "frac_m3_iii":cl["m=3|(iii) vale"]["fracao_nos"],
      "m2_iii":{"n_nos":sub["n_nos"],"ensaios":ens,"eventos":ret,"p0":p0,"esperado":ens*p0,
        "P_X_le_obs_binom_p0":float(stats.binom.cdf(ret,ens,p0)),
        "P_X_le_obs_binom_2p0":float(stats.binom.cdf(ret,ens,2*p0)),
        "z_score_EP_nulo":float((sub["frequencia_media_nodal"]-p0)/((p0*(1-p0)/ens)**0.5))}}
json.dump(out,open(sys.argv[1],"w"),indent=1); print(json.dumps(out,indent=1))
