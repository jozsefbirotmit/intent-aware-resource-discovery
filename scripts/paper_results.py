"""Recompute paper-v8 headline numbers and plotting data from frozen evaluations.
This command is result regeneration, not a fresh solver run. See reproduce.py.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=True)
S=ROOT/'studies';B=S/'final_benchmark';T=S/'final_psa1s_replay/results';M=S/'mechanisms/results';A=S/'alibaba2018_legacy/results/alibaba2018'
def load(p):return [json.loads(l) for l in p.read_text().splitlines()]
def check(v,e,d=2):np.testing.assert_allclose(v,e,rtol=0,atol=.5*10**(-d)+1e-10)
r=load(B/'holdout.jsonl');o=load(B/'online.jsonl');sc=load(B/'scaling.jsonl');assert (len(r),len(o),len(sc))==(90,720,30)
for x in r:
 for c in ('psa','tail8','selective8','multi8'):assert x['opt']<=x[c]['cost']+1e-7 and x[c]['cost']<=x[c]['bound']+1e-7
z=[x for x in r if x['n']>=10];q=[];C=('psa','tail8','selective8','multi8');L=('PSA1','PSA1-E','PSA1-S','PSA1-A')
for c,l in zip(C,L):
 gaps=np.array([100*(x[c]['cost']/x['opt']-1) if x['opt'] else 0 for x in z]);v=[x for x in o if x['algorithm']==c and x['n']>=10];active=[a for x in v for a in (x['times_ms'] if x['found'] else x['times_ms'][:-1])]
 q.append(dict(method=l,mean_gap=gaps.mean(),max_gap=gaps.max(),first_median=np.median([x['times_ms'][0] for x in v]),median=np.median(active),p95=np.quantile(active,.95),mean_saving=np.mean([100*(1-x[c]['cost']/x['no_refresh']) if x['no_refresh'] else 0 for x in z])))
q=pd.DataFrame(q);q.to_csv(out/'table_quality.csv',index=False)
check(q[['mean_gap','max_gap','first_median','median','p95','mean_saving']],[[5.64,25.01,1.16,.68,3.30,16.18],[4.41,23.57,1.25,.66,18.35,17.06],[3.17,17.39,17.13,2.35,41.62,18.02],[2.92,17.39,84.16,9.64,140.09,18.20]])
check([np.mean([x['selective8']['gap'] for x in z if x['family']==f]) for f in ('iid','boundary','probe')],[.14,6.56,2.80])
rng=np.random.default_rng(20261117);ci=[]
for aa,bb in [('psa','tail8'),('tail8','selective8'),('tail8','multi8'),('selective8','multi8')]:
 strata=[np.array([x[aa]['gap']-x[bb]['gap'] for x in z if x['n']==n and x['family']==f]) for n in (10,12) for f in ('iid','boundary','probe')];v=np.concatenate(strata);boot=np.mean([rng.choice(t,(10000,len(t)),replace=True).mean(axis=1) for t in strata],axis=0);ci.append(dict(a=aa,b=bb,mean=v.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),better=int(sum(v>1e-7)),worse=int(sum(v<-1e-7)),ties=int(sum(abs(v)<=1e-7))))
ci=pd.DataFrame(ci);ci.to_csv(out/'paired_ci.csv',index=False);check(ci.loc[[0,1,3],['mean','lo','hi']],[[1.23,.77,1.75],[1.24,.54,2.03],[.24,-.17,.66]]);assert ci.loc[1,['better','worse','ties']].tolist()==[40,7,13]
p=pd.read_csv(T/'philly/policy_summary.csv');a=pd.read_csv(T/'alibaba2018/policy_summary.csv');a=a[a.scope=='per_lag_known'];ps=p[p.method=='psa1_s'];als=a[a.method=='psa1_s'];p0=ps.query("epsilon==0 and lag_min==1 and calibration=='rolling14d' and costs=='heterogeneous' and refresh_ratio==0.05").sort_values(['gpus_requested','replicate']);a0=als.query("protocol=='zero_risk_control' and lag_s==60 and costs=='heterogeneous' and intent==2").sort_values('layout');
check(p0.saving_vs_no_refresh_percent,[14.38,17.27,21.15,6.85,13.89,18.73]);check(a0.saving_vs_no_refresh_percent,[2.54,3.52,3.04]);assert (p0.miss_n==0).all() and (a0.miss_n==0).all();assert a0.n.tolist()==[24010,24059,23286];assert p0.n.tolist()==[61185,62788,64129]*2
p0.assign(dataset='Philly').to_csv(out/'table_tracezero_philly.csv',index=False);a0.assign(dataset='Alibaba').to_csv(out/'table_tracezero_alibaba.csv',index=False)
check(ps.query('epsilon==0').saving_vs_no_refresh_percent.min(),-8.15);check(als.query("protocol=='zero_risk_control'").saving_vs_no_refresh_percent.min(),-3.51)
age=ps.query("replicate==0 and gpus_requested==2 and calibration=='rolling14d' and costs=='heterogeneous' and refresh_ratio==0.05 and epsilon==0").sort_values('lag_min');check(age.saving_vs_no_refresh_percent,[14.38,12.46,11.73,8.76]);age.to_csv(out/'philly_age.csv',index=False)
p1=ps.query('epsilon==0.1');a1=als.query("protocol=='nominal_10pct'");risk=[]
for name,f in [('Philly',p1),('Alibaba',a1)]:
 w=f.loc[f.conditional_miss.idxmax()];risk.append(dict(dataset=name,profiles=len(f),over_10pct=int((f.conditional_miss>.1).sum()),miss=int(w.miss_n),empty=int(w.empty_stop_n),risk=100*w.conditional_miss,lo=100*w.risk_block95_low,hi=100*w.risk_block95_high))
assert [(x['profiles'],x['over_10pct'],x['miss'],x['empty']) for x in risk]==[(112,34,1996,10692),(54,1,2412,23425)];check([[x['risk'],x['lo'],x['hi']] for x in risk],[[18.67,13.41,25.06],[10.30,4.26,16.75]])
case=p.query("replicate==0 and gpus_requested==2 and calibration=='rolling14d' and costs=='heterogeneous' and refresh_ratio==0.05 and epsilon==0.1 and lag_min==1").set_index('method');check(case.loc[['query_only','no_refresh_opt','psa1_s'],'mean_cost'],[3.8356,3.7063,2.1151],4);assert case.loc['psa1_s','miss_n']==630 and case.loc['psa1_s','empty_stop_n']==10308 and case.loc['no_refresh_opt','empty_stop_n']==9678;check(case.loc['psa1_s','saving_vs_no_refresh_percent'],42.93);case.to_csv(out/'philly_positive_budget_case.csv')
assert a1.saving_vs_no_refresh_percent.isna().sum()==18
pd.DataFrame(risk).to_csv(out/'risk_extremes.csv',index=False)
# Independent aggregation of earlier mechanism experiments.
m=pd.read_csv(M/'cases.csv');assert len(m)==3600
for f in ('p3_screen','p4_probe','deferred'):
 x=m[m.family==f];assert len(x)==600 and (x.gap_psa_pct<1e-7).all()
f=pd.read_csv(M/'sensitivity_cases.csv').query('eps==0.10 and r_over_q==0.20');val=[]
for acc,g in f.groupby('accuracy'):assert len(g)==180;val.append(100*(1-g.psa1.mean()/g.no_refresh_opt.mean()))
check(val,[4.0,10.7,32.2],1)
d=pd.read_csv(M/'calibration_drift_cases.csv');dr=[100*g.prob_false_miss.sum()/g.prob_failure.sum() for _,g in d.groupby('delta_logodds')];check(dr,[9.3,14.5,22.0,32.1],1);check(100*d[d.delta_logodds==0].eps.mean(),12.15)
bi=pd.read_csv(A/'brier_scores.csv').query('intent==2 and lag_s==60 and split=="test"');check([bi.prior.mean(),bi.observed_prevalence.mean(),bi.prior_brier.mean(),bi.posterior_brier.mean()],[.756,.497,.327,.100],3)
# Figure data; plots reproduce the numerical content, not the original PDF bytes.
sc=pd.DataFrame(sc);s=sc.groupby(['n','family']).root_ms.agg(['median','min','max']).reset_index();s.to_csv(out/'figure_scaling_data.csv',index=False);check(s.query('n==100').sort_values('family')['median']/1000,[1.19,2.64])
fig,ax=plt.subplots(figsize=(3.4,2.4));
for f,g in s.groupby('family'):
 ax.errorbar(g.n,g['median'],yerr=[g['median']-g['min'],g['max']-g['median']],marker='o',capsize=3,label=f)
ax.set(yscale='log',xlabel='Number of domains',ylabel='First decision time (ms)',xticks=[20,40,60,80,100]);ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(out/'synthetic_scaling.pdf');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(7.0,2.7),layout='constrained');
for ax,(name,f) in zip(axs,[('Philly',p1),('Alibaba',a1)]):
 good=f.saving_vs_no_refresh_percent.notna();f=f.loc[good];f.to_csv(out/f'figure_risk_{name.lower()}_data.csv',index=False);ax.scatter(f.saving_vs_no_refresh_percent,100*f.conditional_miss,s=14,facecolors='none',edgecolors='tab:blue');ax.axhline(10,ls='--',color='tab:red');ax.set(title=name,xlabel='Saving vs. no refresh (%)',ylabel='Conditional misses (%)');ax.grid(alpha=.25)
fig.savefig(out/'telemetry_cost_risk.pdf');plt.close(fig)
print('PASS: frozen v8 numerical claims (quality, paired CIs, savings, telemetry risk, earlier mechanisms, Brier scores), tables and figure data.')
