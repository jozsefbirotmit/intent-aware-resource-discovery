"""Recompute WP4 headline summaries using only published aggregate results.
Usage: python check_telemetry_summaries.py /path/to/checkout-of-v1.0.0 [--replay-root]
With --replay-root, read the equivalent local frozen replay folder instead.
No Alibaba machine-level data are required or redistributed.
"""
import argparse,json,math,hashlib
from pathlib import Path
import pandas as pd
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--replay-root',action='store_true');ap.add_argument('--out',type=Path);a=ap.parse_args()
r=a.root if a.replay_root else a.root/'studies/final_psa1s_replay'
pf=r/'results/philly/policy_summary.csv';af=r/'results/alibaba2018/policy_summary.csv';sf=r/'input/philly/domain_states.csv.gz'
p=pd.read_csv(pf); ab=pd.read_csv(af); p=p[p.method=='psa1_s']; ab=ab[(ab.method=='psa1_s')&(ab.scope=='per_lag_known')];p0=p[p.epsilon==0];a0=ab[ab.protocol=='zero_risk_control'];
def grid(x):
    s=x.saving_vs_no_refresh_percent;assert s.notna().all();return dict(n=len(s),positive=int((s>1e-9).sum()),zero=int((abs(s)<=1e-9).sum()),negative=int((s< -1e-9).sum()),minimum=float(s.min()))
def interval(x):
    assert len(x)==1;y=x.iloc[0];return dict(savings=float(y.saving_vs_no_refresh_percent),low=float(y.saving_no_refresh_block95_low),high=float(y.saving_no_refresh_block95_high),n=int(y.n))
def roundeq(v,e,d=2):assert round(v,d)==e,(v,e)
def counts(x):
    assert len(x)==1;y=x.iloc[0];m,e,n,s=int(y.miss_n),int(y.empty_stop_n),int(y.n),int(y.success_n);assert e+s==n;return dict(n=n,success=s,empty=e,miss=m,conditional=100*m/e,feasible=100*m/(m+s),unconditional=100*m/n)
result={'grid':{'philly':grid(p0),'alibaba':grid(a0)},'bootstrap_examples':{'philly':interval(p0.query("replicate==0 and gpus_requested==2 and calibration=='rolling14d' and costs=='heterogeneous' and refresh_ratio==.05 and lag_min==1")),'alibaba':interval(a0.query("layout==1 and intent==2 and lag_s==60 and costs=='heterogeneous'"))}}
assert [result['grid'][k][m] for k in ('philly','alibaba') for m in ('positive','zero','negative')]==[99,0,13,10,33,11]
for k,e in [('philly',(-8.15,14.38,2.03,25.31)),('alibaba',(-3.51,3.52,-.41,7.87))]:
    roundeq(result['grid'][k]['minimum'],e[0]); z=result['bootstrap_examples'][k];
    for key,v in zip(('savings','low','high'),e[1:]):roundeq(z[key],v)
# Coverage: count all calendar test bins (including missing ones), not only selected rows.
f=pd.read_csv(sf,parse_dates=['available_at_utc']);st=pd.Timestamp('2017-11-01 08:00:00+00:00'); cov=[]
for _,g in f.groupby(['replicate','gpus_requested','mode']):
    denom=int((g.available_at_utc>=st).sum()); known=int(g.test.sum());assert denom==64801;cov.append(100*known/denom)
alibaba_bins=(8*86400-(5*86400+3600))//10
acov=100*a0.n/alibaba_bins
result['coverage']={'philly':[min(cov),max(cov)],'alibaba':[float(acov.min()),float(acov.max())],'philly_bins_per_layout':64801,'alibaba_bins_per_profile':alibaba_bins}
assert [round(x,1) for x in result['coverage']['philly']]==[94.4,99.0];assert [round(x,1) for x in result['coverage']['alibaba']]==[86.3,94.1]
c=ab[ab.protocol=='calibration_selected'];nom=ab[ab.protocol=='nominal_10pct'];w=nom.loc[nom.conditional_miss.idxmax()];wc=c[c.profile==w.profile].iloc[0];result['calibration']={'selected_budget_counts':{str(k):int(v) for k,v in c.epsilon.value_counts().items()},'worst_profile':int(w.profile),'selected_budget_worst':float(wc.epsilon),'nominal_risk':float(w.conditional_miss),'selected_risk':float(wc.conditional_miss)}
assert len(c)==54 and int((c.epsilon==.1).sum())==49 and int((c.epsilon<.1).sum())==5 and wc.epsilon==.1 and wc.conditional_miss==w.conditional_miss
result['risk_examples']={'philly':counts(p[p.scenario==8]),'alibaba':counts(nom[nom.profile==w.profile])}
result['source_sha256']={str(x.relative_to(r)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (pf,af,sf)}
output=json.dumps(result,indent=2)+'\n';print(output,end='');
if a.out:a.out.write_text(output)
