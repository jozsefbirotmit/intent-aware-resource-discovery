"""Summarize frozen benchmark. Stratified paired confidence intervals over models."""
import json,csv,numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).parent
C=('psa','tail8','selective8','multi8'); L=('PSA1','PSA1-E','PSA1-S','PSA1-A')
def load(n):return [json.loads(x) for x in (R/n).read_text().splitlines()]
def csvout(n,z):
 with open(R/n,'w') as f:
  w=csv.DictWriter(f,fieldnames=list(z[0]));w.writeheader();w.writerows(z)
def group(z,tag):
 out=[]
 for c in C:
  g=np.array([x[c]['gap'] for x in z]);save=np.array([100*(1-x[c]['cost']/x['no_refresh']) if x['no_refresh']>0 else 0 for x in z]);
  out.append(dict(group=tag,algorithm=c,n=len(z),mean_gap=g.mean(),median_gap=np.median(g),p95_gap=np.quantile(g,.95),max_gap=g.max(),mean_saving_no_refresh=save.mean(),negative_savings=int(sum(save<-1e-7)),median_root_ms=np.median([x[c]['root_ms'] for x in z])))
 return out
r=load('holdout.jsonl');o=load('online.jsonl');s=load('scaling.jsonl');assert len(r)==90 and len(o)==720 and len(s)==30
z=[x for x in r if x['n']>=10]; groups=group(r,'all')+group(z,'N10-12')
for n in (8,10,12):groups+=group([x for x in r if x['n']==n],str(n))
for f in ('iid','boundary','probe'):groups+=group([x for x in z if x['family']==f],f)
csvout('quality_summary.csv',groups)
rng=np.random.default_rng(20261117);ci=[]
for a,b in [('psa','tail8'),('tail8','selective8'),('tail8','multi8'),('selective8','multi8')]:
 ss=[np.array([x[a]['gap']-x[b]['gap'] for x in z if x['n']==n and x['family']==fam]) for n in (10,12) for fam in ('iid','boundary','probe')];v=np.concatenate(ss);boot=np.mean([rng.choice(t,(10000,len(t)),replace=True).mean(axis=1) for t in ss],axis=0)
 ci.append(dict(a=a,b=b,mean_gap_reduction=v.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),better=int(sum(v>1e-7)),worse=int(sum(v<-1e-7)),ties=int(sum(abs(v)<=1e-7))))
csvout('paired_ci.csv',ci)
on=[]
for n in ('all','N10-12',8,10,12):
 for c in C:
  v=[x for x in o if x['algorithm']==c and (n=='all' or (n=='N10-12' and x['n']>=10) or n==x['n'])];active=[a for x in v for a in (x['times_ms'] if x['found'] else x['times_ms'][:-1])];duration=[sum(x['times_ms'] if x['found'] else x['times_ms'][:-1]) for x in v];root=[x['times_ms'][0] for x in v];on.append(dict(group=n,algorithm=c,trajectories=len(v),active_decisions=len(active),root_median_ms=np.median(root),root_p95_ms=np.quantile(root,.95),active_median_ms=np.median(active),active_p95_ms=np.quantile(active,.95),trajectory_median_ms=np.median(duration),trajectory_p95_ms=np.quantile(duration,.95),max_ms=max(active),peak_tail_table_bytes=max(x['peak_table_bytes'] for x in v)))
csvout('online_summary.csv',on)
sc=[]
for n in (20,40,60,80,100):
 for f in ('iid','stress'):
  v=[x['root_ms'] for x in s if x['n']==n and x['family']==f];sc.append(dict(n=n,family=f,reps=len(v),median_ms=np.median(v),min_ms=min(v),max_ms=max(v)))
csvout('scaling_summary.csv',sc)
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.25,'lines.linewidth':1.15,'lines.markersize':5,'savefig.bbox':'tight'})
fig,ax=plt.subplots(figsize=(5.2,3.2))
for c,l,m in zip(C,L,['o','s','^','D']):
 y=[next(x['mean_gap'] for x in groups if x['group']==str(n) and x['algorithm']==c) for n in (8,10,12)];ax.plot([8,10,12],y,marker=m,markerfacecolor='none',label=l)
ax.set(xlabel='Number of domains',ylabel='Mean optimality gap (%)',xticks=[8,10,12]);ax.legend();fig.tight_layout();fig.savefig(R/'fig_gap.png',dpi=220);plt.close(fig)
fig,ax=plt.subplots(figsize=(5.2,3.2))
for c,l,m in zip(C,L,['o','s','^','D']):
 x=next(a['root_median_ms'] for a in on if a['group']=='N10-12' and a['algorithm']==c);y=next(a['mean_gap'] for a in groups if a['group']=='N10-12' and a['algorithm']==c);ax.plot(x,y,marker=m,markerfacecolor='none',linestyle='none',label=l)
ax.set(xscale='log',xlabel='Median first-decision time (ms)',ylabel='Mean optimality gap (%)');ax.grid(True,which='both',alpha=.25);ax.legend();fig.tight_layout();fig.savefig(R/'fig_tradeoff.png',dpi=220);plt.close(fig)
fig,ax=plt.subplots(figsize=(5.2,3.2))
for f,l,m in [('iid','IID','o'),('stress','Low-feasibility stress','s')]:
 a=[x for x in sc if x['family']==f];n=[x['n'] for x in a];mid=np.array([x['median_ms'] for x in a]);ax.errorbar(n,mid,yerr=[mid-np.array([x['min_ms'] for x in a]),np.array([x['max_ms'] for x in a])-mid],marker=m,markerfacecolor='none',capsize=3,label=l)
ax.set(xlabel='Number of domains',ylabel='First-decision time (ms)',yscale='log',xticks=[20,40,60,80,100]);ax.legend();fig.tight_layout();fig.savefig(R/'fig_scaling.png',dpi=220);plt.close(fig)
for a in groups:
 if a['group'] in ('N10-12','8','10','12'): print(a)
print('CI',ci);print('ONLINE',[a for a in on if a['group']=='N10-12']);print('SCALING',sc)
