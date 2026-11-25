"""Chronologically separated training / budget calibration / frozen holdout.

Costs and empirical conditional miss risk are reported as separate outcomes.
Budget calibration is an engineering gate, not a statistical risk certificate.
"""
import hashlib, json, math, time
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from ia_osr_experiment import Params
from ia_osr_psa_audit import PSASolver,clear_all
import decision_evaluation as de

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/alibaba2018'
DAY=86400; STEP=10
INTENTS=((40,70),(60,80),(80,90));LAGS=(60,600,3600)
BUDGETS=(0.,.01,.025,.05,.075,.10)
METHODS=de.METHODS+('psa2',)
FIELDS=de.FIELDS

def domain_states(cpu,mem,using,cpu_limit,mem_limit):
    """Four domains, two hosts each, three-valued short-circuit predicates."""
    pos=(cpu<=cpu_limit)&(mem<=mem_limit)&(using==1)
    neg=(cpu>cpu_limit)|(mem>mem_limit)|(using==0)
    yes=pos.reshape(4,2,-1).any(axis=1)
    no=neg.reshape(4,2,-1).all(axis=1)
    return np.where(yes,1,np.where(no,0,-1)).astype(np.int8).T

def extended_tables(par):
    base,expected=de.make_tables(par)
    so=PSASolver(par)
    @lru_cache(None)
    def H(s,k):
        if so.stop(s):return 0.
        if k==0:return so.HPSA1fast(s)
        return min(score(s,a,k) for a in so.actions(s))
    def score(s,a,k):
        c,br=so.successors(s,a)
        return c+sum(pr*H(ns,k-1) for pr,ns in br if pr>0)
    @lru_cache(None)
    def choose(s):
        if so.stop(s):return ('STOP',-1)
        order={'STOP':0,'SKIP':1,'Q':2,'R':3}
        return min(so.actions(s),key=lambda a:(round(score(s,a,2),12),order[a[0]],a))
    @lru_cache(None)
    def co(s):
        if so.stop(s):return 0.
        c,br=so.successors(s,choose(s))
        return c+sum(pr*co(ns) for pr,ns in br if pr>0)
    extra=np.empty_like(base[0]);n=par.n()
    for k in range(1<<(2*n)):
        truth=[bool(k&(1<<i)) for i in range(n)];obs=[bool(k&(1<<(i+n))) for i in range(n)]
        s=so.initial;cost=qn=rn=sn=0
        for _ in range(4*n+4):
            op,i=choose(s)
            if op=='STOP':
                risk=so.risk(s);assert risk<=par.eps+1e-10
                extra[k]=[cost,0,1,int(any(truth[j] and x!=3 for j,x in enumerate(s))),qn,rn,sn,risk];break
            if op=='Q':
                cost+=par.q[i];qn+=1
                if truth[i]:extra[k]=[cost,1,0,0,qn,rn,sn,0];break
                s=so.set(s,i,3)
            elif op=='R':cost+=par.r[i];rn+=1;s=so.set(s,i,1 if obs[i] else 2)
            elif op=='SKIP':sn+=1;s=so.set(s,i,s[i]+4)
        else:raise AssertionError('Nonterminating policy')
    exp=float(de.independent_joint(par)@extra[:,0])
    assert abs(exp-co(so.initial))<1e-9
    assert so.exact(so.initial)<=exp+1e-9<=H(so.initial,2)+2e-9
    clear_all()
    return np.concatenate([base,extra[None]],axis=0),np.r_[expected,exp]

def split_masks(t):
    return dict(train=(t>=0)&(t<3*DAY),calibration=(t>=3*DAY+3600)&(t<5*DAY),test=(t>=5*DAY+3600)&(t<8*DAY))

def pairs(z,lag):
    back=lag//STEP; x=z[back:];y=z[:-back]
    known=(x>=0).all(axis=1)&(y>=0).all(axis=1)
    return back,x,y,known

def select_budget(rows,method):
    if method=='query_only':return 0.
    valid=[r['epsilon'] for r in rows if r['method']==method and r['epsilon']>0 and r['empty_stop_n']>=200 and r['conditional_miss']<=.05]
    return max(valid,default=0.)

def summary_values(v):
    n=len(v);s=v.sum(axis=0)
    stops=int(round(s[2]));miss=int(round(s[3]))
    return dict(n=n,mean_cost=float(s[0]/n) if n else np.nan,success_n=int(round(s[1])),empty_stop_n=stops,
                miss_n=miss,conditional_miss=miss/stops if stops else np.nan,miss_per_request=miss/n if n else np.nan,
                mean_queries=float(s[4]/n) if n else np.nan,mean_refreshes=float(s[5]/n) if n else np.nan,
                mean_skips=float(s[6]/n) if n else np.nan,mean_model_stop_risk=float(s[7]/stops) if stops else np.nan)

def calendar_bins(t,vals,baseline,start=5*DAY+3600,end=8*DAY):
    # Ten-minute calendar cells; moving blocks contain six consecutive cells.
    n=math.ceil((end-start)/600);ix=((t-start)//600).astype(int)
    a=np.zeros((n,5));np.add.at(a[:,0],ix,1);np.add.at(a[:,1],ix,vals[:,0]);np.add.at(a[:,2],ix,baseline)
    np.add.at(a[:,3],ix,vals[:,2]);np.add.at(a[:,4],ix,vals[:,3])
    return a

def bootstrap(a,draws=1000,seed=1701):
    rng=np.random.default_rng(seed);n=len(a);length=6
    starts=rng.integers(0,n-length+1,size=(draws,math.ceil(n/length)))
    ix=(starts[:,:,None]+np.arange(length)).reshape(draws,-1)[:,:n]
    sums=a[ix].sum(axis=1)
    def ci(num,den):
        good=den>0
        return tuple(float(x) for x in np.quantile(num[good]/den[good],[.025,.975]))+(int(good.sum()),) if good.any() else (np.nan,np.nan,0)
    c=ci(sums[:,1],sums[:,0]);r=ci(sums[:,4],sums[:,3]);g=ci(100*(sums[:,2]-sums[:,1]),sums[:,2])
    return dict(cost_lo=c[0],cost_hi=c[1],conditional_miss_lo=r[0],conditional_miss_hi=r[1],risk_bootstrap_defined=r[2],
                saving_percent_lo=g[0],saving_percent_hi=g[1])

def main():
    data=np.load(OUT/'alibaba2018_measured_subset.npz');t=data['time_s'];splits=split_masks(t)
    states={};common={};state_rows=[]
    for layout in range(3):
        for it,(cl,ml) in enumerate(INTENTS):
            h=slice(layout*8,(layout+1)*8)
            z=domain_states(data['cpu'][h],data['mem'][h],data['using'][h],cl,ml);states[layout,it]=z
            for part,mask in splits.items():
                for d in range(4):
                    vals=z[mask,d];known=vals>=0;v=vals[known]
                    adjacent=(vals[:-1]>=0)&(vals[1:]>=0)
                    state_rows.append(dict(layout=layout,intent=it,cpu_limit=cl,mem_limit=ml,split=part,domain=d,
                        source_failure_domain_2=float(data['failure_domain_2'][layout*8+2*d]),calendar_points=int(mask.sum()),
                        known_points=int(known.sum()),known_fraction=float(known.mean()),positive_fraction=float(v.mean()) if len(v) else np.nan,
                        adjacent_known_pairs=int(adjacent.sum()),adjacent_transitions=int(((vals[:-1]!=vals[1:])&adjacent).sum())))
        cm=np.ones(len(t),bool)
        for it in range(len(INTENTS)):
            for lag in LAGS:
                b,x,y,k=pairs(states[layout,it],lag);tmp=np.zeros(len(t),bool);tmp[b:]=k;cm&=tmp
        common[layout]=cm
    pd.DataFrame(state_rows).to_csv(OUT/'state_prevalence.csv',index=False)
    np.savez_compressed(OUT/'domain_states.npz',time_s=t,**{f'layout{a}_intent{b}':z for (a,b),z in states.items()},
                         **{f'common_mask_layout{k}':v for k,v in common.items()})
    profiles=[];grid_rows=[];choices=[];models=[];began=time.monotonic()
    # Phase A: training and calibration only. No test outcomes are evaluated here.
    for layout in range(3):
      for it in range(len(INTENTS)):
       for lag in LAGS:
        b,x,y,known=pairs(states[layout,it],lag);tt=t[b:]
        tr=known&splits['train'][b:];ca=known&splits['calibration'][b:]
        if tr.sum()<100 or ca.sum()<100:raise ValueError('Insufficient train/calibration paired observations')
        p,alpha,beta=de.calibrate(x[tr].astype(bool),y[tr].astype(bool));calcode=de.codes(x[ca],y[ca])
        for costname,q in [('equal',(1.,1.,1.,1.)),('heterogeneous',(4.,2.,1.,3.))]:
            pid=len(profiles);meta=dict(profile=pid,layout=layout,intent=it,cpu_limit=INTENTS[it][0],mem_limit=INTENTS[it][1],lag_s=lag,costs=costname)
            tables={};expected={};local=[]
            for eps in BUDGETS:
                par=Params(p,tuple(q),tuple(.05*v for v in q),alpha,beta,eps)
                tab,ex=extended_tables(par);tables[eps]=tab;expected[eps]=ex
                for j,method in enumerate(METHODS):
                    row=dict(meta,method=method,epsilon=eps,**summary_values(tab[j,calcode]));local.append(row);grid_rows.append(row)
            selection={m:select_budget(local,m) for m in METHODS}
            choices.append(dict(meta,selected_epsilon=selection))
            models.append(dict(meta,p=p,alpha=alpha,beta=beta,q=q,r=tuple(.05*v for v in q),train_n=int(tr.sum()),calibration_n=int(ca.sum()),
                               train_correlations=de.correlations(x[tr])))
            profiles.append(dict(meta=meta,tables=tables,expected=expected,selection=selection,models=models[-1],back=b,x=x,y=y,known=known))
            print(json.dumps({'profile_frozen':pid,'layout':layout,'intent':it,'lag_s':lag,'costs':costname,'elapsed_s':round(time.monotonic()-began,1)}),flush=True)
    choicepath=OUT/'calibration_choices_frozen_before_test.json'
    choicepath.write_text(json.dumps(dict(target_test_risk=.10,calibration_gate=.05,min_calibration_empty_stops=200,choices=choices),indent=2)+'\n')
    freezehash=hashlib.sha256(choicepath.read_bytes()).hexdigest()
    pd.DataFrame(grid_rows).to_csv(OUT/'calibration_budget_grid.csv',index=False)
    # Phase B: all decisions above are frozen; evaluate the held-out time interval.
    results=[];reliability=[];briers=[];bins=[]
    for pr in profiles:
        meta=pr['meta'];b=pr['back'];x=pr['x'];y=pr['y'];known=pr['known'];tt=t[b:]
        code=de.codes(np.maximum(x,0),np.maximum(y,0))
        mod=pr['models'];par=Params(tuple(mod['p']),tuple(mod['q']),tuple(mod['r']),tuple(mod['alpha']),tuple(mod['beta']),.10)
        so=PSASolver(par)
        for part,mask0 in splits.items():
            mask=known&mask0[b:];xx=x[mask];yy=y[mask];n=len(xx)
            for d in range(4):
                pred=np.where(yy[:,d]==1,so.pp[d],so.pm[d])
                briers.append(dict(meta,split=part,domain=d,n=n,prior=float(par.p[d]),observed_prevalence=float(xx[:,d].mean()),
                        prior_brier=float(np.mean((xx[:,d]-par.p[d])**2)),posterior_brier=float(np.mean((xx[:,d]-pred)**2))))
                for signal in (0,1):
                    ix=yy[:,d]==signal;nn=int(ix.sum())
                    reliability.append(dict(meta,split=part,domain=d,signal=signal,n=nn,
                        predicted=so.pp[d] if signal else so.pm[d],positive_n=int(xx[ix,d].sum()),observed=float(xx[ix,d].mean()) if nn else np.nan))
        scopes=[('per_lag_known',known&splits['test'][b:])]
        cm=common[meta['layout']][b:]&splits['test'][b:]
        if cm.sum()>=500:scopes.append(('all_intents_lags_common',cm))
        for scope,mask in scopes:
            cc=code[mask];times=tt[mask]
            baseline=pr['tables'][0.][0,cc,0];basecost=float(baseline.mean())
            for protocol in ('zero_risk_control','nominal_10pct','calibration_selected'):
                for j,method in enumerate(METHODS):
                    eps=0. if protocol=='zero_risk_control' else .10 if protocol=='nominal_10pct' else pr['selection'][method]
                    vals=pr['tables'][eps][j,cc]
                    row=dict(meta,scope=scope,protocol=protocol,method=method,epsilon=eps,**summary_values(vals))
                    row.update(query_only_mean_cost=basecost,saving_percent=100*(1-row['mean_cost']/basecost),
                               model_expected_cost=float(pr['expected'][eps][j]),
                               model_optimal_cost=float(pr['expected'][eps][METHODS.index('exact')]))
                    a=calendar_bins(times,vals,baseline);row.update(bootstrap(a))
                    row['risk_interval_status']='undefined_no_empty_stop' if row['empty_stop_n']==0 else 'zero_events_bootstrap_degenerate' if row['miss_n']==0 else 'descriptive_1h_moving_block'
                    assert row['success_n']+row['empty_stop_n']==row['n']
                    if eps==0 or method=='query_only':assert row['miss_n']==0
                    results.append(row)
                    for k,v in enumerate(a):
                        bins.append(dict(profile=meta['profile'],scope=scope,protocol=protocol,method=method,epsilon=eps,calendar_bin_10min=k,
                                         n=int(v[0]),cost_sum=v[1],query_only_cost_sum=v[2],empty_stop_n=int(v[3]),miss_n=int(v[4])))
        clear_all()
    pd.DataFrame(results).to_csv(OUT/'policy_summary.csv',index=False)
    pd.DataFrame(reliability).to_csv(OUT/'posterior_reliability.csv',index=False)
    pd.DataFrame(briers).to_csv(OUT/'brier_scores.csv',index=False)
    pd.DataFrame(bins).to_csv(OUT/'calendar_policy_statistics.csv.gz',index=False,compression='gzip')
    (OUT/'fitted_models.json').write_text(json.dumps(models,indent=2)+'\n')
    metadata=dict(profiles=len(profiles),fitted_probability_models=3*len(INTENTS)*len(LAGS),fitted_parameter_budget_models=len(profiles)*len(BUDGETS),methods=METHODS,policy_result_rows=len(results),
        intents=INTENTS,lags_seconds=LAGS,budgets=BUDGETS,train_seconds=[0,3*DAY],calibration_seconds=[3*DAY+3600,5*DAY],test_seconds=[5*DAY+3600,8*DAY],
        calibration_choices_sha256=freezehash,common_mask_test_points={str(k):int((v&splits['test']).sum()) for k,v in common.items()},
        evaluated_common_mask_minimum=500,bootstrap_block_seconds=3600,bootstrap_draws=1000,bootstrap_seed=1701,
        selection_guarantee='Empirical engineering gate only; no finite-sample or distribution-shift risk guarantee',
        exact_reference='Optimal only under the fitted independent-domain model',elapsed_s=time.monotonic()-began)
    (OUT/'experiment_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':main()
