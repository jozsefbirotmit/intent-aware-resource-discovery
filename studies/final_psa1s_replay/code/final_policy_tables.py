"""Replay the unchanged final benchmark implementation on all binary outcomes.

The wrappers provide measured X/Y inputs; they do not change any solver rule.
Paid-action paths are retained to distinguish harmless SKIP removal from ties.
"""
from pathlib import Path
from functools import lru_cache
import math,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/final_algorithm'))
import numpy as np
import benchmark as final
from ia_osr_experiment import Params
from ia_osr_psa_audit import PSASolver,clear_all
import legacy_evaluation as legacy

METHODS=('query_only','no_refresh_opt','psa1','psa1_s','legacy_exact')
FIELDS=('cost','success','empty_stop','miss','queries','refreshes','skips','model_stop_risk')

def parameter(m,epsilon=None):
    return Params(**{k:tuple(float(v) for v in m[k]) for k in ('p','q','r','alpha','beta')},
                  eps=float(m.get('epsilon',m.get('eps',.1)) if epsilon is None else epsilon))

def execute(sol,choose,truth,signal):
    s=sol.initial;cost=0.;qn=rn=sn=0;paid=[]
    for _ in range(4*sol.n+4):
        op,i=choose(s)
        if op=='STOP':
            risk=sol.risk(s);assert risk<=sol.a.eps+1e-10
            miss=int(any(truth[j] and k!=3 for j,k in enumerate(s)))
            return np.array([cost,0,1,miss,qn,rn,sn,risk],float),';'.join(paid+['STOP'])
        if op=='Q':
            cost+=sol.a.q[i];qn+=1;paid.append(f'Q{i}')
            if truth[i]:return np.array([cost,1,0,0,qn,rn,sn,0],float),';'.join(paid+[f'FOUND{i}'])
            s=sol.set(s,i,3)
        elif op=='R':cost+=sol.a.r[i];rn+=1;paid.append(f'R{i}');s=sol.set(s,i,1 if signal[i] else 2)
        elif op=='SKIP':sn+=1;s=sol.set(s,i,s[i]+4)
        else:raise ValueError(op)
    raise AssertionError('Nonterminating replay')

def no_refresh_table(par):
    """Same 2^N recurrence as benchmark.no_refresh, plus executable actions."""
    n=par.n();v=np.empty(1<<n);surv=np.empty_like(v);actions=np.full(1<<n,-1,np.int16);v[0]=0.;surv[0]=1.
    for m in range(1,1<<n):
        bit=m&-m;i=bit.bit_length()-1;surv[m]=surv[m^bit]*(1-par.p[i])
        if 1-surv[m]<=par.eps+1e-12:v[m]=0.;continue
        best=np.inf;best_i=-1;left=m
        while left:
            b=left&-left;j=b.bit_length()-1;val=par.q[j]+(1-par.p[j])*v[m^b]
            if val<best:best=val;best_i=j
            left^=b
        v[m]=best;actions[m]=best_i
    assert abs(v[-1]-final.no_refresh(par))<1e-10
    return v,actions

@lru_cache(None)
def make_tables(par):
    n=par.n();worlds=1<<(2*n)
    if n>8:raise ValueError('This targeted replay deliberately covers the existing small-domain studies only')
    table=np.empty((len(METHODS),worlds,len(FIELDS)));paths=np.empty((len(METHODS),worlds),object)
    sol=PSASolver(par);nv,na=no_refresh_table(par)
    roots={}
    def nr(s):
        mask=sum(1<<i for i,k in enumerate(s) if k!=3)
        i=int(na[mask]);return ('STOP',-1) if i<0 else ('Q',i)
    for method in ('query_only','no_refresh_opt','legacy_exact'):
        j=METHODS.index(method)
        choose=nr if method=='no_refresh_opt' else lambda s,m=method:legacy.action(sol,s,'exact' if m=='legacy_exact' else m)
        roots[method]=choose(sol.initial)
        for code in range(worlds):
            x=tuple(bool(code&(1<<i)) for i in range(n));y=tuple(bool(code&(1<<(i+n))) for i in range(n))
            table[j,code],paths[j,code]=execute(sol,choose,x,y)
    legacy_opt=sol.exact(sol.initial);legacy_psa=sol.psa_fast_cost(sol.initial)
    clear_all()
    root_values={};meta={}
    for method,name in [('psa1','psa'),('psa1_s','selective8')]:
        val_engine=final.engine(par,name);d=val_engine.evaluate();root_values[method]=d['cost'];final.release(val_engine)
        e=final.engine(par,name);j=METHODS.index(method)
        choose=(lambda s:e.plan(s)[0]) if method=='psa1_s' else (lambda s:e.action(s,1)[0])
        roots[method]=choose(e.sol.initial)
        for code in range(worlds):
            x=tuple(bool(code&(1<<i)) for i in range(n));y=tuple(bool(code&(1<<(i+n))) for i in range(n))
            table[j,code],paths[j,code]=execute(e.sol,choose,x,y)
        if method=='psa1_s':
            meta=dict(selective_refinements=e.refined,tail_builds=e.builds,
                      initial_stop=bool(e.sol.stop(e.sol.initial)),root_depth=int(d['root_depth']))
            assert e.refined==0 and d['root_depth']==0
        final.release(e)
    mass=legacy.independent_joint(par);expected=mass@table[:,:,0].T
    assert abs(expected[METHODS.index('legacy_exact')]-legacy_opt)<1e-9
    assert abs(expected[METHODS.index('no_refresh_opt')]-nv[-1])<1e-9
    assert abs(expected[METHODS.index('psa1')]-legacy_psa)<1e-9
    for name in root_values:assert abs(expected[METHODS.index(name)]-root_values[name])<1e-9
    assert abs(expected[METHODS.index('psa1_s')]-legacy_opt)<1e-9
    assert legacy_opt<=nv[-1]+1e-9
    assert np.all(table[:,:,1]+table[:,:,2]==1)
    if par.eps==0:assert np.all(table[:,:,3]==0)
    si=METHODS.index('psa1_s');li=METHODS.index('legacy_exact')
    flags=dict(cost=np.abs(table[si,:,0]-table[li,:,0])>1e-9,
               miss=table[si,:,3]!=table[li,:,3],empty_stop=table[si,:,2]!=table[li,:,2],
               paid_path=paths[si]!=paths[li])
    meta.update(root_actions=roots,model_expected_costs={k:float(expected[i]) for i,k in enumerate(METHODS)},
                all_outcome_cost_disagreements=int(flags['cost'].sum()),all_outcome_miss_disagreements=int(flags['miss'].sum()),
                all_outcome_stop_disagreements=int(flags['empty_stop'].sum()),all_outcome_paid_path_disagreements=int(flags['paid_path'].sum()))
    return table,expected,flags,meta

def outcome_codes(x,y):
    w=1<<np.arange(x.shape[1]);return x.astype(int)@w+(1<<x.shape[1])*(y.astype(int)@w)

def values_summary(values):
    n=len(values);v=values.sum(axis=0);stops=int(round(v[2]));miss=int(round(v[3]))
    return dict(n=n,mean_cost=float(v[0]/n),success_n=int(round(v[1])),empty_stop_n=stops,miss_n=miss,
                conditional_miss=miss/stops if stops else np.nan,miss_per_request=miss/n,
                mean_queries=float(v[4]/n),mean_refreshes=float(v[5]/n),mean_skips=float(v[6]/n),
                max_model_stop_risk=float(values[values[:,2]==1,7].max()) if stops else np.nan)
