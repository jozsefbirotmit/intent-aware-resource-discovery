"""Exact Bellman vs IA-OSR one-step lookahead for independent domains.
Model: one refresh per domain, terminal when one feasible resource found or residual risk <= eps.
Each domain can be unrefreshed (0), refreshed +/- (1/2), query-failed (3), irrevocably skipped in any belief state (4/5/6).
Only skip actions with already skipped posterior risk <= eps. Query is exact.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import prod
import random
import statistics
import time
import json
import argparse

@dataclass(frozen=True)
class Params:
    p:tuple; q:tuple; r:tuple; alpha:tuple; beta:tuple; eps:float
    def n(self): return len(self.p)

class Solver:
    def __init__(self,par):
        self.a=par
        n=par.n(); self.n=n
        self.g=tuple(par.p[i]*par.alpha[i]+(1-par.p[i])*(1-par.beta[i]) for i in range(n))
        self.h=tuple(1-v for v in self.g)
        self.pp=tuple(par.p[i]*par.alpha[i]/self.g[i] if self.g[i] else 0. for i in range(n))
        self.pm=tuple(par.p[i]*(1-par.alpha[i])/self.h[i] if self.h[i] else 0. for i in range(n))
        self.initial=(0,)*n
    def belief(self,i,k):
        if k in (0,4):return self.a.p[i]
        if k in (1,5):return self.pp[i]
        if k in (2,6):return self.pm[i]
        return 0.
    def unfinished(self,s):return [i for i,k in enumerate(s) if k in (0,1,2)]
    def risk(self,s,skipped_only=False):
        return 1-prod(1-self.belief(i,k) for i,k in enumerate(s) if k!=3 and (not skipped_only or k>=4))
    def stop(self,s): return self.risk(s)<=self.a.eps+1e-12
    def set(self,s,i,v):return s[:i]+(v,)+s[i+1:]
    def actions(self,s):
        if self.stop(s):return [('STOP',-1)]
        acts=[]
        sk=self.risk(s,True)
        for i in self.unfinished(s):
            acts.append(('Q',i))
            if s[i]==0:acts.append(('R',i))
            skip_r=1-(1-sk)*(1-self.belief(i,s[i]))
            if skip_r<=self.a.eps+1e-12:acts.append(('SKIP',i))
        return acts
    def successors(self,s,a):
        name,i=a
        if name=='STOP':return 0.,[]
        if name=='SKIP':return 0.,[(1.,self.set(s,i,s[i]+4))]
        if name=='Q':return self.a.q[i],[(1-self.belief(i,s[i]),self.set(s,i,3))]
        if name=='R':return self.a.r[i],[(self.g[i],self.set(s,i,1)),(self.h[i],self.set(s,i,2))]
        raise ValueError(a)

    @lru_cache(None)
    def exact(self,s):
        if self.stop(s):return 0.
        return min(self.q_exact(s,a) for a in self.actions(s))
    def q_exact(self,s,a):
        c,branches=self.successors(s,a)
        return c+sum(pr*self.exact(sn) for pr,sn in branches if pr>0)

    @lru_cache(None)
    def HQ(self,s):
        if self.stop(s):return 0.
        arr=self.unfinished(s)
        arr.sort(key=lambda i:(self.a.q[i]/self.belief(i,s[i]) if self.belief(i,s[i])>0 else float('inf'),i))
        survival=1.;cost=0.
        for i in arr:
            cost+=survival*self.a.q[i]
            survival*=1-self.belief(i,s[i])
        return cost

    @lru_cache(None)
    def H1(self,s):
        if self.stop(s):return 0.
        arr=self.unfinished(s)
        sk=self.risk(s,True)
        best=self.HQ(s)
        for i in arr:
            if s[i]!=0:continue
            skip_r=1-(1-sk)*(1-self.pm[i])
            if skip_r>self.a.eps+1e-12:continue
            modules=[(self.a.q[j],self.belief(j,s[j])) if j!=i else (self.a.r[i]+self.g[i]*self.a.q[i],self.a.p[i]*self.a.alpha[i]) for j in arr]
            modules.sort(key=lambda x:x[0]/x[1] if x[1]>0 else float('inf'))
            surv=1.; cost=0.
            for c,success in modules:
                cost+=surv*c
                surv*=1-success
            best=min(best,cost)
        return best

    @lru_cache(None)
    def H1SC(self,s):
        """Skip-closed baseline: arbitrary admissible upfront direct skips, then Q-only or <=one screening fixed module.
        Evaluation exponential in n, used as a correctness/quality reference, not proposed scalable H1.
        """
        if self.stop(s):return 0.
        arr=self.unfinished(s)
        already=self.risk(s,True)
        best=float('inf')
        for mask in range(1<<len(arr)):
            skipped=[j for k,j in enumerate(arr) if (mask>>k)&1]
            remain=[j for k,j in enumerate(arr) if not ((mask>>k)&1)]
            sk=1-(1-already)*prod(1-self.belief(j,s[j]) for j in skipped)
            if sk>self.a.eps+1e-12:continue
            if not remain:best=0.;break
            modules=[(self.a.q[j],self.belief(j,s[j])) for j in remain]
            def seq_cost(x):
                z=sorted(x,key=lambda v:v[0]/v[1] if v[1]>0 else float('inf'))
                survival=1.;cost=0.
                for c,success in z:
                    cost+=survival*c
                    survival*=1-success
                return cost
            best=min(best,seq_cost(modules))
            for i in remain:
                if s[i]!=0:continue
                if 1-(1-sk)*(1-self.pm[i])>self.a.eps+1e-12:continue
                with_screen=[(self.a.r[j]+self.g[j]*self.a.q[j],self.a.p[j]*self.a.alpha[j]) if j==i else (self.a.q[j],self.belief(j,s[j])) for j in remain]
                best=min(best,seq_cost(with_screen))
        return best

    @lru_cache(None)
    def rollout_action(self,s,mode='H1'):
        if self.stop(s):return ('STOP',-1)
        H={'H1':self.H1,'HQ':self.HQ,'H1SC':self.H1SC}[mode]
        values=[]
        for a in self.actions(s):
            c,branches=self.successors(s,a)
            val=c+sum(pr*H(sn) for pr,sn in branches if pr>0)
            values.append((round(val,12),{'STOP':0,'SKIP':1,'Q':2,'R':3}[a[0]],a))
        return min(values)[2]

    @lru_cache(None)
    def rollout_cost(self,s,mode='H1'):
        if self.stop(s):return 0.
        a=self.rollout_action(s,mode)
        c,branches=self.successors(s,a)
        return c+sum(pr*self.rollout_cost(sn,mode) for pr,sn in branches if pr>0)

    @lru_cache(None)
    def base_action(self,s,mode='H1'):
        # For actual evaluation of H1 baseline, irrelevant: H1 is evaluated analytically.
        pass

    def info(self):
        s=self.initial
        opt=self.exact(s)
        co=self.rollout_cost(s,'H1')
        cq=self.rollout_cost(s,'HQ')
        csc=self.rollout_cost(s,'H1SC')
        return {'n':self.n,'eps':self.a.eps,'optimal':opt,'IA_OSR_H1':co,'IA_OSR_HQ':cq,'IA_OSR_H1SC':csc,'H1SC_base':self.H1SC(s),'H1_base':self.H1(s),'HQ_base':self.HQ(s),'gap_rel':(co/opt-1) if opt>0 else 0.,'gap_abs':co-opt,'first_opt':min(self.actions(s),key=lambda a:self.q_exact(s,a)),'first_H1':self.rollout_action(s,'H1'),'first_HQ':self.rollout_action(s,'HQ'),'first_H1SC':self.rollout_action(s,'H1SC'),'states_exact':self.exact.cache_info().currsize}


def make_random(n,rng,eps_type='mixed'):
    p=tuple(rng.uniform(.05,.85) for _ in range(n))
    q=tuple(10**rng.uniform(-.2,1.1) for _ in range(n))
    r=tuple(q[i]*10**rng.uniform(-1.4,.15) for i in range(n))
    alpha=tuple(rng.uniform(.65,.995) for _ in range(n))
    beta=tuple(rng.uniform(.65,.995) for _ in range(n))
    if eps_type=='mixed':eps=rng.choice([.01,.03,.05,.1,.2,.3])
    elif eps_type=='strict':eps=rng.choice([.005,.01,.02])
    else:eps=rng.choice([.1,.2,.3])
    return Params(p,q,r,alpha,beta,eps)


def run_study(counts,seed=17,keep=20):
    rng=random.Random(seed); rows=[];fails=[]
    for n,count in counts.items():
        ts=time.monotonic()
        for k in range(count):
            par=make_random(n,rng)
            sol=Solver(par)
            info=sol.info()
            info['params']={'p':par.p,'q':par.q,'r':par.r,'alpha':par.alpha,'beta':par.beta,'eps':par.eps}
            rows.append(info)
            if info['gap_abs']>1e-7:fails.append(info)
            # @lru_cache decorated methods cache by solver instance; clear to avoid retained solver objects.
            for meth in (Solver.exact, Solver.HQ, Solver.H1, Solver.H1SC, Solver.rollout_action, Solver.rollout_cost): meth.cache_clear()
        these=[x for x in rows if x['n']==n]
        print('n',n,'total',len(these),'nonexact',sum(x['gap_abs']>1e-7 for x in these),'mean gap',statistics.mean(x['gap_rel'] for x in these), 'max gap',max(x['gap_rel'] for x in these), 'elapsed',round(time.monotonic()-ts,1),flush=True)
    return rows,sorted(fails,key=lambda z:z['gap_rel'],reverse=True)[:keep]

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--counts',default='1:100,2:400,3:500,4:500,5:250');parser.add_argument('--seed',type=int,default=17);parser.add_argument('--out',default='/mnt/data/ia_osr_results.json')
    args=parser.parse_args()
    counts={int(part.split(':')[0]):int(part.split(':')[1]) for part in args.counts.split(',')}
    rows,fails=run_study(counts,args.seed)
    import pathlib
    pathlib.Path(args.out).write_text(json.dumps({'metadata':{'seed':args.seed,'counts':counts,'policy':'one refresh per domain, exact query, explicit skip with posterior epsilon budget, one successful resource required'},'rows':rows,'largest_gaps':fails},indent=2),encoding='utf8')
    print('saved',args.out,'worst', fails[0] if fails else None,flush=True)
