"""Audit of a polynomial, skip-aware continuation for IA-OSR.

Maintains the exact decision semantics of ia_osr_experiment.py:
independent domains, exact Q, at most one binary R/domain,
irrevocable SKIP, and STOP once combined posterior risk <= epsilon.

H_PSA1: choose at most one upfront admissible direct SKIP; then
an optimally ordered fixed-module plan containing Q-only or one
screening module. Complexity of an unoptimized base evaluation is
O(N^3 log N) since there are <= N+1 skip choices and <=N screens.
"""
import argparse
import csv
import json
import math
import random
import statistics
import time
from functools import lru_cache
from pathlib import Path

from ia_osr_experiment import Params, Solver, make_random


class PSASolver(Solver):
    @lru_cache(maxsize=None)
    def HPSA1(self, s):
        if self.stop(s):
            return 0.0
        active = self.unfinished(s)
        prior_skip_risk = self.risk(s, skipped_only=True)
        best = float('inf')
        # Empty or singleton upfront skip: O(N) candidates.
        skip_options = [()] + [(i,) for i in active]
        for skip in skip_options:
            if skip:
                j = skip[0]
                skip_risk = 1.0 - (1.0-prior_skip_risk)*(1.0-self.belief(j,s[j]))
                if skip_risk > self.a.eps+1e-12:
                    continue
            else:
                skip_risk = prior_skip_risk
            remain = [j for j in active if j not in skip]
            if not remain:
                best = min(best, 0.0)
                continue
            def sequence_cost(modules):
                # A module is (expected internal cost, terminating success prob).
                seq = sorted(modules, key=lambda x: (x[0]/x[1] if x[1]>0 else float('inf')))
                survival, out = 1.0, 0.0
                for cost, success in seq:
                    out += survival*cost
                    survival *= (1.0-success)
                return out
            query_mods = [(self.a.q[j],self.belief(j,s[j])) for j in remain]
            best = min(best,sequence_cost(query_mods))
            # At most one screening module, require negative-branch skip feasibility.
            for i in remain:
                if s[i] != 0:
                    continue
                after_negative_skip = 1.0-(1.0-skip_risk)*(1.0-self.pm[i])
                if after_negative_skip > self.a.eps+1e-12:
                    continue
                modules = [(self.a.r[j]+self.g[j]*self.a.q[j], self.a.p[j]*self.a.alpha[j]) if j==i
                           else (self.a.q[j],self.belief(j,s[j])) for j in remain]
                best = min(best,sequence_cost(modules))
        return best

    @lru_cache(maxsize=None)
    def HPSA1fast(self,s):
        """Exactly the PSA1 policy family, but O(N^2 log N) base evaluation.

        For each possible screen index, sort its fixed module sequence only once,
        then evaluate every possible singleton direct-skip by prefix/suffix
        expected-cost products in O(1) each.
        """
        if self.stop(s):
            return 0.0
        active=self.unfinished(s)
        prior_sk=self.risk(s,skipped_only=True)
        direct_risk={j:1.0-(1.0-prior_sk)*(1.0-self.belief(j,s[j])) for j in active}

        def eval_modules(modules,screen_idx=None):
            # modules = (domain index, expected module cost, success probability)
            ordered=sorted(modules,key=lambda z: (z[1]/z[2] if z[2]>0 else float('inf'),z[0]))
            m=len(ordered)
            prefC=[0.0]*(m+1)
            prefF=[1.0]*(m+1)
            suffixC=[0.0]*(m+1)
            for k,(idx,c,success) in enumerate(ordered):
                prefC[k+1]=prefC[k]+prefF[k]*c
                prefF[k+1]=prefF[k]*(1.0-success)
            for k in range(m-1,-1,-1):
                _,c,success=ordered[k]
                suffixC[k]=c+(1.0-success)*suffixC[k+1]
            val=prefC[m]
            for k,(j,_,__) in enumerate(ordered):
                if screen_idx==j:  # cannot skip the screened domain outright
                    continue
                risk=direct_risk[j]
                if risk > self.a.eps+1e-12:
                    continue
                if screen_idx is not None:
                    neg_skip_risk=1.0-(1.0-risk)*(1.0-self.pm[screen_idx])
                    if neg_skip_risk>self.a.eps+1e-12:
                        continue
                # Query-module j (not screen) removed from fixed sequence.
                candidate=prefC[k]+prefF[k]*suffixC[k+1]
                val=min(val,candidate)
            return val

        queries=[(j,self.a.q[j],self.belief(j,s[j])) for j in active]
        best=eval_modules(queries)
        for i in active:
            if s[i]!=0:
                continue
            neg_skip_risk=1.0-(1.0-prior_sk)*(1.0-self.pm[i])
            if neg_skip_risk>self.a.eps+1e-12:
                continue
            mods=[(j,self.a.r[j]+self.g[j]*self.a.q[j],self.a.p[j]*self.a.alpha[j])
                  if j==i else (j,self.a.q[j],self.belief(j,s[j])) for j in active]
            best=min(best,eval_modules(mods,screen_idx=i))
        return best

    @lru_cache(maxsize=None)
    def psa_fast_action(self,s):
        if self.stop(s):
            return ('STOP',-1)
        scores=[]
        for a in self.actions(s):
            c,branches=self.successors(s,a)
            v=c+sum(pr*self.HPSA1fast(suc) for pr,suc in branches if pr>0)
            scores.append((round(v,12),{'STOP':0,'SKIP':1,'Q':2,'R':3}[a[0]],a))
        return min(scores)[2]

    @lru_cache(maxsize=None)
    def psa_fast_cost(self,s):
        if self.stop(s):
            return 0.0
        a=self.psa_fast_action(s)
        c,branches=self.successors(s,a)
        return c+sum(pr*self.psa_fast_cost(suc) for pr,suc in branches if pr>0)

    @lru_cache(maxsize=None)
    def psa_action(self,s):
        if self.stop(s):
            return ('STOP',-1)
        scores=[]
        for a in self.actions(s):
            c,branches=self.successors(s,a)
            v=c+sum(pr*self.HPSA1(suc) for pr,suc in branches if pr>0)
            # Explicit deterministic tie-breaking.
            scores.append((round(v,12),{'STOP':0,'SKIP':1,'Q':2,'R':3}[a[0]],a))
        return min(scores)[2]

    @lru_cache(maxsize=None)
    def psa_cost(self,s):
        if self.stop(s):
            return 0.0
        a=self.psa_action(s)
        c,branches=self.successors(s,a)
        return c+sum(pr*self.psa_cost(suc) for pr,suc in branches if pr>0)

    def bellman_psa(self,s):
        return min(self.successors(s,a)[0]+sum(pr*self.HPSA1(suc) for pr,suc in self.successors(s,a)[1] if pr>0)
                   for a in self.actions(s)) if not self.stop(s) else 0.0


def clear_all():
    for f in (PSASolver.HPSA1,PSASolver.HPSA1fast,PSASolver.psa_action,PSASolver.psa_cost,
              PSASolver.psa_fast_action,PSASolver.psa_fast_cost,
              Solver.exact,Solver.HQ,Solver.H1,Solver.H1SC,
              Solver.rollout_action,Solver.rollout_cost):
        f.cache_clear()


def numeric(x):
    return round(float(x),10)


def summarize(rows):
    out=[]
    for n in sorted(set(x['N'] for x in rows)):
        sub=[x for x in rows if x['N']==n]
        item={'N':n,'count':len(sub)}
        for mode in ('H1','PSA1','H1SC'):
            vals=[x[f'gap_{mode}'] for x in sub]
            item[f'{mode}_mean_gap_pct']=numeric(100*statistics.mean(vals))
            item[f'{mode}_p95_gap_pct']=numeric(100*sorted(vals)[math.ceil(.95*len(vals))-1])
            item[f'{mode}_max_gap_pct']=numeric(100*max(vals))
            item[f'{mode}_nonoptimal']=sum(x[f'gap_{mode}']>1e-8 for x in sub)
        item['psa_base_better_than_H1']=sum(x['base_PSA1']+1e-10<x['base_H1'] for x in sub)
        item['PSA1_mean_s']=numeric(statistics.mean(x['runtime_PSA1_s'] for x in sub))
        out.append(item)
    return out


def eval_case(par,full=True):
    solver=PSASolver(par)
    s=solver.initial
    t0=time.perf_counter()
    opt=solver.exact(s)
    h1base=solver.H1(s)
    h1=solver.rollout_cost(s,'H1')
    t1=time.perf_counter()
    psabase=solver.HPSA1fast(s)
    psa=solver.psa_fast_cost(s)
    t2=time.perf_counter()
    scbase=solver.H1SC(s) if full else None
    sc=solver.rollout_cost(s,'H1SC') if full else None
    t3=time.perf_counter()
    # Independent slow implementation checks the same continuation-policy family.
    if abs(psabase-solver.HPSA1(s))>1e-9:
        raise AssertionError("Fast and slow PSA1 base costs disagree")
    # As this is a closed base-policy family, T H_PSA <= H_PSA must hold.
    base_consistent=(solver.bellman_psa(s) <=psabase+1e-9)
    if not (opt <= psa+1e-8 and psa<=psabase+1e-8 and psabase<=h1base+1e-8 and base_consistent):
        raise AssertionError(f'Bound violation {par}: {opt=}, {psa=}, {psabase=}, {h1base=}, {base_consistent=}')
    if full and not (psabase>=scbase-1e-8 and sc>=opt-1e-8 and sc<=scbase+1e-8):
        raise AssertionError(f'SC bound violation {par}: {sc=}, {scbase=}, {psabase=}')
    record={'N':par.n(),'eps':par.eps, 'optimal':opt,
            'base_H1':h1base,'base_PSA1':psabase,'base_H1SC':scbase,
            'cost_H1':h1,'cost_PSA1':psa,'cost_H1SC':sc,
            'gap_H1':(h1/opt-1 if opt>0 else 0.),
            'gap_PSA1':(psa/opt-1 if opt>0 else 0.),
            'gap_H1SC':(sc/opt-1 if opt>0 else 0.) if full else 0.,
            'first_opt':min(solver.actions(s),key=lambda a:solver.q_exact(s,a)),
            'first_H1':solver.rollout_action(s,'H1'),
            'first_PSA1':solver.psa_fast_action(s),
            'first_H1SC':solver.rollout_action(s,'H1SC') if full else None,
            'runtime_DP_H1_s':t1-t0,'runtime_PSA1_s':t2-t1,'runtime_H1SC_s':t3-t2,
            'params':par.__dict__}
    clear_all()
    return record


def run(counts,seed=17,outdir='/mnt/data'):
    rng=random.Random(seed)
    rows=[]
    for n,cnt in counts.items():
        t0=time.perf_counter()
        for _ in range(cnt):
            rows.append(eval_case(make_random(n,rng)))
        print('N',n,'count',cnt,'time',round(time.perf_counter()-t0,2),'sec',flush=True)
    summary=summarize(rows)
    target=Params((.10,.15),(10.,10.),(.20,.10),(.80,.80),(.95,.90),.10)
    special=eval_case(target)
    # Verify pre-policy base and rollout at the especially diagnostic parameters.
    outfile=Path(outdir)
    outfile.mkdir(parents=True,exist_ok=True)
    (outfile/'ia_osr_psa1_results.json').write_text(json.dumps({'metadata':{'seed':seed,'counts':counts,'experiment':'exact vs H1 vs polynomial one-upfront-skip H_PSA1 vs H1SC'},'summary':summary,'counterexample':special,'rows':rows},indent=2),encoding='utf8')
    with (outfile/'ia_osr_psa1_summary.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(json.dumps({'summary':summary,'counterexample':{k:special[k] for k in ['optimal','cost_H1','cost_PSA1','cost_H1SC','base_PSA1','first_PSA1','first_opt']}},indent=2),flush=True)
    return summary,special


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--counts',default='1:50,2:300,3:400,4:300,5:100,6:30')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--outdir',default='/mnt/data')
    args=parser.parse_args()
    counts={int(x.split(':')[0]):int(x.split(':')[1]) for x in args.counts.split(',')}
    run(counts,args.seed,args.outdir)
