"""Reproducible audit of near-threshold SKIP commitments and deeper lookahead.

Inputs: CSV from NOMS2027_Synthetic_Experiment_Study_v1/results/cases.csv.
Algorithms: exact Bellman, PSA1 (H_PSA continuation, k=1), depth-2 and
 depth-3 Bellman operator iteration with the same feasible base continuation.
All quantities are *model-exact expected costs*, no sampled trajectories.
"""
from __future__ import annotations
import argparse,csv,functools,json,math,statistics,sys,time
from pathlib import Path

try:
    from ia_osr_experiment import Params
    from ia_osr_psa_audit import PSASolver,clear_all
except ModuleNotFoundError:
    raise SystemExit('Keep ia_osr_experiment.py and ia_osr_psa_audit.py alongside this script')

def from_row(row):
    v=json.loads(row['parameters'])
    return Params(*(tuple(v[k]) for k in ('p','q','r','alpha','beta')),float(v['eps']))

def assess(row,max_depth=3):
    p=from_row(row)
    solver=PSASolver(p)
    initial=solver.initial
    start=time.perf_counter()
    @functools.lru_cache(maxsize=None)
    def Vk(s,k):
        if solver.stop(s): return 0.
        if k==0: return solver.HPSA1fast(s)
        return min(score(s,a,k) for a in solver.actions(s))
    def score(s,a,k):
        c,br=solver.successors(s,a)
        return c+sum(pr*Vk(sn,k-1) for pr,sn in br if pr>0.)
    @functools.lru_cache(maxsize=None)
    def policy(s,k):
        if solver.stop(s):return ('STOP',-1)
        return min(solver.actions(s),key=lambda a:(round(score(s,a,k),12),{'STOP':0,'SKIP':1,'Q':2,'R':3}[a[0]],a[1]))
    @functools.lru_cache(maxsize=None)
    def Ck(s,k):
        if solver.stop(s):return 0.
        c,br=solver.successors(s,policy(s,k))
        return c+sum(pr*Ck(sn,k) for pr,sn in br if pr>0.)
    opt=float(row['optimal'])
    out={'family':row['family'],'kind':row.get('kind',''), 'seed':int(row['seed']),
         'index':int(row['index']),'n':p.n(), 'eps':p.eps,'opt':opt,
         'psa1_original':float(row['psa1']),'skip_closed':float(row['skip_closed']),
         'q0':p.q[0], 'q1':p.q[1], 'p0':p.p[0], 'p1':p.p[1],
         'pair_risk':1-(1-p.p[0])*(1-p.p[1])}
    out['risk_gap']=out['pair_risk']-p.eps
    out['initial_h0']=Vk(initial,0)
    qstar={a:solver.q_exact(initial,a) for a in solver.actions(initial)}
    exact_action=min(qstar,key=lambda a:(round(qstar[a],12),{'STOP':0,'SKIP':1,'Q':2,'R':3}[a[0]],a[1]))
    out['first_exact']=repr(exact_action)
    for k in range(1,max_depth+1):
        out[f'B{k}']=Vk(initial,k)
        out[f'C{k}']=Ck(initial,k)
        out[f'gap{k}_pct']=100*(out[f'C{k}']/opt-1) if opt>0 else 0.
        act=policy(initial,k)
        out[f'first{k}']=repr(act)
        out[f'root_regret{k}']=qstar[act]-opt
        if out[f'C{k}']+1e-7 < opt or out[f'C{k}'] > out[f'B{k}']+1e-7:
            raise AssertionError('Invalid policy bound: '+repr(out))
        if k>1 and out[f'B{k}'] > out[f'B{k-1}']+1e-7:
            raise AssertionError('Non-monotone Bellman upper bounds: '+repr(out))
    if abs(out['C1']-out['psa1_original'])>1e-7:
        raise AssertionError('PSA1 regression check failed: '+repr(out))
    out['runtime_ms']=1000*(time.perf_counter()-start)
    clear_all()
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input', default='cases_input.csv')
    parser.add_argument('--out',default='results_depth.csv')
    parser.add_argument('--max-depth',type=int,default=3)
    parser.add_argument('--iid-per-seed',type=int,default=40)
    parser.add_argument('--controls-per-seed',type=int,default=20)
    args=parser.parse_args()
    inp=list(csv.DictReader(open(args.input,newline='',encoding='utf8')))
    chosen=[r for r in inp if r['family']=='skip_boundary']
    for fam,limit in [('iid',args.iid_per_seed),('p3_screen',args.controls_per_seed),('p4_probe',args.controls_per_seed),('deferred',args.controls_per_seed)]:
        chosen.extend(r for r in inp if r['family']==fam and int(r['index'])<limit)
    print('Selected:',len(chosen),'boundary:',sum(r['family']=='skip_boundary' for r in chosen),flush=True)
    all_rows=[]; t0=time.perf_counter()
    for i,r in enumerate(chosen,1):
        all_rows.append(assess(r,args.max_depth))
        if i%200==0:
            print('Completed',i,'/',len(chosen),'elapsed',round(time.perf_counter()-t0,2),'s',flush=True)
    out=Path(args.out);out.parent.mkdir(exist_ok=True,parents=True)
    with out.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    for family in ('skip_boundary','iid','p3_screen','p4_probe','deferred'):
        for kind in (('below','above') if family=='skip_boundary' else ('',)):
            sub=[r for r in all_rows if r['family']==family and r['kind']==kind]
            if not sub:continue
            print(family,kind,'N',len(sub),
              'gaps',[round(statistics.mean(r[f'gap{k}_pct'] for r in sub),5) for k in range(1,args.max_depth+1)],
              'max',[round(max(r[f'gap{k}_pct'] for r in sub),3) for k in range(1,args.max_depth+1)],
              'opt%', [round(100*sum(r[f'gap{k}_pct']<1e-7 for r in sub)/len(sub),2) for k in range(1,args.max_depth+1)],flush=True)
    print('Written',out,flush=True)

if __name__=='__main__':main()
