"""Adaptive PSA rollout with inherited horizon; exact dense no-SKIP oracle.
All policy costs integrated over the model, no sampled trajectories.
Run: python adaptive_study.py {small,large,check}.
"""
import csv,json,math,random,statistics,time,sys
from functools import lru_cache
from pathlib import Path
import numpy as np
from numba import njit
from ia_osr_experiment import Params,make_random
from ia_osr_psa_audit import PSASolver,clear_all
from skip_depth_study import from_row
ROOT=Path(__file__).parent

@njit(cache=True)
def oracle(p,q,r,alpha,beta,eps):
    n=len(p); m=1<<(2*n)
    prob=np.zeros((n,4)); g=np.empty(n)
    for i in range(n):
        g[i]=p[i]*alpha[i]+(1-p[i])*(1-beta[i]); prob[i,0]=p[i]
        prob[i,1]=p[i]*alpha[i]/g[i] if g[i]>0 else 0.
        prob[i,2]=p[i]*(1-alpha[i])/(1-g[i]) if g[i]<1 else 0.
    v=np.empty(m,dtype=np.float64); stopcount=0
    for s in range(m-1,-1,-1):
        x=1.
        for i in range(n): x*=1-prob[i,(s>>(2*i))&3]
        if 1-x<=eps+1e-12:
            v[s]=0.;stopcount+=1;continue
        best=np.inf
        for i in range(n):
            k=(s>>(2*i))&3; step=1<<(2*i)
            if k==3:continue
            a=q[i]+(1-prob[i,k])*v[s+(3-k)*step]
            if a<best:best=a
            if k==0:
                a=r[i]+g[i]*v[s+step]+(1-g[i])*v[s+2*step]
                if a<best:best=a
        v[s]=best
    return v[0],m,stopcount

def exact(p):return oracle(*(np.asarray(getattr(p,k),dtype=float) for k in ('p','q','r','alpha','beta')),p.eps)

class Engine:
    def __init__(self,p,mode):
        self.sol=PSASolver(p);self.mode=mode;self.depth_hist=[0,0,0,0];self.rootdepth=0
        self.h0_calls=0; self.action_scores=0
        self.H=lru_cache(None)(self._H); self.action=lru_cache(None)(self._action); self.choose=lru_cache(None)(self._choose); self.cost=lru_cache(None)(self._cost)
    def _H(self,s,k):
        if self.sol.stop(s):return 0.
        if k==0:
            self.h0_calls+=1
            return self.sol.HPSA1fast(s)
        return self.action(s,k)[1]
    def _action(self,s,k):
        if self.sol.stop(s):return ('STOP',-1),0.
        vals=[]
        for a in self.sol.actions(s):
            self.action_scores+=1
            c,b=self.sol.successors(s,a)
            v=c+sum(pr*self.H(ns,k-1) for pr,ns in b if pr>0)
            vals.append((round(v,12),{'STOP':0,'SKIP':1,'Q':2,'R':3}[a[0]],a,v))
        _,_,a,v=min(vals);return a,v
    def risk_gate(self,s):
        # Only currently admissible singles whose pair may compete for the same budget.
        so=self.sol; active=so.unfinished(s); m=len(active)
        if m<3 or sum(s[i]==0 for i in active)<2:return False
        if m>16:return False
        b=-math.log1p(-so.a.eps); prior=-math.log1p(-min(so.risk(s,True),1-1e-15)); residual=b-prior
        if residual<=1e-12:return False
        w=[-math.log1p(-so.belief(i,s[i])) for i in active if 0<so.belief(i,s[i])<1 and so.belief(i,s[i]) <= -math.expm1(-residual)+1e-12]
        return any(abs(x+y-residual)<=0.12*residual for i,x in enumerate(w) for y in w[i+1:])
    def _choose(self,s,floor):
        if self.sol.stop(s):return 0
        if isinstance(self.mode,int):return self.mode
        if not self.risk_gate(s):return max(1,floor)
        m=len(self.sol.unfinished(s)); cap=3 if m<=8 else 2
        if self.mode=='risk':return max(floor,cap)
        # Second-stage trigger: further search when deepening changes the action
        # or tightens the bound by >=0.5%; otherwise retain depth two.
        a,h1=self.action(s,1);b,h2=self.action(s,2)
        if cap==3 and (a!=b or h1-h2>0.005*max(h1,1e-12)):return max(floor,3)
        return max(floor,2)
    def _cost(self,s,floor):
        if self.sol.stop(s):return (0.,0.,0.,0.,0.)
        k=self.choose(s,floor);self.depth_hist[k]+=1
        a,bound=self.action(s,k);c,br=self.sol.successors(s,a)
        accum=[c,1.,float(k==1),float(k==2),float(k==3)]
        for pr,ns in br:
            if pr<=0:continue
            res=self.cost(ns,max(1,k-1))
            for j in range(5):accum[j]+=pr*res[j]
        if accum[0]>bound+1e-7:raise AssertionError((s,k,accum[0],bound))
        return tuple(accum)
    def evaluate(self):
        s=self.sol.initial;t=time.perf_counter();k=self.choose(s,1);self.rootdepth=k
        a,b=self.action(s,k) if k else (('STOP',-1),0.)
        rt=time.perf_counter()-t;rootwork=self.h0_calls
        co,steps,d1,d2,d3=self.cost(s,1);total=time.perf_counter()-t
        self.H.cache_clear();self.action.cache_clear();self.choose.cache_clear();self.cost.cache_clear();clear_all()
        return dict(cost=co,root_depth=k,root_action=str(a),bound=b,root_ms=1000*rt,total_ms=1000*total,root_h0=rootwork,h0=self.h0_calls,scores=self.action_scores,steps=steps,depth1=d1,depth2=d2,depth3=d3)

def assess(p,meta,modes,old=None):
    t=time.perf_counter();opt,states,stops=exact(p); et=time.perf_counter()-t
    if old is not None:assert abs(opt-old)<1e-7,(opt,old)
    out=dict(meta,parameters=json.dumps(dict(p=p.p,q=p.q,r=p.r,alpha=p.alpha,beta=p.beta,eps=p.eps)),n=p.n(),eps=p.eps,opt=opt,exact_ms=et*1000,states=states,terminal_states=stops,dense_bytes=8*states)
    for mode in modes:
        e=Engine(p,mode); row=e.evaluate()
        assert opt<=row['cost']+1e-7 and row['cost']<=row['bound']+1e-7
        row['gap_pct']=100*(row['cost']/opt-1) if opt>0 else 0.
        out.update({f'{mode}_{k}':v for k,v in row.items()})
    return out

def write(out,rows):
    with open(ROOT/out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def small():
    inp=list(csv.DictReader(open(ROOT/'cases_input.csv')));keys={(r['family'],r['kind'],r['seed'],r['index']):r for r in csv.DictReader(open(ROOT/'results_depth.csv'))}
    rows=list(csv.DictReader(open(ROOT/'small.csv'))) if (ROOT/'small.csv').exists() else []; done={tuple(str(r[k]) for k in ('family','kind','seed','index')) for r in rows};t=time.perf_counter()
    for r in inp:
        key=tuple(r[k] for k in ('family','kind','seed','index'))
        if key not in keys or key in done:continue
        metadata={k:r[k] for k in ('family','kind','seed','index')};x=assess(from_row(r),metadata,[1,2,3,'risk','progress'],float(r['optimal']))
        for k in (1,2,3):assert abs(x[f'{k}_cost']-float(keys[key][f'C{k}']))<1e-7
        rows.append(x)
        if len(rows)%200==0:print(len(rows),time.perf_counter()-t,flush=True);write('small.csv',rows)
    write('small.csv',rows)

def larger():
    # Fresh seeds. Two scenario families: original IID and low-feasibility/high-cost SKIP stress.
    from mechanism_study import gen_skip_boundary
    rows=[]
    for n,count in [(6,12),(8,12),(9,11)]:
        for j in range(count):
            seed=2026111700+n*100+j;rng=random.Random(seed);family='iid' if j%2==0 else 'extended_boundary'
            if family=='iid':p=make_random(n,rng)
            else:
                base=gen_skip_boundary(rng)[j%4==3] # below/above, balanced within this family
                extra=n-4
                ps=base.p+tuple(rng.uniform(.04,.32) for _ in range(extra));qs=base.q+tuple(rng.uniform(2,12) for _ in range(extra))
                rs=base.r+tuple(q*rng.uniform(.05,.30) for q in qs[4:]);aa=base.alpha+tuple(rng.uniform(.8,.97) for _ in range(extra));bb=base.beta+tuple(rng.uniform(.82,.96) for _ in range(extra))
                p=Params(ps,qs,rs,aa,bb,base.eps)
            # Fixed depth three on n<=10; larger sizes still compare fixed one and adaptive.
            modes=[1,2,3,'risk','progress'] if n<=10 else [1,2,'risk','progress']
            x=assess(p,dict(family=family,kind='',seed=seed,index=j),modes)
            rows.append(x); print(n,j,round(x['exact_ms'],1),round(x['1_gap_pct'],3),round(x['progress_gap_pct'],3),round(x['progress_total_ms'],1),flush=True)
            # Field list follows first row, missing fixed-3 for N>10.
            write('large.csv',rows)

def check():
    p=make_random(4,random.Random(10)); sol=PSASolver(p);print(exact(p),sol.exact(sol.initial));clear_all()
    print(assess(p,{},[1,2,3,'risk','progress']))
if __name__=='__main__':
    {'small':small,'large':larger,'check':check}[sys.argv[1]]()
