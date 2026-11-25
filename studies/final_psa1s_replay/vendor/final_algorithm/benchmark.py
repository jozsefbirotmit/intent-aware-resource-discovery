"""Frozen v2 benchmark. Exact expectations and independent online trajectories."""
from selective import Selective
from iteration import CachedTail, generate, exact, tail_dp, Params, np, random, time, json, ROOT
from adaptive_study import clear_all
import sys,gc
CONFIGS=('psa','tail8','selective8','multi8')
FAMILIES=('iid','boundary','probe')
def engine(p,name):
    return Selective(p,8,True) if name=='selective8' else CachedTail(p,'multi' if name=='multi8' else 1,0 if name=='psa' else 8)
def release(e):
    for name in ('H','action','choose','cost','tail','plan','evalcost'):
        if hasattr(e,name):getattr(e,name).cache_clear()
    e.tables.clear();e.score_rows.clear();clear_all()
def no_refresh(p):
    n=p.n();v=np.empty(1<<n);surv=np.empty(1<<n);v[0]=0.;surv[0]=1.
    for m in range(1,1<<n):
        bit=m&-m;i=bit.bit_length()-1;surv[m]=surv[m^bit]*(1-p.p[i])
        if 1-surv[m]<=p.eps+1e-12:v[m]=0.;continue
        terms=[];left=m
        while left:
            b=left&-left;j=b.bit_length()-1;terms.append(p.q[j]+(1-p.p[j])*v[m^b]);left^=b
        v[m]=min(terms)
    return float(v[-1])
def trajectory(p,name,seed):
    rng=random.Random(seed);x=[rng.random()<v for v in p.p];y=[rng.random()<(p.alpha[i] if x[i] else 1-p.beta[i]) for i in range(p.n())]
    e=engine(p,name);s=e.sol.initial;floor=1;times=[];depth=[];cost=0.;found=False
    for t in range(3*p.n()+1):
        start=time.perf_counter()
        if e.sol.stop(s):a=('STOP',-1);k=0
        elif name=='selective8':a,bound,k=e.plan(s)
        else:k=e.choose(s,floor);a,bound=e.action(s,k)
        times.append(1000*(time.perf_counter()-start));depth.append(k)
        act,i=a
        if act=='STOP':break
        if act=='Q':
            cost+=p.q[i]
            if x[i]:found=True;break
            s=e.sol.set(s,i,3)
        elif act=='R':cost+=p.r[i];s=e.sol.set(s,i,1 if y[i] else 2)
        elif act=='SKIP':s=e.sol.set(s,i,s[i]+4)
        floor=max(1,k-1)
    else:raise RuntimeError('nonterminating')
    risk=0. if found else e.sol.risk(s);assert risk<=p.eps+1e-12
    ans=dict(times_ms=times,depth=depth,cost=cost,found=found,miss=bool(not found and any(x)),rho=risk,tail_states=e.tail_states,peak_table_bytes=e.peak)
    release(e);return ans
def warm():
    exact(generate(2,'iid',12));tail_dp(np.array([.3]),np.array([1.]),np.array([.1]),np.array([.9]),np.array([.9]),np.array([True]),.1,1.)
def models(count=10):
    with open(ROOT/'holdout.jsonl','w') as f, open(ROOT/'online.jsonl','w') as g:
        for n in (8,10,12):
            for j in range(count):
                for fi,fam in enumerate(FAMILIES):
                    seed=73100000+n*10000+fi*1000+j;p=generate(n,fam,seed);t=time.perf_counter();opt,states,_=exact(p);dt=(time.perf_counter()-t)*1000;nr=no_refresh(p)
                    row=dict(n=n,family=fam,seed=seed,parameters=p.__dict__,opt=opt,no_refresh=nr,exact_ms=dt,states=states)
                    order=CONFIGS[j%4:]+CONFIGS[:j%4]
                    for name in order:
                        e=engine(p,name);d=e.evaluate();assert opt<=d['cost']+1e-7 and d['cost']<=d['bound']+1e-7;d['gap']=100*(d['cost']/opt-1) if opt>0 else 0.;row[name]=d
                    f.write(json.dumps(row)+'\n');f.flush()
                    for k in range(2):
                        for name in order:
                            d=trajectory(p,name,seed*100+k);g.write(json.dumps(dict(n=n,family=fam,seed=seed,world=k,algorithm=name,**d))+'\n')
                    g.flush();print('holdout',n,j,fam,round(row['selective8']['gap'],3),flush=True);gc.collect()
def scale(reps=3):
    with open(ROOT/'scaling.jsonl','w') as f:
        for n in (20,40,60,80,100):
            for fam in ('iid','stress'):
                for j in range(reps):
                    seed=73200000+n*1000+j+(100 if fam=='stress' else 0);rng=random.Random(seed)
                    if fam=='iid':p=generate(n,'iid',seed)
                    else:
                        ps=tuple(.018+rng.uniform(0,.008) for _ in range(n));qs=tuple(rng.uniform(4,12) for _ in range(n));p=Params(ps,qs,tuple(.2*q for q in qs),(.95,)*n,(.95,)*n,.1)
                    e=engine(p,'selective8');s=e.sol.initial;t=time.perf_counter();a,b,k=e.plan(s);ms=(time.perf_counter()-t)*1000
                    assert k==1
                    f.write(json.dumps(dict(n=n,family=fam,seed=seed,parameters=p.__dict__,root_ms=ms,action=str(a),depth=k,bound=b,h0=e.h0_calls,scores=e.action_scores))+'\n');f.flush();release(e);gc.collect();print('scale',n,fam,j,round(ms,1),flush=True)
if __name__=='__main__':
    warm();models(int(sys.argv[1]) if len(sys.argv)>1 else 10);scale()
