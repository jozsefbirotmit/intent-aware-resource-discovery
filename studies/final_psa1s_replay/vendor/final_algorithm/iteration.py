"""Exact-tail table reuse and multi-mechanism lookahead on fresh synthetic cases."""
from exact_tail import TailEngine,tail_dp
from adaptive_study import exact,Params,make_random
from mechanism_study import gen_skip_boundary
from collections import OrderedDict
import numpy as np, math, random, time,json,sys
from pathlib import Path
ROOT=Path(__file__).parent
class CachedTail(TailEngine):
    def __init__(self,p,mode,K,reuse=True):
        super().__init__(p,mode,K);self.reuse=reuse;self.tables=OrderedDict();self.bytes=0;self.peak=0;self.hits=0;self.builds=0;self.limit=32*1024*1024;self.score_rows={}
    def _tail(self,s):
        if not self.reuse:return super()._tail(s)
        self.tail_calls+=1;so=self.sol;active=tuple(so.unfinished(s));skip=tuple((i,k) for i,k in enumerate(s) if k>=4);key=(skip,active)
        hit=key if key in self.tables else next((k for k in reversed(self.tables) if k[0]==skip and all(i in k[1] for i in active)),None)
        if hit is not None:
            self.hits+=1;self.tables.move_to_end(hit);idx=hit[1];v,ac=self.tables[hit]
        else:
            idx=active;args=[np.array([getattr(so.a,k)[i] for i in idx]) for k in ('p','q','r','alpha','beta')]
            v,ac=tail_dp(*args,np.ones(len(idx),dtype=np.bool_),so.a.eps,1-so.risk(s,True));self.builds+=1;self.tail_states+=len(v)
            size=v.nbytes+ac.nbytes
            while self.tables and self.bytes+size>self.limit:
                _,(old,oa)=self.tables.popitem(last=False);self.bytes-=old.nbytes+oa.nbytes
            if size<=self.limit:self.tables[key]=(v,ac);self.bytes+=size;self.peak=max(self.peak,self.bytes)
        state=sum(s[i]<<(2*j) for j,i in enumerate(idx));a=int(ac[state]);n=len(idx)
        return float(v[state]),(('STOP',-1) if a<0 else ('Q',idx[a]) if a<n else ('R',idx[a-n]))
    def _action(self,s,k):
        if self.sol.stop(s) or len(self.sol.unfinished(s))<=self.K:return super()._action(s,k)
        vals=[]
        for a in self.sol.actions(s):
            self.action_scores+=1;c,br=self.sol.successors(s,a);v=c+sum(pr*self.H(ns,k-1) for pr,ns in br if pr>0)
            vals.append((round(v,12),{'STOP':0,'SKIP':1,'Q':2,'R':3}[a[0]],a,v))
        if k==1 and self.mode=='multi':self.score_rows[s]=vals
        _,_,a,v=min(vals);return a,v
    def order_gate(self,s):
        so=self.sol;idx=so.unfinished(s);p={i:so.belief(i,s[i]) for i in idx}
        def F(p):
            z=0.;surv=1.
            for i in sorted(idx,key=lambda i:so.a.q[i]/p[i] if p[i]>0 else math.inf):z+=surv*so.a.q[i];surv*=1-p[i]
            return z
        base=None
        for i in idx:
            if s[i]!=0:continue
            q=so.a.q[i];lo=q/so.pp[i] if so.pp[i]>0 else math.inf;hi=q/so.pm[i] if so.pm[i]>0 else math.inf
            if not any(lo<so.a.q[j]/p[j]<hi for j in idx if j!=i and p[j]>0):continue
            if base is None:base=F(p)
            pp=p.copy();pm=p.copy();pp[i]=so.pp[i];pm[i]=so.pm[i]
            if base-so.g[i]*F(pp)-so.h[i]*F(pm)-so.a.r[i]>1e-10:return True
        return False
    def _choose(self,s,floor):
        if self.mode!='multi':return super()._choose(s,floor)
        if self.sol.stop(s):return 0
        m=len(self.sol.unfinished(s))
        if m<=self.K or m<3 or m>16:return max(1,floor)
        if self.risk_gate(s):return max(floor,3 if m<=8 else 2)
        a,h=self.action(s,1);refresh=[x[3] for x in self.score_rows[s] if x[2][0]=='R']
        if (refresh and min(refresh)<=h+0.03*max(h,1e-12)) or self.order_gate(s):
            b,h2=self.action(s,2)
            return max(floor,3 if m<=8 and (a!=b or h-h2>0.005*max(h,1e-12)) else 2)
        return max(1,floor)
    def evaluate(self):
        d=super().evaluate();d.update(builds=self.builds,hits=self.hits,peak_table_bytes=self.peak);self.tables.clear();self.score_rows.clear();return d

def generate(n,family,seed):
    rng=random.Random(seed)
    if family=='iid':return make_random(n,rng)
    if family=='boundary':
        b=gen_skip_boundary(rng)[seed%2];m=n-4;ps=b.p+tuple(rng.uniform(.04,.32) for _ in range(m));qs=b.q+tuple(rng.uniform(2,12) for _ in range(m))
        return Params(ps,qs,b.r+tuple(q*rng.uniform(.05,.30) for q in qs[4:]),b.alpha+tuple(rng.uniform(.8,.97) for _ in range(m)),b.beta+tuple(rng.uniform(.82,.96) for _ in range(m)),b.eps)
    p=tuple(rng.uniform(.18,.7) for _ in range(n));q=tuple(rng.uniform(2,14) for _ in range(n))
    return Params(p,q,tuple(x*rng.uniform(.015,.15) for x in q),tuple(rng.uniform(.78,.95) for _ in p),tuple(rng.uniform(.8,.97) for _ in p),.005)

def run(tag):
    exact(make_random(2,random.Random(1)));tail_dp(np.array([.3]),np.array([1.]),np.array([.1]),np.array([.9]),np.array([.9]),np.array([True]),.1,1.)
    configs=[('psa',1,0,True),('k4',1,4,True),('k6',1,6,True),('k8',1,8,True),('risk6','risk',6,True),('multi6','multi',6,True)]
    sizes=[(6,8),(8,8),(10,6)] if tag=='main' else [(12,3)]
    if tag=='check':sizes=[(8,2),(10,1)];configs=[('old6',1,6,False),('k6',1,6,True),('oldrisk','risk',6,False),('risk6','risk',6,True)]
    with open(ROOT/(tag+'.jsonl'),'w') as f:
        for n,count in sizes:
            for j in range(count):
                for fi,fam in enumerate(['iid','boundary','probe']):
                    seed=2026112700+n*1000+fi*100+j+(100000 if tag=='check' else 0);p=generate(n,fam,seed);t=time.perf_counter();opt,states,_=exact(p);ms=(time.perf_counter()-t)*1000
                    row=dict(n=n,family=fam,seed=seed,opt=opt,exact_ms=ms,parameters=p.__dict__)
                    for name,mode,K,reuse in configs:
                        e=CachedTail(p,mode,K,reuse);d=e.evaluate();assert opt<=d['cost']+1e-7 and d['cost']<=d['bound']+1e-7;d['gap']=100*(d['cost']/opt-1) if opt else 0.;row[name]=d
                    if tag=='check':
                        for a,b in [('old6','k6'),('oldrisk','risk6')]:assert abs(row[a]['cost']-row[b]['cost'])<1e-7 and abs(row[a]['bound']-row[b]['bound'])<1e-7
                    f.write(json.dumps(row)+'\n');f.flush();print(tag,n,j,fam,round(row['k6']['gap'],3),round(row.get('multi6',row['k6'])['gap'],3),flush=True)
if __name__=='__main__':run(sys.argv[1])
