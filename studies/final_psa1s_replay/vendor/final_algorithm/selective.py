"""Selective action-level depth refinement. Each selected action receives one or two
Bellman steps; all others retain their one-step scores. Exact tail is unchanged.
Policy expectation and certification are evaluated separately from global oracle.
"""
from iteration import *
from functools import lru_cache
from adaptive_study import clear_all
class Selective(CachedTail):
    def __init__(self,p,K=8,gated=True):
        super().__init__(p,'multi',K);self.gated=gated;self.plan=lru_cache(None)(self._plan);self.evalcost=lru_cache(None)(self._evalcost);self.refined=0
    def _plan(self,s):
        so=self.sol
        if so.stop(s):return ('STOP',-1),0.,0
        if len(so.unfinished(s))<=self.K:
            v,a=self.tail(s);return a,v,0
        a,h=self.action(s,1);rows=self.score_rows[s]
        refresh=[x[3] for x in rows if x[2][0]=='R']
        # For large active sets, bound the additional search by a fixed cap on m.
        gate=len(so.unfinished(s))<=16 and (not self.gated or self.risk_gate(s) or (refresh and min(refresh)<=h+0.03*max(h,1e-12)) or self.order_gate(s))
        if not gate:return a,h,1
        best=(round(h,12),{'STOP':0,'SKIP':1,'Q':2,'R':3}[a[0]],a,h,1)
        for acttype in ('Q','R','SKIP'):
            candidates=[r for r in rows if r[2][0]==acttype]
            if not candidates:continue
            _,rank,b,old=min(candidates);c,br=so.successors(s,b);val=c+sum(pr*self.H(ns,1) for pr,ns in br if pr>0);self.action_scores+=1;self.refined+=1
            assert val<=old+1e-7
            best=min(best,(round(val,12),rank,b,val,2))
        return best[2],best[3],best[4]
    def _evalcost(self,s):
        if self.sol.stop(s):return 0.
        if len(self.sol.unfinished(s))<=self.K:
            self.tail_entries+=1;return self.tail(s)[0]
        a,b,d=self.plan(s);self.depth_hist[d]+=1;c,br=self.sol.successors(s,a);v=c+sum(pr*self.evalcost(ns) for pr,ns in br if pr>0);assert v<=b+1e-7;return v
    def evaluate(self):
        s=self.sol.initial;t=time.perf_counter();a,b,k=self.plan(s);root_ms=1000*(time.perf_counter()-t);co=self.evalcost(s);total=1000*(time.perf_counter()-t)
        d=dict(cost=co,bound=b,root_action=str(a),root_depth=k,root_ms=root_ms,total_ms=total,h0=self.h0_calls,scores=self.action_scores,tail_states=self.tail_states,tail_calls=self.tail_calls,tail_entries=self.tail_entries,builds=self.builds,hits=self.hits,peak_table_bytes=self.peak,refined=self.refined,depth_hist=self.depth_hist)
        for f in (self.plan,self.evalcost,self.H,self.action,self.choose,self.cost,self.tail):f.cache_clear()
        self.tables.clear();self.score_rows.clear();clear_all();return d

def run(tag):
    exact(make_random(2,random.Random(1)));tail_dp(np.array([.3]),np.array([1.]),np.array([.1]),np.array([.9]),np.array([.9]),np.array([True]),.1,1.)
    if tag=='selective_old':rows=[json.loads(l) for f in ('main.jsonl','large.jsonl') for l in open(ROOT/f) if json.loads(l)['n']>=10]
    else:
        rows=[]
        for n in (10,12):
            for j in range(4):
                for fi,fam in enumerate(('iid','boundary','probe')):
                    seed=2026120100+1000*n+100*fi+j;p=generate(n,fam,seed);t=time.perf_counter();v,_,_=exact(p);rows.append(dict(n=n,family=fam,seed=seed,opt=v,exact_ms=1000*(time.perf_counter()-t),parameters=p.__dict__))
    with open(ROOT/(tag+'.jsonl'),'w') as f:
        for row in rows:
            p=Params(**{k:tuple(v) if isinstance(v,list) else v for k,v in row['parameters'].items()});out={k:row[k] for k in ('n','family','seed','opt','parameters')}
            configs=[('sel8',Selective(p,8,True)),('allsel8',Selective(p,8,False))] if tag=='selective_old' else [('psa',CachedTail(p,1,0)),('k8',CachedTail(p,1,8)),('sel8',Selective(p,8,True)),('multi8',CachedTail(p,'multi',8))]
            for name,e in configs:
                d=e.evaluate();assert row['opt']<=d['cost']+1e-7 and d['cost']<=d['bound']+1e-7; d['gap']=100*(d['cost']/row['opt']-1) if row['opt'] else 0.;out[name]=d
            f.write(json.dumps(out)+'\n');f.flush();print(row['n'],row['family'],round(out['sel8']['gap'],3),round(out['sel8']['total_ms'],2),flush=True)
if __name__=='__main__':run(sys.argv[1])
