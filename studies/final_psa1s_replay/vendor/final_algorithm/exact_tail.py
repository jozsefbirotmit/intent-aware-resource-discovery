"""Exploratory exact-tail hybrid, based on adaptive_study.Engine.
A small remaining active subset is solved exactly with previously skipped risk
retained, and refresh disabled for already refreshed domains. The DP stores
optimal actions in the same table if requested for execution.
"""
from adaptive_study import *

@njit(cache=True)
def tail_dp(p,q,r,alpha,beta,refreshable,eps,skipped_survival):
    n=len(p);m=1<<(2*n);prob=np.zeros((n,4));g=np.zeros(n)
    for i in range(n):
        g[i]=p[i]*alpha[i]+(1-p[i])*(1-beta[i]);prob[i,0]=p[i]
        prob[i,1]=p[i]*alpha[i]/g[i] if g[i]>0 else 0.
        prob[i,2]=p[i]*(1-alpha[i])/(1-g[i]) if g[i]<1 else 0.
    v=np.empty(m);actions=np.empty(m,dtype=np.int16)
    for s in range(m-1,-1,-1):
        x=skipped_survival
        for i in range(n):x*=1-prob[i,(s>>(2*i))&3]
        if 1-x<=eps+1e-12:
            v[s]=0.;actions[s]=-1;continue
        best=np.inf;act=-1
        for i in range(n):
            k=(s>>(2*i))&3;step=1<<(2*i)
            if k==3:continue
            a=q[i]+(1-prob[i,k])*v[s+(3-k)*step]
            if a<best:best=a;act=i
            if k==0 and refreshable[i]:
                a=r[i]+g[i]*v[s+step]+(1-g[i])*v[s+2*step]
                if a<best:best=a;act=n+i
        v[s]=best;actions[s]=act
    return v,actions

class TailEngine(Engine):
    def __init__(self,p,mode,K):
        super().__init__(p,mode);self.K=K;self.tail_calls=0;self.tail_states=0;self.tail_entries=0;self.tail=lru_cache(None)(self._tail)
    def _tail(self,s):
        sol=self.sol;idx=sol.unfinished(s);n=len(idx);self.tail_calls+=1;self.tail_states+=4**n
        vals=np.array([sol.belief(i,s[i]) for i in idx]);q,r,a,b=[np.array([getattr(sol.a,k)[i] for i in idx]) for k in ('q','r','alpha','beta')]
        v,actions=tail_dp(vals,q,r,a,b,np.array([s[i]==0 for i in idx]),sol.a.eps,1-sol.risk(s,True))
        ac=int(actions[0]);action=('STOP',-1) if ac<0 else ('Q',idx[ac]) if ac<n else ('R',idx[ac-n]);return float(v[0]),action
    def _H(self,s,k):
        if self.sol.stop(s):return 0.
        if len(self.sol.unfinished(s))<=self.K:return self.tail(s)[0]
        return super()._H(s,k)
    def _choose(self,s,floor):
        if self.sol.stop(s):return 0
        if len(self.sol.unfinished(s))<=self.K:return max(1,floor)
        return super()._choose(s,floor)
    def _action(self,s,k):
        if self.sol.stop(s):return ('STOP',-1),0.
        if len(self.sol.unfinished(s))<=self.K:
            val,ac=self.tail(s);return ac,val
        return super()._action(s,k)
    def _cost(self,s,floor):
        if self.sol.stop(s):return (0.,0.,0.,0.,0.)
        if len(self.sol.unfinished(s))<=self.K:
            self.tail_entries+=1;return (self.tail(s)[0],0.,0.,0.,0.)
        return super()._cost(s,floor)
    def evaluate(self):
        d=super().evaluate();d.update(tail_calls=self.tail_calls,tail_states=self.tail_states,tail_entries=self.tail_entries);self.tail.cache_clear();return d

def check():
    p=make_random(5,random.Random(930));s=PSASolver(p);e=TailEngine(p,1,6);tested=0
    @lru_cache(None)
    def v(st):
        if s.stop(st):return 0.
        return min(s.successors(st,a)[0]+sum(pr*v(ns) for pr,ns in s.successors(st,a)[1] if pr>0) for a in s.actions(st))
    import itertools
    for st in itertools.product(range(7),repeat=5):
        if s.risk(st,True)>p.eps+1e-12 or s.stop(st):continue
        assert abs(v(st)-e.tail(st)[0])<1e-7,(st,v(st),e.tail(st));tested+=1
    e.tail.cache_clear();clear_all();print('tail check states',tested,flush=True)

def run():
    check();rows=[]
    # Original large completed N=8/9 cases plus a small NEW independent root sample N=10-12.
    cases=[r for r in csv.DictReader(open(ROOT/'large.csv')) if int(r['n'])>=8]
    cases += [r for r in csv.DictReader(open(ROOT/'large_root.csv')) if int(r['n'])>=10 and int(r['seed'])%100 in (0,1)]
    for i,row in enumerate(cases):
        p=from_row(row);n=p.n();opt=float(row['opt']);d=dict(n=n,seed=row['seed'],family=row['family'],parameters=row['parameters'],opt=opt)
        # Full expected costs; stop recursion at exact tails (analytic integration).
        for mode,K in [(1,6),('risk',6)]:
            e=TailEngine(p,mode,K);v=e.evaluate();assert opt<=v['cost']+1e-7
            v['gap_pct']=100*(v['cost']/opt-1) if opt>0 else 0.
            d.update({f'{mode}_{k}':x for k,x in v.items()})
        if '1_cost' in row:d.update(baseline_psa_cost=row['1_cost'],baseline_risk_cost=row['risk_cost'],baseline_psa_gap_pct=row['1_gap_pct'],baseline_risk_gap_pct=row['risk_gap_pct'])
        else:
            # no exhaustive standard policy in this subset; compare root bound from previous run only
            d.update(baseline_psa_cost='',baseline_risk_cost='',baseline_psa_gap_pct='',baseline_risk_gap_pct='')
        rows.append(d);write('tail.csv',rows);print(n,row['family'],round(d['1_gap_pct'],3),round(d['risk_gap_pct'],3),round(d['1_total_ms'],1),round(d['risk_total_ms'],1),flush=True)
if __name__=='__main__':run()
