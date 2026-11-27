import sys,math,random,itertools,json
from functools import lru_cache
sys.path.insert(0,sys.argv[1])
from iteration import CachedTail,generate,exact
from selective import Selective
from ia_osr_experiment import Params
from adaptive_study import clear_all
from benchmark import no_refresh
rng=random.Random(92187)
counts=dict(models=0,states=0,tail_comparisons=0,base_bound=0,full_policy=0,max_abs_exact=0.,max_abs_no_refresh=0.,max_abs_tail=0.,max_h_fast_diff=0.)
for j in range(24):
 n=4 if j<12 else 5
 p=generate(n,('iid','boundary','probe')[j%3],12000+j)
 # Deliberately separate pure-Python implementation, independent of production transition and risk code.
 g=[p.p[i]*p.alpha[i]+(1-p.p[i])*(1-p.beta[i]) for i in range(n)]
 beliefs=[(p.p[i],p.p[i]*p.alpha[i]/g[i],p.p[i]*(1-p.alpha[i])/(1-g[i]),0.) for i in range(n)]
 def b(i,k):return beliefs[i][k-4 if k>=4 else k]
 def state(s,i,k):return s[:i]+(k,)+s[i+1:]
 def risk(s,sk=False):return 1-math.prod(1-b(i,k) for i,k in enumerate(s) if not sk or k>=4)
 def transitions(s,refresh=True,skip=False):
  for i,k in enumerate(s):
   if k>=3:continue
   yield p.q[i],[(1-b(i,k),state(s,i,3))]
   if refresh and k==0:yield p.r[i],[(g[i],state(s,i,1)),(1-g[i],state(s,i,2))]
   if skip and risk(state(s,i,k+4),True)<=p.eps+1e-12:yield 0.,[(1.,state(s,i,k+4))]
 @lru_cache(None)
 def V(s,refresh=True,skip=False):
  if risk(s)<=p.eps+1e-12:return 0.
  return min(c+sum(pr*V(t,refresh,skip) for pr,t in z if pr>0) for c,z in transitions(s,refresh,skip))
 s0=(0,)*n;vo=V(s0);vn=V(s0,False);counts['max_abs_exact']=max(counts['max_abs_exact'],abs(vo-exact(p)[0]));counts['max_abs_no_refresh']=max(counts['max_abs_no_refresh'],abs(vn-no_refresh(p)));assert abs(vo-exact(p)[0])<1e-8 and abs(vn-no_refresh(p))<1e-8
 e=Selective(p,2,True); tail=CachedTail(p,1,5)
 for s in itertools.product(range(7),repeat=n):
  if risk(s,True)>p.eps+1e-12:continue
  # exact skip redundancy and all admissible states, even if not reached by tested policy
  v=V(s);assert abs(v-V(s,True,True))<1e-8;counts['states']+=1
  t=tail.tail(s)[0];counts['max_abs_tail']=max(counts['max_abs_tail'],abs(t-v));assert abs(t-v)<1e-8;counts['tail_comparisons']+=1
  h=e.sol.HPSA1fast(s);hu=e.sol.HPSA1(s);counts['max_h_fast_diff']=max(counts['max_h_fast_diff'],abs(h-hu));assert abs(h-hu)<1e-8
  th=0. if risk(s)<=p.eps+1e-12 else min(c+sum(pr*e.sol.HPSA1fast(t) for pr,t in z if pr>0) for c,z in transitions(s,True,True));assert v<=th+1e-8 and th<=h+1e-8;counts['base_bound']+=1
  a,bnd,d=e.plan(s);co=e.evalcost(s);assert v<=co+1e-8 and co<=bnd+1e-8;counts['full_policy']+=1
 counts['models']+=1
 clear_all()
 print(j,counts['states'],flush=True)
print(json.dumps(counts,indent=2))
