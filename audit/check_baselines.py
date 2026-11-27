import sys,statistics,collections,csv
sys.path.insert(0,sys.argv[1])
from iteration import generate
from benchmark import no_refresh
from ia_osr_psa_audit import PSASolver,clear_all
out=[]
for n in [10,12]:
 for f,fam in enumerate(('iid','boundary','probe')):
  for j in range(10):
   seed=73100000+10000*n+1000*f+j;p=generate(n,fam,seed);s=PSASolver(p);nr=no_refresh(p);row=dict(n=n,family=fam,seed=seed,no_refresh=nr)
   for mode in ('HQ','H1'):
    cost=s.rollout_cost(s.initial,mode);row[mode+'_cost']=cost;row[mode+'_saving']=100*(1-cost/nr)
   out.append(row);clear_all()
with open(sys.argv[2] if len(sys.argv)>2 else 'baseline_60.csv','w') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
for fam in ['all','iid','boundary','probe']:
 t=[r for r in out if fam=='all' or fam==r['family']];print(fam,len(t),[(m,statistics.mean(r[m+'_saving'] for r in t),sum(r[m+'_saving']<0 for r in t)) for m in ['HQ','H1']])
