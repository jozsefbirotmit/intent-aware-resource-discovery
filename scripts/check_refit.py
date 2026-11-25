import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from compare import csv_compare
name=sys.argv[1];orig,new=map(Path,sys.argv[2:]);a,b=[json.loads((p/'fitted_models.json').read_text()) for p in [orig,new]];assert len(a)==len(b)
for x,y in zip(a,b):
 for k in ['p','alpha','beta','q','r']:np.testing.assert_allclose(x[k],y[k],rtol=0,atol=1e-12)
 for k in ['train_n','test_n','calibration_n','test_joint_counts']:
  if k in x:assert x[k]==y[k],k
if name=='alibaba2018':
 with np.load(orig/'domain_states.npz') as x,np.load(new/'domain_states.npz') as y:
  assert x.files==y.files
  for k in x.files:np.testing.assert_array_equal(x[k],y[k])
 csv_compare(orig/'brier_scores.csv',new/'brier_scores.csv')
else:pd.testing.assert_frame_equal(pd.read_csv(orig/'domain_states.csv.gz'),pd.read_csv(new/'domain_states.csv.gz'))
print(json.dumps(dict(dataset=name,fitted_models=len(a),domain_states_equal=True,policy_rows=csv_compare(orig/'policy_summary.csv',new/'policy_summary.csv')),indent=2))
