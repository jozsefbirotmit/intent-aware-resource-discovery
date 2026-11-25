"""Compare regenerated numerical results with immutable study originals."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def csv_compare(a,b,skip=()):
    x,y=pd.read_csv(a),pd.read_csv(b)
    assert x.shape==y.shape,(a,x.shape,y.shape)
    assert list(x.columns)==list(y.columns),a
    for c in x.columns:
        if c in skip:continue
        if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
            np.testing.assert_allclose(x[c],y[c],rtol=1e-9,atol=1e-9,equal_nan=True,err_msg=str(a)+': '+c)
        else:pd.testing.assert_series_equal(x[c],y[c],check_dtype=False)
    return len(x)

def jsonlines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def benchmark(a,b):
    o,n=jsonlines(a/'holdout.jsonl'),jsonlines(b/'holdout.jsonl'); idx={(x['seed'],x['n'],x['family']):x for x in o}
    assert len(n)==len({x['seed'] for x in n}) and len(n)<=len(o)
    for x in n:
        y=idx[x['seed'],x['n'],x['family']];assert x['parameters']==y['parameters']
        for k in ['opt','no_refresh']:np.testing.assert_allclose(x[k],y[k],rtol=1e-10,atol=1e-10)
        for k in ('psa','tail8','selective8','multi8'):
            for f in ('cost','bound','gap'):np.testing.assert_allclose(x[k][f],y[k][f],rtol=1e-9,atol=1e-9)
            for f in ('root_action','root_depth'):assert x[k][f]==y[k][f],(x['seed'],k,f)
    o,n2=jsonlines(a/'online.jsonl'),jsonlines(b/'online.jsonl');idx={(x['seed'],x['world'],x['algorithm']):x for x in o}
    for x in n2:
        y=idx[x['seed'],x['world'],x['algorithm']];
        for f in ('found','miss','depth'):assert x[f]==y[f],(x['seed'],f)
        for f in ('cost','rho'):np.testing.assert_allclose(x[f],y[f],rtol=1e-9,atol=1e-9)
    result=dict(models=len(n),policy_values=len(n)*4,trajectories=len(n2),complete=len(n)==90 and len(n2)==720,timing_values='hardware dependent; not equality tested')
    if (b/'scaling.jsonl').exists():
        o,n3=jsonlines(a/'scaling.jsonl'),jsonlines(b/'scaling.jsonl');idx={x['seed']:x for x in o}
        for x in n3:
            y=idx[x['seed']];assert x['parameters']==y['parameters'] and x['action']==y['action'] and x['depth']==y['depth'];np.testing.assert_allclose(x['bound'],y['bound'],rtol=1e-9,atol=1e-9)
        result['scaling_roots']=len(n3)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['benchmark','telemetry','mechanisms','csv']);p.add_argument('reference',type=Path);p.add_argument('rerun',type=Path);a=p.parse_args()
    if a.kind=='benchmark':out=benchmark(a.reference,a.rerun)
    elif a.kind=='csv':out={'rows':csv_compare(a.reference,a.rerun)}
    elif a.kind=='telemetry':out={name:csv_compare(a.reference/'results'/name/'policy_summary.csv',a.rerun/'results'/name/'policy_summary.csv') for name in ['philly','alibaba2018']}
    else:
        files=['cases.csv','calibration_drift_cases.csv','calibration_drift_summary.csv','sensitivity_cases.csv','sensitivity_summary.csv'];out={}
        for f in files:
            if (a.rerun/f).exists():out[f]=csv_compare(a.reference/f,a.rerun/f,skip=('elapsed_s','runtime_ms','time_ms'))
    print(json.dumps(out,indent=2))
