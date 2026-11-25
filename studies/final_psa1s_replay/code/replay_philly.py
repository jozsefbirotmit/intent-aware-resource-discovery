"""Frozen-parameter replay of all 224 original primary Philly configurations."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from final_policy_tables import *
from replay_statistics import relative_saving

ROOT=Path(__file__).resolve().parents[1];IN=ROOT/'input/philly';OUT=ROOT/'results/philly'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=json.loads((IN/'experiment_metadata.json').read_text());models=json.loads((IN/'fitted_models.json').read_text())
    states=pd.read_csv(IN/'domain_states.csv.gz',parse_dates=['available_at_utc'])
    test_start=pd.Timestamp(source['test_start_utc'])
    groups={}
    for key,f in states.groupby(['replicate','gpus_requested','mode'],sort=False):
        f=f.sort_values('available_at_utc');idx=pd.DatetimeIndex(f.available_at_utc)
        assert np.all(np.diff(idx.asi8)==60*10**9)
        x=f[[f'X{i}' for i in range(4)]].fillna(0).to_numpy(bool)
        test=f.test.to_numpy(bool);codes={}
        for lag in source['lags_minutes']:
            y=np.zeros_like(x);y[lag:]=x[:-lag];codes[lag]=outcome_codes(x,y)
        groups[key]=dict(index=idx,x=x,test=test,codes=codes)
    byscenario={}
    for m in models:byscenario.setdefault(m['scenario'],[]).append(m)
    rows=[];daily=[];audits=[];expected_checks=[];t0=time.monotonic()
    for scenario,fits in byscenario.items():
        fits=sorted(fits,key=lambda m:m['period']);m0=fits[0]
        meta={k:m0[k] for k in ['scenario','replicate','gpus_requested','mode','lag_min','calibration','costs','refresh_ratio','epsilon']}
        g=groups[m0['replicate'],m0['gpus_requested'],m0['mode']];idx=g['index'];test=g['test'];codes=g['codes'][m0['lag_min']]
        test_indices=np.flatnonzero(test);val=np.full((len(METHODS),len(test_indices),8),np.nan);expected_sum=np.zeros(len(METHODS))
        # Calendar periods are identical to the original weekly rolling protocol.
        period_ids=np.maximum(0,((idx.asi8-test_start.value)//(7*86400*10**9))).astype(int)
        for m in fits:
            te=test if m['calibration']=='static' else test&(period_ids==m['period'])
            count=np.bincount(codes[te],minlength=256)
            assert int(te.sum())==m['test_n']
            np.testing.assert_array_equal(count,m['test_joint_counts'])
            tab,expected,flags,md=make_tables(parameter(m))
            assert abs(expected[METHODS.index('psa1')]-m['expected_costs'][4])<1e-9
            assert abs(expected[METHODS.index('legacy_exact')]-m['expected_costs'][5])<1e-9
            loc=np.searchsorted(test_indices,np.flatnonzero(te));val[:,loc,:]=tab[:,codes[te],:]
            expected_sum+=int(te.sum())*expected
            audit=dict(meta,period=m['period'],test_n=int(te.sum()),**md)
            for k,v in flags.items():audit['observed_'+k+'_disagreements']=int(count@v.astype(int))
            audits.append(audit)
        assert not np.isnan(val).any()
        query=val[METHODS.index('query_only'),:,0];nr=val[METHODS.index('no_refresh_opt'),:,0]
        days=pd.date_range(test_start.floor('D'),idx[-1].floor('D'),freq='D')
        for j,method in enumerate(METHODS):
            v=val[j];f=pd.DataFrame(v,columns=FIELDS);f['day']=idx[test].floor('D');f['n']=1
            f['query_cost']=query;f['no_refresh_cost']=nr;f['saving_query']=query-v[:,0];f['saving_no_refresh']=nr-v[:,0]
            bd=f.groupby('day').sum().reindex(days,fill_value=0).reset_index(names='day')
            lo,hi,defined=legacy.weekly_block_bootstrap(bd,'miss','empty_stop',1000,17)
            qlo,qhi,_=legacy.weekly_block_bootstrap(bd,'saving_query','query_cost',1000,17)
            nlo,nhi,_=legacy.weekly_block_bootstrap(bd,'saving_no_refresh','no_refresh_cost',1000,17)
            row=dict(meta,method=method,**values_summary(v));row.update(
                model_expected_cost=float(expected_sum[j]/len(test_indices)),
                query_only_mean_cost=float(query.mean()),no_refresh_same_budget_mean_cost=float(nr.mean()),
                saving_vs_query_percent=relative_saving(row['mean_cost'],query.mean()),
                saving_vs_no_refresh_percent=relative_saving(row['mean_cost'],nr.mean()),
                risk_block95_low=lo,risk_block95_high=hi,risk_bootstrap_defined=defined,
                saving_query_block95_low=100*qlo,saving_query_block95_high=100*qhi,
                saving_no_refresh_block95_low=100*nlo,saving_no_refresh_block95_high=100*nhi)
            row['risk_inference_status']='undefined_no_empty_stops' if row['empty_stop_n']==0 else 'zero_misses_bootstrap_degenerate' if row['miss_n']==0 else 'descriptive_7day_calendar_block'
            if m0['epsilon']==0 or method=='query_only':assert row['miss_n']==0
            rows.append(row)
            for d in bd.to_dict('records'):daily.append(dict(scenario=scenario,method=method,**{k:str(v) if k=='day' else v for k,v in d.items()}))
        if scenario%16==0:print('Philly completed',scenario,'of',len(byscenario),round(time.monotonic()-t0,1),'s',flush=True)
    result=pd.DataFrame(rows);original=pd.read_csv(IN/'policy_summary.csv');checks=[]
    for new,old in [('query_only','query_only'),('psa1','psa1'),('legacy_exact','exact')]:
        a=result[result.method==new].set_index('scenario');b=original[original.method==old].set_index('scenario').loc[a.index]
        np.testing.assert_allclose(a.mean_cost,b.mean_cost,rtol=0,atol=1e-9)
        for k in ['n','success_n','empty_stop_n','miss_n']:np.testing.assert_array_equal(a[k],b[k])
        checks.append(dict(method=new,scenarios=len(a),max_cost_difference=float(np.abs(a.mean_cost-b.mean_cost).max()),counts_identical=True))
    result.to_csv(OUT/'policy_summary.csv',index=False);pd.DataFrame(daily).to_csv(OUT/'daily_policy_statistics.csv.gz',index=False,compression='gzip')
    (OUT/'model_tie_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(list,tuple,dict))} for r in audits]).to_csv(OUT/'model_tie_audit.csv',index=False)
    (OUT/'legacy_reproduction_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    metadata=dict(source='primary full-known telemetry; no job-log availability selection',scenarios=len(byscenario),saved_period_models=len(models),
        methods=METHODS,rows=len(result),parameters_refitted=False,test_joint_counts_reproduced=True,elapsed_s=time.monotonic()-t0,
        selective_refinements=sum(r['selective_refinements'] for r in audits),
        model_expected_S_vs_DP_max_abs_difference=max(abs(r['model_expected_costs']['psa1_s']-r['model_expected_costs']['legacy_exact']) for r in audits))
    (OUT/'replay_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':main()
