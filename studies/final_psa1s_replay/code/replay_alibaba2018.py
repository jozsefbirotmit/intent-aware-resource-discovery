"""Final-policy replay with frozen models and the unchanged calibration gate."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from final_policy_tables import *
from replay_statistics import *

ROOT=Path(__file__).resolve().parents[1];IN=ROOT/'input/alibaba2018';OUT=ROOT/'results/alibaba2018'
BUDGETS=(0.,.01,.025,.05,.075,.10)

def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    source=json.loads((IN/'experiment_metadata.json').read_text());models=json.loads((IN/'fitted_models.json').read_text())
    old_choices=json.loads((IN/'calibration_choices_frozen_before_test.json').read_text())['choices']
    z=np.load(IN/'domain_states.npz');t=z['time_s'];assert np.all(np.diff(t)==10)
    calibr=(t>=3*86400+3600)&(t<5*86400);test=(t>=5*86400+3600)&(t<8*86400)
    profiles=[];calrows=[];choices=[];audits=[]
    # Phase A: only the frozen training model and calibration observations.
    for m in models:
        profile=m['profile'];meta={k:m[k] for k in ['profile','layout','intent','cpu_limit','mem_limit','lag_s','costs']}
        states=z[f'layout{m["layout"]}_intent{m["intent"]}'];back=m['lag_s']//10;x=states[back:];y=states[:-back]
        known=(x>=0).all(1)&(y>=0).all(1);calmask=known&calibr[back:]
        assert int(calmask.sum())==m['calibration_n']
        codes=outcome_codes(np.maximum(x,0),np.maximum(y,0));local=[];tabs={};expected={};flags_by_eps={}
        for eps in BUDGETS:
            tab,ex,flags,md=make_tables(parameter(m,eps));tabs[eps]=tab;expected[eps]=ex;flags_by_eps[eps]=flags
            for j,method in enumerate(METHODS):
                r=dict(meta,method=method,epsilon=eps,**values_summary(tab[j,codes[calmask]]));local.append(r);calrows.append(r)
            a=dict(meta,epsilon=eps,calibration_n=int(calmask.sum()),**md)
            count=np.bincount(codes[calmask],minlength=256)
            for k,v in flags.items():a['calibration_'+k+'_disagreements']=int(count@v.astype(int))
            audits.append(a)
        selection={method:select_budget(local,method) for method in METHODS}
        for new,old in [('query_only','query_only'),('psa1','psa1'),('legacy_exact','exact')]:
            assert selection[new]==old_choices[profile]['selected_epsilon'][old]
        choices.append(dict(meta,selected_epsilon=selection))
        profiles.append(dict(meta=meta,source=m,codes=codes,back=back,known=known,tabs=tabs,expected=expected,flags=flags_by_eps,selection=selection))
        if profile%9==8:print('Alibaba calibration frozen',profile+1,'of',len(models),round(time.monotonic()-start,1),'s',flush=True)
    frozen=OUT/'calibration_choices_frozen_before_test.json'
    frozen.write_text(json.dumps(dict(criterion='At least 200 no-result stops and <=5% empirical conditional miss on calibration only; largest accepted budget; fallback0',choices=choices),indent=2)+'\n')
    choice_hash=hashlib.sha256(frozen.read_bytes()).hexdigest()
    pd.DataFrame(calrows).to_csv(OUT/'calibration_budget_grid.csv',index=False)
    # Phase B: evaluate the untouched test interval after all budgets are saved.
    rows=[];calendar=[];test_audits=[]
    for pr in profiles:
        meta=pr['meta'];back=pr['back'];codes=pr['codes'];tt=t[back:]
        scopes=[('per_lag_known',pr['known']&test[back:]),('all_intents_lags_common',z[f'common_mask_layout{meta["layout"]}'][back:]&test[back:])]
        for scope,mask in scopes:
            assert mask.sum()>=500
            count=np.bincount(codes[mask],minlength=256);cc=codes[mask]
            for eps in BUDGETS:
                r=dict(meta,scope=scope,epsilon=eps,test_n=int(mask.sum()))
                for k,v in pr['flags'][eps].items():r['observed_'+k+'_disagreements']=int(count@v.astype(int))
                test_audits.append(r)
            query=pr['tabs'][0.][METHODS.index('query_only'),cc,0]
            for protocol in ('zero_risk_control','nominal_10pct','calibration_selected'):
                for j,method in enumerate(METHODS):
                    eps=0. if protocol=='zero_risk_control' else .10 if protocol=='nominal_10pct' else pr['selection'][method]
                    tab=pr['tabs'][eps];v=tab[j,cc];nr=tab[METHODS.index('no_refresh_opt'),cc,0]
                    row=dict(meta,scope=scope,protocol=protocol,method=method,epsilon=eps,**values_summary(v))
                    row.update(model_expected_cost=float(pr['expected'][eps][j]),query_only_mean_cost=float(query.mean()),
                               no_refresh_same_budget_mean_cost=float(nr.mean()),
                               saving_vs_query_percent=relative_saving(row['mean_cost'],query.mean()),
                               saving_vs_no_refresh_percent=relative_saving(row['mean_cost'],nr.mean()))
                    b=alibaba_bins(tt[mask],v,query,nr);row.update(alibaba_bootstrap(b))
                    row['risk_inference_status']='undefined_no_empty_stops' if not row['empty_stop_n'] else 'zero_misses_bootstrap_degenerate' if not row['miss_n'] else 'descriptive_1hour_calendar_block'
                    if eps==0 or method=='query_only':assert row['miss_n']==0
                    assert row['success_n']+row['empty_stop_n']==row['n'];rows.append(row)
                    for k,vv in enumerate(b):calendar.append(dict(profile=meta['profile'],scope=scope,protocol=protocol,method=method,epsilon=eps,calendar_bin_10min=k,
                        n=int(vv[0]),cost_sum=vv[1],query_cost_sum=vv[2],no_refresh_same_budget_cost_sum=vv[3],empty_stop_n=int(vv[4]),miss_n=int(vv[5])))
    result=pd.DataFrame(rows);old=pd.read_csv(IN/'policy_summary.csv');checks=[]
    for new,legacy_name in [('query_only','query_only'),('psa1','psa1'),('legacy_exact','exact')]:
        idx=['profile','scope','protocol'];a=result[result.method==new].set_index(idx);b=old[old.method==legacy_name].set_index(idx).loc[a.index]
        np.testing.assert_allclose(a.mean_cost,b.mean_cost,rtol=0,atol=1e-9)
        for k in ['n','success_n','empty_stop_n','miss_n']:np.testing.assert_array_equal(a[k],b[k])
        checks.append(dict(method=new,rows=len(a),max_cost_difference=float(np.abs(a.mean_cost-b.mean_cost).max()),counts_identical=True,calibration_choices_identical=True))
    result.to_csv(OUT/'policy_summary.csv',index=False)
    pd.DataFrame(calendar).to_csv(OUT/'calendar_policy_statistics.csv.gz',index=False,compression='gzip')
    pd.DataFrame(test_audits).to_csv(OUT/'test_tie_audit.csv',index=False)
    (OUT/'model_tie_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    (OUT/'legacy_reproduction_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    meta=dict(profiles=len(models),probability_models=source['fitted_probability_models'],budget_models=len(models)*len(BUDGETS),methods=METHODS,
      rows=len(result),parameters_refitted=False,calibration_choices_sha256=choice_hash,selective_refinements=sum(a['selective_refinements'] for a in audits),
      model_expected_S_vs_DP_max_abs_difference=max(abs(a['model_expected_costs']['psa1_s']-a['model_expected_costs']['legacy_exact']) for a in audits),
      test_interval=source['test_seconds'],calibration_interval=source['calibration_seconds'],elapsed_s=time.monotonic()-start)
    (OUT/'replay_metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':main()
