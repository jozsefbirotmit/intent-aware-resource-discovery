"""Independent artifact consistency and frozen-calibration checks."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from alibaba_replay import INTENTS,LAGS,METHODS,select_budget
from decision_evaluation import calibrate

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/alibaba2018'

def main():
    def reject(x):raise ValueError('Nonstandard JSON numeric constant '+x)
    paths=sorted(ROOT.rglob('*.json'))
    for p in paths:json.loads(p.read_text(),parse_constant=reject)
    models=json.loads((D/'fitted_models.json').read_text())
    exp=json.loads((D/'experiment_metadata.json').read_text())
    frozen=json.loads((D/'calibration_choices_frozen_before_test.json').read_text())
    assert hashlib.sha256((D/'calibration_choices_frozen_before_test.json').read_bytes()).hexdigest()==exp['calibration_choices_sha256']
    grid=pd.read_csv(D/'calibration_budget_grid.csv');states=np.load(D/'domain_states.npz')
    t=states['time_s'];tr=t<3*86400
    for mod,choice in zip(models,frozen['choices']):
        assert mod['profile']==choice['profile']
        z=states[f'layout{mod["layout"]}_intent{mod["intent"]}'];back=mod['lag_s']//10
        x=z[back:];y=z[:-back];mask=tr[back:]&(x>=0).all(1)&(y>=0).all(1)
        pp,aa,bb=calibrate(x[mask].astype(bool),y[mask].astype(bool))
        assert mask.sum()==mod['train_n']
        for key,val in [('p',pp),('alpha',aa),('beta',bb)]:np.testing.assert_allclose(val,mod[key],rtol=0,atol=1e-12)
        local=grid[grid.profile==mod['profile']].to_dict('records')
        for m in METHODS:assert select_budget(local,m)==choice['selected_epsilon'][m]
    rows=pd.read_csv(D/'policy_summary.csv')
    assert len(rows)==2268 and len(grid)==2268 and len(models)==54
    assert np.all(rows.success_n+rows.empty_stop_n==rows.n)
    assert np.all((rows.miss_n>=0)&(rows.miss_n<=rows.empty_stop_n))
    assert np.all(rows.loc[(rows.epsilon==0)|(rows.method=='query_only'),'miss_n']==0)
    defined=rows.empty_stop_n>0
    np.testing.assert_allclose(rows.loc[defined,'conditional_miss'],rows.loc[defined,'miss_n']/rows.loc[defined,'empty_stop_n'],rtol=0,atol=1e-12)
    assert rows.loc[~defined,'conditional_miss'].isna().all()
    assert np.all(rows.mean_model_stop_risk<=rows.epsilon+1e-10)
    for _,g in rows.groupby(['profile','scope','protocol']):assert g.n.nunique()==1 and len(g)==7
    data=np.load(D/'alibaba2018_measured_subset.npz')
    assert data['cpu'].shape==data['mem'].shape==(24,69120)
    assert len(set(data['machine_ids']))==24
    for name in ['cpu','mem']:
        v=data[name];assert np.all((v[np.isfinite(v)]>=0)&(v[np.isfinite(v)]<=100))
    # CSV exports must preserve the full calendar and three-valued domain states.
    x=pd.read_csv(D/'domain_states.csv.gz');assert len(x)==3*3*69120
    for layout in range(3):
        for it in range(3):
            v=x[(x.layout==layout)&(x.intent==it)]
            np.testing.assert_array_equal(v.time_s.to_numpy(),t)
            np.testing.assert_array_equal(v[[f'domain_{i+1}' for i in range(4)]].to_numpy(),states[f'layout{layout}_intent{it}'])
    cd=ROOT/'results/counterexample'
    c=json.loads((cd/'joint_subset_diagnosis.json').read_text())
    assert abs(c['root_expected_regret']-c['total_expected_regret'])<1e-10
    assert abs(c['PSA1_cost']-c['exact_cost']-c['total_expected_regret'])<1e-10
    assert abs(c['depth_comparison'][1]['policy_cost']-c['exact_cost'])<1e-10
    result=dict(status='passed',strict_json_files=len(paths),training_only_parameter_reconstructions=len(models),
      frozen_calibration_choices_reconstructed=len(models)*len(METHODS),test_result_rows=len(rows),calibration_grid_rows=len(grid),
      all_epsilon_zero_and_strict_query_misses=0,full_calendar_domain_export_rows=len(x),
      original_unit_test_run='six targeted tests passed before the full replay; recorded in unit_test_run.txt',
      model_checks='Every fitted model enumerated all 256 truth/refresh outcomes for seven methods during the replay; recursive values and model STOP bounds checked there')
    (ROOT/'provenance/final_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
