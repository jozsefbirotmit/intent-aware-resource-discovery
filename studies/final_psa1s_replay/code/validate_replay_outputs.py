"""Final artifact, frozen-source, accounting and calibration consistency checks."""
from pathlib import Path
import hashlib,json,platform
import numpy as np
import pandas as pd
import numba,llvmlite,matplotlib
from replay_statistics import select_budget

ROOT=Path(__file__).resolve().parents[1]

def main():
    def reject(v):raise ValueError('Nonstandard numeric JSON constant: '+v)
    for p in ROOT.rglob('*.json'):json.loads(p.read_text(),parse_constant=reject)
    sources=json.loads((ROOT/'provenance/frozen_input_sources.json').read_text())
    for x in sources:
        p=ROOT/x['file'];assert p.stat().st_size==x['size_bytes']
        assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==x['sha256'],x['file']
    metadata={};counts={};models_checked=0
    for name,nrows,nmodels in [('philly',1120,896),('alibaba2018',1620,324)]:
        folder=ROOT/'results'/name;d=pd.read_csv(folder/'policy_summary.csv');assert len(d)==nrows
        assert np.all(d.success_n+d.empty_stop_n==d.n)
        assert np.all((d.miss_n>=0)&(d.miss_n<=d.empty_stop_n))
        assert np.all(d.loc[(d.epsilon==0)|(d.method=='query_only'),'miss_n']==0)
        good=d.empty_stop_n>0
        np.testing.assert_allclose(d.loc[good,'conditional_miss'],d.loc[good,'miss_n']/d.loc[good,'empty_stop_n'],rtol=0,atol=1e-12)
        assert d.loc[~good,'conditional_miss'].isna().all()
        assert np.all(d.loc[good,'max_model_stop_risk']<=d.loc[good,'epsilon']+1e-10)
        zero_base=d.no_refresh_same_budget_mean_cost==0
        assert d.loc[zero_base,'saving_vs_no_refresh_percent'].isna().all()
        audit=json.loads((folder/'model_tie_audit.json').read_text());assert len(audit)==nmodels
        for a in audit:
            ex=a['model_expected_costs'];assert abs(ex['psa1_s']-ex['legacy_exact'])<1e-9
            assert ex['psa1_s']<=ex['no_refresh_opt']+1e-9
            assert a['all_outcome_cost_disagreements']==0 and a['selective_refinements']==0
        models_checked+=len(audit)
        old=json.loads((folder/'legacy_reproduction_checks.json').read_text())
        assert all(x['counts_identical'] and x['max_cost_difference']<1e-9 for x in old)
        metadata[name]=json.loads((folder/'replay_metadata.json').read_text());assert not metadata[name]['parameters_refitted']
        counts[name]=len(d)
    folder=ROOT/'results/alibaba2018';choicefile=folder/'calibration_choices_frozen_before_test.json'
    assert hashlib.sha256(choicefile.read_bytes()).hexdigest()==metadata['alibaba2018']['calibration_choices_sha256']
    grid=pd.read_csv(folder/'calibration_budget_grid.csv');choices=json.loads(choicefile.read_text())['choices']
    for c in choices:
        cal=grid[grid.profile==c['profile']].to_dict('records')
        for m,v in c['selected_epsilon'].items():assert select_budget(cal,m)==v
    assert 'Ran 5 tests' in (ROOT/'provenance/unit_test_run.txt').read_text() and '\nOK\n' in (ROOT/'provenance/unit_test_run.txt').read_text()
    env=dict(python=platform.python_version(),numpy=np.__version__,numba=numba.__version__,llvmlite=llvmlite.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,runtime_benchmark_performed=False)
    (ROOT/'provenance/environment.json').write_text(json.dumps(env,indent=2)+'\n')
    result=dict(status='passed',frozen_files_verified=len(sources),saved_model_variants_checked=models_checked,policy_result_rows=counts,
        independent_unit_tests=5,legacy_cost_and_count_reproduction='passed',calibration_choices_reconstructed=54*5,
        all_epsilon_zero_miss_counts=0,all_model_STOP_bounds='passed',selective_refinements=0,
        interpretation='Small N<=K replay verifies the final exact-tail mode, not selective refinement performance')
    (ROOT/'provenance/final_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
