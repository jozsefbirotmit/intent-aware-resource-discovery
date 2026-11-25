"""Find the largest fitted-model PSA1/DP gap and verify the acceleration."""
import argparse
import itertools
import json
import math
from pathlib import Path
from ia_osr_psa_audit import PSASolver,clear_all
from ia_osr_experiment import Params


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',required=True);args=ap.parse_args();r=Path(args.results)
    f=json.loads((r/'fitted_models.json').read_text())
    w=max((x for x in f if x['expected_costs'][-1]>1e-10),key=lambda x:x['expected_costs'][-2]/x['expected_costs'][-1])
    p=Params(tuple(w['p']),tuple(w['q']),tuple(w['r']),tuple(w['alpha']),tuple(w['beta']),w['epsilon'])
    sol=PSASolver(p);count=0
    for state in itertools.product(range(7),repeat=4):
        if sol.risk(state,True)>p.eps+1e-12:continue
        if not math.isclose(sol.HPSA1(state),sol.HPSA1fast(state),rel_tol=1e-10,abs_tol=1e-10):
            raise AssertionError(f'Fast/direct continuation mismatch at {state}')
        count+=1
    opt=sol.exact(sol.initial);psa=sol.psa_fast_cost(sol.initial)
    out=dict(source_model=w,exact_expected_cost=opt,psa_expected_cost=psa,relative_gap=psa/opt-1,
        H_PSA1=sol.HPSA1(sol.initial),PSA_initial_action=sol.psa_fast_action(sol.initial),
        exact_initial_action=min(sol.actions(sol.initial),key=lambda a:sol.q_exact(sol.initial,a)),
        root_action_exact_costs=[dict(action=a,cost=sol.q_exact(sol.initial,a)) for a in sol.actions(sol.initial)],
        fast_vs_direct_continuation_checked_admissible_states=count,
        interpretation='Single learned independent model, not a trace-averaged gap or a violation of the rollout upper bound.')
    (r/'algorithmic_counterexample.json').write_text(json.dumps(out,indent=2));clear_all()
    print('Verified',count,'states; relative fitted-model gap',out['relative_gap'])


if __name__=='__main__':main()
