"""Exact decision-tree diagnosis of the two saved Philly counterexamples."""
import csv
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from ia_osr_experiment import Params
from ia_osr_psa_audit import PSASolver, clear_all

ROOT = Path(__file__).resolve().parents[1]

def analyse(src, label):
    raw = json.loads(src.read_text())
    m = raw['source_model']
    par = Params(**{k: tuple(m[k]) for k in ('p','q','r','alpha','beta')}, eps=m['epsilon'])
    so = PSASolver(par)
    s0 = so.initial
    order = {'STOP':0, 'SKIP':1, 'Q':2, 'R':3}

    @lru_cache(None)
    def H(s,k):
        if so.stop(s): return 0.
        if k == 0: return so.HPSA1fast(s)
        return min(score(s,a,k) for a in so.actions(s))

    def score(s,a,k):
        c, br=so.successors(s,a)
        return c+sum(pr*H(ns,k-1) for pr,ns in br if pr>0)

    def act(s,k):
        if so.stop(s): return ('STOP',-1)
        return min(so.actions(s),key=lambda a:(round(score(s,a,k),12),order[a[0]],a))

    @lru_cache(None)
    def cost(s,k):
        if so.stop(s): return 0.
        a=act(s,k); c,br=so.successors(s,a)
        return c+sum(pr*cost(ns,k) for pr,ns in br if pr>0)

    def optact(s):
        return min(so.actions(s), key=lambda a:(round(so.q_exact(s,a),12),order[a[0]],a))

    roots=[]
    for a in so.actions(s0):
        c,br=so.successors(s0,a)
        roots.append(dict(action=a, one_step_score=score(s0,a,1),two_step_score=score(s0,a,2),
                          exact_action_value=so.q_exact(s0,a),
                          forced_first_then_PSA1=c+sum(pr*cost(ns,1) for pr,ns in br if pr>0)))

    def traverse(policy):
        mass=defaultdict(float); mass[s0]=1.; states=[]; ends=[]
        # Every action monotonically reduces this potential, including SKIP.
        def potential(s): return sum(3 if k==0 else 2 if k in (1,2) else 0 for k in s)
        for pot in range(potential(s0),-1,-1):
            for s,reach in list(mass.items()):
                if potential(s)!=pot or reach<=0: continue
                if so.stop(s):
                    ends.append(dict(state=s, reach=reach, residual_risk=so.risk(s)))
                    continue
                a=policy(s); c,br=so.successors(s,a)
                regret=so.q_exact(s,a)-so.exact(s)
                states.append(dict(state=s,reach=reach,action=a,risk=so.risk(s),
                                   skipped_risk=so.risk(s,True),optimal_value=so.exact(s),
                                   action_exact_value=so.q_exact(s,a),local_regret=regret,
                                   weighted_regret=reach*regret))
                for pr,ns in br:
                    if pr>0: mass[ns]+=reach*pr
        return states,ends

    ps,pe=traverse(lambda s:act(s,1)); os,oe=traverse(optact)
    opt=so.exact(s0); psac=cost(s0,1)
    assert abs(opt-raw['exact_expected_cost'])<1e-10
    assert abs(psac-raw['psa_expected_cost'])<1e-10
    assert abs(sum(r['weighted_regret'] for r in ps)-(psac-opt))<1e-10
    assert all(e['residual_risk']<=par.eps+1e-12 for e in pe+oe)
    assert abs(psac-so.psa_fast_cost(s0))<1e-10
    # Follow only negative REFRESH and failed QUERY branches to expose the cheap path.
    negative=[]; s=s0; pathprob=1.; accumulated=0.
    while not so.stop(s):
        a=optact(s); c,br=so.successors(s,a)
        negative.append(dict(state=s,action=a,risk=so.risk(s),path_probability=pathprob))
        accumulated+=c
        if a[0]=='R': pr,s=br[1]
        else: pr,s=br[0]
        pathprob*=pr
    negative.append(dict(state=s,action=('STOP',-1),risk=so.risk(s),
                         path_probability=pathprob,accumulated_cost=accumulated))

    allnegative=tuple(2 for _ in par.p)
    skips={str(i):so.risk(so.set(allnegative,i,4)) for i in range(so.n)}
    output=dict(source=label,parameters=dict(p=par.p,q=par.q,r=par.r,alpha=par.alpha,beta=par.beta,epsilon=par.eps),
                negative_posteriors=so.pm,positive_posteriors=so.pp,negative_signal_probabilities=so.h,
                root_risk=so.risk(s0),all_negative_risk=so.risk(allnegative),
                all_other_negative_with_unrefreshed_committed_skip_risk=skips,
                exact_cost=opt,PSA1_cost=psac,gap_percent=100*(psac/opt-1),
                root_actions=roots,depth_comparison=[dict(depth=k,action=act(s0,k),
                 root_upper_bound=H(s0,k),policy_cost=cost(s0,k),gap_percent=100*(cost(s0,k)/opt-1)) for k in (1,2,3,4)],
                optimal_all_negative_path=negative,PSA1_states=ps,PSA1_stops=pe,optimal_states=os,optimal_stops=oe,
                total_expected_regret=sum(r['weighted_regret'] for r in ps),
                root_expected_regret=next(r['weighted_regret'] for r in ps if tuple(r['state'])==s0))
    out=ROOT/'results/counterexample';out.mkdir(parents=True,exist_ok=True)
    (out/f'{label}_diagnosis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    with (out/f'{label}_root_actions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=roots[0].keys());w.writeheader();w.writerows(roots)
    print(json.dumps({k:output[k] for k in ('source','exact_cost','PSA1_cost','gap_percent','negative_posteriors','all_negative_risk','all_other_negative_with_unrefreshed_committed_skip_risk','depth_comparison','optimal_all_negative_path','root_expected_regret','total_expected_regret')},indent=2))
    clear_all()
    return output

if __name__=='__main__':
    analyse(ROOT/'inputs/philly_counterexample.json','joint_subset')
    analyse(ROOT/'inputs/philly_full_telemetry_counterexample.json','full_telemetry')
