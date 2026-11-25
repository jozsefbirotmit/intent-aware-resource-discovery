import json,unittest
from pathlib import Path
import numpy as np
from ia_osr_experiment import Params
from alibaba_replay import domain_states,pairs,select_budget,extended_tables,METHODS,FIELDS,bootstrap

class StudyChecks(unittest.TestCase):
    def test_unknown_measurements_do_not_become_negative_and_or_short_circuits(self):
        cpu=np.full((8,3),np.nan);mem=np.full((8,3),np.nan);using=np.ones((8,3),np.int8)
        mem[0]=95  # known failure despite missing CPU
        cpu[1,1]=90  # second host is now also known to fail
        cpu[2]=10;mem[2]=20  # known success despite absent second host
        using[4:6]=0  # source metadata explicitly says unavailable
        s=domain_states(cpu,mem,using,60,80)
        self.assertEqual(s[:,0].tolist(),[-1,0,-1])
        self.assertEqual(s[:,1].tolist(),[1,1,1])
        self.assertEqual(s[:,2].tolist(),[0,0,0])
        self.assertEqual(s[:,3].tolist(),[-1,-1,-1])

    def test_lag_is_calendar_distance_without_compressing_holes(self):
        z=np.array([[0]*4,[-1]*4,[1]*4,[0]*4],dtype=np.int8)
        back,x,y,k=pairs(z,20)
        self.assertEqual(back,2);self.assertEqual(k.tolist(),[True,False])
        self.assertEqual(x[0].tolist(),[1]*4);self.assertEqual(y[0].tolist(),[0]*4)

    def test_budget_gate_uses_calibration_counts_and_returns_zero_if_unsupported(self):
        r=[dict(method='psa1',epsilon=.10,empty_stop_n=300,conditional_miss=.08),
           dict(method='psa1',epsilon=.05,empty_stop_n=300,conditional_miss=.04),
           dict(method='psa1',epsilon=.075,empty_stop_n=10,conditional_miss=0.)]
        self.assertEqual(select_budget(r,'psa1'),.05)
        self.assertEqual(select_budget(r,'h1'),0.)

    def test_zero_budget_never_misses_over_all_256_outcomes(self):
        par=Params((.03,.2,.55,.8),(4.,2.,1.,3.),(.2,.1,.05,.15),(.8,.9,.75,.95),(.9,.8,.95,.7),0.)
        table,expected=extended_tables(par)
        self.assertEqual(table.shape,(7,256,8))
        self.assertTrue(np.all(table[:,:,FIELDS.index('miss')]==0))
        self.assertTrue(np.all(table[:,:,1]+table[:,:,2]==1))

    def test_no_empty_stop_is_undefined_in_resampling(self):
        a=np.zeros((12,5));a[:,0]=1;a[:,1]=2;a[:,2]=3
        r=bootstrap(a,draws=20)
        self.assertTrue(np.isnan(r['conditional_miss_lo']))
        self.assertEqual(r['risk_bootstrap_defined'],0)

    def test_saved_counterexample_regret_is_entirely_at_first_skip(self):
        p=Path(__file__).resolve().parents[1]/'results/counterexample/joint_subset_diagnosis.json'
        d=json.loads(p.read_text())
        self.assertAlmostEqual(d['root_expected_regret'],d['total_expected_regret'],places=10)
        self.assertAlmostEqual(d['depth_comparison'][1]['policy_cost'],d['exact_cost'],places=10)
        self.assertGreater(d['all_other_negative_with_unrefreshed_committed_skip_risk']['0'],.1)
        self.assertLess(d['all_other_negative_with_unrefreshed_committed_skip_risk']['3'],.1)

if __name__=='__main__':unittest.main()
