import json,unittest
from pathlib import Path
import numpy as np
from final_policy_tables import *
from replay_statistics import select_budget

ROOT=Path(__file__).resolve().parents[1]

class FinalReplayChecks(unittest.TestCase):
    def test_final_tail_is_exact_and_no_selective_branch_at_three_domains(self):
        p=Params((.12,.35,.7),(1.,2.,3.),(.05,.1,.15),(.9,.85,.95),(.92,.94,.9),.1)
        tab,ex,flags,meta=make_tables(p)
        self.assertEqual(tab.shape,(5,64,8));self.assertEqual(meta['selective_refinements'],0)
        self.assertAlmostEqual(ex[METHODS.index('psa1_s')],ex[METHODS.index('legacy_exact')],places=10)

    def test_zero_budget_all_256_worlds_have_no_misses(self):
        p=Params((.03,.22,.57,.8),(4.,2.,1.,3.),(.2,.1,.05,.15),(.82,.9,.75,.95),(.9,.85,.95,.7),0.)
        tab,ex,flags,meta=make_tables(p)
        self.assertTrue(np.all(tab[:,:,3]==0));self.assertTrue(np.all(tab[:,:,1]+tab[:,:,2]==1))
        self.assertAlmostEqual(ex[METHODS.index('query_only')],ex[METHODS.index('no_refresh_opt')],places=10)

    def test_no_refresh_reference_never_reads_the_signal(self):
        p=Params((.04,.18,.5,.72),(4.,2.,1.,3.),(.2,.1,.05,.15),(.82,.9,.75,.95),(.9,.85,.95,.7),.1)
        tab,_,_,_=make_tables(p);v=tab[METHODS.index('no_refresh_opt')]
        self.assertTrue(np.all(v[:,5]==0))
        for truth in range(16):np.testing.assert_allclose(v[truth::16],np.tile(v[truth],(16,1)),rtol=0,atol=0)

    def test_same_exact_cost_does_not_imply_same_misses_on_a_real_model(self):
        audits=json.loads((ROOT/'results/philly/model_tie_audit.json').read_text())
        a=next(x for x in audits if x['all_outcome_miss_disagreements']>0)
        models=json.loads((ROOT/'input/philly/fitted_models.json').read_text())
        m=next(m for m in models if m['scenario']==a['scenario'] and m['period']==a['period'])
        tab,ex,flags,meta=make_tables(parameter(m))
        self.assertTrue(flags['miss'].any());self.assertFalse(flags['cost'].any())
        self.assertAlmostEqual(ex[METHODS.index('psa1_s')],ex[METHODS.index('legacy_exact')],places=10)
        self.assertLessEqual(tab[METHODS.index('psa1_s'),:,7].max(),m['epsilon']+1e-10)

    def test_calibration_gate_has_no_test_argument_and_falls_back_to_zero(self):
        rows=[dict(method='psa1_s',epsilon=.1,empty_stop_n=300,conditional_miss=.08),
              dict(method='psa1_s',epsilon=.05,empty_stop_n=300,conditional_miss=.04)]
        self.assertEqual(select_budget(rows,'psa1_s'),.05)
        self.assertEqual(select_budget(rows,'no_refresh_opt'),0.)

if __name__=='__main__':unittest.main()
