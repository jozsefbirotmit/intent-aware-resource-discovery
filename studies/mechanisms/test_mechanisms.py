"""Structural sanity controls; this is numerical testing, not a replacement for proofs."""
import csv,itertools,math,random,json,unittest
from pathlib import Path
from mechanism_study import gen_p4,gen_p3,gen_deferred,exact_true_miss,logit_shift,p4_diagnostic,p3_diagnostic
from ia_osr_psa_audit import PSASolver,clear_all

class TestStructuralMechanisms(unittest.TestCase):
    def test_P1_query_order(self):
        rng=random.Random(37)
        for _ in range(1000):
            p1,p2=rng.uniform(.01,.99),rng.uniform(.01,.99)
            q1,q2=rng.uniform(.1,20),rng.uniform(.1,20)
            a=q1+(1-p1)*q2
            b=q2+(1-p2)*q1
            self.assertEqual(a<=b,q1/p1<=q2/p2)

    def test_P2_module_order(self):
        rng=random.Random(473)
        for _ in range(400):
            n=rng.randint(2,5)
            mod=[(rng.uniform(.05,12),rng.uniform(.04,.95)) for _ in range(n)]
            best=sorted(mod,key=lambda z:z[0]/z[1])
            def cost(seq):
                x=0.;surv=1.
                for c,s in seq:
                    x+=surv*c;surv*=1-s
                return x
            c=cost(best)
            self.assertTrue(all(c<=cost(perm)+1e-10 for perm in itertools.permutations(mod)))

    def test_P3_P4(self):
        for seed in (123,7654,12345):
            rng=random.Random(seed)
            for _ in range(300):
                p4=gen_p4(rng); diag=p4_diagnostic(p4)
                self.assertTrue(diag['p4_cross'])
                self.assertGreater(diag['p4_fixed_saving'],0.)
                p3=gen_p3(rng); diag=p3_diagnostic(p3)
                self.assertTrue(diag['p3_in_interval'])
                self.assertGreater(diag['p3_saving'],0.)

    def test_deferred_query_then_refresh(self):
        rng=random.Random(18)
        for _ in range(300):
            p=gen_deferred(rng);sol=PSASolver(p)
            s0=sol.initial
            self.assertEqual(sol.psa_fast_action(s0),('Q',1))
            sfail=sol.set(s0,1,3)
            self.assertEqual(sol.psa_fast_action(sfail),('R',0))
            clear_all()

    def test_risk_calibration_one_instance(self):
        p=gen_deferred(random.Random(19));sol=PSASolver(p)
        ev=exact_true_miss(sol,p.p)
        self.assertLessEqual(ev['risk_at_failure'],p.eps+1e-10)
        self.assertAlmostEqual(ev['cost_true'],sol.psa_fast_cost(sol.initial),places=10)
        clear_all()

if __name__=='__main__':unittest.main()
