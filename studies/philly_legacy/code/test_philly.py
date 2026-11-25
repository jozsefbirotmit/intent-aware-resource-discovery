"""Targeted checks of scientific semantics; fixtures are not trace results."""
import unittest
import numpy as np
import pandas as pd

from prepare_philly import explicit_minutes,scalar_job_time,interval_bins
from philly_replay import host_states,domain_states,common_mask
import decision_evaluation as d


class PhillyChecks(unittest.TestCase):
    def test_pdt_pst_repeated_hour_is_distinct(self):
        v,ok=explicit_minutes(pd.Series(['2017-11-05 01:30:00 PDT','2017-11-05 01:30:00 PST']))
        self.assertTrue(ok.all());self.assertEqual(v[1]-v[0],60)
        a,amb=scalar_job_time('2017-11-05 01:30:00',True,'America/Los_Angeles')
        b,_=scalar_job_time('2017-11-05 01:30:00',False,'America/Los_Angeles')
        self.assertTrue(amb);self.assertEqual((b-a).total_seconds(),3600)

    def test_utc_job_parse_matches_resource_manager_epoch_example(self):
        t,amb=scalar_job_time('2017-12-16 02:13:26')
        boot=pd.Timestamp(1513390397011,unit='ms',tz='UTC')
        self.assertFalse(amb);self.assertAlmostEqual((t-boot).total_seconds(),8.989,places=3)

    def test_half_open_job_interval_and_censoring(self):
        grid=pd.date_range('2017-10-03T07:00Z',periods=4,freq='min')
        self.assertEqual(interval_bins(grid[1],grid[2],grid),(1,2))
        self.assertEqual(interval_bins(grid[1]+pd.Timedelta(seconds=30),grid[2]+pd.Timedelta(seconds=1),grid),(1,3))
        self.assertEqual(interval_bins(None,grid[2],grid),(0,2))
        self.assertEqual(interval_bins(grid[2],None,grid),(2,4))

    def test_logged_allocation_excludes_an_idle_gpu(self):
        g=np.zeros((1,1,8));f=np.ones((1,1))*.1;b=np.ones_like(g,dtype=bool)
        pos,known=host_states(g,f,b,2,'telemetry');self.assertTrue(pos[0,0])
        pos,known=host_states(g,f,b,2,'logged_filter');self.assertFalse(pos[0,0]);self.assertTrue(known[0,0])

    def test_missing_job_release_yields_unknown_not_busy_or_free(self):
        g=np.zeros((1,1,8));f=np.ones((1,1))*.1;b=np.zeros_like(g,dtype=bool);u=np.ones_like(b)
        pos,known=host_states(g,f,b,2,'logged_filter',uncertain=u)
        self.assertFalse(pos[0,0]);self.assertFalse(known[0,0])

    def test_missing_is_unknown_not_failure_and_or_short_circuits(self):
        g=np.full((1,2,8),np.nan);g[0,0,:2]=0
        f=np.array([[.1,np.nan]]);b=np.zeros_like(g,dtype=bool)
        hp,hk=host_states(g,f,b,2,'telemetry')
        self.assertTrue(hp[0,0]);self.assertTrue(hk[0,0]);self.assertFalse(hk[0,1])
        x,k=domain_states(hp,hk,[[0,1]])
        self.assertTrue(x[0,0]);self.assertTrue(k[0,0])
        hp[:]=False
        x,k=domain_states(hp,hk,[[0,1]]);self.assertFalse(k[0,0])

    def test_all_lags_use_same_known_calendar_grid(self):
        known=np.ones((8,2),bool);known[3,0]=False
        mask=common_mask([known],lags=[1,2])
        np.testing.assert_array_equal(mask,[False,False,True,False,False,False,True,True])

    def test_four_domain_enumeration_matches_bellman(self):
        for eps in [0.,.1]:
            p=d.Params(p=(.2,.4,.6,.8),q=(4.,2.,1.,3.),r=(.2,.1,.05,.15),alpha=(.9,)*4,beta=(.9,)*4,eps=eps)
            tab,ex=d.make_tables(p)
            self.assertEqual(tab.shape,(6,256,8))
            self.assertTrue((tab[0,:,3]==0).all())
            if eps==0:self.assertTrue((tab[:,:,3]==0).all())
            self.assertLessEqual(ex[-1],ex[-2]+1e-9)


if __name__=='__main__':unittest.main(verbosity=2)
