"""Check WP4 synthetic claims against aggregate outputs of v1.0.0 and local audit CSV."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('repository',type=Path);a=ap.parse_args();r=a.repository/'studies'
b=pd.read_csv(Path(__file__).with_name('baseline_60.csv'));assert len(b)==60 and b.seed.nunique()==60
np.testing.assert_allclose(b[['HQ_saving','H1_saving']].mean(),[14.360287108610256,15.24639734218271],rtol=0,atol=1e-9)
q=pd.read_csv(r/'final_benchmark/quality_summary.csv').query('group=="N10-12"').set_index('algorithm');np.testing.assert_allclose(q.loc[['psa','tail8','selective8','multi8'],'mean_saving_no_refresh'],[16.182003315286043,17.055803848729806,18.024033153861307,18.204447613659497],rtol=0,atol=1e-9)
t=pd.read_csv(r/'final_benchmark/online_summary.csv').query('group=="N10-12"').set_index('algorithm');np.testing.assert_allclose(t.loc[['selective8','multi8'],'trajectory_median_ms'],[34.09709949997364,110.77738500000578],rtol=0,atol=1e-9)
s=pd.read_csv(r/'mechanisms/results/sensitivity_summary.csv').query('eps==0.1 and r_over_q==0.2').sort_values('accuracy');np.testing.assert_allclose(100*(1-s.mean_psa_cost/s.mean_no_refresh_cost),s.saving_vs_no_refresh_pct,rtol=0,atol=1e-9);assert s.saving_vs_no_refresh_pct.round(1).tolist()==[4.0,10.7,32.2]
d=pd.read_csv(r/'mechanisms/results/calibration_drift_summary.csv').sort_values('delta_logodds');assert (100*d.weighted_risk_given_failure).round(1).tolist()==[9.3,14.5,22.0,32.1];assert round(100*d.iloc[0].mean_eps,2)==12.15 and d.iloc[0].frac_instance_risk_above_eps==0
print('PASS: all WP4 synthetic aggregate claims and audit baseline means')
