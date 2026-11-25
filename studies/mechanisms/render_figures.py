"""Recreate publication-oriented figures from experiment CSV exports."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create(root):
    root=Path(root);fig=root/'figures';fig.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(root/'family_summary.csv')
    labs=['Unstructured (N=4-5)','P4: ordering probe (N=2)','P3: screening (N=2)',
          'Deferred refresh (N=2)','Skip threshold: below (N=4)','Skip threshold: above (N=4)']
    y=np.arange(len(data))
    saving=data.saving_pct.to_numpy()
    err=np.array([saving-data.saving_low.to_numpy(),data.saving_high.to_numpy()-saving])
    f,ax=plt.subplots(figsize=(9.0,4.25))
    ax.errorbar(saving,y,xerr=err,fmt='o',capsize=3)
    ax.set_yticks(y,labs);ax.invert_yaxis();ax.set_xlabel('PSA1 expected-cost saving relative to query-only (%)')
    ax.grid(axis='x',alpha=.23);ax.set_xlim(0,70)
    f.tight_layout();f.savefig(fig/'fig_family_savings.png',dpi=200);plt.close(f)

    a=pd.read_csv(root/'sensitivity_summary.csv')
    eps=.10
    B=a[a.eps==eps].pivot(index='accuracy',columns='r_over_q',values='saving_vs_no_refresh_pct')
    f,ax=plt.subplots(figsize=(6.3,3.8))
    pic=ax.imshow(B.values,origin='lower',aspect='auto')
    ax.set_xticks(range(len(B.columns)),[f'{v:.2f}' for v in B.columns])
    ax.set_yticks(range(len(B.index)),[f'{v:.2f}' for v in B.index])
    ax.set_xlabel('Refresh-to-query cost ratio (r/q)')
    ax.set_ylabel('Refresh sensitivity and specificity')
    ax.set_title('PSA1 gain over optimal QUERY/SKIP without REFRESH ($\\varepsilon=0.10$)')
    for i in range(len(B.index)):
        for j in range(len(B.columns)):
            ax.text(j,i,f'{B.values[i,j]:.1f}%',ha='center',va='center',color='white' if B.values[i,j]>.6*np.max(B.values) else 'black')
    f.colorbar(pic,ax=ax,label='Expected-cost saving (%)',shrink=.85)
    f.tight_layout();f.savefig(fig/'fig_refresh_sensitivity.png',dpi=200);plt.close(f)

    d=pd.read_csv(root/'calibration_drift_summary.csv')
    f,ax=plt.subplots(figsize=(6.6,4.1))
    y=d.weighted_risk_given_failure.to_numpy()
    ax.errorbar(d.delta_logodds,y, yerr=np.array([y-d.risk_ci_low.to_numpy(),d.risk_ci_high.to_numpy()-y]),fmt='o-',capsize=3,label='Actual conditional missed-resource risk')
    ax.axhline(float(d.mean_eps.mean()),ls='--',label='Average nominal risk tolerance')
    ax.set_xticks(d.delta_logodds)
    ax.set_xlabel('True feasibility log-odds shift relative to belief')
    ax.set_ylabel('Conditional missed-resource probability')
    ax.legend(loc='upper left',fontsize=8)
    ax.grid(alpha=.24)
    f.tight_layout();f.savefig(fig/'fig_calibration_drift.png',dpi=200);plt.close(f)

    t=pd.read_csv(root/'runtime_scaling_summary.csv')
    f,ax=plt.subplots(figsize=(6.2,4))
    ax.plot(t.N,t.median_s,marker='o')
    ax.fill_between(t.N,t.min_s,t.max_s,alpha=.18)
    ax.set_xscale('log');ax.set_yscale('log')
    ax.set_xlabel('Number of domains N')
    ax.set_ylabel('First decision computation time (s)')
    ax.set_title('PSA1 Python first-action runtime; stress distribution')
    ax.grid(alpha=.24,which='both')
    f.tight_layout();f.savefig(fig/'fig_runtime_scaling.png',dpi=200);plt.close(f)

    pairs=pd.read_csv(root/'cases.csv');pairs=pairs[pairs.family=='skip_boundary']
    bp=pairs.pivot(index=['seed','index'],columns='kind',values='psa1')
    diffs=(bp['below']-bp['above']).values
    f,ax=plt.subplots(figsize=(6.3,3.75))
    ax.hist(diffs,bins=28,edgecolor='none')
    ax.axvline(diffs.mean(),linestyle='--',label=f'Mean difference: {diffs.mean():.3f}')
    ax.set_xlabel('PSA1 cost below minus above skip-admissibility threshold')
    ax.set_ylabel('Number of paired instances')
    ax.legend()
    ax.grid(axis='y',alpha=.22)
    f.tight_layout();f.savefig(fig/'fig_skip_threshold_pairs.png',dpi=200);plt.close(f)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('results',nargs='?',default='results')
    a=ap.parse_args();create(a.results)
