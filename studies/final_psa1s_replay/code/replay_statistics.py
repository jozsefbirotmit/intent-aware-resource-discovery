"""Descriptive paired calendar-block statistics; no independence certificate."""
import math
import numpy as np

def alibaba_bins(t,values,query,no_refresh):
    start=5*86400+3600;end=8*86400;n=math.ceil((end-start)/600)
    ix=((t-start)//600).astype(int);a=np.zeros((n,6))
    for j,v in enumerate([np.ones(len(t)),values[:,0],query,no_refresh,values[:,2],values[:,3]]):np.add.at(a[:,j],ix,v)
    return a

def alibaba_bootstrap(a,draws=1000,seed=1701):
    rng=np.random.default_rng(seed);n=len(a);starts=rng.integers(0,n-5,size=(draws,math.ceil(n/6)))
    ix=(starts[:,:,None]+np.arange(6)).reshape(draws,-1)[:,:n];s=a[ix].sum(axis=1)
    def ci(num,den):
        g=den>0
        if not g.any():return np.nan,np.nan,0
        lo,hi=np.quantile(num[g]/den[g],[.025,.975]);return float(lo),float(hi),int(g.sum())
    r=ci(s[:,5],s[:,4]);cq=ci(100*(s[:,2]-s[:,1]),s[:,2]);cn=ci(100*(s[:,3]-s[:,1]),s[:,3])
    return dict(risk_block95_low=r[0],risk_block95_high=r[1],risk_bootstrap_defined=r[2],
                saving_query_block95_low=cq[0],saving_query_block95_high=cq[1],
                saving_no_refresh_block95_low=cn[0],saving_no_refresh_block95_high=cn[1])

def select_budget(cal_rows,method):
    if method=='query_only':return 0.
    ok=[x['epsilon'] for x in cal_rows if x['method']==method and x['epsilon']>0 and x['empty_stop_n']>=200 and x['conditional_miss']<=.05]
    return max(ok,default=0.)

def relative_saving(cost,baseline):return 100*(1-cost/baseline) if baseline>0 else np.nan
