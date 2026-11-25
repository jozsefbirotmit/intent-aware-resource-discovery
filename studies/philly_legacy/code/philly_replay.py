"""Four-domain replay of measured Philly signals and public-subset allocations.

All lags and both occupancy predicates share test timestamps. Telemetry and
job-filtered candidate states are explicitly not scheduler allocatability.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from decision_evaluation import Params,METHODS,FIELDS,calibrate,codes,make_tables,weekly_block_bootstrap,correlations
from prepare_philly import SPLIT

LAGS=(1,5,15,60)
MODES=('telemetry','logged_filter')


def host_states(gpu,free_fraction,busy,k,mode,util_limit=20.,free_limit=.05,uncertain=None):
    observed=np.isfinite(gpu)
    blocked=busy if mode=='logged_filter' else np.zeros_like(busy)
    uncertain=np.zeros_like(busy) if uncertain is None or mode=='telemetry' else uncertain
    observed_low=observed&(gpu<=util_limit)
    low=observed_low&~blocked&~uncertain
    possible=(~observed|observed_low)&~blocked
    mem_known=np.isfinite(free_fraction)
    mem_good=mem_known&(free_fraction>=free_limit)
    positive=mem_good&(low.sum(axis=2)>=k)
    negative=mem_known&((~mem_good)|(possible.sum(axis=2)<k))
    return positive,positive|negative


def domain_states(host_positive,host_known,groups):
    positive=np.column_stack([host_positive[:,g].any(axis=1) for g in groups])
    known=np.column_stack([host_positive[:,g].any(axis=1)|host_known[:,g].all(axis=1) for g in groups])
    return positive,known


def common_mask(known_list,lags=LAGS):
    known=np.logical_and.reduce([k.all(axis=1) for k in known_list])
    ok=known.copy()
    for lag in lags:
        shifted=np.zeros(len(known),dtype=bool)
        shifted[lag:]=known[:-lag]
        ok &= shifted
    return ok


def run(args):
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    z=np.load(args.input,allow_pickle=False)
    # Conservative availability: source minute t is usable only when that minute ends.
    index=pd.to_datetime((z['bin_start_unix_minute']+1)*60,unit='s',utc=True)
    gpu=z['gpu_util'];mem=z['host_mem_free_fraction'];busy=z['logged_busy'];uncertain=z['logged_uncertain'];hosts=z['machine'].tolist()
    test_start=SPLIT+pd.Timedelta(minutes=max(LAGS))
    end=index[-1]+pd.Timedelta(minutes=1)
    days=pd.date_range(test_start.floor('D'),index[-1].floor('D'),freq='D')
    day_labels=index.floor('D')
    order=np.random.default_rng(2026).permutation(len(hosts))
    layouts=[[list(order[8*r+2*d:8*r+2*d+2]) for d in range(4)] for r in range(3)]
    rows=[];daily=[];models=[];cal_rows=[];states_export=[];state_stats=[];layouts_meta=[]
    cache={};scenario=0;start_clock=time.monotonic()
    modes=('telemetry',) if args.telemetry_only else MODES
    for rep,groups in enumerate(layouts):
        layouts_meta.append(dict(replicate=rep,groups=[[hosts[i] for i in g] for g in groups]))
        states={}
        for k in [2,4]:
            for mode in modes:
                hp,hk=host_states(gpu,mem,busy,k,mode,uncertain=uncertain)
                states[k,mode]=domain_states(hp,hk,groups)
        common=common_mask([known for _,known in states.values()])
        train_base=common&(index<SPLIT)
        test_base=common&(index>=test_start)
        if train_base.sum()<1000 or test_base.sum()<1000:
            raise ValueError(f'Insufficient common train/test observations for replicate {rep}.')
        for (k,mode),(x,known) in states.items():
            for phase,mask in [('train',train_base),('test',test_base)]:
                xx=x[mask]
                for i in range(4):
                    state_stats.append(dict(replicate=rep,gpus_requested=k,mode=mode,phase=phase,
                        domain=i,n=int(mask.sum()),prevalence=float(xx[:,i].mean()),
                        all_domains_empty=float((~xx.any(axis=1)).mean())))
            export=pd.DataFrame({'available_at_utc':index,'replicate':rep,'gpus_requested':k,'mode':mode,
                                 'common_all_lags':common,'train':train_base,'test':test_base})
            for i in range(4):export[f'X{i}']=np.where(known[:,i],x[:,i].astype(float),np.nan)
            states_export.append(export)
            for lag in LAGS:
                y=np.zeros_like(x);y[lag:]=x[:-lag]
                code=codes(x,y)
                for calibration in ['static','rolling14d']:
                    if calibration=='static':
                        periods=[(test_start,end,index[0],SPLIT)]
                    else:
                        boundaries=[test_start]+list(pd.date_range(test_start+pd.Timedelta(days=7),end,freq='7D'))+[end]
                        periods=[(a,b,a-pd.Timedelta(days=14),a) for a,b in zip(boundaries[:-1],boundaries[1:]) if a<b]
                    fits=[]
                    for period,(a,b,ta,tb) in enumerate(periods):
                        tr=common&(index>=ta)&(index<tb)
                        te=test_base&(index>=a)&(index<b)
                        if not te.any():continue
                        static_fallback=bool(tr.sum()<100)
                        if static_fallback:
                            # No lookahead: preserve the pre-test fitted model when
                            # missing/uncertain history cannot support a recent fit.
                            tr=common&(index<SPLIT)
                        p,alpha,beta=calibrate(x[tr],y[tr])
                        fits.append((period,tr,te,p,alpha,beta,static_fallback))
                        for i in range(4):
                            g=p[i]*alpha[i]+(1-p[i])*(1-beta[i])
                            for signal in [0,1]:
                                tm=te&(y[:,i]==bool(signal))
                                predicted=p[i]*alpha[i]/g if signal else p[i]*(1-alpha[i])/(1-g)
                                cal_rows.append(dict(replicate=rep,gpus_requested=k,mode=mode,lag_min=lag,
                                    calibration=calibration,period=period,domain=i,signal=signal,
                                    test_n=int(tm.sum()),predicted=predicted,
                                    observed=float(x[tm,i].mean()) if tm.any() else np.nan))
                    for profile,q in [('equal',(1.,1.,1.,1.)),('heterogeneous',(4.,2.,1.,3.))]:
                        ratios=[.05,.2] if rep==0 and k==4 else [.05]
                        for ratio in ratios:
                            for eps in [0.,.1]:
                                scenario+=1
                                base=dict(scenario=scenario,replicate=rep,gpus_requested=k,mode=mode,
                                          lag_min=lag,calibration=calibration,costs=profile,refresh_ratio=ratio,epsilon=eps)
                                results=np.full((len(METHODS),int(test_base.sum()),len(FIELDS)),np.nan)
                                test_indices=np.flatnonzero(test_base)
                                expected_sum=np.zeros(len(METHODS));n=0
                                for period,tr,te,p,alpha,beta,static_fallback in fits:
                                    par=Params(p=p,q=q,r=tuple(ratio*v for v in q),alpha=alpha,beta=beta,eps=eps)
                                    # Reuse identical learned models, never test outcomes.
                                    key=tuple(p)+tuple(alpha)+tuple(beta)+tuple(q)+(ratio,eps)
                                    if key not in cache:cache[key]=make_tables(par)
                                    table,expected=cache[key]
                                    testloc=np.searchsorted(test_indices,np.flatnonzero(te))
                                    results[:,testloc,:]=table[:,code[te],:]
                                    nt=int(te.sum());expected_sum+=nt*expected;n+=nt
                                    models.append(dict(**base,period=period,train_n=int(tr.sum()),test_n=nt,
                                        static_fallback=static_fallback,
                                        p=list(p),alpha=list(alpha),beta=list(beta),q=list(q),r=list(par.r),
                                        expected_costs=expected.tolist(),train_correlations=correlations(x[tr]),
                                        test_correlations=correlations(x[te]),test_joint_counts=np.bincount(code[te],minlength=256).tolist()))
                                if np.isnan(results).any() or n!=len(test_indices):
                                    raise AssertionError('A calibration period failed to cover the common test set.')
                                for mi,method in enumerate(METHODS):
                                    val=results[mi]
                                    f=pd.DataFrame(val,columns=FIELDS);f['day']=day_labels[test_base];f['n']=1
                                    f['baseline_cost']=results[0,:,0];f['cost_saving']=results[0,:,0]-val[:,0]
                                    bd=f.groupby('day').sum().reindex(days,fill_value=0).reset_index(names='day')
                                    stop=int(val[:,2].sum());miss=int(val[:,3].sum())
                                    lo,hi,nb=weekly_block_bootstrap(bd,'miss','empty_stop',args.bootstrap)
                                    cl,ch,_=weekly_block_bootstrap(bd,'cost_saving','baseline_cost',args.bootstrap)
                                    rows.append(dict(**base,method=method,n=n,mean_cost=float(val[:,0].mean()),
                                        model_expected_cost=float(expected_sum[mi]/n),success_n=int(val[:,1].sum()),
                                        empty_stop_n=stop,miss_n=miss,conditional_miss=miss/stop if stop else np.nan,
                                        miss_per_request=miss/n,risk_block95_low=lo,risk_block95_high=hi,
                                        stopping_days=int((bd.empty_stop>0).sum()),nonempty_bootstrap_replicates=nb,
                                        saving_vs_query=1-val[:,0].mean()/results[0,:,0].mean(),
                                        saving_block95_low=cl,saving_block95_high=ch,
                                        mean_queries=float(val[:,4].mean()),mean_refreshes=float(val[:,5].mean()),
                                        mean_skips=float(val[:,6].mean()),
                                        max_model_stop_risk=float(val[val[:,2]==1,7].max()) if stop else np.nan,
                                        risk_inference_status=('undefined_no_empty_stops' if not stop else
                                            'zero_misses_bootstrap_degenerate' if not miss else 'descriptive_block_interval_only')))
                                    for dd in bd.to_dict('records'):
                                        daily.append(dict(scenario=scenario,method=method,**{k:(str(v) if k=='day' else v) for k,v in dd.items()}))
                                if scenario%16==0:
                                    print('completed',scenario,'configs',round(time.monotonic()-start_clock,1),'s',flush=True)
                                    pd.DataFrame(rows).to_csv(root/'policy_summary_checkpoint.csv',index=False)
    summary=pd.DataFrame(rows);summary.to_csv(root/'policy_summary.csv',index=False)
    pd.DataFrame(daily).to_csv(root/'daily_policy_statistics.csv.gz',index=False)
    pd.DataFrame(cal_rows).to_csv(root/'posterior_calibration.csv',index=False)
    pd.DataFrame(state_stats).to_csv(root/'state_prevalence.csv',index=False)
    pd.concat(states_export,ignore_index=True).to_csv(root/'domain_states.csv.gz',index=False)
    (root/'fitted_models.json').write_text(json.dumps(models,allow_nan=False,separators=(',',':')))
    meta=dict(scenarios=scenario,methods=list(METHODS),test_start_utc=str(test_start),split_utc=str(SPLIT),
        layouts=layouts_meta,lags_minutes=list(LAGS),gpu_util_limit=20,host_free_memory_fraction=.05,
        requested_gpus=[2,4],epsilons=[0,.1],refresh_ratios='0.05 for all; 0.20 additionally for replicate 0, 4-GPU intents',
        calibration='static preceding data; or trailing14days reset every7days, strictly preceding predictions; explicit static pre-test fallback if fewer than100 recent paired observations',
        common_test=('same timestamps across four lags and both intents; no job-log availability restriction' if args.telemetry_only else
            'same timestamps across four lags, both intents and both occupancy modes within each layout'),
        mask_scope='telemetry_only' if args.telemetry_only else 'joint_telemetry_and_job',
        seeds=dict(partition=2026,bootstrap=17),bootstrap=dict(draws=args.bootstrap,calendar_block_days=7,
            limitation='Descriptive sensitivity; rare-event zero-count degeneracy and nonstationarity prevent a general assurance claim.'),
        warnings=['Artificial domains group two anonymous measured hosts; no real federation/topology is inferred.',
            'Public-subset job exclusion does not establish scheduler availability.','Per-minute GPU averages are not dynamic VRAM measurements.',
            'All management-action costs are modelled, not measured network RTTs.'],
        elapsed_seconds=time.monotonic()-start_clock)
    (root/'experiment_metadata.json').write_text(json.dumps(meta,indent=2))
    print('Finished',scenario,'configurations,',len(models),'fitted configurations,',round(meta['elapsed_seconds'],1),'seconds',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--telemetry-only',action='store_true')
    ap.add_argument('--bootstrap',type=int,default=1000);run(ap.parse_args())
