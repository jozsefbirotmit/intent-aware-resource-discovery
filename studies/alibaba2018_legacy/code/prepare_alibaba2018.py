"""Audit both original machine tables and extract a train-selected pilot.

No interpolation. Conflicting duplicates are unknown, rather than last-wins.
Only measurement coverage, never feasibility or policy results, selects hosts.
"""
import argparse, hashlib, json, tarfile, time
from pathlib import Path
import numpy as np
import pandas as pd

DAY=86400; STEP=10; END=8*DAY; T=END//STEP
META=['machine_id','time_stamp','failure_domain_1','failure_domain_2','cpu_num','mem_size','status']

def sha_order(x): return hashlib.sha256(('ia-osr-alibaba2018-v1:'+str(x)).encode()).hexdigest()

def read_meta(raw):
    with tarfile.open(raw/'machine_meta.tar.gz') as z:
        member=next(m for m in z if m.isfile() and m.name.endswith('.csv'))
        b=z.extractfile(member).read()
    from io import BytesIO
    return pd.read_csv(BytesIO(b),header=None,names=META), dict(name=member.name,size_bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def main(raw,out):
    out.mkdir(parents=True,exist_ok=True)
    meta,meta_source=read_meta(raw)
    meta.to_csv(out/'machine_metadata.csv.gz',index=False,compression='gzip')
    changes=meta.groupby('machine_id')[['failure_domain_1','failure_domain_2','cpu_num','mem_size']].nunique().gt(1).sum().to_dict()
    initial=meta[meta.time_stamp<=0].sort_values('time_stamp').drop_duplicates('machine_id',keep='last')
    eligible=initial[(initial.status=='USING')&initial.failure_domain_2.notna()]
    labels=sorted(eligible.failure_domain_2.unique().tolist(),key=lambda z:sha_order(str(int(z))))[:12]
    if len(labels)<12: raise ValueError('Fewer than twelve eligible source failure-domain labels')
    candidates=[]; groups=[]
    for g in labels:
        pool=sorted(eligible[eligible.failure_domain_2==g].machine_id.tolist(),key=sha_order)[:16]
        if len(pool)<2: raise ValueError('Insufficient eligible hosts in source domain')
        groups.append(pool);candidates+=pool
    assert len(candidates)==len(set(candidates))
    maxid=int(meta.machine_id.str[2:].astype(int).max())
    known=np.zeros(maxid+1,bool);known[meta.machine_id.str[2:].astype(int).unique()]=True
    cmap=np.full(maxid+1,-1,np.int32)
    for i,h in enumerate(candidates):cmap[int(h[2:])]=i
    n=len(candidates)
    low=np.full((2,n,T),np.inf,np.float32);high=np.full((2,n,T),-np.inf,np.float32)
    bad=np.zeros((2,n,T),bool)
    seen=np.zeros((maxid+1,T),np.uint8)
    rowcounts=np.zeros(maxid+1,np.int64);invalid_cpu=np.zeros_like(rowcounts);invalid_mem=np.zeros_like(rowcounts)
    first=np.full(maxid+1,np.inf);last=np.full(maxid+1,-np.inf)
    previous=np.full(maxid+1,np.nan)
    stats=dict(rows=0,unknown_machine_rows=0,invalid_timestamp_rows=0,off_10s_grid_rows=0,
               outside_8day_window_rows=0,window_grid_rows=0,file_order_time_reversals=0,
               cpu_invalid_rows=0,memory_invalid_rows=0,cpu_noninteger_rows=0,memory_noninteger_rows=0)
    hist=np.zeros((2,101),np.int64);unknown_ids=set();parts={};start=time.monotonic();printed=start
    with tarfile.open(raw/'machine_usage.tar.gz',mode='r:gz') as z:
        member=next(m for m in z if m.isfile() and m.name.endswith('.csv'))
        usage_source=dict(name=member.name,size_bytes=member.size)
        reader=pd.read_csv(z.extractfile(member),header=None,usecols=[0,1,2,3],names=['machine_id','time_stamp','cpu','mem'],
                           dtype={'machine_id':'category','time_stamp':'float64','cpu':'float32','mem':'float32'},chunksize=1000000)
        for chunk in reader:
            # Parse each categorical machine identifier once, rather than per row.
            codes=chunk.machine_id.cat.codes.to_numpy()
            cat=np.array([int(s[2:]) for s in chunk.machine_id.cat.categories],np.int32)
            mid=cat[codes]
            ts=chunk.time_stamp.to_numpy();cpu=chunk.cpu.to_numpy();mem=chunk.mem.to_numpy()
            stats['rows']+=len(chunk)
            validid=(mid>=0)&(mid<=maxid)
            validid[validid]&=known[mid[validid]]
            stats['unknown_machine_rows']+=int((~validid).sum());unknown_ids.update(mid[~validid].tolist())
            validtime=np.isfinite(ts)&(ts>=0)&(ts==np.rint(ts))
            stats['invalid_timestamp_rows']+=int((~validtime).sum())
            ongrid=validtime&(np.remainder(ts,STEP)==0)
            stats['off_10s_grid_rows']+=int((validtime&~ongrid).sum())
            window=ongrid&(ts<END)
            stats['outside_8day_window_rows']+=int((validtime&(ts>=END)).sum())
            use=window&validid
            stats['window_grid_rows']+=int(use.sum())
            safeids=mid[validid]
            rowcounts+=np.bincount(safeids,minlength=maxid+1)
            finite=validtime&validid
            np.minimum.at(first,mid[finite],ts[finite]);np.maximum.at(last,mid[finite],ts[finite])
            # File order is audited separately from the sorted calendar representation.
            dif=chunk.time_stamp.groupby(chunk.machine_id,observed=True,sort=False).diff().to_numpy()
            stats['file_order_time_reversals']+=int(np.nansum(dif<0))
            u,ii=np.unique(mid[validid],return_index=True)
            vv=ts[validid]
            stats['file_order_time_reversals']+=int(np.sum(vv[ii]<previous[u]))
            ur,ir=np.unique(mid[validid][::-1],return_index=True);previous[ur]=vv[::-1][ir]
            for j,vals in enumerate([cpu,mem]):
                good=np.isfinite(vals)&(vals>=0)&(vals<=100)
                key='cpu' if j==0 else 'memory'
                stats[key+'_invalid_rows']+=int((~good).sum())
                stats[key+'_noninteger_rows']+=int((good&(vals!=np.rint(vals))).sum())
                hist[j]+=np.bincount(np.rint(vals[good]).astype(int),minlength=101)
                bc=np.bincount(mid[validid&~good],minlength=maxid+1)
                if j==0:invalid_cpu+=bc
                else:invalid_mem+=bc
            ti=(ts[use]//STEP).astype(np.int32);ids=mid[use]
            np.add.at(seen,(ids,ti),1)
            ci=cmap[ids];keep=ci>=0
            ci=ci[keep];kt=ti[keep]
            for j,vals in enumerate([cpu[use][keep],mem[use][keep]]):
                good=np.isfinite(vals)&(vals>=0)&(vals<=100)
                np.minimum.at(low[j],(ci[good],kt[good]),vals[good])
                np.maximum.at(high[j],(ci[good],kt[good]),vals[good])
                np.logical_or.at(bad[j],(ci[~good],kt[~good]),True)
            if time.monotonic()-printed>=20:
                print(json.dumps({'rows_scanned':stats['rows'],'elapsed_s':round(time.monotonic()-start,1)}),flush=True);printed=time.monotonic()
    # Duplicate counts are exact within the stated eight-day, ten-second window.
    stats['window_unique_machine_time_points']=int(np.count_nonzero(seen))
    stats['window_duplicate_rows']=int(stats['window_grid_rows']-np.count_nonzero(seen))
    stats['window_max_multiplicity']=int(seen.max())
    if seen.max()>=250:raise ValueError('Audit counter requires a wider dtype')
    times=np.arange(T,dtype=np.int64)*STEP
    train=times<3*DAY
    observed=np.isfinite(low)&(low==high)&~bad
    measurements=np.where(observed,low,np.nan)
    complete=observed.all(axis=0)
    coverage=complete[:,train].mean(axis=1)
    selected=[];selection=[]
    for gi,pool in enumerate(groups):
        ranked=sorted(pool,key=lambda h:(-coverage[candidates.index(h)],sha_order(h)))
        selected+=ranked[:2]
        for h in pool:
            k=candidates.index(h)
            selection.append(dict(machine_id=h,source_failure_domain_2=labels[gi],layout=gi//4,domain=gi%4,
                                  training_complete_coverage=float(coverage[k]),selected=h in ranked[:2]))
    pd.DataFrame(selection).to_csv(out/'candidate_selection.csv',index=False)
    ix=[candidates.index(h) for h in selected]
    host_status=np.full((len(selected),T),-1,np.int8)
    for k,h in enumerate(selected):
        events=meta[meta.machine_id==h].sort_values('time_stamp',kind='stable')
        e=events.time_stamp.to_numpy();pos=np.searchsorted(e,times,side='right')-1
        usable=pos>=0
        host_status[k,usable]=(events.status.to_numpy()[pos[usable]]=='USING').astype(np.int8)
    np.savez_compressed(out/'alibaba2018_measured_subset.npz',time_s=times,machine_ids=np.array(selected),
                        cpu=measurements[0,ix],mem=measurements[1,ix],using=host_status,
                        failure_domain_2=np.repeat(np.array(labels),2),layout=np.repeat(np.arange(3),8),domain=np.tile(np.repeat(np.arange(4),2),3))
    hostrows=[]
    for i in range(1,maxid+1):
        if known[i] or rowcounts[i]:
            hostrows.append(dict(machine_id=f'm_{i}',rows=int(rowcounts[i]),first_time_s=float(first[i]) if np.isfinite(first[i]) else None,
              last_time_s=float(last[i]) if np.isfinite(last[i]) else None,window_unique_points=int(np.count_nonzero(seen[i])),
              window_coverage=float(np.count_nonzero(seen[i])/T),cpu_invalid_rows=int(invalid_cpu[i]),memory_invalid_rows=int(invalid_mem[i])))
    pd.DataFrame(hostrows).to_csv(out/'all_machine_measurement_audit.csv',index=False)
    pd.DataFrame({'value':np.arange(101),'cpu_count':hist[0],'memory_count':hist[1]}).to_csv(out/'utilization_histogram.csv',index=False)
    stats.update(metadata_rows=len(meta),metadata_machines=meta.machine_id.nunique(),usage_known_machines=int(np.count_nonzero(rowcounts)),
                 metadata_duplicate_machine_time_rows=int(meta.duplicated(['machine_id','time_stamp']).sum()),
                 metadata_missing_domain1_rows=int(meta.failure_domain_1.isna().sum()),metadata_missing_domain2_rows=int(meta.failure_domain_2.isna().sum()),
                 metadata_min_time_s=int(meta.time_stamp.min()),metadata_max_time_s=int(meta.time_stamp.max()),
                 metadata_records_after_analysis_end=int((meta.time_stamp>=END).sum()),initial_metadata_machines=int(initial.machine_id.nunique()),
                 metadata_changes={k:int(v) for k,v in changes.items()},metadata_status_counts=meta.status.value_counts().to_dict(),
                 machine_cpu_capacities=meta.cpu_num.unique().tolist(),normalized_mem_sizes=meta.mem_size.unique().tolist(),
                 unknown_machine_ids=sorted(unknown_ids),meta_source=meta_source,usage_source=usage_source,
                 candidate_count=n,selected_machines=selected,source_domain_labels=labels,
                 selected_train_coverage=coverage[ix].tolist(),
                 candidate_conflicting_cpu_points=int((np.isfinite(low[0])&(low[0]!=high[0])).sum()),
                 candidate_conflicting_mem_points=int((np.isfinite(low[1])&(low[1]!=high[1])).sum()),
                 elapsed_s=time.monotonic()-start,source_time_semantics='relative seconds; no calendar-date inference',
                 measurement_semantics='exact observed 10-second-grid points; no interpolation or measurement forward-fill')
    (out/'audit_metadata.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    main(a.raw,a.out)
