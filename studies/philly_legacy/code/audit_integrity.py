"""Reproduce integrity and timing diagnostics independently of policy outcomes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_philly import scalar_job_time,START,END


def run(raw,out):
    raw=Path(raw);out=Path(out)
    inv=pd.read_csv(raw/'cluster_machine_list',skipinitialspace=True);inv.columns=inv.columns.str.strip();inv=inv.set_index('machineId')
    jobs=json.loads((raw/'cluster_job_log').read_text());maxgpu={};invalid_details=0;epochs=[];intervals={}
    for job in jobs:
        try:
            epoch=int(job['jobid'].split('_')[1]);seq=int(job['jobid'].split('_')[2])
            submitted=pd.Timestamp(job['submitted_time']).tz_localize('UTC');boot=pd.Timestamp(epoch,unit='ms',tz='UTC')
            epochs.append(dict(cluster_epoch=epoch,application_sequence=seq,submitted_as_utc=str(submitted),
                               cluster_start_utc=str(boot),delay_seconds=(submitted-boot).total_seconds()))
        except (ValueError,KeyError,TypeError):pass
        for at in job.get('attempts') or []:
            st,_=scalar_job_time(at.get('start_time'));en,_=scalar_job_time(at.get('end_time'),False)
            valid=st is not None and en is not None and en>st and st<END and en>START
            for detail in at.get('detail') or []:
                host=detail.get('ip');gg=[int(g[3:]) for g in detail.get('gpus') or [] if isinstance(g,str) and g.startswith('gpu') and g[3:].isdigit()]
                if gg:maxgpu[host]=max(maxgpu.get(host,0),max(gg)+1)
                if host in inv.index and gg and max(gg)>=inv.loc[host,'number of GPUs']:invalid_details+=1
                if valid:
                    a=max(st,START).value/1e9;b=min(en,END).value/1e9
                    for g in gg:intervals.setdefault((host,g),[]).append((a,b))
    e=pd.DataFrame(epochs)
    earliest=e.sort_values('delay_seconds').groupby('cluster_epoch',as_index=False).first()
    earliest.to_csv(out/'job_epoch_time_evidence.csv',index=False)
    overlaps=[]
    for (h,g),seq in intervals.items():
        seq.sort();end=-float('inf');over=[]
        for a,b in seq:
            v=max(0,min(end,b)-a)
            if v:over.append(v)
            end=max(end,b)
        overlaps.append(dict(machine=h,gpu=g,bounded_intervals=len(seq),overlapping_intervals=len(over),
            overlap_gt60_seconds=sum(v>60 for v in over),overlap_gt3600_seconds=sum(v>3600 for v in over),
            max_overlap_seconds=max(over,default=0)))
    ov=pd.DataFrame(overlaps);ov.to_csv(out/'bounded_job_overlap_audit.csv',index=False)
    a=pd.read_csv(out/'gpu_all_host_audit.csv').set_index('machine')
    records=[dict(machine=h,inventory_gpus=int(inv.loc[h,'number of GPUs']),max_logged_gpu_id_plus1=g,
                  mean_reported_gpu_channels=float(a.loc[h,'nonmissing_values']/a.loc[h,'rows'])) for h,g in maxgpu.items()]
    d=pd.DataFrame(records);d.to_csv(out/'inventory_job_telemetry_consistency.csv',index=False)
    # This is a diagnostic of timing assumptions, never an optimization of costs.
    z=np.load(out/'philly_measured_subset.npz',allow_pickle=False)
    u=z['gpu_util'];b=z['logged_busy'];c=z['logged_uncertain'];valid=np.isfinite(u);rows=[]
    for shift in [-480,-60,0,60,480]:
        bb=np.zeros_like(b);cc=np.zeros_like(c)
        if shift>0:bb[shift:]=b[:-shift];cc[shift:]=c[:-shift]
        elif shift<0:bb[:shift]=b[-shift:];cc[:shift]=c[-shift:]
        else:bb[:]=b;cc[:]=c
        known=valid&~cc;busy=known&bb;non=known&~bb
        rows.append(dict(job_shift_relative_to_utc_minutes=shift,known_gpu_bins=int(known.sum()),
            busy_known_bins=int(busy.sum()),mean_util_busy=float(u[busy].mean()) if busy.any() else None,
            mean_util_no_logged_job=float(u[non].mean()) if non.any() else None,
            active_gt20_no_logged_job=int((non&(u>20)).sum()),unlogged_known_gpu_bins=int(non.sum())))
    pd.DataFrame(rows).to_csv(out/'job_time_alignment_sensitivity.csv',index=False)
    totals=dict(job_hosts=len(d),hosts_with_logged_gpu_index_exceeding_inventory=int((d.max_logged_gpu_id_plus1>d.inventory_gpus).sum()),
        allocation_details_exceeding_inventory=invalid_details,
        resource_manager_epochs=len(earliest),first_submission_delays_seconds=earliest.delay_seconds.tolist(),
        utc_inference='Several sequence-0001 submissions follow ResourceManager epoch IDs by about9-12seconds, supporting UTC. Some earliest observed submissions occur later because the job log is a subset. Dataset README does not explicitly confirm the time zone.',
        bounded_gpu_intervals=int(ov.bounded_intervals.sum()),overlapping_bounded_gpu_intervals=int(ov.overlapping_intervals.sum()),
        long_overlap_gt1h_intervals=int(ov.overlap_gt3600_seconds.sum()))
    (out/'integrity_summary.json').write_text(json.dumps(totals,indent=2))
    print(json.dumps(totals,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--raw',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();run(a.raw,a.out)
