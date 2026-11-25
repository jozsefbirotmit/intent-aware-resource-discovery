"""Audit original Philly logs and preserve a deterministic telemetry subset.

GPU/memory observations retain missing values. Job intervals are public-subset
occupancy, never a complete scheduler ground truth. Anonymous machine IDs carry
no rack or federation topology.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TZ = 'America/Los_Angeles'
START = pd.Timestamp('2017-10-03 00:00', tz=TZ).tz_convert('UTC')
END = pd.Timestamp('2017-12-16 00:00', tz=TZ).tz_convert('UTC')
SPLIT = pd.Timestamp('2017-11-01 00:00', tz=TZ).tz_convert('UTC')
GRID = pd.date_range(START, END, freq='min', inclusive='left')
LAGS = [1, 5, 15, 60]


def explicit_minutes(series):
    """Respect the explicit PDT/PST suffix; do not collapse the repeated DST hour."""
    offsets = series.str[-3:].map({'PDT': 7, 'PST': 8})
    naive = pd.to_datetime(series.str[:19], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    utc = naive + pd.to_timedelta(offsets, unit='h')
    values = utc.astype('int64').to_numpy() // 60_000_000_000
    valid = utc.notna().to_numpy() & offsets.notna().to_numpy()
    return values, valid


def scalar_job_time(value, start=True, tz='UTC'):
    if value is None or value == '':
        return None, False
    try:
        t = pd.Timestamp(value)
    except (ValueError,TypeError):
        return None, False
    if pd.isna(t):
        return None, False
    if t.tzinfo is not None:
        return t.tz_convert('UTC'), False
    a = t.tz_localize(tz, ambiguous=True, nonexistent='NaT')
    b = t.tz_localize(tz, ambiguous=False, nonexistent='NaT')
    if pd.isna(a) or pd.isna(b):
        return None, False
    # For a repeated DST hour, cover the union of possible intervals.
    return (a if start else b).tz_convert('UTC'), a != b


def interval_bins(start, end, grid=GRID):
    """Bins [t,t+1min) intersect [start,end); half-open, no future leakage."""
    a = grid[0] if start is None else start
    b = grid[-1] + pd.Timedelta(minutes=1) if end is None else end
    if b <= a:
        return None
    lo = max(0, int((a.value - grid[0].value) // 60_000_000_000))
    hi = min(len(grid), int(np.ceil((b.value - grid[0].value) / 60_000_000_000)))
    return (lo, hi) if lo < hi else None


def audit_csv(path, kind, candidates, out):
    n = len(GRID)
    mapping = {m:i for i,m in enumerate(candidates)}
    # These are intermediate audits of measured signals, never generated workloads.
    arr = np.full((n, len(candidates), 8 if kind=='gpu' else 2), np.nan, dtype=np.float32)
    stats = {}
    duplicates = conflicts = invalid_times = outside = 0
    cols = list(range(10)) if kind == 'gpu' else list(range(4))
    dtype = {c:'float64' for c in ([f'gpu{i}_util' for i in range(8)] if kind=='gpu' else ['mem_total','mem_free'])}
    for ci, chunk in enumerate(pd.read_csv(path, usecols=cols, index_col=False, dtype=dtype, chunksize=250000)):
        name = 'machineId' if kind=='gpu' else 'machine_id'
        raw = chunk.iloc[:,2:].to_numpy(float)
        if kind == 'gpu':
            good_value = np.isfinite(raw) & (raw >= 0) & (raw <= 100)
        else:
            good_value = np.isfinite(raw).all(axis=1) & (raw[:,0] > 0) & (raw[:,1] >= 0) & (raw[:,1] <= raw[:,0])
        for host, inds in chunk.groupby(name, sort=False).indices.items():
            sub = chunk.iloc[inds]
            st = stats.setdefault(host, {'rows':0,'first':None,'last':None,'nonmissing_values':0})
            st['rows'] += len(sub)
            st['first'] = min(st['first'] or sub.time.min(), sub.time.min())
            st['last'] = max(st['last'] or sub.time.max(), sub.time.max())
            st['nonmissing_values'] += int(good_value[inds].sum())
        selected = chunk[name].isin(mapping).to_numpy()
        if selected.any():
            sub = chunk.loc[selected]
            mins, valid = explicit_minutes(sub.time)
            ix = mins - GRID[0].value // 60_000_000_000
            invalid_times += int((~valid).sum())
            inrange = valid & (ix >= 0) & (ix < n)
            outside += int((valid & ~inrange).sum())
            ii = ix[inrange].astype(int)
            jj = sub[name].map(mapping).to_numpy()[inrange].astype(int)
            vals = sub.iloc[:,2:].to_numpy(float)[inrange]
            if kind == 'gpu':
                vals[(vals < 0) | (vals > 100)] = np.nan
            else:
                ok = np.isfinite(vals).all(axis=1) & (vals[:,0] > 0) & (vals[:,1] >= 0) & (vals[:,1] <= vals[:,0])
                vals[~ok] = np.nan
            # Refuse conflicting duplicate keys, including duplicates within a chunk.
            keys = ii * len(candidates) + jj
            if len(np.unique(keys)) != len(keys):
                raise ValueError(f'Duplicate {kind} keys inside a chunk: audit before proceeding.')
            old = arr[ii,jj].astype(float)
            existing = np.isfinite(old).any(axis=1)
            duplicates += int(existing.sum())
            if existing.any():
                equal = np.isclose(old[existing],vals[existing],equal_nan=True,rtol=1e-6,atol=1e-6).all(axis=1)
                conflicts += int((~equal).sum())
                if not equal.all():
                    raise ValueError(f'Conflicting duplicate {kind} values.')
            arr[ii,jj] = vals
        if ci % 12 == 0:
            print(kind, 'processed', sum(x['rows'] for x in stats.values()), 'rows', flush=True)
    pd.DataFrame([dict(machine=k,**v) for k,v in stats.items()]).to_csv(out/f'{kind}_all_host_audit.csv',index=False)
    return arr, dict(rows=sum(s['rows'] for s in stats.values()),hosts=len(stats),selected_duplicate_keys=duplicates,
                    selected_conflicting_keys=conflicts,selected_invalid_times=invalid_times,selected_outside_window=outside)


def jobs_occupancy(path, selected, out, job_timezone='UTC'):
    jobs = json.loads(path.read_text())
    mapping = {m:i for i,m in enumerate(selected)}
    diff = np.zeros((len(GRID)+1,len(selected),8),dtype=np.int16)
    uncertain_diff = np.zeros_like(diff)
    count = dict(jobs=len(jobs),attempts=0,empty_detail=0,missing_start=0,missing_end=0,
                 ambiguous_start=0,ambiguous_end=0,reversed_intervals=0,invalid_gpu_ids=0,
                 details=0,selected_gpu_intervals=0)
    hosts = set()
    intervals = []
    for job in jobs:
        for at in job.get('attempts') or []:
            count['attempts'] += 1
            st, ambs = scalar_job_time(at.get('start_time'),True,job_timezone)
            en, ambe = scalar_job_time(at.get('end_time'),False,job_timezone)
            count['missing_start'] += st is None
            count['missing_end'] += en is None
            count['ambiguous_start'] += ambs
            count['ambiguous_end'] += ambe
            detail = at.get('detail') or []
            count['empty_detail'] += not detail
            uncertain = st is None or en is None
            if st is not None and en is not None and en <= st:
                count['reversed_intervals'] += 1
                uncertain = True
                # Malformed endpoints give uncertain, not confirmed occupancy.
                st,en = min(st,en),max(st,en)+pd.Timedelta(minutes=1)
            bins = interval_bins(st,en)
            for entry in detail:
                count['details'] += 1
                host = entry.get('ip');hosts.add(host)
                if host not in mapping or bins is None:
                    continue
                lo,hi = bins;j=mapping[host]
                for gpu in entry.get('gpus') or []:
                    if not isinstance(gpu,str) or not gpu.startswith('gpu') or not gpu[3:].isdigit() or not 0<=int(gpu[3:])<8:
                        count['invalid_gpu_ids'] += 1;continue
                    g = int(gpu[3:])
                    target = uncertain_diff if uncertain else diff
                    target[lo,j,g] += 1;target[hi,j,g] -= 1
                    count['selected_gpu_intervals'] += 1
                    intervals.append(dict(machine=host,gpu=g,start_utc=str(st) if st is not None else '',
                        end_utc=str(en) if en is not None else '',start_bin=lo,end_bin=hi,uncertain=uncertain))
    active = np.cumsum(diff[:-1],axis=0,dtype=np.int32)
    assert active.min() >= 0
    count['max_simultaneous_logged_jobs_per_gpu_bin'] = int(active.max())
    count['gpu_bins_with_overlapping_logged_jobs'] = int((active>1).sum())
    uncertain_active = np.cumsum(uncertain_diff[:-1],axis=0,dtype=np.int32)>0
    count['gpu_bins_with_uncertain_occupancy'] = int(uncertain_active.sum())
    count['job_hosts'] = len(hosts)
    count['selected_hosts_with_any_job_detail'] = len(set(selected)&hosts)
    count['job_time_zone'] = job_timezone
    count['job_time_zone_status'] = 'UTC inferred from application ID ResourceManager epochs and first submission delays; not explicitly specified by the dataset README.'
    count['occupancy_semantics'] = 'Bounded valid intervals give confirmed public-subset occupancy. Missing/reversed endpoints give uncertain occupancy, censored to window edges, never proof of idle or active. DST ambiguity covered conservatively.'
    pd.DataFrame(intervals).to_csv(out/'selected_logged_gpu_intervals.csv.gz',index=False)
    return active > 0, uncertain_active, count


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--job-timezone',default='UTC',choices=['UTC','America/Los_Angeles'])
    args=ap.parse_args();raw=Path(args.raw);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    inv=pd.read_csv(raw/'cluster_machine_list',skipinitialspace=True);inv.columns=inv.columns.str.strip()
    # The released inventory GPU counts contradict the measured channels and jobs
    # (repository issue #8). Use its identifier set only, NEVER its capacity labels.
    eligible=inv.machineId.tolist()
    candidates=sorted(eligible,key=lambda m:hashlib.sha256(('philly-audit-2026|'+m).encode()).hexdigest())[:96]
    cache=raw/'candidate_measurements_v2.npz'
    auditcache=raw/'candidate_audit_v2.json'
    if cache.exists() and auditcache.exists():
        zz=np.load(cache,allow_pickle=False)
        if zz['machine'].tolist()!=candidates:raise ValueError('Candidate cache differs from fixed selection.')
        gpu=zz['gpu'];mem=zz['mem'];cm=json.loads(auditcache.read_text());gmeta=cm['gpu'];mmeta=cm['memory']
    else:
        gpu,gmeta=audit_csv(raw/'cluster_gpu_util','gpu',candidates,out)
        mem,mmeta=audit_csv(raw/'cluster_mem_util','mem',candidates,out)
        np.savez_compressed(cache,machine=np.array(candidates),gpu=gpu,mem=mem)
        auditcache.write_text(json.dumps(dict(gpu=gmeta,memory=mmeta)))
    train=GRID<SPLIT
    usable=np.isfinite(gpu).all(axis=2)&np.isfinite(mem).all(axis=2)
    cover=usable[train].mean(axis=0)
    # Select by TRAINING coverage only, then fixed hash order for deterministic ties.
    order=sorted(range(len(candidates)),key=lambda i:(-cover[i],i))
    take=order[:24];selected=[candidates[i] for i in take]
    gpu=gpu[:,take];mem=mem[:,take]
    busy,uncertain,jmeta=jobs_occupancy(raw/'cluster_job_log',selected,out,args.job_timezone)
    known=np.isfinite(gpu)
    mem_ok=np.isfinite(mem).all(axis=2)&(mem[:,:,0]>0)
    free_fraction=np.divide(mem[:,:,1],mem[:,:,0],out=np.full(mem.shape[:2],np.nan),where=mem_ok)
    node_meta=[]
    for i,host in enumerate(selected):
        gv=known[:,i];u=gpu[:,i];b=busy[:,i];active=gv&(u>20)
        row=dict(machine=host,train_allgpu_mem_coverage=float(cover[take[i]]),
                 test_allgpu_mem_coverage=float((gv.all(axis=1)&mem_ok[:,i])[~train].mean()),
                 measured_gpu_bins=int(gv.sum()),logged_busy_known_gpu_bins=int((gv&b).sum()),
                 active_gt20_gpu_bins=int(active.sum()),active_without_logged_job=int((active&~b).sum()),
                 low_le20_but_logged_busy=int((gv&(u<=20)&b).sum()),
                 gpu_bins_with_uncertain_job= int((gv&uncertain[:,i]).sum()),
                 busy_mean_util=float(u[gv&b].mean()) if (gv&b).any() else None,
                 unlogged_mean_util=float(u[gv&~b].mean()) if (gv&~b).any() else None)
        node_meta.append(row)
    pd.DataFrame(node_meta).to_csv(out/'selected_host_join_audit.csv',index=False)
    pd.DataFrame(dict(machine=candidates,training_complete_coverage=cover,
                      selected=[m in selected for m in candidates])).to_csv(out/'candidate_selection.csv',index=False)
    np.savez_compressed(out/'philly_measured_subset.npz',
        bin_start_unix_minute=np.array(GRID.asi8//60_000_000_000,dtype=np.int64),
        machine=np.array(selected),gpu_util=gpu,host_mem_total=mem[:,:,0],host_mem_free=mem[:,:,1],
        host_mem_free_fraction=free_fraction.astype(np.float32),logged_busy=busy,logged_uncertain=uncertain)
    meta=dict(inventory_hosts=len(inv),inventory_gpus=int(inv['number of GPUs'].sum()),
        inventory_classes=inv.groupby(['number of GPUs','single GPU mem']).size().reset_index(name='hosts').to_dict('records'),
        gpu=gmeta,memory=mmeta,jobs=jmeta,grid_start_utc=str(GRID[0]),grid_last_bin_start_utc=str(GRID[-1]),
        grid_bins=len(GRID),split_utc=str(SPLIT),candidate_hosts=len(candidates),selected_hosts=selected,
        training_bins=int(train.sum()),test_bins=int((~train).sum()),
        mean_selected_train_coverage=float(usable[train][:,take].mean()),
        mean_selected_test_coverage=float(usable[~train][:,take].mean()),
        inventory_gpu_counts_used=False,
        inventory_warning='Published inventory GPU count/memory classes are inconsistent with telemetry and job allocation indices (issue #8). Only host identifiers are used; eight-channel eligibility selected by training observations.',
        timestamp_semantics='Source minute labelled t is made available at t+1 minute; no forward fill. Job overlap conservatively uses [t,t+1).',
        memory_units='No absolute host-memory unit is assumed; intent uses mem_free/mem_total.',
        warning='Low GPU utilization and absence of a logged job do NOT establish scheduler allocatability. Dynamic GPU VRAM is absent.')
    (out/'audit_metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False))
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':main()
