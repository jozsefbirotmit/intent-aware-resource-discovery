"""Portable driver. All mutable outputs are written to work/, not archived data."""
import argparse,os,sys,shutil,subprocess,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_NUM_THREADS='1',MPLBACKEND='Agg')
def run(cwd,*args):subprocess.run([sys.executable,*args],cwd=cwd,env=ENV,check=True)
def clone(name,work):
    dst=work/name
    if dst.exists():raise SystemExit(f'{dst} already exists; choose a fresh --work directory')
    shutil.copytree(ROOT/'studies'/name,dst);return dst
def stage_alibaba(d,source):
    source=source.resolve()
    legacy=ROOT/'studies/alibaba2018_legacy/results/alibaba2018'
    for name in ('policy_summary.csv','brier_scores.csv'):
        run(ROOT,'scripts/compare.py','csv',str(legacy/name),str(source/name))
    target=d/'input/alibaba2018';target.mkdir(parents=True,exist_ok=True)
    names=('calibration_budget_grid.csv','calibration_choices_frozen_before_test.json','domain_states.npz','experiment_metadata.json','fitted_models.json','policy_summary.csv')
    for name in names:shutil.copy2(source/name,target/name)
    path=d/'provenance/frozen_input_sources.json'
    original=json.loads(path.read_text())
    shutil.copy2(path,path.with_name('original_frozen_input_sources.json'))
    current=[]
    for record in original:
        if record['file'].startswith('input/alibaba2018/'):
            file=d/record['file'];current.append(dict(file=record['file'],size_bytes=file.stat().st_size,sha256=hashlib.file_digest(file.open('rb'),'sha256').hexdigest()))
        else:current.append(record)
    path.write_text(json.dumps(current,indent=2)+'\n')
    (d/'provenance/regenerated_alibaba.json').write_text(json.dumps(dict(source='locally regenerated or locally retained results; original hashes preserved separately',legacy_summary_comparison='passed',byte_identity_to_original=[x['file'] for x,y in zip(current,original) if x['file'].startswith('input/alibaba2018/') and x==y]),indent=2)+'\n')
p=argparse.ArgumentParser();p.add_argument('mode',choices=['quick','benchmark','telemetry','telemetry-philly','mechanisms','refit','alibaba-prepare','stage-alibaba','figures']);p.add_argument('--work',type=Path,default=ROOT/'work');p.add_argument('--alibaba-results',type=Path,help='Local directory containing the regenerated Alibaba legacy results');p.add_argument('--alibaba-raw',type=Path,help='Local directory containing the two original Alibaba archives');a=p.parse_args();w=a.work.resolve();w.mkdir(parents=True,exist_ok=True)
if a.mode in ('telemetry','stage-alibaba') and a.alibaba_results is None:p.error('Provide --alibaba-results from a locally regenerated legacy study (see docs/DATA.md), or use telemetry-philly.')
if a.mode=='alibaba-prepare' and a.alibaba_raw is None:p.error('Provide --alibaba-raw with the original data archives obtained under the provider terms.')
if a.mode=='quick':
    run(ROOT,'scripts/paper_results.py','--out',str(w/'paper_results'))
    run(ROOT/'studies/mechanisms','-m','unittest','test_mechanisms','-v')
    run(ROOT/'studies/final_psa1s_replay','-m','unittest','discover','-s','code','-p','test_final_replay.py','-v')
elif a.mode=='benchmark':
    d=clone('final_benchmark',w);run(d,'benchmark.py','10');run(d,'summarize.py');run(ROOT,'scripts/compare.py','benchmark',str(ROOT/'studies/final_benchmark'),str(d))
elif a.mode=='telemetry':
    d=clone('final_psa1s_replay',w);stage_alibaba(d,a.alibaba_results);run(d,'code/replay_philly.py');run(d,'code/replay_alibaba2018.py');run(d,'-m','unittest','discover','-s','code','-p','test_final_replay.py','-v');run(d,'code/build_replay_report.py');run(d,'code/validate_replay_outputs.py');run(ROOT,'scripts/compare.py','telemetry',str(ROOT/'studies/final_psa1s_replay'),str(d))
elif a.mode=='telemetry-philly':
    d=clone('final_psa1s_replay',w);run(d,'code/replay_philly.py');run(ROOT,'scripts/compare.py','csv',str(ROOT/'studies/final_psa1s_replay/results/philly/policy_summary.csv'),str(d/'results/philly/policy_summary.csv'))
elif a.mode=='stage-alibaba':
    d=clone('final_psa1s_replay',w);stage_alibaba(d,a.alibaba_results)
elif a.mode=='alibaba-prepare':
    d=clone('alibaba2018_legacy',w);run(d,'code/prepare_alibaba2018.py','--raw',str(a.alibaba_raw.resolve()),'--out','results/alibaba2018');run(d,'code/alibaba_replay.py');
    for name in ('policy_summary.csv','brier_scores.csv'):run(ROOT,'scripts/compare.py','csv',str(ROOT/'studies/alibaba2018_legacy/results/alibaba2018'/name),str(d/'results/alibaba2018'/name))
elif a.mode=='mechanisms':
    d=clone('mechanisms',w);run(d,'mechanism_study.py','--out','rerun','--bootstrap','1500','--no-scale');run(d,'-c',"from mechanism_study import run_factorial;run_factorial('rerun')");run(ROOT,'scripts/compare.py','mechanisms',str(d/'results'),str(d/'rerun'))
elif a.mode=='refit':
    d=clone('philly_legacy',w);run(d,'code/philly_replay.py','--input','results/philly_measured_subset.npz','--out','rerun_results_telemetry_full','--telemetry-only','--bootstrap','1000');run(ROOT,'scripts/check_refit.py','philly',str(d/'results_telemetry_full'),str(d/'rerun_results_telemetry_full'))
    print('Philly refit complete. For Alibaba refit from original archives, use alibaba-prepare.')
elif a.mode=='figures':run(ROOT,'scripts/paper_results.py','--out',str(w/'paper_results'))
