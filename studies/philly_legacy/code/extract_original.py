"""Verify the pinned archive and extract only the five expected data files."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from download_philly import SHA256,SIZE,digest

FILES={'cluster_cpu_util','cluster_gpu_util','cluster_mem_util','cluster_job_log','cluster_machine_list'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('archive');ap.add_argument('destination');args=ap.parse_args()
    archive=Path(args.archive);out=Path(args.destination);out.mkdir(parents=True,exist_ok=True)
    if archive.stat().st_size!=SIZE or digest(archive)!=SHA256:raise ValueError('Original archive checksum failed.')
    records=[]
    with tarfile.open(archive,'r:gz') as tf:
        for m in tf:
            if not m.isfile():continue
            name=Path(m.name).name
            if name not in FILES:raise ValueError(f'Unexpected member {m.name}')
            h=hashlib.sha256()
            with tf.extractfile(m) as src,(out/name).open('wb') as dst:
                for b in iter(lambda:src.read(8*1024*1024),b''):
                    h.update(b);dst.write(b)
            records.append(dict(filename=name,bytes=m.size,sha256=h.hexdigest()))
            print(name,m.size,flush=True)
    (out/'original_files.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
