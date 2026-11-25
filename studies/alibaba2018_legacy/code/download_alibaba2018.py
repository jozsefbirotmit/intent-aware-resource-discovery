"""Fetch only the two original Alibaba-2018 machine tables; verify public hashes."""
import argparse, hashlib, json, time, urllib.request
from pathlib import Path

FILES={
 'machine_meta.tar.gz':('b5b1b786b22cd413a3674b8f2ebfb2f02fac991c95df537f363ef2797c8f6d55',92432),
 'machine_usage.tar.gz':('3e6ee87fd204bb85b9e234c5c75a5096580fdabc8f085b224033080090753a7a',1774523160),
}
BASE='https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/'

def download(root,name):
    sha,size=FILES[name];path=root/name
    if path.exists() and path.stat().st_size==size:
        h=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        if h==sha: return dict(file=name,size_bytes=size,sha256=h,url=BASE+name,verified=True)
        raise ValueError('Existing full archive hash mismatch: '+name)
    part=root/(name+'.part')
    for attempt in range(4):
        done=part.stat().st_size if part.exists() else 0
        try:
            req=urllib.request.Request(BASE+name,headers={'Range':f'bytes={done}-'} if done else {})
            with urllib.request.urlopen(req,timeout=45) as r:
                if done and (r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {done}-')):
                    raise ValueError('Server did not honor byte-range resume')
                with part.open('ab' if done else 'wb') as f:
                    last=time.monotonic()
                    while b:=r.read(4*1024*1024):
                        f.write(b);done+=len(b)
                        if time.monotonic()-last>=20:
                            print(f'{name}: {done}/{size} bytes ({100*done/size:.1f}%)',flush=True);last=time.monotonic()
            break
        except Exception as e:
            print(f'{name} attempt {attempt+1}: {type(e).__name__}: {e}',flush=True)
            if attempt==3: raise
    if part.stat().st_size!=size:raise ValueError('Archive length mismatch')
    h=hashlib.file_digest(part.open('rb'),'sha256').hexdigest()
    if h!=sha:raise ValueError('Official SHA256 mismatch')
    part.rename(path)
    return dict(file=name,size_bytes=size,sha256=h,url=BASE+name,verified=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,default=Path('raw'));ap.add_argument('--provenance',type=Path,default=Path('provenance'));args=ap.parse_args()
    args.raw.mkdir(parents=True,exist_ok=True);args.provenance.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n in FILES:
        rows.append(download(args.raw,n))
        (args.provenance/'original_archives.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(rows[-1]),flush=True)
