"""Download and verify Microsoft's pinned original public Philly archive."""
import argparse
import hashlib
import time
import urllib.request
from pathlib import Path

COMMIT = '29a1b87fa2d9ed80b83c9e3a37f3a88d382b031d'
URL = f'https://media.githubusercontent.com/media/msr-fiddle/philly-traces/{COMMIT}/trace-data.tar.gz'
SHA256 = '2037ccf63a725a02be718b0e1aeee6142c90b7f62d0d849295120e9f6ba13d2c'
SIZE = 1055988361


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('destination')
    args = ap.parse_args()
    p = Path(args.destination)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.is_file():
        if p.stat().st_size == SIZE and digest(p) == SHA256:
            print('Existing original archive verified.', flush=True)
            return
        raise ValueError('Existing archive differs from the pinned original; refusing to overwrite.')
    part = p.with_suffix(p.suffix + '.part')
    offset = part.stat().st_size if part.exists() else 0
    if offset > SIZE:
        raise ValueError('Partial download exceeds expected size.')
    if offset < SIZE:
        # Explicit range prevents confusing a previously cached prefix with the full archive.
        req = urllib.request.Request(URL, headers={'Range': f'bytes={offset}-{SIZE-1}'})
        with urllib.request.urlopen(req, timeout=45) as src:
            if offset and src.status != 206:
                raise ValueError('Resume request did not return a partial response.')
            if src.status == 206:
                content_range = src.headers.get('Content-Range', '')
                if not content_range.startswith(f'bytes {offset}-'):
                    raise ValueError(f'Unexpected range: {content_range}')
            t = time.monotonic()
            last = t
            with part.open('ab' if offset else 'wb') as dst:
                while True:
                    b = src.read(4 * 1024 * 1024)
                    if not b:
                        break
                    dst.write(b)
                    offset += len(b)
                    now = time.monotonic()
                    if now - last >= 15:
                        print(f'{offset/1e6:.1f}/{SIZE/1e6:.1f} MB downloaded', flush=True)
                        last = now
    if part.stat().st_size != SIZE or digest(part) != SHA256:
        raise ValueError('Size or SHA-256 mismatch. Retain partial file for an audited retry.')
    part.replace(p)
    print(f'Original archive verified: {SIZE} bytes, SHA-256 {SHA256}', flush=True)


if __name__ == '__main__':
    main()
