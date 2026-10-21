"""Download the official labeled archive with resume support."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip'


def main():
    target = ROOT/'data/raw/hyper-kvasir-labeled-images.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    # curl resumes an interrupted download and fails on HTTP errors.
    
    if not target.exists() or not zipfile.is_zipfile(target):
        subprocess.run(['curl.exe', '-L', '--fail', '--retry', '3', '-C', '-', '-o', str(target), URL], check=True)
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'ZIP checksum failure: {bad}')
        members = len(archive.infolist())
    # Record a locally computed checksum for reproducibility, not publisher authentication.
    
    with target.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    record = dict(url=URL, bytes=target.stat().st_size, sha256=sha, zip_members=members, zip_crc_verified=True)
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/download_provenance.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
