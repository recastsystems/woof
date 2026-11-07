"""Verify the licence and immutable public source bytes before building."""
from pathlib import Path
import hashlib,json

def verify(root=None):
    root=Path(root or Path(__file__).resolve().parent)
    manifest=json.loads((root/'sources.json').read_text())
    assert hashlib.sha256((root/'LICENSE-WRF.txt').read_bytes()).hexdigest()==manifest['wrf']['licence_sha256']
    for row in manifest['sources']:
        data=(root/'upstream'/row['path']).read_bytes()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],row['path']
    return manifest
if __name__=='__main__':
    m=verify();print('licence and',len(m['sources']),'public sources verified')
