"""Device uploads of Rust-seeded radiation table words, bound to provenance."""
from pathlib import Path
import gzip,hashlib,json
from woof.core.device_cache import cuda_cache

TABLE_ROOT=Path(__file__).resolve().parents[1]/'data/wrf_radiation'
TABLES={'goddard-cloud':{'folder':'goddard','file':'cloud-table.le.bin.gz',
 'source_sha256':'ad4bb1dd8446a2b00559ac425e21255a01b79a4cac7b5e3e693a3c98c222a1c8',
 'words':3*2*11*6*109}}

@cuda_cache(maxsize=None)
def radiation_table_words(name):
    import numpy as np
    import cupy as cp
    spec=TABLES[name]
    root=TABLE_ROOT/spec['folder']
    from woof.tables import table_path
    provenance=json.loads(table_path('goddard', root/'provenance.json').read_text())
    if provenance['source_sha256']!=spec['source_sha256']:
        raise ValueError(f'{name}: cloud optics source hash changed; extinction and scattering would use another table')
    raw=gzip.decompress(table_path('goddard', root/spec['file']).read_bytes())
    if len(raw)!=4*spec['words'] or hashlib.sha256(raw).hexdigest()!=provenance['decoded_sha256']:
        raise ValueError(f'{name}: cloud optics table is truncated or changed; optical depths would be corrupted')
    # Rust has already decoded the source. This is a word-preserving device upload.
    return cp.asarray(np.frombuffer(raw,dtype='<f4'))
