"""The TEMPO table programs give the native reference's words on any host.

THE BREAKAGE: tempo_table_init and tempo_table_compute called the host C library's tgammaf. glibc 2.39
(Ubuntu 24.04) returns tgammaf(4) = 6.0000005 where the native reference host (glibc 2.43) returns 6, so on a
2.39 box 268,926 of 772,327 TEMPO table words moved and a strict hrrrv5 forecast (x1.2562, 2023-03-31 12Z, 1 h)
differed from native MPAS in 1,957,713 of 7,901,197 words, on a case that is bitwise on the reference host.
Both programs now evaluate the project's own correctly rounded gamma (rust/physics-tables/src/own_gamma.rs).

CPU tier. Runs the installed (or source-built) programs exactly as FrameworkInitializationRRFS.initialize_tempo_tables
feeds them, for every hail-aware / melting-reflectivity combination, and compares the SHA-256 of their output
with the words the native reference host produces (the words the RRFS framework replay matched, 0 different).
"""
import hashlib
import json
import math
import subprocess

import numpy as np
import pytest

from woof.hex.mpasphys.rrfs import framework_init_rrfs as fi
from woof.hex.mpasphys.rrfs.framework_rrfs import DATA

# SHA-256 of the programs' stdout on the native reference host (glibc 2.43), per (hailaware, melt).
INIT_SHA256 = {
    (False, False): 'd2417cea1316bdc04178749221f1d3c197bc452d81cf66127485f8765b18b217',
    (False, True): '24dcb36f90a568e948a67f2eda2e9700a388d389210f5d509c6f2d96f2f69569',
    (True, False): '42a47f28b7351e96b724f46e443d13cbe01cb15d8bb58d9035607b09781cb379',
    (True, True): '323c73d42cadb95d62986b595dded212b9153ee0ea3d614438892b5512947d35',
}
COMPUTE_SHA256 = 'db16d024461265ec706d7510373ea133f04e0ce8743d7a29951031b0a2870b66'


def _tools():
    from woof.hex.mpasphys.resources_build import PhysicsResourceError, default_cache, locate_decoders
    try:
        return locate_decoders(default_cache())
    except PhysicsResourceError as error:
        pytest.skip(str(error))


def _constants():
    tables = {}
    for name, row in json.loads((DATA / 'metadata.json').read_text())['tempo_constants'].items():
        words = np.array(row['words'], np.uint64 if row['dtype'] == 'float64' else np.uint32)
        value = words.view(np.dtype(row['dtype']).newbyteorder('<')).astype(row['dtype'])
        tables[name] = value.reshape(row['shape']) if row['shape'] else value.reshape(())
    return tables


@pytest.mark.parametrize('hail,melt', sorted(INIT_SHA256))
def test_tempo_tables_are_the_reference_words_on_this_host(hail, melt):
    tools, tables = _tools(), _constants()
    flags = np.array([hail, melt, True], '<i4')
    parts = [flags.tobytes()]
    parts += [np.float32(tables[name]).tobytes() for name in fi.TEMPO_CONSTANT_SCALARS]
    parts += [np.asarray(tables[name], np.float32).tobytes(order='F') for name in fi.TEMPO_CONSTANT_ARRAYS]
    raw = subprocess.run([str(tools['tempo_table_init'])], input=b''.join(parts), check=True,
                         capture_output=True).stdout
    offset = 0
    for name, dtype, shape in (*fi.TEMPO_DERIVED, *fi.TEMPO_HAIL_BINS, *(fi.TEMPO_RADAR_BINS if melt else ())):
        count = math.prod(shape) if shape else 1
        value = np.frombuffer(raw, dtype=dtype, count=count, offset=offset)
        offset += count * value.itemsize
        tables[name] = value.reshape(shape, order='F') if shape else value.reshape(())
    assert offset == len(raw)
    # The word that showed the defect: gamma(4) enters cig(2) (and ccg, crg, cgg) as exactly 6.
    assert tables['cig'][1] == np.float32(6.0)
    assert hashlib.sha256(raw).hexdigest() == INIT_SHA256[(hail, melt)]

    scalars, f32, f64 = fi.FrameworkInitializationRRFS.TEMPO_COMPUTE_INPUTS
    parts = [np.float32(tables[name]).tobytes() for name in scalars]
    parts += [np.asarray(tables[name], np.float32).tobytes(order='F') for name in f32]
    parts += [np.asarray(tables[name], np.float64).tobytes(order='F') for name in f64]
    raw = subprocess.run([str(tools['tempo_table_compute'])], input=b''.join(parts), check=True,
                         capture_output=True).stdout
    assert hashlib.sha256(raw).hexdigest() == COMPUTE_SHA256
