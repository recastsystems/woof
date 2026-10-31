#!/usr/bin/env python
"""Official f1 whole-capture CUDA executable for MPAS's inner NGW boundary.

Rust owns ORCL/NetCDF decode and ORCL encode. Candidate values are produced
only from PRE records, the original model initialization and the CUDA port.
All captured unchanged module state/presence records survive from PRE;
ilaunch is the last-column SAVE value actually returned by the GPU solver.
MPAS: bl_ugwpv1_ngw.F:120-236; cires_ugwpv1_solv2.F:249-253.
"""
import ctypes as C
import json
import os
from pathlib import Path
import sys

# Bind the executable to the supplied checkout, including a lander's trial merge.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import cupy as cp
import numpy as np
from compare_real import Transport, ptr
from woof.hex.mpasphys import ugwp_ngw_v842 as port


def main():
    dump, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True)
    manifest = json.loads((dump / 'manifest.json').read_text())
    sites = {s['site']: s for s in manifest['sites']}
    transport = Transport(Path(os.environ['MPAS_UGWP_TRANSPORT']))
    write = transport.lib.bridge_write_record
    write.argtypes = [C.c_char_p, C.c_char_p, C.c_void_p, C.c_size_t]
    write.restype = C.c_int
    initial = Path(os.environ['MPAS_UGWP_INITIAL'])
    climatology = Path(os.environ['MPAS_UGWP_CLIMATE'])
    climate_base = {n: cp.asarray(transport.netcdf(initial, n)) for n in ('rdzw', 'dzu')}
    climate_base['LATS'] = cp.asarray(transport.netcdf(climatology, 'LATS'))
    audit = []
    try:
        for prefix, capture in manifest['captures'].items():
            if sites[capture['site']]['scheme'] != 'ugwp_ngw':
                raise ValueError('whole-capture replay requires exclusively the inner NGW boundary')
            def field(name):
                return transport.oracle(dump / (prefix + '.pre.' + name + '.bin'))[1]
            if int(field('its')) != 1:
                raise ValueError('captured native lower column bound differs from the resident boundary')
            nlev, ncol = field('ugrs').shape
            if int(field('levs')) != nlev or int(field('ite')) != ncol:
                raise ValueError('captured native column/level extents differ')
            columns = {target: cp.asarray(field(source)) for source, target in (
                ('ugrs','um'), ('vgrs','vm'), ('tgrs','tm'), ('q1','qm'),
                ('prsl','prsl'), ('prslk','prslk'), ('prsi','prsi'), ('zl','zmet'),
                ('zi','zmeti'), ('raincv','raincv'), ('rainncv','rainncv'),
                ('rublten','rublten'), ('rvblten','rvblten'), ('rthblten','rthblten'))}
            reference_handle, _ = transport.oracle(dump / (prefix + '.pre.ugrs.bin'))
            ids = transport.ids(reference_handle)
            columns['lat_r'] = cp.asarray(transport.netcdf(initial, 'latCell', ids))
            climate = dict(climate_base, DAYS=cp.asarray(field('days_limb')), ABSMF=cp.asarray(field('tau_limb')))
            boundary = dict(xlatd=cp.asarray(field('xlatd')),
                indices=cp.stack([cp.asarray(field(n)) for n in ('jindx1_tau','jindx2_tau')]),
                weights=cp.stack([cp.asarray(field(n)) for n in ('ddy_j1tau','ddy_j2tau')]),
                launch_state=cp.empty(ncol, cp.int32))
            result = port.run(columns, climate, dtp=float(field('dtp')),
                day=float(field('r_doy')), amplitude=float(field('state.ugwp_wmsdis_init.tamp_mpa')),
                native_boundary=boundary)
            if cp.count_nonzero(result.status).item():
                raise ValueError('CUDA boundary returned a nonzero runtime status')
            replacements = {n: cp.asnumpy(getattr(result, n)) for n in port.TENDENCY_FIELDS}
            if bool(field('ugwp_diags')):
                replacements.update({n: cp.asnumpy(getattr(result, n)) for n in port.DIAGNOSTIC_FIELDS})
            replacements['state.ugwp_wmsdis_init.ilaunch'] = cp.asnumpy(boundary['launch_state'][-1:])
            names = [n.split('.post.',1)[1][:-4] for n in manifest['files'] if n.startswith(prefix+'.post.')]
            for name in names:
                src = dump / (prefix+'.pre.'+name+'.bin')
                dst = output / (prefix+'.post.'+name+'.bin')
                value = replacements.get(name)
                data = ptr(value) if value is not None else None
                if write(str(src).encode(), str(dst).encode(), data, value.size if value is not None else 0):
                    raise ValueError(transport.lib.bridge_error().decode())
            audit.append(dict(capture=prefix, columns=ncol, levels=nlev, records=len(names),
                generated=list(replacements), preserved_from_pre=[n for n in names if n not in replacements]))
    finally:
        transport.close()
    (output/'cuda-boundary-audit.json').write_text(json.dumps(dict(port_path=str(Path(port.__file__).resolve()), port_sha256=__import__('hashlib').sha256(Path(port.__file__).read_bytes()).hexdigest(), captures=audit),indent=2)+'\n')


if __name__ == '__main__':
    main()
