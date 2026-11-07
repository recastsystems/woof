"""Replay native global geometry and small-step state transitions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.relaxation_wrf471_oracle.replay import grade


def read_capture(path, schema):
    raw = path.read_bytes()
    magic, nx, ny, nz = map(int, np.frombuffer(raw, '<i4', 4))
    if magic != 3114 or min(nx, ny, nz) < 1:
        raise ValueError('invalid global geometry capture')
    offset = 16

    def read(items):
        nonlocal offset
        result = {}
        sizes = dict(nx=nx, ny=ny, nz=nz)
        for name, _, expressions, transpose in items:
            shape = tuple(sizes[expr.split('+')[0]]+int(expr.split('+')[1])
                          if '+' in expr else sizes[expr] for expr in expressions)
            count = int(np.prod(shape))
            value = np.frombuffer(raw, '<f4', count, offset).reshape(shape, order='F')
            offset += 4*count
            result[name] = value.transpose(transpose) if transpose else value
        return result

    data, expected = read(schema['inputs']), read(schema['outputs'])
    if offset != len(raw):
        raise ValueError('geometry capture length mismatch')
    return (nx, ny, nz), data, expected, dict(
        path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def run(directory):
    import cupy as cp
    from woof.core.global_geometry import GlobalGeometry
    from woof.certify.kernel_manifest import kernel_manifest
    schemas = json.loads((directory/'schema.json').read_text())
    results = []
    for kind, schema in schemas.items():
        dims, data, expected, capture = read_capture(directory/(kind+'.bin'), schema)
        nx, ny, nz = dims
        profiles = np.stack([data.get(name, np.zeros(nz+1, np.float32)) for name in
            ('c1h','c2h','c1f','c2f','dnw','fnm','fnp','rdnw')])
        metric_shapes = ((ny,nx),(ny,nx),(ny,nx+1),(ny,nx+1),
                         (ny+1,nx),(ny+1,nx),(ny+1,nx))
        metrics = [data.get(name, np.ones(shape,np.float32)) for name, shape in zip(
            ('tx','ty','ux','uy','vx','vy','vxinv'), metric_shapes)]
        device = lambda value: cp.asarray(np.ascontiguousarray(value))
        a = {name: device(value) for name, value in data.items() if value.ndim > 0}
        for name, value in expected.items():
            if name not in a:
                a[name] = cp.empty(value.shape, cp.float32)
        operator = GlobalGeometry(*dims, profiles=device(profiles), metrics=tuple(map(device,metrics)),
            rdx=float(data.get('rdx',1)), rdy=float(data.get('rdy',1)),
            cfn=float(data.get('cfn',1)), cfn1=float(data.get('cfn1',0)))
        def apply(method, names, **kw):
            getattr(operator,method)(**{name:a[name] for name in names.split()}, **kw)
        if kind == 'mass_faces':
            a['mut'] = cp.empty((ny,nx),cp.float32)
            apply(kind, 'mu mub mut muu muv')
        elif kind == 'couple':
            apply(kind,'muu muv mut u v w ru rv rw')
        elif kind == 'omega':
            a['divv'] = cp.empty((nz,ny,nx),cp.float32)
            apply(kind,'mu mub u v divv ww')
        elif kind == 'prepare':
            operator.mass_faces(a['mu0'],a['mub'],a['muts'],a['muus'],a['muvs'],mode=2)
            apply('prepare_mass','mub mu0 mu mu_save muts')
            apply(kind,'muu muv mut muus muvs muts u0 v0 w0 t0 ph0 pb p alt u v w t ph us vs ws ts phs c2a')
        elif kind == 'finish':
            apply(kind,'muu muv mut muus muvs muts us vs ws ts phs mus ww1 u v w t ph mu ww')
        elif kind == 'pressure':
            apply(kind,'muu muv ph alt p pb al ru_tend rv_tend')
        elif kind == 'rhs_phi':
            apply(kind,'muu muv u v ww ph phb tendency',order=5,phi_adv_z=1)
        else:
            raise ValueError('unknown geometry capture kind')
        outputs = {}
        for name, reference in expected.items():
            actual = cp.asnumpy(a[name])
            outputs[name] = grade(actual, reference)
            if outputs[name]['different_words']:
                indices = np.argwhere(actual.view(np.uint32) != reference.view(np.uint32))
                outputs[name]['first_differences'] = [dict(index=list(map(int,index)),
                    actual=float(actual[tuple(index)]), native=float(reference[tuple(index)]))
                    for index in indices[:8]]
        results.append(dict(kind=kind,capture=capture,outputs=outputs))
    grades = [g for r in results for g in r['outputs'].values()]
    return dict(calls=results,words=sum(g['words'] for g in grades),
        different_words=sum(g['different_words'] for g in grades),
        passed=all(g['different_words']==0 for g in grades),kernel_manifest=kernel_manifest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('receipt',type=Path)
    args=parser.parse_args()
    result=run(args.directory)
    args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='kernel_manifest'},indent=2))
    raise SystemExit(0 if result['passed'] else 1)
