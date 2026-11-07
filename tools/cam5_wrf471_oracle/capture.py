"""Bounded before/after capture of CAM driver calls in a private WRF build.

The instrumentation only writes selected columns. It does not change arguments
or arithmetic. Run with one tile per MPI rank; filenames include tile bounds.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def roster(path, names):
    tree = ast.parse((ROOT / path).read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    out[target.id] = ast.literal_eval(node.value)
    return out


def layouts():
    cu = roster('woof/core/cam5_memory.py', (
        'CAMZM_MASS_INPUTS', 'CAMZM_FULL_INPUTS', 'CAMZM_SURFACE_INPUTS',
        'CAMZM_MASS_OUTPUTS', 'CAMZM_FULL_OUTPUTS', 'CAMZM_SURFACE_OUTPUTS',
        'CAMZM_INT_OUTPUTS'))
    sh = roster('woof/core/physics_inventory.py', (
        'UWSHCU_MASS_INPUTS', 'UWSHCU_FULL_INPUTS', 'UWSHCU_SURFACE_INPUTS',
        'UWSHCU_MASS_OUTPUTS', 'UWSHCU_FULL_OUTPUTS', 'UWSHCU_SURFACE_OUTPUTS'))
    mp = roster('woof/core/cam5_memory.py', (
        'CAMMGMP_MASS_INPUTS', 'CAMMGMP_FULL_INPUTS', 'CAMMGMP_SURFACE_INPUTS',
        'CAMMGMP_INOUT_3D', 'CAMMGMP_INOUT_2D'))
    aliases = {
        'cu7': dict(t='t_phy', pi='pi_phy', u='u_phy', v='v_phy',
                    cldfra='cldfra_mp_all', tpert='tpert_camuwpbl',
                    zmmu='mu_out', zmmd='md_out', dlf='dlf_out',
                    cape='cape_out', rliq='rliq_out'),
        'shcu2': dict(u='u_phy', v='v_phy', t='t_phy', pi='pi_phy',
                      dlf_in='dlf', cmfmc_in='cmfmc', pblh='pblh_in',
                      cutop_in='cutop', cubot_in='cubot', rliq_in='rliq',
                      cush='cush_inout', cmfmc2='cmfmc2_inout',
                      evapcsh='evapcsh_inout', rprdsh='rprdsh_inout',
                      rliq2='rliq2_inout'),
        'mp11': dict(z='z_sea_level', tke='tke_pbl', kvh='kvh3d',
                     icwmrsh='icwmrsh3d', icwmrdp='icwmrdp3d',
                     shfrc='shfrc3d', cmfmc='cmfmc3d', cmfmc2='cmfmc2_3d',
                     qv='qv_curr', qc='qc_curr', qi='qi_curr', qs='qs_curr',
                     qr='qr_curr', nc='nc3d', ni='ni3d', ns='ns3d', nr='nr3d'),
    }
    result = {}
    for key, rs, pre in [('cu7', cu, 'CAMZM'), ('shcu2', sh, 'UWSHCU')]:
        row = {}
        for stage in ('in', 'out'):
            fields = []
            for shape in ('MASS', 'FULL', 'SURFACE'):
                for name in rs[f'{pre}_{shape}_{"INPUTS" if stage == "in" else "OUTPUTS"}']:
                    fields.append(dict(name=name, source=aliases[key].get(name, name),
                                       shape=shape.lower(), dtype='<f4'))
            if key == 'cu7':
                for name in (('kpbl',) if stage == 'in' else rs['CAMZM_INT_OUTPUTS']):
                    fields.append(dict(name=name, source=name, shape='surface', dtype='<i4'))
            if key == 'shcu2':
                fields.append(dict(name='cush', source='cush_inout', shape='surface', dtype='<f4'))
            row[stage] = fields
        result[key] = row
    fields = []
    for shape in ('MASS', 'FULL', 'SURFACE'):
        for name in mp[f'CAMMGMP_{shape}_INPUTS']:
            fields.append(dict(name=name, source=aliases['mp11'].get(name, name),
                               shape=shape.lower(), dtype='<f4'))
    carried = [dict(name=n, source=aliases['mp11'].get(n, n),
                    shape='mass' if dim == '3D' else 'surface', dtype='<f4')
               for dim in ('3D', '2D') for n in mp[f'CAMMGMP_INOUT_{dim}']]
    result['mp11'] = {'in': fields + carried, 'out': carried}
    return result


def block(key, fields, stage, full_step=None):
    selection = (['  if (any(itimestep == (/1, 2, 10, 60, 180/))) then',
                  '    lane_i0 = min(its+10, ite); lane_i1 = min(lane_i0+7, ite)',
                  '    lane_j0 = min(jts+10, jte); lane_j1 = min(lane_j0+1, jte)']
                 if full_step is None else
                 [f'  if (itimestep == {int(full_step)}) then',
                  '    lane_i0 = its; lane_i1 = ite',
                  '    lane_j0 = jts; lane_j1 = jte'])
    lines = selection + [
             f'    write(lane_name,\'(a,i5.5,a,i5.5,a,i5.5,a)\') "{key}-", &',
             f'      its, "-", jts, "-", itimestep, "-{stage}.bin"',
             '    open(newunit=lane_u, file=trim(lane_name), access="stream", &',
             '      form="unformatted", status="replace", action="write", convert="little_endian")',
             '    write(lane_u) 20261101, lane_i1-lane_i0+1, &',
             '      lane_j1-lane_j0+1, kte-kts+1, itimestep, its, jts, kts',
             '    write(lane_u) dt']
    if key == 'cu7':
        lines += ['    write(lane_u) dx, cudt, bl_pbl_physics, sf_sfclay_physics']
    if key == 'shcu2':
        lines += ['    write(lane_u) p_qnc, p_qni, param_first_scalar']
    if key == 'mp11':
        lines += ['    write(lane_u) accum_mode, aitken_mode, coarse_mode, &',
                  '      config_flags%cu_physics, config_flags%shcu_physics']
    for f in fields:
        if f['shape'] == 'surface':
            sl = 'lane_i0:lane_i1,lane_j0:lane_j1'
        else:
            end = 'kte+1' if f['shape'] == 'full' else 'kte'
            sl = f'lane_i0:lane_i1,kts:{end},lane_j0:lane_j1'
        lines.append(f'    write(lane_u) {f["source"]}({sl})')
    lines += ['    close(lane_u)', '  end if', '']
    return '\n'.join(lines)


def install(native, receipt, *, full_step=None):
    if full_step is not None and full_step < 1:
        raise ValueError('full-step must be positive')
    schema = layouts()
    paths = {'cu7': 'module_cu_camzm_driver.F',
             'shcu2': 'module_shcu_camuwshcu_driver.F',
             'mp11': 'module_mp_cammgmp_driver.F'}
    anchors = {'cu7': '!\n! Check to see if this is a convection timestep...',
               'shcu2': '#if ( WRF_CHEM == 1 )\n  !BSINGH:02/01/2013: Sanity',
               'mp11': '    !cmeliq is an output from Macro physics.'}
    ends = {'cu7': 'END SUBROUTINE camzm_driver',
            'shcu2': 'END SUBROUTINE camuwshcu_driver',
            'mp11': '  end subroutine CAMMGMP\n'}
    pins = json.loads((Path(__file__).parent / 'SOURCE.json').read_text())
    pins = {p['path']: p['sha256'] for p in pins['files']}
    receipts = []
    prepared = []
    for key, file in paths.items():
        p = native / 'phys' / file
        s = p.read_text()
        before = hashlib.sha256(p.read_bytes()).hexdigest()
        if before != pins['phys/' + file]:
            raise ValueError(f'{p}: source differs from public pin')
        assert s.count(anchors[key]) == 1 and s.count(ends[key]) == 1
        decl = ('  integer :: lane_u, lane_i0, lane_i1, lane_j0, lane_j1\n'
                '  character(len=160) :: lane_name\n')
        s = s.replace(anchors[key], decl + block(key, schema[key]['in'], 'in', full_step) + anchors[key], 1)
        s = s.replace(ends[key], block(key, schema[key]['out'], 'out', full_step) + ends[key], 1)
        prepared.append((p, s))
        receipts.append(dict(path=str(p), original_sha256=before,
                             capture_sha256=hashlib.sha256(s.encode()).hexdigest()))
    # Validate every pin and insertion point before changing the private tree.
    for p, s in prepared:
        p.write_text(s)
    receipt.write_text(json.dumps(dict(schema=schema, sources=receipts,
                                       full_step=full_step,
                                       physics_arithmetic_changed=False), indent=2) + '\n')


def read(path, fields):
    import numpy as np
    key = path.name.split('-')[0]
    with path.open('rb') as f:
        header = np.fromfile(f, '<i4', 8)
        if header.size != 8:
            raise ValueError('truncated capture header')
        magic, nx, ny, nz, step, its, jts, kts = map(int, header)
        if magic != 20261101:
            raise ValueError('capture magic')
        if min(nx, ny, nz, step, its, jts, kts) < 1:
            raise ValueError('capture bounds')
        meta = dict(nx=nx, ny=ny, nz=nz, step=step, its=its, jts=jts, kts=kts,
                    dt=float(np.fromfile(f, '<f4', 1)[0]))
        if key == 'cu7':
            meta.update(zip(('dx', 'cudt'), map(float, np.fromfile(f, '<f4', 2))))
            meta.update(zip(('pbl', 'sfc'), map(int, np.fromfile(f, '<i4', 2))))
        elif key == 'shcu2':
            meta.update(zip(('p_qnc', 'p_qni', 'first_scalar'), map(int, np.fromfile(f, '<i4', 3))))
        else:
            meta.update(zip(('accum', 'aitken', 'coarse'), map(float, np.fromfile(f, '<f4', 3))))
            meta.update(zip(('cu', 'shcu'), map(int, np.fromfile(f, '<i4', 2))))
        out = {}
        for field in fields:
            nlev = nz + (field['shape'] == 'full')
            shape = (nx, ny) if field['shape'] == 'surface' else (nx, nlev, ny)
            a = np.fromfile(f, field['dtype'], int(np.prod(shape)))
            a = a.reshape(shape, order='F')
            out[field['name']] = np.ascontiguousarray(a.T if a.ndim == 2 else a.transpose(1, 2, 0))
        if f.read(1):
            raise ValueError('capture trailing bytes')
    return meta, out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('native', type=Path)
    ap.add_argument('--receipt', required=True, type=Path)
    ap.add_argument('--full-step', type=int,
                    help='capture every tile column at one step for active-convection coverage')
    args = ap.parse_args()
    install(args.native, args.receipt, full_step=args.full_step)
