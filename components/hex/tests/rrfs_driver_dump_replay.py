#!/usr/bin/env python3
"""Complete real physics_get_tend boundary replay, including unchanged pool records.

MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_todynamics.F:65-550
at 1d6e370e. Single-process global captures only: no halo callback is fabricated.
This proves this boundary, not uncaptured manager/interface/initialization calls.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    decoder = Path(__file__).parents[1] / 'tools/native_physics_reference/oracle/oracle_words.rs'
    bridge = Path(__file__).with_name('rrfs_driver_dump_bridge.rs')
    source = decoder.read_text().replace('//!', '//') + '\n' + bridge.read_text()
    (root / 'bridge.rs').write_text(source)
    (root / 'Makefile').write_text('all: bridge.so\nbridge.so: bridge.rs\n\trustc --edition=2021 --crate-type cdylib -O $< -o $@\n')
    (root / 'source.json').write_text(json.dumps({
        'decoder_sha256': hashlib.sha256(decoder.read_bytes()).hexdigest(),
        'bridge_sha256': hashlib.sha256(bridge.read_bytes()).hexdigest(),
        'generated_sha256': hashlib.sha256(source.encode()).hexdigest()}, indent=2) + '\n')


class Records:
    """Rust owns the typed codec, payload boundaries and exact comparator."""
    def __init__(self, library):
        self.lib = ct.CDLL(str(library.resolve()))
        byteptr = ct.POINTER(ct.c_uint8)
        self.lib.r4_record_open.argtypes = [ct.c_char_p, ct.POINTER(ct.c_int32),
            ct.POINTER(byteptr), ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_char_p)]
        self.lib.r4_record_open.restype = ct.c_void_p
        self.lib.r4_record_close.argtypes = [ct.c_void_p]
        self.lib.r4_record_text.argtypes = [ct.c_void_p, ct.POINTER(ct.c_char_p)]
        self.lib.r4_record_post.argtypes = [ct.c_char_p, ct.c_char_p, byteptr, ct.c_size_t]
        self.lib.r4_record_compare.argtypes = [ct.c_char_p, ct.c_char_p]
        self.lib.r4_record_compare.restype = ct.c_void_p
        self.lib.r4_string_free.argtypes = [ct.c_void_p]

    def load(self, path):
        header = (ct.c_int32 * 21)()
        payload = ct.POINTER(ct.c_uint8)()
        length, name = ct.c_size_t(), ct.c_char_p()
        handle = self.lib.r4_record_open(os.fsencode(path), header, ct.byref(payload),
                                         ct.byref(length), ct.byref(name))
        if not handle:
            raise ValueError(f'Rust decode refused {path}')
        try:
            if header[0] == 5:
                text = ct.c_char_p()
                if self.lib.r4_record_text(handle, ct.byref(text)):
                    raise ValueError(f'Rust character decode refused {path}')
                value = text.value.decode()
            else:
                dtype = {1: 'f4', 2: 'f8', 3: 'i4', 4: 'i4', 6: 'i8', 7: 'c8', 8: 'c16'}[header[0]]
                value = np.ctypeslib.as_array(payload, shape=(length.value,)).copy().view(dtype)
                value = value.reshape(tuple(header[3:3+header[2]]), order='F')
            return value, tuple(header)
        finally:
            self.lib.r4_record_close(handle)

    def post(self, source, target, array=None):
        # An untouched record is passed directly through Rust. Replaced words
        # come solely from PRE and actual GPU results, never expected POST.
        words = None if array is None else np.ascontiguousarray(array.ravel(order='F')).view('u1')
        payload = None if words is None else words.ctypes.data_as(ct.POINTER(ct.c_uint8))
        if self.lib.r4_record_post(os.fsencode(source), os.fsencode(target), payload,
                                    0 if words is None else words.size):
            raise ValueError(f'Rust POST emission refused {target}')

    def compare(self, expected, actual):
        pointer = self.lib.r4_record_compare(os.fsencode(expected), os.fsencode(actual))
        if not pointer:
            raise ValueError(f'Rust comparison refused {expected}')
        try:
            return json.loads(ct.string_at(pointer))
        finally:
            self.lib.r4_string_free(pointer)


def authenticate(root, prefix, manifest_path, *, source="src/core_atmosphere/physics/mpas_atmphys_todynamics.F"):
    """Authenticate the producer's complete case before loading any payload."""
    from woof.hex.mpasphys.rrfs.driver_rrfs import specification
    manifest = json.loads(manifest_path.read_text())
    spec = specification(table_key=manifest.get("table_key", "mpasphys_rrfs"))
    if (manifest.get('schema') != 'r4.real-driver-bank/v1'
            or manifest.get('model_commit') != spec['model_commit']
            or manifest.get('source_sha256', {}).get(source) != spec['source_files'][source]
            or manifest.get('process_count') != 1
            or manifest.get('input_origin') != 'native-tap'
            or not manifest.get('native_binary_sha256')):
        raise ValueError('Native producer identity/source/process count is not bound')
    members = manifest['captures'][prefix]['files']
    observed = {p.name for p in root.glob(prefix+'.*.bin')}
    if observed != set(members):
        raise ValueError('Complete native capture inventory differs from producer manifest')
    for name, sha in members.items():
        if Path(name).name != name or hashlib.sha256((root/name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Native capture member authentication failed: {name}')
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def replay(root, prefix, target, library, manifest_path):
    import cupy as cp
    from woof.hex.cuda_physics_v841 import CudaPhysicsGeometryV841, project_mpas_momentum_v842
    from woof.hex.mpasphys.rrfs.driver_rrfs import StrictPhysicsKernelCache, couple_physics_tendencies
    if not prefix.startswith('boundary__physics_get_tend.'):
        raise ValueError('This consumer implements only the complete physics_get_tend boundary')
    manifest_sha = authenticate(root, prefix, manifest_path)
    codec, cache = Records(library), StrictPhysicsKernelCache()
    pre = {p.name[len(prefix+'.pre.'):-4]: p for p in root.glob(prefix+'.pre.*.bin')}
    post = {p.name[len(prefix+'.post.'):-4]: p for p in root.glob(prefix+'.post.*.bin')}
    if not post or not set(post) <= set(pre):
        raise ValueError('Boundary output inventory is empty or lacks PRE context')
    loaded = {}
    def key(name):
        if name in pre:
            return name
        if name+'.t1' in pre:
            return name+'.t1'
        raise KeyError(f'Native field/default time level was not captured: {name}')
    def field(name):
        name = key(name)
        if name not in loaded:
            loaded[name] = codec.load(pre[name])
        return loaded[name][0]
    def scalar(name):
        return int(field(name).item())
    # A process-local replay cannot synthesize remote halo exchanges.
    if not all(codec.load(p)[1][15] == 0 for p in (pre['mesh.nCells'], pre['argument.rk_step'])):
        raise ValueError('Only native MPI rank zero is admitted')
    nc, ne = scalar('mesh.nCells'), scalar('mesh.nEdges')
    cs, es = scalar('mesh.nCellsSolve'), scalar('mesh.nEdgesSolve')
    if (nc, ne) != (cs, es):
        raise ValueError('Regional/distributed halo callbacks are not captured; global MPI1 only')
    options = {key: field('block.configs.'+key) for key in (
        'config_pbl_scheme', 'config_convection_scheme', 'config_microp_scheme',
        'config_radt_lw_scheme', 'config_radt_sw_scheme')}
    options['config_tempo_aerosolaware'] = bool(scalar('block.configs.config_tempo_aerosolaware'))
    rk, split = scalar('argument.rk_step'), scalar('argument.dynamics_substep')
    mass, mass_edge = field('state.rho_zz.t2'), field('diag.rho_edge')
    scalars = field('state.scalars.t1')
    names = ['scalar_'+str(i) for i in range(scalars.shape[0])]
    for name in ('qv','qc','qr','qi','qs','nc','ni','nifa','nwfa'):
        index = scalar('state.index_'+name)
        if index > 0:
            if index > len(names):
                raise ValueError('Native scalar index exceeds captured carrier extent')
            names[index-1] = name
    def device(array):
        return cp.asarray(np.ascontiguousarray(array))
    diag_u = device(field('diag.tend_u_phys'))
    edges = {kind: device(field('tend_physics.'+name)) for kind, name in
             (('pbl','rublten_Edge'), ('convection','rucuten_Edge'))}
    changed, projection_flags = {}, {}
    if rk == split == 1:
        geometry = CudaPhysicsGeometryV841(
            cells_on_edge=cp.ascontiguousarray(device(field('mesh.cellsOnEdge')[:, :ne]).T)-1,
            east_cell=cp.ascontiguousarray(device(field('mesh.east')).T),
            north_cell=cp.ascontiguousarray(device(field('mesh.north')).T),
            edge_normal_vectors=cp.ascontiguousarray(device(field('mesh.edgeNormalVectors')[:, :ne]).T),
            n_cells=mass.shape[1], n_edges=ne, h2d=None)
        for kind, zonal, meridional, enabled, edge_name in (
            ('pbl','rublten','rvblten', options['config_pbl_scheme'] != 'off','rublten_Edge'),
            ('convection','rucuten','rvcuten', options['config_convection_scheme'] in ('cu_tiedtke','cu_ntiedtke'),'rucuten_Edge')):
            if enabled:
                invalid = cp.zeros(1, dtype=cp.int32)
                result = project_mpas_momentum_v842(geometry=geometry,
                    zonal=device(field('tend_physics.'+zonal)),
                    meridional=device(field('tend_physics.'+meridional)),
                    invalid_flag=invalid, kernel_cache=cache)
                # Nonfinite native words remain inside the exact comparison.
                projection_flags[kind] = int(invalid.get()[0])
                edges[kind][:, :ne] = result
                changed[edge_name] = edges[kind].get()
    rates = {name: device(field('tend_physics.'+name)) for name in (
        'rthblten','rqvblten','rqcblten','rqiblten','rqsblten','rniblten','rncblten','rnifablten','rnwfablten',
        'rthcuten','rqvcuten','rqccuten','rqicuten','rqrcuten','rqscuten','rthratenlw','rthratensw')
        if any(candidate in pre for candidate in ('tend_physics.'+name,'tend_physics.'+name+'.t1'))}
    outputs = couple_physics_tendencies(options=options, kernel_cache=cache,
        mass=device(mass), mass_edge=device(mass_edge), theta_m=device(field('state.theta_m.t1')),
        scalars=device(scalars), scalar_names=names, tendencies=rates,
        pbl_edge=edges['pbl'], convection_edge=edges['convection'], tend_u_phys=diag_u,
        n_cells_solve=cs, n_edges_solve=es, n_edges=ne, rk_step=rk, dynamics_substep=split)
    changed.update({key: value.get() for key, value in outputs.items()})
    changed['tend_u_phys'] = diag_u.get()
    arguments = {'argument.tend_ru_physics':'tend_ru_physics',
                 'argument.tend_rtheta_physics':'tend_rtheta_physics',
                 'argument.tend_rho_physics':'tend_rho_physics'}
    target.mkdir(parents=True, exist_ok=False)
    comparisons, mutations = {}, []
    for name, expected in sorted(post.items()):
        leaf = name.split('.')[-1]
        if leaf.startswith('t') and leaf[1:].isdigit():
            leaf = name.split('.')[-2]
        replacement = changed.get(arguments.get(name, leaf))
        if replacement is not None:
            original, _ = codec.load(pre[name])
            if not isinstance(original, np.ndarray) or original.shape != replacement.shape:
                raise ValueError(f'Native alias has different extent: {name}')
            mutations.append(name)
        actual = target / expected.name
        codec.post(pre[name], actual, replacement)
        comparisons[name] = codec.compare(expected, actual)
    cp.cuda.get_current_stream().synchronize()
    equal = all(row.get('bitwise_equal') for row in comparisons.values())
    return {'schema':'r4.real-coupling-replay/v1', 'scope':'whole physics_get_tend, global MPI1',
        'capture':prefix, 'manifest_sha256':manifest_sha,
        'rk_step':rk, 'dynamics_substep':split, 'options':options,
        'projection_invalid_flags':projection_flags,
        'output_records':len(comparisons), 'words':sum(row['words'] for row in comparisons.values()),
        'different_words':(sum(row['different_words'] for row in comparisons.values())
                           if all(row.get('different_words') is not None for row in comparisons.values()) else None),
        'bitwise_equal':equal, 'mutated_aliases':mutations, 'comparisons':comparisons}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--prefix')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    else:
        required = ('root','prefix','out','library','manifest','receipt')
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error('replay requires '+', '.join('--'+name for name in missing))
        result = replay(args.root, args.prefix, args.out, args.library, args.manifest)
        args.receipt.write_text(json.dumps(result, indent=2)+'\n')
        raise SystemExit(0 if result['bitwise_equal'] else 1)
