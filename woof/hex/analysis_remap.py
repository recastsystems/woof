"""Transfer a native hex analysis into a prepared finer hex initial state.

``python -m woof.hex.analysis_remap --source-grid ... --source-static ...
--analysis ... --target-grid ... --target-static ... --template ...
--operator ... --output ... --receipt ...``

The analysis remap as a hexcore function: the conservative horizontal map
(``woof.hex.moving_remap``), the height-conservative vertical map
(``woof.hex.moving_vertical``) and the atmosphere reconstruction on the target
Jacobian (``woof.hex.moving_state``), the same operators the moving nest uses.
It moves an analysis (or any native hex state) onto a regional or finer mesh.

The target may be a covered regional subset. This is initialization, not a
lateral-boundary generator. The target template supplies terrain, reference
profiles and land state. Every named atmospheric species comes from the
analysis. Winds use the dycore's Cartesian reconstruction and edge projection;
physical w is carried and converted with the target terrain metrics. This
function does not certify a shock-free forecast or supply LES physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np

SPECIES = ('qv', 'qc', 'qr', 'qi', 'qs', 'qg')
#: hydrometeors a cold-start init may omit; they are zero at a cold start
COLD_START_ZERO_SPECIES = frozenset(('qc', 'qr', 'qi', 'qs', 'qg'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_geometry(grid, static, init):
    from woof.hex.mesh import load_precision_preserving_mesh_pair
    from woof.hex.driver import load_mpas_vertical_grid
    from netCDF4 import Dataset
    mesh, _, _ = load_precision_preserving_mesh_pair(grid, static)
    regional = bool(getattr(mesh, 'is_regional', False))
    with Dataset(init) as d:
        counts = np.asarray(d['nEdgesOnCell'][:])
        raw_edges = np.asarray(d['edgesOnCell'][:])
        active = np.arange(raw_edges.shape[1])[None, :] < counts[:, None]
        edges = np.where(active, raw_edges - 1, -1)
        if (not np.array_equal(counts, mesh.arrays['nEdgesOnCell']) or
                not np.array_equal(edges, mesh.arrays['edgesOnCell']) or
                not np.array_equal(np.asarray(d['cellsOnEdge'][:]) - 1, mesh.arrays['cellsOnEdge'])):
            raise ValueError('init topology differs from its grid/static pair')
        for name in ('coeffs_reconstruct', 'edgeNormalVectors'):
            value = np.asarray(d[name][:])
            if value.shape != mesh.arrays[name].shape or not np.isfinite(value).all():
                raise ValueError(f'invalid initialized geometry: {name}')
            if name == 'coeffs_reconstruct':
                if np.any(np.all(value[active] == 0, axis=-1)) or np.any(value[~active] != 0):
                    raise ValueError('invalid active or padding reconstruction coefficients')
            elif not np.allclose(np.linalg.norm(value.astype(float), axis=-1), 1, rtol=0, atol=2e-6):
                raise ValueError('initialized edge normals must be unit length')
            mesh.arrays[name] = value
            mesh.variable_sources[name] = str(init)
    native = load_mpas_vertical_grid(init, mesh, allow_regional_sentinels=regional)
    return SimpleNamespace(mesh=mesh, vertical=native.vertical_grid, terrain_metrics=native.terrain_metrics)


def load(grid, static, init, species):
    from woof.hex.driver import load_mpas_initial_state
    from netCDF4 import Dataset
    with Dataset(init) as d:
        absent = sorted(set(species) - set(d.variables))
    # A cold-start init (GFS/IFS through init-source, MPAS init_atmosphere) writes vapour and at most
    # cloud/rain; the ice species start at zero in the forecast.  Only those may be absent here.
    if absent and ('qv' in absent or not set(absent) <= COLD_START_ZERO_SPECIES):
        raise ValueError(f'analysis/template lacks species: {absent}: {init}')
    present = tuple(name for name in species if name not in absent)
    result = load_geometry(grid, static, init)
    state, reference, saved = load_mpas_initial_state(
        init, result.mesh, result.vertical, scalar_names=present,
        terrain_metrics=result.terrain_metrics, return_saved_diagnostics=True,
        cold_start=True, allow_regional_sentinels=bool(result.mesh.is_regional))
    if absent:
        full = np.zeros((len(species), *state.scalars.shape[1:]), dtype=state.scalars.dtype)
        for index, name in enumerate(species):
            if name in present:
                full[index] = state.scalars[present.index(name)]
        state.scalars = full
    result.state, result.reference, result.saved_diagnostics = state, reference, saved
    result.zero_species = tuple(absent)
    return result


def real_data_reference(vertical):
    """MPAS real-data case's 250 K dry isothermal base on target terrain.

    mpas_init_atm_cases.F:5274-5282, followed by the metric coupling.
    The cold loader then recomputes exner/pressure from the stored base mass.
    """
    from woof.hex.driver import DryReferenceState
    from woof.hex.rkind_libm import powf_rkind
    f = np.float32
    z = np.asarray(vertical.zgrid, dtype=f)
    mid = f(.5) * (z[1:] + z[:-1])
    exponent = -f(9.80616) * mid / (f(287) * f(250))
    pressure = f(100000) * np.exp(exponent.astype(np.float64)).astype(f)
    exner = powf_rkind(pressure / f(100000), f(287) / f(1004.5))
    theta = f(250) / exner
    rho = (pressure / (f(287) * f(250))) / vertical.zz
    rt = rho * theta
    exner = powf_rkind(vertical.zz * f(287 / 100000) * rt, f(287 / (1004.5 - 287)))
    reference = DryReferenceState(rho, rt, vertical.zz * f(287) * exner * rt, exner)
    reference.validate(rho.shape)
    return reference, theta


def native_fields(state, saved, vertical, species):
    """Materialize both public physical fields and exact native momentum carriers."""
    if len(species) != len(set(species)) or 'qv' not in species:
        raise ValueError('unique species including qv required')
    if state.scalars.shape[0] != len(species):
        raise ValueError('species order does not match scalar block')
    dtype = state.rho.dtype
    qv = state.scalars[species.index('qv')]
    theta_m = saved.theta_m
    theta = theta_m / (dtype.type(1) + (dtype.type(461.6) / dtype.type(287)) * qv)
    fields = dict(rho=state.rho * vertical.zz, rho_zz=state.rho,
                  theta=theta, theta_m=theta_m, u=saved.normal_velocity,
                  ru=state.rho_u, w=saved.vertical_velocity, rw=state.rho_w)
    fields.update(zip(species, state.scalars))
    for name, values in fields.items():
        if not np.isfinite(values).all():
            raise ValueError(f'nonfinite transferred field: {name}')
    return fields


def stamp_analysis_time(dataset, source):
    """Keep all three MPAS clock representations on the analysis record time."""
    if 'xtime' not in source.variables:
        raise ValueError('analysis lacks xtime')
    raw = np.asarray(source['xtime'][0]).tobytes().decode('ascii').strip('\x00 ')
    from datetime import datetime
    datetime.strptime(raw, '%Y-%m-%d_%H:%M:%S')
    for name in ('xtime', 'initial_time'):
        if name in dataset.variables:
            variable = dataset[name]
            width = variable.shape[-1]
            if len(raw) > width:
                raise ValueError(f'{name} cannot hold analysis valid time')
            encoded = np.array(list(raw.ljust(width)), dtype='S1')
            if variable.ndim == 1:
                variable[:] = encoded
            else:
                variable[0] = encoded
    dataset.setncattr('config_start_time', raw)
    if 'Time' in dataset.variables:
        dataset['Time'][:] = 0
        dataset['Time'].units = 'seconds since ' + raw.replace('_', ' ')
    return raw


def same_geometry(source, target, source_cells, target_cells):
    """An identity transfer must not smooth winds by reconstructing them."""
    return (source_cells.fingerprint() == target_cells.fingerprint()
            and all(np.array_equal(getattr(source.vertical, name), getattr(target.vertical, name))
                    for name in ('zgrid', 'zz', 'dzw', 'zxu', 'fzm', 'fzp'))
            and all(np.array_equal(getattr(source.terrain_metrics, name), getattr(target.terrain_metrics, name))
                    for name in ('zb_cell', 'zb3_cell'))
            and all(np.array_equal(source.mesh.arrays[name], target.mesh.arrays[name])
                    for name in ('areaCell', 'cellsOnEdge', 'edgesOnCell', 'coeffs_reconstruct', 'edgeNormalVectors')))


def write_fields(dataset, values):
    """Write by dimension names, never by an assumed NetCDF axis order."""
    for name, value in values.items():
        entity = 'nEdges' if name in ('u', 'ru') else 'nCells'
        level = 'nVertLevelsP1' if name in ('w', 'rw') else 'nVertLevels'
        if name not in dataset.variables:
            variable = dataset.createVariable(name, value.dtype, ('Time', entity, level))
        else:
            variable = dataset[name]
        dims = tuple(d for d in variable.dimensions if d != 'Time')
        if len(dims) != 2 or set(dims) != {entity, level}:
            raise ValueError(f'{name}: unsupported dimensions {variable.dimensions}')
        data = value if dims == (level, entity) else value.T
        index = tuple(0 if d == 'Time' else slice(None) for d in variable.dimensions)
        if variable[index].shape != data.shape:
            raise ValueError(f'{name}: target shape differs from transfer')
        variable[index] = data


def _regional_target_of_global(args):
    """True for a regional target (boundary rings) remapped from a global source (none)."""
    from netCDF4 import Dataset

    def regional(path):
        with Dataset(path) as d:
            return 'bdyMaskCell' in d.variables and int(np.max(d['bdyMaskCell'][:])) > 0
    return regional(args.target_grid) and not regional(args.source_grid)


def transfer(args):
    from netCDF4 import Dataset
    from woof.hex.moving_remap import SphericalCells

    began = time.perf_counter()
    if args.operator.suffix != '.npz':
        raise ValueError('operator path must end in .npz')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    paths = [args.output.resolve(), args.receipt.resolve(), args.operator.resolve()]
    if len(set(paths)) != len(paths):
        raise ValueError('output, receipt and operator must be distinct paths')
    temporary = args.output.with_suffix(args.output.suffix + '.partial')
    if temporary.exists():
        raise FileExistsError(temporary)
    species = tuple(args.species.split(','))
    if not set(SPECIES).issubset(species) or len(set(species)) != len(species):
        raise ValueError('species must include qv,qc,qr,qi,qs,qg exactly once each')
    # THE BREAKAGE THIS PREVENTS (hex fast start, 2026-10-23): a 96,912-cell 937.5 m LAM init from the
    # 773,478-cell 80->5 km VR parent took 155.6 s, mostly on one core, loading and reconstructing the
    # whole parent.  A regional target of a global parent reads only the parent cells in a cap around
    # it (woof.hex.nest_boundary.parent_subset); the analysis frame it makes is bit-identical.
    subset = None
    if getattr(args, 'parent_subset', 'auto') == 'auto' and _regional_target_of_global(args):
        from woof.hex.nest_boundary import SUBSET_MARGIN_KM, parent_subset
        whole = {name: str(getattr(args, name)) for name in ('source_grid', 'source_static', 'analysis')}
        cut_grid, cut_static, cut_analysis, _, subset = parent_subset(
            SimpleNamespace(target_grid=args.target_grid, source_grid=args.source_grid,
                            source_static=args.source_static, analysis=args.analysis,
                            parent_subset_margin_km=getattr(args, 'parent_subset_margin_km', SUBSET_MARGIN_KM)),
            [], args.output.with_name(args.output.name + '.parent-subset'), 1)
        subset['whole_parent'] = whole
        args = SimpleNamespace(**{**vars(args), 'source_grid': cut_grid, 'source_static': cut_static,
                                  'analysis': cut_analysis})
    source = load(args.source_grid, args.source_static, args.analysis, species)
    construct = getattr(args, 'template_kind', 'initialized') == 'vertical'
    if construct:
        target = load_geometry(args.target_grid, args.target_static, args.template)
        target.reference, base_theta = real_data_reference(target.vertical)
    else:
        target = load(args.target_grid, args.target_static, args.template, species)
    sg, tg = SphericalCells.read(args.source_grid), SphericalCells.read(args.target_grid)
    identity = not construct and same_geometry(source, target, sg, tg)
    if identity:
        state, saved = source.state, source.saved_diagnostics
        receipt = dict(identity_geometry=True, momentum='native carriers copied without reconstruction')
    else:
        state, saved, receipt = remap(source, target, sg, tg, args.operator, species)
    values = native_fields(state, saved, target.vertical, species)
    if construct:
        values.update(rho_base=target.reference.rho_base, theta_base=base_theta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.template, temporary)
    with Dataset(temporary, 'r+') as d, Dataset(args.analysis) as source_file:
        write_fields(d, values)
        if construct:
            receipt['surface_transfer'] = carry_surface(source_file, d, source, target, sg, tg, args.operator, state)
        valid_time = stamp_analysis_time(d, source_file)
        retained = sorted(set(d.variables) - set(values) - {'Time', 'xtime', 'initial_time'})
        d.setncattr('da_parent_analysis_sha256', digest(args.analysis))
        d.setncattr('da_transfer_method', 'hex conservative physical-height mass and Cartesian momentum')
        d.setncattr('da_transfer_qualification', 'initialization experiment; no forecast or lateral-boundary proof')
    reloaded = load(args.target_grid, args.target_static, temporary, species)
    roundtrip = {}
    for name in ('rho', 'rho_theta', 'rho_u', 'rho_w', 'scalars'):
        before, after = getattr(state, name), getattr(reloaded.state, name)
        roundtrip[name] = dict(bitwise_equal=bool(np.array_equal(before, after)),
                               maximum_absolute_difference=float(np.max(np.abs(before.astype(float)-after))))
    temporary.replace(args.output)
    receipt.update(schema='hex.analysis-remap.v1', analysis=str(args.analysis),
                   output=str(args.output), species=list(species), valid_time=valid_time,
                   template_kind='vertical' if construct else 'initialized',
                   input_paths={name: str(getattr(args, name)) for name in
                                ('source_grid', 'source_static', 'analysis', 'target_grid', 'target_static', 'template')},
                   analysis_sha256=digest(args.analysis), output_sha256=digest(args.output),
                   operator_sha256=None if identity else digest(args.operator),
                   wall_seconds=time.perf_counter()-began, cold_load_roundtrip=roundtrip,
                   template_fields_retained=retained,
                   lateral_boundary_forcing=False, forecast_verified=False, parent_subset=subset,
                   vertical_balance=('mapped parent mass continuity and target terrain conversion' if
                       receipt.get('continuity_balance') else 'target terrain kinematic conversion; no divergence projection'))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def remap(source, target, sg, tg, operator, species):
    from woof.hex.moving_vertical import HeightConservativeMap
    from woof.hex.moving_state import remap_atmosphere
    weights = horizontal_map(source, target, sg, tg, operator)
    weights = HeightConservativeMap(weights, source.vertical.zgrid, target.vertical.zgrid,
                                   source.vertical.dzw, target.vertical.dzw,
                                   endpoint_alignment=True)
    payload = dict(state={name: getattr(source.state, name) for name in
                          ('rho', 'rho_theta', 'rho_u', 'rho_w', 'scalars')},
                   vertical=source.vertical, scalar_names=species, time_seconds=0.0)
    payload['state'].update(normal_velocity=source.saved_diagnostics.normal_velocity,
                            vertical_velocity=source.saved_diagnostics.vertical_velocity)
    return remap_atmosphere(payload, weights, source_mesh=source.mesh,
                                            target_prepared=target, config=None,
                                            balance_continuity=bool(target.mesh.is_regional))


def horizontal_map(source, target, sg, tg, operator):
    from woof.hex.moving_remap import ConservativeMap
    from woof.hex.moving_checkpoint import identify_source_geometry
    if operator.exists():
        weights = ConservativeMap.load(operator, source=sg, target=tg)
        if bool(weights.receipt.get('target_subdomain')) != bool(target.mesh.is_regional):
            raise ValueError('saved operator coverage mode differs from target mesh')
    else:
        weights = ConservativeMap.build(sg, tg, target_subdomain=bool(target.mesh.is_regional))
        operator.parent.mkdir(parents=True, exist_ok=True)
        weights.save(operator)
    radius, scale = identify_source_geometry(source.mesh, sg, target.mesh.attrs['sphere_radius'])
    _, target_scale = identify_source_geometry(target.mesh, tg, radius)
    if scale != target_scale:
        raise ValueError('source and target geometry use different sphere units')
    weights = ConservativeMap(weights.overlap * scale, weights.source_area * scale,
                              weights.target_area * scale, dict(weights.receipt))
    weights = weights.with_physical_areas(source.mesh.arrays['areaCell'], target.mesh.arrays['areaCell'],
                                          source_geometry_area=sg.area * scale,
                                          generator_areas=(sg.area, tg.area), generator_area_scale=scale)
    return weights


def atmospheric_surface_diagnostics(state, vertical):
    """EOS surface pressure from MPAS_to_physics and integrated vapor mass."""
    from woof.hex.rkind_libm import powf_rkind
    f = np.float32
    rt = state.rho_theta
    exner = powf_rkind(vertical.zz * f(287 / 100000) * rt, f(287 / (1004.5 - 287)))
    pressure = vertical.zz * f(287) * exner * rt
    moist_density = (state.rho * vertical.zz) * (f(1) + np.maximum(state.scalars[0], f(0)))
    dz = np.diff(vertical.zgrid, axis=0)
    correction = (f(.5) * (moist_density[1] - moist_density[0]) * dz[0]) / (dz[0] + dz[1])
    surface = ((f(.5) * f(9.80616)) * dz[0]) * (moist_density[0] - correction) + pressure[0]
    vapor = np.sum(state.rho.astype(np.float64) * state.scalars[0] *
                   np.asarray(vertical.dzw)[:, None], axis=0).astype(f)
    return surface, vapor


def carry_surface(source_file, target_file, source, target, sg, tg, operator, state):
    """Carry one coherent land column, retaining the target static geography."""
    from scipy.spatial import cKDTree
    from woof.hex.moving_physics import largest_donors, carry_horizontal_vector
    from woof.hex.vertical_spec import MET_STATE_VARIABLES, ZERO_STATE_VARIABLES
    weights = horizontal_map(source, target, sg, tg, operator)

    def read(dataset, name):
        variable = dataset[name]
        dimensions = tuple(d for d in variable.dimensions if d != 'Time')
        key = tuple(0 if d == 'Time' else slice(None) for d in variable.dimensions)
        value = np.asarray(variable[key])
        if not np.isfinite(value).all():
            raise ValueError(f'nonfinite parent surface field {name}')
        return np.moveaxis(value, dimensions.index('nCells'), -1), dimensions

    old_land = read(source_file, 'landmask')[0].reshape(-1) >= .5
    new_land = read(target_file, 'landmask')[0].reshape(-1) >= .5
    old_veg = read(source_file, 'ivgtyp')[0].reshape(-1)
    new_veg = read(target_file, 'ivgtyp')[0].reshape(-1)
    old_soil = read(source_file, 'isltyp')[0].reshape(-1)
    new_soil = read(target_file, 'isltyp')[0].reshape(-1)
    old_ice = int(np.asarray(source_file['isice_lu'][:]).reshape(-1)[0])
    new_ice = int(np.asarray(target_file['isice_lu'][:]).reshape(-1)[0])
    old_role = np.column_stack((old_land, old_veg == old_ice))
    new_role = np.column_stack((new_land, new_veg == new_ice))
    donors = largest_donors(weights)
    trees = {}; fallback = 0; distance = 0.
    for row in range(len(donors)):
        begin, end = weights.overlap.indptr[row:row+2]
        candidates = weights.overlap.indices[begin:end]
        compatible = np.all(old_role[candidates] == new_role[row], axis=1)
        strict = compatible & (old_veg[candidates] == new_veg[row]) & (old_soil[candidates] == new_soil[row])
        positions = np.flatnonzero(strict if strict.any() else compatible)
        if len(positions):
            selected = positions[np.argmax(weights.overlap.data[begin:end][positions])]
            donors[row] = candidates[selected]
            continue
        role = tuple(new_role[row])
        if role not in trees:
            available = np.flatnonzero(np.all(old_role == new_role[row], axis=1))
            if not len(available):
                raise ValueError(f'no parent surface column matches target land/glacier role {role}')
            trees[role] = available, cKDTree(sg.centers[available])
        available, tree = trees[role]
        chord, index = tree.query(tg.centers[row])
        donors[row] = available[index]; fallback += 1
        distance = max(distance, float(2 * np.arcsin(min(chord / 2, 1)) * target.mesh.attrs['sphere_radius']))

    def write(name, values, dimensions):
        variable = target_file[name]
        target_dims = tuple(d for d in variable.dimensions if d != 'Time')
        if target_dims != dimensions:
            raise ValueError(f'surface dimension order differs for {name}: {dimensions}, {target_dims}')
        data = np.moveaxis(values, -1, dimensions.index('nCells'))
        key = tuple(0 if d == 'Time' else slice(None) for d in variable.dimensions)
        if variable[key].shape != data.shape:
            raise ValueError(f'surface shape differs for {name}')
        variable[key] = data

    atmosphere = set(SPECIES) | {'rho', 'theta', 'rho_base', 'theta_base', 'u', 'w', 'tke', 'relhum'}
    fields = [name for name, _, _, _ in MET_STATE_VARIABLES + ZERO_STATE_VARIABLES if name not in atmosphere]
    copied = []
    for name in fields:
        if name in ('u10', 'v10', 'surface_pressure', 'precipw'):
            continue
        if name not in source_file.variables or name not in target_file.variables:
            raise ValueError(f'parent or target lacks required surface field {name}')
        value, dimensions = read(source_file, name)
        write(name, value[..., donors], dimensions)
        copied.append(name)
    u, dims = read(source_file, 'u10'); v, _ = read(source_file, 'v10')
    tu, tv = carry_horizontal_vector(u, v, weights, source.mesh, target.mesh)
    write('u10', tu, dims); write('v10', tv, dims)
    for name in ('u_init', 'v_init', 'qv_init'):
        old = np.asarray(source_file[name][:])
        if old.shape != target_file[name].shape:
            raise ValueError(f'parent and target reference profile shapes differ: {name}')
        target_file[name][:] = old
    from woof.hex.moving_vertical import HeightConservativeMap
    vertical = HeightConservativeMap(weights, source.vertical.zgrid, target.vertical.zgrid,
                                    source.vertical.dzw, target.vertical.dzw)
    for name in ('t_init', 'tke', 'relhum'):
        value, dims = read(source_file, name)
        if dims not in (('nCells', 'nVertLevels'), ('nVertLevels', 'nCells')):
            raise ValueError(f'unsupported parent diagnostic dimensions for {name}')
        mass = vertical.mass_intensive(value * source.state.rho)
        density = vertical.mass_intensive(source.state.rho)
        write(name, (mass / density).astype(value.dtype), dims)
    # A new terrain mask owns land/ocean identity, while the selected column
    # keeps its coupled soil and snow stores. Sea ice remains part of that donor.
    seaice, dims = read(target_file, 'seaice')
    write('xland', np.where(new_land | (seaice >= .5), 1., 2.).astype(np.float32), dims)
    psfc, precipw = atmospheric_surface_diagnostics(state, target.vertical)
    for name, value in (('surface_pressure', psfc), ('precipw', precipw)):
        write(name, value, ('nCells',))
    return dict(method='one parent column per target, preserving land and glacier roles; prefer overlapping soil and vegetation matches',
                coherent_fields=copied, vector_fields=['u10', 'v10'],
                nearest_compatible_columns=fallback, nearest_compatible_distance_max_m=distance,
                static_geography='target', reference_profiles='parent',
                diagnostic_columns=['t_init', 'tke', 'relhum'],
                atmospheric_surface_diagnostics=['surface_pressure', 'precipw'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-grid', 'source-static', 'analysis', 'target-grid', 'target-static',
                 'template', 'operator', 'output', 'receipt'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--species', default=','.join(SPECIES))
    p.add_argument('--template-kind', choices=('initialized', 'vertical'), default='initialized')
    p.add_argument('--parent-subset', choices=('auto', 'off'), default='auto',
                   help='a regional target of a global parent reads only the parent cells around it')
    args = p.parse_args()
    print(json.dumps(transfer(args), indent=2))


if __name__ == '__main__':
    main()
