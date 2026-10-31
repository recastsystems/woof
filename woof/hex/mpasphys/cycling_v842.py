"""MPAS-native pools at the CUDA JEDI cycling boundary.

The existing JEDI contract owns CDF-5 I/O and coupled-state initialization.
No WOOF physics state or soil interpolation is used on this route.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .. import jedi_contract as contract


def background_fields(session, layout, names):
    """Copy the native physical fields, with selected suite soil/species axes."""
    rows = {}
    for row in layout.fields:
        if row['name'] in rows:
            raise ValueError(f"ambiguous Registry field {row['name']}")
        rows[row['name']] = row
    fields, provenance = {}, {}
    required = tuple(name for name in contract.BACKGROUND_FIELDS if name != 'scalars')
    for name in (*required, 'scalars'):
        if name not in rows:
            raise ValueError(f'native JEDI background requires active Registry field {name}')
        row = rows[name]
        value = np.asarray(session.native.get_field(row['pool'], row['pool_name']))
        # Remove native garbage storage, along the Registry entity axis.
        for axis, dim in enumerate(row['dimensions']):
            if dim in ('nCells', 'nEdges', 'nVertices'):
                value = np.take(value, np.arange(layout.dimensions[dim]), axis=axis)
        if value.dtype != np.float32 or not np.all(np.isfinite(value)):
            raise ValueError(f'native JEDI background {name} has invalid RKIND or non-finite words')
        if name == 'scalars':
            if tuple(row['components']) != tuple(names):
                raise ValueError('native JEDI scalar component order differs')
            for index, species in enumerate(names):
                fields[species] = np.ascontiguousarray(value[index])
                provenance[species] = f"native MPAS pool {row['pool']}.{row['pool_name']}[{species}]"
        else:
            fields[name] = np.ascontiguousarray(value)
            provenance[name] = f"native MPAS pool {row['pool']}.{row['pool_name']}"
    soil = layout.dimensions['nSoilLevels']
    label = contract.NATIVE_SOIL_LABEL.format(levels=soil)
    for name in ('smois', 'tslb'):
        provenance[name] += '; ' + label
    return fields, provenance


def write_background(arguments, session, layout, names, *, step, dt, out):
    """Write one requested boundary through the existing Rust CDF-5 writer."""
    fields, provenance = background_fields(session, layout, names)
    native_soil = layout.dimensions['nSoilLevels']
    selected_soil = getattr(arguments, 'jedi_soil_levels', None)
    if selected_soil is not None and int(selected_soil) != native_soil:
        raise ValueError(f'native JEDI uses the selected suite soil column ({native_soil} levels); '
                         f'--jedi-soil-levels {selected_soil} differs')
    valid = (datetime.strptime(layout.configs['config_start_time'], '%Y-%m-%d_%H:%M:%S')
             + timedelta(seconds=step * dt))
    seconds = int(step * dt)
    if seconds != step * dt:
        raise ValueError('JEDI boundary must have an integral-second timestamp')
    label = f'f{seconds // 3600:03d}' if not seconds % 3600 else f't{seconds:08d}'
    return contract.write_background(Path(out) / f'mpasout.{label}.nc',
        fields=fields, provenance=provenance, valid=valid,
        n_cells=layout.dimensions['nCells'], n_edges=layout.dimensions['nEdges'],
        n_levels=layout.dimensions['nVertLevels'], soil_levels=native_soil,
        scalar_names=names, n_vertices=layout.dimensions['nVertices'],
        identity_path=Path(arguments.init),
        global_attributes={'hex_physics_route': 'MPAS native physics',
                           'source': 'woof hex: MPAS CUDA dynamics and MPAS native physics'})


def _changed_elements(analysis, background, names, mesh, n_cells, n_edges):
    """(cells, edges) where an analysis word differs from the background it was made from, bit for bit.

    A cell changes when any of its rho, theta, w or analysed species words changed; an edge when its u changed or
    either of its cells did (rho_u = u times the edge's mean rho_zz).
    """
    def differs(name, count):
        a, b = np.asarray(analysis[name], np.float32), np.asarray(background[name], np.float32)
        if a.shape != b.shape or a.shape[-1] != count:
            raise ValueError(f'analysis {name} {a.shape} and its background {b.shape} differ in shape')
        flags = a.view(np.uint32) != b.view(np.uint32)
        return flags.reshape(-1, count).any(axis=0)
    cells = np.zeros(n_cells, bool)
    for name in ('rho', 'theta', 'w', *names):
        cells |= differs(name, n_cells)
    edges = differs('u', n_edges)
    neighbours = np.asarray(mesh.arrays['cellsOnEdge'])[:n_edges].astype(np.int64)
    valid = (neighbours >= 0) & (neighbours < n_cells)
    touched = np.zeros(neighbours.shape, bool)
    touched[valid] = cells[neighbours[valid]]
    edges |= touched.any(axis=1)
    return cells, edges


def keep_unchanged(old_state, old_saved, state, saved, cells, edges):
    """The carried words wherever the analysis left its background unchanged; the recoupled words elsewhere."""
    edge_fields = ('rho_u', 'normal_velocity')

    def merge(name, carried, recoupled):
        carried, recoupled = np.asarray(carried), np.asarray(recoupled)
        if carried.shape != recoupled.shape:
            raise ValueError(f'carried {name} {carried.shape} and recoupled {recoupled.shape} differ in shape')
        return np.ascontiguousarray(np.where(edges if name in edge_fields else cells, recoupled, carried))
    for name in ('rho', 'rho_theta', 'rho_u', 'rho_w', 'scalars'):
        setattr(state, name, merge(name, getattr(old_state, name), getattr(state, name)))
    from dataclasses import replace
    saved = replace(saved, **{name: merge(name, getattr(old_saved, name), getattr(saved, name))
                              for name in ('theta_m', 'exner', 'density_perturbation', 'rho_theta_perturbation',
                                           'pressure_perturbation', 'normal_velocity', 'vertical_velocity')})
    return state, saved


def _sha256(path):
    from hashlib import sha256
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def recouple_analysis(arguments, layout, inputs, *, work_dir, background=None, correct=False):
    """Replace analysed species while retaining every unanalysed LES carrier.

    ``background`` (the checkpoint's ``da_background``: the JEDI background this run wrote at the analysis instant,
    path and SHA-256) with ``correct`` (correct mode, the door default) inserts the analysis where it changed its
    background and keeps the carried forecast words where it did not.  MPAS's DA-cycling restart recomputes every
    coupled word from the analysis's rho/theta/species/u/w (atm_init_coupled_diagnostics), so a column the analysis
    never touched still moved by the round trip (rho_zz -> rho -> rho_zz, theta_m -> theta -> theta_m).  THE BREAKAGE
    THIS FIXES (fix/hex-lam-da-v842): an analysis equal to its background did not continue its own forecast, and
    every column outside the observations' reach was disturbed at every cycle.  Exact mode keeps native's recoupling
    of every column.
    """
    analysis = getattr(arguments, 'da_analysis', None)
    if analysis is None:
        return inputs, None
    if not layout.configs.get('config_do_DAcycling'):
        raise ValueError('native --da-analysis requires config_do_DAcycling=true')
    if not layout.configs.get('config_do_restart'):
        raise ValueError('native --da-analysis requires a native restart or moving carrier')
    mesh, grid, profiles, old_state, reference, old_saved, names = inputs
    analysed = tuple(name for name in names if name in contract.DEFAULT_SCALAR_NAMES)
    if not analysed or analysed[0] != 'qv':
        raise ValueError('native JEDI cycling requires the selected suite water-vapour carrier')
    state, saved, receipt = contract.recouple_da_cycling(Path(analysis),
        init_path=Path(arguments.init), mesh=mesh, vertical=grid.vertical_grid,
        terrain_metrics=grid.terrain_metrics, expected_reference=reference,
        expected_valid_time=layout.configs['config_start_time'], scalar_names=analysed,
        n_cells=layout.dimensions['nCells'], n_edges=layout.dimensions['nEdges'],
        n_levels=layout.dimensions['nVertLevels'],
        allow_regional_sentinels=bool(layout.configs.get('config_apply_lbcs', False)),
        time_seconds=old_state.time_seconds, work_dir=Path(work_dir),
        # runtime_v842's loaders form the run's base state (cold_start), as atm_init_coupled_diagnostics does.
        reference_formed=True)
    # JEDI analyses the water species, not TKE/number concentrations held by
    # the native suite. Copy those words from this member's carried state.
    scalars = np.array(old_state.scalars, copy=True, order='C')
    for index, name in enumerate(analysed):
        scalars[names.index(name)] = state.scalars[index]
    state.scalars = scalars
    receipt['native_species_retained'] = [name for name in names if name not in analysed]
    insert = dict(rule='every column recoupled (MPAS-A DA cycling)', background=None)
    if background is not None and correct:
        path = Path(background['path'])
        if _sha256(path) != background['sha256']:
            raise ValueError(f'the checkpoint names JEDI background {path}, which changed since it was written')
        record = contract.analysis_record_for_restart(Path(analysis), init_path=Path(arguments.init),
            expected_valid_time=layout.configs['config_start_time'],
            n_cells=layout.dimensions['nCells'], n_edges=layout.dimensions['nEdges'])
        after = contract.read_contract_fields(Path(analysis), contract.RECOUPLED_INPUTS, scalar_names=analysed,
                                              time_index=record)
        before = contract.read_contract_fields(path, contract.RECOUPLED_INPUTS, scalar_names=analysed)
        cells, edges = _changed_elements(after, before, analysed, mesh, layout.dimensions['nCells'],
                                         layout.dimensions['nEdges'])
        state, saved = keep_unchanged(old_state, old_saved, state, saved, cells, edges)
        insert = dict(rule='analysis inserted where it changed its background; carried words elsewhere (correct mode)',
                      background=str(path), background_sha256=background['sha256'],
                      cells_changed=int(cells.sum()), cells=int(cells.size),
                      edges_changed=int(edges.sum()), edges=int(edges.size))
    receipt['insert'] = insert
    return (mesh, grid, profiles, state, reference, saved, names), receipt
