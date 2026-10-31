"""Prepare the next moving region's class inputs without retaining another GPU forecast.

Rate admission still runs on the carried atmosphere. The initial atmosphere is
only a prediction for work performed beside the previous segment; each class's
complete partition must match before its projected pools and geometry are used.
Changed classes take the normal preparation path. No carried state is predicted.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def same_partition(before, after, *, input_only=False):
    """Exact local ordering, ownership, halo/completeness and global extents."""
    a, b = vars(before), vars(after)
    if a.keys() != b.keys():
        return False
    for key, value in a.items():
        if input_only and key in ('n_owned_cells','n_owned_edges','n_owned_vertices','cell_ring'):
            continue
        other = b[key]
        if isinstance(value, np.ndarray):
            if not isinstance(other, np.ndarray) or value.dtype != other.dtype or not np.array_equal(value, other):
                return False
        elif value != other:
            return False
    return True


def _slice(values, partition):
    from .mpi_v842 import _rank_shells
    from ..partition_local_mesh_v841 import slice_prepared_host
    return slice_prepared_host(_rank_shells(values, {}), partition)


def static_inputs(inputs, partition):
    mesh, grid, profiles, _, reference, _, names = inputs
    local = _slice(SimpleNamespace(mesh=mesh, grid=grid, profiles=profiles, reference=reference), partition)
    return local.mesh, local.grid, local.profiles, None, local.reference, None, tuple(names)


def carried_inputs(inputs, partition, prepared):
    """Fresh state/diagnostics; only immutable member geometry is reused."""
    local = _slice(SimpleNamespace(state=inputs[3], saved=inputs[5]), partition)
    mesh, grid, profiles, _, reference, _, names = prepared
    return mesh, grid, profiles, local.state, reference, local.saved, names


def predict(arguments, layout, payload, inputs, config=None):
    """Host class pools and geometry from this member, before its carrier exists."""
    if (getattr(arguments, 'fullstep_lts_mode', None) != 'active'
            or getattr(arguments, 'samehost_devices', None)):
        return None
    from .lts_v842 import session_partition
    from .partition_v842 import project_payloads
    from ..drivers.fullstep_lts_forecast_v841 import _boundary_rate
    from ..fullstep_lts_admission import admit_mesh_rates, metric_lengths
    from ..fullstep_lts_classes import build_class_plan
    from ..host_parallel import run_tasks
    mesh, grid, _, state, _, saved, _ = inputs
    # Match run_lts_classes' declarations before the common mesh slicer sees
    # initialized deformation planes (their entity axis is not inferred).
    declarations = dict(getattr(mesh, 'variable_dimensions', {}) or {})
    for name in ('defc_a', 'defc_b'):
        if name in mesh.arrays:
            declarations.setdefault(name, ('nCells', 'maxEdges'))
    mesh.variable_dimensions = declarations
    named_dt = getattr(arguments, 'dt', None)
    dt = float(layout.configs['config_dt'] if named_dt in (None, 'auto', 'row') else named_dt)
    max_rate = int(getattr(arguments, 'fullstep_lts_max_rate', None) or 8)
    rates, _ = admit_mesh_rates(
        mesh, state, saved, grid.vertical_grid, dt=dt, max_rate=max_rate,
        acoustic_substeps=int(layout.configs['config_number_of_sub_steps']),
        target_transport=.90 if layout.configs.get('config_scalar_advection', True) else None,
        boundary_rate=_boundary_rate(max_rate))
    if rates.max() == 1:
        return None
    plan = build_class_plan(
        rates=rates, cells_on_cell=np.asarray(mesh.arrays['cellsOnCell']),
        n_edges_on_cell=np.asarray(mesh.arrays['nEdgesOnCell']),
        cells_on_edge=np.asarray(mesh.arrays['cellsOnEdge']),
        cells_on_vertex=np.asarray(mesh.arrays['cellsOnVertex']),
        area_cell=np.asarray(mesh.area_cell, dtype=np.float64),
        dc_edge_m=metric_lengths(mesh)[0], base_dt=dt,
        edges_on_edge=mesh.arrays.get('edgesOnEdge'), n_edges_on_edge=mesh.arrays.get('nEdgesOnEdge'))
    root = Path(payload).parent / 'resident-lts'
    partitions = [tables.layout for tables in plan.classes]
    outputs = [root / f'pool.class{index}' for index in range(len(partitions))]
    layouts = project_payloads(layout, payload, [session_partition(p) for p in partitions], outputs)
    from .mpi_v842 import local_inputs
    local_inputs_ = run_tasks([lambda p=p: local_inputs(inputs, p) for p in partitions])
    records = [dict(rate=tables.rate, partition=p, layout=local, payload=out,
                    inputs=(local_data[0], local_data[1], local_data[2], None,
                            local_data[4], None, local_data[6]))
               for tables, p, local, out, local_data in
               zip(plan.classes, partitions, layouts, outputs, local_inputs_)]
    if config is not None:
        from .moving_class_gpu_v842 import prepare
        def build(index):
            try:
                prepare(records[index], local_inputs_[index], config, arguments,
                        dt * records[index]['rate'])
            except Exception as error:
                # A prediction never replaces admission or the normal build.
                records[index].setdefault('gpu_preparation', {})['error'] = f'{type(error).__name__}: {error}'
        run_tasks([lambda i=i: build(i) for i in range(len(records))])
    return records


def matching(predicted, units, *, input_only=False):
    """A matching class can be reused even when another class's admission changed."""
    result = []
    for unit in units:
        item = next((item for item in predicted or ()
                     if item['rate'] == unit.rate and same_partition(item['partition'],unit.layout)),None)
        if item is None and input_only:
            required = {'cell_l2g','edge_l2g','vertex_l2g','cell_row_complete','edge_row_complete',
                        'vertex_row_complete','_global_cells','_global_edges','_global_vertices'}
            item = next((p for p in predicted or () if p['rate'] == unit.rate
                         and required <= vars(p['partition']).keys()
                         and same_partition(p['partition'],unit.layout,input_only=True)),None)
            if item is not None:
                # Physics/dycore solve ALL local entities (session_partition).
                # The actual coordinator still owns the freshly admitted
                # partition. Only its ownership/ring metadata differed; every
                # gathered input and every local connectivity word is equal.
                item['partition'] = unit.layout
                mesh = item['inputs'][0]
                mesh.partition_layout = mesh.partition_layout_v841 = unit.layout
                item.setdefault('gpu_preparation',{})['reused_for_ownership_change'] = True
        result.append(item)
    return result
