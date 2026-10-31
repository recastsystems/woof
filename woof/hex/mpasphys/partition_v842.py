"""Registry pool transport for process-isolated route-B ranks.

This is metadata and word transport, not a physics implementation. Native
connectivity is one based, unlike the CUDA partition mesh. A missing local
referent must name the local native dummy, never the first solved cell.
"""
from __future__ import annotations

from copy import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from ..partition_local_mesh_v841 import CELL_VALUED, EDGE_VALUED, VERTEX_VALUED
from .pool_v842 import NativePoolHydration, RegistryLayout

_ENTITIES = {
    'nCells': ('cell_l2g', 'n_owned_cells'),
    'nEdges': ('edge_l2g', 'n_owned_edges'),
    'nVertices': ('vertex_l2g', 'n_owned_vertices'),
}
#: A field at least this large projects its partitions concurrently (project_payloads).
_RANKS_PARALLEL_BYTES = 64 << 20

_REFERENCES = {
    **{name: 'nCells' for name in CELL_VALUED | {'cellsOnCellsOnCell'}},
    **{name: 'nEdges' for name in EDGE_VALUED},
    **{name: 'nVertices' for name in VERTEX_VALUED},
}


def _maps(layout, partition):
    maps = {}
    for entity, (ids_name, owned_name) in _ENTITIES.items():
        ids = np.asarray(getattr(partition, ids_name))
        total = layout.dimensions[entity]
        owned = getattr(partition, owned_name)
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or
                not 0 <= owned <= len(ids) or np.any(ids < 0) or
                np.any(ids >= total) or len(np.unique(ids)) != len(ids)):
            raise ValueError(f'invalid {entity} partition map')
        inverse = np.full(total + 2, len(ids) + 1, dtype=np.int32)
        inverse[0] = 0
        inverse[ids + 1] = np.arange(1, len(ids) + 1, dtype=np.int32)
        maps[entity] = (ids, owned, inverse)
    return maps


def partition_maps(layout, partitions):
    """Every partition's validated maps, for many gather_owned_field calls (each _maps sorts every id list)."""
    return [_maps(layout, part) for part in partitions]


def partition_layout(layout: RegistryLayout, partition) -> RegistryLayout:
    """Select local Registry extents, retaining species and pool identities.

    Halo ends are intentionally required from the caller for a provider that
    needs ring-specific exchanges. Cell rings alone cannot classify edge and
    vertex halo rings under every native decomposition convention.
    """
    return _partition_layout(layout, _maps(layout, partition))


def rank_halo_ends(part, cells_on_edge, cells_on_vertex, layers):
    """Cumulative local ends of each of ``layers`` halo layers, per entity, for a same-host rank.

    A row whose native layout prepare reads ring ends (the RRFS fork, driver_rrfs.prepare_native_layout) needs
    them on every rank. Cell layers come from the partition's cell rings, whose local order is ring-ascending
    (partition_assets_v841); the last layer also holds every deeper ring, so the last end is the local total.
    Consumers: the fork diagnostics' native vertex order and halo-w copy (output_diagnostics_rrfs). An edge or
    vertex takes the lowest ring of its local cells, plus one. On a same-host rank those two are metadata only:
    the ported framework refuses its own ring exchanges there (framework_backend_v842.exchange_halo), the rank
    owner exchanges every halo.
    """
    n = int(part._global_cells)
    cell_rings = np.asarray(part.cell_ring, dtype=np.int64)[int(part.n_owned_cells):]
    if cell_rings.size and np.any(np.diff(cell_rings) < 0):
        # THE BREAKAGE THIS PREVENTS: ends counted from rings assume ring-ascending local order; any other order
        # puts ring-2 cells inside layer 1, and the fork's seam smoothing would read a halo layer native never has.
        raise ValueError('partition halo cells are not in ring order; halo layer ends cannot be counted')
    cring = np.full(n + 1, 1 << 20, dtype=np.int64)
    cring[np.asarray(part.cell_l2g, dtype=np.int64)] = np.asarray(part.cell_ring, dtype=np.int64)

    def ends(owned, rings):
        out = [owned + int(np.count_nonzero(rings <= r)) for r in range(1, layers)]
        return tuple(out) + (owned + int(rings.size),)
    coe = np.asarray(cells_on_edge, dtype=np.int64)
    cov = np.asarray(cells_on_vertex, dtype=np.int64)
    coe = np.where((coe < 0) | (coe >= n), n, coe)
    cov = np.where((cov < 0) | (cov >= n), n, cov)
    e = np.asarray(part.edge_l2g, dtype=np.int64)[int(part.n_owned_edges):]
    v = np.asarray(part.vertex_l2g, dtype=np.int64)[int(part.n_owned_vertices):]
    return {'nCells': ends(int(part.n_owned_cells), np.maximum(cell_rings, 1)),
            'nEdges': ends(int(part.n_owned_edges), cring[coe[e]].min(axis=1) + 1),
            'nVertices': ends(int(part.n_owned_vertices), cring[cov[v]].min(axis=1) + 1)}


def _partition_layout(layout, maps):
    local = copy(layout)
    local.dimensions = dict(layout.dimensions)
    local.configs = dict(layout.configs)
    local.indices = dict(layout.indices)
    local.fields = []
    if hasattr(local, 'halo_ends'):
        # Global empty-halo metadata is not a distributed decomposition.
        del local.halo_ends
    for entity, (ids, owned, _) in maps.items():
        local.dimensions.update({entity: len(ids), entity + 'All': len(ids),
                                 entity + 'Solve': owned})
    for row in layout.fields:
        field = dict(row)
        field['dimensions'] = list(row['dimensions'])
        field['components'] = list(row.get('components', ()))
        field['shape'] = [local.dimensions[d] + int(d in _ENTITIES)
                          for d in row['dimensions']]
        local.fields.append(field)
    return local


def project_buffer(buffer, row, layout, partition):
    """Slice every decomposed axis and remap only declared connectivity.

    Real words, including NaN payloads and signed zero, are copied unchanged.
    The final native dummy column is carried from the global dummy column.
    Zero/negative padding is preserved; positive absent/dummy references use
    the local dummy. Global identity arrays are sliced without renumbering.
    """
    return _project_buffer(buffer, row, layout, _maps(layout, partition))


def _project_buffer(buffer, row, layout, maps):
    values = np.asarray(buffer)
    expected = tuple(row['shape'])
    if values.shape != expected:
        raise ValueError(f"{row['pool']}.{row['name']} shape {values.shape} != {expected}")
    dtype = np.dtype('<f4' if row['kind'] == 'real' else '<i4')
    if row['kind'] not in ('real', 'integer') or values.dtype != dtype:
        raise ValueError(f"{row['pool']}.{row['name']} is not a native word buffer")
    fresh = False
    for axis, dim in enumerate(row['dimensions']):
        for entity, (ids, owned, _) in maps.items():
            if dim == entity:
                indices = np.concatenate((ids, [layout.dimensions[entity]]))
            elif dim == entity + 'All':
                indices = ids
            elif dim == entity + 'Solve':
                indices = ids[:owned]
            else:
                continue
            # Native buffers are Fortran-ordered: gather on the transposed (C-ordered) view, so the result is
            # Fortran-ordered already and each entity's words are copied as one contiguous run.  THE COST THIS
            # REMOVES: np.take on the Fortran array returned a C-ordered gather that the final copy then
            # transposed word by word, 13.5 s for four LTS classes of the 98k-cell storm-following payload on one
            # thread (0.8 s now), mostly under the GIL (MPAS-ROUTE-SPEED-2026-10-24).  The words are the same.
            values = np.take(values.T, indices, axis=values.ndim - 1 - axis).T
            fresh = True
            break
    referent = _REFERENCES.get(row['name']) if row['pool'] == 'mesh' else None
    if referent is not None:
        if row['kind'] != 'integer':
            raise ValueError(f"connectivity {row['name']} requires native integers")
        inverse = maps[referent][2]
        if np.any(values > layout.dimensions[referent] + 1):
            raise ValueError(f"connectivity {row['name']} exceeds native {referent} dummy")
        values = values if fresh else values.copy(order='F')
        positive = values > 0
        values[positive] = inverse[values[positive]]
        fresh = True
    if fresh and values.dtype == dtype and values.flags.f_contiguous:
        return values                   # already a private Fortran-ordered copy of the words
    return np.array(values, dtype=dtype, order='F', copy=True)


def project_payload(layout: RegistryLayout, payload: Path, partition,
                    output: Path) -> RegistryLayout:
    """Write an independently validated rank payload to a fresh directory.

    The caller owns process creation and GPU selection. This function does
    not create a native session or claim that an executor supports ranks.
    """
    return project_payloads(layout, payload, [partition], [output])[0]


def project_payloads(layout: RegistryLayout, payload: Path, partitions, outputs, *, workers=None):
    """``project_payload`` for several partitions at once, every field on its own worker.

    Each source field is decoded once for every partition, and the fields are projected, written and hashed on
    all cores.  THE COST THIS REMOVES: the MPAS LTS class executor projected the whole payload once per class,
    field after field (numpy take, file write, sha256), about 30 s before the first step of a storm-following
    segment on box-aws2x and the largest part of a resumed segment's 90 s preparation
    (MPAS-ROUTE-SPEED-2026-10-24).  The bytes and manifests written are project_payload's.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor
    from ..host_parallel import run_tasks
    from .pool_v842 import remember_payload_digest
    source = NativePoolHydration(None, layout, payload)
    # Each partition's maps (np.unique over its ids) on its own thread: ~2.7 s one after another on conus3km.
    maps = run_tasks([lambda partition=partition: _maps(layout, partition) for partition in partitions])
    locals_ = [_partition_layout(layout, item) for item in maps]
    # A fresh directory prevents a failed/reused rank from mixing manifests.
    for output in outputs:
        output.mkdir(parents=True, exist_ok=False)

    def field(index):
        row = layout.fields[index]
        original = source.records[row['pool'], row['name']]
        # A Registry-initial zero field (decode_registry_pools' fill record) projects to zero words on every
        # partition: it stays a fill record, so no rank or class writes, hashes and reads back its zeros and its
        # hydration allocates it untouched.  THE COST THIS REMOVES (MPAS-route host RAM, 2026-10-27): each LTS rate
        # class wrote 6.8 GB of zeros per 100k cells to disk and copied every word of them into its pools.
        filled = row['kind'] != 'character' and original.get('fill') is not None
        buffer = (None if row['kind'] == 'character' or filled
                  else source.field_buffer(row['pool'], row['name']))

        def one(item):
            mapping, local, output = item
            projected = local.fields[index]
            record = {name: projected[name] for name in ('pool', 'name', 'kind', 'dimensions', 'shape')}
            if 'origins' in original:
                # Which input stream filled the words: SuiteIau admits an IAU run only
                # when the iau stream read tend_iau, and a rank payload without origins
                # refused every multi-rank IAU run before its first step.
                record['origins'] = original['origins']
            if row['kind'] == 'character':
                record['text'] = original['text']
            elif filled:
                record.update(payload=None, sha256=None, text=None, fill=original['fill'])
            else:
                # _project_buffer returns a private Fortran-ordered array: its memory is the file's bytes
                # (tobytes(order='F') without the extra copy).
                values = _project_buffer(buffer, row, layout, mapping)
                words = memoryview(values.reshape(-1, order='F')).cast('B')
                name = f'field-{index:06d}.bin'
                (output / name).write_bytes(words)
                sha256 = hashlib.sha256(words).hexdigest()
                remember_payload_digest(output / name, sha256)
                record.update(payload=name, sha256=sha256)
            return record

        items = list(zip(maps, locals_, outputs))
        if buffer is not None and len(items) > 1 and buffer.nbytes >= _RANKS_PARALLEL_BYTES:
            # A large field's partitions are gathered, written and hashed on their own threads.  THE COST THIS
            # REMOVES (conus3km 2.34 M cells on 4 same-host ranks, V4X6000, 2026-10-29): the scalars (~33 GB)
            # were projected for one rank after another on one worker, the ~20 s tail of the payload projection
            # while every other field had finished.  Same bytes and records, in rank order.
            with ThreadPoolExecutor(max_workers=len(items), thread_name_prefix='hex-project-rank') as ranks:
                return list(ranks.map(one, items))
        return [one(item) for item in items]

    count = len(layout.fields)
    workers = max(1, min(count, workers or os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='hex-project-payload') as pool:
        per_field = list(pool.map(field, range(count)))
    for which, (local, output) in enumerate(zip(locals_, outputs)):
        (output / 'manifest.json').write_text(json.dumps({
            'schema': 'mpas-native-pool-payload-v1', 'fields': [records[which] for records in per_field]},
            indent=2) + '\n')
        NativePoolHydration(None, local, output)
    return locals_


def gather_owned_field(row, layout, partitions, buffers, *, maps=None):
    """Assemble one carried/output field using owner words only.

    The result excludes native dummy storage. Connectivity is deliberately
    excluded: its rank-local values need inverse remapping before output.
    Fields with no decomposed axis must agree bit for bit on every rank.
    This supports arbitrary Registry species counts, including LES TKE.
    """
    if not partitions or len(partitions) != len(buffers):
        raise ValueError('one field buffer is required for each partition')
    if row['kind'] not in ('real', 'integer'):
        raise ValueError('owned field assembly requires a numeric Registry field')
    if row['pool'] == 'mesh':
        raise ValueError('mesh fields require global connectivity reconstruction')
    axes = [(axis, entity) for axis, dim in enumerate(row['dimensions'])
            for entity in _ENTITIES if dim in (entity, entity + 'All', entity + 'Solve')]
    if len(axes) > 1:
        raise ValueError('owned field assembly requires at most one decomposed axis')
    # ``maps``: the partitions' _maps, built once by a caller that assembles many fields (partition_maps).
    maps = [_maps(layout, part) for part in partitions] if maps is None else maps
    dtype = np.dtype('<f4' if row['kind'] == 'real' else '<i4')
    for mapping, buffer in zip(maps, buffers):
        dimensions = dict(layout.dimensions)
        for entity, (ids, owned, _) in mapping.items():
            dimensions.update({entity: len(ids), entity + 'All': len(ids),
                               entity + 'Solve': owned})
        expected = tuple(dimensions[d] + int(d in _ENTITIES) for d in row['dimensions'])
        if np.asarray(buffer).shape != expected or np.asarray(buffer).dtype != dtype:
            raise ValueError('rank field shape/dtype differs from local Registry')
    if not axes:
        first = np.asarray(buffers[0])
        if any(first.tobytes(order='F') != np.asarray(b).tobytes(order='F') for b in buffers[1:]):
            raise ValueError('replicated Registry words differ between ranks')
        return first.copy(order='F')
    axis, entity = axes[0]
    count = layout.dimensions[entity]
    seen = np.zeros(count, dtype=np.bool_)
    shape = list(row['shape'])
    shape[axis] = count
    result = np.empty(shape, dtype=dtype, order='F')
    for mapping, buffer in zip(maps, buffers):
        ids, owned, _ = mapping[entity]
        ids = ids[:owned]
        if np.any(seen[ids]):
            raise ValueError('duplicate Registry field owners')
        seen[ids] = True
        target = [slice(None)] * result.ndim
        source = [slice(None)] * result.ndim
        target[axis] = ids
        source[axis] = slice(0, owned)
        result[tuple(target)] = np.asarray(buffer)[tuple(source)]
    if not np.all(seen):
        raise ValueError('Registry field ownership does not cover the global domain')
    return result
