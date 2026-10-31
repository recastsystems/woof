"""Native one-based pool transport under a noncontiguous decomposition."""
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from woof.hex.mpasphys.partition_v842 import (
    gather_owned_field, partition_layout, project_buffer, project_payload,
)
from woof.hex.mpasphys.pool_v842 import NativePoolHydration


def case():
    rows = [
        dict(pool='state', name='scalars', kind='real',
             dimensions=['num_scalars', 'nVertLevels', 'nCells'], shape=[3, 2, 7],
             components=['qv', 'qc', 'tke']),
        dict(pool='mesh', name='cellsOnEdge', kind='integer',
             dimensions=['TWO', 'nEdges'], shape=[2, 6], components=[]),
        dict(pool='mesh', name='indexToCellID', kind='integer',
             dimensions=['nCells'], shape=[7], components=[]),
        dict(pool='state', name='xtime', kind='character', dimensions=[], shape=[], components=[]),
    ]
    for row in rows:
        row.update(pool_name=row['name'], nlevels=2 if row['pool'] == 'state' else 1)
    layout = SimpleNamespace(fields=rows, dimensions=dict(nCells=6, nEdges=5, nVertices=4,
        nCellsSolve=6, nCellsAll=6, nEdgesSolve=5, nEdgesAll=5,
        nVerticesSolve=4, nVerticesAll=4, num_scalars=3, nVertLevels=2, TWO=2),
        configs={'config_do_restart': True}, indices={('state', 'index_qv'): 1},
        halo_ends={'nCells': (6, 6)})
    partition = SimpleNamespace(cell_l2g=np.array([3, 1, 4], np.int32), n_owned_cells=2,
        edge_l2g=np.array([2, 0], np.int32), n_owned_edges=1,
        vertex_l2g=np.array([1, 3], np.int32), n_owned_vertices=1)
    return layout, partition


def test_species_words_dummy_and_native_time_levels_survive_projection():
    layout, partition = case()
    row = layout.fields[0]
    # Include signed zero and distinct NaN payloads. No floating point maths.
    words = (np.arange(42, dtype=np.uint32) + np.uint32(0x7f800001)).reshape(3, 2, 7)
    words[0, 0, 1] = 0x80000000
    output = project_buffer(words.view(np.float32), row, layout, partition)
    assert output.flags.f_contiguous
    np.testing.assert_array_equal(output.view(np.uint32), words[..., [3, 1, 4, 6]])
    local = partition_layout(layout, partition)
    assert local.fields[0]['shape'] == [3, 2, 4]
    assert local.fields[0]['nlevels'] == 2
    assert local.fields[0]['components'] == ['qv', 'qc', 'tke']
    assert local.indices == layout.indices
    assert local.dimensions['nCellsSolve'] == 2
    assert local.dimensions['nEdgesSolve'] == 1
    assert not hasattr(local, 'halo_ends')
    local.configs['config_do_restart'] = False
    local.fields[0]['components'].append('other')
    assert layout.configs['config_do_restart']
    assert len(layout.fields[0]['components']) == 3


def test_native_connectivity_uses_local_dummy_and_retains_padding():
    layout, partition = case()
    # Edge 2 references global cells 4,5; edge 0 references absent cell 1
    # and padding -1. The global dummy column includes the global dummy 7.
    values = np.array([[1, 2, 4, 4, 0, 7], [-1, 3, 5, 6, 0, 0]], np.int32)
    output = project_buffer(values, layout.fields[1], layout, partition)
    np.testing.assert_array_equal(output, [[1, 4, 4], [3, -1, 0]])
    assert output[0, 1] != 1  # An absent donor cannot alias owned cell 0.
    values[0, 2] = 8
    with pytest.raises(ValueError, match='exceeds native nCells dummy'):
        project_buffer(values, layout.fields[1], layout, partition)


def test_global_identity_is_sliced_without_local_renumbering():
    layout, partition = case()
    values = np.array([11, 40, 82, 100, 8, 9, 0], np.int32)
    np.testing.assert_array_equal(project_buffer(values, layout.fields[2], layout, partition),
                                  [100, 40, 8, 0])


@pytest.mark.parametrize('name,entity,values,expected', [
    ('edgesOnCell', 'nCells',
     [[3, 1, 3, 3, 5, 0, 6], [1, 3, 0, 1, 4, -1, 0]],
     [[1, 2, 3, 3], [2, 1, 3, 0]]),
    ('verticesOnEdge', 'nEdges',
     [[2, 1, 4, 0, 5, 5], [4, 4, 1, -1, 2, 0]],
     [[2, 1, 3], [3, 2, 0]]),
])
def test_edge_and_vertex_references_use_their_own_dummy(name, entity, values, expected):
    layout, partition = case()
    values = np.array(values, np.int32)
    row = dict(pool='mesh', name=name, kind='integer', dimensions=['TWO', entity],
               shape=list(values.shape))
    np.testing.assert_array_equal(project_buffer(values, row, layout, partition), expected)


def test_all_and_solve_axes_and_multiple_decomposed_axes():
    layout, partition = case()
    values = np.arange(30, dtype=np.int32).reshape(6, 5)
    row = dict(pool='diag', name='test', kind='integer',
               dimensions=['nCellsSolve', 'nEdgesAll'], shape=[6, 5])
    output = project_buffer(values, row, layout, partition)
    np.testing.assert_array_equal(output, values[np.ix_([3, 1], [2, 0])])


@pytest.mark.parametrize('ids', [[3, 3], [-1, 2], [6], [1.5]])
def test_invalid_partition_fails_before_projection(ids):
    layout, partition = case()
    partition.cell_l2g = np.asarray(ids)
    with pytest.raises(ValueError, match='invalid nCells partition map'):
        partition_layout(layout, partition)


def write_source(path, layout):
    path.mkdir()
    records = []
    buffers = []
    for index, row in enumerate(layout.fields):
        record = {key: row[key] for key in ('pool', 'name', 'kind', 'dimensions', 'shape')}
        if row['kind'] == 'character':
            record['text'] = '2026-10-17_00:00:00'
            buffers.append(None)
        else:
            dtype = np.float32 if row['kind'] == 'real' else np.int32
            values = np.arange(np.prod(row['shape']), dtype=dtype).reshape(row['shape'], order='F')
            if row['name'] == 'cellsOnEdge':
                values %= 8
            data = values.tobytes(order='F')
            name = f'{index}.bin'
            (path / name).write_bytes(data)
            record.update(payload=name, sha256=hashlib.sha256(data).hexdigest())
            buffers.append(values)
        records.append(record)
    (path / 'manifest.json').write_text(json.dumps(dict(schema='mpas-native-pool-payload-v1', fields=records)))
    return buffers


def test_rank_payload_hydrates_all_native_time_levels(tmp_path):
    layout, partition = case()
    buffers = write_source(tmp_path / 'source', layout)
    output = tmp_path / 'rank-0'
    local = project_payload(layout, tmp_path / 'source', partition, output)
    hydrated = NativePoolHydration(None, local, output)
    np.testing.assert_array_equal(hydrated.field_buffer('state', 'scalars'),
                                 buffers[0][..., [3, 1, 4, 6]])
    assert hydrated.records['state', 'xtime']['text'] == '2026-10-17_00:00:00'
    seen = []
    native = SimpleNamespace(config=lambda *args: None, pool=lambda *args: None,
        dimension=lambda *args: None,
        put_field=lambda *args, **kwargs: seen.append((args, kwargs)),
        put_text_field=lambda *args, **kwargs: seen.append((args, kwargs)))
    NativePoolHydration(native, local, output).populate()
    scalar_levels = [kwargs['level'] for args, kwargs in seen if args[:2] == ('state', 'scalars')]
    assert scalar_levels == [1, 2]
    with pytest.raises(FileExistsError):
        project_payload(layout, tmp_path / 'source', partition, output)


def test_resident_class_payload_matches_transport_and_keeps_private_storage(tmp_path):
    from woof.hex.mpasphys import moving_pool_payload_v842 as resident
    layout, partition = case()
    source = tmp_path/'source'
    write_source(source, layout)
    disk = tmp_path/'disk'
    expected_layout = project_payload(layout, source, partition, disk)
    expected = NativePoolHydration(None, expected_layout, disk)
    output = tmp_path/'memory'
    local = resident.project(layout, source, [partition], [output])[0]
    try:
        actual = NativePoolHydration(None, local, output)
        assert local.fields == expected_layout.fields
        assert not list(output.glob('*.bin'))
        assert actual.records['state', 'xtime']['text'] == expected.records['state', 'xtime']['text']
        for row in local.fields:
            if row['kind'] == 'character':
                continue
            a = actual.field_buffer(row['pool'], row['name'])
            b = expected.field_buffer(row['pool'], row['name'])
            assert a.tobytes(order='F') == b.tobytes(order='F')
            assert a.flags.f_contiguous
            a[...] = 0
            again = actual.field_buffer(row['pool'], row['name'])
            assert again.tobytes(order='F') == b.tobytes(order='F')
        from woof.hex.mpasphys.framework_backend_v842 import FrameworkPhysics
        with FrameworkPhysics() as native:
            NativePoolHydration(native, local, output).populate()
            assert native._storage['state', 'scalars'].peek(0) is None
            assert native.get_field('state', 'scalars').tobytes(order='F') == expected.field_buffer(
                'state', 'scalars').tobytes(order='F')
        local.fields[0]['shape'] = [7]
        with pytest.raises(ValueError, match='metadata differs'):
            NativePoolHydration(None, local, output)
    finally:
        resident.clear()
    assert resident.lookup(output) is None


def test_resident_projection_validates_source_before_registration(tmp_path):
    from woof.hex.mpasphys import moving_pool_payload_v842 as resident
    layout, partition = case()
    source = tmp_path/'source'
    write_source(source, layout)
    (source/'0.bin').write_bytes(b'bad')
    output = tmp_path/'memory'
    with pytest.raises(ValueError, match='digest differs'):
        resident.project(layout, source, [partition], [output])
    assert not output.exists()
    assert resident.lookup(output) is None


def test_corrupt_global_payload_fails_before_creating_rank_directory(tmp_path):
    layout, partition = case()
    write_source(tmp_path / 'source', layout)
    (tmp_path / 'source' / '0.bin').write_bytes(b'bad')
    with pytest.raises(ValueError, match='digest differs'):
        project_payload(layout, tmp_path / 'source', partition, tmp_path / 'rank-0')
    assert not (tmp_path / 'rank-0').exists()


def owner_pair():
    layout, first = case()
    second = SimpleNamespace(cell_l2g=np.array([0, 2, 4, 5, 1], np.int32), n_owned_cells=4,
        edge_l2g=np.array([0, 1, 3, 4, 2], np.int32), n_owned_edges=4,
        vertex_l2g=np.array([0, 2, 1, 3], np.int32), n_owned_vertices=3)
    return layout, [first, second]


def test_owner_assembly_ignores_poisoned_halos_and_preserves_all_species():
    layout, partitions = owner_pair()
    row = layout.fields[0]
    original = (np.arange(42, dtype=np.uint32) + np.uint32(0x7f800001)).reshape(3, 2, 7)
    original[1, 0, 3] = 0x80000000
    local = [project_buffer(original.view(np.float32), row, layout, part) for part in partitions]
    for part, values in zip(partitions, local):
        values.view(np.uint32)[..., part.n_owned_cells:] = 0xdeadbeef
    assembled = gather_owned_field(row, layout, partitions, local)
    np.testing.assert_array_equal(assembled.view(np.uint32), original[..., :-1])


def test_owner_assembly_refuses_holes_and_duplicate_owners():
    layout, partitions = owner_pair()
    row = layout.fields[0]
    original = np.zeros(row['shape'], np.float32)
    local = [project_buffer(original, row, layout, part) for part in partitions]
    partitions[1].n_owned_cells = 3
    with pytest.raises(ValueError, match='does not cover'):
        gather_owned_field(row, layout, partitions, local)
    partitions[1].n_owned_cells = 5
    with pytest.raises(ValueError, match='duplicate Registry field owners'):
        gather_owned_field(row, layout, partitions, local)


def test_replicated_words_are_checked_bitwise_including_signed_zero():
    layout, partitions = owner_pair()
    row = dict(pool='state', name='Time', kind='real', dimensions=[], shape=[])
    np.testing.assert_array_equal(gather_owned_field(row, layout, partitions,
                                                   [np.array(0, np.float32)] * 2), 0)
    with pytest.raises(ValueError, match='replicated Registry words differ'):
        gather_owned_field(row, layout, partitions,
                           [np.array(0., np.float32), np.array(-0., np.float32)])


def test_owner_assembly_refuses_stale_shapes_and_mesh_connectivity():
    layout, partitions = owner_pair()
    with pytest.raises(ValueError, match='shape/dtype differs'):
        gather_owned_field(layout.fields[0], layout, partitions,
                           [np.zeros((3, 2, 4), np.float32)] * 2)
    with pytest.raises(ValueError, match='global connectivity reconstruction'):
        gather_owned_field(layout.fields[1], layout, partitions,
                           [np.zeros((2, 3), np.int32)] * 2)
