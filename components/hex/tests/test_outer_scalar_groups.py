"""Direct physics state changes must reach the outer transport state."""
from types import SimpleNamespace as NS

import numpy as np

from woof.hex.mpasphys.runtime_v842 import commit_outer_scalar_groups
from woof.hex.mpasphys import route_table


def test_selected_group_updates_owned_columns_without_other_rows(monkeypatch):
    # An arbitrary group proves this is a route contract, not a smoke branch.
    monkeypatch.setattr(route_table, 'route_row', lambda config: {'outer_scalar_groups': ['tracer']})
    pool = np.arange(7 * 2 * 3, dtype=np.float32).reshape(7, 2, 3)
    target = np.full((9, 2, 4), -1, dtype=np.float32)
    session = NS(config=object(), layout=NS(indices={('state', 'tracer_start'): 4,
                                                   ('state', 'tracer_end'): 6}),
                 field=lambda pool_name, name, level: pool)
    commit_outer_scalar_groups(session, NS(state=NS(scalars=target)), np)
    np.testing.assert_array_equal(target[3:6, :, :3], pool[3:6])
    assert np.all(target[:3] == -1) and np.all(target[6:] == -1)
    assert np.all(target[:, :, 3] == -1)


def test_absent_or_empty_groups_do_not_read_fields(monkeypatch):
    for row in ({}, {'outer_scalar_groups': ['tracer']}):
        monkeypatch.setattr(route_table, 'route_row', lambda config: row)
        commit_outer_scalar_groups(NS(config=object(), layout=NS(indices={})), None, None)


def test_rrfs_declares_direct_chemistry_updates():
    assert tuple(route_table.route_row(NS(table_key='mpasphys_rrfs'))['outer_scalar_groups']) == ('chemistry',)
    assert not route_table.route_row(NS(table_key='mpasphys_v842')).get('outer_scalar_groups')
