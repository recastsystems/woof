"""Both prepared forecast runners bind the hydro settings scope.

Breakage: the item 10 change inserted ``_root_ring_hydrometeor_record``
between ``@_hydro_forecast_scope`` and ``def run_prepared_tree``, so the
decorator wrapped the one-line helper and the tree runner ran without the
scope that binds a WRF-Hydro run's settings (woof.hydro.runtime).
"""

from woof.hydro.runtime import forecast_scope

_SCOPE_CODE = forecast_scope(lambda inputs: inputs).__code__


def test_the_tree_runner_is_inside_the_hydro_scope():
    from woof import prepared_domain_tree_forecast as tree

    assert tree.run_prepared_tree.__code__ is _SCOPE_CODE
    assert tree._root_ring_hydrometeor_record.__code__ is not _SCOPE_CODE


def test_the_single_domain_runner_is_inside_the_hydro_scope():
    from woof import prepared_single_domain_forecast as single

    assert single.run_prepared_forecast.__code__ is _SCOPE_CODE
