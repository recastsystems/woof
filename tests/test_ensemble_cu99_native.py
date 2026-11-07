"""CU99 packed selection and explicit member-owned bootstrap inventory."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("edrates", [0, 1])
def test_native_cu99_route_keeps_the_actual_suite_and_clocks(edrates):
    from test_ensemble_prepared_batch import _source
    from woof.ensemble.prepared_batch import native_prepared_eligibility
    inputs, node = _source()
    node.cfg.run = replace(node.cfg.run, cu_physics=99, kf_edrates=edrates)
    before = vars(node.cfg.run).copy()
    result = native_prepared_eligibility(inputs, node, members=4)
    assert result.eligible, result.reasons
    assert vars(node.cfg.run) == before
    cumulus = next(row for row in result.suite.components if row.component == "cumulus")
    assert cumulus.native_leaf == "woof.core.kf.launch_kf_old"


@pytest.mark.parametrize("edrates", [0, 1])
def test_cu99_bootstrap_banks_every_mutable_adapter_carrier(edrates):
    from woof.core.kf import KainFritsch
    from woof.ensemble.batch_kf import KF_OLD_CARRIERS
    from woof.ensemble.batch_physics_init import bootstrap_physics_snapshot
    cfg = SimpleNamespace(nz=8, ny=3, nx=4, cu_physics=99)
    adapter = KainFritsch(99, kf_edrates=edrates)
    names = KF_OLD_CARRIERS if edrates else KF_OLD_CARRIERS[:1]
    for index, name in enumerate(names, 1):
        shape = (cfg.ny, cfg.nx) if name == "timec_kf" else (cfg.nz, cfg.ny, cfg.nx)
        setattr(adapter, name, np.full(shape, index * .125, np.float32))
    driver = SimpleNamespace(cumulus_callable=adapter)
    record = bootstrap_physics_snapshot(cfg, driver, array_module=np)
    assert "cumulus_callable" not in record.owner_digests
    assert {path for path, _, _, _ in record.structure} == {
        f"driver/cumulus_callable/{name}" for name in names}
    assert all(owner == "member" for _, _, _, owner in record.structure)
    for name in names:
        assert record.member_array(f"driver/cumulus_callable/{name}").tobytes() == getattr(adapter, name).tobytes()
    adapter.w0avg.fill(7.)
    assert np.all(record.member_array("driver/cumulus_callable/w0avg") == .125)
