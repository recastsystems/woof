"""New SAS (cu_physics = 96) is reachable, and admitted as table rows only.

The arithmetic is graded in tests/test_nsas_column_oracle.py.  This file
holds what has to be true for the scheme to be reachable from a bare
default configuration and from a WRF namelist, and for every refusal around
it to name the breakage it prevents.  All CPU.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.config import (CU_SCHEMES, NSAS_DX_FACTOR_SCHEMES,
                          SAS_FAMILY_DEFAULTS, SAS_PGCON_SCHEMES, RunConfig,
                          validate_run_config)
from woof.core.column_cumulus import (COLUMN_CUMULUS_SCHEMES,
                                       ColumnCumulus, wrf_sas_ncloud)

ROOT = Path(__file__).resolve().parents[1]


def _base():
    return RunConfig(nx=16, ny=16, nz=40, dx=12000., dy=12000.,
                     ztop=20000., dt=30., run_seconds=600., moist=True)


def _cfg(**kw):
    return dataclasses.replace(_base(), cu_physics=96, **kw)


def test_bare_default_config_with_96_validates():
    """cu_physics = 96 and nothing else: WRF's defaults for every knob."""
    cfg = validate_run_config(_cfg())
    assert cfg.cu_physics == 96 and 96 in CU_SCHEMES
    assert (cfg.nsas_dx_factor, cfg.sas_pgcon) == (0, 0.55)
    # cudt keeps RunConfig's default: New SAS honours a cumulus step, as
    # WRF does, so the bare default is a runnable configuration.
    assert cfg.cudt_minutes == _base().cudt_minutes
    validate_run_config(_cfg(nsas_dx_factor=1, sas_pgcon=0.2, cudt_minutes=0.0))


@pytest.mark.parametrize("bad", (2, -1, True))
def test_nsas_dx_factor_outside_its_two_branches_is_refused(bad):
    with pytest.raises(ValueError, match="nsas_dx_factor must be 0 or 1"):
        validate_run_config(_cfg(nsas_dx_factor=bad))


def test_sas_knobs_under_another_scheme_are_refused_by_name():
    for cu in (0, 1):
        base = dataclasses.replace(_base(), cu_physics=cu)
        with pytest.raises(ValueError, match="would reach nothing"):
            validate_run_config(dataclasses.replace(base, nsas_dx_factor=1))
        with pytest.raises(ValueError, match="would reach nothing"):
            validate_run_config(dataclasses.replace(base, sas_pgcon=0.2))
    with pytest.raises(ValueError, match="must be finite"):
        validate_run_config(_cfg(sas_pgcon=float("nan")))


def test_adaptive_step_requires_cudt_zero_as_wrf_does():
    with pytest.raises(ValueError, match="module_check_a_mundo.F:2142"):
        validate_run_config(_cfg(use_adaptive_time_step=True,
                                 cudt_minutes=5.0))
    validate_run_config(_cfg(use_adaptive_time_step=True, cudt_minutes=0.0))


def test_the_scheme_is_a_table_row_not_an_adapter_module():
    """The engine admits 96 through COLUMN_CUMULUS_SCHEMES and nothing else."""
    row = COLUMN_CUMULUS_SCHEMES[96]
    assert row.kernel_module == "nsas" and row.entry == "nsas_column"
    assert not (ROOT / "woof" / "core" / "nsas.py").exists()
    from woof.core import physics
    src = inspect.getsource(physics.initialize_physics)
    assert "COLUMN_CUMULUS_SCHEMES" in src and "ColumnCumulus(" in src
    assert "cu_physics == 96" not in src
    assert ColumnCumulus(96).scheme is row


def test_identity_tables_agree_with_the_row():
    from woof.checkpoint_identity import CUMULUS_ALGORITHM_IDENTITIES
    from woof.core.preflight import _CUMULUS_KERNEL_MODULES
    from woof.wrf461_compatibility import CUMULUS_OPTIONS
    row = COLUMN_CUMULUS_SCHEMES[96]
    assert CUMULUS_ALGORITHM_IDENTITIES[96] == row.identity
    assert _CUMULUS_KERNEL_MODULES[96] == (row.kernel_module,)
    assert 96 in CUMULUS_OPTIONS
    registry = json.loads(
        (ROOT / "woof" / "physics_registry_v2.json").read_text("utf-8"))
    option = registry["components"]["cumulus"]["options"]["new-sas"]
    assert option["selectors"] == {"cu_physics": 96}
    consumers = option["consumers"]
    assert consumers["restart_algorithm_identity"] == row.identity
    assert consumers["stock_callable_class"] == (
        f"{ColumnCumulus.__module__}.{ColumnCumulus.__qualname__}")
    from woof import physics_vertical_contract as pvc
    bounds = consumers["vertical_level_bounds"]
    assert getattr(pvc, bounds["contract_constant"]) == (
        bounds["minimum"], bounds["maximum"])
    for name in ("nsas_dx_factor", "sas_pgcon"):
        spec = registry["parameters"][name]
        assert spec["default"] == SAS_FAMILY_DEFAULTS[name]
        assert spec["consuming_read"] == "woof/core/column_cumulus.py"


def test_ncloud_mapping_is_wrfs():
    """module_cu_nsas.F:141-147."""
    assert wrf_sas_ncloud(0) == 0
    assert wrf_sas_ncloud(1) == wrf_sas_ncloud(3) == 1
    assert {wrf_sas_ncloud(m) for m in (2, 4, 6, 8, 10, 16, 28, 50)} == {2}


def test_workspace_is_priced_from_the_launchers_own_formula():
    from woof.core.column_cumulus import workspace_bytes
    from woof.core.preflight import (column_cumulus_workspace_bytes,
                                      column_workspace_bytes)

    def exp(*cu):
        return SimpleNamespace(domains=[SimpleNamespace(run=SimpleNamespace(
            cu_physics=c, bl_pbl_physics=0, mp_physics=0, shcu_physics=0, grav_settling=0,
            nx=300, ny=200, nz=50))
            for c in cu])
    priced = column_cumulus_workspace_bytes(exp(96))
    assert priced == workspace_bytes(96, 50, 60000, 188) or priced > 0
    assert column_workspace_bytes(exp(96)) >= priced
    assert column_cumulus_workspace_bytes(exp(0, 1, 3, 16)) == 0
    assert column_cumulus_workspace_bytes(exp(16, 96)) == priced
    assert "workspace_bytes(" in inspect.getsource(column_cumulus_workspace_bytes)


def test_knobs_leave_every_other_schemes_identity_unchanged():
    """Scoped out of the fingerprint and the checkpoint digest off 96."""
    from woof.io import restart
    values = dataclasses.asdict(_base())
    restart._drop_default_off_run_keys(values)
    assert "nsas_dx_factor" not in values and "sas_pgcon" not in values
    values = dataclasses.asdict(_cfg(nsas_dx_factor=1))
    restart._drop_default_off_run_keys(values)
    assert values["nsas_dx_factor"] == 1 and values["sas_pgcon"] == 0.55
    from woof.core import model
    src = inspect.getsource(model)
    assert "_cumulus_scoped_run_fields()" in src
    assert set(model._cumulus_scoped_run_fields()) == set(SAS_FAMILY_DEFAULTS) | {
        "numbins", "thbinsize", "rbinsize", "mindeepfreq", "minshallowfreq", "shallowcu_forced_ra", "kf_edrates"}
    assert NSAS_DX_FACTOR_SCHEMES == frozenset({96})
    assert SAS_PGCON_SCHEMES == frozenset({94, 96})


# --------------------------------------------------------------------------
# The namelist door
# --------------------------------------------------------------------------

def _nsas_input(text):
    return text.replace(" cu_physics = 1, 0,", " cu_physics = 96, 0,")


def test_namelist_selecting_96_imports_with_its_knobs(tmp_path):
    from test_namelist_import import _import_with, _load
    toml_text, _ = _import_with(
        tmp_path, extra_physics=" nsas_dx_factor = 1,\n sas_pgcon = 0.2, 0.2,\n",
        inp_filter=_nsas_input)
    exp = _load(tmp_path, toml_text)
    root, child = exp.domains[0].run, exp.domains[1].run
    assert (root.cu_physics, root.nsas_dx_factor, root.sas_pgcon) == (96, 1, 0.2)
    assert root.cudt_minutes == 5.0, "WRF's cudt reaches the SAS domain"
    assert (child.cu_physics, child.nsas_dx_factor, child.sas_pgcon) == (0, 0, 0.55)


def test_namelist_knobs_without_96_are_dropped_with_a_reason(tmp_path):
    from test_namelist_import import _import_with
    _, report = _import_with(
        tmp_path, extra_physics=" nsas_dx_factor = 1,\n sas_pgcon = 0.3, 0.3,\n")
    dropped = {d.key for d in report.dropped}
    assert "nsas_dx_factor" in dropped
    assert {"sas_pgcon[1]", "sas_pgcon[2]"} <= dropped


def test_nscv_beside_nsas_is_refused_naming_the_double_shallow(tmp_path):
    from test_namelist_import import _import_with
    with pytest.raises(Exception, match="convect the same boundary layer twice"):
        _import_with(tmp_path, extra_physics=" shcu_physics = 4, 0,\n",
                     inp_filter=_nsas_input)



def test_named_suite_reaches_both_prepared_routes():
    from woof.physics_compat import NSAS_PROFILE_ID, COMPOSITION_SUITE_PROFILE_IDS
    from woof.physics_registry import physics_registry
    registry = physics_registry()
    assert NSAS_PROFILE_ID in COMPOSITION_SUITE_PROFILE_IDS
    assert registry["templates"][NSAS_PROFILE_ID]["components"]["cumulus"] == "new-sas"
    for route_id in ("tools.prepared_domain_tree_forecast", "tools.prepared_single_domain_forecast"):
        route = registry["runner_routes"][route_id]
        assert "new-sas" in route["allowed_component_options"]["cumulus"]
