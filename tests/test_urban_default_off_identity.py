"""With no urban model the engine is the engine it was; with one, the hooks
run where WRF runs them.

Three proofs:

* the Noah kernel with ``urban=None`` and with an urban hand-over that marks
  no column urban are the same function, byte for byte, on every WRF Noah
  oracle fixture (the urban arm is behind ``urban_opt > 0`` and an urban
  CATEGORY test, and neither may move a non-urban column);
* ``sf_urban_physics = 0`` builds a driver with no urban attribute, no urban
  field and no coupler, so its checkpoint inventory is unchanged;
* ``sf_urban_physics > 0`` with a stand-in model module calls ``after_lsm``
  BEFORE the runner consumes RAINBL and ``after_surface_diagnostics`` after
  SFCDIAGS and before the PBL, on every due surface step, with the rural
  values snapshotted and the solar geometry held.

The byte-identity of the default kernel path against the base commit's
kernel was measured separately on test host 1 (all 216 output arrays of the four
Noah oracle fixtures equal); ``tests/test_noah_wrf461_parity.py`` keeps
holding the kernel's distance from WRF.
"""
from __future__ import annotations

import sys
import types
from datetime import datetime

import numpy as np
import pytest

from conftest import requires_gpu


@pytest.mark.gpu
@requires_gpu
def test_a_handover_with_no_urban_column_is_the_default_kernel():
    import cupy as cp

    from woof.core.noah import launch_noah, load_tables, pack_params
    from woof.verify import noah_oracle as N

    params = pack_params(load_tables())
    for name in N.NOAH_ORACLE_FILES:
        fixture = N.load_noah_oracle(name)
        runs = []
        for handover in (False, True):
            device = {"ivgtyp": cp.asarray(fixture.ivgtyp),
                      "isltyp": cp.asarray(fixture.isltyp)}
            for field, value in fixture.inputs.items():
                device[field] = cp.asarray(value.copy())
            shape = fixture.ivgtyp.shape
            for field in ("noahres", "reslin", "chklowq"):
                device[field] = cp.zeros(shape, dtype=cp.float32)
            device["smcrel"] = cp.zeros((4, *shape), dtype=cp.float32)
            device["ebal"] = cp.zeros(shape, dtype=cp.int32)
            urban = None
            if handover:
                zeros = cp.zeros(shape, dtype=cp.float32)
                urban = {"option": 1,
                         "category_mask": cp.zeros(64, dtype=cp.int32),
                         "natural": 14, "frc_urb2d": zeros + 0.5,
                         "ts_urb2d": zeros + 250.0, "tsk_rural_bep": None,
                         "rural_q1": zeros.copy(), "rural_q2k": zeros.copy(),
                         "rural_zlvl": zeros.copy()}
            launch_noah(device, params, N.FIXTURE_DT, N.FIXTURE_DZS,
                        isurban=N.FIXTURE_ISURBAN, isice=N.FIXTURE_ISICE,
                        xice_threshold=N.FIXTURE_XICE_THRESHOLD,
                        itimestep=N.FIXTURE_ITIMESTEP, urban=urban,
                        **fixture.switches)
            runs.append({k: cp.asnumpy(v) for k, v in device.items()})
        off, on = runs
        for field in off:
            assert off[field].tobytes() == on[field].tobytes(), (name, field)


def _small_driver(**cfg_overrides):
    import cupy as cp

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_physics_driver import _base_config
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import DECLARED_CONSTANT_GLW_WM2, initialize_physics

    cfg = _base_config(**cfg_overrides)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 298.0 + 0.004 * np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(
        cfg, coord, base,
        lambda z: 0.010 * np.exp(-np.asarray(z, np.float64) / 2400.0))
    state.u[...] = cp.float32(6.0)
    state.v[...] = cp.float32(0.5)
    shape = (cfg.ny, cfg.nx)
    ivgtyp = np.full(shape, 10, dtype=np.int32)
    ivgtyp[:, :2] = 13                      # MODIS ISURBAN
    driver = initialize_physics(
        state, cfg, glw=DECLARED_CONSTANT_GLW_WM2, swdown=400.0,
        ivgtyp=ivgtyp,
        isltyp=np.full(shape, 8, dtype=np.int32), tsk=295.0,
        radiation_start_time=datetime(2024, 7, 1, 18),
        radiation_latitude=np.full(shape, 40.0),
        radiation_longitude=np.full(shape, -100.0))
    return state, cfg, driver


@pytest.mark.gpu
@requires_gpu
def test_the_default_driver_carries_nothing_urban():
    state, cfg, driver = _small_driver()
    assert driver.urban is None and driver.urban_coupler is None
    assert not any(name.endswith(("_urb2d", "_urb3d", "_urb4d"))
                   or name == "tsk_rural" for name in driver.fields)


@pytest.fixture
def stand_in_ucm(monkeypatch):
    """A model module with the DESIGN 3.4 surface that records its calls."""
    calls = []
    module = types.ModuleType("woof.core.urban_ucm")
    module.STATE_SPEC = {}
    module.DIMENSIONS = {}

    def init_state(state, params, **kw):
        calls.append(("init_state", state.option))

    def after_lsm(state, params, *, lsm, fields, atmosphere, dt, itimestep,
                  solar, cfg):
        import cupy as cp
        calls.append(("after_lsm", itimestep,
                      float(cp.asnumpy(fields["rainbl"]).max()),
                      sorted(state.rural), solar.model_time, lsm))

    def after_surface_diagnostics(state, *, lsm, fields, atmosphere, cfg):
        calls.append(("after_surface_diagnostics", lsm))

    module.init_state = init_state
    module.after_lsm = after_lsm
    module.after_surface_diagnostics = after_surface_diagnostics
    monkeypatch.setitem(sys.modules, "woof.core.urban_ucm", module)
    monkeypatch.setattr("woof.config.urban_model_in_build",
                        lambda option: option == 1)
    return calls


@pytest.mark.gpu
@requires_gpu
def test_the_hooks_run_where_wrf_runs_them(stand_in_ucm):
    import cupy as cp

    state, cfg, driver = _small_driver(sf_urban_physics=1)
    urban = driver.urban
    assert urban is not None and urban.option == 1
    assert stand_in_ucm == [("init_state", 1)]
    utype = cp.asnumpy(driver.fields["utype_urb2d"])
    assert (utype[:, :2] == 2).all() and (utype[:, 2:] == 0).all()
    frc = cp.asnumpy(driver.fields["frc_urb2d"])
    assert (frc[:, :2] == np.float32(0.9)).all()     # FRC_URB_TBL(2)
    driver._pending_rainbl[...] = cp.float32(0.25)
    driver.compute(state, cfg)
    names = [c[0] for c in stand_in_ucm]
    assert names == ["init_state", "after_lsm", "after_surface_diagnostics"]
    _, itimestep, rain_seen, rural, solar_time, lsm = stand_in_ucm[1]
    assert itimestep == 1 and lsm == 2
    assert rain_seen == pytest.approx(0.25)          # before RAINBL is zeroed
    assert {"t1", "sheat", "eta_kinematic", "q1", "q2k", "zlvl"} <= set(rural)
    assert solar_time == 0.0
    assert float(cp.asnumpy(driver.fields["rainbl"]).max()) == 0.0
    # The memory checks price exactly what was allocated.
    from woof.core.urban_state import urban_array_shapes
    priced = urban_array_shapes(cfg)
    for name in urban.names:
        assert priced[name] == driver.fields[name].shape, name
    # The rural Q1 was written on urban columns only.
    q1 = cp.asnumpy(urban.rural["q1"])
    assert (q1[:, :2] > 0).all() and (q1[:, 2:] == 0).all()


@pytest.mark.gpu
@requires_gpu
def test_the_urban_state_rides_a_checkpoint(stand_in_ucm, tmp_path):
    """Every UrbanState array is a ``fields`` entry, so the checkpoint the
    engine already writes carries it and a resume restores it in place."""
    import cupy as cp

    from woof.io import restart

    # Thompson, because the checkpoint's tendency inventory is built for a
    # configured microphysics scheme (a moist mp=0 shim is not a run).
    state, cfg, driver = _small_driver(sf_urban_physics=1, mp_physics=8)
    driver.compute(state, cfg)
    # Move every urban array off its cold-start value, so a resume that
    # re-ran the cold start instead of restoring would be caught.
    for i, name in enumerate(driver.urban.names):
        array = driver.fields[name]
        array[...] = (array + (i + 1)).astype(array.dtype)
    path = restart.write_restart(tmp_path / "urban.npz", state, cfg)
    resumed_state, _, resumed = _small_driver(sf_urban_physics=1,
                                              mp_physics=8)
    views = {name: resumed.fields[name] for name in resumed.urban.names}
    restart.restore_restart(path, resumed_state, cfg)
    for name in driver.urban.names:
        assert resumed.fields[name] is views[name], name   # restored in place
        assert cp.asnumpy(resumed.fields[name]).tobytes() == cp.asnumpy(
            driver.fields[name]).tobytes(), name


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("variant", ["rte-rrtmgp", "rrtmg_legacy", None])
def test_bem_gets_the_direct_and_diffuse_shortwave(variant):
    """BEP_BEM reads SWDDIR/SWDDIF (shadow_mas, short_rad_dd).  RRTMG hands
    over its own swdkdir/swdkdif, RTE+RRTMGP its direct beam, and with no
    shortwave scheme the radiation driver's Ruiz-Arias split applies."""
    import cupy as cp

    kw = dict(sf_urban_physics=3, mp_physics=8)
    if variant is not None:
        kw.update(ra_lw_physics=4, ra_sw_physics=4, ra_rrtmg_variant=variant,
                  radt_minutes=0.1)
    state, cfg, driver = _small_driver(**kw)
    driver.compute(state, cfg)
    f = driver.fields
    swdown = cp.asnumpy(f["swdown"])
    split = cp.asnumpy(f["swddir"]) + cp.asnumpy(f["swddif"])
    assert driver._swdd_from_scheme is (variant is not None)
    assert (cp.asnumpy(f["swddir"]) > 0).all()
    assert (cp.asnumpy(f["swddif"]) > 0).all()
    np.testing.assert_allclose(split, swdown, rtol=1e-5)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("option", [1, 2, 3])
def test_the_health_gate_classifies_every_urban_field(option):
    """The per-step GPU health gate refuses any field it cannot classify,
    and the real forecast path stopped at model time 0 on utype_urb2d
    (int32) before it was excluded like the other category maps."""
    from woof.core.health import collect_state_fields, gpu_integer_policy

    state, cfg, driver = _small_driver(sf_urban_physics=option, mp_physics=8)
    driver.compute(state, cfg)
    fields = collect_state_fields(state, backend="gpu")
    names = [field.name for field in fields]
    assert "surface.utype_urb2d" in names
    for field in fields:
        gpu_integer_policy(field.name, field.values.dtype)  # raises if not


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("option", [1, 2, 3])
def test_a_tiled_run_refuses_the_urban_models_rather_than_misplacing_the_sun(
        option):
    """Tile streaming does not carry the urban models yet, and must say so.

    A tile buffer is built on tilestream's neutral POISON geography and
    serves many tiles, so an urban coupler inside it would compute its
    solar geometry at the buffer's latitudes (not the tile's) and keep a
    column plan made for the first tile it saw.  The tile road's own
    geography check is what refuses that today: it walks the driver, finds
    ``urban_coupler.latitude_deg``, and the gather does not reach it
    (tilestream/driver.py _SCHEME_GEOGRAPHY lists radiation, Noah-MP and
    CAM ozone only).  Adding the coupler to that table without also making
    the column plan per-tile would turn this refusal into a silently wrong
    forecast, which is what this test is here to stop.
    """
    from tilestream.driver import (GeographyNotGatherable,
                                   assert_geography_gathered)

    state, cfg, driver = _small_driver(sf_urban_physics=option, mp_physics=8)
    driver.compute(state, cfg)
    with pytest.raises(GeographyNotGatherable,
                       match=r"driver\.urban_coupler\.latitude_deg"):
        assert_geography_gathered(state)

    state0, cfg0, driver0 = _small_driver(mp_physics=8)
    driver0.compute(state0, cfg0)
    try:
        assert_geography_gathered(state0)
    except GeographyNotGatherable as exc:  # pragma: no cover - not urban's
        assert "urban" not in str(exc)


def test_the_default_noah_launch_compiles_without_the_urban_handover():
    """The default launch is the URBAN = false instantiation, built from the
    pre-hand-over statements.  With the hand-over compiled into one kernel
    behind a runtime flag, NVRTC contracted a different set of products into
    FMAs and a 1 h default forecast drifted from integrate/2.8's from its
    15-minute history on (RTX 5090, 2026-09-30); with this split the five
    histories are byte-identical.  Every hand-over statement must stay under
    ``if constexpr (URBAN)`` and the default path must not take the urban
    entry point."""
    import inspect
    import re
    from pathlib import Path

    from woof.core import noah

    text = (Path(__file__).resolve().parents[1] / "woof" / "core" / "kernels"
            / "noah.cu").read_text(encoding="utf-8")
    declaration = re.search(
        r'extern\s+"C"\s+__global__\s+'
        r'(?:__launch_bounds__\([^)]*\)\s+)?void\s+noah_column\(', text)
    assert declaration is not None, "the default Noah kernel declaration is missing"
    default = text[declaration.start():]
    default = default[:default.index("\n}\n")]
    assert "noah_column_body<false>(" in default and "urban_opt" not in default
    assert 'void noah_column_urban(' in text
    body = text[text.index("void noah_column_body("):]
    body = body[:body.index('\nextern "C"')]
    for token in ("urban_opt", "tsk_rural_bep[idx] =", "rural_q1[idx] ="):
        for match in re.finditer(re.escape(token), body):
            before = body[:match.start()]
            if "{\n" not in before[before.rfind("void noah_column_body("):]:
                continue  # the parameter list
            opened = before.rfind("if constexpr (URBAN) {")
            assert opened >= 0, token
            depth = before[opened:].count("{") - before[opened:].count("}")
            assert depth > 0, f"{token} at {match.start()} is outside if constexpr (URBAN)"
    source = inspect.getsource(noah.launch_noah)
    # The entry point is chosen by name since the FASDAS and irrigation
    # variants (2e0e878fea): the default (urban is None) path takes
    # noah_column, the urban hand-over noah_column_urban, through the one
    # get_kernel(module, name) launch.
    assert 'name = "noah_column" if urban is None else "noah_column_urban"' in source
    assert 'module = "noah_ua" if ua_phys else "noah"' in source
    assert "kern = get_kernel(module, name)" in source
