"""RUC, actually forecasting.

Every other RUC test in this repository measures a routine against an oracle
CSV -- 48 fixture cases of ``LSMRUC``, and hundreds more of the leaves under
it.  None of them runs the model.  This one does: ``sf_surface_physics=3``
over mixed land, water and sea ice, stepped through RK3, and it asserts the
things that separate "ported" from "runs":

* the selector reaches RUC and NOT Noah.  That distinction is the whole
  reason value-based dispatch exists (commit 9362172): the driver used to
  branch on the truthiness of ``sf_surface_physics``, so a RUC request would
  have run Noah and produced a plausible forecast with no error;
* the cold start produces a usable state -- SH2O and SMFR3D are not the
  allocation zeros after ``ruclsminit``, because "no frozen water anywhere"
  is a claim, not an absence of one;
* the driver's own ``ktau==1`` repair block fires, so SOILT1/QVG/QSG/RHOSNF
  leave the zeros ``initialize_physics`` allocated;
* the surface fluxes, the nine-level soil column and the runoff accumulators
  actually move, which is what says the scheme is coupled rather than called;
* nothing goes non-finite;
* a restart round trip reproduces every carried array bit for bit AND the
  next step from the restored state is bit identical -- with a companion test
  that perturbs one word and proves that comparison can fail;
* a real wrfout is written and read back with nine soil layers; and
* the two open RUC leads are settled by measurement rather than by argument:
  ``udrunoff`` is shown to be correctly zero on an unsaturated column and
  nonzero when WRF's own condition for it is met, and ``ilnb`` is shown to be
  DEFINED rather than inherited from the previous column.

The grid is small on purpose.  The column runs on the host -- there is no
device ``sfctmp`` and no device ``lsmruc`` in this tree -- so the per-column
cost is milliseconds and a large domain would make this file unrunnable
rather than more convincing.  That cost is the scheme's scaling blocker, it is
published in the registry warnings, and
``test_the_column_cost_is_what_the_registry_says`` measures it instead of
hiding it.
"""

from __future__ import annotations

import dataclasses
import numpy as np
import pytest

from conftest import requires_gpu

import cupy as cp

from woof.config import RunConfig
from woof.core.ruc_runtime import (DEFINED_ILNB, RUC_DIAGNOSTICS_2D,
                                    RUC_STATE_2D, RUC_STATE_3D,
                                    SEAICE_ALBEDO_DEFAULT, XICE_THRESHOLD)
from woof.core.physics import DECLARED_CONSTANT_GLW_WM2  # noqa: E402

#: The idealised constant downward longwave these fixtures declare.
#:
#: ``woof.core.physics.initialize_physics`` no longer defaults ``glw``
#: (300.0 through 1.8.7): a land-surface suite with no longwave scheme
#: must state where its downward longwave comes from instead of being
#: handed a plausible-looking 300 W m-2 nobody chose.  These are
#: idealised columns; the constant is the right answer for them and this
#: is where they say so.  The VALUE is 1.8.7's default, so every fixture
#: below integrates exactly the numbers it always did.
_IDEALISED_GLW = DECLARED_CONSTANT_GLW_WM2

#: MODI-RUC categories, VEGPARM.TBL:253-274.  Named rather than inlined so a
#: reader can see the vegetated columns are grassland and that the sea-ice
#: test really does select the snow/ice row.
_GRASSLAND = 10
_ICE = 15
_WATER = 17
#: STAS-RUC soil rows.
_LOAM = 6
_WATER_SOIL = 14

_NSOIL = 9

#: Every array a RUC forecast carries across a step: the RUC-only state, the
#: generic soil column, and the generic surface fields RUC writes.
_CARRIED = (*RUC_STATE_2D, *RUC_STATE_3D,
            "tslb", "smois", "sh2o", "snow", "snowh", "snowc", "canwat",
            "lai", "mavail", "tsk", "hfx", "qfx", "lh", "grdflx", "albedo",
            "emiss", "qsfc", "z0", "znt", "sfcrunoff", "udrunoff", "acsnow",
            "acsnom", "smstav", "smstot", "chklowq", "t2", "th2", "q2")


def _host_frame_value(value):
    """One frame field on the host, KEEPING ITS TYPE.

    These two writer tests used to coerce every field to ``float32`` on the
    way out, which was invisible while the frame was all reals and became a
    refusal the moment the land-identity rows joined it: WRF declares
    ISLTYP/IVGTYP ``integer`` (Registry.EM_COMMON:857-858) and
    ``WrfoutWriter`` refuses a float array for an integer-declared variable
    rather than truncate every category.  The blanket cast was the test's
    own artifact -- the production frame builders hand ``state_frame``'s
    output to the writer unchanged, integers included -- so the cast is
    narrowed to the floats it was ever meant for.
    """
    import numpy as _np

    array = _np.asarray(value)
    if array.dtype.kind in "iu":
        return array
    return _np.asarray(array, dtype=_np.float32)


def _maxsmc(soiltyp: int) -> float:
    """MAXSMC for one STAS-RUC row, read from the packaged table.

    Read rather than inlined because ``dqm = MAXSMC - DRYSMC`` is the
    saturation threshold every ``qq.gt.dqm`` branch in ``soilmoist`` tests,
    and a hardcoded 0.451 would be a magic number that silently stops meaning
    "saturated loam" if the table is ever repinned.
    """
    from woof.core.ruc import load_ruc_parameters

    return float(load_ruc_parameters().soil.rows[soiltyp - 1].values[3])


def _build(*, nx: int = 8, ny: int = 6, nz: int = 40, vegtyp: int = _GRASSLAND,
           soiltyp: int = _LOAM, water_columns: int = 2,
           ice_rows: int = 0, snow_mm: float = 0.0,
           snow_depth_m: float = 0.0, soil_moisture: float = 0.28,
           frozen: bool = False, dt: float = 12.0, radiation=None,
           ra_physics: int = 0, radt_minutes: float = 12.0,
           mp_physics: int = 6, sf_sfclay_physics: int = 1,
           bl_pbl_physics: int = 1, nzs: int = _NSOIL,
           ruc_soilprop: str = "wrf_45", ruc_snow: str = "wrf_45",
           fractional_seaice: int = 0):
    """One RUC forecast configuration.

    ``nzs`` defaults to :data:`_NSOIL`, which is what keeps every caller in
    this file the nine-level forecast it was: the argument is threaded to
    ``num_soil_layers`` and to the two soil profiles built from it, and
    nothing else in the builder reads a level count.

    ``frozen=True`` swaps the whole column for a subfreezing one -- a 268 K
    boundary layer over a 265 K surface over a 258..265 K soil profile.  It is
    not cosmetic: RUC's frozen-soil state (SMFR3D, KEEPFR3DFLAG) is unreadable
    on a warm column, because ``soilprop`` only consults it inside
    ``if(keepfr(k).eq.1.)`` and ``keepfr`` only reaches 1 inside
    ``if (soilice(k).gt.0.)``.  Measured: 0 of 216 soil cells have
    KEEPFR3DFLAG==1 on the warm grid, 16 of 216 on this one.
    """
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics

    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0, ztop=16000.0,
                    dt=dt, run_seconds=0.0, time_step_sound=4, moist=True,
                    mp_physics=mp_physics,
                    sf_sfclay_physics=sf_sfclay_physics,
                    sf_surface_physics=3, num_soil_layers=nzs,
                    bl_pbl_physics=bl_pbl_physics, bldt=0.0,
                    ra_physics=ra_physics, radt_minutes=radt_minutes,
                    ruc_soilprop=ruc_soilprop, ruc_snow=ruc_snow,
                    fractional_seaice=fractional_seaice)

    def theta(z):
        z = np.asarray(z, np.float64)
        if frozen:
            return 268.0 + 0.004 * z
        return np.where(z < 1500.0, 300.0,
                        np.where(z < 1700.0, 300.0 + 0.030 * (z - 1500.0),
                                 306.0 + 0.0045 * (z - 1700.0)))

    def qvapor(z):
        z = np.asarray(z, np.float64)
        if frozen:
            return np.maximum(0.0022 - 5.0e-7 * z, 1.0e-6)
        return np.where(z < 1500.0, 0.0110,
                        np.maximum(0.0110 - 5.0e-6 * (z - 1500.0), 1.0e-5))

    coord = make_vertical_coord(cfg.nz, stretch=2.0)
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, qvapor)
    state.u[...] = cp.float32(5.0)
    state.v[...] = cp.float32(1.0)

    landmask = np.ones((cfg.ny, cfg.nx), np.float64)
    if water_columns:
        landmask[:, -water_columns:] = 0.0
    land = landmask == 1.0
    xice = np.zeros((cfg.ny, cfg.nx))
    if ice_rows:
        # Sea ice is a LAND-masked column with XICE past the threshold, which
        # is how WRF's own XLAND/XICE pair spells it: LSMRUC:851 tests XICE
        # after it has already rejected XLAND>=1.5 as water at :824.
        xice[:ice_rows, :cfg.nx - water_columns] = 1.0

    # On the warm grid the surface is deliberately WARMER than the 300 K
    # boundary layer, so the unstable branch a daytime forecast actually takes
    # is the one exercised.
    tsk = np.full((cfg.ny, cfg.nx), 265.0 if frozen else 303.0)
    tsk[~land] = 271.0 if frozen else 294.0
    if ice_rows:
        tsk[:ice_rows, :cfg.nx - water_columns] = 265.0
    spread = 7.0 if frozen else 8.0
    soil_t = np.stack([tsk - level
                       for level in np.linspace(0.0, spread, nzs)])
    soil_m = np.full((nzs, cfg.ny, cfg.nx), soil_moisture)
    soil_m[:, ~land] = 1.0

    vegetation = np.where(land, vegtyp, _WATER)
    if ice_rows:
        vegetation[:ice_rows, :cfg.nx - water_columns] = _ICE
    driver = initialize_physics(
        state, cfg, landmask=landmask, tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m,
        ivgtyp=vegetation,
        isltyp=np.where(land, soiltyp, _WATER_SOIL),
        vegfra=60.0, tmn=270.0 if frozen else 288.0,
        swdown=120.0 if frozen else 700.0,
        glw=220.0 if frozen else 340.0, pblh=500.0,
        xice=xice, snow=snow_mm, snow_depth=snow_depth_m,
        radiation=radiation)
    from woof.config import radiation_scheme_ids
    if radiation_scheme_ids(cfg)[1] == 0:
        # Radiation off: RUC consumes GSW -- the NET shortwave, not the
        # SWDOWN declared above -- and nothing produces it, so the carrier
        # contract would refuse the first surface step.  The forcing door
        # is how an offline-forced RUC run supplies it, and it labels the
        # provenance (external_array).  Writing the allocation's own zeros
        # moves no number these forecasts ever integrated; it names the
        # number's origin, which is the whole of the contract's demand.
        driver.set_forcing(gsw=0.0)
    return state, cfg, driver


def _land(array, cfg, *, water_columns: int = 2):
    return np.asarray(array)[..., :cfg.nx - water_columns]


# ---------------------------------------------------------------------------
# admission
# ---------------------------------------------------------------------------

@requires_gpu
def test_the_selector_reaches_ruc_and_not_noah():
    """The failure this test exists for is a silent one.

    ``PhysicsDriver.compute`` used to branch on the truthiness of
    ``sf_surface_physics``, so ``=3`` ran ``launch_noah``.  A Noah run under a
    RUC selector produces a complete, plausible, finite forecast.  So the
    assertion is not "something ran": it is that the resolved routing names
    RUC's runner, that Noah's parameter bundle was never loaded, and that the
    nine-layer geometry reached the arrays.
    """
    state, cfg, driver = _build(nx=6, ny=4, nz=20)
    assert driver.scheme_dispatch["sf_surface_physics"] == "_run_ruc"
    assert driver.ruc_params is not None
    # Noah's VEGPARM/SOILPARM/GENPARM bundle must NOT be attached: wrfout's
    # historical live-surface gate keyed on it, and loading it here would let
    # a RUC run claim a Noah state.
    assert driver.noah_params is None
    assert driver.noahmp_params is None
    for name in ("tslb", "smois", "sh2o", "smfr3d", "keepfr3dflag"):
        assert driver.fields[name].shape == (_NSOIL, cfg.ny, cfg.nx), name
    # RUC has its own 2-m diagnostic, so it must be absent from the set that
    # selects Noah's SFCDIAGS.
    from woof.core.physics import LAND_SURFACE_SFCDIAGS_SCHEMES
    assert 3 not in LAND_SURFACE_SFCDIAGS_SCHEMES


@requires_gpu
def test_a_forecast_runs_with_every_other_lsm_entry_point_booby_trapped(
        monkeypatch):
    """The routing proof that does not trust a configuration value.

    ``driver.scheme_dispatch["sf_surface_physics"] == "_run_ruc"`` is a
    receipt: it says what ``resolve_physics_dispatch`` decided, not what ran.
    The failure this guards against produces exactly that receipt and still
    runs the wrong scheme, because before value-based dispatch landed the
    driver branched on the TRUTHINESS of ``sf_surface_physics`` and a
    ``=3`` request ran ``launch_noah`` to a complete, plausible, finite
    forecast.

    So this replaces every OTHER land-surface entry point with a raising
    tripwire and then runs twenty RK3 steps.  If any of them is reached the
    forecast dies with that exact message.  Then it removes the tripwire from
    RUC's own seam and shows the tripwire mechanism works at all -- a booby
    trap nobody proved can fire is the same non-evidence as a gate that has
    never failed.
    """
    from woof.core import physics as physics_module
    from woof.core.dycore import step

    def noah_tripwire(*args, **kwargs):
        raise AssertionError(
            "launch_noah was reached under sf_surface_physics=3: the RUC "
            "selector fell through to Noah")

    def noahmp_tripwire(*args, **kwargs):
        raise AssertionError(
            "noahmp_lsm_step was reached under sf_surface_physics=3")

    def sfcdiags_tripwire(*args, **kwargs):
        raise AssertionError(
            "Noah's SFCDIAGS refresh was reached under sf_surface_physics=3; "
            "RUC brings its own 2-m diagnostic and must not borrow it")

    monkeypatch.setattr(physics_module, "launch_noah", noah_tripwire)
    monkeypatch.setattr(physics_module, "noahmp_lsm_step", noahmp_tripwire)
    monkeypatch.setattr(physics_module.PhysicsDriver,
                        "_refresh_surface_diagnostics", sfcdiags_tripwire)

    state, cfg, driver = _build(nx=6, ny=4, nz=20)
    before = cp.asnumpy(driver.fields["tsk"]).copy()
    for _ in range(20):
        step(state, cfg)
    # It ran, it ran RUC, and it moved -- all three, with Noah unreachable.
    assert driver.last_ruc_census is not None
    assert (cp.asnumpy(driver.fields["tsk"]) != before).any()

    # The tripwire itself fires.  Point it at RUC's own seam and the same
    # twenty steps must now fail, which is what makes the twenty above mean
    # something.
    def ruc_tripwire(*args, **kwargs):
        raise AssertionError("ruc_lsm_step reached")

    monkeypatch.setattr(physics_module, "ruc_lsm_step", ruc_tripwire)
    state, cfg, driver = _build(nx=6, ny=4, nz=20)
    with pytest.raises(AssertionError, match="ruc_lsm_step reached"):
        step(state, cfg)


@requires_gpu
def test_a_forecast_runs_with_every_host_sfctmp_leaf_booby_trapped(
        monkeypatch):
    """The device column has to be REACHED, not merely to exist.

    ``woof/core/ruc_gpu.py`` carried four bitwise ``sfctmp`` leaves and a
    bitwise snow-preparation stage with ZERO importers under ``gpuwm/``: the
    only thing that ever launched them was ``tests/``.  A forecast therefore
    ran the host Python path at roughly a thousand times the cost while every
    per-leaf parity gate in the suite stayed green, which is the same failure
    the RUC soil ingest had -- finished, verified, and wired to nothing.

    A timing assertion cannot catch that, because the host path is also
    correct and merely slow.  So this replaces all four host leaves and the
    host stage with raising tripwires and runs the forecast.  If the seam
    forgets to pass the device sets, the driver falls back to
    ``RUC_SFCTMP_HOST_LEAVES``/``RUC_SFCTMP_HOST_STAGES`` and dies here.

    Then it shows the tripwire fires: dropping the device sets from the same
    call reproduces the failure, so the pass above is evidence rather than a
    gate that has never been able to fail.
    """
    from woof.core import ruc as ruc_module
    from woof.core import ruc_runtime as runtime_module
    from woof.core.dycore import step

    def leaf_tripwire(name):
        def call(*args, **kwargs):
            raise AssertionError(
                f"the HOST sfctmp leaf {name!r} was reached in a forecast: "
                "ruc_lsm_step did not pass the device leaf set")
        return call

    monkeypatch.setattr(ruc_module, "RUC_SFCTMP_HOST_LEAVES", {
        name: leaf_tripwire(name)
        for name in ("soil", "sea_ice", "snow_soil", "snow_sea_ice")})
    monkeypatch.setattr(ruc_module, "RUC_SFCTMP_HOST_STAGES",
                        {"snow_prep": leaf_tripwire("snow_prep")})

    state, cfg, driver = _build(nx=6, ny=4, nz=20, snow_mm=15.0,
                                snow_depth_m=0.08)
    before = cp.asnumpy(driver.fields["tsk"]).copy()
    for _ in range(5):
        step(state, cfg)
    assert driver.last_ruc_census is not None
    assert (cp.asnumpy(driver.fields["tsk"]) != before).any()

    # The same run with the LEAF AND STAGE sets withheld must now hit a
    # tripwire.  The array namespace is deliberately left in place: the three
    # go together, and withholding all three fails earlier and differently --
    # the seam hands the driver device arrays, so a host driver dies on
    # "implicit conversion to a NumPy array" before it reaches a leaf at all.
    # That is also a real guard, but it is not THIS one, and a falsification
    # that fires for the wrong reason proves nothing about the gate above.
    from woof.core.ruc_gpu import RUC_DEVICE_ARRAYS
    monkeypatch.setattr(runtime_module, "ruc_device_sfctmp_sets",
                        lambda: (None, None, RUC_DEVICE_ARRAYS))
    state, cfg, driver = _build(nx=6, ny=4, nz=20, snow_mm=15.0,
                                snow_depth_m=0.08)
    with pytest.raises(AssertionError, match="HOST sfctmp leaf"):
        step(state, cfg)


@requires_gpu
def test_the_cold_start_is_usable_and_not_the_allocation_zeros():
    """``ruclsminit`` and the driver's ktau==1 repair, measured separately.

    Two different mechanisms have to fire before RUC has a state.
    ``ruclsminit`` runs at driver construction and derives SH2O/SMFR3D/MAVAIL/
    ZNT from TSLB/SMOIS by the freezing curve.  The rest of RUC's carriers are
    repaired inside ``LSMRUC``'s own ``ktau==1`` block, which is why they are
    still zero before the first step and must not be after it.
    """
    from woof.core.dycore import step

    state, cfg, driver = _build(nx=6, ny=4, nz=20)

    # ruclsminit, at construction.
    for name in ("sh2o", "mavail", "znt"):
        array = cp.asnumpy(driver.fields[name])
        assert np.isfinite(array).all(), name
        assert float(np.abs(array).max()) > 0.0, name
    # A warm column has no frozen water, so SMFR3D is legitimately zero here;
    # what must be true is that SH2O tracks SMOIS rather than staying at zero.
    sh2o = _land(cp.asnumpy(driver.fields["sh2o"]), cfg)
    smois = _land(cp.asnumpy(driver.fields["smois"]), cfg)
    assert float(sh2o.min()) > 0.0
    assert np.all(sh2o <= smois + 1e-6)

    # The driver's ktau==1 repair has NOT run yet.
    for name in ("soilt1", "qvg", "qsg", "rhosnf"):
        assert float(np.abs(cp.asnumpy(driver.fields[name])).max()) == 0.0, name

    step(state, cfg)

    # ...and now it has.  SOILT1 outside 170..400 K is rebuilt from
    # SOILT/TSO(1) (:496-503), QSG from qsn(SOILT) (:513), QVG from
    # QSG*MAVAIL (:520), RHOSNF seeded to -1e3 (:552) and CHKLOWQ to 1.
    soilt1 = cp.asnumpy(driver.fields["soilt1"])
    assert 170.0 < float(soilt1.min()) and float(soilt1.max()) < 400.0
    for name in ("qvg", "qsg"):
        array = cp.asnumpy(driver.fields[name])
        assert float(array.min()) > 0.0, name
        assert float(array.max()) < 0.2, name
    assert float(cp.asnumpy(driver.fields["chklowq"]).max()) > 0.0


@requires_gpu
def test_out_of_identity_configurations_are_refused_before_the_run():
    """The gate has to fire at configuration time, not three hours in."""
    from woof.config import (RUC_OPTION_IDENTITY, RunConfig,
                              validate_run_config)

    base = dict(nx=6, ny=4, nz=20, dx=3000.0, dy=3000.0, ztop=12000.0,
                dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
                sf_sfclay_physics=1, sf_surface_physics=3,
                num_soil_layers=9, bl_pbl_physics=1)
    validate_run_config(RunConfig(**base))
    for name, admitted in RUC_OPTION_IDENTITY.items():
        with pytest.raises(ValueError, match="RUC option identity"):
            validate_run_config(RunConfig(**base, **{name: admitted + 1}))
    # The six-layer geometry WRF also defines is ADMITTED now.  It used to be
    # refused as a port blocker -- the forecast column was pinned to nine --
    # and the RUC_NZS lift retired that pin, so what a six-level request gets
    # is a warning naming the missing WRF forecast oracle, not a refusal.
    # The RUC_OPTION_IDENTITY loop above is a DIFFERENT guard and stays.
    six = RunConfig(**{**base, "num_soil_layers": 6})
    assert validate_run_config(six) is six
    # MYNN's first diagnosis is intentionally overwritten by
    # SFCDIAGS_RUCLSM under WRF's ownership sequence, so the coupled 5/5
    # suite is now an admitted RUC identity.
    validate_run_config(RunConfig(**{
        **base, "sf_sfclay_physics": 5, "bl_pbl_physics": 5}))


@requires_gpu
def test_a_noah_run_gains_no_ruc_arrays():
    """The negative control for the allocation gate.

    Twelve extra 2-D arrays, four diagnostics and two nine-level soil arrays
    in every Noah run would change that run's restart inventory, its VRAM
    budget and its health-descriptor count.
    """
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics

    cfg = RunConfig(nx=6, ny=4, nz=20, dx=3000.0, dy=3000.0, ztop=12000.0,
                    dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
                    sf_sfclay_physics=1, sf_surface_physics=2,
                    bl_pbl_physics=1)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 300.0 + 0.004 * np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, lambda z: 0.008 + 0.0 * z)
    driver = initialize_physics(state, cfg, glw=_IDEALISED_GLW)
    for name in (*RUC_STATE_2D, *RUC_STATE_3D, *RUC_DIAGNOSTICS_2D):
        assert name not in driver.fields, name
    assert driver.ruc_params is None


@requires_gpu
def test_the_vram_preflight_counts_the_ruc_arrays():
    """Under-counting VRAM is a correctness bar on this hardware."""
    from woof.core.preflight import physics_array_shapes

    noah = RunConfig(nx=6, ny=4, nz=20, dx=3000.0, dy=3000.0, ztop=12000.0,
                     dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
                     sf_sfclay_physics=1, sf_surface_physics=2,
                     bl_pbl_physics=1)
    # dataclasses.fields, not __dataclass_fields__: the latter also lists
    # ClassVar constants (RunConfig.constant_k_prandtl), which __init__ refuses.
    ruc = RunConfig(**{**{f.name: getattr(noah, f.name)
                          for f in dataclasses.fields(noah)},
                       "sf_surface_physics": 3, "num_soil_layers": 9})
    noah_shapes = physics_array_shapes(noah)
    ruc_shapes = physics_array_shapes(ruc)
    added = set(ruc_shapes) - set(noah_shapes)
    for name in (*RUC_STATE_2D, *RUC_STATE_3D, *RUC_DIAGNOSTICS_2D):
        assert f"fields/{name}" in added, name
    for name in RUC_STATE_3D:
        assert ruc_shapes[f"fields/{name}"] == (9, noah.ny, noah.nx), name
    # The soil column itself went from four levels to nine.
    for name in ("tslb", "smois", "sh2o", "smcrel"):
        assert noah_shapes[f"fields/{name}"] == (4, noah.ny, noah.nx)
        assert ruc_shapes[f"fields/{name}"] == (9, noah.ny, noah.nx)
    # And the estimate a launch gate compares against is strictly larger.
    words = lambda shapes: sum(int(np.prod(s)) for s in shapes.values())
    assert words(ruc_shapes) > words(noah_shapes)


# ---------------------------------------------------------------------------
# forecasting
# ---------------------------------------------------------------------------

@requires_gpu
def test_ruc_forecasts_and_stays_finite():
    from woof.core.dycore import step, stability_report

    state, cfg, driver = _build()
    before = {name: cp.asnumpy(driver.fields[name]).copy()
              for name in _CARRIED}

    for _ in range(20):
        step(state, cfg)
        assert not stability_report(state, cfg)["nan"]

    census = driver.last_ruc_census
    assert census is not None
    # RUC skips nothing: it dispatches water and sea ice internally, so the
    # census must account for the whole grid.
    assert census["land"] == cfg.ny * (cfg.nx - 2)
    assert census["water"] == cfg.ny * 2
    assert census["sea_ice"] == 0
    assert sum(census.values()) == cfg.ny * cfg.nx

    for name in _CARRIED:
        array = cp.asnumpy(driver.fields[name])
        assert np.isfinite(array).all(), name

    # The coupled quantities must have MOVED on land.  A runner that returned
    # its inputs would satisfy every finiteness assertion above.
    for name in ("tslb", "smois", "sh2o", "tsk", "hfx", "qfx", "lh",
                 "grdflx", "mavail", "qvg", "qsg", "sfcexc", "soilt1",
                 "tsnav", "sfcevp", "smstav", "smstot", "t2", "q2"):
        moved = _land(cp.asnumpy(driver.fields[name]), cfg) != _land(
            before[name], cfg)
        assert moved.any(), f"{name} did not move on any land column"

    # Physical ranges, not just finiteness.
    tsk = _land(cp.asnumpy(driver.fields["tsk"]), cfg)
    assert 250.0 < float(tsk.min()) and float(tsk.max()) < 340.0
    tslb = _land(cp.asnumpy(driver.fields["tslb"]), cfg)
    assert 250.0 < float(tslb.min()) and float(tslb.max()) < 340.0
    smois = _land(cp.asnumpy(driver.fields["smois"]), cfg)
    assert 0.0 <= float(smois.min()) and float(smois.max()) <= 1.0
    albedo = _land(cp.asnumpy(driver.fields["albedo"]), cfg)
    assert 0.05 < float(albedo.min()) and float(albedo.max()) < 0.9

    # The bottom soil level is pinned to TBOT every call (:1057) -- on LAND
    # only.  A water column takes the :824-847 arm, which sets the whole
    # profile to SOILT and CYCLEs before :1057 is reached, so it holds the
    # 294 K water skin temperature rather than the 288 K TBOT.  Asserting the
    # pin over the whole grid was wrong, and measuring both arms is stronger
    # than measuring either.
    tslb_bottom = cp.asnumpy(driver.fields["tslb"])[-1]
    tmn = cp.asnumpy(driver.fields["tmn"])
    np.testing.assert_array_equal(_land(tslb_bottom, cfg), _land(tmn, cfg))
    np.testing.assert_array_equal(
        tslb_bottom[..., cfg.nx - 2:],
        cp.asnumpy(driver.fields["tsk"])[..., cfg.nx - 2:])

    # The sensible heat flux follows the surface-air temperature difference on
    # every land column.  That ties the flux the driver wrote back to the
    # state the column solved for, which "HFX is positive" does not.
    from woof.core.physics import _prepare_atmosphere

    atmosphere = _prepare_atmosphere(state)
    t1 = _land(cp.asnumpy(atmosphere["temperature"][0]), cfg)
    soilt = _land(cp.asnumpy(driver.fields["tsk"]), cfg)
    hfx = _land(cp.asnumpy(driver.fields["hfx"]), cfg)
    assert float(np.abs(hfx).min()) > 0.0
    np.testing.assert_array_equal(np.sign(hfx), np.sign(soilt - t1))

    # SFCDIAGS_RUCLSM's clamp: T2 lies between TSK and the first level.
    t2 = _land(cp.asnumpy(driver.fields["t2"]), cfg)
    assert np.all(t2 >= np.minimum(soilt, t1) - 1e-4)
    assert np.all(t2 <= np.maximum(soilt, t1) + 1e-4)


@requires_gpu
@pytest.mark.parametrize("nzs", [6, 9])
def test_a_forecast_at_every_admitted_geometry_stays_physical(nzs):
    """RUC RUNS at both of WRF's soil geometries, and the answers are ordered.

    Not "compiles at six".  This integrates twenty RK3 steps of a mixed
    land/water grid with ``num_soil_layers`` resolved from the config, and
    then asks the state whether it is a soil column at all.

    Nine is parametrized alongside six on purpose, and the assertions are
    written once for both: every ordering claimed for six is claimed for the
    geometry that has a WRF oracle, at the same numbers, so a bar that six
    could only meet by being wrong in an interesting way would fail at nine
    first.

    WHAT THIS IS NOT.  It is not a correctness statement about six levels
    against WRF.  There is no six-level forecast oracle in this tree and
    none of these bounds is one: they are the properties a RUC soil column
    has to have to be a soil column, and passing them says the column ran
    and stayed physical, not that it is WRF's answer.
    """
    from woof.core.dycore import step, stability_report

    state, cfg, driver = _build(nzs=nzs)
    assert driver.ruc_params.num_soil_layers == nzs
    assert driver.fields["tslb"].shape == (nzs, cfg.ny, cfg.nx)
    before = {name: cp.asnumpy(driver.fields[name]).copy()
              for name in _CARRIED}

    for _ in range(20):
        step(state, cfg)
        assert not stability_report(state, cfg)["nan"]

    for name in _CARRIED:
        array = cp.asnumpy(driver.fields[name])
        assert np.isfinite(array).all(), name

    # Coupled, not merely called: a runner that returned its inputs would
    # satisfy every bound below.
    for name in ("tslb", "smois", "sh2o", "tsk", "hfx", "qfx", "lh",
                 "grdflx", "mavail", "qvg", "qsg", "soilt1", "t2", "q2"):
        moved = _land(cp.asnumpy(driver.fields[name]), cfg) != _land(
            before[name], cfg)
        assert moved.any(), f"{name} did not move on any land column"

    tslb = _land(cp.asnumpy(driver.fields["tslb"]), cfg)
    tmn = _land(cp.asnumpy(driver.fields["tmn"]), cfg)
    # LSMRUC:1057 pins the bottom level to TBOT on land, every call.
    np.testing.assert_array_equal(tslb[-1], tmn)
    # Below the top level the profile is monotone toward TBOT.  The TOP
    # level is excluded and that is not a convenience: it is the one level
    # the surface energy balance drives directly, so it may sit either side
    # of level 1 -- measured at 301.218 K under 301.398 K at six levels and
    # 300.480 under 301.403 at nine, on this fixture.
    assert np.all(np.diff(tslb[1:], axis=0) <= 0.0), (
        "the soil profile below the top level is not ordered toward TBOT")
    assert float(tslb.min()) >= float(tmn.min()) - 1e-4
    assert 250.0 < float(tslb.min()) and float(tslb.max()) < 340.0

    # Moisture inside the soil type's own porosity, read from the packaged
    # STAS-RUC row rather than hardcoded: DRYSMC <= SMOIS <= MAXSMC, and the
    # liquid fraction never exceeds the total.
    from woof.core.ruc import load_ruc_parameters

    row = load_ruc_parameters().soil.rows[_LOAM - 1].values
    drysmc, maxsmc = float(row[1]), float(row[3])
    smois = _land(cp.asnumpy(driver.fields["smois"]), cfg)
    sh2o = _land(cp.asnumpy(driver.fields["sh2o"]), cfg)
    assert float(smois.min()) >= drysmc, (smois.min(), drysmc)
    assert float(smois.max()) <= maxsmc, (smois.max(), maxsmc)
    assert np.all(sh2o <= smois + 1e-6)

    # The receipt must say which of the two claims this run can support.
    identity = driver.ruc_params.restart_identity()
    assert identity["num_soil_layers"] == nzs
    assert identity["soil_geometry_evidence"] == (
        "wrf-oracle" if nzs == 9 else "internal-consistency-only")
    assert len(identity["soil_level_depths_m"]) == nzs


@requires_gpu
def test_two_soil_geometries_run_in_one_process_without_touching_each_other():
    """Two domains, two soil counts, one process, one card, interleaved.

    The tier is a COMPILE-TIME macro, so the question this answers is not
    rhetorical: if the module cache were keyed on the module name alone, the
    second domain would silently get the first domain's translation unit and
    read past every soil scratch array in the frame.  ``ruc_module_defines``
    returning ``()`` at nine and ``(("RUC_NZS", 6),)`` at six is what keys
    them apart, and this is the test that says so on the hardware.

    Three claims, and the third is the one that matters:

    1. both domains step, and neither goes non-finite;
    2. each keeps its own geometry -- its own zs, its own array shapes;
    3. the nine-level domain is BIT-IDENTICAL to the same domain run with no
       six-level domain in the process at all.  Without (3) this would show
       that two geometries coexist, not that the older one is unaffected,
       and unaffected is the whole bar this lane lives on.
    """
    from woof.core.dycore import step, stability_report

    state6, cfg6, driver6 = _build(nzs=6)
    state9, cfg9, driver9 = _build(nzs=9)
    for _ in range(12):
        step(state6, cfg6)
        step(state9, cfg9)
        assert not stability_report(state6, cfg6)["nan"]
        assert not stability_report(state9, cfg9)["nan"]

    assert driver6.fields["tslb"].shape == (6, cfg6.ny, cfg6.nx)
    assert driver9.fields["tslb"].shape == (9, cfg9.ny, cfg9.nx)
    np.testing.assert_array_equal(
        np.asarray(driver6.ruc_params.zs, np.float32),
        np.array([0.0, 0.05, 0.20, 0.40, 1.60, 3.00], np.float32))
    assert len(driver9.ruc_params.zs) == 9
    for driver in (driver6, driver9):
        for name in _CARRIED:
            assert np.isfinite(cp.asnumpy(driver.fields[name])).all(), name

    # The isolation control.
    state_alone, cfg_alone, driver_alone = _build(nzs=9)
    for _ in range(12):
        step(state_alone, cfg_alone)
    differing = [name for name in _CARRIED
                 if not np.array_equal(cp.asnumpy(driver_alone.fields[name]),
                                       cp.asnumpy(driver9.fields[name]))]
    assert differing == [], (
        "the nine-level domain's answer changed because a six-level domain "
        f"existed in the same process: {differing}")


@requires_gpu
def test_water_columns_take_the_water_arm_and_sea_ice_takes_its_own():
    """Three arms of LSMRUC's per-column dispatch, each proved separately."""
    from woof.core.dycore import step

    state, cfg, driver = _build(nx=8, ny=6, ice_rows=2)
    for _ in range(4):
        step(state, cfg)
    census = driver.last_ruc_census
    assert census["water"] == cfg.ny * 2
    assert census["sea_ice"] == 2 * (cfg.nx - 2)
    assert census["land"] == cfg.ny * cfg.nx - census["water"] - \
        census["sea_ice"]

    # :824-847 water: soil moisture forced to 1, snow cleared, CHKLOWQ to 1.
    water = slice(cfg.nx - 2, None)
    np.testing.assert_array_equal(
        cp.asnumpy(driver.fields["smois"])[..., water],
        np.ones((_NSOIL, cfg.ny, 2), np.float32))
    np.testing.assert_array_equal(
        cp.asnumpy(driver.fields["snow"])[..., water],
        np.zeros((cfg.ny, 2), np.float32))
    np.testing.assert_array_equal(
        cp.asnumpy(driver.fields["chklowq"])[..., water],
        np.ones((cfg.ny, 2), np.float32))

    # :862-895 sea ice: ZNT forced to 0.011, SNOALB to 0.75, SMFR3D to 1,
    # SH2O to 0, and TSO capped at 271.4 K.
    ice = (slice(0, 2), slice(0, cfg.nx - 2))
    np.testing.assert_array_equal(
        cp.asnumpy(driver.fields["snoalb"])[ice],
        np.full((2, cfg.nx - 2), 0.75, np.float32))
    np.testing.assert_array_equal(
        cp.asnumpy(driver.fields["smfr3d"])[:, ice[0], ice[1]],
        np.ones((_NSOIL, 2, cfg.nx - 2), np.float32))
    tso_ice = cp.asnumpy(driver.fields["tslb"])[:-1, ice[0], ice[1]]
    assert float(tso_ice.max()) <= 271.4 + 1e-5

    # module_surface_driver.F:3453-3459: the seam's own ALBBCK override, which
    # is outside the FRACTIONAL_SEAICE block and therefore always applies.
    np.testing.assert_array_equal(
        cp.asnumpy(driver.fields["albbck"])[ice],
        np.full((2, cfg.nx - 2), SEAICE_ALBEDO_DEFAULT, np.float32))
    assert float(cp.asnumpy(driver.fields["xice"])[ice].min()) >= \
        XICE_THRESHOLD


@requires_gpu
@pytest.mark.parametrize("lakemodel", [0, 1])
def test_lakemask_bypass_requires_enabled_lake_model(lakemodel):
    """The WRF lake GOTO needs both the model selector and lake mask."""
    from woof.core import physics
    from woof.core.ruc_runtime import ruc_lsm_step
    from woof.core.surface_forcing import SurfacePrecipitationForcing

    state, cfg, driver = _build(nx=4, ny=2, water_columns=0)
    driver.fields["lakemask"][...] = cp.float32(0.0)
    driver.fields["lakemask"][0, 0] = cp.float32(1.0)
    atmosphere = physics._prepare_atmosphere(state)
    driver._run_sfclay(atmosphere, cfg)
    names = ("tsk", "smois", "sh2o", "canwat", "snow")
    before = {name: driver.fields[name].copy() for name in names}
    census = ruc_lsm_step(
        driver.fields, atmosphere,
        params=driver.ruc_params,
        precipitation=SurfacePrecipitationForcing.from_fields(driver.fields),
        dt=cfg.dt, itimestep=1, mosaic_lu=cfg.mosaic_lu,
        mosaic_soil=cfg.mosaic_soil, flag_sm_adj=cfg.flag_sm_adj,
        spp_lsm=cfg.spp_lsm, lakemodel=lakemodel)

    if lakemodel:
        for name in names:
            cp.testing.assert_array_equal(
                driver.fields[name][..., 0, 0], before[name][..., 0, 0])
    else:
        assert any(not bool(cp.array_equal(
            driver.fields[name][..., 0, 0], before[name][..., 0, 0]))
                   for name in names)
    assert any(not bool(cp.array_equal(
        driver.fields[name][..., 0, 1], before[name][..., 0, 1]))
               for name in names)
    assert census == {"land": 8 - lakemodel, "water": 0, "lake": lakemodel, "sea_ice": 0}


@requires_gpu
def test_fractional_sea_ice_reblends_two_distinct_xice_values():
    """The full-ice TSK returned by RUC is retained and WRF-reblended."""
    from woof.core.diagnostics import update_diagnostics
    from tools.surface_forcing_wrf461_oracle.transcribe_surface_forcing import (
        ruc_fractional_post,
    )

    state, cfg, driver = _build(nx=4, ny=1, water_columns=0, fractional_seaice=1)
    xice = np.array([0.55, 0.75], np.float32)
    driver.fields["xice"][0, :2] = cp.asarray(xice)
    driver.fields["ivgtyp"][0, :2] = _ICE
    driver.fields["tsk_save"][0, :2] = cp.asarray(
        np.array([263.0, 266.0], np.float32))
    driver.fields["tsk_sea"][0, :2] = cp.asarray(
        np.array([275.0, 278.0], np.float32))
    driver.fields["tsk"][0, :2] = cp.asarray(ruc_fractional_post(
        ice=cp.asnumpy(driver.fields["tsk_save"][0, :2]),
        sea=cp.asnumpy(driver.fields["tsk_sea"][0, :2]), xice=xice))
    ice_albedo = np.array([0.62, 0.68], np.float32)
    ice_emiss = np.array([0.96, 0.97], np.float32)
    driver.fields["albedo"][0, :2] = cp.asarray(ruc_fractional_post(
        ice=ice_albedo, sea=np.float32(0.08), xice=xice))
    driver.fields["emiss"][0, :2] = cp.asarray(ruc_fractional_post(
        ice=ice_emiss, sea=np.float32(0.98), xice=xice))

    update_diagnostics(state)
    driver.compute(state, cfg)
    expected = ruc_fractional_post(
        ice=cp.asnumpy(driver.fields["tsk_save"][0, :2]),
        sea=cp.asnumpy(driver.fields["tsk_sea"][0, :2]), xice=xice)
    np.testing.assert_array_equal(
        cp.asnumpy(driver.fields["tsk"][0, :2]), expected)


@requires_gpu
def test_ruc_consumes_radiation_time_gsw_unchanged_between_calls(
        monkeypatch):
    """Live SWDOWN/albedo changes cannot rewrite the carried GSW."""
    from woof.core import physics
    from woof.core.physics import RadiationResult

    radiation_calls = []

    def radiation(**kwargs):
        state = kwargs["state"]
        radiation_calls.append(float(state.elapsed_seconds))
        nz, ny, nx = state.p.shape
        zeros = cp.zeros((nz, ny, nx), cp.float32)
        value = cp.float32(111.0 if len(radiation_calls) == 1 else 222.0)
        return RadiationResult(
            zeros, zeros, cp.full((ny, nx), 700.0, cp.float32),
            cp.full((ny, nx), 340.0, cp.float32),
            gsw=cp.full((ny, nx), value, cp.float32))

    state, cfg, driver = _build(
        nx=4, ny=2, water_columns=1, dt=10.0,
        radiation=radiation, ra_physics=90, radt_minutes=1.0)
    seen = []

    def capture(fields, _atmosphere, **_kwargs):
        seen.append(cp.asnumpy(fields["gsw"]).copy())
        return {"land": 6, "water": 2, "lake": 0, "sea_ice": 0}

    monkeypatch.setattr(physics, "ruc_lsm_step", capture)
    driver.compute(state, cfg)
    driver.fields["swdown"][...] = cp.float32(900.0)
    driver.fields["albedo"][...] = cp.float32(0.9)
    state.elapsed_seconds = 10.0
    driver.compute(state, cfg)
    state.elapsed_seconds = 60.0
    driver.compute(state, cfg)

    assert radiation_calls == [0.0, 60.0]
    np.testing.assert_array_equal(seen[0], np.full((2, 4), 111.0,
                                                   np.float32))
    np.testing.assert_array_equal(seen[1], seen[0])
    assert not np.array_equal(
        seen[1], np.full((2, 4), 900.0 * (1.0 - 0.9), np.float32))
    np.testing.assert_array_equal(seen[2], np.full((2, 4), 222.0,
                                                   np.float32))


@requires_gpu
def test_a_snowpack_column_runs_the_snow_branch():
    """Without a pack the whole snow half of SFCTMP is never entered."""
    from woof.core.dycore import step

    state, cfg, driver = _build(nx=6, ny=4, snow_mm=60.0, snow_depth_m=0.30)
    for _ in range(10):
        step(state, cfg)
    snowc = _land(cp.asnumpy(driver.fields["snowc"]), cfg)
    assert float(snowc.max()) > 0.0
    for name in ("snowh", "tsnav", "soilt1", "rhosnf", "snowfallac", "acsnom"):
        array = cp.asnumpy(driver.fields[name])
        assert np.isfinite(array).all(), name
    # The snow-covered albedo must exceed the bare-soil table value: SOILVEGIN
    # gives grassland 0.19 and SNOALB is 0.75-ish, so a blended albedo below
    # the bare value would mean the snow fraction never reached ALB.
    albedo = _land(cp.asnumpy(driver.fields["albedo"]), cfg)
    assert float(albedo.max()) > 0.19
    # TSNAV is the snow-pack mean temperature in CELSIUS (:4410 subtracts
    # 273.15), which is a unit a reader will get wrong if nothing pins it.
    tsnav = _land(cp.asnumpy(driver.fields["tsnav"]), cfg)
    assert -80.0 < float(tsnav.min()) and float(tsnav.max()) < 60.0


@requires_gpu
def test_the_snow_mass_budget_closes_and_a_cold_pack_persists():
    """Where the one-hour forecast's snow pack went, and why it is not a bug.

    The admitting forecast started with SNOW=15 mm / SNOWH=0.08 m and its
    first wrfout frame, at t=600 s, already read SNOW=0.  That looks like a
    dropped pack, and RUC_SMELT reading 0 in the same frame does not clear it
    -- RUC_SMELT is an instantaneous RATE at frame time, not an accumulator.

    Measured step by step, the pack MELTS: ACSNOM reaches 15.0020 mm against
    15.0 mm of initial SWE over eight 6 s steps, and the budget closes to
    0.03%.  What was wrong was the INITIAL STATE, not the scheme -- a 15 mm
    pack was handed a 303 K skin temperature and a 295..303 K soil column, and
    a warm soil layer holds enough heat to melt it (2.5 MJ m-3 K-1 x 0.05 m x
    27 K is 3.4 MJ m-2 against the 5.0 MJ m-2 the pack needs).  RUC resolving
    that inconsistency immediately is correct behaviour, and TSK dropping
    303 -> 292.5 K in the first step is it happening.

    So the assertion is the CONSERVATION, which holds regardless of how fast
    the melt runs, plus the control that says the melt is driven by the warm
    ground rather than by the scheme leaking mass: an identical pack over
    subfreezing air and soil must keep its SWE and accumulate exactly no melt.

    The fast melt above is the WRF v4.6.1 snow scheme's (``ruc_snow =
    "wrf_461"``), whose bottom-melt cap applies only to light packs over
    ground below 283 K, so a 303 K soil melts the pack from below at once;
    that leg runs it by name.  Under explicit ``wrf_45`` (the operational
    RAP/HRRR branch) bottom melt is capped at 5.8e-9 m/s whatever the
    ground temperature, so the same pack persists through these 144 s; the
    budget must close there too.
    """
    from woof.core.dycore import step

    for lineage in ("wrf_461", "wrf_45"):
        state, cfg, driver = _build(nx=6, ny=4, nz=20, snow_mm=15.0,
                                    snow_depth_m=0.08, soil_moisture=0.30,
                                    ruc_snow=lineage)
        swe0 = float(_land(cp.asnumpy(driver.fields["snow"]), cfg).mean())
        assert abs(swe0 - 15.0) < 1e-3
        for _ in range(12):
            step(state, cfg)
        swe = _land(cp.asnumpy(driver.fields["snow"]), cfg)
        melt = _land(cp.asnumpy(driver.fields["acsnom"]), cfg)
        # WRF accumulates SFCEVP twice per land step (module_sf_ruclsm.F
        # :1095 and :1116), so its water budget only closes on half of it.
        # WOOF counts it once (development branch), so the budget closes on
        # SFCEVP itself; a doubled SFCEVP fails this by the evaporated mass.
        evap = _land(cp.asnumpy(driver.fields["sfcevp"]), cfg)
        closure = float((swe + melt + evap).mean()) - swe0
        assert abs(closure) < 0.05 * swe0, (
            f"{lineage}: snow mass budget does not close: SWE "
            f"{swe.mean():.4f} + melt {melt.mean():.4f} + evap "
            f"{evap.mean():.4f} against {swe0} mm initial, residual "
            f"{closure:.4f} mm")
        if lineage == "wrf_461":
            assert float(melt.mean()) > 0.9 * swe0, (
                "the pack vanished without ACSNOM accounting for it, which "
                "is a dropped pack rather than a melted one")
        else:
            assert float(swe.mean()) > 0.9 * swe0, (
                "under the capped bottom melt the pack should persist; "
                f"SWE fell to {swe.mean():.4f} mm")

    # The control: the same pack in a subfreezing column keeps its mass and
    # melts nothing at all, so the melt above is the warm ground and not the
    # scheme losing water.
    state, cfg, driver = _build(nx=6, ny=4, nz=20, snow_mm=15.0,
                                snow_depth_m=0.08, frozen=True,
                                soil_moisture=0.30)
    for _ in range(12):
        step(state, cfg)
    cold_swe = _land(cp.asnumpy(driver.fields["snow"]), cfg)
    cold_melt = _land(cp.asnumpy(driver.fields["acsnom"]), cfg)
    assert float(np.abs(cold_melt).max()) == 0.0, (
        f"a subfreezing pack melted {cold_melt.max()} mm")
    assert float(np.abs(cold_swe - swe0).max()) < 0.01, (
        f"a subfreezing pack's SWE drifted to [{cold_swe.min()},"
        f"{cold_swe.max()}] from {swe0}")
    assert float(_land(cp.asnumpy(driver.fields["snowc"]), cfg).min()) > 0.0

    # RHOSNF is the -1e3 SENTINEL, not a density, until snow actually FALLS.
    # LSMRUC seeds it at :552 in the ktau==1 block and only overwrites it from
    # rhosnfall, so an unrained run reports -1000 kg m-3 for the whole
    # forecast.  Pinned because a user meeting -1000 in a wrfout would
    # reasonably read it as corruption.
    assert float(cp.asnumpy(driver.fields["rhosnf"]).max()) == -1000.0


# ---------------------------------------------------------------------------
# the two open leads
# ---------------------------------------------------------------------------

@requires_gpu
def test_udrunoff_is_zero_because_no_level_oversaturates_and_can_be_nonzero():
    """The ``udrunoff``-is-always-zero lead, settled from the WRF source.

    ``:1041`` is ``udrunoff = udrunoff + runoff2*dt*1000``, and ``runoff2``
    comes from ``SOILMOIST``.  Reading that routine says exactly when it is
    nonzero: ``:5901`` zeroes it on every call, and the ONLY accumulation is
    at ``:6071``/``:6073``, inside ``else if(qq.gt.dqm)`` for levels
    ``k=2..nzs`` -- a SUB-SURFACE level whose implicit solution overshoots
    saturation.  The top level's saturation overshoot goes to ``runoff``
    (surface) instead, at ``:6048``; WRF even has the ``runoff2`` version of
    that line commented out at ``:6043``.

    So ``udrunoff == 0`` in all 48 unsaturated fixture cases is CORRECT and
    not a port defect: none of them drives a deep level past ``dqm``.

    Two things the first attempt at this test got wrong, both found by
    measurement, both worth recording because they are counter-intuitive:

    * RAIN DOES NOT DRIVE ``runoff2``.  Swept over rainbl in
      (0, 1, 10, 40, 200) mm the value is bit-identical -- because
      ``infmax`` caps infiltration at ``:6009`` and every excess millimetre is
      sent to ``runoff`` (surface) at ``:6012``, so water cannot reach a deep
      level fast enough to oversaturate it.  What drives ``runoff2`` is the
      column's INITIAL saturation.
    * the threshold is ``smois >= MAXSMC``, not ``smois > dqm``.  Measured on
      STAS-RUC loam: 0.45 gives exactly zero and 0.451 -- MAXSMC itself, a
      fully saturated column -- gives nonzero.

    The strongest form of the claim is available here and is the one asserted:
    the saturated column is given NO RAIN AT ALL, so it has ``sfcrunoff``
    exactly zero and ``udrunoff`` nonzero -- the exact inverse of the fixture
    pattern that opened the lead (``sfcrunoff`` nonzero in 14 of 48,
    ``udrunoff`` zero in all 48).  Two accumulators that share ``:1040``'s
    shape cannot both be explained by one wiring mistake if each can be made
    to move while the other stays at zero.
    """
    from woof.core.dycore import step

    # Half A: unsaturated, no rain.  Both accumulators must stay exactly zero.
    state, cfg, driver = _build(nx=6, ny=4, nz=20, soil_moisture=0.28)
    for _ in range(10):
        step(state, cfg)
    assert float(np.abs(_land(cp.asnumpy(driver.fields["udrunoff"]),
                              cfg)).max()) == 0.0
    assert float(np.abs(_land(cp.asnumpy(driver.fields["ruc_runoff2"]),
                              cfg)).max()) == 0.0

    # Half B: a fully saturated loam column, still with no rain.  MAXSMC comes
    # off the packaged STAS-RUC table so this stays "saturated" by definition
    # rather than by a remembered constant.
    saturated = _maxsmc(_LOAM)
    assert 0.44 < saturated < 0.46, saturated
    # The witness column runs v4.6.1's SOILPROP, on which it was measured.
    # MEASURED under the default wrf_45 lineage: this saturated column
    # leaves runoff2 at exactly zero in every land cell, so the accumulator
    # under test would never be written.  The wiring is the same for both.
    state, cfg, driver = _build(nx=6, ny=4, nz=20, soil_moisture=saturated,
                                ruc_soilprop="wrf_461")
    step(state, cfg)

    runoff2 = _land(cp.asnumpy(driver.fields["ruc_runoff2"]), cfg)
    udrunoff = _land(cp.asnumpy(driver.fields["udrunoff"]), cfg)
    sfcrunoff = _land(cp.asnumpy(driver.fields["sfcrunoff"]), cfg)
    assert float(runoff2.max()) > 0.0, (
        "WRF's own condition for runoff2 (qq>dqm at a level k>=2) was never "
        "met, so this half proves nothing about the accumulator")

    # ``:1041`` exactly, in the port's own float32 grouping
    # (woof/core/ruc.py: fl32(fl32(runoff2*dt) * 1000)).  An approximate
    # comparison here would pass for a wiring that dropped the 1000 or used
    # cfg.dt where the surface interval belongs.
    dt = np.float32(driver.bldt_seconds)
    expected = (np.float32(runoff2 * dt) * np.float32(1000.0)).astype(
        np.float32)
    np.testing.assert_array_equal(
        udrunoff, expected,
        err_msg="udrunoff is not udrunoff + runoff2*dt*1000 (:1041)")

    # And the inverse of the fixture pattern: no rain, so the SURFACE
    # accumulator stayed at zero while the underground one moved.
    np.testing.assert_array_equal(
        sfcrunoff, np.zeros_like(sfcrunoff),
        err_msg="an unrained column produced surface runoff")

    # The control in the other direction: rain moves sfcrunoff and leaves
    # udrunoff alone on the unsaturated column, so neither accumulator is
    # simply mirroring the other.
    state, cfg, driver = _build(nx=6, ny=4, nz=20, soil_moisture=0.28)
    for _ in range(4):
        driver.set_forcing(rainbl=np.full((cfg.ny, cfg.nx), 40.0, np.float32))
        step(state, cfg)
    assert float(_land(cp.asnumpy(driver.fields["sfcrunoff"]),
                       cfg).max()) > 0.0
    assert float(np.abs(_land(cp.asnumpy(driver.fields["udrunoff"]),
                              cfg)).max()) == 0.0


@requires_gpu
def test_ilnb_is_defined_rather_than_inherited_from_the_previous_column():
    """WRF's uninitialised ``ilnb``, and woof's defined answer.

    ``sfctmp`` declares ``ilnb`` as a plain local (``:1385``) and never
    initialises it; both callees assign it only inside
    ``if(snhei.ge.snth)``, and both read it at ``if(ilnb.gt.1)`` under the
    wider ``if(snhei.gt.0.)`` (``:4410``, ``:5716``).  On
    ``0 < snhei < snth`` the read is undefined by the Fortran standard and in
    practice returns the previous COLUMN's value, which makes ``tsnav``
    depend on grid traversal order.

    woof does not reproduce that.  The runtime passes ``ilnb_chain=False``,
    so the answer is order-independent.  This test proves the divergence is
    real by running the same physical column set in two different memory
    ORDERS and requiring the same answer -- which the chained form cannot
    give -- and then shows the chained form is still available and DOES
    differ, so the claim is a measurement rather than an assertion.
    """
    from woof.core.ruc import ruc_land_surface_step
    from woof.core.ruc_runtime import (C1SN, C2SN, ISNCOVR_OPT,
                                        RucRuntimeParameters)

    params = RucRuntimeParameters()
    nx = 4
    # Two thin packs (below snth) flanked by two deep ones.  The thin columns
    # are the ones that read ilnb; the deep ones are what a chain would feed
    # them.
    snow_mm = np.array([80.0, 2.0, 80.0, 2.0], np.float32)
    snowh = np.array([0.40, 0.004, 0.40, 0.004], np.float32)

    def call(order, *, chain):
        values = {}
        for name in ("soilmois", "sh2o"):
            values[name] = np.full((_NSOIL, nx), 0.25, np.float32)
        values["tso"] = np.full((_NSOIL, nx), 268.0, np.float32)
        values["smfr3d"] = np.full((_NSOIL, nx), 0.05, np.float32)
        values["keepfr3dflag"] = np.zeros((_NSOIL, nx), np.float32)
        scalars = {
            "snow": snow_mm[order], "snowh": snowh[order],
            "snowc": np.ones(nx, np.float32), "canwat": 0.0,
            "snoalb": 0.7, "alb": 0.6, "emiss": 0.98, "lai": 2.0,
            "mavail": 0.5, "sfcexc": 0.02, "z0": 0.05, "znt": 0.05,
            "soilt": 268.0, "hfx": 0.0, "qfx": 0.0, "lh": 0.0,
            "sfcevp": 0.0, "sfcrunoff": 0.0, "udrunoff": 0.0,
            "acrunoff": 0.0, "grdflx": 0.0, "acsnow": 0.0, "snom": 0.0,
            "qvg": 0.002, "qcg": 0.0, "dew": 0.0, "qsfc": 0.002,
            "qsg": 0.002, "chklowq": 1.0, "soilt1": 268.0, "tsnav": -6.0,
            "smavail": 0.0, "smmax": 0.0, "rhosnf": 200.0,
            "precipfr": 0.0, "snowfallac": 0.0,
            "z3d": 40.0, "p8w": 95000.0, "t3d": 266.0, "qv3d": 0.002,
            "qc3d": 0.0, "rho3d": 1.2, "rainbl": 0.0, "frzfrac": 1.0,
            "glw": 250.0, "gsw": 100.0, "chs": 0.02, "flqc": 0.02,
            "flhc": 20.0, "albbck": 0.2, "xland": 1.0, "xice": 0.0,
            "tbot": 275.0, "shdmin": 10.0, "shdmax": 80.0, "vegfra": 60.0,
        }
        for name, value in scalars.items():
            values[name] = (value if isinstance(value, np.ndarray)
                            else np.full(nx, value, np.float32))
        return ruc_land_surface_step(
            values, dt=60.0, ktau=2, zs=params.zs,
            ivgtyp=np.full(nx, _GRASSLAND, np.int32),
            isltyp=np.full(nx, _LOAM, np.int32),
            em_core=0,
            ilnb=DEFINED_ILNB, ilnb_chain=chain,
            c1sn=C1SN, c2sn=C2SN, isncovr_opt=ISNCOVR_OPT,
            mminlu=params.dataset_identifier, parameters=params.bundle)

    forward = np.arange(nx)
    reversed_order = forward[::-1]
    defined_f = np.asarray(call(forward, chain=False).tsnav)
    defined_r = np.asarray(call(reversed_order, chain=False).tsnav)[::-1]
    np.testing.assert_array_equal(
        defined_f, defined_r,
        err_msg="the defined form still depends on column order")

    # The falsification: WRF's chained form is order-dependent on the same
    # columns, so the test above is measuring something.
    chained_f = np.asarray(call(forward, chain=True).tsnav)
    chained_r = np.asarray(call(reversed_order, chain=True).tsnav)[::-1]
    assert not np.array_equal(chained_f, chained_r), (
        "the chained form gave the same answer in both orders, so this "
        "fixture does not reach the uninitialised ilnb read and the "
        "divergence above is untested")
    # And the two forms differ from each other, which is the divergence.
    assert not np.array_equal(defined_f, chained_f)


# ---------------------------------------------------------------------------
# restart
# ---------------------------------------------------------------------------

@requires_gpu
def test_a_ruc_restart_round_trips_and_reproduces_the_next_step(tmp_path):
    from woof.core.dycore import step
    from woof.io.restart import restore_restart, write_restart

    state, cfg, driver = _build(nx=6, ny=4, nz=20)
    for _ in range(6):
        step(state, cfg)

    path = tmp_path / "ruc.npz"
    write_restart(path, state, cfg)
    fresh_state, fresh_cfg, fresh_driver = _build(nx=6, ny=4, nz=20)
    restore_restart(path, fresh_state, fresh_cfg)

    for name in _CARRIED:
        np.testing.assert_array_equal(
            cp.asnumpy(fresh_driver.fields[name]),
            cp.asnumpy(driver.fields[name]), err_msg=name)

    # Storage identity is not trajectory identity.  Step both and compare.
    step(state, cfg)
    step(fresh_state, fresh_cfg)
    for name in _CARRIED:
        np.testing.assert_array_equal(
            cp.asnumpy(fresh_driver.fields[name]),
            cp.asnumpy(driver.fields[name]), err_msg=f"after/{name}")
    for name in ("u", "v", "w", "thp"):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(fresh_state, name)),
            cp.asnumpy(getattr(state, name)), err_msg=f"after/{name}")


#: Every array the round trip compares, so a perturbation is scored against
#: the same inventory the identity check uses rather than a shorter one that
#: would flatter it.
_WATCHED = _CARRIED


def _next_step_delta(tmp_path, field, index, *, add=None, scale=None,
                     set_to=None, build=None):
    """Perturb one carrier after a restore and report which fields moved.

    Returns ``(moved, others)``: everything that changed, and everything that
    changed OTHER than the perturbed field itself.  The split matters.  Eleven
    of RUC's carriers are accumulators or write-once receipts that retain a
    nudge without it ever reaching the next step, so ``len(moved) >= 1`` is
    satisfied by a field that influenced nothing.  ``others`` is the claim.
    """
    from woof.core.dycore import step
    from woof.io.restart import restore_restart, write_restart

    build = build or (lambda: _build(nx=6, ny=4, nz=20))
    state, cfg, driver = build()
    for _ in range(6):
        step(state, cfg)
    path = tmp_path / "ruc.npz"
    write_restart(path, state, cfg)
    step(state, cfg)
    reference = {name: cp.asnumpy(driver.fields[name]).copy()
                 for name in _WATCHED}

    fresh_state, fresh_cfg, fresh_driver = build()
    restore_restart(path, fresh_state, fresh_cfg)
    nudged = cp.asnumpy(fresh_driver.fields[field])
    if scale is not None:
        nudged[index] = np.float32(nudged[index] * scale)
    elif set_to is not None:
        nudged[index] = np.float32(set_to)
    else:
        nudged[index] = np.float32(nudged[index] + add)
    fresh_driver.fields[field][...] = cp.asarray(nudged)
    step(fresh_state, fresh_cfg)
    moved = sorted(
        name for name in _WATCHED
        if not np.array_equal(cp.asnumpy(fresh_driver.fields[name]),
                              reference[name]))
    return moved, [name for name in moved if name != field]


#: The threshold the falsification tests use.  Deliberately NOT a count of
#: watched fields: the Noah-MP lane's ">= 5 of 20 move" held on its own grid
#: and only 2 of 51 moved on a verifier's, because how many fields a
#: perturbation reaches is a property of the grid, the scheme's coupling and
#: the length of the watch list -- none of which the claim depends on.  What
#: is configuration-independent is the CLAIM, that the restored carrier feeds
#: the next step, and its minimal witness is "at least two OTHER carriers
#: moved".  One is not enough: a nudged TSLB reaches GRDFLX alone at any size
#: down to 0.001 K, because ``grdflx = -sflx`` is assigned from the soil solve
#: unconditionally, so a single mover does not distinguish a coupled column
#: from one bookkeeping assignment.
#:
#: Measured margins over this bar, unfrozen grid / frozen grid:
#: SMOIS +1e-5 reaches 16 / 20 others and is flat over three decades of nudge
#: size; TSLB +0.05 K reaches 8 / 21.
_MIN_OTHERS = 2

#: TSLB's measured propagation floor on an UNFROZEN grid, in kelvin.  Below
#: this a nudge reaches GRDFLX and nothing else: measured 1 other field at
#: 0.001, 0.005 and 0.01 K, then 8 at 0.05 K and 10 at 0.1 K.  The floor is a
#: property of the warm column, not of the carrier -- on the frozen grid even
#: 0.001 K reaches 19 others, because there the top soil level also drives the
#: freezing curve.  Recorded rather than rounded up out of sight, because a
#: reader who nudges TSLB by a ULP and sees nothing move should be able to
#: find out here that the wiring is fine and the resolution is not.
_TSLB_PROPAGATION_FLOOR_K = 0.05


@requires_gpu
def test_the_restart_identity_check_can_fail(tmp_path):
    """A check that cannot fail is decoration.

    The perturbation target and its size are both MEASURED, not guessed.  The
    first version of this test added ONE ULP to TSLB and the next step came
    back bit identical on every watched field, which reads like a broken
    restart and is not one: the sweep behind
    :data:`woof.core.ruc_runtime.RUC_MEASURED_LIVE_CARRIERS` nudged all 43
    carriers in turn, and TSLB is live -- a single ULP of a 302 K soil
    temperature simply is not, because on a warm column the top soil level
    reaches the surface energy balance through terms that absorb a 3e-5
    relative change in one 12 s float32 step.

    SMOIS leads here rather than TSLB because it is the robust witness: it
    reaches 16 other carriers at a nudge of 1e-5 m3 m-3 and the count is flat
    from 1e-5 to 2e-2, so the assertion does not sit near a threshold.  TSLB
    follows, at its measured floor, so the soil TEMPERATURE path is checked
    too and the floor itself is pinned.
    """
    moved, others = _next_step_delta(tmp_path, "smois", (0, 0, 0), add=1.0e-5)
    assert "smois" in moved, (
        "a 1e-5 m3 m-3 change in the top soil level did not even survive "
        "into SMOIS, so the perturbation never reached the driver")
    assert len(others) >= _MIN_OTHERS, others

    t_moved, t_others = _next_step_delta(
        tmp_path, "tslb", (0, 0, 0), add=_TSLB_PROPAGATION_FLOOR_K)
    assert "tslb" in t_moved
    assert len(t_others) >= _MIN_OTHERS, t_others

    # And the floor is real, not decoration: an order of magnitude below it
    # the same carrier reaches GRDFLX and nothing else.  This is the half that
    # keeps the docstring accurate -- if RUC's soil coupling ever became
    # sensitive enough that 0.005 K propagated, the comment above would be
    # stale and this would say so.
    _, small_others = _next_step_delta(tmp_path, "tslb", (0, 0, 0), add=0.005)
    assert small_others == ["grdflx"], (
        f"the measured TSLB propagation floor has moved: 0.005 K now reaches "
        f"{small_others}, so _TSLB_PROPAGATION_FLOOR_K is stale")


@requires_gpu
def test_a_frozen_soil_carrier_perturbation_also_breaks_the_next_step(
        tmp_path):
    """SMFR3D and KEEPFR3DFLAG are the two carriers only RUC has.

    A restart that dropped them would still reproduce every generic soil
    field, so the round trip must be shown to have teeth for RUC's OWN state.
    Getting that witness took three corrections, all of them measurements:

    * the column must be FROZEN.  ``soilprop`` rebuilds ``soilice`` from TSO
      and SOILMOIS through the freezing curve (``:2695-2703``) and reads the
      restored SMFR3D only as the cap
      ``soilice(k)=min(soilice(k),smfrkeep(k))`` at ``:2704-2707``, inside
      ``if(keepfr(k).eq.1.)``; ``keepfr`` is assigned 1 at ``:2744`` only
      inside ``if (soilice(k).gt.0.)``.  On the warm grid this file's other
      tests use, 0 of 216 soil cells have KEEPFR3DFLAG==1, so SMFR3D is not
      merely inert there -- it is never read at all.
    * the cell must be one where ``KEEPFR3DFLAG == 1``.  On the frozen grid
      that is 16 of 216 cells and they are all at soil level 1, so the
      obvious ``(0, 0, 0)`` index is the one cell shape that cannot work.
    * the nudge must be DOWNWARD.  The read is a ``min()``; raising a cap
      that is not binding changes nothing.  Measured at the same cell:
      scaling SMFR3D to 0.1x reaches 21 other carriers, adding +0.05 reaches
      3.
    """
    from woof.core.dycore import step

    def frozen():
        # KEEPFR3DFLAG reaches 1 only where soil temperature and moisture
        # both rise in a step (:2742-2744).  The 16 witness cells were
        # measured on v4.6.1's SOILPROP.  MEASURED under the default wrf_45
        # lineage: no cell of this grid reaches 1 in six steps, so the
        # carrier would never be read.  The round trip is the same for both.
        return _build(nx=6, ny=4, nz=20, frozen=True, soil_moisture=0.30,
                      ruc_soilprop="wrf_461")

    state, cfg, driver = frozen()
    for _ in range(6):
        step(state, cfg)
    keep = cp.asnumpy(driver.fields["keepfr3dflag"])
    hot = np.argwhere(keep != 0.0)
    assert len(hot) > 0, (
        "no soil cell reached KEEPFR3DFLAG==1, so nothing on this grid reads "
        "SMFR3D and the test below would be vacuous")
    index = tuple(int(v) for v in hot[0])
    assert index[0] >= 1, (
        f"KEEPFR3DFLAG==1 first at {index}; the surface level is not where "
        "the frozen-soil cap binds")

    moved, others = _next_step_delta(
        tmp_path, "smfr3d", index, scale=0.1, build=frozen)
    assert "smfr3d" in moved
    assert len(others) >= _MIN_OTHERS, (
        f"scaling SMFR3D to 0.1x at {index}, a cell with KEEPFR3DFLAG==1, "
        f"reached only {others}; SMFR3D is not a live carrier even where WRF "
        "reads it")

    # KEEPFR3DFLAG is the gate itself, so flipping it must move the column in
    # BOTH directions: 1 -> 0 removes the cap, 0 -> 1 imposes it.
    #
    # Only ``others`` is asserted here, and that is not a weaker check but the
    # only correct one.  KEEPFR3DFLAG is READ by soilprop at :2704 and then
    # REASSIGNED at :2744 and written back at :1058, so a flipped flag steers
    # the step and is then recomputed to whatever the new state implies --
    # frequently back to the reference value.  Requiring the flag itself to
    # differ afterwards would be asserting that WRF does not overwrite it,
    # which it does.  Measured: 1 -> 0 reaches 21 other carriers while
    # KEEPFR3DFLAG itself comes back identical.
    _, off_others = _next_step_delta(
        tmp_path, "keepfr3dflag", index, set_to=0.0, build=frozen)
    assert len(off_others) >= _MIN_OTHERS, off_others

    cold = np.argwhere((keep == 0.0)
                       & (cp.asnumpy(driver.fields["smfr3d"]) > 0.01))
    assert len(cold) > 0, (
        "no cell has frozen soil water with the cap disengaged, so the "
        "0 -> 1 direction cannot be driven on this grid")
    on_index = tuple(int(v) for v in cold[0])
    _, on_others = _next_step_delta(
        tmp_path, "keepfr3dflag", on_index, set_to=1.0, build=frozen)
    assert len(on_others) >= _MIN_OTHERS, on_others


@requires_gpu
def test_smfr3d_is_unread_on_an_unfrozen_grid_which_is_wrfs_own_structure(
        tmp_path):
    """The negative half of the test above, so the restriction is measured.

    :data:`woof.core.ruc_runtime.RUC_RUNTIME_RESTRICTIONS` claims SMFR3D is
    a CONDITIONAL carrier.  A claim about when a field is not read is only
    worth publishing if the "not read" half is also checked -- otherwise the
    frozen-grid test above could be passing for some unrelated reason.
    """
    from woof.core.dycore import step

    state, cfg, driver = _build(nx=6, ny=4, nz=20)
    for _ in range(6):
        step(state, cfg)
    keep = cp.asnumpy(driver.fields["keepfr3dflag"])
    assert int(np.count_nonzero(keep)) == 0, (
        "a warm column reached KEEPFR3DFLAG==1; the restriction text is "
        "wrong about WRF's :2744 gate")
    frozen_water = cp.asnumpy(driver.fields["smfr3d"])
    assert float(np.abs(frozen_water).max()) > 0.0, (
        "SMFR3D is identically zero, so 'unread' here is trivially true and "
        "proves nothing")
    moved, others = _next_step_delta(tmp_path, "smfr3d", (1, 0, 0), scale=0.1)
    assert others == [], (
        f"SMFR3D reached {others} on an unfrozen grid, where soilprop's "
        "keepfr gate says it cannot be read")


@requires_gpu
def test_the_inert_restart_carriers_are_the_measured_ones(tmp_path):
    """Publish the inert list as a gate, not as prose.

    Every name in :data:`RUC_MEASURED_INERT_CARRIERS` round-trips bit for bit
    -- the round-trip test above covers that -- but its restored value does
    not reach the next step, because LSMRUC recomputes it first.  This is the
    disclosure Noah-MP's lane had to publish after the fact; asserting it
    keeps the published list accurate, and it fails loudly if a future change
    makes one of them live (which would be good news, and should move the
    name rather than be absorbed silently).

    Only a representative subset is driven, because each entry costs two
    restores and a step.  The three chosen are one per mechanism: GRDFLX is
    assigned unconditionally at ``:1096``, EMISS is refreshed from SOILVEGIN's
    table every call, and SH2O is re-derived by the freezing curve.
    """
    from woof.core.ruc_runtime import (RUC_MEASURED_INERT_CARRIERS,
                                        RUC_MEASURED_LIVE_CARRIERS)

    assert not (set(RUC_MEASURED_INERT_CARRIERS)
                & set(RUC_MEASURED_LIVE_CARRIERS))
    for name in (*RUC_MEASURED_INERT_CARRIERS, *RUC_MEASURED_LIVE_CARRIERS):
        assert name in _CARRIED, f"{name} is published but not carried"

    for name, nudge in (("grdflx", 1.0), ("emiss", -0.01), ("sh2o", 0.005)):
        index = (0, 0, 0) if name == "sh2o" else (0, 0)
        moved, others = _next_step_delta(tmp_path, name, index, add=nudge)
        assert others == [], (
            f"{name} is published as INERT but reached {others}; the "
            "published restriction is now wrong")


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

@requires_gpu
def test_a_real_wrfout_carries_nine_soil_layers_and_rucs_own_fields(tmp_path):
    """Write the file with the production writer, then read it with netCDF4.

    ``state_frame`` returning the names is not the claim; the claim is that a
    file exists on disk whose soil axis is NINE long, whose RUC-only carriers
    are on it, and whose values survive the round trip.
    """
    import netCDF4

    from woof.config import soil_layer_count
    from woof.core.dycore import step
    from woof.io.wrfout import WrfoutWriter, state_frame

    state, cfg, driver = _build(nx=6, ny=4, nz=20, snow_mm=20.0,
                                snow_depth_m=0.10)
    for _ in range(4):
        step(state, cfg)
    frame = state_frame(state)
    for name in ("TSLB", "SMOIS", "SH2O", "SMFR3D", "KEEPFR3DFLAG",
                 "SOILT1", "TSNAV", "QVG", "QSG", "SFCEXC", "ACRUNOFF",
                 "RUC_RUNOFF2"):
        assert name in frame, name

    path = tmp_path / "wrfout_d01_2026-07-01_18_00_48"
    writer = WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                          dx=cfg.dx, dy=cfg.dy,
                          soil_layers=soil_layer_count(cfg))
    host = {name: _host_frame_value(value) for name, value in frame.items()}
    writer.write_frame("2026-07-01_18:00:48", host)
    writer.close()

    with netCDF4.Dataset(path) as ds:
        assert len(ds.dimensions["soil_layers_stag"]) == _NSOIL
        for name in ("TSLB", "SMOIS", "SH2O", "SMFR3D", "KEEPFR3DFLAG"):
            variable = ds.variables[name]
            assert variable.dimensions == (
                "Time", "soil_layers_stag", "south_north", "west_east"), name
            stored = np.asarray(variable[0])
            assert stored.shape == (_NSOIL, cfg.ny, cfg.nx), name
            assert np.isfinite(stored).all(), name
            np.testing.assert_array_equal(stored, host[name], err_msg=name)
        for name in RUC_STATE_2D + RUC_DIAGNOSTICS_2D:
            upper = name.upper()
            stored = np.asarray(ds.variables[upper][0])
            assert np.isfinite(stored).all(), upper
            np.testing.assert_array_equal(stored, host[upper], err_msg=upper)
        # The soil column is physical, not a zero-filled axis of the right
        # length -- which is what a wrong ``soil_layers`` would still give.
        tslb = np.asarray(ds.variables["TSLB"][0])
        assert float(tslb.min()) > 200.0 and float(tslb.max()) < 350.0
        smois = np.asarray(ds.variables["SMOIS"][0])
        assert float(smois.min()) >= 0.0 and float(smois.max()) <= 1.0
        # Nine DISTINCT levels, not one value broadcast nine times.
        land = tslb[..., :cfg.nx - 2]
        assert len(np.unique(land[:, 0, 0])) >= 5


@requires_gpu
def test_a_noah_wrfout_gains_no_ruc_variables(tmp_path):
    """The output gate keys on the resolved routing, not on field existence."""
    import netCDF4

    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics
    from woof.io.wrfout import WrfoutWriter, state_frame

    cfg = RunConfig(nx=6, ny=4, nz=20, dx=3000.0, dy=3000.0, ztop=12000.0,
                    dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
                    sf_sfclay_physics=1, sf_surface_physics=2,
                    bl_pbl_physics=1)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 300.0 + 0.004 * np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, lambda z: 0.008 + 0.0 * z)
    initialize_physics(state, cfg, glw=_IDEALISED_GLW)
    frame = state_frame(state)
    for name in RUC_STATE_2D + RUC_STATE_3D + RUC_DIAGNOSTICS_2D:
        assert name.upper() not in frame, name
    path = tmp_path / "wrfout_noah"
    writer = WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                          dx=cfg.dx, dy=cfg.dy, soil_layers=4)
    writer.write_frame("2026-07-01_18:00:00",
                       {name: _host_frame_value(value)
                        for name, value in frame.items()})
    writer.close()
    with netCDF4.Dataset(path) as ds:
        assert len(ds.dimensions["soil_layers_stag"]) == 4
        for name in RUC_STATE_2D + RUC_STATE_3D + RUC_DIAGNOSTICS_2D:
            assert name.upper() not in ds.variables, name


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------

@requires_gpu
def test_the_column_cost_is_what_the_registry_says():
    """Measure the scaling blocker instead of asserting it.

    The column runs on the host, so the cost is per-column and roughly linear
    in the grid.  This times one land-surface call at two grid sizes and
    reports the per-column figure; the bound is deliberately loose because it
    is a property of whichever machine runs the suite, and the point is that
    the number is MEASURED and published rather than that it is small.
    """
    import time

    from woof.core.dycore import step

    timings = {}
    for nx, ny in ((6, 4), (10, 6)):
        state, cfg, driver = _build(nx=nx, ny=ny, nz=20)
        step(state, cfg)  # pay the first-step cold path
        start = time.perf_counter()
        step(state, cfg)
        cp.cuda.runtime.deviceSynchronize()
        timings[(nx, ny)] = time.perf_counter() - start
    per_column = {shape: seconds / (shape[0] * shape[1])
                  for shape, seconds in timings.items()}
    print("\nRUC host column cost, seconds per column:")
    for shape, seconds in sorted(per_column.items()):
        print(f"  {shape[0]}x{shape[1]}: {seconds * 1e3:.3f} ms/column "
              f"(step {timings[shape] * 1e3:.1f} ms)")
    # It is a host loop, so it must be slow enough to be worth publishing and
    # not so slow that the suite becomes unrunnable.
    assert max(per_column.values()) < 1.0
