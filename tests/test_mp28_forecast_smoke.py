"""G4 -- the first multi-step ``mp_physics=28`` forecast, and the measured
lateral-boundary aerosol depletion rate.

WHY THIS FILE EXISTS
--------------------
Before it, every claim in the mp=28 port was a COLUMN claim.  G1 gated device
helpers pointwise, G2 gated single kernels against 48-row Fortran columns, G3
gated the whole adapter against nineteen of those columns -- all of them one
call of ``_apply_thompson_aerosol`` on a state assembled by a test.  Nothing
had ever run mp=28 inside ``dycore.step``: no transport of ``nc``/``nwfa``/
``nifa``, no lateral boundaries, no accumulation across steps, no interaction
between the aerosol tendency accumulators and the persistent scratch arena
that holds them.  A scheme can be column-exact and still be unusable as a
forecast, and the failure modes are exactly the ones a column cannot see --
scratch carried between steps, a transported species that is not advected, a
boundary policy that quietly empties a tracer.

WHAT G4 ASSERTS (MP28_PORT_SPEC.md, "GATES, IN ORDER")
------------------------------------------------------
On a specified-lateral-boundary domain, over a real multi-step integration:

* no NaN anywhere, on any step;
* the aerosol, droplet-number and effective-radius bounds WRF's terminal
  clamp establishes hold on every step, in the region microphysics updates;
* a column WRF leaves at its no-microphysics exit (``:2020``) leaves the
  call with its aerosol above the surface exactly as it entered and its
  droplet number at zero, because that exit comes before the terminal apply;
* the specified-zone ring is BIT-restored by every microphysics call.

THE REGISTERED LBC DEVIATION, MEASURED
--------------------------------------
WOOF carries ``qv`` from an external lateral-boundary snapshot always, and
``nwfa``/``nifa`` beside it whenever the run's aerosol initial state came
from WRF's monthly WIF climatology (``woof/boundary_fields.py``
``external_scalar_fields``; ``woof/ingest/lateral_bc.py``
``COUPLED_SCALAR_STATE_FIELDS``).  WITHOUT that dataset the two aerosol
numbers fall back to WRF's ``flow_dep_bdy`` treatment -- zero gradient on
outflow, ZERO on inflow -- which is the state this file measures below, and
which an externally forced run now has to select deliberately: the doors
that commit to building a forecast refuse it otherwise, before they fetch
anything (``woof.config.validate_experiment_preparation``, measured by
``mp28_aerosol_lateral_forcing_precondition``).  ``nc`` is on the
flow-dependent side either way; ``Registry.EM_COMMON`` declares
``qnwfa``/``qnifa`` with the boundary dimension and ``bdy_interp``, and
stock WRF forces them from the same monthly dataset through
``constants_name``.

The consequence is aerosol-free air advecting in at every inflow face.  It
cannot NaN and cannot go negative.  Where the scheme runs, WRF's terminal
floors (``module_mp_thompson.F:3979-3982``: ``nwfa >= 11.1E6``,
``nifa >= 5.0E3``) hold the air at the floor; in a column with no
condensate and no ice supersaturation WRF returns at ``:2020``, before
the terminal apply, so the clear inflow air keeps its zero aerosol there.
Either way no automated gate anywhere in the tree would ever notice.  It
is a slow, silent, physically wrong trend, and the only defence is to
MEASURE it and publish the number.  That is
:func:`test_lbc_aerosol_depletion_rate_is_measured_on_a_cloud_free_forecast`,
and its result is the headline of
``docs/public/wrf-comparison/mp28-column-evidence.md``.

The measurement is made on a deliberately CLOUD-FREE forecast (WK82's own
sounding with no thermal, which is subsaturated at every level) so that the
aerosol has NO microphysical source or sink at all: the test asserts
``qc == qr == qi == qs == qg == 0`` for the whole run, which makes every
kilogram of lost aerosol attributable to the boundary policy and to nothing
else.  The convective run in
:func:`test_g4_multistep_specified_bc_forecast_is_finite_and_bounded` is the
opposite case and is where the bounds are exercised.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from conftest import requires_gpu

# Consumer expectations come from the independent generated declarations.
from woof.physics_registry import implemented_selector_values as _intake_selectors
from woof.physics_registry import consumer_rows_by_selector as _intake_rows
_INTAKE_MP = tuple(sorted(_intake_selectors("microphysics")))
_INTAKE_REFL = tuple(mp for mp in _INTAKE_MP if mp)
_INTAKE_PRODUCERS = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "reflectivity_input_species").items()
    if row is not None))
from woof.core.column_schemes import COLUMN_SCHEMES
_INTAKE_NATIVE = tuple(sorted((set(_INTAKE_REFL) - set(_INTAKE_PRODUCERS)) | set(COLUMN_SCHEMES)))
_INTAKE_SNOW = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "ring_guard").items()
    if "qs" in (row or {}).get("state_fields", ())))
_INTAKE_SIX = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "ring_guard").items()
    if "qg" in (row or {}).get("state_fields", ())))

# ---------------------------------------------------------------------------
# WRF's own bounds.  Every constant below is transcribed from
# wrf461-pristine/phys/module_mp_thompson.F with its line number,
# and every one of them is a bound the TERMINAL apply establishes -- so it is
# a bound on the state microphysics leaves behind, not on the state transport
# leaves behind.  The distinction is the whole reason the checks below are
# restricted to the microphysics-updated interior; see _INTERIOR_ONLY_NOTE.
#
# "Updated" is a COLUMN property as well as a region.  mp_thompson returns
# at :2020 from a column whose entry cloud, ice, rain, snow and graupel are
# all at or below R1 and which is nowhere supersaturated over ice
# (:1646, :1827-1990), before the terminal apply at :3972-4021, so none of
# these bounds is established there: the column keeps the aerosol transport
# gave it (mp_gt_driver adds the surface emission at k = 0 afterwards,
# :1316-1321), and its droplet number is the zero the entry rewrite wrote
# (:1844-1845).  The adapter records WRF's per-column decision in the
# ``mp_thompson_micro_columns`` scratch slot (1 = updated, 0 = returned at
# :2020), and the checks below read it for every call.
# ---------------------------------------------------------------------------

#: The scratch slot the mp=28 adapter writes WRF's per-column :2020 decision
#: into, taken on the entry state of each call.
MICRO_COLUMNS_SLOT = "mp_thompson_micro_columns"

#: :1805 / :3979-3980.  nwfa is clamped to [11.1E6, 9999.E6] -- a per-kilogram
#: quantity clamped against per-cubic-metre constants, WRF's own unit
#: inconsistency, reproduced literally by thompson_aerosol_state.cu.
NWFA_FLOOR = 11.1e6
NWFA_CEILING = 9999.0e6

#: :1806 / :3981-3982.  naIN1*0.01 with naIN1 = 0.5E6 at :95.
NIFA_FLOOR = 5.0e3
NIFA_CEILING = 9999.0e6

#: :89 Nt_c_max, and :4020's DBLE(Nt_c_max)/rho(k).  The terminal rediagnosis
#: caps the per-kilogram nc at Nt_c_max/rho, so the state bound is
#: density-dependent; :3976's earlier clamp uses the unconverted constant.
NT_C_MAX = 1999.0e6

#: calc_effectRad's clamps (:5623-5652), in the MICRON convention woof's
#: state contract stores (the metre->micron multiply is the writer's last
#: operation; thompson.cu:378-383, thompson_aerosol_state.cu:882-884).
EFFC_BAND_UM = (2.49, 50.0)
EFFI_BAND_UM = (4.99, 125.0)
EFFS_BAND_UM = (9.99, 999.0)

_INTERIOR_ONLY_NOTE = """
The bounds are checked on the microphysics-updated INTERIOR only, i.e.
outside the spec_zone ring.  That is not a convenience: WRF's clipped
microphysics tiles never touch the ring (solve_em.F:3631-3639), WOOF
reproduces that by bit-restoring it (microphysics.spec_zone_ring_slices),
and therefore the ring carries whatever TRANSPORT left there -- which on an
inflow face is exactly zero, below WRF's floor.  Asserting the floor on the
ring would assert that WOOF violates WRF's tile clipping.  Inside the
interior the aerosol floors are asserted on the columns the call updated;
a column WRF returned from at :2020 is held instead to leaving the call
with its aerosol above the surface untouched and its droplet number zero.
""".strip()

#: Repository-relative path of the evidence document this file's measurements
#: are published in.  The lockstep test below reads it.
EVIDENCE_DOC = (pathlib.Path(__file__).resolve().parent.parent
                / "docs" / "public" / "wrf-comparison"
                / "mp28-column-evidence.md")


# ---------------------------------------------------------------------------
# The forecast.
# ---------------------------------------------------------------------------

#: Imposed uniform zonal inflow (m/s).  West face is inflow, east is outflow.
FORECAST_U = 20.0

#: Domain and integration.  dx = 2 km with dt = 12 s is WRF's own 6*dx/1000
#: guidance; time_step_sound = 4 puts the acoustic Courant number at
#: 340*3/2000 = 0.51.  nz = 24 to ztop = 16 km.
FORECAST_NX, FORECAST_NY, FORECAST_NZ = 28, 16, 24
FORECAST_DX = 2000.0
FORECAST_DT = 12.0

#: 150 steps = 1800 s.  At 20 m/s that is a fetch of 36 km across a 56 km
#: domain, i.e. the inflow air reaches 64% of the way across -- enough to fit
#: a depletion-front speed and still leave undisturbed air to measure against.
FORECAST_STEPS = 150

#: The convective run's thermal: WK82/em_quarter_ss shape (3 K, cos^2 inside
#: the unit ellipsoid) with an 8 km horizontal radius so it fits this domain.
BUBBLE_DELT = 3.0
BUBBLE_RADIUS = 8000.0
BUBBLE_ZC = 1500.0


def _forecast_config(**overrides):
    from woof.config import RunConfig, validate_run_config

    values = dict(
        nx=FORECAST_NX, ny=FORECAST_NY, nz=FORECAST_NZ,
        dx=FORECAST_DX, dy=FORECAST_DX, ztop=16000.0,
        dt=FORECAST_DT, run_seconds=FORECAST_STEPS * FORECAST_DT,
        moist=True, mp_physics=28, moist_adv_opt=1,
        specified=True, spec_zone=1, relax_zone=4, spec_bdy_width=5,
    )
    values.update(overrides)
    return validate_run_config(RunConfig(**values))


def _tables_or_skip():
    """Skip only when CCN_ACTIVATE.BIN is genuinely absent.

    The asset ships as of 2026-08-01 (MP28_PORT_SPEC.md blocking unknown 1
    and HARD RULE 5 of the port, both reversed), so this guard does not fire
    on a clean checkout.  It stays as defence for a tree where the file was
    deleted or an override points elsewhere; it names the one asset rather
    than swallowing every load failure.
    """
    from woof.core.thompson_aerosol_contract import (
        MissingAerosolTableAsset, resolve_aerosol_table_root,
        resolve_ccn_activation_path)
    try:
        resolve_ccn_activation_path(None, resolve_aerosol_table_root(None))
    except MissingAerosolTableAsset as exc:                # pragma: no cover
        pytest.skip(f"CCN_ACTIVATE.BIN unavailable: {exc}")


def _build_states(cp, cfg, *, bubble: bool, wind: float):
    """A balanced moist WK82 state, plus three identical driving states.

    ``init_moist_balanced`` is WRF's own quarter_ss moist rebalance
    (``module_initialize_ideal.F:1026-1063``), so the column starts in
    discrete hydrostatic balance under the MOIST equation of state.  That
    matters more here than it looks: seeding qv onto a dry-balanced base
    state leaves an O(1 K) buoyancy imbalance that rings at 15+ m/s within a
    minute and manufactures cloud everywhere, which would make the
    cloud-free attribution of the depletion measurement false.
    """
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.verify.cases.wk82 import wk82_sounding, wk82_theta

    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, wk82_theta, 1.0e5, cfg.ztop)
    y = (np.arange(cfg.ny) + 0.5) * cfg.dy - 0.5 * cfg.ny * cfg.dy

    def thp_func(x, z):
        xr = x[None, None, :] / BUBBLE_RADIUS
        yr = y[None, :, None] / BUBBLE_RADIUS
        zr = (z[:, None, None] - BUBBLE_ZC) / BUBBLE_ZC
        rad = np.sqrt(xr ** 2 + yr ** 2 + zr ** 2)
        thp = np.where(rad <= 1.0,
                       BUBBLE_DELT * np.cos(0.5 * np.pi * rad) ** 2, 0.0)
        return np.broadcast_to(thp, (cfg.nz, cfg.ny, cfg.nx))

    def one(with_bubble):
        state = init_moist_balanced(
            cfg, coord, base, lambda z: wk82_sounding(z)[1],
            thp_func if with_bubble else None)
        state.u[...] = cp.float32(wind)
        state.qv0[...] = state.qv
        return state

    return one(bubble), [one(False) for _ in range(3)]


def _attach_specified_boundaries(state, forcing, cfg):
    from woof.ingest.lateral_bc import (attach_lateral_boundaries,
                                         build_state_lateral_boundaries)

    boundaries = build_state_lateral_boundaries(
        forcing, [0.0, 21600.0, 43200.0],
        spec_bdy_width=cfg.spec_bdy_width, spec_zone=cfg.spec_zone,
        relax_zone=cfg.relax_zone)
    attach_lateral_boundaries(state, boundaries)


def _entry_density(cp, state):
    """WRF's :1802 density, from the state woof's adapter would see."""
    from woof.core.state import DTYPE

    th = state.thb.reshape(-1, 1, 1) + state.thp if state.thb.ndim == 1 \
        else state.thb + state.thp
    pii = (state.p / DTYPE(1.0e5)) ** DTYPE(287.0 / 1004.5)
    temperature = th * pii
    qv = cp.maximum(state.qv, DTYPE(1.0e-10))
    return (DTYPE(0.622) * state.p
            / (DTYPE(287.04) * temperature * (qv + DTYPE(0.622))))


_FINITE_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni",
                  "nwfa", "nifa", "effc", "effi", "effs",
                  "thp", "php", "mup", "u", "v", "w", "h_diabatic")

#: Fields whose spec-zone ring the microphysics call must leave BIT-identical.
#: h_diabatic is excluded on purpose: WRF pins its ring to exactly 0 rather
#: than restoring it, which mp=8's own gate already asserts.
_RING_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg",
                "nc", "nr", "ni", "nwfa", "nifa", "thp")


def _run_forecast(cp, cfg, *, bubble: bool, steps: int, wind: float,
                  initialise: bool = True):
    """Integrate ``steps`` full RK3 steps and record what G4 needs.

    The spec-zone ring check is instrumented INSIDE the step: ``dycore.step``
    runs the RK3 loop (which writes the ring through the lateral-boundary
    path) and only then calls microphysics, so a before/after comparison
    across a whole step could not distinguish "microphysics restored the
    ring" from "the boundary rewrote it".  The wrapper snapshots the ring
    immediately before ``apply_microphysics`` and compares immediately
    after, which is exactly the property WRF's tile clipping guarantees.

    THE INITIALISATION PATH IS THE PRODUCTION ONE.  The default run goes
    through ``woof.core.physics.initialize_physics`` -- woof's ``phy_init``,
    and the caller of WRF's ``mp_init`` -- rather than reaching into
    ``microphysics_init`` directly, so the whole gate integrates the state a
    user's forecast integrates and the driver's ``accept_microphysics``
    contract for mp=28 is exercised on every one of these steps too.

    ``initialise=False`` is the COUNTERFACTUAL: the same domain with the
    production init path skipped entirely, so ``nwfa``/``nifa`` reach the
    scheme as allocated zeros, the terminal apply clamps both to WRF's
    floors (``module_mp_thompson.F:3979-3982``) in every column it
    updates, and the columns WRF returns from at ``:2020`` keep their
    zeros.  It exists so
    :func:`test_the_aerosol_profile_changes_the_forecast_measurably` can
    measure what the profile is WORTH.

    Attaching the driver is not a second variable: it is measured to be
    BITWISE inert over these 150 steps by
    :func:`test_attaching_the_physics_driver_does_not_move_the_forecast`, so
    the only thing that differs between the two runs is the aerosol field.
    """
    from woof.core import dycore, microphysics
    from woof.core.physics import initialize_physics

    state, forcing = _build_states(cp, cfg, bubble=bubble, wind=wind)
    _attach_specified_boundaries(state, forcing, cfg)
    driver = initialize_physics(state, cfg) if initialise else None
    init_receipt = {} if driver is None else driver.microphysics_init_receipt
    cp.cuda.Stream.null.synchronize()

    ring = microphysics.spec_zone_ring_slices(cfg.ny, cfg.nx, cfg.spec_zone)
    sz = int(cfg.spec_zone)
    interior = (Ellipsis, slice(sz, cfg.ny - sz), slice(sz, cfg.nx - sz))

    record = {
        "nwfa_initial": cp.asnumpy(state.nwfa).copy(),
        "nifa_initial": cp.asnumpy(state.nifa).copy(),
        "nwfa2d": cp.asnumpy(state.nwfa2d).copy(),
        "init_receipt": init_receipt,
        "ring_violations": [],
        "bound_violations": [],
        "nonfinite": [],
        "nwfa_interior_mean": [],
        "nifa_interior_mean": [],
        "nwfa_rows": [],
        "condensate_max": [],
        "w_max": [],
        "rain_max": [],
        "rain_sum": [],
        "nc_max": [],
        "qc_max": [],
        "interior": interior,
        "initialised": bool(initialise),
        "driver": driver,
        # Per call: WRF's :2020 decision for every column (True = updated),
        # the interior counts either way, and every way a column WRF
        # returned from came out of the call touched.
        "micro_columns": None,
        "updated_columns": [],
        "skipped_columns": [],
        "skipped_column_violations": [],
    }

    real_apply = dycore.apply_microphysics
    step_index = {"n": 0}

    def instrumented(target, run_cfg, dt, **kwargs):
        before = {name: [getattr(target, name)[slc].copy() for slc in ring]
                  for name in _RING_FIELDS}
        entry_aerosol = {name: getattr(target, name).copy()
                         for name in ("nwfa", "nifa")}
        result = real_apply(target, run_cfg, dt, **kwargs)
        rainnc = getattr(result, "rainnc", None)
        if rainnc is not None:
            record["rain_max"].append(float(cp.max(rainnc)))
            record["rain_sum"].append(float(cp.sum(rainnc)))
        for name in _RING_FIELDS:
            now = getattr(target, name)
            for slc, saved in zip(ring, before[name]):
                if not bool(cp.array_equal(now[slc], saved)):
                    record["ring_violations"].append((step_index["n"], name))
        _check_column_exit(cp, target, entry_aerosol, record,
                           step_index["n"], interior)
        return result

    dycore.apply_microphysics = instrumented
    try:
        for n in range(steps):
            step_index["n"] = n
            dycore.step(state, cfg)
            cp.cuda.Stream.null.synchronize()
            _collect_step(cp, state, cfg, record, n, interior)
    finally:
        dycore.apply_microphysics = real_apply

    record["state"] = state
    return record


def _bits(cp, array):
    return cp.ascontiguousarray(array).view(cp.uint32)


def _check_column_exit(cp, state, entry_aerosol, record, n, interior):
    """WRF's ``:2020`` exit, held around every call from the adapter's flag.

    The flag is read back off the scratch slot the adapter wrote it to, so
    what is checked is the decision this call actually made, on the columns
    of the microphysics interior.  A column WRF returned from must leave the
    call with ``nwfa`` and ``nifa`` above ``k = 0`` BITWISE as they entered
    -- the surface emission (``mp_gt_driver:1320-1321``) is the only aerosol
    write after the return and it lands on ``k = 0`` -- and with ``nc`` zero
    at every level, the zero the entry rewrite (``:1844-1845``) wrote and
    nothing after the return overwrites.
    """
    flag = state.existing_scratch(MICRO_COLUMNS_SLOT)
    if flag is None:
        record["micro_columns"] = None
        record["skipped_column_violations"].append(
            (n, f"the call wrote no {MICRO_COLUMNS_SLOT} flag"))
        return
    updated = flag[interior] != 0
    skipped = ~updated
    record["micro_columns"] = updated
    record["updated_columns"].append(int(cp.count_nonzero(updated)))
    record["skipped_columns"].append(int(cp.count_nonzero(skipped)))
    if not bool(cp.any(skipped)):
        return
    for name in ("nwfa", "nifa"):
        now = _bits(cp, getattr(state, name)[interior][1:])
        was = _bits(cp, entry_aerosol[name][interior][1:])
        changed = int(cp.count_nonzero((now != was) & skipped[None]))
        if changed:
            record["skipped_column_violations"].append(
                (n, f"{name} above k=0 changed in {changed} cells of "
                    "columns WRF returned from"))
    nonzero = int(cp.count_nonzero(
        (state.nc[interior] != 0) & skipped[None]))
    if nonzero:
        record["skipped_column_violations"].append(
            (n, f"nc nonzero in {nonzero} cells of columns WRF returned "
                "from"))


def _collect_step(cp, state, cfg, record, n, interior):
    for name in _FINITE_FIELDS:
        value = getattr(state, name, None)
        if value is None:
            continue
        if not bool(cp.all(cp.isfinite(value))):
            record["nonfinite"].append((n, name))

    nwfa = getattr(state, "nwfa")[interior]
    nifa = getattr(state, "nifa")[interior]
    nc = getattr(state, "nc")[interior]
    rho = _entry_density(cp, state)[interior]

    def bad(label, value, lo, hi):
        if value.size == 0:
            return
        lo_bad = float(cp.min(value)) < lo
        hi_bad = float(cp.max(value)) > hi
        if lo_bad or hi_bad:
            record["bound_violations"].append(
                (n, label, float(cp.min(value)), float(cp.max(value)),
                 lo, hi))

    # The terminal apply's aerosol bounds, on the columns it ran in.  With
    # no flag the whole interior is held to them, which is the stricter
    # reading; the missing flag is itself a recorded violation.
    updated = record["micro_columns"]
    if updated is None:
        updated = cp.ones(nwfa.shape[1:], dtype=bool)
    in_updated = cp.broadcast_to(updated, nwfa.shape)
    in_skipped = ~in_updated
    bad("nwfa", nwfa[in_updated], NWFA_FLOOR * (1.0 - 1e-6),
        NWFA_CEILING * (1.0 + 1e-6))
    bad("nifa", nifa[in_updated], NIFA_FLOOR * (1.0 - 1e-6),
        NIFA_CEILING * (1.0 + 1e-6))
    # A column WRF returned from carries what transport left it: never
    # negative, and not held to a floor the terminal apply never wrote.
    bad("nwfa (returned at :2020)", nwfa[in_skipped], 0.0, float("inf"))
    bad("nifa (returned at :2020)", nifa[in_skipped], 0.0, float("inf"))
    # nc: zero where the terminal apply zeroed qc, else <= Nt_c_max/rho.
    nc_ratio = nc * rho
    bad("nc*rho", nc_ratio, 0.0, NT_C_MAX * (1.0 + 1e-5))
    bad("effc", state.effc[interior], EFFC_BAND_UM[0] * (1.0 - 1e-6),
        EFFC_BAND_UM[1] * (1.0 + 1e-6))
    bad("effi", state.effi[interior], EFFI_BAND_UM[0] * (1.0 - 1e-6),
        EFFI_BAND_UM[1] * (1.0 + 1e-6))
    bad("effs", state.effs[interior], EFFS_BAND_UM[0] * (1.0 - 1e-6),
        EFFS_BAND_UM[1] * (1.0 + 1e-6))

    record["nwfa_interior_mean"].append(float(cp.mean(nwfa)))
    record["nifa_interior_mean"].append(float(cp.mean(nifa)))
    record["nwfa_rows"].append(
        cp.asnumpy(state.nwfa[cfg.nz // 2, cfg.ny // 2, :]).copy())
    record["condensate_max"].append(max(
        float(cp.max(getattr(state, name)))
        for name in ("qc", "qr", "qi", "qs", "qg")))
    record["w_max"].append(float(cp.max(cp.abs(state.w))))
    record["nc_max"].append(float(cp.max(state.nc)))
    record["qc_max"].append(float(cp.max(state.qc)))


_FORECASTS: dict[tuple, dict] = {}


def _forecast(cp, *, bubble: bool, wind: float = FORECAST_U,
              initialise: bool = True):
    """One integration per case per session; the tests share it."""
    key = (bubble, wind, FORECAST_STEPS, initialise)
    if key not in _FORECASTS:
        cfg = _forecast_config()
        record = _run_forecast(cp, cfg, bubble=bubble, steps=FORECAST_STEPS,
                               wind=wind, initialise=initialise)
        record["wind"] = wind
        _FORECASTS[key] = (cfg, record)
    return _FORECASTS[key]


# ---------------------------------------------------------------------------
# G4 proper.
# ---------------------------------------------------------------------------

@requires_gpu
def test_g4_multistep_specified_bc_forecast_is_finite_and_bounded():
    """G4: 150 RK3 steps of mp=28 on a specified-BC convective domain.

    The first multi-step mp=28 forecast that has ever been run.  Per step:
    every prognostic and every radiation-facing effective radius finite, and
    every bound WRF's terminal apply establishes holding in the
    microphysics-updated interior, on the columns the call updated.  Every
    column WRF returned from at ``:2020`` leaves the call with its aerosol
    above the surface bitwise as it entered and its droplet number zero.

    ``nc`` is checked as ``nc*rho`` against ``Nt_c_max`` because that is the
    form WRF's terminal rediagnosis caps (:4020 caps the per-kilogram value
    at ``DBLE(Nt_c_max)/rho(k)``); checking ``nc`` against the raw constant
    would be a DIFFERENT and weaker claim wherever rho < 1.
    """
    import cupy as cp

    _tables_or_skip()
    cfg, record = _forecast(cp, bubble=True)

    assert not record["nonfinite"], (
        "non-finite state during the forecast: "
        + ", ".join(f"step {n} {name}" for n, name in record["nonfinite"][:12]))
    assert not record["skipped_column_violations"], (
        "a column WRF leaves at its no-microphysics exit (:2020) came out of "
        "the call touched:\n  " + "\n  ".join(
            f"step {n}: {what}"
            for n, what in record["skipped_column_violations"][:12]))
    # Both kinds of column were exercised, so neither check above or below
    # holds vacuously.
    assert max(record["updated_columns"]) > 0, record["updated_columns"][:12]
    assert max(record["skipped_columns"]) > 0, record["skipped_columns"][:12]
    assert not record["bound_violations"], (
        "WRF terminal-clamp bound violated in the microphysics interior:\n"
        + "\n".join(
            f"  step {n}: {label} in [{lo_seen:.6e}, {hi_seen:.6e}], "
            f"allowed [{lo:.6e}, {hi:.6e}]"
            for n, label, lo_seen, hi_seen, lo, hi
            in record["bound_violations"][:12])
        + "\n" + _INTERIOR_ONLY_NOTE)

    # Guard the guard: a forecast that produced nothing would satisfy every
    # bound above vacuously.  This one has to have made weather.
    assert max(record["condensate_max"]) > 1.0e-5, (
        "the convective run produced essentially no condensate "
        f"(max {max(record['condensate_max']):.3e} kg/kg); the bounds above "
        "would then hold vacuously")
    assert max(record["w_max"]) > 1.0, (
        f"no vertical motion developed (max |w| "
        f"{max(record['w_max']):.3e} m/s)")
    assert max(record["nc_max"]) > 1.0e6, (
        "prognostic droplet number never became significant, so the mp=28 "
        "path that distinguishes it from mp=8 was never exercised")


#: The long run: 600 steps = 7200 s = 2 hours of model time, which is
#: 2.6 ventilation times of this domain at FORECAST_U.  Long enough that the
#: aerosol field is entirely inflow air -- aerosol-free where WRF's :2020
#: exit leaves a clear column alone, on WRF's floor where the scheme runs --
#: the state a 6-hour operational run would spend most of its life in, and
#: the one no column fixture can produce.
LONG_FORECAST_STEPS = 600


@requires_gpu
def test_g4_a_two_hour_forecast_stays_finite_bounded_and_ring_clean():
    """The longest mp=28 integration that has been run. ~19 s on an RTX 5090.

    The 150-step gate above establishes that the scheme survives being
    stepped; this establishes that it survives being stepped for long enough
    to reach the regime it would actually run in.  Two hours at 20 m/s
    ventilates this 56 km domain 2.6 times, so by the end EVERY cell holds
    inflow air: aerosol-free in the columns WRF returns from at ``:2020``,
    on WRF's floors in the columns the scheme updates.  That is
    a genuinely different state from anything the column fixtures or the
    30-minute run visit, and it is where a slow accumulator leak, a
    persistent-scratch carry or a clamp that is applied one step late would
    finally show.

    This is class-D evidence: self-consistency, no external reference.  It
    says the port does not fall over.  It says nothing about whether the
    trajectory is WRF's.
    """
    import cupy as cp

    _tables_or_skip()
    cfg = _forecast_config(run_seconds=LONG_FORECAST_STEPS * FORECAST_DT)
    record = _run_forecast(cp, cfg, bubble=True, steps=LONG_FORECAST_STEPS,
                           wind=FORECAST_U)
    record["wind"] = FORECAST_U

    assert not record["nonfinite"], record["nonfinite"][:12]
    assert not record["bound_violations"], record["bound_violations"][:12]
    assert not record["skipped_column_violations"], (
        record["skipped_column_violations"][:12])
    assert not record["ring_violations"], record["ring_violations"][:12]
    assert float(record["state"].elapsed_seconds) == \
        LONG_FORECAST_STEPS * FORECAST_DT

    # The domain really has been ventilated.  The inflow air carries no
    # aerosol, the columns WRF returns from at :2020 keep it that way and the
    # columns the scheme updates hold it at WRF's floors, so the interior
    # mean ends below the floors; the initial profile started six times
    # above (6.7e7 kg^-1 mean, nowhere below naCCN1 = 5.0e7).  nwfa stays
    # positive because the surface emission at k = 0 keeps adding.
    final_nwfa = record["nwfa_interior_mean"][-1]
    final_nifa = record["nifa_interior_mean"][-1]
    assert 0.0 < final_nwfa < NWFA_FLOOR, final_nwfa
    assert 0.0 <= final_nifa < NIFA_FLOOR, final_nifa
    # and it made weather while doing it.
    assert max(record["w_max"]) > 5.0
    assert record["rain_sum"][-1] > 0.0

    print(f"\n2-hour mp=28 forecast: {LONG_FORECAST_STEPS} steps, "
          f"max|w| {max(record['w_max']):.2f} m/s, "
          f"final interior nwfa {final_nwfa:.4e} kg^-1 "
          f"(floor {NWFA_FLOOR:.3e}), nifa {final_nifa:.4e} kg^-1 "
          f"(floor {NIFA_FLOOR:.3e}), total RAINNC "
          f"{record['rain_sum'][-1]:.4f} mm, columns WRF returned from "
          f"on the last step {record['skipped_columns'][-1]} of "
          f"{record['skipped_columns'][-1] + record['updated_columns'][-1]}")


@requires_gpu
def test_g4_spec_zone_ring_is_bit_restored_by_every_microphysics_call():
    """WRF's clipped tiles never dispatch the ring; neither may WOOF's.

    Checked INSIDE the step, around the microphysics call only.  A whole-step
    before/after comparison cannot make this claim: ``dycore.step`` writes
    the ring from the lateral-boundary path during RK3, so the two effects
    are indistinguishable from outside.

    ``mp28_runnable`` already pins this for ONE call on a hand-built state.
    Here it is 150 consecutive calls on a state that transport, the Davies
    relaxation and the scheme's own accumulators have all been mutating.
    """
    import cupy as cp

    _tables_or_skip()
    _cfg, record = _forecast(cp, bubble=True)

    assert not record["ring_violations"], (
        "microphysics wrote the specified-zone ring on: "
        + ", ".join(f"step {n} field {name}"
                    for n, name in record["ring_violations"][:20]))


@requires_gpu
def test_g4_persistent_scratch_is_not_carried_between_forecast_steps():
    """The drift a column test structurally cannot find.

    ``state.scratch`` slots survive across steps by design
    (``woof/core/state.py:710-735``): the whole point of the arena is that
    the buffers are not reallocated.  So if the adapter failed to zero the
    three aerosol tendency accumulators at entry, EVERY step would inherit
    the previous step's ``ncten``/``nwfaten``/``nifaten``, and the result
    would be a slow, entirely plausible drift.  A single-call test cannot
    see it -- it allocates a fresh zeroed array and hands it in.

    ``tests/test_thompson_aerosol_adapter.py`` poisons the slots between two
    calls.  This does the same thing 40 times, inside a real integration,
    with EVERY ``mp_thompson_aero_*`` slot poisoned rather than three -- so
    it also covers the four working buffers and WRF's five hydrometeor
    tendency accumulators carried alongside the three aerosol tendency
    accumulators.  The requirement is BITWISE
    identity of the entire prognostic state against an unpoisoned run.
    """
    import cupy as cp

    from woof.core import dycore
    from woof.core.microphysics_aerosol import (AEROSOL_INT_SCRATCH_SLOTS,
                                                 AEROSOL_SCRATCH_SLOTS)
    from woof.core.physics import initialize_physics

    _tables_or_skip()
    cfg = _forecast_config()
    tracked = ("qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni",
               "nwfa", "nifa", "effc", "effi", "effs", "thp")
    steps = 40                       # bitwise identity does not need 150
    expected_slots = (
        "mp_thompson_aero_ncten", "mp_thompson_aero_nwfaten",
        "mp_thompson_aero_nifaten", "mp_thompson_aero_entry_density",
        "mp_thompson_aero_tau1_density", "mp_thompson_aero_nwfa_work_m3",
        # WRF v4.6.1's hydrometeor tendency accumulators (development branch).
        "mp_thompson_aero_qcten", "mp_thompson_aero_qrten",
        "mp_thompson_aero_nrten", "mp_thompson_aero_qiten",
        "mp_thompson_aero_niten",
        # WRF's REAL vapour and temperature tendencies
        # (development branch).
        "mp_thompson_aero_qvten", "mp_thompson_aero_tten",
        # The frozen carriers' snow, graupel and graupel-number accumulators
        # (a860663b5, the delivered frozen-carrier launch).
        "mp_thompson_aero_qsten", "mp_thompson_aero_qgten",
        "mp_thompson_aero_ngten",
        "mp_thompson_aero_condensation_rate",
    )
    assert AEROSOL_SCRATCH_SLOTS == expected_slots
    assert AEROSOL_INT_SCRATCH_SLOTS == ()
    poisoned_calls = {slot: 0 for slot in expected_slots}
    apply_calls = {False: 0, True: 0}

    results = {}
    for poison in (False, True):
        state, forcing = _build_states(cp, cfg, bubble=True, wind=FORECAST_U)
        _attach_specified_boundaries(state, forcing, cfg)
        # Through the production init path, so the state whose bitwise
        # identity is being asserted is the state a forecast integrates.
        initialize_physics(state, cfg)
        real_apply = dycore.apply_microphysics

        def wrapper(target, run_cfg, dt, _poison=poison,
                    _real=real_apply, **kwargs):
            call = apply_calls[_poison]
            apply_calls[_poison] += 1
            owned = {slot for slot in target._scratch
                     if slot.startswith("mp_thompson_aero_")}
            assert owned <= set(expected_slots), owned
            if call:
                assert owned == set(expected_slots), owned
                ranges = []
                for slot in expected_slots:
                    buf = target._scratch[slot]
                    assert buf.dtype == cp.float32 and buf.flags.c_contiguous
                    start = int(buf.data.ptr)
                    end = start + int(buf.nbytes)
                    assert start and end > start, slot
                    for other, other_start, other_end in ranges:
                        assert end <= other_start or other_end <= start, (slot, other)
                    ranges.append((slot, start, end))
            if _poison:
                for slot in expected_slots:
                    buf = target._scratch.get(slot)
                    if buf is None:
                        assert call == 0, slot
                        continue
                    if slot in AEROSOL_INT_SCRATCH_SLOTS:
                        buf.fill(7)
                    else:
                        buf.fill(cp.float32(-3.25e5))
                    poisoned_calls[slot] += 1
            return _real(target, run_cfg, dt, **kwargs)

        dycore.apply_microphysics = wrapper
        try:
            for _ in range(steps):
                dycore.step(state, cfg)
        finally:
            dycore.apply_microphysics = real_apply
        cp.cuda.Stream.null.synchronize()
        results[poison] = {name: cp.asnumpy(getattr(state, name)).copy()
                           for name in tracked}

    # The poisoning must actually have reached live buffers, or this proves
    # nothing: the slots are created on the first call, so from step 2 on
    # every one of them is overwritten before the adapter runs.
    # (This read ``len(AEROSOL_SCRATCH_SLOTS) >= 15``, a count from when the
    # six optional probe outputs were forecast residents; it failed at
    # aa5b9607f with nine.  What it means is asserted directly instead:
    # every slot the adapter draws was live on every call after the first,
    # and so poisoned on each of them.)
    missing = [slot for slot in AEROSOL_SCRATCH_SLOTS
               if slot not in state._scratch]
    assert not missing, missing
    assert apply_calls == {False: steps, True: steps}, apply_calls
    assert poisoned_calls == {slot: steps - 1 for slot in expected_slots}, poisoned_calls

    differing = [name for name in tracked
                 if not np.array_equal(results[False][name],
                                       results[True][name])]
    assert not differing, (
        f"after {steps} steps the poisoned run differs from the clean run "
        f"in {differing}; at least one mp_thompson_aero_* scratch slot is "
        "read before it is written, so the previous step's value leaks "
        "into this one")


@requires_gpu
def test_g4_health_validator_accepts_the_whole_forecast_state():
    """The production health gate, on the state the forecast ended in.

    ``health.collect_state_fields`` gained ``nwfa``/``nifa`` in WP-10 and
    ``rule_for_field`` classifies them as ``moment`` (non-negative, finite).
    A depleted boundary zone must NOT trip it -- that is part of what makes
    the LBC deviation silent, and it is asserted here rather than assumed.

    RE-DERIVED, not re-asserted.  The forecast now runs through
    ``initialize_physics``, so a PhysicsDriver is attached and the census
    covers the driver's surface/held/microphysics fields too -- measured 240
    entries against 127 for the same run with no driver.  The extra 113 are
    the reason this is re-derived: the gate that used to see only the state
    now also sees the seven canonical ``mp_*`` accumulators, and it sees them
    TWICE (once as ``surface.microphysics.*`` and once as
    ``surface.microphysics.scratch.mp_*``) precisely because the driver
    aliases them rather than copying, which is what
    ``microphysics_scratch_slots(28)`` bought.
    """
    import cupy as cp

    _tables_or_skip()
    _cfg, record = _forecast(cp, bubble=True)

    from woof.core import health

    validator = health.StateHealthValidator(record["state"])
    validator.require_healthy(phase="mp28-forecast-smoke-final")

    census = {field.name for field in health.collect_state_fields(
        record["state"])}
    assert {"nwfa", "nifa", "nc"} <= census, (
        "the aerosol tracers are not in the health census, so the gate above "
        f"proved nothing about them; census has {sorted(census)}")
    # The three tracers are classified, not merely present.
    for name in ("nwfa", "nifa", "nc"):
        assert health.rule_for_field(name).status_class == "moment", name

    # The driver's accumulators are in the census, and they ARE the scratch
    # slots -- both spellings present, and the same device buffers.
    for name in ("rainnc", "rainncv", "sr", "snownc", "snowncv",
                 "graupelnc", "graupelncv"):
        assert f"surface.microphysics.{name}" in census, name
        assert f"surface.microphysics.scratch.mp_{name}" in census, name
    driver = record["driver"]
    state = record["state"]
    assert driver.microphysics.rainnc is state.scratch(
        state.mup.shape, "mp_rainnc")
    assert driver.microphysics.hailnc is None, (
        "mp_gt_driver has no hail category; a HAILNC accumulator on an "
        "mp=28 driver is NSSL's set leaking in")

    # And the census really did grow because the driver is attached: an
    # mp=28 census with no driver is state + LBC only.
    assert len(census) > 200, len(census)
    print(f"\nmp=28 health census with the production driver attached: "
          f"{len(census)} fields")


@requires_gpu
def test_attaching_the_physics_driver_does_not_move_the_forecast():
    """WIRING THE INIT PATH DID NOT MOVE THIS GATE'S NUMBERS.  Measured.

    Wave 4's G4 forecast installed the aerosol profile by calling
    ``microphysics.microphysics_init`` directly, on a state with NO
    PhysicsDriver.  Wave 5 routes it through ``initialize_physics``, which
    also attaches the driver -- so every bound, every ring check and every
    published G4 number in this file is now measured on a state that carries
    one.  That is a change to the experiment, and it has to be shown not to
    be a change to the physics before any of those numbers can be reported
    as unchanged.

    The claim is BITWISE, over the full 150-step convective run, on every
    prognostic and every radiation-facing diagnostic -- plus the RAINNC
    accumulation, because the driver's ``accept_microphysics`` is the one
    thing that genuinely runs in one case and not the other.

    It also protects a future edit: if someone gives the driver a feedback
    path into the post-RK microphysics state, this goes red and the numbers
    this file publishes have to be re-measured rather than re-quoted.
    """
    import cupy as cp

    from woof.core import dycore, microphysics
    from woof.core.physics import initialize_physics

    _tables_or_skip()
    cfg = _forecast_config()
    tracked = ("qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni",
               "nwfa", "nifa", "effc", "effi", "effs", "thp", "php",
               "mup", "u", "v", "w", "h_diabatic")

    results = {}
    for with_driver in (False, True):
        state, forcing = _build_states(cp, cfg, bubble=True,
                                       wind=FORECAST_U)
        _attach_specified_boundaries(state, forcing, cfg)
        if with_driver:
            initialize_physics(state, cfg)          # wave-5 production path
        else:
            microphysics.microphysics_init(state, cfg)   # wave-4 path
        cp.cuda.Stream.null.synchronize()
        for _ in range(FORECAST_STEPS):
            dycore.step(state, cfg)
        cp.cuda.Stream.null.synchronize()
        results[with_driver] = {
            name: cp.asnumpy(getattr(state, name)).copy()
            for name in tracked}
        results[with_driver]["rainnc_total"] = float(cp.sum(
            state.scratch(state.mup.shape, "mp_rainnc")))

    differing = [name for name in tracked
                 if not np.array_equal(results[False][name],
                                       results[True][name])]
    assert not differing, (
        f"after {FORECAST_STEPS} steps the driver-attached run differs from "
        f"the driverless one in {differing}; wiring initialize_physics has "
        "changed the trajectory, so every G4 number in this file must be "
        "re-measured")
    assert results[False]["rainnc_total"] == results[True]["rainnc_total"], (
        results[False]["rainnc_total"], results[True]["rainnc_total"])
    print(f"\nPhysicsDriver attachment is bitwise inert over "
          f"{FORECAST_STEPS} mp=28 steps; domain-total RAINNC "
          f"{results[True]['rainnc_total']:.9f} mm either way")


# ---------------------------------------------------------------------------
# The registered LBC deviation, measured.
# ---------------------------------------------------------------------------

def _depletion_metrics(cfg, record):
    """Front position, sweep fraction and loss rate, from one clear run.

    The front position at time t is the x coordinate (m from the inflow
    face) where the mid-level nwfa profile last crosses the midpoint between
    WRF's floor and the undisturbed upstream value.  Using the midpoint
    rather than the floor itself makes the metric insensitive to how sharply
    the 5th-order advection resolves the front.
    """
    rows = np.asarray(record["nwfa_rows"], dtype=np.float64)
    k = cfg.nz // 2
    undisturbed = float(record["nwfa_initial"][k].max())
    threshold = 0.5 * (NWFA_FLOOR + undisturbed)
    sz = int(cfg.spec_zone)

    fronts = []
    for row in rows:
        # Scan from the downstream end back: the last index that is still
        # undisturbed marks the front, so a depleted pocket left behind by a
        # cloud cannot masquerade as the front.
        idx = sz
        for i in range(sz, cfg.nx - sz):
            if row[i] <= threshold:
                idx = i
            else:
                break
        fronts.append((idx - sz + 1) * cfg.dx)
    fronts = np.asarray(fronts)
    t = (np.arange(len(fronts)) + 1) * cfg.dt

    # Fit over the window where the front is inside the domain and has moved
    # past the first cell, so neither the initial ramp nor a saturated front
    # biases the slope.
    usable = (fronts > cfg.dx) & (fronts < (cfg.nx - 2 * sz - 1) * cfg.dx)
    speed = float(np.polyfit(t[usable], fronts[usable], 1)[0])
    wind = float(record["wind"])

    mean = np.asarray(record["nwfa_interior_mean"])
    mean0 = float(record["nwfa_initial"][record["interior"]].mean())
    mean_i = np.asarray(record["nifa_interior_mean"])
    mean_i0 = float(record["nifa_initial"][record["interior"]].mean())

    return {
        "fronts_m": fronts,
        "front_speed_ms": speed,
        "wind_ms": wind,
        "front_speed_ratio": speed / wind,
        "final_front_m": float(fronts[-1]),
        "advective_fetch_m": wind * len(fronts) * cfg.dt,
        "nwfa_retained": float(mean[-1]) / mean0,
        "nifa_retained": float(mean_i[-1]) / mean_i0,
        "nwfa_lost_fraction": 1.0 - float(mean[-1]) / mean0,
        "nifa_lost_fraction": 1.0 - float(mean_i[-1]) / mean_i0,
        # L/U: the time for the inflow air to cross the whole domain, after
        # which nothing of the initial aerosol field is left anywhere.  This
        # is the number that generalises -- 56 km at 20 m/s here, but a
        # 1000 km operational domain at the same wind is 13.9 hours.
        "ventilation_time_s": cfg.nx * cfg.dx / wind,
        "run_seconds": len(mean) * cfg.dt,
        "swept_fraction": min(1.0, wind * len(mean) * cfg.dt
                              / (cfg.nx * cfg.dx)),
        "surface_emission_per_kg_s": float(record["nwfa2d"].max()),
    }


@requires_gpu
def test_lbc_aerosol_depletion_rate_is_measured_on_a_cloud_free_forecast():
    """THE registered-deviation measurement.  This number did not exist.

    A cloud-free specified-BC forecast with a uniform 20 m/s inflow.  The
    test first proves the run really is cloud-free -- ``qc``, ``qr``, ``qi``,
    ``qs`` and ``qg`` are identically zero on every step, so ``nwfa`` has no
    microphysical sink and ``nifa`` has none either -- and then measures what
    the boundary policy alone does.

    The physics is elementary and that is the point: with zero inflow and
    pure advection, the aerosol-free air occupies the upstream ``U*t`` of the
    domain, so the depletion front travels at the inflow WIND speed.  The
    test asserts the measured front speed matches ``FORECAST_U`` to within
    15%, which is what makes the published rate a LAW a user can apply to
    their own domain (``U*t`` of fetch, ``L/U`` to sweep the whole domain)
    rather than one number from one configuration.

    The only aerosol SOURCE in the scheme is the fixed surface emission
    ``nwfa2d`` at k = 0 (mp_gt_driver:1316-1327), and the test measures how
    small it is against the advective loss.
    """
    import cupy as cp

    _tables_or_skip()
    cfg, record = _forecast(cp, bubble=False)

    assert not record["nonfinite"], record["nonfinite"][:8]
    assert max(record["condensate_max"]) == 0.0, (
        "the depletion run is not cloud-free (max condensate "
        f"{max(record['condensate_max']):.3e} kg/kg), so the aerosol loss "
        "cannot be attributed to the boundary policy alone")
    assert max(record["w_max"]) < 0.05, (
        f"the clear run is not at rest (max |w| {max(record['w_max']):.3e} "
        "m/s); a ringing initial state would advect aerosol vertically and "
        "contaminate the horizontal measurement")

    metrics = _depletion_metrics(cfg, record)

    # 1.  The front travels at the inflow wind speed.  This is the law.
    assert metrics["front_speed_ratio"] == pytest.approx(1.0, abs=0.15), (
        f"depletion front speed {metrics['front_speed_ms']:.2f} m/s against "
        f"an imposed inflow of {metrics['wind_ms']:.2f} m/s "
        f"(ratio {metrics['front_speed_ratio']:.3f}); if these do not agree "
        "the published 'U*t of fetch' rule is not the right description")

    # 2.  Aerosol really is being lost, and it is not a rounding effect.
    assert metrics["nwfa_retained"] < 0.6, (
        "the boundary zone did not deplete at all, so this test measures "
        f"nothing (retained {metrics['nwfa_retained']:.3f})")
    assert metrics["nifa_retained"] < 0.6, metrics["nifa_retained"]

    # 3.  The stated signature of the deviation: it stays finite, stays
    #     non-negative, and never leaves the state unhealthy.
    state = record["state"]
    assert float(cp.min(state.nwfa)) >= 0.0
    assert float(cp.min(state.nifa)) >= 0.0
    from woof.core import health
    health.StateHealthValidator(state).require_healthy(
        phase="mp28-lbc-depletion")

    # 4.  The one source in the scheme is far too small to compensate.  The
    #     emission adds nwfa2d*dt per step at k = 0 only, with no clamp.
    emitted = metrics["surface_emission_per_kg_s"] * metrics["run_seconds"]
    initial_k0 = float(record["nwfa_initial"][0].max())
    assert 0.0 < emitted < 0.1 * initial_k0, (
        f"surface emission replaced {emitted:.3e} kg^-1 over the run against "
        f"an initial k=0 loading of {initial_k0:.3e} kg^-1")

    print("\nmp=28 lateral-boundary aerosol depletion, measured")
    for key, value in metrics.items():
        if key == "fronts_m":
            continue
        print(f"  {key:26s} {value!r}")


@requires_gpu
def test_the_specified_zone_ring_ends_at_exactly_zero_aerosol():
    """PINNED, MEASURED CONSEQUENCE of the registered LBC deviation.

    The depletion measurement above is about the INTERIOR.  The
    specified-zone ring itself is a separate and sharper statement, and it
    had never been looked at: WRF's clipped microphysics tiles never touch
    the ring, so the terminal clamp that holds ``nwfa >= 11.1e6`` in the
    columns the scheme updates does NOT run there.  What the ring carries is
    whatever ``flow_dep_bdy`` left, and with no aerosol in the boundary file
    that is exactly ZERO.  In this cloud-free run the same zero reaches the
    interior too: every interior column takes WRF's ``:2020`` exit on every
    call, which leaves it as transport delivered it.  WRF forces
    ``qnwfa``/``qnifa`` at the boundary from the monthly dataset, so in WRF
    neither zero arises; both are this deviation's.

    Measured here with a purely zonal 20 m/s flow: the west (inflow) face,
    and BOTH tangential faces where the normal velocity is zero, end at
    exactly 0.0; only the east (outflow) face retains the interior value,
    and even there the two corner cells are zeroed by the corner treatment.
    So three of four faces, not one.

    This matters to three consumers that are not microphysics:
    ``QNWFA``/``QNIFA`` written to wrfout, any nest whose boundary is fed
    from this ring, and any diagnostic that reads the full array.  It is a
    consequence of deviation D9c, not a separate defect, and it closes when
    the LBC ingest carries qnwfa/qnifa.
    """
    import cupy as cp

    from woof.core import microphysics

    _tables_or_skip()
    cfg, record = _forecast(cp, bubble=False)
    state = record["state"]

    faces = {
        "south": state.nwfa[:, 0, :],
        "north": state.nwfa[:, -1, :],
        "west": state.nwfa[:, :, 0],
        "east": state.nwfa[:, :, -1],
    }
    zero_fraction = {name: float(cp.mean((value == 0.0).astype(cp.float32)))
                     for name, value in faces.items()}
    print("\nfraction of the specified ring at exactly 0 aerosol: "
          + ", ".join(f"{k}={v:.3f}" for k, v in zero_fraction.items()))

    for name in ("south", "north", "west"):
        assert zero_fraction[name] == 1.0, (
            f"the {name} face is no longer uniformly zero "
            f"({zero_fraction[name]:.3f}); the published statement that "
            "three of four faces end at exactly zero aerosol is stale")
    assert 0.0 < zero_fraction["east"] < 0.5, (
        "the outflow face should retain the interior value except at the "
        f"corners; measured zero fraction {zero_fraction['east']:.3f}")

    # Inside the ring the scheme ran on no column: this run never has
    # condensate and is nowhere supersaturated over ice, so every call
    # returned from every interior column at :2020 and left it untouched,
    # and the inflow's zero reaches the interior exactly as WRF would leave
    # it.  The floor holds wherever a call did update a column (the bounds
    # recorded per step), and there is none here to hold it.
    sz = int(cfg.spec_zone)
    assert record["updated_columns"], "no call recorded its :2020 decision"
    assert max(record["updated_columns"]) == 0, (
        "a column of the cloud-free run reached the terminal apply, so the "
        "zero this test and its evidence page describe is not the whole "
        f"interior's any more: {record['updated_columns'][:12]}")
    assert not record["skipped_column_violations"], (
        record["skipped_column_violations"][:12])
    assert not record["bound_violations"], record["bound_violations"][:12]
    assert float(cp.min(state.nwfa[:, sz:cfg.ny - sz, sz:cfg.nx - sz])) == 0.0

    # The ring is exactly the region microphysics does not update.
    ring = microphysics.spec_zone_ring_slices(cfg.ny, cfg.nx, sz)
    ring_min = min(float(cp.min(state.nwfa[slc])) for slc in ring)
    assert ring_min == 0.0


#: The second inflow speed the front-speed law is measured at.  Half of
#: FORECAST_U, so a metric that accidentally encoded the domain geometry or
#: the step count instead of the wind cannot pass both.
FORECAST_U_SLOW = 10.0


@requires_gpu
def test_the_depletion_front_tracks_the_inflow_speed_at_a_second_wind():
    """One measurement is a number; two make it a law.

    The published guidance -- "the upstream ``U*t`` of your domain is at
    WRF's aerosol floor after time t, and the whole domain after ``L/U``" --
    is only usable if the front speed really is the WIND speed and not some
    artefact of this domain, this timestep or this advection order.  So the
    identical experiment runs at half the inflow speed and the front must
    halve with it.

    This also protects the metric itself: a front detector that had latched
    onto a fixed cell index, or onto the step count, would pass at 20 m/s
    and fail here.
    """
    import cupy as cp

    _tables_or_skip()
    fast_cfg, fast = _forecast(cp, bubble=False, wind=FORECAST_U)
    slow_cfg, slow = _forecast(cp, bubble=False, wind=FORECAST_U_SLOW)

    assert max(slow["condensate_max"]) == 0.0
    fast_m = _depletion_metrics(fast_cfg, fast)
    slow_m = _depletion_metrics(slow_cfg, slow)

    assert slow_m["front_speed_ratio"] == pytest.approx(1.0, abs=0.15), (
        f"at {FORECAST_U_SLOW} m/s the front moved at "
        f"{slow_m['front_speed_ms']:.2f} m/s")
    ratio = fast_m["front_speed_ms"] / slow_m["front_speed_ms"]
    assert ratio == pytest.approx(FORECAST_U / FORECAST_U_SLOW, rel=0.20), (
        f"front speed did not scale with the wind: "
        f"{fast_m['front_speed_ms']:.2f} m/s at {FORECAST_U} m/s vs "
        f"{slow_m['front_speed_ms']:.2f} m/s at {FORECAST_U_SLOW} m/s "
        f"(ratio {ratio:.3f}, expected "
        f"{FORECAST_U / FORECAST_U_SLOW:.3f})")
    # Half the wind sweeps half the fetch, so it must retain strictly more.
    assert slow_m["nwfa_retained"] > fast_m["nwfa_retained"] + 0.05, (
        f"halving the inflow did not slow the depletion: retained "
        f"{slow_m['nwfa_retained']:.3f} vs {fast_m['nwfa_retained']:.3f}")

    print(f"\nfront speed vs inflow: {FORECAST_U} m/s -> "
          f"{fast_m['front_speed_ms']:.3f} m/s; {FORECAST_U_SLOW} m/s -> "
          f"{slow_m['front_speed_ms']:.3f} m/s; "
          f"retained {fast_m['nwfa_retained']:.4f} / "
          f"{slow_m['nwfa_retained']:.4f}")


@requires_gpu
def test_the_depletion_measurement_is_published_in_the_evidence_document():
    """Doc/measurement lockstep.

    The evidence document quotes this measurement.  If the measurement moves
    and the document does not, the document becomes a false claim that
    someone will quote back.  The check is on the values a reader acts on --
    the front-speed law, the retained fraction and the ventilation time --
    with a tolerance wide enough to survive a different GPU and narrow enough
    that a real change in behaviour fails.
    """
    import cupy as cp

    _tables_or_skip()
    assert EVIDENCE_DOC.exists(), f"missing evidence document {EVIDENCE_DOC}"
    text = EVIDENCE_DOC.read_text(encoding="utf-8")

    cfg, record = _forecast(cp, bubble=False)
    metrics = _depletion_metrics(cfg, record)

    published = dict(re.findall(
        r"^\|\s*`([A-Za-z0-9_]+)`\s*\|\s*([-+0-9.eE]+)\s*\|", text,
        flags=re.MULTILINE))
    missing = {"front_speed_ms", "front_speed_ratio", "nwfa_retained",
               "nifa_retained", "ventilation_time_s", "swept_fraction",
               "surface_emission_per_kg_s"} - set(published)
    assert not missing, (
        f"the evidence document does not publish {sorted(missing)}; it must "
        "carry the measured depletion table this test recomputes")

    for key, tol in (("front_speed_ms", 0.10),
                     ("front_speed_ratio", 0.10),
                     ("nwfa_retained", 0.10),
                     ("nifa_retained", 0.10),
                     ("surface_emission_per_kg_s", 0.10),
                     ("swept_fraction", 1.0e-9),
                     ("ventilation_time_s", 1.0e-9)):
        assert float(published[key]) == pytest.approx(
            metrics[key], rel=tol), (
            f"{key}: document says {published[key]}, this run measures "
            f"{metrics[key]!r}")


# ---------------------------------------------------------------------------
# Three reachability gaps this file MEASURED in wave 4 and PINNED as passing
# tests that asserted the defect still existed.  All three are now CLOSED,
# and every one of them has been replaced here by its INVERSE rather than
# deleted: a pinned gap that is merely removed leaves no gate at all, and
# each of these defects is silent -- no NaN, no negative, no health trip --
# so only a test standing on the closed side of it can stop it reopening.
# ---------------------------------------------------------------------------

def test_the_runtime_history_lane_admits_28_for_refl_10cm():
    """INVERSE of ``test_gap_runtime_history_lane_still_omits_28...``.

    ``woof/core/refl.py`` admitted mp_physics=28 in WP-11a and the adapter
    threaded ``refl_10cm_due`` correctly, but the RUNTIME that decides when a
    history step is due tested ``cfg.mp_physics in (1, 6, 8, 10, 18)`` at
    three separate sites -- so a forecast driven through ``woof.runtime``
    wrote wrfout frames whose REFL_10CM was never produced.

    WRF's own structure is what puts 28 in: ``mp_gt_driver`` reaches
    ``calc_refl10cm`` from ONE call site (``module_mp_thompson.F:1458``)
    gated on ``diagflag .and. do_radar_ref == 1`` (:1450) and never on
    ``is_aerosol_aware``, so the aerosol-aware package publishes REFL_10CM on
    exactly the same cadence as the classic one.

    The gate is on the NAMED constant, and on the absence of the old inline
    tuples, because the original defect was precisely that three inlined
    copies drifted apart.
    """
    from woof import runtime
    from woof.core import refl

    assert 28 in runtime.REFL_10CM_MICROPHYSICS, (
        "woof/runtime.py no longer admits mp=28 for REFL_10CM; an mp=28 "
        "forecast writes history frames with no radar data and refl.py's "
        "consume-once contract reports an unconsumed stash")
    # 16 (WDM6) joined with the WDM6 port: module_mp_wdm6.F reaches
    # refl10cm_wdm6 from one site (:291) under the same
    # ``diagflag .and. do_radar_ref == 1`` guard, so the history lane must
    # admit it for the same reason it admits 28.  Since 1.9.1 the
    # membership is pinned to the accepted menu itself (every moist
    # accepted scheme; tests/test_mp_accepted_builds.py) -- 9 and 50
    # shipped through the hand-typed set that used to sit here.
    from woof.config import MP_PHYSICS_ACCEPTED

    # FAST_SBM (30) is the one moist scheme out: its polar-scattering
    # diagnostic is not ported, and woof.core.fast_sbm_runtime.apply
    # refuses a due frame, so the history lane never asks it for one.
    # CAM MG (11) never writes REFL_10CM in WRF; its retained zero field is
    # not a diagnostic (woof.da.obsop's cited absence), so it is out too,
    # as are old Eta Ferrier (95) and MADWRF (96), which write none either.
    assert (set(runtime.REFL_10CM_MICROPHYSICS)
            == {mp for mp in MP_PHYSICS_ACCEPTED if mp} - {30, 11, 95, 96})
    # ... and the set is cross-checked against the estimator's own
    # admission set rather than left as one hand-typed tuple, which is how
    # this pin went stale.  The two sets are no longer EQUAL, and the
    # difference is documented scheme behaviour, not drift: preflight
    # prices the shared ``refl`` translation unit for exactly the schemes
    # that LOAD it, while the history lane consumes the stash for every
    # moist scheme.  mp=9 fills REFL_10CM inside its own milbrandt2
    # diagnostics kernel, mp=18 from its own radardd02 diagnostic
    # (woof.core.nssl2_diagnostics) and mp=50 inside p3_main -- on the
    # device for the default CUDA backend, on the host only for the
    # reference arm -- so all three consume without loading ``refl``.
    # Each absence carries its reason in woof/core/preflight.py's
    # _SELF_REFLECTIVITY_MICROPHYSICS, which domain_kernel_modules now
    # requires an entry in, and that table is what the difference is
    # pinned to, so the next scheme that fills the slot itself moves the
    # pin by landing its reason.  Every priced scheme must still be
    # consumed, or a computed field strands at the next history frame.
    from woof.core import preflight as pf

    assert (set(pf._REFLECTIVITY_MICROPHYSICS)
            <= set(runtime.REFL_10CM_MICROPHYSICS))
    # FAST_SBM's row in that table records the third answer: it neither
    # loads refl.cu nor fills the slot, and the history lane leaves it out.
    assert (set(runtime.REFL_10CM_MICROPHYSICS)
            - set(pf._REFLECTIVITY_MICROPHYSICS)
            == set(pf._SELF_REFLECTIVITY_MICROPHYSICS) - {30}
            # 7, 15, 51-53, 55 and 56 landed with their own radar kernels,
            # each with its reason row in _SELF_REFLECTIVITY_MICROPHYSICS;
            # UDM (27) and RCON (29) joined with theirs in 3.0.
            == {4, 5, 7, 9, 15, 18, 24, 27, 29, 50, 51, 52, 53, 55, 56, 97, 900})

    runtime_source = (pathlib.Path(__file__).resolve().parent.parent
                      / "woof" / "runtime.py").read_text(encoding="utf-8")
    assert "mp_physics in (1, 6, 8, 10, 18)" not in runtime_source, (
        "an inlined refl-admission tuple is back in runtime.py; the three "
        "gates must all read REFL_10CM_MICROPHYSICS or they will drift again")
    assert runtime_source.count("REFL_10CM_MICROPHYSICS") >= 4, (
        "runtime.py carries three separate REFL_10CM gates plus the "
        "definition; fewer references means one of them stopped using it")

    refl_source = pathlib.Path(refl.__file__).read_text(encoding="utf-8")
    assert "28" in refl_source


@requires_gpu
def test_an_mp28_forecast_can_be_checkpointed():
    """INVERSE of ``test_gap_an_mp28_forecast_cannot_be_checkpointed``.

    ``woof/io/restart.py::MICROPHYSICS_ALGORITHM_IDENTITIES`` listed only
    {0, 1, 6, 8, 10, 18} and the manifest builder resolved the scheme
    identity through it with no fallback, so the very FIRST checkpoint of an
    mp=28 run raised ``RestartManifestError`` -- the run could not be
    checkpointed, therefore could not be resumed, and a ``woof.runtime``
    forecast with a restart cadence aborted at the first restart instant
    rather than at admission.

    This is the forecast-level half of the claim: a real ``DomainState`` with
    lateral boundaries attached and the production aerosol profile installed,
    written to a real checkpoint.  The per-field round-trip is
    ``tests/test_preflight.py::test_mp28_survives_a_restart_round_trip``.
    """
    import tempfile

    import cupy as cp

    from woof.core.physics import initialize_physics
    from woof.io import restart

    _tables_or_skip()
    assert 28 in restart.MICROPHYSICS_ALGORITHM_IDENTITIES, (
        "restart can no longer identify mp=28, so an mp=28 forecast cannot "
        "be checkpointed at all")

    cfg = _forecast_config()
    state, forcing = _build_states(cp, cfg, bubble=False, wind=FORECAST_U)
    _attach_specified_boundaries(state, forcing, cfg)
    initialize_physics(state, cfg)
    cp.cuda.Stream.null.synchronize()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "mp28.npz"
        restart.write_restart(path, state, cfg)
        assert path.is_file() and path.stat().st_size > 0

    # The rest of the restart contract, which is what made this a one-line
    # gap rather than an unported feature.
    from woof import state_serialization_contract as contract

    assert {"nc", "nwfa", "nifa", "nwfa2d", "nifa2d"} <= set(
        contract.STATE_SERIALIZED_ATTRS)
    assert {"nc0", "nwfa0", "nifa0"} <= set(restart.STATE_REBUILT_ATTRS)


def _production_callers_of_microphysics_init() -> list[str]:
    """Every ``gpuwm/`` module that CALLS ``microphysics_init``.

    Deliberately the same scan that
    ``tests/test_physics_md_aerosol_claims.py`` runs, so the code gate and
    the documentation gate can never disagree about whether the hook is
    wired.  The definition site itself is not a caller: in
    ``microphysics.py`` only text BEFORE ``def microphysics_init`` counts.

    POSIX-spelled, like every other path this suite reports: the caller is a
    location in the source tree, and ``core/physics.py`` is the same location
    whichever separator the host writes it with.  ``str()`` on the relative
    path would make this gate compare ``core\\physics.py`` against the
    ``core/physics.py`` its assertions name, and fail on Windows for a reason
    that has nothing to do with where the hook is called from.
    """
    root = pathlib.Path(__file__).resolve().parent.parent / "woof"
    callers = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if path.name == "microphysics.py" and "def microphysics_init" in text:
            body = text.split("def microphysics_init", 1)[0]
            if "microphysics_init(" in body:
                callers.append(path.relative_to(root).as_posix())
            continue
        if re.search(r"\bmicrophysics_init\s*\(", text):
            callers.append(path.relative_to(root).as_posix())
    return callers


def test_microphysics_init_has_a_production_call_site():
    """THE INVERSE OF THE OLD PINNED GAP.  The profile is installed.

    For four waves this file carried
    ``test_gap_microphysics_init_has_no_production_call_site``, a PASSING
    test asserting that nothing in ``gpuwm/`` called
    ``microphysics.microphysics_init`` -- so every mp=28 forecast began at
    ``nwfa = nifa = 0``, was clamped to WRF's floors
    (``module_mp_thompson.F:3979-3982``) and ran a maritime-clean droplet
    population for its whole length.  That was the single largest
    forecast-relevant defect the port had left.

    It is closed, and this test is what stops it reopening.  The caller is
    ``woof/core/physics.py::initialize_physics``, woof's ``phy_init``,
    which is where WRF calls ``mp_init``
    (``phys/module_physics_init.F:1635``) and therefore where
    ``thompson_init`` runs (``:4522-4538``).  A refactor that dropped the
    call would restore the defect in complete silence -- nothing NaNs,
    nothing goes negative, no health check trips -- which is exactly why the
    scan is a test rather than a review item.
    """
    callers = _production_callers_of_microphysics_init()
    assert "core/physics.py" in callers, (
        "woof/core/physics.py no longer calls microphysics_init, so every "
        "mp=28 forecast is back to starting from nwfa = nifa = 0 and being "
        "clamped to WRF's floors (module_mp_thompson.F:3979-3982).  This is "
        f"the port's largest measured error, reopened.  Callers found: "
        f"{callers}")
    # And it is exactly one place.  A second caller is how "once per domain"
    # silently becomes "twice", or worse "once per step".
    assert callers == ["core/physics.py"], (
        "microphysics_init has more than one production caller "
        f"({callers}); WRF calls thompson_init from mp_init and nowhere "
        "else, and a per-step or duplicated call would overwrite an "
        "advected, activated and scavenged aerosol field with the synthetic "
        "profile")


@requires_gpu
def test_a_freshly_initialised_mp28_domain_carries_the_profile_not_zero():
    """The behavioural half: the state a user's forecast actually starts in.

    The scan above proves a call site exists.  This proves what it does, on a
    real ``DomainState`` built from a validated ``RunConfig`` and taken
    through the real ``initialize_physics`` -- the mp=28 domain the G4
    forecast below integrates.  Before the wiring, ``nwfa`` and ``nifa``
    were identically zero here and ``nwfa2d`` was never derived.

    The floors asserted are WRF's own profile constants
    (``module_mp_thompson.F:96-97`` ``naCCN1 = 50.0E6``, ``:94-95``
    ``naIN1 = 0.5E6``), which are 4.5x and 100x above the terminal apply's
    clamp floors (``:3979-3982``) -- so this cannot be satisfied by a run
    that starts at zero and gets clamped.
    """
    import cupy as cp

    from woof.core.physics import initialize_physics

    _tables_or_skip()
    cfg = _forecast_config()
    state, forcing = _build_states(cp, cfg, bubble=False, wind=FORECAST_U)
    _attach_specified_boundaries(state, forcing, cfg)

    assert float(cp.max(state.nwfa)) == 0.0
    assert float(cp.max(state.nifa)) == 0.0
    assert float(cp.max(state.nwfa2d)) == 0.0

    driver = initialize_physics(state, cfg)
    cp.cuda.Stream.null.synchronize()

    assert driver.microphysics_init_receipt == {
        "thompson_aerosol_profile": {"ccn": True, "in": True}}
    nwfa = cp.asnumpy(state.nwfa)
    nifa = cp.asnumpy(state.nifa)
    assert nwfa.min() >= 50.0e6 * (1.0 - 1.0e-6), nwfa.min()
    assert nifa.min() >= 0.5e6 * (1.0 - 1.0e-6), nifa.min()
    # Strictly above WRF's terminal-clamp floors, which is what makes the
    # difference from the pre-wiring state a physics difference.
    assert nwfa.min() > NWFA_FLOOR
    assert nifa.min() > NIFA_FLOOR
    # Boundary-layer following: monotone decay from the surface.
    assert nwfa[0].min() > nwfa[-1].max()
    # nwfa2d is DERIVED from the filled surface value (:509-510); nifa2d is
    # not derived at all (WRF never writes one) and nc is never touched.
    assert float(cp.min(state.nwfa2d)) > 0.0
    assert float(cp.max(cp.abs(state.nifa2d))) == 0.0
    assert float(cp.max(cp.abs(state.nc))) == 0.0


@requires_gpu
def test_the_aerosol_profile_changes_the_forecast_measurably():
    """WHAT WRF'S SYNTHETIC AEROSOL PROFILE IS WORTH, as a forecast number.

    This test used to be a DEFECT measurement -- the cost of the missing
    ``microphysics_init`` call site.  The call site landed
    (``woof/core/physics.py::initialize_physics``; pinned by
    ``test_microphysics_init_has_a_production_call_site`` above), so the
    default run is now the FILLED one and this is a SENSITIVITY measurement:
    how much of an mp=28 forecast is decided by the CCN/IN loading it starts
    from.  The numbers are the same numbers -- the control run is unchanged
    and the counterfactual is the identical domain with the profile removed
    -- but what they now mean is "this is the part of the forecast the
    aerosol field owns", not "this is what a user is silently losing".

    The sibling column test
    ``test_mp28_runnable.py::test_the_unfilled_aerosol_profile_is_physics_visible_not_cosmetic``
    proves the difference is visible in one call.  Whether it matters to a
    FORECAST is a different question and can only be answered by running two.

    Both runs are IDENTICAL -- same domain, same sounding, same boundaries,
    same wind -- except that the counterfactual skips the production init
    path and therefore starts from ``nwfa = nifa = 0``.  With no aerosol the
    scheme's terminal clamp pins ``nwfa`` at ``11.1e6 kg^-1``
    (``module_mp_thompson.F:3979-3980``) in every column it updates, and the
    entry reads the same floor (``:1805``) wherever cloud forms: a
    maritime-clean CCN population, which activates fewer, larger droplets
    and rains out faster.
    Skipping the init path also leaves the counterfactual without a
    PhysicsDriver, which is measured to be BITWISE irrelevant to this
    forecast (:func:`test_attaching_the_physics_driver_does_not_move_the
    _forecast`), so the aerosol field remains the only live variable.

    The assertion is that the difference is LARGE and in the physically
    expected direction.  The SIZE is printed rather than pinned: pinning it
    would freeze a sensitivity that legitimately depends on the case, the
    resolution and the length of the run.
    """
    import cupy as cp

    _tables_or_skip()
    cfg, filled = _forecast(cp, bubble=True, initialise=True)
    _cfg, stripped = _forecast(cp, bubble=True, initialise=False)

    assert filled["init_receipt"] == {
        "thompson_aerosol_profile": {"ccn": True, "in": True}}, (
        "the control run did not install a profile through the production "
        f"init path (receipt {filled['init_receipt']!r})")
    assert stripped["init_receipt"] == {}

    nwfa_filled = float(filled["nwfa_initial"].mean())
    nwfa_stripped_final = stripped["nwfa_interior_mean"][-1]
    assert float(stripped["nwfa_initial"].max()) == 0.0, (
        "the counterfactual run started with aerosol, so it is not the "
        "aerosol-free comparison this measurement needs")

    rain_filled = filled["rain_sum"][-1]
    rain_stripped = stripped["rain_sum"][-1]
    assert rain_filled > 0.0 and rain_stripped > 0.0, (
        "neither run rained, so this comparison is empty")
    excess = rain_stripped / rain_filled - 1.0

    print("\nmp=28 with and without WRF's synthetic aerosol profile "
          f"({FORECAST_STEPS} steps, dt={FORECAST_DT} s)")
    print("  left column = the production default (profile installed by "
          "initialize_physics); right column = the aerosol-free "
          "counterfactual")
    print(f"  initial nwfa mean       {nwfa_filled:.4e} vs 0.0 kg^-1")
    print(f"  final interior nwfa     "
          f"{filled['nwfa_interior_mean'][-1]:.4e} vs "
          f"{nwfa_stripped_final:.4e} kg^-1")
    print(f"  domain-total RAINNC     {rain_filled:.6f} vs "
          f"{rain_stripped:.6f} mm  ({excess:+.1%})")
    print(f"  peak RAINNC             {max(filled['rain_max']):.6f} vs "
          f"{max(stripped['rain_max']):.6f} mm")
    print(f"  peak nc                 {max(filled['nc_max']):.4e} vs "
          f"{max(stripped['nc_max']):.4e} kg^-1")

    assert abs(excess) > 0.10, (
        "removing WRF's synthetic aerosol profile changed domain-total "
        f"surface rain by only {excess:+.2%}; if that is now true the "
        "published sensitivity must be re-stated")
    assert excess > 0.0, (
        "removing CCN reduced surface rain, which inverts the published "
        f"direction of the sensitivity ({excess:+.2%})")
    assert max(stripped["nc_max"]) < max(filled["nc_max"]), (
        "the aerosol-free run did not produce fewer droplets, so the "
        "mechanism described in the evidence document is not the one acting")
