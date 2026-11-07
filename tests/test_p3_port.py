"""P3 (``mp_physics=50``) port: reachability, table integrity, column sanity.

WHAT THIS MODULE CLAIMS, EXACTLY
--------------------------------
That the scheme is REACHABLE through the shipped seams, that its hard data
dependency FAILS CLOSED, and that a real column integration through those
seams produces a physically sane, conserving state.

WHAT IT DOES NOT CLAIM
----------------------
Any agreement with WRF.  There is no oracle comparison IN THIS MODULE:
every number below is checked against PHYSICS or against woof's own
float64 mirror of its own arithmetic, never against Fortran output.  The
WRF-Fortran oracle campaign ran separately -- twelve fixtures through the
unmodified ``phys/module_mp_p3.F``, results carried verbatim in the
registry row's first warning -- and the row still says
``implemented-unverified`` because per-step agreement with Fortran is not
evidence of forecast skill: no matched WRF forecast run and no comparison
against observations exists.

The conservation and work assertions are guarded by a MUTATION CONTROL
(:func:`test_the_column_assertions_fail_when_the_solver_is_stubbed_out`):
it stubs ``p3_main`` to a no-op and requires the same assertion helper to
fail, so a green run here cannot mean "the test asserts nothing".
"""

from __future__ import annotations

import dataclasses
import hashlib
import shutil

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import RunConfig, validate_run_config

# k = 0 is the surface throughout, matching woof's column slab convention.
NK = 40
NCOL = 3
DT = 20.0
NSTEPS = 15


def _cfg(**overrides) -> RunConfig:
    values = dict(
        nx=4, ny=3, nz=8, dx=1000.0, dy=1000.0, ztop=10000.0,
        dt=10.0, run_seconds=10.0, moist=True, mp_physics=50,
        sf_sfclay_physics=91, sf_surface_physics=2, bl_pbl_physics=1,
        num_soil_layers=4, cu_physics=0,
    )
    values.update(overrides)
    return RunConfig(**values)


# ---------------------------------------------------------------------------
# 1.  The selector is reachable, and its two unported siblings are refused.
# ---------------------------------------------------------------------------

def test_validate_run_config_admits_mp_physics_50():
    cfg = _cfg()
    assert validate_run_config(cfg) is cfg


@pytest.mark.parametrize(
    "value,must_say",
    (
    ),
)
def test_the_unported_p3_variants_are_refused_by_name(value, must_say):
    """A user who asked for two ice categories is owed the reason.

    WRF v4.6.1 maps P3 across mp_physics 50/51/52/53
    (Registry.EM_COMMON:3038-3041); all four packages now run.
    Historical unsupported-package checks have no remaining rows.
    message that makes the number look meaningless", which invites the user
    to conclude P3 is absent entirely.
    """
    with pytest.raises(ValueError) as caught:
        validate_run_config(_cfg(mp_physics=value))
    message = str(caught.value)
    assert f"mp_physics={value}" in message
    assert must_say in message
    # and it must point at the option that DOES exist
    assert "mp_physics=50" in message


def test_the_schema_message_names_50_when_it_refuses_a_neighbour():
    """54 is unassigned in WRF; the refusal must still advertise 50."""
    with pytest.raises(ValueError) as caught:
        validate_run_config(_cfg(mp_physics=54))
    assert "50 (P3 one-category)" in str(caught.value)


def test_the_driver_dispatch_and_slot_map_know_the_option():
    from woof.core.physics import (
        microphysics_scheme_sr_available,
        microphysics_scratch_slots,
    )

    slots = dict(microphysics_scratch_slots(50))
    assert set(slots) == {"rainnc", "rainncv", "sr", "snownc", "snowncv"}
    # P3 has ONE ice category and no graupel accumulator anywhere in its
    # driver arm (module_microphysics_driver.F:1590-1595).
    assert "graupelnc" not in slots and "hailnc" not in slots
    assert microphysics_scheme_sr_available(50) is True


def test_microphysics_apply_rejects_50_only_when_the_state_is_dry():
    """The generic 'unknown mp_physics' arm must no longer catch 50."""
    from woof.core.microphysics import apply

    state = type("S", (), {"qv": None})()
    with pytest.raises(ValueError) as caught:
        apply(state, _cfg(), 10.0)
    assert "requires" in str(caught.value)
    assert "unknown mp_physics" not in str(caught.value)


# ---------------------------------------------------------------------------
# 2.  The registry row is accurate.
# ---------------------------------------------------------------------------

def test_the_registry_row_claims_no_more_than_was_measured():
    import json
    import pathlib

    import woof

    path = pathlib.Path(woof.__file__).parent / "physics_registry_v2.json"
    row = json.loads(path.read_text(encoding="utf-8"))[
        "components"]["microphysics"]["options"]["p3-mp50"]
    assert row["implemented"] is True
    assert row["maturity"] == "implemented-unverified"
    # "template" since the registered HRRR preparation profile
    # (p3-mp50-ysu-mm5-noah-rrtmg-legacy-v1) made P3 a template
    # selection rather than a per-domain override.
    assert row["reachability"]["state"] == "template"
    assert row["selectors"] == {"mp_physics": 50}
    evidence = row["warnings"][0]
    # The whole point of the label: it claims the measurement that was
    # made and nothing past it.  The column oracle (tools/p3_wrf461_oracle)
    # ran, so the warning must carry the measured result and its one
    # declared divergence, and still refuse the forecast-skill claim that
    # would justify a higher maturity rung.
    assert "4 of 12 bit-identical" in evidence
    assert "2-7 ULP" in evidence
    assert "829 ULP" in evidence
    assert "mp_p3_wrapper_wrf" in evidence
    assert "one-ULP nudge" in evidence
    assert "STILL OPEN" in evidence
    assert "no matched WRF forecast run" in evidence
    assert "not evidence of forecast skill" in evidence


def test_the_namelist_importer_takes_mp50_natively():
    """Native import, not substitution: the mapped value must equal 50."""
    from woof.namelist_import import _MP_MAP

    assert _MP_MAP[50][0] == 50
    # The unported siblings must be ABSENT, so no namelist can be silently
    # downgraded onto the one-category solver.
    for unported in ():
        assert unported not in _MP_MAP


# ---------------------------------------------------------------------------
# 3.  The lookup table is a HARD dependency and fails closed.
# ---------------------------------------------------------------------------

def test_the_packaged_table_matches_its_pinned_digest():
    from woof.core.p3_tables import (
        TABLE_1_2MOM_ASSET,
        packaged_p3_table_root,
    )

    path = packaged_p3_table_root() / TABLE_1_2MOM_ASSET.filename
    blob = path.read_bytes()
    assert len(blob) == TABLE_1_2MOM_ASSET.size
    assert hashlib.sha256(blob).hexdigest() == TABLE_1_2MOM_ASSET.sha256


def test_the_table_parses_into_the_fortran_shapes():
    from woof.core.p3_tables import (
        COLLTABSIZE,
        DENSIZE,
        ISIZE,
        RCOLLSIZE,
        RIMSIZE,
        TABSIZE,
        load_lookup_table_1,
        p3_table_root,
    )

    itab, itabcoll = load_lookup_table_1(p3_table_root())
    assert itab.shape == (DENSIZE, RIMSIZE, ISIZE, TABSIZE)
    assert itabcoll.shape == (DENSIZE, RIMSIZE, ISIZE, RCOLLSIZE, COLLTABSIZE)
    assert itab.dtype == np.float32 and itabcoll.dtype == np.float32
    assert np.isfinite(itab).all() and np.isfinite(itabcoll).all()


def test_a_missing_table_is_a_loud_refusal_not_a_fallback(tmp_path):
    from woof.core.p3_tables import load_lookup_table_1

    with pytest.raises(FileNotFoundError) as caught:
        load_lookup_table_1(str(tmp_path))
    # The loader names the selectors that read the table (mp 50/51/52 share
    # the 2momI table since the multi-category ports).
    assert "mp_physics=50/51/52 cannot run" in str(caught.value)
    assert "p3_lookupTable_1.dat-v5.4_2momI" in str(caught.value)


def test_a_byte_modified_table_is_refused(tmp_path):
    """P3's process rates ARE this table; a silent fallback would be wrong.

    One flipped digit is enough: the guard is a SHA-256 over the whole file,
    not a size check, so a same-length edit must still be caught.
    """
    from woof.core.p3_tables import (
        TABLE_1_2MOM_ASSET,
        load_lookup_table_1,
        packaged_p3_table_root,
    )

    src = packaged_p3_table_root() / TABLE_1_2MOM_ASSET.filename
    dst = tmp_path / TABLE_1_2MOM_ASSET.filename
    shutil.copyfile(src, dst)
    blob = bytearray(dst.read_bytes())
    # Flip one digit deep inside the data, keeping the byte count identical.
    for index in range(len(blob) - 1, 0, -1):
        if blob[index : index + 1].isdigit():
            blob[index] = ord("9") if blob[index] != ord("9") else ord("1")
            break
    dst.write_bytes(bytes(blob))
    assert len(bytes(blob)) == TABLE_1_2MOM_ASSET.size  # size guard alone passes

    with pytest.raises(ValueError) as caught:
        load_lookup_table_1(str(tmp_path))
    assert "SHA-256" in str(caught.value)
    assert "Refusing to parse a modified table" in str(caught.value)


def test_the_generated_rain_tables_are_physical():
    """vn/vm are rain fallspeeds [m s-1]; revap is a ventilation integral.

    The authority caps the terminal fallspeed at 9.17 m s-1 for the largest
    bin (module_mp_p3.F:643), so a mass-weighted mean above that is
    impossible and a non-positive one is a dead table.
    """
    from woof.core.p3_tables import generate_rain_tables

    vn, vm, revap = generate_rain_tables()
    for table in (vn, vm, revap):
        assert table.shape == (300, 10)
        assert table.dtype == np.float32
        assert np.isfinite(table).all()
    assert (vn > 0.0).all() and (vm > 0.0).all() and (revap > 0.0).all()
    assert vm.max() <= np.float32(9.17)
    assert vn.max() <= np.float32(9.17)
    # mass-weighted mean fallspeed exceeds the number-weighted one everywhere
    assert (vm >= vn).all()
    # mu_r is the constant 0 (:219), so every column is identical.
    for table in (vn, vm, revap):
        assert (table == table[:, :1]).all()


# ---------------------------------------------------------------------------
# 4.  A real column integration through the solver.
# ---------------------------------------------------------------------------

def _sounding():
    """A deep moist column set: warm rain, rimed ice aloft, and a clear one."""
    from woof.core.p3 import qv_sat

    z = np.arange(NK, dtype=np.float32) * 400.0 + 200.0
    pres = (101300.0 * np.exp(-z / 8500.0)).astype(np.float32)
    temp = np.maximum(300.0 - 0.0075 * z, 200.0).astype(np.float32)
    exner = (pres / 100000.0) ** (287.15 / 1005.0)

    f = dict(
        th=np.tile((temp / exner).astype(np.float32), (NCOL, 1)),
        pres=np.tile(pres, (NCOL, 1)).astype(np.float32),
        dz=np.full((NCOL, NK), 400.0, dtype=np.float32),
    )
    qv = np.empty((NCOL, NK), dtype=np.float32)
    for k in range(NK):
        qv[:, k] = 0.95 * qv_sat(np.float32(temp[k]), np.float32(pres[k]), 0)
    f["qv"] = qv
    for name in ("qc", "qr", "nr", "qi", "qir", "ni", "qib"):
        f[name] = np.zeros((NCOL, NK), dtype=np.float32)
    # column 0: warm cloud over rain.  column 1: rimed ice aloft
    # (rime fraction 0.4, rime density 500).  column 2: clear.
    f["qc"][0, 2:10] = 1.0e-3
    f["qr"][0, 0:6] = 5.0e-4
    f["nr"][0, 0:6] = 1.0e4
    f["qi"][1, 20:30] = 1.0e-3
    f["ni"][1, 20:30] = 1.0e4
    f["qir"][1, 20:30] = 4.0e-4
    f["qib"][1, 20:30] = 4.0e-4 / 500.0
    # Column 2 is dried to 40% RH so it is subsaturated with respect to ICE
    # as well as liquid.  95% RH over liquid is NOT quiescent: below 258 K
    # it is ice-supersaturated by ~12%, which clears P3's Cooper deposition
    # -nucleation threshold (supi >= 0.05, module_mp_p3.F:3264-3266) and
    # grows ice out of nothing.  That is correct physics -- it is pinned by
    # test_an_ice_supersaturated_column_nucleates_ice -- but it makes 95%
    # useless as a "nothing should happen" control.
    f["qv"][2] *= np.float32(0.4 / 0.95)
    f["th_old"] = f["th"].copy()
    f["qv_old"] = f["qv"].copy()
    rho = f["pres"] / (287.15 * (f["th"] * np.tile(exner, (NCOL, 1))))
    f["_mass"] = (rho * f["dz"]).astype(np.float32)   # kg m-2 per layer
    return f


def _integrate(f, *, steps=NSTEPS):
    """Run the shipped WRF wrapper on the slab; returns the diagnostics."""
    from woof.core.p3 import mp_p3_wrapper_wrf, p3_init

    runtime = p3_init()
    surf = {n: np.zeros(NCOL, dtype=np.float32)
            for n in ("rainnc", "rainncv", "sr", "snownc", "snowncv")}
    vol = {n: np.zeros((NCOL, NK), dtype=np.float32)
           for n in ("refl", "re_c", "re_i", "vmi", "di", "rhopo")}
    for it in range(1, steps + 1):
        mp_p3_wrapper_wrf(
            f["th"], f["qv"], f["qc"], f["qr"], f["nr"], f["qi"], f["qir"],
            f["ni"], f["qib"], f["th_old"], f["qv_old"], f["pres"], f["dz"],
            DT, it, surf["rainnc"], surf["rainncv"], surf["sr"],
            surf["snownc"], surf["snowncv"], vol["refl"], vol["re_c"],
            vol["re_i"], vol["vmi"], vol["di"], vol["rhopo"],
            runtime=runtime)
    return surf, vol


def _total_water(f):
    q = f["qv"] + f["qc"] + f["qr"] + f["qi"]
    return (q.astype(np.float64) * f["_mass"]).sum(axis=1)


def _assert_p3_is_physical_and_did_work(f, before, surf, vol):
    """Every invariant this port is entitled to claim, in one place.

    Shared with the mutation control, which requires it to FAIL when the
    solver is stubbed out -- so each assertion below must be one a no-op
    scheme could not satisfy, or must be paired with one that isn't.
    """
    # -- finite, and no negative mass or number anywhere
    for name in ("th", "qv", "qc", "qr", "nr", "qi", "qir", "ni", "qib"):
        assert np.isfinite(f[name]).all(), f"{name} went non-finite"
        assert (f[name] >= 0.0).all(), f"{name} went negative"
    for name, array in vol.items():
        assert np.isfinite(array).all(), f"{name} went non-finite"

    # -- P3's own structural invariants (calc_bulkRhoRime, :6784-6830)
    assert (f["qir"] <= f["qi"] + 1.0e-12).all(), "rime mass exceeds ice mass"
    rimed = f["qib"] > 0.0
    if rimed.any():
        rho_rime = f["qir"][rimed] / f["qib"][rimed]
        assert (rho_rime >= 50.0 - 1e-3).all()
        assert (rho_rime <= 900.0 + 1e-3).all()

    # -- conservation: water lost from the column left through the surface
    after = _total_water(f)
    precip = surf["rainnc"].astype(np.float64)      # mm == kg m-2
    residual = (after - before) + precip
    relative = np.abs(residual) / np.maximum(before, 1e-12)
    assert (relative < 1.0e-4).all(), f"water not conserved: {relative}"

    # -- solid fraction is a fraction
    assert ((surf["sr"] >= 0.0) & (surf["sr"] <= 1.0)).all()

    # -- diagnostics inside physical ranges
    assert vol["refl"].max() < 80.0, "reflectivity above any real storm"
    assert vol["vmi"].min() >= 0.0 and vol["vmi"].max() < 30.0
    cloudy = f["qc"] >= 1.0e-14
    if cloudy.any():
        # effective radii come back in METRES from p3_main (:2279-2281)
        assert (vol["re_c"][cloudy] > 1.0e-6).all()
        assert (vol["re_c"][cloudy] < 1.0e-3).all()

    # -- THE WORK ASSERTIONS.  A stubbed solver satisfies everything above
    #    trivially; these are what make the green meaningful.
    assert precip[0] > 0.0, "the raining column produced no surface precip"
    assert vol["refl"].max() > 0.0, "no column produced positive reflectivity"
    assert vol["vmi"].max() > 0.0, "ice never acquired a fall speed"


def test_a_p3_column_integration_is_physical_and_conserves_water():
    """15 steps x 20 s on three columns through the shipped WRF wrapper."""
    f = _sounding()
    before = _total_water(f)
    surf, vol = _integrate(f)
    _assert_p3_is_physical_and_did_work(f, before, surf, vol)


def test_the_column_assertions_fail_when_the_solver_is_stubbed_out(
        monkeypatch):
    """MUTATION CONTROL for the test above.

    Replace ``p3_main`` with a no-op and require the SAME assertion helper
    to fail.  Without this, "the P3 column test passes" would be a claim
    about the assertions' existence rather than about the scheme.
    """
    import woof.core.p3 as p3

    monkeypatch.setattr(p3, "p3_main", lambda *a, **k: None)
    f = _sounding()
    before = _total_water(f)
    surf, vol = _integrate(f)
    with pytest.raises(AssertionError):
        _assert_p3_is_physical_and_did_work(f, before, surf, vol)


def test_a_dry_column_is_left_alone_while_its_neighbours_rain():
    """Columns never interact inside p3_main; the dry one must not drift.

    Column 2 carries no condensate and is subsaturated with respect to both
    phases, so p3_main's entry test finds neither hydrometeors nor possible
    nucleation and skips it outright (the ``goto 333`` at :2439).  This is
    also the guard against a slab-flattening bug in the adapter's
    (nz, ny, nx) -> (ncol, nz) transpose: a column mix-up shows up here as
    condensate appearing where none was placed.
    """
    f = _sounding()
    dry_qv = f["qv"][2].copy()
    dry_th = f["th"][2].copy()
    _integrate(f, steps=3)
    assert (f["qc"][2] == 0.0).all()
    assert (f["qr"][2] == 0.0).all()
    assert (f["qi"][2] == 0.0).all()
    np.testing.assert_array_equal(f["qv"][2], dry_qv)
    np.testing.assert_array_equal(f["th"][2], dry_th)
    # ...while the neighbour it shares the slab with really did rain
    assert f["qr"][0].max() > 0.0


def test_an_ice_supersaturated_column_nucleates_ice():
    """Cooper (1986) deposition nucleation out of clear, cold, moist air.

    A column at 95% RH over liquid is ~12% supersaturated over ice below
    258 K, which clears the ``supi_cld >= 0.05`` threshold at :3264-3266.
    P3 must then grow ice where there was none -- with the number
    concentration Cooper's exponential prescribes, capped at 100 L-1
    (:3267).  A port that silently skipped the nucleation block would leave
    this column empty and still pass every conservation test in this file,
    which is why the behaviour gets its own pin.
    """
    from woof.core.p3 import qv_sat

    f = _sounding()
    # undo the drying: put column 2 back at the ice-supersaturated 95%
    f["qv"][2] *= np.float32(0.95 / 0.4)
    assert (f["qi"][2] == 0.0).all()
    _integrate(f, steps=5)

    grew = f["qi"][2] > 0.0
    assert grew.any(), "no ice nucleated in an ice-supersaturated column"
    exner = (f["pres"][2] / 100000.0) ** (287.15 / 1005.0)
    temp = f["th"][2] * exner
    # Ice must have appeared in the layers cold enough for Cooper's branch.
    # It is NOT confined to them afterwards -- five steps of sedimentation
    # carry it down -- but nothing may survive above freezing, where
    # p3_main's melting term (:2851-2865) converts it to rain.
    assert grew[temp < 258.15].any()
    assert (temp[grew] < 273.15).all(), "ice survived above the melting point"
    # every ice-bearing level carries a positive number concentration
    assert (f["ni"][2][grew] > 0.0).all()
    # freshly nucleated ice is unrimed: there is no cloud water to rime with
    assert (f["qir"][2] == 0.0).all()
    assert (f["qib"][2] == 0.0).all()
    # the vapour it came from really left the air
    assert float(qv_sat(np.float32(250.0), np.float32(50000.0), 1)) > 0.0


def test_the_first_step_qvs_floor_pins_sup_at_minus_one():
    """The cold-start qvs/qvi floor, DEFAULT-ON (2026-08-31).

    WRF's own first P3 step evaluated 0.0/0.0 for sup and supi: th_old/
    qv_old are plain state that nothing initialises (Registry.EM_COMMON:
    1598-1599), so on call 1 they are zero; the ``max(t_old,1.)`` guard
    at :2329 keeps polysvp1's argument finite but polysvp1(1 K)
    underflows, leaving stock qvs == qvi == 0.  This port used to
    TRANSCRIBE that NaN deliberately, with a containment proof
    (test_the_first_step_nan_in_sup_is_contained_to_comparisons, this
    node's ancestor).  The floor -- ``max(qvs, 1.e-20)`` after both
    qv_sat calls, the remedy later P3 releases carry themselves -- now
    lands in all three arms, default-on: "fixed means default", and
    never bit-exact to a bug (implement defined behaviour where the
    reference computes an undefined 0/0, document the divergence).  It
    pins step-1 sup/supi at exactly -1 (fully subsaturated, the intended
    meaning) and is inert from step 2 on.

    What this pins: (a) no NaN anywhere on step 1, now by construction
    rather than by containment; (b) the floored diagnosis itself, so a
    revert to the stock NaN -- or a floor on one arm only, the rejected
    fork -- fails this node by name.  The declared step-1 delta (the
    trace-condensate clip becomes reachable) is documented at the floor
    site in woof/core/p3.py.
    """
    f = _sounding()
    assert (f["th_old"] != 0.0).any()
    # Force WRF's genuine first-call state: both carriers exactly zero.
    f["th_old"][...] = 0.0
    f["qv_old"][...] = 0.0

    # The floored diagnosis, checked at the leaf before the integration:
    # qvs floors to exactly 1e-20 and sup to exactly -1.
    from woof.core.p3 import f32, qv_sat
    qvs_cold = max(qv_sat(np.float32(1.0), np.float32(85000.0), 0),
                   f32(1.0e-20))
    assert qvs_cold == np.float32(1.0e-20)
    assert np.float32(0.0) / qvs_cold - np.float32(1.0) == np.float32(-1.0)

    surf, vol = _integrate(f, steps=1)

    for name in ("th", "qv", "qc", "qr", "nr", "qi", "qir", "ni", "qib"):
        assert np.isfinite(f[name]).all(), f"first-step NaN leaked into {name}"
    for name, array in vol.items():
        assert np.isfinite(array).all(), f"first-step NaN leaked into {name}"
    for name, array in surf.items():
        assert np.isfinite(array).all(), f"first-step NaN leaked into {name}"
    # and the carriers were written for the next step, ending the floor's
    # authority after exactly one step
    assert (f["qv_old"] > 0.0).any()
    assert (f["th_old"] > 0.0).all()


@pytest.mark.parametrize(
    "kwargs,pattern",
    (
        # mp 52 runs two categories on the CUDA kernels; the CPU reference
        # still refuses them by name.
        ({"n_cat": 2}, "carries one ice category"),
        ({"model": "GEM"}, "only the WRF orientation"),
    ),
)
def test_p3_main_refuses_the_configurations_it_does_not_implement(
        kwargs, pattern):
    """The solver refuses at its own door, not only at config validation.

    config.py is the seam a namelist crosses, but p3_main is callable
    directly by verification tooling; a scope pin that only exists in the
    config layer is one a caller can walk around.
    """
    from woof.core.p3 import p3_main

    slab = lambda: np.zeros((1, 4), np.float32)   # noqa: E731
    surf = lambda: np.zeros(1, np.float32)        # noqa: E731
    with pytest.raises(NotImplementedError, match=pattern):
        p3_main(
            slab(), slab(), slab(), slab(),        # qc, nc, qr, nr
            slab(), slab(), slab(), slab(),        # th_old, th, qv_old, qv
            10.0,                                  # dt
            slab(), slab(), slab(), slab(),        # qitot, qirim, nitot, birim
            slab(), slab(), slab(),                # ssat, pres, dzq
            1,                                     # it
            surf(), surf(),                        # prt_liq, prt_sol
            slab(), slab(), slab(), slab(), slab(), slab(),   # diagnostics
            **kwargs)


# ---------------------------------------------------------------------------
# 5.  The shipped production seam: initialize_physics + microphysics.apply.
# ---------------------------------------------------------------------------

def _p3_domain_state(cp, cfg):
    from woof.core.state import DomainState

    state = DomainState(cfg)
    nz = cfg.nz
    state.p[...] = cp.asarray(
        np.linspace(95000.0, 30000.0, nz, dtype=np.float32)[:, None, None])
    state.thb[...] = cp.asarray(np.full(nz, 300.0, np.float32))
    state.thp[...] = 0.0
    state.phb[...] = cp.asarray(
        np.linspace(0.0, 9.81 * 10000.0, nz + 1, dtype=np.float32))
    state.php[...] = 0.0
    state.qv[...] = 0.006
    state.qc[...] = 2.0e-4
    state.qr[...] = 1.0e-4
    state.nr[...] = 2.0e5
    state.qi[...] = 1.0e-5
    state.ni[...] = 1.0e4
    state.qir[...] = 4.0e-6
    state.qib[...] = 4.0e-6 / 500.0
    return state


@requires_gpu
def test_the_state_allocates_p3s_inventory_and_not_the_six_species_one():
    """P3 has ONE ice category: qs/qg/effs must not exist on an mp=50 state.

    Allocating them would hand advection, output and the nest-transition
    machinery three frozen species P3 never writes -- fields that would read
    as a legitimate zero everywhere instead of as absent.
    """
    import cupy as cp

    state = _p3_domain_state(cp, _cfg())
    for name in ("qi", "qir", "qib", "ni", "nr", "th_old", "qv_old",
                 "effc", "effi"):
        assert getattr(state, name, None) is not None, f"{name} missing"
    for name in ("qs", "qg", "effs"):
        assert getattr(state, name, None) is None, f"{name} must not exist"
    # P3's own background radii, in woof's micron convention (:2279-2281)
    assert float(state.effc.min()) == pytest.approx(10.0)
    assert float(state.effi.min()) == pytest.approx(25.0)


def _background_radius_arms():
    """The scheme tuples of the ``effc/effi/effs`` seeding chain in state.py.

    Read from the source rather than from a constant because the chain IS an
    if/elif ladder: there is no table to import, and the whole point of this
    gate is that a tuple edit there must be visible to a test.
    """
    import ast
    import pathlib

    import woof.core.state as state_module

    tree = ast.parse(
        pathlib.Path(state_module.__file__).read_text(encoding="utf-8"))
    arms = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        seeds_effs = any(
            isinstance(inner, ast.Assign)
            and any(isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "effs"
                    for target in inner.targets)
            for inner in node.body)
        if not seeds_effs:
            continue
        test = node.test
        # The named-scheme arm (``_named_scheme(cfg.mp_physics) is not
        # None``, woof.microphysics_schemes) is a registry lookup, not a
        # literal tuple: its members are exactly the registry's ids.
        if (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.ops[0], ast.IsNot)
                and isinstance(test.left, ast.Call)
                and isinstance(test.left.func, ast.Name)
                and test.left.func.id == "_named_scheme"
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value is None):
            from woof.microphysics_schemes import NAMED_MP_IDS
            arms.append(tuple(NAMED_MP_IDS))
            continue
        if (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and isinstance(test.comparators[0], ast.Constant)
                and isinstance(test.comparators[0].value, int)):
            arms.append((test.comparators[0].value,))
            continue
        if (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.ops[0], ast.In)
                and isinstance(test.comparators[0], ast.Name)
                and test.comparators[0].id == "COLUMN_SCHEMES"):
            from woof.core.column_schemes import COLUMN_SCHEMES
            arms.append(tuple(COLUMN_SCHEMES))
            continue
        if (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.ops[0], ast.In)
                and isinstance(test.comparators[0], ast.BinOp)
                and isinstance(test.comparators[0].op, ast.Add)):
            # A literal tuple extended by a module-level scheme tuple (the
            # table-driven column schemes, woof.core.mp_column_schemes):
            # still a membership row, its members read from the module.
            members = []
            for side in (test.comparators[0].left,
                         test.comparators[0].right):
                if isinstance(side, ast.Tuple):
                    members += [element.value for element in side.elts]
                elif isinstance(side, ast.Name):
                    members += list(getattr(state_module, side.id))
                else:
                    members = None
                    break
            if members is not None:
                arms.append(tuple(members))
                continue
        assert (isinstance(test, ast.Compare)
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.In)
                and isinstance(test.comparators[0], ast.Tuple)), (
            "the background-radius chain stopped being a membership ladder; "
            "re-derive this gate against the new shape")
        arms.append(tuple(element.value
                          for element in test.comparators[0].elts))
    return arms


def test_the_background_radius_rows_are_a_six_species_partition():
    """mp=50 is absent from every background-radius row, and that is the call.

    The three arms in ``DomainState.__init__`` (WSM6/Thompson 2.49/4.99/9.99
    from module_model_constants.F:62-64, Morrison's 2.5/5.0/10.0, NSSL's
    2.51/10.01/25.0) do not decide "which schemes declare effective radii" --
    P3 declares two of them.  They decide which writer's background TRIPLE a
    state that allocated effc AND effi AND effs receives before the first
    microphysics call, and P3 never allocates the third.

    WRF says so three times over: Registry.EM_COMMON:3038 gives mp=50
    ``state:re_cloud,re_ice`` and no ``re_snow`` where all four members of
    the first arm (:3021, :3024, :3031, :3036) register the trio;
    module_physics_init.F names the P3 family in the use_mp_re disjunction
    (:1017), sets has_reqc/has_reqi/has_reqs (:1021-1023), then overrides
    ``has_reqs = 0`` for that family alone (:1027-1033); and the driver's P3
    call binds ``diag_effc_3d=re_cloud`` and ``diag_effi_3d=re_ice`` with no
    snow argument (module_microphysics_driver.F:1596-1597).  The string
    "effs" does not occur in module_mp_p3.F at all.

    So this is a recorded exclusion, not an omission: adding 50 to either
    tuple below is dead code today (P3 leaves the chain 50 lines earlier) and
    an ``AttributeError`` on ``self.effs`` the moment anyone also folds P3
    into the six-species branch -- and a snow radius published for a scheme
    whose snow is a rime mass fraction on one ice category is the
    invented-radius mistake, not a rounding one.

    CPU-only by construction (``array_module=np``), so the decision is pinned
    on every run rather than only where a card is present.
    """
    for arm in _background_radius_arms():
        assert 50 not in arm, (
            f"mp_physics=50 was added to the background-radius row {arm}: "
            "P3 allocates no effs (Registry.EM_COMMON:3038, "
            "module_physics_init.F:1027-1033), so this row would seed a snow "
            "radius the scheme never computes")

    from woof.core.state import DomainState

    rows = {}
    for mp in (6, 8, 16, 28, 9, 10, 18, 50):
        state = DomainState(_cfg(mp_physics=mp), array_module=np)
        rows[mp] = tuple(
            None if getattr(state, name, None) is None
            else float(np.asarray(getattr(state, name)).min())
            for name in ("effc", "effi", "effs"))
        # The species inventory the row keys on, measured not assumed.
        has_snow_species = getattr(state, "qs", None) is not None
        assert has_snow_species is (mp != 50)

    for mp in (6, 8, 16, 28):
        assert rows[mp] == pytest.approx((2.49, 4.99, 9.99), rel=1e-6)
    for mp in (9, 10):
        assert rows[mp] == pytest.approx((2.5, 5.0, 10.0), rel=1e-6)
    assert rows[18] == pytest.approx((2.51, 10.01, 25.0), rel=1e-6)

    # P3: its own pair from module_mp_p3.F:2279/:2281, and no third slot.
    assert rows[50][:2] == pytest.approx((10.0, 25.0), rel=1e-6)
    assert rows[50][2] is None
    p3_state = DomainState(_cfg(), array_module=np)
    assert not hasattr(p3_state, "effs"), (
        "effs is ABSENT on an mp=50 state, not None: an attribute that "
        "exists reads as a legitimate zero to every consumer that walks it")
    # And P3 did not silently land on a six-species row.
    assert rows[50][:2] not in (
        pytest.approx((2.49, 4.99), rel=1e-6),
        pytest.approx((2.5, 5.0), rel=1e-6),
        pytest.approx((2.51, 10.01), rel=1e-6))


def test_p3_takes_kfs_melting_level_closure_not_a_refusal():
    """P3 + Kain-Fritsch resolves, and to the branch WRF's own flow selects.

    KF's feedback cascade is ``IF (warm_rain) / ELSEIF (.NOT. F_QS) /
    ELSEIF (F_QS)`` (module_cu_kfeta.F:2599/:2607/:2622).  P3 leaves
    ``warm_rain`` at mp_init's .false. and declares no qs, so the middle
    branch is the one it takes -- and KF then contributes a rain tendency
    only, asking P3's state for no snow or ice array it does not have.

    Before this, mp=50 + cu_physics=1 raised "KF has no verified
    phase-output contract for mp_physics=50", which put P3 out of reach of
    the DEFAULT cumulus scheme: 82 of the health census's mp=50 rows were
    refused for it.
    """
    from woof.core.kf import KFPhaseMode, kf_phase_mode_for_microphysics
    from woof.core.physics import _cumulus_optional_tendency_components

    assert kf_phase_mode_for_microphysics(50) is KFPhaseMode.NO_SEPARATE_SNOW
    # KF's !F_QS branch zeroes DQIDT and DQSDT, so rqr is the whole set.
    assert _cumulus_optional_tendency_components(
        _cfg(cu_physics=1)) == ("rqr",)


@requires_gpu
def test_p3_runs_under_the_default_cumulus_scheme():
    """The admission above, executed rather than asserted about.

    Kain-Fritsch is the default template's cumulus scheme, so "P3 cannot be
    built with cu_physics=1" would have meant P3 was unreachable in the
    configuration most users would actually assemble.
    """
    import cupy as cp

    from woof.core.microphysics import apply
    from woof.core.physics import (
        DECLARED_CONSTANT_GLW_WM2, initialize_physics)

    cfg = _cfg(cu_physics=1)
    state = _p3_domain_state(cp, cfg)
    # This column runs no longwave scheme under Noah, so it declares its
    # downward longwave rather than letting one be invented: the second
    # remedy the GLW-source refusal names, for exactly this case.  The
    # subject here is the cumulus pairing, not the sky.
    initialize_physics(state, cfg, glw=DECLARED_CONSTANT_GLW_WM2)
    result = apply(state, cfg, float(cfg.dt))
    assert result is not None
    for name in ("qv", "qc", "qr", "qi", "ni", "qir", "qib"):
        field = getattr(state, name)
        assert bool(cp.isfinite(field).all()), f"{name} non-finite"
        assert bool((field >= 0.0).all()), f"{name} negative"


@requires_gpu
def test_microphysics_apply_runs_mp50_end_to_end_on_a_real_domainstate():
    """The production entry point, not the adapter: woof.core.microphysics.

    This is the function ``dycore.step`` calls.  It exercises the ring
    guard, the scratch-slot map, the prep/finish theta bracket and the
    diagnostics contract the PhysicsDriver validates -- none of which the
    column tests above touch.
    """
    import cupy as cp

    from woof.core.microphysics import apply
    from woof.core.physics import (
        DECLARED_CONSTANT_GLW_WM2, initialize_physics)

    cfg = _cfg()
    state = _p3_domain_state(cp, cfg)
    # Declared, for the same reason as the cumulus test above: the subject
    # is the microphysics apply seam, and the column runs no longwave.
    initialize_physics(state, cfg, glw=DECLARED_CONSTANT_GLW_WM2)
    before = {name: getattr(state, name).copy()
              for name in ("qv", "qc", "qr", "qi", "ni", "qir", "qib")}

    result = apply(state, cfg, float(cfg.dt))

    assert result is not None
    for name in ("rainnc", "rainncv", "sr", "snownc", "snowncv"):
        assert getattr(result, name) is not None
    # P3 produces no graupel or hail accumulator at all.
    assert result.graupelnc is None and result.hailnc is None
    for name, prior in before.items():
        current = getattr(state, name)
        assert bool(cp.isfinite(current).all()), f"{name} non-finite"
        assert bool((current >= 0.0).all()), f"{name} negative"
    # something actually happened
    assert not bool(cp.allclose(state.qc, before["qc"]))
    # the solid fraction stayed a fraction
    assert bool(((result.sr >= 0.0) & (result.sr <= 1.0)).all())
    # th_old/qv_old are the NEXT step's carriers and must have been written
    assert bool((state.th_old != 0.0).any())
    assert bool((state.qv_old != 0.0).any())


# ---------------------------------------------------------------------------
# 6.  THE DYCORE.  Sections 1-5 stop at woof.core.microphysics.apply, which
#     is one call inside dycore.step -- so none of them touch the RK time-t
#     copy, the scalar registry, calc_cq or advection.  mp_physics=50 was
#     registered, admitted by validate_run_config, and could not take a
#     single model step: ``extra_moist_species`` handed the generic loops
#     ICE_MASS_SPECIES = ('qi','qs','qg') for a state that deliberately
#     allocates no qs/qg, and dycore.py's time-t copy raised
#     ``AttributeError: 'DomainState' object has no attribute 'qs'`` on the
#     FIRST call.  These tests are the tier that catches that class of gap.
# ---------------------------------------------------------------------------

def test_the_transported_registry_is_p3s_own_inventory(monkeypatch):
    """``extra_moist_species`` on a REAL mp=50 state, not a hand-built one.

    P3 is the first scheme in the tree with ``qi`` and NO ``qs``/``qg``, so
    the generic ICE_MASS_SPECIES filter names two fields the state does not
    have.  Every name the registry returns must exist AND have its ``*0``
    RK time-t copy, because that pairing is exactly what dycore.step
    assumes when it snapshots the scalars at the top of every step.
    """
    import woof.core.state as state_module
    from woof.core.moist import (P3_SPECIES, extra_moist_species,
                                  moist_species)

    monkeypatch.setattr(state_module, "cp", np)
    state = state_module.DomainState(_cfg())

    assert extra_moist_species(state) == P3_SPECIES == (
        "qi", "ni", "nr", "qir", "qib")
    assert moist_species(state) == (
        "qv", "qc", "qr", "qi", "ni", "nr", "qir", "qib")
    for name in moist_species(state):
        assert getattr(state, name, None) is not None, name
        assert getattr(state, name + "0", None) is not None, name + "0"
    for absent in ("qs", "qg"):
        assert absent not in moist_species(state)
        assert getattr(state, absent, None) is None


def test_the_rime_pair_reaches_every_derived_transport_inventory():
    """Four manifests are written by hand against the species registry.

    ``extra_moist_species`` is the source, but the held Smagorinsky
    tendencies, the nest forcing kinds and the two restart classifications
    are separate transcriptions of it.  A rime pair that advected in the
    dycore while sitting still across a nest edge, or vanishing across a
    checkpoint, is the same decoupling bug in a narrower place.
    """
    from woof.core.moist import ABSENT_MASS_SLOT
    from woof.core.preflight import nest_field_kinds, scratch_slot_registry
    from woof.io.restart import (STATE_REBUILT_ATTRS,
                                  STATE_SERIALIZED_ATTRS)

    slots = scratch_slot_registry(_cfg(km_opt=4, diff_6th_opt=2))
    # The absent-mass plane is spelled as a literal on both sides (preflight
    # must import without cupy, and the completeness gate resolves slot
    # names statically), so the two spellings are pinned to each other here.
    assert ABSENT_MASS_SLOT == "moist_absent_mass"
    assert ABSENT_MASS_SLOT in scratch_slot_registry(_cfg(moist_cq=True))
    for name in ("qv", "qc", "qr", "qi", "ni", "nr", "qir", "qib"):
        assert "smag_r" + name in slots, name
    for absent in ("smag_rqs", "smag_rqg", "smag_rns", "smag_rng"):
        assert absent not in slots, absent

    assert nest_field_kinds(_cfg()) == (
        "u", "v", "w", "t", "ph", "mu", "qv", "qc", "qr",
        "qi", "ni", "nr", "qir", "qib")

    for name in ("qir", "qib", "th_old", "qv_old"):
        assert name in STATE_SERIALIZED_ATTRS, name
    for name in ("qir0", "qib0"):
        assert name in STATE_REBUILT_ATTRS, name


def test_the_rime_pair_reaches_its_rk_time_t_copies(monkeypatch):
    """The fifth hand-written transcription of the species registry.

    ``RK_TIME_T_SEED_PAIRS`` is woof's name-list stand-in for something
    WRF does by INDEX: ``dyn_em/solve_em.F`` walks
    ``DO im = PARAM_FIRST_SCALAR, num_3d_s`` (:1915, :2769) and hands
    ``scalar_old(ims,kms,jms,im)`` (:1917, :2783) to the RK update, so
    every member of the active package -- for mp=50 that is
    ``scalar:qni,qnr,qir,qib``, Registry.EM_COMMON:3038, with the rime
    pair declared in the same 4-D array at :555 and :557 -- carries a
    time-t copy with nothing to enumerate.  A name list is equivalent to
    that index range only while it names every member.

    MEASURED before the rime pair was added: a P3 state got twelve of its
    fourteen time-t copies written.  ``qir0``/``qib0`` kept whatever the
    cold-start interpolation left while ``qir``/``qib`` carried the
    relocation transplant, ``seed_rk_time_t_copies`` returned a
    twelve-name list into the birth and relocation receipts, and
    ``woof.verify.cases.nest_relocate.rk_seed_consistency`` -- which
    iterates this same tuple to check WRF ``start_domain``'s
    post-condition -- reported ``pass`` over the twelve fields it could
    see and never looked at the two it could not.

    The completeness half below is stated against the species registry
    rather than as a list of two names, so the next transported field
    added to a one-ice-category state is caught here too.
    """
    import woof.core.state as state_module
    from woof.core.moist import moist_species
    from woof.ingest.nest_init import (RK_TIME_T_SEED_PAIRS,
                                        seed_rk_time_t_copies)
    from woof.verify.cases.nest_relocate import rk_seed_consistency

    monkeypatch.setattr(state_module, "cp", np)
    state = state_module.DomainState(_cfg())

    copied_from = {current for current, _ in RK_TIME_T_SEED_PAIRS}
    for name in moist_species(state):
        assert getattr(state, name + "0", None) is not None, name + "0"
        assert name in copied_from, name

    # Distinct per field, so a row that filled the wrong copy is a
    # mismatch rather than a coincidence.
    for index, name in enumerate(moist_species(state)):
        getattr(state, name)[...] = np.float32(1.0 + index)

    written = seed_rk_time_t_copies(state)
    for name in ("qir", "qib"):
        assert name + "0" in written, name + "0"
        assert np.array_equal(getattr(state, name + "0"),
                              getattr(state, name)), name

    census = rk_seed_consistency(state)
    assert census["pass"]
    for name in ("qir0", "qib0"):
        assert name in census["fields"], name
        assert census["fields"][name]["bit_mismatches"] == 0, name


@requires_gpu
def test_the_rime_pair_advects_exactly_with_the_ice_it_describes():
    """The transport operator, isolated from the microphysics.

    ``qirim/qitot`` selects the lookup table's rime-fraction index and
    ``rho_rime = qirim/birim`` its rime-density index, so a qir that did
    not move with qi would mis-select every ice fall speed and collection
    rate within a few steps.  Seeding qi and qir with the IDENTICAL blob
    and running one scalar stage makes that exact: the two must come out
    bitwise equal to each other AND different from what went in, so this
    fails both if qir is left behind and if it is silently frozen.
    """
    import cupy as cp

    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import advance_scalars_stage, init_moist_balanced

    nx, ny, nz = 16, 8, 12
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=500.0, dy=500.0, ztop=6000.0,
                    dt=10.0, run_seconds=0.0, moist=True, mp_physics=50)
    coord = make_vertical_coord(nz)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.003 * np.asarray(z, float),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base,
                                lambda z: np.full(nz, 1.0e-3))

    blob = np.zeros((nz, ny, nx), dtype=np.float32)
    blob[4:8, 3:6, 6:10] = 1.0e-4
    for name in ("qi", "qir"):
        getattr(state, name)[...] = cp.asarray(blob)
        getattr(state, name + "0")[...] = getattr(state, name)
    state.qib[...] = cp.asarray(blob) / np.float32(500.0)
    state.qib0[...] = state.qib
    before = {name: getattr(state, name).copy()
              for name in ("qi", "qir", "qib")}

    chm = (state.c1h[:, None, None] * state.total_mu()[None]
           + state.c2h[:, None, None])
    ru = cp.zeros((nz, ny, nx + 1), cp.float32)
    ru[...] = 0.5 * (cfg.dx / cfg.dt) * chm[:, :, :1]   # face Courant 0.5
    rv = cp.zeros((nz, ny + 1, nx), cp.float32)
    ww = cp.zeros((nz + 1, ny, nx), cp.float32)

    advance_scalars_stage(state, cfg, ru, rv, ww, cfg.dt, final=True)

    # MOVED: a nonzero advection delta, not a loop that skipped them.
    for name in ("qi", "qir", "qib"):
        delta = float(cp.abs(getattr(state, name) - before[name]).max())
        assert delta > 0.0, f"{name} did not move with the flow"
    # MOVED THE SAME WAY: identical input through the identical operator.
    assert state.qir.tobytes() == state.qi.tobytes()
    assert bool((state.qir <= state.qi + cp.float32(1.0e-12)).all())


def _p3_bubble_setup():
    """A real mp=50 experiment: WK82 moist bubble with an implanted rime blob."""
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.verify.cases import moist_bubble
    from woof.verify.cases.wk82 import wk82_sounding

    cfg = dataclasses.replace(moist_bubble.default_config(), mp_physics=50)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: wk82_sounding(z)[0],
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    rime = np.zeros((cfg.nz, cfg.ny, cfg.nx), dtype=np.float32)
    rime[8:16, :, 40:60] = 2.0e-5
    return cfg, coord, base, rime


def _p3_bubble_state(cfg, coord, base, rime, *, u_background=0.0):
    import cupy as cp

    from woof.core.physics import initialize_physics
    from woof.verify.cases import moist_bubble

    state = moist_bubble.build(cfg, coord, base)
    state.qi[...] = cp.asarray(rime) * np.float32(4.0)
    state.qir[...] = cp.asarray(rime)
    state.qib[...] = cp.asarray(rime) / np.float32(500.0)
    state.ni[...] = cp.asarray(rime) * np.float32(1.0e9)
    if u_background:
        state.u[...] += np.float32(u_background)
    initialize_physics(state, cfg)
    return state


@requires_gpu
def test_mp50_takes_dycore_steps_under_a_background_wind():
    """``woof.core.dycore.run_steps`` on a real mp=50 experiment.

    Before the species-registry fix this raised on the FIRST step, at
    dycore.py's RK time-t copy, for every km_opt and every PBL setting --
    so the registry's ``reachability: component-override`` was false in
    practice.  The two arms differ ONLY in a uniform background x wind,
    which shows the scheme integrates under advection with P3's
    invariants intact.

    It does NOT show that qir/qib are transported, and neither this
    docstring nor the registry evidence string may say it does:
    ``p3_main`` rewrites the rime pair every step, so the two arms
    diverge in qir/qib through the MICROPHYSICS whether or not the
    transport loop carries them.  The transport proof is
    :func:`test_the_rime_pair_advects_exactly_with_the_ice_it_describes`,
    which isolates the operator instead of running the whole scheme.
    """
    import cupy as cp

    from woof.core.dycore import run_steps, stability_report

    cfg, coord, base, rime = _p3_bubble_setup()
    still = _p3_bubble_state(cfg, coord, base, rime)
    moving = _p3_bubble_state(cfg, coord, base, rime, u_background=20.0)
    run_steps(still, cfg, 2)
    run_steps(moving, cfg, 2)

    # It TAKES THE STEPS -- the blocker itself.
    assert still.elapsed_seconds == 2 * cfg.dt
    assert not stability_report(still, cfg)["nan"]
    for name in ("qv", "qc", "qr", "qi", "ni", "nr", "qir", "qib",
                 "th_old", "qv_old"):
        assert bool(cp.isfinite(getattr(still, name)).all()), name
    for name in ("qv", "qc", "qr", "qi", "ni", "nr", "qir", "qib"):
        assert bool((getattr(still, name) >= 0.0).all()), name
    # P3's invariants survive the dycore, not just the column driver.
    assert bool((still.qir <= still.qi + cp.float32(1.0e-12)).all())
    rimed = still.qib > 0.0
    rho_rime = still.qir[rimed] / still.qib[rimed]
    assert 50.0 <= float(rho_rime.min())
    assert float(rho_rime.max()) <= 900.0

    # THE WIND REACHES THE RIME PAIR AT ALL: a live-integration check that
    # the arms are genuinely different runs, NOT a transport proof.  Because
    # p3_main rewrites qir/qib every step, these deltas would be nonzero
    # even under a transport loop that skipped the pair -- which is why the
    # operator-isolation test above, not this one, is what pins transport.
    for name in ("qir", "qib"):
        delta = float(cp.abs(getattr(moving, name)
                             - getattr(still, name)).max())
        assert delta > 0.0, f"{name} is identical in both wind arms"


@requires_gpu
def test_an_mp50_run_writes_and_resumes_a_restart_after_real_steps(tmp_path):
    """A checkpoint taken after real dycore steps restores bit-exactly.

    The interesting names are P3's own: the rime pair (transported
    prognostics) and th_old/qv_old, the cross-step supersaturation
    carriers WRF puts in its own restart stream
    (Registry.EM_COMMON:1598-1599, IO string ``rusd``).  All four are
    nonzero here because the steps above actually ran, so a manifest that
    dropped one fails on content rather than on an all-zero coincidence.
    """
    from woof.core.dycore import run_steps
    from woof.core.p3_tables import TABLE_1_2MOM_ASSET
    from woof.io import restart

    cfg, coord, base, rime = _p3_bubble_setup()
    state = _p3_bubble_state(cfg, coord, base, rime)
    run_steps(state, cfg, 2)

    path = restart.write_restart(tmp_path / "wrfrst.npz", state, cfg)
    header = restart.read_restart_header(path)
    setup = header["physics_setup"]
    assert setup["algorithms"]["microphysics"].startswith(
        "p3-one-category-wrf-v4.6.1")
    # The packaged table's bytes are bound into the checkpoint: P3's
    # process rates ARE that table, so a resume onto other bytes must be
    # refused rather than silently continued.
    assert (setup["microphysics"]["p3"]["assets"][0]["sha256"]
            == TABLE_1_2MOM_ASSET.sha256)
    assert {"state/qir", "state/qib", "state/th_old",
            "state/qv_old"} <= set(header["array_manifest"])

    fresh = _p3_bubble_state(cfg, coord, base, rime)
    restart.restore_restart(path, fresh, cfg)
    for name in ("qi", "ni", "nr", "qir", "qib", "th_old", "qv_old"):
        source = getattr(state, name).get()
        assert source.any(), f"{name} is all zero: the check proves nothing"
        assert getattr(fresh, name).get().tobytes() == source.tobytes(), name


# ---------------------------------------------------------------------------
# 8.  The legacy-RRTMG radiation seam: has_req* is a TRIPLE for P3.
# ---------------------------------------------------------------------------
#
# WHY THIS SECTION EXISTS.  ``woof/config.py``'s ``validate_p3_radiation``
# refuses mp=50 against RTE+RRTMGP and names
# ``ra_rrtmg_variant='rrtmg_legacy'`` as the remedy, so the legacy adapter
# is the only 4/4 radiation an mp=50 run can reach and the shipped P3
# configs take exactly that route.  The adapter used to ask WRF's use_mp_re
# scheme table ONE boolean question and apply the answer to all three of
# effc/effi/effs.  P3 is the one selector whose answer is not one boolean:
# WRF sets has_reqc = has_reqi = 1 in the disjunction
# (module_physics_init.F:1004-1023) and then zeroes has_reqs again in a
# second statement (:1026-1033).  With no row at all the adapter raised
# NotImplementedError at the FIRST radiation call; with a bare ``50: True``
# it would instead have demanded a ``state.effs`` P3 never allocates.


def test_p3_declares_cloud_and_ice_radii_to_legacy_rrtmg_but_no_snow():
    """The triple, and that P3 is the only selector that needs a triple."""
    from woof.config import P3_MP_PHYSICS
    from woof.core.rrtmg_legacy import (
        _MP_DECLARES_RADII, legacy_scheme_has_req)

    # P3 predicts diag_effc/diag_effi (module_mp_p3.F:2279-2281, bound to
    # re_cloud/re_ice at module_microphysics_driver.F:1597-1598) and
    # nothing snow-shaped: Registry/Registry.EM_COMMON:3038 gives mp=50
    # moist:qv,qc,qr,qi and state:re_cloud,re_ice -- ONE ice category with
    # a rime mass/volume pair, no qs, no qg, no re_snow.
    # (has_reqc, has_reqi, has_reqs)
    assert legacy_scheme_has_req(50, 1) == (1, 1, 0)
    # use_mp_re=0 is WRF's outer gate and still zeroes all three.
    assert legacy_scheme_has_req(50, 0) == (0, 0, 0)

    # Every OTHER selector woof can drive answers with one value three
    # times -- which is why one boolean served until P3 arrived, and why
    # "add 50: True" is not the fix.
    for mp_physics in sorted(_MP_DECLARES_RADII):
        flags = set(legacy_scheme_has_req(mp_physics, 1))
        # JENSEN_ISHMAEL shares the override block by name
        # (module_physics_init.F:1029), so it shares the triple.
        if mp_physics in P3_MP_PHYSICS or mp_physics == 55:
            assert flags == {0, 1}
        else:
            assert len(flags) == 1, (
                f"mp_physics={mp_physics} now needs a per-radius has_req*; "
                "give it its own WRF citation rather than sharing P3's")


def test_p3s_zero_snow_flag_is_what_moves_its_ice_into_the_snow_optics():
    """The 0 on effs is a switch, not an omission.

    WRF does not drop P3's ice from the radiation when has_reqs=0 -- it
    moves the ice radius AND the ice mass onto the snow optics slot, zeroes
    the ice slot and pins reice to 10 um
    (module_ra_rrtmg_lw.F:12250-12260, module_ra_rrtmg_sw.F:10851-10861).
    The prep already implements that case; this asserts the flags the gate
    hands it actually select it, so the two halves cannot drift apart.
    """
    from woof.core import rrtmg_legacy_prep as prep
    from woof.core.rrtmg_legacy import legacy_scheme_has_req

    kte = 4
    has_reqc, has_reqi, has_reqs = legacy_scheme_has_req(50, 1)
    # P3's radiation-facing column, in the prep's METER convention.
    re_cloud = np.full(kte, 12.0e-6, np.float32)
    re_ice = np.full(kte, 40.0e-6, np.float32)
    qi = np.full(kte, 1.0e-4, np.float32)
    cldfra = np.ones(kte, np.float32)
    t3d = np.full(kte, 250.0, np.float32)
    zeros = np.zeros(kte, np.float32)

    inflg, iceflg, recloud, reice, resnow, qs1d, qi1d = prep._effective_radii(
        kte, 1, has_reqc, has_reqi, has_reqs, t3d, cldfra, 1.0,
        re_cloud, re_ice, zeros, qi, zeros.copy(), qi.copy())

    # iceflg 5 is the snow-bearing ice optics WRF selects for P3.
    assert (inflg, iceflg) == (5, 5)
    # The ice radius is on the SNOW slot; the ice slot is the constant.
    assert np.allclose(resnow, 40.0)
    assert np.allclose(reice, 10.0)
    assert np.allclose(recloud, 12.0)
    # ...and so is the ice MASS, which is what actually makes a P3 ice
    # cloud radiate rather than vanish.
    assert np.array_equal(qs1d, qi)
    assert not qi1d.any()

    # CONTROL: the same column under a scheme that DOES declare snow keeps
    # its ice on the ice slot, so the assertions above are P3's remap and
    # not something every flag combination produces.
    thompson = legacy_scheme_has_req(8, 1)
    assert thompson == (1, 1, 1)
    _, _, _, reice_t, resnow_t, qs_t, qi_t = prep._effective_radii(
        kte, 1, *thompson, t3d,
        cldfra, 1.0, re_cloud, re_ice, zeros, qi, zeros.copy(), qi.copy())
    assert np.allclose(reice_t, 40.0)
    assert np.allclose(resnow_t, 10.0)
    assert not qs_t.any()
    assert np.array_equal(qi_t, qi)


def test_the_legacy_adapter_never_asks_p3_for_a_snow_radius_it_lacks():
    """A bare ``50: True`` would raise on state.effs; the triple does not.

    The mp=50 state allocates effc and effi and NO effs (its Registry
    package has no re_snow), and the adapter refuses to radiate fallback
    radii silently when a DECLARED radius is missing.  So the only reason
    the run survives is that has_reqs is 0 -- asserted here against the
    shipped allocator's own inventory rather than against the flag alone.
    """
    from woof.core.preflight import state_array_shapes
    from woof.core.rrtmg_legacy import legacy_scheme_has_req

    cfg = _cfg()
    validate_run_config(cfg)
    allocated = set(state_array_shapes(cfg))
    has = legacy_scheme_has_req(cfg.mp_physics, 1)
    for name, flag in zip(("effc", "effi", "effs"), has):
        assert (name in allocated) == bool(flag), (
            f"state.{name} allocated={name in allocated} but has_req says "
            f"{flag}: the legacy adapter would either raise on a missing "
            "declared radius or silently drop a predicted one")

    # And the reason there is no effs, stated where the next reader looks:
    # P3 carries one ice category plus its rime pair and no snow species.
    assert {"qi", "qir", "qib"} <= allocated
    assert not ({"qs", "qg"} & allocated)
