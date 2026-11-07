"""Device behaviour of the generic MPAS column-physics seam.

Two things need arrays.  First, the DEFAULT selection through the generic
entry must be the frozen seam bit for bit (and the slot-table seam's HRRR
family likewise), so moving a caller onto the generic entry changes no
number.  Second, every option the registry marks implemented must actually
run through ``PhysicsDriver.compute`` and ``microphysics.apply`` on the
column state -- the sweep that turns "arbitrary" from a claim into rows.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof.core import mpas_column_batch as mcb  # noqa: E402
from woof.core import mpas_column_suite as suite  # noqa: E402

import mpas_column_physics_cases as cases  # noqa: E402

NZ, NCOL, DT = cases.NZ, cases.NCOL, cases.DT
RATES = mcb._OUTPUT_BUFFERS


def _frozen_frames(steps, *, microphysics, cumulus, plan):
    fields, species = cases.profile(cp, plan)
    surface = cases.surface(plan)
    rng = np.random.default_rng(2)
    seam = mcb.run_mpas_column_batch(
        n_levels=NZ, n_columns=NCOL, dt=DT, radiation_seconds=120.0,
        surface_pbl_seconds=DT,
        cumulus_seconds=None if cumulus is None else DT,
        cumulus_scheme=cumulus, microphysics_scheme=microphysics,
        start_time=cases.START,
        latitude_deg=np.full(NCOL, 38.0), longitude_deg=np.full(NCOL, -100.0),
        terrain_height_m=rng.uniform(0.0, 50.0, NCOL).astype(np.float32),
        z_interface_nominal_m=np.linspace(0.0, cases.ZTOP, NZ + 1),
        p_top_pa=cases.P_TOP, dx_m=12000.0, xice_threshold=0.5, **surface)
    theta = fields["theta"].copy()
    frames = []
    for _ in range(steps):
        result = seam.run_phase1(
            dt=DT, u=fields["u"], v=fields["v"], theta=theta,
            pressure=fields["pressure"],
            pressure_interface=fields["pressure_interface"],
            z_interface=fields["z_interface"], w=fields["w"],
            rho_dry=fields["rho_dry"], **species)
        frame = {name: cp.asnumpy(getattr(result, name)).copy()
                 for name in RATES}
        rho = {"rho_dry": fields["rho_dry"]} if microphysics == "wsm6" else {}
        seam.run_phase2(theta=theta, pressure=fields["pressure"],
                        z_interface=fields["z_interface"], **rho, **species)
        frame["theta"] = cp.asnumpy(theta).copy()
        frame["species"] = {name: cp.asnumpy(value).copy()
                            for name, value in species.items()}
        frames.append(frame)
    return frames, seam.accumulated_precipitation()


@pytest.mark.parametrize("microphysics,mp_physics,cumulus,cu", [
    ("wsm6", 6, None, {}),
    ("wsm6", 6, "gf", {"cu_physics": 3, "ishallow": 1}),
    ("wsm6", 6, "kf", {"cu_physics": 1}),
    ("thompson_aero", 28, "gf", {"cu_physics": 3, "ishallow": 1}),
    ("p3", 50, None, {}),
])
def test_the_frozen_selection_is_bit_identical_through_the_generic_entry(
        microphysics, mp_physics, cumulus, cu):
    physics = dict(suite.MPAS_DEFAULT_PHYSICS, mp_physics=mp_physics, **cu)
    plan = suite.column_physics_plan(physics)
    steps = 6
    frozen, frozen_rain = _frozen_frames(
        steps, microphysics=microphysics, cumulus=cumulus, plan=plan)
    seam = suite.run_mpas_column_physics(
        physics=physics, n_levels=NZ, n_columns=NCOL, dt=DT,
        radiation_seconds=120.0, surface_pbl_seconds=DT,
        cumulus_seconds=DT if cumulus == "kf" else None,
        start_time=cases.START,
        latitude_deg=np.full(NCOL, 38.0), longitude_deg=np.full(NCOL, -100.0),
        terrain_height_m=np.random.default_rng(2).uniform(
            0.0, 50.0, NCOL).astype(np.float32),
        z_interface_nominal_m=np.linspace(0.0, cases.ZTOP, NZ + 1),
        p_top_pa=cases.P_TOP, dx_m=12000.0,
        surface=dict(cases.surface(plan), xice_threshold=0.5))
    generic = cases.run(cp, seam, steps)
    for step in range(steps):
        for name in RATES + ("theta",):
            np.testing.assert_array_equal(
                generic[step][name], frozen[step][name], f"step {step} {name}")
        for name in plan.species:
            np.testing.assert_array_equal(
                generic[step]["species"][name], frozen[step]["species"][name],
                f"step {step} {name}")
    rain = seam.accumulated_precipitation()
    assert set(rain) == set(frozen_rain)
    for name in rain:
        np.testing.assert_array_equal(
            cp.asnumpy(rain[name]), cp.asnumpy(frozen_rain[name]), name)


def test_the_slot_table_family_is_bit_identical_through_the_generic_entry():
    """MYNN + MYNN surface layer + RUC + GSL drag + mp=28, both entries."""
    physics = {"mp_physics": 28, "ra_physics": 4,
               "ra_rrtmg_variant": "rrtmg_legacy", "sf_sfclay_physics": 5,
               "sf_surface_physics": 3, "bl_pbl_physics": 5, "gwd_opt": 3,
               "fractional_seaice": 1}
    plan = suite.column_physics_plan(physics)
    assert plan.n_soil_levels == 9 and len(plan.static_fields) == 20
    seam = cases.build(physics)
    generic = cases.run(cp, seam, 4)
    surface = cases.surface(plan)
    fields, species = cases.profile(cp, plan)
    slot = suite.run_mpas_column_suite(
        n_levels=NZ, n_columns=NCOL, dt=DT, radiation_seconds=120.0,
        surface_pbl_seconds=DT, microphysics_scheme="thompson_aero",
        radiation_scheme="rrtmg_legacy", surface_layer_scheme="mynn",
        land_surface_scheme="ruc", pbl_scheme="mynn", drag_scheme="gsl",
        start_time=cases.START,
        latitude_deg=np.full(NCOL, 38.0), longitude_deg=np.full(NCOL, -100.0),
        terrain_height_m=np.random.default_rng(2).uniform(
            0.0, 50.0, NCOL).astype(np.float32),
        z_interface_nominal_m=np.linspace(0.0, cases.ZTOP, NZ + 1),
        p_top_pa=cases.P_TOP, dx_m=12000.0, fractional_seaice=1,
        orographic_statistics=cases.statics(plan), **surface)
    theta = fields["theta"].copy()
    for step in range(4):
        result = slot.run_phase1(
            dt=DT, u=fields["u"], v=fields["v"], theta=theta,
            pressure=fields["pressure"],
            pressure_interface=fields["pressure_interface"],
            z_interface=fields["z_interface"], w=fields["w"],
            rho_dry=fields["rho_dry"], **species)
        for name in RATES:
            np.testing.assert_array_equal(
                cp.asnumpy(getattr(result, name)), generic[step][name],
                f"step {step} {name}")
        slot.run_phase2(theta=theta, pressure=fields["pressure"],
                        z_interface=fields["z_interface"], **species)
        np.testing.assert_array_equal(
            cp.asnumpy(theta), generic[step]["theta"], f"step {step} theta")
        for name in plan.species:
            np.testing.assert_array_equal(
                cp.asnumpy(species[name]), generic[step]["species"][name],
                f"step {step} {name}")


def test_restart_round_trip_continues_bit_identically():
    physics = dict(suite.MPAS_DEFAULT_PHYSICS, mp_physics=10, cu_physics=16)
    reference = cases.run(cp, cases.build(physics), 4)
    first = cases.build(physics)
    cases.run(cp, first, 2)
    payload = first.export_state()
    second = cases.build(physics)
    second.restore_state(payload)
    fields, species = cases.profile(cp, second.plan)
    theta = cp.asarray(reference[1]["theta"])
    species = {name: cp.asarray(reference[1]["species"][name])
               for name in second.plan.species}
    for step in (2, 3):
        result = second.run_phase1(
            dt=DT, u=fields["u"], v=fields["v"], theta=theta,
            pressure=fields["pressure"],
            pressure_interface=fields["pressure_interface"],
            z_interface=fields["z_interface"], w=fields["w"],
            rho_dry=fields["rho_dry"], species=species)
        for name in RATES:
            np.testing.assert_array_equal(
                cp.asnumpy(getattr(result, name)), reference[step][name],
                f"step {step} {name}")
        second.run_phase2(
            theta=theta, pressure=fields["pressure"],
            z_interface=fields["z_interface"], w=fields["w"],
            rho_dry=fields["rho_dry"], species=species)
    with pytest.raises(ValueError, match="identity"):
        cases.build(dict(physics, mp_physics=8)).restore_state(payload)


def test_species_slack_refuses_both_ways():
    seam = cases.build(dict(suite.MPAS_DEFAULT_PHYSICS, mp_physics=8))
    fields, species = cases.profile(cp, seam.plan)
    call = dict(dt=DT, u=fields["u"], v=fields["v"], theta=fields["theta"],
                pressure=fields["pressure"],
                pressure_interface=fields["pressure_interface"],
                z_interface=fields["z_interface"], w=fields["w"],
                rho_dry=fields["rho_dry"])
    short = {k: v for k, v in species.items() if k != "ni"}
    with pytest.raises(ValueError, match="missing \\['ni'\\]"):
        seam.run_phase1(species=short, **call)
    with pytest.raises(ValueError, match="not its state \\['nc'\\]"):
        seam.run_phase1(species=dict(species, nc=species["ni"]), **call)


#: Options that do not run on the column state yet, each with the breakage
#: the sweep recorded.  A row leaves this table when its option runs; the
#: test below fails if a listed option starts passing, so the table cannot
#: go stale.
NOT_YET = {
    # NAMED GAP (2.9.1): the CCPP sa-TKE-EDMF PBL refuses a positive bldt (its
    # TKE tendency is not in the extra-scalar restart stream), and this seam
    # runs a positive bldt to hold the raw PBL momentum rates it returns, so the
    # option cannot run on the MPAS column state yet.  Owner: the satmedmf
    # port (serialize the held TKE tendency, then remove this row).
    "pbl.ccpp-tke-edmf": "CCPP TKE-EDMF requires bldt=0",
    # NAMED GAP RCON-SPEED-01 (3.0.0): woof.config refuses mp_physics=29
    # for every forecast, not only this seam, until a qualified speed
    # handoff clears the measured 1.885x-MP28 microphysics stage.  Owner:
    # the RCON speed lane; this row leaves with that guard.
    "microphysics.rcon-mp29": "RCON-SPEED-01 speed refusal",
}


@pytest.mark.parametrize("name", sorted(
    name for name in cases.selections()
    if name.split(".")[0] != "turbulence"))
def test_every_registry_option_runs_on_the_column_state(name):
    if name in NOT_YET:
        with pytest.raises(Exception):  # noqa: B017 -- recorded breakage
            cases.run(cp, cases.build_named(name), 3)
        return
    seam = cases.build_named(name)
    frames = cases.run(cp, seam, 3)
    assert not cases._finite(frames)
    assert seam.call_counts["radiation"] >= (
        1 if seam.plan.config.ra_physics or seam.plan.config.ra_sw_physics > 0
        else 0)
    assert set(seam.accumulated_precipitation()) == set(
        seam.plan.surface_accumulators) | {"RAINC"}
