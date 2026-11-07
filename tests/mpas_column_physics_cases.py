"""Synthetic column cases for the generic MPAS column-physics seam.

Shared by the GPU tests and by ``python -m`` style sweeps: one moist land
column profile, surface inputs shaped for whatever land surface a plan
names, and orographic statistics for whatever statics a plan names.  The
selections are built from the registry's own menu
(:func:`woof.core.mpas_column_suite.physics_selector_menu`), one option
changed at a time from the frozen default, with WRF's own pairing partners
added where the engine's validator asks for one.
"""

import datetime
import json
import sys
import traceback

import numpy as np

from woof.core import mpas_column_suite as suite

NZ, NCOL = 30, 8
DT = 60.0
ZTOP = 20000.0
START = datetime.datetime(2026, 10, 3, 18, 0)
P_TOP = float(101325.0 * np.exp(-ZTOP / 8000.0))
#: The ICAO standard atmosphere's pressure at ZTOP (the seam's nominal
#: eta, woof.core.mpas_column_suite._nominal_znu), floored to whole Pa.
P_TOP_STANDARD = float(np.floor(
    101325.0 * (216.65 / 288.15) ** 5.25588
    * np.exp(-(ZTOP - 11000.0) / 6341.62)))

#: The partner settings WRF's own pairing rules require beside one option
#: (each is the remedy the engine's refusal text names).  Test data: a
#: caller reads the refusal and supplies the partner the same way.
PARTNERS = {
    "pbl.myj": {"sf_sfclay_physics": 2},
    "pbl.sase": {"km_opt": 0},
    "pbl.off": {"sf_sfclay_physics": 1},
    "surface_layer.eta-similarity": {"bl_pbl_physics": 2},
    "surface_layer.mynn": {"bl_pbl_physics": 5},
    "surface_layer.off": {"bl_pbl_physics": 0, "sf_surface_physics": 0},
    "cumulus.grell-freitas": {"ishallow": 1},
    # The pairings the ported schemes' own admission names (woof.config):
    # GFS, QNSE and TEMF each run only with their own surface layer and the
    # reverse; CAM deep convection reads MYJ's TKE through get_tpert; SSiB
    # runs with fractional sea ice on its three-layer column.
    "pbl.gfs": {"sf_sfclay_physics": 3},
    "surface_layer.gfs": {"bl_pbl_physics": 3},
    "pbl.qnse": {"sf_sfclay_physics": 4},
    "surface_layer.qnse": {"bl_pbl_physics": 4},
    "pbl.temf": {"sf_sfclay_physics": 10},
    "surface_layer.temf": {"bl_pbl_physics": 10},
    "cumulus.cam-deep": {"bl_pbl_physics": 2, "sf_sfclay_physics": 2},
    "land_surface.ssib": {"fractional_seaice": 1, "num_soil_layers": 3},
    # The PX surface layer runs with the ACM2 PBL (WRF pairs them).
    "surface_layer.pleim-xiu": {"bl_pbl_physics": 7},
    # The idealized SCM surface scheme admits no PBL: WRF 4.7.1 assigns
    # IDEALSCMSFCSCHEME no isfc before its PBL compatibility checks
    # (woof.config's refusal names bl_pbl_physics=0 as the remedy), and
    # its prescribed-flux surface admits no land surface model (nothing
    # else would produce UST/CHS/CHS2/CQS2: sf_surface_physics=0).
    "surface_layer.ideal-scm": {"bl_pbl_physics": 0,
                                "sf_surface_physics": 0},
}


def _one_hot(index, count, ncol=NCOL):
    """Category fractions with every column wholly in ``index`` (1-based)."""
    out = np.zeros((count, ncol), np.float32)
    out[index - 1] = 1.0
    return out

#: ``surface`` keywords an option needs declared, again the engine's own
#: named remedy: with no longwave scheme nothing computes GLW (and with no
#: radiation at all nothing computes SWDOWN either), and
#: ``initialize_physics`` takes a declared constant sky rather than invent
#: one.
SURFACE_PARTNERS = {
    "radiation.dudhia-shortwave": {"glw": 350.0},
    "radiation.gsfc-shortwave": {"glw": 350.0},
    "radiation.off": {"glw": 350.0, "swdown": 400.0},
    # The slab reads THC (LANDUSE.TBL THERIN/100); the refusal names
    # thermal_inertia= as the remedy (0.04 as the other slab fixtures).
    "land_surface.slab": {"thermal_inertia": 0.04},
    # PX land weights the land-use and bottom-soil category FRACTIONS
    # (VEGELAND, SOILPROP); these match the dominant categories the
    # surface() columns declare: MODIS 21-category land use wholly
    # category 10 (ivgtyp) and the 16 STATSGO soils wholly category 6
    # (isltyp).
    "land_surface.pleim-xiu": {"landusef": _one_hot(10, 21),
                               "soilcbot": _one_hot(6, 16)},
}


def _held_suarez_binding(ncol=NCOL):
    """A bound RK-stage edge-momentum binding over the test columns.

    ra_lw_physics=31 damps edge momentum at every RK stage, a caller
    obligation the seam refuses to start without (HeldSuarezMeshMomentum).
    The columns are joined in a ring (edge i links cells i and i+1), with
    the profile's own base pressure.
    """
    import cupy as cp
    from woof.core.held_suarez_mesh import HeldSuarezMeshMomentum
    z_mid = 0.5 * (np.linspace(0.0, ZTOP, NZ + 1)[:-1]
                   + np.linspace(0.0, ZTOP, NZ + 1)[1:])
    cells = np.stack([np.arange(ncol), (np.arange(ncol) + 1) % ncol],
                     axis=1).astype(np.int32)
    pressure = np.repeat((101325.0 * np.exp(-z_mid / 8000.0))[:, None],
                         ncol, 1).astype(np.float32)
    binding = HeldSuarezMeshMomentum()
    binding.bind(cp.ascontiguousarray(cp.asarray(cells)),
                 cp.ascontiguousarray(cp.asarray(pressure)))
    return binding


#: Construction partners a selection's own refusal names (factories, so
#: each seam gets its own).
CONSTRUCTION_PARTNERS = {
    "radiation.relaxation-31": {"rk_momentum": _held_suarez_binding},
    # GFDL-Eta reads the seam's nominal eta pair, which needs p_top_pa at
    # or below the standard-atmosphere pressure of the nominal top (the
    # seam's refusal); P_TOP's 8 km scale height puts it above that.
    "radiation.wrf-gfdleta": {"p_top_pa": lambda ncol: P_TOP_STANDARD},
}

#: Selectors the engine accepts that are not registry components.
EXTRA_SELECTIONS = {
    "drag.gwdo": {"gwd_opt": 1},
    "drag.gsl": {"gwd_opt": 3},
    "topo_wind.jimenez": {"topo_wind": 1},
    "topo_wind.uw": {"topo_wind": 2},
    "radiation.rrtmg-legacy": {"ra_physics": 4,
                               "ra_rrtmg_variant": "rrtmg_legacy"},
}


def selections(components=None) -> dict:
    """``{"component.option": physics}`` over the registry menu."""
    base = dict(suite.MPAS_DEFAULT_PHYSICS)
    out = {}
    for component, row in suite.physics_selector_menu().items():
        for option_id, option in row["options"].items():
            physics = dict(base)
            if component == "radiation":
                physics.pop("ra_physics")
                physics.pop("ra_rrtmg_variant")
            physics.update(option["selectors"])
            physics.update(PARTNERS.get(f"{component}.{option_id}", {}))
            out[f"{component}.{option_id}"] = physics
    for name, extra in EXTRA_SELECTIONS.items():
        out[name] = dict(base, **extra)
    if components:
        out = {name: physics for name, physics in out.items()
               if name.split(".")[0] in components or name in components}
    return out


def profile(xp, plan, ncol=NCOL, seed=0) -> dict:
    """Phase inputs for ``plan.species`` as ``[level, column]`` arrays."""
    rng = np.random.default_rng(seed)
    z_iface = np.linspace(0.0, ZTOP, NZ + 1)
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    p_iface = 101325.0 * np.exp(-z_iface / 8000.0)
    p_mid = 101325.0 * np.exp(-z_mid / 8000.0)
    theta = 300.0 + 30.0 * z_mid / ZTOP
    exner = (p_mid / 1.0e5) ** (287.0 / 1004.0)
    rho_dry = p_mid / (287.0 * theta * exner)

    def cols(values, jitter=1.0e-3):
        base = np.repeat(np.asarray(values, np.float64)[:, None], ncol, 1)
        base *= 1.0 + jitter * rng.standard_normal(base.shape)
        return np.ascontiguousarray(base, dtype=np.float32)

    fields = {
        "u": cols(4.0 + 20.0 * z_mid / ZTOP, 2e-2),
        "v": cols(-2.0 + 6.0 * z_mid / ZTOP, 2e-2),
        "theta": cols(theta),
        "pressure": cols(p_mid, 0.0),
        "pressure_interface": cols(p_iface, 0.0),
        "z_interface": cols(z_iface, 0.0),
        "w": cols(0.02 * np.sin(np.linspace(0, np.pi, NZ + 1)), 0.0),
        "rho_dry": cols(rho_dry, 0.0),
    }
    species = {
        "qv": cols(0.012 * np.exp(-z_mid / 2500.0)),
        "qc": cols(np.where(z_mid < 3000.0, 2.0e-4, 0.0)),
        "qr": cols(np.full(NZ, 1.0e-6)),
    }
    for name in plan.species:
        if name in species:
            continue
        species[name] = np.zeros((NZ, ncol), np.float32)
    # Aerosol number tracers a scheme requires its caller to bring.
    for name, value in (("nwfa", 3.0e8), ("nifa", 1.0e6)):
        if name in species:
            species[name] = cols(value * np.exp(-z_mid / 4000.0))
    return ({name: xp.asarray(value) for name, value in fields.items()},
            {name: xp.asarray(species[name]) for name in plan.species})


def surface(plan, ncol=NCOL) -> dict:
    """``initialize_physics`` keywords for the plan's land surface."""
    n_soil = plan.n_soil_levels
    out = {
        "landmask": np.ones(ncol, np.float32),
        "xland": np.ones(ncol, np.float32),
        "ivgtyp": np.full(ncol, 10, np.int32),
        "isltyp": np.full(ncol, 6, np.int32),
        "vegfra": np.full(ncol, 40.0, np.float32),
        "tsk": np.full(ncol, 298.0, np.float32),
        "tmn": np.full(ncol, 287.0, np.float32),
        "xice": np.zeros(ncol, np.float32),
        "snow": np.zeros(ncol, np.float32),
        "snow_depth": np.zeros(ncol, np.float32),
    }
    if n_soil:
        depth = np.linspace(0.0, 1.0, n_soil, dtype=np.float32)
        out["soil_temperature"] = np.repeat(
            (295.0 - 8.0 * depth)[:, None], ncol, 1).astype(np.float32)
        out["soil_moisture"] = np.repeat(
            (0.22 + 0.08 * depth)[:, None], ncol, 1).astype(np.float32)
    return out


def statics(plan, ncol=NCOL, seed=1) -> dict:
    """Orographic statistics for exactly ``plan.static_fields``."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in plan.static_fields:
        if name.startswith("VAR"):
            value = rng.uniform(50.0, 400.0, ncol)
        elif name.startswith("CON"):
            value = rng.uniform(0.0, 5.0, ncol)
        elif name.startswith("OA"):
            value = rng.uniform(-0.5, 0.5, ncol)
        else:
            value = rng.uniform(0.0, 0.8, ncol)
        out[name] = value.astype(np.float32)
    return out


def build_named(name, **kwargs):
    """The seam of one ``selections()`` row, with its declared partners."""
    ncol = kwargs.get("ncol", NCOL)
    construct = {key: factory(ncol) for key, factory
                 in CONSTRUCTION_PARTNERS.get(name, {}).items()}
    return build(selections()[name],
                 extra_surface=SURFACE_PARTNERS.get(name), **construct,
                 **kwargs)


def build(physics, ncol=NCOL, radiation_seconds=120.0, extra_surface=None,
          **construct):
    plan = suite.column_physics_plan(
        physics, n_levels=NZ, dt=DT, radiation_seconds=radiation_seconds,
        surface_pbl_seconds=DT)
    surf = surface(plan, ncol)
    surf.update(extra_surface or {})
    p_top = construct.pop("p_top_pa", P_TOP)
    rng = np.random.default_rng(2)
    seam = suite.run_mpas_column_physics(
        physics=physics, n_levels=NZ, n_columns=ncol, dt=DT,
        radiation_seconds=radiation_seconds, surface_pbl_seconds=DT,
        start_time=START,
        latitude_deg=np.full(ncol, 38.0), longitude_deg=np.full(ncol, -100.0),
        terrain_height_m=rng.uniform(0.0, 50.0, ncol).astype(np.float32),
        z_interface_nominal_m=np.linspace(0.0, ZTOP, NZ + 1),
        p_top_pa=p_top, dx_m=12000.0,
        cell_area_m2=np.full(ncol, 1.44e8),
        surface=surf, statics=statics(plan, ncol), **construct)
    return seam


def run(xp, seam, steps, ncol=NCOL):
    """Step the seam; return per-step host frames of rates and state."""
    fields, species = profile(xp, seam.plan, ncol)
    theta = fields["theta"].copy()
    frames = []
    for _ in range(steps):
        result = seam.run_phase1(
            dt=DT, u=fields["u"], v=fields["v"], theta=theta,
            pressure=fields["pressure"],
            pressure_interface=fields["pressure_interface"],
            z_interface=fields["z_interface"], w=fields["w"],
            rho_dry=fields["rho_dry"], species=species)
        frame = {name: xp.asnumpy(getattr(result, name)).copy()
                 for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqr",
                              "dqi", "dqs", "dqg", "h_diabatic")}
        frame["scalar_rates"] = {name: xp.asnumpy(value).copy()
                                 for name, value in result.scalar_rates.items()}
        if result.dw is not None and not bool(xp.all(xp.isfinite(result.dw))):
            raise FloatingPointError("the phase-one w rate is non-finite")
        frame["dw"] = result.dw is not None
        receipt = seam.run_phase2(
            theta=theta, pressure=fields["pressure"],
            z_interface=fields["z_interface"], w=fields["w"],
            rho_dry=fields["rho_dry"], species=species)
        frame["theta"] = xp.asnumpy(theta).copy()
        frame["species"] = {name: xp.asnumpy(value).copy()
                            for name, value in species.items()}
        frame["receipt"] = {name: xp.asnumpy(value).copy()
                            for name, value in receipt.items()}
        frames.append(frame)
    return frames


def _finite(frames) -> list:
    bad = []
    for step, frame in enumerate(frames):
        for name, value in frame.items():
            if isinstance(value, bool):
                continue
            items = value.items() if isinstance(value, dict) else (
                (name, value),)
            for key, array in items:
                if not np.all(np.isfinite(array)):
                    bad.append(f"step {step} {name}:{key}")
    return bad


def sweep(components=None, steps=3) -> list:
    """Run every selection; one result row each (never raises)."""
    import cupy as cp
    rows = []
    for name, physics in selections(components).items():
        row = {"option": name, "physics": physics}
        try:
            seam = build_named(name)
            frames = run(cp, seam, steps)
            bad = _finite(frames)
            row.update(
                status="ran" if not bad else "non-finite", bad=bad[:6],
                species=list(seam.species),
                calls=seam.call_counts,
                max_abs_dtheta=float(max(np.abs(f["dtheta"]).max()
                                         for f in frames)),
                max_abs_du=float(max(np.abs(f["du"]).max() for f in frames)),
                scalar_rates=sorted(frames[-1]["scalar_rates"]),
                w_rate=frames[-1]["dw"],
                precip=sorted(seam.accumulated_precipitation()))
            payload = seam.export_state()
            row["restart_arrays"] = len(payload["arrays"])
        except Exception as error:  # noqa: BLE001 -- the row IS the report
            row.update(status="error", error=f"{type(error).__name__}: "
                       f"{str(error)[:700]}",
                       where=traceback.format_exc(limit=-3)[-900:])
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)
        try:
            del seam, frames
        except NameError:
            pass
        cp.get_default_memory_pool().free_all_blocks()
    return rows


if __name__ == "__main__":
    sweep(sys.argv[1:] or None)
