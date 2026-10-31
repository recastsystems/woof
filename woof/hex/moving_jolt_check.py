"""The moving-region jolt check on a reduced mesh: ``woof hex moving check``.

A self-contained proof that a move of the fine region does not jolt the
model, runnable on any machine with this package and the woof engine (on a
CUDA card, or on the CPU without ``--lts``).  Nothing is downloaded: every
mesh is generated here by ``rw_mpas_mesh``.

THE FIXTURE (reproduces the reduced GPU test of the jolt fix):

* a 522-cell spherical Voronoi mesh whose centres are contracted toward a cap
  centre by a smooth angular map (about a 1.4:1 spacing contrast), rebuilt
  through ``rw_mpas_mesh --from-centres`` with the cap at 0, 10, 20 and 30
  degrees east (15 degrees north): four members of a moving family;
* eight layers to 30 km, an isothermal 288 K resting atmosphere, a 1e-5
  relative theta-mass perturbation, one tracer at 1 ppb (see
  ``TRACER_KG_PER_KG``), and the same analytic
  0-100 m terrain evaluated on each member;
* 80 steps of 30 s (3 dynamics splits, 6 acoustic substeps), or with ``--lts``
  the production full-step LTS executor (RK2, 15/30 s classes, scalar reflux)
  with the class boundary shifted at every move;
* moves at 600, 1200 and 1800 s through the production remap
  (:func:`woof.hex.moving_state.remap_atmosphere`, physical-height mapping).

THE STEEP CASE (``steep``, run after the gentle one above; both must pass):
the same family and moves over a 2,700 m plateau north of 20N between 0 and
40E whose escarpment is narrower than a cell, with the production 55 levels
to 30 km (thin low layers), and a horizontally uniform atmosphere at rest:
6.5 K/km to the tropopause and a 16 K inversion between 2,590 and 2,990 m,
so the plateau donors' lowest layers sit in the inversion while the valley
columns under it are a cold pool. THE BREAKAGE it sees: the absolute-height
remap extrapolated a plateau donor's lowest-layer gradients and interface
values 1-2 km down into a valley target column (2025-11-08 12Z smoke case,
240/60 km, move at 1 h: surface-pressure tendency 6.06x the fixed run, peak
over steep terrain at 30.3N 104.1E). The 0-100 m gentle case cannot see
that: measured on the card, the extrapolating remap jolts this case 1.705
against 1.019 for the fix.

THE ARMS: ``fixed`` (no move, the same start state), ``baseline`` (the remap
without the balanced initialization, ``WOOF_HEX_MOVING_BALANCE=0``) and
``balanced`` (the default).  The jolt after each move is the larger of the
two ratios max|w| and max|dpsfc/dt| over the next ten minutes, each against
the fixed arm over the same minutes.  The check passes when every balanced
move is at most ``--limit`` (1.5, the moving acceptance limit) and every
balanced move conserves dry mass, theta mass and tracer mass to ``--budget``
(1e-9 relative).  The baseline arm is the measured defect the balance fixes;
it is reported, never graded.

No divergence damping, terrain smoothing or time filter is enabled.  This is a
dry numerical test (the 1 ppb tracer is weightless to within the calm background), not a cyclone: it says the remap hands the dynamics a
state in discrete force balance; whole-forecast acceptance is a separate run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import multiprocessing
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import numpy as np

SCHEMA = "gpuwm-hex.moving-jolt-check/v1"
EARTH_RADIUS_M = 6371229.0
N_LEVELS = 8
# The steep case (see the module note): production level count, plateau height and the inversion over it.
STEEP_LEVELS = 55
STEEP_PLATEAU_M = 2700.0
STEEP_INVERSION_K = 16.0
STEEP_INVERSION_DEPTH_M = 400.0
CASES = ("gentle", "steep")
# The tracer's mixing ratio.  With physics off MPAS-A counts every carried scalar as water vapour, so a
# tracer has weight in the vertical momentum equation (-cqw g rb qtot) and nothing in the resting state's
# perturbations balances it: at 1e-3 the column free-falls at g q dt = 0.29 m/s per 30 s step, on native
# MPAS-A and on hex's MPAS-exact GPU path alike (measured 2026-10-14: identical words on the same state).
# That made the GPU fixed arm noisy and hid the defect this check exists for (old remap 1.01 against 16.2).
# At 1 ppb the weight is about 3e-7 m/s per step, far under the calm background, while tracer mass
# conservation is still measured (it is relative).  The CPU dry driver ignores the weight either way.
TRACER_KG_PER_KG = 1e-9
STEPS = 80
MOVE_STEPS = (20, 40, 60)
WINDOW_STEPS = 20
ARMS = ("fixed", "baseline", "balanced")
KW = dict(rgas=287.0, cp=1004.5, reference_pressure=1e5)


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------
def _sphere_xyz(lat, lon):
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def build_family(folder: Path, *, members: int = 4, log=print) -> dict[str, Any]:
    """The seed mesh and ``members`` contracted-cap meshes, with each move's overlap."""

    from netCDF4 import Dataset

    from . import engines
    from .moving_remap import ConservativeMap, SphericalCells

    tool = engines.resolve(engines.MESH)
    folder.mkdir(parents=True, exist_ok=True)
    seed = folder / "refined-seed.nc"
    run = dict(stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    subprocess.run([str(tool), "--background-km", "1500", "--out", str(seed), "--cells", "512",
                    "--fit-spacing", "yes", "--sweeps", "5", "--tolerance", ".1", "--clobber"], **run)
    meshes, overlaps, receipts = [], [], []
    for i in range(members):
        centres = folder / f"refined-centres{i}.nc"
        shutil.copyfile(seed, centres)
        with Dataset(centres, "r+") as ds:
            xyz = np.stack([ds[n + "Cell"][:] for n in "xyz"], axis=-1)
            radius = np.linalg.norm(xyz, axis=1)
            unit = xyz / radius[:, None]
            centre = _sphere_xyz(np.deg2rad(15.0), np.deg2rad(10.0 * i))
            cosine = np.clip(unit @ centre, -1.0, 1.0)
            angle = np.arccos(cosine)
            tangent = unit - cosine[:, None] * centre
            tangent /= np.linalg.norm(tangent, axis=1)[:, None]
            mapped = angle * (1.0 - 0.3 * np.exp(-0.5 * (angle / 0.3) ** 2))
            new = np.cos(mapped)[:, None] * centre + np.sin(mapped)[:, None] * tangent
            for j, n in enumerate("xyz"):
                ds[n + "Cell"][:] = new[:, j] * radius
        out = folder / f"refined-{i}.nc"
        receipt = folder / f"refined-mesh{i}.json"
        subprocess.run([str(tool), "--from-centres", str(centres), "--out", str(out), "--clobber",
                        "--receipt", str(receipt)], **run)
        meshes.append(out)
        receipts.append(dict(mesh=out.name, sha256=_sha256(out)))
        if i:
            w = ConservativeMap.build(SphericalCells.read(meshes[i - 1]), SphericalCells.read(out))
            overlaps.append(w)
        log(f"check: member {i} built ({out.name})")
    return dict(tool=str(tool), meshes=meshes, overlaps=overlaps, receipts=receipts)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def levels_for(case: str) -> int:
    if case not in CASES:
        raise ValueError(f"unknown jolt check case {case!r}; the cases are {', '.join(CASES)}")
    return STEEP_LEVELS if case == "steep" else N_LEVELS


def setup(path: Path, levels: int = N_LEVELS):
    """Mesh on the earth sphere in FP32, the dry v8.4.1 configuration and coefficients."""

    from .config_v841 import V841DryDycoreConfig
    from .dynamics_v841 import V841ReferenceWindProfiles
    from .mesh import Mesh
    from .rust_geometry import derived_geometry
    from .transport import build_advection_coefficients, initialize_deriv_two

    mesh = Mesh.from_netcdf(Path(path))
    scale = EARTH_RADIUS_M / float(mesh.attrs["sphere_radius"])
    for entity in ("Cell", "Edge", "Vertex"):
        for axis in ("x", "y", "z"):
            mesh.arrays[axis + entity] *= scale
    for name in ("dcEdge", "dvEdge", "nominalMinDc"):
        mesh.arrays[name] *= scale
    for name in ("areaCell", "areaTriangle", "kiteAreasOnVertex"):
        mesh.arrays[name] *= scale * scale
    mesh.attrs["sphere_radius"] = EARTH_RADIUS_M
    mesh.arrays.update(derived_geometry(mesh))
    for name, a in list(mesh.arrays.items()):
        if a.dtype.kind == "f":
            mesh.arrays[name] = np.ascontiguousarray(a, dtype=np.float32)
    mesh.arrays["nominalMinDc"] = np.asarray(mesh.arrays["nominalMinDc"].reshape(-1)[0], dtype=np.float32)
    for name in ("fEdge", "fVertex"):
        mesh.arrays[name] = (2 * 7.29212e-5 * np.sin(mesh.arrays["lat" + name[1:]])).astype(np.float32)
    for name in ("defc_a", "defc_b"):
        mesh.arrays[name] = np.zeros(mesh.arrays["edgesOnCell"].shape, np.float32)
    mesh.arrays["deriv_two"] = initialize_deriv_two(mesh)
    cfg = V841DryDycoreConfig(
        config_dt=30.0, config_split_dynamics_transport=True, config_dynamics_split_steps=3,
        config_number_of_sub_steps=6, config_smdiv=0.0, config_divergence_damping=False,
        config_terrain_following=True)
    coeff = build_advection_coefficients(mesh, config_scalar_adv_order=3, n_vert_levels=levels)
    winds = V841ReferenceWindProfiles(np.zeros(levels, np.float32), np.zeros(levels, np.float32))
    return mesh, cfg, coeff, winds


def terrain_for(mesh, case: str = "gentle") -> np.ndarray:
    levels_for(case)
    lat, lon = mesh.arrays["latCell"], mesh.arrays["lonCell"]
    if case == "steep":
        lat, lon = np.degrees(np.asarray(lat, np.float64)), np.degrees(np.asarray(lon, np.float64))
        north = 0.5 * (1 + np.tanh((lat - 20.0) / 1.5))
        band = 0.5 * (1 + np.tanh(lon / 3.0)) * 0.5 * (1 + np.tanh((40.0 - lon) / 3.0))
        return STEEP_PLATEAU_M * north * band
    return 100.0 * (np.sin(lat) * np.cos(lon)) ** 2


def _steep_pressure_temperature(z):
    """The steep case's uniform resting atmosphere at heights ``z``: hydrostatic pressure and temperature."""

    def temperature(height):
        free = np.maximum(288.0 - 6.5e-3 * height, 216.65)
        base = 0.96 * STEEP_PLATEAU_M
        return free - STEEP_INVERSION_K * (1.0 - np.clip((height - base) / STEEP_INVERSION_DEPTH_M, 0.0, 1.0))

    fine = np.linspace(0.0, 30000.0, 15001)
    rate = -9.80616 / (KW["rgas"] * temperature(fine))
    log_p = np.log(KW["reference_pressure"]) + np.concatenate(
        ([0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(fine))))
    return np.exp(np.interp(z, fine, log_p)), temperature(z)


def fixture(mesh, terrain, amp=0.0, case: str = "gentle"):
    """A resting atmosphere on ``mesh`` over ``terrain`` (isothermal, or the steep case's), with its diagnostics."""

    from .config_v841 import V841DryDycoreConfig
    from .driver import DryDycoreDriver, DryReferenceState, TerrainMetrics, _pressure_exner
    from .dynamics_v841 import V841ReferenceWindProfiles
    from .state import PrognosticState
    from .terrain import build_terrain_coupling
    from .vertical import build_edge_vertical_metrics, build_vertical_grid

    nc = len(mesh.arrays["areaCell"])
    ne = len(mesh.arrays["dcEdge"])
    nl = levels_for(case)
    v = build_vertical_grid(mesh, terrain.astype(np.float32), n_vert_levels=nl, ztop=30000.0,
                            terrain_smoothing_passes=0, smooth_surfaces=False, xnutr=0)
    v = replace(v, **{name: np.ascontiguousarray(getattr(v, name)) for name in v.__slots__
                      if isinstance(getattr(v, name), np.ndarray)})
    for name in ("dzu", "rdzu", "rdzwp", "rdzwm", "fzm", "fzp"):
        getattr(v, name)[0] = 0.0
    z = 0.5 * (v.zgrid[:-1].astype(float) + v.zgrid[1:])
    if case == "steep":
        p, temperature = _steep_pressure_temperature(z)
    else:
        p, temperature = 1e5 * np.exp(-9.80616 * z / (287.0 * 288)), np.full(z.shape, 288.0)
    r = (p / (287.0 * temperature) / v.zz).astype(np.float32)
    h = (p / (287.0 * temperature) * temperature / (p / 1e5) ** (287.0 / 1004.5) / v.zz).astype(np.float32)
    s = PrognosticState(r, h, np.zeros((nl, ne), np.float32), np.zeros((nl + 1, nc), np.float32),
                        np.full((1, nl, nc), TRACER_KG_PER_KG, np.float32), 0.0)
    p, ex, theta = _pressure_exner(s, v, **KW)
    ref = DryReferenceState(r.copy(), h.copy(), p.copy(), ex.copy())
    pattern = np.cos(mesh.arrays["latCell"]) * np.sin(2 * mesh.arrays["lonCell"])
    s.rho_theta[:] *= (1.0 + amp * pattern[None, :] * np.exp(-z / 6000.0)).astype(np.float32)
    metric = build_edge_vertical_metrics(mesh, v, theta_adv_order=3)
    t = build_terrain_coupling(n_edges_on_cell=mesh.arrays["nEdgesOnCell"], edges_on_cell=mesh.arrays["edgesOnCell"],
                               cells_on_edge=mesh.arrays["cellsOnEdge"], zb=metric.zb, zb3=metric.zb3,
                               config_coef_3rd_order=0.25)
    terrain_metrics = TerrainMetrics(np.ascontiguousarray(t.zb_cell, dtype=np.float32),
                                     np.ascontiguousarray(t.zb3_cell, dtype=np.float32))
    prep = NS(mesh=mesh, vertical=v, reference=ref, terrain_metrics=terrain_metrics)
    driver = DryDycoreDriver(mesh, v, ref, V841DryDycoreConfig(config_terrain_following=True),
                             terrain_metrics=terrain_metrics,
                             reference_wind_profiles=V841ReferenceWindProfiles(np.zeros(nl, np.float32),
                                                                               np.zeros(nl, np.float32)))
    saved = driver._rebuild_saved_diagnostics(s)
    return s, saved, prep


def payload_of(s, saved, prep):
    state = {name: getattr(s, name) for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars")}
    state.update({name: getattr(saved, name) for name in saved.__slots__})
    return dict(state=state, vertical=prep.vertical, mesh=prep.mesh, time_seconds=s.time_seconds,
                scalar_names=["passive"])


def surface_pressure(s, prep):
    """The native physics-preparation surface pressure (dry: qv = 0 in this fixture)."""

    from .driver import _pressure_exner

    pressure, _, _ = _pressure_exner(s, prep.vertical, **KW)
    dz = np.diff(prep.vertical.zgrid, axis=0)
    r = s.rho * prep.vertical.zz
    correction = np.float32(0.5) * (r[1] - r[0]) * dz[0] / (dz[0] + dz[1])
    return pressure[0] + np.float32(0.5) * np.float32(9.80616) * dz[0] * (r[0] - correction)


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------
def _lts_driver(s, saved, p, cfg, coeff, winds, shift, cards=1):
    from .cuda_driver import CudaDryDycoreDriver
    from .cuda_fullstep_lts_active import FullStepLtsActive
    from .fullstep_lts_classes import build_class_plan
    from .partition_local_mesh_v841 import slice_prepared_host

    mesh = p.mesh
    a = mesh.arrays
    decl = dict(mesh.variable_dimensions or {})
    for name in ("defc_a", "defc_b"):
        decl[name] = ("nCells", "maxEdges")
    mesh.variable_dimensions = decl
    base_dt = 15.0
    phase = shift * (s.time_seconds // 600)
    rates = np.where(np.sin(a["lonCell"] + phase) > 0, 1, 2).astype(np.int32)
    plan = build_class_plan(rates=rates, cells_on_cell=a["cellsOnCell"], n_edges_on_cell=a["nEdgesOnCell"],
                            cells_on_edge=a["cellsOnEdge"], cells_on_vertex=a["cellsOnVertex"],
                            area_cell=a["areaCell"].astype(float), dc_edge_m=a["dcEdge"].astype(float),
                            base_dt=base_dt, edges_on_edge=a["edgesOnEdge"], n_edges_on_edge=a["nEdgesOnEdge"])
    complete = NS(mesh=mesh, state=s, saved_diagnostics=saved, vertical=p.vertical, reference=p.reference,
                  terrain_metrics=p.terrain_metrics, advection_coefficients=coeff)
    if cards > 1:
        return _lts_multi_driver(plan, complete, cfg, winds, base_dt, rates, cards)
    drivers, areas = [], []
    for table in plan.classes:
        local = slice_prepared_host(complete, table.layout)
        config = replace(cfg, config_dt=base_dt * table.rate, config_time_integration_order=2)
        drivers.append(CudaDryDycoreDriver.from_host(
            local.mesh, local.state, local.vertical, local.reference, config,
            saved_diagnostics=local.saved_diagnostics, terrain_metrics=local.terrain_metrics,
            advection_coefficients=local.advection_coefficients, reference_wind_profiles=winds))
        areas.append(a["areaCell"][table.layout.cell_l2g])
    active = FullStepLtsActive(plan, drivers, base_dt=base_dt, kernel_cache=drivers[0].cache, area_cell=areas,
                               dzw=p.vertical.dzw)
    active.fixture_rates = rates
    return active


def _lts_multi_driver(plan, complete, cfg, winds, base_dt, rates, cards):
    """The production multi-card class blocks (``FullStepLtsMulti``): every class cut across ``cards`` cards."""

    import cupy as cp

    from .cuda_driver import CudaDryDycoreDriver
    from .cuda_fullstep_lts_multi import FullStepLtsMulti
    from .fullstep_lts_blocks import build_block_plan
    from .partition_local_mesh_v841 import slice_prepared_host

    a = complete.mesh.arrays
    devices = tuple(range(int(cards)))
    xyz = np.stack((np.cos(a["latCell"]) * np.cos(a["lonCell"]), np.cos(a["latCell"]) * np.sin(a["lonCell"]),
                    np.sin(a["latCell"])), axis=1).astype(np.float64)
    bplan = build_block_plan(plan, devices=devices, n_blocks=(len(devices),) * len(plan.classes),
                             cells_on_cell=a["cellsOnCell"], n_edges_on_cell=a["nEdgesOnCell"],
                             cells_on_edge=a["cellsOnEdge"], cells_on_vertex=a["cellsOnVertex"], cell_xyz=xyz,
                             area_cell=a["areaCell"].astype(float), edges_on_edge=a["edgesOnEdge"],
                             n_edges_on_edge=a["nEdgesOnEdge"])
    drivers, areas, caches = [], [], {}
    for tables in bplan.blocks:
        local = slice_prepared_host(complete, tables.layout)
        config = replace(cfg, config_dt=base_dt * tables.rate, config_time_integration_order=2)
        with cp.cuda.Device(int(tables.device)):
            driver = CudaDryDycoreDriver.from_host(
                local.mesh, local.state, local.vertical, local.reference, config,
                saved_diagnostics=local.saved_diagnostics, terrain_metrics=local.terrain_metrics,
                advection_coefficients=local.advection_coefficients, reference_wind_profiles=winds)
            cp.cuda.get_current_stream().synchronize()
        caches.setdefault(int(tables.device), driver.cache)
        drivers.append(driver)
        areas.append(a["areaCell"][tables.layout.cell_l2g])
    active = FullStepLtsMulti(bplan, drivers, base_dt=base_dt, kernel_caches=caches, area_cell=areas,
                              dzw=complete.vertical.dzw)
    active.fixture_rates = rates
    return active


def _advance_lts(d):
    import cupy as cp

    from .driver import DrySavedDiagnostics
    from .state import PrognosticState

    d.advance_macro(2)
    if not d.registers_empty():
        raise RuntimeError("LTS transfer registers are not empty at a complete macro boundary")
    names = ("rho", "rho_theta", "rho_u", "rho_w") + DrySavedDiagnostics.__slots__
    fields = d.gather(names)
    parts = getattr(d, "blocks", None) or d.classes
    first = parts[0].driver.atmosphere.state.scalars
    scalar = np.empty((first.shape[0], d.nlev, len(d.fixture_rates)), np.float32)
    for part in parts:
        layout = part.tables.layout
        n = part.n_owned_cells
        with cp.cuda.Device(int(getattr(part, "device", cp.cuda.Device().id))):
            scalar[:, :, layout.cell_l2g[:n]] = cp.asnumpy(part.driver.atmosphere.state.scalars[:, :, :n])
    s = PrognosticState(*(fields[k] for k in ("rho", "rho_theta", "rho_u", "rho_w")), scalar, d.time_seconds)
    saved = DrySavedDiagnostics(**{k: fields[k] for k in DrySavedDiagnostics.__slots__})
    return s, saved


class _Integrator:
    def __init__(self, *, device: str, lts: bool, cfg, coeff, winds, rate_shift: float, cards: int = 1):
        self.device, self.lts, self.cfg, self.coeff, self.winds = device, lts, cfg, coeff, winds
        self.rate_shift = rate_shift
        self.cards = int(cards)

    def driver(self, s, saved, p):
        from .driver import DryDycoreDriver, DrySavedDiagnostics
        from .state import PrognosticState

        coeff = getattr(p, "coeff", self.coeff)
        if self.lts:
            return _lts_driver(s, saved, p, self.cfg, coeff, self.winds, getattr(p, "rate_shift", 0.0),
                               cards=self.cards)
        kw = dict(terrain_metrics=p.terrain_metrics, reference_wind_profiles=self.winds,
                  advection_coefficients=coeff)
        if self.device == "gpu":
            from .cuda_driver import CudaDryDycoreDriver

            s = PrognosticState(*(np.ascontiguousarray(getattr(s, k))
                                  for k in ("rho", "rho_theta", "rho_u", "rho_w", "scalars")), s.time_seconds)
            saved = DrySavedDiagnostics(**{k: np.ascontiguousarray(getattr(saved, k)) for k in saved.__slots__})
            return CudaDryDycoreDriver.from_host(p.mesh, s, p.vertical, p.reference, self.cfg,
                                                 saved_diagnostics=saved, **kw)
        return DryDycoreDriver(p.mesh, p.vertical, p.reference, self.cfg, **kw)

    def advance(self, d, s, saved):
        from .driver import DrySavedDiagnostics
        from .state import PrognosticState

        if self.lts:
            return _advance_lts(d)
        if self.device != "gpu":
            result = d.step(s, saved_diagnostics=saved)
            return result.state, result.saved_diagnostics
        import cupy as cp

        result = d.step_device()
        d.atmosphere = result.atmosphere
        st, sa = result.atmosphere.state, result.atmosphere.saved
        s = PrognosticState(*(cp.asnumpy(getattr(st, k)) for k in ("rho", "rho_theta", "rho_u", "rho_w", "scalars")),
                            float(st.time_seconds))
        saved = DrySavedDiagnostics(**{k: cp.asnumpy(getattr(sa, k)) for k in DrySavedDiagnostics.__slots__})
        return s, saved


def _close(d) -> None:
    """The multi-card blocks hold one worker thread per card; a replaced integrator releases them."""

    close = getattr(d, "close", None)
    if close is not None:
        close()


def _run_arm(arm: str, family: dict[str, Any], integ: _Integrator, setups: list, log,
             case: str = "gentle") -> dict[str, Any]:
    from .moving_remap import ConservativeMap
    from .moving_state import remap_atmosphere
    from .moving_vertical import HeightConservativeMap

    previous_knob = os.environ.get("WOOF_HEX_MOVING_BALANCE")
    os.environ["WOOF_HEX_MOVING_BALANCE"] = "1" if arm == "balanced" else "0"
    try:
        mesh, cfg, coeff, winds = setups[0]
        s, saved, p = fixture(mesh, terrain_for(mesh, case), amp=1e-5, case=case)
        d = integ.driver(s, saved, p)
        previous = surface_pressure(s, p)
        rows, budgets, rate_changes = [], [], []
        wall = time.monotonic()
        for step in range(STEPS):
            if arm != "fixed" and step in MOVE_STEPS:
                old_rates = getattr(d, "fixture_rates", None)
                old = p
                index = step // WINDOW_STEPS
                tm, _, tc, _ = setups[index]
                _, _, p = fixture(tm, terrain_for(tm, case), case=case)
                p.coeff = tc
                w0 = family["overlaps"][index - 1]
                scale = EARTH_RADIUS_M ** 2
                horizontal = ConservativeMap(w0.overlap * scale, w0.source_area * scale, w0.target_area * scale, {})
                horizontal = horizontal.with_physical_areas(old.mesh.arrays["areaCell"], tm.arrays["areaCell"])
                w = HeightConservativeMap(horizontal, old.vertical.zgrid, p.vertical.zgrid, old.vertical.dzw,
                                          p.vertical.dzw, endpoint_alignment=False)
                began = time.monotonic()
                s, saved, budget = remap_atmosphere(payload_of(s, saved, old), w, source_mesh=old.mesh,
                                                    target_prepared=p, config=cfg)
                remap_seconds = time.monotonic() - began
                jumped = surface_pressure(s, p)
                budgets.append(dict(time=step * cfg.config_dt, remap_seconds=remap_seconds,
                                    surface_pressure_remap_jump_pa=float(abs(
                                        jumped - w.horizontal.intensive(previous)).max()),
                                    mass=budget["mass_total_kg"], theta_mass=budget["theta_mass_total_kg_kelvin"],
                                    tracers=budget["tracer_totals_kg"], balance=budget.get("balance")))
                p.rate_shift = integ.rate_shift
                _close(d)
                d = integ.driver(s, saved, p)
                # The post-move metric starts from the remapped state; the
                # instantaneous cross-mesh jump is recorded separately above.
                previous = surface_pressure(s, p)
                if old_rates is not None:
                    rate_changes.append(dict(time=step * cfg.config_dt,
                                             cells_changed=int(np.sum(old_rates != d.fixture_rates))))
            s, saved = integ.advance(d, s, saved)
            current = surface_pressure(s, p)
            rows.append(dict(time=float(s.time_seconds), w=float(abs(saved.vertical_velocity).max()),
                             dpdt=float(abs(current - previous).max() / cfg.config_dt)))
            previous = current
        _close(d)
        out = dict(rows=rows, budgets=budgets, rate_changes=rate_changes, wall_seconds=time.monotonic() - wall,
                   final_hashes={k: hashlib.sha256(np.ascontiguousarray(getattr(s, k)).tobytes()).hexdigest()
                                 for k in ("rho", "rho_theta", "rho_u", "rho_w", "scalars")},
                   saved_hashes={k: hashlib.sha256(np.ascontiguousarray(getattr(saved, k)).tobytes()).hexdigest()
                                 for k in saved.__slots__})
        log(f"check: {case} arm {arm} done in {out['wall_seconds']:.1f} s")
        return out
    finally:
        if previous_knob is None:
            os.environ.pop("WOOF_HEX_MOVING_BALANCE", None)
        else:
            os.environ["WOOF_HEX_MOVING_BALANCE"] = previous_knob


def _relative(row: dict[str, Any]) -> float:
    before = float(row["before"])
    return abs(float(row["absolute_error"]) / before) if before else abs(float(row["absolute_error"]))


def evaluate(results: dict[str, Any], *, limit: float, budget: float) -> dict[str, Any]:
    metrics = []
    for step in MOVE_STEPS:
        row: dict[str, Any] = dict(move_seconds=step * 30.0)
        for arm, result in results.items():
            window = result["rows"][step:step + WINDOW_STEPS]
            row[arm] = {key: max(x[key] for x in window) for key in ("w", "dpdt")}
        for arm in results:
            if arm == "fixed":
                continue
            ratios = {key: row[arm][key] / row["fixed"][key] for key in ("w", "dpdt")}
            row[arm]["ratios"] = ratios
            row[arm]["jolt"] = max(ratios.values())
        metrics.append(row)
    verdict: dict[str, Any] = dict(limit=limit, budget=budget)
    for arm in results:
        if arm == "fixed":
            continue
        verdict[f"{arm}_worst_jolt"] = max(m[arm]["jolt"] for m in metrics)
        worst = dict(mass=0.0, theta_mass=0.0, tracer=0.0)
        for b in results[arm]["budgets"]:
            worst["mass"] = max(worst["mass"], _relative(b["mass"]))
            worst["theta_mass"] = max(worst["theta_mass"], _relative(b["theta_mass"]))
            for tracer in b["tracers"].values():
                worst["tracer"] = max(worst["tracer"], _relative(tracer))
        verdict[f"{arm}_worst_conservation"] = worst
    balanced_ok = "balanced" in results and verdict["balanced_worst_jolt"] <= limit and all(
        value <= budget for value in verdict["balanced_worst_conservation"].values())
    verdict["passed"] = bool(balanced_ok)
    return dict(metrics=metrics, verdict=verdict)


def _cpu_arm(task):
    """One isolated arm; its balance environment and NetCDF handles are private."""
    arm, case, family = task
    from .stage_timing import stage

    stages, messages = [], []
    with stage(stages, f"{case}/{arm}/host-setup", parallelism={"device": "CPU", "worker_pid": os.getpid()}):
        setups = [setup(path, levels=levels_for(case)) for path in family["meshes"]]
    integ = _Integrator(device="cpu", lts=False, cfg=setups[0][1], coeff=setups[0][2], winds=setups[0][3],
                        rate_shift=0.0)
    with stage(stages, f"{case}/{arm}", parallelism={"device": "CPU", "worker_pid": os.getpid(),
               "arithmetic": "vectorized elements, source-order neighbor sums"}):
        result = _run_arm(arm, family, integ, setups, messages.append, case=case)
    return result, stages, messages


def _cpu_workers(arms):
    from .cpu_budget import available_cpus

    budget = available_cpus()
    for name in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS"):
        value = os.environ.get(name, "")
        if value.isdecimal() and int(value) > 0:
            budget = min(budget, int(value))
    return max(1, min(len(arms), budget))


def run_check(out: Path, *, device: str = "gpu", lts: bool = False, rate_shift: float = 0.25,
              arms=ARMS, limit: float = 1.5, budget: float = 1e-9, keep_meshes: bool = False,
              cards: int = 1, cases=None, log=print) -> dict[str, Any]:
    # Keep the public fixture selection stable; --cases selects either or both
    # on either device. CPU arms execute concurrently, with vectorized operators.
    cases = tuple(CASES if device == "gpu" else ("gentle",)) if cases is None else tuple(cases)
    if not cases:
        raise ValueError("the jolt check needs at least one case")
    for case in cases:
        levels_for(case)
    if lts and device != "gpu":
        raise ValueError("--lts runs the production LTS executor, which is CUDA only; use --device gpu")
    if int(cards) > 1 and not lts:
        raise ValueError("--cards splits the production LTS executor's classes across cards; add --lts")
    if "fixed" not in arms:
        raise ValueError("the fixed arm is the reference every ratio is taken against")
    out = Path(out).expanduser().absolute()
    out.mkdir(parents=True, exist_ok=True)
    # The fixture's host preparation is small and owned by this run: no shared caches.
    os.environ.setdefault("WOOF_HEX_PREP_CACHE", "off")
    from .stage_timing import stage

    began = time.monotonic()
    stages = []
    with stage(stages, "mesh-family", parallelism={"device": "CPU", "threads": int(os.environ.get("RAYON_NUM_THREADS", "1"))}):
        family = build_family(out / "meshes", log=log)
    graded = {}
    for case in cases:
        if device == "cpu":
            workers = _cpu_workers(arms)
            with stage(stages, f"{case}/arms", parallelism={"device": "CPU", "processes": workers}):
                # Spawn, rather than fork, so no NetCDF or numerical-library
                # locks/handles are inherited from mesh generation.
                with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                    completed = list(pool.map(_cpu_arm, ((arm, case, family) for arm in arms)))
            results = {}
            for arm, (result, arm_stages, messages) in zip(arms, completed):
                results[arm] = result
                stages.extend(arm_stages)
                for message in messages:
                    log(message)
            graded[case] = dict(**evaluate(results, limit=limit, budget=budget), results=results)
            continue
        with stage(stages, f"{case}/host-setup", parallelism={"device": "CPU", "threads": 1}):
            setups = [setup(path, levels=levels_for(case)) for path in family["meshes"]]
        integ = _Integrator(device=device, lts=lts, cfg=setups[0][1], coeff=setups[0][2], winds=setups[0][3],
                            rate_shift=rate_shift if lts else 0.0, cards=cards)
        results = {}
        for arm in arms:
            with stage(stages, f"{case}/{arm}", parallelism={"device": device.upper(), "cards": int(cards) if device == "gpu" else 0,
                       "host_threads": 1, "includes": "fixture, integration, existing diagnostic reads, remap, hashes"}):
                results[arm] = _run_arm(arm, family, integ, setups, log, case=case)
        graded[case] = dict(**evaluate(results, limit=limit, budget=budget), results=results)
    # The first case keeps the receipt's top-level fields (metrics, verdict, results); the others are under
    # ``cases``. The top-level ``passed`` is every case's.
    first, *others = cases
    top = graded[first]
    top["verdict"] = dict(top["verdict"], case=first, passed=all(graded[c]["verdict"]["passed"] for c in cases),
                          cases_passed={c: graded[c]["verdict"]["passed"] for c in cases})
    receipt = dict(schema=SCHEMA, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   device=device, lts=lts, cards=int(cards), rate_shift=rate_shift if lts else 0.0, steps=STEPS, dt_seconds=30.0,
                   move_seconds=[s * 30.0 for s in MOVE_STEPS], window_seconds=WINDOW_STEPS * 30.0,
                   meshes=dict(generator=family["tool"], members=family["receipts"]),
                   **top, cases={c: dict(graded[c], levels=levels_for(c)) for c in others},
                   seconds=round(time.monotonic() - began, 1), stages=stages)
    if device == "gpu":
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        receipt["card"] = dict(name=props["name"].decode(), total_memory_bytes=int(props["totalGlobalMem"]))
    try:
        from . import __version__

        receipt["gpuwm_hex"] = __version__
    except ImportError:  # pragma: no cover
        pass
    (out / "moving-check.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not keep_meshes:
        shutil.rmtree(out / "meshes", ignore_errors=True)
    return receipt


def table(receipt: dict[str, Any]) -> str:
    """The receipt as plain text: one row per move and arm."""

    cards = int(receipt.get("cards", 1))
    lines = [f"moving check ({receipt['device']}{', LTS' if receipt['lts'] else ''}"
             f"{f', {cards} cards' if cards > 1 else ''}): "
             f"jolt = max(|w| ratio, |dpsfc/dt| ratio) over the {receipt['window_seconds']:.0f} s after each move"]
    verdict = receipt["verdict"]
    blocks = [(verdict.get("case", "gentle"), receipt)] + list((receipt.get("cases") or {}).items())
    for case, block in blocks:
        arms = [arm for arm in block["results"] if arm != "fixed"]
        lines.append(f"{case} case: " + ("PASSED" if block["verdict"].get("cases_passed", {}).get(
            case, block["verdict"]["passed"]) else "FAILED"))
        lines.append("move s   " + "".join(f"{arm:>22s}" for arm in arms))
        for row in block["metrics"]:
            lines.append(f"{row['move_seconds']:6.0f}   " + "".join(
                f"{'w ' + format(row[arm]['ratios']['w'], '.3f') + ' dp ' + format(row[arm]['ratios']['dpdt'], '.3f'):>22s}"
                for arm in arms))
        for arm in arms:
            cons = block["verdict"][f"{arm}_worst_conservation"]
            lines.append(f"{arm}: worst jolt {block['verdict'][f'{arm}_worst_jolt']:.3f}; worst relative change "
                         f"mass {cons['mass']:.2e}, theta mass {cons['theta_mass']:.2e}, tracer {cons['tracer']:.2e}")
    lines.append(("PASSED" if verdict["passed"] else "FAILED") +
                 f": balanced moves at most {verdict['limit']} and conserved to {verdict['budget']:g}"
                 f" in every case ({', '.join(case for case, _ in blocks)})")
    return "\n".join(lines)
