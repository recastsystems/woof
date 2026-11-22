"""An idealised parent forecast on a small hill, hand-assembled, for GPU tests.

These fixtures lived in the local data assimilation's nested-forecast tests,
which gpuwm 3.0.3 removed with that code.  Other GPU tests still build their
parent from them, so they are kept here unchanged, minus the child nest the
removed module attached.
"""

from __future__ import annotations

from datetime import datetime
from types import MappingProxyType, SimpleNamespace

import numpy as np

from woof.config import RunConfig
from woof.experiment import (DomainConfig, ExperimentConfig,
                              ProjectionConfig, VerticalConfig)
from woof.physics_compat import (WSM6_PROFILE_ID,
                                  single_domain_runtime_switches)
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

#: Small enough to be a correctness burst, large enough that the child
#: has a real interior outside its own five-point specified frame plus
#: four-cell relaxation zone.
PARENT_NX = PARENT_NY = 33
PARENT_NZ = 49
RATIO = 3
CHILD_NX = CHILD_NY = 27
PARENT_DT = 15.0
#: Four parent steps, hence twelve child steps and four FORCE edges.
RUN_SECONDS = 60.0


# ---------------------------------------------------------------------------
# a cheap parent that nevertheless carries terrain
# ---------------------------------------------------------------------------

def _parent_run() -> RunConfig:
    switches = dict(single_domain_runtime_switches(WSM6_PROFILE_ID))
    switches.pop("acknowledgements", None)
    base = dict(
        nx=PARENT_NX, ny=PARENT_NY, nz=PARENT_NZ,
        dx=3000.0, dy=3000.0, ztop=20000.0, dt=PARENT_DT,
        run_seconds=RUN_SECONDS, output_interval_s=RUN_SECONDS,
        # The parent is PERIODIC here, not specified. The nowcast's parent
        # is specified and reads external LBC tables; this fixture has
        # none and does not need them, because the property under test is
        # about the CHILD's presence, and the parent's own lateral
        # treatment is identical in both arms of the comparison. Making it
        # periodic is what lets the ratchet run without a prepared cache.
        specified=False, nested=False, grid_id=1,
        spec_bdy_width=5, spec_zone=1, relax_zone=4,
        moist=True, hypsometric_opt=2, map_proj=1,
        case="nested_forecast_ratchet",
    )
    base.update({key: value for key, value in switches.items()
                 if key in RunConfig.__dataclass_fields__})
    return RunConfig(**base)


def _experiment(run: RunConfig) -> ExperimentConfig:
    root = DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=RUN_SECONDS, run=run, time_step=int(PARENT_DT))
    return ExperimentConfig(
        name="nested_forecast_ratchet",
        start_time=datetime(2024, 5, 21, 21),
        run_seconds=RUN_SECONDS,
        vertical=VerticalConfig((), 0.0, 1, 0.2),
        projection=ProjectionConfig("lambert", 35.0, -97.0, 30.0, 60.0,
                                    -97.0),
        restart_interval_s=0.0, domains=(root,))


def _terrain(run: RunConfig) -> np.ndarray:
    """A smooth hill: enough relief that the terrain SINT branch matters."""
    j = np.arange(run.ny)[:, None]
    i = np.arange(run.nx)[None, :]
    cj, ci = (run.ny - 1) / 2.0, (run.nx - 1) / 2.0
    radius = np.sqrt(((i - ci) / (run.nx / 3.0)) ** 2
                     + ((j - cj) / (run.ny / 3.0)) ** 2)
    return (400.0 * np.exp(-radius ** 2)).astype(np.float64)


def _build_parent_state(run: RunConfig):
    """A hydrostatically balanced parent over that hill, with a thermal.

    ``terrain_opt=1`` is not decoration: it is what the shipped nowcast
    profile carries, and it selects the three-dimensional base-state
    allocation and the terrain branch of ``parent_only_init``'s base
    capture.  A flat fixture would exercise a different branch than the
    one production takes.  ``init_theta_perturbation`` is the terrain-
    capable balanced initialiser (``init_moist_balanced``, which WK82
    uses, refuses a terrain base state by construction).
    """
    import cupy as cp

    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import DTYPE, init_theta_perturbation
    from woof.verify.cases import wk82

    coord = make_vertical_coord(run.nz)
    terrain = _terrain(run)
    base = make_base_state(
        coord, lambda z: wk82.wk82_sounding(z)[0],
        p_surf=run.p_surf, ztop=run.ztop, terrain_z=terrain)

    y = (np.arange(run.ny) + 0.5) * run.dy - 0.5 * run.ny * run.dy

    def thp_func(x, z):
        """A 3 K ellipsoidal thermal, so the child has something to do."""
        xrad = x[None, None, :] / 12_000.0
        yrad = y[None, :, None] / 12_000.0
        zrad = (z - 1_500.0) / 1_500.0
        rad = np.sqrt(xrad ** 2 + yrad ** 2 + zrad ** 2)
        return np.where(rad <= 1.0,
                        3.0 * np.cos(0.5 * np.pi * rad) ** 2, 0.0)

    state = init_theta_perturbation(run, coord, base, thp_func,
                                    terrain_z=terrain)
    # A sheared hodograph so the coupler's boundary tables carry real
    # gradients rather than a uniform field that hides an error.
    u_profile, v_profile = wk82.hodograph_uv(coord, run.ztop)
    state.u[...] = cp.asarray(u_profile, dtype=DTYPE)[:, None, None]
    state.v[...] = cp.asarray(v_profile, dtype=DTYPE)[:, None, None]
    # The EOS diagnostics are what the health gate reads first; the
    # production initialisers refresh them at the end of setup and this
    # fixture has to do the same.
    from woof.core.diagnostics import update_diagnostics
    update_diagnostics(state, run.hypsometric_opt)
    return state, coord


def _synthetic_land(run: RunConfig, terrain: np.ndarray):
    """Parent-grid static and Noah inventories with valid categories.

    Values are synthetic; their JOB in this test is to be a real,
    self-consistent land inventory the child's nest-down and physics
    initialisation must survive, not to be a real place.
    """
    ny, nx = run.ny, run.nx
    rng = np.random.default_rng(20240521)
    landmask = (rng.random((ny, nx)) > 0.25).astype(np.float32)
    lu_index = np.where(landmask >= 0.5,
                        rng.integers(1, 15, size=(ny, nx)),
                        17).astype(np.float32)      # 17 = ISWATER
    sct_dom = np.where(landmask >= 0.5,
                       rng.integers(1, 13, size=(ny, nx)),
                       14).astype(np.float32)       # 14 = water soil
    months = np.linspace(0.1, 0.8, 12)[:, None, None]
    static = {
        "HGT_M": terrain.astype(np.float32),
        "LANDMASK": landmask,
        "LU_INDEX": lu_index,
        "SCT_DOM": sct_dom,
        "SCB_DOM": sct_dom.copy(),
        "SNOALB": np.full((ny, nx), 0.55, np.float32),
        "SOILTEMP": np.full((ny, nx), 287.0, np.float32),
        "TMN": np.full((ny, nx), 287.0, np.float32),
        "LANDUSEF": np.zeros((21, ny, nx), np.float32),
        "SOILCTOP": np.zeros((16, ny, nx), np.float32),
        "SOILCBOT": np.zeros((16, ny, nx), np.float32),
        "GREENFRAC": np.broadcast_to(
            months, (12, ny, nx)).astype(np.float32).copy(),
        "LAI12M": np.broadcast_to(
            2.0 * months, (12, ny, nx)).astype(np.float32).copy(),
        "ALBEDO12M": np.full((12, ny, nx), 0.18, np.float32),
    }
    for k in range(21):
        static["LANDUSEF"][k] = (lu_index == (k + 1)).astype(np.float32)
    for k in range(16):
        static["SOILCTOP"][k] = (sct_dom == (k + 1)).astype(np.float32)
        static["SOILCBOT"][k] = static["SOILCTOP"][k]
    surface = {
        "TSK": np.full((ny, nx), 293.0, np.float32),
        "TSLB": np.full((4, ny, nx), 289.0, np.float32),
        "SMOIS": np.full((4, ny, nx), 0.28, np.float32),
        "SH2O": np.full((4, ny, nx), 0.26, np.float32),
        "TMN": np.full((ny, nx), 287.0, np.float32),
        "SEAICE": np.zeros((ny, nx), np.float32),
        "SNOW": np.zeros((ny, nx), np.float32),
        "SNOWH": np.zeros((ny, nx), np.float32),
        "LANDMASK": landmask,
        "XLAND": np.where(landmask >= 0.5, 1.0, 2.0).astype(np.float32),
    }
    identity = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
                "ISLAKE": 21, "ISICE": 15}
    return static, SimpleNamespace(fields=MappingProxyType(surface)), identity


# ---------------------------------------------------------------------------
# model assembly, exactly as the DA driver does it
# ---------------------------------------------------------------------------

def _wire(exp, *, attach_nest, geometry=None):
    """Hand-build the ExperimentState the DA driver builds, +/- the nest."""
    import cupy as cp

    from woof.core.clock import build_schedule, resolve_clock
    from woof.core.model import (DomainNode, ExperimentState,
                                  ModelRuntimeStatus)
    from woof.core.physics import initialize_physics
    from woof.static.lambert import grids_from_projection_config

    run = exp.root.run
    state, _coord = _build_parent_state(run)
    static, surface, identity = _synthetic_land(run, _terrain(run))
    grid = grids_from_projection_config(exp)[0]

    from woof.core.landuse import initialize_landuse
    lat, lon = grid.latlon_mass()
    landuse = initialize_landuse(
        static["LU_INDEX"], soil_type=static["SCT_DOM"],
        landmask=static["LANDMASK"], snow=surface.fields["SNOW"],
        xice=surface.fields["SEAICE"], valid_time=exp.start_time,
        cen_lat=float(getattr(grid, "cen_lat", grid.ref_lat)),
        mminlu=identity["MMINLU"], iswater=identity["ISWATER"],
        islake=identity["ISLAKE"], isice=identity["ISICE"],
        fractional_seaice=True,
        soil_temperature=surface.fields["TSLB"])
    driver = initialize_physics(
        state, run, landuse=landuse, tsk=surface.fields["TSK"],
        soil_temperature=surface.fields["TSLB"],
        soil_moisture=surface.fields["SMOIS"],
        liquid_moisture=surface.fields["SH2O"],
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=100.0 * static["GREENFRAC"][4],
        tmn=surface.fields["TMN"], xice=surface.fields["SEAICE"],
        snow=surface.fields["SNOW"], snow_depth=surface.fields["SNOWH"],
        sst=surface.fields["TSK"],
        radiation_start_time=exp.start_time,
        radiation_latitude=lat, radiation_longitude=lon, glw=_IDEALISED_GLW)

    if attach_nest:
        raise ValueError("the child nest of the removed local data "
                         "assimilation is not available")
    nested_exp = exp

    tick = resolve_clock(nested_exp)
    schedule = build_schedule(nested_exp, tick)
    clocks = tick.clocks()
    root = DomainNode(exp.root, grid, state, clocks[1], None, [], None)
    nodes = {1: root}
    child_driver = None

    model = ExperimentState(root, MappingProxyType(nodes), schedule, None,
                            "nested-forecast-ratchet")
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    # The shared arena and the shared dycore workspace are what would let
    # two domains write the same bytes; the DA driver disables both and so
    # does this path.  Asserted, not assumed.
    model._scratch_arena = None
    model._dycore_state_workspace = None
    model._io_manager = None
    model._last_checkpoint = None
    model._prepared_by_grid_id = MappingProxyType({})
    return model, root, nodes, driver, child_driver


def _run(model):
    import cupy as cp

    from woof.core.model import execute_experiment

    execute_experiment(model, history_handler=None, progress_callback=None,
                       validate_state=True, skip_feedback_path=True,
                       pool_trim_per_period=True)
    cp.cuda.Stream.null.synchronize()
