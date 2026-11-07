"""The dycore's uncached diffusion compile runs during the input read.

The defect (NV recipe dry run, D5): on a warm kernel cache the first step of
NCAR's CONUS 12 km restart leg still took 22 to 37 s; a sampler put 32 s of
a 33.5 s step 1 inside ``compile_diffusion_source`` (smag2d and diff_opt1
through CuPy's direct, uncached NVRTC route).  The WRF door's worker now
starts those same loader calls on a background thread before the read.

CPU only: the loader and CuPy's device are stood in for.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from woof import kernel_prewarm


def _experiment(**run):
    defaults = {"diff_opt": 1, "km_opt": 4, "diff_6th_opt": 0}
    defaults.update(run)
    return SimpleNamespace(domains=(SimpleNamespace(run=SimpleNamespace(**defaults)),))


def test_ncar_conus_namelist_warms_smag2d_and_diff_opt1():
    # NCAR's v4.4 CONUS benchmark: diff_opt = 1, km_opt = 4, no sixth order.
    assert kernel_prewarm.modules_for(_experiment()) == ("smag2d", "diff_opt1")


def test_diff_opt2_and_sixth_order():
    assert kernel_prewarm.modules_for(_experiment(diff_opt=2, diff_6th_opt=2)) == (
        "smag2d", "diff6")


def test_no_domain_no_module():
    assert kernel_prewarm.modules_for(SimpleNamespace(domains=())) == ()
    assert kernel_prewarm.modules_for(object()) == ()


def test_env_turns_it_off(monkeypatch):
    monkeypatch.setenv(kernel_prewarm.ENV, "0")
    assert kernel_prewarm.start_for(_experiment()) is None


def _fake_cupy(monkeypatch):
    devices = []

    class Device:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            devices.append(self.index)
            return self

        def __exit__(self, *exc):
            return False

    cupy = types.ModuleType("cupy")
    cupy.cuda = SimpleNamespace(Device=Device,
                                runtime=SimpleNamespace(getDeviceCount=lambda: 1))
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    return devices


def test_prewarm_calls_the_runs_own_loader_on_device_0(monkeypatch):
    devices = _fake_cupy(monkeypatch)
    from woof.core import kernels

    loaded = []
    monkeypatch.setattr(kernels, "_load_diffusion_module", lambda name: loaded.append(name))
    receipt = kernel_prewarm.Prewarm(("smag2d", "diff_opt1")).start().join(30)
    assert loaded == ["smag2d", "diff_opt1"]
    assert devices == [0]
    assert receipt["errors"] == {} and set(receipt["seconds"]) == {"smag2d", "diff_opt1"}


def test_prewarm_never_builds_a_physics_parameter_module(monkeypatch):
    _fake_cupy(monkeypatch)
    from woof import physics_params
    from woof.core import kernels

    loaded = []
    monkeypatch.setattr(kernels, "_load_diffusion_module", lambda name: loaded.append(name))
    affected = sorted(physics_params._affected_modules())
    assert affected, "the registry names modules with parameter literals"
    monkeypatch.setattr(kernels, "_DIFFUSION_MODULES",
                        frozenset(kernels._DIFFUSION_MODULES | {affected[0]}))
    receipt = kernel_prewarm.Prewarm((affected[0], "smag2d")).start().join(30)
    assert loaded == ["smag2d"]
    assert "physics-parameter" in receipt["errors"][affected[0]]


def test_prewarm_refuses_a_unit_off_the_diffusion_route(monkeypatch):
    _fake_cupy(monkeypatch)
    from woof.core import kernels

    loaded = []
    monkeypatch.setattr(kernels, "_load_diffusion_module", lambda name: loaded.append(name))
    receipt = kernel_prewarm.Prewarm(("thompson", "smag2d")).start().join(30)
    assert loaded == []
    assert "not diffusion-route units" in receipt["errors"]["*"]


def test_every_warmed_unit_is_a_diffusion_route_unit():
    from woof.core import kernels

    assert set(kernel_prewarm.DIFFUSION_ROUTE_MODULES) <= set(kernels._DIFFUSION_MODULES)


def test_a_failed_compile_is_recorded_not_raised(monkeypatch):
    _fake_cupy(monkeypatch)
    from woof.core import kernels

    def boom(name):
        raise RuntimeError("nvrtc said no")

    monkeypatch.setattr(kernels, "_load_diffusion_module", boom)
    receipt = kernel_prewarm.Prewarm(("smag2d",)).start().join(30)
    assert receipt["errors"]["smag2d"] == "RuntimeError: nvrtc said no"


def test_more_than_one_visible_card_is_not_warmed(monkeypatch):
    _fake_cupy(monkeypatch)
    sys.modules["cupy"].cuda.runtime.getDeviceCount = lambda: 2
    monkeypatch.setattr("woof.local_gpu.no_local_gpu", lambda: False)
    assert kernel_prewarm.start_for(_experiment()) is None


def test_the_worker_starts_it_before_the_read():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "woof" / "wrfinput_forecast.py"
            ).read_text(encoding="utf-8")
    start = text.index("kernel_prewarm.start_for(getattr(run, 'experiment', None))")
    read = text.index("inputs = prepare_wrf_run(run, worker_output/'input'")
    assert start < read


def test_the_diffusion_compile_keeps_its_image_in_the_persistent_cache():
    """compile_diffusion_source compiles through woof.nvrtc_ptx_cache.

    CuPy's direct NVRTC entry point has no disk cache, so before this every
    run recompiled smag2d on step 1 even on a warm kernel cache.  The alias
    keeps the call text (and with it the FTZ route inventory row) as it was.
    """
    import ast
    from pathlib import Path

    from woof.core import kernels

    tree = ast.parse(Path(kernels.__file__).read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "compile_diffusion_source")
    bound = {alias.asname or alias.name: f"{node.module}.{alias.name}"
             for node in ast.walk(function) if isinstance(node, ast.ImportFrom)
             for alias in node.names}
    assert bound["compiler"] == "woof.nvrtc_ptx_cache"


def test_composed_inventory_uses_runtime_factory_not_nonexistent_file(monkeypatch):
    selected = tuple(kernel_prewarm._COMPOSED_LOADERS)
    monkeypatch.setattr(kernel_prewarm, 'static_modules', lambda experiment: selected)
    monkeypatch.setattr(kernel_prewarm, '_inline_jobs', lambda modules: [])
    jobs = kernel_prewarm.static_jobs(_experiment())
    assert {name for name, factory, args in jobs} == set(selected)
    assert all(factory is kernel_prewarm._warm_composed and args == (name,)
               for name, factory, args in jobs)
    # Execute every dispatcher against a substituted factory at the real
    # module/function address; SW must receive its actual table object.
    import importlib
    calls = []
    tables = object()
    for name, (module, function) in kernel_prewarm._COMPOSED_LOADERS.items():
        target = importlib.import_module('woof.core.' + module)
        monkeypatch.setattr(target, function, lambda *args, n=name: calls.append((n,args)))
        if name == 'rrtmg_sw_legacy':
            monkeypatch.setattr(target, '_sw_tables', lambda: tables)
        kernel_prewarm._warm_composed(name)
    assert calls == [(n, (tables,) if n == 'rrtmg_sw_legacy' else ()) for n in selected]


@pytest.mark.parametrize("aerosol_options, expected", [
    ((0,), {(("GPUWM_MSKF_NO_AEROSOL", 1),)}),
    ((1,), {()}),
    ((0, 1), {(), (("GPUWM_MSKF_NO_AEROSOL", 1),)}),
])
def test_mskf_warms_only_selected_runtime_specializations(monkeypatch, aerosol_options, expected):
    from woof.core.kernels.no_ftz import load_module_no_ftz

    experiment = SimpleNamespace(domains=tuple(
        SimpleNamespace(run=SimpleNamespace(cu_physics=11, aercu_opt=value))
        for value in aerosol_options))
    monkeypatch.setattr(kernel_prewarm, "static_modules", lambda experiment: ("mskf",))
    jobs = [(factory, args) for label, factory, args in kernel_prewarm.static_jobs(experiment)
            if label.split("[")[0] == "mskf"]
    assert {args[1] for _, args in jobs} == expected
    assert len(jobs) == len(expected)
    assert all(factory is load_module_no_ftz.__wrapped__ and args[0] == "mskf"
               for factory, args in jobs)


def test_the_forecast_column_is_open_so_the_helicity_diagnostic_runs(monkeypatch):
    # Breakage it prevents (2.9.4 landing, test host 2026-10-20): the synthetic forecast column was periodic, the run's
    # nwp_diagnostics=1 helicity refused it ("WRF's cal_helicity periodic branch is not transcribed"), and the HRRR
    # forecast's step 1 compiled 51 s cold.
    import dataclasses

    import woof.config as config
    from woof import warm_kernels
    from woof.core.uh_diag import _supported_boundary_geometry

    @dataclasses.dataclass
    class Run:
        dt: float = 15.0
        nx: int = 500
        ny: int = 400
        terrain_opt: int = 1
        run_seconds: float = 21600.0
        clock_dt: float = 15.0
        output_interval_s: float = 3600.0
        open_x: bool = False
        open_y: bool = False
        specified: bool = True
        nested: bool = False
        nwp_diagnostics: int = 1
        spec_bdy_width: int = 10
        zadvect_implicit: int = 1

    monkeypatch.setattr(config, "validate_run_config", lambda cfg: cfg)
    # A specified run (the HRRR recipe) warms on a specified column wide enough for its boundary width: the
    # helicity diagnostic accepts it, and so does zadvect_implicit = 1, which refused the open column ("with
    # radiative open boundaries is not wired", hrrr_v4_gsd41, 2026-10-21).
    cfg = warm_kernels.forecast_column_config(Run())
    assert cfg.nwp_diagnostics == 1 and _supported_boundary_geometry(cfg)
    assert cfg.output_interval_s == Run.output_interval_s
    assert cfg.specified and not cfg.nested and not (cfg.open_x or cfg.open_y)
    assert cfg.nx == cfg.ny == 2 * Run.spec_bdy_width + 4
    # A run with no lateral forcing keeps the open column.
    cfg = warm_kernels.forecast_column_config(Run(specified=False, zadvect_implicit=0))
    assert cfg.nwp_diagnostics == 1 and _supported_boundary_geometry(cfg)
    assert not cfg.specified and cfg.open_x and cfg.open_y
    assert (cfg.nx, cfg.ny) == (warm_kernels._NX, warm_kernels._NY)


def test_forecast_column_keeps_the_native_nwp_averaging_window():
    from woof.config import RunConfig, validate_run_config
    from woof.warm_kernels import forecast_column_config

    run = validate_run_config(RunConfig(
        nx=40, ny=32, nz=49, dx=3000., dy=3000., ztop=20000.,
        dt=12., run_seconds=120., output_interval_s=60., nwp_diagnostics=1))
    column = forecast_column_config(run)
    assert column.run_seconds == 24.
    assert column.output_interval_s == 60.
    assert column.nwp_diagnostics == 1


def _column_layer_temperatures(nz, ztop):
    import numpy as np

    from woof.core import constants as c
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.warm_kernels import _theta

    base = make_base_state(make_vertical_coord(nz), _theta, p_surf=1.0e5, ztop=ztop)
    return np.asarray(base.thb) * (np.asarray(base.pb) / c.P0) ** c.RCP


def test_fcn3_nest_column_stays_inside_the_radiation_tables():
    # The Earth-2 FCN3 nest (test host 2026-10-20, repro-e2 g.toml): nz 49, ztop
    # 20 km, p_top 50 hPa.  The constant-N column's top layer was 159.2 K,
    # below RRTMG's 160 K floor, and the pre-warm child exited 1.
    temperature = _column_layer_temperatures(49, 20000.0)
    assert temperature.min() >= 160.0
    assert temperature.max() <= 355.0


def test_column_is_valid_for_any_model_top():
    for nz, ztop in ((40, 16000.0), (60, 30000.0), (80, 45000.0)):
        temperature = _column_layer_temperatures(nz, ztop)
        assert 160.0 <= temperature.min() and temperature.max() <= 355.0, (nz, ztop)


def _hrrr_recipe_experiment():
    from importlib.resources import files

    from woof.experiment import load_experiment

    return load_experiment(str(files("woof") / "data" / "recipes" / "hrrr_v4_gsd41.toml"))


def _loader_route_exists(factory, args) -> bool:
    import importlib

    from woof.core import kernels

    if factory is kernel_prewarm._warm_composed:
        module, function = kernel_prewarm._COMPOSED_LOADERS[args[0]]
    elif factory is kernel_prewarm._warm_device_loader:
        module, function = kernel_prewarm._DEVICE_LOADERS[args[0]]
    elif factory is kernel_prewarm._warm_kf:
        module, function = "kf", "_kf_module"
    elif factory is kernels.load_module.__wrapped__:
        source = "mynn_pbl" if args[0] == "mynn_dmp_sibling" else args[0]
        return (kernels._KDIR / f"{source}.cu").exists()
    else:
        return True
    return callable(getattr(importlib.import_module("woof.core." + module), function, None))


def test_every_static_job_of_the_hrrr_recipe_has_a_loader_route():
    # test host 8x5090, 2026-10-21: the hrrr_v4_gsd41 plan named terrain_drag_composed (gwd_opt = 3), load_module looked
    # for terrain_drag_composed.cu, and the pre-warm child exited 1 with FileNotFoundError.
    from woof.core import kernels
    from woof.core.preflight import CHAINED_TRANSLATION_UNIT_FRAMES

    jobs = kernel_prewarm.static_jobs(_hrrr_recipe_experiment())
    assert "terrain_drag_composed" in {label for label, _factory, _args in jobs}
    for label, factory, args in jobs:
        assert _loader_route_exists(factory, args), label
    # Every chained unit uses its production factory. Eta has a source file,
    # but its factory also supplies the required composed math header.
    for name in CHAINED_TRANSLATION_UNIT_FRAMES:
        if name in kernel_prewarm._KF_TRIGGERS:
            assert _loader_route_exists(
                kernel_prewarm._warm_kf, (kernel_prewarm._KF_TRIGGERS[name],)), name
            continue
        assert name in kernel_prewarm._COMPOSED_LOADERS or name in kernel_prewarm._DEVICE_LOADERS, name
        factory = (kernel_prewarm._warm_composed
                   if name in kernel_prewarm._COMPOSED_LOADERS
                   else kernel_prewarm._warm_device_loader)
        assert _loader_route_exists(factory, (name,)), name


@pytest.mark.parametrize("name,trigger", [("kf", 1), ("kf_trigger2", 2), ("kf_trigger3", 3)])
def test_kf_prewarm_uses_the_production_trigger_variant(monkeypatch, name, trigger):
    from woof.core import kf

    monkeypatch.setattr(kernel_prewarm, "static_modules", lambda _experiment: (name,))
    monkeypatch.setattr(kernel_prewarm, "_inline_jobs", lambda _modules: [])
    monkeypatch.setattr(kernel_prewarm, "_variant_jobs", lambda *_args: [])
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: 7))))
    calls = []
    monkeypatch.setattr(kf, "_kf_module", lambda *args: calls.append(args))
    jobs = kernel_prewarm.static_jobs(_experiment())
    assert len(jobs) == 1
    _label, factory, args = jobs[0]
    factory(*args)
    assert calls == [(7, trigger)]


def test_parallel_plan_covers_selected_physics_and_code_dependent_shapes():
    experiment = _hrrr_recipe_experiment()
    jobs = kernel_prewarm.static_jobs(experiment)
    tiers = {(args[0], args[1]) for _, _, args in jobs
             if len(args) == 2 and isinstance(args[1], tuple)}
    assert ("thompson_aerosol_sed", (("THOMPSON_AA_WRF39", 1),)) in tiers
    assert ("mynn_pbl", (("MYNN_GSD41", 1),)) in tiers
    for cfg in [experiment.domains[0].run, *kernel_prewarm.forecast_columns(experiment)]:
        dims = (("GPUWM_SHAPE_NZ", cfg.nz), ("GPUWM_SHAPE_NY", cfg.ny),
                ("GPUWM_SHAPE_NX", cfg.nx))
        assert ("acoustic", dims) in tiers
        columns = (("GPUWM_SHAPE_NZ", cfg.nz), ("GPUWM_SHAPE_NY", 1),
                   ("GPUWM_SHAPE_NX", cfg.ny * cfg.nx))
        assert ("mynn_pbl", (("MYNN_GSD41", 1),) + columns) in tiers


def test_hrrr_recipe_prewarm_receipt_says_first_step_ready(monkeypatch):
    # The whole warm_static pass on the hrrr_v4_gsd41 recipe with the compiler stubbed: every static loader route
    # is resolved and called, the preparation and forecast-column groups run, and the receipt says
    # first_step_ready.  A job without a loader route fails the way test host's did (FileNotFoundError).
    from woof import kernel_plan
    from woof.ingest import preparation_prewarm

    compiled = []
    real_jobs = kernel_prewarm.static_jobs

    def stubbed(label, ok):
        def factory(*args):
            if not ok:
                raise FileNotFoundError(f"{label}.cu")
            compiled.append(label)
        return factory

    def stubbed_jobs(experiment, **kwargs):
        return [(label, stubbed(label, _loader_route_exists(factory, args)), args)
                for label, factory, args in real_jobs(experiment, **kwargs)]

    monkeypatch.setattr(kernel_prewarm, "static_jobs", stubbed_jobs)
    monkeypatch.setattr(kernel_prewarm, "submit_static",
                        lambda experiment, **kwargs: len(stubbed_jobs(experiment, **kwargs)))
    monkeypatch.setattr(kernel_plan, "install", lambda **kwargs: True)
    monkeypatch.setattr(kernel_plan, "start", lambda: {})
    monkeypatch.setattr(kernel_plan, "close", lambda **kwargs: None)
    monkeypatch.setattr(kernel_plan, "receipt", lambda: {"stub": True})
    monkeypatch.setattr(preparation_prewarm, "warm_preparation_job", lambda options: None)
    monkeypatch.setattr(kernel_prewarm, "warm_forecast_columns", lambda experiment, **kwargs: 1)
    monkeypatch.setattr(kernel_prewarm, "warm_column_reductions", lambda experiment: 1)

    receipt = kernel_prewarm.warm_static(_hrrr_recipe_experiment(), preparation=True, forecast=True)
    assert receipt["errors"] == {}
    assert "terrain_drag_composed" in compiled
    assert receipt["loaded"] == receipt["static_jobs"] == len(compiled)
    assert receipt["preparation"] == {"groups": 4, "loaded": 4}
    assert receipt["forecast_columns"] == 1
    assert receipt["first_step_ready"] is True
    assert kernel_prewarm.first_step_ready({**receipt, "errors": {"x": "y"}}) is False
    assert kernel_prewarm.first_step_ready({**receipt, "forecast_columns": 0}, forecast=True) is False


def test_synthetic_column_does_not_construct_inherited_forecast_hydro(monkeypatch):
    from woof import warm_kernels
    from woof.core import physics
    from woof.hydro.runtime import current_settings, settings_scope

    settings = {"enabled": True, "fabric_manifest": "actual-parent-grid/manifest.json"}
    seen = []

    def initialize(*args, **kwargs):
        seen.append((args, kwargs, current_settings()))
        return "column driver"

    monkeypatch.setattr(physics, "initialize_physics", initialize)
    monkeypatch.setenv("WOOF_NEXTGEN_ENABLED", "1")
    with settings_scope(settings):
        assert warm_kernels._initialize_column_physics("state", cfg="column") == "column driver"
        assert current_settings() is settings
        def fail(*args, **kwargs):
            raise ValueError("column failed")
        monkeypatch.setattr(physics, "initialize_physics", fail)
        import pytest
        with pytest.raises(ValueError, match="column failed"):
            warm_kernels._initialize_column_physics("state")
        assert current_settings() is settings
    assert seen == [(("state",), {"cfg": "column"},
                     {"enabled": False, "activation_reason": "synthetic_kernel_probe"})]


def test_hrrr_recipe_column_carries_mosaic_fractions_and_orographic_statics():
    # 2026-10-21, hrrr_v4_gsd41 (mosaic_lu = mosaic_soil = 1, gwd_opt = 3): the forecast column refused
    # "RUC mosaic needs LANDUSEF category fractions", then "this initialization route hands the physics no static
    # field set", so the run's own variants compiled in step 1.
    import numpy as np

    from woof import warm_kernels
    from woof.core.ruc import load_ruc_parameters
    from woof.static.orographic import required_static_fields

    run = _hrrr_recipe_experiment().domains[0].run
    cfg = warm_kernels.forecast_column_config(run)
    grid = (cfg.ny, cfg.nx)
    fractions = warm_kernels._synthetic_mosaic_fractions(cfg, grid)
    bundle = load_ruc_parameters()
    assert fractions["landusef"].shape == (len(bundle.vegetation_for("MODIFIED_IGBP_MODIS_NOAH").rows), *grid)
    assert fractions["soilctop"].shape == (len(bundle.soil.rows), *grid)
    for value in fractions.values():
        assert np.allclose(value.sum(axis=0), 1.0)
    static = warm_kernels._synthetic_orographic_static(cfg, grid)["terrain_drag_static"]
    assert static["HGT_M"].shape == grid
    for name in required_static_fields(int(cfg.topo_wind), int(cfg.gwd_opt)):
        assert static[name].shape == grid and np.isfinite(static[name]).all()
    # Suites without a mosaic or a drag get nothing extra.
    import dataclasses

    plain = dataclasses.replace(cfg, mosaic_lu=0, mosaic_soil=0, gwd_opt=0, topo_wind=0)
    assert warm_kernels._synthetic_mosaic_fractions(plain, grid) == {}
    assert warm_kernels._synthetic_orographic_static(plain, grid) == {}


def test_the_columns_own_radiation_layer_count_is_in_the_static_plan(monkeypatch):
    # 2.9.4 R11 (2026-10-22): the synthetic column takes its top from its own sounding at ztop, so its
    # RTE compiled for 63 layers against the run's 62 (p_top 50 hPa, nz 49): that one variant compiled
    # alone inside the column, 5.7 s on one core on sm_120 (7 s on the 5090), the pre-warm's single-core tail.
    from woof import warm_kernels
    from woof.core.rrtmgp import rrtmgp_above_model_layer_counts

    experiment = _hrrr_recipe_experiment()
    real_modules = kernel_prewarm.static_modules
    monkeypatch.setattr(kernel_prewarm, "static_modules", lambda e: (*real_modules(e), "rrtmgp_rte"))

    def tiers(columns):
        return {args[1][0][1] for label, _factory, args in kernel_prewarm.static_jobs(experiment, columns=columns)
                if label.startswith("rrtmgp_rte[RRTMGP_MAX_LAYERS=")}

    column = {int(cfg.nz) + extra for cfg in kernel_prewarm.forecast_columns(experiment)
              for extra in rrtmgp_above_model_layer_counts(warm_kernels.column_p_top(cfg))}
    assert column <= tiers(True)
    assert tiers(False) <= tiers(True)
    # The column's top is its sounding's at ztop (float32 on the state), not the run's declared p_top.
    cfg = kernel_prewarm.forecast_columns(experiment)[0]
    assert 0.0 < warm_kernels.column_p_top(cfg) < 20000.0


def test_the_forecast_column_runs_beside_the_static_compile_not_after_it(monkeypatch):
    # 2.9.4 R11: the column ran only after every static module had loaded, so its chain of first-use
    # compiles (one kernel at a time) ran on one core after the parallel pool drained (14 s at 1.09 cores).
    import threading

    from woof import kernel_plan

    events = []
    lock = threading.Lock()

    def note(event):
        with lock:
            events.append(event)

    def job(label):
        return (label, lambda *args: note("load " + label), ())

    monkeypatch.setattr(kernel_plan, "install", lambda **kwargs: True)
    monkeypatch.setattr(kernel_plan, "start", lambda: {})
    monkeypatch.setattr(kernel_plan, "close", lambda **kwargs: None)
    monkeypatch.setattr(kernel_plan, "save", lambda: None)
    monkeypatch.setattr(kernel_plan, "receipt", lambda: {"stub": True})
    monkeypatch.setattr(kernel_prewarm, "plan_tag", lambda experiment: "tag")
    monkeypatch.setattr(kernel_prewarm, "static_jobs", lambda experiment, **kwargs: [job("a"), job("b")])
    monkeypatch.setattr(kernel_prewarm, "submit_static",
                        lambda experiment, **kwargs: note(f"submit columns={kwargs.get('columns')}") or 2)
    monkeypatch.setattr(kernel_prewarm, "warm_column_reductions", lambda experiment: note("reductions") or 1)
    tables = {"staged": True}

    def column(experiment, *, wait_s=kernel_prewarm.FORECAST_TABLE_WAIT_S, reductions=True):
        assert reductions is False
        note(f"column wait={wait_s}")
        if not tables["staged"]:
            tables["staged"] = True
            raise FileNotFoundError("THOMPSON tables not staged yet")
        return 1
    monkeypatch.setattr(kernel_prewarm, "warm_forecast_columns", column)

    receipt = kernel_prewarm.warm_static(SimpleNamespace(domains=()), forecast=True)
    assert receipt["errors"] == {} and receipt["forecast_columns"] == 1 and receipt["first_step_ready"]
    assert events.index("submit columns=True") < events.index("column wait=0.0") < events.index("load a")
    assert "reductions" in events

    # A physics table go has not staged yet sends the column behind the static loads, full wait.
    events.clear()
    tables["staged"] = False
    receipt = kernel_prewarm.warm_static(SimpleNamespace(domains=()), forecast=True)
    assert receipt["errors"] == {} and receipt["forecast_columns"] == 1
    order = [e for e in events if e != "reductions"]
    assert order == ["submit columns=True", "column wait=0.0", "load a", "load b",
                     f"column wait={kernel_prewarm.FORECAST_TABLE_WAIT_S}"]


@pytest.mark.gpu
def test_the_real_domains_staggered_relaxations_and_joined_substep_are_in_the_static_plan():
    # fix/hydro-attach-replay (2026-10-22, cold ERA5 go 3 km 60x48 on a 5070 Ti): step 1 compiled four
    # state-relaxation shapes (u, v, w/phi, mu: 0.5 s each) and the domain's joined acoustic unit (0.4 s)
    # one after another on one core; the plan named only the mass shape and the column has no specified frame.
    from woof.kernel_prewarm import joined_prewarm_units
    from woof.ingest.lateral_bc import relaxation_shapes

    experiment = _hrrr_recipe_experiment()
    jobs = kernel_prewarm.static_jobs(experiment)
    labels = {label for label, _factory, _args in jobs}
    run = experiment.domains[0].run
    for nz, ny, nx in relaxation_shapes(run.nz, run.ny, run.nx):
        assert f"lbc_state[GPUWM_SHAPE_NZ={nz},GPUWM_SHAPE_NY={ny},GPUWM_SHAPE_NX={nx}]" in labels
    units = joined_prewarm_units(run)
    assert units
    joined = {args[0] for label, _factory, args in jobs if label.startswith("acoustic+joined[")}
    assert set(units) <= joined


def test_hydro_routing_units_are_in_the_static_plan_unless_hydro_is_off(monkeypatch):
    # fix/hydro-attach-replay (2026-10-22): hydro attach compiled its routing units on the forecast's one
    # thread after the hydro wait.
    __import__("pytest").importorskip("cupy")
    hydro = {"hydro:troute-mc", "hydro:routed-replay", "hydro:reservoirs", "hydro:runoff-routing",
             "hydro:catchment-exchange"}
    monkeypatch.setattr(kernel_prewarm, "static_modules", lambda experiment: ())
    monkeypatch.setattr(kernel_prewarm, "_inline_jobs", lambda modules: [])
    labels = lambda **kw: {label for label, _f, _a in kernel_prewarm.static_jobs(_experiment(), **kw)}
    assert hydro <= labels(hydro=True)
    assert not hydro & labels()
    monkeypatch.delenv("WOOF_NEXTGEN_ENABLED", raising=False)
    assert kernel_prewarm.hydro_wanted()
    monkeypatch.setenv("WOOF_NEXTGEN_ENABLED", "0")
    assert not kernel_prewarm.hydro_wanted()
