"""The urban selector's door: WRF's keys, WRF's defaults, and refusals that
name what they prevent."""
from __future__ import annotations

import dataclasses
import sys
import types

import pytest

from woof.config import (RunConfig, URBAN_MODEL_MODULES, urban_model_in_build,
                          validate_urban_config)


def _cfg(**kw) -> RunConfig:
    base = dict(nx=8, ny=8, nz=10, dx=3000.0, dy=3000.0, ztop=20000.0,
                dt=10.0, run_seconds=60.0, sf_sfclay_physics=1,
                sf_surface_physics=2, bl_pbl_physics=1)
    base.update(kw)
    return RunConfig(**base)


@pytest.fixture
def all_models(monkeypatch):
    """Pretend the three model lanes are merged, to test the law around them."""
    for name in URBAN_MODEL_MODULES.values():
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr("woof.config.urban_model_in_build", lambda option: True)


def test_defaults_are_wrfs_and_reach_nothing():
    cfg = _cfg()
    assert (cfg.sf_urban_physics, cfg.use_wudapt_lcz, cfg.num_urban_hi) == (
        0, 0, 15)
    validate_urban_config(cfg)
    # With no urban model, the companions are read by nothing: no value of
    # them can make a run wrong, so none is refused.
    validate_urban_config(_cfg(use_wudapt_lcz=7, num_urban_hi=3,
                               sf_surface_physics=3, bl_pbl_physics=5))


@pytest.mark.parametrize("value", [4, -1, 5])
def test_out_of_schema_option_is_refused(value):
    with pytest.raises(ValueError, match="must be 0"):
        validate_urban_config(_cfg(sf_urban_physics=value))


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_a_non_integer_selector_is_refused(value):
    with pytest.raises(ValueError, match="integer"):
        validate_urban_config(_cfg(sf_urban_physics=value))


@pytest.mark.parametrize("option", [1, 2, 3])
def test_an_absent_model_is_not_in_this_build(option, monkeypatch):
    monkeypatch.setattr("woof.config.urban_model_in_build", lambda o: False)
    with pytest.raises(ValueError, match="not in this build"):
        validate_urban_config(_cfg(sf_urban_physics=option))


@pytest.mark.parametrize("option", [1, 2, 3])
@pytest.mark.parametrize("lsm", [0, 3])
def test_ruc_and_no_lsm_never_call_an_urban_model(all_models, option, lsm):
    with pytest.raises(ValueError, match="silently ignored"):
        validate_urban_config(_cfg(sf_urban_physics=option,
                                   sf_surface_physics=lsm))


@pytest.mark.parametrize("option", [2, 3])
@pytest.mark.parametrize("pbl", [0, 5, 11, 900])
def test_bep_needs_a_pbl_that_takes_its_sources(all_models, option, pbl):
    with pytest.raises(ValueError, match="PBL"):
        validate_urban_config(_cfg(sf_urban_physics=option,
                                   bl_pbl_physics=pbl))


@pytest.mark.parametrize("option,lsm,pbl", [
    (1, 2, 1), (1, 4, 5), (1, 2, 11), (2, 2, 1), (2, 4, 2), (3, 2, 2),
    (3, 4, 1)])
def test_the_admitted_pairings(all_models, option, lsm, pbl):
    validate_urban_config(_cfg(sf_urban_physics=option,
                               sf_surface_physics=lsm, bl_pbl_physics=pbl))


def test_companion_keys_are_checked_once_an_urban_model_runs(all_models):
    with pytest.raises(ValueError, match="use_wudapt_lcz"):
        validate_urban_config(_cfg(sf_urban_physics=1, use_wudapt_lcz=2))
    with pytest.raises(ValueError, match="num_urban_hi"):
        validate_urban_config(_cfg(sf_urban_physics=2, num_urban_hi=20))


def test_urban_model_in_build_reads_the_module_table():
    # No lane module ships on this branch yet; the answer is the import
    # system's, not a flag.
    for option, name in URBAN_MODEL_MODULES.items():
        import importlib.util
        assert urban_model_in_build(option) == (
            importlib.util.find_spec(name) is not None)
    assert urban_model_in_build(0) is False
    assert urban_model_in_build(9) is False


def test_the_fields_are_appended_last():
    # Appended right after adaptive_nest_lattice, so every earlier field keeps
    # its positional index.  The namelist-gaps merge appended slope_rad,
    # topo_shading and shadlen after them, the Noah mosaic trio followed,
    # development branch appended diff_opt and mix_full_fields
    # after the trio, development branch appended topo_wind and gwd_opt
    # after those, and development branch (37a99fcd4, merged at
    # dd5ded73f) appended zadvect_implicit_variant last, default "wrf_471";
    # tests/test_config_freeze.py pins the whole tail.
    names = [f.name for f in dataclasses.fields(RunConfig)]
    at = names.index("adaptive_nest_lattice")
    assert names[at + 1:at + 4] == ["sf_urban_physics", "use_wudapt_lcz",
                                    "num_urban_hi"]
    # 2.8.5 appended scalar_pblmix, use_rap_aero_icbc, the CLM lake
    # quartet and the SPP consumer pair after zadvect_implicit_variant.
    at = names.index("sf_surface_mosaic")
    assert names[at:at + 8] == ["sf_surface_mosaic", "mosaic_cat",
                                "mosaic_urban_canopy", "diff_opt",
                                "mix_full_fields", "topo_wind", "gwd_opt",
                                "zadvect_implicit_variant"]


def test_the_memory_checks_price_the_urban_arrays():
    """BEP+BEM's wall and floor stacks are thousands of words per column;
    an unpriced option 3 would pass admission and fail to allocate."""
    import numpy as np

    from woof.core.preflight import physics_array_shapes

    words = {}
    for option in (0, 1, 2, 3):
        cfg = _cfg(sf_urban_physics=option, nx=10, ny=10)
        shapes = physics_array_shapes(cfg)
        words[option] = sum(int(np.prod(v)) for k, v in shapes.items()
                            if k.startswith("fields/") and (
                                "urb" in k or "_bep" in k or "rural" in k))
    assert words[0] == 0
    per_column = {o: words[o] / 100 for o in (1, 2, 3)}
    assert per_column[1] < per_column[2] < per_column[3]
    # tw1/tw2_urb4d alone: 2 x ndm*nwr*nz*nbui = 2 x 2*10*18*15 words.
    assert per_column[3] > 2 * 2 * 10 * 18 * 15


@pytest.mark.parametrize("option,pbl,expected", [
    (1, 1, {"urban_ucm"}),
    (2, 1, {"urban_bep", "urban_bep_couple"}),
    (2, 2, {"urban_bep", "urban_bep_couple", "myjurb"}),
    (3, 1, {"urban_bem_composed", "urban_bep_couple"}),
    (3, 2, {"urban_bem_composed", "urban_bep_couple", "myjurb"}),
])
def test_the_memory_check_prices_the_urban_kernels(option, pbl, expected):
    """An urban run launches kernels the fit gate has to charge for.

    myjurb's 10,256 B frame is a launch-time reservation of (10,256 - 1,024)
    B per resident thread -- 1.7 GiB on an RTX 4090 -- and before the urban
    row existed in woof/core/preflight.py an MYJ + BEP run was admitted
    without it."""
    from datetime import datetime

    from woof.core import preflight as pf
    from woof.experiment import experiment_from_run_config

    run = _cfg(sf_urban_physics=option, bl_pbl_physics=pbl,
               sf_sfclay_physics=2 if pbl == 2 else 1)
    exp = experiment_from_run_config(run, datetime(2024, 7, 1, 18))
    dc = exp.domains[0]
    urban = pf.domain_kernel_modules(dc, prices_refl=False) - \
        pf.domain_kernel_modules(
            dataclasses.replace(dc, run=dataclasses.replace(
                run, sf_urban_physics=0)), prices_refl=False)
    assert urban == expected
    for module in urban:
        assert (module in pf.KERNEL_MAX_LOCAL_SIZE_BYTES
                or module in pf.CHAINED_TRANSLATION_UNIT_FRAMES), module


def test_the_memory_check_prices_the_column_workspaces():
    """BEP's and BEP+BEM's run-time column scratch is not in ``fields`` and
    runs to hundreds of MiB; the admission check has to see it."""
    from woof.core.urban_state import urban_array_shapes

    def mib(shapes, prefix):
        import math
        return sum(math.prod(v) for k, v in shapes.items()
                   if k.startswith(prefix)) * 4 / 2 ** 20

    bep = urban_array_shapes(_cfg(sf_urban_physics=2, nx=200, ny=200, nz=50))
    bem = urban_array_shapes(_cfg(sf_urban_physics=3, nx=200, ny=200, nz=50))
    assert 200 < mib(bep, "bep_column_workspace") <= 256
    assert mib(bep, "bep_class_scratch") == 11
    assert 1000 < mib(bem, "bem_column_workspace") <= 1024
    assert urban_array_shapes(_cfg(sf_urban_physics=1)).keys().isdisjoint(
        {"bep_column_workspace", "bem_column_workspace"})


# --- the UCM's first-level fatal, refused at plan time ----------------------

#: The 59-level ladder of the 750 m San Francisco Bay and Los Angeles sites
#: (the 2026-09-30 urban runs): the first mass level is about 25 m up.
SITE_ETA = (
    1, 0.993814707, 0.985950649, 0.976014256, 0.963557541,
    0.948093116, 0.929123759, 0.90619123, 0.87894237, 0.847207963,
    0.811077714, 0.770949006, 0.727525413, 0.684030771, 0.642961025,
    0.604180932, 0.567562938, 0.532986403, 0.500337601, 0.469508916,
    0.440399021, 0.412912011, 0.386957437, 0.362449884, 0.339308649,
    0.317457527, 0.296824664, 0.277342081, 0.258945674, 0.241574913,
    0.225172549, 0.2096847, 0.195060253, 0.181251153, 0.168211967,
    0.155899644, 0.144273847, 0.133296132, 0.122930467, 0.113142714,
    0.103900604, 0.095173724, 0.0869334266, 0.0791524947, 0.0718053728,
    0.0648678541, 0.0583171472, 0.0521316081, 0.0462909527, 0.0407758839,
    0.0355683193, 0.030651059, 0.026007941, 0.0216237046, 0.0174838807,
    0.0135748768, 0.00988376327, 0.00639845803, 0.00310745789, 0,
)


def _tree(option, lcz, eta=SITE_ETA):
    run = types.SimpleNamespace(sf_urban_physics=option, use_wudapt_lcz=lcz,
                                eta_levels=None, base_temp=290.0)
    vertical = types.SimpleNamespace(eta_levels=tuple(float(v) for v in eta),
                                     p_top=5000.0, hybrid_opt=2, etac=0.2)
    return types.SimpleNamespace(
        vertical=vertical,
        domains=(types.SimpleNamespace(grid_id=1, run=run),
                 types.SimpleNamespace(grid_id=2, run=run)))


def test_the_ucm_with_lcz_high_rise_on_a_25_m_first_level_is_refused_at_plan():
    """The breakage: WRF's own FATAL_ERROR at module_sf_urban.F:825, which
    stopped both 750 m site runs at step 1 after their fetch and
    preparation.  The refusal names it, the classes, and the remedies."""
    from woof.experiment import ucm_first_level_refusal

    text = ucm_first_level_refusal(_tree(1, 1), source="site.toml")
    assert text is not None
    assert "module_sf_urban.F:825" in text and "FATAL_ERROR" in text
    assert "LCZ 1 (compact high-rise, 33.9 m)" in text
    assert "LCZ 4 (open high-rise, 31.5 m)" in text
    assert "LCZ 2 " not in text and "LCZ 5 " not in text
    assert "25.0 m above the ground" in text
    assert "NLCD with use_wudapt_lcz = 0" in text
    assert "eta_levels[1] = 0.9907 or lower (now 0.9938)" in text
    assert "sf_urban_physics = 2 or 3" in text


@pytest.mark.parametrize("option,lcz", [(1, 0), (2, 1), (3, 1), (0, 1)])
def test_the_same_grid_runs_everything_that_has_no_such_limit(option, lcz):
    """NLCD's URBPARM classes (tallest canopy 10.2 m) clear the 25 m level,
    and BEP/BEP+BEM spread the buildings over the levels."""
    from woof.experiment import ucm_first_level_refusal

    assert ucm_first_level_refusal(_tree(option, lcz)) is None


def test_the_remedy_the_refusal_prints_clears_the_check():
    from woof.experiment import ucm_first_level_refusal

    eta = list(SITE_ETA)
    eta[1] = 0.9907
    assert ucm_first_level_refusal(_tree(1, 1, eta)) is None


def test_a_marginal_grid_is_left_to_the_exact_step_one_check():
    """A first level just under the tallest canopy in the base state but
    above it in a warm column is not refused: the plan refuses only what
    no plausible column can run, and the kernel's step-1 stop stays exact."""
    from woof.core.urban_tables import ucm_canopy_heights
    from woof.experiment import _first_layer_depth, ucm_first_level_refusal

    canopy = max(ucm_canopy_heights(1).values())
    vertical = _tree(1, 1).vertical
    lo, hi = 0.9, 1.0
    for _ in range(60):   # base-state first level 1 m below the canopy
        mid = 0.5 * (lo + hi)
        if 0.5 * _first_layer_depth(mid, vertical, 290.0) >= canopy - 1.0:
            lo = mid
        else:
            hi = mid
    eta = list(SITE_ETA)
    eta[1] = lo
    assert ucm_first_level_refusal(_tree(1, 1, eta)) is None


def test_every_front_door_meets_the_check_at_the_shared_load():
    import inspect

    from woof import experiment

    source = inspect.getsource(experiment.build_experiment)
    assert "ucm_first_level_refusal(experiment, source=source)" in source
