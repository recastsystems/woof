"""The cold preparation cache must be populated beside fetch, not at the root."""
import os
import subprocess
import sys

import pytest

from conftest import requires_gpu


def test_offline_prewarm_keeps_chemistry_without_external_source_binding():
    from woof.config import RunConfig
    from woof.ingest.preparation_prewarm import synthetic_preparation
    run = RunConfig(nx=33, ny=32, nz=7, dx=12000., dy=12000.,
                    ztop=20000., dt=30., run_seconds=60.,
                    chem_sets="gocart_primary", chem_sources="geos-fp",
                    aer_ra_feedback=1, ra_lw_physics=4, ra_sw_physics=4)
    source, grid, cfg = synthetic_preparation(run)
    assert cfg.chem_sources == ""
    assert cfg.chem_sets == run.chem_sets
    assert cfg.aer_ra_feedback == 1
    assert run.chem_sources == "geos-fp"
    assert source.valid_time.year == 2000


@requires_gpu
@pytest.mark.gpu
def test_preparation_prewarm_caches_root_initialize_kernels(tmp_path):
    cache = tmp_path / "cupy"
    cache.mkdir()
    env = dict(os.environ, CUPY_CACHE_DIR=str(cache))
    script = """
from concurrent.futures import ThreadPoolExecutor
from woof.ingest.preparation_prewarm import preparation_jobs, warm_preparation_job
jobs = preparation_jobs()
with ThreadPoolExecutor(4) as pool:
    list(pool.map(warm_preparation_job, [options for _, options in jobs]))
"""
    assert not list(cache.glob("*.cubin"))
    subprocess.run([sys.executable, "-c", script], env=env, check=True)
    warmed = {path.name for path in cache.glob("*.cubin")}
    assert warmed, "real preparation paths compiled no kernels into the fresh cache"
    # New interpreter: no CuPy memoized function/module can serve this root.
    subprocess.run([sys.executable, "-c", script], env=env, check=True)
    after = {path.name for path in cache.glob("*.cubin")}
    assert after == warmed, sorted(after - warmed)
    print(f"preparation cubins: 0 -> {len(warmed)}; second root_initialize: +{len(after - warmed)}")


def test_preparation_prewarm_needs_no_wif_dataset_for_an_mp28_hrrr_run(tmp_path, monkeypatch):
    """The pre-warm runs before fetch stages the WIF climatology.

    woof-1.1:r11 run_62bd631d2ba8284c: every preparation job of an mp28
    HRRR run (aer_init_opt=1, wif_input_opt=1, specified) refused with the
    mp28 lateral-forcing precondition on a fresh container.
    """
    from woof.config import RunConfig, mp28_aerosol_lateral_forcing_precondition
    from woof.ingest.preparation_prewarm import synthetic_preparation

    for name in ("WOOF_WIF_CLIMATOLOGY", "WOOF_WIF_CLIMATOLOGY_ROOT", "WOOF_WIF_DATA_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    run = RunConfig(nx=33, ny=32, nz=7, dx=3000.0, dy=3000.0, ztop=20000.0, dt=20.0,
                    run_seconds=60.0, mp_physics=28, aer_init_opt=1, wif_input_opt=1,
                    specified=True)
    assert mp28_aerosol_lateral_forcing_precondition(run)  # the run itself still asks
    _, _, cfg = synthetic_preparation(run)
    assert cfg.mp_physics == 28
    assert mp28_aerosol_lateral_forcing_precondition(cfg) is None


def _packaged_recipe_runs():
    from importlib.resources import files

    from woof.experiment import load_experiment

    recipes = files("woof") / "data" / "recipes"
    for entry in sorted(recipes.iterdir(), key=lambda item: item.name):
        if entry.name.endswith(".toml"):
            yield entry.name, load_experiment(str(entry)).domains[0].run


def test_probe_column_is_valid_input_for_every_shipped_recipe(tmp_path, monkeypatch):
    """The probe carries the aerosol pair an aerosol-aware recipe reads.

    test host 8x5090, 2026-10-21, packaged hrrr_v4_gsd41 (mp_physics = 28,
    use_rap_aero_icbc = true): every preparation pre-warm job refused with
    "analyzed aerosol IC/BC requires QNWFA and QNIFA on every initial and
    boundary frame" and the preparation compiled cold.  The floor in
    initialize_real is asked here through its own helpers, with no WIF
    dataset installed.
    """
    import numpy as np

    from woof.config import run_preparation_preconditions
    from woof.ingest.analyzed_numbers import (
        AEROSOL_NUMBER_FIELDS, METGRID_NUMBER_FIELDS, analyzed_aerosol_fields,
        metgrid_number_targets)
    from woof.ingest.preparation_prewarm import synthetic_preparation
    from woof.ingest.real import _resolved_mp28_aerosol_source

    for name in ("WOOF_WIF_CLIMATOLOGY", "WOOF_WIF_CLIMATOLOGY_ROOT", "WOOF_WIF_DATA_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    recipes = dict(_packaged_recipe_runs())
    assert "hrrr_v4_gsd41.toml" in recipes
    for recipe, run in recipes.items():
        for specific in (True, False):
            source, _grid, cfg = synthetic_preparation(run, specific=specific)
            assert run_preparation_preconditions(cfg) == (), recipe
            if int(cfg.mp_physics) != 28:
                continue
            decoded = tuple(name for name in METGRID_NUMBER_FIELDS if name in source.fields)
            carried = analyzed_aerosol_fields(decoded, metgrid_number_targets(cfg))
            assert _resolved_mp28_aerosol_source(cfg) == "analysis", recipe
            assert set(carried) == set(AEROSOL_NUMBER_FIELDS), (recipe, carried)
            for name in AEROSOL_NUMBER_FIELDS:
                values = source.fields[name]
                assert values.shape == source.fields["T"].shape
                assert np.isfinite(values).all() and (values > 0).all(), (recipe, name)
    # The run itself keeps its own selectors; only the probe changes.
    hrrr = recipes["hrrr_v4_gsd41.toml"]
    assert hrrr.use_rap_aero_icbc and _resolved_mp28_aerosol_source(hrrr) == "analysis"


def test_probe_aerosol_profile_is_thompson_init_climatology():
    import numpy as np

    from woof.core.thompson_aerosol_state import NA_CCN0, NA_CCN1, NA_IN0, NA_IN1
    from woof.ingest.preparation_prewarm import _aerosol_number_profile

    height = np.array([0.0, 1000.0, 5000.0, 15000.0])[:, None, None] * np.ones((1, 2, 2))
    profile = _aerosol_number_profile(height)
    assert np.allclose(profile["QNWFA"][0], NA_CCN1 + NA_CCN0)
    assert np.allclose(profile["QNIFA"][0], NA_IN1 * 0.01 + NA_IN0)
    assert (np.diff(profile["QNWFA"], axis=0) < 0).all()
    assert (np.diff(profile["QNIFA"], axis=0) < 0).all()
