"""WRF 4.6.1 upper damping (damp_opt=3 inside advance_w, and w_damp).

``tests/data/upper_damping_wrf461.npz`` holds the 96 regime columns of
``tools/upper_damping_wrf461_oracle/columns.py`` (unity map and map
factors) and the words WRF 4.6.1's own compiled ``advance_w`` and
``w_damp`` wrote for them (gfortran 13.3.0 -O0, glibc 2.39 scalar sinf;
the WRF stock-flag build writes the same words; see the tool README).
Under the strict build (GPUWM_WRF_EXACT=1) WOOF must reproduce every w
and phi word of every damped, undamped, lid and dry case, and every
rw_tend word of w_damp, bit for bit.

The coefficient test is CPU only: WRF forms ``dampmag = dts*dampcoef`` as
one float32 product, and the double-precision product WOOF used to round
once is a different word for dtau 4.5 s (an 18 s step with 4 sound steps).
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "data" / "upper_damping_wrf461.npz"
MAPS = ("unity_map", "map_factors")


def test_damp_magnitude_is_wrfs_float32_product():
    from woof.core.acoustic import damp_magnitude
    on = SimpleNamespace(damp_opt=3, dampcoef=0.2)
    # dtau 4.5 s, dampcoef 0.2: WRF's float32 product is 0.900000036; the
    # double product rounded once is 0.899999976, one word lower.
    got = damp_magnitude(on, 4.5)
    assert got.dtype == np.float32
    assert got.view(np.uint32) == np.float32(0.90000004).view(np.uint32)
    assert got != np.float32(4.5 * 0.2)
    rng = np.random.default_rng(0)
    for dtau, coef in zip(rng.uniform(0.25, 12.0, 400), rng.uniform(0.01, 1.0, 400)):
        want = np.float32(dtau) * np.float32(coef)        # numpy: one rounding
        got = damp_magnitude(SimpleNamespace(damp_opt=3, dampcoef=float(coef)), float(dtau))
        assert got.view(np.uint32) == want.view(np.uint32)
    assert damp_magnitude(SimpleNamespace(damp_opt=0, dampcoef=0.2), 4.5) == 0
    assert damp_magnitude(SimpleNamespace(damp_opt=2, dampcoef=0.2), 4.5) == 0


def test_every_damping_coefficient_site_uses_the_float32_product():
    """The single and ensemble launchers both take damp_magnitude."""
    for rel in ("woof/core/acoustic.py", "woof/ensemble/batch_acoustic.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "dtau * cfg.dampcoef" not in text, rel
        assert "damp_magnitude(cfg, dtau)" in text, rel


def test_regime_columns_cover_the_stated_set():
    from tools.upper_damping_wrf461_oracle import columns
    raw = columns.build_raw(ROOT)
    ny, nx = raw["MU"].shape
    assert ny * nx >= 64
    assert set(columns.REGIMES) >= {"convective", "stable_night", "snow_ice", "ocean",
                                    "high_terrain", "edge_cases"}
    assert np.all(raw["HGT"][4] == 0)                       # ocean row
    assert raw["HGT"][5:7].max() >= 4000                    # high terrain
    assert raw["QSNOW"][:, 3].max() > 1e-3                  # snow row
    w = raw["W"]
    assert np.all((w[:, 7, 1] == 0) & np.signbit(w[:, 7, 1]))  # signed zeros
    zdamp, level = columns.edge_zdamp(raw)
    h = columns.column_heights(raw)
    j, i = columns.EDGE_COLUMN
    assert np.float32(h[-1, j, i] - np.float32(zdamp)) == h[level, j, i]
    # The HRRR layer (zdamp 5000) reaches into every column.
    assert columns.damping_layer_levels(raw, 5000.0).min() >= 5


def _fixture():
    if not FIXTURE.exists():
        pytest.skip("the upper-damping oracle fixture is not in this tree")
    with np.load(FIXTURE, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _raw(fx, map_name):
    prefix = f"raw/{map_name}/"
    return {key[len(prefix):]: value.copy() for key, value in fx.items()
            if key.startswith(prefix)}


@requires_gpu
def test_strict_advance_w_damper_matches_wrf_461_word_for_word():
    if os.environ.get("GPUWM_WRF_EXACT") != "1":
        pytest.skip("the 0 ULP standard is the strict build (GPUWM_WRF_EXACT=1)")
    import cupy  # noqa: F401  (device test: the conftest marker reads this import)
    from tools.smallstep_wrf471_oracle import vertical_workspace
    from tools.upper_damping_wrf461_oracle import columns, run_oracle
    module, workspace = run_oracle._vertical_module()
    fx = _fixture()
    checked = 0
    for map_name in MAPS:
        raw = _raw(fx, map_name)
        edge, _ = columns.edge_zdamp(raw)
        for case, metadata in run_oracle.advance_w_cases(edge):
            with patch.object(vertical_workspace, "workspace_source", workspace):
                got = module.vertical_port_outputs(raw, metadata)
            for field in ("w", "ph"):
                want = fx[f"{map_name}/{case}/{field}_wrf"]
                differ = int((got[field].view(np.uint32) != want.view(np.uint32)).sum())
                assert differ == 0, (map_name, case, field, differ)
                checked += 1
    assert checked == 2 * 2 * 8


@requires_gpu
def test_w_damp_matches_wrf_461_word_for_word():
    """w_damp's kernel fixes every rounding point, so both builds must match."""
    import cupy  # noqa: F401  (device test: the conftest marker reads this import)
    from tools.upper_damping_wrf461_oracle import run_oracle
    fx = _fixture()
    for map_name in MAPS:
        raw = _raw(fx, map_name)
        for dt in (run_oracle.DT_HRRR, 30.0):
            inp = dict(ww=fx[f"w_damp/{map_name}/dt{int(dt)}/ww"],
                       rw_t=fx[f"w_damp/{map_name}/dt{int(dt)}/rw_t"], w=raw["W"],
                       mub=raw["MUB"], mup=raw["MU"], u=raw["U"], v=raw["V"],
                       msfu=raw["MAPFAC_U"], msfv=raw["MAPFAC_V"], c1f=raw["C1F"],
                       c2f=raw["C2F"], rdnw=raw["RDNW"])
            for case, crit, ieva in run_oracle.W_DAMP_CASES:
                key = f"w_damp/{map_name}/dt{int(dt)}/{case}"
                rw, vmax, hmax, _ = run_oracle._woof_w_damp(inp, dt, 3000.0, 3000.0, crit, ieva)
                want = fx[key + "/rw_wrf"]
                assert np.array_equal(rw.view(np.uint32), want.view(np.uint32)), key
                cfl = np.array([vmax, hmax], np.float32)
                assert np.array_equal(cfl.view(np.uint32),
                                      fx[key + "/max_cfl_wrf"].view(np.uint32)), key
