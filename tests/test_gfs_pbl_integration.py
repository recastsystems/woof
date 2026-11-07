"""The GFS nonlocal PBL (bl_pbl_physics=3): its doors and its seams.

No device is opened here.  What is pinned:

* the three doors (the RunConfig key, the WRF namelist importer, the
  registry option) and the surface-layer admission, each refusal naming
  its breakage;
* the dispatch row, the checkpoint identity, the vertical bound and the
  workspace geometry, each held equal to the one authority it restates;
* the PBL driver's scalar words, spelled as WRF's binary32 fold;
* default-off: nothing GFS-specific is reached unless 3 is selected.

The kernel against WRF's words is tests/test_gfs_pbl_wrf461_parity.py
(CPU reference) and tests/test_gfs_pbl_wrf461_gpu.py (card).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.config import (GFS_PBL_SCHEME, GFS_PBL_SURFACE_LAYERS, RunConfig,
                          validate_gfs_pbl_config, validate_run_config)

ROOT = Path(__file__).resolve().parents[1]


def _cfg(**overrides) -> RunConfig:
    base = dict(nx=16, ny=16, nz=30, dx=3000.0, dy=3000.0, dt=15.0,
                ztop=15000.0, run_seconds=60.0, moist=True, mp_physics=8,
                bl_pbl_physics=GFS_PBL_SCHEME,
                sf_sfclay_physics=3, sf_surface_physics=2)
    base.update(overrides)
    return RunConfig(**base)


def test_the_gfs_pbl_is_admitted_with_its_native_surface_layer():
    assert GFS_PBL_SCHEME == 3
    assert GFS_PBL_SURFACE_LAYERS == (3,)
    for sfclay in GFS_PBL_SURFACE_LAYERS:
        validate_run_config(_cfg(sf_sfclay_physics=sfclay))
    # A positive PBL cadence is WRF's bldt: the scheme holds nothing
    # outside the tendency manifest, so it is admitted.
    validate_run_config(_cfg(bldt=5.0))
    validate_run_config(_cfg(moist=False, mp_physics=0))


@pytest.mark.parametrize("sfclay", [0, 1, 5, 91])
def test_a_layer_without_the_six_words_is_refused_by_name(sfclay):
    with pytest.raises(ValueError) as caught:
        validate_run_config(_cfg(sf_sfclay_physics=sfclay,
                                 sf_surface_physics=0 if sfclay == 0 else 2))
    message = str(caught.value)
    assert "GZ1OZ0" in message and "bl_pbl_physics=3 (GFS)" in message
    assert "+Inf" in message


def test_the_eta_layer_is_refused():
    with pytest.raises(ValueError):
        validate_run_config(_cfg(sf_sfclay_physics=2))


def test_bep_and_bem_urban_are_wrfs_own_fatal():
    for urban in (2, 3):
        cfg = SimpleNamespace(bl_pbl_physics=GFS_PBL_SCHEME,
                              sf_sfclay_physics=3, sf_urban_physics=urban)
        with pytest.raises(ValueError) as caught:
            validate_gfs_pbl_config(cfg)
        assert "module_physics_init.F:3744-3745" in str(caught.value)
    validate_gfs_pbl_config(SimpleNamespace(
        bl_pbl_physics=GFS_PBL_SCHEME, sf_sfclay_physics=3,
        sf_urban_physics=1))


def test_dispatch_routes_three_to_its_own_runner():
    from woof.core.physics import PHYSICS_SLOT_DISPATCH, PhysicsDriver
    assert PHYSICS_SLOT_DISPATCH["bl_pbl_physics"][GFS_PBL_SCHEME] == \
        "_run_gfs_pbl"
    assert callable(getattr(PhysicsDriver, "_run_gfs_pbl"))


def test_the_checkpoint_identity_names_the_wrf_version_ported():
    from woof.checkpoint_identity import PBL_ALGORITHM_IDENTITIES
    assert PBL_ALGORITHM_IDENTITIES[GFS_PBL_SCHEME] == "gfs-pbl-wrf-v4.6.1-v2-fast-default"


def test_the_registry_option_is_the_selector():
    registry = json.loads((ROOT / "woof/physics_registry_v2.json")
                          .read_text(encoding="utf-8"))
    option = registry["components"]["pbl"]["options"]["gfs"]
    assert option["selectors"] == {"bl_pbl_physics": GFS_PBL_SCHEME}
    assert option["implemented"] is True
    assert option["constraints"]["requires_components"]["surface_layer"] \
        == ["gfs"]
    assert option["consumers"]["restart_algorithm_identity"] == \
        "gfs-pbl-wrf-v4.6.1-v2-fast-default"
    assert option["maturity"] == "implemented-unverified"


def test_the_namelist_importer_carries_three_natively():
    from woof.namelist_import import _BL_MAP
    assert _BL_MAP[3][0] == GFS_PBL_SCHEME


def test_the_launcher_bound_is_the_contract_bound():
    source = (ROOT / "woof/core/gfs_pbl.py").read_text(encoding="utf-8")
    assert "VERTICAL_LEVEL_BOUNDS = GFS_PBL_VERTICAL_LEVEL_BOUNDS" in source
    from woof.physics_vertical_contract import GFS_PBL_VERTICAL_LEVEL_BOUNDS
    assert GFS_PBL_VERTICAL_LEVEL_BOUNDS == (4, None)


def test_the_workspace_geometry_is_the_kernels():
    from woof.core.physics_inventory import (
        GFS_PBL_BLOCK, GFS_PBL_WS_SLOTS, gfs_pbl_chunk_columns,
        gfs_pbl_workspace_bytes)
    source = (ROOT / "woof/core/kernels/gfs_pbl.cu").read_text("utf-8")
    assert int(re.search(r"#define GFS_WS_SLOTS (\d+)", source).group(1)) \
        == GFS_PBL_WS_SLOTS
    assert gfs_pbl_chunk_columns(50, 10) == 10
    assert gfs_pbl_chunk_columns(50, 10 ** 6, budget_bytes=1) == 1
    chunk = gfs_pbl_chunk_columns(50, 10 ** 6)
    assert chunk % GFS_PBL_BLOCK == 0
    assert gfs_pbl_workspace_bytes(50, 10 ** 6) <= 256 * 1024 * 1024


def test_the_pbl_driver_words_are_wrfs_binary32_fold():
    """CP, G, ROVCP, R, ROVG, EP1, KARMAN as module_model_constants.F folds
    them (cp = 7.*r_d/2., rcp = r_d/cp, rovg = r_d/g, ep_1 = r_v/r_d-1.);
    the oracle's grade is the end-to-end proof, this pins the words."""
    from woof.core.pbl_driver_constants import pbl_driver_constants_f32
    words = [np.float32(w).view(np.uint32) for w in pbl_driver_constants_f32()]
    f = np.float32
    want = [f(1004.5), f(9.81), f(f(287.0) / f(1004.5)), f(287.0),
            f(f(287.0) / f(9.81)), f(f(f(461.6) / f(287.0)) - f(1.0)),
            f(0.4)]
    assert words == [w.view(np.uint32) for w in want]


def test_default_off_reaches_no_gfs_seam():
    """The runner is reached only through the dispatch row, and the
    preflight prices the workspace only for selector 3."""
    physics = (ROOT / "woof/core/physics.py").read_text(encoding="utf-8")
    assert physics.count("gfs_pbl_step(") == 1
    assert physics.count("from woof.core.gfs_pbl import") == 1
    from woof.core.preflight import _PBL_KERNEL_MODULES
    assert _PBL_KERNEL_MODULES[GFS_PBL_SCHEME] == ("gfs_pbl",)
    assert all("gfs_pbl" not in v for k, v in _PBL_KERNEL_MODULES.items()
               if k != GFS_PBL_SCHEME)


def test_launcher_uses_the_shared_no_ftz_route():
    from woof.core.kernels import module_options
    from woof.numerics import mode
    expected = ("-std=c++17", "--ftz=false")
    if mode() == "strict":
        expected += ("--fmad=false", "-DGPUWM_WRF_EXACT=1")
    assert module_options("gfs_pbl") == expected
    source = (ROOT / "woof/core/gfs_pbl.py").read_text("utf-8")
    assert 'return load_module("gfs_pbl")' in source
    assert "compile_using_nvrtc" not in source


def test_gfs_mass_heights_are_rebuilt_at_restart():
    from woof.io.restart import classify_scratch_slot
    assert classify_scratch_slot("gfs_pbl_z") == "rebuild"
