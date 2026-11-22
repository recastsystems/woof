"""WRF 4.8.0 MYNN-EDMF (8f8e8b59) route: admission, isolation and image binding.

The column itself is graded word for word against native WRF 4.8.0 strict by
tools/mynn_wrf480_port/replay.py (woof/data/mynn480/README.md); these tests
pin what a CPU run can check: the default and wrf_471 doors never reach the
4.8.0 unit, unqualified settings are refused by name, and the shipped images
bind the exact source the route compiles.
"""
from dataclasses import replace
import hashlib
import json
import sys

import pytest

from woof.config import ConfigurationRefused, RunConfig, validate_run_config


def config(**kw):
    settings = dict(nx=8, ny=8, nz=49, dx=3000., dy=3000., ztop=20000.,
                    dt=12., run_seconds=120., moist=True, mp_physics=8,
                    bl_pbl_physics=5, sf_sfclay_physics=1)
    settings.update(kw)
    return RunConfig(**settings)


def test_wrf480_default_identity_validates():
    cfg = config(bl_mynn_version="wrf_480")
    validate_run_config(cfg)
    assert cfg.bl_mynn_version == "wrf_480"


def test_default_and_471_doors_do_not_select_480():
    assert config().bl_mynn_version == "wrf_461"
    assert config(bl_mynn_edmf=0).bl_mynn_version == "wrf_471"


@pytest.mark.parametrize("key,value", [
    ("bl_mynn_edmf", 0), ("bl_mynn_cloudpdf", 1), ("bl_mynn_closure", 2.5),
    ("bl_mynn_mixlength", 2), ("tke_budget", 1), ("bl_mynn_output", 1),
])
def test_wrf480_refuses_unqualified_settings(key, value):
    with pytest.raises(ConfigurationRefused, match="wrf_480"):
        validate_run_config(config(bl_mynn_version="wrf_480", **{key: value}))


def test_wrf480_refuses_mixscalars_key():
    # mp_physics=6 has no number species, so only the 4.8.0 meaning refuses.
    with pytest.raises(ConfigurationRefused, match="wrf_480"):
        validate_run_config(config(bl_mynn_version="wrf_480", mp_physics=6,
                                   bl_mynn_mixscalars=1))


def test_wrf480_refuses_aerosol_microphysics_until_staged():
    with pytest.raises(ConfigurationRefused, match="aerosol"):
        validate_run_config(config(bl_mynn_version="wrf_480", mp_physics=28))


def test_selector_comment_carries_wrf480():
    from woof.physics_source_defaults import read_physics_selector_comment
    text = '! gpuwm-physics-selectors-v1: {"bl_mynn_version":"wrf_480"}\n&physics\n/\n'
    assert read_physics_selector_comment(text) == {"bl_mynn_version": "wrf_480"}


def test_vertical_schema_has_aligned_nonoverlapping_argument_slabs():
    from woof.core.mynn480_gpu import schema
    for nz in (20, 49, 50, 64, 128):
        meta = schema(nz)
        assert meta["scratch_bytes"] % 4 == 0
        for slab, count in meta["slab_fields"].items():
            ranges = sorted((f["offset"], f["offset"] + f["count"])
                            for f in meta["arguments"].values() if f.get("slab") == slab)
            assert ranges[0][0] == 0
            assert ranges[-1][1] == count
            assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))


def test_schema_binds_the_pinned_native_source():
    from woof.core.mynn480_gpu import DATA, SOURCE_SHA256
    for name in ("schema49.json", "schema50.json"):
        meta = json.loads((DATA / name).read_text())
        assert meta["source_sha256"] == SOURCE_SHA256["module_bl_mynnedmf.F90"]


def test_prebuilt_images_bind_the_route_source():
    from woof.core.mynn480_gpu import DATA, source
    receipts = sorted((DATA / "images").glob("*.json"))
    assert {r.name.split("-")[0] for r in receipts} >= {"sm_89", "sm_120"}
    for receipt in receipts:
        manifest = json.loads(receipt.read_text())
        # Images are per level count (the source bakes nz in), so each
        # binds source(nz) for its own nz, not source(49).
        nz = int(receipt.name.split("-")[1].removeprefix("nz"))
        digest = hashlib.sha256(source(nz).encode()).hexdigest()
        assert manifest["source_sha256"] == digest
        image = receipt.with_suffix(".cubin")
        assert hashlib.sha256(image.read_bytes()).hexdigest() == manifest["image_sha256"]


def test_wrf480_history_carries_the_new_edmf_fields_and_default_does_not():
    from woof.io.history_layout import produced_history_shapes
    new = {"EXCESS_H", "EXCESS_Q", "MAXMF_DD", "MAXWIDTH_DD", "MAXTKEPROD",
           "CLDTOP_COOLING", "ENT_EFF"}
    with480 = produced_history_shapes(config(bl_mynn_version="wrf_480"))
    default = produced_history_shapes(config())
    assert set(with480) - set(default) == new
    assert all(with480[name] == (8, 8) for name in new)


def test_wrf461_scratch_and_diagnostics_do_not_import_480():
    sys.modules.pop("woof.core.mynn480_gpu", None)
    from woof.core.mynn_pbl_scratch import mynn_pbl_scratch_shapes
    mynn_pbl_scratch_shapes(64, 49, bl_mynn_version="wrf_461")
    assert "woof.core.mynn480_gpu" not in sys.modules
    shapes = mynn_pbl_scratch_shapes(64, 49, bl_mynn_version="wrf_480")
    assert "mynn480_work" in shapes and "mynn471_work" not in shapes
