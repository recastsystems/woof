"""PX surface flux state survives the real engine restart path byte for byte."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import zipfile

import pytest

from conftest import requires_gpu
from woof.io import restart


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("pbl,sfclay", [(0, 7), (7, 7), (7, 1), (7, 91)])
def test_px_surface_restart_roundtrip_and_continuation(tmp_path, pbl, sfclay):
    import cupy as cp
    from woof.core.dycore import run_steps
    from test_restart import _physics_state, _assert_restart_equal

    def make():
        return _physics_state(cp, sf_sfclay_physics=sfclay, bl_pbl_physics=pbl,
                              cu_physics=0, mp_physics=0)

    reference, cfg, driver = make()
    run_steps(reference, cfg, 12)
    full = restart.write_restart(tmp_path / "full.npz", reference, cfg)

    split, cfg, driver = make()
    run_steps(split, cfg, 6)
    mid = restart.write_restart(tmp_path / "mid.npz", split, cfg)
    fresh, cfg, fresh_driver = make()
    info = restart.restore_restart(mid, fresh, cfg)
    assert info.elapsed_seconds == 60.0
    roundtrip = restart.write_restart(tmp_path / "roundtrip.npz", fresh, cfg)
    _assert_restart_equal(mid, roundtrip)
    run_steps(fresh, cfg, 6)
    final = restart.write_restart(tmp_path / "final.npz", fresh, cfg)
    _assert_restart_equal(full, final)
    assert bool(cp.any(fresh_driver.fields["ust"] > 0))
    assert bool(cp.any(fresh_driver.fields["hfx"] != 0))
    assert bool(cp.any(fresh_driver.fields["chs2"] > 0))
    with zipfile.ZipFile(full) as archive:
        hashes = {name: hashlib.sha256(archive.read(name)).hexdigest()
                  for name in sorted(archive.namelist())}
    report = {"steps": 12, "split_after_steps": 6,
              "mode": "strict" if os.environ.get("GPUWM_WRF_EXACT") == "1" else "default",
              "roundtrip": "byte-identical array payloads",
              "continuation": "byte-identical array payloads",
              "sf_sfclay_physics": sfclay, "sf_surface_physics": 2, "bl_pbl_physics": pbl,
              "full_restart_members_sha256": hashes}
    target = os.environ.get("PXSFCLAY_PARITY_RECEIPT_DIR")
    if target:
        (Path(target) / f"restart.pbl{pbl}.sf{sfclay}.{report['mode']}.json").write_text(
            json.dumps(report, indent=2) + "\n")
    for path in (full, mid, roundtrip, final):
        size = path.stat().st_size
        path.unlink()
        if target:
            with (Path(target) / "deleted-model-files.log").open("a") as log:
                log.write(f"{path} {size} bytes\n")
