"""Shortwave resource accounting and source-bound proof guards."""
from datetime import datetime

import pytest
from conftest import requires_gpu

from woof.config import RunConfig
from woof.experiment import experiment_from_run_config
from woof.core import preflight as pf
from woof.core import kernel_frame_recordings as recordings


def test_gsfc_selector_prices_its_launched_module():
    run = RunConfig(nx=8, ny=8, nz=4, dx=1000.0, dy=1000.0,
                    ztop=10000.0, dt=1.0, run_seconds=10.0,
                    ra_physics=0, ra_lw_physics=0, ra_sw_physics=2)
    domain = experiment_from_run_config(run, datetime(2024, 1, 1)).domains[0]
    assert "gsfcsw" in pf.domain_kernel_modules(domain, prices_refl=False)
    assert "gsfcsw" not in pf.assumed_bound_modules({"gsfcsw"})


def test_gsfc_frame_ceiling_has_a_measured_recording():
    rows = [row for row in recordings.KERNEL_LOCAL_FRAME_RECORDINGS
            if "gsfcsw" in row.frames]
    assert rows
    assert pf.KERNEL_MAX_LOCAL_SIZE_BYTES["gsfcsw"] == recordings.frame_ceiling()["gsfcsw"]
    assert all("gsfcsw" in row.frames for row in
               recordings.KERNEL_LOCAL_FRAME_RECORDINGS if row.complete)


@pytest.mark.gpu
@requires_gpu
def test_gsfc_frame_price_covers_the_production_driver_reading():
    from woof.core.kernels import load_module, kernel_local_size_bytes
    from woof.certify.compile_platform import compile_platform_fingerprint
    measured = kernel_local_size_bytes(
        load_module("gsfcsw").get_function("gsfcsw_columns_kernel"))
    assert measured <= pf.KERNEL_MAX_LOCAL_SIZE_BYTES["gsfcsw"]
    row = recordings.recording_for(compile_platform_fingerprint())
    if row is not None and "gsfcsw" in row.frames:
        from woof.numerics import is_strict
        if is_strict() and row.strict_recording is not None:
            row = row.strict_recording
        assert measured == row.frames["gsfcsw"]
