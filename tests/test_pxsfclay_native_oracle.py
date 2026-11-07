"""The PX proof fixture was captured inside native WRF, not constructed."""
import os

import pytest

from conftest import requires_gpu
from woof.verify.pxsfclay_native_oracle import (
    FIXTURE_DIR, fixture_manifest, read_capture, replay)


def test_native_capture_inventory_and_provenance():
    manifest = fixture_manifest()
    files = [name for name in manifest["sha256"] if name.endswith(".bin.gz")]
    assert len(files) == 12
    assert manifest["origin"] == "real WRF forecast, instrumented PXSFCLAY driver"
    records = [read_capture(FIXTURE_DIR / name)[0] for name in files]
    assert {record["itimestep"] for record in records} == {1, 2, 3}
    assert sum(record["columns"] * 24 for record in records) == 43200


@pytest.mark.gpu
@requires_gpu
def test_real_native_input_replay_is_bitwise():
    if os.environ.get("GPUWM_WRF_EXACT") != "1":
        pytest.skip("native replay requires GPUWM_WRF_EXACT=1")
    from woof.wrf_exact import referee_glibc
    if ".".join(map(str, referee_glibc())) != fixture_manifest()["referee_glibc"]:
        pytest.skip("these real native captures use GPUWM_WRF_GLIBC=2.43")
    report = replay()
    assert report["words"] == 43200
    assert report["differing"] == 0, report
