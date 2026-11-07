"""The container contract, judged by the OTHER lane's real decoder.

Everything else in this suite proves the Python reader agrees with the
Python fixture writer, which is two readings of one contract and still
one lane's opinion.  This module hands a pack to the Rust ``rw_goes
verify`` -- the binary the bridge actually ships -- and requires it to
say PASS, and requires it to say something else when a byte is wrong.

Skipped, not failed, when the binary is absent: ``rw_goes`` lives on
``development branch`` and this lane branched off ``feature/da-scorecard``,
so until the two are merged the crate is not in this tree.  Point
``WOOF_RW_GOES`` at a built binary to run it before then.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from woof.obs.goes_cwp import read_cloudtop_pack, read_cwp_pack

from goes_pack_fixtures import (CLOUDTOP_SCHEMA_V2, CWP_SCHEMA_V2,
                                sibling_block, write_cloudtop_pack,
                                write_cwp_pack)

REPO = Path(__file__).resolve().parent.parent


def _binary() -> Path | None:
    override = os.environ.get("WOOF_RW_GOES")
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    for name in ("rw_goes.exe", "rw_goes"):
        candidate = REPO / "tools" / "rustwx" / "target" / "release" / name
        if candidate.is_file():
            return candidate
    found = shutil.which("rw_goes")
    return Path(found) if found else None


RW_GOES = _binary()
requires_bridge = pytest.mark.skipif(
    RW_GOES is None,
    reason="rw_goes is not built in this tree; set WOOF_RW_GOES to run "
           "the cross-lane container check")


def _verify(path):
    return subprocess.run([str(RW_GOES), "verify", "--pack", str(path)],
                          capture_output=True, text=True)


#: The shipped `rw_goes` reads v2 ONLY -- it refuses a v1 pack by name
#: rather than reading it as the additive base it is.  So the cross-lane
#: check is written in v2, and `test_this_reader_still_takes_v1` records
#: that the Python side deliberately did NOT follow: v1 packs exist on
#: disk with digests in earlier receipts, and dropping them would strand
#: every product built before the bump.
def _cwp(tmp_path, name="cwp.goespack"):
    dqf = np.array([[0.0, 512.0], [256.0, np.nan]], np.float32)
    return write_cwp_pack(
        tmp_path / name,
        cod=np.array([[10.0, 20.0], [0.0, np.nan]], np.float32),
        cps=np.array([[15.0, 30.0], [0.0, 5.0]], np.float32),
        phase=np.array([[1.0, 4.0], [0.0, np.nan]], np.float32),
        lat=np.array([[35.0, 35.0], [34.0, 34.0]], np.float32),
        lon=np.array([[-97.0, -96.0], [-97.0, -96.0]], np.float32),
        x_scan_rad=[-0.001, 0.001], y_scan_rad=[0.081, 0.079],
        schema=CWP_SCHEMA_V2,
        dqf_planes={"COD": dqf, "CPS": dqf,
                    "ACTP": np.zeros((2, 2), np.float32)})


@requires_bridge
def test_the_rust_decoder_accepts_a_cwp_pack_this_suite_writes(tmp_path):
    path = _cwp(tmp_path)
    result = _verify(path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema"] == "gpuwm-obs.goes-cwp-verify.v1"
    assert report["status"] == "PASS"
    assert report["pack_schema"] == "gpuwm-obs.goes-cwp.v2"
    assert report["planes"] == ["cwp", "phase", "cod", "cps", "lat", "lon",
                                "cod_dqf", "cps_dqf", "actp_dqf"]
    assert report["nx"] == 2 and report["ny"] == 2
    # Both lanes hash the same bytes to the same digest.
    assert report["content_sha256"] == read_cwp_pack(path).meta[
        "content_sha256"]


@requires_bridge
def test_the_rust_decoder_accepts_a_cloudtop_pack_this_suite_writes(tmp_path):
    cwp_path = _cwp(tmp_path)
    path = write_cloudtop_pack(
        tmp_path / "ct.goespack",
        cloud_top_height_m=np.array([[9000.0, np.nan]], np.float32),
        cloud_top_pressure_hpa=np.array([[300.0, np.nan]], np.float32),
        lat=np.array([[34.5, 34.5]], np.float32),
        lon=np.array([[-97.0, -96.0]], np.float32),
        x_scan_rad=[-0.001, 0.001], y_scan_rad=[0.080],
        schema=CLOUDTOP_SCHEMA_V2, sibling=sibling_block(cwp_path))
    result = _verify(path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "PASS"
    assert report["pack_schema"] == "gpuwm-obs.goes-cloudtop.v2"
    assert report["planes"] == ["cloud_top_height_m",
                                "cloud_top_pressure_hpa", "lat", "lon",
                                "acha_dqf", "ctp_dqf"]
    assert read_cloudtop_pack(path).meta["sibling"]["content_sha256"] == \
        read_cwp_pack(cwp_path).meta["content_sha256"]


@requires_bridge
def test_both_lanes_refuse_the_same_corrupted_payload(tmp_path):
    """The refusal has to be shared, not just the acceptance."""

    from woof.obs.goes_pack import GoesPackError, read_goes_pack

    path = _cwp(tmp_path)
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    assert _verify(path).returncode != 0
    with pytest.raises(GoesPackError, match="payload hashes to"):
        read_goes_pack(path)


@requires_bridge
def test_both_lanes_read_v1_and_v2(tmp_path):
    """Old products stay readable, and the tool says which capabilities.

    The v2 bump briefly made the shipped binary v2-only, which stranded
    every v1 pack whose digest an earlier receipt recorded.  That was
    fixed upstream (rw-goes acda95029) rather than worked around here,
    and this pins the agreed posture on both sides: `verify` reads every
    schema the tool has ever written, while `cwp`/`cloud-top` still write
    v2 only.

    `per_pixel_dqf` is the capability flag both lanes branch on.  It is a
    fact about the version, not a fault, which is why a v1 pack reports
    False and still PASSes.
    """

    from woof.obs.goes_cwp import read_cwp_pack

    v1 = write_cwp_pack(
        tmp_path / "v1.goespack",
        cod=np.array([[10.0]], np.float32), cps=np.array([[15.0]], np.float32),
        phase=np.array([[1.0]], np.float32),
        lat=np.array([[35.0]], np.float32),
        lon=np.array([[-97.0]], np.float32),
        x_scan_rad=[0.0], y_scan_rad=[0.08])
    result = _verify(v1)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "PASS"
    assert report["pack_schema"] == "gpuwm-obs.goes-cwp.v1"
    assert report["per_pixel_dqf"] is False

    pack = read_cwp_pack(v1)
    assert pack.schema_version == 1
    assert pack.has_dqf_planes is False

    # ...and the v2 sibling, through the same two readers.
    v2 = _cwp(tmp_path, name="v2.goespack")
    v2_report = json.loads(_verify(v2).stdout)
    assert v2_report["per_pixel_dqf"] is True
    v2_pack = read_cwp_pack(v2)
    assert v2_pack.schema_version == 2
    assert v2_pack.has_dqf_planes is True
