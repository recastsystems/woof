"""The New Tiedtke derivative rows are of the forward text this build runs.

THE BREAKAGE THIS PREVENTS: the 26 rows of ``ad/phys/cumulus/kset`` are generated from one forward text (the scheme's
NVRTC source followed by the driver's forcing fold), and their image holds that forward beside its tangent and its
reverse. An edit to the scheme's port, to the fold, or an engine version whose New Tiedtke source differs, moves the
forward a forecast runs while the rows stay as they were: every window on a forecast with New Tiedtke on would then
linearise another model than the one recorded, and nothing would say so. This test fails first, and names the rebuild.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CUMULUS = ROOT / "ad/phys/cumulus"
REBUILD = ("rebuild the image and the rows (ad/phys/cumulus/build.sh, then gen/mkspec_nt.py), prove them "
           "(host/proof.sh), and give the kernel set that holds them a new version")


def _rows():
    if not (CUMULUS / "kset/rows.json").is_file():
        pytest.skip("an sdist carries no ad/ folder; the repository checkout does")
    return json.loads((CUMULUS / "kset/rows.json").read_text(encoding="utf-8"))


def test_every_row_names_the_one_image_and_its_own_spec():
    rows = _rows()
    assert len(rows) == 26 and len({r["function"] for r in rows}) == 26
    assert {r["derivative"].get("shared_image") for r in rows} == {"nt_ad.fatbin"}
    forward = set()
    for row in rows:
        spec = json.loads((CUMULUS / "kset" / row["derivative"]["spec"]["path"]).read_text(encoding="utf-8"))
        assert spec["kernel"] == row["function"]
        assert [a["name"] for a in row["arguments"]], row["function"]
        forward.add(spec["forward_source_sha256"])
    assert len(forward) == 1, f"the rows are of {len(forward)} forward texts"


def test_the_rows_are_of_this_builds_forward_text():
    rows = _rows()
    spec = json.loads((CUMULUS / "kset" / rows[0]["derivative"]["spec"]["path"]).read_text(encoding="utf-8"))
    pytest.importorskip("woof.core.kernels", reason="the forward text is assembled from the engine's New Tiedtke source")
    from woof.hex.mpasphys import ntiedtke_v842
    from woof.hex.mpasphys.rrfs import driver_rrfs

    # build.sh: the scheme's text, a newline, the driver's forcing fold (one image holds both)
    text = ntiedtke_v842.cuda_source() + "\n" + driver_rrfs._CONVECTION_FORCING_CUDA
    now = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert now == spec["forward_source_sha256"], (
        f"the New Tiedtke forward text of this build is {now}; the derivative rows are of "
        f"{spec['forward_source_sha256']}: {REBUILD}")


def test_the_upper_damping_rows_are_of_this_builds_forward_text():
    """The two hand-written rows of NOAA's upper damping follow the forward's kernels statement by statement; the
    same breakage as above if those kernels or their float helpers change under them."""
    kset = ROOT / "ad/phys/upper_damping/kset"
    if not (kset / "rows.json").is_file():
        pytest.skip("an sdist carries no ad/ folder; the repository checkout does")
    rows = json.loads((kset / "rows.json").read_text(encoding="utf-8"))
    assert [r["function"] for r in rows] == ["upper_cam_f32", "upper_rayleigh_f32"]
    from woof.hex import cuda_upper_damping_v841

    now = hashlib.sha256(cuda_upper_damping_v841._CUDA_SOURCE.encode("utf-8")).hexdigest()
    for row in rows:
        spec = json.loads((kset / row["derivative"]["spec"]["path"]).read_text(encoding="utf-8"))
        assert spec["kernel"] == row["function"]
        assert spec["forward_source_sha256"] == now, (
            f"the upper-damping forward text of this build is {now}; row {row['function']} is written against "
            f"{spec['forward_source_sha256']}: read ad/phys/upper_damping/cuda against the forward again, rebuild "
            "(build.sh) and give the kernel set that holds the rows a new version")
