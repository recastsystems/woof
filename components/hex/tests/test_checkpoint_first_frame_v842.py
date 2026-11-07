"""A checkpointing run writes the same frames as the unbroken run, initial frame included.

THE BREAKAGE THIS GUARDS (fix/hex-ckpt-surface-fields): in correct mode (the door default) a fresh run that asked
for --checkpoint-hours wrote hpbl, hfx, qfx, lh, th2m and refl10cm* as zeros in its initial frame, because the
first-call completion of that frame (woof.hex.f000_first_call) was skipped for any checkpoint route.  Every hex
checkpoint run was affected, DA background forecasts included.  Measured on x1.2562 2023-03-31 12Z before the fix:
history 12,739 and diag 10,248 words differ in the 12:00 frame, every other frame 0.

The real door runs three times on one small global mesh (2 h, 30 min frames, a checkpoint at 1 h): straight,
checkpointed, and resumed from that checkpoint.  Every word of every frame the checkpointed and the resumed runs
write must equal the straight run's frame at the same valid time.

Inputs: WOOF_HEX_CKPT_INIT and WOOF_HEX_CKPT_GRID name an x1.2562 init and grid from `woof hex init`
memory and 90 s per run on an RTX 5090.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Inputs come from the environment (below), not the byte-pinned authority tree, so no `assets` marker.
pytestmark = [pytest.mark.gpu, pytest.mark.slow]

INIT = os.environ.get("WOOF_HEX_CKPT_INIT")
GRID = os.environ.get("WOOF_HEX_CKPT_GRID")
#: The three runs share one prep/kernel cache (conftest turns it off for the test process). Without it each run
#: compiles every kernel again through NVRTC on one thread, about 15 min a run on a loaded node.
#: WOOF_HEX_CKPT_PREP_CACHE names a warm one (e.g. ~/.cache/gpuwm-hex/prep); else the runs share a fresh one.
PREP_CACHE = os.environ.get("WOOF_HEX_CKPT_PREP_CACHE")


def _forecast(out: Path, *extra: str, cache: Path) -> None:
    command = [sys.executable, "-m", "woof.hex.cli", "forecast", "--mesh", "x1.2562", "--grid", GRID,
               "--init", INIT, "--hours", "2", "--history-every-minutes", "30", "--no-products",
               "--out", str(out), *extra]
    env = {**os.environ, "WOOF_HEX_PREP_CACHE": str(cache)}
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr[-4000:]


def _differing_words(a: Path, b: Path) -> dict[str, int]:
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(a) as left, netCDF4.Dataset(b) as right:
        left.set_auto_maskandscale(False)
        right.set_auto_maskandscale(False)
        assert set(left.variables) == set(right.variables)
        differ = {}
        for name in left.variables:
            x = np.atleast_1d(np.asarray(left.variables[name][...]))
            y = np.atleast_1d(np.asarray(right.variables[name][...]))
            if x.dtype.kind in "SUO":
                count = int(not np.array_equal(x, y))
            else:
                assert x.shape == y.shape, name
                count = int(np.any(x.view(np.uint8).reshape(x.size, -1) != y.view(np.uint8).reshape(y.size, -1),
                                   axis=1).sum()) if x.size else 0
            if count:
                differ[name] = count
        return differ


@pytest.mark.skipif(not (INIT and GRID), reason="WOOF_HEX_CKPT_INIT and WOOF_HEX_CKPT_GRID name the x1.2562 inputs")
def test_checkpoint_and_resume_frames_equal_the_unbroken_run(tmp_path):
    straight, checkpointed, resumed = tmp_path / "straight", tmp_path / "checkpointed", tmp_path / "resumed"
    cache = Path(PREP_CACHE).expanduser() if PREP_CACHE else tmp_path / "prep-cache"
    _forecast(straight, cache=cache)
    _forecast(checkpointed, "--checkpoint-hours", "1", cache=cache)
    carriers = sorted(checkpointed.glob("checkpoint.*.pkl"))
    assert len(carriers) == 1, carriers
    _forecast(resumed, "--restart-from", str(carriers[0]), cache=cache)

    frames = sorted(path.name for path in straight.glob("*.nc"))
    initial = [name for name in frames if name.endswith("12.00.00.nc")]
    assert {"history.2023-03-31_12.00.00.nc", "diag.2023-03-31_12.00.00.nc"} <= set(initial)
    # The initial frame carries the first-call surface diagnostics, never placeholder zeros.
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(checkpointed / "history.2023-03-31_12.00.00.nc") as frame:
        for name in ("hpbl", "hfx", "th2m"):
            assert np.count_nonzero(np.asarray(frame.variables[name][...])) > 0, name

    # The resumed run writes the frames after its checkpoint (13:00); the checkpointed run writes them all.
    after = [name for name in frames if name.split(".", 1)[1] > "2023-03-31_13.00.00.nc"]
    assert after
    for run, expected in ((checkpointed, frames), (resumed, after)):
        written = sorted(path.name for path in run.glob("*.nc"))
        assert set(expected) <= set(written), sorted(set(expected) - set(written))
        for name in expected:
            assert _differing_words(straight / name, run / name) == {}, (run.name, name)
