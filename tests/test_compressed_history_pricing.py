"""Disk admission prices lossless compressed history frames at their measured size.

Breakage it prevents: woof 2.9.10 writes history frames lossless-compressed
by default (shuffle + deflate NetCDF-4, the same words in about a third of the
disk), while disk admission still priced every frame at its classic layout, so
a run was projected at 2.7-3.9x the history it writes and refused on disks it
fits (the 12-3-1 km proof run: 52.97 GB of classic history, 15.05 GB written).
"""

import math

import pytest

from woof import disk_budget
from test_quicklook_disk_projection import _experiment, _project

#: The least compressible frame of the measured run (12 km, 44,877,020 ->
#: 16,385,928 bytes): the price must never fall below it.
WORST_MEASURED = 16_385_928 / 44_877_020


def test_the_price_keeps_a_margin_over_the_least_compressible_frame_measured():
    assert disk_budget.COMPRESSED_HISTORY_FRACTION >= 1.2 * WORST_MEASURED
    assert disk_budget.COMPRESSED_HISTORY_FRACTION < 0.5


def test_the_default_prices_compressed_frames_and_none_prices_classic(tmp_path, monkeypatch):
    exp = _experiment(tmp_path, "quicklook = { every_s = 300.0 }")
    monkeypatch.delenv("GPUWM_HISTORY_COMPRESSION", raising=False)
    monkeypatch.delenv("WOOF_WRFOUT_WRITER", raising=False)
    assert disk_budget.history_stored_fraction() == disk_budget.COMPRESSED_HISTORY_FRACTION
    compressed = _project(exp)
    monkeypatch.setenv("GPUWM_HISTORY_COMPRESSION", "none")
    assert disk_budget.history_stored_fraction() == 1.0
    classic = _project(exp)
    assert compressed["history_stored_fraction"] == disk_budget.COMPRESSED_HISTORY_FRACTION
    assert classic["history_stored_fraction"] == 1.0
    for small, whole in zip(compressed["domains"], classic["domains"]):
        # Frame counts and per-frame layout are the same; only the stored bytes shrink.
        assert small["history_frames"] == whole["history_frames"]
        assert small["history_frame_bytes"] == whole["history_frame_bytes"]
        assert small["history_bytes"] == math.ceil(
            whole["history_bytes"] * disk_budget.COMPRESSED_HISTORY_FRACTION)
    assert compressed["history_bytes"] < 0.5 * classic["history_bytes"]
    assert (classic["total_bytes"] - compressed["total_bytes"]
            == classic["history_bytes"] - compressed["history_bytes"])


@pytest.mark.parametrize("env", [{"GPUWM_HISTORY_COMPRESSION": "none"},
                                 {"WOOF_WRFOUT_WRITER": "python"}])
def test_uncompressed_writers_are_priced_whole(monkeypatch, env):
    monkeypatch.delenv("GPUWM_HISTORY_COMPRESSION", raising=False)
    monkeypatch.delenv("WOOF_WRFOUT_WRITER", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert disk_budget.history_stored_fraction() == 1.0
