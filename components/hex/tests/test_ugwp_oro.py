"""woof.hex.ugwp_oro: which runs need the GSL drag statistics, and the file layout they are written in.

The breakage these pin: an RRFS-shape run (rrfs_v2_dev) on a mesh RRFS publishes
no ugwp_oro_data.nc for stopped before its first step (2026-10-24); the
prepare stages build the file only for runs whose physics reads it.
"""
from __future__ import annotations

from woof.hex import ugwp_oro


def test_rrfs_shape_suites_read_the_statistics():
    assert ugwp_oro.suite_reads_ugwp("rrfs_v2_dev")
    assert ugwp_oro.suite_reads_ugwp("rrfs_v2_tempo_candidate")
    assert not ugwp_oro.suite_reads_ugwp("arwen_mpas_column")
    assert not ugwp_oro.suite_reads_ugwp(None)


def test_forecast_argv_spellings():
    assert ugwp_oro.forecast_args_read_ugwp(["forecast", "--forecast-arg=--physics-suite=rrfs_v2_dev"])
    assert ugwp_oro.forecast_args_read_ugwp(["forecast", "--physics-suite", "rrfs_v2_dev"])
    assert ugwp_oro.forecast_args_read_ugwp(["--physics", "config_gwdo_scheme=bl_ugwp_gwdo"])
    assert not ugwp_oro.forecast_args_read_ugwp(["--physics-suite", "rrfs_v2_dev",
                                                 "--physics", "config_gwdo_scheme=bl_ysu_gwdo"])
    assert not ugwp_oro.forecast_args_read_ugwp(["forecast", "--mesh", "x1.40962"])


def test_prepare_skips_runs_that_read_no_statistics(tmp_path):
    assert ugwp_oro.prepare_for_forecast(["forecast", "--mesh", "m"], tmp_path / "m.grid.nc") is None
    named = ["forecast", "--physics-suite", "rrfs_v2_dev", "--orographic-statistics", "x.ugwp_oro_data.nc"]
    assert ugwp_oro.prepare_for_forecast(named, tmp_path / "m.grid.nc") is None


def test_file_layout_is_rrfs_order():
    assert len(ugwp_oro.ORDER) == 20
    assert ugwp_oro.ORDER[:2] == ("var2dls", "conls") and ugwp_oro.ORDER[10:12] == ("var2dss", "conss")
    assert ugwp_oro.VARIABLES["oa3ss"] == (
        "unitless", "asymmetry of subgrid-scale orography for south-westerly flow (small-scale GWD)")
    assert ugwp_oro.default_path("/w/mesh.grid.nc", "region-ab").name == "region-ab.ugwp_oro_data.nc"
