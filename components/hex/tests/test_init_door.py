"""Refusal-surface tests for the ``mpas-port init`` front door.

These exercise the door's own contract, engine discovery, argument
completeness, met-source resolution, with no engine binary and no real
data.  The numeric path is proven against the artifact (the built
rw_mpas_init exe on real factory inputs), not mocked here. The aerosol
module replay against the pristine Fortran needs the home lane workspace
and lives in test_init_door_pool.py.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from woof.hex.cli import main
from woof.hex.init_door import (
    InitDoorRefusal,
    _probe_wps_header,
    resolve_engine,
    resolve_met_source,
)


def _wps_probe_bytes(version: int = 5, endian: str = ">") -> bytes:
    return struct.pack(f"{endian}ii", 4, version)


COMPLETE_SWITCHES = [
    "--start-time", "2025-01-01_00:00:00",
    "--nfglevels", "38",
    "--nfgsoillevels", "4",
    "--extrap-airtemp", "lapse-rate",
    "--use-spechumd", "no",
    "--theta-adv-order", "3",
    "--coef-3rd-order", "0.25",
    "--virtual-factor", "reproduce-fortran",
    "--deep-soil-moisture", "reproduce-fortran",
    "--landuse-table", "MODIFIED_IGBP_MODIS_NOAH",
    "--frac-seaice", "yes",
    "--tsk-seaice-threshold", "100.0",
    "--oned-underflow", "preserve",
]


def _paths(tmp_path: Path) -> list[str]:
    return [
        "--met", str(tmp_path / "met"),
        "--static", str(tmp_path / "static.nc"),
        "--capsule", str(tmp_path / "capsule.nc"),
        "--reference", str(tmp_path / "reference.nc"),
        "--out", str(tmp_path / "init.nc"),
    ]


def test_missing_switch_refusal_names_the_switch_and_the_breakage(tmp_path, capsys):
    argv = ["init", *_paths(tmp_path), *COMPLETE_SWITCHES]
    index = argv.index("--extrap-airtemp")
    del argv[index : index + 2]
    rc = main(argv)
    err = capsys.readouterr().err
    assert rc == 2
    assert "--extrap-airtemp was not given" in err
    assert "no default" in err
    assert "config_extrap_airtemp" in err


def _no_engines_anywhere(monkeypatch):
    """A machine with no rw_mpas_init on any rung.

    Deleting the environment variables is not enough: the ladder also reads
    woof's bridge directories, and a box that has ever run
    ``woof fetch-bridges`` has the binary there.  That is the point of the
    rung, and it means a refusal test has to remove it deliberately.
    """

    from woof.hex import engines

    for name in ("WOOF_HEX_RW_MPAS_INIT", "RW_MPAS_INIT", "WOOF_RW_MPAS_INIT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: ())


def test_engine_absent_refusal_names_env_var_and_build_line(monkeypatch):
    _no_engines_anywhere(monkeypatch)
    with pytest.raises(InitDoorRefusal) as caught:
        resolve_engine(None)
    message = str(caught.value)
    assert "RW_MPAS_INIT" in message
    assert "cargo build" in message
    # One command stages this binary; naming it first is the difference
    # between a remedy and a research project.
    assert "woof fetch-bridges" in message


def test_engine_path_must_exist(tmp_path):
    with pytest.raises(InitDoorRefusal) as caught:
        resolve_engine(tmp_path / "no-such-exe")
    message = str(caught.value)
    assert "which is not a file" in message
    assert "--engine" in message


def test_a_staged_engine_resolves_with_no_environment_variable(
    monkeypatch, tmp_path
):
    """``woof fetch-bridges`` alone is enough to open this door."""

    from woof.hex import engines

    staged = tmp_path / engines.executable_name("rw_mpas_init")
    staged.write_bytes(b"")
    staged.chmod(0o755)
    for name in ("WOOF_HEX_RW_MPAS_INIT", "RW_MPAS_INIT", "WOOF_RW_MPAS_INIT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: (staged,))

    assert resolve_engine(None) == staged.resolve()


def test_met_source_missing_is_refused_by_name(tmp_path):
    with pytest.raises(InitDoorRefusal) as caught:
        resolve_met_source(tmp_path / "absent")
    assert "does not exist" in str(caught.value)
    assert "ungrib" in str(caught.value)


def test_met_directory_with_no_intermediate_is_refused(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"not an intermediate file")
    with pytest.raises(InitDoorRefusal) as caught:
        resolve_met_source(tmp_path)
    assert "no WPS intermediate file" in str(caught.value)


def test_met_directory_with_one_candidate_resolves_to_it(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"prose")
    candidate = tmp_path / "MET_2025-01-01_00"
    candidate.write_bytes(_wps_probe_bytes())
    assert resolve_met_source(tmp_path) == candidate


def test_met_directory_with_two_candidates_demands_an_explicit_file(tmp_path):
    for name in ("MET_2025-01-01_00", "MET_2025-01-02_00"):
        (tmp_path / name).write_bytes(_wps_probe_bytes())
    with pytest.raises(InitDoorRefusal) as caught:
        resolve_met_source(tmp_path)
    message = str(caught.value)
    assert "exactly one valid time" in message
    assert "MET_2025-01-01_00" in message and "MET_2025-01-02_00" in message


@pytest.mark.parametrize("endian", [">", "<"])
@pytest.mark.parametrize("version", [3, 4, 5])
def test_probe_accepts_every_wps_version_both_endians(tmp_path, endian, version):
    path = tmp_path / "candidate"
    path.write_bytes(_wps_probe_bytes(version, endian))
    assert _probe_wps_header(path)


def test_probe_rejects_non_wps_bytes(tmp_path):
    path = tmp_path / "candidate"
    path.write_bytes(b"\x00" * 8)
    assert not _probe_wps_header(path)
    short = tmp_path / "short"
    short.write_bytes(b"\x00")
    assert not _probe_wps_header(short)


def test_missing_out_dir_is_refused_before_any_engine_run(tmp_path, monkeypatch, capsys):
    engine = tmp_path / "rw_mpas_init"
    engine.write_bytes(b"#!/bin/sh\n")
    engine.chmod(0o755)
    monkeypatch.setenv("RW_MPAS_INIT", str(engine))
    argv = [
        "init",
        "--met", str(tmp_path / "met"),
        "--static", str(tmp_path / "static.nc"),
        "--capsule", str(tmp_path / "capsule.nc"),
        "--reference", str(tmp_path / "reference.nc"),
        "--out", str(tmp_path / "no-such-dir" / "init.nc"),
        *COMPLETE_SWITCHES,
    ]
    rc = main(argv)
    err = capsys.readouterr().err
    assert rc == 2
    assert "output directory" in err
    assert "does not exist" in err


# ---------------------------------------------------------------------------
# run declarations: the init carries its own clock on every route
# ---------------------------------------------------------------------------
def _declaring_namespace():
    import argparse

    return argparse.Namespace(
        start_time="2026-09-13_15:00:00", nfglevels=51, nfgsoillevels=4,
        use_spechumd="yes", frac_seaice="no", extrap_airtemp="constant",
        tsk_seaice_threshold="100.0", landuse_table="MODIFIED_IGBP_MODIS_NOAH",
    )


def test_run_declarations_are_the_constructed_routes_keys():
    from woof.hex.init_door import run_declarations

    declared = run_declarations(_declaring_namespace())
    assert declared == {
        "config_start_time": "2026-09-13_15:00:00",
        "config_nfglevels": 51,
        "config_nfgsoillevels": 4,
        "config_nsoillevels": 4,
        "config_use_spechumd": True,
        "config_frac_seaice": False,
        "config_extrap_airtemp": "constant",
        "config_tsk_seaice_threshold": 100.0,
        "config_landuse_data": "MODIFIED_IGBP_MODIS_NOAH",
    }
    assert all(key.startswith("config_") for key in declared)


def test_stamping_writes_only_what_the_capsule_copy_left_absent(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    import numpy as np

    from woof.hex.init_door import run_declarations, stamp_run_declarations

    init = tmp_path / "init.nc"
    with netCDF4.Dataset(init, mode="w") as dataset:
        dataset.createDimension("nCells", 3)
        dataset.setncattr("config_nvertlevels", np.int32(55))
        dataset.setncattr("config_landuse_data", "USGS")

    record = stamp_run_declarations(init, run_declarations(_declaring_namespace()))

    assert record["already_present"] == {"config_landuse_data": "USGS"}
    assert set(record["stamped"]) == {
        "config_start_time", "config_nfglevels", "config_nfgsoillevels",
        "config_nsoillevels", "config_use_spechumd", "config_frac_seaice",
        "config_extrap_airtemp", "config_tsk_seaice_threshold",
    }
    with netCDF4.Dataset(init) as dataset:
        assert dataset.getncattr("config_start_time") == "2026-09-13_15:00:00"
        assert dataset.getncattr("config_landuse_data") == "USGS"
        assert dataset.getncattr("config_use_spechumd") == "YES"
        assert dataset.getncattr("config_frac_seaice") == "NO"
        nfg = dataset.getncattr("config_nfglevels")
        assert nfg == 51 and nfg.dtype == np.int32
        threshold = dataset.getncattr("config_tsk_seaice_threshold")
        assert threshold == 100.0 and threshold.dtype == np.float32
        assert dataset.getncattr("config_nvertlevels") == 55


def test_stamping_a_native_capsules_init_changes_nothing(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    import hashlib

    from woof.hex.init_door import run_declarations, stamp_run_declarations
    from woof.hex.vertical_spec import encode_config_attribute

    declared = run_declarations(_declaring_namespace())
    init = tmp_path / "init.nc"
    with netCDF4.Dataset(init, mode="w") as dataset:
        dataset.createDimension("nCells", 3)
        for key, value in declared.items():
            dataset.setncattr(key, encode_config_attribute(value))
    before = hashlib.sha256(init.read_bytes()).hexdigest()

    record = stamp_run_declarations(init, declared)

    assert record["stamped"] == {}
    assert set(record["already_present"]) == set(declared)
    assert hashlib.sha256(init.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("terrain_present", [False, True])
def test_source_terrain_requirement_is_format_metadata(monkeypatch, tmp_path, terrain_present):
    from types import SimpleNamespace
    from woof.hex import wps_intermediate
    from woof.hex.init_door import scan_met_file

    names = ["LANDSEA", "TT", "ST000010", "SM000010"]
    if terrain_present:
        names.append("SOILHGT")

    class Reader:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_fields(self, *, load_values):
            assert load_values is False
            return iter(SimpleNamespace(field=name, level=200100.0, map_source="synthetic",
                                        valid_time="2026-10-03_15:00:00") for name in names)

    monkeypatch.setattr(wps_intermediate, "WpsIntermediateReader", Reader)
    arguments = dict(nfglevels=1, start_time="2026-10-03_15:00:00")
    if terrain_present:
        assert "SOILHGT" in scan_met_file(tmp_path / "arbitrary.MET", **arguments).fields
    else:
        with pytest.raises(InitDoorRefusal, match="source terrain would remain zero"):
            scan_met_file(tmp_path / "arbitrary.MET", **arguments)


def test_cached_vertical_readers_share_the_artifact(tmp_path, monkeypatch):
    """Member inits of one cycle read the cached vertical artifact at once (unified image, 2026-10-16:
    six GEFS members ran rw_mpas_init one at a time behind an exclusive lock); a run with other
    stamped declarations waits for those readers, then stamps."""
    import argparse
    import json
    import threading
    from netCDF4 import Dataset
    from woof.hex import init_door, vertical_spec

    built = []

    def materialize(*, grid, static, spec_path, output, receipt_path, run_config):
        built.append(output)
        with Dataset(output, "w", format="NETCDF3_64BIT_OFFSET") as dataset:
            for key in init_door.STAMPED_DECLARATIONS:
                dataset.setncattr(key, run_config[key])
        receipt_path.write_text(json.dumps({"output": str(output), "inputs": {"vertical_spec": {
            "canonical_sha256": "0" * 64}}, "invariants": {}}), encoding="utf-8")

    monkeypatch.setattr(init_door, "_vertical_cache_key", lambda arguments: ("k" * 64, {"m": 1}))
    monkeypatch.setattr(vertical_spec, "materialize_vertical_artifact", materialize)
    args = argparse.Namespace(**vars(_declaring_namespace()), grid=None, static=None, vertical_spec=None)
    first = init_door._cached_vertical_source(args, tmp_path, {})
    assert first.summary["vertical_cache"]["hit"] is False and len(built) == 1
    got = {}

    def again(name, namespace):
        got[name] = init_door._cached_vertical_source(namespace, tmp_path, {})

    reader = threading.Thread(target=again, args=("same", args))
    reader.start()
    reader.join(timeout=10)
    assert "same" in got, "a run with the same declarations waited for the first reader"
    assert got["same"].summary["vertical_cache"]["hit"] is True and len(built) == 1
    other = argparse.Namespace(**{**vars(args), "start_time": "2026-09-13_18:00:00"})
    writer = threading.Thread(target=again, args=("other", other))
    writer.start()
    writer.join(timeout=1)
    assert "other" not in got, "a re-stamp must wait for readers of the old declarations"
    first.lock.close()
    got["same"].lock.close()
    writer.join(timeout=10)
    assert "other" in got
    stamp = got["other"].summary["vertical_cache"]
    assert stamp["config_start_time"] == "2026-09-13_18:00:00"
    assert stamp["config_start_time_before"] == "2026-09-13_15:00:00"
    got["other"].lock.close()
    with Dataset(tmp_path / ("k" * 24 + ".vertical.nc")) as dataset:
        assert dataset.getncattr("config_start_time") == "2026-09-13_18:00:00"


# ---------------------------------------------------------------------------
# --first-guess-hydrometeors: the first guess's own condensate reaches the
# MPAS-A init by default (hex-meshes-sources review, 2026-10-29: an HRRR start
# on MPAS-A physics had no reflectivity at f000), never under strict numerics.
HRRR_FIELDS = ("GHT", "PRES", "QC", "QG", "QI", "QNC", "QNI", "QNR", "QR", "QS", "SPECHUMD", "TT", "UU", "VV")
GFS_FIELDS = ("GHT", "RH", "TT", "UU", "VV")


def _choose(monkeypatch, fields, *, speaks=True, requested=None, numerics="fast"):
    from argparse import Namespace
    from woof.hex import init_door
    from woof.hex import numerics as hex_numerics
    monkeypatch.setenv("WOOF_HEX_NUMERICS", numerics)
    hex_numerics._reset_for_tests()
    monkeypatch.setattr(init_door, "engine_reads_first_guess_hydrometeors", lambda engine: speaks)
    try:
        return init_door.first_guess_hydrometeors(Namespace(first_guess_hydrometeors=requested), "rw_mpas_init", fields)
    finally:
        hex_numerics._reset_for_tests()


def test_an_hrrr_first_guess_starts_with_its_own_hydrometeors_by_default(monkeypatch):
    record = _choose(monkeypatch, HRRR_FIELDS)
    assert record["argv"] == ["--first-guess-hydrometeors", "yes"] and record["applied"]
    assert record["first_guess_records"] == ["QC", "QG", "QI", "QNC", "QNI", "QNR", "QR", "QS"]


def test_strict_numerics_keeps_native_case_7(monkeypatch):
    record = _choose(monkeypatch, HRRR_FIELDS, numerics="strict")
    assert record["argv"] == [] and record["requested"] == "no" and not record["applied"]


def test_no_is_honoured_and_a_dry_first_guess_changes_nothing(monkeypatch):
    assert _choose(monkeypatch, HRRR_FIELDS, requested="no")["argv"] == []
    gfs = _choose(monkeypatch, GFS_FIELDS)
    assert gfs["argv"] == [] and gfs["first_guess_records"] == [] and not gfs["applied"]


def test_an_engine_without_the_flag_is_a_named_gap_not_a_silent_dry_start(monkeypatch, capsys):
    record = _choose(monkeypatch, HRRR_FIELDS, speaks=False)
    assert record["argv"] == [] and not record["applied"]
    assert "predates --first-guess-hydrometeors" in record["gap"]
    assert "named gap" in capsys.readouterr().err
