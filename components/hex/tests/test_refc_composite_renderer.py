"""A frame whose REFC holds more than its volume draws its composite reflectivity map from REFC: the real renderer.

THE BREAKAGE THIS PREVENTS (hrrrv5 + storm scheme zoom globe, GFS 2026-10-31 06Z, f006, 2026-10-31; production
run_c9412ffb836c65e3): rw_mpas_convert adds the cumulus scheme's rain to REFC and writes GPUWM_REFC_NOTE, and the map's
title prints that note, but a frame that also has a REFL_10CM volume was drawn from the volume, which holds only the
microphysics. The map read "incl. parameterised convection" with 1.2 % of a 2,700 km window above 10 dBZ against 27.4 %
in the frame's own REFC, and a 3 km core read "No values to plot" under showers its rain map showed.

The frame says what its REFC holds, so nothing here names a suite: a frame with no note (WRF, WOOF, a hex run with no
cumulus scheme) keeps drawing its volume.
"""
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import engines
from woof.hex.cuda_arwen_physics_v841 import history_cumulus_rain_rate

netCDF4 = pytest.importorskip("netCDF4")

NOTE = "incl. parameterised convection"
FROM_REFC = "Reading native column-maximum reflectivity REFC"
FROM_VOLUME = "Reading native WRF REFL_10CM reflectivity volume"
NY, NX, NZ = 8, 10, 3


def renderer():
    path, _ = engines.locate(engines.RENDERER)
    if path is None:
        pytest.skip("no rw_wrfbatch on this install")
    return path


def frame(path, *, note, volume):
    """A converted hex frame as small as the renderer takes: REFC 30 dBZ everywhere, its volume (if any) no echo."""
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_DATA") as d:
        for name, size in (("Time", None), ("DateStrLen", 19), ("south_north", NY), ("west_east", NX),
                           ("bottom_top", NZ), ("bottom_top_stag", NZ + 1)):
            d.createDimension(name, size)
        d.setncatts({"SIMULATION_START_DATE": "2026-10-31_06:00:00", "START_DATE": "2026-10-31_06:00:00",
                     "WEST-EAST_GRID_DIMENSION": NX + 1, "SOUTH-NORTH_GRID_DIMENSION": NY + 1,
                     "BOTTOM-TOP_GRID_DIMENSION": NZ + 1, "GRID_ID": 1, "PARENT_ID": 0, "MAP_PROJ": 6,
                     "DX": np.float32(9360.0), "DY": np.float32(9360.0), "TRUELAT1": np.float32(0.0),
                     "TRUELAT2": np.float32(0.0), "STAND_LON": np.float32(23.0), "CEN_LAT": np.float32(0.5),
                     "CEN_LON": np.float32(23.0), "MOAD_CEN_LAT": np.float32(0.5), "POLE_LAT": np.float32(90.0),
                     "POLE_LON": np.float32(0.0), "GPUWM_MODEL_LABEL": "hex", "GPUWM_GRID_LABEL": "test"})
        if note:
            d.setncattr("GPUWM_REFC_NOTE", NOTE)
        times = d.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = netCDF4.stringtochar(np.array(["2026-10-31_12_00_00"], dtype="S19"))[0]
        surface, mass, stag = (("Time", "south_north", "west_east"), ("Time", "bottom_top", "south_north", "west_east"),
                               ("Time", "bottom_top_stag", "south_north", "west_east"))
        lat, lon = np.meshgrid(np.linspace(0.2, 0.8, NY), np.linspace(22.5, 23.5, NX), indexing="ij")
        levels = np.arange(NZ + 1, dtype=np.float32)[:, None, None] * np.float32(5000.0)
        for name, dims, value in (("XLAT", surface, lat), ("XLONG", surface, lon), ("HGT", surface, 300.0),
                                  ("SINALPHA", surface, 0.0), ("COSALPHA", surface, 1.0), ("REFC", surface, 30.0),
                                  ("CUPREC", surface, 1.0e-3), ("PHB", stag, levels), ("PH", stag, 0.0),
                                  ("T", mass, 0.0), ("P", mass, 0.0), ("PB", mass, 90000.0), ("QVAPOR", mass, 0.01)):
            d.createVariable(name, "f4", dims)[0, ...] = value
        if volume:
            d.createVariable("REFL_10CM", "f4", mass)[0, ...] = -35.0


def draw(tmp_path, name, **kwargs):
    work = tmp_path / name
    work.mkdir()
    file = work / "wrfout_d01_2026-10-31_12_00_00"
    frame(file, **kwargs)
    done = subprocess.run([str(renderer()), "--store-root", str(work / "store"), "--out-dir", str(work / "out"),
                           "--products", "composite_reflectivity", "--frames", "all", str(file)],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    pictures = list((work / "out").rglob("*composite_reflectivity*.png"))
    assert len(pictures) == 1, done.stdout
    return done.stdout + done.stderr, pictures[0].read_bytes()


def test_a_frame_whose_refc_holds_the_cumulus_rain_draws_refc_not_its_volume(tmp_path):
    log, noted = draw(tmp_path, "noted", note=True, volume=True)
    assert FROM_REFC in log and FROM_VOLUME not in log
    # The same frame with no note is the volume's picture, which has no echo at all: the map changed, not just its log.
    log, plain = draw(tmp_path, "plain", note=False, volume=True)
    assert FROM_VOLUME in log and FROM_REFC not in log
    assert noted != plain


def test_a_frame_with_no_volume_draws_refc_with_or_without_a_note(tmp_path):
    log, _ = draw(tmp_path, "mpas", note=True, volume=False)
    assert FROM_REFC in log
    log, _ = draw(tmp_path, "mpas-no-cumulus", note=False, volume=False)
    assert FROM_REFC in log


def test_the_arwen_route_writes_the_schemes_mean_rain_rate_since_its_last_snapshot():
    """The WOOF route's frame carries a rate only with a scheme, so its map draws REFC only with a scheme."""
    owner = SimpleNamespace()
    seam = SimpleNamespace(_driver=SimpleNamespace(rainc=object()), elapsed_seconds=0.0)
    first = np.array([0.0, 1.0], np.float32)
    assert history_cumulus_rain_rate(owner, seam, first, np)["cuprec"].tolist() == [0.0, 0.0]
    seam.elapsed_seconds = 3600.0
    later = np.array([3.6, 0.5], np.float32)  # 3.6 mm in the hour; a bucket reset is no negative rain
    expected = pytest.approx([1.0e-3, 0.0], rel=1.0e-5)
    assert history_cumulus_rain_rate(owner, seam, later, np)["cuprec"].tolist() == expected
    # A second snapshot at the same boundary reads the same rate, and a rollback to an earlier one starts over.
    assert history_cumulus_rain_rate(owner, seam, later, np)["cuprec"].tolist() == expected
    seam.elapsed_seconds = 1800.0
    assert history_cumulus_rain_rate(owner, seam, first, np)["cuprec"].tolist() == [0.0, 0.0]
    # No cumulus scheme: the driver holds no accumulator and the frame writes no field.
    assert history_cumulus_rain_rate(SimpleNamespace(), SimpleNamespace(_driver=SimpleNamespace(rainc=None)), first, np) == {}
    assert history_cumulus_rain_rate(SimpleNamespace(), SimpleNamespace(), first, np) == {}
