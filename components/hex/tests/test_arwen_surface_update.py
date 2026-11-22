"""Real Rust record reads and CUDA conversions across a clock month boundary."""
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.surface_series_reader import covering_reader, discover


def test_cold_start_ice_mixture_has_physical_endmembers():
    from woof.hex.arwen_surface_update import initialize_fractional_ice
    ice = np.array([0, .026, .2, .7, 1], dtype="f4")
    fields = dict(xice=ice, **{key: np.full(5, .2, dtype="f4")
                              for key in ("albedo", "albbck", "emiss", "embck")})
    initialize_fractional_ice(fields, .02, np)
    # Independently recover the ice endmember used by the next record's
    # alpha_new = .08 + (alpha_old-.08)*x_new/x_old operation.
    recovered = .08 + (fields["albedo"][1:]-.08)/ice[1:]
    np.testing.assert_allclose(recovered, .8, rtol=2e-7)
    assert fields["albedo"][0] == np.float32(.2)
    np.testing.assert_array_equal(fields["albbck"][1:], np.float32(.8))
    np.testing.assert_array_equal(fields["emiss"][1:], np.float32(.98))


def test_initial_surface_alarm_refreshes_detached_boundary_receipt(tmp_path, monkeypatch):
    from woof.hex import arwen_surface_update as update, surface_series_reader
    old = {"open_water_columns": 2, "sea_ice_columns": 0}
    new = {"open_water_columns": 1, "sea_ice_columns": 1}
    backend = SimpleNamespace(_constructor=SimpleNamespace(n_columns=2),
        _seam=SimpleNamespace(elapsed_seconds=0), _last_receipt={
            "phase": "boundary", "surface_classification": old})
    class Service:
        def __init__(self, *args, **kwargs):
            self.classification = old
        def records(self, when):
            self.classification = new
        def receipt(self):
            return {"enabled": True, "classification": dict(self.classification)}
    monkeypatch.setattr(surface_series_reader, "discover", lambda root: tmp_path / "surface.nc")
    monkeypatch.setattr(surface_series_reader, "covering_reader", lambda *args, **kwargs: object())
    monkeypatch.setattr(update, "ArwenSurfaceUpdate", Service)
    update.attach(backend, work_dir=tmp_path, init_path=tmp_path / "init.nc",
                  start=datetime(2026, 10, 17), end=datetime(2026, 11, 16))
    assert backend._last_receipt["surface_classification"] == new
    assert backend._last_receipt["surface_update"]["enabled"]
    assert backend._last_receipt["phase"] == "boundary"


def test_constructor_preserves_restored_radiation_words(tmp_path, monkeypatch):
    import sys
    from woof.hex.arwen_surface_update import ArwenSurfaceUpdate
    monkeypatch.setitem(sys.modules, "cupy", np)
    fields = {key: np.array([.321, .654], dtype="f4") for key in
              ("albedo", "albbck", "emiss", "embck", "tsk", "tmn")}
    fields["xice"] = np.array([.2, .8], dtype="f4")
    saved = {key: value.copy() for key, value in fields.items()}
    backend = SimpleNamespace(_surface_restart={"restored": True},
        _constructor=SimpleNamespace(n_columns=2, xice_threshold=.02,
            expected_surface_classification=lambda: {}),
        _seam=SimpleNamespace(_driver=SimpleNamespace(fields=fields,
            noahmp_params=SimpleNamespace(land_use=SimpleNamespace(isice=15, iswater=17)))))
    monkeypatch.setattr(ArwenSurfaceUpdate, "_read_initial", lambda self, name,
        levels=1, axis=None, dtype=np.float32: np.zeros((levels, 2) if axis else 2, dtype=dtype))
    start = datetime(2026, 10, 17)
    reader = SimpleNamespace(n_cells=2, record_interval=timedelta(hours=6), times=[start])
    service = ArwenSurfaceUpdate(backend, reader, tmp_path / "init.nc", start=start)
    assert (service.isice, service.iswater) == (15, 17)
    for key, value in saved.items():
        np.testing.assert_array_equal(fields[key].view("u4"), value.view("u4"))


def write_inputs(root):
    from netCDF4 import Dataset
    start = datetime(2024, 1, 31)
    series = root / "surface_series.nc"
    with Dataset(series, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        for key, size in (("Time", 5), ("nCells", 4), ("StrLen", 64)):
            ds.createDimension(key, size)
        stamps = ds.createVariable("xtime", "S1", ("Time", "StrLen"))
        for k in range(5):
            stamps[k] = np.frombuffer((start+timedelta(hours=6*k)).strftime(
                "%Y-%m-%d_%H:%M:%S").ljust(64).encode(), dtype="S1")
        ds.createVariable("sst", "f4", ("Time", "nCells"))[:] = np.array(
            [[290+k, 291+k, 285, 295+k] for k in range(5)], dtype="f4")
        ice = np.array([[0, .7, 0, 0]]*4 + [[.6, 0, 0, 0]], dtype="f4")
        ds.createVariable("xice", "f4", ("Time", "nCells"))[:] = ice
    init = root / "init.nc"
    with Dataset(init, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        ds.createDimension("nCells", 4)
        ds.createDimension("nMonths", 12)
        for key, multiplier in (("greenfrac", 5), ("albedo12m", 2)):
            ds.createVariable(key, "f4", ("nCells", "nMonths"))[:] = np.broadcast_to(
                np.arange(1, 13, dtype="f4")*multiplier, (4, 12))
        ds.createVariable("landmask", "i4", ("nCells",))[:] = [0, 0, 1, 0]
    return start, series, init


@pytest.fixture
def inputs(tmp_path):
    library = os.environ.get("WOOF_HEX_SURFACE_READER")
    if not library:
        pytest.skip("requires source-built rw-surface, WOOF_HEX_SURFACE_READER")
    return (*write_inputs(tmp_path), library)


def test_series_discovery_coverage_and_clock(inputs):
    start, series, _, library = inputs
    assert discover(series.parent / "forecast") == series
    assert discover(series.parent / "empty" / "forecast") is None
    reader = covering_reader(series, n_cells=4, start=start, end=start+timedelta(days=1),
                             library=library)
    assert reader.record_interval == timedelta(hours=6)
    assert len(reader.times) == 5
    with pytest.raises(ValueError, match="nCells"):
        covering_reader(series, n_cells=3, start=start, end=start,
                        library=library)
    with pytest.raises(ValueError, match="does not cover"):
        covering_reader(series, n_cells=4, start=start, end=start+timedelta(days=2),
                        library=library)


def test_month_boundary_device_words_and_restart(inputs):
    if os.environ.get("S2S_SURFACE_GPU") != "1":
        pytest.skip("requires a CUDA card to itself and S2S_SURFACE_GPU=1")
    import cupy as cp
    from woof.hex.arwen_surface_update import ArwenSurfaceUpdate
    start, series, init, library = inputs
    f = {name: cp.full((1, 4), value, dtype="f4") for name, value in dict(
        vegfra=20, tmn=280, snow=3, snowc=1, snowh=.2, albbck=.2,
        shdmin=10, shdmax=80, albedo=.2, emiss=.98, embck=.98,
        glw=300, swdown=200, hfx=10, qfx=.0001, ust=.2).items()}
    for name, values in dict(xland=[2, 1, 1, 2], xice=[0, .7, 0, 0],
                              tsk=[290, 270, 285, 295]).items():
        f[name] = cp.array([values], dtype="f4")
    f["ivgtyp"] = cp.array([[17, 15, 7, 17]], dtype="i4")
    f["isltyp"] = cp.array([[14, 16, 5, 14]], dtype="i4")
    for name, value in dict(tslb=280, smois=.3, sh2o=.2).items():
        f[name] = cp.full((4, 1, 4), value, dtype="f4")
    counts = dict(xland_source="native", xice_threshold=.02, xland_land_columns=2,
        xland_water_columns=2, sea_ice_columns=1, open_water_columns=2,
        sflx_land_columns=1, glacier_columns=0)
    driver = SimpleNamespace(fields=f, surface_classification=counts,
        last_noahmp_census=None, noahmp_params=SimpleNamespace(
            land_use=SimpleNamespace(isice=15, iswater=17)))
    backend = SimpleNamespace(_constructor=SimpleNamespace(n_columns=4,
        xice_threshold=.02, expected_surface_classification=lambda: counts),
        _seam=SimpleNamespace(_driver=driver), _rollback_snapshot=True)
    reader = covering_reader(series, n_cells=4, start=start, end=start+timedelta(days=1),
                             library=library)
    service = ArwenSurfaceUpdate(backend, reader, init, start=start)
    backend._surface_updates = service
    service.records(start)
    before = cp.asnumpy(f["vegfra"]).view("u4").copy()
    expected_january = np.float32((np.float32(5)*15+np.float32(10)*16)/31)
    assert before[0, 2] == expected_january.view("u4")
    saved = service.state()
    saved_fields = {k: v.copy() for k, v in f.items()}
    service.advance(86340, 60)
    service.records(start+timedelta(days=1))
    assert service.counts == dict(surface=3, green=2, skin=1, deep=1)
    expected_february = np.float32((np.float32(5)*14+np.float32(10)*17)/31)
    after = cp.asnumpy(f["vegfra"]).view("u4")
    assert after[0, 2] == expected_february.view("u4") != before[0, 2]
    np.testing.assert_array_equal(cp.asnumpy(service.owned["sst"]).view("u4"),
                                 np.array([294, 295, 285, 299], dtype="f4").view("u4"))
    np.testing.assert_array_equal(cp.asnumpy(f["xland"]), [[1, 2, 1, 2]])
    np.testing.assert_array_equal(cp.asnumpy(f["ivgtyp"]), [[15, 17, 7, 17]])
    np.testing.assert_array_equal(cp.asnumpy(f["isltyp"]), [[16, 14, 5, 14]])
    for key in ("snow", "snowc", "snowh"):
        assert float(f[key][0, 1]) == 0
    np.testing.assert_array_equal(cp.asnumpy(f["sh2o"][:, 0, 0]), np.zeros(4))
    np.testing.assert_array_equal(cp.asnumpy(f["sh2o"][:, 0, 1]), np.ones(4))
    assert float(f["tmn"][0, 0]) == float(np.float32(271.4))
    assert all(r["h2d_transfers"] == 1 and r["d2h_bytes"] == 0 for r in reader.reads)
    assert all(bool(cp.all(cp.isfinite(v))) for v in service.owned.values())
    # The backend restores engine fields; the service restores its own pools,
    # alarm phase and monthly calendar, then reclassifies the restored fields.
    for k, v in saved_fields.items():
        f[k][...] = v
    service.restore(saved)
    for k, v in saved["arrays"].items():
        np.testing.assert_array_equal(cp.asnumpy(service.owned[k]).view("u4"), v.view("u4"))
    assert not service.alarms.surface_rings(start)["surface"]
    receipt = os.environ.get("S2S_SURFACE_RECEIPT")
    if receipt:
        Path(receipt).write_text(json.dumps(dict(test="month boundary CUDA words",
            january_word=int(before[0, 2]), february_word=int(after[0, 2]),
            changed_surface_classes=2, reads=reader.reads), indent=2)+"\n")


def test_analysis_alarm_keeps_the_preceding_physics_census():
    from woof.hex.drivers.run_cuda_v841_full_physics_x4 import require_arwen_v2_surface_execution
    used = dict(xland_source="native", xice_threshold=.02, xland_land_columns=2,
        xland_water_columns=2, sea_ice_columns=1, open_water_columns=2,
        sflx_land_columns=1, glacier_columns=0)
    current = dict(used, xland_land_columns=3, xland_water_columns=1,
                   sea_ice_columns=2, open_water_columns=1)
    payload = dict(surface_classification=current,
        last_noahmp_census=dict(land=1, water=2, sea_ice=1, glacier=0),
        surface_update=dict(enabled=True, classification=current,
                            physics_classification=used))
    result = require_arwen_v2_surface_execution(payload, executed=True, label="analysis alarm")
    assert result["surface_classification"] == current
    assert result["last_noahmp_census"]["sea_ice"] == 1
    with pytest.raises(ValueError, match="surface classification changed"):
        require_arwen_v2_surface_execution(dict(payload, surface_classification=used),
                                          executed=True, label="stale current state")
    with pytest.raises(ValueError, match="census/provenance changed"):
        require_arwen_v2_surface_execution(dict(payload,
            last_noahmp_census=dict(land=1, water=1, sea_ice=2, glacier=0)),
            executed=True, label="future execution census")
