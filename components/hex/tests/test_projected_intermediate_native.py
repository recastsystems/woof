"""Whole-MET byte identity and publication checks against the independent oracle.

The reference covers Noah layers and preserved source soil depth nodes.
It is test-only code from the predecessor, never a runtime route.
Run with the real projected-intermediate library and engine loader on a box.
"""
from datetime import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import hrrr_intermediate as hi
from projected_reference import regrid_hour as reference_regrid, reference_window


def _needs_native():
    """Skip, naming why, when the projected-intermediate library is not staged.

    The library ships with the engine's bridge bundle from the release that
    carries it; woof 2.8.7 does not, and a runner with no engine has none.
    The runtime refusal these tests would otherwise hit is held by
    test_missing_explicit_native_artifact_cannot_select_another_library.
    """
    try:
        hi._native_projected_entry()
    except ModuleNotFoundError as error:
        pytest.skip(f"no woof engine is installed: {error}")
    except hi.IntermediateRefusal as error:
        if "was not found" not in str(error):
            raise
        pytest.skip(f"the projected-intermediate library is not staged: {error}")


@pytest.mark.parametrize("point", [(37.77, -122.42), (41.95, -94.79), (47.50, -111.30), (35.0, -98.0)])
@pytest.mark.parametrize("radius", [20.0, 100.0])
def test_native_crop_bounds_match_the_independent_projection(point, radius):
    _needs_native()
    target = hi.target_for_cap(point, radius)
    row = hi.source_row("hrrr")
    assert hi.source_window_for(target, row) == reference_window(target, row)


def test_native_crop_refuses_points_outside_source_coverage():
    _needs_native()
    target = hi.target_for_cap((10.0, 0.0), 100.0)
    with pytest.raises(hi.IntermediateRefusal, match="coverage"):
        hi.source_window_for(target, hi.source_row("hrrr"))


def snapshot(tmp_path, target, seed=4):
    from woof.ingest import hrrr
    row = hi.source_row("hrrr")
    window = hi.source_window_for(target, row)
    rng = np.random.default_rng(seed)
    fields = {}
    specs = {**{name: (50, 1.0, 8.0) for name in row.atmosphere_3d},
             **{name: (50, -30.0, 30.0) for name in row.wind_3d},
             **{name: (1, 1.0, 8.0) for name in row.surface},
             **{name: (1, -30.0, 30.0) for name in row.wind_10m},
             row.soil_temperature: (9, 170.0, 400.0), row.soil_moisture: (9, 0.0, 1.0)}
    specs.update(SPFH=(50, 0.0, 0.09), Q2=(1, 0.0, 0.09),
                 LANDSEA=(1, 0.0, 1.0), XICE=(1, 0.0, 1.0))
    for name, (levels, low, high) in specs.items():
        shape = ((levels,) if levels != 1 else ()) + (window.ny, window.nx)
        values = rng.uniform(low, high, shape).astype("<f4")
        if name in ("LANDSEA", "XICE"):
            values = (values >= 0.5).astype("<f4")
        if name in ("SPFH", "Q2", "SNOW"):
            values.reshape(-1)[::7] = 0.0
        path = tmp_path / f"{name}.f32le"
        values.tofile(path)
        fields[name] = np.memmap(path, dtype="<f4", mode="r", shape=shape)
    snap = hrrr.HrrrNativeSnapshot(valid_time=datetime(2026, 9, 13, 15), forecast_hour=0,
                                  i_start=window.i_start, j_start=window.j_start,
                                  ny=window.ny, nx=window.nx, fields=fields)
    return row, snap


@pytest.mark.parametrize("workers", [1, 8, 16])
def test_real_native_entry_matches_every_met_byte(tmp_path, workers):
    _needs_native()
    from woof.ingest import hrrr, soil
    from woof.ingest.preprocess_backend import resolve_preprocess_backend
    target = hi.LatLonTarget(41.90, -94.85, 0.025, 0.025, 7, 5)
    row, snap = snapshot(tmp_path, target)
    pieces = hi._engine_pieces(row)
    pieces.update(ranges=hrrr._require_source_physical_ranges, interp_nodes=soil._interp_nodes,
                  plan=hrrr._ProjectedCpuPlan, rotation=hrrr._source_window_rotation)
    backend = resolve_preprocess_backend("cpu", workers=workers)
    records, _ = reference_regrid(row, pieces, snap, target,
                                  backend=SimpleNamespace(indexed_donor_interp=False))
    reference = tmp_path / "reference.met"
    hi.write_intermediate(reference, records, valid_time=snap.valid_time, forecast_hour=0,
                          map_source=row.map_source, target=target)
    native = tmp_path / "native.met"
    written, plan = hi.regrid_hour(row, pieces, snap, target, backend=backend, path=native)
    assert native.read_bytes() == reference.read_bytes()
    assert written["sha256"] == hi.sha256_file(reference)
    assert written["field_records"] == len(records)  # every record the reference wrote (the HRRR row grew QNC/QNI/QNR/RH2)
    # Reuse geometry for a second output, including its owned pointer lifetime.
    again, retained = hi.regrid_hour(row, pieces, snap, target, backend=backend,
                                    path=tmp_path / "again.met", plan=plan)
    assert retained is plan and again["sha256"] == written["sha256"]


def test_native_admission_does_not_publish_a_broken_open_water_soil(tmp_path):
    _needs_native()
    from woof.ingest.preprocess_backend import resolve_preprocess_backend
    target = hi.LatLonTarget(41.90, -94.85, 0.025, 0.025, 3, 3)
    row, snap = snapshot(tmp_path, target)
    for name in ("LANDSEA", "XICE"):
        path = Path(snap.fields[name].filename)
        np.zeros(snap.fields[name].shape, dtype="<f4").tofile(path)
    np.full(snap.fields["SOILT"].shape, 160.0, dtype="<f4").tofile(snap.fields["SOILT"].filename)
    output = tmp_path / "bad.met"
    with pytest.raises(hi.IntermediateRefusal, match="broken soil record"):
        hi.regrid_hour(row, hi._engine_pieces(row), snap, target,
                       backend=resolve_preprocess_backend("cpu"), path=output)
    assert not output.exists()


def test_file_metadata_entry_matches_the_oracle_without_python_mapping(monkeypatch, tmp_path):
    _needs_native()
    from woof.ingest import hrrr, soil
    from woof.ingest.preprocess_backend import resolve_preprocess_backend
    target = hi.LatLonTarget(41.90, -94.85, 0.025, 0.025, 3, 3)
    row, reference_snapshot = snapshot(tmp_path, target)
    pieces = hi._engine_pieces(row)
    pieces.update(ranges=hrrr._require_source_physical_ranges, interp_nodes=soil._interp_nodes,
                  plan=hrrr._ProjectedCpuPlan, rotation=hrrr._source_window_rotation)
    records, _ = reference_regrid(row, pieces, reference_snapshot, target,
                                  backend=SimpleNamespace(indexed_donor_interp=False))
    reference = tmp_path / "reference.met"
    hi.write_intermediate(reference, records, valid_time=reference_snapshot.valid_time,
                          forecast_hour=0, map_source=row.map_source, target=target)
    for name, value in reference_snapshot.fields.items():
        role = row.decoded_directories[int(name in (row.soil_temperature, row.soil_moisture))]
        directory = tmp_path / role.format(hour=0)
        directory.mkdir(exist_ok=True)
        os.link(value.filename, directory / f"{name}.f32le")
    files = hi.decoded_window_files(row, tmp_path, reference_snapshot.valid_time, 0,
                                    hi.source_window_for(target, row))
    def forbidden(*args, **kwargs):
        raise AssertionError("production must let Rust open the weather fields")
    monkeypatch.setattr(np, "memmap", forbidden)
    native = tmp_path / "file-metadata.met"
    _, plan = hi.regrid_hour(row, pieces, files, target,
                            backend=resolve_preprocess_backend("cpu"), path=native)
    plan.close()
    assert native.read_bytes() == reference.read_bytes()


def test_missing_explicit_native_artifact_cannot_select_another_library(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOF_PROJECTED_INTERMEDIATE_BRIDGE", str(tmp_path / "absent.so"))
    with pytest.raises(hi.IntermediateRefusal, match="is not a file"):
        hi._native_projected_entry()


def test_ledger_append_failure_never_raises_even_with_warning_as_error(monkeypatch, tmp_path):
    import warnings
    monkeypatch.setenv("WOOF_TIME_LEDGER", str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hi._ledger_emit(tmp_path / "timeline.jsonl", "prep.intermediate.hour", "mark", {"hour": 0})


def test_c1_event_has_the_frozen_keys_and_keeps_prior_lines(monkeypatch, tmp_path):
    monkeypatch.delenv("WOOF_TIME_LEDGER", raising=False)
    path = tmp_path / "timeline.jsonl"
    path.write_text("prior\n")
    attrs = {"hour": 2, "decode_s": 3.5, "regrid_s": 0.8}
    hi._ledger_emit(path, "prep.intermediate.hour", "mark", attrs)
    lines = path.read_text().splitlines()
    assert lines[0] == "prior"
    event = json.loads(lines[1])
    assert set(event) == {"v", "t_utc", "t_mono", "host", "rental_id", "run_id", "product",
                          "stage", "phase", "card", "pid", "attrs"}
    assert event["v"] == 1 and event["product"] == "hex" and event["attrs"] == attrs


def test_hour_markers_publish_in_order_while_the_next_fetch_is_pending(monkeypatch, tmp_path):
    """Orchestration scheduling check; numerical checks use the real entry above."""
    from threading import Event
    from datetime import timedelta
    ready = [Event() for _ in range(3)]
    row = hi.source_row("hrrr")
    target = hi.LatLonTarget(41.90, -94.85, 0.025, 0.025, 3, 3)
    request = hi.IntermediateRequest(row, tmp_path / "grib", datetime(2026, 9, 13, 15),
                                     (0, 1, 2), target, tmp_path / "out")
    monkeypatch.delenv("WOOF_TIME_LEDGER", raising=False)
    monkeypatch.setattr(hi, "_native_projected_entry", lambda: None)
    def ensure(request, hour, log):
        if hour:
            # The preceding marker releases the simulated slow transfer.
            assert ready[hour-1].wait(10), "decode/regrid did not overlap the next fetch"
            marker = request.out_dir / f"MET:{request.cycle+timedelta(hours=hour-1):%Y-%m-%d_%H}.ready.json"
            assert marker.is_file()
        return request.grib_dir
    class Producer:
        def __init__(self, **kwargs):
            self.output = kwargs["output"]
            self.output.mkdir()
        def start(self): pass
        def wait_preflight(self): pass
        def wait_hour(self, hour): return self.output
        def finish(self): return {"status": "PASS"}
        def cancel(self): pass
    def regrid(row, pieces, snap, target, *, backend, path, plan, **kwargs):
        path.write_bytes(b"verified-test-publication")
        return {"path": str(path), "field_records": 318, "bytes": path.stat().st_size,
                "sha256": hi.sha256_file(path), "levels": 50, "operator": "scheduling-test"}, plan
    monkeypatch.setattr(hi, "_ensure_hour", ensure)
    monkeypatch.setattr(hi, "_admit_hour", lambda *args: {})
    monkeypatch.setattr(hi, "regrid_hour", regrid)
    decoder = tmp_path / "decoder"
    decoder.write_bytes(b"scheduling-test-only")
    monkeypatch.setattr(hi, "resolve_decoder", lambda *args: decoder)
    monkeypatch.setattr(hi, "source_window_for", lambda *args: hi.SourceWindow(0, 8, 0, 8))
    monkeypatch.setattr(hi, "_engine_pieces", lambda row: {"resolve_backend": lambda x, **kwargs: None,
                                                         "load": lambda root, hour: None,
                                                         "producer": Producer})
    # The synthetic transfer writes no GRIB. Predict its configured roots.
    request.grib_dir.mkdir()
    for hour in request.hours:
        for path in row.hour_files(request.grib_dir, request.cycle, hour):
            path.touch()
    def log(message):
        if message.startswith("WROTE"):
            for hour in request.hours:
                marker = request.out_dir / f"MET:{request.cycle+timedelta(hours=hour):%Y-%m-%d_%H}.ready.json"
                if marker.is_file():
                    ready[hour].set()
    receipt = hi.build_intermediates(request, log=log)
    events = [json.loads(line) for line in (request.out_dir / "timeline.jsonl").read_text().splitlines()]
    assert [e["attrs"]["hour"] for e in events if e["phase"] == "mark"] == [0, 1, 2]
    assert [e["phase"] for e in events] == ["begin", "end", "mark"] * 3
    assert all(Path(item["completion_marker"]).is_file() for item in receipt["files"])
