"""woof.hex.column_store: the any-point column store of a hex run, every output hour.

Breakage it prevents: a hex run whose soundings have no store for its last hour (a run ending off the hour), a
surface export that wrote no store though its frames carry the column, a store failure that fails the door, and a
reader that cannot tell which grid a view's store holds.
"""
from __future__ import annotations

from pathlib import Path

from woof.hex import column_store


def test_whole_hours_and_the_last_frame():
    frames = [Path(f"/x/wrfout_d01_2024-04-26_{h:02d}_{m:02d}_00") for h in (12, 13) for m in (0, 30)]
    picked = [f.name for f in column_store.chosen(frames)]
    assert picked == ["wrfout_d01_2024-04-26_12_00_00", "wrfout_d01_2024-04-26_13_00_00",
                      "wrfout_d01_2024-04-26_13_30_00"]


def test_switched_off_writes_nothing(monkeypatch):
    monkeypatch.setenv("WOOF_COLUMNS", "off")
    assert column_store.write([Path("/x/wrfout_d01_2024-04-26_12_00_00")], Path("/o"),
                              say=lambda *a, **k: None)["skipped"] == "WOOF_COLUMNS=off"


def test_a_surface_export_writes_its_store(monkeypatch, tmp_path):
    """The surface set carries every field a sounding reads (T, P, PB, QVAPOR, PH, PHB, U, V, W). The refusal that
    said it "carries no 3-D fields for soundings" was untrue on every woof this package runs on, and kept the
    surface export (the WOOF site's, every ensemble member's) from writing the store its frames make."""
    import woof.soundings as soundings

    def fake(frame, out):
        path = Path(out) / "d01" / "20240426T120000Z.wcol"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RWSTORE2")
        return [{"path": str(path), "bytes": 8}]
    monkeypatch.setattr(soundings, "write_columns", fake)
    receipt = column_store.write([tmp_path / "wrfout_d01_2024-04-26_12_00_00"], tmp_path / "columns",
                                 field_set="surface", say=lambda *a, **k: None)
    assert (receipt["files"], receipt["field_set"]) == (1, "surface")


def test_the_columns_index_lists_every_views_store_finest_first(tmp_path):
    """Every view's store sits under its own view folder but names its grid d01 (the frame file's own name), so a
    reader that filed stores by that name took a 96 -> 3 km zoom globe's ~25 km globe for its 3 km core (WOOF
    production, 2026-10-28). The index names each view's store, its spacing and its window, finest first."""
    import json

    from woof.hex.products_door import write_columns_index

    case = tmp_path / "case"
    globe = {"kind": "latlon", "spacing_degrees": 0.2293, "west": -180.0, "south": -90.0, "east": 180.0, "north": 90.0}
    core = {"kind": "lambert", "dx_metres": 3021.0, "centre_lat": 35.5, "centre_lon": -97.5, "nx": 300, "ny": 260}
    for view in ("global", "refined-core"):
        for hour in (18, 19):
            path = case / view / "columns" / "d01" / f"20250519T{hour}0000Z.wcol"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"RWSTORE2" + view.encode())
    # The core was drawn live: its window is in the render manifest beside its maps, not in this pass's record.
    (case / "refined-core" / "render-manifest.json").write_text(json.dumps({"convert": {"window_spec": core}}))
    views = [{"view": "global", "window": "global", "window_spec": globe}, {"view": "refined-core", "window": "mesh"},
             {"view": "points"}, {"view": "event", "window": "region:35.5,-97.5,1000,1000"}]
    doc = json.loads(write_columns_index(case, views).read_text())
    assert [(v["view"], v["rank"], v["spacing_km"]) for v in doc["views"]] == [("refined-core", 0, 3.021),
                                                                               ("global", 1, 25.497)]
    assert doc["views"][0]["folder"] == "refined-core/columns/d01"
    assert doc["views"][0]["valid_times"] == ["2025-05-19T18:00:00Z", "2025-05-19T19:00:00Z"]
    assert doc["views"][0]["centre"]["lat"] == 35.5 and doc["views"][0]["bounds"] is None
    assert doc["views"][1]["bounds"] == [-180.0, -90.0, 180.0, 90.0]
    assert write_columns_index(tmp_path / "nothing", views) is None


def test_a_frame_the_live_loop_stored_is_kept_not_written_again(monkeypatch, tmp_path):
    """The zoom globe's products pass converts every frame again for its whole-run maps; the frames whose store the live
    loop wrote keep it (RESULT.md defect 3: 162 s after a 3 h 3 km zoom globe, 15-20 min after a 24 h one)."""
    import woof.soundings as soundings

    calls = []

    def fake(frame, out):
        m = column_store._FRAME.match(Path(frame).name)
        path = Path(out) / m[1] / f"{m[2]}{m[3]}{m[4]}T{m[5]}{m[6]}{m[7]}Z.wcol"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RWSTORE2")
        calls.append(Path(frame).name)
        return [{"path": str(path), "bytes": 8}]
    monkeypatch.setattr(soundings, "write_columns", fake)
    frames = [tmp_path / f"wrfout_d01_2025-05-19_{h:02d}_00_00" for h in (18, 19, 20, 21)]
    out = tmp_path / "columns"
    # The live loop: one frame at a time as each lands.
    for frame in frames[:3]:
        assert column_store.write([frame], out, say=lambda *a, **k: None)["files"] == 1
    # The products pass: every frame; only the one the live loop did not reach is written.
    calls.clear()
    receipt = column_store.write(frames, out, say=lambda *a, **k: None)
    assert calls == ["wrfout_d01_2025-05-19_21_00_00"]
    assert (receipt["files"], receipt["kept"]) == (1, 3)
    # Everything stored already: nothing written, and the writer is not even loaded.
    calls.clear()
    assert column_store.write(frames, out, say=lambda *a, **k: None)["kept"] == 4 and calls == []
    # An empty file (never left by rw-store, which renames whole files in) is written again.
    (out / "d01" / "20250519T180000Z.wcol").write_bytes(b"")
    column_store.write(frames, out, say=lambda *a, **k: None)
    assert calls == ["wrfout_d01_2025-05-19_18_00_00"]


def test_a_frame_that_fails_is_named_and_the_door_goes_on(monkeypatch):
    import woof.soundings as soundings

    def boom(frame, out):
        raise RuntimeError("column store not written: no PH")
    monkeypatch.setattr(soundings, "write_columns", boom)
    receipt = column_store.write([Path("/x/wrfout_d01_2024-04-26_12_00_00")], Path("/o"), say=lambda *a, **k: None)
    assert receipt["files"] == 0 and "no PH" in receipt["failed"][0]["error"]
