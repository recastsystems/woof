"""A resumed HRRR fetch lands every hour of a product in one byte mode.

Breakage this prevents (updraft factory, 2026-10-27, 2023-08-07 and
2019-06-09): ``woof fetch --mode idx-subset --analyzed-aerosol`` resuming
a folder an interrupted full-file fetch had started kept the whole files
and fetched only the missing hours as subsets.  A whole wrfnat carries
PBLH/LCDC/MCDC/HCDC, which hrrr_grib2_bridge selects wherever they are
published, so the preparation refused the series 80-146 s in ("f16
selected inventory does not exactly equal the first series frame").
The run's own fetch (default full-file) then relabelled the folder
"full-file".
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from woof import fetch
from tools import download_hrrr_native_subset as hrrr_transport

#: Census of a whole object, well above any subset selection.
WHOLE = {"atmosphere": 1136, "soil": 743}
CYCLE = datetime(2023, 8, 7, 12)


def _grib2_stream(messages: int) -> bytes:
    one = (b"GRIB" + b"\x00\x00" + b"\x00" + b"\x02"
           + (20).to_bytes(8, "big") + b"7777")
    return one * messages


def _install_backbone(monkeypatch, calls: list[tuple[int, str, str]],
                      whole_hours=()):
    """A backbone that lands what the mode asks: whole or subset.

    ``whole_hours``: hours whose index is absent, so ``auto`` lands them
    whole, as the real probe does.
    """

    def fake(*, cycle, hour, kind, mode, out, analyzed_aerosol=False,
             **_ignored):
        calls.append((hour, kind, mode))
        if mode == "auto":
            mode = "full-file" if hour in whole_hours else "idx-subset"
        if mode == "full-file":
            records = WHOLE[kind]
        elif kind == "atmosphere":
            records = hrrr_transport.atmosphere_record_count(
                analyzed_aerosol=analyzed_aerosol)
        else:
            records = hrrr_transport.SOIL_RECORD_COUNT
        name = (f"hrrr.t{cycle:%H}z.wrfnatf{hour:02d}.grib2"
                if kind == "atmosphere"
                else f"hrrr.t{cycle:%H}z.wrfprsf{hour:02d}.grib2")
        (out / name).write_bytes(_grib2_stream(records))
        return {
            "name": name, "idx_name": None,
            "mode": "full-file" if mode == "full-file" else "idx-subset",
            "mode_reason": "fixture", "bytes": records * 20,
            "wall_seconds": 0.1, "source": "aws",
            "grib_url": f"https://example.invalid/{name}",
            "selected_record_count": records,
            "probe": {"idx_covers_object": False,
                      "idx_record_count": records},
            # A current atmosphere subset names the optional surface
            # records it took (none listed by this fixture's index).
            **({"optional_surface": []}
               if mode != "full-file" and kind == "atmosphere" else {}),
        }

    monkeypatch.setattr(fetch, "_rw_fetch_hrrr", fake)


def _fetch(out: Path, hours, mode: str, **options) -> dict:
    fetch.fetch_hrrr(cycle=CYCLE, hours=tuple(hours), area=None, out=out,
                     transport="s3", engine="rust",
                     engine_bin=Path("rw_fetch"), mode=mode,
                     progress=lambda _line: None, **options)
    return json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())


def test_idx_subset_resume_over_whole_files_lands_one_series(
        tmp_path, monkeypatch):
    from test_fetch import _install_runtime_surface_inputs
    _install_runtime_surface_inputs(monkeypatch)
    calls: list[tuple[int, str, str]] = []
    _install_backbone(monkeypatch, calls)
    out = tmp_path / "hrrr"

    # The interrupted full-file fetch: f00 and f01 whole.
    _fetch(out, (0, 1), "full-file")

    # The resume that mixed them: now every hour is a subset.
    calls.clear()
    manifest = _fetch(out, (0, 1, 2), "idx-subset", analyzed_aerosol=True)
    payload = [f for f in manifest["files"]
               if f["role"] in ("atmosphere", "soil")]
    assert {f["mode"] for f in payload} == {"idx-subset"}
    assert manifest["mode"] == "idx-subset"
    subset = hrrr_transport.atmosphere_record_count(analyzed_aerosol=True)
    assert {f["records"] for f in payload if f["role"] == "atmosphere"} \
        == {subset}
    assert sorted(calls) == [(hour, kind, "idx-subset") for hour in (0, 1, 2)
                             for kind in ("atmosphere", "soil")]
    # The whole files were moved aside, not deleted.
    assert len(list(out.glob("*.rejected-*"))) == 4

    # The run's own fetch (default full-file) reuses the uniform series
    # as it is and says what the files are.
    calls.clear()
    manifest = _fetch(out, (0, 1, 2), "full-file", analyzed_aerosol=True)
    assert calls == []
    assert manifest["mode"] == "idx-subset"
    assert manifest["requested_mode"] == "full-file"


def test_auto_keeps_whole_and_current_subset_hours_side_by_side(
        tmp_path, monkeypatch):
    """The default byte mode: a whole-file fallback hour and subset hours
    carry the same records for the bridge, so the series is not refused
    and a resume moves nothing."""
    from test_fetch import _install_runtime_surface_inputs
    _install_runtime_surface_inputs(monkeypatch)
    calls: list[tuple[int, str, str]] = []
    _install_backbone(monkeypatch, calls, whole_hours=(1,))
    out = tmp_path / "hrrr"

    manifest = _fetch(out, (0, 1, 2), "auto")
    modes = {(f["forecast_hour"], f["role"]): f["mode"]
             for f in manifest["files"] if f["role"] in ("atmosphere", "soil")}
    assert modes[(1, "atmosphere")] == "full-file"
    assert modes[(0, "atmosphere")] == modes[(2, "atmosphere")] == "idx-subset"
    assert all("optional_surface" in f for f in manifest["files"]
               if f["role"] == "atmosphere" and f["mode"] == "idx-subset")

    calls.clear()
    _fetch(out, (0, 1, 2), "auto")
    assert calls == []
    assert not list(out.glob("*.rejected-*"))


def test_a_subset_from_before_the_optional_records_is_fetched_again(
        tmp_path, monkeypatch):
    """A subset whose receipt names no optional surface records predates
    them: beside a current hour it would select a different record set,
    so it is set aside (not deleted) and fetched again."""
    from test_fetch import _install_runtime_surface_inputs
    _install_runtime_surface_inputs(monkeypatch)
    calls: list[tuple[int, str, str]] = []
    _install_backbone(monkeypatch, calls)
    out = tmp_path / "hrrr"
    _fetch(out, (0, 1), "auto")
    receipt = out / fetch.FETCH_MANIFEST_NAME
    old = json.loads(receipt.read_text())
    for entry in old["files"]:
        entry.pop("optional_surface", None)
    receipt.write_text(json.dumps(old))

    calls.clear()
    manifest = _fetch(out, (0, 1, 2), "auto")
    assert sorted(hour for hour, kind, _mode in calls
                  if kind == "atmosphere") == [0, 1, 2]
    assert len(list(out.glob("*wrfnat*.rejected-*"))) == 2
    assert all("optional_surface" in f for f in manifest["files"]
               if f["role"] == "atmosphere")


def test_an_unclaimed_current_subset_is_verified_not_fetched_again(
        tmp_path, monkeypatch):
    """A half-fetched hour no receipt claims keeps its current subset: its
    census equals the required records plus the optional records its kept
    index lists, so it is verified in place rather than downloaded again."""
    from test_fetch import _install_runtime_surface_inputs
    _install_runtime_surface_inputs(monkeypatch)
    calls: list[tuple[int, str, str]] = []
    optional = hrrr_transport.optional_surface_selectors(
        hrrr_transport.OPTIONAL_SURFACE_FIELDS)

    def fake(*, cycle, hour, kind, mode, out, analyzed_aerosol=False,
             **_ignored):
        calls.append((hour, kind, mode))
        if kind == "atmosphere":
            name = f"hrrr.t{cycle:%H}z.wrfnatf{hour:02d}.grib2"
            records = hrrr_transport.ATMOSPHERE_RECORD_COUNT + len(optional)
            (out / f"{name}.idx").write_text("".join(
                f"{n}:{n * 20}:d=2023080712:{selector}:anl:\n"
                for n, selector in enumerate(optional, 1)), encoding="ascii")
        else:
            name = f"hrrr.t{cycle:%H}z.wrfprsf{hour:02d}.grib2"
            records = hrrr_transport.SOIL_RECORD_COUNT
        (out / name).write_bytes(_grib2_stream(records))
        return {"name": name, "idx_name": None, "mode": "idx-subset",
                "mode_reason": "fixture", "bytes": records * 20,
                "wall_seconds": 0.1, "source": "aws",
                "grib_url": f"https://example.invalid/{name}",
                "selected_record_count": records,
                "probe": {"idx_covers_object": True, "idx_record_count": records},
                **({"optional_surface": list(optional)}
                   if kind == "atmosphere" else {})}

    monkeypatch.setattr(fetch, "_rw_fetch_hrrr", fake)
    out = tmp_path / "hrrr"
    _fetch(out, (0, 1), "auto")
    receipt = out / fetch.FETCH_MANIFEST_NAME
    kept = json.loads(receipt.read_text())
    kept["files"] = [f for f in kept["files"] if f.get("forecast_hour") != 1]
    kept["forecast_hours"] = [0]
    receipt.write_text(json.dumps(kept))

    calls.clear()
    manifest = _fetch(out, (0, 1), "auto")
    assert (1, "atmosphere", "auto") not in calls
    assert not list(out.glob("*wrfnat*.rejected-*"))
    hour1 = next(f for f in manifest["files"]
                 if f["forecast_hour"] == 1 and f["role"] == "atmosphere")
    assert hour1["optional_surface"] == list(optional)
