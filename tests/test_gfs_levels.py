"""The GFS pressure ladder follows the case's model top, not a constant.

WOOF-ArWen's GFS route used to fetch a fixed 21-level ladder topping out at
100 hPa.  Nothing said so: the flag did not exist, the manifest did not
record it, and the refusal a user eventually hit came from the vertical
contract three steps later ("source atmosphere stops at 10000 Pa but
requested p_top is 5000 Pa").  Every WRF-Runner GFS namelist declares
``p_top = 5000``, and NOMADS publishes the levels to satisfy it -- so
this was a silent science ceiling, not a syntax difference.

Every gate here runs against the real inventory captured in
``tests/fixtures/gfs-inventory/`` (see its README for provenance).  No
live request is issued.  The one live smoke is opt-in and touches a
single ``.idx`` on S3, never the NOMADS filter.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest

import woof.fetch as fetch
from woof import fetch_bars
from tools import download_gfs_native_subset as gfs_transport

_FIXTURE = (Path(__file__).parent / "fixtures" / "gfs-inventory"
            / "gfs.t12z.pgrb2.0p25.f000.idx")
_AREA = "30,-100,40,-90"
_CYCLE = datetime(2026, 7, 28, 6)


@pytest.fixture(scope="module")
def real_index() -> str:
    return _FIXTURE.read_text(encoding="ascii")


@pytest.fixture(scope="module")
def real_levels(real_index) -> tuple[float, ...]:
    return gfs_transport.available_levels_from_index(real_index)


def _grib2(messages: int) -> bytes:
    one = (b"GRIB" + b"\x00\x00" + b"\x00" + b"\x02"
           + (20).to_bytes(8, "big") + b"7777")
    return one * messages


# ---------------------------------------------------------------------------
# The captured inventory, and the constants taken from it
# ---------------------------------------------------------------------------

def test_the_real_inventory_publishes_every_field_on_every_level(real_index):
    """A level is only usable when all five 3-D fields sit on it."""

    per_field: dict[str, set[float]] = {
        name: set() for name in gfs_transport.PRESSURE_FIELDS}
    for line in real_index.splitlines():
        columns = line.split(":")
        variable, level = columns[3], columns[4]
        if variable in per_field and level.endswith(" mb"):
            per_field[variable].add(float(level[:-3]))
    assert len({frozenset(levels) for levels in per_field.values()}) == 1
    assert len(next(iter(per_field.values()))) == 41


def test_the_certified_ladder_matches_the_captured_inventory(real_levels):
    """The fallback constant cannot drift from the file it came from."""

    assert real_levels == tuple(
        float(level)
        for level in gfs_transport.CERTIFIED_AVAILABLE_LEVELS_HPA)


def test_the_certified_ladder_is_the_top_of_the_available_one(real_levels):
    """A deeper top EXTENDS the certified ladder; it never replaces it."""

    certified = tuple(float(level)
                      for level in gfs_transport.PRESSURE_LEVELS_HPA)
    assert real_levels[-len(certified):] == certified


def test_the_record_bar_is_linear_in_the_ladder_length():
    assert gfs_transport.record_count_for_levels(
        len(gfs_transport.PRESSURE_LEVELS_HPA)) == \
        fetch_bars.CERTIFIED_RECORD_BARS["gfs"] == 124
    assert gfs_transport.record_count_for_levels(23) == 134
    assert gfs_transport.record_count_for_levels(41) == 224


def test_the_derived_bar_agrees_with_the_real_index(real_index, real_levels):
    """Both sides of the bar, measured against one real inventory."""

    for levels in (gfs_transport.PRESSURE_LEVELS_HPA,
                   gfs_transport.levels_for_top(5000.0,
                                                available=real_levels),
                   real_levels):
        derived = fetch.gfs_derived_record_bar(
            _CYCLE, progress=lambda line: None, levels_hpa=levels,
            index_text=real_index)
        assert derived == gfs_transport.record_count_for_levels(len(levels)), \
            levels


# ---------------------------------------------------------------------------
# Level selection: the dimension, at more than one value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("top_pa,expected_levels,expected_top_hpa", [
    # The certified ladder already reaches a 100 hPa-class top: byte for
    # byte the request ArWen has always made.
    (None, 21, 100.0),
    (10000.0, 22, 70.0),
    (20000.0, 21, 100.0),
    # A 50 hPa-class top -- what every WRF-Runner GFS namelist declares.
    # 70 hPa is not low enough, 50 hPa is: exactly the 23-level request
    # his own generator makes.
    (5000.0, 24, 40.0),
    (7000.0, 23, 50.0),
    # Deeper still, walking the published ladder.
    (1000.0, 29, 7.0),
    (100.0, 34, 0.7),
    (2.0, 41, 0.01),
])
def test_the_ladder_follows_the_requested_top(real_levels, top_pa,
                                              expected_levels,
                                              expected_top_hpa):
    levels = gfs_transport.levels_for_top(top_pa, available=real_levels)
    assert len(levels) == expected_levels
    assert levels[0] == expected_top_hpa
    # Whatever the top, the ladder covers it and stays sorted and unique.
    if top_pa is not None:
        assert levels[0] * 100.0 < top_pa
    assert list(levels) == sorted(levels)
    assert len(set(levels)) == len(levels)
    # And the certified ladder is still entirely inside it.
    assert levels[-21:] == tuple(
        float(level) for level in gfs_transport.PRESSURE_LEVELS_HPA)


def test_a_50_hpa_top_adds_a_level_strictly_above_the_model_top(
        real_levels):
    """The reconciliation delta, stated as a gate.

    WRF-Runner's gfs.py requests five fields on 23 pressure levels with
    p_top = 5000; WOOF-ArWen requested 21 and stopped at 10000.  The two lists
    now agree.
    """

    levels = gfs_transport.levels_for_top(5000.0, available=real_levels)
    certified = tuple(float(level)
                      for level in gfs_transport.PRESSURE_LEVELS_HPA)
    assert set(levels) - set(certified) == {40.0, 50.0, 70.0}
    assert len(levels) == 24


def test_an_unsatisfiable_top_refuses_and_names_the_deepest_available(
        real_levels):
    """The refusal has to be useful: say how far up the source reaches."""

    with pytest.raises(gfs_transport.TopPressureUnavailable) as caught:
        gfs_transport.levels_for_top(0.5, available=real_levels)
    message = str(caught.value)
    assert "0.01 hPa" in message and "1 Pa" in message
    assert "HGT, TMP, RH, UGRD, VGRD" in message
    assert "--p-top-pa" in message


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_nonsense_top_is_refused_as_such(real_levels, bad):
    with pytest.raises(ValueError, match="positive pressure"):
        gfs_transport.levels_for_top(bad, available=real_levels)


def test_an_index_that_names_no_complete_level_yields_nothing():
    """Four fifths of a level is not a level."""

    partial = "\n".join([
        "1:0:d=2026073012:HGT:50 mb:anl:",
        "2:100:d=2026073012:TMP:50 mb:anl:",
        "3:200:d=2026073012:RH:50 mb:anl:",
        "4:300:d=2026073012:UGRD:50 mb:anl:",
        # VGRD absent at 50 mb
    ])
    assert gfs_transport.available_levels_from_index(partial) == ()


def test_a_partly_published_level_is_dropped_from_a_real_index(real_index):
    kept = [line for line in real_index.splitlines()
            if not (line.split(":")[3] == "VGRD"
                    and line.split(":")[4] == "50 mb")]
    levels = gfs_transport.available_levels_from_index("\n".join(kept))
    assert 50.0 not in levels
    assert 70.0 in levels and 100.0 in levels


# ---------------------------------------------------------------------------
# The query the transport actually builds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("top_pa,wanted", [
    (None, {"lev_100_mb", "lev_1000_mb"}),
    (5000.0, {"lev_50_mb", "lev_70_mb", "lev_100_mb", "lev_1000_mb"}),
    (100.0, {"lev_1_mb", "lev_2_mb", "lev_50_mb", "lev_1000_mb"}),
])
def test_the_nomads_query_names_every_selected_level(real_levels, top_pa,
                                                     wanted):
    levels = gfs_transport.levels_for_top(top_pa, available=real_levels)
    url = gfs_transport.nomads_query(
        _CYCLE, 0, left_lon=260.0, right_lon=270.0, bottom_lat=30.0,
        top_lat=40.0, pressure_levels_hpa=levels)
    query = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
    assert wanted <= set(query)
    named = {key for key in query if key.startswith("lev_")
             and key.endswith("_mb")}
    assert len(named) == len(levels)
    # The filter spells levels without trailing zeros; "lev_100.0_mb" is
    # a level NOMADS does not have and would silently drop.
    assert not [key for key in named if ".0_mb" in key]


def test_sub_hpa_levels_keep_their_decimal_spelling(real_levels):
    levels = gfs_transport.levels_for_top(2.0, available=real_levels)
    url = gfs_transport.nomads_query(
        _CYCLE, 0, left_lon=260.0, right_lon=270.0, bottom_lat=30.0,
        top_lat=40.0, pressure_levels_hpa=levels)
    query = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
    for key in ("lev_0.01_mb", "lev_0.07_mb", "lev_0.4_mb"):
        assert key in query, key


# ---------------------------------------------------------------------------
# End to end through fetch_gfs: the ladder reaches the receipt
# ---------------------------------------------------------------------------

def _fetch(tmp_path, monkeypatch, real_index, real_levels, **kwargs):
    urls: list[str] = []
    levels_seen: list[int] = []

    def download(url, destination, **_kw):
        urls.append(url)
        query = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
        count = len([key for key in query
                     if key.startswith("lev_") and key.endswith("_mb")])
        levels_seen.append(count)
        destination.write_bytes(_grib2(
            gfs_transport.record_count_for_levels(count)))

    monkeypatch.setattr(gfs_transport, "_download", download)
    manifest_path = fetch.fetch_gfs(
        cycle=_CYCLE, hours=(0, 3), area=fetch.parse_area(_AREA),
        out=tmp_path / "gfs", progress=lambda line: None,
        derived_bar=lambda cycle, **kw: fetch.gfs_derived_record_bar(
            cycle, index_text=real_index,
            levels_hpa=kw.get("levels_hpa"),
            progress=lambda line: None),
        available_levels=lambda cycle, **kw: real_levels,
        **kwargs)
    return json.loads(manifest_path.read_text()), urls, levels_seen


@pytest.mark.parametrize("top_pa,levels,top_hpa", [
    (None, 21, 100.0), (10000.0, 22, 70.0), (5000.0, 24, 40.0),
    (1000.0, 29, 7.0),
])
def test_the_receipt_records_the_ladder_and_the_source_top(
        tmp_path, monkeypatch, real_index, real_levels, top_pa, levels,
        top_hpa):
    manifest, _urls, seen = _fetch(
        tmp_path, monkeypatch, real_index, real_levels,
        top_pressure_pa=top_pa)

    assert seen == [levels, levels]
    assert len(manifest["pressure_levels_hpa"]) == levels
    assert min(manifest["pressure_levels_hpa"]) == top_hpa
    assert manifest["source_top_pressure_pa"] == top_hpa * 100.0
    # The record bar is this request's, and it is recorded as certified
    # rather than as an accepted upstream change.
    (bar,) = manifest["record_bars"]
    assert bar["expected"] == gfs_transport.record_count_for_levels(levels)
    assert bar["inventory_change_accepted"] is False


def test_all_levels_takes_the_whole_published_ladder(
        tmp_path, monkeypatch, real_index, real_levels):
    manifest, _urls, seen = _fetch(
        tmp_path, monkeypatch, real_index, real_levels, all_levels=True)
    assert seen == [41, 41]
    assert manifest["source_top_pressure_pa"] == 1.0
    assert len(manifest["pressure_levels_hpa"]) == 41


def test_the_default_request_is_unchanged_byte_for_byte(
        tmp_path, monkeypatch, real_index, real_levels):
    """The whole point of extending rather than replacing."""

    _manifest, urls, _seen = _fetch(
        tmp_path, monkeypatch, real_index, real_levels)
    query = dict(parse_qsl(urlsplit(urls[0]).query, keep_blank_values=True))
    named = sorted(key for key in query if key.startswith("lev_")
                   and key.endswith("_mb"))
    assert named == sorted(
        f"lev_{level}_mb" for level in gfs_transport.PRESSURE_LEVELS_HPA)


def test_a_top_and_all_levels_together_are_refused(tmp_path, monkeypatch,
                                                   real_index, real_levels):
    with pytest.raises(ValueError, match="two answers to the same question"):
        _fetch(tmp_path, monkeypatch, real_index, real_levels,
               top_pressure_pa=5000.0, all_levels=True)


def test_an_unsatisfiable_top_refuses_before_any_download(
        tmp_path, monkeypatch, real_index, real_levels):
    def refuse(url, destination, **_kw):
        raise AssertionError("downloaded despite an unsatisfiable top")

    monkeypatch.setattr(gfs_transport, "_download", refuse)
    with pytest.raises(gfs_transport.TopPressureUnavailable,
                       match="deepest this product publishes"):
        fetch.fetch_gfs(
            cycle=_CYCLE, hours=(0,), area=fetch.parse_area(_AREA),
            out=tmp_path / "gfs", progress=lambda line: None,
            top_pressure_pa=0.5,
            derived_bar=lambda cycle, **kw: 124,
            available_levels=lambda cycle, **kw: real_levels)


def test_the_certified_ladder_stands_in_when_the_index_is_unreadable(
        tmp_path, monkeypatch):
    """A transient S3 blip must not silently shrink the ladder."""

    monkeypatch.setattr(fetch, "gfs_live_index",
                        lambda cycle, **kw: None)
    lines: list[str] = []
    levels = fetch.gfs_available_levels(_CYCLE, progress=lines.append)
    assert levels == gfs_transport.CERTIFIED_AVAILABLE_LEVELS_HPA
    assert gfs_transport.levels_for_top(5000.0, available=levels)[0] == 40.0


# ---------------------------------------------------------------------------
# The front door carries the ladder to the ingest side
# ---------------------------------------------------------------------------

def test_the_front_door_manifest_declares_the_ladder_it_fetched(
        tmp_path, monkeypatch, real_index, real_levels):
    from woof import gfs_direct

    out = tmp_path / "gfs"
    _manifest, _urls, _seen = _fetch(
        tmp_path, monkeypatch, real_index, real_levels,
        top_pressure_pa=5000.0)
    for name in ("bridge", "namelist.wps", "experiment.toml"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    path, _digest = fetch.author_gfs_front_door_manifest(
        out=out, bridge=tmp_path / "bridge",
        wps_namelist=tmp_path / "namelist.wps",
        experiment_config=tmp_path / "experiment.toml",
        progress=lambda line: None)
    identity = json.loads(path.read_text())["source"]
    assert len(identity["pressure_levels_hpa"]) == 24
    assert identity["top_pressure_pa"] == 4000.0
    # And the front door reads that top instead of its own constant --
    # which is what let a p_top of 5000 Pa through at all.
    assert gfs_direct._manifest_source_top_pa(
        {"source": identity}) == 4000.0


def test_a_receipt_without_a_ladder_still_means_the_certified_top():
    """Directories fetched before the ladder was recorded keep working."""

    from woof import gfs_direct

    assert gfs_direct._manifest_source_top_pa({"source": {}}) == 10000.0
    assert gfs_direct._manifest_source_top_pa({}) == 10000.0


def test_the_reconciliation_delta_itself(real_levels):
    """The science difference, stated where it is decided.

    WRF-Runner's GFS namelists declare ``p_top_requested = 5000``.  The
    vertical contract compares that against the source top, and the
    source top is decided by the fetched ladder.  Under the old fixed
    ladder it was 10000 Pa and the run was refused -- the interop
    verification's delta 9, the one reconciliation that changed science
    rather than syntax.  Under a ladder fetched for that top it is 5000
    Pa and the same eta grid validates.
    """

    import numpy as np

    from woof.vertical_contract import validate_explicit_eta_grid

    eta = np.linspace(1.0, 0.0, 81, dtype=np.float64)
    old_top = min(gfs_transport.PRESSURE_LEVELS_HPA) * 100.0
    assert old_top == 10000.0
    with pytest.raises(ValueError,
                       match="source atmosphere stops at 10000 Pa"):
        validate_explicit_eta_grid(
            eta, nz=80, p_top=5000.0, source_top_pressure_pa=old_top,
            context="the ladder WOOF-ArWen used to be able to fetch")

    new_top = min(gfs_transport.levels_for_top(
        5000.0, available=real_levels)) * 100.0
    assert new_top == 4000.0
    checked = validate_explicit_eta_grid(
        eta, nz=80, p_top=5000.0, source_top_pressure_pa=new_top,
        context="the ladder --p-top-pa 5000 fetches")
    assert np.array_equal(checked, eta)


_CERTIFIED = [float(level) for level in gfs_transport.PRESSURE_LEVELS_HPA]


@pytest.mark.parametrize("ladder,message", [
    # Shorter than the certified ladder: incompleteness, not a request.
    ([70.0, 100.0, 150.0], "fewer than"),
    # A level inserted inside the certified ladder is a different ladder.
    ([50.0, 100.0, 125.0] + _CERTIFIED[1:], "does not end with"),
    # A certified level dropped from the middle.
    ([50.0, 70.0] + _CERTIFIED[:8] + _CERTIFIED[9:], "does not end with"),
    # A level appended below the bottom.
    (_CERTIFIED + [1100.0], "does not end with"),
    # Out of order, and duplicated.
    ([100.0, 50.0] + _CERTIFIED[1:], "not strictly increasing"),
    ([50.0, 50.0] + _CERTIFIED, "not strictly increasing"),
])
def test_the_front_door_refuses_a_ladder_that_is_not_an_extension(
        ladder, message):
    from woof import gfs_direct

    with pytest.raises(ValueError, match=message):
        gfs_direct._manifest_source_top_pa(
            {"source": {"pressure_levels_hpa": ladder}})


@pytest.mark.parametrize("ladder,top", [
    (_CERTIFIED, 10000.0),
    ([50.0, 70.0] + _CERTIFIED, 5000.0),
    ([0.01, 0.02] + [50.0, 70.0] + _CERTIFIED, 1.0),
])
def test_the_front_door_accepts_every_upward_extension(ladder, top):
    from woof import gfs_direct

    assert gfs_direct._manifest_source_top_pa(
        {"source": {"pressure_levels_hpa": ladder}}) == top


def test_a_declared_top_that_contradicts_the_ladder_is_refused():
    from woof import gfs_direct

    ladder = [float(v) for v in gfs_transport.PRESSURE_LEVELS_HPA]
    with pytest.raises(ValueError, match="declares a source top"):
        gfs_direct._manifest_source_top_pa({"source": {
            "pressure_levels_hpa": ladder, "top_pressure_pa": 5000.0}})


# ---------------------------------------------------------------------------
# The decode side: a deeper ladder survives the bridge handoff
# ---------------------------------------------------------------------------

_TWO_D_NAMES = (
    "PSFC", "SOURCE_OROGRAPHY", "SKINTEMP", "SNOW", "SNOWH",
    "T2", "RH2", "U10", "V10", "LANDSEA", "XICE",
    "GFS_ST000010", "GFS_ST010040", "GFS_ST040100", "GFS_ST100200",
    "GFS_SM000010", "GFS_SM010040", "GFS_SM040100", "GFS_SM100200",
)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bridge_output(root: Path, levels_hpa: tuple[float, ...],
                   hours: tuple[int, ...] = (0, 3)) -> Path:
    """A synthetic decode of ``levels_hpa``, shaped as the bridge writes it."""

    import numpy as np

    root.mkdir(parents=True)
    for hour in hours:
        time_root = root / f"f{hour:03}"
        time_root.mkdir()
        base = np.tile(np.arange(4, dtype="<f4"), (2, 1))
        for name in ("GHT", "T", "RH", "U", "V"):
            np.tile(base, (len(levels_hpa), 1, 1)).tofile(
                time_root / f"{name}.f32le")
        for name in _TWO_D_NAMES:
            base.tofile(time_root / f"{name}.f32le")
    inventory = root / "inventory.tsv"
    inventory.write_text("test inventory\n", encoding="utf-8")
    decoded_manifest = root / "decoded-sha256.tsv"
    lines = ["hour\tvariable\tbytes\tsha256\tfilename"]
    for hour in hours:
        for name in ("GHT", "T", "RH", "U", "V") + _TWO_D_NAMES:
            relative = f"f{hour:03d}/{name}.f32le"
            path = root / relative
            lines.append(f"{hour}\t{name}\t{path.stat().st_size}\t"
                         f"{_sha256(path)}\t{relative}")
    decoded_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dummy = root.parent / "dummy.grib2"
    dummy.write_bytes(b"GRIB")
    source_digest = _sha256(dummy)
    (root / "gate.tsv").write_text(
        "status\tPASS\n"
        "schema\tgpuwm-gfs-grib2-bridge-v1\n"
        "cycle\t2026-07-20 00:00:00\n"
        "forecast_hours\t" + ",".join(str(hour) for hour in hours) + "\n"
        "pressure_levels_pa\t"
        + ",".join(str(int(round(level * 100))) for level in levels_hpa)
        + "\n"
        "nx\t4\nny\t2\nlat1\t20\nlon1\t250\ndx\t0.25\ndy\t0.25\n"
        "lat2\t20.25\nlon2\t250.75\nnum_data_points\t8\n"
        "scan_mode\t0x40\n"
        "originating_center\t7\nmaster_table_version\t2\n"
        "local_table_version\t1\n"
        "forecast_generating_process_ids\t"
        + ",".join(f"{hour}:{81 if hour == 0 else 96}" for hour in hours)
        + "\n"
        "land_mask_parameter\t0\n"
        "invariant_fields\tSOURCE_OROGRAPHY,LANDSEA\n"
        "invariant_fingerprint_fnv1a64\t"
        "SOURCE_OROGRAPHY:abc,LANDSEA:def\n"
        + "source_sha256\t"
        + ",".join(f"{hour}:{source_digest}" for hour in hours) + "\n"
        + f"inventory_sha256\t{_sha256(inventory)}\n"
        + f"decoded_manifest_sha256\t{_sha256(decoded_manifest)}\n",
        encoding="utf-8")
    return dummy


@pytest.mark.parametrize("top_pa,expected", [
    (None, 21), (10000.0, 22), (5000.0, 24), (1000.0, 29),
])
def test_the_bridge_loader_carries_whatever_ladder_was_decoded(
        tmp_path, real_levels, top_pa, expected):
    """The seam the 100 hPa ceiling actually lived at.

    The loader used to assert the gate's ladder EQUALLED its own
    21-level constant, so a correctly fetched and correctly decoded
    23-level column was refused at the last step.  It now takes the
    ladder the bridge reports, shapes the arrays from it, and hands it
    to the snapshot the vertical interpolation reads.
    """

    from woof.gfs_direct import _load_bridge_snapshots

    levels = gfs_transport.levels_for_top(top_pa, available=real_levels)
    assert len(levels) == expected
    root = tmp_path / "decoded"
    dummy = _bridge_output(root, levels)
    snapshots = _load_bridge_snapshots(
        root, datetime(2026, 7, 20), ((0, dummy), (3, dummy)),
        expected_levels_hpa=levels)

    assert len(snapshots) == 2
    for snapshot in snapshots:
        assert tuple(snapshot.levels_hpa) == levels
        assert snapshot.fields["T"].shape == (expected, 2, 4)
        # The source top the vertical contract compares p_top against.
        assert float(min(snapshot.levels_hpa)) * 100.0 == levels[0] * 100.0


def test_the_loader_refuses_a_decode_that_is_not_what_was_fetched(
        tmp_path, real_levels):
    """A bridge that decoded a different ladder than the receipt declares
    is a mismatch between what was verified and what was read."""

    from woof.gfs_direct import _load_bridge_snapshots

    fetched = gfs_transport.levels_for_top(5000.0, available=real_levels)
    decoded = gfs_transport.levels_for_top(7000.0, available=real_levels)
    root = tmp_path / "decoded"
    dummy = _bridge_output(root, decoded)
    with pytest.raises(ValueError, match="input manifest does not declare"):
        _load_bridge_snapshots(
            root, datetime(2026, 7, 20), ((0, dummy), (3, dummy)),
            expected_levels_hpa=fetched)


def test_the_loader_refuses_a_gate_ladder_that_is_not_an_extension(tmp_path):
    from woof.gfs_direct import _load_bridge_snapshots

    root = tmp_path / "decoded"
    dummy = _bridge_output(root, tuple(_CERTIFIED[:-1] + [1100.0]))
    with pytest.raises(ValueError, match="does not end with the certified"):
        _load_bridge_snapshots(
            root, datetime(2026, 7, 20), ((0, dummy), (3, dummy)))


# ---------------------------------------------------------------------------
# Opt-in live smoke: one .idx on S3, never the NOMADS filter
# ---------------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.skipif(os.environ.get("WOOF_NETWORK_TESTS") != "1",
                    reason="set WOOF_NETWORK_TESTS=1 for the live smoke")
def test_live_inventory_still_publishes_the_certified_ladder():
    """One small S3 index; the rate-limited NOMADS filter is untouched."""

    cycle = fetch.resolve_latest_cycle("gfs", 0)
    levels = fetch.gfs_available_levels(cycle, progress=lambda line: None)
    certified = tuple(float(level)
                      for level in gfs_transport.PRESSURE_LEVELS_HPA)
    assert levels[-len(certified):] == certified
    assert gfs_transport.levels_for_top(5000.0, available=levels)[0] <= 50.0
