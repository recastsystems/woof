"""The Rust fetch backbone, the probe rule, and the record-count bars.

Three contracts are pinned here.

**Provider name aliases.** NOMADS spells HRRR hybrid cloud water
``CLWMR``; AWS S3 spells the same record ``CLMR``.  Both index forms
have to satisfy the same 561-record selection, because that is a
spelling difference and not an inventory difference.

**The tripwire.** A record *count* that moves is the opposite case: the
fetch stops and names both numbers until an operator accepts the change
in one flag, at which point the live count becomes the bar and the
manifest records that it happened.

**The transport fallback.** A host publishing an inventory this WOOF
does not recognise costs that host, not the run: ``--transport auto``
says why and moves on.

One optional live smoke (``WOOF_NETWORK_TESTS=1``) drives the built
binary against a real object.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from woof import cli
from woof import fetch
from woof import fetch_bars
from woof import rustwx_fetch
from woof.fetch_bars import ACCEPT_FLAG, CERTIFIED_RECORD_BARS, resolve_bar
from tools import download_hrrr_native_subset as hrrr_transport


# ---------------------------------------------------------------------------
# Synthetic HRRR indexes in both provider spellings
# ---------------------------------------------------------------------------

def _grib2_stream(messages: int) -> bytes:
    one = (b"GRIB" + b"\x00\x00" + b"\x00" + b"\x02"
           + (20).to_bytes(8, "big") + b"7777")
    return one * messages


def _atmosphere_index(*, cloud_water_name: str = "CLMR",
                      cloud_ice_name: str = "CIMIXR",
                      extra: tuple[tuple[str, str], ...] = (),
                      ) -> tuple[str, int]:
    """A complete wrfnat index in one provider's vocabulary.

    Returns ``(text, object_bytes)``.  Every record is 100 bytes wide,
    which is enough for the offset monotonicity and end-of-object rules
    the strict parsers apply; the payload itself is never read here.
    """

    rows: list[tuple[str, str]] = []
    for role in hrrr_transport.HYBRID_FIELDS:
        name = {"CLMR": cloud_water_name,
                "CIMIXR": cloud_ice_name}.get(role, role)
        for level in range(1, 51):
            rows.append((name, f"{level} hybrid level"))
    rows.extend(hrrr_transport.SURFACE_FIELDS)
    # HRRR really does publish an accumulated twin of a surface field
    # beside the instantaneous one; the selection must not take both.
    rows.append(("WEASD", "surface"))
    rows.extend(extra)
    lines = []
    for index, (variable, level) in enumerate(rows, 1):
        forecast = ("0-1 hr acc fcst"
                    if index == len(rows) - len(extra) else "anl")
        lines.append(f"{index}:{(index - 1) * 100}:d=2026072812:"
                     f"{variable}:{level}:{forecast}:")
    return "\n".join(lines) + "\n", len(rows) * 100


def _soil_index() -> tuple[str, int]:
    rows = [(variable, level)
            for variable in ("TSOIL", "SOILW")
            for level in hrrr_transport.SOIL_LEVELS]
    lines = [f"{index}:{(index - 1) * 100}:d=2026072812:{variable}:{level}:anl:"
             for index, (variable, level) in enumerate(rows, 1)]
    return "\n".join(lines) + "\n", len(rows) * 100


# ---------------------------------------------------------------------------
# Provider name aliases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spelling", ["CLMR", "CLWMR"])
def test_atmosphere_selection_accepts_either_cloud_water_spelling(spelling):
    """The pilot's finding: one index says CLMR, the other CLWMR."""

    text, size = _atmosphere_index(cloud_water_name=spelling)
    rows = hrrr_transport._parse_index(text.encode("ascii"), size)
    selected = hrrr_transport._atmosphere_selection(rows)
    assert len(selected) == hrrr_transport.ATMOSPHERE_RECORD_COUNT == 561
    chosen = {rows[index].variable for index in selected}
    assert spelling in chosen
    # The accumulated twin is excluded, which is what keeps the count at
    # 561 even though the index carries a 562nd WEASD:surface row.
    assert sum(1 for index in selected
               if rows[index].variable == "WEASD") == 1


@pytest.mark.parametrize("spelling", ["CLMR", "CLWMR"])
def test_selectors_count_the_same_in_either_index(spelling):
    """``atmosphere_selectors()`` is what the Rust backbone is given."""

    text, _ = _atmosphere_index(cloud_water_name=spelling)
    observed = fetch.count_selectors_in_index(
        text, hrrr_transport.atmosphere_selectors(),
        hrrr_transport.ACCUMULATION_EXCLUSION)
    assert observed == hrrr_transport.ATMOSPHERE_RECORD_COUNT


def test_the_cloud_water_selector_carries_both_spellings():
    selectors = hrrr_transport.atmosphere_selectors()
    assert "CLMR|CLWMR:1 hybrid level" in selectors
    assert "CIMIXR|CICE:1 hybrid level" in selectors
    # ...and nothing else grew an alternation by accident.
    alternating = {selector.split(":")[0] for selector in selectors
                   if "|" in selector}
    assert alternating == {"CLMR|CLWMR", "CIMIXR|CICE"}


@pytest.mark.parametrize("engine", ["python", "rust-selectors"])
def test_a_wrfnat_index_from_before_hrrr_v3_selects_its_cice_cloud_ice(engine):
    """HRRRv1 and v2 index hybrid cloud ice as CICE and carry no CIMIXR.

    Keyed on CIMIXR alone, every cycle before July 2018 was refused with
    ``levels={'CIMIXR': []}`` on the idx-subset transport.
    """

    text, size = _atmosphere_index(cloud_ice_name="CICE")
    if engine == "python":
        rows = hrrr_transport._parse_index(text.encode("ascii"), size)
        selected = hrrr_transport._atmosphere_selection(rows)
        assert len(selected) == hrrr_transport.ATMOSPHERE_RECORD_COUNT
        assert sum(rows[index].variable == "CICE"
                   for index in selected) == 50
    else:
        observed = fetch.count_selectors_in_index(
            text, hrrr_transport.atmosphere_selectors(),
            hrrr_transport.ACCUMULATION_EXCLUSION)
        assert observed == hrrr_transport.ATMOSPHERE_RECORD_COUNT


def test_an_index_carrying_both_cloud_ice_codes_is_a_count_change_not_absorbed():
    """No published HRRR file carries both; one that did is not guessed at."""

    both = tuple(("CICE", f"{level} hybrid level") for level in range(1, 51))
    text, size = _atmosphere_index(extra=both)
    rows = hrrr_transport._parse_index(text.encode("ascii"), size)
    with pytest.raises(hrrr_transport.IndexInventoryError,
                       match="count=611"):
        hrrr_transport._atmosphere_selection(rows)


def test_an_absent_field_is_still_a_hard_error_under_aliasing():
    """Alias tolerance must not become "any name will do"."""

    text, size = _atmosphere_index(cloud_water_name="CLOUDWATER")
    rows = hrrr_transport._parse_index(text.encode("ascii"), size)
    with pytest.raises(hrrr_transport.IndexInventoryError) as error:
        hrrr_transport._atmosphere_selection(rows)
    assert "CLMR" in str(error.value)


def test_soil_index_selects_exactly_eighteen():
    text, size = _soil_index()
    rows = hrrr_transport._parse_index(text.encode("ascii"), size)
    assert len(hrrr_transport._soil_selection(rows)) == 18
    assert fetch.count_selectors_in_index(
        text, hrrr_transport.soil_selectors()) == 18


# ---------------------------------------------------------------------------
# The record-count tripwire
# ---------------------------------------------------------------------------

def test_a_matching_count_resolves_silently():
    said: list[str] = []
    bar = resolve_bar("hrrr-atmosphere", 561, progress=said.append)
    assert (bar.expected, bar.certified, bar.tripped) == (561, 561, False)
    assert said == []
    assert bar.as_manifest()["inventory_change_accepted"] is False


def test_a_changed_count_refuses_and_names_both_numbers():
    with pytest.raises(ValueError) as error:
        resolve_bar("hrrr-atmosphere", 572)
    message = str(error.value)
    assert "572" in message and "561" in message
    assert "Nothing was downloaded" in message
    assert ACCEPT_FLAG in message


def test_an_accepted_change_becomes_the_bar_and_is_recorded():
    said: list[str] = []
    bar = resolve_bar("gfs", 130, accept_inventory_change=True,
                      progress=said.append)
    assert bar.expected == 130
    assert bar.certified == CERTIFIED_RECORD_BARS["gfs"] == 124
    assert bar.tripped
    assert bar.as_manifest()["inventory_change_accepted"] is True
    assert said and "130" in said[0]


def test_an_unreadable_inventory_stands_the_certified_constant_in():
    bar = resolve_bar("hrrr-soil", None)
    assert (bar.expected, bar.derived) == (18, None)
    assert not bar.tripped
    assert "unavailable" in bar.source


def test_an_unknown_bar_kind_is_refused():
    with pytest.raises(ValueError, match="unknown record bar"):
        resolve_bar("hrrr-stratosphere", 1)


def test_the_gfs_selector_pairs_come_from_the_cgi_declaration():
    from tools import download_gfs_native_subset as gfs_transport

    pairs = fetch_bars.nomads_selector_pairs(
        gfs_transport.NOMADS_VARIABLES, gfs_transport.NOMADS_LEVELS,
        gfs_transport.PRESSURE_LEVELS_HPA)
    # The CGI's own vocabulary, mapped mechanically onto .idx columns.
    assert ("TSOIL", "0.1-0.4 m below ground") in pairs
    assert ("TMP", "2 m above ground") in pairs
    assert ("HGT", "500 mb") in pairs
    assert ("TMP", "lev_2_m_above_ground") not in pairs


def test_gfs_counting_is_exact_on_the_level_column():
    """``1 mb`` must not swallow ``100 mb``, ``150 mb`` or ``1000 mb``."""

    wanted = frozenset({("HGT", "1000 mb")})
    text = ("1:0:d=2026072812:HGT:1 mb:anl:\n"
            "2:100:d=2026072812:HGT:100 mb:anl:\n"
            "3:200:d=2026072812:HGT:1000 mb:anl:\n")
    assert fetch_bars.count_index_selection(text, wanted) == 1


# ---------------------------------------------------------------------------
# Transport fallback on an unrecognised inventory
# ---------------------------------------------------------------------------

def _fake_products(failing_host: str | None, seen: list[str]):
    """A ``_download_product`` that refuses one host's inventory."""

    def product(request, *, workers, retries, expected_count=-1):
        host = "nomads" if "nomads" in request.url else "s3"
        seen.append(f"{request.kind}:{host}")
        request.index_path.write_text(f"1:0:{host}\n", encoding="ascii")
        if host == failing_host:
            raise hrrr_transport.IndexInventoryError(
                "HRRR atmosphere index lacks the exact native bridge "
                "inventory: levels={'CLMR': []}, surfaces={}, count=511")
        request.destination.write_bytes(_grib2_stream(
            hrrr_transport.SOIL_RECORD_COUNT if request.kind == "soil"
            else hrrr_transport.ATMOSPHERE_RECORD_COUNT))
        return {"kind": request.kind}

    return product


def test_auto_falls_back_to_the_next_host_on_an_unrecognised_inventory(
        tmp_path, monkeypatch):
    seen: list[str] = []
    said: list[str] = []
    monkeypatch.setattr(hrrr_transport, "_download_product",
                        _fake_products("nomads", seen))
    out = tmp_path / "hrrr"
    manifest = fetch.fetch_hrrr(
        cycle=datetime(2026, 7, 28, 5, tzinfo=timezone.utc).replace(
            tzinfo=None),
        hours=(0,), area=None, out=out, transport="nomads",
        transport_fallback=("s3",), progress=said.append)

    # NOMADS was tried first for EACH product and S3 served both.  The
    # pooled transport may interleave the two products, so the pin is
    # per product, not on the global call order.
    per_product = {"atmosphere": [], "soil": []}
    for attempt in seen:
        kind, _, host = attempt.partition(":")
        per_product[kind].append(host)
    assert per_product == {"atmosphere": ["nomads", "s3"],
                           "soil": ["nomads", "s3"]}
    explanation = [line for line in said if "does not recognise" in line]
    assert len(explanation) == 2
    assert "falling back to s3" in explanation[0]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert {item["transport"] for item in payload["files"]
            if item["role"] in ("atmosphere", "soil")} == {"s3"}
    # The refused host's index was quarantined, not deleted or reused.
    assert list(out.glob("*.idx.rejected-*"))


def test_a_pinned_transport_reports_the_mismatch_instead_of_wandering(
        tmp_path, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(hrrr_transport, "_download_product",
                        _fake_products("nomads", seen))
    with pytest.raises(hrrr_transport.IndexInventoryError,
                       match="lacks the exact native bridge inventory"):
        fetch.fetch_hrrr(
            cycle=datetime(2026, 7, 28, 5), hours=(0,), area=None,
            out=tmp_path / "hrrr", transport="nomads", progress=lambda _: None)
    # The pinned host was the only one ever tried -- no product wandered
    # to another transport.  (The pooled soil product may have started
    # before the atmosphere refusal propagated; that changes which
    # attempts exist, never which host they target.)
    assert "atmosphere:nomads" in seen
    assert {attempt.partition(":")[2] for attempt in seen} == {"nomads"}


def test_the_manifest_records_the_engine_mode_and_bars(tmp_path,
                                                       monkeypatch):
    from test_fetch import _install_runtime_surface_inputs
    _install_runtime_surface_inputs(monkeypatch)
    monkeypatch.setattr(hrrr_transport, "_download_product",
                        _fake_products(None, []))
    manifest = fetch.fetch_hrrr(
        cycle=datetime(2026, 7, 28, 5), hours=(0,), area=None,
        out=tmp_path / "hrrr", transport="s3", progress=lambda _: None)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["engine"] == "python"
    assert payload["mode"] == "auto"
    assert {bar["kind"] for bar in payload["record_bars"]} == {
        "hrrr-atmosphere", "hrrr-soil"}
    assert all(bar["inventory_change_accepted"] is False
               for bar in payload["record_bars"])
    # Every payload file records its own census, so a later resume knows
    # what the file was supposed to contain without assuming a subset.
    records = {item["name"]: item.get("records")
               for item in payload["files"] if item["role"] != "checksums"}
    # The soil bar remains 18; its payload also carries the one source-
    # declared analyzed vegetation record, with its own frozen binding.
    assert set(records.values()) == {561, 19}
    assert {bar["kind"]: bar["derived"] for bar in payload["record_bars"]} == {
        "hrrr-atmosphere": 561, "hrrr-soil": 18}
    soil = next(item for item in payload["files"] if item["role"] == "soil")
    assert [record["field"] for record in soil["runtime_surface"]] == ["VEGFRA"]


# ---------------------------------------------------------------------------
# Engine resolution and the binary contract
# ---------------------------------------------------------------------------

def test_an_unknown_engine_is_refused():
    with pytest.raises(ValueError, match="unknown fetch engine"):
        fetch.resolve_fetch_engine("wget")


def test_the_env_override_fails_loudly_when_it_names_nothing(monkeypatch):
    monkeypatch.setenv(rustwx_fetch.FETCH_ENV, "/no/such/rw_fetch")
    with pytest.raises(FileNotFoundError, match="names a missing file"):
        rustwx_fetch.find_fetch_bin()


def test_the_abi_marker_is_shared_by_python_and_the_packager():
    from woof.native_wrf_distribution import (
        BRIDGE_NAMES, _BRIDGE_ABI_MARKERS, _BRIDGE_USAGE_MARKERS)

    assert "rw_fetch" in BRIDGE_NAMES
    assert _BRIDGE_USAGE_MARKERS["rw_fetch"] == "usage: rw_fetch"
    assert (_BRIDGE_ABI_MARKERS["rw_fetch"].decode("ascii")
            == rustwx_fetch.FETCH_ABI_MARKER)
    assert rustwx_fetch.FETCH_ABI_MARKER.startswith(
        rustwx_fetch.FETCH_RECORD_SCHEMA)


def test_cli_refuses_a_byte_transport_the_python_engine_cannot_serve(
        tmp_path, capsys):
    rc = cli.main(["fetch", "--source", "hrrr", "--engine", "python",
                   "--mode", "full-file", "--cycle", "2026-07-28T05",
                   "--hours", "1", "--out", str(tmp_path / "hrrr")])
    assert rc == 2
    assert "needs the rust fetch backbone" in capsys.readouterr().err


#: The backbone's checkout build as each shell must receive it, written
#: out rather than derived so a generator that loses its shell rule cannot
#: also rewrite what it is judged against.  Windows PowerShell 5.1 rejects
#: `&&` with a parser error.
BACKBONE_BUILD_FOR_SHELL = {
    False: "cd tools/rustwx && cargo build --release --locked --offline "
           "&& cd ../..",
    True: "cd tools\\rustwx; cargo build --release --locked --offline; "
          "cd ..\\..",
}


@pytest.mark.parametrize("windows", (False, True))
def test_the_python_engine_refusal_spells_the_backbone_build_for_the_shell(
        tmp_path, capsys, monkeypatch, windows):
    """The build line in this refusal is one a Windows user can paste.

    It printed `cd tools/rustwx && cargo build ...` on every OS, which
    Windows PowerShell 5.1 cannot parse.
    """

    from woof import bridges

    monkeypatch.setattr(bridges, "WINDOWS_SHELL", windows)
    rc = cli.main(["fetch", "--source", "hrrr", "--engine", "python",
                   "--mode", "full-file", "--cycle", "2026-07-28T05",
                   "--hours", "1", "--out", str(tmp_path / "hrrr")])
    assert rc == 2
    err = capsys.readouterr().err
    assert BACKBONE_BUILD_FOR_SHELL[windows] in err, err
    if windows:
        assert "&&" not in err, (
            f"Windows PowerShell 5.1 cannot parse '&&': {err}")


def test_the_backbone_flags_serve_hrrr_and_the_gfs_fullfile_route(
        tmp_path, capsys):
    """--engine/--mode used to be hrrr-only; the GFS full-file route
    (the 4090 user-zero finding) now takes them too.  What stays
    refused: ERA5 never moves bytes, and for GFS --engine chooses how
    WHOLE objects move, so without --mode full-file it names the route
    it belongs to.  Route selection itself is pinned in test_fetch.py
    (test_cli_fetch_gfs_fullfile_routes_through_the_new_transport)."""

    rc = cli.main(["fetch", "--source", "era5", "--engine", "rust",
                   "--cycle", "2026-07-28T00", "--hours", "3",
                   "--area", "30,-100,40,-90", "--out", str(tmp_path / "e")])
    assert rc == 2
    assert "--source hrrr or gfs/gdas only" in capsys.readouterr().err

    rc = cli.main(["fetch", "--source", "gfs", "--engine", "rust",
                   "--cycle", "2026-07-28T00", "--hours", "3",
                   "--area", "30,-100,40,-90", "--out", str(tmp_path / "g")])
    assert rc == 2
    assert "belong to '--mode full-file'" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The built binary
# ---------------------------------------------------------------------------

BINARY = rustwx_fetch.find_fetch_bin()
needs_binary = pytest.mark.skipif(
    BINARY is None,
    reason="rw_fetch not built (cd tools/rustwx && cargo build --release "
           "--locked --offline)")


@needs_binary
def test_the_built_binary_answers_its_three_probe_contracts():
    ok, evidence = rustwx_fetch.probe_fetch_bin(BINARY)
    assert ok, evidence
    # `woof doctor` accepts exit 0 from --version...
    version = subprocess.run([str(BINARY), "--version"],
                             capture_output=True, text=True)
    assert version.returncode == 0
    assert version.stdout.startswith("rw_fetch ")
    # ...and the distribution packager requires a NONZERO no-argument
    # exit that still prints the usage marker.
    bare = subprocess.run([str(BINARY)], capture_output=True, text=True)
    assert bare.returncode != 0
    assert "usage: rw_fetch" in (bare.stdout + bare.stderr).lower()


@needs_binary
def test_the_built_binary_refuses_an_hrrr_product_it_would_downgrade():
    """``build_hrrr_url`` ends in ``_ => "wrfsfc"``; rw_fetch must not.

    An unrecognised product token silently becomes the surface file
    upstream, which would hand a caller who asked for native levels a
    2-D object and surface the mistake three bars later.
    """

    result = subprocess.run(
        [str(BINARY), "probe", "--model", "hrrr", "--date", "20260728",
         "--cycle", "12", "--hours", "0", "--product", "wrfnative"],
        capture_output=True, text=True)
    # A refusal about the request, not a malformed command line.
    assert result.returncode == rustwx_fetch.EXIT_REFUSED
    assert "not an HRRR product token" in result.stderr
    assert "wrfnat" in result.stderr


@needs_binary
def test_the_built_binary_refuses_an_unknown_flag_rather_than_ignoring_it():
    result = subprocess.run(
        [str(BINARY), "fetch", "--model", "hrrr", "--turbo"],
        capture_output=True, text=True)
    assert result.returncode == 2
    assert '"--turbo"' in result.stderr


@pytest.mark.network
@needs_binary
@pytest.mark.skipif(os.environ.get("WOOF_NETWORK_TESTS") != "1",
                    reason="live network smoke; set WOOF_NETWORK_TESTS=1")
def test_live_probe_reports_the_transport_decision(tmp_path):
    """One live probe: does the rule answer against a real object?

    Deliberately a ``probe``, not a ``fetch``: it moves a few kilobytes
    of index and one bounded tail range, and it exercises the whole
    decision path -- URL algebra, strict index validation, and the
    coverage proof -- without pulling a 700 MB object onto the disk.
    """

    from datetime import timedelta

    # Three hours back: published on S3, and old enough that the run has
    # certainly finished landing.
    moment = datetime.now(timezone.utc) - timedelta(hours=3)
    report = rustwx_fetch.run_probe(
        BINARY, model="hrrr", date=f"{moment:%Y%m%d}", cycle=moment.hour,
        hours=(0,), product="nat", source="aws",
        cache_dir=tmp_path / "cache")
    hour = report["hours"][0]
    assert hour["mode"] in ("idx-subset", "full-file"), hour["mode_reason"]
    if hour["mode"] == "idx-subset":
        probe = hour["probe"]
        assert probe["idx_covers_object"] is True
        assert probe["object_bytes"] == (probe["idx_last_offset"]
                                         + probe["idx_last_message_bytes"])
        assert "accounts for all" in hour["mode_reason"]
    else:
        # The other legal answer: the index could not be proven to cover
        # the object, so the whole file is the safe transport.
        assert hour["probe"]["idx_covers_object"] is not True


# ---------------------------------------------------------------------------
# What the tripwire leaves behind
# ---------------------------------------------------------------------------
# The Rust backbone can only report a census after it has written the
# object, so a tripped tripwire on that path refuses with a payload
# already on disk.  Until v1.0.1 the refusal said "Nothing was
# downloaded" and left an unverified GRIB in a manifestless directory,
# which the next ordinary run then refused as well.

def _install_fake_backbone(monkeypatch, atmosphere: int,
                           soil: int | None = None,
                           dedup: dict | None = None):
    """Replace the Rust backbone with one that lands a payload first.

    That ordering is the whole point: the fixture writes the object and
    the ``.idx`` beside it, then reports a census, exactly as the real
    backbone does.  Per-product counts, so one product's inventory can
    move while the other's stays certified.

    ``dedup`` is the cache accounting a current backbone reports per
    transfer; omitting it is what an older binary looks like.
    """

    counts = {"atmosphere": atmosphere,
              "soil": (CERTIFIED_RECORD_BARS["hrrr-soil"]
                       if soil is None else soil)}

    def fake(*, binary, cycle, hour, kind, host, mode, out, cache_dir,
             progress, retries=0, shown_name=None, streams=None,
             byte_relay=None):
        records = counts[kind]
        name = (f"hrrr.t{cycle:%H}z.wrfnatf{hour:02d}.grib2"
                if kind == "atmosphere"
                else f"hrrr.t{cycle:%H}z.wrfprsf{hour:02d}.grib2")
        (out / name).write_bytes(_grib2_stream(records))
        (out / f"{name}.idx").write_text("1:0:d=2026072805:PRES:surface:anl:\n",
                                         encoding="ascii")
        entry = {
            "name": name, "idx_name": f"{name}.idx", "mode": "idx-subset",
            "mode_reason": "fixture", "bytes": records * 20,
            "wall_seconds": 0.1, "source": "aws",
            "grib_url": f"https://example.invalid/{name}",
            "selected_record_count": records,
            "probe": {"idx_covers_object": True,
                      "idx_record_count": records},
        }
        if dedup is not None:
            entry["dedup"] = dict(dedup)
        if kind == "atmosphere":
            # A current backbone call names the optional surface records
            # its subset took (none listed by this fixture's index).
            entry["optional_surface"] = []
        return entry

    monkeypatch.setattr(fetch, "_rw_fetch_hrrr", fake)


def test_a_late_tripwire_quarantines_the_payload_and_says_so(
        tmp_path, monkeypatch):
    """Ordering and accuracy, together: the two halves of one failure."""

    _install_fake_backbone(monkeypatch, CERTIFIED_RECORD_BARS[
        "hrrr-atmosphere"] + 11)
    out = tmp_path / "hrrr"
    said: list[str] = []
    with pytest.raises(ValueError) as error:
        fetch.fetch_hrrr(
            cycle=datetime(2026, 7, 28, 5), hours=(0,), area=None, out=out,
            transport="s3", engine="rust", engine_bin=Path("rw_fetch"),
            progress=said.append)

    message = str(error.value)
    assert "572" in message and "561" in message
    assert ACCEPT_FLAG in message
    # The old lie is gone...
    assert "Nothing was downloaded" not in message
    # ...replaced by what is actually true of this directory.
    assert "payload is on disk" in message
    assert "moved aside" in message
    assert "Nothing was deleted" in message

    # Ordering: no unverified GRIB is left where a consumer would read
    # it as a fetch product, and neither is the index whose census
    # disagreed.
    assert not (out / "hrrr.t05z.wrfnatf00.grib2").exists()
    assert not (out / "hrrr.t05z.wrfnatf00.grib2.idx").exists()
    quarantined = sorted(p.name for p in out.glob("*.inventory-change-*"))
    assert len(quarantined) == 2, quarantined
    # Nothing deleted: the bytes are all still there.
    assert sum(p.stat().st_size for p in out.glob("*.inventory-change-*")) > 0
    # And no manifest was published, so a later run sees an unfinished
    # directory rather than a blessed one.
    assert not (out / fetch.FETCH_MANIFEST_NAME).exists()


def test_the_early_tripwire_still_says_nothing_was_downloaded():
    """The GFS path derives the bar first, so that sentence stays true."""

    with pytest.raises(ValueError) as error:
        resolve_bar("gfs", 130)
    assert fetch_bars.NOTHING_DOWNLOADED in str(error.value)


def test_an_accepted_change_survives_a_completed_resume(tmp_path,
                                                        monkeypatch):
    """accept -> resume -> the acceptance is still in the manifest.

    A resume that finds every file present downloads nothing, resolves
    no bars, and used to republish ``record_bars: []`` -- dropping the
    record that this directory was fetched under an accepted inventory
    change, which DATA promises is kept.
    """

    changed = CERTIFIED_RECORD_BARS["hrrr-atmosphere"] + 11
    from test_fetch import _install_runtime_surface_inputs
    _install_runtime_surface_inputs(monkeypatch)
    _install_fake_backbone(monkeypatch, changed)
    out = tmp_path / "hrrr"
    args = dict(cycle=datetime(2026, 7, 28, 5), hours=(0,), area=None,
                out=out, transport="s3", engine="rust",
                engine_bin=Path(os.environ.get(rustwx_fetch.FETCH_ENV, "rw_fetch")),
                progress=lambda _: None)

    manifest = fetch.fetch_hrrr(accept_inventory_change=True, **args)
    accepted = json.loads(manifest.read_text(encoding="utf-8"))
    by_kind = {bar["kind"]: bar for bar in accepted["record_bars"]}
    assert set(by_kind) == {"hrrr-atmosphere", "hrrr-soil"}
    assert by_kind["hrrr-atmosphere"]["inventory_change_accepted"] is True
    assert by_kind["hrrr-atmosphere"]["derived"] == changed
    # The soil selection did not move, so nothing was accepted for it.
    assert by_kind["hrrr-soil"]["inventory_change_accepted"] is False

    # The resume: every file is present and verified, nothing is
    # downloaded, and the flag is NOT passed again -- which is exactly
    # the run that used to erase the provenance.
    resumed_manifest = fetch.fetch_hrrr(**args)
    resumed = json.loads(resumed_manifest.read_text(encoding="utf-8"))
    assert resumed["record_bars"] == accepted["record_bars"]
    assert next(bar for bar in resumed["record_bars"]
                if bar["kind"] == "hrrr-atmosphere"
                )["inventory_change_accepted"] is True


def test_a_resume_that_re_resolves_a_bar_records_the_new_one(tmp_path,
                                                             monkeypatch):
    """Seeding must not freeze a stale bar over a real re-download."""

    from test_fetch import _install_runtime_surface_inputs
    _install_runtime_surface_inputs(monkeypatch)
    _install_fake_backbone(monkeypatch, CERTIFIED_RECORD_BARS[
        "hrrr-atmosphere"] + 11)
    out = tmp_path / "hrrr"
    args = dict(cycle=datetime(2026, 7, 28, 5), hours=(0,), area=None,
                out=out, transport="s3", engine="rust",
                engine_bin=Path(os.environ.get(rustwx_fetch.FETCH_ENV, "rw_fetch")),
                progress=lambda _: None)
    fetch.fetch_hrrr(accept_inventory_change=True, **args)

    # Upstream goes back to the certified census and --force re-fetches.
    _install_fake_backbone(monkeypatch, CERTIFIED_RECORD_BARS[
        "hrrr-atmosphere"])
    manifest = fetch.fetch_hrrr(force=True, **args)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    atmosphere = next(bar for bar in payload["record_bars"]
                      if bar["kind"] == "hrrr-atmosphere")
    assert atmosphere["derived"] == CERTIFIED_RECORD_BARS["hrrr-atmosphere"]
    assert atmosphere["inventory_change_accepted"] is False


# ---------------------------------------------------------------------------
# The download cache's own cost.  One full-file object reaches the backbone's
# cache under two key shapes, and those two entries used to be two whole
# copies on disk -- a 15.5 GB fetch left 62 GB of cache with nothing on the
# receipt to say where it went.  The entries now share one content-addressed
# payload and the manifest carries the arithmetic.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dedup, expected", [
    pytest.param({"cache_bytes_written": 700, "cache_bytes_deduplicated": 500,
                  "reference_entries": 1},
                 {"transfers": 2, "cache_bytes_written": 1400,
                  "cache_bytes_deduplicated": 1000, "reference_entries": 2}, id="accounted"),
    pytest.param(None, {"transfers": 0, "cache_bytes_written": 0,
                        "cache_bytes_deduplicated": 0, "reference_entries": 0}, id="older-backbone"),
    pytest.param({"cache_bytes_written": "lots", "reference_entries": 3},
                 {"transfers": 2, "cache_bytes_written": 0,
                  "cache_bytes_deduplicated": 0, "reference_entries": 6}, id="malformed-count"),
])
def test_the_manifest_carries_the_cache_dedup_summary(tmp_path, monkeypatch, dedup, expected):
    """Account valid arithmetic, absent older records and malformed counts separately."""
    from test_fetch import _install_runtime_surface_inputs, _assert_runtime_surface_requests
    runtime_calls = _install_runtime_surface_inputs(monkeypatch)
    _install_fake_backbone(monkeypatch, CERTIFIED_RECORD_BARS["hrrr-atmosphere"], dedup=dedup)
    out = tmp_path / "hrrr"
    manifest = fetch.fetch_hrrr(
        cycle=datetime(2026, 7, 28, 5), hours=(0,), area=None, out=out,
        transport="s3", engine="rust", engine_bin=Path("rw_fetch"), progress=lambda _: None)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    _assert_runtime_surface_requests(runtime_calls, payload, datetime(2026, 7, 28, 5))
    assert payload["dedup"] == expected
    assert (out / "hrrr.t05z.wrfnatf00.grib2").is_file()
    assert {bar["kind"] for bar in payload["record_bars"]} == {"hrrr-atmosphere", "hrrr-soil"}


# ---------------------------------------------------------------------------
# The default byte transport.  ``--mode auto`` resolved to ``idx-subset``
# against every healthy host, so the documented fast path was reachable only
# by typing three flags -- and a field user paid 560 s for a 419 MB file that
# the whole-file route moves in 27-35 s.
# ---------------------------------------------------------------------------

def _hrrr_fetch_plan(monkeypatch, argv, *, backbone: Path | None,
                     resolved_transport: str = "s3"):
    """Run ``woof fetch --source hrrr`` far enough to read its plan.

    The transfer itself is replaced; what is under test is the pair
    (engine, mode) the front door decides on before any byte moves.
    """

    seen: dict[str, object] = {}
    reached: list[object] = []

    def capture(**kwargs):
        seen.update(kwargs)
        raise _StopBeforeTransfer

    def no_network(*args, **kwargs):
        reached.append(args)
        raise OSError("a fetch plan test does not reach the network")

    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: backbone)
    monkeypatch.setattr(rustwx_fetch, "probe_fetch_bin",
                        lambda path: (True, "usable"))
    monkeypatch.setattr(fetch, "fetch_hrrr", capture)
    monkeypatch.setattr(fetch, "require_published_cycle",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(fetch, "check_prior_request",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(fetch, "resolve_hrrr_transport",
                        lambda cycle, requested, **kwargs: resolved_transport)
    # The default fetch runs as posted, and its posting loop asks each
    # lead's host before the transfer moves it.  Every lead answers
    # posted here, and a socket that tries to leave this process is
    # refused and counted, so these tests (not marked network) never
    # wait on a real host's HEAD.
    monkeypatch.setattr(fetch, "_head_answer", lambda url: True)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    parser = cli.build_parser()
    args = parser.parse_args(argv)
    try:
        fetch.fetch_main(args)
    except _StopBeforeTransfer:
        pass
    assert reached == [], f"the fetch plan reached the network: {reached}"
    return seen


class _StopBeforeTransfer(Exception):
    """Raised by the stand-in transfer once the plan is readable."""


def _hrrr_argv(tmp_path, *extra):
    return ["fetch", "--source", "hrrr", "--cycle", "2026-07-30T12",
            "--hours", "1", "--out", str(tmp_path / "out"), *extra]


def test_the_hrrr_default_is_the_record_subset_through_the_backbone(
        tmp_path, monkeypatch):
    """Move only the bytes the preparation reads (the project owner, 2026-10-29).

    ``auto`` takes the ``.idx`` record subset wherever the index is
    complete and the whole object where it is not.  Measured on a development machine
    over AWS, 12Z f00-f03: whole files 4,601 MB, the subset 1,823 MB,
    quicker and with byte-identical prepared files.
    """

    plan = _hrrr_fetch_plan(monkeypatch, _hrrr_argv(tmp_path),
                            backbone=tmp_path / "rw_fetch")
    assert plan["engine"] == "rust"
    assert plan["mode"] == fetch.HRRR_DEFAULT_MODE == "auto"


def test_both_byte_modes_stay_available_when_named(tmp_path, monkeypatch):
    for named in ("idx-subset", "full-file"):
        plan = _hrrr_fetch_plan(
            monkeypatch, _hrrr_argv(tmp_path, "--mode", named),
            backbone=tmp_path / "rw_fetch")
        assert plan["mode"] == named


def test_an_explicit_auto_still_means_the_probe_rule(tmp_path, monkeypatch):
    plan = _hrrr_fetch_plan(
        monkeypatch, _hrrr_argv(tmp_path, "--mode", "auto"),
        backbone=tmp_path / "rw_fetch")
    assert plan["mode"] == "auto"


def test_an_install_without_the_backbone_is_told_what_it_is_paying(
        tmp_path, monkeypatch, capsys):
    """The fallback is correct and slow, and now it says so up front.

    Negative control for the line above: with the backbone present the
    same invocation prints no such warning (asserted below).
    """

    plan = _hrrr_fetch_plan(monkeypatch, _hrrr_argv(tmp_path), backbone=None)
    assert plan["engine"] == "python"
    assert plan["mode"] == "auto"
    warned = capsys.readouterr().err
    assert "Python transport" in warned
    assert "woof setup" in warned

    _hrrr_fetch_plan(monkeypatch, _hrrr_argv(tmp_path),
                     backbone=tmp_path / "rw_fetch")
    assert "Python transport" not in capsys.readouterr().err


def test_choosing_the_python_engine_is_not_second_guessed(
        tmp_path, monkeypatch, capsys):
    plan = _hrrr_fetch_plan(
        monkeypatch, _hrrr_argv(tmp_path, "--engine", "python"),
        backbone=None)
    assert plan["engine"] == "python"
    assert "Python transport" not in capsys.readouterr().err


def test_an_auto_resolved_nomads_whole_file_names_what_it_costs(
        tmp_path, monkeypatch, capsys):
    """The throughput split between the two hosts, said once.

    Measured on one box against one cycle, four objects, the same
    backbone and the same default mode: 348/209/418/255 s from NOMADS
    and 69/34/45/44 s from S3.  ``auto`` prefers S3 for exactly that
    reason and reaches NOMADS only when S3 cannot serve the window yet.

    So this line names the COST and not a remedy: pointing a reader at
    `--transport s3` here would send them at a host that has just been
    asked and answered no, and a printed next step that 404s is worse
    than no next step.
    """

    _hrrr_fetch_plan(monkeypatch, _hrrr_argv(tmp_path),
                     backbone=tmp_path / "rw_fetch",
                     resolved_transport="nomads")
    printed = capsys.readouterr().out
    assert "paces whole-file transfers" in printed
    assert "the only host that has this window yet" in printed
    assert "--transport s3" not in printed


def test_a_pinned_transport_is_not_second_guessed(tmp_path, monkeypatch,
                                                  capsys):
    """Negative control, watched firing: an operator who named the host.

    The line above fires for the same resolved host when it was chosen
    by ``auto``; naming ``--transport nomads`` is a decision, and a
    decision does not get advice.
    """

    _hrrr_fetch_plan(monkeypatch,
                     _hrrr_argv(tmp_path, "--transport", "nomads"),
                     backbone=tmp_path / "rw_fetch",
                     resolved_transport="nomads")
    assert "paces whole-file transfers" not in capsys.readouterr().out


def test_an_as_posted_fetch_names_the_cost_once_before_its_first_nomads_prefix():
    """As posted, the host is chosen for each verified prefix.

    The cost is said before the first prefix the operational server
    serves and not again for later ones; a prefix the archive serves, an
    unresolved ``auto``, a named host, a subset transfer and a cached
    window say nothing.
    """

    said: list[str] = []
    advise = fetch._hrrr_pacing_advice(pinned=None, mode="full-file",
                                       cached=False, say=said.append)
    advise("auto")
    advise("s3")
    assert said == []
    advise("nomads")
    assert len(said) == 1 and "paces whole-file transfers" in said[0]
    advise("nomads")
    advise("s3")
    assert len(said) == 1

    for quiet in ({"pinned": "nomads", "mode": "full-file", "cached": False},
                  {"pinned": None, "mode": "idx-subset", "cached": False},
                  {"pinned": None, "mode": "full-file", "cached": True}):
        silent: list[str] = []
        fetch._hrrr_pacing_advice(**quiet, say=silent.append)("nomads")
        assert silent == [], quiet


# ---------------------------------------------------------------------------
# A failed rw_fetch reaches the run's failed event as its real reason
# ---------------------------------------------------------------------------

#: What a 2.7.7 rw_fetch printed on stderr when one 16 MiB chunk of a
#: real wrfprs object timed out: wx-core's carriage-return progress, the
#: reason glued onto it, then the whole usage text.  Captured from the
#: built binary against noaa-hrrr-bdp-pds, abridged in the usage only.
_GLUED_STDERR = (
    "\r  Downloading chunks 1/25...\r  Downloading chunks 9/25..."
    "rw_fetch: HTTP error: failed to read https://noaa-hrrr-bdp-pds.s3."
    "amazonaws.com/hrrr.20260925/conus/hrrr.t06z.wrfprsf01.grib2: "
    "timeout: global\n"
    "usage: rw_fetch <fetch|probe|latest> [OPTIONS]\n"
    "       rw_fetch --version | --help | --abi\n\n"
    "  fetch    download a model-run window and print a fetch record\n\n"
    "common options\n"
    "  --model NAME            hrrr, gfs, gdas, rap, nam, ...\n\n"
    "fetch options\n"
    "  --mode MODE             auto (default) | full-file | idx-subset\n")


def _stand_in(tmp_path: Path, *, stderr: str, code: int) -> list[str]:
    """A command that prints ``stderr`` byte for byte and exits ``code``."""

    script = tmp_path / "stand_in.py"
    script.write_text(
        "import sys\n"
        f"sys.stderr.buffer.write({stderr.encode('utf-8')!r})\n"
        f"sys.exit({code})\n", encoding="utf-8")
    return [sys.executable, str(script)]


def test_a_reason_glued_to_chunk_progress_is_the_reason_reported(tmp_path):
    """The reason, not the first usage heading.

    A 22.9 GiB HRRR fetch that lost one chunk 38 minutes in reached the
    run's failed event as ``rw_fetch fetch: common options``.
    """

    with pytest.raises(RuntimeError) as caught:
        rustwx_fetch._run(_stand_in(tmp_path, stderr=_GLUED_STDERR, code=2),
                          what="fetch")
    message = str(caught.value)
    assert message == (
        "rw_fetch fetch: HTTP error: failed to read https://noaa-hrrr-bdp-"
        "pds.s3.amazonaws.com/hrrr.20260925/conus/hrrr.t06z.wrfprsf01.grib2"
        ": timeout: global")
    assert "common options" not in message


def test_a_transfer_cut_off_is_transient_and_its_remedy_reaches_the_event(
        tmp_path):
    from woof import runplan

    stderr = ("\r  Downloading chunks 26/27...\n"
              "rw_fetch: HTTP error: failed to read https://x/y: connection "
              "reset (gave up after 4 attempts)\n")
    out = tmp_path / "downloads"
    with pytest.raises(rustwx_fetch.RwFetchError) as caught:
        rustwx_fetch._run(
            _stand_in(tmp_path, stderr=stderr,
                      code=rustwx_fetch.EXIT_TRANSFER),
            what="fetch", out=out)
    error = caught.value
    assert error.transient
    assert error.reason.endswith("(gave up after 4 attempts)")
    assert str(out) in error.remedy
    # The run-plan front door relays it rather than `remedy: null`.
    assert runplan._remedy(error) == error.remedy


def test_a_refusal_is_not_transient_and_a_silent_exit_still_says_so(tmp_path):
    with pytest.raises(rustwx_fetch.RwFetchError) as caught:
        rustwx_fetch._run(
            _stand_in(tmp_path, stderr="\nrw_fetch: f000: no source served "
                      "this object -- aws: the object is absent\n", code=1),
            what="fetch")
    assert not caught.value.transient
    assert caught.value.remedy is None
    assert str(caught.value).endswith("the object is absent")
    assert rustwx_fetch.failure_reason("", -9) == (
        "rw_fetch was stopped by signal 9 before it reported a reason")
    assert rustwx_fetch.failure_reason("  Downloading chunks 3/9...", 5) == (
        "rw_fetch exited 5 without reporting a reason")


def _one_file_record(out: Path) -> dict:
    return {"schema": rustwx_fetch.FETCH_RECORD_SCHEMA, "files": [{
        "name": "hrrr.t06z.wrfprsf18.grib2", "bytes": 1, "wall_seconds": 1.0,
        "source": "aws", "mode": "full-file", "mode_reason": "requested",
        "grib_url": "https://example/hrrr.t06z.wrfprsf18.grib2"}]}


def test_the_rust_hrrr_route_asks_again_after_a_network_cut_off(
        tmp_path, monkeypatch):
    """One object cut off by the network costs that object a retry.

    It used to cost the run: the Rust route spent none of the fetch's
    retry budget, so f18 soil failing after the backbone's own retries
    ended a fetch whose other 43 files had landed.
    """

    calls: list[int] = []

    def flaky(binary, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise rustwx_fetch.RwFetchError(
                "fetch", "HTTP error: failed to read https://x: connection "
                "reset", returncode=rustwx_fetch.EXIT_TRANSFER)
        return _one_file_record(kwargs["out"])

    monkeypatch.setattr(rustwx_fetch, "run_fetch", flaky)
    waits: list[float] = []
    monkeypatch.setattr(fetch.fetch_pool, "sleep_unless_stopped",
                        lambda seconds, **_: waits.append(seconds))
    said: list[str] = []
    entry = fetch._rw_fetch_hrrr(
        binary=Path("rw_fetch"), cycle=datetime(2026, 9, 25, 6), hour=18,
        kind="soil", host="s3", mode="full-file", out=tmp_path,
        cache_dir=None, progress=said.append, retries=2,
        shown_name="hrrr.t06z.soilf18.grib2")
    assert entry["name"] == "hrrr.t06z.wrfprsf18.grib2"
    assert len(calls) == 2
    assert any("asking again (1 of 2) in 2 s" in line for line in said)
    # The shared backoff, not an immediate re-ask.
    assert waits == [2.0]
    # One completion line, under the name the file is filed as.
    done = [line for line in said if " B in " in line]
    assert len(done) == 1 and "soilf18" in done[0], said


def test_the_rust_hrrr_route_does_not_retry_a_refusal_or_past_its_budget(
        tmp_path, monkeypatch):
    for code, budget, expected_calls, expected_waits in (
            (rustwx_fetch.EXIT_REFUSED, 3, 1, []),
            (rustwx_fetch.EXIT_TRANSFER, 2, 3, [2.0, 4.0])):
        calls: list[int] = []
        waits: list[float] = []
        monkeypatch.setattr(fetch.fetch_pool, "sleep_unless_stopped",
                            lambda seconds, **_: waits.append(seconds))

        def failing(binary, **kwargs):
            calls.append(1)
            raise rustwx_fetch.RwFetchError("fetch", "no", returncode=code)

        monkeypatch.setattr(rustwx_fetch, "run_fetch", failing)
        with pytest.raises(rustwx_fetch.RwFetchError):
            fetch._rw_fetch_hrrr(
                binary=Path("rw_fetch"), cycle=datetime(2026, 9, 25, 6),
                hour=0, kind="atmosphere", host="s3", mode="full-file",
                out=tmp_path, cache_dir=None, progress=lambda _: None,
                retries=budget)
        assert len(calls) == expected_calls
        assert waits == expected_waits


def test_the_rust_hrrr_route_counts_its_asks_against_the_budget_it_spends(
        tmp_path, monkeypatch):
    """Each "asking again (N of M)" line names the asks the loop will make.

    A fetch handed the Python transport's budget of five was told "1 of
    5" and then stopped after the fourth ask, because the shared schedule
    caps the loop at ``TRANSIENT_ATTEMPTS - 1``; the reader waited for a
    fifth ask that never came.
    """

    from woof import fetch_endpoints

    calls: list[int] = []
    monkeypatch.setattr(fetch.fetch_pool, "sleep_unless_stopped",
                        lambda seconds, **_: None)

    def cut_off(binary, **kwargs):
        calls.append(1)
        raise rustwx_fetch.RwFetchError(
            "fetch", "HTTP error: connection reset",
            returncode=rustwx_fetch.EXIT_TRANSFER)

    monkeypatch.setattr(rustwx_fetch, "run_fetch", cut_off)
    budget = fetch_endpoints.TRANSIENT_ATTEMPTS - 1
    said: list[str] = []
    with pytest.raises(rustwx_fetch.RwFetchError):
        fetch._rw_fetch_hrrr(
            binary=Path("rw_fetch"), cycle=datetime(2026, 9, 25, 6), hour=0,
            kind="atmosphere", host="s3", mode="full-file", out=tmp_path,
            cache_dir=None, progress=said.append, retries=budget + 1)
    asks = [line for line in said if "asking again" in line]
    assert len(calls) == budget + 1
    assert len(asks) == budget
    for number, line in enumerate(asks, start=1):
        assert f"asking again ({number} of {budget}) in" in line, said


def test_a_cut_off_rust_transfer_moves_the_gfs_ladder_on():
    from woof import fetch_endpoints

    cut = rustwx_fetch.RwFetchError(
        "fetch", "HTTP error: connection reset",
        returncode=rustwx_fetch.EXIT_TRANSFER)
    refused = rustwx_fetch.RwFetchError(
        "fetch", "--model is required", returncode=rustwx_fetch.EXIT_REFUSED)
    assert fetch_endpoints.fault_reason(cut) == (
        "the connection failed -- HTTP error: connection reset")
    assert fetch_endpoints.fault_reason(refused) is None


@needs_binary
def test_the_built_binary_prints_a_runtime_reason_without_the_usage_text(
        tmp_path):
    """A refusal after the command line parsed is one line, exit 1."""

    result = subprocess.run(
        [str(BINARY), "fetch", "--model", "hrrr", "--date", "20260925",
         "--cycle", "06", "--hours", "0", "--product", "wrfprs"],
        capture_output=True, text=True)
    assert result.returncode == rustwx_fetch.EXIT_REFUSED
    assert "usage:" not in result.stderr
    assert rustwx_fetch.failure_reason(result.stderr, result.returncode) == (
        "--out DIR is required")


#: What a current rw_fetch prints when an origin answers a chunk's range
#: request with the whole object: an answer the origin gave on purpose,
#: exit 1, not a transfer the network cut off.
_RANGE_IGNORED_STDERR = (
    "\r  Downloading chunks 1/27...\n"
    "rw_fetch: HTTP error: range request for https://mirror.example/hrrr."
    "t06z.wrfnatf00.grib2 (bytes=16777216-33554431) returned HTTP 200, not "
    "206; the origin did not serve the requested span\n")


def test_an_origin_that_ignores_range_is_not_retried_or_blamed_on_the_network(
        tmp_path, monkeypatch):
    """A refusal from the origin gets no retry loop and no network remedy.

    Every payload failure used to exit 3, so a mirror that ignores Range
    or refuses access was downloaded again up to the retry budget and the
    reader was told to wait for a steadier connection.
    """

    from woof import fetch_endpoints, runplan

    command = _stand_in(tmp_path, stderr=_RANGE_IGNORED_STDERR,
                        code=rustwx_fetch.EXIT_REFUSED)
    with pytest.raises(rustwx_fetch.RwFetchError) as caught:
        rustwx_fetch._run(command, what="fetch", out=tmp_path / "downloads")
    error = caught.value
    assert not error.transient
    assert error.remedy is None
    assert error.reason.endswith("the origin did not serve the requested span")
    assert fetch_endpoints.fault_reason(error) is None
    assert "network cut" not in (runplan._remedy(error) or "")

    calls: list[int] = []

    def refused(binary, **kwargs):
        calls.append(1)
        return rustwx_fetch._run(command, what="fetch", out=kwargs["out"])

    monkeypatch.setattr(rustwx_fetch, "run_fetch", refused)
    said: list[str] = []
    with pytest.raises(rustwx_fetch.RwFetchError):
        fetch._rw_fetch_hrrr(
            binary=Path("rw_fetch"), cycle=datetime(2026, 9, 25, 6), hour=0,
            kind="atmosphere", host="s3", mode="full-file", out=tmp_path,
            cache_dir=None, progress=said.append, retries=5)
    assert len(calls) == 1
    assert not any("asking again" in line for line in said)


def test_exit_2_from_an_older_backbone_is_not_called_a_command_line_dispute(
        tmp_path):
    """A pre-split backbone exits 2 for everything, a dropped link too.

    Its probe still passes (the record ABI did not change), so exit 2
    alone does not mean the command line was refused.  Only a current
    backbone's usage-error line does.
    """

    with pytest.raises(rustwx_fetch.RwFetchError) as caught:
        rustwx_fetch._run(_stand_in(tmp_path, stderr=_GLUED_STDERR, code=2),
                          what="fetch")
    older = caught.value.remedy
    assert "disagree about its command line" not in older
    assert "older than this woof" in older

    usage = ("\nrw_fetch: unknown option \"--bogus\"\n"
             "rw_fetch --help lists every option\n")
    with pytest.raises(rustwx_fetch.RwFetchError) as caught:
        rustwx_fetch._run(_stand_in(tmp_path, stderr=usage, code=2),
                          what="fetch")
    assert "disagree about its command line" in caught.value.remedy
    assert caught.value.reason == 'unknown option "--bogus"'
