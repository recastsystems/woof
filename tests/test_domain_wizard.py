"""``woof domain`` wizard gates: sizing fit, round trips, edge rejects.

The wizard owns no memory arithmetic and no projection math of its own,
so these tests bind its glue: emitted configs load through the REAL
experiment/case-data loaders, their estimator envelope fits the declared
card budget, the companion namelist.wps agrees with the [projection]
table through the real grid builders, and every documented refusal
(pole containment, bad cycle, missing budget) fails loudly with an
actionable message.  Worldwide contract: the projection is
auto-selected by |lat| (mercator < 25 <= lambert <= 60 < polar),
both hemispheres emit, and antimeridian-crossing footprints produce
wrap-aware (W > E) fetch boxes instead of refusals.
"""
from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
import json
import os
import re

import tomllib
from pathlib import Path

import numpy as np
import pytest

from woof.case_data import load_experiment_case
from woof.cli import main as cli_main
from woof.core.preflight import (GIB, estimate_experiment,
                                  estimate_phases,
                                  observed_peak_envelope_bytes)
from woof.domain_wizard import (CARD_VRAM_GIB, DEFAULT_PHYSICS_PROFILE,
                                 LADDER_RATIOS, WIZARD_PHYSICS_PROFILES,
                                 DomainFitError, _dims_for_scale,
                                 _ladder_dx_km, card_assumed_free_gib,
                                 experiment_from_text, fit_headroom_bytes,
                                 profile_switches, radt_ladder_minutes,
                                 render_config, sizing_budget_bytes,
                                 vram_reserve_gib)
from woof.experiment import load_experiment
from woof.fetch import validate_fetch_hints
from woof.hrrr_route_inputs import HrrrRouteInputError, route_input_paths
from woof.physics_compat import (MORRISON_PROFILE_ID,
                                  single_domain_runtime_switches)
from woof.source_adapters import (source_coverage_window,
                                   wizard_planable_source_ids)
from woof.ingest.grib import parse_vtable
from woof.static.lambert import (grids_from_projection_config,
                                  grids_from_wps_namelist)

BUNDLE = Path(os.environ.get("WOOF_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
MAY99 = Path(os.environ.get("WOOF_TEST_MAY99_DATA",
                    "gpuwm-fixture-unset/may99-data"))
requires_staged_real_inputs = pytest.mark.skipif(
    not (MAY99 / "era5_may1999_pl.grib").is_file()
    or not (BUNDLE / "static/WPS_GEOG/topo_gmted2010_30s").is_dir(),
    reason="staged May-1999 ERA5 or the WPS_GEOG tree is absent",
)


def _as_windows(monkeypatch, pf):
    """Price the card as Windows does, without breaking this host's TLS.

    ``pf.sys`` is the interpreter's one ``sys``: under ``platform =
    "win32"`` a first ``urllib`` opener on Linux reaches
    ``ssl._load_windows_store_certs`` and dies (NameError
    enum_certificates).  The keyless ERA5 default reads the ARCO store's
    bounds document over HTTPS, so these tests answer that read the way an
    unreadable document is answered (None: the declared publication delay
    stands) instead of opening a socket under a faked platform.
    """
    from woof import source_availability

    monkeypatch.setattr(source_availability, "published_stop",
                        lambda window, **_kw: None)
    monkeypatch.setattr(pf.sys, "platform", "win32")


def _run_wizard(tmp_path, *extra, point="39.7,-96.6", card="16gb",
                ladder="12-3", source="era5", cycle="1999-05-03T12"):
    out = tmp_path / "area.toml"
    # --point=VALUE form: a leading "-" (southern latitude) must not be
    # parsed as an option flag.  card=None omits --card entirely, for the
    # cases that pass --vram-gib instead (the two are exclusive).
    rc = cli_main([
        "domain", f"--point={point}",
        *(() if card is None else ("--card", card)), "--ladder", ladder,
        "--source", source, "--cycle", cycle, "--out", str(out), *extra])
    return rc, out


@contextmanager
def _runtime_card_pricing(monkeypatch, *, vram_gib, free_bytes):
    """Re-price a fit through the runtime probe for its declared card."""
    import sys

    from woof.core import mynn_pbl_scratch as scratch

    # CUDA measured less than the nominal total on the 16 GB and 32 GB
    # cards. At 49 levels that crosses the MYNN width quantum, so nominal
    # totals would price a wider workspace than those cards actually run.
    # Keep the probe samples independent of the planner's capacity table.
    total_bytes = {16.0: 16611278848, 32.0: 33711521792}.get(
        vram_gib, int(vram_gib * GIB))
    card = {
        "sm_count": 70, "max_threads_per_sm": 1536,
        "registers_per_sm": 65536, "warp_size": 32,
        "total_bytes": total_bytes, "free_bytes": int(free_bytes),
        "device_name": "declared-card-runtime",
    }
    assert scratch._PRICING_MEMORY.get() is None
    with monkeypatch.context() as runtime:
        runtime.delenv(scratch.MYNN_PBL_COLUMN_CHUNK_ENV, raising=False)
        runtime.setattr(scratch, "probe_mynn_card", lambda device=None: card)
        runtime.setattr(scratch, "_RESOLVED", {})
        runtime.setattr(scratch, "_PINNED", None)
        runtime.setattr(scratch, "_TILE_WALKED", {})
        # Runtime resolution publishes its choice to already loaded modules.
        # Register their original values so repricing cannot leak that choice.
        for name in scratch._CHUNK_MODULES:
            module = sys.modules.get(name)
            if module is not None and hasattr(module, "MYNN_PBL_COLUMN_CHUNK"):
                runtime.setattr(module, "MYNN_PBL_COLUMN_CHUNK",
                                module.MYNN_PBL_COLUMN_CHUNK)
        yield


# ---------------------------------------------------------------------------
# Input rejection: every documented refusal is a stderr message + exit 2
# through the CLI dispatch boundary -- never a Python traceback.
# ---------------------------------------------------------------------------

def _assert_refused(capsys, needle: str, rc: int) -> None:
    assert rc == 2
    err = capsys.readouterr().err
    assert needle in err
    assert "Traceback" not in err


@pytest.mark.parametrize("point, needle", [
    ("35.3", "lat,lon"),
    ("abc,-97.5", "decimal degrees"),
    ("95.0,-60.0", "[-90, 90]"),
    ("35.3,-400.0", "[-180, 180]"),
])
def test_point_rejections(tmp_path, capsys, point, needle):
    rc, out = _run_wizard(tmp_path, point=point)
    _assert_refused(capsys, needle, rc)
    assert not out.exists()


def test_out_of_convention_longitude_wraps_with_a_warning(tmp_path, capsys):
    """Warn-not-block: 170E spelled as -190 is a real longitude; the
    wizard wraps it to the [-180, 180] convention, says so in one line,
    and proceeds.

    The artifacts are checked, not just the sentence.  This test used to
    assert the warning alone, and passed for two releases while the
    wrapped value reached the emitted TOML as a quoted STRING and the
    emitted namelist.wps as ``array(170.)``.
    """

    rc, out = _run_wizard(tmp_path, point="35.3,-190.0")
    captured = capsys.readouterr()
    assert rc == 0
    assert out.exists()
    assert "warning:" in captured.err
    assert "wrapped to 170" in captured.err

    projection = tomllib.loads(out.read_text())["projection"]
    for key in ("ref_lon", "stand_lon", "ref_lat", "truelat1", "truelat2"):
        assert isinstance(projection[key], float), (key, projection[key])
    assert projection["ref_lon"] == pytest.approx(170.0)

    namelist = (out.parent / f"{out.stem}.namelist.wps").read_text()
    for line in namelist.splitlines():
        key, _, value = line.partition("=")
        if key.strip() in ("ref_lat", "ref_lon", "truelat1", "truelat2",
                           "stand_lon"):
            # Fortran has to be able to read it.
            assert float(value.strip().rstrip(",")) == pytest.approx(
                projection[key.strip()])


def test_a_wrapped_longitude_matches_its_unwrapped_twin_byte_for_byte(
        tmp_path, capsys):
    """--point 35.3,-190 and --point 35.3,170 name the same meridian, so
    the two emissions may not differ in type or in text."""

    rc_a, out_a = _run_wizard(tmp_path / "a", point="35.3,-190.0")
    rc_b, out_b = _run_wizard(tmp_path / "b", point="35.3,170.0")
    capsys.readouterr()
    assert (rc_a, rc_b) == (0, 0)
    assert tomllib.loads(out_a.read_text())["projection"] == \
        tomllib.loads(out_b.read_text())["projection"]
    assert (out_a.parent / f"{out_a.stem}.namelist.wps").read_text() == \
        (out_b.parent / f"{out_b.stem}.namelist.wps").read_text()


def test_toml_emitter_refuses_to_quote_a_value_it_cannot_type(tmp_path):
    """A number emitted as a string is valid TOML under the right key
    and the wrong type, so it survives review.  The emitter renders
    scalars it recognises and refuses the rest rather than quoting."""
    from woof.domain_wizard import _toml_value

    assert _toml_value(np.float32(-160.0)) == repr(-160.0)
    assert _toml_value(np.asarray(-160.0)) == repr(-160.0)
    assert _toml_value(np.int64(7)) == "7"
    assert _toml_value("lambert") == '"lambert"'
    with pytest.raises(TypeError, match="cannot render"):
        _toml_value(np.asarray([1.0, 2.0]))
    with pytest.raises(TypeError, match="cannot render"):
        _toml_value(object())


def test_point_longitude_refusal_names_the_range_it_enforces(
        tmp_path, capsys):
    """One wrap is accepted, so a message claiming [-180, 180] alone
    described a refusal that does not happen."""
    rc, out = _run_wizard(tmp_path, point="35.3,-400.0")
    _assert_refused(capsys, "[-360, 360]", rc)
    assert not out.exists()
    # and the accepted-with-a-wrap case really is accepted
    rc_ok, out_ok = _run_wizard(tmp_path / "ok", point="35.3,270.0")
    capsys.readouterr()
    assert rc_ok == 0 and out_ok.exists()


def test_card_and_vram_gib_are_mutually_exclusive(tmp_path, capsys):
    rc, _ = _run_wizard(tmp_path, "--vram-gib", "20")
    _assert_refused(capsys, "mutually exclusive", rc)


def test_vram_below_reserve_rejected(tmp_path, capsys):
    """A card too small to size is refused, and says which wall it hit.

    Two walls now, not one.  Below one CUDA context plus the external
    margin there is nothing to size against at all; above that, the fit
    loop refuses and names the layout, the arithmetic and the share of it
    that no smaller grid can move.  The flat 4 GiB reserve used to draw
    the line for both, which refused cards the suite-priced reserve would
    have sized.
    """
    out = tmp_path / "area.toml"
    # Bare default: one preset ("12"), so the fit loop's own refusal
    # propagates directly, naming the preset and the reserve arithmetic.
    rc = cli_main(["domain", "--point", "39.7,-96.6", "--vram-gib", "2.5",
                   "--cycle", "1999-05-03T12", "--out", str(out)])
    _assert_refused(capsys, "no budget for ladder 12 at all", rc)
    assert not out.exists()

    # Explicit auto: the walk tries every preset and the final refusal
    # reports that even the shallowest one's smallest layout is over.
    rc = cli_main(["domain", "--point", "39.7,-96.6", "--vram-gib", "2.5",
                   "--ladder", "auto",
                   "--cycle", "1999-05-03T12", "--out", str(out)])
    _assert_refused(capsys, "smallest layout exceeds the budget", rc)
    assert not out.exists()

    rc = cli_main(["domain", "--point", "39.7,-96.6", "--vram-gib", "0.9",
                   "--cycle", "1999-05-03T12", "--out", str(out)])
    _assert_refused(capsys, "leaves no budget", rc)
    assert not out.exists()


def test_bad_cycle_and_gfs_synoptic_hours(tmp_path, capsys):
    rc, _ = _run_wizard(tmp_path, cycle="not-a-time")
    _assert_refused(capsys, "YYYY-MM-DDTHH", rc)
    # GFS cycles are synoptic-only; parse_cycle enforces per source.
    rc, _ = _run_wizard(tmp_path, source="gfs", cycle="2026-07-28T05")
    _assert_refused(capsys, "00/06/12/18", rc)


def test_hours_minimum(tmp_path, capsys):
    rc, _ = _run_wizard(tmp_path, "--hours", "0")
    _assert_refused(capsys, "at least 1", rc)


def test_antimeridian_footprint_emits_wrapping_fetch_box(tmp_path):
    # Worldwide contract: a footprint straddling 180E is supported end
    # to end; the fetch hint wraps (W > E in the signed convention).
    rc, out = _run_wizard(tmp_path, point="52.0,179.5")
    assert rc == 0
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    s, w, n, e = (float(v) for v in raw["fetch"]["area"].split(","))
    assert w > 0.0 > e, (w, e)  # crossing box: west near 170E, east near -170W
    from woof.fetch import parse_area
    area = parse_area(raw["fetch"]["area"])
    assert area.crosses_antimeridian


def test_negative_coordinates_parse_in_both_forms(tmp_path, capsys):
    """`--point -33.87,151.21` must work, not just `--point=-33.87,...`.

    argparse reads a leading `-` as an option prefix unless the token
    matches its negative-number regex, which a `lat,lon` pair never
    does.  Every documented example was a positive CONUS latitude, so
    the whole southern hemisphere failed with "expected one argument"
    -- on the release whose headline claim is worldwide forecasts.
    """
    from woof.cli import _join_negative_coordinates

    assert _join_negative_coordinates(
        ["domain", "--point", "-33.87,151.21"]
    ) == ["domain", "--point=-33.87,151.21"]
    assert _join_negative_coordinates(
        ["fetch", "--area", "-58.58,119.65,-8.38,-177.23"]
    ) == ["fetch", "--area=-58.58,119.65,-8.38,-177.23"]
    # Positive values and non-coordinate flags are left exactly alone.
    assert _join_negative_coordinates(
        ["domain", "--point", "35.3,-97.5", "--hours", "6"]
    ) == ["domain", "--point", "35.3,-97.5", "--hours", "6"]
    # A following token that is not all-numeric stays an option string,
    # so `--point --help` still errors the way it should.
    assert _join_negative_coordinates(
        ["domain", "--point", "--help"]
    ) == ["domain", "--point", "--help"]

    sydney = tmp_path / "spaced"
    sydney.mkdir()
    out = sydney / "area.toml"
    rc = cli_main(["domain", "--point", "-33.87,151.21", "--card", "24gb",
                   "--ladder", "12", "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--out", str(out), "--explain"])
    assert rc == 0, capsys.readouterr()
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert raw["projection"]["ref_lat"] == pytest.approx(-33.87)
    assert raw["projection"]["map_proj"] == "lambert"
    # Hemisphere-correct: southern standard parallels.
    assert raw["projection"]["truelat1"] < 0.0
    assert raw["projection"]["truelat2"] < 0.0
    # And the printed next: line is pasteable -- negative area in = form.
    printed = capsys.readouterr().out
    assert "--area=-" in printed


def test_point_refusal_names_the_equals_form(tmp_path, capsys):
    rc, _ = _run_wizard(tmp_path, point="35.3")
    assert rc == 2
    err = capsys.readouterr().err
    assert "--point=-33.87,151.21" in err


def test_cycle_latest_resolves_instead_of_contradicting_itself(
        tmp_path, capsys, monkeypatch):
    """v1.0.0: "--cycle 'latest' must be YYYY-MM-DDTHH (UTC) or 'latest'".

    Worse than self-contradictory: the documented order is
    wizard-then-fetch, so nothing told a user which cycle was current
    and they had to run a throwaway fetch to find out.  The resolver
    already existed.
    """
    import woof.fetch as fetch_module
    from datetime import datetime

    resolved = datetime(2026, 7, 29, 18)
    calls = []

    def fake_resolve(source, last_hour, **kwargs):
        calls.append((source, last_hour))
        return resolved

    monkeypatch.setattr(fetch_module, "resolve_latest_cycle", fake_resolve)
    rc, out = _run_wizard(tmp_path, "--explain", source="gfs", cycle="latest",
                          ladder="12")
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert calls == [("gfs", 6)]
    assert "--cycle latest resolved to 2026-07-29T18Z" in printed
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    # The emitted config records the resolved time, never the query.
    assert raw["experiment"]["start_time"] == resolved
    assert raw["fetch"]["cycle"] == "2026-07-29T18"
    assert "--cycle 2026-07-29T18" in printed


def test_cycle_latest_resolves_for_a_reanalysis(tmp_path, capsys):
    """A reanalysis has a latest, and this door writes it into the file.

    It used to refuse here -- "a reanalysis with weeks of latency" --
    which is a statement about a DELAY, and a delay is a number, and a
    number resolves.  The wizard asks the resolver rather than carrying
    a list of the sources that may ask.
    """

    from datetime import datetime, timedelta, timezone

    rc, out = _run_wizard(tmp_path, source="era5", cycle="latest")
    assert rc == 0, capsys.readouterr().err
    emitted = out.read_text(encoding="utf-8")
    # The RESOLVED cycle, never the literal query.  (Not a bare "latest"
    # search: pytest's tmp_path carries this test's own name.)
    assert 'cycle = "latest"' not in emitted
    printed = capsys.readouterr().out
    assert "--cycle latest resolved to" in printed
    start = [line for line in emitted.splitlines()
             if line.startswith("start_time = ")][0]
    resolved = datetime.strptime(start.split(" = ")[1].strip(),
                                 "%Y-%m-%dT%H:%M:%S")
    # Behind real time by the delay the row declares, on its own grid.
    assert resolved < datetime.now(timezone.utc).replace(
        tzinfo=None) - timedelta(days=4)
    assert resolved.hour in (0, 6, 12, 18)


def test_every_source_this_door_can_plan_can_also_resolve_latest():
    """`latest` is offered exactly where it resolves, with no list.

    The interactive door defaults the cycle question to ``latest`` for
    any source it can plan; a source it offers that cannot resolve one
    would default the reader straight into a refusal.  The two used to
    be kept in step by hand -- one branch naming era5, another asking
    whether a fetch door existed -- which is how a reanalysis came to be
    offered nothing at all.
    """

    from woof import domain_interactive
    from woof.domain_wizard import planable_sources
    from woof.source_cycles import cycle_grid_for

    for source in planable_sources():
        offered = domain_interactive.default_cycle_answer(source)
        resolves = cycle_grid_for(source) is not None
        assert (offered == "latest") is resolves, (
            f"{source} is offered {offered!r} but "
            f"{'can' if resolves else 'cannot'} resolve one")
    # Not vacuous: the reanalysis that used to be excluded by name is
    # offered `latest` now, and it resolves.
    assert domain_interactive.default_cycle_answer("era5") == "latest"


def test_tropical_points_get_the_halved_root_clock(tmp_path, capsys):
    """|lat| < 25 emits 2.5 s/km, with the reason in the file.

    A 12 km Mercator domain at Manila on the wizard's own 60 s clock
    destabilised at +1 h; the same domain at a shorter step completed
    6 h.  The emitted rationale names the co-located v1.1 CFL gate.
    """
    from woof.domain_wizard import (ROOT_TIME_STEP_S,
                                     TROPICAL_ROOT_TIME_STEP_S,
                                     root_time_step_s)

    assert TROPICAL_ROOT_TIME_STEP_S * 2 == ROOT_TIME_STEP_S
    assert root_time_step_s(14.6) == TROPICAL_ROOT_TIME_STEP_S
    assert root_time_step_s(-14.6) == TROPICAL_ROOT_TIME_STEP_S
    assert root_time_step_s(24.99) == TROPICAL_ROOT_TIME_STEP_S
    assert root_time_step_s(25.0) == ROOT_TIME_STEP_S
    assert root_time_step_s(-33.87) == ROOT_TIME_STEP_S

    rc, out = _run_wizard(tmp_path / "manila", point="14.6,120.98",
                          source="gfs", cycle="2026-07-29T18")
    assert rc == 0, capsys.readouterr()
    text = out.read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    assert raw["domain"][0]["time_step"] == TROPICAL_ROOT_TIME_STEP_S
    assert "TROPICAL CLOCK" in text
    assert "co-located vertical" in text
    # The chain still derives exactly, and the config still loads.
    exp = experiment_from_text(text, source=str(out))
    assert exp.root.time_step == TROPICAL_ROOT_TIME_STEP_S
    assert float(exp.dt_exact(2)) == TROPICAL_ROOT_TIME_STEP_S / 4

    # A mid-latitude point is untouched by all of this.
    rc, out = _run_wizard(tmp_path / "kansas", point="39.7,-96.6",
                          source="gfs", cycle="2026-07-29T18")
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert tomllib.loads(text)["domain"][0]["time_step"] == ROOT_TIME_STEP_S
    assert "TROPICAL CLOCK" not in text


@pytest.mark.parametrize("lat", [90.0, -90.0, 89.0])
def test_pole_containing_polar_domain_roundtrips(tmp_path, capsys, lat):
    rc, out = _run_wizard(tmp_path, point=f"{lat},0", ladder="12",
                          source="gfs", cycle="2026-07-29T18")
    assert rc == 0, capsys.readouterr().err
    raw = tomllib.loads(out.read_text())
    assert raw["projection"]["map_proj"] == "polar"
    south, west, north, east = map(float, raw["fetch"]["area"].split(","))
    assert (west, east) == (-180.0, 180.0)
    assert (north if lat > 0 else south) == (90.0 if lat > 0 else -90.0)
    exp = load_experiment(out)
    assert exp.projection.map_proj == "polar"
    grids = grids_from_wps_namelist(out.with_suffix(".namelist.wps"))
    assert grids[0].map_proj == "polar"


def test_margined_span_over_180_is_never_emitted_as_a_flipped_box():
    """The emission gate itself, still armed and still a refusal.

    Audit reproduction point: auto projection, point 34,0, GFS.  The
    880x704 root's raw span (165.1 deg) fits, but the GFS source margin
    (15 deg per side) pushes the box to 195.1 deg -- and
    :func:`woof.fetch.parse_area` reads a box that wide back as the
    complementary antimeridian crossing, which is the wrong crop with
    nothing to signal it.  ``_fetch_area`` refuses to write one.  This
    is the gate; the test below is about who is supposed to hit it.
    """
    from woof.domain_wizard import _fetch_area, _fetch_margin_deg, \
        _projection_entries

    projection = _projection_entries(34.0, 0.0)
    with pytest.raises(ValueError, match="boxes wider than 180 degrees"):
        _fetch_area(projection, 880, 704,
                    margin_deg=_fetch_margin_deg("gfs"))


def test_a_card_filling_gfs_layout_uses_existing_full_longitude_coverage(
        tmp_path, capsys):
    """A GFS forcing band may widen without resizing the forecast to fit it.

    Driven from a DRAWN area since 2.7.3.  The layout this needs is one
    whose forcing box, once the GFS margin is added, has outgrown a
    servable crop -- and a `--point` can no longer produce one, because
    a centre carries no extent and the fit now bounds the extent it
    chooses (`POINT_FIT_MAX_EXTENT_KM`).  A drawn area carries its own
    extent and is still sized to the drawing, so the property lives
    where the layout still comes from.
    """
    polygon = tmp_path / "wide.geojson"
    polygon.write_text(json.dumps({
        "type": "Polygon",
        "coordinates": [[[-70.0, 20.0], [70.0, 20.0], [70.0, 48.0],
                         [-70.0, 48.0], [-70.0, 20.0]]]}), encoding="utf-8")
    out = tmp_path / "area.toml"
    rc = cli_main([
        "domain", f"--polygon={polygon}", "--vram-gib", "64",
        "--ladder", "12", "--source", "gfs", "--cycle", "2026-07-28T06",
        "--out", str(out)])
    assert rc == 0, capsys.readouterr().out
    area = tomllib.loads(out.read_text(encoding="utf-8"))["fetch"]["area"]
    south, west, north, east = (float(v) for v in area.split(","))
    assert (west, east) == (-180.0, 180.0), area
    from woof.fetch import parse_area
    parsed = parse_area(area)
    assert not parsed.crosses_antimeridian
    assert parsed.lon_west == pytest.approx(west, abs=0.01)
    assert parsed.lon_east == pytest.approx(east, abs=0.01)


def test_the_servable_crop_bound_is_the_same_arithmetic_as_the_gate():
    """One expression, two readers -- the drift this pair forbids.

    A sizing bound computed one way and an emission gate computed
    another is how a wizard sizes for twelve seconds and then refuses
    the file it just sized.  Both are :func:`_margined_longitude_span`.
    """
    from woof.domain_wizard import (_fetch_area, _fetch_margin_deg,
                                     _projection_entries,
                                     fetch_crop_refusal)

    projection = _projection_entries(34.0, 0.0)
    margin = _fetch_margin_deg("gfs")
    for nx in range(560, 900, 8):
        refused = fetch_crop_refusal(projection, nx, 704, source="gfs")
        try:
            _fetch_area(projection, nx, 704, margin_deg=margin,
                        allow_full_longitude=True)
        except ValueError as error:
            assert refused is not None, nx
            assert "180 degrees" in str(error)
        else:
            assert refused is None, (nx, refused)


def test_saved_southern_gfs_polygon_widens_only_forcing_longitude():
    from woof.domain_wizard import (fetch_area_hint, fetch_crop_refusal,
                                     _projection_entries, _root_grid,
                                     _fetch_margin_deg, max_fetch_abs_lat)
    from woof.fetch import parse_area, validate_fetch_hints
    projection = _projection_entries(-39.91377935078366, -23.72743785729405)
    before = dict(projection)
    notes = []
    hint = fetch_area_hint(projection, 874, 574, source="gfs",
                           root_dx_m=12000.0, target_option="--polygon", notes=notes)
    area = parse_area(hint)
    assert (area.lon_west, area.lon_east, area.longitude_span_degrees) == (-180.0, 180.0, 360.0)
    box = area.as_nomads()
    assert (box["left_lon"], box["right_lon"]) == (0.0, 360.0)
    lat, _ = _root_grid(projection, 874, 574, 12000.0).latlon_c()
    margin = _fetch_margin_deg("gfs")
    pole = max_fetch_abs_lat(12000.0)
    assert area.lat_south == pytest.approx(max(-pole, float(lat.min()) - margin), abs=.005)
    assert area.lat_north == pytest.approx(min(pole, float(lat.max()) + margin), abs=.005)
    validate_fetch_hints({"source": "gfs", "cycle": "2026-09-08T18", "hours": 6,
                          "area": hint, "out": "unused", "cadence": 3}, source="<global-band-test>")
    assert fetch_crop_refusal(projection, 874, 574, source="gfs", root_dx_m=12000.0) is None
    assert any("only forcing coverage is expanded" in note for note in notes)
    assert projection == before


def test_fetch_area_just_under_the_limit_round_trips_unflipped():
    """Spans just under the 180-degree refusal must survive the
    emit -> parse_area round trip as the same box (parse_area flips
    only spans OVER 180 into the complementary crossing)."""
    from woof.domain_wizard import _fetch_area, _fetch_margin_deg, \
        _projection_entries
    from woof.fetch import parse_area

    projection = _projection_entries(34.0, 0.0)
    margin = _fetch_margin_deg("gfs")
    area = None
    for nx in range(880, 400, -8):  # widest layout the margined gate admits
        try:
            area = _fetch_area(projection, nx, 704, margin_deg=margin)
        except ValueError:
            continue
        break
    assert area is not None, "no layout fit under the margined gate"
    lat_s, lon_w, lat_n, lon_e = area
    assert lon_w < 0.0 < lon_e  # centered on ref_lon = 0, not crossing
    span = lon_e - lon_w
    assert 160.0 < span <= 180.0, span  # genuinely near the limit
    parsed = parse_area(",".join(f"{v:.2f}" for v in area))
    assert not parsed.crosses_antimeridian
    assert parsed.lon_west == pytest.approx(lon_w, abs=0.01)
    assert parsed.lon_east == pytest.approx(lon_e, abs=0.01)


def test_projection_auto_selection_bands():
    from woof.domain_wizard import _projection_entries, auto_projection

    assert auto_projection(1.3) == "mercator"
    assert auto_projection(-17.8) == "mercator"
    assert auto_projection(-27.5) == "lambert"
    assert auto_projection(39.7) == "lambert"
    assert auto_projection(64.8) == "polar"
    assert auto_projection(-77.85) == "polar"
    # Hemisphere-correct Lambert truelats (both signed with the point).
    sh = _projection_entries(-27.5, 153.0)
    assert sh["map_proj"] == "lambert"
    assert sh["truelat1"] == -17.5 and sh["truelat2"] == -37.5
    # Explicit override wins.
    forced = _projection_entries(-27.5, 153.0, "mercator")
    assert forced["map_proj"] == "mercator"
    assert forced["truelat1"] == -27.5
    with pytest.raises(ValueError, match="--projection"):
        _projection_entries(10.0, 0.0, "cassini")


@pytest.mark.parametrize("point, source, cycle, map_proj, wrf_code", [
    ("-27.5,153.0", "gfs", "2026-07-28T06", "lambert", 1),
    ("1.3,103.8", "gfs", "2026-07-28T06", "mercator", 3),
    ("64.8,-147.7", "gfs", "2026-07-28T06", "polar", 2),
    ("-17.8,178.5", "gfs", "2026-07-28T06", "mercator", 3),
])
def test_worldwide_points_emit_and_round_trip(tmp_path, point, source,
                                              cycle, map_proj, wrf_code):
    """The four worldwide gate sites (Brisbane, Singapore, Fairbanks,
    Fiji) emit, declare the right projection, and round-trip through
    the real loaders and grid builders."""
    rc, out = _run_wizard(tmp_path, "--ladder", "12", point=point,
                          source=source, cycle=cycle)
    assert rc == 0
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert raw["projection"]["map_proj"] == map_proj
    assert raw["shared"]["map_proj"] == wrf_code
    # New single-domain preparations have a usable checkpoint cadence.
    assert raw["experiment"]["restart_interval_s"] == 3600.0
    exp = load_experiment(out)
    grids = grids_from_projection_config(exp)
    assert len(grids) == 1
    wps = out.parent / f"{out.stem}.namelist.wps"
    wps_text = wps.read_text(encoding="utf-8")
    assert f"map_proj = '{map_proj}'" in wps_text
    wps_grids = grids_from_wps_namelist(wps)
    lat_a, lon_a = grids[0].latlon_mass()
    lat_b, lon_b = wps_grids[0].latlon_mass()
    np.testing.assert_allclose(lat_a, lat_b, rtol=0, atol=1e-9)
    np.testing.assert_allclose(lon_a, lon_b, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# Layout invariants (quantization rules the loader will re-check)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ladder", sorted(LADDER_RATIOS))
@pytest.mark.parametrize("scale", [0.55, 1.0, 2.3, 5.7])
def test_dims_even_divisible_and_clear(ladder, scale):
    ratios = LADDER_RATIOS[ladder]
    dims = _dims_for_scale(scale, ratios)
    assert len(dims) == len(ratios) + 1
    for nx, ny in dims:
        assert nx % 2 == 0 and ny % 2 == 0
    for (pnx, pny), (nx, ny), ratio in zip(dims, dims[1:], ratios):
        assert nx % ratio == 0 and ny % ratio == 0
        # Centered child leaves >= 10 parent rows (Davies + blend zones).
        assert (pnx - nx // ratio) // 2 >= 10
        assert (pny - ny // ratio) // 2 >= 10


# ---------------------------------------------------------------------------
# Sizing fit: emitted dims' estimator envelope fits the card budget
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("card", sorted(CARD_VRAM_GIB))
def test_fit_fills_card_budget(tmp_path, monkeypatch, card):
    rc, out = _run_wizard(tmp_path, card=card, ladder="auto")
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    exp = experiment_from_text(text, source=str(out))
    vram = CARD_VRAM_GIB[card]
    # The budget is the CANDIDATE's own -- its suite's reserve out of the
    # free VRAM a card that size really presents, not the nameplate.
    free_bytes = int(card_assumed_free_gib(vram) * GIB)
    with _runtime_card_pricing(monkeypatch, vram_gib=vram,
                               free_bytes=free_bytes):
        estimate = estimate_experiment(exp, forcing_interval_seconds=21600.0,
                                       vram_gib=vram)
        budget = sizing_budget_bytes(exp, free_bytes=free_bytes, vram_gib=vram,
                                     forcing_interval_seconds=21600.0)
    envelope = estimate.peak_envelope_bytes
    assert envelope <= budget
    # ...and it must stop SHORT of it.  Every ladder v1.4.0 emitted landed
    # 0.01-0.19 GiB from the wall, which is a rounding error away from a
    # refusal on the machine that then runs it.
    assert envelope <= budget - fit_headroom_bytes(budget), (
        "the fit loop must leave headroom, not touch the budget")
    # The bisection must still actually spend the budget, not stop at the
    # floor: an envelope model is not licence to size timidly.
    assert envelope >= 0.7 * budget
    # Certified clock/dx conventions on the emitted chain.
    root = exp.root
    assert root.time_step == 60 and root.run.dx == 12000.0
    for dc in exp.domains:
        assert exp.dx_exact(dc.grid_id) == exp.dx_exact(1) / np.prod(
            [d.parent_grid_ratio for d in exp.domains
             if 1 < d.grid_id <= dc.grid_id], dtype=int)


def test_the_wizard_budgets_with_the_platform_envelope_factor(
        tmp_path, capsys, monkeypatch):
    """Same card, same ladder: each platform sizes and says what it applied.

    HISTORY, because the assertion inverted.  This test used to demand
    ``linux_cells > windows_cells``, from the era when Windows carried a
    1.75 multiplier and 4.12 GiB of pool constants that Linux did not.
    Both of those are retired, and 2026-08-20 (task 206) measured the
    last surviving difference -- the pool-slack fraction -- and found it
    is not a driver-model property at all: it is the legacy-RRTMG lane's
    retained call-peak workspace, and Linux showed the same 1.17-1.19x
    the WDDM calibration had measured.  Charging it by platform meant
    Linux under-predicted every one of fifteen instrumented forecasts.

    So the two platforms now size the SAME grid for the same card and
    the same suite, and what this test holds is that each still prices
    its own envelope, names its own basis, and lands inside its own
    budget with the grid it chose.
    """
    import woof.core.preflight as pf

    _as_windows(monkeypatch, pf)
    rc, out = _run_wizard(tmp_path / "win", "--explain", card="24gb")
    windows_out = capsys.readouterr().out
    assert rc == 0
    windows_exp = experiment_from_text(
        out.read_text(encoding="utf-8"), source=str(out))
    assert "peak envelope" in windows_out
    assert "envelope basis: windows;" in windows_out

    monkeypatch.setattr(pf.sys, "platform", "linux")
    rc, out = _run_wizard(tmp_path / "lin", "--explain", card="24gb")
    linux_out = capsys.readouterr().out
    assert rc == 0
    linux_exp = experiment_from_text(
        out.read_text(encoding="utf-8"), source=str(out))
    assert "local-memory backing store" in linux_out
    assert "envelope basis: linux;" in linux_out

    windows_cells = windows_exp.root.run.nx * windows_exp.root.run.ny
    linux_cells = linux_exp.root.run.nx * linux_exp.root.run.ny
    assert linux_cells == windows_cells

    # Each still fits its own platform's budget, measured by the
    # estimator -- which now itemizes differently per platform too, so
    # each config has to be re-priced under the platform that built it.
    vram = CARD_VRAM_GIB["24gb"]
    free_bytes = int(card_assumed_free_gib(vram) * GIB)
    for exp, platform in ((windows_exp, "win32"), (linux_exp, "linux")):
        monkeypatch.setattr(pf.sys, "platform", platform)
        with _runtime_card_pricing(monkeypatch, vram_gib=vram,
                                   free_bytes=free_bytes):
            estimate = estimate_experiment(exp, forcing_interval_seconds=21600.0,
                                           vram_gib=vram)
            budget = sizing_budget_bytes(
                exp, free_bytes=free_bytes, vram_gib=vram,
                forcing_interval_seconds=21600.0)
        assert estimate.peak_envelope_bytes <= budget, platform
        assert estimate.peak_envelope_bytes >= 0.7 * budget, platform


@pytest.mark.parametrize("ladder", sorted(LADDER_RATIOS))
def test_windows_12gib_sizes_under_the_measured_model(
        tmp_path, capsys, monkeypatch, ladder):
    """A 12 GiB Windows card sizes, with the measured accounting.

    Its predecessor was an EXPERIMENTAL tier whose advisory asked for
    exactly one measurement -- a small Windows card's real peak.  The
    2026-08-19 RTX 3080 calibration delivered it (six whole forecasts,
    machine-wide sampling), so the tier is retired into the one measured
    Windows model and the pioneer warning is gone: the accounting is no
    longer a guess, and saying it is would be false.
    """
    import woof.core.preflight as pf

    _as_windows(monkeypatch, pf)
    rc, out = _run_wizard(tmp_path / ladder, "--explain", card="12gb",
                          ladder=ladder)
    assert rc == 0, ladder
    printed = capsys.readouterr().out

    assert "peak envelope" in printed
    assert "envelope basis: windows;" in printed
    # The pool-slack term appears when the SUITE has the mechanism, not
    # when the driver model does: it is the legacy-RRTMG engines'
    # retained call-peak workspace (task 206).  The wizard's default
    # suite is rte-rrtmgp, so this card is not charged for it -- and on
    # Linux it now would be charged, identically, if it were.
    assert "pool slack" not in printed
    assert "small-card threshold" not in printed
    assert "Please report your measured peak" not in printed
    assert "windows-small" not in printed

    # And the emitted config really fits the measured accounting.
    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    free_bytes = int(card_assumed_free_gib(12.0) * GIB)
    with _runtime_card_pricing(monkeypatch, vram_gib=12.0,
                               free_bytes=free_bytes):
        estimate = estimate_experiment(exp, forcing_interval_seconds=21600.0,
                                       vram_gib=12.0)
        budget = sizing_budget_bytes(
            exp, free_bytes=free_bytes, vram_gib=12.0,
            forcing_interval_seconds=21600.0)
    assert estimate.peak_envelope_bytes <= budget


def test_every_windows_card_takes_the_one_measured_accounting(
        tmp_path, capsys, monkeypatch):
    """Card size never selects a different formula (the #162 split)."""
    import woof.core.preflight as pf

    _as_windows(monkeypatch, pf)
    rc, _ = _run_wizard(tmp_path / "sixteen", "--explain", card="16gb")
    assert rc == 0
    printed = capsys.readouterr().out
    assert "peak envelope" in printed
    assert "envelope basis: windows;" in printed
    assert "EXPERIMENTAL" not in printed

    # The accounting seam itself, without going through the wizard: one
    # family per platform, whatever the card and whether one was named
    # at all (`woof check` measures free VRAM and names no card).
    for vram in (11.0, 12.0, 16.0, None):
        assert pf.envelope_platform("win32", vram) == "windows"
    assert pf.envelope_platform("linux", 12.0) == "linux"
    # The 5090-derived projection constants stay display-only, and they
    # no longer depend on the card either.
    for vram in (12.0, 16.0, None):
        assert pf.platform_projection_constants("win32", vram) == (
            pf.pool_retention_residual_bytes(),
            pf.PROBE_DEVICE_OVERHEAD_BYTES)


def test_auto_picks_deepest_ladder_on_32gb(tmp_path):
    rc, out = _run_wizard(tmp_path, card="32gb", ladder="auto")
    assert rc == 0
    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    assert len(exp.domains) == 4  # 12-3-1-0.5
    assert [dc.parent_grid_ratio for dc in exp.domains] == [1, 4, 3, 2]
    assert float(exp.dx_exact(4)) == 500.0


def test_no_ladder_flag_emits_the_single_domain_go_shape(tmp_path):
    """The flags door's DEFAULT is one 12 km domain, not the deepest tree.

    `--ladder auto` used to be the default, so `woof domain --point ...
    --card 24gb --source gfs` -- the obvious first invocation -- emitted
    a four-domain tree that `woof go` then refused (4090 user-zero
    stress run, 2026-08-03).  The interactive door was already fixed for
    exactly this (domain_interactive.DEFAULT_LADDER = "12"); the flags
    door now agrees.  Trees are explicit opt-in: --ladder (including
    `auto`, unchanged above) or --root-dx/--chain.
    """

    out = tmp_path / "bare.toml"
    rc = cli_main([
        "domain", "--point=39.7,-96.6", "--card", "24gb",
        "--source", "gfs", "--cycle", "2026-07-29T18", "--out", str(out)])
    assert rc == 0
    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    assert len(exp.domains) == 1
    assert exp.root.run.dx == 12000.0
    assert exp.restart_interval_s == 3600.0


# ---------------------------------------------------------------------------
# The default physics suite, pinned
# ---------------------------------------------------------------------------

def test_wizard_default_suite_is_the_certified_full_radiation_profile(
        tmp_path):
    """The wizard's real-case default is the certified Morrison profile
    (owner directive 2026-08-06, after a shipped 48 h case ran a
    longwave-OFF validation suite through two nights): a shipped
    ``wrf-matched-run`` template with BOTH radiation streams on, emitted
    switch for switch from woof.physics_compat so the emitted file
    passes the runners' profile guard as written, and re-derived from
    the registry here so the choice cannot outlive its evidence."""
    from woof.domain_wizard import DEFAULT_PHYSICS_PROFILE
    from woof.physics_compat import (MORRISON_PROFILE_ID,
                                      single_domain_runtime_switches)
    from woof.physics_registry import physics_registry

    assert DEFAULT_PHYSICS_PROFILE == MORRISON_PROFILE_ID

    rc, out = _run_wizard(tmp_path, "--explain")
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    exp = experiment_from_text(text, source=str(out))
    root = exp.domains[0].run

    # Registry evidence, not vibes: the default template is certified
    # (top conformance rung) and runs full lw+sw radiation.
    template = physics_registry()["templates"][DEFAULT_PHYSICS_PROFILE]
    assert template["maturity"] == "wrf-matched-run"

    from woof.config import radiation_scheme_ids
    switches = single_domain_runtime_switches(DEFAULT_PHYSICS_PROFILE)
    assert switches["ra_lw_physics"] == root.ra_lw_physics == 4
    assert switches["ra_sw_physics"] == root.ra_sw_physics == 4
    assert root.ra_physics == 0
    assert radiation_scheme_ids(root) == (4, 4)
    # Kain-Fritsch on the 12 km root only; every nest runs cumulus off.
    assert root.cu_physics == switches["cu_physics"] == 1
    assert all(dc.run.cu_physics == 0 for dc in exp.domains[1:])
    # Every remaining profile switch lands on the root verbatim (the
    # per-domain cadence keys are carried by the [[domain]] tables).
    for key, value in switches.items():
        assert getattr(root, key) == value, key
    assert all(dc.run.mp_physics == 10 for dc in exp.domains)
    # The default emission is nocturnally valid and says so in ink; the
    # declared-experiment acknowledgement belongs only to explicitly
    # selected asymmetric suites.
    assert "# NOCTURNALLY VALID" in text
    assert "acknowledgements" not in text
    # Product decision (STEP17): wizard configs ship the UP_HELI_MAX
    # diagnostic ON -- this audience reads UH products.
    assert all(dc.run.nwp_diagnostics == 1 for dc in exp.domains)


# ---------------------------------------------------------------------------
# Round trips through the real loaders
# ---------------------------------------------------------------------------

def test_emitted_era5_config_round_trips(tmp_path, capsys):
    geog = tmp_path / "GEOG"
    geog.mkdir()
    rc, out = _run_wizard(tmp_path, "--geog-root", str(geog),
                          ladder="12-3-1")
    assert rc == 0
    printed = capsys.readouterr().out
    # Forcing is not on disk yet.  By default the deferral is not a
    # stanza of its own -- it is step 2 of the next-steps block, with
    # the exact follow-up command and the note that it waits on step 1.
    assert "woof check: deferred" not in printed
    assert "woof check" in printed and "--free-gib" in printed
    assert "after the fetch lands" in printed

    # --explain restores the inventory and the geog story, verbatim.
    rc, _ = _run_wizard(tmp_path, "--explain", "--geog-root", str(geog),
                        ladder="12-3-1")
    assert rc == 0
    explained = capsys.readouterr().out
    assert "woof check: deferred" in explained
    assert "WPS_GEOG" in explained

    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert raw["fetch"]["source"] == "era5"
    assert raw["case_data"]["wps_namelist"] == "area.namelist.wps"
    # The packaged ERA5 Vtable was copied beside the TOML and parses.
    vtable = out.parent / "Vtable.ERA5_CDO"
    assert vtable.is_file()
    assert len(parse_vtable(vtable)) > 20

    # Create the declared forcing; the full case loader then accepts the
    # emitted file as-is ([fetch] split off and validated, not rejected).
    forcing = out.parent / Path(raw["case_data"]["forcing"][0])
    forcing.parent.mkdir(parents=True, exist_ok=True)
    forcing.write_bytes(b"stub")
    exp, data = load_experiment_case(out)
    assert len(exp.domains) == 3
    assert exp.run_seconds == 6 * 3600.0
    assert data.geog_root == geog
    assert data.source_orography is None  # era5_z_invariant provider
    assert data.forcing_interval_s == 21600.0


def test_wps_namelist_agrees_with_projection_config(tmp_path):
    rc, out = _run_wizard(tmp_path, ladder="12-3-1")
    assert rc == 0
    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    from_toml = grids_from_projection_config(exp)
    from_wps = grids_from_wps_namelist(out.parent / "area.namelist.wps")
    assert len(from_toml) == len(from_wps) == 3
    for a, b in zip(from_toml, from_wps):
        assert (a.e_we, a.e_sn, a.dx) == (b.e_we, b.e_sn, b.dx)
        for attr in ("ref_lat", "ref_lon", "truelat1", "truelat2",
                     "stand_lon", "known_x", "known_y"):
            assert getattr(a, attr) == pytest.approx(
                getattr(b, attr), abs=1e-9), attr
        lat_a, lon_a = a.latlon_mass()
        lat_b, lon_b = b.latlon_mass()
        np.testing.assert_allclose(lat_a, lat_b, atol=1e-9)
        np.testing.assert_allclose(lon_a, lon_b, atol=1e-9)


def test_children_are_centered_on_the_point(tmp_path):
    rc, out = _run_wizard(tmp_path, ladder="12-3-1", point="39.7,-96.6")
    assert rc == 0
    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    for grid in grids_from_projection_config(exp):
        # The wizard's centering arithmetic makes every domain's grid
        # center coincide with the parent's exactly (child center in
        # parent coordinates = P/2 + 0.5 = the parent center), so each
        # projected center maps back to the requested point.
        lat, lon = grid.ij_to_latlon(grid.e_we / 2.0, grid.e_sn / 2.0)
        assert float(lat) == pytest.approx(39.7, abs=1e-9)
        assert float(lon) == pytest.approx(-96.6, abs=1e-9)


# ---------------------------------------------------------------------------
# GFS/HRRR prepared routes and their saved geography inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source", ["gfs", "hrrr"])
def test_native_source_keeps_explicit_geography_root(
        tmp_path, monkeypatch, capsys, source):
    """D2: a root given to domain survives into check and bare go."""
    import shlex
    from woof import go_cli
    from woof.geog_assets import config_geog_root, geog_datasets
    from woof.static import terrain_autosmooth

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for key in ("GPUWM_CASE_DATA_ROOT", "XDG_DATA_HOME", "CDSAPI_KEY", "CDSAPI_URL"):
        monkeypatch.delenv(key, raising=False)
    geog = tmp_path / "elsewhere" / "WPS_GEOG"
    from _staged_geog import staged_geog_tree
    staged_geog_tree(geog)
    # These fixtures cover index discovery, not terrain sampling.
    monkeypatch.setattr(terrain_autosmooth, "survey_domain_terrain",
                        lambda *args, **kwargs: [])
    checked_roots = []
    geography_refusal = go_cli.geography_refusal

    def check_geography(root, **kwargs):
        checked_roots.append(Path(root))
        return geography_refusal(root, **kwargs)

    monkeypatch.setattr(go_cli, "geography_refusal", check_geography)
    rc, out = _run_wizard(
        tmp_path, "--geog-root", str(geog), "--hours", "1",
        "--point-extent-km", "1500", point="37.5,-90", ladder="12",
        source=source, cycle="2026-10-17T00")
    assert rc == 0
    printed = capsys.readouterr()
    raw = tomllib.loads(out.read_text())
    # The wizard writes the root with forward slashes (TOML-safe on every
    # platform); the hosted Windows CI compared that text with str(Path)
    # and refused 2.9.2's first publish.  Compare paths, not spellings.
    assert Path(raw["static"]["geog_root"]) == geog.resolve()
    assert "case_data" not in raw
    assert config_geog_root(out) == geog.resolve()
    assert "geography BLOCKER" not in printed.err
    assert "geography staging is the next step" not in printed.out
    command = printed.out.split("next:")[-1].strip().splitlines()[0]
    assert "--geog-root" not in command
    # The printed command spells the configuration with forward slashes
    # (shlex-safe on Windows, where a backslash path would lose its
    # separators); compare the path, not the spelling.
    words = shlex.split(command)
    assert words[:2] == ["woof", "go"] and len(words) == 3
    assert Path(words[2]) == out

    checked_roots.clear()
    assert cli_main(["check", str(out), "--budget-gib", "20"]) == 0
    assert checked_roots == [geog.resolve()]
    assert "geography BLOCKER" not in capsys.readouterr().err
    checked_roots.clear()
    assert cli_main([*shlex.split(command)[1:], "--dry-run"]) == 0
    assert checked_roots == [geog.resolve()]
    preview = capsys.readouterr()
    assert "missing geography indexes" not in preview.err
    assert not (home / ".local/share/woof/WPS_GEOG").exists()


@pytest.mark.parametrize("source", ["gfs", "hrrr"])
def test_native_sources_omit_case_data(tmp_path, capsys, source):
    cycle = "2026-07-28T06" if source == "gfs" else "2026-07-28T05"
    rc, out = _run_wizard(tmp_path, source=source, cycle=cycle)
    assert rc == 0
    printed = capsys.readouterr().out
    assert f"woof go {_posix(out)}" in printed.split("next:")[-1]
    assert "--data-dir" not in printed.split("next:")[-1]
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert "case_data" not in raw
    assert raw["fetch"]["source"] == source
    assert raw["fetch"]["cycle"] == cycle
    # Validate the public syntax and its real shared planning seam before
    # decoder/readiness checks. This configuration test needs no built bridge.
    import hashlib
    import json
    from woof import go_cli, runplan
    from woof.cli import build_parser

    exp = load_experiment(out)
    assert len(exp.domains) == 2
    before = out.read_bytes()
    run_dir, data_dir = tmp_path / "run", tmp_path / "data-preview"
    args = build_parser().parse_args([
        "go", str(out), "--dry-run", "--outdir", str(run_dir),
        "--data-dir", str(data_dir)])
    assert args.func is go_cli.go_main
    assert args.dry_run
    if runplan.prepared_chain_for_source(source) == "prepared:go":
        plan = go_cli.plan_from_config(
            args.config, outdir=args.outdir, data_dir=args.data_dir)
        assert plan["config"] == out
        assert plan["source"] == source
        assert plan["cycle"] == cycle
        assert plan["hours"] == raw["fetch"]["hours"]
        assert plan["area"] == raw["fetch"]["area"]
        assert plan["domains"] == 2
        assert plan["runner"] == go_cli.TREE_RUNNER_MODULE
    else:
        declaration = {
            "schema": runplan.PLAN_SCHEMA, "name": out.stem,
            "route": "prepared", "config": {"path": str(args.config.resolve())},
            "output_root": str(args.outdir.resolve()),
            "run_options": {"data_dir": str(args.data_dir.resolve())}}
        plan = runplan.build_plan(
            declaration, source=f"woof go {out}", base_dir=out.parent,
            sha256=hashlib.sha256(
                json.dumps(declaration, sort_keys=True).encode()).hexdigest())
        view, resolved, data = runplan.resolve_plan(plan, require_inputs=False)
        assert plan.config_bytes() == before
        assert view["plan"]["config_sha256"] == hashlib.sha256(before).hexdigest()
        assert data is None
        assert [d.run for d in resolved.domains] == [d.run for d in exp.domains]
    capsys.readouterr()
    assert out.read_bytes() == before
    assert not run_dir.exists()
    assert not data_dir.exists()


@pytest.mark.parametrize("source", ["gfs", "hrrr"])
def test_explicit_download_directory_survives_printed_launch(tmp_path, capsys, source):
    import shlex
    from woof.cli import build_parser

    data = tmp_path / "Chosen forcing folder"
    rc, out = _run_wizard(tmp_path, "--data-dir", str(data), source=source,
                          cycle="2026-07-28T06")
    assert rc == 0
    printed = capsys.readouterr().out.split("next:")[-1]
    command = next(line.strip() for line in printed.splitlines()
                   if line.strip().startswith("woof go "))
    args = build_parser().parse_args(shlex.split(command)[1:])
    assert args.config.resolve() == out.resolve()
    assert args.data_dir.resolve() == data.resolve()


@pytest.mark.parametrize("source", ["gfs", "hrrr"])
def test_explained_manual_fetch_and_go_use_the_same_download_directory(tmp_path, capsys, source):
    import shlex
    from woof.cli import build_parser

    rc, out = _run_wizard(tmp_path, "--explain", source=source, cycle="2026-07-28T06")
    assert rc == 0
    printed = capsys.readouterr().out.split("next:")[-1]
    commands = {}
    for line in printed.splitlines():
        if "woof fetch " in line or "woof go " in line:
            argv = shlex.split(line[line.index("woof "):])[1:]
            commands[argv[0]] = build_parser().parse_args(argv)
    assert commands["go"].config.resolve() == out.resolve()
    assert commands["go"].data_dir.resolve() == commands["fetch"].out.resolve()


def test_check_without_case_data_runs_memory_preflight(tmp_path, capsys):
    """A declared memory budget can be checked before forcing is fetched.

    The preparation route still owns actual input validation; this result
    neither verifies missing forcing nor certifies a working GPU.
    """
    rc, out = _run_wizard(tmp_path, source="gfs", cycle="2026-07-28T06")
    assert rc == 0
    capsys.readouterr()
    rc = cli_main(["check", str(out), "--budget-gib", "20"])
    printed = capsys.readouterr().out
    assert rc == 0
    assert "not applicable -- no [case_data]" in printed
    assert "preparation route validates its own inputs" in printed
    assert "memory preflight" in printed


def test_existing_divergent_vtable_is_never_overwritten(tmp_path, capsys):
    """Warn-not-block: the user's Vtable is kept (never overwritten),
    the wizard says so in one line, and the emission succeeds."""

    marker = "not the packaged table"
    (tmp_path / "Vtable.ERA5_CDO").write_text(marker)
    rc, out = _run_wizard(tmp_path)
    captured = capsys.readouterr()
    assert rc == 0
    assert out.exists()
    assert "warning:" in captured.err
    assert "kept your existing Vtable.ERA5_CDO" in captured.err
    # The refusal is gone; the protection is not.
    assert (tmp_path / "Vtable.ERA5_CDO").read_text() == marker


# ---------------------------------------------------------------------------
# [fetch] hints schema
# ---------------------------------------------------------------------------

def test_fetch_hints_validation():
    good = {"source": "era5", "cycle": "1999-05-03T12", "hours": 6,
            "area": "25,-112,45,-83", "out": "data/x", "cadence": 6}
    validate_fetch_hints(good, source="unit")
    with pytest.raises(ValueError, match="unknown key"):
        validate_fetch_hints({"source": "era5", "extra": 1}, source="unit")
    with pytest.raises(ValueError, match="must carry source"):
        validate_fetch_hints({"hours": 6}, source="unit")
    with pytest.raises(ValueError, match="not one of"):
        validate_fetch_hints({"source": "cfs"}, source="unit")
    with pytest.raises(ValueError, match="hours.*whole number"):
        validate_fetch_hints({"source": "era5", "hours": [1, 2]},
                             source="unit")


def test_loaders_reject_bad_fetch_table(tmp_path):
    rc, out = _run_wizard(tmp_path, source="gfs", cycle="2026-07-28T06")
    assert rc == 0
    text = out.read_text(encoding="utf-8").replace(
        'source = "gfs"', 'source = "gfs"\nbogus_key = 1')
    bad = tmp_path / "bad.toml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="bogus_key"):
        load_experiment(bad)


# ---------------------------------------------------------------------------
# HRRR area contract: the hint and the guard share ONE grid-derived coverage
# ---------------------------------------------------------------------------

def test_the_field_hrrr_root_emits_an_area_the_fetch_validator_accepts():
    """Field repro (2026-08, RTX PRO 6000): center 39,-98, 3 km root,
    1234 x 986 mass points.  The wizard emitted ``--area
    21.98,-126.17,54.39,-69.83`` as its own next command and ``woof
    fetch`` refused it: the wizard's box clamped only at the pole while
    the fetch guard held a hand-held 52.7 latitude cap
    (woof/fetch.py HRRR_CONUS_LAT) -- two definitions of HRRR coverage.
    Both now derive from the native grid, and the proof here is the REAL
    validator over the exact emitted bytes, post-formatting."""

    from woof import fetch
    from woof.domain_wizard import (_projection_entries, _root_grid,
                                     fetch_area_hint)

    projection = _projection_entries(39.0, -98.0)
    notes: list[str] = []
    hint = fetch_area_hint(projection, 1234, 986, source="hrrr",
                           root_dx_m=3000.0, coverage_notes=notes)
    # The real parser and the real coverage gate, on the exact string.
    fetch.validate_fetch_area("hrrr", fetch.parse_area(hint))
    # ...and the loaders' own hint validation, which every emission
    # round-trips through before the file is written.
    validate_fetch_hints(
        {"source": "hrrr", "cycle": "2026-07-28T05", "hours": 3,
         "area": hint, "out": "data/x"}, source="unit")
    # The clamp bit only the 2-degree margin, never the footprint: every
    # root corner still lies inside the emitted box.
    south, west, north, east = (float(v) for v in hint.split(","))
    lat_c, lon_c = _root_grid(projection, 1234, 986, 3000.0).latlon_c()
    assert south <= float(np.min(lat_c)) and float(np.max(lat_c)) <= north
    assert west <= float(np.min(lon_c)) and float(np.max(lon_c)) <= east
    # The clamp is disclosed, not silent, and names the edge it moved.
    assert notes and any("north" in note for note in notes)


def test_an_hrrr_area_hint_inside_coverage_is_untouched():
    """A request the source covers with margin to spare emits exactly the
    box it always did -- the coverage bound changes nothing inside it."""

    from woof import fetch
    from woof.domain_wizard import (_fetch_area, _fetch_margin_deg,
                                     _projection_entries, fetch_area_hint)

    projection = _projection_entries(35.3, -97.5)
    notes: list[str] = []
    hint = fetch_area_hint(projection, 60, 48, source="hrrr",
                           root_dx_m=12000.0, coverage_notes=notes)
    bare = _fetch_area(projection, 60, 48,
                       margin_deg=_fetch_margin_deg("hrrr"),
                       root_dx_m=12000.0)
    assert hint == ",".join(f"{value:.2f}" for value in bare)
    assert notes == []
    fetch.validate_fetch_area("hrrr", fetch.parse_area(hint))


def test_a_fetch_hint_area_outside_hrrr_coverage_is_refused_at_load():
    """The rotten-hint rule the [fetch] table already applies to windows,
    extended to the area: a hint naming latitudes the native grid does
    not carry is refused at config load in the fetch's own words.  The
    box below is the 1.4.1 field workaround -- clamped to the guard's
    hand-held 52.70 cap, which sits NORTH of every mass point the real
    grid carries (it tops out at 52.6157)."""

    with pytest.raises(ValueError, match="coverage"):
        validate_fetch_hints(
            {"source": "hrrr", "cycle": "2026-07-28T05", "hours": 3,
             "area": "21.10,-126.17,52.70,-69.83", "out": "d"},
            source="unit")
    # An unparseable area hint is equally rotten.
    with pytest.raises(ValueError, match="lat0,lon0,lat1,lon1"):
        validate_fetch_hints(
            {"source": "era5", "area": "21.98,-126.17"}, source="unit")


def test_an_hrrr_emission_near_the_grids_north_edge_passes_its_own_fetch(
        tmp_path, capsys):
    """The whole product seam: a card-filling 12 km HRRR root against the
    top of the grid emits a next command whose --area its own fetch
    accepts, with the clamp disclosed as an advisory (never a refusal:
    the DOMAIN was already bounded by source coverage during fitting)."""

    from woof import fetch

    # 24 GiB, not 12: the 1.8 full-radiation HRRR default (Thompson mp8
    # + legacy RRTMG) costs about 1.8 GiB more peak envelope than the
    # wsm6/no-longwave suite it replaced, and 12 GiB no longer fits the
    # minimum 12 km layout.  The seam under test is the emitted --area,
    # so it runs on a card the shipped default fits.
    rc, out = _run_wizard(tmp_path, "--explain", point="48.5,-98.0", card="24gb",
                          ladder="12", source="hrrr",
                          cycle="2026-07-28T05")
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    hints = tomllib.loads(out.read_text(encoding="utf-8"))["fetch"]
    area_hint = str(hints["area"])
    # The stored hint and the printed command carry the same string, and
    # the real fetch validators accept it.
    assert area_hint in captured.out
    fetch.validate_fetch_area("hrrr", fetch.parse_area(area_hint))
    # A box this size was clamped on at least one edge, and said so.
    assert "clamped" in captured.err


# ---------------------------------------------------------------------------
# Full composed-check integration on staged real inputs (gated)
# ---------------------------------------------------------------------------

@requires_staged_real_inputs
@pytest.mark.slow
def test_wizard_full_check_passes_on_staged_inputs(tmp_path, capsys):
    """point -> emitted config -> composed `woof check` rc 0, in-process.

    A genuinely new area (central Oklahoma) against the staged May-1999
    ERA5 GRIB1 pair and the standard WPS_GEOG tree: the wizard's final
    step runs the real composed check (input preflight decode + geog tile
    coverage + memory estimate vs budget) and must report PASS.
    """
    out = tmp_path / "okc.toml"
    rc = cli_main([
        "domain", "--point", "35.3,-97.5", "--card", "24gb",
        "--cycle", "1999-05-03T12", "--hours", "6",
        "--out", str(out),
        "--forcing", str(MAY99 / "era5_may1999_pl.grib"),
        str(MAY99 / "era5_may1999_sl.grib"),
        "--geog-root", str(BUNDLE / "static/WPS_GEOG")])
    printed = capsys.readouterr().out
    assert rc == 0
    assert "woof input preflight: PASS" in printed
    assert "woof check: PASS (rc 0)" in printed
    assert "WARNING" not in printed  # envelope fit keeps check warning-free


def test_gfs_fetch_hint_margin_is_the_front_door_coverage_margin():
    """The wizard's suggested GFS --area must pass the front door's own
    donor-coverage proof: its margin comes from the one shared function
    (woof.fetch.gfs_suggested_fetch_margin_deg) instead of a private
    too-small constant the coverage check then rejects."""
    from woof.domain_wizard import _FETCH_MARGIN_DEG, _fetch_margin_deg
    from woof.fetch import (GFS_LAKE_DONOR_MARGIN_DEG,
                             gfs_suggested_fetch_margin_deg)

    assert _fetch_margin_deg("gfs") == gfs_suggested_fetch_margin_deg()
    assert gfs_suggested_fetch_margin_deg() == GFS_LAKE_DONOR_MARGIN_DEG
    # The acceptance lane measured +8..15 deg beyond the old 2-deg hint
    # as the empirical requirement for an interior-CONUS domain.
    assert gfs_suggested_fetch_margin_deg() >= 8.0
    # ERA5 needs only interpolation halo; HRRR must stay inside its own
    # CONUS coverage box, so both keep the small margin.
    assert _fetch_margin_deg("era5") == _FETCH_MARGIN_DEG
    assert _fetch_margin_deg("hrrr") == _FETCH_MARGIN_DEG


def test_fetch_area_applies_and_clamps_the_margin():
    from woof.domain_wizard import _fetch_area, _projection_entries

    projection = _projection_entries(35.0, -97.5)
    small = _fetch_area(projection, 60, 48, margin_deg=2.0)
    wide = _fetch_area(projection, 60, 48, margin_deg=15.0)
    # 13 more degrees on every side (S grows down, N up, W down, E up).
    assert wide[0] == pytest.approx(small[0] - 13.0)
    assert wide[1] == pytest.approx(small[1] - 13.0)
    assert wide[2] == pytest.approx(small[2] + 13.0)
    assert wide[3] == pytest.approx(small[3] + 13.0)
    # Near the dateline the margin wraps across the seam instead of
    # truncating: the donor margin is honoured on both sides, and the
    # resulting box is the W > E crossing form the fetch layer serves.
    west = _projection_entries(52.0, -170.0)
    wrapped = _fetch_area(west, 60, 48, margin_deg=15.0)
    assert wrapped[1] > 0.0 > wrapped[3]
    from woof.fetch import Area
    assert Area(wrapped[0], wrapped[1], wrapped[2],
                wrapped[3]).crosses_antimeridian


# ---------------------------------------------------------------------------
# Custom ladders: --root-dx / --chain alongside the presets.
# ---------------------------------------------------------------------------

def test_chain_and_root_dx_parsing_and_refusals(capsys):
    from woof.domain_wizard import (MAX_CHAIN_DEPTH, MAX_CHAIN_RATIO,
                                     MAX_ROOT_DX_KM, MIN_CHAIN_RATIO,
                                     MIN_ROOT_DX_KM, ROOT_DX_M,
                                     parse_chain, parse_custom_ladder)

    assert parse_chain("4,3,3") == (4, 3, 3)
    assert parse_chain(" 4 , 3 ") == (4, 3)
    assert parse_chain("") == ()
    # KEEP-HARD: an unparseable ratio and a non-refinement stay refusals.
    with pytest.raises(ValueError, match="not an integer"):
        parse_chain("4,3.5")
    with pytest.raises(ValueError, match="not a refinement"):
        parse_chain(str(MIN_CHAIN_RATIO - 1))
    # Warn-not-block: the conservative upper bounds report and continue.
    capsys.readouterr()
    assert parse_chain(str(MAX_CHAIN_RATIO + 1)) == (MAX_CHAIN_RATIO + 1,)
    err = capsys.readouterr().err
    assert "warning:" in err and "exceeds the blessed maximum" in err
    deep = tuple([2] * (MAX_CHAIN_DEPTH + 1))
    assert parse_chain(",".join(map(str, deep))) == deep
    err = capsys.readouterr().err
    assert "warning:" in err and "nests" in err

    # A preset run stays a preset run.
    assert parse_custom_ladder(
        root_dx_km=None, chain=None, ladder="auto") is None
    assert parse_custom_ladder(
        root_dx_km=None, chain=None, ladder="12-3") is None
    # Either flag alone switches to the custom form.
    assert parse_custom_ladder(
        root_dx_km=3.0, chain=None, ladder="auto") == (3000.0, ())
    assert parse_custom_ladder(
        root_dx_km=None, chain="4", ladder="auto") == (ROOT_DX_M, (4,))
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_custom_ladder(root_dx_km=3.0, chain="4", ladder="12-3")
    # Warn-not-block: the km-typo window warns and continues; a
    # non-positive spacing stays a refusal.
    capsys.readouterr()
    for odd in (MIN_ROOT_DX_KM / 2, MAX_ROOT_DX_KM * 2):
        root_m, _ = parse_custom_ladder(
            root_dx_km=odd, chain="4", ladder="auto")
        assert root_m == odd * 1000.0
        err = capsys.readouterr().err
        assert "warning:" in err and "--root-dx" in err
    with pytest.raises(ValueError, match="positive spacing"):
        parse_custom_ladder(root_dx_km=-1.0, chain="4", ladder="auto")


def test_custom_ladder_3km_to_750m_emits_and_checks(tmp_path, capsys):
    """The r4 case: an arbitrary root dx with an integer ratio.

    Validated by the same estimator fit loop, the same experiment
    loader, and the same `woof check` the presets go through.
    """
    out = tmp_path / "r4.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "24gb",
                   "--root-dx", "3", "--chain", "4", "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert "woof check: PASS (rc 0)" in printed

    text = out.read_text(encoding="utf-8")
    exp = experiment_from_text(text, source=str(out))
    assert [float(exp.dx_exact(d.grid_id)) for d in exp.domains] == [
        3000.0, 750.0]
    # 5 s/km at 35 N: 15 s root, exactly quartered on the nest.
    assert exp.root.time_step == 15
    assert float(exp.dt_exact(2)) == 15 / 4
    # The companion namelist.wps agrees through the real grid builders.
    wps = out.parent / "r4.namelist.wps"
    assert " dx = 3000," in wps.read_text(encoding="utf-8")
    from_wps = grids_from_wps_namelist(wps)
    from_toml = grids_from_projection_config(load_experiment(out))
    assert len(from_wps) == len(from_toml) == 2
    for a, b in zip(from_wps, from_toml):
        lat_a, lon_a = a.latlon_mass()
        lat_b, lon_b = b.latlon_mass()
        assert a.dx == b.dx
        np.testing.assert_allclose(lat_a, lat_b, rtol=0, atol=1e-9)
        np.testing.assert_allclose(lon_a, lon_b, rtol=0, atol=1e-9)


@pytest.mark.parametrize("ladder_flags", [
    ("--root-dx", "3", "--chain", "4"),
    ("--ladder", "12-3-1"),
])
def test_every_domain_inherits_the_profiles_epssm(tmp_path, capsys,
                                                  ladder_flags):
    """The nest gets the profile's epssm, not WRF's Registry default.

    The regression this pins killed a reported nested forecast: the
    wizard wrote ``epssm = 0.1`` on every nest while the root took 0.5
    from the physics profile, stripping the vertical-acoustic
    off-centering exactly where nest terrain is steepest.  A 3 km ->
    750 m ladder over the Cascades grew w to non-finite in seven
    acoustic substeps at the child's steepest cell; the same geometry
    with the nest at the profile's 0.5 ran clean.

    Asserted against ``profile_switches`` rather than the number 0.5, so
    a profile that ships a different epssm still propagates to depth.
    """
    from woof.domain_wizard import profile_switches

    out = tmp_path / "epssm.toml"
    rc = cli_main(["domain", "--point=46.9,-121.8", "--card", "24gb",
                   *ladder_flags, "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed

    exp = load_experiment(out)
    expected = float(profile_switches(None)["epssm"])
    assert len(exp.domains) > 1
    assert [float(d.run.epssm) for d in exp.domains] == (
        [expected] * len(exp.domains))
    # Stated per domain in the file the reader opens, so the value that
    # matters is visible where it is set -- and overridable there.
    text = out.read_text(encoding="utf-8")
    assert text.count("epssm = ") == len(exp.domains)
    assert "epssm = 0.1" not in text


def test_a_named_profiles_epssm_reaches_the_nest(tmp_path, capsys):
    """Same contract when --physics-profile names the suite."""
    from woof.domain_wizard import profile_switches

    out = tmp_path / "epssm-profile.toml"
    assert cli_main([
        "domain", "--point=46.9,-121.8", "--card", "24gb",
        "--ladder", "12-3", "--source", "gfs", "--cycle",
        "2026-07-29T18", "--hours", "6", "--physics-profile",
        MORRISON_PROFILE_ID, "--out", str(out)]) == 0
    capsys.readouterr()
    exp = load_experiment(out)
    expected = float(profile_switches(MORRISON_PROFILE_ID)["epssm"])
    assert [float(d.run.epssm) for d in exp.domains] == [expected] * 2


def test_custom_ladder_deep_chain_reaches_the_hundred_metre_scale(
        tmp_path, capsys):
    out = tmp_path / "deep.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "32gb",
                   "--root-dx", "3", "--chain", "3,3,3", "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--hours", "3",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert "woof check: PASS (rc 0)" in printed

    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    dxs = [float(exp.dx_exact(d.grid_id)) for d in exp.domains]
    assert len(dxs) == 4
    assert dxs[0] == 3000.0
    assert dxs[-1] == pytest.approx(3000.0 / 27, rel=1e-12)
    assert 110.0 < dxs[-1] < 112.0
    # Exact rational clock all the way down: 15 s / 27.
    assert exp.root.time_step == 15
    assert exp.dt_exact(4) == Fraction(15, 27)


def test_the_gray_zone_advisory_warns_and_never_refuses(tmp_path, capsys):
    from woof.domain_wizard import (GRAY_ZONE_DX_KM, _SHARED_CERTIFIED,
                                     gray_zone_advisory)

    # Above the gray zone: silent.
    assert gray_zone_advisory([12.0, 3.0, 1.0], _SHARED_CERTIFIED) == []
    # PBL scheme off: the overlap it warns about does not exist.
    assert gray_zone_advisory(
        [3.0, 0.75], {**_SHARED_CERTIFIED, "bl_pbl_physics": 0}) == []

    lines = gray_zone_advisory([3.0, 0.75, 0.25], _SHARED_CERTIFIED)
    assert len(lines) == 1, "one accurate sentence, not a lecture"
    assert "GRAY ZONE" in lines[0]
    assert "2 domain(s)" in lines[0]
    assert "finest 250 m" in lines[0]
    # The advisory must name the REMEDY, not a roadmap item.  It used to
    # say the proper tool "is a 3-D turbulence closure (SASE, planned)";
    # every closure it names is implemented now, so pointing a user at
    # vapourware would be worse than saying nothing.  SASE ships on this
    # line (lane/sase-sota, bl_pbl_physics = 900), so the old
    # ``"SASE" not in lines[0]`` vapourware guard is superseded -- what
    # survives of it is that a SASE mention MUST carry its maturity
    # label, so nobody is pointed at an unverified closure unlabeled.
    assert "SASE" in lines[0] and "EXPERIMENTAL" in lines[0]
    assert "not WRF-verified" in lines[0]
    assert "km_opt = 3" in lines[0] and "km_opt = 2" in lines[0]
    assert "bl_pbl_physics = 0" in lines[0]
    assert f"below {GRAY_ZONE_DX_KM:g} km" in lines[0]

    # End to end it is an advisory: rc 0, in the file and on stdout.
    out = tmp_path / "gray.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "24gb",
                   "--root-dx", "3", "--chain", "4", "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0
    assert "advisory: GRAY ZONE" in printed
    assert "finest 750 m" in printed
    assert "GRAY ZONE" in out.read_text(encoding="utf-8")


def test_the_deepest_preset_also_declares_its_gray_zone(tmp_path, capsys):
    """12-3-1-0.5 lands at 500 m; the advisory is not custom-only."""
    out = tmp_path / "preset.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "32gb",
                   "--ladder", "12-3-1-0.5", "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--hours", "3",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert "advisory: GRAY ZONE" in printed
    assert "finest 500 m" in printed


def test_the_cumulus_advisory_warns_at_convection_permitting_dx(
        tmp_path, capsys):
    from woof.domain_wizard import (CUMULUS_CONVECTION_PERMITTING_DX_KM,
                                     cumulus_gray_zone_advisory)

    # 12 km root with active cumulus: the scheme is doing its real job.
    assert cumulus_gray_zone_advisory([12.0, 3.0], [1, 0]) == []
    # Cumulus off everywhere: nothing to say at any spacing.
    assert cumulus_gray_zone_advisory([3.0, 0.75], [0, 0]) == []

    lines = cumulus_gray_zone_advisory([3.0, 0.75], [1, 0])
    assert len(lines) == 1, "one accurate sentence, not a lecture"
    assert "CUMULUS" in lines[0]
    assert "1 domain(s)" in lines[0]
    assert "finest 3 km" in lines[0]
    assert f"below {CUMULUS_CONVECTION_PERMITTING_DX_KM:g} km" in lines[0]
    assert "cu_physics = 1" in lines[0]
    # The remedy is pasteable: name the switch and its off value.
    assert "cu_physics = 0" in lines[0]

    # End to end: a suite the user NAMED keeps its cumulus scheme at
    # --root-dx 3, and the pairing is an advisory, never a refusal --
    # rc 0, the finding on stdout, the full sentence in the header.
    out = tmp_path / "cp.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "24gb",
                   "--root-dx", "3", "--chain", "4", "--source", "gfs",
                   "--physics-profile", MORRISON_PROFILE_ID,
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0
    assert "advisory: CUMULUS" in printed
    assert "CUMULUS" in out.read_text(encoding="utf-8")


def test_the_wizard_defaults_cumulus_off_below_the_permitting_bound(
        tmp_path, capsys):
    """A bare 3 km run emits cu_physics = 0, and says nothing about
    double counting, because there is nothing left to double count.

    The wizard used to hand every unnamed suite the profile's
    Kain-Fritsch on the root at ANY spacing, print the sentence naming
    the heating and rainfall it counts twice, and write the file
    anyway.  The bound is the one this module already declares --
    CUMULUS_CONVECTION_PERMITTING_DX_KM -- not a second number.
    """
    from woof.domain_wizard import (CUMULUS_CONVECTION_PERMITTING_DX_KM,
                                     cumulus_by_domain)

    out = tmp_path / "bare3km.toml"
    # A 1.5 km nest, not 750 m: below 1 km the default is the spacing
    # table's cumulus-free suite (woof.physics_menu.SPACING_DEFAULTS), so
    # nothing would be retired; this probe is the derived Kain-Fritsch
    # suite meeting a 3 km root.
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "24gb",
                   "--root-dx", "3", "--chain", "2", "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    text = out.read_text(encoding="utf-8")
    exp = experiment_from_text(text, source=str(out))
    assert [dc.run.cu_physics for dc in exp.domains] == [0, 0]
    # cudt paces Kain-Fritsch and nothing else; with the scheme off the
    # emitted cadence is the registry's own spelling for "no scheme".
    assert exp.root.run.cudt_minutes == 0.0
    # Nothing double-counts, so the double-counting sentence is gone
    # from both surfaces.
    assert "counted twice" not in printed
    assert "counted twice" not in text
    assert "CUMULUS GRAY ZONE" not in printed
    # The moved switch is REPORTED, not silently applied: the emission
    # changed a switch the derived suite carries, and it says which one
    # and how to get it back.  The summary line describes the FILE.
    assert "NO cumulus parameterization" in printed
    assert "NO cumulus parameterization" in text
    assert "advisory: CUMULUS OFF AT 3 KM" in printed
    assert "CUMULUS OFF AT 3 KM" in text
    assert "--physics-profile" in text
    # And the header stops claiming a verbatim identity the file lacks.
    assert "Taken verbatim from woof.physics_compat EXCEPT" in text

    # The threshold is the declared bound: AT it the profile's scheme
    # survives (the gray zone is a softer finding, not this defect).
    assert CUMULUS_CONVECTION_PERMITTING_DX_KM == 4.0
    dims, ratios = [(120, 120)], ()
    assert cumulus_by_domain(
        dims, ratios, profile=MORRISON_PROFILE_ID,
        root_dx_m=CUMULUS_CONVECTION_PERMITTING_DX_KM * 1000.0) == [1]
    assert cumulus_by_domain(
        dims, ratios, profile=MORRISON_PROFILE_ID,
        root_dx_m=(CUMULUS_CONVECTION_PERMITTING_DX_KM - 0.5) * 1000.0
    ) == [0]


def test_an_explicitly_named_cumulus_suite_survives_at_three_km(
        tmp_path, capsys):
    """Naming --physics-profile asserts the config IS that suite, so the
    wizard emits it verbatim and keeps the double-counting sentence."""
    out = tmp_path / "named3km.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "24gb",
                   "--root-dx", "3", "--chain", "4", "--source", "gfs",
                   "--physics-profile", MORRISON_PROFILE_ID,
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    text = out.read_text(encoding="utf-8")
    exp = experiment_from_text(text, source=str(out))
    switches = single_domain_runtime_switches(MORRISON_PROFILE_ID)
    assert exp.root.run.cu_physics == switches["cu_physics"] == 1
    assert "advisory: CUMULUS" in printed
    assert "counted twice" in text


def test_the_convective_gray_zone_gets_a_softer_note():
    from woof.domain_wizard import (CUMULUS_CONVECTION_PERMITTING_DX_KM,
                                     CUMULUS_GRAY_ZONE_TOP_DX_KM,
                                     cumulus_gray_zone_advisory,
                                     cumulus_gray_zone_headline)

    # 4-10 km with active cumulus: the genuine gray zone, softer words.
    for dx in (CUMULUS_CONVECTION_PERMITTING_DX_KM, 9.0,
               CUMULUS_GRAY_ZONE_TOP_DX_KM):
        lines = cumulus_gray_zone_advisory([dx], [1])
        assert len(lines) == 1
        assert "CUMULUS GRAY ZONE" in lines[0]
        assert "common operational practice" in lines[0]
    # Above the band: silent (the shipped 12 km presets stay unchanged).
    assert cumulus_gray_zone_advisory([12.0], [1]) == []
    # Off in the band: silent.
    assert cumulus_gray_zone_advisory([9.0], [0]) == []

    # Both findings at once, strong first; headlines derive from the
    # same call, first clause only.
    both = cumulus_gray_zone_advisory([8.0, 2.0], [1, 1])
    assert len(both) == 2
    assert "counted twice" in both[0] or "twice" in both[0]
    assert "CUMULUS GRAY ZONE" in both[1]
    heads = cumulus_gray_zone_headline([8.0, 2.0], [1, 1])
    assert len(heads) == 2
    assert all(head in (line.split(", so ", 1)[0] + ".")
               for head, line in zip(heads, both))


def test_a_twelve_km_root_with_cumulus_stays_silent(tmp_path, capsys):
    """The shipped preset ladders root at 12 km: no cumulus advisory,
    and the wizard's output is otherwise unchanged."""
    out = tmp_path / "preset12.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "24gb",
                   "--ladder", "12-3", "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert "advisory: CUMULUS" not in printed
    assert "CUMULUS" not in out.read_text(encoding="utf-8")


def test_custom_root_dx_in_the_tropics_keeps_an_exact_half_second(
        tmp_path, capsys):
    out = tmp_path / "trop.toml"
    rc = cli_main(["domain", "--point=14.6,120.98", "--card", "24gb",
                   "--root-dx", "3", "--chain", "4", "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    text = out.read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    # 2.5 s/km at 3 km = 7.5 s -> WRF's exact rational clock keys.
    assert raw["domain"][0]["time_step"] == 7
    assert raw["domain"][0]["time_step_fract_num"] == 1
    assert raw["domain"][0]["time_step_fract_den"] == 2
    exp = experiment_from_text(text, source=str(out))
    assert exp.dt_exact(1) == Fraction(15, 2)
    assert exp.dt_exact(2) == Fraction(15, 8)
    assert "TROPICAL CLOCK" in text


# ---------------------------------------------------------------------------
# The documented GFS -> GPU route: physics representation and accuracy.
# ---------------------------------------------------------------------------

def test_emitted_radiation_uses_the_representation_the_guard_compares(
        tmp_path, capsys):
    """v1.0.0's wizard config could never pass the runner's guard.

    Every shipped profile writes radiation as the split pair
    (`ra_physics = 0` + `ra_lw_physics`/`ra_sw_physics`); the wizard
    wrote the legacy combined `ra_physics = 4`.  Both resolve to (4, 4),
    the guard even printed that both sides resolved to (4, 4), and it
    rejected them anyway because it compares the raw switch dicts.
    """
    from woof.config import radiation_scheme_ids

    rc, out = _run_wizard(tmp_path, "--explain", ladder="12", source="gfs",
                          cycle="2026-07-29T18")
    capsys.readouterr()
    assert rc == 0
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert raw["shared"]["ra_physics"] == 0
    assert raw["shared"]["ra_lw_physics"] == 4
    assert raw["shared"]["ra_sw_physics"] == 4
    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    assert radiation_scheme_ids(exp.root.run) == (4, 4)


@pytest.mark.parametrize("profile", [
    "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1",
    "thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1",
    "wsm6-ysu-mm5-noah-no-radiation-v1",
])
def test_physics_profile_configs_pass_the_runner_guard_as_emitted(
        tmp_path, capsys, pinned_thompson_tables, profile):
    """--physics-profile emits a config the prepared runner accepts.

    Not "nearly accepts": this is the runner's own validator, run over
    the exact bytes the wizard wrote, with no hand edits -- the loop
    that cost node 2 three full 200 s front-door cycles to escape.
    """
    import tools.prepared_single_domain_forecast as runner

    out = tmp_path / f"{profile[:12]}.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "24gb",
                   "--ladder", "12", "--source", "gfs",
                   "--physics-profile", profile, "--explain",
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert "every runner enforces it switch for switch" in printed

    exp = load_experiment(out)
    validation = runner._validate_profile_switches(
        exp, source="gfs", profile=profile)
    assert validation["profile"] == profile
    # And the whole physics gate, not just the switch comparison.
    runner._validate_physics(exp, profile, exp.run_seconds,
                             float(exp.root.history_interval_s),
                             source="gfs")
    # The file states, in words, what it will actually run.
    text = out.read_text(encoding="utf-8")
    assert "# PHYSICS:" in text
    assert profile in text


def test_the_default_suite_states_its_physics_and_runs(
        tmp_path, capsys):
    """Status is stated, never a gate (owner ruling 2026-07-31); and the
    default is BOUND (owner directive 2026-08-06).

    The gfs/era5 default is now the certified Morrison profile, so the
    default screen's physics line names that profile with the resolved
    radiation in words, plus the bound-profile enforcement note.  A
    pilot once read the profiles' `ra_physics: 0` as "radiation off",
    so the words, never the raw switch, still carry the resolved
    behaviour.
    """
    from woof.domain_wizard import (DEFAULT_PHYSICS_PROFILE,
                                     physics_summary,
                                     prepared_route_physics_notice)

    rc, _ = _run_wizard(tmp_path, "--explain", ladder="12", source="gfs",
                        cycle="2026-07-29T18")
    printed = capsys.readouterr().out
    assert rc == 0
    # The physics line names the bound certified profile, in words...
    assert DEFAULT_PHYSICS_PROFILE in printed
    assert "longwave RTE+RRTMGP, shortwave RTE+RRTMGP" in printed
    # ...with the enforcement note for a bound profile...
    assert "bound to a shipped profile" in printed
    # ...and no refusal talk anywhere.
    assert "will refuse" not in printed
    assert "refuses it as emitted" not in printed

    # Words, never the raw switch: full-physics profiles must not read
    # as "off", and reduced ones must not read as full.
    full = physics_summary("morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1")
    assert "longwave RTE+RRTMGP, shortwave RTE+RRTMGP" in full
    assert "Kain-Fritsch cumulus" in full
    reduced = physics_summary("thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1")
    assert "longwave OFF, shortwave Dudhia" in reduced
    assert "NO cumulus parameterization" in reduced

    # The unnamed-suite catalog branch survives for direct callers and
    # still names what each candidate ACTUALLY runs.
    unnamed = "\n".join(prepared_route_physics_notice(None, "gfs"))
    assert "longwave OFF, shortwave Dudhia" in unnamed

    # ERA5 does not go through that door, so it gets no such notice.
    assert prepared_route_physics_notice(None, "era5") == []


# ---------------------------------------------------------------------------
# Output layering: the wizard's default is one screen ending in the
# next-steps block, and --explain restores every word of the long form.
#
# The field exhibit this pins: a first-run ERA5 wizard printed 20 lines
# whose correct `woof fetch` command sat at line 15, under a gray-zone
# advisory and above a nine-name dataset inventory, and the user's
# public verdict was "still can't get it working".  The commands were
# right; they were not findable.
# ---------------------------------------------------------------------------

#: What a first run may print before the reader has to scroll.  Not a
#: style preference: the exhibit's wall was 20 lines and the block that
#: matters is the last four, so the cap is what keeps the whole thing on
#: one screen alongside a shell prompt.
WIZARD_DEFAULT_LINE_CAP = 14


@pytest.mark.parametrize("explain", [False, True])
def test_wizard_output_is_layered_and_the_default_fits_a_screen(
        tmp_path, capsys, explain):
    """Terse by default, complete under --explain -- both, every time."""

    extra = ("--explain",) if explain else ()
    rc, out = _run_wizard(tmp_path, *extra, ladder="12-3-1-0.5",
                          card="24gb", source="era5")
    printed = capsys.readouterr().out
    assert rc == 0
    lines = printed.splitlines()

    if explain:
        # Every word that moved is back, in its original wording.
        assert "sizing (itemized preflight estimator, in-process):" in printed
        assert "peak envelope" in printed
        assert "envelope basis:" in printed
        assert "woof check: deferred" in printed
        assert "static geography:" in printed
        assert "treat sub-kilometre PBL structure as indicative" in printed
    else:
        assert len(lines) <= WIZARD_DEFAULT_LINE_CAP, printed
        # The advisory still fires -- it is shortened, not dropped.
        assert "GRAY ZONE" in printed
        assert "treat sub-kilometre PBL structure as indicative" \
            not in printed
        # One sizing line, carrying the numbers that decide whether it runs.
        assert "sizing (itemized" not in printed
        assert "peak envelope" in printed and "headroom" in printed
        # And the way back to everything above.
        assert "--explain" in printed


def test_wizard_ends_with_three_numbered_commands_and_nothing_after(
        tmp_path, capsys):
    """The last thing on screen is the only thing asking for an action.

    Nothing prints after step 3.  The exhibit's failure was a correct
    next command with more output beneath it, which reads as "and then
    this happened" rather than "do this".
    """

    rc, out = _run_wizard(tmp_path, ladder="12-3", source="era5")
    printed = capsys.readouterr().out
    assert rc == 0
    lines = [line for line in printed.splitlines() if line.strip()]

    block = lines[lines.index("next:"):]
    # Exactly three numbered steps, in order, and nothing numbered four.
    numbered = [line for line in block if line.lstrip()[:2] in
                ("1.", "2.", "3.", "4.")]
    assert len(numbered) == 3
    assert numbered[0].startswith("  1. woof fetch ")
    assert numbered[1].startswith("  2. woof check ")
    assert numbered[2].startswith("  3. woof run ")
    # Step 3 is the last line of output.  Anything after it competes
    # with the one thing the reader is being asked to do.
    assert lines[-1] == numbered[2]
    # Step 2 carries the deferral instead of a stanza of its own.
    assert "after the fetch lands" in numbered[1]


def test_the_default_era5_provider_is_keyless_without_a_cds_key(
        tmp_path, capsys, monkeypatch):
    """No CDS key: the default reads the keyless ARCO store and says why.

    Replaces test_the_era5_next_block_names_the_missing_cds_key_only_when_missing
    (2.9.2, 003fe55ab): the next block no longer names a missing key because the
    default needs none.

    Breakage it prevents: the default `woof domain` wrote a CDS-only
    [fetch] table, so the default `woof go` stopped at acquisition on a
    computer with no ``~/.cdsapirc`` while `woof doctor` reported the
    keyless ARCO store available.  With a key the CDS stays the default,
    byte for byte the table it always was.
    """

    from woof import fetch

    for name in ("CDSAPI_KEY", "CDSAPI_URL", "CDSAPI_RC"):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(fetch.Path, "home", staticmethod(lambda: home))

    rc, out = _run_wizard(tmp_path / "no-key", "--explain", source="era5")
    absent = capsys.readouterr().out
    assert rc == 0
    config = tomllib.loads(out.read_text(encoding="utf-8"))
    assert config["fetch"]["era5_provider"] == "arco"
    assert config["case_data"]["forcing"][0].endswith(
        fetch.era5_combined_name("arco"))
    chosen = [line for line in absent.splitlines()
              if line.startswith("source: ERA5 from Google's keyless ARCO store")]
    assert len(chosen) == 1 and str(home / fetch.CDSAPIRC_NAME) in chosen[0]
    step = next(line for line in absent.splitlines() if "1. woof fetch" in line)
    assert "--era5-provider arco" in step
    # No pointer telling a keyless user to go and register for a CDS key.
    assert "Copernicus CDS key needed" not in absent

    (home / fetch.CDSAPIRC_NAME).write_text("url: x\nkey: y\n")
    rc, out = _run_wizard(tmp_path / "with-key", "--explain", source="era5")
    present = capsys.readouterr().out
    assert rc == 0
    config = tomllib.loads(out.read_text(encoding="utf-8"))
    assert "era5_provider" not in config["fetch"]
    assert config["case_data"]["forcing"][0].endswith(fetch.era5_combined_name("cds"))
    assert "ARCO" not in present
    assert "1. woof fetch --source era5" in present
    assert "--era5-provider" not in present

    # A provider named on the command line is kept, key or no key.
    (home / fetch.CDSAPIRC_NAME).unlink()
    rc, out = _run_wizard(tmp_path / "named", "--era5-provider", "cds", source="era5")
    named = capsys.readouterr().out
    assert rc == 0
    assert tomllib.loads(out.read_text(encoding="utf-8"))["fetch"]["era5_provider"] == "cds"
    assert "keyless ARCO store" not in named
    assert "Copernicus CDS key" in named


def test_the_default_source_runs_on_gfs_when_the_keyless_store_lacks_the_window(
        tmp_path, capsys, monkeypatch):
    """No CDS key and a recent date: the default runs on GFS and says why.

    Breakage it prevents: ERA5 is published about a week behind real
    time, so a bare `woof domain` for yesterday's date wrote an ARCO
    [fetch] table the store could not serve and the default `woof go`
    failed outright at the fetch, after the plan had been accepted.  A
    window the store holds stays on ERA5; `--source era5` named for a
    window it lacks is refused here, naming the newest cycle it holds.
    """

    from datetime import datetime

    from woof import fetch

    for name in ("CDSAPI_KEY", "CDSAPI_URL", "CDSAPI_RC"):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(fetch.Path, "home", staticmethod(lambda: home))
    newest = datetime(2026, 10, 11, 12)

    def fake_resolve(source, last_hour, **selection):
        assert source == "era5" and selection.get("provider") == "arco"
        return newest

    monkeypatch.setattr(fetch, "resolve_latest_cycle", fake_resolve)

    def bare(folder, cycle, *extra):
        out = tmp_path / folder / "area.toml"
        rc = cli_main(["domain", "--point=39.7,-96.6", "--card", "16gb",
                       "--ladder", "12-3", "--cycle", cycle, "--out", str(out),
                       "--explain", *extra])
        captured = capsys.readouterr()
        return rc, out, captured.out + captured.err

    rc, out, text = bare("recent", "2026-10-16T00")
    assert rc == 0
    config = tomllib.loads(out.read_text(encoding="utf-8"))
    assert config["fetch"]["source"] == "gfs"
    assert "era5_provider" not in config["fetch"]
    chosen = [line for line in text.splitlines() if line.startswith("source: GFS, because ERA5 for 2026-10-16T00Z")]
    assert len(chosen) == 1 and "2026-10-11T12Z" in chosen[0] and "no Copernicus CDS key" in chosen[0]
    assert "keyless ARCO store, because" not in text

    rc, out, text = bare("held", "2026-10-10T00")
    assert rc == 0
    config = tomllib.loads(out.read_text(encoding="utf-8"))
    assert config["fetch"]["source"] == "era5" and config["fetch"]["era5_provider"] == "arco"
    assert "source: GFS" not in text and "keyless ARCO store, because" in text

    rc, out, text = bare("named", "2026-10-16T00", "--source", "era5")
    assert rc != 0 and not out.exists()
    assert "not in Google's keyless ARCO store yet" in text
    assert "2026-10-11T12Z" in text and "--source gfs" in text


def test_the_next_block_credential_line_is_derived_from_the_registry_row(
        tmp_path, capsys, monkeypatch):
    """THE arbitrary acceptance test for the wizard's credential line.

    The line used to be an ``if source == "era5"`` arm, so a second
    source that needed an account key would have needed a second arm.
    It reads the row's CREDENTIAL column now: a credential declared on
    any row -- here grafted onto a source that has never needed one --
    produces the same pointer, gated the same way, with nothing edited
    in the wizard.
    """

    import dataclasses

    from woof import source_adapters as registry
    from woof.source_credentials import CredentialLocation, SourceCredential

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(registry.Path, "home", staticmethod(lambda: home))
    credential = SourceCredential(
        credential_id="probe-arbitrary-credential",
        display_name="Probe Arbitrary key",
        location_kind=CredentialLocation.HOME_FILE,
        location=".probe-arbitrary-key",
        needed_for="acquisition",
        breakage="the download is rejected by the provider",
        obtain_url="https://example.invalid/keys")
    donor = registry.get_source_adapter("gfs")
    grafted = dataclasses.replace(donor, credentials=(credential,))
    monkeypatch.setattr(
        registry, "_ADAPTERS",
        tuple(grafted if adapter.source_id == "gfs" else adapter
              for adapter in registry.source_adapters()))
    monkeypatch.setitem(registry._ALIASES, "gfs", grafted)  # noqa: SLF001

    rc, _ = _run_wizard(tmp_path / "no-key", ladder="12", source="gfs",
                        cycle="2026-07-29T18")
    absent = capsys.readouterr().out
    assert rc == 0
    assert "Probe Arbitrary key" in absent
    assert str(home / ".probe-arbitrary-key") in absent
    assert "https://example.invalid/keys" in absent

    (home / ".probe-arbitrary-key").write_text("key: x\n")
    rc, _ = _run_wizard(tmp_path / "with-key", ladder="12", source="gfs",
                        cycle="2026-07-29T18")
    present = capsys.readouterr().out
    assert rc == 0
    assert "Probe Arbitrary key" not in present


def test_a_gfs_wizard_run_gets_no_era5_credential_line(tmp_path, capsys):
    """The pointer belongs to the route that needs it, and to no other."""

    rc, _ = _run_wizard(tmp_path, ladder="12", source="gfs",
                        cycle="2026-07-29T18")
    printed = capsys.readouterr().out
    assert rc == 0
    assert "Copernicus CDS key" not in printed
    assert "woof go " in printed.split("next:")[-1]


# ---------------------------------------------------------------------------
# The closing block names the route the emitted file is actually on
# ---------------------------------------------------------------------------

def _emit(tmp_path, capsys, *extra, source="gfs", name="area"):
    """Emit one config through the real CLI; return its printed output."""

    out = tmp_path / f"{name}.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--source", source,
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--card", "12gb", "--out", str(out), *extra])
    assert rc == 0
    return out, capsys.readouterr().out


def test_a_gfs_emission_never_points_at_gpuwm_run(tmp_path, capsys):
    """The bug an owner hit on 1.3.0, in the shape it hit him.

    ``woof run`` executes the ``[case_data]`` config-driven route,
    which is ERA5's; it refuses a GFS config by design and says so.  The
    closing block used to print it for every source, so following the
    numbered list to the end produced a refusal -- the tool telling its
    own user it was broken.  The block branches on the source now.
    """

    out, printed = _emit(
        tmp_path, capsys, "--ladder", "12", "--physics-profile",
        MORRISON_PROFILE_ID)
    block = printed.split("next:")[-1]
    assert "woof run " not in block
    assert f"woof go {_posix(out)}" in block


def test_the_gfs_emission_the_block_names_passes_gos_plan_reader(
        tmp_path, capsys):
    """What it names is not merely different -- it is accepted."""

    from woof.go_cli import plan_from_config

    out, printed = _emit(
        tmp_path, capsys, "--ladder", "12", "--physics-profile",
        MORRISON_PROFILE_ID)
    assert "woof go " in printed.split("next:")[-1]
    plan = plan_from_config(out)
    assert plan["profile"] == MORRISON_PROFILE_ID
    assert plan["source"] == "gfs"


def test_an_era5_emission_still_points_at_gpuwm_run(tmp_path, capsys):
    """ERA5 is the route `woof run` exists for; nothing changed there."""

    out, printed = _emit(tmp_path, capsys, "--ladder", "12", source="era5")
    assert f"woof run {_posix(out)}" in printed.split("next:")[-1]


def test_an_hrrr_emission_names_its_automatic_native_launch(tmp_path, capsys):
    out, printed = _emit(tmp_path, capsys, "--ladder", "12", "--card", "24gb",
                         source="hrrr", name="hrrr-area")
    block = printed.split("next:")[-1]
    assert f"woof go {_posix(out)}" in block
    assert "woof run " not in block
    assert "sha256" not in block
    assert all(path.exists() for path in route_input_paths(out).values())



def test_a_nested_hrrr_emission_drives_the_route_with_no_hand_edits(
        tmp_path, capsys, pinned_thompson_tables):
    """The acceptance this lane exists for, run through the ROUTE's gates.

    ``woof domain --source hrrr`` used to emit one file of the five the
    nested HRRR route consumes, and the namelist.wps it did emit was
    missing ``&share/interval_seconds`` -- the single key that route's
    first gate demands.  The gate run that proved the route worked at
    all had to author the rest with a lane's proof harness.

    So this test does not check that four files exist.  It runs the
    route's own validators, imported from the route, over the exact
    bytes the wizard wrote: the raw-WPS contract, the native/stock
    delta, the namelist import (which carries the Lambert contract
    check), the public hierarchy slice, and the root preparer's profile
    binding and vertical grid.  Every one of them is the thing that
    refused wizard output before.
    """
    from woof.hrrr_hierarchy_direct import (
        _native_experiment, _require_raw_stock_delta,
        _require_raw_wps_contract, _supported_hierarchy_slice)
    from woof.ingest.hrrr_target import (load_hrrr_target_domain,
                                          required_hrrr_source_window)
    from woof.physics_menu import default_profile_for
    from woof.vertical_contract import explicit_vertical_from_wrf_namelist
    from tools.hrrr_single_domain_benchmark import (
        _validate_native_hrrr_physics_profile)

    out = tmp_path / "wind.toml"
    rc = cli_main(["domain", "--point=46.4,-118.3", "--card", "24gb",
                   "--root-dx", "3", "--chain", "4", "--source", "hrrr",
                   "--cycle", "2026-07-29T18", "--hours", "3",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed

    paths = route_input_paths(out)
    exp = load_experiment(out)
    assert len(exp.domains) == 2

    # The route's raw contract gates, on the emitted bytes.
    assert _require_raw_wps_contract(
        paths["wps_namelist"], len(exp.domains))["status"] == "PASS"
    assert _require_raw_stock_delta(
        paths["namelist_input"],
        paths["stock_namelist_input"])["status"] == "PASS"

    # The target-domain document loads, and its HRRR source window fits.
    target = load_hrrr_target_domain(paths["target_domain"])
    assert (target.nx, target.ny, target.nz) == (
        exp.root.run.nx, exp.root.run.ny, exp.root.run.nz)
    required_hrrr_source_window(target)

    # The importer + Lambert contract check the hierarchy runs, and the
    # public slice gate, for this run's own forcing inventory.
    native_exp, _resolved, _report = _native_experiment(
        paths["wps_namelist"], paths["namelist_input"])
    hours = tuple(range(int(exp.run_seconds // 3600) + 1))
    _supported_hierarchy_slice(native_exp, target, forcing_hours=hours)

    # The namelist describes the same tree as the TOML beside it --
    # including the epssm that made this lane necessary.
    assert [(d.run.nx, d.run.ny, float(d.run.epssm))
            for d in native_exp.domains] == [
        (d.run.nx, d.run.ny, float(d.run.epssm)) for d in exp.domains]
    assert {float(d.run.epssm) for d in native_exp.domains} == {0.5}

    # And the ROOT preparation's own two gates over the same file.
    # Bound to the default this tree takes rather than a literal: the
    # point is that the wizard emitted namelists the root preparer
    # accepts for the suite it emitted, whichever suite that is.  Its
    # 750 m nest binds the sub-km default, which a nested hrrr tree takes
    # now that the hierarchy stage pins the soil its land surface runs.
    suite = default_profile_for(
        "hrrr", min(d.run.dx for d in exp.domains), len(exp.domains))
    binding = _validate_native_hrrr_physics_profile(
        paths["namelist_input"], suite)
    assert binding["profile"] == suite
    # ... and that suite runs BOTH radiation streams, which is the
    # property 1.8 exists to give this route.
    assert binding["resolved"]["ra_lw_physics"] == 4
    assert binding["resolved"]["ra_sw_physics"] == 4
    # Its microphysics tables were staged and byte-checked at binding.
    assert binding["microphysics_table_authority"]["table_set"]
    vertical = explicit_vertical_from_wrf_namelist(
        paths["namelist_input"], expected_nz=target.nz,
        context="native HRRR initializer")
    assert vertical.eta_levels == exp.vertical.eta_levels
    assert vertical.p_top == exp.vertical.p_top


def test_a_nested_hrrr_next_block_names_the_automatic_native_launch(tmp_path, capsys):
    out, printed = _emit(tmp_path, capsys, "--root-dx", "3", "--chain", "4",
                         "--card", "24gb", source="hrrr", name="hrrr-tree")
    block = printed.split("next:")[-1]
    assert f"woof go {_posix(out)}" in block
    assert "sha256" not in block
    assert "--materialize-authorities" not in block
    assert all(path.exists() for path in route_input_paths(out).values())



def test_a_nested_hrrr_chain_at_a_lead_hands_both_stages_the_same_two_values(
        tmp_path, capsys):
    """The lead is printed beside the cycle, on every stage of the chain.

    At lead 0 the cycle and model time zero are the same instant, so one
    string served both stages for four releases.  At lead 6 the printed
    chain has to say which is which, and both stages have to be able to
    derive the other.
    """
    out = tmp_path / "lead-tree.toml"
    assert cli_main([
        "domain", "--point=46.4,-118.3", "--card", "24gb",
        "--root-dx", "3", "--chain", "4", "--source", "hrrr",
        "--cycle", "2026-07-29T18", "--hours", "3",
        "--forecast-start-hour", "6", "--out", str(out), "--explain"]) == 0
    printed = capsys.readouterr().out
    block = printed.split("next:")[-1]

    from woof.domain_wizard import hrrr_route_commands
    from woof.experiment import load_experiment
    block += "\n" + hrrr_route_commands(
        out, load_experiment(out), profile=None, data_dir="data",
        forecast_start_hour=6)

    # The fetch downloads f06..f09, not f00..f03, and both preparation
    # stages carry the same cycle and the same lead.
    assert "woof fetch --source hrrr --cycle 2026-07-29T18" in block
    assert block.count("--forecast-start-hour 6") == 3  # fetch + both stages
    assert block.count("--valid-time 2026-07-29_18:00:00") == 2
    assert "--cycle 2026-07-29_18:00:00" not in block
    # The emitted config, and therefore the namelist the hierarchy reads
    # and compares against model time zero, start at cycle + 6 h.  Before
    # the lead was reachable here the namelist could only say the cycle
    # hour, which the nested route refuses at any nonzero lead.
    text = out.read_text(encoding="utf-8")
    assert "start_time = 2026-07-30T00:00:00" in text
    assert "forecast_start_hour = 6" in text
    namelist = route_input_paths(out)["namelist_input"].read_text(
        encoding="utf-8")
    start = {line.split("=")[0].strip(): line.split("=")[1].strip()
             for line in namelist.splitlines()
             if line.strip().startswith("start_")}
    assert start["start_day"] == "30, 30,"
    assert start["start_hour"] == "00, 00,"


def test_hrrr_sizing_respects_hrrrs_own_grid_not_only_the_card(
        tmp_path, capsys):
    """VRAM is not the only bound on how large an HRRR domain may be.

    HRRR's native grid is 1799 x 1059, and the interpolation stencil
    needs real source cells outside the target on every side.  A ladder
    sized purely against VRAM is a legal, well-sized experiment that no
    HRRR fetch can force: on a 24 GiB Linux card, a 3 km root near the
    Washington/Oregon border ran its halo nine rows off the top of the
    HRRR grid, and the root preparation found out after the download.

    Sized against the SOURCE's own window function -- the one the root
    preparer calls -- the fit loop stops where HRRR does, and says so.
    The soil donor search does not stop it: its box stops at HRRR's own
    edge, where there is nothing to search, so the fitted domain's
    interpolation reaches the edge and its donor box stops there.
    """
    from dataclasses import replace

    from woof.hrrr_route_inputs import coverage_refusal, target_domain
    from woof.ingest.hrrr_target import (HRRR_SOURCE_NY,
                                          required_hrrr_source_window)

    out = tmp_path / "edge.toml"
    # The assertion below is that HRRR's GRID, not the card, is what
    # stopped the fit -- so the card has to be big enough that it is not
    # the bound.  1.8's full-radiation default raised the per-cell cost,
    # and at 32 GiB the card became the binding constraint again, which
    # would have made this test pass for the wrong reason.
    rc = cli_main(["domain", "--point=46.35,-118.10", "--vram-gib", "48",
                   "--root-dx", "3", "--chain", "4", "--source", "hrrr",
                   "--cycle", "2026-07-29T18", "--hours", "1",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed

    exp = load_experiment(out)
    assert coverage_refusal(exp) is None
    target = target_domain(exp)
    assert target.surface_fallback_radius_cells == 24
    # It stopped where the interpolation reaches HRRR's top edge -- HRRR's
    # grid, not the card, was the bound -- and the printed advisory says so.
    atmosphere = required_hrrr_source_window(
        replace(target, surface_fallback_radius_cells=0))
    assert atmosphere.j_end == HRRR_SOURCE_NY - 1
    # The donor box, 24 cells wide, stops on the same edge.
    assert required_hrrr_source_window(target).j_end == HRRR_SOURCE_NY - 1
    assert "bounded by HRRR's own grid, not by your card" in printed


def _conus_polygon(tmp_path):
    """A KTBW-shaped box deep inside HRRR coverage, as the nowcast draws it."""
    import json

    box = {"type": "Polygon", "coordinates": [[
        [-84.287, 25.628], [-80.275, 25.628], [-80.275, 29.189],
        [-84.287, 29.189], [-84.287, 25.628]]]}
    polygon = tmp_path / "box.geojson"
    polygon.write_text(json.dumps(box), encoding="utf-8")
    return polygon


def test_a_1p5km_hrrr_root_carries_its_7p5s_clock_exactly(tmp_path):
    """dx 1.5 km -> 7.5 s root clock, spec -> wizard -> namelists, exact.

    Field 2026-08-06: the rung-0 1.5 km screen's first HRRR case refused
    at the domain stage -- "the root domain spec carries an integer time
    step; this ladder's root clock is 7.5 s" -- while the SAME grid had
    run the GFS route all night, whose experiment TOML spells the clock
    ``time_step = 7`` + ``time_step_fract_num/den = 1/2`` (WRF's own
    registry spelling).  The target-domain spec now carries exactly that
    decomposition, and both emitted WRF namelists spell it the same way.
    """
    import json
    import re

    from woof.ingest.hrrr_target import load_hrrr_target_domain

    out = tmp_path / "clock.toml"
    rc = cli_main(["domain", "--polygon", str(_conus_polygon(tmp_path)),
                   "--root-dx", "1.5", "--source", "hrrr",
                   "--cycle", "2026-07-29T18", "--hours", "4",
                   "--name", "clockcase", "--out", str(out)])
    assert rc == 0

    spec_path = tmp_path / "clock.d01-target.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["time_step_seconds"] == 7
    assert spec["time_step_fract_num"] == 1
    assert spec["time_step_fract_den"] == 2

    target = load_hrrr_target_domain(spec_path)
    assert target.time_step_exact == Fraction(15, 2)

    # Both halves of the namelist pair spell the rational clock the way
    # stock WRF reads it -- never a bare 7, never a float 7.5.
    for name in ("clock.namelist.input", "clock.stock.namelist.input"):
        namelist = (tmp_path / name).read_text(encoding="utf-8")
        assert re.search(r"time_step\s*=\s*7\s*,", namelist), name
        assert re.search(r"time_step_fract_num\s*=\s*1\s*,", namelist), name
        assert re.search(r"time_step_fract_den\s*=\s*2\s*,", namelist), name

    # And a whole-second clock's spec payload is unchanged by the
    # widening: no fract keys, so every stored identity survives.
    whole = json.loads(hrrr_route_inputs_render_target_domain_at_3km(
        tmp_path))
    assert "time_step_fract_num" not in whole


def hrrr_route_inputs_render_target_domain_at_3km(tmp_path) -> str:
    """A 3 km emission (15 s clock), for the payload-stability check."""
    from woof.experiment import load_experiment
    from woof.hrrr_route_inputs import render_target_domain

    out = tmp_path / "whole.toml"
    rc = cli_main(["domain", "--polygon", str(_conus_polygon(tmp_path)),
                   "--root-dx", "3", "--source", "hrrr",
                   "--cycle", "2026-07-29T18", "--hours", "4",
                   "--name", "wholecase", "--out", str(out)])
    assert rc == 0
    return render_target_domain(load_experiment(out))


def test_native_hrrr_polygon_fits_about_the_emitted_cell_center(tmp_path):
    """A deep polygon stays whole when its requested center is off lattice."""
    from dataclasses import asdict

    from woof.domain_wizard import (load_polygon_footprint,
                                     verify_polygon_containment)
    from woof.static.external_source import (crop_window, source_grid,
                                               static_source_row)
    from woof.static.source_defaults import (align_source_projection,
                                               source_static_defaults)

    row = static_source_row(source_static_defaults("hrrr")["source"])
    source = source_grid(row)
    lat, lon = source.ij_to_latlon(900.01, 500.01)
    lat, lon = float(lat), float(lon)
    polygon = tmp_path / "off-lattice.geojson"
    polygon.write_text(json.dumps({"type": "Polygon", "coordinates": [[
        [lon - .005, lat - .005], [lon + .005, lat - .005],
        [lon + .005, lat + .005], [lon - .005, lat + .005],
        [lon - .005, lat - .005]]]}), encoding="utf-8")
    footprint = load_polygon_footprint(polygon)
    x, y = source.latlon_to_ij(footprint.center_lat, footprint.center_lon)
    # 6a69b356f, lane/hrrr-statics, snaps exact-spacing roots to source
    # mass points. This center is nearly half a root cell from that lattice,
    # amplified by 27 on d04; sizing before the snap used to clip the polygon.
    assert abs(float(x) - (round(float(x) - .5) + .5)) > .45
    assert abs(float(y) - (round(float(y) - .5) + .5)) > .45
    out = tmp_path / "off-lattice.toml"
    rc = cli_main([
        "domain", "--polygon", str(polygon), "--root-dx", "3",
        "--chain", "3,3,3", "--source", "hrrr", "--vram-gib", "48",
        "--cycle", "2026-07-29T18", "--hours", "1", "--out", str(out)])
    assert rc == 0
    exp = load_experiment(out)
    assert len(exp.domains) == 4
    verify_polygon_containment(exp, footprint, (0., 0., 0., 0.))
    grids = grids_from_projection_config(exp)
    for key in ("map_proj", "truelat1", "truelat2", "stand_lon"):
        assert getattr(grids[0], key) == row.grid_map[key]
    assert crop_window(row, grids[0]) is not None
    projection = asdict(exp.projection)
    aligned = align_source_projection(
        projection, (grids[0].e_we - 1, grids[0].e_sn - 1), 3000., "hrrr")
    assert aligned == projection  # Final alignment cannot move the fitted center.


def test_a_spec_refusal_is_not_dressed_as_a_coverage_refusal(
        tmp_path, monkeypatch, capsys):
    """Each refusal path states its own cause and its own remedy.

    Field 2026-08-06: the integer-clock refusal surfaced as "polygon
    ...  falls outside HRRR coverage: <the clock sentence>.  Move
    --polygon inside the HRRR grid" -- two unrelated errors in one
    sentence, and the named remedy (moving the polygon) could never
    have fixed the actual problem.  A spec that cannot be constructed
    now refuses with its own words; the coverage sentence is reserved
    for coverage answers.
    """
    import woof.domain_wizard as wizard
    from woof.hrrr_route_inputs import HrrrRouteInputError

    def broken_spec(exp):
        raise HrrrRouteInputError(
            "the spec itself cannot be built; its own remedy")

    monkeypatch.setattr(wizard, "coverage_refusal", broken_spec)
    rc = cli_main(["domain", "--polygon", str(_conus_polygon(tmp_path)),
                   "--root-dx", "3", "--source", "hrrr",
                   "--cycle", "2026-07-29T18", "--hours", "4",
                   "--name", "sepcase", "--out", str(tmp_path / "sep.toml")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "the spec itself cannot be built; its own remedy" in err
    assert "falls outside HRRR coverage" not in err
    assert "Move --polygon" not in err


def test_hrrr_coverage_test_lets_the_donor_search_stop_at_hrrrs_edge():
    """Field 2026-08 (RTX PRO 6000, 1 km nest at 39,-98).

    The auto-fitted 1234x986 3 km root's radius-8 window is
    i=250..1523, j=36..1058, exactly on the native top edge, and HRRR
    soil mapping could not fill two land cells within 8 cells.  Raising
    surface_fallback_radius_cells to 16 asked for source rows past the
    edge (j=28..1066 against j max 1058) and was refused, so the
    coverage test then demanded a whole radius of margin from every edge
    and refused this domain although HRRR covers its atmosphere.

    The donor search's box now stops at HRRR's own edge, where no donor
    exists: the domain is accepted, and a wider search is accepted with
    it, its window on the same edge.
    """
    from dataclasses import replace

    from woof.hrrr_route_inputs import target_coverage_refusal
    from woof.ingest.hrrr_target import (HRRR_SOURCE_NY, HrrrTargetDomain,
                                          required_hrrr_source_window)

    def field_target(nx, ny):
        return HrrrTargetDomain(
            name="field_39n_98w_3km", map_proj="lambert",
            nx=nx, ny=ny, nz=49, dx_m=3000.0, dy_m=3000.0,
            ref_lat=39.0, ref_lon=-98.0,
            truelat1=29.0, truelat2=49.0, stand_lon=-98.0,
            time_step_seconds=15)

    fitted = field_target(1234, 986)
    window = required_hrrr_source_window(fitted)
    assert (window.j_start, window.j_end) == (36, HRRR_SOURCE_NY - 1)
    wider = required_hrrr_source_window(
        replace(fitted, surface_fallback_radius_cells=16))
    assert (wider.j_start, wider.j_end) == (28, HRRR_SOURCE_NY - 1)
    assert target_coverage_refusal(fitted) is None
    assert target_coverage_refusal(
        replace(fitted, surface_fallback_radius_cells=16)) is None
    # The field workaround, 24 cells trimmed from every side, still fits.
    assert target_coverage_refusal(field_target(1186, 938)) is None


def test_a_point_hrrr_cannot_force_at_all_is_refused_by_the_fit_loop(
        tmp_path, capsys):
    """Watched firing: outside HRRR's coverage, nothing fits.

    Not a silent shrink to zero and not a refusal about memory -- the
    smallest layout this ladder has still needs source cells HRRR does
    not have, and the message says so and names the flag.
    """
    out = tmp_path / "atlantic.toml"
    rc = cli_main(["domain", "--point=25.0,-40.0", "--card", "12gb",
                   "--ladder", "12", "--source", "hrrr",
                   "--cycle", "2026-07-29T18", "--hours", "1",
                   "--out", str(out)])
    captured = capsys.readouterr()
    assert rc != 0
    message = captured.err + captured.out
    assert "cannot be forced by hrrr" in message
    assert "leaves HRRR coverage" in message
    assert "Move --point" in message
    assert not out.exists()


def test_hrrr_emission_preserves_an_explicit_profile_beyond_the_default(
        tmp_path, capsys):
    """Named-profile evidence does not limit implemented physics choices."""
    from woof.domain_wizard import resolved_physics_profile
    from woof.hrrr_route_inputs import verify_round_trip

    out = tmp_path / "morrison.toml"
    rc = cli_main(["domain", "--point=46.4,-118.3", "--card", "24gb",
                   "--ladder", "12-3", "--source", "hrrr",
                   "--cycle", "2026-07-29T18", "--hours", "3",
                   "--physics-profile", MORRISON_PROFILE_ID,
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0
    assert f"woof go {_posix(out)}" in printed.split("next:")[-1]
    exp = load_experiment(out)
    expected = single_domain_runtime_switches(MORRISON_PROFILE_ID)
    for key, value in expected.items():
        assert getattr(exp.root.run, key) == value, key
    assert exp.root.run.cu_physics == 1
    assert [domain.run.cu_physics for domain in exp.domains] == [1, 0]
    assert all(domain.run.mp_physics == expected["mp_physics"]
               for domain in exp.domains)
    paths = route_input_paths(out)
    verify_round_trip(exp, paths["wps_namelist"], paths["namelist_input"])
    before = out.read_bytes()
    assert cli_main(["go", str(out), "--dry-run"]) == 0
    capsys.readouterr()
    assert out.read_bytes() == before

    # Removing the membership ban did not replace the source's recommendation.
    good = tmp_path / "default.toml"
    assert cli_main([
        "domain", "--point=46.4,-118.3", "--card", "24gb",
        "--ladder", "12-3", "--source", "hrrr", "--cycle",
        "2026-07-29T18", "--hours", "3", "--out", str(good)]) == 0
    capsys.readouterr()
    default = load_experiment(good)
    recommended = resolved_physics_profile("hrrr", None)
    assert recommended != MORRISON_PROFILE_ID
    for key, value in single_domain_runtime_switches(recommended).items():
        assert getattr(default.root.run, key) == value, key


def _named_licence_opt_in(source: str) -> tuple[str, ...]:
    """The licence(s) a licence-gated row's own refusal names, else none.

    Rows whose weights are not permissive refuse every default path and
    name the exact ``--accept-licence`` value that opts in.  A sweep over
    every planable row opts in with exactly that value, as a user would;
    the refusal itself is held by the licensing tests.
    """

    import shlex

    from woof.earth2_licensing import check_argv

    try:
        check_argv(["domain", "--source", source])
    except ValueError as refusal:
        match = re.search(r"--accept-licence (.+)$", str(refusal))
        assert match, refusal
        return tuple(shlex.split(match.group(1)))
    return ()


def _advised_lighter_profiles(message: str) -> list[str]:
    """The suite names a no-fit refusal advises, in printed order."""

    match = re.search(r"lighter --physics-profile \(([^)]+)\)", message)
    if match is None:
        return []
    return [name.strip() for name in match.group(1).split(",")]


@pytest.mark.parametrize("source", wizard_planable_source_ids())
def test_no_fit_advice_names_only_profiles_the_same_source_accepts(
        tmp_path, capsys, source):
    """A refusal must never advise a suite the very same command refuses.

    Measured on the 2.5.0 line: ``woof domain --source gfs --vram-gib
    4`` refused the ladder and ranked a RUC-LSM suite FIRST in its
    lighter-profile advice -- a suite the same wizard refuses outright
    for gfs, because the registry's land-surface table withdraws ruc-lsm
    from the GFS route.  Advice admissibility was checked for one
    hard-coded source only, so every other source's advice went out
    unfiltered.  The concrete breakage: a reader follows the refusal's
    first-ranked remedy and is refused again by the same door.

    Every planable source, through the real CLI: force the no-fit
    refusal, then re-run the same command with each advised profile.
    ACCEPTED means neither the pairing gate nor the memory fit refuses
    it: rc 0, or a refusal about something else entirely (a nocturnal
    acknowledgement), never the source-pairing refusal, and never the
    budget the advice was offered as a way out of.
    """

    window = source_coverage_window(source)
    if window is None:
        point = "39.1,-94.6"
    elif hasattr(window, "grid"):
        # A Lambert window's lat/lon ENVELOPE can wrap the antimeridian
        # (RAP reaches past Alaska), which puts the box midpoint an
        # ocean away from the grid; the grid's own center point cannot.
        lat, lon = window.grid().ij_to_latlon(
            0.5 * (window.nx + 1), 0.5 * (window.ny + 1))
        point = f"{float(lat):.2f},{float(lon):.2f}"
    else:
        south, west, north, east = window.envelope()
        point = f"{0.5 * (south + north):.2f},{0.5 * (west + east):.2f}"

    # 315803306, lane/20cr-site, admits the historical 1836..2015 archive.
    # Advice ranking needs a valid request to reach memory sizing, not a
    # 2026 window refused by the source's unchanged temporal guard.
    cycle = "1957-12-18T12" if source == "20crv3-cf" else "2026-08-12T00"
    # The dated GEFS archive rows (lane/cap-gefs-archive, 11c73fea3b9 and
    # 6a3b8fd50bf) are closed publication eras, and the GEFS reforecast
    # publishes no f000: the request is placed inside the source's own
    # declared era and starts at its own first published lead, both read
    # from the route table.
    start = []
    from woof import fetch_routes
    canonical = fetch_routes.canonical_source(source)
    if canonical in fetch_routes.route_ids():
        route = fetch_routes.route_for(canonical)
        from datetime import datetime
        asked = datetime.fromisoformat(cycle)
        eras = route.publication_eras
        if eras and not any(era.valid_from <= asked and (
                era.valid_until is None or asked < era.valid_until)
                for era in eras):
            # The newest era the route prepares, at its own first cycle:
            # fix/ifs-0p1-first-cycle (17d77cbbaa) gives ecmwf-open-data-0p1
            # a first-cycle era with a prep_refusal and starts its 06/18Z
            # era at 18Z, so midnight of that day predates the set.
            preparable = [era for era in eras if not era.prep_refusal]
            cycle = f"{preparable[-1].valid_from:%Y-%m-%dT%H}"
        if route.first_lead:
            start = ["--forecast-start-hour", str(route.first_lead)]

    opt_in = [token for licence in _named_licence_opt_in(source)
              for token in ("--accept-licence", licence)]

    def _argv(vram: str) -> list[str]:
        return ["domain", f"--point={point}", "--ladder", "12",
                "--source", source, "--cycle", cycle,
                "--hours", "6", "--vram-gib", vram, *start, *opt_in]

    # Force the fit loop's own refusal -- the one that ranks lighter
    # profiles.  The card size that reaches it depends on the source's
    # default suite: below its reserve wall the wizard refuses earlier
    # ("no budget at all", no ranking), and on a large enough card the
    # ladder simply fits, so walk up until the ranking refusal fires.
    message = vram = None
    for candidate in ("4", "5", "6", "8"):
        rc = cli_main([*_argv(candidate),
                       "--out", str(tmp_path / "refused.toml")])
        captured = capsys.readouterr()
        text = captured.err + captured.out
        if rc != 0 and "does not fit" in text:
            message, vram = text, candidate
            break
    # Watched firing: the no-fit refusal this test is about, not some
    # other gate reached first.
    assert message is not None, f"{source}: no-fit refusal never forced"
    for rank, profile in enumerate(_advised_lighter_profiles(message)):
        rc = cli_main([*_argv(vram), "--physics-profile", profile,
                       "--out", str(tmp_path / f"advised-{rank}.toml")])
        captured = capsys.readouterr()
        follow = captured.err + captured.out
        for needle in ("cannot be prepared with --source",
                       "cannot drive the nested HRRR route"):
            assert needle not in follow, (
                f"{source} advice ranked {profile} at #{rank + 1} and "
                f"the same wizard refuses the pairing: {follow}")
        # Nor the budget: a suite named as a way out of a memory refusal
        # must fit the budget that refused.
        for needle in ("does not fit", "no budget for ladder"):
            assert needle not in follow, (
                f"{source} advice ranked {profile} at #{rank + 1} and "
                f"the same card refuses it for memory: {follow}")


def test_hypsometric_opt_is_emitted_where_wrf_declares_it(tmp_path):
    """The key that made wrf.exe FATAL before its first timestep.

    WRF v4.6.1 declares ``hypsometric_opt`` as a &domains SCALAR
    (Registry.EM_COMMON:2283, ``namelist,domains``, nentries 1; the
    compiled binary carries ``NAMELIST /domains/ hypsometric_opt``).
    The emitter wrote it into &dynamics, so the Fortran namelist read of
    that group failed and every rank died before a timestep -- measured
    on a campaign node.  Both emitted flavors are checked, section
    scoped: a whole-file substring would pass on a file that still put
    the key in the wrong group.
    """
    from woof.namelist_import import parse_namelist

    out = tmp_path / "hypsometric.toml"
    assert cli_main([
        "domain", "--point=46.4,-118.3", "--card", "24gb",
        "--ladder", "12-3", "--source", "hrrr", "--cycle",
        "2026-07-29T18", "--hours", "3", "--out", str(out)]) == 0
    paths = route_input_paths(out)
    exp = load_experiment(out)
    assert len(exp.domains) == 2

    for flavor in ("namelist_input", "stock_namelist_input"):
        sections = parse_namelist(paths[flavor])
        assert "hypsometric_opt" not in sections["dynamics"], flavor
        # A scalar, not a per-domain column: a namelist object declared
        # with nentries 1 takes exactly one value, so mirroring the
        # config's per-domain field here would fail the &domains read
        # for the other reason.
        assert sections["domains"]["hypsometric_opt"] == [
            exp.root.run.hypsometric_opt], flavor


def test_no_experiment_can_carry_a_per_domain_hypsometric_opt(tmp_path):
    """What makes the scalar emission above lossless rather than lossy.

    WRF has ONE hypsometric_opt for the whole run, woof's RunConfig
    carries one per domain, and the emitter writes the root's.  That is
    safe only because the experiment schema refuses the key as a
    [[domain]] override -- it reaches every domain from [shared].  Pinned
    here, so a future widening of the per-domain key set has to answer
    the namelist question rather than silently leave the WRF arm running
    d01's choice on every nest.
    """

    out = tmp_path / "split.toml"
    assert cli_main([
        "domain", "--point=46.4,-118.3", "--card", "24gb",
        "--ladder", "12-3", "--source", "hrrr", "--cycle",
        "2026-07-29T18", "--hours", "3", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "hypsometric_opt = 2" in text  # [shared], for the whole tree

    split = tmp_path / "split_nest.toml"
    split.write_text(
        text + "\n[[domain]]\ngrid_id = 2\nhypsometric_opt = 1\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="NOT per domain"):
        load_experiment(split)


def test_the_emitted_namelists_are_refused_if_they_drift_from_the_config(
        tmp_path):
    """The round trip that makes 'zero hand edits' mechanical.

    The route reads the namelists, not the TOML.  A set that describes a
    different tree than the config beside it is a defect that surfaces
    only after a fetch and a root preparation, so the writer re-imports
    what it wrote through the REAL importer and refuses on any
    difference.  Watched firing: one edited geometry key, and it fires.
    """
    from woof.hrrr_route_inputs import verify_round_trip

    out = tmp_path / "drift.toml"
    assert cli_main([
        "domain", "--point=46.4,-118.3", "--card", "24gb",
        "--ladder", "12-3", "--source", "hrrr", "--cycle",
        "2026-07-29T18", "--hours", "3", "--out", str(out)]) == 0
    paths = route_input_paths(out)
    exp = load_experiment(out)

    # Unedited, the round trip is silent.
    verify_round_trip(exp, paths["wps_namelist"], paths["namelist_input"])

    # The edit is this lane's own bug, written into the namelist: the
    # nest back on WRF's Registry default while the config says 0.5.
    namelist = paths["namelist_input"]
    text = namelist.read_text(encoding="utf-8")
    edited = text.replace("0.5, 0.5,", "0.5, 0.1,", 1)
    assert edited != text
    namelist.write_text(edited, encoding="utf-8")
    with pytest.raises(HrrrRouteInputError, match="epssm"):
        verify_round_trip(exp, paths["wps_namelist"], namelist)


def test_a_tree_emission_names_the_automatic_native_launch(tmp_path, capsys):
    out, printed = _emit(tmp_path, capsys, "--ladder", "12-3", "--name", "treecase")
    block = printed.split("next:")[-1]
    assert f"woof go {_posix(out)}" in block
    assert "woof run " not in block
    assert "--materialize-authorities" not in block
    assert "sha256" not in block



def test_a_tree_emissions_launch_command_selects_the_tree_runner(tmp_path, capsys):
    from woof.go_cli import TREE_RUNNER_MODULE, plan_from_config
    out, printed = _emit(tmp_path, capsys, "--ladder", "12-3", "--name", "treecase")
    assert "woof go " in printed.split("next:")[-1]
    plan = plan_from_config(out)
    assert plan["domains"] == 2
    assert plan["runner"] == TREE_RUNNER_MODULE



def test_the_manual_chain_pointer_is_reachable_without_a_checkout():
    """A-10.  The wheel ships no docs tree, so a repo-relative path was
    the whole of an instruction the reader provably could not follow."""
    from woof.go_cli import MANUAL_CHAIN

    assert "docs/public/FIRST-LIGHT.md" in MANUAL_CHAIN
    assert MANUAL_CHAIN.count("https://") == 1
    assert "recastsystems/woof" in MANUAL_CHAIN


def test_a_profileless_gfs_emission_points_at_gpuwm_go(tmp_path, capsys):
    """Converted (owner ruling 2026-07-31): the chain runs the default
    suite as written, so a profileless single-domain GFS emission gets
    the same one-command next step a bound one does."""

    _, printed = _emit(tmp_path, capsys, "--ladder", "12")
    block = printed.split("next:")[-1]
    assert "woof run " not in block
    assert "woof go " in block


def _posix(path) -> str:
    return str(path).replace("\\", "/")


# ---------------------------------------------------------------------------
# The advisory for a domain sized to the card rather than to the weather
# ---------------------------------------------------------------------------

def test_a_card_filling_footprint_gets_an_advisory_not_a_refusal():
    """An owner's 32 GiB emission spanned 152 degrees of longitude.

    Legal arithmetic -- the sizer's job is to use the card it was given
    -- and an absurd first run.  This says so once, names the flag that
    makes it smaller, and refuses nothing.
    """

    from woof.domain_wizard import oversized_footprint_advisory

    wide = oversized_footprint_advisory("-6.39,-159.63,73.19,-35.37")
    assert len(wide) == 1
    assert "--vram-gib" in wide[0]
    assert "124 x" in wide[0]
    # It names the flag that CAUSED the box as well as the one that
    # shrinks it, and says why narrowing the download alone is wrong.
    assert "--area" in wide[0]
    assert "starve" in wide[0]

    # A continental domain is not remarkable and gets no line.
    assert oversized_footprint_advisory("6.24,-135.55,63.02,-59.45") == []
    # Nor is a shape this function cannot read a reason to say anything.
    assert oversized_footprint_advisory("not-a-box") == []

    # A box that is merely TALL used to pass unremarked, because only
    # longitude was measured: the wheel user's Linux --card 24gb
    # --ladder 12 emitted 88 degrees of latitude.  This one is 84 tall
    # and 70 wide -- under the longitude bar, over the latitude one.
    tall = oversized_footprint_advisory("-8.00,-120.00,76.00,-50.00")
    assert len(tall) == 1
    assert "70 x 84" in tall[0]


def test_the_advisory_reaches_the_terminal_on_a_large_card(tmp_path,
                                                           capsys):
    """Emitted for real, at the size that provoked it.

    The sentence no longer opens "sized to fill your card": on this very
    invocation the fit stops on a bound that is not memory, and the
    wizard says so, so an advisory naming the card as the cause would
    contradict the line above it.

    Since 2.7.3 the bound that stops this particular fit is the extent
    a point request is sized to, and the assertions below are the whole
    agreement: the plan summary's plain fact and the advisory name one
    bound and one flag between them, on stdout, with nothing added to
    stderr -- the cap is the ordinary sizing of an ordinary request and
    a warning would say otherwise.  The card-shaped remedy is tested
    where it still applies, in
    ``test_domain_wizard_point_extent_cap.py``.
    """

    out = tmp_path / "wide.toml"
    assert cli_main(["domain", "--point=35.3,-97.5", "--source", "gfs",
                     "--cycle", "2026-07-29T18", "--hours", "6",
                     "--ladder", "12", "--vram-gib", "32.00",
                     "--physics-profile", MORRISON_PROFILE_ID,
                     "--out", str(out)]) == 0
    captured = capsys.readouterr()
    printed = captured.out
    assert "advisory: this domain is much wider than the documented" \
        in printed
    advisory = [line for line in printed.splitlines()
                if line.startswith("advisory: this domain is much wider")]
    assert len(advisory) == 1
    assert "stopped on the REQUESTED EXTENT, not on the card" in advisory[0]
    assert "--polygon" in advisory[0]
    # The flag the fit made inert is not the one the line under it names.
    assert "--vram-gib" not in advisory[0]
    # The bound is stated once, as fact, on stdout -- and nowhere on
    # stderr, which this door's default emission is held to keep empty.
    fact = [line for line in printed.splitlines()
            if line.startswith("domain: point request:")]
    assert len(fact) == 1
    assert "extent capped at 6000 km" in fact[0]
    assert "memory allows more" in fact[0] and "--polygon" in fact[0]
    assert "not the card" not in captured.err
    assert "point request:" not in captured.err
    assert "woof domain: FAIL" not in printed
    assert "sized to fill your card" not in printed + captured.err


# ---------------------------------------------------------------------------
# 2026-08-01 sizing calibration: the tier, the reserve and the two checks
# ---------------------------------------------------------------------------

#: Free VRAM real cards of each tier hand to a fresh CUDA context.
#:
#: The 16 GiB row is MEASURED: an idle headless RTX 4080 (16,376 MiB
#: physical) presents 15.33 GiB.  The others are the same 0.66 GiB
#: driver/nameplate gap applied to the tier's nominal size, which is the
#: assumption the tier has to be conservative against.
REAL_CARD_FREE_GIB = {"12gb": 11.34, "16gb": 15.33, "24gb": 23.33,
                      "32gb": 30.27}


@pytest.mark.parametrize("card", sorted(CARD_VRAM_GIB))
def test_the_card_tier_is_conservative_against_a_real_card(card):
    """A tier may never assume more free VRAM than its class delivers.

    The 16 GiB tier assumed the card would hand over its whole nominal
    size.  It does not -- a real RTX 4080 presents 15.33 GiB of a 15.99
    GiB card -- so every ladder the tier emitted was sized against VRAM
    that does not exist, landed 0.13-0.32 GiB over the real budget, and
    failed the product's own `woof check` minutes after the wizard
    printed PASS.
    """
    assumed = card_assumed_free_gib(CARD_VRAM_GIB[card])
    assert assumed <= REAL_CARD_FREE_GIB[card], card
    # ...and not so conservative that the tier stops being useful.
    assert assumed >= REAL_CARD_FREE_GIB[card] - 1.0, card


@pytest.mark.parametrize("card", sorted(CARD_VRAM_GIB))
@pytest.mark.parametrize("ladder", ["12", "12-3", "12-3-1-0.5"])
def test_an_emitted_config_fits_the_card_it_was_sized_for(
        tmp_path, monkeypatch, card, ladder):
    """The A-0 regression, as an inequality rather than a subprocess.

    Every `--card 16gb` ladder v1.4.0 emitted exceeded the budget a real
    16 GB card leaves.  Re-priced here against that card's real free
    VRAM, runtime MYNN width and its own suite's reserve, with nothing
    added back.
    """
    rc, out = _run_wizard(tmp_path, card=card, ladder=ladder, source="gfs",
                          cycle="2026-07-28T00")
    assert rc == 0
    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    vram = CARD_VRAM_GIB[card]
    interval = 10800.0
    real_free = int(REAL_CARD_FREE_GIB[card] * GIB)
    with _runtime_card_pricing(monkeypatch, vram_gib=vram,
                               free_bytes=real_free):
        estimate = estimate_experiment(exp, forcing_interval_seconds=interval,
                                       vram_gib=vram)
        phases = estimate_phases(exp, source="gfs",
                                 forcing_interval_seconds=interval,
                                 vram_gib=vram)
        budget = sizing_budget_bytes(exp, free_bytes=real_free, vram_gib=vram,
                                     forcing_interval_seconds=interval)
    assert phases.peak_envelope_bytes <= budget, (
        f"{card} {ladder}: emitted envelope "
        f"{phases.peak_envelope_bytes / GIB:.2f} GiB over a real budget of "
        f"{budget / GIB:.2f} GiB")
    assert estimate.alloc_estimate_bytes <= budget


def test_the_wizard_labels_its_verdict_as_an_estimate_for_a_declared_card(
        tmp_path, capsys):
    """The wizard never measures a card; its verdict must say so.

    The 4090 stress run certified "fits with 0.27 GiB to spare" for a
    card not in the machine and the config landed 0.015 GiB from the
    budget.  Both renderings of the sizing verdict carry the label now:
    the one-line summary and the --explain table.
    """
    rc, _ = _run_wizard(tmp_path / "s", card="24gb", ladder="12",
                        source="gfs", cycle="2026-07-28T00")
    assert rc == 0
    printed = capsys.readouterr().out
    assert "estimate for a declared 24 GiB card" in printed
    assert "not a measurement of hardware in this machine" in printed

    rc, _ = _run_wizard(tmp_path / "e", "--explain", card="24gb",
                        ladder="12", source="gfs", cycle="2026-07-28T00")
    assert rc == 0
    printed = capsys.readouterr().out
    assert "ESTIMATE FOR HARDWARE NOT PRESENT" in printed
    assert "never more" in printed and "optimistic" in printed


NSSL2_PROFILES = (
    "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-wrf-comparison-candidate-v1",
    "nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-wrf-comparison-candidate-v1",
)


@pytest.mark.parametrize("physics_profile", NSSL2_PROFILES)
@pytest.mark.parametrize("vram", [12.0, 15.0, 24.0, 32.0])
def test_a_suite_with_a_large_backing_store_still_sizes(
        tmp_path, physics_profile, vram, capsys):
    """Defect 4: the reserve's overhead term is SUITE-dependent.

    It tracks the local-memory backing store of the selected kernel set
    -- 1.93 GiB for WSM6+MYNN, 3.94 for NSSL2 double-moment -- while the
    fit loop assumed the documented flat 4.0.  Any suite whose overhead
    pushed the reserve past that flat figure was sized against one budget
    and verified against a smaller one, so BOTH NSSL2 profiles emitted a
    config that failed their own check at EVERY card size.  The loop
    prices the reserve from the candidate now, which is the same call
    check makes.

    Since the 2026-08-03 stress finding, an ABSENT card is priced at the
    conservative measured reference intercept (170 SMs), not a per-class
    SM discount -- so NSSL2's 15,504 B frame reserves 3.78 GiB before a
    grid cell exists, and the smallest tier accurately refuses rather than
    certifying a margin the class discount invented.  The refusal must
    name the arithmetic; every larger tier must still size and pass its
    own verifier.

    WHICH tiers refuse is a platform fact and is deliberately not
    asserted.  The peak envelope carries a 1.75x WDDM floor on Windows
    and none on Linux (``PEAK_ENVELOPE_FACTORS``), so this suite at 12
    GiB is refused on one and fits with 0.35 GiB of headroom on the
    other -- both correct, and pinning the Windows answer is what made
    this parametrization a standing Linux red.  What is asserted is the
    property that does not vary: the wizard never emits a config its own
    verifier fails, and when it refuses instead it names the arithmetic.
    """
    out = tmp_path / "n.toml"
    rc = cli_main([
        "domain", "--point=35.22,-97.44", "--vram-gib", str(vram),
        "--root-dx", "12", "--hours", "2", "--source", "gfs",
        "--cycle", "2026-07-28T00", "--physics-profile", physics_profile,
        "--out", str(out)])
    if rc == 2:
        # The accurate refusal: grid-independent constants dominate, and
        # the message says which ones and why shrinking cannot help.
        assert vram == 12.0, (
            f"{physics_profile} at {vram} GiB refused; only the smallest "
            "tier may, and it must be the constants that bind")
        message = capsys.readouterr().err
        assert "grid-independent" in message
        assert "local-memory backing store" in message
        assert not out.exists(), "a refusal writes no file"
        return
    assert rc == 0, f"{physics_profile} at {vram} GiB emitted rc {rc}"

    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    interval = 10800.0
    phases = estimate_phases(exp, source="gfs",
                             forcing_interval_seconds=interval,
                             vram_gib=vram)
    # The budget the VERIFIER would use, on a card that really is this
    # size -- the number the fit loop has to have targeted.
    free_bytes = int(card_assumed_free_gib(vram) * GIB)
    budget = sizing_budget_bytes(exp, free_bytes=free_bytes, vram_gib=vram,
                                 forcing_interval_seconds=interval)
    assert phases.peak_envelope_bytes <= budget


def test_the_reserve_the_loop_targets_is_the_one_the_verifier_uses(tmp_path):
    """Two budgets in one command's output was the mechanism.

    The wizard printed "peak envelope 10.97 GiB of a 11.00 GiB budget"
    and then, four lines later, "EXCEEDS the 10.33 GiB budget by 0.64".
    One file, one invocation, two budgets.
    """
    from woof.core.preflight import (EXTERNAL_MARGIN_BYTES, ReservePolicy,
                                      card_local_memory_profile)

    rc, out = _run_wizard(tmp_path, card="16gb", ladder="12-3",
                          source="gfs", cycle="2026-07-28T00")
    assert rc == 0
    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    interval = 10800.0
    estimate = estimate_experiment(exp, forcing_interval_seconds=interval,
                                   vram_gib=16.0)
    free_bytes = int(card_assumed_free_gib(16.0) * GIB)
    fit_budget = sizing_budget_bytes(
        exp, free_bytes=free_bytes, vram_gib=16.0,
        forcing_interval_seconds=interval)
    # 2026-08-20 (task 206): the fit budget is free VRAM minus what the
    # ENVELOPE does not model, which is other processes.  It used to
    # subtract the allocation reserve, and that reserve carries the CUDA
    # context and the local-memory backing store the envelope already
    # contains -- one process charged twice for its own bytes, which is
    # what refused the smallest hrrr layout on a 10 GiB card.
    assert fit_budget == free_bytes - EXTERNAL_MARGIN_BYTES
    # The property that mattered is unchanged and now provable rather
    # than pinned: what the wizard accepts, the verifier accepts.
    verifier = ReservePolicy.n0_alloc(
        exp, profile=card_local_memory_profile(16.0),
        estimate_bytes=estimate.alloc_estimate_bytes)
    assert estimate.peak_envelope_bytes <= fit_budget
    assert estimate.alloc_estimate_bytes <= verifier.budget_bytes(free_bytes)


def test_the_price_the_loop_targets_is_the_one_the_verifier_uses(
        tmp_path, capsys):
    """The sibling of the budget test above, for the other half.

    Two BUDGETS in one command's output was the earlier mechanism. This
    is two PRICES: the wizard sized the HRRR 12/3 km ladder on a declared
    16 GiB card at a 13.73 GiB peak envelope, inside the 14.54 GiB budget
    it printed, and the `woof check` it then runs on its own output
    priced the same file at 14.72 GiB and refused it, exit 4. One
    command, one file, two prices, and a shipped door that refuses what
    it just wrote. Measured against the 2.7.5 wheel from PyPI as well as
    the branch, so it reached users.

    The mechanism was a guard that outlived its defect.
    ``estimate_experiment`` priced the legacy shortwave chunk workspace
    with ``resident_threads=0`` whenever the caller handed no device
    profile, which ``rrtmg_sw.sw_batch_column_chunk`` reads as "no
    device" and answers with the fixed no-device width. That answer
    equalled the reference profile's until c5f942ad128 (2026-09-15)
    retired the 2,048-column ceiling and made the width the device's own
    saturation width. `woof check` fills an absent profile from
    ``card_local_memory_profile``; the estimator's own legacy branch did
    not, so the two halves of one command stopped agreeing.

    THE SECOND MECHANISM, and the reason this test now reads the printed
    numbers rather than the exit code alone. An exit code of 0 says only
    that the two prices landed on the same side of the budget. They were
    still two: the loop sized the ladder at the producer's own 3,600 s
    boundary cadence and printed `peak envelope 13.74 GiB`, and the check
    read the emitted file, found no `cadence` key in the `[fetch]` table
    that producer's fetch does not take, fell back to the 21,600 s
    constant and printed `forecast needs 13.69 GiB` -- for a file whose
    own emitted namelist.wps says `interval_seconds = 3600`. The schedule
    read now answers the producer's published cadence when the file
    carries no other one, so both doors price the file at the cadence it
    will be prepared with and the command prints one number.

    Asserted as the PROPERTY, not the number, and to the BYTE: an
    estimate for a declared card is the same estimate whether or not the
    caller spells the reference profile out, and the estimate at the
    cadence the emitting loop sized against is the same estimate as at
    the cadence the verifying read returns, because those are what the
    verifier does. Two non-vacuity assertions keep the equalities from
    holding for want of a difference to find.
    """
    from woof.core.preflight import (DEFAULT_FORCING_INTERVAL_SECONDS,
                                      card_local_memory_profile,
                                      config_forcing_schedule,
                                      config_forcing_source, estimate_phases)
    from woof.core.rrtmg_sw import (SW_BATCH_COLUMN_CHUNK_NO_DEVICE,
                                     sw_batch_column_chunk)
    from woof.source_adapters import source_forcing_interval_seconds

    rc, out = _run_wizard(tmp_path, card="16gb", ladder="12-3",
                          source="hrrr", cycle="2026-07-28T05")
    printed = capsys.readouterr().out
    assert rc == 0, (
        "the wizard wrote a configuration and then exited nonzero on its "
        "own memory check, which is the defect this test is about")
    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))

    # ONE FILE, ONE PRICE, AS PRINTED. The sizing line is the loop's
    # answer and the binding-phase line is the check's answer on the file
    # the loop just wrote; a reader sees both in one command's output.
    sized = re.search(r"peak envelope (\d+\.\d+) GiB", printed)
    verified = re.search(r"BINDING PHASE: \w+ needs (\d+\.\d+) GiB", printed)
    assert sized is not None and verified is not None, printed
    assert sized.group(1) == verified.group(1), (
        "one command printed two prices for the one file it emitted: "
        f"{sized.group(1)} GiB from the sizing loop and "
        f"{verified.group(1)} GiB from the check it ran on its own "
        "output")

    # ...AND TO THE BYTE, through the two routes that produced them. The
    # emitted file carries no cadence key, so what the verifying read
    # returns for it is the whole question.
    producer = config_forcing_source(out, priced_only=False)
    emitted_cadence, retained = config_forcing_schedule(out, exp)
    sizing_cadence = source_forcing_interval_seconds(producer)

    def envelope(cadence):
        return estimate_phases(
            exp, source=producer, forcing_intervals=retained,
            ingest_forcing_interval_seconds=cadence,
            forcing_interval_seconds=cadence, vram_gib=16.0,
            profile=card_local_memory_profile(16.0)).peak_envelope_bytes

    # The envelope comparison comes BEFORE the cadence comparison, so a
    # read that answered a different cadence is caught here, as a
    # different envelope in bytes, rather than three lines earlier by an
    # assertion that would make this one unable to fail.
    assert envelope(sizing_cadence) == envelope(emitted_cadence), (
        "one configuration, two boundary cadences, two envelopes: "
        f"{envelope(sizing_cadence)} against {envelope(emitted_cadence)} "
        "bytes")
    assert emitted_cadence == sizing_cadence, (
        "the file the loop emitted reads back at a different boundary "
        f"cadence than the loop sized it at: {emitted_cadence!r} against "
        f"{sizing_cadence!r} seconds")
    assert sizing_cadence != DEFAULT_FORCING_INTERVAL_SECONDS, (
        "this producer's published cadence equals the fallback constant, "
        "so the equality above holds whatever the read returns")
    assert envelope(DEFAULT_FORCING_INTERVAL_SECONDS) != envelope(
        sizing_cadence), (
        "the envelope does not move with the boundary cadence on this "
        "configuration, so the equality above proves nothing")
    interval = 3600.0
    bare = estimate_experiment(exp, forcing_interval_seconds=interval,
                               vram_gib=16.0)
    declared = estimate_experiment(exp, forcing_interval_seconds=interval,
                                   vram_gib=16.0,
                                   profile=card_local_memory_profile(16.0))
    assert bare.workspace_bytes == declared.workspace_bytes, (
        "the radiation workspace is priced differently depending on "
        "whether the caller spelled out the profile the estimator would "
        f"have filled in: {bare.workspace_bytes} against "
        f"{declared.workspace_bytes} bytes")
    assert bare.peak_envelope_bytes == declared.peak_envelope_bytes, (
        "one configuration, one declared card, two peak envelopes: "
        f"{bare.peak_envelope_bytes} against "
        f"{declared.peak_envelope_bytes} bytes")
    nlay = exp.domains[0].run.nz + 1
    saturation = sw_batch_column_chunk(
        nlay,
        resident_threads=card_local_memory_profile(
            16.0).resident_thread_capacity)
    assert saturation != SW_BATCH_COLUMN_CHUNK_NO_DEVICE, (
        "the no-device shortwave width and the reference profile's "
        "saturation width are equal again, so the equalities above hold "
        "whatever the estimator passes and prove nothing")


def test_detailed_steps_keep_the_measured_check_and_declared_alternative(tmp_path, capsys):
    """Two documented commands, one file, one machine, opposite verdicts.

    `woof check CONFIG` returned 4 while the wizard's own printed
    `woof check CONFIG --budget-gib 12 --vram-gib 16` returned 0, because
    --budget-gib re-declared the free figure the tier had invented.  The
    bare form is the next step now, and the declared form is printed
    beside it saying what it is for.
    """
    rc, out = _run_wizard(tmp_path, "--explain", card="16gb", ladder="12", source="gfs",
                          cycle="2026-07-28T00")
    assert rc == 0
    printed = capsys.readouterr().out
    assert f"2. woof check {out}" in printed.replace("\\", "/") or (
        "2. woof check" in printed)
    assert "that measures THIS machine's free VRAM" in printed
    assert "--free-gib" in printed, "the declared form is still offered"


def test_the_minimum_layout_refusal_does_not_contradict_itself(
        tmp_path, capsys):
    """"the other 0.00 GiB (0% of the projection) is grid-independent
    calibration constants, so a smaller grid cannot help" -- if 0% is
    grid-independent then the grid is exactly what would help."""

    out = tmp_path / "tiny.toml"
    # 4 GiB, not 5: this probe needs a budget the minimum layout cannot
    # fit, and the fit boundary MOVES when pricing improves -- the RRTMGP
    # register-derivation drop and the model-top default move carried the
    # 12-3-1-0.5 ladder's envelope under a 5 GiB card's budget, which
    # left this test asserting a refusal that correctly no longer fires.
    # 4 GiB is the probe that still exercises THIS arm: at 3 GiB the
    # size-independent floor (context + kernel backing store) overflows
    # first and the no-layout-can-help refusal answers instead.
    # The suite is named, with the grid deciding its cumulus exactly as
    # the unnamed default does: this probe is the minimum-layout arm for
    # the 12 km default suite, and below 1 km the unnamed default is the
    # spacing table's heavier suite, whose kernel set alone overflows a
    # 4 GiB card (the no-layout-can-help arm, pinned in
    # tests/test_domain_wizard_memory_remedy.py).
    # 5 GiB since 97e9a3ca061 (cap-hail): automatic HAILCAST's 7,392-B
    # kernel frame lifted the grid-independent floor to 2.88 GiB, over the
    # 4 GiB card's 2.75 GiB budget, so 4 GiB now answers with the
    # no-layout-can-help arm; 5 GiB (3.75 GiB budget) sits between that
    # floor and the minimum layout's 4.34 GiB envelope.
    from woof.physics_menu import default_profile_for

    rc = cli_main(["domain", "--point=35.22,-97.44", "--vram-gib", "5",
                   "--ladder", "12-3-1-0.5", "--source", "gfs",
                   "--physics-profile", default_profile_for("gfs"),
                   "--cumulus", "grid",
                   "--cycle", "2026-07-28T00", "--out", str(out)])
    assert rc == 2
    message = capsys.readouterr().err
    assert "minimum layout" in message
    assert "there is no smaller grid on this ladder" in message
    # The self-contradiction: a share and a claim that disagree.
    share = float(message.split("% of the envelope")[0].split("(")[-1])
    if share < 25.0:
        assert "so a smaller grid cannot help" not in message
    assert not out.exists(), "a refusal writes no file"


@pytest.mark.parametrize("spelling", ["nan", "inf", "-inf"])
def test_non_finite_vram_is_refused_without_inventing_a_capacity(
        tmp_path, capsys, spelling):
    """E-10.  The finite CHECK existed; the MESSAGE did not respect it.

    It shared a sentence with the too-small-card branch, which describes
    the card it was given -- and card_assumed_free_gib launders a
    non-finite value through max(), which returns the finite operand.  So
    `--vram-gib nan` was reported as a card presenting "about 0.00 GiB
    free": a specific, false, plausible number invented for an input that
    names no quantity.
    """
    # =VALUE form: a leading "-" (-inf) must not be read as an option.
    rc, out = _run_wizard(tmp_path, f"--vram-gib={spelling}", card=None)
    assert rc == 2
    err = capsys.readouterr().err
    assert "is not a size" in err
    assert "0.00 GiB" not in err
    assert "Traceback" not in err
    assert not out.exists()


def test_help_and_gfs_emission_preserve_available_ruc_profiles(tmp_path, capsys):
    """Retired template membership must not withdraw the existing RUC owner."""
    from woof.domain_wizard import (_profile_help_route_note,
                                     profile_route_blocker,
                                     profiles_blocked_on_source)

    ruc_profiles = tuple(
        profile for profile in WIZARD_PHYSICS_PROFILES
        if single_domain_runtime_switches(profile)["sf_surface_physics"] == 3)
    assert ruc_profiles, "the menu must exercise an actual RUC selection"
    assert not set(ruc_profiles).intersection(profiles_blocked_on_source("gfs"))
    assert "--source gfs cannot prepare" not in _profile_help_route_note()

    with pytest.raises(SystemExit) as help_exit:
        cli_main(["domain", "--help"])
    assert help_exit.value.code == 0
    printed = capsys.readouterr().out
    assert "--physics-profile" in printed
    # argparse may wrap a long hyphenated profile id across lines.
    compact_help = "".join(printed.split())
    for profile in ruc_profiles:
        assert profile in compact_help
        assert profile_route_blocker(profile, "gfs") is None
        out = tmp_path / f"{profile}.toml"
        assert cli_main([
            "domain", "--point=35.3,-97.5", "--card", "24gb",
            "--ladder", "12", "--source", "gfs", "--cycle",
            "2026-07-29T18", "--hours", "6",
            "--physics-profile", profile, "--out", str(out)]) == 0
        printed = capsys.readouterr().out
        assert f"woof go {_posix(out)}" in printed.split("next:")[-1]
        exp = load_experiment(out)
        expected = single_domain_runtime_switches(profile)
        assert exp.root.run.sf_surface_physics == 3
        assert exp.root.run.num_soil_layers == expected["num_soil_layers"]
        for key, value in expected.items():
            assert getattr(exp.root.run, key) == value, (profile, key)


def test_the_help_caveat_is_derived_not_listed(monkeypatch):
    """A hard-coded pair would go stale in the direction that lies.

    If a route regains the component, the caveat has to disappear on its
    own -- otherwise `--help` starts refusing a pairing that works.
    """
    from woof import domain_wizard

    monkeypatch.setattr(domain_wizard, "profile_route_blocker",
                        lambda profile, source: None)
    assert domain_wizard.profiles_blocked_on_source("gfs") == ()
    assert domain_wizard._profile_help_route_note() == ""


def test_an_mp8_hrrr_chain_prints_the_two_exports_its_runners_demand(
        tmp_path, capsys):
    """The wizard's own generated chain carries its launch environment.

    mp8 is first-class through the library, but the runners under
    ``tools/`` kept the two-variable launch contract on purpose, and the
    printed chain said nothing about it.  A field run of the shipped
    1.5.0 wheel discovered the pair by being refused twice and then had
    to locate the table root by hand.  The exports are printed BEFORE
    preparation because preparation launches the forecast runner as a
    subprocess: one pair in one shell covers both stages.
    """
    from woof.physics_compat import (EXPERIMENTAL_THOMPSON_ENV,
                                      THOMPSON_PROFILE_ID,
                                      THOMPSON_TABLE_ROOT_ENV,
                                      thompson_guard_exports,
                                      thompson_table_root)

    out = tmp_path / "mp8.toml"
    assert cli_main([
        "domain", "--point=39.0,-98.0", "--card", "24gb",
        "--root-dx", "3", "--source", "hrrr",
        "--physics-profile", THOMPSON_PROFILE_ID,
        "--cycle", "2026-07-29T18", "--hours", "1",
        "--out", str(out)]) == 0
    block = capsys.readouterr().out.split("next:")[-1]

    from woof.domain_wizard import hrrr_route_commands
    from woof.experiment import load_experiment
    block += "\n" + hrrr_route_commands(
        out, load_experiment(out), profile=THOMPSON_PROFILE_ID, data_dir="data",
        forecast_start_hour=0)

    for line in thompson_guard_exports():
        assert line in block
    # Both variables named, and the root is a VALUE, not a placeholder:
    # the reader pastes it rather than going looking for it.
    assert EXPERIMENTAL_THOMPSON_ENV in block
    assert THOMPSON_TABLE_ROOT_ENV in block
    assert thompson_table_root() in block
    # Before the stage that reads them, not after it.
    assert block.index(EXPERIMENTAL_THOMPSON_ENV) \
        < block.index("woof prep --source hrrr")


def test_a_chain_for_a_suite_with_no_launch_guard_prints_no_exports(
        tmp_path, capsys):
    """Only the guarded suite pays for the block."""
    from woof.physics_compat import (EXPERIMENTAL_THOMPSON_ENV,
                                      THOMPSON_TABLE_ROOT_ENV)

    out = tmp_path / "wsm6.toml"
    assert cli_main([
        "domain", "--point=39.0,-98.0", "--card", "24gb",
        "--root-dx", "3", "--source", "hrrr",
        "--cycle", "2026-07-29T18", "--hours", "1",
        "--out", str(out)]) == 0
    block = capsys.readouterr().out.split("next:")[-1]
    assert "woof go " in block
    assert EXPERIMENTAL_THOMPSON_ENV not in block
    assert THOMPSON_TABLE_ROOT_ENV not in block


# ---------------------------------------------------------------------------
# Radiation cadence: a nest inherits its parent's radt, it does not refine
# it with dx.  Until 2.5.0 the wizard wrote `radt = max(1.0, dx_km)` on
# every nest, so the whole sub-km half of a ladder ran radiation once a
# simulated MINUTE -- measured at 79% of a real 1 km run's wall clock --
# and the deepest shipped ladder (12-3-1-0.5) emitted 12/3/1/1: the 500 m
# nest paid the floor and got no cadence of its own out of it either.
# Radiative transfer varies on cloud timescales, not on grid scales, and
# WRF's own guidance is one radt for the root and the same value for
# every nest.  The fix ships DEFAULT-ON: there is no flag for it.
# ---------------------------------------------------------------------------

_RADT_PROJECTION = {
    "map_proj": "lambert", "ref_lat": 35.3, "ref_lon": -97.5,
    "truelat1": 25.3, "truelat2": 45.3, "stand_lon": -97.5,
}


def _emitted_radt(ratios, profile=DEFAULT_PHYSICS_PROFILE) -> list[float]:
    """The ``radt`` column of the bytes the wizard actually writes.

    Read out of the rendered TOML rather than off ``_domain_tables``, so
    a refinement re-entering anywhere between the physics profile and
    the emitted file is caught by these tests.
    """
    from datetime import datetime

    text = render_config(
        name="radtladder", start_time=datetime(2026, 7, 29, 18), hours=6,
        projection=dict(_RADT_PROJECTION),
        dims=_dims_for_scale(1.0, ratios), ratios=ratios,
        fetch_hints={"source": "era5"}, case_data=None, profile=profile)
    return [float(table["radt"]) for table in tomllib.loads(text)["domain"]]


def test_the_deepest_shipped_ladder_emits_one_radiation_cadence():
    """12-3-1-0.5 emitted 12/3/1/1.  It emits 12/12/12/12."""
    ratios = LADDER_RATIOS["12-3-1-0.5"]
    assert _ladder_dx_km(ratios) == [12.0, 3.0, 1.0, 0.5]
    root_radt = profile_switches(DEFAULT_PHYSICS_PROFILE)["radt"]
    assert root_radt == 12.0
    assert _emitted_radt(ratios) == [12.0, 12.0, 12.0, 12.0]


def test_the_les_target_ladder_emits_one_radiation_cadence():
    """The 12-3-1-0.5-0.25 ladder: the 250 m LES target this program
    exists for, and the rung the old floor taxed hardest.

    Asserted on :func:`radt_ladder_minutes` rather than on a rendered
    file because ``--chain 4,3,2,2`` still dies in ``_dims_for_scale`` /
    ``_DIFF6_FACTORS`` at four nests (a separate 2.5.0 repair).  That
    function is the ONE the emission reads -- ``_domain_tables`` indexes
    its result -- so this pins the cadence the deep ladder will emit the
    moment the depth limit lifts, and cannot drift from it.
    """
    ratios = (4, 3, 2, 2)
    assert _ladder_dx_km(ratios) == [12.0, 3.0, 1.0, 0.5, 0.25]
    root_radt = profile_switches(DEFAULT_PHYSICS_PROFILE)["radt"]
    assert radt_ladder_minutes(root_radt, len(ratios) + 1) \
        == [12.0, 12.0, 12.0, 12.0, 12.0]


@pytest.mark.parametrize("profile", WIZARD_PHYSICS_PROFILES)
@pytest.mark.parametrize("ladder", ["12-3", "12-3-1", "12-3-1-0.5"])
def test_no_emitted_nest_departs_from_its_parents_radiation_cadence(
        profile, ladder):
    """Neither finer NOR coarser, under every shipped suite.

    The old rule broke both ways: under the 12-minute suites it refined
    3.0/1.0/1.0 out of a 12.0 root, and under the radt = 1.0 suites
    (the mp8 validation profile and the four no-radiation profiles) it
    handed the 3 km nest a 3.0 its own parent did not have -- a nest
    running radiation THREE TIMES LESS often than the domain feeding it.
    """
    root_radt = profile_switches(profile)["radt"]
    emitted = _emitted_radt(LADDER_RATIOS[ladder], profile=profile)
    assert emitted == [root_radt] * len(emitted)


def test_the_radiation_cadence_is_default_on_and_takes_no_flag():
    """"Fixed means default": the inheritance is not opt-in.

    ``domain_main``'s parser is the whole front door for the wizard, and
    a remedy reachable only through a flag is a workaround.  Nothing in
    it names radt, and the bare emission above already carries the fix.
    """
    import argparse

    from woof.domain_wizard import register_cli

    subparsers = argparse.ArgumentParser().add_subparsers()
    register_cli(subparsers)
    parser = subparsers.choices["domain"]
    options = {string for action in parser._actions
               for string in action.option_strings}
    assert not [opt for opt in options
                if "radt" in opt or "radiation" in opt]
    # ...and the bare emission -- no flags at all -- is already fixed.
    assert _emitted_radt(LADDER_RATIOS["12-3-1"]) == [12.0, 12.0, 12.0]

# ---------------------------------------------------------------------------
# The cadence rule is SPOKEN, not only applied ("fixed means default"
# ships the remedy default-on WITH an advisory) -- and LAYERED like every
# other advisory on this door, because the default screen's line cap is a
# measured gate.  Default: a clause on the physics line, zero extra
# lines.  --explain: one full advisory line naming the mechanism and the
# per-domain override in the emitted file -- the only radt input the
# wizard honours, since it takes no radt flag.
# ---------------------------------------------------------------------------

def _radt_advisories(printed: str) -> list[str]:
    return [line for line in printed.splitlines()
            if line.startswith("advisory:") and "radt" in line]


def test_a_nested_emission_speaks_the_cadence_on_the_default_screen(
        tmp_path, capsys):
    """--ladder 12-3, default suite, NO flags: the physics line itself
    says one cadence governs and that nests inherit it, without
    spending a line of the capped default screen."""
    rc, _ = _run_wizard(tmp_path, ladder="12-3")
    assert rc == 0
    printed = capsys.readouterr().out
    physics = [line for line in printed.splitlines()
               if line.startswith("physics:")]
    assert len(physics) == 1
    assert "one radiation cadence" in physics[0]
    assert "nests inherit" in physics[0]
    assert _radt_advisories(printed) == []  # the cap stays honoured


def test_explain_speaks_the_full_cadence_advisory(tmp_path, capsys):
    """--explain: exactly one advisory line, carrying the root's
    cadence, the word "inherit", and the override path."""
    rc, _ = _run_wizard(tmp_path, "--explain", ladder="12-3")
    assert rc == 0
    lines = _radt_advisories(capsys.readouterr().out)
    assert len(lines) == 1
    assert "radt 12" in lines[0]
    assert "inherit" in lines[0]
    assert "emitted config" in lines[0]


def test_the_cadence_advisory_is_silent_without_a_nest(tmp_path, capsys):
    """One domain has nothing to inherit; the physics summary line
    already names its radt, and any more would be noise."""
    rc, _ = _run_wizard(tmp_path, "--explain", ladder="12")
    assert rc == 0
    printed = capsys.readouterr().out
    assert _radt_advisories(printed) == []
    assert "nests inherit" not in printed
    assert "radt 12 min" in printed  # the summary still speaks it


@pytest.mark.parametrize("profile", WIZARD_PHYSICS_PROFILES)
def test_the_cadence_advisory_fires_for_every_offered_suite(profile):
    """Once per nested emission under EVERY suite, never for a single
    domain -- and it speaks the suite's OWN root radt, not a number.

    There is no radiation-off case to be silent for: every profile the
    wizard offers runs at least shortwave (the ``*-no-radiation-*``
    names mean longwave OFF with Dudhia shortwave still on, radt = 1),
    so radt paces real work in all of them.

    RETIRED, with the reader it duplicated: the premise used to be
    checked by a local two-line read of the switch map that fell back to
    ``ra_physics`` only when a split key was ABSENT.  The aggregate
    radiation option states the split keys as -1 and the pair in the
    combined key, so that read called a fully RTE+RRTMGP suite
    "radiation off" the first time the wizard offered one.  The premise
    is now asked of ``radiation_scheme_ids``, which is the reader every
    door uses, so the two cannot disagree again.
    """
    from woof.domain_wizard import radiation_cadence_advisory
    from woof.physics_menu import radiation_scheme_ids

    switches = profile_switches(profile)
    lw, sw = radiation_scheme_ids(switches)
    if lw == sw == 0:
        assert radiation_cadence_advisory(profile, 4) == []
        assert radiation_cadence_advisory(profile, 1) == []
        return
    notes = radiation_cadence_advisory(profile, 4)
    assert len(notes) == 1
    assert f"radt {float(switches['radt']):g}" in notes[0]
    assert radiation_cadence_advisory(profile, 1) == []


@pytest.mark.parametrize("source", ["gfs", "ifs", "aifs", "gdas"])
def test_polar_cap_full_longitude_uses_global_coverage_metadata(source):
    from woof.domain_wizard import _projection_entries, fetch_area_hint
    projection = _projection_entries(90.0, 0.0)
    area = tuple(map(float, fetch_area_hint(
        projection, 250, 200, source=source, root_dx_m=12000.0).split(',')))
    assert area[1:] == (-180.0, 90.0, 180.0)
