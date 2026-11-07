"""A point request is sized to a domain, not to whatever the card holds.

``--point`` carries a centre and no extent, so the fit has to choose one.
Until 2.7.3 the only thing that chose was memory: the scale bisection grew
the root until the envelope met the budget.  That produced two failures at
once on the door the desktop's forecast request uses.

* With streaming on -- ``--tiles auto``, the desktop's default -- memory
  never binds, because the card holds one tile and not the domain.  A
  card with about 7 GiB free sized a 12 km root of 2326 x 1860 mass
  points: 27,912 x 22,320 km, wider than the Earth's circumference at
  mid-latitudes, wrapped around the projection pole.  A mid-latitude
  point was then REFUSED at plan review for reaching a pole, which the
  reader was told to fix by drawing a smaller area -- advice that does
  not apply to a request that drew nothing.
* Resident fits reached the same place more slowly: a 180 GiB budget
  sized 26,952 km on the same ladder.

The fit now carries two bounds of its own, both properties of the REQUEST
rather than of the card, and both of which SHRINK the domain instead of
refusing it: a maximum extent (:data:`POINT_FIT_MAX_EXTENT_KM`), and the
projection's own polar envelope.

What stays a refusal is the case a smaller domain cannot fix: an area the
user OUTLINED that reaches the pole, and a point so close to one that even the
minimum layout contains it.  Those two keep their wording, which the
desktop maps to its own sentence.

And the cap is a QUIET default.  Its first version announced itself with a
``warning:`` on stderr, which is a claim that something unusual happened --
but a mid-latitude point on any card from about 16 GiB up is the ordinary
request, so the warning fired on essentially every one of them and turned
``tests/test_go_chain.py`` red on the door's own default emission.  The
bound is now stated once, as a plain fact, in the plan summary on stdout
(:func:`woof.domain_wizard.point_fit_cap_note`), where the reader is
already being told what was sized.  stderr is left for the requests that
really are unusual: the two refusals above.
"""
from __future__ import annotations

import datetime
import json

import pytest

from woof.cli import main as cli_main
from woof import domain_wizard as dw, provenance

GIB = 1024 ** 3
START = datetime.datetime(2026, 9, 9, 0, 0)

#: Latitudes a forecast is actually requested from, spanning both
#: auto-selected projections the mid-latitudes use.
MID_LATITUDES = (30.0, 41.5, 48.5, 60.0)


def _fit(lat: float, *, gib: float = 180.0, tiles: str | None = None,
         ladder: str = "12", stop: dict | None = None):
    projection = dw._projection_entries(lat, -98.0)
    dims, exp = dw.fit_ladder(
        ladder=ladder, free_bytes=int(gib * GIB), vram_gib=float(gib),
        hours=6, start_time=START, projection=projection, source="gfs",
        name="point-extent-cap", tiles=tiles, stop_out=stop)
    return projection, dims, exp


def _stderr_beyond_the_banner(err: str) -> str:
    """``err`` minus the one startup notice a healthy door writes.

    ``woof.provenance_gate.announce`` prints ``woof <door>: <banner>``
    once per process so a log names the tree that ran, and it is the
    line ``tests/test_go_chain.py`` strips for exactly the same reason
    before holding a door to an empty stderr.  Everything else is this
    file's subject.
    """

    banner = provenance.resolve().banner()
    lines = err.splitlines()
    # ... and the check door's planning-time model-top verdict, which goes
    # to stderr so `--json` stdout stays one document (8f7e4c27247).  Only
    # the PASS form is a healthy notice, as tests/test_go_chain.py reads it.
    top_pass = "woof check: model-top check PASS: source "
    for index, line in enumerate(lines):
        if line.startswith(top_pass):
            del lines[index]
            break
    for index, line in enumerate(lines):
        door, sep, text = line.partition(": ")
        if sep and text == banner and door.startswith("woof "):
            del lines[index]
            break
    return "\n".join(lines)


def _extent_km(dims) -> float:
    nx, ny = dims[0]
    return max(nx, ny) * dw.ROOT_DX_M / 1000.0


@pytest.mark.parametrize("lat", MID_LATITUDES)
@pytest.mark.parametrize("tiles", [None, "auto"])
def test_a_point_fit_is_a_valid_domain_on_a_budget_that_never_binds(
        lat: float, tiles: str | None) -> None:
    """The defect, at the four latitudes it was reported from.

    A large budget (and, separately, streaming, which removes the budget
    as a bound altogether) must still produce a domain that PASSES the
    pole guard, because the guard runs on what the fit chose.
    """
    projection, dims, exp = _fit(lat, tiles=tiles)
    # The refusal the desktop was showing, asked exactly where the door
    # asks it: after the fit, on the fitted root.
    dw._pole_clearance_refusal(projection, *dims[0], dw.ROOT_DX_M)
    assert len(exp.domains) == 1
    assert _extent_km(dims) <= dw.POINT_FIT_MAX_EXTENT_KM + 1e-6
    # A real domain, not a degenerate one shrunk to nothing by the caps.
    assert dims[0][0] >= 110 and dims[0][1] >= 88


@pytest.mark.parametrize("lat", MID_LATITUDES)
def test_streaming_does_not_unbind_the_point_fit(lat: float) -> None:
    """The reported shape: a small card, streaming on, no drawn area.

    Streaming is what removed memory as a bound, so the streamed fit on
    a small card and the resident fit on a very large one must land on
    the same request-shaped cap rather than on two different runaways.
    """
    _, streamed, _ = _fit(lat, gib=10.0, tiles="auto")
    _, resident, _ = _fit(lat, gib=180.0)
    assert streamed[0] == resident[0]
    assert _extent_km(streamed) <= dw.POINT_FIT_MAX_EXTENT_KM + 1e-6


@pytest.mark.parametrize("lat", [70.0, 80.0, -75.0])
def test_polar_domains_use_the_extent_cap_without_pole_shrinking(lat: float) -> None:
    projection, dims, _ = _fit(lat)
    dw._pole_clearance_refusal(projection, *dims[0], dw.ROOT_DX_M)
    assert _extent_km(dims) <= dw.POINT_FIT_MAX_EXTENT_KM
    assert _extent_km(dims) > 0.95 * dw.POINT_FIT_MAX_EXTENT_KM


def test_a_bound_that_bit_is_reported_and_not_silent(capsys) -> None:
    """`test_domain_wizard_budget_monotonic` states the property: a fit
    that stops growing for a reason other than memory must say so, or a
    reader cannot tell it from a comfortable fit.

    Said to the CALLER here, in ``stop_out``, and to the reader by the
    door as one line of plan summary.  Not on stderr: this is the
    ordinary request on an ordinary card, and the test below is the
    other half of the same ruling.
    """
    stop: dict = {}
    _fit(41.5, stop=stop)
    assert stop["scope"] == dw.POINT_FIT_EXTENT_SCOPE
    assert "6000 km" in stop["reason"]
    assert capsys.readouterr().err == ""


def test_the_cap_adds_nothing_to_stderr_on_an_ordinary_request(
        tmp_path, capsys) -> None:
    """The blocker the first version shipped, stated as its own test.

    ``tests/test_go_chain.py`` runs this exact emission and holds the
    door to an empty stderr; a cap that warned made the release suite
    red and told every ordinary user that the normal case was a
    problem.  Asserted here too, at the door, so the ruling is pinned in
    the file that owns the cap rather than only in the chain suite that
    noticed it.
    """
    out = tmp_path / "ordinary.toml"
    assert cli_main(["domain", "--point=35.3,-97.5", "--card", "24gb",
                     "--ladder", "12", "--source", "gfs",
                     "--cycle", "2026-07-29T18", "--hours", "6",
                     "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert _stderr_beyond_the_banner(captured.err) == "", captured.err
    # Quiet is not silent: the fact is on stdout, once.
    fact = [line for line in captured.out.splitlines()
            if line.startswith("domain: point request:")]
    assert len(fact) == 1, captured.out
    assert "extent capped at 6000 km" in fact[0]
    assert "memory allows more" in fact[0]
    assert "--polygon" in fact[0]


def test_the_extent_cap_is_default(capsys) -> None:
    """"Fixed means default": a bare fit is the fixed one.

    ``fit_ladder`` is called here with nothing but the arguments a
    point request always supplies -- no extent is passed -- and the
    result is already bounded by the default.
    """
    projection = dw._projection_entries(41.5, -98.0)
    dims, _ = dw.fit_ladder(
        ladder="12", free_bytes=180 * GIB, vram_gib=180.0, hours=6,
        start_time=START, projection=projection, source="gfs",
        name="point-extent-cap")
    capsys.readouterr()
    assert _extent_km(dims) <= dw.POINT_FIT_MAX_EXTENT_KM + 1e-6
    import inspect
    assert "cap" not in inspect.signature(dw.fit_ladder).parameters


def test_near_pole_point_uses_polar_route(tmp_path, capsys) -> None:
    out = tmp_path / "near-pole.toml"
    rc = cli_main(["domain", "--point=89.0,-100.0", "--card", "16gb",
                   "--ladder", "12", "--source", "gfs",
                   "--cycle", "2026-07-28T06", "--out", str(out)])
    assert rc == 0, capsys.readouterr().err
    assert out.exists()


def test_drawn_area_reaching_pole_uses_polar_route(tmp_path, capsys) -> None:
    area = tmp_path / "area.geojson"
    area.write_text(json.dumps({"type": "Polygon", "coordinates": [[
        [-150.0, 80.0], [150.0, 80.0], [150.0, 89.9],
        [-150.0, 89.9], [-150.0, 80.0]]]}))
    out = tmp_path / "drawn.toml"
    rc = cli_main(["domain", f"--polygon={area}", "--card", "16gb",
                   "--ladder", "12", "--source", "gfs",
                   "--cycle", "2026-07-28T06", "--out", str(out)])
    assert rc == 0, capsys.readouterr().err
    assert out.exists()


# ---------------------------------------------------------------------------
# The sweep the cap owes its own advisories: a bound that makes a flag
# inert takes that flag out of the sentence that recommends it.
# ---------------------------------------------------------------------------

#: The desktop's forecast request, verbatim, minus the parts that vary:
#: no drawn area, a 12 km root, streaming on.
DESKTOP_SHAPE = ["--root-dx=12", "--tiles=auto", "--source", "gfs",
                 "--cycle", "2026-09-09T00", "--hours", "6"]


def _door(tmp_path, capsys, lat: float, gib: float = 10.0):
    """Run the door the way the desktop runs it; return (stdout, stderr)."""

    out = tmp_path / f"pt{lat}.toml"
    rc = cli_main(["domain", f"--point={lat},-84", *DESKTOP_SHAPE,
                   "--vram-gib", f"{gib:g}", "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    return captured.out, captured.err


def _advisory_line(stdout: str) -> str:
    lines = [line for line in stdout.splitlines()
             if line.startswith("advisory: this domain is much wider")]
    assert len(lines) == 1, stdout
    return lines[0]


@pytest.mark.parametrize("lat", MID_LATITUDES)
def test_the_advisory_stops_naming_the_lever_the_cap_made_inert(
        lat: float, tmp_path, capsys) -> None:
    """The contradiction the cap created, at every latitude it created it.

    Capping the point fit made the oversized-footprint advisory fire on
    EVERY mid-latitude point request -- a 6,000 x 4,800 km conformal
    root fans out to a 108 to 360 degree fetch box -- while that
    advisory still said "pass --vram-gib N ... for a smaller first run".
    Two lines above it the fit says the opposite in as many words ("a
    larger card buys no more grid here"), and the flag is inert in fact:
    with streaming on the card holds one tile, so memory never binds and
    every budget above the refusal floor emits the identical grid.

    The advisory's own docstring records this being fixed once, for the
    source bound.  The request bounds get the same sweep.
    """

    stdout, stderr = _door(tmp_path, capsys, lat)
    line = _advisory_line(stdout)
    assert ("domain: point request: extent capped at 6000 km "
            "(500x400 at 12 km)") in stdout
    assert "not the card" not in stderr
    # The lever that still moves the answer, and no lever that does not.
    assert "--polygon" in line
    assert "--vram-gib" not in line and "--root-dx" not in line
    # It names the same bound the fit named, rather than a second story.
    assert "stopped on the REQUESTED EXTENT, not on the card" in line
    # And it still says the thing it exists to say.
    assert "degree --area box" in line and "starve" in line


def test_the_card_shaped_remedy_survives_where_the_card_still_decides(
        ) -> None:
    """Not a blanket rewrite: the old sentence is still the right one.

    A drawn area was sized to the drawing and a point fit that memory
    or the source stopped short of its own bounds is still a fit a
    smaller budget shrinks, so both keep the remedy -- byte for byte,
    because a wheel user's report is what worded it.
    """

    box = "-6.39,-159.63,73.19,-35.37"
    unbounded = dw.oversized_footprint_advisory(box)
    assert unbounded == dw.oversized_footprint_advisory(
        box, request_bound=None)
    assert len(unbounded) == 1
    assert "--vram-gib N (or a finer --root-dx KM)" in unbounded[0]
    # And a box nobody needs to hear about stays silent on both routes.
    assert dw.oversized_footprint_advisory(
        "6.24,-135.55,63.02,-59.45",
        request_bound="REQUESTED EXTENT") == []


@pytest.mark.parametrize("lat,gib", [(30.0, 10.0), (41.5, 180.0),
                                     (60.0, 6.0), (70.0, 10.0),
                                     (80.0, 10.0), (-75.0, 10.0)])
def test_the_fit_and_the_advisory_read_one_bound(lat: float, gib: float,
                                                 capsys) -> None:
    """The two callers cannot disagree, which is the actual fix.

    The advisory is printed by a different function, from a different
    scope, well after the fit returned; the only thing keeping its
    sentence true is that it is handed the bound the SEARCH stopped on
    rather than one re-derived from the emitted root.  This asserts the
    agreement directly, including at the latitudes where the answer is
    "no request bound was in force" (70 N on a 10 GiB streamed budget
    stops on memory at 412 x 328) -- where a false positive would take
    a working lever out of the sentence.
    """

    stop: dict = {}
    _, dims, _ = _fit(lat, gib=gib, tiles="auto", stop=stop)
    assert capsys.readouterr().err == ""
    scope = stop.get("scope")
    assert scope is None or scope in dw.POINT_FIT_SCOPES
    if scope is not None:
        # The fact the door prints and the bound the search stopped on
        # are one sentence built from one answer.
        note = dw.point_fit_cap_note(scope, dims, dw.ROOT_DX_M)
        assert note.startswith("point request: extent capped at ")
        assert "memory allows more" in note


def test_the_bound_is_read_from_the_search_not_from_a_repriced_neighbour(
        ) -> None:
    """The misattribution the review found, pinned both ways.

    The first version answered "which bound is this emitted root sitting
    against?" by re-pricing the root one grid step larger per axis and
    asking the bounds about THAT.  A layout the request bounds never
    touched -- memory stopped it exactly one discretisation step under
    the cap -- prices over the cap as its own neighbour and is then
    reported as cap-bound, which takes ``--vram-gib`` out of the
    advisory on a fit where the card is still the lever.

    The two halves below are the whole argument: the neighbour of a root
    exactly at the cap crosses it (so the neighbour cannot tell the two
    cases apart), and the fit reports from its own search instead --
    empty when memory stopped it, named when a bound did.
    """

    projection = dw._projection_entries(41.5, -98.0)
    dx_km = dw.ROOT_DX_M / 1000.0
    at_cap = int(round(dw.POINT_FIT_MAX_EXTENT_KM / dx_km))
    assert at_cap % 2 == 0
    # A root exactly at the cap is inside every request bound ...
    assert dw.point_request_bound(projection, at_cap, at_cap * 4 // 5,
                                  dw.ROOT_DX_M) is None
    # ... and its neighbour one grid step out is not.
    neighbour = dw.point_request_bound(projection, at_cap + 2,
                                       at_cap * 4 // 5 + 2, dw.ROOT_DX_M)
    assert neighbour is not None
    assert neighbour[0] == dw.POINT_FIT_EXTENT_SCOPE

    # The fit answers from the search that stopped, not from a
    # neighbour: named when a request bound bound ...
    bound: dict = {}
    _fit(41.5, gib=180.0, stop=bound)
    assert bound["scope"] == dw.POINT_FIT_EXTENT_SCOPE
    # ... and EMPTY when memory did, on a resident fit small enough that
    # the card is still the lever.
    memory: dict = {}
    _, small, _ = _fit(41.5, gib=8.0, tiles="off", stop=memory)
    assert memory == {}
    assert max(small[0]) * dx_km < dw.POINT_FIT_MAX_EXTENT_KM


def test_a_capped_fit_still_says_its_forcing_spans_the_whole_band(
        tmp_path, capsys) -> None:
    """A consequence of the cap, said out loud rather than discovered.

    A 4,800 km tall conformal domain centred at 60 N reaches into the
    high Arctic, where its lat/lon bounding box is every longitude:
    the emitted fetch is the source's full band, which on 2.7.2 was a
    request that got refused instead.  It is a download the reader is
    entitled to hear about before `woof go` starts it, so this pins
    that both sentences are spoken at plan review -- the band warning
    and the advisory that names the flag for a smaller one.
    """

    stdout, stderr = _door(tmp_path, capsys, 60.0)
    assert "full longitude band (-180..180)" in stderr
    assert "only forcing coverage is expanded" in stderr
    line = _advisory_line(stdout)
    assert "360 x" in line and "--polygon" in line


def test_the_template_point_route_is_capped_by_the_same_bound(
        tmp_path) -> None:
    """`woof domain-fit --point` is the other point sizer.

    :func:`fit_ladder` is not reached only from ``woof domain``: the
    starter-template route hands it the template's own dimensions
    builder and the template's own root dx, and it is the same
    bisection with the same missing bound, so a template sized on a
    large card ran away in exactly the same way.  It is capped because
    the bound lives in the fit rather than in the door -- and nothing
    pinned that, so this does.
    """

    from argparse import Namespace
    import tomllib
    from woof import starter_template as st

    dx_m = 12125.125
    text = dw.render_config(
        name="template-point-cap", start_time=START, hours=6,
        projection=dw._projection_entries(41.5, -98.0, "auto"),
        dims=dw._dims_for_scale(1, ()), ratios=(), root_dx_m=dx_m,
        fetch_hints=dict(source="gfs", cycle="2026-09-09T00", hours=6,
                         out="data/test", cadence=3), case_data=None)
    raw = tomllib.loads(text)
    raw["domain"][0]["dx"] = dx_m
    path = tmp_path / "starter.toml"
    path.write_text(st.render_tables(raw), encoding="utf-8")

    out = tmp_path / "resolved.toml"
    assert st.fit_main(Namespace(
        template=path, out=out, point="41.5,-98", polygon=None,
        buffer_km=None, source=None, card=None, vram_gib=180.0,
        start_time=None, hours=None, write=True)) == 0
    fitted = tomllib.loads(out.read_text(encoding="utf-8"))["domain"][0]
    dx_km = float(fitted["dx"]) / 1000.0
    extent_km = max(fitted["nx"], fitted["ny"]) * dx_km
    assert extent_km <= dw.POINT_FIT_MAX_EXTENT_KM + 1e-6
    # Bound, not merely small: the template's own dx is not 12 km, so a
    # cap read off the wrong number would pass this by accident.
    assert extent_km > dw.POINT_FIT_MAX_EXTENT_KM - 2.0 * dx_km


#: 2.7.3 gives :func:`fit_ladder` a SECOND road: `candidate_scales` walks
#: an authored ladder of rungs largest-first instead of bisecting a
#: continuous scale, because tiling and host admission are not monotonic
#: in grid size and a bisection assumes they are.  The cyclone door is its
#: first caller, and that door is a `--point` door.
#:
#: So the bound above is not a property of the bisection; it is a property
#: of the REQUEST, and both roads have to carry it.  Without these the
#: same door would have answered the same question two ways: the bounded
#: road growing a high-latitude request into the projection pole on a
#: large card while the bisecting road shrank away from it.
#: 8.0 down to 0.2 in 0.2 steps -- 40 rungs, the same bracket the
#: bisection is allowed to reach.  The top rung sizes a 880 x 704 root at
#: 12 km, 10,560 km across, so a road that ignored the request bounds
#: would return a domain nearly twice the cap.
_BOUNDED_LADDER = tuple(step / 5 for step in range(40, 0, -1))


def _bounded_fit(lat: float, *, gib: float = 180.0, tiles: str | None = None,
                 stop: dict | None = None):
    projection = dw._projection_entries(lat, -98.0)
    dims, exp = dw.fit_ladder(
        ladder="12", free_bytes=int(gib * GIB), vram_gib=float(gib),
        hours=6, start_time=START, projection=projection, source="gfs",
        name="bounded-point-cap", tiles=tiles, stop_out=stop,
        candidate_scales=_BOUNDED_LADDER)
    return projection, dims, exp


@pytest.mark.parametrize("lat", MID_LATITUDES)
def test_the_bounded_search_is_capped_by_the_same_extent_bound(lat) -> None:
    """The rung ladder reaches past the cap; the fit does not."""

    _projection, dims, _exp = _bounded_fit(lat)
    extent_km = max(dims[0]) * 12.0
    assert extent_km <= dw.POINT_FIT_MAX_EXTENT_KM + 1e-6
    # Bound, not merely small: the ladder's largest rung is well past the
    # cap, so a road that ignored the bound would have returned it.
    largest = dw._dims_for_scale(max(_BOUNDED_LADDER), dw.LADDER_RATIOS["12"])
    assert max(largest[0]) * 12.0 > dw.POINT_FIT_MAX_EXTENT_KM


@pytest.mark.parametrize("lat", MID_LATITUDES)
def test_the_bounded_search_hands_back_the_bound_that_stopped_it(lat) -> None:
    """One `stop_out` contract, so one plan-summary sentence.

    The note the door prints is built from what the SEARCH says stopped
    it, never re-derived from the emitted root -- a re-derivation off a
    neighbour misattributes a fit that stopped one discretisation step
    under a cap.  A road that filled nothing would print nothing and the
    reader would see a grid stop for no stated reason.
    """

    stop: dict = {}
    _projection, dims, _exp = _bounded_fit(lat, stop=stop)
    assert stop["scope"] in dw.POINT_FIT_SCOPES
    note = dw.point_fit_cap_note(stop["scope"], dims)
    assert "point request: extent capped at" in note
    assert not note.lower().startswith("warning")


def test_the_bounded_search_reports_no_bound_when_memory_stopped_it() -> None:
    """A memory stop leaves `stop_out` empty on this road too.

    Memory is visible from the outside -- the sizing line prints the
    envelope against the budget -- so claiming a cap that did not bind
    would state a fact that is not one.
    """

    stop: dict = {}
    _projection, dims, _exp = _bounded_fit(41.5, gib=8.0, stop=stop)
    assert max(dims[0]) * 12.0 < dw.POINT_FIT_MAX_EXTENT_KM
    assert stop == {}


def test_both_fit_roads_admit_polar_cap() -> None:
    projection, dims, _ = _bounded_fit(72.0, tiles="auto")
    assert dw._footprint_contains_pole(projection, *dims[0], dw.ROOT_DX_M)
    bisected_projection, bisected_dims, _ = _fit(72.0, tiles="auto")
    assert dw._footprint_contains_pole(bisected_projection, *bisected_dims[0], dw.ROOT_DX_M)


def _road_fit(*, bounded: bool, source: str = "gfs", lat: float = 41.5,
              gib: float = 180.0, stop: dict | None = None,
              scales=None):
    projection = dw._projection_entries(lat, -98.0)
    extra = {"candidate_scales": scales or _BOUNDED_LADDER} if bounded else {}
    dims, exp = dw.fit_ladder(
        ladder="12", free_bytes=int(gib * GIB), vram_gib=float(gib),
        hours=6, start_time=START, projection=projection, source=source,
        name="source-stop", stop_out=stop, **extra)
    return projection, dims, exp


@pytest.mark.parametrize("bounded", [False, True])
def test_a_source_stop_is_warned_about_on_both_roads(capsys, bounded) -> None:
    """The reconciliation, in the other direction.

    A source-shaped ceiling stops the search below what the card affords,
    and from the outside that is indistinguishable from a comfortable fit
    -- the sizing line prints an envelope well under budget.  The
    bisection warned about it; the bounded road filled the same
    ``stop_out`` and said nothing on stderr, so a bounded-road caller
    stopped by a coverage window would have been told nothing at all.
    """

    stop: dict = {}
    _projection, dims, _exp = _road_fit(bounded=bounded, source="hrrr",
                                        stop=stop)
    err = _stderr_beyond_the_banner(capsys.readouterr().err)
    assert stop["scope"] == "SOURCE"
    assert f"stopped at {dims[0][0]}x{dims[0][1]} on the SOURCE" in err
    assert "a larger card buys no more grid here" in err


def test_bounded_ladder_admits_near_pole_point() -> None:
    projection, dims, _ = _road_fit(bounded=True, lat=89.0)
    assert dw._footprint_contains_pole(projection, *dims[0], dw.ROOT_DX_M)


def test_a_memory_exhausted_ladder_claims_no_request_bound() -> None:
    """The carried bound is cleared by a memory rejection, so a refusal
    that memory caused never names a bound that did not bind."""

    with pytest.raises(dw.DomainFitError) as caught:
        _road_fit(bounded=True, gib=0.5)
    assert "the centre is what moves" not in str(caught.value)
    assert caught.value.resource in {"vram", "host", "memory"}


# ---------------------------------------------------------------------------
# The extent is the request's, so a front end can pass it.
#
# 6,000 km was only ever a default: a point carries no extent, and the
# number stands in for one.  It had no door, so a front end asking for a
# wider domain from a point had to draw a polygon instead.  The flag moves
# the default and nothing else: the projection pole, the source's coverage
# and the card still bound the fit.
# ---------------------------------------------------------------------------


def _root_extent_km(out) -> float:
    exp = dw.experiment_from_text(out.read_text(encoding="utf-8"),
                                  source=str(out))
    root = exp.domains[0].run
    return max(root.nx, root.ny) * root.dx / 1000.0


def test_the_point_extent_is_an_argument_the_fit_sizes_to(
        tmp_path, capsys) -> None:
    out = tmp_path / "wide.toml"
    assert cli_main(["domain", "--point=30,-98", "--vram-gib", "180",
                     "--ladder", "12", "--source", "gfs",
                     "--cycle", "2026-07-29T18", "--hours", "6",
                     "--point-extent-km", "9000", "--out", str(out)]) == 0
    captured = capsys.readouterr()
    extent = _root_extent_km(out)
    assert dw.POINT_FIT_MAX_EXTENT_KM < extent <= 9000.0 + 1e-6
    # The plan summary states the extent used and the limit it was sized
    # under, on stdout, and the cap that bound it once.
    header = [line for line in captured.out.splitlines()
              if line.startswith("woof domain: ")]
    assert f"root extent {extent:.0f} km (--point-extent-km 9000)" in (
        header[-1]), captured.out
    fact = [line for line in captured.out.splitlines()
            if line.startswith("domain: point request:")]
    assert len(fact) == 1, captured.out
    assert f"extent capped at {extent:.0f} km" in fact[0]
    assert "point request:" not in captured.err


def test_the_template_point_route_takes_the_same_extent(tmp_path,
                                                         capsys) -> None:
    from argparse import Namespace
    import tomllib
    from woof import starter_template as st

    dx_m = 12000.0
    text = dw.render_config(
        name="template-point-extent", start_time=START, hours=6,
        projection=dw._projection_entries(30.0, -98.0, "auto"),
        dims=dw._dims_for_scale(1, ()), ratios=(), root_dx_m=dx_m,
        fetch_hints=dict(source="gfs", cycle="2026-09-09T00", hours=6,
                         out="data/test", cadence=3), case_data=None)
    raw = tomllib.loads(text)
    path = tmp_path / "starter.toml"
    path.write_text(st.render_tables(raw), encoding="utf-8")
    out = tmp_path / "resolved.toml"
    assert st.fit_main(Namespace(
        template=path, out=out, point="30,-98", polygon=None,
        buffer_km=None, source=None, card=None, vram_gib=180.0,
        start_time=None, hours=None, write=True,
        point_extent_km=9000.0)) == 0
    printed = capsys.readouterr().out
    fitted = tomllib.loads(out.read_text(encoding="utf-8"))["domain"][0]
    extent = max(fitted["nx"], fitted["ny"]) * float(fitted["dx"]) / 1000.0
    assert dw.POINT_FIT_MAX_EXTENT_KM < extent <= 9000.0 + 1e-6
    assert "point request:" in printed


def test_the_point_extent_takes_the_door_on_the_command_line() -> None:
    from woof.cli import build_parser

    parser = build_parser()
    for command in (["domain", "--point=30,-98", "--cycle", "2026-07-29T18",
                     "--out", "x.toml"],
                    ["domain-fit", "t.toml", "--point=30,-98",
                     "--out", "x.toml"]):
        assert parser.parse_args(command).point_extent_km == (
            dw.POINT_FIT_MAX_EXTENT_KM)
        assert parser.parse_args(
            [*command, "--point-extent-km", "7500"]).point_extent_km == 7500.0
        for bad in ("0", "-5", "nan", "wide"):
            with pytest.raises(SystemExit):
                parser.parse_args([*command, "--point-extent-km", bad])


def test_a_point_extent_beside_a_drawn_area_is_refused(tmp_path,
                                                        capsys) -> None:
    """A polygon is sized to the drawing and never reads the extent, so
    a value given with it would be dropped; it is refused instead."""

    area = tmp_path / "area.geojson"
    area.write_text(json.dumps({
        "type": "Polygon",
        "coordinates": [[[-100.0, 30.0], [-96.0, 30.0], [-96.0, 33.0],
                         [-100.0, 33.0], [-100.0, 30.0]]]}), encoding="utf-8")
    out = tmp_path / "drawn.toml"
    rc = cli_main(["domain", f"--polygon={area}", "--card", "16gb",
                   "--source", "gfs", "--cycle", "2026-07-28T06",
                   "--point-extent-km", "9000", "--out", str(out)])
    assert rc == 2
    assert "--point-extent-km" in capsys.readouterr().err
    assert not out.exists()


# ---------------------------------------------------------------------------
# What the extent cannot move, and what it cannot shrink.
#
# Once the extent became an argument two things the fixed 6,000 km had
# hidden were reachable.  An extent below the smallest root a ladder
# hosts was quietly exceeded under a line saying the cap bound it, and a
# Mercator point fit had nothing to stop it wrapping round the globe.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ladder,extent", [("12", 50.0), ("12-3", 500.0)])
def test_an_extent_below_the_smallest_root_gets_that_root_and_says_so(
        ladder: str, extent: float) -> None:
    stop: dict = {}
    projection = dw._projection_entries(39.0, -98.0)
    dims, _ = dw.fit_ladder(
        ladder=ladder, free_bytes=int(24 * GIB), vram_gib=24.0, hours=6,
        start_time=START, projection=projection, source="gfs",
        name="floor", stop_out=stop, point_extent_km=extent)
    smallest = dw._dims_for_scale(dw._MIN_SCALE, dw.LADDER_RATIOS[ladder])
    assert dims[0] == smallest[0]
    assert _extent_km(dims) > extent
    assert stop["scope"] == dw.POINT_FIT_FLOOR_SCOPE
    assert stop["scope"] in dw.POINT_FIT_SCOPES
    note = dw.point_fit_cap_note(stop["scope"], dims,
                                 point_extent_km=extent)
    assert "is the smallest root this ladder hosts" in note
    assert f"larger than --point-extent-km {extent:g}" in note
    assert "capped" not in note and "raise" not in note


def test_the_door_states_the_floor_rather_than_a_cap(tmp_path,
                                                     capsys) -> None:
    """The reported shape: 50 km on the 12 km ladder emitted a 720 km
    root and told the reader to raise the value for a larger domain."""

    out = tmp_path / "small.toml"
    assert cli_main(["domain", "--point=39,-98", "--source", "gfs",
                     "--cycle", "2026-07-29T18", "--hours", "6",
                     "--vram-gib", "24", "--ladder", "12",
                     "--point-extent-km", "50", "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert _root_extent_km(out) == 720.0
    header = [line for line in captured.out.splitlines()
              if line.startswith("woof domain: ")]
    assert "root extent 720 km (--point-extent-km 50)" in header[-1]
    fact = [line for line in captured.out.splitlines()
            if line.startswith("domain: point request:")]
    assert len(fact) == 1, captured.out
    assert ("is the smallest root this ladder hosts, larger than "
            "--point-extent-km 50") in fact[0]
    assert "capped" not in captured.out
    assert "raise --point-extent-km" not in captured.out
    assert "point request:" not in captured.err


def test_the_longitude_span_is_measured_without_wrapping() -> None:
    from woof.static.projection import (EARTH_RADIUS_M,
                                         footprint_longitude_span)
    import math

    merc = dw._projection_entries(0.0, -30.0)
    assert merc["map_proj"] == "mercator"
    # Mercator longitude is linear in x: nx cells of dx at the equator.
    expected = math.degrees(5000 * 12000.0 / EARTH_RADIUS_M)
    assert expected > 500.0
    assert footprint_longitude_span(merc, 5000, 4000, 12000.0) == (
        pytest.approx(expected, rel=1e-9))
    # A footprint that does not wrap measures what its full corner grid
    # measures: the perimeter carries the extremes.
    lam = dw._projection_entries(41.5, -98.0)
    _, lon_c = dw._root_grid(lam, 500, 400, 12000.0).latlon_c()
    assert footprint_longitude_span(lam, 500, 400, 12000.0) == (
        pytest.approx(float(lon_c.max() - lon_c.min()), abs=1e-6))


def test_a_root_past_one_trip_around_the_globe_is_a_request_bound() -> None:
    merc = dw._projection_entries(0.0, -30.0)
    scope, reason = dw.point_request_bound(merc, 5000, 4000, 12000.0,
                                           max_extent_km=1.0e6)
    assert scope == dw.POINT_FIT_BAND_SCOPE
    assert "more than once around the globe" in reason
    assert dw.point_request_bound(merc, 3000, 2400, 12000.0,
                                  max_extent_km=1.0e6) is None


@pytest.mark.parametrize("lat,projection_name", [(0.0, "mercator"),
                                                  (26.0, "lambert")])
def test_a_point_fit_stays_inside_one_trip_around_the_globe(
        lat: float, projection_name: str) -> None:
    """The reported shape: --point-extent-km 60000 at (0, -30) on a 2000
    GiB budget sized a 5000 x 4000 Mercator root and printed PASS.  The
    Lambert cone at 26 N crosses the band before its pole margin, so the
    same bound is what holds it."""

    from woof.static.projection import footprint_longitude_span

    stop: dict = {}
    projection = dw._projection_entries(lat, -30.0)
    assert projection["map_proj"] == projection_name
    dims, _ = dw.fit_ladder(
        ladder="12", free_bytes=int(2000 * GIB), vram_gib=2000.0, hours=6,
        start_time=START, projection=projection, source="gfs",
        name="band", stop_out=stop, point_extent_km=60000.0)
    span = footprint_longitude_span(projection, *dims[0], dw.ROOT_DX_M)
    assert 300.0 < span < 360.0
    assert stop["scope"] == dw.POINT_FIT_BAND_SCOPE
    dw._pole_clearance_refusal(projection, *dims[0], dw.ROOT_DX_M)
    note = dw.point_fit_cap_note(stop["scope"], dims)
    assert "less than once around the globe" in note


def test_the_advisory_names_the_extent_flag_and_the_floor_names_root_dx(
        ) -> None:
    box = "-6.39,-159.63,73.19,-35.37"
    for bound in (dw.POINT_FIT_EXTENT_SCOPE, dw.POINT_FIT_PROJECTION_SCOPE,
                  dw.POINT_FIT_BAND_SCOPE):
        (line,) = dw.oversized_footprint_advisory(box, request_bound=bound)
        assert "lower --point-extent-km" in line and "--polygon" in line
        assert "--vram-gib" not in line and "--root-dx" not in line
    (floor,) = dw.oversized_footprint_advisory(
        box, request_bound=dw.POINT_FIT_FLOOR_SCOPE)
    assert "a finer --root-dx KM" in floor
    assert "no --point-extent-km makes it smaller" in floor
