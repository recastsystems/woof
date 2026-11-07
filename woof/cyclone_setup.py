"""Author a source-selected 12/3 km moving-nest setup from an explicit center.

The map request reads only the named source's published f000 MSLP and wind
fields. Configuration creation reuses the ordinary domain author, physics
suite and vortex tracker; neither entry point starts a forecast or computes a
new tracking algorithm.

WHICH SOURCE IS TABLE WORK, NOT A BRANCH HERE.  Every source-derived fact on
the emitted document -- the cycle grid, the forcing interval that prices the
tree road and stamps the WPS namelist, the coverage window, the member
grammar, the physics profile, the preparation recipe -- is read from that
source's own registry row through :mod:`woof.cyclone_sources`, which is the
one place the planable rows and the acquisition routes are intersected.
Adding a model to this door is adding its row, and this file has no name of
any model in it.
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import sys
import textwrap
import tomllib

#: v2, and the bump is the point.  v1 documents were a map or a
#: configuration; a v2 result can also be a PROPOSAL -- `kind` gains
#: "proposal", the result gains `fitting` and `created`, and `--out` on
#: a proposal exits 0 having written nothing until `--accept-fit` names
#: the reviewed fit.  A caller that reads exit 0 plus `--out` as "the
#: file is there" is correct under v1 and wrong under v2, and the
#: version string is the only part of the document such a caller is
#: guaranteed to look at.  Left at v1 the change would have been silent.
#:
#: `kind` gained a THIRD value, "sources", and that one is additive: it is
#: the source menu ``--list-sources`` emits, it moves no existing key and
#: changes no existing meaning, so it is a kind a v2 reader may not know
#: rather than a document a v2 reader would misread.  The same is true of
#: `member`, `forcing_interval_seconds` and `seed` on a map or a
#: configuration: new keys beside the old ones, with the old ones meaning
#: exactly what they meant.  That is why the version does not move again.
SCHEMA = "arwen.cyclone-setup.v2"
#: The forcing source ``--source`` binds when none is named.  ONE
#: constant, read by the flag, by every keyword default in this module and
#: by the door's help, because a default spelled at each site is a default
#: that drifts.  It names the source this door shipped with, so a caller
#: that never learned the flag reads the document it already read; every
#: other source is the same registry row reached by naming it.
DEFAULT_SOURCE = "gfs"
ROOT_DIMS = (200, 160)
CHILD_DIMS = (160, 160)
ROOT_DX_M = 12000.0
RATIO = 4
#: The grid this door's one following nest is, in every file it emits.
FOLLOWER_GRID_ID = 2


def _default_profile(forcing_source: str, nest_km: float = 3.) -> str | None:
    """The suite this setup binds with none named, at its finest grid.

    The row `woof domain` binds
    (:func:`woof.domain_wizard.resolved_physics_profile`), read at the
    following nest's spacing for the two domains every setup authors, so
    a nest that ever reaches below 1 km binds the sub-km suite as every
    other door does.
    """

    from woof import domain_wizard as dw

    return dw.resolved_physics_profile(
        forcing_source, None, finest_dx_m=nest_km * 1000., domains=2)


#: The preset nest measured in the unit a nest is actually sized in: whole
#: PARENT cells.  A following nest is square and registers on whole parent
#: cells, so every size this door can choose is ``RATIO`` times an even
#: number of them, and this is the floor a budget grows up from.
PRESET_NEST_PARENT_CELLS = CHILD_DIMS[0] // RATIO
# Five-percent, aspect-preserving rungs. Both child axes stay divisible by
# 2*RATIO, so the shared author can center each nest on whole parent cells.
FIT_SCALE_STEPS = 20


def _config_name(adapter, name: str | None, nest_km: float = 3.) -> str:
    """The configuration name, resolved ONCE for both authoring seams.

    ``configuration_text`` resolved it and the fit ladder was handed the
    raw ``None`` beside it.  That is harmless only while this door always
    supplies ``candidate_builder``, because that is the branch which keeps
    ``fit_ladder`` from rendering a configuration of its own; a caller
    that dropped the builder would have written ``name = None`` into a
    configuration.  One resolver, both call sites.
    """
    return name or f"{adapter.display_title} cyclone {nest_km*RATIO:g} km to {nest_km:g} km"


def _cycle(raw: str, *, latest: bool = False,
           forcing_source: str = DEFAULT_SOURCE) -> datetime:
    """The named source's own cycle grid, never a hardcoded 00/06/12/18."""
    from woof.cyclone_sources import resolve_cycle
    return resolve_cycle(raw, source=forcing_source, latest=latest)


def _start_hour(value, *, source: str, moment: datetime, hours: int = 0) -> int:
    """The lead this setup begins at, checked by the SOURCE's own contract."""
    from woof.cyclone_sources import resolve_start_hour
    return resolve_start_hour(value, source=source, moment=moment, hours=hours)


def latest_map(cycle: str = "latest", *, source: str = DEFAULT_SOURCE,
               member: str | None = None, start_hour: int = 0) -> dict:
    """The selection map for the lead the run will START at.

    THE MAP AND THE RUN SHOW THE SAME MOMENT.  The centre is clicked on
    this picture and the configuration is initialised from the field the
    picture was drawn from, so a preview pinned to f000 while the run
    began at f186 asked the reader to place a storm using a map that does
    not contain it.  ``forecast_hour`` was already on the request; what
    was missing was a way to ask for anything but zero.
    """
    from woof.cyclone_sources import selected_member, source_adapter
    adapter = source_adapter(source)
    source = adapter.source_id
    selection = selected_member(source, member)
    map_member = 0
    if selection is not None:
        from woof.forcing_member import member_contract
        contract = member_contract(source, selection)
        if contract is not None:
            map_member = contract[1].member(selection).ordinal
    moment = _cycle(cycle, latest=True, forcing_source=source)
    start_hour = _start_hour(start_hour, source=source, moment=moment)
    # A regional source's map is its own grid, clipped to the drawable
    # band: showing a global frame for a window that stops at 60 N invites
    # a click the configuration door then has to refuse.
    bounds = [-85., -180., 85., 180.]
    if adapter.coverage_window is not None:
        south, west, north, east = adapter.coverage_window.envelope()
        bounds = [max(-85., south), west, min(85., north), east]
    return {
        "schema": SCHEMA, "kind": "map", "cycle": moment.strftime("%Y%m%d%H"),
        "source": source, "member": selection,
        "forecast_start_hour": start_hour,
        "valid_time": (moment + timedelta(hours=start_hour)).isoformat(sep=" "),
        "map_request": {"source": source, "date": moment.strftime("%Y-%m-%d"),
                        "hour": moment.hour, "forecast_hour": start_hour,
                        "member": map_member,
                        "product": "mslp_10m_winds", "bounds": bounds},
        "forecast_started": False,
        "selection": ("Click the circulation center on this exact "
                      f"{adapter.display_title} f{start_hour:03d} "
                      "pressure-and-wind map."),
    }


def follow_table_for_nest(child_nx: int, child_ny: int) -> dict:
    """Derive proposal reach from this nest's size and overlap floor.

    The single-axis limit is N * (1 - floor). The runner clamps each
    actual vector to the floor, so a diagonal proposal can move less
    without imposing that smaller limit on west/east motion. Both the
    emitted configuration and receipt use this table. Search margin
    grows with the nest; the floor and cooldown remain selectable.
    """
    from woof.companion_domains import VORTEX_PRESET
    from woof.core.nest_relocation import max_parent_cells_for_overlap

    table = dict(VORTEX_PRESET)
    bound = max_parent_cells_for_overlap(
        table["min_overlap_fraction"], parent_grid_ratio=RATIO,
        child_nx=int(child_nx), child_ny=int(child_ny), diagonal=False)
    if bound is None:
        return table
    if bound < 1:
        # NAMED, because the alternative is the tracker's own
        # "max_shift_cells is below min_shift_cells" reaching a reader
        # who wrote neither number.  What breaks: this nest is narrow
        # enough that its own overlap floor admits no move at all, so a
        # following nest on it could never follow.  No rung of this
        # door's ladder reaches it -- the smallest is 24 parent cells
        # and admits 3 -- so this is the bound stated rather than a
        # table emitted that the run door would reject.
        raise ValueError(
            f"A {child_nx}x{child_ny} nest at ratio {RATIO} is "
            f"{min(int(child_nx), int(child_ny)) // RATIO} parent cells "
            "wide, and min_overlap_fraction = "
            f"{table['min_overlap_fraction']} admits no move at all on "
            "it; run a wider nest, or lower min_overlap_fraction "
            "deliberately")
    for key in ("max_shift_cells", "max_move_parent_cells"):
        table[key] = bound
    table["min_shift_cells"] = min(int(table["min_shift_cells"]),
                                   int(table["max_shift_cells"]))
    margin = int(VORTEX_PRESET["search_margin_cells"])
    table["search_margin_cells"] = max(
        margin, (min(int(child_nx), int(child_ny)) // RATIO) * margin
        // PRESET_NEST_PARENT_CELLS)
    return table


def _following_nest_dims(experiment) -> tuple[int, int]:
    """(nx, ny) of the domain this door marks as following.

    Grid 2 is the following nest on every document this door writes, and
    the receipt's own ``domains`` rows say so with the same test, so the
    dimensions the follow table is derived from and the dimensions the
    reader sees beside it cannot come apart.
    """
    child = next(d for d in experiment.domains if int(d.grid_id) == 2)
    return int(child.run.nx), int(child.run.ny)


#: The inner nest's ratio to the following nest: 3 km -> 1 km is 3.
INNER_NEST_RATIOS = (2, 3, 4, 5)


def parse_inner_nest(text: str) -> tuple[int, float]:
    """``--inner-nest RATIO,KM``: the ratio to the following nest and the side, km."""
    import argparse
    try:
        ratio_text, side_text = (part.strip() for part in str(text).split(","))
        ratio, side = int(ratio_text), float(side_text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"must be RATIO,KM (for example 3,200), not {text!r}") from None
    if ratio not in INNER_NEST_RATIOS or not (math.isfinite(side) and side > 0):
        raise argparse.ArgumentTypeError(
            f"RATIO must be one of {INNER_NEST_RATIOS} and KM a size above "
            f"zero, not {text!r}")
    return ratio, side


def inner_nest_dimensions(follower_dims, inner_nest, *, nest_km: float) -> tuple[int, int]:
    """(nx, ny) of the inner nest: the asked side, inside the follower's ring.

    The side is rounded to whole following-nest cells (WPS: n * ratio + 1
    staggered points), and capped so the inner nest clears the follower's
    own spec + blend rows on every side -- a follower the memory budget
    shrank keeps a valid inner nest, smaller, rather than a refused file.
    """
    from woof import domain_wizard as dw

    ratio, side_km = int(inner_nest[0]), float(inner_nest[1])
    ring = dw._SPEC_BDY_WIDTH + dw._BLEND_WIDTH
    spans = []
    for parent_cells in follower_dims:
        want = max(1, round(side_km / float(nest_km)))
        room = int(parent_cells) - 2 * ring - 2
        if room < 4:
            raise ValueError(
                f"the following nest ({follower_dims[0]}x{follower_dims[1]}) "
                "has no room for an inner nest inside its boundary ring")
        spans.append(min(want, room))
    return spans[0] * ratio, spans[1] * ratio


def configuration_text(*, cycle: str, point: tuple[float, float], hours: int = 6,
                       name: str | None = None, tiles: str = "auto",
                       source: str = "cyclone-setup.toml",
                       forcing_source: str = DEFAULT_SOURCE,
                       member: str | None = None,
                       start_hour: int = 0,
                       dimensions=None,
                       isftcflx: int | None = None,
                       history_interval_s: float | None = None,
                       nest_history_interval_s: float | None = None,
                       nest_km: float = 3., track_selection: dict | None = None,
                       model: str | None = None,
                       inner_nest: tuple[int, float] | None = None,
                       quicklook_interval_s: float | None = None) -> tuple[str, object]:
    from woof import domain_wizard as dw
    from woof.companion_domains import VORTEX_PRESET_SOURCE
    from woof.cyclone_sources import (declared_case_data, fetch_hints,
                                       moving_nest_note, source_adapter,
                                       validate_center)
    from woof.starter_template import render_tables

    if nest_km not in (3., 2.5):
        raise ValueError("--nest-km must be 3 or 2.5; 2.5 km uses a 10 km parent at integer ratio 4")
    root_dx_m = float(nest_km) * 1000. * RATIO
    from woof.cyclone_sources import acquisition_hints
    model = acquisition_hints(forcing_source, model).get("model")
    adapter = source_adapter(forcing_source)
    forcing_source = adapter.source_id
    moment = _cycle(cycle, forcing_source=forcing_source)
    # THE MODEL'S TIME ZERO.  The cycle says which run of the source this
    # comes from; the lead says where in that run the forecast begins.  A
    # storm that exists only at f186 is initialised from f186 and forced
    # from f186 onward, and every downstream stage reads the difference
    # from the pair the emitted file already carries -- `[fetch]
    # forecast_start_hour` and `start_time` -- rather than from a second
    # convention of this door's own.
    start_hour = _start_hour(start_hour, source=forcing_source, moment=moment)
    start_time = moment + timedelta(hours=start_hour)
    if tiles not in ("off", "auto", "on"):
        raise ValueError("Tile mode must be off, auto or on")
    lat, lon = point
    # The finiteness bound and the source's own coverage window in ONE
    # call, so this door and the seeder cannot disagree about whether a
    # center is on the grid.  Out of coverage is the one genuine refusal
    # on this path and it names the sources that do cover the point.
    validate_center(forcing_source, point)
    dims = [ROOT_DIMS, CHILD_DIMS] if dimensions is None else list(dimensions)
    if (len(dims) not in (2, 3) or any(len(pair) != 2 for pair in dims)
            or any(type(n) is not int or n <= 0 for pair in dims for n in pair)):
        raise ValueError("Cyclone dimensions must be two positive integer axis pairs")
    dims = dims[:2]
    projection = dw._projection_entries(lat, lon, "auto")
    if track_selection is not None:
        from woof.cyclone_tracks import layout
        from types import SimpleNamespace
        projection, _ = layout(point=point, fixes=track_selection["fixes"],
            dimensions=dims, dx_m=root_dx_m,
            clearance=_nest_clearance_cells(SimpleNamespace(spec_bdy_width=dw._SPEC_BDY_WIDTH, blend_width=dw._BLEND_WIDTH), dims[1]))
    # The fetch block the SOURCE declares: its cadence, its rounding to
    # its own forcing interval, its lead horizon, its member vocabulary
    # and whether its transport takes an area window.  The wizard's own
    # validator runs over the result, so a row that cannot state a
    # fetchable request is refused here and not at acquisition.  The
    # duration bound comes from the source's published horizon rather
    # than from one model's 384-hour ceiling written down here.
    hints = fetch_hints(source=forcing_source, moment=moment, hours=hours,
                        projection=projection, dims=tuple(dims[0]),
                        dx_m=root_dx_m, member=member, start_hour=start_hour, model=model)
    profile = _default_profile(forcing_source, nest_km)
    # THE INNER NEST (--inner-nest RATIO,KM): a third domain centred in the
    # following nest, which rides with it as it moves (the prepared tree's
    # descendant regrounding), sized to the asked side and kept inside the
    # following nest's own boundary ring.
    ratios = (RATIO,)
    if inner_nest is not None:
        inner = inner_nest_dimensions(dims[1], inner_nest, nest_km=nest_km)
        dims = [*dims, inner]
        ratios = (RATIO, int(inner_nest[0]))
    text = dw.render_config(
        name=_config_name(adapter, name, nest_km),
        start_time=start_time, hours=hours, projection=projection,
        dims=dims, ratios=ratios, root_dx_m=root_dx_m,
        profile=profile, cumulus_requested=False, tiles=tiles,
        fetch_hints=hints,
        case_data=declared_case_data(forcing_source, hints, source),
        history_interval_s=3600. if history_interval_s is None else float(history_interval_s),
        nest_history_interval_s=(900. if nest_history_interval_s is None
                                 else nest_history_interval_s))
    raw = tomllib.loads(text)
    child = next(row for row in raw["domain"] if row["grid_id"] == 2)
    # The maximums come from the dimensions this document actually
    # carries, not from the preset's own nest: a reduced child is
    # narrower and its floor admits fewer parent cells, a child grown to
    # a memory budget is wider and its floor admits more.
    child["follow"] = {**follow_table_for_nest(*dims[1]),
                       "track": {"path": "storm-track.d02.csv"}}
    if inner_nest is not None:
        # The inner core can move relative to the coarse parent vortex.
        # Carrying a centered inner grid with a parent-only fix lost the
        # eye outside the fine footprint in the whole storm run. Refine
        # steering on the existing inner state before moving both grids.
        child["follow"]["refine_grid_id"] = raw["domain"][-1]["grid_id"]
    if track_selection is not None:
        import re
        identifier = str(track_selection["lineage"].get("track_id", "")).upper()
        if re.fullmatch(r"[A-Z]{2}[0-9]{6}", identifier):
            child["follow"]["track"]["storm_id"] = identifier
        from woof.cyclone_tracks import toml_lineage
        grid = dw._root_grid(projection, *dims[0], root_dx_m)
        x, y = grid.latlon_to_ij(lat, lon)
        from woof.core.storm_tracking import _round_cells
        for key, axis, side in (("i_parent_start", float(x), dims[1][0]),
                                ("j_parent_start", float(y), dims[1][1])):
            child[key] = _round_cells(axis - (side / RATIO - 1.) / 2.)
        raw["lineage"] = {"cyclone_tracks": toml_lineage(track_selection["lineage"])}
    # This is one immediate following nest. Spawn/retire decisions are not part
    # of this quick-start; the chosen center is its initial registration.
    # The run door's own verdict on the nest this file declares, written
    # into the file that carries it: a configuration whose following nest
    # its source's chain cannot feed says so where it is read, instead of
    # leaving that sentence to the launch.
    moving = moving_nest_note(forcing_source,
                              follower_grid_ids=(FOLLOWER_GRID_ID,), point=point)
    # The heading says which question the sentence answers.  A row that
    # reaches no launch chain is not answering about the nest at all, and
    # heading it as if it were would point the reader at the wrong knob.
    heading = ("Launch route for this source: "
               if moving["launch_refusal"] is not None
               else "Moving nest on this source's chain: ")
    limit = "" if moving["integrates_moving_nest"] else "".join(
        "# " + line + "\n" for line in textwrap.wrap(
            heading + moving["note"], 76))
    began = (f"f{start_hour:03d} forecast (valid {start_time:%Y-%m-%d %H} UTC)"
             if start_hour else "f000 analysis")
    text = (f"# {adapter.display_title} cyclone quick-start: {root_dx_m/1000.:g} km parent and "
            f"{nest_km:g} km following nest.\n"
            "# Center and cycle were selected explicitly on that source's "
            f"{began}.\n"
            f"# Existing vortex-lock preset: {VORTEX_PRESET_SOURCE}"
            + ("" if tuple(dims[1]) == CHILD_DIMS else
               f" (movement bounds re-derived for a {dims[1][0]}x{dims[1][1]} nest)")
            + "\n"
            "# Following uses the MSLP minimum, with 850 hPa circulation as fallback.\n"
            + limit
            + render_tables(raw))
    text = dw.with_surface_flux_option(text, isftcflx)
    # The quick-look stream on the finest domain (woof domain --quicklook-interval's own writer).
    text = dw.with_quicklook(text, quicklook_interval_s)
    experiment = dw.experiment_from_text(text, source=source)
    return text, experiment


def _forcing_interval(forcing_source: str, *, cycle=None, start_hour=0,
                      hours=None) -> float:
    """The selected source's own boundary cadence, in seconds.

    An INGEST OPERAND, not a label: it sets the lateral-boundary store
    every phase estimate carries, so it reaches the flat budget AND the
    tree road's admission through the same ``operands`` dict every
    candidate on this door is priced with.  Read from the registry row
    rather than written down here, which is why a six-hourly source is
    priced as six-hourly without a line of its own.

    With the window (``cycle``, ``start_hour``, ``hours``) it is the
    spacing the fetch table takes (:func:`woof.cyclone_sources.
    window_cadence_hours`) where the cycle's ladder coarsens inside the
    window: a 240 h IFS setup downloads 6-hourly files, and a namelist
    or a price at the usual 3 h would describe files that never arrive.
    """
    from woof.cyclone_sources import source_adapter, window_cadence_hours
    interval = float(source_adapter(forcing_source).forcing_interval_seconds)
    if (type(hours) is not int or hours < 1 or type(start_hour) is not int
            or start_hour < 0):
        return interval
    try:
        moment = _cycle(cycle, forcing_source=forcing_source)
    except ValueError:
        # configuration_text refuses an unreadable cycle with its own
        # words, after the checks that come before it on this door; the
        # interval only prices a configuration that gets that far.
        return interval
    cadence = window_cadence_hours(forcing_source, moment=moment,
                                   start_hour=start_hour, hours=hours)
    if cadence is not None and cadence * 3600 > interval:
        return float(cadence * 3600)
    return interval


def _nest_parent_cells(child_dims) -> int:
    """The nest's own width in PARENT cells, on its narrow axis."""
    return min(int(child_dims[0]), int(child_dims[1])) // RATIO


def _nest_dimensions(parent_cells: int):
    """A square nest of ``parent_cells`` whole parent cells, parent as is."""
    side = RATIO * int(parent_cells)
    return [tuple(ROOT_DIMS), (side, side)]


def _nest_clearance_cells(experiment, child_dims) -> int:
    """Rows the nest needs between itself and the parent's edge.

    The boundary and blend zones, the tracker's search window and one
    maximum move -- read from the follow table THIS nest will carry,
    through the one derivation that emits it
    (:func:`follow_table_for_nest`), so the clearance and the emitted
    bounds cannot disagree.
    """
    follow = follow_table_for_nest(*child_dims)
    return (experiment.spec_bdy_width + experiment.blend_width
            + int(follow["search_margin_cells"])
            + max(int(follow["max_shift_cells"]),
                  int(follow["max_move_parent_cells"])))


def _nest_moves_at_all(child_dims) -> bool:
    """Whether this nest's OWN overlap floor admits any move.

    The same bound :func:`follow_table_for_nest` derives its maximums
    from, asked as a yes or no.  A ladder probes layouts far below
    anything it would propose -- the reduction ladder walks down to a
    twentieth of the requested nest -- and a nest that narrow admits no
    move at all: 6 parent cells against a 0.7 floor admits zero.  Such a
    rung is not a following-nest layout, so it is not a rung, and the
    emit path's named refusal stays a refusal rather than being raised
    out of a geometric probe.
    """
    from woof.companion_domains import VORTEX_PRESET
    from woof.core.nest_relocation import max_parent_cells_for_overlap

    bound = max_parent_cells_for_overlap(
        VORTEX_PRESET["min_overlap_fraction"], parent_grid_ratio=RATIO,
        child_nx=int(child_dims[0]), child_ny=int(child_dims[1]), diagonal=False)
    return bound is None or bound >= 1


def _nest_fits_parent(experiment, dims) -> bool:
    parent, child = dims
    if not _nest_moves_at_all(child):
        return False
    margin = _nest_clearance_cells(experiment, child)
    return all((axis - child[index] // RATIO) // 2 >= margin
               for index, axis in enumerate(parent))


def _nest_ladder(experiment) -> tuple[int, ...]:
    """Nest sizes a budget may choose, largest first, floor last.

    Square, in whole parent cells, and EVEN in them so both child axes
    stay divisible by ``2 * RATIO`` and the shared author can center the
    nest on whole parent cells.  The parent is kept as it is, so the
    ladder ends where the growing tracker window would reach the parent's
    boundary and blend zone -- a geometric bound, asked of each rung with
    the bounds that rung would carry.  Bounded at 64 rungs because that is
    what ``fit_ladder`` accepts for a bounded largest-first search.
    """
    cells, rung = [], PRESET_NEST_PARENT_CELLS
    while len(cells) < 64 and _nest_fits_parent(experiment, _nest_dimensions(rung)):
        cells.append(rung)
        rung += 2
    return tuple(reversed(cells))


def _fit_dimensions(scale, base=None):
    from woof import domain_wizard as dw
    root, child = ((ROOT_DIMS, CHILD_DIMS) if base is None
                   else (tuple(base[0]), tuple(base[1])))
    return [tuple(dw._even(n * scale) for n in root),
            tuple(RATIO * dw._even(n * scale / RATIO) for n in child)]


def _reduction_dimensions(base=None):
    """The reduction ladder's dimension builder for ``base``.

    A base that IS the preset layout -- which is every proposal on this
    door until a budget sizes the nest -- reduces from the preset, and is
    built by the module function with one argument, exactly as it was
    before a base could be anything else.  Anything else is a layout this
    door chose, and the ladder scales that one instead of the preset it
    was never asked for.
    """
    if base is None or [list(pair) for pair in base] == [list(ROOT_DIMS),
                                                         list(CHILD_DIMS)]:
        return _fit_dimensions
    return lambda scale: _fit_dimensions(scale, base)


def _fit_scales(experiment, base=None):
    # Keep the whole tracker search window clear of the boundary/blend zone
    # even after one maximum requested move. Runtime still enforces overlap,
    # movement bounds and containment on EVERY actual relocation.
    dims_at = _reduction_dimensions(base)
    scales = tuple(step / FIT_SCALE_STEPS for step in range(FIT_SCALE_STEPS - 1, 0, -1)
                   if _nest_fits_parent(experiment, dims_at(step / FIT_SCALE_STEPS)))
    if not scales:
        # NAMED, because the alternative is fit_ladder's internal contract
        # message ("candidate_scales must be a tuple of 1..64 decreasing
        # positive finite scales") reaching a reader who never chose a
        # scale ladder.  What breaks: every rung of this ladder puts the
        # tracker's search window inside the parent's boundary/blend zone,
        # so a following nest could relocate into cells the parent does
        # not integrate.  The way out is the requested domain itself.
        raise ValueError(
            "No smaller cyclone layout keeps the following nest's tracker "
            f"search window {_nest_clearance_cells(experiment, CHILD_DIMS)} "
            "cells clear of the 12 km parent's boundary and blend zone, so "
            "there is nothing to propose; run the requested domain with "
            "--tiles off, or use a larger card")
    return scales


#: The two budgets these floors are compared against are NOT the same
#: number, and the direction is what makes that safe.  The streaming
#: floor is the term `decide_tree` compares against its own, TIGHTER
#: tree budget (`_tree_budget_bytes`); `budget` here is the wizard's
#: `sizing_budget_bytes`, which is larger.  So a floor that exhausts THIS
#: budget certainly exhausts the tree's: the refusal cannot fire falsely
#: on the loose comparison, only stay silent where the tighter one would
#: have spoken -- and staying silent costs a bounded search, not a wrong
#: answer.  Reversing the direction (or swapping in the tree budget
#: without saying so) would turn it into a false "resizing cannot help".
def _fixed_floors(phases, budget, tiles):
    # Read the estimator/planner's own lower bounds, not a second byte model.
    forecast = getattr(phases, "forecast", None)
    resident = getattr(forecast, "fixed_envelope_bytes", None)
    streamed = getattr(getattr(phases, "tree_road", None),
                       "streaming_fixed_floor_bytes", None)
    required = ([resident] if tiles == "off" else [streamed] if tiles == "on"
                else [resident, streamed])
    # A STREAMING floor alone cannot rule out a smaller all-resident AUTO tree.
    impossible = all(floor is not None and floor >= budget for floor in required)
    return impossible, {"resident_fixed_floor_bytes": resident,
                        "streaming_fixed_floor_bytes": streamed}


def _recommended_mode(tiles: str) -> str:
    """The mode this door would tell a ``tiles`` reader to re-run with.

    ONE function, because the price and the sentence must not be able to
    disagree: :func:`_unreduced_resident_admission` prices this mode and
    :func:`_keeps_coverage_sentence` names it.  Under ``--tiles on`` that
    is ``auto`` -- auto weighs a tree resident before it consults the
    planner, so withdrawing the mode is a smaller change than turning
    streaming off.  Under ``auto`` there is no mode to withdraw and
    ``off`` is the answer.
    """
    return "auto" if tiles == "on" else "off"


def _admits_resident(phases) -> bool:
    """Did the tree admission keep EVERY domain resident on this road?

    ``--tiles auto`` is not priced by the resident envelope alone: it goes
    through ``streaming.decide_tree``, which withholds a moving nest's
    rebuild from the admission budget.  In the band between the withheld
    and unwithheld budgets the envelope is under the budget and auto still
    refuses, so the envelope comparison alone is not the question auto
    answers, and a door that recommended auto off that comparison
    recommended a mode that then refused.
    """
    road = getattr(phases, "tree_road", None)
    rows = () if road is None else tuple(getattr(road, "rows", ()) or ())
    return bool(rows) and getattr(road, "refusal", None) is None \
        and getattr(road, "report_error", None) is None \
        and all(row["road"] == "resident" for row in rows)


#: What the REQUESTED, unreduced cyclone tree costs in the mode this door
#: would name as the way to keep it, and whether the declared card admits
#: it THAT way.  The mode is :func:`_recommended_mode`'s, and it is the
#: same call :func:`_keeps_coverage_sentence` prints: pricing one mode and
#: recommending another is how the door came to recommend ``--tiles auto``
#: on the strength of what ``--tiles off`` costs, in a band where auto
#: refuses.
#:
#: The reduction this door performs is a SCIENCE reduction: a 2,400 x
#: 1,920 km 12 km parent is there to carry the steering environment, and
#: 1,560 x 1,248 km does not.  On the 6-8 GiB band the tile planner's
#: tree road refuses layouts the same card holds resident, so the door
#: was proposing degraded coverage -- or refusing outright -- while a
#: mode the user can select ran the domain they asked for, and never
#: said so.  It is REPORTED, never applied: streaming is off by choice
#: and a door that silently switched a memory mode would be lying about
#: what it emitted.
#:
#: Priced by the same estimator, on the same operands, against the same
#: budget, and through the SAME one hardware snapshot as every other
#: candidate here -- nothing is redetected and no number is inflated to
#: make the resident route look admissible.  The operands and the machine
#: reach this function only inside ``budget_of`` and ``price_off``, which
#: is the point: there is no second set of them to get wrong, and no
#: argument here a caller could vary to change what is priced.  A
#: resident experiment does not consult the tile planner at all, so the
#: snapshot is carried only so that this probe cannot become a second,
#: differently-measured machine.
#: ``None`` means the unreduced request is not admitted that way either,
#: and there is nothing to name.
def _unreduced_resident_admission(intent, budget_of, price_mode):
    """Price the unreduced request in the mode the sentence will NAME."""
    if intent["tiles"] == "off":
        return None
    mode = _recommended_mode(intent["tiles"])
    try:
        _text, exp = configuration_text(**{**intent, "tiles": mode})
        phases = price_mode(exp)
        budget = budget_of(exp)
    except (ValueError, OSError):
        # Not an admission answer -- the recommended route could not even
        # be priced.  Claim nothing; the caller's own refusal stands as it
        # is.
        return None
    if phases.peak_envelope_bytes > budget:
        return None
    if mode == "auto" and not _admits_resident(phases):
        return None
    return {"tiles": mode, "dimensions": _requested_dimensions(intent),
            "peak_envelope_bytes": phases.peak_envelope_bytes,
            "budget_bytes": _admitting_budget_bytes(phases, mode, budget)}


def _admitting_budget_bytes(phases, mode: str, budget: int) -> int:
    """The budget the admission this door NAMES was actually judged against.

    ``budget_of`` is the whole-process allowance, undiminished.  Under
    ``auto`` the tree walk does not judge against that number: it
    withholds a moving nest's rebuild transient from it first
    (``streaming._resident_admission``, ``withheld_bytes``) and every
    comparison downstream spends the reduced allowance.  So the sentence
    quoted an allowance 0.5 GiB larger than the one that admitted the
    tree, and a reader checking the arithmetic against `woof check`'s
    own streaming block -- which prints the withheld figure -- found two
    budgets for one decision.  ``off`` withholds nothing and is
    unchanged; a walk that could not be priced has no budget of its own
    to quote and keeps the caller's.
    """
    if mode != "auto":
        return budget
    road = getattr(phases, "tree_road", None)
    walked = int(getattr(road, "total_budget_bytes", 0) or 0)
    return walked if walked else budget


#: What `--tiles off` actually authors when the resident route cannot
#: hold the requested domain either.  Asked ONLY on the refusal path,
#: where the door is about to hand back nothing at all: a refusal that
#: names no way through is the defect the refusal law is about, and
#: "try --tiles off" without a layout behind it is a suggestion, not an
#: answer.  So it is measured -- the same bounded ladder, the same
#: estimator, the same budget -- and reported with the dimensions it
#: found.  It is a REDUCTION too, and says so; it is not silently
#: applied, and the requested mode is what the caller asked for.
#: It answers TWO questions, so it returns two things.  ``found`` is the
#: layout, for the refusal that names a way through.  ``measured`` says
#: whether the resident ladder was actually WALKED -- because a refusal
#: with no layout is either "the resident route was priced and admits
#: nothing on this card" (a measurement, and the one thing that makes
#: "the computer cannot admit" a true sentence for a tiled request) or
#: "the resident route could not be priced at all" (no evidence, claim
#: nothing).  A memory-typed refusal out of the ladder IS the first:
#: every rung was priced and every rung was refused.  Any other failure
#: -- an unloadable candidate, a coverage or extent bound, an OSError --
#: is the second, and does not license a claim about the card.
def _resident_alternative(intent, sizing, dims_of, scales_of):
    """``(measured, found)``: the largest resident rung `--tiles off`
    admits, and whether the resident route was priced to find out.

    Priced on the SELECTED source, like every other candidate on this
    door: the forcing interval is an ingest operand, so a six-hourly
    source's resident ladder is not the three-hourly one's, and a probe
    that quoted one source's numbers inside another's refusal would be
    offering a layout nobody measured.
    """
    from woof import domain_wizard as dw

    if intent["tiles"] == "off":
        return False, None
    resident = dict(intent, tiles="off")
    forcing_source = intent["forcing_source"]
    try:
        _text, exp = configuration_text(**resident)
        dims, _fitted = dw.fit_ladder(
            ratios=(RATIO,), free_bytes=sizing.free_bytes,
            vram_gib=sizing.vram_gib, device_profile=sizing.device_profile,
            target_machine=None, hours=intent["hours"],
            start_time=(_cycle(intent["cycle"], forcing_source=forcing_source)
                        + timedelta(hours=intent["start_hour"])),
            projection=tomllib.loads(_text)["projection"],
            source=forcing_source,
            name=intent["name"], root_dx_m=float(intent.get("nest_km", 3.))*1000.*RATIO,
            profile=_default_profile(forcing_source, intent.get("nest_km", 3.)),
            tiles="off",
            forcing_interval_seconds=_forcing_interval(
                forcing_source, cycle=intent["cycle"],
                start_hour=intent["start_hour"], hours=intent["hours"]),
            candidate_builder=lambda proposed: configuration_text(
                **{**resident, "dimensions": proposed})[1],
            dimensions_builder=dims_of, candidate_scales=scales_of(exp),
            layout_label="cyclone 12/3 km resident")
    except dw.DomainFitError as error:
        return error.resource in {"vram", "host", "memory"}, None
    except (ValueError, OSError):
        return False, None
    return True, {"tiles": "off", "dimensions": [list(pair) for pair in dims]}


def _resident_alternative_sentence(found) -> str:
    dims = found["dimensions"]
    return (f"--tiles off is not refused here: it authors "
            f"{dims[0][0]}x{dims[0][1]} / {dims[1][0]}x{dims[1][1]} on this "
            "computer as one resident allocation.  That is smaller ground "
            "than was requested, so review it as a reduction")


#: The way out when there is no tile mode left to try.  Said only after
#: the resident ladder was walked and admitted nothing, so it is a
#: measurement and not a guess: the refusal that carries it has already
#: named the computer as the bound, and this is the sentence that says
#: what would move it.  Without it the hardware refusal on this band
#: ended at "no smaller candidate passed" -- true, and no way out.
def _no_resident_sentence(budget) -> str:
    return ("--tiles off is not a way through here either: the same "
            "bounded ladder, priced resident on this same computer, "
            f"admitted no layout under its {budget} byte budget, so what "
            "this needs is more free VRAM -- a larger card, or this one "
            "to itself -- not a different tile mode")


#: The way out of the refusal that has NO layout to point at.
#:
#: ``impossible`` means the estimator's own fixed floors -- the process
#: and radiation costs a run pays before one grid cell is stored --
#: already reach the budget, so no rung of any ladder is walked and there
#: is nothing measured to offer.  The refusal law still wants a way out,
#: and this was the one refusal on this door that named none: the HARDER
#: refused case got less guidance than the softer one beside it, which
#: names "more free VRAM -- a larger card, or this one to itself".
#:
#: What is said here is what the floors already on the payload show, and
#: nothing further.  The floor is compared against the BUDGET, so more
#: free VRAM moves the comparison; the floor IS the selected physics'
#: fixed cost, so a lighter selection moves the floor; the grid moves
#: neither, which is what the head sentence already says.  Where the
#: other memory mode's floor was measured and sits UNDER the budget it is
#: named as a floor that is not exhausted -- never as a layout, because
#: on this path no layout was priced.
def _fixed_floor_way_out(floors, budget, tiles) -> str:
    other, mode = (("resident_fixed_floor_bytes", "off") if tiles == "on" else
                   ("streaming_fixed_floor_bytes", "on") if tiles == "off" else
                   (None, None))
    value = floors.get(other) if other else None
    unexhausted = (
        f"; --tiles {mode}'s own fixed floor is {value} bytes, under that "
        "budget, so its floor is not what is exhausted here -- but no "
        "layout was priced on that road, and none is offered"
        if value is not None and value < budget else "")
    return ("What moves this is the budget or the floor itself, never the "
            f"grid: more free VRAM raises the {budget} byte budget -- a "
            "larger card, or this one to itself -- and a lighter physics "
            "selection lowers the fixed cost" + unexhausted)


#: Why the proposal stopped where it did, when something other than the
#: card stopped it.  The reason string is the fitter's own -- the same
#: one `stop_out` carries to the wizard's plan summary -- so this door
#: and that one cannot disagree about what bound a fit.  Stated as fact
#: on the document, never as a warning, and it names no flag this door
#: does not have.
def _stopped_by_sentence(stopped) -> str:
    return (f"The next larger layout was rejected on the {stopped['scope']}, "
            f"not the card: {stopped['reason']}")


def _keeps_coverage_sentence(admitted, tiles: str) -> str:
    """The one sentence that names the way to keep the requested ground,
    with the numbers that make it a claim.

    WHICH WAY DEPENDS ON WHAT COMPELLED THE TILED ROAD.  Under ``--tiles
    on`` the tiled road was ASKED FOR: the computer holds this tree, and
    ``--tiles auto`` weighs a tree resident before it consults the planner
    at all, so auto is the remedy and ``off`` is a bigger change than the
    reader needs.  Under ``auto`` there is no mode to withdraw, and ``off``
    remains the sentence.

    THE MODE NAMED IS THE MODE PRICED.  ``admitted`` carries the mode
    :func:`_unreduced_resident_admission` measured it in, and this
    sentence reads it from there rather than deciding a second time: the
    door recommended ``--tiles auto`` on the strength of what ``--tiles
    off`` costs, and in the band where auto withholds a moving nest's
    rebuild the recommended mode then refused.
    """

    dims = admitted["dimensions"]
    mode = admitted["tiles"]
    ground = (f"the requested {dims[0][0]}x{dims[0][1]} / "
              f"{dims[1][0]}x{dims[1][1]} domain on this computer as one "
              f"resident allocation ({admitted['peak_envelope_bytes']} bytes "
              f"against a {admitted['budget_bytes']} byte budget)")
    if tiles == "on":
        return (f"--tiles on is what compels the tiled road here, not the "
                f"computer: --tiles {mode} admits {ground}, so re-run with "
                f"--tiles {mode} to keep the requested coverage")
    return (f"--tiles {mode} admits {ground}; re-run with --tiles {mode} to "
            "keep the requested coverage")


#: HOW THE PROPOSED TREE RUNS, on the document that proposes it.
#:
#: ``--tiles auto`` is a request, not an outcome: the same layout runs
#: resident on one card and streamed on another, and until this block
#: existed the cyclone document said only which mode was ASKED for.  A
#: reviewer reading "tiles: auto" had no way to tell whether the proposal
#: in front of them keeps the whole tree on the card or pins a host store
#: and cycles tiles through it, which is a difference of 1.2x-1.4x in
#: wall time and of tens of gigabytes of host RAM.
#:
#: Read off the SAME walk the run door takes -- ``phases.tree_road`` is
#: :func:`woof.core.streaming.decide_tree`'s own decisions -- so the
#: document cannot claim a road the run will not take.  Same shape as
#: ``downscale-plan.json``'s ``streaming`` block: the mode, the road, the
#: budget it was decided against, and the per-domain verdict with its
#: tiling where there is one.  Additive: the schema version is unchanged
#: because no existing key moves or changes meaning.
def _streaming_entry(phases, tiles: str, budget: int) -> dict:
    """The ``streaming`` block: which road each domain takes, and why."""
    road_plan = getattr(phases, "tree_road", None)
    rows = tuple(getattr(road_plan, "rows", ()) or ())
    common = {"mode": tiles, "budget_bytes": int(budget),
              "peak_envelope_bytes": phases.peak_envelope_bytes}
    if rows:
        return {**common,
                "road": ("streamed" if any(row["road"] == "streamed"
                                           for row in rows) else "resident"),
                "domains": [{"grid_id": row["grid_id"], "road": row["road"],
                             "why": row["reason"], "tile": row.get("tile")}
                            for row in rows]}
    if tiles == "off":
        # NOT an unpriced road: "off" IS the answer, and it needs no walk.
        return {**common, "road": "resident",
                "reason": "[tiles] mode = 'off': every domain is resident by "
                          "configuration, so no tree road is priced",
                "domains": []}
    # AN UNPRICED ROAD IS NOT A RESIDENT ONE.  Saying "resident" with no row
    # behind it is a claim about how the run goes, made where the walk
    # produced nothing -- a refusal, a report failure, or a tree this
    # surface never priced -- and a reviewer sizing a card reads it as a
    # verdict.  Say that it could not be priced, and say what said so.
    return {**common, "road": None,
            "reason": (getattr(road_plan, "refusal", None)
                       or getattr(road_plan, "report_error", None)
                       or "this configuration prices no tree road"),
            "domains": []}


def _requested_dimensions(intent) -> list:
    """The layout this plan is ABOUT, preset or budget-sized.

    Every probe on this door re-prices the requested layout, and once a
    budget can choose the nest there is no longer a module constant that
    names it.  One reader, so a probe cannot price the preset while the
    proposal is a different size.
    """
    dims = intent.get("dimensions")
    return ([list(ROOT_DIMS), list(CHILD_DIMS)] if dims is None
            else [list(pair) for pair in dims])


def _budget_sizing(sizing, budget_gib: float | None):
    """The card allowance this plan is priced against, narrowed to a budget.

    A nest budget is a statement about how much of the card this run may
    occupy, so it is applied where the card's own availability is applied
    and NOT as a second comparison further down: every phase estimate, the
    tile planner's tree road and the final admission then see one number,
    and the budget cannot admit a layout the tree road would refuse.
    ``sizing_budget_bytes`` is free bytes minus the external margin, so the
    free bytes that express a budget of B are B plus that margin.

    A budget larger than the card is not a refusal and not a promise: the
    card is still the card, the smaller of the two is what sizes the nest,
    and the document says which one bound it.
    """
    import dataclasses
    from woof.core.preflight import EXTERNAL_MARGIN_BYTES
    from woof.domain_wizard import GIB

    if budget_gib is None:
        return sizing, None
    wanted = int(float(budget_gib) * GIB) + EXTERNAL_MARGIN_BYTES
    if wanted >= int(sizing.free_bytes):
        return sizing, "card"
    return dataclasses.replace(sizing, free_bytes=wanted), "request"


def _floor_price(intent, price, floor_dims):
    """What the preset nest costs here, measured, or ``None`` if the floor
    could not be priced at all.  A refusal states a number it took from the
    same estimator that refused it, or it states none.

    A REFUSED FLOOR IS STILL A PRICED FLOOR.  Under the default `--tiles
    auto` the estimator raises on the preset nest rather than returning
    phases -- the tile planner's tree road refuses before it answers --
    and the refusal then named no price at all on exactly the tile mode
    every user meets, while `--tiles off` named one.  The envelope the
    fitter measured travels on the error, so it is read from there.
    """
    from woof import domain_wizard as dw

    try:
        _text, exp = configuration_text(**{**intent, "dimensions": floor_dims})
        return price(exp).peak_envelope_bytes
    except dw.DomainFitError as error:
        phases = getattr(error, "phases", None)
        return None if phases is None else phases.peak_envelope_bytes
    except (ValueError, OSError):
        return None


def _unbudgeted_alternative(intent, *, sizing, operands):
    """What dropping ``--nest-budget-gib`` actually authors on this card.

    Asked only where the CARD is what refused the preset nest, which is
    the one case in which raising the flag moves nothing at all: the
    budget named is already larger than the card's own free memory, so
    the way through is to stop naming a budget and let this door's own
    reduction road propose a smaller layout.  That way through is
    MEASURED here -- the same bounded ladder, the same estimator, the
    same card as the run that refused -- because a refusal naming a
    remedy nobody priced is a suggestion and not an answer.  ``None``
    means the ladder was not walked or admitted nothing, and the sentence
    then claims no layout.
    """
    from woof import domain_wizard as dw
    from woof.cyclone_sources import source_adapter

    plain = {**intent, "dimensions": None}
    forcing_source = plain["forcing_source"]
    requested = [list(ROOT_DIMS), list(CHILD_DIMS)]
    machine = None
    if plain["tiles"] != "off":
        from woof.core.streaming import planner_machine
        machine = planner_machine(vram_bytes=sizing.free_bytes,
                                  name="woof cyclone budget",
                                  device_profile=sizing.device_profile)
        if machine is None:
            return None
    try:
        text, exp = configuration_text(**plain)
        dims, _fitted = dw.fit_ladder(
            ratios=(RATIO,), free_bytes=sizing.free_bytes,
            vram_gib=sizing.vram_gib, device_profile=sizing.device_profile,
            target_machine=machine, hours=plain["hours"],
            start_time=(_cycle(plain["cycle"], forcing_source=forcing_source)
                        + timedelta(hours=plain["start_hour"])),
            projection=tomllib.loads(text)["projection"],
            source=forcing_source,
            name=_config_name(source_adapter(forcing_source), plain["name"], plain.get("nest_km", 3.)),
            root_dx_m=float(intent.get("nest_km", 3.))*1000.*RATIO,
            profile=_default_profile(forcing_source, intent.get("nest_km", 3.)),
            tiles=plain["tiles"],
            forcing_interval_seconds=operands["forcing_interval_seconds"],
            candidate_builder=lambda proposed: configuration_text(
                **{**plain, "dimensions": proposed})[1],
            dimensions_builder=_reduction_dimensions(requested),
            candidate_scales=_fit_scales(exp, requested),
            layout_label="cyclone 12/3 km")
    except (dw.DomainFitError, ValueError, OSError):
        return None
    return [list(pair) for pair in dims]


def _tile_road_bound(error):
    """The number the TILE ROAD actually weighed the floor against.

    A refusal states the budget it compared against, and on the default
    ``--tiles auto`` that budget is NOT the flat sizing budget this door
    computes: the tree walk withholds the following nest's rebuild
    transient first, so its admission budget is smaller, and a refusal
    quoting the flat number printed a target ABOVE the price beside a
    sentence saying the price did not fit.  Measured on the 16 GiB
    fixture card at a 5.06 GiB nest budget: the floor prices
    5,141,378,237 bytes, the flat fit target is 5,161,476,948, and the
    tile road refused it against 5,139,501,921 with 293,631,708 bytes
    withheld for the nest's rebuild.  The walk carries all of that on
    its plan (:class:`woof.core.streaming.TreeRoadPlan`), so it is read
    from there rather than recomputed or parsed back out of the
    sentence.  ``None`` on the flat road, where no such budget exists
    and the fit target IS what bound the floor.
    """
    road = getattr(getattr(error, "phases", None), "tree_road", None)
    budget = getattr(road, "admission_budget_bytes", None)
    if budget is None:
        return None
    return {"budget_bytes": int(budget),
            "withheld_bytes": int(getattr(road, "admission_withheld_bytes", 0) or 0),
            "withheld_for": getattr(road, "admission_withheld_for", None),
            "remedy": getattr(road, "admission_remedy", None)}


def _nest_budget_floor_refusal(floor_dims, *, budget: int, cost,
                               budget_bound, requested_bytes,
                               alternative=None, fit_bound=None) -> str:
    """Why the preset nest cannot be had here, naming what actually bound it.

    TWO DIFFERENT BREAKAGES REACH THIS SENTENCE and they do not have the
    same way out.  ``budget_bound`` is the answer :func:`_budget_sizing`
    already computed a hundred lines above: ``request`` means the named
    budget is the smaller of the two numbers and raising it moves the
    wall; ``card`` means the card's own free memory is, the named budget
    is larger than anything this machine could offer, and raising it
    moves nothing.  Saying "raise --nest-budget-gib" to a caller who
    asked for 24 GiB on a 6 GB card names a remedy that cannot work and
    blames a number that did not refuse, and a form reading ``bound_by``
    is told the caller's own budget refused a run the card refused.  The
    card-bound reader's way through is to drop the flag, and that is a
    layout this door measures rather than a mode it suggests.

    The floor is the shipped quick-start's own nest, so the sentence
    names it in both units a reader has -- cells and the parent cells it
    registers on -- and what it actually costs on this card, measured by
    the same estimator that refused it rather than asserted.

    WHAT THE FLOOR IS COMPARED AGAINST IS THE BUDGET, the same number the
    door compares the preset against without the flag
    (:func:`_admitted_as_requested`), because the floor IS that layout.
    It was the fit target -- the budget less ``fit_headroom_bytes`` --
    while the flagless door used the budget, so on the band between the
    two (5.25 to 5.475 GiB free on a 6 GiB card at the 1.13 margin) this
    sentence said the card could not hold the preset nest and named a
    smaller layout as the way out, and dropping the flag authored the
    whole preset unchanged (A175).  Quoting the budget is safe from the
    contradiction the fit target was once introduced to cure: that one
    printed the budget while comparing against the target, so the
    printed number sat above the price; here the printed number is the
    one compared, so a floor the flat road refuses on card memory prices
    above it.

    AND THE BUDGET IS ONLY THE FLAT ROAD'S BOUND.  On the DEFAULT
    ``--tiles auto`` the tree walk withholds the following nest's rebuild
    transient before it compares anything, so it refuses against a
    SMALLER budget, and quoting the flat number there prints a figure
    above the price beside a sentence saying the price did not fit -- at
    a 5.06 GiB budget the flat road's number was 5,161,476,948 bytes
    against a 5,141,378,237 byte price, and the number that bound was
    5,139,501,921.  ``fit_bound`` (:func:`_tile_road_bound`) is that
    number where the tile road is what refused, and it is quoted in place
    of the flat one whenever the two differ, with the withholding that
    produced it and the walk's own way out beside it.  The way out is
    re-aimed with it: "raise --nest-budget-gib until it clears what the
    floor prices" is false advice on that road, where the flat number
    already cleared the price and the run was refused anyway.

    ``requested_bytes`` is what the caller asked for and is required on
    the card-bound branch, which states it.  It once had a ``None`` arm
    that left that clause out and started the next one lowercase ("...
    against {target}. the card is the bound"); no call site ever took it,
    so the arm is gone rather than being given a sentence of its own.
    """
    priced = "" if cost is None else f", which prices {cost} bytes here"
    floor = (f"the floor is {floor_dims[1][0]}x{floor_dims[1][1]} "
             f"({_nest_parent_cells(floor_dims[1])} parent cells at ratio "
             f"{RATIO}){priced}")
    target = (f"the {budget} byte sizing budget this card's free memory "
              "leaves" if budget_bound == "card" else
              f"a {budget} byte budget")
    aim = "raise --nest-budget-gib until it clears what the floor prices"
    road_remedy = ""
    if (fit_bound is not None
            and int(fit_bound["budget_bytes"]) != budget):
        withheld = int(fit_bound.get("withheld_bytes") or 0)
        for_whom = fit_bound.get("withheld_for")
        held = ("" if not (withheld and for_whom) else
                f", which withholds {withheld} bytes for {for_whom}'s rebuild")
        target = (f"the {int(fit_bound['budget_bytes'])} byte admission "
                  "budget the tile road this run takes weighed it "
                  f"against{held}")
        # AIMED AT THE NUMBER THAT BINDS, and the flat road's budget is
        # not printed beside it, so there is no second figure for a
        # reader to raise the flag against.  No derivation of the road's
        # budget from the flag is claimed: measured, it is this budget
        # less the withholding, which is a coincidence of one
        # configuration.  What is claimed is that it
        # MOVES with the flag, which it does -- 5.045 GiB gives
        # 5,123,395,794 bytes and 5.06 GiB gives 5,139,501,921.
        aim = ("raise --nest-budget-gib until THAT budget clears what the "
               "floor prices, which it does as the flag moves")
        if fit_bound.get("remedy"):
            road_remedy = (". The walk's own way out on that road: "
                           + fit_bound["remedy"])
    if budget_bound == "card":
        if alternative:
            way_out = (f"drop --nest-budget-gib: this door's own reduction "
                       f"road then authors {alternative[0][0]}x"
                       f"{alternative[0][1]} / {alternative[1][0]}x"
                       f"{alternative[1][1]} on this computer, which is "
                       "smaller ground than the preset nest, so review it as "
                       "a reduction")
        else:
            way_out = ("drop --nest-budget-gib and let this door's own "
                       "reduction road propose the largest layout this "
                       "computer does admit")
        return ("This computer's free memory cannot hold the cyclone "
                f"preset's own nest: {floor}, against {target}. The "
                f"{requested_bytes} byte --nest-budget-gib asked for is "
                "larger than that, so it is not what refused this run and "
                "raising it cannot move it: the card is the bound. The nest "
                "is sized UP from that floor and never below it, so "
                f"{way_out}")
    return (f"--nest-budget-gib is under the cyclone preset's own nest: "
            f"{floor}, against {target}. The nest is sized UP from that "
            f"floor and never below it; {aim}, or drop it and let the "
            f"card's own available memory size the run{road_remedy}")


def _admitted_as_requested(exp, price, budget: int, cancelled=None):
    """``(phases, refusal)`` for a layout judged AS REQUESTED; ``refusal``
    is ``None`` when it is admitted.

    A requested layout is admitted when the estimator prices it without a
    memory refusal (the tile planner's tree road and the host wall raise
    one) and its peak envelope is within ``budget``.  No fit headroom is
    held back: the headroom is what a search leaves unspent when it GROWS
    or shrinks a grid toward the budget, and a requested layout is
    neither.  ONE test, asked by both doors of the same layout -- the
    flagless door of the preset it was asked for, and the
    ``--nest-budget-gib`` door of its floor, which is that preset.  They
    were two tests, the floor's with the headroom, and on a card between
    the two the flag refused the preset nest that dropping it authored
    unchanged (A175).
    """
    from woof import domain_wizard as dw

    try:
        phases = price(exp)
    except dw.DomainFitError as error:
        dw.check_fit_cancelled(cancelled)
        if error.resource not in {"vram", "host", "memory"}:
            raise
        return error.phases, error
    if phases.peak_envelope_bytes > budget:
        return phases, dw.DomainFitError(phases.verdict(budget),
                                         resource="vram", phases=phases)
    return phases, None


def _size_nest_to_budget(intent, *, sizing, target_machine, operands, price,
                         cancelled=None):
    """The largest square nest this budget holds, parent kept as it is.

    THE SAME ADMISSION THE PROPOSAL IS JUDGED BY, and deliberately not a
    second one: ``fit_ladder``'s bounded largest-first search prices every
    rung through the same estimator, against the same budget, and stops
    short of it by the same ``fit_headroom_bytes`` the reduction road
    leaves, so the nest this returns is admitted on the terms the document
    then reports.  The rungs are nest sizes rather than scale factors --
    square, whole even parent cells, the parent untouched -- and the
    largest that fits wins because the ladder is walked from the top.

    THE FLOOR IS THE PRESET ITSELF, so when no rung clears the fit target
    the floor is judged the way the door judges the preset without the
    flag (:func:`_admitted_as_requested`): against the budget, with the
    same estimator and the same machine.  Judged against the fit target
    instead, the flag refused, on a card the flagless door filled with
    the whole preset, the very layout that door authored (A175).  A floor
    admitted that way still faces the bound the ladder asks of every
    rung's parent, which the memory refusal kept it from reaching.

    Returns ``(dimensions, searched)``: ``searched`` is True when the
    ladder chose the nest, and so held the fit headroom back, and False
    when the floor was admitted as requested, which holds none back.
    The document's ``headroom_bytes`` says which.
    """
    from woof import domain_wizard as dw
    from woof.cyclone_sources import source_adapter

    floor_dims = _nest_dimensions(PRESET_NEST_PARENT_CELLS)
    floor_text, floor_exp = configuration_text(
        **{**intent, "dimensions": floor_dims})
    projection = tomllib.loads(floor_text)["projection"]
    forcing_source = intent["forcing_source"]
    try:
        dims, _exp = dw.fit_ladder(
            ratios=(RATIO,), free_bytes=sizing.free_bytes,
            vram_gib=sizing.vram_gib, device_profile=sizing.device_profile,
            target_machine=target_machine, hours=intent["hours"],
            start_time=(_cycle(intent["cycle"], forcing_source=forcing_source)
                        + timedelta(hours=intent["start_hour"])),
            projection=projection, source=forcing_source,
            name=_config_name(source_adapter(forcing_source), intent["name"], intent.get("nest_km", 3.)),
            root_dx_m=float(intent.get("nest_km", 3.))*1000.*RATIO,
            profile=_default_profile(forcing_source, intent.get("nest_km", 3.)),
            tiles=intent["tiles"],
            forcing_interval_seconds=operands["forcing_interval_seconds"],
            candidate_builder=lambda proposed: configuration_text(
                **{**intent, "dimensions": proposed})[1],
            dimensions_builder=lambda rung: _nest_dimensions(int(rung)),
            candidate_scales=tuple(float(rung) for rung in _nest_ladder(floor_exp)),
            layout_label="cyclone following nest", cancelled=cancelled)
    except dw.DomainFitError as error:
        if error.resource not in {"vram", "host", "memory"}:
            raise
        _phases, refusal = _admitted_as_requested(
            floor_exp, price, dw.sizing_budget_bytes(floor_exp, **operands),
            cancelled)
        if refusal is not None:
            raise refusal
        # Every rung shares this parent, so this is the bound the ladder
        # would have named had memory let the floor reach it.
        bound = dw.point_request_bound(projection, *ROOT_DIMS, ROOT_DX_M)
        if bound is not None:
            raise dw.DomainFitError(
                f"{bound[1]}; {dw._exhausted_point_bound_remedy(bound[0])}",
                resource="extent") from error
        return [list(pair) for pair in floor_dims], False
    return [list(pair) for pair in dims], True


def plan_cyclone(*, cycle: str, point: tuple[float, float], sizing, target_machine=None,
                 hours: int = 6, name: str | None = None,
                 tiles: str = "auto", source: str = "cyclone-setup.toml",
                 forcing_source: str = DEFAULT_SOURCE, member: str | None = None,
                 start_hour: int = 0, nest_budget_gib: float | None = None,
                 cancelled=None, isftcflx: int | None = None,
                 history_interval_s: float | None = None,
                 nest_history_interval_s: float | None = None,
                 nest_km: float = 3., track_selection: dict | None = None,
                 model: str | None = None,
                 inner_nest: tuple[int, float] | None = None,
                 nest_cells: int | None = None,
                 quicklook_interval_s: float | None = None) -> dict:
    from woof import domain_wizard as dw
    from woof.companion_domains import VORTEX_PRESET_SOURCE
    from woof.configuration_recovery import MemoryAdmissionError
    from woof.cyclone_sources import (moving_nest_note, selected_member,
                                       source_adapter)
    from woof.starter_template import changes

    root_dx_m = float(nest_km) * 1000. * RATIO
    dw.check_fit_cancelled(cancelled)
    from woof.cyclone_sources import acquisition_hints
    model = acquisition_hints(forcing_source, model).get("model")
    adapter = source_adapter(forcing_source)
    forcing_source = adapter.source_id
    interval_s = _forcing_interval(forcing_source, cycle=cycle,
                                   start_hour=start_hour, hours=hours)
    # ONE intent, carried by every re-price on this door -- the requested
    # layout, each rung of the ladder, the unreduced admission probe and
    # the resident probe.  The source and member live in it for the same
    # reason the point and the hours do: a probe that dropped them would
    # be pricing a different configuration from the one being proposed.
    intent = dict(cycle=cycle, point=point, hours=hours, name=name, tiles=tiles,
                  source=source, forcing_source=forcing_source, member=member,
                  start_hour=start_hour, isftcflx=isftcflx,
                  history_interval_s=history_interval_s,
                  nest_history_interval_s=nest_history_interval_s,
                  nest_km=nest_km, track_selection=track_selection, model=model,
                  inner_nest=inner_nest, quicklook_interval_s=quicklook_interval_s)
    if nest_budget_gib is not None and (
            not isinstance(nest_budget_gib, (int, float))
            or isinstance(nest_budget_gib, bool)
            or not math.isfinite(float(nest_budget_gib))
            or float(nest_budget_gib) <= 0):
        raise ValueError(
            "--nest-budget-gib is the memory the sized tree may occupy and "
            "must be a finite positive size in GiB; omit it to size against "
            "the card's own available memory")
    # BEFORE anything is priced, because the budget is the allowance every
    # later number is measured against -- including the tile planner's.
    sizing, budget_bound = _budget_sizing(sizing, nest_budget_gib)
    # --nest-cells N: the following nest's side in its own cells, chosen
    # directly (the slider's size) rather than through a memory budget.
    # The requested layout is then priced and admitted like the preset's.
    if nest_cells is not None:
        if nest_budget_gib is not None:
            raise ValueError("--nest-cells and --nest-budget-gib both size the "
                             "following nest; give one")
        n = int(nest_cells)
        if n % RATIO or n < 4 * RATIO:
            raise ValueError(f"--nest-cells {n} must be a whole number of parent "
                             f"cells (a multiple of {RATIO}) of at least {4 * RATIO}")
        intent["dimensions"] = [list(ROOT_DIMS), [n, n]]
    # One hardware snapshot for the whole search. Never redetect/inflate VRAM
    # or force a different streaming mode to make a candidate appear to fit.
    if tiles != "off" and target_machine is None:
        from woof.core.streaming import planner_machine
        target_machine = planner_machine(vram_bytes=sizing.free_bytes,
                                         name="woof cyclone budget",
                                         device_profile=sizing.device_profile)
        if target_machine is None:
            # Same breakage and same way out the shared planner names ten
            # lines into dw._sizing_phases -- said once, in one wording,
            # and typed so it leaves plan_cyclone with a memory payload
            # instead of as an untyped ValueError with no resource.
            raise dw.DomainFitError(
                f"--tiles {tiles}: --tiles needs host RAM available to the "
                "shared planner; run the wizard on the forecast host or use "
                "--tiles off",
                resource="host")
        # The planner machine carries the card that was MEASURED, not the
        # reference card, so the four sites in woof/core/streaming.py that
        # fall back to machine.device_profile cannot price one card's
        # shader count as another's.
        #
        # MEASURED on an RTX 5070 Ti (70 SM, Linux, CUDA 13), on this
        # door's own 12/3 km cyclone tree, at two real card states --
        # 15.28 GiB free and 5.56 GiB free, the band where the tree road
        # binds, the second produced by holding 9.5 GiB on the card.  What was
        # measured: every number _sizing_phases produces (phase peak,
        # binding phase, the forecast non-pool/intercept/column-workspace/
        # fixed-envelope/subtotal/alloc terms, and the tree road's priced
        # flag, refusal, resource and streaming fixed floor), under three
        # planner machines identical but for this field: none, the measured
        # profile, and a deliberately absurd 999-SM profile.  All three are
        # byte-identical at both card states, including the refusal string
        # and the 6,978,986,310 B streaming fixed floor.  Calibration of
        # the instrument: the absurd arm is the positive control -- priced
        # through the estimator directly, 999 SMs move the same term from
        # 1,181,036,544 B to 7,214,964,736 B, so an arm that reached this
        # field could not have come back equal.
        #
        # So this line moves no admission threshold on this door:
        # _sizing_phases builds the resident estimate itself with
        # profile=sizing.device_profile and hands it to all four sites,
        # where machine.device_profile is only the fallback for a missing
        # estimate.  It is kept because that fallback, if a future caller
        # reaches it, must price this card -- 1,181,036,544 B here against
        # the reference card's 2,322,194,432 B, 1,088.3 MiB apart on the
        # one term shrinking the grid cannot move.  It is now set where the
        # machine is BUILT rather than patched on afterwards, because
        # preflight.admission_estimate takes the device from the machine
        # and from nowhere else, so a machine built without it is a
        # different envelope from the run door's on the same card.
    operands = dict(free_bytes=sizing.free_bytes, vram_gib=sizing.vram_gib,
                    profile=sizing.device_profile,
                    forcing_interval_seconds=interval_s)

    def price(exp):
        dw.check_fit_cancelled(cancelled)
        phases = dw._sizing_phases(exp, source=forcing_source,
                                   machine=target_machine, **operands)
        dw.check_fit_cancelled(cancelled)
        return phases

    # Whether a search chose the layout, and so held the fit headroom back;
    # the reduction road below is the other search.
    nest_searched = False
    if nest_budget_gib is not None:
        # The nest is chosen FIRST and the chosen layout is then the
        # requested one: everything downstream -- the admission, the
        # reduction road, the probes, the document -- reads one layout out
        # of the intent, so nothing here has a second idea of what was
        # asked for.
        try:
            chosen, nest_searched = _size_nest_to_budget(
                intent, sizing=sizing, target_machine=target_machine,
                operands=operands, price=price, cancelled=cancelled)
        except dw.DomainFitError as error:
            if error.resource not in {"vram", "host", "memory"}:
                raise
            floor_dims = _nest_dimensions(PRESET_NEST_PARENT_CELLS)
            _floor_text, floor_exp = configuration_text(
                **{**intent, "dimensions": floor_dims})
            floor_budget = dw.sizing_budget_bytes(floor_exp, **operands)
            # WHOSE refusal this is, read off the answer _budget_sizing
            # already gave rather than assumed.  A budget larger than the
            # card never bound anything: naming it as the bound sends the
            # reader to raise a flag that cannot move, and reports to a
            # form reading `bound_by` that the caller's own number refused
            # a run the hardware refused.
            card_bound = budget_bound == "card"
            alternative = (_unbudgeted_alternative(intent, sizing=sizing,
                                                   operands=operands)
                           if card_bound else None)
            raise MemoryAdmissionError(
                _nest_budget_floor_refusal(
                    floor_dims, budget=floor_budget,
                    cost=_floor_price(intent, price, floor_dims),
                    budget_bound=budget_bound,
                    requested_bytes=int(float(nest_budget_gib) * dw.GIB),
                    alternative=alternative,
                    fit_bound=_tile_road_bound(error)),
                reason="nest-floor-card" if card_bound else "nest-budget-floor",
                bound_by="card" if card_bound else "nest-budget",
                resource=error.resource, budget_bytes=floor_budget,
                keeps_coverage=None, resident_alternative=None,
                resident_fixed_floor_bytes=None,
                streaming_fixed_floor_bytes=None,
                unbudgeted_alternative=alternative) from error
        intent["dimensions"] = chosen
    baseline_text, baseline_exp = configuration_text(**intent)
    baseline_dims = _requested_dimensions(intent)
    track_grown = False
    track_growth_reason = None
    if track_selection is not None:
        from woof.cyclone_tracks import layout
        base_dims = _requested_dimensions(intent)
        _, needed = layout(point=point, fixes=track_selection["fixes"],
            dimensions=base_dims, dx_m=root_dx_m,
            clearance=_nest_clearance_cells(baseline_exp, base_dims[1]))
        if tuple(base_dims[0]) != tuple(needed):
            intent["dimensions"] = [needed, tuple(base_dims[1])]
            track_grown = True
    try:
        original_text, experiment = (configuration_text(**intent) if track_grown
                                     else (baseline_text, baseline_exp))
    except ValueError as error:
        if not track_grown:
            raise
        # Source/projection extent may stop parent growth before memory does.
        # Preserve the admitted preset and report the unavailable corridor.
        track_growth_reason = str(error)
        intent["dimensions"] = base_dims
        original_text, experiment = baseline_text, baseline_exp
        track_grown = False
    text = original_text
    requested_dims = _requested_dimensions(intent)

    dims_of = _reduction_dimensions(requested_dims)

    def scales_of(exp):
        return _fit_scales(exp, requested_dims)

    budget = dw.sizing_budget_bytes(experiment, **operands)

    # What stopped the SEARCH, carried out of the fitter rather than
    # re-derived from the emitted grid.  A cyclone fit is a POINT fit on
    # the shared bounded road, so the same bounds apply to it as to the
    # wizard's own point door: a high-latitude centre grows a 12 km parent
    # toward the projection pole and the fit shrinks off it, and a source
    # window can stop the search below what the card affords.  Without
    # this the door emitted a smaller grid than the budget allows and said
    # nothing about why -- the invisible saturation the plan summary
    # exists to prevent.  Reported on the document, which is this door's
    # only surface: its stdout is the JSON result and everything else is
    # redirected to stderr.
    fit_stop: dict = {}
    # The bounds the REQUEST carries, asked of the requested layout
    # itself (see below) as well as of every rung the fit tries.
    request_bound = dw.point_request_bound(
        tomllib.loads(original_text)["projection"], *requested_dims[0], root_dx_m)
    phases, refusal = _admitted_as_requested(experiment, price, budget,
                                             cancelled)

    if refusal is None and track_grown and phases.peak_envelope_bytes > budget - dw.fit_headroom_bytes(budget):
        refusal = dw.DomainFitError("The grown track corridor exceeds memory admission with fit headroom",
                                    resource="vram", phases=phases)
    if refusal is None and request_bound is not None:
        # The card is not the only thing that decides how big a domain
        # grown from a single point may be, and on this door it was the
        # only thing that did: the REQUESTED 200x160 was authored without
        # ever facing the bounds the fit below is bounded by.  Above
        # about 81 N that emitted a 12 km parent whose footprint contains
        # the projection pole -- where lat-lon source interpolation and
        # static-tile windowing are not pole-capable -- while the SAME
        # door, one gigabyte lower, refused a SMALLER pole-reaching rung
        # for exactly that reason.  One door, two answers, and the larger
        # one could not be prepared.  So the request faces the bound
        # first, and a request the bound stops is a reduction like any
        # other: the ladder shrinks off the pole and the proposal says
        # why.
        refusal = dw.DomainFitError(request_bound[1], resource="extent",
                                    phases=phases)

    reduced = refusal is not None
    memory_bound = reduced and refusal.resource in {"vram", "host", "memory"}
    keeps_coverage = None
    if reduced:
        # Asked BEFORE anything is proposed or refused, so both the
        # proposal's notice and the refusal's text can name it.  Only for
        # a MEMORY refusal AND only when the REQUESTED layout faces no
        # request bound of its own: `--tiles off` moves where the bytes
        # land, not where the domain sits, so offering it against a
        # request bound names a remedy that cannot help.
        #
        # Both tests, not either: a request can be bound BOTH ways at
        # once.  The memory refusal is raised first and leaves
        # `refusal.resource` reading `vram`, so the memory test alone
        # sent the UNREDUCED 200x160 through the estimator, saw it
        # admitted resident, and said so -- in the same notice that then
        # reported the projection had rejected a SMALLER root for
        # reaching the pole.  One notice, two contradictory claims, and
        # the named remedy did not do what it said: `--tiles off` at that
        # point and budget authors the reduced layout auto had already
        # proposed, because the pole bound stops the resident ladder in
        # exactly the same place.  Reproduced on the real door at
        # --point=82,-20 --vram-gib 8.45.  `request_bound` above is asked
        # of the requested dimensions, which is the layout this admission
        # would be naming, so it is the test that belongs here.
        keeps_coverage = _unreduced_resident_admission(
            intent,
            lambda exp: dw.sizing_budget_bytes(exp, **operands),
            lambda exp: dw._sizing_phases(exp, source=forcing_source,
                                          machine=target_machine, **operands)
        ) if memory_bound and request_bound is None else None
        # Measured once, on the refusal path only, and memoised: a refusal
        # has to name a way through, and the way through has to be a
        # layout somebody measured rather than a mode somebody suggested.
        probed: list = []

        def _probe(exhausted: bool) -> tuple[bool, dict | None]:
            # Nothing to measure when BOTH fixed floors exhaust the budget:
            # the resident ladder is then refused by its own floor before
            # any rung, so a search would cost prices to prove what the
            # floor already said.  `measured` stays False there because
            # this probe did not run -- but `exhausted` is itself the
            # measurement that the resident route admits nothing, and the
            # caller reads it directly.
            if not probed:
                probed.append(
                    (False, None) if keeps_coverage or exhausted else
                    _resident_alternative(intent, sizing, dims_of, scales_of))
            return probed[0]

        def _alternative(exhausted: bool):
            return _probe(exhausted)[1]

        def _way_through(exhausted: bool, floors: dict) -> str:
            if keeps_coverage:
                return f"  {_keeps_coverage_sentence(keeps_coverage, tiles)}."
            measured, found = _probe(exhausted)
            if found:
                return f"  {_resident_alternative_sentence(found)}."
            if measured and tiles != "off":
                return f"  {_no_resident_sentence(budget)}."
            if exhausted:
                # The refusal law's harder half: nothing was priced, so
                # nothing is offered, but what would move this IS known.
                return f"  {_fixed_floor_way_out(floors, budget, tiles)}."
            return ""

        impossible, floors = _fixed_floors(phases, budget, tiles)
        if impossible:
            raise MemoryAdmissionError(
                "The selected computer's fixed process/radiation costs exhaust its memory "
                "budget before grid storage; resizing cannot help with the selected physics "
                "and tile mode."
                + _way_through(True, floors),
                reason="fixed-floor", bound_by="computer", budget_bytes=budget,
                keeps_coverage=keeps_coverage,
                resident_alternative=_alternative(True), **floors)
        candidates = {}

        def build(dims):
            proposed_text, exp = configuration_text(**{**intent, "dimensions": dims})
            candidates[tuple(dims)] = proposed_text
            return exp

        try:
            dims, experiment = dw.fit_ladder(
                ratios=(RATIO,), free_bytes=sizing.free_bytes, vram_gib=sizing.vram_gib,
                device_profile=sizing.device_profile, target_machine=target_machine,
                hours=hours,
                start_time=(_cycle(cycle, forcing_source=forcing_source)
                            + timedelta(hours=start_hour)),
                projection=tomllib.loads(original_text)["projection"],
                source=forcing_source, name=_config_name(adapter, name, nest_km),
                root_dx_m=root_dx_m,
                profile=_default_profile(forcing_source, nest_km),
                tiles=tiles, forcing_interval_seconds=interval_s,
                candidate_builder=build, dimensions_builder=dims_of,
                candidate_scales=scales_of(experiment),
                layout_label="cyclone 12/3 km", cancelled=cancelled,
                stop_out=fit_stop)
            text = candidates[tuple(dims)]
            # Re-admit the final, fully authored moving tree, including the
            # wizard's headroom. No stale candidate or admission bypass.
            phases = price(experiment)
            budget = dw.sizing_budget_bytes(experiment, **operands)
            if phases.peak_envelope_bytes > budget - dw.fit_headroom_bytes(budget):
                raise dw.DomainFitError("The final cyclone proposal no longer fits with headroom",
                                        resource="vram", phases=phases)
        except dw.DomainFitError as error:
            if error.resource not in {"vram", "host", "memory"}:
                raise
            impossible, floors = _fixed_floors(error.phases or phases, budget, tiles)
            # WHOSE refusal this is.  "The selected computer cannot admit"
            # is a claim about HARDWARE, and it was false on exactly the
            # band this door serves: at 6 GiB the resident fixed floor sat
            # a gigabyte and a half under the budget and `--tiles off`
            # authored a real layout, so what refused was the tile
            # planner's tree road for `--tiles auto`, not the card.  A
            # refusal that misnames what refused sends the reader after
            # the wrong remedy -- a bigger card -- and hides the one that
            # works.  The hardware claim is made only when the fixed floor
            # the estimator itself reports actually exhausts the budget.
            resident_floor = floors.get("resident_fixed_floor_bytes")
            # And WHOSE it is not.  The floor comparison alone is an
            # INFERENCE -- "a resident tree could start here" -- and it
            # misattributed in the opposite direction on a band this door
            # reaches in ordinary use: budgets between the resident fixed
            # floor and the smallest rung's actual cost (roughly 3-5 GiB
            # free on this tree, which "measured-available" sizing hands a
            # busy 16 GiB card).  There the floor sits under the budget,
            # so the door blamed the tree road, while `--tiles off` on the
            # same card was refused too.  The measurement that settles it
            # is already computed: the resident ladder is WALKED on this
            # path, and when it admits nothing the computer is the bound.
            # So the tile planner is blamed only when a resident layout
            # was actually found -- either the unreduced request priced
            # resident (`keeps_coverage`) or a rung of the resident ladder
            # (`resident_alternative`).
            #
            # `--tiles off` has no tree road to blame, so a refusal there is
            # the resident route against the card and the original sentence
            # is the true one.  Only a TILED request can be refused by the
            # tile planner while the card still holds a resident tree.
            hardware_bound = impossible or tiles == "off" or not (
                resident_floor is not None and resident_floor < budget) or (
                keeps_coverage is None and _alternative(impossible) is None)
            # ONE SENTENCE, THEN THE WAY OUT.  What the reader needs is what
            # refused, with its numbers, and what to do; three further
            # sentences of policy behind them is a paragraph nobody finishes.
            # The clauses below carry the same facts the sentences did.
            if hardware_bound:
                head = "The selected computer cannot admit a cyclone proposal: "
                tail = ("; fixed process/radiation costs exhaust the budget, so resizing "
                        "cannot help." if impossible else
                        "; no smaller candidate in this bounded policy passed, and "
                        "physics, duration, halos and movement margins were not reduced.")
            else:
                head = f"No --tiles {tiles} cyclone proposal was admitted: "
                # UNDER `on`, THE MODE IS WHAT REFUSED.  "the tile planner's
                # tree road refusing" is true either way, but under `on` the
                # planner was asked for by the request itself, so naming the
                # planner sends the reader looking for a defect where there
                # is a flag.  Under `auto` the mode chose the planner and the
                # planner is the right name.
                tail = (("; no smaller candidate passed, and the requested "
                         "--tiles on is what compels the tiled road here, not "
                         "the computer: a resident tree's fixed floor is "
                         f"{resident_floor} bytes against a {budget} byte budget."
                         ) if tiles == "on" else
                        ("; no smaller candidate passed, and this is the tile planner's "
                         "tree road refusing, not the computer: its resident fixed floor "
                         f"is {resident_floor} bytes against a {budget} byte budget."))
            raise MemoryAdmissionError(
                head + str(error) + tail + _way_through(impossible, floors),
                reason="fixed-floor" if impossible else "bounded-search",
                bound_by="computer" if hardware_bound else f"tiles-{tiles}-tree-road",
                resource=error.resource, budget_bytes=budget,
                keeps_coverage=keeps_coverage,
                resident_alternative=_alternative(impossible), **floors) from error

    dw.check_fit_cancelled(cancelled)
    stopped_by = dict(fit_stop) if fit_stop else None
    review_required = reduced or track_grown
    comparison_text = baseline_text if track_grown else original_text
    changed_fields = [{"field": field, "before": before, "after": after}
                      for field, before, after in changes(tomllib.loads(comparison_text),
                                                        tomllib.loads(text))]
    result = {
        "schema": SCHEMA, "kind": "proposal" if review_required else "configuration",
        "cycle": _cycle(cycle, forcing_source=forcing_source).strftime("%Y%m%d%H"),
        "source": forcing_source, "member": selected_member(forcing_source, member),
        "forcing_interval_seconds": interval_s,
        # WHEN THE RUN BEGINS, in the two forms a reader needs: the lead
        # within the named cycle, and the wall-clock moment that lead is
        # valid at.  `cycle` alone stopped being the start time the moment
        # a run could begin at a lead, and a document that carried only
        # the cycle would have every downstream consumer re-deriving the
        # start from a flag it never saw.
        "forecast_start_hour": start_hour,
        "start_time": (_cycle(cycle, forcing_source=forcing_source)
                       + timedelta(hours=start_hour)).isoformat(sep=" "),
        "hours": hours, "point": list(point), "tiles": tiles,
        "domains": [{"grid_id": d.grid_id, "parent_id": d.parent_id,
                     "nx": d.run.nx, "ny": d.run.ny, "nz": d.run.nz,
                     "dx_m": d.run.dx, "dy_m": d.run.dy,
                     "following": d.grid_id == 2} for d in experiment.domains],
        # The table the emitted configuration carries, read off the
        # dimensions this proposal settled on, so the desktop reads the
        # numbers the toml beside it holds rather than the preset's
        # unfitted ones.
        "follow": {**follow_table_for_nest(*_following_nest_dims(experiment)),
                   **({"refine_grid_id": experiment.domains[1].follow.tracker.refine_grid_id}
                      if experiment.domains[1].follow.tracker.refine_grid_id is not None else {})},
        "follow_preset_source": VORTEX_PRESET_SOURCE,
        # THE NEST AS A DECISION, not just as two numbers in `domains`.
        # A reader sizing a card asks three questions of a following nest
        # -- how big it ended up, what that costs, and what stopped it
        # growing -- and all three are answers this door already computed.
        # Additive: every existing key means exactly what it meant, which
        # is why the schema version does not move for it.
        "nest": {
            "dimensions": [experiment.domains[1].run.nx,
                           experiment.domains[1].run.ny],
            "parent_cells": _nest_parent_cells(
                (experiment.domains[1].run.nx, experiment.domains[1].run.ny)),
            "floor_dimensions": list(CHILD_DIMS),
            "floor_parent_cells": PRESET_NEST_PARENT_CELLS,
            "ratio": RATIO,
            "budget_gib": nest_budget_gib,
            "budget_bytes": budget if nest_budget_gib is not None else None,
            "budget_bound_by": budget_bound,
            # What the admission of THIS layout held back: the fit headroom
            # when a search grew or shrank it, nothing when it was admitted
            # as requested -- the preset without the flag, or the floor
            # with it, which A175 made the same test.  It was the headroom
            # whatever chose the layout, so a preset admitted inside it
            # claimed bytes unspent that the envelope had spent.
            "headroom_bytes": (dw.fit_headroom_bytes(budget)
                               if review_required or nest_searched else 0),
            "peak_envelope_bytes": phases.peak_envelope_bytes,
            "sized_to_budget": nest_budget_gib is not None,
        },
        # Asked of the same table the run door resolves against, so this
        # document and the launch cannot disagree about whether the nest
        # this setup authors can be integrated on the selected chain.
        "follow_statics": moving_nest_note(
            forcing_source, follower_grid_ids=(FOLLOWER_GRID_ID,), point=point),
        "profile": _default_profile(forcing_source, nest_km),
        "streaming": _streaming_entry(phases, tiles, budget),
        "memory": {"peak_envelope_bytes": phases.peak_envelope_bytes, "budget_bytes": budget,
                   "binding_phase": phases.binding_phase, "free_bytes": sizing.free_bytes,
                   "fit_headroom_bytes": dw.fit_headroom_bytes(budget) if review_required else 0,
                   "sizing_basis": "measured-available" if sizing.measured else "declared-capacity"},
        "fitting": {"changed": review_required, "review_required": review_required,
                    "fit_id": hashlib.sha256(text.encode()).hexdigest(),
                    "original_dimensions": baseline_dims if track_grown else requested_dims,
                    "proposed_dimensions": [[d.run.nx, d.run.ny] for d in experiment.domains],
                    "changes": changed_fields,
                    "reason": str(refusal) if reduced else ("Parent growth reserves the tracked corridor" if track_grown else None),
                    "policy": "largest admitted 5%-scale rung; preserve tracker search and movement clearance",
                    "keeps_coverage": keeps_coverage,
                    "stopped_by": stopped_by,
                    "notice": (("Smaller geographic coverage proposed. Review dimensions and all field "
                                "changes; use --accept-fit FIT_ID to save this exact proposal."
                                + (f"  {_keeps_coverage_sentence(keeps_coverage, tiles)}."
                                   if keeps_coverage else "")
                                + (f"  {_stopped_by_sentence(stopped_by)}."
                                   if stopped_by else ""))
                               if reduced else
                               "Original dimensions fit; configuration unchanged.")},
        "config_text": text, "created": False, "forecast_started": False,
    }
    if track_selection is not None:
        from woof.cyclone_tracks import placement_report
        result["lineage"] = {"cyclone_tracks": track_selection["lineage"]}
        result["track_placement"] = placement_report(text, track_selection)
        result["track_placement"]["growth_limit"] = track_growth_reason
        if track_grown:
            result["fitting"]["notice"] = ("Parent dimensions changed to reserve the tracked corridor. "
                "Review all field changes; use --accept-fit FIT_ID to save this exact proposal.")
        if result["track_placement"]["leaves_follower_clearance"]:
            result["fitting"]["notice"] += (" Track corridor does not fit; parent is centered on the start fix. "
                "See track_placement.leaves_parent and leaves_follower_clearance for UTC fixes.")
    return result


def _cancelled(error) -> bool:
    """Is this the author cancelling a fit, rather than a fit failing?

    Resolved late because this module imports the domain wizard lazily,
    and asked of an exception that has already been raised, so the import
    only happens on a path that already loaded it.
    """

    from woof import domain_wizard as dw
    return isinstance(error, dw.DomainFitCancelled)


def main(args) -> int:
    from woof.earth2_licensing import licence_scope, accepted_licences
    with licence_scope((*accepted_licences(), *(getattr(args, "accept_licence", ()) or ()))):
        return _main(args)


def _main(args) -> int:
    try:
        with contextlib.redirect_stdout(sys.stderr):
            forcing_source = getattr(args, "source", None) or DEFAULT_SOURCE
            member = getattr(args, "member", None)
            start_hour = int(getattr(args, "start_hour", 0) or 0)
            if getattr(args, "list_sources", False):
                # A THIRD KIND, not a third schema.  `kind` already tells a
                # reader which document this is; a menu is one more value of
                # it, and every existing key on a map or a configuration is
                # untouched, so a v2 reader that does not know "sources"
                # skips it the way it skips any kind it did not ask for.
                from woof.cyclone_sources import source_options
                result = {"schema": SCHEMA, "kind": "sources",
                          "sources": source_options(),
                          # The sizing floor belongs on the menu because the
                          # menu is what a picker builds its form from: the
                          # nest a budget grows UP from, in both units, so a
                          # --nest-budget-gib field can be bounded before
                          # anything is priced.  The per-source lead ceiling
                          # rides on each row for the same reason.
                          "nest_floor": {
                              "dimensions": list(CHILD_DIMS),
                              "parent_cells": PRESET_NEST_PARENT_CELLS,
                              "parent_dimensions": list(ROOT_DIMS),
                              "ratio": RATIO},
                          "created": False, "forecast_started": False}
            elif args.latest_map:
                result = latest_map(args.cycle, source=forcing_source, member=member,
                                    start_hour=start_hour)
            else:
                from woof import domain_wizard as dw
                from woof.companion_query import inspect_configuration
                from woof.cyclone_seed import load_seed_fields, seed_cyclone
                from woof.cyclone_sources import companion_input_files
                from woof.hrrr_prepared_bundle import render_wps_namelist
                from woof.starter_template import _publish_new_files
                from woof.cyclone_sources import source_adapter, acquisition_hints
                model = acquisition_hints(forcing_source, getattr(args, "model", None)).get("model")
                forcing_source = source_adapter(forcing_source).source_id
                moment = _cycle(args.cycle, forcing_source=forcing_source)
                tracks_path = getattr(args, "tracks", None)
                storm = getattr(args, "storm", None)
                track_selection = None
                acquired_observations = False
                if tracks_path is None and (storm is not None or getattr(args, "search_area", None) is not None):
                    if args.point is not None or getattr(args, "advisory_position", None) or getattr(args, "seed_fields", None):
                        raise ValueError("track-position-conflict: name/area selection is mutually exclusive with point/advisory/seed fields")
                    from woof.cyclone_tracks import acquire_tracks
                    storm = storm or "auto"
                    acquired_observations = True
                    tracks_path = acquire_tracks(storm=storm,
                        start=moment + timedelta(hours=start_hour),
                        source=getattr(args, "track_source", "auto"),
                        cache=getattr(args, "track_cache", None))
                if tracks_path is not None:
                    if not storm:
                        raise ValueError("track-storm-required: --tracks requires --storm auto or TRACK_ID")
                    if args.point is not None or getattr(args, "advisory_position", None) or getattr(args, "seed_fields", None):
                        raise ValueError("track-position-conflict: --storm is mutually exclusive with --point, --advisory-position and --seed-fields")
                    from woof.cyclone_tracks import select_track
                    from woof.cyclone_sources import source_options
                    track_selection = select_track(tracks_path, storm=storm, member=None if acquired_observations else member,
                        source=forcing_source, start=moment + timedelta(hours=start_hour), hours=args.hours,
                        search_area=getattr(args, "search_area", None))
                    # A trajectory identity need not be an atmospheric member token.
                    # Use it only when this same source declares it in its menu.
                    mid = track_selection["lineage"]["member_id"]
                    row = next(r for r in source_options() if r["source"] == forcing_source)
                    provenance_source = track_selection["lineage"]["member"].get("source")
                    if not acquired_observations:
                        member = mid if provenance_source == forcing_source and mid in row["members"] else None
                seed_path = getattr(args, "seed_fields", None)
                fields = (load_seed_fields(seed_path, source=forcing_source,
                                           cycle=args.cycle, member=member)
                          if seed_path else None)
                advisory = getattr(args, "advisory_position", None)
                # WHERE THE CENTER COMES FROM is one function with a stated
                # fallback chain -- an explicit point, then the source's own
                # declared fields, then the advisory -- and it reports which
                # rung answered.  A missing center is the only outcome that
                # stops here, and it names both flags that supply one.
                seed = seed_cyclone(
                    source=forcing_source, fields=fields,
                    point=((track_selection["lineage"]["center_fix"]["lat"], track_selection["lineage"]["center_fix"]["lon"])
                           if track_selection is not None else
                           dw._parse_point(args.point) if args.point is not None else None),
                    advisory=(dw._parse_point(advisory) if advisory is not None
                              else None),
                    search_radius_km=getattr(args, "seed_radius_km", 500.))
                if seed.point is None:
                    raise ValueError("; ".join(seed.messages))
                sizing, machine, _ = dw._domain_target_hardware(args)
                out = args.out.expanduser().resolve() if args.out else None
                result = plan_cyclone(cycle=args.cycle, point=seed.point,
                    hours=args.hours, name=args.name, tiles=args.tiles, sizing=sizing,
                    target_machine=machine, source=str(out or "cyclone-setup.toml"),
                    forcing_source=forcing_source, member=member,
                    start_hour=start_hour,
                    nest_budget_gib=getattr(args, "nest_budget_gib", None),
                    isftcflx=getattr(args, "isftcflx", None),
                    history_interval_s=getattr(args, "history_interval", None),
                    nest_history_interval_s=getattr(args, "nest_history_interval", None),
                    nest_km=getattr(args, "nest_km", 3.), track_selection=track_selection, model=model,
                    inner_nest=getattr(args, "inner_nest", None),
                    nest_cells=getattr(args, "nest_cells", None),
                    quicklook_interval_s=getattr(args, "quicklook_interval", None))
                result["seed"] = seed.to_dict()
                if result.get("track_placement", {}).get("leaves_follower_clearance"):
                    print("warning: " + result["fitting"]["notice"])
                if track_selection is not None:
                    result["seed"]["method"] = "track"
                    result["seed"]["diagnostics"] = track_selection["lineage"]["interpolation"]
                # THE PLAN, on the human channel, in one line: where the run
                # begins, how big the following nest ended up and what the
                # priced tree costs against the budget that admitted it.
                # The document carries the same numbers for a machine; this
                # is the reader who is about to decide whether to save it.
                nest = result["nest"]
                print(f"plan: start f{result['forecast_start_hour']:03d} "
                      f"({result['start_time']} UTC), "
                      f"following nest {nest['dimensions'][0]}x"
                      f"{nest['dimensions'][1]} "
                      f"({nest['parent_cells']} parent cells), priced "
                      f"{nest['peak_envelope_bytes']} bytes against a "
                      f"{result['memory']['budget_bytes']} byte budget")
                # ONCE, on the human channel, where the reader who chose
                # the source can still change it.  The document carries
                # the same sentence for a machine, and the run door
                # raises it as its refusal; nothing here is a second
                # derivation of the fact.
                if not result["follow_statics"]["integrates_moving_nest"]:
                    print("warning: " + result["follow_statics"]["note"])
                acceptance = getattr(args, "accept_fit", None)
                if acceptance is not None and acceptance != result["fitting"]["fit_id"]:
                    raise ValueError("The reviewed fit does not match this proposal; review the new proposal before saving")
                if acceptance is not None:
                    result["fitting"]["review_required"] = False
                    result["kind"] = "configuration"
                if out is not None and not result["fitting"]["review_required"]:
                    if out.suffix.lower() != ".toml":
                        raise ValueError("Save the cyclone configuration as a new .toml file")
                    text = result["config_text"]
                    experiment = dw.experiment_from_text(text, source=str(out))
                    receipt = out.with_suffix(".cyclone.json")
                    # A source whose preparation recipe names a companion
                    # table writes it beside the configuration, so the
                    # never-overwrite rule has to cover that file too.
                    inputs = companion_input_files(forcing_source, out)
                    # The namelist carries the SELECTED source's interval,
                    # passed as the number it is rather than patched into
                    # the rendered string afterwards.
                    wps_text = render_wps_namelist(
                        experiment,
                        interval_seconds=result["forcing_interval_seconds"])
                    # This door authors on any planable source, the native
                    # regional one included, and that route reads namelists
                    # beside the configuration: written short, the
                    # configuration this door just authored is refused at
                    # the prepare precheck before anything starts. Asked
                    # through the one helper every publishing door asks,
                    # with the published configuration's own [fetch].source,
                    # so the files it gets and the files its run reads are
                    # decided by one function.
                    from woof.hrrr_route_inputs import candidate_companions
                    companions = candidate_companions(
                        out, experiment, wps_text=wps_text,
                        source=(tomllib.loads(text).get("fetch") or {}).get("source"))
                    if any(path.exists()
                           for path in (out, receipt, *(p for p, _ in inputs),
                                        *(p for p, _ in companions))):
                        raise ValueError("Choose a new output path; cyclone setup never overwrites an existing configuration")
                    proof = {key: value for key, value in result.items() if key != "config_text"}
                    proof.update(created=True, output=str(out), output_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                 wps_sha256=hashlib.sha256(wps_text.encode()).hexdigest(),
                                 route_companions=[str(path) for path, _ in companions])
                    out.parent.mkdir(parents=True, exist_ok=True)
                    _publish_new_files((*inputs, *companions,
                        (receipt, json.dumps(proof, indent=2, allow_nan=False) + "\n"), (out, text)))
                    result["configuration"] = inspect_configuration(out)
                    result.update(created=True, config_path=str(out), receipt_path=str(receipt))
        print(json.dumps(result, allow_nan=False, default=str))
        return 0
    except KeyboardInterrupt:
        print(json.dumps({"schema": SCHEMA, "error": "Cyclone fitting cancelled",
                          "cancelled": True, "created": False, "forecast_started": False}))
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        # ONE cancellation answer for both seams.  The predicate path
        # raises DomainFitCancelled, which IS a RuntimeError, so this
        # handler used to turn a cancelled fit into exit 1 with no
        # `cancelled` key -- indistinguishable from a fit that failed by
        # the only two things a caller reads, the exit code and that key.
        if _cancelled(error):
            print(json.dumps({"schema": SCHEMA, "error": "Cyclone fitting cancelled",
                              "cancelled": True, "created": False,
                              "forecast_started": False}))
            return 130
        result = {"schema": SCHEMA, "error": str(error), "created": False,
                  "forecast_started": False}
        if hasattr(error, "memory"):
            result["memory"] = error.memory
        print(json.dumps(result, allow_nan=False))
        return 1


def register_cli(subparsers):
    from woof.cli_numbers import positive_float, positive_float_list, positive_int
    parser = subparsers.add_parser("cyclone-setup", help="select a cyclone on any planable source's chosen cycle and lead and author a 12/3 km following nest")
    from woof.earth2_licensing import add_licence_argument
    add_licence_argument(parser)
    # NO `choices=`, deliberately, and for the same reason the domain
    # wizard's --source carries none: the admissible set is the registry
    # intersected with the fetch routes, it grows by a table row, and an
    # argparse list would answer a valid source with "invalid choice"
    # while the door itself could plan it.  A source with no route is
    # refused by cyclone_sources, with the menu flag named.
    parser.add_argument("--source", default=DEFAULT_SOURCE, metavar="SOURCE",
                        help=f"forcing source to initialize from (default {DEFAULT_SOURCE}); "
                             "--list-sources prints the planable set")
    parser.add_argument("--member", metavar="MEMBER",
                        help="track member id with --tracks; otherwise member in the source route grammar")
    parser.add_argument("--model", metavar="MODEL", help="optional model selection for a registry acquisition owner")
    parser.add_argument("--list-sources", action="store_true",
                        help="emit the planable sources with their members, cycle hours, "
                             "forcing interval and coverage envelope")
    parser.add_argument("--latest-map", action="store_true")
    parser.add_argument("--cycle", default="latest")
    parser.add_argument("--point")
    parser.add_argument("--track-source", choices=("auto", "nhc", "hurdat2", "ibtracs", "jtwc"), default="auto",
                        help="observed best-track archive used when no --tracks file is supplied")
    parser.add_argument("--track-cache", type=Path, help="cache for automatic best-track acquisition")
    parser.add_argument("--tracks", type=Path, metavar="FILE", help="track JSON or CSV guidance for selection and placement only")
    parser.add_argument("--storm", metavar="auto|ID|NAME", help="strongest eligible system at the cycle, or a track id, ATCF id or storm name")
    from woof.cyclone_search import parse_area
    parser.add_argument("--search-area", type=parse_area, metavar="S,W,N,E",
                        help="limit track selection by position at the cycle within these bounds")
    parser.add_argument("--nest-km", type=float, choices=(3., 2.5), default=3., help="nest spacing; 2.5 km authors a 10 km parent at ratio 4")
    parser.add_argument("--seed-fields", type=Path, metavar="NPZ",
                        help="canonical source-analysis arrays carrying that source's own "
                             "cycle and member identity, to locate the center from")
    parser.add_argument("--advisory-position", metavar="LAT,LON",
                        help="advisory center; bounds the field search and is the last fallback")
    parser.add_argument("--seed-radius-km", type=positive_float, default=500.,
                        help="how far from the advisory position the field search may look")
    parser.add_argument("--start-hour", type=int, default=0, metavar="N",
                        help="forecast lead of the selected cycle to begin at "
                             "(default 0, the analysis); the run initialises "
                             "from fN and is forced from fN onward at the "
                             "source's own cadence")
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--isftcflx", type=int, choices=(0, 1, 2), default=None,
                        help="surface flux over water on both grids (WRF "
                             "isftcflx), as `woof domain --isftcflx`; "
                             "default: the suite's own (0)")
    parser.add_argument("--history-interval", type=positive_float, default=None, metavar="SECONDS",
                        help="how often the 12 km parent writes a wrfout (default 3600)")
    parser.add_argument("--quicklook-interval", type=positive_float, default=None, metavar="SECONDS",
                        help="a quick-look subset on the finest domain every SECONDS between its full frames "
                             "(quicklook/wrfout_dNN_*); must divide that domain's history interval")
    parser.add_argument("--nest-cells", type=positive_int, default=None, metavar="N",
                        help="the following nest's side in its own cells (a multiple of the parent ratio), "
                             "instead of the preset or --nest-budget-gib")
    parser.add_argument("--inner-nest", type=parse_inner_nest, default=None, metavar="RATIO,KM",
                        help="a third domain centred in the following nest and moving with it: "
                             "RATIO to the following nest (3 turns 3 km into 1 km) and its side in km")
    parser.add_argument("--nest-history-interval", type=positive_float_list, default=None, metavar="SECONDS[,SECONDS]",
                        help="how often the following nest writes a wrfout (default 900); a "
                             "longer interval is how a multi-day run fits its disk")
    parser.add_argument("--name", help="configuration name (default: the selected source's own title)")
    parser.add_argument("--tiles", choices=("off", "auto", "on"), default="auto")
    parser.add_argument("--hardware-json", type=Path)
    parser.add_argument("--target-host-memory-json", type=Path)
    parser.add_argument("--vram-gib", type=float)
    parser.add_argument("--nest-budget-gib", type=float, metavar="GIB",
                        help="grow the following nest, square and in whole "
                             "parent cells, to the largest the priced tree "
                             "holds inside this much memory; the parent is "
                             "unchanged and the preset nest is the floor")
    parser.add_argument("--card", help="a tier (12gb/16gb/24gb/32gb), a size "
                        "('10gb') or a model with a recorded size ('RTX 3080')")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--accept-fit", metavar="FIT_ID",
                        help="save only the exact reviewed proposal identified by fitting.fit_id")
    parser.add_argument("--json", action="store_true", help="emit the JSON result (the default)")
    parser.set_defaults(func=main)
