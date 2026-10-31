"""What a run downloads and what its preparation writes, priced before either starts.

A run's disk holds four things: the source files it downloads, the files
its preparation writes from them, its history files and checkpoints, and
its pictures.  :mod:`woof.disk_budget` projects the last two from the
grid.  This module prices the first two, from sizes measured on real
downloads and real preparations, kept as rows of the packaged table
``woof/data/download-bytes.v1.json``:

* **The download** is one figure per plan: the objects the fetch will
  request, each priced at the size measured for its source, byte
  transport and role.  The objects are counted the way the fetch counts
  them, by the fetch's own functions: the table route's resolved object
  list, or the lead ladder of a transport that predates the route table.
  A source whose transport crops to an area (the NOMADS grib filter, the
  ERA5 retrieval) is priced per grid point of the requested area.  A
  route that composes files after the transfer (a GEFS a/b pair, a GDPS
  valid time) keeps the objects beside the composed copy, and both are
  priced.  The full-file default stays the default; it is priced, not
  replaced.
* **The preparation** is priced per cell of the grids, from the bytes a
  measured preparation of the same chain wrote: per cell and forcing
  time on the outer grid (the forcing series the preparation writes for
  every time it was given), plus per cell once (the initial state and
  the static fields), plus per cell of every nest.
* **What is already on disk** is the files a fetch receipt of the same
  request names in the folder the download lands in, and nothing else
  in that folder (:func:`present_bytes`).

The transport priced is the one the fetch will take, asked through the
fetch's own functions, including the switch to whole archive objects
for a GFS or GDAS cycle the grib-filter host no longer keeps.

Adding a source is a row in that table, never a function here: the
arbitrary acceptance test.  A source or transport with no row is
reported as unpriced, by name, rather than priced at a guess.

Both figures were left out of every disk projection until 2026-09-26: an
18 hour HRRR run downloaded 22 GB that no estimate showed, and an event
page layout projected at 15.2 GiB had written 28.9 GB by its first
forecast step, 18.9 GB of it downloads and 9.6 GB preparation.
"""

from __future__ import annotations

import functools
import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

#: The packaged table this module reads.
TABLE_NAME = "download-bytes.v1.json"
TABLE_SCHEMA = "gpuwm-download-bytes-v1"

#: The keys of a request (a ``[fetch]`` table, or a ``woof fetch`` argv
#: read back into one) that decide what is downloaded.
REQUEST_KEYS = ("source", "cycle", "hours", "area", "point", "radius_km",
                "cadence", "forecast_start_hour", "source_root",
                "era5_provider", "era5_product", "member", "mode",
                "transport", "p_top_pa", "all_levels", "out")


def table_path() -> Path:
    return Path(__file__).with_name("data") / TABLE_NAME


@functools.lru_cache(maxsize=1)
def table() -> Mapping[str, Any]:
    """The packaged table, checked for its schema once per process."""

    document = json.loads(table_path().read_text(encoding="utf-8"))
    if document.get("schema") != TABLE_SCHEMA:
        raise ValueError(f"{TABLE_NAME} is not a {TABLE_SCHEMA} document")
    return document


def request_from_arguments(arguments: Sequence[str]) -> dict[str, Any] | None:
    """The ``[fetch]`` table spelling of one ``woof fetch`` argv, or None.

    Read by woof's own fetch parser, so there is no second copy of the
    flag table here.  None when the parser refuses the list; the fetch
    would refuse it too, and says why when it runs.
    """

    from woof.cli import parse_fetch_arguments

    try:
        args = parse_fetch_arguments(arguments)
    except SystemExit:
        return None
    request: dict[str, Any] = {}
    for key in REQUEST_KEYS:
        value = getattr(args, key, None)
        if value is None or value is False:
            continue
        request[key] = str(value) if isinstance(value, Path) else value
    return request


def _canonical(source: str) -> str:
    from woof import fetch_routes

    return fetch_routes.canonical_source(str(source))


def _legacy(source: str) -> Mapping[str, Any] | None:
    return (table().get("legacy_sources") or {}).get(source)


def _archive_mode(source: str, request: Mapping[str, Any]) -> str | None:
    """The transport an unnamed mode takes for a cycle only the archive still holds, or None.

    The fetch asks the same question, through the same function: a GFS or
    GDAS cycle older than the grib-filter host keeps is read as whole
    objects from the archive when no mode was named.  Pricing that
    request as a crop put a 4.6 GB download at 34 MB.
    """

    legacy = _legacy(source)
    if legacy is None or not legacy.get("archive_mode"):
        return None
    if _named_mode(source, legacy, request):
        return None
    cycle = _cycle(request, source)
    if cycle is None:
        return None
    from woof import fetch

    return str(legacy["archive_mode"]) if fetch.archive_only_cycle(source, cycle) else None


def _named_mode(source: str, legacy: Mapping[str, Any],
                request: Mapping[str, Any]) -> str | None:
    """The transport ``request`` names itself, or None for the default.

    ``auto`` on a source whose table has no row of that name asks for the
    default choice, which is what the fetch then takes; pricing it as a
    mode of its own left the request unpriced.
    """

    value = request.get(str(legacy.get("mode_key") or "mode"))
    if not value:
        return None
    if value == "auto" and _row(source, "auto") is None:
        return None
    return str(value)


def _mode(source: str, request: Mapping[str, Any]) -> str:
    """The byte transport the fetch will use for ``request``."""

    legacy = _legacy(source)
    if legacy is None:
        from woof import fetch_routes

        return fetch_routes.resolve_mode(source, request.get("mode"))
    value = _named_mode(source, legacy, request)
    if value:
        return value
    return _archive_mode(source, request) or str(legacy["default_mode"])


def _row(source: str, mode: str) -> Mapping[str, Any] | None:
    rows = [row for row in table().get("downloads") or ()
            if row.get("source") == source and row.get("mode") == mode]
    if not rows:
        return None
    row = rows[0]
    if row.get("same_as"):
        base = _row(source, str(row["same_as"]))
        if base is None:
            return None
        return dict(base, mode=mode, priced_as=str(row["same_as"]),
                    why=str(row.get("why") or ""))
    return row


def _cycle(request: Mapping[str, Any], source: str) -> datetime | None:
    raw = request.get("cycle")
    if raw is None or str(raw).strip().lower() == "latest":
        return None
    if isinstance(raw, datetime):
        return raw.replace(tzinfo=None)
    from woof import fetch

    return fetch.parse_cycle(str(raw)[:13], source)


def _int(request: Mapping[str, Any], key: str, default: int | None) -> int | None:
    value = request.get(key)
    if value is None:
        return default
    return int(value)


def _area(request: Mapping[str, Any]):
    """The request's area, resolved the way the fetch resolves it, or None."""

    from woof import fetch

    radius = request.get("radius_km")
    return fetch._resolve_area(SimpleNamespace(
        area=request.get("area"), point=request.get("point"),
        radius_km=None if radius is None else float(radius)))


def _points(request: Mapping[str, Any], grid_deg: float, convention: str) -> int:
    """Grid points of a ``grid_deg`` lattice inside the request's area."""

    area = _area(request)
    if area is None:
        raise ValueError("this transport crops to an area and the request names none")
    if convention == "nomads":
        box = area.as_nomads()
        south, north = box["bottom_lat"], box["top_lat"]
        west, east = box["left_lon"], box["right_lon"]
    else:
        south, north = area.lat_south, area.lat_north
        west = area.lon_west
        east = west + area.longitude_span_degrees

    def count(low: float, high: float) -> int:
        return max(1, math.floor(high / grid_deg + 1e-6) - math.ceil(low / grid_deg - 1e-6) + 1)

    return count(south, north) * count(west, east)


def _rows(request: Mapping[str, Any], grid_deg: float) -> int:
    """Latitude rows of a ``grid_deg`` lattice inside the request's area."""

    area = _area(request)
    if area is None:
        raise ValueError("this transport crops to an area and the request names none")
    return max(1, math.floor(area.lat_north / grid_deg + 1e-6)
               - math.ceil(area.lat_south / grid_deg - 1e-6) + 1)


def _planned_objects(source: str, request: Mapping[str, Any], row: Mapping[str, Any]
                     ) -> tuple[list[tuple[str, int | None]], int, list[tuple[str, int]],
                                list[tuple[str, ...]]]:
    """Every object the fetch will request, as (role, lead), its lead count, its donors and its composed files.

    The last holds, for each composed file a table route writes after
    the transfer (a GEFS a/b pair, a GDPS valid time), the roles of the
    objects copied into it.  The fetch keeps those objects and writes
    the composed copy beside them, so each part is on disk twice.
    """

    from woof import fetch, fetch_routes

    hours = _int(request, "hours", None)
    if hours is None:
        raise ValueError("the request names no hours")
    start = _int(request, "forecast_start_hour", 0) or 0
    cadence = _int(request, "cadence", None)
    cycle = _cycle(request, source)
    legacy = _legacy(source)
    if legacy is None:
        route = fetch_routes.route_for(source)
        stand_ins = ([cycle] if cycle is not None else
                     fetch_routes.planning_cycles(route))
        refusal: Exception | None = None
        for when in stand_ins:
            try:
                plan = fetch_routes.resolve_request(
                    source, cycle=when, hours=hours, cadence=cadence,
                    start_hour=start, member=request.get("member"))
            except ValueError as error:
                refusal = error
                continue
            objects = [(obj.role, obj.lead) for obj in plan.objects]
            donors = [(_canonical(donor.source), len(donor.leads), bool(donor.records))
                      for donor in plan.donors]
            composed = [tuple(part.role for part in step.parts) for step in plan.compose]
            return objects, len(plan.leads), donors, composed
        raise refusal or ValueError("the route resolves no object")
    rule = str(legacy["leads"])
    if rule == "hourly-window":
        leads = (tuple(range(start, start + hours + 1)) if cycle is None
                 else fetch.hrrr_forecast_hours(hours, cycle, start))
    elif rule == "container-ladder":
        leads = fetch.container_forecast_hours(source, hours, cadence, start)
    elif rule == "analysis-times":
        leads = tuple(range(len(fetch._era5_times(
            cycle or datetime(2001, 1, 1), hours, cadence or 6))))
    else:
        raise ValueError(f"{TABLE_NAME} names an unknown lead rule {rule!r} for {source}")
    roles = list((row.get("objects") or row.get("objects_per_point") or {}).keys())
    return [(role, lead) for lead in leads for role in roles], len(leads), [], []


def _record_scale(source: str, request: Mapping[str, Any], row: Mapping[str, Any]
                  ) -> tuple[float, str | None]:
    """How much larger this request's objects are than the measured ones.

    A crop's size follows the records it selects, and the records follow
    the request's pressure ladder: a model top above the certified ladder
    adds five records per level.  A row that says how many records its
    measurement selected (``measured_records``) is scaled by the records
    this request selects, counted by the fetch's own function for the
    source's record rule; any other row is priced as measured.
    """

    measured = row.get("measured_records")
    rule = (_legacy(source) or {}).get("records")
    if not measured or rule is None:
        return 1.0, None
    if rule != "container-subset":
        raise ValueError(f"{TABLE_NAME} names an unknown record rule {rule!r} for {source}")
    from woof import fetch

    records = fetch.container_subset_record_count(
        source, top_pressure_pa=request.get("p_top_pa"),
        all_levels=bool(request.get("all_levels")))
    if records == int(measured):
        return 1.0, None
    return records / float(measured), (
        f"{records} records per file against the {int(measured)} measured, "
        "scaled per record")


def _unpriced(source: str | None, mode: str | None, why: str, **extra) -> dict[str, Any]:
    return {"bytes": None, "transfer_bytes": None, "objects": None, "leads": extra.get("leads"),
            "source": source, "mode": mode, "basis": why}


def download_estimate(request: Mapping[str, Any] | None) -> dict[str, Any]:
    """What downloading ``request`` leaves on disk and moves over the network.

    ``request`` is a ``[fetch]`` table, or a ``woof fetch`` argv read
    back by :func:`request_from_arguments`.  The answer carries
    ``bytes`` (left on disk), ``transfer_bytes`` (moved over the network,
    more than ``bytes`` for a source read in compressed chunks or chunk
    blocks and cropped locally), ``objects``, ``leads`` (the forcing times), ``source``,
    ``mode`` and a ``basis`` in words.  ``bytes`` is None only when the
    table has no measured size for the source and transport, or the
    fetch itself would refuse the request, and the basis says which.
    """

    if not request or not request.get("source"):
        return {"bytes": 0, "transfer_bytes": 0, "objects": 0, "leads": None,
                "source": None, "mode": None,
                "basis": "this run downloads nothing: it has no fetch request"}
    if request.get("source_root"):
        leads = None
        try:
            source = _canonical(str(request["source"]))
            leads = _planned_objects(source, request, _row(source, _mode(source, request)) or {})[1]
        except (ValueError, KeyError):
            source = str(request["source"])
        return {"bytes": 0, "transfer_bytes": 0, "objects": 0, "leads": leads,
                "source": source, "mode": "local",
                "basis": "the inputs are read from source_root on this computer; nothing is downloaded"}
    try:
        source = _canonical(str(request["source"]))
        mode = _mode(source, request)
    except ValueError as error:
        return _unpriced(str(request.get("source")), request.get("mode"),
                         f"the fetch refuses this request: {error}")
    row = _row(source, mode)
    try:
        objects, leads, donors, composed = _planned_objects(source, request, row or {})
    except ValueError as error:
        return _unpriced(source, mode, f"the fetch refuses this request: {error}")
    if row is None:
        return _unpriced(source, mode,
                         f"no measured size for {source} ({mode}) in woof/data/{TABLE_NAME}, "
                         "so the download is left out of this figure", leads=leads)
    fixed = row.get("objects") or {}
    per_point = row.get("objects_per_point") or {}
    transfer = row.get("transfer_per_object") or {}
    points = None
    if per_point:
        try:
            points = _points(request, float(row["grid_deg"]), str(row.get("area") or "request"))
        except ValueError as error:
            return _unpriced(source, mode, f"the fetch refuses this request: {error}", leads=leads)
    # A store read by the compressed blocks that hold the window's latitude
    # rows (rw_zarr over blosc Zarr chunks) moves that share of each whole
    # chunk: the window's rows plus, on average, one block's worth more.
    rows_share = 1.0
    window_rows = row.get("transfer_rows")
    if window_rows and per_point:
        rows = _rows(request, float(row["grid_deg"])) + int(window_rows.get("margin_rows", 0))
        rows_share = min(1.0, (rows + float(window_rows["rows_per_block"]))
                         / float(window_rows["grid_rows"]))
    try:
        scale, scale_words = _record_scale(source, request, row)
    except ValueError as error:
        return _unpriced(source, mode, f"the fetch refuses this request: {error}", leads=leads)

    def size_of(role: str) -> float | None:
        if role in fixed:
            return float(fixed[role]) * scale
        if role in per_point:
            return float(per_point[role]) * points * scale
        return None

    disk = moved = 0.0
    unpriced_roles = set()
    for role, _lead in objects:
        size = size_of(role)
        if size is None:
            unpriced_roles.add(role)
            continue
        disk += size
        moved += float(transfer[role]) * rows_share if role in transfer else size
    # A composed file is a copy of objects already counted: it takes disk
    # and moves nothing.  Leaving it out priced a GEFS download at half
    # of what it left on disk.
    parts = [role for step in composed for role in step]
    for role in parts:
        size = size_of(role)
        if size is None:
            unpriced_roles.add(role)
            continue
        disk += size
    if unpriced_roles:
        return _unpriced(source, mode,
                         f"no measured size for the {', '.join(sorted(unpriced_roles))} objects of "
                         f"{source} ({mode}) in woof/data/{TABLE_NAME}", leads=leads)
    donor_words = []
    for donor_source, donor_leads, donor_records in donors:
        # A donor that takes its bound records by index is priced as those
        # records, not as the whole object it reads them from.
        donor_row = _row(donor_source, "donor-records" if donor_records else "full-file")
        if donor_row is None or not donor_row.get("objects"):
            return _unpriced(source, mode, f"no measured size for the {donor_source} donor this "
                             f"route also downloads, in woof/data/{TABLE_NAME}", leads=leads)
        size = sum(float(v) for v in donor_row["objects"].values()) * donor_leads
        disk += size
        moved += size
        donor_words.append(f"{donor_leads} {donor_source} donor "
                           f"{'record file' if donor_records else 'file'}(s)")
    count = len(objects) + sum(donor[1] for donor in donors)
    what = f"{len(objects)} object(s) over {leads} forcing time(s)"
    if points is not None:
        what += f", cropped to {points:,} points of the {row['grid_deg']} degree grid"
    if scale_words is not None:
        what += f", {scale_words}"
    if rows_share < 1.0:
        what += (f", over the network the compressed blocks holding its latitude rows, "
                 f"{rows_share:.0%} of each whole chunk")
    if donor_words:
        what += " and " + ", ".join(donor_words)
    basis = (f"{what}, each at the size measured for {source} ({row.get('priced_as') or mode}) "
             f"in woof/data/{TABLE_NAME}")
    if row.get("priced_as"):
        basis += f"; {mode} is priced as {row['priced_as']}: {row.get('why')}"
    if composed:
        basis += (f"; the fetch also writes {len(composed)} composed file(s) from {len(parts)} "
                  "of those objects and keeps the objects, so those are counted twice on disk")
    if _archive_mode(source, request):
        basis += (f"; the cycle is older than the {_legacy(source)['default_mode']} host keeps, "
                  f"so the fetch switches to {mode} and reads the archive")
    return {"bytes": int(round(disk)), "transfer_bytes": int(round(moved)), "objects": count,
            "leads": leads, "source": source, "mode": mode, "basis": basis,
            "points": points}


def _receipts(directory: Path):
    """Every fetch receipt in ``directory``, as (recorded request, [(file it names, its lead)]).

    The lead is the forecast hour or route lead the receipt gives the
    file, or None for a file of no one lead (a static field, a checksum
    list, an ERA5 window).
    """

    from woof import era5_acquisition, era5_arco, fetch, fetch_routes

    def load(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def names(entries, *keys: str) -> list[tuple[str, Any]]:
        found = []
        for entry in entries if isinstance(entries, list) else ():
            if isinstance(entry, Mapping):
                name = next((entry.get(key) for key in keys if entry.get(key)), None)
                # A receipt names files inside its own folder, nothing else.
                if name and not Path(str(name)).is_absolute() and ".." not in Path(str(name)).parts:
                    lead = entry.get("forecast_hour", entry.get("lead"))
                    found.append((str(name), lead))
        return found

    manifest = load(directory / fetch.FETCH_MANIFEST_NAME)
    if isinstance(manifest, Mapping):
        if manifest.get("schema") == fetch.FETCH_MANIFEST_SCHEMA:
            recorded = {key: manifest.get(key) for key in ("source", "cycle", "area")}
            yield recorded, names(manifest.get("files"), "name")
        elif manifest.get("schema") == fetch_routes.ROUTE_MANIFEST_SCHEMA:
            recorded = manifest.get("request")
            if isinstance(recorded, Mapping):
                yield recorded, (names(manifest.get("files"), "relpath", "name")
                                 + names(manifest.get("composed"), "name"))
    # An interrupted table-route fetch records each object it verified.
    recovery = directory / fetch_routes._RECOVERY_DIRECTORY
    if recovery.is_dir():
        for path in sorted(recovery.glob("*.json")):
            record = load(path)
            if isinstance(record, Mapping) and isinstance(record.get("request"), Mapping):
                yield record["request"], names([record.get("file")], "relpath", "name")
    for module in (era5_arco, era5_acquisition):
        receipt = load(directory / module._RECEIPT)
        if (isinstance(receipt, Mapping) and receipt.get("schema") == module._SCHEMA
                and isinstance(receipt.get("request"), Mapping)):
            yield receipt["request"], names([receipt.get("artifact")], "name")


def present_bytes(directory: str | Path | None, request: Mapping[str, Any] | None) -> int:
    """The bytes of ``request``'s download already on disk in ``directory``.

    Only files that a fetch receipt in ``directory`` names count, and only
    a receipt that recorded this request: the same source and cycle, and
    the same area and member where the receipt records them, which is
    what the fetch itself checks before it resumes into a folder.  A
    folder named by hand can hold anything, a user's own source files or
    another request's download, and counting all of it priced a download
    that had not happened at nothing.

    Of those, a file the receipt gives a lead counts only when this
    request asks for that lead.  The fetch resumes a window that moved
    (the same cycle from a later start hour) into the same folder, and
    the leads it no longer asks for are still there but are not part of
    this download.
    """

    if directory is None or not request or not request.get("source"):
        return 0
    directory = Path(directory)
    try:
        if not directory.is_dir():
            return 0
        source = _canonical(str(request["source"]))
        cycle = _cycle(request, source)
        area = _area(request)
    except (OSError, ValueError):
        return 0
    if cycle is None:
        return 0
    wanted_area = None if area is None else area.as_manifest()
    member = request.get("member")
    try:
        planned = _planned_objects(source, request, _row(source, _mode(source, request)) or {})[0]
        wanted_leads = {lead for _role, lead in planned if lead is not None}
    except (ValueError, KeyError):
        # The fetch would refuse this request; nothing it names is resumed.
        return 0

    def same(recorded: Mapping[str, Any]) -> bool:
        try:
            if _canonical(str(recorded.get("source"))) != source:
                return False
            if _cycle({"cycle": recorded.get("cycle")}, source) != cycle:
                return False
        except (ValueError, TypeError):
            return False
        if "area" in recorded and recorded.get("area") != wanted_area:
            return False
        return not (member is not None and "member" in recorded
                    and str(recorded.get("member")) != str(member))

    def asked(lead: Any) -> bool:
        if lead is None:
            return True
        try:
            return int(lead) in wanted_leads
        except (TypeError, ValueError):
            return False

    named: set[Path] = set()
    for recorded, files in _receipts(directory):
        if same(recorded):
            named.update(directory / name for name, lead in files if asked(lead))
    total = 0
    for path in named:
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def preparation_estimate(exp, *, chain: str | None, forcing_times: int | None) -> dict[str, Any]:
    """What the preparation of ``exp`` on ``chain`` writes to disk.

    ``chain`` is the route the run takes (``prepared:hrrr``,
    ``prepared:go``, ``prepared:staged`` or ``experiment``), or None when
    nothing is prepared (an existing prepared bundle is reused).
    ``forcing_times`` is how many times the preparation is given, from
    :func:`download_estimate`; without it the outer grid's hours are
    counted hourly, which is the densest any source is prepared at.
    """

    if chain is None:
        return {"bytes": 0, "chain": None,
                "basis": "nothing is prepared: the run reuses a prepared bundle"}
    row = (table().get("preparation") or {}).get(chain)
    if row is None:
        return {"bytes": None, "chain": chain,
                "basis": f"no measured preparation size for the {chain} chain in woof/data/{TABLE_NAME}"}
    if forcing_times is None:
        forcing_times = int(float(exp.run_seconds) // 3600) + 1
    root_id = min((int(domain.grid_id) for domain in exp.domains), default=1)
    total = 0.0
    for domain in exp.domains:
        run = domain.run
        cells = int(run.nx) * int(run.ny) * int(run.nz)
        if int(domain.grid_id) == root_id:
            total += cells * (float(row.get("root_per_cell_per_time", 0.0)) * forcing_times
                              + float(row.get("root_per_cell", 0.0)))
            width = int(getattr(run, "spec_bdy_width", 5) or 5)
            boundary = 2 * (int(run.nx) + int(run.ny)) * width * int(run.nz)
            total += boundary * float(row.get("boundary_per_cell_per_time", 0.0)) * forcing_times
        else:
            total += cells * float(row.get("nest_per_cell", 0.0))
    return {"bytes": int(round(total)), "chain": chain, "forcing_times": forcing_times,
            "basis": f"per grid cell, as measured on a {chain} preparation ({row.get('measured')})"}


#: Grid cells the atmospheric window keeps beyond the target's own
#: footprint on every side: the parabolic horizontal stencil reaches two
#: source cells past the point it interpolates, and the staggered U and V
#: rows one more.
WINDOW_MARGIN_CELLS = 3


def _window_points(exp, source: str, grid_points: int) -> int:
    """Source grid points the atmospheric window keeps for ``exp``'s root.

    The window is the target's footprint in the source's own index space
    (every nest lies inside the root), so it is read from the source's
    declared native grid (:func:`woof.source_adapters.source_coverage_window`),
    the table row that already carries the grid for the coverage refusal.
    A root that leaves that grid is refused by the preparation, and a
    source or target this cannot place is priced at the whole grid.
    """

    import numpy as np

    projection = getattr(exp, "projection", None)
    if projection is None:
        return grid_points
    try:
        from woof.domain_wizard import _root_grid
        from woof.source_adapters import source_coverage_window
        from woof.source_coverage import LambertGridWindow

        window = source_coverage_window(source)
        if window is None or not window.nx or not window.ny:
            return grid_points
        root = exp.root.run
        latitude, longitude = _root_grid(
            {"map_proj": projection.map_proj, "ref_lat": projection.ref_lat,
             "ref_lon": projection.ref_lon, "truelat1": projection.truelat1,
             "truelat2": projection.truelat2, "stand_lon": projection.stand_lon},
            int(root.nx), int(root.ny), float(root.dx)).latlon_mass()
        if isinstance(window, LambertGridWindow):
            x, y = window._ij(latitude, longitude)
            x, y = x - 1.0, y - 1.0
        else:
            shifted = window._shift(longitude)
            x = (shifted - window.west) * (window.nx - 1) / (window.east - window.west)
            y = ((np.asarray(latitude, dtype=np.float64) - window.south)
                 * (window.ny - 1) / (window.north - window.south))
    except (AttributeError, KeyError, TypeError, ValueError):
        return grid_points
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return grid_points

    def span(values, size: int) -> int:
        low = max(0, int(np.floor(float(np.min(values)))) - WINDOW_MARGIN_CELLS)
        high = min(size, int(np.ceil(float(np.max(values)))) + WINDOW_MARGIN_CELLS + 1)
        return max(0, high - low)

    return min(grid_points, span(x, int(window.nx)) * span(y, int(window.ny)))


def _global_window_points(exp, axes: Mapping[str, Any], grid_points: int, *,
                          ring: bool = True) -> int | None:
    """Points of a whole-globe source the atmospheric window keeps for ``exp``'s root.

    ``axes`` is the row's ``global_axes``: the canonical axes the engine
    publishes a whole ring on (ascending latitude from ``latitude_first``,
    longitude from ``longitude_first``, one ``step_degrees`` apart).  The
    window is granted only where every stencil of the target is clear of
    the ring's stored cut (:func:`woof.ingest.horiz.global_ring_cut`, the
    rule :func:`woof.ingest.atmospheric_window.atmospheric_window_for_grids`
    applies), and then it is the target's footprint on those axes.  None
    when the target's stencils reach the cut: the preparation re-cuts the
    ring and keeps it whole.  With ``ring`` false (a transport that
    downloads only an area, whose crop has no cut) the footprint is taken
    as it is.  A target this cannot place is priced at ``grid_points``.
    """

    import numpy as np

    projection = getattr(exp, "projection", None)
    if projection is None:
        return grid_points
    try:
        from woof.domain_wizard import _root_grid
        from woof.ingest.horiz import global_source_cut

        root = exp.root.run
        grid = _root_grid(
            {"map_proj": projection.map_proj, "ref_lat": projection.ref_lat,
             "ref_lon": projection.ref_lon, "truelat1": projection.truelat1,
             "truelat2": projection.truelat2, "stand_lon": projection.stand_lon},
            int(root.nx), int(root.ny), float(root.dx))
        pairs = (grid.latlon_mass(), grid.latlon_u(), grid.latlon_v())
        step = float(axes["step_degrees"])
        nx, ny = int(axes["nx"]), int(axes["ny"])
        west = float(axes["longitude_first"])
        longitude_axis = west + step * np.arange(nx, dtype=np.float64)
        longitudes = [np.asarray(longitude, dtype=np.float64) for _, longitude in pairs]
        latitude_axis = float(axes["latitude_first"]) + step * np.arange(ny, dtype=np.float64)
        if ring and global_source_cut(
                latitude_axis, longitude_axis, *longitudes,
                target_latitudes=tuple(lat for lat, _ in pairs)) is not None:
            return None
        latitude = np.concatenate([np.asarray(lat, dtype=np.float64).ravel() for lat, _ in pairs])
        longitude = np.concatenate([values.ravel() for values in longitudes])
        x = np.mod(longitude - west, 360.0) / step
        y = (latitude - float(axes["latitude_first"])) / step
    except (AttributeError, KeyError, TypeError, ValueError):
        return grid_points
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return grid_points

    def span(values, size: int) -> int:
        low = max(0, int(np.floor(float(np.min(values)))) - WINDOW_MARGIN_CELLS)
        high = min(size, int(np.ceil(float(np.max(values)))) + WINDOW_MARGIN_CELLS + 1)
        return max(0, high - low)

    return min(grid_points, span(x, nx) * span(y, ny))


def _normalized_points(exp, document: str) -> int | None:
    """Points of the regular window a normalized source is composed on, for ``exp``.

    A source published on an unstructured mesh is remapped by its
    normalization document onto a regular window around the target before
    it is composed (:mod:`woof.source_normalization`): the document's step
    and halo around every corner point of the target.  That window is the
    frame grid, so it is asked of the same function the normalization runs.
    None when the target cannot be placed or the window would be refused.
    """

    projection = getattr(exp, "projection", None)
    if projection is None:
        return None
    try:
        from woof.domain_wizard import _root_grid
        from woof.source_normalization import load_document, target_from_points

        spec = load_document(Path(__file__).resolve().parent / "authorities" / document)
        root = exp.root.run
        latitude, longitude = _root_grid(
            {"map_proj": projection.map_proj, "ref_lat": projection.ref_lat,
             "ref_lon": projection.ref_lon, "truelat1": projection.truelat1,
             "truelat2": projection.truelat2, "stand_lon": projection.stand_lon},
            int(root.nx), int(root.ny), float(root.dx)).latlon_c()
        window = target_from_points(spec, [float(value) for value in latitude.ravel()],
                                    [float(value) for value in longitude.ravel()])
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return None
    return int(window.nx) * int(window.ny)


def compose_scratch_estimate(exp, *, chain: str | None, source: str | None,
                             forcing_times: int | None,
                             points: int | None = None) -> dict[str, Any]:
    """The frame stream a preparation stages in its compose scratch, priced before the download.

    A chain that composes through the mapped engine stages every decoded
    valid time on disk before the preparation reads it back: every field
    the source's mapping publishes, on the source's own grid, as 8-byte
    values, except the layers the atmospheric window crops.  The stream
    scales with the SOURCE grid, not the target, so none of the other
    figures reaches it: a GDPS 48 hour window whose target reaches the
    globe's stored longitude cut stages about 82 GB for a domain of any
    size.  It lives while the preparation runs and is removed when the
    preparation ends.

    The figures are rows of the packaged table's ``compose_scratch``
    section: per source, the grid points, the layers each valid time
    publishes, the bytes per value, and how many of those layers the
    atmospheric window may crop.  Those layers are priced over the
    target's footprint on the source grid: a regional source's from its
    coverage window, a global source's from the row's ``global_axes``.  A
    global source takes the window only where the target's stencils are
    clear of its stored longitude cut (:func:`_global_window_points`); a
    target on the cut keeps the ring whole, and the whole stream is then
    certain.  ``points`` is the request's own crop, when its transport
    downloads only an area: the frames then cover only that area.

    The answer carries ``bytes`` (the estimate), ``min_bytes`` (the part
    that does not depend on the window: the whole stream for a global
    source whose target reaches its stored cut), ``max_bytes`` (no window
    at all), ``valid_times``,
    ``per_valid_time``, ``source``, ``composes`` (whether the chain
    stages a stream at all, priced or not) and a ``basis``.  ``bytes`` is
    None when the chain composes and the table has no row for the source.
    """

    section = table().get("compose_scratch") or {}
    if chain is None or chain not in (section.get("chains") or ()):
        return {"bytes": 0, "min_bytes": 0, "max_bytes": 0, "valid_times": None,
                "per_valid_time": 0, "source": source, "composes": False,
                "basis": "this preparation stages no decoded frame stream"}
    row = (section.get("sources") or {}).get(source) if source else None
    if row is None:
        return {"bytes": None, "min_bytes": None, "max_bytes": None, "valid_times": None,
                "per_valid_time": None, "source": source, "composes": True,
                "basis": f"no compose scratch row for {source} in woof/data/{TABLE_NAME}, "
                         "so the frame stream its preparation stages is left out of this figure"}
    if not forcing_times:
        return {"bytes": None, "min_bytes": None, "max_bytes": None, "valid_times": None,
                "per_valid_time": None, "source": source, "composes": True,
                "basis": "the number of valid times the preparation composes is not known"}
    if row.get("normalization"):
        normalized = _normalized_points(exp, str(row["normalization"]))
        if normalized is None:
            return {"bytes": None, "min_bytes": None, "max_bytes": None, "valid_times": None,
                    "per_valid_time": None, "source": source, "composes": True,
                    "basis": f"{source} is composed on a window its normalization sizes around "
                             "the target, and this target could not be placed on it"}
        grid_points = normalized
    else:
        grid_points = int(points) if points else int(row["grid_points"])
    layers = int(row["layers_per_valid_time"])
    windowed = int(row.get("windowed_layers", 0))
    per_value = int(row["bytes_per_value"])
    on_cut = False
    if not windowed:
        kept = grid_points
    elif row.get("global_axes"):
        kept = _global_window_points(exp, row["global_axes"], grid_points, ring=not points)
        if kept is None:
            # The target's stencils reach the ring's stored cut: the ring
            # is re-cut and kept whole, and that is certain.
            on_cut, windowed, kept = True, 0, grid_points
    else:
        kept = _window_points(exp, source, grid_points)
    whole = grid_points * layers * per_value
    fixed = grid_points * (layers - windowed) * per_value
    per_time = fixed + kept * windowed * per_value
    times = int(forcing_times)
    basis = (f"{times} valid time(s) of {per_time:,} bytes: {grid_points:,} grid points "
             f"of {source} by {layers} layers of {per_value}-byte values")
    if row.get("normalization"):
        basis += ", on the regular window its normalization sizes around the target"
    elif points:
        basis += ", over the area the request downloads"
    if windowed:
        basis += (f", {windowed} of those layers cropped to the {kept:,} points the target's "
                  "footprint keeps on the source grid")
    elif on_cut:
        basis += (", all of them whole: the target reaches the global grid's stored "
                  "longitude cut, so the atmospheric window is not taken")
    basis += f" ({row.get('measured')})"
    return {"bytes": per_time * times, "min_bytes": fixed * times, "max_bytes": whole * times,
            "valid_times": times, "per_valid_time": per_time, "source": source,
            "composes": True, "basis": basis}


__all__ = ["REQUEST_KEYS", "TABLE_NAME", "TABLE_SCHEMA", "WINDOW_MARGIN_CELLS",
           "compose_scratch_estimate", "download_estimate",
           "preparation_estimate", "present_bytes", "request_from_arguments", "table",
           "table_path"]
