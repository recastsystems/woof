"""Download a manifest-bound GFS subset for ``rw-wps --source gfs``.

The NOMADS selector is deliberately explicit.  In particular, GFS soil
temperature is published as ``TSOIL`` rather than the generic ``TMP`` name
used for atmospheric and surface temperature.  Omitting it creates a valid
GRIB2 file that the fail-closed native bridge correctly rejects.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request

from woof.nomads_governor import paced_urlopen


#: The pressure ladder ArWen was certified against: 21 levels topping
#: out at 100 hPa.  It is the DEFAULT, not the ceiling -- see
#: :func:`levels_for_top`.  A run whose model top sits above 100 hPa
#: (WRF-Runner's GFS namelists all declare ``p_top = 5000`` Pa, i.e. 50
#: hPa) needs the ladder extended upward, and the product publishes the
#: levels to do it.
PRESSURE_LEVELS_HPA = (
    100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600,
    650, 700, 750, 800, 850, 900, 925, 950, 975, 1000,
)

#: Every isobaric level ``pgrb2.0p25`` publishes for ALL FIVE of the
#: fields the 3-D selection takes (HGT, TMP, RH, UGRD, VGRD).
#:
#: Captured from a live inventory on 2026-07-30 -- the unedited index in
#: ``tests/fixtures/gfs-inventory/``, whose README records the object,
#: the digest and the census.  This constant is a tripwire and a
#: fallback, exactly as ``fetch_bars.CERTIFIED_RECORD_BARS`` is for the
#: record count: the live index is consulted first
#: (:func:`available_levels_from_index`), and this stands in only when it
#: cannot be read.  A test binds the two together, so the constant
#: cannot quietly drift from the file it was taken from.
#:
#: Note the last 21 entries are exactly :data:`PRESSURE_LEVELS_HPA`: a
#: deeper top extends the certified ladder upward, it never replaces it.
CERTIFIED_AVAILABLE_LEVELS_HPA = (
    0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.4, 0.7, 1, 2, 3, 5, 7, 10,
    15, 20, 30, 40, 50, 70,
    100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600,
    650, 700, 750, 800, 850, 900, 925, 950, 975, 1000,
)

#: The fields the selection takes on every isobaric level.  One level
#: therefore contributes exactly this many records, which is what makes
#: the record bar a linear function of the ladder length.
PRESSURE_FIELDS = ("HGT", "TMP", "RH", "UGRD", "VGRD")

#: Records the selection takes that do not sit on an isobaric level:
#: HGT/PRES/TMP/WEASD/SNOD/LAND/ICEC at the surface (7), TMP/RH at 2 m
#: (2), UGRD/VGRD at 10 m (2), and TSOIL+SOILW on the four below-ground
#: layers (8).
SINGLE_LEVEL_RECORD_COUNT = 19


class TopPressureUnavailable(ValueError):
    """The requested model top is above everything the product carries.

    Distinct from a malformed request so callers can say *how far up the
    source actually reaches* rather than merely refusing.
    """


def available_levels_from_index(index_text: str, *, model: str = "gfs") -> tuple[float, ...]:
    """The isobaric levels this object publishes for every 3-D field.

    A level counts only when all five of :data:`PRESSURE_FIELDS` appear
    on it: four-fifths of a level is not a level the bridge can decode,
    and silently selecting one would make the fetch pass its own record
    bar and fail at ingest.
    """

    required = tuple("SPFH" if model == "gdas" and name == "RH" else name for name in PRESSURE_FIELDS)
    per_field: dict[str, set[float]] = {name: set() for name in required}
    for line in index_text.splitlines():
        fields = line.strip().split(":")
        if len(fields) < 6:
            continue
        variable, level = fields[3], fields[4]
        if variable not in per_field or not level.endswith(" mb"):
            continue
        try:
            per_field[variable].add(float(level[:-3]))
        except ValueError:
            continue
    if not all(per_field.values()):
        return ()
    usable = set.intersection(*per_field.values())
    return tuple(sorted(usable))


def levels_for_top(
        top_pressure_pa: float | None, *,
        available: tuple[float, ...] = CERTIFIED_AVAILABLE_LEVELS_HPA,
        certified: tuple[int, ...] = PRESSURE_LEVELS_HPA,
) -> tuple[float, ...]:
    """The ladder that covers a model top of ``top_pressure_pa``.

    ``None`` asks for the certified ladder unchanged -- today's
    behaviour, byte for byte, for every caller that does not care.

    Otherwise the certified ladder is extended UPWARD along whatever the
    product actually publishes until a level sits strictly above the
    requested top (i.e. at a pressure below it), because that is
    the condition ``woof.vertical_contract`` enforces at ingest: the
    source atmosphere has to reach the model top, not stop under it.
    Extending rather than replacing keeps every level a certified run
    already used, so a deeper top adds data and changes nothing that was
    there before.

    Raises :class:`TopPressureUnavailable` when the request is above
    everything the product carries, naming the deepest top it can serve.
    """

    ladder = tuple(float(level) for level in certified)
    if top_pressure_pa is None:
        return ladder
    top = float(top_pressure_pa)
    if not top > 0.0:
        raise ValueError(
            f"the requested model top must be a positive pressure in Pa, "
            f"got {top_pressure_pa!r}")
    if min(ladder) * 100.0 < top:
        return ladder
    higher = sorted(
        (level for level in available if float(level) < min(ladder)),
        reverse=True)
    extension: list[float] = []
    for level in higher:
        extension.append(float(level))
        if float(level) * 100.0 < top:
            return tuple(sorted(extension)) + ladder
    deepest = min(available) if available else min(ladder)
    raise TopPressureUnavailable(
        f"a model top of {top:g} Pa needs a GFS pgrb2.0p25 level strictly "
        f"above it, and the deepest this product publishes for all of "
        f"{', '.join(PRESSURE_FIELDS)} is {deepest:g} hPa "
        f"({deepest * 100.0:g} Pa).  Raise --p-top-pa to "
        f"{deepest * 100.0:g} Pa (strictly greater), or use a source whose "
        "atmosphere reaches higher.")


def record_count_for_levels(level_count: int) -> int:
    """How many records the selection takes for a ladder that long."""

    return len(PRESSURE_FIELDS) * int(level_count) + SINGLE_LEVEL_RECORD_COUNT
NOMADS_LEVELS = (
    "lev_surface",
    "lev_2_m_above_ground",
    "lev_10_m_above_ground",
    "lev_0-0.1_m_below_ground",
    "lev_0.1-0.4_m_below_ground",
    "lev_0.4-1_m_below_ground",
    "lev_1-2_m_below_ground",
)
NOMADS_VARIABLES = (
    "var_HGT", "var_TMP", "var_RH", "var_UGRD", "var_VGRD",
    "var_PRES", "var_WEASD", "var_SNOD", "var_LAND", "var_ICEC",
    "var_TSOIL", "var_SOILW",
)


def declared_variables(model: str = "gfs") -> tuple[str, ...]:
    """The variables this module declares; GDAS takes specific humidity
    (pressure levels and 2 m) where GFS takes relative humidity."""
    return tuple("var_SPFH" if model == "gdas" and name == "var_RH" else name for name in NOMADS_VARIABLES)


def nomads_variables(model: str = "gfs") -> tuple[str, ...]:
    """Every variable a ``model`` crop asks the grib-filter for.

    The declared variables, then every variable the source's packaged
    mapping binds that they do not name, read from the mapping table
    itself (:func:`woof.fetch.mapped_record_levels`).  GDAS prepares
    through its packaged mapping, which requires the five hydrometeor
    mixing ratios since 2.9.9; a crop without CLMR/ICMR/RWMR/SNMR/GRLE was
    refused at prepare ("mapped frame ... lacks required fields
    ['cloud_ice_mixing_ratio', ...]", 2026-10-29).  GFS decodes
    through ``gfs_grib2_bridge`` and binds no mapping, so its request is
    the declared one.  The mapping's variables are ``.idx`` names; the
    filter spells cloud mixing ratio ``CLWMR`` where the index says
    ``CLMR`` (:data:`woof.fetch_bars.FILTER_TO_IDX_NAMES`).  A mapping
    that cannot be named in an index adds nothing here, and the
    preparation names what it lacks.
    """

    from woof.fetch import mapped_record_levels
    from woof.fetch_bars import filter_variable

    declared = declared_variables(model)
    mapped = mapped_record_levels(model) or {}
    extra = tuple(filter_variable(name) for name in sorted(mapped))
    return declared + tuple(name for name in extra if name not in declared)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cycle(raw: str) -> datetime:
    value = datetime.strptime(raw, "%Y-%m-%d_%H:%M:%S")
    if value.hour not in {0, 6, 12, 18} or value.minute or value.second:
        raise ValueError("GFS cycle must be exactly 00/06/12/18 UTC")
    return value


def _forecast_hours(raw: str) -> tuple[int, ...]:
    try:
        hours = tuple(int(value) for value in raw.split(","))
    except ValueError as error:
        raise ValueError("forecast hours must be comma-separated integers") from error
    if len(hours) < 2 or hours[0] != 0 or any(
            later <= earlier for earlier, later in zip(hours, hours[1:])):
        raise ValueError("forecast hours must contain increasing f000 plus a later time")
    steps = {later - earlier for earlier, later in zip(hours, hours[1:])}
    if len(steps) != 1 or steps.pop() not in {1, 3}:
        raise ValueError("forecast-hour cadence must be uniformly 1 or 3 hours")
    if hours[-1] > 384:
        raise ValueError("forecast hours exceed the supported GFS f384 horizon")
    return hours


#: The NCEP models this selector serves, keyed by woof source name.
#:
#: GDAS is the GFS assimilation cycle's own output: the *same*
#: ``pgrb2.0p25`` container -- 0.25-degree regular lat/lon, the same
#: variable and level codes, the same 124-record census under this
#: selector, the same originating centre and table versions -- published
#: under a different directory and grib-filter script.  Verified against
#: a live cycle: 124 records, scan 0x40, DRT 5.0, shape 6, centre 7,
#: master table 2, local table 1, PDT 4.0, generating process 81.  Only
#: the naming and the forecast horizon differ, so the certified GFS
#: mapping, bridge and front door serve it unchanged.
NOMADS_MODELS = {
    "gfs": {"script": "filter_gfs_0p25.pl", "prefix": "gfs"},
    "gdas": {"script": "filter_gdas_0p25.pl", "prefix": "gdas"},
}


def _level_parameter(level_hpa: float) -> str:
    """``lev_<L>_mb`` exactly as the grib filter spells it.

    The filter's level names carry no trailing zeros: 0.5 hPa is
    ``lev_0.5_mb`` and 100 hPa is ``lev_100_mb``, never ``lev_100.0_mb``.
    """

    value = float(level_hpa)
    text = f"{value:g}"
    return f"lev_{text}_mb"


def nomads_query(
        cycle: datetime, forecast_hour: int, *, left_lon: float,
        right_lon: float, bottom_lat: float, top_lat: float,
        model: str = "gfs",
        pressure_levels_hpa: tuple[float, ...] = PRESSURE_LEVELS_HPA,
) -> str:
    if not 0.0 <= left_lon < right_lon <= 360.0:
        raise ValueError("GFS subset longitudes must increase within [0, 360]")
    if not -90.0 <= bottom_lat < top_lat <= 90.0:
        raise ValueError("GFS subset latitudes must increase within [-90, 90]")
    try:
        spec = NOMADS_MODELS[model]
    except KeyError:
        raise ValueError(
            f"unknown NOMADS model {model!r}; expected one of "
            f"{sorted(NOMADS_MODELS)}") from None
    prefix = spec["prefix"]
    hour = cycle.strftime("%H")
    parameters: list[tuple[str, str]] = [
        ("file", f"{prefix}.t{hour}z.pgrb2.0p25.f{forecast_hour:03d}"),
        ("subregion", ""),
        ("leftlon", format(left_lon, "g")),
        ("rightlon", format(right_lon, "g")),
        ("toplat", format(top_lat, "g")),
        ("bottomlat", format(bottom_lat, "g")),
    ]
    parameters.extend(
        (_level_parameter(level), "on") for level in pressure_levels_hpa)
    parameters.extend((name, "on") for name in NOMADS_LEVELS)
    parameters.extend((name, "on") for name in nomads_variables(model))
    parameters.append((
        "dir", f"/{prefix}.{cycle:%Y%m%d}/{hour}/atmos",
    ))
    return ("https://nomads.ncep.noaa.gov/cgi-bin/" + spec["script"] + "?"
            + urlencode(parameters))


class _IncompleteGrib(RuntimeError):
    """A 200 answer whose body is not a whole GRIB2 stream; a fresh transfer can repair it."""


def _retry_delay(error: BaseException, attempt: int) -> float | None:
    """Seconds before the next attempt, or None when asking again cannot help.

    The network faults are the tree's shared classification
    (:func:`woof.fetch_endpoints.retry_delay`): a 404 or another status
    that will not change is not asked again, and a busy host's own
    ``Retry-After`` is served up to the shared limit.  An incomplete
    body is this transport's own, asked again on the same ladder.
    """

    from woof import fetch_endpoints

    if isinstance(error, _IncompleteGrib):
        return 2.0 ** attempt
    return fetch_endpoints.retry_delay(
        error, attempt, wait_limit_s=fetch_endpoints.TRANSIENT_WAIT_LIMIT_S)


def _download(url: str, destination: Path, *, retries: int | None = None) -> None:
    """Stream one NOMADS CGI subset onto ``destination``, governed.

    Every request goes through :mod:`woof.nomads_governor`, so this
    transport observes the same node-wide NOMADS spacing and cooldown
    the Rust backbone does instead of racing it.

    The staging name carries this process's pid and a nanosecond stamp.
    A fixed ``<name>.part`` is a file two runs collide on and a killed
    run leaves behind under a name the next run cannot tell from its
    own; a unique one is only ever this attempt's, and is removed when
    the attempt fails.
    """

    from woof import fetch_endpoints
    from woof.fetch_pool import (TransferCancelled, raise_if_stopped,
                                  sleep_unless_stopped)

    # The tree's one attempt budget (fetch_endpoints.TRANSIENT_ATTEMPTS);
    # a caller may ask for fewer, never more.
    attempts = fetch_endpoints.TRANSIENT_ATTEMPTS
    if retries is not None:
        attempts = max(1, min(int(retries), attempts))
    partial = destination.with_suffix(
        f"{destination.suffix}.{os.getpid()}-{time.time_ns()}.part")
    for attempt in range(1, attempts + 1):
        try:
            raise_if_stopped()
            request = Request(url, headers={"User-Agent": "rw-wps-gfs-fetch/1"})
            with paced_urlopen(request, timeout=300) as response, \
                    partial.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
                    # Another file already failed this request: the
                    # rest of this one would be bytes nobody uses.
                    raise_if_stopped()
                output.flush()
                os.fsync(output.fileno())
            with partial.open("rb") as stream:
                signature = stream.read(4)
                stream.seek(-4, os.SEEK_END)
                terminator = stream.read(4)
            if (signature != b"GRIB" or terminator != b"7777"
                    or partial.stat().st_size <= 1024):
                raise _IncompleteGrib("NOMADS response is not a complete GRIB2 stream")
            os.replace(partial, destination)
            return
        except Exception as error:
            partial.unlink(missing_ok=True)
            if attempt == attempts or isinstance(error, TransferCancelled):
                raise
            # The shared wait: 2, 4, 8 and 16 s, or the server's own
            # Retry-After when that is longer and within the shared
            # limit.  None is a fault no retry changes (a 404, a refused
            # client, a name that does not resolve), which used to be
            # asked five times over most of a minute.  (A NOMADS
            # over-rate-limit answer additionally opened the node-wide
            # cooldown inside paced_urlopen, which the next attempt's
            # pacing serves in full.)  Cut short, and the attempt not
            # made, once another file has failed the request.
            delay = _retry_delay(error, attempt)
            if delay is None:
                raise
            sleep_unless_stopped(delay, sleep=time.sleep)


def _write_manifest(
        output: Path, *, cycle: datetime, records: tuple[tuple[int, Path], ...],
        series: Path, bridge: Path, wps_namelist: Path,
        experiment_config: Path,
) -> tuple[Path, str]:
    roles = {
        "series": series,
        "bridge": bridge,
        "wps_namelist": wps_namelist,
        "experiment_config": experiment_config,
        **{f"grib-f{hour:03d}": path for hour, path in records},
    }
    for path in roles.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = {
        "schema": "gpuwm-gfs-direct-input-manifest-v1",
        "source": {
            "model": "GFS",
            "product": "pgrb2.0p25",
            "cycle": cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "files": {
            role: {"name": path.name, "sha256": _sha256(path)}
            for role, path in roles.items()
        },
    }
    path = output / "input-manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = _sha256(path)
    (output / "input-manifest.sha256").write_text(digest + "\n", encoding="ascii")
    return path, digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--forecast-hours", default="0,3")
    parser.add_argument("--left-lon", type=float, required=True)
    parser.add_argument("--right-lon", type=float, required=True)
    parser.add_argument("--bottom-lat", type=float, required=True)
    parser.add_argument("--top-lat", type=float, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--wps-namelist", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cycle = _cycle(args.cycle)
    hours = _forecast_hours(args.forecast_hours)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "input-manifest.json").exists():
        raise FileExistsError("refusing to replace an existing GFS input manifest")
    records = []
    for forecast_hour in hours:
        path = output / f"gfs.t{cycle:%H}z.pgrb2.0p25.f{forecast_hour:03d}.subset.grib2"
        _download(nomads_query(
            cycle, forecast_hour, left_lon=args.left_lon,
            right_lon=args.right_lon, bottom_lat=args.bottom_lat,
            top_lat=args.top_lat,
        ), path)
        records.append((forecast_hour, path))
    records_tuple = tuple(records)
    series = output / "gfs-series.tsv"
    series.write_text(
        "".join(
            f"{hour}\t{path}\t{81 if hour == 0 else 96}\n"
            for hour, path in records_tuple),
        encoding="utf-8",
    )
    manifest, digest = _write_manifest(
        output, cycle=cycle, records=records_tuple, series=series,
        bridge=args.bridge.resolve(), wps_namelist=args.wps_namelist.resolve(),
        experiment_config=args.experiment_config.resolve(),
    )
    print(json.dumps({
        "status": "READY",
        "series": str(series),
        "input_manifest": str(manifest),
        "input_manifest_sha256": digest,
        "records": [
            {"forecast_hour": hour, "path": str(path), "bytes": path.stat().st_size,
             "sha256": _sha256(path)}
            for hour, path in records_tuple
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
