"""Observed lake ice: start the lakes the forcing cannot see as observed.

Why this module exists
----------------------
A forcing analysis says nothing about a lake it does not resolve.  On the
GFS route a land-use lake takes its skin temperature and sea-ice fraction
from metgrid's whole-source nearest-water search, which is what WPS and
real.exe give it, so a lake GFS does not resolve inherits the ice (none)
and the skin temperature of the nearest GFS water.  Measured on a GFS start
of 2025-02-20 12Z (3 km, Great Lakes and Boundary Waters), against IMS 1 km
for that day: of 191 Boundary Waters / Rainy Lake cells IMS showed frozen,
0 started with ice, at skin temperatures of 246-274 K under 251 K air, and
in the first hour they put up a mean +89 W m-2 of sensible heat (to about
250) and +37 W m-2 of latent heat where frozen lakes give -13.  Over every
lake the forcing does not resolve, 1,150 of 3,498 observed-frozen cells
started frozen.  No setting reached ice: the two skin arms move only the
temperature.

What it does
------------
It reads a daily observed snow and ice analysis (rows of
``woof/data/surface_obs/lake-ice-sources.v1.toml``: IMS 1 km, then 4 km),
samples it on the target grid's lake cells in Rust
(``tools/grib1_bridge/src/ims_snow_ice.rs``), and decides each lake cell:

* where the analysis maps water (open water 1 or ice 3) in the cell's
  window, ice wins when ice pixels outnumber open-water pixels;
* where it maps no water there (a lake smaller than its water mask), the
  ground is read instead: snow-covered (4) outnumbering snow-free (2) is a
  frozen, snow-covered lake;
* a tie takes the centre pixel's reading;
* a cell the analysis does not cover keeps the forcing's own answer.

A frozen cell takes SEAICE = XICE = 1 and a skin temperature of at most
271.4 K (the ice-column base real.exe's adjust_for_seaice_post uses); the
soil router then builds its ice column exactly as for sea ice.  An open
cell keeps the forcing's ice and skin rule unchanged.  The forcing's own
ice is never removed: a cell the forcing already froze stays frozen.

The window is the target cell: ``floor(dx / (2 * pixel))`` pixels each
side of the centre (one pixel at 1 km, a 3 x 3 window at 3 km).

When no analysis can be read (offline, nothing posted for the date, a
library that predates the reader) the run says so on stderr, names why,
and keeps the forcing's ice -- per run, never as a changed default.
The analysis is asked for only when a domain of the tree has lake cells.
``[ingest] gfs_lake_ice = "wps"`` (or ``woof go --lake-ice wps``) selects
the forcing's ice outright (the WPS and real.exe identity arm) and asks
for nothing; ``[case_data]`` still accepts the key.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from pathlib import Path
import sys
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from woof.ingest.backend_contract import _sha256

#: ``gfs_lake_ice`` values; the first is the default.
LAKE_ICE_CHOICES = ("observed", "wps")

#: The skin temperature cap of an observed-frozen lake cell, K: the base of
#: the ice column real.exe's adjust_for_seaice_post builds (TMN = 271.4 K,
#: module_soil_pre.F), so the column never warms upward through the ice.
FROZEN_LAKE_SKIN_CAP_K = 271.4

#: The cache the analyses are kept in: ``$WOOF_SURFACE_OBS_CACHE`` or
#: ``~/.woof/cache/surface-obs``.  A file placed there by hand under the
#: row's own file name is used without any network access.
CACHE_ENV = "WOOF_SURFACE_OBS_CACHE"

SOURCES_TABLE = (Path(__file__).resolve().parent.parent / "data"
                 / "surface_obs" / "lake-ice-sources.v1.toml")

IMS_ENTRY = "gpuwm_ims_sample_codes"

_OPEN_WATER, _SNOW_FREE, _ICE, _SNOW = 1, 2, 3, 4


@dataclass(frozen=True)
class LakeIceSource:
    id: str
    describes: str
    format: str
    url: str
    mirrors: tuple[str, ...]
    pixel_m: float
    lookback_days: int
    licence: str
    citation: str

    def urls(self, day: datetime) -> tuple[str, ...]:
        fields = {"year": f"{day.year:04d}",
                  "doy": f"{day.timetuple().tm_yday:03d}"}
        return tuple(template.format(**fields)
                     for template in (self.url, *self.mirrors))


def load_lake_ice_sources(path: Path | None = None) -> tuple[LakeIceSource, ...]:
    """Every row of the lake-ice source table, in ladder order."""

    import tomllib

    document = tomllib.loads(Path(path or SOURCES_TABLE).read_text(encoding="utf-8"))
    if document.get("schema") != "gpuwm-lake-ice-sources-v1":
        raise ValueError(f"{path or SOURCES_TABLE}: not a gpuwm-lake-ice-sources-v1 table")
    rows = []
    for raw in document.get("source", ()):
        if raw.get("format") != "ims-ascii-v1.3":
            raise ValueError(
                f"lake-ice source {raw.get('id')!r} declares format "
                f"{raw.get('format')!r}; the reader knows 'ims-ascii-v1.3'")
        rows.append(LakeIceSource(
            id=str(raw["id"]), describes=str(raw["describes"]),
            format=str(raw["format"]), url=str(raw["url"]),
            mirrors=tuple(str(m) for m in raw.get("mirrors", ())),
            pixel_m=float(raw["pixel_m"]),
            lookback_days=int(raw["lookback_days"]),
            licence=str(raw["licence"]), citation=str(raw["citation"])))
    if not rows:
        raise ValueError(f"{path or SOURCES_TABLE}: the table has no rows")
    return tuple(rows)


def default_cache_root() -> Path:
    value = os.environ.get(CACHE_ENV, "").strip()
    return Path(value) if value else Path.home() / ".woof" / "cache" / "surface-obs"


@dataclass(frozen=True)
class ObservedLakeIce:
    """One acquired analysis file, bound by its bytes."""

    source: LakeIceSource
    path: Path
    sha256: str
    valid_time: datetime
    url: str

    def identity(self) -> dict[str, object]:
        return {
            "provider": self.source.id,
            "describes": self.source.describes,
            "analysis_valid_time": self.valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "file": self.path.name,
            "sha256": self.sha256,
            "url": self.url,
            "licence": self.source.licence,
        }


def _download(url: str, destination: Path, *, opener, timeout_s: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + f".part{os.getpid()}")
    try:
        request = Request(url, headers={"User-Agent": "gpuwm-lake-ice"})
        with opener(request, timeout=timeout_s) as response, partial.open("wb") as stream:
            while block := response.read(1 << 20):
                stream.write(block)
        with partial.open("rb") as stream:
            if stream.read(2) != b"\x1f\x8b":
                raise ValueError(f"{url} did not answer with a gzip file")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def acquire_observed_lake_ice(
        valid_time: datetime, *, cache_root: Path | None = None,
        sources: tuple[LakeIceSource, ...] | None = None,
        opener=urlopen, timeout_s: float = 60.0,
) -> tuple[ObservedLakeIce | None, list[str]]:
    """The newest analysis valid at or before ``valid_time``, or why none.

    Returns ``(observation, attempts)``; ``attempts`` lists every file tried
    and what it answered, so a fallback can say exactly why.
    """

    rows = sources if sources is not None else load_lake_ice_sources()
    root = Path(cache_root) if cache_root is not None else default_cache_root()
    start = datetime(valid_time.year, valid_time.month, valid_time.day)
    attempts: list[str] = []
    offline = False
    for row in rows:
        for back in range(row.lookback_days + 1):
            day = start - timedelta(days=back)
            urls = row.urls(day)
            target = root / row.id / Path(urls[0]).name
            if target.is_file():
                return ObservedLakeIce(row, target, _sha256(target), day, urls[0]), attempts
            if offline:
                # No network: keep walking the ladder through the cache only,
                # so an offline machine with an older day or the 4 km row
                # cached still starts from an observed analysis (the
                # documented "put the file there" route).  2.9.1 verify: the
                # first refused connection used to end the search.
                continue
            for url in urls:
                try:
                    _download(url, target, opener=opener, timeout_s=timeout_s)
                except HTTPError as error:
                    attempts.append(f"{url}: HTTP {error.code}")
                    continue
                except (URLError, TimeoutError, ConnectionError, OSError, ValueError) as error:
                    attempts.append(f"{url}: {error}")
                    if isinstance(error, (URLError, TimeoutError, ConnectionError)):
                        # No network: every later URL on this host fails the
                        # same way, and each would cost a full timeout.
                        offline = True
                        break
                    continue
                return ObservedLakeIce(row, target, _sha256(target), day, url), attempts
    return None, attempts


def _ims_backend():
    from woof.ingest.cpu_backend import masked_fields_cpu_backend

    backend = masked_fields_cpu_backend()
    try:
        entry = getattr(backend._library, IMS_ENTRY)
    except AttributeError as error:
        raise RuntimeError(
            f"the CPU preprocessing library at {backend.path} predates the "
            f"lake-ice reader ({IMS_ENTRY}); `woof fetch-bridges` stages "
            "the current one") from error
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    entry.argtypes = [pointer, size, pointer, pointer, size, ctypes.c_uint32,
                      pointer, pointer, pointer]
    entry.restype = ctypes.c_int32
    return backend, entry


def sample_ims_codes(path: Path, lat, lon, *, radius: int):
    """Per-point code counts ``(n, 5)``, centre codes ``(n,)`` and a receipt."""

    backend, entry = _ims_backend()
    lat = np.ascontiguousarray(lat, dtype=np.float64).ravel()
    lon = np.ascontiguousarray(lon, dtype=np.float64).ravel()
    n = lat.size
    counts = np.zeros((n, 5), dtype=np.uint32)
    centre = np.zeros(n, dtype=np.uint8)
    receipt = np.zeros(3, dtype=np.uint64)
    encoded = str(path).encode("utf-8")
    code = int(entry(encoded, len(encoded),
                     ctypes.c_void_p(lat.ctypes.data), ctypes.c_void_p(lon.ctypes.data),
                     n, int(radius),
                     ctypes.c_void_p(counts.ctypes.data),
                     ctypes.c_void_p(centre.ctypes.data),
                     ctypes.c_void_p(receipt.ctypes.data)))
    if code:
        message = backend._native_message() or f"native error {code}"
        raise ValueError(f"the lake-ice analysis could not be read: {message}")
    return counts, centre, {"dimension": int(receipt[0]),
                            "rows_read": int(receipt[1]),
                            "off_grid": int(receipt[2])}


def classify_frozen(counts, centre):
    """Frozen / observed per point from IMS window counts (module docstring)."""

    counts = np.asarray(counts, dtype=np.int64)
    centre = np.asarray(centre)
    water, ice = counts[:, _OPEN_WATER], counts[:, _ICE]
    bare, snow = counts[:, _SNOW_FREE], counts[:, _SNOW]
    mapped_water = (water + ice) > 0
    frozen = np.where(
        mapped_water,
        (ice > water) | ((ice == water) & np.isin(centre, (_ICE, _SNOW))),
        (snow > bare) | ((snow == bare) & (centre == _SNOW)))
    observed = mapped_water | ((bare + snow) > 0)
    return frozen & observed, observed


def window_radius(dx_m: float, pixel_m: float) -> int:
    return max(0, int(float(dx_m) // (2.0 * float(pixel_m))))


@dataclass(frozen=True)
class LakeIceDecision:
    """Which lake cells an analysis froze on one grid."""

    frozen: np.ndarray       # (ny, nx) bool: observed frozen lake cells
    receipt: Mapping[str, object]


def decide_lake_ice(observation: ObservedLakeIce, lat, lon, lake_mask, *,
                    dx_m: float) -> LakeIceDecision:
    lake = np.asarray(lake_mask, dtype=bool)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    radius = window_radius(dx_m, observation.source.pixel_m)
    counts, centre, read = sample_ims_codes(
        observation.path, lat[lake], lon[lake], radius=radius)
    frozen_cells, observed_cells = classify_frozen(counts, centre)
    frozen = np.zeros(lake.shape, dtype=bool)
    frozen[lake] = frozen_cells
    receipt = {
        **observation.identity(),
        "window_radius_pixels": radius,
        "lake_cells": int(lake.sum()),
        "observed_cells": int(observed_cells.sum()),
        "frozen_cells": int(frozen_cells.sum()),
        "frozen_by_ice_pixels": int((frozen_cells & ((counts[:, 1] + counts[:, 3]) > 0)).sum()),
        "frozen_by_snow_cover": int((frozen_cells & ((counts[:, 1] + counts[:, 3]) == 0)).sum()),
        "reader": read,
        # No wall time here.  This receipt is bound into the prepared
        # cache's source identity (gfs_direct, nest_init), and a wall time
        # made that identity differ on every preparation of the same
        # inputs.  THE BREAKAGE (run_515470d250a1060a, 2026-10-26): a
        # checkpoint resume's forcing continuation re-prepares the head and
        # requires the identical cache, so it always refused with
        # "continuation changes its immutable prepared head cache"
        # (0.082 vs 0.08 s here and nothing else).
    }
    return LakeIceDecision(frozen=frozen, receipt=receipt)


def apply_lake_ice(fields: Mapping[str, object], skin, frozen, *,
                   skin_name: str | None = None):
    """Freeze the observed-frozen cells in a metgrid field set.

    Returns ``(fields, skin, receipt)``: a shallow copy of ``fields`` whose
    sea-ice fraction (``XICE`` and/or ``SEAICE``, whichever it carries;
    ``SEAICE`` when neither) is 1 on ``frozen``, and ``skin`` (host float64)
    capped at :data:`FROZEN_LAKE_SKIN_CAP_K` there.  ``skin_name`` also
    writes the capped skin back into ``fields`` under that name, for a
    route whose soil call reads the skin from the field set.  Arrays keep
    their own kind (NumPy or CuPy) and dtype.
    """

    frozen = np.asarray(frozen, dtype=bool)
    out = dict(fields)
    names = [name for name in ("XICE", "SEAICE") if name in out] or ["SEAICE"]
    before_ice = None
    for name in names:
        template = out.get(name)
        if template is None:
            host = np.zeros(frozen.shape, dtype=np.float64)
        else:
            host = _host64(template)
            if before_ice is None:
                before_ice = host >= 0.5
        host = host.copy()
        host[frozen] = 1.0
        out[name] = host if template is None else _like(host, template)
    skin_host = _host64(skin).copy()
    capped = frozen & (skin_host > FROZEN_LAKE_SKIN_CAP_K)
    skin_host[frozen] = np.minimum(skin_host[frozen], FROZEN_LAKE_SKIN_CAP_K)
    if skin_name is not None:
        out[skin_name] = _like(skin_host, fields[skin_name])
    newly = frozen if before_ice is None else frozen & ~before_ice
    return out, skin_host, {
        "cells_frozen": int(frozen.sum()),
        "cells_newly_frozen": int(newly.sum()),
        "skin_capped_cells": int(capped.sum()),
        "skin_cap_K": FROZEN_LAKE_SKIN_CAP_K,
    }


def _host64(value) -> np.ndarray:
    if hasattr(value, "get") and not isinstance(value, np.ndarray):
        value = value.get()
    return np.asarray(value, dtype=np.float64)


def _like(host: np.ndarray, template):
    dtype = getattr(template, "dtype", np.float64)
    if hasattr(template, "__cuda_array_interface__"):
        import cupy

        return cupy.asarray(host, dtype=dtype)
    return np.asarray(host, dtype=dtype)


def announce(receipt: Mapping[str, object], *, where: str) -> None:
    # "note:" and "warning:" lines are the ones `woof go` relays from a
    # passing preparation stage, so both answers reach the terminal.
    print(
        f"note: lake ice: {receipt['provider']} analysis valid "
        f"{receipt['analysis_valid_time']} ({receipt['file']}) froze "
        f"{receipt['frozen_cells']} of {receipt['lake_cells']} lake cell(s) "
        f"on {where} ({receipt['frozen_by_ice_pixels']} by its lake ice, "
        f"{receipt['frozen_by_snow_cover']} by snow cover over a lake its "
        f"water mask does not carry; {receipt['cells_newly_frozen']} were "
        f"open in the forcing); their skin temperature is at most "
        f"{FROZEN_LAKE_SKIN_CAP_K} K.  {WPS_SWITCH} keeps the forcing's "
        "lake ice instead.",
        file=sys.stderr)


#: The switch that selects ``wps`` on the GFS route, as the notes name it.
#: It was "[case_data] gfs_lake_ice", which a `woof domain` GFS config does
#: not carry and cannot carry without the ERA5 route's inputs (door item 7).
WPS_SWITCH = ('[ingest] gfs_lake_ice = "wps" in the experiment config '
              "(or woof go --lake-ice wps)")


def announce_no_lakes(*, where: str) -> None:
    print(f"note: lake ice: {where} has no land-use lake cells, so no "
          "lake-ice analysis was requested for it.", file=sys.stderr)


def announce_fallback(reason: str, *, where: str) -> None:
    print(
        f"warning: lake ice: no observed lake-ice analysis for {where}: "
        f"{reason}.  This run keeps the forcing's own lake ice (the WPS "
        "nearest-water search), so lakes the forcing does not resolve start "
        "as open water even if they are frozen.  Restore network access, or "
        f"put the analysis file in {default_cache_root()} (one folder per "
        "source row of woof/data/surface_obs/lake-ice-sources.v1.toml), "
        f"or set {WPS_SWITCH} to keep the forcing's ice without asking.",
        file=sys.stderr)


class LakeIceProvider:
    """What a preparation carries to every domain it initializes.

    The analysis is acquired the first time a domain WITH LAKE CELLS asks
    (:meth:`freeze`), once for the whole tree, so a nest is frozen from the
    same analysis on its own grid.  A domain with no lake cells asks for
    nothing.  Before this the analysis was downloaded at the start of
    every observed-choice GFS preparation, lakes or none (door item 7).

    ``observation`` is ``None`` when the choice is ``wps``, before any
    domain with lakes has asked, or when no analysis could be read
    (``fallback`` then says why).  A provider built with an
    ``observation`` or a ``fallback`` is already resolved and never asks.
    """

    def __init__(self, choice: str, observation: ObservedLakeIce | None = None,
                 fallback: str | None = None, *,
                 valid_time: datetime | None = None, where: str | None = None):
        self.choice = choice
        self._observation = observation
        self._fallback = fallback
        self._valid_time = valid_time
        self._where = where
        self._resolved = (choice == "wps" or valid_time is None
                          or observation is not None or fallback is not None)
        import threading
        # Nests are prepared on a thread pool (woof.ingest.nest_init); the
        # first domain with lakes acquires, the rest wait for its answer.
        self._lock = threading.Lock()

    def __deepcopy__(self, memo):
        # One provider per tree by design: a copied catalog still shares the
        # one analysis (and a lock cannot be copied).
        return self

    def __repr__(self) -> str:
        return (f"LakeIceProvider(choice={self.choice!r}, "
                f"observation={self._observation!r}, "
                f"fallback={self._fallback!r})")

    @property
    def observation(self) -> ObservedLakeIce | None:
        return self._observation

    @property
    def fallback(self) -> str | None:
        return self._fallback

    @property
    def wants_analysis(self) -> bool:
        """True when a grid with lake cells should call :meth:`freeze`."""
        return self.choice != "wps"

    def acquire(self) -> ObservedLakeIce | None:
        """The analysis, acquired on the first call; ``None`` when unreadable."""
        with self._lock:
            if not self._resolved:
                self._resolved = True
                try:
                    observation, attempts = acquire_observed_lake_ice(self._valid_time)
                except (OSError, ValueError) as error:
                    observation, attempts = None, [str(error)]
                if observation is None:
                    tried = ("; ".join(attempts[-4:]) if attempts
                             else "no source row answered")
                    self._fallback = f"none could be read ({tried})"
                    announce_fallback(self._fallback, where=self._where)
                else:
                    self._observation = observation
        return self._observation

    def identity(self) -> dict[str, object]:
        out: dict[str, object] = {"choice": self.choice}
        if self.observation is not None:
            out.update(self.observation.identity())
        if self.fallback is not None:
            out["fallback"] = self.fallback
        return out

    def freeze(self, fields, skin, *, lat, lon, lake_mask, dx_m, where,
               skin_name=None):
        """Apply the observation on one grid; identity when there is none.

        Returns ``(fields, skin, receipt)``; with no observation ``fields``
        and ``skin`` come back unchanged and the receipt is the identity.
        A grid with no lake cells asks for nothing and says so.  A read
        failure on this grid falls back loudly, for this grid only.
        """

        if self.wants_analysis and not np.asarray(lake_mask, dtype=bool).any():
            announce_no_lakes(where=where)
            return fields, skin, {"choice": self.choice, "lake_cells": 0,
                                  "analysis": "not requested: no lake cells"}
        if self.wants_analysis:
            self.acquire()
        if self.observation is None:
            return fields, skin, self.identity()
        try:
            decision = decide_lake_ice(self.observation, lat, lon, lake_mask, dx_m=dx_m)
        except (RuntimeError, ValueError, OSError) as error:
            announce_fallback(str(error), where=where)
            return fields, skin, {**self.identity(), "fallback": str(error)}
        fields, skin, applied = apply_lake_ice(
            fields, skin, decision.frozen, skin_name=skin_name)
        receipt = {"choice": self.choice, **decision.receipt, **applied}
        announce(receipt, where=where)
        return fields, skin, receipt


def resolve_lake_ice_provider(choice: str, valid_time: datetime, *,
                              where: str) -> LakeIceProvider:
    """The provider a preparation's choice asks for.

    Nothing is downloaded here: the analysis is acquired once, by the first
    domain of the tree that has lake cells (:meth:`LakeIceProvider.freeze`).
    """

    if choice not in LAKE_ICE_CHOICES:
        raise ValueError(
            "gfs_lake_ice must be one of "
            + ", ".join(repr(value) for value in LAKE_ICE_CHOICES))
    if choice == "wps":
        return LakeIceProvider(choice)
    return LakeIceProvider(choice, valid_time=valid_time, where=where)
