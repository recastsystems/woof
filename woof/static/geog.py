"""WPS_GEOG dataset readers (geogrid's source-data layer, CPU only).

Each WPS_GEOG dataset directory is self-describing: an ASCII ``index`` file
gives the storage layout (wordsize / signedness / endianness / tile
dimensions / border width / z levels), the value semantics (scale factor,
missing value, category range) and the regular lat/lon georeferencing
(``dx``/``dy``/``known_x``/``known_y``/``known_lat``/``known_lon``); the
data itself is a set of flat binary tiles named
``XSTART-XEND.YSTART-YEND`` (1-based, inclusive, x fastest within a row,
row 1 first, big-endian by default -- the write_geogrid.c convention).

Conventions implemented here (arbitrated against the bundle's geo_em files
where the format leaves room):

- ``missing_value`` is compared against the *raw* (unscaled) integers, then
  ``scale_factor`` is applied (e.g. greenfrac: raw byte 200 = missing,
  raw 0..100 scaled by 0.01).
- ``tile_bdr`` border cells are duplicated halo, never authoritative; the
  reader crops them and mosaics interior cells only.
- The index remains authoritative for ``tile_z`` when a tile contains extra
  complete, all-zero trailing planes.  Some official low-resolution WPS_GEOG
  soil tiles carry this padding; nonzero or partial surplus data is rejected.
- Negative ``dy`` (albedo-style north-to-south grids) is pure
  georeferencing: tile storage order is unchanged, only latlon<->xy flips.
- A dataset whose tiles span 360 degrees of longitude wraps in x.
"""
from __future__ import annotations

import contextlib
import contextvars
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


#: The MODIS 21-class land-use attributes WRF writes when the geography
#: names no other legend (wrfout and wrfinput globals). Lives here, in the
#: preprocessing package, so static builds and geography selection read it
#: without importing the forecast output writer (woof/io/wrfout.py, which
#: the standalone RW-WPS wheel must not carry).
DEFAULT_LANDUSE_ATTRS = {
    "MMINLU": "MODIFIED_IGBP_MODIS_NOAH",
    "ISWATER": 17,
    "ISLAKE": 21,
    "ISICE": 15,
    "ISURBAN": 13,
    "ISOILWATER": 14,
    "NUM_LAND_CAT": 21,
}

_TILE_RE = re.compile(r"^(\d{1,6})-(\d{1,6})\.(\d{1,6})-(\d{1,6})$")

_TRUE_STRINGS = ("yes", "true", ".true.", "1")


def _raw_index(path) -> dict[str, str]:
    """Read an index file into a {lowercased key: unquoted value} dict."""
    out: dict[str, str] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip()
        if len(val) >= 2 and val[0] in "'\"" and val[-1] == val[0]:
            val = val[1:-1]
        out[key.strip().lower()] = val
    return out


@dataclass(frozen=True)
class GeogIndex:
    """Typed view of a WPS_GEOG ``index`` file (defaults per WPS geogrid)."""
    type: str
    projection: str
    dx: float
    dy: float
    known_x: float
    known_y: float
    known_lat: float
    known_lon: float
    wordsize: int
    tile_x: int
    tile_y: int
    tile_z: int
    tile_z_start: int
    tile_z_end: int
    tile_bdr: int = 0
    signed: bool = False
    endian: str = "big"
    scale_factor: float = 1.0
    missing_value: float | None = None
    category_min: int | None = None
    category_max: int | None = None
    units: str = ""
    description: str = ""
    mminlu: str = ""
    iswater: int | None = None
    islake: int | None = None
    isice: int | None = None
    isurban: int | None = None
    row_order: str = "bottom_top"
    interp_option: str = ""

    @property
    def nz(self) -> int:
        return self.tile_z_end - self.tile_z_start + 1

    @property
    def dtype(self) -> np.dtype:
        base = {1: "i1", 2: "i2", 4: "i4"}[self.wordsize]
        if not self.signed:
            base = "u" + base[1:]
        order = ">" if self.endian == "big" else "<"
        return np.dtype(order + base) if self.wordsize > 1 else np.dtype(base)


def parse_index(path) -> GeogIndex:
    """Parse a WPS_GEOG index file into a :class:`GeogIndex`."""
    kv = _raw_index(path)

    def _f(key, default=None):
        return float(kv[key]) if key in kv else default

    def _i(key, default=None):
        return int(float(kv[key])) if key in kv else default

    if "tile_z" in kv:
        z0, z1 = 1, _i("tile_z")
    else:
        z0, z1 = _i("tile_z_start", 1), _i("tile_z_end", 1)
    return GeogIndex(
        type=kv.get("type", "continuous").lower(),
        projection=kv.get("projection", "regular_ll").lower(),
        dx=_f("dx"), dy=_f("dy"),
        known_x=_f("known_x", 1.0), known_y=_f("known_y", 1.0),
        known_lat=_f("known_lat"), known_lon=_f("known_lon"),
        wordsize=_i("wordsize"),
        tile_x=_i("tile_x"), tile_y=_i("tile_y"),
        tile_z=z1 - z0 + 1, tile_z_start=z0, tile_z_end=z1,
        tile_bdr=_i("tile_bdr", 0),
        signed=kv.get("signed", "no").strip().lower() in _TRUE_STRINGS,
        endian=kv.get("endian", "big").strip().lower(),
        scale_factor=_f("scale_factor", 1.0),
        missing_value=_f("missing_value"),
        category_min=_i("category_min"), category_max=_i("category_max"),
        units=kv.get("units", ""), description=kv.get("description", ""),
        mminlu=kv.get("mminlu", ""),
        iswater=_i("iswater"), islake=_i("islake"),
        isice=_i("isice"), isurban=_i("isurban"),
        row_order=kv.get("row_order", "bottom_top").strip().lower(),
        interp_option=kv.get("interp_option", "").strip(),
    )


#: Index projections this reader can actually resolve.  Everything else is
#: refused by name, through the one predicate below, so the geography
#: inventory and dataset construction cannot disagree about one tree.
SUPPORTED_INDEX_PROJECTIONS: tuple[str, ...] = ("regular_ll",)


def unsupported_projection_detail(projection: str | None) -> str | None:
    """Why a declared index projection cannot be read, else ``None``.

    One predicate, both doors: the geography inventory
    (:func:`woof.geog_assets.validate_dataset_dir`, reached by the
    fetch listing, the pre-fetch skip decision and post-stage
    validation) and :class:`GeogDataset` construction.  The detail names
    the projection the index declared, the breakage it prevents and the
    way out, because a refusal that names none of them is why a real
    WPS_GEOG tree of projected tiles reads as an internal error.
    """

    name = (projection or "regular_ll").strip().lower()
    if name in SUPPORTED_INDEX_PROJECTIONS:
        return None
    return (
        f"index declares projection {name!r}; this reader is "
        "equirectangular by construction, resolving a source cell by "
        "dividing a longitude difference by the index dx and taking the "
        "global axis from 360/dx by 180/dy, so a projected index would "
        "resolve the wrong source cells rather than fail. Stage the "
        "regular_ll variant of this dataset instead."
    )


@dataclass
class GeogWindow:
    """A mosaicked window of source data in native (raw) storage.

    ``raw`` has shape ``(nz, ny, nx)``; element ``[z, j, i]`` is source cell
    ``(x0 + i, y0 + j)`` (1-based dataset coordinates, x wrap already
    resolved).  :meth:`values` applies missing-value masking (on the raw
    integers) then the scale factor, returning float64 with NaN missing.
    """
    index: GeogIndex
    x0: int
    y0: int
    raw: np.ndarray
    coverage: np.ndarray | None = None

    @property
    def x1(self) -> int:
        return self.x0 + self.raw.shape[2] - 1

    @property
    def y1(self) -> int:
        return self.y0 + self.raw.shape[1] - 1

    def values(self, z: int = 0) -> np.ndarray:
        a = self.raw[z].astype(np.float64)
        if self.index.missing_value is not None:
            a[self.raw[z] == self.index.missing_value] = np.nan
        if self.index.scale_factor != 1.0:
            a *= self.index.scale_factor
        return a


class GeogDataset:
    """One WPS_GEOG dataset directory: index + tile inventory + windowing."""

    def __init__(self, path, *, sparse: bool | None = None):
        self.path = Path(path)
        self.index = parse_index(self.path / "index")
        raw_index = _raw_index(self.path / "index")
        declared = raw_index.get("sparse", raw_index.get("is_sparse", "no"))
        self.declared_sparse = (
            declared.strip().lower() in _TRUE_STRINGS
            if sparse is None else bool(sparse)
        )
        projection_detail = unsupported_projection_detail(
            self.index.projection)
        if projection_detail is not None:
            raise NotImplementedError(
                f"{self.path / 'index'}: {projection_detail}")
        self.tiles: dict[tuple[int, int], Path] = {}
        self._tile_cache: dict[tuple[int, int, bool], np.ndarray | None] = {}
        xs_min = ys_min = None
        xe_max = ye_max = 0
        for f in self.path.iterdir():
            m = _TILE_RE.match(f.name)
            if not m:
                continue
            xs, xe, ys, ye = (int(g) for g in m.groups())
            self.tiles[(xs, ys)] = f
            xs_min = xs if xs_min is None else min(xs_min, xs)
            ys_min = ys if ys_min is None else min(ys_min, ys)
            xe_max, ye_max = max(xe_max, xe), max(ye_max, ye)
        if not self.tiles:
            raise FileNotFoundError(f"no data tiles found in {self.path}")
        assert xs_min is not None and ys_min is not None
        self.tile_inventory_bounds = (xs_min, xe_max, ys_min, ye_max)

        def declared_cells(*keys: str) -> int | None:
            for key in keys:
                if key in raw_index:
                    value = int(float(raw_index[key]))
                    if value <= 0:
                        raise ValueError(
                            f"WPS GEOG index {self.path / 'index'} declares "
                            f"non-positive {key}={value}")
                    return value
            return None

        def regular_ll_cells(span: float, spacing: float) -> int | None:
            if not np.isfinite(spacing) or spacing == 0.0:
                return None
            cells = int(round(span / abs(spacing)))
            if cells <= 0:
                return None
            tolerance = max(1.0e-9, span * 2.0e-6)
            if abs(abs(spacing) * cells - span) > tolerance:
                return None
            return cells

        declared_nx = declared_cells("global_nx", "nx_global")
        declared_ny = declared_cells("global_ny", "ny_global")
        inferred_nx = regular_ll_cells(360.0, self.index.dx)
        inferred_ny = regular_ll_cells(180.0, self.index.dy)
        staged_inventory = (
            self.declared_sparse or xs_min > 1 or ys_min > 1)

        if declared_nx is not None and declared_nx < xe_max:
            raise ValueError(
                f"WPS GEOG global_nx={declared_nx} is smaller than tile "
                f"extent {xe_max} in {self.path}")
        if declared_ny is not None and declared_ny < ye_max:
            raise ValueError(
                f"WPS GEOG global_ny={declared_ny} is smaller than tile "
                f"extent {ye_max} in {self.path}")

        # A footprint-minimized WPS_GEOG staging tree keeps original global
        # tile coordinates.  Inferring dimensions solely from the largest
        # staged filename would disable longitude wrapping and can turn every
        # western-hemisphere lookup into an out-of-extent zero.  Preserve the
        # regular-LL global geometry when the tile origins prove staging (or
        # when explicit global dimensions are present).  A regional tree
        # starting at tile origin 1 retains its observed extent unless it
        # explicitly opts in or already spans the complete axis.
        use_inferred_nx = (
            inferred_nx is not None and inferred_nx >= xe_max
            and (staged_inventory or inferred_nx == xe_max))
        use_inferred_ny = (
            inferred_ny is not None and inferred_ny >= ye_max
            and (staged_inventory or inferred_ny == ye_max))
        self.nx_global = (
            declared_nx if declared_nx is not None else
            inferred_nx if use_inferred_nx else xe_max)
        self.ny_global = (
            declared_ny if declared_ny is not None else
            inferred_ny if use_inferred_ny else ye_max)
        inferred_extent_expands_inventory = (
            (use_inferred_nx and (xs_min > 1 or inferred_nx > xe_max))
            or (use_inferred_ny and (ys_min > 1 or inferred_ny > ye_max)))
        self.extent_basis = (
            "declared_global" if declared_nx is not None or declared_ny is not None
            else "regular_ll_staged_inventory"
            if staged_inventory and inferred_extent_expands_inventory
            else "regular_ll_complete"
            if use_inferred_nx or use_inferred_ny
            else "tile_inventory")
        #: does the dataset span 360 deg of longitude (x wraps)?
        self.wraps_x = (
            inferred_nx is not None and self.nx_global == inferred_nx)

    # -- georeferencing ------------------------------------------------------

    def latlon_to_xy(self, lat, lon):
        """(lat, lon) degrees -> fractional 1-based source coordinates."""
        idx = self.index
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        d = lon - idx.known_lon
        if self.wraps_x:
            d = np.mod(d, 360.0)
            x = idx.known_x + d / idx.dx
            x = np.where(x >= self.nx_global + 0.5, x - self.nx_global, x)
        else:
            d = np.mod(d + 180.0, 360.0) - 180.0
            x = idx.known_x + d / idx.dx
        y = idx.known_y + (lat - idx.known_lat) / idx.dy
        return x, y

    def xy_to_latlon(self, x, y):
        """Fractional 1-based source coordinates -> (lat, lon) degrees."""
        idx = self.index
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        lat = idx.known_lat + (y - idx.known_y) * idx.dy
        lon = idx.known_lon + (x - idx.known_x) * idx.dx
        lon = np.mod(lon + 180.0, 360.0) - 180.0
        return lat, lon

    # -- tile IO --------------------------------------------------------------

    def _read_tile(self, xs: int, ys: int,
                   include_border: bool = False) -> np.ndarray | None:
        """Read tile with 1-based origin (xs, ys).

        By default only the authoritative interior is returned.  With
        ``include_border``, WPS's duplicated tile halo is retained for the
        tile-local interpolation path.  Returns None for a sparse missing
        tile.
        """
        key = (xs, ys, include_border)
        if key in self._tile_cache:
            return self._tile_cache[key]
        f = self.tiles.get((xs, ys))
        if f is None:
            self._tile_cache[key] = None
            return None
        idx = self.index
        b = idx.tile_bdr
        ny, nx = idx.tile_y + 2 * b, idx.tile_x + 2 * b
        raw = np.fromfile(f, dtype=idx.dtype)
        plane_words = ny * nx
        expect = idx.nz * plane_words
        if raw.size != expect:
            if raw.size < expect or raw.size % plane_words:
                raise ValueError(f"tile {f} has {raw.size} words, expected "
                                 f"{expect} ({idx.nz}x{ny}x{nx})")
            actual_nz = raw.size // plane_words
            raw = raw.reshape(actual_nz, ny, nx)
            padding = raw[idx.nz:]
            nonzero = np.argwhere(padding != 0)
            if nonzero.size:
                z, y, x = (int(value) for value in nonzero[0])
                raise ValueError(
                    f"tile {f} has {actual_nz} complete z planes but its "
                    f"index declares {idx.nz}; undeclared trailing planes "
                    f"contain nonzero data (first at z={idx.nz + z + 1}, "
                    f"y={y + 1}, x={x + 1})")
            raw = raw[:idx.nz]
        else:
            raw = raw.reshape(idx.nz, ny, nx)
        if b and not include_border:
            raw = raw[:, b:-b, b:-b]
        if idx.row_order == "top_bottom":
            raw = raw[:, ::-1, :]
        self._tile_cache[key] = raw
        return raw

    def read_tile_window(self, xs: int, ys: int) -> GeogWindow | None:
        """Read one native tile, retaining its interpolation border.

        WPS ``get_point`` chooses a tile from the requested source
        coordinate and confines every fallback interpolator to that tile.
        The returned coordinate origin therefore includes ``tile_bdr``.
        """
        raw = self._read_tile(xs, ys, include_border=True)
        if raw is None:
            return None
        b = self.index.tile_bdr
        return GeogWindow(index=self.index, x0=xs - b, y0=ys - b, raw=raw)

    def tile_coverage_mask(self, x0: int, x1: int,
                           y0: int, y1: int) -> np.ndarray:
        """Boolean source-cell coverage for an inclusive requested window.

        Longitude wrapping follows :meth:`read_window`.  Cells outside a
        non-wrapping dataset, rows outside its latitude extent, and cells
        belonging to absent tiles are false.  This is metadata-only: tile
        payloads are not read, so preflight can prove completeness cheaply.
        """

        if x1 < x0 or y1 < y0:
            raise ValueError("empty window")
        x = np.arange(x0, x1 + 1, dtype=np.int64)
        y = np.arange(y0, y1 + 1, dtype=np.int64)
        if self.wraps_x:
            source_x = (x - 1) % self.nx_global + 1
            x_inside = np.ones(x.size, dtype=bool)
        else:
            source_x = x
            x_inside = (x >= 1) & (x <= self.nx_global)
        y_inside = (y >= 1) & (y <= self.ny_global)
        x_origins = ((source_x - 1) // self.index.tile_x
                     * self.index.tile_x + 1)
        y_origins = ((y - 1) // self.index.tile_y
                     * self.index.tile_y + 1)
        # Tile presence depends only on the (x_origin, y_origin) pair, and a
        # model window spans a couple of dozen distinct origins at most.  Probe
        # each distinct pair once and expand through the unique inverses rather
        # than asking the inventory the same question once per source cell: a
        # 30-arcsec domain window is millions of cells but ~15 origins.
        ux, xinv = np.unique(x_origins, return_inverse=True)
        uy, yinv = np.unique(y_origins, return_inverse=True)
        present = np.fromiter(
            ((int(xs), int(ys)) in self.tiles for ys in uy for xs in ux),
            dtype=bool, count=ux.size * uy.size,
        ).reshape(uy.size, ux.size)
        return (present[yinv.reshape(-1)][:, xinv.reshape(-1)]
                & x_inside[None, :] & y_inside[:, None])

    def _extent_mask(self, x0: int, x1: int,
                     y0: int, y1: int) -> np.ndarray:
        """Cells in a requested window that belong to the dataset extent."""

        x = np.arange(x0, x1 + 1, dtype=np.int64)
        y = np.arange(y0, y1 + 1, dtype=np.int64)
        x_inside = (np.ones(x.size, dtype=bool) if self.wraps_x else
                    (x >= 1) & (x <= self.nx_global))
        y_inside = (y >= 1) & (y <= self.ny_global)
        return y_inside[:, None] & x_inside[None, :]

    def missing_tiles(self, x0: int, x1: int, y0: int,
                      y1: int) -> tuple[tuple[int, int], ...]:
        """Expected tile origins absent from the in-extent window cells."""

        mask = self.tile_coverage_mask(x0, x1, y0, y1)
        missing_cells = ~mask & self._extent_mask(x0, x1, y0, y1)
        rows, cols = np.nonzero(missing_cells)
        if rows.size == 0:
            return ()
        x = x0 + cols.astype(np.int64)
        y = y0 + rows.astype(np.int64)
        if self.wraps_x:
            x = (x - 1) % self.nx_global + 1
        xs = (x - 1) // self.index.tile_x * self.index.tile_x + 1
        ys = (y - 1) // self.index.tile_y * self.index.tile_y + 1
        # unique over (y_origin, x_origin) rows sorts lexicographically on that
        # column order, which is exactly the (item[1], item[0]) ordering the
        # scalar implementation produced.
        origins = np.unique(np.stack((ys, xs), axis=1), axis=0)
        return tuple((int(row[1]), int(row[0])) for row in origins)

    def required_tile_origins(
            self, x0: int, x1: int, y0: int,
            y1: int) -> tuple[tuple[int, int], ...]:
        """Tile origins intersecting the in-extent part of a source window.

        Unlike :meth:`missing_tiles`, this includes present and absent
        origins.  It is metadata-only and deliberately independent of the
        dataset's ``sparse`` declaration so a staged, footprint-minimized
        tree can prove that every tile required by a model domain is present.
        """

        if x1 < x0 or y1 < y0:
            raise ValueError("empty window")
        x = np.arange(x0, x1 + 1, dtype=np.int64)
        y = np.arange(y0, y1 + 1, dtype=np.int64)
        if self.wraps_x:
            source_x = (x - 1) % self.nx_global + 1
        else:
            source_x = x[(x >= 1) & (x <= self.nx_global)]
        source_y = y[(y >= 1) & (y <= self.ny_global)]
        if source_x.size == 0 or source_y.size == 0:
            return ()
        x_origins = np.unique(
            (source_x - 1) // self.index.tile_x * self.index.tile_x + 1)
        y_origins = np.unique(
            (source_y - 1) // self.index.tile_y * self.index.tile_y + 1)
        return tuple(
            (int(xs), int(ys))
            for ys in y_origins
            for xs in x_origins
        )

    def read_window(self, x0: int, x1: int, y0: int, y1: int) -> GeogWindow:
        """Mosaic the (1-based, inclusive) index window ``[x0..x1, y0..y1]``.

        x wraps modulo the global width for global datasets; rows outside
        ``[1, ny_global]`` and absent tiles are filled with the dataset's
        missing value (or 0 when the index declares none).
        """
        idx = self.index
        if x1 < x0 or y1 < y0:
            raise ValueError("empty window")
        nxw, nyw = x1 - x0 + 1, y1 - y0 + 1
        # A complete longitude band plus interpolation margins legitimately
        # repeats seam columns. Periodic mosaics already split every wrap
        # below; only a nonperiodic dataset has a hard single-width limit.
        if nxw > self.nx_global and not self.wraps_x:
            raise ValueError("window wider than the global grid")
        coverage = self.tile_coverage_mask(x0, x1, y0, y1)
        unexplained = ~coverage & self._extent_mask(x0, x1, y0, y1)
        if not self.declared_sparse and bool(np.any(unexplained)):
            j, i = np.argwhere(unexplained)[0]
            source_x, source_y = x0 + int(i), y0 + int(j)
            if self.wraps_x:
                source_x = (source_x - 1) % self.nx_global + 1
            tile_x = ((source_x - 1) // idx.tile_x * idx.tile_x + 1)
            tile_y = ((source_y - 1) // idx.tile_y * idx.tile_y + 1)
            raise FileNotFoundError(
                f"WPS GEOG dataset {self.path} has unexplained fill at "
                f"source index (x={source_x}, y={source_y}); expected tile "
                f"origin ({tile_x}, {tile_y}) is absent. Declare this "
                "dataset sparse only when missing coverage is intentional."
            )
        fill = idx.missing_value if idx.missing_value is not None else 0
        out = np.full((idx.nz, nyw, nxw), fill, dtype=idx.dtype)

        # split the x range into wrap-contiguous segments of absolute coords
        if self.wraps_x:
            segs = []
            a = x0
            while a <= x1:
                wa = (a - 1) % self.nx_global + 1
                run = min(x1 - a, self.nx_global - wa) + 1
                segs.append((a, wa, run))
                a += run
        else:
            segs = [(x0, x0, nxw)]

        ty, tx = idx.tile_y, idx.tile_x
        yy0, yy1 = max(y0, 1), min(y1, self.ny_global)
        for ys in range((yy0 - 1) // ty * ty + 1, yy1 + 1, ty):
            for abs_x0, wx0, run in segs:
                wx1 = wx0 + run - 1
                for xs in range((wx0 - 1) // tx * tx + 1, wx1 + 1, tx):
                    tile = self._read_tile(xs, ys)
                    if tile is None:
                        continue
                    # overlap in wrapped source coordinates
                    ox0, ox1 = max(wx0, xs), min(wx1, xs + tx - 1)
                    oy0, oy1 = max(yy0, ys), min(yy1, ys + ty - 1)
                    if ox1 < ox0 or oy1 < oy0:
                        continue
                    di = (abs_x0 - x0) + (ox0 - wx0)
                    dj = oy0 - y0
                    out[:, dj:dj + oy1 - oy0 + 1,
                        di:di + ox1 - ox0 + 1] = \
                        tile[:, oy0 - ys:oy1 - ys + 1, ox0 - xs:ox1 - xs + 1]
        return GeogWindow(index=idx, x0=x0, y0=y0, raw=out,
                          coverage=coverage)


def tile_coverage_mask(dataset: GeogDataset | str | Path,
                       x0: int, x1: int, y0: int, y1: int, *,
                       sparse: bool | None = None) -> np.ndarray:
    """Public metadata-only WPS tile-coverage mask API."""

    ds = (dataset if isinstance(dataset, GeogDataset)
          else GeogDataset(dataset, sparse=sparse))
    return ds.tile_coverage_mask(x0, x1, y0, y1)


# -- the 180-degree seam of global category tiles ------------------------------

#: Bumped whenever :func:`repair_seam_columns` changes; part of the key of
#: every repaired copy, so a copy made under an older rule is never read.
SEAM_REPAIR_VERSION = 1

#: ``off`` reads the category tiles at 180 degrees exactly as published, the
#: way WPS geogrid reads them; for a comparison with geogrid only.
SEAM_REPAIR_ENV = "GPUWM_GEOG_SEAM_REPAIR"

#: Folder the repaired copies are kept in (default: the user cache's
#: ``woof/geog-seam``).
SEAM_CACHE_ENV = "WOOF_GEOG_SEAM_CACHE"

_SEAM_REPAIRED: dict[str, Path] = {}


def repair_seam_columns(last, first, wet, *, water):
    """Repair water, lake and ice written into the columns at 180 degrees.

    THE BREAKAGE THIS REPAIRS.  WPS_GEOG's 30 arc-second MODIS land use
    (``modis_landuse_20class_30s`` and ``..._with_lakes``) carries water
    (17), lake (21) or ice (15) in its last column (x = 43200) and its
    first two (x = 1, 2) where the ground either side is land: ice on
    tropical Taveuni and Vanua Levu (16.1-17.0 S), and lake pixels in
    Chukotka and on Wrangel Island (65-71 N).  Read from the tiles
    themselves, 2026-11-13: this rule changes 85 pixels of
    ``..._with_lakes``.  geogrid reads the tiles as they are, so a
    1 km domain on Taveuni got a 1-2 cell strip of water and ice down the
    island along 180 degrees.

    What it costs: the same rule run on interior columns (0, 90 W, 150 W,
    70 E) changes 33 to 270 pixels each, nearly all real lakes one or two
    pixels wide.  It cannot tell such a lake sitting exactly on 180 from
    the defect, so it is applied at 180 only, where a few lake pixels in
    Chukotka may be read as tundra.

    ``last``: (ny, 2) categories of columns nx-1 and nx, the last two
    (just west of 180).  ``first``: (ny, 3) categories of columns 1, 2
    and 3 (just east of 180).  -1 marks a row whose tile is not staged.
    ``wet``: the index's water, lake and ice classes; ``water``: its
    water class (``iswater``), which rule 2 never changes.

    The rule, for a pixel of column nx, 1 or 2 in a wet class; its clean
    neighbours are columns nx-1 and 3 in its own row and the rows above
    and below:

    1. When more than half of its clean neighbours are land it is read as
       land: the land pixel inland of it on its own side, or when that one
       is wet too the commonest land class among the neighbours (the lower
       class on a tie).  A real coast along 180 keeps its water: its
       neighbours are not mostly land.
    2. Otherwise a lake or ice pixel none of whose clean neighbours is of
       its class takes the pixel inland of it on its own side.  This is the
       coast: ice along the shore of tropical Taveuni, which rule 1 leaves
       (half its neighbours are sea), became land ice in the land model.
       Ice or lake with its own class beside it (the Ross Ice Shelf) stays.

    Returns the repaired ``(last, first)`` and one ``(row, column, was,
    now)`` per changed pixel (row 0-based, column ``"nx"``, ``"1"`` or
    ``"2"``).
    """
    last = np.array(last, dtype=np.int64, copy=True)
    first = np.array(first, dtype=np.int64, copy=True)
    if (last.ndim != 2 or last.shape[1] != 2
            or first.shape != (last.shape[0], 3)):
        raise ValueError("last must be (ny, 2) and first (ny, 3)")
    wet_classes = np.asarray(sorted({int(v) for v in wet}), dtype=np.int64)
    wetset = {int(v) for v in wet_classes}
    clean = np.stack((last[:, 0], first[:, 2]), axis=1)
    suspects = ((last, 1, 0, "nx"), (first, 0, 2, "1"), (first, 1, 2, "2"))
    rows = np.nonzero(np.isin(last[:, 1], wet_classes)
                      | np.isin(first[:, 0], wet_classes)
                      | np.isin(first[:, 1], wet_classes))[0]
    changes = []
    for j in rows:
        j = int(j)
        hood = clean[max(0, j - 1):j + 2].ravel()
        hood = hood[hood >= 0]
        land = hood[~np.isin(hood, wet_classes)]
        mostly_land = 2 * land.size > hood.size
        if mostly_land:
            values, counts = np.unique(land, return_counts=True)
            common = int(values[int(np.argmax(counts))])
        for array, column, inland, name in suspects:
            was = int(array[j, column])
            if was < 0 or was not in wetset:
                continue
            side = int(array[j, inland])
            if mostly_land:
                now = side if side >= 0 and side not in wetset else common
            elif was != water and side >= 0 and not (hood == was).any():
                now = side
            else:
                continue
            if now != was:
                array[j, column] = now
                changes.append((j, name, was, now))
    return last, first, changes


def default_geog_cache_root() -> Path:
    """The per-user folder repaired GEOG copies are kept in."""
    import os
    if os.environ.get(SEAM_CACHE_ENV):
        return Path(os.environ[SEAM_CACHE_ENV])
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
    elif os.environ.get("XDG_CACHE_HOME"):
        base = Path(os.environ["XDG_CACHE_HOME"])
    else:
        base = Path.home() / ".cache"
    return base / "woof" / "geog-seam"


#: False inside :func:`quiet_seam_notes`: the repaired copy is still read,
#: and its ``note:`` lines are not printed.
_SEAM_NOTES = contextvars.ContextVar("geog_seam_notes", default=True)


@contextlib.contextmanager
def quiet_seam_notes():
    """Read the repaired copies without their ``note: geography:`` lines.

    For a domain that does not reach 180 degrees: every domain reads the
    repaired copy, so that all builds read one set of bytes, but the lines
    describe pixels at 180 degrees that such a domain never samples.

    THE BREAKAGE THIS PREVENTS (3.0.3 candidate, 2026-11-14): a 12 km and
    3 km forecast over Iowa printed, on every preparation, three lines
    about water pixels "in the columns at 180 degrees" and a soil void
    "west of 180 degrees".  A ``warning:`` line still prints.
    """

    token = _SEAM_NOTES.set(False)
    try:
        yield
    finally:
        _SEAM_NOTES.reset(token)


def _seam_note(text: str, *, kind: str = "note") -> None:
    # "note:" and "warning:" lines are the ones `woof go` and `woof prep`
    # relay from a passing preparation stage (go_cli.LaunchOutput,
    # prep_output), so the line reaches the terminal and the run's own log.
    # A bare "geography:" line reached only the stage log, which a finished
    # run does not keep.
    import sys
    if kind == "note" and not _SEAM_NOTES.get():
        return
    print(f"{kind}: geography: {text}", file=sys.stderr, flush=True)


def _seam_cache_roots(cache_root) -> list[Path]:
    """Where a repaired copy may be written, best first: the named folder
    (or the user cache), then a per-user folder under the system temp."""
    import getpass
    import tempfile
    roots = [Path(cache_root) if cache_root else default_geog_cache_root()]
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - no user name is not a reason to fail
        user = "user"
    spare = Path(tempfile.gettempdir()) / f"gpuwm-geog-seam-{user}"
    if spare not in roots:
        roots.append(spare)
    return roots


def _write_json_atomic(path: Path, payload) -> None:
    import json
    import os
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    os.replace(partial, path)


def seam_repaired_dataset(path, *, cache_root=None) -> Path:
    """The directory a build reads a global category dataset from: ``path``
    itself, or a copy whose columns at 180 degrees are repaired
    (:func:`repair_seam_columns`).

    The copy holds the dataset's ``index``, a link to every tile that the
    repair leaves alone and a rewritten file for each tile it changes;
    ``seam-repair.json`` beside it names every changed pixel.  It is made
    once (its key is the rule's version, the dataset's path, index and tile
    names, and the edge tiles' sizes and times) and read by every later
    build, so a corridor crop and a direct build read the same bytes.  A
    dataset the rule does not apply to (not global, not categorical, no
    water, lake or ice class) or does not change is read where it is, and
    nothing is written for it.

    The copy is written under ``cache_root`` (default: ``GPUWM_GEOG_SEAM_
    CACHE``, else the user cache), else under the system temp folder; when
    neither can be written the build stops and says so, rather than read
    the tiles as published.  The line saying what was repaired starts
    ``note: geography:``, which ``woof go`` relays to the terminal and the
    run's log.
    """
    import hashlib
    import json
    import os

    path = Path(path)
    if os.environ.get(SEAM_REPAIR_ENV, "").strip().lower() in (
            "0", "off", "no", "false"):
        return path
    try:
        idx = parse_index(path / "index")
    except (OSError, ValueError, KeyError, TypeError):
        return path
    wet = {int(v) for v in (idx.iswater, idx.islake, idx.isice)
           if v is not None}
    if (idx.type != "categorical" or not wet or idx.tile_bdr or idx.nz != 1
            or not idx.tile_x or idx.tile_x < 3
            or idx.projection != "regular_ll"):
        return path
    try:
        ds = GeogDataset(path)
    except (OSError, ValueError, NotImplementedError):
        return path
    nx, ny = int(ds.nx_global), int(ds.ny_global)
    if not ds.wraps_x or nx % idx.tile_x:
        return path
    # Columns nx and 1 are the ones either side of 180 degrees only when
    # column 1 starts at 180 W, as in every WPS land-use set.  An archive
    # whose column 1 sits elsewhere is read as stored here (rw-mpas's reader
    # finds the line wherever it falls).
    # Measured the short way round, in pixels, so a geoindex's rounded dx
    # is not multiplied by a whole globe of pixels.
    edge = (float(idx.known_lon) - (float(idx.known_x) - 1.0) * float(idx.dx)
            - 0.5 * float(idx.dx))
    offset = (((edge + 360.0) % 360.0) - 180.0) / float(idx.dx)
    if abs(offset) > 1.0e-2:
        return path
    last_x = nx - idx.tile_x + 1
    bands = sorted(ys for (xs, ys) in ds.tiles
                   if xs == last_x and (1, ys) in ds.tiles)
    if not bands:
        return path
    names = sorted(f.name for f in ds.tiles.values())
    stamp = [f"seam-repair v{SEAM_REPAIR_VERSION}", str(path.resolve()),
             (path / "index").read_text(), *names]
    for ys in bands:
        for xs in (last_x, 1):
            st = ds.tiles[(xs, ys)].stat()
            stamp.append(f"{xs}.{ys}:{st.st_size}:{st.st_mtime_ns}")
    key = hashlib.sha256("\n".join(stamp).encode("utf-8")).hexdigest()[:20]
    if key in _SEAM_REPAIRED:
        return _SEAM_REPAIRED[key]
    roots = _seam_cache_roots(cache_root)
    for root in roots:
        home = root / key
        target = home / path.name
        try:
            receipt = json.loads(
                (home / "seam-repair.json").read_text(encoding="utf-8"))
            if receipt.get("changed_pixels") and (target / "index").is_file():
                _SEAM_REPAIRED[key] = target
                _seam_note(f"{path.name}: {receipt['changed_pixels']} "
                           "pixel(s) of water, lake or ice on land in the "
                           "columns at 180 degrees read as land, from "
                           f"{target}")
                return target
        except (OSError, ValueError, KeyError, TypeError):
            pass

    # Scan the two columns west of the line and the three east of it, in
    # every band whose tiles are staged on both sides.
    last = np.full((ny, 2), -1, dtype=np.int64)
    first = np.full((ny, 3), -1, dtype=np.int64)
    tiles = {}
    for ys in bands:
        tl, tf = ds._read_tile(last_x, ys), ds._read_tile(1, ys)
        if tl is None or tf is None:
            continue
        n = min(tl.shape[1], tf.shape[1], ny - ys + 1)
        last[ys - 1:ys - 1 + n] = tl[0, :n, -2:]
        first[ys - 1:ys - 1 + n] = tf[0, :n, :3]
        tiles[ys] = (tl, tf, n)
    new_last, new_first, changes = repair_seam_columns(
        last, first, wet, water=idx.iswater)
    lat0, dy = float(idx.known_lat), float(idx.dy)
    record = {
        "schema": "gpuwm-geog-seam-repair.v1",
        "rule_version": SEAM_REPAIR_VERSION,
        "dataset": path.name,
        "source": str(path.resolve()),
        "wet_classes": sorted(wet),
        "rule": ("a pixel of column nx, 1 or 2 in a water, lake or ice class "
                 "is read as land when more than half of its clean "
                 "neighbours (columns nx-1 and 3, rows j-1..j+1) are land: "
                 "the land pixel inland of it on its own side, else the "
                 "commonest land class among the neighbours; otherwise a "
                 "lake or ice pixel with none of its class among those "
                 "neighbours takes the pixel inland of it on its own side"),
        "changed_pixels": len(changes),
        "changes": [{"y": j + 1, "lat": round(lat0 + j * dy, 4),
                     "column": column, "was": was, "now": now}
                    for j, column, was, now in changes],
        "tiles_rewritten": [],
    }
    if not changes:
        _SEAM_REPAIRED[key] = path
        return path
    rewritten = {}
    for ys, (tl, tf, n) in tiles.items():
        rows = slice(ys - 1, ys - 1 + n)
        for xs, tile, before, after, cols in (
                (last_x, tl, last[rows], new_last[rows], slice(-2, None)),
                (1, tf, first[rows], new_first[rows], slice(0, 3))):
            if np.array_equal(before, after):
                continue
            fixed = np.array(tile, copy=True)
            fixed[0, :n, cols] = after.astype(fixed.dtype)
            if idx.row_order == "top_bottom":
                fixed = fixed[:, ::-1, :]
            rewritten[ds.tiles[(xs, ys)].name] = fixed
    record["tiles_rewritten"] = sorted(rewritten)
    failures = []
    for root in roots:
        home = root / key
        target = home / path.name
        try:
            _write_seam_copy(path, home, target, names, rewritten, idx)
            record["copy"] = str(target)
            _write_json_atomic(home / "seam-repair.json", record)
        except OSError as error:
            failures.append(f"{home} ({error})")
            continue
        _SEAM_REPAIRED[key] = target
        _seam_note(f"{path.name}: {len(changes)} pixel(s) of water, lake or "
                   "ice on land in the columns at 180 degrees read as land, "
                   f"from {target}")
        return target
    # THE BREAKAGE THIS REFUSAL PREVENTS: reading the tiles as published
    # here puts the strip of water and ice back down Taveuni along 180
    # degrees, with nothing in the run to say so.  It is reached only when
    # neither the cache nor the system temp folder can be written.
    raise OSError(
        f"{path.name}: the columns at 180 degrees carry {len(changes)} "
        "pixel(s) of water, lake or ice on land, and the repaired copy the "
        "build reads could not be written in any of: "
        + "; ".join(failures)
        + f".  Set {SEAM_CACHE_ENV} to a writable folder, or "
        f"{SEAM_REPAIR_ENV}=off to read the tiles as published (geogrid's "
        "strip along 180 included)")


def _write_seam_copy(path: Path, home: Path, target: Path, names, rewritten,
                     idx) -> None:
    """Write the repaired copy of ``path`` at ``target`` (atomically, by a
    rename): its index, the ``rewritten`` tiles, and a link to every other
    tile (a file link only: no folder of the copy links into WPS_GEOG, so
    no recursive delete of the cache can reach the published tiles)."""
    import os
    import shutil
    home.mkdir(parents=True, exist_ok=True)
    partial = home / f".{path.name}.{os.getpid()}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    shutil.copyfile(path / "index", partial / "index")
    for name in names:
        source = path / name
        if name in rewritten:
            np.ascontiguousarray(rewritten[name]).astype(
                idx.dtype).tofile(partial / name)
            continue
        try:
            os.symlink(source.resolve(), partial / name)
        except OSError:
            os.link(source.resolve(), partial / name)
    try:
        os.rename(partial, target)
    except OSError:
        shutil.rmtree(partial, ignore_errors=True)
        if not (target / "index").is_file():
            raise


# -- the void of the SoilGrids texture sets beside 180 degrees ----------------

#: How far from 180 degrees a soil void is looked for, in columns (1 degree
#: at 30 arc-seconds); the reference column is this far out.  The three
#: numbers are rw-mpas's (static_geog.rs SOIL_VOID_*), so a hex static and an
#: ArWen domain read one soil map.
SOIL_VOID_REACH = 120
#: A column is void when it holds the water class on at least this share of
#: the rows where the reference column holds land (elsewhere: 1-5 %, coasts).
SOIL_VOID_SHARE = 0.95
#: Fewer land rows at the reference than this and no void is declared.
SOIL_VOID_MIN_ROWS = 50
#: WPS's soil water category (GEOGRID.TBL ``isoilwater``), for a soil index
#: that declares none.
SOIL_WATER = 14
#: Bumped whenever :func:`repair_soil_void` changes.
SOIL_VOID_VERSION = 1


def repair_soil_void(west, east, water=SOIL_WATER):
    """Fill the void a soil set carries beside 180 degrees.

    THE BREAKAGE THIS REPAIRS.  WPS_GEOG's SoilGrids texture sets
    (``soilgrids/texture_top``, ``texture_bot``, ``texture_layer1..4``) hold
    the water class (14) on 96 % of the land pixels of the 0.5 degrees west
    of 180 (columns 43143-43200; 1-2 % in every other half degree), from
    60 S to 75 N (read from the tiles, 2026-11-13).  The WRF route gave
    every land cell
    there silty clay loam (real.exe's soil for land on water soil,
    :func:`woof.core.landuse.soil_category_matched_to_land`) instead of its
    own soil; MPAS's rule made it ocean (Vanua Levu's east end, west
    Taveuni, a 20 km strip of Chukotka).

    ``west``: (ny, REACH + 1) categories of the columns from REACH west of
    the line to the column on it; ``east``: (ny, REACH + 1) from the column
    on the line to REACH east of it; -1 where no tile is staged.  The
    outermost column of each is the reference.  A void is the run of columns
    from the line outward whose water share on the reference's land rows is
    at least SOIL_VOID_SHARE.  Each water pixel of the void takes the nearest
    soil class on its row outside the void that is not water (west of the
    void or east of the line, the west on a tie, within the reach); sea there
    is water in the land use, which decides the land mask.

    Returns ``(west, east, info)``: repaired copies and a dict with the
    void's widths (``west``, ``east``), its least water share and the count
    of ``filled`` pixels; ``info`` is None when there is no void.
    """
    west = np.array(west, dtype=np.int64, copy=True)
    east = np.array(east, dtype=np.int64, copy=True)
    ny, width = west.shape
    reach = width - 1
    if east.shape != (ny, width) or reach < 1:
        raise ValueError("west and east must both be (ny, reach + 1)")

    def void_run(reference, toward_line):
        rows = (reference >= 0) & (reference != water)
        if int(rows.sum()) < SOIL_VOID_MIN_ROWS:
            return 0, 0.0
        run, least = 0, 1.0
        for column in toward_line:
            share = float((column[rows] == water).sum()) / float(rows.sum())
            if share < SOIL_VOID_SHARE:
                break
            run += 1
            least = min(least, share)
        return run, least

    nw, w_share = void_run(west[:, 0], [west[:, reach - k] for k in range(reach)])
    ne, e_share = void_run(east[:, reach], [east[:, k] for k in range(reach)])
    if nw == 0 and ne == 0:
        return west, east, None
    # Per row: the nearest non-water class outside the void and how many
    # columns beyond the void's edge it lies.
    w_val = np.full(ny, -1, dtype=np.int64)
    w_far = np.zeros(ny, dtype=np.int64)
    e_val = np.full(ny, -1, dtype=np.int64)
    e_far = np.zeros(ny, dtype=np.int64)
    for j in range(ny):
        for extra in range(reach - nw + 1):
            v = west[j, reach - nw - extra]
            if v >= 0 and v != water:
                w_val[j], w_far[j] = v, extra
                break
        for extra in range(reach - ne + 1):
            v = east[j, ne + extra]
            if v >= 0 and v != water:
                e_val[j], e_far[j] = v, extra
                break
    filled = 0
    # Void pixels: west offset d (0 on the line) -> column reach - d of west;
    # east offset d -> column d of east.
    for side, run in (("west", nw), ("east", ne)):
        for d in range(run):
            if side == "west":
                to_west, to_east = nw - d, d + 1 + ne
                column = west[:, reach - d]
            else:
                to_west, to_east = d + 1 + nw, ne - d
                column = east[:, d]
            take_west = (w_val >= 0) & ((e_val < 0)
                                        | (to_west + w_far <= to_east + e_far))
            take_east = (e_val >= 0) & ~take_west
            value = np.where(take_west, w_val, np.where(take_east, e_val, -1))
            hit = (column == water) & (value >= 0) & (value != water)
            column[hit] = value[hit]
            filled += int(hit.sum())
    share = (e_share if nw == 0 else w_share if ne == 0
             else min(w_share, e_share))
    return west, east, {"west": nw, "east": ne, "share": share,
                        "filled": filled}


def soil_void_repaired_dataset(path, *, cache_root=None) -> Path:
    """The directory a build reads a soil category set from: ``path``
    itself, or a copy whose void beside 180 degrees is filled
    (:func:`repair_soil_void`), made, cached, announced and refused the way
    :func:`seam_repaired_dataset` makes its copy.  A set with no void, or
    one the rule does not apply to (not global, not one plane of whole
    tiles, column 1 not at 180 W), is read where it is.
    """
    import hashlib
    import json
    import os

    path = Path(path)
    if os.environ.get(SEAM_REPAIR_ENV, "").strip().lower() in (
            "0", "off", "no", "false"):
        return path
    try:
        idx = parse_index(path / "index")
    except (OSError, ValueError, KeyError, TypeError):
        return path
    reach = SOIL_VOID_REACH
    # An index with no tile_x is not a set of whole tiles: it is read where
    # it is, and the build that opens it reports the index in its own words.
    if (idx.type != "categorical" or idx.mminlu or idx.tile_bdr
            or idx.nz != 1 or not idx.tile_x or idx.tile_x <= reach
            or idx.projection != "regular_ll"):
        return path
    try:
        ds = GeogDataset(path)
    except (OSError, ValueError, NotImplementedError):
        return path
    nx, ny = int(ds.nx_global), int(ds.ny_global)
    if not ds.wraps_x or nx % idx.tile_x:
        return path
    edge = (float(idx.known_lon) - (float(idx.known_x) - 1.0) * float(idx.dx)
            - 0.5 * float(idx.dx))
    if abs((((edge + 360.0) % 360.0) - 180.0) / float(idx.dx)) > 1.0e-2:
        return path
    water = SOIL_WATER if idx.iswater is None else int(idx.iswater)
    last_x = nx - idx.tile_x + 1
    bands = sorted(ys for (xs, ys) in ds.tiles
                   if xs == last_x and (1, ys) in ds.tiles)
    if not bands:
        return path
    names = sorted(f.name for f in ds.tiles.values())
    stamp = [f"soil-void v{SOIL_VOID_VERSION}", str(path.resolve()),
             (path / "index").read_text(), *names]
    for ys in bands:
        for xs in (last_x, 1):
            st = ds.tiles[(xs, ys)].stat()
            stamp.append(f"{xs}.{ys}:{st.st_size}:{st.st_mtime_ns}")
    key = hashlib.sha256("\n".join(stamp).encode("utf-8")).hexdigest()[:20]
    if key in _SEAM_REPAIRED:
        return _SEAM_REPAIRED[key]
    roots = _seam_cache_roots(cache_root)

    def say(info, target):
        _seam_note(f"{path.name}: {info['west']} column(s) west and "
                   f"{info['east']} east of 180 degrees hold the water class "
                   f"on {100.0 * info['share']:.1f}% of the rows where land "
                   "lies 1 degree away (a void in the source); their water "
                   "pixels are read as the nearest soil class on the row "
                   "outside the void that is not water, within 1 degree: "
                   f"{info['filled']} pixel(s), from {target}")

    for root in roots:
        home = root / key
        target = home / path.name
        try:
            receipt = json.loads(
                (home / "seam-repair.json").read_text(encoding="utf-8"))
            if receipt.get("changed_pixels") and (target / "index").is_file():
                _SEAM_REPAIRED[key] = target
                say(receipt["void"], target)
                return target
        except (OSError, ValueError, KeyError, TypeError):
            pass

    west = np.full((ny, reach + 1), -1, dtype=np.int64)
    east = np.full((ny, reach + 1), -1, dtype=np.int64)
    tiles = {}
    for ys in bands:
        tl, tf = ds._read_tile(last_x, ys), ds._read_tile(1, ys)
        if tl is None or tf is None:
            continue
        n = min(tl.shape[1], tf.shape[1], ny - ys + 1)
        west[ys - 1:ys - 1 + n] = tl[0, :n, -(reach + 1):]
        east[ys - 1:ys - 1 + n] = tf[0, :n, :reach + 1]
        tiles[ys] = (tl, tf, n)
    new_west, new_east, info = repair_soil_void(west, east, water)
    if info is None or not info["filled"]:
        _SEAM_REPAIRED[key] = path
        return path
    rewritten = {}
    for ys, (tl, tf, n) in tiles.items():
        rows = slice(ys - 1, ys - 1 + n)
        for xs, tile, before, after, cols in (
                (last_x, tl, west[rows], new_west[rows],
                 slice(-(reach + 1), None)),
                (1, tf, east[rows], new_east[rows], slice(0, reach + 1))):
            if np.array_equal(before, after):
                continue
            fixed = np.array(tile, copy=True)
            fixed[0, :n, cols] = after.astype(fixed.dtype)
            if idx.row_order == "top_bottom":
                fixed = fixed[:, ::-1, :]
            rewritten[ds.tiles[(xs, ys)].name] = fixed
    record = {
        "schema": "gpuwm-geog-soil-void-repair.v1",
        "rule_version": SOIL_VOID_VERSION,
        "dataset": path.name,
        "source": str(path.resolve()),
        "water_class": water,
        "rule": ("columns from 180 degrees outward that hold the water class "
                 f"on at least {SOIL_VOID_SHARE:.0%} of the rows where the "
                 f"column {reach} away holds land are a void; each water "
                 "pixel of the void takes the nearest class on its row "
                 "outside the void that is not water (the west on a tie)"),
        "void": info,
        "changed_pixels": info["filled"],
        "tiles_rewritten": sorted(rewritten),
    }
    failures = []
    for root in roots:
        home = root / key
        target = home / path.name
        try:
            _write_seam_copy(path, home, target, names, rewritten, idx)
            record["copy"] = str(target)
            _write_json_atomic(home / "seam-repair.json", record)
        except OSError as error:
            failures.append(f"{home} ({error})")
            continue
        _SEAM_REPAIRED[key] = target
        say(info, target)
        return target
    # THE BREAKAGE THIS REFUSAL PREVENTS: reading the void as published gives
    # the land beside 180 degrees water soil, with nothing in the run to say
    # so.  Reached only when neither the cache nor the system temp folder
    # can be written.
    raise OSError(
        f"{path.name}: {info['west'] + info['east']} column(s) beside 180 "
        "degrees are a void of water soil, and the repaired copy the build "
        "reads could not be written in any of: " + "; ".join(failures)
        + f".  Set {SEAM_CACHE_ENV} to a writable folder, or "
        f"{SEAM_REPAIR_ENV}=off to read the tiles as published")
