"""``woof render`` -- forecast product PNGs from wrfout NetCDF files.

Two engines render:

* ``rust`` -- the vendored Rusty Weather production renderer
  (``tools/rustwx``, :mod:`woof.rustwx`): the campaign product sheets'
  quality tier, with basemaps and typography.  Its vendored catalog
  carries 324 entries, 151 of which the runtime lister evaluates as
  implicit-render candidates per file -- surface fields, the full
  200-850 mb isobaric chart families, CAPE/CIN/SRH/shear/STP severe
  suite, the heavy ECAPE family (``--heavy``), and multi-hour windowed
  accumulations -- and everything a file's stored fields prove out
  renders (``--list-products`` shows the per-file verdict and every
  reason).
  The ONLY engine ``--engine auto`` will select, and the only one any
  in-process caller or chain reaches.
* ``matplotlib`` -- the wrf-rust + matplotlib path below, reachable
  only by typing ``--engine matplotlib``, and announcing itself as a
  ``WORKAROUND:`` line on every run.

The project render law ( 2026-08-06) reserves weather-field
product plots for ``rw_wrfbatch``.  The one fallback it named, the DA
nowcast's multi-panel sheet tool, served none of this door's products and
was removed with the local data assimilation in 3.0.3.  ``--engine auto``
therefore does not degrade: with no usable
renderer it REFUSES, naming the missing artifact and ``woof
fetch-bridges``, at a nonzero exit.  It used to draw five weather
fields with matplotlib and report success, which is how a box with a
failed bridge staging produced pictures that looked like the
production catalog and were not it.

``--pair A_DIR B_DIR`` composes two runs' rendered PNGs into labeled
side-by-side comparison sheets (:mod:`woof.pair_compose`) -- either
engine's output pairs.

``--streamlines`` / ``--barbs`` choose how the wind is drawn on every
product carrying a wind layer.  Neither given, the engine keeps its
automatic per-grid choice (streamlines on curvilinear and projected
grids, barbs on plain lat/lon) and the invocation is byte-identical to
every earlier release.  ``RUSTWX_WIND_STREAMLINES`` still works and
these two outrank it: a flag a stale shell export could silently
overrule would be a flag that lies.

``--products var:<stored name>`` draws any 2-D field the wrfout carries,
INCLUDING one added to a user's own WRF Registry -- ``MSLP_ANOM`` is
imported as ``wrf_mslp_anom`` and rendered by ``--products
var:wrf_mslp_anom`` with no product definition written anywhere.
``--list-products`` names every ``var:`` row a given file can serve.

Both engines file what they draw through :mod:`woof.render_layout`:
``<--out>/<domain>/<product>/<valid-day>/<file>.png``, by default and
without a flag.  Every picture used to land in one directory, which on
a three-nest run of the rust catalog is five figures of files of every
product and every valid time with nothing but filenames to sort them.
``--layout flat`` restores the v2.4.1 single directory for a consumer
that still needs it.  See ``docs/render-output-layout.md``.

Every output filename carries a domain + resolution token
(:func:`domain_token`: ``d02-3km``, sub-kilometre nests as ``d05-111m``)
read from the file's own ``GRID_ID`` and ``DX``.  Without it two nests
of one run share every other filename component and the second render
at a lead silently overwrote the first -- a forecast lost with no error
and an exit code of 0.  The same spacing appears in the plot subtitle
as ``Δx 3 km``, and plots are labelled with the model that produced
them (``--source-label``, default :func:`default_source_label`) rather
than the GDEX fetch source the rust engine's ``wrf`` store identity
inherits.  That default carries the version of the code that is
EXECUTING (``WOOF 1.8.7``), which is not the same number as
``woof.__version__``: see :func:`default_source_label`.

Every derived quantity comes from the mandated ``wrf`` package (pip
distribution ``wrf-rust``): destaggering, earth-rotation, and unit
conversion are ``wrf.getvar`` calls, never local formulas.  When a file
carries no earth-rotation fields (idealized/minimal wrfouts without
SINALPHA/COSALPHA), the 10 m wind panel falls back to grid-relative raw
``U10``/``V10`` in the model's native m/s and labels both the barbs and
the colorbar accordingly -- an accurate degradation, not a hand-rolled
conversion.  The one composition performed here is WRF's own
accumulation-bucket total (``RAINC + RAINNC``), which is bookkeeping
over Registry accumulators rather than a diagnostic.  Composite reflectivity is the column maximum
of the model-native ``REFL_10CM`` -- a direct field product, matching
``tools/matched_wrfout_refl_figures.py`` -- not a re-derived simulated
reflectivity.

Rendering conventions follow that tool: Agg backend, the standard NWS
5-dBZ reflectivity scale, recessive axes, per-panel colorbars labeled
with units, and valid-time titles.  Geographic extent is the file's own
XLAT/XLONG curvilinear coordinates (``wrf.latlon_coords``), so terrain
and projection shape the frame instead of bare array indices.

The module imports ``wrf`` and matplotlib lazily: ``woof --help`` and
every non-render command work without the render extra installed, and
the failure names the missing dependency and the install command.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from woof import explain, render_georef, render_layout, run_stamp
from woof.cli_numbers import positive_int
from woof.io.history_selection import PRODUCT_HISTORY_INPUTS
from woof.science_core import SCIENCE_CORE_REQUIREMENT

#: The pip requirement the render products are certified against, quoted
#: verbatim in the install hint below.  Sourced from woof.science_core so
#: the hint and the pyproject extra cannot drift apart.
WRF_PACKAGE_REQUIREMENT = SCIENCE_CORE_REQUIREMENT

#: Standard NWS 5-dBZ-step reflectivity scale (5..75), as in
#: tools/matched_wrfout_refl_figures.py (domain convention).
_NWS_COLORS = (
    "#04e9e7", "#019ff4", "#0300f4", "#02fd02", "#01c501", "#008e00",
    "#fdf802", "#e5bc00", "#fd9500", "#fd0000", "#d40000", "#bc0000",
    "#f800fd", "#9854c6",
)
_NWS_LEVELS = tuple(np.arange(5.0, 76.0, 5.0))

#: Accumulated-precipitation bounds (mm) on an NWS-style QPF ladder.
_PRECIP_LEVELS = (0.1, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0,
                  35.0, 50.0, 75.0, 100.0, 150.0)

_WRFOUT_DOMAIN = re.compile(r"wrfout_(d\d{2})")

#: Provenance label for locally-imported runs.  The rust engine inherits
#: a GDEX fetch source from its ``wrf`` store-model identity, which this
#: lane never fetched from; ``--source-label`` renames it for stock-WRF
#: files, which ArWen did not produce and must not claim.
#:
#: The BRAND half only.  What actually reaches a plot is
#: :func:`default_source_label`, which appends the executing version.
#: (WOOF 1.1.7) "WOOF-ArWen", the WRF engine's product name (was "WOOF"): a map says which of the two engines
#: drew it ("WOOF-ArWen 1.1.7"; the MPAS engine's say "WOOF-HEX-A 1.1.7", woof.hex.render_door).
DEFAULT_SOURCE_LABEL = "WOOF-ArWen"



#: netCDF4 over HDF5 is not thread safe, and netCDF4 1.7 releases the GIL
#: around its library calls: two threads rendering in one process (two
#: launches filed at once, as tests/test_render_georef_record.py races
#: them) crashed the interpreter with an access violation inside a
#: header read about one run in thirteen.  Every history read here holds
#: this lock for its whole open.
_NETCDF_LOCK = threading.RLock()


@contextlib.contextmanager
def _netcdf_read(path):
    import netCDF4

    with _NETCDF_LOCK:
        with netCDF4.Dataset(path) as dataset:
            yield dataset

def default_source_label() -> str:
    """``WOOF 1.8.7`` -- the brand plus the version that is EXECUTING.

    THE product stamp, and the reason it is a function rather than a
    constant.  Both engines put this string on the plot verbatim: the
    matplotlib engine through :func:`plot_context`, and the rust engine
    through ``--source-label``, which ``rw_wrfbatch`` renders as its
    ``subtitle_right`` without interpreting it.  So whatever this
    returns is literally the label a reader sees on the image, and it
    is the only place on a PNG where the producing build can be named
    at all.

    The number comes from :func:`woof.provenance_gate.executing_version`
    and NOT from ``woof.__version__``, because those two are not the
    same claim.  ``__version__`` asks distribution metadata for the
    version of the distribution NAMED woof, which on a box with a
    stale editable install answers for a tree that is not the one
    running -- exactly the field report this program was opened for,
    where plots were labelled 1.6.2 and the reader believed they had
    installed 1.8, and nothing in the product could say which of them
    was right.  ``executing_version`` prefers the running code's own
    declaration, so the label describes the code that drew the image.

    An install that genuinely cannot name its version contributes
    nothing rather than the ``0+unknown`` sentinel: a plot is not a
    diagnostics channel, and a bare ``WOOF`` is the accurate label there.
    ``woof version`` and the render receipt carry the full story.
    """

    from woof.provenance import UNKNOWN_VERSION
    from woof.provenance_gate import executing_version

    try:
        version = executing_version()
    except Exception:                                   # noqa: BLE001
        return DEFAULT_SOURCE_LABEL
    if not version or version == UNKNOWN_VERSION:
        return DEFAULT_SOURCE_LABEL
    return f"{DEFAULT_SOURCE_LABEL} {version}"


def _import_wrf():
    try:
        import wrf
    except ImportError as exc:
        raise RuntimeError(
            "woof render needs the mandated 'wrf' package (pip "
            f"distribution {WRF_PACKAGE_REQUIREMENT}); install it with "
            "pip install 'recast-woof[render]' or pip install "
            f"'{WRF_PACKAGE_REQUIREMENT}'") from exc
    return wrf


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


#: The slug an output carries when the file's domain identity cannot be
#: established.  Spelled exactly as the rust engine spells it
#: (``rw-wrfbatch``'s ``native_grid``): a reader comparing two engines'
#: outputs should not have to learn two words for one fact.
NATIVE_GRID_SLUG = "native_grid"


def _domain_tag(path: Path) -> str | None:
    """``d0X`` when the file PROVES its domain, else ``None``.

    Deliberately the same rule ``rw-wrfbatch::file_grid_identity`` uses,
    in the same precedence: the ``GRID_ID`` global attribute when it is a
    plausible domain number, otherwise a ``wrfout_dNN`` filename,
    otherwise nothing.  Two engines that degrade differently are two
    behaviours to document and one of them will be wrong.

    ``None`` is the whole point of the change.  This used to return
    ``"d01"`` for a file carrying neither piece of evidence, which
    labelled an anonymous input as the parent domain on no evidence at
    all -- and, because the token is what separates one nest's output
    filenames from another's, let two anonymous inputs at one valid time
    overwrite each other in exactly the silence this release exists to
    end.
    """

    try:
        with _netcdf_read(path) as ds:
            if "GRID_ID" in ds.ncattrs():
                grid_id = int(ds.getncattr("GRID_ID"))
                # The rust side accepts 1..=99; anything else is not a
                # domain number and is treated as no evidence.
                if 1 <= grid_id <= 99:
                    return f"d{grid_id:02d}"
    except Exception:
        pass
    match = _WRFOUT_DOMAIN.search(path.name)
    return match.group(1) if match else None


def history_episode(path) -> int | None:
    """The lifecycle episode one history file's own folder declares.

    ``None`` when it declares none, which is every run that does not
    declare ``retire``/``rearm`` and therefore every run shipped so far.

    The evidence is the folder the HISTORY WRITER filed the frame in.
    A nest that retires and re-arms puts a second history run through
    one slot, and ``woof.io.wrfout`` separates those as
    ``d05/episode-002/`` using
    :func:`woof.core.nest_lifecycle.output_episode`'s number -- so the
    number read back here is the lifecycle's own, not a second count
    kept by the renderer.  Read from the IMMEDIATE parent, because that
    is exactly where the writer puts it: a deeper search would let a
    case folder somebody named ``episode-002`` relabel every frame of
    every nest under it.

    The spelling is :func:`woof.render_layout.episode_number`'s, the
    inverse of the one the writer used, so the two trees cannot drift
    into disagreeing about which episode a picture belongs to.
    """

    return render_layout.episode_number(Path(path).parent.name)


def _grid_spacing_m(path: Path) -> float | None:
    """The file's own ``DX`` in metres, or None when it declares none."""

    try:
        with _netcdf_read(path) as ds:
            if "DX" not in ds.ncattrs():
                return None
            spacing = float(ds.getncattr("DX"))
    except Exception:
        return None
    if not np.isfinite(spacing) or spacing <= 0.0:
        return None
    return spacing


def _spacing_parts(spacing_m: float | None) -> tuple[str, str] | None:
    """``(value, unit)`` for a grid spacing: ``('3', 'km')``/``('333', 'm')``.

    At and above a kilometre the number is trimmed kilometres (a 3:1 nest
    of a 12 km parent is ``1.333 km``); below it, integer metres, because
    ``0.333 km`` is a worse label for a 333 m nest than ``333 m`` is.
    """

    if spacing_m is None:
        return None
    metres = round(float(spacing_m))
    if metres >= 1_000:
        value = f"{spacing_m / 1_000.0:.3f}".rstrip("0").rstrip(".")
        return (value or "0"), "km"
    return f"{metres:d}", "m"


def resolution_token(spacing_m: float | None) -> str | None:
    """``3km``/``333m`` -- the resolution half of the filename token."""

    parts = _spacing_parts(spacing_m)
    return None if parts is None else f"{parts[0]}{parts[1]}"


def spacing_label(spacing_m: float | None) -> str | None:
    """``Δx 3 km`` -- the same number, spelled for the plot title."""

    parts = _spacing_parts(spacing_m)
    return None if parts is None else f"Δx {parts[0]} {parts[1]}"


def domain_token(domain: str | None, spacing_m: float | None) -> str:
    """``d02-3km`` -- the filename token that separates a run's nests.

    Two nests of one run share every other filename component (product,
    valid time, and on the rust engine model/cycle/lead as well), so
    without this the second domain rendered at a lead silently
    overwrites the first.

    The three degradation steps are ``rw-wrfbatch::native_domain_slug``'s,
    exactly: identity plus a usable ``DX`` gives ``d02-3km``; identity
    without one gives a bare ``d02``, because the domain alone still
    separates the nests, which is the token's whole job; and no identity
    gives :data:`NATIVE_GRID_SLUG` with no resolution appended -- a
    resolution is not an identity, and pretending otherwise is how
    ``d01`` got printed on files that never claimed to be d01.
    """

    if domain is None:
        return NATIVE_GRID_SLUG
    resolution = resolution_token(spacing_m)
    return domain if resolution is None else f"{domain}-{resolution}"


def plot_context(domain: str | None, spacing_m: float | None, stamp: str,
                 source_label: str) -> str:
    """The second title line every matplotlib product carries.

    Mirrors the rust engine's subtitle: grid identity, the spacing the
    file declares, the valid time, and who produced the forecast.
    """

    segments = [domain_token(domain, spacing_m)]
    spacing = spacing_label(spacing_m)
    if spacing is not None:
        segments.append(spacing)
    segments.append(f"valid {stamp}")
    segments.append(source_label)
    return " | ".join(segments)


def _figure(plt, lat, lon):
    """One recessive-axes lat/lon panel sized to the domain aspect."""
    ny, nx = np.asarray(lat).shape
    width = 8.0
    height = max(4.0, width * ny / max(nx, 1) * 0.9)
    fig, axis = plt.subplots(figsize=(width, height))
    axis.set_xlabel("longitude (degrees east)", fontsize=9, color="#444444")
    axis.set_ylabel("latitude (degrees north)", fontsize=9, color="#444444")
    axis.tick_params(labelsize=8, colors="#444444")
    for spine in axis.spines.values():
        spine.set_color("#999999")
    return fig, axis


def _finish(fig, axis, mappable, *, title: str, cbar_label: str,
            out_png: Path, dpi: int, ticks=None) -> None:
    cbar = fig.colorbar(mappable, ax=axis, shrink=0.9, pad=0.02,
                        ticks=ticks)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    axis.set_title(title, fontsize=11)
    fig.tight_layout()
    # Same ceiling as the rust engine's placement seam, same remedy: the
    # layout's three folders push the longest product names past
    # Windows' MAX_PATH from any real case root, and a savefig that
    # cannot open its target loses the picture outright here.
    spelled = render_layout.fs_path(out_png)
    Path(spelled).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(spelled, dpi=dpi)


def _render_refl(wf, timeidx, lat, lon, *, plt, wrf, context,
                 out_png: Path, dpi: int) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    refl = np.asarray(wrf.getvar(wf, "REFL_10CM", timeidx=timeidx))
    composite = refl.max(axis=0)
    cmap = ListedColormap(_NWS_COLORS)
    cmap.set_under("none")
    cmap.set_over(_NWS_COLORS[-1])
    norm = BoundaryNorm(_NWS_LEVELS, cmap.N)
    fig, axis = _figure(plt, lat, lon)
    mesh = axis.pcolormesh(lon, lat, composite, cmap=cmap, norm=norm,
                           shading="auto")
    _finish(fig, axis, mesh,
            title=f"composite reflectivity (column-max REFL_10CM)\n"
                  f"{context}",
            cbar_label="composite dBZ", out_png=out_png, dpi=dpi,
            ticks=_NWS_LEVELS[::2])
    plt.close(fig)


def _render_t2(wf, timeidx, lat, lon, *, plt, wrf, context,
               out_png: Path, dpi: int) -> None:
    t2 = np.asarray(wrf.getvar(wf, "t2", timeidx=timeidx, units="degC"))
    fig, axis = _figure(plt, lat, lon)
    # Spectral temperature scale (meteorological convention), symmetric
    # ticks left to matplotlib; the data range sets the limits.
    mesh = axis.pcolormesh(lon, lat, t2, cmap="RdYlBu_r", shading="auto")
    _finish(fig, axis, mesh,
            title=f"2 m temperature\n{context}",
            cbar_label="deg C", out_png=out_png, dpi=dpi)
    plt.close(fig)


def _render_wind10(wf, timeidx, lat, lon, *, plt, wrf, context,
                   out_png: Path, dpi: int) -> None:
    try:
        # Earth-rotated components (needs SINALPHA/COSALPHA in the file);
        # knots come from wrf.getvar's own unit handling.
        uv = np.asarray(wrf.getvar(wf, "uvmet10", timeidx=timeidx,
                                   units="kt"))
        u_barb, v_barb = uv[0], uv[1]
        speed = np.asarray(wrf.getvar(wf, "wspd10", timeidx=timeidx,
                                      units="kt"))
        rotation, speed_units = "earth-rotated", "kt"
    except Exception:
        # Idealized/minimal files carry no rotation fields; the fallback
        # is grid-relative raw U10/V10 in the model's native m/s, labeled
        # as such -- unit conversion is wrf.getvar's job, never a local
        # formula, so the whole panel degrades to m/s and says so rather
        # than converting by hand.
        u_barb = np.asarray(wrf.getvar(wf, "U10", timeidx=timeidx))
        v_barb = np.asarray(wrf.getvar(wf, "V10", timeidx=timeidx))
        speed = np.asarray(wrf.getvar(wf, "wspd10", timeidx=timeidx))
        rotation, speed_units = "grid-relative", "m s-1"
    fig, axis = _figure(plt, lat, lon)
    mesh = axis.pcolormesh(lon, lat, speed, cmap="viridis", shading="auto")
    ny, nx = speed.shape
    # ~18 barbs along the long axis keeps flags legible at any nest size.
    step = max(1, int(np.ceil(max(ny, nx) / 18)))
    sub = (slice(step // 2, None, step), slice(step // 2, None, step))
    axis.barbs(np.asarray(lon)[sub], np.asarray(lat)[sub],
               u_barb[sub], v_barb[sub], length=5.5,
               linewidth=0.6, color="#222222")
    _finish(fig, axis, mesh,
            title=f"10 m wind ({rotation} barbs)\n{context}",
            cbar_label=f"10 m wind speed ({speed_units})",
            out_png=out_png, dpi=dpi)
    plt.close(fig)


def _render_precip(wf, timeidx, lat, lon, *, plt, wrf, context,
                   out_png: Path, dpi: int) -> None:
    from matplotlib.colors import BoundaryNorm

    buckets = []
    for name in ("RAINC", "RAINNC"):
        try:
            buckets.append(
                np.asarray(wrf.getvar(wf, name, timeidx=timeidx)))
        except Exception:
            continue
    if not buckets:
        raise RuntimeError(
            "neither RAINC nor RAINNC is present; there is no "
            "accumulated-precipitation field to render")
    total = buckets[0]
    for bucket in buckets[1:]:
        total = total + bucket
    import matplotlib

    fig, axis = _figure(plt, lat, lon)
    cmap = matplotlib.colormaps["YlGnBu"].resampled(
        len(_PRECIP_LEVELS) - 1)
    cmap.set_under("none")
    norm = BoundaryNorm(_PRECIP_LEVELS, cmap.N)
    mesh = axis.pcolormesh(lon, lat, total, cmap=cmap, norm=norm,
                           shading="auto")
    _finish(fig, axis, mesh,
            title=f"accumulated precipitation (RAINC + RAINNC)\n"
                  f"{context}",
            cbar_label="mm since simulation start", out_png=out_png,
            dpi=dpi, ticks=_PRECIP_LEVELS)
    plt.close(fig)


def _render_olr(wf, timeidx, lat, lon, *, plt, wrf, context,
                out_png: Path, dpi: int) -> None:
    """Top-of-atmosphere outgoing longwave, drawn as synthetic satellite IR.

    ``OLR`` is what the longwave scheme was already computing and wrfout
    frames have carried since 1.5.1, so this is the one product here whose
    subject is the model as a satellite would see it.

    Inverted grayscale is what makes it read that way: an IR image is dark
    where the emitting surface is warm and bright where it is cold, and OLR
    runs the same direction -- deep cold cloud tops emit least, the clear
    warm surface emits most -- so ``gray_r`` puts storm tops white against a
    dark background exactly as the imagery convention does.

    The panel stays in W m-2 rather than converting to a brightness
    temperature.  That conversion is a derived quantity and derived
    quantities are the ``wrf`` package's job, never a formula written at a
    plotting site (the same rule that keeps ``_render_wind10`` degrading to
    m s-1 instead of converting knots by hand).

    Limits come from the frame, as they do for ``t2`` and ``wind10``: there
    is no standard OLR ladder to hard-code, and inventing one here would put
    unsourced numbers between the reader and the field.  So shade is
    comparable within a panel, not between panels -- read the colorbar.
    """

    olr = np.asarray(wrf.getvar(wf, "OLR", timeidx=timeidx))
    fig, axis = _figure(plt, lat, lon)
    mesh = axis.pcolormesh(lon, lat, olr, cmap="gray_r", shading="auto")
    _finish(fig, axis, mesh,
            title=f"outgoing longwave radiation at TOA "
                  f"(synthetic IR)\n{context}",
            cbar_label="OLR (W m-2)", out_png=out_png, dpi=dpi)
    plt.close(fig)


#: Product registry: CLI name -> renderer.  ``all`` expands to this order.
PRODUCTS = {
    "refl": _render_refl,
    "t2": _render_t2,
    "wind10": _render_wind10,
    "precip": _render_precip,
    "olr": _render_olr,
}

#: The shared product names, as the rust engine's catalog slugs.
#:
#: All five of :data:`PRODUCTS`.  ``olr`` maps to ``var:wrf_olr``, the
#: engine's own chart of the history's stored ``OLR`` plane (the same
#: W m-2 flux the matplotlib panel draws), and NOT to the catalog's
#: ``simulated_ir_satellite``: that is a brightness temperature from a
#: radiative-transfer operator, a different quantity that the history
#: lane does not serve, and pointing ``olr`` at it would be the
#: ``wind10`` bug below again.  Before this entry ``woof render --help``
#: listed ``olr`` and the engine refused it as an unknown slug (2.8.7 and
#: 2.8.8 candidates alike).
#:
#: Each value is the slug of a chart whose SUBJECT is what the key
#: names.  ``wind10`` used to map to ``mslp_10m_winds`` under a comment
#: calling that "the catalog's standalone surface-wind product"; it is
#: not.  ``mslp_10m_winds`` is a mean-sea-level PRESSURE analysis with
#: 10 m barbs drawn over it, so ``--products wind10 --engine rust``
#: returned a pressure chart to anyone who asked for wind, while the
#: matplotlib engine's own ``wind10`` drew a genuine wind map.  During a
#: wildfire that asymmetry is what pushed an operator onto the fallback
#: engine to get a wind map at all.  The rust catalog now carries a real
#: standalone wind chart and this points at it.
#:
#: Any other token is passed through as a raw catalog slug, which the
#: renderer validates strictly (see :func:`parse_products_rust`).
RUST_PRODUCT_ALIASES = {
    # column-max simulated reflectivity
    "refl": "composite_reflectivity",
    # 2 m temperature fill
    "t2": "2m_temperature",
    # 10 m wind speed fill + 10 m barbs, one frame, nothing else on it
    "wind10": "10m_wind_speed_and_direction",
    # run-total accumulated precipitation
    "precip": "total_qpf",
    # top-of-atmosphere outgoing longwave, the stored OLR plane in W m-2
    "olr": "var:wrf_olr",
}

#: What each alias must actually DRAW, as the canonical store selector
#: its fill resolves to.  This is the assertion a string-equality test
#: could not make: the ``wind10`` bug was a correct-looking mapping to a
#: real slug that filled with the wrong quantity, and any test comparing
#: ``RUST_PRODUCT_ALIASES["wind10"]`` to a hardcoded string would have
#: agreed with the broken code.  ``tests/test_render_rust.py`` checks
#: these against the renderer's own catalog, from the built binary.
RUST_PRODUCT_ALIAS_FILL_SELECTORS = {
    "refl": "composite_reflectivity_entire_atmosphere",
    "t2": "temperature_2m_agl",
    "wind10": "wind_speed_10m_agl",
    "precip": "total_precipitation_surface",
}


def parse_products_rust(spec: str) -> str:
    """CLI product spec -> the rust renderer's ``--products`` value.

    ``all`` expands to the renderer's own inspected catalog (its "all"),
    the four shared names map through :data:`RUST_PRODUCT_ALIASES`, and
    unknown tokens pass through as catalog slugs for the renderer's own
    strict validation -- so ``--products sbcape,srh_0_1km`` works without
    this module re-declaring the rust catalog.

    The list is read with the engine's own tokenizer
    (:func:`woof.rustwx.product_spec_terms`), so a section's level list
    stays inside its section and duplicates are whole products, never
    level numbers.  ``all`` stands for every named product the frames
    can draw, so a named product beside it adds nothing and is folded
    into it (the engine refuses a keyword beside a named product in one
    list).  What ``all`` does not draw keeps its place beside it: the
    ``variables`` keyword and the storeless families (``xsec:``,
    ``mesh:``), which the engine splits off before the store list and
    the render door answers per term.
    """

    from woof.rustwx import (MESH_PREFIXES, SECTION_PREFIX,
                              VARIABLES_KEYWORD, product_spec_terms)

    slugs: list[str] = []
    for token in product_spec_terms(spec):
        slug = RUST_PRODUCT_ALIASES.get(token, token)
        if slug.lower() == "all":
            slug = "all"
        if slug not in slugs:
            slugs.append(slug)
    if not slugs:
        raise ValueError("no products requested")
    if "all" in slugs:
        slugs = [slug for slug in slugs
                 if slug == "all" or slug.startswith(SECTION_PREFIX)
                 or slug.lower().startswith(MESH_PREFIXES)
                 or slug.lower() == VARIABLES_KEYWORD]
    return ",".join(slugs)


def parse_size(spec: str) -> tuple[int, int] | None:
    """``1200x900`` -> (width, height); rust-engine output pixels.

    ``auto`` -> ``None``: the engine sizes each canvas from its domain's
    own shape, so a square nest is not drawn on a landscape canvas.
    """

    if spec.strip().lower() == "auto":
        return None
    parts = spec.lower().split("x")
    if len(parts) != 2:
        raise ValueError(
            f"--size must be WIDTHxHEIGHT (e.g. 1200x900), got {spec!r}")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(
            f"--size must be WIDTHxHEIGHT (e.g. 1200x900), got {spec!r}"
        ) from None
    if width < 320 or height < 240:
        raise ValueError("--size must be at least 320x240")
    return width, height


def parse_products(spec: str) -> tuple[str, ...]:
    """``refl,t2`` -> product tuple; ``all`` -> every product, in order.

    Read with the engine's own tokenizer
    (:func:`woof.rustwx.product_spec_terms`), so a section term this
    engine cannot draw is refused whole rather than by its level list's
    pieces.
    """
    from woof.rustwx import product_spec_terms

    names: list[str] = []
    for token in product_spec_terms(spec):
        if token == "all":
            names.extend(name for name in PRODUCTS if name not in names)
            continue
        if token not in PRODUCTS:
            raise ValueError(
                f"unknown product {token!r}; choose from "
                f"{', '.join(PRODUCTS)} or 'all'")
        if token not in names:
            names.append(token)
    if not names:
        raise ValueError("no products requested")
    return tuple(names)


def parse_timeidx(spec: str) -> int | None:
    """``N`` -> that frame, ``all`` -> None (every frame in the file)."""
    if spec == "all":
        return None
    try:
        index = int(spec)
    except ValueError:
        raise ValueError(
            f"--timeidx must be an integer or 'all', got {spec!r}"
        ) from None
    if index < 0:
        raise ValueError("--timeidx must be non-negative")
    return index


def _stamp_for_filename(stamp: str) -> str:
    """Valid-time stamp with Windows-illegal colons replaced."""
    return stamp.replace(":", "-")


def render_wrfouts(paths, *, products: tuple[str, ...],
                   timeidx: int | None, outdir: Path,
                   dpi: int = 150,
                   source_label: str | None = None,
                   layout: str = render_layout.DEFAULT_LAYOUT,
                   ) -> tuple[list[Path], list[str],
                              list[tuple[str, str]]]:
    """Render every requested product/frame.

    Returns ``(written, failures, skipped)``.  The third list is the one
    this function used to fold into the second, and folding them was a
    wrong answer with an exit code attached: a product whose *declared*
    inputs (:data:`_MPL_PRODUCT_NEEDS`) are simply not in the file has
    not failed at anything.  The cold-start history frame is the standing
    example -- no microphysics call precedes it, so it carries no
    ``REFL_10CM``, a registered deviation -- and reflectivity coming back
    empty from it used to fail the render, and with it the whole ``woof
    go`` chain, on a forecast that had completed.

    The distinction is drawn from the table, not from the exception, so
    it holds for every product and stays true when the catalog grows:

    * declared inputs absent  -> **skipped**, with the names of what is
      absent; the caller reports it and the exit code does not move.
    * declared inputs present -> rendered, or **failed** if it raised.
      That is a real failure and still is one.

    Nothing is silent either way: both lists come back for the CLI to
    print.
    """
    # Resolved here rather than in the signature: a default argument is
    # evaluated at import, and this one resolves provenance (a git
    # subprocess).  `woof --help` must not pay for it.
    if source_label is None:
        source_label = default_source_label()
    wrf = _import_wrf()
    plt = _pyplot()
    written: list[Path] = []
    failures: list[str] = []
    skipped: list[tuple[str, str]] = []
    #: output path -> the input that claimed it, this invocation.
    claims: dict[Path, Path] = {}
    for path in (Path(p) for p in paths):
        try:
            wrffile = wrf.WrfFile(str(path))
            stamps = list(wrffile.times())
        except Exception as exc:
            failures.append(f"{path}: unreadable wrfout ({exc})")
            continue
        present = _file_variables(path)
        domain = _domain_tag(path)
        spacing_m = _grid_spacing_m(path)
        token = domain_token(domain, spacing_m)
        # Which LIFE of this nest, when it has more than one.  Both
        # engines file the same way, so a run that renders half its
        # frames through each does not end up with two trees.
        episode = history_episode(path)
        if timeidx is None:
            indices = range(len(stamps))
        elif timeidx >= len(stamps):
            failures.append(
                f"{path}: --timeidx {timeidx} out of range; file has "
                f"{len(stamps)} frame(s)")
            continue
        else:
            indices = (timeidx,)
        for index in indices:
            stamp = stamps[index]
            try:
                lat, lon = wrf.latlon_coords(wrffile, timeidx=index)
            except Exception as exc:
                failures.append(
                    f"{path}[{index}]: no XLAT/XLONG coordinates ({exc})")
                break
            # Antimeridian-crossing domains: XLONG jumps between +180
            # and -180 mid-array, which pcolormesh renders as a smear
            # across the full axis.  Unwrap onto the branch nearest the
            # domain centre (the axis may then extend past +/-180,
            # which matplotlib labels correctly).
            lon_values = np.asarray(lon)
            if lon_values.size and (lon_values.max()
                                    - lon_values.min()) > 180.0:
                center = float(lon_values[tuple(
                    s // 2 for s in lon_values.shape)])
                lon = center + ((lon - center + 180.0) % 360.0 - 180.0)
            context = plot_context(domain, spacing_m, stamp, source_label)
            for product in products:
                absent = ([] if present is None
                          else missing_declared_inputs(present, product))
                if absent:
                    skipped.append((product, f"{path}[{index}] carries no "
                                             f"{', '.join(absent)}"))
                    continue
                # WHERE the picture goes is one decision, made in one
                # place, for both engines: woof.render_layout.  The
                # filename is unchanged from v2.4.1 -- only the
                # directory beneath `--out` is new -- so a reader who
                # knows the old name still recognises the file.
                out_png = render_layout.place(
                    outdir, domain=token, product=product,
                    day=render_layout.valid_day(stamp), layout=layout,
                    episode=episode,
                    filename=(f"{product}_{token}_"
                              f"{_stamp_for_filename(stamp)}.png"))
                # The token separates nests, but it cannot separate two
                # inputs it could not identify: both are `native_grid`.
                # Refusing the second is the same call this release made
                # for named nests -- an overwrite that reports success
                # is the failure, not the collision.
                claimed = claims.get(out_png)
                if claimed is not None and claimed != path:
                    failures.append(
                        f"{path}[{index}] {product}: would overwrite "
                        f"{out_png.name}, already rendered from "
                        f"{claimed}.  Both inputs resolve to the same "
                        f"output name because neither declares a "
                        f"distinguishable domain identity (GRID_ID or a "
                        f"wrfout_dNN filename).  Render them into "
                        f"separate --out directories, or give the files "
                        f"their GRID_ID.")
                    continue
                claims[out_png] = path
                try:
                    PRODUCTS[product](
                        wrffile, index, lat, lon, plt=plt, wrf=wrf,
                        context=context, out_png=out_png, dpi=dpi)
                except Exception as exc:
                    failures.append(f"{path}[{index}] {product}: {exc}")
                    continue
                written.append(out_png)
                print(f"render: {out_png}")
    return written, failures, skipped


#: The matplotlib engine's per-product wrfout variable needs, READ OFF
#: woof's own render product catalog rather than declared again here.
#:
#: Three readers now: the --list-products availability report, the render
#: loop's own pre-check (:func:`missing_declared_inputs`), and the
#: ``[output]`` history-selection plan-time dependency warning
#: (:func:`woof.io.history_selection.lost_products`).  They read ONE
#: table, so the status a listing prints, the decision a render makes,
#: and the products a config is warned it is about to lose cannot
#: disagree about which variable a product needs.
#:
#: This view is restricted to :data:`PRODUCTS` because it is the
#: matplotlib engine's own catalog; the shared table additionally
#: carries the rust catalog slugs those names alias onto, which this
#: engine cannot draw.
_MPL_PRODUCT_NEEDS = {
    name: PRODUCT_HISTORY_INPUTS[name] for name in PRODUCTS}


def missing_declared_inputs(present, product: str) -> list[str]:
    """This product's declared inputs that ``present`` does not carry.

    ``present`` is the file's variable-name set.  Each entry of
    :data:`_MPL_PRODUCT_NEEDS` is one requirement, and ``A|B`` means
    either satisfies it, so what comes back reads as the requirement did.

    Empty means every declared input is there -- which is the whole
    point of asking: a product that then fails to render failed for a
    reason this table does not know about, and that is a real failure.
    """

    missing = []
    for need in _MPL_PRODUCT_NEEDS.get(product, ()):
        options = need.split("|")
        if not any(option in present for option in options):
            missing.append(" or ".join(options))
    return missing


#: The global attribute a trimmed run stamps into its own wrfout, naming
#: the variables it deliberately did not write
#: (:meth:`woof.io.history_selection.HistorySelection.wrfout_attrs`).
#: A full-default run stamps nothing, so its absence means "nothing was
#: dropped" and never "this file is old".
HISTORY_DROPPED_ATTR = "GPUWM_HISTORY_DROPPED"


def history_dropped_variables(path) -> frozenset[str]:
    """The variables this wrfout's own run chose not to write.

    Empty when the file does not say -- a full-default run, or any
    artifact written before ``[output]`` existed.  Empty is also the
    answer for a file that cannot be opened at all: this is a
    DIAGNOSTIC, and it must never be the reason a render dies.  A file
    that is really unreadable fails loudly a moment later, in the render
    itself, with a message about the file rather than about a table.
    """

    try:
        with _netcdf_read(Path(path)) as dataset:
            raw = getattr(dataset, HISTORY_DROPPED_ATTR, "")
    except Exception:
        return frozenset()
    return frozenset(
        name for name in str(raw).split(",") if name.strip()) or frozenset()


def history_killed_products(paths, product_tokens) -> dict[str, str]:
    """Requested products whose inputs THIS RUN dropped -> why, by name.

    The mapping comes from woof's own render product catalog
    (:data:`woof.io.history_selection.PRODUCT_HISTORY_INPUTS`), which is
    also what the ``[output]`` plan-time warning reads -- so a product
    the config warned about is the same product named here.

    Two deliberate abstentions.  ``all`` asks for whatever the file
    supports and is not a request for a specific product, so it is never
    reported.  A token the table does not carry is not claimed: the
    renderer owns 151 products and answers for their inputs itself
    through ``--list-products``; inventing a dependency here for a recipe
    this table has not verified is the correct-looking-wrong-mapping
    defect ``wind10`` already cost once.
    """

    from woof.io.history_selection import PRODUCT_HISTORY_INPUTS

    dropped: set[str] = set()
    for path in paths:
        dropped |= history_dropped_variables(path)
    if not dropped:
        return {}
    killed: dict[str, str] = {}
    for token in product_tokens:
        token = str(token).strip()
        if token == "all":
            continue
        requirements = PRODUCT_HISTORY_INPUTS.get(token)
        if not requirements:
            continue
        lost = [" or ".join(requirement.split("|"))
                for requirement in requirements
                if all(option in dropped
                       for option in requirement.split("|"))]
        if lost:
            killed[token] = (
                f"{token} is drawn from {', '.join(lost)}, which this "
                "run's [output] history_drop / preset did not write")
    return killed


def _history_selection_action(killed: dict[str, str], *, everything: bool
                              ) -> str:
    lines = "\n".join(f"  {detail}." for detail in killed.values())
    head = ("woof render: every product you asked for needs a variable "
            "this run did not write."
            if everything else
            "note: render cannot draw "
            f"{', '.join(sorted(killed))} from these file(s) -- the "
            "variable(s) they need were not written.")
    return (
        f"{head}\n{lines}\n"
        "  remedy: re-run without dropping the variable(s) -- delete the "
        "name from [output] history_drop, or widen the preset (severe "
        "keeps the storm-scale volumes, full keeps everything).\n"
        "  # to see what THIS file can still draw:\n"
        "  #   woof render --list-products FILE")


_HISTORY_SELECTION_WHY = (
    "The file says so itself: a run that trimmed its history stamps "
    f"{HISTORY_DROPPED_ATTR} into every wrfout it writes, naming the "
    "variables it chose not to write.  Without that stamp the only "
    "available answer would be 'not in file', which reads like a "
    "damaged artifact rather than a configuration choice made before "
    "the run -- and the remedy for the two is not the same.\n\n"
    "The product-to-variable mapping is woof's own render product "
    "catalog (woof.io.history_selection.PRODUCT_HISTORY_INPUTS), the "
    "same table the [output] plan-time warning reads at config "
    "resolution, so the product named here is the product that warning "
    "named hours earlier.")


def history_selection_refusal(paths, product_tokens) -> str | None:
    """Refusal text when EVERY requested product lost an input, else None.

    Nothing can be drawn, so this is a refusal and not a note: it is
    raised at the front door, before a run directory is claimed and
    before a wrfout is opened for rendering, and it names the variable,
    the product and the way back.
    """

    tokens = [str(token).strip() for token in product_tokens
              if str(token).strip()]
    killed = history_killed_products(paths, tokens)
    if not killed or set(killed) != set(tokens):
        return None
    return explain.layered(_history_selection_action(killed, everything=True),
                           _HISTORY_SELECTION_WHY)


def history_selection_notice(paths, product_tokens) -> str | None:
    """One ``note:`` line for the requested products this run cannot draw.

    The survivors are still drawn -- refusing the whole command because
    one product of four lost its input would be worse than the silence
    it replaces.
    """

    killed = history_killed_products(paths, product_tokens)
    if not killed:
        return None
    return explain.layered(_history_selection_action(killed, everything=False),
                           _HISTORY_SELECTION_WHY)


def _file_variables(path: Path) -> set[str] | None:
    """The wrfout's variable names, or None when they cannot be read.

    None is not a failure here: it means the pre-check abstains and
    rendering decides, which is exactly the behaviour that predates the
    pre-check.  The read is names only -- no field values -- so it is
    not a derived quantity and does not belong to the mandated ``wrf``
    package.
    """

    try:
        with _netcdf_read(path) as dataset:
            return set(dataset.variables)
    except Exception:
        return None


def _list_products_matplotlib(path: Path) -> list[tuple[str, str, str, str]]:
    """(product, kind, status, detail) rows for the matplotlib catalog."""

    with _netcdf_read(path) as ds:
        present = set(ds.variables)
    rows = []
    for product in _MPL_PRODUCT_NEEDS:
        missing = missing_declared_inputs(present, product)
        if missing:
            rows.append((product, "matplotlib", "missing-fields",
                         f"not in file: {', '.join(missing)}"))
        else:
            rows.append((product, "matplotlib", "renderable",
                         "required variables present"))
    return rows


def list_products_main(args: argparse.Namespace, engine: str) -> int:
    """``woof render --list-products WRFOUT...``: catalog + availability."""

    failures = 0
    try:
        groups = (group_history_series(args.wrfout)
                  if getattr(args, "series", False) else [[path] for path in args.wrfout])
    except ValueError as error:
        print(f"render FAIL: {error}", file=sys.stderr)
        return 1
    for group in groups:
        path = group[-1]
        print(f"render: product catalog for {path} (engine {engine})")
        try:
            if engine == "rust":
                from woof import rustwx

                renderer = rustwx.find_renderer()
                import tempfile
                with tempfile.TemporaryDirectory(
                        prefix="gpuwm-rwlist-") as store:
                    if getattr(args, "series", False):
                        rows, summary = rustwx.list_products_series(
                            renderer, group, store_root=Path(store), heavy=args.heavy)
                    else:
                        rows, summary = rustwx.list_products(
                            renderer, path, store_root=Path(store), heavy=args.heavy)
            else:
                rows = _list_products_matplotlib(path)
                counts: dict[str, int] = {}
                for _, _, status, _ in rows:
                    counts[status] = counts.get(status, 0) + 1
                summary = f"total={len(rows)} " + " ".join(
                    f"{status}={count}"
                    for status, count in sorted(counts.items()))
        except (RuntimeError, OSError) as exc:
            print(f"render FAIL: {exc}", file=sys.stderr)
            failures += 1
            continue
        for slug, kind, status, detail in rows:
            print(f"  {status:<14} {kind:<9} {slug:<40} {detail}")
        print(f"render: {path}: {summary}")
    return 1 if failures else 0


def catalog_main(args) -> int:
    """``woof render --list-products`` with no file: the vocabulary.

    Two different questions share the flag.  With a wrfout it means
    "which of these can THIS file render, and why not the rest" --
    availability, which needs the file.  Without one it means "what may
    I put in ``--products``", which needs only the build, and which is
    the question someone has before they have run anything at all.

    Answered by asking the engine, never from a copy kept here: the rust
    catalog is the renderer's, and a second list in this module is the
    enumeration-drift failure this project keeps paying for.  With no
    usable rust engine this REFUSES rather than answering from the other
    engine's five: ``--engine auto`` would not draw those products
    either (render law, audit F7), so listing them would be a menu
    nothing serves.  ``--engine matplotlib`` still lists its own.
    """

    from woof import rustwx

    try:
        engine, why = _resolve_engine(args.engine)
    except (RuntimeError, FileNotFoundError) as exc:
        # ``--engine rust`` is a refusal now, not a fallback, and this
        # entry point is reached before the render path's own try.  The
        # other two failures here print one line and exit 2; a renderer
        # that cannot answer for itself is the same shape of answer.
        print(f"render: {exc}", file=sys.stderr)
        return 2
    notice = matplotlib_workaround_notice(engine)
    if notice is not None:
        print(notice, file=sys.stderr)
    print(f"render: product catalog (engine {engine})")
    if engine == "matplotlib":
        for product in PRODUCTS:
            print(f"  {product}")
        print(f"render: {len(PRODUCTS)} products")
        return 0
    renderer = rustwx.find_renderer()
    try:
        result = subprocess.run(
            [str(renderer), "--list-products"], capture_output=True,
            text=True, errors="replace", env=rustwx.renderer_env())
    except OSError as exc:
        print(f"render: renderer failed to launch: {exc}", file=sys.stderr)
        return 2
    if result.returncode != 0:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        print(f"render: {tail[-1] if tail else f'exit {result.returncode}'}",
              file=sys.stderr)
        return 2
    print(result.stdout, end="")
    # The short names --help advertises, with the engine product each one
    # asks for, so the two answers to "what may I put in --products" agree.
    print("render: shared names: " + ", ".join(
        f"{name} -> {slug}" for name, slug in RUST_PRODUCT_ALIASES.items()))
    print("render: `woof render --list-products WRFOUT` adds which of "
          "these that file can actually render, and why not the rest")
    return 0


def require_renderer() -> Path:
    """The usable ``rw_wrfbatch``, or the refusal that names its absence.

    The SAME gate :func:`_resolve_engine` reads, at the LAST seam before
    the engine is launched.  The CLI already asked, but the in-process
    callers do not go through the CLI (``woof go``'s first-products
    leg, the DA gallery, the verification door's synoptic panels,
    tests), and a gate that only one caller passes through is a gate the
    next caller walks around.  There is no fallback to offer here --
    weather fields belong to this binary -- so the reason is raised.
    """

    from woof import rustwx

    renderer = rustwx.find_renderer()
    if renderer is None:
        raise RuntimeError(_unresolvable_renderer_refusal(
            "it is not built (nothing on the resolution ladder holds a "
            "rw_wrfbatch binary)"))
    reason = renderer_refusal(renderer)
    if reason is not None:
        raise RuntimeError(_unresolvable_renderer_refusal(
            f"the renderer at {renderer} may not be used by this tree: "
            f"{reason}"))
    return renderer


def render_series_rust(paths, *, products: str, timeidx: int | None,
                       outdir: Path, size: tuple[int, int] | None,
                       heavy: bool = False,
                       source_label: str | None = None,
                       layout: str = render_layout.DEFAULT_LAYOUT,
                       overlays: Path | None = None,
                       annotate: Path | None = None,
                       streamlines: bool | None = None,
                       theme: str | None = None,
                       section: str | None = None,
                       isotherms: str | None = None,
                       section_across_km: float | None = None,
                       section_size: tuple[int, int] | None = None,
                       section_top_km: float | None = None,
                       fills: list | None = None,
                       context_paths=(),
                       ) -> tuple[list[Path], list[str],
                                  list[tuple[str, str]]]:
    """One store over a whole wrfout SERIES; ``(written, failures, skipped)``.

    :func:`render_wrfouts_rust` renders one file per store, which is the
    campaign convention and the right default for independent inputs.
    It is also the one thing a WINDOWED product cannot be drawn under:
    ``qpf_6h`` is defined as F012 minus F006, so a store holding only
    F012 has nothing to difference and the engine skips it by name.
    This renders the whole series into ONE store, which is what a
    verification door's 6 h accumulation -- two separate hourly wrfout
    files -- actually needs.

    The domain token comes from the LAST file, the frame whose valid
    time the panels carry; every file in a series is one nest by
    construction, because a store that mixed two nests would merge two
    grids into one run.  The lifecycle episode comes from the same file
    for the same reason: a store spanning a retire/re-arm boundary would
    difference one life of a nest against another.  Placement and
    rebranding are :func:`_place_engine_output`'s, unchanged, so these
    panels land in the same ``<out>/<domain>/<product>/<valid-day>/``
    layout as every other render.
    """

    from woof import rustwx

    series = [Path(item) for item in paths]
    if not series:
        raise ValueError("a render series needs at least one wrfout file")
    renderer = require_renderer()
    if source_label is None:
        source_label = default_source_label()
    outdir.mkdir(parents=True, exist_ok=True)
    subject = series[-1]
    token = domain_token(_domain_tag(subject), _grid_spacing_m(subject))
    episode = history_episode(subject)
    width, height = size if size is not None else (None, None)
    context = {Path(path).resolve() for path in context_paths}
    wanted_times = ({stamp for path in series if path.resolve() not in context
                     for stamp in _history_series_record(path)[1]}
                    if context else None)
    prior_georef = render_georef.read(outdir / render_georef.GEOREF_FILENAME)
    engine_georef = None
    drawn_at: dict[Path, Path] = {}
    with scratch_store(outdir) as store:
        # Context frames supply accumulation baselines. Render them only in
        # owned scratch so previously published first pictures retain bytes
        # and timestamps; only requested history frames are delivered.
        engine_out = store / "png" if context else outdir
        imported = store / "catalog-import.json"
        available, unavailable = _available_window_request(
            renderer, subject, products, store, heavy=heavy, paths=series,
            section=section, save_import=imported)
        if not available:
            return [], [], unavailable
        batches = (["all"] if timeidx is None else [str(timeidx)])
        if timeidx is None and wanted_times is not None:
            batches = _wanted_slots(
                series, wanted_times,
                each=_joined_across_moves(subject, context_paths)) or batches
        written, failures, skipped = [], [], []
        for frames in batches:
            batch = rustwx.run_renderer_series(
                renderer, series, store_root=store, out_dir=engine_out,
                products=available, frames=frames, width=width,
                height=height, heavy=heavy,
                source_label=source_label, overlays=overlays,
                annotate=annotate, streamlines=streamlines, theme=theme,
                section=section, isotherms=isotherms,
                section_across_km=section_across_km,
                section_size=section_size,
                section_top_km=section_top_km, fills=fills,
                from_import=imported if imported.is_file() else None)
            written.extend(batch[0])
            failures.extend(batch[1])
            skipped.extend(batch[2])
            if engine_out != outdir:
                # The engine drew into the store, so its map record is
                # there too and goes with the store: read it after EVERY
                # launch, since each launch may replace what the last one
                # wrote.  Drawn into ``outdir``, the record stays put and
                # file_pictures reads it under the lock.
                launched = render_georef.read(
                    engine_out / render_georef.GEOREF_FILENAME)
                if launched is not None:
                    engine_georef = (
                        launched if engine_georef is None else
                        render_georef.merge(engine_georef, launched,
                                            root=engine_out))
        skipped = unavailable + skipped
        if wanted_times is not None:
            selected = []
            for png in written:
                moment = render_layout.engine_output_time(png.name)
                if moment is None:
                    # Which frame drew it is unknowable from its name, so
                    # it cannot be told from a discarded baseline; it is
                    # named and left undelivered, never the end of a run
                    # whose frames are already drawn.
                    print(f"render: warning: not delivered, the frame time "
                          f"cannot be read from the file name: {png}",
                          file=sys.stderr)
                    skipped.append((png.stem, "frame time not readable from "
                                    "the file name; picture not delivered"))
                    continue
                if moment not in wanted_times:
                    continue
                # A native SFIRE panel is drawn already filed under
                # <domain>/<product>/<valid-day>/; it keeps that path.
                try:
                    relative = png.relative_to(engine_out)
                except ValueError:
                    relative = Path(png.name)
                delivered = outdir / relative
                os.makedirs(render_layout.fs_path(delivered.parent),
                            exist_ok=True)
                os.replace(render_layout.fs_path(png),
                           render_layout.fs_path(delivered))
                selected.append(delivered)
                drawn_at[delivered] = png
            written = selected
    written = render_georef.file_pictures(
        outdir, written,
        lambda png: _place_engine_output(png, outdir, token, layout,
                                         episode=episode),
        batch_dir=engine_out, engine=engine_georef, drawn_at=drawn_at,
        prior=prior_georef)
    return written, failures, skipped


def _joined_across_moves(subject: Path, context_paths) -> bool:
    """Whether a context frame is the subject's nest at another place
    (:func:`history_series_groups` joined it across a move)."""

    mine = _history_series_identity(subject)[1]
    for path in context_paths:
        theirs = _history_series_identity(path)[1]
        if _another_place(mine, theirs):
            return True
    return False


def _another_place(mine, theirs) -> bool:
    """Whether two nest keys are one nest at two places on the ground.

    Either the recorded place differs (the nest moved in its parent), or
    the recorded place is the same and the coordinates are not: a nest
    that rides a parent which moved (a cyclone run's 1 km d03, fixed in
    its following 3 km d02) records its place in that parent, which never
    changes, while its ground moves with the parent.  Keyed on the
    recorded place alone, its frames on either side of a parent's move
    were two unrelated series, and its 1 h rain was drawn at no hour in
    which the parent moved (run_c12e1e07306546b4: 10 of 12 hours).
    """

    return mine[:3] == theirs[:3] and (mine[3] != theirs[3]
                                       or mine[5][0] != theirs[5][0])


def _wanted_slots(series, wanted_times, *, each: bool = False) -> list[str] | None:
    """``[index]`` when ONE frame of a context series is wanted, else ``None``.

    ``each``: one launch per wanted frame, however many are wanted, for a
    series joined across a moving nest's moves.  Its baselines are the
    nest's frames at other places, usually many more than the frames the
    series delivers, and a launch into the same store reuses the first
    one's import.  Measured on test host D over a 3 km 48 x 48 storm nest's
    series of 20 frames (17 baselines, 3 wanted, every product): 17.7 s
    drawing every frame, 6.2 s for three single-frame launches, and the
    206 pictures both drew byte-identical.  Drawing every frame also put a
    product the engine draws once per store (terrain height, on the
    store's first frame) on a discarded baseline, so a place the nest
    moved to never delivered it; the single-frame launches draw it at
    each wanted frame.

    A series with context frames exists so a windowed product has its
    baselines, not so the baselines are drawn: with ``--frames all`` the
    engine draws every baseline and the caller throws those pictures
    away.  The engine draws one stored slot per launch (``--frames N``,
    ordinal over the store's ascending valid times), and this was
    decided when every launch imported the whole series again: measured
    at 1 km (414x402) on the 5070 Ti host, about 6 s per frame to import
    against 3.5 s to draw one frame's 204 pictures, so one launch per
    wanted frame won only when a single frame was wanted.  A later launch
    into the same store now reuses the first one's import, but it still
    reads and hashes every frame of the series to find it, so the rule
    is kept until that cost is measured against drawing the baselines.
    """

    stamps = sorted({stamp for path in series
                     for stamp in _history_series_record(path)[1]})
    wanted = sorted(set(wanted_times) & set(stamps))
    if each and wanted and len(stamps) > len(wanted):
        return [str(stamps.index(stamp)) for stamp in wanted]
    if len(wanted) != 1 or len(stamps) < 2:
        return None
    return [str(stamps.index(wanted[0]))]


def _history_identity_attribute(name: str) -> bool:
    """Immutable run/grid/science identity, excluding the per-frame carrier ledger.

    Wrfout's radiation carrier SOURCE and LAST_UPDATE describe which update
    supplies the current frame, including analysis-to-radiation handoff. They
    change while one forecast advances. Initial-condition source/cycle/lead,
    physics selectors, domain episode, projection and actual coordinates still
    distinguish independent histories; none of those are carrier ledger keys.
    """
    names = {"START_DATE", "SIMULATION_START_DATE", "GRID_ID", "PARENT_ID",
             "I_PARENT_START", "J_PARENT_START", "PARENT_GRID_RATIO", "DX", "DY",
             "MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON", "CEN_LAT",
             "CEN_LON", "MOAD_CEN_LAT", "POLE_LAT", "POLE_LON", "HYBRID_OPT", "ETAC",
             "DT", "TITLE"}
    # A running ledger (GPUWM_CHEM_LEDGER: the smoke budget, rewritten at
    # every history write) is per-frame state like the carrier ledger, not
    # identity.  Counted as identity it made every frame of a smoke run its
    # own series, so a windowed product never had a second whole-hour frame
    # to fold: the default HRRR v4 recipe's go render exited 1 on its
    # windowed pass ("'windowed' names no product these frames can draw").
    return (name in names or name.endswith("_PHYSICS")
            or name.startswith("GPUWM_") and not name.startswith("GPUWM_CARRIER_")
            and not name.endswith("_LEDGER"))


#: What one moving nest's history frames differ by and nothing else: where
#: the nest sits in its parent, and the centre that follows from it (with
#: XLAT/XLONG).  The history writer refreshes all of them after a move.
_PLACEMENT_ATTRIBUTES = frozenset({"I_PARENT_START", "J_PARENT_START",
                                   "CEN_LAT", "CEN_LON"})


def _history_series_record(path: Path) -> tuple[tuple, tuple[datetime.datetime, ...]]:
    """Identify one actual grid and run before sharing its native time store."""
    key, _nest, stamps = _history_series_identity(path)
    return key, stamps


def _history_series_identity(path: Path
                             ) -> tuple[tuple, tuple, tuple[datetime.datetime, ...]]:
    """``(series key, nest key, valid times)`` of one history file.

    The series key is the grid's whole identity, its place included, which
    is what one store may hold.  The nest key is ``(folder, episode, the
    identity without the place, the recorded place, (ratio, nx, ny), the
    ground)``, the place being :data:`_PLACEMENT_ATTRIBUTES` and the
    coordinates: two frames whose nest keys differ only in the place are
    one moving nest before and after a move, whose earlier frames a later
    series needs to close its windows (:func:`history_series_groups`).
    That includes frames at one RECORDED place with other coordinates: a
    nest fixed in a parent that moves keeps its recorded place while its
    ground moves (:func:`_another_place`).  The ground is ``(coordinates
    digest, lat min, lat max, lon min, lon max)``; it and the extent say
    whether two places share ground (:func:`_places_share_ground`).
    """
    import netCDF4

    path = Path(path).resolve()
    try:
        with _netcdf_read(path) as dataset:
            attributes = dataset.ncattrs()
            metadata = {name: np.asarray(dataset.getncattr(name)).tolist()
                        for name in attributes
                        if _history_identity_attribute(name)}
            for name in ("GRID_ID", "DX", "DY"):
                if name not in metadata:
                    raise ValueError(f"missing {name}")
            if not metadata.get("START_DATE") and not metadata.get("SIMULATION_START_DATE"):
                raise ValueError("missing model initialization date")
            dimensions = [(name, len(value)) for name, value in dataset.dimensions.items()
                          if name not in {"Time", "DateStrLen"}]
            nest = hashlib.sha256(json.dumps(
                {name: value for name, value in metadata.items()
                 if name not in _PLACEMENT_ATTRIBUTES},
                sort_keys=True, allow_nan=False).encode("utf-8"))
            nest.update(repr(sorted(dimensions)).encode("utf-8"))
            digest = hashlib.sha256(json.dumps(metadata, sort_keys=True,
                                               allow_nan=False).encode("utf-8"))
            digest.update(repr(sorted(dimensions)).encode("utf-8"))
            ground = hashlib.sha256()
            coordinates = {}
            for name in ("XLAT", "XLONG"):
                variable = dataset.variables[name]
                values = np.asarray(variable[:])
                if variable.dimensions[0] == "Time":
                    first = values[0]
                    if not all(np.array_equal(frame, first) for frame in values):
                        raise ValueError(f"{name} changes within this file")
                else:
                    first = values
                if not np.isfinite(first).all():
                    raise ValueError(f"{name} contains nonfinite coordinates")
                for hashed in (digest, ground):
                    hashed.update(name.encode("ascii"))
                    hashed.update(str(first.dtype).encode("ascii"))
                    hashed.update(np.ascontiguousarray(first).tobytes())
                coordinates[name] = np.asarray(first, dtype=np.float64)
            stamps =tuple(datetime.datetime.strptime(str(stamp), "%Y-%m-%d_%H:%M:%S")
                           for stamp in netCDF4.chartostring(dataset.variables["Times"][:]))
            if not stamps or any(left >= right for left, right in zip(stamps, stamps[1:])):
                raise ValueError("history times are empty, duplicate, or not increasing")
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        raise ValueError(f"Cannot group history series {path}: {error}") from error
    # Separate folders are separate run/lifecycle authorities. In particular,
    # never difference independent case folders or a nest's retire/rearm lives.
    episode = history_episode(path)
    place = (metadata.get("I_PARENT_START"), metadata.get("J_PARENT_START"))
    sizes = dict(dimensions)
    extent = (metadata.get("PARENT_GRID_RATIO"), sizes.get("west_east"),
              sizes.get("south_north"))
    # The ground the frame covers: its coordinates' digest, and their
    # latitude span and longitude span (unwrapped about the first cell, so
    # a grid across 180 degrees keeps one span).
    lat, lon = coordinates["XLAT"], coordinates["XLONG"]
    anchor = float(lon.flat[0])
    lon = (lon - anchor + 180.0) % 360.0 - 180.0 + anchor
    span = (ground.hexdigest(), float(lat.min()), float(lat.max()),
            float(lon.min()), float(lon.max()))
    return ((str(path.parent), episode, digest.hexdigest()),
            (str(path.parent), episode, nest.hexdigest(), place, extent, span), stamps)


def _places_share_ground(mine, theirs) -> bool:
    """Whether two recorded places of one nest have a mass cell in common.

    A nest moves by whole parent cells, so place ``(i, j)`` and place
    ``(i', j')`` are ``(i - i') * ratio`` nest cells apart in x (and so in
    y), and they overlap exactly when that is less than the grid's width
    in both.  A place whose numbers are not all recorded shares nothing:
    no move can be read from it.

    At one recorded place with other coordinates (the parent moved, see
    :func:`_another_place`) the recorded places say nothing about the
    ground, so the coordinates' spans decide: grids whose latitude and
    longitude spans do not both overlap have no cell in common.  The
    renderer then reads the whole-cell move from the coordinates and
    checks it on every shared cell (rw_wrfbatch ``locate_shift``).
    """

    if mine[3] == theirs[3]:
        (_, south, north, west, east) = mine[5]
        (_, south_other, north_other, west_other, east_other) = theirs[5]
        if east_other - west_other < 180.0 and east - west < 180.0:
            # Both spans unwrapped about their own first cell: bring the
            # other's longitudes to within 180 degrees of this one's.
            offset = round(((west + east) - (west_other + east_other)) / 720.0) * 360.0
            west_other, east_other = west_other + offset, east_other + offset
        return (south < north_other and south_other < north
                and west < east_other and west_other < east)
    (i, j), (ratio, nx, ny) = mine[3], mine[4]
    (i_other, j_other) = theirs[3]
    try:
        dx = abs(int(i) - int(i_other)) * int(ratio)
        dy = abs(int(j) - int(j_other)) * int(ratio)
        return int(ratio) > 0 and dx < int(nx) and dy < int(ny)
    except (TypeError, ValueError):
        return False


def group_history_series(paths, *, context_paths=()) -> list[list[Path]]:
    """Group compatible files, with explicit earlier context for a continuation.

    The file lists of :func:`history_series_groups`, which says which frames
    of each are context.
    """
    return [series for series, _context in
            history_series_groups(paths, context_paths=context_paths)]


def history_series_groups(paths, *, context_paths=()
                          ) -> list[tuple[list[Path], list[Path]]]:
    """Group compatible files, with explicit earlier context for a continuation.

    ``(series, context)`` per group, ``context`` being the frames of the
    series that are there to close its windows and are not drawn.

    Ordinary inputs retain their separate directory authorities. An explicit
    context frame in another directory may join one unique target series only
    when its grid, episode and scientific identity match and its times precede
    that series. This lets a restart's saved history supply its first window
    without silently joining independent target runs.

    A MOVING NEST's frames at an earlier place join each later place's series
    as context: same folder, same episode, the same identity but for the place
    (:func:`_history_series_identity`), valid before that series' last frame,
    at a place that shares ground with the series' own
    (:func:`_places_share_ground`).  The renderer moves them onto the series'
    place by the move the frames record, with the ground they did not cover
    missing, so the first window after a move is drawn (rw_wrfbatch
    ``nest_move``).  Split by place alone, a 3 km storm-following nest drew
    its 1 h rain only at the hours it did not move (4 of 12) and the render
    stage failed.  A place the nest has travelled a whole width away from
    holds no value on the series' ground: joined anyway, every such frame was
    imported (and with ``--frames all`` drawn) again for every later place,
    and the renderer refused the whole series over it.  A nest fixed in a
    parent that moves is joined the same way (:func:`_another_place`); the
    renderer reads its move from the coordinates.  Ground a nest newly
    covers stays missing for a window that began before it arrived: its
    accumulations there count only the nest's own rain since it arrived
    (nest-only totals, ``physics_continuation.STRIP_ACCUMULATORS``).
    """
    groups = {}
    nests = {}
    context = {Path(path).resolve() for path in context_paths}
    earlier = []
    for raw in paths:
        path = Path(raw)
        key, nest, stamps = _history_series_identity(path)
        nests[key] = nest
        if path.resolve() in context:
            earlier.append((path, key, stamps))
            continue
        groups.setdefault(key, []).append((path, stamps))
    target_keys = tuple(groups)
    first_target = {key: min(stamps[0] for _, stamps in groups[key]) for key in target_keys}
    for path, key, stamps in earlier:
        if key not in target_keys:
            matches = [target for target in target_keys
                       if target[1:] == key[1:] and stamps[-1] < first_target[target]]
            if len(matches) > 1:
                # Joining it to one of them would difference that run's
                # rain against another run's history.
                raise ValueError(f"History context {path} matches {len(matches)} runs "
                                 "rendered together, so it cannot say which one it "
                                 "continues; render one continuation at a time")
            if matches:
                key = matches[0]
        groups.setdefault(key, []).append((path, stamps))
    def moved_from(other, key):
        # Same folder, episode and identity, another place (recorded, or
        # on the ground under a parent that moved), and ground in common
        # with this one.
        mine, theirs = nests.get(key), nests.get(other)
        return (other != key and mine is not None and theirs is not None
                and _another_place(mine, theirs)
                and _places_share_ground(mine, theirs))

    moved = {}
    for key, rows in groups.items():
        if all(path.resolve() in context for path, _stamps in rows):
            continue
        last = max(stamps[-1] for _path, stamps in rows)
        own = {stamp for _path, stamps in rows for stamp in stamps}
        moved[key] = [
            (path, stamps)
            for other, other_rows in groups.items() if moved_from(other, key)
            for path, stamps in other_rows
            if stamps[-1] < last and own.isdisjoint(stamps)]
    result = []
    for key, rows in groups.items():
        before = moved.get(key, [])
        rows = sorted([*rows, *before], key=lambda item: item[1][0])
        seen = set()
        for path, stamps in rows:
            overlap = seen.intersection(stamps)
            if overlap:
                raise ValueError(f"History series has overlapping valid times at {path}; "
                                 "select one run and one copy of each history frame")
            seen.update(stamps)
        joined = {path.resolve() for path, _stamps in before}
        result.append((
            [path for path, _stamps in rows],
            [path for path, _stamps in rows
             if path.resolve() in context or path.resolve() in joined]))
    return result


def _available_window_request(renderer: Path, path: Path, products: str,
                              store: Path, *, heavy: bool, paths=None,
                              section=None, save_import: Path | None = None,
                              ) -> tuple[str, list[tuple[str, str]]]:
    """The request this invocation can actually draw; ``(spec, skipped)``.

    Every NAMED slug whose catalog row is not ``renderable`` is dropped
    here and reported as a skip carrying the engine's own reason.  The
    catalog is asked about exactly the files this invocation is about to
    render -- one file on the per-file route, the whole series store on
    the series route -- so its verdict is not a guess about a different
    store, and Python computes no time axis, no accumulation and no
    field list of its own.

    WHAT BREAKAGE THIS PREVENTS (gate law).  A named slug the catalog
    has refused, forwarded to the renderer anyway, is not skipped by it:
    ``rusty-weather``'s batch lane turns an unavailable window into
    ``ItemFailed``, and ``rw_wrfbatch`` exits nonzero when
    ``summary.failed > 0`` -- one product takes the WHOLE invocation
    down.  On the downscale route that invocation is the finalize render
    of a finished child, so six hours of integration ended with
    "Forecast failed" and a partial picture tree.

    This used to drop window-axis rows by matching two of the engine's
    English sentences.  The catalog has had FIVE windowed outcomes
    (``rw-wrfbatch/src/main.rs``): excluded on the whole-hour axis,
    excluded on the ordinal axis (renderers built before 2.8.0, which
    refused windows on a sub-hourly history), renderable, ``blocked``
    with a per-slug reason, and excluded because the window compute
    itself was unavailable -- whose sentence is composed at run time and
    can never be in a frozen set.  Two of five matched, and only while the wording
    held.  Matching on the STATUS matches all five and every status the
    catalog grows later, which is why the verdict is asked of
    :func:`woof.rustwx.catalog_verdict` rather than spelled again here.

    A token with no catalog row passes through untouched: the group
    keywords name no row and the engine expands them itself, leaving out
    what it cannot draw.  A non-empty request that is ONLY group
    keywords is returned unchanged without asking, because there is no
    named promise in it to check; an EMPTY request is not, and comes
    back empty so the caller refuses rather than running an empty
    render.

    The three STORELESS families are dropped first, by
    :func:`woof.rustwx.drop_storeless_terms` and before anything is
    launched: a store listing cannot decide a ``mesh:``, ``meshdiff:``
    or ``xsec:`` term, and the renderer answers one it cannot draw by
    refusing the whole invocation.  Doing it here rather than only at
    the front door covers every caller of this render path, and costs a
    mesh-only request no catalog listing it could not have used.

    Corrupt metadata and every other native refusal keep the original
    request, so the import/launch failure stays visible.
    """

    from woof import rustwx

    products, storeless = rustwx.drop_storeless_terms(
        products, section=section)
    # Whole terms: a section's level list is one product, never several.
    requested = rustwx.product_spec_terms(products)
    if not requested:
        # Nothing named and no group keyword: an empty or comma-only
        # request.  It comes back as an empty spec, which is the
        # caller's signal to refuse before launching -- the short
        # circuit below cannot decide it, because all() of nothing is
        # True and it would return the comma-only spelling straight
        # into the renderer.
        return "", storeless
    if all(token.lower() in rustwx.GROUP_KEYWORDS for token in requested):
        # The native automatic catalog already excludes what it cannot
        # draw from a group it expands itself.
        return products, storeless
    try:
        rows, _summary = rustwx.catalog_rows(
            renderer, (path,) if paths is None else paths,
            store_root=store, heavy=heavy, products=products,
            save_import=save_import)
    except (OSError, RuntimeError) as error:
        # No trustworthy availability verdict: run the unchanged native
        # request so its import/metadata/launch failure remains visible.
        #
        # SAYING SO is the difference between that and a silent forward.
        # This arm is the one path on which a named slug the catalog
        # would have refused still reaches the renderer, and some refused
        # slugs fail the whole invocation there (a `var:` term the store
        # does not hold) -- so a reader who got "exit 1" and a render
        # command had no way to learn that the availability question was
        # never answered.  It is a note, not a refusal: the request is
        # still attempted, and the engine's own failure is still what
        # decides the outcome.
        print(f"note: product availability could not be read for {path} "
              f"({error}); the request is sent to the renderer unchanged, "
              "so a product these frames cannot draw will fail this render "
              "rather than be skipped", file=sys.stderr)
        return products, storeless
    available, excluded = rustwx.catalog_verdict(rows, requested)
    if not excluded:
        return products, storeless
    # A catalog verdict is about every frame the listing imported, not
    # about one of them, so a series names its span rather than filing
    # the whole verdict against its last file.
    frames = [Path(item) for item in ((path,) if paths is None else paths)]
    subject = (str(frames[0]) if len(frames) == 1 else
               f"{frames[0]} to {frames[-1].name} ({len(frames)} frames)")
    codes = {row[0]: rustwx.catalog_code(row) for row in rows}
    return available, storeless + [(slug, f"{detail}; {subject}" if codes.get(slug)=="inactive-fire-domain" else f"{subject}: {detail}")
                                   for slug, detail in excluded]


def render_wrfouts_rust(paths, *, products: str, timeidx: int | None,
                        outdir: Path, size: tuple[int, int] | None,
                        heavy: bool = False,
                        source_label: str | None = None,
                        layout: str = render_layout.DEFAULT_LAYOUT,
                        overlays: Path | None = None,
                        annotate: Path | None = None,
                        streamlines: bool | None = None,
                        theme: str | None = None,
                        section: str | None = None,
                        isotherms: str | None = None,
                        section_across_km: float | None = None,
                        section_size: tuple[int, int] | None = None,
                        section_top_km: float | None = None,
                        fills: list | None = None,
                        series: bool = False, context_paths=(),
                        ) -> tuple[list[Path], list[str],
                                   list[tuple[str, str]]]:
    """Rusty Weather engine; ``(written, failures, skipped)``.

    One renderer invocation per wrfout file, each with its own scratch
    store (two files in one store would merge into one run's timeline;
    per-file isolation keeps every input independently comparable, the
    campaign convention).  The scratch store lives under the output
    directory and is removed as soon as the file's render finishes.

    ``overlays``/``annotate`` are JSON files in the renderer's own
    geographic-overlay schema (coordinates in degrees; see
    ``tools/rustwx/crates/rustwx-products/src/geographic_overlays.rs``).
    They are what the tile-streamed and DA lanes needed and could not
    have: a boundary-zone box, tile seams, storm-report markers, radar
    sites and range rings, on a panel the production renderer drew.
    Omitted, the renderer runs no overlay code and the PNGs are
    byte-identical to every earlier build.

    ``theme`` is a built-in theme name (``default``, ``dark``) or a JSON
    theme file in the engine's own schema
    (``tools/rustwx/crates/rustwx-render/src/theme.rs``): the surface,
    the inks, the basemap linework, the colorbar chrome, the fonts and
    the colormap overrides, as one file.  Omitted, the engine draws its
    own look and the PNGs are byte-identical to every earlier build.
    ``section`` is the line ``xsec:`` products are cut along
    (``lat,lon,lat,lon`` or a JSON file), ``isotherms`` the isotherm set
    drawn on them, ``section_across_km`` an optional second frame across
    the line, ``section_top_km`` the ceiling of the fitted height range
    (1-40 km; the engine's own 14 km when none is named, which is why a
    shallow feature used to occupy the bottom fourteenth of every
    published cut).

    ``fills``, when a list is given, collects the range each vertical
    cut's colour bar was drawn over and the rule that set it, for the
    receipt.  A section's bar is fitted to its own frame at both ends,
    so two cuts of one line are compared through that record rather
    than by colour.
    """

    from woof import rustwx

    if series:
        context = {Path(path).resolve() for path in context_paths}
        written, failures, skipped = [], [], []
        for group, group_context in history_series_groups(
                [*paths, *context_paths], context_paths=context_paths):
            if all(path.resolve() in context for path in group):
                continue
            options = dict(products=products, timeidx=timeidx, outdir=outdir,
                           size=size, heavy=heavy, source_label=source_label,
                           layout=layout, overlays=overlays, annotate=annotate,
                           streamlines=streamlines, theme=theme, section=section,
                           isotherms=isotherms, section_across_km=section_across_km,
                           section_size=section_size,
                           section_top_km=section_top_km, fills=fills)
            if len(group) == 1:
                batch = render_wrfouts_rust(group, **options)
            else:
                batch = render_series_rust(
                    group, context_paths=group_context, **options)
                for png in batch[0]:
                    print(f"render: {png}")
            written.extend(batch[0])
            failures.extend(batch[1])
            skipped.extend(batch[2])
        return written, failures, skipped
    renderer = require_renderer()
    if source_label is None:
        source_label = default_source_label()
    outdir.mkdir(parents=True, exist_ok=True)
    frames = "all" if timeidx is None else str(timeidx)
    width, height = size if size is not None else (None, None)
    written: list[Path] = []
    failures: list[str] = []
    skipped: list[tuple[str, str]] = []
    for path in (Path(p) for p in paths):
        # The domain this file PROVES, read the same way the matplotlib
        # engine reads it, from the file's own GRID_ID/DX.  The engine
        # spells the same token into its filenames, but evidence from
        # the file beats a transcription, and one file's outputs all
        # belong to one nest whatever any filename says.
        token = domain_token(_domain_tag(path), _grid_spacing_m(path))
        # And which LIFE of that nest, read from the folder the history
        # writer filed the file in.  Two episodes of one nest can carry
        # one valid time -- the retiring one's last frame and the
        # re-armed one's activation frame -- and the engine names them
        # identically, so without this the second delivery replaces the
        # first.
        episode = history_episode(path)
        prior_georef = render_georef.read(
            outdir / render_georef.GEOREF_FILENAME)
        with scratch_store(outdir) as store:
            # Availability can drop a product and therefore change the scoped
            # run name. Draw from the listing's import, which already contains
            # every surviving product's inputs, instead of importing again.
            imported = store / "catalog-import.json"
            available, unavailable = _available_window_request(
                renderer, path, products, store, heavy=heavy,
                section=section, save_import=imported)
            skipped.extend(unavailable)
            if not available:
                # Main still returns nonzero when the entire request
                # produced no image, and names these exclusions.
                continue
            file_written, file_failures, file_skipped = rustwx.run_renderer(
                renderer, path, store_root=store, out_dir=outdir,
                products=available, frames=frames, width=width,
                height=height, heavy=heavy, source_label=source_label,
                overlays=overlays, annotate=annotate,
                streamlines=streamlines, theme=theme, section=section,
                isotherms=isotherms, section_across_km=section_across_km,
                section_size=section_size, section_top_km=section_top_km,
                fills=fills, from_import=imported if imported.is_file() else None)
        # Filed and recorded in one hold of the manifest's lock: the
        # engine keyed this launch's pictures by the flat names they are
        # about to leave, and the next launch drops any entry whose file
        # has left, so the move and the re-key cannot be apart.
        file_written = render_georef.file_pictures(
            outdir, file_written,
            lambda png, token=token, episode=episode: _place_engine_output(
                png, outdir, token, layout, episode=episode),
            prior=prior_georef)
        written.extend(file_written)
        failures.extend(file_failures)
        skipped.extend(file_skipped)
        for png in file_written:
            print(f"render: {png}")
    return written, failures, skipped


#: What the shipped product calls its rendered files.
OUTPUT_FILENAME_PREFIX = "arwen_"

#: What the vendored engine calls them: rustwx-products hardcodes
#: ``rustwx_`` into every output filename format string (e.g.
#: ``tools/rustwx/crates/rustwx-products/src/derived.rs``).
_ENGINE_FILENAME_PREFIX = "rustwx_"


def _rebrand_engine_output(png: Path) -> Path:
    """The product's filename for one engine-written PNG.

    The vendored engine stamps its own name into every file it writes;
    the product these files come out of is WOOF-ArWen, and a PNG named for
    the vendored crate is a first-run head-scratcher (the field run
    asked what "rustwx" was).  The rename happens here -- the one seam
    every wheel-shipped render flows through -- rather than in the
    vendored Rust, which keeps the engine byte-identical to its
    campaign builds.  Filenames only: module names, store identities
    and the pair-compose lead marker (which reads both spellings) are
    untouched.

    A name the engine did not brand passes through unchanged, and so
    does one whose rename fails -- a drawn image under yesterday's name
    beats a failure over branding.
    """

    if not png.name.startswith(_ENGINE_FILENAME_PREFIX):
        return png
    target = png.with_name(
        OUTPUT_FILENAME_PREFIX + png.name[len(_ENGINE_FILENAME_PREFIX):])
    try:
        os.replace(png, target)
    except OSError:
        return png
    return target


def _place_engine_output(png: Path, outdir: Path, domain: str,
                         layout: str, *,
                         episode: int | None = None) -> Path:
    """Rebrand one engine-written PNG and file it under the layout.

    The rust engine writes every picture straight into ``--out-dir``
    with a name of its own, which is what made a real run's output a
    single directory of thousands of files.  It is not asked to change:
    the vendored crate stays byte-identical to its campaign builds, and
    the placement happens here, at the same seam the rebrand already
    happened at -- one move per file, after the renderer has exited and
    the file is complete.

    Every failure degrades to leaving the picture where the engine put
    it, and SAYS SO on stderr, naming the file and why.  A drawn image
    at the old path beats no image over filing -- layout is not
    correctness -- but the silent form of this exact fallback is how a
    parser that could not read the engine's one-digit cycle hours
    shipped every 00Z-09Z run flat under a front door printing ``layout
    nested``.  The caller's returned list names wherever the file
    actually is, so nothing downstream is told about a file that is not
    there.
    """

    png = _rebrand_engine_output(png)
    if layout == render_layout.FLAT:
        return png
    # Native panels can already be filed at their declared domain/product/day.
    # Preserve that path instead of treating a valid native filename as flat.
    try:
        native_parts = png.relative_to(outdir).parts
    except ValueError:
        native_parts = ()
    if len(native_parts) == 4:
        native_domain, native_product, native_day, native_name = native_parts
        native_stamp = render_layout.NATIVE_PANEL.fullmatch(native_name)
        if (re.fullmatch(r"d[0-9]+(?:-[0-9]+(?:\.[0-9]+)?[mk]+)?|native_grid", native_domain)
                and native_stamp and native_product == native_stamp.group(1)
                and native_day == native_stamp.group(2)[:8]):
            return png
    parsed = render_layout.parse_engine_output(png.name, domain=domain)
    if parsed is None:
        # Not a name this engine's grammar produces -- so nothing here
        # knows which product or valid time it is, and inventing a
        # folder for it would file it under a guess.
        print(f"render: warning: left flat, filename does not parse as "
              f"engine output: {png}", file=sys.stderr)
        return png
    filed_domain, product, day = parsed
    # The delivered name drops the domain and product tokens, because
    # the two folders it is about to sit in spell exactly those and a
    # frame carrying them twice cost a measured delivery 310 characters
    # -- past MAX_PATH for every tool the recipient opens it WITH, which
    # `fs_path` does nothing about.  Done here, at the organisation
    # step, rather than in the engine: the vendored crate stays
    # byte-identical to its campaign builds, and `--layout flat` keeps
    # the v2.4.1 name (this branch is nested-only; flat returned above).
    # The episode is the CALLER's, read from the history file this
    # picture was drawn from: the engine's filename cannot carry it,
    # because the engine knows nothing about a nest's lifecycle.  Absent
    # -- every run that declares no retire/rearm -- the path is the
    # three segments it has always been.
    target = render_layout.place(
        outdir, domain=filed_domain, product=product, day=day,
        episode=episode,
        filename=render_layout.delivered_name(
            png.name, domain=filed_domain, product=product),
        layout=layout)
    if target == png:
        return png
    try:
        # Through render_layout.fs_path: three folders plus the engine's
        # filename put the longest products past Windows' MAX_PATH from
        # any real case root, and without this the warning below fires
        # for exactly those products -- the layout ruling inverted for
        # `composite_reflectivity` while `2m_temperature` beside it
        # files correctly.
        spelled = render_layout.fs_path(target)
        Path(spelled).parent.mkdir(parents=True, exist_ok=True)
        os.replace(render_layout.fs_path(png), spelled)
    except OSError as error:
        print(f"render: warning: left flat, could not move into layout "
              f"({error}): {png}", file=sys.stderr)
        return png
    return target


#: A render's working scratch lives BESIDE the delivered tree, in
#: ``<delivery><SCRATCH_SUFFIX>/``, never inside it.
SCRATCH_SUFFIX = render_layout.SCRATCH_SUFFIX


def scratch_root_for(outdir) -> Path:
    """Where working scratch for a delivery to ``outdir`` belongs.

    A SIBLING of the delivered directory.  Scratch used to be created
    inside it (``mkdtemp(dir=outdir)``), which breaks a delivery two
    ways, both seen on 2026-08-20: 25 ``.rwstore-*`` directories were
    left inside a delivered product folder, with paths long enough that
    Windows could not list the folder at all; and a ``tar`` pull of the
    tree raced a live render -- ``tar: .rwstore-p_4f7zzy: File removed
    before we read it``.  The removal is best-effort by nature (a
    memory-mapped hour file can hold its handle past process exit), so
    the guarantee "a delivered tree contains only products" cannot rest
    on cleanup.  It rests on the location.

    The sibling is taken beside the CASE ROOT, not beside the run
    folder: what gets pulled, tarred and browsed is the case folder the
    caller named with ``--out``, so scratch parked one level inside it
    would still be in the archive and still be in the way.  A run folder
    is recognised by its own name (``run_stamp.is_run_folder``), so this
    needs nothing threaded through from the door.
    """

    outdir = Path(os.path.abspath(outdir))
    anchor = outdir.parent if run_stamp.is_run_folder(outdir) else outdir
    return anchor.parent / f"{anchor.name}{SCRATCH_SUFFIX}"


#: The prefix a render's working stores take unless a door minted one
#: for a single stage.  Named because two functions match on it.
DEFAULT_SCRATCH_PREFIX = "rwstore-"

#: How a door hands its per-stage scratch prefix to the render
#: subprocess it spawns.  An environment variable rather than a flag
#: because the render command line is also PRINTED for a reader to
#: paste, and a pasted line carrying one run's private token would be
#: wrong the moment it is reused.
SCRATCH_PREFIX_ENV = "WOOF_RENDER_SCRATCH_PREFIX"

_SCRATCH_PREFIX_SHAPE = re.compile(
    rf"^{re.escape(DEFAULT_SCRATCH_PREFIX)}[0-9a-zA-Z]+-$")

#: A working store a door's minted token names: the token, then the
#: characters ``tempfile.mkdtemp`` appended.  The plain-prefix store of a
#: render nobody spawned does not match, so no marker can be written for
#: one and no later run can remove one.
_SCRATCH_STORE_SHAPE = re.compile(
    rf"^{re.escape(DEFAULT_SCRATCH_PREFIX)}[0-9a-zA-Z]+-[0-9a-zA-Z_]+$")


def stage_scratch_prefix() -> str:
    """Mint the prefix naming ONE render stage's working stores.

    A door that spawns a render stage owns exactly the stores that stage
    creates, and owns no other.  Matching on a time window cannot say
    which is which: a second render into the same delivery whose store
    opens a second after this stage started is indistinguishable from
    this stage's own.  A token minted here, handed to the stage through
    :data:`SCRATCH_PREFIX_ENV` and matched on afterwards, says it
    exactly -- nothing another door created can carry this token.

    It keeps :data:`DEFAULT_SCRATCH_PREFIX` as its head so that every
    working store, minted or not, is still recognisable as one, and so
    a reader clearing a scratch root by hand needs one glob.

    EIGHT hex digits, not a whole uuid.  The token is on the front of
    every working-store path under ``<case>.render-scratch/``, a
    directory this project's own layout page already records as long
    enough to break a Windows directory listing, and 32 bits of a random
    uuid separate the handful of stages one delivery ever has open at
    once exactly as well as 128 do.
    """

    import uuid

    return f"{DEFAULT_SCRATCH_PREFIX}{uuid.uuid4().hex[:8]}-"


def scratch_prefix() -> str:
    """The prefix THIS process's working stores take.

    The door's minted token when one was handed down, and
    :data:`DEFAULT_SCRATCH_PREFIX` otherwise -- a render nobody spawned
    from a door (``woof render`` typed directly) is nobody's to sweep
    and takes the plain prefix.

    A value that is not a mintable token is not a reason to refuse a
    render: the store is created under the plain prefix instead, and the
    line says so once, because the only consequence is that the door
    which handed it down will not recognise this store as its own.
    """

    named = os.environ.get(SCRATCH_PREFIX_ENV, "").strip()
    if not named:
        return DEFAULT_SCRATCH_PREFIX
    if _SCRATCH_PREFIX_SHAPE.match(named):
        return named
    global _WARNED_SCRATCH_PREFIX
    if not _WARNED_SCRATCH_PREFIX:
        _WARNED_SCRATCH_PREFIX = True
        print(f"render: warning: {SCRATCH_PREFIX_ENV}={named!r} is not a "
              f"{DEFAULT_SCRATCH_PREFIX}<token>- prefix; this render's "
              "working store takes the plain prefix and the door that "
              "spawned it will leave the store alone if it dies.",
              file=sys.stderr)
    return DEFAULT_SCRATCH_PREFIX


_WARNED_SCRATCH_PREFIX = False


def sweep_abandoned_scratch(outdir, *, prefix: str) -> list[Path]:
    """Remove working stores a dead render left beside ``outdir``.

    :func:`scratch_store` removes its own store on the way out, including
    on an exception -- but only if the process lives to run the ``finally``.
    A renderer that exits nonzero usually does; one that is OOM-killed, or
    a stage subprocess that dies on a signal, does not, and the store stays
    in ``<case>.render-scratch/`` holding the whole tiled hour file it was
    working on.  MEASURED on one failed 1 km render: 3 abandoned stores,
    11.4 GiB, beside a delivery that had published nothing.  The door that
    saw the stage fail is the one place that knows the render is over, so
    it sweeps here and SAYS what it removed -- silent deletion beside a
    failure is how evidence disappears.

    ``prefix`` IS THE OWNERSHIP TEST, and a door passes the token it
    minted with :func:`stage_scratch_prefix` and handed to the stage that
    died.  :func:`scratch_store` promises that concurrent renders into
    one delivery each keep their own store and none removes another's,
    and a sweep of everything under the plain prefix breaks that promise
    on POSIX, where ``rmtree`` deletes a directory whose files another
    process still has open and the live render simply loses its work.
    Matching on the minted token cannot reach another door's store
    whenever it was created; matching on when a store appeared can, and
    the interleaving it misses (a second render whose store opens just
    after this stage started) is the common one.

    A store under the plain prefix is therefore never swept by a door,
    and this function REFUSES the plain prefix rather than defaulting to
    it: a default that matches every working store is the blanket sweep
    two paragraphs up, one keyword away.  It is somebody's live render or
    an earlier crash's leavings, and ``woof render`` typed directly
    cleans up after itself.

    Returns the stores actually removed, in path order.  Never raises,
    and never reports a store it did not remove: a directory that will
    not go (a Windows handle still mapping an hour file) stays where it
    is and stays out of the returned list, and the caller's failure
    remains the failure being reported.
    """

    import shutil

    if not _SCRATCH_PREFIX_SHAPE.match(str(prefix)):
        raise ValueError(
            f"sweep_abandoned_scratch needs the token a door minted with "
            f"stage_scratch_prefix(), not {prefix!r}: sweeping every store "
            f"under the plain {DEFAULT_SCRATCH_PREFIX!r} prefix would "
            f"remove a concurrent render's live store, which on POSIX "
            f"loses that render's work. Mint a prefix, hand it to the "
            f"stage through WOOF_RENDER_SCRATCH_PREFIX, and sweep on it.")
    removed: list[Path] = []
    for store in owned_scratch_stores(outdir, prefix=prefix):
        # The extended-length spelling, for the reason
        # :func:`_remove_scratch_store` gives: the engine's hour files
        # sit past the Windows path ceiling, and a sweep that cannot
        # reach them removes nothing.
        shutil.rmtree(render_layout.fs_path(store, descend=True),
                      ignore_errors=True)
        if not _store_exists(store):
            removed.append(store)
    try:
        scratch_root_for(outdir).rmdir()
    except OSError:
        pass
    return removed


def owned_scratch_stores(outdir, *, prefix: str) -> list[Path]:
    """The working stores beside ``outdir`` that carry ``prefix``, in path order.

    What a door asks after one of its render stages has exited: anything
    still here under the stage's token is a store that stage opened and
    did not remove.  Refuses the plain prefix on the same terms as
    :func:`sweep_abandoned_scratch`, because a listing on it would claim
    every concurrent render's live store as this door's.
    """

    if not _SCRATCH_PREFIX_SHAPE.match(str(prefix)):
        raise ValueError(
            f"owned_scratch_stores needs the token a door minted with "
            f"stage_scratch_prefix(), not {prefix!r}: the plain "
            f"{DEFAULT_SCRATCH_PREFIX!r} prefix names every render's "
            f"store, not one door's.")
    try:
        return sorted(entry for entry in scratch_root_for(outdir).iterdir()
                      if entry.is_dir() and entry.name.startswith(prefix))
    except OSError:
        return []


def _store_exists(store) -> bool:
    return os.path.lexists(render_layout.fs_path(store, descend=True))


#: The name a door gives the marker it leaves beside a working store it
#: owned and still could not remove after all of its renders had exited:
#: ``<store><ABANDONED_SUFFIX>``.  The marker is the ownership proof a
#: later run needs, because a store's own name cannot say whether the
#: door that minted its token is still running.
ABANDONED_SUFFIX = ".abandoned"


def mark_abandoned_scratch(stores, *, run=None) -> list[Path]:
    """Mark stores whose owning run has ended so a later run removes them.

    Called by a door only once every render it spawned has exited and
    its own sweep still could not remove these (a handle another program
    holds on an hour file).  A store without this marker is never
    removed by :func:`sweep_marked_scratch`, so a concurrent render's
    live store cannot be reached by it.  Returns the markers written.
    """

    import json

    written: list[Path] = []
    for store in stores:
        store = Path(store)
        if not _SCRATCH_STORE_SHAPE.match(store.name):
            continue
        marker = store.with_name(store.name + ABANDONED_SUFFIX)
        try:
            marker.write_text(json.dumps(
                {"store": store.name,
                 "run": None if run is None else str(run)}) + "\n",
                encoding="utf-8")
        except OSError:
            continue
        written.append(marker)
    return written


def sweep_marked_scratch(folder) -> list[Path]:
    """Remove the stores earlier runs in ``folder`` marked as abandoned.

    The scratch roots of runs in ``folder`` sit at most two levels down:
    ``<folder>/<delivery>.render-scratch`` for a run written straight
    into it, and ``<folder>/<run>/<delivery>.render-scratch`` for a
    stamped run folder (only a folder named as one is looked inside, so
    a folder full of other things costs one listing).  Only a store
    carrying a marker from :func:`mark_abandoned_scratch` is touched; its
    marker goes with it.  Returns the stores removed, in path order.
    Never raises.
    """

    import shutil

    folder = Path(folder)
    removed: list[Path] = []
    try:
        roots = sorted({*folder.glob(f"*{SCRATCH_SUFFIX}"),
                        *(root for run in folder.iterdir()
                          if run_stamp.is_run_folder(run) and run.is_dir()
                          for root in run.glob(f"*{SCRATCH_SUFFIX}"))})
    except OSError:
        return removed
    for root in roots:
        try:
            markers = sorted(root.glob(f"*{ABANDONED_SUFFIX}"))
        except OSError:
            continue
        cleared = False
        for marker in markers:
            store = marker.with_name(marker.name[:-len(ABANDONED_SUFFIX)])
            if not _SCRATCH_STORE_SHAPE.match(store.name):
                continue
            if _store_exists(store):
                shutil.rmtree(render_layout.fs_path(store, descend=True),
                              ignore_errors=True)
                if _store_exists(store):
                    continue
                removed.append(store)
            try:
                marker.unlink()
                cleared = True
            except OSError:
                pass
        if cleared:
            # Only a root this sweep emptied: an empty root nobody marked
            # may be the one a live render is about to open its store in.
            try:
                root.rmdir()
            except OSError:
                pass
    return sorted(removed)


@contextlib.contextmanager
def scratch_store(outdir, *, prefix: str | None = None):
    """One renderer invocation's working store, outside the delivery.

    Removed on exit; the sibling root goes with the last store in it, so
    concurrent renders into the same delivery each keep their own and
    none removes another's (``rmdir`` refuses a non-empty directory).
    That promise survives a door's cleanup too: ``prefix`` defaults to
    :func:`scratch_prefix`, which is the token the door that spawned
    this process minted, and :func:`sweep_abandoned_scratch` matches on
    that token rather than on when a store appeared.

    A parent directory that cannot hold the sibling (read-only, or a
    delivery written straight into a mount point) is not a reason to
    fail a render OR to fall back into the delivered tree: the store
    goes to the system temp area instead, named so it is recognisable.
    """

    import tempfile

    if prefix is None:
        prefix = scratch_prefix()
    root = scratch_root_for(outdir)
    try:
        root.mkdir(parents=True, exist_ok=True)
        store = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    except OSError as error:
        root = Path(tempfile.mkdtemp(prefix="gpuwm-render-scratch-"))
        store = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
        print(f"render: scratch beside {outdir} was not writable "
              f"({error}); working store is {store}", file=sys.stderr)
    try:
        yield store
    finally:
        _remove_scratch_store(store)
        try:
            root.rmdir()
        except OSError:
            pass


def _remove_scratch_store(store: Path) -> bool:
    """Delete a per-file scratch store; True when it is gone.

    WALKED IN THE EXTENDED-LENGTH SPELLING, which is what makes this
    work on Windows at all.  The engine files each hour at
    ``<store>/wrf/local_<init>_<64 hex>_<profile>_science_v1/f000.rws``
    and writes it through the extended-length API.  MEASURED on the
    engine's own stores: an hour file sits 147 characters below its
    store, and a door's store sits 43 to 45 characters below the run
    folder, so an hour file passes the 260-character ceiling the
    ordinary API enforces (unless long paths are switched on for the
    whole machine) once the run folder's own path is about 70
    characters.  On the 2026-09-26 user sweep the run folder was 104
    characters and the hour files 294 to 296.  ``rmtree`` of the plain
    spelling failed on every try with WinError 145 ("the directory is
    not empty"), and every successful Windows run kept its stores
    (741 MB on a 24 h 3 km forecast, finding F7).

    The retries were written for handle lag after the renderer exits;
    the failure they kept meeting was this path ceiling (F7).  They stay
    only for a file another program (a scanner or an indexer) may hold
    for a moment after it is written, which no Windows render has
    measured yet.  A store that STILL cannot be removed is worth a
    warning, never a failed render; when a door minted this store's
    token the door removes it once its render has exited, and the line
    says so rather than calling it lost.
    """

    import shutil
    import time

    walk = render_layout.fs_path(store, descend=True)
    for delay in (0.0, 0.25, 0.5, 1.0, 2.0, 2.0, 2.0, 2.0):
        if delay:
            time.sleep(delay)
        try:
            shutil.rmtree(walk)
        except FileNotFoundError:
            if not _store_exists(store):
                return True
        except OSError:
            continue
        else:
            return True
    shutil.rmtree(walk, ignore_errors=True)
    if not _store_exists(store):
        return True
    if _SCRATCH_STORE_SHAPE.match(Path(store).name):
        print(f"render: warning: scratch store could not be removed yet; "
              f"the run that started this render removes it once its "
              f"renders have exited: {store}", file=sys.stderr)
    else:
        print(f"render: warning: scratch store left behind: {store}",
              file=sys.stderr)
    return False


def renderer_refusal(renderer) -> str | None:
    """Why this tree may not draw with ``renderer``, or None if it may.

    THE single answer to "is this renderer usable", and the only one.
    Both seams that can launch the engine read it -- :func:`_resolve_engine`
    for the CLI, :func:`render_wrfouts_rust` for every in-process caller
    (``woof go``'s first-products leg, the DA gallery) -- so a renderer
    can never be usable at one seam and foreign at the other.  Two
    independently correct gates landed here in one release and would
    have been exactly that: two definitions of usable, diverging on the
    first case they disagreed about.

    Two questions, in the order a reader wants them answered:

    * THE CONTRACT (task #106).  :func:`woof.rustwx.probe_renderer`
      runs ``--help``, then the ``--abi`` handshake against
      :data:`woof.rustwx.RENDERER_ABI_MARKER`.  This is the decisive
      one, because it is a statement about the BINARY's own answer
      rather than about where it sits: two builds of ``rw_wrfbatch``
      with different md5s both passed the old ``--help``-only probe and
      both reported ``verified``, and a build predating the handshake
      says ``unknown option --abi`` on exit 2.
    * THE PROVENANCE (task #125).
      :func:`woof.provenance_gate.renderer_bridge_refusal` refuses a
      binary that answers the contract correctly and still belongs to
      another tree -- the case the handshake cannot see, because a
      sibling checkout at the same contract version answers it
      perfectly.  ``find_renderer`` returns the first candidate that is
      a file and its last candidate is ``~/.woof/bridges``, a directory
      every woof on the machine writes into and none of them owns, so
      "nobody chose this binary" is a real and silent state.

    The reason is returned rather than raised, because what to DO about
    it belongs to the caller: ``auto`` degrades and says why, an
    explicit ``--engine rust`` refuses, and a direct in-process render
    has no fallback to offer at all.
    """

    from woof import rustwx
    from woof.provenance_gate import renderer_bridge_refusal

    ok, evidence = rustwx.probe_renderer(renderer)
    if not ok:
        return evidence
    # The provenance clause carries its own remedies (build here, or
    # declare the staged binary through the environment), so it is
    # returned verbatim rather than re-worded.
    return renderer_bridge_refusal(renderer, env_var=rustwx.RENDERER_ENV)


def _unresolvable_renderer_refusal(reason: str) -> str:
    """The one refusal both ``auto`` and ``rust`` raise, layered.

    THE render-law fix (hidden-scope audit F7, 2026-08-18).  Weather
    fields belong to ``rw_wrfbatch``; the law (project render law,
    2026-08-06) named exactly one fallback, the DA nowcast's sheet tool
    (removed in 3.0.3), which drew multi-panel sheets from a DA case
    directory -- it had no product for a wrfout's reflectivity, temperature, wind,
    precipitation or OLR panel.  So there is nothing lawful to fall back
    to here, and the answer is a refusal that names the missing artifact
    and stages it.

    ``woof fetch-bridges`` leads because it is the remedy that works on
    a wheel install, which is where the defect was found: staging failed
    silently, ``--engine auto`` drew five matplotlib weather fields, and
    the command exited 0.  The cargo one-liner follows for a checkout,
    and the explicit matplotlib workaround is named last -- naming the
    exit is not the same as taking it (1.8.8 refusal sweep).
    """

    from woof import rustwx

    # The reason may itself be a LAYERED message -- the provenance
    # clause is one -- and nesting two `[[explain]]` marks in one string
    # truncates this remedy at the inner mark for every consumer that
    # splits on the first one (`explain.render`, and the JSON catalog
    # that carries the action half into a document).  So the inner
    # explanation is lifted out and appended to the outer one, which is
    # where it belonged: one message, one action half, one why half.
    reason, nested_why = explain.split(str(reason))
    reason = reason.rstrip()
    # No `woof render:` prefix here.  Every terminal caller adds one
    # (`render_main` and `catalog_main` both print "render: " + this),
    # and a message that carries its own prefix printed "render: woof
    # render: ..." on the first line a refused user sees.  The subject
    # is named in the sentence instead, which also reads correctly
    # where there is no prefix at all -- the run-plan catalog document
    # carries this string verbatim in its `error` field.
    action = (
        f"the rust render engine ({rustwx.RENDERER_NAME}) is "
        f"not usable here: {reason}\n"
        "  Weather-field product plots come from the real Rust renderer, "
        "so this is refused rather than drawn by something else.\n"
        "  remedy: woof fetch-bridges\n"
        f"  # stages {rustwx.RENDERER_NAME} and the basemap assets it "
        "draws from\n"
        f"  #   ... or, from a checkout: {rustwx.CARGO_BUILD_HINT}\n"
        f"  #   ... or set {rustwx.RENDERER_ENV} to a built binary you "
        "mean to use\n"
        "  # `woof render --engine matplotlib` still exists as a NAMED "
        "WORKAROUND;\n"
        "  # it draws five analysis-grade panels, not the production "
        "catalog.")
    why = _RENDER_LAW_WHY
    if nested_why.strip():
        why = f"{why}\n\n{nested_why.strip()}"
    return explain.layered(action, why)


#: How every message here names the one engine allowed to draw a
#: weather field, so the phrase cannot drift between call sites.
_RENDERER_SUBJECT = "the Rust renderer rw_wrfbatch, through woof.rustwx"

_RENDER_LAW_WHY = (
    "The project render law ( 2026-08-06) reserves weather fields for the "
    "Rust renderer; the one fallback it named composed DA nowcast sheets, "
    "served none of this door's products, and is no longer shipped.  "
    "`--engine auto` used to "
    "degrade to a SECOND fallback: on a box where bridge staging failed "
    "it drew composite reflectivity, 2 m temperature, 10 m wind, "
    "accumulated precipitation and OLR with matplotlib and exited 0, so "
    "an operator got pictures and no signal that the 151-product "
    "production catalog had never run.  `auto` now has two answers, the "
    "rust engine or this refusal, and the exit code says which.")


def _resolve_engine(requested: str) -> tuple[str, str]:
    """(engine, why) for ``--engine auto|rust|matplotlib``.

    Rust is selected exactly when the binary resolves AND
    :func:`renderer_refusal` has nothing to say about it.  ``auto`` and
    ``rust`` now differ only in wording: NEITHER degrades, because the
    thing they would degrade to is matplotlib drawing weather fields,
    which the render law does not allow (audit F7).  ``auto`` means
    "resolve the engine for me", not "draw with whatever is lying
    around".

    The explicit path used to return the resolved binary WITHOUT the
    probe -- only ``auto`` asked the contract question -- so the one
    caller that pinned ``--engine rust`` to be certain of the real
    renderer was exactly the caller that could still get a foreign one:
    with a stale bridge staged, ``auto`` fell back naming the mismatch
    while ``rust`` ran the stale build silently.  An explicit request is
    a statement about which engine must draw, so failing its contract is
    a refusal, never a silent substitution and never a fallback.

    ``--engine matplotlib`` remains reachable, by that name only, and
    every run of it prints :func:`matplotlib_workaround_notice`.
    """

    if requested == "matplotlib":
        return "matplotlib", "requested"
    from woof import rustwx

    renderer = rustwx.find_renderer()
    if renderer is None:
        if requested == "rust":
            raise RuntimeError(
                "--engine rust: the renderer is not built; "
                + rustwx.renderer_remedy())
        raise RuntimeError(_unresolvable_renderer_refusal(
            "it is not built (nothing on the resolution ladder, including "
            "~/.woof/bridges, holds a rw_wrfbatch binary)"))
    reason = renderer_refusal(renderer)
    if reason is None:
        return "rust", str(renderer)
    if requested == "rust":
        # NO SILENT SUBSTITUTION -- an explicit --engine rust that got a
        # different engine is the defect this refusal exists for -- but
        # the exit is named, because one demonstrably exists: the same
        # command with --engine matplotlib renders the analysis-grade
        # subset on this identical tree.  Refusing to choose for the
        # caller is not the same as refusing to tell them what the
        # choices are (1.8.8 refusal sweep).
        raise RuntimeError(
            f"--engine rust: the renderer at {renderer} failed its "
            f"contract check: {reason}\n"
            f"  Rebuild the renderer ({rustwx.CARGO_BUILD_HINT}), stage "
            f"one with `woof fetch-bridges`, or pass --engine matplotlib "
            f"to take the named workaround outright.")
    raise RuntimeError(_unresolvable_renderer_refusal(
        f"the renderer at {renderer} failed its contract check: {reason}"))


def matplotlib_engine_gap():
    """The requirement that stops the matplotlib engine drawing, or ``None``.

    The fallback engine is not a fallback until this is ``None``.  It
    computes every derived field through ``wrf`` (pip distribution
    ``wrf-rust``), the mandated science core, which lives behind the
    same ``render`` extra the rust engine's documentation offers as the
    thing to fall back FROM.  Asked without importing, so this is safe
    to call on any path including ``--help``.
    """

    from woof import capabilities

    if capabilities.is_installed(capabilities.SCIENCE_CORE.module):
        return None
    return capabilities.SCIENCE_CORE


def drawable_engine() -> tuple[str | None, str]:
    """Which engine ``woof go`` may chain into, and why -- or ``(None, why)``.

    Exactly one engine is chainable: the rust one.  ``(None, why)`` when
    it is not staged or fails its contract, and the chain then SKIPS its
    render stage with that reason printed, which is the lawful shape of
    "no imagery here" -- a forecast that finished plus a named absence.

    The matplotlib engine is deliberately absent from this answer even
    when its science core is installed (audit F7).  It draws weather
    fields, the render law reserves those for ``rw_wrfbatch``, and a
    chain that reaches it automatically is precisely the second fallback
    the law forbids.  It stays reachable by name at the ``woof render``
    front door, where a human typed it.
    """

    try:
        engine, why = _resolve_engine("auto")
    except Exception as error:                              # noqa: BLE001
        # Deliberately broad, and only here.  This function is asked by
        # `woof go` DURING a chain, purely to decide whether to run the
        # render stage; resolving the rust engine means launching it for
        # its contract handshake, and a launch that fails for any reason
        # must degrade this answer, never end a forecast that has
        # already succeeded.  The reason is carried, not swallowed --
        # and the first line is what the chain prints, so the layered
        # explain half is dropped here rather than pasted into a stage
        # log.
        first = str(error).split("[[explain]]")[0].strip()
        return None, first or f"the rust engine could not be probed: {error}"
    if engine == "rust":
        return "rust", why
    return None, (
        f"the rust render engine is not available ({why}); weather-field "
        f"product plots come from {_RENDERER_SUBJECT}, so the chain draws "
        "nothing rather than substituting another engine")


def engine_refusal(engine: str, requested: str, why: str) -> str | None:
    """The refusal for an engine that cannot draw, or ``None``.

    THE fix for the wrong-remedy-first defect.  A bare install used to
    print ``rust render engine not available ... Build it with: woof
    fetch-bridges`` -- an advisory about the OTHER engine -- and then
    die in a traceback whose tail carried the remedy that would actually
    have worked.  The first line a reader sees is now the one that fixes
    the thing that is broken.

    Both routes are named, in the order that costs least: staging the
    rust engine is one command, needs no extra, and draws the whole
    catalog (measured 161 PNGs against 5 renderable matplotlib products
    on the same file).  An explicit ``--engine matplotlib`` is a request
    for the fallback, so its own remedy leads there instead.
    """

    if engine != "matplotlib":
        return None
    gap = matplotlib_engine_gap()
    if gap is None:
        return None
    if requested == "matplotlib":
        action = (
            "woof render --engine matplotlib: the matplotlib engine "
            f"needs {gap.label}, which is not installed.\n"
            "  The fallback engine computes every derived field through "
            "it, so it needs the same extra as the rust engine it falls "
            "back from.\n"
            f"{gap.remedy}")
    else:
        action = (
            f"woof render: neither render engine can draw here.\n"
            f"  The rust engine is not available ({why}), and the "
            f"matplotlib fallback needs {gap.label}, which is not "
            "installed.\n"
            "  remedy: woof setup\n"
            "  # stages the rust render engine (no extra needed); it "
            "draws the\n"
            f"  # full catalog, against {len(PRODUCTS)} products for the "
            "fallback\n"
            "  #   ... or, for the matplotlib fallback engine:\n"
            "  #   pip install 'recast-woof[render]'")
    return explain.layered(action, _ENGINE_WHY)


_ENGINE_WHY = (
    "Refused at the front door, before a single wrfout was opened.  The "
    "version this replaces printed an advisory naming the rust engine's "
    "build command -- a remedy for the engine that was NOT selected -- "
    "and then raised, so the correct install line arrived at the tail of "
    "a traceback, below the wrong one.\n\n"
    "`woof render --list-products FILE` is deliberately not refused "
    "here: the matplotlib catalog reads the file with netCDF4 alone and "
    "answers correctly on an install with no render extra at all.")


#: How many products each engine can draw, for the fallback notice.
#: The rust catalog's 151 is its implicit-render candidate count on a
#: typical wrfout; the matplotlib count is read from :data:`PRODUCTS`,
#: so adding a fallback product cannot leave this notice overstating
#: what the fallback costs.
_RUST_CATALOG_PRODUCTS = 151


def matplotlib_workaround_notice(engine: str) -> str | None:
    """ONE line saying the matplotlib engine is drawing, in the workaround voice.

    Silence here cost a pilot a whole session: four products came out,
    they looked right, and nothing said the 151-product rust catalog was
    simply not installed.  An engine that does not announce itself is
    indistinguishable from the real thing until someone counts.

    It is a WORKAROUND now, not a fallback, and the word is the point.
    Nothing degrades into this engine any more (audit F7): it is
    reachable only by typing ``--engine matplotlib``, and "Fixed means
    default" (project ruling, 2026-08-10) says an opt-in remedy must be REPORTED as
    a workaround every time it runs.  The prefix is this tree's existing
    spelling for that -- ``woof/static/rust_bridge.py`` prints the same
    ``WORKAROUND:`` line whenever the Python geog reader stands in.

    One line, and it carries everything needed to act on it: the engine
    actually used, the products available against the rust catalog, and
    a fix that is true on THIS install.  The multi-line remedy belongs
    to ``woof doctor``; a notice that scrolls is a notice that gets
    skimmed.

    That last clause used to be the bare ``cd tools/rustwx; cargo build``
    one-liner, naming a directory a wheel install does not have.  The
    1.0.1 remedy contract fixed that everywhere except here, because
    this call site assembled its own string.  It goes through the same
    install-aware machinery now, in the one-line form: the one-liner
    where the crate exists, a pointer to ``woof doctor`` where it does
    not -- because the accurate answer there is a whole bootstrap and this
    notice gets exactly one physical line.
    """

    if engine != "matplotlib":
        return None
    from woof import bridges, rustwx

    fix = bridges.install_aware_one_line_hint(
        rustwx.CARGO_BUILD_HINT, rustwx.RUSTWX_CRATE_RELATIVE)
    return (f"WORKAROUND: render engine matplotlib -- the render law "
            f"reserves weather-field product plots for {_RENDERER_SUBJECT}; "
            f"{len(PRODUCTS)} of the rust catalog's "
            f"{_RUST_CATALOG_PRODUCTS} products are renderable this way. "
            f"Stage the real engine with `woof fetch-bridges`, or build "
            f"it: {fix}")


def _skip_reason(detail: str, sources=()) -> str:
    """One skip's reason as the note's line gives it: without its file.

    Every skip is recorded against the input it came from: ``PATH:
    reason`` from the renderer, ``PATH[frame] carries no FIELD`` from the
    matplotlib engine, ``FIRST to LAST (N frames): reason`` for a series
    verdict.  On the one line ``woof go`` relays, that path is a run
    folder's absolute path repeated once per product, and it pushed the
    part a reader acts on -- which field, which window -- off the end.
    The path stays in the per-item rows behind ``--explain``.

    ``sources`` are the invocation's inputs, so only a prefix that IS one
    of them is taken off: a reason whose text merely contains a colon (a
    storeless term's sentence) is given whole.  A reason no source opens
    is given whole too, which costs length and never the reason.
    """

    text = str(detail)
    spellings = {str(item) for item in sources}
    spellings |= {str(Path(item)) for item in sources}
    for source in sorted(spellings, key=len, reverse=True):
        if not source or not text.startswith(source):
            continue
        rest = text[len(source):]
        frame = re.match(r"\[\d+\]", rest)
        if frame is not None:
            rest = rest[frame.end():]
            if rest.startswith(" "):
                # ``PATH[frame] carries no FIELD``: the file is the
                # sentence's subject, and it lacks the field in every
                # frame (the matplotlib engine reads presence per file).
                return " ".join(("the file" + rest).split()).rstrip(".")
        span = re.match(r" to .+? \(\d+ frames\)", rest)
        if span is not None:
            rest = rest[span.end():]
        if rest.startswith(": "):
            reason = " ".join(rest[2:].split()).rstrip(".")
            if reason:
                return reason
    return " ".join(text.split()).rstrip(".") or "no reason given"


def inactive_domain_skips(skipped) -> bool:
    """Only native catalog declarations of inactive fire domains succeed empty."""
    return bool(skipped) and all(detail.startswith("inactive-fire-domain: IFIRE=0 ")
                                for _product, detail in skipped)


#: How the engine opens the reason of a product the input can never draw,
#: as opposed to one it lacks a field or a frame for (``rusty-weather``
#: ``windowed_store::NOT_APPLICABLE_PREFIX``): ``qpf_1h`` on a grid whose
#: history is written every 3 h.
NOT_APPLICABLE_SKIP = "not-applicable: "


def not_applicable_skips(skipped) -> bool:
    """Whether every skip is a product that does not apply to these inputs.

    WHAT BREAKAGE THIS PREVENTS (gate law): the finalize stage's windowed
    pass over a 3-hourly nest asked for ``qpf_1h`` alone, the engine
    declared it not applicable, nothing was drawn, the render exited 1 and
    ``woof go`` printed ``FAILED render`` on a run whose every picture
    was in place.  Such a pass succeeds empty, as an inactive fire domain
    does; a skip for a missing field or a gap still exits 1.
    """
    return bool(skipped) and all(
        detail.startswith("inactive-fire-domain: IFIRE=0 ")
        or NOT_APPLICABLE_SKIP in detail
        for _product, detail in skipped)


def skip_notice(skipped: list[tuple[str, str]],
                wrote_any: bool = True, drawn=None,
                sources=()) -> str | None:
    """ONE line for the products this render did not draw, or ``None``.

    A skip is not a failure and must not print as one -- but it must
    print, and it must print the PRODUCT NAMES.  A bare count is the
    shape of message a reader cannot act on: it says something is
    missing from the directory without saying what, so the only way to
    find out is to diff a listing against the catalog.

    Names and each name's first reason in the action half, then, as
    ``refl: the file carries no REFL_10CM``: which field or window a
    product lacked is what a reader needs to act, so it is on the line.
    The file each reason was recorded against is not (see
    :func:`_skip_reason`; ``sources`` are the invocation's inputs), and
    the per-item evidence -- every skip, which file, which frame --
    stays behind ``--explain``, verbatim from the engine.

    The ``note:`` prefix is essential rather than decorative.  It is
    this tree's word for "true and worth knowing, not a fault", and
    ``woof go``'s stage runner surfaces a passing stage's ``warning:``
    and ``note:`` lines through its output capture -- so without it this
    sentence is printed by ``woof render`` and swallowed by the chain,
    which is the silent success this change exists to avoid.

    The action half also states what the skips mean for the exit code,
    and it must state it TRUTHFULLY in both arms.  With at least one
    image drawn (``wrote_any``), skips do not change the exit code and
    the notice says so.  With NOTHING drawn, the skips are exactly why
    the command exits nonzero (the deliberate ask-for-one-absent-product
    arm of the main dispatch), and the old reassurance -- "not a failure
    and does not change the exit code" -- was false there: a real user
    who requested ``lifted_index`` from a wrfout got rendered=0, exit 1,
    and a note insisting nothing had failed.  That arm now says nothing
    rendered and points at ``--list-products``, which prints the
    per-product verdict and reason for THIS file.

    ``drawn`` is the set of product folders this render DID fill.  With
    it, the note tells apart a product that drew no picture at all from
    one that was only skipped at some frames -- ``qpf_1h`` at F000, where
    no hour has ended yet -- because naming both the same way put a
    product with 24 pictures in the same sentence as three with none.

    Each name carries the first reason recorded for it, on the one line
    ``woof go`` relays, as :func:`woof.render_receipts.undrawn_note`
    gives the run's closing note.  The line used to give one fixed cause
    for every row, "the file(s) do not carry their declared input fields
    or required time windows", which is wrong for a frame too large for
    the renderer's pressure-level volume: that frame carries every field,
    and the engine's own note says the volume passed its ceiling.
    """

    if not skipped:
        return None
    from woof.render_receipts import _drawn_family

    names = sorted({product for product, _detail in skipped})
    first_reason: dict[str, str] = {}
    for product, detail in skipped:
        first_reason.setdefault(product, _skip_reason(detail, sources))
    why = "; ".join(f"{name}: {first_reason[name]}" for name in names)
    if inactive_domain_skips(skipped):
        verdict = "The domains declare no active fire model; this is a successful skip"
    elif not wrote_any and not_applicable_skips(skipped):
        verdict = ("These products do not apply to these inputs; this is a "
                   "successful skip")
    elif wrote_any:
        verdict = ("That is not a failure and does not change the "
                   "exit code")
    else:
        verdict = ("Nothing else was drawn, so nothing rendered and the "
                   "exit code is nonzero; run --list-products to see "
                   "which products this file can support")
    split = ""
    if drawn is not None:
        drawn = set(drawn)
        never = [name for name in names if _drawn_family(name) not in drawn]
        partial = [name for name in names if name not in never]
        clauses = []
        if never:
            clauses.append(f"{', '.join(never)} drew no picture in this "
                           "render")
        if partial:
            verb = "was" if len(partial) == 1 else "were"
            clauses.append(f"{', '.join(partial)} drew pictures and "
                           f"{verb} skipped only at the frames named below")
        split = "; ".join(clauses) + ".  "
    return explain.layered(
        f"note: render skipped {len(skipped)} product render(s) "
        f"({', '.join(names)}) -- {why}.  {split}{verdict}",
        "\n".join(f"  skipped {product}: {detail}"
                  for product, detail in skipped))


#: The line layers the renderer's cartopy fallback draws, relative to the
#: cartopy Natural Earth root.  Mirrors ``rustwx-render``'s
#: ``default_conus_feature_paths``: coastline, national borders, state
#: lines.  A cache holding only some of them draws only those.
_CARTOPY_LINE_LAYERS = (
    ("physical", "ne_10m_coastline.shp"),
    ("cultural", "ne_10m_admin_0_boundary_lines_land.shp"),
    ("cultural", "ne_50m_admin_1_states_provinces_lines.shp"),
)


def _cartopy_draws_every_line_layer() -> bool:
    from woof import rustwx

    root = rustwx.cartopy_natural_earth_root()
    return root is not None and all(
        (root / folder / name).is_file()
        for folder, name in _CARTOPY_LINE_LAYERS)


def basemap_gap(renderer) -> str | None:
    """What a picture from ``renderer`` lacks, and the fix, or None.

    The other way to ship a plot believing it is something it is not.
    A renderer with no basemaps does not fail, warn, or exit non-zero:
    it draws the weather over a blank rectangle, and 1.4.0 shipped a
    tropical cyclone with no coastline that way (F4/B-11).  2.7.x shipped
    every picture of every wheel install that way: the shapefiles came
    only from ``woof fetch-bridges``, which no install text ran.  They
    ship in the ``recast-woof-data`` companion now, so this answers only on an
    install where the companion was removed or edited -- and says so.

    Checked against the renderer's own resolution ladder plus the roots
    :func:`woof.rustwx.renderer_env` hands it, and it names the one
    command that fixes it.
    """

    from woof import rustwx

    if renderer is None or rustwx.resolve_basemap_dir(renderer) is not None:
        return None
    if _cartopy_draws_every_line_layer():
        # The renderer's own last-resort fallback, which it consults per
        # layer after the candidate roots.  Coastlines, borders and state
        # lines will be drawn, so warning here would be a false alarm on
        # every workstation that has ever run cartopy at those scales --
        # and a notice that cries wolf is a notice nobody reads.  A cache
        # that holds only some of the three (a coastline and no state
        # lines, say) still warns, because the picture lacks the rest.
        return None
    return ("no map assets resolve for the renderer, so pictures are drawn "
            "with no coastlines, borders or state lines; they ship in the "
            "recast-woof-data package, so reinstall it: "
            f"{rustwx.basemap_remedy()}")


def missing_basemap_notice(renderer) -> str | None:
    """ONE line when the renderer resolves but its map assets do not.

    The terminal form of :func:`basemap_gap`, printed by ``woof
    render``.  A run reports the same gap as a
    :data:`BASEMAP_MISSING_CODE` event instead
    (:func:`announce_missing_basemap`), because the render it spawns
    prints this to a stream nobody reads.
    """

    gap = basemap_gap(renderer)
    return None if gap is None else f"render: WARNING: {gap}"


#: The run event every in-run render path emits when the renderer it
#: drives has no map assets (:func:`announce_missing_basemap` says how
#: often).  A code rather than a
#: sentence, so the desktop, the terminal and the web page can each say
#: it in their own words.
BASEMAP_MISSING_CODE = "render_basemap_missing"

_BASEMAP_CHECKED: set[tuple[str, str]] = set()
_BASEMAP_CHECKED_LOCK = threading.Lock()


def _renderer_on_disk():
    """The renderer a render subprocess would drive, without resolving it.

    :func:`woof.rustwx.find_renderer` can refresh a stale staged bundle
    and can raise on a bad override; this is a reader asking where the
    binary sits so its map assets can be looked up, so it takes the
    first candidate on disk and leaves every judgement to the render.
    """

    from woof import rustwx

    for candidate in rustwx.renderer_candidates():
        if candidate.is_file():
            return candidate
    return None


def renderer_basemap_gap() -> str | None:
    """:func:`basemap_gap` for the renderer a render would drive, or None.

    For a caller that keeps a RECORD of the gap rather than writing to an
    event stream: a remote machine's picture gallery status and a local
    DA products record.  It answers every time it is asked, where
    :func:`announce_missing_basemap` answers once per process, because a
    record rewritten on every pass would otherwise lose the warning on
    the second pass.  Never raises: a check about the picture must not
    stop the picture.
    """

    try:
        return basemap_gap(_renderer_on_disk())
    except Exception:  # noqa: BLE001 - telemetry never fails a render
        return None


#: The two moments a run reports the gap: when it draws its first
#: picture while the forecast runs, and when the finalize stage draws.
ANNOUNCE_STAGES = ("as-drawn", "finalize")


def announce_missing_basemap(warn, render_dir, *, stage: str) -> bool:
    """Report a renderer with no map assets to a run, once per stage.

    THE DEFECT THIS CLOSES: every picture of a wheel install was drawn
    with no coastlines, borders or state lines, and the only warning was
    :func:`missing_basemap_notice` printed by the ``woof render``
    subprocess -- to a stderr the early render, the every-frame render
    and the finalize stage all capture and drop.  No event carried it,
    so the desktop, the terminal workspace and the web page showed
    nothing.

    ``warn`` takes ``(code, message, **fields)`` like every run
    observer's.  Called by each in-run render path before it draws.  The
    first call for one ``render_dir`` and ``stage`` checks and the rest
    return at once, so the early render and the every-frame render share
    one report however many frames they draw.  ``stage`` is one of
    :data:`ANNOUNCE_STAGES`, and the finalize stage reports again because
    the terminal workspace and a remote machine's status read only the
    last 512 KiB of a run's events: on a long run the early report has
    left that window by the time the pictures are finished.  Returns
    whether it warned.  Never raises: a check about the picture must not
    stop the picture.
    """

    try:
        if stage not in ANNOUNCE_STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        key = (os.path.normcase(os.path.abspath(os.fspath(render_dir))),
               stage)
        with _BASEMAP_CHECKED_LOCK:
            if key in _BASEMAP_CHECKED:
                return False
            _BASEMAP_CHECKED.add(key)
        gap = basemap_gap(_renderer_on_disk())
        if gap is None:
            return False
        from woof import rustwx

        warn(BASEMAP_MISSING_CODE, gap, remedy=rustwx.basemap_remedy(),
             render_dir=str(render_dir), render_stage=stage)
        return True
    except Exception:  # noqa: BLE001 - telemetry never fails a run
        return False


def _pair_main(args: argparse.Namespace) -> int:
    # Pillow is checked HERE, by resolution, rather than by catching the
    # import of `woof.pair_compose`.  Two defects in one line:
    #
    # * the message named `recast-woof[render]`, which is wrf-rust + pyshp and
    #   has never contained Pillow.  Pillow arrives transitively with
    #   matplotlib, a BASE dependency, so the extra it named would not
    #   have installed it and the reader would have run the command,
    #   waited for a wheel, and met the same refusal.
    # * the guard was inert: `pair_compose` imports PIL inside its
    #   functions, so this module imports cleanly with no Pillow at all
    #   and the ImportError arrived later, from inside `compose_pairs`,
    #   as a traceback.
    from woof import capabilities

    if not capabilities.is_installed(capabilities.PILLOW.module):
        print(explain.render(
            capabilities.refusal("woof render --pair", capabilities.PILLOW),
            explain=explain.explain_enabled(args),
            command="woof render"), file=sys.stderr)
        return 2
    try:
        from woof.pair_compose import compose_pairs
    except ImportError as error:
        # Kept as the floor under the resolution check above: a Pillow
        # that resolves and then fails to import (a broken wheel, a
        # partial uninstall) is still a refusal, not a traceback.
        remedy = capabilities.remedy_for_error(error) or ""
        print(f"render: --pair could not load its image toolkit: {error}\n"
              + remedy, file=sys.stderr)
        return 2
    left, right = (Path(p) for p in args.pair)
    labels = args.pair_labels or (None, None)
    # A pair sheet is an output of this invocation like any picture, so
    # it gets its own run folder too.  No inputs are passed: a
    # comparison of two runs belongs to neither one's cycle, so the
    # stamp carries the launch instant and no init time.
    _claim_run_dir(args)
    try:
        try:
            sheets = compose_pairs(
                left, right, args.out, title=args.pair_title,
                subtitle=args.pair_subtitle, left_label=labels[0],
                right_label=labels[1], theme=args.theme)
        except ValueError as exc:
            print(f"render: {exc}", file=sys.stderr)
            return 2
        for sheet in sheets:
            print(f"render: {sheet}")
        print(f"render: {len(sheets)} pair sheet(s) -> {args.out}")
        return 0
    finally:
        _publish_run_dir(args)


# The renderer's refusals of products that have no difference, in a
# difference run (rustwx-render difference.rs): a sheet composed of several
# panels (refuse_composed), and a product drawn as category codes (Haines,
# fog flag), whose difference is not a quantity.  Marker -> what is said.
_DIFF_REFUSALS = {
    "is a sheet composed of several panels":
        "a sheet composed of several panels; a difference is drawn per map product",
    "draws category codes, and the difference of two codes is not a quantity":
        "drawn as category codes; the difference of two codes is not a quantity",
}


def _diff_main(args: argparse.Namespace) -> int:
    """``woof render --diff A_RUN B_RUN``: products as run A minus run B.

    Orchestration only: the frames are paired by their recorded valid time
    (:mod:`woof.render_difference`), and each pair is drawn
    by the Rust renderer, which reads both runs, checks that they share
    the valid time and the grid, and refuses by name when they do not.
    """

    from woof import render_difference, rustwx

    try:
        renderer = require_renderer()
        size = parse_size(args.size)
        timeidx = parse_timeidx(args.timeidx)
        products = parse_products_rust(args.products)
        if args.list_products:
            raise ValueError("--list-products describes one run; ask it of "
                             "each run without --diff")
        if getattr(args, "context_wrfout", ()):
            raise ValueError("--context-wrfout names one run's history, so it "
                             "cannot identify which --diff side it continues; "
                             "include earlier frames in the matching run folder")
        a_run, b_run = args.diff
        pairing = render_difference.pair_frames_by_valid_time(
            render_difference.run_frames(a_run),
            render_difference.run_frames(b_run),
            reader=_history_series_record, domain_reader=_domain_tag,
            series_groups=history_series_groups,
            timeidx=timeidx,
            series=getattr(args, "series", False) or Path(a_run).is_dir())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print("render: " + explain.render(
            str(exc), explain=explain.explain_enabled(args),
            command="woof render"), file=sys.stderr)
        return 2
    notice = render_difference.unpaired_notice(pairing)
    if notice is not None:
        print(f"render: note: {notice}", file=sys.stderr)
    if not pairing.pairs:
        print("render: --diff found no selected valid time both runs hold, so there "
              "is nothing to difference", file=sys.stderr)
        return 2
    labels = tuple(args.diff_labels) if args.diff_labels else (
        Path(a_run).name or "A", Path(b_run).name or "B")
    source_label = args.source_label or default_source_label()
    if getattr(args, "radar_colors", None):
        os.environ[rustwx.RADAR_COLORS_ENV] = args.radar_colors
    diff_table = getattr(args, "diff_table", None)
    if diff_table:
        if diff_table not in rustwx.DIFFERENCE_TABLE_CHOICES:
            if not Path(diff_table).is_file():
                print(f"render: --diff-table {diff_table}: no such file; give "
                      f"{' or '.join(rustwx.DIFFERENCE_TABLE_CHOICES)}, or the "
                      "path of a JSON difference table", file=sys.stderr)
                return 2
            diff_table = str(Path(diff_table).resolve())
    _claim_run_dir(args, [pair[2] for pair in pairing.pairs])
    written: list[Path] = []
    failures: list[str] = []
    skipped_rows: list[tuple[str, str]] = []
    # 'all' reaches the multi-panel sheets and the category-code products
    # too, which the renderer refuses to difference; under 'all' they are
    # skipped once each and said so, and only such a product named on the
    # command line is a failure.  Without this every default --diff exited 1.
    every_product = str(args.products).strip().lower() == "all"
    sheets_skipped: set[str] = set()
    try:
        args.out.mkdir(parents=True, exist_ok=True)
        for domain, valid, a_file, b_file in pairing.pairs:
            a_paths, b_paths, a_index = pairing.contexts[(domain, valid)]
            token = domain_token(_domain_tag(a_file), _grid_spacing_m(a_file))
            prior_georef = render_georef.read(
                args.out / render_georef.GEOREF_FILENAME)
            with scratch_store(args.out) as store:
                drawn, failed, skipped, differences = (
                    rustwx.run_renderer_difference(
                        renderer, a_paths, b_paths, timeidx=a_index, store_root=store,
                        out_dir=args.out, products=products, labels=labels,
                        sheet=args.diff_sheet, source_label=source_label,
                        difference_table=diff_table,
                        theme=args.theme,
                        overlays=args.overlays, annotate=args.annotate,
                        streamlines=args.streamlines,
                        width=size[0] if size else None,
                        height=size[1] if size else None))
            drawn = render_georef.file_pictures(
                args.out, drawn,
                lambda png, token=token: _place_engine_output(
                    png, args.out, token, args.layout,
                    episode=history_episode(a_file)),
                prior=prior_georef)
            for png in drawn:
                print(f"render: {png}")
            for row in differences:
                print(f"render: difference {row['key']} {valid:%Y-%m-%d %H:%MZ}"
                      f" bar +/-{row.get('half_range')} {row.get('units', '')}"
                      f" ({row.get('rule')}) cells={row.get('defined_cells')}"
                      f" differing={row.get('differing_cells')}"
                      f" max_abs={row.get('max_abs')}")
            for slug, reason in skipped:
                print(f"render: skipped {slug}: {reason}", file=sys.stderr)
            if every_product:
                kept = []
                for failure in failed:
                    marker = next((m for m in _DIFF_REFUSALS if m in failure), None)
                    if marker is None:
                        kept.append(failure)
                        continue
                    key = failure.split(marker)[0].split()[-1]
                    if key not in sheets_skipped:
                        sheets_skipped.add(key)
                        print(f"render: skipped {key}: {_DIFF_REFUSALS[marker]}",
                              file=sys.stderr)
                failed = kept
            for failure in failed:
                print(f"render: failed {failure}", file=sys.stderr)
            written.extend(drawn)
            failures.extend(failed)
            skipped_rows.extend(skipped)
        from woof.render_receipts import _drawn_family, publish_invocation

        publish_invocation(root=args.out, engine="rust",
            requested_spec=args.products, written=written,
            failures=failures,
            skipped=[(_drawn_family(slug) + "_difference", reason)
                     for slug, reason in skipped_rows], layout=args.layout,
            inputs=list(dict.fromkeys(path for pair in pairing.pairs for path in pair[2:])),
            context_inputs=list(dict.fromkeys(
                path for sides in pairing.contexts.values()
                for paths in sides[:2] for path in paths)))
        print(f"render: {len(written)} difference picture(s) from "
              f"{len(pairing.pairs)} valid time(s) -> {args.out}")
        return 1 if failures or not written else 0
    finally:
        _publish_run_dir(args)


def _claim_run_dir(args: argparse.Namespace, inputs=()) -> Path:
    """Rebind ``args.out`` to this render's own run folder, and say so.

    One render is one run of the renderer, and two renders into one
    ``--out`` used to drop their pictures among each other -- same
    directory, same filenames for the same product and valid time, the
    second silently overwriting the first.  The stamped level fixes
    that above the layout: ``<domain>/<product>/<valid-day>`` is
    untouched underneath it.

    ``args.out`` is REBOUND rather than threaded through as a second
    value because every path this door writes -- the pictures, the pair
    sheets, the per-file scratch store, the closing summary line -- is
    computed from it.  Two names for the output directory is how one of
    them ends up being the one nobody updated.

    The initialisation time comes from the FIRST INPUT THAT PROVES ONE
    (``SIMULATION_START_DATE``), and is omitted when none does; the
    ``--pair`` route passes no inputs and takes a launch-only stamp,
    because a comparison sheet belongs to no single run's cycle.

    A freshly allocated folder is NOT yet published: ``latest-run.txt``
    stays where it was until :func:`_publish_run_dir` sees at least one
    PNG in the folder, and a render that produced none removes the
    folder again.  A failed render used to leave an empty stamped
    folder with the pointer naming it, so a script following the
    documented pointer could not tell a failed render from a successful
    empty one (UX finding N5).
    """

    out = Path(args.out)
    enabled = run_stamp.run_stamp_enabled(args)
    if enabled and run_stamp.is_run_folder(out):
        # The caller named the run folder.  A second stamp level inside
        # it would bury the path they typed, so it is honoured verbatim
        # -- and said out loud, because "I asked for a stamp and got no
        # new folder" is otherwise a silent surprise.
        print(f"render: run folder {out} (named on the command line; "
              f"drawing into it rather than stamping inside it)")
    run_dir = run_stamp.resolve(
        out, init=run_stamp.first_wrfout_init(inputs), enabled=enabled,
        publish=False)
    if enabled and run_dir != out:
        print(f"render: run folder {run_stamp.relative_to_case(run_dir, out)}"
              f" under {out}")
    args.out = run_dir
    # What _publish_run_dir needs on every exit path: whether THIS
    # invocation allocated a fresh folder, and under which case root.
    args._run_claim_created = run_dir != out
    args._run_claim_root = out
    return run_dir


def _publish_run_dir(args: argparse.Namespace) -> None:
    """Publish or retract this render's run folder by what it produced.

    Called on EVERY exit path after :func:`_claim_run_dir` -- the
    normal return and both exception returns -- and a no-op when this
    invocation allocated no fresh folder (``--run-stamp off``, or an
    ``--out`` already inside a run).

    At least one PNG under the folder publishes it: ``latest-run.txt``
    moves to name it, exactly as before.  No PNG at all retracts it:
    the folder is removed and the pointer is left untouched, so a
    script reading the pointer always lands on the newest run that
    actually drew something, and a failed render leaves no new folder
    to mistake for output.
    """

    if not getattr(args, "_run_claim_created", False):
        return
    args._run_claim_created = False
    run_dir = Path(args.out)
    case_root = Path(args._run_claim_root)
    if any(run_dir.rglob("*.png")):
        run_stamp.record_latest(case_root, run_dir)
        return
    import shutil

    try:
        shutil.rmtree(run_dir)
    except OSError:
        # A folder that cannot be removed is left; it still was not
        # published, which is the half a pointer-following script needs.
        return
    print("render: no images were produced, so run folder "
          f"{run_stamp.relative_to_case(run_dir, case_root)} was not "
          f"published (removed; {run_stamp.LATEST_POINTER} untouched)",
          file=sys.stderr)


#: The file ``--inputs-from`` reads: the frames to draw and the frames
#: that only supply accumulation baselines, one path per entry.
RENDER_INPUTS_SCHEMA = "gpuwm.render-inputs/v1"


def write_render_inputs(path: Path, frames, context_frames=()) -> Path:
    """Write the frame lists a render stage hands ``woof render``.

    A forecast's whole series used to travel as arguments, one path per
    frame.  Windows starts no process whose command line passes 32,767
    characters, and 48 hours of hourly parent frames plus a 15 minute
    nest under an ordinary Documents folder came to 36,519: the forecast
    finished and its pictures were never drawn.  A file has no such
    ceiling, and every frame of the series still reaches the renderer,
    which is what an accumulation window needs.
    """

    record = {"schema": RENDER_INPUTS_SCHEMA,
              "wrfout": [str(frame) for frame in frames],
              "context_wrfout": [str(frame) for frame in context_frames]}
    Path(path).write_text(json.dumps(record, indent=1), encoding="utf-8")
    return Path(path)


def _fold_render_inputs(args: argparse.Namespace) -> str | None:
    """Add ``--inputs-from``'s frames to the command's; the refusal, or None.

    Done once: ``woof.cli`` folds the file in before its provenance gate
    and capability preflight, so their refusal lines name the frames too,
    and ``inputs_from`` is cleared so :func:`render_main` does not add the
    same frames a second time.
    """

    source = getattr(args, "inputs_from", None)
    if source is None:
        return _expand_render_folders(args)
    try:
        record = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        reason = getattr(error, "strerror", None) or str(error)
        return f"the frame list {source} could not be read ({reason})"
    if not isinstance(record, dict) or record.get("schema") != RENDER_INPUTS_SCHEMA:
        return (f"{source} is not a frame list this version reads "
                f"(it needs \"schema\": \"{RENDER_INPUTS_SCHEMA}\")")
    lists = {}
    for key in ("wrfout", "context_wrfout"):
        value = record.get(key, [])
        if not isinstance(value, list) or not all(
                isinstance(item, str) and item for item in value):
            return f"{source}: \"{key}\" must be a list of file paths"
        lists[key] = [Path(item) for item in value]
    args.wrfout = [*(args.wrfout or []), *lists["wrfout"]]
    args.context_wrfout = [*(getattr(args, "context_wrfout", None) or []),
                           *lists["context_wrfout"]]
    args.inputs_from = None
    _respell_invocation(lists)
    return _expand_render_folders(args)


def _expand_render_folders(args) -> str | None:
    from woof.render_compare import expand_inputs

    try:
        args.wrfout = expand_inputs(getattr(args, "wrfout", ()) or ())
        args.context_wrfout = expand_inputs(getattr(args, "context_wrfout", ()) or ())
    except (OSError, ValueError) as error:
        return str(error)
    return None


def _respell_invocation(lists: dict) -> None:
    """The "run ... --explain" line names the frames, when that line fits.

    The render stage removes its frame file when the stage succeeds, so a
    pointer that repeated ``--inputs-from`` would send the reader to a
    file that is gone.  It names the frames themselves, as the stage's
    command did before the file existed.

    Unless the frames spelled out make a line past the command-line
    budget (:data:`woof.rustwx.COMMAND_LINE_BUDGET`), which is the very
    series the file exists for.  Spelled out, 242 frames were a 48,000
    character line: no Windows shell runs it, and as the last line of a
    refusal it filled the diagnostic the desktop shows and pushed the
    refusal itself out.  Such a line keeps ``--inputs-from`` and the
    file, which a failed stage leaves in place for it
    (:func:`woof.go_cli.render_inputs_file`).
    """

    from woof import explain
    from woof.rustwx import COMMAND_LINE_BUDGET

    typed = explain.invocation()
    if typed is None:
        return
    frames = [str(path) for path in lists["wrfout"]]
    spelled: list[str] = []
    skip = False
    for token in typed:
        if skip:
            skip = False
        elif token == "--inputs-from":
            skip = True
            spelled += frames
        elif token.startswith("--inputs-from="):
            spelled += frames
        else:
            spelled.append(token)
    for path in lists["context_wrfout"]:
        spelled += ["--context-wrfout", str(path)]
    explain.set_invocation(spelled)
    pasted = explain.reinvocation() or ""
    if len(f"{pasted} --explain") > COMMAND_LINE_BUDGET:
        explain.set_invocation(typed)


def _cf_main(args: argparse.Namespace) -> int:
    """CF diagnostics use the Rust reader and the same production panel path."""
    from woof import rustwx

    if args.wrfout or not args.cf_variable:
        print("render: --cf needs --variable NAME and takes its input from --cf FILE.nc", file=sys.stderr)
        return 2
    try:
        size = parse_size(args.size)
        return rustwx.run_cf_renderer(
            require_renderer(), args.cf_file, variable=args.cf_variable,
            out_dir=args.out, valid=args.cf_valid, bounds=args.cf_bounds,
            table=args.cf_table, contours=args.cf_contours,
            contour_file=args.cf_contour_file, observation=args.cf_observation,
            observation_variable=args.cf_observation_variable,
            overlays=args.overlays, size=size, receipt=args.cf_receipt,
            theme=args.theme)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"render: {error}", file=sys.stderr)
        return 2


def render_main(args: argparse.Namespace) -> int:
    # Which tree is drawing these plots.  Idempotent -- `woof.cli.main`
    # has normally already announced -- and here as well because this
    # handler is reachable without that front door.
    from woof.provenance_gate import announce

    announce("woof render")
    if getattr(args, "cf_file", None):
        return _cf_main(args)
    refusal = _fold_render_inputs(args)
    if refusal is not None:
        print(f"render: {refusal}", file=sys.stderr)
        return 2
    try:
        from woof.hubwind import render as render_hubwind
        hubwind_result = render_hubwind(args)
        if hubwind_result is not None:
            return hubwind_result
    except (ValueError, OSError, RuntimeError) as error:
        print(f"render: {error}", file=sys.stderr)
        return 2
    comparisons = [name for name in ("pair", "compare", "diff")
                   if getattr(args, name, None)]
    if len(comparisons) > 1:
        print("render: " + ", ".join("--" + name for name in comparisons)
              + " name separate comparison inputs; choose one comparison",
              file=sys.stderr)
        return 2
    if getattr(args, "compare", None):
        # A run beside a reference model's own fields: its own engine
        # (rw_compare), its own product names and its own inputs (frames
        # or a run folder), so it leaves this handler before any of the
        # single-run engine resolution below.
        if args.pair:
            print("render: --compare draws from history frames and --pair "
                  "composes already-rendered PNG directories; they do not "
                  "combine", file=sys.stderr)
            return 2
        from woof.render_compare import compare_main

        return compare_main(args)
    if args.pair:
        if args.wrfout:
            print("render: --pair composes already-rendered PNG "
                  "directories; wrfout arguments do not combine with it",
                  file=sys.stderr)
            return 2
        return _pair_main(args)
    if getattr(args, "diff", None):
        if args.wrfout:
            print("render: --diff takes the two runs as A_RUN B_RUN; wrfout "
                  "arguments do not combine with it", file=sys.stderr)
            return 2
        return _diff_main(args)
    if not args.wrfout and args.list_products:
        # "What may I put in --products?" is a question about this build,
        # not about a file, and a forecaster who has not run anything yet
        # is exactly who asks it.  Demanding a wrfout first is why the
        # only way to discover a product slug was to read the source.
        # The renderer already answers it (`rw_wrfbatch --list-products`
        # with no inputs prints its vocabulary), so this asks rather than
        # keeping a second copy of the catalog here to go stale.
        return catalog_main(args)
    if not args.wrfout:
        print("render: at least one WRFOUT file is required "
              "(or --list-products alone for the product catalog, or "
              "--pair A_DIR B_DIR)", file=sys.stderr)
        return 2
    try:
        timeidx = parse_timeidx(args.timeidx)
        size = parse_size(args.size)
        engine, why = _resolve_engine(args.engine)
        if engine == "rust":
            rust_products = parse_products_rust(args.products)
        else:
            products = parse_products(args.products)
        if getattr(args, "series", False) and engine != "rust":
            raise ValueError("--series uses the native Rust timeline renderer")
        if getattr(args, "context_wrfout", ()) and not getattr(args, "series", False):
            raise ValueError("context history needs --series")
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        # Through the layering boundary, not raw: the bridge refusal
        # `_resolve_engine` can raise is composed with `explain.layered`,
        # and printing it unrendered would put the [[explain]] sentinel
        # on a terminal.
        print("render: " + explain.render(
            str(exc), explain=explain.explain_enabled(args),
            command="woof render"), file=sys.stderr)
        return 2
    # BEFORE the fallback advisory, and before any wrfout is opened: an
    # engine that cannot draw is refused with ITS OWN remedy first.
    # `--list-products` is exempt because the matplotlib catalog needs
    # only netCDF4 and is a working path on a bare install.
    if not args.list_products:
        refusal = engine_refusal(engine, args.engine, why)
        if refusal is not None:
            print(explain.render(
                refusal, explain=explain.explain_enabled(args),
                command="woof render"), file=sys.stderr)
            return 2
    notice = matplotlib_workaround_notice(engine)
    if notice is not None:
        print(notice, file=sys.stderr)
    if engine == "rust":
        from woof import rustwx

        blank = missing_basemap_notice(rustwx.find_renderer())
        if blank is not None:
            print(blank, file=sys.stderr)
    if args.list_products:
        return list_products_main(args, engine)
    # THE [output] PRE-CHECK, before a run directory is claimed and
    # before a wrfout is opened for rendering.  A run that trimmed its
    # history says so in its own files, so a request for a product whose
    # input that run dropped is answerable HERE, by name, with the way
    # back -- instead of the bare "not in file" this used to become,
    # which reads like a damaged artifact rather than the configuration
    # choice it is.  Exempt: --list-products (it exists to answer this),
    # and --products all (a request for whatever the file supports).
    from woof.rustwx import product_spec_terms
    requested = product_spec_terms(args.products)
    history_refusal = history_selection_refusal(args.wrfout, requested)
    if history_refusal is not None:
        print(explain.render(
            history_refusal, explain=explain.explain_enabled(args),
            command="woof render"), file=sys.stderr)
        return 2
    history_note = history_selection_notice(args.wrfout, requested)
    if history_note is not None:
        print(explain.render(
            history_note, explain=explain.explain_enabled(args),
            command="woof render"), file=sys.stderr)
    # THE STORELESS FAMILIES, answered here rather than by the renderer.
    # `mesh:` and `meshdiff:` are drawn from a mesh file's cell
    # boundaries and this door passes none; `xsec:` is cut along a line
    # and may have none.  The renderer refuses any of the three for the
    # WHOLE invocation before it draws anything, so a term forwarded
    # from here used to cost every other requested product its pictures:
    # `--products composite_reflectivity,mesh:cell_area` exited 1 with
    # no pictures where it could have drawn every reflectivity frame.
    # Dropped per PRODUCT, which is what the renderer will not do, and
    # named -- and when the drop leaves nothing, refused here, before a
    # run directory is claimed.
    door_skips: list[tuple[str, str]] = []
    requested_spec = rust_products if engine == "rust" else None
    if engine == "rust":
        from woof import rustwx

        rust_products, door_skips = rustwx.drop_storeless_terms(
            rust_products, section=args.section)
        if not rust_products:
            print("render: " + explain.render(explain.layered(
                ", ".join(term for term, _reason in door_skips)
                + ": this render has nothing left to draw.\n"
                + "\n".join(f"  {term}: {reason}"
                            for term, reason in door_skips),
                "These families do not come from the history store, so the "
                "store's own catalog cannot decide them and the renderer "
                "refuses the whole invocation rather than skipping the "
                "term.  Asked for beside products that ARE drawable, they "
                "are dropped and the rest is drawn; asked for alone, there "
                "is nothing to draw."),
                explain=explain.explain_enabled(args),
                command="woof render"), file=sys.stderr)
            return 2
        # `reason`, never `why`: the name beside it holds how the ENGINE
        # was resolved and is printed one line below.  Reusing it here
        # made the engine line quote a dropped product's sentence as the
        # basis on which the renderer had been chosen.
        for term, reason in door_skips:
            print(f"render: note: {term} is dropped from this render and "
                  f"the other requested products are still drawn. {reason}.",
                  file=sys.stderr)
    print(f"render: engine {engine} ({why})")
    grib2_exe = None
    if getattr(args, "grib2_out", None) is not None:
        from woof.grib2_live import find_exporter
        try:
            grib2_exe = find_exporter()
        except (RuntimeError, FileNotFoundError) as error:
            print(f"render: GRIB2 exporter: {error}", file=sys.stderr)
            return 3
    # AFTER every refusal and after --list-products: this creates a
    # directory, and a command that draws nothing must leave none --
    # which is also why every exit path below runs _publish_run_dir.
    _claim_run_dir(args, args.wrfout)
    try:
        # WHERE the pictures will be, printed BEFORE they are drawn, so
        # a script driving this can compute the path it is going to
        # watch rather than glob for it afterwards.
        # One separator on both arms: the reader's own --out arrives
        # with this platform's, and gluing a forward-slash template
        # behind it printed a path in two spellings (UX finding N24).
        # An input filed under a lifecycle episode gets one more
        # segment, and the sentence has to say so: a script handed the
        # three-segment template would watch a directory this render
        # never writes to.  Asked of the INPUTS, which are known here,
        # rather than assumed.
        episodic = any(history_episode(path) is not None
                       for path in args.wrfout)
        print(f"render: layout {args.layout} -- "
              + (render_layout.describe(str(args.out), episode=episodic)
                 if args.layout == render_layout.NESTED
                 else f"{args.out}{os.sep}<file>.png (legacy flat)"))
        if engine == "rust":
            # The receipt line for WHICH engine binary drew these
            # products.  Recorded on the passing path too: "the in-tree
            # engine ran" is the fact that was missing when a foreign
            # engine's 153 products had to be explained after the fact.
            from woof import rustwx
            from woof.provenance_gate import bridge_tree_match

            # The radar colour set reaches the engine by its environment
            # name, which every renderer subprocess inherits
            # (rustwx.renderer_env): one switch for every radar-table
            # product, no per-product argument.
            if getattr(args, "radar_colors", None):
                os.environ[rustwx.RADAR_COLORS_ENV] = args.radar_colors

            verdict = bridge_tree_match(rustwx.find_renderer(),
                                        env_var=rustwx.RENDERER_ENV)
            # Filled by the renderer's own event stream: the range each
            # vertical cut was drawn over.  It reaches the receipt below.
            section_fills: list = []
            print(f"render: engine bridge {verdict.verdict} "
                  f"({verdict.basis})")
            try:
                written, failures, skipped = render_wrfouts_rust(
                    args.wrfout, fills=section_fills,
                    products=rust_products, timeidx=timeidx,
                    outdir=args.out, size=size, heavy=args.heavy,
                    source_label=args.source_label, layout=args.layout,
                    overlays=args.overlays, annotate=args.annotate,
                    streamlines=args.streamlines, theme=args.theme,
                    section=args.section, isotherms=args.isotherms,
                    section_across_km=args.section_across_km,
                    section_size=args.section_size,
                    section_top_km=args.section_top_km,
                    series=getattr(args, "series", False),
                    context_paths=getattr(args, "context_wrfout", ()))
            except (RuntimeError, ValueError) as exc:
                print("render: " + explain.render(
                    str(exc), explain=explain.explain_enabled(args),
                    command="woof render"), file=sys.stderr)
                return 2
        else:
            written, failures, skipped = render_wrfouts(
                args.wrfout, products=products, timeidx=timeidx,
                outdir=args.out, dpi=args.dpi,
                source_label=args.source_label, layout=args.layout)
        from woof.render_receipts import publish_invocation
        # How tall the cuts were: the ceiling the door forwarded, or the
        # engine's own when a section was drawn and none was named.  A
        # receipt that records the products but not the height axis
        # cannot explain two different pictures of one line.  Read off
        # the spec that REACHES the renderer, so a section term the door
        # dropped for want of a line records no ceiling.
        section_top_km = None
        if engine == "rust" and rustwx.split_section_spec(rust_products)[1]:
            section_top_km = (args.section_top_km
                              if args.section_top_km is not None
                              else rustwx.SECTION_TOP_KM_DEFAULT)
        # WHAT WAS ASKED FOR, not what survived the door: the receipt's
        # request and its skips are read together, and a request the
        # door had already trimmed would name no term to match the skip
        # against.
        render_summary = publish_invocation(root=args.out, engine=engine,
            requested_spec=requested_spec if engine == "rust" else ",".join(products),
            written=written, failures=failures, skipped=door_skips + skipped,
            layout=args.layout,
            inputs=args.wrfout, context_inputs=getattr(args, "context_wrfout", ()),
            section_top_km=section_top_km,
            section_fills=section_fills if engine == "rust" else ())
        for failure in failures:
            print(f"render FAIL: {failure}", file=sys.stderr)
        from woof.render_receipts import drawn_families
        notice = skip_notice(skipped, wrote_any=bool(written),
                             drawn=drawn_families(args.out, written,
                                                  args.layout),
                             sources=(*args.wrfout,
                                      *getattr(args, "context_wrfout", ())))
        if notice is not None:
            print(explain.render(
                notice, explain=explain.explain_enabled(args),
                command="woof render"), file=sys.stderr)
        print(f"render: {len(written)} file(s) -> {args.out}")
        print(f"render: result receipt -> {render_summary['summary_path']}")
    finally:
        _publish_run_dir(args)
    # Written-and-no-failures, where a SKIP IS NOT A FAILURE.  The two
    # were one list until 1.4.1 and the exit code could not tell them
    # apart, so a registered absence -- the cold-start frame's missing
    # REFL_10CM -- returned 1 from a render that had drawn every product
    # the file could support, and `woof go` stopped the chain on a
    # forecast that had succeeded.
    #
    # The two arms that still return 1 are the ones that mean something
    # went wrong: any failure at all, and a render that produced no
    # image.  The second is what keeps a skip from becoming a silent
    # success -- ask for one product, have its inputs be absent, and the
    # answer is still a nonzero exit, because nothing was drawn.
    if grib2_exe is not None:
        from woof import grib2_export
        from woof.grib2_live import request_for
        code = grib2_export.invoke(grib2_exe, request_for(args.wrfout, args.grib2_out))
        if code:
            return code
    return 0 if not failures and (written or not_applicable_skips(door_skips + skipped)) else 1


def _section_size(value: str) -> tuple[int, int]:
    """``WxH`` for ``--section-size``, refused by name when it is not."""

    parts = re.split("[xX]", value.strip())
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--section-size '{value}' is not WxH, e.g. 2400x1200")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--section-size '{value}' is not WxH in whole pixels") from None
    for name, side in (("width", width), ("height", height)):
        if not 200 <= side <= 12_000:
            raise argparse.ArgumentTypeError(
                f"--section-size {name} {side} is not 200-12000 pixels")
    return width, height


def _section_top_km(value: str) -> float:
    """``N`` for ``--section-top-km``, refused in the engine's sentence.

    The engine owns the range (``rw_wrfbatch`` takes 1-40 km), so the
    door repeats the engine's own refusal rather than inventing a second
    wording for the same rule -- and repeats it HERE, before a render
    launches, instead of after the renderer has opened the frames.
    """

    from woof import rustwx

    problem = rustwx.section_top_problem(value)
    if problem is not None:
        raise argparse.ArgumentTypeError(problem)
    return float(value)


def _section_across_km(value: str) -> float:
    """``KM`` for ``--section-across``, refused in the engine's sentence."""

    from woof import rustwx

    problem = rustwx.section_across_problem(value)
    if problem is not None:
        raise argparse.ArgumentTypeError(problem)
    return float(value)


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "render",
        help="render forecast product PNGs from wrfout files "
             "(composite reflectivity, 2 m temperature, 10 m wind, "
             "accumulated precipitation, TOA outgoing longwave "
             "radiation in W m-2)")
    parser.add_argument(
        "wrfout", type=Path, nargs="*", metavar="WRFOUT",
        help="wrfout NetCDF files, a wrfout directory, or the output folder printed by woof go")
    parser.add_argument("--cf", dest="cf_file", type=Path, metavar="FILE.nc",
                        help="render a native CF diagnostic grid with file source provenance")
    parser.add_argument("--variable", dest="cf_variable", metavar="NAME",
                        help="CF field selected by variable name; product and units come from the CF table")
    parser.add_argument("--valid", dest="cf_valid", metavar="UTC",
                        help="CF valid time in ISO UTC (default: first record)")
    parser.add_argument("--bounds", dest="cf_bounds", metavar="W,E,S,N",
                        help="CF geographic extent in degrees, e.g. --bounds=-125,-66,24,50")
    parser.add_argument("--cf-table", type=Path, metavar="FILE.json",
                        help="CF variable product table (default: packaged production rows)")
    parser.add_argument("--contours", dest="cf_contours", metavar="NAME[,NAME]",
                        help="CF probability contours at each table row's contour_level")
    parser.add_argument("--contour-file", dest="cf_contour_file", type=Path, metavar="FILE.nc",
                        help="CF contour companion on the fill grid (default: --cf file)")
    parser.add_argument("--observation", dest="cf_observation", type=Path, metavar="FILE",
                        help="CF observation grid or MRMS GRIB2/gz beside the diagnostic on one ladder and extent")
    parser.add_argument("--observation-variable", dest="cf_observation_variable", metavar="NAME",
                        help="variable in a CF observation file (default: --variable)")
    parser.add_argument("--receipt", dest="cf_receipt", type=Path, metavar="FILE.json",
                        help="CF input, output and per-stage timing receipt")
    parser.add_argument("--grib2-out", type=Path, metavar="DIR",
                        help="also export every input frame as native surface and pressure-level GRIB2")
    parser.add_argument(
        "--engine", choices=("auto", "rust", "matplotlib"),
        default="auto",
        help="render engine: the vendored Rusty Weather renderer "
             "(campaign plot quality; 151 implicit-render catalog "
             "candidates per file) or the matplotlib workaround; 'auto' "
             "(default) uses rust whenever its binary is built and "
             "probes as runnable, and REFUSES otherwise rather than "
             "drawing weather fields with matplotlib -- 'matplotlib' "
             "asks for that workaround by name and announces itself")
    parser.add_argument("--hubwind-config", type=Path, default=None, metavar="JSON|TOML",
                        help="hub heights, turbine, exceedance and ramp settings; JSON options or [hubwind] TOML")
    parser.add_argument(
        "--products", default="all", metavar="LIST",
        help="comma-separated products: "
             f"{', '.join(PRODUCTS)}, or 'all' (default); with the rust "
             "engine these are "
             + ", ".join(f"{name}={slug}" for name, slug
                         in RUST_PRODUCT_ALIASES.items())
             + ", raw catalog slugs (sbcape, srh_0_1km, ...) also "
             "work and 'all' renders its full catalog")
    parser.add_argument(
        "--timeidx", default="all", metavar="N|all",
        help="frame index within each file (within each timeline with --series), or 'all' (default)")
    parser.add_argument(
        "--series", action="store_true",
        help="render compatible files from each run/domain/episode as one timeline, including multi-hour products")
    parser.add_argument(
        "--context-wrfout", action="append", type=Path, default=[], metavar="FILE",
        help=argparse.SUPPRESS)
    # The render stage's own hand-off: a long series in a file rather than
    # on a command line (see write_render_inputs).
    parser.add_argument(
        "--inputs-from", dest="inputs_from", type=Path, default=None,
        metavar="FILE", help=argparse.SUPPRESS)
    parser.add_argument(
        "--out", type=Path, default=Path("out/render"), metavar="DIR",
        help="where the PNGs go (default out/render).  Each render "
             "claims its own timestamped run folder under it, so two "
             "renders never overwrite each other; point it at an "
             "existing run-... folder to draw into that one")
    parser.add_argument(
        "--layout", choices=render_layout.LAYOUTS,
        default=render_layout.DEFAULT_LAYOUT,
        help="how the PNGs are arranged inside this render's run "
             "folder: 'nested' (default) files each picture at "
             f"{render_layout.describe('<run folder>', sep='/')}, "
             "so a run's thousands of frames are separated by nest, by "
             "chart and by day and a script can predict a path without "
             "globbing; 'flat' writes every picture directly into the "
             "run folder, which is what releases up to 2.4.1 did and is "
             "kept only for consumers still written against it (with "
             "--run-stamp off it is the v2.4.1 tree exactly)")
    run_stamp.add_argument(parser, artifacts="PNGs")
    parser.add_argument(
        "--dpi", type=positive_int, default=150, metavar="N",
        help="PNG resolution, matplotlib engine (default 150)")
    parser.add_argument(
        "--size", default="auto", metavar="WxH|auto",
        help="output pixels, rust engine; 'auto' (the default) sizes each "
             "canvas from its domain's shape, WxH draws a fixed canvas")
    parser.add_argument(
        # None, not the string: the default is resolved at render time
        # by `default_source_label()`, which asks provenance which tree
        # is executing.  Computing it here would put a git subprocess in
        # `woof --help`.
        "--source-label", default=None, metavar="TEXT",
        help="model/provenance label stamped on every plot (default "
             f"'{DEFAULT_SOURCE_LABEL} <the executing version>'); set it "
             "when rendering wrfout files this model did not produce, so "
             "the sheet does not claim them")
    parser.add_argument(
        "--heavy", action="store_true",
        help="rust engine: also compute the heavy ECAPE product family "
             "at import (SBECAPE/SBNCAPE/SBECIN, ECAPE SCP/EHI/...; "
             "adds substantial per-frame import time)")
    wind = parser.add_mutually_exclusive_group()
    wind.add_argument(
        "--streamlines", dest="streamlines", action="store_true",
        default=None,
        help="rust engine: draw the wind as STREAMLINES instead of "
             "barbs on every product that carries a wind layer.  "
             "Without either flag the engine keeps its automatic "
             "choice (streamlines on curvilinear and projected grids, "
             "barbs on plain lat/lon), and the RUSTWX_WIND_STREAMLINES "
             "environment variable still works; this flag and --barbs "
             "outrank it")
    wind.add_argument(
        "--barbs", dest="streamlines", action="store_false",
        default=None,
        help="rust engine: draw the wind as BARBS, overruling both the "
             "automatic choice and any inherited "
             "RUSTWX_WIND_STREAMLINES")
    parser.add_argument(
        "--overlays", type=Path, metavar="FILE.json",
        help="rust engine: draw map overlays given in geographic DEGREES "
             "on every panel -- lines, closed boxes, markers, labels and "
             "range rings.  This is the seam a boundary-zone frame, tile "
             "seams, storm-report markers and radar sites needed; the "
             "schema is documented in "
             "tools/rustwx/crates/rustwx-products/src/"
             "geographic_overlays.rs.  Omitted, the renderer runs no "
             "overlay code and the PNGs are byte-identical")
    parser.add_argument(
        "--annotate", type=Path, metavar="FILE.json",
        help="rust engine: override the panel title and the three "
             "subtitle slots (title, title_suffix, subtitle_left, "
             "subtitle_center, subtitle_right).  A short badge belongs "
             "in the centre slot; anything sentence-length belongs on "
             "the left, which owns the row's width")
    parser.add_argument(
        "--theme", metavar="NAME|FILE.json", default=None,
        help="rust engine: the render theme -- a built-in name (default, "
             "dark) or a JSON theme file naming the surface, the inks, "
             "the basemap linework, the colorbar chrome, the fonts and "
             "the colormap overrides (schema in tools/rustwx/crates/"
             "rustwx-render/src/theme.rs; RUSTWX_THEME is the "
             "environment spelling).  Omitted, the engine draws its own "
             "look and the PNGs are byte-identical")
    parser.add_argument(
        "--radar-colors", choices=("standard", "classic"), default=None,
        help="rust engine: the colour tables the reflectivity and radial "
             "velocity products draw with -- standard (the radar tables, "
             "the default) or classic (the reflectivity ladder and "
             "blue-red velocity scale before 2.8.5).  One name selects "
             "every radar-table product; RUSTWX_RADAR_COLORS is the "
             "environment spelling, which `woof go` and `woof run` "
             "renders also read")
    parser.add_argument(
        "--section", metavar="lat,lon,lat,lon|FILE.json", default=None,
        help="rust engine: the line the vertical-section products "
             "(xsec:<fill>[/<overlay>...] in --products, any 3-D wrfout "
             "field on a height axis) are cut along; a JSON file gives "
             "{start, end} or a {points, extend_km} polyline")
    parser.add_argument(
        "--isotherms", metavar="L,L,...[@H]", default=None,
        help="rust engine: the isotherms (C) drawn on every section, "
             "with an optional highlighted one after '@' "
             "(e.g. 0,-5,-10,-15,-20@-10); default 0,-10,-20,-30,-40")
    parser.add_argument(
        "--section-size", dest="section_size", type=_section_size,
        metavar="WxH",
        help="the size a cross-section is drawn at; absent, a section is "
             "landscape 2:1 at the map's width, because a vertical cut "
             "handed the map's own size comes out portrait")
    parser.add_argument(
        "--section-across", dest="section_across_km", type=_section_across_km,
        metavar="KM", default=None,
        help="rust engine: also draw each section product across the "
             "line, this many km long, through the fill's maximum column")
    parser.add_argument(
        "--section-top-km", dest="section_top_km", type=_section_top_km,
        metavar="N", default=None,
        help="rust engine: the ceiling of a section's height axis, 1-40 "
             "km; absent, the engine fits up to 14 km, which draws a "
             "shallow feature in the bottom fourteenth of the frame -- "
             "give it 3 for a boundary-layer cut")
    parser.add_argument(
        "--list-products", action="store_true",
        help="list the engine's product catalog with per-file "
             "availability (why each product is or is not renderable "
             "from this wrfout) instead of rendering")
    parser.add_argument(
        "--pair", nargs=2, metavar=("A_DIR", "B_DIR"), type=Path,
        help="compose two runs' rendered PNG directories into labeled "
             "side-by-side comparison sheets (no wrfout arguments)")
    parser.add_argument(
        "--pair-title", default="Paired comparison", metavar="TITLE",
        help="pair-sheet title (default 'Paired comparison')")
    parser.add_argument(
        "--pair-subtitle", default="", metavar="TEXT",
        help="optional pair-sheet subtitle")
    parser.add_argument(
        "--pair-labels", nargs=2, metavar=("LEFT", "RIGHT"),
        help="panel labels (default: the two directory names)")
    from woof.render_compare import register_arguments

    register_arguments(parser)
    parser.add_argument(
        "--diff", nargs=2, metavar=("A_RUN", "B_RUN"), type=Path,
        help="draw each product as run A minus run B: two folders of wrfout "
             "frames (or two files), paired by valid time; refused by name "
             "when the runs do not share a grid (rust engine; no wrfout "
             "arguments)")
    parser.add_argument(
        "--diff-labels", nargs=2, metavar=("A_NAME", "B_NAME"),
        help="the two runs' names on the difference panels (default: the "
             "two folder names)")
    parser.add_argument(
        "--diff-sheet", action="store_true",
        help="also draw an A | B | A minus B sheet per product")
    parser.add_argument(
        "--diff-table", metavar="{table,auto,FILE}", default=None,
        help="the difference colour bars: table (default: the fixed "
             "per-product bars, so the frames of a loop share one), auto "
             "(every bar fitted to its own difference) or a JSON file whose "
             "keys replace the built-in table's (schema in tools/rustwx/"
             "crates/rustwx-render/data/difference.json; "
             "RUSTWX_DIFFERENCE_TABLE is the environment spelling).  Under "
             "every choice a fixed bar that would draw a nonzero difference "
             "as blank is fitted to it, and every panel prints the largest "
             "|A-B| and how many cells differ"),
    parser.set_defaults(func=render_main)
    return parser


__all__ = ["DEFAULT_SOURCE_LABEL", "PRODUCTS", "RUST_PRODUCT_ALIASES",
           "WRF_PACKAGE_REQUIREMENT", "default_source_label",
           "domain_token", "list_products_main",
           "matplotlib_workaround_notice",
           "parse_products", "parse_products_rust", "parse_size",
           "parse_timeidx", "plot_context", "register_cli", "render_main",
           "BASEMAP_MISSING_CODE", "announce_missing_basemap",
           "basemap_gap", "renderer_basemap_gap",
           "missing_basemap_notice", "missing_declared_inputs",
           "render_series_rust", "render_wrfouts", "render_wrfouts_rust",
           "require_renderer", "resolution_token", "skip_notice",
           "spacing_label"]
