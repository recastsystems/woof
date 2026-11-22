"""Figures from a streamed big-domain forecast.

DEPRECATED FALLBACK, under the project render law ( 2026-08-06):
weather-field product plots come from the real Rust renderer
``rw_wrfbatch`` driven through :mod:`woof.rustwx`.  This module's
matplotlib panels are the fallback, reachable only with
``--engine matplotlib``, and they are not the product tier.

The premise this file was written on is gone.  It said the production
renderer "cannot be pointed at a pinned host store, so it is not the tool
for a domain that never becomes a wrfout" -- and the domain DOES become a
wrfout now: ``tilestream/run_bigdomain.py`` writes one beside every
``.npz`` by default (``woof.io.surface_wrfout``), and ``--engine rust``
here materialises one from any older ``.npz`` that predates that.
MEASURED 2026-08-17 on a 256x256x33 streamed forecast: 20 production
products per frame through the real executable, the reflectivity on the
NWS ladder the panels below imitate.

What the Rust route does NOT yet draw, and why it is not a silent second
tier: the SIX-PANEL SHEET, the evolution STRIP, the zoom crop and the
tile-seam overlay are multi-panel COMPOSITION, and ``MapRenderRequest``
composes one panel.  Composition of finished PNGs is
:mod:`woof.pair_compose`'s job; until those figures are built that way
they stay here, and they are the only reason this module still runs.

Every colour convention below is imported from :mod:`woof.render`
rather than reinvented:

* ``_NWS_COLORS`` / ``_NWS_LEVELS`` -- the standard NWS 5 dBZ reflectivity
  ladder, with ``set_under("none")`` so sub-5 dBZ is transparent instead of
  a wash of blue over a clear domain;
* ``_PRECIP_LEVELS`` -- the NWS-style QPF ladder, on ``YlGnBu``;
* ``spacing_label`` -- the ``Δx 3 km`` subtitle token;
* recessive axes (``#444444`` labels, ``#999999`` spines), per-panel
  colorbars LABELLED WITH UNITS, and lon/lat pcolormesh on the file's own
  curvilinear coordinates so the map projection shapes the frame.

Two things this deliberately does NOT do.

It does not stretch a colour scale to the data.  Reflectivity is on the fixed
NWS ladder and precipitation on the fixed QPF ladder, so a weak field looks
weak.  The scales that ARE data-dependent (2 m temperature, 10 m wind, column
w) print their own limits in the colourbar label.

It does not hide a zero.  ``panel_or_note`` writes "field is uniformly X"
across a panel whose field has no variation, because a blank panel and a
panel of genuinely constant data look identical and only one of them is a
result.  At t+0 of this configuration that is the accurate state of every
surface diagnostic: Noah-MP and MYNN have not been called yet, so ``T2``,
``U10``, ``PBLH`` and the rest are still zero, and the reflectivity field is
uniformly the -35 dBZ floor because the initial state carries no condensate.

THE CROP
--------
``--crop`` cells are removed from every edge before plotting.  The domain is
PERIODIC on a REAL Lambert grid, so the wrap is a geographic discontinuity --
latitude, Coriolis and the solar zenith all jump across it -- and the rows
next to it are responding to an artefact.  The crop is stated in the figure
footer, not applied quietly.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np


#: Every figure carries this.  It is the whole accuracy budget of the exercise
#: in one paragraph, and it belongs ON the picture rather than in a report
#: that gets separated from it.
IC_FOOTER = (
    "IDEALISED INITIAL CONDITION - NOT A FORECAST OF ANY REAL DAY. "
    "No GRIB and no WPS anywhere in this path. The state is the "
    "Weisman-Klemp 1982 analytic sounding and its quarter-circle hodograph "
    "(WRF em_quarter_ss), HORIZONTALLY UNIFORM - so there is no front, no "
    "jet, no dryline and no synoptic pattern - broken only by 0.1 K "
    "boundary-layer theta noise and 3 K / 30 km warm bubbles, WK82's own "
    "trigger. What IS real: the Lambert conformal projection, its map "
    "factors and Coriolis, the per-column solar geometry every radiation "
    "and land-surface path reads, the terrain, and the physics. "
    "Vapour is added to a dry-balanced column because WOOF-ArWen's balanced "
    "moist initialiser refuses a terrain-following base state, so the first "
    "~15 min is a hydrostatic adjustment and carries no echo.")

#: Width the title notes are wrapped to.  Tuned against the narrowest figure
#: here (the 12.5 inch hero) at 12 pt: wider and the text leaves the canvas.
WRAP_COLUMNS = 118

#: Set by ``--config-note``: the physics and tiling the run actually used.
#: The figures carry it because "streamed" is a claim about HOW the numbers
#: were produced, and a reader cannot check it from the weather.
CONFIG_NOTE: list = [""]


#: The measured resident ceiling, put on the figure by ``--ceiling-note``.
#: A picture whose whole point is "the card cannot hold this" should carry
#: the measurement that says so, not rely on a report travelling with it.
CEILING_NOTE: list = [""]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _axes_style(axis) -> None:
    """``woof.render._figure``'s recessive axes, applied to a shared figure."""
    axis.set_xlabel("longitude (degrees east)", fontsize=8, color="#444444")
    axis.set_ylabel("latitude (degrees north)", fontsize=8, color="#444444")
    axis.tick_params(labelsize=7, colors="#444444")
    for spine in axis.spines.values():
        spine.set_color("#999999")


def _suptitle(fig, text: str, *, floor: float) -> float:
    """Set the figure title and return the ``top`` margin it needs.

    The title is 2 lines bare and 4 with ``--config-note`` and
    ``--ceiling-note``, and a fixed margin is wrong for one of those: too
    small and the last line lands on the panel titles, too large and a short
    title floats in a band of white.  Measured off the line count and the
    figure height rather than tuned per figure.
    """
    fig.suptitle(text, fontsize=12)
    lines = text.count("\n") + 1
    height_in = fig.get_size_inches()[1]
    # 12 pt at 1.35 line spacing, plus a panel-title's worth of clearance.
    used = (lines * 12 * 1.35 + 16) / 72.0 / height_in
    return min(floor, 0.985 - used)


def _colorbar(fig, axis, mappable, label, ticks=None):
    cbar = fig.colorbar(mappable, ax=axis, shrink=0.86, pad=0.02, ticks=ticks)
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return cbar


def _flat(field) -> bool:
    finite = np.isfinite(field)
    if not finite.any():
        return True
    lo, hi = float(np.min(field[finite])), float(np.max(field[finite]))
    return hi - lo <= 1e-6 * max(1.0, abs(hi))


def panel_or_note(fig, axis, lon, lat, field, *, title, units, note_fmt,
                  **kw):
    """Draw the field, or say in words that there is nothing to draw.

    ``note_fmt`` is formatted with the constant value.  A panel that reads
    "uniformly 0.000 (physics has not been called yet)" is a result; a blank
    panel is a bug report waiting to happen.
    """
    _axes_style(axis)
    axis.set_title(title, fontsize=9)
    if _flat(field):
        value = float(np.nanmax(field)) if np.isfinite(field).any() else np.nan
        axis.text(0.5, 0.5, note_fmt.format(value=value), ha="center",
                  va="center", fontsize=9, color="#8a1c1c",
                  transform=axis.transAxes, wrap=True)
        axis.set_xlim(float(lon.min()), float(lon.max()))
        axis.set_ylim(float(lat.min()), float(lat.max()))
        return None
    mesh = axis.pcolormesh(lon, lat, field, shading="auto", **kw)
    _colorbar(fig, axis, mesh, units)
    return mesh


def refl_panel(fig, axis, lon, lat, refl, *, title):
    """Composite reflectivity on the NWS ladder, sub-5 dBZ transparent."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    from woof.render import _NWS_COLORS, _NWS_LEVELS

    _axes_style(axis)
    axis.set_title(title, fontsize=9)
    axis.set_facecolor("#f2f4f6")
    if float(np.nanmax(refl)) < _NWS_LEVELS[0]:
        axis.text(0.5, 0.5,
                  f"no echo anywhere: column-max REFL_10CM peaks at "
                  f"{float(np.nanmax(refl)):.1f} dBZ,\nbelow the 5 dBZ floor "
                  f"of the NWS scale.\nThe initial state carries no "
                  f"condensate; this is an accurate zero.",
                  ha="center", va="center", fontsize=9, color="#8a1c1c",
                  transform=axis.transAxes)
        axis.set_xlim(float(lon.min()), float(lon.max()))
        axis.set_ylim(float(lat.min()), float(lat.max()))
        return None
    cmap = ListedColormap(_NWS_COLORS)
    cmap.set_under("none")
    cmap.set_over(_NWS_COLORS[-1])
    mesh = axis.pcolormesh(lon, lat, refl, cmap=cmap,
                           norm=BoundaryNorm(_NWS_LEVELS, cmap.N),
                           shading="auto")
    _colorbar(fig, axis, mesh, "composite dBZ", ticks=_NWS_LEVELS[::2])
    return mesh


def precip_panel(fig, axis, lon, lat, total, *, title):
    import matplotlib
    from matplotlib.colors import BoundaryNorm

    from woof.render import _PRECIP_LEVELS

    _axes_style(axis)
    axis.set_title(title, fontsize=9)
    axis.set_facecolor("#f2f4f6")
    if float(np.nanmax(total)) < _PRECIP_LEVELS[0]:
        axis.text(0.5, 0.5,
                  f"no accumulation anywhere: RAINC + RAINNC peaks at "
                  f"{float(np.nanmax(total)):.3f} mm,\nbelow the 0.1 mm "
                  f"floor of the QPF scale.",
                  ha="center", va="center", fontsize=9, color="#8a1c1c",
                  transform=axis.transAxes)
        axis.set_xlim(float(lon.min()), float(lon.max()))
        axis.set_ylim(float(lat.min()), float(lat.max()))
        return None
    cmap = matplotlib.colormaps["YlGnBu"].resampled(len(_PRECIP_LEVELS) - 1)
    cmap.set_under("none")
    mesh = axis.pcolormesh(lon, lat, total, cmap=cmap,
                           norm=BoundaryNorm(_PRECIP_LEVELS, cmap.N),
                           shading="auto")
    _colorbar(fig, axis, mesh, "mm since forecast start",
              ticks=_PRECIP_LEVELS)
    return mesh


def wind_panel(fig, axis, lon, lat, u10, v10, *, title, barbs: int = 22):
    """10 m wind speed with barbs, labelled GRID-RELATIVE because it is.

    ``woof.render`` prefers ``wrf.getvar("uvmet10")`` -- the earth-rotated
    components -- and falls back to raw ``U10``/``V10`` in the model's native
    m/s when a file carries no SINALPHA/COSALPHA, labelling both the barbs
    and the colourbar accordingly.  This store carries ``setup/sina`` and
    ``setup/cosa``, but the rotation is not applied here: the fallback path's
    labelling is used instead, so the panel says what it is rather than
    claiming a rotation it did not perform.
    """
    _axes_style(axis)
    axis.set_title(title, fontsize=9)
    speed = np.hypot(u10, v10)
    if _flat(speed):
        axis.text(0.5, 0.5,
                  f"10 m wind is uniformly {float(np.nanmax(speed)):.3f} m/s "
                  "-- the surface layer scheme\nhas not been called yet at "
                  "this lead time.",
                  ha="center", va="center", fontsize=9, color="#8a1c1c",
                  transform=axis.transAxes)
        # Without this the panel keeps matplotlib's default 0-1 axes and the
        # note sits in a frame that is not the domain, next to three panels
        # that are -- which reads as a plotting failure rather than a zero.
        axis.set_xlim(float(lon.min()), float(lon.max()))
        axis.set_ylim(float(lat.min()), float(lat.max()))
        axis.set_facecolor("#f2f4f6")
        return None
    mesh = axis.pcolormesh(lon, lat, speed, cmap="viridis", shading="auto")
    ny, nx = speed.shape
    step = max(1, int(np.ceil(max(ny, nx) / barbs)))
    sub = (slice(step // 2, None, step), slice(step // 2, None, step))
    axis.barbs(lon[sub], lat[sub], u10[sub], v10[sub], length=4.6,
               linewidth=0.45, color="#222222")
    _colorbar(fig, axis, mesh, "10 m wind speed (m s-1, grid-relative)")
    return mesh


def seam_statistics(d: dict, *, tile: int, crop: int,
                    field: str = "WMAX", halfwidth: int = 3) -> dict:
    """Is there a discontinuity AT the tile boundaries?  Measured, not eyeballed.

    The seam figure asks the reader to look for a kink.  This asks the array.
    For each column ``i`` take the mean over rows of the second difference
    ``|F[j,i-1] - 2 F[j,i] + F[j,i+1]|`` -- a roughness that is large wherever
    the field is not smooth, which in a convecting domain is most places.

    The control is LOCAL, and that matters more than it sounds.  A domain has
    only ``n / tile - 1`` interior seams -- four per axis at 1440 with tile
    288 -- so comparing four columns against the domain mean measures where
    the storms happened to be, not the seams.  Each seam is therefore
    compared with the ``halfwidth`` columns on either side of it, which are
    in the same weather and differ only in that no tile boundary runs
    through them.

    The reported ratio is the mean over seams of ``roughness(seam) /
    roughness(its neighbours)``.  A halo that is too narrow, a tile that read
    a neighbour already advanced (the read-at-time-t rule in
    :mod:`tilestream.driver`), or a ring arena patching the wrong rectangle
    all put their error exactly on those columns, so any of them would push
    this well above 1.  Near 1 says the boundaries are not special -- which
    is the only thing it can accurately claim.  It does not prove the interior
    is right; the digest comparison does that.

    ``WMAX`` is the default rather than reflectivity because it is smooth
    where reflectivity is a thresholded step function, so a real seam has
    somewhere to show.
    """
    f = np.asarray(d[field], dtype=np.float64)
    ny, nx = f.shape
    out: dict = {"field": field, "tile": tile}
    for axis, n in (("x", nx), ("y", ny)):
        g = f if axis == "x" else f.T
        # rough[k] describes column k+1 of the plotted (cropped) array.
        rough = np.abs(g[:, :-2] - 2.0 * g[:, 1:-1] + g[:, 2:]).mean(axis=0)
        ratios, seams, controls = [], [], []
        for k in range(1, (n + crop) // tile + 1):
            c = k * tile - crop - 1          # index into rough
            if c - halfwidth < 0 or c + halfwidth >= rough.size:
                continue
            near = np.concatenate([rough[c - halfwidth:c],
                                   rough[c + 1:c + 1 + halfwidth]])
            seams.append(float(rough[c]))
            controls.append(float(near.mean()))
            ratios.append(float(rough[c] / max(near.mean(), 1e-30)))
        if not ratios:
            continue
        out[f"{axis}_n"] = len(ratios)
        out[f"{axis}_seam"] = float(np.mean(seams))
        out[f"{axis}_other"] = float(np.mean(controls))
        out[f"{axis}_ratio"] = float(np.mean(ratios))
    return out


def _load(path: Path, crop: int) -> dict:
    raw = np.load(path)
    out = {k: raw[k] for k in raw.files}
    if crop:
        c = slice(crop, -crop)
        for k, v in list(out.items()):
            if isinstance(v, np.ndarray) and v.ndim == 2:
                out[k] = v[c, c]
    return out


def _precip_total(d: dict):
    """WRF's own accumulation buckets, summed -- and only the ones that exist.

    ``woof.render`` composes ``RAINC + RAINNC`` because that is WRF's
    Registry bookkeeping, not a diagnostic.  With ``cu_physics=0`` there is
    no cumulus scheme and therefore no ``RAINC`` carrier at all, so the sum
    is over whichever buckets the configuration actually allocated; the
    caller is told which in the panel title.
    """
    have = [k for k in ("RAINC", "RAINNC") if k in d]
    total = d[have[0]].copy()
    for k in have[1:]:
        total = total + d[k]
    return total, " + ".join(have)


def _context(d: dict, run_label: str) -> str:
    from woof.render import spacing_label

    nx, ny, nz = int(d["nx"]), int(d["ny"]), int(d["nz"])
    dx = float(d["dx"])
    across = nx * dx / 1000.0
    line = (f"{nx}x{ny}x{nz} = {nx*ny*nz/1e6:.1f} Mcell | "
            f"{spacing_label(dx)} | {across:.0f} x {across:.0f} km | "
            f"{run_label}")
    # Hard-wrapped, because matplotlib does not wrap a suptitle: an
    # unwrapped provenance line runs off both edges of the canvas and the
    # measurement it carries is the half that got cut.
    for note in (CONFIG_NOTE[0], CEILING_NOTE[0]):
        if note:
            line += "\n" + textwrap.fill(note, width=WRAP_COLUMNS)
    return line


def _valid(d: dict) -> str:
    from datetime import timedelta

    from tilestream.bigdomain import start_datetime

    lead = float(d["elapsed_s"])
    stamp = start_datetime() + timedelta(seconds=lead)
    return (f"t+{lead/60:.0f} min  (valid {stamp:%Y-%m-%d %H:%M} UTC, "
            f"idealised clock)")


def sheet(path: Path, out_png: Path, *, crop: int, run_label: str,
          dpi: int = 130) -> None:
    """The four-panel forecaster's sheet at one lead time."""
    plt = _plt()
    d = _load(path, crop)
    lon, lat = d["LON"], d["LAT"]
    ny, nx = lat.shape
    fig, axes = plt.subplots(3, 2, figsize=(15.0, 19.4))
    top = _suptitle(fig, f"WOOF-ArWen out-of-core streamed forecast -- "
                         f"{_valid(d)}\n{_context(d, run_label)}", floor=0.955)

    refl_panel(fig, axes[0][0], lon, lat, d["REFL_COMPOSITE"],
               title="composite reflectivity (column-max REFL_10CM, "
                     "Morrison two-moment)")
    wind_panel(fig, axes[0][1], lon, lat, d["U10"], d["V10"],
               title="10 m wind (grid-relative barbs)")
    t2c = d["T2"] - 273.15
    panel_or_note(fig, axes[1][0], lon, lat, t2c,
                  title="2 m temperature (cold pools are the storm outflow)",
                  units=f"deg C  [{t2c.min():.1f} to {t2c.max():.1f}]",
                  # The panel plots degrees C, so the note has to say degrees
                  # C or it reads as a temperature of -273 K.  An unset T2 is
                  # 0 K exactly, which is what -273.15 C here means.
                  note_fmt=("2 m temperature is uniformly {value:.2f} deg C, "
                            "i.e. the carrier is still 0 K --\nNoah-MP and "
                            "MYNN have not been called yet at this lead "
                            "time."),
                  cmap="RdYlBu_r")
    total, names = _precip_total(d)
    precip_panel(fig, axes[1][1], lon, lat, total,
                 title=f"accumulated precipitation ({names})")
    # The two panels a hash cannot fake: w is a prognostic the model
    # integrates and is exactly zero in a state that is not moving, and PBLH
    # is MYNN's own diagnosis of how deep the mixing got.
    wmax = d["WMAX"]
    panel_or_note(fig, axes[2][0], lon, lat, wmax,
                  title="column-maximum vertical velocity (updraught cores)",
                  units=f"m s-1  [0 to {wmax.max():.1f}]",
                  note_fmt="column-max w is uniformly {value:.3f} m/s",
                  cmap="magma", vmin=0.0, vmax=max(float(wmax.max()), 1.0))
    pblh = d.get("PBLH")
    if pblh is None:
        axes[2][1].axis("off")
    else:
        panel_or_note(fig, axes[2][1], lon, lat, pblh,
                      title="PBL depth (MYNN)",
                      units=f"m  [{pblh.min():.0f} to {pblh.max():.0f}]",
                      note_fmt=("PBL depth is uniformly {value:.3f} m -- "
                                "MYNN has not been called yet\nat this lead "
                                "time."),
                      cmap="cividis")
    _footer(fig, crop, nx, ny, top=top, bottom=0.075)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def evolution(paths, out_png: Path, *, crop: int, run_label: str,
              field: str = "REFL_COMPOSITE", dpi: int = 130) -> None:
    """One field at several lead times: does the model integrate or sit there?"""
    plt = _plt()
    frames = [_load(Path(p), crop) for p in paths]
    n = len(frames)
    lon, lat = frames[0]["LON"], frames[0]["LAT"]
    ny, nx = lat.shape
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 7.6), squeeze=False)
    label = {"REFL_COMPOSITE": "composite reflectivity",
             "WMAX": "column-maximum vertical velocity"}.get(field, field)
    top = _suptitle(fig, f"WOOF-ArWen out-of-core streamed forecast -- {label} "
                         f"evolution\n{_context(frames[0], run_label)}",
                    floor=0.88)
    # One scale across the whole strip, set by the largest frame.  A
    # per-frame scale would make a growing field look steady, which is the
    # one thing an evolution figure exists to rule out.
    hi = max(float(np.nanmax(f[field])) for f in frames) if field != \
        "REFL_COMPOSITE" else 0.0
    for axis, d in zip(axes[0], frames):
        if field == "REFL_COMPOSITE":
            refl_panel(fig, axis, lon, lat, d[field], title=_valid(d))
        else:
            panel_or_note(fig, axis, lon, lat, d[field], title=_valid(d),
                          units=f"m s-1  [0 to {hi:.1f}], common scale",
                          note_fmt="uniformly {value:.3f} m/s",
                          cmap="magma", vmin=0.0, vmax=max(hi, 1.0))
    _footer(fig, crop, nx, ny, top=top, bottom=0.22)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def hero(path: Path, out_png: Path, *, crop: int, run_label: str,
         field: str = "REFL_COMPOSITE", dpi: int = 170) -> None:
    """One field, one panel, as large as the domain deserves.

    The dpi is chosen against the CELL COUNT, not by eye: at 12.5 inches
    wide the axes are about 10 of them, so 170 dpi puts ~1700 pixels across
    a 1408-cell crop and every model cell survives to the file.  Below about
    140 the storms start being resampled away, which on a convection-
    allowing domain quietly deletes the thing the figure is of.
    """
    plt = _plt()
    d = _load(path, crop)
    lon, lat = d["LON"], d["LAT"]
    ny, nx = lat.shape
    fig, axis = plt.subplots(figsize=(12.5, 11.4))
    if field == "REFL_COMPOSITE":
        refl_panel(fig, axis, lon, lat, d[field],
                   title="composite reflectivity "
                         "(column-max REFL_10CM, Morrison two-moment)")
    elif field == "WMAX":
        hi = float(np.nanmax(d[field]))
        panel_or_note(fig, axis, lon, lat, d[field],
                      title="column-maximum vertical velocity",
                      units=f"m s-1  [0 to {hi:.1f}]",
                      note_fmt="uniformly {value:.3f} m/s",
                      cmap="magma", vmin=0.0, vmax=max(hi, 1.0))
    else:
        raise ValueError(f"no hero renderer for {field!r}")
    top = _suptitle(fig, f"WOOF-ArWen out-of-core streamed forecast -- "
                         f"{_valid(d)}\n{_context(d, run_label)}", floor=0.90)
    _footer(fig, crop, nx, ny, top=top, bottom=0.13)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def tile_seams(path: Path, out_png: Path, *, crop: int, run_label: str,
               tile: int, dpi: int = 170) -> None:
    """Reflectivity with the tile interiors drawn on it.

    This is the figure a digest cannot replace.  Every cell in the domain was
    computed inside one of these boxes, from a 16-cell halo gathered out of
    pinned host RAM, and the boxes were stepped in an order the physics knows
    nothing about.  If the halo were too narrow, if a tile read a neighbour
    that had already been advanced (the read-at-time-t rule in
    :mod:`tilestream.driver`), or if the ring arena patched the wrong
    rectangle, the error would appear AT THESE LINES and nowhere else -- a
    storm would step, an outflow boundary would kink, a gradient would
    terminate.  A reader can check that claim with their eyes and no tooling.

    ``tile`` is the INTERIOR edge; the lines are drawn at its multiples in
    the original index space, offset by ``crop`` because the plotted array
    has already had its periodic-wrap margin removed.
    """
    plt = _plt()
    d = _load(path, crop)
    lon, lat, refl = d["LON"], d["LAT"], d["REFL_COMPOSITE"]
    ny, nx = lat.shape
    fig, axis = plt.subplots(figsize=(12.5, 11.4))
    refl_panel(fig, axis, lon, lat, refl,
               title=f"composite reflectivity with the {tile}x{tile} tile "
                     "interiors drawn: no seam appears on any of them")
    edges = [k * tile - crop for k in range(1, (nx + crop) // tile + 1)]
    for e in edges:
        if 0 <= e < nx:
            axis.plot(lon[:, e], lat[:, e], color="#111111", lw=0.7,
                      alpha=0.55, zorder=5)
        if 0 <= e < ny:
            axis.plot(lon[e, :], lat[e, :], color="#111111", lw=0.7,
                      alpha=0.55, zorder=5)
    top = _suptitle(fig, f"WOOF-ArWen out-of-core streamed forecast -- "
                         f"{_valid(d)}\n{_context(d, run_label)}", floor=0.90)
    _footer(fig, crop, nx, ny, top=top, bottom=0.13)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def storm_window(d: dict, side: int) -> tuple[slice, slice]:
    """The ``side``x``side`` window centred on the domain's strongest updraught.

    The first rule tried here was "most cells above 40 dBZ", and it was a
    bad rule on this domain: the terrain ridge goes convective EVERYWHERE
    once the upslope flow and the surface fluxes get going, so the densest
    echo is a 500 km blanket of grid-scale cells and the window landed on it
    every time.  That blanket is real and the full-domain figure shows it;
    it is just not what a zoom is for.

    Column-max ``w`` picks the other thing.  A supercell updraught is the
    largest vertical velocity in the domain by a wide margin -- 45 m/s
    against a few m/s for the ridge convection -- so centring on its maximum
    lands on a storm with structure, and it does so without a hand-picked
    coordinate that would have to be re-picked for every frame.
    """
    w = np.asarray(d["WMAX"])
    ny, nx = w.shape
    side = int(min(side, ny, nx))
    j, i = np.unravel_index(int(w.argmax()), w.shape)
    j0 = int(np.clip(j - side // 2, 0, ny - side))
    i0 = int(np.clip(i - side // 2, 0, nx - side))
    return slice(j0, j0 + side), slice(i0, i0 + side)


def zoom(path: Path, out_png: Path, *, crop: int, run_label: str,
         side: int = 400, dpi: int = 170) -> None:
    """A window of the big domain at full model resolution.

    The hero figure is accurate and nearly unreadable: 4320 km across, a storm
    is 30 cells wide, and a reader cannot tell a supercell from a speck.
    This crops the busiest ``side``x``side`` cells -- the SAME array, no
    resampling, no second run -- so the structure inside those specks is
    visible: hook echoes, forward-flank precipitation shields, and the cold
    pools driving them.  Both figures are needed; neither replaces the
    other.
    """
    plt = _plt()
    d = _load(path, crop)
    jj, ii = storm_window(d, side)
    lon, lat = d["LON"][jj, ii], d["LAT"][jj, ii]
    km = side * float(d["dx"]) / 1000.0
    fig, axes = plt.subplots(1, 2, figsize=(17.0, 8.6))
    refl_panel(fig, axes[0], lon, lat, d["REFL_COMPOSITE"][jj, ii],
               title="composite reflectivity (column-max REFL_10CM)")
    t2c = d["T2"][jj, ii] - 273.15
    panel_or_note(fig, axes[1], lon, lat, t2c,
                  title="2 m temperature -- the cold pools under the storms",
                  units=f"deg C  [{t2c.min():.1f} to {t2c.max():.1f}]",
                  note_fmt="uniformly {value:.2f} deg C",
                  cmap="RdYlBu_r")
    ny, nx = d["LAT"].shape
    top = _suptitle(fig, f"WOOF-ArWen out-of-core streamed forecast -- "
                         f"{_valid(d)}\nZOOM: {side}x{side} cells "
                         f"({km:.0f} x {km:.0f} km) centred on the domain's "
                         f"strongest updraught, at full model resolution\n"
                         f"{_context(d, run_label)}", floor=0.86)
    _footer(fig, crop, nx, ny, top=top, bottom=0.19)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def terrain(path: Path, out_png: Path, *, crop: int, run_label: str,
            dpi: int = 140) -> None:
    """The domain itself: what geography the tiles were gathering."""
    plt = _plt()
    d = _load(path, crop)
    lon, lat, ht = d["LON"], d["LAT"], d["HT"]
    ny, nx = lat.shape
    fig, axis = plt.subplots(figsize=(10.5, 9.6))
    _axes_style(axis)
    mesh = axis.pcolormesh(lon, lat, ht, cmap="terrain", shading="auto")
    _colorbar(fig, axis, mesh, "terrain height (m)")
    axis.set_title("domain geography: terrain gathered per tile, never rebuilt",
                   fontsize=10)
    top = _suptitle(fig, f"WOOF-ArWen out-of-core streamed forecast -- domain\n"
                         f"{_context(d, run_label)}", floor=0.90)
    _footer(fig, crop, nx, ny, top=top, bottom=0.13)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _footer(fig, crop: int, nx: int, ny: int, *, top: float,
            bottom: float) -> None:
    """The caption band, with the margins reserved for it EXPLICITLY.

    ``bbox_inches="tight"`` grows the canvas to fit stray artists but does
    not move them apart, so a footer placed at y=0.01 of a short figure
    lands on top of the x-axis labels and a three-line suptitle lands on the
    panel titles.  The margins are therefore reserved per figure shape
    rather than left to the layout engine.
    """
    text = IC_FOOTER
    if crop:
        text += (f" Outer {crop} cells cropped on every edge ({nx}x{ny} "
                 "shown): the domain is periodic on a real projection, so "
                 "the wrap is a geographic discontinuity.")
    fig.subplots_adjust(top=top, bottom=bottom, left=0.06, right=0.97,
                        wspace=0.28, hspace=0.22)
    fig.text(0.5, bottom * 0.42, text, ha="center", va="center",
             fontsize=7.5, color="#444444", wrap=True)


#: Products the Rust route asks for.  Both keywords expand from the store
#: catalog, which is what the streamed lane wants: a snapshot carries
#: whatever the run wrote, and naming a fixed list here would silently
#: drop a field the day one is added to the snapshot.  ``all`` is every
#: named product the frames can draw; ``variables`` is every stored field
#: no named product draws, which is how the column updraft extremes
#: (``W_UP_MAX``/``W_DN_MAX``) reach a panel at all.
RUST_PRODUCTS = "all,variables"


def sibling_wrfout(npz_path: Path) -> Path:
    """The wrfout beside an ``.npz``, written from it if it is not there.

    ``tilestream/run_bigdomain.py`` writes one by default now, so the
    usual answer is "it already exists".  The write-it path is for the
    ``.npz`` files that predate that default: they carry the coordinates,
    the composite and the surface fields, so the frame is recoverable
    exactly rather than approximately.
    """

    from woof.io.surface_wrfout import (snapshot_wrfout_path,
                                         write_surface_wrfout)
    from tilestream.run_bigdomain import (SNAPSHOT_EPOCH,
                                          _snapshot_time_string)

    target = snapshot_wrfout_path(npz_path)
    if target.is_file():
        return target
    snapshot = dict(np.load(npz_path))
    write_surface_wrfout(
        target, snapshot, time_str=_snapshot_time_string(snapshot),
        dx=float(snapshot["dx"]), grid_id=1, start_time=SNAPSHOT_EPOCH,
        title=f"woof tile-streamed big domain ({npz_path.stem})")
    print(f"materialised {target.name} from {npz_path.name}")
    return target


def render_rust(paths, out: Path, *, source_label: str) -> int:
    """Drive the production renderer over the frames; its exit code.

    A thin driver on purpose: nothing here draws, so there is no second
    theme to drift from the product tier.
    """

    import subprocess
    import sys as _sys

    frames = [sibling_wrfout(path) for path in paths]
    out.mkdir(parents=True, exist_ok=True)
    command = [_sys.executable, "-m", "woof.cli", "render",
               "--engine", "rust", "--products", RUST_PRODUCTS,
               "--out", str(out), "--source-label", source_label]
    command.extend(str(frame) for frame in frames)
    result = subprocess.run(command)
    drawn = sorted(out.rglob("*.png"))
    print(f"done: {len(drawn)} product panels from the production renderer")
    return result.returncode


def resolve_engine(request: str) -> tuple[str, str]:
    """``(engine, why)`` -- the three-way contract ``woof render`` uses.

    An explicit ``rust`` that cannot be honoured raises rather than
    quietly drawing with the other engine; ``auto`` degrades and NAMES
    the reason, because a fallback nobody was told about is how a
    matplotlib panel ends up in a product gallery.
    """

    if request == "matplotlib":
        return "matplotlib", "requested"
    from woof import rustwx

    engine = rustwx.find_renderer()
    if engine is None:
        reason = f"rw_wrfbatch is not built ({rustwx.CARGO_BUILD_HINT})"
        if request == "rust":
            raise SystemExit(f"bigdomain_render: {reason}")
        return "matplotlib", reason
    usable, evidence = rustwx.probe_renderer(engine)
    if not usable:
        if request == "rust":
            raise SystemExit(f"bigdomain_render: {engine}: {evidence}")
        return "matplotlib", f"{engine}: {evidence}"
    return "rust", str(engine)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--engine", default="auto", choices=("auto", "rust", "matplotlib"),
        help="which renderer draws the panels (default auto).  The render "
             "law puts weather fields on the Rust renderer; 'auto' uses it "
             "when this checkout has built it and falls back to matplotlib "
             "with the reason named, 'rust' refuses rather than "
             "substituting, and 'matplotlib' selects the DEPRECATED "
             "fallback outright -- which is also where the multi-panel "
             "sheets and evolution strips still live")
    ap.add_argument("--crop", type=int, default=16)
    ap.add_argument("--prefix", default="bigdomain")
    ap.add_argument("--run-label", default="WOOF-ArWen, streamed out-of-core")
    ap.add_argument("--ceiling-note", default="",
                    help="measured resident-ceiling line for every figure")
    ap.add_argument("--zoom", type=int, default=400,
                    help="edge in cells of the full-resolution zoom window")
    ap.add_argument("--tile", type=int, default=0,
                    help="interior tile edge the run used; draws the tile "
                         "seam overlay when given")
    ap.add_argument("--sheet-leads", default="",
                    help="comma-separated forecast minutes that get a "
                         "six-panel sheet; empty means every frame")
    ap.add_argument("--config-note", default="",
                    help="physics and tiling line for every figure")
    args = ap.parse_args()
    CEILING_NOTE[0] = args.ceiling_note
    CONFIG_NOTE[0] = args.config_note
    args.out.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.npz)

    engine, why = resolve_engine(args.engine)
    print(f"bigdomain_render: engine {engine} ({why})")
    if engine == "rust":
        raise SystemExit(render_rust(paths, args.out,
                                     source_label=args.run_label))
    print("bigdomain_render: WARNING -- the matplotlib panels below are "
          "the render law's DEPRECATED FALLBACK, not the product tier")

    head = np.load(paths[0])
    stem = f"{args.prefix}_streamed_{int(head['nx'])}sq"

    def lead(path) -> str:
        return f"T{int(round(float(np.load(path)['elapsed_s']) / 60)):03d}min"

    written = []

    def emit(name, fn, *a, **kw):
        out = args.out / name
        fn(*a, out, **kw)
        written.append(out)
        print(f"wrote {out}")

    # The evolution strips want every frame; the six-panel sheets do not.
    # At 1440^2 each sheet is six 2 Mcell QuadMeshes, so rendering one per
    # dump is minutes of matplotlib for figures a reader will not compare
    # side by side anyway.  ``--sheet-leads`` picks the ones worth having.
    want = (None if not args.sheet_leads else
            {int(v) for v in args.sheet_leads.split(",")})
    for path in paths:
        minutes = int(round(float(np.load(path)["elapsed_s"]) / 60))
        if want is not None and minutes not in want:
            continue
        emit(f"{stem}_sheet_{lead(path)}.png", sheet, path,
             crop=args.crop, run_label=args.run_label)
    emit(f"{stem}_evolution_refl.png", evolution, paths,
         crop=args.crop, run_label=args.run_label)
    emit(f"{stem}_evolution_wmax.png", evolution, paths,
         crop=args.crop, run_label=args.run_label, field="WMAX")
    emit(f"{stem}_refl_hero_{lead(paths[-1])}.png", hero, paths[-1],
         crop=args.crop, run_label=args.run_label)
    emit(f"{stem}_zoom_{lead(paths[-1])}.png", zoom, paths[-1],
         crop=args.crop, run_label=args.run_label, side=args.zoom)
    emit(f"{stem}_domain_terrain.png", terrain, paths[0],
         crop=args.crop, run_label=args.run_label)
    if args.tile:
        for path in paths:
            stats = seam_statistics(_load(path, args.crop), tile=args.tile,
                                    crop=args.crop)
            if "x_ratio" in stats:
                print(f"seam roughness {lead(path)} ({stats['field']}, "
                      f"vs local neighbours): "
                      f"x {stats['x_seam']:.4g} / {stats['x_other']:.4g} = "
                      f"{stats['x_ratio']:.3f} over {stats['x_n']} seams, "
                      f"y {stats['y_seam']:.4g} / {stats['y_other']:.4g} = "
                      f"{stats['y_ratio']:.3f} over {stats['y_n']} seams")
        emit(f"{stem}_tileseams_{lead(paths[-1])}.png", tile_seams,
             paths[-1], crop=args.crop, run_label=args.run_label,
             tile=args.tile)
    print(f"done: {len(written)} figures")


if __name__ == "__main__":
    main()
