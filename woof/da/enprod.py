"""``woof enprod`` -- the products that make an ensemble legible.

An ensemble of N members is N wrfout timelines; nobody reads N of
anything.  The WoFS-style suite below collapses them into five pictures
that a forecaster can actually use:

* ``mean``      -- the ensemble mean of a scalar product field.
* ``spread``    -- the ensemble standard deviation (1 sigma, ddof=1):
  where the members disagree.
* ``prob``      -- probability of exceedance at a threshold, optionally
  over a neighborhood of a given radius (max-in-neighborhood per member
  first, THEN the ensemble fraction: the NMEP convention).
* ``paintball`` -- every member's threshold contour on one panel, one
  stable colour per member: the ensemble's spatial envelope with the
  membership still readable.
* ``pmm``       -- the probability-matched mean for reflectivity-like
  fields: the mean's spatial pattern carrying the pooled members'
  intensity distribution, because a plain ensemble mean of reflectivity
  smears every storm into a blob that verifies as nothing.

INPUT is an ensemble root directory: ``member_NNN/`` run directories
holding ordinary wrfouts, plus an ``ensemble-manifest.json`` of schema
``gpuwm-ensemble-manifest.v1`` (see :func:`load_manifest`).  Nothing here
scans for member directories on its own -- the manifest is the roster,
and a member the manifest declares but the disk does not have is a
refusal naming that member, never a quietly smaller ensemble.  A
19-member "30-member" probability field is wrong by 37% and looks
exactly like a right one.

CONVENTIONS are :mod:`woof.render`'s, imported rather than restated:
the domain+resolution filename token (``d02-3km``), the subtitle line
(``d02-3km | dx 3 km | valid ... | WOOF``), the recessive-axes panel,
the NWS 5-dBZ reflectivity scale, and the claims-map refusal that made
v1.0.1 stop silently overwriting one nest's PNG with another's.  On top
of them this module adds the two facts an ensemble product must carry:
the member count, and an EXPERIMENTAL stamp -- these products are new in
v1.2, and none of them has been calibrated against a verification
archive.

Filenames extend the v1.0.1 shape ``{product}_{domain-token}_{stamp}``
by putting an ensemble token in the product slot::

    refl-ens-mean_d02-3km_1974-04-03_18-00-00.png
    refl-ens-spread_d02-3km_1974-04-03_18-00-00.png
    refl-ens-p40dbz_d02-3km_1974-04-03_18-00-00.png
    refl-ens-p40dbz-r5km_d02-3km_1974-04-03_18-00-00.png
    refl-ens-paintball40dbz_d02-3km_1974-04-03_18-00-00.png
    refl-ens-pmm_d02-3km_1974-04-03_18-00-00.png

ENGINES: ``--engine`` selects which renderer draws the panels, and the
default is ``auto``, which means the Rust one.  ``rw_ensbatch`` is the
engine this module's docstring used to say did not exist -- the same
reductions (``rustwx-ensemble``, unit-tested against the same probes the
functions below name) on the same production map/basemap/design/PNG path
a deterministic ``woof render --engine rust`` panel comes off.

The matplotlib suite below (:func:`run_suite` and everything it calls) is
therefore the render law's DEPRECATED FALLBACK: reachable, documented,
and NOT the product tier.  ``auto`` does not arrive here at all: it
answers rust or a refusal naming the engine, the build line and the
workaround (:func:`woof.rustwx_lanes.resolve_lane_engine`).  The one
way in is ``--engine matplotlib``, by name, and taking it prints
:func:`woof.rustwx_lanes.ensemble_workaround_notice` on stderr every
run, because an opt-in route is a workaround and says so.  Nothing is
kept here that the Rust route cannot do: ``--domain dNN`` is passed to
the engine as the nest choice and a member directory holding several
nests is refused once, in the same sentence whichever engine was asked
for, and ``--dpi`` is the same picture on both routes, scaled from the
engine's 1200x900 at the default ``--dpi 150`` (:func:`panel_pixels`).
MEASURED 2026-08-17: five real tile-streamed members, all five products
through the real executable at the default engine.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Callable

import numpy as np

from woof.cli_numbers import positive_int

# The v1.0.1 product conventions, imported rather than re-implemented.
# The underscored names are deliberate: they are woof.render's own
# private helpers, and reaching for them here is a statement that the
# ensemble suite draws the SAME panel with the SAME tokens as the
# deterministic suite.  A second implementation of `_figure` would be a
# second theme, and the two would drift within one release.
from woof.explain import warn
from woof.render import (DEFAULT_SOURCE_LABEL, _NWS_COLORS, _NWS_LEVELS,
                          _PRECIP_LEVELS, _domain_tag, _figure, _finish,
                          _grid_spacing_m, _import_wrf, _pyplot,
                          _stamp_for_filename, domain_token, plot_context)

# The frame-inventory contract, imported from the lane that WRITES it.  A
# second spelling of it here would be a second contract, and the count
# check this replaced is what a second contract decays into.
from woof.ensemble.wrfout_inventory import (
    WRFOUT_INVENTORY_CONTRACT, WRFOUT_INVENTORY_KEY,
    canonical_relative_path as inventory_relative_path,
    domain_token as inventory_domain_token,
    duplicate_paths as inventory_duplicate_paths,
    entry_frames as inventory_entry_frames,
    member_inventory as build_member_inventory,
    read_inventory as read_member_inventory,
    verify_entry as verify_inventory_entry)

#: The manifest schema this module reads.  Checked by VALUE, not by which
#: key carries it: the sibling lane owns the writer, and a disagreement
#: about the key's spelling should not be a data-loss-shaped failure.
MANIFEST_SCHEMA = "gpuwm-ensemble-manifest.v1"

#: The manifest's filename inside the ensemble root.
MANIFEST_FILENAME = "ensemble-manifest.json"

#: The schema ``woof go --members`` writes, one manifest per domain at
#: ``<run>/<domain>/ensemble-manifest.json``
#: (:attr:`woof.ensemble.batch_product_output.NativeDiagnosticSpool.CONTRACT`).
#: It carries no per-member status: its roster is ``member_order`` and the
#: members' retained histories are ``member_files``, paths relative to the
#: run folder, present only under ``--keep-member-files``.  2.8.7 and the
#: 2.8.8 candidate wrote it and this module read only v1, so enprod on such
#: a run printed a schema warning and then the engine's usage line.
V2_MANIFEST_SCHEMA = "gpuwm-ensemble-output.v2"

#: Top-level keys inspected for the schema string.
_SCHEMA_KEYS = ("schema", "schema_version", "format", "kind")

#: Per-member keys inspected for the run directory, most authoritative
#: first.  ``member_dir`` is the writer's own spelling
#: (:func:`woof.ensemble.manifest.new_ensemble_manifest`) and is listed
#: first for that reason: without it this module fell through to its
#: ``member_{NNN:03d}`` guess, which happened to agree with the writer
#: today and would have gone silently wrong the day the layout changed.
_DIRECTORY_KEYS = ("member_dir", "dir", "directory", "path")

#: Member statuses accepted without an explicit opt-in.  Anything else is
#: a refusal that PRINTS the statuses it actually found, so the operator
#: learns the right ``--accept-status`` value from the refusal itself
#: instead of from this docstring.
#:
#: Both spellings are real and both mean "this member finished".  ``DONE``
#: is what :mod:`woof.ensemble.manifest` -- the writer of record for
#: ``gpuwm-ensemble-manifest.v1`` -- records
#: (``MEMBER_STATUSES = PENDING|RUNNING|DONE|FAILED``); ``complete`` is the
#: spelling this module read before that writer landed and is kept because
#: manifests written against it exist.  The set is deliberately not
#: case-folded and deliberately short: ``RUNNING`` and ``FAILED`` are
#: exactly the members that must not be averaged, and admitting them has to
#: stay an explicit ``--accept-status`` on the operator's part.
DEFAULT_ACCEPT_STATUS = ("DONE", "complete")

def experimental_stamp() -> str:
    """The uncalibrated-ensemble stamp, versioned by the RUNNING engine.

    The warning half is still true and stays: these ensemble products
    have never been calibrated against verification, and a panel that
    does not say so will be screenshotted into a briefing that assumes
    it was.

    The VERSION half used to be the literal string ``v1.2``, frozen at
    the release the suite was written for and stamped onto every panel
    every release since -- so a 1.8.7 plot claimed to come from a 1.2
    ensemble.  A version on a product is a provenance claim; the only
    accurate source for it is the engine that produced the product, which
    is :data:`woof.__version__` (read from the installed
    distribution's metadata, so an editable checkout and a wheel both
    answer for themselves).  Read at CALL time rather than at import
    time so a process that reloads or reinstalls the package cannot
    keep stamping the version it started with.
    """

    from woof import __version__

    return f"EXPERIMENTAL (v{__version__} ensemble, uncalibrated)"


def __getattr__(name: str):
    """``EXPERIMENTAL_STAMP`` is served at ACCESS time, never frozen.

    The attribute survives for compatibility, but as PEP 562 module
    ``__getattr__`` rather than a module constant: a constant evaluated
    at import time is exactly the frozen-at-import stamping that
    :func:`experimental_stamp`'s docstring says this suite no longer
    does.  Every access re-reads the running engine's version; the
    function is the authority and this name is a view of it.
    """

    if name == "EXPERIMENTAL_STAMP":
        return experimental_stamp()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

#: What this suite does with a non-finite member value.  ONE policy, named
#: and applied identically by every product, because the two ways it used
#: to be handled implicitly were both wrong in the same direction --
#: quietly:
#:
#: ``"mask"`` (the default) -- MASKED AND COUNTED.  A non-finite value is
#:   not data -- NaN and +/-Inf alike, which is what ``np.isfinite`` means
#:   and what every reduction here uses -- so it is excluded from every
#:   reduction at that grid point
#:   and the DENOMINATOR shrinks with it.  A point where no member is
#:   finite is NaN in the output: there is nothing to say there, and
#:   saying nothing is the answer.  The coverage this leaves is recorded
#:   in the provenance and printed on the panel, which is what answers the
#:   real objection to nan-aware statistics -- not that they exclude
#:   missing members, but that they do it without disclosing the count.
#:
#: ``"refuse"`` -- fail closed.  Any non-finite value anywhere in the
#:   member stack refuses the whole product, naming the members and the
#:   point counts.  For an operational sheet where a silently masked
#:   member is worse than no sheet at all.
#:
#: What is NOT on offer is the old implicit behaviour, where ``NaN >
#: threshold`` evaluated ``False`` and an invalid member was counted as a
#: MISS against the full ensemble denominator.  That understated
#: probability and broke the advertised radius monotonicity outright: a
#: one-member probe of ``[10, NaN, 0]`` at threshold 5 gave ``[1, 0, 0]``
#: at radius 0 and ``[0, 0, 0]`` at radius 1, because ``np.maximum``
#: propagated the NaN across the neighborhood.
NAN_POLICIES = ("mask", "refuse")
DEFAULT_NAN_POLICY = "mask"

#: How the probability-matched mean breaks ties in the ensemble mean.
#:
#: ``"flat-index"`` (the default) -- Ebert's algorithm exactly: the pooled
#:   intensity distribution is preserved value for value, and points whose
#:   means are equal receive their pooled values in flat-index order.
#:   That order is deterministic and carries NO information: a four-cell
#:   plateau at mean 1.5 comes back as ``[3, 2, 1, 0]``, which is a
#:   row-major artifact of the tie and not a feature of the forecast.  The
#:   PMM's defining property is that the output's value distribution IS
#:   the pooled members' one, so this is the rule that keeps the product
#:   what it claims to be, and the tie statistics are reported so a reader
#:   can see how much of a panel is plateau.
#:
#: ``"average"`` -- every point in a tie group receives the mean of the
#:   pooled values that group was allotted.  No artificial gradient, at
#:   the cost of the exact pooled distribution (and, on a tied maximum, of
#:   the retained ensemble maximum).  Offered because on a field with a
#:   large exact floor the artifact is visible and the distribution claim
#:   is not what the reader is using.
PMM_TIE_RULES = ("flat-index", "average")
DEFAULT_PMM_TIE_RULE = "flat-index"

#: Paintball member colours.  40 entries (matplotlib ``tab20`` + ``tab20b``)
#: covers the 30-member target with headroom.  Held as literal hex rather
#: than pulled from matplotlib so member->colour assignment is testable
#: without the render extra installed, and so the mapping cannot shift
#: under a matplotlib release.
PAINTBALL_PALETTE = (
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
    "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939",
    "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39",
    "#e7ba52", "#e7cb94", "#843c39", "#ad494a", "#d6616b",
    "#e7969c", "#7b4173", "#a55194", "#ce6dbd", "#de9ed6",
)

#: Probability shading ladder: 10% steps, with zero left transparent so
#: the plot shows where the ensemble said something, not where it did not.
_PROB_LEVELS = tuple(np.round(np.arange(0.1, 1.001, 0.1), 3))


class EnsembleRefusal(RuntimeError):
    """A fail-closed refusal: the ensemble is not what was claimed.

    Every message names the members at fault.  "some members are
    missing" is not a diagnosis, and an operator cannot act on it.
    """


# ---------------------------------------------------------------------------
# Scalar product fields
# ---------------------------------------------------------------------------


def _extract_refl(wrf, wrffile, timeidx):
    """Composite reflectivity: column max of model-native REFL_10CM.

    The same direct-field product ``woof.render`` draws -- not a
    re-derived simulated reflectivity.
    """

    refl = np.asarray(wrf.getvar(wrffile, "REFL_10CM", timeidx=timeidx))
    return refl.max(axis=0)


def _extract_uh(wrf, wrffile, timeidx):
    """Updraft helicity: WRF's own stored ``UP_HELI_MAX`` accumulator."""

    return np.asarray(wrf.getvar(wrffile, "UP_HELI_MAX", timeidx=timeidx))


def _uh_layer(variable):
    """A low-level updraft helicity layer's stored interval max (``UP_HELI_MAX01/02/03``, WOOF uh-layers 2026-11-02)."""

    def extract(wrf, wrffile, timeidx):
        return np.asarray(wrf.getvar(wrffile, variable, timeidx=timeidx))

    return extract


def _extract_uh_band(name):
    def extract(wrf, wrffile, timeidx):
        """Low-level updraft helicity: the stored interval maximum ``name``."""
        return np.asarray(wrf.getvar(wrffile, name, timeidx=timeidx))
    return extract


def _extract_t2(wrf, wrffile, timeidx):
    return np.asarray(
        wrf.getvar(wrffile, "t2", timeidx=timeidx, units="degC"))


def _extract_wspd10(wrf, wrffile, timeidx):
    return np.asarray(wrf.getvar(wrffile, "wspd10", timeidx=timeidx))


def _extract_precip(wrf, wrffile, timeidx):
    """WRF's accumulation-bucket total (RAINC + RAINNC).

    Bookkeeping over Registry accumulators, exactly as ``woof.render``
    does it; a file carrying neither bucket is an error, not a zero.
    """

    buckets = []
    for name in ("RAINC", "RAINNC"):
        try:
            buckets.append(
                np.asarray(wrf.getvar(wrffile, name, timeidx=timeidx)))
        except Exception:
            continue
    if not buckets:
        raise RuntimeError(
            "neither RAINC nor RAINNC is present; there is no "
            "accumulated-precipitation field to ensemble")
    total = buckets[0]
    for bucket in buckets[1:]:
        total = total + bucket
    return total


@dataclass(frozen=True)
class FieldSpec:
    """A scalar 2D product field the ensemble suite can operate on."""

    name: str
    title: str
    units: str
    #: Filename-safe unit token for exceedance thresholds (``p40dbz``).
    unit_slug: str
    #: Default exceedance threshold, in ``units``.
    default_threshold: float
    extract: Callable
    cmap: str = "viridis"
    #: Discrete shading ladder for mean/pmm panels, when the field has a
    #: conventional one.  ``None`` lets the data range set the limits.
    levels: tuple | None = None
    #: Draw mean/pmm on the NWS 5-dBZ scale.
    nws_scale: bool = False
    #: Whether ``--products all`` includes the probability-matched mean.
    #: PMM is a reflectivity-like construct: it presumes a skewed,
    #: positive-definite intensity distribution whose peaks the mean
    #: smears.  On 2 m temperature it computes, and it means nothing.
    pmm_in_all: bool = False
    #: Engine-declared probability thresholds, all in native units.
    default_thresholds: tuple[float, ...] = ()


#: Field registry: ``--field`` name -> spec.  Generic product fields
#: only; no case, campaign, or source identity appears here or anywhere
#: downstream of it.
def _visibility_native_only(*args, **kwargs):
    raise RuntimeError("visibility requires the native Rust column diagnostic; use --engine rust")

FIELDS: dict[str, FieldSpec] = {
    "visibility": FieldSpec(name="visibility",title="visibility",units="m",unit_slug="m",
        default_threshold=1000.0,extract=_visibility_native_only),
    "fog_prob": FieldSpec(name="fog_prob",title="visibility",units="m",unit_slug="m",
        default_threshold=1000.0,extract=_visibility_native_only),
    "refl": FieldSpec(
        name="refl", title="composite reflectivity", units="dBZ",
        unit_slug="dbz", default_threshold=40.0, extract=_extract_refl,
        cmap="turbo", nws_scale=True, pmm_in_all=True),
    "uh": FieldSpec(
        name="uh", title="updraft helicity", units="m2 s-2",
        unit_slug="m2s2", default_threshold=75.0, extract=_extract_uh,
        cmap="YlOrRd"),
    # The low-level layers (WOOF uh-layers, 2026-11-02): an export carries them when its run wrote them; the Rust
    # engine's own rows (rw_ensbatch) name the same fields.
    "uh01": FieldSpec(
        name="uh01", title="0-1 km updraft helicity", units="m2 s-2",
        unit_slug="m2s2", default_threshold=10.0, extract=_uh_layer("UP_HELI_MAX01"),
        cmap="YlOrRd", default_thresholds=(10.0, 20.0, 40.0)),
    "uh02": FieldSpec(
        name="uh02", title="0-2 km updraft helicity", units="m2 s-2",
        unit_slug="m2s2", default_threshold=50.0, extract=_uh_layer("UP_HELI_MAX02"),
        cmap="YlOrRd", default_thresholds=(25.0, 50.0, 75.0)),
    "uh03": FieldSpec(
        name="uh03", title="0-3 km updraft helicity", units="m2 s-2",
        unit_slug="m2s2", default_threshold=50.0, extract=_uh_layer("UP_HELI_MAX03"),
        cmap="YlOrRd", default_thresholds=(25.0, 50.0, 75.0)),
    **{f"uh{suffix}": FieldSpec(
        name=f"uh{suffix}", title=f"0-{top} km updraft helicity", units="m2 s-2",
        unit_slug="m2s2", default_threshold=25.0,
        extract=_extract_uh_band(f"UP_HELI_MAX{suffix}"), cmap="YlOrRd")
       for suffix, top in (("01", 1), ("02", 2), ("03", 3))},
    "precip": FieldSpec(
        name="precip", title="accumulated precipitation", units="mm",
        unit_slug="mm", default_threshold=25.0, extract=_extract_precip,
        cmap="YlGnBu", levels=_PRECIP_LEVELS, pmm_in_all=True),
    "t2": FieldSpec(
        name="t2", title="2 m temperature", units="deg C",
        unit_slug="degc", default_threshold=30.0, extract=_extract_t2,
        cmap="RdYlBu_r"),
    "wspd10": FieldSpec(
        name="wspd10", title="10 m wind speed", units="m s-1",
        unit_slug="ms", default_threshold=25.0, extract=_extract_wspd10,
        cmap="viridis"),
}

#: The suite, in the order ``all`` expands to.
PRODUCTS = ("mean", "spread", "prob", "paintball", "pmm", "max")


# ---------------------------------------------------------------------------
# The ensemble mathematics -- pure numpy, no wrfout and no matplotlib.
# ---------------------------------------------------------------------------


def _as_stack(stack, *, nan_policy: str = DEFAULT_NAN_POLICY,
              member_numbers=None) -> np.ndarray:
    array = np.asarray(stack, dtype=float)
    if array.ndim != 3:
        raise ValueError(
            f"member stack must be (members, ny, nx); got shape "
            f"{array.shape}")
    if array.shape[0] < 1:
        raise ValueError("member stack is empty")
    if nan_policy not in NAN_POLICIES:
        raise ValueError(
            f"unknown nan policy {nan_policy!r}; choose from "
            f"{', '.join(NAN_POLICIES)}")
    if nan_policy == "refuse":
        _refuse_nonfinite(array, member_numbers)
    return array


def _refuse_nonfinite(array: np.ndarray, member_numbers=None) -> None:
    bad = ~np.isfinite(array)
    if not bad.any():
        return
    counts = bad.reshape(array.shape[0], -1).sum(axis=1)
    numbers = (list(member_numbers) if member_numbers is not None
               else list(range(array.shape[0])))
    named = ", ".join(
        f"member {numbers[index]}: {int(count)} point(s)"
        for index, count in enumerate(counts) if count)
    raise EnsembleRefusal(
        f"--nan-policy refuse: {int(bad.sum())} non-finite value(s) in the "
        f"member stack ({named}). Nothing was drawn. Use --nan-policy mask "
        f"to exclude them from the reductions and stamp the coverage on "
        f"the panel instead.")


def missingness_report(stack, *, member_numbers=None,
                       nan_policy: str = DEFAULT_NAN_POLICY) -> dict:
    """What is missing from this member stack, and how much is left.

    Computed once per field per valid time and carried into the caption
    and the provenance.  A masked reduction that does not publish its
    denominator is exactly the thing the original propagate-NaN policy
    was right to refuse; publishing it is what makes masking accurate.
    """

    array = _as_stack(stack, nan_policy=nan_policy,
                      member_numbers=member_numbers)
    n_members, ny, nx = array.shape
    finite = np.isfinite(array)
    per_point = finite.sum(axis=0)
    numbers = (list(member_numbers) if member_numbers is not None
               else list(range(n_members)))
    affected = [numbers[index] for index in range(n_members)
                if not finite[index].all()]
    total = int(finite.size)
    return {
        "policy": nan_policy,
        "members": n_members,
        "grid_points": int(ny * nx),
        "nonfinite_values": int(total - int(finite.sum())),
        "members_affected": affected,
        "min_finite_members": int(per_point.min()),
        "fully_missing_points": int((per_point == 0).sum()),
        #: Fraction of the (members x points) cells that carried data.
        "coverage": float(finite.sum() / total) if total else 0.0,
    }


def coverage_caption(report: dict) -> str | None:
    """One line for the panel, or ``None`` when nothing was missing."""

    if not report or not report.get("nonfinite_values"):
        return None
    affected = report.get("members_affected") or []
    shown = ", ".join(str(number) for number in affected[:6])
    if len(affected) > 6:
        shown += f", +{len(affected) - 6} more"
    missing_points = report.get("fully_missing_points", 0)
    tail = ("" if not missing_points
            else f"; {missing_points} point(s) blank (no finite member)")
    return (f"MASKED: {report['nonfinite_values']} non-finite value(s) from "
            f"member(s) {shown}; coverage {100.0 * report['coverage']:.1f}% "
            f"of member-points, min {report['min_finite_members']} of "
            f"{report['members']} members per point{tail}")


def _finite_counts(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(finite-member array with non-finite entries zeroed, counts)``.

    The one place ``mask`` decides what "excluded from the reduction"
    means, so mean and spread cannot drift apart about it.

    ``np.nansum`` is NOT that place, and using it was the bug: it ignores
    NaN and propagates INFINITY.  The policy counts every non-finite value
    as missing -- :func:`missingness_report` classifies with
    ``np.isfinite`` -- so a stack of ``[1.0, 2.0, +Inf]`` reported
    ``nonfinite_values=1`` and ``min_finite_members=2`` on the panel while
    the mean and the spread it stamped that coverage onto were both
    ``+Inf``.  A product that contradicts its own coverage stamp is worse
    than one that has none.

    Zero is the right substitution precisely because the DENOMINATOR is
    the finite count and not the member count: an excluded member
    contributes nothing to the sum and nothing to the divisor.
    """

    finite = np.isfinite(array)
    return np.where(finite, array, 0.0), finite.sum(axis=0)


def ensemble_mean(stack, *, nan_policy: str = DEFAULT_NAN_POLICY,
                  member_numbers=None) -> np.ndarray:
    """Arithmetic mean over the member axis, under the stated NaN policy.

    ``mask`` averages the finite members at each point and leaves points
    with no finite member NaN; the denominator it used is published in
    :func:`missingness_report` and on the panel, which is the whole
    difference between this and a bare ``np.nanmean``.  EVERY non-finite
    value is excluded, infinities included -- see :func:`_finite_counts`.
    ``refuse`` fails the product outright.
    """

    array = _as_stack(stack, nan_policy=nan_policy,
                      member_numbers=member_numbers)
    if nan_policy == "refuse":
        return array.mean(axis=0)
    with np.errstate(invalid="ignore"):
        values, counts = _finite_counts(array)
        out = values.sum(axis=0) / np.where(counts > 0, counts, 1)
        return np.where(counts > 0, out, np.nan)


def ensemble_max(stack, *, nan_policy: str = DEFAULT_NAN_POLICY,
                 member_numbers=None) -> np.ndarray:
    """Member maximum at each point, under the stated NaN policy.

    The "ensemble max" map (max composite reflectivity, max updraft
    helicity).  ``mask`` takes the maximum over the finite members only,
    infinities excluded exactly as in :func:`ensemble_mean`, and leaves a
    point no member has NaN.  ``refuse`` fails the product outright.
    Mirrors ``rustwx_ensemble::ensemble_max``.
    """

    array = _as_stack(stack, nan_policy=nan_policy,
                      member_numbers=member_numbers)
    if nan_policy == "refuse":
        return array.max(axis=0)
    finite = np.isfinite(array)
    out = np.where(finite, array, -np.inf).max(axis=0)
    return np.where(finite.any(axis=0), out, np.nan)


def ensemble_spread(stack, *, ddof: int = 1,
                    nan_policy: str = DEFAULT_NAN_POLICY,
                    member_numbers=None) -> np.ndarray:
    """Ensemble standard deviation -- the SAMPLE one (ddof=1) by default.

    The ensemble is a sample from the forecast distribution, not the
    population, and every spread/skill comparison in the literature is
    the n-1 form.  Two members are the minimum; one member has no
    spread, and reporting 0.0 for it would read as perfect agreement.
    Under ``mask`` that minimum is applied PER POINT: a point with one
    finite member is NaN, not zero.
    """

    array = _as_stack(stack, nan_policy=nan_policy,
                      member_numbers=member_numbers)
    if array.shape[0] <= ddof:
        raise EnsembleRefusal(
            f"ensemble spread with ddof={ddof} needs more than {ddof} "
            f"member(s); this ensemble has {array.shape[0]}")
    if nan_policy == "refuse":
        return array.std(axis=0, ddof=ddof)
    with np.errstate(invalid="ignore", divide="ignore"):
        finite = np.isfinite(array)
        counts = finite.sum(axis=0)
        usable = counts > ddof
        mean = ensemble_mean(array, nan_policy="mask")
        # The deviation is taken only where the member is finite, and the
        # excluded members contribute a hard zero to the sum of squares.
        # ``np.nansum`` over ``(array - mean)**2`` was the same infinity
        # hole as the mean's: ``(+Inf - 1.5)**2`` is ``+Inf``, which
        # np.nansum keeps, so the spread went to infinity at a point the
        # coverage stamp called two-thirds covered.
        deviation = np.where(finite, array - mean[None, :, :], 0.0)
        squares = (deviation ** 2).sum(axis=0)
        variance = squares / np.where(usable, counts - ddof, 1)
        return np.where(usable, np.sqrt(variance), np.nan)


def disc_offsets(radius_cells: float) -> tuple[tuple[int, int], ...]:
    """Integer ``(dy, dx)`` offsets inside a disc of the given radius.

    Nested by construction: every offset inside radius r is inside
    radius R >= r.  That nesting is what makes neighborhood probability
    monotone in the radius, so it is a property of this function and not
    a coincidence of the caller.
    """

    if not np.isfinite(radius_cells) or radius_cells < 0.0:
        raise ValueError(f"radius must be finite and >= 0, got {radius_cells}")
    limit = int(math.floor(radius_cells))
    squared = float(radius_cells) ** 2
    return tuple(
        (dy, dx)
        for dy in range(-limit, limit + 1)
        for dx in range(-limit, limit + 1)
        if dy * dy + dx * dx <= squared)


def _in_domain_offsets(radius_cells: float, ny: int, nx: int):
    """Disc offsets that can reach any in-domain cell of an ny x nx grid.

    An offset further out than the domain is wide produces a source slice
    and a destination slice of different shapes -- one empty, one not --
    and ``np.maximum`` raised ``operands could not be broadcast together
    with shapes (0,5) (2,5) (0,5)``.  A radius larger than the domain is a
    legitimate request (it means "the whole domain"), so the disc is
    CLIPPED rather than the request refused.
    """

    return tuple((dy, dx) for dy, dx in disc_offsets(radius_cells)
                 if abs(dy) < ny and abs(dx) < nx)


def neighborhood_footprint(shape, radius_cells: float) -> np.ndarray:
    """How many IN-DOMAIN cells each point's disc actually covers.

    Exists so "the maximum is over the clipped disc" is a checkable
    structural claim and not only an assertion about output values: for a
    maximum operator, replicating an edge value outward can never change
    an in-domain result, so no comparison of maxima can tell clipping from
    replicate padding.  Counting the footprint can, and does.  It is also
    what the masked NMEP reports as its neighborhood coverage.
    """

    ny, nx = (int(shape[0]), int(shape[1]))
    out = np.zeros((ny, nx), dtype=int)
    if radius_cells <= 0.0:
        return out + 1
    for dy, dx in _in_domain_offsets(radius_cells, ny, nx):
        dst = (slice(max(0, -dy), ny - max(0, dy)),
               slice(max(0, -dx), nx - max(0, dx)))
        out[dst] += 1
    return out


def neighborhood_max(field, radius_cells: float, *,
                     nan_policy: str = DEFAULT_NAN_POLICY) -> np.ndarray:
    """Maximum over a disc of ``radius_cells`` around every grid point.

    Pure numpy on purpose: ``scipy.ndimage.maximum_filter`` would be
    faster, and scipy is not a declared dependency of this project.
    Beyond the domain edge there is no data, so the maximum is taken over
    the in-domain part of the disc only -- NOT over edge values replicated
    outward, which would invent exceedances in the boundary rows.

    Under ``mask`` a non-finite cell contributes nothing to its
    neighbours' maxima and a point whose whole clipped disc is non-finite
    comes back NaN.  Letting ``np.maximum`` propagate one NaN across a
    neighbourhood, as it did, turned a hit into a miss at every point that
    could see the bad cell.
    """

    array = np.asarray(field, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"field must be 2D; got shape {array.shape}")
    if radius_cells <= 0.0:
        return array.copy()
    ny, nx = array.shape
    masking = nan_policy != "refuse"
    source = np.where(np.isfinite(array), array, -np.inf) if masking \
        else array
    out = np.full(array.shape, -np.inf, dtype=float)
    for dy, dx in _in_domain_offsets(radius_cells, ny, nx):
        dst = (slice(max(0, -dy), ny - max(0, dy)),
               slice(max(0, -dx), nx - max(0, dx)))
        src = (slice(max(0, dy), ny - max(0, -dy)),
               slice(max(0, dx), nx - max(0, -dx)))
        np.maximum(out[dst], source[src], out=out[dst])
    if masking:
        # -inf survives only where every in-disc cell was non-finite, and
        # that is genuinely "no data here", not "very small".
        out = np.where(np.isneginf(out), np.nan, out)
    return out


def exceedance_probability(stack, threshold: float, *,
                           radius_cells: float = 0.0,
                           nan_policy: str = DEFAULT_NAN_POLICY,
                           member_numbers=None) -> np.ndarray:
    """Fraction of members exceeding ``threshold``; strictly greater.

    With a radius, this is the neighborhood-maximum ensemble probability
    (NMEP): each member is first reduced to its neighborhood maximum, and
    the ensemble fraction is taken of THAT.  The order matters -- taking
    the neighborhood maximum of the probability field instead would let
    one member's hit be reported at full ensemble confidence.

    Under ``mask`` the DENOMINATOR at a point is the number of members
    that have a finite value AT THAT POINT -- its own value, not its
    neighborhood's.  Two things follow, and both are deliberate:

    * counting a non-finite member as a non-exceedance against the full
      roster understated the probability, and because ``np.maximum`` then
      spread the NaN over the neighborhood it turned hits into misses:
      the probe ``[10, NaN, 0]`` at threshold 5 fell from ``1`` at radius
      0 to ``0`` at radius 1;
    * fixing the voting roster to the point rather than to the disc keeps
      the denominator independent of the radius, which is what makes the
      advertised structural monotonicity EXACT rather than approximate.
      A denominator that grew with the radius would let a probability
      fall as the radius rose even with the masking right, and this
      product's one structural promise would still be false.

    So: a member with no valid forecast at a grid point does not vote on
    that grid point at any radius, and a point where no member votes is
    NaN at every radius.
    """

    array = _as_stack(stack, nan_policy=nan_policy,
                      member_numbers=member_numbers)
    reduced = np.empty(array.shape, dtype=float)
    for index in range(array.shape[0]):
        member = array[index]
        if radius_cells > 0.0:
            member = neighborhood_max(member, radius_cells,
                                      nan_policy=nan_policy)
        reduced[index] = member
    if nan_policy == "refuse":
        return (reduced > threshold).mean(axis=0)
    votes = np.isfinite(array)
    exceeds = np.where(np.isfinite(reduced), reduced, -np.inf) > threshold
    hits = (votes & exceeds).sum(axis=0)
    counts = votes.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, hits / np.where(counts > 0, counts, 1),
                        np.nan)


def probability_matched_mean(stack, *,
                             nan_policy: str = DEFAULT_NAN_POLICY,
                             tie_rule: str = DEFAULT_PMM_TIE_RULE,
                             member_numbers=None) -> np.ndarray:
    """Exact probability-matched mean (Ebert 2001).

    Three steps, no approximation:

    1. the ensemble mean supplies the spatial PATTERN -- where the
       features are, which is what averaging gets right;
    2. every member value is pooled and sorted, and every M-th value is
       taken (M = member count), yielding exactly one value per grid
       point drawn from the pooled intensity distribution -- which is
       what averaging destroys;
    3. the pooled values are laid onto the grid in the mean's rank order:
       the highest goes to the mean's highest point, and so on down.

    The result has the mean's structure and the members' amplitude
    statistics.  Ties in the mean resolve by :data:`PMM_TIE_RULES`.

    **Missingness stays where it is.**  A point with no finite member is
    NaN in the output and takes no pooled value; the pooled distribution
    is drawn from the finite values only, and the every-M-th stride
    becomes "n_assignable values spread evenly through the finite pool",
    which reduces to Ebert's exact stride when nothing is missing.  The
    previous implementation sorted NaNs to the front of the reversed pool
    and to the back of the mean's descending order, so it assigned the
    pooled NaN to the *highest finite mean point*: a two-member probe of
    ``[[100, 1, NaN], [90, 2, 0]]`` returned ``[NaN, 90, 1]``, erasing the
    panel's strongest feature and filling the actually-invalid point with
    a finite number.
    """

    if tie_rule not in PMM_TIE_RULES:
        raise ValueError(
            f"unknown pmm tie rule {tie_rule!r}; choose from "
            f"{', '.join(PMM_TIE_RULES)}")
    array = _as_stack(stack, nan_policy=nan_policy,
                      member_numbers=member_numbers)
    mean = ensemble_mean(array, nan_policy=nan_policy)
    flat_mean = mean.reshape(-1)
    assignable = np.isfinite(flat_mean)
    n_assign = int(assignable.sum())
    out = np.full(flat_mean.size, np.nan, dtype=float)
    if n_assign == 0:
        return out.reshape(mean.shape)

    values = array.reshape(-1)
    pool = np.sort(values[np.isfinite(values)])[::-1]
    if pool.size == 0:
        return out.reshape(mean.shape)
    # Evenly spaced draws from the pooled distribution: with a complete
    # stack this is exactly indices 0, M, 2M, ... of the descending pool.
    picks = np.floor(np.arange(n_assign) * (pool.size / n_assign)
                     ).astype(int)
    picked = pool[np.clip(picks, 0, pool.size - 1)]

    finite_index = np.flatnonzero(assignable)
    finite_mean = flat_mean[finite_index]
    order = np.argsort(-finite_mean, kind="stable")
    if tie_rule == "average":
        ranked = finite_mean[order]
        # Group boundaries of equal means in the descending order.
        starts = np.flatnonzero(
            np.concatenate(([True], ranked[1:] != ranked[:-1])))
        group = np.zeros(ranked.size, dtype=int)
        group[starts] = 1
        group = np.cumsum(group) - 1
        sums = np.bincount(group, weights=picked, minlength=group.max() + 1)
        sizes = np.bincount(group, minlength=group.max() + 1)
        picked = (sums / sizes)[group]
    out[finite_index[order]] = picked
    return out.reshape(mean.shape)


def pmm_tie_report(stack, *, nan_policy: str = DEFAULT_NAN_POLICY) -> dict:
    """How much of the mean field is a plateau, and how big the worst is.

    A flat-index tie break is deterministic but paints an artificial
    row-major gradient across every plateau, so a reader needs to know
    whether the panel has one.  Reported rather than silently tolerated.
    """

    array = _as_stack(stack, nan_policy=nan_policy)
    flat = ensemble_mean(array, nan_policy=nan_policy).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"tied_points": 0, "largest_tie_group": 0,
                "tied_fraction": 0.0}
    _, counts = np.unique(finite, return_counts=True)
    tied = int(counts[counts > 1].sum())
    return {
        "tied_points": tied,
        "largest_tie_group": int(counts.max()),
        "tied_fraction": float(tied / finite.size),
    }


def member_color(member_number: int) -> str:
    """The paintball colour for a member NUMBER -- not for its position.

    Keyed to the member's own identity so a plot of members 1..30 and a
    plot of members 3, 7, 11 give member 7 the same colour.  Colour is
    the only label a paintball plot has; if it moved when the roster was
    filtered, comparing two paintball plots would be actively misleading.
    """

    return PAINTBALL_PALETTE[int(member_number) % len(PAINTBALL_PALETTE)]


# ---------------------------------------------------------------------------
# Filename tokens
# ---------------------------------------------------------------------------


def number_slug(value: float) -> str:
    """A filename-safe number: ``40``, ``42p5``, ``m10``.

    ``.`` becomes ``p`` so the token cannot be misread as an extension,
    and a leading ``-`` becomes ``m`` so it cannot be misread as the next
    token.
    """

    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def threshold_slug(value: float, unit_slug: str) -> str:
    """``40`` + ``dbz`` -> ``40dbz``."""

    return f"{number_slug(value)}{unit_slug}"


def radius_slug(radius_km: float | None) -> str | None:
    """``5`` -> ``r5km``; no neighborhood -> no token at all.

    A ``-r0km`` on a plain point probability would claim a neighborhood
    of zero radius was applied, which is true and useless; the absence of
    the token is the clearer statement.
    """

    if radius_km is None or radius_km <= 0.0:
        return None
    return f"r{number_slug(radius_km)}km"


def ensemble_token(product: str, *, threshold: float | None = None,
                   unit_slug: str | None = None,
                   radius_km: float | None = None) -> str:
    """The ensemble half of the filename's product slot.

    ``ens-mean``, ``ens-spread``, ``ens-pmm``, ``ens-max``, ``ens-p40dbz``,
    ``ens-p40dbz-r5km``, ``ens-paintball40dbz``.
    """

    if product in ("mean", "spread", "pmm", "max"):
        return f"ens-{product}"
    if threshold is None or unit_slug is None:
        raise ValueError(
            f"product {product!r} is threshold-based; it has no filename "
            f"token without a threshold and its unit")
    marker = threshold_slug(threshold, unit_slug)
    if product == "prob":
        token = f"ens-p{marker}"
        radius = radius_slug(radius_km)
        return token if radius is None else f"{token}-{radius}"
    if product == "paintball":
        return f"ens-paintball{marker}"
    raise ValueError(f"unknown ensemble product {product!r}")


#: The pixel size the rust ensemble engine draws at by default, and the
#: ``--dpi`` that is the same picture on the matplotlib route.  The pair
#: is the anchor :func:`panel_pixels` scales from, so at the default the
#: two routes agree byte for byte with every earlier release.
_ENGINE_PANEL_PIXELS = (1200, 900)
_REFERENCE_DPI = 150.0

#: The smallest panel worth drawing, the same floor
#: ``woof.render.parse_size`` applies to a pixel request.
_MIN_PANEL_PIXELS = (320, 240)


def panel_pixels(dpi: float) -> tuple[int, int]:
    """``--dpi`` as the pixel size both engines draw, one meaning.

    ``--dpi`` used to mean a figure resolution on one route and nothing
    at all on the other, where it was accepted and dropped with a
    warning.  A user's request had an implementation and no door.

    The mapping is anchored on the pair the two defaults already assert
    are the same picture: ``--dpi 150`` and the engine's 1200x900.  At
    that dpi the output is byte-identical to every earlier release;
    above and below it scales linearly, and the floor is named when it
    bites rather than silently applied.
    """

    scale = float(dpi) / _REFERENCE_DPI
    width = max(_MIN_PANEL_PIXELS[0], round(_ENGINE_PANEL_PIXELS[0] * scale))
    height = max(_MIN_PANEL_PIXELS[1], round(_ENGINE_PANEL_PIXELS[1] * scale))
    if (width, height) == _MIN_PANEL_PIXELS and scale * _ENGINE_PANEL_PIXELS[0] < _MIN_PANEL_PIXELS[0]:
        warn(f"--dpi {float(dpi):g} asks for a panel smaller than "
             f"{_MIN_PANEL_PIXELS[0]}x{_MIN_PANEL_PIXELS[1]} px, which is "
             "the floor this tree draws at; the panels are that size.")
    return width, height


def delivered_segments(field_name: str, token: str, domain: str,
                       stamp: str) -> tuple[str, str, str]:
    """``(domain, product, valid day)`` for one ensemble panel.

    The 2026-08-06 folder ruling's three segments, spelled once for both
    ensemble engines.  The product segment is the same
    ``<field>-<ensemble token>`` the filename carries, so a reader sees
    one name in the folder and in the file rather than two spellings of
    one product.

    An unreadable stamp files under :data:`woof.render_layout.UNDATED`
    rather than under a guess, exactly as the deterministic route does.
    """

    from woof import render_layout

    return (domain or render_layout.NATIVE_GRID, f"{field_name}-{token}",
            render_layout.valid_day(stamp) or render_layout.UNDATED)


def delivered_path(outdir, field_name: str, token: str, domain: str,
                   stamp: str):
    """Where one ensemble panel belongs under ``outdir``."""

    from woof import render_layout

    domain_segment, product, day = delivered_segments(
        field_name, token, domain, stamp)
    return render_layout.place(
        outdir, domain=domain_segment, product=product, day=day,
        filename=product_filename(field_name, token, domain, stamp))


def product_filename(field_name: str, token: str, domain: str,
                     stamp: str) -> str:
    """``refl-ens-p40dbz-r5km_d02-3km_1974-04-03_18-00-00.png``.

    v1.0.1's ``{product}_{domain-token}_{stamp}.png`` with the ensemble
    token folded into the product slot, so the ``_`` separators keep
    meaning exactly what they mean for the deterministic suite.
    """

    return f"{field_name}-{token}_{domain}_{_stamp_for_filename(stamp)}.png"


def ensemble_plot_context(domain: str | None, spacing_m: float | None,
                          stamp: str, source_label: str,
                          n_members: int, *, notes=()) -> str:
    """v1.0.1's subtitle plus the facts an ensemble product must carry.

    TWO lines minimum: the v1.0.1 subtitle with the member count
    appended, then the experimental stamp on its own.  One line carried
    all three and ran off the right edge of an 8-inch panel at 30 members
    -- a stamp that is clipped out of the figure is not a stamp, and its
    own line is also the prominence it should have had from the start.

    ``notes`` adds a line each for the two things that make a panel mean
    something different from what it looks like: masked missing members,
    and a roster widened past the default accepted statuses.  Both used
    to be invisible on the graphic, which is where a reader meets it.
    """

    base = plot_context(domain, spacing_m, stamp, source_label)
    lines = [f"{base} | {n_members} members", experimental_stamp()]
    lines.extend(note for note in notes if note)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnsembleMember:
    number: int
    directory: Path
    status: str
    seed: int | None = None
    provenance: dict = dataclass_field(default_factory=dict)
    #: ``wrfout_count`` as the writer recorded it, when it did.  ``None``
    #: for a member the writer never completed.  Kept as a cross-check on
    #: the inventory's length; it is NOT what admits a member, because one
    #: stale file replacing one real file satisfies a count exactly.
    declared_wrfout_count: int | None = None
    #: The per-file frame inventory the writer recorded
    #: (:data:`woof.ensemble.wrfout_inventory.WRFOUT_INVENTORY_CONTRACT`),
    #: or ``None`` for a member that declared none.  This is what an
    #: overridden member is bound against, and a member with ``None`` here
    #: is refused rather than rendered.
    declared_inventory: tuple | None = None
    #: True when this member is here only because the operator widened
    #: ``--accept-status`` past the default.
    overridden: bool = False
    #: The error the writer recorded for a FAILED member, if any.
    error_type: str | None = None


@dataclass(frozen=True)
class EnsembleManifest:
    root: Path
    schema: str
    n_members: int
    members: tuple[EnsembleMember, ...]
    #: The manifest's own top-level status.  Read because
    #: ``--accept-status`` names MEMBER statuses and says nothing about
    #: whether the ensemble as a whole ever finished.
    status: str | None = None

    @property
    def overridden(self) -> tuple[EnsembleMember, ...]:
        return tuple(member for member in self.members if member.overridden)


def _schema_of(document: dict) -> str | None:
    for key in _SCHEMA_KEYS:
        value = document.get(key)
        if isinstance(value, str):
            return value
    return None


def _member_number(record: dict) -> int | None:
    """The member's own number, from whichever field carries it.

    Strict about the VALUE, tolerant about the key: the sibling lane owns
    the writer and this consumer will not fail an otherwise-good roster
    over the difference between ``member`` and ``id``.  The directory
    name is the last resort because ``member_007`` is the layout the
    ensemble is defined by.
    """

    for key in ("member", "id", "index", "number"):
        value = record.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
            return int(value)
    for key in _DIRECTORY_KEYS:
        value = record.get(key)
        if isinstance(value, str):
            digits = "".join(ch for ch in Path(value).name if ch.isdigit())
            if digits:
                return int(digits)
    return None


def manifest_path(root, domain: str | None = None) -> Path:
    """The ensemble manifest ``root`` names, wherever its writer put it.

    ``root/ensemble-manifest.json`` (v1, and a v2 domain folder named
    directly), else the one v2 domain manifest under a ``woof go
    --members`` run folder (``root/<domain>/ensemble-manifest.json``, the
    folder go prints as "products under ...").  Several domains are
    refused unless ``domain`` picks one, naming them, because reducing one
    of them silently would answer for a nest nobody asked about.  When
    nothing is found the v1 path is returned, which the caller refuses.

    A manifest one folder down is a domain only when it declares
    :data:`V2_MANIFEST_SCHEMA`.  A v1 manifest there is another ensemble
    root (a DA cycle's ``cycle_NNN/``, written by
    :mod:`woof.ensemble.cycle`) whose ``member_dir`` entries are
    relative to ITS folder: read from here it would resolve its members
    against the outer folder, and two cycles would be offered as two
    domains with ``--domain`` then filtering wrfout names by a cycle
    folder.  Such a root is named directly, as before.
    """

    root = Path(root)
    direct = root / MANIFEST_FILENAME
    if direct.is_file():
        return direct
    # go's printed output folder wraps the forecast under run/ or chain/run/.
    # Only those declared layouts are followed, never arbitrary ensembles
    # such as a DA cycle or a member folder.
    for relative in ("run", "chain/run"):
        forecast = root / relative
        if ((forecast / MANIFEST_FILENAME).is_file()
                or any(_manifest_schema(path) == V2_MANIFEST_SCHEMA
                       for path in forecast.glob(f"*/{MANIFEST_FILENAME}"))):
            return manifest_path(forecast, domain)
    found = sorted(path for path in root.glob(f"*/{MANIFEST_FILENAME}")
                   if path.is_file()
                   and _manifest_schema(path) == V2_MANIFEST_SCHEMA)
    if domain is not None:
        chosen = root / domain / MANIFEST_FILENAME
        if chosen in found:
            return chosen
        if found:
            raise EnsembleRefusal(
                f"--domain {domain}: {root} holds ensemble manifests for "
                f"{', '.join(path.parent.name for path in found)} only")
        return direct
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise EnsembleRefusal(
            f"{root} holds one ensemble manifest per domain "
            f"({', '.join(path.parent.name for path in found)}); name the "
            "one to reduce with --domain, since an ensemble product of "
            "mixed nests is a picture of nothing")
    return direct


def load_manifest(root, *, accept_status=DEFAULT_ACCEPT_STATUS,
                  domain: str | None = None) -> EnsembleManifest:
    """Read and VALIDATE the ensemble manifest under ``root``.

    Every failure is an :class:`EnsembleRefusal` naming what is wrong and
    which member it is wrong for.  All member problems are collected and
    reported together: an operator fixing a 30-member ensemble should
    learn about all six broken members in one run, not in six.  Both
    schemas are read: ``gpuwm-ensemble-manifest.v1`` and the
    ``gpuwm-ensemble-output.v2`` that ``woof go --members`` writes
    (:func:`_load_v2_manifest`).
    """

    root = Path(root)
    path = manifest_path(root, domain)
    if not path.is_file():
        raise EnsembleRefusal(
            f"no ensemble manifest at {path}.  woof enprod reads the "
            f"roster from the manifest and never guesses it from the "
            f"directory listing, because a member that failed to run "
            f"leaves no directory and would silently shrink the "
            f"ensemble.  Write one of schema {MANIFEST_SCHEMA}, or "
            f"generate a synthetic ensemble to try the suite against "
            f"with: woof enprod --make-fixture {root}")
    # Member paths in v1 are relative to the manifest, including when
    # the outer go output folder was supplied. V2 uses the same basis.
    root = path.parent
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EnsembleRefusal(f"{path}: unreadable manifest ({exc})") from exc
    if not isinstance(document, dict):
        raise EnsembleRefusal(
            f"{path}: manifest must be a JSON object, got "
            f"{type(document).__name__}")

    schema = _schema_of(document)
    if schema == V2_MANIFEST_SCHEMA:
        return _load_v2_manifest(path, document)
    if schema is None:
        warn(f"{path.name} declares no schema string; reading it as "
             f"{MANIFEST_SCHEMA} (every field used is still validated)",
             why=f"Looked under {', '.join(_SCHEMA_KEYS)}; top-level "
                 f"keys present: "
                 f"{', '.join(sorted(document)) or '(none)'}.")
        schema = MANIFEST_SCHEMA
    elif schema != MANIFEST_SCHEMA:
        warn(f"{path.name} declares schema {schema!r}; this build reads "
             f"{MANIFEST_SCHEMA!r} -- proceeding, since every field "
             "used is still validated per member")

    records = document.get("members")
    if not isinstance(records, list):
        raise EnsembleRefusal(
            f"{path}: 'members' must be a list of member records, got "
            f"{type(records).__name__}")
    raw_count = document.get("n_members")
    if not isinstance(raw_count, int) or isinstance(raw_count, bool) \
            or raw_count < 1:
        warn(f"{path.name}: n_members is {raw_count!r}; using the "
             f"member list itself ({len(records)} record(s))")
        raw_count = len(records)
    elif len(records) != raw_count:
        warn(f"{path.name}: n_members={raw_count} but the roster lists "
             f"{len(records)} record(s); the roster is authoritative "
             "and every panel stamps the count actually averaged")
        raw_count = len(records)

    accepted = tuple(accept_status)
    members: list[EnsembleMember] = []
    problems: list[str] = []
    skipped: list[tuple[int, str]] = []
    seen_numbers: dict[int, int] = {}
    statuses_seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            problems.append(
                f"members[{index}]: not an object ({type(record).__name__})")
            continue
        number = _member_number(record)
        if number is None:
            problems.append(
                f"members[{index}]: no member number; expected one of "
                f"member/id/index/number, or a dir/path naming one")
            continue
        if number in seen_numbers:
            problems.append(
                f"members[{index}]: member {number} is already declared "
                f"at members[{seen_numbers[number]}]")
            continue
        seen_numbers[number] = index
        raw_dir = None
        for key in _DIRECTORY_KEYS:
            value = record.get(key)
            if isinstance(value, str) and value:
                raw_dir = value
                break
        directory = (root / raw_dir) if raw_dir is not None \
            else (root / f"member_{number:03d}")
        status = record.get("status")
        if not isinstance(status, str):
            problems.append(
                f"member {number}: no 'status' string in the manifest "
                f"record; a member of unstated status is not a member "
                f"this suite will average")
            continue
        statuses_seen.add(status)
        if status not in accepted:
            skipped.append((number, status))
            continue
        if not directory.is_dir():
            problems.append(
                f"member {number}: manifest says {status!r} but its run "
                f"directory {directory} does not exist")
            continue
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            seed = None
        provenance = record.get("provenance")
        declared = record.get("wrfout_count")
        if isinstance(declared, bool) or not isinstance(declared, int):
            declared = None
        try:
            inventory = read_member_inventory(record)
        except ValueError as exc:
            problems.append(f"member {number}: {exc}")
            continue
        error = record.get("error")
        members.append(EnsembleMember(
            number=number, directory=directory, status=status, seed=seed,
            provenance=dict(provenance) if isinstance(provenance, dict)
            else {},
            declared_wrfout_count=declared,
            declared_inventory=inventory,
            overridden=status not in DEFAULT_ACCEPT_STATUS,
            error_type=(error.get("type") if isinstance(error, dict)
                        else None)))

    if problems:
        # Structural problems -- unreadable records, duplicate member
        # numbers, missing directories for accepted members -- stay a
        # refusal: they mean the manifest misdescribes what is on disk.
        raise EnsembleRefusal(
            f"{path}: {len(problems)} of {raw_count} declared member(s) "
            f"are unusable, so this is not the {raw_count}-member "
            f"ensemble the manifest claims:\n  "
            + "\n  ".join(problems))
    if skipped:
        # A member whose status is simply not accepted is DROPPED with
        # one warning line; the panels stamp the count actually
        # averaged, so a thinner ensemble is disclosed, never silent.
        rejected = sorted({status for _, status in skipped})
        warn(f"skipping {len(skipped)} member(s) whose status is not "
             f"accepted ({', '.join(repr(s) for s in rejected)}); "
             f"rendering the remaining {len(members)} -- accepting "
             "them takes --accept-status "
             f"{','.join(sorted(set(accepted) | set(rejected)))}")
    if not members:
        raise EnsembleRefusal(
            f"{path}: no usable member remains (declared {raw_count}, "
            f"skipped {len(skipped)} by status); there is nothing to "
            "ensemble")

    document_status = document.get("status")
    return EnsembleManifest(
        root=root, schema=schema, n_members=len(members),
        members=tuple(sorted(members, key=lambda m: m.number)),
        status=document_status if isinstance(document_status, str) else None)


def _load_v2_manifest(path: Path, document: dict) -> EnsembleManifest:
    """The roster of a ``gpuwm-ensemble-output.v2`` domain manifest.

    The members are ``member_order``; each one's histories are the
    ``member_files`` rows for this domain, relative to the run folder (the
    manifest's grandparent), and each must exist at its recorded size.  A
    run that kept no histories (``keep_member_files`` false, the default)
    is refused naming where its own aggregate maps are and the flag that
    keeps the histories: there is nothing here to reduce, and a member
    missing its files would be a smaller ensemble than the one the run
    describes.  The writer records no member status; a member is in the
    roster because the run listed it, and its files are what admit it.
    """

    domain_dir = path.parent
    run_root = domain_dir.parent
    domain = domain_dir.name
    order = document.get("member_order")
    if (not isinstance(order, list) or not order
            or any(isinstance(m, bool) or not isinstance(m, int)
                   for m in order)
            or len(set(order)) != len(order)):
        raise EnsembleRefusal(
            f"{path}: 'member_order' must list each member id once, got "
            f"{order!r}")
    files = document.get("member_files")
    if not document.get("keep_member_files") or not files:
        maps = run_root / "maps" / domain
        raise EnsembleRefusal(
            f"{path}: this {len(order)}-member ensemble kept no member "
            f"histories (schema {V2_MANIFEST_SCHEMA}, keep_member_files "
            f"{bool(document.get('keep_member_files'))}), so there is "
            "nothing for enprod to reduce.  The run drew its own mean, "
            f"spread, probability and paintball maps under {maps}.  To "
            "reduce the members here, run the ensemble again with woof "
            "go --members N --keep-member-files")
    if not isinstance(files, list):
        raise EnsembleRefusal(
            f"{path}: 'member_files' must be a list of file records")
    by_member: dict[int, list[Path]] = {member: [] for member in order}
    problems: list[str] = []
    missing_histories = False
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            problems.append(f"member_files[{index}]: not an object")
            continue
        if record.get("domain", domain) != domain:
            continue
        member = record.get("member_id")
        relative = record.get("path")
        if member not in by_member or not isinstance(relative, str):
            problems.append(
                f"member_files[{index}]: member {member!r} is not in "
                f"member_order {order} or has no path")
            continue
        file_path = run_root / relative
        if not file_path.is_file():
            missing_histories = True
            problems.append(
                f"member {member}: its history {file_path} is missing")
            continue
        recorded = record.get("bytes")
        if isinstance(recorded, int) and not isinstance(recorded, bool) \
                and file_path.stat().st_size != recorded:
            problems.append(
                f"member {member}: {file_path} holds "
                f"{file_path.stat().st_size} bytes where the run recorded "
                f"{recorded}")
            continue
        by_member[member].append(file_path)
    counts = {member: len(paths) for member, paths in by_member.items()}
    for member, count in counts.items():
        if count == 0:
            problems.append(f"member {member}: no {domain} history recorded")
    expected = max(counts.values())
    short = sorted(member for member, count in counts.items()
                   if 0 < count < expected)
    if short:
        problems.append(
            f"member(s) {', '.join(str(m) for m in short)} recorded fewer "
            f"{domain} histories than the others ({expected}), so the "
            "ensemble does not hold every member at every time")
    members = []
    for member, paths in sorted(by_member.items()):
        directories = sorted({file_path.parent for file_path in paths})
        if len(directories) > 1:
            problems.append(
                f"member {member}: its histories span {len(directories)} "
                "directories, and a member is read from one")
            continue
        if directories:
            members.append(EnsembleMember(
                number=member, directory=directories[0], status="DONE",
                declared_wrfout_count=len(paths)))
    if problems:
        remedy = (
            "\nRestore the missing member history files, or run the ensemble "
            "again with woof go --members N --keep-member-files and retain "
            "those files until enprod has finished. The existing aggregate "
            f"maps are under {run_root / 'maps' / domain}."
            if missing_histories else "")
        raise EnsembleRefusal(
            f"{path}: {len(problems)} problem(s), so this is not the "
            f"{len(order)}-member ensemble the run describes:\n  "
            + "\n  ".join(problems) + remedy)
    return EnsembleManifest(
        root=run_root, schema=V2_MANIFEST_SCHEMA, n_members=len(members),
        members=tuple(members), status=None)


def engine_roster(manifest: EnsembleManifest, store_root: Path) -> Path:
    """A v1 roster of ``manifest`` for ``rw_ensbatch``, in ``store_root``.

    The engine reads ``gpuwm-ensemble-manifest.v1`` (``member`` and
    ``member_dir`` per record); a v2 roster resolved above is handed to it
    in that shape, every member directory absolute and every status the
    engine's default ``DONE``, so the engine reduces exactly the members
    this module validated.
    """

    target = Path(store_root) / "roster" / MANIFEST_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "schema": MANIFEST_SCHEMA,
        "n_members": manifest.n_members,
        "members": [{"member": member.number, "status": "DONE",
                     "member_dir": str(Path(member.directory).resolve())}
                    for member in manifest.members],
    }, indent=2) + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# The status override: members, not bytes
# ---------------------------------------------------------------------------


def _member_inventory_problems(member: EnsembleMember,
                               entry: MemberFrames | None) -> tuple:
    """``(problems, detail)`` for one overridden member's frame inventory.

    Seven bindings, all of them required, none of them merely a count:

    1. every record is a well-formed
       :data:`woof.ensemble.wrfout_inventory.WRFOUT_INVENTORY_CONTRACT`
       entry -- exact contract, canonical member-relative path, a domain
       agreeing with its own filename, non-negative integer size, a real
       sha256 digest, and unique valid frame indices;
    2. no path is declared twice, because every check downstream keys on
       the path and a dict keeps whichever record came last;
    3. every file the manifest declared is present at its declared
       relative path, at its declared size, and hashes to its declared
       sha256 -- the check a stale replacement cannot pass;
    4. every frame the suite indexed comes from a file the manifest
       declared -- so an extra wrfout dropped into the member directory is
       a refusal and not an extra member-time in the mean;
    5. every (valid time -> frame index) pair the suite resolved is the
       pair the manifest recorded for that file, AND every frame declared
       for the indexed domain was resolved -- the two directions together
       are a bijection.  Only the first direction existed, so an inventory
       could declare a frame at index 99 in 2099 that nothing on disk
       carries and still verify: ``declared_frames=2, bound_frames=1``
       was reported as VERIFIED;
    6. at least one declared file supplied at least one indexed frame that
       satisfied all of those checks -- verification is evidence, not the
       vacuous absence of a counterexample in an empty loop;
    7. the declared ``wrfout_count``, where the writer recorded one,
       agrees with the length of the declared inventory.

    The bijection is scoped to the domain the suite INDEXED, because
    :func:`index_member_frames` selects one domain and a member that
    legitimately wrote d01 and d02 histories declares both.  Scoping it
    to the indexed domain is what makes "every declared frame was bound"
    a statement about this product rather than about the run.
    """

    declared = member.declared_inventory
    detail = {
        "member": member.number,
        "status": member.status,
        "error_type": member.error_type,
        "declared_wrfout_count": member.declared_wrfout_count,
        "declared_wrfout_files": None if declared is None else len(declared),
        "declared_frames": None,
        "bound_files": 0,
        "bound_frames": 0,
    }
    if declared is None:
        return ([
            f"member {member.number} (status {member.status!r}) is accepted "
            f"by --accept-status, but its manifest record declares no "
            f"'{WRFOUT_INVENTORY_KEY}'.  There is nothing to bind its files "
            f"to, so admitting it would put unverified bytes into every "
            f"product.  A member the writer never finished has no inventory "
            f"BECAUSE it never finished; render the ensemble without it"
        ], detail)

    problems: list[str] = []
    by_relative: dict[str, dict] = {}
    for record in declared:
        problems.extend(
            f"member {member.number}: {problem}"
            for problem in verify_inventory_entry(
                record, member_dir=member.directory))
        by_relative[str(record.get("path"))] = record
    for repeated in inventory_duplicate_paths(declared):
        problems.append(
            f"member {member.number}: the manifest inventory declares "
            f"{repeated} more than once. Every check downstream keys on the "
            f"path, so only one of the records would ever be bound and "
            f"which one depends on list order")
    declared_frames = 0
    for record in declared:
        try:
            declared_frames += len(inventory_entry_frames(record))
        except ValueError as exc:
            problems.append(f"member {member.number}: {exc}")
    detail["declared_frames"] = declared_frames

    if member.declared_wrfout_count is not None \
            and member.declared_wrfout_count != len(declared):
        # The per-file sha256 binding below, not this redundant scalar,
        # is what proves identity -- report the drift and continue.
        warn(f"member {member.number}: manifest wrfout_count="
             f"{member.declared_wrfout_count} disagrees with its "
             f"{len(declared)}-file inventory; the per-file inventory "
             "binding is authoritative")

    found = {} if entry is None else entry.frames
    bound: set[tuple] = set()
    # The domain the suite actually indexed, read off the frames it
    # resolved rather than taken on trust from the reader: one spelling of
    # "which domain is this" (wrfout_inventory.domain_token) on both sides
    # of the comparison, so the bijection cannot be satisfied by two
    # functions disagreeing about a filename.
    indexed_domains = {inventory_domain_token(path)
                       for path, _ in found.values()}
    for stamp, (path, index) in sorted(found.items()):
        try:
            relative = inventory_relative_path(path, member.directory)
        except ValueError as exc:
            problems.append(f"member {member.number}: {exc}")
            continue
        record = by_relative.get(relative)
        if record is None:
            problems.append(
                f"member {member.number}: frame {stamp} was read from "
                f"{relative}, which the manifest inventory does not "
                f"declare.  An undeclared wrfout under an admitted member "
                f"is exactly the stale-run case --accept-status must not "
                f"average")
            continue
        try:
            frames = inventory_entry_frames(record)
        except ValueError:
            continue                      # already reported above
        if stamp not in frames:
            problems.append(
                f"member {member.number}: {relative} carries valid time "
                f"{stamp}, which the manifest inventory does not declare "
                f"for it")
            continue
        if frames[stamp] != index:
            problems.append(
                f"member {member.number}: {relative} has {stamp} at frame "
                f"{index}; the manifest inventory recorded frame "
                f"{frames[stamp]}")
            continue
        bound.add((relative, stamp))
        detail["bound_frames"] += 1

    # The other direction.  Proving every FOUND frame was declared leaves
    # an inventory free to declare frames nothing on disk carries; the
    # probe declared one real frame plus a phantom at index 99 in 2099 and
    # the member verified with declared_frames=2, bound_frames=1.
    unbound: list[str] = []
    for record in declared:
        relative = record.get("path")
        if not isinstance(relative, str):
            continue                      # already reported by the schema
        if indexed_domains and record.get("domain") not in indexed_domains:
            continue                      # a domain this product did not draw
        try:
            frames = inventory_entry_frames(record)
        except ValueError:
            continue                      # already reported above
        for stamp in sorted(frames):
            if (relative, stamp) not in bound:
                unbound.append(f"{relative}@{stamp}")
    if unbound:
        problems.append(
            f"member {member.number}: the manifest inventory declares "
            f"{len(unbound)} frame(s) the suite did not resolve from the "
            f"files on disk ({', '.join(sorted(unbound))}). The binding is "
            f"a bijection or it is not a binding: an inventory free to "
            f"declare frames nothing carries can declare the frame that "
            f"makes a short roster look complete")
    detail["bound_files"] = len({relative for relative, _stamp in bound})
    if detail["bound_files"] == 0 or detail["bound_frames"] == 0:
        problems.append(
            f"member {member.number}: verification bound "
            f"{detail['bound_files']} file(s) and "
            f"{detail['bound_frames']} frame(s). VERIFIED means at least one "
            f"declared file supplied at least one indexed frame whose path, "
            f"bytes, valid time, and frame index were all checked; an empty "
            f"inventory or absent indexed roster is no evidence")
    return (problems, detail)


def verify_override_inventory(manifest: EnsembleManifest,
                              indexed: list) -> dict:
    """Bind an overridden member's frames to the manifest's own inventory.

    ``--accept-status`` used to be a byte-level decision: name ``FAILED``
    and the suite recursively globbed ``wrfout*`` under that member's
    directory and averaged whatever it found, without checking the
    top-level manifest status, without binding an inventory, and without
    stamping the override on the graphic.  A stale wrfout from an earlier,
    unrelated attempt went into the mean and the PNG was indistinguishable
    from an all-``DONE`` one.

    The first fix compared the manifest's ``wrfout_count`` with the number
    of files the suite indexed, and that is not an identity either: ONE
    stale file replacing ONE real file passes it exactly.  A focused probe
    put arbitrary bytes behind a single indexed frame, declared
    ``wrfout_count=1``, and the verifier reported the member VERIFIED.

    So the override binds against
    :data:`woof.ensemble.wrfout_inventory.WRFOUT_INVENTORY_CONTRACT` --
    canonical relative path, domain, valid times with their frame indices,
    size and sha256, recorded per file by the writer when the member
    finished.  Every declared file must be present and hash to what was
    recorded; every indexed frame must come from a declared file at the
    declared frame index; nothing undeclared may contribute.

    **A member with no verification evidence is refused, not admitted.**
    A missing inventory used to be reported as "unverifiable" and rendered
    anyway; an explicitly empty inventory later passed vacuously because
    every binding loop had zero iterations.  Both put bytes with no checked
    identity behind a verification-shaped decision.  There is no accurate
    reading of "the roster was widened to include a member whose files
    nothing can identify"; if that member is wanted, the operator can say
    so by producing an inventory that binds at least one indexed frame.
    """

    overridden = {member.number for member in manifest.overridden}
    if not overridden:
        return {"overridden": False}
    by_number = {entry.member.number: entry for entry in indexed}
    problems: list[str] = []
    detail: list[dict] = []
    verified: list[int] = []
    for member in manifest.overridden:
        member_problems, member_detail = _member_inventory_problems(
            member, by_number.get(member.number))
        detail.append(member_detail)
        if member_problems:
            problems.extend(member_problems)
        else:
            verified.append(member.number)
    if problems:
        raise EnsembleRefusal(
            "--accept-status admitted member(s) whose wrfout files do not "
            "bind to the manifest's frame inventory, so nothing was "
            "drawn:\n  " + "\n  ".join(problems))
    return {
        "overridden": True,
        "contract": WRFOUT_INVENTORY_CONTRACT,
        "members": sorted(overridden),
        "verified": sorted(verified),
        "ensemble_status": manifest.status,
        "detail": detail,
    }


def override_caption(report: dict) -> str | None:
    """The line that goes on the panel when the roster was widened."""

    if not report or not report.get("overridden"):
        return None
    parts = []
    for entry in report["detail"]:
        note = "" if entry["error_type"] is None \
            else f"/{entry['error_type']}"
        parts.append(f"{entry['member']}:{entry['status']}{note}")
    bound = sum(int(entry.get("bound_frames", 0))
                for entry in report["detail"])
    files = sum(int(entry.get("declared_wrfout_files") or 0)
                for entry in report["detail"])
    status = report.get("ensemble_status")
    ensemble = "" if status in (None, "COMPLETE") else \
        f"; ensemble status {status}"
    return (f"ROSTER OVERRIDE: --accept-status admitted "
            f"{', '.join(parts)}; {files} file(s)/{bound} frame(s) bound to "
            f"the manifest inventory by sha256{ensemble}")


# ---------------------------------------------------------------------------
# Locating each member's frames
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemberFrames:
    member: EnsembleMember
    domain: str | None
    spacing_m: float | None
    #: valid-time stamp -> (wrfout path, frame index within that file)
    frames: dict


def _wrfout_candidates(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("wrfout*") if p.is_file())


def index_member_frames(member: EnsembleMember, *, wrf,
                        domain: str | None) -> MemberFrames:
    """Map one member's valid times to (file, frame index).

    Refuses a member whose files claim one valid time twice: two frames
    for one time is an ambiguity about which forecast is the forecast,
    and picking either silently would put an arbitrary one in the mean.
    """

    candidates = _wrfout_candidates(member.directory)
    if not candidates:
        raise EnsembleRefusal(
            f"member {member.number}: no wrfout files under "
            f"{member.directory}")
    by_domain: dict[str | None, list[Path]] = {}
    for path in candidates:
        by_domain.setdefault(_domain_tag(path), []).append(path)
    if domain is not None:
        if domain not in by_domain:
            raise EnsembleRefusal(
                f"member {member.number}: no {domain} wrfout under "
                f"{member.directory} (found: "
                f"{', '.join(str(d) for d in sorted(by_domain, key=str))})")
        chosen = domain
    elif len(by_domain) == 1:
        chosen = next(iter(by_domain))
    else:
        raise EnsembleRefusal(
            f"member {member.number}: {member.directory} holds "
            f"{len(by_domain)} domains "
            f"({', '.join(str(d) for d in sorted(by_domain, key=str))}); "
            f"name the one to plot with --domain")

    frames: dict[str, tuple[Path, int]] = {}
    spacing_m = None
    for path in by_domain[chosen]:
        if spacing_m is None:
            spacing_m = _grid_spacing_m(path)
        try:
            wrffile = wrf.WrfFile(str(path))
            stamps = list(wrffile.times())
        except Exception as exc:
            raise EnsembleRefusal(
                f"member {member.number}: unreadable wrfout {path} "
                f"({exc})") from exc
        for index, stamp in enumerate(stamps):
            if stamp in frames:
                raise EnsembleRefusal(
                    f"member {member.number}: valid time {stamp} appears "
                    f"in both {frames[stamp][0].name} and {path.name}; "
                    f"refusing to choose which frame is the forecast")
            frames[stamp] = (path, index)
    return MemberFrames(member=member, domain=chosen, spacing_m=spacing_m,
                        frames=frames)


def index_ensemble(manifest: EnsembleManifest, *, wrf,
                   domain: str | None) -> tuple[list[MemberFrames],
                                                list[str]]:
    """Index every member and agree on the domain, spacing, and times.

    Returns ``(per-member indexes, common valid-time stamps)``.  An
    ensemble whose members disagree about which valid times exist is
    refused naming the divergent members -- a "mean" over members that
    are not at the same time is not a mean of anything.
    """

    indexed: list[MemberFrames] = []
    problems: list[str] = []
    for member in manifest.members:
        try:
            indexed.append(index_member_frames(member, wrf=wrf,
                                               domain=domain))
        except EnsembleRefusal as exc:
            problems.append(str(exc))
    if problems:
        raise EnsembleRefusal(
            "the ensemble is not readable as declared:\n  "
            + "\n  ".join(problems))

    domains = {entry.domain for entry in indexed}
    if len(domains) > 1:
        raise EnsembleRefusal(
            "members disagree about the domain: "
            + ", ".join(
                f"member {entry.member.number}={entry.domain}"
                for entry in indexed))
    spacings = {entry.spacing_m for entry in indexed}
    if len(spacings) > 1:
        raise EnsembleRefusal(
            "members disagree about the grid spacing DX: "
            + ", ".join(
                f"member {entry.member.number}={entry.spacing_m}"
                for entry in indexed))

    stamp_sets = {entry.member.number: set(entry.frames) for entry in indexed}
    common = set.intersection(*stamp_sets.values())
    divergent = {
        number: sorted(stamps - common)
        for number, stamps in stamp_sets.items() if stamps - common}
    if not common:
        raise EnsembleRefusal(
            "the members share no valid time at all; there is nothing to "
            "ensemble")
    if divergent:
        # Surplus frames cannot corrupt a reduction computed on the
        # intersection; name the dropped times and render what is
        # shared (warn-not-block).
        dropped = ", ".join(
            f"member {number}: {', '.join(extra)}"
            for number, extra in sorted(divergent.items()))
        warn(f"members do not share one set of valid times; rendering "
             f"the {len(common)} shared time(s) and ignoring the "
             f"surplus frames ({dropped})")
    return indexed, sorted(common)


def load_member_stack(indexed: list[MemberFrames], spec: FieldSpec,
                      stamp: str, *, wrf) -> np.ndarray:
    """The ``(members, ny, nx)`` stack for one field at one valid time.

    Members are stacked in member-number order, which is the order
    :func:`index_ensemble` guarantees, so a stack index maps back to a
    member identity for the paintball plot.
    """

    planes = []
    shapes: dict[int, tuple] = {}
    for entry in indexed:
        path, timeidx = entry.frames[stamp]
        wrffile = wrf.WrfFile(str(path))
        plane = np.asarray(spec.extract(wrf, wrffile, timeidx), dtype=float)
        if plane.ndim != 2:
            raise EnsembleRefusal(
                f"member {entry.member.number}: field {spec.name} came "
                f"back with shape {plane.shape}, which is not a 2D "
                f"product field")
        shapes[entry.member.number] = plane.shape
        planes.append(plane)
    distinct = set(shapes.values())
    if len(distinct) > 1:
        raise EnsembleRefusal(
            "members disagree about the grid shape: "
            + ", ".join(f"member {number}={shape}"
                        for number, shape in sorted(shapes.items())))
    return np.stack(planes, axis=0)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _unwrap_lon(lon):
    """Antimeridian unwrap onto the branch nearest the domain centre.

    The same correction ``woof.render.render_wrfouts`` applies inline;
    without it a domain crossing +/-180 renders as a smear across the
    full axis.
    """

    values = np.asarray(lon)
    if values.size and (values.max() - values.min()) > 180.0:
        center = float(values[tuple(s // 2 for s in values.shape)])
        return center + ((values - center + 180.0) % 360.0 - 180.0)
    return lon


def _shaded_mesh(axis, lon, lat, values, spec: FieldSpec):
    """One shaded panel on the field's conventional scale."""

    from matplotlib.colors import BoundaryNorm, ListedColormap
    import matplotlib

    if spec.nws_scale:
        cmap = ListedColormap(_NWS_COLORS)
        cmap.set_under("none")
        cmap.set_over(_NWS_COLORS[-1])
        norm = BoundaryNorm(_NWS_LEVELS, cmap.N)
        return axis.pcolormesh(lon, lat, values, cmap=cmap, norm=norm,
                               shading="auto"), _NWS_LEVELS[::2]
    if spec.levels is not None:
        cmap = matplotlib.colormaps[spec.cmap].resampled(len(spec.levels) - 1)
        cmap.set_under("none")
        norm = BoundaryNorm(spec.levels, cmap.N)
        return axis.pcolormesh(lon, lat, values, cmap=cmap, norm=norm,
                               shading="auto"), spec.levels
    return axis.pcolormesh(lon, lat, values, cmap=spec.cmap,
                           shading="auto"), None


def render_mean(stack, spec, *, lat, lon, plt, context, out_png, dpi,
                nan_policy=DEFAULT_NAN_POLICY):
    values = ensemble_mean(stack, nan_policy=nan_policy)
    fig, axis = _figure(plt, lat, lon)
    mesh, ticks = _shaded_mesh(axis, lon, lat, values, spec)
    _finish(fig, axis, mesh,
            title=f"ensemble mean {spec.title}\n{context}",
            cbar_label=spec.units, out_png=out_png, dpi=dpi, ticks=ticks)
    plt.close(fig)


def render_max(stack, spec, *, lat, lon, plt, context, out_png, dpi,
               nan_policy=DEFAULT_NAN_POLICY):
    values = ensemble_max(stack, nan_policy=nan_policy)
    fig, axis = _figure(plt, lat, lon)
    mesh, ticks = _shaded_mesh(axis, lon, lat, values, spec)
    _finish(fig, axis, mesh,
            title=f"ensemble max {spec.title}\n{context}",
            cbar_label=spec.units, out_png=out_png, dpi=dpi, ticks=ticks)
    plt.close(fig)


def render_spread(stack, spec, *, lat, lon, plt, context, out_png, dpi,
                  nan_policy=DEFAULT_NAN_POLICY):
    values = ensemble_spread(stack, nan_policy=nan_policy)
    fig, axis = _figure(plt, lat, lon)
    # Spread is a positive-definite disagreement measure regardless of
    # the field's own scale, so it gets a sequential map and never the
    # field's ladder (a QPF ladder on a spread field reads as rainfall).
    mesh = axis.pcolormesh(lon, lat, values, cmap="magma", shading="auto")
    _finish(fig, axis, mesh,
            title=f"ensemble spread (1 sigma) {spec.title}\n{context}",
            cbar_label=f"standard deviation ({spec.units})",
            out_png=out_png, dpi=dpi)
    plt.close(fig)


def render_probability(stack, spec, *, threshold, radius_km, radius_cells,
                       lat, lon, plt, context, out_png, dpi,
                       nan_policy=DEFAULT_NAN_POLICY):
    from matplotlib.colors import BoundaryNorm
    import matplotlib

    values = exceedance_probability(stack, threshold,
                                    radius_cells=radius_cells,
                                    nan_policy=nan_policy)
    fig, axis = _figure(plt, lat, lon)
    cmap = matplotlib.colormaps["plasma"].resampled(len(_PROB_LEVELS) - 1)
    cmap.set_under("none")
    norm = BoundaryNorm(_PROB_LEVELS, cmap.N)
    mesh = axis.pcolormesh(lon, lat, values, cmap=cmap, norm=norm,
                           shading="auto")
    if radius_cells > 0.0:
        scope = (f"\nwithin {float(radius_km):g} km "
                 f"({radius_cells:.2f} grid cells) of each point")
    else:
        scope = " at the grid point"
    _finish(fig, axis, mesh,
            title=f"probability {spec.title} > {threshold:g} {spec.units}"
                  f"{scope}\n{context}",
            cbar_label="fraction of members", out_png=out_png, dpi=dpi,
            ticks=_PROB_LEVELS)
    plt.close(fig)


def _finish_legend(fig, axis, handles, labels, *, title, out_png, dpi,
                   ncol):
    """``_finish``'s layout for a panel whose key is a legend, not a bar."""

    axis.legend(handles, labels, loc="center left",
                bbox_to_anchor=(1.01, 0.5), fontsize=6, ncol=ncol,
                frameon=False, handlelength=1.4, columnspacing=0.8,
                labelspacing=0.3, title="member", title_fontsize=7)
    axis.set_title(title, fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)


def render_paintball(stack, spec, members, *, threshold, lat, lon, plt,
                     context, out_png, dpi):
    from matplotlib.lines import Line2D

    fig, axis = _figure(plt, lat, lon)
    handles, labels = [], []
    for index, member in enumerate(members):
        color = member_color(member.number)
        values = np.asarray(stack[index], dtype=float)
        # A member entirely below the threshold contours nothing;
        # matplotlib says so with a warning and an empty collection.  It
        # still earns a legend entry: "member 12 drew nothing" is a fact
        # about the forecast, and dropping it from the key would make the
        # reader count the colours to notice.
        if np.nanmax(values) > threshold and np.nanmin(values) <= threshold:
            axis.contour(lon, lat, values, levels=[float(threshold)],
                         colors=[color], linewidths=0.9)
        handles.append(Line2D([0], [0], color=color, linewidth=1.4))
        labels.append(str(member.number))
    ncol = max(1, math.ceil(len(members) / 12))
    _finish_legend(
        fig, axis, handles, labels,
        title=f"paintball {spec.title} > {threshold:g} {spec.units}"
              f"\n{context}",
        out_png=out_png, dpi=dpi, ncol=ncol)
    plt.close(fig)


def render_pmm(stack, spec, *, lat, lon, plt, context, out_png, dpi,
               nan_policy=DEFAULT_NAN_POLICY,
               tie_rule=DEFAULT_PMM_TIE_RULE):
    values = probability_matched_mean(stack, nan_policy=nan_policy,
                                      tie_rule=tie_rule)
    fig, axis = _figure(plt, lat, lon)
    mesh, ticks = _shaded_mesh(axis, lon, lat, values, spec)
    _finish(fig, axis, mesh,
            title=f"probability-matched mean {spec.title}\n{context}",
            cbar_label=spec.units, out_png=out_png, dpi=dpi, ticks=ticks)
    plt.close(fig)


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProductRequest:
    """One panel to draw: a product, and the knobs that name its file."""

    field: str
    product: str
    threshold: float | None = None
    radius_km: float | None = None


def expand_requests(fields, products, thresholds, radii, *,
                    pmm_explicit: bool = True) -> list[ProductRequest]:
    """The cross product of fields x products x thresholds x radii.

    Non-threshold products (mean/spread/pmm/max) appear once per field, not
    once per threshold: they do not depend on one.

    ``pmm_explicit=False`` (the ``--products all`` case) drops ``pmm``
    from the fields that declare it inapplicable, PER FIELD.  Asking for
    the whole suite over ``refl,uh`` should give the reflectivity PMM and
    not an updraft-helicity one; the alternative -- one field's
    applicability deciding for every other field -- puts a meaningless
    panel in a sheet the operator will read as meaningful.
    """

    requests: list[ProductRequest] = []
    for name in fields:
        spec = FIELDS[name]
        field_thresholds = thresholds or ((1000.0, 400.0) if name in ("visibility", "fog_prob") else (spec.default_threshold,))
        for product in products:
            if product == "pmm" and not pmm_explicit and not spec.pmm_in_all:
                continue
            if product in ("mean", "spread", "pmm", "max"):
                requests.append(ProductRequest(field=name, product=product))
                continue
            for threshold in field_thresholds:
                if product == "paintball":
                    requests.append(ProductRequest(
                        field=name, product=product, threshold=threshold))
                    continue
                for radius in radii:
                    requests.append(ProductRequest(
                        field=name, product=product, threshold=threshold,
                        radius_km=radius))
    return requests


def radius_in_cells(radius_km: float | None,
                    spacing_m: float | None) -> float:
    """Kilometres of neighborhood radius -> grid cells.

    A file that declares no DX cannot be given a neighborhood at all: the
    radius would be in unknown units, and guessing one is how a 5 km
    neighborhood becomes a 5-cell one on a 250 m nest -- a 20x error that
    produces a perfectly plausible-looking plot.
    """

    if radius_km is None or radius_km <= 0.0:
        return 0.0
    if spacing_m is None or spacing_m <= 0.0:
        raise EnsembleRefusal(
            f"a {radius_km:g} km neighborhood needs the grid spacing, and "
            f"these wrfouts declare no usable DX global attribute")
    return float(radius_km) * 1000.0 / float(spacing_m)


def run_suite(root, *, fields, products, thresholds, radii, domain,
              timeidx, outdir, dpi=150,
              source_label=DEFAULT_SOURCE_LABEL,
              accept_status=DEFAULT_ACCEPT_STATUS,
              pmm_explicit=True,
              nan_policy=DEFAULT_NAN_POLICY,
              tie_rule=DEFAULT_PMM_TIE_RULE,
              provenance=None,
              ) -> tuple[list[Path], list[str]]:
    """Draw the requested suite with matplotlib; ``(written, failures)``.

    DEPRECATED FALLBACK under the render law: :func:`run_suite_rust` is
    the product tier and ``--engine auto`` reaches it by default.  This
    path stays because two of its options have no Rust equivalent
    (``--domain``, ``--dpi``) and because a checkout that has not built
    the workspace must still be able to look at its ensemble -- with the
    reason it fell back printed, never assumed.


    Roster problems refuse up front (:class:`EnsembleRefusal`) -- an
    ensemble that is not the ensemble it claims to be produces no
    products at all.  A single panel that cannot be drawn (a field absent
    from the files, say) fails that panel with a recorded message and the
    rest continue, which is ``woof render``'s per-product behaviour.
    """

    if nan_policy not in NAN_POLICIES:
        raise EnsembleRefusal(
            f"unknown --nan-policy {nan_policy!r}; choose from "
            f"{', '.join(NAN_POLICIES)}")
    wrf = _import_wrf()
    plt = _pyplot()
    outdir = Path(outdir)
    manifest = load_manifest(root, accept_status=accept_status, domain=domain)
    indexed, stamps = index_ensemble(manifest, wrf=wrf, domain=domain)
    # Before ANY panel is drawn: a widened roster is checked against the
    # manifest's own inventory, not merely accepted as a status string.
    override = verify_override_inventory(manifest, indexed)
    override_note = override_caption(override)
    if override_note:
        print(f"enprod: {override_note}", file=sys.stderr)
    if provenance is not None:
        provenance["roster_override"] = override
        provenance["nan_policy"] = nan_policy
        provenance["pmm_tie_rule"] = tie_rule
        provenance["missingness"] = {}
    if timeidx is not None:
        if timeidx >= len(stamps):
            raise EnsembleRefusal(
                f"--timeidx {timeidx} is out of range; the ensemble shares "
                f"{len(stamps)} valid time(s)")
        stamps = [stamps[timeidx]]

    members = [entry.member for entry in indexed]
    n_members = len(members)
    domain_name = indexed[0].domain
    spacing_m = indexed[0].spacing_m
    token = domain_token(domain_name, spacing_m)
    requests = expand_requests(fields, products, thresholds, radii,
                               pmm_explicit=pmm_explicit)
    if not requests:
        raise EnsembleRefusal(
            "no products to draw: every requested product was dropped as "
            "inapplicable to every requested field")

    written: list[Path] = []
    failures: list[str] = []
    #: output path -> the request that claimed it, this invocation.
    claims: dict[Path, str] = {}
    for stamp in stamps:
        first_path, first_index = indexed[0].frames[stamp]
        try:
            lat, lon = wrf.latlon_coords(wrf.WrfFile(str(first_path)),
                                         timeidx=first_index)
        except Exception as exc:
            failures.append(
                f"{stamp}: no XLAT/XLONG coordinates in {first_path} "
                f"({exc})")
            continue
        lon = _unwrap_lon(lon)
        stacks: dict[str, np.ndarray] = {}
        coverage: dict[str, str | None] = {}
        for request in requests:
            spec = FIELDS[request.field]
            try:
                ensemble_key = ensemble_token(
                    request.product, threshold=request.threshold,
                    unit_slug=spec.unit_slug, radius_km=request.radius_km)
            except ValueError as exc:
                failures.append(f"{stamp} {request.field}: {exc}")
                continue
            out_png = delivered_path(
                outdir, request.field, ensemble_key, token, stamp)
            # v1.0.1's claims map: two requests that resolve to one
            # filename are two forecasts and one file, and the second
            # write would report success while destroying the first.
            description = (f"{request.field} {request.product} "
                           f"threshold={request.threshold} "
                           f"radius_km={request.radius_km}")
            claimed = claims.get(out_png)
            if claimed is not None and claimed != description:
                failures.append(
                    f"{stamp}: would overwrite {out_png.name}, already "
                    f"claimed by [{claimed}]; [{description}] resolves to "
                    f"the same filename token.  Give the two requests "
                    f"distinguishable thresholds/radii, or render them "
                    f"into separate --out directories")
                continue
            claims[out_png] = description
            try:
                if request.field not in stacks:
                    stacks[request.field] = load_member_stack(
                        indexed, spec, stamp, wrf=wrf)
                    report = missingness_report(
                        stacks[request.field],
                        member_numbers=[m.number for m in members],
                        nan_policy=nan_policy)
                    coverage[request.field] = coverage_caption(report)
                    if provenance is not None:
                        provenance["missingness"].setdefault(
                            stamp, {})[request.field] = report
                    if coverage[request.field]:
                        print(f"enprod: {stamp} {request.field}: "
                              f"{coverage[request.field]}", file=sys.stderr)
                stack = stacks[request.field]
                context = ensemble_plot_context(
                    domain_name, spacing_m, stamp, source_label, n_members,
                    notes=(coverage.get(request.field), override_note))
                _draw(request, spec, stack, members, lat=lat, lon=lon,
                      plt=plt, context=context, out_png=out_png, dpi=dpi,
                      spacing_m=spacing_m, nan_policy=nan_policy,
                      tie_rule=tie_rule)
            except (EnsembleRefusal, RuntimeError, ValueError, OSError,
                    KeyError) as exc:
                failures.append(
                    f"{stamp} {request.field}-{ensemble_key}: {exc}")
                continue
            written.append(out_png)
            print(f"enprod: {out_png}")
    return written, failures


def _draw(request: ProductRequest, spec: FieldSpec, stack, members, *,
          lat, lon, plt, context, out_png, dpi, spacing_m,
          nan_policy=DEFAULT_NAN_POLICY,
          tie_rule=DEFAULT_PMM_TIE_RULE) -> None:
    if request.product == "mean":
        render_mean(stack, spec, lat=lat, lon=lon, plt=plt, context=context,
                    out_png=out_png, dpi=dpi, nan_policy=nan_policy)
    elif request.product == "max":
        render_max(stack, spec, lat=lat, lon=lon, plt=plt, context=context,
                   out_png=out_png, dpi=dpi, nan_policy=nan_policy)
    elif request.product == "spread":
        render_spread(stack, spec, lat=lat, lon=lon, plt=plt,
                      context=context, out_png=out_png, dpi=dpi,
                      nan_policy=nan_policy)
    elif request.product == "pmm":
        render_pmm(stack, spec, lat=lat, lon=lon, plt=plt, context=context,
                   out_png=out_png, dpi=dpi, nan_policy=nan_policy,
                   tie_rule=tie_rule)
    elif request.product == "prob":
        cells = radius_in_cells(request.radius_km, spacing_m)
        render_probability(stack, spec, threshold=request.threshold,
                           radius_km=request.radius_km, radius_cells=cells,
                           lat=lat, lon=lon, plt=plt, context=context,
                           out_png=out_png, dpi=dpi, nan_policy=nan_policy)
    elif request.product == "paintball":
        render_paintball(stack, spec, members, threshold=request.threshold,
                         lat=lat, lon=lon, plt=plt, context=context,
                         out_png=out_png, dpi=dpi)
    else:
        raise ValueError(f"unknown ensemble product {request.product!r}")


# ---------------------------------------------------------------------------
# Synthetic fixture generator
# ---------------------------------------------------------------------------


def write_synthetic_ensemble(root, *, n_members: int = 5, nx: int = 24,
                             ny: int = 20, nz: int = 4, dx: float = 3000.0,
                             domain_id: int = 2,
                             stamps=("1970-01-01_18:00:00",
                                     "1970-01-01_19:00:00"),
                             seed: int = 0) -> Path:
    """Write a runnable ``member_NNN/`` + manifest ensemble under ``root``.

    Exists so the suite has a real artifact to be exercised against -- by
    the tests, and by an operator who wants to see what the products look
    like before an ensemble finishes running.  The members are a drifting
    Gaussian reflectivity blob plus noise, which is meteorologically
    meaningless and statistically sufficient: it gives the mean a
    pattern, the spread a maximum on the blob's flanks, and the members
    enough displacement diversity that the paintball and probability
    plots are not all one contour.

    The manifest it writes is the shape
    :func:`woof.ensemble.manifest.new_ensemble_manifest` writes -- the
    writer of record for ``gpuwm-ensemble-manifest.v1`` -- so a fixture
    root and a real ensemble root are the same kind of thing to every
    consumer: zero-based ``index``, ``member_dir``, and the writer's
    ``DONE`` terminal status.  A fixture in a spelling nothing produces
    would test a contract nothing writes.
    """

    import datetime
    from types import SimpleNamespace

    from woof.io.wrfout import WrfoutWriter, wrf_global_attrs

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    lat = np.tile(np.linspace(38.0, 40.0, ny)[:, None], (1, nx))
    lon = np.tile(np.linspace(-98.0, -95.0, nx)[None, :], (ny, 1))

    # The PRODUCTION global-attribute profile, not a bare GRID_ID.
    #
    # A fixture whose files are not the kind of file the product writes is
    # a fixture that tests a contract nothing produces.  This one used to
    # carry `GRID_ID` alone, which the matplotlib engine happens not to
    # read -- so the fixture passed every test in this suite while being
    # unreadable by the RUST renderer, whose fail-closed preflight wants a
    # run origin (`START_DATE`) and a projection like every real wrfout
    # has.  `woof enprod --engine rust` on a `--make-fixture` ensemble
    # refused with "no sound WRF run origin" until this changed.
    fixture_grid = SimpleNamespace(
        truelat1=38.5, truelat2=39.5, stand_lon=-96.5, ref_lat=39.0,
        ref_lon=-96.5)
    attrs = wrf_global_attrs(
        fixture_grid,
        datetime.datetime.strptime(stamps[0], "%Y-%m-%d_%H:%M:%S"),
        grid_id=domain_id, parent_id=max(domain_id - 1, 1),
        i_parent_start=5, j_parent_start=5, parent_grid_ratio=3, dt=6.0)

    records = []
    for number in range(n_members):
        # np.iinfo rather than 2**31 - 1: the repo's float32-total-order
        # gate (tests/test_fp32_ulp.py) reads any subtraction whose left
        # operand folds to the sign bit as a hand-rolled ordering, and it
        # is right to be blunt about that -- the constant is the one this
        # spelling should have used anyway.
        member_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        member_rng = np.random.default_rng(member_seed)
        directory = root / f"member_{number:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"wrfout_d{domain_id:02d}_" \
                           f"{_stamp_for_filename(stamps[0])}.nc"
        # Each member's storm sits somewhere else: that displacement is
        # the whole point of an ensemble product.
        y0 = ny / 2.0 + member_rng.normal(0.0, ny / 12.0)
        x0 = nx / 2.0 + member_rng.normal(0.0, nx / 12.0)
        with WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=dx, dy=dx,
                          global_attrs=attrs) as writer:
            for step, stamp in enumerate(stamps):
                drift = 1.5 * step
                blob = np.exp(-(((yy - y0 - drift) ** 2
                                 + (xx - x0 - drift) ** 2)
                                / (2.0 * (min(ny, nx) / 6.0) ** 2)))
                surface = 60.0 * blob + member_rng.normal(0.0, 2.0, (ny, nx))
                column = np.stack(
                    [surface * weight for weight in (1.0, 0.9, 0.7, 0.4)
                     ][:nz], axis=0)
                writer.write_frame(stamp, {
                    "T": np.zeros((nz, ny, nx), np.float32),
                    "MU": np.zeros((ny, nx), np.float32),
                    "REFL_10CM": column.astype(np.float32),
                    "UP_HELI_MAX": (180.0 * blob ** 2).astype(np.float32),
                    "T2": (292.0 + 4.0 * blob).astype(np.float32),
                    "U10": member_rng.normal(0.0, 5.0,
                                             (ny, nx)).astype(np.float32),
                    "V10": member_rng.normal(0.0, 5.0,
                                             (ny, nx)).astype(np.float32),
                    "RAINC": (5.0 * blob).astype(np.float32),
                    "RAINNC": (20.0 * blob).astype(np.float32),
                    "XLAT": lat.astype(np.float32),
                    "XLONG": lon.astype(np.float32),
                    "HGT": np.zeros((ny, nx), np.float32),
                    "SINALPHA": np.zeros((ny, nx), np.float32),
                    "COSALPHA": np.ones((ny, nx), np.float32),
                })
        # The same frame inventory the real writer records
        # (woof.ensemble.engine), built by the same function: a fixture
        # that declared only a count would exercise a contract the engine
        # no longer writes and the override path no longer accepts.
        inventory = build_member_inventory([path], member_dir=directory)
        records.append({
            "index": number,
            "member_dir": f"member_{number:03d}",
            "seed": member_seed,
            "status": "DONE",
            "wrfout_count": len(inventory),
            WRFOUT_INVENTORY_KEY: [dict(entry) for entry in inventory],
            "provenance": {
                "generator": "woof.da.enprod.write_synthetic_ensemble",
                "synthetic": True,
            },
        })

    manifest = root / MANIFEST_FILENAME
    manifest.write_text(json.dumps({
        "schema": MANIFEST_SCHEMA,
        "stability": "experimental",
        "experimental": True,
        "status": "COMPLETE",
        "n_members": n_members,
        "members": records,
    }, indent=2) + "\n", encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def engine_field_specs(engine_path) -> dict:
    """The field registry the rust route draws with: the ENGINE's values.

    Title, units, default threshold and unit slug come off
    ``--list-fields``; everything the engine cannot supply (the
    extractor the matplotlib fallback needs, the colour ladder) stays
    local.  A field the engine knows and this module does not gets a row
    with no extractor, which is exactly what the rust route needs and
    exactly what the fallback must refuse by name.

    Empty when the engine cannot be asked; the caller then keeps the
    local registry and says which one it used.
    """

    from dataclasses import replace

    from woof import rustwx_lanes

    rows = rustwx_lanes.list_ensemble_fields(engine_path)
    if not rows:
        return {}
    resolved: dict = {}
    for name, row in rows.items():
        local = FIELDS.get(name)
        if local is not None:
            resolved[name] = replace(
                local, title=row.get("title", local.title),
                units=row.get("units", local.units),
                unit_slug=row.get("unit_slug", local.unit_slug),
                default_threshold=float(row.get("default_threshold",
                                                local.default_threshold)),
                default_thresholds=tuple(row.get("default_thresholds", ())))
            continue
        resolved[name] = FieldSpec(
            name=name, title=row.get("title", name),
            units=row.get("units", ""),
            unit_slug=row.get("unit_slug", ""),
            default_threshold=float(row.get("default_threshold", 0.0)),
            # No local drawing code: the engine draws this one, and the
            # matplotlib fallback refuses it by name rather than
            # pretending it can.
            extract=None,
            default_thresholds=tuple(row.get("default_thresholds", ())))
    return resolved


def parse_fields(spec: str, vocabulary=None) -> tuple[str, ...]:
    """The requested fields, checked against the vocabulary that will draw.

    ``vocabulary`` is the ENGINE's own field list when the rust route is
    resolved, and the local registry otherwise.  A field the engine
    carries and this module has no drawing code for used to be refused
    at argument check with a list of five names, even though the engine
    it was about to be handed to knew it perfectly well.
    """

    vocabulary = FIELDS if vocabulary is None else vocabulary
    names: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "all":
            names.extend(name for name in vocabulary if name not in names)
            continue
        if token not in vocabulary:
            raise ValueError(
                f"unknown field {token!r}; choose from "
                f"{', '.join(vocabulary)} or 'all'; rw_ensbatch "
                f"--list-fields prints the engine's own vocabulary")
        if token not in names:
            names.append(token)
    if not names:
        raise ValueError("no fields requested")
    return tuple(names)


def parse_products(spec: str) -> tuple[tuple[str, ...], bool]:
    """``mean,prob`` -> ``(products, pmm_was_named)``.

    The second element distinguishes ``--products pmm`` from
    ``--products all``: a probability-matched mean of 2 m temperature is
    a number this suite can compute and nobody should read, so ``all``
    leaves it out per field (see :func:`expand_requests`) while naming it
    outright still draws it, with a note.
    """

    names: list[str] = []
    explicit_pmm = False
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "all":
            for product in PRODUCTS:
                if product not in names:
                    names.append(product)
            continue
        if token not in PRODUCTS:
            raise ValueError(
                f"unknown product {token!r}; choose from "
                f"{', '.join(PRODUCTS)} or 'all'")
        if token == "pmm":
            explicit_pmm = True
        if token not in names:
            names.append(token)
    if not names:
        raise ValueError("no products requested")
    return tuple(names), explicit_pmm


def parse_floats(spec: str, *, what: str,
                 minimum: float | None = None) -> tuple[float, ...]:
    values: list[float] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError:
            raise ValueError(
                f"{what} must be comma-separated numbers, got "
                f"{token!r}") from None
        if not math.isfinite(value):
            raise ValueError(f"{what} must be finite, got {token!r}")
        if minimum is not None and value < minimum:
            raise ValueError(f"{what} must be >= {minimum:g}, got {value:g}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"no {what} given")
    return tuple(values)


def parse_timeidx(spec: str) -> int | None:
    if spec == "all":
        return None
    try:
        index = int(spec)
    except ValueError:
        raise ValueError(
            f"--timeidx must be an integer or 'all', got {spec!r}") from None
    if index < 0:
        raise ValueError("--timeidx must be non-negative")
    return index


def enprod_main(args: argparse.Namespace) -> int:
    if args.make_fixture is not None:
        if args.ens_root is not None:
            print("enprod: --make-fixture writes a new ensemble; it does "
                  "not take an existing ENS_ROOT argument", file=sys.stderr)
            return 2
        if args.members < 2:
            warn(f"--members {args.members} makes a one-member "
                 "synthetic ensemble; spread and probabilities will "
                 "refuse their own panels, everything else renders")
        manifest = write_synthetic_ensemble(args.make_fixture,
                                            n_members=args.members)
        print(f"enprod: synthetic {args.members}-member ensemble -> "
              f"{manifest.parent}")
        print(f"enprod: manifest {manifest}")
        return 0
    if args.ens_root is None:
        print("enprod: an ensemble root directory is required (or "
              "--make-fixture DIR to write a synthetic one)",
              file=sys.stderr)
        return 2
    # The ENGINE is resolved before the field is checked, because the
    # engine owns the vocabulary the field is checked against.  Both
    # refusals still fire here, at argument check, before a member is
    # read.
    print(f"enprod: {experimental_stamp()}")
    try:
        engine, why = resolve_enprod_engine(args.engine)
    except RuntimeError as refusal:
        print(f"enprod: refused: {str(refusal).split('[[explain]]')[0].strip()}",
              file=sys.stderr)
        return 2
    print(f"enprod: engine {engine} ({why})")
    vocabulary = None
    if engine == "rust":
        from woof import rustwx_lanes

        vocabulary = engine_field_specs(rustwx_lanes.find_ensemble_bin()) or None
    try:
        fields = parse_fields(args.field, vocabulary)
        products, pmm_explicit = parse_products(args.products)
        thresholds = (parse_floats(args.threshold, what="--threshold")
                      if args.threshold else ())
        radii = parse_floats(args.neighborhood_km, what="--neighborhood-km",
                             minimum=0.0)
        timeidx = parse_timeidx(args.timeidx)
        accept_status = tuple(
            s.strip() for s in args.accept_status.split(",") if s.strip())
    except ValueError as exc:
        print(f"enprod: {exc}", file=sys.stderr)
        return 2
    if not accept_status:
        warn("--accept-status resolved to nothing; using the default "
             f"({','.join(DEFAULT_ACCEPT_STATUS)})")
        accept_status = tuple(DEFAULT_ACCEPT_STATUS)
    specs = FIELDS if vocabulary is None else vocabulary
    if engine == "matplotlib":
        # The fallback draws with local code, so a field the engine
        # knows and this module cannot draw is refused BY NAME with the
        # route that can draw it.
        undrawable = [name for name in fields
                      if getattr(specs.get(name), "extract", None) is None]
        if undrawable:
            print(f"enprod: --engine matplotlib cannot draw "
                  f"{', '.join(undrawable)}: the fallback has no extractor "
                  f"for those fields and only the rust engine does.  Draw "
                  f"them with --engine rust.", file=sys.stderr)
            return 2
    if "pmm" in products and pmm_explicit:
        for name in fields:
            if not getattr(specs.get(name), "pmm_in_all", False):
                print(f"enprod: note: pmm on {name} is computable but not "
                      f"meaningful; the probability-matched mean assumes a "
                      f"reflectivity-like intensity distribution",
                      file=sys.stderr)
    if set(accept_status) - set(DEFAULT_ACCEPT_STATUS):
        warn(f"--accept-status widens the roster past "
             f"{','.join(DEFAULT_ACCEPT_STATUS)}",
             why="Every admitted member's frame inventory is checked "
                 "against the manifest before anything renders, and "
                 "the override is stamped on every panel.")
    if engine == "matplotlib":
        # An opt-in route is a workaround and says so every run, not
        # once in a manual page.
        from woof import rustwx_lanes

        print(f"enprod: {rustwx_lanes.ensemble_workaround_notice()}",
              file=sys.stderr)
    # The nest choice is about the REQUEST, so it is answered once for
    # both engines and before either opens anything.  A manifest this
    # cannot read is not refused here: the route that reads it properly
    # refuses it with its own sentence a moment later.
    if args.ens_root is not None:
        try:
            roster = load_manifest(args.ens_root, accept_status=accept_status,
                                   domain=args.domain)
        except EnsembleRefusal:
            roster = None
        if roster is not None:
            problem = domain_choice_problem(roster, args.domain)
            if problem is not None:
                print(f"enprod: {problem}", file=sys.stderr)
                return 2
    if engine == "rust":
        width, height = panel_pixels(args.dpi)
        if (width, height) != _ENGINE_PANEL_PIXELS:
            print(f"enprod: --dpi {args.dpi:g} renders {width}x{height} px, "
                  f"scaled from the engine's "
                  f"{_ENGINE_PANEL_PIXELS[0]}x{_ENGINE_PANEL_PIXELS[1]} at "
                  f"the default --dpi {_REFERENCE_DPI:g}")
        return run_suite_rust(
            args.ens_root, fields=fields, products=products,
            thresholds=thresholds, radii=radii, timeidx=timeidx,
            outdir=args.out, source_label=args.source_label,
            accept_status=accept_status, nan_policy=args.nan_policy,
            tie_rule=args.pmm_tie_rule, domain=args.domain,
            width=width, height=height, specs=specs)
    provenance: dict = {}
    try:
        written, failures = run_suite(
            args.ens_root, fields=fields, products=products,
            thresholds=thresholds, radii=radii, domain=args.domain,
            timeidx=timeidx, outdir=args.out, dpi=args.dpi,
            source_label=args.source_label, accept_status=accept_status,
            pmm_explicit=pmm_explicit, nan_policy=args.nan_policy,
            tie_rule=args.pmm_tie_rule, provenance=provenance)
    except EnsembleRefusal as exc:
        print(f"enprod: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"enprod: {exc}", file=sys.stderr)
        return 2
    for failure in failures:
        print(f"enprod FAIL: {failure}", file=sys.stderr)
    _publish_receipt(args.out, engine="matplotlib",
                     requested_spec=_requested_spec(fields, products),
                     written=written, failures=failures, skipped=(),
                     families=None)
    print(f"enprod: {len(written)} file(s) -> {args.out}")
    return 0 if written and not failures else 1



# ---------------------------------------------------------------------------
# The Rust engine route
# ---------------------------------------------------------------------------
#
# This module's own docstring used to say there was no second engine to
# switch to -- "the vendored rust renderer's catalog has no ensemble
# entries, so this module is matplotlib-only".  ``rw_ensbatch`` is that
# second engine: the reductions are `rustwx-ensemble` (the same policies,
# unit-tested against the same probes this module's docstrings name) and
# the panels come off the same production map/basemap/design/PNG path a
# deterministic `woof render --engine rust` panel does.


def resolve_enprod_engine(request: str) -> tuple[str, str]:
    """``(engine, why)`` for ``--engine``, through the shared contract.

    The contract itself lives in
    :func:`woof.rustwx_lanes.resolve_lane_engine`, because the
    observation-grid lane asks the identical question and two lanes
    answering one question differently is the drift this delegation
    ends.  ``matplotlib`` by name only, ``rust`` raises, and ``auto`` is
    rust or a named refusal -- it does NOT degrade, which is what
    ``woof render``'s own door was changed to when the render law's
    one-fallback clause was enforced.
    """

    from woof import rustwx_lanes

    return rustwx_lanes.resolve_ensemble_engine(request)


def domain_choice_problem(manifest, domain: str | None = None) -> str | None:
    """One refusal for an ensemble whose members hold more than one nest.

    ONE function, both engines, asked at the front door before anything
    is opened.  An ensemble product of mixed nests is not an ensemble
    product -- averaging one member's d01 against another's d02 is a
    picture of nothing -- so this is a refusal about the REQUEST, not
    about either engine, and it names the way out.

    It fired in two places with two different sentences before: the
    matplotlib route refused per member deep inside frame indexing, and
    the rust route refused the ``--domain`` flag itself as unimplemented.
    """

    from woof.ensemble import wrfout_inventory

    by_member: dict[int, list[str]] = {}
    for member in manifest.members:
        found = sorted({
            token for token in (
                wrfout_inventory.domain_token(path)
                for path in _wrfout_candidates(Path(member.directory)))
            if token})
        by_member[member.number] = found
    present = sorted({token for found in by_member.values() for token in found})
    if domain is not None:
        without = sorted(number for number, found in by_member.items()
                         if found and domain not in found)
        if without:
            return (f"--domain {domain} names a nest member(s) "
                    f"{', '.join(str(number) for number in without)} do not "
                    f"carry (these members hold "
                    f"{', '.join(present) or 'no wrfout_dNN file'}).  Every "
                    f"member of an ensemble must contribute the same nest; "
                    f"name one they all hold, or fix the member roster.")
        return None
    if len(present) <= 1:
        return None
    return (f"the members hold more than one domain "
            f"({', '.join(present)}); an ensemble product of mixed nests "
            f"is not an ensemble product.  Name the one to plot with "
            f"--domain dNN.")


def _rust_delivery_facts(manifest, *, domain: str | None,
                         frames: int | None) -> tuple[str, str]:
    """``(domain token, valid day)`` for the rust route's delivered tree.

    Read in PYTHON, from the member wrfout, with netCDF4 only: this
    route exists so ensemble panels can be drawn without the wrf
    package, and reading the facts through it would give that back.

    Neither fact is worth failing over.  An unreadable member files
    under the same two degradations the deterministic route uses --
    ``native_grid`` and ``undated`` -- because a picture filed under a
    stated unknown is findable and a picture refused is not.
    """

    from woof import render, render_layout
    from woof.ensemble import wrfout_inventory

    token = render_layout.NATIVE_GRID
    day = render_layout.UNDATED
    for member in manifest.members:
        for path in _wrfout_candidates(Path(member.directory)):
            found = wrfout_inventory.domain_token(path)
            if domain is not None and found != domain:
                continue
            token = render.domain_token(found, _grid_spacing_m(path))
            try:
                stamps = wrfout_inventory.wrfout_valid_times(path)
            except Exception:
                stamps = ()
            if stamps:
                index = 0 if frames is None else min(int(frames), len(stamps) - 1)
                day = render_layout.valid_day(stamps[index]) or render_layout.UNDATED
            return token, day
    return token, day


def run_suite_rust(ens_root, *, fields, products, thresholds, radii,
                   timeidx, outdir, source_label, accept_status,
                   nan_policy, tie_rule, domain: str | None = None,
                   width: int = 1200, height: int = 900, specs=None) -> int:
    """Drive ``rw_ensbatch`` once per (field, threshold, radius).

    The roster gate stays in the engine, which reads the same manifest
    this module reads and refuses the same statuses; what this function
    owns is only the request fan-out, because the CLI takes LISTS of
    thresholds and radii and one engine invocation renders one of each.
    """

    from woof import render, render_layout, rustwx_lanes

    engine_path = rustwx_lanes.find_ensemble_bin()
    try:
        manifest = manifest_path(ens_root, domain)
    except EnsembleRefusal as exc:
        print(f"enprod: {exc}", file=sys.stderr)
        return 2
    if not manifest.is_file():
        print(f"enprod: no ensemble manifest at {manifest}", file=sys.stderr)
        return 2
    outdir = Path(outdir)
    written: list[Path] = []
    failures: list[str] = []
    skipped_rows: list[tuple[str, str]] = []
    families: dict[str, str] = {}
    frames = None if timeidx is None else int(timeidx)
    try:
        roster = load_manifest(ens_root, accept_status=accept_status,
                               domain=domain)
        domain_token_value, valid_day = _rust_delivery_facts(
            roster, domain=domain, frames=frames)
    except EnsembleRefusal as exc:
        roster = None
        if _manifest_schema(manifest) == V2_MANIFEST_SCHEMA:
            # The engine reads v1 only, so for a v2 run this module's
            # refusal is the only one there is.
            print(f"enprod: {exc}", file=sys.stderr)
            return 2
        # The engine reads the same manifest and refuses the same roster
        # with its own sentence; this read is only for the delivered
        # path, so it must not become a second refusal.
        domain_token_value, valid_day = (render_layout.NATIVE_GRID,
                                         render_layout.UNDATED)
    # The engine's member store is WORKING SCRATCH, so it lives beside
    # the delivered panels, not in them: a delivered product tree holds
    # products only (woof.render.scratch_store names the breakage).
    # It used to be `outdir / "_ens_store"`, which was never removed at
    # all.
    with render.scratch_store(outdir, prefix="ensstore-") as store_root:
        # The engine draws into the SCRATCH store and Python files the
        # results, exactly as `woof render` does: the engine's own
        # segments are a member count and the literal word "ensemble",
        # so two nests of one ensemble under one --out shared a folder,
        # and the vendored crate stays byte-identical either way.
        engine_out = store_root / "png"
        engine_out.mkdir(parents=True, exist_ok=True)
        if roster is not None and roster.schema == V2_MANIFEST_SCHEMA:
            manifest = engine_roster(roster, store_root)
        for name in fields:
            # The ENGINE's own row when it could be asked: a threshold
            # and a unit slug kept in two places meant one panel's
            # threshold read 30 degrees Celsius here and 303 kelvin
            # there.
            spec = (specs or FIELDS)[name]
            field_thresholds = thresholds or ((1000.0, 400.0) if name in ("visibility", "fog_prob") else (spec.default_thresholds or (spec.default_threshold,)))
            for threshold in field_thresholds:
                for radius_km in (radii or (0.0,)):
                    paths, problems, skipped, report = (
                        rustwx_lanes.run_ensemble_renderer(
                            engine_path, manifest, store_root=store_root,
                            out_dir=engine_out, field=name,
                            products=",".join(products),
                            threshold=threshold,
                            neighborhood_km=radius_km, frames=frames,
                            nan_policy=nan_policy, pmm_tie_rule=tie_rule,
                            accept_status=",".join(accept_status),
                            source_label=source_label, domain=domain,
                            width=width, height=height))
                    for product, path in _rendered_pairs(report, paths):
                        try:
                            ensemble_key = ensemble_token(
                                product, threshold=threshold,
                                unit_slug=spec.unit_slug, radius_km=radius_km)
                        except ValueError:
                            ensemble_key = f"ens-{product}"
                        filename = product_filename(
                            name, ensemble_key, domain_token_value, valid_day)
                        target = render_layout.place(
                            outdir, domain=domain_token_value,
                            product=f"{name}-{ensemble_key}", day=valid_day,
                            filename=filename)
                        if target.exists():
                            # Two panels resolving to one delivered name
                            # would be two forecasts and one file, and
                            # the second write would report success while
                            # destroying the first.  The engine's own
                            # name is what tells them apart, so it stays.
                            filename = f"{target.stem}_{Path(path).stem}.png"
                        delivered, note = render_layout.deliver(
                            outdir, path,
                            domain=domain_token_value,
                            product=f"{name}-{ensemble_key}",
                            day=valid_day, filename=filename)
                        if note:
                            print(f"enprod: {note}", file=sys.stderr)
                        families[str(Path(delivered).resolve())] = \
                            f"{name}-{ensemble_key}"
                        written.append(delivered)
                    failures.extend(problems)
                    skipped_rows.extend(skipped)
                    for slug, reason in skipped:
                        print(f"enprod: skipped {slug}: {reason}")
                    for key in ("members", "coverage", "pmm_ties",
                                "paintball_legend"):
                        if key in report:
                            print(f"enprod: {key} {report[key]}")
    for path in written:
        print(f"enprod: {path}")
    _publish_receipt(outdir, engine="rust",
                     requested_spec=_requested_spec(fields, products),
                     written=written, failures=failures,
                     skipped=skipped_rows, families=families)
    if failures:
        for failure in failures:
            print(f"enprod: {failure}", file=sys.stderr)
        return 1
    if not written:
        print("enprod: the rust engine drew nothing", file=sys.stderr)
        return 1
    print(f"enprod: {len(written)} panel(s) -> {outdir}")
    return 0


def _manifest_schema(path: Path) -> str | None:
    """The schema string a manifest file declares, or ``None``."""

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return _schema_of(document) if isinstance(document, dict) else None


def _rendered_pairs(report: dict, paths) -> list[tuple[str, object]]:
    """``(product, path)`` for each RENDERED event, in engine order."""

    rows = report.get("rendered") or []
    if len(rows) == len(paths):
        return list(rows)
    # A build that answered no product names: the paths are still
    # delivered, under the product the engine's own folder spells.
    return [(Path(path).parent.parent.name or "product", path)
            for path in paths]


def _requested_spec(fields, products) -> str:
    """The request, spelled for the receipt."""

    return ",".join(f"{field}:{product}"
                    for field in fields for product in products)


def _publish_receipt(outdir, *, engine, requested_spec, written, failures,
                     skipped, families) -> None:
    """Record this delivery the way every other render delivery is.

    A render receipt is a property of a DELIVERY, not of one door: the
    desktop and remote surfaces read ``render-summary.json`` and saw
    nothing at all for an ensemble suite, on either engine, with the
    absence itself unannounced.  It is a record, so a failure to write
    one is reported and never turned into a failed render.
    """

    from woof import render_layout, render_receipts

    if not written:
        return
    try:
        render_receipts.deliver(
            root=Path(outdir), engine=engine, requested_spec=requested_spec,
            written=[Path(path) for path in written],
            failures=[str(reason) for reason in failures],
            skipped=[(str(slug), str(reason)) for slug, reason in skipped],
            layout=render_layout.NESTED, families=families)
    except Exception as error:
        print(f"enprod: warning: no render receipt was published ({error})",
              file=sys.stderr)

def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "enprod",
        help="ensemble product PNGs from a member_NNN/ ensemble root "
             "(mean, spread, exceedance probability with neighborhoods, "
             "paintball, probability-matched mean)")
    parser.add_argument(
        "ens_root", type=Path, nargs="?", metavar="ENS_ROOT",
        help=f"ensemble root holding member_NNN/ run directories and "
             f"{MANIFEST_FILENAME} (schema {MANIFEST_SCHEMA}), or the run "
             f"folder of woof go --members --keep-member-files (schema "
             f"{V2_MANIFEST_SCHEMA}, one manifest per domain)")
    parser.add_argument(
        "--field", default="refl", metavar="LIST",
        help=f"comma-separated product fields: {', '.join(FIELDS)}, or "
             f"'all' (default refl; 'refl,uh' is the severe-convective "
             f"pair)")
    parser.add_argument(
        "--products", default="all", metavar="LIST",
        help=f"comma-separated products: {', '.join(PRODUCTS)}, or 'all' "
             f"(default)")
    parser.add_argument(
        "--threshold", default="", metavar="LIST",
        help="comma-separated thresholds in the field's own "
             "units; default is the field's own (refl 40 dBZ, uh 75 "
             "m2 s-2; visibility strictly below 1000 and 400 m). Every threshold gets its own probability and "
             "paintball plot")
    parser.add_argument(
        "--neighborhood-km", default="0", metavar="LIST",
        help="comma-separated neighborhood radii in km for the "
             "probability product (default 0 = point probability).  Each "
             "member uses its maximum (minimum for visibility) within the radius before "
             "the ensemble fraction is taken")
    parser.add_argument(
        "--domain", metavar="dNN",
        help="which domain to plot when members hold more than one "
             "(default: the single domain present, else a refusal)")
    parser.add_argument(
        "--timeidx", default="all", metavar="N|all",
        help="index into the valid times every member shares, or 'all' "
             "(default)")
    parser.add_argument(
        "--out", type=Path, default=Path("out/enprod"), metavar="DIR",
        help="output directory for the PNGs (default out/enprod)")
    parser.add_argument(
        "--dpi", type=positive_int, default=150, metavar="N",
        help="PNG resolution (default 150)")
    parser.add_argument(
        "--source-label", default=DEFAULT_SOURCE_LABEL, metavar="TEXT",
        help=f"model/provenance label stamped on every plot (default "
             f"{DEFAULT_SOURCE_LABEL})")
    parser.add_argument(
        "--accept-status", default=",".join(DEFAULT_ACCEPT_STATUS),
        metavar="LIST",
        help=f"comma-separated manifest member statuses to accept "
             f"(default {','.join(DEFAULT_ACCEPT_STATUS)}); any other "
             f"status is a refusal naming the members")
    parser.add_argument(
        "--nan-policy", default=DEFAULT_NAN_POLICY, choices=NAN_POLICIES,
        help="what to do with a non-finite member value -- NaN or "
             f"+/-Inf (default {DEFAULT_NAN_POLICY}): 'mask' excludes it "
             "from every reduction at that point, shrinks the denominator "
             "with it, and stamps the resulting coverage on the panel; "
             "'refuse' fails the whole product naming the members")
    parser.add_argument(
        "--pmm-tie-rule", default=DEFAULT_PMM_TIE_RULE,
        choices=PMM_TIE_RULES,
        help="how the probability-matched mean resolves equal means "
             f"(default {DEFAULT_PMM_TIE_RULE}): 'flat-index' is Ebert's "
             "algorithm exactly and paints a deterministic but "
             "meaningless row-major gradient across a plateau; 'average' "
             "gives every point in a tie the group's mean intensity and "
             "gives up the exact pooled distribution")
    parser.add_argument(
        "--engine", default="auto", choices=("auto", "rust", "matplotlib"),
        help="which renderer draws the panels (default auto).  The render "
             "law puts weather fields on the Rust renderer; 'auto' uses "
             "rw_ensbatch when this checkout has built it and falls back "
             "to matplotlib with the reason named, 'rust' refuses rather "
             "than substituting, and 'matplotlib' selects the fallback "
             "outright")
    parser.add_argument(
        "--make-fixture", type=Path, metavar="DIR",
        help="write a synthetic ensemble (members + manifest) to DIR and "
             "exit, for exercising the suite without a real ensemble")
    parser.add_argument(
        "--members", type=positive_int, default=5, metavar="N",
        help="--make-fixture member count (default 5)")
    parser.set_defaults(func=enprod_main)
    return parser


__all__ = [
    "DEFAULT_ACCEPT_STATUS", "DEFAULT_NAN_POLICY", "DEFAULT_PMM_TIE_RULE",
    "EXPERIMENTAL_STAMP", "FIELDS", "MANIFEST_FILENAME", "MANIFEST_SCHEMA",  # noqa: F822 -- module __getattr__
    "V2_MANIFEST_SCHEMA", "engine_roster", "manifest_path",
    "experimental_stamp",
    "NAN_POLICIES", "PAINTBALL_PALETTE", "PMM_TIE_RULES", "PRODUCTS",
    "EnsembleManifest", "EnsembleMember", "EnsembleRefusal", "FieldSpec",
    "MemberFrames", "ProductRequest", "coverage_caption", "disc_offsets",
    "ensemble_max", "ensemble_mean", "ensemble_plot_context",
    "ensemble_spread",
    "ensemble_token", "exceedance_probability", "expand_requests",
    "resolve_enprod_engine", "run_suite_rust",
    "index_ensemble", "index_member_frames", "load_manifest",
    "load_member_stack", "member_color", "missingness_report",
    "neighborhood_footprint", "neighborhood_max", "number_slug",
    "override_caption", "pmm_tie_report", "probability_matched_mean",
    "product_filename", "radius_in_cells", "radius_slug", "register_cli",
    "run_suite", "threshold_slug", "verify_override_inventory",
    "write_synthetic_ensemble",
]
