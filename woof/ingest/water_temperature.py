"""Assemble the finished water-surface temperature before soil preprocessing.

Why this module exists
----------------------
The shipped chain chose a water temperature PER CELL, between two fields
that were mapped by two different algorithms::

    water_temperature = where(finite(SST) and 170 <= SST <= 400, SST, SKINTEMP)

SST reaches the target through METGRID.TBL's ``sixteen_pt+four_pt``, and both
of those operators require every point of their stencil to be usable.  An SST
analysis carries no values over land, so within two source cells of any
coastline the stencil is incomplete, both operators decline, and the target
takes ``fill_missing=0.``.  Zero is not a temperature, so the selector above
reads it as "no SST here" and switches that column to SKINTEMP -- which over
lakes is ERA5's FLake model state, several K warmer than the analysis and
quantized on the 0.25 degree source cell.

The consequence is a lake painted as a quilt of two source fields joined at
the SST-validity boundary.  Measured on the reproducing case (Lake Erie,
1985-05-31 12Z, 24 km parent with a ratio-8 3 km nest) by
``tools/water_temperature_probe.py``, over the nest's 4195 water cells and
the 7915 adjacencies between them, with the shipped operators:

===============================================  ===========================
quantity                                         value
===============================================  ===========================
water cells taking SST                           1955 (46.6 %)
water cells taking SKINTEMP                      2240 (53.4 %)
adjacent-step P99 across the water               7.81 K
adjacent-step P99 INSIDE the SST region          0.13 K
adjacent-step P99 INSIDE the SKINTEMP region     0.85 K
adjacent pairs straddling the boundary           397, median jump 5.64 K
===============================================  ===========================

Each adjacency is counted once.  Restricted to Erie itself (the largest
connected body, 2684 cells) 257 of the 269 steps above 1 K sit exactly on the
switch boundary, which is the attribution: neither field is rough on its own,
and the discontinuity is manufactured by switching between them.  So the
remedy is to stop switching: decide ONE provider for a whole connected body
of water and interpolate that provider coherently across it.

What this module guarantees
---------------------------
* Surface classes come from the target statics: land is ``LANDMASK >= 0.5``,
  lakes are the land-use table's own inland-water category (``ISLAKE``),
  ocean is the remaining water.
* A lake and the ocean can never share a connected component, so a lake can
  never be handed ocean water.
* Donors for a component are drawn from that component alone.  There is no
  global search, so a landlocked lake with no analysis of its own cannot
  import another basin's water.
* Every water cell is attributed in ``WATER_TEMP_SOURCE``.
* A source can additionally declare lake-model water and ice state. Where
  the lake has no usable analysis of its own, explicitly ice-free model
  water supplies the cells where that provider is admissible. Declined
  cells use the component skin field and are counted in the receipt. This
  represents the source model's local lake state even where its grid does
  not resolve that lake, without searching for another basin's water.
* A water cell its provider still leaves without an admissible
  temperature is filled, never refused: from its own body's water, else
  from the nearest source water on the domain, else from the skin
  temperature around it.  That is the one place a body can take another
  basin's water, and only for a body the route hands no admissible
  temperature of its own.  Every such cell is counted and named in the
  receipt's ``water_fill`` and in the advisory.  The masked skin chain
  no longer hands a lake 0 K where the source holds no water (it takes
  the source's land skin there, see
  ``woof.ingest.horiz._skin_temperature_on_both_surfaces``), so this
  is what keeps a route that maps its skin some other way from refusing
  a lake over its source's coverage.

Where the guarantee is enforced
-------------------------------
Not route by route.  :func:`require_assembled_water_temperature` is called by
the soil preprocessing routers, which are the single seam every forcing route
already crosses, and it REFUSES a mapping that still carries a raw SST beside
its SKINTEMP with no assembled field to consume.  A new forcing source
therefore cannot reintroduce the per-cell fuse by omission: it either routes
its own statics through :func:`assemble_for_route` or it is refused by name.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sys
from types import MappingProxyType
from typing import Mapping

import numpy as np
from woof.ingest.lake_temperature import (
    LAKE_WATER_PROVIDER, MAX_LISTED_DECLINED_CELLS,
)

#: The MODIS inland-water category.  Kept as documentation of the value the
#: MODIS land-use tables carry; NOTHING selects lakes with it.  The lake
#: category is read from the selected land-use table's own ``ISLAKE``
#: (:func:`lake_mask_from_landuse`), because USGS 24-category has no
#: inland-water class at all and WPS writes ``ISLAKE = -1`` there.
MODIS_LAKE_CATEGORY = 21

#: Physical admissibility, the same window the soil reconciler applies.
MIN_WATER_TEMPERATURE_K = 170.0
MAX_WATER_TEMPERATURE_K = 400.0

#: Declared policies.  Silence selects the first one.
WATER_TEMPERATURE_POLICIES = (
    "era5_class_coherent",
    "wrf_compat",
    "external_overlay",
)
DEFAULT_WATER_TEMPERATURE_POLICY = "era5_class_coherent"

#: WATER_TEMP_SOURCE provider ids.  Diagnostic only; nothing branches on them.
SOURCE_LAND = 0
SOURCE_ANALYSIS = 1        #: ERA5 SST, same-component donors
SOURCE_COMPONENT_SKIN = 2  #: coherent SKINTEMP for a whole component
SOURCE_PER_CELL = 3        #: the historical per-cell fuse (wrf_compat)
SOURCE_LAKE_WATER = 4      #: explicit ice-free source lake-model state
SOURCE_NEAREST_WATER = 5   #: filled from the nearest source water temperature
SOURCE_SURROUNDING_SKIN = 6  #: filled from the skin temperature around it
SOURCE_NAMES = {
    SOURCE_LAND: "land",
    SOURCE_ANALYSIS: "era5_sst_component",
    SOURCE_COMPONENT_SKIN: "era5_skintemp_component",
    SOURCE_PER_CELL: "per_cell_legacy",
    SOURCE_LAKE_WATER: LAKE_WATER_PROVIDER,
    SOURCE_NEAREST_WATER: "nearest_source_water",
    SOURCE_SURROUNDING_SKIN: "surrounding_skin",
}

#: A component takes the analysis when its own donors cover at least this
#: much of it.  Below the bar the whole component takes SKINTEMP instead,
#: because a sparse donor set cannot describe the body coherently.
MIN_COMPONENT_COVERAGE = 0.5


def validate_water_temperature_policy(policy):
    """Return a known policy name, or refuse by name."""
    if policy is None:
        return DEFAULT_WATER_TEMPERATURE_POLICY
    text = str(policy)
    if text not in WATER_TEMPERATURE_POLICIES:
        raise ValueError(
            f"water_temperature_policy {text!r} is not one of "
            + ", ".join(repr(name) for name in WATER_TEMPERATURE_POLICIES))
    return text


def resolve_water_temperature_policy(case_data):
    """The policy a case's declarations select.

    Declaration wins.  Absent a declaration, a declared overlay selects
    ``external_overlay``, because an overlay has already replaced both
    source fields over water and its precedence predates this policy.
    Silence with no overlay is the default, ``era5_class_coherent``.
    """
    declared = getattr(case_data, "water_temperature_policy", None)
    if declared is not None:
        return validate_water_temperature_policy(declared)
    if getattr(case_data, "water_temperature_overlay", None) is not None:
        return "external_overlay"
    return DEFAULT_WATER_TEMPERATURE_POLICY


# ---------------------------------------------------------------------------
# connected components, without adding a dependency
# ---------------------------------------------------------------------------
def _label_components(mask):
    """Label 8-connected components of a boolean mask (1-based, 0 = off).

    Eight-connectivity on purpose: a lake pinched to a diagonal on a coarse
    grid is one lake, not two.  Merging distinct BODIES is prevented by the
    caller, which labels each surface class separately.

    Runs in the Rust preprocessing library (``gpuwm_label_components_8``),
    whose labels equal the per-cell union-find kept as its test oracle
    (``woof/verify/water_blend_oracle.py``).
    """
    from woof.ingest.cpu_backend import water_blend_backend

    return water_blend_backend().label_components(mask)


#: Component labels are a pure function of two invariant statics, but the
#: assembly runs once per forcing time, so an N-time run paid for the same
#: union-find N times: measured 0.19 s at 550 squared and 0.61 s at 1000
#: squared, on the launch-to-first-plot path.  Keyed by the CONTENT of the
#: two masks, so a parent and its children never alias and a caller that
#: rebuilds its statics still hits.
#:
#: Four entries because the live set is one per domain and this tree's
#: deepest hierarchy is four; each holds one int32 label field, so the bound
#: also caps what a large domain can park here (16 MiB per entry at 2000
#: squared).  Concurrent child preparation can race the eviction, which
#: costs a recompute and never a wrong answer: the value is a pure function
#: of the key.
_LABEL_CACHE: dict[tuple, tuple] = {}
_LABEL_CACHE_LIMIT = 4


def _mask_key(mask):
    packed = np.packbits(np.ascontiguousarray(mask, dtype=bool))
    return (mask.shape, hashlib.blake2b(packed.tobytes(),
                                        digest_size=16).hexdigest())


def label_surface_components(target_land, target_lake):
    """Components of water, with lakes and ocean kept strictly apart.

    Returns ``(labels, classes)`` where ``classes[label]`` is ``"lake"`` or
    ``"ocean"``.  Labelling each class separately is what makes "never an
    ocean donor for a lake" a property of the data structure rather than a
    check someone has to remember to write.

    The result is memoized on the two masks' content.  ``labels`` is handed
    back read-only, because it is shared with every other caller holding the
    same statics.
    """
    land = np.asarray(target_land, dtype=bool)
    lake = np.asarray(target_lake, dtype=bool) & ~land
    key = (_mask_key(land), _mask_key(lake))
    cached = _LABEL_CACHE.get(key)
    if cached is not None:
        return cached
    ocean = (~land) & ~lake

    lake_labels, n_lake = _label_components(lake)
    ocean_labels, n_ocean = _label_components(ocean)
    labels = np.where(lake_labels > 0, lake_labels,
                      np.where(ocean_labels > 0, ocean_labels + n_lake, 0))
    classes = {}
    for k in range(1, n_lake + 1):
        classes[k] = "lake"
    for k in range(1, n_ocean + 1):
        classes[k + n_lake] = "ocean"
    labels = labels.astype(np.int32)
    labels.flags.writeable = False
    entry = (labels, MappingProxyType(classes))
    if len(_LABEL_CACHE) >= _LABEL_CACHE_LIMIT:
        _LABEL_CACHE.pop(next(iter(_LABEL_CACHE)))
    _LABEL_CACHE[key] = entry
    return entry


# ---------------------------------------------------------------------------
# coherent interpolation
# ---------------------------------------------------------------------------
def _target_longitude_in_source_frame(source_lon, target_lon):
    """Target longitudes expressed in the source's own 360-degree window.

    A Lambert grid reports longitude in -180..180 while an ERA5 crop may be
    stored in 0..360 or the other way round.  Comparing them directly puts
    every target a whole turn away from its donors, so shift by whole turns
    until the target sits in the source's window.
    """
    lon = np.asarray(source_lon, dtype=np.float64)
    target = np.asarray(target_lon, dtype=np.float64).copy()
    origin = lon[0]
    return origin + np.mod(target - origin + 180.0, 360.0) - 180.0


def _bilinear_corners(source_lat, source_lon, target_lat, target_lon):
    ny, nx = np.asarray(source_lat).size, np.asarray(source_lon).size
    lat = np.asarray(source_lat, dtype=np.float64)
    lon = np.asarray(source_lon, dtype=np.float64)
    y = (np.asarray(target_lat, dtype=np.float64) - lat[0]) / (lat[1] - lat[0])
    x = (_target_longitude_in_source_frame(lon, target_lon)
         - lon[0]) / (lon[1] - lon[0])
    y = np.clip(y, 0.0, ny - 1.0000001)
    x = np.clip(x, 0.0, nx - 1.0000001)
    j0 = np.floor(y).astype(np.intp)
    i0 = np.floor(x).astype(np.intp)
    fy, fx = y - j0, x - i0
    j1 = np.minimum(j0 + 1, ny - 1)
    i1 = np.minimum(i0 + 1, nx - 1)
    return ((j0, i0, (1.0 - fy) * (1.0 - fx)),
            (j0, i1, (1.0 - fy) * fx),
            (j1, i0, fy * (1.0 - fx)),
            (j1, i1, fy * fx))


def normalized_masked_bilinear(field, donors, corners, shape,
                               denominator_floor=1e-6, *, workers=None):
    """Bilinear interpolation renormalized over the donors that exist.

    Interpolates ``field * donors`` and ``donors`` separately and divides.
    A target whose stencil holds one usable donor still gets that donor's
    value at full weight instead of being abandoned, and a target with no
    donor weight at all returns NaN rather than a fill that later reads as
    a temperature.  Runs in the Rust preprocessing library
    (``gpuwm_masked_bilinear_blend_f64``), byte-identical to the NumPy
    code kept as its test oracle (``woof/verify/water_blend_oracle.py``).
    """
    from woof.ingest.cpu_backend import water_blend_backend

    return water_blend_backend().masked_bilinear_blend(
        field, donors, corners, shape, denominator_floor=denominator_floor,
        workers=workers)


def _fill_within_component(values, component, max_sweeps=1000, *,
                           workers=None):
    """Close residual holes from the component's OWN cells, never elsewhere.

    Runs in the Rust preprocessing library (``gpuwm_component_fill_f64``),
    byte-identical to the NumPy sweep kept as its test oracle
    (``woof/verify/water_blend_oracle.py``).
    """
    from woof.ingest.cpu_backend import water_blend_backend

    return water_blend_backend().component_fill(
        values, component, max_sweeps=max_sweeps, workers=workers)


def _admissible(values):
    """Finite and inside the physical window the soil reconciler applies."""
    return (np.isfinite(values)
            & (values >= MIN_WATER_TEMPERATURE_K)
            & (values <= MAX_WATER_TEMPERATURE_K))


def _fill_missing_water_temperature(values, source, water, labels, *,
                                    workers=None):
    """Give a temperature to every water cell its provider left without one.

    A cell reaches here when the provider its body chose left it no
    admissible temperature.  On a regional crop of a coarse source that is
    the ordinary case for a lake: the source resolves no water anywhere
    near it, the masked skin interpolation had no water donor to read, and
    the chain left its fill value, zero.  Refusing the preparation for it
    refused a domain over the extent of the source crop, so each such cell
    is filled, in this order, and every fill is counted:

    1. from its own body's admissible water, ring by ring from the cells
       that have it;
    2. a body with none takes, as one value, the nearest admissible water
       on the domain: source water one of the providers above carried in.
       "Near" has the reach the masked source-water search already has,
       the whole of the data the preparation holds;
    3. with no admissible water on the domain, the body takes the skin
       temperature around it, ring by ring from its shore, and a body
       whose shore has none takes the nearest admissible skin.

    Returns ``(values, source, counts, filled)``.  A cell none of the three
    can reach stays inadmissible for the caller's refusal: nothing on the
    domain carries an admissible surface temperature, which is a defect of
    the source rather than of its crop.

    Runs in the Rust preprocessing library (``gpuwm_water_repair_f64``),
    byte-identical to the NumPy box repairs kept as its test oracle
    (``woof/verify/water_blend_oracle.py``).
    """
    from woof.ingest.cpu_backend import water_blend_backend

    return water_blend_backend().water_repair(
        values, source, water, labels, minimum=MIN_WATER_TEMPERATURE_K,
        maximum=MAX_WATER_TEMPERATURE_K,
        nearest_water_code=SOURCE_NEAREST_WATER,
        surrounding_skin_code=SOURCE_SURROUNDING_SKIN, workers=workers)


def _component_owner_of_source(labels, source_lat, source_lon,
                               target_lat, target_lon, source_shape, *,
                               workers=None):
    """Which target component each source cell belongs to (0 = none).

    A source cell is claimed by the component holding most of the target
    cells that fall nearest to it, and among components holding equally
    many by the highest label.  That is what confines a component's
    donor set to its own body: donors are selected by identity, not by
    distance, so no radius can leak another basin in.

    Runs in the Rust preprocessing library (``gpuwm_component_owner_f64``),
    byte-identical to the NumPy code kept as its test oracle
    (``woof/verify/water_blend_oracle.py``).  That code sorted the counts
    with NumPy's default unstable ``argsort``, so which of two equally
    large claims won depended on the machine's sort kernel; the oracle
    now sorts stably, which is the rule stated above.
    """
    from woof.ingest.cpu_backend import water_blend_backend

    return water_blend_backend().component_owner(
        labels, source_lat, source_lon, target_lat, target_lon,
        source_shape, workers=workers)


# ---------------------------------------------------------------------------
# the assembled field
# ---------------------------------------------------------------------------
def _water_temperature_refusal(*, bad, values, skin, mapped_sst, source,
                               labels, component_rows, diagnostic_context,
                               diagnostic_latlon):
    """Bounded evidence for a refusal, without changing any provider choice.

    Reached only when no cell of the domain, water or land, carries an
    admissible temperature to fill a water cell from, so the source's
    surface temperature itself is missing here.
    """
    bad_count = int(bad.sum())
    cells = np.argwhere(bad)[:8]
    lines = [
        f"{bad_count} water cells have no admissible water temperature "
        "after class-coherent assembly, and no cell of this domain carries "
        "an admissible water or skin temperature to fill them from: the "
        "source's surface temperature is missing here",
        f"Context: {diagnostic_context or 'direct water-temperature assembly'}; "
        f"mass grid {values.shape[0]}x{values.shape[1]}",
        f"Bad cells (zero-based row/column; showing {len(cells)} of "
        f"{bad_count}; admissible range {MIN_WATER_TEMPERATURE_K:g}.."
        f"{MAX_WATER_TEMPERATURE_K:g} K):",
    ]
    latlon = None
    if diagnostic_latlon is not None:
        latitude, longitude = (np.asarray(axis) for axis in diagnostic_latlon)
        if latitude.shape == values.shape and longitude.shape == values.shape:
            latlon = latitude, longitude
    sst = None if mapped_sst is None else np.asarray(mapped_sst)
    if sst is not None and sst.shape != values.shape:
        sst = None
    shown_labels = set()
    for row, column in cells:
        label = int(labels[row, column])
        shown_labels.add(label)
        location = f"row={row}, column={column}"
        if latlon is not None:
            location += (f", lat={latlon[0][row, column]:.6f}, "
                         f"lon={latlon[1][row, column]:.6f}")
        sst_value = ("unavailable" if sst is None
                     else f"{sst[row, column]:.8g}")
        lines.append(
            f"  {location}: assembled={values[row, column]:.8g} K, "
            f"mapped_SKINTEMP={skin[row, column]:.8g} K, "
            f"mapped_SST={sst_value} K, "
            f"provider={SOURCE_NAMES[int(source[row, column])]}, "
            f"component={label}")
    lines.append("Components containing the shown cells:")
    for item in component_rows:
        if item["label"] in shown_labels:
            lines.append(
                f"  component={item['label']}, class={item['class']}, "
                f"cells={item['cells']}, SST_donors={item['donors']}, "
                f"SST_coverage={item['coverage']:.6g}, "
                f"selected_provider={item['provider']}")
    return ValueError("\n".join(lines))


def assemble_water_temperature(
        *, mapped_sst, mapped_skin, target_land, target_lake,
        source_sst=None, source_lat=None, source_lon=None,
        target_lat=None, target_lon=None,
        policy=DEFAULT_WATER_TEMPERATURE_POLICY,
        diagnostic_context=None, diagnostic_latlon=None,
        mapped_lake_water=None, lake_ice=None, workers=None):
    """Return ``(water_temperature, water_temperature_source, receipt)``.

    ``water_temperature`` is finished: every water cell carries a physical
    temperature attributed to its provider. Analysis is chosen for a whole
    connected body; optional lake-model water falls back to component skin
    at declined cells. Land cells carry mapped skin so the array is total,
    and the soil reconciler still decides what land does with it.  A water
    cell its provider left without an admissible temperature is filled by
    :func:`_fill_missing_water_temperature` and counted in the receipt's
    ``water_fill``.

    ``lake_ice`` is the mask of lake cells the source lake model covers
    with ice (:func:`woof.ingest.lake_temperature.map_lake_ice`).  It
    changes no value here: an ice cell's skin is the ice's, set by the
    caller, and the soil route never reads an ice cell's water
    temperature.  It only keeps those cells out of the receipt's fallback
    count, because they did not fall back to anything.

    The optional diagnostic context and geographic latitude/longitude pair
    are used only to explain a refusal; they never enter interpolation.
    ``workers`` is the preparation's host-step thread count
    (:func:`woof.ingest.cpu_backend.host_step_workers`); ``None`` takes
    the automatic count.  No value depends on it.
    """
    policy = validate_water_temperature_policy(policy)
    skin = np.asarray(mapped_skin, dtype=np.float64)
    land = np.asarray(target_land, dtype=bool)
    if skin.shape != land.shape:
        raise ValueError("mapped_skin and target_land shapes differ")
    shape = skin.shape
    water = ~land
    values = skin.copy()
    source = np.full(shape, SOURCE_LAND, dtype=np.int8)

    if policy in ("wrf_compat", "external_overlay"):
        # Byte-for-byte the historical selector, kept reachable so a
        # stock-WRF certification can still ask for it by name.  An absent
        # SST is the selector's own "no SST here", which is how the shipped
        # code read a source that carries none.
        #
        # A declared overlay takes this arm too, and takes it unchanged: the
        # overlay has already written one analysis into BOTH source fields
        # over water, so the per-cell choice has nothing left to switch
        # between, and every fingerprint an overlay run published before
        # this module existed still reproduces.
        sst = (np.full(shape, np.nan, dtype=np.float64) if mapped_sst is None
               else np.asarray(mapped_sst, dtype=np.float64))
        admissible = (np.isfinite(sst)
                      & (sst >= MIN_WATER_TEMPERATURE_K)
                      & (sst <= MAX_WATER_TEMPERATURE_K))
        values = np.where(admissible, sst, skin)
        source = np.where(water, np.int8(SOURCE_PER_CELL),
                          np.int8(SOURCE_LAND)).astype(np.int8)
        receipt = {
            "policy": policy,
            "water_cells": int(water.sum()),
            "components": 0,
            "per_provider": {
                SOURCE_NAMES[SOURCE_PER_CELL]: int(water.sum())},
            "components_on_analysis": 0,
            "components_on_skin": 0,
        }
        return values.astype(np.float64), source, receipt

    lake = np.asarray(target_lake, dtype=bool) & water
    lake_water = (None if mapped_lake_water is None else
                  np.asarray(mapped_lake_water, dtype=np.float64))
    if lake_water is not None and lake_water.shape != shape:
        raise ValueError("mapped_lake_water and target_land shapes differ")
    labels, classes = label_surface_components(land, lake)

    per_provider = {name: 0 for name in SOURCE_NAMES.values()}
    on_analysis = 0
    on_skin = 0
    on_lake_water = 0
    lake_fallback = 0
    lake_fallback_cells: list = []
    component_rows = []

    have_source = (source_sst is not None and source_lat is not None
                   and source_lon is not None and target_lat is not None
                   and target_lon is not None)
    if have_source:
        source_sst = np.asarray(source_sst, dtype=np.float64)
        corners = _bilinear_corners(source_lat, source_lon,
                                    target_lat, target_lon)
        owner = _component_owner_of_source(
            labels, source_lat, source_lon, target_lat, target_lon,
            source_sst.shape, workers=workers)
    else:
        corners = None
        owner = None

    # Every body's cells, donors, blend, fill and provider in one pass of
    # the Rust preprocessing library (gpuwm_water_bodies_f64), rather than
    # masks over the whole domain once per body; the NumPy loop is its
    # test oracle (woof/verify/water_blend_oracle.py).
    from woof.ingest.cpu_backend import water_blend_backend

    ordered = sorted(classes)
    if ordered != list(range(1, len(ordered) + 1)):
        raise ValueError(
            "water bodies must be labelled 1..N for the per-body assembly")
    lake_class = np.zeros(len(ordered) + 1, dtype=bool)
    for label in ordered:
        lake_class[label] = classes[label] == "lake"
    stats, coverages, listed = water_blend_backend().water_bodies(
        labels=labels, lake_class=lake_class, skin=skin, values=values,
        source=source, lake_water=lake_water,
        codes=(SOURCE_ANALYSIS, SOURCE_COMPONENT_SKIN, SOURCE_LAKE_WATER),
        sst=source_sst if have_source else None,
        owner=owner, corners=corners,
        min_coverage=MIN_COMPONENT_COVERAGE,
        minimum=MIN_WATER_TEMPERATURE_K, maximum=MAX_WATER_TEMPERATURE_K,
        max_listed=MAX_LISTED_DECLINED_CELLS, workers=workers)
    provider_names = {1: SOURCE_NAMES[SOURCE_ANALYSIS],
                      2: SOURCE_NAMES[SOURCE_COMPONENT_SKIN],
                      3: SOURCE_NAMES[SOURCE_LAKE_WATER]}
    for label in ordered:
        (cells, donor_count, provider, analysis_cells, skin_cells,
         lake_cells, declined) = (int(value) for value in stats[label])
        if cells == 0:
            continue
        per_provider[SOURCE_NAMES[SOURCE_ANALYSIS]] += analysis_cells
        per_provider[SOURCE_NAMES[SOURCE_COMPONENT_SKIN]] += skin_cells
        per_provider[SOURCE_NAMES[SOURCE_LAKE_WATER]] += lake_cells
        if provider == 1:
            on_analysis += 1
        elif provider == 3:
            on_lake_water += 1
        else:
            on_skin += 1
        lake_fallback += declined
        component_rows.append({
            "label": int(label), "class": classes[label], "cells": cells,
            "donors": donor_count, "coverage": float(coverages[label]),
            "provider": provider_names[provider]})
    lake_fallback_cells.extend([int(j), int(i)] for j, i in listed)
    ice_cells = None
    if lake_ice is not None and lake_water is not None:
        ice = np.asarray(lake_ice, dtype=bool)
        if ice.shape != shape:
            raise ValueError("lake_ice and target_land shapes differ")
        # The cells the assembly counted as declined: every cell of a lake
        # body that did not take the analysis and that the lake-water
        # provider left without a value (water_blend_oracle's rule).
        counted = [row["label"] for row in component_rows
                   if row["class"] == "lake"
                   and row["provider"] != SOURCE_NAMES[SOURCE_ANALYSIS]]
        declined = np.isin(labels, counted) & ~np.isfinite(lake_water)
        if int(declined.sum()) != lake_fallback:
            raise ValueError(
                "the lake fallback count disagrees with the declined lake "
                f"cells ({lake_fallback} against {int(declined.sum())})")
        ice &= lake
        ice_cells = int(ice.sum())
        still_open = declined & ~ice
        lake_fallback = int(still_open.sum())
        lake_fallback_cells = [
            [int(j), int(i)] for j, i in
            np.argwhere(still_open)[:MAX_LISTED_DECLINED_CELLS]]

    values, source, fill_counts, filled = _fill_missing_water_temperature(
        values, source, water, labels, workers=workers)
    bad = water & ~_admissible(values)
    if np.any(bad):
        raise _water_temperature_refusal(
            bad=bad, values=values, skin=skin, mapped_sst=mapped_sst,
            source=source, labels=labels, component_rows=component_rows,
            diagnostic_context=diagnostic_context,
            diagnostic_latlon=diagnostic_latlon)
    if np.any(water & (source == SOURCE_LAND)):
        raise ValueError("a water cell was left without a declared provider")
    if np.any(filled):
        # A filled cell changed provider, so the tally is read back off
        # the provider field rather than kept beside it.
        per_provider = {
            name: int(np.count_nonzero(water & (source == key)))
            for key, name in SOURCE_NAMES.items()}

    receipt = {
        "policy": policy,
        "water_cells": int(water.sum()),
        "lake_cells": int(lake.sum()),
        "ocean_cells": int((water & ~lake).sum()),
        "components": len(component_rows),
        "components_on_analysis": on_analysis,
        "components_on_skin": on_skin,
        "per_provider": {k: v for k, v in per_provider.items() if v},
        "component_detail": component_rows,
    }
    if lake_water is not None:
        receipt["components_on_lake_water"] = on_lake_water
        receipt["lake_fallback_cells"] = lake_fallback
        receipt["lake_fallback_cell_indices"] = lake_fallback_cells
    if ice_cells is not None:
        receipt["lake_ice_cells"] = ice_cells
    if np.any(filled):
        receipt["water_fill"] = {
            "cells": int(filled.sum()),
            **fill_counts,
            "cell_indices": [
                [int(j), int(i)]
                for j, i in np.argwhere(filled)[:MAX_LISTED_DECLINED_CELLS]],
        }
    return values, source, receipt


# ---------------------------------------------------------------------------
# the one seam every forcing route crosses
# ---------------------------------------------------------------------------
#: The mapped fields the historical per-cell selector chose BETWEEN.  A
#: mapping that carries both is a mapping the fuse can silently quilt.
_RAW_WATER_PAIR = ("SST", "SKINTEMP")


def lake_mask_from_landuse(lu_index, landuse_attrs, *, route):
    """``(mask, lake_category)`` for the target's inland water.

    The category is the selected land-use table's OWN ``ISLAKE``, never a
    constant: MODIS writes 21, and USGS 24-category has no inland-water
    class at all, for which WPS writes ``ISLAKE = -1``.

    ``lake_category`` is ``None`` in exactly that second case, and the mask
    is then all-False because the table genuinely cannot name a lake.  The
    caller must state that rather than let it pass as "this domain has no
    lakes" -- which is the failure mode this returns two values to prevent.
    A mapping with no ``ISLAKE`` at all is a refusal, because that is a
    caller that never resolved its land-use table.
    """
    if landuse_attrs is None or "ISLAKE" not in landuse_attrs:
        raise ValueError(
            f"{route}: the water-temperature assembly needs the selected "
            "land-use table's ISLAKE to tell an inland body from the ocean; "
            "the caller supplied no land-use attributes.  Resolve them from "
            "the GEOG index (GeogSelection.landuse_global_attrs) rather "
            "than assuming a category number.")
    category = int(landuse_attrs["ISLAKE"])
    if category < 1:
        return np.zeros(np.shape(lu_index), dtype=bool), None
    return np.asarray(lu_index) == category, category


#: ``eq=False`` on both containers below: they hold arrays, and a
#: generated ``__eq__`` would compare those elementwise and then die on
#: the truth value of the result.  Identity comparison is what callers
#: of these actually want.
@dataclass(frozen=True, eq=False)
class WaterTemperatureStatics:
    """One route's declaration of the surface it is assembling over.

    ``route`` names the caller in every refusal and rides the receipt, so a
    false or missing assembly can be attributed without reading the
    traceback.  ``lake_category`` is ``None`` when the land-use table names
    no lake class; the advisory says so out loud.
    """

    route: str
    land: np.ndarray
    lake: np.ndarray
    lake_category: int | None
    policy: str
    #: The route's daily-mean 2 m air temperature for inland water the
    #: source does not resolve (:class:`woof.ingest.horiz.DailyMeanAirTemperature`,
    #: WRF's TAVGSFC); ``None`` takes each forcing time's own 2 m air
    #: temperature there.
    inland_air_temperature: object = None

    @classmethod
    def for_route(cls, *, route, policy, landmask, lu_index, landuse_attrs,
                  lake_override=False):
        """Build the statics from a route's own geogrid fields.

        ``landmask`` is the geogrid ``LANDMASK`` the soil router will decide
        land and water with, so the two agree by construction.

        ``lake_override=True`` when the caller ALSO hands the soil router a
        ``lake_mask``/``lake_skin_temperature`` pair.  That override makes
        every land-use lake cell water inside the router, so the assembly
        has to class it as water too; otherwise the router would read a land
        value out of the assembled field on exactly those cells.
        """
        land = np.asarray(landmask) >= 0.5
        lake, category = lake_mask_from_landuse(
            lu_index, landuse_attrs, route=route)
        if lake.shape != land.shape:
            raise ValueError(f"{route}: LU_INDEX and LANDMASK shapes differ")
        if lake_override:
            land = land & ~lake
        return cls(route=str(route), land=land, lake=lake,
                   lake_category=category,
                   policy=validate_water_temperature_policy(policy))


@dataclass(frozen=True, eq=False)
class WaterTemperatureAssembly:
    """What a route hands the soil router, and the receipt that names it."""

    values: np.ndarray
    provider: np.ndarray
    receipt: Mapping[str, object]
    statics: WaterTemperatureStatics


def assemble_horizontal_water_temperature(horizontal, statics, *,
                                          workers=None):
    """Attach the common finished water surface to a mapped snapshot.

    Adapter-specific interpolation has already happened. This step depends
    only on canonical surface fields and the target's declared geography.
    """
    from dataclasses import replace

    def host(value):
        return value.get() if hasattr(value, "get") else value

    fields = horizontal.fields
    assembly = assemble_for_route(
        statics, mapped_skin=host(fields["SKINTEMP"]),
        mapped_sst=(None if "SST" not in fields else host(fields["SST"])),
        diagnostic_context=(
            f"valid_time={horizontal.valid_time.isoformat()} UTC"),
        workers=workers)
    return replace(horizontal, water_temperature=assembly.values,
                   water_temperature_source=assembly.provider,
                   water_temperature_receipt=assembly.receipt)


def assemble_for_route(statics, *, mapped_sst, mapped_skin, source_sst=None,
                       source_lat=None, source_lon=None,
                       target_lat=None, target_lon=None,
                       diagnostic_context=None, diagnostic_latlon=None,
                       mapped_lake_water=None, lake_ice=None, workers=None):
    """THE assembly entry point.  Every forcing route reaches it here.

    Closing this route by route is what produced the quilt in the first
    place, so there is one function, it takes the route's own statics, and
    :func:`require_assembled_water_temperature` refuses the routes that
    have not been through it.
    """
    if not isinstance(statics, WaterTemperatureStatics):
        raise TypeError("assemble_for_route needs WaterTemperatureStatics")
    values, provider, receipt = assemble_water_temperature(
        mapped_sst=mapped_sst, mapped_skin=mapped_skin,
        target_land=statics.land, target_lake=statics.lake,
        source_sst=source_sst, source_lat=source_lat, source_lon=source_lon,
        target_lat=target_lat, target_lon=target_lon,
        policy=statics.policy,
        diagnostic_context=(statics.route if diagnostic_context is None
                            else f"{statics.route}; {diagnostic_context}"),
        diagnostic_latlon=diagnostic_latlon,
        mapped_lake_water=mapped_lake_water, lake_ice=lake_ice,
        workers=workers)
    receipt = dict(receipt)
    receipt["route"] = statics.route
    receipt["lake_class"] = (
        "declared" if statics.lake_category is not None
        else "absent_from_land_use_table")
    return WaterTemperatureAssembly(
        values=values, provider=provider, receipt=receipt, statics=statics)


def require_assembled_water_temperature(*, route, fields, water_temperature,
                                        policy):
    """Refuse a soil call that would silently rerun the per-cell fuse.

    The fuse is ``where(admissible(SST), SST, SKINTEMP)``.  It can only
    DIFFER from the assembled field where the mapping actually carries an
    SST beside its SKINTEMP -- absent an SST the selector reduces to
    ``SKINTEMP`` on every water cell, which is exactly what the assembly
    returns for a body with no donors, so those routes are the identity and
    are let through.  Where a mapped SST does exist, silence is the defect:
    the caller has two differently-mapped fields and no decision, so it is
    refused by name and told where the decision belongs.

    ``wrf_compat`` is the one declaration that reopens the fuse, because
    reproducing stock WRF byte for byte is the whole reason that policy has
    a name.
    """
    if water_temperature is not None:
        return
    if policy == "wrf_compat":
        return
    if not all(name in fields for name in _RAW_WATER_PAIR):
        return
    raise ValueError(
        f"{route or 'an unnamed forcing route'} handed soil preprocessing a "
        "raw SST beside its SKINTEMP with no assembled water temperature.  "
        "Choosing between two differently-mapped fields cell by cell is "
        "what painted lakes as a quilt of both.  Route this source's own "
        "statics through "
        "woof.ingest.water_temperature.assemble_for_route and forward the "
        "result as water_temperature, or declare "
        "water_temperature_policy = \"wrf_compat\" to ask for the "
        "historical selector on purpose.")


def water_temperature_advisory(receipt):
    """One plain line naming the policy and what it did, or None."""
    if not receipt or receipt.get("policy") != "era5_class_coherent":
        return None
    if not receipt.get("water_cells"):
        return None
    counts = ", ".join(
        f"{name} {count}"
        for name, count in sorted(receipt.get("per_provider", {}).items()))
    route = receipt.get("route")
    where = f" on {route}" if route else ""
    # LOUD when the guarantee is narrower than the headline: a land-use
    # table with no lake class cannot keep a lake and the ocean in separate
    # provider decisions, and silence there would read as "this domain has
    # no lakes".
    caveat = ""
    if receipt.get("lake_class") == "absent_from_land_use_table":
        caveat = (" The selected land-use table names no lake category, so "
                  "inland water is classed with the ocean here and a lake "
                  "joined to the sea by a coarse coastline can share its "
                  "provider.")
    # NAMED: the lake cells that start as ice from the source lake model.
    ice_cells = int(receipt.get("lake_ice_cells") or 0)
    if ice_cells:
        caveat += (
            f" {ice_cells} lake cell(s) start as lake ice from the source "
            "lake model (ice depth above zero over at least half the cell's "
            "interpolation weight), at its ice-surface temperature (see "
            "lake_ice_mapping in the receipt).")
    # NAMED, never silent: every lake cell the lake-model provider declined
    # (frozen or unknown ice depth, inadmissible temperature), that the lake
    # model does not cover with ice, and that kept the component skin
    # temperature the pre-lake-model route used (ENG-008).
    fallback = int(receipt.get("lake_fallback_cells") or 0)
    if fallback:
        listed = receipt.get("lake_fallback_cell_indices") or []
        cells = ", ".join(f"({j}, {i})" for j, i in listed)
        more = ("" if len(listed) >= fallback
                else f" and {fallback - len(listed)} more")
        caveat += (
            f" {fallback} lake cell(s) had no ice-free lake-model water "
            "(partly frozen or unknown ice depth, or an out-of-range "
            "temperature; see lake_water_mapping in the receipt) and kept "
            "the component "
            f"skin temperature the pre-lake-model route used: cells (j, i) "
            f"{cells}{more}.")
    # NAMED, never silent: every water cell no provider could give an
    # admissible temperature, and what it was filled from.
    fill = receipt.get("water_fill") or {}
    if fill.get("cells"):
        listed = fill.get("cell_indices") or []
        cells = ", ".join(f"({j}, {i})" for j, i in listed)
        more = ("" if len(listed) >= fill["cells"]
                else f" and {fill['cells'] - len(listed)} more")
        caveat += (
            f" {fill['cells']} water cell(s) had no water temperature in "
            f"the source and were filled: {fill.get('own_body', 0)} from "
            f"their own body's water, {fill.get('nearest_water', 0)} from "
            "the nearest source water, "
            f"{fill.get('surrounding_skin', 0)} from the skin temperature "
            f"around them (see water_fill in the receipt): cells (j, i) "
            f"{cells}{more}.")
    return (
        f"water temperature: policy era5_class_coherent{where} over "
        f"{receipt['water_cells']} water cells in {receipt['components']} "
        f"connected bodies ({counts}).{caveat} Declaring "
        "water_temperature_overlay with an observational analysis remains "
        "the higher-accuracy option.")


def announce_water_temperature(receipt, *, seen=None, scope=None):
    """Print the advisory once per ``(scope, message)``, or do nothing.

    ``scope`` is whatever identifies the domain to the caller (the mass-grid
    shape, on the mapping path).  Every forcing time of a domain assembles
    the same providers over the same cells, so the line belongs to the
    domain and not to the snapshot.
    """
    advisory = water_temperature_advisory(receipt)
    if advisory is None:
        return None
    if seen is not None:
        key = (scope, advisory)
        if key in seen:
            return advisory
        seen.add(key)
    print(advisory, file=sys.stderr)
    return advisory

def model_sst_with_skin_fill(raw_sst, *, skin, landmask):
    """The model's SST carrier: the source SST, with TSK where water has none.

    Returns ``(sst, receipt)``.  Every forcing route's real-case
    initialization (:func:`woof.runtime._initialize_real_case_physics`)
    calls this once, so the rule is one rule for every source, not a
    per-source branch.

    The mapped SST is an analysis that is undefined over the source's land,
    and the horizontal stencil leaves ``fill_missing = 0`` within a source
    cell or two of every coastline.  The finished water TEMPERATURE
    (:func:`assemble_water_temperature`, carried in TSK) already fills
    those cells; the raw SST handed to the model as its ``sst`` carrier did
    not, so it reached the physics as 0 K.  real.exe's rule where a water
    column has no SST is ``sst = tsk`` (WRF 4.6.1 module_initialize_real.F
    :2815-2823, a water cell whose SST is outside 150..400 K, and
    :3655-3656, the ``flag_sst /= 1`` arm), and that is what is done here: a water cell (``landmask < 0.5``) whose SST is non-finite or
    outside 170..400 K takes this cell's TSK.  Land cells and every water
    cell with an admissible SST keep their bytes; with nothing to fill the
    input array itself is returned, so a case with no such cell is
    byte-identical to before.

    THE BREAKAGE THIS PREVENTS (WOOF 2026-11-06, Pickens ERA5 WoFS
    ensemble, ERA5 2011-04-27T16Z, 12 km d01 reaching the Gulf coast): 44
    of 210 open-water cells carried sst = 0 K while TSK was valid on all
    210, so every member's SST-offset draw was refused ("offset ocean
    temperature is outside 170..400 K") after member 0.
    """
    def host(value):
        return value.get() if hasattr(value, "get") else np.asarray(value)

    skin_host = host(skin)
    if raw_sst is None:
        return skin, {"source_cells": 0, "filled_from_tsk": 0,
                      "rule": "no source SST: sst = tsk"}
    sst = host(raw_sst)
    water = host(landmask) < 0.5
    if sst.shape != water.shape or skin_host.shape != water.shape:
        raise ValueError(
            f"SST {sst.shape}, TSK {skin_host.shape} and landmask "
            f"{water.shape} shapes differ")
    fill = water & ~_admissible(sst)
    filled = int(np.count_nonzero(fill))
    receipt = {"source_cells": int(np.count_nonzero(water)) - filled,
               "filled_from_tsk": filled,
               "rule": "water cell with no admissible SST: sst = tsk"}
    if not filled:
        return raw_sst, receipt
    bad_skin = fill & ~_admissible(skin_host)
    if np.any(bad_skin):
        raise ValueError(
            f"{int(np.count_nonzero(bad_skin))} water cell(s) have neither an "
            "admissible SST nor an admissible TSK (170..400 K) to fill it from")
    out = np.array(sst, copy=True)
    out[fill] = skin_host[fill].astype(out.dtype, copy=False)
    receipt["filled_cells"] = [
        [int(j), int(i)] for j, i in np.argwhere(fill)[:MAX_LISTED_DECLINED_CELLS]]
    return out, receipt


__all__ = [
    "MODIS_LAKE_CATEGORY",
    "WATER_TEMPERATURE_POLICIES",
    "DEFAULT_WATER_TEMPERATURE_POLICY",
    "SOURCE_LAND", "SOURCE_ANALYSIS", "SOURCE_COMPONENT_SKIN",
    "SOURCE_PER_CELL", "SOURCE_LAKE_WATER", "SOURCE_NEAREST_WATER",
    "SOURCE_SURROUNDING_SKIN", "SOURCE_NAMES",
    "WaterTemperatureAssembly",
    "WaterTemperatureStatics",
    "validate_water_temperature_policy",
    "resolve_water_temperature_policy",
    "lake_mask_from_landuse",
    "label_surface_components",
    "normalized_masked_bilinear",
    "assemble_water_temperature",
    "assemble_for_route",
    "model_sst_with_skin_fill",
    "require_assembled_water_temperature",
    "water_temperature_advisory",
    "announce_water_temperature",
]
