"""WRF ``module_soil_pre.F`` soil ingest for the RUC LSM's soil levels.

This is ``init_soil_depth_3`` (:1153-1194) and ``init_soil_3_real``
(:1817-2158) of WRF v4.6.1 ``share/module_soil_pre.F``, the pair
``process_soil_real`` (:817-830) selects when ``sf_surface_physics ==
RUCLSMSCHEME``.  It is a separate module from :mod:`woof.ingest.soil`
because RUC's geometry is not a longer version of Noah's:

* Noah's ``zs`` are layer **midpoints** and its ``dzs`` are the layer
  thicknesses those midpoints sit inside.  RUC's ``zs`` are **level (node)
  depths**, and the shallowest one is ``0.0`` -- the ground surface itself,
  not the centre of a layer.  A RUC soil value is the value *at a depth*, not
  the mean *over a slab*.
* RUC's ``dzs`` is a derived control volume around each node, and WRF's
  derivation is not a partition of [0, 3] m -- see :func:`ruc_soil_depths`.

Everything here is ``float32``, in WRF's own expression order, because the
reference this module is measured against is WRF's own ``REAL`` arithmetic:
``woof/data/ruc/oracle/soil_ingest.csv``, produced by
``tools/ruc_soil_ingest_wrf461_oracle`` driving the byte-unmodified routines.
Computing in float64 and rounding once is a third function, neither float32
nor correctly rounded, and it is how this project has produced real bugs
before.

Two WRF behaviours are deliberately NOT reproduced.  Both are refused with an
error instead, and both refusals are proved against oracle rows that show
what WRF does:

1. **An unsorted layer source.**  ``init_soil_3_real``'s sort (:1881-1899,
   :1913-1931) permutes ``st_input(1..n)``, but a layer source lives in
   ``st_input(2..n+1)`` -- WRF itself puts it there
   (``share/module_optional_input.F:1364-1370``) and then writes the skin
   temperature into slot 1 and the deep temperature into slot n+2.  The sort
   therefore permutes the wrong slots and drags uninitialised memory into the
   soil column.  The oracle's ``era5_layers_reversed_adj`` rows show the
   result: soil temperatures of order -1e29 K.  metgrid emits its levels
   shallow-to-deep so WRF never trips over this, but a source adapter that
   does not is silently corrupted rather than told.
2. **A zero-moisture land column under the moisture adjustment.**
   :2079 divides by ``0.9*smtotr``; with an all-zero source profile
   ``smtotr`` is zero and the quotient is 0/0.  The oracle's
   ``era5_layers_adj_nosst`` column 10 records what this toolchain then
   returns from ``MAX(0.02, NaN)`` -- 0.02 -- but that is a property of
   gfortran's ``maxss`` operand order, not of Fortran, and a soil column
   whose value depends on it is not a number this project will ship.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import sys
from typing import Mapping

import numpy as np


#: ``init_soil_depth_3`` (:1161-1167) tabulates the level depths in metres and
#: supports exactly these two lengths.  The nine-level table is the one a
#: nine-layer RUC run uses, and it is bit-identical to the node depths a
#: RUC-native gridded source already carries.
RUC_LEVEL_DEPTHS_M: dict[int, tuple[float, ...]] = {
    6: (0.00, 0.05, 0.20, 0.40, 1.60, 3.00),
    9: (0.00, 0.01, 0.04, 0.10, 0.30, 0.60, 1.00, 1.60, 3.00),
}

#: :2110-2117 hardcodes these five factors, applied to levels 1-5 of a land
#: column whose moisture increases with depth over its top three levels.
_MOISTURE_DAYTIME_FACTORS = (0.75, 0.8, 0.85, 0.9, 0.95)

#: :1930-1934 forms the Noah source bucket with these weights against
#: ``sm_input(2..5)``.  They are hardcoded in WRF and they are NOT Noah's own
#: ``dzs`` (``init_soil_depth_2``: 0.1, 0.3, 0.6, 1.0): the total is the same
#: 2 m of column, the second and third weights are not.  Reproduced as WRF
#: has them, because this is the definition of the quantity the adjustment
#: conserves, not an independent statement about Noah's geometry.
_NOAH_BUCKET_WEIGHTS = (0.1, 0.2, 0.7, 1.0)

_LAYER_SOURCE = "layers"
_LEVEL_SOURCE = "levels"


@dataclass(frozen=True)
class RucSoilColumns:
    """``float32`` RUC soil state on the scheme's own levels."""

    soil_temperature: np.ndarray   # (num_soil_layers, ...), WRF TSLB
    soil_moisture: np.ndarray      # (num_soil_layers, ...), WRF SMOIS
    skin_temperature: np.ndarray   # (...), WRF TSK after the open-water fill
    level_depths: np.ndarray       # (num_soil_layers,), WRF ZS
    level_thicknesses: np.ndarray  # (num_soil_layers,), WRF DZS


def ruc_soil_depths(num_soil_layers: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(zs, dzs)`` exactly as ``init_soil_depth_3`` builds them.

    ``zs`` is tabulated; ``dzs`` is derived (:1174-1183) through an interface
    array ``zs2``.  The derivation has an index shift that is reproduced here
    on purpose, because ``dzs`` is an *input* to the moisture adjustment and
    the reference is WRF's value, not a corrected one:

    ``zs2(1)`` is set to 0 and ``zs2(2)`` to the first half-interval, so
    ``dzs(1)`` is that half-interval.  The loop then overwrites ``zs2(2)``
    with the interface *below* level 2 and computes ``dzs(2) = zs2(2) -
    zs2(1)`` against the zero still in ``zs2(1)``, not against the
    half-interval it just discarded.  The result is that ``dzs(1)`` and
    ``dzs(2)`` both cover [0, 0.005] m and ``sum(dzs)`` is 3.005, not 3.0.
    A partition-of-the-column assumption about ``dzs`` is therefore wrong by
    5 mm at nine levels.
    """

    try:
        table = RUC_LEVEL_DEPTHS_M[int(num_soil_layers)]
    except KeyError:
        raise ValueError(
            f"num_soil_layers={num_soil_layers}: init_soil_depth_3 tabulates "
            f"only {sorted(RUC_LEVEL_DEPTHS_M)} level depths, and :1189-1192 "
            "makes 4 and 5 fatal.  A different count needs WRF's table "
            "extended first, not a remap invented here"
        ) from None

    count = len(table)
    zs = np.asarray(table, dtype=np.float32)
    zs2 = np.zeros(count, dtype=np.float32)
    dzs = np.zeros(count, dtype=np.float32)
    half = np.float32(0.5)

    zs2[0] = np.float32(0.0)
    zs2[1] = (zs[1] + zs[0]) * half
    dzs[0] = zs2[1] - zs2[0]
    for index in range(1, count - 1):
        zs2[index] = (zs[index + 1] + zs[index]) * half
        dzs[index] = zs2[index] - zs2[index - 1]
    zs2[count - 1] = zs[count - 1]
    dzs[count - 1] = zs2[count - 1] - zs2[count - 2]
    return zs, dzs


def _as_profile(value, name: str, count: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim < 1 or array.shape[0] != count:
        raise ValueError(
            f"{name} must have {count} leading source levels, got shape "
            f"{array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _bracket(target: np.float32, samples: np.ndarray) -> int:
    """Return WRF's chosen source interval for one target depth, or -1.

    :1958-1968 walks the source intervals in order and takes the FIRST that
    contains the target, with both comparisons inclusive -- so a target that
    coincides with an interior sample depth is taken from the interval ABOVE
    it, not the one below.  A target outside every interval leaves ``tslb`` /
    ``smois`` at whatever was in the INTENT(OUT) array, which is why the
    caller must be told rather than given a clamped value.
    """

    for index in range(samples.size - 1):
        if samples[index] <= target <= samples[index + 1]:
            return index
    return -1


def _interpolate(
    samples: np.ndarray, depths: np.ndarray, targets: np.ndarray, label: str,
) -> np.ndarray:
    """WRF's linear interpolation, in WRF's float32 expression order."""

    out = np.empty((targets.size,) + samples.shape[1:], dtype=np.float32)
    for want, target in enumerate(targets):
        have = _bracket(target, depths)
        if have < 0:
            raise ValueError(
                f"{label}: RUC level {want + 1} at {float(target)} m lies "
                f"outside the source depths "
                f"{[float(d) for d in depths]}; init_soil_3_real:1958-1968 "
                "leaves such a level UNSET in an INTENT(OUT) array rather "
                "than extrapolating, so there is no value to reproduce"
            )
        upper = depths[have]
        lower = depths[have + 1]
        out[want] = (
            samples[have] * (lower - target)
            + samples[have + 1] * (target - upper)
        ) / (lower - upper)
    return out


def remap_soil_to_ruc_levels(
    *,
    source_temperature,
    source_moisture,
    source_levels_cm,
    source_geometry: str,
    skin_temperature,
    deep_temperature,
    landmask,
    sea_surface_temperature=None,
    num_soil_layers: int = 9,
    moisture_adjustment: bool = False,
    source_depth_scale: int = 100,
) -> RucSoilColumns:
    """Build RUC ``TSLB``/``SMOIS`` the way ``real.exe`` does.

    ``source_geometry`` is ``"layers"`` for a slab source that reaches WRF as
    metgrid's ``FLAG_SOIL_LAYERS=1`` (ERA5, GFS), and ``"levels"`` for a node
    source already on RUC's own depths (``FLAG_SOIL_LEVELS=1``; a RUC-native
    or HRRR grid).  The two are not interchangeable and WRF's two arms differ
    in more than bookkeeping:

    ``"layers"``
        ``source_levels_cm`` are the layer MIDPOINTS in whole centimetres,
        which is how WRF's reader forms them -- ``char2int2``
        (``share/module_optional_input.F:1949-1954``) and the
        ``flag_soil_layers`` block (:1339-1352) both compute
        ``(top + bottom) / 2`` in INTEGER arithmetic, so a 0-7 cm ERA5 layer
        is sampled at 3 cm, not 3.5 and not 7.  The profile is then anchored
        at 0 m by the skin temperature and at 3 m by the deep temperature
        (:1899-1907), and moisture is anchored at 0 m by a linear
        extrapolation off the top two layers and at 3 m by repeating the
        deepest (:1938-1948).
    ``"levels"``
        The nodes are the samples, with no anchors at all.  A target level
        outside them is an error, not an extrapolation.

    ``source_depth_scale`` is the integer unit ``source_levels_cm`` is
    carried in, declared by the source's own remap policy
    (:func:`woof.ingest.soil_contract.ruc_soil_remap_policy`).  The
    default 100 is WRF's own: ``char2int2``
    (``share/module_optional_input.F:1949-1954``) forms soil level depths
    by integer division into CENTIMETRES and :1994 divides the INTEGER by
    100, so every metgrid-shaped source runs at 100 and is bit-identical
    to the oracle.  A producer whose published ladder is finer than a
    centimetre (ICON's shallowest interior node is 0.005 m) declares a
    finer scale rather than being refused: the limit at 100 is metgrid's
    file format, not the soil column, and rounding 0.5 cm onto the 0.0 m
    node would hand :1958-1968 two samples at one depth.  The arithmetic
    is unchanged -- ``REAL(level)/scale`` in the same float32 expression
    order.

    ``moisture_adjustment`` is WRF's ``flag_sm_adj`` (namelist default 0,
    ``Registry.EM_COMMON:2537``).  It applies only to a layer source
    (:2067) and rescales the top ``num_soil_layers - 1`` levels so their
    ``dzs``-weighted bucket matches the source's own four-layer bucket
    divided by 0.9, then floors each at 0.02, then -- if moisture rises
    with depth across the top three levels -- multiplies levels 1-5 by
    :data:`_MOISTURE_DAYTIME_FACTORS`.  The deepest level is never adjusted.
    """

    if source_geometry not in {_LAYER_SOURCE, _LEVEL_SOURCE}:
        raise ValueError(
            f"source_geometry={source_geometry!r} must be "
            f"{_LAYER_SOURCE!r} or {_LEVEL_SOURCE!r}"
        )

    zs, dzs = ruc_soil_depths(num_soil_layers)
    count = zs.size

    levels_cm = np.asarray(source_levels_cm)
    if levels_cm.ndim != 1 or levels_cm.size < 2:
        raise ValueError("source_levels_cm must be at least two depths")
    if levels_cm.dtype.kind not in "iu":
        raise ValueError(
            "source_levels_cm must be INTEGER depths in units of "
            "source_depth_scale per metre: WRF's readers form them with "
            "integer division and init_soil_3_real:1994 divides the INTEGER "
            "by the scale."
        )
    scale_value = int(source_depth_scale)
    if scale_value <= 0:
        raise ValueError(
            f"source_depth_scale={source_depth_scale} must be a positive "
            "integer number of units per metre")
    if np.any(np.diff(levels_cm) <= 0):
        raise ValueError(
            "source_levels_cm must be strictly increasing.  WRF sorts them "
            "(init_soil_3_real:1881-1899) but the accompanying swap moves "
            "st_input(1..n) while a layer source occupies st_input(2..n+1), "
            "so an unsorted source is not reordered -- it is corrupted.  See "
            "the oracle's era5_layers_reversed_adj rows, whose soil "
            "temperatures are of order -1e29 K."
        )
    if np.any(levels_cm < 0):
        raise ValueError("source_levels_cm must be non-negative depths")

    nlev = int(levels_cm.size)
    temperature = _as_profile(source_temperature, "source_temperature", nlev)
    moisture = _as_profile(source_moisture, "source_moisture", nlev)
    field_shape = temperature.shape[1:]
    if moisture.shape[1:] != field_shape:
        raise ValueError("source temperature and moisture shapes differ")

    tsk = np.array(skin_temperature, dtype=np.float32, copy=True)
    tmn = np.asarray(deep_temperature, dtype=np.float32)
    mask = np.asarray(landmask, dtype=np.float32)
    for name, value in (
        ("skin_temperature", tsk), ("deep_temperature", tmn),
        ("landmask", mask),
    ):
        if value.shape != field_shape:
            raise ValueError(
                f"{name} shape {value.shape} differs from the source field "
                f"shape {field_shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")

    hundred = np.float32(scale_value)
    if source_geometry == _LEVEL_SOURCE:
        depths = (levels_cm.astype(np.float32) / hundred).astype(np.float32)
        temperature_samples = temperature
        moisture_samples = moisture
        moisture_depths = depths
    else:
        depths = np.empty(nlev + 2, dtype=np.float32)
        depths[0] = np.float32(0.0)
        depths[1:nlev + 1] = levels_cm.astype(np.float32) / hundred
        # :1990 and :2033 spell the deep anchor as 300./100., not as 3, so
        # it is formed at the declared scale rather than written as 3.0.
        depths[nlev + 1] = np.float32(3.0 * scale_value) / hundred
        temperature_samples = np.concatenate(
            (tsk[None, ...], temperature, tmn[None, ...]), axis=0)
        # :1938-1943.  Note it divides by the difference of the TEMPERATURE
        # level depths; WRF reads st_levels_input here even though the array
        # being extrapolated is moisture.  They coincide for every metgrid
        # source, and this function enforces that by taking one level list.
        span = np.float32(int(levels_cm[1]) - int(levels_cm[0]))
        shallow = np.float32(int(levels_cm[0]))
        surface = (moisture[0] - moisture[1]) / span * shallow + moisture[0]
        moisture_samples = np.concatenate(
            (surface[None, ...], moisture, moisture[-1:]), axis=0)
        moisture_depths = depths

    soil_t = _interpolate(
        temperature_samples, depths, zs, "RUC soil temperature")
    soil_m = _interpolate(
        moisture_samples, moisture_depths, zs, "RUC soil moisture")

    if moisture_adjustment:
        if source_geometry != _LAYER_SOURCE:
            raise ValueError(
                "moisture_adjustment applies only to a layer source: "
                "init_soil_3_real:2067 guards the whole block with "
                "flag_soil_levels /= 1, so asking for it on a node source "
                "would silently do nothing"
            )
        if nlev != len(_NOAH_BUCKET_WEIGHTS):
            raise ValueError(
                f"moisture_adjustment needs exactly "
                f"{len(_NOAH_BUCKET_WEIGHTS)} source layers: "
                "init_soil_3_real:1930-1934 indexes sm_input(2..5) with "
                "hardcoded weights and has no general form"
            )
        soil_m = _apply_moisture_adjustment(soil_m, moisture, mask, dzs)

    land = mask >= np.float32(0.5)
    water = ~land
    if np.any(water):
        # :2131-2157.  With an SST field the whole water column, including
        # TSK, is reset to SST; without one it is reset to TSK.
        if sea_surface_temperature is None:
            fill = tsk
        else:
            sst = np.asarray(sea_surface_temperature, dtype=np.float32)
            if sst.shape != field_shape:
                raise ValueError(
                    "sea_surface_temperature shape differs from the source "
                    "field shape")
            if not np.isfinite(sst).all():
                raise ValueError(
                    "sea_surface_temperature contains non-finite values")
            # :2138 assigns tsk(i,j) = sst(i,j) on every water column with no
            # validity test at all -- and it is the ONE line by which the RUC
            # arm differs from Noah's, whose init_soil_2_real:1732-1754 writes
            # tslb/smois/sh2o over water and never touches tsk.  A gap in the
            # SST analysis therefore kills a RUC run and not a Noah one:
            # stock real.exe aborts a few hundred lines later at
            # dyn_em/module_initialize_real.F:3283 with "grid%tsk
            # unreasonable".  Measured on one production coarse-domain
            # metgrid frame of this project's own case: 1,494 of 14,397
            # water cells (10.4%) carry SST = 0, and real.exe at
            # sf_surface_physics=3 dies on the first of them while the same
            # input at sf_surface_physics=2 writes its wrfinput without
            # complaint.
            # Refused at the seam, where the caller can substitute a skin
            # temperature the way preprocess_noah_soil already does, rather
            # than propagated as a 0 K ocean.
            unusable = water & (
                (sst < np.float32(170.0)) | (sst > np.float32(400.0)))
            if np.any(unusable):
                raise ValueError(
                    f"{int(unusable.sum())} of {int(water.sum())} water "
                    "columns have a nonphysical sea_surface_temperature.  "
                    "init_soil_3_real:2131-2144 writes it into TSK unchecked "
                    "-- unlike Noah's init_soil_2_real, which never writes "
                    "TSK -- and stock real.exe then aborts at "
                    "module_initialize_real.F:3283.  Repair the SST field "
                    "against a valid skin temperature before this seam"
                )
            fill = sst
            tsk = np.where(water, sst, tsk).astype(np.float32)
        soil_t = np.where(water[None, ...], fill[None, ...], soil_t)
        soil_m = np.where(
            water[None, ...], np.float32(1.0), soil_m).astype(np.float32)

    return RucSoilColumns(
        soil_temperature=np.ascontiguousarray(soil_t, dtype=np.float32),
        soil_moisture=np.ascontiguousarray(soil_m, dtype=np.float32),
        skin_temperature=np.ascontiguousarray(tsk, dtype=np.float32),
        level_depths=zs,
        level_thicknesses=dzs,
    )


def _apply_moisture_adjustment(
    soil_m: np.ndarray,
    source_moisture: np.ndarray,
    landmask: np.ndarray,
    dzs: np.ndarray,
) -> np.ndarray:
    """``init_soil_3_real``:2062-2129, the Noah-to-RUC bucket rescale."""

    count = soil_m.shape[0]
    # :1930-1934, formed before the surface/deep anchors are written and so
    # from the source layers alone.
    bucket_source = source_moisture[0] * np.float32(_NOAH_BUCKET_WEIGHTS[0])
    for index in range(1, len(_NOAH_BUCKET_WEIGHTS)):
        bucket_source = bucket_source + (
            source_moisture[index] * np.float32(_NOAH_BUCKET_WEIGHTS[index])
        )

    # :2072-2077.  The RUC bucket omits the deepest level, and it is
    # accumulated in level order because float32 addition is not associative.
    bucket_target = np.zeros(soil_m.shape[1:], dtype=np.float32)
    for index in range(count - 1):
        bucket_target = bucket_target + soil_m[index] * dzs[index]

    land = landmask > np.float32(0.5)
    if np.any(land & (bucket_target == np.float32(0.0))):
        raise ValueError(
            "a land column has zero RUC soil-moisture bucket, so "
            "init_soil_3_real:2079 divides by zero.  WRF then evaluates "
            "MAX(0.02, NaN), whose result is gfortran's operand order "
            "rather than anything Fortran defines -- the oracle's "
            "era5_layers_adj_nosst column 10 records 0.02 on gfortran 13.3.0. "
            "Refused rather than reproduced"
        )

    adjusted = soil_m.copy()
    scale = bucket_source / (np.float32(0.9) * bucket_target)
    for index in range(count - 1):
        candidate = (adjusted[index] * bucket_source) / (
            np.float32(0.9) * bucket_target)
        adjusted[index] = np.where(
            land, np.maximum(np.float32(0.02), candidate), adjusted[index],
        ).astype(np.float32)
    del scale

    # :2100-2107.  Strict >, so an exactly flat profile does NOT arm it.
    daytime = land & (adjusted[1] > adjusted[0]) & (adjusted[2] > adjusted[1])
    for index in range(count - 1):
        if index < len(_MOISTURE_DAYTIME_FACTORS):
            factor = np.float32(_MOISTURE_DAYTIME_FACTORS[index])
        else:
            factor = np.float32(1.0)
        adjusted[index] = np.where(
            daytime, factor * adjusted[index], adjusted[index],
        ).astype(np.float32)
    return adjusted


# ---------------------------------------------------------------------------
# The runtime seam.
#
# Everything above is ``share/module_soil_pre.F`` and answers "what does
# real.exe compute".  Everything below answers the separate question this
# module existed for a week without answering: **how does a RUC run actually
# receive it.**  Until this section landed, ``woof/ingest/ruc_soil.py`` had
# zero importers anywhere under ``gpuwm/`` -- the remap was finished, oracle
# matched, and unreachable, so every initializer called
# ``preprocess_noah_soil`` and a ``sf_surface_physics=3`` request received
# Noah's four layer MIDPOINTS.
# ---------------------------------------------------------------------------

#: WRF's ``RUCLSMSCHEME``.  Spelled here rather than imported so this module
#: keeps depending on nothing but numpy and :mod:`woof.ingest.soil`.
RUCLSMSCHEME = 3

#: Every land-surface selector whose soil geometry is Noah's four layer
#: midpoints.  ``2`` is Noah and ``4`` is Noah-MP; both take
#: :func:`woof.ingest.soil.preprocess_noah_soil` unchanged.
_NOAH_GEOMETRY_SCHEMES = (2, 4)


@dataclass(frozen=True)
class RucSoilState:
    """A RUC initial soil state in ``NoahSoilState``'s slot.

    The first ten fields are spelled exactly as :class:`
    woof.ingest.soil.NoahSoilState`'s, because every consumer downstream of
    the initializers -- ``initialize_physics``,
    ``woof.native_wrf_contract.canonical_noah_surface``,
    ``initialize_landuse`` -- reads them by name.  What differs is the leading
    axis: ``soil_temperature``/``soil_moisture``/``liquid_moisture`` are
    ``(num_soil_layers, ny, nx)`` on RUC's LEVEL depths, not ``(4, ny, nx)``
    on Noah's layer midpoints.  ``level_depths``/``level_thicknesses`` are
    the ZS/DZS that go with them and have no ``NoahSoilState`` counterpart.

    ``liquid_moisture`` is a copy of ``soil_moisture``.  That is not a
    partition and is not meant to be one: WRF derives a RUC run's SH2O in
    ``ruclsminit`` (``module_sf_ruclsm.F``), not in ``real.exe``, and
    :func:`woof.core.ruc_runtime.ruc_cold_start` runs ``ruclsminit`` at
    driver construction and overwrites every value here.  Noah's
    ``sh2o_init`` freezing-curve partition is deliberately NOT applied --
    it is Noah's table lookup on Noah's geometry, and running it would put a
    Noah-derived number into a RUC field that is about to be recomputed.
    """

    soil_temperature: np.ndarray       # (num_soil_layers, ny, nx), WRF TSLB
    soil_moisture: np.ndarray          # (num_soil_layers, ny, nx), WRF SMOIS
    liquid_moisture: np.ndarray        # (num_soil_layers, ny, nx), WRF SH2O
    deep_soil_temperature: np.ndarray  # (ny, nx), WRF TMN
    tsk: np.ndarray                    # (ny, nx)
    landmask: np.ndarray               # WPS 1 land / 0 water
    xland: np.ndarray                  # WRF 1 land / 2 water
    xice: np.ndarray
    snow_water: np.ndarray             # kg m-2
    snow_depth: np.ndarray             # m
    level_depths: np.ndarray           # (num_soil_layers,), WRF ZS
    level_thicknesses: np.ndarray      # (num_soil_layers,), WRF DZS
    #: Receipt from :mod:`woof.ingest.soil_downscale`, spelled exactly as
    #: :class:`woof.ingest.soil.NoahSoilState`'s so the run receipt binds
    #: it the same way for either land surface.  RUC's copy is measured on
    #: RUC's OWN ``STAS-RUC`` air-dry and saturation constants, because the
    #: column it describes is the one ``remap_soil_to_ruc_levels`` builds
    #: and ``LSMRUC`` integrates -- not Noah's discarded four layers.
    soil_texture_downscale: Mapping[str, object] = field(default_factory=dict)
    #: real.exe's TSLB reasonableness rebuild on RUC's own levels, spelled
    #: as :class:`woof.ingest.soil.NoahSoilState`'s; EMPTY when no land
    #: column needed it.
    soil_temperature_repair: Mapping[str, object] = field(default_factory=dict)
    #: real.exe's ``SMOIS = MAX(SMOIS, 0.005)`` for a layer source
    #: (:func:`account_for_zero_ruc_soil_moisture`); EMPTY when nothing was
    #: below it.
    soil_moisture_floor: Mapping[str, object] = field(default_factory=dict)
    #: The sea-ice cells of the Noah surface call, spelled as
    #: :class:`woof.ingest.soil.NoahSoilState`'s ``sea_ice_cells``.
    sea_ice_cells: np.ndarray | None = None
    #: :mod:`woof.ingest.land_state`'s receipt (soil-moisture index remap
    #: and same-model donor), spelled as NoahSoilState's.
    land_state: Mapping[str, object] = field(default_factory=dict)


def _midpoint_cm(name: str, prefix: str) -> int:
    """The INTEGER centimetre midpoint WRF's reader forms from a layer name.

    ``share/module_optional_input.F:1949-1954`` (``char2int2``) and the
    ``flag_soil_layers`` block at ``:1339-1352`` both compute
    ``(top + bottom) / 2`` in INTEGER arithmetic, so ``ST000007`` is sampled
    at 3 cm and not 3.5.  Parsed off the field name rather than tabulated,
    because the name IS the depth encoding and a hand table would be a second
    place for it to be wrong.
    """

    digits = name[len(prefix):]
    if len(digits) != 6 or not digits.isdigit():
        raise ValueError(f"{name!r} is not a <prefix><top3><bottom3> layer name")
    return (int(digits[:3]) + int(digits[3:])) // 2


def _source_soil_profiles(fields, soil_layer_contract, *, land=None):
    """Return ``(temperature, moisture, levels, geometry, depth_scale)``.

    The four source modes are exactly the four
    :func:`woof.ingest.soil.preprocess_noah_soil` dispatches on, and the
    marker names are imported from it rather than restated, so a source whose
    field names move cannot leave this function reading the old ones.
    """

    from woof.ingest.soil import (_GFS_MOIST_NAMES, _GFS_TEMP_NAMES,
                                   _MOIST_NAMES, _TEMP_NAMES, _host)
    from woof.ingest.soil_contract import (MAPPED_SOIL_MOISTURE,
                                            MAPPED_SOIL_TEMPERATURE)

    if soil_layer_contract is not None or MAPPED_SOIL_TEMPERATURE in fields:
        # The declarative mapped-source arm.  Its RUC target is declared
        # explicitly in woof/ingest/soil_contract.py -- the way that
        # module's Noah comparison demands -- and the remap policy is
        # LOOKED UP in that module's RUC_REMAP_POLICIES table, not decided
        # here: a node contract selects init_soil_3_real's flag_soil_levels
        # arm and a layer contract its flag_soil_layers arm.  Both arms are
        # the ones below this branch already run for the native SOILT/SOILW
        # and ST/GFS_ST dispatches, so mapped-vs-native equality on
        # identical inputs stays structural and adding a future model is a
        # composition document, not a branch here.
        from woof.ingest.soil_contract import ruc_soil_remap_policy

        if soil_layer_contract is None:
            raise ValueError(
                "mapped soil arrays require an explicit soil_layer_contract")
        policy = ruc_soil_remap_policy(soil_layer_contract)
        contract = policy["contract"]
        if (MAPPED_SOIL_TEMPERATURE not in fields
                or MAPPED_SOIL_MOISTURE not in fields):
            raise KeyError(
                "declarative mapped soil input requires temperature and "
                "moisture arrays together")
        temperature = _host(fields[MAPPED_SOIL_TEMPERATURE])
        moisture = _host(fields[MAPPED_SOIL_MOISTURE])
        sample_count = len(policy["sample_depths"])
        if (temperature.shape != moisture.shape
                or temperature.shape[0] != sample_count):
            raise ValueError(
                "declarative mapped soil arrays must both carry "
                f"{sample_count} leading {policy['declaration']} samples; got "
                f"temperature {temperature.shape} and moisture "
                f"{moisture.shape}")
        # The mapped route's OWN moisture admission, applied to the source
        # samples RUC integrates exactly as woof/ingest/soil.py's mapped
        # branch applies it to the Noah copy (which this arm discards):
        # the bound-kissing decode clamp, then interpolation overshoot put
        # on 0..1 and a land value beyond the operator's reach refused
        # (clamp_soil_moisture_overshoot), judged on the same land the
        # Noah call judges, since water columns become 1.0 either way.
        # A non-finite value is left as decoded, so the fail-closed check
        # downstream (woof/core/ruc.py:ruc_initialize_cold_start's 0..1
        # check, the refusal that found this seam) still sees it.
        from woof.ingest.quantization import clamp_bound_kissing
        from woof.ingest.soil import clamp_soil_moisture_overshoot

        moisture, _ = clamp_bound_kissing(moisture, minimum=0.0, maximum=1.0)
        moisture, _ = clamp_soil_moisture_overshoot(
            moisture, land=land,
            subject=f"mapped soil moisture (RUC {policy['policy']})")
        del contract
        levels_cm = np.array(policy["sample_depths"], dtype=np.int64)
        return (temperature, moisture, levels_cm,
                str(policy["geometry"]),
                int(policy["depth_scale_per_m"]))

    if "SOILT" in fields and "SOILW" in fields:
        # HRRR's native depth nodes ARE RUC's nine levels, bit for bit --
        # woof.ingest.soil.HRRR_SOIL_NODE_DEPTHS_M is
        # RUC_LEVEL_DEPTHS_M[9] in metres.  So this arm is WRF's
        # ``flag_soil_levels`` case and the remap is the identity.
        levels_cm = np.array(
            [int(round(depth * 100.0)) for depth in RUC_LEVEL_DEPTHS_M[9]],
            dtype=np.int64)
        return (_host(fields["SOILT"]), _host(fields["SOILW"]),
                levels_cm, _LEVEL_SOURCE, 100)

    if all(name in fields for name in _GFS_TEMP_NAMES):
        names, moist_names, prefix = (
            _GFS_TEMP_NAMES, _GFS_MOIST_NAMES, "GFS_ST")
    elif all(name in fields for name in _TEMP_NAMES):
        names, moist_names, prefix = (_TEMP_NAMES, _MOIST_NAMES, "ST")
    else:
        raise KeyError(
            "no recognised soil source: expected HRRR SOILT/SOILW nodes, the "
            f"four GFS layers {_GFS_TEMP_NAMES}, or the four ERA5 layers "
            f"{_TEMP_NAMES}")
    levels_cm = np.array(
        [_midpoint_cm(name, prefix) for name in names], dtype=np.int64)
    return (np.stack([_host(fields[name]) for name in names]),
            np.stack([_host(fields[name]) for name in moist_names]),
            levels_cm, _LAYER_SOURCE, 100)


def preprocess_ruc_soil(
    fields,
    *,
    soil_type,
    deep_soil_temperature=None,
    lake_mask=None,
    lake_skin_temperature=None,
    soil_layer_contract=None,
    num_soil_layers: int = 9,
    landmask=None,
    terrain=None,
    source_orography=None,
    water_temperature=None,
    water_temperature_policy=None,
    soil_mesh=None,
    route=None,
    fractional_seaice: bool = False,
    seaice_threshold: float = 100.0,
) -> RucSoilState:
    """Build a RUC initial soil state, in ``preprocess_noah_soil``'s signature.

    The surface bookkeeping -- the land/water/sea-ice partition, the lake
    override, the SST and TSK repair, the TMN repair, the 271.4 K sea-ice
    deep temperature and the SNOW/SNOWH reconciliation -- is
    ``dyn_em/module_initialize_real.F`` and ``share/module_soil_pre.F``'s
    surface half, which WRF runs identically for every ``sf_surface_physics``.
    So this function CALLS :func:`woof.ingest.soil.preprocess_noah_soil` for
    it rather than restating it.  That is deliberate and it is what makes
    "Noah initialization is bitwise unchanged" a structural property instead
    of a measurement: the Noah routine is not edited, not copied and not
    parameterised, only called.  Its four-layer TSLB/SMOIS/SH2O are then
    DISCARDED, and the soil column is rebuilt from the source profiles by
    :func:`remap_soil_to_ruc_levels`, which is ``init_soil_3_real``.

    Three divergences from stock ``real.exe``, each with a reason:

    ``flag_sm_adj``
        Not applied.  ``woof.config.RUC_OPTION_IDENTITY_EVIDENCE`` admits
        only ``flag_sm_adj = 0``, so ``moisture_adjustment`` is False here and
        the two must not drift apart.

    the open-water fill
        ``init_soil_3_real:2131-2144`` writes SST into TSK on every water
        column with no validity test, and stock ``real.exe`` then aborts a few
        hundred lines later at ``module_initialize_real.F:3283`` on any gap in
        the SST analysis -- measured at 10.4% of the water cells of one
        production coarse-domain frame.  ``preprocess_noah_soil`` has
        already resolved TSK to the valid SST wherever one exists and to the
        skin temperature where it does not, so this seam takes WRF's
        **no-SST** arm against that already-repaired TSK.  The result equals
        WRF's wherever WRF would not have aborted.

    sea ice
        Left to the runtime.  ``LSMRUC:862-895`` resets SOILMOIS, SMFR3D,
        SH2O, KEEPFR3DFLAG and ``min(271.4, TSO)`` on every sea-ice column on
        **every** step, not once at initialization, so writing an ice column
        here would be overwritten before it was read.  Noah's four-layer
        equispaced ice profile is therefore not carried over; TMN is, because
        ``preprocess_noah_soil`` sets it to 271.4 K on ice and RUC reads TMN
        as the bottom boundary at ``LSMRUC:1057``.

    ``landmask``, ``terrain`` and ``source_orography`` are the ERA5/mapped
    initializers' soil seam, with :func:`woof.ingest.soil.
    preprocess_noah_soil`'s exact semantics: the static-build landmask is
    the terrestrial/water decision surface (WRF's ``process_soil_real``
    drives every branch with ``grid%landmask``, never the met-source
    LANDSEA), and terrain/source_orography (all-or-none) arm WRF's
    ``adjust_soil_temp_new`` elevation lapse.  That lapse is applied by
    ``process_soil_real`` to TSK and to EVERY ``st_input`` level BEFORE
    either ``init_soil_*_real`` arm runs, so it is scheme-independent:
    the Noah call below returns the adjusted TSK anchor, and the same
    float64 increment -- computed by the same helper on the same decision
    surface, lakes excluded -- is added to the RUC source temperature
    samples (layer or level form alike) before the float32 remap.
    Moisture and the 3 m deep anchor are not adjusted, exactly as on the
    Noah path.

    ``num_soil_layers`` reaches :func:`ruc_soil_depths`, which refuses
    anything but 6 or 9.  woof's CUDA leaves index a nine-element
    ``__constant__`` table, so a six-level request is refused further in.

    ``soil_mesh`` is the sub-source-cell reconstitution
    (:mod:`woof.ingest.soil_downscale`).  It is applied HERE, to the source
    profiles, and NOT forwarded to the Noah call below -- deliberately, and
    for the two reasons this whole seam exists.  Noah's four layers are
    discarded a few lines later, so downscaling them would be work thrown
    away and an advisory printed twice; and RUC's air-dry and saturation
    constants are ``SOILPARM.TBL``'s ``STAS-RUC`` block, not ``STAS``, so the
    ratio has to be formed against the table the scheme itself integrates.
    The Noah call therefore stays bitwise what it was.
    """
    if not isinstance(fractional_seaice, bool):
        raise TypeError("fractional_seaice must be boolean")


    from woof.ingest.soil import (_host, _soil_temperature_elevation_delta,
                                   preprocess_noah_soil)

    # The land the Noah call below decides on (the static landmask over the
    # met-source LANDSEA, lake-override cells as water), which the mapped
    # arm's moisture admission judges; the Noah call names a source
    # without either.
    decision = landmask if landmask is not None else fields.get("LANDSEA")
    terrestrial = None if decision is None else _host(decision) >= 0.5
    if terrestrial is not None and lake_mask is not None:
        terrestrial = terrestrial & ~_host(lake_mask).astype(bool)
    # Selected BEFORE the Noah call, so a source RUC has no arm for is
    # refused by name rather than by whatever preprocess_noah_soil says
    # about it first.
    (temperature, moisture, levels_cm, geometry,
     depth_scale) = _source_soil_profiles(
         fields, soil_layer_contract, land=terrestrial)

    surface = preprocess_noah_soil(
        fields,
        **({"fractional_seaice": True} if fractional_seaice else {}),
        **_threshold_keyword(seaice_threshold),
            soil_type=soil_type,
        deep_soil_temperature=deep_soil_temperature,
        lake_mask=lake_mask, lake_skin_temperature=lake_skin_temperature,
        soil_layer_contract=soil_layer_contract,
        landmask=landmask, terrain=terrain,
        source_orography=source_orography,
        water_temperature=water_temperature,
        water_temperature_policy=water_temperature_policy,
        route=route)

    if terrain is not None:
        # The Noah call above validated the all-or-none pairing and
        # adjusted the TSK anchor; mirror the identical increment onto the
        # RUC source samples.  The decision surface is rebuilt exactly as
        # preprocess_noah_soil builds it (static landmask over met-source
        # LANDSEA; lake-override cells are water), because the increment
        # must be zero on every column WRF would not adjust.
        decision = (_host(landmask) if landmask is not None
                    else _host(fields["LANDSEA"])) >= 0.5
        if lake_mask is not None:
            decision = decision.copy()
            decision[_host(lake_mask).astype(bool)] = False
        temperature = temperature + _soil_temperature_elevation_delta(
            terrain, source_orography, decision)

    # The sub-source-cell reconstitution, on the SOURCE profiles and before
    # the vertical remap.  It is a purely horizontal, per-slab affine map,
    # so it commutes with the vertical remap on either geometry arm: doing
    # it here means the levels RUC integrates carry the target grid's own
    # texture without this seam having to know RUC's level table.
    downscale_receipt: Mapping[str, object] = {}
    if soil_mesh is not None:
        from woof.core.noah import load_tables, pack_params
        from woof.ingest.soil_downscale import downscale_soil_moisture

        terrestrial = (_host(landmask) if landmask is not None
                       else _host(fields["LANDSEA"])) >= 0.5
        if lake_mask is not None:
            terrestrial = terrestrial.copy()
            terrestrial[_host(lake_mask).astype(bool)] = False
        moisture, downscale_receipt = downscale_soil_moisture(
            moisture, soil_type=soil_type, terrestrial=terrestrial,
            params=pack_params(load_tables(mminsl="STAS-RUC")),
            plan=soil_mesh,
            # RUC's source profiles are not on Noah's four thicknesses, and
            # the layer/level geometry differs per source, so the soil-water
            # budget is left unweighted rather than weighted by the wrong
            # column.  Every other number on the receipt is exact.
            layer_thickness_m=())
        downscale_receipt = dict(downscale_receipt)
        downscale_receipt["soil_table"] = "STAS-RUC"

    # real.exe's TSLB reasonableness rebuild on RUC's column: the land
    # columns the Noah call above rebuilt (the same samples, lapse, land,
    # skin and snow) are held at TSK through the remap and rebuilt
    # TSK-to-TMN on RUC's own levels after it (woof.ingest.soil.
    # unreasonable_land_soil_columns and snow_soil_below_skin_columns).
    # The Noah call announced them.
    from woof.ingest.soil import (snow_soil_below_skin_columns,
                                   soil_temperature_repair_receipt,
                                   tsk_tmn_soil_profile,
                                   unreasonable_land_soil_columns)

    repair_land = (_host(landmask) if landmask is not None
                   else _host(fields["LANDSEA"])) >= 0.5
    if lake_mask is not None:
        repair_land = repair_land & ~_host(lake_mask).astype(bool)
    band_columns = unreasonable_land_soil_columns(temperature, repair_land)
    snow_columns = snow_soil_below_skin_columns(
        temperature, repair_land, skin=_host(surface.tsk),
        snow_water=_host(surface.snow_water))
    rebuilt_columns = band_columns | snow_columns
    temperature_repair = soil_temperature_repair_receipt(
        temperature, band_columns, repair_land, snow_columns=snow_columns,
        skin=_host(surface.tsk))
    if temperature_repair:
        temperature = np.array(temperature, copy=True)
        temperature[:, rebuilt_columns] = _host(surface.tsk)[rebuilt_columns]

    columns = remap_soil_to_ruc_levels(
        source_temperature=temperature,
        source_moisture=moisture,
        source_levels_cm=levels_cm,
        source_geometry=geometry,
        skin_temperature=surface.tsk,
        deep_temperature=surface.deep_soil_temperature,
        landmask=surface.landmask,
        sea_surface_temperature=None,
        num_soil_layers=num_soil_layers,
        moisture_adjustment=False,
        source_depth_scale=depth_scale,
    )

    soil_temperature = columns.soil_temperature
    if temperature_repair:
        soil_temperature = np.array(soil_temperature, copy=True)
        soil_temperature[:, rebuilt_columns] = tsk_tmn_soil_profile(
            columns.level_depths, surface.tsk,
            surface.deep_soil_temperature)[:, rebuilt_columns].astype(
                soil_temperature.dtype)

    soil_moisture, moisture_floor = account_for_zero_ruc_soil_moisture(
        columns.soil_moisture, geometry)
    liquid_moisture = np.array(soil_moisture, copy=True)
    tsk = surface.tsk
    if fractional_seaice:
        sea_water = None
        if water_temperature is not None:
            sea_water = water_temperature
        elif fields.get("SST") is not None:
            sea_water = fields["SST"]
        tsk, soil_temperature, soil_moisture, liquid_moisture = (
            fractional_sea_ice_post(
                xice=surface.xice, tsk=tsk, sst=sea_water,
                ice=surface.sea_ice_cells,
                deep_soil_temperature=surface.deep_soil_temperature,
                soil_temperature=soil_temperature,
                soil_moisture=soil_moisture,
                liquid_moisture=liquid_moisture))

    return RucSoilState(
        soil_temperature=soil_temperature,
        soil_moisture=soil_moisture,
        liquid_moisture=liquid_moisture,
        deep_soil_temperature=surface.deep_soil_temperature,
        tsk=tsk,
        landmask=surface.landmask,
        xland=surface.xland,
        xice=surface.xice,
        snow_water=surface.snow_water,
        snow_depth=surface.snow_depth,
        level_depths=columns.level_depths,
        level_thicknesses=columns.level_thicknesses,
        soil_texture_downscale=downscale_receipt,
        soil_temperature_repair=temperature_repair,
        soil_moisture_floor=moisture_floor,
        sea_ice_cells=surface.sea_ice_cells,
    )


def _threshold_keyword(seaice_threshold) -> dict:
    """``seaice_threshold`` for a soil call, empty at WRF's 100 K default
    so a default call is the call it was before the field existed."""
    if float(seaice_threshold) == 100.0:
        return {}
    return {"seaice_threshold": float(seaice_threshold)}


#: real.exe's floor for a RUC column built from a LAYER source
#: (``dyn_em/module_initialize_real.F`` v4.6.1 :3435-3446,
#: ``account_for_zero_soil_moisture``, ``CASE ( RUCLSMSCHEME )``).
_WRF_RUC_LAYER_MOISTURE_FLOOR = np.float32(0.005)


def account_for_zero_ruc_soil_moisture(soil_moisture, geometry: str):
    """WRF's ``account_for_zero_soil_moisture`` for ``sf_surface_physics=3``.

    The linear surface anchor from a dry top layer can be negative even
    when the input layer samples are within 0..1. The remapped 0 and 1 cm
    nodes then fail cold-start admission without real.exe's layer floor.

    real.exe never hands RUC that column: after the soil remap,
    ``module_initialize_real.F`` :3435-3446 takes, for a
    ``flag_soil_layers`` source, ``SMOIS = MAX(SMOIS, 0.005)`` on every
    level of every column (the ``lqmi`` residual beside it is commented
    out), and for a ``flag_soil_levels`` source does nothing.  This is that
    branch, element for element, in float32.  Water columns already hold
    1.0 and are unaffected.  It is announced with its count and minimum
    rather than applied silently, and the receipt rides
    :class:`RucSoilState`.  Returns ``(soil_moisture, receipt)``; the input
    array is returned untouched (same object) when nothing is below.
    """

    if geometry != _LAYER_SOURCE:
        return soil_moisture, {}
    moisture = np.asarray(soil_moisture)
    below = np.isfinite(moisture) & (moisture < _WRF_RUC_LAYER_MOISTURE_FLOOR)
    if not below.any():
        return soil_moisture, {}
    per_level = {
        f"SMOIS_L{level + 1}": int(np.count_nonzero(below[level]))
        for level in range(moisture.shape[0]) if below[level].any()}
    receipt = {
        "policy": "wrf-ruc-layer-source-smois-max-0.005",
        "wrf_citation": (
            "dyn_em/module_initialize_real.F v4.6.1 :3435-3446 "
            "account_for_zero_soil_moisture CASE (RUCLSMSCHEME), "
            "flag_soil_layers arm"),
        "floored_values": int(np.count_nonzero(below)),
        "floored_columns": int(np.count_nonzero(below.any(axis=0))),
        "per_level": per_level,
        "min_pre_floor": float(np.min(moisture[below])),
    }
    from woof.static.rust_bridge import floor_ruc_layer_moisture
    floored = floor_ruc_layer_moisture(moisture)
    # "note:" on stderr: the prep door (woof.prep_output) puts only
    # warning/note lines on screen, so a bare stdout line reached the
    # details log alone and the floor was invisible at the door (2.8.8
    # acceptance D-02).
    print(
        f"note: RUC soil moisture: {receipt['floored_values']} value(s) in "
        f"{receipt['floored_columns']} column(s) below "
        f"{float(_WRF_RUC_LAYER_MOISTURE_FLOOR):.3g} on {sorted(per_level)} "
        f"(min {receipt['min_pre_floor']:.6g}, the layer source's "
        "extrapolated surface anchor) raised to 0.005, as real.exe does for "
        "RUC from a layer source (module_initialize_real.F:3441)",
        file=sys.stderr)
    return floored, receipt


def fractional_sea_ice_post(*, xice, tsk, sst, deep_soil_temperature,
                            soil_temperature, soil_moisture, liquid_moisture,
                            ice=None):
    """real.exe's RUC sea-ice column under ``fractional_seaice = 1``.

    ``share/module_soil_pre.F`` adjust_for_seaice_post, the RUCLSMSCHEME arm
    of the HRRR v4.1.21 fork (:337-391): every cell left holding ice (the
    Noah surface call above keeps the fraction at ``xice >= 0.02`` and
    zeroes the rest, :392-393) takes

    * the blended skin temperature (:357-363)
      ``TSK = XICE*min(271.4, TSK) + (1 - XICE)*SST``, with ``SST`` where the
      input has one (``flag_sst = 1``) and ``TSK`` itself where it has none;
    * the ice column (:375-385): ``TSLB(1) = TSK``, ``TSLB(n) = TMN`` and,
      for the levels between, the quarter-layer midpoints of a 3 m column,
      ``mid = (3/n)/4 + (k-2)*(3/n)``, ``TSLB(k) = ((3 - mid)*TSK +
      mid*TMN)/3``;
    * ``SMOIS = 1`` and ``SH2O = 0`` on every level (:387-390).

    float32 throughout, as real.exe's REAL arithmetic.  A cell whose SST is
    missing or outside 170..400 K blends with its own TSK, which is the
    ``flag_sst = 0`` arm for that cell.  Only called under
    ``fractional_seaice = 1``: at the default the binary arm is unchanged.

    ``ice`` is the Noah call's ``sea_ice_cells``: the ``xice >= 0.02``
    cells and the ``seaice_threshold`` cells (water colder than the
    threshold, :341-342), which take this column with their XICE as
    analysed -- possibly 0, so the blend is then the SST.  ``None`` keeps
    the fraction-only decision ``xice > 0``.
    """
    f32 = np.float32
    xice = np.asarray(xice, dtype=f32)
    ice = (xice > f32(0.0)) if ice is None else np.asarray(ice, dtype=bool)
    tsk = np.array(tsk, dtype=f32, copy=True)
    if sst is None:
        water = tsk
    else:
        candidate = np.asarray(sst, dtype=f32)
        valid = (np.isfinite(candidate) & (candidate >= f32(170.0))
                 & (candidate <= f32(400.0)))
        water = np.where(valid, candidate, tsk).astype(f32)
    blended = (xice * np.minimum(f32(271.4), tsk)
               + (f32(1.0) - xice) * water).astype(f32)
    tsk = np.where(ice, blended, tsk).astype(f32)
    soil_temperature = np.array(soil_temperature, copy=True)
    soil_moisture = np.array(soil_moisture, copy=True)
    liquid_moisture = np.array(liquid_moisture, copy=True)
    if not np.any(ice):
        return tsk, soil_temperature, soil_moisture, liquid_moisture
    n = soil_temperature.shape[0]
    tmn = np.asarray(deep_soil_temperature, dtype=f32)
    total = f32(3.0)
    layer = total / f32(n)
    column = np.empty((n,) + tsk.shape, dtype=f32)
    column[0] = tsk
    column[n - 1] = tmn
    for k in range(2, n):
        mid = layer / f32(4.0) + f32(k - 2) * layer
        column[k - 1] = ((total - mid) * tsk + mid * tmn) / total
    soil_temperature[:, ice] = column[:, ice].astype(soil_temperature.dtype)
    soil_moisture[:, ice] = 1.0
    liquid_moisture[:, ice] = 0.0
    return tsk, soil_temperature, soil_moisture, liquid_moisture


#: WRF's SLABSCHEME selector value.
SLABSCHEME = 1

#: process_soil_real's fix_bottom_level_for_temp, SLABSCHEME arm
#: (share/module_soil_pre.F:639-686): the slab's TMN is the source soil
#: temperature whose level lies closest to this depth, in cm -- the first of
#: equals, since the scan keeps a level only when it is strictly closer.
SLAB_TMN_SOURCE_DEPTH_CM = 30.0


def preprocess_slab_soil(
    fields,
    *,
    soil_type,
    deep_soil_temperature=None,
    lake_mask=None,
    lake_skin_temperature=None,
    soil_layer_contract=None,
    landmask=None,
    terrain=None,
    source_orography=None,
    water_temperature=None,
    water_temperature_policy=None,
    soil_mesh=None,
    route=None,
    seaice_threshold: float = 100.0,
):
    """The slab scheme's initial soil state, as real.exe builds it.

    The surface bookkeeping (land/water/sea-ice partition, lakes, the water
    temperature, snow, the elevation lapse) is :func:`preprocess_noah_soil`'s,
    exactly as for RUC.  What is the slab's own, from WRF v4.6.1:

    * **SEAICE_THRESHOLD** (``adjust_for_seaice_pre`` SLABSCHEME arm,
      share/module_soil_pre.F:103-140): an open-water column whose met skin
      temperature is below the threshold becomes sea ice (XICE = 1, then
      the ordinary ice conversion: XLAND = 1, the ice land-use row,
      TMN = 271.4 K).  WRF's default, 100 K, converts nothing.
    * **TMN** is the source soil temperature at the level closest to 30 cm
      (``fix_bottom_level_for_temp``), lapse-adjusted with the other soil
      levels (``adjust_soil_temp_new``), not the 3 m annual mean Noah reads:
      the slab's deepest layer is centred at 0.23 m.  A land value outside
      170..400 K takes TSK (module_initialize_real.F:3305-3321); water takes
      the water temperature; sea ice 271.4 K.
    * **TSLB** is ``init_soil_1_real``: linear in depth from TSK at zs(1) to
      TMN at zs(5) on land, the water temperature in every layer on water.
    * SMOIS/SH2O stay zero: init_soil_1_real writes none, and nothing in the
      slab reads soil water.

    DECLARED DIVERGENCE (default on): sea-ice columns take the LAND profile,
    TSK down to 271.4 K.  real.exe calls ``process_soil_real`` -- and so
    init_soil_1_real -- at module_initialize_real.F:3000, BEFORE
    ``adjust_for_seaice_pre`` at :3219 turns those columns into land, so it
    builds them the water profile (SST, or the ice skin where the source has
    no SST, in every layer) despite the comment above the call that says the
    adjustment comes first.  The slab never integrates its deepest layer, so
    WRF's order pins an ice column's lower boundary at the initial skin
    temperature for the whole run, and the TMN = 271.4 K it sets a few lines
    later is never read.  ``GPUWM_WRF_EXACT=1`` restores WRF's order.
    """
    import dataclasses

    from woof.core.slab import NUM_SOIL_LAYERS, initial_soil_temperature
    from woof.ingest.soil import (_host, _soil_temperature_elevation_delta,
                                   preprocess_noah_soil)
    from woof.wrf_exact import ENABLED as WRF_EXACT

    decision = landmask if landmask is not None else fields.get("LANDSEA")
    if decision is None:
        raise KeyError("the slab soil ingest needs LANDMASK or LANDSEA")
    from woof.slab_init_bridge import threshold_ice, initialize
    skin = _host(fields["SKINTEMP"])
    zeros_surface = np.zeros_like(skin)
    raw_ice = fields.get("XICE", fields.get("SEAICE"))
    terrestrial, xice = threshold_ice(
        _host(decision), zeros_surface if lake_mask is None else _host(lake_mask),
        skin, zeros_surface if raw_ice is None else _host(raw_ice), seaice_threshold)
    fields = dict(fields)
    fields["XICE"] = xice

    temperature, _moisture, levels, _geometry, scale = \
        _source_soil_profiles(fields, soil_layer_contract, land=terrestrial)
    surface = preprocess_noah_soil(
        fields,
        soil_type=soil_type,
        deep_soil_temperature=deep_soil_temperature,
        lake_mask=lake_mask, lake_skin_temperature=lake_skin_temperature,
        soil_layer_contract=soil_layer_contract,
        landmask=landmask, terrain=terrain,
        source_orography=source_orography,
        water_temperature=water_temperature,
        water_temperature_policy=water_temperature_policy,
        soil_mesh=soil_mesh, route=route)

    tsk = np.asarray(surface.tsk, np.float64)
    delta = delta_zero = np.zeros_like(tsk)
    if terrain is not None:
        delta = _soil_temperature_elevation_delta(terrain, source_orography, terrestrial)
        delta_zero = _soil_temperature_elevation_delta(terrain, np.zeros_like(_host(terrain)), terrestrial)
    tmn, tslb = initialize(
        temperature, levels, scale, terrestrial, surface.landmask, tsk,
        _host(fields["SST"]) if "SST" in fields else tsk,
        delta, delta_zero, terrain is not None, WRF_EXACT)
    zeros = np.zeros((NUM_SOIL_LAYERS,) + tsk.shape, np.float64)
    return dataclasses.replace(
        surface,
        soil_temperature=tslb.astype(np.float64),
        soil_moisture=zeros, liquid_moisture=zeros.copy(),
        deep_soil_temperature=tmn.astype(np.float32).astype(np.float64))


def preprocess_land_surface_soil(
    fields,
    *,
    sf_surface_physics: int,
    soil_type,
    deep_soil_temperature=None,
    lake_mask=None,
    lake_skin_temperature=None,
    soil_layer_contract=None,
    num_soil_layers: int | None = None,
    landmask=None,
    terrain=None,
    source_orography=None,
    water_temperature=None,
    water_temperature_policy=None,
    soil_mesh=None,
    route=None,
    fractional_seaice: bool = False,
    soil_no_source_land=None,
    seaice_threshold: float = 100.0,
    land_state=None,
):
    """Route a soil source to the selected land surface's own geometry.

    This is the one seam every initializer calls, and it exists because the
    seven of them called :func:`woof.ingest.soil.preprocess_noah_soil`
    unconditionally with no land-surface selector in sight -- so a
    ``sf_surface_physics=3`` request was initialised on Noah's four layer
    midpoints, silently, and the nine-level remap beside it was dead code.

    **Fail-closed on the selector, not on the shape.**  An unrecognised
    ``sf_surface_physics`` raises here rather than falling through to Noah.
    That direction matters: the failure this replaces was not a crash, it was
    a complete plausible forecast on the wrong soil discretization.  The
    shape check downstream (``_as_soil``) would have caught a four-layer array
    handed to a nine-layer allocation, but only as an unexplained broadcast
    error, and only for the schemes whose counts differ.

    ``soil_no_source_land`` is the horizontal snapshot's mask of land the
    source holds no land for (an island in a source area of open sea).
    Those cells carry METGRID.TBL fill_missing in ``fields``, and they take
    :func:`woof.ingest.soil.island_soil_columns` instead, before either
    geometry reads its source profiles, so every scheme builds its levels
    from the same column.  ``None`` or an empty mask changes nothing.

    ``land_state`` (:class:`woof.ingest.land_state.LandStateContext`) is
    where and when the domain is.  Every call converts a source's soil
    moisture to the target land model's hydraulics when their soil models
    differ (the soil-moisture index remap, woof/data/soil/land-state.v1.json);
    with a context, a same-model analysis that covers the domain (HRRR's RUC
    soil for a RUC run over CONUS) then replaces it on the land it covers.
    Named breakage: an ERA5 start put HTESSEL soil moisture into RUC as is,
    0.36 against HRRR's 0.21, and T2 ran 1-2 C cold with no storms
    (2026-10-26 cold-start study).
    """

    if not isinstance(fractional_seaice, bool):
        raise TypeError("fractional_seaice must be boolean")
    if soil_no_source_land is not None:
        from woof.ingest.soil import island_soil_columns

        fields, _ = island_soil_columns(
            fields, no_source_land=soil_no_source_land, soil_type=soil_type,
            landmask=landmask, lake_mask=lake_mask)
    scheme = int(sf_surface_physics)
    deep_receipt = None
    from woof.ingest.deep_soil_donor import has_deep_donor
    if has_deep_donor(soil_layer_contract):
        fields, soil_layer_contract, deep_receipt = _apply_deep_soil_donor(
            fields, soil_layer_contract, scheme=scheme, context=land_state,
            landmask=landmask)
    from woof.ingest.land_state import (apply_land_state_donor,
                                         convert_source_soil_moisture)

    fields, land_receipt = convert_source_soil_moisture(
        fields, sf_surface_physics=scheme, soil_type=soil_type,
        landmask=landmask, lake_mask=lake_mask,
        soil_layer_contract=soil_layer_contract)
    state = _route_land_surface_soil(
        fields, scheme=scheme, soil_type=soil_type,
        deep_soil_temperature=deep_soil_temperature, lake_mask=lake_mask,
        lake_skin_temperature=lake_skin_temperature,
        soil_layer_contract=soil_layer_contract,
        num_soil_layers=num_soil_layers, landmask=landmask, terrain=terrain,
        source_orography=source_orography,
        water_temperature=water_temperature,
        water_temperature_policy=water_temperature_policy,
        soil_mesh=soil_mesh, route=route,
        fractional_seaice=fractional_seaice,
        seaice_threshold=seaice_threshold)
    state, land_receipt = apply_land_state_donor(
        state, sf_surface_physics=scheme, context=land_state,
        receipt=land_receipt, landmask=landmask, lake_mask=lake_mask)
    if deep_receipt is not None:
        land_receipt = {**land_receipt, "deep_soil": deep_receipt}
    import dataclasses
    return dataclasses.replace(state, land_state=land_receipt)


def _apply_deep_soil_donor(fields, contract, *, scheme, context, landmask):
    """The declared deep-soil donor's layers under the source's own.

    ``(fields, contract, receipt)``.  A source that declares one
    (woof/ingest/deep_soil_donor.py; ECCC's GEM rows, which publish one
    0-10 cm slab) gets the donor's layers stacked below its own, and the
    stacked contract goes on to the Noah or RUC arm.  When the donor cannot
    be read: RUC refuses, naming why (it cannot start from the one slab:
    init_soil_3_real's 0 m moisture anchor needs two layers); Noah and
    Noah-MP fall back to the source's own layer for this run, loudly, since
    their remap has always carried one layer down to the deep-soil
    temperature.
    """

    from woof.ingest.deep_soil_donor import (extend_mapped_soil_fields,
                                              primary_soil_layer_contract)
    from woof.ingest.soil_contract import (MAPPED_SOIL_MOISTURE,
                                            MAPPED_SOIL_TEMPERATURE)

    if MAPPED_SOIL_TEMPERATURE not in fields or MAPPED_SOIL_MOISTURE not in fields:
        return fields, primary_soil_layer_contract(contract), None
    try:
        return extend_mapped_soil_fields(fields, contract, context=context,
                                         landmask=landmask)
    except Exception as error:
        if scheme == RUCLSMSCHEME:
            raise RuntimeError(
                "the RUC land surface needs the soil column below this "
                "source's one published layer, from its declared deep-soil "
                f"donor, and the donor could not be read: {error}") from error
        print("warning: deep soil: the declared donor could not be read "
              f"({type(error).__name__}: {error}); this run carries the "
              "source's own top layer down instead", file=sys.stderr,
              flush=True)
        return (fields, primary_soil_layer_contract(contract),
                {"fallback": f"{type(error).__name__}: {error}"})


def _route_land_surface_soil(
    fields, *, scheme, soil_type, deep_soil_temperature, lake_mask,
    lake_skin_temperature, soil_layer_contract, num_soil_layers, landmask,
    terrain, source_orography, water_temperature, water_temperature_policy,
    soil_mesh, route, fractional_seaice, seaice_threshold,
):
    """The dispatch on the land-surface selector (see the caller)."""
    if scheme == 0:
        from woof.ingest.soil import preprocess_noah_soil
        # NOLSMSCHEME has no init_soil_* arm in module_soil_pre.F:785-899.
        # Keep the shared skin/snow/land/ice preparation, but never remap,
        # partition or downscale an inactive soil column.
        return preprocess_noah_soil(
            fields, soil_type=soil_type,
            deep_soil_temperature=deep_soil_temperature,
            lake_mask=lake_mask, lake_skin_temperature=lake_skin_temperature,
            soil_layer_contract=soil_layer_contract, landmask=landmask,
            terrain=terrain, source_orography=source_orography,
            water_temperature=water_temperature,
            water_temperature_policy=water_temperature_policy,
            route=route, fractional_seaice=fractional_seaice,
            seaice_threshold=seaice_threshold,
            inactive_soil_layers=4 if num_soil_layers is None else int(num_soil_layers))
    if scheme == RUCLSMSCHEME:
        # The landmask/terrain/source-orography seam is forwarded whole:
        # preprocess_ruc_soil implements it with preprocess_noah_soil's
        # exact semantics (static-mask decision surface, scheme-independent
        # adjust_soil_temp_new lapse on TSK and every source temperature
        # level).  This replaced a fail-closed raise that existed while the
        # RUC side of the seam was unported; the genuine refusals -- an
        # unpaired terrain/source_orography, a non-boolean landmask, the
        # Noah-bounds mapped contract -- live in the callees and survive.
        return preprocess_ruc_soil(
            fields,
            **({"fractional_seaice": True} if fractional_seaice else {}),
            **_threshold_keyword(seaice_threshold),
            soil_type=soil_type,
            deep_soil_temperature=deep_soil_temperature,
            lake_mask=lake_mask, lake_skin_temperature=lake_skin_temperature,
            soil_layer_contract=soil_layer_contract,
            landmask=landmask, terrain=terrain,
            source_orography=source_orography,
            water_temperature=water_temperature,
            water_temperature_policy=water_temperature_policy,
            soil_mesh=soil_mesh,
            route=route,
            **({} if num_soil_layers is None
               else {"num_soil_layers": int(num_soil_layers)}))
    if scheme == SLABSCHEME:
        if fractional_seaice:
            # module_check_a_mundo.F:1008-1017, WRF's own fatal.
            raise ValueError(
                "fractional_seaice=1 cannot run with the slab scheme "
                "(sf_surface_physics=1): WRF refuses the pair, because the "
                "slab has no open-water fraction to blend")
        return preprocess_slab_soil(
            fields,
            soil_type=soil_type,
            deep_soil_temperature=deep_soil_temperature,
            lake_mask=lake_mask, lake_skin_temperature=lake_skin_temperature,
            soil_layer_contract=soil_layer_contract,
            landmask=landmask, terrain=terrain,
            source_orography=source_orography,
            water_temperature=water_temperature,
            water_temperature_policy=water_temperature_policy,
            soil_mesh=soil_mesh, route=route,
            seaice_threshold=seaice_threshold)
    if scheme in _NOAH_GEOMETRY_SCHEMES:
        from woof.ingest.soil import preprocess_noah_soil

        # Called with exactly preprocess_noah_soil's own argument list and
        # nothing added, so a Noah or Noah-MP run's soil state is the same
        # object it was before this seam existed.
        return preprocess_noah_soil(
            fields,
            **({"fractional_seaice": True} if fractional_seaice else {}),
            **_threshold_keyword(seaice_threshold),
            soil_type=soil_type,
            deep_soil_temperature=deep_soil_temperature,
            lake_mask=lake_mask, lake_skin_temperature=lake_skin_temperature,
            soil_layer_contract=soil_layer_contract,
            landmask=landmask, terrain=terrain,
            source_orography=source_orography,
            water_temperature=water_temperature,
            water_temperature_policy=water_temperature_policy,
            soil_mesh=soil_mesh,
            route=route)
    from woof.physics_registry import column_surface_contract
    contract = column_surface_contract(scheme)
    ingest = (contract or {}).get("soil_ingest")
    if ingest is not None:
        if ingest["method"] != "wrf_init_soil_2_real":
            raise ValueError("the registered soil ingest method is not implemented")
        if num_soil_layers is not None and int(num_soil_layers) != len(ingest["depths_m"]):
            raise ValueError("num_soil_layers differs from the registered soil ingest geometry")
        from woof.ingest.soil import preprocess_noah_soil
        return preprocess_noah_soil(
            fields, soil_type=soil_type, deep_soil_temperature=deep_soil_temperature,
            **_threshold_keyword(seaice_threshold),
            lake_mask=lake_mask, lake_skin_temperature=lake_skin_temperature,
            soil_layer_contract=soil_layer_contract, landmask=landmask,
            terrain=terrain, source_orography=source_orography,
            water_temperature=water_temperature, water_temperature_policy=water_temperature_policy,
            soil_mesh=soil_mesh, route=route, fractional_seaice=fractional_seaice,
            target_geometry=ingest)
    raise ValueError(
        f"sf_surface_physics={scheme} has no soil ingest.  woof implements "
        f"the slab geometry for {SLABSCHEME} "
        "(module_soil_pre.F:init_soil_1_real), "
        f"the Noah geometry for {_NOAH_GEOMETRY_SCHEMES} "
        "(module_soil_pre.F:init_soil_2_real) and the RUC level geometry for "
        f"{RUCLSMSCHEME} (init_soil_3_real); routing anything else through "
        "either would initialise a forecast on a soil discretization the "
        "scheme does not use"
    )


__all__ = [
    "RUCLSMSCHEME",
    "SLABSCHEME",
    "SLAB_TMN_SOURCE_DEPTH_CM",
    "preprocess_slab_soil",
    "RUC_LEVEL_DEPTHS_M",
    "RucSoilColumns",
    "RucSoilState",
    "preprocess_land_surface_soil",
    "preprocess_ruc_soil",
    "remap_soil_to_ruc_levels",
    "ruc_soil_depths",
]
