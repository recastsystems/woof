"""Config-driven initial-state theta bubbles for real-data experiments.

The [perturbation] block (:mod:`woof.experiment`) declares warm bubbles
in geographic coordinates; this module evaluates them on one domain's
grid and writes them into that domain's initial theta (and, under
``rh_preserve``, qv) columns inside
:func:`woof.ingest.real.initialize_real`, before the specific volume
and geopotential are formed.  The shape is WRF's em_quarter_ss bubble
(module_initialize_ideal.F; the port's transcription is
woof/verify/cases/moist_bubble.py): peak ``amplitude_k`` at the
center, ``cos^2(pi/2 * r)`` taper on the normalized ellipse radius
``r = sqrt((d_h/radius)^2 + ((z - z_c)/depth)^2)``, exactly zero at and
beyond ``r = 1``.

PER-DOMAIN APPLICATION, deliberately.  woof's real-data nest init is
real.exe-per-domain, not ndown: every child re-ingests the source
analysis on its own grid (woof/ingest/nest_init.py), so a bubble
written into the parent's prepared state would be invisible to a
child's initial conditions.  Each domain that initializes at the
experiment start time therefore evaluates the same geographic bubble on
its own grid inside its own ``initialize_real`` -- the one init path
every domain shares -- and the per-domain receipts record what each
grid actually received.  A domain with a delayed start initializes from
the analysis at its activation time, by which point the parent has
already evolved the bubble, so it takes no fresh analytic bubble.

OFF is absolute: with no [perturbation] block
:func:`build_initial_state_perturbation` is never called, callers pass
``initial_perturbation=None``, and ``initialize_real`` executes not one
instruction of this module -- the prepared state is byte-identical to a
build without this file.

Refusals (loud, before any integration):

* a bubble center outside the coarse domain (``require_containment``);
* an enabled bubble whose center is inside the domain but which touches
  zero cells (radius below the grid's resolving power) -- a refusal,
  never a silent no-op, per the treatment-proof rule;
* a perturbed layer hotter than the top of the temperature table the
  domain's radiation reads (RTE+RRTMGP: 355 K), because RRTMGP refuses
  that layer at its first call and the forecast stops at step 1;
* ``rh_preserve`` building water vapour past
  :data:`RH_PRESERVE_QV_LIMIT_KG_KG`, the measured mixing ratio past which
  the forecast stops being finite (see the constant).

This is an ArWen-over-WRF extension (PROVENANCE.md): stock WRF v4.6.1
has no config-driven real-data initial-condition perturbation; its warm
bubbles exist only in the idealized initializers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from woof.core import constants as c
from woof.core import portable_math as pm
from woof.static.projection import EARTH_RADIUS_M

APPLICATION_SCHEMA = "gpuwm-initial-perturbation-apply-v1"

#: The largest water vapour mixing ratio ``rh_preserve`` may build in a
#: bubble cell.  Holding RH through a large warming builds vapour fast
#: (saturation vapour pressure roughly doubles every 10 K), and past
#: this the forecast stops being finite.  Measured with 12/3 km trees
#: at 35.45 N, 97.95 W (bubble radius 10 km, half-depth 1500 m,
#: centred 1500 m above ground on the 3 km grid, 1 h forecasts, RTX
#: 5090): the bubbles that built 0.046 (GFS, 30 K), 0.054 (ERA5, 30 K)
#: and 0.084 kg/kg (ERA5, 40 K) ran to the end with every field finite;
#: the ones that built 0.092 (GFS, 45 K), 0.104, 0.131, 0.166, 0.211
#: (ERA5, 45 to 60 K) and 0.185 kg/kg (GFS, 60 K) went non-finite on
#: the 3 km grid at model step 21 or sooner, inside six minutes.
RH_PRESERVE_QV_LIMIT_KG_KG = 0.09


class TemperatureCeiling(NamedTuple):
    """The hottest layer a domain's radiation accepts, and what it does."""

    kelvin: float
    tables: str            # what the ceiling is the top of
    failure: str           # what that radiation does past it


def radiation_temperature_ceiling(cfg) -> TemperatureCeiling | None:
    """The layer-temperature ceiling of a domain's radiation, or ``None``.

    A table, not a code path: each radiation implementation that refuses
    a layer temperature above a fixed value names that value here.
    Only RTE+RRTMGP does today.  Its gas-optics tables span 160..355 K
    and every call refuses a layer outside them, so a forecast whose
    initial state holds a hotter layer stops at step 1 (measured: a 60 K
    bubble on a GFS 12/3 km tree, applied before the prepared-tree
    geopotential rebalance existed, left a 363 K layer and the forecast
    exited 2 at step 1 with "tlay range [205.987, 363.344269] K is
    outside allowed range [160, 355] K").  The legacy RRTMG port, RRTM,
    Dudhia and the analytic scheme carry no such refusal, and a domain
    with radiation off reads none.
    """
    from woof.config import radiation_scheme_ids
    from woof.physics_compat import RRTMG_VARIANT_RTE_RRTMGP, rrtmg_variant

    if rrtmg_variant(cfg) != RRTMG_VARIANT_RTE_RRTMGP:
        return None
    kinds = tuple(kind for kind, selector
                  in zip(("lw", "sw"), radiation_scheme_ids(cfg))
                  if selector == 4)
    if not kinds:
        return None
    from woof.core.rrtmgp import gas_table_temperature_range_k

    return TemperatureCeiling(
        kelvin=min(gas_table_temperature_range_k(kind)[1] for kind in kinds),
        tables="RTE+RRTMGP gas-optics tables",
        failure=("RRTMGP refuses a layer outside them at its first call "
                 "(\"tlay range ... is outside allowed range\"), so the "
                 "forecast would stop at step 1"))


def _great_circle_km(lat, lon, center_lat, center_lon) -> np.ndarray:
    """Haversine distance in km on the WPS sphere (EARTH_RADIUS_M)."""
    lat = np.asarray(lat, dtype=np.float64) * (np.pi / 180.0)
    lon = np.asarray(lon, dtype=np.float64) * (np.pi / 180.0)
    lat0 = float(center_lat) * (np.pi / 180.0)
    lon0 = float(center_lon) * (np.pi / 180.0)
    half_dlat = 0.5 * (lat - lat0)
    half_dlon = 0.5 * (lon - lon0)
    h = (pm.sin(half_dlat) ** 2
         + pm.cos(lat) * pm.cos(lat0) * pm.sin(half_dlon) ** 2)
    return (2.0 * EARTH_RADIUS_M / 1000.0) * pm.arcsin(np.sqrt(h))


@dataclass(frozen=True)
class _PlacedBubble:
    """One bubble resolved onto a specific domain's grid."""

    spec: object                 # woof.experiment.BubbleConfig
    index: int                   # 1-based position in the config block
    center_x: float              # 1-based projection coordinate (mass i)
    center_y: float              # 1-based projection coordinate (mass j)
    inside: bool
    horizontal_km: np.ndarray | None = field(repr=False, default=None)


class InitialStatePerturbation:
    """The [perturbation] block evaluated for one domain's grid.

    Built by :func:`build_initial_state_perturbation` (which returns
    ``None`` for an absent block -- the OFF contract), consumed exactly
    once by ``initialize_real``.  ``apply`` mutates the host FP64
    theta/qv columns in place and returns the per-domain receipt.
    ``temperature_ceiling`` is :func:`radiation_temperature_ceiling` of
    the domain's configuration; ``None`` checks no layer temperature.
    """

    def __init__(self, bubbles, grid, *, grid_id: int,
                 require_containment: bool,
                 temperature_ceiling: TemperatureCeiling | None = None):
        self.grid_id = int(grid_id)
        self.temperature_ceiling = temperature_ceiling
        nx = int(grid.e_we) - 1
        ny = int(grid.e_sn) - 1
        lat = lon = None
        placed = []
        for index, spec in enumerate(bubbles, start=1):
            x, y = grid.latlon_to_ij(spec.center_lat, spec.center_lon)
            x, y = float(x), float(y)
            # Mass point (i, j) sits at projection coordinate (i, j),
            # 1-based, i in [1, e_we-1], j in [1, e_sn-1] (static/lambert.py
            # grid-registration contract).
            inside = (1.0 <= x <= float(nx)) and (1.0 <= y <= float(ny))
            if require_containment and not inside:
                raise ValueError(
                    f"perturbation.bubbles #{index} center "
                    f"({spec.center_lat:g}, {spec.center_lon:g}) lies "
                    f"outside domain d{self.grid_id:02d} (projection "
                    f"coordinate ({x:.2f}, {y:.2f}) of a {nx} x {ny} mass "
                    "grid); every bubble must sit inside the coarse "
                    "domain. Move the bubble or widen the domain.")
            horizontal_km = None
            if inside:
                if lat is None:
                    lat, lon = grid.latlon_mass()
                horizontal_km = _great_circle_km(
                    lat, lon, spec.center_lat, spec.center_lon)
            placed.append(_PlacedBubble(
                spec=spec, index=index, center_x=x, center_y=y,
                inside=inside, horizontal_km=horizontal_km))
        self._placed = tuple(placed)

    def apply(self, *, theta, qv, pressure, z_half_agl, allow_empty=False) -> dict:
        """Add every contained bubble to ``theta`` (and qv) in place.

        ``theta``/``qv``/``pressure`` are the final host FP64
        ``(nz, ny, nx)`` columns of ``initialize_real``; ``z_half_agl``
        the matching half-level heights AGL.  Cells outside every
        bubble's ``r < 1`` ellipse are byte-untouched.  Returns the
        per-domain application receipt; raises when an in-domain bubble
        touches zero cells, when ``rh_preserve`` builds more vapour than
        :data:`RH_PRESERVE_QV_LIMIT_KG_KG`, or when a perturbed layer at
        ``pressure`` is hotter than the domain's radiation accepts.
        """
        theta = np.asarray(theta)
        rows = []
        masks = []
        for placed in self._placed:
            spec = placed.spec
            if not placed.inside:
                rows.append({
                    "bubble": placed.index,
                    "applied": False,
                    "reason": "center outside this domain",
                    "center_xy": [placed.center_x, placed.center_y],
                    "cells_touched": 0,
                })
                continue
            radial = np.sqrt(
                (placed.horizontal_km[None, :, :] / spec.radius_km) ** 2
                + ((np.asarray(z_half_agl, dtype=np.float64)
                    - spec.center_height_m) / spec.depth_m) ** 2)
            mask = radial < 1.0
            cells = int(np.count_nonzero(mask))
            if cells == 0 and not allow_empty:
                raise ValueError(
                    f"perturbation.bubbles #{placed.index} is enabled and "
                    f"centered inside domain d{self.grid_id:02d} "
                    f"(projection coordinate ({placed.center_x:.2f}, "
                    f"{placed.center_y:.2f})) but touches zero cells: "
                    f"radius_km = {spec.radius_km:g} / depth_m = "
                    f"{spec.depth_m:g} are below this grid's resolving "
                    "power. An applied perturbation that writes nothing "
                    "is a refusal, not a silent no-op -- enlarge the "
                    "bubble or drop it.")
            delta = np.zeros_like(theta[mask])
            delta[...] = (spec.amplitude_k
                          * pm.cos(0.5 * np.pi * radial[mask]) ** 2)
            row = {
                "bubble": placed.index,
                "applied": True,
                "center_xy": [placed.center_x, placed.center_y],
                "cells_touched": cells,
                "max_theta_added_k": float(delta.max()) if cells else 0.0,
                "rh_preserve": bool(spec.rh_preserve),
            }
            if spec.rh_preserve and cells:
                row["max_qv_delta_kg_kg"] = self._preserve_rh(
                    theta, qv, pressure, mask, delta)
                self._refuse_vapour_past_limit(placed, qv, mask)
            theta[mask] += delta
            rows.append(row)
            masks.append((placed.index, mask))
        self._refuse_layer_past_radiation_ceiling(theta, pressure, masks)
        return {
            "schema": APPLICATION_SCHEMA,
            "grid_id": self.grid_id,
            "bubbles": rows,
        }

    def _refuse_vapour_past_limit(self, placed, qv, mask) -> None:
        """Refuse an ``rh_preserve`` bubble that built too much vapour."""
        built = float(np.max(np.asarray(qv)[mask]))
        if built <= RH_PRESERVE_QV_LIMIT_KG_KG:
            return
        spec = placed.spec
        raise ValueError(
            f"perturbation.bubbles #{placed.index} (amplitude_k = "
            f"{spec.amplitude_k:g} K, rh_preserve = true) builds water "
            f"vapour up to {built:.3f} kg/kg on domain "
            f"d{self.grid_id:02d}, above the "
            f"{RH_PRESERVE_QV_LIMIT_KG_KG:g} kg/kg this check allows: "
            "3 km forecasts whose bubbles built 0.092 kg/kg and more "
            "went non-finite within six minutes, and ones that built "
            "0.084 kg/kg and less ran. Lower amplitude_k, or set "
            "rh_preserve = false to keep the analysed vapour.")

    def _refuse_layer_past_radiation_ceiling(self, theta, pressure,
                                             masks) -> None:
        """Refuse a perturbed layer hotter than the radiation accepts."""
        if self.temperature_ceiling is None or not masks:
            return
        ceiling = self.temperature_ceiling.kelvin
        touched = np.zeros(np.shape(theta), dtype=bool)
        for _, mask in masks:
            touched |= mask
        if not touched.any():
            return
        p_cells = np.asarray(pressure, dtype=np.float64)[touched]
        temperature = (np.asarray(theta, dtype=np.float64)[touched]
                       * pm.power(p_cells / c.P0, c.RCP))
        hottest = int(np.argmax(temperature))
        if temperature[hottest] <= ceiling:
            return
        cell = tuple(int(axis[hottest]) for axis in np.nonzero(touched))
        names = ", ".join(f"#{index}" for index, mask in masks if mask[cell])
        raise ValueError(
            f"perturbation.bubbles {names} heats a layer on domain "
            f"d{self.grid_id:02d} to {temperature[hottest]:.1f} K at "
            f"{p_cells[hottest] / 100.0:.0f} hPa, above {ceiling:g} K, "
            f"the top of the {self.temperature_ceiling.tables} this "
            f"domain's radiation reads. {self.temperature_ceiling.failure}. "
            "Lower amplitude_k, or raise center_height_m into colder "
            "air.")

    @staticmethod
    def _preserve_rh(theta, qv, pressure, mask, delta) -> float:
        """Adjust qv on ``mask`` so RH survives the theta change.

        WRF-faithful pair from :mod:`woof.ingest.real` (imported lazily
        -- real.py must not import this module back): RH diagnosed from
        the unperturbed T/p/qv with the exact algebraic inverse of
        rh_to_mxrat1, then qv rebuilt from that RH at the perturbed
        temperature and the SAME analyzed pressure.  Only masked cells
        are written.
        """
        from woof.ingest.real import (
            _mixing_ratio_to_relative_humidity,
            _saturation_mixing_ratio,
            _temperature_from_potential_temperature,
        )

        qv = np.asarray(qv)
        p_cells = np.asarray(pressure, dtype=np.float64)[mask]
        theta_cells = np.asarray(theta, dtype=np.float64)[mask]
        t_old = _temperature_from_potential_temperature(
            theta_cells, p_cells)
        # allow_wps_undershoot: the native-HRRR lane legitimately
        # carries bounded negative SPFH undershoot in its columns (the
        # same envelope real.py's own RH diagnosis admits); such a cell
        # diagnoses a negative RH, which the saturation relation below
        # clips to WRF's own 1e-6 floor.
        rh = _mixing_ratio_to_relative_humidity(
            t_old, p_cells, qv[mask], allow_wps_undershoot=True)
        t_new = _temperature_from_potential_temperature(
            theta_cells + delta, p_cells)
        qv_new = _saturation_mixing_ratio(t_new, p_cells, rh)
        if not np.isfinite(qv_new).all() or np.any(qv_new < 0.0):
            raise ValueError(
                "rh_preserve produced an invalid qv inside the bubble")
        max_delta = float(np.max(np.abs(qv_new - qv[mask])))
        qv[mask] = qv_new
        return max_delta


    def apply_to_state(self, state, *, allow_empty=False) -> dict:
        """Add the bubbles to an already-initialized :class:`DomainState`.

        The prepared-cache route (``woof.prepared_domain_tree_forecast``)
        restores sealed, unperturbed per-domain states and never passes
        through ``initialize_real``; this is that route's application
        point, run after restore and BEFORE ``initialize_prepared_physics``
        whose ``update_diagnostics`` then rederives p/al/alt from the
        perturbed prognostics.  The bubble is evaluated in FP64 on host
        against the restored full theta, diagnosed pressure ``state.p``
        and geopotential heights AGL, then written back to ``thp`` (and
        ``qv`` under ``rh_preserve``) on the bubble cells ONLY.

        The column is then rebalanced hydrostatically at the HELD
        pressure, as WRF's em_quarter_ss initializer does after its
        bubble ("rebalance hydrostatically", module_initialize_ideal.F;
        the port's ``init_moist_balanced`` in woof/core/moist.py): the
        dry column mass and ``p`` stay, and every layer the bubble warmed
        is thickened by the ratio of its new to old moist potential
        temperature, which is exactly how the specific volume changes at
        fixed pressure through the EOS ``update_diagnostics`` inverts
        (either hypsometric option; the dry-mass factor cancels).  The
        thickening accumulates up the column into ``php``.  Without it
        the diagnostic would hold the geopotential and raise the
        pressure instead: +25 % (20.4 kPa) at the core of a 60 K bubble
        on a GFS 12/3 km tree, which RRTMGP then refused at step 1 as a
        363 K layer.  Cells outside every bubble and the columns' levels
        below their lowest bubble cell keep their exact restored bytes.
        The receipt names the route so the two application points stay
        distinguishable.
        """
        def host(value):
            return np.asarray(value.get() if hasattr(value, "get")
                              else value, dtype=np.float64)

        def profile(value):
            value = host(value)
            return value[:, None, None] if value.ndim == 1 else value

        thb = profile(state.thb)          # flat base state: 1-D column
        theta = thb + host(state.thp)
        qv = host(state.qv)
        use_theta_m = int(getattr(state, "use_theta_m", 0))
        if use_theta_m:
            theta = theta / (1.0 + c.RVOVRD * qv)
        theta_before = theta.copy()
        qv_before = qv.copy()
        pressure = host(state.p)
        php = host(state.php)
        full_phi = profile(state.phb) + php
        z_half_agl = (0.5 * (full_phi[:-1] + full_phi[1:])
                      - full_phi[:1]) / c.G
        receipt = self.apply(theta=theta, qv=qv, pressure=pressure,
                             z_half_agl=z_half_agl, allow_empty=allow_empty)
        receipt["application_point"] = "restored-prepared-state"
        receipt["geopotential"] = "rebalanced at the held pressure"
        changed = theta != theta_before
        qv_changed = qv != qv_before
        if changed.any() or qv_changed.any():
            # alt ~ theta * (1 + Rv/Rd qv) at fixed p and mu, and each
            # layer's thickness is alt times a factor of mu alone, so the
            # ratio IS the thickness ratio.  Exactly 1 on untouched cells.
            ratio = ((theta * (1.0 + c.RVOVRD * qv))
                     / (theta_before * (1.0 + c.RVOVRD * qv_before)))
            thickness = full_phi[1:] - full_phi[:-1]
            resid = getattr(state, "dphb_resid", None)
            if resid is not None:
                # The base thickness the EOS kernel actually reads.
                thickness = thickness + profile(resid)
            rise = np.zeros_like(php)
            np.cumsum(thickness * (ratio - 1.0), axis=0, out=rise[1:])
            _write_cells(state.php, php + rise, rise != 0.0)
        if use_theta_m:
            _write_cells(state.thp, theta * (1.0 + c.RVOVRD * qv) - thb,
                         changed | qv_changed)
            state.thp0[...] = state.thp
        else:
            _write_cells(state.thp, theta - thb, changed)
        _write_cells(state.qv, qv, qv_changed)
        return receipt


def _write_cells(target, values, cells) -> None:
    """Write ``values[cells]`` into a host or device FP32 field."""
    if not cells.any():
        return
    if type(target).__module__.startswith("cupy"):
        import cupy as cp
        target[cp.asarray(cells)] = cp.asarray(
            values[cells].astype(np.float32))
    else:
        target[cells] = values[cells].astype(np.float32)


def build_initial_state_perturbation(perturbation, grid, *, grid_id: int,
                                     require_containment: bool, cfg):
    """The construction entry: ``None`` unless a block is configured.

    ``perturbation`` is ``ExperimentConfig.perturbation``.  The ``None``
    return is the whole OFF contract -- the caller holds no object and
    ``initialize_real`` executes no instruction of this module.
    ``require_containment`` is set by the COARSE domain (and the
    single-domain path): a center outside it is a configuration error.
    A child legitimately may not contain a bubble; its receipt says so.
    ``cfg`` is the domain's own RunConfig; its radiation names the
    layer-temperature ceiling the application checks.
    """
    if perturbation is None:
        return None
    return InitialStatePerturbation(
        perturbation.bubbles, grid, grid_id=grid_id,
        require_containment=require_containment,
        temperature_ceiling=radiation_temperature_ceiling(cfg))


__all__ = [
    "APPLICATION_SCHEMA",
    "InitialStatePerturbation",
    "RH_PRESERVE_QV_LIMIT_KG_KG",
    "TemperatureCeiling",
    "build_initial_state_perturbation",
    "radiation_temperature_ceiling",
]
