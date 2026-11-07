"""WRF v4.6.1 NSSL two-moment (``mp_physics=18``) contract.

This module is deliberately CPU-only.  It pins the unified NSSL selector,
resolved default mode, transported state, radiation diagnostics, and numerical
setup before the complete CUDA driver is admitted.  The authority is NCAR WRF
v4.6.1 commit ``d66e442fccc04111067e29274c9f9eaccc3cef28``:

* ``Registry/Registry.EM_COMMON:2410-2425,3033,3049-3056``;
* ``share/module_check_a_mundo.F:3377-3465``;
* ``phys/module_physics_init.F:4615-4652``; and
* ``phys/module_mp_nssl_2mom.F``.

Since WRF v4.6 the former NSSL IDs are compatibility spellings of option 18.
The target here is the native option-18 default: two moments, hail, predicted
CCN, and predicted graupel/hail volume (density), without the optional sixth
moments.  Importing this contract does not make the scheme executable.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping


MP_PHYSICS = 18
WRF_REFERENCE_VERSION = "v4.6.1"
WRF_REFERENCE_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"
CONTRACT_ID = "wrf-v4.6.1-nssl-mp18-two-moment-hail-ccn-density-v1"

# Registry packages nssl_2mom + nssl2mconc + nssl_hail + nssl_ccn_opt
# + nssl_hailvol after module_check_a_mundo resolves all -1 selectors.
BASE_MASS_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg")
TWO_MOMENT_NUMBER_FIELDS = ("qndrop", "qnr", "qni", "qns", "qng")
HAIL_MASS_FIELD = "qh"
HAIL_NUMBER_FIELD = "qnh"
PREDICTED_CCN_FIELD = "qnn"
GRAUPEL_VOLUME_FIELD = "qvolg"
HAIL_VOLUME_FIELD = "qvolh"
SIXTH_MOMENT_FIELDS = ("qzr", "qzg", "qzh")

RADIATION_EFFECTIVE_RADIUS_FIELDS = ("re_cloud", "re_ice", "re_snow")
REFLECTIVITY_FIELDS = ("refl_10cm",)
PRECIPITATION_FIELDS = (
    "rainnc", "rainncv", "snownc", "snowncv",
    "graupelnc", "graupelncv", "hailnc", "hailncv",
)
HAIL_DIAGNOSTIC_FIELDS = ("hail_maxk1", "hail_max2d")

# Registry defaults are the runtime authority.  In particular, v4.6.1's
# Registry says 900 kg m-3 for nssl_rho_qhl although README.NSSLmp still says
# 800; module_physics_init forwards the Registry value into nssl_2mom_init.
WRF_NAMELIST_DEFAULTS: Mapping[str, int | float] = MappingProxyType({
    "nssl_cccn": 0.5e9,
    "nssl_alphah": 0.0,
    "nssl_alphahl": 1.0,
    "nssl_cnoh": 4.0e5,
    "nssl_cnohl": 4.0e4,
    "nssl_cnor": 8.0e5,
    "nssl_cnos": 3.0e6,
    "nssl_rho_qh": 500.0,
    "nssl_rho_qhl": 900.0,
    "nssl_rho_qs": 100.0,
    "nssl_icdx": 6,
    "nssl_icdxhl": 6,
    "nssl_2moment_on": -1,
    "nssl_hail_on": -1,
    "nssl_ccn_on": -1,
    "nssl_density_on": -1,
    "nssl_3moment": 0,
})


@dataclass(frozen=True)
class NSSL2Mode:
    """Selectors after WRF's option-18 consistency pass.

    ``density_moments`` follows the Registry package selector: 0 is fixed
    density, 1 predicts graupel volume, and 2 predicts graupel and hail volume.
    ``sixth_moments`` is 0, 1 (rain/graupel), or 2 (rain/graupel/hail).
    """

    two_moment: bool
    hail: bool
    predicted_ccn: bool
    density_moments: int
    sixth_moments: int

    @property
    def transported_fields(self) -> tuple[str, ...]:
        fields = list(BASE_MASS_FIELDS)
        if self.hail:
            fields.append(HAIL_MASS_FIELD)
        if self.two_moment:
            fields.extend(TWO_MOMENT_NUMBER_FIELDS)
            if self.hail:
                fields.append(HAIL_NUMBER_FIELD)
        if self.predicted_ccn:
            fields.append(PREDICTED_CCN_FIELD)
        if self.density_moments >= 1:
            fields.append(GRAUPEL_VOLUME_FIELD)
        if self.density_moments >= 2:
            fields.append(HAIL_VOLUME_FIELD)
        if self.sixth_moments >= 1:
            fields.extend(SIXTH_MOMENT_FIELDS[:2])
        if self.sixth_moments >= 2:
            fields.append(SIXTH_MOMENT_FIELDS[2])
        return tuple(fields)

    @property
    def radiation_fields(self) -> tuple[str, ...]:
        return RADIATION_EFFECTIVE_RADIUS_FIELDS if self.two_moment else ()


def nssl2_cccn_concentration(nssl_cccn: float = 0.5e9) -> np.float32:
    """Stock WRF's nssl_params(1) route: ABS of a finite REAL value.

    WRF 4.7.1 module_mp_nssl_2mom.F:1370 and module_physics_init.F:4642
    pass the parameter array. The optional scalar argument's >1 guard at
    :1407 does not apply to that route. There is no WRF coefficient range.
    Reject nonfinite/unrepresentable values before they poison activation;
    zero and negative namelist values retain WRF's ABS semantics.
    """
    if isinstance(nssl_cccn, bool):
        raise TypeError("nssl_cccn must be a finite float32 concentration")
    try:
        value = float(nssl_cccn)
    except (TypeError, ValueError) as exc:
        raise TypeError("nssl_cccn must be a finite float32 concentration") from exc
    if not math.isfinite(value) or abs(value) > float(np.finfo(np.float32).max):
        raise ValueError("nssl_cccn must be a finite float32 concentration; "
                         "undefined values would contaminate the CCN reservoir")
    return np.abs(np.float32(value))


def nssl2_background_ccn_per_kg(nssl_cccn: float = 0.5e9) -> np.float32:
    """WRF nssl_2mom_init:1997, one REAL division for the seed and inflow."""
    return nssl2_cccn_concentration(nssl_cccn) / np.float32(1.225)


def _selector(name: str, value: int, allowed: tuple[int, ...]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer selector, got {value!r}")
    if value not in allowed:
        options = ", ".join(str(item) for item in allowed)
        raise ValueError(f"{name} must be one of {options}, got {value}")
    return value


def resolve_nssl2_mode(
        *, mp_physics: int = MP_PHYSICS, nssl_2moment_on: int = -1,
        nssl_hail_on: int = -1, nssl_ccn_on: int = -1,
        nssl_density_on: int = -1, nssl_3moment: int = 0,
        ) -> NSSL2Mode:
    """Apply WRF v4.6.1's native option-18 default resolution.

    WRF rewrites nssl_3moment=1 to package value 2 with hail enabled.
    An explicit 2 is also a Registry package value and allocates qzh, including
    when hail is disabled. Only the public value 1 forces number prediction;
    module_check_a_mundo.F:3457-3463 applies that rewrite after defaults.
    Deprecated scheme IDs are intentionally not accepted here; an importer
    must canonicalize them explicitly rather than disguising a legacy setup.
    """
    if mp_physics != MP_PHYSICS:
        raise ValueError(
            f"NSSL-2 contract requires mp_physics={MP_PHYSICS}, got "
            f"{mp_physics}")
    two = _selector("nssl_2moment_on", nssl_2moment_on, (-1, 0, 1))
    hail = _selector("nssl_hail_on", nssl_hail_on, (-1, 0, 1, 2))
    ccn = _selector("nssl_ccn_on", nssl_ccn_on, (-1, 0, 1))
    density = _selector("nssl_density_on", nssl_density_on, (-1, 0, 1, 2))
    three = _selector("nssl_3moment", nssl_3moment, (0, 1, 2))

    two = 1 if two < 0 else two
    # module_check_a_mundo.F:3442-3448: an unset hail flag resolves to the
    # one-moment hail package (2) when number concentrations are off and to
    # the two-moment hail package (1) otherwise.
    hail = (1 if two else 2) if hail < 0 else hail
    ccn = 1 if ccn < 0 else ccn
    # module_check_a_mundo.F:3450-3455 keys the density default on
    # nssl_hail_on == 1 exactly, not on truthiness.
    density = (2 if hail == 1 else 1) if density < 0 else density
    # WRF applies this after hail and density defaults, so a pre-resolved
    # one-moment hail package (hail=2) is not silently changed to hail=1.
    if three == 1:
        two = 1
    if hail == 2 and two:
        # WRF's Registry allocates only the hail MASS field for
        # nssl_hail_on=2 (package nssl_hail1m, Registry.EM_COMMON:3053).
        # The scheme itself never sees the 2: module_physics_init.F:4648
        # passes `nssl_hail_on = config_flags%nssl_hail_on > 0` as a
        # LOGICAL, which module_mp_nssl_2mom.F:1251-1257 coerces back to
        # hail_on=1 -- indistinguishable inside the module from
        # nssl_hail_on=1.  With two moments that leaves lhl > 1, so the
        # gather at :2775, `IF ( lhl > 1 ) an(ix,1,kz,lnhl) = chl(...)`,
        # reads CHL.  module_microphysics_driver.F:1968 passes CHL as
        # qnh_curr unconditionally and with no f_ flag beside it, so with
        # qnh unallocated that is the null scalar slot -- an
        # inactive-package read with no defined value.  woof refuses
        # instead of replicating it (project rule: never bit-exact to a
        # bug).
        raise ValueError(
            "nssl_hail_on=2 (one-moment hail) is only defined with "
            "nssl_2moment_on=0; with two-moment concentrations WRF reads "
            "the unallocated hail-number field")
    if density == 2 and not hail:
        raise ValueError(
            "nssl_density_on=2 predicts hail volume and requires "
            "nssl_hail_on=1")
    if density == 1 and hail == 1:
        # The module cannot tell nssl_density_on=1 from 2 either:
        # module_physics_init.F:4647 passes the LOGICAL
        # `(config_flags%nssl_density_on > 0)`, coerced to density_on=1 at
        # module_mp_nssl_2mom.F:1259-1265, so with hail on it predicts
        # hail volume (lvhl > 0, :1674-1679) under BOTH settings.  The
        # Registry allocates qvolh only under nssl_density_on=2 (package
        # nssl_hailvol, Registry.EM_COMMON:3056); under =1 (nssl_graupelvol,
        # :3055) the field does not exist.  The load at :2778 and the store
        # at :3336 are `present( vhl )`-guarded, but the guard cannot fire:
        # dyn_em/solve_em.F:3869 passes
        # QVOLH_CURR=scalar(ims,kms,jms,P_QVOLH) unconditionally and
        # module_microphysics_driver.F:1970 forwards it as VHL, so
        # present(vhl) is TRUE while P_QVOLH is the null scalar index.  The
        # one flag that would say otherwise, f_vhl, is declared "not used
        # yet" at :2296 and never consulted.  So the scheme reads and
        # writes the inactive qvolh slot; woof refuses rather than
        # reproducing it.
        raise ValueError(
            "nssl_density_on=1 with nssl_hail_on=1 makes WRF predict hail "
            "volume through the unallocated qvolh field; use "
            "nssl_density_on=2 (or -1) with hail on")
    if three and not two:
        raise ValueError(
            "nssl_3moment=2 with nssl_2moment_on=0 leaves allocated sixth "
            "moments unchanged by the one-moment driver")

    return NSSL2Mode(
        two_moment=bool(two),
        hail=bool(hail),
        predicted_ccn=bool(ccn),
        density_moments=density,
        sixth_moments=(2 if hail else 1) if three == 1 else three,
    )


#: The five ``RunConfig`` fields that carry WRF's NSSL variant selectors.
NSSL2_SELECTOR_FIELDS = (
    "nssl_2moment_on", "nssl_hail_on", "nssl_ccn_on", "nssl_density_on",
    "nssl_3moment",
)


def resolve_nssl2_mode_for_config(cfg) -> NSSL2Mode:
    """Resolve the variant mode carried by a configuration object.

    ``cfg`` is anything exposing the five :data:`NSSL2_SELECTOR_FIELDS`
    attributes -- in practice :class:`woof.config.RunConfig`.  This is the
    ONE spelling of "which NSSL variant is this run", so a receipt, a
    binding and a kernel launch cannot disagree about it.  It does not
    gate: callers that need the ported-path refusal wrap the result in
    :func:`require_ported_nssl2_mode`.
    """
    # Spelled out rather than looped over NSSL2_SELECTOR_FIELDS: this is
    # the file that CONSUMES these five knobs, and the registry's
    # consuming-read gate (tools/check_parameter_claims.py) only counts an
    # identifier read.  A getattr loop would leave every NSSL selector
    # claiming to be implemented with nothing in woof/core reading it.
    return resolve_nssl2_mode(
        nssl_2moment_on=cfg.nssl_2moment_on,
        nssl_hail_on=cfg.nssl_hail_on,
        nssl_ccn_on=cfg.nssl_ccn_on,
        nssl_density_on=cfg.nssl_density_on,
        nssl_3moment=cfg.nssl_3moment,
    )


DEFAULT_MODE = resolve_nssl2_mode()
DEFAULT_RESTART_FIELDS = DEFAULT_MODE.transported_fields

# ---------------------------------------------------------------------------
# Deprecated WRF scheme IDs (module_check_a_mundo.F:3377-3421).  Since v4.5
# the old NSSL mp_physics values are compatibility spellings that WRF rewrites
# to option 18 plus explicit variant flags before any physics runs.  The
# mapping below is the pinned v4.6.1 oracle transcription. Stock
# v4.7.1 import uses its separately versioned mapping below. The
# resolver applies the subsequent nssl_3moment=1 number-prediction
# rewrite in native order; numerical admission is a separate check.
DEPRECATED_MP_PHYSICS_FLAGS: Mapping[int, Mapping[str, int]] = (
    MappingProxyType({
        17: MappingProxyType({
            "nssl_2moment_on": 1, "nssl_hail_on": 1,
            "nssl_ccn_on": 0, "nssl_density_on": 2}),
        19: MappingProxyType({
            "nssl_2moment_on": 0, "nssl_hail_on": 2,
            "nssl_density_on": 1}),
        21: MappingProxyType({
            "nssl_2moment_on": 0, "nssl_hail_on": 0,
            "nssl_density_on": 0}),
        22: MappingProxyType({
            "nssl_2moment_on": 1, "nssl_hail_on": 0,
            "nssl_ccn_on": 0, "nssl_density_on": 1}),
    }))


# WRF v4.7.1 f52c197ed39d12e087d02c50f412d90d418f6186,
# share/module_check_a_mundo.F:3397-3416 (WRF public domain).
# These legacy spellings now force predicted CCN even when the namelist
# explicitly requests diagnostic CCN. Keep the v4.6.1 oracle contract above
# intact: changing its fixture meaning would invalidate the native proof.
DEPRECATED_MP_PHYSICS_FLAGS_WRF471: Mapping[int, Mapping[str, int]] = (
    MappingProxyType({
        selector: MappingProxyType({
            **flags,
            **({"nssl_ccn_on": 1} if selector in (17, 22) else
               {"nssl_ccn_on": 0} if selector == 21 else {}),
        })
        for selector, flags in DEPRECATED_MP_PHYSICS_FLAGS.items()
    }))


def canonicalize_deprecated_mp_physics(
        mp_physics: int) -> tuple[int, dict[str, int]]:
    """Rewrite a deprecated NSSL scheme ID exactly as WRF v4.6.1 does.

    Returns ``(mp_physics, flags)`` where ``flags`` holds the variant
    selectors WRF's ``module_check_a_mundo.F:3377-3421`` would force.  A
    non-deprecated ID passes through with no flags.  Callers own refusing
    unported resolved modes via :func:`require_ported_nssl2_mode`.
    """
    if isinstance(mp_physics, bool) or not isinstance(mp_physics, int):
        raise TypeError(f"mp_physics must be an integer, got {mp_physics!r}")
    flags = DEPRECATED_MP_PHYSICS_FLAGS.get(mp_physics)
    if flags is None:
        return mp_physics, {}
    return MP_PHYSICS, dict(flags)


# The variant modes with a ported, executable numerical path.  Everything on
# this list runs through the one shipped option-18 implementation with its
# variant switches; nothing is substituted.  Keys are
# (two_moment, hail, predicted_ccn, density_moments, sixth_moments).
PORTED_VARIANT_MODES: tuple[tuple[bool, bool, bool, int, int], ...] = (
    (False, True, True, 1, 0),   # legacy19, Registry CCN on; physics lccn=0
    (False, True, False, 1, 0),
    (False, False, True, 0, 0),  # legacy21
    (False, False, False, 0, 0),
    (True, True, True, 2, 0),    # native default (shipped mp18 lane)
    (True, True, False, 2, 0),   # diagnostic CCN; v4.6.1 legacy mp17
    (True, False, True, 1, 0),   # mp18 with hail off
    (True, False, False, 1, 0),  # diagnostic CCN; v4.6.1 legacy mp22
)


def require_ported_nssl2_mode(mode: NSSL2Mode) -> NSSL2Mode:
    """Refuse any resolved NSSL mode without a ported numerical path."""
    if not isinstance(mode, NSSL2Mode):
        raise TypeError("mode must be NSSL2Mode")
    key = (mode.two_moment, mode.hail, mode.predicted_ccn,
           mode.density_moments, mode.sixth_moments)
    if key in PORTED_VARIANT_MODES:
        return mode
    if mode.sixth_moments:
        raise ValueError(
            "the three-moment NSSL extension (nssl_3moment=1/2) is not "
            "ported")
    if mode.density_moments == 0:
        raise ValueError(
            "nssl_density_on=0 (fixed graupel/hail density) is not "
            "plumbed; the ported two-moment modes predict graupel volume "
            "(and hail volume when hail is on)")
    raise ValueError(
        f"NSSL mode {key!r} has no ported numerical path")


# Workspace slots that a variant pins to exact zero for the whole run: the
# corresponding WRF Registry fields simply do not exist in that package
# combination, and the ported kernels' category gates (mass/number minimums)
# reproduce the absent-category behaviour bit-for-bit from all-zero moments.
def pinned_zero_fields(mode: NSSL2Mode) -> tuple[str, ...]:
    """Registry-named fields a ported variant keeps identically zero."""
    require_ported_nssl2_mode(mode)
    pinned = [name for name in DEFAULT_RESTART_FIELDS
              if name not in mode.transported_fields]
    return tuple(pinned)


def nssl2_contract_receipt(mode: NSSL2Mode, *, nssl_cccn: float = 0.5e9) -> dict[str, object]:
    """The receipt block describing ONE resolved NSSL configuration.

    Every emitter of an NSSL contract receipt calls this, so a run's
    receipt cannot describe a mode the run is not in.  ``is_default_lane``
    is stated rather than left to be inferred: it is the difference
    between the validation-candidate lane and an implemented-unverified
    variant, and a reader should not have to compare five booleans to find
    out which one produced the file.
    """
    return {
        "selector": MP_PHYSICS,
        "contract_id": CONTRACT_ID,
        "wrf_reference_version": WRF_REFERENCE_VERSION,
        "wrf_reference_commit": WRF_REFERENCE_COMMIT,
        "resolved_mode": asdict(mode),
        "is_default_lane": mode == DEFAULT_MODE and nssl_cccn == 0.5e9,
        "transported_fields": list(mode.transported_fields),
        "absent_fields": list(pinned_zero_fields(mode)),
        "wrf_namelist_defaults": dict(WRF_NAMELIST_DEFAULTS),
        **({"coefficient_overrides": {"nssl_cccn": float(nssl_cccn)}}
           if nssl_cccn != 0.5e9 else {}),
    }


__all__ = [
    "nssl2_background_ccn_per_kg",
    "nssl2_cccn_concentration",
    "BASE_MASS_FIELDS",
    "nssl2_contract_receipt",
    "CONTRACT_ID",
    "DEFAULT_MODE",
    "DEFAULT_RESTART_FIELDS",
    "DEPRECATED_MP_PHYSICS_FLAGS",
    "DEPRECATED_MP_PHYSICS_FLAGS_WRF471",
    "PORTED_VARIANT_MODES",
    "canonicalize_deprecated_mp_physics",
    "pinned_zero_fields",
    "require_ported_nssl2_mode",
    "GRAUPEL_VOLUME_FIELD",
    "HAIL_DIAGNOSTIC_FIELDS",
    "HAIL_MASS_FIELD",
    "HAIL_NUMBER_FIELD",
    "HAIL_VOLUME_FIELD",
    "MP_PHYSICS",
    "NSSL2Mode",
    "NSSL2_SELECTOR_FIELDS",
    "resolve_nssl2_mode_for_config",
    "PRECIPITATION_FIELDS",
    "PREDICTED_CCN_FIELD",
    "RADIATION_EFFECTIVE_RADIUS_FIELDS",
    "REFLECTIVITY_FIELDS",
    "SIXTH_MOMENT_FIELDS",
    "TWO_MOMENT_NUMBER_FIELDS",
    "WRF_NAMELIST_DEFAULTS",
    "WRF_REFERENCE_COMMIT",
    "WRF_REFERENCE_VERSION",
    "resolve_nssl2_mode",
]
