"""MMM-physics generation for the hex MPAS routes: YSU PBL and the revised MM5 surface layer.

Defaults are the latest released upstream scheme versions; pins are for bitwise proof (project policy,
2026-11-08). The official single-source repository is NCAR/MMM-physics (BSD-3-Clause, (c) 2022 NCAR):

- ``550b5b4`` (NCAR/MMM-physics 550b5b494758, 2026-05-26): the commit WRF v4.8.0 (tag 06d4240ae989,
  2026-06-08) ships as phys/physics_mmm. A scheme commit shipped inside a model release counts as
  released (2026-11-08). Against a4baf7f it carries, for these two schemes:
    * bl_ysu.F90:855, the top-down (stratocumulus) entrainment efficiency
      ``ent_eff = min(0.2*8*tmp1, 0.4)`` (was ``0.2*8*tmp1 + 0.2``; Zhang et al. 2024): no 0.2 floor,
      capped at 0.4. It acts only where cloud-topped mixing runs, i.e. config_ysu_pblmix=true.
    * sf_sfclayrev.F90:707-720,722-725,779-783, the KIM turbulent orographic form drag option
      (Koo et al. 2018; Koo and Hong 2025; MMM PR #15, e9065ee). Upstream default off (WRF Registry
      kim_tofd=.false., tofd_factor=.003); with it off the scheme is the a4baf7f arithmetic word for
      word (cd/cda gain +0.0 on a square, the ust branch is the old expression).
- ``a4baf7f`` (NCAR/MMM-physics tag 20250616-MPASv8.3, a4baf7f3243d): the Externals pin of MPAS-A
  v8.4.2 and RRFS-MPAS 44c0849, which the bitwise claims and the landing gate are proven against.

Selection (engine extension; these keys never enter the native MPAS Registry):
``--physics mmm_version=550b5b4|a4baf7f`` (aliases ``latest`` and ``pinned``), ``--physics kim_tofd=true``
and ``--physics tofd_factor=0.003``. A run that names no version gets 550b5b4 in correct mode (the forecast
door's default) and a4baf7f in exact mode, whose definition is native MPAS v8.4.2 bit for bit.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

LATEST = "550b5b4"
PINNED = "a4baf7f"
VERSIONS = (LATEST, PINNED)
ALIASES = {"latest": LATEST, "pinned": PINNED, "pin": PINNED,
           "550b5b494758": LATEST, "a4baf7f3243d": PINNED,
           "wrf_480": LATEST, "v842": PINNED, "20250616-mpasv8.3": PINNED}
#: WRF v4.8.0 Registry/Registry.EM_COMMON: kim_tofd default .false., tofd_factor default .003
TOFD_FACTOR_DEFAULT = 0.003
KEYS = ("mmm_version", "kim_tofd", "tofd_factor")
SCHEMES_WITH_TOFD = ("sf_monin_obukhov_rev",)


def normalise(value: Any) -> str:
    text = str(value).strip().strip("\"'").casefold()
    version = ALIASES.get(text, text)
    if version not in VERSIONS:
        raise ValueError(f"mmm_version must be one of {VERSIONS} (or latest/pinned), not {value!r}")
    return version


def _logical(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().strip("\"'").casefold().strip(".")
    if text in ("true", "t", "1", "on", "yes"):
        return True
    if text in ("false", "f", "0", "off", "no"):
        return False
    raise ValueError(f"kim_tofd must be true or false, not {value!r}")


def default_version(mode: str) -> str:
    """The generation a run that names none gets: the latest in correct mode, the pin in exact mode."""
    return PINNED if mode == "exact" else LATEST


def resolve(explicit: Mapping[str, Any] | None, *, mode: str) -> Mapping[str, Any]:
    """The run's MMM-physics choice: version, KIM form drag and its factor (always all three keys)."""
    explicit = dict(explicit or {})
    unknown = sorted(set(explicit) - set(KEYS))
    if unknown:
        raise ValueError(f"unknown MMM-physics options {unknown}; known: {list(KEYS)}")
    version = normalise(explicit["mmm_version"]) if "mmm_version" in explicit else default_version(mode)
    tofd = _logical(explicit.get("kim_tofd", False))
    factor = float(explicit.get("tofd_factor", TOFD_FACTOR_DEFAULT))
    if not factor > 0.0:
        raise ValueError("tofd_factor must be positive (it scales the orographic standard deviation into zf)")
    if tofd and version != LATEST:
        # THE BREAKAGE THIS PREVENTS: MMM a4baf7f has no form-drag term; the run would silently carry none.
        raise ValueError(f"kim_tofd needs mmm_version={LATEST} (WRF v4.8.0's MMM-physics); "
                         f"MMM {version} has no orographic form drag")
    return MappingProxyType({"mmm_version": version, "kim_tofd": tofd, "tofd_factor": factor})


PINNED_OPTIONS = resolve({}, mode="exact")


def of(call: Any) -> Mapping[str, Any]:
    """The MMM choice a SlotCall carries; a call built without one (module replays) runs the pin."""
    value = getattr(call, "mmm", None)
    return PINNED_OPTIONS if not value else value


#: The slot/scheme pairs the MMM-physics generation reaches on the MPAS routes.
MMM_SLOTS = {"pbl": ("bl_ysu",), "surface_layer": ("sf_monin_obukhov_rev", "sf_monin_obukhov")}


def configure(config: Any, explicit: Mapping[str, Any] | None) -> Any:
    """The route's DriverConfig with its MMM choice laid on (``config.mmm``).

    A slot that runs MPAS's own Fortran (native-cpu) runs the library's a4baf7f source, so a run that names no
    version there gets the pin; one that names 550b5b4 (or kim_tofd) there is refused."""
    from dataclasses import replace
    explicit = dict(explicit or {})
    choice = resolve(explicit, mode=getattr(config, "mode", "exact"))
    active = getattr(config, "active", lambda slot: True)
    reached = [(slot, config.selectors.get(slot)) for slot, schemes in MMM_SLOTS.items()
               if config.selectors.get(slot) in schemes and active(slot)]
    native = [(slot, scheme) for slot, scheme in reached if str(config.backends.get(slot)) != "hex-cuda"]
    if not reached and "mmm_version" not in explicit and not choice["kim_tofd"]:
        # No YSU or revised-MM5 slot is selected and active, so the generation changes nothing: the run records the
        # pin (no "mmm" receipt entry), as the MYNN suite version does.  THE BREAKAGE THIS PREVENTS: every
        # convection_permitting receipt named MMM 550b5b4, a scheme generation the run never executes.
        choice = resolve({**explicit, "mmm_version": PINNED}, mode="exact")
    if native and choice["mmm_version"] != PINNED:
        if "mmm_version" in explicit or choice["kim_tofd"]:
            # THE BREAKAGE THIS PREVENTS: the native library is MMM a4baf7f; the run would carry the pin
            # under the 550b5b4 label.
            raise ValueError(f"mmm_version={choice['mmm_version']} runs on the GPU port only; "
                             f"{', '.join(f'{s}={n}' for s, n in native)} runs native-cpu (MMM {PINNED})")
        choice = resolve({**explicit, "mmm_version": PINNED}, mode="exact")
    if choice["kim_tofd"]:
        if config.selectors.get("surface_layer") not in SCHEMES_WITH_TOFD or not active("surface_layer"):
            # THE BREAKAGE THIS PREVENTS: kim_tofd on any other surface layer would run with no form drag.
            raise ValueError("kim_tofd applies to config_sfclayer_scheme='sf_monin_obukhov_rev' only")
    return replace(config, mmm=choice)
