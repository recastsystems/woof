"""Orographic gravity-wave-drag generation for the hex MPAS routes: KIM (MPAS PR #1409) or YSU (v8.4.2).

Defaults are the latest released upstream scheme versions; pins are for bitwise proof (project policy,
2026-11-08):

- ``kim2026``: NCAR/MMM-physics bl_gwdo.F90 at tag 20260319-WRFv4.8.0 (63c806d), the KIM drag WRF v4.8.0
  ships as gwd_opt=1, with the MPAS-Model PR #1409 driver slice (selector ``bl_kim_gwdo``, knobs
  ``config_gwdo_factor`` default 2.0 and ``config_gwdo_nonhyd`` default true) and the MPAS-Model PR #1408
  orographic statistics (``elvmax`` plus the revised var2d/con/oa/ol), built per mesh by
  ``rw_mpas_static --gwd-only --gwd-recipe kim2026`` (woof.hex.kim_gwd_static) and read over the init's.
- ``v842``: MPAS-A v8.4.2 ``bl_ysu_gwdo`` (woof.hex.mpasphys.ysu_gwdo_v842) on the init's own v8.4.2
  statistics, the pin the bitwise claims and the landing gate are proven against.

Selection (engine extension; ``gwdo_version`` never enters the native Registry):
``--physics gwdo_version=kim2026|v842`` (aliases ``latest``/``pinned``), ``--physics config_gwdo_factor=X``
and ``--physics config_gwdo_nonhyd=false``. A run that names no version gets v842 in every mode. kim2026 did not
pass the default-flip check (the 2026-11-08 default-flip rule: obs-skill beside the pin, sheets looked at): on hex 60 km
x1.163842, GFS 2023-03-31 12Z, 120 h, scored against the GFS analysis at the same 20 valid times, it was worse than
v842 at every lead in NH 500 hPa height (68.7 vs 52.4 m RMSE at +120 h), NH MSLP and 10 m wind over terrain
> 1000 m (speed bias +1.30 vs +0.80 m/s, larger by day), at the same step time (1537.6 vs 1536.6 s, same card);
evidence the upstream-port verification record. kim2026 stays selectable by name. The
version maps the suite's YSU drag only: ``config_gwdo_scheme='bl_ysu_gwdo'`` named explicitly keeps the
v8.4.2 kernel (an explicit selector is never rewritten), ``'bl_kim_gwdo'`` named explicitly runs KIM in
either mode, and the GSL drag (``bl_ugwp_gwdo``, hrrrv5 and the RRFS suites) is a different scheme that
the version does not touch.
"""
from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Any, Mapping

LATEST = "kim2026"
PINNED = "v842"
VERSIONS = (LATEST, PINNED)
ALIASES = {"latest": LATEST, "pinned": PINNED, "pin": PINNED, "kim": LATEST, "kim2026": LATEST,
           "wrf_480": LATEST, "20260319-wrfv4.8.0": LATEST, "63c806d": LATEST,
           "v842": PINNED, "v8.4.2": PINNED, "ysu": PINNED}
KIM = "bl_kim_gwdo"
YSU = "bl_ysu_gwdo"
#: MPAS PR #1409 Registry defaults; WRF v4.8.0 Registry gwd_dx_factor = 2., gwd_if_nonhyd = .true.
DEFAULT_FACTOR = 2.0
DEFAULT_NONHYD = True
KNOBS = ("config_gwdo_factor", "config_gwdo_nonhyd")
KEYS = ("gwdo_version", *KNOBS)


def normalise(value: Any) -> str:
    text = str(value).strip().strip("\"'").casefold()
    version = ALIASES.get(text, text)
    if version not in VERSIONS:
        raise ValueError(f"gwdo_version must be one of {VERSIONS} (or latest/pinned), not {value!r}")
    return version


def _logical(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().strip("\"'").casefold().strip(".")
    if text in ("true", "t", "1", "on", "yes"):
        return True
    if text in ("false", "f", "0", "off", "no"):
        return False
    raise ValueError(f"config_gwdo_nonhyd must be true or false, not {value!r}")


def default_version(mode: str) -> str:
    """The generation a run that names none gets: the pin in every mode (kim2026 failed the obs-skill check
    beside v842, module docstring); ``mode`` is kept so the door predicates stay one call."""
    del mode
    return PINNED


def _named_selector(text: str | None) -> str | None:
    """``config_gwdo_scheme`` as the run's namelist names it, or None when it is left to the suite."""
    if not text or not text.strip():
        return None
    from .driver_v842 import parse_fortran_namelist
    for options in parse_fortran_namelist(text).values():
        for key, value in options.items():
            if key.casefold() == "config_gwdo_scheme":
                value = str(value).strip()
                return None if value == "suite" else value
    return None


def configure(config: Any, explicit: Mapping[str, Any] | None, text: str | None = None) -> Any:
    """The route's DriverConfig with its drag generation laid on.

    kim2026 rewrites the suite's ``bl_ysu_gwdo`` to ``bl_kim_gwdo`` (selector and option) and carries the two
    PR #1409 knobs in ``config.gwdo``; v842 (and every run whose drag is not the suite's YSU) is returned
    unchanged, so a pinned run is word for word what it was before the knob existed."""
    explicit = dict(explicit or {})
    unknown = sorted(set(explicit) - set(KEYS))
    if unknown:
        raise ValueError(f"unknown orographic-drag options {unknown}; known: {list(KEYS)}")
    if config is None:
        if explicit:
            raise ValueError(f"{sorted(explicit)} apply to the MPAS-physics route (--physics-suite); this run has none")
        return config
    mode = getattr(config, "mode", "exact")
    version = normalise(explicit["gwdo_version"]) if "gwdo_version" in explicit else default_version(mode)
    selector = config.selectors.get("gwdo")
    named = _named_selector(text)
    if selector == YSU and version == LATEST and named != YSU:
        backend = str(config.backends.get("gwdo"))
        if backend != "hex-cuda":
            if "gwdo_version" in explicit:
                # THE BREAKAGE THIS PREVENTS: native MPAS v8.4.2 has no KIM drag; the run would carry YSU
                # under the kim2026 label.
                raise ValueError("gwdo_version=kim2026 runs on the GPU port only; config_gwdo_backend="
                                 f"{backend} runs MPAS v8.4.2's own YSU drag")
            version = PINNED
        else:
            selectors = dict(config.selectors, gwdo=KIM)
            options = dict(config.options, config_gwdo_scheme=KIM)
            config = replace(config, selectors=MappingProxyType(selectors), options=MappingProxyType(options))
            selector = KIM
    if selector != KIM:
        if any(key in explicit for key in KNOBS) or ("gwdo_version" in explicit and version == LATEST
                                                     and named == YSU):
            # THE BREAKAGE THIS PREVENTS: the knobs (and kim2026) would be accepted and never read.
            raise ValueError(f"{sorted(explicit)} select or tune the KIM drag (config_gwdo_scheme='{KIM}'); "
                             f"this run's drag is config_gwdo_scheme={selector!r}")
        return config
    if str(config.backends.get("gwdo")) != "hex-cuda":
        raise ValueError(f"config_gwdo_scheme='{KIM}' runs on the GPU port only (woof.hex.mpasphys.kim_gwdo_2026); "
                         "native MPAS v8.4.2 has no KIM drag")
    if "gwdo_version" in explicit and version == PINNED:
        raise ValueError(f"gwdo_version=v842 runs MPAS v8.4.2's {YSU}; config_gwdo_scheme='{KIM}' names the KIM drag")
    factor = float(explicit.get("config_gwdo_factor", DEFAULT_FACTOR))
    if not factor > 0.0:
        raise ValueError("config_gwdo_factor must be positive: it divides the base stress (bl_gwdo.F90:478)")
    nonhyd = _logical(explicit.get("config_gwdo_nonhyd", DEFAULT_NONHYD))
    choice = MappingProxyType({"gwdo_version": LATEST, "config_gwdo_factor": factor,
                               "config_gwdo_nonhyd": nonhyd})
    options = dict(config.options, config_gwdo_scheme=KIM, config_gwdo_factor=factor, config_gwdo_nonhyd=nonhyd)
    return replace(config, options=MappingProxyType(options), gwdo=choice)


def selected(config: Any) -> bool:
    """True when the run's drag is the KIM generation (it reads the PR #1408 statistics)."""
    return config is not None and getattr(config, "selectors", {}).get("gwdo") == KIM and (
        config.active("gwdo") if hasattr(config, "active") else True)
