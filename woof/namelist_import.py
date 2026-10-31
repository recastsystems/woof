"""WRF namelist importer: namelist.wps + namelist.input -> experiment TOML.

Phase-5 Task 1 (recon G10, panel lane L1).  ``import_namelists`` reads both
WRF namelists into the typed experiment schema (woof/experiment.py),
converts WRF's STAGGERED ``e_we``/``e_sn``/``e_vert`` to woof mass
dimensions explicitly (one fewer point: the bundle's 251 x 201 x 50
namelist domain is woof's 250 x 200 x 49), maps only implemented physics
schemes, and emits the resolved TOML text plus a structured
:class:`SubstitutionReport` -- NEVER silent: every namelist key is either
translated, recorded as a ratified substitution (mp 55 -> 10 Morrison,
RRTMG -> RTE+RRTMGP; bl_pbl 11 imports natively since the Shin-Hong
port), recorded as consumed-without-counterpart, or a hard error.

Child ``dx``/``dy`` namelist entries (the bundle hand-types
``333.333333`` for d04) are cross-checked against the exact parent-chain
rational and then DROPPED from the output: the resolved TOML never
hand-types child dx/dt -- woof/experiment.py derives them exactly
(dx 12000 -> 3000 -> 1000 -> 1000/3 m; dt 60 -> 15 -> 5 -> 5/3 s for the
bundle's 1,4,3,3 ratios).  The namelist chain is authoritative; the
"500 m d04" prose value is a pinned hard error, not a rounding choice.

Omitted namelist keys resolve to their WRF v4.6.1 Registry defaults
(review F2), so an implicitly two-way namelist (omitted ``feedback`` =>
Registry default 1) enables the plainly labelled experimental feedback
path instead of silently importing as one-way; ``km_opt`` (Registry default -1 =
must-set) is a hard error when omitted.  Values woof supplies WITHOUT a
Registry source (the ``ztop`` scaffold; ``time_step_sound`` auto -> 4)
are recorded as :class:`AppliedDefault` entries.  Registry defaults that
bind for the reference run and differ from woof's frozen RunConfig
defaults are emitted explicitly (Phase-4 ratified native-dt baseline):
``emdiv = 0.01``, ``hypsometric_opt = 2``, ``h_sca_adv_order = 5``.

Knob-parity contract (product/knobs lane): every namelist key the model
GENUINELY HONORS maps to its experiment-TOML counterpart, and the
report classifies every consumed key into exactly one of three explicit
sections -- TRANSLATED (a TOML value came out of it, including the
ratified substitutions), FIXED BY ARWEN (the key was validated against
the single implemented value and recorded with that value and why), or
NOT IMPLEMENTED (consumed without a counterpart, each with a reason).
RunConfig-honored keys the importer previously refused as unmapped now
translate: the per-domain turbulence row ``c_s``, ``c_k``,
``mix_isotropic``, ``mix_upper_bound``, ``tke_upper_bound``,
``tke_heat_flux``, ``tke_drag_coefficient`` (every one ``max_domains`` in
the Registry and admitted per domain by
``woof.experiment._DOMAIN_RUN_OVERRIDES``, so a PBL parent may carry a
PBL-off LES child that names its own closure constants), the per-domain
moist-filter switch ``moist_mix6_off`` (Registry.EM_COMMON:2889,
divergence-ledger entry L4), plus
``diff_6th_thresh``, ``no_mp_heating``,
``mp_tend_lim``, ``ysu_topdown_pblmix``, ``isftcflx``, ``iz0tlnd``,
``usemonalb``, ``rdlai2d``, ``opt_thcnd`` (each emitted only when the
namelist supplies it -- their Registry defaults equal woof's frozen
RunConfig defaults, so the established imports stay byte-identical).
Keys the core pins where WRF has options are validated against the
pinned value and refused otherwise (``rk_ord = 3``,
``h_mom_adv_order = 5``, ``v_mom_adv_order = v_sca_adv_order = 3``,
``momentum_adv_opt = 1``, ``use_mp_re = 1``, the
MYNN/Noah-MP/RUC option identities, supported &stoch controls, no FDDA
nudging) -- never silently reinterpreted.

Scheme-generation hazard (closed 2026-08-30): WRF v4.8.0 rebinds
``cu_physics = 3`` to Grell-Freitas-Li while adding ZERO namelist options
and changing ZERO defaults, so a v4.8.0 namelist is byte-indistinguishable
from a v4.6.1 one at the GF option level and no importer can detect which
generation was meant.  Every GF import therefore records
:data:`GF_SCHEME_GENERATION_NOTICE` in the report (rendered by ``format``,
printed by ``woof import-namelist``); the one namelist spelling that DOES
reveal spreading-generation intent, ``cugd_avedx`` != 1 beside
``cu_physics = 3``, is a hard refusal naming the wrong-scheme breakage.
"""

from __future__ import annotations

import contextvars
import math
import re
import tomllib
from dataclasses import dataclass, replace as _dataclass_replace
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from numbers import Real
from typing import Mapping

from woof.config import EPSSM_AUTO, advection_order_refusal
from woof.experiment import _reject_moving_nest_keys, build_experiment
from woof.fortran_namelist import parse_namelist, parse_namelist_text
from woof.static.projection import WRF_MAP_PROJ_CODES
from woof.physics_compat import (
    RRTMG_VARIANT_LEGACY,
    RRTMG_VARIANT_RTE_RRTMGP,
    WRF_RRTMG_COMPATIBILITY_TOKENS,
    WRF_RRTMG_LEGACY,
    WRF_RRTMG_TO_RTE_RRTMGP,
    require_ready_wrf_physics,
)

# Omission is distinct from the historical explicit None (WRF legacy).
_AUTO_RRTMG_VARIANT = object()

#: Relative tolerance for the namelist child-dx cross-check (matches
#: woof.experiment._REL_TOL): the bundle's truncated ``333.333333`` vs
#: the exact 1000/3 m passes (~1e-9); a hand-typed 500 m fails hard.
_REL_TOL = 1.0e-6


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------

#: The &fdda selectors that turn nudging ON.  Any nonzero entry is a
#: request to nudge.
_ACTIVE_NUDGING_SELECTORS = ("grid_fdda", "grid_sfdda", "obs_nudge_opt")

FDDA_OBSERVATION_INPUT = (
    "surface and point-observation nudging require observation forcing; "
    "the gridded analysis source does not supply those inputs")

#: The values of ``&dynamics/use_theta_m`` WRF defines.
THETA_M_ADMITTED = (0, 1)

#: WRF's Registry default for an OMITTED ``&dynamics/use_theta_m``, by the
#: WRF line the namelist was written for.  V4.0 flipped it: the V3.9
#: Registry (NOAA-EMC/HRRR v4.1.21, sorc/hrrr_wrfarw.fd/WRFV3.9/Registry/
#: Registry.EM_COMMON:2633) reads ``rconfig integer use_theta_m
#: namelist,dynamics 1 0``, dry theta; the bundled V4.7.1 registry
#: (woof/data/wrf_namelist/registry_v4.7.1.json) reads 1, moist theta.
#: Operational HRRR omits the key and runs the V3.9 line, so an HRRR
#: namelist imports with ``wrf_version="3"``; the default stays the V4
#: reading every earlier import used.
OMITTED_USE_THETA_M_BY_WRF_VERSION = {"3": 0, "4": 1}
WRF_VERSION_DEFAULT = "4"

#: Where each line's Registry states that default, for the "Namelist
#: defaults" row an omitted key books.  The V4 row is the copy in this tree
#: (tools/lake_wrf461_oracle/upstream/Registry.EM_COMMON:2860); the V3 row
#: is the operational fork's file fetched at its tag (sha256 8eed8499de58
#: 3e7d11a28fe4c5faba0ea2720b9992e8bd750ef7595ad9a1e552, 3350 lines).
OMITTED_USE_THETA_M_SOURCE = {
    "3": "WRFV3.9 Registry/Registry.EM_COMMON:2633 default "
         "(NOAA-EMC/HRRR v4.1.21), dry theta",
    "4": "WRF v4.6.1 Registry/Registry.EM_COMMON:2860 default (the "
         "bundled v4.7.1 registry agrees), moist theta",
}

#: What a caller that did not say how it chose the WRF line is reported as
#: having done: nothing chose it, the importer's default did.
WRF_VERSION_SOURCE_DEFAULT = (
    "the importer's default (pass --wrf-version 3 for a namelist written "
    "for WRF 3.x)")


def wrf_line_of(wrf_version) -> str:
    """The WRF line ("3" or "4") a version spelling names."""
    major = str(wrf_version).strip().split(".")[0]
    if major not in OMITTED_USE_THETA_M_BY_WRF_VERSION:
        raise ValueError(
            f"wrf_version must name WRF line 3 or 4, got {wrf_version!r}")
    return major


def omitted_use_theta_m(wrf_version) -> int:
    """The Registry default an omitted ``use_theta_m`` takes.

    ``wrf_version`` is the WRF line, "3" or "4" (a dotted spelling such as
    "3.9" or "4.7.1" names its line by its first component).
    """
    return OMITTED_USE_THETA_M_BY_WRF_VERSION[wrf_line_of(wrf_version)]


#: The engine generation selectors whose stock-WRF code changed in a V4
#: release, as ``(selector, first release that runs the engine default,
#: generation every earlier V4 release runs)``.  The engine defaults are
#: the newest generations (woof.core.solar DEFAULT_SOLAR_GENERATION and
#: woof.ysu_generations YSU_DEFAULT_GENERATION are WRF 4.8's; Noah-MP
#: snow aging is WRF 4.7's; the urban water balance WRF 4.8's).  The same
#: four rows are what a stock WRF v4.6.1 recording replays with
#: (woof.verify_exact.replay.WRF_461_REFERENCE_GENERATIONS).  Adding a
#: release that changes a scheme is one row here.  Origin: the OU
#: Oman-UAE case (2026-10-28, run B) imported a WRF 4.6 namelist and ran
#: WRF 4.8's YSU entrainment cap and solar constants, because the
#: importer only knew the line ("4").
WRF_RELEASE_GENERATIONS = (
    ("solar_generation", (4, 8), "wrf_461"),
    ("ysu_version", (4, 8), "wrf_461"),
    ("urban_water_balance", (4, 8), "wrf_471"),
    ("noahmp_snow_age", (4, 7), "wrf_461"),
)


def wrf_release_of(wrf_version) -> tuple[int, ...] | None:
    """``(major, minor[, patch])`` of a dotted WRF version, else ``None``.

    A bare line ("4") names no release: the importer cannot know which V4
    code the namelist was written for and keeps the engine defaults.
    """
    parts = str(wrf_version).strip().lstrip("vV").split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:3]):
        return None
    return tuple(int(part) for part in parts[:3])


def wrf_version_argument(text: str) -> str:
    """argparse type for ``--wrf-version``: a line (3, 4) or a release."""
    import argparse
    value = str(text).strip()
    try:
        wrf_line_of(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    if "." in value and wrf_release_of(value) is None:
        raise argparse.ArgumentTypeError(
            f"--wrf-version {value!r}: name a line (3, 4) or a release "
            "such as 4.6.1")
    return value


def wrf_release_generations(wrf_version) -> dict[str, str]:
    """The generation selectors a named WRF V4 release runs.

    Empty for a bare line or a release at or after every row's change
    (the engine defaults are then that release's code already).
    """
    release = wrf_release_of(wrf_version)
    if release is None or release[0] != 4:
        return {}
    return {key: generation for key, since, generation
            in WRF_RELEASE_GENERATIONS if release[:2] < since}

#: How each import route recovers the dry-theta state it integrates.  The
#: moist theta_m prognostic is implemented nowhere in the engine and the
#: initial and boundary state is recovered exactly on all three routes,
#: so the answer is one announced substitution rather than a refusal on
#: whichever door the user happened to arrive through.
_THETA_M_ROUTES = {
    "metgrid": ("metgrid TT is physical temperature, so the initial and "
                "boundary state is recovered exactly"),
    "wrf_boundary": ("the moist wrfbdy THM/QV/MU are converted exactly at "
                     "each forcing time into the dry-theta state"),
    "native": ("woof's native initialization constructs the dry-theta "
               "state from physical temperature (woof/ingest/real.py:1724), "
               "so the initial and boundary state is recovered exactly"),
}


def active_nudging_selectors(fdda: dict) -> list[tuple[str, list]]:
    """Every &fdda selector in ``fdda`` that asks for nudging.

    One predicate, called by the importer and by the support report, so
    the two doors cannot answer one &fdda block differently.
    """

    active: list[tuple[str, list]] = []
    for key in _ACTIVE_NUDGING_SELECTORS:
        values = fdda.get(key)
        if values is None:
            continue
        try:
            requested = any(int(value) != 0 for value in values)
        except (TypeError, ValueError):
            # An unreadable selector is a nudging request nobody can
            # price; it is named here rather than escaping as a
            # traceback out of int().
            requested = True
        if requested:
            active.append((key, list(values)))
    return active


@dataclass(frozen=True)
class ThetaMDecision:
    """The single answer every door gives about ``use_theta_m``.

    ``moist_theta`` is true when WRF would integrate the moist theta_m
    prognostic and WOOF integrates dry theta instead -- a DECLARED
    DIVERGENCE with ``reason``, never a refusal: the divergence is the
    same on every door, so refusing on one of them and announcing it on
    the others gave one namelist two answers.
    """

    moist_theta: bool
    route: str
    reason: str


def theta_m_decision(use_theta_m: int, *,
                     metgrid_initialization: bool = False,
                     wrf_boundary_use_theta_m: int | None = None
                     ) -> ThetaMDecision:
    """Book ``&dynamics/use_theta_m`` for whichever door is asking."""

    if metgrid_initialization:
        route = "metgrid"
    elif wrf_boundary_use_theta_m is not None:
        route = "wrf_boundary"
    else:
        route = "native"
    if int(use_theta_m) == 0:
        return ThetaMDecision(
            moist_theta=False, route=route,
            reason=("metgrid TT is physical temperature; native "
                    "initialization constructs the shared dry-theta state "
                    "and boundaries" if route == "metgrid" else
                    "woof transcribes the non-moist-theta use_theta_m=0 "
                    "branch"))
    return ThetaMDecision(
        moist_theta=True, route=route,
        reason=("WRF would integrate moist theta; WOOF integrates dry theta "
                "and the moist-theta branch is not implemented.  "
                + _THETA_M_ROUTES[route]
                + "; the integration itself differs from a use_theta_m = 1 "
                "WRF run.  Set use_theta_m = 0 in the producing namelist to "
                "run WRF on the same variable."))


#: ``&time_control/fine_input_stream`` selects which input stream
#: initializes each nest.  WRF defines exactly two values, and only two
#: (Registry.EM_COMMON `rconfig integer fine_input_stream`, consumed at
#: share/mediation_integrate.F:758-766): 0 takes every field from the
#: nest's own input, and 2 takes only the static and masked
#: land-surface fields from it and interpolates the rest from the
#: parent, which is WRF's delayed-nest-start pattern.
FINE_INPUT_STREAM_OWN_INPUT = 0
FINE_INPUT_STREAM_DELAYED_NEST = 2
FINE_INPUT_STREAM_ADMITTED = (FINE_INPUT_STREAM_OWN_INPUT,
                              FINE_INPUT_STREAM_DELAYED_NEST)

#: The way out of both fine_input_stream answers, in the one place both
#: doors read it from.
FINE_INPUT_STREAM_WAY_OUT = (
    f"Set fine_input_stream to {FINE_INPUT_STREAM_OWN_INPUT} or "
    f"{FINE_INPUT_STREAM_DELAYED_NEST} on every domain.")
FINE_INPUT_STREAM_DELAYED_WAY_OUT = (
    "Nothing to change: the delayed child starts at its declared start "
    f"time. Set fine_input_stream = {FINE_INPUT_STREAM_OWN_INPUT} to take "
    "every field from the child's own input instead.")

#: The adaptive clock's &domains keys, as (key, WRF Registry default,
#: cast), stated once for the importer that reads them and the HRRR route
#: writer (:func:`woof.hrrr_route_inputs.render_namelist_input`) that
#: spells them, so the two cannot drift.  Registry.EM_COMMON:2269-2281
#: declares the SCALARS scope 1, one value for the run; the COLUMNS are
#: max_domains, and woof carries them per domain
#: (``woof.experiment._DOMAIN_RUN_OVERRIDES``).
ADAPTIVE_CLOCK_SCALARS = (
    ("use_adaptive_time_step", False, bool),
    ("step_to_output_time", True, bool),
    ("adaptation_domain", 1, int),
)
ADAPTIVE_CLOCK_COLUMNS = (
    ("target_cfl", 1.2, float),
    ("target_hcfl", 0.84, float),
    ("max_step_increase_pct", 5, int),
    ("starting_time_step", -1, int),
    ("starting_time_step_den", 0, int),
    ("max_time_step", -1, int),
    ("max_time_step_den", 0, int),
    ("min_time_step", -1, int),
    ("min_time_step_den", 0, int),
)


@dataclass(frozen=True)
class FineInputStreamDecision:
    """The single answer every door gives about ``fine_input_stream``.

    ``refusal`` is set only for an index WRF does not define.  A declared
    ``2`` is a DECLARED DIVERGENCE carried in ``divergence``, never a
    refusal: the delayed child starts at its declared start time on both
    prepared routes, and only the provenance of its masked surface state
    differs.  The support report used to PASS the pair while the importer
    raised an unmapped-key ValueError on it, which is one namelist with
    two answers.
    """

    streams: tuple[int, ...]
    undefined: tuple[int, ...]
    delayed_domains: tuple[int, ...]
    refusal: str | None
    divergence: str | None
    way_out: str


def fine_input_stream_decision(streams) -> FineInputStreamDecision:
    """Book one ``fine_input_stream`` column for whichever door is asking.

    ``streams`` is the per-domain column in d01..dNN order.  A value that
    is not a Fortran integer token raises here rather than being coerced,
    so no door reads a different column from the one the namelist wrote.
    """

    column = tuple(streams)
    for value in column:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "fine_input_stream must contain Fortran integer tokens in "
                f"d01..dNN order, got {value!r}. "
                + FINE_INPUT_STREAM_WAY_OUT)
    undefined = tuple(sorted({value for value in column
                              if value not in FINE_INPUT_STREAM_ADMITTED}))
    delayed = tuple(index + 1 for index, value in enumerate(column)
                    if value == FINE_INPUT_STREAM_DELAYED_NEST)
    if undefined:
        return FineInputStreamDecision(
            streams=column, undefined=undefined, delayed_domains=delayed,
            refusal=(
                f"fine_input_stream={list(column)} declares "
                f"{list(undefined)}; WRF defines two values for this key: "
                f"{FINE_INPUT_STREAM_OWN_INPUT} (every field from the "
                f"nest's own input) and {FINE_INPUT_STREAM_DELAYED_NEST} "
                "(only the static and masked land-surface fields from it, "
                "the rest interpolated from the parent)."),
            divergence=None, way_out=FINE_INPUT_STREAM_WAY_OUT)
    if delayed:
        return FineInputStreamDecision(
            streams=column, undefined=(), delayed_domains=delayed,
            refusal=None,
            divergence=(
                f"fine_input_stream={list(column)}: domains {list(delayed)} "
                "take the delayed-nest-start route. Both prepared routes "
                "satisfy it -- the stock export writes wrfinput_d0N for "
                "every domain at that domain's configured start time, and "
                "the woof runtime initializes a delayed child from its "
                "own analysis at activation. The one difference from stock "
                "WRF: the masked surface state comes from the child's "
                "own-grid analysis rather than from a real.exe wrfinput."),
            way_out=FINE_INPUT_STREAM_DELAYED_WAY_OUT)
    return FineInputStreamDecision(
        streams=column, undefined=(), delayed_domains=(), refusal=None,
        divergence=None, way_out=FINE_INPUT_STREAM_WAY_OUT)


@dataclass(frozen=True)
class Substitution:
    """One ratified physics substitution applied by the importer.

    ``reason`` is set on a DECLARED DIVERGENCE: the WRF value is admitted
    and WOOF integrates something else, for a stated reason the user must
    see (the terminal prints it, the receipt carries it).  A substitution
    without a reason is a package replacement, which the WRF doors refuse
    (:func:`woof.wrfinput_door.require_preserved_wrf_selectors`).
    """

    key: str            # WRF namelist key
    wrf_value: object
    wrf_name: str       # WRF scheme name
    gpuwm_key: str      # resolved TOML key
    gpuwm_value: object
    gpuwm_name: str     # woof scheme name
    reason: str | None = None


@dataclass(frozen=True)
class DroppedKey:
    """A namelist key consumed without a woof TOML counterpart."""

    section: str
    key: str
    values: tuple
    reason: str


@dataclass(frozen=True)
class FixedKey:
    """A namelist key validated against the single implemented value.

    The key never reaches the emitted TOML because WOOF has exactly one
    implemented behavior for it; the importer checked the supplied value
    against that pin and records the pin and its evidence here.  Any
    other value is a hard error, never a silent reinterpretation --
    except where the key reaches no woof code at any value, which the
    reason says in those words, because there is no breakage to refuse.
    """

    section: str
    key: str
    values: tuple
    fixed_value: object
    reason: str


@dataclass(frozen=True)
class TranslatedKey:
    """A parsed control outside the explicit fixed/dropped decision buckets.

    The compatibility name predates this distinction. Membership proves that
    the importer consumed a key, not that it emitted an equivalent TOML value.
    """

    section: str
    key: str


@dataclass(frozen=True)
class AppliedDefault:
    """A resolved-TOML value the namelist did not supply.

    Omitted WRF keys default to their v4.6.1 Registry values (F2 fix)
    and are visible in the emitted TOML; entries here additionally record
    every value whose woof resolution DEVIATES from (or has no) WRF
    Registry source -- the never-silent contract's last mile.
    """

    key: str
    value: object
    reason: str


@dataclass(frozen=True)
class NamelistDefault:
    """An omitted key replaced by its upstream namelist default."""

    role: str
    section: str
    key: str
    value: object
    reason: str


@dataclass(frozen=True)
class SubstitutionReport:
    """Structured record of every non-1:1 importer decision.

    ``format`` renders the three explicit knob-parity sections --
    translated / fixed-by-WOOF / not-implemented -- plus the
    gpuwm-supplied-values footnote (the F2 never-silent last mile).
    ``notices`` carries scheme-generation statements the namelist itself
    cannot express (:data:`GF_SCHEME_GENERATION_NOTICE`): substitutions
    the importer CANNOT detect and therefore must declare.
    """

    substitutions: tuple[Substitution, ...]
    dropped: tuple[DroppedKey, ...]
    defaults_applied: tuple[AppliedDefault, ...] = ()
    fixed: tuple[FixedKey, ...] = ()
    translated: tuple[TranslatedKey, ...] = ()
    notices: tuple[str, ...] = ()
    #: Values the namelist left to WRF's Registry default that the TOML
    #: hands to the model as ``"auto"`` (A171: ``epssm``), one sentence
    #: each, so the reader learns the run may change them and why.
    model_choices: tuple[str, ...] = ()
    namelist_defaults: tuple[NamelistDefault, ...] = ()
    #: The WRF line ("3" or "4") whose Registry defaults the omitted keys
    #: took, and what chose it (the ``--wrf-version`` flag, the producing
    #: files' ``TITLE``, or the importer's default).  The line decides
    #: whether an omitted ``use_theta_m`` is dry theta (3) or the
    #: moist-theta substitution (4), so a receipt has to say which was
    #: read and why; the "Namelist defaults" row for that key repeats it.
    wrf_line: str = WRF_VERSION_DEFAULT
    wrf_line_source: str = WRF_VERSION_SOURCE_DEFAULT

    def format(self) -> str:
        lines = []
        if self.translated:
            lines.append(f"Other parsed controls (not a configuration-equivalence claim): "
                         f"{len(self.translated)} key(s)")
            by_section: dict[str, list[str]] = {}
            for t in self.translated:
                by_section.setdefault(t.section, []).append(t.key)
            for section, keys in by_section.items():
                lines.append(f"  &{section}: {', '.join(sorted(keys))}")
        if self.notices:
            lines.append("Scheme-generation notices:")
            for notice in self.notices:
                lines.append(f"  {notice}")
        if self.model_choices:
            lines.append("Left to the model (written \"auto\"):")
            for choice in self.model_choices:
                lines.append(f"  {choice}")
        lines.append("Physics substitutions (ratified):")
        if self.substitutions:
            for s in self.substitutions:
                lines.append(
                    f"  {s.key} {s.wrf_value} ({s.wrf_name}) -> "
                    f"{s.gpuwm_key} {s.gpuwm_value} ({s.gpuwm_name})"
                    + (f": {s.reason}" if s.reason else ""))
        else:
            lines.append("  (none)")
        if self.namelist_defaults:
            lines.append("Namelist defaults (substituted for omitted keys):")
            for entry in self.namelist_defaults:
                lines.append(f"  {entry.role} &{entry.section} {entry.key} = "
                             f"{entry.value}: {entry.reason}")
        if self.fixed:
            lines.append("Fixed by WOOF (validated against the only "
                         "implemented value):")
            for f in self.fixed:
                values = ", ".join(str(v) for v in f.values)
                lines.append(f"  &{f.section} {f.key} = {values} "
                             f"[fixed: {f.fixed_value}]: {f.reason}")
        if self.defaults_applied:
            lines.append("gpuwm-supplied values without a WRF Registry "
                         "source:")
            for a in self.defaults_applied:
                lines.append(f"  {a.key} = {a.value}: {a.reason}")
        lines.append("Not implemented (namelist keys consumed without a "
                     "woof counterpart):")
        for d in self.dropped:
            values = ", ".join(str(v) for v in d.values)
            lines.append(f"  &{d.section} {d.key} = {values}: {d.reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fortran-namelist parsing lives in woof.fortran_namelist, the one reader
# every namelist door shares; parse_namelist / parse_namelist_text keep
# their long-standing home here for the callers that import them from it.
# ---------------------------------------------------------------------------

def read_namelist_role(path: str | Path, role: str) -> dict[str, dict[str, list]]:
    """:func:`parse_namelist`, with an unreadable file as one sentence.

    THE shared refusal for "you named a namelist that is not there".
    ``woof import-namelist`` has always produced it; ``rw-wps
    --namelist-support-report``, documented as step one of
    migrating-from-wps.md, called :func:`parse_namelist` directly and
    handed the reader a five-frame ``FileNotFoundError`` traceback for
    the identical mistake.  Two surfaces answering the same condition
    differently is the bug; one function is the fix.
    """

    try:
        return parse_namelist(path)
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {role} {path}: {error}") from None


# ---------------------------------------------------------------------------
# The d01-only view of a namelist that may describe a whole hierarchy
# ---------------------------------------------------------------------------

#: Sentence a single-domain consumer appends when the namelist it was
#: handed describes more than one domain.  Kept here so the diagnosis is
#: spelled the same wherever it is raised.
MULTI_DOMAIN_ROOT_VIEW_HINT = (
    "multi-domain namelist passed to a single-domain root preparer: "
    "d01 is the domain being prepared, so its own (FIRST) column value is "
    "the one that binds; the differing per-domain column(s) below belong "
    "to the nests.  Fix by correcting d01's value, or by preparing the "
    "root from a max_dom = 1 namelist")


def root_domain_namelist_view(
        sections, *, source: str = "",
) -> tuple[dict[str, dict[str, object]], tuple[str, ...]]:
    """``(d01 view, per-domain columns that differ across domains)``.

    WRF per-domain arrays are ordered d01..dNN, so the head grid's value
    is element zero of every column -- for ``&physics``, ``&dynamics``,
    ``&time_control`` and the rest alike.  A single-domain consumer handed
    a hierarchy namelist wants exactly that view, and nothing else in the
    file.

    The second return value names every column that is NOT uniform.  Those
    are the keys where reading the wrong end of the array silently
    prepares a different domain, and a consumer that refuses on one of
    them owes the user :data:`MULTI_DOMAIN_ROOT_VIEW_HINT` rather than a
    sentence blaming the value it happened to read.

    ``source`` appears in refusals about a malformed column.
    """

    view: dict[str, dict[str, object]] = {}
    differing: list[str] = []
    for section_name, entries in sections.items():
        resolved: dict[str, object] = {}
        for key, values in entries.items():
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"&{section_name}/{key}"
                    + (f" of {source}" if source else "")
                    + " has no value; a WRF namelist key must carry at "
                    "least its d01 column entry")
            resolved[key] = values[0]
            if any(value != values[0] for value in values[1:]):
                differing.append(f"&{section_name}/{key}")
        view[section_name] = resolved
    return view, tuple(sorted(differing))


# ---------------------------------------------------------------------------
# Ratified scheme maps.  Every entry is (woof value, WRF scheme name,
# woof scheme name); identity entries share the name.  Any WRF value
# absent from its map is a hard error -- only implemented schemes map.
# ---------------------------------------------------------------------------

_MP_MAP = {
    0: (0, "none", "none"),
    1: (1, "Kessler", "Kessler"),
    2: (2, "Purdue Lin", "Purdue Lin"),
    # Ferrier-Aligo (fer_mp_hires, Registry.EM_COMMON:3019), native since
    # the Ferrier port: the mapped value equals the WRF value.  Its two
    # siblings stay ABSENT and keep the importer's unmapped-value error:
    # 15 advects the rime factor as qrimef, which this port carries as
    # non-advected state, and 95 is the old Eta scheme whose qs holds the
    # total ice.
    5: (5, "Ferrier-Aligo", "Ferrier-Aligo"),
    # WSM3, native since the WSM3 port: a TRANSLATION (3 -> 3), not a
    # substitution.  Mapping 3 -> 6 would hand a user who asked for WSM3's
    # simple ice (cloud ice inside qc, snow inside qr) WSM6's separate
    # ice, snow and graupel classes.
    3: (3, "WSM3", "WSM3"),
    4: (4, "WSM5", "WSM5"),
    6: (6, "WSM6", "WSM6"),
    # WSM6R, the revised WSM6 (Registry.EM_COMMON:3022, wsm6rscheme), native
    # since its port: a TRANSLATION, not a substitution.  Mapping 106 -> 6
    # would hand a user who asked for the revised scheme (one-at-a-time
    # process updates, smoothif switches, explicit sub-stepped fallout) the
    # original.  WRF's hail_opt does not reach wsm6r, so no hail knob is
    # written for it.
    106: (106, "WSM6R", "WSM6R"),
    8: (8, "Thompson", "Thompson"),
    # A namelist that says mp_physics = 9 gets Milbrandt-Yau, natively.  The
    # mapped value equals the WRF value, so this is not a substitution and
    # PHYSICS.md's substitution count is untouched.  Mapping 9 -> 10 would
    # have been one, and a lie: Morrison carries ONE rimed-ice category
    # selected by morr_rimed_ice while Milbrandt-Yau carries graupel and
    # hail simultaneously, and Morrison diagnoses droplet number where
    # Milbrandt-Yau prognoses it.
    9: (9, "Milbrandt-Yau 2-moment", "Milbrandt-Yau 2-moment"),
    10: (10, "Morrison 2-moment", "Morrison 2-moment"),
    # WDM6, native since the WDM6 port.  A TRANSLATION, not a substitution
    # (the mapped value equals the WRF value), for the same reason the 28
    # row below is one: WDM6 predicts cloud/rain number and a CCN
    # reservoir, so mapping 16 -> 6 would silently hand a user who asked
    # for double-moment warm rain a single-moment scheme.  WDM5 (14) and
    # WDM7 (26) are deliberately ABSENT and get the named refusal in
    # woof/config.py: they carry different hydrometeor sets and WDM6
    # cannot stand in for either.
    16: (16, "WDM6", "WDM6"),
    18: (18, "NSSL 2-moment", "NSSL 2-moment"),
    30: (30, "FAST_SBM spectral bin", "FAST_SBM spectral bin"),
    # WSM7, native since the WSM7 port: WSM6's six classes plus a separate
    # hail class.  A TRANSLATION (the mapped value equals the WRF value);
    # mapping 24 -> 6 would drop the hail category a user asked for.
    24: (24, "WSM7", "WSM7"),
    # A TRANSLATION, not a substitution: the mapped value equals the WRF
    # value, so PHYSICS.md's "exactly three ratified substitutions" claim is
    # untouched.  Mapping 28 -> 8 would have been the substitution, and it
    # would have been a lie: classic Thompson pins nc at Nt_c = 100e6 and
    # runs Cooper nucleation, while 28 carries prognostic nc/nwfa/nifa,
    # CCN activation and iceDeMott.  Silently downgrading a namelist that
    # asked for aerosol-aware physics is precisely the failure this map
    # exists to prevent.
    28: (28, "Thompson aerosol-aware", "Thompson aerosol-aware"),
    # P3 one-category (Registry.EM_COMMON:3038).  A NATIVE import, not a
    # substitution: the mapped value equals the WRF value because woof
    # ports this exact configuration.  Its three siblings (51/52/53) are
    # deliberately ABSENT from this map -- an absent key raises the
    # importer's unmapped-value error, and woof.config's
    # _P3_UNPORTED_VARIANTS then names each one's missing physics.  Mapping
    # any of them to 50 would be the substitution that silently downgrades
    # two ice categories or a third moment to the one-category solver.
    50: (50, "P3 one-category", "P3 one-category"),
    # 55 (Jensen-ISHMAEL) is deliberately ABSENT.  It was (10, "ISHMAEL",
    # "Morrison 2-moment") until 2026-10-12: a package replacement that ran
    # Morrison for a namelist asking for ISHMAEL.  woof.config's
    # _MP_UNPORTED_SCHEMES now refuses it by name, before this map is read.
}

#: WRF's aerosol-aware-Thompson namelist surface, section -> key -> what
#: ArWen does with it.  ``_Section.finish`` refuses any key it was not
#: offered, so every one of these must be consumed explicitly; the choice is
#: only ever between an INERT drop (the key changes nothing under the
#: selected scheme, in WRF either) and a hard refusal that names what is
#: missing.
#:
#: Under ``mp_physics != 28`` every key here is inert in WRF as well -- the
#: thompsonaero package is the only consumer -- so they drop with that
#: reason.  Under ``mp_physics == 28`` each one is refused by name, because
#: ArWen's established posture is to REFUSE where WRF silently overwrites:
#: WRF's ``share/module_check_a_mundo.F`` forces ``grav_settling`` to 0
#: (:2459-2474) and ``scalar_pblmix`` to 1 (:2477-2495) under mp=28 without
#: failing, and a user who asked for the other value would otherwise be
#: given a different model than the one they wrote down.
_MP28_AEROSOL_NAMELIST_KEYS: dict[str, dict[str, str]] = {
    "physics": {
        "qna_update":
            "no aerosol IC/BC lane exists to update from; the knob only "
            "means anything alongside use_aero_icbc",
        "scalar_pblmix":
            "PBL scalar mixing of the aerosol scalars qnc/qnwfa/qnifa is "
            "not wired -- "
            "woof/core/mynn_pbl.py passes flag_qnc/flag_qnwfa/flag_qnifa "
            "as literal False.  WRF SILENTLY forces scalar_pblmix=1 under "
            "mp_physics=28 with use_aero_icbc "
            "(share/module_check_a_mundo.F:2477-2495); WOOF refuses rather "
            "than accepting a value it would not honour",
        "grav_settling":
            "WRF SILENTLY forces grav_settling=0 under mp_physics=28 "
            "because the scheme already has gravitational fog settling "
            "(share/module_check_a_mundo.F:2459-2474); WOOF has no fog "
            "settling option at all, so it refuses the key instead of "
            "accepting a request it would overwrite",
        "wif_fire_emit":
            "no biomass-burning aerosol emission source "
            "(Registry.EM_COMMON:2657); WOOF's only nwfa2d is "
            "thompson_init's derived surface emission "
            "(phys/module_mp_thompson.F:510)",
        "wif_fire_inj":
            "no biomass-burning aerosol injection profile "
            "(Registry.EM_COMMON:2659), for the same reason.  Refused even "
            "at its WRF Registry default of 1, because with no emission "
            "source there is no accurate value: 1 describes a vertical "
            "distribution WOOF never performs",
        "dust_emis":
            "no dust emission source (Registry.EM_COMMON:2591); WOOF's "
            "nifa2d is identically zero, exactly as "
            "module_mp_thompson.F leaves it",
    },
    "domains": {},
}

#: The WIF KEY TRIPLE a real WRF namelist writes to select the monthly
#: aerosol climatology, and the only combination ArWen admits:
#:
#:   &physics  use_aero_icbc = .true.     (-> aer_init_opt = 1)
#:   &domains  wif_input_opt = 1          (the use_wif_input package)
#:   &domains  num_wif_levels = 30        (the dataset's own vertical axis)
#:
#: Admitted rather than refused because the ingest EXISTS: it is
#: ``woof/ingest/wif_climatology.py``, oracle-measured against WRF-4.7.1
#: real.exe.  A namelist that writes this triple described a run ArWen
#: can now produce, and dead-ending it was the refusal outliving the
#: defect it named.
#:
#: ``wif_input_opt = 2`` STAYS REFUSED, by name: it additionally allocates
#: the black-carbon scalar ``qnbca`` (Registry/registry.new3d_wif:82),
#: which has no consumer here -- no transport, no microphysical sink, and
#: nothing that would write it into history.  Admitting it would allocate
#: a field the model never touches and report a configuration ArWen does
#: not run.
WIF_INPUT_OPT_CLIMATOLOGY = 1
WIF_INPUT_OPT_WITH_BLACK_CARBON = 2

#: The dataset's own vertical axis: 30 sigma levels, read from the file's
#: records, not assumed (``woof/ingest/wif_climatology.py``).  WRF's
#: Registry default is the same 30.  A namelist naming any other count
#: describes a dataset ArWen has never seen, so it is refused with the
#: number it expected rather than silently reinterpreted.
WIF_DATASET_LEVELS = 30

_BL_MAP = {
    0: (0, "none", "none"),
    1: (1, "YSU", "YSU"),
    # The coupled MYNN 5/5 suite.  Two of the three legs the comment below
    # requires were already in place -- physics_compat admits the pair and
    # PHYSICS_SLOT_DISPATCH runs it, which is what the shipped MYNN fixed
    # profile forecasts with.  Only the importer leg lagged, so MYNN was
    # reachable at mp_physics=6 and nowhere else: pairing it with Thompson,
    # Morrison or NSSL had no front door at all.  Adding the map entry
    # widens no gate -- require_ready_wrf_physics still previews the whole
    # suite below and refuses every pairing it refused before.  (The
    # RUC/Noah-MP pairing refusals named in an earlier version of this
    # comment were retired by the surface-driver ownership port.)
    # MYJ (Mellor-Yamada-Janjic level 2.5) imports natively, never as a
    # substitution: the scheme is transcribed from the byte-frozen WRF
    # v4.6.1 module_bl_myjpbl.F and dispatched by _run_myj_pbl.  It travels
    # with sf_sfclay_physics=2 -- the importer accepts a namelist naming
    # both and woof.config.validate_myj_pairing refuses one without the
    # other, which is WRF's own fatal at
    # phys/module_physics_init.F:3770-3772.
    2: (2, "MYJ", "MYJ"),
    5: (5, "MYNN2.5", "MYNN"),
    # 900 IS NOT A WRF SELECTOR and this row does not pretend it is: SASE
    # is ArWen's own closure, out of WRF's namespace on purpose
    # (woof.config.SASE_PBL_SCHEME, admitted by validate_sase_config,
    # declared out of the WRF compatibility matrix's axes in
    # woof.wrf461_compatibility.AXIS_EXCLUSIONS).  The row exists because
    # the native route's namelist is GPUWM'S OWN FILE: the configuration
    # door writes bl_pbl_physics = 900 into it and the route reads it
    # back, so an importer with no row for the value made the one shipped
    # suite that selects the closure unwritable -- refused at its own
    # emission, on the route that runs it.  Both names say what it is,
    # because a substitution ledger that printed a WRF scheme name here
    # would claim a transcription that does not exist.
    900: (900, "none (no WRF counterpart)", "SASE"),
    901: (901, "none (CCPP scheme, no WRF counterpart)", "CCPP TKE-EDMF"),
    # Native since the Shin-Hong port (certified CPU authority, max ULP 0
    # against WRF v4.6.1; see the physics registry's shinhong option).  The
    # row was (1, "Shin-Hong", "YSU") -- a ratified substitution -- until
    # the scheme itself was admitted; per the doc block below, this map row
    # widens together with the physics_compat readiness row (the WRF
    # matrix's four (11, sfclay) cells) and the PHYSICS_SLOT_DISPATCH row.
    11: (11, "Shin-Hong", "Shin-Hong"),
    # The UW moist-turbulence PBL (CAMUWPBLSCHEME) imports natively: the
    # scheme is transcribed from WRF v4.7.1 module_bl_camuwpbl_driver.F and
    # the CAM modules it calls, dispatched by _run_uwpbl and graded against
    # tools/uwpbl_wrf471_oracle.  WRF names no surface layer for it; its one
    # pairing fatal (BEP/BEM urban) is woof.config.validate_uwpbl_config.
    9: (9, "UW (CAM5 Bretherton-Park)", "UW moist turbulence"),
}
_RA_LW_MAP = {
    0: (0, "none", "none"),
    1: (1, "RRTM", "WRF RRTM"),
    4: (4, "RRTMG", "RTE+RRTMGP"),
    5: (5, "New Goddard", "WRF New Goddard"),
}
_RA_SW_MAP = {
    0: (0, "none", "none"),
    1: (1, "Dudhia", "WRF Dudhia"),
    4: (4, "RRTMG", "RTE+RRTMGP"),
    5: (5, "New Goddard", "WRF New Goddard"),
}
#: WRF values the importer may emit into a woof configuration.  These are
#: the RUNNABLE sets, deliberately narrower than woof/config.py's schema
#: tables: importing a namelist writes a file someone will later run, so a
#: value that config would accept but physics_compat still refuses must fail
#: here rather than produce an unrunnable config.  Admitting a scheme means
#: widening these together with its physics_compat row and its
#: PHYSICS_SLOT_DISPATCH row -- never one of the three alone.
# 2 is WRF's Eta similarity surface layer, native since the MYJ port and
# admissible only beside bl_pbl_physics=2 (validate_myj_pairing).
_SFCLAY_ALLOWED = {0, 1, 2, 5, 91}
_SFSFC_ALLOWED = {0, 2, 3, 4}
# 16 is WRF's own NTIEDTKESCHEME number (module_cumulus_driver.F), and
# woof.config.CU_SCHEMES has admitted it since the New Tiedtke phase-2
# edit.  It was absent here alone, so the one shipped suite that selects
# it emitted a namelist this importer refused -- the emitter and the
# importer disagreeing about a selector the emitter writes, which is a
# configuration that cannot be read back as itself.  The cudt law that
# comes with the scheme is applied at emission below.
#
# 5 is WRF's G3SCHEME, Grell 3D (woof/core/kernels/g3.cu), admitted with
# its cudt law (emitted below) and its cugd_avedx key.  96 is WRF's
# NSASSCHEME, New SAS with NSCV (woof.core.column_cumulus).
_CU_ALLOWED = {0, 1, 3, 5, 6, 16, 96}

#: The Grell-Freitas scheme-generation notice, recorded on EVERY import
#: that selects cu_physics = 3 and printed by ``woof import-namelist``.
#:
#: Named breakage it prevents: a SILENT WRONG-SCHEME IMPORT.  WRF v4.8.0
#: removed Grell-Freitas from its tree and rebound cu_physics = 3 to
#: Grell-Freitas-Li (GFL, an external submodule adding prognostic cold
#: pools and G3-style subsidence spreading), while introducing zero new
#: namelist options and zero changed defaults -- clos_choice and ishallow
#: keep identical names and identical defaults on both sides while
#: meaning different schemes.  A v4.8.0 namelist is therefore
#: byte-indistinguishable from a v4.6.1 one at the GF option level, so a
#: user bringing one asks for GFL and would otherwise get v4.6.1 GF with
#: no word said.  No refusal can close that: a refusal that cannot detect
#: its trigger must not pretend to.  The defined behaviour is this
#: notice, stated on every GF import rather than only on suspicion,
#: because the suspicion is undetectable by construction.  The single
#: namelist spelling that DOES reveal spreading-generation intent,
#: ``cugd_avedx`` != 1 beside cu_physics = 3, is a hard refusal at its
#: consumption site in :func:`import_namelists`.
GF_SCHEME_GENERATION_NOTICE = (
    "cu_physics = 3 imports as the WRF v4.6.1 Grell-Freitas scheme "
    "(pinned dicycle = 1 Bechtold diurnal-cycle closure, MSE-launched "
    "k22 updraft origin searched from start_k22 = 2; "
    "woof/core/kernels/gf.cu), NOT WRF v4.8.0's Grell-Freitas-Li. "
    "v4.8.0 rebinds this same selector to GFL (prognostic cold pools, "
    "G3-style subsidence spreading) with zero new namelist options and "
    "zero changed defaults, so a v4.8.0-era namelist is "
    "byte-indistinguishable at the Grell-Freitas option level and the "
    "importer cannot detect which generation was meant.  It records "
    "this notice on every Grell-Freitas import instead, so the "
    "generation substitution is never silent.")


#: Recorded on every MYNN import whose &physics omits bl_mynn_mixscalars.
#: The omitted key takes the Registry default of the MYNN generation the
#: solver ports, and that default moved between WRF lines: the V3.9 line
#: (HRRR; gsd_41) has no such key, v4.6.1 (wrf_461) reads 0
#: (Registry.EM_COMMON:2479), v4.7.x reads 1 (Registry.EM_COMMON:2482)
#: under the MYNN-EDMF submodule woof does not carry.  A v4.7-era MYNN
#: namelist is byte-indistinguishable from a v4.6.1 one at this key, so
#: the importer cannot detect which default was meant; it records this
#: notice instead, so the divergence is never silent.
MYNN_MIXSCALARS_DEFAULT_NOTICE = (
    "bl_mynn_mixscalars omitted: imported as 0, the WRF v4.6.1 Registry "
    "default (Registry.EM_COMMON:2479) of the MYNN generation woof ports; "
    "the V3.9 line HRRR runs has no such key and mixes its aerosols "
    "through scalar_pblmix = 1.  WRF v4.7.x defaults the key to 1 "
    "(Registry.EM_COMMON:2482) under the MYNN-EDMF submodule woof does "
    "not carry; a namelist written for that line that relies on the "
    "default must say bl_mynn_mixscalars = 1 to mix the qn family "
    "(nc/ni/nwfa/nifa under mp_physics = 28; WRF's no-op under a "
    "microphysics with no number species; refused by name under a "
    "scheme WRF mixes in part, such as Thompson's qni, until the port "
    "carries per-species solves).")


#: The namelist groups the importer reads, by file.  Any other group is
#: refused by name.  Exported by woof.namelist_contract.
WPS_AUXILIARY_SECTIONS = {
    "domain_wizard": "WRF Domain Wizard GUI metadata; geogrid and metgrid "
                     "do not read this section",
    "mod_levs": "WPS mod_levs utility pressure-level selection; geogrid "
                "and metgrid do not read this section",
    "plotfmt": "WPS plotfmt utility plotting controls; geogrid and "
               "metgrid do not read this section",
}
WPS_SECTIONS = ("share", "geogrid", "ungrib", "metgrid",
                *WPS_AUXILIARY_SECTIONS)
#: namelist.input groups read whole.  Each is inert when its selector leaves
#: that machinery off (or, for &logging, always: it routes MPI-rank log
#: text).  With the selector on, &dfi_control is a DECLARED DIVERGENCE the
#: terminal prints (no filter: only the first hour's noise differs), and
#: &chem maps onto WOOF's chemistry or is refused by name
#: (woof/namelist_chem.py; owner ruling 2026-10-12: never run with no chemistry).
#: Selector key and the value that means "off", per group.
INPUT_MACHINERY_SECTIONS: dict[str, tuple[str | None, object]] = {
    "dfi_control": ("dfi_opt", 0),
    "chem": ("chem_opt", 0),
    "logging": (None, None),
}
INPUT_SECTIONS = ("time_control", "domains", "physics", "fdda", "dynamics",
                  "bdy_control", "grib2", "namelist_quilt", "noah_mp",
                  "stoch", "fire", *INPUT_MACHINERY_SECTIONS)
#: Groups whose every key is recorded and dropped: ungrib/metgrid staging
#: is replaced by woof's own ingest; FDDA (once no nudging selector is on),
#: GRIB2 and quilt-server keys have no woof counterpart.
WPS_DROPPED_SECTIONS = ("ungrib", "metgrid", *WPS_AUXILIARY_SECTIONS)
INPUT_DROPPED_SECTIONS = ("fdda", "grib2", "namelist_quilt")
STOCH_SELECTORS = frozenset({
    "rand_perturb", "skebs", "sppt", "spp", "spp_conv", "spp_pbl", "spp_lsm",
    "stoch_force_opt", "multi_perturb", "perturb_bdy", "perturb_chem_bdy",
    "pert_cld3", "pert_deng", "pert_farms", "pert_mynn", "pert_noah", "pert_thom",
})
#: WRF's numbered auxiliary input/history stream keys, recorded and dropped.
AUX_STREAM_KEY = re.compile(r"^aux(hist|input)\d+_")

#: WRF's generated I/O boilerplate also puts the stream number at the END
#: of format and frame-count controls.  All three spellings belong to one
#: import decision; a stream filename without its io_form must not pass
#: while the matching format declaration refuses as an unknown setting.
AUX_STREAM_FORMAT_KEY = re.compile(r"^(io_form|frames_per)_aux(hist|input)\d+$")

_WRFINPUT_OUTPUT_REASON = (
    "ignored WRF auxiliary wrfinput snapshot output; woof writes its "
    "forecast history products without this extra input-file stream")
_AUX_HISTORY_REASON = (
    "ignored WRF auxiliary history stream controls; woof writes its "
    "fixed per-domain wrfout product set, one frame per file")
_AUX_INPUT_REASON = (
    "ignored WRF auxiliary input file controls; woof's prepared "
    "initial/boundary forcing route does not consume WRF auxiliary "
    "file metadata; physics and FDDA selectors are validated separately")
_WRF_FILENAME_REASON = (
    "ignored WRF file naming control; woof's prepared input and history "
    "writers manage their own filenames")
_WRF_FORMAT_REASON = (
    "ignored WRF I/O format/fill control; woof's input and NetCDF history "
    "writers manage their own file format")
_WRF_DIAGNOSTIC_OUTPUT_REASON = (
    "ignored WRF diagnostic output controls; woof writes its forecast "
    "history product set without this auxiliary diagnostic stream")

#: Keys with no forecast-state effect.  This table is shared by the import
#: report, preprocessing compatibility report and generated site contract.
#: Forecast clocks (history_interval/begin/end) and forcing selectors do
#: not belong here: they keep their own translation or named refusal.
TIME_CONTROL_IGNORED_KEYS: dict[str, str] = {
    "write_input": _WRFINPUT_OUTPUT_REASON,
    "input_outname": _WRFINPUT_OUTPUT_REASON,
    **{f"inputout_{part}_{unit}": _WRFINPUT_OUTPUT_REASON
       for part in ("begin", "end") for unit in "ydhms"},
    "inputout_interval": _WRFINPUT_OUTPUT_REASON,
    **{f"inputout_interval_{unit}": _WRFINPUT_OUTPUT_REASON
       for unit in "dhms"},
    **{key: _WRF_FILENAME_REASON for key in (
        "history_inname", "history_outname", "input_inname", "bdy_inname",
        "bdy_outname", "rst_inname", "rst_outname")},
    **{key: _WRF_FORMAT_REASON for key in (
        "use_netcdf_classic", "ncd_nofill")},
    "iofields_filename": _AUX_HISTORY_REASON,
    "ignore_iofields_warning": _AUX_HISTORY_REASON,
    "output_ready_flag": "ignored WRF output-ready marker; woof records "
                         "history completion in its output manifest",
    **{key: _WRF_DIAGNOSTIC_OUTPUT_REASON for key in (
        "diag_print", "diurnal_diag", "mean_diag", "mean_diag_interval",
        "mean_diag_interval_d", "mean_diag_interval_h",
        "mean_diag_interval_m", "mean_diag_interval_mo",
        "mean_diag_interval_s", "mean_freq", "mean_interval")},
    "write_hist_at_0h_rst": "ignored WRF restart history-output switch; "
                            "woof's restart route manages its own history",
    "write_restart_at_0h": "ignored WRF initial restart-output switch; "
                           "woof's checkpoint route manages its own output",
}


_MACHINERY_OFF_REASON = {
    "dfi_control": "digital-filter initialization is off (dfi_opt = 0); "
                   "its filter controls are inert",
    "chem": "WRF-Chem is off (chem_opt = 0); its chemistry controls are "
            "inert",
    "logging": "WRF MPI-rank log routing; WOOF writes its own run log",
}


def _is_off(values, off) -> bool:
    return all(value == off for value in values)


def _consume_machinery_sections(inp: dict, drop, substitutions, fix=None):
    """Read &dfi_control, &chem and &logging whole (see
    :data:`INPUT_MACHINERY_SECTIONS`).

    Only the domains the run integrates are judged: a per-domain selector
    left on for an undeclared nest (the operational file's chem_opt = 18, 0
    under max_dom = 1 is the d01 value only) is read at those columns.

    Returns the :class:`woof.namelist_chem.ChemImport` of an active &chem
    group (mapped onto WOOF's chemistry, or refused by name there), else
    ``None``.
    """
    max_dom = (inp.get("domains", {}).get("max_dom") or [1])[0]
    max_dom = max_dom if isinstance(max_dom, int) and max_dom > 0 else 1
    chem_import = None
    for section_name, (selector, off) in INPUT_MACHINERY_SECTIONS.items():
        entries = inp.get(section_name)
        if entries is None:
            continue
        values = list(entries.get(selector) or [off]) if selector else [off]
        if selector == "chem_opt":
            values = values[:max_dom]
        if _is_off(values, off):
            for key, key_values in entries.items():
                drop(section_name, key, key_values,
                     _MACHINERY_OFF_REASON[section_name])
            continue
        if section_name == "dfi_control":
            # Refused, not declared (2026-10-12).  Until then this imported
            # as "forecast from the unfiltered analysis" and claimed the
            # later hours were the requested configuration; they are not:
            # the filter changes the state the forecast integrates from.
            raise _err("dfi_control", "dfi_opt", values, DFI_REFUSAL)
        else:
            # Mapped onto WOOF's chemistry or refused by name, never run
            # with none (owner ruling 2026-10-12; woof/namelist_chem.py).  Until
            # then this was a declared divergence to "no chemistry".
            from woof.namelist_chem import import_chem_section

            chem_import = import_chem_section(
                entries, max_dom=max_dom,
                fork=bool(operational_fork_signature(inp)),
                drop=drop, fix=fix, error=_err)
            continue
        for key, key_values in entries.items():
            if key != selector:
                drop(section_name, key, key_values,
                     f"companion control of {selector} = {values[0]}, "
                     "which WOOF does not run (declared divergence above)")
    return chem_import


#: Why a requested digital filter is refused (namelist door rule:
#: a setting that changes the forecast runs or is refused by name).
DFI_REFUSAL = (
    "digital-filter initialization requested.  WRF integrates the model "
    "backward and forward around the start time and starts the forecast "
    "from the filtered state (share/dfi.F); WOOF has no digital filter, so "
    "the forecast would start from the unfiltered analysis: the fast "
    "gravity-wave and surface-pressure noise the filter removes would be "
    "present in the first hours, and every later hour integrates from a "
    "different initial state than the one requested.  Next: set "
    "dfi_opt = 0.")


def _dfi_requested(inp: dict) -> bool:
    values = inp.get("dfi_control", {}).get("dfi_opt") or [0]
    return values[0] != 0


def _chem_requested(inp: dict, max_dom: int) -> bool:
    values = inp.get("chem", {}).get("chem_opt") or [0]
    return not _is_off(values[:max_dom], 0)


def _drop_dfi_companion(section, key: str, inp: dict, drop) -> None:
    drop(section.name, key, section.take(key),
         "companion of digital-filter initialization, which WOOF does not "
         "run (dfi_opt is refused by name)" if _dfi_requested(inp) else
         "digital-filter initialization is off (dfi_opt = 0); this DFI-pass "
         "control is inert")


#: real.exe vertical-interpolation controls, with the value WOOF's own
#: initialization always uses (woof/ingest/real.py, the WRF v4.6.1
#: module_initialize_real.F operator) and what that value means.  A
#: supplied match is recorded as fixed; any other value is a declared
#: divergence of the INITIAL STATE only (how the analysis is placed on
#: the model levels), never of the model that integrates it.
REAL_INIT_VERTICAL_CONTROLS: dict[str, tuple[object, str]] = {
    "interp_type": (2, "vertical interpolation in log pressure"),
    "lagrange_order": (2, "quadratic Lagrange vertical interpolation"),
    "force_sfc_in_vinterp": (1, "the surface value used for the lowest "
                                "model level"),
    "zap_close_levels": (500.0, "source levels within 500 Pa of the "
                                "surface removed"),
    "lowest_lev_from_sfc": (False, "the lowest level interpolated, not "
                                   "copied from the surface"),
    "adjust_heights": (False, "pressure-level heights used as analysed"),
}


def _consume_unrun_domain_controls(dm, inp: dict, drop, fix,
                                   substitutions) -> None:
    """&domains controls of initialization passes WOOF does differently."""
    for key, (used, meaning) in REAL_INIT_VERTICAL_CONTROLS.items():
        values = dm.take(key)
        if values is None:
            continue
        supplied = values[0]
        same = (float(supplied) == float(used)
                if isinstance(used, float) else supplied == used)
        if same:
            fix("domains", key, values, used,
                f"WOOF's initialization uses {meaning} (woof/ingest/real.py)")
            continue
        substitutions.append(Substitution(
            key=key, wrf_value=supplied, wrf_name=f"real.exe {key} = "
                                                  f"{supplied}",
            gpuwm_key=key, gpuwm_value=used, gpuwm_name=meaning,
            reason="initial state only: WOOF's initialization always uses "
                   f"{meaning}, so the analysis is placed on the model "
                   "levels slightly differently between source levels; the "
                   "integrating model is the one requested"))
    rebalance = dm.take("rebalance")
    if rebalance is not None:
        if rebalance[0] == 0:
            fix("domains", "rebalance", rebalance, 0,
                "WOOF has no separate start-time rebalance pass")
        else:
            substitutions.append(Substitution(
                key="rebalance", wrf_value=rebalance[0],
                wrf_name="start-time hydrostatic rebalance of the input",
                gpuwm_key="rebalance", gpuwm_value=0,
                gpuwm_name="initial state balanced by WOOF's initialization",
                reason="WOOF computes its initial pressure and geopotential "
                       "hydrostatically from the analysed temperature and "
                       "moisture inside its own initialization, so there "
                       "is no separate rebalance pass to switch on"))
    _drop_dfi_companion(dm, "time_step_dfi", inp, drop)


#: ``mp_tend_radar = 1`` (the operational WRF 3.9 fork's Registry; stock WRF
#: 4.x has no such key).  The fork reads the cloud analysis's
#: RAD_TTEN_DFI_1..4 / TTEN_TIMES out of the WRF input file and, on every
#: microphysics step for the whole run (the slot loop stops at slot 4, so
#: the last slot persists), replaces the clamped microphysics heating with
#: that tendency where the radar covers
#: (dyn_em/module_big_step_utilities_em.F:5901-6005 of NOAA-EMC/HRRR
#: v4.1.21).  A WRF input that carries the four slots and their times is
#: applied exactly so (woof/ingest/wrfinput_radar.py, on the device kernel
#: of woof/da/radar_tten.py).  Refused by name where no such tendency
#: exists: a namelist imported over GRIB or met_em input has no cloud
#: analysis, and wrf.exe given mp_tend_radar = 1 with no tendency reads
#: zero-filled slots, which it treats as "covered, no heating" and so
#: removes the microphysics heating everywhere.
MP_TEND_RADAR_REFUSAL = (
    "radar latent heating requested on {domains}, and the input has no "
    "radar tendency to apply.  The operational WRF fork replaces the "
    "microphysics heating with the radar-derived temperature tendency the "
    "cloud analysis writes into the WRF input (RAD_TTEN_DFI_1..4 and "
    "TTEN_TIMES); with no such fields wrf.exe reads zero-filled slots, which "
    "it treats as covered with zero heating, so storms would get no latent "
    "heating anywhere.  Next: start from the WRF input that carries the "
    "tendency (woof run --wrfinput <real.exe directory>), where it is "
    "applied as the fork applies it; or set mp_tend_radar = 0 and add "
    "[radar_heating] with a reflectivity window root "
    "(docs/public/CONFIGURATION.md), which builds the same NOAA tendency on "
    "the device from observed reflectivity.")

#: How an applied ``mp_tend_radar = 1`` is recorded.
MP_TEND_RADAR_APPLIED = (
    "radar latent heating on {domains}: the WRF input's RAD_TTEN_DFI_1..4 "
    "and TTEN_TIMES replace the clamped microphysics heating where the "
    "radar covers, on every microphysics step, as the operational WRF fork "
    "does (module_big_step_utilities_em.F); mp_tend_lim is the namelist's")


def _ra_sw_eclipse_decision(values, active, *, run_window, ra_sw, fix):
    """Record the reachable GPU eclipse switch without discarding it."""
    from woof.eclipse_windows import eclipses_during
    if any(int(value) not in (0, 1) for value in active):
        raise _err("physics", "ra_sw_eclipse", values, "must be 0 or 1")
    start, end = run_window
    during = eclipses_during(start, end)
    reason = ("GPU Besselian obscuration on every model step: "
              + ", ".join(w.label() for w in during) if during else
              "no eclipse between " + f"{start:%Y-%m-%d %H:%M} and {end:%Y-%m-%d %H:%M} UTC; "
              "GPU obscuration is zero and shortwave remains undimmed")
    fix("physics", "ra_sw_eclipse", values, list(active), reason)


def _consume_unrun_physics_controls(ph, inp: dict, max_dom: int, cu: list,
                                    drop, fix, substitutions, *,
                                    run_window, ra_sw,
                                    radar_tendency_domains=frozenset(),
                                    grid_ids=()) -> None:
    """&physics controls of packages WOOF does not run.

    Each is inert when its package is off.  A requested package is refused
    by name unless it provably does nothing in this run (``ra_sw_eclipse``
    outside every eclipse). Interval precipitation is translated separately into native GPU buckets.
    """
    def column(key):
        values = ph.take(key)
        return values, (None if values is None else values[:max_dom])

    drop("physics", "co2tf", ph.take("co2tf"),
         "GFDL radiation's CO2 flag; read only under ra_*_physics = 99, "
         "which WOOF does not offer")
    # The Grell ensemble dimensions are read (and, without Grell 3D on any
    # domain, dropped as inert) by _translate_namelists beside cu_diag.
    values = ph.take("mp_tend_radar")
    if values is not None:
        # One value for every domain: the fork's Registry declares it with
        # one entry (Registry.EM_COMMON:2441, "rconfig integer
        # mp_tend_radar namelist,physics 1 0").
        # A per-domain list that states the same value everywhere (the
        # common "0, 0" of a nested namelist) says what the single entry
        # says and is read as it; 2.9.0 accepted "0, 0" as inert.
        if (not values or any(type(v) is not int or v not in (0, 1)
                              for v in values)):
            raise _err("physics", "mp_tend_radar", values,
                       "must be 0 or 1 (the operational WRF fork's "
                       "Registry.EM_COMMON:2441 declares one integer)")
        if len(set(values)) != 1:
            raise _err("physics", "mp_tend_radar", values,
                       "differs by domain, but the operational WRF fork "
                       "declares one value for every domain "
                       "(Registry.EM_COMMON:2441), so the per-domain choice "
                       "this list states is one no build of it can make; "
                       "write one value")
        if values[0] == 0:
            drop("physics", "mp_tend_radar", values,
                 "radar-derived latent-heating tendency is off; inert")
        else:
            forced = list(grid_ids) or list(range(1, max_dom + 1))
            missing = [gid for gid in forced
                       if gid not in radar_tendency_domains]
            if missing:
                raise _err("physics", "mp_tend_radar", values,
                           MP_TEND_RADAR_REFUSAL.format(
                               domains=", ".join(f"d{gid:02d}"
                                                 for gid in missing)))
            fix("physics", "mp_tend_radar", values, 1,
                MP_TEND_RADAR_APPLIED.format(
                    domains=", ".join(f"d{gid:02d}" for gid in forced)))
    tracer_opt = inp.get("dynamics", {}).get("tracer_opt") or [0]
    if _is_off(tracer_opt[:max_dom], 0):
        drop("physics", "tracer_pblmix", ph.take("tracer_pblmix"),
             "PBL mixing switch of the passive-tracer array, which does not "
             "exist with tracer_opt = 0 (the chemistry is the chem array)"
             if _chem_requested(inp, max_dom) else
             "no passive-tracer array exists (tracer_opt = 0, no "
             "chemistry), so there is nothing for the PBL to mix")


def _consume_unrun_dynamics_controls(dyn, inp: dict, max_dom: int,
                                     km_opt_col: list, drop, fix) -> dict:
    """&dynamics array switches for arrays WOOF does or does not carry.

    Returns the chem-array switches to emit into ``[shared]`` when the
    namelist's chemistry is mapped onto WOOF's (woof/namelist_chem.py
    CARRIED_DYNAMICS_KEYS): RunConfig carries them under the same names,
    and the chem law (woof.config.validate_chem_config) judges their
    values.
    """
    from woof.namelist_chem import CARRIED_DYNAMICS_KEYS

    chem = _chem_requested(inp, max_dom)
    carried: dict = {}
    for key in CARRIED_DYNAMICS_KEYS:
        values = dyn.take(key)
        if values is None:
            continue
        if not chem:
            drop("dynamics", key, values,
                 "transport/filter switch of the chemistry array, which "
                 "does not exist with chemistry off")
            continue
        column = values[:max_dom]
        if len(set(map(repr, column))) != 1:
            raise _err("dynamics", key, values,
                       "WOOF carries one value for the whole tree; this "
                       f"namelist gives {column} on its domains")
        carried[key] = column[0]
    tracer_opt = dyn.entries.get("tracer_opt") or [0]
    if _is_off(tracer_opt[:max_dom], 0):
        drop("dynamics", "tracer_opt", dyn.take("tracer_opt"),
             "no passive-tracer package is requested")
        drop("dynamics", "tracer_mix6_off", dyn.take("tracer_mix6_off"),
             "filter switch of the passive-tracer array, which does not "
             "exist with tracer_opt = 0")
    tke = dyn.entries.get("tke_mix6_off")
    if tke is not None:
        if all(value != 2 for value in km_opt_col[:max_dom]):
            drop("dynamics", "tke_mix6_off", dyn.take("tke_mix6_off"),
                 "filter switch of the prognostic TKE array, which exists "
                 "only under km_opt = 2")
        elif all(value is False for value in tke[:max_dom]):
            fix("dynamics", "tke_mix6_off", dyn.take("tke_mix6_off"), False,
                "the sixth-order filter acts on TKE under km_opt = 2 "
                "(woof/core/dycore.py diff6 rows)")
    scalar = dyn.entries.get("scalar_mix6_off")
    if scalar is not None and all(value is False for value in scalar[:max_dom]):
        fix("dynamics", "scalar_mix6_off", dyn.take("scalar_mix6_off"),
            False, "the sixth-order filter acts on the scalar array "
                   "(woof/core/dycore.py diff6 rows)")
    for key in ("km_opt_dfi", "moist_adv_dfi_opt"):
        _drop_dfi_companion(dyn, key, inp, drop)
    return carried


def ignored_time_control_reason(key: str) -> str | None:
    """Why a WRF I/O declaration is inert after forecast selectors pass."""
    if key in TIME_CONTROL_IGNORED_KEYS:
        return TIME_CONTROL_IGNORED_KEYS[key]
    if AUX_STREAM_KEY.match(key) or AUX_STREAM_FORMAT_KEY.fullmatch(key):
        return _AUX_INPUT_REASON if "auxinput" in key else _AUX_HISTORY_REASON
    return None


#: &physics keys pinned to the one value woof implements, as
#: (key, pin, why).  Exported by woof.namelist_contract.
PHYSICS_PINS: tuple[tuple[str, int, str], ...] = (
    ("shcu_physics", 0,
     "no separate shallow-cumulus scheme (1 Grell 3D shallow, 2 UW, 3 "
     "GRIMS, 4 NSCV, 5 Deng) is ported in this release; the shallow "
     "convection inside Grell-Freitas and Grell 3D (ishallow = 1) and New "
     "SAS's built-in NSCV do run"),
    ("kf_edrates", 0,
     "KF entrainment/detrainment rate diagnostics are not "
     "implemented"),
    ("flag_sm_adj", 0,
     "RUC option identity: woof has no RUC soil ingest for the "
     "adjustment to act on (share/module_soil_pre.F:2063)"),
    ("sst_update", 0,
     "SST update cycling is not implemented (single-analysis "
     "case runs)"),
    ("sst_skin", 0,
     "skin-SST diurnal adjustment is not implemented"),
    ("tmn_update", 0,
     "deep-soil temperature update is not implemented"),
)

#: WRF package selectors whose schemes are not ported, as key -> (the off
#: value, {on value: reason}), with the keys WRF reads only when the
#: selector is on.
#:
#: Named breakage (2.9.0): these selectors were not read at all, so even
#: their OFF value (sf_ocean_physics = 0, windfarm_opt = 0, as hurricane and
#: operational templates write them) was refused as "unmapped key(s) ...
#: extend the ratified map", and the ON values got that importer sentence
#: instead of the physics reason.
#:
#: They are consumed LATE (just before &physics is finished), after every
#: scheme's own reader: a port that admits one of these selectors takes the
#: key earlier and this table never sees it, so landing the port and this
#: table in either order cannot refuse the ported value.  The row is then
#: retired with the port.
UNPORTED_PACKAGE_SELECTORS: dict[str, tuple[int, dict[int, str]]] = {
    # sf_ocean_physics is consumed by the Pollard / PWP reader below.
    "windfarm_opt": (0, {
        2: "the MAV wind-farm drag scheme (mavscheme) is not ported in this "
           "release",
    }),
}

#: Keys WRF reads only when one of those selectors (or tmn_update, pinned
#: in PHYSICS_PINS) is on, as (section, key) -> (selector, why the key is
#: inert with it off).
UNPORTED_SELECTOR_PARAMETERS: dict[tuple[str, str], tuple[str, str]] = {
    # oml_hml0, oml_gamma and oml_relaxation_time are the ported ocean
    # mixed layer's own settings, read by its reader.
    **{("physics", key): ("windfarm_opt",
                          "wind-farm parameter; WRF reads it only when "
                          "windfarm_opt is on")
       for key in ("windfarm_ij", "windfarm_deg", "windfarm_tke_factor",
                   "windfarm_wake_model", "windfarm_overlap_method")},
    ("physics", "lagday"): ("tmn_update",
                            "deep-soil temperature lag; WRF reads it only "
                            "when tmn_update = 1"),
}


def _consume_unported_package_selectors(section, name: str, max_dom: int,
                                        drop) -> None:
    """Read :data:`UNPORTED_PACKAGE_SELECTORS` and drop
    :data:`UNPORTED_SELECTOR_PARAMETERS` for one namelist group.

    A selector written at its off value drops as inert; any other value is
    refused with the scheme's name.  Its parameters drop as inert: with
    the selector off WRF never reads them, and with it on the selector's
    refusal names the run's problem (the refusal census sets the selector
    aside and keeps checking the rest)."""
    if name == "physics":
        for key, (off, reasons) in UNPORTED_PACKAGE_SELECTORS.items():
            raw = section.take(key)
            if raw is None:
                continue
            active = list(raw[:max_dom])
            if all(type(value) is int and value == off for value in active):
                drop("physics", key, raw,
                     f"{key} = {off} is WRF's off value; inert")
                continue
            named = [reasons[value] for value in dict.fromkeys(active)
                     if type(value) is int and value in reasons]
            why = ("; ".join(named) if named else
                   f"WRF defines {key} = {off} or "
                   + ", ".join(str(value) for value in reasons))
            raise _err("physics", key, raw,
                       f"{why[0].upper()}{why[1:]}. Nothing is substituted "
                       f"for it; set {key} = {off}.")
    for (group, key), (selector, why) in UNPORTED_SELECTOR_PARAMETERS.items():
        if group != name:
            continue
        values = section.take(key)
        if values is not None:
            drop(group, key, values, why)

#: &dynamics keys pinned to the one value the dycore implements, as
#: (key, pin, why).  Exported by woof.namelist_contract.
DYNAMICS_PINS: tuple[tuple[str, int, str], ...] = (
    ("rk_ord", 3,
     "the dycore integrates WRF's 3-stage RK3 only "
     "(woof/core/dycore.py stage table)"),
    ("h_mom_adv_order", 5,
     "horizontal momentum advection is the WRF flux5 stencil, "
     "hardcoded (woof/core/kernels/advection.cu)"),
    # v_mom_adv_order and v_sca_adv_order left this table with
    # lane/286-vadv5: WRF's vert_order 3 and 5 ladders are both carried
    # (woof/core/kernels/advection.cu zface_half and advect_w's faces)
    # and the keys import as per-domain columns below.
    ("momentum_adv_opt", 1,
     "standard (non-positive-definite) momentum advection only"),
    # gwd_opt left this table with lane/282-terrain-drag: WRF v4.7.1's
    # gwd_opt = 1 and 3 are ported (woof.core.terrain_drag) and the key
    # imports from &dynamics (v4) or &physics (the v3 placement) below.
)

#: &dynamics keys with one implemented value on every domain, as
#: key -> (value, why any other is refused).  The translation below reads
#: its pins from here and woof.namelist_contract exports them.
DYNAMICS_REQUIRED_VALUES: dict[str, tuple[object, str]] = {
    "non_hydrostatic": (True, "woof is nonhydrostatic-only."),
    "moist_adv_opt": (1, "only the matched WRF positive-definite option 1 "
                         "is implemented for moisture mass."),
    "scalar_adv_opt": (1, "only the matched WRF positive-definite option "
                          "1 is implemented for scalar/number fields."),
    "h_sca_adv_order": (5, "only the WRF Registry default 5 imports: "
                           "woof's transported-scalar stencils are fixed "
                           "at WRF's 5th-order horizontal/3rd-order "
                           "vertical forms."),
}


#: &physics integer switches the importer admits at a closed set of
#: values, key -> (admitted values, why any other is refused).  Read
#: by the translation below and exported by woof.namelist_contract,
#: so the site checks an upload against the rule the engine runs.
PHYSICS_CHOICES: dict[str, tuple[tuple[int, ...], str]] = {
    "bl_mynn_mixlength": ((1, 2),
     'must be 1 or 2 (the implemented WRF MYNN mixing-length branches).'),
    "scalar_pblmix": ((0, 1),
     'must be 0 (off) or 1 (WRF post-PBL scalar diffusion).'),
    "bl_mynn_mixscalars": ((0, 1),
     'must be 0 (off) or 1 (MYNN scalar plume transport).'),
    "mp_zero_out": ((0, 1, 2),
     'must be 0 (off), 1 (zero non-vapour species below '
     'mp_zero_out_thresh) or 2 (also floor vapour at zero), WRF '
     'phys/module_microphysics_zero_out.F.'),
    "mp_zero_out_all": ((0, 1),
     'must be 0 (moist array only, the WRF v4.6.1 default) or 1 (also '
     'the scalar arrays).'),
    "no_mp_heating": ((0, 1),
     'must be 0 (microphysics latent heating on, the WRF default) '
     'or 1 (heating off, '
     'module_big_step_utilities_em.F:5770-5782).'),
    "ysu_topdown_pblmix": ((0, 1),
     'must be 0 or 1 (top-down radiation-driven PBL mixing).'),
    "swint_opt": ((0, 1),
     "must be 0 (hold the radiation call's surface shortwave for the "
     "interval) or 1 (fit it to the zenith cosine at every radiation call "
     "and evaluate the fit on every step, woof.core.swint)."),
    "alb_sol": ((0, 1),
     "must be 0 (unchanged land albedo) or 1 (sun-angle-dependent "
     "ALBSOL/ALBBCKSOL updated on radiation steps)."),
    "isfflx": ((0, 1),
     'must be 0 (surface heat/moisture fluxes off) or 1 (on); WRF '
     "value 2 also needs woof's unported prescribed-"
     'flux/diffusion forcing path.'),
    "use_mp_re": ((0, 1),
     'must be 0 (legacy-RRTMG calculated radii) or 1 (use the WRF '
     'microphysics scheme table).'),
    "isftcflx": ((0, 1, 2),
     'must be 0 (standard MM5 water-point roughness), 1 (Garratt),'
     ' or 2 (Donelan).'),
    "iz0tlnd": ((0, 1, 2),
     'must be 0 (standard CZIL), 1 (Chen-Zhang), or 2 (fixed CZIL '
     '0.1).'),
    "opt_thcnd": ((1, 2),
     'must be 1 (Johansen soil thermal conductivity) or 2 '
     '(McCumber-Pielke).'),
}

#: &domains nest-interpolation keys pinned to WRF's default, as
#: key -> (the one admitted value, why).  Exported by
#: woof.namelist_contract.
NEST_GUARDS: dict[str, tuple[object, str]] = {
    "interp_method_type": (2,
     'only SINT (2, the WRF default per Registry.EM_COMMON:2301) '
     'is implemented.'),
    "nest_interp_coord": (0,
     'isobaric nest re-interpolation is not implemented.'),
    "vert_refine_method": (0,
     'vertical nest refinement is rejected (identical '
     'e_vert/eta_levels/p_top on all domains).'),
    "input_from_hires": (False,
     'high-resolution child terrain input is rejected (children '
     'SINT + blend the parent terrain).'),
}


def _unported_tail(key: str, value) -> str:
    """The named reason an unported WRF selector value is refused, as a
    sentence to append to the menu refusal (woof.config
    WRF_UNPORTED_SCHEME_REASONS), or ``""``."""
    from woof.config import wrf_unported_refusal_tail

    return wrf_unported_refusal_tail(key, value)


def _port_receipt(**selection) -> str:
    """The fail-closed port receipt for an in-port selector, or ``""``."""
    from woof.physics_compat import pending_wrf_physics_components

    request = {"mp_physics": 0, "sf_sfclay_physics": 0, "bl_pbl_physics": 0,
               "sf_surface_physics": 0, "num_soil_layers": 4}
    request.update(selection)
    blockers = pending_wrf_physics_components(**request)
    if not blockers:
        return ""
    return " Port status: " + "; ".join(
        item.format() for item in blockers)


class NamelistRefusal(ValueError):
    """One importer refusal that names the namelist setting(s) it is about.

    ``keys`` holds ``(section, key)`` pairs, ``key == "*"`` for a whole
    section.  :func:`import_namelists` uses them to set the named settings
    aside and keep checking the rest of the pair, so one refusal lists
    every problem instead of the first one (a WOOF user found five
    refusals in one namelist on five GPU boxes, one per boot).
    """

    def __init__(self, message: str, keys=()):
        super().__init__(message)
        self.keys = tuple((str(section), str(key)) for section, key in keys)


_KEY_TOKEN = re.compile(r"^[a-z][a-z0-9_]*$")


def _err(section: str, key: str, value, why: str) -> NamelistRefusal:
    # "moist_adv_opt/scalar_adv_opt" names two keys; a composite label
    # such as "end_* datetime" names none the census could set aside.
    keys = [(section, part) for part in str(key).split("/")
            if _KEY_TOKEN.match(part)]
    return NamelistRefusal(f"&{section} {key} = {value!r}: {why}", keys)


def _identity_matches(value, admitted) -> bool:
    """Type-strict comparison for option-identity pins.

    Booleans must be Fortran logicals (never 0/1 integers), floats accept
    exact integer spellings (``soiltstep = 0``), and integers refuse
    fractional or logical tokens -- no coercion that could smuggle a
    different option through the pin.
    """
    if isinstance(admitted, bool):
        return isinstance(value, bool) and value == admitted
    if isinstance(admitted, float):
        return (isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(value) == admitted)
    return (isinstance(value, int) and not isinstance(value, bool)
            and value == admitted)


#: Keys the importer requires to be PRESENT (contract values are checked
#: later, one rule at a time).  Censused in one sweep up front so a
#: namelist missing several keys yields ONE report naming all of them
#: instead of one refusal per re-run.  The scattered ``required()``
#: raises stay as fail-closed backstops.
_REQUIRED_KEYS = {
    ("wps", "geogrid"): (
        "dx", "e_sn", "e_we", "i_parent_start", "j_parent_start",
        "parent_grid_ratio", "ref_lat", "ref_lon",
        "stand_lon", "truelat1"),
    ("input", "domains"): ("time_step",),
    ("input", "dynamics"): ("km_opt",),
    ("input", "physics"): (
        "bl_pbl_physics", "mp_physics", "ra_lw_physics", "ra_sw_physics"),
}

_REQUIRED_KEY_REASONS = {
    ("wps", "geogrid"): {
        "dx": "WPS initializes an unset sentinel, not usable grid spacing",
        "e_sn": "WPS initializes INVALID, not a usable north-south grid size",
        "e_we": "WPS initializes INVALID, not a usable east-west grid size",
        "i_parent_start": "WPS has no initialized parent-grid x offset",
        "j_parent_start": "WPS has no initialized parent-grid y offset",
        "parent_grid_ratio": "WPS initializes INVALID, not a usable nest ratio",
        "ref_lat": "WPS initializes an unset sentinel, not a domain latitude",
        "ref_lon": "WPS initializes an unset sentinel, not a domain longitude",
        "stand_lon": "WPS initializes an unset sentinel, not a projection longitude",
        "truelat1": "WPS initializes an unset sentinel, not a projection latitude",
    },
    ("input", "domains"): {
        "time_step": "WRF's -1 sentinel does not specify the imported fixed clock",
    },
    ("input", "dynamics"): {
        "km_opt": "WRF's -1 sentinel does not select a turbulence closure",
    },
    ("input", "physics"): {
        key: "WRF's -1 sentinel needs a physics_suite selection; this importer "
             "requires an explicit supported scheme instead of inventing a suite"
        for key in ("bl_pbl_physics", "mp_physics", "ra_lw_physics", "ra_sw_physics")
    },
}

# WRF v4.7.1 Registry.EM_COMMON. Values are initialized on every domain;
# geometry and clock cross-checks below still reject incompatible pairs.
_NAMELIST_DEFAULTS = {
    ("input", "time_control"): {
        "start_year": (1993, 2173), "start_month": (3, 2174),
        "start_day": (13, 2175), "end_year": (1993, 2179),
        "end_month": (3, 2180), "end_day": (14, 2181),
    },
    ("input", "domains"): {
        "e_we": (32, 2270), "e_sn": (32, 2272), "e_vert": (31, 2274),
        "parent_id": (0, 2319), "i_parent_start": (1, 2320),
        "j_parent_start": (1, 2321), "parent_grid_ratio": (1, 2322),
        "parent_time_step_ratio": (1, 2323),
    },
    ("input", "dynamics"): {"mix_full_fields": (False, 2908)},
}

#: The same keys' rows in the WRFV3.9 Registry.EM_COMMON operational HRRR
#: builds (NOAA-EMC/HRRR v4.1.21, the file OMITTED_USE_THETA_M_SOURCE
#: pins).  Every default VALUE is the same on both lines (read off that
#: file); only the row moved, and a default row cites the Registry the
#: namelist was written for.
_NAMELIST_DEFAULT_LINES_WRF3 = {
    "start_year": 2046, "start_month": 2047, "start_day": 2048,
    "end_year": 2052, "end_month": 2053, "end_day": 2054,
    "e_we": 2143, "e_sn": 2145, "e_vert": 2147,
    "parent_id": 2190, "i_parent_start": 2191, "j_parent_start": 2192,
    "parent_grid_ratio": 2193, "parent_time_step_ratio": 2194,
    "mix_full_fields": 2674,
}


def _registry_default_citation(key: str, v4_line: int, line: str) -> str:
    if line == "3":
        return ("WRFV3.9 Registry/Registry.EM_COMMON:"
                f"{_NAMELIST_DEFAULT_LINES_WRF3[key]} default "
                "(NOAA-EMC/HRRR v4.1.21)")
    return f"WRF v4.7.1 Registry/Registry.EM_COMMON:{v4_line} default"

MIX_FULL_FIELDS_SUBSTITUTION = (
    "The generic importer retains its established full-field mixing "
    "resolution under the metric operator. Named source requests retain "
    "their declared logicals and may choose their own omitted default.")

# GUI staging templates may leave date components unresolved. The imported
# forecast clock is concrete in &time_control; actual WPS dates still cross-check.
_WPS_DATE_TEMPLATE = re.compile(
    r"(?:\d{4}|@[A-Za-z][A-Za-z0-9_]*@)-"
    r"(?:\d{2}|@[A-Za-z][A-Za-z0-9_]*@)-"
    r"(?:\d{2}|@[A-Za-z][A-Za-z0-9_]*@)_"
    r"(?:\d{2}|@[A-Za-z][A-Za-z0-9_]*@):"
    r"(?:\d{2}|@[A-Za-z][A-Za-z0-9_]*@):"
    r"(?:\d{2}|@[A-Za-z][A-Za-z0-9_]*@)")


def _apply_namelist_defaults(
        wps: dict, inp: dict, *, wrf_version: str = WRF_VERSION_DEFAULT,
        wrf_version_source: str | None = None) -> list[NamelistDefault]:
    """Fill omitted controls before the required-key inventory.

    ``wrf_version`` is the WRF line the namelist was written for and
    ``wrf_version_source`` what chose it; each default row cites that
    line's Registry.  An omitted ``&dynamics/use_theta_m`` is RECORDED
    here and not filled: its default is the one that differs between the
    lines, the translation resolves it (and books the moist-theta
    substitution on the V4 line), and this row is where the reader sees
    which Registry answered and why that one.
    """
    line = wrf_line_of(wrf_version)
    source = (WRF_VERSION_SOURCE_DEFAULT if wrf_version_source is None
              else wrf_version_source)
    count = inp.get("domains", {}).get("max_dom", [1])[0]
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 21:
        raise _err("domains", "max_dom", count,
                   "must be an integer in [1, 21] (WRF's compiled "
                   "max_domains default); refusing to expand per-domain "
                   "arrays for an implausible domain count.")
    recorded = []
    for (role, section), defaults in _NAMELIST_DEFAULTS.items():
        entries = (wps if role == "wps" else inp).setdefault(section, {})
        for key, (value, registry_line) in defaults.items():
            if key in entries:
                continue
            if (section == "time_control" and key.startswith("end_")
                    and _run_length_seconds(entries) > 0):
                continue
            entries[key] = [value] * count
            recorded.append(NamelistDefault(
                role, section, key, value,
                _registry_default_citation(key, registry_line, line)))
    if "use_theta_m" not in inp.get("dynamics", {}):
        recorded.append(NamelistDefault(
            "input", "dynamics", "use_theta_m",
            OMITTED_USE_THETA_M_BY_WRF_VERSION[line],
            f"{OMITTED_USE_THETA_M_SOURCE[line]}; WRF line {line} from "
            f"{source}"))
    geo = wps["geogrid"]
    if "parent_id" not in geo:
        geo["parent_id"] = [1] * count
        recorded.append(NamelistDefault(
            "wps", "geogrid", "parent_id", 1,
            "WPS v4.6.0 geogrid/src/gridinfo_module.F:94 default"))
    if (str(geo.get("map_proj", ["lambert"])[0]).lower() == "lambert"
            and "truelat2" not in geo and "truelat1" in geo):
        geo["truelat2"] = list(geo["truelat1"])
        recorded.append(NamelistDefault(
            "wps", "geogrid", "truelat2", geo["truelat1"][0],
            "WPS v4.6.0 geogrid/src/gridinfo_module.F:448-451 "
            "defaults the second Lambert latitude to truelat1"))
    return recorded


#: WRF's own physics suites, transcribed from WRF v4.6.1
#: share/module_check_a_mundo.F setup_physics_suite (:3116-:3147).  WRF
#: replaces a selector with the suite's value ONLY where the namelist wrote
#: -1 for that domain, and stops on any other suite name (:3148).  Table
#: work, not a code path: a suite WRF adds is one more row.
WRF_PHYSICS_SUITES = {
    "conus": {"cu_physics": 6, "mp_physics": 8, "ra_lw_physics": 4,
              "ra_sw_physics": 4, "bl_pbl_physics": 2,
              "sf_sfclay_physics": 2, "sf_surface_physics": 2},
    "tropical": {"cu_physics": 16, "mp_physics": 6, "ra_lw_physics": 4,
                 "ra_sw_physics": 4, "bl_pbl_physics": 1,
                 "sf_sfclay_physics": 91, "sf_surface_physics": 2},
}
_WRF_PHYSICS_SUITE_LINES = {"conus": 3116, "tropical": 3133}


def _expand_physics_suite(inp: dict, drop) -> list[NamelistDefault]:
    """Resolve ``&physics physics_suite`` the way wrf.exe does.

    Named breakage it prevents: NCAR's own benchmark namelists (and every
    namelist built from WRF's README examples) write
    ``physics_suite = 'conus'`` with the seven selectors at -1; this
    importer refused all seven as unset sentinels, so a WRF run directory
    that wrf.exe runs unchanged could not be read at all.  Each replaced
    selector is recorded as a namelist default citing the suite row, and
    the suite key itself is consumed with that reason.  'none' (WRF's
    Registry default, Registry.EM_COMMON:2402) leaves every selector as
    written, so a -1 there is still refused by the sentinel census.
    """
    physics = inp.get("physics")
    if not physics or "physics_suite" not in physics:
        return []
    raw = physics.pop("physics_suite")
    values = raw if isinstance(raw, list) else [raw]
    suite = str(values[0]).strip().lower() if values else "none"
    drop("physics", "physics_suite", values,
         "expanded into explicit selectors exactly as WRF's "
         "setup_physics_suite does (share/module_check_a_mundo.F)")
    if suite == "none":
        return []
    table = WRF_PHYSICS_SUITES.get(suite)
    if table is None:
        raise _err("physics", "physics_suite", values,
                   f"is not a WRF physics suite; WRF v4.6.1 knows "
                   f"{sorted(WRF_PHYSICS_SUITES)} and 'none', and stops "
                   "on any other name (share/module_check_a_mundo.F:3148)")
    recorded = []
    for key, suite_value in table.items():
        column = physics.get(key)
        if column is None:
            continue
        expanded = [suite_value if value == -1 else value for value in column]
        if expanded == list(column):
            continue
        physics[key] = expanded
        recorded.append(NamelistDefault(
            "input", "physics", key, suite_value,
            f"physics_suite '{suite}' resolves -1 to {suite_value} "
            "(WRF v4.6.1 share/module_check_a_mundo.F:"
            f"{_WRF_PHYSICS_SUITE_LINES[suite]})"))
    return recorded


def _wrf_domain_steps(root_step: Fraction, parent_id, tratio) -> list[Fraction]:
    """Each domain's exact step: the root's, divided down the parent chain."""
    steps: list[Fraction] = []
    for n, parent in enumerate(parent_id):
        if n == 0 or parent in (0, None):
            steps.append(Fraction(root_step))
        else:
            steps.append(steps[int(parent) - 1] / int(tratio[n]))
    return steps


def _wrf_cadence_minutes(key: str, n: int, minutes: float, step: Fraction,
                         fix) -> float:
    """A WRF physics cadence in minutes, as the steps wrf.exe takes.

    WRF never runs a fractional cadence: phys/module_physics_init.F:2148
    sets ``STEPRA = MAX(NINT(RADT*60./DT), 1)`` (STEPCU and STEPBL alike)
    and fires the scheme on whole steps.  Named breakage: NCAR's v4.4
    CONUS 12 km namelist writes radt = 15 at a 72 s step (12.5 steps,
    which wrf.exe runs as 13) and was refused.  A commensurate cadence is
    returned unchanged; otherwise the minutes of WRF's whole-step count,
    recorded as a fix naming both numbers.  When those minutes have no
    exact decimal spelling the namelist's own value is kept: since 2.9.10
    the loader and clock round it to the same whole steps wrf.exe takes
    (woof.experiment.wrf_physics_cadences), so the refusal this branch
    used to raise is retired.
    """
    if minutes <= 0.0:
        return minutes
    exact = Fraction(str(minutes)) * 60 / step
    if exact.denominator == 1:
        return minutes
    # Fortran NINT rounds half away from zero; every value here is positive.
    whole = max(int(exact + Fraction(1, 2)), 1)
    effective = Fraction(whole) * step / 60
    value = float(effective)
    if Fraction(str(value)) != effective:
        return minutes
    fix("physics", f"{key}[{n + 1}]", [minutes], value,
        f"{minutes:g} min is {exact} steps of {_seconds_text(step)}; wrf.exe "
        f"runs whole steps, NINT = {whole} (phys/module_physics_init.F:2148), "
        f"which is {value:g} min")
    return value


def _wrf_restart_interval_s(seconds: float, run_seconds: float,
                            step: Fraction, adaptive: bool, fix) -> float:
    """The d01 restart alarm as the steps wrf.exe writes restart files on.

    WRF's repeating alarm rings at the first step at or after each whole
    interval and then advances by the interval, not to the step it rang on
    (external/esmf_time_f90/ESMF_Clock.F90, ESMF_ClockAdvance: PRED3 and
    ``PrevRingTime = PrevRingTime + RingInterval``).  So an interval longer
    than the run never rings, and an interval that is not a whole number
    of steps still writes, on WRF's next step.  woof writes restart files
    on a fixed whole-step cadence and refused every interval that was not
    a whole number of steps.  Named breakage: NCAR's v4.4 CONUS 12 km
    namelist with the common ``restart_interval = 99999`` (no restart
    file) at its 72 s step was refused at import.

    A whole-step interval is returned unchanged.  An interval past the run
    end imports as 0 (no restart file, as in wrf.exe).  Otherwise, when
    WRF's ring steps inside the run are a fixed whole-step cadence, that
    cadence is returned and recorded as a fix; when they are not (WRF
    writes on an uneven step pattern woof's fixed cadence cannot
    reproduce), the interval is refused naming WRF's steps.
    """
    if seconds <= 0.0:
        return seconds
    interval = Fraction(str(seconds))
    if (interval / step).denominator == 1:
        return seconds
    window = Fraction(str(run_seconds))
    minutes = float(interval / 60)
    if interval > window:
        fix("time_control", "restart_interval", [minutes], 0.0,
            f"{minutes:g} min is past the {float(window / 60):g} min run, "
            "so wrf.exe's restart alarm never rings and it writes no "
            "restart file (ESMF_Clock.F90 rings at PrevRingTime + "
            "RingInterval); imported as 0, no restart file")
        return 0.0
    if adaptive:
        # Adaptive steps vary, so WRF's ring steps are not known here; the
        # experiment admission keeps its whole-step check.
        return seconds
    cadence = math.ceil(interval / step)
    rings = [math.ceil(k * interval / step)
             for k in range(1, int(window // interval) + 1)]
    effective = cadence * step
    value = float(effective)
    if (any(ring != (k + 1) * cadence for k, ring in enumerate(rings))
            or Fraction(str(value)) != effective):
        raise _err("time_control", "restart_interval", [minutes],
                   f"{minutes:g} min is {interval / step} steps of "
                   f"{_seconds_text(step)}; wrf.exe writes restart files on "
                   f"steps {', '.join(map(str, rings[:6]))}"
                   f"{', ...' if len(rings) > 6 else ''} (the first step at "
                   "or after each interval, ESMF_Clock.F90), which is not a "
                   "fixed step cadence and woof writes restarts on one; "
                   "write an interval that is a whole number of steps")
    fix("time_control", "restart_interval", [minutes], float(effective / 60),
        f"{minutes:g} min is {interval / step} steps of "
        f"{_seconds_text(step)}; wrf.exe writes its restart files inside "
        f"the run on steps {', '.join(map(str, rings))} (the first step at "
        "or after each interval, ESMF_Clock.F90), every "
        f"{cadence} steps, which is {float(effective / 60):g} min")
    return value


def _run_length_seconds(time_control: dict) -> float:
    """WRF's run_days/hours/minutes/seconds, summed (0 when unset)."""
    total = 0.0
    for key, scale in (("run_days", 86400), ("run_hours", 3600),
                       ("run_minutes", 60), ("run_seconds", 1)):
        values = time_control.get(key)
        if values:
            try:
                total += float(values[0]) * scale
            except (TypeError, ValueError):
                return 0.0
    return total


def _census_optional(section: str, wps: dict, inp: dict) -> frozenset:
    """Keys the census may not demand, because the translation does not
    read them for this pair.

    * Mercator and polar stereographic ignore truelat2 (and Mercator
      stand_lon) in WPS; the translation fills them from truelat1/ref_lon,
      so demanding them refused a pair WRF runs.
    * WRF takes the run length from run_* when it is positive and never
      reads end_* then; the translation derives the end from start + run_*.
    """
    if section == "geogrid":
        projection = wps.get("geogrid", {}).get("map_proj", ["lambert"])
        if str(projection[0]).strip().lower() != "lambert":
            return frozenset({"truelat2", "stand_lon"})
    if section == "time_control":
        if _run_length_seconds(inp.get("time_control", {})) > 0.0:
            return frozenset({"end_year", "end_month", "end_day"})
    return frozenset()


def _census_missing_keys(wps: dict, inp: dict,
                         wps_path, input_path) -> None:
    """One report of every missing required key across both namelists."""

    parsed = {"wps": (wps, wps_path), "input": (inp, input_path)}
    lines = []
    for (role, section), keys in _REQUIRED_KEYS.items():
        document, path = parsed[role]
        entries = document.get(section, {})
        keys = tuple(key for key in keys
                     if key not in _census_optional(section, wps, inp))
        absent = sorted(key for key in keys if key not in entries)
        if absent:
            lines.append(f"  &{section} of {path}: {', '.join(absent)}")
            lines.extend(f"    {key}: {_REQUIRED_KEY_REASONS[(role, section)][key]}"
                         for key in absent)
    if lines:
        raise ValueError(
            "the namelist pair is missing required key(s):\n"
            + "\n".join(lines)
            + "\nadd every listed key and re-run (this is the complete "
            "missing-key inventory, not the first failure).")


def _misplaced_key_hint(section: str, entries) -> str:
    """Where WRF itself reads each unmapped key, when that is elsewhere.

    Only for namelist.input groups WRF's Registry declares: the WPS groups
    and &namelist_quilt are read by other programs.
    """
    from woof.wrf_namelist_registry import placement_note, wrf_namelist_keys

    if section not in {group for group, _ in wrf_namelist_keys()}:
        return ""
    notes = [note for note in (placement_note(section, key)
                               for key in sorted(entries)) if note]
    return (" " + "; ".join(notes) + ".") if notes else ""


#: When set to a set, every key a translation ASKS a section for -- found
#: or not -- is added as ``(section, key)``.  woof.namelist_contract reads
#: the importer's vocabulary this way instead of keeping a second copy of
#: it; nothing else sets it, so an ordinary import records nothing.
_TAKE_TRACE: contextvars.ContextVar = contextvars.ContextVar(
    "namelist_take_trace", default=None)


# Keys unique to the operational WRF 3.9 fork's Registry.  Their presence
# identifies the requested configuration even when the namelist is renamed.
OPERATIONAL_FORK_ONLY_KEYS = {
    "time_control": ("gsd_diagnostics", "history_interval2",
                     "history_interval_change", "diag_int", "wind_int"),
    "physics": ("alb_sol", "prec_acc_dt1", "mp_tend_radar",
                "bl_mynn_tkebudget"),
    "dynamics": ("diff_6th_factor2",),
}

OPERATIONAL_FORK_THOMPSON_DEFAULTS = {
    "thompson_version": "wrf_39_noaa",
    "thompson_fork_snow_fall": "wrf_39_noaa",
}


# Explicit configuration-door choices. Data source and ordinary RunConfig
# defaults do not select these values.
OPERATIONAL_FORK_RUC_DEFAULTS = {
    "ruc_soilprop": "wrf_45",
    "ruc_irrigation": "wrf_45",
    "ruc_snow": "wrf_45",
    "ruc_qvg_cold_start": "air",
    "ruc_2m_diagnostic": "log_profile",
}


def operational_fork_signature(sections):
    """Fork-only namelist keys, independent of the filename and data source."""
    return sorted(
        f"&{section} {key}"
        for section, keys in OPERATIONAL_FORK_ONLY_KEYS.items()
        for key in keys if key in sections.get(section, {}))


def operational_fork_ruc_defaults(sections):
    """RUC lineage values when a namelist requests the operational fork."""
    values = sections.get("physics", {}).get("sf_surface_physics", [])
    if values and values[0] == 3 and operational_fork_signature(sections):
        return dict(OPERATIONAL_FORK_RUC_DEFAULTS)
    return {}


def operational_fork_thompson_defaults(sections):
    """Owned selector values when the namelist requests the fork's MP28."""
    values = sections.get("physics", {}).get("mp_physics", [])
    if (values and values[0] == 28
            and operational_fork_signature(sections)):
        return dict(OPERATIONAL_FORK_THOMPSON_DEFAULTS)
    return {}


class _Section:
    """One namelist section with consume-or-fail key accounting."""

    def __init__(self, name: str, entries: dict[str, list], source: str):
        self.name = name
        self.entries = dict(entries)
        self.source = source
        #: Keys actually popped from the namelist, in consumption order --
        #: the raw material of the report's Translated section (keys later
        #: recorded as fixed/dropped are subtracted there).
        self.consumed: list[str] = []
        # The refusal belongs to the loader that owns these keys, and it
        # is raised here from the loader: the copy that stood here said
        # "woof implements static nests only", which is not true -- a
        # nest that follows weather is shipped, expressed as DISCRETE
        # [relocation] rather than as per-step namelist keys -- so the
        # importer, the loader and the support report each described one
        # namelist differently.  The location the loader renders is
        # TOML-shaped ("[domains] of ..."); making it namelist-shaped
        # needs a keyword on woof/experiment.py:1530, out of this lane.
        _reject_moving_nest_keys(name, self.entries, source)

    def take(self, key: str, default=None) -> list | None:
        trace = _TAKE_TRACE.get()
        if trace is not None:
            trace.add((self.name, key))
        if key in self.entries:
            self.consumed.append(key)
            return self.entries.pop(key)
        return default

    def col(self, key: str, n: int, default=None) -> list | None:
        """Per-domain column using the importer's ratified last-value fill.

        This is not generic Fortran namelist behavior. Registry arrays whose
        omitted tail must retain its initialized default use ``registry_col``.
        """
        values = self.take(key)
        if values is None:
            return None if default is None else [default] * n
        if len(values) > n:
            values = values[:n]
        return values + [values[-1]] * (n - len(values))

    def registry_col(self, key: str, n: int, default) -> list:
        """Per-domain column whose omitted tail keeps its Registry default.

        Fortran namelist assignment does not broadcast a scalar across an
        array: elements not named by the input retain their initialized WRF
        Registry values.  Use this for Registry arrays where that distinction
        changes the resolved experiment rather than for columns WRF explicitly
        normalizes elsewhere.
        """
        values = self.take(key)
        if values is None:
            return [default] * n
        values = values[:n]
        return values + [default] * (n - len(values))

    def scalar(self, key: str, default=None):
        values = self.take(key)
        if values is None:
            return default
        return values[0]

    def required(self, key: str):
        values = self.take(key)
        if values is None:
            raise ValueError(
                f"&{self.name} of {self.source} is missing required "
                f"key {key}.")
        return values[0]

    def finish(self) -> None:
        if self.entries:
            raise NamelistRefusal(
                f"unmapped key(s) {sorted(self.entries)} in &{self.name} "
                f"of {self.source}: the importer refuses to drop namelist "
                "settings silently -- extend the ratified map or remove "
                "the key(s)." + _misplaced_key_hint(self.name, self.entries),
                [(self.name, key) for key in sorted(self.entries)])


def _uniform(section: str, key: str, values: list):
    first = values[0]
    if any(v != first for v in values[1:]):
        raise _err(section, key, values,
                   "per-domain values must be identical (woof shares "
                   "this setting across domains).")
    return first


def _require_bools(section: str, key: str, values: list) -> list[bool]:
    """Fortran-logical columns must decode to real booleans -- a residual
    string (e.g. an unparsed token) must never be truth-tested (shadow
    review S2's silent false->true hazard)."""
    for v in values:
        if not isinstance(v, bool):
            raise _err(section, key, v,
                       "must be a Fortran logical (.true./.false.).")
    return [bool(v) for v in values]


def _fmt(value) -> str:
    """TOML literal for a scalar value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat(sep="T")
    return '"' + str(value) + '"'


#: The fire fields that are not ``&fire`` keys: the refinement ratios
#: live in ``&domains`` (read below), and the static, fuel-table and
#: smoke switches are woof extensions WRF has no key for.
_FIRE_NON_FIRE_GROUP_FIELDS = frozenset(
    {"sr_x", "sr_y", "fire_static", "fire_fuel_namelist", "fire_smoke"})


def imported_fire_fields() -> tuple[str, ...]:
    """Every ``&fire`` key :func:`_fire_columns` reads, by table.

    The importer reads the WRF v4.7.1 fire group through
    :data:`woof.sfire_config.FIRE_CONFIG_FIELDS` rather than one key at a
    time, so these names are spelled in a pair although none of them is
    written out in this module.  tests/test_route_round_trip_fields.py
    reads this to tell a table-read field from one the pair cannot carry.
    """
    from woof.sfire_config import FIRE_CONFIG_FIELDS
    return tuple(name for name in FIRE_CONFIG_FIELDS
                 if name not in _FIRE_NON_FIRE_GROUP_FIELDS)


def _fire_columns(fire, domains, count):
    """Translate Registry fire columns without broadcasting omitted tails."""
    from woof.sfire_config import FireRunFields, FIRE_DOMAIN_FIELDS
    defaults = FireRunFields()
    result = {}
    for name in imported_fire_fields():
        raw = fire.take(name)
        if raw is None:
            continue
        default = getattr(defaults, name)
        for value in raw:
            valid = (isinstance(value, bool) if isinstance(default, bool) else
                     isinstance(value, int) and not isinstance(value, bool) if isinstance(default, int) else
                     isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value))
            if value is not None and not valid:
                raise _err("fire", name, raw, f"must use the Registry {type(default).__name__} type")
        supplied = [default if value is None else value for value in raw]
        if name in FIRE_DOMAIN_FIELDS:
            result[name] = supplied[:count] + [default] * max(0, count - len(supplied))
        else:
            if len(supplied) != 1:
                raise _err("fire", name, raw, "is a scalar shared by all domains in WRF")
            result[name] = supplied * count
    for name in ("sr_x", "sr_y"):
        raw = domains.take(name)
        if raw is None:
            continue
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0) for value in raw):
            raise _err("domains", name, raw, "must contain nonnegative integer refinement ratios")
        result[name] = [0 if value is None else value for value in raw[:count]] + [0] * max(0, count - len(raw))
    fire.finish()
    return result


# ---------------------------------------------------------------------------
# &geogrid geog_data_res: which static datasets a namelist asks for
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StaticLandcover:
    """The ``[static.highres]`` land cover an imported namelist selects.

    ``source_id`` is the row of
    :data:`woof.static.highres_fetch.LANDCOVER_SOURCES` whose
    ``wps_geog_tokens`` the namelist's ``geog_data_res`` names;
    ``grid_ids`` the domains that name it and ``max_dx_m`` the block's
    spacing scope (``None``: every domain).  ``values`` are the per-domain
    ``geog_data_res`` strings as WPS reads them, for the report.
    """

    source_id: str
    grid_ids: tuple[int, ...]
    max_dx_m: float | None
    values: tuple[str, ...]


def geog_data_res_tokens() -> dict[str, str]:
    """Every WPS ``geog_data_res`` token woof builds, and how.

    ``{token: "geog"}`` for the GEOG-tree selectors the static builder
    resolves (:class:`woof.static.build.GeogSelection`), and
    ``{token: <landcover_source id>}`` for a land-cover collection built
    through ``[static.highres]``.  Both halves are tables owned elsewhere;
    a token reaches this map by a row there, never by a branch here.
    """
    from woof.static.build import _SUPPORTED_GEOG_TOKENS
    from woof.static.highres_fetch import landcover_source_ids_by_wps_token

    tokens = {token: "geog" for token in sorted(_SUPPORTED_GEOG_TOKENS)}
    tokens.update(landcover_source_ids_by_wps_token())
    return tokens


def _geog_thirty_second_notice(values, grid_ids: list[int], *,
                               highres_landcover: str | None = None) -> str | None:
    """Declare the field identities a bare WPS ``30s`` token selects."""
    if values is None:
        return None
    from woof.static.build import (
        GeogSelection, _DEFAULT_GEOG_DIRS, _THIRTY_SECOND_GEOG_DIRS,
    )
    entries = []
    for grid_id, value in zip(grid_ids, values):
        tokens = [part.strip().lower() for part in str(value).split("+")]
        if "30s" not in tokens:
            continue
        selection = GeogSelection.from_tokens(
            ".", str(value), highres_landcover=highres_landcover)
        entries.append(
            f"d{grid_id:02d}: " + "; ".join(
                f"{field}={getattr(selection, field)}" for field in _DEFAULT_GEOG_DIRS))
    if not entries:
        return None
    explicit = ", ".join(_THIRTY_SECOND_GEOG_DIRS)
    return (
        "Static geography &geogrid geog_data_res '30s' follows WPS 4.6 "
        f"GEOGRID.TBL.ARW: the bare token selects {explicit}; fields without "
        "that selector fall through to the next token or WPS's default "
        "dataset, rather than acquiring a different source resolution. "
        + " | ".join(entries)
        + ". The static builder reads these datasets from [case_data] "
        "geog_root; no dataset substitution is applied.")


def _geog_data_res_landcover(values, *, max_dom: int, grid_ids: list[int],
                             spacings: list[float]
                             ) -> StaticLandcover | None:
    """Read ``geog_data_res`` the way WPS does and say what it builds.

    WPS v4.6 gridinfo_module.F initializes every element to ``'default'``
    and the namelist replaces only the elements listed, so an unlisted
    domain is ``'default'`` (not the last value repeated).  Each value is a
    ``+``-joined token list.  WPS applies a land-cover table entry wherever
    its token appears in the list (GEOGRID.TBL.ARW_LCZ's priority-2
    LANDUSEF entry knows only ``cglc_modis_lcz``, so ``default`` before it
    does not shadow it).

    Returns ``None`` when no domain names a land-cover collection.  A token
    woof builds from nothing is refused by name: building another dataset
    under its name would be a silent substitution.
    """
    if values is None:
        return None
    known = geog_data_res_tokens()
    column = [str(value) for value in values[:max_dom]]
    column += ["default"] * (max_dom - len(column))
    unknown: list[str] = []
    selected: dict[int, str] = {}
    for n, value in enumerate(column):
        tokens = [token.strip().lower() for token in value.split("+")
                  if token.strip()]
        unknown += [token for token in tokens
                    if token not in known and token not in unknown]
        sources = sorted({known[token] for token in tokens
                          if known.get(token, "geog") != "geog"})
        if len(sources) > 1:
            raise _err(
                "geogrid", "geog_data_res", values,
                f"d{grid_ids[n]:02d}'s {value!r} names {len(sources)} "
                f"land-cover collections ({', '.join(sources)}); one domain "
                "builds its land use from one, so which one this namelist "
                "means cannot be read from it.  Keep one token.")
        if sources:
            selected[n] = sources[0]
    if unknown:
        geog = sorted(token for token, how in known.items() if how == "geog")
        landcover = sorted(token for token, how in known.items()
                           if how != "geog")
        raise _err(
            "geogrid", "geog_data_res", values,
            f"token(s) {unknown} name GEOG dataset(s) woof does not "
            f"build.  It builds {geog} from the GEOG tree (read from the "
            "namelist.wps that [case_data] wps_namelist names) and "
            f"{landcover} through [static.highres]; building a domain from "
            "another dataset under the name the namelist gives would be a "
            "silent substitution.  Replace or remove the token(s).")
    if not selected:
        return None
    source_ids = sorted(set(selected.values()))
    if len(source_ids) > 1:
        raise _err(
            "geogrid", "geog_data_res", values,
            f"the domains name {len(source_ids)} land-cover collections "
            f"({', '.join(source_ids)}); one [static.highres] block "
            "carries one landcover_source.  Keep one collection.")
    inside = sorted(selected)
    outside = [n for n in range(max_dom) if n not in selected]
    max_dx_m = None
    if outside:
        finest_outside = min(spacings[n] for n in outside)
        coarsest_inside = max(spacings[n] for n in inside)
        if coarsest_inside >= finest_outside:
            raise _err(
                "geogrid", "geog_data_res", values,
                f"{source_ids[0]} is named on "
                f"{', '.join(f'd{grid_ids[n]:02d}' for n in inside)} but "
                f"not on {', '.join(f'd{grid_ids[n]:02d}' for n in outside)}, "
                "and [static.highres] reaches domains by grid spacing "
                f"(max_dx_m) alone: a domain at {coarsest_inside:g} m names "
                f"it while one at {finest_outside:g} m does not, so no "
                "spacing bound builds it on exactly those domains.  Name "
                "it on every domain at or below one spacing.")
        max_dx_m = coarsest_inside
    return StaticLandcover(
        source_id=source_ids[0],
        grid_ids=tuple(grid_ids[n] for n in inside),
        max_dx_m=max_dx_m, values=tuple(column))


def _innermost_int(section, key: str, default: int, max_dom: int):
    """The innermost domain's value of a per-domain integer, read without
    consuming the key (WRF resets every domain to it,
    share/module_check_a_mundo.F:1063-1077); ``None`` when it is not an
    integer, which the key's own reader refuses later."""
    raw = section.entries.get(key)
    if raw is None:
        return default
    values = list(raw[:max_dom]) + [raw[-1]] * max(0, max_dom - len(raw))
    value = values[-1]
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _built_land_categories(static_landcover: StaticLandcover | None, *,
                           grid_ids, urban, use_wudapt_lcz, geog_data_res
                           ) -> tuple[dict[int, int], str]:
    """``({grid_id: NUM_LAND_CAT the static build stamps}, why)``.

    The GEOG tree's land use is the MODIS set geog_data_res selects: 21
    categories with the lake class (the default), 20 for the 15
    arc-second set ``modis_15s`` names, as geogrid stamps; a domain whose
    land cover ``[static.highres]`` builds takes that collection's legend
    under the run's urban canopy (:func:`woof.static.highres.
    landcover_legend`, the one authority the static build itself calls).
    A legend the canopy cannot run is refused here with that function's
    reason, which is WRF's own stop, instead of at preparation.
    """
    from woof.static.build import (geog_landuse_category_count,
                                    geog_landuse_dataset)
    from woof.static.highres import MODIS21_CATEGORY_COUNT, landcover_legend
    from woof.static.highres_fetch import LANDCOVER_SOURCES

    # The GEOG land use each domain's geog_data_res selects (WPS reads an
    # unlisted domain as 'default'): the lake inventories carry 21
    # categories, the 15 arc-second MODIS set (modis_15s) 20.
    column = [str(value) for value in (geog_data_res or [])][:len(grid_ids)]
    column += ["default"] * (len(grid_ids) - len(column))
    counts = {int(gid): geog_landuse_category_count(value)
              for gid, value in zip(grid_ids, column)}
    geog_note = "; ".join(
        f"{dataset} ({geog_landuse_category_count(value)} categories) on "
        + ", ".join(f"d{int(gid):02d}" for gid, v in zip(grid_ids, column)
                    if geog_landuse_dataset(v) == dataset)
        for dataset, value in {geog_landuse_dataset(v): v
                               for v in column}.items())
    if static_landcover is None:
        if set(counts.values()) == {MODIS21_CATEGORY_COUNT} and all(
                geog_landuse_dataset(v) == geog_landuse_dataset("default")
                for v in column):
            return counts, ("the GEOG tree's MODIS land use on every domain, "
                            f"{MODIS21_CATEGORY_COUNT} categories")
        return counts, f"the GEOG tree's MODIS land use: {geog_note}"
    row = LANDCOVER_SOURCES[static_landcover.source_id]
    urban = urban or 0
    lcz = use_wudapt_lcz or 0
    try:
        _, count = landcover_legend(static_landcover.source_id,
                                    sf_urban_physics=urban,
                                    use_wudapt_lcz=lcz)
    except ValueError as error:
        raise _err(
            "geogrid", "geog_data_res", geog_data_res,
            f"it selects {row.label} land cover, and with the urban canopy "
            f"on (sf_urban_physics = {urban}) and use_wudapt_lcz = {lcz} "
            f"its built classes have no urban parameter table: {error}.  "
            "Set use_wudapt_lcz = 1, or remove the "
            f"{'/'.join(row.wps_geog_tokens)} token to keep the MODIS "
            f"land use ({MODIS21_CATEGORY_COUNT} categories, so "
            f"num_land_cat = {MODIS21_CATEGORY_COUNT}).") from None
    for gid in static_landcover.grid_ids:
        counts[int(gid)] = int(count)
    where = ", ".join(f"d{gid:02d}" for gid in static_landcover.grid_ids)
    if int(count) > MODIS21_CATEGORY_COUNT:
        note = (f"{row.label} land cover on {where}, whose built classes "
                f"the urban canopy (sf_urban_physics = {urban}"
                + (", use_wudapt_lcz = 1" if lcz else "")
                + f") keeps as categories 51-{int(count)}")
    else:
        note = (f"{row.label} land cover on {where}, its built classes "
                "collapsed to MODIS urban (13) because no urban canopy runs "
                "(sf_urban_physics = 0)")
    if len(static_landcover.grid_ids) < len(counts):
        outside = [v for gid, v in zip(grid_ids, column)
                   if gid not in static_landcover.grid_ids]
        if all(geog_landuse_dataset(v) == geog_landuse_dataset("default")
               for v in outside):
            note += ("; the GEOG tree's MODIS land use, "
                     f"{MODIS21_CATEGORY_COUNT} categories, elsewhere")
        else:
            note += ("; the GEOG tree's MODIS land use elsewhere: "
                     + "; ".join(
                         f"{geog_landuse_dataset(v)} "
                         f"({geog_landuse_category_count(v)} categories) "
                         f"on d{int(gid):02d}"
                         for gid, v in zip(grid_ids, column)
                         if gid not in static_landcover.grid_ids))
    return counts, note


def _static_landcover_block(selection: StaticLandcover, *, cache_root,
                            legend_note: str) -> tuple[list[str], str]:
    """The ``[static.highres]`` lines an imported land-cover token becomes,
    and the report notice that says what else that block replaces.

    ``fields = "all"`` is the block's only mode that builds land cover, and
    it replaces terrain and soil texture too, which WPS's land-cover token
    does not: that is stated, never left for the reader to find.  The
    cache folder is the per-user one the engine's own default terrain row
    uses unless the caller names another (``--static-cache-root``).
    """
    from woof.static.highres_fetch import LANDCOVER_SOURCES
    from woof.static.highres_production import default_highres_cache_root

    row = LANDCOVER_SOURCES[selection.source_id]
    root = (default_highres_cache_root().as_posix() if cache_root is None
            else (Path(cache_root).as_posix()
                  if isinstance(cache_root, Path) else str(cache_root)))
    if len(set(selection.values)) == 1:
        named = f"'{selection.values[0]}' on every domain"
    else:
        named = ", ".join(f"'{value}'" for value in selection.values)
    where = ("every domain" if selection.max_dx_m is None else
             ", ".join(f"d{gid:02d}" for gid in selection.grid_ids)
             + f" (max_dx_m = {selection.max_dx_m:g}, the domains at or "
               "below that spacing)")
    replaces = (
        "That block also replaces terrain (terrain_source \"auto\": USGS "
        "3DEP 1/3 arc-second inside the conterminous United States, "
        "Copernicus DEM GLO-30 elsewhere, where WPS reads GMTED2010 30 s) "
        "and soil texture (SoilGrids v2 250 m, where WPS reads the 30 s "
        "soil category tables), and recomputes TMN from the new terrain; "
        "the monthly climatologies stay on the GEOG tree.")
    notice = (
        f"Static land cover imported from &geogrid geog_data_res = {named}: "
        f"[static.highres] enabled = true, fields = \"all\", "
        f"landcover_source = \"{selection.source_id}\" ({row.label}, "
        f"{row.coverage.nominal_resolution}, representing "
        f"{row.first_year}"
        + (f"-{row.last_year}" if row.last_year != row.first_year else "")
        + f") on {where}; {legend_note}.  {replaces}  The GEOG-tree "
        "tokens beside it are read from the namelist.wps that [case_data] "
        f"wps_namelist names.  Its tiles are cached in {root}.")
    lines = [
        "",
        "# &geogrid geog_data_res "
        + ", ".join(f"'{value}'" for value in selection.values) + ":",
        f"# {row.label} land cover, built by the high-resolution static "
        "block.  fields = \"all\"",
        "# also replaces terrain and soil texture, which WPS's token does "
        "not (see the",
        "# import report).  The other tokens are read from the namelist.wps "
        "that",
        "# [case_data] wps_namelist names.",
        "[static.highres]",
        "enabled = true",
        f"cache_root = {_toml_string(root)}",
        'fields = "all"',
        f"landcover_source = {_toml_string(selection.source_id)}",
    ]
    if selection.max_dx_m is not None:
        lines.append(f"max_dx_m = {_fmt(float(selection.max_dx_m))}")
    return lines, notice


def _toml_string(value: str) -> str:
    """A TOML basic string (JSON's escapes are TOML's)."""
    import json

    return json.dumps(str(value), ensure_ascii=True)


# ---------------------------------------------------------------------------
# The importer
# ---------------------------------------------------------------------------

def _default_rrtmg_variant_for_namelist(inp: dict, input_path=None,
                                      source_text: str | None = None) -> str:
    physics = inp.get("physics", {})
    requested_aerosol = physics.get("aer_opt", ())
    # aer_opt 1, 2 and 3 exist only on the legacy RRTMG shortwave.
    if requested_aerosol and all(type(value) is int and value in (1, 2, 3) for value in requested_aerosol):
        return RRTMG_VARIANT_LEGACY
    budget = physics.get("bl_mynn_tkebudget")
    pbl = physics.get("bl_pbl_physics", [])
    if input_path is not None and 5 in pbl:
        from woof.physics_source_defaults import (
            namelist_physics_defaults, read_physics_selector_comment,
        )
        selected = namelist_physics_defaults(input_path)
        selected.update(read_physics_selector_comment(source_text or ""))
        if selected.get("bl_mynn_version") == "gsd_41":
            return RRTMG_VARIANT_LEGACY
        if selected.get("bl_mynn_version") == "wrf_461":
            return RRTMG_VARIANT_RTE_RRTMGP
    if budget is not None and all(type(value) is int and value == 0 for value in budget) \
            and 5 in pbl:
        return RRTMG_VARIANT_LEGACY
    return RRTMG_VARIANT_RTE_RRTMGP


def import_namelists(wps_path: str | Path, input_path: str | Path,
                     name: str | None = None,
                     urban_ucm_generation: str | None = None,
                     urban_controls: Mapping[str, object] | None = None,
                     rrtmg_variant: str | None = _AUTO_RRTMG_VARIANT,
                     rrtmg_compatibility: str | None = None,
                     acknowledgements: tuple[str, ...] = (),
                      landuse_identity: Mapping[str, object] | None = None,
                      wrf_boundary_use_theta_m: int | None = None,
                      wrfinput_qv_domains: tuple[int, ...] = (),
                      wrfinput_radar_tendency_domains: tuple[int, ...] = (),
                     static_cache_root: str | Path | None = None,
                     metgrid_initialization: bool = False,
                     geogrid_tbl: str | Path | None = None,
                     terrain_smoothing_precision: str | None = None,
                     wrf_version: str = WRF_VERSION_DEFAULT,
                     wrf_version_source: str | None = None,
                     request_source: str | None = None,
                     preserve_wrf_mixing: bool = False
                     ) -> tuple[str, SubstitutionReport]:
    """Translate namelist.wps + namelist.input into (TOML text, report).

    ``wrf_version`` names the WRF line the namelist was written for ("3"
    or "4", the CLI ``--wrf-version`` flag) and selects only the Registry
    default an OMITTED ``&dynamics/use_theta_m`` takes
    (:func:`omitted_use_theta_m`): 0 on the V3 line, 1 on V4.
    ``wrf_version_source`` says what chose that line (the flag, the
    producing files' ``TITLE``); the report records both, and the
    "Namelist defaults" row an omitted key books cites that line's
    Registry.

    ``request_source`` binds an authored route to the existing source
    request-default table even when its output filename is arbitrary.
    The ordinary file and parsed imports leave it unset, retaining their
    historical defaults. Supplied stock keys and generation selectors win.

    ``preserve_wrf_mixing`` keeps WRF's declared/default mixing logicals
    for the native wrfinput door. Its real-data base-profile check establishes
    support for both values; ordinary translation keeps its existing policy.

    ``name`` sets ``[experiment].name`` (the CLI ``--name`` flag);
    the default is derived deterministically from the start time and
    domain count.  ``rrtmg_variant`` selects which woof implementation a
    WRF RRTMG 4/4 request maps to. An omitted argument follows the named
    source's radiation generation, otherwise the established
    ``"rte-rrtmgp"`` substitution with byte-identical ordinary output.
    Explicit ``None`` retains WRF's legacy RRTMG selection; explicit
    ``"rrtmg_legacy"`` selects that same port and compatibility token.
    An explicit mapping wins over the named default. An omitted
    ``aer_opt=3`` request selects the implemented legacy aerosol optics.
    The returned
    TOML is validated through :func:`woof.experiment.build_experiment`
    before being returned, so every section-A rule (root/child flags,
    ratios, clearance, vertical identity, cadence divisibility, derived
    dt/dx chain) binds at import time.

    ``rrtmg_compatibility`` is the SECOND fact a WRF namelist cannot
    spell about a 4/4 pair, and it was derived here instead of asked
    for.  The token is not a label: the RTE+RRTMGP arm reads it to
    choose its snow treatment and stamps it into the restart algorithm
    identity, so 'none' and the mapping token are two different runs of
    one selector pair.  Deriving it made every 4/4 import claim the
    mapping receipt, and a shipped suite that declares 'none' could not
    be written and read back as itself.  ``None`` keeps the derivation,
    so every established import stays byte-identical; a caller holding
    the authoritative configuration passes what that configuration says.

    ``acknowledgements`` carries declared-experiment acknowledgement ids
    into the emitted ``[experiment]`` table (the CLI ``--ack`` flag).  A
    WRF namelist has no spelling for a woof governance declaration, so
    without this an imported configuration that needs one -- e.g. a
    shortwave-on/longwave-off pairing across a window that includes
    local night -- would refuse validation HERE, before the TOML the
    reader could have added the declaration to even exists.  Empty adds
    nothing and keeps every existing import byte-identical.

    ``wrfinput_radar_tendency_domains`` identifies native input domains
    whose WRF input carries the cloud analysis's radar temperature tendency
    (``RAD_TTEN_DFI_1..4`` and ``TTEN_TIMES``).  ``mp_tend_radar = 1`` on
    those domains is applied from that tendency
    (:mod:`woof.ingest.wrfinput_radar`); on any other domain it is
    refused by name, because there is no tendency to apply.

    ``wrfinput_qv_domains`` identifies native input domains whose variable
    headers contain QVAPOR. Those domains retain a moisture state even
    with microphysics off; an empty tuple preserves the generic dry
    namelist contract for that selection.

    ``static_cache_root`` is the ``cache_root`` written into the
    ``[static.highres]`` block a land-cover ``geog_data_res`` token
    (``cglc_modis_lcz``) becomes (the CLI ``--static-cache-root`` flag);
    ``None`` writes the engine's per-user high-resolution cache, the folder
    its default terrain row already uses.  A relative path is resolved
    against the TOML's own folder when it is loaded.  A namelist naming no
    land-cover token writes no block and ignores it.
    """
    if rrtmg_variant not in (_AUTO_RRTMG_VARIANT, None, RRTMG_VARIANT_RTE_RRTMGP,
                             RRTMG_VARIANT_LEGACY):
        raise ValueError(
            f"rrtmg_variant must be '{RRTMG_VARIANT_RTE_RRTMGP}' or "
            f"'{RRTMG_VARIANT_LEGACY}', got {rrtmg_variant!r}")
    if rrtmg_compatibility is not None and rrtmg_compatibility not in (
            "none", *WRF_RRTMG_COMPATIBILITY_TOKENS):
        raise ValueError(
            "rrtmg_compatibility must be 'none' or one of "
            f"{sorted(WRF_RRTMG_COMPATIBILITY_TOKENS)}, got "
            f"{rrtmg_compatibility!r}")
    if not isinstance(metgrid_initialization, bool):
        raise TypeError("metgrid_initialization must be boolean")
    if metgrid_initialization and wrf_boundary_use_theta_m is not None:
        raise ValueError("metgrid initialization and restored WRF boundaries are distinct input contracts")
    wps_path, input_path = Path(wps_path), Path(input_path)

    wps = read_namelist_role(wps_path, "namelist.wps")
    # Capture one source snapshot for this file door. Passing it explicitly
    # keeps parsing and selectors on the same text without a carrier reread.
    try:
        input_text = input_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read namelist.input {input_path}: {error}") from None
    inp = parse_namelist_text(input_text, source=str(input_path))
    return import_parsed_namelists(
        wps, inp, wps_path=wps_path, input_path=input_path, name=name,
        source_text=input_text,
        urban_ucm_generation=urban_ucm_generation,
        urban_controls=urban_controls,
        rrtmg_variant=rrtmg_variant,
        rrtmg_compatibility=rrtmg_compatibility,
        acknowledgements=acknowledgements,
        landuse_identity=landuse_identity,
        wrf_boundary_use_theta_m=wrf_boundary_use_theta_m,
        wrfinput_qv_domains=wrfinput_qv_domains,
        wrfinput_radar_tendency_domains=wrfinput_radar_tendency_domains,
        static_cache_root=static_cache_root,
        metgrid_initialization=metgrid_initialization,
        geogrid_tbl=geogrid_tbl,
        terrain_smoothing_precision=terrain_smoothing_precision,
        wrf_version=wrf_version, wrf_version_source=wrf_version_source,
        request_source=request_source, preserve_wrf_mixing=preserve_wrf_mixing)


#: A refused physics selector the importer cannot simply drop (the key is
#: required, or dropping it moves the rest of the namelist onto another
#: code path) is replaced by the representative below while the rest of
#: the pair is checked.  The representatives are the ones WOOF's default
#: suite runs, so the branches a real suite takes stay the branches that
#: are checked; a later refusal whose text names a replaced key is a
#: consequence of the replacement, not of the user's namelist, and is not
#: reported.
_REFUSED_SELECTOR_STAND_INS = {
    ("physics", "mp_physics"): 8,
    ("physics", "bl_pbl_physics"): 1,
    ("physics", "sf_sfclay_physics"): 1,
    ("physics", "sf_surface_physics"): 2,
    ("physics", "cu_physics"): 0,
    ("physics", "ra_lw_physics"): 4,
    ("physics", "ra_sw_physics"): 4,
    ("dynamics", "km_opt"): 4,
    ("dynamics", "diff_opt"): 2,
    # A refused geog_data_res is checked as WPS's own default.  Which
    # land-use category count the pair builds hangs on it, so the
    # num_land_cat = 61 refusal a removed key would add (it names
    # geog_data_res) is that consequence, not a second fix to make: an
    # LCZ token's num_land_cat = 61 is right once the token imports.
    ("geogrid", "geog_data_res"): "default",
}

#: Keys a refusal can never be continued past: the geometry and the clock
#: every later check is computed from.
_UNREMOVABLE_KEYS = frozenset(
    (section, key)
    for (_role, section), keys in _REQUIRED_KEYS.items()
    for key in keys) - frozenset(_REFUSED_SELECTOR_STAND_INS)

#: The most settings one import sets aside before it stops looking.  A
#: namelist has a few hundred keys at most; the bound only keeps a
#: pathological file from looping.
_MAX_SET_ASIDE_ROUNDS = 256


def _without(document: dict, section: str, key: str) -> bool:
    """Remove ``key`` (or the whole section for ``*``) in place."""
    entries = document.get(section)
    if entries is None:
        return False
    if key == "*":
        del document[section]
        return True
    if key in entries:
        del entries[key]
        return True
    return False


def _format_refusals(problems: list[str]) -> str:
    lines = [
        f"the importer refuses {len(problems)} setting(s) in this namelist "
        "pair; every one is listed, so they can all be fixed before the "
        "next import:"]
    for index, problem in enumerate(problems, 1):
        lines.append(f"  {index}. {problem}")
    return "\n".join(lines)


def _refuse_latlon_grid(wps: dict) -> None:
    """Refuse a lat-lon namelist.wps by name, before anything else.

    A lat-lon grid (WPS 'lat-lon', WRF 6) is geometry every later check
    is computed from, so it is refused first and alone: checked further,
    its degree spacing and absent true latitudes surfaced as a missing
    ``truelat1`` or a dx/dy "mismatch", refusals that send the reader to
    fix the wrong thing.
    """
    from woof.static.projection import (implemented_projections,
                                         is_latlon_map_proj, latlon_blocker,
                                         latlon_refusal)

    value = wps.get("geogrid", {}).get("map_proj", ["lambert"])[0]
    if (isinstance(value, str) and is_latlon_map_proj(value)
            and value.strip().lower().replace("_", "-")
            not in implemented_projections()):
        raise _err("geogrid", "map_proj", value,
                   f"{latlon_refusal()}; {latlon_blocker()}.")


def import_parsed_namelists(wps: dict, inp: dict, *, wps_path, input_path,
                            **options) -> tuple[str, SubstitutionReport]:
    """:func:`import_namelists` on parsed documents, every refusal at once.

    The translation stops at its first refusal, so on its own a namelist
    with five unsupported settings needed five imports to discover them;
    on WOOF each import ran on a freshly booted GPU box.  This runs the
    translation, sets the refused setting(s) aside and runs it again until
    it succeeds or reaches a refusal it cannot continue past (a missing
    geometry key, a refusal that names no setting), then raises ONE
    ValueError listing every refusal it met, each with its own reason.

    Settings are set aside, never repaired: a set-aside key takes its WRF
    Registry default for the rest of the check, the way an omitted key
    does, and a refused physics selector is checked as its
    :data:`_REFUSED_SELECTOR_STAND_INS` representative.  A namelist with
    exactly one refusal raises that refusal unchanged.

    ``wps_path`` and ``input_path`` are provenance labels for parsed data.
    Pass ``source_text`` when the caller already has the input snapshot;
    it remains authoritative even if the input label names another file.
    Without that snapshot, an existing input file supplies its selector
    carrier for compatibility. Virtual labels need no backing file.
    Named source defaults still follow the input label.
    """

    from copy import deepcopy

    from woof.explain import muted_warnings

    _refuse_latlon_grid(wps)
    work_wps, work_inp = deepcopy(wps), deepcopy(inp)
    # 083ac7cd0, lane/sw-excess, made explicit source snapshots authoritative.
    # Retain that behavior while preserving 7ba801346, lane/286-fork-thompson:
    # a real-file carrier must still reach strict selector validation when
    # the parsed caller supplies no snapshot; a virtual label stays usable.
    if options.get("source_text") is None:
        carrier = Path(input_path)
        if carrier.is_file():
            try:
                options["source_text"] = carrier.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError) as error:
                raise ValueError(f"cannot read namelist.input selector carrier {carrier}: {error}") from None
    options.setdefault("rrtmg_variant", _AUTO_RRTMG_VARIANT)
    problems: list[str] = []
    collected = options.pop("_collect", None)
    first_error: BaseException | None = None
    stand_ins: dict[tuple[str, str], object] = {}
    for attempt in range(_MAX_SET_ASIDE_ROUNDS):
        try:
            if attempt == 0:
                return _translate_namelists(
                    work_wps, work_inp, wps_path=Path(wps_path),
                    input_path=Path(input_path), **options)
            with muted_warnings():
                _translate_namelists(
                    work_wps, work_inp, wps_path=Path(wps_path),
                    input_path=Path(input_path), **options)
            break
        except (ValueError, NotImplementedError) as error:
            text = str(error)
            if stand_ins and any(re.search(rf"\b{re.escape(key)}\b", text)
                                 for _, key in stand_ins):
                # A consequence of a stand-in, not of the namelist.
                keys = getattr(error, "keys", ())
                if not keys or not _set_aside(keys, work_wps, work_inp,
                                              stand_ins):
                    break
                continue
            if first_error is None:
                first_error = error
            if text not in problems:
                problems.append(text)
                if collected is not None:
                    collected.append(error)
            keys = getattr(error, "keys", ())
            if keys:
                if not _set_aside(keys, work_wps, work_inp, stand_ins):
                    break
                continue
            from woof.physics_compat import UnsupportedPhysicsSuiteError
            if isinstance(error, UnsupportedPhysicsSuiteError):
                # The suite preview refuses a combination, not one key:
                # check the rest of the pair with the representative suite.
                keys = [pair for pair in _REFUSED_SELECTOR_STAND_INS
                        if pair[1] in work_inp.get("physics", {})
                        and pair[1] in text]
                if keys and _set_aside(keys, work_wps, work_inp, stand_ins):
                    continue
            break
    if first_error is None:      # pragma: no cover - loop always records
        raise AssertionError("namelist import finished without a result")
    if len(problems) == 1:
        raise first_error
    raise NamelistRefusal(_format_refusals(problems))


def namelist_refusals(wps: dict, inp: dict, *, wps_path="namelist.wps",
                      input_path="namelist.input", **options) -> list:
    """Every refusal the importer gives a parsed pair, as exceptions.

    Empty when the pair imports.  Each element is the refusal the
    translation raised (a :class:`NamelistRefusal` carries the
    ``(section, key)`` pairs it names), in the order met; the same loop as
    :func:`import_parsed_namelists`, which raises them as one.
    """
    collected: list = []
    try:
        import_parsed_namelists(wps, inp, wps_path=wps_path,
                                input_path=input_path, _collect=collected,
                                **options)
    except (ValueError, NotImplementedError):
        if not collected:
            raise
    return collected


def _set_aside(keys, work_wps: dict, work_inp: dict,
               stand_ins: dict) -> bool:
    """Set the refused settings aside; False when one cannot be."""
    wps_sections = {"share", "geogrid", "ungrib", "metgrid"}
    progressed = False
    for section, key in keys:
        document = work_wps if section in wps_sections else work_inp
        pair = (section, key)
        if pair in _REFUSED_SELECTOR_STAND_INS:
            stand_in = _REFUSED_SELECTOR_STAND_INS[pair]
            column = document.get(section, {}).get(key)
            if pair in stand_ins or column is None:
                return False
            stand_ins[pair] = stand_in
            document[section][key] = [stand_in] * len(column)
            progressed = True
            continue
        if pair in _UNREMOVABLE_KEYS:
            return False
        progressed = _without(document, section, key) or progressed
    return progressed


def _seconds_text(step: Fraction) -> str:
    step = Fraction(step)
    if step.denominator == 1:
        return f"{step.numerator} s"
    return f"{step.numerator}/{step.denominator} s"


def _wrf_interval(whole: int, den: int) -> Fraction | None:
    """A WRF ``Sn/Sd`` clock bound as an exact rational; ``None`` for WRF's
    -1 "unset" (start_em.F substitutes a grid-spacing value for it)."""
    whole = int(whole)
    if whole <= 0:
        return None
    return Fraction(whole, int(den) if int(den) > 0 else 1)


def largest_mass_map_factor(projection: Mapping, map_proj: str, *,
                            dx: float, dy: float, e_we: int, e_sn: int
                            ) -> float:
    """``MAXVAL(grid%msftx)`` over the root's mass grid, as start_em.F
    reads it under the adaptive clock.

    The map factor is a function of latitude alone and the grid's
    latitude extremes lie on its boundary ring (latitude is monotone in
    projected distance from the pole point), so the ring is exact.
    """
    import numpy as np

    from woof.static.projection import projection_class

    grid = projection_class(map_proj)(
        ref_lat=float(projection["ref_lat"]),
        ref_lon=float(projection["ref_lon"]),
        truelat1=float(projection["truelat1"]),
        truelat2=float(projection["truelat2"]),
        stand_lon=float(projection["stand_lon"]),
        dx=float(dx), dy=float(dy), e_we=int(e_we), e_sn=int(e_sn))
    nx, ny = int(e_we) - 1, int(e_sn) - 1
    i = np.arange(1, nx + 1, dtype=np.float64)
    j = np.arange(1, ny + 1, dtype=np.float64)
    ring_x = np.concatenate((i, i, np.full(j.size, 1.0),
                             np.full(j.size, float(nx))))
    ring_y = np.concatenate((np.full(i.size, 1.0),
                             np.full(i.size, float(ny)), j, j))
    latitude, _longitude = grid.ij_to_latlon(ring_x, ring_y)
    factor = np.asarray(grid.map_factor(latitude), dtype=np.float64)
    if not np.isfinite(factor).all():
        raise ValueError("the root grid's map factor is not finite")
    return float(factor.max())


def _fixed_step_through_adaptive_clock(
        use_adaptive: bool, starting: tuple[int, int], largest: tuple[int, int],
        least: tuple[int, int], *, projection: Mapping, map_proj: str,
        dx: float, dy: float, e_we: int, e_sn: int, max_dom: int = 1
        ) -> tuple[Fraction, int, float] | None:
    """WRF's fixed step through the adaptive clock, and the acoustic
    count WRF derives for it: ``(step, num_sound_steps, max_msft)``.

    ``None`` unless ``use_adaptive_time_step`` is set with
    ``starting_time_step = max_time_step = min_time_step``, all set, on
    a single-domain run.  That shape clamps every step to one value
    (adapt_timestep_em.F), so the run integrates a fixed step;
    start_em.F then forces ``time_step_sound`` to 0 and solve_em.F
    derives the count from the live step and ``MAXVAL(msftx)``:
    ``max(2 * (INT(300 * dt / (min(dx, dy) / max_msft) - 0.01) + 1), 4)``
    (:func:`woof.core.adaptive_clock.wrf_num_sound_steps`).  HRRR v4
    (20/20/20 at 3 km, largest map factor 1.0443 at 21.14 N) derives 6
    where its namelist spells 4.

    A nested namelist is left to the adaptive clock it names: WRF clamps
    each grid to its OWN bounds and then rounds a nest's step to a whole
    divisor of its parent's (adapt_timestep_em.F, the per-grid max/min
    clamps and the ``grid%nested`` block after them), and derives each
    grid's count from that grid's step and map factor.  A root pinned by
    its triplet does not pin its nests, so a fixed tree at
    ``parent_time_step_ratio`` with the root's count is not what WRF
    integrates there.
    """
    if not use_adaptive or int(max_dom) != 1:
        return None
    bounds = [_wrf_interval(*pair) for pair in (starting, largest, least)]
    if any(bound is None for bound in bounds):
        return None
    if not (bounds[0] == bounds[1] == bounds[2]):
        return None
    from woof.core.adaptive_clock import wrf_num_sound_steps

    step = bounds[0]
    factor = largest_mass_map_factor(projection, map_proj, dx=dx, dy=dy,
                                     e_we=e_we, e_sn=e_sn)
    count = int(wrf_num_sound_steps(float(step), float(dx), float(dy),
                                    factor))
    return step, count, factor


def _translate_namelists(wps: dict, inp: dict, *, wps_path: Path,
                         input_path: Path, name: str | None = None,
                         urban_ucm_generation: str | None = None,
                         urban_controls: Mapping[str, object] | None = None,
                         rrtmg_variant: str | None = _AUTO_RRTMG_VARIANT,
                         rrtmg_compatibility: str | None = None,
                         acknowledgements: tuple[str, ...] = (),
                         landuse_identity: Mapping[str, object] | None = None,
                         wrf_boundary_use_theta_m: int | None = None,
                         wrfinput_qv_domains: tuple[int, ...] = (),
                         wrfinput_radar_tendency_domains: tuple[int, ...] = (),
                         static_cache_root: str | Path | None = None,
                         metgrid_initialization: bool = False,
                         geogrid_tbl: str | Path | None = None,
                         terrain_smoothing_precision: str | None = None,
                         wrf_version: str = WRF_VERSION_DEFAULT,
                         wrf_version_source: str | None = None,
                         request_source: str | None = None,
                         source_text: str | None = None,
                         preserve_wrf_mixing: bool = False
                         ) -> tuple[str, SubstitutionReport]:
    """The translation itself; stops at its first refusal."""
    requested_aerosol = inp.get("physics", {}).get("aer_opt", ())
    inferred_aerosol_legacy = rrtmg_variant is _AUTO_RRTMG_VARIANT and bool(requested_aerosol) and all(
        type(value) is int and value in (2, 3) for value in requested_aerosol)
    from woof.physics_source_defaults import (
        namelist_physics_defaults, recipe_physics_defaults,
        read_physics_selector_comment,
        with_physics_defaults_text)
    named_source_defaults = namelist_physics_defaults(input_path)
    # An authored source route retains its table request when its emitted
    # namelist has an arbitrary output stem. Ordinary imports pass no source.
    named_source_defaults.update(recipe_physics_defaults(request_source))
    source_defaults = dict(named_source_defaults)
    if source_text is not None and not isinstance(source_text, str):
        raise TypeError("source_text must be namelist text or None")
    carried_selectors = read_physics_selector_comment(source_text or "")
    source_defaults.update(carried_selectors)
    # A named WRF V4 release pins the generations its own code runs; the
    # namelist's comment and the named source's table still win.
    release_generations = {
        key: value for key, value in wrf_release_generations(wrf_version).items()
        if key not in source_defaults}
    source_defaults.update(release_generations)
    if rrtmg_variant is not _AUTO_RRTMG_VARIANT:
        # Coupled modern imports omit this key and read the RunConfig
        # default, so a source fallback must not reinsert legacy there.
        source_defaults.pop("ra_rrtmg_variant", None)
    if (rrtmg_variant not in (_AUTO_RRTMG_VARIANT, None, RRTMG_VARIANT_LEGACY)
            and "rrtmg_cloud_optics_form" not in carried_selectors):
        # An explicit modern mapping has no legacy cloud wrapper. Retain
        # an explicit comment for validation, but omit the source default.
        source_defaults.pop("rrtmg_cloud_optics_form", None)
    operational_ruc_requested = bool(operational_fork_ruc_defaults(inp))
    prescribed_surface_supplied = {
        key for key in ("rdlai2d", "usemonalb")
        if key in inp.get("physics", {})}
    # Supplied standard keys win even when default-value normalization
    # removes their output lines. Explicit OFF cannot become source ON.
    for key in ("mp_zero_out", "mp_zero_out_thresh", "mp_zero_out_all", "alb_sol"):
        if key in inp.get("physics", {}):
            source_defaults.pop(key, None)
    if carried_selectors.get("diff_6th_form") == "wrf_461":
        # The legacy form does not consume the fork-only factor. Keep a
        # supplied WRF factor for validation, but do not invent one from
        # the source request when an explicit comment chose the old form.
        source_defaults.pop("diff_6th_factor2", None)
    substitutions: list[Substitution] = []
    dropped: list[DroppedKey] = []
    fixed: list[FixedKey] = []
    notices: list[str] = []
    model_choices: list[str] = []
    defaults_applied: list[AppliedDefault] = [AppliedDefault(
        key="ztop", value=20000.0,
        reason="gpuwm-only vertical scaffold height (no WRF namelist "
               "counterpart); the real path derives heights from "
               "p_top/eta_levels -- the certified reference profile "
               "value")]

    if inferred_aerosol_legacy:
        defaults_applied.append(AppliedDefault(
            key="ra_rrtmg_variant", value=RRTMG_VARIANT_LEGACY,
            reason=f"aer_opt={requested_aerosol[0]} requires the implemented legacy RRTMG aerosol optics"))

    def drop(section: str, key: str, values, reason: str) -> None:
        if values is None:
            return
        if not isinstance(values, list):
            values = [values]
        dropped.append(DroppedKey(section=section, key=key,
                                  values=tuple(values), reason=reason))

    def fix(section: str, key: str, values, fixed_value, reason: str
            ) -> None:
        if values is None:
            return
        if not isinstance(values, list):
            values = [values]
        fixed.append(FixedKey(section=section, key=key,
                              values=tuple(values), fixed_value=fixed_value,
                              reason=reason))

    for section_name in WPS_DROPPED_SECTIONS:
        if section_name in wps:
            for key, values in wps[section_name].items():
                drop(section_name, key, values,
                     WPS_AUXILIARY_SECTIONS.get(
                         section_name, "ungrib/metgrid staging is replaced "
                         "by woof's direct forcing ingest"))
            if not wps[section_name] and section_name in WPS_AUXILIARY_SECTIONS:
                drop(section_name, "*", [], WPS_AUXILIARY_SECTIONS[section_name])
    active_fdda = False
    for key, values in active_nudging_selectors(inp.get("fdda", {})):
        if key == "grid_fdda":
            active_fdda = True
        else:
            raise _err("fdda", key, values,
                       "FDDA_OBSERVATION_INPUT: surface and point-observation nudging need observation forcing that the gridded analysis source does not supply")
    for section_name in INPUT_DROPPED_SECTIONS:
        if section_name == "fdda" and active_fdda:
            continue
        if section_name in inp:
            for key, values in inp[section_name].items():
                drop(section_name, key, values,
                     "FDDA/GRIB2/quilt-server machinery has no woof "
                     "counterpart")
    chem_import = _consume_machinery_sections(inp, drop, substitutions, fix)

    known_wps = set(WPS_SECTIONS)
    unknown = sorted(set(wps) - known_wps)
    if unknown:
        raise NamelistRefusal(
            f"unknown section(s) {unknown} in {wps_path}; known: "
            f"{sorted(known_wps)}.",
            [(section, "*") for section in unknown])
    known_inp = set(INPUT_SECTIONS)
    unknown = sorted(set(inp) - known_inp)
    if unknown:
        raise NamelistRefusal(
            f"unknown section(s) {unknown} in {input_path}; known: "
            f"{sorted(known_inp)}.",
            [(section, "*") for section in unknown])
    for required, parsed, path in (("share", wps, wps_path),
                                   ("geogrid", wps, wps_path),
                                   ("time_control", inp, input_path),
                                   ("domains", inp, input_path)):
        if required not in parsed:
            raise ValueError(f"{path} has no &{required} section.")
    mix_full_supplied_count = len(
        inp.get("dynamics", {}).get("mix_full_fields", ()))
    namelist_defaults = _apply_namelist_defaults(
        wps, inp, wrf_version=wrf_version,
        wrf_version_source=wrf_version_source)
    namelist_defaults.extend(_expand_physics_suite(inp, drop))
    _census_missing_keys(wps, inp, wps_path, input_path)

    share = _Section("share", wps["share"], str(wps_path))
    geo = _Section("geogrid", wps["geogrid"], str(wps_path))
    from woof.static.terrain_smoothing import (
        geogrid_tbl_smoothing, WPS_DEFAULT, static_inline, with_precision)
    named_table = wps["geogrid"].get("opt_geogrid_tbl_path")
    named_table = named_table[0] if isinstance(named_table, (list, tuple)) else named_table
    table_path = Path(geogrid_tbl) if geogrid_tbl is not None else Path(named_table or "geogrid")
    if geogrid_tbl is None and not table_path.is_absolute():
        table_path = Path(wps_path).parent / table_path
    if table_path.is_dir() or (geogrid_tbl is None and not table_path.is_file()):
        table_path = table_path / "GEOGRID.TBL"
    terrain_smoothing = WPS_DEFAULT
    if table_path.is_file():
        terrain_smoothing = geogrid_tbl_smoothing(table_path)
        if not terrain_smoothing.is_default:
            notices.append(f"Terrain smoothing imported from {table_path}: {terrain_smoothing.label()} on every domain.")
    elif geogrid_tbl is not None or named_table is not None:
        notices.append(f"GEOGRID.TBL {table_path} was not found; the terrain smoothing it may set was not imported.")
    if terrain_smoothing_precision is not None:
        # --terrain-smoothing-precision: the default smoother's arithmetic
        # (wps-float32 = geogrid.exe's HGT_M exactly); every other smoother
        # has one arithmetic, WPS's, and keeps its setting.
        chosen = with_precision(terrain_smoothing, terrain_smoothing_precision)
        if chosen != terrain_smoothing:
            notices.append(f"Terrain smoothing {chosen.label()} on every domain (--terrain-smoothing-precision {terrain_smoothing_precision}).")
        elif not terrain_smoothing.precision_is_a_choice:
            notices.append(f"--terrain-smoothing-precision {terrain_smoothing_precision} leaves terrain smoothing {terrain_smoothing.label()} as it is: only smth-desmth_special x1 has a choice of arithmetic, and every other smoother already runs WPS's.")
        terrain_smoothing = chosen
    tc = _Section("time_control", inp["time_control"], str(input_path))
    dm = _Section("domains", inp["domains"], str(input_path))
    ph = _Section("physics", inp.get("physics", {}), str(input_path))
    dyn = _Section("dynamics", inp.get("dynamics", {}), str(input_path))
    # diff_6th_factor2 is declared only by the NOAA-EMC WRFV3.9 fork
    # (HRRR v4.1.21 Registry.EM_COMMON:2629); public WRF v4.6.1 has no
    # such key.  A namelist that names it was written for that fork, and
    # two of the fork's behaviours follow it: its sixth-order filter form
    # (woof.core.dycore.DIFF6_FORMS) and its unguarded mp_zero_out call
    # set (fork solve_em.F:4078-4141, v4.6.1's mp_zero_out_all = 1).
    noaa_wrf39_namelist = "diff_6th_factor2" in dyn.entries
    bdy = _Section("bdy_control", inp.get("bdy_control", {}),
                   str(input_path))
    noahmp = _Section("noah_mp", inp.get("noah_mp", {}), str(input_path))
    stoch = _Section("stoch", inp.get("stoch", {}), str(input_path))
    # Keys only the operational WRF 3.9 fork (NOAA-EMC/HRRR) defines: a
    # namelist that carries any of them was written for that fork, so its
    # Thompson runs as the fork's (thompson_version = "wrf_39_noaa", emitted
    # below under mp_physics = 28).  Peeked before any section consumes
    # its keys; each is still recorded and dropped where it always was.
    operational_fork_keys = operational_fork_signature(
        {section.name: section.entries for section in (tc, ph, dyn)})
    operational_thompson_defaults = operational_fork_thompson_defaults(
        {section.name: section.entries for section in (tc, ph, dyn)})

    operational_ruc_defaults = operational_fork_ruc_defaults(inp)
    # Explicit generation metadata wins over the content-derived defaults.
    # Do this before emitting inferred lines: the later shared-default helper
    # intentionally fills only missing settings, never replaces a choice.
    for key in carried_selectors:
        operational_ruc_defaults.pop(key, None)
        operational_thompson_defaults.pop(key, None)
    climate_values = {key: tc.scalar(key, default) for key, default in
                      (("output_diagnostics", 0), ("diag_int", 15), ("wind_int", 5))}
    for key, value in climate_values.items():
        if type(value) is not int or (value not in (0, 1) if key == "output_diagnostics" else value <= 0):
            raise _err("time_control", key, value,
                       "must be 0 or 1" if key == "output_diagnostics" else "must be positive whole minutes")
    gsd_diagnostics = tc.scalar("gsd_diagnostics", 0)
    if gsd_diagnostics not in (0, 1):
        raise _err("time_control", "gsd_diagnostics", gsd_diagnostics,
                   "only operational package 1 is carried; research package 2 is not")
    fire = _Section("fire", inp.get("fire", {}), str(input_path))

    # ---- &noah_mp: option-identity validation --------------------------
    # Every Noah-MP option is identity-pinned (woof/config.py
    # NOAHMP_OPTION_IDENTITY_EVIDENCE: the ported column is validated at
    # exactly one value of each).  A key at the identity value is recorded
    # as fixed; any other value refuses with the identity's evidence.
    from woof.config import (MYNN_PBL_OPTION_IDENTITY,
                              NOAHMP_OPTION_IDENTITY_EVIDENCE,
                              NOAHMP_OPTIONS_WITHOUT_CONSUMER)
    for option, (admitted, evidence) in \
            NOAHMP_OPTION_IDENTITY_EVIDENCE.items():
        values = noahmp.take(option)
        if values is None:
            continue
        if any(not _identity_matches(value, admitted) for value in values):
            # A knob with no consumer at any value has no breakage to
            # name, so a namelist that sets one is imported rather than
            # refused, and the record says the knob reaches nothing.
            # The run door reaches the same verdict from the same table
            # (woof.config.NOAHMP_OPTIONS_WITHOUT_CONSUMER), so the two
            # doors cannot disagree about one namelist.
            if option not in NOAHMP_OPTIONS_WITHOUT_CONSUMER:
                raise _err(
                    "noah_mp", option, values,
                    f"woof's Noah-MP port implements {option} = "
                    f"{admitted!r} only ({evidence}); no nearby branch "
                    "is substituted for an unported one.")
            fix("noah_mp", option, values, admitted,
                f"{option} reaches no woof code ({evidence}), so the "
                "imported value changes nothing and the pin is written")
            continue
        fix("noah_mp", option, values, admitted,
            f"Noah-MP option identity ({evidence})")
    noahmp.finish()

    # ---- &share ---------------------------------------------------------
    wrf_core = share.scalar("wrf_core", "ARW")
    if str(wrf_core).upper() != "ARW":
        raise _err("share", "wrf_core", wrf_core, "only ARW is supported.")
    max_dom = dm.scalar("max_dom", 1)
    if isinstance(max_dom, bool) or not isinstance(max_dom, int) \
            or not 1 <= max_dom <= 21:
        raise _err("domains", "max_dom", max_dom,
                   "must be an integer in [1, 21] (WRF's compiled "
                   "max_domains default); refusing to expand per-domain "
                   "arrays for an implausible domain count.")
    reset_interval1_column = tc.col("reset_interval1", max_dom, 60)
    # Validate the domain count before expanding any stochastic arrays.
    # A coefficient does not enable a scheme. The disabled route retains
    # its original output bytes; active controls bind a complete provider.
    from woof.namelist_stochastic import import_stochastic_section, SEED_NOTICE
    stochastic_controls, stochastic_flags = import_stochastic_section(stoch,
        max_dom=max_dom, fix=fix, drop=drop, error=_err)
    if stochastic_controls is not None:
        notices.append(SEED_NOTICE)
    wps_max_dom = share.scalar("max_dom")
    if wps_max_dom is not None and wps_max_dom != max_dom:
        raise ValueError(
            f"max_dom mismatch: {wps_path} says {wps_max_dom}, "
            f"{input_path} says {max_dom}.")
    fire_columns = _fire_columns(fire, dm, max_dom)

    # ---- &time_control: start/run length --------------------------------
    def _dt_columns(prefix: str) -> list[datetime]:
        parts: dict[str, list[int]] = {}
        for unit, default in (("year", None), ("month", None),
                              ("day", None), ("hour", 0), ("minute", 0),
                              ("second", 0)):
            col = tc.col(f"{prefix}_{unit}", max_dom,
                         default=default)
            if col is None:
                raise ValueError(
                    f"{input_path} &time_control is missing "
                    f"{prefix}_{unit}.")
            parts[unit] = list(col)
        try:
            return [
                datetime(parts["year"][index], parts["month"][index],
                         parts["day"][index], parts["hour"][index],
                         parts["minute"][index], parts["second"][index])
                for index in range(max_dom)
            ]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{input_path} &time_control has an invalid per-domain "
                f"{prefix}_* datetime column: {error}") from None

    start_times = _dt_columns("start")
    start_time = start_times[0]
    run_days = int(tc.scalar("run_days", 0))
    run_hours = int(tc.scalar("run_hours", 0))
    run_minutes = int(tc.scalar("run_minutes", 0))
    run_secs = int(tc.scalar("run_seconds", 0))
    run_seconds = float(run_days * 86400 + run_hours * 3600
                        + run_minutes * 60 + run_secs)
    if run_seconds > 0.0:
        # WRF reads no end_* when run_* is positive: the end is start +
        # run length, which is what a present end_* is checked against.
        end_time = start_time + timedelta(seconds=run_seconds)
        for unit in ("year", "month", "day", "hour", "minute", "second"):
            key = f"end_{unit}"
            drop("time_control", key, tc.take(key),
                 "WRF derives the forecast end from positive run_*; "
                 "this end_* staging value is not read for the integration clock")
    else:
        end_times = _dt_columns("end")
        end_time = _uniform("time_control", "end_* datetime", end_times)
    if run_seconds <= 0.0:
        run_seconds = (end_time - start_time).total_seconds()
    if run_seconds <= 0.0:
        raise ValueError(
            f"{input_path} &time_control yields a non-positive run "
            f"length: run_days/hours/minutes/seconds all zero and "
            f"end {end_time} <= start {start_time}.")
    expected_end = start_time + timedelta(seconds=run_seconds)
    if end_time != expected_end:
        raise ValueError(
            f"{input_path} &time_control run_* ends the root at "
            f"{expected_end}, but the uniform end_* columns declare "
            f"{end_time}.")

    wps_start = share.take("start_date")
    if wps_start is not None:
        if len(wps_start) > max_dom:
            drop("share", "start_date (inactive domains)",
                 wps_start[max_dom:],
                 "WPS reads only the max_dom active-domain dates; "
                 "trailing template entries do not change the run")
            wps_start = wps_start[:max_dom]
        wps_start = list(wps_start) + [wps_start[-1]] * (
            max_dom - len(wps_start))
        parsed = []
        templates = []
        for index, value in enumerate(wps_start):
            if "@" in str(value) and _WPS_DATE_TEMPLATE.fullmatch(str(value)):
                templates.append(value)
                parsed.append(start_times[index])
            else:
                parsed.append(datetime.strptime(str(value), "%Y-%m-%d_%H:%M:%S"))
        drop("share", "start_date (templates)", templates or None,
             "unresolved WPS staging date templates are replaced by the "
             "concrete &time_control start_* clock for direct forcing ingest")
        if parsed != start_times:
            raise ValueError(
                f"per-domain start mismatch: {wps_path} start_date = "
                f"{[value.isoformat() for value in parsed]} but "
                f"{input_path} start_* = "
                f"{[value.isoformat() for value in start_times]}.")
    drop("share", "end_date", share.take("end_date"),
         "run length comes from &time_control run_*/end_*")
    for section, obj, key, reason in (
            ("share", share, "interval_seconds",
             "forcing cadence is discovered and validated from the input "
             "catalog, not declared"),
            ("time_control", tc, "interval_seconds",
             "forcing cadence is discovered and validated from the input "
             "catalog, not declared"),
            ("share", share, "io_form_geogrid",
             "WPS geogrid staging is replaced by woof's static builder"),
            ("share", share, "opt_output_from_geogrid_path",
             "WPS geogrid output directory; woof's static builder manages its own output"),
            ("share", share, "debug_level", "WPS logging control"),
            ("share", share, "nocolons",
             "woof filenames are always colon-free (H_M_S)"),
            ("time_control", tc, "nocolons",
             "woof filenames are always colon-free (H_M_S)"),
            ("time_control", tc, "frames_per_outfile",
             "woof writes one frame per wrfout file"),
            ("time_control", tc, "restart",
             "resuming is the `woof run --restart` flag, not a config "
             "key"),
            ("time_control", tc, "io_form_history", "WRF I/O layer key"),
            ("time_control", tc, "io_form_restart", "WRF I/O layer key"),
            ("time_control", tc, "io_form_input", "WRF I/O layer key"),
            ("time_control", tc, "io_form_boundary", "WRF I/O layer key"),
            # override_restart_timers steers WRF's restart-alarm
            # bookkeeping.  Resuming is the `woof run --restart` flag
            # (the reason the &time_control restart key drops above), so
            # there is no timer to override; runtime-only in the same
            # report line, and consumed here for the same reason.
            ("time_control", tc, "override_restart_timers",
             "resuming is the `woof run --restart` flag, not a config "
             "key"),
    ):
        drop(section, key, obj.take(key), reason)

    # nwp_diagnostics (Registry.EM_COMMON:2210, &time_control, default 0):
    # mapped 1:1 onto the RunConfig field of the same name (knob-parity
    # conventions -- emitted only when supplied, WRF Registry default ==
    # frozen RunConfig default).  woof implements the UP_HELI_MAX member
    # of WRF's nwp_output family per step (woof/core/uh_diag.py); the
    # unimplemented members (WSPD10MAX, W_UP_MAX/W_DN_MAX, W_MEAN,
    # GRPL_MAX, HAIL_MAX*) are simply absent from wrfouts, and WRF's
    # every-step diagflag forcing (solve_em.F:369) changes only which
    # instant computes diagnostics WRF then discards -- the emitted
    # instantaneous REFL_10CM at history time is the same field either
    # way, so the old scope refusal is retired.
    nwp_diagnostics_values = tc.take("nwp_diagnostics")
    nwp_diagnostics = None
    if nwp_diagnostics_values is not None:
        value = nwp_diagnostics_values[0]
        if isinstance(value, bool) or not isinstance(value, int) \
                or value not in (0, 1):
            raise _err(
                "time_control", "nwp_diagnostics", nwp_diagnostics_values,
                "must be 0 (off, the WRF default) or 1 (per-step "
                "UP_HELI_MAX running-max diagnostic).")
        nwp_diagnostics = value

    # cycling (Registry.EM_COMMON, &time_control, default .false.): the start
    # carries a previous forecast's scheme state.  Mapped onto RunConfig
    # .cycling and emitted only when .true., so every namelist that omits
    # it or spells the default imports to the same bytes as before.  Its
    # consumers are the first MYNN call of EITHER generation (keep the input
    # QKE unless its lowest-level maximum is below 0.0002, and QC_BL and
    # CLDFRA_BL; woof/core/mynn_pbl_runtime.py) and the wrfinput carry of
    # that subgrid cloud (woof/ingest/wrfinput.py).  validate_run_config
    # admits it under both: WOOF's v4.6.1 generation carries the three
    # fields its WRF original zeroes, a declared fix.  This is the ONE place
    # the key is read, on every import route including the operational
    # fork's, with the same carried-state ownership.
    cycling_values = tc.take("cycling")
    cycling = None
    if cycling_values is not None:
        if _require_bools("time_control", "cycling", cycling_values[:1])[0]:
            cycling = True

    # WRF's Registry default is .true. for EVERY element -- Registry.EM_COMMON:
    # `rconfig logical input_from_file namelist,time_control max_domains
    # .true.` -- so an omitted key and an omitted tail both import (the
    # ordinary single-domain namelist never names it), and it is the
    # shipped documentation's pin (docs/public/CONFIGURATION.md).  The
    # importer briefly read the default as .false. and refused every
    # namelist that left the key out, with a message asserting the
    # opposite of the Registry (ENG-015).  Only an EXPLICIT .false. is
    # refused, and the refusal names why.
    input_from_file = _require_bools(
        "time_control", "input_from_file",
        tc.registry_col("input_from_file", max_dom, True))
    if not all(input_from_file):
        raise _err("time_control", "input_from_file", input_from_file,
                   "this entry route requires a file initial condition for each domain. "
                   "WRF's Registry default is .true. for every domain (Registry.EM_COMMON), "
                   "so an omitted key or an omitted tail imports; an explicit .false. "
                   "selects parent-interpolated (ndown-style) initialization, which is "
                   "not wired to this route. Set input_from_file=.true. for that domain "
                   "or drop the entry.")
    fix("time_control", "input_from_file", input_from_file, True,
        "per-domain file initialization implements the input_from_file=T "
        "branch (med_nest_initial, share/mediation_integrate.F:509-952)")

    # fine_input_stream is the OTHER per-domain nest-initialization
    # selector, and it is answered by the same function the RW-WPS
    # support report calls (:func:`fine_input_stream_decision`).  It used
    # to be consumed nowhere, so every namelist carrying it died on the
    # unmapped-key refusal in _Section.finish while the support report
    # returned PASS on the identical pair: one configuration, two
    # answers.  The column is built with the report's own rule -- extra
    # domain values rejected rather than truncated, the declared tail
    # filled from the last value -- so the two doors read one column.
    fine_input_stream_values = tc.take("fine_input_stream")
    if fine_input_stream_values is not None:
        if len(fine_input_stream_values) > max_dom:
            raise _err(
                "time_control", "fine_input_stream", fine_input_stream_values,
                f"declares {len(fine_input_stream_values)} values but "
                f"max_dom = {max_dom}; extra domain values are rejected "
                f"rather than truncated. Declare at most {max_dom} "
                "values, one per domain, or raise max_dom to cover "
                "them.")
        fine_input_stream_column = list(fine_input_stream_values) + [
            fine_input_stream_values[-1]] * (
                max_dom - len(fine_input_stream_values))
        # Both the type gate and the undefined-index refusal are the
        # shared function's, named here with the file they came from.
        # They are raised whole rather than through _err, which would
        # restate the column the sentences already carry.
        try:
            stream_decision = fine_input_stream_decision(
                fine_input_stream_column)
        except ValueError as error:
            raise ValueError(
                f"&time_control of {input_path}: {error}") from error
        if stream_decision.refusal is not None:
            raise ValueError(
                f"&time_control of {input_path}: "
                f"{stream_decision.refusal} {stream_decision.way_out}")
        if stream_decision.delayed_domains:
            # A SUBSTITUTION, NOT A REFUSAL AND NOT A DROP: the delayed
            # child starts at its declared start time either way, and
            # the divergence (where its masked surface state comes from)
            # is a trajectory statement the terminal must print, not a
            # line only the receipt file carries.
            substitutions.append(Substitution(
                key="fine_input_stream",
                wrf_value=FINE_INPUT_STREAM_DELAYED_NEST,
                wrf_name="delayed-nest start with parent-interpolated "
                         "3-D meteorology",
                gpuwm_key="fine_input_stream",
                gpuwm_value=FINE_INPUT_STREAM_OWN_INPUT,
                gpuwm_name="delayed-nest start from the child's own-grid "
                           "analysis",
                reason=stream_decision.divergence))
        else:
            drop("time_control", "fine_input_stream",
                 list(stream_decision.streams),
                 "every domain takes its initial fields from its own-grid "
                 "analysis, which is what stream "
                 f"{FINE_INPUT_STREAM_OWN_INPUT} selects; there is no "
                 "[domains] counterpart key to carry it to")

    # ---- the history alarm: interval, begin and end -------------------
    # WRF builds every output alarm from one macro
    # (share/set_timekeeping_alarms.m4): the interval is
    # D/H/M/S of <stream>_interval_d/_h/_m/_s with the bare <stream>_interval
    # (minutes) standing in for _m when _m is 0, and the alarm first rings
    # at the domain's start + <stream>_begin_d/_h/_m/_s, then every
    # interval, and never after start + <stream>_end_d/_h/_m/_s
    # (frame/module_domain.F:2260-2332 domain_alarm_create; the ESMF clock
    # stops an alarm once the time passes its StopTime).  _y is read only
    # to test whether a begin or end is set and never enters the offset.
    def _alarm_column(stem: str, *, bare_minutes: bool,
                      fill) -> list[float] | None:
        """Seconds per domain, WRF's D/H/M/S sum; None when unset."""
        parts = {unit: tc.take(f"{stem}_{unit}")
                 for unit in ("d", "h", "m", "s")}
        bare = tc.take(stem) if bare_minutes else None
        if bare is None and all(value is None for value in parts.values()):
            return None
        columns = {}
        for unit, raw in (*parts.items(), ("bare", bare)):
            if raw is None:
                columns[unit] = [0] * max_dom
                continue
            label = stem if unit == "bare" else f"{stem}_{unit}"
            if any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 0 for value in raw):
                raise _err("time_control", label, raw,
                           "WRF declares it a non-negative integer "
                           "(registry.io_boilerplate).")
            columns[unit] = fill(raw)
        if bare is not None and parts["m"] is not None and all(
                columns["m"][n] for n in range(max_dom)) and any(
                columns["bare"][n] != columns["m"][n]
                for n in range(max_dom)):
            # Recorded, so the unread spelling never vanishes unexplained.
            drop("time_control", stem, bare,
                 f"WRF reads {stem} only where {stem}_m is 0 "
                 "(share/set_timekeeping_alarms.m4), and "
                 f"{stem}_m is set on every domain")
        seconds = []
        for n in range(max_dom):
            minutes = columns["m"][n] or columns["bare"][n]
            seconds.append(float(columns["d"][n] * 86400
                                 + columns["h"][n] * 3600
                                 + minutes * 60 + columns["s"][n]))
        return seconds

    def _last_value_fill(raw: list) -> list:
        # The importer's ratified fill for the history cadence (``col``).
        values = list(raw[:max_dom])
        return values + [values[-1]] * (max_dom - len(values))

    def _registry_fill(raw: list) -> list:
        # Fortran leaves an unassigned tail at the Registry default, 0.
        values = list(raw[:max_dom])
        return values + [0] * (max_dom - len(values))

    history_interval2_s = _alarm_column(
        "history_interval2", bare_minutes=True, fill=_last_value_fill)
    history_interval_change_s = _alarm_column(
        "history_interval_change", bare_minutes=True, fill=_last_value_fill)
    history_interval_s = _alarm_column(
        "history_interval", bare_minutes=True, fill=_last_value_fill)
    if history_interval_s is None or any(v <= 0.0
                                         for v in history_interval_s):
        raise _err("time_control", "history_interval",
                   history_interval_s,
                   "every domain needs a positive history cadence.")
    # WRF's bare history_begin / history_end are declared in minutes
    # (registry.io_boilerplate:62, :82) but the alarm macro never reads
    # them: it reads _m, and substitutes the bare key only for the
    # INTERVAL.  ArWen applies them the way WRF applies history_interval,
    # as minutes where _m is 0, and says so (below), because a namelist
    # that writes history_begin = 600 is asking for output from ten hours.
    bare_offsets = {key: tc.entries.get(key)
                    for key in ("history_begin", "history_end")}
    history_begin_s = _alarm_column(
        "history_begin", bare_minutes=True, fill=_registry_fill)
    history_end_s = _alarm_column(
        "history_end", bare_minutes=True, fill=_registry_fill)
    for key in ("history_begin_y", "history_end_y"):
        drop("time_control", key, tc.take(key),
             "WRF reads the year entry only to test whether an offset is "
             "set and never adds years to it (share/set_timekeeping_alarms"
             ".m4), so it changes no alarm")
    for key, raw in bare_offsets.items():
        if raw is not None and any(int(value) for value in raw):
            substitutions.append(Substitution(
                key=key, wrf_value=list(raw),
                wrf_name="declared in minutes, never read by WRF's alarm "
                         "setup",
                gpuwm_key=f"[[domain]] {key}_s",
                gpuwm_value=[value * 60 for value in raw],
                gpuwm_name=f"{key} applied as minutes",
                reason=(f"WRF v4.7.1 declares {key} in minutes "
                        "(registry.io_boilerplate) but its alarm setup "
                        f"(share/set_timekeeping_alarms.m4) reads only "
                        f"{key}_d/_h/_m/_s, so wrf.exe ignores it; WOOF "
                        "applies it as minutes where the _m entry is 0, the "
                        "rule WRF uses for history_interval.  Write "
                        f"{key}_m to get the same output from wrf.exe.")))
    if history_end_s is not None:
        for n in range(max_dom):
            if history_end_s[n] and history_end_s[n] <= (
                    history_begin_s[n] if history_begin_s else 0.0):
                raise _err("time_control", "history_end",
                           history_end_s,
                           f"domain {n + 1} ends its history at "
                           f"{history_end_s[n]:g} s, not after its begin "
                           "offset, so it would write nothing.")
    restart_parts = _alarm_column(
        "restart_interval", bare_minutes=True, fill=_last_value_fill)
    # Restart alarms ring on the d01 clock only (WRF reads id 1).
    restart_interval_s = restart_parts[0] if restart_parts else 0.0
    # Output-stream and logging keys with no woof counterpart: woof
    # writes its fixed per-domain wrfout product set on the history
    # cadence, so auxiliary stream declarations and I/O field overrides
    # are recorded, never silently influential.  ``adjust_output_times``
    # is inert on woof's exact rational clock: fixed-step admission
    # requires aligned intervals, and the adaptive driver shortens its
    # live step to land exactly on each requested history alarm.
    drop("time_control", "debug_level", tc.take("debug_level"),
         "WRF logging control")
    drop("time_control", "adjust_output_times",
         tc.take("adjust_output_times"),
         "inert on the exact rational clock (alarms land "
         "exactly on the history cadence)")
    for key in sorted(tc.entries):
        reason = ignored_time_control_reason(key)
        if reason is not None:
            drop("time_control", key, tc.take(key), reason)
    tc.finish()
    share.finish()

    # ---- &geogrid vs &domains cross-checks -------------------------------
    def _int_col(section: _Section, key: str, required=True):
        values = section.col(key, max_dom)
        if values is None:
            if required:
                raise ValueError(
                    f"&{section.name} of {section.source} is missing "
                    f"{key}.")
            return None
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in values):
            raise _err(
                section.name, key, values,
                "must contain Fortran integer tokens; fractional, logical, "
                "and string values are rejected without coercion.")
        return list(values)

    geo_parent_id = _int_col(geo, "parent_id")
    geo_ratio = _int_col(geo, "parent_grid_ratio")
    geo_i = _int_col(geo, "i_parent_start")
    geo_j = _int_col(geo, "j_parent_start")
    geo_e_we = _int_col(geo, "e_we")
    geo_e_sn = _int_col(geo, "e_sn")
    parent_id = _int_col(dm, "parent_id")
    grid_ids = _int_col(dm, "grid_id", required=False) \
        or list(range(1, max_dom + 1))
    ratio = _int_col(dm, "parent_grid_ratio")
    tratio = _int_col(dm, "parent_time_step_ratio")
    i_start = _int_col(dm, "i_parent_start")
    j_start = _int_col(dm, "j_parent_start")
    e_we = _int_col(dm, "e_we")
    e_sn = _int_col(dm, "e_sn")
    e_vert = _int_col(dm, "e_vert")

    expected_grid_ids = list(range(1, max_dom + 1))
    if grid_ids != expected_grid_ids:
        raise ValueError(
            "WRF hierarchy grid_id values must be contiguous and listed "
            f"parent-before-child: expected {expected_grid_ids}, got "
            f"{grid_ids}.")
    if parent_id[0] not in (0, grid_ids[0]):
        raise ValueError(
            f"d01 parent_id must be 0 or its own grid_id {grid_ids[0]}, "
            f"got {parent_id[0]}: the head grid cannot have an external parent.")
    if parent_id[0] != 0:
        drop("domains", "parent_id (root)", [parent_id[0]],
             "WRF's head grid has no parent; its self-parent spelling "
             "is normalized to the experiment's root marker 0")
        parent_id[0] = 0
    declared_parent_ids = {grid_ids[0]}
    for grid_id, declared_parent in zip(grid_ids[1:], parent_id[1:]):
        if declared_parent not in declared_parent_ids:
            raise ValueError(
                f"parent_id = {declared_parent} of grid_id = {grid_id} "
                "does not name a previously declared domain (orphan and "
                "cyclic hierarchies are rejected before geometry "
                "derivation).")
        declared_parent_ids.add(grid_id)

    wps_root = {
        "parent_id": geo_parent_id[0],
        "parent_grid_ratio": geo_ratio[0],
        "i_parent_start": geo_i[0],
        "j_parent_start": geo_j[0],
    }
    expected_wps_root = {
        "parent_id": 1,
        "parent_grid_ratio": 1,
        "i_parent_start": 1,
        "j_parent_start": 1,
    }
    if wps_root != expected_wps_root:
        raise ValueError(
            "WPS d01 root topology must use parent_id=1, "
            "parent_grid_ratio=1, i_parent_start=1, j_parent_start=1; "
            f"got {wps_root}.")

    # WPS lists d01's parent as itself (parent_id = 1); namelist.input
    # uses 0 for the head grid -- compare children only.  The head grid's
    # i/j_parent_start place nothing either: they locate a nest's corner
    # in its parent, WPS writes 1 and namelist.input templates carry 1 or
    # 0 (the operational 3 km file has 0), so the root values are
    # recorded, not compared.
    for key, wps_col, inp_col in (("i_parent_start", geo_i, i_start),
                                  ("j_parent_start", geo_j, j_start)):
        if inp_col[0] != wps_col[0]:
            drop("domains", f"{key} (root)", [inp_col[0]],
                 "the head grid has no parent, so its parent-start "
                 "coordinate places nothing; WPS's value "
                 f"{wps_col[0]} stands")
            inp_col[0] = wps_col[0]
    for key, wps_col, inp_col in (
            ("parent_id", geo_parent_id[1:], parent_id[1:]),
            ("parent_grid_ratio", geo_ratio[1:], ratio[1:]),
            ("i_parent_start", geo_i[1:], i_start[1:]),
            ("j_parent_start", geo_j[1:], j_start[1:]),
            ("e_we", geo_e_we, e_we), ("e_sn", geo_e_sn, e_sn)):
        if wps_col != inp_col:
            raise ValueError(
                f"nest layout mismatch for {key}: {wps_path} says "
                f"{wps_col} but {input_path} says {inp_col}.")

    wps_dx = float(geo.required("dx"))
    wps_dy = float(geo.scalar("dy", wps_dx))
    # WRF derives omitted child spacing from the parent ratio. Keep the
    # supplied prefix instead of applying the usual last-value column fill:
    # the public namelist specifies only root dx/dy.
    dm_dx = dm.take("dx")
    dm_dy = dm.take("dy")
    if dm_dx is not None:
        dm_dx = dm_dx[:max_dom]
    if dm_dy is not None:
        dm_dy = dm_dy[:max_dom]
    root_dx = float(dm_dx[0]) if dm_dx is not None else wps_dx
    root_dy = float(dm_dy[0]) if dm_dy is not None else wps_dy
    if abs(wps_dx - root_dx) > _REL_TOL * abs(root_dx) \
            or abs(wps_dy - root_dy) > _REL_TOL * abs(root_dy):
        raise ValueError(
            f"d01 dx/dy mismatch: {wps_path} says {wps_dx}/{wps_dy} but "
            f"{input_path} says {root_dx}/{root_dy}.")
    if root_dx != root_dy:
        raise ValueError(
            f"d01 dx = {root_dx} must equal dy = {root_dy} (conformal "
            "projected grids are isotropic).")

    # Child dx/dy: cross-check the hand-typed decimals against the exact
    # chain, then DROP them -- the resolved TOML derives child dx at load.
    dx_exact = [Fraction(root_dx)]
    for n in range(1, max_dom):
        parent_index = grid_ids.index(parent_id[n])
        dx_exact.append(dx_exact[parent_index] / ratio[n])
        for key, col in (("dx", dm_dx), ("dy", dm_dy)):
            if col is None or n >= len(col):
                continue
            supplied = float(col[n])
            derived = float(dx_exact[n])
            if abs(supplied - derived) > _REL_TOL * abs(derived):
                raise ValueError(
                    f"&domains {key}[{n + 1}] = {supplied!r} in "
                    f"{input_path} contradicts the parent chain: "
                    f"d0{grid_ids[n]} derives {key} = {dx_exact[n]} m "
                    f"(parent {dx_exact[parent_index]} m / ratio "
                    f"{ratio[n]}).  The namelist chain is authoritative "
                    "-- child dx is never hand-typed.")
    if max_dom > 1:
        for key, col in (("dx (children)", dm_dx), ("dy (children)",
                                                    dm_dy)):
            drop("domains", key,
                 [float(v) for v in col[1:]] if col else None,
                 "child dx/dy derive exactly from the parent chain at "
                 "load; the hand-typed namelist decimals are "
                 "cross-checked, not copied")

    # ---- &domains: clock, vertical grid, guards --------------------------
    time_step = int(dm.required("time_step"))
    fract_den_supplied = "time_step_fract_den" in dm.entries
    fract_num = int(dm.scalar("time_step_fract_num", 0))
    fract_den = int(dm.scalar("time_step_fract_den", 1))
    if fract_den_supplied and fract_num == 0 and fract_den != 1:
        drop("domains", "time_step_fract_den", [fract_den],
             "denominator of a zero fractional step (time_step_fract_num "
             "= 0): the step is time_step whatever it is")
    nz = _uniform("domains", "e_vert", e_vert) - 1
    eta_levels = [float(v) for v in (dm.take("eta_levels") or [])]
    if metgrid_initialization:
        from woof.ingest.eta import wrf_automatic_eta_requested
        if wrf_automatic_eta_requested(eta_levels):
            fix("domains", "eta_levels", eta_levels, [],
                "WRF automatic-grid marker; the shared Rust generator "
                "materializes eta during metgrid initialization")
            eta_levels = []
    # Omitted namelist keys take their WRF v4.6.1 Registry defaults
    # (review F2 -- silently substituting gpuwm-convenient values is a
    # never-silent violation): p_top_requested 5000 Pa
    # (Registry.EM_COMMON:2275), feedback 1 (:2322), smooth_option 2
    # (:2323) -- an implicitly two-way namelist therefore selects the
    # experimental runtime path instead of importing as one-way.
    p_top = float(dm.scalar("p_top_requested", 5000))
    # hypsometric_opt is a &domains SCALAR, not a &dynamics column
    # (Registry.EM_COMMON:2283, `namelist,domains`, nentries 1;
    # run/README.namelist documents it inside &domains).  woof read it
    # from &dynamics, which is a section no real WRF namelist can carry
    # it in -- wrf.exe fails the &dynamics namelist read outright -- so
    # the importer could only ever have accepted namelists woof itself
    # wrote.  Ratified Registry-default binding when the namelist leaves
    # it unset (Phase-4 native-dt baseline): 2.
    hypsometric_opt = int(dm.scalar("hypsometric_opt", 2))

    # ADAPTIVE TIME STEP.  The refusal that used to stand here said woof
    # "integrates on the fixed namelist clock", and that was true when it
    # was written: several invariants were built on a constant step.  They
    # are not any more -- see docs/ADAPTIVE-TIMESTEP.md -- so the guard comes
    # out and the keys are carried.  It is the LAST step of that work by
    # design: a refusal is cheap to keep and expensive to have removed
    # early.
    #
    # SCOPE, from Registry.EM_COMMON:2269-2281 -- use_adaptive_time_step,
    # step_to_output_time and adaptation_domain are scope 1 (one scalar
    # for the run); the rest are max_domains, and woof carries them per
    # domain (woof.experiment._DOMAIN_RUN_OVERRIDES: a parent and its
    # nest reach target_cfl at different steps, and upstream's own
    # guidance runs max_step_increase_pct at 5 on a parent and 51 on a
    # nest).  So a column is read per domain on the importer's
    # last-value fill, the root's value goes to [shared] and a domain
    # that differs gets its own [[domain]] row below, on epssm's rule.  A
    # uniform column, which is every column that imported before, emits
    # the same TOML it always did.  The refusal of a DISAGREEING column
    # that stood here ("one value for the tree") predates the per-domain
    # overrides and turned away every nested adaptive namelist, among
    # them the HRRR route's own.
    use_adaptive_time_step, step_to_output_time, adaptation_domain = (
        cast(dm.scalar(key, default))
        for key, default, cast in ADAPTIVE_CLOCK_SCALARS)
    adaptive_columns = {
        key: [cast(value) for value in dm.col(key, max_dom, default)]
        for key, default, cast in ADAPTIVE_CLOCK_COLUMNS}
    (target_cfl, target_hcfl, max_step_increase_pct, starting_time_step,
     starting_time_step_den, max_time_step, max_time_step_den,
     min_time_step, min_time_step_den) = (
        adaptive_columns[key][0] for key, _, _ in ADAPTIVE_CLOCK_COLUMNS)

    for key, (ok, why) in NEST_GUARDS.items():
        raw = dm.take(key)
        value = ok if raw is None else raw[0]
        if value != ok:
            raise _err("domains", key, value, why)
        fix("domains", key, raw, ok, why.rstrip("."))

    # smooth_cg_topo (Registry.EM_COMMON:2301, logical, scope 1): real.exe
    # blends domain 1's outer rows toward the input model's terrain
    # (woof.ingest.cg_topo ports it).  Emitted into [experiment] only
    # when true, so every other import is byte-identical.
    smooth_cg_topo_values = dm.take("smooth_cg_topo")
    smooth_cg_topo = False
    if smooth_cg_topo_values is not None:
        smooth_cg_topo = _require_bools(
            "domains", "smooth_cg_topo", smooth_cg_topo_values[:1])[0]
    # WRF process/tile decomposition layout: woof's GPU decomposition is
    # internal, so these carry no science and drop with a reason.
    # tile_sz_x/tile_sz_y size the CPU build's shared-memory tiles and
    # belong to exactly this family; they are recorded here rather than
    # refused so that the support report, which states them as a note
    # (woof/namelist_compat.py:1079), and this importer give one answer
    # about one namelist.
    for key in ("numtiles", "nproc_x", "nproc_y", "tile_sz_x", "tile_sz_y"):
        drop("domains", key, dm.take(key),
             "WRF parallel tile/process decomposition layout; woof's "
             "GPU decomposition is internal")
    for key in ("s_we", "s_sn", "s_vert"):
        values = dm.take(key)
        if values is None:
            continue
        if any(not _identity_matches(value, 1) for value in values[:max_dom]):
            raise _err("domains", key, values,
                       "the imported grid starts at WRF index 1; a different "
                       "start index would shift its extent")
        fix("domains", key, values, 1,
            "WRF grid indexing starts at 1; experiment dimensions encode "
            "the same extent without an index-origin control")

    # Explicit eta bypasses WRF generation. The metgrid door materializes
    # omitted eta through the shared Rust generator before initialization.
    # Standalone import has no preparation contract and still requires eta.
    for key in ("auto_levels_opt", "max_dz", "dzbot", "dzstretch_s",
                "dzstretch_u"):
        raw = dm.take(key)
        if raw is None:
            continue
        if eta_levels:
            drop("domains", key, raw,
                 "inert: explicit eta_levels bypass WRF's automatic level generation")
        elif metgrid_initialization:
            from woof.ingest.eta import wrf_eta_options
            resolved = wrf_eta_options({key: raw})[key]
            fix("domains", key, raw, resolved,
                "consumed by the shared Rust WRF eta generator during metgrid initialization")
        else:
            raise _err(
                "domains", key, raw,
                "automatic eta-level generation is consumed by woof run --met-em DIR; "
                "standalone import requires explicit eta_levels.")

    use_sh_qv = dm.take("use_sh_qv")
    if use_sh_qv is not None:
        if len(use_sh_qv) != 1 or not isinstance(use_sh_qv[0], bool):
            raise _err("domains", "use_sh_qv", use_sh_qv, "requires one logical value")
        if not metgrid_initialization:
            raise _err("domains", "use_sh_qv", use_sh_qv,
                       "this preprocessing control is consumed by woof run --met-em DIR")
        fix("domains", "use_sh_qv", use_sh_qv, use_sh_qv[0],
            "metgrid initialization passes this humidity interpolation choice to shared initialize_real")
    # perturb_input (Registry.EM_COMMON:2368, default .false.): wrf.exe's
    # only consumer is frame/module_io.F:2257-2262, which perturbs every
    # 3-D float field it reads.  Named breakage: NCAR's benchmark namelists
    # write the inert .false. and were refused as an unmapped key.
    perturb_input = dm.take("perturb_input")
    if perturb_input is not None:
        if any(value is not False for value in perturb_input):
            raise _err("domains", "perturb_input", perturb_input,
                       "asks wrf.exe to perturb every 3-D field it reads "
                       "(frame/module_io.F:2257-2262 perturb_real); woof "
                       "has no such input perturbation.  Write .false. or "
                       "remove the key")
        drop("domains", "perturb_input", perturb_input,
             "inert at .false.: wrf.exe perturbs input fields only when "
             "it is .true. (frame/module_io.F:2257-2262)")
    # use_baseparam_fr_nml (&dynamics): WRF's start_em.F:516-523 takes the
    # base-state constants from &dynamics base_* instead of the input file.
    # woof's own initialization already reads its base state from the
    # namelist (base_temp above; the other constants are WRF's defaults and
    # any other value is refused as an unmapped key), and the WRF-input
    # doors compare the files' P00/T00/TLP/TISO/TLP_STRAT/P_STRAT with the
    # namelist's (woof.wrfinput_door.check_namelist_base_state).  Named
    # breakage: NCAR's benchmark namelists write .t. and were refused.
    use_baseparam = dyn.take("use_baseparam_fr_nml")
    if use_baseparam is not None:
        drop("dynamics", "use_baseparam_fr_nml", use_baseparam,
             "base-state constants are the namelist's in woof already; "
             "the WRF-input door refuses files whose P00/T00/TLP/TISO/"
             "TLP_STRAT/P_STRAT differ from them")
    feedback = int(dm.scalar("feedback", 1))
    smooth_option = int(dm.scalar("smooth_option", 2))
    blend_width = int(dm.scalar("blend_width", 5))
    for key, reason in (
            ("num_metgrid_levels", "forcing level count is derived from "
                                   "the input catalog at ingest"),
            ("num_metgrid_soil_levels", "soil level count is derived "
                                        "from the input catalog"),
            ("sfcp_to_sfcp", "interpolation policy key declared in "
                             "[case_data], not imported"),
    ):
        raw = dm.take(key)
        if metgrid_initialization and raw is not None:
            meaning = ("validated against the actual metgrid field/soil inventory"
                       if key != "sfcp_to_sfcp" else
                       "resolved from this requested pressure policy and actual input pressure order")
            fix("domains", key, raw, raw, "metgrid initialization: " + meaning)
        else:
            drop("domains", key, raw, reason)

    # ---- the &domains half of the mp=28 aerosol sweep --------------------
    # It has to run HERE, before dm.finish(), and the &physics half runs
    # with the rest of the physics block far below.  WRF splits WIF's two
    # keys into &domains (Registry/registry.new3d_wif:16-17) while every
    # other aerosol knob is in &physics, and dm.finish() -- which refuses
    # any unconsumed key -- fires long before mp_physics has been mapped.
    # So mp_physics is PEEKED off the unconsumed &physics entries rather
    # than read from the mapped value; peeking cannot consume, so the
    # physics block below still validates mp_physics itself.
    # Any domain asking for 28 is enough to refuse: a mixed column is
    # rejected later by _uniform anyway, and treating the root's value as
    # the whole answer would let `mp_physics = 6, 28` drop a WIF key as
    # inert on its way to that rejection.
    _peeked_mp = ph.entries.get("mp_physics") or ()
    _selects_28 = any(
        isinstance(value, int) and not isinstance(value, bool)
        and value == 28 for value in _peeked_mp)
    for _aero_key, _why in _MP28_AEROSOL_NAMELIST_KEYS["domains"].items():
        _values = dm.take(_aero_key)
        if _values is None:
            continue
        if not _selects_28:
            drop("domains", _aero_key, _values,
                 "inert: WRF consumes this only inside the thompsonaero "
                 "package (Registry/Registry.EM_COMMON:3036, "
                 "mp_physics = 28)")
            continue
        raise _err("domains", _aero_key, _values, _why + ".")

    # ---- the WIF key triple, &domains half -------------------------------
    # ADMITTED, because the ingest exists.  The two keys are consumed here
    # and the decision is carried to the physics block below, which is
    # where use_aero_icbc -- the third leg -- is read: WRF splits the
    # triple across two namelist groups and dm.finish() runs long before
    # &physics is touched, so the halves cannot be validated in one place.
    wif_input_opt_imported = 0
    _wif_values = dm.take("wif_input_opt")
    _levels_values = dm.take("num_wif_levels")
    if _wif_values is not None:
        if not _selects_28:
            drop("domains", "wif_input_opt", _wif_values,
                 "inert: WRF consumes this only inside the thompsonaero "
                 "package (Registry/Registry.EM_COMMON:3036, "
                 "mp_physics = 28)")
        else:
            _wif_opt = int(_uniform("domains", "wif_input_opt",
                                    list(_wif_values)))
            if _wif_opt == WIF_INPUT_OPT_WITH_BLACK_CARBON:
                # One spelling of one configuration fact: the canonical
                # table states what wif_input_opt=2 asks for and what is
                # ported, and this door adds only the way out.  The
                # wrfinput door raises the same two sentences from the
                # same table.
                from woof.config import MP28_AEROSOL_SOURCE_OPTIONS

                _, _, _why = MP28_AEROSOL_SOURCE_OPTIONS["wif_input_opt"]
                raise _err(
                    "domains", "wif_input_opt", _wif_values,
                    f"{_why}. Set wif_input_opt="
                    f"{WIF_INPUT_OPT_CLIMATOLOGY} with aer_init_opt=1 for "
                    "the ported monthly climatology, or remove the "
                    "black-carbon request")
            if _wif_opt not in (0, WIF_INPUT_OPT_CLIMATOLOGY):
                raise _err(
                    "domains", "wif_input_opt", _wif_values,
                    "WRF declares 0, 1 and 2 only "
                    "(Registry/registry.new3d_wif:17)")
            wif_input_opt_imported = _wif_opt
    if _levels_values is not None:
        if wif_input_opt_imported != WIF_INPUT_OPT_CLIMATOLOGY:
            drop("domains", "num_wif_levels", _levels_values,
                 "inert: the WIF vertical axis has no consumer unless "
                 "wif_input_opt=1 selects the climatology ingest "
                 "(Registry/registry.new3d_wif:14-16)")
        else:
            _levels = int(_uniform("domains", "num_wif_levels",
                                   list(_levels_values)))
            if _levels != WIF_DATASET_LEVELS:
                raise _err(
                    "domains", "num_wif_levels", _levels_values,
                    "the ported ingest reads the dataset's own vertical "
                    f"axis, which is {WIF_DATASET_LEVELS} sigma levels "
                    "(QNWFA_QNIFA_SIGMA_MONTHLY.dat, and WRF's Registry "
                    "default is the same 30); a different count describes "
                    "a dataset WOOF has never decoded, and reinterpreting "
                    "it silently would interpolate the wrong column")

    ocean_profile_settings = {}
    for key in ('ocean_levels', 'ocean_z', 'ocean_t', 'ocean_s'):
        raw = dm.take(key)
        if raw is None:
            continue
        if (ph.entries.get('sf_ocean_physics') or [0])[0] != 2:
            drop('domains', key, raw, 'PWP profile is inert unless sf_ocean_physics=2')
            continue
        if key == 'ocean_levels':
            value = _uniform('domains', key, raw)
            if type(value) is not int or not 2 <= value <= 256:
                raise _err('domains', key, raw, 'must be an integer from 2 through 256')
        else:
            value = list(raw)
            if any(isinstance(x, bool) or not isinstance(x, (int,float)) or not math.isfinite(x) for x in value):
                raise _err('domains', key, raw, 'must contain finite profile values')
        ocean_profile_settings[key] = value
        fix('domains', key, raw, value, 'WRF PWP ocean reference profile')
    _consume_unrun_domain_controls(dm, inp, drop, fix, substitutions)
    _consume_unported_package_selectors(dm, "domains", max_dom, drop)
    dm.finish()

    # ---- &geogrid projection ---------------------------------------------
    map_proj = str(geo.scalar("map_proj", "lambert")).lower()
    if map_proj not in ("lambert", "mercator", "polar"):
        from woof.static.projection import (is_latlon_map_proj,
                                             latlon_blocker, latlon_refusal)
        blocker = (
            f" {latlon_refusal()}; {latlon_blocker()}."
            if is_latlon_map_proj(map_proj) else "")
        raise _err("geogrid", "map_proj", map_proj,
                   "implemented projections: 'lambert', 'mercator', "
                   f"'polar'.{blocker}")
    ref_lat = float(geo.required("ref_lat"))
    ref_lon = float(geo.required("ref_lon"))
    truelat1 = float(geo.required("truelat1"))
    if map_proj == "lambert":
        truelat2 = float(geo.required("truelat2"))
        stand_lon = float(geo.required("stand_lon"))
    else:
        # WPS semantics: Mercator ignores truelat2/stand_lon and polar
        # stereographic ignores truelat2; a namelist may omit them.  The
        # emitted [projection] table always carries all six keys.
        truelat2 = float(geo.scalar("truelat2", truelat1))
        stand_lon = float(geo.scalar("stand_lon", ref_lon))
    projection = {
        "map_proj": map_proj,
        "ref_lat": ref_lat,
        "ref_lon": ref_lon,
        "truelat1": truelat1,
        "truelat2": truelat2,
        "stand_lon": stand_lon,
    }
    # s_we/s_sn are the domain's own 1-based start index in its own
    # grid, which runs 1..e_we in WPS and in WRF alike.  They are NOT an
    # offset into the parent -- that is i_parent_start -- so folding a
    # declared start into the layout would translate the domain and
    # produce a different grid.  Normalized to 1 and reported, which is
    # what the WPS reader does with the same key.
    from woof.static.projection import WPS_WINDOW_START

    for key in ("s_we", "s_sn"):
        values = geo.take(key)
        if values is not None:
            fix("geogrid", key, values, WPS_WINDOW_START,
                f"a domain's own grid runs {WPS_WINDOW_START}..e_we, so "
                "the window start normalizes to it; the start is not an "
                "offset into the parent (i_parent_start is), and nothing "
                "about the grid moves")
    # ref_x/ref_y move the reference point off the WPS default centre
    # cell.  That is index arithmetic on a projection already fully
    # resolved, so it is carried exactly rather than refused: the root
    # grid is built with known_x/known_y, then the SAME grid is emitted
    # through the six [projection] keys by re-expressing the reference
    # point at the default centre cell -- the round trip
    # woof/static/projection.py:148-152 already performs.
    ref_x_values = geo.take("ref_x")
    ref_y_values = geo.take("ref_y")
    if ref_x_values is not None or ref_y_values is not None:
        from woof.static.projection import projection_class

        known_x = (float(e_we[0]) / 2.0 if ref_x_values is None
                   else float(ref_x_values[0]))
        known_y = (float(e_sn[0]) / 2.0 if ref_y_values is None
                   else float(ref_y_values[0]))
        if (known_x, known_y) == (float(e_we[0]) / 2.0, float(e_sn[0]) / 2.0):
            # Already the default centre cell: the round trip is the identity,
            # and computing it anyway left platform libm bits in ref_lat
            # (38.49999999999998 on Linux, 38.5 on Windows for a declared 38.5).
            centre_lat, centre_lon = ref_lat, ref_lon
        else:
            declared_grid = projection_class(map_proj)(
                ref_lat=ref_lat, ref_lon=ref_lon, truelat1=truelat1,
                truelat2=truelat2, stand_lon=stand_lon,
                dx=root_dx, dy=root_dy,
                e_we=int(e_we[0]), e_sn=int(e_sn[0]),
                known_x=known_x, known_y=known_y)
            centre_lat, centre_lon = declared_grid.ij_to_latlon(
                float(e_we[0]) / 2.0, float(e_sn[0]) / 2.0)
        projection["ref_lat"] = float(centre_lat)
        projection["ref_lon"] = float(centre_lon)
        for key, values in (("ref_x", ref_x_values), ("ref_y", ref_y_values)):
            if values is None:
                continue
            fix("geogrid", key, values,
                "[projection] ref_lat/ref_lon at the grid centre",
                f"the reference point is carried exactly: the root grid is "
                f"built with known_x/known_y = ({known_x}, {known_y}) and "
                "the emitted six-key [projection] table names the same "
                "geometry at the WPS default centre cell, ref_lat/ref_lon "
                f"= ({float(centre_lat):.6f}, {float(centre_lon):.6f}); "
                "[projection] has no ref_x/ref_y key to carry")
    # geog_data_res: on a route that builds its own statics (no input file
    # supplies them), a land-cover token becomes the [static.highres]
    # block that builds that collection, and a token nothing here builds
    # is refused by name.  The GEOG-tree tokens (default, 30s, 5m, modis_lai)
    # stay where they are read: the namelist.wps [case_data] names.
    geog_data_res = geo.take("geog_data_res")
    static_landcover = None
    if landuse_identity is None:
        static_landcover = _geog_data_res_landcover(
            geog_data_res, max_dom=max_dom, grid_ids=grid_ids,
            spacings=[float(value) for value in dx_exact])
        geog_notice = _geog_thirty_second_notice(
            geog_data_res, grid_ids,
            highres_landcover=(None if static_landcover is None
                              else static_landcover.source_id))
        if geog_notice is not None:
            notices.append(geog_notice)
    if static_landcover is None:
        drop("geogrid", "geog_data_res", geog_data_res,
             "GEOG dataset/resolution selection is static-build "
             "configuration, not imported")
    for key, reason in (
            ("geog_data_path", "geography root is declared in "
                               "[case_data] (or --geog-root), not "
                               "imported"),
            ("opt_geogrid_tbl_path", "WPS table path"),
    ):
        drop("geogrid", key, geo.take(key), reason)
    geo.finish()

    # ---- &physics: ratified scheme maps ----------------------------------
    # Inspect the complete requested suite before consuming individual keys.
    # This keeps unfinished ports fail-closed while reporting Thompson, the
    # coupled MYNN stack, and RUC together instead of stopping at the first
    # unsupported selector.  ``peek`` follows WRF's repeat-last convention but
    # does not mutate _Section, so ordinary parsing remains authoritative.
    def _peek_uniform_int(key: str, default=None) -> int | None:
        values = ph.entries.get(key)
        if values is None:
            return default
        values = list(values[:max_dom])
        values += [values[-1]] * (max_dom - len(values))
        return int(_uniform("physics", key, values))

    def _peek_col_int(key: str) -> list | None:
        """The padded per-domain column, WRF's repeat-last convention."""
        values = ph.entries.get(key)
        if values is None:
            return None
        values = list(values[:max_dom])
        values += [values[-1]] * (max_dom - len(values))
        return [int(v) for v in values]

    # bl_pbl_physics is a PER-DOMAIN column (see the _mapped call below):
    # woof.experiment._DOMAIN_RUN_OVERRIDES admits it per domain so a PBL
    # parent can carry a PBL-off LES child, and the importer has to be able
    # to read back what the experiment schema can express.  The readiness
    # preview runs once per DISTINCT value so every domain's suite is
    # refused on exactly the grounds it was refused on before.
    bl_preview_col = _peek_col_int("bl_pbl_physics")
    preview = {
        "mp_physics": _peek_uniform_int("mp_physics"),
        "sf_sfclay_physics": _peek_uniform_int("sf_sfclay_physics"),
        "bl_pbl_physics": (None if bl_preview_col is None
                           else bl_preview_col[0]),
        "sf_surface_physics": _peek_uniform_int("sf_surface_physics", 0),
        # Registry.EM_COMMON:2504 defaults to 5.  The target suite supplies
        # 9 explicitly; retaining the real default here avoids inventing a
        # RUC geometry in diagnostics.
        "num_soil_layers": _peek_uniform_int("num_soil_layers", 5),
    }
    if all(preview[key] is not None for key in (
            "mp_physics", "sf_sfclay_physics", "bl_pbl_physics")):
        for _bl in dict.fromkeys(bl_preview_col):
            require_ready_wrf_physics(**{**preview, "bl_pbl_physics": _bl})

    def _mapped(key: str, table: dict, per_domain=False):
        values = ph.col(key, max_dom)
        if values is None:
            raise ValueError(
                f"{input_path} &physics is missing {key}.")
        values = [int(v) for v in values]
        if not per_domain:
            value = _uniform("physics", key, values)
            values = [value]
        out = []
        for value in values:
            if value not in table:
                raise _err("physics", key, value,
                           f"no ratified woof mapping (implemented: "
                           f"{sorted(table)})."
                           + _unported_tail(key, value))
            out.append(table[value])
        return values, out

    # WRF v4.6.1 no longer has separate NSSL scheme IDs: 17/19/21/22 are
    # compatibility spellings that share/module_check_a_mundo.F:3382-3421
    # rewrites onto mp_physics=18 plus explicit variant flags, printing a
    # deprecation CAUTION as it goes.  Doing the same rewrite here is a
    # canonicalization, not a substitution -- the scheme the user gets is
    # the scheme WRF would have given them -- so it is reported as an
    # applied default rather than added to the substitution ledger.
    _nssl_deprecated_flags: dict[str, int] = {}
    _nssl_deprecated_id: int | None = None
    if "mp_physics" in ph.entries:
        from woof.core.nssl2_contract import (
            DEPRECATED_MP_PHYSICS_FLAGS as _NSSL_DEPRECATED,
        )
        _raw_mp = ph.entries["mp_physics"]
        _raw_mp = _raw_mp if isinstance(_raw_mp, list) else [_raw_mp]
        _deprecated_seen = {
            int(value) for value in _raw_mp
            if isinstance(value, int) and int(value) in _NSSL_DEPRECATED}
        if len(_deprecated_seen) > 1:
            raise _err(
                "physics", "mp_physics", sorted(_deprecated_seen),
                "two different deprecated NSSL scheme IDs in one namelist "
                "resolve to different variant flags; WRF's per-domain "
                "rewrite (module_check_a_mundo.F:3382-3421) cannot be "
                "represented by woof's single NSSL selector set")
        if _deprecated_seen:
            _nssl_deprecated_id = _deprecated_seen.pop()
            if any(int(value) != _nssl_deprecated_id for value in _raw_mp):
                raise _err(
                    "physics", "mp_physics", _raw_mp,
                    "a deprecated NSSL scheme ID must apply to every domain: "
                    "its variant flags are whole-run in woof, while WRF's "
                    "nssl_hail_on is per-domain "
                    "(Registry.EM_COMMON:2420)")
            _nssl_deprecated_flags = dict(_NSSL_DEPRECATED[_nssl_deprecated_id])
            ph.entries["mp_physics"] = [18 for _ in _raw_mp]

    # Every unported WRF package (the P3 and WDM siblings, ISHMAEL) would
    # otherwise fall out of _mapped as the generic "no ratified woof
    # mapping" error, which tells a user who asked for two ice categories
    # only that a number is unknown.  Raise woof.config's per-package
    # explanation instead, before the map lookup, keyed so the all-refusals
    # loop can set the selector aside and keep checking the pair.
    # PEEK, never ph.col: col() CONSUMES the entry, and _mapped below needs
    # it.  ph.entries is the same non-consuming read the WIF block at :1285
    # uses for exactly this reason.
    from woof.config import unported_mp_scheme_refusal
    for _mp_value in (ph.entries.get("mp_physics") or ()):
        try:
            _mp_int = int(str(_mp_value).strip().rstrip(","))
        except ValueError:
            continue
        _named = unported_mp_scheme_refusal(_mp_int)
        if _named is not None:
            raise NamelistRefusal(f"{input_path} &physics: " + _named,
                                  [("physics", "mp_physics")])

    mp_wrf, mp_mapped = _mapped("mp_physics", _MP_MAP)
    mp_physics, wrf_name, gp_name = mp_mapped[0]
    if mp_physics != mp_wrf[0]:
        substitutions.append(Substitution(
            key="mp_physics", wrf_value=mp_wrf[0], wrf_name=wrf_name,
            gpuwm_key="mp_physics", gpuwm_value=mp_physics,
            gpuwm_name=gp_name))
    # Scalar WRF option, Registry.EM_COMMON:2663-2666: default 1 = hail,
    # explicit 0 = graupel.  It selects AG/BG/RHOG before Morrison derives
    # CG and its gamma/exponent products (module_mp_morr_two_moment.F:
    # 337-411, :483-510).
    # WRF ccn_conc (Registry.EM_COMMON:2664, default 1.0E8 # m-3).  Only
    # WDM5/6/7 and NTU consume the namelist value. NSSL overwrites its
    # grid%ccn_conc with nssl_cccn/1.225 in start_em.F:1754-1758 before
    # flow_dep_bdy_qnn uses it. Emit this value only for supported mp=16.
    ccn_conc_supplied = "ccn_conc" in ph.entries
    ccn_conc = float(ph.scalar("ccn_conc", 1.0e8))
    if not (ccn_conc > 0.0):
        raise _err("physics", "ccn_conc", ccn_conc,
                   "must be a positive CCN number concentration in # m-3.")
    morr_rimed_ice = int(ph.scalar("morr_rimed_ice", 1))
    if morr_rimed_ice not in (0, 1):
        raise _err("physics", "morr_rimed_ice", morr_rimed_ice,
                   "must be 0 (graupel) or 1 (hail, WRF default).")
    # WRF Registry.EM_COMMON:2665, scalar default 0.  This is a distinct
    # WSM6 switch; do not alias Morrison's opposite-default selection.
    hail_opt_supplied = "hail_opt" in ph.entries
    wsm6_hail_opt = int(ph.scalar("hail_opt", 0))
    if wsm6_hail_opt not in (0, 1):
        raise _err("physics", "hail_opt", wsm6_hail_opt,
                   "must be 0 (graupel, WRF WSM6 default) or 1 (hail).")
    # Recorded where inert, so a supplied value never vanishes unexplained
    # (the 2026-10-12 door audit found both consumed with no record).
    if hail_opt_supplied and mp_physics not in (6, 16):
        drop("physics", "hail_opt", [wsm6_hail_opt],
             "read only by WSM6 and WDM6 (module_physics_init.F:4487, "
             f":4582-4584); mp_physics = {mp_physics} does not read it")
    if ccn_conc_supplied and mp_physics != 16:
        drop("physics", "ccn_conc", [ccn_conc],
             "read only by the WDM schemes and NTU (start_em.F; NSSL "
             f"overwrites it from nssl_cccn); mp_physics = {mp_physics} "
             "does not read it")

    # ---- RunConfig-honored &physics knobs (knob-parity lane) -----------
    # Each maps 1:1 onto an existing consumed RunConfig field and is
    # emitted only when the namelist supplies it: every WRF Registry
    # default below equals woof's frozen RunConfig default, so omission
    # resolves identically on both sides and established imports stay
    # byte-identical.
    def _optional_scalar_int(key: str, allowed: tuple, why: str):
        values = ph.take(key)
        if values is None:
            return None
        value = values[0]
        if isinstance(value, bool) or not isinstance(value, int) \
                or value not in allowed:
            raise _err("physics", key, values, why)
        return value

    def _optional_scalar_bool(key: str):
        values = ph.take(key)
        if values is None:
            return None
        return _require_bools("physics", key, values)[0]

    no_mp_heating = _optional_scalar_int(
        "no_mp_heating", *PHYSICS_CHOICES["no_mp_heating"])
    # mp_zero_out (Registry.EM_COMMON, both sources, &physics, one value)
    # and its threshold, mapped 1:1; v4.6.1's mp_zero_out_all likewise.
    # The fork declares no mp_zero_out_all and applies the pass to every
    # array (solve_em.F:4078-4141), so a fork namelist that sets
    # mp_zero_out imports with mp_zero_out_all = 1, the v4.6.1 spelling of
    # the same call set.
    mp_zero_out = _optional_scalar_int(
        "mp_zero_out", *PHYSICS_CHOICES["mp_zero_out"])
    mp_zero_out_thresh_values = ph.take("mp_zero_out_thresh")
    mp_zero_out_thresh = None
    if mp_zero_out_thresh_values is not None:
        mp_zero_out_thresh = float(mp_zero_out_thresh_values[0])
        if not math.isfinite(mp_zero_out_thresh):
            raise _err("physics", "mp_zero_out_thresh",
                       mp_zero_out_thresh_values,
                       "must be a finite mixing ratio in kg/kg (WRF "
                       "Registry default 1e-8).")
    mp_zero_out_all = _optional_scalar_int(
        "mp_zero_out_all", *PHYSICS_CHOICES["mp_zero_out_all"])
    if (mp_zero_out_all is None and noaa_wrf39_namelist
            and mp_zero_out):
        mp_zero_out_all = 1
        notices.append(
            "mp_zero_out_all = 1: the namelist names diff_6th_factor2, a "
            "key only the NOAA-EMC WRFV3.9 fork declares, and that fork "
            "applies mp_zero_out to the scalar arrays as well "
            "(solve_em.F:4078-4141).")
    mp_tend_lim_values = ph.take("mp_tend_lim")
    mp_tend_lim = None
    if mp_tend_lim_values is not None:
        mp_tend_lim = float(mp_tend_lim_values[0])
        if not math.isfinite(mp_tend_lim) or mp_tend_lim <= 0.0:
            raise _err("physics", "mp_tend_lim", mp_tend_lim_values,
                       "must be a finite positive microphysics theta-"
                       "tendency clamp in K/s (WRF Registry default 10.0).")
    ysu_topdown_pblmix = _optional_scalar_int(
        "ysu_topdown_pblmix", *PHYSICS_CHOICES["ysu_topdown_pblmix"])
    # swint_opt is a radiation-DRIVER option (module_radiation_driver.F
    # update_swinterp_parameters / interp_sw_radiation), independent of
    # the shortwave scheme, so it is honoured here beside the other
    # RunConfig knobs; validate_run_config refuses 1 under ra_sw_physics=0.
    swint_opt = _optional_scalar_int(
        "swint_opt", *PHYSICS_CHOICES["swint_opt"])
    if swint_opt == 0:
        # The default: receipted, no TOML line (byte-identical to the
        # import of a namelist that omits it, as while it was pinned).
        swint_opt = None
    alb_sol_values = ph.take("alb_sol")
    alb_sol = None
    if alb_sol_values is not None:
        alb_sol = _uniform("physics", "alb_sol", alb_sol_values)
        if type(alb_sol) is not int or alb_sol not in (0, 1):
            raise _err("physics", "alb_sol", alb_sol_values,
                       PHYSICS_CHOICES["alb_sol"][1])
    if alb_sol == 0:
        # Explicit off and omission retain the same configuration bytes.
        alb_sol = None
    def _optional_mynn_choice(key):
        values = ph.take(key)
        if values is None:
            return None
        allowed, why = PHYSICS_CHOICES[key]
        if any(type(value) is not int or value not in allowed for value in values):
            raise _err("physics", key, values, why)
        if len(set(values)) != 1:
            raise _err("physics", key, values,
                       "is a shared setting in woof; differing domain "
                       "values would silently change the selected physics.")
        return values[0]

    bl_mynn_mixlength = _optional_mynn_choice("bl_mynn_mixlength")
    tkeadvect_values = ph.take("bl_mynn_tkeadvect")
    bl_mynn_tkeadvect = None
    if tkeadvect_values is not None:
        if any(type(value) is not bool for value in tkeadvect_values):
            raise _err("physics", "bl_mynn_tkeadvect", tkeadvect_values,
                       "must be a logical (.true. or .false.)")
        bl_mynn_tkeadvect = _uniform("physics", "bl_mynn_tkeadvect", tkeadvect_values)
        if bl_mynn_tkeadvect is False:
            # A stated .false. is the RunConfig default and what every
            # earlier import of the same namelist ran; it is consumed here
            # and not written, so those imports keep their bytes (the HRRR
            # operational namelists all state .false.).
            bl_mynn_tkeadvect = None
    scalar_pblmix = _optional_mynn_choice("scalar_pblmix")
    bl_mynn_mixscalars = _optional_mynn_choice("bl_mynn_mixscalars")
    # bl_mynn_tkebudget is the WRF 3.x spelling of MYNN's TKE-budget switch
    # (renamed tke_budget, &dynamics, in WRF 4.0).  A namelist that spells
    # it this way was written for a WRF 3.x-generation MYNN, and the one
    # such generation the solver carries is the GSD MYNN v4.1 of the
    # NOAA-EMC WRF 3.9 branch, so the import selects bl_mynn_version =
    # "gsd_41".  Only 0 imports: the column solver writes no budget terms
    # (the driver refuses tke_budget = 1 for the same reason).
    bl_mynn_version = None
    tkebudget_values = ph.take("bl_mynn_tkebudget")
    if tkebudget_values is not None:
        if any(type(value) is not int or value != 0
               for value in tkebudget_values):
            raise _err("physics", "bl_mynn_tkebudget", tkebudget_values,
                       "must be 0: the MYNN column solver writes no TKE "
                       "budget terms.")
        bl_mynn_version = "gsd_41"
    if "bl_mynn_version" in carried_selectors:
        bl_mynn_version = carried_selectors["bl_mynn_version"]
    isfflx = _optional_scalar_int(
        "isfflx", *PHYSICS_CHOICES["isfflx"])
    if isfflx == 1:
        fix(
            "physics", "isfflx", [1], 1,
            "the established surface heat/moisture-flux-on value")
        isfflx = None
    use_mp_re = _optional_scalar_int(
        "use_mp_re", *PHYSICS_CHOICES["use_mp_re"])
    if use_mp_re == 1:
        fix(
            "physics", "use_mp_re", [1], 1,
            "the established WRF microphysics effective-radius scheme-table "
            "value")
        use_mp_re = None
    isftcflx = _optional_scalar_int(
        "isftcflx", *PHYSICS_CHOICES["isftcflx"])
    iz0tlnd = _optional_scalar_int(
        "iz0tlnd", *PHYSICS_CHOICES["iz0tlnd"])
    usemonalb = _optional_scalar_bool("usemonalb")
    rdlai2d = _optional_scalar_bool("rdlai2d")
    if operational_ruc_requested:
        for key in ("rdlai2d", "usemonalb"):
            if key not in prescribed_surface_supplied:
                if key == "rdlai2d":
                    rdlai2d = True
                else:
                    usemonalb = True
                defaults_applied.append(AppliedDefault(
                    key=key, value=True,
                    reason="operational RUC configuration requests the "
                           "prescribed monthly surface field"))
    rdmaxalb = _optional_scalar_bool("rdmaxalb")
    seaice_albedo_default_values = ph.take("seaice_albedo_default")
    seaice_albedo_default = None
    if seaice_albedo_default_values is not None:
        seaice_albedo_default = float(seaice_albedo_default_values[0])
        if (not math.isfinite(seaice_albedo_default)
                or not 0.0 <= seaice_albedo_default <= 1.0):
            raise _err(
                "physics", "seaice_albedo_default",
                seaice_albedo_default_values,
                "must be a finite albedo fraction in [0, 1].")
    opt_thcnd = _optional_scalar_int(
        "opt_thcnd", *PHYSICS_CHOICES["opt_thcnd"])

    # ---- &physics keys the model pins where WRF has options ------------
    # Present keys are validated against the single implemented value and
    # recorded as fixed; any other value is a hard error, never a silent
    # reinterpretation.
    # NSCV beside New SAS is refused by WRF itself
    # (module_check_a_mundo.F:1094) and here BY NAME, ahead of the generic
    # shcu_physics pin, because the reason is not that NSCV is unported:
    # cu_physics = 96 already runs NSCV (nscv2d) inside its own call, so a
    # second NSCV would convect the same boundary layer twice -- double the
    # shallow mass flux, heating and drying for one physical process.
    _peeked_cu = ph.entries.get("cu_physics") or ()
    _peeked_shcu = ph.entries.get("shcu_physics") or ()
    for _n, _shcu in enumerate(_peeked_shcu):
        _cu_n = (_peeked_cu[min(_n, len(_peeked_cu) - 1)]
                 if _peeked_cu else None)
        if _shcu == 4 and _cu_n == 96:
            raise _err(
                "physics", "shcu_physics", list(_peeked_shcu),
                "shcu_physics = 4 (NSCV) beside cu_physics = 96 is refused, "
                "as WRF refuses it (module_check_a_mundo.F:1094): New SAS "
                "already runs NSCV shallow convection inside its own call, "
                "so a second NSCV would convect the same boundary layer "
                "twice. Set shcu_physics = 0.")
    for key, pin, why in PHYSICS_PINS:
        raw = ph.take(key)
        if raw is None:
            continue
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value != pin for value in raw):
            raise _err("physics", key, raw,
                       f"woof implements {key} = {pin} only ({why}).")
        fix("physics", key, raw, pin, why)
    from woof.surface_ocean import ocean_namelist_defaults
    ocean_selection_explicit = bool(ph.entries.get("sf_ocean_physics"))
    ocean_settings = {}
    for key, default in {"sf_ocean_physics": 0, **ocean_namelist_defaults()}.items():
        raw = ph.take(key)
        value = default if raw is None else _uniform("physics", key, raw)
        if key == "sf_ocean_physics":
            if type(value) is not int or value not in (0, 1, 2):
                raise _err("physics", key, raw, "must be 0 (off), 1 (Pollard) or 2 (3DPWP)")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise _err("physics", key, raw, "must be finite; nonfinite settings corrupt mixed-layer state")
        ocean_settings[key] = value
        if raw is not None:
            fix("physics", key, raw, value, "WRF scalar ocean setting")
    if ocean_settings["sf_ocean_physics"] and ocean_settings["oml_gamma"] <= 0:
        raise _err("physics", "oml_gamma", ocean_settings["oml_gamma"], "must be positive: OML1D divides by the lapse rate and buoyancy frequency")
    raw = ph.take('omdt')
    if raw is not None:
        if ocean_settings['sf_ocean_physics'] == 2:
            value=_uniform('physics','omdt',raw)
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<=0:
                raise _err('physics','omdt',raw,'must be positive minutes; DPWP multiplies OMDT by 60')
            ocean_settings['omdt']=value
            fix('physics','omdt',raw,value,'maximum PWP substep in minutes; the surface interval is fully advanced')
        else:
            drop('physics','omdt',raw,'only DPWP reads OMDT; Pollard uses the surface physics interval')
    raw = ph.take('omlcall')
    if raw is not None:
        if any(value != 0 for value in raw):
            raise _err('physics','omlcall',raw,'retired WRF rename sentinel, not an ocean cadence; a nonzero value does not select a defined coupled scheme')
        drop('physics','omlcall',raw,'zero retired WRF rename sentinel, Registry.EM_COMMON:2613')
    lake_raw = ph.take("sf_lake_physics")
    lake_column = [0] * max_dom
    if lake_raw is not None:
        for value in lake_raw[:max_dom]:
            if type(value) is not int or value not in (0, 1):
                raise _err("physics", "sf_lake_physics", lake_raw,
                           "must be integer 0 (off) or 1 (CLM lake).")
        lake_column[:min(max_dom, len(lake_raw))] = lake_raw[:max_dom]
        fix("physics", "sf_lake_physics", lake_raw, lake_column,
            "WRF per-domain selector; an omitted domain retains default 0")
    lake_settings = {"sf_lake_physics": lake_column}
    for key, default in (("use_lakedepth", 1), ("lakedepth_default", 50.0),
                         ("lake_min_elev", 5.0)):
        raw = ph.take(key)
        column = [default] * max_dom
        if raw is not None:
            for index, value in enumerate(raw[:max_dom]):
                if key == "use_lakedepth":
                    if type(value) is not int or value not in (0, 1):
                        raise _err("physics", key, raw, "must be integer 0 or 1.")
                elif (isinstance(value, bool) or not isinstance(value, (int, float))
                      or not math.isfinite(value)):
                    raise _err("physics", key, raw,
                               "must be finite in metres.")
                column[index] = value
            fix("physics", key, raw, column,
                f"WRF per-domain lake setting; omitted domains retain {default}")
        lake_settings[key] = column
    # ---- slope-dependent shortwave and terrain shadowing ---------------
    # WRF's &physics slope_rad and topo_shading are max_domains columns
    # (Registry.EM_COMMON; an omitted tail keeps the Registry default 0)
    # and shadlen is a scalar.  woof.core.topo_radiation ports them from
    # WRF v4.7.1 (module_radiation_driver.F toposhad,
    # module_surface_driver.F TOPO_RAD_ADJ), so they translate 1:1 and are
    # emitted only where set: an omitted key keeps every import
    # byte-identical.
    topo_columns = {}
    terrain_drag_columns: dict[str, list[int]] = {}
    for key in ("slope_rad", "topo_shading"):
        raw = ph.take(key)
        if raw is None:
            continue
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value not in (0, 1) for value in raw):
            raise _err("physics", key, raw,
                       "must be 0 or 1 on every domain (WRF "
                       "Registry.EM_COMMON: 1 turns it on).")
        column = list(raw[:max_dom])
        topo_columns[key] = column + [0] * (max_dom - len(column))
    shadlen_values = ph.take("shadlen")
    shadlen = None
    if shadlen_values is not None:
        shadlen = float(shadlen_values[0])
        if not math.isfinite(shadlen) or shadlen <= 0.0:
            raise _err("physics", "shadlen", shadlen_values,
                       "must be a finite positive length in metres (WRF "
                       "default 25000.).")
    cu_rad_feedback = ph.take("cu_rad_feedback")
    if cu_rad_feedback is not None:
        if any(_require_bools("physics", "cu_rad_feedback",
                              cu_rad_feedback)):
            raise _err("physics", "cu_rad_feedback", cu_rad_feedback,
                       "cumulus cloud fraction does not feed radiation in "
                       "woof (WRF Registry default .false.); only the "
                       "disabled branch is implemented.")
        fix("physics", "cu_rad_feedback", cu_rad_feedback, False,
            "cumulus-radiation feedback is not implemented (the WRF "
            "Registry default .false. branch is the implemented one)")

    # ---- MYNN option identity ------------------------------------------
    # woof's MYNN port is validated at exactly one value of each option
    # (woof/config.py MYNN_PBL_OPTION_IDENTITY); a present key at the
    # identity value is fixed, anything else refuses.
    for option, admitted in MYNN_PBL_OPTION_IDENTITY.items():
        values = ph.take(option)
        if values is None:
            continue
        if any(not _identity_matches(value, admitted) for value in values):
            raise _err(
                "physics", option, values,
                f"woof implements {option} = {admitted!r} only (MYNN "
                "option identity, woof/config.py); no nearby branch is "
                "substituted for an unported one.")
        fix("physics", option, values, admitted,
            "MYNN option identity: the ported solver is validated at "
            "exactly this value")

    # ---- NSSL parameters -------------------------------------------------
    # The mp18 port runs at the WRF v4.6.1 Registry defaults pinned by
    # woof/core/nssl2_contract.py; tunable NSSL parameters are not yet
    # plumbed.  Under any other scheme the keys are inert in WRF too.
    #: The variant selectors, which import natively.  Everything else under
    #: the nssl_ prefix is a tunable coefficient and stays pinned.
    _NSSL_VARIANT_SELECTORS = (
        "nssl_2moment_on", "nssl_hail_on", "nssl_ccn_on",
        "nssl_density_on", "nssl_3moment",
    )
    nssl_selectors: dict[str, int] = {}
    nssl_keys = sorted(key for key in ph.entries if key.startswith("nssl_"))
    if nssl_keys or _nssl_deprecated_flags:
        from woof.core.nssl2_contract import (
            CONTRACT_ID as _NSSL_CONTRACT_ID,
            WRF_NAMELIST_DEFAULTS as _NSSL_DEFAULTS,
        )
        for key in nssl_keys:
            values = ph.take(key)
            if key not in _NSSL_DEFAULTS:
                raise _err(
                    "physics", key, values,
                    "no ratified NSSL parameter mapping (extend "
                    "woof/core/nssl2_contract.py first).")
            if mp_physics != 18:
                drop("physics", key, values,
                     "inert: NSSL parameters are consumed only under "
                     "mp_physics = 18")
                continue
            if key in _NSSL_VARIANT_SELECTORS:
                # WRF's nssl_hail_on is the one per-domain NSSL selector
                # (Registry.EM_COMMON:2420, max_domains); the other four are
                # scalars.  woof resolves ONE variant for the whole run, so
                # a per-domain hail split is refused by name rather than
                # collapsed to whichever domain happened to be first.
                if len({int(value) for value in values}) > 1:
                    raise _err(
                        "physics", key, values,
                        "woof resolves one NSSL variant for the whole run; "
                        "a per-domain split of this selector would change "
                        "which prognostic fields exist between domains")
                if key in _nssl_deprecated_flags and (
                        int(values[0]) != _nssl_deprecated_flags[key]):
                    raise _err(
                        "physics", key, values,
                        f"mp_physics = {_nssl_deprecated_id} already forces "
                        f"{key} = {_nssl_deprecated_flags[key]} "
                        "(module_check_a_mundo.F:3382-3421); the namelist "
                        "asks for a value WRF would have overwritten")
                nssl_selectors[key] = int(values[0])
                continue
            admitted = _NSSL_DEFAULTS[key]
            if any(not _identity_matches(value, admitted)
                   for value in values):
                raise _err(
                    "physics", key, values,
                    f"the NSSL two-moment port runs at the WRF v4.6.1 "
                    f"Registry default {key} = {admitted!r} only "
                    f"(contract {_NSSL_CONTRACT_ID}); tunable NSSL "
                    "parameters are not plumbed.")
            fix("physics", key, values, admitted,
                f"NSSL parameter pinned at its Registry default by "
                f"contract {_NSSL_CONTRACT_ID}")

        for _flag, _value in _nssl_deprecated_flags.items():
            nssl_selectors.setdefault(_flag, _value)

        if mp_physics == 18:
            # Resolve once here so an unported variant is named at import
            # time, against the namelist key the user wrote, instead of
            # surfacing later as a RunConfig validation error about
            # selectors they never typed.
            from woof.core.nssl2_contract import (
                require_ported_nssl2_mode as _nssl_require,
                resolve_nssl2_mode as _nssl_resolve,
            )
            try:
                _nssl_require(_nssl_resolve(**nssl_selectors))
            except ValueError as _nssl_exc:
                raise _err(
                    "physics",
                    "mp_physics"
                    if _nssl_deprecated_id is not None
                    else "/".join(sorted(nssl_selectors)),
                    _nssl_deprecated_id
                    if _nssl_deprecated_id is not None
                    else nssl_selectors,
                    f"{_nssl_exc}. No nearby NSSL branch is substituted for "
                    "an unported one") from _nssl_exc
            if _nssl_deprecated_id is not None:
                defaults_applied.append(AppliedDefault(
                    key="mp_physics",
                    value=("18 with "
                           + ", ".join(f"{name} = {value}" for name, value
                                       in sorted(
                                           _nssl_deprecated_flags.items()))),
                    reason=(
                        f"WRF v4.6.1 deprecated mp_physics = "
                        f"{_nssl_deprecated_id} and rewrites it onto option "
                        "18 plus these variant flags before any physics runs "
                        "(share/module_check_a_mundo.F:3382-3421). woof "
                        "performs the same rewrite, so this is WRF's own "
                        "canonicalization and not a woof substitution."),
                ))

    # ---- aerosol-aware Thompson keys ------------------------------------
    # Same shape as the NSSL sweep above and for the same structural
    # reason: _Section.finish() refuses any unconsumed key, so every one of
    # WRF's mp=28 aerosol knobs must be answered here or an otherwise valid
    # namelist becomes unimportable.  Inert under any other scheme; refused
    # by name under 28.  See _MP28_AEROSOL_NAMELIST_KEYS for the per-key
    # citation.
    # The &domains half already ran above, before dm.finish().
    for _aero_key, _why in _MP28_AEROSOL_NAMELIST_KEYS["physics"].items():
        _values = ph.take(_aero_key)
        if _values is None:
            continue
        if mp_physics != 28:
            drop("physics", _aero_key, _values,
                 "inert: WRF consumes this only inside the "
                 "thompsonaero package "
                 "(Registry/Registry.EM_COMMON:3036, mp_physics = 28)")
            continue
        if _aero_key == "grav_settling" and all(
                value == 0 for value in _values[:max_dom]):
            # 0 is the value WRF itself runs under mp=28 (it forces it
            # there) and the only behaviour ArWen has: Thompson's own
            # settling, no separate fog-settling pass.  Nothing is
            # overwritten, so there is no breakage to refuse.
            fix("physics", _aero_key, _values, 0,
                "no separate fog-settling pass, which is what WRF runs "
                "under mp_physics = 28 (share/module_check_a_mundo.F:"
                "2459-2474); the scheme's own settling applies")
            continue
        raise _err("physics", _aero_key, _values, _why + ".")
    # ---- the WIF key triple, &physics half -------------------------------
    # use_aero_icbc is what real.exe reads to DERIVE aer_init_opt=1
    # (dyn_em/module_initialize_real.F:2325-2732).  The two halves are
    # cross-checked because either alone is a namelist that does not do
    # what its author thinks: WRF fatals mp=28 with wif_input_opt=0
    # (:2735-2736), and wif_input_opt=1 without use_aero_icbc allocates
    # the WIF arrays with nothing to fill them.
    _aero_icbc_values = ph.take("use_aero_icbc")
    _rap_aero_values = ph.take("use_rap_aero_icbc")
    analyzed_aerosol_imported = False
    if _rap_aero_values is not None:
        if mp_physics != 28:
            drop("physics", "use_rap_aero_icbc", _rap_aero_values,
                 "inert: analyzed water/ice-friendly aerosols require mp_physics=28")
        else:
            analyzed_aerosol_imported = bool(_uniform(
                "physics", "use_rap_aero_icbc", list(_rap_aero_values)))
    _aero_icbc = False
    if _aero_icbc_values is not None:
        if mp_physics != 28:
            drop("physics", "use_aero_icbc", _aero_icbc_values,
                 "inert: WRF consumes this only inside the thompsonaero "
                 "package (Registry/Registry.EM_COMMON:3036, "
                 "mp_physics = 28)")
        else:
            _aero_icbc = bool(_uniform("physics", "use_aero_icbc",
                                       list(_aero_icbc_values)))
    _wif_selected = (wif_input_opt_imported == WIF_INPUT_OPT_CLIMATOLOGY)
    wif_climatology_imported = False
    if _wif_selected and mp_physics != 28:
        # The &domains sweep PEEKED at the whole mp_physics column and any
        # domain asking for 28 let the key through; the mapped, uniform
        # answer is the one that governs.  `mp_physics = 6, 28` reaches
        # here with the key consumed and mp_physics 6, and dropping it as
        # inert at that point would accept a WIF selection under a scheme
        # that has no WIF package at all.
        raise _err(
            "domains", "wif_input_opt", (wif_input_opt_imported,),
            "wif_input_opt=1 selects the use_wif_input package, which WRF "
            "declares only inside thompsonaero "
            "(Registry/Registry.EM_COMMON:3036, mp_physics = 28); this "
            f"namelist resolves to mp_physics={mp_physics}.  WRF fatals "
            "mp_physics=28 with wif_input_opt=0 "
            "(dyn_em/module_initialize_real.F:2735-2736) and has no "
            "consumer for the reverse, so the pair has to move together")
    # The index of the mp=28 synthetic-fallback receipt row, if one is
    # written below, so the run-door clause can be appended to it once
    # &bdy_control has been parsed (the clause is about ``specified``).
    _mp28_fallback_row: int | None = None
    # The analyzed request beside a COMPLETE WIF triple (use_aero_icbc
    # with wif_input_opt = 1) keeps the triple in the TOML: the HRRR
    # route's renderer writes exactly this for a configuration stating
    # aer_init_opt = wif_input_opt = 1 with use_rap_aero_icbc (the shipped
    # configs/recipes/hrrr_v4_gsd41.toml and hrrr_configuration_cut.toml),
    # and reading it back as (0, 0, 'analysis') refused that round trip.
    # Both spellings run the analyzed source: use_rap_aero_icbc decides it
    # before the pair or mp28_aerosol_source is consulted
    # (woof/ingest/real.py _resolved_mp28_aerosol_source).  A namelist
    # with only one half of the triple, the operational hrrr_wrf.nl among
    # them, imports exactly as before.
    analyzed_with_wif_triple = (analyzed_aerosol_imported and _aero_icbc
                                and _wif_selected)
    if analyzed_aerosol_imported:
        defaults_applied.append(AppliedDefault(
            key="mp28 aerosol initial state", value="analyzed aerosol IC/BC",
            reason="use_rap_aero_icbc selects QNWFA/QNIFA from the driving analysis on every initial and boundary frame; missing fields refuse instead of substituting climatology"
                   + ("; the namelist's complete WIF triple (use_aero_icbc with wif_input_opt = 1) is carried as aer_init_opt = wif_input_opt = 1, which the analyzed request overrides" if analyzed_with_wif_triple else "")))
    elif mp_physics == 28 and (_aero_icbc or _wif_selected):
        if not (_aero_icbc and _wif_selected):
            raise _err(
                "physics", "use_aero_icbc",
                _aero_icbc_values if _aero_icbc_values is not None
                else (False,),
                "the WIF climatology is a KEY TRIPLE and this namelist "
                "wrote half of it: &physics use_aero_icbc = .true. WITH "
                "&domains wif_input_opt = 1 (and num_wif_levels = 30).  "
                f"Here use_aero_icbc={_aero_icbc!r} and "
                f"wif_input_opt={wif_input_opt_imported!r}.  WRF fatals "
                "mp_physics=28 with wif_input_opt=0 "
                "(dyn_em/module_initialize_real.F:2735-2736), and "
                "wif_input_opt=1 without use_aero_icbc allocates the WIF "
                "arrays with nothing to fill them; neither half alone "
                "names a run either model performs")
        wif_climatology_imported = True
        defaults_applied.append(AppliedDefault(
            key="mp28 aerosol initial state",
            value="WIF monthly climatology (aer_init_opt=1, "
                  "wif_input_opt=1)",
            reason=(
                "use_aero_icbc=.true. with wif_input_opt=1 selects the "
                "ported QNWFA_QNIFA_SIGMA_MONTHLY.dat ingest "
                "(woof/ingest/wif_climatology.py), the same metgrid "
                "constants_name route real.exe reads.  The 215 MiB "
                "dataset is not redistributed: stage it with `woof "
                "fetch-tables --wif`, or point wif_climatology_path / "
                "WOOF_WIF_CLIMATOLOGY at a copy.  On a domain with "
                "external lateral boundaries a missing dataset is a named "
                "refusal at the run door -- before any fetch and before "
                "step 0 -- never a silent fall back to the synthetic "
                "profile; set mp28_aerosol_source='synthetic' in the "
                "emitted configuration to take that profile deliberately")))
    elif mp_physics == 28:
        # The never-silent last mile.  A user who imports an mp=28 namelist
        # gets a printed line saying exactly which aerosol initial state
        # their run will use and which WRF line refuses the same
        # configuration -- rather than discovering months later that their
        # ArWen and WRF runs were never comparable.
        from woof.config import (MP28_AEROSOL_SOURCE_DEFAULT,
                                  MP28_AEROSOL_SYNTHETIC_FALLBACK)
        from woof.ingest.wif_climatology import (portable_wif_statement,
                                                  resolve_wif_climatology)
        # The printed line now reports a RESOLUTION, not a deviation.  It is
        # still printed unconditionally, for the same reason it always was:
        # which aerosol initial state a run used is the single fact that
        # decides whether an ArWen mp=28 forecast and a WRF mp=28 forecast
        # are the same experiment.  What changed is which answer is normal.
        # The resolver is run HERE, at import time, so the answer is the
        # real one for this machine and this working directory rather than
        # a description of what will probably happen.
        try:
            _wif = resolve_wif_climatology()
        except Exception:      # an explicit override that does not exist
            _wif = None        # -- initialize_real raises on it by name
        # Stated by CONTENT and by rule, never by this process's absolute
        # paths: this report is bound into the WRF-input door's preparation
        # receipt, and a cwd or home path in it made a --restart launched
        # from another directory a "different run" (5 km radar DA,
        # 2026-10-23).  The run's own warning and aerosol receipt keep the
        # paths.
        if _wif is not None and _wif.resolved:
            defaults_applied.append(AppliedDefault(
                key="mp28 aerosol initial state",
                value="WRF monthly WIF aerosol climatology "
                      f"({portable_wif_statement(_wif)})",
                reason=MP28_AEROSOL_SOURCE_DEFAULT))
        else:
            reason = MP28_AEROSOL_SYNTHETIC_FALLBACK
            if _wif is not None and _wif.fallback_reason:
                reason = reason + " " + portable_wif_statement(_wif)
            # AND WHAT THE RUN DOOR WILL DO WITH IT -- but only when it
            # will.  This import writes a TOML and runs nothing, so it
            # does not refuse; if these namelists describe a domain with
            # external lateral boundaries the run door does, and the
            # reader is told here, where the file they would edit is
            # about to be written, which is the only door at which the
            # second way out can be taken.  ``specified`` is parsed from
            # &bdy_control further down this same translation, so the
            # sentence is APPENDED THERE, to the record remembered here;
            # an idealized namelist set gets the resolution and no
            # run-door clause, because that door will not refuse it.  The
            # sentence itself is the run door's own, imported rather than
            # restated.
            _mp28_fallback_row = len(defaults_applied)
            defaults_applied.append(AppliedDefault(
                key="mp28 aerosol initial state",
                value="SYNTHETIC FALLBACK -- thompson_init CCN/IN profile",
                reason=reason))

    bl_wrf, bl_mapped = _mapped("bl_pbl_physics", _BL_MAP, per_domain=True)
    bl_pbl_col = [entry[0] for entry in bl_mapped]
    bl_pbl_physics, wrf_name, gp_name = bl_mapped[0]
    if bl_mynn_mixscalars is None and any(value == 5 for value in bl_pbl_col):
        notices.append(MYNN_MIXSCALARS_DEFAULT_NOTICE)
    # WRF does not run the scalar_pblmix this namelist states under the
    # aerosol-aware Thompson scheme: share/module_check_a_mundo.F:2477-2495
    # (v4.6.1) resets it to 1 for mp_physics = 28 with use_aero_icbc or
    # use_rap_aero_icbc, at debug level 1, so wrf.exe prints nothing and
    # namelist.output still shows the stated value; :2497-2511 then turns
    # it off again under MYNN with bl_mynn_mixscalars = 1.  Importing the
    # stated 0 gave a run that never mixed QNC/QNI/QNWFA/QNIFA in the PBL
    # while wrf.exe on the same namelist did: on the stock-WRF door case
    # the lowest-level QNWFA drifted 3 % from WRF in 10 minutes and 14 %
    # in 90.  Resolved here as WRF resolves it, and said so.
    if (mp_physics == 28 and (_aero_icbc or analyzed_aerosol_imported)
            and not (bl_pbl_physics == 5 and bl_mynn_mixscalars == 1)
            and scalar_pblmix != 1):
        defaults_applied.append(AppliedDefault(
            key="scalar_pblmix", value=1,
            reason=(f"the namelist states {scalar_pblmix!r}, but WRF "
                    "resets scalar_pblmix to 1 for mp_physics = 28 with "
                    "aerosol IC/BCs (share/module_check_a_mundo.F:"
                    "2477-2495) and runs that, silently")))
        scalar_pblmix = 1
    elif (bl_pbl_physics == 5 and bl_mynn_mixscalars == 1
          and scalar_pblmix == 1):
        defaults_applied.append(AppliedDefault(
            key="scalar_pblmix", value=0,
            reason=("WRF turns scalar_pblmix off when MYNN mixes the "
                    "scalars itself (bl_mynn_mixscalars = 1; "
                    "share/module_check_a_mundo.F:2497-2511)")))
        scalar_pblmix = 0
    if bl_pbl_physics != bl_wrf[0]:
        substitutions.append(Substitution(
            key="bl_pbl_physics", wrf_value=bl_wrf[0], wrf_name=wrf_name,
            gpuwm_key="bl_pbl_physics", gpuwm_value=bl_pbl_physics,
            gpuwm_name=gp_name))
    lw_wrf, lw_mapped = _mapped("ra_lw_physics", _RA_LW_MAP, per_domain=True)
    sw_wrf, sw_mapped = _mapped("ra_sw_physics", _RA_SW_MAP, per_domain=True)
    ra_lw_physics, lw_wrf_name, lw_gp_name = lw_mapped[0]
    ra_sw_physics, sw_wrf_name, sw_gp_name = sw_mapped[0]
    radiation_pairs = [(lw[0], sw[0]) for lw, sw in zip(lw_mapped, sw_mapped)]
    # An unspecified mapping follows the named source generation. An
    # ordinary namelist retains the existing modern RRTMG default; an
    # explicit mapping still wins over a named source's selection.
    if rrtmg_variant is _AUTO_RRTMG_VARIANT:
        rrtmg_variant = source_defaults.get(
            "ra_rrtmg_variant", _default_rrtmg_variant_for_namelist(
                inp, input_path, source_text))
    legacy_rrtmg_col = [rrtmg_variant == RRTMG_VARIANT_LEGACY or
        (rrtmg_variant is None and 4 in pair) for pair in radiation_pairs]
    any_rrtmg = any(4 in pair for pair in radiation_pairs)
    any_modern_rrtmg = any(4 in pair and not legacy
        for pair, legacy in zip(radiation_pairs, legacy_rrtmg_col))
    any_dudhia = any(sw == 1 for _, sw in radiation_pairs)
    # Preserve the frozen aggregate representation for the already shipped
    # coupled RTE+RRTMGP configurations.  Every other WRF pair is emitted in
    # the native split schema, including RRTM LW + Dudhia SW (1/1).
    coupled_legacy = (ra_lw_physics == ra_sw_physics
                      and ra_lw_physics in (0, 4))
    ra_physics = ra_lw_physics if coupled_legacy else 0
    legacy_rrtmg = legacy_rrtmg_col[0]
    if use_mp_re == 0 and any_modern_rrtmg:
        raise _err(
            "physics", "use_mp_re", [use_mp_re],
            "use_mp_re=0 is implemented only by the exact legacy-RRTMG "
            "wrapper; a selected modern-RRTMG spectrum has no equivalent "
            "calculated-radius branch.")
    o3input = None
    ghg_input = None
    # STATED BY THE CALLER WHERE THERE IS A CALLER TO STATE IT.  The
    # derived answer below is what a bare `woof import-namelist` has
    # always produced and stays the default; a caller round-tripping its
    # own configuration holds the authoritative value and passes it,
    # because 'none' and the mapping token are two different runs of one
    # 4/4 selector pair (the RTE+RRTMGP arm reads the token to choose its
    # snow treatment) and nothing in the namelist distinguishes them.
    # The stated value applies only where the pair IS 4/4: on any other
    # pair the token has no consumer and woof.config refuses it, so the
    # derivation's 'none' stands.
    compatibility_col = [
        (rrtmg_compatibility if rrtmg_compatibility is not None
         else (WRF_RRTMG_LEGACY if legacy else WRF_RRTMG_TO_RTE_RRTMGP))
        if pair == (4, 4) else "none"
        for pair, legacy in zip(radiation_pairs, legacy_rrtmg_col)]
    wrf_rrtmg_compatibility = compatibility_col[0]
    adapter_44 = ("WRF legacy RRTMG" if legacy_rrtmg else lw_gp_name)
    if (ra_lw_physics, ra_sw_physics) == (4, 4):
        substitutions.append(Substitution(
            key="ra_lw_physics/ra_sw_physics", wrf_value=lw_wrf[0],
            wrf_name=lw_wrf_name, gpuwm_key="ra_physics",
            gpuwm_value=ra_physics, gpuwm_name=adapter_44))
    elif ra_lw_physics != lw_wrf[0] or (ra_lw_physics == 4 and not legacy_rrtmg):
        substitutions.append(Substitution(
            key="ra_lw_physics", wrf_value=lw_wrf[0],
            wrf_name=lw_wrf_name, gpuwm_key="ra_lw_physics",
            gpuwm_value=ra_lw_physics,
            gpuwm_name=("RTE+RRTMGP longwave" if ra_lw_physics == 4 else lw_gp_name)))
    if (ra_lw_physics, ra_sw_physics) != (4, 4) \
            and (ra_sw_physics != sw_wrf[0] or (ra_sw_physics == 4 and not legacy_rrtmg)):
        substitutions.append(Substitution(
            key="ra_sw_physics", wrf_value=sw_wrf[0],
            wrf_name=sw_wrf_name, gpuwm_key="ra_sw_physics",
            gpuwm_value=ra_sw_physics,
            gpuwm_name=("RTE+RRTMGP shortwave" if ra_sw_physics == 4 else sw_gp_name)))
    for n, (pair, is_legacy) in enumerate(zip(radiation_pairs, legacy_rrtmg_col)):
        if n and 4 in pair and (pair, is_legacy) != (
                radiation_pairs[0], legacy_rrtmg_col[0]):
            substitutions.append(Substitution(
                key=f"ra_lw_physics/ra_sw_physics[d{grid_ids[n]:02d}]",
                wrf_value=[lw_wrf[n], sw_wrf[n]], wrf_name="WRF radiation selection",
                gpuwm_key=f"domain[grid_id={grid_ids[n]}].ra_lw_physics/ra_sw_physics",
                gpuwm_value=list(pair),
                gpuwm_name="WRF legacy RRTMG" if is_legacy else "RTE+RRTMGP"))
    if any_dudhia:
        icloud = int(ph.scalar("icloud", 1))
        swrad_scat = float(ph.scalar("swrad_scat", 1.0))
        if icloud not in (0, 1, 2):
            raise _err("physics", "icloud", icloud, "choose 0, 1 or 2; icloud=3 needs unported cal_cldfra3 condensate adjustment.")
        if not math.isfinite(swrad_scat) or swrad_scat < 0.0:
            raise _err("physics", "swrad_scat", swrad_scat,
                       "must be finite and non-negative.")
    else:
        icloud = 1
        swrad_scat = 1.0
        raw_icloud = ph.take("icloud")
        adapter_label = ("legacy RRTMG"
                         if 4 in (ra_lw_physics, ra_sw_physics)
                         and legacy_rrtmg else "RTE+RRTMGP")
        if any_rrtmg and raw_icloud is not None and any(int(value) == 0
                                         for value in raw_icloud):
            raise _err("physics", "icloud", raw_icloud,
                "icloud=0 lacks wired RRTMGP clear-sky cloud optics and a "
                "recorded legacy clear-sky LW/SW oracle; choose the 1/1 "
                "pair for clear sky or icloud=1/2.")
        if raw_icloud is not None:
            icloud = int(_uniform("physics", "icloud", raw_icloud))
            if icloud not in (0, 1, 2):
                raise _err("physics", "icloud", raw_icloud, "choose 0, 1 or 2; icloud=3 needs unported cal_cldfra3 condensate adjustment.")
        if icloud == 1:
            fix("physics", "icloud", raw_icloud, 1,
                f"cloud-radiation coupling is fixed on in the {adapter_label} "
                "driver")
        drop("physics", "swrad_scat", ph.take("swrad_scat"),
             "Dudhia shortwave is not selected")

    if any_rrtmg:
        # RRTMG-family radiation options, ratified with fail-closed ranges.
        # Both 4/4 adapters implement McICA maximum-random overlap
        # (cldovrlp=2), constant decorrelation (idcor=0), analytic
        # year-dependent well-mixed gases (ghg_input=0), and no aerosol
        # (aer_opt=0).  Legacy RRTMG additionally implements the wrapper's
        # O3DATA branch (o3input=0); both variants retain CAM climatology
        # (o3input=2).  Any other request would be a silent WRF semantic
        # change, so an explicit other value refuses instead of importing.
        # ghg_input: both values are implemented (woof.core.camtr_gases
        # ports WRF's CAMtr reader).  An omitted key on the WRF V4 line is
        # the V4 Registry default 1 (Registry.EM_COMMON:2452) on the legacy
        # arm, the one held to wrf.exe; the RTE+RRTMGP arm keeps its own
        # dated gases unless the namelist asks for 1, so the shipped
        # RTE+RRTMGP configurations do not change.  Origin: the OU
        # Oman-UAE case (2026-10-28) ran fixed gases where wrf.exe 4.6.1
        # read CAMtr, and longwave differed in every column.
        adapter_label = ("RTE+RRTMGP" if any_modern_rrtmg else "legacy RRTMG")
        raw_o3input = ph.take("o3input")
        if raw_o3input is not None:
            value = int(_uniform("physics", "o3input", raw_o3input))
            admitted = (2,) if any_modern_rrtmg else (0, 2)
            if value not in admitted:
                raise _err(
                    "physics", "o3input", raw_o3input,
                    f"woof's {adapter_label} adapter implements "
                    f"o3input values {admitted}; o3input=0 belongs to the "
                    "legacy wrapper's O3DATA branch.")
            if value == 2:
                fix(
                    "physics", "o3input", raw_o3input, 2,
                    f"the established CAM-climatology value in the "
                    f"{adapter_label} adapter")
            else:
                o3input = value
        for key, supported, why in (
                ("cldovrlp", 2,
                 "McICA maximum-random overlap is the only implemented "
                 "subcolumn walk"),
                ("idcor", 0,
                 "constant 2500 m decorrelation is the only implemented "
                 "choice (inert at cldovrlp=2)")):
            raw = ph.take(key)
            if raw is None:
                continue
            if any(int(value) != supported for value in raw):
                raise _err(
                    "physics", key, raw,
                    f"woof's {adapter_label} adapter only implements "
                    f"{key} = {supported} ({why}).")
            fix("physics", key, raw, supported,
                f"the only implemented value in the {adapter_label} "
                f"adapter ({why})")

    if any_rrtmg:
        raw_ghg = ph.take("ghg_input")
        if raw_ghg is not None:
            ghg_input = int(_uniform("physics", "ghg_input",
                                     raw_ghg if isinstance(raw_ghg, list) else [raw_ghg]))
            if ghg_input not in (0, 1):
                raise _err("physics", "ghg_input", raw_ghg,
                           "WRF defines ghg_input 0 (fixed gases) and 1 "
                           "(CAMtr time-varying gases).")
        elif (wrf_version_source is not None and wrf_line_of(wrf_version) == "4"
              and not any_modern_rrtmg):
            # Only for a namelist NAMED as V4 (the WRF files' TITLE through
            # the WRF-file door, or --wrf-version): a bare import keeps the
            # engine's fixed gases.  The operational HRRR namelist is a WRF
            # 3.9-fork file that the bare default line ("4") does not
            # describe, and the HRRR route's native namelist leaves
            # ghg_input out by contract (stock-only ghg_input = 0,
            # woof.hrrr_route_inputs); with the V4 default on every import,
            # each HRRR-route emission was refused at its round trip
            # ("d01 ghg_input: config 1 vs namelist 0", 2.9.10 fix batch).
            ghg_input = 1
            defaults_applied.append(AppliedDefault(
                key="ghg_input", value=1,
                reason="the namelist omits ghg_input; the WRF v4 Registry "
                       "default is 1 (time-varying CO2/N2O/CH4/CFCs from "
                       "CAMtr_volume_mixing_ratio, SSP245 as WRF ships it)"))
        if ghg_input == 0:
            ghg_input = None  # the engine default: write nothing
    if not any_rrtmg and any(5 in pair for pair in radiation_pairs):
        drop('physics', 'ghg_input', ph.take('ghg_input'),
             'New Goddard uses its fixed module gas concentrations; ghg_input belongs to RRTMG')
    if not any_rrtmg:
        raw_o3input = ph.take("o3input")
        if raw_o3input is not None:
            o3input = int(_uniform("physics", "o3input", raw_o3input))
            if o3input not in (0, 2):
                raise _err("physics", "o3input", raw_o3input, "must be 0 or 2.")

    # aer_opt: 0 (no aerosol) on every radiation pair; 3 (the Thompson
    # water- and ice-friendly aerosol shortwave optics the HRRR fork's
    # radiation driver builds through gt_aod and calc_aerosol_rrtmg_sw) on
    # the legacy RRTMG pair with mp_physics = 28, which is the gate WRF
    # itself applies (F_QNWFA and RRTMG shortwave,
    # module_radiation_driver.F:951-953). User aerosol (2) uses the
    # WRF 4.6.1 spectral producer; ECMWF aerosol (1) is refused by name; woof.config.validate_radiation_driver_
    # options restates the same rules on the loaded RunConfig.
    aer_opt = None
    raw_aer_opt = ph.take("aer_opt")
    if raw_aer_opt is not None:
        aer_opt = _uniform("physics", "aer_opt", raw_aer_opt)
        if type(aer_opt) is not int:
            raise _err("physics", "aer_opt", raw_aer_opt,
                       "must be integer 0, 1, 2 or 3 (Registry.EM_COMMON:2389).")
        if aer_opt not in (0, 1, 2, 3):
            raise _err("physics", "aer_opt", raw_aer_opt, "must be 0, 1, 2 or 3 (Registry.EM_COMMON:2389).")
        if aer_opt == 1 and not legacy_rrtmg:
            raise _err("physics", "aer_opt", raw_aer_opt,
                       "aer_opt=1 needs --rrtmg-variant rrtmg_legacy: RRTMGP has no checked six-type aerosol mapping and would discard or mis-map these optics.")
        if aer_opt == 1 and max_dom != 1:
            raise _err("physics", "aer_opt", raw_aer_opt,
                       "aer_opt=1 on a resident nest needs domain-1 aerod parent forcing; child-grid climatology would change WRF's root-only aerosol routing.")
        if aer_opt == 2 and not legacy_rrtmg:
            raise _err("physics", "aer_opt", raw_aer_opt,
                       "aer_opt=2 requires legacy RRTMG shortwave; another solver would discard its optics.")
        if aer_opt == 3 and not legacy_rrtmg:
            raise _err(
                "physics", "aer_opt", raw_aer_opt,
                "aer_opt = 3 is transcribed for the legacy RRTMG shortwave "
                "composition only (ra_sw_physics = 4 mapped to "
                f"ra_rrtmg_variant = '{RRTMG_VARIANT_LEGACY}'); this "
                "namelist's shortwave would silently discard the aerosol "
                "optics.  Import with --rrtmg-variant "
                f"{RRTMG_VARIANT_LEGACY} (rrtmg_variant="
                f"'{RRTMG_VARIANT_LEGACY}') to run the exact RRTMG port "
                "with the aerosol, or set aer_opt = 0.")
        if aer_opt == 3 and mp_physics != 28:
            raise _err(
                "physics", "aer_opt", raw_aer_opt,
                "aer_opt = 3 needs mp_physics = 28 (WRF gates the branch "
                "on the water-friendly aerosol package, F_QNWFA); with "
                f"mp_physics = {mp_physics} WRF runs zero aerosol under a "
                "namelist claiming aerosol-aware radiation.")
        if aer_opt == 0:
            # WRF's and the RunConfig's default: the read is receipted and
            # the emitted TOML stays byte-identical to an import without
            # the key, as it was while aer_opt was pinned at 0.
            aer_opt = None

    aerosol_settings = {}
    from woof.core.aer_opt2 import OPTION_KEYS, VALUE_KEYS, DEFAULT_OPTIONS, DEFAULT_VALUES, validate_options
    for key, default in zip(("aer_type",)+OPTION_KEYS+VALUE_KEYS,
                            (1,)+DEFAULT_OPTIONS+DEFAULT_VALUES):
        raw = ph.take(key)
        val = default if raw is None else _uniform("physics", key, raw)
        if aer_opt == 2:
            aerosol_settings[key] = val
        elif raw is not None and val != default:
            raise _err("physics",key,raw,"this aerosol setting requires aer_opt=2; its value would otherwise be discarded.")
    aux_name = tc.take("auxinput15_inname")
    aux_minutes = tc.take("auxinput15_interval_m")
    aux_interval = tc.take("auxinput15_interval")
    aux_io = tc.take("io_form_auxinput15")
    if aer_opt == 2:
        validate_options(aer_type=aerosol_settings["aer_type"],
                         options=tuple(aerosol_settings[k] for k in OPTION_KEYS),
                         values=tuple(aerosol_settings[k] for k in VALUE_KEYS))
        if any(aerosol_settings[k]==2 for k in OPTION_KEYS):
            if aux_name is None:
                raise _err("time_control","auxinput15_inname",None,"gridded aerosol options require the auxiliary aerosol input; absent fields cannot become fixed values.")
            if aux_io is not None and _uniform("time_control","io_form_auxinput15",aux_io) != 2:
                raise _err("time_control","io_form_auxinput15",aux_io,"aerosol auxiliary input must be NetCDF (2).")
            name = _uniform("time_control","auxinput15_inname",aux_name)
            aerosol_settings["aer_auxinput15"] = str((Path(input_path).resolve().parent / name).resolve())
            interval = aux_minutes if aux_minutes is not None else aux_interval
            aerosol_settings["aer_auxinput15_interval"] = 0 if interval is None else 60*int(_uniform("time_control","auxinput15_interval",interval))

    sfclay_values = ph.col("sf_sfclay_physics", max_dom)
    if sfclay_values is None:
        raise _err(
            "physics", "sf_sfclay_physics", None,
            "must-set namelist key (WRF Registry default -1 is rejected "
            "by the surface-layer driver); woof will not silently "
            "disable the scheme.")
    sfclay = int(_uniform(
        "physics", "sf_sfclay_physics", sfclay_values))
    if sfclay not in _SFCLAY_ALLOWED:
        raise _err("physics", "sf_sfclay_physics", sfclay,
                   f"no woof mapping (implemented: "
                   f"{sorted(_SFCLAY_ALLOWED)})."
                   + _unported_tail("sf_sfclay_physics", sfclay)
                   + _port_receipt(sf_sfclay_physics=sfclay))
    sfsfc = int(_uniform("physics", "sf_surface_physics",
                         ph.col("sf_surface_physics", max_dom, 0)))
    # The recipe's prescribed-monthly request is for the land schemes that
    # read the monthly fields (physics_source_defaults.land_scoped_defaults).
    from woof.physics_source_defaults import land_scoped_defaults
    source_defaults = land_scoped_defaults(source_defaults, sfsfc, mp_physics)
    if sfsfc not in _SFSFC_ALLOWED:
        raise _err("physics", "sf_surface_physics", sfsfc,
                   f"no woof mapping (implemented: "
                   f"{sorted(_SFSFC_ALLOWED)})."
                   + _unported_tail("sf_surface_physics", sfsfc)
                   + _port_receipt(sf_surface_physics=sfsfc))
    if (seaice_albedo_default is not None
            and seaice_albedo_default != 0.65 and sfsfc not in (2, 3)):
        raise _err(
            "physics", "seaice_albedo_default",
            [seaice_albedo_default],
            "a nondefault value is implemented only by Noah "
            "(sf_surface_physics=2, seaice_noah) and RUC LSM "
            "(sf_surface_physics=3).")
    if rdmaxalb is False and sfsfc != 2:
        raise _err(
            "physics", "rdmaxalb", [rdmaxalb],
            "rdmaxalb=false is implemented by Noah LSMINIT and requires "
            "sf_surface_physics=2.")
    if isfflx == 0 and sfclay == 0:
        raise _err(
            "physics", "isfflx", [isfflx],
            "isfflx=0 has no consumer when sf_sfclay_physics=0; woof "
            "implements the gate in its MM5 and MYNN surface layers.")
    from types import SimpleNamespace
    from woof.config import (WRF_NO_LAND_SURFACE_SOIL_LAYERS, soil_layer_count,
                              validated_soil_layer_count)

    resolved_soil_layers = (WRF_NO_LAND_SURFACE_SOIL_LAYERS if sfsfc == 0
                            else validated_soil_layer_count(sfsfc))
    requested_soil_layers = ph.take("num_soil_layers")
    soil_reason = f"resolved soil geometry for sf_surface_physics={sfsfc}"
    if requested_soil_layers is not None:
        requested = int(_uniform("physics", "num_soil_layers", requested_soil_layers))
        try:
            resolved_soil_layers = soil_layer_count(SimpleNamespace(
                sf_surface_physics=sfsfc, num_soil_layers=requested))
        except ValueError as error:
            if sfsfc in (2, 4) and requested != resolved_soil_layers:
                # A WRF namelist means what wrf.exe makes of it: Noah and
                # Noah-MP have one soil geometry, and WRF v4.6.1 writes it
                # over any request (share/module_check_a_mundo.F:3552-3555).
                # Named breakage: NCAR's v4.4 CONUS benchmarks write
                # num_soil_layers = 1 beside Noah and were refused; the
                # files' own soil geometry is still checked against the
                # resolved count by every WRF-input door.  RUC keeps its
                # refusal: there WRF honours 6 or 9 and the request matters.
                soil_reason = (
                    f"wrf.exe sets num_soil_layers = {resolved_soil_layers} "
                    f"for sf_surface_physics={sfsfc} whatever the namelist "
                    "asks (share/module_check_a_mundo.F:3552-3555); the "
                    f"request {requested} is replaced, as WRF replaces it")
            else:
                raise _err("physics", "num_soil_layers", requested_soil_layers,
                           str(error)) from error
    fix(
        "physics", "num_soil_layers", requested_soil_layers,
        resolved_soil_layers, soil_reason)
    cu_values = ph.col("cu_physics", max_dom)
    if cu_values is None:
        raise _err(
            "physics", "cu_physics", None,
            "must-set namelist key (WRF Registry default -1 is rejected "
            "by the cumulus driver); woof will not silently disable the "
            "scheme.")
    cu = [int(v) for v in cu_values]
    for value in cu:
        if value not in _CU_ALLOWED:
            raise _err("physics", "cu_physics", value,
                       f"no woof mapping (implemented: "
                       f"{sorted(_CU_ALLOWED)})."
                       + _unported_tail("cu_physics", value))
    cudt = [float(v) for v in ph.col("cudt", max_dom, 0)]
    # The two Grell-family keys, WRF v4.6.1 Registry defaults (both 0,
    # single-instance).  Read whenever present; consumed only where
    # cu_physics = 3, exactly as WRF's cumulus driver reads them only for
    # the Grell schemes.
    clos_choice = int(_uniform(
        "physics", "clos_choice", ph.col("clos_choice", max_dom, 0)))
    ishallow = int(_uniform(
        "physics", "ishallow", ph.col("ishallow", max_dom, 0)))
    # 0..16 are admitted as written: 0 is the ensemble mean, 1..16 one
    # closure member alone, and the load of the emitted TOML says the
    # single-member arms are implemented but not verified against WRF.
    # Only a value with no meaning in WRF's closure code is refused, with
    # the same sentence the run door prints.
    from woof.config import gf_clos_choice_refusal
    clos_refusal = gf_clos_choice_refusal(clos_choice)
    if clos_refusal is not None:
        raise _err("physics", "clos_choice", [clos_choice], clos_refusal)
    if ishallow not in (0, 1):
        raise _err("physics", "ishallow", [ishallow],
                   "must be 0 or 1 (CUP_gf_sh off/on).")
    if (clos_choice or ishallow) and all(value not in (3, 5) for value in cu):
        drop("physics", "clos_choice/ishallow",
             [clos_choice, ishallow],
             "Grell-family keys with no Grell scheme selected "
             "(cu_physics has no 3 or 5); WRF's cumulus driver would not "
             "read them either")
        clos_choice = 0
        ishallow = 0
    # New SAS's two keys (WRF Registry.EM_COMMON:2493 and :2651):
    # nsas_dx_factor is scope 1, sas_pgcon a max_domains column.  Read
    # whenever present, emitted only on the cu_physics = 96 domains, and
    # dropped with a reason elsewhere, exactly as WRF's cumulus driver
    # reads them only for the SAS schemes.
    from woof.config import (NSAS_DX_FACTOR_SCHEMES, SAS_FAMILY_DEFAULTS,
                              SAS_PGCON_SCHEMES)
    nsas_dx_factor = ph.scalar("nsas_dx_factor",
                               SAS_FAMILY_DEFAULTS["nsas_dx_factor"])
    if (isinstance(nsas_dx_factor, bool) or not isinstance(nsas_dx_factor, int)
            or nsas_dx_factor not in (0, 1)):
        raise _err("physics", "nsas_dx_factor", [nsas_dx_factor],
                   "must be 0 or 1 (module_cu_nsas.F:567: 1 scales the "
                   "trigger and closure velocities by 250/dx).")
    sas_pgcon = [float(v) for v in ph.registry_col(
        "sas_pgcon", max_dom, SAS_FAMILY_DEFAULTS["sas_pgcon"])]
    if not all(math.isfinite(v) for v in sas_pgcon):
        raise _err("physics", "sas_pgcon", sas_pgcon, "must be finite.")
    if (nsas_dx_factor != SAS_FAMILY_DEFAULTS["nsas_dx_factor"]
            and not any(value in NSAS_DX_FACTOR_SCHEMES for value in cu)):
        drop("physics", "nsas_dx_factor", [nsas_dx_factor],
             "read only by cu_physics = 96 (New SAS), which no domain "
             "selects; WRF's cumulus driver would not read it either")
        nsas_dx_factor = SAS_FAMILY_DEFAULTS["nsas_dx_factor"]
    for n, value in enumerate(sas_pgcon):
        if (value != SAS_FAMILY_DEFAULTS["sas_pgcon"]
                and cu[n] not in SAS_PGCON_SCHEMES):
            drop("physics", f"sas_pgcon[{n + 1}]", [value],
                 "read only by the SAS cumulus family (cu_physics = 96 "
                 "here), which this domain does not select")
            sas_pgcon[n] = SAS_FAMILY_DEFAULTS["sas_pgcon"]
    if any(value == 3 for value in cu):
        # The scheme-generation declaration.  A v4.8.0 namelist selecting
        # GFL is byte-indistinguishable from this one at the GF option
        # level (GFL added zero namelist options and changed zero
        # defaults), so the wrong-scheme import cannot be refused -- it is
        # declared, once, on every GF import.
        notices.append(GF_SCHEME_GENERATION_NOTICE)
    # cugd_avedx -- "number of grid boxes over which subsidence is
    # spread" (Registry.EM_COMMON:2543, integer scalar, default 1) -- is
    # the ONE Grell-family key whose non-default value reveals a namelist
    # written for a spreading-generation Grell scheme.  In v4.6.1 the
    # cumulus driver hands it only to G3 (module_cumulus_driver.F:1250)
    # and applies the spreading only under `IF (cu_physics .eq. 5)`
    # (conv_grell_spread3d, :1628); GFDRV never receives it, and woof's
    # GF port matches GFDRV.  v4.8.0's Grell-Freitas-Li reuses
    # cu_physics = 3 with subsidence spreading as a headline mechanism
    # (GF_SCHEME_GENERATION_NOTICE above).  Until now the key had no
    # import path at all and died in _Section.finish's generic
    # "unmapped key(s)" sentence -- true, but a sentence about the
    # importer's map rather than about the scheme generation the user
    # actually selected.
    from woof.config import G3_RUN_DEFAULTS, validate_g3_dimensions
    g3_controls = {}
    g3_named = {key for key in G3_RUN_DEFAULTS if key in ph.entries}
    for control_key, default in G3_RUN_DEFAULTS.items():
        if control_key == "cugd_avedx": continue
        if control_key == "cu_diag":
            # A max_domains Registry array: entries the namelist leaves out
            # keep WRF's default 0, they are not filled from the last value.
            g3_controls[control_key] = ph.registry_col(control_key, max_dom, default)[:max_dom]
            continue
        values = ph.col(control_key, max_dom, default)
        g3_controls[control_key] = _uniform("physics", control_key, values)
    cugd_avedx = 1
    cugd_avedx_values = ph.take("cugd_avedx")
    if cugd_avedx_values is not None:
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in cugd_avedx_values):
            raise _err("physics", "cugd_avedx", cugd_avedx_values,
                       "must be an integer (WRF Registry scalar, "
                       "Registry.EM_COMMON:2543).")
        cugd_avedx = int(_uniform(
            "physics", "cugd_avedx", cugd_avedx_values))
        if any(value == 5 for value in cu):
            # Grell 3D reads it (G3DRV's high-resolution arm and
            # conv_grell_spread3d).  WRF's key is one scalar for every
            # domain; Grell-Freitas domains beside a G3 domain ignore it
            # exactly as GFDRV does, and are emitted at the default.
            pass
        elif any(value == 3 for value in cu):
            if cugd_avedx != 1:
                raise _err(
                    "physics", "cugd_avedx", [cugd_avedx],
                    "requests subsidence spreading over a multi-box "
                    "neighbourhood, which no v4.6.1-generation "
                    "Grell-Freitas implements: WRF v4.6.1's cumulus "
                    "driver hands cugd_avedx only to G3 and spreads only "
                    "when cu_physics = 5 (module_cumulus_driver.F:1250, "
                    ":1628), and woof's GF port matches GFDRV, which "
                    "never receives it.  A non-default value beside "
                    "cu_physics = 3 is the one namelist spelling that "
                    "reveals a WRF v4.8.0-era Grell-Freitas-Li (or G3) "
                    "configuration, and importing it as v4.6.1 GF would "
                    "silently run convection without the requested "
                    "mechanism.  Remove the key (or set 1, the Registry "
                    "default) to accept the v4.6.1 Grell-Freitas scheme "
                    "generation this importer maps.")
            fix("physics", "cugd_avedx", cugd_avedx_values, 1,
                "Registry default, inert under v4.6.1 Grell-Freitas: the "
                "cumulus driver hands cugd_avedx only to G3's "
                "conv_grell_spread3d (cu_physics = 5), never to GFDRV")
        else:
            drop("physics", "cugd_avedx", cugd_avedx_values,
                 "Grell-family subsidence-spreading key with no Grell "
                 "scheme selected (cu_physics has no 3 or 5); consumed "
                 "only by G3 (cu_physics = 5), and WRF's v4.6.1 cumulus "
                 "driver would not read it either")
            cugd_avedx = 1
    radt = [float(v) for v in ph.col("radt", max_dom, 0)]
    bldt = float(_uniform("physics", "bldt", ph.col("bldt", max_dom, 0)))
    if not use_adaptive_time_step:
        domain_steps = _wrf_domain_steps(
            Fraction(time_step) + Fraction(fract_num, fract_den),
            parent_id, tratio)
        radt = [_wrf_cadence_minutes("radt", n, value, domain_steps[n], fix)
                for n, value in enumerate(radt)]
        bldt = _wrf_cadence_minutes("bldt", 0, bldt, domain_steps[0], fix)
        cudt = [(_wrf_cadence_minutes("cudt", n, value, domain_steps[n], fix)
                 if cu[n] == 1 else value) for n, value in enumerate(cudt)]
    restart_interval_s = _wrf_restart_interval_s(
        restart_interval_s, run_seconds,
        Fraction(time_step) + Fraction(fract_num, fract_den),
        use_adaptive_time_step, fix)
    for key, reason in (
            ("ifsnow", "Noah snow physics is always active"),
            ("surface_input_source", "surface fields come from the "
                                     "ingest catalog"),

    ):
        drop("physics", key, ph.take(key), reason)
    # ---- land-use identity keys ----------------------------------------
    # Both are ordinary keys in a WRF-Runner/WPS-generated namelist and both
    # used to reach ph.finish() unmapped, which refused the whole import
    # with "unmapped key(s) ['fractional_seaice', 'num_land_cat']" -- a
    # sentence about the importer's map, not about the run.
    #
    # num_land_cat is the geography's land-use category count.  woof never
    # reads it from the namelist: the count comes from LANDUSE.TBL for the
    # MMINLU the static build stamped (woof/core/landuse.py
    # load_landuse_table + the lucats bound check), and every woof
    # geography is the MODIS table: its 21 categories (20 for the 15
    # arc-second set geog_data_res = 'modis_15s' selects, which has no lake
    # class, as geogrid stamps), or 61 where a
    # [static.highres] land cover keeps its urban classes under the urban
    # canopy (woof.static.highres.landcover_legend).  So it is validated
    # against the count the static build will stamp and recorded -- a
    # namelist declaring USGS's 24 describes static data woof does not
    # build, and refusing is the only accurate answer.
    if landuse_identity is None:
        # The count the static build will stamp, per domain: the GEOG
        # tree's MODIS set (21, or 20 for modis_15s), or for a domain whose land cover the
        # [static.highres] block builds, that collection's legend under the
        # run's urban canopy (woof.static.highres.landcover_legend): 61
        # with the canopy on, where the LCZ / NLCD intensity classes keep
        # categories 51-61, else 21 with them collapsed to MODIS urban.
        # The urban keys are peeked, not consumed: the loop below reads
        # them, and WRF resets every domain to the innermost value.
        num_land_cat = ph.take("num_land_cat")
        built_categories, legend_note = _built_land_categories(
            static_landcover, grid_ids=grid_ids,
            urban=_innermost_int(ph, "sf_urban_physics", 0, max_dom),
            use_wudapt_lcz=_innermost_int(ph, "use_wudapt_lcz", 0, max_dom),
            geog_data_res=geog_data_res)
        counts = sorted(set(built_categories.values()))
        if num_land_cat is not None:
            if any(isinstance(value, bool) or not isinstance(value, int)
                   or value not in counts or len(counts) > 1
                   for value in num_land_cat):
                built = " and ".join(
                    f"{count} categories on "
                    + ", ".join(f"d{gid:02d}" for gid, c
                                in built_categories.items() if c == count)
                    for count in counts)
                # Only the urban legend's own count earns the pointer to
                # geog_data_res: it is the value a land-cover token makes
                # right, and the pointer is also what marks this refusal as
                # a consequence when a refused geog_data_res stands in as
                # 'default' (_REFUSED_SELECTOR_STAND_INS).  Any other count
                # is wrong whatever geog_data_res says, and is reported.
                from woof.core.landuse import load_landuse_table
                urban_count = int(load_landuse_table().lucats)
                hint = ""
                if static_landcover is None and any(
                        not isinstance(value, bool) and value == urban_count
                        for value in num_land_cat):
                    tokens = sorted(token for token, how
                                    in geog_data_res_tokens().items()
                                    if how != "geog")
                    hint = (f"  A geog_data_res land-cover token ({tokens}) "
                            "with the urban canopy on and use_wudapt_lcz = "
                            f"1 builds the {urban_count}-category Local "
                            "Climate Zone legend.")
                raise _err(
                    "physics", "num_land_cat", num_land_cat,
                    f"this pair builds {built} of the "
                    f"MODIFIED_IGBP_MODIS_NOAH table ({legend_note}); the "
                    "static builder stamps that count into the prepared "
                    "geography, so this value describes geography the run "
                    "will not initialize from."
                    + ("  WRF reads one num_land_cat for the whole run, so "
                       "no single value describes these domains."
                       if len(counts) > 1 else
                       f"  Set num_land_cat = {counts[0]} (or leave it "
                       "out).")
                    + ("  WRF itself stops on this mismatch: WPS 4.6 "
                       "geogrid writes NUM_LAND_CAT = 20 for the 15 "
                       "arc-second MODIS land use (modis_15s has no lake "
                       "class), and real.exe refuses a namelist "
                       "num_land_cat that differs (share/input_wrf.F)."
                       if 20 in counts else "")
                    + hint)
        fix("physics", "num_land_cat", num_land_cat,
            counts[0] if len(counts) == 1 else counts,
            "the land-use category count is read from LANDUSE.TBL for the "
            "static build's MMINLU (MODIFIED_IGBP_MODIS_NOAH), never from the "
            "namelist" + ("" if static_landcover is None
                          else f" ({legend_note})"))
    else:
        dataset = landuse_identity.get("MMINLU")
        category_count = landuse_identity.get("NUM_LAND_CAT")
        if (not isinstance(dataset, str) or not dataset
                or isinstance(category_count, bool)
                or not isinstance(category_count, Real)
                or not math.isfinite(category_count)
                or float(category_count) != int(category_count)
                or int(category_count) <= 0):
            raise ValueError("WRF land-use identity requires MMINLU and positive integer NUM_LAND_CAT")
        num_land_cat = ph.take("num_land_cat")
        if num_land_cat is not None and any(
                isinstance(value, bool) or not isinstance(value, int)
                or value != int(category_count) for value in num_land_cat):
            raise _err("physics", "num_land_cat", num_land_cat,
                       f"input file MMINLU={dataset} carries NUM_LAND_CAT={category_count}")
        fix("physics", "num_land_cat", num_land_cat, int(category_count),
            f"the WRF input file supplies land-use identity {dataset}")
    # fractional_seaice selects WHICH of WRF's two sea-ice thresholds the
    # surface runs (module_surface_driver.F:1365-1368: 0.5, or 0.02 under
    # 1).  RunConfig.fractional_seaice carries it for the RUC seam, the only
    # land surface whose seam has the open-water second surface-layer call
    # and the post-LSM blend; under any other LSM the forecast runs WRF's
    # 0.5 branch, so the value is recorded as fixed at 0 rather than
    # emitted into a configuration woof.config would refuse.
    fractional_seaice_values = ph.take("fractional_seaice")
    fractional_seaice = None
    if fractional_seaice_values is not None:
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value not in (0, 1) for value in fractional_seaice_values):
            raise _err(
                "physics", "fractional_seaice", fractional_seaice_values,
                "must be 0 (WRF's XICE >= 0.5 threshold, the Registry "
                "default) or 1 (the 0.02 fractional threshold).")
        if sfsfc == 3:
            fractional_seaice = (int(fractional_seaice_values[0])
                                if fractional_seaice_values[0] != 0 else None)
        elif fractional_seaice_values[0] == 0:
            fix("physics", "fractional_seaice", fractional_seaice_values, 0,
                "WRF's XICE >= 0.5 threshold, the branch every woof land "
                "surface runs")
        else:
            # A DECLARED DIVERGENCE, not a pin.  Until 2026-10-12 this was
            # booked in the "fixed by ArWen (validated against the only
            # implemented value)" bucket, which said the request was met;
            # it is not met under this LSM, so the terminal now prints it
            # with the cells it changes.  (A WRF input carrying the flag
            # still reads it for land-use initialization,
            # woof/wrfinput_forecast.py prepare_wrf_run.)
            substitutions.append(Substitution(
                key="fractional_seaice", wrf_value=1,
                wrf_name="fractional sea ice (ice/water flux blend, 0.02 "
                         "threshold)",
                gpuwm_key="fractional_seaice", gpuwm_value=0,
                gpuwm_name="whole-cell sea ice at XICE >= 0.5",
                reason="the fractional blend is implemented by the RUC seam "
                       "only (sf_surface_physics = 3: the open-water second "
                       "surface-layer call and the post-LSM flux blend); "
                       f"sf_surface_physics = {sfsfc} treats a cell as all "
                       "ice at XICE >= 0.5 and all water below it, so the "
                       "surface fluxes differ wherever the analysis has a "
                       "sea-ice fraction strictly between 0.02 and 1 "
                       "(marginal ice zones, freezing lakes) and nowhere "
                       "else"))
            drop("physics", "fractional_seaice", fractional_seaice_values,
                 "declared divergence above")
    # seaice_threshold is real.exe's rule (share/module_soil_pre.F
    # adjust_for_seaice_post, :245-246 and :341-342 of the HRRR v4.1.21
    # fork): open water whose input skin is colder than it becomes sea ice.
    # RunConfig.seaice_threshold carries it into the Noah/RUC/Noah-MP soil
    # ingest (woof/ingest/soil.py preprocess_noah_soil).  The Registry
    # declares one REAL (default 100 K, which converts nothing).
    seaice_threshold_values = ph.take("seaice_threshold")
    seaice_threshold = None
    if seaice_threshold_values is not None:
        if any(isinstance(value, bool)
               or not isinstance(value, (int, float))
               for value in seaice_threshold_values):
            raise _err("physics", "seaice_threshold", seaice_threshold_values,
                       "WRF declares one REAL skin temperature in K.")
        if len({float(value) for value in seaice_threshold_values}) != 1:
            raise _err("physics", "seaice_threshold", seaice_threshold_values,
                       "differs by domain, but WRF's Registry declares one "
                       "value for the whole run; write one value")
        threshold_value = float(seaice_threshold_values[0])
        if threshold_value == 100.0:
            fix("physics", "seaice_threshold", seaice_threshold_values, 100.0,
                "WRF's Registry default, which converts no water cell to ice")
        elif threshold_value <= 170.0 or sfsfc in (2, 3, 4):
            seaice_threshold = threshold_value
        else:
            substitutions.append(Substitution(
                key="seaice_threshold", wrf_value=threshold_value,
                wrf_name=f"open water below {threshold_value:g} K treated "
                         "as ice",
                gpuwm_key="seaice_threshold", gpuwm_value=None,
                gpuwm_name="sea ice from the analysed ice fraction",
                reason="the threshold rule runs in the Noah, RUC and Noah-MP "
                       "soil ingest; sf_surface_physics = "
                       f"{sfsfc} has none, so water colder than the "
                       "threshold stays water"))
            drop("physics", "seaice_threshold", seaice_threshold_values,
                 "declared divergence above")
    urban_settings = {}
    for key, default in (("sf_urban_physics", 0), ("use_wudapt_lcz", 0),
                         ("num_urban_hi", 15)):
        raw = ph.take(key)
        values = ([default] * max_dom if raw is None else
                  list(raw[:max_dom]) + [raw[-1]] * max(0, max_dom - len(raw)))
        if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise _err("physics", key, raw, "WRF declares an integer namelist key.")
        urban_settings[key] = values[-1]
        fix("physics", key, raw, values[-1],
            "WRF resets every domain to the innermost domain's value "
            "(share/module_check_a_mundo.F:1063-1077)")
    for key, default in (("slucm_distributed_drag", False),
                         ("distributed_ahe_opt", 0)):
        raw = ph.take(key)
        if raw is not None and any(value != default for value in raw):
            raise _err("physics", key, raw,
                       "the arm is not transcribed, the setting would be ignored")
        fix("physics", key, raw, default, "the unported arm is disabled")
    ruc_mosaic_settings = {}
    for key in ("sf_surface_mosaic", "mosaic_lu", "mosaic_soil"):
        values = ph.take(key)
        if key == "sf_surface_mosaic":
            sf_surface_mosaic = 0 if values is None else values[0]
            mosaic_values = values
            if sf_surface_mosaic == 0:
                fix("physics", key, values, 0,
                    ("WRF run-wide integer: reads the first value "
                     "(Registry.EM_COMMON:2536); Noah mosaic off"
                     if values is not None and len(values) > 1 else
                     "Noah mosaic off (WRF default); sf_surface_mosaic = 1 "
                     "runs Noah land-use tiles"))
            continue
        value = 0 if values is None else values[0]
        if type(value) is not int or value not in (0, 1):
            raise _err("physics", key, values,
                       "WRF declares a run-wide integer switch, 0 or 1.")
        if value and sfsfc != 3:
            raise _err(
                "physics", key, values,
                "RUC mosaic requires sf_surface_physics=3; other land "
                "models do not consume these weighted parameters.")
        ruc_mosaic_settings[key] = value
        fix("physics", key, values, value,
            "WRF run-wide integer: reads the first value; RUC weighted "
            "land and soil parameters, off by default")
    mosaic_cat_values = ph.take("mosaic_cat")
    mosaic_cat = 3 if mosaic_cat_values is None else mosaic_cat_values[0]
    from types import SimpleNamespace
    from woof.config import validate_noah_mosaic_config
    try:
        validate_noah_mosaic_config(SimpleNamespace(
            sf_surface_mosaic=sf_surface_mosaic, mosaic_cat=mosaic_cat,
            sf_surface_physics=sfsfc))
    except ValueError as exc:
        raise _err("physics", "sf_surface_mosaic", mosaic_values, str(exc)) from None
    if sf_surface_mosaic == 1:
        fix("physics", "sf_surface_mosaic", mosaic_values, sf_surface_mosaic,
            "WRF run-wide integer: reads the first value (Registry.EM_COMMON:2536)")
    if mosaic_cat_values is not None or sf_surface_mosaic == 1:
        fix("physics", "mosaic_cat", (mosaic_cat_values if mosaic_cat_values is not None
                                     else [mosaic_cat]), mosaic_cat,
            "WRF run-wide integer: reads the first value (Registry.EM_COMMON:2537); unread when mosaic is off")
    # woof's mosaic_urban_canopy (woof/config.py MOSAIC_URBAN_CANOPY_RULES):
    # no WRF spelling, read as a per-domain column of rule names; an
    # omitted tail is "dominant", WRF's own rule, so a namelist without the
    # key imports byte-identically.
    from woof.config import (MOSAIC_URBAN_CANOPY_DEFAULT,
                              MOSAIC_URBAN_CANOPY_RULES)
    canopy_values = ph.take("mosaic_urban_canopy")
    canopy_column = None
    if canopy_values is not None:
        if any(not isinstance(value, str)
               or value not in MOSAIC_URBAN_CANOPY_RULES
               for value in canopy_values):
            raise _err(
                "physics", "mosaic_urban_canopy", canopy_values,
                f"must name a rule on every domain, one of "
                f"{sorted(MOSAIC_URBAN_CANOPY_RULES)} (a woof key; WRF has "
                f"no such switch and runs 'dominant').")
        column = list(canopy_values[:max_dom])
        canopy_column = column + [MOSAIC_URBAN_CANOPY_DEFAULT] * (
            max_dom - len(column))
        for value in sorted(set(canopy_column)):
            try:
                validate_noah_mosaic_config(SimpleNamespace(
                    sf_surface_mosaic=sf_surface_mosaic, mosaic_cat=mosaic_cat,
                    sf_surface_physics=sfsfc, mosaic_urban_canopy=value,
                    sf_urban_physics=urban_settings["sf_urban_physics"]))
            except ValueError as exc:
                raise _err("physics", "mosaic_urban_canopy", canopy_values,
                           str(exc)) from None
        fix("physics", "mosaic_urban_canopy", canopy_values, canopy_column[0],
            "woof per-domain key: where Noah mosaic runs the urban canopy "
            "('dominant' is WRF v4.7.1's rule)")
    # A public WRF namelist has no release-version spelling. The generic
    # corrected default applies; the metadata-owned extension reproduces
    # the older water balance only when explicitly written.
    from woof.config import (URBAN_WATER_BALANCE_DEFAULT,
                              URBAN_WATER_BALANCE_GENERATIONS)
    water_values = ph.take("urban_water_balance")
    water_generation = URBAN_WATER_BALANCE_DEFAULT
    if water_values is not None:
        if (not water_values or any(value not in URBAN_WATER_BALANCE_GENERATIONS
                                   for value in water_values)
                or len(set(water_values)) != 1):
            raise _err("physics", "urban_water_balance", water_values,
                       f"must select one run-wide generation from "
                       f"{tuple(URBAN_WATER_BALANCE_GENERATIONS)}; the weak-rain "
                       "evaporation rule differs between generations")
        water_generation = water_values[0]
        fix("physics", "urban_water_balance", water_values, water_generation,
            "woof water-balance generation; both retain rho_air/1000")
    # ---- sub-grid terrain drag (lane/282-terrain-drag) ------------------
    # WRF's &physics topo_wind is a max_domains column (Registry default 0)
    # and gwd_opt one too, declared in &dynamics since v4.0 and in &physics
    # before it.  woof.core.terrain_drag ports topo_wind 1 and 2 (YSU's
    # ctopo arm) and gwd_opt 1 and 3, so they translate 1:1 and are emitted
    # only where set: an omitted key keeps every import byte-identical.
    topo_wind_raw = ph.take("topo_wind")
    if topo_wind_raw is not None:
        # WRF reads max_dom entries of a max_domains column.
        topo_wind_raw = list(topo_wind_raw[:max_dom])
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value not in (0, 1, 2) for value in topo_wind_raw):
            raise _err("physics", "topo_wind", topo_wind_raw,
                       "must be 0, 1 or 2 on every domain (WRF v4.7.1: 1 "
                       "Jimenez-Dudhia, 2 the VAR form).")
        column = list(topo_wind_raw[:max_dom])
        column = column + [0] * (max_dom - len(column))
        inert = [n for n, value in enumerate(column)
                 if value and bl_pbl_col[n] != 1]
        if any(bl_pbl_col[n] == 11 for n in inert):
            raise _err("physics", "topo_wind", topo_wind_raw,
                       "WRF's Shin-Hong PBL (bl_pbl_physics = 11) reads the "
                       "topo_wind coefficients too, and woof carries that "
                       "arm in YSU only, so the drag WRF would apply under "
                       "Shin-Hong would be missing.")
        if inert:
            drop("physics", "topo_wind", topo_wind_raw,
                 "inert on domain(s) "
                 + ", ".join(str(n + 1) for n in inert)
                 + ": WRF reads the topo_wind coefficients only in YSU and "
                 "Shin-Hong, and those domains run neither")
            for n in inert:
                column[n] = 0
        if any(column):
            terrain_drag_columns["topo_wind"] = column
    precipitation_columns = {key: ph.col(key, max_dom, 0.0)
                             for key in ("prec_acc_dt", "prec_acc_dt1")}
    precipitation_columns["reset_interval1"] = reset_interval1_column
    hail_values = {
        "hailcast_opt": _uniform("physics", "hailcast_opt", ph.col("hailcast_opt", max_dom, -1)),
        "haildt": _uniform("physics", "haildt", ph.col("haildt", max_dom, 60.0)),
    }
    from woof.lightning_contract import LIGHTNING_DEFAULTS
    lightning_values = {key: _uniform("physics", key, ph.col(key, max_dom, default))
                        for key, default in LIGHTNING_DEFAULTS.items()}
    if lightning_values["lightning_option"] == 0:
        # Off reads none of the lightning controls: the emitted TOML stays
        # byte-identical to an import made before they existed, and a
        # moved control is reported as dropped, not silently carried.
        for key, default in LIGHTNING_DEFAULTS.items():
            if lightning_values[key] != default:
                drop("physics", key, lightning_values[key],
                     "inert: lightning_option = 0 reads no lightning control")
        lightning_values = {}
    gwd_physics_raw = ph.take("gwd_opt")
    # cu_diag per domain.  WRF decides it from cu_physics, not from the
    # namelist: module_check_a_mundo stops the run when cu_diag = 1 sits on
    # a domain whose scheme is not Grell-Devenyi, Grell-Freitas, KF-CuP or
    # G3, and then sets cu_diag = 1 on those four and 0 everywhere else.
    _cu_diag_schemes = (3, 5, 10, 93)
    for n in range(max_dom):
        if g3_controls["cu_diag"][n] != 0 and cu[n] not in _cu_diag_schemes:
            raise _err("physics", "cu_diag", g3_controls["cu_diag"],
                       f"cu_diag = 1 on domain {n + 1} needs cu_physics 3, 5, "
                       f"10 or 93 there; wrf.exe itself stops this namelist "
                       f"(module_check_a_mundo: 'Using cu_diag=1 requires use "
                       f"of one of the following CU schemes'), and domain "
                       f"{n + 1} runs cu_physics = {cu[n]}.")
    if any(value == 5 for value in cu):
        validate_g3_dimensions(*(g3_controls[key] for key in
            ("maxiens", "maxens", "maxens2", "maxens3", "ensdim")), cugd_avedx)
    else:
        # No domain runs Grell 3D, so none of these controls changes the
        # forecast, and wrf.exe accepts and ignores them.  The five ensemble
        # dimensions only size closure arrays that the cumulus driver hands
        # to G3DRV and GDDRV (module_cumulus_driver.F ENSDIM=... arguments);
        # Grell-Freitas carries its own maxens3 parameter
        # (module_cu_gf_deep.F) and every other scheme never sees them.
        # NCAR's CONUS benchmark namelist sets all five to 1 under Tiedtke.
        # convtrans_avglen_m and cu_diag only drive convtrans_prep's
        # time-averaged cloud and rain diagnostics; outside G3 woof does not
        # write those, and for the schemes left here WRF overwrites cu_diag
        # itself.  2.9.0 refused all of this as "cu_physics=5 is required"
        # and so refused NCAR's benchmark as shipped.
        for key, default in G3_RUN_DEFAULTS.items():
            if key == "cugd_avedx":
                continue
            if key not in g3_named:
                continue
            value = g3_controls[key]
            if key == "cu_diag":
                reason = ("WRF sets cu_diag from cu_physics itself "
                          "(module_check_a_mundo); it only switches on "
                          "convtrans_prep's time-averaged cloud and rain "
                          "diagnostics, which woof writes for Grell 3D only")
            elif key == "convtrans_avglen_m":
                reason = ("averaging window of convtrans_prep's time-averaged "
                          "diagnostics; woof writes those for Grell 3D only "
                          "and no domain selects it")
            else:
                reason = ("Grell 3D / Grell-Devenyi ensemble dimension; inert "
                          "with neither scheme on any domain")
            drop("physics", key, value if isinstance(value, list) else [value],
                 reason)
    # The eclipse window: WRF evaluates obscuration at xtime + radt
    # (module_radiation_driver.F), so the last call can see radt minutes
    # past the forecast end.
    _eclipse_end = (start_time + timedelta(seconds=run_seconds)
                    + timedelta(minutes=max([0.0, *radt])))
    eclipse_sw = [int(v) for v in ph.col("ra_sw_eclipse", max_dom, 1)]
    _ra_sw_eclipse_decision(eclipse_sw, eclipse_sw,
        run_window=(start_time, _eclipse_end), ra_sw=list(sw_wrf), fix=fix)
    _consume_unrun_physics_controls(
        ph, inp, max_dom, cu, drop, fix, substitutions,
        run_window=(start_time, _eclipse_end), ra_sw=list(sw_wrf),
        radar_tendency_domains=frozenset(
            int(gid) for gid in wrfinput_radar_tendency_domains),
        grid_ids=tuple(grid_ids[:max_dom]))
    windfarm_col = [0] * max_dom
    windfarm_tke_factor = 0.25
    if any(value != 0 for value in ph.entries.get("windfarm_opt", [])):
        windfarm_col = ph.col("windfarm_opt", max_dom, 0)
        if any(type(value) is not int or value not in (0, 1) for value in windfarm_col):
            raise _err("physics", "windfarm_opt", windfarm_col,
                       "only 0 (off) and 1 (Fitch) are implemented; MAV option 2 requires subgrid wakes that this coupling does not calculate")
        if any(option and pbl != 5 for option, pbl in zip(windfarm_col, bl_pbl_col)):
            raise _err("physics", "windfarm_opt", windfarm_col,
                       "Fitch requires MYNN bl_pbl_physics=5; other PBL schemes do not receive the turbine TKE source")
        ij = ph.take("windfarm_ij")
        if ij is not None and any(value != 0 for value in ij):
            raise _err("physics", "windfarm_ij", ij,
                       "grid-index turbine ownership is not mapped across nested domains; use geographic windturbines.txt and windfarm_ij=0")
        factor = ph.take("windfarm_tke_factor")
        if factor is not None:
            windfarm_tke_factor = _uniform("physics", "windfarm_tke_factor", factor)
            if isinstance(windfarm_tke_factor, bool) or not isinstance(windfarm_tke_factor, (int, float)) or not math.isfinite(windfarm_tke_factor) or windfarm_tke_factor < 0:
                raise _err("physics", "windfarm_tke_factor", factor,
                           "must be finite and nonnegative to prevent a negative or nonfinite TKE source")
        fix("physics", "windfarm_opt", windfarm_col, windfarm_col,
            "Fitch GPU drag and turbine-added TKE; geographic windturbines.txt beside namelist.input")
    _consume_unported_package_selectors(ph, "physics", max_dom, drop)
    ph.finish()

    # ---- switches WRF's namelist cannot state -----------------------------
    # ``moist_cq`` has no WRF namelist key.  It lands in every prepared-cache
    # domain identity, so the importer, the shipped physics profiles and the
    # domain wizard have to give the same answer for the same suite or a
    # root sealed from a profile can never match a hierarchy imported from
    # the same namelist.  physics_compat owns that answer; this is a lookup,
    # not a second opinion.  (``top_lid`` HAS a namelist key, and an omitted
    # one is WRF's Registry default below, not this lookup's answer.)
    from woof.physics_compat import implicit_runtime_switches

    implicit = implicit_runtime_switches(
        mp_physics=mp_physics, sf_sfclay_physics=sfclay,
        sf_surface_physics=sfsfc, bl_pbl_physics=bl_pbl_physics,
        cu_physics=cu[0], num_soil_layers=resolved_soil_layers,
        ra_lw_physics=ra_lw_physics, ra_sw_physics=ra_sw_physics)
    # A native file can carry passive water vapor with microphysics off.
    # Its variable header is the authority; a generic namelist without
    # native inputs retains the dry off-profile contract.
    qv_domains = frozenset(wrfinput_qv_domains)
    unknown_qv_domains = qv_domains - set(grid_ids)
    if unknown_qv_domains:
        raise ValueError(
            f"native water-vapor domain(s) are not declared by the namelist: "
            f"{sorted(unknown_qv_domains)}")
    moist_column = tuple(mp_physics > 0 or grid_id in qv_domains
                         for grid_id in grid_ids)

    # ---- &dynamics --------------------------------------------------------
    # Omitted keys take WRF Registry defaults (F2): hybrid_opt 2
    # (registry.hyb_coord:62), damp_opt 3 (Registry.EM_COMMON:2848);
    # km_opt's Registry default is -1 = MUST-SET, so omission (or -1) is
    # a hard error rather than a silently invented scheme.
    use_theta_m_values = dyn.take("use_theta_m")
    # An omitted scalar takes the Registry default of the WRF line the
    # namelist was written for: V4 Registry.EM_COMMON:2860 says 1, the
    # V3.9 line operational HRRR runs says 0 (OMITTED_USE_THETA_M_BY_WRF_
    # VERSION).  woof's h_diabatic/moist-physics transcription implements
    # only the dry-theta (use_theta_m=0) branch, so a V4 omission must not
    # silently select 0; it is booked below as the substitution it is.
    use_theta_m = (omitted_use_theta_m(wrf_version)
                   if use_theta_m_values is None
                   else int(use_theta_m_values[0]))
    if wrf_boundary_use_theta_m is not None:
        if (isinstance(wrf_boundary_use_theta_m, bool) or
                wrf_boundary_use_theta_m not in (0, 1)):
            raise ValueError("WRF boundary thermodynamic identity must be 0 or 1")
        if use_theta_m != wrf_boundary_use_theta_m:
            raise ValueError("namelist use_theta_m differs from the producing WRF files")
    if use_theta_m not in THETA_M_ADMITTED:
        raise _err("dynamics", "use_theta_m", use_theta_m, "requires 0 or 1")
    # A SUBSTITUTION, NOT A FIX, AND NOT A REFUSAL.  use_theta_m = 1 (WRF's
    # Registry default when omitted) selects the moist-theta prognostic for
    # the whole integration; ArWen integrates dry theta and has no such
    # branch.  The initial and boundary fields are recovered exactly on all
    # three routes -- metgrid TT is physical temperature, a moist wrfbdy's
    # THM/QV/MU are converted at each forcing time, and native
    # initialization builds dry theta from physical temperature -- but the
    # INTEGRATION differs, so it is booked where the doors' no-substitution
    # gate and the terminal announcement can see it, with the reason (not
    # in the bucket for keys with exactly one implemented value, where it
    # reached the user through nothing but the receipt file, ENG-016).
    # The bare door used to REFUSE the same configuration the other two
    # announced, which is one namelist with two answers.
    decision = theta_m_decision(
        use_theta_m, metgrid_initialization=metgrid_initialization,
        wrf_boundary_use_theta_m=wrf_boundary_use_theta_m)
    if decision.moist_theta:
        substitutions.append(Substitution(
            key="use_theta_m", wrf_value=1,
            wrf_name="moist potential temperature (theta_m) prognostic",
            gpuwm_key="use_theta_m", gpuwm_value=0,
            gpuwm_name="dry potential temperature",
            reason=decision.reason))
    else:
        fix("dynamics", "use_theta_m", use_theta_m_values, 0, decision.reason)
    top_lid_values = dyn.col("top_lid", max_dom)
    if top_lid_values is None:
        # WRF's Registry default (Registry.EM_COMMON:2898, .false.): an
        # omitted top_lid is an OPEN top in wrf.exe, so it is one here.
        # Until 2026-10-12 an omission resolved to the shipped profiles'
        # certified rigid lid instead -- a different upper boundary from the
        # one the namelist describes, recorded only as a footnote.  Measured
        # cost on NCAR's CONUS 12 km benchmark restart hour (2.8.9,
        # UCAR-BENCHMARK-PARITY-2026-10-09/STATUS-289.md): PH RMS 28.2 m2/s2
        # with the rigid lid against 2.55 with WRF's open top (NCAR's own
        # ifort-vs-gnu spread 1.43), MU 6.63 Pa against 0.75.  The guard
        # that lid cited (the 2026-07-18 two-domain NaN probe) is kept where
        # it binds -- RunConfig's default for ArWen's OWN configurations
        # (woof/config.py top_lid) -- and every namelist woof itself
        # writes states top_lid explicitly (hrrr_route_inputs.py), so a
        # profile's emitted namelist still reads back as its rigid lid.
        top_lid = False
    else:
        top_lid = bool(_uniform(
            "dynamics", "top_lid",
            _require_bools("dynamics", "top_lid", top_lid_values)))
    hybrid_opt = int(dyn.scalar("hybrid_opt", 2))
    etac = float(dyn.scalar("etac", 0.2))
    w_damping = int(dyn.scalar("w_damping", 0))
    # WRF zadvect_implicit (Registry.EM_COMMON:2873, one value for every
    # domain, default 0).  CHK_IEVA tests ``> 0`` alone -- its 2 and 3
    # variants are commented out of WRF 4.7.1 -- so every positive value
    # is the one scheme woof runs as 1, and zero or below is off.
    zadvect_implicit = 1 if int(dyn.scalar("zadvect_implicit", 0)) > 0 else 0
    # WRF w_crit_cfl (Registry.EM_COMMON:2872, one value for every domain,
    # default 1.0): where w_damp measures its excess from, and its onset
    # under zadvect_implicit (A165).  Refused here, keyed, by the rule
    # RunConfig applies, so a one-pass import lists it with the others.
    w_crit_cfl = dyn.scalar("w_crit_cfl")
    if w_crit_cfl is not None:
        from woof.config import w_crit_cfl_refusal
        why = w_crit_cfl_refusal(w_crit_cfl, w_damping=w_damping,
                                 zadvect_implicit=zadvect_implicit)
        if why is not None:
            raise _err("dynamics", "w_crit_cfl", w_crit_cfl, why)
        w_crit_cfl = float(w_crit_cfl)
    # Registry.EM_COMMON initializes every domain to 0.1.  A scalar
    # ``epssm = 0.5`` namelist assignment changes d01 only; its unassigned
    # tail remains 0.1 (as confirmed by WRF's effective namelist.output).
    # Preserve that per-domain state instead of broadcasting the scalar.
    # A domain the namelist did not assign runs WRF's DEFAULT, not a value
    # the user chose, and is written as the "auto" sentinel: 0.1, raised
    # to the measured off-centering floor over ground 0.1 does not hold
    # (woof.acoustic_adaptation), where a chosen value is held as written.
    epssm_given = dyn.take("epssm") or []
    epssm_assigned = min(len(epssm_given), max_dom)
    epssm = [float(value) for value in epssm_given[:max_dom]] \
        + [0.1] * (max_dom - epssm_assigned)
    if epssm_assigned < max_dom:
        unassigned = ", ".join(f"d{n + 1:02d}"
                               for n in range(epssm_assigned, max_dom))
        model_choices.append(
            ("epssm is not set, so every domain takes"
             if not epssm_assigned else
             f"epssm lists {epssm_assigned} value(s), so {unassigned} "
             "take(s)")
            + " WRF's Registry default 0.1, written epssm = \"auto\": it "
            "runs 0.1 except over terrain steeper than 0.1 was measured "
            "stable on, where the run raises it to the measured "
            "off-centering floor and says so (woof/acoustic_adaptation.py)."
            "  Write a value for a domain to hold it as chosen.")
    km_opt_given = dyn.take("km_opt")
    if not km_opt_given or km_opt_given[0] == -1:
        raise _err("dynamics", "km_opt", None,
                   "must-set namelist key (WRF Registry default -1 "
                   "refuses to run); woof will not invent a mixing "
                   "scheme for you.")
    # WRF module_check_a_mundo fills each unset (-1) child from domain 1,
    # including a tail after a mixed column. It does not repeat the last child.
    km_opt_col = km_opt_given[:max_dom] + [km_opt_given[0]] * max(
        0, max_dom - len(km_opt_given))
    km_opt_col = [km_opt_given[0] if value == -1 else value
                  for value in km_opt_col]
    # PER-DOMAIN, for the same reason bl_pbl_physics is: km_opt is in
    # woof.experiment._DOMAIN_RUN_OVERRIDES ("a PBL parent may carry a
    # PBL-off Smagorinsky child"), and a nested LES tree is exactly a
    # namelist whose km_opt column is not constant.  The root value is the
    # [shared] one; a domain that differs emits its own override.
    km_opt_col = [int(value) for value in km_opt_col]
    km_opt = km_opt_col[0]
    diff_opt_raw = dyn.take("diff_opt")
    if not diff_opt_raw or diff_opt_raw[0] == -1:
        raise _err("dynamics", "diff_opt", diff_opt_raw,
                   "must-set root namelist key: WRF's Registry default -1 "
                   "selects no diffusion operator and WRF refuses it. "
                   "Set the root diff_opt to 1 (model-coordinate horizontal "
                   "diffusion) or 2 (metric stress/scalar diffusion).")
    diff_opt_col = diff_opt_raw[:max_dom] + [diff_opt_raw[0]] * max(
        0, max_dom - len(diff_opt_raw))
    diff_opt_col = [diff_opt_raw[0] if value == -1 else value
                    for value in diff_opt_col]
    if any(isinstance(v, bool) or not isinstance(v, int) or v not in (1, 2)
           for v in diff_opt_col):
        raise _err("dynamics", "diff_opt", diff_opt_col,
                   "must select 1 (model-coordinate horizontal diffusion) "
                   "or 2 (metric stress/scalar diffusion); other values "
                   "do not select an implemented horizontal operator.")
    for index, value in enumerate(diff_opt_col):
        if value == 1 and km_opt_col[index] not in (2, 4):
            raise _err("dynamics", "diff_opt/km_opt", diff_opt_col,
                       "model-coordinate diffusion is implemented for "
                       "km_opt=2 or 4, which supply its exchange coefficients.")
    fix("dynamics", "diff_opt", diff_opt_raw,
        tuple(diff_opt_col) if len(set(diff_opt_col)) > 1 else diff_opt_col[0],
        "selects WRF's model-coordinate or metric horizontal operator")
    mix_full = _require_bools(
        "dynamics", "mix_full_fields",
        dyn.registry_col("mix_full_fields", max_dom, False))
    # Real-data base profiles are zero, so the metric operator admits
    # either runtime value. Named source requests retain supplied logical
    # values; ordinary importer emissions keep their historical full-field
    # rule under diff_opt 2. The coordinate operator keeps its own rule.
    # Consume this stock-key request here. The shared-default text helper
    # must not reapply False after explicit True was normalized out of TOML.
    named_mix_default = source_defaults.pop("mix_full_fields", None)
    mix_full_resolved = [
        value if preserve_wrf_mixing or diff_opt_col[index] == 1
        or (named_mix_default is not None and index < mix_full_supplied_count)
        else named_mix_default if named_mix_default is not None else True
        for index, value in enumerate(mix_full)]
    affected = [index + 1 for index, value in enumerate(mix_full)
                if not value and diff_opt_col[index] == 2
                and mix_full_resolved[index]]
    if affected:
        substitutions.append(Substitution(
            key="mix_full_fields", wrf_value=tuple(mix_full),
            wrf_name="WRF perturbation/full-field mixing by domain",
            gpuwm_key="mix_full_fields",
            gpuwm_value=(tuple(mix_full_resolved)
                         if 1 in diff_opt_col or not all(mix_full_resolved) else True),
            gpuwm_name=(f"full-field substitution on domains {affected}"
                        if 1 in diff_opt_col or not all(mix_full_resolved)
                        else "full-field mixing on every domain"),
            reason=(MIX_FULL_FIELDS_SUBSTITUTION
                    + (f" Only domains {affected} use this substitution."
                       if 1 in diff_opt_col else ""))))
    elif (all(value == 2 for value in diff_opt_col)
          and all(mix_full_resolved)):
        fix("dynamics", "mix_full_fields", mix_full, True,
            "explicit WRF full-field mode matches woof's implemented mixing")
    diff_6th_opt = int(_uniform("dynamics", "diff_6th_opt",
                                dyn.col("diff_6th_opt", max_dom, 0)))
    diff_6th_factor = [float(v)
                       for v in dyn.col("diff_6th_factor", max_dom, 0.12)]
    # diff_6th_factor2 (NOAA-EMC WRFV3.9 fork only, Registry.EM_COMMON:
    # 2629, &dynamics, max_domains, default 0.04): the fork's factor for
    # the moist and scalar arrays, applied on the full step.  Naming it
    # selects the fork's filter form (diff_6th_form = "noaa_wrf39").  Read
    # on the Fortran-assignment rule: an unassigned tail keeps the fork
    # Registry default.
    diff_6th_factor2 = None
    if noaa_wrf39_namelist:
        diff_6th_factor2 = [
            float(v) for v in dyn.registry_col("diff_6th_factor2", max_dom,
                                               0.04)]
        if any(not math.isfinite(v) or not 0.0 <= v <= 1.0
               for v in diff_6th_factor2):
            raise _err("dynamics", "diff_6th_factor2", diff_6th_factor2,
                       "must be a filter factor in [0, 1] (fork Registry "
                       "default 0.04).")
    # PER DOMAIN, on the same rule as diff_6th_factor beside it.  WRF
    # declares diff_6th_slopeopt max_domains (Registry.EM_COMMON) and it
    # is half of one knob whose other half (diff_6th_factor) was already
    # a column here, so a tree could tune the filter's strength per nest
    # and not its slope limiter.  Read with the importer's last-value
    # fill, exactly the column _uniform used to be handed, so a uniform
    # namelist emits a byte-identical TOML; a namelist whose column is
    # NOT uniform used to be refused and now emits [[domain]] overrides.
    diff_6th_slopeopt_col = [
        int(v) for v in dyn.col("diff_6th_slopeopt", max_dom, 0)]
    diff_6th_slopeopt = diff_6th_slopeopt_col[0]
    # moist_mix6_off (Registry.EM_COMMON:2889, &dynamics, max_domains,
    # default .false.): WRF's own switch for taking the 6th-order filter
    # off the moist scalars (dyn_em/module_em.F:1421), mapped 1:1 onto the
    # RunConfig field of the same name and spelling.  Supplied-only, on
    # the knob-parity rule: the Registry default equals woof's frozen
    # RunConfig default, so omission emits nothing and established imports
    # stay byte-identical.  Per domain on epssm's Fortran-assignment
    # semantics: an unassigned tail keeps the Registry default rather
    # than broadcasting the head value.
    moist_mix6_off_raw = dyn.take("moist_mix6_off")
    moist_mix6_off_col = None
    if moist_mix6_off_raw is not None:
        supplied = _require_bools("dynamics", "moist_mix6_off",
                                  moist_mix6_off_raw[:max_dom])
        moist_mix6_off_col = supplied + [False] * (max_dom - len(supplied))
    # The km_opt=2/3 turbulence parameter row (knob-parity lane).  Every
    # key here is `max_domains` in Registry.EM_COMMON (c_s :2862, c_k
    # :2863, mix_isotropic :2896, mix_upper_bound :2897, tke_upper_bound
    # :2899, tke_drag_coefficient :2900, tke_heat_flux :2901) and every
    # one sits in woof.experiment._DOMAIN_RUN_OVERRIDES, so each is read
    # as a COLUMN for the same reason km_opt and bl_pbl_physics are: a PBL
    # parent carrying a PBL-off LES child is exactly a namelist whose
    # turbulence columns are not constant.
    #
    # Before this, only c_s was read (and it was forced uniform); c_k,
    # mix_isotropic, mix_upper_bound, tke_upper_bound, tke_heat_flux and
    # tke_drag_coefficient were not read at all.  That left NO pair of
    # inputs able to express a km_opt=2 LES child: omitting c_k resolved
    # woof's RunConfig default 0.15 against an LES TOML's em_les 0.10 and
    # the prepared cache -- which binds the turbulence row per domain --
    # refused the forecast on `run.c_k`, while supplying c_k in the
    # namelist hit _Section.finish()'s unmapped-key refusal instead.  Both
    # directions were closed, so the whole km_opt=2 hierarchy gate was.
    #
    # Each Registry default equals woof's frozen RunConfig default, so an
    # omitted key resolves identically and established imports stay
    # byte-identical.  The root value is the [shared] one; a domain that
    # differs emits its own override, on epssm's existing rule.
    def _real_col(key: str, ok, why: str) -> list[float] | None:
        """A supplied per-domain real column, validated element-wise."""
        raw = dyn.col(key, max_dom)
        if raw is None:
            return None
        values = [float(value) for value in raw]
        if not all(ok(value) for value in values):
            raise _err("dynamics", key, raw, why)
        return values

    c_s_col = _real_col(
        "c_s", lambda v: math.isfinite(v) and v > 0.0,
        "must be a finite positive Smagorinsky constant "
        "(WRF Registry default 0.25).")
    c_k_col = _real_col(
        "c_k", lambda v: math.isfinite(v) and v > 0.0,
        "must be a finite positive TKE-closure constant "
        "(WRF Registry default 0.15; the em_les reference sets 0.10).")
    mix_upper_bound_col = _real_col(
        "mix_upper_bound", lambda v: math.isfinite(v) and v > 0.0,
        "must be a finite positive non-dimensional K cap "
        "(WRF Registry default 0.1).")
    tke_upper_bound_col = _real_col(
        "tke_upper_bound", lambda v: math.isfinite(v) and v > 0.0,
        "must be a finite positive TKE ceiling in m2 s-2 "
        "(WRF Registry default 1000.).")
    tke_heat_flux_col = _real_col(
        "tke_heat_flux", math.isfinite,
        "must be a finite kinematic heat flux in K m s-1 "
        "(WRF Registry default 0.).")
    tke_drag_coefficient_col = _real_col(
        "tke_drag_coefficient", lambda v: math.isfinite(v) and v >= 0.0,
        "must be a finite non-negative drag coefficient "
        "(WRF Registry default 0.).")
    mix_isotropic_raw = dyn.col("mix_isotropic", max_dom)
    mix_isotropic_col = None
    if mix_isotropic_raw is not None:
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value not in (0, 1) for value in mix_isotropic_raw):
            raise _err("dynamics", "mix_isotropic", mix_isotropic_raw,
                       "must be 0 (anisotropic mixing lengths) or 1 "
                       "(isotropic (dx*dy*dz)^(1/3)).")
        mix_isotropic_col = [int(value) for value in mix_isotropic_raw]
    #: The row in emission order: (TOML key, supplied column or None).
    turbulence_row = (
        ("c_s", c_s_col),
        ("c_k", c_k_col),
        ("mix_isotropic", mix_isotropic_col),
        ("mix_upper_bound", mix_upper_bound_col),
        ("tke_upper_bound", tke_upper_bound_col),
        ("tke_heat_flux", tke_heat_flux_col),
        ("tke_drag_coefficient", tke_drag_coefficient_col),
    )
    # Per domain with its slopeopt partner, and validated element-wise:
    # a column is only per-domain if every element is checked.
    diff_6th_thresh_raw = dyn.col("diff_6th_thresh", max_dom)
    diff_6th_thresh_col = (None if diff_6th_thresh_raw is None
                           else [float(v) for v in diff_6th_thresh_raw])
    diff_6th_thresh = (None if diff_6th_thresh_col is None
                       else diff_6th_thresh_col[0])
    if diff_6th_thresh_col is not None and any(
            not math.isfinite(value) or value <= 0.0
            for value in diff_6th_thresh_col):
        raise _err("dynamics", "diff_6th_thresh", diff_6th_thresh_raw,
                   "must be a finite positive terrain slope in m/m "
                   "(WRF Registry default 0.10).")
    # &dynamics keys the dycore pins where WRF has options: validated
    # against the single implemented value, recorded as fixed.
    for key, pin, why in DYNAMICS_PINS:
        raw = dyn.col(key, max_dom)
        if raw is None:
            continue
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value != pin for value in raw):
            raise _err("dynamics", key, raw,
                       f"woof implements {key} = {pin} only ({why}).")
        fix("dynamics", key, raw, pin, why)
    gwd_dynamics_raw = dyn.take("gwd_opt")
    if gwd_dynamics_raw is not None and gwd_physics_raw is not None             and list(gwd_dynamics_raw[:max_dom])             != list(gwd_physics_raw[:max_dom]):
        raise _err("dynamics", "gwd_opt", gwd_dynamics_raw,
                   f"&physics also sets gwd_opt = {gwd_physics_raw!r}; WRF "
                   "v4 reads &dynamics only, so the namelist says two "
                   "things at once.")
    gwd_raw = (gwd_dynamics_raw if gwd_dynamics_raw is not None
               else gwd_physics_raw)
    if gwd_raw is not None:
        # WRF reads max_dom entries of a max_domains column.
        gwd_raw = list(gwd_raw[:max_dom])
    gwd_section = "dynamics" if gwd_dynamics_raw is not None else "physics"
    if gwd_raw is not None:
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value not in (0, 1, 3) for value in gwd_raw):
            raise _err(gwd_section, "gwd_opt", gwd_raw,
                       "must be 0, 1 or 3 on every domain (WRF v4.7.1: 1 the "
                       "KIM drag, 3 the GSL drag suite; 2 is not ported).")
        column = list(gwd_raw[:max_dom])
        column = column + [0] * (max_dom - len(column))
        no_pbl = [n for n, value in enumerate(column)
                  if value and bl_pbl_col[n] == 0]
        if no_pbl:
            drop(gwd_section, "gwd_opt", gwd_raw,
                 "inert on domain(s) "
                 + ", ".join(str(n + 1) for n in no_pbl)
                 + ": WRF adds the drag inside its PBL driver, which a "
                 "domain without a PBL scheme never calls")
            for n in no_pbl:
                column[n] = 0
        if any(column):
            terrain_drag_columns["gwd_opt"] = column
    drop("dynamics", "gwd_diags", dyn.take("gwd_diags"),
         "output only: the gravity-wave-drag diagnostics (DTAUX3D, DUSFCG "
         "and the GSL components) are not written to history")
    # tke_adv_opt (Registry.EM_COMMON:2880, max_domains, default 1).  The
    # old rationale here -- "inert: no prognostic-TKE mixing scheme is
    # selectable (km_opt 1 and 4 only)" -- was falsified by km_opt=2, and
    # a dropped key that DOES change the answer is exactly the silent loss
    # _Section.finish() exists to prevent.  Which of the two it is now
    # depends on the mixing scheme actually selected:
    #
    # * with a km_opt=2 domain in the tree, TKE is a prognostic carrier
    #   that woof advects through the moist-scalar rows, and it takes
    #   WRF's POSITIVE-DEFINITE update -- tke_adv_opt = 1's branch
    #   (woof/core/moist.py advance_tke_stage, WRF solve_em.F:2100-2119)
    #   -- unconditionally.  woof implements no other transport for it,
    #   so 1 is pinned and 0/2 are refused rather than dropped;
    # * with no km_opt=2 domain there is no TKE carrier to transport, so
    #   WRF would not consume the key either and it is dropped as inert.
    tke_adv_opt_raw = dyn.take("tke_adv_opt")
    if 2 in km_opt_col:
        if tke_adv_opt_raw is not None:
            column = list(tke_adv_opt_raw[:max_dom])
            column += [column[-1]] * (max_dom - len(column))
            if any(isinstance(value, bool) or not isinstance(value, int)
                   or value != 1 for value in column):
                raise _err(
                    "dynamics", "tke_adv_opt", tke_adv_opt_raw,
                    "woof implements tke_adv_opt = 1 only (the "
                    "positive-definite RK3 TKE transport of WRF "
                    "solve_em.F:2100-2119, which km_opt = 2 makes live).")
            fix("dynamics", "tke_adv_opt", tke_adv_opt_raw, 1,
                "the km_opt=2 TKE carrier takes WRF's positive-definite "
                "RK3 transport, the only one woof implements")
    else:
        drop("dynamics", "tke_adv_opt", tke_adv_opt_raw,
             "inert: no domain selects a prognostic-TKE mixing scheme "
             "(km_opt = 2), so there is no TKE carrier to transport and "
             "WRF would not consume it either")
    base_temp = float(dyn.scalar("base_temp", 290.0))
    damp_opt = int(dyn.scalar("damp_opt", 3))
    # THE DAMPING AND CONSTANT-K ROW, per domain.  Every key here is
    # max_domains in Registry.EM_COMMON and none of them was exposed per
    # domain, so a tree could not damp its inner nest differently from
    # its root -- and on a 10/2/0.667 km tree it must, because the
    # relaxation sponge is 40 km wide on the root and 2.7 km on the inner
    # nest at the same cell count.  Same last-value fill the refusal
    # above was reading, so every uniform namelist still emits a
    # byte-identical TOML.
    zdamp_col = [float(v) for v in dyn.col("zdamp", max_dom, 5000.0)]
    dampcoef_col = [float(v) for v in dyn.col("dampcoef", max_dom, 0.2)]
    khdif_col = [float(v) for v in dyn.col("khdif", max_dom, 0)]
    kvdif_col = [float(v) for v in dyn.col("kvdif", max_dom, 0)]
    zdamp, dampcoef = zdamp_col[0], dampcoef_col[0]
    khdif, kvdif = khdif_col[0], kvdif_col[0]
    # Element-wise, per domain: the two mixing schemes conflict on the
    # domain that selects both, not on the tree.
    for index, km_value in enumerate(km_opt_col):
        if km_value == 4 and (khdif_col[index] > 0.0
                              or kvdif_col[index] > 0.0):
            raise _err(
                "dynamics", "km_opt", km_opt_col,
                f"selects WRF Smagorinsky mixing on domain {index + 1}, "
                "but khdif/kvdif also enable the km_opt=1 constant-K "
                "operator there; choose exactly one mixing scheme.")
    non_hydro = dyn.col("non_hydrostatic", max_dom)
    if non_hydro is not None and not all(_require_bools(
            "dynamics", "non_hydrostatic", non_hydro)):
        raise _err("dynamics", "non_hydrostatic", non_hydro,
                   DYNAMICS_REQUIRED_VALUES["non_hydrostatic"][1])
    fix("dynamics", "non_hydrostatic", non_hydro, True,
        "woof is nonhydrostatic-only")
    # Per domain in the Registry and per domain here, though only one
    # value imports today: the column is read and checked element-wise so
    # a tree that really did carry a mixed advection option is refused
    # naming it, rather than passing a uniformity check that never saw
    # the tail.
    moist_adv_opt_col = [
        int(v) for v in dyn.col("moist_adv_opt", max_dom, 1)]
    moist_adv_opt = moist_adv_opt_col[0]
    scalar_adv = dyn.col("scalar_adv_opt", max_dom, 1)
    scalar_adv_opt = int(_uniform(
        "dynamics", "scalar_adv_opt", scalar_adv))
    if any(value != 1 for value in moist_adv_opt_col) or scalar_adv_opt != 1:
        raise _err(
            "dynamics", "moist_adv_opt/scalar_adv_opt",
            (moist_adv_opt_col, scalar_adv_opt),
            "only the matched WRF positive-definite option 1 is "
            "implemented for both moisture mass and scalar/number fields.")
    fix("dynamics", "scalar_adv_opt", scalar_adv, 1,
        "validated equal to the supported moist_adv_opt=1 path")
    time_step_sound = int(dyn.scalar("time_step_sound", 0))
    if time_step_sound == 0 and not use_adaptive_time_step:
        # WRF Registry default 0 on a fixed step: solve_em.F derives the
        # count itself, max(2 * (INT(300 * dt / (min(dx, dy) / max_msft)
        # - 0.01) + 1), 4), from the step and the root's largest map
        # factor.  That is the count wrf.exe integrates with this
        # namelist, so it is the count the import writes (4 at the usual
        # 6 x dx step, 6 from about 10 x dx).
        from woof.core.adaptive_clock import wrf_num_sound_steps

        rule_step = Fraction(time_step) + Fraction(fract_num, fract_den)
        rule_factor = largest_mass_map_factor(
            projection, map_proj, dx=root_dx, dy=root_dy,
            e_we=int(e_we[0]), e_sn=int(e_sn[0]))
        time_step_sound = int(wrf_num_sound_steps(
            float(rule_step), float(root_dx), float(root_dy), rule_factor))
        defaults_applied.append(AppliedDefault(
            key="time_step_sound", value=time_step_sound,
            reason="WRF Registry default 0 means WRF derives the count "
                   "(solve_em.F: max(2 * (INT(300 * dt / (min(dx, dy) / "
                   "max_msft) - 0.01) + 1), 4)); at "
                   f"{_seconds_text(rule_step)} on {root_dx:g} m with a "
                   f"largest map factor of {rule_factor:.6f} that is "
                   f"{time_step_sound}"))
    elif time_step_sound == 0:
        # WRF Registry default 0 = automatic selection; WRF picks 4 for
        # these dt/dx regimes and woof needs an explicit even count --
        # recorded, never silent (F2).
        time_step_sound = 4
        defaults_applied.append(AppliedDefault(
            key="time_step_sound", value=4,
            reason="WRF Registry default 0 means automatic selection; "
                   "4 is WRF's choice at conventional dt/dx and woof "
                   "requires an explicit even value"))
    # ---- WRF's fixed step through the adaptive clock ---------------------
    # A single-domain namelist that sets use_adaptive_time_step with
    # starting = max = min (HRRR v4: 20/20/20) names a FIXED step, and
    # WRF then derives the acoustic substep count itself: start_em.F
    # forces time_step_sound to 0 under the adaptive clock and solve_em.F
    # takes num_sound_steps from the live step and the grid's largest map
    # factor.  The namelist's own time_step_sound is not what integrates.
    # At a source request that explicitly selects the pinned clock, the
    # TOML says what runs: the fixed step, the derived count, and a
    # pinned terrain clock, because the clock is the namelist's own and
    # the engine's measured maps were never taken on an operational grid.
    # Other imports retain their established adaptive emission.
    #
    # A pair whose own woof selector comment states terrain_clock was
    # written by woof from a configuration (the HRRR route's renderer
    # carries terrain_clock on every emission), so its adaptive keys are
    # that configuration's clock, not a WRF namelist to be read the way
    # WRF runs it: the configuration's use_adaptive_time_step and bounds
    # come back as written.  Collapsing them refused the route's round
    # trip of an adaptive tree pinned at equal bounds
    # (configs/recipes/hrrr_v4_gsd41.toml, 20/20/20 s), and the prepared
    # cache binds use_adaptive_time_step, so a tree prepared from the pair
    # would have been refused by the forecast reading the configuration.
    clock_stated_by_gpuwm = "terrain_clock" in carried_selectors
    terrain_clock = None
    fixed_through_adaptive = (_fixed_step_through_adaptive_clock(
        use_adaptive_time_step,
        (starting_time_step, starting_time_step_den),
        (max_time_step, max_time_step_den),
        (min_time_step, min_time_step_den),
        projection=projection, map_proj=map_proj, dx=root_dx, dy=root_dy,
        e_we=int(e_we[0]), e_sn=int(e_sn[0]), max_dom=int(max_dom))
        if (named_source_defaults.get("terrain_clock") == "pinned"
            and source_defaults.get("terrain_clock") == "pinned"
            and not clock_stated_by_gpuwm) else None)
    if fixed_through_adaptive is not None:
        step, derived, largest_map_factor = fixed_through_adaptive
        namelist_step = Fraction(time_step) + Fraction(fract_num, fract_den)
        fix("domains", "use_adaptive_time_step", [True], False,
            "starting_time_step = max_time_step = min_time_step is WRF's "
            f"fixed step of {_seconds_text(step)} through the adaptive "
            "clock (adapt_timestep_em.F clamps every step to that value); "
            "the engine runs it as its fixed clock")
        if namelist_step != step:
            fix("domains", "time_step", [time_step], _seconds_text(step),
                "under use_adaptive_time_step WRF integrates "
                "starting_time_step (start_em.F: grid%dt = "
                "grid%starting_time_step), not time_step")
            time_step = int(step.numerator // step.denominator)
            remainder = step - time_step
            fract_num, fract_den = (int(remainder.numerator),
                                    int(remainder.denominator))
        if time_step_sound != derived:
            fix("dynamics", "time_step_sound", [time_step_sound], derived,
                "under use_adaptive_time_step WRF sets time_step_sound to 0 "
                "(start_em.F) and derives the count from the step and the "
                "grid's largest map factor (solve_em.F: max(2 * (INT(300 * "
                "dt / (dx / max_msft) - 0.01) + 1), 4)); at "
                f"{_seconds_text(step)} on {root_dx:g} m with a largest map "
                f"factor of {largest_map_factor:.6f} that is {derived}")
        time_step_sound = int(derived)
        use_adaptive_time_step = False
        terrain_clock = "pinned"
    # ---- WRF's own acoustic count on a fixed step (D8, 2.8.9) -----------
    # A WRF namelist on a fixed step names the acoustic substep count
    # wrf.exe integrates: its time_step_sound, or WRF's own rule for 0
    # (above).  An import runs that count by default, to spec: the clock
    # is pinned, and the terrain clock's measured verdict is carried as
    # advice in the receipt and as a caution line where it differs.  The
    # breakage this prevents: NCAR's CONUS 12 km restart, which WRF runs
    # on 4 substeps, ran on 6 under the terrain clock (10 percent per step,
    # 117.6 s against 72.9 s end to end on the restart leg) and sat
    # further from WRF's own output than WRF's 4, which ran stable there.
    # The terrain clock stays the default where the input does not name
    # WRF's count: an adaptive namelist (WRF derives the count from a step
    # the clock chooses), and a selector comment or named-source table
    # that states terrain_clock (that statement is the input's choice).
    if (terrain_clock is None and not use_adaptive_time_step
            and "terrain_clock" not in source_defaults):
        terrain_clock = "pinned"
        defaults_applied.append(AppliedDefault(
            key="terrain_clock", value="pinned",
            reason="a fixed-step WRF namelist names the acoustic substep "
                   f"count wrf.exe runs ({time_step_sound}); the import "
                   "runs it, and the measured terrain clock's verdict is "
                   "advice in the receipt (state terrain_clock in a "
                   "gpuwm-physics-selectors-v1 comment to let it adapt "
                   "the clock)"))
    # smdiv/emdiv/h_sca_adv_order are max_domains in the Registry and were
    # read here as scalars, which silently took element 1 and dropped a
    # tail the file really carried.  Columns now, on the same last-value
    # fill: a scalar namelist resolves to the identical value on every
    # domain, so established imports are byte-identical.
    smdiv_col = [float(v) for v in dyn.col("smdiv", max_dom, 0.1)]
    smdiv = smdiv_col[0]
    # WRF Registry defaults the namelist leaves unset, ratified binding
    # (Phase-4 native-dt baseline): emdiv 0.01, h_sca_adv_order 5.
    # (hypsometric_opt's binding is the same, but it is read from
    # &domains, where WRF declares it -- see the &domains block above.)
    emdiv_col = [float(v) for v in dyn.col("emdiv", max_dom, 0.01)]
    emdiv = emdiv_col[0]
    h_sca_adv_order_col = [
        int(v) for v in dyn.col("h_sca_adv_order", max_dom, 5)]
    h_sca_adv_order = h_sca_adv_order_col[0]
    # tke_budget (Registry.EM_COMMON, &dynamics, max_domains, default 0),
    # supplied-only on the knob-parity rule: the Registry default equals
    # woof's frozen RunConfig default, so an omitted key emits nothing
    # and every established import stays byte-identical.  Per domain
    # because the diagnostic's cost scales with the grid, so a tree can
    # accumulate the budget on the nest being read and leave it off the
    # rest -- the same argument sase_flux_diag carries.
    tke_budget_raw = dyn.take("tke_budget")
    tke_budget_col = None
    if tke_budget_raw is not None:
        supplied = [int(v) for v in tke_budget_raw[:max_dom]]
        tke_budget_col = supplied + [0] * (max_dom - len(supplied))
        if any(value not in (0, 1) for value in tke_budget_col):
            raise _err("dynamics", "tke_budget", tke_budget_raw,
                       "must be 0 (off) or 1 (per-step term-by-term TKE "
                       "budget accumulation).")
    # v_sca_adv_order / v_mom_adv_order (Registry.EM_COMMON, &dynamics,
    # max_domains, default 3), supplied-only on the same knob-parity
    # rule as tke_budget: the Registry default equals the frozen
    # RunConfig default, so an omitted key emits nothing and every
    # established import stays byte-identical.  Per domain because WRF
    # declares them so and operational HRRR runs 5 on its root with 3 on
    # the nests it declares.  The vertical ladder (woof/core/kernels/
    # advection.cu zface_half, advect_w's mass-level faces) carries WRF's
    # vert_order 3 and 5; any other value is refused by name.
    vertical_order_columns = {}
    for _order_key in ("v_sca_adv_order", "v_mom_adv_order"):
        _order_raw = dyn.take(_order_key)
        vertical_order_columns[_order_key] = None
        if _order_raw is None:
            continue
        # The omitted tail keeps the Registry default 3, as wrf.exe reads
        # a max_domains array (namelist_defaults sets every entry, the
        # namelist read overwrites only the supplied ones) and as
        # tke_budget above fills: `v_sca_adv_order = 5,` with two domains
        # runs the nest at 3 in WRF, so it imports that way.
        _supplied = [v for v in _order_raw[:max_dom]]
        _column = _supplied + [3] * (max_dom - len(_supplied))
        for _value in _column:
            _why = advection_order_refusal(
                v_sca_adv_order=_value if _order_key == "v_sca_adv_order" else 3,
                v_mom_adv_order=_value if _order_key == "v_mom_adv_order" else 3,
                h_mom_adv_order=5)
            if _why is not None:
                raise _err("dynamics", _order_key, _order_raw, _why)
        # The Registry default column was omitted by the established
        # importer. Preserve those exact bytes, while retaining any
        # mixed or nondefault column and its explicit nest entries.
        if any(value != 3 for value in _column):
            vertical_order_columns[_order_key] = [int(v) for v in _column]
    v_sca_adv_order_col = vertical_order_columns["v_sca_adv_order"]
    v_mom_adv_order_col = vertical_order_columns["v_mom_adv_order"]
    # The importer refuses rather than tolerates the old placement.  A
    # namelist carrying hypsometric_opt in &dynamics is one wrf.exe
    # cannot read at all, so silently honouring it here would hand a
    # woof forecast a setting its mirrored WRF arm could never run;
    # generic "unmapped key" would be true but would not say where the
    # key belongs, and this refusal has one obvious repair.
    stale_hypsometric = dyn.take("hypsometric_opt")
    if stale_hypsometric is not None:
        raise _err(
            "dynamics", "hypsometric_opt", stale_hypsometric,
            "WRF declares hypsometric_opt in &domains as a scalar "
            "(Registry.EM_COMMON:2283, `namelist,domains`, nentries 1); "
            "in &dynamics it fails wrf.exe's own namelist read before "
            "the first timestep.  Move the key to &domains.")
    if any(value != DYNAMICS_REQUIRED_VALUES["h_sca_adv_order"][0]
           for value in h_sca_adv_order_col):
        raise _err(
            "dynamics", "h_sca_adv_order", h_sca_adv_order_col,
            "only the WRF Registry default 5 imports: woof's transported-"
            "scalar stencils are fixed at WRF's 5th-order horizontal form "
            "(the vertical order is v_sca_adv_order), and the configurable "
            "h_sca_adv_order feeds only the geopotential advection "
            "(rhs_ph) -- a non-default value here would not mean what it "
            "means in WRF.")
    # The base helper drops tracer_opt (and its filter switch) only when no
    # domain asks for a tracer package; SFIRE's option 3 is left for the
    # fire tracer row below.
    chem_dynamics = _consume_unrun_dynamics_controls(dyn, inp, max_dom,
                                                     km_opt_col, drop, fix)
    tracer_values = dyn.take("tracer_opt")
    if tracer_values is not None:
        if any(isinstance(value, bool) or not isinstance(value, int) or value not in (0, 3)
               for value in tracer_values if value is not None):
            raise _err("dynamics", "tracer_opt", tracer_values,
                       "SFIRE bulk import implements 0 and native fire tracer option 3")
        tracer_column = [0 if value is None else value for value in tracer_values[:max_dom]]
        tracer_column += [0] * (max_dom - len(tracer_column))
        if any(value == 3 for value in tracer_column):
            # Carry the same row across the nesting tree; only attached
            # ifire=2 domains emit. Parent carriers have zero source.
            fire_columns["fire_smoke"] = [True] * max_dom
        fix("dynamics", "tracer_opt", tracer_values, tracer_column[0],
            "native bulk SFIRE tracer, converted from g/kg air to ug/kg dry air")
    for key in ("umove", "vmove"):
        values = dyn.take(key)
        if values is not None and any(value != 0 for value in values if value is not None):
            raise _err("dynamics", key, values,
                       "umove/vmove require flat idealized Cartesian domains; this real-data import fixes terrain and land use to the ground, which would need translation each step")
    dyn.finish()

    # ---- &bdy_control -----------------------------------------------------
    spec_bdy_width = int(bdy.scalar("spec_bdy_width", 5))
    spec_zone = int(bdy.scalar("spec_zone", 1))
    relax_zone = int(bdy.scalar("relax_zone", 4))
    spec_exp = float(bdy.scalar("spec_exp", 0.0))
    # WRF defaults when &bdy_control omits the flags: the head grid takes
    # external specified LBCs, every child is nested (the bundle sets
    # specified = T,F,F,F / nested = F,T,T,T explicitly).
    specified_col = bdy.col("specified", max_dom)
    specified = _require_bools("bdy_control", "specified", specified_col) \
        if specified_col is not None \
        else [True] + [False] * (max_dom - 1)
    if _mp28_fallback_row is not None and any(specified):
        # The clause the mp=28 receipt above deferred until this line
        # parsed the boundary condition it is conditional on.  Appended
        # to the row that is already there rather than added as a second
        # one, so the reader sees one statement about their aerosol
        # initial state.
        from woof.config import MP28_AEROSOL_LATERAL_FORCING_PRECONDITION

        _row = defaults_applied[_mp28_fallback_row]
        defaults_applied[_mp28_fallback_row] = _dataclass_replace(
            _row,
            reason=_row.reason + " "
            + MP28_AEROSOL_LATERAL_FORCING_PRECONDITION)
    nested_col = bdy.col("nested", max_dom)
    nested = _require_bools("bdy_control", "nested", nested_col) \
        if nested_col is not None \
        else [False] + [True] * (max_dom - 1)
    bdy.finish()

    if stochastic_controls is not None:
        from types import SimpleNamespace
        from woof.config import validate_spp_config
        for n in range(max_dom):
            current_flags = {key: column[n] for key, column in stochastic_flags.items()}
            try:
                validate_spp_config(SimpleNamespace(**current_flags,
                    cu_physics=cu[n], bl_pbl_physics=bl_pbl_col[n],
                    sf_sfclay_physics=sfclay, sf_surface_physics=sfsfc))
            except ValueError as problem:
                key = next((key for key in stochastic_flags if key in str(problem)), "spp")
                raise _err("stoch", key, current_flags[key] if key in current_flags else 1,
                    str(problem)) from problem

    # ---- emit the resolved TOML ------------------------------------------
    if name is None:
        name = f"wrf_{start_time:%Y%m%d%H}_{max_dom}dom"
    lines = [
        "# woof experiment translated from WRF namelists by "
        "`woof import-namelist`",
        f"# ({wps_path.name} + {input_path.name}).  WRF staggered "
        "e_we/e_sn/e_vert convert",
        "# to woof mass dimensions nx/ny/nz (one fewer point).  Child "
        "dx/dt are NEVER",
        "# written: woof/experiment.py derives them exactly from the "
        "parent chain",
        "# (the namelist chain is authoritative).",
    ]
    if substitutions:
        lines.append("# Physics substitutions (ratified, never silent):")
        for s in substitutions:
            lines.append(f"#   {s.key} {s.wrf_value} ({s.wrf_name}) -> "
                         f"{s.gpuwm_key} {s.gpuwm_value} ({s.gpuwm_name})")
    if use_theta_m_values is None and wrf_version_source is not None:
        # Written only when something CHOSE the line (the flag, the
        # files' TITLE): an import on the default line keeps the bytes it
        # always produced, and its moist-theta row is in the block above.
        lines.append(
            "# &dynamics use_theta_m is omitted in the namelist: read as "
            f"{use_theta_m} on WRF line {wrf_line_of(wrf_version)}, from "
            f"{wrf_version_source}.")
    lines += [
        "",
        "[experiment]",
        f'name = "{name}"',
        f"start_time = {start_time.isoformat(sep='T')}",
        f"run_seconds = {_fmt(run_seconds)}",
        f"feedback = {feedback}",
        f"smooth_option = {smooth_option}",
        f"blend_width = {blend_width}",
        f"spec_bdy_width = {spec_bdy_width}",
        *(["smooth_cg_topo = true"] if smooth_cg_topo else []),
        f"restart_interval_s = {_fmt(restart_interval_s)}",
        *([
            "# Declared-experiment acknowledgements, carried through the "
            "import (--ack);",
            "# a WRF namelist has no spelling for them.",
            "acknowledgements = ["
            + ", ".join(f'"{ack}"' for ack in acknowledgements) + "]",
        ] if acknowledgements else []),
        "",
        "[projection]",
        f'map_proj = "{projection["map_proj"]}"',
    ]
    for key in ("ref_lat", "ref_lon", "truelat1", "truelat2", "stand_lon"):
        lines.append(f"{key} = {_fmt(projection[key])}")
    lines += [
        "",
        "[shared]",
        f"# WRF e_vert = {nz + 1} staggered full levels -> nz = {nz} "
        "mass levels.",
        f"nz = {nz}",
        "# Nominal vertical scaffold height; the real path derives "
        "heights from",
        "# p_top/eta_levels.",
        "ztop = 20000.0",
        f"p_top = {_fmt(p_top)}",
    ]
    if eta_levels:
        lines.append("eta_levels = [")
        for i in range(0, len(eta_levels), 5):
            chunk = ", ".join(repr(v) for v in eta_levels[i:i + 5])
            lines.append(f"    {chunk},")
        lines.append("]")
    lines += [
        f"hybrid_opt = {hybrid_opt}",
        f"etac = {_fmt(etac)}",
        f"base_temp = {_fmt(base_temp)}",
        f"time_step_sound = {time_step_sound}",
        # [shared] carries d01; differing assigned values are emitted as
        # explicit [[domain]] overrides below, and an unassigned Registry
        # tail as the "auto" sentinel.
        (f"epssm = {_fmt(epssm[0])}" if epssm_assigned
         else f'epssm = "{EPSSM_AUTO}"'),
        "# WRF Registry defaults the namelist leaves unset, ratified "
        "binding for the",
        "# reference configuration (Phase-4 native-dt baseline): emdiv "
        "0.01,",
        "# hypsometric_opt 2, h_sca_adv_order 5.",
        f"emdiv = {_fmt(emdiv)}",
        f"hypsometric_opt = {hypsometric_opt}",
        "# Adaptive time step (Registry.EM_COMMON:2269-2281).  Off unless",
        "# the namelist asked for it; with it off every value below is",
        "# WRF's Registry default and nothing reads them.",
        f"use_adaptive_time_step = {_fmt(use_adaptive_time_step)}",
        f"step_to_output_time = {_fmt(step_to_output_time)}",
        f"adaptation_domain = {adaptation_domain}",
        f"target_cfl = {_fmt(target_cfl)}",
        f"target_hcfl = {_fmt(target_hcfl)}",
        f"max_step_increase_pct = {max_step_increase_pct}",
        f"starting_time_step = {starting_time_step}",
        f"starting_time_step_den = {starting_time_step_den}",
        f"max_time_step = {max_time_step}",
        f"max_time_step_den = {max_time_step_den}",
        f"min_time_step = {min_time_step}",
        f"min_time_step_den = {min_time_step_den}",
        *([
            "# The namelist's starting = max = min step is WRF's fixed "
            "step through",
            "# the adaptive clock, with the acoustic count WRF derives from "
            "it; the",
            "# engine runs that clock as written and records the terrain "
            "clock's",
            "# verdict as advice in the receipt.",
            f'terrain_clock = "{terrain_clock}"',
        ] if terrain_clock is not None and fixed_through_adaptive is not None
          else [
            "# The namelist's fixed step and acoustic count are the ones "
            "wrf.exe runs;",
            "# the engine runs them as written and records the terrain "
            "clock's verdict",
            "# as advice in the receipt.",
            f'terrain_clock = "{terrain_clock}"',
        ] if terrain_clock is not None else []),
        f"h_sca_adv_order = {h_sca_adv_order}",
        f"smdiv = {_fmt(smdiv)}",
        f"top_lid = {_fmt(top_lid)}",
        f"moist = {_fmt(moist_column[0] or any(value == 2 for value in fire_columns.get('ifire', ())))}",
        # WRF has no moist_cq namelist key: it derives calc_cq from the
        # moist species that exist, including passive vapor at mp=0.
        # physics_compat supplies the same enabled policy as the shipped
        # profiles; the acoustic driver bypasses states without vapor.
        f"moist_cq = {_fmt(bool(implicit['moist_cq']))}",
        f"mp_physics = {mp_physics}",
        f"morr_rimed_ice = {morr_rimed_ice}",
        f"moist_adv_opt = {moist_adv_opt}",
        f"km_opt = {km_opt}",
        f"diff_6th_opt = {diff_6th_opt}",
        f"diff_6th_slopeopt = {diff_6th_slopeopt}",
        f"w_damping = {w_damping}",
        # Emitted only when on, so a namelist without it imports to the
        # same bytes it always did.
        *((f"zadvect_implicit = {zadvect_implicit}",)
          if zadvect_implicit else ()),
        *((f"w_crit_cfl = {_fmt(w_crit_cfl)}",)
          if w_crit_cfl is not None else ()),
        f"damp_opt = {damp_opt}",
        f"zdamp = {_fmt(zdamp)}",
        f"dampcoef = {_fmt(dampcoef)}",
        # The pair is written whenever either selector leaves the engine
        # default (diff_opt 2, mix_full_fields true), so a namelist that
        # omits mix_full_fields under diff_opt 2 (WRF's default false, the
        # operational HRRR spelling) resolves to the TOML that runs it.
        *([f"diff_opt = {diff_opt_col[0]}"]
          if 1 in diff_opt_col else []),
        *([f"mix_full_fields = {_fmt(mix_full_resolved[0])}"]
          if 1 in diff_opt_col or not mix_full_resolved[0] else []),
        f"khdif = {_fmt(khdif)}",
        f"kvdif = {_fmt(kvdif)}",
    ]
    if wif_climatology_imported:
        # Emitted only when the triple was written, so every namelist that
        # imported before this lane still emits byte-identical TOML.
        lines += [
            "# &physics use_aero_icbc = .true. + &domains wif_input_opt = 1:",
            "# the ported QNWFA_QNIFA_SIGMA_MONTHLY.dat climatology ingest.",
            "# Leave wif_climatology_path unset to read the staged copy",
            "# (`woof fetch-tables --wif`, ~/.woof/wif).",
            "aer_init_opt = 1",
            f"wif_input_opt = {WIF_INPUT_OPT_CLIMATOLOGY}",
        ]
    if analyzed_with_wif_triple:
        lines += ["aer_init_opt = 1",
                  f"wif_input_opt = {WIF_INPUT_OPT_CLIMATOLOGY}",
                  "use_rap_aero_icbc = true"]
    elif analyzed_aerosol_imported:
        lines += ["use_rap_aero_icbc = true", "mp28_aerosol_source = \"analysis\""]
    if mp_physics == 28 and operational_thompson_defaults:
        lines += [
            "# The namelist carries keys only the operational WRF 3.9 fork",
            "# defines (" + ", ".join(operational_fork_keys) + "):",
            "# Thompson runs as that fork's generation (fork tables under",
            "# WOOF_THOMPSON_FORK_TABLE_ROOT or ~/.woof/tables/thompson-wrf39-noaa).",
        ]
        lines.extend(f"{key} = {_fmt(value)}"
                     for key, value in operational_thompson_defaults.items())
    if chem_import is not None:
        # Emitted only for an active &chem group, which until 2026-10-12
        # imported as "no chemistry"; a chem-off namelist is byte-identical.
        lines += chem_import.toml_lines()
        lines += [f"{key} = {_fmt(value)}"
                  for key, value in chem_dynamics.items()]
    # Supplied-only knobs (knob-parity lane): each maps 1:1 onto a
    # consumed RunConfig field whose default equals the WRF Registry
    # default, so absence emits nothing and the established imports stay
    # byte-identical.
    for _turb_key, _turb_col in turbulence_row:
        if _turb_col is not None:
            lines.append(f"{_turb_key} = {_fmt(_turb_col[0])}")
    if diff_6th_thresh is not None:
        lines.append(f"diff_6th_thresh = {_fmt(diff_6th_thresh)}")
    if moist_mix6_off_col is not None:
        lines.append(f"moist_mix6_off = {_fmt(moist_mix6_off_col[0])}")
    for key in ("diff_6th_form", "upper_wind_limiter_form"):
        # An explicit carried generation wins over the factor-presence
        # inference. Ordinary WRF imports emit neither selector.
        value = source_defaults.get(
            key, "noaa_wrf39" if diff_6th_factor2 is not None else None)
        if value is not None:
            lines.append(f"{key} = {_fmt(value)}")
    for key, value, default in (("mp_zero_out", mp_zero_out, 0),
                               ("mp_zero_out_thresh", mp_zero_out_thresh, 1e-8),
                               ("mp_zero_out_all", mp_zero_out_all, 0)):
        if value is not None and value != default:
            lines.append(f"{key} = {_fmt(value)}")
    if tke_budget_col is not None:
        lines.append(f"tke_budget = {tke_budget_col[0]}")
    for _order_key, _order_col in vertical_order_columns.items():
        if _order_col is not None:
            lines.append(f"{_order_key} = {_order_col[0]}")
    for _topo_key, _topo_col in topo_columns.items():
        lines.append(f"{_topo_key} = {_topo_col[0]}")
    for _drag_key, _drag_col in terrain_drag_columns.items():
        lines.append(f"{_drag_key} = {_drag_col[0]}")
    if shadlen is not None:
        lines.append(f"shadlen = {_fmt(shadlen)}")
    lines += [
        f"spec_zone = {spec_zone}",
        f"relax_zone = {relax_zone}",
        "terrain_opt = 1",
        f"map_proj = {WRF_MAP_PROJ_CODES[projection['map_proj']]}",
        f"sf_sfclay_physics = {sfclay}",
        f"sf_surface_physics = {sfsfc}",
        f"bl_pbl_physics = {bl_pbl_physics}",
        f"ra_physics = {ra_physics}",
    ]
    if ocean_selection_explicit and ocean_settings["sf_ocean_physics"] == 0:
        lines.append("sf_ocean_physics = 0")
    if any(windfarm_col):
        lines.append(f"windfarm_tke_factor = {_fmt(windfarm_tke_factor)}")
    if ocean_settings["sf_ocean_physics"]:
        lines.extend(f"{key} = {_fmt(value)}" for key, value in ocean_settings.items())
        for key, value in ocean_profile_settings.items():
            literal = '[' + ', '.join(_fmt(x) for x in value) + ']' if isinstance(value, list) else _fmt(value)
            lines.append(f'{key} = {literal}')
    for key, value in urban_settings.items():
        lines.append(f"{key} = {value}")
    if urban_settings["sf_urban_physics"] == 1 or water_values is not None:
        lines.append(f"urban_water_balance = {_fmt(water_generation)}")
    if urban_ucm_generation is not None:
        if urban_ucm_generation not in ("wrf_471", "wrf_480"):
            raise ValueError("urban_ucm_generation must be wrf_471 or wrf_480")
        lines.append(f'urban_ucm_generation = "{urban_ucm_generation}"')
    if urban_controls:
        admitted = {"urban_treeoption", "urban_ttscheme", "urban_fvg", "urban_table_dir"}
        if set(urban_controls) - admitted:
            raise ValueError(f"unknown urban controls {sorted(set(urban_controls)-admitted)}")
        for key,value in urban_controls.items():
            lines.append(f'{key} = {_toml_string(value) if isinstance(value,str) else _fmt(value)}')
    for key, column in stochastic_flags.items():
        if column[0]:
            lines.append(f"{key} = {column[0]}")
    if resolved_soil_layers != 4:
        lines.append(f"num_soil_layers = {resolved_soil_layers}")
    if sf_surface_mosaic == 1:
        lines += ["sf_surface_mosaic = 1", f"mosaic_cat = {mosaic_cat}"]
    if lake_column[0]:
        lines.append(f"sf_lake_physics = {lake_column[0]}")
    for key, default in (("use_lakedepth", 1), ("lakedepth_default", 50.0),
                         ("lake_min_elev", 5.0)):
        if lake_settings[key][0] != default:
            lines.append(f"{key} = {_fmt(lake_settings[key][0])}")
    for key, value in ruc_mosaic_settings.items():
        if value:
            lines.append(f"{key} = {value}")
    if operational_ruc_defaults:
        lines.append("# RUC lineage choices requested by the operational-fork namelist.")
        for key, value in operational_ruc_defaults.items():
            lines.append(f"{key} = {_fmt(value)}")
            defaults_applied.append(AppliedDefault(
                key=key, value=value,
                reason="the namelist requests the operational WRF 3.9 fork's RUC configuration"))
    if canopy_column is not None and canopy_column[0] != "dominant":
        lines.append(f"mosaic_urban_canopy = {_fmt(canopy_column[0])}")
    for key,value in aerosol_settings.items():
        lines.append(f"{key} = {_fmt(value)}")
    for key, value in (("no_mp_heating", no_mp_heating),
                       ("mp_tend_lim", mp_tend_lim),
                       ("ysu_topdown_pblmix", ysu_topdown_pblmix),
                       ("ysu_version", carried_selectors.get("ysu_version")),
                       ("ra_sw_eclipse", eclipse_sw[0]),
                       ("swint_opt", swint_opt),
                       ("aer_opt", aer_opt),
                       ("alb_sol", alb_sol),
                       ("bl_mynn_mixlength", bl_mynn_mixlength),
                       ("bl_mynn_tkeadvect", bl_mynn_tkeadvect),
                       ("scalar_pblmix", scalar_pblmix),
                       ("bl_mynn_mixscalars", bl_mynn_mixscalars),
                       ("bl_mynn_version", bl_mynn_version),
                       ("bl_mynn_cloud_tendency_form", carried_selectors.get("bl_mynn_cloud_tendency_form")),
                       ("isfflx", isfflx),
                       ("isftcflx", isftcflx),
                       ("iz0tlnd", iz0tlnd),
                       ("usemonalb", usemonalb),
                       ("rdlai2d", rdlai2d),
                       ("fractional_seaice", fractional_seaice),
                       ("seaice_threshold", seaice_threshold),
                       ("opt_thcnd", opt_thcnd),
                       ("o3input", o3input),
                       ("ghg_input", ghg_input),
                       ("use_mp_re", use_mp_re),
                       ("seaice_albedo_default",
                        seaice_albedo_default),
                       ("rdmaxalb", rdmaxalb),
                       ("nwp_diagnostics", nwp_diagnostics),
                       ("gsd_diagnostics", gsd_diagnostics if gsd_diagnostics else None),
                       ("cycling", cycling)):
        if value is not None:
            lines.append(f"{key} = {_fmt(value)}")
    for key, default in (("output_diagnostics", 0), ("diag_int", 15), ("wind_int", 5)):
        if climate_values["output_diagnostics"] or climate_values[key] != default:
            lines.append(f"{key} = {_fmt(climate_values[key])}")
    for key, column in precipitation_columns.items():
        # reset_interval1 is emitted only when the namelist moves it off
        # its 60 min Registry default, so an import that never names it
        # keeps the bytes it had before the field existed.
        if (any(value != 60 for value in column) if key == "reset_interval1"
                else any(column)):
            lines.append(f"{key} = {_fmt(column[0])}")
    for key, value in hail_values.items():
        lines.append(f"{key} = {_fmt(value)}")
    for key, value in lightning_values.items():
        lines.append(f"{key} = {_fmt(value)}")
    if wrf_rrtmg_compatibility != "none":
        lines.append(
            f'wrf_rrtmg_compatibility = "{wrf_rrtmg_compatibility}"')
    if legacy_rrtmg:
        lines.append(f'ra_rrtmg_variant = "{RRTMG_VARIANT_LEGACY}"')
    elif any_rrtmg and not coupled_legacy:
        # Preserve the selected mixed-spectrum implementation explicitly.
        # Compatibility tokens alter cloud optics and remain coupled-only.
        lines.append(f'ra_rrtmg_variant = "{RRTMG_VARIANT_RTE_RRTMGP}"')
    if mp_physics == 6:
        lines.append(f"wsm6_hail_opt = {wsm6_hail_opt}")
    for _nssl_key in (
            "nssl_2moment_on", "nssl_hail_on", "nssl_ccn_on",
            "nssl_density_on", "nssl_3moment"):
        if _nssl_key in nssl_selectors:
            lines.append(f"{_nssl_key} = {nssl_selectors[_nssl_key]}")
    if mp_physics == 16:
        # WRF's ONE `hail_opt` key drives wsm6 and wdm6 alike
        # (module_physics_init.F:4487 and :4582-4584), so the WDM6 field is
        # written from the same parsed scalar; no namelist can make the two
        # woof fields disagree.
        lines.append(f"wdm6_hail_opt = {wsm6_hail_opt}")
        lines.append(f"wdm6_ccn_conc = {_fmt(ccn_conc)}")
    if not coupled_legacy:
        lines += [
            f"ra_lw_physics = {ra_lw_physics}",
            f"ra_sw_physics = {ra_sw_physics}",
        ]
    if any_dudhia:
        lines += [
            f"icloud = {icloud}",
            f"swrad_scat = {_fmt(swrad_scat)}",
        ]
    elif icloud != 1:
        lines.append(f"icloud = {icloud}")
    lines.append(f"bldt = {_fmt(bldt)}")
    # The eleven numerics WRF declares max_domains that this importer read
    # as a tree-wide value.  Stated once, so the read side, the [shared]
    # emission and the per-domain tail cannot drift the way the split
    # between them did.  `None` means the namelist never supplied the key
    # (tke_budget and diff_6th_thresh are supplied-only).
    _per_domain_numerics = (
        ("diff_6th_slopeopt", diff_6th_slopeopt_col, str),
        ("diff_6th_thresh", diff_6th_thresh_col, _fmt),
        ("zdamp", zdamp_col, _fmt),
        ("dampcoef", dampcoef_col, _fmt),
        ("emdiv", emdiv_col, _fmt),
        ("smdiv", smdiv_col, _fmt),
        ("khdif", khdif_col, _fmt),
        ("kvdif", kvdif_col, _fmt),
        ("h_sca_adv_order", h_sca_adv_order_col, str),
        ("moist_adv_opt", moist_adv_opt_col, str),
        ("tke_budget", tke_budget_col, str),
        ("v_sca_adv_order", v_sca_adv_order_col, str),
        ("v_mom_adv_order", v_mom_adv_order_col, str),
    )
    for _fire_key, _fire_column in fire_columns.items():
        lines.append(f"{_fire_key} = {_fmt(_fire_column[0])}")
    for n in range(max_dom):
        is_root = parent_id[n] == 0
        lines += [
            "",
            "[[domain]]",
            f"# WRF e_we = {e_we[n]}, e_sn = {e_sn[n]} (staggered) -> "
            "mass dimensions.",
            f"grid_id = {grid_ids[n]}",
            f"parent_id = {parent_id[n]}",
            f"i_parent_start = {i_start[n]}",
            f"j_parent_start = {j_start[n]}",
            f"parent_grid_ratio = {ratio[n]}",
            f"parent_time_step_ratio = {tratio[n]}",
            f"nx = {e_we[n] - 1}",
            f"ny = {e_sn[n] - 1}",
        ]
        if moist_column[n] != moist_column[0]:
            lines.append(f"moist = {_fmt(moist_column[n])}")
        for key, column in stochastic_flags.items():
            if column[n] != column[0]:
                lines.append(f"{key} = {column[n]}")
        for _fire_key, _fire_column in fire_columns.items():
            if _fire_column[n] != _fire_column[0]:
                lines.append(f"{_fire_key} = {_fmt(_fire_column[n])}")
        if not terrain_smoothing.is_default:
            lines.append(static_inline(terrain_smoothing))
        if is_root:
            lines.append(f"time_step = {time_step}")
            if fract_num:
                lines.append(f"time_step_fract_num = {fract_num}")
                lines.append(f"time_step_fract_den = {fract_den}")
            lines.append(f"dx = {_fmt(root_dx)}")
            # The exponential sponge acts only on the specified (root)
            # branch (module_bc_em.F:1320); children carry spec_exp = 0
            # by the nested-branch rule.
            lines.append(f"spec_exp = {_fmt(spec_exp)}")
        lines += [
            f"specified = {_fmt(specified[n])}",
            f"nested = {_fmt(nested[n])}",
            f"history_interval_s = {_fmt(history_interval_s[n])}",
        ]
        if history_interval2_s is not None and history_interval2_s[n]:
            lines.append(f"history_interval2_s = {_fmt(history_interval2_s[n])}")
        if history_interval_change_s is not None and history_interval_change_s[n]:
            lines.append(f"history_interval_change_s = {_fmt(history_interval_change_s[n])}")
        # Emitted only where set, so every namelist without a begin or
        # end offset keeps producing a byte-identical TOML.
        if history_begin_s is not None and history_begin_s[n]:
            lines.append(f"history_begin_s = {_fmt(history_begin_s[n])}")
        if history_end_s is not None and history_end_s[n]:
            lines.append(f"history_end_s = {_fmt(history_end_s[n])}")
        if start_times[n] != start_time:
            lines.append(
                f"start_time = {start_times[n].isoformat(sep='T')}")
        if n >= epssm_assigned:
            # WRF's Registry default, not a choice: under a [shared]
            # number it is said here; under a [shared] "auto" inherited.
            if epssm_assigned:
                lines.append(f'epssm = "{EPSSM_AUTO}"')
        elif epssm[n] != epssm[0]:
            lines.append(f"epssm = {_fmt(epssm[n])}")
        # The turbulence column, emitted on exactly the same rule as
        # epssm: only where a domain differs from the root, so every
        # uniform namelist keeps producing a byte-identical TOML.
        if km_opt_col[n] != km_opt_col[0]:
            lines.append(f"km_opt = {km_opt_col[n]}")
        if diff_opt_col[n] != diff_opt_col[0]:
            lines.append(f"diff_opt = {diff_opt_col[n]}")
        if mix_full_resolved[n] != mix_full_resolved[0]:
            lines.append(f"mix_full_fields = {_fmt(mix_full_resolved[n])}")
        if bl_pbl_col[n] != bl_pbl_col[0]:
            lines.append(f"bl_pbl_physics = {bl_pbl_col[n]}")
        if windfarm_col[n]:
            lines.append("wind_farms = " + _toml_string(str(Path(input_path).resolve().parent / "windturbines.txt")))
        # The km_opt=2/3 turbulence parameter row, same rule: c_k on an
        # LES child beside its parents' default is the case that opened
        # this path (see the read side above).
        for _turb_key, _turb_col in turbulence_row:
            if _turb_col is not None and _turb_col[n] != _turb_col[0]:
                lines.append(f"{_turb_key} = {_fmt(_turb_col[n])}")
        if moist_mix6_off_col is not None \
                and moist_mix6_off_col[n] != moist_mix6_off_col[0]:
            lines.append(
                f"moist_mix6_off = {_fmt(moist_mix6_off_col[n])}")
        # The damping, divergence-damping, constant-K and filter columns,
        # emitted on epssm's rule: only where a domain differs from the
        # root, so a namelist whose columns are uniform -- which is every
        # namelist that imported before these were read as columns --
        # produces a byte-identical TOML.
        for _dom_key, _dom_col, _dom_fmt in _per_domain_numerics:
            if _dom_col is None:
                continue
            if _dom_col[n] != _dom_col[0]:
                lines.append(f"{_dom_key} = {_dom_fmt(_dom_col[n])}")
        for _topo_key, _topo_col in topo_columns.items():
            if _topo_col[n] != _topo_col[0]:
                lines.append(f"{_topo_key} = {_topo_col[n]}")
        if canopy_column is not None and canopy_column[n] != canopy_column[0]:
            lines.append(f"mosaic_urban_canopy = {_fmt(canopy_column[n])}")
        for _drag_key, _drag_col in terrain_drag_columns.items():
            if _drag_col[n] != _drag_col[0]:
                lines.append(f"{_drag_key} = {_drag_col[n]}")
        for key, column in precipitation_columns.items():
            if column[n] != column[0]:
                lines.append(f"{key} = {_fmt(column[n])}")
        # The adaptive clock's max_domains targets and clamps, same rule.
        for _clock_key, _, _ in ADAPTIVE_CLOCK_COLUMNS:
            _clock_col = adaptive_columns[_clock_key]
            if _clock_col[n] != _clock_col[0]:
                lines.append(f"{_clock_key} = {_fmt(_clock_col[n])}")
        # WRF declares LW/SW as max_domains arrays. Emit only differences
        # from the inherited root choice, preserving uniform documents.
        if (radiation_pairs[n], legacy_rrtmg_col[n]) != (
                radiation_pairs[0], legacy_rrtmg_col[0]):
            lw, sw = radiation_pairs[n]
            variant = (RRTMG_VARIANT_LEGACY if legacy_rrtmg_col[n]
                       else RRTMG_VARIANT_RTE_RRTMGP)
            lines += [
                "ra_physics = 0", f"ra_lw_physics = {lw}",
                f"ra_sw_physics = {sw}", f'ra_rrtmg_variant = "{variant}"',
                f'wrf_rrtmg_compatibility = "{compatibility_col[n]}"',
            ]
        for key, column in lake_settings.items():
            if column[n] != column[0]:
                lines.append(f"{key} = {_fmt(column[n])}")
        if eclipse_sw[n] != eclipse_sw[0]:
            lines.append(f"ra_sw_eclipse = {eclipse_sw[n]}")
        if radt[n] > 0.0:
            lines.append(f"radt = {_fmt(radt[n])}")
        else:
            # WRF radt = 0 -> radiation every step.  RunConfig's compat
            # `radt` key cannot express that (a positive radt overrides
            # radt_minutes, and radt = 0 falls back to the 12-minute
            # radt_minutes default), so emit radt_minutes = 0.0 (review
            # F3).
            lines.append("radt_minutes = 0.0")
        lines.append(f"cu_physics = {cu[n]}")
        if cu[n] == 1:
            lines.append(f"cudt_minutes = {_fmt(cudt[n])}")
        elif cu[n] == 3:
            # GF runs on the model step (STEPCU = 1, WRF's usual GF
            # configuration); RunConfig validation enforces the same.
            lines.append("cudt_minutes = 0.0")
            lines.append(f"clos_choice = {clos_choice}")
            lines.append(f"ishallow = {ishallow}")
            if cudt[n]:
                drop("physics", f"cudt[{n + 1}]", [cudt[n]],
                     "GF carries no cudt cadence: WRF's GF runs every "
                     "model step and so does woof's")
        elif cu[n] == 5:
            # Grell 3D runs on the model step (woof.config refuses
            # cudt_minutes != 0 on it by name: the driver adds its per-call
            # RAINCV once, where WRF re-adds the held PRATEC every step).
            lines.append("cudt_minutes = 0.0")
            lines.append(f"clos_choice = {clos_choice}")
            lines.append(f"ishallow = {ishallow}")
            lines.append(f"cugd_avedx = {cugd_avedx}")
            for key, value in g3_controls.items():
                lines.append(f"{key} = {value[n] if key == 'cu_diag' else value}")
            if cudt[n]:
                drop("physics", f"cudt[{n + 1}]", [cudt[n]],
                     "Grell 3D carries no cudt cadence in woof: the scheme "
                     "runs on the model step")
        elif cu[n] == 16:
            # New Tiedtke runs on the model step and carries no NCA hold,
            # and woof.config refuses cudt_minutes != 0 on it by name.
            # Stated rather than omitted: an omitted key inherits
            # RunConfig's 5.0, and this scheme's RAINCV is a per-call rate
            # with no persistence, so a five-minute hold would reapply it
            # every step.
            lines.append("cudt_minutes = 0.0")
            if cudt[n]:
                drop("physics", f"cudt[{n + 1}]", [cudt[n]],
                     "New Tiedtke carries no cudt cadence: the scheme runs "
                     "on the model step and its RAINCV is a per-call rate "
                     "with no NCA hold")
        elif cu[n] == 6:
            lines.append("cudt_minutes = 0.0")
            if cudt[n]:
                drop("physics", f"cudt[{n + 1}]", [cudt[n]],
                     "Original cu6 runs every model step and returns per-call rain without an NCA hold")
        elif cu[n] == 96:
            # New SAS honours cudt as WRF does (its rates are formed over
            # dt*stepcu and held between calls) -- except under an
            # adaptive step, where WRF resets cudt to 0 for every SAS
            # scheme and says so (module_check_a_mundo.F:2142-2160), and
            # woof.config refuses the pair for the same reason.
            if use_adaptive_time_step and cudt[n]:
                fix("physics", f"cudt[{n + 1}]", [cudt[n]], 0.0,
                    "use_adaptive_time_step requires cudt = 0 for every SAS "
                    "scheme; WRF resets it the same way "
                    "(module_check_a_mundo.F:2142-2160)")
                lines.append("cudt_minutes = 0.0")
            else:
                lines.append(f"cudt_minutes = {_fmt(cudt[n])}")
            lines.append(f"nsas_dx_factor = {nsas_dx_factor}")
            lines.append(f"sas_pgcon = {_fmt(sas_pgcon[n])}")
        elif cudt[n]:
            drop("physics", f"cudt[{n + 1}]", [cudt[n]],
                 "cudt is consumed only where cu_physics = 1 or 96")
        # An omitted cudt_minutes inherits RunConfig's live 5.0 while the
        # shipped physics profiles and the domain wizard write 0.0 for the
        # same cumulus-off suite.  Both spellings mean the same dead
        # switch -- woof/core/clock.py builds no cumulus calendar and
        # woof/core/physics.py takes no cumulus step at cu_physics = 0 --
        # so the two are reconciled where identities are COMPARED
        # (woof.ingest.prepared_cache.effective_prepared_domain_config),
        # not by writing a value into a receipt this importer has always
        # reproduced byte for byte.
        lines.append(f"diff_6th_factor = {_fmt(diff_6th_factor[n])}")
        if diff_6th_factor2 is not None:
            lines.append(
                f"diff_6th_factor2 = {_fmt(diff_6th_factor2[n])}")
    if stochastic_controls is not None:
        # WRF nens is a seed label. An imported stochastic run contains
        # one forecast unless its explicit engine member count is changed.
        lines += ["", "[ensemble]", "members = 1"]
        def stochastic_table(path, values):
            lines.extend(("", "[" + path + "]"))
            lines.extend(f"{key} = {_fmt(value)}" for key, value in values.items()
                         if not isinstance(value, dict))
            for key, value in values.items():
                if isinstance(value, dict):
                    stochastic_table(path + "." + key, value)
        stochastic_table("ensemble.stochastic", stochastic_controls)
    if static_landcover is not None:
        block_lines, notice = _static_landcover_block(
            static_landcover, cache_root=static_cache_root,
            legend_note=legend_note)
        lines += block_lines
        notices.append(notice)
    fdda_section = None
    if active_fdda:
        from woof.spectral_nudging.namelist import translate as translate_nudging
        from woof.toml_document import emit_experiment_toml
        fdda_section = _Section("fdda", inp["fdda"], str(input_path))
        try:
            nudging_table = translate_nudging(fdda_section, grid_ids, drop)
        except ValueError as error:
            raise _err("fdda", "grid_fdda", inp["fdda"].get("grid_fdda"), str(error)) from error
        lines.extend(("", emit_experiment_toml({"nudging": nudging_table})))
    text = "\n".join(lines) + "\n"

    text = with_physics_defaults_text(text, source_defaults)
    defaults_applied.extend(AppliedDefault(
        key=key, value=value,
        reason=("scheme generation explicitly carried by the namelist comment"
                if key in carried_selectors else
                f"scheme generation WRF {wrf_version} runs (from "
                f"{WRF_VERSION_SOURCE_DEFAULT if wrf_version_source is None else wrf_version_source})"
                if key in release_generations else
                "scheme generation declared by the named source namelist"))
        for key, value in source_defaults.items())

    # Validate the emitted TOML through the schema loader: every
    # section-A rule binds at import time.
    if static_landcover is None:
        build_experiment(
            tomllib.loads(text),
            source=f"import of {wps_path.name} + {input_path.name}")
    else:
        # The [static] table is the one-file case schema's, validated by
        # its owner (woof.static.highres_production.parse_static_table)
        # on the way past, exactly as every config file door does.
        from woof.experiment import build_experiment_from_config_tables
        build_experiment_from_config_tables(
            tomllib.loads(text),
            source=f"import of {wps_path.name} + {input_path.name}",
            base_dir=Path(wps_path).parent)
    # Every consumed key lands in exactly one report section: keys
    # recorded above as fixed or dropped are subtracted from each
    # section's consumption trace and the remainder is what produced the
    # TOML -- the Translated section.
    handled = ({(d.section, d.key) for d in dropped}
               | {(f.section, f.key) for f in fixed})
    translated: list[TranslatedKey] = []
    for section_obj in (share, geo, tc, dm, ph, dyn, bdy, noahmp, stoch, fire) + ((fdda_section,) if fdda_section is not None else ()):
        for key in section_obj.consumed:
            if (section_obj.name, key) not in handled:
                translated.append(TranslatedKey(section=section_obj.name,
                                                key=key))
    report = SubstitutionReport(
        substitutions=tuple(substitutions), dropped=tuple(dropped),
        defaults_applied=tuple(defaults_applied), fixed=tuple(fixed),
        translated=tuple(translated), notices=tuple(notices),
        model_choices=tuple(model_choices),
        namelist_defaults=tuple(namelist_defaults),
        wrf_line=wrf_line_of(wrf_version),
        wrf_line_source=(WRF_VERSION_SOURCE_DEFAULT
                         if wrf_version_source is None
                         else wrf_version_source))
    return text, report
