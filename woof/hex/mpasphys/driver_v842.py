"""Driver orchestration from pinned MPAS-A v8.4.2, without scheme arithmetic.

Authority: mpas_atmphys_driver.F:111-384, mpas_atmphys_manager.F:137-766,
mpas_atmphys_control.F:92-332, mpas_atmphys_init.F:83-481. The hooks own
numerical preparation, persistent MPAS pools and framework I/O. This module
never aggregates raw tendencies: physics_get_tend must run at the native RK
boundary, with separate PBL, convection, LW and SW carriers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace as _replace
from enum import StrEnum
from fractions import Fraction
import importlib.util
import re
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Protocol

from ..config import parse_fortran_namelist
from ..physics_suites import load_table

SCHEMA = "mpas-port.physics-driver-v842/v1"
_RELEASED_TENDENCIES = object()

#: The correct-mode switch (the correct-mode key of 2026-10-13): ``config_physics_mode`` in
#: &nhyd_model, "exact" (default, MPAS-A v8.4.2 word for word) or "correct".  An optional
#: ``config_physics_correct_skip`` lists finding ids to leave uncorrected inside correct mode (the
#: per-correction observation A/B).  Any other mode value is refused by name.
PHYSICS_MODES = ("exact", "correct")

#: Corrections this driver knows, by finding id (TRIAGE.md cluster 4, mpas-rad-pbl-lsm-seaice).  "driver"
#: corrections are applied here; "slots" lists the (slot, schemes) whose implementation must declare the id
#: when the correction is active, so a correct-mode run never silently executes an uncorrected scheme.  A
#: native-cpu slot is pristine MPAS v8.4.2 and declares none; a hex-cuda slot declares what its table
#: descriptor lists under "corrections".  When an active slot's implementation does not declare an active
#: finding, the run is not refused (coordinator ruling 2026-10-14, default correct): the finding is left out
#: of the run as a whole and listed as a NAMED GAP in the receipt and on the console.  "owners" names the
#: follow-up owner of each such gap (the lane that owns the slot implementation; a dict when it depends on the
#: scheme), from the 2026-10-13 radiation/PBL/LSM/sea-ice scheme review ("Scheme lanes still to do").
#: Follow-up owners of the findings that act outside the MPAS suite route (rows with "hex_route" below).
_ENGINE_OWNER = "woof engine with woof.correct_mode (engine 2.9.0 and later)"
_LBC_OWNER = "woof engine rw_mpas_lbc with --physics-mode (engine 2.9.0 and later)"
_IAU_OWNER = "hex IAU door (woof.hex.iau_v842 correct kernels wired into the forecast)"
CORRECTIONS_V842: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    # RP1: bl_mynn_post.F multiplies condensate by 1+sqv (not 1/(1-sqv)) and bl_mynn_pre.F divides snow by
    # 1+qs: with zero MYNN tendencies cloud water is lost every call (1.7 % per 900 s at dt 20 s).
    "RP1": {"slots": {"pbl": ("bl_mynn",)},
            "owners": {"pbl": "p12-bl-mynn"},
            "breakage": "MYNN removes cloud water, ice and snow on every call even with zero tendencies"},
    # RP3: mpas_atmphys_lsm_shared.F:42-52 feeds the blended skin temperature back as the ice temperature;
    # the surface layer's ice part uses the blend too (mpas_atmphys_driver_sfclayer.F:428).
    "RP3": {"slots": {"seaice": None, "surface_layer": None},
            "owners": {"seaice": "p18-seaice",
                       "surface_layer": {"sf_mynn": "p10-sf-mynn", "sf_monin_obukhov": "p9-sfclay-mm5",
                                         "sf_monin_obukhov_rev": "p9-sfclay-mm5"}},
            "breakage": "fractional sea-ice cells lose their ice temperature between calls (up to 29 K)"},
    # RP7: mynn_tendencies (bl_mynn_subroutines.F90:3102) takes the lowest layer's pressure thickness as the full
    # surface pressure minus the hydrostatic interface pressure; a near-surface non-hydrostatic deficit makes it
    # zero or negative and moisture_check (F90:4149-4150) divides by it (bl_mynn_v842.MYNN_TEXT_CORRECTIONS).
    "RP7": {"slots": {"pbl": ("bl_mynn",)},
            "owners": {"pbl": "p12-bl-mynn"},
            "breakage": "MYNN's lowest-layer pressure thickness can be zero, and 0/0 in its moisture check made the "
                        "vapour tendency NaN in one column; the dycore then refused the step with ~4,100 whole "
                        "columns non-finite (x1.40962 120 km plain hex, 2021-07-01 start at day 19.2 and 2020-01-01 "
                        "start at day 77.2)"},
    # RP4: refused at resolution, see resolve_namelist (native correct mode refuses it too).
    "RP4": {"slots": {}, "owners": {},
            "breakage": "config_mynn_tkeadvect resets TKE to the never-advected qke_adv every call"},
    # RP6: config_pbl_interval stretches dt_pbl but the surface/PBL schemes are still called every step.
    # Noah-MP steps by the model dt (mpas_atmphys_lsm_noahmpinit.F:289), so under the cadence its
    # implementation must step by dt_pbl and declare RP6; the other dt_pbl schemes already do.
    # Under the cadence the land-surface and sea-ice schemes must also receive the precipitation of every step
    # since their last call (WRF RAINBL, module_surface_driver.F); the driver does that for every
    # implementation with the surface precipitation window (accumulate/open/close glue, see begin_step).
    "RP6": {"slots": {"land_surface": ("sf_noahmp",)},
            "owners": {"land_surface": "p17-noahmp"},
            "breakage": "config_pbl_interval integrates the surface and PBL schemes interval/dt times too long"},
    # Findings that act OUTSIDE the MPAS suite route: the ArWen column route's engine corrections and the
    # IAU and boundary doors' findings.  One finding table for every route (integrator ruling 2026-10-14;
    # lane H1).  "slots" and "owners" stay empty because no MPAS physics slot implements them, so the suite
    # route never activates them; "hex_route" says where each one acts, which woof.hex.correct_mode reads.
    "WB4": {"slots": {}, "owners": {},
            "hex_route": {"route": "arwen-column", "slot": "convection", "scheme": "ntiedtke",
                          "lives": "engine (woof.correct_mode)", "owner": _ENGINE_OWNER,
                          "what": "New Tiedtke water tendencies coupled on dry mass (ntiedtke_dry_mass_coupling)"},
            "breakage": "New Tiedtke tendencies per unit of moist frame mass are added on dry mass: about 1 per "
                        "cent of convective precipitation is never taken out of the column"},
    "IN1": {"slots": {}, "owners": {},
            "hex_route": {"route": "iau", "slot": "iau", "lives": "hexcore.iau_v842", "owner": _IAU_OWNER,
                          "what": "the IAU theta_m conversion no longer re-counts the physics vapour tendency"},
            "breakage": "the IAU theta_m conversion counts the physics vapour tendency a second time"},
    "IN2": {"slots": {}, "owners": {},
            "hex_route": {"route": "iau", "slot": "iau", "lives": "hexcore.iau_v842", "owner": _IAU_OWNER,
                          "what": "exact chain rule for a density increment (theta_m and edge momentum)"},
            "breakage": "a density increment reaches theta_m and edge momentum without its chain-rule terms"},
    "IN3": {"slots": {}, "owners": {},
            "hex_route": {"route": "iau", "slot": "iau", "lives": "hexcore.iau_v842", "owner": _IAU_OWNER,
                          "what": "one IAU dose per window: weight = overlap of the step with the window, "
                                  "position carried through restarts"},
            "breakage": "a window that does not divide by the step, or a restart inside it, doses the increment "
                        "more or less than once"},
    "IN4": {"slots": {}, "owners": {},
            "hex_route": {"route": "lbc", "slot": "lateral_boundary", "lives": "engine (rw_mpas_lbc --physics-mode)",
                          "owner": _LBC_OWNER,
                          "what": "boundary edge winds extrapolated like the initial conditions"},
            "breakage": "boundary winds below and above the first-guess column disagree with the initial "
                        "conditions at the same valid time (up to 4.7 m/s at the lowest levels)"},
    "IN5": {"slots": {}, "owners": {},
            "hex_route": {"route": "lbc", "slot": "lateral_boundary", "lives": "engine (rw_mpas_lbc --physics-mode)",
                          "owner": _LBC_OWNER,
                          "what": "every boundary file must carry its own valid time"},
            "breakage": "a first-guess file of another valid time is accepted for a boundary time"},
    # WB3 (cluster mpas-water-budget): precip_to_MPAS divides rho_p by (1+qv), but microphysics_from_MPAS
    # already made rho_p the DRY density (zz*rho_zz), so precipitable water is 0.6 % low on average.  Every
    # microphysics implementation computes the diagnostic, so every one must declare it.
    # WB5 (cluster mpas-water-budget): WSM6's rain-out books the below-ground part of the straddling cell at
    # the flat cell mean (mp_wsm6.F90:1984, :2254) while the remap moves it with a sloped profile; the
    # surface accounting then gains or loses water with the slope's sign.
    "WB5": {"slots": {"microphysics": ("mp_wsm6",)},
            "owners": {"microphysics": "p2-wsm6"},
            # WB5 also acts on the ArWen column route, in the engine (one row, lane H1).
            "hex_route": {"route": "arwen-column", "slot": "microphysics", "scheme": "wsm6",
                          "lives": "engine (woof.correct_mode)", "owner": _ENGINE_OWNER,
                          "what": "WSM6 below-ground partial cell booked with the remap's own profile "
                                  "(wsm6_conservative_surface)"},
            "breakage": "WSM6 books the below-ground part of the falling cell at the flat mean while the remap "
                        "moves it with a sloped profile, so column plus surface water is not conserved"},
    "WB3": {"slots": {"microphysics": None},
            "owners": {"microphysics": {"mp_wsm6": "p2-wsm6", "mp_thompson": "p3-thompson",
                                        "mp_thompson_aerosols": "p4-thompson-aerosols", "mp_kessler": "p1-kessler"}},
            "breakage": "precipitable water divides the dry density by (1+qv) a second time (0.6 % low, up to 1.3 %)"},
})
#: The reason every slot gap carries, and what the run does instead.
GAP_REASON = "slot does not implement it"
NATIVE_CPU_GAP_NOTE = ("native-cpu slots run pristine MPAS v8.4.2 and can never implement a correction; "
                       "a hex-cuda implementation that declares the finding closes this gap")
GAP_RUNS = "native MPAS v8.4.2 behaviour for this finding (as exact mode)"
#: RP6 under a native-file restart: the surface precipitation window lives on the device and the pristine
#: Registry restart stream cannot hold it (see runtime_v842.run_native_dynamics).  Owner from REPORT.md section 3.
RP6_RESTART_GAP = {"slot": "restart", "scheme": None, "backend": None,
                   "reason": "the port's native-file restart cannot hold the surface precipitation window",
                   "owner": "MPAS integrator (with f3's checkpoint provider)"}
#: RP6 (correct mode with config_pbl_interval): driver glue that hands the land-surface and sea-ice schemes the
#: precipitation added up since their last call, as the native correct-mode build does with rainncv_bl, raincv_bl,
#: snowncv_bl, graupelncv_bl and sr_bl (mpas_atmphys_lsm_shared.F precip_window_add/precip_window_reset).
#: accumulate runs every step before the surface group; open puts the window totals in rainncv, raincv,
#: snowncv, graupelncv and sr just before the land-surface call and close puts the step's own values back and
#: empties the window just after the sea-ice call, so the pristine slots read exactly what native correct mode
#: reads and every later reader sees the step's own precipitation.
SURFACE_PRECIPITATION_WINDOW = ("accumulate_surface_precipitation", "open_surface_precipitation_window",
                                "close_surface_precipitation_window")
_INTERVAL = re.compile(r"^(?:(\d+)_)?(\d+):(\d+):(\d+)(?:\.(\d+))?$")


class Backend(StrEnum):
    OFF = "off"
    NATIVE_CPU = "native-cpu"
    HEX_CUDA = "hex-cuda"


#: The MPAS suite routes run every slot on its GPU port by default (the table row's ``default_backend``,
#: hex-cuda for the v8.4.2 suites).  THE BREAKAGE THIS PREVENTS: with native-cpu as the route default, a run
#: that selected an MPAS suite ran MPAS's own Fortran on the CPU in every slot unless each slot was moved to
#: hex-cuda by hand, although every v8.4.2 scheme but CAM radiation has a GPU port proven bitwise against
#: native MPAS.  A slot whose scheme this install cannot run on the GPU falls back to native-cpu, and the run
#: names it at start-up (lines beginning ``PHYSICS ON CPU:``) and in its receipt (``cpu_slots``), never
#: silently.  native-cpu stays selectable per slot (config_*_backend) or for every slot
#: (``--physics-default-backend native-cpu``): the native referee.
CPU_WARNING_PREFIX = "PHYSICS ON CPU:"


@dataclass(frozen=True)
class CpuSlot:
    """An active slot that runs MPAS's own Fortran on the CPU in this run, and why."""
    slot: str
    scheme: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"slot": self.slot, "scheme": self.scheme, "backend": str(Backend.NATIVE_CPU),
                "reason": self.reason}


def gpu_port_absence(row_key: str, slot: str, scheme: str) -> str | None:
    """None when this install runs ``scheme`` in ``slot`` on its GPU port by default, else why it cannot."""
    row = load_table()[row_key]["slots"][slot]
    entry = row.get("cuda_entries", {}).get(scheme)
    if entry is None:
        return f"this release has no GPU slot for {scheme}"
    name = f"woof.hex.mpasphys.{entry['module']}"
    try:
        present = importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        present = False
    if not present:
        return f"its GPU port ({name}) is not in this distribution"
    if entry.get("resources"):
        from .resources_build import NOT_BUILT
        if scheme in NOT_BUILT:
            return (f"`woof hex physics-resources build` does not build its GPU packet ({NOT_BUILT[scheme]}); "
                    f"{row['backend_selector']}='hex-cuda' with a --physics-resources packet that carries "
                    f"{scheme} runs it on the GPU")
    return None


def default_slot_backends(row_key: str, selectors: Mapping[str, str],
                          default: str) -> tuple[dict[str, Backend], dict[str, str]]:
    """Each slot's backend before any explicit selection, and why each slot this leaves on native-cpu is.

    ``default`` hex-cuda puts every slot on its GPU port and falls back to native-cpu only where this install
    has none (:func:`gpu_port_absence`); native-cpu or off applies to every slot, as before."""
    default = Backend(default)
    chosen, reasons = {}, {}
    for slot, scheme in selectors.items():
        absent = (gpu_port_absence(row_key, slot, scheme)
                  if default == Backend.HEX_CUDA and scheme != "off" else None)
        chosen[slot] = Backend.NATIVE_CPU if absent else default
        if absent:
            reasons[slot] = absent
        elif default == Backend.NATIVE_CPU:
            reasons[slot] = "this run's default MPAS physics backend is native-cpu"
    return chosen, reasons


def cpu_slot_record(config: Any, row_key: str, reasons: Mapping[str, str],
                    explicit: Iterable[str]) -> tuple[CpuSlot, ...]:
    """Every active slot of ``config`` that runs on the CPU, with the reason (explicit, default or no port)."""
    rows = load_table()[row_key]["slots"]
    explicit = set(explicit)
    result = []
    for slot, backend in config.backends.items():
        if backend != Backend.NATIVE_CPU or not config.active(slot):
            continue
        reason = (f"selected with {rows[slot]['backend_selector']}='native-cpu'" if slot in explicit
                  else reasons.get(slot, "this run's default MPAS physics backend is native-cpu"))
        result.append(CpuSlot(slot, config.selectors[slot], reason))
    return tuple(result)


def interval_seconds(value: str) -> Fraction | None:
    """Fixed MPAS interval; mpas_timekeeping.F:1347-1591.

    Calendar-month intervals need the framework calendar and are rejected here
    because replacing a month by fixed seconds changes the next ring date.
    """
    value = str(value).strip()
    if value == "none":
        return None
    match = _INTERVAL.fullmatch(value)
    if match is None:
        raise ValueError(f"physics interval {value!r} requires DD_HH:MM:SS[.fraction]")
    days, hours, minutes, seconds, fraction = match.groups()
    result = Fraction(int(days or 0) * 86400 + int(hours) * 3600
                      + int(minutes) * 60 + int(seconds))
    if fraction:
        result += Fraction(int(fraction), 10 ** len(fraction))
    if result <= 0:
        raise ValueError("a recurring physics alarm requires a positive interval")
    return result


@dataclass(frozen=True)
class CorrectionGap:
    """A correction correct mode requested but this run cannot apply: a named gap, never a refusal."""
    finding: str
    slot: str
    scheme: str | None
    backend: str | None
    reason: str
    owner: str
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        # finding / slot / reason / owner are the names downstream readers key on (integrator, 2026-10-14);
        # the rest is detail.  owner is the follow-up owner of the gap.
        result = {"finding": self.finding, "slot": self.slot, "reason": self.reason, "owner": self.owner,
                  "scheme": self.scheme, "backend": self.backend, "runs": GAP_RUNS,
                  "breakage": CORRECTIONS_V842[self.finding]["breakage"]}
        if self.note:
            result["note"] = self.note
        return result


@dataclass(frozen=True)
class DriverConfig:
    suite: str
    selectors: Mapping[str, str]
    options: Mapping[str, Any]
    backends: Mapping[str, Backend]
    #: "exact" or "correct" (PHYSICS_MODES); corrections are the finding ids APPLIED in this run.
    mode: str = "exact"
    corrections: frozenset[str] = frozenset()
    correct_skip: frozenset[str] = frozenset()
    #: Correct mode only: active findings this run does not apply, each a named gap (see CORRECTIONS_V842).
    gaps: tuple[CorrectionGap, ...] = ()
    service_backends: Mapping[str, Backend] = field(default_factory=dict)
    #: The active slots that run on the CPU in this run (a route resolution records them; a library
    #: resolution for module replays records none).
    cpu_slots: tuple[CpuSlot, ...] = ()
    # Engine extension; never insert these keys into the native Registry.
    tc_surface: Mapping[str, str] = field(default_factory=dict)
    #: --ocean-mode pwp on this route (woof.hex.mpasphys.ocean_pwp_v842): the PWP column table the OML slot runs.
    ocean_pwp: Mapping[str, Any] = field(default_factory=dict)

    def cpu_warning(self) -> str | None:
        """The loud start-up lines naming each active slot that runs on the CPU, or None when none does."""
        if not self.cpu_slots:
            return None
        lines = [f"{CPU_WARNING_PREFIX} {len(self.cpu_slots)} active MPAS physics slot(s) run MPAS's own "
                 f"Fortran on the CPU in this run, not on the GPU:"]
        for item in self.cpu_slots:
            lines.append(f"{CPU_WARNING_PREFIX}   {item.slot}={item.scheme}: {item.reason}")
        gpu = [slot for slot, backend in self.backends.items()
               if backend == Backend.HEX_CUDA and self.active(slot)]
        lines.append(f"{CPU_WARNING_PREFIX}   on the GPU (hex-cuda): {', '.join(gpu) or 'no active slot'}")
        return "\n".join(lines)

    def correct(self, finding: str) -> bool:
        """True when correct mode applies this finding's correction in this run."""
        return finding in self.corrections

    def correct_mode_record(self) -> dict[str, Any]:
        """The run record's ``correct_mode`` block: mode, applied findings and named gaps."""
        return {"mode": self.mode, "applied": sorted(self.corrections), "skipped": sorted(self.correct_skip),
                "gaps": [gap.as_dict() for gap in self.gaps]}

    def gap_warning(self) -> str | None:
        """The loud multi-line startup warning naming each gap, or None when there is none.

        It begins ``CORRECT MODE: named gaps`` (the spelling downstream readers match, integrator 2026-10-14)
        and every line carries the ``CORRECT MODE:`` prefix."""
        if self.mode == "exact" or not self.gaps:
            return None
        lines = [f"CORRECT MODE: named gaps ({len(self.gaps)}): WARNING, these corrections are NOT applied "
                 f"in this run, which runs {GAP_RUNS} instead:"]
        for gap in self.gaps:
            where = gap.slot if gap.scheme is None else f"{gap.slot}/{gap.scheme} ({gap.backend})"
            lines.append(f"CORRECT MODE:   {gap.finding} on {where}: {gap.reason}; owner {gap.owner}. "
                         f"Native defect left in: {CORRECTIONS_V842[gap.finding]['breakage']}.")
        lines.append(f"CORRECT MODE:   applied corrections: {', '.join(sorted(self.corrections)) or 'none'}; "
                     f"--physics-mode exact runs native MPAS v8.4.2 bit for bit.")
        return "\n".join(lines)

    # Active native selector; mpas_atmphys_control.F:310-319.
    def active(self, slot: str) -> bool:
        row = load_table()["mpasphys_v842"]["slots"][slot]
        parent = row.get("derived_from")
        return (self.selectors[slot] != "off" and self.backends[slot] != Backend.OFF
                and (parent is None or self.active(parent))
                and all(self.selectors[key] == value and self.active(key)
                        for key, value in row.get("requires_selection", {}).items()))

    def native_embedded(self, slot: str) -> bool:
        """One native parent call contains NGW; mpas_atmphys_driver_gwdo.F:692-834."""
        parent = load_table()["mpasphys_v842"]["slots"][slot].get("embedded_in")
        return (parent is not None and self.active(slot) and self.active(parent)
                and self.backends[slot] == self.backends[parent] == Backend.NATIVE_CPU)

    @property
    def execution_options(self) -> Mapping[str, Any]:
        """Mask disabled backends at native glue; mpas_atmphys_control.F:310-319."""
        result = dict(self.options)
        for slot, row in load_table()["mpasphys_v842"]["slots"].items():
            if self.backends[slot] == Backend.OFF and row["selector"] is not None:
                result[row["selector"]] = row.get("option_values", {}).get("off", "off")
        return MappingProxyType(result)

    @property
    def moist_physics(self) -> bool:
        """mpas_atmphys_control.F:310-319; excludes cloudiness and GWDO."""
        return any(self.active(s) for s in (
            "microphysics", "convection", "land_surface", "pbl",
            "radiation_lw", "radiation_sw", "surface_layer"))

    @property
    def driver_physics(self) -> bool:
        """Pre-dynamics driver gate; mpas_atmphys_driver.F:179-184."""
        return any(self.active(s) for s in (
            "convection", "land_surface", "pbl", "radiation_lw",
            "radiation_sw", "surface_layer"))

    # Orchestration receipt of native selectors; mpas_atmphys_control.F:92-332.
    def receipt(self) -> dict[str, Any]:
        result = {"schema": SCHEMA, "suite": self.suite,
                  "selectors": dict(self.selectors), "options": dict(self.options),
                  "backends": {s: str(b) for s, b in self.backends.items()}}
        # Service backends (the surface-update glue, i5) appear only when one is moved off its native-cpu
        # default, so default and exact receipts stay word for word what they were (integrator, 2026-10-14).
        if any(b != Backend.NATIVE_CPU for b in self.service_backends.values()):
            result["service_backends"] = {s: str(b) for s, b in self.service_backends.items()}
        if self.cpu_slots:
            result["cpu_slots"] = [item.as_dict() for item in self.cpu_slots]
        if self.tc_surface:
            result["tc_surface"] = dict(self.tc_surface)
        if self.ocean_pwp:
            result["ocean_pwp"] = dict(self.ocean_pwp)
        # Exact-mode receipts stay word for word what they were before the switch existed.
        if self.mode != "exact":
            result.update(physics_mode=self.mode, corrections=sorted(self.corrections),
                          correct_skip=sorted(self.correct_skip), correct_mode=self.correct_mode_record())
        return result


def _physics_mode(groups: Mapping[str, Mapping[str, Any]], physics_mode: str | None,
                  correct_skip: Iterable[str] | None) -> tuple[str, frozenset[str]]:
    """config_physics_mode / config_physics_correct_skip from &nhyd_model, or the caller's values."""
    supplied = {k.casefold(): v for k, v in groups.get("nhyd_model", {}).items()}
    mode = supplied.get("config_physics_mode", None)
    if physics_mode is not None:
        if mode is not None and str(mode).strip() != physics_mode:
            raise ValueError("conflicting config_physics_mode in the namelist and the caller")
        mode = physics_mode
    mode = "exact" if mode is None else str(mode).strip()
    if mode not in PHYSICS_MODES:
        raise ValueError(f"config_physics_mode must be 'exact' or 'correct', not {mode!r}")
    skip = supplied.get("config_physics_correct_skip", "")
    names = {s.strip() for s in str(skip).split(",") if s.strip()} if skip else set()
    names |= {s.strip() for s in (correct_skip or ())}
    unknown = sorted(names - set(CORRECTIONS_V842))
    if unknown:
        raise ValueError(f"config_physics_correct_skip names unknown findings {unknown}; "
                         f"known: {sorted(CORRECTIONS_V842)}")
    if names and mode != "correct":
        raise ValueError("config_physics_correct_skip applies only with config_physics_mode='correct'")
    return mode, frozenset(names)


def _active_corrections(selectors: Mapping[str, str], options: Mapping[str, Any],
                        backends: Mapping[str, Backend], skip: frozenset[str]) -> frozenset[str]:
    """Finding ids whose correction changes this configuration (correct mode only)."""
    def on(slot: str) -> bool:
        return selectors[slot] != "off" and backends[slot] != Backend.OFF
    active = set()
    if on("pbl") and selectors["pbl"] == "bl_mynn":
        active.add("RP1")
        active.add("RP7")
    if on("seaice") and on("land_surface") and options.get("config_frac_seaice", False):
        active.add("RP3")
    if on("pbl") and selectors["pbl"] == "bl_mynn" and options.get("config_mynn_tkeadvect", False):
        active.add("RP4")
    if interval_seconds(options.get("config_pbl_interval", "none")) is not None and (
            on("pbl") or on("surface_layer") or on("land_surface")):
        active.add("RP6")
    if on("microphysics"):
        active.add("WB3")
    if on("microphysics") and selectors["microphysics"] == "mp_wsm6":
        active.add("WB5")
    return frozenset(active - skip)


def correct_source(source: str, substitutions: Mapping[str, Iterable[tuple[str, str, int]]],
                   corrections: Iterable[str], where: str) -> str:
    """A slot's CUDA text for the run: exact text unchanged, each applied finding's lines substituted.

    ``substitutions`` maps a finding id to (exact text, corrected text, expected count) rows.  Exact mode
    passes no corrections and compiles the original text byte for byte.  Breakage the count check
    prevents: a regenerated kernel source that no longer contains the exact text would silently drop the
    fix while the receipt still names it applied."""
    for finding in sorted(set(corrections) & set(substitutions)):
        for old, new, count in substitutions[finding]:
            found = source.count(old)
            if found != count:
                raise RuntimeError(f"{where}: the correct-mode text for {finding} occurs {found} times, not "
                                   f"{count}; the source changed and the correction would be dropped")
            source = source.replace(old, new)
    return source


def slot_corrections(call: Any, substitutions: Mapping[str, Any]) -> tuple[str, ...]:
    """The applied findings a slot implementation has text for (empty in exact mode)."""
    return tuple(sorted(set(getattr(call, "corrections", ())) & set(substitutions)))


#: What a slot implementation declares it applies: (slot, scheme, backend) -> finding ids.
Declarations = Callable[[str, str, Backend], Iterable[str]]


def table_declarations(slot: str, scheme: str, backend: Backend) -> frozenset[str]:
    """The finding ids the published implementation of a slot applies (the slot-declaration contract).

    native-cpu is pristine MPAS v8.4.2 and declares none; a hex-cuda slot declares its table descriptor's
    ``corrections`` (device_pools_v842.register_cuda_slots registers exactly these)."""
    if Backend(backend) != Backend.HEX_CUDA:
        return frozenset()
    entry = load_table()["mpasphys_v842"]["slots"][slot].get("cuda_entries", {}).get(scheme) or {}
    return frozenset(entry.get("corrections", ()))


def _gap_owner(finding: str, slot: str, scheme: str | None) -> str:
    owner = CORRECTIONS_V842[finding].get("owners", {}).get(slot)
    if isinstance(owner, Mapping):
        owner = owner.get(scheme)
    return owner or "MPAS integrator (no lane named for this slot)"


def settle_corrections(config: DriverConfig, requested: Iterable[str],
                       declarations: Declarations = table_declarations,
                       restart: bool = False) -> tuple[frozenset[str], tuple[CorrectionGap, ...]]:
    """Split the requested findings into those this run applies and its named gaps.

    A finding is applied only when every active slot it lives in has an implementation that declares it;
    otherwise none of it is applied (no half-corrected finding) and each slot lacking it is a named gap.
    Coordinator ruling 2026-10-14: a gap is listed in the receipt and printed loudly, never refused."""
    applied, gaps = set(), []
    for finding in sorted(requested):
        missing = []
        for slot, schemes in CORRECTIONS_V842[finding]["slots"].items():
            scheme = config.selectors[slot]
            if not config.active(slot) or (schemes is not None and scheme not in schemes):
                continue
            backend = config.backends[slot]
            if finding not in frozenset(declarations(slot, scheme, backend)):
                missing.append(CorrectionGap(
                    finding, slot, scheme, str(backend), GAP_REASON, _gap_owner(finding, slot, scheme),
                    NATIVE_CPU_GAP_NOTE if backend == Backend.NATIVE_CPU else None))
        if finding == "RP6" and restart and config.active("land_surface"):
            # RP6 under a native-file restart (runtime_v842 keeps the refusal as its backstop).
            missing.append(CorrectionGap("RP6", **RP6_RESTART_GAP))
        if missing:
            gaps.extend(missing)
        else:
            applied.add(finding)
    return frozenset(applied), tuple(gaps)


def resolve_namelist(text: str, *, backends: Mapping[str, str] | None = None,
                     default_backend: str | None = None, physics_mode: str | None = None,
                     correct_skip: Iterable[str] | None = None,
                     declarations: Declarations = table_declarations) -> DriverConfig:
    """MPAS control resolution; mpas_atmphys_control.F:92-332.

    Only the native suite route uses this resolver. WOOF's resolver and its
    default remain separate. All physics Registry values travel to native hooks.

    ``default_backend`` None is the library default for module replays: every slot native-cpu and no
    ``cpu_slots`` record, as before.  The routes pass their table default (hex-cuda: each slot on its GPU
    port, :func:`default_slot_backends`), and every active slot left on the CPU is named in ``cpu_slots``.
    """
    table = load_table()
    spec = table["mpasphys_v842"]
    options = {k: r["value"] for k, r in spec["registry_defaults"].items()}
    for key, row in spec["registry_defaults"].items():
        if row["type"] == "real":
            options[key] = float(options[key])
    groups = parse_fortran_namelist(text)
    supplied = groups.get("physics", {})
    for name, value in supplied.items():
        key = name.casefold()
        if key not in options:
            raise ValueError(f"MPAS v8.4.2 Registry has no physics option {name!r}")
        kind = spec["registry_defaults"][key]["type"]
        valid = ((kind == "character" and isinstance(value, str))
                 or (kind == "logical" and isinstance(value, bool))
                 or (kind == "integer" and type(value) is int)
                 or (kind == "real" and type(value) in (int, float)))
        if not valid:
            raise ValueError(f"{name} requires MPAS Registry type {kind}")
        options[key] = float(value) if kind == "real" else value
    suite = options["config_physics_suite"].strip()
    if suite not in spec["suite_routes"]:
        raise ValueError(f"{suite!r} has no MPAS v8.4.2 driver suite route")
    suite_values = table["mpas_suites"]["suites"][suite]
    selectors = {}
    for slot, row in spec["slots"].items():
        key = row["selector"]
        selected = selectors[row["derived_from"]] if key is None else options[key]
        if "option_values" in row:
            selected = next((name for name, value in row["option_values"].items()
                             if type(value) is type(selected) and value == selected), None)
        else:
            selected = selected.strip()
        if selected == "suite":
            selected = suite_values[key]
        if selected not in row["schemes"]:
            raise ValueError(f"illegal MPAS v8.4.2 {key}={selected!r}")
        selectors[slot] = selected
    # Native resolves these AFTER validating the individual selectors.
    if (selectors["radiation_lw"] != "off" or selectors["radiation_sw"] != "off") \
            and selectors["cloudiness"] == "off":
        selectors["cloudiness"] = "cld_incidence"
    if selectors["pbl"] == "bl_mynn":
        selectors["surface_layer"] = "sf_mynn"
    if selectors["pbl"] == "bl_ysu" and selectors["surface_layer"] not in (
            "sf_monin_obukhov", "sf_monin_obukhov_rev"):
        raise ValueError("YSU PBL requires an MPAS Monin-Obukhov surface layer")
    if selectors["land_surface"] != "off" and selectors["surface_layer"] == "off":
        raise ValueError("MPAS land surface requires a surface layer")
    for slot, row in spec["slots"].items():
        if row["selector"] is not None:
            options[row["selector"]] = row.get("option_values", {}).get(
                selectors[slot], selectors[slot])
    library = default_backend is None
    chosen, cpu_reasons = default_slot_backends(
        "mpasphys_v842", selectors, Backend.NATIVE_CPU if library else default_backend)
    explicit = set()
    backend_names = {r["backend_selector"]: s for s, r in spec["slots"].items()}
    services = spec.get("services", {})
    chosen_services = {name: Backend(row["default_backend"])
                       for name, row in services.items()}
    # A GPU route that updates SST, sea ice, greenness and deep soil from a surface series takes the
    # surface-update service on the GPU too. Left on its native-cpu table default, config_sst_update
    # made framework_only false, so the route asked for the native MPAS library and Fortran referee
    # resources a user install does not carry (hex-hybrid-longrange 10-day run, 2026-10-26).
    # An explicit config_surface_update_backend still wins; module replays (library) keep native-cpu.
    if (not library and options.get("config_sst_update") and "surface_update" in services
            and str(default_backend) in services["surface_update"]["backends"]):
        chosen_services["surface_update"] = Backend(str(default_backend))
    service_names = {row["backend_selector"]: name for name, row in services.items()}
    supplied_backends = groups.get(spec["backend_namelist_group"], {})
    for key, value in supplied_backends.items():
        if key in service_names:
            name = service_names[key]
            if value not in services[name]["backends"]:
                raise ValueError(f"unsupported {name} service backend {value!r}")
            chosen_services[name] = Backend(value)
            continue
        if key not in backend_names:
            raise ValueError(f"unknown MPAS slot backend option {key!r}")
        chosen[backend_names[key]] = Backend(value)
        explicit.add(backend_names[key])
    for slot, backend in (backends or {}).items():
        if slot not in selectors:
            raise ValueError(f"unknown physics backend slot {slot!r}")
        name = spec["slots"][slot]["backend_selector"]
        if name in supplied_backends and chosen[slot] != Backend(backend):
            raise ValueError(f"conflicting MPAS slot backend selections for {slot}")
        chosen[slot] = Backend(backend)
        explicit.add(slot)

    def record(config: DriverConfig) -> DriverConfig:
        if library:
            return config
        return _replace(config, cpu_slots=cpu_slot_record(config, "mpasphys_v842", cpu_reasons, explicit))

    mode, skip = _physics_mode(groups, physics_mode, correct_skip)
    if mode == "exact":
        return record(DriverConfig(suite, MappingProxyType(selectors),
                                   MappingProxyType(options), MappingProxyType(chosen),
                                   service_backends=MappingProxyType(chosen_services)))
    requested = _active_corrections(selectors, options, chosen, skip)
    if "RP4" in requested:
        # RP4 (correct mode). MPAS keeps qke_adv in diag_physics (Registry.xml:2806-2812) and its dynamics
        # never transports it, so bl_mynn.F90:766-770 resets qke to that stale field on every call.  This is
        # the correction itself (the native correct-mode referee stops the same way), not an unimplemented one.
        raise ValueError("config_mynn_tkeadvect=true is refused in correct mode (RP4): MPAS does not "
                         "transport qke_adv, so MYNN would reset TKE to a stale field on every call; "
                         "set config_mynn_tkeadvect=false or skip RP4 with config_physics_correct_skip")
    base = DriverConfig(suite, MappingProxyType(selectors), MappingProxyType(options), MappingProxyType(chosen))
    restart = {k.casefold(): v for k, v in groups.get("restart", {}).items()}.get("config_do_restart") is True
    corrections, gaps = settle_corrections(base, requested, declarations, restart=restart)
    return record(DriverConfig(suite, MappingProxyType(selectors), MappingProxyType(options),
                               MappingProxyType(chosen), mode, corrections, skip, gaps,
                               service_backends=MappingProxyType(chosen_services)))


def route_namelist(text: str, *, physics_mode: str | None = None,
                   correct_skip: Iterable[str] | None = None,
                   default_backend: str | None = None) -> DriverConfig | None:
    """Explicit native-suite door route; mpas_atmphys_control.F:92-332.

    The table selects the backend row. A missing suite retains the existing
    WOOF default. Returning None means the caller uses its existing resolver;
    slot backend overrides require an explicitly selected native suite.
    ``default_backend`` (``--physics-default-backend``) replaces the row's default for every slot that
    names no backend of its own.
    """
    groups = parse_fortran_namelist(text)
    spec = load_table()["mpasphys_v842"]
    suite = groups.get("physics", {}).get("config_physics_suite")
    if isinstance(suite, str) and suite in spec["suite_routes"]:
        return resolve_namelist(text, default_backend=default_backend or spec["default_backend"],
                                physics_mode=physics_mode, correct_skip=correct_skip)
    if groups.get(spec["backend_namelist_group"]):
        raise ValueError("MPAS slot backend overrides require an explicit MPAS suite route")
    return None


@dataclass
class Alarm:
    period: Fraction
    next_ring: Fraction

    def ringing(self, now: Fraction, lookahead: Fraction = Fraction(0)) -> bool:
        """Forward recurring envelope; mpas_timekeeping.F:876-921."""
        return self.next_ring <= now + lookahead

    def reset(self, now: Fraction, lookahead: Fraction = Fraction(0)) -> None:
        """Skip elapsed rings, keep phase; mpas_timekeeping.F:925-985."""
        target = now + lookahead
        if self.ringing(now, lookahead):
            self.next_ring += ((target - self.next_ring) // self.period + 1) * self.period


@dataclass(frozen=True)
class StepFlags:
    radiation_lw: bool = False
    radiation_sw: bool = False
    convection: bool = False
    greeness: bool = False
    surface: bool = False
    cam_abs: bool = False
    cam_restart: bool = False
    rain_bucket: bool = False
    radiation_bucket: bool = False
    diagnostics: bool = False


class PhysicsManager:
    """Fixed-interval manager; mpas_atmphys_manager.F:137-766."""

    # Create manager alarms; mpas_atmphys_manager.F:385-766.
    def __init__(self, config: DriverConfig, dt: float, *,
                 streams: Mapping[str, str] | None = None):
        self.config = config
        self.dt = Fraction(str(dt))
        if self.dt <= 0:
            raise ValueError("physics dynamical timestep must be positive")
        self.alarms: dict[str, Alarm] = {}
        self.scheme_dt: dict[str, Fraction] = {}
        self.last_time: Fraction | None = None
        o = config.execution_options
        streams = dict(streams or {})
        for slot, key in (("radiation_lw", "config_radtlw_interval"),
                          ("radiation_sw", "config_radtsw_interval"),
                          ("convection", "config_conv_interval"),
                          ("pbl", "config_pbl_interval")):
            period = interval_seconds(o[key])
            self.scheme_dt[slot] = self.dt if period is None else period
            if period is not None:
                self.alarms[slot] = Alarm(period, Fraction(0))
        self.scheme_dt["oml"] = self.scheme_dt["pbl"]
        # MPAS mpas_atmphys_driver_seaice.F:495-536 uses dt_pbl.
        self.scheme_dt["seaice"] = self.scheme_dt["pbl"]
        self.scheme_dt["gwdo"] = self.scheme_dt["pbl"]
        self.scheme_dt["ngw"] = self.scheme_dt["pbl"]
        #: RP6 (correct mode): False on the steps between the calls config_pbl_interval sets.
        self.pbl_due = True
        if config.correct("RP6"):
            # Noah (mpas_atmphys_driver_lsm.F:726) and the RP6-declaring Noah-MP step by dt_pbl.
            self.scheme_dt["land_surface"] = self.scheme_dt["pbl"]
        self._add("greeness", o["config_greeness_update"], required=True)
        if o["config_sst_update"]:
            self._add("surface", streams.get("surface", "none"), required=True)
        if "cam_lw" == o["config_radt_lw_scheme"] \
                or "cam_sw" == o["config_radt_sw_scheme"]:
            self._add("cam_abs", o["config_camrad_abs_update"], required=True)
        if o["config_radt_lw_scheme"] == "cam_lw":
            self._add("cam_restart", streams.get("restart", "none"), delayed=True)
        for name in ("rain_bucket", "radiation_bucket"):
            self._add(name, o["config_bucket_update"], delayed=True)
        periods = [p for s in ("output", "diagnostics")
                   if (p := interval_seconds(streams.get(s, "none"))) is not None]
        if periods:
            self.alarms["diagnostics"] = Alarm(min(periods), Fraction(0))

    def _add(self, name: str, interval: str, *, delayed: bool = False,
             required: bool = False) -> None:
        """Alarm creation; mpas_atmphys_manager.F:549-644."""
        period = interval_seconds(interval)
        if period is None:
            if required:
                raise ValueError(f"MPAS {name} alarm requires an enabled stream interval")
            return
        self.alarms[name] = Alarm(period, period if delayed else Fraction(0))

    def _ring(self, name: str, now: Fraction, *, lookahead: bool = False) -> bool:
        """Check then reset; mpas_atmphys_manager.F:263-363."""
        alarm = self.alarms.get(name)
        if alarm is None:
            return False
        ahead = self.dt if lookahead else Fraction(0)
        ringing = alarm.ringing(now, ahead)
        if ringing:
            alarm.reset(now, ahead)
        return ringing

    def advance(self, time_seconds: float) -> StepFlags:
        """Manager flags; mpas_atmphys_manager.F:137-375.

        Native creates PBL's alarm but does not query it. Bucket/CAM restart
        checks include dt lookahead; the diagnostic check resets only at now.
        """
        now = Fraction(str(time_seconds))
        if now < 0 or (self.last_time is not None and now <= self.last_time):
            raise ValueError("physics manager requires strictly increasing elapsed time")
        self.last_time = now
        c = self.config
        flags = {s: c.active(s) and (
            self._ring(s, now) if s in self.alarms else True)
                 for s in ("radiation_lw", "radiation_sw", "convection")}
        flags["greeness"] = self._ring("greeness", now)
        flags["surface"] = self._ring("surface", now)
        flags["cam_abs"] = self._ring("cam_abs", now)
        flags["cam_restart"] = self._ring("cam_restart", now, lookahead=True)
        flags["rain_bucket"] = c.active("convection") and self._ring(
            "rain_bucket", now, lookahead=True)
        flags["radiation_bucket"] = (c.active("radiation_lw") or c.active("radiation_sw")) \
            and self._ring("radiation_bucket", now, lookahead=True)
        self._ring("diagnostics", now)
        alarm = self.alarms.get("diagnostics")
        flags["diagnostics"] = alarm is not None and alarm.ringing(now, self.dt)
        # RP6 (correct mode): config_pbl_interval is the time between calls to the surface-layer, ocean
        # mixed-layer, land-surface, sea-ice, PBL and gravity wave drag schemes, which all step by dt_pbl;
        # between calls their state and tendencies are held, as WRF does with bldt/STEPBL
        # (module_surface_driver.F, module_pbl_driver.F).  Exact mode never consults this alarm.
        self.pbl_due = self._ring("pbl", now) if c.correct("RP6") else True
        return StepFlags(**flags)

    def restart_state(self) -> dict[str, Any]:
        """Framework alarm state; mpas_timekeeping.F:739-766,925-985."""
        return {"schema": SCHEMA, "config": self.config.receipt(), "dt": str(self.dt),
                "last_time": None if self.last_time is None else str(self.last_time),
                "alarms": {k: {"period": str(a.period), "next_ring": str(a.next_ring)}
                           for k, a in self.alarms.items()}}

    def restore_restart_state(self, payload: Mapping[str, Any]) -> None:
        """Restore phase without restarting alarms; mpas_timekeeping.F:925-985."""
        if payload.get("schema") != SCHEMA or payload.get("config") != self.config.receipt() \
                or payload.get("dt") != str(self.dt):
            raise ValueError("physics restart configuration differs from live driver")
        if set(payload["alarms"]) != set(self.alarms):
            raise ValueError("physics restart alarm set differs from live driver")
        candidate = {}
        for k, item in payload["alarms"].items():
            period, next_ring = Fraction(item["period"]), Fraction(item["next_ring"])
            if period != self.alarms[k].period or next_ring < 0 or next_ring % period:
                raise ValueError(f"invalid physics restart phase for {k}")
            candidate[k] = Alarm(period, next_ring)
        last = None if payload["last_time"] is None else Fraction(payload["last_time"])
        if last is not None and last < 0:
            raise ValueError("invalid physics restart elapsed time")
        self.alarms = candidate
        self.last_time = last


@dataclass(frozen=True)
class SlotCall:
    slot: str
    scheme: str
    backend: Backend
    itimestep: int
    time_seconds: float
    dt: float
    flags: StepFlags
    pools: Any
    options: Mapping[str, Any]
    #: Finding ids correct mode applies in this run (empty in exact mode); a slot implementation that
    #: declared an id at registration applies that correction when the id is present here.
    corrections: frozenset[str] = frozenset()


class DriverHooks(Protocol):
    """Numerical/framework adapter. No Python array arithmetic in these hooks."""

    # Adapter boundary; mpas_atmphys_driver.F:209-384.
    def invoke(self, operation: str, call: SlotCall) -> Any: ...
    # Adapter persistent fields; mpas_atmphys_vars.F:24-953.
    def restart_state(self) -> Mapping[str, Any]: ...
    # Adapter persistent fields; mpas_atmphys_vars.F:24-953.
    def restore_restart_state(self, payload: Mapping[str, Any]) -> None: ...


class SlotRegistry:
    """Exact slot/scheme/backend registrations, with no implicit fallback."""

    # Dispatch table for native driver slots; mpas_atmphys_driver.F:209-384.
    def __init__(self):
        self._callbacks: dict[tuple[str, str, Backend], Callable[[SlotCall], Any]] = {}
        self._corrections: dict[tuple[str, str, Backend], frozenset[str]] = {}

    # Bind native driver slot implementation; mpas_atmphys_driver.F:209-384.
    def register(self, slot: str, scheme: str, backend: str,
                 callback: Callable[[SlotCall], Any], *,
                 corrections: Iterable[str] = ()) -> None:
        """``corrections`` are the correct-mode finding ids this implementation applies."""
        key = slot, scheme, Backend(backend)
        spec = load_table()["mpasphys_v842"]["slots"]
        if slot not in spec or scheme not in spec[slot]["schemes"] or scheme == "off":
            raise ValueError(f"unknown MPAS slot/scheme {slot}/{scheme}")
        if key[2] == Backend.OFF or not callable(callback):
            raise ValueError("register an executable native-cpu or hex-cuda slot")
        if key in self._callbacks:
            raise ValueError(f"physics slot already registered: {key}")
        unknown = sorted(set(corrections) - set(CORRECTIONS_V842))
        if unknown:
            raise ValueError(f"slot {key} declares unknown corrections {unknown}")
        self._callbacks[key] = callback
        self._corrections[key] = frozenset(corrections)

    # Check executable native driver slots; mpas_atmphys_driver.F:179-384.
    def require(self, config: DriverConfig) -> None:
        if not config.moist_physics:
            return
        missing = [(s, config.selectors[s], str(config.backends[s]))
                   for s in config.selectors if config.active(s)
                   and (s == "microphysics" or config.driver_physics)
                   and not config.native_embedded(s)
                   and (s, config.selectors[s], config.backends[s]) not in self._callbacks]
        if missing:
            raise RuntimeError(f"MPAS physics slot implementations not registered: {missing}")
        # Correct mode: a finding a slot does not implement was settled at resolution into a NAMED GAP
        # (settle_corrections; coordinator ruling 2026-10-14, never a refusal), so config.corrections holds
        # only findings every active slot implements.  What remains to check is that the registered
        # implementations are the ones that settlement read.  Breakage this prevents: a registry whose
        # implementation contradicts the declarations would run a finding uncorrected while the receipt
        # names it applied (pristine physics under a correct label).
        contradicted = []
        for finding in sorted(config.corrections):
            for slot, schemes in CORRECTIONS_V842[finding]["slots"].items():
                if not config.active(slot) or (schemes is not None
                                               and config.selectors[slot] not in schemes):
                    continue
                key = slot, config.selectors[slot], config.backends[slot]
                if finding not in self._corrections.get(key, frozenset()):
                    contradicted.append((finding, slot, config.selectors[slot], str(config.backends[slot])))
        if contradicted:
            raise RuntimeError(
                f"the run's correct-mode record names {sorted({f for f, *_ in contradicted})} applied, but the "
                f"registered slot implementations {contradicted} do not declare them: resolve the namelist "
                "with the declarations of the implementations that run (resolve_namelist(declarations=...)) "
                "so each is listed as a named gap instead")

    def declared(self, slot: str, scheme: str, backend: Backend) -> frozenset[str]:
        """The finding ids a registered implementation declared (a Declarations callable for resolution)."""
        return self._corrections.get((slot, scheme, Backend(backend)), frozenset())

    # Invoke native driver slot; mpas_atmphys_driver.F:209-384.
    def run(self, call: SlotCall) -> Any:
        return self._callbacks[call.slot, call.scheme, call.backend](call)


_NATIVE_GLUE = {
    "MPAS_to_physics": "mpas_to_physics",
    "o3climatology_from_MPAS": "o3climatology_from_mpas",
}


class NativePoolArguments:
    """Named native driver arguments; mpas_atmphys_driver.F:209-384.

    Pool names/handles and native one-based solve extents are supplied by the
    forecast adapter. The calendar mapping must come from native timekeeping,
    and ``synchronize`` must copy the current input words to their native pools
    before returning. No field arithmetic or implicit pool creation occurs.
    """

    # Bind native pool extents; mpas_atmphys_driver.F:209-384.
    def __init__(self, native: Any, *, pools: Mapping[str, Any],
                 its: int, ite: int,
                 synchronize: Callable[[str, SlotCall], None]):
        if its < 1 or ite < its:
            raise ValueError("native physics requires one-based nonempty solve extents")
        self.native, self.pools = native, dict(pools)
        self.its, self.ite, self.synchronize = its, ite, synchronize
        self.calendar: Mapping[str, Any] = {}

    def update_calendar(self, values: Mapping[str, Any]) -> None:
        """Native manager metadata; mpas_atmphys_manager.F:204-213."""
        self.calendar = dict(values)

    def __call__(self, entry: str, call: SlotCall) -> Mapping[str, Any]:
        """Driver time levels/arguments; mpas_atmphys_driver.F:209-384."""
        micro = entry in ("pre_microphysics", "driver_microphysics", "post_microphysics",
                          "microphysics_from_mpas", "microphysics_to_mpas")
        values = dict(self.pools)
        if entry == "physics_get_tend":
            values.update(rk_step=call.pools["rk_step"],
                          dynamics_substep=call.pools["dynamics_substep"])
        values.update(its=self.its, ite=self.ite, itimestep=call.itimestep,
                      xtime_s=call.time_seconds, dt=call.dt,
                      time_lev=2 if micro else 1)
        values.update(self.calendar)
        for name in ("config_frac_seaice", "config_sfc_albedo"):
            values[name] = call.options[name]
        result = {}
        for name in self.native.abi["entries"][entry]["arguments"]:
            if name in values:
                result[name] = values[name]
            elif name in ("dminfo", "clock", "stream_manager", "domain", "block"):
                result[name] = self.native.framework_object(name)
            else:
                raise RuntimeError(f"native {entry}.{name} has no bound pool or calendar value")
        self.synchronize(entry, call)
        return result


class NativeCalendarState:
    """Strict native clock/flag glue; mpas_atmphys_manager.F:204-213,263-375.

    The shim is linked to the same strict f3 library, using MPAS timekeeping
    and its public manager/vars globals. It must export native_manager_state;
    this constructor fails immediately if that binding is absent.
    """

    # Bind native calendar shim; mpas_atmphys_manager.F:204-213.
    def __init__(self, native: Any, library: Any, arguments: NativePoolArguments):
        if getattr(native, 'framework_only', False):
            self.native, self.arguments = native, arguments
            return
        import ctypes as ct
        self.native, self.arguments, self.ct = native, arguments, ct
        self.library = library
        self.entry = library.native_manager_state
        self.entry.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int), ct.c_char_p,
                              ct.POINTER(ct.c_int), ct.POINTER(ct.c_float),
                              ct.POINTER(ct.c_int)]
        self.entry.restype = None

    def __call__(self, call: SlotCall) -> Mapping[str, Any]:
        """Set calendar globals before updates; mpas_atmphys_manager.F:204-213."""
        if getattr(self.native, 'framework_only', False):
            values = self.native.calendar_state(call.flags)
            self.arguments.update_calendar(values)
            return values
        ct = self.ct
        f = call.flags
        flags = (ct.c_int * 8)(f.radiation_lw, f.radiation_sw, f.convection,
                              f.cam_abs, f.cam_restart, f.rain_bucket,
                              f.radiation_bucket, f.diagnostics)
        stamp, dates = ct.create_string_buffer(512), (ct.c_int * 5)()
        julian, status = ct.c_float(), ct.c_int()
        self.entry(self.native.framework_object("clock"), flags, stamp, dates,
                   ct.byref(julian), ct.byref(status))
        if status.value:
            raise RuntimeError(f"native calendar update failed: status={status.value}")
        values = {"current_date": stamp.value.decode("ascii"),
                  "julian": julian.value, "julian_in": julian.value,
                  "leapyear": bool(dates[4])}
        self.arguments.update_calendar(values)
        return values


class NativeDriverHooks:
    """Complete f3 driver_X calls; mpas_atmphys_driver.F:209-384.

    ``arguments`` supplies exact named ABI arguments and synchronizes input
    pools. ``after_call`` copies changed pools back. Native Fortran dimensions
    and lower bounds are preserved by the provider; this adapter does not
    transpose, convert carriers or replace missing native glue with a no-op.
    External glue implements operations absent from the native ABI, including
    the RK/halo boundary and native framework manager initialization.
    """

    # Bind native entry adapter; mpas_atmphys_driver.F:209-384.
    def __init__(self, native: Any,
                 arguments: Callable[[str, SlotCall], Mapping[str, Any]], *,
                 after_call: Callable[[str, SlotCall], Any],
                 external: Mapping[str, Callable[[SlotCall], Any]],
                 save_fields: Callable[[], Mapping[str, Any]],
                 restore_fields: Callable[[Mapping[str, Any]], None]):
        self.native, self.arguments, self.after_call = native, arguments, after_call
        self.external = dict(external)
        self.save_fields, self.restore_fields = save_fields, restore_fields
        self._execution_options = {}

    def missing_operations(self, config: DriverConfig) -> tuple[str, ...]:
        """Required native closure; mpas_atm_core.F:589-596; mpas_atmphys_driver.F:179-384."""
        if not config.moist_physics:
            return ()
        operations = {
            "physics_tables_init", "physics_registry_init", "physics_run_init",
            "physics_init", "physics_timetracker_state", "physics_update_surface",
            "physics_get_tend", "post_transport_prepare", "atmphys_finalize",
        }
        o = config.execution_options
        for key, operation in (
                ("config_sst_update", "physics_update_sst"),
                ("config_sstdiurn_update", "physics_update_sstskin"),
                ("config_deepsoiltemp_update", "physics_update_deepsoiltemp")):
            if o[key]:
                operations.add(operation)
        if config.active("microphysics"):
            operations.update(("pre_microphysics", "post_microphysics"))
        if config.driver_physics:
            operations.update(("allocate_forall_physics", "MPAS_to_physics",
                               "deallocate_forall_physics", "update_convection_step1"))
            for slot, stem in (("radiation_lw", "radiation_lw"),
                               ("radiation_sw", "radiation_sw"),
                               ("surface_layer", "sfclayer"),
                               ("pbl", "pbl"), ("gwdo", "gwdo"),
                               ("convection", "convection")):
                if config.active(slot) and (slot != "pbl" or config.active("surface_layer")):
                    operations.update(("allocate_" + stem, "deallocate_" + stem))
            if o["config_bucket_update"] != "none" and o["config_bucket_radt"] > 0:
                operations.add("update_radiation_diagnostics")
            if config.active("radiation_lw") or config.active("radiation_sw"):
                operations.add("deallocate_cloudiness")
                if config.active("cloudiness"):
                    operations.add("allocate_cloudiness")
                if o["config_o3climatology"] and (
                        config.selectors["radiation_lw"] == "rrtmg_lw" or
                        config.selectors["radiation_sw"] == "rrtmg_sw"):
                    operations.add("o3climatology_from_MPAS")
            if config.active("land_surface"):
                operations.add("atmphys_sfc_diagnostics")
                if config.correct("RP6"):
                    operations.update(SURFACE_PRECIPITATION_WINDOW)
                if config.active("seaice"):
                    operations.update(("allocate_seaice", "deallocate_seaice"))
                if config.selectors["land_surface"] == "sf_noah":
                    operations.update(("allocate_lsm", "deallocate_lsm"))
            if config.active("convection"):
                operations.add("update_convection_step2")
        if config.active("ngw") and not config.native_embedded("ngw"):
            operations.add("snapshot_gwdo_tendencies")
        return tuple(sorted(op for op in operations if op not in self.external
                            and _NATIVE_GLUE.get(op, op) not in self.native.abi["entries"]))

    def preflight(self, config: DriverConfig) -> None:
        """Reject absent closure before native mutation; mpas_atmphys_init.F:83-481."""
        missing = self.missing_operations(config)
        if missing:
            raise RuntimeError(f"native physics glue is not bound: {', '.join(missing)}")

    def configure(self, config: DriverConfig) -> None:
        """Resolved Registry configs; mpas_atmphys_control.F:92-332; mpas_atmphys_init.F:83-481."""
        self._execution_options = dict(config.execution_options)
        for name, value in self._execution_options.items():
            self.native.config(name, value)

    def invoke(self, operation: str, call: SlotCall) -> Any:
        """Invoke native entry with exact ABI names; mpas_atmphys_driver.F:209-384."""
        if operation in self.external:
            return self.external[operation](call)
        if getattr(self, 'resident_operations', None) is not None and operation != 'atmphys_finalize':
            raise RuntimeError(f'resident physics has no GPU operation: {operation}')
        entry = _NATIVE_GLUE.get(operation, operation)
        if entry not in self.native.abi["entries"]:
            raise RuntimeError(f"native physics glue is not bound: {operation}")
        arguments = dict(self.arguments(entry, call))
        expected = set(self.native.abi["entries"][entry]["arguments"])
        if set(arguments) != expected:
            raise ValueError(f"native {entry} argument names differ: expected {sorted(expected)}")
        # MPAS mpas_atmphys_driver_gwdo.F:735-829: the native parent owns NGW
        # unless mixed backend dispatch masks this call's actual config pool.
        original = self._execution_options.get("config_ngw_scheme",
                                                call.options.get("config_ngw_scheme", False))
        masked = entry == "driver_gwdo" and call.options.get("config_ngw_scheme", False) != original
        if masked:
            self.native.config("config_ngw_scheme", call.options["config_ngw_scheme"])
        try:
            self.native.call(entry, **arguments)
            return self.after_call(entry, call)
        finally:
            if masked:
                self.native.config("config_ngw_scheme", original)

    def register_slots(self, registry: SlotRegistry, config: DriverConfig) -> None:
        """Register selected native drivers; mpas_atmphys_driver.F:209-365."""
        rows = load_table()["mpasphys_v842"]["slots"]
        for slot, scheme in config.selectors.items():
            if not config.active(slot) or config.backends[slot] != Backend.NATIVE_CPU:
                continue
            if config.native_embedded(slot):
                continue
            entry = rows[slot]["native_entries"][scheme]
            if entry not in self.native.abi["entries"]:
                raise RuntimeError(f"native ABI has no slot entry: {entry}")
            registry.register(slot, scheme, Backend.NATIVE_CPU,
                              lambda call, entry=entry: self.invoke(entry, call))

    def restart_state(self) -> Mapping[str, Any]:
        """Provider owns all persistent pools; mpas_atmphys_vars.F:24-953."""
        return self.save_fields()

    def restore_restart_state(self, payload: Mapping[str, Any]) -> None:
        """Provider restores transactionally; mpas_atmphys_init.F:174-396."""
        self.restore_fields(payload)


class PhysicsDriver:
    """One step's driver calls; mpas_atmphys_driver.F:111-384.

    Native assembles tendencies once before the RK/substep loops
    (mpas_atm_time_integration.F:2140-2176), then uses the held arrays in each
    stage. Microphysics follows transport.
    """

    # Bind ordered driver orchestration; mpas_atmphys_driver.F:111-384.
    def __init__(self, config: DriverConfig, manager: PhysicsManager,
                 registry: SlotRegistry, hooks: DriverHooks):
        if config != manager.config:
            raise ValueError("physics driver and manager configurations differ")
        registry.require(config)
        preflight = getattr(hooks, "preflight", None)
        if preflight is not None:
            preflight(config)
        self.config, self.manager, self.registry, self.hooks = config, manager, registry, hooks
        self.calls: list[dict[str, Any]] = []
        self._pending: tuple[int, float, StepFlags, Any] | None = None
        self._initialized = False
        self._tendencies_ready = False
        self._tendencies = None

    def _call(self, operation: str, step: int, time: float, flags: StepFlags,
              pools: Any, *, slot: str | None = None) -> Any:
        """Scheme glue invocation; mpas_atmphys_driver.F:209-365."""
        c = self.config
        options = dict(c.execution_options)
        if slot == "gwdo" and c.active("ngw") and not c.native_embedded("ngw"):
            options["config_ngw_scheme"] = False
        service = next((name for name, row in load_table()["mpasphys_v842"].get("services", {}).items()
                        if slot is None and operation in row["operations"]), None)
        glue_backend = c.service_backends.get(service, Backend.NATIVE_CPU)
        if slot is None and operation in getattr(self.hooks, 'resident_operations', ()):
            glue_backend = Backend.HEX_CUDA
        call = SlotCall(operation if slot is None else slot,
                        "driver" if slot is None else c.selectors[slot],
                        glue_backend if slot is None else c.backends[slot],
                        step, time, float(self.manager.scheme_dt.get(slot, self.manager.dt)),
                        flags, pools, MappingProxyType(options), c.corrections)
        self.calls.append({"operation": operation, "slot": call.slot,
                           "backend": str(call.backend), "itimestep": step,
                           "time_seconds": time, "flags": asdict(call.flags)})
        return self.hooks.invoke(operation, call) if slot is None else self.registry.run(call)

    def initialize(self, pools: Any) -> None:
        """Framework init glue; mpas_atmphys_init.F:83-481."""
        if self._initialized:
            raise RuntimeError("MPAS physics driver already initialized")
        if self.config.moist_physics:
            configure = getattr(self.hooks, "configure", None)
            if configure is not None:
                configure(self.config)
            # mpas_atm_core.F:589-596: native globals and input pools precede
            # scheme initialization. Packages are set before field allocation
            # by the framework adapter, using execution_options.
            for operation in ("physics_tables_init", "physics_registry_init",
                              "physics_run_init", "physics_init"):
                self._call(operation, 0, 0.0, StepFlags(), pools)
        self._initialized = True

    def begin_step(self, *, itimestep: int, time_seconds: float, pools: Any) -> StepFlags:
        """Pre-dynamics driver; mpas_atm_core.F:1036-1040; mpas_atmphys_driver.F:111-384."""
        if not self._initialized or self._pending is not None or itimestep < 1:
            raise RuntimeError("physics step requires initialization and no pending step")
        c = self.config
        flags = self.manager.advance(time_seconds) if c.moist_physics else StepFlags()
        self.calls = []
        self._pending = itimestep, time_seconds, flags, pools
        self._tendencies_ready = False
        self._tendencies = None
        if not c.moist_physics:
            return flags
        # Named glue in driver order; mpas_atmphys_driver.F:209-384.
        def glue(op):
            self._call(op, itimestep, time_seconds, flags, pools)
        # Named slot in driver order; mpas_atmphys_driver.F:209-384.
        def run(slot):
            self._call(slot, itimestep, time_seconds, flags, pools, slot=slot)
        glue("physics_timetracker_state")
        # Manager boundary updates precede all physics preparation.
        if flags.greeness:
            glue("physics_update_surface")
        if flags.surface:
            glue("physics_update_sst")
        if c.options["config_sstdiurn_update"]:
            glue("physics_update_sstskin")
        if c.options["config_deepsoiltemp_update"]:
            glue("physics_update_deepsoiltemp")
        if c.options["config_o3climatology"] and (
                flags.radiation_lw and c.selectors["radiation_lw"] == "rrtmg_lw"
                or flags.radiation_sw and c.selectors["radiation_sw"] == "rrtmg_sw"):
            glue("o3climatology_from_MPAS")
        if not c.driver_physics:
            return flags
        glue("allocate_forall_physics")
        glue("MPAS_to_physics")
        if flags.radiation_lw or flags.radiation_sw:
            if c.active("cloudiness"):
                glue("allocate_cloudiness")
                run("cloudiness")
        if flags.radiation_sw:
            glue("allocate_radiation_sw")
            run("radiation_sw")
        if flags.radiation_lw:
            glue("allocate_radiation_lw")
            run("radiation_lw")
        if c.options["config_bucket_update"] != "none" and c.options["config_bucket_radt"] > 0:
            glue("update_radiation_diagnostics")
        if c.active("radiation_sw") or c.active("radiation_lw"):
            glue("deallocate_cloudiness")
        for slot in ("radiation_sw", "radiation_lw"):
            if c.active(slot):
                glue("deallocate_" + slot)
        # RP6: pbl_due is False only in correct mode, between the calls config_pbl_interval sets.
        due = self.manager.pbl_due
        # RP6 (correct mode with config_pbl_interval): every step's precipitation is added to the window the
        # land-surface and sea-ice schemes receive at their next call (native: physics_driver, l_precip_bl).
        window = c.correct("RP6") and c.active("land_surface")
        if window:
            glue("accumulate_surface_precipitation")
        if c.active("surface_layer") and due:
            glue("allocate_sfclayer")
            run("surface_layer")
            glue("deallocate_sfclayer")
        if c.active("oml") and due:
            run("oml")
        if c.active("land_surface") and due:
            if window:
                glue("open_surface_precipitation_window")
            if c.selectors["land_surface"] == "sf_noah":
                glue("allocate_lsm")
            run("land_surface")
            if c.selectors["land_surface"] == "sf_noah":
                glue("deallocate_lsm")
            if c.active("seaice"):
                glue("allocate_seaice")
                run("seaice")
                glue("deallocate_seaice")
            if window:
                glue("close_surface_precipitation_window")
            glue("atmphys_sfc_diagnostics")
        if getattr(c, 'tc_surface', None) and c.active("surface_layer") and due:
            glue("tc_surface_spray")
        if c.active("pbl") and c.active("surface_layer") and due:
            glue("allocate_pbl")
            run("pbl")
            glue("deallocate_pbl")
        # GWDO adds to the PBL tendencies (mpas_atmphys_driver_gwdo.F:540-547), so it follows the PBL.
        if c.active("gwdo") and due:
            glue("allocate_gwdo")
            separate_ngw = c.active("ngw") and not c.native_embedded("ngw")
            if separate_ngw:
                glue("snapshot_gwdo_tendencies")
            run("gwdo")
            if separate_ngw:
                run("ngw")
            glue("deallocate_gwdo")
        glue("update_convection_step1")
        if flags.convection:
            glue("allocate_convection")
            run("convection")
            glue("deallocate_convection")
        if c.active("convection"):
            glue("update_convection_step2")
        glue("deallocate_forall_physics")
        return flags

    def physics_get_tend(self, *, rk_step: int, dynamics_substep: int, pools: Any) -> Any:
        """Held sources; mpas_atm_time_integration.F:2140-2176; mpas_atmphys_todynamics.F:98-498."""
        if self._pending is None or rk_step not in (1, 2, 3) or dynamics_substep < 1:
            raise RuntimeError("physics tendency requested outside an active RK stage")
        # DO_PHYSICS calls this even for suite none/config_moist_physics=false;
        # (the time-integration lines cited in the docstring above). Its zero sources also make
        # the final dynamics heating available to native restart output.
        if self._tendencies_ready:
            if self._tendencies is _RELEASED_TENDENCIES:
                raise RuntimeError("physics sources were already handed to the runtime and released this step")
            return self._tendencies
        if (rk_step, dynamics_substep) != (1, 1):
            raise RuntimeError("native physics sources must be assembled before the first RK stage")
        step, time, flags, _ = self._pending
        self._tendencies = self._call("physics_get_tend", step, time, flags,
                                     {"pools": pools, "rk_step": rk_step,
                                      "dynamics_substep": dynamics_substep})
        self._tendencies_ready = True
        return self._tendencies

    def release_held_tendencies(self):
        """Hand ownership to ResidentNativeCoupling after its once-per-step assembly.

        The coupling retains sources through their last use. Keeping this second
        reference prevents release_dynamics_sources from recycling the mass,
        momentum and theta arrays during scalar transport and microphysics.
        Standalone callers retain the held-source contract until they hand off.
        """
        if self._tendencies_ready:
            self._tendencies = _RELEASED_TENDENCIES

    def finish_step(self, *, pools: Any) -> Any:
        """Post-transport microphysics; mpas_atm_time_integration.F:2770-2816."""
        if self._pending is None:
            raise RuntimeError("no physics step awaits post-transport microphysics")
        step, time, flags, _ = self._pending
        result = None
        if self.config.moist_physics:
            self._call("post_transport_prepare", step, time, flags, pools)
            if self.config.active("microphysics"):
                self._call("pre_microphysics", step, time, flags, pools)
                result = self._call("microphysics", step, time, flags, pools, slot="microphysics")
                self._call("post_microphysics", step, time, flags, pools)
        self._pending = None
        return result

    def restart_state(self) -> dict[str, Any]:
        """Persistent driver pools and alarm phase; mpas_atmphys_vars.F:24-953."""
        if self._pending is not None:
            raise RuntimeError("physics restart requires a completed timestep")
        return {"schema": SCHEMA, "manager": self.manager.restart_state(),
                "fields": dict(self.hooks.restart_state()), "initialized": self._initialized}

    def restore_restart_state(self, payload: Mapping[str, Any]) -> None:
        """Restore persistent physics pools; mpas_atmphys_init.F:174-396."""
        if self._pending is not None or payload.get("schema") != SCHEMA:
            raise ValueError("invalid physics restart boundary or schema")
        # Validate alarm payload in a private manager before touching live pools.
        old = self.manager.restart_state()
        self.manager.restore_restart_state(payload["manager"])
        try:
            self.hooks.restore_restart_state(payload["fields"])
        except BaseException:
            self.manager.restore_restart_state(old)
            raise
        self._initialized = bool(payload["initialized"])

    def finalize(self, pools: Any) -> None:
        """Native cleanup entry; mpas_atmphys_finalize.F:37-67."""
        if not self._initialized or self._pending is not None:
            raise RuntimeError("physics finalize requires a completed initialized run")
        if self.config.moist_physics:
            self._call("atmphys_finalize", 0, 0.0, StepFlags(), pools)
        self._initialized = False
