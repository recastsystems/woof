"""MPAS-A v8.4.2 Registry defaults, the defaults a bare hex run gets.

A bare run uses MPAS's actual defaults.
A bare ``woof hex forecast`` integrates the dycore schedule, the physics
cadences and the timestep MPAS-Atmosphere v8.4.2 itself defaults to.  Every
value below cites the line of the upstream source it is read from
(MPAS-Model tag ``v8.4.2``, ``src/core_atmosphere/Registry.xml`` SHA-256
``c2eff7d95ec3867af69a945908955076d5f97dfe943f2d25ad756d98b94d86b3``; the
cited lines are kept in ``verification/mpas-v8.4.2-registry/``).  Every
value stays overridable from the CLI (``--dt``,
``--time-integration-order``, ``--acoustic-substeps``,
``--radiation-seconds``, ``--pbl-cadence``, ``--no-monotonic``) and from a
``--physics-namelist``.

Re-pinned from v8.4.1 on 2026-10-11 (:data:`UPSTREAM_PIN`).  One default
moved with the pin: ``config_number_of_sub_steps`` went from 2 to 4, because
two acoustic sub-steps can put noise in the jet streams of
convection-permitting runs and unphysical kinetic energy at the highest
wavenumbers (v8.4.2 release notes).  Two stays selectable
(``--acoustic-substeps 2``) and is what every earlier bare hex run took.

The configuration the native x4 reference was integrated at (RK3, six
acoustic sub-steps, 120 s, radiation every 600 s) is NOT a default any more.
It is :data:`X4_RECORDED_SCHEDULE`, which the hex self-reproducibility tests and the x4
proof pass explicitly; a run that matches it says so in its receipt.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping

#: Where the cited lines come from.
REGISTRY_SOURCE = (
    "MPAS-Model v8.4.2 src/core_atmosphere/Registry.xml "
    "(sha256 c2eff7d95ec3867af69a945908955076d5f97dfe943f2d25ad756d98b94d86b3)"
)

#: The upstream release the dycore is pinned to, and the one it was pinned to
#: before.  ``changed_files`` is every file of ``src/core_atmosphere`` whose
#: bytes differ between the two tags, with what the difference is; every other
#: file the port cites has matching source hashes at both tags, so a citation of it
#: holds at either.  Native v8.4.1 and v8.4.2, built alike and run from one
#: namelist, wrote matching numerical arrays at 2 and at 4 acoustic sub-steps
#: (verification/mpas-842-repin-20261011/).
UPSTREAM_PIN: Mapping[str, Any] = MappingProxyType({
    "repository": "https://github.com/MPAS-Dev/MPAS-Model",
    "tag": "v8.4.2",
    "tag_object": "2eee75b147d83d67835cecfb30c6fb3793e3042b",
    "commit": "f34984b2c89353cf31c235b08d3b4acd96b37bb3",
    "archive_sha256": "03e2d518c2f32eccbfa84a48d2c7efa223967819f4f72856f36b87c3b1a796f5",
    "pinned_on": "2026-10-11",
    "previous": MappingProxyType({
        "tag": "v8.4.1",
        "tag_object": "2a934b5008a7446df96d550bf2e21466feaec686",
        "commit": "91c5eac175eebeaf4206bacd5cb50c39dff3c152",
        "archive_sha256": "772f565c2bd66999492085eff8ffa0b9aa9a2edd1e7f2c0e5d1a8bedc1160861",
    }),
    "changed_files": MappingProxyType({
        "src/core_atmosphere/Registry.xml": MappingProxyType({
            "sha256": "c2eff7d95ec3867af69a945908955076d5f97dfe943f2d25ad756d98b94d86b3",
            "previous_sha256": "0731a09dd5c258cc510edfa90fbcf963e126e2f5ae1d09ae7cbe9e3b8eecde28",
            "change": (
                "version string; config_number_of_sub_steps default 2 -> 4; "
                "lbc_ni declared before lbc_nr; no line moved"),
        }),
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F": MappingProxyType({
            "sha256": "579acb2819b565760822194401d5af9e33c129a4114ac03e6882d7abdbabc757",
            "previous_sha256": "937e3191a646b0f3f14aaf1678f57b0d6880574f06e402a5053ff6ed12ab706b",
            "change": (
                "line 597, one OpenACC directive (rho_zz copied to the device "
                "instead of created there); no arithmetic, no line moved"),
        }),
        "src/core_atmosphere/dynamics/mpas_atm_dissipation_models.F": MappingProxyType({
            "sha256": "cc6e6b3b211270b7affc81fa44720e9fb6c8e39f9a42ab32c57e7c4d9bb6bcbf",
            "previous_sha256": "9771c6552582475dc01fa6d80f6678460e00fe3fdbb144f2b4fd0c2464bb4dcd",
            "change": (
                "two lines added after 409: a 0.01 floor on the eddy "
                "viscosities of the 1.5-order prognostic-TKE LES closure, "
                "which hex ports with the floor (woof.hex.mixing_v842_les, "
                "run on the MPAS route); lines after 409 move down by two"),
        }),
    }),
})

#: Each default with the upstream line that sets it.
REGISTRY_DEFAULTS: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "config_time_integration_order": MappingProxyType({
        "value": 2, "citation": "Registry.xml:69"}),
    "config_dt": MappingProxyType({
        "value": 720.0,
        "citation": (
            "Registry.xml:74 -- 720 s for the default 120 km x1.40962 mesh, "
            "i.e. 6 s per km of grid spacing")}),
    "config_split_dynamics_transport": MappingProxyType({
        "value": True, "citation": "Registry.xml:99"}),
    "config_number_of_sub_steps": MappingProxyType({
        "value": 4,
        "citation": (
            "Registry.xml:104 -- 4 from v8.4.2 (commit 0f57f834); v8.4.1 and "
            "earlier defaulted to 2")}),
    "config_dynamics_split_steps": MappingProxyType({
        "value": 3, "citation": "Registry.xml:109"}),
    "config_monotonic": MappingProxyType({
        "value": True, "citation": "Registry.xml:239"}),
    "config_coef_3rd_order": MappingProxyType({
        "value": 0.25, "citation": "Registry.xml:244"}),
    "config_radtlw_interval": MappingProxyType({
        "value": "00:30:00", "citation": "Registry.xml:2316"}),
    "config_radtsw_interval": MappingProxyType({
        "value": "00:30:00", "citation": "Registry.xml:2321"}),
    "config_conv_interval": MappingProxyType({
        "value": "none",
        "citation": (
            "Registry.xml:2326; 'none' means every step: "
            "mpas_atmphys_manager.F:522-523 sets dt_cu = dt")}),
    "config_pbl_interval": MappingProxyType({
        "value": "none",
        "citation": (
            "Registry.xml:2331; 'none' means every step: "
            "mpas_atmphys_manager.F:541-542 sets dt_pbl = dt")}),
})

DEFAULT_TIME_INTEGRATION_ORDER: int = 2
DEFAULT_ACOUSTIC_SUBSTEPS: int = 4
#: The Registry default of v8.4.1 and earlier, and of every bare hex run before
#: the 2026-10-11 re-pin.  Still a legal ``--acoustic-substeps`` value.
PREVIOUS_ACOUSTIC_SUBSTEPS: int = 2
DEFAULT_DYNAMICS_SPLITS: int = 3
DEFAULT_MONOTONIC: bool = True
DEFAULT_COEF_3RD_ORDER: float = 0.25
#: config_radtlw_interval = config_radtsw_interval = 00:30:00.
DEFAULT_RADIATION_SECONDS: float = 1800.0
#: The rule config_dt = 720 s at 120 km states: 6 s of model time per km of
#: the mesh's finest nominal spacing.
SECONDS_PER_KM: float = 6.0

#: The configuration the one native MPAS-A v8.4.1 reference (x4.163842, 24
#: ranks, 30 steps) was integrated at.  The hex self-reproducibility tests and the x4
#: proof pass it explicitly; it is evidence, not a default.
X4_RECORDED_SCHEDULE: Mapping[str, Any] = MappingProxyType({
    "config_dt": 120.0,
    "config_time_integration_order": 3,
    "config_number_of_sub_steps": 6,
    "config_dynamics_split_steps": 3,
    "config_radt_seconds": 600.0,
    "config_monotonic": False,
})


def _clock_closes(dt: float, steps: int) -> bool:
    """Whether every multiple of ``dt`` up to ``steps`` is exact in binary64.

    The drivers advance ``time_seconds`` by adding ``config_dt`` and compare
    it to ``step * dt`` every step, so a timestep whose multiples round
    differently would stop a run mid-forecast on the clock check.
    """

    accumulated = 0.0
    for step in range(1, steps + 1):
        accumulated += dt
        if accumulated != step * dt:
            return False
    return True


def default_dt_seconds(
    nominal_dx_m: float,
    *,
    radiation_seconds: float = DEFAULT_RADIATION_SECONDS,
) -> dict[str, Any]:
    """The timestep a bare run takes on a mesh, and how it was derived.

    ``6 s per km`` of the mesh's ``nominalMinDc`` (MPAS's own "nominal
    minimum distance between cell centers", which the bind checks against
    the static file FP32-exactly), rounded DOWN to the largest whole number
    of seconds that divides the radiation interval into a whole number of
    steps.  15 km -> 90 s, 30 km -> 180 s, 60 km -> 360 s, 25 km -> 150 s,
    3 km -> 18 s, 937.5 m -> 5 s.  120 km gives 600 s rather than the
    Registry's 720 s, because 720 s does not divide the 1800 s radiation
    interval and hex calls radiation on whole steps (the sealed physics
    constructor refuses a cadence that is not one).  Only a mesh finer than
    about 170 m, where no whole second fits, falls back to an exact
    fractional divisor.  The stability check at bind
    (woof.hex.timestep_admission) then measures the result against the
    mesh's real dcEdge.
    """

    dx = float(nominal_dx_m)
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError(f"nominal_dx_m={nominal_dx_m!r} must be finite and positive")
    cadence = float(radiation_seconds)
    if not math.isfinite(cadence) or cadence <= 0.0:
        raise ValueError(
            f"radiation_seconds={radiation_seconds!r} must be finite and positive")
    target = SECONDS_PER_KM * dx / 1000.0
    rejected: list[float] = []
    chosen: float | None = None
    whole = min(int(math.floor(target + 1.0e-9)), int(math.floor(cadence)))
    for seconds in range(whole, 0, -1):
        if cadence % seconds == 0.0:
            chosen = float(seconds)
            break
    if chosen is None:
        count = max(1, math.ceil(cadence / target - 1.0e-12))
        while count <= int(cadence) * 4096:
            candidate = cadence / count
            if candidate <= target * (1.0 + 1.0e-12):
                if _clock_closes(candidate, max(1, int(round(86_400.0 / candidate)))):
                    chosen = candidate
                    break
                rejected.append(candidate)
            count += 1
    if chosen is None:  # pragma: no cover - a positive cadence always has 1/2^k divisors
        raise ValueError(f"no timestep divides {cadence:g} s below {target:g} s")
    return {
        "dt_seconds": float(chosen),
        "source": "mesh",
        "rule": (
            f"{SECONDS_PER_KM:g} s per km of nominalMinDc ({dx:g} m -> "
            f"{target:g} s), rounded down to divide the {cadence:g} s "
            f"radiation interval ({int(round(cadence / chosen))} steps)"
        ),
        "nominal_dx_m": dx,
        "target_seconds": target,
        "radiation_seconds": cadence,
        "rejected_for_clock_closure": rejected,
        "citation": REGISTRY_DEFAULTS["config_dt"]["citation"],
    }


def lts_class_clock_dt(selection: dict[str, Any], *, max_rate: int, events: dict[str, float]) -> dict[str, Any]:
    """The default step moved down so full-step LTS's coarsest class step lands on every event.

    THE BREAKAGE THIS PREVENTS (multi-zoom run3, 2026-10-26, 60 km globe with 3.75/7.5/15 km zones): the default
    step for the 3.75 km cells is 20 s, the coarsest class steps 8 x 20 = 160 s, which does not divide the 1800 s
    radiation interval, so full-step LTS refused every bare run on that mesh ("not a whole number of the coarsest
    class step") and the user had to find 18.75 s by hand.  Only a DEFAULT step is moved (an explicit ``--dt``
    or a namelist ``config_dt`` is the user's and is still refused by the class clock if it misses): the largest
    step on the eighth-second grid, at most the default, for which ``step * max_rate`` divides every event.  The
    cap ``max_rate`` is the class ladder's ceiling, so any class top the mesh grades to (a power of two at most
    the cap) divides the events too.
    """

    if selection.get("source") not in ("mesh", "physics cap"):
        return selection
    base = float(selection["dt_seconds"])
    values = [float(v) for v in events.values() if v and float(v) > 0.0]

    def fits(step: float) -> bool:
        unit = step * int(max_rate)
        return all(abs(v / unit - round(v / unit)) <= 1.0e-9 * max(v / unit, 1.0) and v / unit >= 1.0 - 1.0e-9
                   for v in values)

    if fits(base):
        return selection
    for eighths in range(int(math.floor(base * 8 + 1.0e-9)), 0, -1):
        step = eighths / 8.0
        if fits(step):
            return {**selection, "dt_seconds": step, "source": "lts class clock",
                    "default_dt_seconds": base, "lts_max_rate": int(max_rate),
                    "rule": (f"{selection.get('rule', '')}; moved to {step:g} s so the coarsest LTS class step "
                             f"({int(max_rate)} x {step:g} s) divides every event {sorted(set(values))}")}
    return selection


def resolve_dt(
    requested: Any,
    *,
    nominal_dx_m: float | None,
    row_dt_seconds: float | None,
    radiation_seconds: float = DEFAULT_RADIATION_SECONDS,
) -> dict[str, Any]:
    """The run's timestep from a ``--dt`` request.

    ``None`` or ``"auto"`` is the default (:func:`default_dt_seconds`);
    ``"row"`` is the timestep the mesh row was registered with; a number is
    an explicit selection.  The answer says which, so the receipt does.  A
    row that offers no nominal spacing (a constructed double) has no rule to
    apply, so ``auto`` takes its registered timestep and says so.
    """

    text = None if requested is None else str(requested).strip().lower()
    if text in (None, "", "auto"):
        if nominal_dx_m is None:
            if row_dt_seconds is None:
                raise ValueError("the mesh row offers neither a nominal spacing nor a timestep")
            return {"dt_seconds": float(row_dt_seconds), "source": "row",
                    "rule": "the row offers no nominal spacing, so its registered timestep"}
        return default_dt_seconds(nominal_dx_m, radiation_seconds=radiation_seconds)
    if text == "row":
        if row_dt_seconds is None:
            raise ValueError("--dt row: this mesh row declares no timestep")
        return {"dt_seconds": float(row_dt_seconds), "source": "row",
                "rule": "the timestep the mesh row was registered with"}
    try:
        value = float(text)
    except ValueError:
        raise ValueError(
            f"--dt {requested!r} is not 'auto', 'row' or a number of seconds"
        ) from None
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"--dt {requested!r} must be a finite positive number of seconds")
    return {"dt_seconds": value, "source": "explicit", "rule": "--dt"}


def x4_recorded_match(config: Any) -> dict[str, Any]:
    """Whether a configuration is the native x4 reference's own schedule.

    Recorded in receipts as evidence.  It admits and refuses nothing.
    """

    differences = {}
    for knob, recorded in X4_RECORDED_SCHEDULE.items():
        actual = getattr(config, knob, None)
        same = (
            isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and not isinstance(recorded, bool)
            and float(actual) == float(recorded)
        ) or (isinstance(recorded, bool) and bool(actual) is recorded)
        if not same:
            differences[knob] = {"run": actual, "recorded": recorded}
    return {
        "matched": not differences,
        "recorded_schedule": dict(X4_RECORDED_SCHEDULE),
        "differences": differences,
        "meaning": (
            "the native MPAS-A v8.4.1 x4.163842 reference was integrated at "
            "the recorded schedule; x4 tests broad full-physics consistency and "
            "hex self-reproducibility. Native dry-step agreement is measured "
            "separately by verify-dycore; a run elsewhere is not refused"
        ),
    }


__all__ = [
    "DEFAULT_ACOUSTIC_SUBSTEPS",
    "PREVIOUS_ACOUSTIC_SUBSTEPS",
    "UPSTREAM_PIN",
    "DEFAULT_COEF_3RD_ORDER",
    "DEFAULT_DYNAMICS_SPLITS",
    "DEFAULT_MONOTONIC",
    "DEFAULT_RADIATION_SECONDS",
    "DEFAULT_TIME_INTEGRATION_ORDER",
    "REGISTRY_DEFAULTS",
    "REGISTRY_SOURCE",
    "SECONDS_PER_KM",
    "X4_RECORDED_SCHEDULE",
    "default_dt_seconds",
    "resolve_dt",
    "x4_recorded_match",
]
