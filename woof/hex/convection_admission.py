"""Convection admission: OFF below 3 km, New Tiedtke at 20 km and coarser.

The default has three bands since 2026-10-05: no cumulus scheme below 3 km
(the 2026-08-26 ruling below), Grell-Freitas from 3 km up to 20 km, and New
Tiedtke at 20 km and coarser (``NTIEDTKE_AT_OR_ABOVE_M``, the ruling of
2026-10-05, with its measurement beside the constant).

It was ruled on 2026-08-26 that convection is switched off below 3 km,
answering directly whether a fine swath runs the cumulus scheme at all.
This module is that ruling made executable, and it is deliberately the only
place the threshold exists.

THE BREAKAGE THIS GATE PREVENTS, MEASURED (2026-08-26,
``evidence/dt-anchors-20260826/RECEIPT.md``).  WRF pins ``cudt = 0`` for
``cu_physics = 3``, so Grell-Freitas recomputes on every model step and no
configuration relaxes it -- the cumulus cadence IS the timestep.  A finer
mesh declares a smaller timestep, so it calls the closure proportionally
more often: 30 times an hour at the proven 120 s, 180 at 20 s (a 3 km
swath), 720 at 5 s (a 750 m swath).  At 5 s on ``x1.40962`` the measured
|w| mean over four half-hour windows ran 12.0 / 49.6 / 73.5 / 87.5 m/s
against a same-card 120 s control's 1.15 / 1.17 / 1.20 / 1.48, still
climbing at 2 h, with |w| max 102.67 m/s against 1.680.  Every step was
finite and two arms were byte-identical, so it is a different SOLUTION and
not an unstable one -- which is worse, because nothing crashes.

Every mesh the ruling covers sits between six and twenty-four times the
proven call rate.  So the ruling and the measurement point the same way, and
this gate is where they meet.

**Fixed means default.**  A bare run on a sub-3-km mesh selects no cumulus
scheme, with nobody passing a flag.  The explicit switch is an A/B
instrument -- it is how the attribution above is measured -- and a decision
taken through it says ``source: "explicit"`` in its own receipt so it can
never be read as the remedy.

**On a uniform mesh the finest spacing decides.**  The decision takes the
smaller of the row's declared nominal spacing and the mesh's own measured
``min(dcEdge)``, so a row that declares one resolution and carries another
cannot smuggle the closure back on.

**On a variable-resolution mesh each cell's SCHEME spacing decides
(2026-10-11).**  MPAS-A hands every cumulus scheme the cell's own length
scale, ``len_disp / meshDensity**0.25`` (``mpas_atmphys_driver_convection.F
:718``, ``:840``), and the scale-aware schemes (Grell-Freitas, New Tiedtke)
fade their closure out toward convection-permitting cells from that number.
A hexagon's ``min(dcEdge)`` is not that length: on the 48-3 km patch mesh
it is 2,778.75 m where the finest cell's scheme spacing is 3,000.28 m, and
on the 60-3 km mesh the finest cell's is 3,034.9 m.  Read at ``min(dcEdge)``
the 3 km ruling switched the scheme off for the WHOLE mesh, its 48-60 km
cells included.  THE BREAKAGE THIS CLOSES, MEASURED (2026-10-11, the 48-3 km
patch mesh, GFS 2026-10-03 12Z):
the bind logged "finest spacing 2778.8 m vs 3000 m threshold -> off" and
the run made no convective rain in any cell.  So on a mesh whose per-cell
spacing varies, the rulings' bands are read at the finest cell's scheme
spacing, the scheme runs in every cell with that cell's own spacing, and
its scale awareness does the fading -- MPAS-A's own way for meshes that
reach convection-permitting cells (its ``convection_permitting`` suite runs
scale-aware Grell-Freitas on the 60-3 and 15-3 km meshes).  Reading the
bands at the finest cell keeps every registered variable row on the scheme
it was anchored with.  The column seam has no per-column cumulus switch, so
a variable mesh whose finest cell is under 3 km keeps the whole-mesh
``off`` it had before, and its receipt names the scheme its coarse cells
would take.

**Convection-off is a NEW admitted configuration, not a mutation of the
proven one.**  How often a scheme is called is part of what a timestep
anchor certifies (:mod:`woof.hex.dt_admission`), so a convection-off run
earns its own anchors and cannot borrow the Grell-Freitas rows'.  The
frozen v8.4.1 column-physics lane's proven configuration stays exactly
reachable and byte-unchanged.

**The arbitrary acceptance test.**  Selecting the scheme is table work in
one place: a threshold, a named set, and one translation between the config
vocabulary and the sealed constructor's.  Adding a scheme is a row here, not
a new configuration subclass per combination.
"""

from __future__ import annotations

import math
from typing import Any


class ConvectionAdmissionError(RuntimeError):
    """A convection selection is refused, by name."""


#: The config-level name of the proven cumulus scheme.
SCHEME_GRELL_FREITAS: str = "cu_grell_freitas"

#: The config-level name for "no cumulus scheme is called at all".
SCHEME_OFF: str = "off"

#: The config-level name of the New Tiedtke scheme, MPAS's own spelling
#: (``config_convection_scheme = 'cu_ntiedtke'``).  It is the scheme RRFS v2
#: runs on its 12 km mesh: the public workflow substitutes it for the suite's
#: default when ``FCST_CONVECTION_SCHEME`` is true (rrfs-workflow 85ea55c0,
#: ``scripts/exrrfs_fcst.sh:62-66``, ``workflow/config_resources/
#: config.meshdep:31-35``).  Since 2026-10-05 it is also the default at 20
#: km and coarser (``NTIEDTKE_AT_OR_ABOVE_M``); below that a run reaches it
#: only by asking for it (``requested="ntiedtke"``, the RRFS v2 profile).
#:
#: The column arithmetic is the engine's own ``cu_physics = 16`` adapter
#: (``woof.core.ntiedtke.NewTiedtke``), which the frozen column batch cannot
#: reach because its cadence table admits ``gf``, ``kf`` and ``None`` only.
#: :mod:`woof.hex.cuda_ntiedtke_column_v841` is the row's seam: the frozen
#: batch class, unmodified, with its driver built at ``cu_physics = 16`` and
#: the scheme's raw momentum rates added to ``du``/``dv`` exactly as native
#: MPAS adds ``rucuten``/``rvcuten`` (``mpas_atmphys_todynamics.F``).
SCHEME_NTIEDTKE: str = "cu_ntiedtke"

#: Every cumulus selection the column-physics lane admits.  A value outside
#: this set is refused by ``V841MpasColumnPhysicsConfig.validate``.
ADMITTED_CONVECTION_SCHEMES: tuple[str, ...] = (
    SCHEME_GRELL_FREITAS,
    SCHEME_OFF,
    SCHEME_NTIEDTKE,
)

#: The config name -> the sealed constructor's name.  One table, so adding a
#: scheme is a row and the two vocabularies cannot drift.
_CONSTRUCTOR_NAMES: dict[str, str | None] = {
    SCHEME_GRELL_FREITAS: "gf",
    SCHEME_OFF: None,
    SCHEME_NTIEDTKE: "ntiedtke",
}

#: The human label a receipt, roster or refusal uses for each selection.
_LABELS: dict[str, str] = {
    SCHEME_GRELL_FREITAS: "GF",
    SCHEME_OFF: "convection off",
    SCHEME_NTIEDTKE: "New Tiedtke",
}

#: The engine's ``cu_physics`` number per constructor name.  The generic
#: column seam is handed the number; the names are hex's receipt vocabulary.
_CU_PHYSICS: dict[str | None, int] = {"gf": 3, None: 0, "ntiedtke": 16}


def register_engine_scheme(
    config_name: str, *, constructor_name: str, cu_physics: int, label: str
) -> None:
    """Admit a cumulus scheme WOOF's physics registry declares.

    The three rows above are the schemes the hex rulings name (the
    resolution default).  Every other cumulus option is the engine's: the
    physics resolver (:mod:`woof.hex.physics_suites`) reads it from
    ``woof/physics_registry_v2.json`` and registers it here, so the config
    name, the constructor name, the label and the ``--convection`` request
    exist for it without a row written in this file.  A name already
    registered must be registered the same way.
    """

    global ADMITTED_CONVECTION_SCHEMES, REQUESTS
    config = str(config_name)
    constructor = str(constructor_name)
    number = int(cu_physics)
    if config in _CONSTRUCTOR_NAMES:
        held = _CONSTRUCTOR_NAMES[config]
        if held != constructor or _CU_PHYSICS.get(held) != number:
            raise ConvectionAdmissionError(
                f"config_convection_scheme={config!r} is already registered as "
                f"{held!r} (cu_physics={_CU_PHYSICS.get(held)}); it cannot also "
                f"be {constructor!r} (cu_physics={number})"
            )
        return
    if _CU_PHYSICS.get(constructor, number) != number:
        raise ConvectionAdmissionError(
            f"constructor name {constructor!r} already means cu_physics="
            f"{_CU_PHYSICS[constructor]}, not {number}"
        )
    _CONSTRUCTOR_NAMES[config] = constructor
    _CU_PHYSICS[constructor] = number
    _LABELS[config] = str(label)
    _REQUESTED_SCHEMES.setdefault(constructor, config)
    ADMITTED_CONVECTION_SCHEMES = tuple(_CONSTRUCTOR_NAMES)
    if constructor not in REQUESTS:
        REQUESTS = REQUESTS + (constructor,)


def cu_physics_for_constructor_scheme(scheme: str | None) -> int:
    """The engine's ``cu_physics`` for a constructor name."""

    if scheme in _CU_PHYSICS:
        return _CU_PHYSICS[scheme]
    raise ConvectionAdmissionError(
        f"cumulus_scheme={scheme!r} is not a registered cumulus selection "
        f"(registered: {sorted(str(k) for k in _CU_PHYSICS)})"
    )


#: Schemes WRF and MPAS call on every model step (WRF pins ``cudt = 0`` for
#: ``cu_physics = 3`` and ``16``; the RRFS v2 namelists set no
#: ``config_conv_interval``), so their cadence IS the timestep.  Constructor
#: vocabulary.
EVERY_STEP_SCHEMES: frozenset[str] = frozenset({"gf", "ntiedtke"})

#: Convection is switched off BELOW this spacing.
#: From here up to ``NTIEDTKE_AT_OR_ABOVE_M`` the proven Grell-Freitas
#: configuration is unchanged.
CONVECTION_OFF_BELOW_M: float = 3_000.0

#: New Tiedtke, not Grell-Freitas, AT and above
#: this spacing.  It replaces the earlier Grell-Freitas default for these
#: meshes only.  20 km is where the registered rows change timestep: every
#: row at 20 km or coarser declares dt 120 s and every finer row declares
#: 100 s or less, so the band is exactly the set the dt 120 s New Tiedtke
#: anchor admits, and no finer row loses its Grell-Freitas anchor.
NTIEDTKE_AT_OR_ABOVE_M: float = 20_000.0

#: The ruling, quoted where the threshold lives so the number never travels
#: without the authority that set it.
RULING: str = (
    "2026-08-26: convection is switched off below 3 km, answering directly "
    "whether a fine swath runs the convection scheme at all.  Changing this "
    "threshold is a decision of record and not a tuning knob: the number is "
    "the boundary a measured breakage sits on, so moving it needs the "
    "measurement redone, not an edit"
)

#: The 2026-10-05 ruling, quoted where its threshold lives.
NTIEDTKE_RULING: str = (
    "2026-10-05: New Tiedtke becomes the hex "
    "coarse-mesh default once a second case confirms it scores better "
    "against observations.  Two 72 h x1.40962 cases confirmed it, so it "
    "supersedes the 2026-08-26 default's Grell-Freitas answer at 20 km and "
    "coarser"
)

#: What the 2026-10-05 change fixes, measured, so the default names its own
#: evidence (docs/declared-divergences.md section 1).
NTIEDTKE_MEASUREMENT: str = (
    "Grell-Freitas leaves a cold drift in the upper tropical troposphere. "
    "MEASURED (2026-10-05, x1.40962, 72 h, radiosondes from IGRA2, tropics "
    "20S-20N): 250 hPa bias/RMS -2.03/2.24 K with GF against -0.71/1.04 K "
    "with New Tiedtke from GFS 2026-10-01 00Z, and -2.57/2.97 K against "
    "-0.96/1.52 K from GFS 2026-01-15 00Z.  Against GFS analyses the "
    "global 250 hPa bias goes -1.11 to -0.50 K and -1.18 to -0.63 K.  "
    "evidence/dt-anchor-ntiedtke-20261005/RECEIPT.md"
)

#: The measured breakage, kept beside the threshold because a gate that
#: cannot name what it prevents does not exist (gate law, 2026-08-16).
BREAKAGE: str = (
    "WRF pins cudt=0 for cu_physics=3, so Grell-Freitas is called every "
    "model step and a finer mesh calls it proportionally more often -- 30 "
    "times an hour at the proven 120 s, 180 at the 20 s a 3 km swath "
    "declares, 720 at the 5 s a 750 m swath declares.  MEASURED "
    "(2026-08-26, evidence/dt-anchors-20260826/RECEIPT.md, x1.40962): at "
    "5 s the |w| mean over four half-hour windows ran 12.0/49.6/73.5/87.5 "
    "m/s against a same-card 120 s control's 1.15/1.17/1.20/1.48, still "
    "climbing at 2 h, with |w| max 102.67 m/s against 1.680.  Finite at "
    "every one of 1,440 steps and byte-identical across two arms, so it is "
    "a different solution rather than an unstable one and nothing crashes "
    "to announce it"
)

#: What ``--convection`` accepts.  ``auto`` is the ruling; ``off``, ``gf``
#: and ``ntiedtke`` are explicit selections that record themselves as such
#: (``ntiedtke`` is also the RRFS v2 profile's), and ``auto`` resolves to
#: ``ntiedtke`` at 20 km and coarser.
REQUESTS: tuple[str, ...] = ("auto", "off", "gf", "ntiedtke")

#: Request -> config-level scheme for the explicit requests.
_REQUESTED_SCHEMES: dict[str, str] = {
    "off": SCHEME_OFF,
    "gf": SCHEME_GRELL_FREITAS,
    "ntiedtke": SCHEME_NTIEDTKE,
}


def _positive_length(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ConvectionAdmissionError(
            f"{name}={value!r} m must be finite and positive: the convection "
            f"ruling is taken on a real grid spacing, and a spacing that is "
            f"not one cannot be compared to the {CONVECTION_OFF_BELOW_M:g} m "
            f"threshold"
        )
    return result


def convection_for_spacing(finest_spacing_m: float) -> str:
    """The cumulus scheme the rulings select at this grid spacing."""

    spacing = _positive_length("finest_spacing_m", finest_spacing_m)
    if spacing < CONVECTION_OFF_BELOW_M:
        return SCHEME_OFF
    if spacing >= NTIEDTKE_AT_OR_ABOVE_M:
        return SCHEME_NTIEDTKE
    return SCHEME_GRELL_FREITAS


def default_convection_scheme(
    *, nominal_dx_m: float, minimum_dc_edge_m: float | None = None
) -> str:
    """The scheme a BARE run gets -- no flag, no argument, no opt-in."""

    return convection_decision(
        nominal_dx_m=nominal_dx_m, minimum_dc_edge_m=minimum_dc_edge_m
    )["scheme"]


def finest_spacing(
    *, nominal_dx_m: float, minimum_dc_edge_m: float | None = None
) -> float:
    """The smallest cell spacing the mesh carries, declared or measured."""

    nominal = _positive_length("nominal_dx_m", nominal_dx_m)
    if minimum_dc_edge_m is None:
        return nominal
    measured = _positive_length("minimum_dc_edge_m", minimum_dc_edge_m)
    return min(nominal, measured)


#: Per-cell spacings that differ by more than this relative amount make a
#: variable-resolution mesh.  A uniform mesh's ``meshDensity`` is 1.0 in
#: every cell (UCAR's x1 meshes, and the generator's uniform output), so its
#: per-cell spacing is one number; float noise in a density field is orders
#: of magnitude below one percent, and the coarsest refinement any hex mesh
#: carries is a factor of two.
VARIABLE_MESH_SPREAD: float = 0.01


def cell_spacing_m(mesh_density: Any, len_disp_m: float) -> Any:
    """MPAS-A's per-cell scheme length scale, ``len_disp / meshDensity**0.25``.

    The same construction ``woof.hex.cuda_gwdo_v841.native_cell_dx_m`` builds
    for the column physics, in vectorised FP32 here because the admission
    only classifies cells against band thresholds and must not loop over a
    835,586-cell mesh in Python.
    """

    import numpy as np

    density = np.asarray(mesh_density, dtype=np.float32)
    if density.ndim != 1 or density.size == 0:
        raise ConvectionAdmissionError("meshDensity must be a non-empty cell vector")
    if (
        not np.all(np.isfinite(density))
        or np.any(density <= 0.0)
        or np.any(density > 1.0)
    ):
        raise ConvectionAdmissionError(
            "meshDensity must be finite and within (0, 1]: the per-cell "
            "spacing the convection decision classifies is built from it"
        )
    length = np.float32(_positive_length("len_disp_m", len_disp_m))
    return length / np.power(density, np.float32(0.25))


def _per_cell_summary(
    *,
    nominal_dx_m: float,
    cell_spacing: Any | None,
    coarsest_dx_m: float | None,
) -> dict[str, Any] | None:
    """Finest/coarsest per-cell spacing and the count under 3 km, or None.

    ``None`` means the mesh is uniform (or nothing per-cell is known), and
    the whole-mesh decision applies unchanged.
    """

    if cell_spacing is not None:
        import numpy as np

        spacing = np.asarray(cell_spacing, dtype=np.float64)
        if spacing.ndim != 1 or spacing.size == 0 or not np.all(
            np.isfinite(spacing)
        ) or np.any(spacing <= 0.0):
            raise ConvectionAdmissionError(
                "the per-cell spacing must be a non-empty vector of finite, "
                "positive lengths"
            )
        finest = float(spacing.min())
        coarsest = float(spacing.max())
        below = int(np.count_nonzero(spacing < CONVECTION_OFF_BELOW_M))
        cells = int(spacing.size)
        basis = "mesh meshDensity: len_disp / meshDensity**0.25 per cell"
    elif coarsest_dx_m is not None:
        finest = _positive_length("nominal_dx_m", nominal_dx_m)
        coarsest = _positive_length("coarsest_dx_m", coarsest_dx_m)
        below = 0 if finest >= CONVECTION_OFF_BELOW_M else -1
        cells = None
        basis = "row: nominal (finest) and background (coarsest) spacing"
    else:
        return None
    if coarsest <= finest * (1.0 + VARIABLE_MESH_SPREAD):
        return None
    return {
        "basis": basis,
        "finest_cell_spacing_m": finest,
        "coarsest_cell_spacing_m": coarsest,
        "cells": cells,
        # -1: the row alone says cells finer than 3 km exist, uncounted.
        "cells_below_threshold": below,
    }


def constructor_scheme(config_scheme: str) -> str | None:
    """Translate the config name into the sealed constructor's name.

    ``SealedArwenConstructorV841`` takes ``'gf'``, ``'ntiedtke'`` or
    ``None``.  One translation, in one place, so the two vocabularies cannot
    drift.
    """

    if config_scheme in _CONSTRUCTOR_NAMES:
        return _CONSTRUCTOR_NAMES[config_scheme]
    raise ConvectionAdmissionError(
        f"config_convection_scheme={config_scheme!r} is not a cumulus "
        f"selection this run resolved (resolved: "
        f"{', '.join(ADMITTED_CONVECTION_SCHEMES)}).  Any cumulus option of "
        f"WOOF's physics registry is selected through the physics resolver "
        f"(--physics config_convection_scheme=<MPAS name, registry option id "
        f"or cu_physics number>), which registers it here; a name it never "
        f"resolved would be recorded in the receipt and never called"
    )


def gf_ishallow(config_scheme: str) -> int:
    """GF's shallow branch: on with GF, and meaningless without it.

    Native MPAS v8.4.1 hardwires ``ishallow = 1``
    (``mpas_atmphys_vars.F:340``), and the sealed constructor refuses
    ``gf_ishallow=1`` when no GF is selected.  Derived here rather than
    written as a literal beside every constructor mapping, so switching the
    scheme off cannot leave a shallow flag behind to refuse the run.
    """

    return 1 if constructor_scheme(config_scheme) == "gf" else 0


def label(config_scheme: str) -> str:
    """The human name a receipt, roster or refusal uses for one selection."""

    if config_scheme in _LABELS:
        return _LABELS[config_scheme]
    raise ConvectionAdmissionError(
        f"config_convection_scheme={config_scheme!r} is not a cumulus "
        "selection this run resolved, so there is no label to print for it"
    )


def label_for_constructor_scheme(scheme: str | None) -> str:
    """The same label, from the sealed constructor's vocabulary."""

    if scheme is None:
        return "convection off"
    for config, constructor in _CONSTRUCTOR_NAMES.items():
        if constructor == scheme and config != SCHEME_GRELL_FREITAS:
            return _LABELS[config]
    return str(scheme).upper()


def convection_decision(
    *,
    nominal_dx_m: float,
    minimum_dc_edge_m: float | None = None,
    requested: str = "auto",
    cell_spacing_m: Any | None = None,
    coarsest_dx_m: float | None = None,
) -> dict[str, Any]:
    """Decide the run's cumulus selection, and record why.

    Returns a JSON-ready mapping that rides into the run's own receipt: the
    spacing the decision was taken on, the threshold, the ruling, the
    measured breakage, and whether the decision came from the ruling
    (``source: "resolution"``) or from an explicit A/B arm
    (``source: "explicit"``).

    ``cell_spacing_m`` is the mesh's per-cell scheme spacing
    (:func:`cell_spacing_m`); ``coarsest_dx_m`` is the row-only stand-in
    (a generated row's background spacing) preflight has before any file
    is read.  When either shows a variable-resolution mesh the decision is
    taken per cell (module docstring); on a uniform mesh both are ignored
    and the decision is the whole-mesh one, unchanged.
    """

    per_cell = _per_cell_summary(
        nominal_dx_m=nominal_dx_m,
        cell_spacing=cell_spacing_m,
        coarsest_dx_m=coarsest_dx_m,
    )
    if per_cell is not None:
        return _variable_mesh_decision(
            nominal_dx_m=nominal_dx_m,
            minimum_dc_edge_m=minimum_dc_edge_m,
            requested=requested,
            per_cell=per_cell,
        )
    return _uniform_mesh_decision(
        nominal_dx_m=nominal_dx_m,
        minimum_dc_edge_m=minimum_dc_edge_m,
        requested=requested,
    )


def _check_request(requested: str) -> None:
    if requested not in REQUESTS:
        raise ConvectionAdmissionError(
            f"convection request {requested!r} is not one of "
            f"{list(REQUESTS)}: 'auto' applies the rulings (off below "
            f"{CONVECTION_OFF_BELOW_M:g} m, New Tiedtke at "
            f"{NTIEDTKE_AT_OR_ABOVE_M:g} m and coarser, GF between), and "
            f"every other request is an explicit selection that records "
            f"itself as such"
        )


def _variable_mesh_decision(
    *,
    nominal_dx_m: float,
    minimum_dc_edge_m: float | None,
    requested: str,
    per_cell: dict[str, Any],
) -> dict[str, Any]:
    """The per-cell decision on a variable-resolution mesh."""

    _check_request(requested)
    finest = float(per_cell["finest_cell_spacing_m"])
    coarsest = float(per_cell["coarsest_cell_spacing_m"])
    below_cells = per_cell["cells_below_threshold"]
    # The bands are read at the finest cell's SCHEME spacing.
    ruled = convection_for_spacing(finest)
    below = finest < CONVECTION_OFF_BELOW_M
    coarse_scheme = convection_for_spacing(coarsest)
    if below and coarse_scheme != SCHEME_OFF:
        # The 3 km ruling per cell needs a per-column cumulus switch the
        # column seam does not have, so this mesh keeps the whole-mesh
        # answer it had before rather than running the closure in cells the
        # ruling switches off.
        per_cell["status"] = (
            f"{'some' if below_cells is None or below_cells < 0 else below_cells}"
            f" cells are finer than {CONVECTION_OFF_BELOW_M:g} m and the "
            f"column seam has no per-column cumulus switch, so the whole mesh "
            f"stays off as before (its coarse cells would take "
            f"{label(coarse_scheme)})"
        )
    elif ruled != SCHEME_OFF:
        per_cell["status"] = (
            f"every cell is {CONVECTION_OFF_BELOW_M:g} m or coarser; "
            f"{label(ruled)} runs in every cell with that cell's own spacing "
            f"({finest:g}-{coarsest:g} m), and the scheme's scale awareness "
            f"fades it toward the finest cells"
        )
    else:
        per_cell["status"] = (
            f"the coarsest cell ({coarsest:g} m) is below the "
            f"{CONVECTION_OFF_BELOW_M:g} m threshold"
        )
    coarse = finest >= NTIEDTKE_AT_OR_ABOVE_M
    if requested == "auto":
        scheme = ruled
        source = "resolution"
        note = (
            f"the ruling decided this per cell on a variable-resolution mesh "
            f"({finest:g}-{coarsest:g} m): {per_cell['status']}; the run "
            f"selects {label(scheme)} with no flag passed"
        )
    else:
        scheme = _REQUESTED_SCHEMES[requested]
        source = "explicit"
        if scheme == ruled:
            note = (
                f"explicit A/B arm; it happens to agree with the per-cell "
                f"ruling on this {finest:g}-{coarsest:g} m mesh, which would "
                f"have selected {label(ruled)} anyway"
            )
        else:
            note = (
                f"explicit A/B arm OVERRIDING the per-cell ruling: on this "
                f"{finest:g}-{coarsest:g} m mesh the ruling selects "
                f"{label(ruled)} and this run selects {label(scheme)}.  This "
                f"is an experiment arm, never a remedy -- the remedy is the "
                f"default"
            )
    return {
        "schema": "gpuwm-hex.convection-decision/v1",
        "scheme": scheme,
        "constructor_scheme": constructor_scheme(scheme),
        "gf_ishallow": gf_ishallow(scheme),
        "source": source,
        "requested": requested,
        "ruled_scheme": ruled,
        "finest_spacing_m": finest,
        "nominal_dx_m": float(nominal_dx_m),
        "minimum_dc_edge_m": (
            None if minimum_dc_edge_m is None else float(minimum_dc_edge_m)
        ),
        "threshold_m": CONVECTION_OFF_BELOW_M,
        "below_threshold": below,
        "ntiedtke_threshold_m": NTIEDTKE_AT_OR_ABOVE_M,
        "coarse_band": coarse,
        "variable_resolution": per_cell,
        "ruling": RULING,
        "ntiedtke_ruling": NTIEDTKE_RULING,
        "breakage": BREAKAGE,
        "variable_mesh_breakage": VARIABLE_MESH_BREAKAGE,
        "note": note,
    }


#: What the per-cell reading fixes, measured, so the default names its own
#: evidence (gate law).
VARIABLE_MESH_BREAKAGE: str = (
    "On a variable-resolution mesh the old reading compared min(dcEdge), "
    "not the scheme's per-cell spacing, with the 3 km threshold: the 48-3 km "
    "patch mesh's min(dcEdge) of 2778.75 m (finest scheme spacing 3000.28 m) "
    "switched cumulus off for the whole mesh, its 48 km cells included.  "
    "MEASURED (2026-10-11, GFS 2026-10-03 12Z): no convective rain in any cell"
)


def _uniform_mesh_decision(
    *,
    nominal_dx_m: float,
    minimum_dc_edge_m: float | None,
    requested: str,
) -> dict[str, Any]:
    """The whole-mesh decision, unchanged from before 2026-10-11."""

    if requested not in REQUESTS:
        raise ConvectionAdmissionError(
            f"convection request {requested!r} is not one of "
            f"{list(REQUESTS)}: 'auto' applies the rulings (off below "
            f"{CONVECTION_OFF_BELOW_M:g} m, New Tiedtke at "
            f"{NTIEDTKE_AT_OR_ABOVE_M:g} m and coarser, GF between), and "
            f"every other request is an explicit selection that records "
            f"itself as such"
        )

    spacing = finest_spacing(
        nominal_dx_m=nominal_dx_m, minimum_dc_edge_m=minimum_dc_edge_m
    )
    ruled = convection_for_spacing(spacing)
    below = spacing < CONVECTION_OFF_BELOW_M
    coarse = spacing >= NTIEDTKE_AT_OR_ABOVE_M

    if requested == "auto":
        scheme = ruled
        source = "resolution"
        if coarse:
            band = (
                f"at or above the {NTIEDTKE_AT_OR_ABOVE_M:g} m New Tiedtke "
                f"threshold"
            )
        else:
            band = (
                f"{'below' if below else 'at or above'} the "
                f"{CONVECTION_OFF_BELOW_M:g} m threshold"
            )
        note = (
            f"the ruling decided this: the finest spacing on the mesh is "
            f"{spacing:g} m, which is {band}, so the run selects "
            f"{label(scheme)} with no flag passed"
        )
    else:
        scheme = _REQUESTED_SCHEMES[requested]
        source = "explicit"
        if scheme == SCHEME_NTIEDTKE and scheme != ruled:
            note = (
                f"explicit profile selection of {label(scheme)} (the scheme "
                f"RRFS v2 runs at 12 km); the resolution ruling, which "
                f"selects {label(ruled)} at {spacing:g} m, is not what chose "
                f"it"
                + (
                    ".  This mesh is below the ruling's threshold, so the "
                    "selection OVERRIDES the ruling"
                    if below
                    else ""
                )
            )
        elif scheme == ruled:
            note = (
                f"explicit A/B arm; it happens to agree with the ruling at "
                f"{spacing:g} m, which would have selected {label(ruled)} "
                f"anyway"
            )
        else:
            note = (
                f"explicit A/B arm OVERRIDING the ruling: at {spacing:g} m "
                f"the ruling selects {label(ruled)} and this run selects "
                f"{label(scheme)}.  This is an experiment arm, never a "
                f"remedy -- the remedy is the default"
            )

    return {
        "schema": "gpuwm-hex.convection-decision/v1",
        "scheme": scheme,
        "constructor_scheme": constructor_scheme(scheme),
        "gf_ishallow": gf_ishallow(scheme),
        "source": source,
        "requested": requested,
        "ruled_scheme": ruled,
        "finest_spacing_m": spacing,
        "nominal_dx_m": float(nominal_dx_m),
        "minimum_dc_edge_m": (
            None if minimum_dc_edge_m is None else float(minimum_dc_edge_m)
        ),
        "threshold_m": CONVECTION_OFF_BELOW_M,
        "below_threshold": below,
        "ntiedtke_threshold_m": NTIEDTKE_AT_OR_ABOVE_M,
        "ruling": RULING,
        "ntiedtke_ruling": NTIEDTKE_RULING,
        "breakage": BREAKAGE,
        "note": note,
    }


__all__ = [
    "ADMITTED_CONVECTION_SCHEMES",
    "BREAKAGE",
    "CONVECTION_OFF_BELOW_M",
    "NTIEDTKE_AT_OR_ABOVE_M",
    "NTIEDTKE_MEASUREMENT",
    "NTIEDTKE_RULING",
    "ConvectionAdmissionError",
    "REQUESTS",
    "RULING",
    "SCHEME_GRELL_FREITAS",
    "SCHEME_OFF",
    "SCHEME_NTIEDTKE",
    "EVERY_STEP_SCHEMES",
    "VARIABLE_MESH_BREAKAGE",
    "VARIABLE_MESH_SPREAD",
    "cell_spacing_m",
    "constructor_scheme",
    "cu_physics_for_constructor_scheme",
    "register_engine_scheme",
    "convection_decision",
    "convection_for_spacing",
    "default_convection_scheme",
    "finest_spacing",
    "gf_ishallow",
    "label",
    "label_for_constructor_scheme",
]
