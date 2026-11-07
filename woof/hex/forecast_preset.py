"""Named forecast presets, as DATA.

A preset is one row naming what a forecast run changes relative to the
proven configuration: which physics suite the column seam runs, and how
often the surface layer, the land-surface model and the PBL are called.
``woof hex forecast --preset NAME`` selects a row; the door turns the row
into the knobs it already has (``--pbl-cadence``), so a preset is never a
code path of its own and adding one is adding a row.

WHAT A PRESET HOLDS, AND WHAT IT DOES NOT.  A row holds only choices that
change the forecast.  A change that moves no output byte is not a preset
choice: it is simply on, for every row (the per-step physics rollback
export, which a run declines unless ``--stop-on-refusal`` keeps a refused
step's boundary; see :mod:`woof.hex.cuda_arwen_physics_v841`).  Holding a
free saving behind a preset would make the default slower for no reason.

THE PHYSICS SUITE IS DECLARED, AND CHECKED AGAINST THE ENGINE.  The pinned
engine's column seam (``woof.core.mpas_column_batch``) builds its physics
configuration with the radiation, land-surface, surface-layer and PBL
selections written as literals, and publishes no constructor argument for
any of them.  Every row therefore declares :data:`ENGINE_COLUMN_SUITE`, and
:func:`resolve_preset` refuses a row that declares anything else.  THE
BREAKAGE THIS PREVENTS: a row naming RTE-RRTMGP or Noah would run legacy
RRTMG and Noah-MP while its receipt said otherwise.  When the engine's seam
takes the suite as data, a row naming another suite becomes a table edit
here plus that engine pin, and nothing else.

THE CADENCE IS A HOLD, NOT A NUMBER OF SECONDS FOR EVERY MESH.  A held
cadence must be a whole number of model steps (:mod:`woof.hex.pbl_cadence`
refuses anything else), and the meshes this program runs step at 5 s to
120 s.  A row therefore names the longest hold it accepts, and
:func:`pbl_cadence_request` turns that into the largest whole number of
this mesh's steps that fits inside it: 30 s is six steps at dt 5 s, one
step (the weld) at dt 20 s or longer.  A row never holds the stack longer
than it says.

WHY THE DEFAULT IS ``reference``.  A row becomes the default only when it
scores at least as well against observations as the row it replaces.  The
``fast`` row was graded on four 3 h HRRR-driven forecasts of the 937.5 m
point cull (docs/hex-point-hrrr.md, the forecast presets section): it
takes about 6 per cent less time per step and its reflectivity and
precipitation threshold scores are indistinguishable from ``reference``,
but its near-surface forecast is worse: 10 m wind RMSE against ASOS is
higher on all four cases (0.05 m/s on average, an interval that excludes
zero), 2 m dewpoint and temperature RMSE are higher on three of four, and
hourly precipitation RMSE against Stage-IV is slightly higher.  So it
stays selectable and ``reference`` stays the default.

RTE-RRTMGP, the other physics saving, was measured through this seam as a
measurement arm outside the shipped options (docs/hex-point-hrrr.md): its
radiation call costs about a twentieth of legacy RRTMG's on the point cull.
It becomes a row here once the engine's seam takes the radiation variant
as an argument.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

from .errors import ConfigurationRefusal


__all__ = [
    "DEFAULT_PRESET",
    "ENGINE_COLUMN_SUITE",
    "FAST_PRESET",
    "PRESETS",
    "REFERENCE_PRESET",
    "ForecastPreset",
    "pbl_cadence_request",
    "preset_names",
    "resolve_preset",
]


#: The column physics the pinned engine seam constructs, whatever a caller
#: asks for: ``MpasColumnBatchPhysics.__init__`` builds its ``RunConfig`` with
#: ``ra_physics=4, ra_rrtmg_variant=RRTMG_VARIANT_LEGACY``,
#: ``sf_surface_physics=4``, ``sf_sfclay_physics=1`` and ``bl_pbl_physics=1``
#: (gpuwm 2.8.0, ``woof/core/mpas_column_batch.py``), one of the sixteen
#: engine files :mod:`woof.hex.cuda_arwen_physics_v841` pins by SHA-256.
ENGINE_COLUMN_SUITE: Mapping[str, str] = MappingProxyType(
    {
        "radiation": "rrtmg_legacy",
        "land_surface": "noahmp",
        "surface_layer": "revised_mo",
        "boundary_layer": "ysu",
    }
)


@dataclass(frozen=True, slots=True)
class ForecastPreset:
    """One named forecast preset."""

    #: What the row is called on the command line.  A capability name, never
    #: a case, a site or a customer.
    name: str
    #: One line for ``--help`` and the receipt.
    summary: str
    #: The physics suite the column seam runs; see :data:`ENGINE_COLUMN_SUITE`.
    suite: Mapping[str, str]
    #: The longest the surface layer, the land-surface model and the PBL may
    #: be held between calls, in seconds.  ``None`` is the weld: every model
    #: step, the proven configuration.
    surface_pbl_hold_seconds: float | None
    #: Where the row's measurement and its verdict against observations are
    #: written down.
    evidence: str
    #: ``physics.mode`` (:mod:`woof.hex.correct_mode`) the row runs when the
    #: command line names none.  Every shipped row says ``"correct"``: owner
    #: ruling 2026-10-13, correct mode is the standard (the mode key;
    #: fixed means default).  ``--physics-mode exact`` still selects the matched
    #: MPAS/WRF arithmetic, byte for byte, and every parity or identity proof
    #: names it explicitly.  Not in :meth:`receipt`: the run's own mode is
    #: recorded by the door (and only in correct mode, so an exact receipt is
    #: the one written before the key existed).
    physics_mode: str = "correct"

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip().lower():
            raise ValueError(
                f"a preset name is a lowercase word with no surrounding space, got {self.name!r}"
            )
        if set(self.suite) != set(ENGINE_COLUMN_SUITE):
            raise ValueError(
                f"preset {self.name!r} must declare exactly the suite slots "
                f"{sorted(ENGINE_COLUMN_SUITE)}, got {sorted(self.suite)}"
            )
        if self.physics_mode not in ("exact", "correct"):
            raise ValueError(
                f"preset {self.name!r} physics_mode must be 'exact' or 'correct', "
                f"got {self.physics_mode!r}"
            )
        hold = self.surface_pbl_hold_seconds
        if hold is not None and not (math.isfinite(float(hold)) and float(hold) > 0.0):
            raise ValueError(
                f"preset {self.name!r} surface_pbl_hold_seconds must be a positive "
                f"number of seconds or None (the weld), got {hold!r}"
            )

    def receipt(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary,
            "suite": dict(self.suite),
            "surface_pbl_hold_seconds": self.surface_pbl_hold_seconds,
            "evidence": self.evidence,
        }


REFERENCE_PRESET = ForecastPreset(
    name="reference",
    summary=(
        "the proven configuration: legacy RRTMG, Noah-MP, revised MO and YSU, "
        "with the surface layer, Noah-MP and YSU called every model step"
    ),
    suite=ENGINE_COLUMN_SUITE,
    surface_pbl_hold_seconds=None,
    evidence="docs/hex-point-hrrr.md, the forecast presets section",
    physics_mode="correct",
)

FAST_PRESET = ForecastPreset(
    name="fast",
    summary=(
        "the same schemes, with the surface layer, Noah-MP and YSU held for "
        "up to 30 s between calls (six steps at dt 5 s; the weld where dt is "
        "20 s or longer); about 6 per cent less time per step, and a worse "
        "near-surface forecast against ASOS (10 m wind, 2 m dewpoint and "
        "temperature) on the four cases it was graded on"
    ),
    suite=ENGINE_COLUMN_SUITE,
    surface_pbl_hold_seconds=30.0,
    evidence="docs/hex-point-hrrr.md, the forecast presets section",
    physics_mode="correct",
)

#: Every row, by name.
PRESETS: Mapping[str, ForecastPreset] = MappingProxyType(
    {row.name: row for row in (REFERENCE_PRESET, FAST_PRESET)}
)

#: The row a run gets when it names none.  A constant, not "the first row".
DEFAULT_PRESET = REFERENCE_PRESET.name


def preset_names() -> tuple[str, ...]:
    """Every preset name, the default first."""

    return (DEFAULT_PRESET,) + tuple(
        sorted(name for name in PRESETS if name != DEFAULT_PRESET)
    )


def resolve_preset(name: str | None = None) -> ForecastPreset:
    """The row a run selects, or a refusal naming the rows that exist."""

    key = DEFAULT_PRESET if name is None else str(name).strip().lower()
    row = PRESETS.get(key)
    if row is None:
        raise ConfigurationRefusal(
            "preset",
            name,
            f"no forecast preset carries that name.  Presets: {list(preset_names())}",
            f"--preset in {list(preset_names())}",
        )
    moved = {
        slot: value
        for slot, value in row.suite.items()
        if ENGINE_COLUMN_SUITE[slot] != value
    }
    if moved:
        raise ConfigurationRefusal(
            "preset",
            row.name,
            (
                f"preset {row.name!r} declares {moved}, and the pinned engine's "
                "column seam builds "
                f"{ {slot: ENGINE_COLUMN_SUITE[slot] for slot in moved} } "
                "unconditionally (woof/core/mpas_column_batch.py takes no "
                "argument for it).  The run would execute the engine's suite "
                "under this row's name"
            ),
            "an engine whose column seam takes the suite as a constructor argument",
        )
    return row


def pbl_cadence_request(row: ForecastPreset, dt_seconds: float) -> str:
    """The ``--pbl-cadence`` value ``row`` asks for at this timestep.

    ``"auto"`` (the weld) when the row holds nothing or when not even two
    steps fit inside its hold; otherwise the largest whole number of steps
    that does, in seconds.
    """

    dt = float(dt_seconds)
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError(f"dt_seconds must be a positive number, got {dt_seconds!r}")
    hold = row.surface_pbl_hold_seconds
    if hold is None:
        return "auto"
    # The small allowance keeps 30/5 at six steps in binary arithmetic; it is
    # far below any timestep this program admits.
    steps = int(math.floor(float(hold) / dt + 1.0e-9))
    if steps <= 1:
        return "auto"
    return f"{steps * dt:g}"
