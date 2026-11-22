"""Timestep anchors for the v8.4.1 column-physics lane: evidence, not a gate.

RETIRED AS A GATE (2026-10-10).  Nothing in this module refuses a
run any more for lacking an anchor or for running at cadences an anchor was
not earned at.  ``anchor_record`` answers whether a configuration matches a
registered row, and the run writes that answer into its receipt
(``dt_admission.anchor``); the refusals that remain name a concrete
breakage -- a physics cadence that is not a whole number of steps, which the
sealed constructor cannot schedule, and a timestep whose multiples are not
exact in binary64, which stops a run on the clock check
(``require_cadences``).  Stability is measured from the mesh at bind
(:mod:`woof.hex.timestep_admission`).  The history below is kept because it
is the record of what the registered rows measured.


The frozen column-physics lane executes at exactly one model timestep.
``V841MpasColumnPhysicsConfig.validate`` used to say so as a literal --
``config_dt`` had to equal ``120.0`` or the configuration was refused
"because the matched v8.4.1 real-x4 column-physics lane is exact" -- and the
same literal pinned ``config_bldt_seconds`` and ``config_cudt_seconds``
beside it.

The premise of that refusal is **unproven at other dt, not wrong at other
dt**, and the distinction matters because the two have different remedies.
Nothing in the dycore is 120-shaped: the same v8.4.1 dry dycore is already
run and cross-checked CPU-against-GPU at ``config_dt`` 1.0 s
(``tools/run_cuda_v841_numeric_ruler.py``), 3600.0 s
(``tools/run_cuda_jw_dualrun_trust.py``, 24 steps, two independent device
arms) and arbitrary dt (``tools/run_cuda_v841_jw_dualrun.py``).  What is
120-shaped is the EVIDENCE: every archived receipt of the column-physics
lane, and the one native MPAS-A v8.4.1 reference this program holds, were
integrated at 120 s.

So this module is the same shape as
:mod:`woof.hex.cuda_backend.arch_admission` and
:mod:`woof.hex.cuda_backend.regional_admission`: a flat constant becomes a
registry of EARNED anchors.  A timestep with no anchor is refused by name
with the remedy; a timestep with one names the evidence that admits it.

An anchor is earned, not declared.  Each entry names, in this repository:

* a **schedule receipt** -- the host-derivable half, minted by
  ``tools/mint_dt_anchor.py``: the RK stage timesteps and acoustic sub-step
  timestep this dt produces, the resolved physics cadence step counts, the
  WSM6 minor-loop count and its FP32 SR roundoff envelope, and the clock
  closure proof (that the run's step endpoints are exact in float64 so the
  post-RK endpoint equality can never fail on accumulation); and
* an **integration anchor** -- real forecasts at this dt on named hardware,
  finite at every step, with two runs byte-identical under masked digests.

A third field is deliberately nullable and deliberately named:

* a **native reference** -- a native MPAS-A v8.4.1 integration of the same
  case at this dt.  120 s has one (24 ranks, 30 steps, masked-content
  history digests; ``tools/run_cuda_v841_full_physics_x4.py::AUTHORITY_PINS``).
  No other timestep can have one without a fresh native run, and the port's
  standing position retired whole-model parity anyway -- so an anchor
  without a native reference is admissible, and says so, rather than being
  quietly conflated with one that has it.

An anchor certifies a CONFIGURATION at a timestep, not a timestep alone.
The registry is keyed by the pair ``(dt, cumulus selection)``, because the
2026-08-26 ruling that convection is off below 3 km created a second real
configuration at exactly the timesteps a fine mesh declares.  How often
a scheme is CALLED is part of what an anchor's forecasts measured -- WRF
pins ``cudt = 0`` for Grell-Freitas, so the cumulus cadence IS the timestep
-- and switching the scheme off changes that from "every step" to "never".
A convection-off run reading a Grell-Freitas row would be quoting
measurements of a forcing it does not apply.  See
:mod:`woof.hex.convection_admission` for the ruling and the threshold.

The surface/PBL cadence is the third fragment of the key, and it is the one
a run may SELECT (``--pbl-cadence``, :mod:`woof.hex.pbl_cadence`).  A held
cadence at an anchored ``(dt, cumulus)`` is admitted through a DERIVED row
(:func:`derived_held_cadence_anchor`, 2026-09-13): the dycore half of the
anchor -- the outer step, its RK schedule, its clock closure and its
byte-identical dual run -- is a property of the timestep and is unchanged
by how often the seam calls the surface/PBL stack, so it is the welded
row's own; the host half is re-minted for the held cadence at admission;
the physics band is stamped NOT MEASURED at the held cadence rather than
borrowed from the welded row.  The derived row occupies its own key, so the
breakage the key fragment prevents (a held run quoting the welded band)
cannot recur, and an EARNED held row -- minted and registered -- still wins
the exact lookup.  A held cadence at a timestep whose welded configuration
holds no anchor is refused exactly as the welded run would be.

Adding a row is TABLE WORK once the evidence exists -- and it is not an
agent's call.  Moving the frozen lane off 120 s is a ruling; this module is
the mechanism that ruling would use, not the ruling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping


class DtAdmissionError(RuntimeError):
    """A model timestep holds no anchor, or contradicts the one it names."""


#: The timestep the frozen v8.4.1 column-physics lane was proven at, and the
#: timestep of the only native MPAS-A v8.4.1 reference this program holds.
PROVEN_DT_SECONDS: float = 120.0

#: The physics radiation cadence the frozen lane holds fixed at every dt.
#: A dt must divide it exactly; see :func:`schedule_receipt`.
RADIATION_CADENCE_SECONDS: float = 600.0

#: WSM6's internal maximum cloud-microphysics timestep (WRF ``dtcldcr``).
#: This is a SCHEME constant, not the model timestep: the minor-loop count is
#: ``max(round(dt / 120), 1)``, so every dt at or below 179 s runs exactly
#: one minor loop, as 120 s does.  It is quoted here because it is the one
#: place a literal 120 in the physics path is genuinely dt-dependent physics
#: rather than proof bookkeeping.
WSM6_MINOR_DT_SECONDS: float = 120.0


@dataclass(frozen=True, slots=True)
class DtAnchor:
    """The evidence that admits one model timestep for the frozen lane."""

    dt_seconds: float
    radiation_seconds: float
    surface_pbl_seconds: float
    cumulus_seconds: float | None
    cumulus_scheme: str | None
    meshes: tuple[str, ...]
    card: str
    admitted_on: str
    schedule_receipt: str
    integration_anchor: str
    native_reference: str | None
    basis: str
    #: What the measured physics band DID, against a control at the proven
    #: timestep on the same card, mesh and init.  Required, with no default,
    #: so a row cannot be added without answering it -- and it leads with a
    #: one-word verdict so a reader scanning receipts sees the answer before
    #: the reasoning.  MEASURED 2026-08-26: two of the four timesteps earned
    #: that day track their control and two diverge from it, and an anchor
    #: that recorded only "two byte-identical forecasts" would have said the
    #: same thing about all four.  Determinism and physical plausibility are
    #: different questions and this field is the second one.
    physics_health: str
    #: The registry key of the WELDED row this row was derived from, when
    #: it admits a held surface/PBL cadence at that row's timestep
    #: (:func:`derived_held_cadence_anchor`); ``None`` for an earned row.
    #: Carried so a receipt reading a derived row can never mistake it for
    #: a registered one.
    derived_from: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dt_seconds": self.dt_seconds,
            "radiation_seconds": self.radiation_seconds,
            "surface_pbl_seconds": self.surface_pbl_seconds,
            "cumulus_seconds": self.cumulus_seconds,
            "cumulus_scheme": self.cumulus_scheme,
            "meshes": list(self.meshes),
            "card": self.card,
            "admitted_on": self.admitted_on,
            "schedule_receipt": self.schedule_receipt,
            "integration_anchor": self.integration_anchor,
            "native_reference": self.native_reference,
            "basis": self.basis,
            "physics_health": self.physics_health,
            "derived_from": self.derived_from,
        }

    @property
    def derived(self) -> bool:
        """Whether this row admits a held cadence on the welded row's dycore evidence."""

        return self.derived_from is not None

    @property
    def physics_health_verdict(self) -> str:
        """The leading one-word verdict of :attr:`physics_health`."""

        return self.physics_health.split(None, 1)[0].rstrip(":.,").upper()


def cumulus_key(cumulus_scheme: str | None) -> str:
    """The registry key fragment naming one cumulus selection."""

    if cumulus_scheme is None:
        return "off"
    scheme = str(cumulus_scheme)
    if not scheme:
        raise DtAdmissionError(
            "a cumulus selection is a scheme name or None (convection off); "
            "the empty string is neither"
        )
    return scheme


def surface_pbl_key(
    dt_seconds: float, surface_pbl_seconds: float | None = None
) -> str:
    """The registry key fragment naming one surface/PBL cadence.

    ``None`` and "exactly dt" are the same statement -- the proven weld,
    ``config_bldt_seconds == config_dt`` -- and normalise to ``"dt"`` so the
    rows earned before the cadence was holdable keep a readable key.  Any
    other cadence is spelled by its exact ``repr``.
    """

    dt = float(dt_seconds)
    if not math.isfinite(dt) or dt <= 0.0:
        raise DtAdmissionError(
            f"a model timestep must be finite and positive, got {dt_seconds!r}"
        )
    if surface_pbl_seconds is None:
        return "dt"
    seconds = float(surface_pbl_seconds)
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise DtAdmissionError(
            f"a surface/PBL cadence must be finite and positive, got "
            f"{surface_pbl_seconds!r}"
        )
    return "dt" if seconds == dt else f"{seconds!r}"


def dt_key(
    dt_seconds: float,
    cumulus_scheme: str | None = "gf",
    surface_pbl_seconds: float | None = None,
) -> str:
    """The registry key for one CONFIGURATION at one timestep.

    Keyed by the exact float's ``repr``: two timesteps that differ in the
    last bit are different configurations and must not share an anchor.

    Keyed by the cumulus selection beside it for the same reason, and it is
    not a refinement -- it is the difference between two runs that share
    every other number.  How often a scheme is CALLED is part of what an
    anchor's forecasts measured, and switching the scheme off changes that
    from "every step" to "never".  It was ruled on 2026-08-26 that convection
    is off below 3 km, so the lane now has two real configurations at the
    small timesteps and each earns its own evidence.  A convection-off run
    borrowing a Grell-Freitas anchor would be quoting measurements of a
    forcing it does not apply.

    Keyed by the surface/PBL cadence for the third time on the same
    argument.  ``config_bldt_seconds`` is welded to dt exactly as ``cudt``
    is, so a smaller timestep calls the surface layer, the land-surface
    model and the PBL proportionally more often -- 30 times an hour at the
    proven 120 s, 720 at 5 s.  THE BREAKAGE THIS FRAGMENT PREVENTS: without
    it a run holding that cadence would occupy the SAME slot as the welded
    row at the same ``(dt, cumulus)`` and silently replace it, after which
    every ordinary welded run at that timestep would be admitted against a
    band measured at one twenty-fourth of its own call rate.  MEASURED
    (2026-08-26, evidence/convection-off-20260826/RECEIPT.md): the
    registered 5 s convection-off row certifies |w| mean
    8.19/49.05/65.83/81.07 m/s with the stack called 720 times an hour.  See
    :mod:`woof.hex.pbl_cadence`.
    """

    value = float(dt_seconds)
    if not math.isfinite(value) or value <= 0.0:
        raise DtAdmissionError(
            f"a model timestep must be finite and positive, got {dt_seconds!r}"
        )
    return (
        f"{value!r}|cumulus={cumulus_key(cumulus_scheme)}"
        f"|surface_pbl={surface_pbl_key(value, surface_pbl_seconds)}"
    )


#: The mint each non-120 anchor below was earned by.  One init, one mesh, one
#: card per row, three arms per row: a control at the proven timestep and two
#: candidate arms at the timestep being earned.
_MINT_CAMPAIGN = "evidence/dt-anchors-20260826/RECEIPT.md"
_MINT_INIT = (
    "native MPAS-A v8.4.1 init_atmosphere, GFS 2026-08-12 06Z, "
    "sha256 2d5e41f3db86dda4eda4aab86f2fc94c7046eb226eaee7180dbf0272faf10fcd"
)

#: Timesteps holding a verified anchor.  This registry records what has been
#: EARNED; every other value is refused by name below.
#:
#: 120 s was the only row until 2026-08-26, when it was ruled that the frozen
#: lane stops being pinned to it and four more were minted against that ruling --
#: on the already-registered x1.40962, whose own Courant limit measures
#: 698.95 s, so every value below sits far beneath it and every finer mesh
#: inherits the anchor.  An anchor is a property of the TIMESTEP, not the
#: mesh, which is why the cheap place to earn one is the smallest registered
#: mesh rather than the large ones it unblocks.
#:
#: Only 120 s carries a native reference and only it ever can: the one native
#: MPAS-A v8.4.1 integration this program holds was run at 120 s.  The four
#: rows added below say ``native_reference=None`` rather than being quietly
#: conflated with it.
ADMITTED_TIMESTEPS: Mapping[str, DtAnchor] = MappingProxyType(
    {
        dt_key(120.0, "gf"): DtAnchor(
            dt_seconds=120.0,
            radiation_seconds=600.0,
            surface_pbl_seconds=120.0,
            cumulus_seconds=120.0,
            cumulus_scheme="gf",
            meshes=("x4.163842", "x1.40962", "u96.64002"),
            card=(
                "RTX 5090 (170 SM), RTX 5070 Ti (70 SM) and "
                "RTX 3080 (68 SM, desktop, sm_86 tier)"
            ),
            admitted_on="2026-08-26",
            schedule_receipt=(
                "evidence/dt-admission-20260826/schedule-receipt-dt120.json"
            ),
            integration_anchor=(
                "evidence/obs-referee-283/RECEIPT.md, "
                "evidence/restart-step16-327/, "
                "evidence/sm86-tier-20260825/RECEIPT.md"
            ),
            native_reference=(
                "tools/run_cuda_v841_full_physics_x4.py::AUTHORITY_PINS -- "
                "MPAS-A v8.4.1, x4.163842, 24 MPI ranks, config_dt 120 s, "
                "30 steps, masked-content history digests"
            ),
            basis=(
                "the timestep every archived column-physics receipt was "
                "integrated at.  Its integration half is measured twice over: "
                "the obs-referee suite ran the four cases twice and all seven "
                "outputs match, the x4 restart gate produced a restarted "
                "history byte-identical to the uninterrupted one, and the "
                "sm_86 tier ran the same 2 h x1.40962 forecast twice "
                "byte-identically under masked digests on a third card.  Its "
                "native half is the one native MPAS-A v8.4.1 integration this "
                "program holds, which was run at this dt"
            ),
            physics_health=(
                "REFERENCE.  This is the band every other anchor is read "
                "against: a 120 s control on the candidate's own card, mesh "
                "and init.  On x1.40962 from the 2026-08-12 06Z native init "
                "it runs |w| mean 1.15/1.17/1.20/1.48 m/s over four half-hour "
                "windows, |w| max 1.680, theta_m max 892.65 K -- and it "
                "reproduces to every printed digit on two different cards"
            ),
        ),
        dt_key(120.0, "ntiedtke"): DtAnchor(
            dt_seconds=120.0,
            radiation_seconds=600.0,
            surface_pbl_seconds=120.0,
            cumulus_seconds=120.0,
            cumulus_scheme="ntiedtke",
            meshes=("x1.40962",),
            card="RTX 5090 (170 SM, sm_120) and RTX 4090 (128 SM, sm_89)",
            admitted_on="2026-10-05",
            schedule_receipt=(
                "evidence/dt-anchor-ntiedtke-20261005/"
                "schedule-receipt-dt120-ntiedtke.json"
            ),
            integration_anchor=(
                "evidence/dt-anchor-ntiedtke-20261005/"
                "anchor-dt120-ntiedtke-sm120.json, "
                "evidence/dt-anchor-ntiedtke-20261005/"
                "anchor-dt120-ntiedtke-sm89.json, "
                "evidence/dt-anchor-ntiedtke-20261005/RECEIPT.md"
            ),
            native_reference=None,
            basis=(
                "the 2026-10-05 coarse-mesh default (hexcore."
                "convection_admission.NTIEDTKE_AT_OR_ABOVE_M).  Two 72 h "
                "x1.40962 forecasts from GFS 2026-10-01 00Z in separate "
                "processes on two RTX 5090 cards wrote 13 of 13 "
                "byte-identical history frames, finite in every frame, and "
                "a second pair on one RTX 4090 did the same.  New Tiedtke is "
                "called every step, as Grell-Freitas is, so the cadences are "
                "the proven 120 s row's.  No native reference: the frozen "
                "native integration ran Grell-Freitas"
            ),
            physics_health=(
                "TRACKS THE CONTROL, and scores better than it against "
                "observations.  Against the 120 s Grell-Freitas control on "
                "the same mesh and init, the six-hourly |w| mean runs "
                "0.011-0.015 m/s against 0.011-0.018, |w| max over 72 h "
                "4.59 m/s against 6.48, theta max 937.53 K against 937.49.  "
                "Against radiosondes the tropical 250 hPa bias/RMS is "
                "-0.71/1.04 K against -2.03/2.24 K at 72 h, and a second "
                "case (GFS 2026-01-15 00Z) gives -0.96/1.52 K against "
                "-2.57/2.97 K"
            ),
        ),
        dt_key(100.0, "gf"): DtAnchor(
            dt_seconds=100.0,
            radiation_seconds=600.0,
            surface_pbl_seconds=100.0,
            cumulus_seconds=100.0,
            cumulus_scheme="gf",
            meshes=("x1.40962",),
            card="RTX 5090 (170 SM)",
            admitted_on="2026-08-26",
            schedule_receipt=(
                "evidence/dt-anchors-20260826/dt100/schedule-receipt-dt100.json"
            ),
            integration_anchor=(
                "evidence/dt-anchors-20260826/dt100/anchor-dt100.json, "
                "evidence/dt-anchors-20260826/RECEIPT.md"
            ),
            native_reference=None,
            basis=(
                "72 steps per arm over 2 h, two arms in separate processes, "
                "finite at every step in both and all five history frames "
                "identical.  Health measured against a 120 s control on the "
                "SAME card, mesh and init: every band field within a few parts "
                "in 1e4 -- theta_m max -0.138 K, qv max -1.1e-5, exner min "
                "-2.5e-5, rho min -9.6e-7 -- and the half-hour vertical-"
                "velocity trend tracks the control (1.15/1.13/1.42/1.55 m/s "
                "against 1.15/1.17/1.20/1.48).  Grell-Freitas is called 36 "
                "times an hour here against the proven 30.  Largest value at "
                "or below the 103.67 s Courant limit of the finest registered "
                "graded mesh that also divides the 600 s radiation cadence "
                "exactly and closes its clock in binary64.  Unblocks "
                "v16.66.195630 (195,630 cells, 16.5 km).  No native reference "
                "exists at this timestep and none can"
            ),
            physics_health=(
                "TRACKS.  Every band field within parts in 1e4 of the 120 s "
                "control on the same card, mesh and init: theta_m max "
                "-0.138 K, qv max -1.1e-5, exner min -2.5e-5, rho min "
                "-9.6e-7.  |w| max 1.805 against the control's 1.680, and the "
                "half-hour |w| means 1.15/1.13/1.42/1.55 track the control's "
                "1.15/1.17/1.20/1.48 with no trend away from it"
            ),
        ),
        dt_key(75.0, "gf"): DtAnchor(
            dt_seconds=75.0,
            radiation_seconds=600.0,
            surface_pbl_seconds=75.0,
            cumulus_seconds=75.0,
            cumulus_scheme="gf",
            meshes=("x1.40962",),
            card="RTX 5070 Ti (70 SM)",
            admitted_on="2026-08-26",
            # The evidence tree names this arm's directory for the lab host it ran
            # on; host-a is that host's neutral label in shipped text.
            schedule_receipt=(
                "evidence/dt-anchors-20260826/dt75-host-a/schedule-receipt-dt75.json"
            ),
            integration_anchor=(
                "evidence/dt-anchors-20260826/dt75-host-a/anchor-dt75.json, "
                "evidence/dt-anchors-20260826/RECEIPT.md"
            ),
            native_reference=None,
            basis=(
                "96 steps per arm over 2 h, two arms in separate processes, "
                "finite at every step in both and all five history frames "
                "identical.  Against a 120 s control on the same card, mesh "
                "and init: theta_m max -0.456 K, qv max -2.2e-5, exner min "
                "-7.4e-5, |w| max 2.092 against 1.680 m/s, and the half-hour "
                "trend tracks the control.  Grell-Freitas is called 48 times "
                "an hour against the proven 30.  600/7 = 85.714... clears the "
                "95.84 s Courant limit of the finest registered graded mesh "
                "but is inexact in binary64, so 600/8 is the largest value "
                "that satisfies all three constraints.  Unblocks "
                "v15.60.224197 and v15.60.224210.  No native reference exists "
                "at this timestep and none can"
            ),
            physics_health=(
                "TRACKS.  theta_m max -0.456 K, qv max -2.2e-5, exner min "
                "-7.4e-5 against the 120 s control on the same card, mesh and "
                "init.  |w| max 2.092 against 1.680, half-hour means "
                "1.14/1.21/1.88/1.47 against 1.15/1.17/1.20/1.48 -- one "
                "raised window, no trend away from the control"
            ),
        ),
        dt_key(20.0, "gf"): DtAnchor(
            dt_seconds=20.0,
            radiation_seconds=600.0,
            surface_pbl_seconds=20.0,
            cumulus_seconds=20.0,
            cumulus_scheme="gf",
            meshes=("x1.40962",),
            card="RTX 5070 Ti (70 SM)",
            admitted_on="2026-08-26",
            schedule_receipt=(
                "evidence/dt-anchors-20260826/dt20/schedule-receipt-dt20.json"
            ),
            integration_anchor=(
                "evidence/dt-anchors-20260826/dt20/anchor-dt20.json, "
                "evidence/dt-anchors-20260826/RECEIPT.md"
            ),
            native_reference=None,
            basis=(
                "360 steps per arm over 2 h, two arms in separate processes, "
                "finite at every step in both and all five history frames "
                "identical -- so the integration is deterministic and stable.  "
                "THE BAND IS NOT QUIET AND THE ANCHOR SAYS SO: against a 120 s "
                "control on the same card, mesh and init, the vertical-"
                "velocity mean per half-hour climbs monotonically "
                "1.99/3.48/4.11/5.53 m/s against 1.15/1.17/1.20/1.48, is still "
                "climbing at the end of the arm, and |w| max reaches 7.511 "
                "against 1.680; theta_m max stops rising, 887.54 K against "
                "892.65 K.  Grell-Freitas is called 180 times an hour against "
                "the proven 30.  The comparison is confounded by the mesh and "
                "unavoidably so: x1.40962 is 120 km and 20 s is 35x below its "
                "own 698.95 s Courant limit, a configuration nobody runs for "
                "weather, while 20 s belongs with a 3 km swath where several "
                "m/s is ordinary.  Whether the growth is GF's call rate or "
                "resolved dynamics is NOT MEASURED; what settles it is the "
                "same trend on a mesh whose Courant limit is near this "
                "timestep, refereed by obs-skill.  This anchor certifies that "
                "the timestep integrates finitely and deterministically at the "
                "cadences named here, and nothing more.  No native reference "
                "exists at this timestep and none can"
            ),
            physics_health=(
                "DIVERGES.  The |w| mean climbs monotonically over the four "
                "half-hour windows -- 1.99, 3.48, 4.11, 5.53 m/s against the "
                "120 s control's 1.15, 1.17, 1.20, 1.48 -- is still climbing "
                "at the end of the 2 h arm, and reaches |w| max 7.511 against "
                "1.680, while theta_m max stops rising (887.54 K against "
                "892.65 K).  Finite at every one of 360 steps and byte-"
                "identical across arms, so it is a DIFFERENT SOLUTION and not "
                "an unstable one.  Cause NOT MEASURED: this mesh is 120 km and "
                "this timestep is 35x below its own 698.95 s Courant limit, so "
                "the comparison cannot separate GF's 6x call rate from "
                "resolved dynamics.  What settles it is the same trend on a "
                "mesh whose Courant limit is near this timestep -- a 3 km "
                "swath -- refereed by obs-skill"
            ),
        ),
        dt_key(5.0, "gf"): DtAnchor(
            dt_seconds=5.0,
            radiation_seconds=600.0,
            surface_pbl_seconds=5.0,
            cumulus_seconds=5.0,
            cumulus_scheme="gf",
            meshes=("x1.40962",),
            card="RTX 5090 (170 SM)",
            admitted_on="2026-08-26",
            schedule_receipt=(
                "evidence/dt-anchors-20260826/dt5/schedule-receipt-dt5.json"
            ),
            integration_anchor=(
                "evidence/dt-anchors-20260826/dt5/anchor-dt5.json, "
                "evidence/dt-anchors-20260826/RECEIPT.md"
            ),
            native_reference=None,
            basis=(
                "1,440 steps per arm over 2 h, two arms in separate processes, "
                "finite at every step in both and all five history frames "
                "identical.  Grell-Freitas is called 720 times an hour here -- "
                "24x its proven call rate -- because WRF pins cudt=0 for "
                "cu_physics=3 and no configuration can relax it.  The band "
                "against the 120 s control on the same card, mesh and init is "
                "recorded in the campaign receipt and carries the same "
                "confound the 20 s row names: 5 s is 140x below x1.40962's own "
                "698.95 s Courant limit and belongs with a 750 m swath.  This "
                "anchor certifies that the timestep integrates finitely and "
                "deterministically at the cadences named here, and nothing "
                "more.  No native reference exists at this timestep and none "
                "can"
            ),
            physics_health=(
                "DIVERGES SEVERELY, and this row must not be read as a "
                "certificate that 5 s produces good weather on any mesh.  The "
                "|w| mean over the four half-hour windows runs 12.0, 49.6, "
                "73.5, 87.5 m/s against the 120 s control's 1.15, 1.17, 1.20, "
                "1.48, still climbing at the end of the arm, with |w| max "
                "102.67 against 1.680 -- 61x -- and theta_m max falling to "
                "885.60 K against 892.65 K.  A 102 m/s updraft is not "
                "physical on 120 km cells and the grid cannot resolve one, so "
                "the measurement points AWAY from resolved dynamics and "
                "TOWARD a forcing that scales with call count rather than "
                "with elapsed time; GF runs 720 times an hour here against "
                "the proven 30.  That attribution is NOT MEASURED and this "
                "row does not assert it.  The run is finite at every one of "
                "1,440 steps and byte-identical across arms, which is what "
                "this anchor certifies and is the whole of what it "
                "certifies.  MEASURED SINCE (2026-08-26, RTX 5070 Ti, "
                "evidence/convection-off-20260826/): the A/B this row asked "
                "for was run once the convection ruling made it "
                "reachable, and GRELL-FREITAS IS NOT THE CAUSE.  With the "
                "closure never called, the same timestep on the same mesh "
                "and init still reaches |w| max 93.96 and half-hour means "
                "8.19/49.05/65.83/81.07 -- 91.4 % of this row's excess over "
                "the control survives with no convection at all, and "
                "theta_m max is 887.3353 K either way, to every printed "
                "digit.  The call-rate SHAPE of the hypothesis survives and "
                "only its subject is excluded: config_bldt_seconds is welded "
                "to dt exactly as cudt is, so the surface/PBL stack is also "
                "called 720 times an hour here against the proven 30.  What "
                "settles the rest, MEASURED SINCE (2026-08-26, RTX 5070 Ti, "
                "evidence/pbl-cadence-20260826/): the surface/PBL cadence "
                "was held at 120 s while dt shrank, and it is NOT the "
                "cause either -- at 20 s holding it removes 4.6 % of the "
                "excess, and at 5 s it makes the climb WORSE (|w| max "
                "134.55 against 81.77 over the identical 964 steps) and "
                "the run stops integrating at step 964 of 1,440.  Both "
                "per-call physics candidates that had a cadence knob are "
                "now excluded.  What is left and NOT MEASURED: WSM6, "
                "which has no cadence in the sealed constructor to hold, "
                "and the same trend on a 750 m swath where 5 s sits at "
                "the Courant limit instead of 140x below it"
            ),
        ),
        dt_key(20.0, None): DtAnchor(
            dt_seconds=20.0,
            radiation_seconds=600.0,
            surface_pbl_seconds=20.0,
            cumulus_seconds=None,
            cumulus_scheme=None,
            meshes=("x1.40962",),
            card="RTX 5070 Ti (70 SM)",
            admitted_on="2026-08-26",
            schedule_receipt=(
                "evidence/convection-off-20260826/dt20/schedule-receipt-dt20.json"
            ),
            integration_anchor=(
                "evidence/convection-off-20260826/dt20/anchor-dt20.json, "
                "evidence/convection-off-20260826/RECEIPT.md"
            ),
            native_reference=None,
            basis=(
                "It was ruled on 2026-08-26 that convection is switched off "
                "below 3 km, and 20 s is the timestep a 3 km swath declares.  "
                "360 steps per arm over 2 h, two arms in separate processes, "
                "finite at every step in both and all five history frames "
                "identical.  No cumulus closure is called at any step "
                "(stepcu = 0, no cumulus cadence at all), which is what "
                "makes this a different configuration from the 20 s "
                "Grell-Freitas row and not a refinement of it.  Earned on "
                "the SAME card, mesh and init as that row, so the pair is a "
                "clean A/B in which only the cumulus selection moves.  No "
                "native reference exists for this configuration and none can"
            ),
            physics_health=(
                "DIVERGES.  The |w| mean climbs monotonically over the four "
                "half-hour windows -- 1.16, 2.43, 3.21, 3.94 m/s against the "
                "120 s control's 1.15, 1.17, 1.20, 1.48 -- is still climbing "
                "at the end of the 2 h arm, and reaches |w| max 5.016 "
                "against 1.680.  Switching the closure off removes 42.8 % of "
                "the 20 s Grell-Freitas row's excess over that control and "
                "LEAVES 57.2 %, so the closure is a contributor here and not "
                "the cause; theta_m max is 887.54 K with the scheme and "
                "887.54 K without it.  What is still driving the remainder "
                "is NOT MEASURED: this mesh is 120 km and this timestep is "
                "35x below its own 698.95 s Courant limit.  MEASURED "
                "SINCE (2026-08-26, RTX 5070 Ti, "
                "evidence/pbl-cadence-20260826/): config_bldt_seconds is "
                "welded to dt exactly as cudt is, so this row also calls "
                "the surface/PBL stack 180 times an hour against the "
                "proven 30 -- and HOLDING it at 120 s removes only 4.6 % "
                "of this row's excess over the control, leaving 95.4 %.  "
                "The held arm runs |w| max 4.861 against this row's 5.016 "
                "and half-hour means 1.19/2.48/3.27/3.93 against "
                "1.16/2.43/3.21/3.94, which is the same curve.  The "
                "surface/PBL call rate is NOT what is driving the "
                "remainder.  This anchor certifies that the timestep "
                "integrates finitely and deterministically with no "
                "convection at the cadences named here, and nothing more"
            ),
        ),
        dt_key(5.0, None): DtAnchor(
            dt_seconds=5.0,
            radiation_seconds=600.0,
            surface_pbl_seconds=5.0,
            cumulus_seconds=None,
            cumulus_scheme=None,
            meshes=("x1.40962",),
            card="RTX 5070 Ti (70 SM)",
            admitted_on="2026-08-26",
            schedule_receipt=(
                "evidence/convection-off-20260826/dt5/schedule-receipt-dt5.json"
            ),
            integration_anchor=(
                "evidence/convection-off-20260826/dt5/anchor-dt5.json, "
                "evidence/convection-off-20260826/RECEIPT.md"
            ),
            native_reference=None,
            basis=(
                "It was ruled on 2026-08-26 that convection is switched off "
                "below 3 km, and 5 s is the timestep a 750 m swath declares.  "
                "1,440 steps per arm over 2 h, two arms in separate "
                "processes, finite at every step in both and all five "
                "history frames identical.  No cumulus closure is called at "
                "any step.  This row and the 5 s Grell-Freitas row differ in "
                "exactly one input, which is what makes the pair able to "
                "answer an attribution question rather than only describe "
                "two runs -- and it answered one: see physics_health.  No "
                "native reference exists for this configuration and none can"
            ),
            physics_health=(
                "DIVERGES SEVERELY, and this row must not be read as a "
                "certificate that 5 s produces good weather on any mesh.  "
                "The |w| mean over the four half-hour windows runs 8.19, "
                "49.05, 65.83, 81.07 m/s against the 120 s control's 1.15, "
                "1.17, 1.20, 1.48, still climbing at the end of the arm, "
                "with |w| max 93.96 against 1.680.  THE POINT OF THIS ROW IS "
                "THAT IT LOOKS LIKE THE GRELL-FREITAS ROW WITH THE CLOSURE "
                "NEVER CALLED: 91.4 % of that row's excess over the control "
                "survives here, and theta_m max is 887.3353 K in both, to "
                "every printed digit.  Grell-Freitas' call rate is therefore "
                "NOT the cause of the runaway, which the 5 s GF row "
                "explicitly declined to assert and was right to.  What IS "
                "the cause is NOT MEASURED.  The call-rate shape of the "
                "hypothesis survives with a different subject: "
                "config_bldt_seconds is welded to dt exactly as cudt is, so "
                "the surface/PBL stack is called 720 times an hour here "
                "against the proven 30, and WSM6 runs every step too; "
                "radiation is on a fixed 600 s cadence and does not scale.  "
                "The mesh confound is untouched by this arm -- 5 s is still "
                "140x below x1.40962's own 698.95 s Courant limit.  "
                "MEASURED SINCE (2026-08-26, RTX 5070 Ti, "
                "evidence/pbl-cadence-20260826/): the surface/PBL "
                "cadence this row named as the next measurement was "
                "HELD at 120 s -- 30 calls an hour against this row's "
                "720 -- and the climb did not flatten.  Over the "
                "identical first 964 steps the held arm reaches |w| max "
                "134.55 against this row's 81.77, with quarterly means "
                "3.74/28.18/52.64/71.00 against 3.57/27.92/54.68/63.49, "
                "and it then STOPS INTEGRATING at step 964 of 1,440 -- "
                "reproducibly, in two processes -- where this "
                "configuration completed all 1,440.  Cutting the call "
                "rate 24-fold makes it worse, so the surface/PBL call "
                "rate is NOT the cause either.  What remains NOT "
                "MEASURED: WSM6, which runs every step and has no "
                "cadence in the sealed constructor to hold, and the "
                "resolved dynamics of a configuration 140x below its "
                "own Courant limit.  What settles those: the same trend "
                "on a sub-3-km swath where 5 s is the natural step, "
                "refereed by obs-skill"
            ),
        ),
    }
)


def admitted_timestep(
    dt_seconds: float,
    cumulus_scheme: str | None = "gf",
    surface_pbl_seconds: float | None = None,
) -> DtAnchor | None:
    """The anchor admitting this CONFIGURATION at ``dt_seconds``, or ``None``.

    The cumulus selection and the surface/PBL cadence are part of the
    question, not details checked afterwards: convection-off at a timestep
    is a different configuration from Grell-Freitas at the same timestep,
    and holding the surface/PBL cadence while dt shrinks is a third.  Each
    holds its own row.

    A held cadence with no registered row of its own is answered from the
    welded row at the same ``(dt, cumulus)`` through
    :func:`derived_held_cadence_anchor`; a held cadence that is not a whole
    number of steps RAISES :class:`DtAdmissionError` naming both numbers --
    a refusal, not an absence -- rather than answering ``None`` as though
    the timestep itself were unknown.
    """

    registered = registered_anchor(dt_seconds, cumulus_scheme, surface_pbl_seconds)
    if registered is not None:
        return registered
    try:
        held = surface_pbl_key(dt_seconds, surface_pbl_seconds) != "dt"
    except DtAdmissionError:
        return None
    if not held:
        return None
    return derived_held_cadence_anchor(dt_seconds, cumulus_scheme, surface_pbl_seconds)


def registered_anchor(
    dt_seconds: float,
    cumulus_scheme: str | None = "gf",
    surface_pbl_seconds: float | None = None,
) -> DtAnchor | None:
    """The REGISTERED row for this exact configuration, or ``None``.

    The exact lookup alone: no derivation.  This is the question a mint asks
    ("is there anything left to earn?") and the question the mesh-row
    override asks ("does this configuration hold a REAL anchor?"), and both
    would be answered wrongly by a derived row -- a held cadence would read
    as already earned and could never be minted.
    """

    try:
        key = dt_key(dt_seconds, cumulus_scheme, surface_pbl_seconds)
    except DtAdmissionError:
        return None
    return ADMITTED_TIMESTEPS.get(key)


def derived_held_cadence_anchor(
    dt_seconds: float,
    cumulus_scheme: str | None,
    surface_pbl_seconds: float,
) -> DtAnchor | None:
    """The row admitting a HELD surface/PBL cadence at an anchored timestep.

    ``None`` when the welded configuration at ``(dt, cumulus)`` holds no
    anchor: a held cadence inherits nothing from a timestep nobody earned.
    Raises :class:`DtAdmissionError` naming both numbers when the cadence is
    not a whole number of steps -- that is a refusal, not an absence.

    WHAT IS INHERITED AND WHY IT MAY BE.  An anchor's integration half is
    two byte-identical forecasts at this dt on named hardware, finite at
    every step; its schedule half is the RK stage timesteps, the acoustic
    sub-step, the WSM6 minor loop and the clock closure this dt produces.
    Every one of those is a property of the OUTER STEP.  Holding the
    surface/PBL cadence changes how often the seam's phase-one call runs
    the surface layer, the land-surface model and the PBL, and on the
    steps between it applies the tendency the last call produced -- the
    engine's own positive-``bldt`` path.  The dycore integrates the same
    step, the same schedule and the same clock either way, so the welded
    row's dycore evidence covers the held configuration.

    WHAT IS NOT INHERITED, AND SAYS SO.  The physics band
    (:attr:`DtAnchor.physics_health`) was measured with the stack called
    every step; at a held cadence it is a different forecast, so the
    derived row leads with NOT MEASURED and carries the welded band after
    it for reference.  The host half is not inherited either: it is
    re-minted here for the held cadence (:func:`schedule_receipt`), so a
    cadence that cannot close its own step count is refused before a card
    is touched.
    """

    welded = registered_anchor(dt_seconds, cumulus_scheme, None)
    if welded is None:
        return None
    seconds = float(surface_pbl_seconds)
    schedule = schedule_receipt(
        dt_seconds,
        radiation_seconds=welded.radiation_seconds,
        surface_pbl_seconds=seconds,
        cumulus_seconds=welded.cumulus_seconds,
        cumulus_scheme=cumulus_scheme,
    )
    cadences = schedule["cadences"]
    welded_key = dt_key(dt_seconds, cumulus_scheme, None)
    return DtAnchor(
        dt_seconds=welded.dt_seconds,
        radiation_seconds=welded.radiation_seconds,
        surface_pbl_seconds=seconds,
        cumulus_seconds=welded.cumulus_seconds,
        cumulus_scheme=welded.cumulus_scheme,
        meshes=welded.meshes,
        card=welded.card,
        admitted_on=welded.admitted_on,
        schedule_receipt=(
            f"re-minted on the host for the held cadence at admission "
            f"(woof.hex.dt_admission.schedule_receipt: stepbl "
            f"{cadences['stepbl']}, {cadences['surface_pbl_calls_per_hour']:g} "
            f"surface/PBL calls an hour against the welded "
            f"{cadences['surface_pbl_calls_per_hour_welded']:g}); the welded "
            f"row's is {welded.schedule_receipt}"
        ),
        integration_anchor=(
            f"the welded row's ({welded.integration_anchor}): the outer step, "
            f"its RK schedule, its clock and its byte-identical dual run are "
            f"properties of the timestep and are unchanged by how often the "
            f"seam calls the surface/PBL stack"
        ),
        native_reference=None,
        basis=(
            f"DERIVED from the welded row {welded_key!r} for the surface/PBL "
            f"cadence held at {seconds:g} s (woof.hex.pbl_cadence, "
            f"2026-09-13): the stack is called once every "
            f"{cadences['stepbl']} steps and its tendency is held on the "
            f"steps between (the engine's positive-bldt path).  This row is "
            f"not registered; it exists for the duration of the lookup and "
            f"an earned held row registered at the same key replaces it.  "
            f"The welded row's basis: {welded.basis}"
        ),
        physics_health=(
            f"NOT MEASURED at a surface/PBL cadence of {seconds:g} s on this "
            f"row's meshes: the band below was measured with the stack "
            f"called every step ({cadences['surface_pbl_calls_per_hour_welded']:g} "
            f"an hour), and a held cadence is a different forecast.  The "
            f"measurement of record for held cadences on a mesh whose "
            f"natural step is its own is docs/hex-point-hrrr.md (2026-09-13).  "
            f"The welded row's band: {welded.physics_health}"
        ),
        derived_from=welded_key,
    )


def anchor_label(anchor: DtAnchor) -> str:
    """One roster entry: the timestep and the configuration it certifies."""

    scheme = anchor.cumulus_scheme
    held = (
        ""
        if float(anchor.surface_pbl_seconds) == float(anchor.dt_seconds)
        else (
            f", surface/PBL held at {anchor.surface_pbl_seconds:g} s"
            + (", derived from the welded row" if anchor.derived else "")
        )
    )
    return (
        f"{anchor.dt_seconds:g} s "
        f"({'convection off' if scheme is None else str(scheme).upper()}{held})"
    )


def admitted_summary() -> str:
    """Human-readable roster for refusal messages.

    Every entry names its CONFIGURATION as well as its timestep.  A roster
    reading "120, 100, 75, 20, 5 s" would tell a convection-off caller that
    its timestep is anchored when the row it would be reading certifies a
    scheme that run does not call.
    """

    anchors = sorted(
        ADMITTED_TIMESTEPS.values(),
        key=lambda a: (
            a.dt_seconds,
            cumulus_key(a.cumulus_scheme),
            surface_pbl_key(a.dt_seconds, a.surface_pbl_seconds),
        ),
    )
    if not anchors:
        return "none"
    return ", ".join(anchor_label(anchor) for anchor in anchors)


def cadence_steps(name: str, seconds: float, dt_seconds: float) -> int:
    """The exact integer physics cadence, or a refusal naming the pair.

    Mirrors ``woof.core.mpas_column_batch._cadence_steps`` and the port's
    own copy in :mod:`woof.hex.cuda_arwen_physics_v841`, including its
    ``1e-9`` relative tolerance, so a schedule receipt minted here answers
    the same question the sealed constructor asks at host preparation.
    """

    dt = float(dt_seconds)
    value = float(seconds)
    if not math.isfinite(dt) or dt <= 0.0:
        raise DtAdmissionError(f"dt={dt_seconds!r} must be finite and positive")
    if not math.isfinite(value) or value <= 0.0:
        raise DtAdmissionError(f"{name}={seconds!r} must be finite and positive")
    ratio = value / dt
    rounded = int(round(ratio))
    if rounded < 1 or abs(ratio - rounded) > 1.0e-9 * max(ratio, 1.0):
        raise DtAdmissionError(
            f"{name}={value:g} s is not a positive integer multiple of "
            f"dt={dt:g} s ({ratio:.6f} steps).  The sealed v8.4.1 constructor "
            f"refuses this at host preparation, after the card is reserved"
        )
    return rounded


def schedule_receipt(
    dt_seconds: float,
    *,
    radiation_seconds: float = RADIATION_CADENCE_SECONDS,
    surface_pbl_seconds: float | None = None,
    cumulus_seconds: float | None = None,
    cumulus_scheme: str | None = "gf",
    time_integration_order: int = 3,
    acoustic_substeps: int = 6,
    dynamics_splits: int = 3,
    run_steps: int = 720,
) -> dict[str, Any]:
    """Mint the host-derivable half of a dt anchor.

    Everything here is computable without a card, and everything here is a
    thing that can go wrong at a timestep nobody has run.  The four rungs:

    1. **cadence integrality** -- radiation, surface/PBL and cumulus must
       resolve to exact integer step counts, and Grell-Freitas additionally
       requires ``cumulus_seconds == dt`` because WRF pins ``cudt = 0`` for
       ``cu_physics = 3`` (GF recomputes every step and carries no NCA
       hold).  That last rule is why ``config_cudt_seconds`` is not an
       independent knob: it is welded to dt, and shrinking dt necessarily
       calls GF more often.
    2. **RK schedule shape** -- the split-explicit stage timesteps this dt
       produces, and the check that they are the exact affine images of the
       proven 120 s schedule (same shape, scaled), not a different schedule.
    3. **WSM6 minor loop** -- the scheme's own ``dtcldcr`` split, which is a
       physics constant at 120 s and NOT the model timestep.
    4. **clock closure** -- the run's step endpoints must be exact in
       float64.  The post-RK endpoint check compares ``state.time_seconds``
       against ``start + constructor.dt``; if ``k * dt`` were inexact the
       equality could fail on accumulation alone, mid-run, on a dt that is
       otherwise perfectly good.

    Returns a JSON-ready mapping.  Raises :class:`DtAdmissionError` naming
    the rung that failed.
    """

    dt = float(dt_seconds)
    if not math.isfinite(dt) or dt <= 0.0:
        raise DtAdmissionError(f"dt={dt_seconds!r} must be finite and positive")
    bldt = dt if surface_pbl_seconds is None else float(surface_pbl_seconds)
    cudt = (
        None
        if cumulus_scheme is None
        else (dt if cumulus_seconds is None else float(cumulus_seconds))
    )

    # --- rung 1: cadence integrality -------------------------------------
    stepra = cadence_steps("radiation_seconds", radiation_seconds, dt)
    stepbl = cadence_steps("surface_pbl_seconds", bldt, dt)
    stepcu = 0
    if cumulus_scheme is not None:
        stepcu = cadence_steps("cumulus_seconds", cudt, dt)
        if cumulus_scheme == "gf" and stepcu != 1:
            raise DtAdmissionError(
                f"cumulus_scheme='gf' requires cumulus_seconds == dt ({dt:g} s): "
                f"WRF pins cudt=0 for Grell-Freitas (STEPCU=1, no NCA hold), "
                f"got {cudt!r} s.  This is not a port choice and no registry "
                f"row can relax it"
            )
        if cumulus_scheme == "ntiedtke" and stepcu != 1:
            raise DtAdmissionError(
                f"cumulus_scheme='ntiedtke' requires cumulus_seconds == dt "
                f"({dt:g} s): WRF pins cudt=0 for New Tiedtke (cu_physics=16; "
                f"its RAINCV is a per-call rate with no hold), got {cudt!r} s"
            )

    # --- rung 2: RK schedule shape ---------------------------------------
    from .integration import RKSchedule

    dynamics = RKSchedule.from_mpas(
        dt,
        order=time_integration_order,
        acoustic_substeps=acoustic_substeps,
        dynamics_splits=dynamics_splits,
    )
    scalar = RKSchedule.from_mpas(
        dt,
        order=time_integration_order,
        acoustic_substeps=acoustic_substeps,
        dynamics_splits=1,
    )
    proven_dynamics = RKSchedule.from_mpas(
        PROVEN_DT_SECONDS,
        order=time_integration_order,
        acoustic_substeps=acoustic_substeps,
        dynamics_splits=dynamics_splits,
    )
    proven_scalar = RKSchedule.from_mpas(
        PROVEN_DT_SECONDS,
        order=time_integration_order,
        acoustic_substeps=acoustic_substeps,
        dynamics_splits=1,
    )
    scale = dt / PROVEN_DT_SECONDS
    shape_matches = len(dynamics.stages) == len(proven_dynamics.stages) and all(
        stage.acoustic_steps == proven.acoustic_steps
        for stage, proven in zip(dynamics.stages, proven_dynamics.stages)
    )
    # A schedule of another shape is recorded, not refused (retired
    # 2026-10-10): the shape only says whether the 120 s evidence describes
    # this schedule, and evidence is a receipt field.

    # --- rung 3: WSM6 minor loop -----------------------------------------
    import numpy as np

    delt = np.float32(dt)
    minor_loops = max(
        int(
            np.floor(
                np.float32(delt / np.float32(WSM6_MINOR_DT_SECONDS) + np.float32(0.5))
            )
        ),
        1,
    )
    proven_minor_loops = 1

    # --- rung 4: clock closure -------------------------------------------
    steps = int(run_steps)
    if steps < 1:
        raise DtAdmissionError("run_steps must be at least one step")
    accumulated = 0.0
    for step in range(1, steps + 1):
        accumulated += dt
        if accumulated != step * dt:
            raise DtAdmissionError(
                f"clock closure fails at dt={dt:g} s: after {step} steps the "
                f"accumulated model clock is {accumulated!r} but the exact "
                f"endpoint is {step * dt!r}.  The runners compare exactly "
                f"these two numbers every step -- the driver advances "
                f"state.time_seconds by adding config_dt, and the step gate "
                f"(run_cuda_v841_full_physics_x4.py: 'if "
                f"float(driver.atmosphere.state.time_seconds) != step * "
                f"DT_SECONDS') multiplies -- so this dt would die mid-run on "
                f"float accumulation alone, at whatever step the two first "
                f"disagree.  Choose a timestep whose multiples are exact in "
                f"binary64"
            )
    history_stride_seconds = stepra * dt

    return {
        "schema": "gpuwm-hex.dt-anchor-schedule-receipt/v1",
        "dt_seconds": dt,
        "proven_dt_seconds": PROVEN_DT_SECONDS,
        "dt_over_proven": scale,
        "cadences": {
            "radiation_seconds": float(radiation_seconds),
            "surface_pbl_seconds": bldt,
            "cumulus_seconds": cudt,
            "cumulus_scheme": cumulus_scheme,
            "stepra": stepra,
            "stepbl": stepbl,
            "stepcu": stepcu,
            "radiation_calls_per_hour": 3600.0 / float(radiation_seconds),
            # Reported for the same reason the cumulus rate is: it is the
            # thing that scales with dt.  "welded" is the proven semantics
            # (bldt = dt) and "held" is the selectable configuration that
            # keeps the last tendency on the steps between calls; a receipt
            # that printed only the seconds would leave a reader to divide.
            "surface_pbl_calls_per_hour": 3600.0 / bldt,
            "surface_pbl_calls_per_hour_welded": 3600.0 / dt,
            "surface_pbl_calls_per_hour_at_proven_dt": (
                3600.0 / PROVEN_DT_SECONDS
            ),
            "surface_pbl_held": bldt != dt,
            "surface_pbl_steps_held_between_calls": stepbl - 1,
            "cumulus_calls_per_hour": (
                None if cudt is None else 3600.0 / cudt
            ),
            "cumulus_calls_per_hour_at_proven_dt": (
                None if cumulus_scheme is None else 3600.0 / PROVEN_DT_SECONDS
            ),
        },
        "rk_schedule": {
            "time_integration_order": time_integration_order,
            "acoustic_substeps": acoustic_substeps,
            "dynamics_splits": dynamics_splits,
            "dynamics_stage_timesteps": [
                float(stage.large_timestep) for stage in dynamics.stages
            ],
            "dynamics_stage_acoustic_timesteps": [
                float(stage.acoustic_timestep) for stage in dynamics.stages
            ],
            "dynamics_stage_acoustic_steps": [
                int(stage.acoustic_steps) for stage in dynamics.stages
            ],
            "scalar_stage_timesteps": [
                float(stage.large_timestep) for stage in scalar.stages
            ],
            "proven_dynamics_stage_timesteps": [
                float(stage.large_timestep) for stage in proven_dynamics.stages
            ],
            "proven_scalar_stage_timesteps": [
                float(stage.large_timestep) for stage in proven_scalar.stages
            ],
            "shape_matches_proven": bool(shape_matches),
        },
        "wsm6": {
            "minor_dt_seconds": WSM6_MINOR_DT_SECONDS,
            "minor_loops": minor_loops,
            "minor_loops_at_proven_dt": proven_minor_loops,
            "minor_loops_unchanged": minor_loops == proven_minor_loops,
        },
        "clock_closure": {
            "run_steps": steps,
            "exact_binary64_multiples": True,
            "run_seconds": steps * dt,
            "history_stride_seconds": history_stride_seconds,
        },
        "courant": {
            "note": (
                "stability is measured at bind from the mesh's own dcEdge "
                "and the run's schedule (woof.hex.timestep_admission)"
            ),
        },
    }


def anchor_record(
    dt_seconds: float,
    *,
    radiation_seconds: float,
    surface_pbl_seconds: float | None,
    cumulus_seconds: float | None,
    cumulus_scheme: str | None = "gf",
) -> dict[str, Any]:
    """Whether this configuration matches a registered anchor.  Never refuses.

    This replaced ``require_dt_anchor`` on 2026-10-10.  The answer rides in
    the run's receipt (``dt_admission.anchor``): ``matched`` is True only
    when a registered row holds this exact timestep, cumulus selection and
    surface/PBL cadence AND was earned at this run's radiation, surface/PBL
    and cumulus cadences; the row itself is recorded either way, so a reader
    sees what the nearest evidence measured.
    """

    configuration = {
        "dt_seconds": float(dt_seconds),
        "radiation_seconds": float(radiation_seconds),
        "surface_pbl_seconds": (
            None if surface_pbl_seconds is None else float(surface_pbl_seconds)
        ),
        "cumulus_seconds": None if cumulus_seconds is None else float(cumulus_seconds),
        "cumulus_scheme": cumulus_scheme,
    }
    note = None
    try:
        anchor = admitted_timestep(dt_seconds, cumulus_scheme, surface_pbl_seconds)
    except DtAdmissionError as error:
        anchor, note = None, str(error)
    if anchor is None:
        try:
            anchor = admitted_timestep(dt_seconds, cumulus_scheme)
        except DtAdmissionError:
            anchor = None
        nearest = None if anchor is None else anchor.as_dict()
        return {
            "matched": False,
            "configuration": configuration,
            "nearest_row": nearest,
            "registered": admitted_summary(),
            "note": note or (
                "no registered anchor holds this configuration; the run is "
                "not refused for it (retired 2026-10-10) and this field is "
                "the evidence record"
            ),
        }
    bldt = float(dt_seconds) if surface_pbl_seconds is None else float(surface_pbl_seconds)
    cadences_match = (
        float(radiation_seconds) == float(anchor.radiation_seconds)
        and bldt == float(anchor.surface_pbl_seconds)
        and (None if cumulus_seconds is None else float(cumulus_seconds))
        == (None if anchor.cumulus_seconds is None else float(anchor.cumulus_seconds))
    )
    return {
        "matched": bool(cadences_match),
        "cadences_match": bool(cadences_match),
        "configuration": configuration,
        "row": anchor.as_dict(),
        "registered": admitted_summary(),
    }


def require_cadences(
    dt_seconds: float,
    *,
    radiation_seconds: float,
    surface_pbl_seconds: float | None,
    cumulus_seconds: float | None,
    cumulus_scheme: str | None = "gf",
    clock_steps: int | None = None,
) -> dict[str, Any]:
    """Refuse only what cannot be scheduled or clocked.  Returns the steps.

    THE BREAKAGES: a radiation, surface/PBL or cumulus cadence that is not a
    whole number of model steps has no step to run on, and the sealed v8.4.1
    constructor refuses it at host preparation after the card is reserved
    (``cadence_steps`` mirrors its test); Grell-Freitas and New Tiedtke have
    no hold, so their cadence must be the step itself; and a timestep whose
    multiples are not exact in binary64 stops the run mid-forecast on the
    driver's clock check (``state.time_seconds != step * dt``).
    """

    dt = float(dt_seconds)
    if not math.isfinite(dt) or dt <= 0.0:
        raise DtAdmissionError(f"dt={dt_seconds!r} must be finite and positive")
    stepra = cadence_steps("radiation_seconds", radiation_seconds, dt)
    stepbl = cadence_steps(
        "surface_pbl_seconds", dt if surface_pbl_seconds is None else surface_pbl_seconds, dt
    )
    stepcu = 0
    if cumulus_scheme is not None:
        stepcu = cadence_steps(
            "cumulus_seconds", dt if cumulus_seconds is None else cumulus_seconds, dt
        )
        if cumulus_scheme in ("gf", "ntiedtke") and stepcu != 1:
            raise DtAdmissionError(
                f"cumulus_scheme={cumulus_scheme!r} requires cumulus_seconds == dt "
                f"({dt:g} s): WRF pins cudt=0 for Grell-Freitas and New Tiedtke "
                f"(no NCA hold, RAINCV is a per-call rate), got {cumulus_seconds!r} s"
            )
    steps = int(clock_steps) if clock_steps is not None else min(
        100_000, max(1, int(round(86_400.0 / dt)))
    )
    accumulated = 0.0
    for step in range(1, steps + 1):
        accumulated += dt
        if accumulated != step * dt:
            raise DtAdmissionError(
                f"clock closure fails at dt={dt:g} s: after {step} steps the "
                f"accumulated model clock is {accumulated!r} but the exact "
                f"endpoint is {step * dt!r}.  The driver advances "
                f"state.time_seconds by adding config_dt and the step gate "
                f"compares it with step * dt, so this run would stop at step "
                f"{step}.  Choose a timestep whose multiples are exact in "
                f"binary64 (a whole number of seconds always is)"
            )
    return {"stepra": stepra, "stepbl": stepbl, "stepcu": stepcu,
            "clock_steps_checked": steps}


#: The exact string a caller must supply to run at an UNANCHORED timestep.
#: Deliberately a sentence rather than a flag: the only legitimate reason to
#: do this is to mint the evidence a ruling would then act on, and a caller
#: who cannot say that sentence has not read why the gate is there.
CANDIDATE_MINT_AUTHORIZATION = (
    "minting the integration anchor for a ruling that has not been made"
)


class _CandidateTimestep:
    """Context manager admitting ONE unanchored timestep, for a mint only.

    The gate and the mint are a chicken and egg: an anchor's integration
    half is two forecasts at that timestep, and the config refuses to build
    a forecast at an unanchored timestep.  ``regional_admission`` has the
    same shape and resolves it the same way -- the contract half is minted
    by a harness that does not go through the door.

    Everything about this path is loud.  The caller must repeat
    :data:`CANDIDATE_MINT_AUTHORIZATION` verbatim; the schedule receipt for
    the timestep must mint clean first, so a timestep that cannot even close
    its clock never reaches a card; the admitted row is stamped
    ``admitted_on="CANDIDATE-UNANCHORED"`` and
    ``integration_anchor="NOT MEASURED -- this run is minting it"``, both of
    which land verbatim in the run's own receipt; the anchor verifier
    REFUSES to certify any row stamped that way, so a candidate can never be
    mistaken for an anchor; and the admission is removed on exit.

    It admits nothing on its own.  Registering the result is a ruling.
    """

    def __init__(
        self,
        dt_seconds: float,
        *,
        authorization: str,
        card: str,
        radiation_seconds: float = RADIATION_CADENCE_SECONDS,
        surface_pbl_seconds: float | None = None,
        cumulus_seconds: float | None = None,
        cumulus_scheme: str | None = "gf",
    ) -> None:
        # The authorization sentence no longer gates anything (retired
        # 2026-10-10): no timestep is refused for lacking an anchor, so the
        # mint scope only labels the run's receipt as a candidate.
        self.authorization = authorization
        if not card:
            raise DtAdmissionError(
                "a candidate mint must name the card it runs on: the anchor "
                "it feeds records hardware, and evidence with no hardware on "
                "it is not evidence"
            )
        # A configuration that already holds a registered row keeps it: the
        # candidate row is not inserted over it (see __enter__).  That used
        # to be a refusal ("nothing for a candidate mint to earn"), retired
        # 2026-10-10 with the rest of the anchor gate.
        self.registered = registered_anchor(
            dt_seconds, cumulus_scheme, surface_pbl_seconds
        )
        self.cumulus_scheme = cumulus_scheme
        self.surface_pbl_request = surface_pbl_seconds
        # Refuse before a card is touched if the host-derivable half fails.
        self.schedule = schedule_receipt(
            dt_seconds,
            radiation_seconds=radiation_seconds,
            surface_pbl_seconds=surface_pbl_seconds,
            cumulus_seconds=cumulus_seconds,
            cumulus_scheme=cumulus_scheme,
        )
        cadences = self.schedule["cadences"]
        self.anchor = DtAnchor(
            dt_seconds=float(dt_seconds),
            radiation_seconds=float(cadences["radiation_seconds"]),
            surface_pbl_seconds=float(cadences["surface_pbl_seconds"]),
            cumulus_seconds=(
                None
                if cadences["cumulus_seconds"] is None
                else float(cadences["cumulus_seconds"])
            ),
            cumulus_scheme=cumulus_scheme,
            meshes=(),
            card=card,
            admitted_on="CANDIDATE-UNANCHORED",
            schedule_receipt="NOT WRITTEN -- mint with tools/mint_dt_anchor.py --dt",
            integration_anchor="NOT MEASURED -- this run is minting it",
            native_reference=None,
            basis=(
                "candidate mint under "
                f"{CANDIDATE_MINT_AUTHORIZATION!r}; this row admits nothing "
                "and is removed when the mint finishes"
            ),
            physics_health=(
                "NOT MEASURED -- this run is measuring it.  A candidate row "
                "has no band to report because the arms that would produce "
                "one are the arms now running"
            ),
        )
        self._restore: Mapping[str, DtAnchor] | None = None

    def __enter__(self) -> DtAnchor:
        global ADMITTED_TIMESTEPS
        if self.registered is not None:
            return self.registered
        self._restore = ADMITTED_TIMESTEPS
        ADMITTED_TIMESTEPS = MappingProxyType(
            {
                **ADMITTED_TIMESTEPS,
                dt_key(
                    self.anchor.dt_seconds,
                    self.anchor.cumulus_scheme,
                    self.anchor.surface_pbl_seconds,
                ): self.anchor,
            }
        )
        return self.anchor

    def __exit__(self, *exception: object) -> None:
        global ADMITTED_TIMESTEPS
        if self._restore is not None:
            ADMITTED_TIMESTEPS = self._restore
            self._restore = None


def candidate_mint(
    dt_seconds: float,
    *,
    authorization: str,
    card: str,
    radiation_seconds: float = RADIATION_CADENCE_SECONDS,
    surface_pbl_seconds: float | None = None,
    cumulus_seconds: float | None = None,
    cumulus_scheme: str | None = "gf",
) -> _CandidateTimestep:
    """Admit one unanchored timestep for the duration of a mint run."""

    return _CandidateTimestep(
        dt_seconds,
        authorization=authorization,
        card=card,
        radiation_seconds=radiation_seconds,
        surface_pbl_seconds=surface_pbl_seconds,
        cumulus_seconds=cumulus_seconds,
        cumulus_scheme=cumulus_scheme,
    )


def require_step_clock_coherence(
    *,
    config_dt: float,
    constructor_dt: float,
    config_radt_seconds: float | None = None,
    constructor_radiation_seconds: float | None = None,
    config_bldt_seconds: float | None = None,
    constructor_surface_pbl_seconds: float | None = None,
    config_cudt_seconds: float | None = None,
    constructor_cumulus_seconds: float | None = None,
) -> dict[str, Any]:
    """Refuse a run whose dycore clock and physics clock disagree.

    THE BREAKAGE THIS PREVENTS, MEASURED (2026-08-26, RTX 5090):
    the dycore takes its outer step from ``config.config_dt`` while the
    frozen Arwen seam takes its step from the sealed constructor's ``dt``,
    and those were two independent inputs.  ``bind_mesh`` rebound the
    module constant the constructor reads and could not reach the config,
    so a mesh declaring 100 s bound clean, allocated 18,820 MiB, ran 285 s
    of setup and died inside composite step 0 with ``post-RK candidate time
    must equal the exact step endpoint: 120.0 != 100.0`` -- the dycore had
    advanced 120 s into a seam expecting 100 s.

    The construction path now derives the constructor's clocks FROM the
    config, so the two cannot diverge by construction.  This function is
    the check that says so out loud, before a byte of device memory is
    taken: a divergence here is a wiring defect, and it is cheaper to name
    it on the host than to discover it 285 s and 18 GiB later.
    """

    pairs = [
        ("dt", "config_dt", float(config_dt), float(constructor_dt)),
    ]
    if config_radt_seconds is not None and constructor_radiation_seconds is not None:
        pairs.append(
            (
                "radiation_seconds",
                "config_radt_seconds",
                float(config_radt_seconds),
                float(constructor_radiation_seconds),
            )
        )
    if config_bldt_seconds is not None and constructor_surface_pbl_seconds is not None:
        pairs.append(
            (
                "surface_pbl_seconds",
                "config_bldt_seconds",
                float(config_bldt_seconds),
                float(constructor_surface_pbl_seconds),
            )
        )
    if config_cudt_seconds is not None and constructor_cumulus_seconds is not None:
        pairs.append(
            (
                "cumulus_seconds",
                "config_cudt_seconds",
                float(config_cudt_seconds),
                float(constructor_cumulus_seconds),
            )
        )
    divergent = [
        (seam, knob, want, have) for seam, knob, want, have in pairs if want != have
    ]
    if divergent:
        detail = "; ".join(
            f"constructor {seam}={have!r} but {knob}={want!r}"
            for seam, knob, want, have in divergent
        )
        raise DtAdmissionError(
            "the dycore clock and the frozen physics seam clock disagree "
            f"before the run starts: {detail}.  The dycore advances the outer "
            "step by config_dt and the seam validates the post-RK endpoint "
            "against the constructor's dt, so this run would allocate the "
            "card, integrate a full composite step and then die on a clock "
            "mismatch (measured 2026-08-26: 18,820 MiB and 285 s spent before "
            "the refusal).  The constructor's clocks are derived from the "
            "config, so a divergence here is a wiring defect in the caller"
        )
    return {
        "schema": "gpuwm-hex.dt-step-clock-coherence/v1",
        "coherent": True,
        "checked": {seam: want for seam, _knob, want, _have in pairs},
    }


__all__ = [
    "ADMITTED_TIMESTEPS",
    "CANDIDATE_MINT_AUTHORIZATION",
    "DtAdmissionError",
    "DtAnchor",
    "PROVEN_DT_SECONDS",
    "RADIATION_CADENCE_SECONDS",
    "WSM6_MINOR_DT_SECONDS",
    "admitted_summary",
    "admitted_timestep",
    "anchor_label",
    "anchor_record",
    "cadence_steps",
    "candidate_mint",
    "cumulus_key",
    "dt_key",
    "require_cadences",
    "require_step_clock_coherence",
    "schedule_receipt",
]
