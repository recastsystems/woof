"""Earned-anchor admission for regional (limited-area) CUDA execution.

Ruled 2026-08-25 as part of the LAM campaign: *regional execution
is refused by name until a registered regional anchor exists*, mirroring the
per-architecture pattern in :mod:`woof.hex.cuda_backend.arch_admission`.

The reasoning is the same one that admits an architecture below the proven
floor.  The port's CUDA numbers are trusted because they are checked against
something: the global lane against the frozen v8.4.1 authority set, an
unproven architecture against its own minted anchor.  A regional
configuration has neither until somebody mints one, so a regional forecast
run without an anchor produces numbers no receipt could be compared to.  The
concrete breakage this gate prevents is therefore not a crash: it is a
regional forecast that looks finished, carries a receipt, and was never
checked against anything at all.

AN ANCHOR HAS TWO HALVES AND THEY DO NOT KEY THE SAME WAY.  This is the
2026-08-27 re-keying, and it retires a measured tax rather than relaxing a
gate.

* the **contract deck** measures every regional kernel's device bits against
  the CPU authority's host-derived expected bits ON ONE CULL'S OWN ZONE
  GEOMETRY, with a mutation control per deck proving the deck can fail.  What
  it certifies is a geometry.  It stays keyed to the geometry -- one row per
  ``bdyMask`` digest -- and it is cheap: 39 s at 4,440 cells, 142 s at 15,755
  (``evidence/nest-ratio-20260827/RECEIPT.md`` section 3b).

* the **forecast mint** runs the same configuration twice in two independent
  processes and compares the published history under masked digests.  What it
  certifies is that this KERNEL SET, at this TIMESTEP, on a domain of this
  SHAPE CLASS, reproduces itself.  It is two 1,080-step integrations -- 5.5
  to 8.7 minutes of card measured on four new meshes, against 4-6 minutes for
  the forecast it is admitting.

Keying the MINT to a cull's own bytes charged that 5.5-8.7 minutes again for
every re-placed cull.  A storm-following swath re-places itself every cycle
and a re-placed cull is a new digest, so the cascade paid roughly three
forecasts of card time per swath per cycle for permission to run one.

**Under the gate law that per-digest mint cannot name the breakage it
prevents.**  The measurement is in the nest-ratio receipt: five concentric
culls of ONE parent, 4,440 to 15,755 cells, all carrying the parent's own
finest edge of 4,457.233 m, all seven rings wide, all at dt 20 s -- five
independent mints, five identical verdicts, and all five earned the same 20 s
anchor at the same 1.605x Courant margin.  Nothing the mint measures
distinguished them.  the project rule is that a guard must name the concrete
breakage it prevents, and "this cull's bytes are new" is not one when every
input the mint reads is the same.

So the mint keys to the CONFIGURATION CLASS, and the class key is exactly the
set of inputs the mint's own instrument reads and its verdict depends on:

``boundary_zone_width``
    How many rings the culler grew.  Every regional stage -- the specified
    zone, the relaxation zone's Rayleigh and Laplacian terms, the mask-4/5
    scalar downgrade -- is indexed by ring, so a zone of a different width is
    a different set of stages and the mint says nothing about it.

``n_vert_levels``
    Every regional kernel launches over ``(levels, elements)``; a different
    column height is a different launch geometry for all 22 of them.

``finest_edge_m``
    ``min(dcEdge)``, measured off the mesh in hand.  It is what the Courant
    admission keys on, what ``config_len_disp`` follows, and what
    ``convection_admission`` takes its decision on.  It is also the one
    length a cull cannot change: a cull moves no cell centre, so every cull
    containing the parent's fine core carries the parent's own finest edge.

``dt_seconds``
    The relaxation-zone coefficients are hardwired at 50 and 10 times dt, so
    the boundary stages are dt-dependent by construction.  A mint at one
    timestep says nothing about another.

``kernel_set_sha256``
    A digest over the source of every translation unit the regional step
    launches through.  The mint is a statement about bytes; if the bytes
    move, the statement lapses.  Nothing checked this before -- the mint
    RECORDED its source digests and no admission ever re-read them.

**What class keying does NOT relax.**  A cull of an admitted class is still
refused until its own contract deck has been run and its receipt presented:
8 of 8 decks bitwise against the v8.4.1 CPU authority on THIS cull's zone
geometry, 22 of 22 declared kernels covered, every mutation control with
teeth.  The residual per-geometry admission cost is that deck and nothing
else.

**And what the pair of halves still does not admit** is unchanged and stated
on every row: the mint pair is the DRY regional dycore, so an anchor
certifies the 22 boundary kernels and says nothing on its own about the WOOF
physics seam a full-physics run also drives; and endpoint byte-identity
against the compiled reference is NOT measured here, and is not measured for
the global lane either.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .. import shipped_sources

#: Every translation unit the regional step launches through, as bytes.
#:
#: This is the mint instrument's own list (``tools/run_cuda_regional_forecast``)
#: minus the instrument itself: the instrument decides what is WRITTEN, never
#: what is COMPUTED, and hashing it would make an edit to a print statement
#: lapse every class.  ``regional_admission`` is deliberately absent for the
#: same reason it is absent from the mint receipt -- it decides ADMISSION, not
#: arithmetic, and the rows' own prose lives in it, so hashing it would make
#: every edit to a basis sentence invalidate the evidence that sentence names.
REGIONAL_KERNEL_SOURCES: tuple[str, ...] = (
    "src/hexcore/cuda_column_order_v841.py",
    "src/hexcore/cuda_acoustic_fusion_v841.py",
    "src/hexcore/cuda_upper_damping_v841.py",
    "src/hexcore/cuda_acoustic_v841.py",
    "src/hexcore/cuda_backend/containers.py",
    "src/hexcore/cuda_backend/aot.py",
    "src/hexcore/cuda_backend/runtime.py",
    "src/hexcore/cuda_driver.py",
    "src/hexcore/cuda_dynamics_v841.py",
    "src/hexcore/cuda_fp32.py",
    "src/hexcore/cuda_horizontal.py",
    "src/hexcore/cuda_horizontal_v841.py",
    "src/hexcore/cuda_regional_forecast_v841.py",
    "src/hexcore/cuda_regional_v841.py",
    "src/hexcore/cuda_transport.py",
    "src/hexcore/cuda_transport_v841.py",
    "src/hexcore/cuda_v841.py",
    "src/hexcore/regional_v841.py",
)

class RegionalAdmissionRefusal(RuntimeError):
    """A regional configuration is refused, and the message says why."""


def kernel_set_sha256(root: Path | None = None) -> str:
    """One digest over every source the regional step launches through.

    Framed name-then-payload with a NUL separator, in the declared order, so
    the digest cannot be moved by renaming a file into another's bytes.  The
    same framing convention ``mesh.regional_boundary_mask_digest`` uses, and
    for the same reason: this program has already lost a day to two
    classifiers hashing the same arrays two different ways.

    EACH NAME RESOLVES TO THE FILE THIS INTERPRETER WILL LAUNCH, through
    :mod:`woof.hex.shipped_sources`, and that is the wheel-reach defect's fix.  The
    constant this replaced walked three directories up from here and assumed
    a checkout underneath; from an installed wheel that lands on
    ``<venv>/lib/python3.13``, where no ``src/`` exists, so this function
    refused and took the whole limited-area lane -- 0.2.0's headline -- with
    it on every machine, with no flag that could open it.  The names are
    unchanged, the bytes under them in a checkout are unchanged, and the
    digest is therefore unchanged: ``MINTED_KERNEL_SET_SHA256`` did not move
    and no admitted class lapsed.  Measured both ways in
    ``evidence/wheel-reach-20260827``.  That paragraph is about the wheel-reach defect
    and nothing else.  0.2.0's package rename moved this digest twice over --
    every declared NAME went from ``src/mpas_port/`` to ``src/hexcore/``, and
    eleven of the fourteen PAYLOADS moved with them, because the same token is
    an import line and an NVRTC ``module_key`` inside those files.  The digest
    therefore lapsed, which is the cost a name-framed digest exists to charge;
    the alternative, keeping a retired directory name in the declaration so the
    arithmetic would not notice, is a gate reporting on a path nothing
    executes.  What the move was NOT is measured and stated at
    :data:`MINTED_KERNEL_SET_SHA256`.

    The gate did not lose its teeth in the move.  It gained them where it had
    none: from a wheel it now digests the modules in ``site-packages`` that
    the run actually imports, so altering one of those bytes lapses every
    class, which is exactly what a statement about bytes has to do.
    """

    digest = hashlib.sha256()
    for name in REGIONAL_KERNEL_SOURCES:
        path = shipped_sources.resolve(name, root)
        try:
            payload = path.read_bytes()
        except OSError as error:  # pragma: no cover - a broken install
            raise RegionalAdmissionRefusal(
                f"the regional kernel set cannot be digested: {name} is not "
                f"readable at {path} under "
                f"{shipped_sources.describe_root(root)} ({error}).  A "
                f"regional anchor is a statement about these bytes, so an "
                f"install that cannot produce them cannot be admitted "
                f"against one"
            ) from error
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


#: A class whose finest edge was never measured in a checkout holding its
#: mesh.  Exactly one class carries it -- the 120 km v8.4.1 reference cull,
#: whose grid publishes UNIT-SPHERE ``dcEdge`` and whose Earth-scaled metrics
#: live in a static file this repository does not contain (§9 of STATE.md: the
#: regional reference bundle is a mirror on the nodes, with no fetch path).
#: Guessing the number would be a fabricated key, so the class declares it
#: unmeasured and PAYS FOR IT: a class with an unmeasured edge admits only the
#: cull geometries already listed in :data:`SHIPPED_CONTRACTS`, never a
#: presented receipt, so it cannot grow to cover a cull nobody looked at.
UNMEASURED_FINEST_EDGE_MM = -1

#: The relative finest-edge band a same-ladder class admits (see
#: :meth:`RegionalClassKey.finest_edge_within`).  Five per cent: wide enough
#: for the relaxation noise between two generations of one ladder spec at
#: different points, narrow enough that the two sub-kilometre classes (869 m
#: and 711 m, 18 % apart) can never both admit one measured edge.
SAME_LADDER_FINEST_EDGE_BAND = 0.05


@dataclass(frozen=True, slots=True)
class RegionalClassKey:
    """The identity a forecast MINT actually measures, geometry removed.

    Integers, deliberately.  ``finest_edge_mm`` and ``dt_ms`` are quantised
    so a key can be compared for equality without a tolerance and without a
    float repr appearing in a refusal message.  A cull moves no cell centre,
    so two culls of one parent carry ``min(dcEdge)`` in identical float64
    bits; the quantisation is protection against a value that arrived through
    a printed number, not against real drift.
    """

    boundary_zone_width: int
    n_vert_levels: int
    finest_edge_mm: int
    dt_ms: int
    kernel_set_sha256: str

    @classmethod
    def build(
        cls,
        *,
        boundary_zone_width: int,
        n_vert_levels: int,
        finest_edge_m: float,
        dt_seconds: float,
        kernel_set: str,
    ) -> "RegionalClassKey":
        return cls(
            boundary_zone_width=int(boundary_zone_width),
            n_vert_levels=int(n_vert_levels),
            finest_edge_mm=int(round(float(finest_edge_m) * 1000.0)),
            dt_ms=int(round(float(dt_seconds) * 1000.0)),
            kernel_set_sha256=str(kernel_set),
        )

    @property
    def finest_edge_measured(self) -> bool:
        return self.finest_edge_mm != UNMEASURED_FINEST_EDGE_MM

    def finest_edge_within(self, other: "RegionalClassKey", band: float) -> bool:
        """Does ``other``'s finest edge sit inside this class's band?

        ``band`` is RELATIVE (0.05 is five per cent either way) and 0.0 is
        exact equality in millimetres, which is what every class declared
        until 2026-09-13.  The quantisation was "protection against a value
        that arrived through a printed number, not against real drift"
        (above): two culls of ONE parent carry ``min(dcEdge)`` in identical
        bits.  A parent GENERATED AT ANOTHER POINT from the same ladder spec
        carries the same delivered rung and a finest edge that differs by
        relaxation noise -- a few tenths of a per cent to a few per cent --
        and exact millimetres would refuse it as a different resolution
        class, which it is not.  A class that admits a band says so in its
        row (``finest_edge_band_relative``) and the receipt records the
        measured edge beside the class's own.
        """

        if not self.finest_edge_measured:
            return True
        if band <= 0.0:
            return self.finest_edge_mm == other.finest_edge_mm
        return abs(other.finest_edge_mm - self.finest_edge_mm) <= band * self.finest_edge_mm

    def matches(self, other: "RegionalClassKey", *, band: float = 0.0) -> bool:
        """Is ``other`` -- a key measured off a real run -- this class?

        Every field compares exactly, except that a class declaring an
        unmeasured finest edge does not compare one (such a class buys that
        looseness back by refusing every geometry that is not already a
        shipped contract row, see :func:`require_regional_anchor`), and a
        class declaring a finest-edge band compares the edge within it
        (:meth:`finest_edge_within`).
        """

        if (
            self.boundary_zone_width != other.boundary_zone_width
            or self.n_vert_levels != other.n_vert_levels
            or self.dt_ms != other.dt_ms
            or self.kernel_set_sha256 != other.kernel_set_sha256
        ):
            return False
        return self.finest_edge_within(other, band)

    def differing_fields(self, other: "RegionalClassKey", *, band: float = 0.0) -> list[str]:
        names = ["boundary_zone_width", "n_vert_levels", "dt_ms", "kernel_set_sha256"]
        differing = [name for name in names if getattr(self, name) != getattr(other, name)]
        if self.finest_edge_measured and not self.finest_edge_within(other, band):
            differing.append("finest_edge_mm")
        return differing

    def as_dict(self) -> dict[str, object]:
        return {
            "boundary_zone_width": self.boundary_zone_width,
            "n_vert_levels": self.n_vert_levels,
            "finest_edge_m": (
                None
                if not self.finest_edge_measured
                else self.finest_edge_mm / 1000.0
            ),
            "dt_seconds": self.dt_ms / 1000.0,
            "kernel_set_sha256": self.kernel_set_sha256,
        }

    def describe(self) -> str:
        edge = (
            "finest edge NOT MEASURED"
            if not self.finest_edge_measured
            else f"finest edge {self.finest_edge_mm / 1000.0:.3f} m"
        )
        return (
            f"zone width {self.boundary_zone_width}, "
            f"{self.n_vert_levels} levels, "
            f"{edge}, "
            f"dt {self.dt_ms / 1000.0:g} s, "
            f"kernel set {self.kernel_set_sha256[:16]}..."
        )


@dataclass(frozen=True, slots=True)
class RegionalClass:
    """One minted configuration class: the FORECAST half of an anchor."""

    class_id: str
    key: RegionalClassKey
    parent: str
    card: str
    admitted_on: str
    mint_receipts: tuple[str, ...]
    minted_on_geometries: tuple[str, ...]
    basis: str
    #: How far, relatively, a measured finest edge may sit from this class's
    #: own and still be this class.  0.0 (the default) is exact millimetres.
    #: A class declares a band only when its basis says why; see
    #: :meth:`RegionalClassKey.finest_edge_within`.
    finest_edge_band_relative: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "key": self.key.as_dict(),
            "finest_edge_band_relative": self.finest_edge_band_relative,
            "parent": self.parent,
            "card": self.card,
            "admitted_on": self.admitted_on,
            "mint_receipts": list(self.mint_receipts),
            "minted_on_geometries": list(self.minted_on_geometries),
            "basis": self.basis,
        }


@dataclass(frozen=True, slots=True)
class RegionalContract:
    """One cull's own contract deck: the GEOMETRY half of an anchor."""

    bdy_mask_sha256: str
    n_cells: int
    boundary_zone_width: int
    class_id: str
    mesh_row: str | None
    card: str
    admitted_on: str
    contract_receipt: str
    basis: str

    def as_dict(self) -> dict[str, object]:
        return {
            "bdy_mask_sha256": self.bdy_mask_sha256,
            "n_cells": self.n_cells,
            "boundary_zone_width": self.boundary_zone_width,
            "class_id": self.class_id,
            "mesh_row": self.mesh_row,
            "card": self.card,
            "admitted_on": self.admitted_on,
            "contract_receipt": self.contract_receipt,
            "basis": self.basis,
        }


# ---------------------------------------------------------------------------
# the two registries
# ---------------------------------------------------------------------------

#: The kernel-set digest the shipped classes were minted at.  It is a
#: MEASURED constant of this tree, not a declaration: ``tests/
#: test_regional_forecast_anchor.py`` re-computes it from the sources and
#: refuses a mismatch by name, so a kernel edit that silently kept an anchor
#: alive is impossible.
#: RE-MINTED 2026-08-27 for the provenance scrub.  Three of the
#: fourteen hashed sources moved -- ``cuda_driver.py`` and ``regional_v841.py``
#: carried a person's name, a node label and a machine-shaped rig description
#: in comments and docstrings, and those had to leave the PRIVATE tree rather
#: than the public copy, because this digest is evaluated at ADMISSION time
#: from the shipped tree: a copy scrubbed after the fact refuses every regional
#: CUDA run at the door of the published package while its wheel builds clean
#: and its audit passes.  The move is measured inert
#: (``evidence/scrub-pins-20260827/``: identical ASTs with docstrings stripped
#: and string constants normalised), but a mint is a statement about BYTES, so
#: it lapsed and was RE-TAKEN rather than re-pointed -- both rows below carry
#: fresh two-process mint pairs on this digest.
#:
#: RE-DERIVED 2026-08-28 for the 0.2.0 package rename, and this is a different
#: KIND of move from the one above, which is why it is written out rather than
#: folded into that paragraph.  The provenance scrub moved BYTES: three sources were edited and
#: the two shipped classes were RE-MINTED on the card, two fresh 1,080-step
#: pairs, 17 min 29 s of a 5090.  The rename moved NAMES and the spelling of
#: one identifier: all fourteen declared names changed prefix, and eleven of
#: the fourteen files changed bytes by the substitution of the token
#: ``mpas_port`` with ``woof.hex`` AND NOTHING ELSE -- measured, not argued.
#: Each of those eleven is byte-identical to its pre-rename self once that one
#: identifier is substituted, which is a stronger statement than the provenance scrub's
#: AST comparison and needs no docstring-stripping to make.  The 32 lines
#: carrying the token are 19 comment, docstring and cross-reference lines, 12
#: NVRTC ``module_key`` / module-name labels, and one import.  Not one is
#: inside a CUDA source string, so no kernel's compiled text differs, and the
#: only thing a ``module_key`` selects is a cache slot and an observer label.
#:
#: WHAT IS THEREFORE OWED, said plainly rather than dodged: the two rows below
#: name mint receipts taken at kernel set ``237630342e...``, and this constant
#: is no longer that value.  Those receipts are NOT re-pointed and their prose
#: is not edited to pretend otherwise.  A re-mint at this digest -- the four
#: 1,080-step integrations the provenance scrub paid for, on a card -- is an OPEN follow-up of
#: the 0.2.0 cut and was not taken when this constant was re-derived.
#: The constant moved anyway because the alternative is worse by a wide margin:
#: a stale value here is not a failing test, it is ``require_regional_anchor``
#: refusing every limited-area run at the door of the shipped package, which is
#: the wheel-reach defect's breakage restaged.  Only one half of the anchor is in
#: question: every cull's contract deck still measures device bits against the
#: v8.4.1 CPU authority on its own zone geometry, and a spelling cannot touch
#: that.
#:
#: RE-DERIVED 2026-09-14 for the host path of the forecast step (the hex
#: performance campaign's hostpath lane).  Three of the fourteen sources
#: moved, all of them HOST Python and none of them a CUDA source string:
#: ``cuda_backend/runtime.py`` (the admission is measured once per process
#: and answered from memory; ``launch_checked`` launches once with no event
#: and no synchronize), ``cuda_driver.py`` (its two ``recover_state`` calls
#: pass ``timing_repeats=0``, so each recovery kernel is launched once instead
#: of twice), and ``cuda_regional_forecast_v841.py`` (the garbage discipline
#: restores every padded argument of a launch in one launch, and the density
#: and recovered-state validations set the step's flag on the device instead
#: of draining the stream per array).  Every regional kernel's source is
#: byte-unchanged -- ``cuda_regional_v841.py`` is not among the three, and the
#: contract deck re-run on the point cull at this digest records the same
#: ``translation_unit.source_sha256`` as the receipt taken at the previous
#: digest -- and the shared translation units' compile keys are the same in
#: both arms' receipts.  What was RE-MINTED at this digest: the point class
#: ``graded-869m-dt5-z7``, on the 43,884-cell point cull, in the same session
#: as a forecast of the untouched tree (its row says what was measured).  The
#: other four classes carry this digest through the constant and record the
#: lapse here rather than pretend a re-mint; their kernels launch the same
#: compiled text they were minted on.
#:
#: RE-DERIVED 2026-09-14 for the element-parallel launch geometry of the
#: level-independent dycore kernels (the hex performance campaign's kernels
#: lane).  Six of the fourteen sources moved, and this time the CUDA source
#: strings themselves moved: ``cuda_driver.py``, ``cuda_dynamics_v841.py``,
#: ``cuda_horizontal.py``, ``cuda_horizontal_v841.py``,
#: ``cuda_regional_v841.py`` and ``cuda_transport.py`` re-map every kernel
#: whose levels are independent from one thread per owner looping over
#: levels to one thread per ``(tracer, level, owner)`` element in a
#: grid-stride loop, and ``cuda_regional_forecast_v841.py`` launches them
#: per element; the momentum kernel gathers a once-per-run reference wind
#: instead of evaluating cosf/sinf per neighbour per stage.  What did NOT
#: move is the per-element arithmetic: each element runs the former loop
#: body for its level with the same operands in the same order, the
#: vertically implicit column solve keeps its per-column form, and the claim
#: is measured rather than argued -- the cull's own contract deck at this
#: digest (8 of 8 decks bitwise against the v8.4.1 CPU authority, 22 of 22
#: kernels, dual run identical) and a per-kernel capture-and-replay of every
#: re-mapped kernel against the previous translation units on the point
#: cull's live state (``tools/kernel_element_ab.py``).  What was RE-MINTED
#: at this digest: the point class ``graded-869m-dt5-z7``, on the 43,884-cell
#: point cull, in the same session as a forecast of the untouched tree (its
#: row says what was measured).  The other four classes carry this digest
#: through the constant and record the lapse here rather than pretend a
#: re-mint: their kernels now launch re-mapped text, and the evidence that
#: the re-mapped text computes the minted bits is the deck and the per-kernel
#: replay above, not a re-run of their own mints.
#:
#: RE-DERIVED 2026-09-14 for the landing of the two moves above together
#: (the hex performance campaign's kernels-land lane).  The host-path move
#: and the element-parallel re-map were taken on separate branches from the
#: same base and each re-derived this constant on its own; merged, the
#: fourteen sources carry both: ``cuda_driver.py``,
#: ``cuda_backend/recovery.py`` and ``cuda_regional_forecast_v841.py`` hold
#: a lane each in different lines (the CUDA source strings are the re-map's,
#: the host path's once-per-kernel launch helper around them), and the other
#: eleven are one lane's or neither's.  One line moved beyond the union:
#: ``cuda_regional_forecast_v841.py`` gained a public reading of the garbage
#: discipline's unit-pool registry (``bound_to_unit_pool``), so the
#: per-kernel replay can reproduce a launch's scrub after the run; it reads
#: a dictionary and launches nothing.  What was RE-MINTED at this digest:
#: the point class ``graded-869m-dt5-z7`` on the 43,884-cell point cull (its
#: row says what was measured).  The other four classes carry this digest
#: through the constant and record the lapse here rather than pretend a
#: re-mint, on the reasoning of the two notes above: the cull's contract
#: deck at the merged text and the per-kernel replay against the integrated
#: tree's translation units are the evidence that the merged text computes
#: the minted bits.
#: Integration, 2026-10-04: approved device selection and rank constructor
#: dispatch change two host source files. All twelve arithmetic source files
#: remain byte-identical to e6feb2f. The former source set was
#: 8a716f28862a3a89006b017397ad004cb05699a539ac74c9d9b283ba86657818.
#: verification/hex-next/native-point.json records eight matched native operator decks,
#: twenty-two kernels, identical dual runs and effective mutation controls on
#: the current source set. This is a point contract proof. Other class records
#: are carried, not newly measured; forecast and regional rank gates remain
#: separate from this dry contract.
#: Startup lane, 2026-10-05: ``cuda_driver.py`` gained the host-preparation
#: cache (woof.hex.prep_cache) on its two host validations, which run before
#: any CUDA interaction.  No CUDA source string, launch or arithmetic line
#: moved; the former source set was
#: 6e92b86e418fb748931f0469eb0e0a7007cc72c381b5e1128ec2f95a7e3b3394.  The
#: conus12km same-card before/after pair in
#: evidence/hex-startup-20261005/ compares every step's execution boundary
#: and both history files byte for byte.  Class records are carried, not
#: newly minted.
#: Certificate-gate retirement, 2026-10-10 (branch hex/retire-cert-gates):
#: this digest moved twice that day and the first move carried no note.
#: (1) ``cuda_driver.py`` and ``cuda_regional_forecast_v841.py`` stopped
#: refusing for a missing certificate (scalar_advection off,
#: positive_definite, apvm=0, an unregistered regional class) and record the
#: match instead; the former source set was
#: 6a38fdc7a60e94fb61595d25dd80a9777c2dce62213103d0cf7952f623b92d10, then
#: 31598b50e5618979b10243489b92127de71b61dd1aa6a6962b6903ae8ac2c159.
#: (2) The regional scalar path in ``cuda_driver.py`` takes the limited
#: routine under config_monotonic OR config_positive_definite, as native and
#: the global path do.  That is an ARITHMETIC change for a regional run with
#: positive_definite on and monotonic off, and for no other configuration:
#: with monotonic on, or both off, the same kernels launch with the same
#: arguments.  NO REGIONAL MINT RE-RAN for either move.  Every class record
#: below is carried from the source set it was measured on, and since
#: 2026-10-10 a class record is evidence the driver reports, not a
#: condition of running.
#: (3) 2026-10-11: ``cuda_transport_v841.py`` releases the limiter's work
#: arrays early (`del` only; no launch or arithmetic line moved); the former
#: source set was
#: 94388a96d39ab23367c06fe51079be3f9eae4d39048400bbbfb07b9197d09998.
#: Integration, 2026-10-11: forward merges retain the certificate retirement,
#: ops changes and the one-card ordering fix. The one-card ordering branch
#: also moved this digest without a regional measurement. This value is the
#: combined source set, not a new regional mint. No regional proof has been
#: rerun for this integration. Existing class records keep their old scope.
#: Forecast batch, 2026-10-13 (branch hex/gpu-forecast-batch):
#: ``cuda_driver._validate_v841_host_mesh`` evaluates its row laws over whole
#: tables instead of row loops (host validation before any CUDA interaction;
#: tests/test_mesh_rows_vectorized.py keeps the loops and proves the same
#: verdict and message).  No CUDA source string, launch or arithmetic line
#: moved; the former source set was
#: af121cffb085e14bd90c5fea0526529ff18351659c9c9b9e4c6d82d2600aec79.  Class
#: records are carried, not newly minted.
#: SG2 lagged microphysics heating, 2026-10-13 (branch correct/storm-growth;
#: moved by 0c3a9624 and again by its repair): ``cuda_driver.py`` gained
#: MPAS-A's rt_diabatic_tend consumer (add_lagged_diabatic_theta_v842_f32 in
#: every RK stage, mpas_atm_time_integration.F:6937), its removal at the final
#: recovery inside recover_cells_f32 (F:4436-4437) and, in the repair, the
#: specified-zone zero (zero_speczone_lagged_diabatic_theta_v842_f32, F:7949).
#: This is an ARITHMETIC change for every run WITH physics, regional runs
#: included: the ArWen and suite backends carry the heating from the second
#: step on.  With physics off nothing produces a carry, neither new kernel
#: launches and recover_cells_f32 runs with has_diabatic = 0; the dry derecho
#: fixed-step and active-LTS final states are byte-identical before and after,
#: every word (the correct/storm-growth report).  NO REGIONAL MINT RE-RAN.
#: The former source set was
#: 77b7ce2c24fc7d6dc5f98df4e1a358ea3b871c50d3f35f7fd048482289fff387, then
#: 92e658ffd812e7e8c81db0287a31d9b7ac13d4342f30c00fc3c99b5caa87acd9 at
#: 0c3a9624 (before the specified-zone zero).  Class records are carried,
#: not newly minted; re-minting the regional classes with physics is a named
#: follow-up of that report.
#: (4) 2026-10-14 (hex 0.4.0 release, lane H7): the MPAS v8.4.2 port's LAM
#: identity landing (d4-lam-identity: the native lateral-boundary module,
#: native regional acoustic rounding, gradual underflow, native LBC scalar
#: storage) and the golden 0.4.0 merge moved the regional source set from
#: af121cffb085e14bd90c5fea0526529ff18351659c9c9b9e4c6d82d2600aec79.  Regional
#: runs moved to native MPAS behaviour by design and are proven bitwise
#: against native by that unit's replay and the G1/G2 regional decks.  The
#: value below is the combined source set re-derived by
#: tools/repin_source_tables.py, not a new regional mint; the class records
#: keep their old scope (evidence the driver reports, not a condition of
#: running).
#: (5) 2026-10-14 (hex 0.4.0 release, H1 merged onto the H7 merge): the
#: forecast-batch and SG2 moves noted above, the LAM identity move of (4)
#: and the 0.4.0 release scrub's comment and docstring lines meet in one
#: source set.  The two lines carried
#: d3ef6d459251862710b4e4542b42750366dda586c27262ed1c8e4190d169d552 (H7) and
#: b215702d1f4d1c620cc9c190ebe3f5328145baa6af2cfc5f0dbc304a8299d62b (H1).
#: The value below is the combined source set re-derived by
#: tools/repin_source_tables.py, not a new regional mint; no regional proof
#: re-ran for this merge, and the class records keep their old scope.
#: (6) 2026-10-15 (hex 0.4.0, CUDA 12 support): ``cuda_backend/runtime.py``
#: moved and nothing else in the set.  ``require_cuda``'s default
#: ``min_runtime_version`` went from 13000 to 12000 and its docstring gained
#: one paragraph saying why: HOST Python, no CUDA source string, launch or
#: arithmetic line moved.  The former source set was
#: 7ce8d4c44a2fca15742c79e823f3d07d7426c19b1e343dae596829cad93b3822.  The
#: value below is re-derived by tools/repin_source_tables.py, not a new
#: regional mint; the class records keep their old scope.
#: (7) 2026-10-15 (hex 0.4.0 fix, LES on the MPAS route): cuda_driver.py and
#: config_v841.py moved to wire MPAS-A v8.4.2's LES dissipation into the RK
#: step. With LES off the launches are unchanged (the moist w Euler kernel
#: takes a has_mixing flag, 0 on that path, and subtracts from 0.0f as
#: before). The value below is re-derived by tools/repin_source_tables.py on
#: the composed tree (CUDA 12 support plus LES), not a new regional mint; no
#: regional proof re-ran, and the class records keep their old scope.
#: (8) 2026-10-16 (hex memory census cuts, branch hex/mem-census-cuts):
#: cuda_driver.py (mem/state-levels: read-only time levels bound, not copied;
#: regional keeps its private images), cuda_transport.py, containers.py and
#: recovery.py (mem/mesh: device mesh tables at the stride the mesh uses) and
#: cuda_transport_v841.py and partition_executor_v841.py (scalar transport a
#: species per pass, round E split into one part per pass on ranks;
#: regional included: the same kernels on per-species slices)
#: moved.  No kernel source or arithmetic line moved for the regional lane.
#: The value below is re-derived by tools/repin_source_tables.py on the
#: composed tree, not a new regional mint; no regional proof re-ran, and the
#: class records keep their old scope.
#: (9) 2026-10-16 (hex/speed-1): corrected theta finish reuses the same
#: cell stencil across four levels. The two sums retain their original
#: operand and helper order. Direct old/new uint32 tests cover 1/4/5/55
#: levels and 0/5/6/10/11 neighbors; the 60 km six-hour forecast matches
#: all 78 published field hashes. G1/G2, including hl-3, are the landing
#: proof for this source set (see the lane's REVIEW-speed-1.md). This is
#: a source-table refresh by tools/repin_source_tables.py, not a new mint
#: of the archived regional classes; their original scope is unchanged.
#: (10) 2026-10-16 (hex/global-defects, carried by hex/speed-2): aot.py
#: now loads the source-bound prebuilt kernel pack, with the existing NVRTC
#: fallback. It is the only changed member of this 18-file set since (7);
#: no CUDA source string in this set changed. The value below is re-derived
#: by tools/repin_source_tables.py. This is NOT a new regional class mint:
#: archived receipts and their original scope remain unchanged. The speed-2
#: landing decks are separate current-tree evidence, not replacement mints.
#: (11) 2026-10-16 (hex/speed-3): cuda_driver.py hashes stable host arrays
#: through byte views and bounded parallel SHA-256 calls. No CUDA source or
#: arithmetic changed. This current-source refresh is not a regional mint;
#: archived receipts retain their original scope, with separate landing proof.
#: (12) 2026-10-16 (hex 0.4.1 landing batch): the composed tree of (8)-(11)
#: together. The value below is re-derived by tools/repin_source_tables.py
#: on that tree; not a new regional mint, archived scope unchanged.
#: (13) 2026-10-16 (hex/fast-numerics, owner ruling: fast numerics is the
#: default, strict the opt-in): cuda_regional_v841.py and runtime.py take
#: their compile options from woof.hex.numerics. Under --numerics strict the
#: options are the unit's original tuple, unchanged; no CUDA source string
#: changed. Re-derived by tools/repin_source_tables.py; not a new regional
#: mint, archived scope unchanged (the mints are strict-numerics statements).
#: (14) 2026-10-17 (hex land-1017 batch): the LES-with-ArWen-physics change
#: (cuda_driver.py keeps dpdz alive for the LES w-tendency read; species row
#: from arwen_les.model_row) and the routeb-door merge, composed. No CUDA
#: source string changed. Re-derived by tools/repin_source_tables.py; not a
#: new regional mint, archived scope unchanged.
#: (15) 2026-10-17 (hex/fix-startup-onecore): containers.py constructs
#: edge signs on the device; host validation and compaction use bounded
#: exact array blocks. Direct CUDA sign tests preserve active signs and
#: inactive +0 words. This refresh is not a new regional forecast mint.
#: (16) 2026-10-19: refresh current sources after f65ab11c specializes
#: immutable dimensions in six kernels. Generated CUDA source and cache keys
#: now include shape. Archived mints keep their recorded scope; this source
#: refresh is not a new regional forecast or installed device proof.
#: (17) 2026-10-19: refresh lookup identities after the LTS hook repair.
#: Generic keys and the specialized acoustic tap now agree; CUDA source
#: arithmetic is unchanged. This is not a new forecast mint.
#: (18) 2026-10-20: the AOT loader resolves pinned on-demand assets instead
#: of package-data packs. Compiler checks, cubin checks, CUDA source and
#: arithmetic are unchanged. This source-table refresh is not a new regional
#: forecast mint; archived evidence retains its recorded scope.
#: (19) 2026-10-20 (capabilities campaign landing, lane/hexcap-base):
#: cuda_dynamics_v841.py keys unspecialized kernels by (cache, name) so the
#: moving-LTS interface taps (b6ebb526) replace the same entry the step
#: launches. A host-side cache key only; no CUDA source string or
#: arithmetic changed. Re-derived by tools/repin_source_tables.py; not a
#: new regional forecast mint, archived scope unchanged.
#: (20) 2026-10-24 (hex 0.4.5 landing batch): host-side changes only in
#: this set since (19): aot.py accepts pack banners through
#: compile_contract.build_id_accepted (865a117); runtime.py memoises kernel
#: resolution and records shape origins (U-MPASPHYS-1, 5a270ec);
#: cuda_driver.py loads every moist species on a limited-area MPAS-suite run
#: (47f7459; the regional qv-only path is unchanged for the dycore's own
#: runs); cuda_horizontal_v841.py adds solve_vorticity, which launches the
#: existing vertex_diagnostics_v841_f32 kernel unchanged (G1);
#: cuda_transport_v841.py a comment. No CUDA source string or arithmetic
#: changed. Re-derived by tools/repin_source_tables.py; not a new regional
#: forecast mint, archived scope unchanged.
#: (21) 2026-10-27 (hex 0.4.9 all-fixes batch): runtime.py makes the host
#: wait on the card with blocking sync instead of CUDA's spin (c151b3e9; the
#: main thread held one core at 100 %). The only changed member of this set
#: since (20); no CUDA source string or arithmetic changed. Re-derived by
#: tools/repin_source_tables.py; not a new regional forecast mint, archived
#: scope unchanged.
#: (22) 2026-10-27 (hex 0.4.9 fold-everything batch): cuda_transport_v841.py
#: tells an exchanger that declares begin_species_pass which species each
#: limiter pass carries, and an exchanger declaring fct_scale_parts None takes
#: the route default (fix/lts-share-0.4.9 7b2ac42a, the one-card active LTS
#: class exchanger). Regional exchangers declare neither, so their passes are
#: unchanged. The only changed member of this set since (21); no CUDA source
#: string or arithmetic changed. Re-derived by tools/repin_source_tables.py;
#: not a new regional forecast mint, archived scope unchanged.
#: (23) 2026-10-28 (HEX-LTS-REGRESSION, fix/lts-zoom-speed): cuda_driver.py
#: lets the one-card full-step LTS executor read a class step's four-byte
#: validation flag at the macro boundary (deferred_validation_v841, a list
#: only that executor sets; flush_deferred_validation_v841). A regional
#: driver never carries the list, so its step reads the flag in the step as
#: before. The only changed member of this set since (22); no CUDA source
#: string or arithmetic changed. Re-derived by tools/repin_source_tables.py;
#: not a new regional forecast mint, archived scope unchanged.
#: (24) 2026-10-28: bounded transport batches and repeated-zero result reuse.
#: CUDA arithmetic source strings and boundary kernels are unchanged. Host
#: launch checks compare every output word against the one-species path,
#: including regional masks; halo round schedules retain their declared parts.
#: This is a current-source repin, not a new regional forecast mint. The
#: archived scope is unchanged; release still requires the device bitwise
#: transport gate and same-card timing receipt (docs/transport-batching.md).
#: (25) 2026-10-28: integer zero classification reduces cells per level,
#: then levels per species. Floating-point CUDA source is unchanged; fast
#: and strict transport checks pass. This updates current source identity,
#: not the archived regional forecast scope or its recorded hashes.
#: (26) 2026-10-28 (rel/hex-0.4.10 fold): (23) and (24)-(25) landed on separate lines; the merged
#: tree's digest is re-derived by tools/repin_source_tables.py, not a new mint.
#: (27) 2026-10-28 (rel/hex-0.4.10 fold): guide/l11-regional-iau lets open_regional_forecast_v841 take the
#: IAU stage-density driver type on the one-card plain MPAS-A regional dycore; host construction only, no CUDA
#: source string or arithmetic changed. Re-derived by tools/repin_source_tables.py; not a new mint.
#: (28) 2026-10-29 (rel/hex-0.4.10 fold HB7): fix/hex-mc-validate-once validates the device mesh and context once
#: per mesh (containers.py, cuda_regional_forecast_v841.py host checks); no CUDA source or arithmetic changed.
#: Re-derived by tools/repin_source_tables.py; not a new mint.
#: (29) 2026-10-29 (rel/hex-0.4.10 fold HB9): rrfs/speed3-dycore d38acc4 runs validate_recovered_v841_f32 and
#: reconstruct_winds_v841_f32 one lane per (level, cell/edge) instead of a per-cell level loop (same checks, same
#: per-element arithmetic and order; hl-3 BITWISE, strict RRFS 0 of 29,644,818), and the 44c0849 re-pin
#: (guide/repin-fold-0410) adds NOAA-only host paths to cuda_driver.py (config_lbc_w nearest, scalar eddy mix)
#: and forms regional rthdynten from the boundary-adjusted density tendency, as native MPAS-A v8.4.2 and 44c0849
#: both do (a physics-input diagnostic; the dry mint's prognostic arithmetic is unchanged). Re-derived by
#: tools/repin_source_tables.py; not a new mint.
#: (30) 2026-10-29 (rel/hex-0.4.10 fold HB10): fix/c3-4card-vram-gated (rrfs/c3-4card-vram 0e6fcb7) holds the
#: limited-area driving pool on the boundary rings plus three neighbour rings and scatters full-extent arrays from
#: it (cuda_regional_forecast_v841.py, host CuPy indexing; zeros elsewhere, the same words at every element a
#: boundary kernel reads; c3s 1 h 0 of 322,061,158 words vs its base); no CUDA source or arithmetic changed.
#: Re-derived by tools/repin_source_tables.py; not a new mint.
#: (31) 2026-10-29 (rel/hex-0.4.10 fold HB11, rrfs/one-tree-fast 39a1af5b): cuda_driver._shares_start_levels lets
#: the limited-area lane bind its step-start and subcycle-start levels by reference, as the global lane already
#: does (WOOF_HEX_MEM_STATE_LEVELS_REGIONAL=0 restores the private images); host-side array lifetimes only, no
#: CUDA source string or arithmetic changed. The only changed member of this set since (30). Re-derived by
#: tools/repin_source_tables.py; not a new mint.
#: (32) 2026-10-29 (rel/hex-0.4.10 fold HB12, fix/rrfs-c3km-startup 1535cae8): cuda_backend/aot.py computes the
#: kernel pack digest and the compile-platform binding once per process instead of once per kernel cache; host
#: caching only, no CUDA source string or arithmetic changed (every prepared rank input identical base vs fix, 23,817
#: entries). The only changed member of this set since (31). Re-derived by tools/repin_source_tables.py; not a new
#: mint.
#: (33) 2026-10-29 (rel/hex-0.4.10 fold HB15, fix/c3-4card-ring-hb12b 1c8210c8): cuda_regional_forecast_v841.py runs
#: the limited-area boundary scalars and the LBC admission on the boundary support (host CuPy indexing, a4cd216e and
#: 3476a320) and launches the regional scalar-transport kernels per tracer group of <= 2^28 edge values instead of all
#: tracers at once (1c8210c8: the 4-rank conus3km int grid-stride overflow); every tracer is computed on its own, no
#: CUDA source string or arithmetic changed (mv3km 1 h 0 of 683,465,844 words vs 236be507's paths). The only changed
#: member of this set since (32). Re-derived by tools/repin_source_tables.py; not a new mint.
#: (34) 2026-10-29 (fix/hex-nvrtc-parallel): cuda_backend/aot.py moves the raw NVRTC compile into one function
#: (raw_nvrtc) the kernel plan's pool workers also run, and raw_image first takes an image the pool already
#: compiled; the same createProgram/compileProgram/getCUBIN call, no CUDA source string or arithmetic changed.
#: The only changed member of this set since (33). Re-derived by tools/repin_source_tables.py; not a new mint.
#: (35) 2026-10-29 (fix/rrfs-shared-dycore-8card): cuda_driver._step_device_v841 hands the step's four-byte
#: validation flag to deferred_checks_v842 when an MPAS-physics forecast loop collects (read at most one step
#: later, before any output), as cuda_driver_rrfs has since cb2ae586; elsewhere it is read at once as before. Host
#: read timing only, no CUDA source string or arithmetic changed. The only changed member of this set since (34).
#: Re-derived by tools/repin_source_tables.py; not a new mint.
#: (36) 2026-10-30 (woof hex 0.4.11 fold HB2, feat/hex-moving-resident 78111207): cuda_driver.py's from-host
#: constructor passes an optional upload_cache through to the advection-coefficient upload (None keeps the old call),
#: and cuda_backend/runtime.py + containers.py reuse unstepped predicted-class arrays with exact incremental uploads;
#: host upload bookkeeping only, no CUDA source string or arithmetic changed (the row's whole run: 0 of 1,240,994,505
#: words vs A2). Re-derived by tools/repin_source_tables.py; not a new mint.
MINTED_KERNEL_SET_SHA256 = (
    "1726f6bd1f317f24fbe78c2fbe01a5499c07ef5cd8efaf89fa966253c0dfb4a6"
)

_DRY_MINT_CAVEAT = (
    "WHAT THIS CLASS DOES NOT ADMIT: the mint pair is the DRY regional "
    "dycore, as every regional mint this program has run is.  It certifies "
    "the 22 boundary kernels and says nothing on its own about the WOOF "
    "physics seam a full-physics run also drives; the full-physics arms are "
    "admitted on the reasoning the regional-physics lane recorded, that the "
    "boundary machinery is the same 22 kernels either way, and that "
    "reasoning is inherited rather than re-earned.  Endpoint byte-identity "
    "against the compiled reference is NOT measured here, and is not "
    "measured for the global lane either."
)

#: Configuration classes holding a verified forecast mint.  Adding a row is
#: table work -- one entry naming evidence that exists in this repository.
ADMITTED_CLASSES: Mapping[str, RegionalClass] = MappingProxyType(
    {
        "graded-3315m-dt20-z7": RegionalClass(
            class_id="graded-3315m-dt20-z7",
            key=RegionalClassKey(
                boundary_zone_width=7,
                n_vert_levels=55,
                finest_edge_mm=3_314_810,
                dt_ms=20_000,
                # Minted at the kernel set of the species-row generalization
                # (the door lane's chain landed 2026-09-01), ce63c3dd...,
                # measured at the mint; the host-path move and the
                # element-parallel re-map of 2026-09-14 each carried it onto
                # the tree's constant like every other row, and the
                # constant's own note says what moved, what was measured
                # and what was re-minted.
                kernel_set_sha256=MINTED_KERNEL_SET_SHA256,
            ),
            parent="generated graded parent: 3.75 km cap of 480 km radius at "
            "42.0 N 86.0 W on a 120 km background, 120,941 cells",
            card="NVIDIA GeForce RTX 3080 (sm_86)",
            admitted_on="2026-09-02",
            mint_receipts=(
                "evidence/hex-mod-20260901/class-mint/mint-run1/forecast.json",
                "evidence/hex-mod-20260901/class-mint/mint-run2/forecast.json",
            ),
            minted_on_geometries=("r3.75.120.35842",),
            basis=(
                "The class of a 3.75 km limited-area cull of a generated graded "
                "parent: seven boundary rings, 55 levels, the cull's own finest "
                "edge of 3,314.810 m, dt 20 s (a 23.867 s Courant limit at "
                "125 m/s x 0.9), and this tree's regional kernel set after the "
                "species-row generalization.  Two independent processes ran "
                "1,080 steps -- six forecast hours from 2026-08-12_06:00:00, "
                "config_apply_lbcs=true, config_len_disp 3,750 m, hourly GFS "
                "boundaries -- in 915 s and 866 s on the desktop RTX 3080 "
                "beside another tenant, and all THIRTEEN published history frames "
                "agree under masked digests while every whole-file digest differs "
                "on its own file_id.  Minted on the geometry it will run on; the "
                "class inherits the graded-parent lane's finding that the "
                "geometry is not in the key, and pays for that inheritance with "
                "the cull's own contract deck below.  "
                + _DRY_MINT_CAVEAT
            ),
        ),
        "graded-4457m-dt20-z7": RegionalClass(
            class_id="graded-4457m-dt20-z7",
            key=RegionalClassKey(
                boundary_zone_width=7,
                n_vert_levels=55,
                finest_edge_mm=4_457_233,
                dt_ms=20_000,
                kernel_set_sha256=MINTED_KERNEL_SET_SHA256,
            ),
            parent="v4.75.121182",
            card="NVIDIA GeForce RTX 5090 (sm_120)",
            admitted_on="2026-09-01",
            mint_receipts=(
                "evidence/door-species-20260901/graded-class-mint/"
                "mint-run1/forecast.json",
                "evidence/door-species-20260901/graded-class-mint/"
                "mint-run2/forecast.json",
            ),
            minted_on_geometries=("r4.75.14050",),
            basis=(
                "The class every storm-following cull of a 4.6 km graded "
                "parent belongs to: seven boundary rings, 55 levels, the "
                "parent's own finest edge of 4,457.233 m, dt 20 s, and this "
                "tree's regional kernel set.  Two independent processes ran "
                "1,080 steps -- six forecast hours from 2026-08-12_06:00:00, "
                "config_apply_lbcs=true, config_len_disp 4,000 m -- and all "
                "THIRTEEN published history frames agree under masked digests "
                "while every whole-file digest differs on its own file_id.  "
                "THE CLASS IS EARNED ONCE AND THE EVIDENCE THAT IT IS A CLASS "
                "IS SEPARATE FROM THE MINT: the nest-ratio lane minted FIVE "
                "concentric culls of this same parent independently -- 4,440, "
                "7,975, 11,020, 14,050 and 15,755 cells, domains from 155 to "
                "565 km in radius, interface ratios from 13.5:1 to 3.6:1 -- "
                "and every one of the five returned the same verdict at the "
                "same 1.605x Courant margin under a 32.092 s limit.  Not one "
                "input this instrument reads distinguished them, which is why "
                "the geometry is not in the key.  RE-MINTED 2026-08-27 at "
                "the kernel set the provenance scrub produced, on the "
                "same 14,050-cell cull and the same namelist: 1,080 steps in "
                "two independent processes, 192 s and 184 s, all THIRTEEN "
                "published frames identical under masked digests and every "
                "whole-file digest differing on its own file_id.  The "
                "superseded pair was taken at kernel set bdcfe014... and its "
                "receipts remain in the tree as the record of that run.  "
                "  RE-MINTED 2026-09-01 at the kernel set the species-row door generalization produced: cuda_driver.py and cuda_regional_forecast_v841.py both moved, so every class lapsed exactly as a whole-file digest is meant to make them.  Same 14,050-cell cull, same namelist, 1,080 steps in two independent processes, 192 s and 183 s, all THIRTEEN published frames identical under masked digests (0 of 13 differing) and every whole-file digest differing on its own file_id (13 of 13).  What moved in the kernel set is measured rather than assumed: of the six kernels in the shared translation unit, only the two moist-coefficient kernels changed PTX at compute_120, and the regional step launches neither (evidence/door-species-20260901/regional-ptx-diff.txt).  "
                + _DRY_MINT_CAVEAT
            ),
        ),
        "conus-x1-120km-dt120-z7": RegionalClass(
            class_id="conus-x1-120km-dt120-z7",
            key=RegionalClassKey(
                boundary_zone_width=7,
                n_vert_levels=55,
                finest_edge_mm=UNMEASURED_FINEST_EDGE_MM,
                dt_ms=120_000,
                kernel_set_sha256=MINTED_KERNEL_SET_SHA256,
            ),
            parent="x1.2562 (the v8.4.1 regional reference)",
            card="NVIDIA GeForce RTX 5090 (sm_120)",
            admitted_on="2026-09-01",
            mint_receipts=(
                "evidence/door-species-20260901/conus-class-mint/"
                "mint-run1/forecast.json",
                "evidence/door-species-20260901/conus-class-mint/"
                "mint-run2/forecast.json",
            ),
            minted_on_geometries=("conus-x1.2971",),
            basis=(
                "The reference-cull class, at the coarse end: 120 km cells, "
                "dt 120 s, seven rings, 55 levels.  FOUR independent "
                "processes ran the CANDIDATE-REGIONAL-DRY namelist "
                "(config_apply_lbcs=true) for 90 steps -- three forecast "
                "hours from 2026-08-12_06:00:00.  Two were assembled with the "
                "gate bypassed, which is what earned it; two went through the "
                "gate afterwards and reproduced it.  All SEVEN published "
                "history frames agree under masked digests across all four, "
                "while every whole-file digest differs on its own file_id.  "
                "RE-MINTED at the merged tip after the NVRTC "
                "reciprocal-rewrite fix, which moved the third-order "
                "stencil denominator off a source literal in both cuda_driver "
                "and cuda_transport: the pre-fix pair is superseded.  ONE "
                "MEMBER ONLY, and that is a statement about the evidence "
                "rather than about the class -- no second cull of this parent "
                "has ever been made, so the five-member independence "
                "measurement the 4.6 km class carries has no counterpart "
                "here.  THE FINEST EDGE OF THIS CLASS IS NOT MEASURED and "
                "the row says so rather than guessing: this cull's grid "
                "publishes UNIT-SPHERE dcEdge and the Earth-scaled metrics "
                "sit in a static file no copy of this repository contains, "
                "which is why even the Courant admission reads the static for "
                "it.  The class therefore admits ONLY the geometries listed "
                "in SHIPPED_CONTRACTS and refuses a presented receipt, so it "
                "cannot grow to cover a cull nobody looked at.  Measuring the "
                "edge on a machine holding the reference mirror retires that "
                "restriction.  RE-MINTED 2026-08-27 at the kernel set the "
                "provenance scrub produced, on the same cull and the "
                "same CANDIDATE-REGIONAL-DRY namelist: 90 steps in two "
                "independent processes, 13 s each on an RTX 5090, all SEVEN "
                "published frames identical under masked digests and every "
                "whole-file digest differing on its own file_id.  The "
                "superseded 2026-08-26 pair was taken at kernel set "
                "bdcfe014... on an RTX 5070 Ti and its receipts remain in the "
                "tree as the record of that run.  " + _DRY_MINT_CAVEAT
            ),
        ),
        "graded-824m-dt5-z7": RegionalClass(
            class_id="graded-824m-dt5-z7",
            key=RegionalClassKey(
                boundary_zone_width=7,
                n_vert_levels=55,
                finest_edge_mm=824_453,
                dt_ms=5_000,
                kernel_set_sha256=MINTED_KERNEL_SET_SHA256,
            ),
            parent="generated graded 937.5 m core on a 120 km background",
            card="NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120)",
            admitted_on="2026-10-08",
            mint_receipts=(
                "docs/proofs/regional-class-824/mint-run1/forecast.json",
                "docs/proofs/regional-class-824/mint-run2/forecast.json",
            ),
            minted_on_geometries=("q0.9375.120.43642.n47.50w111.30",),
            basis=(
                "Exact millimetre class, with no finest-edge band. Two independent "
                "processes ran 2,160 dry regional steps at 5 s on the measured "
                "43,642-cell, 55-level cull. The initial and three-hour endpoint "
                "frames agree in masked file hashes and every published field "
                "hash; the exact Rust reader also compared every NetCDF variable "
                "word, including character clock bytes. Whole files have distinct "
                "file ids. Both runs used the same fourteen source hashes and "
                "configuration hash. The cull's independent contract passed eight "
                "decks, twenty-two kernels, dual identity and mutation controls. "
                "docs/proofs/regional-class-824/identity.json records the complete "
                "endpoint comparison. " + _DRY_MINT_CAVEAT
            ),
        ),
        "graded-869m-dt5-z7": RegionalClass(
            class_id="graded-869m-dt5-z7",
            key=RegionalClassKey(
                boundary_zone_width=7,
                n_vert_levels=55,
                finest_edge_mm=869_251,
                dt_ms=5_000,
                kernel_set_sha256=MINTED_KERNEL_SET_SHA256,
            ),
            parent="v0.9.120.110533",
            card="NVIDIA GeForce RTX 5070 Ti (sm_120)",
            admitted_on="2026-09-02",
            mint_receipts=(
                "evidence/hex-speed-20260902/class-mint-937/mint-run1/forecast.json",
                "evidence/hex-speed-20260902/class-mint-937/mint-run2/forecast.json",
            ),
            minted_on_geometries=("r0.9.120.40520",),
            finest_edge_band_relative=SAME_LADDER_FINEST_EDGE_BAND,
            basis=(
                "BAND 2026-09-13 (point door): a cull of a parent "
                "GENERATED AT ANOTHER POINT from the same 937.5 m ladder spec "
                "(woof hex mesh-plan --point) delivers the same rung and a "
                "finest edge that differs from 869.251 m by relaxation noise; "
                "it is this resolution class, and the key admits it within "
                "5 % either way -- a band that cannot reach the 711 m class "
                "beside it (18 % apart) and that the Courant admission "
                "re-measures against the cull's own dcEdge regardless.  "
                "The class of a 937.5 m limited-area cull of the registered "
                "v0.9.120.110533 parent, the first sub-kilometre class this "
                "program has minted: seven boundary rings, 55 levels, the "
                "cull's own finest edge of 869.251 m (the parent's, in the "
                "same float64 bits), dt 5 s (a 6.259 s Courant limit at "
                "125 m/s x 0.9, 1.252x), and this tree's regional kernel set "
                "at MINTED_KERNEL_SET_SHA256, measured at the mint.  Two "
                "independent processes ran 1,080 steps -- 1.5 forecast hours "
                "from 2026-08-28_18:00:00, config_apply_lbcs=true, "
                "config_len_disp 937.5 m, hourly GFS boundaries -- in 764 s "
                "and 562 s on the RTX 5070 Ti of the proving node beside other "
                "tenants (the first beside a 10.8 GiB one), and all THIRTEEN "
                "published history frames agree under masked digests while "
                "every whole-file digest differs on its own file_id.  Minted "
                "on the geometry it will run on, the 40,520-cell cull of the "
                "speed ladder; the class inherits the graded-parent lane's "
                "finding that the geometry is not in the key, and pays for "
                "that inheritance with the cull's own contract deck below.  "
                "RE-MINTED 2026-09-14 at the kernel set the step's host-path "
                "change produced (three host sources moved, no CUDA source "
                "string): on the 43,884-cell 937.5 m point cull of a parent "
                "generated at 41.95 N 94.79 W, HRRR 15Z init and hourly "
                "boundaries, 720 steps of 5 s, the hostpath tree ran twice in "
                "two processes on the proving RTX 5090 (215.3 s and "
                "214.3 s of integration) and both history frames agree "
                "byte for byte -- and agree byte for byte with the untouched "
                "0.3.0 tree's run of the same hour in the same session, so the "
                "kernel set moved and the published bytes did not; the cull's "
                "own contract deck passed at this digest (8 of 8 bitwise, 22 "
                "of 22 kernels, dual run identical).  "
                "RE-MINTED 2026-09-14 at the kernel set the element-parallel "
                "launch geometry produced, landed together with the host-path "
                "move (six CUDA-bearing sources moved, the per-element "
                "arithmetic did not): on the same 43,884-cell point cull, "
                "HRRR 15Z init and hourly boundaries, 2,160 steps of 5 s on "
                "the proving RTX 5090 in one process (567.7 s of integration), "
                "all four history frames agree byte for byte with the "
                "untouched 0.3.0 tree's run of the same three hours and with "
                "the integrated tree's; the cull's own contract deck passed at "
                "this digest (8 of 8 bitwise, 22 of 22 kernels, dual run "
                "identical) and a per-kernel capture-and-replay of the 49 "
                "re-mapped kernels the run launches was bitwise against the "
                "integrated tree's translation units on the run's own state.  "
                + _DRY_MINT_CAVEAT
            ),
        ),
        "graded-711m-dt5-z7": RegionalClass(
            class_id="graded-711m-dt5-z7",
            key=RegionalClassKey(
                boundary_zone_width=7,
                n_vert_levels=55,
                finest_edge_mm=710_869,
                dt_ms=5_000,
                kernel_set_sha256=MINTED_KERNEL_SET_SHA256,
            ),
            parent="generated 800 m parent: the registered 937.5 m ladder scaled "
            "by 0.8/0.9375 with the cap radii kept, 155,138 cells",
            card="NVIDIA GeForce RTX 5070 Ti (sm_120)",
            admitted_on="2026-09-02",
            mint_receipts=(
                "evidence/hex-speed-20260902/class-mint-800/mint-run1/forecast.json",
                "evidence/hex-speed-20260902/class-mint-800/mint-run2/forecast.json",
            ),
            minted_on_geometries=("r0.8.102.56619",),
            finest_edge_band_relative=SAME_LADDER_FINEST_EDGE_BAND,
            basis=(
                "BAND 2026-09-13 (point door): the same-ladder band the "
                "937.5 m class carries, for the same reason; the two bands do "
                "not overlap.  "
                "The finest class this program has minted: an 800 m limited-area "
                "cull of a generated 800 m parent, seven boundary rings, 55 "
                "levels, the cull's own finest edge of 710.869 m, dt 5 s (a "
                "5.118 s Courant limit at 125 m/s x 0.9, 1.024x, the tightest "
                "margin of any registered row), and this tree's regional kernel "
                "set at MINTED_KERNEL_SET_SHA256, measured at the mint.  Two "
                "independent processes ran 1,080 steps -- 1.5 forecast hours "
                "from 2026-08-28_18:00:00, config_apply_lbcs=true, "
                "config_len_disp 800 m, hourly GFS boundaries -- in 697 s and "
                "677 s on the proving node's NVIDIA GeForce RTX 5070 Ti, and all "
                "THIRTEEN published history frames agree under masked digests "
                "while every whole-file digest differs on its own file_id.  "
                "Minted on the geometry it will run on, the 56,619-cell cull of "
                "the speed ladder; the class inherits the graded-parent lane's "
                "finding that the geometry is not in the key, and pays for that "
                "inheritance with the cull's own contract deck below.  "
                + _DRY_MINT_CAVEAT
            ),
        ),
    }
)


#: Cull geometries whose own contract deck has been run and passed.  Keyed by
#: ``bdyMask`` digest, because that IS the zone geometry the deck measured.
SHIPPED_CONTRACTS: Mapping[str, RegionalContract] = MappingProxyType(
    {
        anchor.bdy_mask_sha256: anchor
        for anchor in (
            RegionalContract(
                bdy_mask_sha256=(
                    "62d1b5ef8626628ddecd87955132dc64fb742a68420a3b90956bbf9bd59d1d3d"
                ),
                n_cells=35_842,
                boundary_zone_width=7,
                class_id="graded-3315m-dt20-z7",
                mesh_row="r3.75.120.35842",
                card="NVIDIA GeForce RTX 3080 (sm_86)",
                admitted_on="2026-09-02",
                contract_receipt=(
                    "evidence/hex-mod-20260901/contract/records.json"
                ),
                basis=(
                    "8 of 8 decks bitwise against the v8.4.1 CPU authority over "
                    "124,848,922 float32 values compared as raw bit patterns (42 payloads), "
                    "22 of 22 declared kernels covered, dual-run stable across "
                    "two passes in one process, and every deck re-run with a "
                    "deliberately wrong zone geometry FAILS."
                ),
            ),
            RegionalContract(
                bdy_mask_sha256=(
                    "acc95da7ecc58253e0085332eb5acc827d42287ecae4fe9ea88bd64060f4d67e"
                ),
                n_cells=2_971,
                boundary_zone_width=7,
                class_id="conus-x1-120km-dt120-z7",
                mesh_row="conus-x1.2971",
                card="NVIDIA GeForce RTX 5070 Ti (sm_120)",
                admitted_on="2026-08-26",
                contract_receipt=(
                    "evidence/regional-cuda-l5-20260826/contract/contract-run1.json"
                ),
                basis=(
                    "8 of 8 decks bitwise against the v8.4.1 CPU authority "
                    "over 10,443,332 float32 values compared as raw bit "
                    "patterns, 22 of 22 declared kernels covered, dual-run "
                    "stable across two independent processes (84 of 84 "
                    "payload digests identical), and every deck re-run with a "
                    "deliberately wrong zone geometry FAILS."
                ),
            ),
            RegionalContract(
                bdy_mask_sha256=(
                    "2baf091d718efcbbcb3f9385d55b0f224c1ea54fc3d75bfa8f108fc8a1fca158"
                ),
                n_cells=11_020,
                boundary_zone_width=7,
                class_id="graded-4457m-dt20-z7",
                mesh_row="r4.75.11020",
                card="NVIDIA GeForce RTX 5070 Ti (sm_120)",
                admitted_on="2026-08-27",
                contract_receipt=(
                    "evidence/swath-as-lam-20260827/contract/contract-run1.json"
                ),
                basis=(
                    "The first regional geometry earned on a mesh this "
                    "program placed for itself: 121,182 cells become 11,020, "
                    "rings 9,220/240/245/251/257/263/269/275, Euler "
                    "characteristic 1.  8 of 8 decks bitwise against the "
                    "v8.4.1 CPU authority on THIS cull's zone geometry (544 "
                    "spec cells, 1,016 relax cells, 1,644 spec edges, 3,072 "
                    "relax edges), 22 of 22 declared kernels covered, "
                    "dual-run stable, every mutation control with teeth."
                ),
            ),
            RegionalContract(
                bdy_mask_sha256=(
                    "e2d18b34c866e2768896467f21ec11d908c00425ed36409e100d25827ed6b68e"
                ),
                n_cells=4_440,
                boundary_zone_width=7,
                class_id="graded-4457m-dt20-z7",
                mesh_row="r4.75.4440",
                card="NVIDIA GeForce RTX 5090 (sm_120)",
                admitted_on="2026-08-27",
                contract_receipt=(
                    "evidence/nest-ratio-20260827/anchors/d045/contract.json"
                ),
                basis=(
                    "The 0.45x cull of the same parent: 4,440 cells, rings "
                    "2,937/207/209/212/215/218/217/225.  8 of 8 decks bitwise "
                    "on THIS cull's zone geometry (442 spec cells, 854 relax "
                    "cells, 1,332 spec edges, 2,570 relax edges), 22 of 22 "
                    "kernels covered, every deck dual-run identical in "
                    "process, all eight mutation controls with teeth."
                ),
            ),
            RegionalContract(
                bdy_mask_sha256=(
                    "401c8c160948dadffef45cf75632970d73e8491b1799ea1639a3da794e76e29f"
                ),
                n_cells=7_975,
                boundary_zone_width=7,
                class_id="graded-4457m-dt20-z7",
                mesh_row="r4.75.7975",
                card="NVIDIA GeForce RTX 5090 (sm_120)",
                admitted_on="2026-08-27",
                contract_receipt=(
                    "evidence/nest-ratio-20260827/anchors/d070/contract.json"
                ),
                basis=(
                    "The 0.70x cull: 7,975 cells, rings "
                    "6,269/254/251/249/242/241/237/232.  Those ring "
                    "populations SHRINK outward, and that is the mesh being "
                    "variable resolution rather than torn.  8 of 8 decks "
                    "bitwise on THIS cull's zone geometry (469 spec cells, "
                    "983 relax cells, 1,399 spec edges, 2,935 relax edges), "
                    "22 of 22 kernels covered, all eight mutation controls "
                    "with teeth."
                ),
            ),
            RegionalContract(
                bdy_mask_sha256=(
                    "a8e66046452db881bb4a9da08952610207ee5aa2e0a58d48b1d2348b48f84088"
                ),
                n_cells=14_050,
                boundary_zone_width=7,
                class_id="graded-4457m-dt20-z7",
                mesh_row="r4.75.14050",
                card="NVIDIA GeForce RTX 5090 (sm_120)",
                admitted_on="2026-08-27",
                contract_receipt=(
                    "evidence/nest-ratio-20260827/anchors/d135/contract.json"
                ),
                basis=(
                    "The 1.35x cull -- the measured knee, and the cut every "
                    "shipped swath row takes from 2026-08-27: 14,050 cells, "
                    "rings 12,094/297/298/293/285/270/264/249, driven-ring "
                    "cells 10.86 to 14.30 km wide against a 71.0 km coarse "
                    "parent, about 5.7:1 at the interface.  8 of 8 decks "
                    "bitwise on THIS cull's zone geometry (513 spec cells, "
                    "1,146 relax cells, 1,508 spec edges, 3,405 relax "
                    "edges), 22 of 22 kernels covered, all eight mutation "
                    "controls with teeth."
                ),
            ),
            RegionalContract(
                bdy_mask_sha256=(
                    "1726f1a23b6140d21a2d96897939d7b9eda990b78d31579722cb6293f66533db"
                ),
                n_cells=15_755,
                boundary_zone_width=7,
                class_id="graded-4457m-dt20-z7",
                mesh_row="r4.75.15755",
                card="NVIDIA GeForce RTX 5090 (sm_120)",
                admitted_on="2026-08-27",
                contract_receipt=(
                    "evidence/nest-ratio-20260827/anchors/d170/contract.json"
                ),
                basis=(
                    "The gentlest boundary interface this program has run: "
                    "15,755 cells, rings 14,208/240/239/229/225/217/205/192, "
                    "driven-ring cells 15.89 to 25.27 km wide, about 3.4:1 at "
                    "the interface.  8 of 8 decks bitwise on THIS cull's zone "
                    "geometry (397 spec cells, 910 relax cells, 1,169 spec "
                    "edges, 2,696 relax edges), 22 of 22 kernels covered, all "
                    "eight mutation controls with teeth.  The zone is SMALLER "
                    "in element count than the 1.35x cull's while the domain "
                    "is larger, because the rings out here are made of much "
                    "wider cells."
                ),
            ),
            RegionalContract(
                bdy_mask_sha256=(
                    "e52578a9bcffc1a019918f6e9fd01f6ac7bf54914dc916857860b539268e35b5"
                ),
                n_cells=40_520,
                boundary_zone_width=7,
                class_id="graded-869m-dt5-z7",
                mesh_row="r0.9.120.40520",
                card="NVIDIA GeForce RTX 5070 Ti (sm_120)",
                admitted_on="2026-09-02",
                contract_receipt=(
                    "evidence/hex-speed-20260902/class-mint-937/contract/records.json"
                ),
                basis=(
                    "The first sub-kilometre geometry earned: the 125 km cap "
                    "at 35.0 N 97.0 W culled from the registered 937.5 m "
                    "parent, 110,533 cells become 40,520, rings "
                    "38,052/399/387/366/351/333/326/306.  8 of 8 decks "
                    "bitwise against the v8.4.1 CPU authority on THIS cull's "
                    "zone geometry (632 spec cells, 1,437 relax cells, 1,861 "
                    "spec edges, 4,250 relax edges) over 140,794,482 float32 "
                    "values compared as raw bit patterns (42 payloads), 22 of "
                    "22 declared kernels covered, dual-run stable across two "
                    "passes in one process, and every deck re-run with a "
                    "deliberately wrong zone geometry FAILS.  468 s on the "
                    "proving node's RTX 5070 Ti."
                ),
            ),
            RegionalContract(
                bdy_mask_sha256=(
                    "de0c858251b36b8592c8416f2dba8876127035229d4f3d88184831732c10b996"
                ),
                n_cells=56_619,
                boundary_zone_width=7,
                class_id="graded-711m-dt5-z7",
                mesh_row="r0.8.102.56619",
                card="NVIDIA GeForce RTX 5070 Ti (sm_120)",
                admitted_on="2026-09-02",
                contract_receipt=(
                    "evidence/hex-speed-20260902/class-mint-800/contract/records.json"
                ),
                basis=(
                    "The finest geometry earned: the same 125 km cap at 35.0 N "
                    "97.0 W culled from the generated 800 m parent, 155,138 "
                    "cells become 56,619, rings 53,412/508/497/477/463/448/422/392.  "
                    "8 of 8 decks bitwise against the v8.4.1 CPU authority on "
                    "THIS cull's zone geometry (814 spec cells, "
                    "1,885 relax cells, 2,397 spec edges, "
                    "5,580 relax edges) over 196,714,772 float32 values "
                    "compared as raw bit patterns (42 payloads), 22 of 22 "
                    "declared kernels covered, dual-run stable across two passes "
                    "in one process, and every deck re-run with a deliberately "
                    "wrong zone geometry FAILS.  552 s on the "
                    "proving node's NVIDIA GeForce RTX 5070 Ti."
                ),
            ),
        )
    }
)


# ---------------------------------------------------------------------------
# the presented-receipt route: a cull the cascade made this cycle
# ---------------------------------------------------------------------------

#: Where a freshly-run contract deck's receipt is looked for.  A cycling
#: cascade re-places its swath every cycle, so the cull it must admit did not
#: exist when this file was written and cannot be a row in it.
#:
#: THE ROUTE IS EVIDENCE, NOT A DECLARATION, and the difference is the whole
#: of it.  A presented receipt admits a geometry only if it says, in its own
#: recorded numbers, everything a shipped row's basis sentence says: it was
#: written by ``tools/run_cuda_regional_contract.py``, it names THIS mesh's
#: ``bdyMask`` digest and cell count, it was produced at the same kernel-set
#: digest as the class admitting it, all 8 decks are bitwise against the CPU
#: authority, all 22 declared kernels are covered, and every mutation control
#: had teeth.  Anything short of that is refused by name.  A receipt cannot
#: admit a configuration a shipped row could not.
CONTRACT_LEDGER_ENVIRONMENT = "WOOF_HEX_REGIONAL_CONTRACT_DIR"

#: What a presented receipt must assert about itself before it admits
#: anything.  Each entry is (json key path, required value, the sentence the
#: refusal uses when it is not there).
_REQUIRED_CONTRACT_CLAIMS: tuple[tuple[str, Any, str], ...] = (
    ("instrument", "run_cuda_regional_contract",
     "it was not written by tools/run_cuda_regional_contract.py"),
    ("all_decks_bitwise", True,
     "it does not report all 8 decks bitwise against the v8.4.1 CPU authority"),
    ("all_kernels_covered", True,
     "it does not report all 22 declared regional kernels covered by a deck"),
    ("all_controls_have_teeth", True,
     "it does not report every mutation control failing on a deliberately "
     "wrong zone geometry, so its decks are not proven able to fail"),
    ("dual_run_identical", True,
     "it does not report its decks reproducing across two runs"),
    ("decks_selected", False,
     "it was run with --decks, so it is a subset of the deck set and not the "
     "contract"),
)


def _ledger_directories(extra: Sequence[Path | str] = ()) -> list[Path]:
    directories: list[Path] = [Path(item) for item in extra]
    raw = os.environ.get(CONTRACT_LEDGER_ENVIRONMENT, "")
    for piece in raw.split(os.pathsep):
        if piece.strip():
            directories.append(Path(piece.strip()))
    return directories


def _read_contract_receipt(path: Path) -> Mapping[str, Any] | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def contract_receipt_defects(
    document: Mapping[str, Any],
    *,
    bdy_mask_sha256: str,
    n_cells: int | None,
    kernel_set: str,
) -> list[str]:
    """Everything wrong with a presented contract receipt, by name."""

    defects: list[str] = []
    for key, required, sentence in _REQUIRED_CONTRACT_CLAIMS:
        if document.get(key) != required:
            defects.append(sentence)
    measured = str(document.get("bdy_mask_sha256") or "")
    if measured != bdy_mask_sha256:
        defects.append(
            f"it was run on bdyMask digest {measured[:16] or '(absent)'}... "
            f"and this mesh's is {bdy_mask_sha256[:16]}..., so it measured a "
            f"different zone geometry"
        )
    if n_cells is not None and document.get("n_cells") is not None:
        if int(document["n_cells"]) != int(n_cells):
            defects.append(
                f"it was run on {int(document['n_cells'])} cells and this "
                f"mesh carries {int(n_cells)}"
            )
    recorded = str(document.get("kernel_set_sha256") or "")
    if recorded != kernel_set:
        defects.append(
            f"it was produced at kernel set {recorded[:16] or '(absent)'}... "
            f"and this tree's regional kernel set digests to "
            f"{kernel_set[:16]}..., so its decks measured different bytes "
            f"from the ones this run would launch"
        )
    return defects


def presented_contract(
    bdy_mask_sha256: str,
    *,
    n_cells: int | None,
    kernel_set: str,
    directories: Sequence[Path | str] = (),
) -> tuple[RegionalContract | None, list[str]]:
    """A contract deck receipt presented for this exact geometry.

    Returns the admitted contract and the reasons every candidate receipt was
    rejected, so a refusal can say what it looked at rather than only that it
    found nothing.
    """

    rejected: list[str] = []
    for directory in _ledger_directories(directories):
        if not directory.is_dir():
            rejected.append(f"{directory} is not a directory")
            continue
        for path in sorted(directory.glob("*.json")):
            document = _read_contract_receipt(path)
            if document is None:
                rejected.append(f"{path.name} is not a readable JSON object")
                continue
            defects = contract_receipt_defects(
                document,
                bdy_mask_sha256=bdy_mask_sha256,
                n_cells=n_cells,
                kernel_set=kernel_set,
            )
            if defects:
                rejected.append(f"{path.name}: {defects[0]}")
                continue
            class_id = str(document.get("class_id") or "")
            return (
                RegionalContract(
                    bdy_mask_sha256=bdy_mask_sha256,
                    n_cells=int(document.get("n_cells") or (n_cells or 0)),
                    boundary_zone_width=int(
                        document.get("boundary_zone_width") or 0
                    ),
                    class_id=class_id,
                    mesh_row=document.get("mesh_row"),
                    card=str(document.get("card") or "unrecorded"),
                    admitted_on=str(document.get("date_utc") or "unrecorded"),
                    contract_receipt=str(path),
                    basis=(
                        "PRESENTED contract receipt, checked by content: 8 of "
                        "8 decks bitwise against the v8.4.1 CPU authority on "
                        "this cull's own zone geometry, 22 of 22 declared "
                        "kernels covered, every mutation control with teeth, "
                        "dual-run identical, at this tree's own regional "
                        "kernel-set digest."
                    ),
                ),
                rejected,
            )
    return None, rejected


# ---------------------------------------------------------------------------
# lookups
# ---------------------------------------------------------------------------
def admitted_summary() -> str:
    """Human-readable roster for refusal messages."""

    if not ADMITTED_CLASSES:
        return "none"
    return ", ".join(sorted(ADMITTED_CLASSES))


def contract_summary() -> str:
    if not SHIPPED_CONTRACTS:
        return "none"
    return ", ".join(
        sorted(
            row.mesh_row or row.bdy_mask_sha256[:16]
            for row in SHIPPED_CONTRACTS.values()
        )
    )


def admitted_class(class_id: str | None) -> RegionalClass | None:
    if not class_id:
        return None
    held = ADMITTED_CLASSES.get(str(class_id))
    if held is not None:
        return held
    from woof.hex.regional_mint import presented_classes
    return next((row for row in presented_classes() if row.class_id == str(class_id)), None)


def admitted_class_for_key(key: RegionalClassKey) -> RegionalClass | None:
    # Exact matches first, then the banded ones, so a class that names a
    # measured edge to the millimetre is never out-ranked by a band around
    # a neighbour.
    from woof.hex.regional_mint import presented_classes
    rows = tuple(ADMITTED_CLASSES.values()) + tuple(presented_classes())
    for row in rows:
        if row.key.matches(key):
            return row
    for row in ADMITTED_CLASSES.values():
        if row.finest_edge_band_relative > 0.0 and row.key.matches(
            key, band=row.finest_edge_band_relative
        ):
            return row
    return None


def shipped_contract(bdy_mask_sha256: str | None) -> RegionalContract | None:
    if not bdy_mask_sha256:
        return None
    return SHIPPED_CONTRACTS.get(str(bdy_mask_sha256))


def contract_for_row(mesh_row: str | None) -> RegionalContract | None:
    if not mesh_row:
        return None
    for row in SHIPPED_CONTRACTS.values():
        if row.mesh_row == str(mesh_row):
            return row
    return None


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
def unanchored_class_refusal(
    key: RegionalClassKey, mesh_row: str | None
) -> str:
    """The named refusal for a configuration class holding no mint."""

    named = "an unregistered regional mesh" if not mesh_row else repr(mesh_row)
    return (
        f"regional CUDA execution is refused for {named}: its configuration "
        f"class ({key.describe()}) holds no forecast mint, so no pair of runs "
        f"has ever shown that THIS kernel set reproduces itself at this "
        f"timestep on a domain of this shape -- the run would produce a "
        f"receipt nobody could verify, which is the one thing this gate "
        f"exists to prevent.  A class is minted ONCE, on any one cull of it, "
        f"by tools/run_cuda_regional_forecast.py --mint-anchor run twice, "
        f"and is then one row in "
        f"cuda_backend/regional_admission.ADMITTED_CLASSES.  Minted classes: "
        f"{admitted_summary()}.  NOTE WHAT IS AND IS NOT IN THE KEY: the "
        f"cull's own bytes are NOT, because five concentric culls of one "
        f"parent were minted independently and no input the mint reads "
        f"distinguished them; the kernel set IS, so an edit to a regional "
        f"translation unit lapses every class until it is re-minted"
    )


def uncontracted_geometry_refusal(
    bdy_mask_sha256: str | None,
    n_cells: int | None,
    class_id: str,
    rejected: Sequence[str] = (),
) -> str:
    """The named refusal for a cull whose own zone geometry is unchecked."""

    digest = (
        "no bdyMask digest was supplied"
        if not bdy_mask_sha256
        else f"bdyMask digest {bdy_mask_sha256[:16]}..."
    )
    cells = "" if n_cells is None else f", {int(n_cells)} cells"
    looked = ""
    if rejected:
        looked = "  Receipts looked at and why each was rejected: " + "; ".join(
            rejected[:6]
        )
    return (
        f"regional CUDA execution is refused for this cull ({digest}{cells}): "
        f"its configuration class {class_id!r} IS minted, but no contract "
        f"deck has been run on THIS cull's own zone geometry.  The concrete "
        f"breakage that prevents: the 22 regional kernels are indexed by "
        f"ring, and a deck run on another cull's rings measured another "
        f"cull's specified and relaxation zones -- a mask-4/5 downgrade, a "
        f"spec-zone pgrad mask or a one-cell ring-7 edge that is wrong HERE "
        f"would be reproduced identically by two runs and caught by neither.  "
        f"Run `python -m woof.hex.drivers.run_cuda_regional_contract` (in a "
        f"checkout, tools/run_cuda_regional_contract.py) against this grid and "
        f"present its receipt through the {CONTRACT_LEDGER_ENVIRONMENT} "
        f"directory (or add a row to SHIPPED_CONTRACTS).  Contracted "
        f"geometries: {contract_summary()}.{looked}"
    )


def class_mismatch_refusal(
    mesh_row: str, expected: RegionalClassKey, measured: RegionalClassKey,
    *, band: float = 0.0,
) -> str:
    """The named refusal when a contracted geometry is run off its class."""

    differing = expected.differing_fields(measured, band=band)
    return (
        f"regional CUDA execution is refused for {mesh_row!r}: its contract "
        f"deck belongs to a class minted at [{expected.describe()}] and this "
        f"run would execute [{measured.describe()}].  The fields that differ "
        f"are {differing}.  The mint measured a different configuration from "
        f"the one this run would execute, so it says nothing about it"
    )


def digest_mismatch_refusal(mesh_row: str, expected: str, measured: str) -> str:
    """The named refusal when a registered row's bdyMask bytes moved."""

    return (
        f"regional CUDA execution is refused for {mesh_row!r}: the contract "
        f"deck registered bdyMask digest {expected[:16]}... but this mesh's "
        f"boundary masks digest {measured[:16]}... -- the zone geometry the "
        f"deck was run on is not the zone geometry this run would execute, so "
        f"its per-kernel receipt says nothing about it"
    )


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RegionalAnchor:
    """What admitted one regional run: a class mint plus a geometry deck.

    Kept under the name the two call sites and every archived receipt already
    use.  ``mesh_row``, ``n_cells``, ``boundary_zone_width``,
    ``bdy_mask_sha256``, ``card``, ``admitted_on``, ``contract_receipt``,
    ``forecast_anchor`` and ``basis`` all still read, so a caller that
    recorded an anchor before the re-keying records the same fields after it.
    """

    mesh_row: str | None
    n_cells: int
    boundary_zone_width: int
    bdy_mask_sha256: str
    card: str
    admitted_on: str
    contract_receipt: str
    forecast_anchor: str
    basis: str
    class_id: str = ""
    class_key: RegionalClassKey | None = None
    contract_route: str = "shipped"

    def as_dict(self) -> dict[str, object]:
        return {
            "mesh_row": self.mesh_row,
            "n_cells": self.n_cells,
            "boundary_zone_width": self.boundary_zone_width,
            "bdy_mask_sha256": self.bdy_mask_sha256,
            "card": self.card,
            "admitted_on": self.admitted_on,
            "contract_receipt": self.contract_receipt,
            "forecast_anchor": self.forecast_anchor,
            "basis": self.basis,
            "class_id": self.class_id,
            "class_key": None if self.class_key is None else self.class_key.as_dict(),
            "contract_route": self.contract_route,
        }


def _compose(
    contract: RegionalContract,
    klass: RegionalClass,
    *,
    n_cells: int | None,
    route: str,
) -> RegionalAnchor:
    return RegionalAnchor(
        mesh_row=contract.mesh_row,
        n_cells=int(contract.n_cells or (n_cells or 0)),
        boundary_zone_width=int(
            contract.boundary_zone_width or klass.key.boundary_zone_width
        ),
        bdy_mask_sha256=contract.bdy_mask_sha256,
        card=contract.card,
        admitted_on=contract.admitted_on,
        contract_receipt=contract.contract_receipt,
        forecast_anchor=klass.mint_receipts[0] if klass.mint_receipts else "",
        basis=(
            f"CLASS {klass.class_id}: {klass.basis}  GEOMETRY "
            f"{contract.bdy_mask_sha256[:16]}...: {contract.basis}"
        ),
        class_id=klass.class_id,
        class_key=klass.key,
        contract_route=route,
    )


def require_regional_anchor(
    mesh_row: str | None,
    *,
    bdy_mask_sha256: str | None,
    n_cells: int | None = None,
    boundary_zone_width: int | None = None,
    n_vert_levels: int | None = None,
    finest_edge_m: float | None = None,
    dt_seconds: float | None = None,
    contract_directories: Sequence[Path | str] = (),
    kernel_set: str | None = None,
) -> RegionalAnchor:
    """Admit one regional configuration for CUDA execution, or refuse by name.

    Two halves, checked in the order that makes the message useful.  First the
    GEOMETRY: has a contract deck ever been run on these rings?  Then the
    CLASS: is the configuration this run would execute one a forecast mint has
    ever been taken of?

    Callers that know the configuration -- the regional forecast opener, which
    holds the mesh and the namelist -- pass ``boundary_zone_width``,
    ``n_vert_levels``, ``finest_edge_m`` and ``dt_seconds``, and the class key
    is MEASURED from this run rather than trusted from a table.  The
    validation site inside the pinned global driver holds only a mesh, so it
    resolves the class through the geometry's own contract row and the class
    key it names; that site is reachable only by handing a culled mesh to the
    global door, which is a configuration nobody runs deliberately.

    Raises :class:`RuntimeError` (``RegionalAdmissionRefusal``) carrying a
    message that names the concrete breakage, per the gate law.  Callers
    convert it to their own refusal type when a typed refusal is the local
    convention.
    """

    digest = None if not bdy_mask_sha256 else str(bdy_mask_sha256)
    resolved_kernel_set = kernel_set or kernel_set_sha256()

    # A named row whose bytes moved is refused before anything else, because
    # the name and the bytes disagreeing is a different defect from either
    # half being missing.
    named = contract_for_row(mesh_row)
    if named is not None and digest and named.bdy_mask_sha256 != digest:
        raise RegionalAdmissionRefusal(
            digest_mismatch_refusal(str(mesh_row), named.bdy_mask_sha256, digest)
        )
    # NAMING A ROW IS A CLAIM, AND THE BYTES ARE THE PROOF: both must hold.
    # A caller that names a row nobody has a deck for is refused even when
    # its bytes happen to match a geometry that IS contracted.  Falling
    # through to the digest would admit a run whose own receipt names an
    # identity no registry holds -- two classifiers disagreeing about which
    # mesh they are looking at, which is a defect shape this program has
    # already lost a day to.
    if mesh_row and named is None and shipped_contract(digest) is not None:
        raise RegionalAdmissionRefusal(
            f"regional CUDA execution is refused for {mesh_row!r}: no "
            f"contract deck is registered under that name, and the bdyMask "
            f"digest supplied belongs to "
            f"{shipped_contract(digest).mesh_row!r}.  The name and the bytes "
            f"disagree about which cull this is, so any receipt this run "
            f"produced would name an identity no registry holds.  Contracted "
            f"geometries: {contract_summary()}"
        )

    contract = shipped_contract(digest) or named
    route = "shipped"
    rejected: list[str] = []
    if contract is None and digest:
        contract, rejected = presented_contract(
            digest,
            n_cells=n_cells,
            kernel_set=resolved_kernel_set,
            directories=contract_directories,
        )
        route = "presented"

    # The class key.  Measured from this run when the caller knows it, taken
    # from the geometry's contract row when it does not.
    measured_key: RegionalClassKey | None = None
    if None not in (boundary_zone_width, n_vert_levels, finest_edge_m, dt_seconds):
        measured_key = RegionalClassKey.build(
            boundary_zone_width=int(boundary_zone_width),  # type: ignore[arg-type]
            n_vert_levels=int(n_vert_levels),  # type: ignore[arg-type]
            finest_edge_m=float(finest_edge_m),  # type: ignore[arg-type]
            dt_seconds=float(dt_seconds),  # type: ignore[arg-type]
            kernel_set=resolved_kernel_set,
        )

    if contract is None:
        from woof.hex.speculative_contract import pending_contract
        contract = pending_contract(digest=digest, cells=n_cells, kernel_set=resolved_kernel_set,
                                    measured_key=measured_key, mesh_row=mesh_row)
        if contract is not None:
            route = "speculative"

    if contract is None:
        class_id = (
            "(unresolved)"
            if measured_key is None
            else (
                admitted_class_for_key(measured_key).class_id
                if admitted_class_for_key(measured_key) is not None
                else "(unminted)"
            )
        )
        if measured_key is not None and class_id == "(unminted)":
            raise RegionalAdmissionRefusal(
                unanchored_class_refusal(measured_key, mesh_row)
            )
        raise RegionalAdmissionRefusal(
            uncontracted_geometry_refusal(digest, n_cells, class_id, rejected)
        )

    klass = admitted_class(contract.class_id)
    if klass is None:
        raise RegionalAdmissionRefusal(
            f"regional CUDA execution is refused for this cull (bdyMask "
            f"digest {contract.bdy_mask_sha256[:16]}...): its contract deck "
            f"names configuration class {contract.class_id!r}, which holds no "
            f"forecast mint.  Minted classes: {admitted_summary()}"
        )

    if measured_key is not None and not klass.key.matches(
        measured_key, band=klass.finest_edge_band_relative
    ):
        raise RegionalAdmissionRefusal(
            class_mismatch_refusal(
                str(mesh_row or contract.mesh_row or "an unregistered cull"),
                klass.key,
                measured_key,
                band=klass.finest_edge_band_relative,
            )
        )
    if route == "presented" and not klass.key.finest_edge_measured:
        raise RegionalAdmissionRefusal(
            f"regional CUDA execution is refused for this cull (bdyMask "
            f"digest {contract.bdy_mask_sha256[:16]}...): its contract deck "
            f"was presented at run time and it names class "
            f"{klass.class_id!r}, whose own finest edge was never measured -- "
            f"so the class key cannot tell this cull's resolution from any "
            f"other at the same zone width, column count and timestep, and "
            f"the mint would be inherited by a domain nobody compared it to.  "
            f"A class with an unmeasured finest edge admits only the "
            f"geometries written into SHIPPED_CONTRACTS"
        )
    if measured_key is None and klass.key.kernel_set_sha256 != resolved_kernel_set:
        raise RegionalAdmissionRefusal(
            f"regional CUDA execution is refused for "
            f"{mesh_row or contract.mesh_row!r}: class {klass.class_id!r} was "
            f"minted at kernel set {klass.key.kernel_set_sha256[:16]}... and "
            f"this tree's regional kernel set digests to "
            f"{resolved_kernel_set[:16]}...  The mint is a statement about "
            f"bytes and those bytes have moved, so it no longer describes "
            f"what this run would launch; re-mint the class"
        )

    if n_cells is not None and contract.n_cells and int(n_cells) != int(contract.n_cells):
        raise RegionalAdmissionRefusal(
            f"regional CUDA execution is refused for "
            f"{mesh_row or contract.mesh_row!r}: the contract deck was run on "
            f"{contract.n_cells} cells and this mesh carries {int(n_cells)}.  "
            f"The bdyMask digest matched, so this is a mesh whose rings are "
            f"those of a different domain -- exactly the disagreement the "
            f"digest cross-check exists to catch.  For the GEOMETRY half of "
            f"an anchor a different cull is a different configuration, and "
            f"that did not change when the forecast mint moved to a class: "
            f"the 22 regional kernels are indexed by ring and a deck run on "
            f"other rings measured another domain's zones"
        )

    return _compose(contract, klass, n_cells=n_cells, route=route)


# ---------------------------------------------------------------------------
# retired surface, kept resolvable
# ---------------------------------------------------------------------------
#: RETIRED 2026-08-27 by the class re-keying, and kept only so a reader who
#: meets the name in an archived receipt can resolve it.  It was a mapping
#: from mesh-row name to a single row carrying BOTH halves of an anchor, so a
#: re-placed cull of an already-minted class could not be admitted without
#: paying for a second forecast mint.  The two halves now live in
#: ``ADMITTED_CLASSES`` and ``SHIPPED_CONTRACTS``; this view composes them
#: back so an old caller reads the same fields.
ADMITTED_REGIONS: Mapping[str, RegionalAnchor] = MappingProxyType(
    {
        contract.mesh_row: _compose(
            contract,
            ADMITTED_CLASSES[contract.class_id],
            n_cells=contract.n_cells,
            route="shipped",
        )
        for contract in SHIPPED_CONTRACTS.values()
        if contract.mesh_row and contract.class_id in ADMITTED_CLASSES
    }
)


def admitted_region(mesh_row: str) -> RegionalAnchor | None:
    """The anchor admitting ``mesh_row``, or ``None``."""

    return ADMITTED_REGIONS.get(str(mesh_row))


def admitted_region_by_digest(bdy_mask_sha256: str | None) -> RegionalAnchor | None:
    """The anchor whose registered boundary masks are exactly these bytes."""

    contract = shipped_contract(bdy_mask_sha256)
    if contract is None or contract.class_id not in ADMITTED_CLASSES:
        return None
    return _compose(
        contract,
        ADMITTED_CLASSES[contract.class_id],
        n_cells=contract.n_cells,
        route="shipped",
    )


def unanchored_refusal(mesh_row: str | None, bdy_mask_sha256: str | None) -> str:
    """The named refusal for a regional configuration holding no anchor."""

    return uncontracted_geometry_refusal(
        bdy_mask_sha256, None, "(unresolved)", ()
    )


__all__ = [
    "ADMITTED_CLASSES",
    "ADMITTED_REGIONS",
    "CONTRACT_LEDGER_ENVIRONMENT",
    "MINTED_KERNEL_SET_SHA256",
    "REGIONAL_KERNEL_SOURCES",
    "RegionalAdmissionRefusal",
    "RegionalAnchor",
    "RegionalClass",
    "RegionalClassKey",
    "RegionalContract",
    "SHIPPED_CONTRACTS",
    "UNMEASURED_FINEST_EDGE_MM",
    "admitted_class",
    "admitted_class_for_key",
    "admitted_region",
    "admitted_region_by_digest",
    "admitted_summary",
    "class_mismatch_refusal",
    "contract_for_row",
    "contract_receipt_defects",
    "contract_summary",
    "digest_mismatch_refusal",
    "kernel_set_sha256",
    "presented_contract",
    "require_regional_anchor",
    "shipped_contract",
    "unanchored_class_refusal",
    "unanchored_refusal",
    "uncontracted_geometry_refusal",
]
