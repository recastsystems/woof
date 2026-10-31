"""The ONE device-memory admission surface for the v8.4.1 CUDA lane.

Every "will this mesh fit this card?" decision in the project answers from
this module: the forecast door's ``--preflight`` verdict, the door's real-run
admission, and the driver's own floor (``MIN_FREE_DEVICE_BYTES``, rebound per
mesh by ``src/hexcore/drivers/mpas_mesh_binding.py``) all call :func:`required_free_bytes`
with the same model, so they agree to the byte -- and therefore to the cell.
Before this module existed the door predicted from a measured affine row
while the driver held a separate linear floor scaled from an asserted 24 GiB
constant; a card between the two answers passed one gate and died on the
other, which is the concrete breakage a single surface prevents.

THE MODEL HAS A SHAPE NOW, AND THE SHAPE IS NOT A LINE (2026-08-27).
--------------------------------------------------------------------

Every row this module has ever shipped was ``fixed + slope * cells``.  That
form was refuted by measurement, not by argument:

* Across five full-physics forecast-door runs on placed variable-resolution
  meshes the affine row's error spanned **+3.89 % to -6.76 %**, and one mesh
  with 15,343 MORE cells peaked 318 MiB LOWER.  A function linear in cell
  count cannot do that.
* One of those runs (``v6.75.112676``) peaked **96 MiB past its own
  ``required_free``** -- the flat 512 MiB headroom did not cover it.
* On the limited-area path the affine row over-predicted every one of five
  concentric culls by 1.2-1.9 GiB: 4,440 -> 15,755 cells (3.5x) moved the
  measured peak only 5,552 -> 7,728 MiB (1.39x).

The per-allocation ledger (``tools/device_memory_ledger/``) says why,
and the arithmetic below reproduces its measured block sizes exactly.  The
process footprint is

    peak(cells, card) = core(card, configuration)
                      + SUM over tiled workspaces of ws(min(cells, tile(card)))
                      + bytes_per_cell * cells

THE TILED WORKSPACES ARE THE MISSING TERM.  Three physics kernels keep their
column arrays in a GLOBAL device workspace sized to the threads the CARD can
hold in flight, capped by the mesh -- ``min(cells, tile)``.  Below the tile
the term is proportional to cells; above it the term is CONSTANT.  That knee
is the shape a line cannot have, and the tiles are large enough to matter:

    Grell-Freitas  gf.py::gf_tile_columns    tile = SMs x 4 x 64
    YSU            ysu.py::ysu_tile_columns  tile = SMs x 16 x 32
    RRTMG SW/LW    rrtmg_lw.py::batch_column_chunk
                                             chunk = quantised(resident
                                             threads / g-points), ceiling

On the 170 SM part the knees are 43,520 cells (GF) and 87,040 cells (YSU).
Both published fit meshes straddle them: ``x1.40962`` sits BELOW both and
``x4.163842`` ABOVE both, so the affine fit charged the tile growth between
them as if it were per-cell.  That is where 2,167 B of the retired row's
98,748 B/cell slope came from, and it is why the retired row over-predicts
every small mesh and every limited-area cull.

MEASURED PROOF THE TILE ARITHMETIC IS THE RIGHT ARITHMETIC.  The per-allocation ledger
records each site's largest single allocation at both meshes.  Every site in
the model scales exactly 4.00x with the 4x cell count EXCEPT three, and the
formulas above reproduce all three to the byte:

    site                          x1 MiB     x4 MiB   ratio   this module
    woof/core/gf.py (workspace)  1,262.0    1,338.8  1.06    641/680 blocks
    woof/core/ysu.py (workspace)   157.6      334.7  2.12    1,281/2,720
    woof/core/rrtmg_sw.py (wk_d) 1,745.6    1,745.6  1.00    2,048 columns

THE PEAK CHANGES SCHEME WITH MESH SIZE, which is the same finding from the
other side.  At 40,962 cells the largest thing live at the global peak is the
RRTMG shortwave ``spcvmc`` workspace -- one mesh-independent 1,830,420,480 B
block.  At 151,649 and 163,842 cells the peak instant is in the dycore and
that block is not live at it at all.  A single line through one radiation-set
point and one dycore-set point describes neither regime.

WHAT THE MARGIN ABSORBS, BY NAME (the gate law, 2026-08-16).
---------------------------------------------------------------

The retired 512 MiB headroom named nothing and failed by 96 MiB.  This model
carries a margin with two named, measured components and no padding:

1. **Arena placement of the largest mesh-independent block.**  There is no
   arena in this project: allocation goes to CuPy's default ``MemoryPool``,
   whose high-water is the set of blocks it has had to ``cudaMalloc``, not
   the peak live set.  Whether the 1,745.6 MiB shortwave workspace is
   servable from the pool's free list at the instant radiation runs depends
   on the arena layout, and the layout depends on the exact byte sizes of
   every per-cell array allocated before it.  This is MEASURED, twice, on an
   idle card: freeing 38.4 MiB EARLIER in the run moved ``x1.40962``'s pool
   high-water from 5,404.5 to 7,111.7 MiB -- **a 1,707.2 MiB rise from a
   38 MiB saving**, because that block stopped being servable from the free
   list (``STATE.md``, the construction-diagnostic hold).  The margin is that
   block, priced at the card's own chunk width: 1,745.6 MiB on a 170 SM
   part, 872.8 MiB on a 68 or 70 SM part.  The concrete breakage: a card
   admitted without it dies inside the CuPy allocation at the first
   radiation call, minutes into a run.
2. **Instrument convention.**  The fitted rows are per-process
   ``nvidia-smi``; the door faces whole-device free memory.  Measured side by
   side, the two conventions differ by 11.2 MiB (x4 arm) and 0.75 MiB
   (graded arm).  The margin carries the larger.

Nothing else is in the margin.  If a term is not measured it is not here.

PER-CARD ROWS, AND WHY ONE SHARED ROW WAS A DEFECT.
-----------------------------------------

The retired module carried :data:`CARD_TIER_ROWS` and **nothing ever read
it**: ``forecast_door._resolve_model`` returned the 170 SM row unless the
user typed ``--device-fixed-mib`` and ``--device-bytes-per-cell`` by hand.
So the 10 GiB desktop was priced with a 32 GiB card's fixed term and refused
``x1.40962`` and ``v15.150.38857`` -- two meshes it had been MEASURED running
with 2,244 MiB to spare.  That is the borrowed-row defect.  The door now reads the card's
multiprocessor count at decision time and selects or derives its row, so a
card that was never measured still gets its own arithmetic.

A DERIVED row is not a measured one and says so.  It scales the reference
card's core by the two mechanisms that are known and measured to scale with
the card -- the CUDA local-memory backing store (per-context, sized to the
widest launched frame times resident threads) and the radiation chunk width
-- and it lands CONSERVATIVE on both cards that have their own measurement:
+182.7 MiB on the 70 SM part and +490.8 MiB on the 68 SM part, both
over-predictions.  Direction matters more than magnitude for a gate.

CONFIGURATIONS ARE DATA, NOT CODE PATHS (the arbitrary acceptance test).
------------------------------------------------------------------------

The global and limited-area paths build different device stacks -- the
regional one carries a padded atmosphere, two levels of lateral-boundary
state and 22 more kernels -- so they carry different cores.  They are two
rows of one table, not two functions.  Adding a card or a configuration is a
row; it is never a new branch.

STATED LIMITS, ON PAGE ONE OF THE MODULE.
-----------------------------------------

* The limited-area core is an ENVELOPE, not a fit: the per-allocation
  instrument has never been run on a regional mesh, so the core is the
  largest residue over five whole-device forecast-door samples.  One per-allocation ledger
  arm on a cull replaces it.
* The 16 GiB and 10 GiB cores were measured at hex ``5252421``, not at the
  merged tip, and carry :data:`PIN_RESTATEMENT_BYTES` for it.
* Every measurement in this module is 55 vertical levels, float32, WSM6 + GF
  + YSU + YSU-GWDO + revised-MO + NoahMP + cloud fraction + legacy RRTMG.
  The tiled-workspace terms are functions of ``nz``; nothing else here is.
* The four-swath spread that opened this lane is EXPLAINED IN MECHANISM but
  NOT SEPARATED IN MEASUREMENT: the four runs share card, pin, timestep,
  schedule and radiation cadence, so arena placement is the only candidate
  left standing, and no A/B on those four meshes has been run to prove it.

This module is stdlib-only at import, deliberately: the forecast door
imports it before any refusal, on boxes with no CUDA lane at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

__all__ = [
    "CARD_TIER_ROWS",
    "ROLLBACK_ARMED_BYTES_PER_CELL",
    "STACK_RESTATEMENT_BYTES_PER_CELL",
    "CONVENTION_MARGIN_BYTES",
    "CardProfile",
    "DEFAULT_HEADROOM_BYTES",
    "FLOOR_DERIVATION",
    "FOOTPRINT_MODEL",
    "FootprintModel",
    "GRELL_FREITAS_WORKSPACE",
    "KNOWN_CARDS",
    "MIB",
    "MODEL_VERTICAL_LEVELS",
    "NATIVE_MESH_CELLS",
    "PIN_RESTATEMENT_BYTES",
    "REFERENCE_CARD",
    "RETIRED_AFFINE_ROW_20260826",
    "RETIRED_CARD_TIER_ROWS_20260826",
    "RETIRED_FLAT_HEADROOM_BYTES",
    "RETIRED_LINEAR_FLOOR_BYTES",
    "RETIRED_ROW_20260825",
    "SHAPED_ROWS",
    "ShapedFootprintModel",
    "TiledWorkspace",
    "YSU_WORKSPACE",
    "card_profile_from_attributes",
    "model_for_card",
    "native_device_floor_bytes",
    "radiation_chunk_columns",
    "required_free_bytes",
    "MPAS_ROUTE_RUN_PEAKS",
    "PHYSICS_ROUTES",
    "ROUTE_RUN_MARGIN",
    "RouteRunPeak",
    "route_required_free_bytes",
    "route_run_curve_mib",
    "harmonic_basis_bytes",
    "retired_affine_row_floor_bytes",
    "retired_converged_row_floor_bytes",
    "retired_linear_floor_bytes",
    "row_key",
    "shortwave_workspace_bytes",
    "tier_model",
]

MIB = 1024**2


@dataclass(frozen=True)
class CoTenancyRow:
    """Conservative resident-member limit, independent of forecast case names."""

    mesh_kind: str
    maximum_cells: int
    maximum_levels: int
    minimum_card_bytes: int
    maximum_members: int


# These are scheduling limits, not measured footprint fits. Admission still
# charges the full existing solo floor for EVERY member, with no sharing credit.
CO_TENANCY_ROWS = (
    CoTenancyRow("point-cull", 50_000, 55, 0, 1),
    CoTenancyRow("point-cull", 50_000, 55, 16 * 1024**3, 2),
    CoTenancyRow("point-cull", 50_000, 55, 32 * 1024**3, 3),
    CoTenancyRow("point-cull", 50_000, 55, 64 * 1024**3, 4),
    # Global quasi-uniform meshes up to the 30 km x1.655362 (B5, the resident global ensemble).  The card
    # thresholds sit under the nameplate sizes (a 32 GB RTX 5090 reports 31.8 GiB) so each card class
    # gets its row; the floor sum below is what keeps a group inside the card's memory.
    CoTenancyRow("global", 655_362, 55, 0, 1),
    CoTenancyRow("global", 655_362, 55, 15 * 1024**3, 2),
    CoTenancyRow("global", 655_362, 55, 30 * 1024**3, 3),
    CoTenancyRow("global", 655_362, 55, 60 * 1024**3, 4),
)


def co_tenancy_limits(*, mesh_kind: str, cells: int, levels: int, card_bytes: int, free_bytes: int,
                      solo_required_bytes: int, member_additional_bytes: Sequence[int] = ()) -> tuple[int, int]:
    """``(table limit, memory limit)`` for resident members of one mesh on one card.

    The ONE pricing both the ensemble planner (``global_ensemble.group_limit``) and the runtime
    (:func:`admit_co_tenancy`) decide on.  Each member is charged its full solo floor plus its own
    additional allocations (the stochastic arrays); the largest such member sets the memory limit.
    THE BREAKAGE THIS PREVENTS (RTX 5090, 60 km x1.163842 GEFS members 1-2, 2026-10-20): the planner
    read only the table (3 per 32 GB card) while the runtime charged 14.68 GB solo + 4.97 GB stochastic per
    member, so every first group was refused and split; the refusal also printed "memory limit 2" because it
    divided by the solo floor alone, when only 1 member fits.
    """
    rows = [row for row in CO_TENANCY_ROWS if row.mesh_kind == mesh_kind
            and cells <= row.maximum_cells and levels <= row.maximum_levels
            and card_bytes >= row.minimum_card_bytes]
    table = max((row.maximum_members for row in rows), default=1)
    per_member = solo_required_bytes + max(member_additional_bytes, default=0)
    return table, free_bytes // per_member


def admit_co_tenancy(*, mesh_kind: str, cells: int, levels: int,
                     members: int, card_bytes: int, free_bytes: int,
                     solo_required_bytes: int,
                     member_additional_bytes: tuple[int, ...] = ()) -> dict[str, Any]:
    """Refuse before construction rather than OOM at a member's radiation step."""
    values = (cells, levels, members, card_bytes, free_bytes, solo_required_bytes)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("co-tenancy needs positive integer dimensions and memory sizes")
    if member_additional_bytes and (len(member_additional_bytes) != members or
            any(type(value) is not int or value < 0 for value in member_additional_bytes)):
        raise ValueError("additional allocations need one nonnegative byte count per member")
    limit, memory_limit = co_tenancy_limits(
        mesh_kind=mesh_kind, cells=cells, levels=levels, card_bytes=card_bytes, free_bytes=free_bytes,
        solo_required_bytes=solo_required_bytes, member_additional_bytes=member_additional_bytes)
    additional = sum(member_additional_bytes)
    required = members * solo_required_bytes + additional
    if members > limit or required > free_bytes:
        raise MemoryError(
            f"co-tenancy refuses {members} members: table limit {limit}, "
            f"memory limit {memory_limit} at {solo_required_bytes} solo + "
            f"{max(member_additional_bytes, default=0)} additional bytes per member, "
            f"required {required} bytes, available {free_bytes} bytes, "
            f"shortfall {max(0, required - free_bytes)} bytes; prevents "
            "CUDA out-of-memory during resident-member construction or the "
            "first radiation workspace allocation; reduce members per card")
    return {"mesh_kind": mesh_kind, "cells": cells, "levels": levels,
            "members": members, "maximum_members": min(limit, memory_limit),
            "table_maximum_members": limit,
            "memory_limited_members": memory_limit,
            "solo_required_bytes": solo_required_bytes,
            "required_free_bytes": required, "measured_free_bytes": free_bytes,
            "measured_total_bytes": card_bytes,
            "member_additional_bytes": list(member_additional_bytes),
            "basis": "sum of full solo admission floors; no unmeasured sharing credit"}


def harmonic_basis_bytes(cells: int, degree: int) -> int:
    """Exact float32 basis allocation for a closed spherical Voronoi mesh.

    Euler's formula gives nVertices = 2*nCells - 4 on the trivalent dual.
    MeshHarmonics stores all cell and vertex harmonics, excluding degree zero.
    This prices an allocation, not a measured stochastic peak-memory fit.
    """
    if type(cells) is not int or cells < 4:
        raise ValueError("a closed spherical Voronoi mesh needs at least four cells")
    if type(degree) is not int or not 1 <= degree <= 512:
        raise ValueError("harmonic degree must be 1..512")
    return (3*cells - 4) * ((degree + 1)**2 - 1) * 4


def stochastic_array_ledger(cells: int, levels: int, degree: int, *,
                            extra_scales: int = 0, skeb: bool = False) -> dict[str, int]:
    """Logical array floor for one stochastic member on a closed global mesh.

    Counts the basis, retained mesh/member arrays, and named forcing arrays
    live at the tendency-write kernels. Each member pays its own mesh
    resource without a sharing credit. These are allocation shapes, not a
    measured peak: construction scratch, intermediate expression outputs,
    reduction/cuBLAS workspaces and allocator placement remain unpriced.
    No mesh data or GPU context is needed to evaluate this stdlib-only sum.
    """
    basis = harmonic_basis_bytes(cells, degree)
    if type(levels) is not int or levels < 1:
        raise ValueError("stochastic levels must be a positive integer")
    if type(extra_scales) is not int or not 0 <= extra_scales <= 5:
        raise ValueError("stochastic extra scales must be 0..5")
    if type(skeb) is not bool:
        raise ValueError("stochastic skeb must be logical")
    vertices, edges = 2*cells-4, 3*cells-6
    points, modes, processes = cells+vertices, (degree+1)**2-1, 1+extra_scales
    arrays = dict(
        harmonic_basis=basis,
        mesh_mode_degrees=8*modes,
        mesh_topology=16*edges+12*vertices,
        mesh_geometry=4*cells+12*edges,
        member_tapers=4*levels*points,
        member_spectral_coefficients=12*modes*processes,
        member_spectral_weights=4*modes*processes,
        member_pattern_outputs=12*points*processes,
        forcing_sppt=4*points,
    )
    if skeb:
        arrays.update(forcing_sink=4*levels*edges,
                      forcing_dissipation=4*levels,
                      forcing_streamfunction=4*vertices,
                      forcing_curl=4*edges,
                      forcing_norm=4,
                      forcing_temperature=4*cells)
    return arrays

#: The vertical level count every measurement in this module was taken at.
#: The tiled-workspace terms are functions of it; the per-cell slope and the
#: cores are NOT re-derivable at another ``nz`` from anything measured here.
MODEL_VERTICAL_LEVELS = 55

#: The native x4.163842 cell count the frozen proof runs at.  Restated here
#: (rather than imported from the registry) because this module must import
#: with no numpy on the box; the registry test asserts the two agree.
NATIVE_MESH_CELLS = 163_842


# ---------------------------------------------------------------------------
# the card
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CardProfile:
    """The card properties every card-scaled term in the model reads.

    Two numbers, both published by the CUDA driver and both read at decision
    time by the door: the multiprocessor count and the maximum resident
    threads per multiprocessor.  Nothing here is a nameplate memory size --
    the door measures free memory separately and never carries it between
    decisions.
    """

    name: str
    multiprocessors: int
    max_threads_per_sm: int = 1536

    @property
    def resident_threads(self) -> int:
        """Threads the card can hold resident: what CUDA prices a per-thread
        local frame at, and what the radiation chunk width is derived from."""

        return int(self.multiprocessors) * int(self.max_threads_per_sm)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "multiprocessors": int(self.multiprocessors),
            "max_threads_per_sm": int(self.max_threads_per_sm),
            "resident_threads": self.resident_threads,
        }


def card_profile_from_attributes(
    name: str, attributes: Mapping[str, Any]
) -> CardProfile:
    """A :class:`CardProfile` from a CuPy ``Device.attributes`` mapping.

    The door calls this with what the driver reports right now.  A card this
    project has never seen gets a real profile rather than a default, which
    is the whole point of the borrowed-row defect's fix.
    """

    sms = attributes.get("MultiProcessorCount")
    threads = attributes.get("MaxThreadsPerMultiProcessor")
    if not sms or int(sms) <= 0:
        raise ValueError(
            "MultiProcessorCount missing or non-positive; the device-memory "
            "model prices the CUDA local-memory backing store and the "
            "radiation chunk width from it, so a card with no SM count "
            "cannot be given a footprint row"
        )
    if not threads or int(threads) <= 0:
        threads = 1536
    return CardProfile(
        name=str(name), multiprocessors=int(sms), max_threads_per_sm=int(threads)
    )


#: The cards this project has run on.  A row here is a convenience for
#: reports and tests; the door never looks a card up by name, it reads the
#: driver.
KNOWN_CARDS: Mapping[str, CardProfile] = MappingProxyType(
    {
        "32gib-170sm": CardProfile("RTX 5090 (32,607 MiB, sm_120)", 170, 1536),
        "24gib-128sm": CardProfile("RTX 4090 (24,564 MiB, sm_89)", 128, 1536),
        "16gib-70sm": CardProfile("RTX 5070 Ti (16,303 MiB, sm_120)", 70, 1536),
        "10gib-68sm": CardProfile("RTX 3080 (10,240 MiB, sm_86, WDDM)", 68, 1536),
        "96gib-188sm": CardProfile(
            "RTX PRO 6000 Blackwell Server (97,887 MiB, sm_120)", 188, 1536
        ),
    }
)

#: The card every measured row of record was taken on, and the card a
#: derived row is derived FROM.
REFERENCE_CARD = KNOWN_CARDS["32gib-170sm"]


@dataclass(frozen=True)
class CardCountRow:
    """Scheduling policy, separate from the measured memory admission sum."""

    key: str
    min_cells: int
    max_cells: int
    min_card_bytes: int
    max_card_bytes: int
    counts: tuple[int, ...]


# Descending counts use the visible rental allocation. A small mesh keeps
# the existing single-card loop. Exact local halo sizes still pass the
# shaped footprint admission before any device stack is constructed.
CARD_COUNT_ROWS = (
    CardCountRow("small-mesh", 1, 65_535, 0, 2**63 - 1, (1,)),
    CardCountRow("medium-small-memory", 65_536, 131_071, 0, 24 * 1024**3 - 1, (8, 4, 2, 1)),
    CardCountRow("medium-capacity-memory", 65_536, 131_071, 24 * 1024**3, 2**63 - 1, (1,)),
    CardCountRow("large-small-memory", 131_072, 2**63 - 1, 0, 24 * 1024**3 - 1, (8, 4, 2, 1)),
    CardCountRow("large-medium-memory", 131_072, 2**63 - 1, 24 * 1024**3, 48 * 1024**3 - 1, (8, 4, 2, 1)),
    CardCountRow("large-capacity-memory", 131_072, 2**63 - 1, 48 * 1024**3, 2**63 - 1, (8, 4, 2, 1)),
)


def select_card_count(cells: int, card_memory_bytes: int, visible_cards: int) -> dict[str, Any]:
    """Choose a table count without CUDA or a case-specific branch."""
    if cells < 1 or card_memory_bytes < 0 or visible_cards < 1:
        raise ValueError(
            f"cannot schedule {cells} cells on {visible_cards} visible cards "
            f"with {card_memory_bytes} memory bytes; a nonempty mesh and card allocation are required"
        )
    rows = [row for row in CARD_COUNT_ROWS
            if row.min_cells <= cells <= row.max_cells
            and row.min_card_bytes <= card_memory_bytes <= row.max_card_bytes]
    if len(rows) != 1:
        raise ValueError(f"card-count table has {len(rows)} rows for {cells} cells and {card_memory_bytes} card bytes")
    row = rows[0]
    count = next(count for count in row.counts if count <= visible_cards)
    return {"row": row.key, "cells": int(cells), "card_memory_bytes": int(card_memory_bytes),
            "visible_cards": int(visible_cards), "selected_cards": count,
            "basis": "scheduling row; exact local footprint admission follows"}


def single_card_rows(cells: int) -> bool:
    """Whether every memory tier for this mesh selects the existing loop."""
    rows = [row for row in CARD_COUNT_ROWS if row.min_cells <= cells <= row.max_cells]
    return bool(rows) and all(row.counts == (1,) for row in rows)


# ---------------------------------------------------------------------------
# the tiled workspaces -- the term the affine row did not have
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TiledWorkspace:
    """A physics workspace sized to ``min(cells, tile)`` where the tile is a
    property of the CARD.

    Grell-Freitas and YSU both moved their column arrays out of the CUDA
    per-thread local frame and into a global workspace sized to the threads
    in flight (the local-frame cut).  A local frame is priced by CUDA at the
    card's WHOLE resident-thread capacity; a workspace is priced at the tile.
    That is why these terms exist at all, and why they are card-scaled.

    ``slots`` and ``block`` are the engine's own constants, restated here
    because this module imports with no engine on the box.
    ``tests/test_device_admission.py`` asserts they still agree with
    ``woof.core`` whenever the engine checkout is importable.
    """

    name: str
    slots: int
    block: int
    blocks_per_sm: int
    level_offset: int
    level_floor: int
    dtype_bytes: int
    site: str
    provenance: str

    def tile_columns(self, cells: int, card: CardProfile) -> int:
        """Columns kept in flight: the card's capacity, capped by the mesh."""

        tile = int(self.blocks_per_sm) * int(self.block) * int(card.multiprocessors)
        return max(int(self.block), min(int(cells), tile))

    def bytes_for(
        self, cells: int, card: CardProfile, levels: int = MODEL_VERTICAL_LEVELS
    ) -> int:
        """The workspace this card allocates for this mesh, in bytes.

        Rounded up to whole blocks, because both kernels interleave the
        workspace by lane within a block: the unit of allocation is one
        block's region, not one column's.
        """

        columns = self.tile_columns(cells, card)
        blocks = -(-columns // int(self.block))
        extent = max(int(self.level_floor), int(levels)) + int(self.level_offset)
        return blocks * int(self.slots) * extent * int(self.block) * int(self.dtype_bytes)

    def knee_cells(self, card: CardProfile) -> int:
        """The cell count above which this term stops growing."""

        return int(self.blocks_per_sm) * int(self.block) * int(card.multiprocessors)


#: Grell-Freitas.  ``GFWS_SLOT_COUNT_COL + GFWS_SLOT_COUNT_DRV = 90 + 36``,
#: ``GF_BLOCK = 64``, ``GF_TILE_BLOCKS_PER_SM = 4``, extent
#: ``gf_kernel_capacity(nz) + 9`` with ``gf_kernel_capacity`` floored at the
#: compiled ``_GF_KMAX_DEFAULT = 40``.  Reproduces the ledger's measured
#: 1,262.0 MiB (x1, 641 blocks) and 1,338.8 MiB (x4, 680 blocks) exactly.
GRELL_FREITAS_WORKSPACE = TiledWorkspace(
    name="grell_freitas_column_workspace",
    slots=90 + 36,
    block=64,
    blocks_per_sm=4,
    level_offset=9,
    level_floor=40,
    dtype_bytes=4,
    site="woof/core/gf.py::gf_workspace_floats via gf_tile_columns",
    provenance=(
        "engine constants GFWS_SLOT_COUNT_COL=90, GFWS_SLOT_COUNT_DRV=36, "
        "GF_BLOCK=64, GF_TILE_BLOCKS_PER_SM=4, _GF_KMAX_DEFAULT=40.  "
        "Verified against the merged-tip per-allocation ledger: 1,262.0 MiB at 40,962 "
        "cells and 1,338.8 MiB at 163,842 on the 170 SM card, ratio 1.06, "
        "which is 680/641 blocks and not 4.00"
    ),
)

#: YSU.  ``YSUWS_SLOTS = 18``, ``YSU_BLOCK = 32``,
#: ``YSU_TILE_BLOCKS_PER_SM = 16``, extent ``nz + 1``.  Reproduces the
#: ledger's 157.6 MiB (x1, 1,281 blocks) and 334.7 MiB (x4, 2,720 blocks).
YSU_WORKSPACE = TiledWorkspace(
    name="ysu_column_workspace",
    slots=18,
    block=32,
    blocks_per_sm=16,
    level_offset=1,
    level_floor=0,
    dtype_bytes=4,
    site="woof/core/physics_inventory.py::ysu_workspace_floats via ysu_tile_columns",
    provenance=(
        "engine constants YSUWS_SLOTS=18, YSU_BLOCK=32, "
        "YSU_TILE_BLOCKS_PER_SM=16.  Verified against the per-allocation ledger merged-tip "
        "ledger: 157.6 MiB at 40,962 cells and 334.7 MiB at 163,842 on the "
        "170 SM card, ratio 2.12, which is 2,720/1,281 blocks and not 4.00"
    ),
)

#: The workspaces the model charges, in order.
TILED_WORKSPACES: tuple[TiledWorkspace, ...] = (
    GRELL_FREITAS_WORKSPACE,
    YSU_WORKSPACE,
)


# ---------------------------------------------------------------------------
# the radiation chunk -- the card-scaled step function
# ---------------------------------------------------------------------------
#: ``woof.core.rrtmg_lw.BATCH_CHUNK_QUANTUM``.
RADIATION_CHUNK_QUANTUM = 256
#: ``woof.core.rrtmg_sw.SW_BATCH_COLUMN_CHUNK_CEILING`` and ``NGPTSW``.
SHORTWAVE_CHUNK_CEILING = 2048
SHORTWAVE_GPOINTS = 112
#: ``SPCVMC_WK_ARRAYS`` -- the float32 arrays the spcvmc workspace holds per
#: thread, one thread per (column, g-point) pair.
SPCVMC_WK_ARRAYS = 35


def radiation_chunk_columns(
    card: CardProfile,
    threads_per_column: int = SHORTWAVE_GPOINTS,
    ceiling: int = SHORTWAVE_CHUNK_CEILING,
    quantum: int = RADIATION_CHUNK_QUANTUM,
) -> int:
    """Columns per radiation chunk on this card -- ``batch_column_chunk``.

    The engine's own narrowing (the device-sized radiation chunk): the smallest multiple of ``quantum``
    whose launch covers the card's resident threads, clamped to the ceiling.
    Restated here rather than imported because this module must price a
    footprint on a box with no CUDA lane; the test asserts the two agree.

    Measured consequence, which is why it is in the model: on the 70 SM part
    the shortwave chunk halves from 2,048 to 1,024 columns and the step-6
    pool event shrinks from +1,745.7 MiB to +872.8 MiB.
    """

    resident = int(card.resident_threads)
    if resident <= 0:
        return int(ceiling)
    need = -(-resident // int(threads_per_column))
    need = -(-need // int(quantum)) * int(quantum)
    return max(int(quantum), min(int(ceiling), need))


def shortwave_workspace_bytes(
    card: CardProfile, levels: int = MODEL_VERTICAL_LEVELS
) -> int:
    """The RRTMG shortwave ``spcvmc`` workspace: ONE contiguous block.

    ``cp.zeros((nc * NGPTSW, SPCVMC_WK_ARRAYS * nl1), float32)`` at
    ``woof/core/rrtmg_sw.py``, where ``nc`` is the chunk width and
    ``nl1 = nlayers + 1 = levels + 2``.  At 2,048 columns and 55 levels that
    is exactly 1,830,420,480 B = 1,745.6 MiB, which is the block the per-allocation ledger
    ledger records as the largest single allocation live at the 40,962-cell
    global peak -- **identical at 163,842 cells**.

    This is the margin's first named component: it is the largest thing in
    the footprint that does not scale with the mesh, so it is the thing whose
    arena placement can move the pool high-water without any input changing.
    """

    columns = radiation_chunk_columns(card)
    return columns * SHORTWAVE_GPOINTS * SPCVMC_WK_ARRAYS * (int(levels) + 2) * 4


#: The margin's second named component: the gap between the per-process
#: ``nvidia-smi`` convention the rows are fitted on and the whole-device
#: convention a door faces.  MEASURED both ways in one process, twice:
#: 11.2 MiB on the x4 arm and 0.75 MiB on the graded arm.  The larger is
#: carried.
CONVENTION_MARGIN_BYTES = int(round(11.2 * MIB))

#: What a row measured before the merged tip must be restated by.  The
#: 2026-08-26 re-fit measured the same mesh on the same card at the merged
#: tip and at hex ``7fe514b``: ``x1.40962`` rose 484.0 MiB.  A row taken at
#: an older tip is an under-statement by about that, and a gate must not
#: quote an under-statement without saying so.
PIN_RESTATEMENT_BYTES = int(round(484.0 * MIB))

#: What a global row measured on the 2026-08 proof stack must be restated
#: by, per cell, to price the stack the door runs today.  THE BREAKAGE IT
#: PREVENTS (measured 2026-10-11, RTX PRO 6000 host, RTX PRO 6000, 188 SM, hex
#: lane/hex-arbitrary-physics + engine 9f3f52565): a bare-defaults 30 km
#: global run (655,362 cells) peaked 82,404 MiB on its process after the
#: door priced it at 66,086 MiB from the 170 SM row; the door admitted it
#: and, before the late-health fix, it ran out of card memory.  The proof
#: rows were fitted on the x4 recorded schedule, which runs the monotonic
#: scalar limiter OFF, before CUDA-graph subcycle staging and before New
#: Tiedtke became the 30 km default; the bare MPAS-A defaults run all
#: three.  The per-allocation hook at the peak put the limiter's work
#: arrays alone at 11,580 MiB live (two species x level x edge, two
#: species x interface x cell and six species x level x cell arrays, in the
#: run that ran out of memory) and the graph's staged inputs
#: at 7,018 MiB, all mesh-scaled.  The restatement is the measured 188 SM
#: slope (:data:`_SHAPED_ROWS` ``global/188sm``) less the 170 SM proof
#: slope; it is a property of the build, as the slope is, and is carried
#: by every global row measured on the older stack and by every derived
#: row.  Not measured on those cards: a ledger arm on each replaces it.
STACK_RESTATEMENT_BYTES_PER_CELL = 124_655.2996 - 96_581.5833

#: What arming the forecast rollback (``--stop-on-refusal``) adds, per cell.
#: The plain forecast loop then keeps two committed boundaries for the late
#: health gate and a per-step physics boundary export.  MEASURED
#: 2026-10-11 on the 30 km global run above: the two-deep snapshot's own
#: receipt reports 24,070,131,576 device bytes, and the per-allocation hook
#: puts one seam export at 7,462.6 MiB (half of the two slots' 14,925.2
#: MiB; the per-step physics export copies the same arrays).  31.9 GB over
#: 655,362 cells.  THE BREAKAGE IT PREVENTS: an armed run admitted at the
#: bare-run price dies part-way into its first steps, as the 30 km run did
#: at 97 GB while it carried the two-deep snapshot unread.
ROLLBACK_ARMED_BYTES_PER_CELL = (24_070_131_576 + 7_462.6 * MIB) / 655_362

#: The CUDA per-thread local frame the widest LAUNCHED kernel carries, from
#: the ledger's ``local_memory_model.max_local_size_bytes`` at the merged tip
#: (``wsm6_column``, 7,216 B).  CUDA reserves one per-context backing store
#: of this times the card's resident threads, at first launch, and never
#: returns it -- so it is the largest card-scaled term in the core and the
#: one a derived row rescales.  The reservation CUDA actually takes is for
#: achievable occupancy and is therefore <= this product, which makes a
#: derived row conservative.
WIDEST_LAUNCHED_LOCAL_FRAME_BYTES = 7_216


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StagePeakFit:
    """A device high-water fit, including retained pool and outside-pool bytes.

    The intercept belongs to this measured stage trend, not to the physical
    core. Taking its maximum with the existing shaped row preserves the
    smaller-mesh floor where a large-mesh trend extrapolates below it.
    """

    stage: str
    cells: tuple[int, int]
    peak_bytes: tuple[int, int]

    @property
    def bytes_per_cell(self) -> float:
        return max(0.0, (self.peak_bytes[1] - self.peak_bytes[0])
                   / (self.cells[1] - self.cells[0]))

    @property
    def intercept_bytes(self) -> float:
        return max(peak - self.bytes_per_cell * cells
                   for cells, peak in zip(self.cells, self.peak_bytes))

    def predict_bytes(self, cells: int) -> float:
        return self.intercept_bytes + self.bytes_per_cell * cells

    def as_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "measured_cells": list(self.cells),
                "measured_peak_bytes": list(self.peak_bytes),
                "intercept_bytes": self.intercept_bytes,
                "bytes_per_cell": self.bytes_per_cell,
                "scope": "two measured sizes; extrapolated outside their interval"}


@dataclass(frozen=True)
class ShapedFootprintModel:
    """One card's, one configuration's footprint row -- core plus shape.

    ``predict_bytes`` and ``max_cells`` keep the signatures the affine
    :class:`FootprintModel` had, so every caller of the admission sum is
    unchanged; what moved is what those methods compute.
    """

    core_bytes: float
    bytes_per_cell: float
    card: CardProfile
    configuration: str
    provenance: str
    levels: int = MODEL_VERTICAL_LEVELS
    measured: bool = True
    derived_from: str | None = None
    #: :data:`STACK_RESTATEMENT_BYTES_PER_CELL` on a row measured on the
    #: 2026-08 proof stack; zero on a row measured on today's.
    restated_bytes_per_cell: float = 0.0
    #: :data:`ROLLBACK_ARMED_BYTES_PER_CELL` when the run arms the forecast
    #: rollback (``--stop-on-refusal``); zero otherwise.
    rollback_bytes_per_cell: float = 0.0
    stage_peak_fits: tuple[StagePeakFit, ...] = ()
    radiation_chunk_override: int | None = None
    shortwave_arrays_override: int | None = None

    # -- the terms, each nameable on its own ------------------------------
    @property
    def effective_bytes_per_cell(self) -> float:
        """Every per-cell term this row charges, summed."""

        return (
            float(self.bytes_per_cell)
            + float(self.restated_bytes_per_cell)
            + float(self.rollback_bytes_per_cell)
        )

    def tiled_bytes(self, cells: int) -> int:
        return sum(
            ws.bytes_for(int(cells), self.card, self.levels)
            for ws in TILED_WORKSPACES
        )

    def tiled_breakdown(self, cells: int) -> dict[str, int]:
        return {
            ws.name: ws.bytes_for(int(cells), self.card, self.levels)
            for ws in TILED_WORKSPACES
        }

    def predict_bytes(self, cells: int) -> float:
        if cells <= 0:
            raise ValueError("cells must be positive")
        shaped = (
            float(self.core_bytes)
            + (self.effective_bytes_per_cell - self.rollback_bytes_per_cell) * int(cells)
            + float(self.tiled_bytes(int(cells)))
        )
        peak = max((fit.predict_bytes(int(cells)) for fit in self.stage_peak_fits), default=shaped)
        return max(shaped, peak) + self.rollback_bytes_per_cell * int(cells)

    # -- the margin, and what each part of it absorbs ----------------------
    def margin_terms(self) -> dict[str, dict[str, Any]]:
        """The margin, itemised.  Every entry names its breakage."""

        columns = self.radiation_chunk_override or radiation_chunk_columns(self.card)
        arrays = self.shortwave_arrays_override or SPCVMC_WK_ARRAYS
        workspace = columns * SHORTWAVE_GPOINTS * arrays * (int(self.levels) + 2) * 4
        return {
            "arena_placement": {
                "bytes": workspace,
                "columns": columns,
                "workspace_arrays": arrays,
                "absorbs": (
                    "the RRTMG shortwave spcvmc workspace being mapped "
                    "fresh instead of served from the CuPy pool's free list "
                    "at the instant radiation runs"
                ),
                "breakage": (
                    "the run dies inside a CuPy allocation part-way through "
                    "the first radiation call, minutes in, on a card the gate "
                    "had already admitted"
                ),
                "measured": ((
                    "24 km census: 8,192 columns x 112 g-points x 16 arrays x "
                    "57 levels x 4 bytes = 3,192 MiB, observed as one pool "
                    "request; 22 km auto width is 7,936 columns. Holds the "
                    "larger observed block for arena placement, a derived "
                    "margin rather than another measured device peak"
                ) if self.radiation_chunk_override is not None else (
                    "freeing 38.4 MiB EARLIER in the run moved x1.40962's pool "
                    "high-water 5,404.5 -> 7,111.7 MiB, +1,707.2 MiB, "
                    "reproduced across two runs one of which was on an idle "
                    "card (STATE.md, the construction-diagnostic hold)"
                )),
                "scales_with": "the card's radiation chunk width",
            },
            "instrument_convention": {
                "bytes": CONVENTION_MARGIN_BYTES,
                "absorbs": (
                    "the gap between the per-process nvidia-smi rows these "
                    "cores are fitted on and the whole-device free memory a "
                    "door reads"
                ),
                "breakage": (
                    "a card with exactly the predicted peak free is short by "
                    "the convention gap and refuses or dies at the margin"
                ),
                "measured": (
                    "11.2 MiB on the x4 arm and 0.75 MiB on the graded arm, "
                    "both conventions sampled in one process"
                ),
                "scales_with": "nothing",
            },
        }

    def margin_bytes(self) -> int:
        return int(sum(term["bytes"] for term in self.margin_terms().values()))

    # -- the sums the gates use -------------------------------------------
    def required_bytes(self, cells: int, margin_bytes: int | None = None) -> int:
        margin = self.margin_bytes() if margin_bytes is None else int(margin_bytes)
        return int(round(self.predict_bytes(int(cells)))) + margin

    def max_cells(self, budget_bytes: int, margin_bytes: int | None = None) -> int:
        """Largest cell count this budget admits.

        Solved by bisection rather than by division because the model is
        piecewise linear: the tiled terms stop growing at their knees, so
        there is no single slope to divide by.
        """

        margin = self.margin_bytes() if margin_bytes is None else int(margin_bytes)
        budget = int(budget_bytes) - margin
        if budget <= 0 or self.effective_bytes_per_cell <= 0.0:
            return 0
        if self.predict_bytes(1) > budget:
            return 0
        low, high = 1, 1
        while self.predict_bytes(high) <= budget:
            low = high
            high *= 2
            if high > 1 << 40:  # pragma: no cover - no mesh is this large
                return low
        while low + 1 < high:
            mid = (low + high) // 2
            if self.predict_bytes(mid) <= budget:
                low = mid
            else:
                high = mid
        return low

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": ("max(shaped footprint, measured stage peaks) + rollback staging"
                      if self.stage_peak_fits else
                      "core + sum(tiled workspaces at min(cells, tile)) + slope * cells"),
            "core_bytes": float(self.core_bytes),
            "core_mib": float(self.core_bytes) / MIB,
            "bytes_per_cell": float(self.bytes_per_cell),
            "restated_bytes_per_cell": float(self.restated_bytes_per_cell),
            "rollback_bytes_per_cell": float(self.rollback_bytes_per_cell),
            "effective_bytes_per_cell": self.effective_bytes_per_cell,
            "configuration": self.configuration,
            "levels": int(self.levels),
            "card": self.card.as_dict(),
            "measured": bool(self.measured),
            "derived_from": self.derived_from,
            "tiled_workspace_knee_cells": {
                ws.name: ws.knee_cells(self.card) for ws in TILED_WORKSPACES
            },
            "stage_peak_fits": [fit.as_dict() for fit in self.stage_peak_fits],
            "radiation_chunk_columns": self.radiation_chunk_override or radiation_chunk_columns(self.card),
            "margin_bytes": self.margin_bytes(),
            "margin_terms": self.margin_terms(),
            "provenance": self.provenance,
            # Kept so a receipt written against the affine row still finds a
            # field with this name; it is the core, and it is NOT the retired
            # row's fixed term.
            "fixed_bytes": float(self.core_bytes),
            "fixed_mib": float(self.core_bytes) / MIB,
        }


# ---------------------------------------------------------------------------
# the measured rows
# ---------------------------------------------------------------------------
def row_key(configuration: str, card: CardProfile) -> str:
    return f"{configuration}/{int(card.multiprocessors)}sm"


#: The measured rows.  Keyed by configuration and multiprocessor count,
#: because those are the two things that actually move the core: what device
#: stack is built, and how many threads the card holds resident.
_SHAPED_ROWS: dict[str, dict[str, Any]] = {
    "global/170sm": {
        "core_bytes": 3_860_338_719.0,
        "bytes_per_cell": 96_581.5833,
        "card": "32gib-170sm",
        "configuration": "global",
        "measured": True,
        "stack": "proof-2026-08",
        "provenance": (
            "two-point fit on the per-allocation ledger session measured "
            "2026-08-26 on the proving RTX 5090 (170 SM) at the merged tip, "
            "hex 2009db7 + engine 26daaab7e: x1.40962 peaked 8,874.0 MiB and "
            "x4.163842 20,446.0 MiB, both by this process's nvidia-smi row.  "
            "The tiled-workspace terms are subtracted from both points BEFORE "
            "the fit, which is the whole difference from the retired affine "
            "row: 2,167 B/cell of that row's slope was GF and YSU tile growth "
            "between a mesh below both knees and a mesh above both.  "
            "evidence/memory-row-refit-20260826/"
        ),
    },
    "global/70sm": {
        "core_bytes": 1_084.9401 * MIB + PIN_RESTATEMENT_BYTES,
        "bytes_per_cell": 115_143.1,
        "card": "16gib-70sm",
        "configuration": "global",
        "measured": True,
        "stack": "proof-2026-08",
        "provenance": (
            "two-point fit on the per-allocation ledger session measured 2026-08-25 on "
            "the proving RTX 5070 Ti (70 SM) at hex 5252421 + engine "
            "26daaab7e: x1.40962 peaked 6,272.0 MiB and u96.64002 8,802.0 MiB.  "
            "NOT MEASURED AT THE MERGED TIP -- the core carries "
            "PIN_RESTATEMENT_BYTES (+484.0 MiB, what the tip moved x1 by on "
            "the 170 SM card) so the row is not quoted as an under-statement.  "
            "CAVEAT CARRIED FROM THE 08-24 FIT ON THIS CARD: the two peaks "
            "land at different instants and the pair separates core from slope "
            "imperfectly, which is why this row's slope is 19 % above the 170 "
            "SM card's over a lever arm one fifth as long.  One per-allocation ledger arm at "
            "the tip on a third mesh replaces it.  "
            "evidence/l6-capacity-20260825/ (the 70 SM card's run)"
        ),
    },
    "global/68sm": {
        "core_bytes": 755.7227 * MIB + PIN_RESTATEMENT_BYTES,
        "bytes_per_cell": 96_581.5833,
        "card": "10gib-68sm",
        "configuration": "global",
        "measured": True,
        "stack": "proof-2026-08",
        "provenance": (
            "single-point core from the per-allocation ledger row measured 2026-08-25 on "
            "the desktop RTX 3080 (68 SM, sm_86, WDDM) at hex 5252421 + engine "
            "26daaab7e: x1.40962 peaked 6,340.5 MiB device-view.  WDDM "
            "publishes no per-process row, so that figure INCLUDES the "
            "desktop's 1,142.5 MiB run-start baseline and the baseline is "
            "REMOVED here -- the door compares against free memory, which "
            "already has the desktop taken out of it, so a core carrying the "
            "baseline charges the desktop twice.  The retired tier row "
            "(2,483.0 MiB fixed) did exactly that.  Slope BORROWED from the "
            "170 SM row because the slope is a property of the build; core "
            "carries PIN_RESTATEMENT_BYTES.  "
            "evidence/l6-capacity-20260825/desktop3080/"
        ),
    },
    "limited-area/170sm": {
        "core_bytes": 6_324.7 * MIB,
        "bytes_per_cell": 96_581.5833,
        "card": "32gib-170sm",
        "configuration": "limited-area",
        "measured": True,
        # The pinned regional record runs the limiter off and its five
        # culls were measured whole-run at the door; nothing measured
        # restates it.
        "stack": "regional-2026-08",
        "provenance": (
            "ENVELOPE, NOT A FIT.  The per-allocation instrument has never "
            "been run on a regional mesh, so this core is the LARGEST residue "
            "(peak minus tiled terms minus slope*cells) over the five "
            "concentric culls measured 2026-08-27 on the proving RTX 5090, "
            "all 1,080 steps of full physics at dt 20 s through "
            "`woof hex forecast --lbc-dir`, whole-device sampler on an idle "
            "card: r4.75.4440 5,552 MiB, r4.75.7975 7,336, r4.75.11020 7,670, "
            "r4.75.14050 7,712, r4.75.15755 7,728.  The five residues span "
            "4,988-6,325 MiB and the largest is taken, so this row "
            "over-predicts the other four by up to 1,337 MiB.  The regional "
            "device stack is a different configuration -- padded atmosphere, "
            "two levels of lateral-boundary state, 22 more kernels -- and it "
            "is 85 % mesh-independent at these cell counts, which is why the "
            "retired affine row over-predicted every cull by 1.2-1.9 GiB.  "
            "Slope BORROWED from the global 170 SM row: four of the five "
            "points are flat to within 228 MiB across a 1.98x cell count, so "
            "these five cannot determine a slope.  ONE per-allocation ledger arm on a "
            "cull replaces this whole row.  evidence/nest-ratio-20260827/"
        ),
    },
    "global/188sm": {
        "core_bytes": 2_771_989_299.0,
        "bytes_per_cell": 124_655.2996,
        "card": "96gib-188sm",
        "configuration": "global",
        "measured": True,
        "stack": "bare-defaults-2026-10",
        "provenance": (
            "single-mesh row, split by the per-allocation hook, measured "
            "2026-10-11 on an RTX PRO 6000 Blackwell Server (188 SM, "
            "97,887 MiB) at hex lane/hex-arbitrary-physics + the "
            "late-health fix, engine 9f3f52565, through the bare forecast "
            "door: x1.655362 (655,362 cells, 55 levels), GFS 2026-10-03 "
            "12Z, MPAS-A defaults (monotonic limiter, CUDA-graph "
            "subcycles, New Tiedtke at 30 km).  The 6 h run (120 steps) "
            "peaked 82,963 MiB on the card, 82,404 MiB above the card's "
            "559 MiB idle floor; the 1 h run peaked 81,311.  CORE: at the "
            "1 h run's device peak the hook found 2,127.2 MiB of live "
            "blocks that are not a whole multiple of the cell, edge or "
            "vertex count and 2,367 MiB of device memory outside the CuPy "
            "pool (context, module images, local-memory backing store); "
            "their sum less this card's tiled workspaces (1,850.6 MiB) is "
            "the core.  SLOPE: the rest of the 6 h peak over the cell "
            "count, which therefore carries the pool's cached blocks.  ONE "
            "mesh: core and slope are separated by the ledger, not by a "
            "second mesh; a second-mesh arm replaces the split"
        ),
    },
}

SHAPED_ROWS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {key: MappingProxyType(dict(row)) for key, row in _SHAPED_ROWS.items()}
)

#: The global row the door prices TODAY's build with.  The four global rows
#: above are measurements of earlier builds and stay as the record they are
#: (their receipts, the restatement and the tests that refit them); none of
#: them is the door's price for the global configuration any more.
#:
#: THE BREAKAGE IT PREVENTS.  The hex memory census cuts (branch
#: hex/mem-census-cuts, 2026-10-16: census items 3, 4, 5, 7, 9 and 11 with
#: the mesh, state-levels, acoustic, diagnostics, pool and io-halo memory
#: lanes) cut route A's device peak to under half at 60 km.  Priced with the
#: earlier rows the door refused every run the cuts made fit -- x1.163842 on
#: an RTX 5090 with 18.4 GB free was refused at 24,832.6 MiB predicted while
#: it peaks at 12,244 MiB -- and at the same time UNDER-priced the build it
#: was fitted for: that build measured 27,882 MiB at x1.163842 against the
#: 24,832.6 MiB the 170 SM row predicted, so a card with 26.6 GB free was
#: admitted into an out-of-memory.
#:
#: MEASURED 2026-10-16 on the proving RTX 5090 (170 SM), route A
#: (default ArWen physics) through the bare forecast door, one hour each, the
#: run's own nvidia-smi row sampled every 500 ms, the same inputs and card for
#: both arms:
#:
#:   ==========  ===============================  =================================
#:   mesh        before: hex 3b6175b3 + engine    after: hex/mem-census-cuts e212d4a5
#:               088511604c6 (graphs on)          + engine cb2a7f4af54 (eager)
#:   ==========  ===============================  =================================
#:   x1.40962    9,892 MiB (6 steps)              5,726 MiB
#:   x1.163842   27,882 MiB (10 steps)            12,244 MiB
#:   ==========  ===============================  =================================
#:
#: Every history word of every frame identical between the arms (46
#: variables x 3 frames; 140,211,882 and 560,830,122 words).  The row is the
#: two-point fit of the "after" arm the way every row here is fitted: the
#: tiled workspaces (1,419.6 and 1,673.4 MiB) come off both points first.
#: The core carries the card-sized radiation chunk and the CUDA local-memory
#: store, so another card is DERIVED from this row by model_for_card's two
#: card-scaled mechanisms, exactly as before.
CURRENT_GLOBAL_ROW: Mapping[str, Any] = MappingProxyType({
    "core_bytes": 2_326_008_761.0,
    "bytes_per_cell": 53_454.1167,
    "card": "32gib-170sm",
    "configuration": "global",
    "measured": True,
    "stack": "census-cuts-2026-10",
    "provenance": (
        "two-point fit, hex/mem-census-cuts e212d4a5 + engine "
        "lane/292-hex-memory cb2a7f4af54, RTX 5090 (170 SM), route A, bare "
        "forecast door, 1 h: x1.40962 peaked 5,726 MiB and x1.163842 12,244 "
        "MiB by this process's nvidia-smi row; the same runs on hex 3b6175b3 + "
        "engine 088511604c6 peaked 9,892 and 27,882 MiB with every history "
        "word identical.  Tiled workspaces subtracted before the fit.  "
        "Measured 2026-10-16."
    ),
})

#: What one same-host rank costs per LOCAL column on today's build (its own
#: cells plus its K=3 halo), measured with two ranks on one RTX 5090
#: (``--samehost-devices 0,0``), route A, 1 h, hex/mem-census-cuts e212d4a5:
#: x1.40962 (21,553 + 21,557 local cells) peaked 6,054 MiB and x1.163842
#: (83,946 + 83,940) 14,818 MiB, one process holding both ranks.  The fit
#: (tiled workspaces per rank first) is 58,267.3 B per local cell and 1,082.8
#: MiB per rank; a rank on its own card is charged that card's whole single-card
#: core instead, which is larger.  The same 120 km pair on hex 3b6175b3 peaked
#: 12,904 MiB, history words identical.
#:
#: THE BREAKAGE IT RETIRES, AND THE ONE IT STILL PREVENTS.  The 2026-10-12
#: envelope (:data:`SAMEHOST_RANK_ENVELOPE_PEAK_BYTES`, 65,123 MiB at 334,756
#: local columns) priced the same-host overhead the single-card census could
#: not see: about 57.6 KB per column of construction-stream blocks stranded in
#: an arena the stepping stream never reused and halo fields held to the end
#: of the step.  The io-halo memory lane fixed both (merged in
#: hex/mem-census-cuts), so the envelope stops being the price; a rank is still
#: never priced below this measured same-host slope, which is what keeps the
#: four-card 15 km run that died at step 43 refused on a build that still
#: carries the overhead (that build no longer prices from this row).
SAMEHOST_RANK_BYTES_PER_CELL = 58_267.3


#: The largest per-card peak measured for ONE same-host rank on a 188 SM
#: card, and the local column count it carried: 15 km x1.2621442 on 8 x RTX
#: PRO 6000, 334,756 local columns per rank, 65,123 MiB (nvidia-smi, the
#: largest of the 8 cards; hex 15km-speed branch d04c30d + native ozone
#: bridge, the wide radiation chunk this tree runs).  The same rank size on
#: two cards (30 km, 327,681 columns) peaked 57,533-61,345 MiB in four arms.
#: Receipts: the 15 km speed sprint report of 2026-10-11
#: ("Before and after") and evidence/g1/g1-30km-2card-table.md.
SAMEHOST_RANK_ENVELOPE_PEAK_BYTES = 65_123 * MIB
SAMEHOST_RANK_ENVELOPE_LOCAL_CELLS = 334_756


@dataclass(frozen=True)
class SameHostRankPeak:
    """One same-host rank's whole-run per-card peak, measured on a card of this SM count."""

    multiprocessors: int
    levels: int
    local_cells: int
    peak_mib: float
    receipt: str
    #: 1 for the single-card loop, otherwise the same-host rank count measured.
    ranks: int = 2


#: Whole-run per-card peaks of ONE same-host rank on its own card, measured.
#:
#: THE BREAKAGE IT PREVENTS (2026-10-24, 2 x RTX PRO 6000): the 30 km
#: globe on two cards, 331,942 local columns per rank, was priced from the
#: single-card core plus :data:`SAMEHOST_RANK_BYTES_PER_CELL` at 22,705 MiB
#: (25,907 MiB with the margin), so the run declared 26,000 MiB to the box
#: lease; each card peaked at 29,538 MiB, and the pool admitted other jobs to
#: those cards against 3.5 GB that was not there.  The slope two ranks on ONE
#: card measured does not see what a rank on its own card holds (its own
#: context, pool arena and halo staging), so a rank is never priced below the
#: line through its card's measured peak.
SAMEHOST_RANK_PEAKS: tuple[SameHostRankPeak, ...] = (
    SameHostRankPeak(
        multiprocessors=188, levels=55, local_cells=331_942, peak_mib=29_538.0,
        receipt=("x1.655362 30 km globe, 2 cards, 24 h, GFS 2026-10-22 12Z, WOOF physics, fast numerics, "
                 "the hex 0.4.5 multi-card fix + woof 2.9.5; the pool over-budget record "
                 "(nvidia-smi, the larger of the two cards), 2026-10-24")),
    # The single-card loop on the same case and card: its row predicts 39,877 MiB (43,080 MiB with
    # the margin) and the run peaked at 48,032 MiB.  Priced only into the lease estimate
    # (woof.hex.drivers.samehost_forecast_v841.lease_vram_mib), not the single-card admission.
    SameHostRankPeak(
        multiprocessors=188, levels=55, local_cells=655_362, peak_mib=48_032.0, ranks=1,
        receipt=("x1.655362 30 km globe, 1 card, 24 h, GFS 2026-10-22 12Z, WOOF physics, fast numerics, "
                 "the hex 0.4.5 multi-card fix + woof 2.9.5; the pool over-budget record "
                 "(nvidia-smi), 2026-10-24")),
)


def _through_measured_rank_peaks(model: ShapedFootprintModel, *, single_card: bool = False) -> ShapedFootprintModel:
    """``model`` with its per-column slope raised so it is never below a measured rank peak."""

    for peak in SAMEHOST_RANK_PEAKS:
        if ((peak.ranks == 1) != single_card
                or peak.multiprocessors != model.card.multiprocessors or peak.levels != model.levels
                or model.predict_bytes(peak.local_cells) >= peak.peak_mib * MIB):
            continue
        cells = peak.local_cells
        slope = (peak.peak_mib * MIB - float(model.core_bytes) - float(model.tiled_bytes(cells))
                 - float(model.restated_bytes_per_cell) * cells) / cells
        model = replace(model, bytes_per_cell=slope,
                        provenance=model.provenance + "; MEASURED RANK PEAK: slope "
                        f"{slope:,.1f} B/column through the measured {peak.peak_mib:,.0f} MiB at "
                        f"{cells:,} local columns ({peak.receipt})")
    return model


def samehost_rank_model(model: ShapedFootprintModel, *, ranks: int = 2) -> ShapedFootprintModel:
    """Price one resident same-host rank from same-host measurements.

    THE BREAKAGE IT PREVENTS (measured 2026-10-12, G3, 4 x RTX PRO 6000, hex
    a4a9bab + engine 7008877b6): global 15 km on four cards, 664,586 local
    columns per rank, bare door.  The previous pricing took the single-card
    global/188sm row and SUBTRACTED 7,017 MiB of CUDA-graph staging that
    same-host ranks do not allocate ("derived, not measured"); it predicted
    76,385 MiB per rank, the door admitted, every card filled to 94.9-96.9 GB
    by step 5 and rank 0 died out of memory at step 43 (pool 96,598,311,936
    bytes, a 2,635,825,152-byte limiter temporary unmappable): at least
    98,088 MiB with the 3.37 GiB measured outside the pool.  A same-host rank
    is HEAVIER per column than the single-card loop, not lighter: one 30 km
    rank of 655,362 columns runs out of memory at 97.7 GB where the
    single-card door peaks at 83.8 GB (15km-speed lane).

    The row: the single-card core and tiled workspaces, and a slope fitted
    so the line passes through the largest same-host peak measured
    (:data:`SAMEHOST_RANK_ENVELOPE_PEAK_BYTES`).  It is an ENVELOPE through
    one rank size, used for bare and armed ranks alike: the one bare
    same-host point (above) lies above the armed line minus the rollback
    term, so bare ranks are not given the rollback discount.  It predicts
    124,860 MiB at 664,586 columns (measured >= 98,088, refused either way
    on a 96 GB card) and 63,842 MiB at 327,681 (measured 57,533-61,345).
    A bare same-host fit at a second rank size replaces it.  Other cards
    keep their rows: no same-host rank has been measured on them, and their
    smaller pools strand less (30 km on four 32 GiB cards peaked 31.6 GiB).
    """
    if (model.configuration == "global"
            and model.bytes_per_cell == CURRENT_GLOBAL_ROW["bytes_per_cell"]
            and model.restated_bytes_per_cell == 0.0):
        # Today's build (CURRENT_GLOBAL_ROW, measured or derived for this card):
        # the rank carries the card's single-card core and tiled workspaces and
        # the measured same-host slope per local column.
        # The measured 2-rank peak line (SAMEHOST_RANK_PEAKS) prices a rank that
        # shares the run with others; a run of ONE rank is not raised to it.
        # Breakage: extrapolated to one 1,216,922-column rank it refused a
        # 96 GB card the single-card measurements admit.
        through = _through_measured_rank_peaks if int(ranks) > 1 else (lambda value: value)
        if model.bytes_per_cell >= SAMEHOST_RANK_BYTES_PER_CELL:
            return through(model)
        return through(replace(model, bytes_per_cell=SAMEHOST_RANK_BYTES_PER_CELL,
                       rollback_bytes_per_cell=0.0, measured=model.measured,
                       derived_from=model.derived_from,
                       provenance=model.provenance + "; SAME-HOST RANK: "
                       f"{SAMEHOST_RANK_BYTES_PER_CELL:,.1f} B per local column, measured "
                       "with two ranks on one RTX 5090 at x1.40962 and x1.163842 "
                       "(hex/mem-census-cuts); bare and armed ranks priced alike"))
    row = SHAPED_ROWS["global/188sm"]
    if (model.configuration != "global" or model.card.multiprocessors != 188
            or model.bytes_per_cell != row["bytes_per_cell"]
            or model.core_bytes != row["core_bytes"]):
        return model
    cells = SAMEHOST_RANK_ENVELOPE_LOCAL_CELLS
    slope = (SAMEHOST_RANK_ENVELOPE_PEAK_BYTES - float(model.core_bytes)
             - float(model.tiled_bytes(cells))) / cells
    return replace(model, bytes_per_cell=slope, restated_bytes_per_cell=0.0,
                   rollback_bytes_per_cell=0.0, measured=True,
                   derived_from="samehost-rank-envelope/188sm",
                   provenance=model.provenance + "; SAME-HOST RANK ENVELOPE: slope "
                   f"{slope:,.1f} B/column through the largest measured same-host rank "
                   f"peak ({SAMEHOST_RANK_ENVELOPE_PEAK_BYTES / MIB:,.0f} MiB at "
                   f"{cells:,} local columns, 15 km on 8 cards); bare and armed "
                   "ranks priced alike; one rank size, not a two-size fit")


def _current_global_model() -> ShapedFootprintModel:
    """The measured row of today's build (:data:`CURRENT_GLOBAL_ROW`)."""

    row = CURRENT_GLOBAL_ROW
    return ShapedFootprintModel(
        core_bytes=float(row["core_bytes"]),
        bytes_per_cell=float(row["bytes_per_cell"]),
        card=KNOWN_CARDS[str(row["card"])],
        configuration=str(row["configuration"]),
        provenance=str(row["provenance"]),
        measured=bool(row["measured"]),
    )


def _row_model(key: str) -> ShapedFootprintModel:
    row = _SHAPED_ROWS[key]
    return ShapedFootprintModel(
        core_bytes=float(row["core_bytes"]),
        bytes_per_cell=float(row["bytes_per_cell"]),
        card=KNOWN_CARDS[str(row["card"])],
        configuration=str(row["configuration"]),
        provenance=str(row["provenance"]),
        measured=bool(row["measured"]),
        restated_bytes_per_cell=(
            STACK_RESTATEMENT_BYTES_PER_CELL
            if row["stack"] == "proof-2026-08"
            else 0.0
        ),
    )


#: The row of record: the 170 SM card, the global configuration, today's
#: build (:data:`CURRENT_GLOBAL_ROW`).  It is the DEFAULT and the derivation
#: reference, not a universal answer -- the door reads the card and
#: re-selects.  The 2026-08 proof row it replaced is ``_row_model("global/170sm")``.
FOOTPRINT_MODEL = _current_global_model()


# Whole-device stage high waters, including live and retained CuPy blocks,
# diagnostic/GF temporaries and history capture. Re-derived by the regression
# from tools/device_memory_ledger/admission-stage-peaks-20261017.json.
# The 24 km case commits 60 steps; 22 km commits 360 with 3-hour history.
LARGE_GLOBAL_STAGE_PEAK_BYTES = (
    ('construct', 32_679_198_720, 38_960_168_960),
    ('cuda_context', 32_907_788_288, 39_230_701_568),
    ('dynamics.solve_diagnostics', 72_418_131_968, 91_107_950_592),
    ('history.capture', 70_121_750_528, 85_101_707_264),
    ('physics._run_cumulus', 73_825_320_960, 96_763_969_536),
    ('physics._run_noahmp', 65_426_227_200, 88_067_080_192),
    ('physics._run_radiation', 65_426_227_200, 88_067_080_192),
    ('physics._run_sfclay', 65_426_227_200, 88_067_080_192),
    ('physics._run_ysu', 65_426_227_200, 88_067_080_192),
    ('step', 75_119_263_744, 96_763_969_536),
)


def _with_large_global_census(model: ShapedFootprintModel) -> ShapedFootprintModel:
    if (model.card.multiprocessors != 188 or model.card.max_threads_per_sm != 1536
            or model.levels != 55):
        return model
    return replace(model,
        stage_peak_fits=tuple(StagePeakFit(stage, (1_022_592, 1_216_922), (a, b))
                             for stage, a, b in LARGE_GLOBAL_STAGE_PEAK_BYTES),
        radiation_chunk_override=8192, shortwave_arrays_override=16,
        provenance=(
            "MEASURED large-global stage envelope, 2026-10-17, 188 SM / 1536 threads, "
            "55 levels, fast WOOF physics, one same-host rank, graphs off: "
            "1,022,592 cells / 60 steps and 1,216,922 cells / 360 steps. "
            "Includes dynamics diagnostics workspace, GF convection scratch, "
            "retained CuPy pool blocks, outside-pool memory and history staging. "
            "Maximum of two-point stage fits, extrapolated outside these sizes, "
            "and the existing smaller-mesh shaped floor. The placement margin "
            "prices the observed 8192-column, 16-array shortwave request. "
            "Evidence: tools/device_memory_ledger/admission-stage-peaks-20261017.json. "
            "Smaller-mesh floor provenance: " + model.provenance))


def model_for_card(
    card: CardProfile | None = None,
    configuration: str = "global",
    levels: int = MODEL_VERTICAL_LEVELS,
    *,
    rollback_armed: bool = False,
) -> ShapedFootprintModel:
    """The footprint row for this card and configuration.

    Measured if this card has been measured; DERIVED, and labelled so,
    otherwise.  This is the function the borrowed-row defect was open for: the retired
    module had per-card rows and no caller, so a 10 GiB desktop was priced
    with a 32 GiB card's fixed term and refused two meshes it had been
    measured running.
    """

    rollback = ROLLBACK_ARMED_BYTES_PER_CELL if rollback_armed else 0.0
    if card is None:
        card = REFERENCE_CARD
    key = row_key(configuration, card)
    if configuration == "global":
        # Today's build: measured on the 170 SM card (CURRENT_GLOBAL_ROW),
        # DERIVED for every other card by the two card-scaled mechanisms
        # below.  The earlier global rows in _SHAPED_ROWS priced earlier
        # builds and stay as their record.
        reference = _current_global_model()
        if card.multiprocessors == reference.card.multiprocessors:
            return replace(reference, card=card, levels=int(levels),
                           rollback_bytes_per_cell=rollback)
        return _with_large_global_census(_derived_model(
            reference, row_key("global", REFERENCE_CARD), card,
            configuration, levels, rollback))
    if key in _SHAPED_ROWS:
        model = _row_model(key)
        if levels != model.levels:
            model = ShapedFootprintModel(
                core_bytes=model.core_bytes,
                bytes_per_cell=model.bytes_per_cell,
                card=model.card,
                configuration=model.configuration,
                provenance=model.provenance,
                levels=int(levels),
                measured=model.measured,
                restated_bytes_per_cell=model.restated_bytes_per_cell,
            )
        # A row is measured on ONE card; if the caller hands us the same SM
        # count under a different name, the arithmetic is identical and the
        # name is the caller's.
        return ShapedFootprintModel(
            core_bytes=model.core_bytes,
            bytes_per_cell=model.bytes_per_cell,
            card=card,
            configuration=model.configuration,
            provenance=model.provenance,
            levels=int(levels),
            measured=True,
            restated_bytes_per_cell=model.restated_bytes_per_cell,
            rollback_bytes_per_cell=rollback,
        )

    reference_key = row_key(configuration, REFERENCE_CARD)
    if reference_key not in _SHAPED_ROWS:
        raise KeyError(
            f"no measured row for configuration {configuration!r} on any card, "
            "so nothing can be derived for "
            f"{card.name!r}; add a measured row before admitting on it"
        )
    reference = _row_model(reference_key)
    return _derived_model(reference, reference_key, card, configuration, levels, rollback)


def _derived_model(reference: ShapedFootprintModel, reference_key: str,
                   card: CardProfile, configuration: str, levels: int,
                   rollback: float) -> ShapedFootprintModel:
    """``reference`` rescaled to ``card`` by the two card-scaled mechanisms."""

    local_store_saving = WIDEST_LAUNCHED_LOCAL_FRAME_BYTES * (
        REFERENCE_CARD.resident_threads - card.resident_threads
    )
    radiation_saving = shortwave_workspace_bytes(
        REFERENCE_CARD, levels
    ) - shortwave_workspace_bytes(card, levels)
    core = float(reference.core_bytes) - float(local_store_saving) - float(
        radiation_saving
    )
    if core < 0.0:
        core = float(reference.core_bytes)
    return ShapedFootprintModel(
        core_bytes=core,
        bytes_per_cell=reference.bytes_per_cell,
        restated_bytes_per_cell=reference.restated_bytes_per_cell,
        rollback_bytes_per_cell=rollback,
        card=card,
        configuration=configuration,
        levels=int(levels),
        measured=False,
        derived_from=reference_key,
        provenance=(
            f"DERIVED, NOT MEASURED: no per-allocation ledger arm exists for "
            f"{card.multiprocessors} SM on the {configuration} configuration, "
            f"so this row rescales the {reference_key} core by the two "
            "mechanisms that are known and measured to scale with the card -- "
            "the CUDA per-context local-memory backing store (widest launched "
            f"frame {WIDEST_LAUNCHED_LOCAL_FRAME_BYTES} B x resident threads, "
            "the upper bound CUDA prices, so the saving is understated and the "
            "row is conservative) and the RRTMG shortwave chunk width.  "
            "Checked against the two cards that DO have their own "
            "measurement, this derivation over-predicts by 182.7 MiB on the "
            "70 SM part and 490.8 MiB on the 68 SM part -- conservative on "
            "both, which is the direction a gate needs.  Slope is the "
            "reference's: the slope is a property of the build.  One per-allocation ledger arm "
            "on this card replaces this row"
        ),
    )


# ---------------------------------------------------------------------------
# the ONE admission sum
# ---------------------------------------------------------------------------
def required_free_bytes(
    cells: int,
    model: ShapedFootprintModel | None = None,
    margin_bytes: int | None = None,
) -> int:
    """Free device bytes this mesh needs on this model: the ONE admission sum.

    ``predicted peak + margin``, rounded once.  The door's verdict, the
    driver argv the door builds, and the per-mesh floor the mesh binding
    installs all call this function; a test asserts they carry the same
    number to the byte.

    The margin is the model's own -- two named, measured components, not a
    flat constant.  Passing ``margin_bytes`` overrides it, which is what
    ``--headroom-mib`` does and why that flag now reports the model's figure
    as its default rather than 512 MiB.
    """

    model = FOOTPRINT_MODEL if model is None else model
    return model.required_bytes(int(cells), margin_bytes)


#: The physics routes a member can be priced for.  Every footprint row above was measured on the ArWen route
#: (route A); the MPAS route (``woof.hex.mpasphys``, the hex default since 0.4.6) holds more.
PHYSICS_ROUTES = ("arwen", "mpas")


@dataclass(frozen=True)
class RouteRunPeak:
    """One forecast's whole-run peak on its card, measured with nothing else of its own on that card."""

    route: str
    multiprocessors: int
    levels: int
    cells: int
    peak_mib: float
    receipt: str


#: Whole-run peaks of ONE member on the MPAS physics route (fixed step, fast numerics).
#:
#: THE BREAKAGE IT PREVENTS (U-H4, global synradar cycle, 2026-10-27 00:28Z): ``woof hex global --row
#: global-30km --source gefs --members-per-card auto`` on two RTX PRO 6000 (97,887 MiB) priced each 30 km MPAS
#: member at the ArWen-route floor (43,080 MiB), planned 2 per card, and all four members ran out of memory about
#: 3 min in (CuPy: "Out of memory allocating 432,537,600 bytes (allocated so far: 48,761,238,528 bytes)"); the
#: next 2-member groups met the same end.  One member alone on that card peaked 75,069 MiB.
MPAS_ROUTE_RUN_PEAKS: tuple[RouteRunPeak, ...] = (
    RouteRunPeak(
        route="mpas", multiprocessors=188, levels=55, cells=655_362, peak_mib=75_069.0,
        receipt=("x1.655362 30 km uniform globe, GEFS 2025-05-19 06Z members, 6 h, hex 0.4.8 MPAS physics "
                 "convection_permitting, dt 180 s, one member alone on an RTX PRO 6000 (97,887 MiB); the card's "
                 "nvidia-smi row sampled every 5 s over five member rounds, 2026-10-27")),
    RouteRunPeak(
        route="mpas", multiprocessors=170, levels=55, cells=40_962, peak_mib=12_664.0,
        receipt=("x1.40962 120 km uniform globe, GEFS 2025-05-19 06Z members 1-2, 1 h (6 steps of 600 s, radiation "
                 "at steps 0 and 3), hex 0.4.8 MPAS physics convection_permitting, fast numerics, one member per "
                 "group on an RTX 5090; the member processes' nvidia-smi compute-apps rows sampled every 1 s peaked "
                 "12,664 and 12,400 MiB (an earlier run of member 1 alone peaked 10,940 MiB), 2026-10-27")),
)
#: Uniform quasi-uniform globes only (the rows the global ensemble door runs).  Variable-resolution and multi-zoom
#: globes are priced by their own whole runs in ``card_fit``; a 3-zone multi-zoom globe of 143,420 cells peaked
#: 26,800 MiB on the MPAS route, above this line, so this curve is not their price.

#: Headroom over the MPAS-route whole-run curve, the same 10 % the card-fit whole-run curve carries
#: (``card_fit.WHOLE_RUN_MARGIN``): the curve passes through sampled peaks a sampler can miss by a spike.
ROUTE_RUN_MARGIN = 0.10


def route_run_curve_mib(cells: int, route: str = "mpas", levels: int = MODEL_VERTICAL_LEVELS) -> float | None:
    """Peak MiB of one ``route`` member on ``cells`` columns, from that route's measured whole runs, or None.

    Straight between neighbouring measured runs (peaks made non-decreasing in cells), the end segments' slopes
    past the smallest and largest; one run alone scales cell-linear.  The runs were measured on different cards;
    the per-card terms (CUDA local-memory store, radiation chunk) differ between them by a few hundred MiB, far
    inside :data:`ROUTE_RUN_MARGIN`.  Derived between and past the measured sizes, not measured there."""

    points: dict[int, float] = {}
    for peak in MPAS_ROUTE_RUN_PEAKS if route == "mpas" else ():
        if peak.levels == int(levels):
            points[peak.cells] = max(points.get(peak.cells, 0.0), float(peak.peak_mib))
    if not points:
        return None
    xs = sorted(points)
    ys: list[float] = []
    for x in xs:
        ys.append(max([points[x], *ys[-1:]]))
    x = float(cells)
    if len(xs) == 1:
        return ys[0] * x / xs[0]
    segment = next((i for i in range(1, len(xs)) if x <= xs[i]), len(xs) - 1)
    x0, x1, y0, y1 = xs[segment - 1], xs[segment], ys[segment - 1], ys[segment]
    return max(0.0, y0 + (y1 - y0) * (x - x0) / (x1 - x0))


def route_required_free_bytes(cells: int, model: ShapedFootprintModel | None = None,
                              margin_bytes: int | None = None, *, route: str = "arwen") -> int:
    """:func:`required_free_bytes` for one member on ``route``.

    The WOOF route is exactly :func:`required_free_bytes`.  The MPAS route is never below it, and never below
    its own measured whole-run curve (:func:`route_run_curve_mib`) plus :data:`ROUTE_RUN_MARGIN` plus the
    model's margin.  The co-tenancy price both the ensemble planner and the roster admission charge per member
    (U-H4, :data:`MPAS_ROUTE_RUN_PEAKS`)."""

    if route not in PHYSICS_ROUTES:
        raise ValueError(f"physics route {route!r} is not one of {PHYSICS_ROUTES}")
    model = FOOTPRINT_MODEL if model is None else model
    base = required_free_bytes(cells, model, margin_bytes)
    if route == "arwen":
        return base
    curve = route_run_curve_mib(int(cells), route, int(model.levels))
    if curve is None:
        return base
    margin = model.margin_bytes() if margin_bytes is None else int(margin_bytes)
    return max(base, int(round(curve * (1.0 + ROUTE_RUN_MARGIN) * MIB)) + margin)


def native_device_floor_bytes() -> int:
    """The re-derived ``NATIVE_DEVICE_FLOOR``: the x4.163842 requirement."""

    return required_free_bytes(NATIVE_MESH_CELLS)


# ---------------------------------------------------------------------------
# retired arms -- computable, dated, never a gate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FootprintModel:
    """A RETIRED affine row: fixed term plus a per-cell slope.

    This is the shape the footprint-spread and borrowed-row defects retired on 2026-08-27.  It survives
    only so the before arm of a comparison has one home -- see
    :func:`retired_affine_row_floor_bytes`.  It is not a gate and must never
    become one.
    """

    fixed_bytes: float
    bytes_per_cell: float
    provenance: str

    def predict_bytes(self, cells: int) -> float:
        if cells <= 0:
            raise ValueError("cells must be positive")
        return self.fixed_bytes + self.bytes_per_cell * cells

    def max_cells(self, budget_bytes: int, headroom_bytes: int) -> int:
        usable = float(budget_bytes) - float(headroom_bytes) - self.fixed_bytes
        if usable <= 0.0 or self.bytes_per_cell <= 0.0:
            return 0
        return int(usable // self.bytes_per_cell)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fixed_bytes": self.fixed_bytes,
            "fixed_mib": self.fixed_bytes / MIB,
            "bytes_per_cell": self.bytes_per_cell,
            "provenance": self.provenance,
        }


#: The flat headroom the shaped margin retired.  It named no breakage, and it
#: failed by 96 MiB on v6.75.112676 and by 28.2 MiB on v20.80.151649.  Kept
#: computable so a before/after table has one home.  NEVER a gate.
RETIRED_FLAT_HEADROOM_BYTES = 512 * MIB

#: Retired alias, kept so an existing import does not break at a tree that
#: still names it.  It is the RETIRED flat constant, not the model's margin;
#: call ``model.margin_bytes()`` for the number a gate uses.
DEFAULT_HEADROOM_BYTES = RETIRED_FLAT_HEADROOM_BYTES

#: The asserted constant the 2026-08-25 re-derivation retired.
RETIRED_LINEAR_FLOOR_BYTES = 24 * 1024**3


def retired_linear_floor_bytes(cells: int) -> int:
    """What the RETIRED 24 GiB floor demanded at ``cells`` -- before arm only.

    ``24 GiB * cells / 163,842``: 157,285 B per cell, no fixed term.
    NEVER call this to admit or refuse a run.
    """

    if cells <= 0:
        raise ValueError("cells must be positive")
    return RETIRED_LINEAR_FLOOR_BYTES * int(cells) // NATIVE_MESH_CELLS


#: The row the 2026-08-26 re-fit retired.
RETIRED_ROW_20260825 = FootprintModel(
    fixed_bytes=4339.1 * MIB,
    bytes_per_cell=103_696.0,
    provenance=(
        "RETIRED 2026-08-26.  Measured 2026-08-25 on the proving RTX 5090 "
        "(170 SM) at the converged seam pin, hex 7fe514b + engine 26daaab7e, "
        "both published meshes in one per-allocation ledger session; "
        "evidence/pin-move-335-20260825/.  Superseded by the merged-tip "
        "re-fit at hex 2009db7 -- the same two meshes on the same card "
        "measured 8,874.0 / 20,446.0 MiB there against this row's "
        "8,390.0 / 20,542.0"
    ),
)


def retired_converged_row_floor_bytes(cells: int) -> int:
    """What the RETIRED 2026-08-25 row demanded -- before arm only."""

    if cells <= 0:
        raise ValueError("cells must be positive")
    return (
        int(round(RETIRED_ROW_20260825.predict_bytes(int(cells))))
        + RETIRED_FLAT_HEADROOM_BYTES
    )


#: The row THIS lane retired: the affine row of record from 2026-08-26.  It
#: was measured impeccably and it is the third generation of retired arm in
#: this module, and the three are different KINDS of wrong.  The 24 GiB proxy
#: was never measured.  The 08-25 row measured a different tree.  This one
#: measured the right tree on the right card with the right instrument and
#: had **the wrong shape**: a line in cell count, fitted across a mesh below
#: both tiled-workspace knees and a mesh above both, so it charged card-sized
#: workspace growth as if it were per-cell.
RETIRED_AFFINE_ROW_20260826 = FootprintModel(
    fixed_bytes=5016.5 * MIB,
    bytes_per_cell=98_748.0,
    provenance=(
        "RETIRED 2026-08-27.  Measured 2026-08-26 on the proving RTX 5090 "
        "(170 SM) at the merged tip, hex 2009db7 + engine 26daaab7e, both "
        "published meshes in one per-allocation ledger session -- the same session this "
        "module's global/170sm core is fitted from.  Retired for SHAPE, not "
        "for provenance: five placed-mesh forecast-door runs spanned "
        "+3.89 % to -6.76 % against it, one mesh with 15,343 more cells "
        "peaked 318 MiB lower, one run overran its own required_free by "
        "96 MiB, and five limited-area culls were over-predicted by "
        "1.2-1.9 GiB each.  evidence/memory-shape-20260827/"
    ),
)


def retired_affine_row_floor_bytes(cells: int) -> int:
    """What the RETIRED 2026-08-26 affine row demanded -- before arm only.

    Every receipt and chart that quotes the affine requirement computes it
    here rather than hand-typing it, because hand-typed before-arm numbers
    are exactly what outlive the constant they came from.

    NEVER call this to admit or refuse a run.
    """

    if cells <= 0:
        raise ValueError("cells must be positive")
    return (
        int(round(RETIRED_AFFINE_ROW_20260826.predict_bytes(int(cells))))
        + RETIRED_FLAT_HEADROOM_BYTES
    )


#: The per-card tier table the shaped rows retired.  It was correct data with
#: no caller: ``forecast_door._resolve_model`` never read it, which is ledger
#: the borrowed-row defect.  Kept computable so the before arm of the borrowed-row defect table has one home;
#: :func:`tier_model` still builds the affine model each row described.
RETIRED_CARD_TIER_ROWS_20260826: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "32gib-170sm": MappingProxyType(
            {
                "tier": "32 GiB",
                "card": "RTX 5090 (32,607 MiB, 170 SM, sm_120)",
                "measured": True,
                "fixed_mib": 5016.5,
                "bytes_per_cell": 98_748.0,
                "basis": "two-point per-allocation ledger fit at the merged tip, 2026-08-26",
            }
        ),
        "16gib-70sm": MappingProxyType(
            {
                "tier": "16 GiB",
                "card": "RTX 5070 Ti (16,303 MiB, 70 SM, sm_120)",
                "measured": True,
                "fixed_mib": 1774.0,
                "bytes_per_cell": 115_143.0,
                "basis": "two-point per-allocation ledger fit at hex 5252421, 2026-08-25",
            }
        ),
        "10gib-68sm": MappingProxyType(
            {
                "tier": "10 GiB",
                "card": "RTX 3080 (10,240 MiB, 68 SM, sm_86, WDDM)",
                "measured": True,
                "fixed_mib": 2483.0,
                "bytes_per_cell": 98_748.0,
                "basis": (
                    "single-point per-allocation ledger row at hex 5252421, 2026-08-25, slope "
                    "borrowed.  RETIRED ALSO FOR A SECOND DEFECT: the "
                    "6,340.5 MiB device-view peak it was derived from includes "
                    "the desktop's 1,142.5 MiB baseline, and the door compares "
                    "against free memory which already excludes it, so this "
                    "row charged the desktop twice"
                ),
            }
        ),
        "12gib": MappingProxyType(
            {
                "tier": "12 GiB",
                "card": "no 12 GiB card exists in the fleet",
                "measured": False,
                "fixed_mib": 5016.5,
                "bytes_per_cell": 98_748.0,
                "basis": (
                    "DERIVED, NOT MEASURED: the of-record 170 SM model quoted "
                    "at a 12 GiB budget.  Superseded by "
                    "device_admission.model_for_card, which derives from the "
                    "card's SM count instead of from its memory size"
                ),
            }
        ),
    }
)

#: Retired alias for the tier table, kept so an existing import resolves.
CARD_TIER_ROWS = RETIRED_CARD_TIER_ROWS_20260826


def tier_model(key: str) -> FootprintModel:
    """The RETIRED affine model one tier row described -- before arm only."""

    row = RETIRED_CARD_TIER_ROWS_20260826[key]
    label = "measured row" if row["measured"] else "DERIVED (not measured)"
    return FootprintModel(
        fixed_bytes=float(row["fixed_mib"]) * MIB,
        bytes_per_cell=float(row["bytes_per_cell"]),
        provenance=f"RETIRED 2026-08-27.  {row['card']} -- {label}: {row['basis']}",
    )


# ---------------------------------------------------------------------------
# the derivation, machine-readable
# ---------------------------------------------------------------------------
FLOOR_DERIVATION: Mapping[str, Any] = MappingProxyType(
    {
        "derived": "2026-08-27",
        "schema": "gpuwm-hex.device-admission.shaped/v1",
        "ruling": (
            "the four-swaths receipt's named follow-up, 2026-08-27: 'A linear "
            "model in cell count is the wrong shape for a placed mesh: "
            "something other than the cell count is setting the peak.  Until "
            "that is understood the 512 MiB headroom is not sufficient on "
            "this class of mesh.'  This is the shape, and it is arithmetic on "
            "the per-allocation ledger rather than another fit"
        ),
        "model": (
            "peak(cells, card) = core(card, configuration) "
            "+ sum(tiled workspace at min(cells, tile(card))) "
            "+ bytes_per_cell * cells; "
            "required = peak + arena_placement_margin(card) + convention_margin"
        ),
        "what_moved": (
            "the tiled physics workspaces came OUT of the fixed term and OUT "
            "of the slope and became their own term.  Grell-Freitas and YSU "
            "size a global device workspace to min(cells, SMs x blocks/SM x "
            "block) after the local-frame cut, and RRTMG sizes its column "
            "chunk from the card's resident threads after the device-sized radiation chunk.  Both fit "
            "meshes straddle the GF knee (43,520 cells) and the YSU knee "
            "(87,040) on the 170 SM card, so the affine fit charged 2,167 "
            "B/cell of tile growth as per-cell and put the saturated remainder "
            "in a fixed term that a smaller card does not have"
        ),
        "reproduces_the_ledger": {
            "grell_freitas_x1_mib": 1261.97,
            "grell_freitas_x4_mib": 1338.75,
            "ysu_x1_mib": 157.62,
            "ysu_x4_mib": 334.69,
            "shortwave_spcvmc_bytes_170sm": 1_830_420_480,
            "note": (
                "the three sites the per-allocation ledger records as NOT scaling 4.00x "
                "with a 4x mesh, and the only three; the formulas in this "
                "module reproduce all three from card constants alone"
            ),
        },
        "margin_replaces": (
            "512 MiB flat, shared by every card and every mesh, naming no "
            "breakage.  It failed by 96 MiB on v6.75.112676 (measured 16,236 "
            "MiB against a 16,139.6 MiB required_free) and by 28.2 MiB on "
            "v20.80.151649.  Still computable at "
            "device_admission.RETIRED_FLAT_HEADROOM_BYTES"
        ),
        "retired_rows_computable_at": (
            "retired_affine_row_floor_bytes (2026-08-26 affine row of record), "
            "retired_converged_row_floor_bytes (2026-08-25 row), "
            "retired_linear_floor_bytes (the 24 GiB proxy), "
            "tier_model (the per-card affine tier table)"
        ),
        "open_and_stated": {
            "limited_area_core_is_an_envelope": (
                "the per-allocation instrument has never run on a regional "
                "mesh; the core is the largest of five whole-device residues "
                "and over-predicts four of them by up to 1,337 MiB.  ONE per-allocation ledger "
                "arm on a cull replaces it"
            ),
            "four_swath_spread_not_separated": (
                "the +3.89 %/-6.76 % spread across four placed meshes has a "
                "named mechanism -- arena placement of the 1,745.6 MiB "
                "shortwave block, measured at 1,707.2 MiB in the "
                "construction-diagnostic A/B -- but NO A/B on those four "
                "meshes has been run, so the attribution is inference.  The "
                "settling arm is one per-allocation ledger run per swath mesh with the "
                "pool high-water and the peak-instant site census recorded"
            ),
            "70sm_slope": (
                "115,143 B/cell against the 170 SM card's 96,582 over a lever "
                "arm one fifth as long, same engine pin.  Probably a "
                "short-baseline artefact and treated as the card's own row "
                "because a gate must not under-predict; one per-allocation ledger arm on a "
                "third mesh settles it"
            ),
            "nz_dependence": (
                "the tiled terms are functions of nz; the cores and the slope "
                "were measured only at nz=55 and are NOT re-derivable at "
                "another level count from anything in this module"
            ),
        },
    }
)
