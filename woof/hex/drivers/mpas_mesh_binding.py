#!/usr/bin/env python
"""Mesh, geometry, timestep, and memory binding for the frozen v8.4.1 path.

The frozen x4 proof remains an asserted no-op at its recorded timestep.  Every
other bind is made from a registry entry whose grid/static bytes, dimensions
and nominal resolution are explicit; the run's timestep is the caller's
(``dt_seconds``) or MPAS-A's 6 s per km of nominal spacing.  The stability
length is the finite positive ``dcEdge`` array in the supplied Earth-scaled
static file; ``nominalMinDc`` is never used as its substitute for stability.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from woof.hex import convection_admission
from woof.hex import pbl_cadence as pbl_cadence_module
from woof.hex import device_admission
from woof.hex import dt_admission
from woof.hex import cascade_row
from woof.hex import mesh_row_candidate
from woof.hex import mesh_rows
from woof.hex import mpas_defaults
from woof.hex.cell_coordination_admission import (
    CellCoordinationAdmissionError,
    CellCoordinationPolicy,
    admit_cell_coordination,
)
from woof.hex.dual_edge_admission import (
    DualEdgeAdmission,
    DualEdgeAdmissionError,
    DualEdgePolicy,
    admit_dual_edges,
)
from woof.hex.timestep_admission import (
    CourantPolicy,
    EdgeLengthAuthority,
    SplitExplicitSchedule,
    TimestepAdmissionError,
    admit_timestep,
    edge_length_authority,
)

__all__ = [
    "MeshBindingError",
    "MeshBindingMismatch",
    "MeshBinding",
    "MESH_BINDINGS",
    "constants_fingerprint",
    "binding_fingerprint",
    "bind_mesh",
    "regional_mask_digest",
    "admit_regional_row",
]


class MeshBindingError(RuntimeError):
    """Base class: a run is refused rather than executed under an unproved bind."""


class MeshBindingMismatch(MeshBindingError):
    """The declared mesh and supplied bytes/geometry/timestep do not agree."""


# The radiation cadence a run gets when it names none: MPAS-A's own
# config_radtlw_interval/config_radtsw_interval default, 00:30:00
# (Registry.xml:2316, 2321).  The run's actual cadence reaches bind_mesh as
# ``radiation_seconds`` and is the one a timestep must divide.
PHYSICS_RADIATION_CADENCE_SECONDS = mpas_defaults.DEFAULT_RADIATION_SECONDS

# The timestep the native x4 reference was integrated at, kept as a module
# name because the registry rows quote it.  It admits and refuses nothing:
# since 2026-10-10 a run's timestep is --dt or MPAS-A's 6 s per km, and the
# earned-anchor registry is evidence recorded in the receipt.
FROZEN_LANE_DT_SECONDS = dt_admission.PROVEN_DT_SECONDS


@dataclass(frozen=True)
class MeshBinding:
    name: str
    n_cells: int
    n_edges: int
    n_levels: int
    n_interfaces: int
    n_soil_levels: int
    nominal_dx_m: float
    #: The timestep the row was registered with.  Since 2026-10-10 it is not
    #: the run's timestep: a run takes ``--dt``, or by default 6 s per km of
    #: nominal spacing (woof.hex.mpas_defaults.default_dt_seconds); ``--dt row``
    #: selects this value.
    dt_seconds: float
    grid_bytes: int
    grid_sha256: str
    static_bytes: int
    static_sha256: str
    #: The largest horizontal wind the stability check bounds the advective
    #: Courant number with (woof.hex.timestep_admission.CourantPolicy).
    courant_wave_speed_m_s: float = 125.0
    frozen_native: bool = False
    scale_admission_floor: bool = True
    drop_carried_deformation: bool = False
    # Regional (limited-area) row fields -- pure table work.  A regional cull
    # is registered by filling these three slots, never by new code paths:
    # the ring count of the boundary zone (7 on every measured native cull),
    # the digest of the grid's bdyMaskCell/Edge/Vertex triple as computed by
    # :func:`regional_mask_digest`, and the lateral-boundary source slot --
    # nullable today, and a regional row with an empty slot is REFUSED at
    # bind by name, because no boundary stream exists yet to force the zone.
    boundary_zone_width: int | None = None
    bdy_mask_sha256: str | None = None
    lbc_source: str | None = None
    # The native invariant stream a mesh's own workflow configures, as the
    # file NAME its fix directory carries beside the grid (table data: a new
    # mesh fills the slot, no code path keys on it).  When the slot is filled
    # and that file sits beside --grid or --static, the forecast door hands it
    # to the driver by default, so the vertical grid comes from it as it does
    # in MPAS-A.  THE BREAKAGE THIS CLOSES (measured
    # 2026-10-05): RRFS v2 configures the stream on conus12km, and hex read
    # the vertical grid from init.nc instead, 2.7 mm off in zgrid and 8e-5
    # relative in zz; native v8.4.1 with versus without the stream differs by
    # 2.0e-4 K RMS theta at hour 1, 7 times NOAA's rounding envelope.
    invariant_file: str | None = None
    notes: str = ""
    #: The coarsest cell spacing a generated variable-resolution row
    #: declares (its background), so preflight can take the per-cell
    #: convection decision from the row alone.  ``None`` on every uniform
    #: row; the bind decides from the grid's own meshDensity either way.
    coarsest_dx_m: float | None = None

    @property
    def regional(self) -> bool:
        return self.boundary_zone_width is not None or self.bdy_mask_sha256 is not None

    def courant_policy(self, max_wind_m_s: float | None = None) -> CourantPolicy:
        """The stability policy: the run's ``--max-wind``, else this row's
        wind bound.  Both are assumed bounds, recorded with their source."""

        if max_wind_m_s is not None:
            return CourantPolicy(
                max_characteristic_speed_m_s=float(max_wind_m_s),
                max_characteristic_speed_source="--max-wind",
            )
        return CourantPolicy(
            max_characteristic_speed_m_s=self.courant_wave_speed_m_s,
            max_characteristic_speed_source=(
                "default" if float(self.courant_wave_speed_m_s) == 125.0
                else "the mesh row's courant_wave_speed_m_s"
            ),
        )

    def dual_edge_policy(self, *, apvm_upwinding: float = 0.5) -> DualEdgePolicy:
        """One floor for every row, deliberately with no per-row override.

        The Courant policy is per-row because a wave speed is a modelling
        choice. Dual-edge amplification is not: it is a property of the mesh
        the operators inherit, so a row cannot admit itself past it. Changing
        the floor means editing woof.hex.dual_edge_admission, where the
        measured anchors that set it live.
        """

        return DualEdgePolicy(apvm_upwinding=apvm_upwinding)

    def cell_coordination_policy(self) -> CellCoordinationPolicy:
        """One floor for every row, for the same reason the dual-edge one is.

        How many edges a cell has is a property of the mesh, not a modelling
        choice, so a row cannot admit itself past it. Changing the floor means
        editing woof.hex.cell_coordination_admission, where the measurement
        that set it lives.
        """

        return CellCoordinationPolicy()


MESH_BINDINGS: Mapping[str, MeshBinding] = MappingProxyType(
    {
        "x4.163842": MeshBinding(
            name="x4.163842",
            n_cells=163_842,
            n_edges=491_520,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=25_000.0,
            dt_seconds=120.0,
            grid_bytes=224_139_172,
            grid_sha256="48e747157bb1f0b83b96505e268699dfb562b4c1428468cb91457fbb03b1be55",
            static_bytes=298_860_376,
            static_sha256="f064ee8f8d40085db4bf77a3d5fc6081cd92368b7d3dd32d98110b8b64d177e8",
            frozen_native=True,
            scale_admission_floor=False,
            notes=(
                "Frozen native proof shape. Binding must change no module constant or "
                "trajectory; geometry is still independently Courant-admitted. "
                "Admitted under the named floors NATIVE_DEVICE_FLOOR and "
                "NATIVE_RESTART_FLOOR, re-fitted at the merged tip 2026-08-26 "
                "(2026-08-26) from the measured row -- "
                "woof.hex.device_admission.FLOOR_DERIVATION is the derivation of "
                "record; the measured x4 peak at that tip is 20,446 MiB "
                "(2026-08-26, RTX 5090). "
                "Static provenance (read from the file, 2026-08-24): native MPAS-A "
                "v8.4.1 init_atmosphere with config_native_gwd_static=YES; no writer "
                "of ours ever touched it, so the retired writer's antipodal drag band "
                "does not apply. Rebuilding it would change the frozen mesh/reference relationship "
                "anchor and is refused."
            ),
        ),
        "x1.40962": MeshBinding(
            name="x1.40962",
            n_cells=40_962,
            n_edges=122_880,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=120_000.0,
            dt_seconds=120.0,
            grid_bytes=56_039_332,
            grid_sha256="9a9e1909a755dac209462ceb0bfffd77ac1b37503169568b7f296707ee612bb9",
            static_bytes=94_766_584,
            static_sha256="cf1a47d4168327f06a8403555d6ed8b2fe1aff7f8b916bb7f6a754c34a10ac82",
            drop_carried_deformation=True,
            notes=(
                "Published 120 km mesh. Device admission scales with columns; timestep "
                "is explicit and independently checked against physical dcEdge. "
                "Static provenance (read from the file, 2026-08-24): the NCAR-published "
                "static, built by native init_atmosphere v8.2.0 on glade with "
                "config_native_gwd_static=YES; no writer of ours produced it, so the "
                "retired writer's antipodal drag band does not apply and it is kept."
            ),
        ),
        "v15.150.38857": MeshBinding(
            name="v15.150.38857",
            n_cells=38_857,
            n_edges=116_565,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=15_000.0,
            dt_seconds=60.0,
            grid_bytes=53_162_368,
            grid_sha256="0e6ac7c46140b24010764e840f3b1b77d52adb4abb16cac7cae4ce384b84c3b6",
            static_bytes=74_304_272,
            static_sha256="199c16ca993edfca9335b9e63b63db0a67e0eb201179d3dd1df1f9510420635f",
            drop_carried_deformation=True,
            notes=(
                "Generated 15-to-150 km variable-resolution mesh. It deliberately does "
                "not inherit x4's 120 s timestep: registry declares 60 s and the actual "
                "finite positive dcEdge minimum must admit it before CUDA contact. "
                "Static rebuilt 2026-08-24 by the unified rw_mpas_static (82-variable "
                "union): the previously pinned static came from the retired writer, "
                "whose drag band sampled terrain 180 degrees of longitude from every "
                "cell (archive-origin assumption); measured corr(old var2d, new var2d) "
                "= +0.003 at the same cell and +0.697 at lon+180. The retired pin "
                "a326fad338a4 also omitted deriv_two, cell_gradient_coef_x/y, "
                "defc_a/defc_b and the soil-composition group; no file matching it "
                "survives on any reachable machine. The rebuilt static carries real "
                "defc tables, so like the published mesh this row drops them at "
                "attach: the frozen v8.4.1 path runs deformation inactive. "
                "REFUSED AT BIND since 2026-08-24 by the dual-edge admission, and "
                "the row is kept because it is the measurement that sets the floor: "
                "the density-biased Fibonacci seed this mesh was generated from is "
                "polycrystalline, so its Delaunay carries 3,447 heptagons beside "
                "3,459 pentagons and the near-cocircular dislocation quads collapse "
                "dvEdge to 6.514 m at edge 19786 (dcEdge 38,657 m, ratio 1.685e-04, "
                "TRiSK tangential amplification 5,935x). Measured consequence "
                "(an RTX 5070 Ti): every runaway magnitude in the first outer "
                "step sits on that edge and the run dies at composite step 0. It has "
                "never completed a forecast and no timestep makes it able to."
            ),
        ),
        "conus-x1.2971": MeshBinding(
            name="conus-x1.2971",
            n_cells=2_971,
            n_edges=9_116,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=120_000.0,
            dt_seconds=120.0,
            grid_bytes=4_229_936,
            grid_sha256="d2ec651a0e03e045029b9179b270ebe987f311c8b052da16ecbb2d75215f9b51",
            static_bytes=5_755_728,
            static_sha256="7762b60f6002c111e49f2edb657e641bfaaaf05c115883d8cbb5b6f15118edc7",
            drop_carried_deformation=True,
            boundary_zone_width=7,
            bdy_mask_sha256=(
                "acc95da7ecc58253e0085332eb5acc827d42287ecae4fe9ea88bd64060f4d67e"
            ),
            lbc_source="CANDIDATE-REGIONAL-DRY/lbc-x1",
            notes=(
                "The first regional row whose lbc_source slot is FILLED, so the "
                "bind admits execution instead of refusing for want of a "
                "boundary stream. Culled from the registered x1.40962 pair by "
                "MPAS-Limited-Area v2.2 (commit edc556e1) on the verbatim "
                "conus.custom.pts of the L0 record, so the parent bytes are the "
                "ones this registry already pins. Its boundary stream is the "
                "three-file native case-9 series of the CANDIDATE-REGIONAL-DRY "
                "record set (init a461950569842e61af750cd90d9b9074138ceaf5c8eb"
                "10706397dfec99d4edcd), and the runtime that consumes it is "
                "woof.hex.regional_v841. Registered as the per-step byte "
                "ladder of the regional CPU authority lane: 2,971 cells is the "
                "size at which the Python whole-step driver runs a step in "
                "minutes rather than hours, while the 44,770-cell x4 cull "
                "stays the endpoint target. Grid dcEdge is unit-sphere as "
                "published; the static carries the Earth-scaled metrics, which "
                "is why the Courant admission reads the static."
            ),
        ),
        "v15.60.224210": MeshBinding(
            name="v15.60.224210",
            n_cells=224_210,
            n_edges=672_624,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=15_000.0,
            dt_seconds=75.0,
            grid_bytes=306_733_600,
            grid_sha256="6096d58406e30c5e2920d12072d38e04544ad4793afc432277081f7f069e5118",
            static_bytes=428_699_212,
            static_sha256="47be91587f711356533c0b2dee1ba69ab2bcbe2ee1d0305931af0c09ac036c85",
            drop_carried_deformation=True,
            notes=(
                "Generated graded 15-to-60 km mesh, the first variable-resolution "
                "row this project generated that clears every floor: hierarchical "
                "Goldberg ladder (GP(86,61) level 0, two refinement levels, "
                "count-changing surgery in the annuli), 224,210 cells against "
                "224,208 predicted, min dvEdge/dcEdge 0.0407 -- 2.04x the "
                "admission floor, zero edges under 0.04 -- min dvEdge 925.7 m, "
                "admitted under the 2026-08-25 length floor (the old "
                "7,500 m anchor guarded a retired load check and refused "
                "x4.163842 itself). Census 2,133 defect cells (0.95%), zero "
                "outside the transition annuli beyond the base twelve. Declared "
                "dt 75 s: the measured Courant limit is 95.84 s (min dcEdge "
                "13,311.8 m), and 75 s is the largest value at or below that "
                "limit which also divides the 600 s radiation cadence exactly "
                "(eight steps) and closes the model clock in binary64 -- "
                "600/7 = 85.714... clears Courant and does neither. This row "
                "read '90 s' until 2026-08-26 and the row itself never did: "
                "90 s was moved to 75 s when the cadence rule landed and the "
                "sentence was not. Admission re-measures from this static's own "
                "dcEdge at bind. Static from the unified rw_mpas_static; like "
                "every non-native row the carried defc tables drop at attach. "
                "DEVICE ADMISSION, corrected 2026-08-26 and re-shaped "
                "2026-08-27: the requirement is whatever "
                "woof.hex.device_admission.required_free_bytes returns for "
                "this row's cell count ON THE CARD THE DOOR IS LOOKING AT -- "
                "quoted as a fixed GiB figure here until 2026-08-27, which is "
                "how a 32 GiB card's number ends up refusing a 10 GiB card -- "
                "and the RTX 5090 holds it: measured ADMITTED on an RTX 5090, "
                "2026-08-26, "
                "evidence/device-floor-rederive-20260826/. The refusal this "
                "row carried until then belonged to the retired linear proxy, "
                "which demanded more than the card's total. TIMESTEP, "
                "corrected the same day: 75 s earned an anchor on an RTX 5070 Ti "
                "(dt_admission, evidence/dt-anchors-20260826/, the dt 75 s anchor), "
                "so the row's declared timestep is admitted too. Both of this "
                "row's recorded refusals are retired and NEITHER was ever a "
                "property of the mesh; it has not yet been run. "
                "Evidence: tree/evidence/graded-goldberg-20260825/."
            ),
        ),
        "v15.60.224197": MeshBinding(
            name="v15.60.224197",
            n_cells=224_197,
            n_edges=672_585,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=15_000.0,
            dt_seconds=75.0,
            grid_bytes=306_715_724,
            grid_sha256="981f95a058597394d933df3ab40ad2545a5d435d6575b5036889866330d9c741",
            static_bytes=428_674_356,
            static_sha256="0bcd6cb5922446d7c10827b9d7d57c1b88c61d1c0bed3b3d80d406ce7afc84fe",
            drop_carried_deformation=True,
            notes=(
                "The SAME graded 15-to-60 km request at different window "
                "coordinates, and the arbitrary-acceptance proof: it took "
                "ZERO code changes -- one JSON spec's center_deg moved, and "
                "the generator, the emit gate, this registry and the bind "
                "path all took it as data. Independently generated on the "
                "hierarchical ladder: 224,197 cells, min dvEdge/dcEdge "
                "0.04041 (2.02x the admission floor), min dvEdge 981.2 m, "
                "delivered median 1.0087, census 2,146 defect cells. "
                "Declared dt 75 s (Courant limit 89.95 s, min dcEdge "
                "12,492.8 m; 75 s also divides the 600 s radiation cadence). "
                "Evidence: "
                "tree/evidence/graded-goldberg-20260825/."
            ),
        ),
        "v16.66.195630": MeshBinding(
            name="v16.66.195630",
            n_cells=195_630,
            n_edges=586_884,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=16_507.912109375,
            dt_seconds=100.0,
            grid_bytes=267_636_876,
            grid_sha256="a53db2c1f1b39f56ed11e43eefc21eb4c5590bd71d3f2f43a80742455fe66563",
            static_bytes=374_054_260,
            static_sha256="92da723f11dd565343ea01a22316bce84ed249c3be6b159085a93117ea856848",
            drop_carried_deformation=True,
            notes=(
                "A 4x-request rescaling that a 32 GiB card holds with room: "
                "23.63 GiB required on the re-derived admission floor "
                "(predicted peak 23.13 GiB). It was registered on 2026-08-26 "
                "as the capacity BOUNDARY, and that framing is retired the "
                "same day: the boundary belonged to the retired linear proxy, "
                "and against the measured row the 32 GiB part carries about "
                "271,000 cells, so this row is simply a registered mesh. Its "
                "declared 100 s ALSO earned an anchor the same day "
                "(dt_admission, evidence/dt-anchors-20260826/dt100/), "
                "retiring the second refusal this row carried -- the frozen "
                "lane is no longer pinned to 120 s. THE BLOW-UP IS EXPLAINED, "
                "AND THIS ROW NOW REFUSES AT BIND FOR IT "
                "(evidence/graded-blowup-20260826/, an RTX 5090, "
                "2026-08-26). On 2026-08-26 the row bound at 100 s, "
                "integrated 22 steps and died at step 23 of 36. The cause is "
                "THIS MESH'S SINGLE 4-COORDINATED CELL, 195615 at 33.74N "
                "117.65W, inserted by the generator's count-changing defect "
                "surgery: in every arm the model's theta maximum AND its |w| "
                "maximum both sit on that one cell at the top model level, "
                "and theta there grows 197.4 K (100 s), 197.7 K (75 s) and "
                "181.3 K (20 s) above its initial value by 1,800 s while "
                "every global minimum stays flat to four figures. The three "
                "candidates the dt-anchor campaign left unseparated are "
                "SEPARATED AND TWO ARE DEAD: the Courant margin is not it -- "
                "at 75 s (28% margin against 3.5%) the run died at the same "
                "MODEL time, 2,325 s against 2,300 s, not the same step "
                "count; the TRiSK amplification is not it -- v20.80.151649 "
                "completed 6 h at 94.9% of its own Courant limit with a WORSE "
                "amplification, 24.34x against this mesh's 24.03x, and "
                "carries no cell below coordination 5. A smaller timestep "
                "only buys time: at 20 s this mesh finishes 1 h (180/180 "
                "steps, two arms, byte identical) still carrying a 181 K "
                "error at that cell, and then the SAME cell's vertical "
                "velocity runs 4.83 -> 35.04 m/s between 07:00Z and 07:30Z "
                "and reaches 281 m/s at step 495 of 540, where the "
                "step-health gate refuses it at 2 h 45 m. No timestep tested "
                "gives this mesh a forecast. THIS ROW HAS NEVER COMPLETED ONE "
                "BEYOND AN HOUR. REMEDY: regenerate -- and two of the three "
                "graded meshes reachable that day carried the same defect, so "
                "the durable fix is generator-side. The gate is "
                "woof.hex.cell_coordination_admission. 16.51 km inside a "
                "66.03 km background "
                "-- the canonical 4x request rescaled by one factor with "
                "every ratio preserved (--fit-spacing, the door's own printed "
                "remedy): gradient 1.50 %/cell at published parity, region "
                "attainment 1.048, 195,630 cells against 195,636 predicted. "
                "Hierarchical Goldberg ladder, min dvEdge/dcEdge 0.04162 "
                "(2.08x the admission floor, zero edges under 0.04), min "
                "dvEdge 944.2 m, delivered median 1.0069, max adjacent "
                "spacing ratio 1.1698, census 2,046 defect cells (1.05%) with "
                "none outside the transition annuli beyond the base twelve. "
                "Declared dt 100 s: the measured Courant limit is 103.67 s "
                "(min dcEdge 14,398.0 m) and 100 s divides the 600 s "
                "radiation cadence exactly six times. WHY NOT THE 15 km "
                "CANONICAL (v15.60.224210, registered above): on 2026-08-26 "
                "the answer was memory, and it no longer is -- the floor "
                "re-derivation puts that mesh at 26.39 GiB required against "
                "a 32 GiB card that holds it, measured admitted on an RTX 5090 "
                "the same day (evidence/device-floor-rederive-20260826/). "
                "What separated the two rows after that was the timestep, and "
                "that separation is gone too: 100 s and 75 s BOTH earned "
                "anchors on 2026-08-26 (evidence/dt-anchors-20260826/). "
                "Nothing in the registry refuses either row now; what neither "
                "has is a completed forecast. Evidence: "
                "tree/evidence/graded-goldberg-20260825/. THE DEFECT IS FIXED "
                "AT THE PRODUCER AND THIS ROW IS SUPERSEDED BY v16.66.195629 "
                "below (2026-08-26, woof the meshgen coordination work): the same "
                "spec row, unchanged, regenerated by a generator whose surgery "
                "carries a coordination clause, delivers 195,629 cells with no "
                "cell below coordination 5. This row stays registered, and "
                "stays refused, because it is the bytes that measured the cost."
            ),
        ),
        "v16.66.195629": MeshBinding(
            name="v16.66.195629",
            n_cells=195_629,
            n_edges=586_881,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=16_507.912109375,
            dt_seconds=100.0,
            grid_bytes=267_635_840,
            grid_sha256="98d4d9fda0a2da4b5a3286792b760fb5c9c0dea840cb35659f854d3c653a5088",
            static_bytes=374_052_348,
            static_sha256="4c7b44be4c97df89b9f75fcdd7d109e2690c3d3ccfff34d4473790fd05f01b04",
            drop_carried_deformation=True,
            notes=(
                "v16.66.195630 REGENERATED FROM ITS OWN SPEC ROW, UNCHANGED, "
                "BY A FIXED GENERATOR (2026-08-26, woof "
                "the meshgen coordination work). The spec bytes are identical -- "
                "16.508 km inside a 66.032 km background, cap at 37N 97W, the "
                "canonical 4x request rescaled by one factor -- and the only "
                "thing that changed is the engine: rw-mpas mesh surgery now "
                "reads coordination as half of its own repair test and "
                "re-anneals the cells its insertions damage instead of pinning "
                "them. WHAT THE OLD ROW DIED OF, AND WHY THIS ONE CANNOT: S2, "
                "the insertion operator, places a generator at the spacing-true "
                "point of a near-cocircular quad's long diagonal, which on such "
                "a quad IS the quad's common circumcentre -- the new cell's "
                "Delaunay ring is then exactly the four quad cells, so it is "
                "born a QUADRILATERAL and its two opposite neighbours go 6 -> "
                "7. Measured at 18 of 18 and 13 of 13 insertions on the two "
                "regenerated meshes: it is the rule, not an accident. Measured "
                "in the shipped v16.66.195630 bytes themselves: cell 195615's "
                "four neighbours lie on a circle of radius 20.783 km to within "
                "0.10 km, and 195615 sits 0.564 km from that circle's centre -- "
                "2.7 % of the radius. The local polish is what anneals such a "
                "cell into a legal one, and it PINNED every cell outside the "
                "current batch's seeds, so a cell damaged in one round was "
                "never moved again; nothing in the loop or in the emit gate "
                "ever read a coordination number, because a quadrilateral plus "
                "the two heptagons it makes leaves sum(6 - nEdgesOnCell) at "
                "exactly 12. CENSUS: {5: 1037, 6: 193568, 7: 1023, 8: 1}, zero "
                "cells below coordination 5, coordination defect 12. GEOMETRY: "
                "min dvEdge/dcEdge 0.040237 (2.01x the admission floor), min "
                "dvEdge 944.2 m, min dcEdge 14,398.0 m, max adjacent spacing "
                "ratio 1.1698, delivered median 1.0069, region attainment "
                "1.0477, gradient 1.4999 %/cell at published parity, 195,629 "
                "cells against 195,636 predicted. Declared dt 100 s: the "
                "measured Courant limit is 103.67 s and 100 s divides the 600 s "
                "radiation cadence exactly six times (the timestep-anchor campaign, "
                "evidence/dt-anchors-20260826/dt100/). Evidence: "
                "tree/evidence/meshgen-coordination-20260826/."
            ),
        ),
        "v20.80.151649": MeshBinding(
            name="v20.80.151649",
            n_cells=151_649,
            n_edges=454_941,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=20_000.0,
            dt_seconds=120.0,
            grid_bytes=207_471_508,
            grid_sha256="c1d0d5bc8eacde2d18020778f744c391d0d6e78c0b073798a2ddca08f997f4e4",
            static_bytes=289_962_580,
            static_sha256="183a3e436c4d9f252b1ac795e68ac0c840d689946276a822219302e119346003",
            drop_carried_deformation=True,
            notes=(
                "The graded mesh that RUNS: 20 km inside an 80 km background, "
                "the canonical 4x request rescaled by one factor with every "
                "ratio preserved, and the first variable-resolution mesh this "
                "project generated that clears BOTH measured frozen-lane "
                "boundaries at once. Inside the re-derived device admission "
                "floor (19.38 GiB required, predicted peak 18.88 GiB, against "
                "a 32 GiB card carrying about 271,000 cells), and "
                "coarse enough for the frozen lane's pinned 120 s: measured "
                "Courant limit 126.44 s (min dcEdge 17,561.1 m), and 120 s "
                "divides the 600 s radiation cadence five times. Hierarchical "
                "Goldberg ladder, gradient 1.50 %/cell at published parity, "
                "region attainment 1.048, min dvEdge/dcEdge 0.04108 (2.05x the "
                "admission floor, zero edges under 0.04), min dvEdge 1,322.3 m, "
                "delivered median 1.0060, max adjacent spacing ratio 1.1734. "
                "Evidence: tree/evidence/graded-goldberg-20260825/."
            ),
        ),
        "v4.75.123423": MeshBinding(
            name="v4.75.123423",
            n_cells=123_423,
            n_edges=370_263,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=4_000.0,
            dt_seconds=20.0,
            grid_bytes=168_858_348,
            grid_sha256="45cad8a00fa95c2c0ae72f215ed534dee06251f91be54bb818f5586d1c2e8d27",
            static_bytes=235_994_464,
            static_sha256="69c53f3335f2acaec4328f802fcb69c61f2dec94de2d28ee10e064c59e53adf1",
            drop_carried_deformation=True,
            notes=(
                "The first mesh this project PLACED rather than requested: the "
                "grid comes from a swath spec the placement machinery emitted "
                "for a detected feature, generated unedited by rw_mpas_mesh, "
                "and the static from the unified rw_mpas_static against that "
                "same grid (42.32 s, 12 geography datasets, 82 variables, "
                "FP32-exact nominalMinDc 4,000 m). A 4 km-requested core "
                "inside a 75 km background: attained finest spacing 4.700 km, "
                "min dcEdge 4,302.90 m, coarsest 89,859.4 m, spacing ratio "
                "17.60, region attainment 1.175 over an interior 271.97 km "
                "deep. Coordination 5->702, 6->122,032, 7->688, 8->1 with "
                "nothing below five edges; min dvEdge/dcEdge 0.040003 (2.00x "
                "the admission floor), shortest dual edge 308.34 m. "
                "Declared dt 20 s against a measured Courant limit of 30.98 s "
                "-- 1.55x of margin, the first registered row whose timestep "
                "sits near its own mesh's limit rather than tens of times "
                "beneath it -- and 20 s divides the 600 s radiation cadence "
                "thirty times. Finest spacing 4.000 m-declared / 4,302.90 m "
                "measured is ABOVE the 3 km convection threshold, so the "
                "3 km convection threshold leaves Grell-Freitas ON here and the row "
                "binds against the anchored (20 s, gf) configuration. The "
                "static carries real defc tables, so like the other generated "
                "rows this one drops them at attach. "
                "Evidence: tree/evidence/swath-first-forecast-20260826/."
            ),
        ),
        "v4.75.121182": MeshBinding(
            name="v4.75.121182",
            n_cells=121_182,
            n_edges=363_540,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=4_000.0,
            dt_seconds=20.0,
            grid_bytes=165_791_160,
            grid_sha256="efd363d0145842121e7c91791a4009586f8bd5aef0945face256b920c82369c5",
            static_bytes=231_709_672,
            static_sha256="ef7fe2ea44c0c2b7bd05c86b27b35385a7809f60b64f96370700f400fd074c54",
            drop_carried_deformation=True,
            notes=(
                "The first mesh placed on a storm this program DETECTED in its "
                "own model output rather than in a fixture: the 24 h u96.64002 "
                "global forecast of 2026-08-12 06Z carried a 944.6 hPa Southern "
                "Ocean cyclone south of Australia, the deepest low on the "
                "planet in that forecast, and the placement layer ranked it "
                "first of 41 cyclone tracks and emitted this spec for it. The "
                "grid is that spec generated unedited by rw_mpas_mesh, and the "
                "static is rw_mpas_static against that same grid (42 s, 12 "
                "geography datasets, 82 variables, FP32-exact nominalMinDc "
                "4,000 m). A 4 km-requested core inside a 75 km background: "
                "attained finest spacing 4.597 km, min dcEdge 4,457.23 m, "
                "coarsest 79,447.8 m, spacing ratio 17.28, delivered/requested "
                "median 1.0122. The mesh is GLOBAL -- sum(areaCell)/4pi = 1.0, "
                "Euler characteristic 2 -- so no lateral-boundary stream is "
                "involved. Coordination 5->595, 6->120,004, 7->583 with nothing "
                "below five edges; min dvEdge/dcEdge 0.041638, 2.08x the "
                "admission floor. Declared dt 20 s against a measured Courant "
                "limit of 32.09 s -- 1.60x of margin -- and 20 s divides the "
                "600 s radiation cadence thirty times. Finest spacing 4,457.23 m "
                "measured is above the 3 km convection threshold, so the "
                "3 km convection threshold leaves Grell-Freitas ON and the row binds "
                "the anchored (20 s, gf) configuration. The static carries real "
                "defc tables, so like the other generated rows this one drops "
                "them at attach. "
                "Evidence: tree/evidence/swath-real-cascade-20260826/."
            ),
        ),
        "v6.75.112676": MeshBinding(
            name="v6.75.112676",
            n_cells=112_676,
            n_edges=338_022,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=6_000.0,
            dt_seconds=20.0,
            grid_bytes=154_154_208,
            grid_sha256="36406384c93e9e8b2ee8b93bc9eacbdcb7b0db07cc609a6fdfb922a739a7d7b7",
            static_bytes=215_446_200,
            static_sha256="4cfa3008961c0503dfea23ed87f7ac6cf514d3c723e52330d60547b67c902f10",
            drop_carried_deformation=True,
            notes=(
                "Slot s01 of the first cycle this program ran with FOUR "
                "independent placed grids at once. The threat is an "
                "ATMOSPHERIC RIVER, not a cyclone: a moisture corridor in the "
                "central North Pacific at 14.23 N 140.89 W, ranked first of "
                "541 candidates in the 24 h u96.64002 forecast of 2026-08-12 "
                "06Z. Its row asks for 6 km, not the 4 km a cyclone row asks "
                "for, so the mesh is coarser than the placed cyclone rows and "
                "that is the threat row's own choice. "
                "Delivered 112,676 cells against the sizing pass's "
                "generator dry-run prediction of 112,675.1 -- 0.0008 %. "
                "Attained finest spacing 8.670 km against the sizing pass's "
                "inscribed-cap estimate of 7.941 km: the cap stand-in is "
                "9.2 % optimistic on a long thin corridor, where on a "
                "cyclone's compact ring it was 0.17 %. Coarsest 88,083.4 m, "
                "spacing ratio 10.16, coordination 5->427, 6->111,834, "
                "7->415 with nothing below five edges. GLOBAL: "
                "sum(areaCell)/4pi = 1.0. min(dvEdge/dcEdge) 0.040126, "
                "2.01x the 0.02 admission floor, shortest dual edge 647.5 m. "
                "Declared dt 20 s against a measured Courant limit of "
                "62.42 s -- 3.12x of margin -- and 20 s divides the 600 s "
                "radiation cadence thirty times. Finest spacing 8,670 m is "
                "far above the 3 km convection threshold, so Grell-Freitas "
                "stays on with no flag passed and the row binds the anchored "
                "(20 s, gf) configuration. The static carries real defc "
                "tables, so like every generated row it drops them at attach. "
                "Evidence: tree/evidence/four-swaths-20260827/."
            ),
        ),
        "v6.75.114257": MeshBinding(
            name="v6.75.114257",
            n_cells=114_257,
            n_edges=342_765,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=6_000.0,
            dt_seconds=20.0,
            grid_bytes=156_316_772,
            grid_sha256="4ed2a3a9e97878de32baa1ca580018a6f1fed804994211c2e6118292f91a7e42",
            static_bytes=218_469_072,
            static_sha256="2d2dcaf18113ebe3b0adaba72a90fc9c860eb2afc80ec1e6c3c954cc5b039c01",
            drop_carried_deformation=True,
            notes=(
                "Slot s03 of the same four-grid cycle: the SECOND atmospheric "
                "river, a separate corridor in the north-west Pacific at "
                "28.31 N 164.82 E, ranked third. Two grids on the same threat "
                "CLASS in one cycle is what the class budget permits and no "
                "more, and the two are 2,700 km apart on different corridors. "
                "Delivered 114,257 cells against a predicted 114,265.4 -- "
                "0.007 %. Attained finest spacing 7.936 km against the "
                "inscribed-cap estimate of 7.941 km, 0.07 % -- the same "
                "estimator that missed by 9.2 % on s01, which is a property "
                "of the polygon's shape and not of the estimator's accuracy "
                "in general. Coarsest 84,007.5 m, spacing ratio 10.59, "
                "coordination 5->470, 6->113,330, 7->456, 8->1 with nothing "
                "below five edges. GLOBAL: sum(areaCell)/4pi = 1.0. "
                "min(dvEdge/dcEdge) 0.042660, 2.13x the admission floor, "
                "shortest dual edge 655.1 m. Declared dt 20 s against a "
                "measured Courant limit of 57.14 s -- 2.86x of margin. "
                "Finest spacing is above the 3 km convection threshold, so "
                "the row binds the anchored (20 s, gf) configuration. "
                "Evidence: tree/evidence/four-swaths-20260827/."
            ),
        ),
        "v4.75.128019": MeshBinding(
            name="v4.75.128019",
            n_cells=128_019,
            n_edges=384_051,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=4_000.0,
            dt_seconds=20.0,
            grid_bytes=175_144_472,
            grid_sha256="550545393a2d196c48a591f6d305fd1a89d663726c96eaaad48084b833e1da8f",
            static_bytes=244_782_016,
            static_sha256="5fae3113f68403a357e54f48fc29071dab45e5af69df64dc84b1e6247f49ceba",
            drop_carried_deformation=True,
            notes=(
                "Slot s02 of the four-grid cycle, and the only WINTER STORM "
                "row this project has placed. Antarctic coast at 66.41 S "
                "159.52 E, ranked second of 541. Its threat row asks for "
                "4 km where the atmospheric-river rows ask for 6, so this is "
                "the finest of the four and the largest at 128,019 cells "
                "against a predicted 128,014.0 -- 0.004 %. Attained finest "
                "4.566 km, coarsest 95,470.0 m, spacing ratio 20.91, the "
                "widest of the cycle. Coordination 5->740, 6->126,551, "
                "7->728 with nothing below five edges. GLOBAL: "
                "sum(areaCell)/4pi = 1.0. min(dvEdge/dcEdge) 0.042668, "
                "2.13x the admission floor, shortest dual edge 342.7 m. "
                "Declared dt 20 s against a measured Courant limit of "
                "32.88 s -- 1.64x of margin, the tightest of the four and "
                "the closest any registered row sits to its own mesh's "
                "limit after v4.75.121182's 1.60x. "
                "ITS STATIC IS THE FIRST BUILT BY THE NEIGHBOUR-FILL PATH. "
                "The stock rw_mpas_static refused this mesh outright -- "
                "'albedo_modis mapped no valid pixels to cell 111276' -- "
                "because MODIS surface albedo is land-only and the Antarctic "
                "sea-ice margin is where the land-use archive calls a cell "
                "ice while every albedo pixel in it is fill. woof's static "
                "builder now carries a value to such a cell from the "
                "neighbours that have one; rebuilding v6.75.112676's static "
                "with the same binary reproduced it BYTE-IDENTICALLY, so no "
                "existing static moved. Evidence: "
                "tree/evidence/four-swaths-20260827/."
            ),
        ),
        "v6.75.120989": MeshBinding(
            name="v6.75.120989",
            n_cells=120_989,
            n_edges=362_961,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=6_000.0,
            dt_seconds=20.0,
            grid_bytes=165_527_000,
            grid_sha256="1427172bd3354f6c5a0a75c3bf79411756db1d24793fec82566bb5d076b80647",
            static_bytes=231_340_656,
            static_sha256="71d885f6138174624e8c4351d86ea8afc9b885132f03e919b2e425f6c85c604b",
            drop_carried_deformation=True,
            notes=(
                "Slot s04 of the four-grid cycle: the 944.6 hPa Southern "
                "Ocean EXTRATROPICAL CYCLONE at 60.09 S 139.46 E, the same "
                "storm v4.75.121182 was placed on under the previous "
                "rulebook. The two rows are a controlled pair -- same storm, "
                "same init, same 75 km background, same dt -- differing only "
                "in what the threat row asked for: 4 km there, 6 km here, "
                "because the extratropical-cyclone row's own spacing moved "
                "when the metrics document was rewritten. 120,989 cells "
                "against a predicted 120,999.5 -- 0.009 %. Attained finest "
                "6.034 km, coarsest 89,273.9 m, spacing ratio 14.79, "
                "coordination 5->754, 6->119,495, 7->738, 8->2 with nothing "
                "below five edges. GLOBAL: sum(areaCell)/4pi = 1.0. "
                "min(dvEdge/dcEdge) 0.040929, 2.05x the admission floor, "
                "shortest dual edge 432.2 m. Declared dt 20 s against a "
                "measured Courant limit of 43.45 s -- 2.17x of margin. "
                "Finest spacing is above the 3 km convection threshold, so "
                "the row binds the anchored (20 s, gf) configuration. "
                "Evidence: tree/evidence/four-swaths-20260827/."
            ),
        ),
        "r4.75.11020": MeshBinding(
            name="r4.75.11020",
            n_cells=11_020,
            n_edges=33_338,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=4_000.0,
            dt_seconds=20.0,
            grid_bytes=15_466_312,
            grid_sha256="140698dc4125440c9ca5b4dadcd3559a03a4715141ebfe9fd8f6a328b1c6f5f3",
            static_bytes=21_200_308,
            static_sha256="00262343feeffd81f6f724faddb9cb9f7a02fd7049621cbae442c6b9da4579a5",
            drop_carried_deformation=True,
            boundary_zone_width=7,
            bdy_mask_sha256=(
                "2baf091d718efcbbcb3f9385d55b0f224c1ea54fc3d75bfa8f108fc8a1fca158"
            ),
            lbc_source="u96.64002-parent-forecast/lbc-s01",
            notes=(
                "The FIRST limited-area row this program culled out of a mesh "
                "it placed for itself. The parent is v4.75.121182 -- the "
                "graded global mesh the placement layer emitted for the "
                "deepest low in a 24 h u96.64002 forecast -- and the region "
                "row is that same layer's own cull-region document, handed to "
                "rw_mpas_mesh --cull-parent unedited. 121,182 cells become "
                "11,020: 9,220 free interior plus seven boundary rings of "
                "240/245/251/257/263/269/275, so 92.4 percent of the parent "
                "was background the card no longer holds. Euler "
                "characteristic 1 (a bounded disk), zone width 7 measured, "
                "mesh-check passed on the grid alone. The core keeps the "
                "parent's spacing exactly, because a cull moves no cell "
                "centre: the rings grow OUTWARD from the requested polygon "
                "into the parent's transition ramp, so the 4 km-requested "
                "interior is untouched by the relaxation zone. Same declared "
                "dt as the parent, 20 s, so the two arms differ by domain and "
                "boundary treatment and by nothing else. lbc_source names the "
                "boundary series rw_mpas_lbc built from the SAME u96.64002 "
                "forecast that placed the swath, entered through the "
                "unstructured-port-stream driving-source row. "
                "Evidence: tree/evidence/swath-as-lam-20260827/."
            ),
        ),
        "r4.75.15755": MeshBinding(
            name="r4.75.15755",
            n_cells=15_755,
            n_edges=47_445,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=4_000.0,
            dt_seconds=20.0,
            grid_bytes=22_026_176,
            grid_sha256="9de51a8c020323ff2e2afd1a0fbe958c547753c2edd9c5d4fe90760ae5ac4b54",
            static_bytes=30_212_452,
            static_sha256="8f2091d754ecc35c41fc0e75830f3b08e6d63b414985bd5d134ce6a96e733a7d",
            drop_carried_deformation=True,
            boundary_zone_width=7,
            bdy_mask_sha256=(
                "1726f1a23b6140d21a2d96897939d7b9eda990b78d31579722cb6293f66533db"
            ),
            lbc_source="u96.64002-parent-forecast/lbc-s01-1.70",
            notes=(
                "The GENTLEST boundary interface this program has run, and it "
                "cost no extra mesh. Culling the same parent at 1.70x the "
                "placed region keeps the 4.6 km core exactly where it was -- a "
                "cull moves no cell centre -- and picks up the parent's own "
                "coarsening ramp on the way out, so the seven driven rings "
                "land on cells 15.89 to 25.27 km wide. Against the 71.0 km "
                "coarse parent that is about 3.4:1 at the interface, inside "
                "the 3:1 to 5:1 band a WRF nest would use, where the placed "
                "1.00 cull is about 7.6:1 and a 0.45 cull about 13.7:1. The "
                "ramp between the rings and the core IS the intermediate "
                "resolution level, in the form MPAS has rather than the form "
                "WRF has: a region of one mesh, not another forecast. 15,755 "
                "cells against the 1.00 cull's 11,020 -- 43 percent more -- "
                "and against the uncut parent's 121,182. Rings "
                "240/239/229/225/217/205/192: populations shrink outward "
                "because the cells are widening 59 percent across the zone, "
                "which is the mesh being variable resolution rather than torn. "
                "min(dcEdge) is 4,457.233 m, the parent's finest edge, so this "
                "row shares the 20 s anchor and its 1.605x Courant margin with "
                "every other cull of this parent. "
                "Evidence: tree/evidence/nest-ratio-20260827/."
            ),
        ),
        "r4.75.14050": MeshBinding(
            name="r4.75.14050",
            n_cells=14_050,
            n_edges=42_381,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=4_000.0,
            dt_seconds=20.0,
            grid_bytes=19_675_052,
            grid_sha256="59e4b1ef3b07471f045723d6c2a85ccf541da131f63beb3645c69371bb974827",
            static_bytes=26_974_320,
            static_sha256="c178405c0c33d610fe37bf7971b548328a063d33d94da2de3633c040968a3285",
            drop_carried_deformation=True,
            boundary_zone_width=7,
            bdy_mask_sha256=(
                "a8e66046452db881bb4a9da08952610207ee5aa2e0a58d48b1d2348b48f84088"
            ),
            lbc_source="u96.64002-parent-forecast/lbc-s01-1.35",
            notes=(
                "The intermediate step of the domain-size ladder: the same "
                "parent culled at 1.35x the placed region, 14,050 cells, "
                "driven rings on cells 10.86 to 14.30 km wide -- about 5.7:1 "
                "against the 71.0 km coarse parent, between the placed cull's "
                "7.6:1 and the 1.70 cull's 3.4:1. It exists so the ladder has "
                "a middle point and the trend in interior agreement can be "
                "read as a curve rather than as two ends. Rings "
                "297/298/293/285/270/264/249. Same 20 s anchor at 1.605x "
                "Courant margin as every other cull of this parent. "
                "Evidence: tree/evidence/nest-ratio-20260827/."
            ),
        ),
        "r4.75.7975": MeshBinding(
            name="r4.75.7975",
            n_cells=7_975,
            n_edges=24_151,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=4_000.0,
            dt_seconds=20.0,
            grid_bytes=11_216_472,
            grid_sha256="8d2afd4aa7aa60309e1da1069bcefb20c5acf49ff1cf394e6739d13d904070f5",
            static_bytes=15_356_780,
            static_sha256="f1e1db3387af02cd135b84a9b326a0f6ca79c1cac4ddad0d022cfba1cc29382f",
            drop_carried_deformation=True,
            boundary_zone_width=7,
            bdy_mask_sha256=(
                "401c8c160948dadffef45cf75632970d73e8491b1799ea1639a3da794e76e29f"
            ),
            lbc_source="u96.64002-parent-forecast/lbc-s01-0.70",
            notes=(
                "The MIDDLE domain of the nest-ratio measurement, and it "
                "exists to answer one question: does a limited-area interior "
                "change when its boundary moves further away? Same parent as "
                "r4.75.11020 (v4.75.121182), same binary, same run, same "
                "cell centres -- a cull moves none -- and the SAME requested "
                "region scaled to 0.70 of its centroid-to-vertex arcs, so "
                "this domain and the 1.00 one are similar shapes over "
                "concentric ground rather than two different pieces of "
                "atmosphere. 6,269 free interior cells in seven rings of "
                "254/251/249/242/241/237/232. Those populations SHRINK "
                "outward and that is correct: the rings grow into the "
                "parent's coarsening ramp, 6.31 to 7.71 km, so the shell's "
                "perimeter grows 11.6 percent while its population falls 8.7. "
                "min(dcEdge) is 4,457.233 m -- the parent's own finest edge, "
                "identical across all three domains -- so the 20 s anchor "
                "sits at 1.605x margin under a 32.092 s Courant limit, the "
                "same margin the 11,020-cell row got. "
                "Evidence: tree/evidence/nest-ratio-20260827/."
            ),
        ),
        "r3.75.120.35842": MeshBinding(
            name="r3.75.120.35842",
            n_cells=35_842,
            n_edges=108_249,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=3750.0,
            dt_seconds=20.0,
            grid_bytes=50_226_220,
            grid_sha256="548143032f166b85e38862e9cb1a8c814895be38252d038cc2a49b54ac24dca6",
            static_bytes=73_181_844,
            static_sha256="88a08a8257b4ef2c0962755b9b92351c227138f76948adb6aa237c6365e726d9",
            drop_carried_deformation=True,
            boundary_zone_width=7,
            bdy_mask_sha256=(
                "62d1b5ef8626628ddecd87955132dc64fb742a68420a3b90956bbf9bd59d1d3d"
            ),
            lbc_source="gfs-2026-08-12-06z-wps-intermediate/lbc-hourly",
            notes=(
                "Limited-area cull of a generated graded parent (3.75 km cap of 480 km radius at 42.0 N 86.0 W on a 120 km background, 120,941 cells) around a 600 x 600 km box over the Lower Great Lakes convection of the GFS 2026-08-12 06Z cycle. Init is the parent's own woof hex init culled; the boundary series is rw_mpas_lbc on the wps-intermediate route from the same GFS cycle, hourly 06-12Z. Staged for a forecast, not yet run: no regional anchor of this class exists and no forecast has been integrated on it."
            ),
        ),
        "r4.75.4440": MeshBinding(
            name="r4.75.4440",
            n_cells=4_440,
            n_edges=13_540,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=4_000.0,
            dt_seconds=20.0,
            grid_bytes=6_293_136,
            grid_sha256="d91d2a4ce8bd94ec0ae4b566174344f5ee3edbd35fbb0d85ac68e347c4a13525",
            static_bytes=8_595_292,
            static_sha256="f45d339426d99934f402a8363b39689098d5c6d1d3fa32fe55d46daa7b16dc91",
            drop_carried_deformation=True,
            boundary_zone_width=7,
            bdy_mask_sha256=(
                "e2d18b34c866e2768896467f21ec11d908c00425ed36409e100d25827ed6b68e"
            ),
            lbc_source="u96.64002-parent-forecast/lbc-s01-0.45",
            notes=(
                "The SMALLEST domain of the nest-ratio measurement, and the "
                "one that carries the claim. Its free interior is the patch "
                "all three domains are compared over, so it is the arm whose "
                "boundary is nearest the ground under test: 2,937 free cells "
                "whose furthest point is about 135 km from a driven ring, "
                "against 300 km for the 1.00 domain. It is also the HARSHEST "
                "interface of the three, because at 0.45 the rings still sit "
                "inside the parent's 4.6 km core (5.03 to 5.49 km across the "
                "seven) instead of out in the ramp where the wider domain's "
                "rings land -- so a 71 km parent state is imposed on cells "
                "about fourteen times finer. If THIS interior tracks the "
                "wider domains, a coarser interface further away cannot be "
                "starving anything. Rings 207/209/212/215/218/217/225; the "
                "single dip at ring 6 is one cell of hexagonal tiling against "
                "a curved cut, not a tear -- every ring cell touches the ring "
                "inside it. Same 20 s anchor at 1.605x Courant margin. "
                "Evidence: tree/evidence/nest-ratio-20260827/."
            ),
        ),
        "u96.64002": MeshBinding(
            name="u96.64002",
            n_cells=64_002,
            n_edges=192_000,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=96_000.0,
            dt_seconds=120.0,
            grid_bytes=87_560_436,
            grid_sha256="57f4965a81d25dbc16b4fcbdb06474ca1a4b39adf406a58b04852be72f93f305",
            static_bytes=122_381_512,
            static_sha256="005cb9e7363283ec98a0cb027956d0245978a3a9a7c12cb3032d31e815561e27",
            drop_carried_deformation=True,
            notes=(
                "Generated uniform 96 km mesh, and the first generated mesh to "
                "complete a full-physics forecast. Both files are ours: the grid "
                "from rw_mpas_mesh seeded on the icosahedral Goldberg subdivision "
                "GP(80,0) -- 64,002 cells exactly, snap 0.0 percent -- and the "
                "static from the unified rw_mpas_static against that same grid. "
                "The seed is what separates it from v15.150.38857: coordination is "
                "exactly 12 pentagons and no heptagons, so no dislocation quad "
                "exists for a collapsed dual edge to form around, and "
                "min(dvEdge/dcEdge) measures 0.394671 -- the published x1.40962's "
                "own class. Declared dt 120 s against a measured Courant limit of "
                "511.4 s (min dcEdge 71,031.6 m). The static carries real defc "
                "tables, so like the other non-native rows this one drops them at "
                "attach."
            ),
        ),
        "v0.9.120.110533": MeshBinding(
            name="v0.9.120.110533",
            n_cells=110_533,
            n_edges=331_593,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=937.5,
            dt_seconds=5.0,
            grid_bytes=151_228_504,
            grid_sha256="39f0c8ae2b4b0373cf3889dec6837d0f0a842becbb75e157fe2c1a426423cbc6",
            static_bytes=224_613_208,
            static_sha256="ff495ce89104629a73b5921d8174408fe4cd099a75b656f0d7d06acb463f8c4b",
            drop_carried_deformation=True,
            notes=(
                "THE FIRST SUB-KILOMETRE ROW THIS PROGRAM HAS REGISTERED. A "
                "937.5 m refined core of 100 km radius over central Oklahoma "
                "(35.0 N 97.0 W) on a 120 km global background, reached in "
                "seven nested cap rows of ONE spec -- each halving the "
                "spacing, each ramp eighteen times its own target -- so the "
                "gradient the surgery-locality gate reads is 10.05 % per "
                "cell against its 12.25 % ceiling. Delivered 110,533 cells "
                "against a predicted 110,437.7, 0.086 %. Attained finest "
                "946.99 m, coarsest 124,804.8 m, spacing ratio 131.79, the "
                "widest any registered row carries. Coordination 5->1,835, "
                "6->106,876, 7->1,821, 8->1, nothing below five edges. "
                "GLOBAL: sum(areaCell)/4pi = 1.0 exactly, Euler 2, "
                "orthogonality 1.265e-11 against a 1e-10 emission limit, "
                "TRiSK antisymmetry 3.886e-16, area decomposition 2.69 eps "
                "of 32. min(dvEdge/dcEdge) 0.040048, 2.00x the admission "
                "floor; SHORTEST DUAL EDGE 58.879 m, which the binary32 "
                "200 m floor refuses and which this grid's own "
                "binary64_earth_centred storage admits at 3.725e-7 m -- the "
                "2026-08-29 dual-edge unlock is what makes the row "
                "expressible at all. "
                "Declared dt 5 s against a measured Courant limit of "
                "6.259 s (min dcEdge 869.25 m) -- 1.252x of margin, the "
                "tightest any registered row sits at, and the FIRST row "
                "where 5 s is the mesh's natural step rather than 140x "
                "below its limit. Convection is off here by the 2026-08-26 "
                "sub-3-km default with no flag passed, so the row binds the "
                "anchored (5 s, convection off) configuration, whose own "
                "anchor named this measurement as the one that would "
                "settle its confound. "
                "ITS STATIC IS THE FIRST BUILT AT A DERIVED CATEGORICAL "
                "SAMPLING RATE. The stock builder sampled every geography "
                "dataset once per source pixel and refused this mesh -- "
                "'modis_landuse_20class_30s_with_lakes mapped no valid "
                "category pixels to cell 49103' at 34.79 N 96.84 W -- "
                "because a 30 arc-second pixel is 926 m and this mesh's "
                "cells are 947. The refusal is unchanged; the rate is now "
                "derived per categorical dataset from the mesh's own "
                "min(dcEdge) -- ceil(1310.43 / min(dcEdge)), the pixel "
                "diagonal over the smallest cell's inradius -- and reads 2 "
                "here and 1 for every mesh with min(dcEdge) at or above "
                "1,310.43 m. The finest REGISTERED mesh before this row has "
                "min(dcEdge) 4,302.90 m, so x1.40962's and x4.163842's statics rebuild "
                "BYTE-IDENTICALLY and no registered static moved. "
                "Evidence: woof evidence/fine-mesh-20260829/. "
                "RE-PINNED 2026-09-01 by the 0.2.2 re-pin campaign: grid and "
                "static rebuilt from the same spec through the PUBLISHED "
                "2.6.1 artifacts (rw_mpas_mesh/rw_mpas_static from the "
                "v2.6.1 bridge bundle, geography from woof fetch-geog). "
                "Both digests moved off the 2026-08-29 build's "
                "(grid 40e9e9f7... -> 39f0c8ae..., static 6e941739... -> "
                "ff495ce8...), and the movement is MEASURED rather than "
                "assumed. Determinism: two builds from the same spec and "
                "the same published binary emitted byte-identical grids. "
                "Attribution, old grid against new, variable by variable: "
                "the same mesh to roundoff. The seven index and topology "
                "arrays (indexToCellID, indexToEdgeID, indexToVertexID, "
                "nEdgesOnCell, nEdgesOnEdge, densityFunctionCode, "
                "nominalMinDc) are byte-equal; every old cell centre has a "
                "new counterpart within 3.1 mm (median 0.7 um, one-to-one "
                "over all 110,533); 26 cells, 78 edges and 40 vertices "
                "swapped index on near-tie ordering (193 cellsOnCell rows "
                "follow them) and every other coordinate sits in place to "
                "under 6e-10 rad; none of the embedded mesh receipt's 235 "
                "integer keys changed (the same insertion, anneal and "
                "surgery sequence), its float keys moved at roundoff or "
                "in the extremal statistics that amplify it, its "
                "steepest-gradient reading moved 10.05 -> 10.97 % per "
                "cell because the 0.2.1 repaired meter reads the same "
                "spec (still admitted against 12.25), and "
                "the published receipt gains thirteen fields "
                "(graded_levels[].insert_front_spacing_km, "
                "gradient_probe_coverage, gradient_probe_points, "
                "mesh.well_centredness.*), which is the 536 extra bytes. "
                "So the published generator relaxes the same mesh with "
                "roundoff-level arithmetic differences from the 08-29 "
                "engine build, and the static moved with its input. The "
                "derived categorical supersample reads 2 on all six "
                "categorical passes in the static's own receipt. Evidence: "
                "evidence/repin-261-20260901/subkm/."
            ),
        ),
        "u60.163632": MeshBinding(
            name="u60.163632",
            n_cells=163_632,
            n_edges=490_890,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=60_000.0,
            dt_seconds=120.0,
            grid_bytes=223_857_676,
            grid_sha256="3ffcd78e03f85d2e8fd5d56792d173b172b27cb5d8e32f04b2a1f025d90d38a4",
            static_bytes=332_510_380,
            static_sha256="b820e95c647f5646ab8d630e3dd4c99c2392d90ef7714906190bc9fcfeedd476",
            drop_carried_deformation=True,
            notes=(
                "HEX-SPEED 2026-09-02: generated uniform 60 km mesh, the 60 km rung of the "
                "global speed ladder (the published x1.163842 is 163,842 cells; this is the "
                "generator's own 60 km sizing, rw_mpas_mesh --background-km 60 from the v2.6.3 "
                "bridge bundle, 26.6 s on one machine's CPU). 163,632 cells, 490,890 edges, "
                "dcEdge min 41,666.3 m, median 60,721.9 m, max 65,332.8 m; the static from "
                "rw_mpas_static against that grid through the default geography ladder. Declared "
                "dt 120 s against the door's Courant limit of 300.0 s (2.50x); the 120 s "
                "Grell-Freitas anchor is a property of the timestep. Registered to run one "
                "GFS-initialised 6 h forecast per card; nothing about its physics is proven "
                "beyond that run's own receipt. Evidence: evidence/hex-speed-20260902/."
            ),
        ),
        "r0.9.120.40520": MeshBinding(
            name="r0.9.120.40520",
            n_cells=40_520,
            n_edges=121_848,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=937.5,
            dt_seconds=5.0,
            grid_bytes=56_551_320,
            grid_sha256="348592df56e53dfeb8dd1efa7cde38f1e4373f008b8c7ba0b543a11290a013eb",
            static_bytes=82_483_960,
            static_sha256="0ba6dfbb79ca82b8771f23390e18acb890fdc7a1b6b1000a861fdc1626a197ed",
            drop_carried_deformation=True,
            boundary_zone_width=7,
            bdy_mask_sha256=(
                "e52578a9bcffc1a019918f6e9fd01f6ac7bf54914dc916857860b539268e35b5"
            ),
            lbc_source="gfs-2026-08-28-18z-wps-intermediate/lbc-hourly",
            notes=(
                "HEX-SPEED 2026-09-02: a 125 km cap at 35.0 N 97.0 W culled from the registered v0.9.120.110533 parent (937.5 m core of 100 km radius on a 120 km background, the same bytes the 0.2.2 row pins) "
                "with woof hex cull (rw_mpas_mesh --cull-parent from the v2.6.3 bridge bundle), "
                "so the child's terrain, vertical grid and initial state ARE the parent's cell "
                "for cell. 40,520 cells: the whole 937.5 m core plus the first 25 km of the 1.875 km ramp, where the seven boundary rings sit. "
                "Rings 38,052/399/387/366/351/333/326/306 (zone width 7); "
                "dcEdge min 869.251 m, median 976.9 m, max 4324.6 m (the outermost rings reach into the "
                "ramp); declared dt 5 s against the door's Courant limit of 6.259 s "
                "(1.252x). Convection is off by the sub-3-km default with no flag "
                "passed. Init is the parent's own woof hex init (GFS 2026-08-28 18Z analysis, native-free "
                "tc55-v1 vertical) culled; the boundary series is rw_mpas_lbc on the wps-intermediate route "
                "from the same GFS cycle, hourly f000-f006 (18Z-00Z). Sibling of r0.8.102.56619 on the same cap. "
                "Evidence: evidence/hex-speed-20260902/."
            ),
        ),
        "r0.8.102.56619": MeshBinding(
            name="r0.8.102.56619",
            n_cells=56_619,
            n_edges=170_231,
            n_levels=55,
            n_interfaces=56,
            n_soil_levels=4,
            nominal_dx_m=800.0,
            dt_seconds=5.0,
            grid_bytes=78_998_752,
            grid_sha256="f09d5b1ac5f8cc9a0af782dd8817b34b4cc075b04365c0782cfc94bc886e5d93",
            static_bytes=115_237_376,
            static_sha256="533492c42e600324902797de817d234ba165ccd833378c003ab4eed3fba9375d",
            drop_carried_deformation=True,
            boundary_zone_width=7,
            bdy_mask_sha256=(
                "de0c858251b36b8592c8416f2dba8876127035229d4f3d88184831732c10b996"
            ),
            lbc_source="gfs-2026-08-28-18z-wps-intermediate/lbc-hourly",
            notes=(
                "HEX-SPEED 2026-09-02: a 125 km cap at 35.0 N 97.0 W culled from an 800 m parent generated for this measurement: the registered 937.5 m ladder with every spacing, transition and the 102.4 km background scaled by 0.8/0.9375 and the cap radii kept (155,138 cells, 147 s on one machine's CPU) "
                "with woof hex cull (rw_mpas_mesh --cull-parent from the v2.6.3 bridge bundle), "
                "so the child's terrain, vertical grid and initial state ARE the parent's cell "
                "for cell. 56,619 cells: the whole 800 m core plus the first 25 km of the 1.6 km ramp, where the seven boundary rings sit. THE FINEST REGISTERED ROW: min dcEdge 710.869 m. "
                "Rings 53,412/508/497/477/463/448/422/392 (zone width 7); "
                "dcEdge min 710.869 m, median 831.3 m, max 3394.1 m (the outermost rings reach into the "
                "ramp); declared dt 5 s against the door's Courant limit of 5.118 s "
                "(1.024x). Convection is off by the sub-3-km default with no flag "
                "passed. Init is the parent's own woof hex init (GFS 2026-08-28 18Z analysis, native-free "
                "tc55-v1 vertical) culled; the boundary series is rw_mpas_lbc on the wps-intermediate route "
                "from the same GFS cycle, hourly f000-f006 (18Z-00Z). Sibling of r0.9.120.40520 on the same cap. "
                "Evidence: evidence/hex-speed-20260902/."
            ),
        ),
        "conus12km": MeshBinding(
            name="conus12km",
            n_cells=157_859,
            n_edges=475_138,
            n_levels=59,
            n_interfaces=60,
            n_soil_levels=4,
            nominal_dx_m=12_000.0,
            dt_seconds=60.0,
            grid_bytes=167_235_640,
            grid_sha256="245c511b0ba39d9df74ff5d611634e481646ad003bff8ea6e36f081d5b483c56",
            static_bytes=275_923_012,
            static_sha256="55b7cb11ec6cad7418e47d86d9ade248b2a6d84ba0b71bcc798e4389cda89458",
            boundary_zone_width=7,
            bdy_mask_sha256="0620f99c2618d4bbc000d53225dc2826155aab53fc967d4dd68991f2b6ccd1e2",
            lbc_source="gfs-hourly-wps-intermediate",
            # RRFS v2's L60 invariant for GFS-driven runs (source profile
            # rrfs-v2, mesh_set.invariant_rule): 957,584,940 bytes, SHA-256
            # cabfea7d4136f0eb5f1314039e6e6a1c1c6633fc6dfb6d49957c70d46412d17f.
            invariant_file="conus12km.invariant.nc_L60_GFS",
            notes=(
                "Public 12 km limited-area mesh and static data from the "
                "20251017 set. The atmosphere has 59 layers and 60 interfaces; "
                "the column-physics soil ABI retains four Noah layers. "
                "All geometric and timestep admissions are measured at bind. "
                "The 60 s timestep requires a measured anchor or a recorded mint. "
                "This row supplies no regional forecast contract or skill claim."
            ),
        ),
        "conus3km": MeshBinding(
            name="conus3km",
            n_cells=2_341_818,
            n_edges=7_031_657,
            n_levels=59,
            n_interfaces=60,
            n_soil_levels=4,
            nominal_dx_m=3_000.0,
            dt_seconds=20.0,
            grid_bytes=2_587_577_704,
            grid_sha256="773ffde88ef8c21b356523c2aa0ad2461c32ac75e1692d3e7c9e41c1791ac444",
            static_bytes=4_086_429_764,
            static_sha256="41f93c75ba2d601ddfd8b5f96a9de2b47d11fd3f8f0cd4cb8782a3afd1870a6c",
            boundary_zone_width=7,
            bdy_mask_sha256="a5e29f2deb8396d1601e6198a4ab58c4ea6b4c7ba84db21c1178d72255a14c09",
            lbc_source="gfs-hourly-wps-intermediate",
            # RRFS v2's L60 invariant for GFS-driven runs (source profile
            # rrfs-v2, mesh_set.invariant_rule; 14,164,674,656 bytes there).
            invariant_file="conus3km.invariant.nc_L60_GFS",
            # RETIRED 2026-10-10: courant_safety_factor=0.95, the per-row
            # loosening of the old 0.90 outer-step policy that had refused
            # RRFS v2's own 20 s step here (rrfs-3km-proof, 2026-10-05).  The
            # stability check is now physical (woof.hex.timestep_admission):
            # 20 s with 4 sub-steps measures 0.942 against the 1.160
            # advective limit and 0.23 against the 0.816 acoustic one.
            notes=(
                "Public 3 km limited-area mesh and static data from the "
                "20251017 set (RRFS v2's conus3km). The atmosphere has 59 layers "
                "and 60 interfaces from the L60 vertical file; the column-physics "
                "soil ABI retains four Noah layers. The row's step is RRFS v2's "
                "20 s (--dt row): min(dcEdge) is 2,652.95 m, an advective "
                "Courant number of 0.942 at 125 m/s against the 1.160 limit. "
                "Hashes, sizes and the "
                "boundary-mask digest were read from the published files on "
                "2026-10-05. This row supplies no regional forecast contract or "
                "skill claim."
            ),
        ),
    }
)

# An anchor-mint run may override ONE row's declared timestep while it is in
# flight (woof.hex.mesh_row_candidate), because an anchor is a property of the
# TIMESTEP and the cheap place to earn one is the smallest registered mesh --
# whose Courant limit is an upper bound every candidate sits far beneath.  This
# returns the mapping above UNCHANGED on every ordinary run: an override exists
# only inside a live dt_admission.candidate_mint, under the same authorization
# sentence, and the overridden row stamps itself CANDIDATE-UNANCHORED-DT in the
# bind log and the run receipt.  Applied here, where the mapping is built, so
# the forecast door's by-path re-execution of this file picks it up as the
# first copy did.
MESH_BINDINGS = mesh_row_candidate.apply_overrides(MESH_BINDINGS)

# A mesh GENERATED AT A POINT -- ``woof hex mesh-plan --point --generate``
# -- cannot be a row in this file either: the point is arbitrary and the
# mesh did not exist when this file was written.  ``woof.hex.mesh_rows``
# adds the generated global parent and its cull from the row file the
# door leaves beside the mesh (``$WOOF_HEX_MESH_ROWS``), and returns the
# mapping UNCHANGED on every ordinary run.  Applied BEFORE the cascade rows
# so a cascade may cull a generated parent, and every admission behind such
# a row is a measurement of the files it names at bind.
MESH_BINDINGS = mesh_rows.apply_rows(MESH_BINDINGS, MeshBinding)

# A CYCLING CASCADE cuts a new limited-area mesh every cycle -- a
# storm-following swath re-places itself -- so its culls cannot be rows in
# this file: they did not exist when it was written.  ``woof.hex.cascade_row``
# adds them from the cull receipt that made them, and returns the mapping
# UNCHANGED on every ordinary run.  Every admission behind such a row is a
# MEASUREMENT of the files it names rather than a declaration about them, and
# the bind below re-hashes both files regardless, so a cascade row that lies
# about its bytes refuses exactly as a hand-written one would.  What a cascade
# row does not have is a person who read it; what stands in for that is the
# per-geometry contract deck the cascade runs before the forecast.
MESH_BINDINGS = cascade_row.apply_rows(MESH_BINDINGS, MeshBinding)

NATIVE_MESH_NAME = "x4.163842"

# The device-memory floors come from the ONE admission surface
# (woof.hex.device_admission), re-fitted at the merged tip 2026-08-26 under
# the 2026-08-26 re-derivation; the linear ``24 GiB * cells / 163,842`` scaling
# they replace both refused meshes the measured row says fit (large cell
# counts) and admitted runs that die inside a CuPy allocation mid-run
# (x1.40962 at 6,144 MiB free against a measured 8,874 MiB peak) -- the two
# concrete breakages this floor exists to prevent.
# The restart worker executes the same device stack the run does, so the same
# measured requirement governs it; the retired 22-vs-24 GiB split encoded
# margin between two asserted constants, not a measured difference.
NATIVE_DEVICE_FLOOR = device_admission.native_device_floor_bytes()
NATIVE_RESTART_FLOOR = NATIVE_DEVICE_FLOOR

_FINGERPRINT_SCALARS = (
    "N_CELLS",
    "N_EDGES",
    "N_LEVELS",
    "N_INTERFACES",
    "N_SOIL_LEVELS",
    "DT_SECONDS",
    "NOMINAL_DX_M",
    "MIN_FREE_DEVICE_BYTES",
    "RESTART_WORKER_MIN_FREE_DEVICE_BYTES",
)
_FINGERPRINT_PINS = (
    "INIT_RECONSTRUCTION_COEFFICIENTS_PIN",
    "INIT_EDGE_NORMAL_VECTORS_PIN",
    "PHYSICS_GEOMETRY_CARRIER_PIN",
    "LANDMASK_CONSTRUCTOR_CAST_PIN",
    "AUTHORITY_PINS",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return repr(value.item())
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def constants_fingerprint(
    proof: Any,
    *,
    edge_authority_sha256: str | None = None,
    edge_minimum_m: float | None = None,
) -> dict[str, Any]:
    """Fingerprint every shape/timestep-sensitive value plus real-edge authority."""

    body: dict[str, Any] = {}
    for name in _FINGERPRINT_SCALARS:
        body[name] = _plain(getattr(proof, name, None))
    for name in _FINGERPRINT_PINS:
        body[name] = _plain(getattr(proof, name, None))
    body["EDGE_LENGTH_AUTHORITY_SHA256"] = edge_authority_sha256
    body["EDGE_LENGTH_MINIMUM_M"] = (
        None if edge_minimum_m is None else repr(float(edge_minimum_m))
    )
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"fields": body, "sha256": hashlib.sha256(blob).hexdigest()}


def _static_edge_authority(static_path: Path) -> EdgeLengthAuthority:
    """Build the physical edge-length authority through its public constructor.

    Reading ``dcEdge`` here rather than accepting a previously built mapping is
    deliberate: a hand-assembled dict can carry a minimum nothing measured.
    """

    import netCDF4

    with netCDF4.Dataset(str(static_path)) as dataset:
        raw_dc = dataset.variables.get("dcEdge")
        if raw_dc is None:
            raise MeshBindingMismatch(
                f"{static_path}: static carries no physical dcEdge; there is no "
                "length authority to admit a timestep or to fingerprint against"
            )
        dc_edge = np.asarray(raw_dc[:])
    try:
        return edge_length_authority(dc_edge, source="static.dcEdge")
    except TimestepAdmissionError as error:
        raise MeshBindingMismatch(f"{static_path}: {error}") from error


def _grid_cell_spacing(grid_path: Path, len_disp_m: float) -> Any:
    """Each cell's scheme spacing, read from the grid's own ``meshDensity``.

    ``len_disp`` is the row's nominal spacing, the value the forecast driver
    hands :func:`woof.hex.cuda_gwdo_v841.native_cell_dx_m` for the column
    physics, so the convection decision classifies the same per-cell length
    the scheme receives.  A grid without ``meshDensity`` is uniform by
    construction and returns ``None`` (the whole-mesh decision).
    """

    import netCDF4

    with netCDF4.Dataset(str(grid_path)) as dataset:
        raw = dataset.variables.get("meshDensity")
        if raw is None:
            return None
        density = np.asarray(raw[:])
    try:
        return convection_admission.cell_spacing_m(density, len_disp_m)
    except convection_admission.ConvectionAdmissionError as error:
        raise MeshBindingMismatch(f"{grid_path}: {error}") from error


def _grid_cell_coordination(grid_path: Path) -> Any:
    """Read how many edges each cell has, from the file topology binds against.

    Read here rather than taken from the inspection mapping, for the reason
    :func:`_static_dual_edges` is: a mapping assembled earlier can carry a
    count nothing measured.
    """

    import netCDF4

    with netCDF4.Dataset(str(grid_path)) as dataset:
        raw = dataset.variables.get("nEdgesOnCell")
        if raw is None:
            raise MeshBindingMismatch(
                f"{grid_path}: grid carries no nEdgesOnCell; cell coordination "
                "cannot be measured and the mesh cannot bind topology"
            )
        return np.asarray(raw[:], dtype=np.int64)


def _static_dual_edges(static_path: Path) -> tuple[Any, Any, Any]:
    """Read the DUAL edge lengths the TRiSK tangential terms divide by.

    Read from the same file the timestep authority reads, for the same reason:
    a mapping assembled earlier can carry a minimum nothing measured.
    """

    import netCDF4

    with netCDF4.Dataset(str(static_path)) as dataset:
        missing = [
            name
            for name in ("dvEdge", "dcEdge")
            if dataset.variables.get(name) is None
        ]
        if missing:
            raise MeshBindingMismatch(
                f"{static_path}: static carries no physical {', '.join(missing)}; "
                "the dual-edge amplification the TRiSK operators inherit cannot be measured"
            )
        dv_edge = np.asarray(dataset.variables["dvEdge"][:])
        dc_edge = np.asarray(dataset.variables["dcEdge"][:])
        raw_cells = dataset.variables.get("cellsOnEdge")
        cells_on_edge = None if raw_cells is None else np.asarray(raw_cells[:])
    return dv_edge, dc_edge, cells_on_edge


def _fingerprint_with_authority(
    proof: Any, authority: EdgeLengthAuthority
) -> dict[str, Any]:
    """The single site that supplies the fingerprint's authority keywords.

    Both sides of every fingerprint comparison route through here, so a keyword
    added to :func:`constants_fingerprint` cannot reach one side only.
    """

    return constants_fingerprint(
        proof,
        edge_authority_sha256=authority.raw_sha256,
        edge_minimum_m=authority.minimum_m,
    )


def binding_fingerprint(proof: Any, static: Path) -> dict[str, Any]:
    """The digest a bind against ``static`` produces, computed as ``bind_mesh`` does.

    Any baseline compared against a bind receipt MUST come from here.  A bare
    :func:`constants_fingerprint` digests ``EDGE_LENGTH_AUTHORITY_SHA256=None``
    and can never equal a bound digest, which would make the x4 frozen no-op
    report a moved fingerprint on a run that rebound nothing.
    """

    return _fingerprint_with_authority(proof, _static_edge_authority(Path(static)))


def _sha256_file(path: Path) -> str:
    # The file's real SHA-256, computed once per file version
    # (woof.hex.file_digest): the bind re-read the 2 GB grid and 3 GB static
    # of the 20 km mesh on every forecast, 3.9 s of the door (rented box, 2026-10-13).
    from woof.hex.file_digest import sha256_file as file_version_sha256

    return file_version_sha256(path)


#: The regional mask triple, in the fixed digest order.
_REGIONAL_MASK_NAMES = ("bdyMaskCell", "bdyMaskEdge", "bdyMaskVertex")


def _mask_triple_digest(masks: Mapping[str, Any]) -> str:
    """The one digest definition lives with the mesh contract; delegate."""

    from woof.hex.mesh import regional_boundary_mask_digest

    return regional_boundary_mask_digest(masks)


def regional_mask_digest(grid_path: Path) -> str:
    """The digest a regional registry row pins its boundary rings by."""

    import netCDF4

    with netCDF4.Dataset(str(grid_path)) as dataset:
        missing = [
            name
            for name in _REGIONAL_MASK_NAMES
            if dataset.variables.get(name) is None
        ]
        if missing:
            raise MeshBindingMismatch(
                f"{grid_path}: no regional mask digest exists -- the grid lacks "
                f"{', '.join(missing)}, so it is not a regional cull"
            )
        masks = {
            name: np.asarray(dataset.variables[name][:])
            for name in _REGIONAL_MASK_NAMES
        }
    return _mask_triple_digest(masks)


def admit_regional_row(binding: MeshBinding, grid_observed: Mapping[str, Any]) -> dict[str, Any]:
    """Cross-examine the regional row fields against the inspected grid.

    Runs for EVERY row before any constant is rebound.  Each refusal names
    the concrete breakage: a regional cull bound as global would integrate an
    unforced boundary, and a regional row without a boundary source has
    nothing to force its rings with.
    """

    observed = grid_observed.get("regional_masks") or {}
    present = bool(observed.get("present"))
    if binding.regional and (
        binding.boundary_zone_width is None or binding.bdy_mask_sha256 is None
    ):
        raise MeshBindingMismatch(
            f"mesh {binding.name!r}: regional row fields are all-or-nothing -- "
            "boundary_zone_width and bdy_mask_sha256 must both be declared; a "
            "half-declared zone can be neither verified nor staged"
        )
    if not binding.regional:
        if present:
            raise MeshBindingMismatch(
                f"mesh {binding.name!r}: the grid carries the "
                "bdyMaskCell/Edge/Vertex triple (a regional cull) but the "
                "registry row declares no boundary zone.  Bound as global, the "
                "run would treat ring-7 absent-neighbour slots as real cells "
                "and integrate an unforced boundary inward.  Register the row "
                "with boundary_zone_width, bdy_mask_sha256 and an lbc_source"
            )
        return {"regional": False}
    if not present:
        raise MeshBindingMismatch(
            f"mesh {binding.name!r}: the registry row declares a "
            f"{binding.boundary_zone_width}-ring boundary zone but the grid "
            "carries no bdyMask triple; the supplied file is not the "
            "registered regional cull"
        )
    observed_width = int(observed.get("zone_width", -1))
    if observed_width != int(binding.boundary_zone_width):
        raise MeshBindingMismatch(
            f"mesh {binding.name!r}: the grid's outermost ring is "
            f"{observed_width}, the registry declares "
            f"{binding.boundary_zone_width}; zone-staged tendencies are sized "
            "to the declared width, so they would force rings that do not "
            "exist or skip rings that do"
        )
    observed_digest = str(observed.get("sha256", ""))
    if observed_digest != binding.bdy_mask_sha256:
        raise MeshBindingMismatch(
            f"mesh {binding.name!r}: bdyMask triple SHA-256 {observed_digest} "
            f"!= declared {binding.bdy_mask_sha256}; the boundary rings are "
            "not the ones this row admitted, so the specified and relaxation "
            "zones would follow the wrong cells"
        )
    if binding.lbc_source is None:
        raise MeshBindingMismatch(
            f"mesh {binding.name!r} is regional and its lbc_source slot is "
            "empty: no lateral boundary stream is registered to force the "
            f"{binding.boundary_zone_width}-ring zone, and the frozen dycore "
            "carries no regional branch, so a run would integrate an unforced "
            "boundary.  The row stays registrable; execution waits until a "
            "boundary source fills the slot"
        )
    return {
        "regional": True,
        "boundary_zone_width": int(binding.boundary_zone_width),
        "bdy_mask_sha256": observed_digest,
        "lbc_source": binding.lbc_source,
    }


def _require_file(role: str, path: Path, want_bytes: int, want_sha: str, mesh: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise MeshBindingMismatch(f"mesh {mesh!r}: {role} authority is missing: {resolved}")
    size = resolved.stat().st_size
    if size != want_bytes:
        raise MeshBindingMismatch(
            f"mesh {mesh!r}: {role} byte count {size} != declared {want_bytes} ({resolved}); "
            "the supplied file is not the registered authority"
        )
    digest = _sha256_file(resolved)
    if digest != want_sha:
        raise MeshBindingMismatch(
            f"mesh {mesh!r}: {role} SHA-256 {digest} != declared {want_sha} ({resolved}); "
            "the supplied file is not the registered authority"
        )
    return {"path": str(resolved), "bytes": size, "sha256": digest}


def _inspect_grid(grid_path: Path, binding: MeshBinding) -> dict[str, Any]:
    import netCDF4

    with netCDF4.Dataset(str(grid_path)) as dataset:
        dims = {key: len(value) for key, value in dataset.dimensions.items()}
        observed_cells = dims.get("nCells")
        observed_edges = dims.get("nEdges")
        max_edges = dims.get("maxEdges")
        if not isinstance(max_edges, int) or max_edges < 1:
            raise MeshBindingMismatch(
                f"mesh {binding.name!r}: no positive maxEdges dimension; the "
                "reconstruction coefficients cannot be bound to the cell topology"
            )
        if observed_cells != binding.n_cells or observed_edges != binding.n_edges:
            raise MeshBindingMismatch(
                f"mesh {binding.name!r}: grid declares nCells={observed_cells}, nEdges={observed_edges}; "
                f"registry declares nCells={binding.n_cells}, nEdges={binding.n_edges}"
            )
        radius = float(getattr(dataset, "sphere_radius", 0.0) or 0.0)
        raw_nominal = dataset.variables.get("nominalMinDc")
        nominal = (
            float(np.float32(np.asarray(raw_nominal[:]).ravel()[0]))
            if raw_nominal is not None
            else None
        )
        if "nEdgesOnCell" not in dataset.variables or "cellsOnEdge" not in dataset.variables:
            raise MeshBindingMismatch(
                f"mesh {binding.name!r}: grid lacks nEdgesOnCell/cellsOnEdge required to bind topology"
            )
        n_edges_on_cell = np.asarray(dataset.variables["nEdgesOnCell"][:], dtype=np.int64)
        cells_on_edge = np.ascontiguousarray(
            np.asarray(dataset.variables["cellsOnEdge"][:], dtype="<i4")
        )
        mask_variables = {
            name: dataset.variables.get(name) for name in _REGIONAL_MASK_NAMES
        }
        if all(value is not None for value in mask_variables.values()):
            masks = {
                name: np.asarray(variable[:])
                for name, variable in mask_variables.items()
            }
            regional_masks = {
                "present": True,
                "zone_width": int(max(int(mask.max()) for mask in masks.values())),
                "sha256": _mask_triple_digest(masks),
            }
        else:
            regional_masks = {"present": False}
    active_slots = int(n_edges_on_cell.sum())
    return {
        "regional_masks": regional_masks,
        "nCells": observed_cells,
        "nEdges": observed_edges,
        "maxEdges": max_edges,
        "grid_sphere_radius": radius,
        "grid_nominalMinDc_f32": nominal,
        "active_slots": active_slots,
        "active_components": active_slots * 3,
        "cellsOnEdge_shape": [int(cells_on_edge.shape[0]), int(cells_on_edge.shape[1])],
        "cellsOnEdge_raw_sha256": hashlib.sha256(cells_on_edge.tobytes(order="C")).hexdigest(),
    }


def _inspect_static(
    static_path: Path,
    binding: MeshBinding,
    grid_observed: Mapping[str, Any],
) -> dict[str, Any]:
    import netCDF4

    with netCDF4.Dataset(str(static_path)) as dataset:
        dims = {key: len(value) for key, value in dataset.dimensions.items()}
        radius = float(getattr(dataset, "sphere_radius", 0.0) or 0.0)
        raw_nominal = dataset.variables.get("nominalMinDc")
        if raw_nominal is None:
            raise MeshBindingMismatch(
                f"mesh {binding.name!r}: static carries no nominalMinDc; nominal dx cannot be reported"
            )
        observed_dx = np.float32(np.asarray(raw_nominal[:]).ravel()[0])
        raw_dc = dataset.variables.get("dcEdge")
        if raw_dc is None:
            raise MeshBindingMismatch(
                f"mesh {binding.name!r}: static carries no physical dcEdge; timestep admission has no length authority"
            )
        dc_edge = np.asarray(raw_dc[:])
    want_dx = np.float32(binding.nominal_dx_m)
    if observed_dx.view(np.uint32) != want_dx.view(np.uint32):
        raise MeshBindingMismatch(
            f"mesh {binding.name!r}: static nominalMinDc={float(observed_dx)} is not FP32-exact "
            f"equal to registry nominal dx={float(want_dx)}"
        )
    grid_dc = grid_observed.get("grid_nominalMinDc_f32")
    if grid_dc and radius:
        implied = float(grid_dc) * radius
        if abs(implied - float(want_dx)) > 1.0e-3 * float(want_dx):
            raise MeshBindingMismatch(
                f"mesh {binding.name!r}: unit-sphere grid nominalMinDc={grid_dc} rad times "
                f"static sphere_radius={radius} implies {implied:.3f} m, not {float(want_dx)} m; "
                "grid and static are not the same mesh"
            )
    levels = dims.get("nVertLevels")
    soil = dims.get("nSoilLevels")
    if levels is not None and levels != binding.n_levels:
        raise MeshBindingMismatch(
            f"mesh {binding.name!r}: static nVertLevels={levels}, registry={binding.n_levels}"
        )
    if soil is not None and soil != binding.n_soil_levels:
        raise MeshBindingMismatch(
            f"mesh {binding.name!r}: static nSoilLevels={soil}, registry={binding.n_soil_levels}"
        )
    try:
        edge_authority = edge_length_authority(dc_edge, source="static.dcEdge")
    except TimestepAdmissionError as error:
        raise MeshBindingMismatch(f"mesh {binding.name!r}: {error}") from error
    return {
        "nVertLevels": levels,
        "nSoilLevels": soil,
        "static_sphere_radius": radius,
        "nominalMinDc_f32": float(observed_dx),
        "edge_length_authority": edge_authority.as_dict(),
    }


def _zero_digest(shape: tuple[int, ...]) -> str:
    return hashlib.sha256(np.zeros(shape, dtype="<f4").tobytes(order="C")).hexdigest()


def _admit_vertical_dimensions(proof: Any, binding: MeshBinding) -> None:
    """Check the atmospheric shape and the fixed column-physics soil ABI."""

    if binding.n_levels < 3 or binding.n_interfaces != binding.n_levels + 1:
        raise MeshBindingMismatch(
            f"mesh {binding.name!r}: n_levels={binding.n_levels}, "
            f"n_interfaces={binding.n_interfaces}; the lower-boundary "
            "coefficients require at least three layers and one more interface"
        )
    if binding.n_soil_levels != proof.N_SOIL_LEVELS:
        raise MeshBindingMismatch(
            f"mesh {binding.name!r}: n_soil_levels={binding.n_soil_levels}, "
            f"column physics requires {proof.N_SOIL_LEVELS}; changing the soil "
            "extent without converting its layer geometry would feed the "
            "land-surface model the wrong soil depths"
        )
    if binding.frozen_native:
        for attr, want in (
            ("N_LEVELS", binding.n_levels),
            ("N_INTERFACES", binding.n_interfaces),
        ):
            have = getattr(proof, attr)
            if have != want:
                raise MeshBindingMismatch(
                    f"mesh {binding.name!r} is the frozen native mesh, so "
                    f"binding must be a no-op, but {attr}={have}, registry={want}"
                )


def _rebind_atmospheric_dimensions(
    proof: Any, binding: MeshBinding, forecast: Any = None,
) -> dict[str, Any]:
    """Carry a verified row's atmospheric extents to every runtime consumer."""

    rebindings: dict[str, Any] = {}
    for attr, want in (
        ("N_LEVELS", binding.n_levels),
        ("N_INTERFACES", binding.n_interfaces),
    ):
        rebindings[attr] = [int(getattr(proof, attr)), int(want)]
        setattr(proof, attr, int(want))
        if forecast is not None:
            setattr(forecast, attr, int(want))
    return rebindings


def _rebind_gwdo_vertical_dimension(gwdo: Any, binding: MeshBinding) -> list[int]:
    """Keep the runtime GWD shape guard on the verified atmosphere's extent."""

    before = int(gwdo.X4_N_VERT_LEVELS)
    if binding.frozen_native:
        if before != binding.n_levels:
            raise MeshBindingMismatch(
                f"mesh {binding.name!r} is the frozen native mesh, but the "
                f"GWD layer guard moved from {binding.n_levels} to {before}"
            )
        return [before, before]
    gwdo.X4_N_VERT_LEVELS = int(binding.n_levels)
    return [before, int(binding.n_levels)]


def _rebind_gwdo_kernel_dimension(
    gwdo: Any, binding: MeshBinding, physics: Any, config: Any,
) -> dict[str, Any]:
    """Bind CUDA column storage to the same verified extent as its carriers."""

    source = gwdo._CUDA_SOURCE
    definitions = [line for line in source.splitlines()
                   if line.startswith("#define GWDO_NLEV ")]
    if len(definitions) != 1:
        raise MeshBindingMismatch("GWD CUDA source needs one GWDO_NLEV definition")
    definition = definitions[0]
    try:
        before = int(definition.removeprefix("#define GWDO_NLEV "))
    except ValueError as error:
        raise MeshBindingMismatch("GWD CUDA extent is not an integer") from error
    after = int(binding.n_levels)
    before_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if gwdo.CUDA_GWDO_V841_KERNEL_SHA256 != before_sha:
        raise MeshBindingMismatch("GWD CUDA source and recorded kernel hash differ")
    if binding.frozen_native and before != after:
        raise MeshBindingMismatch("frozen native GWD CUDA extent moved")
    rebound = source.replace(definition, f"#define GWDO_NLEV {after}", 1)
    after_sha = hashlib.sha256(rebound.encode("utf-8")).hexdigest()
    if rebound != source:
        gwdo._CUDA_SOURCE = rebound
        # Compiled kernels cache the former source. A new binding must compile
        # the new allocation, and both receipt consumers need its actual hash.
        gwdo._KERNELS.clear()
    gwdo.CUDA_GWDO_V841_KERNEL_SHA256 = after_sha
    physics.CUDA_GWDO_V841_KERNEL_SHA256 = after_sha
    config.V841_GWDO_KERNEL_SHA256 = after_sha
    return {"layers": [before, after], "sha256": [before_sha, after_sha]}


def claim_for_convection(claim: str, scheme: str) -> str:
    """The forecast claim with its convection scheme named as ``scheme``.

    The drivers' claim sentences are written for GF ("WSM6 + GF + YSU ...").
    A GF decision returns the sentence unchanged; any other scheme replaces
    the single "GF + " term with its own label, and convection off drops it.
    The label is the one :mod:`woof.hex.convection_admission` holds for the
    scheme -- the three ruled schemes, and every cumulus option of WOOF's
    registry a run resolved -- so no scheme list lives here.  A scheme no
    selection resolved refuses rather than leaving a receipt that names a
    scheme the run did not execute."""

    if scheme == convection_admission.SCHEME_OFF:
        label = None
    else:
        try:
            label = convection_admission.label(scheme)
        except convection_admission.ConvectionAdmissionError as error:
            raise ValueError(f"no claim label for convection scheme {scheme!r}: {error}") from error
    if label == "GF" or "GF + " not in claim:
        return claim
    return claim.replace("GF + ", "" if label is None else f"{label} + ", 1)


def claim_for_selection(claim: str, labels: Sequence[str]) -> str:
    """The forecast claim with its scheme list replaced by ``labels``.

    THE BREAKAGE THIS PREVENTS: the claim sentence lists the frozen suite
    ("WSM6 + ... + legacy RRTMG") whatever the run selected.  A Thompson
    mp_physics=8 run on the generic column seam (global 30 km, 2026-10-11)
    wrote a receipt whose claim said WSM6 while its constructor record said
    mp_physics=8.  The labels are the engine registry's for the mapping the
    run hands the engine, so no scheme list lives here.  A claim that is not
    shaped "... using <schemes>, at dt = ..." refuses rather than keeping a
    list the run did not execute."""

    head, using, _schemes, at, tail = _claim_parts(claim)
    if not labels:
        raise ValueError("a claim needs at least one scheme label")
    return f"{head}{using}{' + '.join(labels)}{at}{tail}"


def claim_scheme_terms(claim: str) -> tuple[str, ...]:
    """The scheme terms of a claim sentence, in the order it lists them."""

    return tuple(_claim_parts(claim)[2].split(" + "))


def _claim_parts(claim: str) -> tuple[str, str, str, str, str]:
    head, using, rest = claim.partition(" using ")
    schemes, at, tail = rest.partition(", at dt = ")
    if not using or not at or not schemes:
        raise ValueError(
            "the claim sentence has no 'using <schemes>, at dt = ' term to "
            f"name the selected schemes in: {claim!r}")
    return head, using, schemes, at, tail


def bind_mesh(
    proof: Any,
    mesh_name: str,
    *,
    grid: Path,
    static: Path,
    forecast: Any = None,
    verify_frozen_sources: bool = True,
    convection: str = "auto",
    pbl_cadence: str = "auto",
    time_integration_order: int = mpas_defaults.DEFAULT_TIME_INTEGRATION_ORDER,
    acoustic_substeps: int = mpas_defaults.DEFAULT_ACOUSTIC_SUBSTEPS,
    radiation_seconds: float = mpas_defaults.DEFAULT_RADIATION_SECONDS,
    dt_seconds: float | None = None,
    dynamics_splits: int = mpas_defaults.DEFAULT_DYNAMICS_SPLITS,
    max_wind_m_s: float | None = None,
    apvm_upwinding: float = 0.5,
    log=print,
) -> dict[str, Any]:
    """Cross-examine, stability-admit, and bind one registered mesh before CUDA.

    ``max_wind_m_s`` is the run's ``--max-wind``: the largest horizontal
    wind the advective stability estimate assumes, in place of the row's
    (125 m/s by default).  The estimate is not a measurement of the case,
    so the run may state its own bound (2026-10-10 audit: dt 105 s on the
    15 km mesh was refused at 1.189 against 1.160 although it is inside the
    estimate for every wind below 122 m/s).

    ``dt_seconds`` is the run's timestep.  ``None`` takes MPAS-A's rule, 6 s
    per km of the row's nominal spacing rounded down to divide the radiation
    interval (woof.hex.mpas_defaults.default_dt_seconds); the forecast door
    always passes the timestep it resolved.  The row's own ``dt_seconds`` is
    the timestep it was registered with and is recorded beside it.
    """

    if mesh_name not in MESH_BINDINGS:
        raise MeshBindingMismatch(
            f"unknown mesh {mesh_name!r}; registered meshes are {sorted(MESH_BINDINGS)}. "
            "Register dimensions, file pins and nominal dx before running"
        )
    binding = MESH_BINDINGS[mesh_name]
    registered_dt = float(binding.dt_seconds)
    if dt_seconds is None:
        dt_choice = mpas_defaults.default_dt_seconds(
            float(binding.nominal_dx_m), radiation_seconds=float(radiation_seconds)
        )
    else:
        dt_choice = {"dt_seconds": float(dt_seconds), "source": "caller"}
    dt_choice["registered_row_dt_seconds"] = registered_dt
    if float(dt_choice["dt_seconds"]) != registered_dt:
        # The run's timestep travels on the binding from here on, so every
        # rebind below (DT_SECONDS in the proof and forecast modules, the
        # pbl and convection decisions, the receipt) reads ONE number.  A
        # frozen native row bound at another timestep is not a no-op, so it
        # binds like any other row.
        from dataclasses import replace as _replace

        binding = _replace(
            binding, dt_seconds=float(dt_choice["dt_seconds"]), frozen_native=False
        )

    frozen: dict[str, Any] | None = None
    if verify_frozen_sources:
        frozen = proof.measure_execution_sources()
        log(
            f"[mesh-binding] execution sources recorded: {len(frozen['files'])} modules, "
            f"receipt {frozen['sha256'][:16]}, "
            f"{'all match' if frozen['matches_pins'] else str(len(frozen['moved_from_pins'])) + ' differ from'} "
            "the x4 proof's pins"
        )

    files = {
        "grid": _require_file("grid", Path(grid), binding.grid_bytes, binding.grid_sha256, mesh_name),
        "static": _require_file("static", Path(static), binding.static_bytes, binding.static_sha256, mesh_name),
    }
    observed = _inspect_grid(Path(files["grid"]["path"]), binding)
    # Regional cross-examination runs for EVERY row, before any constant is
    # rebound: a regional cull bound as a global row, or a regional row with
    # an empty boundary-source slot, is refused by name right here.
    observed["regional_admission"] = admit_regional_row(binding, observed)
    observed.update(_inspect_static(Path(files["static"]["path"]), binding, observed))
    # Reconstruct the immutable authority through its public constructor -- the
    # same read a fingerprint baseline performs -- rather than trusting the
    # mapping the inspection pass already built.
    authority = _static_edge_authority(Path(files["static"]["path"]))
    try:
        timestep = admit_timestep(
            binding.dt_seconds,
            authority,
            policy=binding.courant_policy(max_wind_m_s),
            schedule=SplitExplicitSchedule(
                time_integration_order=int(time_integration_order),
                acoustic_substeps=int(acoustic_substeps),
                dynamics_splits=int(dynamics_splits),
            ),
            cadence_seconds=float(radiation_seconds),
        )
    except TimestepAdmissionError as error:
        raise MeshBindingMismatch(f"mesh {mesh_name!r}: {error}") from error
    observed["timestep_admission"] = timestep.as_dict()
    # Positive dual geometry is required for every mesh. The ratio floor
    # prevents the measured APVM gradient runaway only when that operator
    # is selected; disabling APVM keeps the roughness census in the receipt.
    dv_edge, dc_edge, cells_on_edge = _static_dual_edges(
        Path(files["static"]["path"])
    )
    try:
        dual_edges = admit_dual_edges(
            dv_edge,
            dc_edge,
            policy=binding.dual_edge_policy(apvm_upwinding=apvm_upwinding),
            cells_on_edge=cells_on_edge,
            mesh_name=mesh_name,
        )
    except DualEdgeAdmissionError as error:
        raise MeshBindingMismatch(str(error)) from error
    observed["dual_edge_admission"] = dual_edges.as_dict()
    # Cell-coordination admission runs for EVERY mesh, and its histogram is
    # recorded whether it admits or refuses.  A cell the icosahedral seeding
    # cannot produce -- one with fewer than five edges, which only
    # count-changing defect surgery creates -- was measured to carry a 197 K
    # standing theta error that no timestep removes, so it is refused here
    # rather than at step 23 on a validation flag that names no cell.
    try:
        coordination = admit_cell_coordination(
            _grid_cell_coordination(Path(files["grid"]["path"])),
            policy=binding.cell_coordination_policy(),
            mesh_name=mesh_name,
        )
    except CellCoordinationAdmissionError as error:
        raise MeshBindingMismatch(str(error)) from error
    observed["cell_coordination_admission"] = coordination.as_dict()
    # RETIRED 2026-10-10:
    # the refusal of every timestep the earned-anchor registry did not hold
    # ("this row is STABLE and merely unrunnable HERE").  Stability is the
    # physical check above; the cadences are checked below against the run's
    # own radiation interval; whether the configuration matches a registered
    # anchor is recorded in observed["dt_admission"].
    #
    # The cumulus selection is decided from this mesh's own finest spacing:
    # it was ruled on 2026-08-26 that convection is switched off below 3 km,
    # and on 2026-10-05 that New Tiedtke runs at 20 km and coarser;
    # ``convection='auto'`` is that default and needs no flag.
    #
    # On a variable-resolution mesh the cells' own scheme spacing decides
    # (len_disp/meshDensity**0.25, the length the scheme is handed): read
    # at min(dcEdge), the 3 km ruling switched cumulus off for the whole
    # 48-3 km patch mesh, its 48 km cells included
    # (woof.hex.convection_admission).
    convection_decision = convection_admission.convection_decision(
        nominal_dx_m=float(binding.nominal_dx_m),
        minimum_dc_edge_m=float(authority.minimum_m),
        requested=convection,
        cell_spacing_m=_grid_cell_spacing(
            Path(files["grid"]["path"]), float(binding.nominal_dx_m)
        ),
    )
    observed["convection_admission"] = convection_decision
    cumulus_scheme = convection_decision["constructor_scheme"]
    log(
        f"[mesh-binding] convection: finest spacing "
        f"{convection_decision['finest_spacing_m']:.1f} m vs "
        f"{convection_decision['threshold_m']:.0f} m threshold -> "
        f"{convection_decision['scheme']} ({convection_decision['source']})"
    )
    # The surface/PBL cadence is taken here from the run's timestep.  'auto'
    # is the weld (config_bldt_seconds = config_dt), which is MPAS-A's own
    # default (config_pbl_interval = 'none' -> dt_pbl = dt,
    # mpas_atmphys_manager.F:541-542) and needs no flag.  See
    # woof.hex.pbl_cadence.
    try:
        pbl_decision = pbl_cadence_module.pbl_cadence_decision(
            dt_seconds=float(binding.dt_seconds), requested=pbl_cadence,
            radiation_seconds=float(radiation_seconds),
        )
    except pbl_cadence_module.PblCadenceError as error:
        largest = mpas_defaults.default_dt_seconds(
            float(binding.dt_seconds) * 1000.0 / mpas_defaults.SECONDS_PER_KM,
            radiation_seconds=float(radiation_seconds),
        )["dt_seconds"]
        raise MeshBindingMismatch(
            f"mesh {mesh_name!r}: {error}.  The largest whole-second timestep "
            f"at or below {binding.dt_seconds:g} s that divides the "
            f"{float(radiation_seconds):g} s radiation interval is {largest:g} s"
        ) from error
    observed["pbl_cadence"] = pbl_decision
    surface_pbl_seconds = float(pbl_decision["surface_pbl_seconds"])
    log(
        f"[mesh-binding] surface/PBL cadence: {pbl_decision['label']} "
        f"({pbl_decision['calls_per_hour']:g} calls/hour vs the proven "
        f"{pbl_decision['calls_per_hour_at_proven_dt']:g}) "
        f"({pbl_decision['source']})"
    )
    # THE BREAKAGE THIS PREVENTS: a radiation, surface/PBL or cumulus
    # cadence that is not a whole number of this timestep's steps has no step
    # to run on, and the sealed constructor refuses it at host preparation
    # after the card is reserved (measured 2026-08-26: a row at 90 s against
    # the then 600 s radiation cadence died on both 6 h legs).  Checked here
    # against the RUN's radiation interval, not a constant.
    cumulus_seconds = None if cumulus_scheme is None else float(binding.dt_seconds)
    try:
        cadence_steps = dt_admission.require_cadences(
            binding.dt_seconds,
            radiation_seconds=float(radiation_seconds),
            surface_pbl_seconds=surface_pbl_seconds,
            cumulus_seconds=cumulus_seconds,
            cumulus_scheme=cumulus_scheme,
        )
    except dt_admission.DtAdmissionError as error:
        largest = mpas_defaults.default_dt_seconds(
            float(binding.dt_seconds) * 1000.0 / mpas_defaults.SECONDS_PER_KM,
            radiation_seconds=float(radiation_seconds),
        )["dt_seconds"]
        raise MeshBindingMismatch(
            f"mesh {mesh_name!r}: {error}.  The largest whole-second timestep "
            f"at or below {binding.dt_seconds:g} s that divides the "
            f"{float(radiation_seconds):g} s radiation interval is {largest:g} s"
        ) from error
    from types import SimpleNamespace
    from woof.hex.step_anchors import step_anchor_record

    observed["dt_admission"] = {
        "refuses": False,
        "dt_seconds": float(binding.dt_seconds),
        "dt_selection": dt_choice,
        "cadence_steps": cadence_steps,
        "anchor": dt_admission.anchor_record(
            binding.dt_seconds,
            radiation_seconds=float(radiation_seconds),
            surface_pbl_seconds=surface_pbl_seconds,
            cumulus_seconds=cumulus_seconds,
            cumulus_scheme=cumulus_scheme,
        ),
        "dycore_anchor": step_anchor_record(SimpleNamespace(
            config_dt=float(binding.dt_seconds),
            config_time_integration_order=int(time_integration_order),
            config_number_of_sub_steps=int(acoustic_substeps),
            config_dynamics_split_steps=int(dynamics_splits),
        )),
    }
    log(
        f"[mesh-binding] mesh {mesh_name}: nCells={observed['nCells']} "
        f"nEdges={observed['nEdges']} nominal={observed['nominalMinDc_f32']} m; "
        f"min(dcEdge)={authority.minimum_m:.3f} m; dt={binding.dt_seconds:.3f} s "
        f"({dt_choice['source']}); stability limit="
        f"{timestep.maximum_admitted_dt_seconds:.3f} s (advective "
        f"{timestep.estimated_outer_courant:.3f}/{timestep.admitted_courant:.3f}, "
        f"acoustic {timestep.acoustic_courant:.3f}/"
        f"{timestep.acoustic_courant_limit:.3f})"
    )
    log(
        f"[mesh-binding] mesh {mesh_name}: min(dvEdge/dcEdge)="
        f"{dual_edges.minimum_ratio:.6g} at edge {dual_edges.minimum_ratio_edge} "
        f"(dvEdge={dual_edges.minimum_ratio_dv_edge_m:.3f} m), "
        f"TRiSK tangential amplification {dual_edges.amplification:.4g}x; "
        f"floor {dual_edges.policy.minimum_dv_over_dc:.6g}; "
        f"APVM={dual_edges.policy.apvm_upwinding:g}, "
        f"ratio floor active={dual_edges.policy.apvm_upwinding > 0.0}"
    )

    edge_sha = authority.raw_sha256
    fingerprint_before = _fingerprint_with_authority(proof, authority)

    _admit_vertical_dimensions(proof, binding)

    native = MESH_BINDINGS[NATIVE_MESH_NAME]
    if binding.frozen_native:
        mismatches: list[str] = []
        for attr, want in (
            ("N_CELLS", binding.n_cells),
            ("N_EDGES", binding.n_edges),
            ("MIN_FREE_DEVICE_BYTES", NATIVE_DEVICE_FLOOR),
        ):
            have = getattr(proof, attr)
            if int(have) != int(want):
                mismatches.append(f"{attr}={have!r} != {want!r}")
        if float(getattr(proof, "DT_SECONDS")) != float(binding.dt_seconds):
            mismatches.append(
                f"DT_SECONDS={getattr(proof, 'DT_SECONDS')!r} != {binding.dt_seconds!r}"
            )
        have_dx = np.float32(proof.NOMINAL_DX_M)
        if have_dx.view(np.uint32) != np.float32(binding.nominal_dx_m).view(np.uint32):
            mismatches.append(f"NOMINAL_DX_M={float(have_dx)} != {binding.nominal_dx_m}")
        if mismatches:
            raise MeshBindingMismatch(
                f"mesh {mesh_name!r} is the frozen native mesh, so binding must be a no-op, "
                f"but constants already moved: {'; '.join(mismatches)}"
            )
        fingerprint_after = _fingerprint_with_authority(proof, authority)
        if fingerprint_after["sha256"] != fingerprint_before["sha256"]:
            raise MeshBindingError(
                "native x4 bind was required to change nothing, but fingerprint moved "
                f"{fingerprint_before['sha256'][:16]} -> {fingerprint_after['sha256'][:16]}"
            )
        log(
            f"[mesh-binding] {mesh_name} frozen no-op: fingerprint unchanged "
            f"at {fingerprint_after['sha256'][:16]}"
        )
        return {
            "mesh": mesh_name,
            "rebound": False,
            "files": files,
            "observed": observed,
            "frozen_execution_sources": frozen,
            "constants_fingerprint_before": fingerprint_before["sha256"],
            "constants_fingerprint_after": fingerprint_after["sha256"],
            "edge_length_authority_sha256": edge_sha,
            "timestep_admission": timestep.as_dict(),
            "rebindings": {},
            "notes": binding.notes,
        }

    rebindings: dict[str, Any] = {}
    nc, ne = binding.n_cells, binding.n_edges

    for role in ("grid", "static"):
        pin = dict(proof.AUTHORITY_PINS[role])
        pin["bytes"] = files[role]["bytes"]
        pin["sha256"] = files[role]["sha256"]
        pin["relative_path"] = files[role]["path"]
        proof.AUTHORITY_PINS[role] = pin
    rebindings["AUTHORITY_PINS"] = {
        role: files[role]["sha256"][:16] for role in ("grid", "static")
    }

    proof.N_CELLS, proof.N_EDGES = nc, ne
    rebindings["N_CELLS"] = [native.n_cells, nc]
    rebindings["N_EDGES"] = [native.n_edges, ne]
    rebindings.update(_rebind_atmospheric_dimensions(proof, binding, forecast))

    before_dt = float(proof.DT_SECONDS)
    proof.DT_SECONDS = float(binding.dt_seconds)
    rebindings["DT_SECONDS"] = [before_dt, float(binding.dt_seconds)]

    max_edges = observed["maxEdges"]
    reconstruction_shape = (nc, max_edges, 3)
    rc_pin = dict(proof.INIT_RECONSTRUCTION_COEFFICIENTS_PIN)
    rc_pin["shape"] = reconstruction_shape
    rc_pin["static_placeholder_raw_sha256"] = _zero_digest(reconstruction_shape)
    rc_pin["active_slots"] = observed["active_slots"]
    rc_pin["active_components"] = observed["active_components"]
    rc_pin["nonzero_components"] = observed["active_components"]
    proof.INIT_RECONSTRUCTION_COEFFICIENTS_PIN = MappingProxyType(rc_pin)
    rebindings["INIT_RECONSTRUCTION_COEFFICIENTS_PIN"] = {
        "shape": list(reconstruction_shape),
        "active_slots": observed["active_slots"],
        "active_components": observed["active_components"],
    }

    en_pin = dict(proof.INIT_EDGE_NORMAL_VECTORS_PIN)
    en_pin["shape"] = (ne, 3)
    en_pin["static_placeholder_raw_sha256"] = _zero_digest((ne, 3))
    proof.INIT_EDGE_NORMAL_VECTORS_PIN = MappingProxyType(en_pin)
    rebindings["INIT_EDGE_NORMAL_VECTORS_PIN"] = {"shape": [ne, 3]}

    geometry = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in proof.PHYSICS_GEOMETRY_CARRIER_PIN.items()
    }
    geometry["cellsOnEdge"]["shape"] = (ne, 2)
    geometry["cellsOnEdge"]["raw_sha256"] = observed["cellsOnEdge_raw_sha256"]
    proof.PHYSICS_GEOMETRY_CARRIER_PIN = MappingProxyType(geometry)
    rebindings["PHYSICS_GEOMETRY_CARRIER_PIN"] = {
        "cellsOnEdge_shape": [ne, 2],
        "cellsOnEdge_raw_sha256": observed["cellsOnEdge_raw_sha256"][:16],
    }

    if hasattr(proof, "LANDMASK_CONSTRUCTOR_CAST_PIN"):
        landmask = dict(proof.LANDMASK_CONSTRUCTOR_CAST_PIN)
        landmask["shape"] = (nc,)
        landmask["source_array_sha256"] = None
        landmask["rebound_for_mesh"] = mesh_name
        proof.LANDMASK_CONSTRUCTOR_CAST_PIN = MappingProxyType(landmask)
        rebindings["LANDMASK_CONSTRUCTOR_CAST_PIN"] = {"shape": [nc]}

    dx = np.float32(binding.nominal_dx_m)
    proof.NOMINAL_DX_M = dx
    rebindings["NOMINAL_DX_M"] = [float(native.nominal_dx_m), float(dx)]

    if binding.scale_admission_floor:
        # The per-mesh floor is the same admission sum the forecast door
        # computes -- the measured affine row plus headroom, from the ONE
        # surface -- so a --preflight verdict and the driver's own gate agree
        # to the byte.  The linear ``NATIVE_DEVICE_FLOOR * nc / native``
        # scaling this replaced admitted x1.40962 at 6,144 MiB free against a
        # measured 8,874 MiB peak: an admission that dies inside a CuPy
        # allocation mid-run, which is the breakage this floor prevents.
        floor = device_admission.required_free_bytes(nc)
        inner = proof.gpu_memory_admission

        def _admit(cp, *, minimum=None, _inner=inner, _floor=floor):
            return _inner(cp, minimum=_floor if minimum is None else minimum)

        proof.gpu_memory_admission = _admit
        proof.MIN_FREE_DEVICE_BYTES = floor
        if hasattr(proof, "RESTART_WORKER_MIN_FREE_DEVICE_BYTES"):
            # The worker allocates the same device stack; same requirement.
            proof.RESTART_WORKER_MIN_FREE_DEVICE_BYTES = floor
        rebindings["MIN_FREE_DEVICE_BYTES"] = [
            NATIVE_DEVICE_FLOOR,
            floor,
            f"{floor / 1024**3:.2f} GiB",
        ]

    dropped: dict[str, Any] = {}
    if binding.drop_carried_deformation:
        attach = proof.attach_inactive_zero_deformation

        def _attach(mesh, _attach=attach, _dropped=dropped):
            arrays = getattr(mesh, "arrays", None)
            if isinstance(arrays, dict):
                for name in ("defc_a", "defc_b"):
                    if name in arrays:
                        array = np.ascontiguousarray(arrays.pop(name))
                        _dropped[name] = {
                            "shape": list(array.shape),
                            "nonzero": int(np.count_nonzero(array)),
                            "raw_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                        }
            return _attach(mesh)

        proof.attach_inactive_zero_deformation = _attach
        rebindings["attach_inactive_zero_deformation"] = "carried defc dropped"

    if forecast is not None:
        forecast.N_CELLS, forecast.N_EDGES = nc, ne
        forecast.NOMINAL_DX_M = dx
        forecast.DT_SECONDS = float(binding.dt_seconds)
        # ONE convection decision, taken here from this mesh's own measured
        # spacing and read by the config builder -- the same shape the
        # 2026-08-26 clock fix gave DT_SECONDS.  A second decision taken from
        # a default in the runner is exactly how config_dt and the seam's dt
        # came to disagree.
        forecast.CONVECTION_DECISION = dict(convection_decision)
        # The surface/PBL cadence rides the same road for the same reason: a
        # second decision taken from a default in the runner is exactly how
        # config_dt and the seam's dt came to disagree.
        forecast.PBL_CADENCE_DECISION = dict(pbl_decision)
        rebindings["forecast_reexports"] = [
            "N_CELLS",
            "N_EDGES",
            "N_LEVELS",
            "N_INTERFACES",
            "NOMINAL_DX_M",
            "DT_SECONDS",
            "CONVECTION_DECISION",
            "PBL_CADENCE_DECISION",
        ]
        # THE BREAKAGE THIS PREVENTS: a receipt for a run on this mesh that
        # says "x4.163842" in its own claim sentence and profile slug.  Both
        # constants are written verbatim into every receipt, and a receipt
        # that names the wrong mesh is worse than no receipt -- a reader
        # comparing two runs would take them for the same shape.  The native
        # row rebinds nothing, so its claim and profile stay byte-identical.
        native_label, mesh_label = native.name, binding.name
        if mesh_label != native_label:
            for module, attribute in (
                (forecast, "CLAIM"),
                (proof, "PROFILE"),
                (proof, "CLAIM"),
            ):
                text = getattr(module, attribute, None)
                if not isinstance(text, str) or native_label not in text:
                    continue
                setattr(
                    module,
                    attribute,
                    text.replace(native_label, mesh_label).replace(
                        f"dt = {native.dt_seconds:g} s",
                        f"dt = {binding.dt_seconds:g} s",
                    ),
                )
                rebindings[f"{attribute}_mesh_label"] = [native_label, mesh_label]
        # THE BREAKAGE THIS PREVENTS: the claim sentence names "GF" as the
        # convection scheme whatever the decision above selected.  The RRFS v2
        # conus12km run with New Tiedtke (2026-10-05)
        # carried a receipt claiming GF.  A GF run's claim is left byte-identical.
        claim = getattr(forecast, "CLAIM", None)
        if isinstance(claim, str):
            rewritten = claim_for_convection(claim, convection_decision["scheme"])
            if rewritten != claim:
                forecast.CLAIM = rewritten
                rebindings["CLAIM_convection_label"] = convection_decision["scheme"]
        prepare = forecast.prepare_forecast_host

        import woof.hex.cuda_gwdo_v841 as gwdo
        import woof.hex.cuda_arwen_physics_v841 as arwen_physics
        import woof.hex.config_v841 as config

        rebindings["cuda_gwdo_v841.kernel_dimension"] = (
            _rebind_gwdo_kernel_dimension(gwdo, binding, arwen_physics, config)
        )

        def _prepare(
            *args,
            _prepare=prepare,
            _dx=dx,
            _dt=np.float32(binding.dt_seconds),
            _binding=binding,
            **kwargs,
        ):
            import woof.hex.cuda_gwdo_v841 as gwdo

            before = np.float32(gwdo.X4_NOMINAL_MIN_DC_M_F32)
            gwdo.X4_NOMINAL_MIN_DC_M_F32 = _dx
            # The GWDO run guard compares the composite dt against this module
            # constant bit-for-bit.  On the frozen native mesh nothing is
            # rebound and the guard still demands exactly 120 s; a registered
            # mesh runs at ITS declared, Courant-admitted timestep, so the
            # guard must demand exactly that value instead -- the kernel takes
            # dt as a runtime argument and is dt-general.  Without this
            # rebind every registered mesh whose dt is not 120 s dies at
            # step 0 in run_bl_ysu_gwdo_cuda_v841.
            before_dt = np.float32(gwdo.X4_DT_SECONDS_F32)
            gwdo.X4_DT_SECONDS_F32 = _dt
            before_levels, after_levels = _rebind_gwdo_vertical_dimension(gwdo, _binding)
            log(
                f"[mesh-binding] GWD effective len_disp {float(before)} -> {float(_dx)} m; "
                f"GWDO dt guard {float(before_dt)} -> {float(_dt)} s; "
                f"GWD layer guard {before_levels} -> {after_levels}"
            )
            return _prepare(*args, **kwargs)

        forecast.prepare_forecast_host = _prepare
        rebindings["cuda_gwdo_v841.X4_NOMINAL_MIN_DC_M_F32"] = [
            float(native.nominal_dx_m),
            float(dx),
        ]
        rebindings["cuda_gwdo_v841.X4_DT_SECONDS_F32"] = [
            float(native.dt_seconds),
            float(binding.dt_seconds),
        ]
        rebindings["cuda_gwdo_v841.X4_N_VERT_LEVELS"] = [
            native.n_levels,
            binding.n_levels,
        ]

    fingerprint_after = _fingerprint_with_authority(proof, authority)
    if fingerprint_after["sha256"] == fingerprint_before["sha256"]:
        raise MeshBindingError(
            f"mesh {mesh_name!r} was supposed to rebind shape/timestep constants, but "
            "the fingerprint did not move; the treatment never engaged"
        )
    log(
        f"[mesh-binding] bound {mesh_name}: nCells={nc}, nEdges={ne}, "
        f"dx={float(dx)} m, dt={binding.dt_seconds} s; fingerprint "
        f"{fingerprint_before['sha256'][:16]} -> {fingerprint_after['sha256'][:16]}"
    )
    return {
        "mesh": mesh_name,
        "rebound": True,
        "files": files,
        "observed": observed,
        "frozen_execution_sources": frozen,
        "constants_fingerprint_before": fingerprint_before["sha256"],
        "constants_fingerprint_after": fingerprint_after["sha256"],
        "edge_length_authority_sha256": edge_sha,
        "timestep_admission": timestep.as_dict(),
        "rebindings": rebindings,
        "deformation_dropped": dropped,
        "notes": binding.notes,
    }
