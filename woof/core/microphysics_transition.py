"""Explicit one-way nest microphysics transition contracts.

Stock WRF v4.6.1 accepts an ``mp_physics(max_domains)`` namelist vector but
``share/module_check_a_mundo.F:774-790`` replaces every entry with the
innermost-domain selector.  WRF therefore has no executable mixed-scheme nest
edge.  The matrix in this module is a WOOF extension: shared mass species
are mapped, absent target mass species use their scheme cold-start default,
and target number/volume moments are diagnosed from target mass.

The previously ratified Thompson MP8 -> NSSL-2 MP18 path retains its original
policy id and CUDA entry point.  All other mixed edges use one matrix policy.

mp_physics=50 (P3, one-category ice with a prognostic rime pair) is a matrix
member with its own defined closure, because WRF itself defines neither
direction: entering P3 merges every frozen source species into the single ice
category mass-conservingly and DIAGNOSES the rime pair from named constants;
leaving P3 splits the single category back into qi/qs/qg by rime state,
mass-conservingly, with the degenerate cases (zero ice, zero rime) exact.
The constants and both closed forms are documented at the ``P3_EDGE_*``
constants below and recorded in every touching edge's receipt.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace as dataclasses_replace
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

# The constants module, not the table contract: this module is staged
# into the standalone RW-WPS preparation wheel and the contract (with the
# correctly rounded libm behind it) is not, so the edge reads the three
# scalars from the import-free module both of them share.
from woof.core.thompson_aerosol_constants import (
    NIFA_FLOOR, NT_C, NWFA_FLOOR)
from woof.core.wdm6_constants import WDM6_NUMBER_SPECIES


SAME_SCHEME_POLICY = "same-scheme-only"
MP8_TO_MP18_POLICY = "mp8-to-mp18-mass-diagnosed-v1"
EDGE_MATRIX_POLICY = "mp-edge-mass-diagnosed-v1"
TRANSITION_ORDER = "diagnose-parent-then-spatially-interpolate"
#: The FEEDBACK direction of the same matrix: the CHILD's scheme is the
#: source and the parent's is the target.  A separate id because the order
#: is separate -- the diagnosis runs on the child's own grid and the
#: restriction that follows it is an average, not an interpolation -- and
#: because a receipt that called both directions by one name could not say
#: which way the mass went.
REVERSE_EDGE_POLICY = "mp-edge-mass-diagnosed-reverse-v1"
REVERSE_TRANSITION_ORDER = "diagnose-child-then-spatially-restrict"
NSSL2_BACKGROUND_CCN_PER_KG = 408163264.0

# ---------------------------------------------------------------------------
# mp_physics=50 (P3) mixed-edge closure constants.  WRF v4.6.1 defines
# neither direction of a P3 mixed nest edge (share/module_check_a_mundo.F:
# 774-790 forbids mixed selectors outright), so every constant here is a
# DEFINED, documented ArWen choice anchored to a WRF constant that exists.
# ---------------------------------------------------------------------------

#: Rime density assigned to a six-species parent's SNOW mass when it enters
#: P3's single ice category.  100 kg m-3 is WRF's own bulk snow density --
#: Morrison spells it ``RHOSN = 100.`` (module_mp_morr_two_moment.F:377) and
#: WSM6/NSSL carry the same 100 for their snow categories -- and it sits
#: inside P3's admitted rime-density range [50, 900] kg m-3
#: (module_mp_p3.F:225-226, ``rho_rimeMin``/``rho_rimeMax``).
P3_EDGE_FRESH_SNOW_RIME_DENSITY = 100.0

#: Rime density assigned to a six-species parent's GRAUPEL and HAIL mass
#: entering P3.  400 kg m-3 is Morrison's graupel bulk density
#: (``RHOG = 400.`` in the ``IHAIL.EQ.0`` arm, module_mp_morr_two_moment.F:
#: 378-382) -- the same constant this matrix already binds as
#: ``target_morrison_rimed_density`` for a graupel-mode Morrison child.
#: DOCUMENTED DIVERGENCE: Thompson/WSM6/NSSL use 500 for graupel and
#: Morrison-hail/NSSL-hail use 900; one dense-rime constant covers both
#: rimed species so the split back out of P3 is a two-endpoint partition
#: against exactly the two densities the entry diagnosis used.
P3_EDGE_DENSE_RIME_DENSITY = 400.0

#: P3's own smallness threshold for mass mixing ratios
#: (``qsmall = 1.e-14``, module_mp_p3.F:230).  Ice below it returns to
#: vapor on entry, exactly as p3_main itself does (:4919-4925 equivalent,
#: woof/core/p3.py:1616-1622).
P3_EDGE_QSMALL_KG_PER_KG = 1.0e-14

#: Diagnosed rain number per unit rain mass [kg-1 per kg kg-1] entering P3.
#: This is the closed form of what P3's own first call computes for rain
#: mass with no number: ``get_rain_dsd2`` floors nr at nsmall, lands below
#: ``lammin = (mu_r + 1) * inv_Drmax`` and reconstructs
#: ``nr = lamr**3 * qr * gamma(mu_r+1) / (gamma(mu_r+4) * cons1)``
#: (module_mp_p3.F:6705-6780, the lammin branch; ``mu_r = 0`` :219,
#: ``inv_Drmax = 1./0.002`` :222, ``cons1 = piov6*1000`` :259).  Computed
#: in the same float32 chain p3_init uses; the kernel carries the identical
#: value as the literal ``39788.727f``.
def _p3_edge_rain_number_per_mass() -> float:
    f = np.float32
    lammin = (f(0.0) + f(1.0)) * (f(1.0) / f(0.002))   # (mu_r+1)*inv_Drmax
    cons1 = f(3.14159265) * (f(1.0) / f(6.0)) * f(1000.0)   # piov6*rhow
    return float((lammin * lammin) * lammin / (f(6.0) * cons1))


P3_EDGE_RAIN_NUMBER_PER_RAIN_MASS = _p3_edge_rain_number_per_mass()

#: Mass of one freshly nucleated P3 ice crystal (``mi0 = 4.*piov3*900.*
#: 1.e-18``, module_mp_p3.F:242).  The entry diagnosis treats the merged
#: frozen mass as freshly nucleated crystals of this mass, then caps the
#: number with P3's own total-ice-number cap, so the diagnosed ni is a
#: state P3's own clamps admit.  Kernel literal: ``3.7699116e-15f``.
def _p3_edge_ice_nucleation_mass() -> float:
    f = np.float32
    piov3 = f(3.14159265) * (f(1.0) / f(3.0))
    return float(f(4.0) * piov3 * f(900.0) * f(1.0e-18))


P3_EDGE_ICE_NUCLEATION_MASS_KG = _p3_edge_ice_nucleation_mass()

#: P3's total ice number cap (``max_total_Ni = 2000.e+3`` m-3,
#: module_mp_p3.F:186, applied per ``impose_max_total_Ni`` :6833-6855).
P3_EDGE_MAX_TOTAL_NI_PER_M3 = 2000.0e3

#: P3's admitted rime-density interval (module_mp_p3.F:225-226).  Both edge
#: directions hold the derived density ``qir/qib`` inside it by
#: construction: entry diagnoses densities in [100, 400] and exit runs the
#: exact ``calc_bulkRhoRime`` clamps (:6784-6830) before splitting.
P3_RIME_DENSITY_BOUNDS_KG_M3 = (50.0, 900.0)
#: WDM6's entry closure on a mixed nest edge (mp_physics=16 as TARGET).
#:
#: nc and nr enter at zero and nn at the domain's ``wdm6_ccn_conc``.  This
#: is not a new number: it is the identical triple
#: ``ingest/microphysics_cold_start.source_absent_microphysics`` writes for
#: a WDM6 domain whose analyzed input carries no number species, and
#: ``ccn_conc`` is the value WRF's ``flow_dep_bdy_qnn`` pushes through an
#: inflow face (module_bc.F; ported at ``ingest/lateral_bc.py``).  A
#: one-way nest edge is a lateral boundary, so the edge uses the boundary's
#: own value rather than inventing one.  WDM6's clamp
#: ``min(max(nn,1.e8),2.e10)`` (module_mp_wdm6.F:584) admits it, and
#: ``validate_run_config`` already holds ``wdm6_ccn_conc`` inside that
#: interval for mp=16, so the seed cannot be outside the scheme's own bound.
WDM6_EDGE_ENTRY_CLOUD_NUMBER = 0.0
WDM6_EDGE_ENTRY_RAIN_NUMBER = 0.0

#: Thompson-aerosol's entry closure on a mixed nest edge (mp_physics=28 as
#: TARGET), for the three species classic Thompson does not carry.
#:
#: These are WRF's OWN non-aerosol-aware values -- the ELSE branch
#: ``mp_gt_driver`` takes when ``is_aerosol_aware`` is FALSE
#: (module_mp_thompson.F:1248-1255): ``nc = Nt_c/rho``,
#: ``nwfa = 11.1E6/rho``, ``nifa = naIN1*0.01/rho == 5.0E3/rho``.  They are
#: read from :mod:`woof.core.thompson_aerosol_constants`, the module the
#: table contract itself re-exports them from, rather than re-typed, so the
#: edge and the scheme's own floors cannot drift.  nr and
#: ni are NOT here: they are diagnosed by the same two Thompson closures
#: the ratified mp=8 edge already runs, with the same field codes.
MP28_EDGE_ENTRY_CLOUD_NUMBER_PER_M3 = NT_C
MP28_EDGE_ENTRY_NWFA_PER_M3 = NWFA_FLOOR
MP28_EDGE_ENTRY_NIFA_PER_M3 = NIFA_FLOOR

#: Per-field receipt reasons for the two entry closures, so a
#: ``species_actions`` row names the seeded value's authority rather than
#: the generic mass-moment closure.
_WDM6_EDGE_ENTRY_REASONS = {
    "nn": "wdm6_ccn_reservoir_seeded_at_domain_ccn_conc_as_inflow_face",
    "nc": "wdm6_cloud_number_enters_at_zero_as_on_a_cold_start",
    "nr": "wdm6_rain_number_enters_at_zero_as_on_a_cold_start",
}
_MP28_EDGE_ENTRY_REASONS = {
    "nc": "thompson_non_aerosol_aware_droplet_number_fallback",
    "nwfa": "thompson_water_friendly_aerosol_floor_fallback",
    "nifa": "thompson_ice_friendly_aerosol_floor_fallback",
}
_MP28_EDGE_ENTRY_VALUES = {
    "nc": MP28_EDGE_ENTRY_CLOUD_NUMBER_PER_M3,
    "nwfa": MP28_EDGE_ENTRY_NWFA_PER_M3,
    "nifa": MP28_EDGE_ENTRY_NIFA_PER_M3,
}

#: One line per selector whose mixed-edge ENTRY seeds a moment from a
#: ported default instead of mapping it from the parent's own field.
#: Plan review prints it for the child that will run the edge, so the
#: mapping is named before the run rather than found in a receipt.  A
#: scheme whose every moment maps from the parent has no row here.
MIXED_EDGE_ENTRY_NOTES = {
    13: ("SBU-YLin child: RIMI is diagnosed from cloud water, snow and "
         "density using WRF 4.6.1 module_mp_sbu_ylin.F:500-520, including "
         "the first column maximum propagated downward."),
    16: ("WDM6 child: nc and nr enter at zero and the CCN reservoir nn at "
         "this domain's wdm6_ccn_conc, the seed a WDM6 inflow face uses."),
    28: ("Thompson aerosol-aware child: nr and ni are carried from a "
         "Morrison or Thompson parent and otherwise diagnosed as on the "
         "mp=8 edge; nc, nwfa and nifa enter at WRF's non-aerosol-aware "
         "values."),
}


def mixed_edge_entry_note(contract) -> str | None:
    """The plan-review line for ``contract``, or ``None``.

    Only a MIXED edge into a selector with a seeded moment has one; a
    same-scheme edge and a fully mapped mixed edge print nothing.
    """
    if not getattr(contract, "mixed", False):
        return None
    return MIXED_EDGE_ENTRY_NOTES.get(int(contract.target_mp_physics))


#: Selectors with a ported MIXED nest edge.  APPEND ONLY: ``_ALL_EDGE_FIELDS``
#: iterates this tuple in order and ``_EDGE_FIELD_CODES`` is ``enumerate``
#: over the result, so inserting a selector anywhere but the end silently
#: renumbers the stable host field codes ``kernels/nest_microphysics.cu``
#: switches on -- which would re-point the ratified MP8 -> MP18 nest edge at
#: different fields with nothing raising.  That is why the tuple is no
#: longer in ascending order: 16 and 28 were APPENDED when their closures
#: were ratified, and moving them to their numeric places would move
#: everything after them.
#:
#: mp_physics=50 (P3) was APPENDED after its rime-pair closure was defined,
#: documented and unit-tested (the ``P3_EDGE_*`` constants above): its
#: qv/qc/qr/qi and nr/ni reuse existing codes and its rime pair took the
#: next free codes 20/21, moving nothing -- exactly the append discipline
#: this tuple's comment demands.
#: mp_physics=9 (Milbrandt-Yau) was APPENDED after its edge closure was
#: read off the scheme itself: MY2 runs a mass-to-number consistency block
#: on entry (module_mp_milbrandt2mom.F:1459-1528, transcribed at
#: woof/core/kernels/milbrandt2.cu:547-600), so the numbers a mixed edge
#: has to diagnose are the numbers the scheme would build from those masses
#: at its own first call -- no intercept is invented and nothing is
#: borrowed from another scheme.  Its qv/qc/qr/qi/qs/qg/qh reuse existing
#: codes and only nc and nh were new, taking 22 and 23, so the append moved
#: nothing.  It is the second DUAL-RIMED selector beside mp=18 (graupel AND
#: hail as separate categories), which is what
#: :data:`_DUAL_RIMED_SELECTORS` below exists to say once.
#:
#: mp_physics=16 (WDM6) and mp_physics=28 (Thompson aerosol-aware) were
#: APPENDED next, on the same discipline and with the same shape of
#: closure: every moment they add is diagnosed by an operation this tree
#: ALREADY ships as the correct one for a laterally forced WDM6 or
#: aerosol-Thompson domain, so nothing here is a new approximation.
#:
#:   * WDM6 entry seeds nc=0, nr=0 and nn=``wdm6_ccn_conc``.  That is
#:     byte for byte what ``ingest/microphysics_cold_start.py`` and
#:     ``ingest/wrfinput.py`` already write for a fresh WDM6 domain, and
#:     ``ccn_conc`` is exactly what WRF's ``flow_dep_bdy_qnn`` feeds in
#:     through an INFLOW face (ported at ``ingest/lateral_bc.py``).  A
#:     one-way nest edge IS a lateral boundary, so the edge and the
#:     boundary now agree instead of the edge refusing what the boundary
#:     does every step.
#:   * Thompson-aerosol entry seeds nr/ni from the same two Thompson
#:     closures the mp=8 edge already runs, and nc/nwfa/nifa from WRF's
#:     own non-aerosol-aware fallbacks (module_mp_thompson.F:1248-1255,
#:     the ELSE branch mp_gt_driver takes when is_aerosol_aware is FALSE),
#:     whose constants are packaged in ``core/thompson_aerosol_contract``.
#:
#: EXIT edges needed no closure at all and were being refused anyway: the
#: target's moments are diagnosed from target mass by the arms already in
#: the kernel and the departing scheme's reservoir/aerosol fields are
#: dropped, which is what ``species_actions`` has always receipted.
PORTED_MP_PHYSICS = (1, 6, 8, 10, 18, 50, 9, 16, 28, 2, 24, 13)

#: Selectors that carry graupel and hail as SEPARATE prognostic categories,
#: so ``qg`` means graupel unambiguously and ``qh`` is its own species.
#: Every other ported scheme has ONE rimed category whose physical meaning
#: is a namelist switch (:func:`_rimed_category`), and the mass mapping
#: turns on which of the two shapes each end of an edge has.  Spelled once,
#: because ``mass_source`` asked ``== 18`` in four places and a second
#: dual-rimed scheme would otherwise have had its hail dropped and its
#: graupel mapped by a rimed-category comparison that returns None for it.
_DUAL_RIMED_SELECTORS = (9, 18, 24)

#: Schemes ArWen has ported but whose MIXED nest edges are refused rather
#: than approximated.  Consulted by :func:`resolve_microphysics_transition`
#: purely so the refusal names a reason instead of falling through to the
#: generic "ported selectors are ..." message, which would read as
#: "the scheme is not implemented" when in fact only the edge closure is
#: missing.
#:
#: WDM5 (14) awaits its five-class entry closure. The mp=50 row left this list
#: when its rime-pair closure was ratified; mp=16 and mp=28 left it
#: together (R-004) when the closures described on
#: :data:`PORTED_MP_PHYSICS` were ratified -- each is an operation this
#: tree already performs on a laterally forced domain of that scheme, so
#: refusing the edge was refusing at a nest boundary what the lateral
#: boundary does every step.  The refusal MACHINERY stays: it is
#: scheme-independent, it is held to its two tables by the import-time
#: check below, and it is how the next scheme with a genuinely missing
#: closure gets named instead of falling into the generic message.
#:
#: mp=3 (WSM3) is a member for a reason of its own: WSM3 has no
#: moments at all; what a mixed edge lacks for it is a PHASE closure.
UNVALIDATED_MIXED_EDGE_SELECTORS: tuple[int, ...] = (3, 5, 14, 15, 26, 27, 29, 30, 38, 40, 52, 55, 56, 95, 97)

_MASS_FIELDS = {
    2: ("qv", "qc", "qr", "qi", "qs", "qg"),
    13: ("qv", "qc", "qr", "qi", "qs"),
    56: ("qv", "qc", "qr", "qi", "qs", "qg", "qh"),
    1: ("qv", "qc", "qr"),
    # Milbrandt-Yau (Registry.EM_COMMON:3025): the six-species set plus a
    # separate hail category, the same mass shape NSSL carries.
    9: ("qv", "qc", "qr", "qi", "qs", "qg", "qh"),
    6: ("qv", "qc", "qr", "qi", "qs", "qg"),
    8: ("qv", "qc", "qr", "qi", "qs", "qg"),
    10: ("qv", "qc", "qr", "qi", "qs", "qg"),
    18: ("qv", "qc", "qr", "qi", "qs", "qg", "qh"),
    # P3 one-category (Registry.EM_COMMON:3038): moist:qv,qc,qr,qi with no
    # qs and no qg.  The rime pair rides _MOMENT_FIELDS below: like NSSL's
    # volume moments it is DIAGNOSED on a mixed edge, never mapped.
    50: ("qv", "qc", "qr", "qi"),
    # WDM6 (Registry.EM_COMMON:3031) carries WSM6's six masses unchanged;
    # only the number moments below are new.
    16: ("qv", "qc", "qr", "qi", "qs", "qg"),
    # Thompson aerosol-aware (Registry.EM_COMMON:3036) carries classic
    # Thompson's six masses unchanged and adds no rimed category.
    28: ("qv", "qc", "qr", "qi", "qs", "qg"),
    # WSM7 (Registry.EM_COMMON:3034): WSM6's six masses plus a separate
    # hail category, no moments.
    24: ("qv", "qc", "qr", "qi", "qs", "qg", "qh"),
}
_MOMENT_FIELDS = {
    2: (),
    13: ("rimi",),
    1: (),
    # A number moment for every one of the six hydrometeors; the WRF
    # driver binds qnc/qnr/qni/qns/qng/qnh at
    # module_microphysics_driver.F:1857-1862 and woof/core/state.py's mp=9
    # arm allocates nc/nr/ni/ns/ng/nh.  ``nr``/``ni``/``ns``/``ng`` already
    # hold codes 6..9; ``nc`` and ``nh`` took 22 and 23.
    9: ("nc", "nr", "ni", "ns", "ng", "nh"),
    6: (),
    24: (),
    8: ("nr", "ni"),
    10: ("nr", "ni", "ns", "ng"),
    18: (
        "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
        "qvolg", "qvolh",
    ),
    50: ("nr", "ni", "qir", "qib"),
    # WDM6: the CCN reservoir and the two warm-rain numbers, spelled
    # ONCE in woof/core/wdm6_constants.WDM6_NUMBER_SPECIES so the
    # allocator, the ring guard and this edge cannot drift apart.
    16: WDM6_NUMBER_SPECIES,
    # Thompson aerosol-aware, in woof/core/state.py's allocation order.
    28: ("nc", "nr", "ni", "nwfa", "nifa"),
}

#: Number moments a mixed edge CARRIES instead of re-diagnosing: rain and
#: cloud-ice numbers between schemes whose rain and cloud ice are both
#: inverse-exponential (mu = 0) -- Morrison (10), Thompson (8) and aerosol
#: Thompson (28).  Between those, the same mass and number describe the same
#: size distribution, so the parent's number IS the child's.
#:
#: THE BREAKAGE THIS PREVENTS (CP-CHILD-PHYSICS-2026-10-24): a 1 km
#: Thompson child of a 3 km Morrison parent re-diagnosed rain number from
#: rain mass; Thompson's closure gave about 400 drops per kg where the parent
#: carried about 3,500, nine times larger drops by volume, so the child's
#: analysis frame drew a 15-28 dBZ blanket over 9,704 columns that the parent
#: does not have (gone by +15 min), and the ring drew 10-25 dBZ strips along
#: the inflow edges for the whole run.
_CARRIED_NUMBER_SCHEMES = frozenset({8, 10, 28})
_CARRIED_NUMBERS = ("nr", "ni")

#: What a mixed edge touching each refused selector WOULD have to move,
#: recorded so the refusal is specific and so a future package does not have
#: to rediscover it.  Every tuple must lead with the moments that ALREADY
#: have a stable host field code, so appending the selector to
#: :data:`PORTED_MP_PHYSICS` extends the code table and reorders nothing.
#:
#: WDM5's number row remains. mp=50 left when its rime-pair closure was ratified;
#: mp=16's and mp=28's left together with theirs.  The moments the two
#: rows described are now REAL rows in :data:`_MOMENT_FIELDS` with real
#: field codes (nn = 24 for WDM6; nwfa = 25 and nifa = 26 for
#: Thompson-aerosol, both schemes' nr/ni/nc reusing 6/7/22), which is what
#: ratification means -- the append discipline this comment describes was
#: followed and no pre-existing code moved.
from woof.core.fast_sbm_contract import BIN_SPECIES as _FSBM_BIN_SPECIES
UNVALIDATED_MIXED_EDGE_MOMENTS: dict[int, tuple[str, ...]] = {
    40: ("nr", "ni", "ns", "ng", "nc"),
    # WSM3 carries no number moment; its refusal is about the phase split
    # of its two condensate masses (see its reason below).
    97: (),
    3: (),
    14: WDM6_NUMBER_SPECIES,
    26: ("qh",) + WDM6_NUMBER_SPECIES,
    27: ("qh",) + WDM6_NUMBER_SPECIES,
    5: ("f_rimef",),
    15: ("qrimef",),
    30: _FSBM_BIN_SPECIES,
    52: ("nc", "ni", "qir", "qib", "ni2", "qir2", "qib2"),
    55: ("nr", "ni", "ni2", "ni3", "qvoli", "qaoli", "qvoli2", "qaoli2", "qvoli3", "qaoli3"),
    29: ("nc", "nr", "ni", "nwfa", "nifa"),
    38: ("ng", "qvolg"),
    # NTU (mp=56): the six numbers lead (nr/ni/ns/ng already hold field
    # codes), then the aerosol budget and the ice volume, shape and
    # moment fields no other scheme carries.
    56: ("nr", "ni", "ns", "ng", "nc", "nh", "qdcn", "qtcn",
         "qccn", "qrcn", "qnin", "fi", "fs", "vi", "vs", "vg",
         "ai", "as_", "ag", "ah", "i3m"),
    95: ("qt", "f_ice_phy", "f_rain_phy", "f_rimef_phy"),
}

#: The scheme's own name, and the paragraph that says why ITS closure is
#: missing.  One row per :data:`UNVALIDATED_MIXED_EDGE_SELECTORS` entry: the
#: refusals are NOT the same refusal, and an operator must never be handed
#: another scheme's reason, another scheme's fallback constants or another
#: scheme's Fortran citation.  Each ``reason`` is a sentence fragment that
#: completes "... but it has no validated cross-scheme entry closure for
#: its moments (...) -- ".
#:
#: WDM5 and MP38 name their missing entry closures below. The two rows that
#: stood here (WDM6's CCN reservoir, Thompson-aerosol's three aerosol
#: numbers) were retired by R-004: both named a seeding operation as
#: unmeasured while the same operation shipped as the correct one on the
#: LATERAL boundary of a domain running that very scheme.  The closures
#: they asked for are now on :data:`PORTED_MP_PHYSICS`, with the WRF lines
#: that define them.
_UNVALIDATED_MIXED_EDGE_REASONS: dict[int, tuple[str, str]] = {
    97: ("Goddard GCE three-ice",
         "gsfcgce_hail selects the meaning of qg and gsfcgce_2ice removes "
         "frozen classes; a mixed edge has no defined mapping for those "
         "switches and could populate a removed class or relabel hail as graupel"),
    55: ("ISHMAEL", "A mixed edge has no closure for both volume moments of all three adaptive-habit categories; copying mass alone would change particle habit and deposition rates"),
    40: ("Morrison aerosol", "Mixed-scheme conversion of the passive aercu_opt=0 QNC tracer has no native closure. Use a same-scheme edge."),
    3: ("WSM3",
        "WSM3 keeps cloud ice inside qc and snow inside qr below 0 C "
        "(phys/module_mp_wsm3.F:172-180, the phase split its own "
        "effectRad_wsm3 reads), so a WSM3 parent's cold qc would enter a "
        "child with separate ice, snow and graupel classes as cloud WATER, "
        "and a six-class parent's qi/qs/qg have no WSM3 field to land in "
        "unless folded into qc/qr; nest_microphysics.cu defines neither "
        "temperature split, so the edge is refused rather than "
        "mis-phased"),
    5: ("Ferrier-Aligo", "qi carries all ice together and f_rimef carries "
        "its fall-speed history. Mapping a separated-ice parent into qi "
        "without a rime-factor closure changes sedimentation. Same-scheme "
        "nests preserve the Ferrier state; mixed edges need that closure."),
    30: ("FAST_SBM", "142 spectral distributions have no validated cross-scheme entry closure; transferring bulk moisture would not specify their shape"),
    14: ("WDM5", "a mixed edge can introduce rain or cloud mass without its transported number, and this five-class entry closure has not been validated"),
    52: ("P3 two-category", "a mixed edge would need a category partition and moment closure for both free ice populations; the one-category closure cannot preserve that inventory"),
    27: ("UDM", "A mixed edge has no validated graupel/hail partition and cloud/rain/CCN number closure."),
    29: ("RCON", "No mixed-scheme entry closure has been qualified for RCON aerosol and hydrometeor number moments."),
    38: ("Thompson graupel-hail", "A cross-scheme edge has no density-consistent graupel number and volume closure."),
    15: ("Ferrier-Aligo advected rime", "no cross-scheme mass-weighted rime "
         "closure; a child would lose QRIMEF and change ice fall speeds"),
    56: ("NTU multi-moment",
         "its pristine-ice, snow, graupel and hail volume, shape and "
         "second/third moments and its five-field aerosol budget have "
         "no ratified entry closure from another scheme's two moments, "
         "and inventing one would hand the child an ice habit and a "
         "CCN cut-off the parent never had; a same-scheme NTU edge "
         "interpolates every field and is not affected"),
    95: ("Eta Ferrier", "no cross-scheme closure for total qt with carried phase and rime fractions"),
}

# P3 with predicted droplet number (mp=51) runs same-scheme nests (an
# mp=51 parent and child carry the same field set, woof/core/nest_fields.py
# forcing nc with the rest).  A MIXED edge into or out of it is refused by
# name: the cross-scheme P3 closure is the ``target_mp == 50`` /
# ``source_mp == 50`` arm of kernels/nest_microphysics.cu, which carries no
# droplet number.  Entering MP51 through it would hand the child cloud water
# with no matching nc (P3 then rebuilds nc from its lambda limiter,
# module_mp_p3.F:6679-6685, a number no scheme predicted); leaving it would
# drop nc where the target carries one (code 22, mp=9/16/28).
UNVALIDATED_MIXED_EDGE_SELECTORS += (51,)
UNVALIDATED_MIXED_EDGE_MOMENTS[51] = ("nc", "nr", "ni", "qir", "qib")
_UNVALIDATED_MIXED_EDGE_REASONS[51] = (
    "P3 one-category, predicted droplet number",
    "the P3 entry/exit closure of kernels/nest_microphysics.cu (its "
    "mp == 50 arms) carries no droplet number, so an edge into MP51 would "
    "seed nc from P3's lambda limiter rather than from any predicted "
    "number and an edge out of it would drop nc")

# Named schemes (woof.microphysics_schemes) run same-scheme nests; a mixed
# edge into or out of one has no validated entry closure for its moments
# yet, so it is refused by name with the scheme's own moment list.
from woof.microphysics_schemes import NAMED_SCHEMES as _NAMED_MP_SCHEMES
for _s in _NAMED_MP_SCHEMES.values():
    UNVALIDATED_MIXED_EDGE_SELECTORS += (_s.mp_id,)
    UNVALIDATED_MIXED_EDGE_MOMENTS[_s.mp_id] = _s.moment_species + _s.aerosol_species
    _UNVALIDATED_MIXED_EDGE_REASONS[_s.mp_id] = (
        _s.name.upper(),
        "a named WOOF scheme whose cross-scheme moment closure (graupel "
        "number and volume included) has not been written or measured")

# P3 three-moment ice (mp=53) runs same-scheme nests (the generic SINT rows
# carry its nc and qzi, woof/ingest/nest_init.py).  A MIXED edge into it
# would have to diagnose the sixth ice moment Z -- carried as qzi =
# (N*Z)**0.5 -- and a prognostic droplet number from a parent that has
# neither, and no WRF operation defines that closure: real.exe seeds no P3
# moment at all, and a qzi of zero is not neutral, because p3_main floors Z
# at zsmall and then clamps it to the mu_i = 20 bound (module_mp_p3.F:2612,
# :2654-2660), so every child ice PSD would enter at the narrowest shape
# the table allows.  An edge OUT of it would drop Z silently from a
# parent-side receipt that never names it.  Refused by name until that
# closure is written and measured.
UNVALIDATED_MIXED_EDGE_SELECTORS += (53,)
UNVALIDATED_MIXED_EDGE_MOMENTS[53] = ("nr", "ni", "qir", "qib", "nc", "qzi")
_UNVALIDATED_MIXED_EDGE_REASONS[53] = (
    "P3 three-moment ice",
    "the edge would have to invent the sixth ice moment (qzi) and a "
    "prognostic droplet number from a parent that carries neither, and a "
    "zero qzi puts every child ice PSD at the mu_i = 20 bound on its first "
    "call (module_mp_p3.F:2612, :2654-2660)")

# A selector may not join UNVALIDATED_MIXED_EDGE_SELECTORS without bringing
# its own moments and its own sentence.  Adding 16 to the selector tuple and
# not to the moments table turned the named refusal into a bare KeyError(16),
# with every advertised gate still claiming the refusal worked; this check
# turns that into an import-time failure that no code path can reach past.
_UNVALIDATED_MIXED_EDGE_REASONS[26] = ("WDM7", "a mixed edge needs a conservative graupel/hail partition and hail exit, which are not implemented")

_missing_edge_rows = sorted(
    mp for mp in UNVALIDATED_MIXED_EDGE_SELECTORS
    if mp not in UNVALIDATED_MIXED_EDGE_MOMENTS
    or mp not in _UNVALIDATED_MIXED_EDGE_REASONS
)
if _missing_edge_rows:
    raise RuntimeError(
        "UNVALIDATED_MIXED_EDGE_SELECTORS entries without a moments row and "
        f"a scheme-specific reason: {_missing_edge_rows}; the refusal in "
        "resolve_microphysics_transition would raise KeyError instead of "
        "naming the scheme")
_TARGET_FIELDS = {
    mp: _MASS_FIELDS[mp] + _MOMENT_FIELDS[mp]
    for mp in PORTED_MP_PHYSICS
}
_ALL_EDGE_FIELDS = tuple(dict.fromkeys(
    name
    for mp in PORTED_MP_PHYSICS
    for name in _TARGET_FIELDS[mp]
))
#: The stable host field codes ``kernels/nest_microphysics.cu`` switches on:
#:
#:   qv/qc/qr/qi/qs/qg = 0..5, nr/ni/ns/ng = 6..9, qh = 10,
#:   qndrop/qnr/qni/qns/qng/qnh/qnn/qvolg/qvolh = 11..19,
#:   qir = 20, qib = 21   (mp_physics=50, allocated with its ratification),
#:   nc = 22, nh = 23     (mp_physics=9, allocated with its ratification),
#:   nn = 24              (mp_physics=16, allocated with its ratification),
#:   nwfa = 25, nifa = 26 (mp_physics=28, allocated with its ratification).
#:
#: 22..26 were allocated by APPENDING 9, then 16, then 28 to
#: :data:`PORTED_MP_PHYSICS`, so every code below kept the value it had --
#: the discipline mp=50's ratification set.  ``nc`` is ONE field name and
#: therefore ONE code (22), shared by mp=9, mp=16 and mp=28, which is why
#: WDM6 adds only ``nn`` and Thompson-aerosol only ``nwfa``/``nifa``.
#: mp=16's masses reuse 0..5 and its nr reuses 6; mp=28's masses reuse
#: 0..5 and its nr/ni reuse 6/7.  This block and the kernel's own comment
#: decode the same table; the kernel is the consumer, so changing one
#: without the other is the defect the append rule exists to prevent.
_EDGE_FIELD_CODES = {
    name: code for code, name in enumerate(_ALL_EDGE_FIELDS)
}
_SOURCE_MASS_CODES = {
    name: code
    for code, name in enumerate(("qv", "qc", "qr", "qi", "qs", "qg", "qh"))
}

# Kept as the exact original MP8->MP18 field-code table.  The content identity
# test and the ratified CUDA entry point both intentionally bind this object.
_MP18_FIELDS = _TARGET_FIELDS[18]
_FIELD_CODES = {name: code for code, name in enumerate(_MP18_FIELDS)}
_SOURCE_MASS_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg")
_DIAGNOSED_FIELDS = ("qndrop", "qnr", "qni", "qns", "qng", "qnn", "qvolg")
_ZEROED_FIELDS = ("qh", "qnh", "qvolh")
#: Child-local state a microphysics-scheme boundary resets rather than
#: inherits.  ``rthften``/``rqvften`` -- the dycore's exported advective
#: forcing pair -- are DELIBERATELY ABSENT and the reason is the membership
#: rule: this tuple is the state the DEPARTING SCHEME owned, and a
#: scheme boundary invalidates a closure's retained products (h_diabatic's
#: latent heating, the held rates, the accumulators) but says nothing about
#: the dynamics.  The advective theta/qv rates are a dycore product,
#: identical whichever microphysics ran, so they interpolate from the
#: parent with every other transported field (woof/ingest/nest_init.py).
_RESET_FIELDS = (
    "h_diabatic", "held_tendencies", "precipitation_accumulators",
)
_IGNORED_SOURCE_FIELDS = ("nr", "ni")
_CANONICALIZATION_THRESHOLDS = {
    "cxmin_number_per_m3": 1.0e-8,
    "initial_mass_kg_per_kg": 1.0e-8,
    "cloud_ice_snow_mass_kg_per_kg": 1.0e-13,
    "rain_graupel_mass_kg_per_kg": 1.0e-12,
    "subthreshold_mass_action": "return_to_vapor",
}
_THREADS = 256


def _rimed_category(cfg) -> str | None:
    """Return the physical meaning of a scheme's single ``qg`` category."""

    mp = int(getattr(cfg, "mp_physics", 0))
    if mp in (8, 28):
        # Thompson's single rimed category is graupel in both entries:
        # module_mp_thompson.F declares one rho_g/am_g/bm_g set used by the
        # classic and the aerosol-aware path alike, and mp=28 adds no hail
        # category.
        return "graupel"
    if mp == 2:
        return "graupel"
    if mp == 6:
        return "hail" if int(getattr(cfg, "wsm6_hail_opt", 0)) else "graupel"
    if mp == 16:
        # WDM6 carries WSM6's rimed category and WSM6's hail_opt arm
        # (module_mp_wdm6.F:2096-2108 sets the same five constants), under
        # its own RunConfig field because the two schemes' namelist knob is
        # read inside the scheme it belongs to.
        return "hail" if int(getattr(cfg, "wdm6_hail_opt", 0)) else "graupel"
    if mp == 10:
        return "hail" if int(getattr(cfg, "morr_rimed_ice", 1)) else "graupel"
    return None


@dataclass(frozen=True)
class MicrophysicsTransitionContract:
    """Resolved directed parent-to-child microphysics policy."""

    source_mp_physics: int
    target_mp_physics: int
    policy_id: str
    mixed: bool
    source_rimed_category: str | None = None
    target_rimed_category: str | None = None
    target_morrison_rimed_density: float = 900.0
    #: The child's ``wdm6_ccn_conc``, carried so the WDM6 entry closure
    #: seeds the reservoir from the DOMAIN's own value rather than from a
    #: constant restated in the kernel.  Inert unless the target is mp=16.
    target_wdm6_ccn_conc: float = 1.0e8
    #: The target NSSL domain's concentration. Its entry closure uses the
    #: same REAL ABS(cccn)/1.225 seed as its cold-start and inflow paths.
    target_nssl_cccn: float = 0.5e9
    target_nssl_shapes: tuple[float, float] = (0.0, 1.0)
    #: Which way this contract runs, in words, for the receipt.  A FORCE
    #: edge diagnoses on the parent and interpolates; a FEEDBACK edge
    #: diagnoses on the child and restricts.  The kernel is the same and is
    #: column-local either way; what differs is whose grid it runs on and
    #: what the spatial operator after it does.
    translation_order: str = TRANSITION_ORDER
    #: Whether this edge carries the parent's rain and cloud-ice numbers
    #: (:data:`_CARRIED_NUMBER_SCHEMES`).  The carry is part of correct-mode
    #: finding HN1 (key ``hydrometeor_handover_numbers``, read off the
    #: TARGET domain's configuration by the resolver): exact mode switches
    #: it off and the edge re-diagnoses both numbers from mass, as 2.9.6
    #: did, so an exact-mode mixed edge is byte-identical to 2.9.6.
    carry_numbers: bool = True

    def number_source(self, target_field: str) -> str | None:
        """The source number a mixed edge carries into ``target_field``.

        ``None`` when the number is diagnosed from mass (or seeded) instead:
        see :data:`_CARRIED_NUMBER_SCHEMES` for the edges that carry, and
        :attr:`carry_numbers` for the exact-mode switch.
        """

        if (self.mixed and self.carry_numbers
                and target_field in _CARRIED_NUMBERS
                and self.source_mp_physics in _CARRIED_NUMBER_SCHEMES
                and self.target_mp_physics in _CARRIED_NUMBER_SCHEMES
                and target_field in _MOMENT_FIELDS[self.source_mp_physics]
                and target_field in _MOMENT_FIELDS[self.target_mp_physics]):
            return target_field
        return None

    def mass_source(self, target_field: str) -> str | None:
        """Source mass field for one target mass, or ``None`` for default."""

        if target_field not in _MASS_FIELDS[self.target_mp_physics]:
            raise ValueError(
                f"{target_field!r} is not a target mass field for "
                f"MP{self.target_mp_physics}")
        if target_field in ("qv", "qc", "qr"):
            return target_field
        if target_field in ("qi", "qs"):
            return (target_field
                    if target_field in _MASS_FIELDS[self.source_mp_physics]
                    else None)
        if target_field == "qg":
            if self.target_mp_physics in _DUAL_RIMED_SELECTORS:
                if self.source_mp_physics in _DUAL_RIMED_SELECTORS:
                    # Both ends name graupel and hail separately, so the
                    # two categories map straight across.
                    return "qg"
                return ("qg"
                        if self.source_rimed_category == "graupel" else None)
            if self.source_mp_physics in _DUAL_RIMED_SELECTORS:
                return ("qh" if self.target_rimed_category == "hail"
                        else "qg")
            return ("qg"
                    if self.source_rimed_category
                    == self.target_rimed_category else None)
        if target_field == "qh":
            if self.source_mp_physics in _DUAL_RIMED_SELECTORS:
                return "qh"
            return ("qg" if self.source_rimed_category == "hail" else None)
        raise AssertionError(target_field)

    def species_actions(self) -> tuple[Mapping[str, object], ...]:
        """One explicit receipt row for every target and dropped source."""

        rows: list[Mapping[str, object]] = []
        consumed: set[str] = set()
        # The two P3 special shapes.  Entering P3 (target 50), every frozen
        # source species is MERGED into the single ice category rather than
        # mapped one-to-one; leaving P3 (source 50), qi/qs/qg are all cut
        # from the single category by rime state, provided the target has an
        # ice inventory at all (a Kessler child drops frozen mass exactly as
        # it does from any other parent).
        p3_entry = self.mixed and self.target_mp_physics == 50
        p3_exit = (
            self.mixed and self.source_mp_physics == 50
            and "qi" in _MASS_FIELDS[self.target_mp_physics])
        for target in _MASS_FIELDS[self.target_mp_physics]:
            if p3_entry and target == "qi":
                frozen = [name for name in ("qi", "qs", "qg", "qh")
                          if name in _MASS_FIELDS[self.source_mp_physics]]
                if not frozen:
                    rows.append({
                        "action": "defaulted",
                        "source_field": None,
                        "target_field": target,
                        "reason": "source_species_absent",
                        "default": "zero_mass_mixing_ratio",
                    })
                for name in frozen:
                    consumed.add(name)
                    rows.append({
                        "action": "mapped",
                        "source_field": name,
                        "target_field": "qi",
                        "reason": "merged_into_p3_single_ice_category",
                    })
                continue
            if p3_exit and target in ("qi", "qs", "qg"):
                consumed.add("qi")
                rows.append({
                    "action": "mapped",
                    "source_field": "qi",
                    "target_field": target,
                    "reason": {
                        "qi": "p3_unrimed_ice_after_rime_split",
                        "qs": "p3_rime_split_fresh_snow_fraction",
                        "qg": "p3_rime_split_dense_rime_fraction",
                    }[target],
                })
                continue
            source = self.mass_source(target)
            if source is None:
                rows.append({
                    "action": "defaulted",
                    "source_field": None,
                    "target_field": target,
                    "reason": "source_species_absent",
                    "default": "zero_mass_mixing_ratio",
                })
            else:
                consumed.add(source)
                rows.append({
                    "action": "mapped",
                    "source_field": source,
                    "target_field": target,
                    "reason": "shared_physical_mass_species",
                })
        carried: set[str] = set()
        for target in _MOMENT_FIELDS[self.target_mp_physics]:
            source = self.number_source(target)
            if source is not None:
                carried.add(source)
                rows.append({
                    "action": "mapped",
                    "source_field": source,
                    "target_field": target,
                    "reason": "shared_exponential_number_moment",
                })
                continue
            row = {
                "action": "diagnosed",
                "source_field": None,
                "target_field": target,
                "reason": (
                    "p3_rime_state_diagnosed_from_source_frozen_species"
                    if p3_entry and target in ("qir", "qib")
                    else "target_scheme_mass_moment_closure"),
            }
            # The seeded entries name their VALUE and its authority, so a
            # receipt reader can see what a mixed edge put in the child
            # rather than only that something was diagnosed.
            if self.mixed and self.target_mp_physics == 13:
                row["reason"] = "wrf_sbu_ylin_cloud_snow_riming_diagnosis"
                row["units"] = "fraction"
                row["source"] = "WRF 4.6.1 module_mp_sbu_ylin.F:500-520"
            elif self.mixed and self.target_mp_physics == 16:
                row["reason"] = _WDM6_EDGE_ENTRY_REASONS[target]
                row["seeded_value"] = (
                    self.target_wdm6_ccn_conc if target == "nn"
                    else 0.0)
                row["units"] = ("number_per_m3" if target == "nn"
                                else "number_per_kg")
            elif self.mixed and self.target_mp_physics == 28 and (
                    target in _MP28_EDGE_ENTRY_REASONS):
                row["reason"] = _MP28_EDGE_ENTRY_REASONS[target]
                row["seeded_value"] = _MP28_EDGE_ENTRY_VALUES[target]
                row["units"] = "number_per_m3_divided_by_air_density"
            rows.append(row)
        for source in _MASS_FIELDS[self.source_mp_physics]:
            if source not in consumed:
                rows.append({
                    "action": "dropped",
                    "source_field": source,
                    "target_field": None,
                    "reason": "target_species_absent_or_rimed_category_mismatch",
                })
        for source in _MOMENT_FIELDS[self.source_mp_physics]:
            if p3_exit and source in ("qir", "qib"):
                rows.append({
                    "action": "mapped",
                    "source_field": source,
                    "target_field": "qg",
                    "reason": (
                        "p3_rime_mass_partitioned_into_snow_and_graupel"
                        if source == "qir"
                        else "p3_rime_volume_sets_partition_weight"),
                })
                continue
            if source in carried:
                continue
            rows.append({
                "action": "dropped",
                "source_field": source,
                "target_field": None,
                "reason": "scheme_closure_mismatch",
            })
        return tuple(rows)

    def receipt(self) -> Mapping[str, object]:
        if not self.mixed:
            return {
                "policy_id": self.policy_id,
                "source_mp_physics": self.source_mp_physics,
                "target_mp_physics": self.target_mp_physics,
                "mixed": False,
                "stock_wrf_equivalent": True,
            }
        species = self.species_actions()
        counts = Counter(str(row["action"]) for row in species)
        receipt: dict[str, object] = {
            "policy_id": self.policy_id,
            "source_mp_physics": self.source_mp_physics,
            "target_mp_physics": self.target_mp_physics,
            "mixed": True,
            "stock_wrf_equivalent": False,
            "stock_wrf_reason": (
                "WRF v4.6.1 normalizes all domains to the innermost "
                "mp_physics selector"
            ),
            "translation_order": self.translation_order,
            "source_rimed_category": self.source_rimed_category,
            "target_rimed_category": self.target_rimed_category,
            "species_actions": [dict(row) for row in species],
            "species_action_counts": {
                action: int(counts.get(action, 0))
                for action in ("mapped", "defaulted", "diagnosed", "dropped")
            },
            "reset_child_local_fields": list(_RESET_FIELDS),
            "implementation": transition_implementation_identity(),
        }
        # Preserve every established receipt field for the ratified pair.
        if ((self.source_mp_physics, self.target_mp_physics)
                == (8, 18)):
            receipt.update({
                "canonicalized_source_mass_fields": list(_SOURCE_MASS_FIELDS),
                "diagnosed_from_mass_fields": list(_DIAGNOSED_FIELDS),
                "zeroed_fields": list(_ZEROED_FIELDS),
                "field_actions": {
                    **{
                        name:
                        "canonicalize_from_source_mass_then_interpolate"
                        for name in _SOURCE_MASS_FIELDS
                    },
                    **{
                        name: "diagnose_from_source_mass_then_interpolate"
                        for name in _DIAGNOSED_FIELDS
                    },
                    **{
                        name: "zero_then_interpolate"
                        for name in _ZEROED_FIELDS
                    },
                },
                "ignored_source_fields": {
                    name: "ignored_due_to_scheme_closure_mismatch"
                    for name in _IGNORED_SOURCE_FIELDS
                },
                "nssl2_background_ccn_per_kg": float(
                    _nssl2_background_ccn(self.target_nssl_cccn)),
                "canonicalization_thresholds": dict(
                    _CANONICALIZATION_THRESHOLDS),
            })
        if self.target_mp_physics == 18 and self.target_nssl_cccn != 0.5e9:
            receipt["nssl2_cccn"] = self.target_nssl_cccn
            receipt["nssl2_background_ccn_per_kg"] = float(
                _nssl2_background_ccn(self.target_nssl_cccn))
        if self.target_mp_physics == 18 and self.target_nssl_shapes != (0.0, 1.0):
            receipt["nssl2_shapes"] = list(self.target_nssl_shapes)
        # Every edge touching P3 records the closure it executes, the same
        # way the ratified pair records its constants: the diagnosis is a
        # DEFINED ArWen behaviour where WRF defines nothing, so the receipt
        # carries the exact constants rather than pointing at source.
        if 50 in (self.source_mp_physics, self.target_mp_physics):
            receipt["p3_edge"] = {
                "direction": (
                    "enter" if self.target_mp_physics == 50 else "leave"),
                "fresh_snow_rime_density_kg_m3":
                    P3_EDGE_FRESH_SNOW_RIME_DENSITY,
                "dense_rime_density_kg_m3": P3_EDGE_DENSE_RIME_DENSITY,
                "rain_number_per_rain_mass":
                    P3_EDGE_RAIN_NUMBER_PER_RAIN_MASS,
                "ice_nucleation_mass_kg": P3_EDGE_ICE_NUCLEATION_MASS_KG,
                "max_total_ice_number_per_m3": P3_EDGE_MAX_TOTAL_NI_PER_M3,
                "qsmall_kg_per_kg": P3_EDGE_QSMALL_KG_PER_KG,
                "rime_density_bounds_kg_m3":
                    list(P3_RIME_DENSITY_BOUNDS_KG_M3),
                "mass_conserving": True,
                "degenerate_cases_exact": ["zero_ice", "zero_rime"],
            }
        return receipt


def transition_implementation_identity() -> Mapping[str, str]:
    """Content identity for the executable conversion policy."""

    kernel = Path(__file__).with_name("kernels") / "nest_microphysics.cu"
    kernel_sha = _canonical_source_sha256(kernel)
    driver_sha = _canonical_source_sha256(Path(__file__))
    live_coupler_sha = _canonical_source_sha256(
        Path(__file__).with_name("nest.py"))
    parent_init_sha = _canonical_source_sha256(
        Path(__file__).parents[1] / "ingest" / "nest_init.py")
    metadata = {
        "policy_ids": [MP8_TO_MP18_POLICY, EDGE_MATRIX_POLICY],
        "translation_order": TRANSITION_ORDER,
        "ratified_field_codes": dict(_FIELD_CODES),
        "matrix_field_codes": dict(_EDGE_FIELD_CODES),
        "mass_fields": {str(k): list(v) for k, v in _MASS_FIELDS.items()},
        "moment_fields": {str(k): list(v) for k, v in _MOMENT_FIELDS.items()},
        "background_ccn_per_kg": NSSL2_BACKGROUND_CCN_PER_KG,
        "canonicalization_thresholds": dict(_CANONICALIZATION_THRESHOLDS),
        "p3_edge_constants": {
            "fresh_snow_rime_density_kg_m3": P3_EDGE_FRESH_SNOW_RIME_DENSITY,
            "dense_rime_density_kg_m3": P3_EDGE_DENSE_RIME_DENSITY,
            "rain_number_per_rain_mass": P3_EDGE_RAIN_NUMBER_PER_RAIN_MASS,
            "ice_nucleation_mass_kg": P3_EDGE_ICE_NUCLEATION_MASS_KG,
            "max_total_ice_number_per_m3": P3_EDGE_MAX_TOTAL_NI_PER_M3,
            "qsmall_kg_per_kg": P3_EDGE_QSMALL_KG_PER_KG,
            "rime_density_bounds_kg_m3": list(P3_RIME_DENSITY_BOUNDS_KG_M3),
        },
        "wdm6_edge_constants": {
            "entry_cloud_number": WDM6_EDGE_ENTRY_CLOUD_NUMBER,
            "entry_rain_number": WDM6_EDGE_ENTRY_RAIN_NUMBER,
            "entry_reservoir": "domain wdm6_ccn_conc",
        },
        "mp28_edge_constants": {
            "entry_cloud_number_per_m3":
                MP28_EDGE_ENTRY_CLOUD_NUMBER_PER_M3,
            "entry_nwfa_per_m3": MP28_EDGE_ENTRY_NWFA_PER_M3,
            "entry_nifa_per_m3": MP28_EDGE_ENTRY_NIFA_PER_M3,
        },
    }
    payload = json.dumps(
        {
            "driver_sha256": driver_sha,
            "kernel_sha256": kernel_sha,
            "live_coupler_sha256": live_coupler_sha,
            "parent_init_sha256": parent_init_sha,
            "metadata": metadata,
        },
        sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    return {
        "driver_sha256": driver_sha,
        "kernel_sha256": kernel_sha,
        "live_coupler_sha256": live_coupler_sha,
        "parent_init_sha256": parent_init_sha,
        "contract_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _p3_edge_bulk_rho_rime(qi_tot, qi_rim, bi_rim):
    """``calc_bulkRhoRime`` (module_mp_p3.F:6784-6830) in float32.

    P3's own admission function for a (qitot, qirim, birim) triple: rime
    volume below 1e-15 zeroes the pair, the derived density clamps to
    [50, 900] kg m-3, rime mass clamps to total mass, and rime mass below
    qsmall zeroes the pair.  Returns ``(qi_rim, bi_rim, rho_rime)``.  Both
    edge directions run it so ``qirim <= qitot`` and the density bound hold
    BY CONSTRUCTION on every output.
    """

    f = np.float32
    qi_tot, qi_rim, bi_rim = f(qi_tot), f(qi_rim), f(bi_rim)
    rho_rime = f(0.0)
    if bi_rim >= f(1.0e-15):
        rho_rime = f(qi_rim / bi_rim)
        if rho_rime < f(P3_RIME_DENSITY_BOUNDS_KG_M3[0]):
            rho_rime = f(P3_RIME_DENSITY_BOUNDS_KG_M3[0])
            bi_rim = f(qi_rim / rho_rime)
        elif rho_rime > f(P3_RIME_DENSITY_BOUNDS_KG_M3[1]):
            rho_rime = f(P3_RIME_DENSITY_BOUNDS_KG_M3[1])
            bi_rim = f(qi_rim / rho_rime)
    else:
        qi_rim = f(0.0)
        bi_rim = f(0.0)
        rho_rime = f(0.0)
    if qi_rim > qi_tot and rho_rime > f(0.0):
        qi_rim = qi_tot
        bi_rim = f(qi_rim / rho_rime)
    if qi_rim < f(P3_EDGE_QSMALL_KG_PER_KG):
        qi_rim = f(0.0)
        bi_rim = f(0.0)
    return qi_rim, bi_rim, rho_rime


def p3_edge_entry_reference(qv, qc, qr, qi, qs, qg, qh, inv_rho):
    """Float32 CPU mirror of the kernel's ``target_mp == 50`` arm, scalar.

    Entering P3 from a six-species parent (absent species passed as 0):

    * ``qi_child = qi + qs + qg + qh`` -- mass-conserving merge; below P3's
      own qsmall the merged ice returns to vapor exactly as p3_main does
      (woof/core/p3.py:1616-1622).
    * ``qir = qs + qg + qh`` and ``qib = qs/100 + (qg+qh)/400`` -- the rime
      pair diagnosed at the named densities, then passed through P3's own
      ``calc_bulkRhoRime``; ``qirim <= qitot`` holds by construction and the
      derived density lies in [100, 400], inside P3's [50, 900].
    * ``nr``/``ni`` diagnosed from target mass with P3's own entry closure
      constants (see ``P3_EDGE_RAIN_NUMBER_PER_RAIN_MASS`` and
      ``P3_EDGE_ICE_NUCLEATION_MASS_KG``).

    ``inv_rho`` is 1/rho, i.e. the ``alt`` value the kernel reads.  Returns
    a dict of the eight P3 target fields.  The kernel mirrors this function
    operation for operation with round-to-nearest intrinsics, so the GPU
    shard's equivalence test compares bitwise.
    """

    f = np.float32
    qv, qc, qr = f(qv), f(qc), f(qr)
    qi, qs, qg, qh = f(qi), f(qs), f(qg), f(qh)
    inv_rho = f(inv_rho)
    qsmall = f(P3_EDGE_QSMALL_KG_PER_KG)
    vapor = qv
    frozen_dense = f(qg + qh)
    qitot = f(f(qi + qs) + frozen_dense)
    qirim = f(qs + frozen_dense)
    birim = f(f(qs / f(P3_EDGE_FRESH_SNOW_RIME_DENSITY))
              + f(frozen_dense / f(P3_EDGE_DENSE_RIME_DENSITY)))
    if qitot < qsmall:
        vapor = f(vapor + qitot)
        qitot = f(0.0)
        qirim = f(0.0)
        birim = f(0.0)
    qirim, birim, _rho = _p3_edge_bulk_rho_rime(qitot, qirim, birim)
    if qr < qsmall:
        rain_number = f(0.0)
    else:
        rain_number = f(qr * f(P3_EDGE_RAIN_NUMBER_PER_RAIN_MASS))
    if qitot < qsmall:
        ice_number = f(0.0)
    else:
        ice_number = min(
            f(qitot / f(P3_EDGE_ICE_NUCLEATION_MASS_KG)),
            f(f(P3_EDGE_MAX_TOTAL_NI_PER_M3) * inv_rho))
    return {
        "qv": vapor, "qc": qc, "qr": qr, "qi": qitot,
        "nr": rain_number, "ni": ice_number, "qir": qirim, "qib": birim,
    }


def p3_edge_exit_reference(qitot, qirim, birim):
    """Float32 CPU mirror of the kernel's ``source_mp == 50`` split, scalar.

    Leaving P3, the single ice category splits back into (qi, qs, qg):

    * inputs floor at zero (a defined guard for adversarial state), then
      P3's own ``calc_bulkRhoRime`` canonicalizes the triple;
    * unrimed mass ``qitot - qirim`` becomes qi -- exact when rime is zero;
    * rimed mass partitions between qs and qg so that BOTH the rime mass
      and the rime volume are conserved against the same two densities the
      entry diagnosis used: ``qg = (qirim - 100*qib) * 400/300`` clamped to
      [0, qirim], ``qs = qirim - qg``.  A pure fresh-snow rime density
      (100) yields all snow, a pure dense-rime density (400) all graupel,
      and a two-density mix inverts the entry diagnosis exactly (up to
      float32 rounding).

    Returns ``(qi, qs, qg)`` with ``qi + qs + qg == qitot`` by algebra
    (each term is a telescoping difference).  Zero ice and zero rime are
    exact.  The kernel mirrors this operation for operation.
    """

    f = np.float32
    qitot = max(f(qitot), f(0.0))
    qirim = max(f(qirim), f(0.0))
    birim = max(f(birim), f(0.0))
    qirim, birim, _rho = _p3_edge_bulk_rho_rime(qitot, qirim, birim)
    graupel = f(0.0)
    if qirim > f(0.0):
        rho_s = f(P3_EDGE_FRESH_SNOW_RIME_DENSITY)
        rho_g = f(P3_EDGE_DENSE_RIME_DENSITY)
        scale = f(rho_g / f(rho_g - rho_s))
        graupel = f(f(qirim - f(rho_s * birim)) * scale)
        graupel = min(max(graupel, f(0.0)), qirim)
    snow = f(qirim - graupel)
    ice = f(qitot - qirim)
    return ice, snow, graupel


def _canonical_source_sha256(path: Path) -> str:
    """Hash source text independently of checkout newline policy."""

    canonical = path.read_text(encoding="utf-8").replace(
        "\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def resolve_microphysics_transition(
        parent_cfg, child_cfg) -> MicrophysicsTransitionContract:
    """Resolve one ported ordered edge or fail closed.

    A same-scheme edge resolves for ANY ported ``mp_physics`` before the
    mixed-edge matrix is consulted.  Every MIXED pair drawn from
    :data:`PORTED_MP_PHYSICS` now resolves too, mp=16 and mp=28 included
    (R-004): their entry closures are the documented ones on that tuple,
    and their exit edges never needed a closure at all.  The named-refusal
    branch below survives for the next scheme whose closure is genuinely
    missing; its two tables are empty today.

    THE DEFAULT IS THE EDGE THE PAIR TAKES.  ``nest_microphysics_transition``
    left at its default (``same-scheme-only``) resolves a mixed edge to the
    one closure this matrix defines for it, with the receipt recording the
    requested and the effective policy side by side (woof.core.nest).
    Until 2.7.3 the default REFUSED every mixed edge for want of the key,
    which put an opt-in string in front of a working correctness path.
    Naming a policy still means something: the id the pair takes is
    admitted, and the OTHER mixed id is a contradiction and is refused.
    """

    source = int(getattr(parent_cfg, "mp_physics", 0))
    target = int(getattr(child_cfg, "mp_physics", 0))
    policy = str(getattr(
        child_cfg, "nest_microphysics_transition", SAME_SCHEME_POLICY))
    if source == target:
        if policy != SAME_SCHEME_POLICY:
            raise ValueError(
                f"nest microphysics edge MP{source}->MP{target} is already "
                f"same-scheme and must use {SAME_SCHEME_POLICY!r}, got "
                f"{policy!r}")
        return MicrophysicsTransitionContract(
            source_mp_physics=source, target_mp_physics=target,
            policy_id=policy, mixed=False)

    from woof.physics_registry import column_microphysics_binding
    for mp in (source, target):
        binding = column_microphysics_binding(mp)
        if binding is not None and binding.get('mixed_edge_refusal'):
            raise ValueError(f'mixed nest microphysics edge MP{source}->MP{target}: '
                             + binding['mixed_edge_refusal'])

    # NAMED refusal before the generic "not a ported selector" message,
    # for a scheme that IS ported but whose cross-scheme entry closure is
    # missing: an operator who reads "ported selectors are (...)" would
    # reasonably conclude the scheme itself is unavailable.  The message
    # body is looked up per scheme, never shared -- two closures are never
    # missing for the same reason.  No scheme is in this state today.
    unvalidated = sorted(
        {source, target} & set(UNVALIDATED_MIXED_EDGE_SELECTORS))
    if unvalidated:
        mp = unvalidated[0]
        moments = (", ".join(UNVALIDATED_MIXED_EDGE_MOMENTS[mp])
                   or "none: its two condensate classes change phase at "
                   "0 C")
        scheme, reason = _UNVALIDATED_MIXED_EDGE_REASONS[mp]
        raise ValueError(
            f"mixed nest microphysics edge MP{source}->MP{target} is REFUSED: "
            f"MP{mp} ({scheme}) is ported and runs, but it has "
            f"no defined cross-scheme entry mapping "
            f"(number moments: {moments or 'none'}) because {reason}. "
            f"Configure both domains with "
            f"mp_physics={mp}, or keep MP{mp} on a single domain.")

    if source not in PORTED_MP_PHYSICS or target not in PORTED_MP_PHYSICS:
        raise ValueError(
            f"unsupported mixed nest microphysics edge MP{source}->MP{target}; "
            f"ported selectors are {PORTED_MP_PHYSICS}")
    required_policy = (
        MP8_TO_MP18_POLICY if (source, target) == (8, 18)
        else EDGE_MATRIX_POLICY
    )
    if policy == SAME_SCHEME_POLICY:
        # The unset default: the edge resolves to the closure this pair
        # takes.  The coupler's receipt carries requested_policy beside
        # effective_policy, so a reader sees that nothing was named.
        policy = required_policy
    elif policy != required_policy:
        raise ValueError(
            f"MP{source}->MP{target} takes nest_microphysics_transition="
            f"{required_policy!r}; {policy!r} is the closure of another "
            f"edge. Set {required_policy!r}, or leave the key out and the "
            "edge resolves to it.")
    missing = []
    for role, cfg in (("parent", parent_cfg), ("child", child_cfg)):
        if not bool(cfg.moist):
            missing.append(f"{role}.moist=true")
        if not bool(cfg.moist_cq):
            missing.append(f"{role}.moist_cq=true")
    if missing:
        raise ValueError(
            f"MP{source}->MP{target} nest transition requires the validated "
            "moist/CQ contract on both domains; missing " + ", ".join(missing))
    target_density = (
        400.0 if target == 10
        and int(getattr(child_cfg, "morr_rimed_ice", 1)) == 0
        else 900.0
    )
    # HN1: the target domain's key decides whether the edge carries numbers
    # (off in exact mode and in a strict WRF process).
    from woof.ingest.handover_numbers import closure_active
    return MicrophysicsTransitionContract(
        source_mp_physics=source, target_mp_physics=target,
        policy_id=policy, mixed=True,
        carry_numbers=closure_active(child_cfg),
        source_rimed_category=_rimed_category(parent_cfg),
        target_rimed_category=_rimed_category(child_cfg),
        target_morrison_rimed_density=target_density,
        target_wdm6_ccn_conc=float(
            getattr(child_cfg, "wdm6_ccn_conc", 1.0e8)),
        target_nssl_cccn=(float(_nssl2_concentration(
            getattr(child_cfg, "nssl_cccn", 0.5e9))) if target == 18 else 0.5e9),
        target_nssl_shapes=(float(getattr(child_cfg,"nssl_alphah",0.0)),
                            float(getattr(child_cfg,"nssl_alphahl",1.0))),
    )


def resolve_reverse_microphysics_transition(
        parent_cfg, child_cfg, *,
        policy: str | None = None) -> MicrophysicsTransitionContract:
    """Resolve the FEEDBACK edge: the CHILD's scheme into the PARENT's.

    Same matrix, opposite order.  The ordered pair is
    ``(child.mp_physics -> parent.mp_physics)``, which is the reverse of the
    edge :func:`resolve_microphysics_transition` resolves for the same two
    domains, and every mixed pair drawn from :data:`PORTED_MP_PHYSICS`
    resolves in both directions because the matrix is a matrix.

    THE POLICY IS AN ARGUMENT, NOT A KEY READ OFF THE TARGET.  The forward
    resolver reads ``nest_microphysics_transition`` off its target, which is
    the child.  Reversed, the target is the PARENT, and a middle parent's
    key is the policy of its OWN upward edge: d1(mp8) -> d2(mp18) puts
    ``mp8-to-mp18-mass-diagnosed-v1`` on d2, so a reverse d3(mp6) -> d2(mp18)
    edge that read the target's key would hit the "takes
    nest_microphysics_transition=" mismatch for a pair whose closure is the
    edge matrix.  So the caller states the policy or takes the matrix's.

    ``policy`` is checked against the pair, never silently honoured: the two
    spellings a reverse edge may carry are the forward matrix id the pair
    takes and :data:`REVERSE_EDGE_POLICY` itself.  The contract that comes
    back is stamped with :data:`REVERSE_EDGE_POLICY` and
    :data:`REVERSE_TRANSITION_ORDER`, so a receipt says which way the mass
    went rather than leaving a reader to infer it from the mp numbers.
    """

    source = int(getattr(child_cfg, "mp_physics", 0))
    target = int(getattr(parent_cfg, "mp_physics", 0))
    matrix_policy = (
        SAME_SCHEME_POLICY if source == target
        else MP8_TO_MP18_POLICY if (source, target) == (8, 18)
        else EDGE_MATRIX_POLICY
    )
    if policy is not None and policy not in (
            matrix_policy, REVERSE_EDGE_POLICY):
        raise ValueError(
            f"reverse nest microphysics edge MP{source}->MP{target} takes "
            f"{REVERSE_EDGE_POLICY!r} (or the forward matrix id "
            f"{matrix_policy!r}); {policy!r} is the closure of another edge. "
            "Pass one of those, or leave policy out and the edge resolves "
            "to the matrix.")
    contract = resolve_microphysics_transition(
        child_cfg, _WithPolicy(parent_cfg, matrix_policy))
    if not contract.mixed:
        return contract
    return dataclasses_replace(
        contract, policy_id=REVERSE_EDGE_POLICY,
        translation_order=REVERSE_TRANSITION_ORDER)


class _WithPolicy:
    """``cfg`` with ``nest_microphysics_transition`` replaced, read-only.

    The reverse resolver reuses the forward one rather than restating the
    matrix, and the forward one reads that one key off the config object it
    is handed.  This substitutes the key and forwards everything else --
    ``mp_physics``, ``moist``, ``moist_cq``, ``morr_rimed_ice``,
    ``wdm6_ccn_conc`` -- to the real config, so no second copy of the
    resolver's input list exists to go stale.
    """

    __slots__ = ("_cfg", "nest_microphysics_transition")

    def __init__(self, cfg, policy: str):
        object.__setattr__(self, "_cfg", cfg)
        object.__setattr__(self, "nest_microphysics_transition", policy)

    def __getattr__(self, name):
        return getattr(self._cfg, name)


def transition_handles_field(
        contract: MicrophysicsTransitionContract, field_name: str) -> bool:
    return bool(
        contract.mixed
        and field_name in _TARGET_FIELDS[contract.target_mp_physics]
    )


def transition_target_fields(
        contract: MicrophysicsTransitionContract) -> tuple[str, ...]:
    """Every field the edge kernel writes for ``contract``'s target scheme.

    The target's transported masses followed by its moments, in the
    order the field-code table declares them.  This is the inventory a
    caller that has no ``DomainState`` to iterate (the offline downscale
    lane converts a parent ARCHIVE) walks to run the same kernel the live
    nest edge runs, so the two routes cannot disagree about which fields
    a scheme boundary produces.  Empty for a same-scheme contract, which
    converts nothing.
    """
    if not contract.mixed:
        return ()
    return tuple(_TARGET_FIELDS[contract.target_mp_physics])


def transition_source_field_shape(state, field_name: str) -> tuple[int, ...]:
    if field_name not in _ALL_EDGE_FIELDS:
        raise ValueError(f"unsupported microphysics edge field {field_name!r}")
    shape = tuple(int(value) for value in state.qv.shape)
    if len(shape) != 3:
        raise ValueError(f"parent moisture field must be 3-D, got {shape}")
    return shape


#: The source planes :func:`launch_microphysics_edge_field` may read,
#: horizontally windowed.  THE LIST IS THE KERNEL'S INPUT SET and must stay
#: it: the windowed namespace is the whole parent as far as the launcher can
#: see, so a plane an arm reads and this tuple omits is not a slow path, it
#: is an ``AttributeError`` on a run the registry already admitted at plan
#: review.  Milbrandt-Yau's arm was exactly that -- it reads ``thp`` and
#: ``p`` beside the masses, the tile-streamed nest route
#: (woof/ingest/reconstruction_store.py -> parent_only_init(window=...))
#: is the only route that windows, and a windowed mp=9 edge died on
#: ``thb`` while the same edge on a resident parent ran.
_WINDOWED_EDGE_PLANES = (
    "alt", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
    "qir", "qib", "mub2d", "mup", "thp", "p",
    # The numbers a mixed edge carries (_CARRIED_NUMBERS); a parent without
    # them skips them, as an mp8 parent skips qh.
    "nr", "ni",
)


def edge_parent_planes() -> tuple[str, ...]:
    """The source planes an edge launcher may read, for every puller.

    One list, two consumers: :func:`transition_source_window` cuts it for a
    tile-streamed child, and :meth:`woof.core.nest.NestCoupler.
    _coupled_parent_field` pulls it out of a streamed parent's store before
    the launcher reads the state.  Both go through this accessor so the
    kernel's input set stays single-sourced -- a plane added to one arm and
    not the other is the tile-streamed mp=9 defect again, in a second place.
    """
    return _WINDOWED_EDGE_PLANES


def transition_source_window(state, window):
    """Bounded, contiguous inputs for the existing column-local edge kernel.

    The caller obtains ``window`` from the SINT registration's exact donor
    halo. No transition is recomputed on the full parent just to interpolate
    a child slab. Vertical coefficients are borrowed unchanged, and so is a
    columnar ``thb``: a base profile has no horizontal extent to cut.
    """
    from types import SimpleNamespace
    import cupy as cp
    ny, nx = state.qv.shape[-2:]
    if not isinstance(window, tuple) or len(window) != 2:
        raise ValueError("transition window must be two bounded slices")
    for sl, extent in zip(window, (ny, nx)):
        if (not isinstance(sl, slice) or sl.step not in (None, 1)
                or sl.start is None or sl.stop is None
                or not 0 <= sl.start < sl.stop <= extent):
            raise ValueError("transition window is outside the parent")
    fields = {}
    for name in _WINDOWED_EDGE_PLANES:
        value = getattr(state, name, None)
        fields[name] = (None if value is None else
                        cp.ascontiguousarray(cp.asarray(value[(...,)+window])))
    thb = getattr(state, "thb", None)
    fields["thb"] = (None if thb is None else cp.ascontiguousarray(cp.asarray(
        thb if thb.ndim == 1 else thb[(...,)+window])))
    fields["use_theta_m"] = int(getattr(state, "use_theta_m", 0) or 0)
    fields.update(c1h=cp.ascontiguousarray(cp.asarray(state.c1h)),
                  c2h=cp.ascontiguousarray(cp.asarray(state.c2h)))
    return SimpleNamespace(**fields)


def _validate_transition_arrays(contract, state, out, shape) -> None:
    import cupy as cp

    source_names = _MASS_FIELDS[contract.source_mp_physics]
    if contract.source_mp_physics == 50:
        # The exit split reads the rime pair beside the mass fields, so it
        # is validated with them rather than trusted to exist.
        source_names = source_names + ("qir", "qib")
    for name in source_names:
        value = getattr(state, name, None)
        if value is None or tuple(value.shape) != shape:
            raise ValueError(
                f"microphysics transition source {name} must have "
                f"shape {shape}")
        if value.dtype != cp.float32:
            raise TypeError(
                f"microphysics transition source {name} must be float32")
        if not value.flags.c_contiguous:
            raise ValueError(
                f"microphysics transition source {name} must be contiguous")
    if tuple(out.shape) != shape or out.dtype != cp.float32:
        raise ValueError(
            f"microphysics transition output must be float32 with shape "
            f"{shape}")
    if not out.flags.c_contiguous:
        raise ValueError("microphysics transition output must be contiguous")
    ny, nx = shape[1:]
    checks = [
        ("alt", state.alt, shape),
        ("mub2d", state.mub2d, (ny, nx)),
        ("mup", state.mup, (ny, nx)),
        ("c1h", state.c1h, (shape[0],)),
        ("c2h", state.c2h, (shape[0],)),
    ]
    if contract.target_mp_physics == 9:
        # Entering Milbrandt-Yau the kernel forms the scheme's absolute
        # temperature per cell, so thb/thp/p are inputs exactly like the
        # masses and are checked exactly like them.  Named here rather than
        # left to the launch: a parent namespace missing one of the three
        # used to reach the kernel as a bare AttributeError, on a mixed
        # edge the registry had already admitted at plan review.
        thb = getattr(state, "thb", None)
        if thb is None:
            raise ValueError(
                "the Milbrandt-Yau nest edge diagnoses the scheme's own "
                "numbers from absolute temperature and needs the parent's "
                "thb; the state it was handed carries none. A windowed "
                "donor namespace comes from transition_source_window, "
                "whose plane list is the kernel's input set")
        checks.append(
            ("thb", thb, (shape[0],) if thb.ndim == 1 else shape))
        for name in ("thp", "p"):
            value = getattr(state, name, None)
            if value is None:
                raise ValueError(
                    "the Milbrandt-Yau nest edge diagnoses the scheme's own "
                    f"numbers from absolute temperature and needs the "
                    f"parent's {name}; the state it was handed carries "
                    "none. A windowed donor namespace comes from "
                    "transition_source_window, whose plane list is the "
                    "kernel's input set")
            checks.append((name, value, shape))
    for name, value, expected in checks:
        if tuple(value.shape) != expected or value.dtype != cp.float32:
            raise ValueError(
                f"microphysics transition {name} must be float32 with "
                f"shape {expected}")
        if name in ("thb", "thp", "p") and not value.flags.c_contiguous:
            # The three MY2 planes are the ones a windowed donor produces by
            # slicing, so contiguity is the property that route can lose.
            raise ValueError(
                f"microphysics transition {name} must be contiguous")


def launch_microphysics_edge_field(
        contract: MicrophysicsTransitionContract, state, field_name: str,
        *, out, coupled: bool) -> object:
    """Write one target field diagnosed on ``state``'s own grid.

    SOURCE-NEUTRAL, and the name says so because the function always was.
    The kernel is parameterized by ``(source_mp, target_mp)`` and is
    column-local; nothing in it knows whether ``state`` is the parent of an
    edge about to be interpolated down or the child of one about to be
    restricted up.  The FORCE path passes the parent and ``coupled=True``;
    the FEEDBACK path passes the child and ``coupled=False``.
    """


    if not transition_handles_field(contract, field_name):
        raise ValueError(
            f"MP{contract.source_mp_physics}->MP"
            f"{contract.target_mp_physics} does not handle {field_name!r}")
    if not isinstance(coupled, bool):
        raise TypeError("coupled must be bool")
    if (contract.source_mp_physics, contract.target_mp_physics) == (8, 18):
        return launch_mp8_to_mp18_parent_field(
            state, field_name, out=out, coupled=coupled,
            nssl_cccn=contract.target_nssl_cccn,
            nssl_shapes=contract.target_nssl_shapes)
    if (contract.number_source(field_name) is not None
            and getattr(state, field_name, None) is not None):
        # A parent that does not hold the number (a frame written without
        # it) falls through to the target's own closure below.
        return _carry_number_field(
            state, contract.number_source(field_name), out=out,
            coupled=coupled)

    shape = transition_source_field_shape(state, field_name)
    _validate_transition_arrays(contract, state, out, shape)
    placeholder = state.qv
    if contract.target_mp_physics == 13 and field_name == "rimi":
        # Preserve the existing mass translations and field-code authorities.
        # This column diagnostic carries no source moment across an edge.
        from woof.core.kernels import get_kernel
        snow = getattr(state, "qs", None)
        columns = shape[1] * shape[2]
        get_kernel("sbu_ylin_edge", "sbu_ylin_edge_rimi")(
            ((columns + _THREADS - 1) // _THREADS,), (_THREADS,), (
                state.alt, state.qc, placeholder if snow is None else snow,
                getattr(state, "qi", placeholder), getattr(state, "qir", placeholder),
                getattr(state, "qib", placeholder),
                state.mub2d, state.mup, state.c1h, state.c2h, out,
                np.int32(snow is not None), np.int32(contract.source_mp_physics),
                np.int32(coupled),
                np.int32(shape[0]), np.int32(columns)))
        return out
    source_arrays = []
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh"):
        value = getattr(state, name, None)
        source_arrays.append(placeholder if value is None else value)
    rime_arrays = []
    for name in ("qir", "qib"):
        value = getattr(state, name, None)
        rime_arrays.append(placeholder if value is None else value)
    # Milbrandt-Yau's entry closure is the only arm that reads a
    # temperature (Cooper's ice number and Thompson's snow intercept are
    # both N(T)), a pressure (the scheme's own de = pres/(Rd*T), :3400) and
    # the scheme's constant vector.  The temperature is not built here: the
    # kernel forms it from thb/thp/p per cell, so the arm needs no array of
    # its own and reads the same three planes whether ``state`` is a
    # resident DomainState or the bounded namespace transition_source_window
    # hands a tile-streamed nest.  Every other target gets the same
    # placeholder plane the rime pair gets off a P3 edge.
    if contract.target_mp_physics == 9:
        # The scheme's constant TABLE, not the scheme: this module is staged
        # into the standalone RW-WPS preparation wheel and
        # woof.core.milbrandt2 is not, so importing the scheme here left an
        # unresolvable internal import that the wheel's staging gate refuses
        # (tools/build_rw_wps_release.py, tests/test_native_wrf_distribution.py).
        # milbrandt2_constants is pure numpy and holds the one device cache
        # both callers use.
        from woof.core.milbrandt2_constants import ck_vector_device
        from woof.core import constants as _c

        my2_arrays = (state.thb, state.thp, state.p, ck_vector_device())
        my2_scalars = (np.float32(_c.P0), np.float32(_c.RCP),
                       np.int32(1 if state.thb.ndim == 1 else 0))
    else:
        my2_arrays = (placeholder, placeholder, placeholder, placeholder)
        my2_scalars = (np.float32(0.0), np.float32(0.0), np.int32(0))
    mass_sources = [
        _SOURCE_MASS_CODES.get(contract.mass_source(name), -1)
        if name in _MASS_FIELDS[contract.target_mp_physics] else -1
        for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
    ]
    if contract.target_mp_physics == 50:
        # Entering P3, qs/qg/qh are not TARGET mass fields (P3 has none),
        # but their mass is not dropped: the kernel merges every frozen
        # source species into the single ice category, so their slots
        # resolve to the parent species wherever the parent has them.
        for slot, name in ((4, "qs"), (5, "qg"), (6, "qh")):
            if name in _MASS_FIELDS[contract.source_mp_physics]:
                mass_sources[slot] = _SOURCE_MASS_CODES[name]
    count = int(out.size)
    ny, nx = shape[1:]
    kernel, ccn_args = _nssl2_edge_kernel(
        "microphysics_edge_field", contract.target_nssl_cccn, contract.target_nssl_shapes,
        extra_defines=(("GPUWM_USE_THETA_M", 1),) if getattr(state, "use_theta_m", 0) else ())
    kernel(
        ((count + _THREADS - 1) // _THREADS,), (_THREADS,), (
            state.alt, *source_arrays, *rime_arrays, *my2_arrays,
            state.mub2d, state.mup,
            state.c1h, state.c2h, out,
            *[np.int32(code) for code in mass_sources],
            np.int32(_EDGE_FIELD_CODES[field_name]),
            np.int32(contract.source_mp_physics),
            np.int32(contract.target_mp_physics),
            np.float32(contract.target_morrison_rimed_density),
            *my2_scalars,
            np.float32(contract.target_wdm6_ccn_conc),
            np.int32(coupled), np.int32(shape[0]), np.int32(ny),
            np.int32(nx),
            *ccn_args,
        ))
    return out


_CARRY_NUMBER_KERNEL = None


def _carry_number_field(state, source_name: str, *, out, coupled: bool):
    """The source number, mass-coupled exactly as the edge kernel couples.

    ``coupled`` multiplies by the hybrid column mass
    ``(c1h*mub + c2h) + c1h*mup`` in the kernel's own rounding order.
    """

    import cupy as cp

    global _CARRY_NUMBER_KERNEL
    source = getattr(state, source_name, None)
    shape = tuple(int(v) for v in state.qv.shape)
    if source is None or tuple(source.shape) != shape:
        raise ValueError(
            f"microphysics transition source {source_name} must have "
            f"shape {shape}")
    if tuple(out.shape) != shape or out.dtype != cp.float32:
        raise ValueError(
            f"microphysics transition output must be float32 with shape {shape}")
    if not coupled:
        out[...] = cp.asarray(source, dtype=cp.float32)
        return out
    if _CARRY_NUMBER_KERNEL is None:
        _CARRY_NUMBER_KERNEL = cp.ElementwiseKernel(
            "float32 value, float32 c1, float32 c2, float32 mub, float32 mup",
            "float32 out",
            "out = __fmul_rn(__fadd_rn(__fadd_rn(__fmul_rn(c1, mub), c2), "
            "__fmul_rn(c1, mup)), value)",
            "gpuwm_edge_carry_number")
    nz, ny, nx = shape
    _CARRY_NUMBER_KERNEL(
        cp.asarray(source, dtype=cp.float32),
        cp.asarray(state.c1h, dtype=cp.float32).reshape(nz, 1, 1),
        cp.asarray(state.c2h, dtype=cp.float32).reshape(nz, 1, 1),
        cp.asarray(state.mub2d, dtype=cp.float32).reshape(1, ny, nx),
        cp.asarray(state.mup, dtype=cp.float32).reshape(1, ny, nx), out)
    return out


def _nssl2_concentration(value):
    from woof.core.nssl2_contract import nssl2_cccn_concentration
    return nssl2_cccn_concentration(value)


def _nssl2_background_ccn(value):
    from woof.core.nssl2_contract import nssl2_background_ccn_per_kg
    return nssl2_background_ccn_per_kg(value)


def _nssl2_edge_kernel(name, nssl_cccn, nssl_shapes=(0.0, 1.0), *, extra_defines=()):
    from woof.core.kernels import get_kernel, get_kernel_int_defines
    from woof.core.nssl2_shape import shape_defines
    defines = () if nssl_shapes == (0.0, 1.0) else shape_defines(nssl_shapes)
    defines += tuple(extra_defines)
    coefficient_args = ()
    if _nssl2_concentration(nssl_cccn) != np.float32(0.5e9):
        defines += (("NSSL2_VARIABLE_CCCN", 1),)
        coefficient_args = (_nssl2_background_ccn(nssl_cccn),)
    kernel = (get_kernel_int_defines("nest_microphysics", name, tuple(sorted(defines)))
              if defines else get_kernel("nest_microphysics", name))
    return kernel, coefficient_args


def launch_mp8_to_mp18_parent_field(
        state, field_name: str, *, out, coupled: bool,
        nssl_cccn: float = 0.5e9, nssl_shapes=(0.0, 1.0)) -> object:
    """Original bit-specified MP8 -> MP18 translation entry point."""

    import cupy as cp

    if field_name not in _FIELD_CODES:
        raise ValueError(f"unsupported MP8->MP18 transition field {field_name!r}")
    if not isinstance(coupled, bool):
        raise TypeError("coupled must be bool")
    shape = transition_source_field_shape(state, field_name)
    arrays = {
        "alt": state.alt, "qv": state.qv, "qc": state.qc,
        "qr": state.qr, "qi": state.qi, "qs": state.qs, "qg": state.qg,
    }
    for name, value in arrays.items():
        if value is None or tuple(value.shape) != shape:
            raise ValueError(
                f"MP8 transition source {name} must have shape {shape}")
        if value.dtype != cp.float32:
            raise TypeError(f"MP8 transition source {name} must be float32")
        if not value.flags.c_contiguous:
            raise ValueError(f"MP8 transition source {name} must be contiguous")
    if tuple(out.shape) != shape or out.dtype != cp.float32:
        raise ValueError(
            f"MP8 transition output must be float32 with shape {shape}")
    if not out.flags.c_contiguous:
        raise ValueError("MP8 transition output must be contiguous")
    ny, nx = shape[1:]
    for name, value, expected in (
            ("mub2d", state.mub2d, (ny, nx)),
            ("mup", state.mup, (ny, nx)),
            ("c1h", state.c1h, (shape[0],)),
            ("c2h", state.c2h, (shape[0],))):
        if tuple(value.shape) != expected or value.dtype != cp.float32:
            raise ValueError(
                f"MP8 transition {name} must be float32 with shape {expected}")

    count = int(out.size)
    kernel, ccn_args = _nssl2_edge_kernel(
        "mp8_to_mp18_mass_diagnosed_field", nssl_cccn, nssl_shapes)
    kernel(
        ((count + _THREADS - 1) // _THREADS,), (_THREADS,), (
            state.alt, state.qv, state.qc, state.qr, state.qi, state.qs,
            state.qg, state.mub2d, state.mup, state.c1h, state.c2h, out,
            np.int32(_FIELD_CODES[field_name]), np.int32(coupled),
            np.int32(shape[0]), np.int32(ny), np.int32(nx),
            *ccn_args,
        ))
    return out


# ---------------------------------------------------------------------------
# RETIRED NAMES.  The three entry points above were called ``..._parent_...``
# because the FORCE path was the only caller; the feedback path calls the
# same code with the child as the source, so the names moved and these thin
# aliases hold the existing call sites (woof/ingest/nest_init.py and the
# edge gates) while they follow.  No behaviour hangs off the spelling.
launch_microphysics_edge_parent_field = launch_microphysics_edge_field
transition_parent_window = transition_source_window
transition_parent_field_shape = transition_source_field_shape


__all__ = [
    "EDGE_MATRIX_POLICY", "MP8_TO_MP18_POLICY",
    "MicrophysicsTransitionContract", "NSSL2_BACKGROUND_CCN_PER_KG",
    "P3_EDGE_DENSE_RIME_DENSITY", "P3_EDGE_FRESH_SNOW_RIME_DENSITY",
    "P3_EDGE_ICE_NUCLEATION_MASS_KG", "P3_EDGE_MAX_TOTAL_NI_PER_M3",
    "P3_EDGE_QSMALL_KG_PER_KG", "P3_EDGE_RAIN_NUMBER_PER_RAIN_MASS",
    "P3_RIME_DENSITY_BOUNDS_KG_M3",
    "PORTED_MP_PHYSICS", "SAME_SCHEME_POLICY", "TRANSITION_ORDER",
    "UNVALIDATED_MIXED_EDGE_MOMENTS", "UNVALIDATED_MIXED_EDGE_SELECTORS",
    "REVERSE_EDGE_POLICY", "REVERSE_TRANSITION_ORDER",
    "edge_parent_planes",
    "launch_microphysics_edge_field",
    "launch_microphysics_edge_parent_field",
    "launch_mp8_to_mp18_parent_field", "p3_edge_entry_reference",
    "MIXED_EDGE_ENTRY_NOTES", "mixed_edge_entry_note",
    "p3_edge_exit_reference", "resolve_microphysics_transition",
    "resolve_reverse_microphysics_transition",
    "transition_handles_field", "transition_implementation_identity",
    "transition_parent_field_shape", "transition_parent_window",
    "transition_source_field_shape", "transition_source_window",
    "transition_target_fields",
]


# ---------------------------------------------------------------------------
# AGREEMENT WITH THE REGISTRY, AT IMPORT.  ``PORTED_MP_PHYSICS`` stays
# hand-written because its ORDER is the host field-code table; the registry
# publishes each option's ``consumers.nest_transition`` row from it
# (tools/build_registry.py), and this holds the two to each other so a
# scheme appended here without a rebuilt registry -- or a registry edited
# without this tuple -- fails this import instead of a tree load.  The two
# selectors in neither tuple are cited by defect id; a mixed edge touching
# them falls through to the generic refusal today.
MIXED_EDGE_ABSENCES = {
    96: "MAD-WRF has no qualified mixed-scheme cloud/tracer nest edge.",
    4: "WSM5 has no validated mixed-edge closure for parent graupel entering its five-class package; same-scheme edges do not convert species.",
    0: ("mp_physics=0 carries no hydrometeors to close a mixed edge "
        "over; a 0->X or X->0 edge falls through to the generic "
        "refusal"),
    106: ("mp_physics=106 (WSM6R) has no ratified mixed-edge "
          "receipt: its masses are WSM6's, but no per-species "
          "ratification of a 106<->X edge has been run, so a mixed "
          "edge touching it falls through to the generic refusal; "
          "a same-scheme WSM6R edge resolves before either tuple "
          "is consulted"),
}


def _column_unvalidated_options():
    from woof.physics_registry import component_options
    return {int(option['selectors']['mp_physics']): option
            for option in component_options('microphysics').values()
            if option.get('extensions', {}).get('column_kernel', {}).get('mixed_edge_refusal')}

def _require_agreement_with_the_registry() -> None:
    from woof.physics_registry import require_consumer_rows_agreement

    observed = {
        mp: {"mixed_edge_ported": True,
             "mass_fields": list(_MASS_FIELDS[mp]),
             "moment_fields": list(_MOMENT_FIELDS[mp])}
        for mp in PORTED_MP_PHYSICS
    }
    observed.update({
        mp: {"mixed_edge_ported": False}
        for mp in UNVALIDATED_MIXED_EDGE_SELECTORS
    })

    def project(row):
        if row.get("mixed_edge_ported") is True:
            return {"mixed_edge_ported": True,
                    "mass_fields": list(row["mass_fields"]),
                    "moment_fields": list(row["moment_fields"])}
        return {"mixed_edge_ported": False}

    require_consumer_rows_agreement(
        "woof.core.microphysics_transition (PORTED_MP_PHYSICS, "
        "UNVALIDATED_MIXED_EDGE_SELECTORS, _MASS_FIELDS, _MOMENT_FIELDS)",
        "microphysics", "nest_transition", observed, project=project,
        cited_absences={**MIXED_EDGE_ABSENCES, **{
            mp: option['extensions']['column_kernel']['mixed_edge_refusal']
            for mp, option in _column_unvalidated_options().items()}})


_require_agreement_with_the_registry()
