"""External scalar forcing selected by the implemented WRF operations.

WRF v4.6.1 solve_em.F:2803-2839,2904-2930: water vapour is specified;
supplied aerosols are specified when aer_init_opt > 0. Other scalar
moments retain flow-dependent boundaries unless have_bcs_scalar is set.

ANALYSED HYDROMETEORS ON THE ROOT'S SPECIFIED BOUNDARY.  WRF carries the
other moist species on a specified domain only under ``have_bcs_moist``
(WRF v4.7.1 dyn_em/solve_em.F:2265-2267 relax and spec, :2346 no
flow-dependent boundary, :4701-4703 spec_bdy_final), whose Registry
default is off (Registry/Registry.EM_COMMON:2979), and stock real.exe
writes water vapour only (main/real_em.F:956, :1156, :1374).  WOOF does
not take that default where the source publishes the hydrometeors: a
flow-dependent boundary has zero inflow, so an analysed cloud or snow
field drains out of the domain and is never resupplied (a HRRR-forced
two-hour winter case kept 1.0 million t of snow over its inner domain
against the 3.1 in HRRR's own analysis).  Which masses a source publishes
on EVERY forcing frame is a fact of its table row: the canonical
hydrometeor fields its mapping declares (:func:`mapping_boundary_species`)
or, for a native route whose inventory lives in its decoder, the row's
``boundary_species`` column (:func:`source_boundary_species`).  A source
that publishes none keeps water vapour only, byte for byte.

THE NUMBER MOMENTS AT THE RING.  WRF's ``have_bcs_scalar``
(solve_em.F:2812-2815, :2912-2917) specifies every scalar.  Here the ring
carries exactly the number moments the cold start seeds from the masses it
carries (:data:`COLD_START_SEEDED_NUMBERS`: real.exe's make_RainNumber and
make_IceNumber on mp=8 and mp=28, and make_DropletNumber on mp=28), taken
from the same forcing frame.  Every forcing frame is initialized through
the same cold start as the start time, so the ring's number is the one
the rule gives its own analysed mass, and the ring's sizes are the start
state's sizes.  Left flow-dependent the ring would bring the analysed
mass in with zero number, and Thompson's entry block would size it at its
limits (2.5 mm rain, drizzle-sized droplets).  Morrison (mp=10) joins
them through the handover closure (:mod:`woof.ingest.handover_numbers`):
real.exe seeds no Morrison number, and a flow-dependent Morrison ring
brought the analysed mass in at N = 0, which Morrison sizes at its slope
limit (2.8 mm rain, 2 mm snow and graupel): 469-518 such columns per hour
in a hex-forced 3 km child's outer rows, drawn as 40-55 dBZ edge bands.
Exact mode (correct-mode key hydrometeor_handover_numbers off) and a strict
WRF process keep real.exe's flow-dependent Morrison ring.  A
scheme whose cold start seeds no number (WSM6, WDM6, NSSL, Milbrandt-Yau,
P3) keeps its numbers flow-dependent, which is the zero the start state
gives them too.
"""
from __future__ import annotations

from typing import Mapping


AEROSOL_BOUNDARY_FIELDS = ("nwfa", "nifa")

#: The analysed hydrometeor masses a source may publish on its forcing
#: frames, by state name, in WRF's moist-array order.
BOUNDARY_HYDROMETEOR_MASSES = ("qc", "qr", "qi", "qs", "qg")

#: The number moments a Thompson cold start seeds from a carried mass, per
#: scheme: (mass, number) in the order the closure runs.  WRF v4.7.1
#: dyn_em/module_initialize_real.F:4829-4852 (make_DropletNumber on
#: THOMPSONAERO only, make_RainNumber and make_IceNumber on THOMPSON and
#: THOMPSONAERO).  The cold-start closure in :mod:`woof.ingest.real`
#: reads its pairs from here, so the ring and the start state cannot seed
#: different moments.
COLD_START_SEEDED_NUMBERS = {
    8: (("qr", "nr"), ("qi", "ni")),
    28: (("qc", "nc"), ("qr", "nr"), ("qi", "ni")),
}

#: The supplied boundary scalars WRF carries in its ``scalar`` array rather
#: than ``moist``.  WRF's end-of-step ``spec_bdy_final`` forces a
#: scalar-array species back onto its boundary value only on a NESTED domain
#: (solve_em.F ``scalar_species_bdy_loop_3``).  On a SPECIFIED domain its
#: ring moves by the boundary tendency alone (``spec_bdy_scalar``), which the
#: RK scalar update integrates onto the table.  Water vapour and the
#: hydrometeor masses are moist-array species and are forced back on both
#: kinds of domain.  The seeded number moments are scalar-array species.
SCALAR_ARRAY_BOUNDARY_FIELDS = (*AEROSOL_BOUNDARY_FIELDS, "nc", "nr", "ni", "qke_adv")

def scalar_array_boundary_fields(cfg):
    """Active WRF scalar package fields, with the hail package extension."""
    return SCALAR_ARRAY_BOUNDARY_FIELDS + (("ng", "qvolg")
        if int(getattr(cfg, "mp_physics", 0)) == 38 else ())

#: The supplied scalars whose specified-domain lateral tendency is captured
#: once per model step into a held full-domain array
#: (``lbc_<name>_held``).  The hydrometeor masses and their numbers are not
#: among them: their tendency is recomputed on every RK stage from the
#: time-t copy and the time-t mass, which is the same number, and holds no
#: full-domain array per species.
HELD_BOUNDARY_FIELDS = ("qv", *AEROSOL_BOUNDARY_FIELDS)


def _carried_masses(cfg, species) -> tuple[str, ...]:
    """The masses in ``species`` that ``cfg``'s state allocates, in order."""
    wanted = {str(name).lower() for name in species}
    unknown = sorted(wanted - set(BOUNDARY_HYDROMETEOR_MASSES))
    if unknown:
        raise ValueError(
            f"boundary hydrometeor species {unknown} are not among "
            f"{BOUNDARY_HYDROMETEOR_MASSES}")
    if not wanted:
        return ()
    from woof.core.device_inventory import state_array_shapes

    allocated = state_array_shapes(cfg)
    return tuple(name for name in BOUNDARY_HYDROMETEOR_MASSES
                 if name in wanted and name in allocated)


def boundary_hydrometeor_fields(cfg, species) -> tuple[str, ...]:
    """Masses ``species`` supplies that the scheme carries, then their numbers.

    ``species`` is what the SOURCE publishes on every forcing frame; the
    result is what the root's specified boundary carries for ``cfg``.
    """
    if not getattr(cfg, "moist", False) or not getattr(cfg, "specified", True):
        return ()
    masses = _carried_masses(cfg, species)
    # The seeded numbers: real.exe's Thompson rows, and the Morrison
    # handover closure (woof.ingest.handover_numbers), so a ring that
    # brings a mass in brings its number too.
    from woof.ingest.handover_numbers import handover_number_pairs
    numbers = tuple(
        number for mass, number in handover_number_pairs(
            int(getattr(cfg, "mp_physics", 0)), cfg)
        if mass in masses)
    return (*masses, *numbers)


def external_scalar_fields(cfg, *, aerosol_from_input: bool | None = None,
                           boundary_species=()):
    """Ordered scalar inventory, with an optional resolved ingest decision.

    The decision records whether initialization actually read aerosols. It
    must not be inferred from their values: a supplied zero is still data.
    ``boundary_species`` names the analysed hydrometeor masses the source
    published on this forcing frame and the initializer installed; the
    default, none, is water vapour only.
    """
    if not cfg.moist:
        return ()
    if aerosol_from_input is None:
        aerosol_from_input = (
            int(getattr(cfg, "aer_init_opt", 0)) > 0
            or getattr(cfg, "mp28_aerosol_source", "auto") == "climatology")
    hydrometeors = boundary_hydrometeor_fields(cfg, boundary_species)
    if int(cfg.mp_physics) in (28, 38) and aerosol_from_input:
        return ("qv", *hydrometeors, *AEROSOL_BOUNDARY_FIELDS)
    from woof.core.chem_mp_coupling import source_number_fields
    composition = source_number_fields(cfg) if int(cfg.mp_physics) == 28 else ()
    return ("qv", *hydrometeors, *composition)


def potential_external_scalar_fields(cfg, *, boundary_species=()):
    """Cold pricing includes an auto-resolved aerosol dataset, if available.

    No filesystem probe or dataset read is performed by the estimator. An
    auto selection which falls back to synthetic profiles conservatively
    retains the two possible aerosol rows in its allowance.
    ``boundary_species`` is the source's published hydrometeor inventory
    (:func:`source_boundary_species`); a caller that does not know the
    source prices water vapour and aerosol only.
    """
    return external_scalar_fields(
        cfg, aerosol_from_input=(
            int(getattr(cfg, "aer_init_opt", 0)) > 0
            or getattr(cfg, "mp28_aerosol_source", "auto") != "synthetic"),
        boundary_species=boundary_species)


def admissible_boundary_inventory(cfg, scalars) -> bool:
    """Whether a sealed boundary table's scalar inventory fits ``cfg``.

    Water vapour always; supplied aerosol fields on mp=28 only; any subset of the masses the scheme carries, each with exactly
    the number moments the cold start seeds from it.  Order-free: a cache
    records its fields sorted.
    """
    scalars = tuple(scalars)
    rest = set(scalars)
    if len(rest) != len(scalars) or "qv" not in rest:
        return False
    rest.discard("qv")
    if "qke_adv" in rest:
        if not (int(getattr(cfg, "bl_pbl_physics", 0)) == 5
                and getattr(cfg, "bl_mynn_tkeadvect", False)):
            return False
        rest.discard("qke_adv")
    aerosol = rest & set(AEROSOL_BOUNDARY_FIELDS)
    if aerosol and int(getattr(cfg, "mp_physics", 0)) not in (28, 38):
        return False
    if aerosol and aerosol != set(AEROSOL_BOUNDARY_FIELDS):
        from woof.core.chem_mp_coupling import source_number_fields
        if aerosol != set(source_number_fields(cfg)):
            return False
    rest -= aerosol
    # The chem rows a data-store source fills on every forcing frame
    # (WRF-Chem have_bcs_chem, woof.chem_source_init.chem_boundary_fields):
    # exactly those, so a table sealed before the source was enabled (or
    # with another one) is refused rather than run on default inflow.
    from woof.state_serialization_contract import CHEM_STATE_PREFIX
    carried_chem = {name for name in rest if name.startswith(CHEM_STATE_PREFIX)}
    expected_chem: set = set()
    if getattr(cfg, "chem_sets", ""):
        from woof.chem_source_init import chem_boundary_fields
        from woof.chem_table import load as load_chem_table
        try:
            expected_chem = set(chem_boundary_fields(load_chem_table(cfg), cfg))
        except (AttributeError, TypeError, ValueError):
            return False
    if carried_chem != expected_chem:
        return False
    rest -= carried_chem
    masses = tuple(name for name in BOUNDARY_HYDROMETEOR_MASSES
                   if name in rest)
    try:
        expected = boundary_hydrometeor_fields(cfg, masses)
    except (AttributeError, TypeError, ValueError):
        return False
    return set(expected) == rest


def mapping_boundary_species(mapping: Mapping[str, object]) -> tuple[str, ...]:
    """The hydrometeor masses a mapping document declares, by state name.

    A declared canonical field is decoded from every frame the mapping
    reads, so the lateral boundary may carry it.  Read off the mapping's
    own ``fields`` table: adding a source that publishes hydrometeors is a
    mapping row, not a code path.
    """
    from woof.mapped_source import HYDROMETEOR_LEGACY_NAMES

    declared = mapping.get("fields") or {}
    names = {legacy.lower() for canonical, legacy
             in HYDROMETEOR_LEGACY_NAMES.items() if canonical in declared}
    return tuple(name for name in BOUNDARY_HYDROMETEOR_MASSES
                 if name in names)


def sealed_boundary_species(intervals) -> tuple[str, ...] | None:
    """The hydrometeor masses a sealed boundary inventory carries.

    ``intervals`` is a prepared cache's retained interval rows, each
    naming the ``fields`` its tables hold (the header's
    ``metadata.lbc.intervals``).  A forecast door about to restore that
    cache prices what the cache holds rather than what its source's row
    says: a forecast from a user's own mapping runs as ``mapped``, a
    name whose row publishes nothing although its cache carries every
    mass the mapping declares, and a cache sealed before the boundary
    carried hydrometeors holds water vapour alone whatever its source
    publishes today.  ``None`` when no row names its fields, and the
    caller prices its source's row instead.
    """
    carried: set[str] = set()
    readable = False
    for row in intervals if isinstance(intervals, (list, tuple)) else ():
        fields = row.get("fields") if isinstance(row, Mapping) else None
        if isinstance(fields, (list, tuple)):
            readable = True
            carried.update(str(name).lower() for name in fields)
    if not readable:
        return None
    return tuple(name for name in BOUNDARY_HYDROMETEOR_MASSES
                 if name in carried)


def stale_boundary_species(cfg, published, carried, *,
                           analysed) -> tuple[str, ...]:
    """The analysed masses a sealed boundary lets drain out of the domain.

    ``published`` is what the cache's source publishes on every frame
    today (:func:`source_boundary_species`), ``carried`` what its sealed
    interval tables hold (:func:`sealed_boundary_species`), ``analysed``
    the masses its start state holds nonzero (:func:`analysed_start_species`).
    A fresh preparation for ``cfg`` carries :func:`boundary_hydrometeor_fields`
    of what is published; an analysed one among them that the sealed
    tables do not hold flows out through a boundary that never resupplies
    it.  Empty when nothing analysed drains: a cache whose source
    publishes none, whose scheme carries none of it, or whose start state
    holds none of what its boundary lacks.
    """
    expected = boundary_hydrometeor_fields(cfg, published)
    held = {str(name).lower() for name in (carried or ())}
    present = {str(name).lower() for name in (analysed or ())}
    return tuple(name for name in expected
                 if name in BOUNDARY_HYDROMETEOR_MASSES
                 and name not in held and name in present)


def analysed_start_species(metadata) -> tuple[str, ...]:
    """The hydrometeor masses a prepared cache's start state holds nonzero.

    Read off its initialization receipt
    (``metadata.hydrometeor_initialization.initialized_state_species``,
    one fingerprint per installed mass with its ``nonzero_count``).  A
    cache without the receipt names none.
    """
    receipt = (metadata or {}).get("hydrometeor_initialization")         if isinstance(metadata, Mapping) else None
    installed = (receipt.get("initialized_state_species")
                 if isinstance(receipt, Mapping) else None)
    if not isinstance(installed, Mapping):
        return ()
    nonzero = {str(name).lower() for name, fingerprint in installed.items()
               if isinstance(fingerprint, Mapping)
               and int(fingerprint.get("nonzero_count") or 0) > 0}
    return tuple(name for name in BOUNDARY_HYDROMETEOR_MASSES
                 if name in nonzero)


def stale_boundary_refusal(*, subject: str, source: str, lacking,
                           action: str) -> str:
    """The refusal for a prepared boundary sealed before it carried masses.

    WRF's real.exe writes water vapour only, and so did every preparation
    here until a source that publishes the hydrometeors carried them on
    the root's specified boundary (this module's docstring), or until its
    mapping declared them.  A cache sealed without them runs on a flow-dependent
    boundary for those masses, whose inflow is zero, so the analysed
    cloud and snow drain out of the domain's edges and are never
    resupplied (a HRRR-forced two-hour winter case kept 1.0 million t of
    snow over its inner domain against the 3.1 in HRRR's own analysis).
    That is the breakage the refusal prevents; ``action`` is how the
    caller's door re-prepares.
    """
    return (
        f"{subject} was sealed before its lateral boundary carried the "
        f"hydrometeors {source} publishes on every forcing frame "
        f"({', '.join(lacking)}): its boundary carries none of them, so "
        "the analysed cloud, rain, ice and snow drain out of the domain "
        f"edges and are never resupplied.  Re-prepare it: {action}")


def source_boundary_species(source) -> tuple[str, ...]:
    """The hydrometeor masses a registered source publishes on every frame.

    A row that names a packaged profile states it in that profile's
    mapping; a native row states it in its ``boundary_species`` column.
    A name the registry does not know (a met_em directory, a WRF input
    pair) publishes none here, so a cold estimate for it prices water
    vapour only.  A route that prepares from a mapping document it was
    handed rather than a registry name (a user's own mapping) passes the
    document itself, and its own ``fields`` table answers
    (:func:`mapping_boundary_species`), so every price reads the same
    table the initializer does.  A door about to restore a sealed cache
    passes the masses that cache carries (a tuple, from
    :func:`sealed_boundary_species`), which answer for themselves.
    """
    if isinstance(source, Mapping):
        return mapping_boundary_species(source)
    if isinstance(source, (tuple, list, frozenset, set)):
        carried = {str(name).lower() for name in source}
        return tuple(name for name in BOUNDARY_HYDROMETEOR_MASSES
                     if name in carried)
    if not source:
        return ()
    from woof.source_adapters import get_source_adapter

    try:
        adapter = get_source_adapter(str(source))
    except ValueError:
        return ()
    species = set(adapter.boundary_species)
    if adapter.packaged_profile is not None:
        import json

        from woof.source_authorities import packaged_authorities

        mapping = json.loads(packaged_authorities(
            adapter.packaged_profile)["mapping"].read_text(encoding="utf-8"))
        species.update(mapping_boundary_species(mapping))
    return tuple(name for name in BOUNDARY_HYDROMETEOR_MASSES
                 if name in species)
