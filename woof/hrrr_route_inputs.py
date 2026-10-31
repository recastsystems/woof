"""Author every input file the nested HRRR route needs, from one experiment.

The HRRR domain-tree route takes five documents, and until now the
product emitted exactly one of them.  ``woof domain --source hrrr``
wrote an experiment TOML and a ``namelist.wps`` that was missing
``&share/interval_seconds`` -- the one key that route's own raw-WPS
contract gate demands -- and nothing at all for
``--root-domain-spec``, ``--namelist-input`` or
``--stock-wrf-namelist-input``.  So wizard output could not drive the
route it was sized for, and the gate that proved the route ran at all
had to borrow a separate proof harness to author the missing files.

This module is that authoring, in the product, derived from ONE source
of truth: the :class:`~woof.experiment.ExperimentConfig` the wizard
already emitted.  Every geometry number, every physics switch, every
clock key here is read off that experiment rather than restated, and
:func:`write_hrrr_route_inputs` re-imports what it wrote through the
REAL importer and refuses if the round trip does not reproduce the
experiment it started from.  A set of files that disagrees with the
TOML beside them is exactly the failure this replaces; it cannot be
written and then discovered three stages downstream.

What the route requires of these files is not restated here as prose --
it is enforced by the gates in :mod:`woof.hrrr_hierarchy_direct`,
which the tests run over these exact bytes.
"""

from __future__ import annotations

from datetime import timedelta
from fractions import Fraction
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from woof.config import (GRELL_FREITAS_CU_PHYSICS, effective_radt_minutes,
                          radiation_scheme_ids)
from woof.core.microphysics_transition import PORTED_MP_PHYSICS
from woof.ingest.hrrr_target import TARGET_DOMAIN_SCHEMA
from woof.wrf_physics_inventory import EXPORT_USE_THETA_M

#: The route's fixed forcing cadence.  HRRR publishes hourly and the
#: raw-WPS/raw-runtime contract gates pin the integer 3600 in both
#: namelists; 3600.0 is refused, so the spelling matters.
FORCING_INTERVAL_SECONDS = 3600

#: HRRR's native vertical and soil inventories, as the route's runtime
#: contract pins them.
NUM_METGRID_LEVELS = 51
NUM_METGRID_SOIL_LEVELS = 9

#: Microphysics the route admits: DERIVED from the nest-transition
#: resolver's ported set rather than re-spelled, so this door can never
#: again sit behind a ratified port.  It did exactly that twice -- the set
#: stayed at the pre-P3 five after mp=50's rime-pair closure was ratified
#: into ``microphysics_transition.PORTED_MP_PHYSICS``, refusing a scheme
#: whose ingest (``HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE[50]``), nest
#: edges (both directions, all ported partners) and history inventory
#: (presence-guarded QIR/QIB rows in ``woof/io/wrfout.py``; a P3 run
#: omits QSNOW/QGRAUP exactly as stock WRF's Registry does) were all
#: already defined -- and, until R-004/R-044, it subtracted mp=28 a second
#: time on this route's own feet.
#:
#: That subtraction is GONE (R-044).  It claimed the route had no aerosol
#: lateral boundary condition; the route has had one since the WIF
#: climatology ingest landed (``woof/ingest/wif_climatology.py``, default
#: through ``mp28_aerosol_source='auto'``) and nwfa/nifa are carried in
#: ``lateral_bc.COUPLED_SCALAR_STATE_FIELDS`` on every route, this one
#: included.  What is actually conditional is the DATASET, not the source
#: name, and the dataset precondition is measured once for every route by
#: :func:`woof.config.mp28_aerosol_lateral_forcing_precondition` -- raised
#: before the fetch by ``validate_experiment_preparation`` at every door
#: that commits to building a forecast, kept as a floor at real
#: initialization and reported at plan review from the registry's row --
#: which also closes the hole
#: this table left open, where every specified-BC route that is not spelled
#: "hrrr" ran into the same depletion with no refusal at all.
SUPPORTED_MICROPHYSICS = frozenset(PORTED_MP_PHYSICS)

#: The recommended default. It does not restrict other valid suites.
#:
#: The configuration-door ruling selects the fork source version here.
#: Its in-cloud water requires the fork's legacy radiation merge. Other
#: sources and existing generic compositions retain their defaults.
ROUTE_DEFAULT_PHYSICS_PROFILE = "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"

#: The three -- and only three -- differences between the native namelist
#: woof integrates and the stock-WRF namelist beside it.  The route
#: compares the two parsed files key for key and refuses any other
#: difference, so they are generated from one renderer with one switch.
#: ``use_theta_m`` used to be a fourth (0 -> 1).  It is 0 on both halves
#: now: the engine integrates dry theta, the export the stock arm runs
#: from holds dry theta (woof.wrf_physics_inventory.EXPORT_USE_THETA_M),
#: and a stock arm told 1 either stops at WRF's input gate or, on the
#: older mixed export, reads dry boundary rows as moist theta.
#: Under the (4, 4) RRTMG pair the longwave delta collapses -- both
#: arms run 4 -- and the two stock-only keys stay stock-only for the
#: same reason: each names a setting the native arm answers in CODE, so
#: the native namelist would be claiming to control something woof does
#: not read from it.  ``ghg_input = 0`` mirrors the FIXED-gas
#: configuration the native arm runs on either 4/4 engine --
#: woof/core/rrtmg_legacy.py pins it in its switch table, and the
#: RTE+RRTMGP arm selects one annual mean CO2 for the run's
#: calendar year (woof/core/rrtmgp.py) rather than WRF's
#: time-varying CAM gas table; ``do_radar_ref = 1`` mirrors the
#: REFL_10CM woof evaluates unconditionally at output time.  The
#: certified raw-runtime contract
#: (:func:`woof.hrrr_hierarchy_direct._require_raw_stock_delta`)
#: requires both ABSENT from the native namelist and pins both in the
#: stock one; it is the enforcing half of this sentence.
#: When selected, the fork-only alb_sol is native-only. Stock WRF 4.6.1
#: has no such namelist key, so its surface keeps solar albedo disabled.
_STOCK_DELTAS = ("ra_lw_physics 0->1 (native longwave off only), "
                 "stock-only ghg_input=0, stock-only do_radar_ref=1")

STOCK_ALB_SOL_REASON = (
    "WRF 4.6.1 has no &physics alb_sol namelist key; including it makes "
    "the stock Fortran namelist read fail. The stock arm omits the key "
    "and runs without the fork's sun-angle land-albedo correction.")


class HrrrRouteInputError(ValueError):
    """This experiment cannot drive the nested HRRR route.

    ``namelist_values`` names the fields whose namelist column every
    domain agrees on, with that column's value.  A refusal that carries
    it says the EDIT that makes the configuration and the files agree
    rather than only the difference between them, so the door that
    refused can offer a way out a reader carries out in one step and a
    caller can follow it instead of copying it out of the sentence by
    hand.  Empty where the columns disagree, where the difference is not
    a per-domain field, or where the value has no spelling in a
    settings document.
    """

    def __init__(self, *args, namelist_values=None):
        super().__init__(*args)
        self.namelist_values = dict(namelist_values or {})


def _f(value) -> str:
    """A namelist float that never renders as an integer."""
    text = repr(float(value))
    return text if ("." in text or "e" in text or "E" in text) else text + "."


def _column(values) -> str:
    """One WRF per-domain column: comma separated, trailing comma."""
    return ", ".join(str(value) for value in values) + ","


def _logical(value: bool) -> str:
    return ".true." if value else ".false."


def _repeated(value, count: int) -> str:
    return _column([value] * count)


def _chem_namelist_rows(runs, *, stock: bool):
    """``(&dynamics rows, &chem group)`` for a chem run, or two empty lists.

    The route prepares from these namelists, and the preparation is what
    takes the chem start and edge state (woof/ingest/real.py
    fill_boundary_sources), so a chem TOML whose namelist carried no
    ``&chem`` prepared with no chemistry at all.  The group is written in
    the importer's own vocabulary (woof/namelist_chem.py): ``chem_opt``
    from the set's WRF package, the input/boundary switches when a
    boundary source row is enabled, every carried ``&chem`` key, and the
    operational fork's keys for a fork package (``chem_opt = 18``).  The
    round trip then demands the re-import reproduce chem_sets and
    chem_sources.  Native half only: the stock WRF arm is a non-chem build
    (woof.hrrr_hierarchy_direct._require_raw_stock_delta records it).
    Empty for a chem-off tree, so every chem-off emission keeps its bytes.
    """
    root = runs[0]
    if stock or not getattr(root, "chem_sets", ""):
        return [], []
    from woof.chem_table import catalog, chem_names
    from woof.namelist_chem import (CARRIED_DYNAMICS_KEYS, CARRIED_KEYS,
                                     FORK_CHEM_OPT_SETS)

    count = len(runs)
    for run in runs[1:]:
        if run.chem_sets != root.chem_sets or run.chem_sources != root.chem_sources:
            raise HrrrRouteInputError(
                "chem_sets/chem_sources differ between domains; WRF's &chem "
                "carries one package for the whole tree")
    sets = chem_names(root.chem_sets, "chem_sets")
    table = catalog()
    chem_opt = None
    if len(sets) == 1:
        row = table.sets.get(sets[0])
        if row is not None and row.wrf_chem_opt is not None:
            chem_opt = row.wrf_chem_opt
        else:
            chem_opt = {name: value for value, name
                        in FORK_CHEM_OPT_SETS.items()}.get(sets[0])
    if chem_opt is None:
        raise HrrrRouteInputError(
            f"chem_sets = {root.chem_sets!r} has no WRF &chem package the "
            "route's namelist can spell, so the preparation would run "
            "without that chemistry")
    fork = chem_opt in FORK_CHEM_OPT_SETS
    sources = chem_names(root.chem_sources, "chem_sources")

    def value(raw):
        if isinstance(raw, bool):
            return _logical(raw)
        if isinstance(raw, float):
            return _f(raw)
        return raw

    def row(key, raw):
        return f" {key:<36s}= {_repeated(value(raw), count)}"

    dynamics = [row(key, getattr(root, key)) for key in CARRIED_DYNAMICS_KEYS]
    group = ["&chem", row("chem_opt", chem_opt)]
    if any(table.sources[name].kind == "boundary" for name in sources
           if name in table.sources):
        group += [row("chem_in_opt", 1), row("have_bcs_chem", True)]
    group += [row(key, getattr(root, key)) for key in CARRIED_KEYS]
    if fork:
        group += [
            row("simple_dir_fdb", int(getattr(root, "simple_dir_fdb", 0))),
            *([row("plumerise_flag", 2)]
              if root.plume_fire_properties == "frp" else []),
            *([row("bb_dcycle", True)]
              if root.fire_emission_mode == "trailing_24h_dcycle" else []),
        ]
        group.append(row("enh_vermix", bool(root.enh_vermix)))
        if float(root.flam_part) != -1.0:
            group.append(row("flam_part", float(root.flam_part)))
        if int(root.aer_drydep_opt) != 0:
            group.append(row("aer_drydep_opt", int(root.aer_drydep_opt)))
    group += ["/", ""]
    return dynamics, group


#: Every switch the route's physics gate reads -- resolved per domain
#: at emission, per shipped profile at the wizard's pairing gate.
ROUTE_GATED_SWITCHES = ("mp_physics",)


def route_physics_problems(switches, *, label: str = "") -> list[str]:
    """The route's objections to one resolved switch set, or ``[]``.

    THE one spelling of the physics slice this route admits.
    :func:`validate_route_physics` applies it to every domain of a real
    experiment at emission, and the wizard's pairing predicate
    (``woof.domain_wizard.profile_route_blocker``) applies it to a
    shipped profile's switch table both when refusing a pairing and
    when RANKING lighter-suite advice.  One predicate, shared, so a
    refusal can never advise a suite this gate turns away and the two
    readings can never drift.
    """

    problems: list[str] = []
    # Actual analyzed-input initialization requirements remain authoritative.
    # Surface, turbulence, cumulus and radiation use the common RunConfig
    # validators; membership in a measured preset is not a capability.
    #
    # This list is EMPTY today and that is the correct state, not a stub.
    # Its one row (mp_physics=28) was retired with R-044: the breakage it
    # named -- no aerosol lateral boundary condition -- was fixed by the
    # WIF climatology ingest and the coupled nwfa/nifa boundary, and the
    # residue that IS real (the dataset may be missing) is source
    # independent: every door that commits to a forecast refuses it
    # before it fetches anything, and plan review reports it. The function
    # stays because it is THE spelling of this route's physics slice,
    # shared by the emission gate and the wizard's pairing predicate; a
    # future route-specific objection is a row here and nowhere else.
    return problems


def route_physics_blocker(switches) -> str | None:
    """Why this switch set cannot drive the nested HRRR route, or None.

    The profile-level projection of :func:`validate_route_physics`, for
    callers holding a suite's switch table rather than a whole
    experiment -- the wizard's pairing predicate.  The offending
    switches ride in the FIRST clause, ahead of the layered remedy,
    because a refusal that hides them behind ``--explain`` names no
    breakage.
    """

    problems = route_physics_problems(switches)
    if not problems:
        return None
    return ("this suite cannot drive the nested HRRR route -- "
            + "; ".join(problems)
            + ": pass --physics-profile with a route-compatible suite "
              "(the wizard's --source hrrr default is one)")


def validate_route_physics(exp) -> None:
    """Refuse, at emission, a suite the route's own gate will refuse.

    The alternative is what the field met: a wizard that reports PASS,
    a fetch, a root preparation, and only then a refusal naming a
    switch the wizard had already chosen.
    """
    problems = []
    for domain in exp.domains:
        run = domain.run
        problems.extend(route_physics_problems(
            {switch: getattr(run, switch)
             for switch in ROUTE_GATED_SWITCHES},
            label=f"d{domain.grid_id:02d} "))
    # No feedback clause.  Two-way feedback is a runtime coupling the
    # tree executor runs (woof.prepared_domain_tree_forecast
    # resolve_execution_plan): this route's artifacts -- initial states,
    # statics, boundary series -- are authored identically at feedback 0
    # and 1, and the coupler still refuses, by name, the trees feedback
    # cannot serve.  The one-way refusal that stood here outlived that
    # and turned away every two-way layout before anything was fetched.
    if problems:
        raise HrrrRouteInputError(
            "this suite cannot drive the nested HRRR route: "
            + "; ".join(problems)
            + ".  Pass --physics-profile with a route-compatible suite "
              "(the wizard's --source hrrr default is one).")


def verify_axis_authored_keys(exp, text: str, *, stock: bool = False) -> None:
    """Every key the physics-fidelity axis governs, in the emitted bytes.

    The axis (:mod:`woof.physics_mode`) is the AUTHOR of its resolved
    keys: the loader writes the resolved vector onto every domain's
    RunConfig and refuses any other author.  This renderer therefore has
    exactly one correct source for such a key -- the resolved config --
    and a literal spelled here instead is invisible until a root
    preparation refuses the drift.  That is what happened to the first
    Shin-Hong arm: the config resolved ``bl_pbl_physics`` to 11, the
    renderer said 1, and the profile validator was right to stop.

    So the renderer reads back its own bytes and holds every governed key
    to the RESOLVED value.  This is deliberately generic -- it names no
    ledger entry and no scheme number, and it grows with the ledger --
    because the defect class is "a governed key is spelled as a literal
    somewhere in this function", not "L3 was missed".

    An APPLIED patch whose key never reaches the namelist is refused
    outright: a WRF arm that cannot express the patch is not a mirror of
    the woof arm, and finding that out at emission costs nothing while
    finding it out after a root preparation costs a node.  A patch
    resolved to its faithful side is not held to that, because the
    faithful arm is what the emitter has always written.
    """
    from woof.namelist_import import parse_namelist_text

    resolution = getattr(exp, "physics_mode", None)
    if resolution is None or not getattr(resolution, "governed", False):
        return

    sections = parse_namelist_text(text)
    flavor = "stock-WRF" if stock else "native woof"
    problems = []
    for patch in resolution.resolved:
        # parse_namelist_text lower-cases every key it reads, and Fortran
        # namelist names are case-insensitive, so the ledger's spelling is
        # folded rather than trusted to already match.
        wanted = patch.key.lower()
        found = [(name, values) for name, entries in sections.items()
                 for key, values in entries.items()
                 if key == wanted]
        if not found:
            if patch.patched:
                problems.append(
                    f"{patch.entry_id} resolves {patch.key} = "
                    f"{patch.value!r} (patched) but the {flavor} namelist "
                    "carries no such key, so the WRF arm would run this "
                    "entry unpatched")
            continue
        if len(found) > 1:
            problems.append(
                f"{patch.key} appears in {sorted(n for n, _ in found)}; a "
                "governed key must be authored in exactly one section")
            continue
        section, values = found[0]
        # Every column element, not just d01: one experiment is one arm,
        # so a nest carrying a different resolved value would be a tree
        # that cannot be compared across its own boundary.
        if any(value != patch.value for value in values):
            problems.append(
                f"{patch.entry_id} resolves {patch.key} = {patch.value!r} "
                f"but &{section}/{patch.key} was emitted as {values!r}; the "
                "renderer must author this key from the resolved config, "
                "never as a literal")
    if problems:
        raise HrrrRouteInputError(
            "the emitted namelist contradicts the physics-fidelity axis "
            f"it was rendered from (physics_mode = {resolution.mode!r}, "
            f"patchset = {resolution.patchset!r}): "
            + "; ".join(problems)
            + ".  woof.physics_mode.governed_keys() is the list every "
              "emitter owes the resolved value.")


def render_target_domain(exp) -> str:
    """The ``--root-domain-spec`` document for this experiment's d01.

    Its key set is closed: the loader accepts the dataclass fields and
    nothing else, so this is generated from the experiment rather than
    hand-kept.
    """
    root = exp.domains[0]
    run = root.run
    projection = exp.projection
    if projection.map_proj != "lambert":
        raise HrrrRouteInputError(
            f"the nested HRRR route is Lambert-only; this domain is "
            f"{projection.map_proj}.  Re-run the wizard with "
            "--projection lambert, or pick a point where 'auto' selects "
            "it (|lat| between 25 and 60 degrees).")
    # The root clock, decomposed exactly the way WRF's registry and the
    # experiment TOML spell it: whole seconds plus a proper rational
    # remainder.  A 1.5 km root runs 7.5 s = 7 + 1/2; the integer-only
    # spelling this replaced refused that ladder outright while the
    # same clock ran all night on the GFS route.
    dt = Fraction(exp.dt_exact(root.grid_id))
    dt_whole = dt.numerator // dt.denominator
    dt_rem = dt - dt_whole
    document = {
        "schema": TARGET_DOMAIN_SCHEMA,
        "name": exp.name,
        "map_proj": "lambert",
        "nx": int(run.nx),
        "ny": int(run.ny),
        "nz": int(run.nz),
        "dx_m": float(exp.dx_exact(root.grid_id)),
        "dy_m": float(exp.dx_exact(root.grid_id)),
        "ref_lat": float(projection.ref_lat),
        "ref_lon": float(projection.ref_lon),
        "truelat1": float(projection.truelat1),
        "truelat2": float(projection.truelat2),
        "stand_lon": float(projection.stand_lon),
        "time_step_seconds": int(dt_whole),
        "spec_bdy_width": int(run.spec_bdy_width),
        "spec_zone": int(run.spec_zone),
        "relax_zone": int(run.relax_zone),
    }
    if dt_rem:
        document["time_step_fract_num"] = dt_rem.numerator
        document["time_step_fract_den"] = dt_rem.denominator
    document["surface_fallback_radius_cells"] = SURFACE_DONOR_RADIUS_CELLS
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


#: The soil donor search every emitted HRRR target carries, in HRRR cells.
#: Breakage it prevents: a fixed radius of 8 cells (24 km), the value a
#: document that omits the key still carries, refused a fitted 3 km root
#: over the Gulf coast whose own refusal measured that 14 cells reach a
#: donor for every land cell.  Donors are chosen nearest first, so every
#: cell that found one within 8 cells keeps exactly the same donor.  The
#: search box stops at HRRR's own edge
#: (:func:`woof.ingest.hrrr_target.required_hrrr_source_window`), so a
#: domain anywhere on the grid takes this radius; it used to shrink toward
#: 8 near an edge, where a wider box was refused.
SURFACE_DONOR_RADIUS_CELLS = 24


def target_domain(exp):
    """This experiment's d01 as the route's own target-domain object."""
    from woof.ingest.hrrr_target import HrrrTargetDomain

    document = json.loads(render_target_domain(exp))
    document.pop("schema")
    return HrrrTargetDomain(**document)


def target_coverage_refusal(target) -> str | None:
    """Why HRRR cannot initialize this d01, or ``None`` when it can.

    The root preparer's own demand (``required_hrrr_source_window``):
    every source cell the atmospheric interpolation reads lies inside the
    native 1799 x 1059 grid.  The soil donor search needs no margin past
    that: its box stops at HRRR's edge, where no donor exists, and a
    wider search can always be asked for.  A margin demanded past it
    refused domains near an edge whose atmosphere HRRR covers.
    """
    from woof.ingest.hrrr_target import required_hrrr_source_window

    try:
        required_hrrr_source_window(target)
    except ValueError as error:
        return str(error)
    return None


def coverage_refusal(exp) -> str | None:
    """Why HRRR cannot force this d01, or ``None`` when it can.

    HRRR's native grid is 1799 x 1059 cells, and the atmospheric
    interpolation stencil needs real source cells on every side of the
    target.  A domain sized purely against VRAM can
    therefore be a perfectly legal experiment that no HRRR fetch can
    ever force -- and that is what a card-filling ladder near the edge
    of HRRR's coverage produces.  Asked at sizing time, this makes the
    fit loop pick a domain HRRR can carry; asked at emission, it names
    the overflow instead of leaving it to a root preparation minutes
    later.

    This answers the COVERAGE question and nothing else.  A spec that
    cannot even be constructed (a non-Lambert projection, a malformed
    clock) raises :class:`HrrrRouteInputError` with its own cause and
    remedy; it is NOT returned as if the polygon were off the grid.
    Absorbing those errors here once wrapped "the root domain spec
    carries an integer time step" inside "falls outside HRRR coverage
    ...  Move --polygon inside the HRRR grid", and the user it refused
    was sent chasing a phantom coverage problem (field, 2026-08-06,
    dx 1.5 km).
    """
    return target_coverage_refusal(target_domain(exp))


def coverage_advisory(exp) -> list[str]:
    """Say when HRRR's grid, not the card, is what stopped the sizing.

    The sizing line reports headroom in GiB, so a domain the fit loop
    stopped growing for a reason that is not memory reads as an
    unexplained shortfall.  When the source cells the atmospheric
    interpolation reads reach an edge of HRRR's native grid, that IS the
    reason.  The soil donor search is not: its box stops at the edge.
    """
    from dataclasses import replace

    from woof.ingest.hrrr_target import (HRRR_SOURCE_NX, HRRR_SOURCE_NY,
                                          required_hrrr_source_window)

    if coverage_refusal(exp) is not None:
        return []
    # Radius 0: the window of the atmospheric stencil alone.
    window = required_hrrr_source_window(replace(
        target_domain(exp), surface_fallback_radius_cells=0))
    touched = [name for name, at_edge in (
        ("west", window.i_start <= 0),
        ("east", window.i_end >= HRRR_SOURCE_NX - 1),
        ("south", window.j_start <= 0),
        ("north", window.j_end >= HRRR_SOURCE_NY - 1),
    ) if at_edge]
    if not touched:
        return []
    return [
        "this domain is bounded by HRRR's own grid, not by your card: "
        "the HRRR cells its interpolation reads reach the "
        f"{'/'.join(touched)} edge of the "
        f"{HRRR_SOURCE_NX}x{HRRR_SOURCE_NY} HRRR grid, so headroom "
        "left on the card cannot be spent here"]


def _grell_family_scalars(domains) -> tuple[int, int] | None:
    """``(clos_choice, ishallow)`` for the namelist pair, or None.

    None when no domain runs Grell-Freitas: WRF's cumulus driver reads
    neither key then, and the configuration holds both at 0.  WRF
    declares each with one entry for the whole run, so the pair can state
    one closure and one shallow arm; a tree whose Grell-Freitas domains
    set different values is refused, because the stock-WRF namelist would
    run one of those domains on another's closure while the woof arm ran
    the configured one.
    """

    grell = [domain for domain in domains
             if domain.run.cu_physics == GRELL_FREITAS_CU_PHYSICS]
    if not grell:
        return None
    values = {(int(domain.run.clos_choice), int(domain.run.ishallow))
              for domain in grell}
    if len(values) == 1:
        return next(iter(values))
    stated = "; ".join(
        f"d{domain.grid_id:02d} clos_choice = {domain.run.clos_choice}, "
        f"ishallow = {domain.run.ishallow}" for domain in grell)
    raise HrrrRouteInputError(
        "the Grell-Freitas domains of this tree set different closures "
        f"({stated}), and WRF reads clos_choice and ishallow once for the "
        "whole run, so the stock-WRF namelist beside this config would run "
        "one of them on another's closure. Next: set clos_choice and "
        "ishallow once in [shared], where they reach every Grell-Freitas "
        "domain, or run this tree on a source whose route reads the "
        "configuration itself")


def _adaptive_clock_rows(runs) -> list[str]:
    """The adaptive clock's &domains rows, from the resolved config.

    Written at every value, defaults included, on the rule the turbulence
    row follows: the hierarchy stage rebuilds the experiment from these
    bytes and the stock twin runs them, so an omitted key hands both the
    Registry default while the woof forecast reads the TOML.  Omitted,
    an adaptive tree was refused at emission by the round trip below
    (config use_adaptive_time_step true, namelist false), and a tree
    that got past it would have been prepared for a fixed clock.  The
    three scope-1 keys come from the root, where the loader holds them
    for the whole tree; the nine max_domains keys are columns.
    """

    from woof.namelist_import import (ADAPTIVE_CLOCK_COLUMNS,
                                       ADAPTIVE_CLOCK_SCALARS)

    def spell(value) -> str:
        if isinstance(value, bool):
            return _logical(value)
        return _f(value) if isinstance(value, float) else str(value)

    rows = [f" {key:<35} = {spell(cast(getattr(runs[0], key)))},"
            for key, _, cast in ADAPTIVE_CLOCK_SCALARS]
    rows += [f" {key:<35} = "
             f"{_column(spell(cast(getattr(run, key))) for run in runs)}"
             for key, _, cast in ADAPTIVE_CLOCK_COLUMNS]
    return rows


def _terrain_radiation_and_mosaic_rows(runs) -> list[str]:
    """WRF &physics rows for slope radiation, terrain shading and Noah mosaic.

    All five are stock WRF 4.6.1 &physics keys the route's importer reads
    (``slope_rad`` and ``topo_shading`` per-domain columns, ``shadlen``,
    ``sf_surface_mosaic`` and ``mosaic_cat`` run-wide), so both halves of
    the pair carry them alike and the hierarchy's raw native-to-stock
    delta is unchanged.  Before these rows the renderer never wrote them:
    a TOML with ``slope_rad = 1`` or ``sf_surface_mosaic = 1`` emitted a
    pair that said 0, so the root preparation, the hierarchy stage and the
    mirrored WRF arm ran without the setting the forecast integrated.
    Each row is written only where the configuration leaves WRF's
    default, so every emission that does not use them keeps its bytes.
    """

    rows: list[str] = []
    for key in ("slope_rad", "topo_shading"):
        column = [int(getattr(run, key)) for run in runs]
        if any(column):
            rows.append(f" {key:<35} = {_column(column)}")
    for key in ("shadlen", "sf_surface_mosaic"):
        values = {getattr(run, key) for run in runs}
        if len(values) > 1:
            raise HrrrRouteInputError(
                f"{key} differs between domains; WRF reads one value for "
                f"the whole run and the domains carry {sorted(values)}")
    if float(runs[0].shadlen) != 25000.0:
        rows.append(f" {'shadlen':<35} = {_f(runs[0].shadlen)},")
    if int(runs[0].sf_surface_mosaic):
        # mosaic_cat is read only while mosaic is on (the importer writes
        # it only then, and the experiment fingerprint drops it when off,
        # woof/core/model.py), so it rides with the switch.
        categories = {int(run.mosaic_cat) for run in runs}
        if len(categories) > 1:
            raise HrrrRouteInputError(
                "mosaic_cat differs between domains; WRF reads one value "
                f"for the whole run and the domains carry {sorted(categories)}")
        rows.append(f" {'sf_surface_mosaic':<35} = "
                    f"{int(runs[0].sf_surface_mosaic)},")
        rows.append(f" {'mosaic_cat':<35} = {categories.pop()},")
    return rows


def _run_wide(runs, key):
    """The one value a WRF run-wide key carries for this tree, or a refusal."""

    values = {getattr(run, key) for run in runs}
    if len(values) > 1:
        raise HrrrRouteInputError(
            f"{key} differs between domains; WRF reads one value for the "
            f"whole run and the domains carry {sorted(values)}")
    return values.pop()


def _land_lake_and_drag_rows(runs) -> list[str]:
    """WRF &physics rows for RUC mosaic, the CLM lake, sea-ice albedo and
    the YSU topographic wind correction.

    Every one is a stock WRF 4.6.1 &physics key the route's importer reads
    (``mosaic_lu`` and ``mosaic_soil`` run-wide, ``seaice_albedo_default``
    a scalar, ``topo_wind`` and the four lake keys max_domains columns),
    so both halves of the pair carry them alike and the hierarchy's raw
    native-to-stock delta is unchanged.  The renderer never wrote them:
    a TOML with ``mosaic_lu = 1`` or ``sf_lake_physics = 1`` emitted a
    pair that said 0, so the round trip refused the shipped HRRR recipes
    (configs/recipes/hrrr_v4_gsd41.toml, hrrr_configuration_cut.toml) on
    this route, and had it not, the root preparation, the hierarchy stage
    and the mirrored WRF arm would have run without the setting the
    forecast integrated.  Each row is written only where the
    configuration leaves the WRF Registry default (which is also what the
    importer fills for an omitted key: no recipe row names them), so
    every emission that does not use them keeps its bytes.
    """

    from woof.config import RunConfig

    defaults = RunConfig.__dataclass_fields__
    rows: list[str] = []
    for key in ("mosaic_lu", "mosaic_soil"):
        value = int(_run_wide(runs, key))
        if value != defaults[key].default:
            rows.append(f" {key:<35} = {value},")
    albedo = float(_run_wide(runs, "seaice_albedo_default"))
    if albedo != defaults["seaice_albedo_default"].default:
        rows.append(f" {'seaice_albedo_default':<35} = {_f(albedo)},")
    for key in ("topo_wind", "sf_lake_physics", "use_lakedepth"):
        column = [int(getattr(run, key)) for run in runs]
        if any(value != defaults[key].default for value in column):
            rows.append(f" {key:<35} = {_column(column)}")
    for key in ("lakedepth_default", "lake_min_elev"):
        column = [float(getattr(run, key)) for run in runs]
        if any(value != defaults[key].default for value in column):
            rows.append(f" {key:<35} = {_column(_f(v) for v in column)}")
    return rows


def _drag_and_implicit_advection_rows(runs) -> list[str]:
    """WRF &dynamics rows for gravity-wave drag and implicit vertical advection.

    ``gwd_opt`` is a max_domains column WRF v4 reads in &dynamics (the
    importer reads it there, or in &physics for a pre-v4 namelist), and
    ``zadvect_implicit`` one value for the run (Registry.EM_COMMON:2873).
    Both are stock WRF 4.6.1 keys, written to both halves alike and only
    where the configuration leaves WRF's default 0.  Unwritten, a TOML
    with ``gwd_opt = 3`` (the GSL drag suite of
    configs/recipes/hrrr_v4_gsd41.toml) or ``zadvect_implicit = 1`` was
    read back as 0 and refused by the round trip.
    """

    rows: list[str] = []
    column = [int(run.gwd_opt) for run in runs]
    if any(column):
        rows.append(f" {'gwd_opt':<35} = {_column(column)}")
    implicit = int(_run_wide(runs, "zadvect_implicit"))
    if implicit:
        rows.append(f" {'zadvect_implicit':<35} = {implicit},")
    return rows


def wps_land_categories(wps_text: str, count: int) -> int:
    """NUM_LAND_CAT the land use ``wps_text``'s geog_data_res selects.

    WRF reads one num_land_cat for the run and real.exe stops when it
    differs from the geography's (share/input_wrf.F); geogrid stamps 21
    for the MODIS land use with its lake class and 20 for the 15
    arc-second set (``modis_15s``).  Domains that disagree keep 21, which
    the round trip's importer then refuses by name.
    """
    from woof.fortran_namelist import parse_namelist_text
    from woof.static.build import geog_landuse_category_count

    values = (parse_namelist_text(wps_text).get("geogrid", {})
              .get("geog_data_res") or [])
    column = [str(value) for value in values][:count]
    column += ["default"] * (count - len(column))
    counts = {geog_landuse_category_count(value) for value in column}
    return counts.pop() if len(counts) == 1 else 21


def render_namelist_input(exp, *, stock: bool = False,
                          num_land_cat: int = 21) -> str:
    """One WRF ``namelist.input`` for this experiment.

    ``stock`` renders the unchanged-WRF twin.  It differs from the
    native file in exactly three ways -- the route parses both and
    refuses a fourth -- so both come out of this one function.
    ``num_land_cat`` is the geography's category count
    (:func:`wps_land_categories`); it was a literal 21, so the route
    refused NOAA's own namelist.wps (modis_15s, 20 categories) at its
    round trip.
    """
    from woof.config import RunConfig
    from woof.physics_source_defaults import recipe_physics_defaults

    validate_route_physics(exp)
    domains = list(exp.domains)
    count = len(domains)
    root = domains[0]
    runs = [domain.run for domain in domains]
    factor2 = [run.diff_6th_factor2 for run in runs]
    # The importer's recipe fill for the standard WRF keys written below
    # (the fork inferences touch only the RUC and Thompson generations,
    # which are handled after the sections are rendered).  A standard key
    # it fills differently from the TOML is written even at its WRF
    # default: an omitted mp_zero_out or alb_sol was read back as the
    # recipe's 2 or 1.
    recipe = recipe_physics_defaults(ROUTE_REQUEST_SOURCE)

    def recipe_fills_otherwise(key, value) -> bool:
        return key in recipe and value != recipe[key]

    for key in ("fractional_seaice", "seaice_threshold"):
        values = {getattr(run, key) for run in runs}
        if len(values) > 1:
            raise HrrrRouteInputError(
                f"{key} differs between domains; WRF reads one value for "
                f"the whole run and the domains carry {sorted(values)}")
    if any(value is None for value in factor2) and any(
            value is not None for value in factor2):
        raise HrrrRouteInputError(
            "diff_6th_factor2 is set on only some domains; a WRF column "
            "cannot carry an omitted factor between numeric entries. "
            "Set it explicitly on every domain or omit the whole column")

    start = exp.start_time
    end = start + timedelta(seconds=float(exp.run_seconds))
    run_seconds = int(exp.run_seconds)
    hours, remainder = divmod(run_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    def dt_columns(moment, prefix):
        parts = {
            "year": f"{moment.year:04d}", "month": f"{moment.month:02d}",
            "day": f"{moment.day:02d}", "hour": f"{moment.hour:02d}",
            "minute": f"{moment.minute:02d}",
            "second": f"{moment.second:02d}",
        }
        return "".join(
            f" {prefix}_{key:<28} = {_repeated(value, count)}\n"
            for key, value in parts.items())

    eta = exp.vertical.eta_levels
    eta_rows = []
    for offset in range(0, len(eta), 5):
        chunk = ", ".join(repr(float(value)) for value in eta[offset:offset + 5])
        eta_rows.append(("              " if offset else "") + chunk + ",")
    clock = {"time_step": root.time_step}
    if root.time_step_fract_num:
        clock["time_step_fract_num"] = root.time_step_fract_num
        clock["time_step_fract_den"] = root.time_step_fract_den

    # Radiation is rendered from the config, per domain, not restated as
    # a literal.  The stock twin substitutes RRTM longwave (1) exactly
    # where the native arm runs with longwave off (0) -- WRF cannot run
    # the native preparation's disabled longwave -- and carries every
    # other radiation selection unchanged, so under the (4, 4) RRTMG
    # pair the longwave delta collapses and the two arms are identical
    # here.
    #
    # The RESOLVED pair, not the split fields raw.  A configuration may
    # state its radiation in the aggregate spelling -- ``ra_physics = 4``
    # with the split pair left at -1, which is what
    # ``woof import-namelist`` writes -- and the raw fields then
    # rendered ``ra_lw_physics = -1``, which the route's own importer
    # refuses as a scheme with no mapping, so an imported configuration
    # could not be given the namelists this route runs from.
    pairs = [radiation_scheme_ids(r) for r in runs]
    native_lw = [int(longwave) for longwave, _ in pairs]
    shortwave = [int(shortwave) for _, shortwave in pairs]
    longwave = ([1 if value == 0 else value for value in native_lw]
                if stock else native_lw)
    # Dry theta on both halves: the native one because the engine
    # integrates it, the stock one because the export it runs from is
    # written that way and WRF's input gate stops a namelist that
    # disagrees with its files (EXPORT_USE_THETA_M).
    theta_m = EXPORT_USE_THETA_M if stock else 0
    ghg = " ghg_input                           = 0,\n" if stock else ""
    # STOCK-ONLY, and mandatory there: the mirrored arm has to PRODUCE
    # what the registration scores.  woof evaluates REFL_10CM at output
    # time unconditionally -- it is an output product, not a
    # namelist-gated one, which is why the native namelist has no
    # counterpart to render from and the certified raw-runtime contract
    # requires the key ABSENT there.  WRF gates the same quantity three
    # deep on this one switch, Registry default 0:
    #
    #   * module_check_a_mundo.F:3477 -- do_radar_ref == 1 sets the
    #     derived compute_radar_ref;
    #   * Registry.EM_COMMON:3059 -- `package radar_refl
    #     compute_radar_ref==1 - state:refl_10cm,refd_max` allocates the
    #     array only then;
    #   * module_mp_thompson.F:1450 -- `if (diagflag .and.
    #     do_radar_ref == 1)` computes it, and solve_em.F:365 raises
    #     diagflag on every history step with ke_diag at full column
    #     depth, which is the column maximum the registration scores.
    #
    # At the default, under a scheme that is not Milbrandt/NSSL, the
    # array is never allocated: every history frame comes out missing
    # REFL_10CM with all other variables intact -- what the first
    # WRF-arm scoring attempt measured.  A SCALAR:
    # Registry.EM_COMMON:2447 declares it `namelist,physics` nentries 1,
    # run/README.namelist:1057 spells it without the `(max_dom)` its
    # per-domain neighbours there carry, and the Fortran above tests it
    # without a domain index.
    radar_ref = " do_radar_ref                        = 1," if stock else ""

    # WRF's WIF climatology request, (aer_init_opt, wif_input_opt) = (1, 1);
    # real.exe derives aer_init_opt from use_aero_icbc.  Run-wide in WRF.
    wif_pair = (int(_run_wide(runs, "aer_init_opt")),
                int(_run_wide(runs, "wif_input_opt"))) != (0, 0)
    # WRF's &time_control cycling (Registry.EM_COMMON: `rconfig logical
    # cycling namelist,time_control 1 .false.`, run-wide).  The importer
    # reads it onto RunConfig.cycling, so a pair that omits it reads back
    # as a fresh start: every imported operational HRRR namelist states
    # .true. and was refused here by the round trip (acceptance defect
    # D-04, 2.8.8).  Written on BOTH halves, only when .true., so stock
    # WRF's MYNN first call runs the same cycled start and every uncycled
    # emission keeps its bytes.
    cycling = bool(_run_wide(runs, "cycling"))
    stock_deltas = _STOCK_DELTAS
    if int(getattr(root.run, "alb_sol", 0)) == 1:
        stock_deltas += ", native-only alb_sol=1 (absent from stock WRF 4.6.1)"
    elif recipe_fills_otherwise("alb_sol", int(getattr(root.run, "alb_sol", 0))):
        # The native half writes alb_sol = 0 below so the route's recipe
        # fill cannot read it as 1; stock WRF 4.6.1 has no key and runs
        # without the correction, so the two halves run the same physics.
        stock_deltas += (", native-only alb_sol=0 (absent from stock WRF "
                         "4.6.1, which runs the same physics without it)")
    lines = [
        "! Generated by `woof domain --source hrrr`.  This is the "
        + ("stock-WRF" if stock else "native woof")
        + " half of the pair;",
        "! the two differ only by " + stock_deltas + ", which the "
        "HRRR hierarchy",
        "! route verifies key for key before it prepares anything.",
        "&time_control",
        " run_days                            = 0,",
        f" run_hours                           = {hours},",
        f" run_minutes                         = {minutes},",
        f" run_seconds                         = {seconds},",
    ]
    for run in runs:
        if any(getattr(run, key) % 60 for key in (
                "history_interval2_s", "history_interval_change_s")):
            raise HrrrRouteInputError(
                "the fork's history_interval2/history_interval_change namelist "
                "controls require whole minutes; fractional minutes would "
                "silently shift the history alarms")
    lines.append(dt_columns(start, "start").rstrip("\n"))
    lines.append(dt_columns(end, "end").rstrip("\n"))
    lines.extend([
        f" interval_seconds                    = "
        f"{FORCING_INTERVAL_SECONDS},",
        f" input_from_file                     = "
        f"{_repeated('.true.', count)}",
        f" history_interval_s                  = "
        f"{_column(int(d.full_history_interval_s or d.history_interval_s) for d in domains)}",
        *([f" history_interval2                   = {_column(int(r.history_interval2_s / 60) for r in runs)}",
           f" history_interval_change             = {_column(int(r.history_interval_change_s / 60) for r in runs)}"]
          if any(r.history_interval2_s for r in runs) else []),
        f" frames_per_outfile                  = {_repeated(1, count)}",
        *([f" output_diagnostics                  = {root.run.output_diagnostics},",
           f" diag_int                            = {root.run.diag_int},",
           f" wind_int                            = {root.run.wind_int},"]
          if root.run.output_diagnostics or root.run.diag_int != 15 or root.run.wind_int != 5 else []),
        *([f" gsd_diagnostics                     = {root.run.gsd_diagnostics},",
           f" reset_interval1                     = {_column(r.reset_interval1 for r in runs)}"]
          if root.run.gsd_diagnostics else []),
        *([" cycling                             = .true.,"]
          if cycling else []),
        " restart                             = .false.,",
        " io_form_history                     = 2,",
        " io_form_restart                     = 2,",
        " io_form_input                       = 2,",
        " io_form_boundary                    = 2,",
        "/",
        "",
        "&domains",
    ])
    for key, value in clock.items():
        lines.append(f" {key:<35} = {value},")
    lines.extend([
        f" max_dom                             = {count},",
        f" e_we                                = "
        f"{_column(r.nx + 1 for r in runs)}",
        f" e_sn                                = "
        f"{_column(r.ny + 1 for r in runs)}",
        f" e_vert                              = "
        f"{_column(r.nz + 1 for r in runs)}",
        " eta_levels = " + ("\n".join(eta_rows)),
        f" p_top_requested                     = "
        f"{_f(exp.vertical.p_top)},",
        # &domains, and a SCALAR.  WRF declares this key
        # `namelist,domains` with nentries 1 (Registry.EM_COMMON:2283;
        # run/README.namelist documents it inside the &domains block as
        # a single value), so BOTH halves of the old spelling were
        # fatal: emitted into &dynamics, the Fortran namelist read of
        # that group fails and wrf.exe stops before the first timestep
        # -- measured on the campaign nodes -- and a per-domain column
        # would overrun a scalar namelist object here for the other
        # reason.  The root's value is the tree's: woof/experiment.py
        # refuses hypsometric_opt as a [[domain]] override, so it
        # reaches every domain from [shared] and this loses nothing.
        f" hypsometric_opt                     = "
        f"{root.run.hypsometric_opt},",
        f" dx                                  = "
        f"{_column(_f(exp.dx_exact(d.grid_id)) for d in domains)}",
        f" dy                                  = "
        f"{_column(_f(exp.dx_exact(d.grid_id)) for d in domains)}",
        f" grid_id                             = "
        f"{_column(d.grid_id for d in domains)}",
        f" parent_id                           = "
        f"{_column(d.parent_id for d in domains)}",
        f" i_parent_start                      = "
        f"{_column(d.i_parent_start for d in domains)}",
        f" j_parent_start                      = "
        f"{_column(d.j_parent_start for d in domains)}",
        f" parent_grid_ratio                   = "
        f"{_column(d.parent_grid_ratio for d in domains)}",
        f" parent_time_step_ratio              = "
        f"{_column(d.parent_time_step_ratio for d in domains)}",
        # From the config, never a literal: the hierarchy stage rebuilds
        # the experiment from these bytes, so a literal 0 would prepare
        # a two-way tree as one-way and the stock twin would run it so.
        f" feedback                            = {int(exp.feedback)},",
        f" smooth_option                       = "
        f"{int(exp.smooth_option)},",
        *_adaptive_clock_rows(runs),
        f" num_metgrid_levels                  = {NUM_METGRID_LEVELS},",
        f" num_metgrid_soil_levels             = "
        f"{NUM_METGRID_SOIL_LEVELS},",
        # The WIF key triple's &domains half.  Written for every gsd_41
        # tree (its established bytes) and wherever the configuration
        # selects the climatology pair: unwritten, a non-gsd_41 TOML with
        # aer_init_opt = wif_input_opt = 1 (configs/hrrr_v4_vertical_order5)
        # was read back as (0, 0) and refused by the round trip.
        *([f" wif_input_opt                       = {root.run.wif_input_opt},"]
          if root.run.bl_mynn_version == "gsd_41" or wif_pair else []),
        " sfcp_to_sfcp                        = .true.,",
        "/",
        "",
        "&physics",
        f" prec_acc_dt                         = {_column(_f(r.prec_acc_dt) for r in runs)}",
        f" prec_acc_dt1                        = {_column(_f(r.prec_acc_dt1) for r in runs)}",
        f" mp_physics                          = "
        f"{_column(r.mp_physics for r in runs)}",
        *([f" use_aero_icbc                       = {_logical(root.run.aer_init_opt == 1)},"]
          if root.run.bl_mynn_version == "gsd_41" or wif_pair else []),
        f" ra_lw_physics                       = "
        f"{_column(longwave)}",
        f" ra_sw_physics                       = {_column(shortwave)}",
        # The EFFECTIVE cadence, not the compatibility field beside it:
        # a domain whose row never mentions radiation carries radt=0,
        # which means "use radt_minutes" to the engine and "every model
        # step" to WRF, so writing it raw emitted a namelist that does
        # not reproduce the config it sits beside.  cudt one line below
        # already reads the modern field; this is the same reading.
        f" radt                                = "
        f"{_column(_f(effective_radt_minutes(r)) for r in runs)}",
        f" icloud                              = {root.run.icloud},",
        f" swrad_scat                          = "
        f"{_f(root.run.swrad_scat)},",
        f" sf_sfclay_physics                   = {_column(r.sf_sfclay_physics for r in runs)}",
        f" sf_surface_physics                  = {_column(r.sf_surface_physics for r in runs)}",
        # Per domain, from the RESOLVED config -- never a literal.  The
        # physics-fidelity axis writes its resolved vector onto every
        # domain's RunConfig (woof/experiment.py), so an arm selecting
        # divergence-ledger entry L3 arrives here already carrying its
        # PBL selector.  A literal 1 shadowed it: the arm resolved to
        # Shin-Hong, the mirrored namelist said YSU, and the root
        # preparation refused the drift it was right to refuse.
        f" bl_pbl_physics                      = "
        f"{_column(r.bl_pbl_physics for r in runs)}",
        *([f" bl_mynn_mixlength                   = {root.run.bl_mynn_mixlength},",
           " bl_mynn_tkebudget                   = 0,",
           f" scalar_pblmix                       = {root.run.scalar_pblmix},"]
          if root.run.bl_mynn_version == "gsd_41" else []),
        f" bldt                                = "
        f"{_column(_f(r.bldt) for r in runs)}",
        f" cu_physics                          = {_column(r.cu_physics for r in runs)}",
        f" cudt                                = "
        f"{_column(_f(r.cudt_minutes) for r in runs)}",
        f" isfflx                              = {root.run.isfflx},",
        " ifsnow                              = 1,",
        " surface_input_source                = 1,",
        f" num_soil_layers                     = {root.run.num_soil_layers},",
        *[
            f" {key:<35} = "
            + (_f(getattr(root.run, key)) if key == "mp_zero_out_thresh"
               else str(int(getattr(root.run, key)))) + ","
            for key, default in (("mp_zero_out", 0),
                                 ("mp_zero_out_thresh", 1e-8),
                                 ("mp_zero_out_all", 0))
            if getattr(root.run, key) != default
            or recipe_fills_otherwise(key, getattr(root.run, key))
        ],
        # WRF &physics fractional_seaice (Registry single-valued, default
        # 0): which sea-ice threshold the RUC seam runs.  Written when the
        # config moves it; omitted at 0, so every default emission keeps
        # its bytes.  Unwritten, a configured 1 was read back as 0.
        *([f" fractional_seaice                   = "
           f"{int(root.run.fractional_seaice)},"]
          if int(root.run.fractional_seaice) != 0 else []),
        # WRF &physics seaice_threshold (Registry single REAL, default
        # 100): real.exe's open-water-to-ice skin threshold.  Written when
        # the config moves it, omitted at 100 so default emissions keep
        # their bytes.
        *([f" seaice_threshold                    = "
           f"{_f(float(root.run.seaice_threshold))},"]
          if float(getattr(root.run, "seaice_threshold", 100.0)) != 100.0
          else []),
        f" num_land_cat                        = {int(num_land_cat)},",
        f" sf_urban_physics                    = {_repeated(0, count)}",
        " sst_update                          = 0,",
    ])
    if root.run.mp_physics == 16:
        # WRF's scalar names are shared with other microphysics schemes;
        # the importer maps them back to WDM6's active RunConfig fields.
        lines.extend([
            f" hail_opt                            = {root.run.wdm6_hail_opt},",
            f" ccn_conc                            = {_f(root.run.wdm6_ccn_conc)},",
        ])
    grell = _grell_family_scalars(domains)
    if grell is not None:
        # Written wherever a domain runs Grell-Freitas, defaults included:
        # an omitted key hands the stock-WRF arm the Registry default 0
        # while the woof arm runs the configured closure.
        clos_choice, ishallow = grell
        lines.extend([
            f" clos_choice                         = {clos_choice},",
            f" ishallow                            = {ishallow},",
        ])
    # The two stock-only &physics keys, together: each is a setting the
    # native arm answers in code and the mirrored arm can only be told.
    if ghg:
        lines.append(ghg.rstrip("\n"))
    if radar_ref:
        lines.append(radar_ref)
    # Preserve the analyzed aerosol donor request when this route writes
    # the namelist it will later run.  The importer maps this public pair
    # back to use_rap_aero_icbc and mp28_aerosol_source='analysis'.
    # With the switch off no row is added, retaining the existing bytes.
    analyzed_aerosol = {bool(getattr(r, "use_rap_aero_icbc", False)) for r in runs}
    if len(analyzed_aerosol) > 1:
        raise ValueError(
            "use_rap_aero_icbc is one value for the whole run in WRF; "
            "the domains request different analyzed aerosol sources")
    if True in analyzed_aerosol:
        lines.extend([
            " use_aero_icbc                       = .true.,",
            " use_rap_aero_icbc                   = .true.,",
        ])
    # WRF's radiation-driver options (single-valued
    # in WRF's Registry), written when the config leaves WRF's default 0.
    # The route runs from these files, so a configured 1 or 3 left
    # unwritten would run at 0 unnoticed; omitted at 0, so every default
    # emission keeps its bytes.
    for key in ("swint_opt", "aer_opt", "alb_sol"):
        values = {int(getattr(r, key, 0) or 0) for r in runs}
        if len(values) > 1:
            raise ValueError(
                f"{key} is one value for the whole run in WRF (Registry "
                f"single-valued); the domains carry {sorted(values)}")
        (value,) = values
        # alb_sol belongs to the operational fork. The stock comparison
        # must omit it because WRF 4.6.1's namelist reader has no key.
        # The paired raw-delta receipt records this physical difference.
        # A 0 the recipe fills as 1 is written in the native half as well,
        # which the hierarchy's raw delta records as physics-equivalent.
        if ((value or recipe_fills_otherwise(key, value))
                and not (stock and key == "alb_sol")):
            lines.append(f" {key:<36s}= {value},")
    lines.extend(_terrain_radiation_and_mosaic_rows(runs))
    lines.extend(_land_lake_and_drag_rows(runs))
    lines.extend([
        "/",
        "",
        "&fdda",
        "/",
        "",
        "&dynamics",
        f" use_theta_m                         = {theta_m},",
        f" hybrid_opt                          = "
        f"{exp.vertical.hybrid_opt},",
        f" etac                                = "
        f"{_f(exp.vertical.etac)},",
        f" top_lid                             = "
        f"{_repeated(_logical(root.run.top_lid), count)}",
        f" w_damping                           = {root.run.w_damping},",
        # Per domain, explicitly.  A scalar `epssm = 0.5` assigns d01
        # only and leaves every nest on WRF's Registry default of 0.1 --
        # which is how a nested run over steep terrain came to lose its
        # vertical-acoustic off-centering exactly where it needed it.
        f" epssm                               = "
        f"{_column(_f(r.epssm) for r in runs)}",
        # Per domain, from each domain's own run config: a hard-coded 2 and
        # .true. made a diff_opt = 1 (or mix_full_fields = false) run fail
        # its config-versus-namelist round trip with a misleading mismatch.
        f" diff_opt                            = "
        f"{_column(r.diff_opt for r in runs)}",
        f" km_opt                              = "
        f"{_column(r.km_opt for r in runs)}",
        f" mix_full_fields                     = "
        f"{_column(_logical(r.mix_full_fields) for r in runs)}",
        f" diff_6th_opt                        = "
        f"{_column(r.diff_6th_opt for r in runs)}",
        f" diff_6th_factor                     = "
        f"{_column(_f(r.diff_6th_factor) for r in runs)}",
        *([
            " diff_6th_factor2                    = "
            + _column(_f(value) for value in factor2),
        ] if factor2[0] is not None else []),
        f" diff_6th_slopeopt                   = "
        f"{_column(r.diff_6th_slopeopt for r in runs)}",
        # WRF's own moist-filter switch (Registry.EM_COMMON:2889,
        # max_domains, default .false.; divergence-ledger entry L4).
        # Rendered explicitly, per domain, so the mirrored WRF arm runs
        # the same moist-filter policy the config declares -- an omitted
        # key would silently hand the WRF arm the Registry default while
        # the ArWen arm read the TOML.
        f" moist_mix6_off                      = "
        f"{_column(_logical(r.moist_mix6_off) for r in runs)}",
        # THE WHOLE TURBULENCE ROW, per domain, not c_s alone.  Every key
        # here is max_domains in Registry.EM_COMMON (c_s :2862, c_k :2863,
        # mix_isotropic :2896, mix_upper_bound :2897, tke_upper_bound
        # :2899, tke_drag_coefficient :2900, tke_heat_flux :2901), every
        # one sits in woof.experiment._DOMAIN_RUN_OVERRIDES, and
        # woof.namelist_import reads every one back as a column.  With
        # only c_s authored, a suite that departs from the WRF defaults on
        # any of the others emitted a namelist describing a different run
        # from the TOML beside it: the prognostic-TKE suite writes c_k =
        # 0.1 and a wall stress, and the re-import read WRF's 0.15 and
        # 0.0, so the route's own round-trip check refused the set at
        # emission and the suite had no HRRR route at all.  Authored
        # explicitly at every value, defaults included, on the rule
        # moist_mix6_off already follows: an omitted key hands the WRF arm
        # the Registry default while the woof arm reads the TOML.
        f" c_s                                 = "
        f"{_column(_f(r.c_s) for r in runs)}",
        f" c_k                                 = "
        f"{_column(_f(r.c_k) for r in runs)}",
        f" mix_isotropic                       = "
        f"{_column(r.mix_isotropic for r in runs)}",
        f" mix_upper_bound                     = "
        f"{_column(_f(r.mix_upper_bound) for r in runs)}",
        f" tke_upper_bound                     = "
        f"{_column(_f(r.tke_upper_bound) for r in runs)}",
        f" tke_heat_flux                       = "
        f"{_column(_f(r.tke_heat_flux) for r in runs)}",
        f" tke_drag_coefficient                = "
        f"{_column(_f(r.tke_drag_coefficient) for r in runs)}",
        f" diff_6th_thresh                     = "
        f"{_column(_f(r.diff_6th_thresh) for r in runs)}",
        f" base_temp                           = "
        f"{_f(root.run.base_temp)},",
        f" damp_opt                            = {root.run.damp_opt},",
        f" zdamp                               = "
        f"{_column(_f(r.zdamp) for r in runs)}",
        f" dampcoef                            = "
        f"{_column(_f(r.dampcoef) for r in runs)}",
        f" khdif                               = "
        f"{_column(_f(r.khdif) for r in runs)}",
        f" kvdif                               = "
        f"{_column(_f(r.kvdif) for r in runs)}",
        f" non_hydrostatic                     = "
        f"{_repeated('.true.', count)}",
        f" moist_adv_opt                       = {_repeated(1, count)}",
        f" scalar_adv_opt                      = {_repeated(1, count)}",
        f" time_step_sound                     = "
        f"{_column(r.time_step_sound for r in runs)}",
        f" smdiv                               = "
        f"{_column(_f(r.smdiv) for r in runs)}",
        f" emdiv                               = "
        f"{_column(_f(r.emdiv) for r in runs)}",
        f" h_sca_adv_order                     = "
        f"{_column(r.h_sca_adv_order for r in runs)}",
        # WRF's vertical advection orders, written when any domain leaves
        # the Registry default 3.  The route runs from these files, not
        # the TOML, and the round trip compares the prepared identity the
        # orders do not enter, so a configured 5 left unwritten would run
        # at 3 unnoticed.  Omitted at 3, so every default emission keeps
        # its bytes.
        *([f" v_mom_adv_order                     = "
           f"{_column(r.v_mom_adv_order for r in runs)}",
           f" v_sca_adv_order                     = "
           f"{_column(r.v_sca_adv_order for r in runs)}"]
          if any(r.v_mom_adv_order != 3 or r.v_sca_adv_order != 3
                 for r in runs) else []),
        *_drag_and_implicit_advection_rows(runs),
        # WRF w_crit_cfl (Registry.EM_COMMON:2872, one value for the tree),
        # written when the config moves it off the 1.0 the importer fills
        # for a silent namelist, so default emissions keep their bytes.
        *([f" w_crit_cfl                          = "
           f"{_f(float(root.run.w_crit_cfl))},"]
          if float(root.run.w_crit_cfl) != 1.0 else []),
        *_chem_namelist_rows(runs, stock=stock)[0],
        "/",
        "",
        "&bdy_control",
        f" spec_bdy_width                      = "
        f"{root.run.spec_bdy_width},",
        f" spec_zone                           = {root.run.spec_zone},",
        f" relax_zone                          = {root.run.relax_zone},",
        f" spec_exp                            = "
        f"{_f(root.run.spec_exp)},",
        f" specified                           = "
        f"{_column(_logical(r.specified) for r in runs)}",
        f" nested                              = "
        f"{_column(_logical(r.nested) for r in runs)}",
        "/",
        "",
        *_chem_namelist_rows(runs, stock=stock)[1],
        "&grib2",
        "/",
        "",
        "&namelist_quilt",
        " nio_tasks_per_group                 = 0,",
        " nio_groups                          = 1,",
        "/",
        "",
    ])
    text = "\n".join(lines)
    from woof.physics_source_defaults import (
        PHYSICS_SELECTOR_VALUES, with_physics_selector_comment)
    # Use the exact rendered WRF sections so generation selection follows
    # every admitted fork signature, including a factor2 request.
    from woof.namelist_import import (operational_fork_ruc_defaults,
        operational_fork_thompson_defaults, parse_namelist_text)
    rendered_sections = parse_namelist_text(text)
    inferred_ruc = operational_fork_ruc_defaults(rendered_sections)
    inferred_thompson = operational_fork_thompson_defaults(rendered_sections)
    # The importer treats these monthly surface choices as shared.
    # A mixed tree cannot be represented by one namelist value without
    # changing a domain's selected surface physics.
    for key in ("rdlai2d", "usemonalb"):
        values = [getattr(run, key) for run in runs]
        if any(value != values[0] for value in values[1:]):
            raise HrrrRouteInputError(
                f"{key} differs between domains; the namelist monthly "
                "surface setting carries one value for the entire tree")
    # What the route's importer fills where this namelist is silent: the
    # recipe table's settings for the route's request source and the fork
    # inferences from these sections.  A value that differs from it has
    # to be written, or the route reads the recipe's value instead of the
    # TOML's.
    fallback = route_fallback_settings(rendered_sections)

    def importer_fills_otherwise(key, value) -> bool:
        return value != fallback.get(
            key, RunConfig.__dataclass_fields__[key].default)

    # Carry monthly surface fields for the land schemes that read them
    # when they are on, and wherever the importer would otherwise fill a
    # different value.  A RUC fork signature also needs explicit false
    # values so inference cannot change a roundtrip, and so does the
    # recipe's prescribed-monthly request: a TOML that leaves them off
    # (Noah or RUC without the monthly fields) was read back as on.
    prescribed_rows = [
        f" {key:<35} = {_logical(getattr(root.run, key))},"
        for key in ("rdlai2d", "usemonalb")
        if (root.run.sf_surface_physics in (2, 3)
            and (getattr(root.run, key) or inferred_ruc))
        or importer_fills_otherwise(key, getattr(root.run, key))]
    if prescribed_rows:
        text = text.replace(
            "&physics\n", "&physics\n" + "\n".join(prescribed_rows) + "\n", 1)
    selectors = {}
    for key in PHYSICS_SELECTOR_VALUES:
        default = RunConfig.__dataclass_fields__[key].default
        values = [getattr(run, key, default) for run in runs]
        if any(value != values[0] for value in values[1:]):
            raise HrrrRouteInputError(
                f"{key} differs between domains; the namelist generation "
                "comment carries one value for the entire tree")
        # A second filter factor also identifies the fork's limiter to
        # the importer. Carry a deliberately selected legacy limiter even
        # at its default value so that heuristic cannot change the run.
        explicit_legacy_limiter = (
            key == "upper_wind_limiter_form" and factor2[0] is not None)
        # A generation the recipe table fills differently is carried as
        # well: a TOML on the generic RUC snow or filter form was read
        # back as the recipe's fork form when the comment left it out.
        # ruc_soilprop joined the comment after the inferred RUC keys had
        # fixed their bytes, so it is carried only where it is needed:
        # its default is already the fork's wrf_45.
        inferred = ((key in inferred_ruc and key != "ruc_soilprop")
                    or key in inferred_thompson)
        if (values[0] != default or explicit_legacy_limiter or inferred
                or importer_fills_otherwise(key, values[0])):
            selectors[key] = values[0]
    text = with_physics_selector_comment(text, selectors)
    verify_axis_authored_keys(exp, text, stock=stock)
    return text


def _settings_value(value) -> str:
    """One setting's value as a settings document spells it."""

    return str(value).lower() if isinstance(value, bool) else repr(value)


def _settings_phrase(settings) -> str:
    """``a = 1, b = false`` -- an edit a reader can carry out as printed."""

    return ", ".join(f"{key} = {_settings_value(value)}"
                     for key, value in sorted(settings.items()))


def route_input_paths(config_path: Path) -> dict[str, Path]:
    """Where :func:`write_hrrr_route_inputs` puts each file.

    Named once, here, so the writer, the printed next steps and the
    tests cannot drift apart.
    """
    stem = config_path.stem
    parent = config_path.parent
    return {
        "wps_namelist": parent / f"{stem}.namelist.wps",
        "target_domain": parent / f"{stem}.d01-target.json",
        "namelist_input": parent / f"{stem}.namelist.input",
        "stock_namelist_input": parent / f"{stem}.stock.namelist.input",
    }


#: Per-domain keys of woof's own schema that THIS route carries once
#: for the whole tree.  Two mechanisms, one consequence, and both were
#: measured through the real door rather than read off a table:
#:
#: * the importer refuses a column whose entries differ --
#:   ``mp_physics`` and ``sf_sfclay_physics`` (``_mapped`` and the
#:   surface-layer read in :mod:`woof.namelist_import`), ``bldt`` and
#:   ``diff_6th_opt`` (its ``_uniform`` reads in &physics and
#:   &dynamics);
#: * the file pair states one value for the tree -- ``isfflx``, which
#:   WRF declares with one entry and :func:`render_namelist_input`
#:   writes from the root.
#:
#: So an edit that sets one of these on ONE domain of a regional
#: forecast cannot be written into the files that route runs from.  A
#: door that publishes a candidate asks :func:`route_shared_domain_keys`
#: and refuses such an edit with the sentence it already uses for a
#: tree-wide setting, instead of letting the round trip below refuse the
#: same edit in the importer's words with no way out.
ROUTE_SHARED_DOMAIN_KEYS = ("bldt", "diff_6th_opt", "isfflx",
                            "mp_physics", "sf_sfclay_physics",
                            "mynn_sfclay_variant", "ysu_version", "terrain_clock",
                            "diff_6th_form", "upper_wind_limiter_form",
                            "mp_zero_out", "mp_zero_out_thresh",
                            "mp_zero_out_all", "ruc_irrigation", "ruc_snow",
                            "ruc_qvg_cold_start", "ruc_2m_diagnostic",
                            "thompson_version", "thompson_fork_snow_fall")


def route_shared_domain_keys(source) -> frozenset[str]:
    """Which per-domain keys this candidate's route holds tree-wide.

    ``source`` is the candidate's own ``[fetch].source``, resolved
    through the dispatcher's own branch exactly as
    :func:`candidate_companions` resolves it, so the answer cannot
    disagree with the route the run will take.  Every other route reads
    the configuration itself and holds none of them.
    """

    from woof.source_drivability import candidate_route_chain

    if candidate_route_chain(source) != "prepared:hrrr":
        return frozenset()
    return frozenset(ROUTE_SHARED_DOMAIN_KEYS)


#: The request source this route's namelists are read under, by
#: :func:`verify_round_trip` here and by the hierarchy stage
#: (:func:`woof.hrrr_hierarchy_direct._native_experiment`).  The
#: importer fills every setting the recipe table declares for it
#: (``woof/data/physics_sources/request-defaults.v1.toml``) wherever the
#: namelist is silent, so the emission has to state any value that
#: differs from that fill.
ROUTE_REQUEST_SOURCE = "hrrr"

#: The fork Registry default a ``diff_6th_factor2`` of ``None`` runs under
#: ``diff_6th_form = "noaa_wrf39"`` (RunConfig's own documentation of the
#: field; the integrator reads it as
#: :data:`woof.core.dycore.DIFF6_FACTOR2_FORK_DEFAULT`, and a test binds
#: the two).  Restated here only so the emission path does not import the
#: GPU dycore to compare two spellings of one setting.
DIFF6_FACTOR2_FORK_DEFAULT = 0.04


def route_fallback_settings(rendered_sections) -> dict[str, object]:
    """What the route's importer fills for a setting the namelist omits.

    The recipe table's defaults for :data:`ROUTE_REQUEST_SOURCE`, with the
    fork inferences the importer makes from the rendered sections on top
    (the operational RUC and Thompson generations a fork signature
    selects).  A RunConfig field absent here falls back to its RunConfig
    default.
    """

    from woof.namelist_import import (operational_fork_ruc_defaults,
                                       operational_fork_thompson_defaults)
    from woof.physics_source_defaults import (land_scoped_defaults,
                                               recipe_physics_defaults)

    surface = rendered_sections.get("physics", {}).get(
        "sf_surface_physics", [None])
    microphysics = rendered_sections.get("physics", {}).get(
        "mp_physics", [None])
    fallback = land_scoped_defaults(
        recipe_physics_defaults(ROUTE_REQUEST_SOURCE), surface[0],
        microphysics[0])
    fallback.update(operational_fork_ruc_defaults(rendered_sections))
    fallback.update(operational_fork_thompson_defaults(rendered_sections))
    return fallback


#: Run fields the WOOF forecast reads from the TOML itself and no namelist
#: of this pair can spell, each with where the forecast reads it.
#:
#: Every RunConfig field falls in one of three groups on this route:
#:
#: (a) a field the route's importer reads from the namelist pair (a WRF
#:     key, a native-only key, or the selector comment).  The pair is what
#:     the root preparation, the hierarchy stage and the mirrored WRF arm
#:     run, so :func:`verify_round_trip` holds every such field to the
#:     TOML and refuses a drift by name;
#: (b) a field with no spelling in the pair that the forecast reads from
#:     the TOML it is bound to (``--experiment-config``: the tree runner's
#:     ``load_experiment`` in woof/prepared_domain_tree_forecast.py, and
#:     the single-domain root's ``resolve_root_experiment`` in
#:     woof/hrrr_configuration.py).  Such a value reaches the model
#:     without the pair, so the round trip does not compare it.  This
#:     table is that group, together with the write cadences and output
#:     diagnostics :func:`round_trip_exempt_run_fields` adds;
#: (c) a field neither the pair nor the forecast carries: a TOML setting
#:     that silently never reaches the model.  That is a defect to fix in
#:     the emission or the reader, never an exemption, and the round trip
#:     refuses it like (a) until it is fixed.
#:
#: The breakage a (b) entry would hide if it were wrong is a value the
#: preparation stamps differently from the TOML, so every entry must be
#: one the prepared-cache identity rules preparation-inert
#: (``PREPARATION_INERT_RUN_FIELDS``): preparation reads it nowhere, and
#: the forecast's own cache comparison drops it.
#: tests/test_route_round_trip_fields.py holds each entry to that, to its
#: cited reader naming the field, and to the importer having no spelling
#: for it.  Before this table the round trip either compared none of
#: these (the prepared identity drops them) or, briefly, all of them,
#: which refused the shipped tornado LES configs for an
#: ``inflow_perturbation`` the forecast does run.
ROUTE_FORECAST_TOML_FIELDS: Mapping[str, str] = MappingProxyType({
    # The aer_opt = 2 user-aerosol option and value controls (815912d563b):
    # read by the legacy RRTMG shortwave's aerosol optics on each radiation
    # call, from the TOML the forecast is bound to; the namelist pair spells
    # aer_type and the auxinput15 pair, not these.
    "aer_aod550_opt": "woof/core/aer_opt2.py, the aer_opt = 2 optics",
    "aer_angexp_opt": "woof/core/aer_opt2.py, the aer_opt = 2 optics",
    "aer_ssa_opt": "woof/core/aer_opt2.py, the aer_opt = 2 optics",
    "aer_asy_opt": "woof/core/aer_opt2.py, the aer_opt = 2 optics",
    "aer_aod550_val": "woof/core/aer_opt2.py, the aer_opt = 2 optics",
    "aer_angexp_val": "woof/core/aer_opt2.py, the aer_opt = 2 optics",
    "aer_ssa_val": "woof/core/aer_opt2.py, the aer_opt = 2 optics",
    "aer_asy_val": "woof/core/aer_opt2.py, the aer_opt = 2 optics",
    # WRF's lightning diagnostic package controls (cd75865eb6e): read by
    # the forecast's lightning driver on its own cadence, from the bound
    # TOML; the namelist pair spells lightning_option itself.
    "lightning_dt": "woof/core/lightning.py, the lightning driver",
    "lightning_start_seconds": "woof/core/lightning.py, the lightning driver",
    "flashrate_factor": "woof/core/lightning.py, the lightning driver",
    "iccg_method": "woof/core/lightning.py, the lightning driver",
    "iccg_prescribed_num": "woof/core/lightning.py, the lightning driver",
    "iccg_prescribed_den": "woof/core/lightning.py, the lightning driver",
    "cellcount_method": "woof/core/lightning.py, the lightning driver",
    "cldtop_adjustment": "woof/core/lightning.py, the lightning driver",
    "do_radar_ref": "woof/core/refl.py, the reflectivity diagnostic",
    # The Thompson fork table directory: read by the Thompson column at
    # load, never by preparation.
    "thompson_table_dir": "woof/core/thompson_column.py, the Thompson table loader",
    "inflow_perturbation": (
        "woof/core/inflow_perturbation.py build_inflow_perturbation, "
        "constructed per child by the NestCoupler (woof/core/nest.py)"),
    "inflow_perturbation_seed": (
        "woof/core/inflow_perturbation.py build_inflow_perturbation"),
    "inflow_perturbation_amplitude_scale": (
        "woof/core/inflow_perturbation.py build_inflow_perturbation"),
    "inflow_perturbation_faces": (
        "woof/core/inflow_perturbation.py build_inflow_perturbation"),
    "sase_moist_n2": "woof/core/sase.py, the SASE closure",
    "sase_stable_dissipation": "woof/core/sase.py, the SASE closure",
    "sase_additive_dissipation": "woof/core/sase.py, the SASE closure",
    "bl_mynn_gsd41_unsquared_qtke": (
        "woof/core/mynn_pbl_gpu.py, the gsd_41 MYNN column solver"),
    "zadvect_implicit_variant": (
        "woof/core/ieva.py, the final RK stage's implicit vertical "
        "advection"),
    "min_time_step_sound": (
        "woof/core/adaptive_clock.py, the adaptive clock's substep floor"),
    "adaptive_nest_lattice": (
        "woof/core/adaptive_clock.py, the adaptive clock's nest lattice"),
    "rrtmg_smoke_manifest": (
        "woof/core/rrtmg_smoke_manifest.py, prescribed smoke in the "
        "radiation call"),
    # The five RRFS v2 switches 809cdc87b added: no WRF namelist key the
    # importer spells, each read by the forecast from the bound TOML.
    # Without these entries the round trip compared them against the
    # importer's defaults, so a TOML on this route that set one was
    # refused for a value the forecast does read.
    "rrtmg_aer3_type_source": (
        "woof/core/rrtmg_legacy.py, the aer_opt=3 shortwave aerosol type"),
    "bl_mynn_mixaerosols": (
        "woof/config.py mynn_mixed_number_species (MYNN_QN_MIX_KEYS), the "
        "aerosol numbers the MYNN solve couples"),
    "bl_mynn_mixnumcon": (
        "woof/config.py mynn_mixed_number_species (MYNN_QN_MIX_KEYS), the "
        "cloud numbers the MYNN solve couples"),
    "tempo_ml_for_bl_nc": (
        "woof/core/tempo.py, TEMPO's boundary-layer cloud number"),
    "tempo_surface_emission_before": (
        "woof/core/tempo.py, TEMPO's surface emission order"),
    # WRF has no namelist key for it (calc_cq runs whenever water vapor
    # exists), and only the forecast's acoustic and big-step drivers read
    # it.  The importer states the physics_compat answer, True, for every
    # pair, so before this entry a TOML moist_cq = false (the verification
    # counterfactual of configs/les_tornado_100m_mayfield_20211210_attempt2,
    # attempt2b, configs/recipes/conus_hrrr_configuration.toml and the
    # three configs/frozen/ archives) was refused on this route for a
    # value no prepared array depends on.  The mirrored WRF arm runs
    # WRF's own calc_cq whatever the TOML says; a counterfactual has no
    # WRF twin on any route.
    "moist_cq": (
        "woof/core/acoustic.py prepare_moist_cq, the acoustic and "
        "big-step moisture coefficients"),
    # The zero-layer sea ice model (2.9.4): WRF has no such model and no
    # namelist key for it, so the pair cannot spell it, and the mirrored
    # WRF arm never runs it.  The forecast builds it from the bound TOML at
    # physics initialization.  Before this entry the HRRR v4 recipe's
    # seaice_prognostic = false (operational HRRR keeps RUC's own ice
    # column) was refused at emission for a value the forecast does read.
    "seaice_prognostic": (
        "woof/ocean/seaice/__init__.py, the sea ice model's initialization"),
    # THE CHEM OPERATOR'S SWITCHES (woof/config.py CHEM_RUN_FIELDS).  This
    # route's emitted pair carries no &chem group (render_namelist_input
    # writes none), so the route's pair cannot spell these -- although a
    # WRF namelist's own &chem group has mapped onto them since 2026-10-12
    # (woof/namelist_chem.py; until then it was dropped whole).  The
    # forecast's chem step is built from the bound TOML (chem_sets decides
    # whether it runs at all, and is compared in its own right), and each
    # switch below is read there.  Before these entries the route compared
    # them against the importer's defaults, so a TOML that set any of them
    # on a chem run was refused at emission for a value the forecast does
    # read.  The three whose only reader is the chem door's validation
    # (woof/config.py validate_chem_config) are read there: it refuses
    # dmsemis_opt != 0 and chem_conv_tr != 0 under a cumulus scheme, and
    # type-checks biomass_burn_opt, which the smoke set's own
    # fire_emission_mode and plume rise supersede.
    "aerosol_mp_coupling": (
        "woof/core/chem_mp_coupling.py, aerosol to aerosol-aware Thompson"),
    "biomass_burn_opt": (
        "woof/config.py validate_chem_config, the chem door"),
    "chem_conv_tr": (
        "woof/config.py validate_chem_config, the chem door"),
    "chem_mix2_off": (
        "woof/core/chem_transport.py, the chem array's horizontal mixing"),
    "chemdt": "woof/core/chem_driver.py, the chem step cadence",
    "dmsemis_opt": (
        "woof/config.py validate_chem_config, the chem door"),
    "dust_alpha": "woof/core/chem_dust.py, AFWA dust tuning",
    "dust_gamma": "woof/core/chem_dust.py, AFWA dust tuning",
    "dust_opt": "woof/core/chem_dust.py, the dust emission scheme",
    "dust_smtune": "woof/core/chem_dust.py, AFWA dust tuning",
    "dust_ustune": "woof/core/chem_dust.py, AFWA dust tuning",
    "fire_emission_mode": (
        "woof/core/chem_fire.py, fire emission timing"),
    "kemit": "woof/core/chem_inventory.py, the emission layer count",
    "mynn_chem_vertmx": (
        "woof/core/chem_vertmx.py, the chem vertical mixing"),
    "plume_fire_properties": (
        "woof/core/chem_plumerise.py, plume-rise fire properties"),
    "plumerisefire_frq": (
        "woof/core/chem_plumerise.py, the plume-rise cadence"),
    "seas_opt": "woof/core/chem_seasalt.py, the sea-salt emission scheme",
    "vertmix_onoff": (
        "woof/core/chem_vertmx.py, the chem vertical mixing"),
    "wetscav_onoff": (
        "woof/core/chem_wetdep.py, large-scale washout"),
    "simple_dir_fdb": (
        "woof/core/radiation_composition.py transported_smoke_attr, the "
        "transported smoke's AOD in the aer_opt=3 shortwave"),
    "enh_vermix": (
        "woof/core/chem_vertmx.py, the fork's mixing over burning columns"),
    "flam_part": (
        "woof/core/chem_plumerise.py, the fork's flaming fraction"),
    "aer_drydep_opt": (
        "woof/core/chem_vertmx.py, the fork's smoke deposition velocity"),
    # The per-run smoke-off sentence (woof/smoke_execution.py): launch
    # metadata with no WRF spelling, read by the run's receipts.  A run
    # whose card could not hold the smoke envelope was refused at emission
    # for it, and its tree authority did not reload without it (a 4090 host and
    # a rented box, 2026-10-21).
    "smoke_off_reason": (
        "woof/smoke_receipts.py, the run receipt's smoke-off sentence"),
})


def round_trip_exempt_run_fields() -> frozenset[str]:
    """Run fields the round trip does not compare: group (b) above.

    :data:`ROUTE_FORECAST_TOML_FIELDS`, plus the write cadences and the
    output-only diagnostic toggles, read from the two tables the prepared
    cache and the restart identity already publish for them
    (``NON_TRAJECTORY_IDENTITY_FIELDS`` and
    ``INERT_DIAGNOSTIC_IDENTITY_FIELDS``) rather than listed a third time.
    None of the cadences changes one model step, and the namelist pair is
    not their authority: preparation writes no frame, and the forecast
    takes when and what it writes from the TOML it is bound to
    (``--experiment-config``).  ``run_seconds`` is compared on its own
    below.  Comparing them anyway refused every emission, measured:
    the pair states no ``nwp_diagnostics`` or ``restart_interval``, and
    writing them would change every default emission's bytes.
    """

    from woof.ingest.prepared_cache import (
        INERT_DIAGNOSTIC_IDENTITY_FIELDS, NON_TRAJECTORY_IDENTITY_FIELDS)

    return frozenset(
        path.partition(".")[2]
        for path in NON_TRAJECTORY_IDENTITY_FIELDS
        | INERT_DIAGNOSTIC_IDENTITY_FIELDS
        if path.startswith("run.")) | frozenset(ROUTE_FORECAST_TOML_FIELDS)


def verify_round_trip(exp, wps_namelist: Path, namelist_input: Path) -> None:
    """Re-import what was just written and demand the same experiment.

    The route reads these namelists, not the TOML, so a set of files
    that describes a different tree than the TOML beside it is a defect
    that only surfaces after a fetch and a root preparation.  Running
    the REAL importer over the REAL bytes at emission is the only check
    that cannot drift from what the route will do.
    """
    from woof.experiment import build_experiment
    from dataclasses import fields as dataclass_fields
    from woof.ingest.prepared_cache import (
        _json_copy, effective_prepared_domain_config,
        prepared_domain_config_identity)
    from woof.namelist_import import import_namelists
    import tomllib

    # The TOML's governance declarations ([experiment].acknowledgements)
    # have no namelist spelling, so the round trip inherits them from
    # the authoritative experiment rather than failing the very guard
    # the TOML beside these files already satisfies.
    #
    # WHICH RRTMG is the same class of fact, and it was missed.  A WRF
    # namelist spells ``ra_lw_physics = ra_sw_physics = 4`` for both the
    # legacy-RRTMG transcription and the RTE+RRTMGP substitution -- the
    # selection between them is woof's, carried in ``ra_rrtmg_variant``,
    # and WRF has no key for it.  Re-importing without saying so resolved
    # every 4/4 emission to RTE+RRTMGP and then reported the difference
    # as though the emitted files were wrong, which made the route's own
    # legacy-RRTMG profiles unemittable.  The route already inherits this
    # out of band everywhere else it matters
    # (:func:`woof.hrrr_hierarchy_direct._native_experiment` takes it
    # from the prepared-cache identity header); this is the same
    # inheritance at emission.
    #
    # WHICH RRTMG LINEAGE is the third fact of the same kind, and it was
    # missed with the second.  WRF has no key for
    # ``wrf_rrtmg_compatibility`` either, and it is not decoration: the
    # RTE+RRTMGP arm reads it to choose its snow treatment and stamps it
    # into the restart algorithm identity, so 'none' and the mapping
    # token are two different runs of one 4/4 pair.  Re-importing without
    # saying so gave every 4/4 emission the mapping token and reported
    # the difference against a shipped suite that declares 'none' --
    # which made that suite unemittable on this route.
    variant = getattr(exp.root.run, "ra_rrtmg_variant", None)
    compatibility = getattr(exp.root.run, "wrf_rrtmg_compatibility", None)
    text, _report = import_namelists(
        wps_namelist, namelist_input, name=exp.name,
        request_source=ROUTE_REQUEST_SOURCE,
        acknowledgements=tuple(exp.acknowledgements),
        **({} if variant is None else {"rrtmg_variant": variant}),
        **({} if compatibility is None
           else {"rrtmg_compatibility": compatibility}))
    imported = build_experiment(
        tomllib.loads(text),
        source=f"round trip of {namelist_input.name}")
    exempt = round_trip_exempt_run_fields()

    def identity(candidate):
        # EVERY RunConfig field of groups (a) and (c) above, not the
        # prepared-cache identity's subset; group (b), the fields the
        # forecast reads from the TOML alone, is ``exempt``.
        # That identity drops the preparation-inert fields because
        # preparation never reads them, and comparing through it is what
        # let an emission whose namelist the importer reads differently
        # from the TOML pass silently: 1ec83bf01's stability policy first
        # (a nest epssm of 0.1 against a config of 0.5), then rdlai2d,
        # usemonalb, the RUC and filter generations and the rest, which
        # the importer fills from the recipe table wherever the namelist
        # is silent.  The namelist pair is what the preparation and the
        # mirrored WRF arm run, so a field it reads differently is a
        # different tree from the TOML beside it, refused here by name.
        # The prepared identity's normalizations (one radiation cadence,
        # a dead cumulus interval, the radiation aggregate) still apply:
        # each equates two spellings of one forecast.
        rows = []
        for domain in candidate.domains:
            effective = effective_prepared_domain_config(
                prepared_domain_config_identity(domain))
            run = effective.setdefault("run", {})
            every_field = _json_copy({
                field.name: getattr(domain.run, field.name)
                for field in dataclass_fields(domain.run)})
            for key, value in every_field.items():
                if key not in exempt:
                    run.setdefault(key, value)
            # Dead Noah-mosaic state, on the experiment fingerprint's own
            # rule (woof/core/model.py): with mosaic off nothing reads the
            # tile count or the canopy rule, and the importer writes
            # mosaic_cat only while mosaic is on, so a TOML mosaic_cat
            # beside mosaic off is one forecast with the omitted one.
            if not run.get("sf_surface_mosaic", 0):
                run.pop("mosaic_cat", None)
                run.pop("mosaic_urban_canopy", None)
            # Two spellings of one filter: an omitted second factor runs
            # the fork Registry's 0.04 under the fork form, the value the
            # importer states for it.
            if (run.get("diff_6th_form") == "noaa_wrf39"
                    and run.get("diff_6th_factor2") is None):
                run["diff_6th_factor2"] = DIFF6_FACTOR2_FORK_DEFAULT
            rows.append(effective)
        return rows

    differences = []
    #: Per field, the value each domain's namelist column carries where
    #: it differs from the configuration.  A field every domain agrees
    #: on becomes the way out below.
    columns: dict[str, list] = {}
    if len(imported.domains) != len(exp.domains):
        differences.append(
            f"domain count {len(imported.domains)} != {len(exp.domains)}")
    else:
        for grid_id, (left, right) in enumerate(
                zip(identity(exp), identity(imported)), start=1):
            left_run = left.get("run", {})
            right_run = right.get("run", {})
            for key in sorted(set(left_run) | set(right_run)):
                if left_run.get(key) != right_run.get(key):
                    differences.append(
                        f"d{grid_id:02d} {key}: config "
                        f"{left_run.get(key)!r} vs namelist "
                        f"{right_run.get(key)!r}")
                    columns.setdefault(key, []).append(right_run.get(key))
    if imported.start_time != exp.start_time:
        differences.append(
            f"start_time {imported.start_time} != {exp.start_time}")
    if float(imported.run_seconds) != float(exp.run_seconds):
        differences.append(
            f"run_seconds {imported.run_seconds} != {exp.run_seconds}")
    if differences:
        # The way out, exact where it can be: a field whose column every
        # domain agrees on has ONE value that makes the two documents
        # say the same thing, and a reader told that value carries the
        # remedy out in one step.  A field the columns disagree on is
        # left to the per-domain lines above, which already print both
        # sides.
        stated = {key: values[0] for key, values in columns.items()
                  if len(values) == len(exp.domains)
                  and all(value == values[0] for value in values)
                  and isinstance(values[0], (bool, int, float, str))}
        raise HrrrRouteInputError(
            "the emitted HRRR namelists do not reproduce the emitted "
            "config -- refusing to write a set the route would read "
            "differently from the TOML beside it: "
            + "; ".join(differences[:8])
            + ("" if len(differences) <= 8
               else f" (+{len(differences) - 8} more)")
            # A refusal names the way out, and each difference above
            # already names both values, so the way out can be exact:
            # this route integrates what the namelists say, and the
            # fields whose namelist column the pair cannot carry per
            # domain are the ones :data:`ROUTE_SHARED_DOMAIN_KEYS`
            # names.  Nothing here is a workaround for the defect this
            # module exists to fix: it is what to do with a setting this
            # route has no spelling for.
            # Not "this route runs the namelists rather than the TOML":
            # the forecast does read the TOML, and the fields it reads
            # from there alone are exempt above (ROUTE_FORECAST_TOML_FIELDS).
            # What reads the namelists is the preparation, the hierarchy
            # stage and the mirrored WRF arm, which is what the sentence
            # has to say.
            + ". This route prepares the tree from these namelists and "
            "its mirrored WRF arm runs them, so the namelist value above "
            "is what they would use while the forecast read the TOML's. "
            "Next: set each "
            "field to the value its namelist column carries"
            + (f" ({_settings_phrase(stated)})" if stated else "")
            + ", or keep the setting and run this forecast on a source "
            "whose route reads the configuration itself",
            namelist_values=stated)


def write_hrrr_route_inputs(config_path: Path, exp, *, wps_text: str,
                            writer) -> list[Path]:
    """Write every file the nested HRRR route needs beside the config.

    ``writer`` is the caller's atomic write function so the whole set
    lands the same way the config does.  Returns the paths written, in
    the order a reader meets them.
    """
    paths = route_input_paths(Path(config_path))
    writer(paths["wps_namelist"], wps_text)
    writer(paths["target_domain"], render_target_domain(exp))
    land = wps_land_categories(wps_text, len(exp.domains))
    writer(paths["namelist_input"],
           render_namelist_input(exp, num_land_cat=land))
    writer(paths["stock_namelist_input"],
           render_namelist_input(exp, stock=True, num_land_cat=land))
    verify_round_trip(exp, paths["wps_namelist"], paths["namelist_input"])
    return [paths["wps_namelist"], paths["target_domain"],
            paths["namelist_input"], paths["stock_namelist_input"]]


def configuration_reading_sources(source) -> tuple[str, ...]:
    """Sources of ``source``'s model whose route reads the configuration itself.

    Derived from the source table, never listed: every runnable row that
    names the same ``upstream_model_id`` and dispatches to a chain other
    than this route's.  It is the way out a refusal of this route can
    name for a configuration its namelists cannot carry, and it is empty
    for a model this route is the only way to run.
    """

    from woof.source_adapters import get_source_adapter, source_adapters
    from woof.source_drivability import candidate_route_chain, drivability_for

    try:
        adapter = get_source_adapter(str(source or ""))
    except ValueError:
        return ()
    model = adapter.upstream_model_id
    if model is None:
        return ()
    found = []
    for row in source_adapters():
        if row.upstream_model_id != model or row.source_id == adapter.source_id:
            continue
        verdict = drivability_for(row.source_id) or {}
        if verdict.get("refusal") is not None or not verdict.get("chain"):
            continue
        if candidate_route_chain(row.source_id) == "prepared:hrrr":
            continue
        found.append(row.source_id)
    return tuple(found)


def beside_root_disagreement(config_path, exp) -> tuple[str, ...]:
    """The d01 trajectory fields the set beside ``config_path`` holds differently.

    The root preparation reads the configuration itself
    (``tools.prepare_hrrr_wrf --experiment-config``) and the hierarchy
    reads d01 out of the namelist pair, then binds the two by
    :func:`woof.hrrr_hierarchy_direct.root_binding_differences`.  A set a
    door wrote before the configuration last changed therefore cannot run:
    WOOF canary run_0771cd8814b6e7f8 (HRRR 2026-03-10 12Z, 3 km + 1 km)
    had its set written by ``woof domain``, then ``[shared]
    gsd_diagnostics = 1`` added to the TOML, and was refused after a
    20 GB fetch and the root preparation with "native namelist d01
    trajectory controls differ from the sealed root preparation:
    run.gsd_diagnostics".

    Asked by rendering the set from the configuration (keeping the beside
    WPS namelist) and binding the two d01 imports by the hierarchy's own
    rule, so the importer reads both sides the same way.  Equal namelist
    text, a configuration the writer cannot carry, and a beside pair
    the importer refuses all answer ``()``: the set beside is then read as
    before, and the hierarchy reports anything wrong with it in its own
    words.  Only d01 is compared, because only d01 is bound; a hand edit
    to a child domain in the set beside still runs.
    """

    import tempfile

    config_path = Path(config_path)
    beside = route_input_paths(config_path)
    with tempfile.TemporaryDirectory(prefix="arwen-route-inputs-") as staging:
        target = Path(staging) / config_path.name
        try:
            write_hrrr_route_inputs(
                target, exp,
                wps_text=beside["wps_namelist"].read_text(encoding="utf-8"),
                writer=lambda path, content: path.write_text(
                    content, encoding="utf-8"))
        except ValueError:
            return ()
        rendered = route_input_paths(target)
        if (rendered["namelist_input"].read_text(encoding="utf-8")
                == beside["namelist_input"].read_text(encoding="utf-8")):
            return ()
        from woof.hrrr_hierarchy_direct import (
            _native_experiment, _sealed_root_rrtmg_compatibility,
            _sealed_root_rrtmg_variant, root_binding_differences)
        from woof.ingest.prepared_cache import prepared_domain_config_identity

        # The keywords the hierarchy imports under come from the sealed
        # root, which the preparation resolved from this configuration.
        root = {"domain_config": prepared_domain_config_identity(exp.domains[0])}
        keywords = {
            "rrtmg_variant": _sealed_root_rrtmg_variant(root),
            "rrtmg_compatibility": _sealed_root_rrtmg_compatibility(root),
            "acknowledgements": tuple(getattr(exp, "acknowledgements", ())),
        }
        try:
            ours, _, _ = _native_experiment(
                rendered["wps_namelist"], rendered["namelist_input"], **keywords)
            theirs, _, _ = _native_experiment(
                beside["wps_namelist"], beside["namelist_input"], **keywords)
        except ValueError:
            return ()
    return tuple(sorted(root_binding_differences(
        prepared_domain_config_identity(ours.domains[0]), theirs.domains[0])))


def run_route_inputs(config_path, exp, *, raw, into=None, report=None):
    """The four files one run of ``config_path`` hands this route.

    The set beside the configuration when it is COMPLETE and agrees with
    it: that is what a door wrote, and it is what the route has always
    read.  A complete set whose d01 disagrees with the configuration
    (:func:`beside_root_disagreement`: it was written before the
    configuration last changed) is not read, because the root preparation
    reads the configuration and the hierarchy would refuse the pair after
    the fetch; the run writes the set from the configuration instead and
    ``report["disagreeing"]`` names the fields.  Otherwise the
    run writes the set itself, from the configuration, through
    :func:`write_hrrr_route_inputs` and therefore through
    :func:`verify_round_trip` -- into ``into``, never beside the user's
    file.  A WPS namelist that is present is kept, because its geography
    choices are the user's and the round trip holds its geometry to the
    configuration; an absent one is rendered from the experiment by the
    doors' own renderer (:func:`woof.companion_domains.candidate_wps_text`).
    ``raw`` is the configuration's parsed TOML, whose ``[fetch]`` source
    and cadence that renderer and the refusal read.

    Before this, a configuration nobody's door had saved -- one written
    by a program, by hand, or by ``woof import-namelist`` -- passed
    ``woof go --dry-run`` and was refused three seconds into the real
    run, because the missing files were found only by the chain.

    ``into=None`` is the door's question: the set is rendered into a
    scratch folder and discarded, and ``None`` comes back.  The dry run
    and plan review ask it, so a configuration the chain would refuse is
    refused before anything is fetched or a run folder is claimed.

    Raises :class:`HrrrRouteInputError` naming the missing files, what
    the writer refused, and the sources of the same model whose route
    reads the configuration itself.
    """

    import tempfile

    config_path = Path(config_path)
    beside = route_input_paths(config_path)
    if all(path.is_file() for path in beside.values()):
        disagreeing = beside_root_disagreement(config_path, exp)
        if report is not None:
            report["disagreeing"] = disagreeing
        if not disagreeing:
            return beside
    missing = sorted(role for role, path in beside.items()
                     if not path.is_file())

    def render(folder: Path) -> Path:
        target = folder / config_path.name
        try:
            if beside["wps_namelist"].is_file():
                wps_text = beside["wps_namelist"].read_text(encoding="utf-8")
            else:
                from woof.companion_domains import candidate_wps_text

                wps_text = candidate_wps_text(raw, exp, exp, config_path)
            write_hrrr_route_inputs(
                target, exp, wps_text=wps_text,
                writer=lambda path, content: path.write_text(
                    content, encoding="utf-8"))
        except ValueError as error:
            siblings = configuration_reading_sources(
                (raw.get("fetch") or {}).get("source"))
            raise HrrrRouteInputError(
                f"the HRRR route prepares from WRF namelists written from the TOML, "
                f"{config_path.name} has none beside it ({', '.join(missing)} "
                "missing), and they cannot be written from it without "
                "changing the forecast it describes: " + str(error).rstrip(".")
                + (". A source of the same model whose route reads the "
                   "configuration itself: "
                   + ", ".join(f'[fetch] source = "{name}"' for name in siblings)
                   if siblings else ""),
                namelist_values=getattr(error, "namelist_values", None)) from None
        return target

    if into is None:
        with tempfile.TemporaryDirectory(prefix="arwen-route-inputs-") as staging:
            render(Path(staging))
        return None
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    return route_input_paths(render(into))


def candidate_companions(config_path, exp, *, wps_text: str, source):
    """Every file a candidate configuration must carry beside it.

    ONE answer for every door that publishes a configuration -- the
    domain editor, the forcing editor, the schedule editor, the
    saved-setup start, ``woof domain-fit``, ``woof domain-tiles`` and
    the following-nest authoring door -- because the question belongs to
    the ROUTE that configuration will run on, not to the edit, the refit
    or the authoring that produced it.  Answered per door it was
    answered wrong: the domain editor wrote the configuration and its
    WPS namelist and nothing else, so EVERY edit it made to a
    native-route configuration -- any action, any grid -- produced a
    candidate without the namelists the route runs from.  The fit and
    tile doors published the same short set for the same reason.  A run
    of an incomplete set no longer stops for it
    (:func:`run_route_inputs` writes the set into the run folder), but
    the set a door writes is still the one a user sees and edits beside
    the configuration, and a complete set is the one the route reads.

    ``source`` is the candidate's own ``[fetch].source`` -- the very key
    the dispatcher reads to choose the chain -- or ``None`` when it has
    none.  Not "``None`` when the configuration carries ``[case_data]``":
    a configuration with both would then be given a short copy and still
    routed to the regional chain.  The chain is resolved through
    :func:`woof.source_drivability.candidate_route_chain`, which is
    where that one branch lives so that a preprocessing install carrying
    this module without the forecast dispatcher can still ask it;
    :mod:`woof.runplan` keeps the name as a re-export, so a reader of
    the dispatcher still finds it.  One branch, so the set of files
    written here and the set the run reads cannot be decided
    differently.

    Returns ``[(destination, text), ...]`` with the WPS namelist first.
    Text rather than written files, because every one of these doors
    publishes its whole set through one create-only helper that owns
    rollback, and a file written here would sit outside it.  The route's
    own writer still runs, into a staging directory, so what comes back
    has been through :func:`verify_round_trip` against the real importer
    -- the nest cadence an edit just set is read back out of the bytes
    that carry it rather than assumed into them.
    """
    import tempfile

    from woof.source_drivability import candidate_route_chain

    config_path = Path(config_path)
    paths = route_input_paths(config_path)
    if candidate_route_chain(source) != "prepared:hrrr":
        return [(paths["wps_namelist"], wps_text)]
    with tempfile.TemporaryDirectory(prefix="arwen-candidate-") as staging:
        staged = Path(staging) / config_path.name
        written = write_hrrr_route_inputs(
            staged, exp, wps_text=wps_text,
            writer=lambda path, content: path.write_text(
                content, encoding="utf-8"))
        return [(config_path.parent / path.name,
                 path.read_text(encoding="utf-8")) for path in written]


__all__ = [
    "FORCING_INTERVAL_SECONDS",
    "HrrrRouteInputError",
    "candidate_companions",
    "configuration_reading_sources",
    "ROUTE_DEFAULT_PHYSICS_PROFILE",
    "ROUTE_SHARED_DOMAIN_KEYS",
    "route_shared_domain_keys",
    "SUPPORTED_MICROPHYSICS",
    "render_namelist_input",
    "render_target_domain",
    "route_input_paths",
    "route_physics_blocker",
    "route_physics_problems",
    "run_route_inputs",
    "validate_route_physics",
    "verify_axis_authored_keys",
    "verify_round_trip",
    "write_hrrr_route_inputs",
]
