"""Aerosol-aware Thompson (mp_physics=28) coupling: ``nwfa``/``nifa`` from the
GOCART species (process ``coupling.thompson``, DESIGN 6.7).

WRF-Chem v4.7.1 has no chem-to-Thompson aerosol path (there is no ``nwfa``
anywhere under ``chem/``).  The public coupling is NOAA GSL's: with
``merra2_aerosol_aware`` the RRFS Thompson driver overwrites its water- and
ice-friendly aerosol numbers from binned aerosol mass every call, through
``get_niwfa`` (ccpp-physics ``physics/MP/Thompson/mp_thompson.F90:1025-1068``,
Apache-2.0).  WOOF-ArWen ports that routine and says so; the kernel is
``woof/core/kernels/chem_mp_coupling.cu``.

WHAT IS DATA.  Each participating species row carries an ``mp_coupling``
object::

    {"target": "nifa" | "nwfa",   # which Thompson number field
     "group": int,                 # the parenthesised group of get_niwfa
     "group_factor": number,       # 9. (sea salt), 5 (sulfate), 8 (OC), 1
     "order": int,                 # position inside the group
     "unit_mass": number,          # get_niwfa's denominator, as written
     "mass_multiplier": number,    # optional native mass -> GSL mass basis
     "from": "file:line"}

GSL's denominators are single-precision literals dividing a REAL(8), so the
launcher widens ``float32(unit_mass)`` exactly as gfortran does.  The row's
native units become kg/kg by a factor derived from the row, never from its
name: ``ug kg-1`` rows multiply by 1e-9; ``ppmv`` rows by
``1e-6 * molar_mass_g_mol / 28.966`` (``mwdry``, share/module_model_constants.F:34),
the same conversion sum_pm_gocart applies to sulfate (module_gocart_aerosols.F:
100).  That conversion is WOOF-ArWen's: GSL is handed MERRA-2 kg/kg directly.

WHEN.  After every chem step (DESIGN 3, step 10), so the next microphysics
call reads numbers diagnosed from the species it will act beside.  GSL
diagnoses at the start of each microphysics call instead; between the two
points WOOF-ArWen's ``nwfa``/``nifa`` are advected as the scalars they are, the
species beside them, so the two differ by one step of transport of each.
Thompson's own activation sinks do not feed back into species mass, as in
GSL's coupling.

This process moves no species mass, so its ledger bucket (``coupling``)
closes at zero by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KEY = "coupling.thompson"
LEDGER = "coupling"
#: The kernel translation units step() launches (priced by the preflight).
KERNEL_MODULES = ("chem_mp_coupling",)
REQUIRES: tuple[str, ...] = ()
ALLOCATES: tuple = ()

#: get_niwfa's final multiplier, ``1.e15``: a single-precision literal against
#: a REAL(8), so the widened float32, 999999986991104.0.
SCALE = float(np.float32(1.0e15))

#: WRF's dry-air molar mass (share/module_model_constants.F:34, REAL 28.966).
#: The ppmv -> kg/kg factor is ArWen's own conversion, evaluated in float64.
MWDRY = 28.966

TARGETS = ("nifa", "nwfa")


def rows(table):
    return table.rows_for(KEY)


def effective_mode(cfg):
    """Source-backed aerosol-aware Thompson diagnoses numbers by default."""
    mode = getattr(cfg, "aerosol_mp_coupling", "auto")
    if mode != "auto":
        return mode
    if int(getattr(cfg, "mp_physics", 0)) != 28:
        return "none"
    from woof.chem_table import load
    from woof.chem_source_init import chem_boundary_fields
    table = load(cfg)
    if table is None:
        return "none"
    filled = set(chem_boundary_fields(table, cfg))
    return ("diagnose" if any(row.state_attr in filled for row in rows(table))
            else "none")



def source_number_fields(cfg):
    """Number targets whose composition mass is supplied on every frame."""
    if effective_mode(cfg) != "diagnose":
        return ()
    from woof.chem_table import load
    from woof.chem_source_init import chem_boundary_fields
    table = load(cfg)
    if table is None:
        return ()
    filled = set(chem_boundary_fields(table, cfg))
    targets = {r.mp_coupling["target"] for r in rows(table)
               if r.state_attr in filled and r.mp_coupling}
    return tuple(name for name in ("nwfa", "nifa") if name in targets)


def refusal(cfg):
    """Why ``aerosol_mp_coupling = 'diagnose'`` cannot run as configured, or None.

    get_niwfa OVERWRITES a target's number field every chem step from the
    rows that feed it; GSL hands it MERRA-2 aerosol, which is never empty.
    Here the rows start at zero unless a data-store source fills them
    (WRF-Chem ``have_bcs_chem``, :func:`woof.chem_source_init.chem_boundary_fields`),
    so with no such source the first diagnosis replaces Thompson's aerosol
    with almost none and every cloud forms in near-clean air.  A target no
    active row feeds is left to the scheme (``launch_niwfa``), so only
    targets with feeding rows are checked.
    """
    if effective_mode(cfg) != "diagnose":
        return None
    from woof.chem_source_init import chem_boundary_fields
    from woof.chem_table import load
    table = load(cfg)
    if table is None:
        return None
    filled = set(chem_boundary_fields(table, cfg))
    for target in TARGETS:
        feeding = [row for row in rows(table)
                   if row.mp_coupling and row.mp_coupling.get("target") == target]
        if feeding and not any(row.state_attr in filled for row in feeding):
            names = sorted(row.name for row in feeding)
            return (f"aerosol_mp_coupling = 'diagnose' overwrites {target} every "
                    f"chem step from rows {names}, and none of them takes its "
                    "start and edge values from a data-store source (chem_sources "
                    "enables none that fills them, such as 'cams-global'), so they "
                    f"start at zero and {target} would collapse to almost nothing; "
                    "enable an aerosol source that fills these rows or set "
                    "aerosol_mp_coupling = 'none'")
    return None


@dataclass(frozen=True)
class CouplingTerm:
    """One row's place in get_niwfa, packed at the routine's precision."""

    name: str
    target: int          # index into TARGETS
    group: int
    order: int
    group_factor: float  # float64
    denominator: float   # float64 value of the float32 literal
    to_kg_per_kg: float  # float64


def to_kg_per_kg(units: str, molar_mass_g_mol: float | None) -> float:
    if units == "ug kg-1":
        return 1.0e-9
    if units == "ppmv":
        if not molar_mass_g_mol:
            raise ValueError(
                "a ppmv row coupled to Thompson needs molar_mass_g_mol for "
                "its kg/kg conversion")
        return 1.0e-6 * float(molar_mass_g_mol) / MWDRY
    raise ValueError(f"units {units!r} have no kg/kg conversion")


def pack_terms(row_dicts) -> list[CouplingTerm]:
    """Rows (``SpeciesRow`` or plain mappings) -> sorted coupling terms.

    Sorted by (target, group, order), the order get_niwfa sums in.  Two rows
    claiming one (target, group, order) slot are refused: the sum would
    depend on table order, which fixes arena slots and nothing else.
    """
    terms = []
    seen = {}
    for row in row_dicts:
        get = (row.get if isinstance(row, dict)
               else lambda key, default=None, _r=row: getattr(_r, key, default))
        spec = get("mp_coupling")
        name = get("name")
        if not spec:
            raise ValueError(f"row {name!r} names {KEY} but has no mp_coupling")
        target = spec["target"]
        if target not in TARGETS:
            raise ValueError(f"row {name!r}: mp_coupling.target {target!r} "
                             f"is not one of {TARGETS}")
        slot = (target, int(spec["group"]), int(spec["order"]))
        if slot in seen:
            raise ValueError(f"rows {seen[slot]!r} and {name!r} both claim "
                             f"get_niwfa slot {slot}")
        seen[slot] = name
        terms.append(CouplingTerm(
            name=name,
            target=TARGETS.index(target),
            group=int(spec["group"]),
            order=int(spec["order"]),
            group_factor=float(spec["group_factor"]),
            denominator=float(np.float32(spec["unit_mass"])),
            to_kg_per_kg=to_kg_per_kg(get("units"),
                                      get("molar_mass_g_mol", None))
                            * float(spec.get("mass_multiplier", 1.0))))
    terms.sort(key=lambda t: (t.target, t.group, t.order))
    for t in terms:
        if t.group_factor <= 0 or t.denominator <= 0:
            raise ValueError(f"row {t.name!r}: group_factor and unit_mass "
                             "must be positive")
    return terms


def launch_niwfa(fields, terms, nifa, nwfa, *, nifa_f64=None, nwfa_f64=None):
    """Overwrite ``nifa``/``nwfa`` (float32, any shape) from ``fields``.

    ``fields`` maps each term's row name to its float32 device field, all the
    shape of ``nifa``.  A target with no term is left untouched.  The float64
    arrays, when given, receive the value before the float32 rounding (the
    oracle grades both).
    """
    if isinstance(nifa, np.ndarray):
        from woof.ingest.cpu_backend import CpuPreprocessBackend
        CpuPreprocessBackend().aerosol_numbers(
            fields, terms, nifa, nwfa, scale=SCALE,
            nifa_f64=nifa_f64, nwfa_f64=nwfa_f64)
        return
    import cupy as cp

    from woof.core.kernels import get_kernel

    if not terms:
        return
    shape = nifa.shape
    for t in terms:
        f = fields[t.name]
        if f.dtype != cp.float32 or f.shape != shape or not f.flags.c_contiguous:
            raise ValueError(f"field for {t.name!r} must be C-contiguous "
                             f"float32 {shape}")
    for arr in (nifa, nwfa):
        if arr.dtype != cp.float32 or arr.shape != shape or not arr.flags.c_contiguous:
            raise ValueError("nifa/nwfa must be C-contiguous float32 of one shape")
    ptr = cp.asarray(np.array([fields[t.name].data.ptr for t in terms],
                              dtype=np.uint64))
    conv = cp.asarray(np.array([t.to_kg_per_kg for t in terms], np.float64))
    den = cp.asarray(np.array([t.denominator for t in terms], np.float64))
    tgt = cp.asarray(np.array([t.target for t in terms], np.int32))
    grp = cp.asarray(np.array([t.group for t in terms], np.int32))
    gfac = cp.asarray(np.array([t.group_factor for t in terms], np.float64))
    ncell = int(np.prod(shape))
    threads = 256
    blocks = (ncell + threads - 1) // threads
    kern = get_kernel("chem_mp_coupling", "chem_mp_niwfa")
    null = np.uint64(0)
    kern((blocks,), (threads,),
         (ptr, conv, den, tgt, grp, gfac, np.int32(len(terms)),
          np.float64(SCALE), nifa, nwfa,
          nifa_f64 if nifa_f64 is not None else null,
          nwfa_f64 if nwfa_f64 is not None else null,
          np.int64(ncell)))


def init(ctx):
    mode = effective_mode(ctx.cfg)
    if mode != "diagnose":
        return
    for name in TARGETS:
        if getattr(ctx.state, name, None) is None:
            raise ValueError(
                f"aerosol_mp_coupling='diagnose' writes state.{name}, which "
                "only the aerosol-aware Thompson scheme (mp_physics=28) "
                "carries")
    pack_terms(rows(ctx.table))


def step(ctx, dt, ktau):
    if effective_mode(ctx.cfg) != "diagnose":
        return
    terms = pack_terms(rows(ctx.table))
    fields = {t.name: ctx.field(ctx.table.row(t.name)) for t in terms}
    launch_niwfa(fields, terms, ctx.state.nifa, ctx.state.nwfa)


def initialize_from_sources(state, cfg, receipts):
    """Bind source aerosol mass before the first microphysics call."""
    if not receipts or effective_mode(cfg) != "diagnose":
        return None
    terms = pack_terms(rows(state.chem.table))
    filled = {name for receipt in receipts for name in receipt["rows"]}
    for target in TARGETS:
        feeding = [term for term in terms if term.target == TARGETS.index(target)]
        if feeding and not any(term.name in filled for term in feeding):
            raise ValueError(f"initial {target} diagnosis has no populated aerosol source")
    for target in TARGETS:
        if getattr(state, target, None) is None:
            raise ValueError(f"initial aerosol coupling needs state.{target}; select aerosol-aware Thompson")
    fields = {t.name: getattr(state, state.chem.table.row(t.name).state_attr) for t in terms}
    launch_niwfa(fields, terms, state.nifa, state.nwfa)
    initialized = [name for name in ("nwfa", "nifa")
                   if any(TARGETS[t.target] == name for t in terms)]
    result = dict(aerosol_source="composition-chemistry", dataset_used=True,
                source_rows=[r["source"] for r in receipts],
                aerosol_source_statement=("Thompson " + ", ".join(initialized) + " diagnosed from populated composition before first microphysics; other fields retain their recorded initialization policy"),
                policy="source-mass-to-number-before-cold-start-closure",
                operator="NOAA GSL get_niwfa", initialized_fields=initialized)
    if set(initialized) == set(TARGETS):
        result.update(synthetic_fallback_in_use=False, awaiting_profile_fill=False)
    return result
