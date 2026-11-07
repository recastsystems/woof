"""Active boundary-field inventory shared by initialization and memory pricing."""
from __future__ import annotations

from woof.config import p3_variant
from woof.config import RunConfig, is_p3, p3_predicts_nc
from woof.core.mp_column_schemes import (SIX_MASS_TABLE_SCHEMES,
                                         TABLE_SCHEMES)


def nest_field_kinds(cfg: RunConfig) -> tuple[str, ...]:
    """Child forcing field kinds: u/v/w/t(thm)/ph/mu + ALL active
    moist/scalar species including Thompson/Morrison numbers (architecture
    section D;
    module_bc_em.F:320-345 w coupling; Registry scalar set qnr/qni/qns/
    qng at Registry.EM_COMMON:3026).

    ``nc`` is scheme-dependent, and the reason is the advection copy.  For
    mp_physics=10 it is EXCLUDED: Morrison allocates ``nc`` but has no
    ``nc0``, does not transport it, and diagnoses a fixed 250 cm-3 every
    call, so forcing it across a nest edge would carry a field the child
    immediately overwrites.  For mp_physics=28 it is INCLUDED: aerosol-aware
    Thompson makes droplet number prognostic, allocates ``nc0`` and advects
    it alongside nwfa/nifa (woof/core/moist.py::THOMPSON_AERO_NUMBER_SPECIES),
    so it is a real forced boundary field like ``nr``/``ni``.  The two facts
    are not in tension -- the inventory follows what is transported, not what
    is allocated.

    The three ice MASSES are scheme-dependent for the same reason, and
    ``mp_physics=50`` is EXCLUDED from that block deliberately.  What this
    inventory decides is not "does the scheme have ice" but "which of WRF's
    boundary-forced Registry members does the scheme activate": every kind
    named here is declared ``ikjftb`` with ``i0rhusdf=(bdy_interp:dt)`` --
    the ``moist`` masses at Registry.EM_COMMON:452-469 and the ``scalar``
    moments at :520-558.  P3's package is
    ``moist:qv,qc,qr,qi;scalar:qni,qnr,qir,qib`` (Registry.EM_COMMON:3038):
    ONE ice mass, no ``qs`` and no ``qg``, which is why the mp=50 arm below
    brings its own ``qi`` instead of joining the tuple.  WRF's driver says
    the same thing -- the ``P3_1CATEGORY`` arm passes no snow and no
    graupel array at all (module_microphysics_driver.F:1557-1602, against
    ``mp_p3_wrapper_wrf``'s signature at module_mp_p3.F:690-699).  Folding
    50 in would name ``qi`` twice and price sixteen rolling boundary tables
    for two species an mp=50 child never allocates (woof/core/state.py's
    mp=50 arm), and the first ``force`` would then raise ``state has no
    active nest field 'qs'`` (woof/core/nest.py:186-193).

    P3's remaining package members -- ``state:re_cloud,re_ice,vmi3d,
    rhopo3d,di3d,refl_10cm,th_old,qv_old`` -- are absent for a DIFFERENT
    reason, recorded here so it is not read as a second omission: they are
    ``misc`` state with no ``b`` in the io string (th_old/qv_old at
    Registry.EM_COMMON:1598-1599), so WRF builds no boundary arrays for
    them and ``bdy_interp`` never touches them.  ``p3_main`` rewrites
    th_old/qv_old at the end of every call (module_mp_p3.F:5018-5021);
    they are cross-step carriers the child regenerates, not forced
    boundary fields.

    This boundary-forcing inventory contains REAL prognostics handled by
    ``copy_fcn``. Surface feedback has its own Registry-derived inventory
    in ``nest_surface_fields``: those fields carry no boundary strips and
    use WRF's masked/integer operators in the feedback transaction.
    Selected Noah-MP uses its own native-flag inventory in
    ``noahmp_nest.feedback_fields`` for surface feedback.
    Other carried physics fields use ``nest_physics_feedback`` with their
    Registry-selected runtime feedback operators.
    """
    kinds = ["u", "v", "w", "t", "ph", "mu"]
    if cfg.moist:
        kinds += ["qv", "qc", "qr"]
        from woof.physics_registry import consumer_row_for_selector
        native = consumer_row_for_selector("microphysics", "native_column", cfg.mp_physics)
        if native:
            kinds += [n for n in native["transport"] if n not in ("qv", "qc", "qr")]
        from woof.physics_registry import column_microphysics_binding
        binding = column_microphysics_binding(cfg.mp_physics)
        if binding is not None:
            kinds += binding['transported_extras']
        from woof.core.column_schemes import COLUMN_SCHEMES
        if cfg.mp_physics in COLUMN_SCHEMES:
            kinds += [n for n in COLUMN_SCHEMES[cfg.mp_physics].species if n not in ("qv", "qc", "qr")]
        from woof.core.wdm6_constants import wdm_scheme
        family = wdm_scheme(cfg.mp_physics)
        if family:
            kinds += [n for n in family.mass_species if n not in ("qv", "qc", "qr")]
            kinds += list(family.number_species)
        elif cfg.mp_physics in (2, 6, 8, 9, 10, 16, 18, 24, 27, 28, 29, 38, 40, 106) + SIX_MASS_TABLE_SCHEMES:
            kinds += ["qi", "qs", "qg"]
        if cfg.mp_physics == 30:
            from woof.core.fast_sbm_contract import EXTRA_SPECIES
            kinds += list(EXTRA_SPECIES)
        if cfg.mp_physics == 24:
            # WSM7's hail mass (woof/core/moist.py
            # SINGLE_MOMENT_HAIL_SPECIES) crosses a nest edge with the rest.
            kinds += ["qh"]
        if cfg.mp_physics == 27:
            kinds += ["qh", "nn", "nc", "nr"]
        if cfg.mp_physics == 56:
            from woof.core.ntu import NTU_SPECIES
            kinds += list(NTU_SPECIES)
        if cfg.mp_physics == 95:
            kinds += ["qs", "qt"]
        if cfg.mp_physics == 9:
            # The inventory follows what is transported
            # (woof/core/moist.py::MY2_SPECIES): hail mass plus all six
            # number moments cross a nest edge with the masses they
            # describe (1.9.1 D1's route: mp=9 had no arm here, so a
            # nested Milbrandt-Yau child would have been forced with
            # WSM6's field set).
            kinds += ["qh", "nc", "nr", "ni", "ns", "ng", "nh"]
        if cfg.mp_physics == 96:
            kinds += ["qi", "qs"]
        if cfg.mp_physics == 13:
            # SBU-YLin (Registry.EM_COMMON:3029): qi and qs cross a nest
            # edge; rimi is a WRF ``state`` field with no interpolation
            # flag, rewritten by the child's own first call.
            kinds += ["qi", "qs"]
        if cfg.mp_physics == 8:
            kinds += ["nr", "ni"]
        if cfg.mp_physics in THOMPSON_AEROSOL_FAMILY:
            kinds += ["nr", "ni", "nc", "nwfa", "nifa"]
            kinds += list(THOMPSON_AEROSOL_FAMILY[cfg.mp_physics].graupel_species)
        if cfg.mp_physics in (10, 40):
            kinds += ["nr", "ni", "ns", "ng"]
            if cfg.mp_physics == 40:
                kinds += ["nc"]
        if cfg.mp_physics == 18:
            kinds += ["qh", "qndrop", "qnr", "qni", "qns", "qng",
                      "qnh", "qnn", "qvolg", "qvolh"]
        from woof.core.fer_hires import package_for
        ferrier_package = package_for(cfg.mp_physics)
        if ferrier_package:
            # MP5 transports total ice; MP15 also transports mass-weighted
            # rime. MP5's f_* state and qt do not cross the nest boundary.
            kinds += ferrier_package["transported"]
        _p3 = p3_variant(cfg.mp_physics)
        if _p3 is not None:
            # P3 one-category: the inventory follows what is transported
            # (woof/core/moist.py::P3_SPECIES / P3_NC_SPECIES), so the rime mass/volume
            # pair is forced across a nest edge exactly like the number
            # moments.  Forcing qi without them would hand the child ice
            # whose rime fraction and rime density came from whatever the
            # child's own last step left behind.  DELIBERATELY not folded
            # into the qi/qs/qg tuple above: this arm carries P3's single
            # ice mass itself, because Registry.EM_COMMON:3038 gives the
            # scheme no qs and no qg (the docstring carries the
            # consequence of widening that tuple).
            from woof.core.p3_tables import P3_TRANSPORT_FIELDS
            kinds += list(P3_TRANSPORT_FIELDS[cfg.mp_physics])
        # A named scheme (woof.microphysics_schemes): its capability row is
        # the transported set, the same rows moist.extra_moist_species
        # returns for it, so each crosses a nest edge with its masses.
        from woof.microphysics_schemes import scheme as _named_scheme
        named = _named_scheme(int(cfg.mp_physics))
        if named is not None:
            kinds += [name for name in (named.ice_mass_species
                                        + named.moment_species
                                        + named.aerosol_species)
                      if name not in kinds]
    # Read like chem_sets below: a caller may hand a partial configuration
    # (the offline-child inventory passes only the microphysics fields).
    if (getattr(cfg, "bl_pbl_physics", 0) == 5
            and getattr(cfg, "bl_mynn_tkeadvect", False)):
        kinds.append("qke_adv")
    if getattr(cfg, "bl_pbl_physics", 0) == 16:
        from woof.core.pbl_scalar_inventory import pbl_transported_scalars
        kinds.extend(pbl_transported_scalars(cfg))

    if getattr(cfg, "bl_pbl_physics", 0) == 17:
        from woof.core.physics_inventory import pbl_advected_scalars
        kinds.extend(pbl_advected_scalars(cfg))
    # Chem species (woof/core/chem_state.py), named by the chem table: WRF
    # declares every chem species with the nest interpolation and feedback
    # flags (registry.chem ``i0{12}rhusdf=(bdy_interp:dt)``) and forces a
    # nest's chem with relax plus spec (solve_em.F:2509-2545), the generic
    # scalar coupling of the moments above.  Empty while chem is off.
    if getattr(cfg, "chem_sets", ()):
        from woof.chem_table import load as load_chem_table
        kinds += [row.state_attr for row in load_chem_table(cfg).transported]
    from woof.core.passive_tracer_contract import active_tracers
    kinds += active_tracers(cfg)
    return tuple(kinds)

from woof.config import THOMPSON_AEROSOL_FAMILY
