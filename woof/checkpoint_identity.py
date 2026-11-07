"""The name a checkpoint gives every scheme, and the gate that asks first.

A restart file records the identity of every scheme that integrated it, so
a resume onto a different implementation fails BEFORE it restores rather
than continuing a trajectory that is not the one it claims.  The tables
below are those identities, and the gate below them is the question
``validate_run_config`` asks at plan review: can the checkpoint writer name
everything this configuration selected?

WHY THEY LIVE IN A TOP-LEVEL MODULE rather than beside the writer in
:mod:`woof.io.restart`, which is where they were written and where every
reader in the tree still spells them.  The gate is called from
:func:`woof.config.validate_run_config` -- every door's plan review --
and ``woof/config.py`` ships in distributions that stage no forecast
executor and therefore no ``woof/io`` at all: the standalone RW-WPS
preprocessing project stages ``woof/*.py`` and an explicit handful of
``woof/io`` modules, and ``restart.py`` is deliberately not one of them
(it reaches the CUDA side).  A gate imported out of ``woof.io.restart``
is therefore unresolvable in a preparation-only install -- the plan review
every front door runs would raise ``ModuleNotFoundError`` on a
configuration it should simply have accepted.

Answering that with an exception in the packaging tables would have left
the same import waiting to be reached at run time.  These are pure
configuration facts -- integers, strings and dicts -- so the module that
holds them imports nothing from ``woof.io``, nothing from CuPy, and
resolves its two configuration helpers inside the one function that needs
them.  It stages, and imports, wherever ``woof/config.py`` does.

:mod:`woof.io.restart` imports every name here and re-exports it under
the spelling the tree already uses, so there is ONE copy of each table and
``woof.io.restart.MICROPHYSICS_ALGORITHM_IDENTITIES`` is that copy.
"""


from __future__ import annotations


#: The 2.9.4 capabilities campaign's start-only and door keys at the value
#: every earlier header ran: development branch's ocean door (no ocean
#: contract) and development branch's lake-state analysis (an initialization
#: input; a resumed lake continues from its checkpointed columns).  Absent
#: means this value on either side of a resume; any other value binds.
CAMPAIGN_IMPLICIT_RUN_KEYS = {
    # The shared fog/KF defaults are absent in both writer and reader.
    "grav_settling": 0,
    "kfeta_trigger": 1,
    "smoke_off_reason": "",
    "ocean_config": "",
    "waves": "off",
    "waves_options": "",
    "lake_state_source": "auto",
    "lake_state_path": "",
    "lake_state_accept_licence": "",
    # 2.9.10 (fix/wrf461-2dom-exact): WRF's ghg_input, read only by legacy
    # RRTMG and RRTM longwave; 0 (the engine's fixed gases) is absent so
    # every earlier checkpoint resumes and every public document keeps its
    # bytes, and 1 binds.  Named breakage: with it always present every
    # checkpoint digest, echo and public document moved (2.9.10 pass 2).
    "ghg_input": 0,
    "cu_rad_feedback": False,
    "rrtmg_aer3_source": "operational_fork",
    # Disabled moving frames preserve all earlier document/restart identities.
    "umove": 0.0,
    "vmove": 0.0,
    "pxlsm_smois_init": 1,
    "pxlsm_modis_veg": 0,
    "pxlsm_soil_nudge": 0,
    "sgfdda_interval_m": 180,
    "sgfdda_inname": "sgfdda_d<domain>",
    "io_form_sgfdda": 2,
}

#: development branch: prognostic sea ice and its parameters are read only
#: where a land surface runs it (woof.ocean.seaice: ``seaice_prognostic``
#: and ``sf_surface_physics``).
SEAICE_RUN_KEYS = ("seaice_prognostic", "seaice_initial_thickness",
                   "seaice_new_ice_thickness", "seaice_ocean_heat_flux")

#: development branch: HAILCAST's output-only selectors at their defaults
#: (CONFIG_DIAGNOSTIC_FIELDS keeps them out of the resume walk).
HAILCAST_DEFAULT_RUN_KEYS = {"hailcast_opt": -1, "haildt": 60.0,
                             "hailcast_version": "wrf_39_noaa"}


def seaice_read(values: dict) -> bool:
    return bool(values.get("sf_surface_physics", 0)) and bool(
        values.get("seaice_prognostic", False))


def drop_unread_mp_switches(values: dict) -> None:
    """Keep only switches read by the selected column microphysics scheme."""
    from woof.core.mp_column_schemes import MP_COLUMN_SCHEMES
    selected = MP_COLUMN_SCHEMES.get(int(values.get("mp_physics", 0) or 0))
    owned = set(selected.switches) if selected is not None else set()
    for row in MP_COLUMN_SCHEMES.values():
        for name in row.switches:
            if name not in owned:
                values.pop(name, None)


def drop_capability_campaign_keys(values: dict) -> None:
    """Omit the campaign keys while they reach nothing.

    Shared by the checkpoint echo and digest (woof.io.restart) and the
    public experiment document (woof.experiment), so a default run keeps
    the 2.9.3 bytes in both and an active value binds in both.  Named
    breakage: every 2.9.3 checkpoint refused on resume ("lake_state_path:
    absent from the restart file") and every public document's bytes moved.
    """
    # lic-snicar's seven controls are read only by Noah-MP opt_alb=3.
    # Keeping them in an inactive echo changed every pre-port checkpoint's
    # configuration hash despite identical stored arrays (test_restart).
    # This projection is shared by public documents, writers and resume.
    if (int(values.get("sf_surface_physics", 0) or 0) != 4
            or int(values.get("opt_alb", 2)) != 3):
        for name in ("snicar_snowshape_opt", "snicar_snowbc_intmix",
                     "snicar_snowdust_intmix", "snicar_bc_ppb",
                     "snicar_dust_ppm", "snicar_ice_density",
                     "snicar_bubble_radius_um"):
            values.pop(name, None)
    drop_unread_mp_switches(values)
    # Only NTU reads its aerosol population selector. Earlier non-NTU
    # checkpoint and experiment identities contain no such key.
    if int(values.get("mp_physics", 0) or 0) != 56:
        values.pop("ntu_ccnty", None)
    if int(values.get("mp_physics", 0) or 0) != 40 and values.get("aercu_opt", 0) == 0:
        values.pop("aercu_opt", None)
    # The QNSE-only switch did not exist in earlier configurations.
    # Keep inactive checkpoint and experiment identities unchanged.
    if values.get("bl_pbl_physics", 0) != 4:
        values.pop("mfshconv", None)

    from woof.level_output_config import drop_inert as drop_inert_levels
    drop_inert_levels(values)
    from woof.core.misc_buckets import drop_inert as drop_inert_buckets
    drop_inert_buckets(values)
    from woof.core.physics_tendency_output import drop_inert as drop_inert_tendencies
    drop_inert_tendencies(values)
    from woof.core.misc_diag_print import drop_inert as drop_inert_print
    drop_inert_print(values)
    from woof.afwa_config import drop_inert as drop_inert_afwa
    drop_inert_afwa(values)
    # Newly admitted RUN5 alone reads WTDDT. Preserve earlier default
    # experiment and checkpoint bytes while binding every active cadence.
    if int(values.get("sf_surface_physics",0)) != 4 or int(values.get("opt_run",3)) != 5:
        values.pop("wtddt",None)
    # Only Noah-MP's papers-based snow optics reads these controls. Keeping
    # them in an inactive identity refuses checkpoints written before the
    # controls existed and changes default public experiment documents.
    if (int(values.get("sf_surface_physics", 0)) != 4
            or int(values.get("opt_alb", 2)) != 3):
        for name in ("snicar_snowshape_opt", "snicar_snowbc_intmix",
                     "snicar_snowdust_intmix", "snicar_bc_ppb",
                     "snicar_dust_ppm", "snicar_ice_density",
                     "snicar_bubble_radius_um"):
            values.pop(name, None)
    for name, default in CAMPAIGN_IMPLICIT_RUN_KEYS.items():
        if values.get(name, default) == default:
            values.pop(name, None)
    if not seaice_read(values):
        for name in SEAICE_RUN_KEYS:
            values.pop(name, None)
    for name, default in HAILCAST_DEFAULT_RUN_KEYS.items():
        if values.get(name, default) == default:
            values.pop(name, None)
    if not int(values.get("dfi_opt",0)):
        from woof.dfi_control import DFI_KEYS
        for name in DFI_KEYS:
            values.pop(name,None)

    if not values.get("obs_nudge_opt", 0):
        # Disabled FDDA allocates nothing and preserves earlier public
        # documents and checkpoint identities byte for byte.
        for key in tuple(values):
            if key.startswith("obs_") or key in ("max_obs", "fdda_start", "fdda_end"):
                values.pop(key, None)
    # 4c82704f8f's seaice_threshold is a real.exe rule read only by the
    # initial surface; the Registry default 100 K converts nothing, so it is
    # absent there and a stated threshold binds.  Named breakage: every
    # default public experiment document's bytes moved (the checkpoint echo
    # already omitted it, woof.io.restart).
    if float(values.get("seaice_threshold", 100.0)) == 100.0:
        values.pop("seaice_threshold", None)


def drop_default_output_controls(values: dict) -> None:
    """Keep pre-output-control document and checkpoint bytes at their defaults.

    Nondefault output requests remain explicit. The GSD reset cadence is
    implicit only while its diagnostic package is off at the default cadence.
    """
    if not values.get("tslist", ""):
        for key in ("tslist", "ts_buf_size", "max_ts_locs", "max_ts_level", "tslist_unstagger_winds"):
            values.pop(key, None)
    for key in ("solar_diagnostics", "mean_diag", "diurnal_diag", "mean_freq", "mean_interval", "auxhist3_interval", "do_avgflx_em", "do_avgflx_cugd",
                "prec_acc_dt", "prec_acc_dt1", "history_interval2_s",
                "history_interval_change_s"):
        if float(values.get(key, 0.0) or 0.0) == 0.0:
            values.pop(key, None)
    if int(values.get("output_diagnostics", 0) or 0) == 0:
        values.pop("output_diagnostics", None)
        for key, default in (("diag_int", 15), ("wind_int", 5)):
            if values.get(key, default) == default:
                values.pop(key, None)
    if int(values.get("uh_layers", 0) or 0) == 0:
        values.pop("uh_layers", None)
    if int(values.get("gsd_diagnostics", 0) or 0) == 0:
        values.pop("gsd_diagnostics", None)
        if int(values.get("reset_interval1", 60)) == 60:
            values.pop("reset_interval1", None)


def drop_default_cumulus_selectors(values: dict) -> None:
    """Keep earlier identity bytes while binding every consumed setting."""
    from woof.config import CUMULUS_PARAMETER_IDENTITY_RULES
    for name, (default, readers) in CUMULUS_PARAMETER_IDENTITY_RULES.items():
        if (values.get(name, default) == default
                and (not readers or values.get("cu_physics", 0) not in readers)):
            values.pop(name, None)


def drop_default_sms_selectors(values: dict) -> None:
    """Preserve identity bytes when the added SMS controls are unchanged."""
    # These controls arrived with SMS-3DTKE. Their defaults preserve the
    # earlier transport and stress paths, so old checkpoint and experiment
    # bytes must omit them. Explicit nondefaults still bind on continuation.
    for key, default in (("tke_adv_opt", 1), ("sfs_opt", 0), ("m_opt", 0),
                         ("tke_mix2_off", False), ("tke_mix6_off", False)):
        if values.get(key, default) == default:
            values.pop(key, None)


def drop_default_surface_port_keys(values: dict) -> None:
    """The unchanged surface options have the identity of older checkpoints.

    Lives here, not in woof.io.restart, because woof.experiment (staged in
    the standalone RW-WPS package) applies it to public config documents and
    the restart writer is forecast-executor code that package never carries.
    """
    for name, default in (("seaice_albedo_opt",0),("seaice_thickness_opt",0),
                          ("seaice_snowdepth_opt",0),("seaice_thickness_default",3.0),
                          ("seaice_snowdepth_min",.001),("seaice_snowdepth_max",1.e10)):
        if values.get(name,default)==default: values.pop(name,None)
    if not values.get('ua_phys',False): values.pop('ua_phys',None)
    if not values.get('sf_surf_irr_scheme', 0):
        for key in ('sf_surf_irr_scheme', 'irr_daily_amount', 'irr_start_hour',
                    'irr_num_hours', 'irr_start_julianday', 'irr_end_julianday',
                    'irr_freq', 'irr_ph'):
            values.pop(key, None)


def drop_default_diffusion_selectors(values: dict) -> None:
    """Preserve pre-selector identity bytes for the unchanged metric form."""
    drop_default_sms_selectors(values)
    if values.get("diff_opt", 2) == 2 and values.get("mix_full_fields", True) is True:
        values.pop("diff_opt", None)
        values.pop("mix_full_fields", None)


def drop_default_spp_selectors(values: dict) -> None:
    """Keep newly added SPP selectors absent when disabled.

    spp_lsm predates the SPP consumer port and must keep its existing echo.
    """
    for name in ("spp_conv", "spp_pbl"):
        if values.get(name, 0) == 0:
            values.pop(name, None)

#: THE table of output-only RunConfig switches: each one decides what a
#: forecast WRITES, never what it integrates, so it may differ between
#: the run that wrote a checkpoint and the run that resumes it, and
#: between the run that prepared a cache and the run that reads it.
#: Every place that asks "does this switch change the model" reads this
#: one set: the member restart walk and the configuration digest in
#: :mod:`woof.io.restart` (re-exported there under this name), the tree
#: restart identity comparison beside them, and the prepared-cache
#: identity in :mod:`woof.ingest.prepared_cache`.  Two copies of this
#: list drifted once already: the prepared cache knew only the first
#: member, so switching any of the other three on refused an unchanged
#: cache and forced a second preparation of identical arrays.
#:
#: Membership is a proof, not a label: the switch reads model state and
#: writes only its own diagnostic buffers.
#:
#: * ``nwp_diagnostics`` -- inertness pinned by tests/test_uh_lifecycle.py;
#:   its accumulator payloads are tolerant in both directions (missing in
#:   the file restores as zeros with a note; present under a
#:   diagnostics-off resume is dropped with a note).
#: * ``tke_budget`` -- the accumulator writes only its own scratch
#:   (tests/test_tke_budget.py).
#: * ``sase_flux_diag`` -- four history buffers filled from arrays the
#:   SASE step already holds (tests/test_sase_gpu.py, every prognostic
#:   byte-identical with the switch on).
#: * ``hmix_k_diag`` -- two history buffers copied from the eddy
#:   viscosity the run's own mixing producer computed; nothing reads them
#:   back (tests/test_sase_gpu.py, every prognostic byte-identical with
#:   the switch on, under both producers).
#: * ``surface_energy_diag`` -- publishes the physics driver's existing
#:   albedo, emissivity and GSW arrays to history and leaves the
#:   configuration digest unchanged (tests/test_urban_history_energy.py).
#:
#: A newly switched-on diagnostic that a positive PBL cadence carries
#: between steps restores as zeros, its cold value, and is refilled by
#: the next due step.  The physics selectors beside them
#: (``sase_moist_n2``, ``sase_stable_dissipation``,
#: ``sase_additive_dissipation``) move the trajectory and are NOT here.
CONFIG_DIAGNOSTIC_FIELDS = frozenset(
    {"hailcast_opt", "haildt", "hailcast_version", "nwp_diagnostics", "tke_budget", "sase_flux_diag", "hmix_k_diag",
     "surface_energy_diag",
     # The 0-1/0-2/0-3 km updraft-helicity maxima (woof/core/uh_layers.py):
     # output only, they change no prognostic word.
     "uh_layers",
     # Correct mode WB1 (woof/correct_mode.py OUTPUT_ONLY_KEYS): books the
     # settled-cloud total CLOUDNC and changes no prognostic word.
     "thompson_cloud_surface_booking"})

#: Versioned semantic identities.  These are deliberately explicit instead
#: of inferred from scheme numbers: a trajectory-changing implementation or
#: policy change must advance its tag, causing an incompatible restart to
#: fail before restore.  Asset bytes and resolved per-run values are bound
#: by the writer in :mod:`woof.io.restart`.
#:
#: WHAT DOES NOT GET A ROW, so the next reader does not add one and call
#: it completeness.  A row exists for an input CONFIG CANNOT PROVE: a
#: resolved callable, packaged asset bytes, a resolved cadence, a policy
#: chosen at setup.  Every selector config DOES prove is already bound,
#: twice: the writer's ``configuration_sha256`` hashes the whole RunConfig
#: minus the run-length and diagnostic fields, and the resume path
#: compares every field by name before it compares that hash -- so a
#: closure change refuses with the field named, not with a digest
#: mismatch.  ``km_opt`` and its constants (c_k, c_s, khdif, kvdif,
#: mix_isotropic, mix_full_fields, diff_opt, diff_6th_opt) are the worked
#: example: none is excluded from the fingerprint, none has a packaged
#: asset or a resolved callable, and giving the turbulence closure a table
#: here would add a SIXTH authority over the five registry rows that
#: already describe it -- the drift surface these tables exist to avoid --
#: while invalidating every v2 checkpoint on disk, since any added key
#: moves the physics-setup fingerprint.  The residual a row would close
#: (same selector, differently transcribed kernels) is the whole dycore's,
#: not km_opt's, and it is closed by ONE string,
#: :data:`DYCORE_MIXING_ALGORITHM_IDENTITY` below, not a per-option table.
MICROPHYSICS_ALGORITHM_IDENTITIES = {
    96: "madwrf-wrf471-passive-cloud-v1",
    40: "morrison-aero-wrf461-v1-fixed-droplets-pinned-t0-finite-freezing",
    2: "lin-wrf461-six-class-v1",
    30: "fast-sbm-wrf461-33bin-corrected-d1-d20-v1",
    95: "eta-ferrier-qt-partition-wrf-v4.6.1-v1",
    0: "disabled",
    # "real-clamp-product" (2.8.8) on every row whose scheme finishes
    # through woof.core.microphysics.moist_physics_finish (mp=1, 6, 9, 10,
    # 16, 18, 28, 50): the finish now clamps the theta increment at WRF's
    # REAL product float32(mp_tend_lim) * float32(dt) rather than the double
    # product rounded once, which classic Thompson (mp=8) already did.  For
    # some mp_tend_lim, dt pairs (HRRR's 0.07 K/s at dt = 12 s) the two
    # bounds differ by one float32 unit, so every clamped cell's thp and
    # h_diabatic move, and a checkpoint written before the change may not
    # continue under it.
    1: "kessler-warm-rain-v2-real-clamp-product",
    6: "wsm6-single-moment-six-class-wrf-v4.6.1-v2-real-clamp-product",
    # v5 (2.8.8): rain collecting graupel reads the one slab WRF builds
    # (thompson_racg_index; WRF v4.6.1 subscripts it at idx_bg1=5, a
    # declared divergence), and the finish clamps at the REAL product
    # mp_tend_lim*dt.  Both move mp=8 trajectories, so a v4 checkpoint is
    # refused before restore instead of resuming onto them.
    8: ("classic-thompson-wrf-v4.6.1-v5-cloud-fallout-"
        "refl10cm-ng-shadow-snow-rime-mass-number-velocity-rain-density-condensation-history-"
        "one-slab-rain-graupel-real-clamp-product"),
    3: "wsm3-simple-ice-three-class-wrf-v4.6.1-d1-default-stock-strict-d2-top-freeze-d3-default-remap-v3",
    4: "wsm5-wrf-v4.6.1-remap-psevp-v1",
    # WSM6R: the trajectory-defining facts are WRF's own module word for
    # word plus woof's two declared choices (per-step RAINNCV, WSM6's
    # radii diagnosed for RTE+RRTMGP).
    106: ("wsm6r-revised-wsm6-wrf-v4.6.1-v1-step-rainncv-"
          "wsm6-effectrad-diagnosis"),
    7: "gsfcgce-four-ice-hail-nuwrf-wrf-v4.6.1-v1",
    # Milbrandt-Yau two-moment (WRF v4.6.1 MILBRANDT2MOM).  Named at the
    # mp=8/28/50 granularity -- the trajectory-defining configuration,
    # not the scheme name --
    # because that is what makes an incompatible resume fail BEFORE
    # restore.  "six-category-2mom" is the change everything else follows
    # from: hail is a category of its own beside graupel and every one of
    # the six carries a number moment, which the WRF driver binds as
    # qnc/qnr/qni/qns/qng/qnh (module_microphysics_driver.F:1857-1862) and
    # woof transports as MY2_SPECIES (woof/core/moist.py).  The four
    # tokens after it are the switches the WRF driver HARD-CODES, so they
    # are part of this port's identity rather than of its configuration
    # (each is pinned with its line in
    # woof.config.MILBRANDT2_FIXED_IDENTITY): "ccntype2" is the
    # continental CCN spectrum (module_mp_milbrandt2mom.F:3615),
    # "meyers-contact-nucl" is prim_ice_nucl=1 (:1175), "nonspherical-snow"
    # is snow_spherical=.false. (:1174), and "full-sedimentation" is the
    # precipDiag/sedi/warmphase/autoconv/icephase/snow block all left on
    # (:3618-3623).  A build that flipped any of them would integrate a
    # different trajectory while staying finite, so it advances this tag
    # rather than resuming onto it.
    9: ("milbrandt-yau-wrf-v4.6.1-v2-six-category-2mom-ccntype2-"
        "meyers-contact-nucl-nonspherical-snow-full-sedimentation-"
        "real-clamp-product"),
    # Range-safe freezing, retained cleanup vapor and in-range number
    # preservation change subsequent tendencies even from finite inputs.
    10: ("morrison-two-moment-v5-nonnegative-sedimentation-kf-number-seeding-finite-freezing-"
         "final-vapor-in-range-number-real-clamp-product"),
    # WDM6 (WRF v4.6.1 WDM6SCHEME, Registry/Registry.EM_COMMON:3031).  Named
    # at the mp=8/28 granularity -- the trajectory-defining pieces, not the
    # scheme name.  "prognostic-nc-nr-ccn" is the change everything else
    # follows from (module_mp_wdm6.F carries qnn/qnc/qnr as scalars);
    # "gamma-mu1-rain" names the rain PSD whose intercept is diagnosed from
    # nr rather than fixed (:2251-2261); "ccn-activation" names the
    # supersaturation activation that moves mass and number out of the
    # reservoir (:1951-1969); "xland-autoconversion" names the per-column
    # maritime/continental threshold (:607-614), which makes the LAND MASK
    # part of this scheme's trajectory identity in a way no other woof
    # microphysics has been; "ccn-conc-init" records that the CCN reservoir
    # starts from the namelist constant fill (:220-227) rather than an
    # ingested aerosol field, so a future ingest must advance this tag
    # instead of silently resuming onto it.  v4 (A144): "zero-rate-rain-no-
    # evaporation" names the rain condensation cap that no longer turns a
    # zero rate (rain with no number) into evaporation, which changes the
    # vapour, ice and theta trajectory wherever such rain meets
    # subsaturated air (docs/wdm6_oracle_known_deltas.md, section 6).
    14: "wdm5-wrf461-v2-stock-rain-plm-condensation-melt-collapse",
    # v6 (development branch, landed on 2.8.9's v5 by golden/parity):
    # "wrf-words" -- the coefficients are wdm6init's own float32 words, the
    # transcendentals WOOF's own float32 exp/log/pow, every expression in
    # the Fortran's operation order and no multiply-add contraction, so the
    # column kernel lands on WRF v4.6.1's words (tools/wdm6_wrf461_oracle);
    # the last bits of every rate move, so a v5 checkpoint must not resume
    # onto it.
    16: ("wdm6-double-moment-warm-rain-wrf-v4.6.1-v6-prognostic-nc-nr-ccn-"
         "gamma-mu1-rain-ccn-activation-xland-autoconversion-ccn-conc-init-"
         "conservative-rain-interface-flux-bounded-transport-time-"
         "zero-rate-rain-no-evaporation-real-clamp-product-wrf-words"),
    18: ("nssl-two-moment-state-transport-v2-process-boundary-fail-loud-"
         "real-clamp-product"),
    # WSM7 (WRF v4.6.1 WSM7SCHEME, Registry/Registry.EM_COMMON:3034).  The
    # string is the scheme row's own (woof/data/wsm_family/schemes.json):
    # "single-moment-hail" is the seventh class, "remap-search-reaches-
    # top-interface" the declared divergence D1 that moves the trajectory
    # where WRF's PLM remap drops mass, "refl-hail-term" D2.
    24: "wsm7-single-moment-hail-wrf-v4.6.1-v1-remap-search-reaches-"
        "top-interface-refl-hail-term",
    # Independent stock WRF unit. Earlier modified-reference checkpoints
    # cannot resume across its rain, melting and number arithmetic changes.
    26: "wdm7-wrf461-v2-stock-rain-plm-condensation-hail-melt-number",

    # Thompson AEROSOL-AWARE (WRF v4.6.1 THOMPSONAERO,
    # Registry/Registry.EM_COMMON:3036).  Named at the granularity the mp=8
    # row uses -- the trajectory-defining pieces, not the scheme name --
    # because that is what makes an incompatible resume fail BEFORE restore.
    # "prognostic-nc" is the change everything else follows from
    # (module_mp_thompson.F:1795-1812 freezes nc1d at entry and :3972-4021
    # applies the single terminal ncten/nwfaten/nifaten clamp); "nwfa-nifa"
    # names the two transported aerosol tracers; "ccn-activate-table" names
    # the tnccn_act asset the activation reads
    # (:5102-5108); "demott-koop" names the ice-nucleation pair that replaces
    # classic Cooper (iceDeMott called at :2574/:2623, iceKoop at :2637;
    # the functions themselves at :5447 and :5521); "scavenging" names the
    # six aerosol wet-removal rates; "surface-emission" names the unclamped
    # nwfa2d/nifa2d injection mp_gt_driver applies AFTER the terminal clamp
    # (:1310-1327), which is a real ordering choice a reimplementation could
    # get wrong while leaving every bound intact.  "synthetic-aerosol-init"
    # records that this build's aerosol profile comes from thompson_init's
    # fill (:493-551) and not from a WIF metgrid stream: a future
    # wif_input_opt ingest is a DIFFERENT initial condition and must advance
    # this tag rather than silently resume onto it.
    # v2 (2.8.8): "wrf-arithmetic" names the generation graded word for
    # word on the 157-column WRF oracle.  Every EXP, LOG, LOG10 and ** takes
    # WOOF's own libm words (kernels/thompson_aerosol_libm.cuh), the cold
    # and warm process rates follow WRF's operation order, snow and graupel
    # accumulation, fallout and final latent heating take WRF's terms, and
    # a level at exactly 273.15 K no longer takes the wet-bulb melting
    # branch.  "rain-graupel-slab1" names the read of rain collecting
    # graupel from the one graupel-density slab WRF allocates, where 2.8.7
    # reproduced WRF's out-of-bounds index 5.  Each moves the moments,
    # theta and aerosol from finite inputs in both thompson_version
    # generations (they share these kernels), so a 2.8.7 (v1) checkpoint
    # may not continue under v2.  "real-clamp-product" names the finish
    # clamp at WRF's REAL product (above mp=1); it joins the v2 string
    # rather than opening a v3 because v2 has not shipped, and the string
    # still changes, so a checkpoint an earlier 2.8.8 build wrote under the
    # double-product clamp is refused like a 2.8.7 one.
    28: ("thompson-aerosol-aware-wrf-v4.6.1-v2-prognostic-nc-nwfa-nifa-"
         "ccn-activate-table-demott-koop-scavenging-surface-emission-"
         "synthetic-aerosol-init-wrf-arithmetic-rain-graupel-slab1-"
         "real-clamp-product"),
    27: "udm-wrf471-v1-native-kinds-frozen-tables-native-radius-flags-zero",
    29: "rcon-wrf-v4.7.1-v1-native-kinds-frozen-initialized-tables",
    38: "thompson-graupel-hail-wrf-v4.6.1-v1-prognostic-ng-qvolg-density-binned-tables",
    # P3 (WRF v4.6.1 P3_1CATEGORY, Registry.EM_COMMON:3038).  Named at the
    # granularity the mp=8/28 rows use -- the trajectory-defining
    # configuration, not the scheme name -- because that is what makes an
    # incompatible resume fail BEFORE restore.  "1cat" and "2mom-ice" name
    # the nCat=1 / log_3momentIce=.false. build (module_mp_p3.F:1043-1050);
    # "specified-nc" names log_predictNc=.false., which is the difference
    # between this row and mp=51's; "diagnosed-ssat" names
    # log_predictSsat=.false., the branch that makes th_old/qv_old the
    # cross-step carriers this restart serializes (:2325-2337); "rime-mass-
    # volume-transported" records that qir/qib advect with qi
    # (woof/core/moist.py::P3_SPECIES) -- a build that stopped
    # transporting them would integrate a DIFFERENT trajectory while
    # staying finite, which is exactly the silent resume this string is
    # here to refuse.
    # Native five-class SBU-YLin with unadvected diagnostic RIMI.
    13: "sbu-ylin-five-class-wrf-v4.7.1-native-v3-rimi-state",
    50: ("p3-one-category-wrf-v4.6.1-v2-2mom-ice-specified-nc-"
         "diagnosed-ssat-rime-mass-volume-transported-real-clamp-product"),
    # Ferrier-Aligo (WRF v4.6.1 FER_MP_HIRES, Registry.EM_COMMON:3019).
    # "total-ice-transported" records that qi advects and the f_* state
    # does not; "rimef-carried" that F_RIMEF crosses steps (and a
    # checkpoint); "etampnew-expanded-rain" names the table; "rcw-from-lwc"
    # names the one statement fixed rather than transcribed (the droplet
    # radius from RHO*QW, not its square; kernels/fer_hires.cu), which
    # changes riming trajectories wherever cloud water exceeds QAUT0.
    15: "ferrier-aligo-advect-wrf-v4.6.1-v1-qrimef-transported-stock-rcw",
    5: ("ferrier-aligo-wrf-v4.6.1-v1-total-ice-transported-rimef-carried-"
        "etampnew-expanded-rain-rcw-from-lwc"),
    # P3 with predicted droplet number (WRF v4.6.1 P3_1CATEGORY_NC,
    # Registry.EM_COMMON:3039): the mp=50 build with log_predictNc=.true.
    # (module_mp_p3.F:819-820).  "predicted-nc" is the difference from the
    # mp=50 row; "ccn-activation-2mode-derf" names the prescribed two-mode
    # lognormal activation (:268-294, :3313-3339) evaluated with P3's own
    # DERF (:6090-6204) -- a prognostic-aerosol replacement of those modes
    # is a different trajectory and must advance this tag; "nc-transported"
    # records that the droplet number advects with the rest of the package
    # (woof/core/moist.py::P3_NC_SPECIES).
    51: ("p3-one-category-wrf-v4.6.1-v2-2mom-ice-predicted-nc-"
         "ccn-activation-2mode-derf-diagnosed-ssat-rime-mass-volume-nc-"
         "transported-real-clamp-product"),
    52: "p3-two-category-wrf-v4.6.1-v1-prognostic-nc-iceice-merge-table2-v5.3",
    # Unmodified WRF 4.7.1 column operation order; the recovered modified
    # source trajectory is incompatible. CCNTY remains part of setup.
    56: "ntu-native-wrf-v4.7.1-v1-log10-native-aero-moments",
}
# Every further P3 package carries its identity in its woof.config
# P3_VARIANTS row (mp=53: "3mom-ice-predicted-nc", qnc and qzi transported).
# The mp=50 literal above stays the one written into existing checkpoints;
# the import-time check below holds the row to it.
from woof.config import P3_VARIANTS as _P3_VARIANTS
MICROPHYSICS_ALGORITHM_IDENTITIES.update({
    _mp: _row.algorithm_identity for _mp, _row in _P3_VARIANTS.items()
    if _mp != 50})
if _P3_VARIANTS[50].algorithm_identity != MICROPHYSICS_ALGORITHM_IDENTITIES[50]:
    raise RuntimeError(
        "woof.config.P3_VARIANTS[50].algorithm_identity disagrees with the "
        "mp=50 restart identity every existing P3 checkpoint carries")
from woof.microphysics_schemes import NAMED_SCHEMES as _NAMED_MP_SCHEMES
MICROPHYSICS_ALGORITHM_IDENTITIES.update({
    _s.mp_id: _s.algorithm_identity for _s in _NAMED_MP_SCHEMES.values()})

# Table-admitted column schemes carry their identity in their row
# (woof.core.mp_column_schemes.ColumnScheme.restart_identity).  mp=97:
# "three-ice" is the class set the kernel integrates; the gsfcgce_hail and
# gsfcgce_2ice switches are bound separately as scheme-scoped run fields.
from woof.core.mp_column_schemes import (  # noqa: E402
    MP_COLUMN_SCHEMES as _TABLE_ROWS)
MICROPHYSICS_ALGORITHM_IDENTITIES.update(
    {mp: row.restart_identity for mp, row in _TABLE_ROWS.items()})
MICROPHYSICS_ALGORITHM_IDENTITIES[55] = "ishmael-wrf471-3ice-habit-volume-fast-default-native-strict-v2"
SURFACE_LAYER_ALGORITHM_IDENTITIES = {
    89: "ideal-scm-surface-wrf471-native-real32-forcing-scalars-v1",
    10: "temf-surface-wrf-v4.6.1-v2-F3-cr-math-fast-default",
    0: "disabled",
    # v2 (2.8.8): the sfclay unit compiles without FMA contraction and
    # keeps denormal values instead of flushing them to zero
    # (woof.core.kernels.module_options) on WOOF's own float32
    # log/exp/pow/atan; with isfflx = 0 it keeps the
    # incoming CHS, CHS2 and CQS2 as WRF does instead of recomputing them;
    # and an overflowing scalar-roughness exponential stops before FH goes
    # infinite.  Each changes the exchange coefficients and fluxes the next
    # step reads, so a 2.8.7 (v1) checkpoint may not continue under v2.
    1: "revised-mm5-surface-layer-v2-no-fma-denormals-kept-flux-off-"
       "keeps-chs-finite-scalar-roughness",
    # The Eta similarity surface layer.  The identity binds the WRF version
    # whose byte-frozen module_sf_myjsfc.F the port transcribes AND the
    # similarity tables it interpolates: MYJSFCINIT builds PSIM/PSIH by
    # accumulating ZETA in float32, so a different table construction is a
    # different scheme even at the same WRF version, and a checkpoint may
    # not resume across one.
    # v2 (2.8.8): that is what happened.  The tables are now built with
    # WOOF's own float32 log/atan/exp (woof/core/myjsfc_tables.py), which
    # moved 2,277 of 10,001 PSIM entries by up to 16 ULP and 1,382 PSIH
    # entries, and the column compiles without FMA contraction or a flush
    # of denormals to zero, on WOOF's libm words.  A 2.8.7 (v1) checkpoint
    # may not continue under v2.
    2: "eta-similarity-surface-layer-wrf-v4.6.1-v2-myjsfcinit-tables-"
       "woof-libm-no-fma-denormals-kept",
    # v2 (2.8.8): no FMA contraction, denormals kept rather than flushed to
    # zero, WOOF's libm words (kernels/mynn_libm.cuh), and the first-step
    # seed through the same unit.  It changes UST, MOL, QSFC and the fluxes
    # the next step reads, so a 2.8.7 (v1) checkpoint may not continue
    # under v2.
    5: "mynn-surface-layer-wrf-v4.6.1-v2-no-fma-denormals-kept",
    # v2 (2.8.8): the classic MM5 column (kernels/sfclay_classic.cuh) is in
    # the sfclay unit and takes its arithmetic (no FMA, denormals kept), so
    # a 2.8.7 (v1) checkpoint may not continue under v2.
    91: "classic-mm5-surface-layer-v2-no-fma-denormals-kept",
    # The GFS layer: WRF v4.6.1 module_sf_gfs.F in binary64 over
    # GFUNCPHYS's FPVS table, with the FRACTIONAL_SEAICE wrapper's two
    # declared repairs (the open-water call gets XLAND_SEA; the open-water
    # TSK_SEA is published).  A build without either computes a different
    # sea-ice cell, so a checkpoint may not resume across one.
    3: "gfs-surface-layer-wrf-v4.6.1-v2-fast-default-xland-sea-tsk-sea",
    4: "qnse-surface-layer-wrf-v4.7.1-v3-fast-intrinsics-frozen-init",

    # The Pleim-Xiu surface layer.  It carries UST, ZNT (Charnock over
    # water), QSFC and the fluxes across steps and averages the new u* with
    # the old one over land, so a resume onto a different transcription is
    # a different trajectory; the identity binds the WRF version of the
    # byte-frozen module_sf_pxsfclay.F and its sea-ice wrapper.
    7: "pleim-xiu-surface-layer-wrf-v4.6.1-v1",
}
#: WRF v4.7.1 lsm_mosaic; bound only for the enabled tile path.
NOAH_MOSAIC_ALGORITHM_IDENTITY = "noah-mosaic-wrf-v4.7.1-v1"

LAND_SURFACE_ALGORITHM_IDENTITIES = {
    0: "disabled",
    # WRF v4.6.1 module_sf_slab.F SLAB1D, word for word.
    1: "wrf-v4.7.1-slab-5layer-snow-fast-v3",
    2: "noah-lsm-v2-post-sflx-chs2-source-water-lake-skin",
    # v2 (2.8.5): a LAKEMASK column is bypassed only when the lake model is
    # selected (module_sf_ruclsm.F:824, lakemodel==1 .and. lakemask==1).
    # v1 bypassed every LAKEMASK column, so with sf_lake_physics = 0 a lake
    # column was never advanced; it now runs RUC's water branch.  That
    # changes the surface state and fluxes of every RUC run with lake
    # cells, so a v1 checkpoint may not continue under v2.
    # v3 (2.8.5): SOILPROP's soil-water diffusivity and conductivity take
    # the WRF v4.0-4.5 normalisation by default (ruc_soilprop = "wrf_45",
    # woof/core/ruc_tier.py); v2 ran the v4.6.1 form over total porosity,
    # which moves 2.5 to 8 times more water up into a dry top soil level.
    # That changes the soil water and surface fluxes of every RUC run, so
    # a v2 checkpoint may not continue under v3.
    # v4: the post-SFCTMP irrigation takes the WRF v4.0-4.5 crop-fraction-
    # scaled floor by default (ruc_irrigation = "wrf_45",
    # woof/core/ruc_mosaic.py), whatever mosaic_lu says; v3 ran the v4.6.1
    # relaxation under mosaic_lu = 1 and nothing without it.  That changes
    # the root-zone soil water of every RUC run with cropland, so a v3
    # checkpoint may not continue under v4.
    # v5: the snow scheme takes the WRF v4.0-4.5 form the operational
    # RAP/HRRR branch carries by default (ruc_snow = "wrf_45",
    # woof/core/ruc_tier.py); v4 ran the v4.6.1 snow conductivity, cover,
    # melt and albedo.  That changes the snow pack and the surface over it
    # in every RUC run with snow, so a v4 checkpoint may not continue under
    # v5.
    # v6 restores generic irrigation and snow to wrf_461. Unpublished v4/v5
    # headers omitted wrf_45, so interpreting those missing selectors as the
    # restored defaults would change their continuing trajectory. The new
    # identity refuses those checkpoints before restoring live forecast arrays.
    # v7 (2.8.8): every EXP, LOG, LOG10, TANH and REAL**REAL of the column
    # takes WOOF's float32 words and the soil-resistance COS glibc's cosf
    # (kernels/ruc.cu), where v6 rounded float64 stand-ins once; the
    # sea-ice blend and skin reset run only under fractional_seaice = 1, as
    # in WRF (kernels/ruc_fused_driver.cuh), where v6 ran them on every ice
    # cell; and SFCEVP accumulates QFX*dt once, not WRF's duplicated twice.
    # The first two change soil water, soil ice, fluxes and the ice-cell
    # surface every step, so a 2.8.7 (v6) checkpoint may not continue
    # under v7.  (The 2.8.8 RUC initialization carry and the real.exe soil
    # moisture floor act at cold start only and need no identity.)
    3: "ruc-lsm-wrf-v4.6.1-v7-woof-libm-fractional-seaice-gate",
    4: "noahmp-lsm-wrf-v4.6.1-v2-wrf470-night-snow-aging",
    7: "pleim-xiu-land-wrf-v4.6.1-v1",
    8: "ssib-wrf-v4.6.1-v1-seaice-unit-repair",
}
NOAHMP_SNOW_AGE_ALGORITHM_IDENTITIES = {
    "wrf_461": "noahmp-lsm-wrf-v4.6.1-v1",
    "wrf_470": LAND_SURFACE_ALGORITHM_IDENTITIES[4],
}

PBL_ALGORITHM_IDENTITIES = {
    10: "temf-pbl-wrf-v4.6.1-v2-native-arithmetic-fast-default",
    0: "disabled",
    # v2 (2.8.8): YSU compiles without FMA contraction on WOOF's float32
    # pow/exp (gfk_pow/gfk_exp, kernels/ysu.cu), reads theta as WRF's
    # driver hands it (th*pi/pi), makes the bulk Richardson sign test in
    # double, and takes WRF's surface-drag form on every topo_wind = 0
    # column.  Each moves the exchange coefficients and tendencies, so a
    # 2.8.7 (v1) checkpoint may not continue under v2.
    1: "ysu-wrf-v4.6.1-v2-no-fma-woof-libm-driver-theta-double-ri-"
       "wrf-surface-drag",
    # The exact driver changes the former rounded-length generation's
    # diffusivities and tendencies in both stock and fork forms. A checkpoint
    # from that generation cannot promise the same continuation here.
    # v3 was opened in 2.8.8 and has never shipped, so it also names the
    # rest of that release's MYNN generation: the gsd_41 fork's driver
    # arithmetic and length-1 formulation, and the moisture variance that
    # now keeps denormals rather than flushing them (kernels/mynn_pbl.cu).
    # Every 2.8.7 (v2) checkpoint is refused by this one bump; a v4 would
    # refuse nothing more that a release wrote.
    5: "mynn-edmf-pbl-wrf-v4.6.1-v3-exact-driver",
    # Adding a scheme means adding its row, not relaxing the check.  The
    # identity binds the WRF version whose byte-frozen module_bl_shinhong.F
    # the certified CPU authority transcribes (max ULP 0, both arms); a
    # future re-transcription against a different WRF advances the suffix
    # rather than silently resuming onto this one.
    # MYJ carries genuinely prognostic state -- TKE_MYJ is read as 2*TKE at
    # the top of every call and rewritten at the bottom, and the Eta surface
    # layer's PBLH scan reads it too -- so a resume that dropped it would
    # continue a different boundary layer while staying finite.  The
    # identity binds the WRF version the port transcribes; a
    # re-transcription against another WRF advances the suffix rather than
    # silently resuming onto this one.
    2: "myj-pbl-wrf-v4.6.1-v1-mellor-yamada-2.5-janjic",
    # QNSE carries MYJ's prognostic TKE column and, with mfshconv = 1, the
    # mass flux's RC_MF.  The identity binds the WRF version the port
    # transcribes (module_bl_qnsepbl.F and module_bl_mfshconvpbl.F at
    # v4.6.1); mfshconv itself is a RunConfig value the restart header
    # compares, so it is not repeated here.
    4: "qnse-pbl-wrf-v4.7.1-v2-fast-default-edmf-mass-flux",
    # The UW moist-turbulence PBL carries prognostic state across steps --
    # the interface diffusivities it reads back as kvm_in/kvh_in and the
    # residual surface stress -- and its saturation table is WRF's own
    # words (woof/core/uwpbl_constants.py).  The identity binds the WRF
    # version the port transcribes, v4.7.1.
    # The GFS nonlocal PBL carries no state of its own between calls, but
    # its rewrite of UST/WSPD and its held tendencies are its arithmetic;
    # the identity binds the WRF version the port transcribes.
    3: "gfs-pbl-wrf-v4.6.1-v2-fast-default",

    # BouLac carries prognostic TKE_PBL (INOUT every call, floored at
    # temin) and, under BEP, writes DLG_BEP back; a resume that dropped the
    # TKE would continue a different boundary layer.  The identity binds
    # the WRF version whose module_bl_boulac.F object the port is graded
    # against word for word.
    # v3 retains native WU_TUR/WV_TUR/WT_TUR/WQ_TUR restart diagnostics.
    8: "boulac-pbl-wrf-v4.7.1-v3",
    9: "uw-moist-turbulence-pbl-wrf-v4.7.1-v1",
    # v2 (2.8.8): Shin-Hong compiles without FMA contraction on WOOF's
    # float32 pow/exp (kernels/shinhong.cu), which moved its exchange
    # coefficient by up to 8 ULP, so a 2.8.7 (v1) checkpoint may not
    # continue under v2.
    11: "shinhong-pbl-wrf-v4.6.1-v2-no-fma-woof-libm",
    # GBM carries prognostic TKE_PBL across steps (read at the top of every
    # call, rewritten with the first GBM2D pass's TKE).  The identity binds
    # the WRF version whose module_bl_gbmpbl.F the port transcribes.
    # v4 uses warp-register recurrences, shared live-range reuse and removes
    # calculations whose results WRF does not publish.
    # Compiler FMA placement can change, so older fast checkpoints must not
    # silently continue under this schedule. Strict replay bytes are retained.
    12: "gbm-moist-tke-pbl-wrf-v4.6.1-v4-live-shared-fast",
    # EEPS carries prognostic TKE and dissipation across steps (the
    # transported PEK_ADV/PEP_ADV and the PBL's PEK_PBL/PEP_PBL).  The
    # identity binds the WRF version the port transcribes and its one
    # declared divergence, RADSUM initialised to zero.
    16: "eeps-pbl-wrf-v4.6.1-v1-radsum-zero",
    # MRF carries no prognostic state of its own (UST, PBLH, KPBL and HOL
    # are fields every restart already carries); the identity binds the
    # WRF version whose module_bl_mrf.F the kernel is graded against.
    99: "mrf-pbl-wrf-v4.6.1-v1",

    # K-epsilon carries prognostic TKE, dissipation and TPE across calls.
    # The identity binds the WRF version the port transcribes and the
    # declared fixes it carries (tools/keps_wrf461_oracle/
    # declared-divergences.patch); a change to either advances the suffix.
    17: "keps-pbl-wrf-v4.7.1-v2-defined-wv-native-undefined-initialized",

    # ACM2 carries PBLH across steps (it reads the previous height for the
    # convective velocity scale before writing the new one) and EXCH_H/
    # EXCH_M's top level; the identity binds the WRF version whose
    # byte-pinned module_bl_acm.F the port transcribes.
    7: "acm2-pbl-wrf-v4.6.1-v1",
    # SASE carries no WRF version in its identity because there is no WRF
    # scheme it transcribes.  What the identity DOES have to bind is the
    # closure's constant registry: sase_config_id() is a SHA-256 over
    # every registered coefficient, so a checkpoint written under one set
    # of constants cannot be resumed under another -- which is the whole
    # job of this table.
    900: "sase-experimental-v1",
    901: "ccpp-satmedmf-tke-edmf-1d9b0765-v1",
}
#: sf_surface_physics -> the ``PhysicsDriver`` attribute holding that
#: scheme's packed parameter bundle, and the packaged-asset roles whose
#: bytes it was built from.  A land-surface scheme with no row here cannot
#: be restart-identified: a checkpoint that omitted its parameters would
#: resume against a silently different table set.  Adding a scheme means
#: adding its row, not relaxing the check.
LAND_SURFACE_PARAMETER_SOURCES = {
    # The slab's bundle is its geometry, ifsnow and constants
    # (woof.core.slab.SlabParameters); the one table value it reads, THC,
    # comes from LANDUSE.TBL at initialization and rides the fields.
    1: ("slab_params", ("noah_landuse",)),
    7: ("pxlsm_params", ("pxlsm_tables",)),
    2: ("noah_params", ("noah_vegparm", "noah_soilparm",
                        "noah_genparm", "noah_landuse")),
    # RUC reads the RUC SECTIONS of the same three files Noah reads --
    # VEGPARM's MODI-RUC/USGS-RUC blocks and SOILPARM's STAS-RUC block -- so
    # the asset roles are shared while the bundle object is not.  LANDUSE.TBL
    # is absent: woof.core.ruc never opens it, because RUC's roughness,
    # albedo and emissivity come from its own VEGPARM rows.
    3: ("ruc_params", ("noah_vegparm", "noah_soilparm", "noah_genparm")),
    4: ("noahmp_params", ("noahmp_mptable", "noahmp_soilparm",
                          "noahmp_genparm")),
    8: ("column_surface_params", ("ssib_vegetation", "noah_landuse")),
}
LONGWAVE_ALGORITHM_IDENTITIES = {
    7: "wrf461-flg-rad-all-v1",
    31: "wrf-v4.6.1-relaxation-31-v1",
    5: "wrf461-goddard-defined-visible-depth-v1",
    99: "wrf-v4.6.1-gfdleta-lw-v1",
    0: "disabled",
    1: "wrf-v4.6.1-rrtm-longwave-v1",
    3: "wrf-v4.6.1-cam3-v1",
    4: "rte-rrtmgp-v1",
    90: "analytic-clear-sky-v1",
}
SHORTWAVE_ALGORITHM_IDENTITIES = {
    7: "wrf461-flg-rad-all-v1",
    5: "wrf461-goddard-defined-visible-depth-v1",
    99: "wrf-v4.6.1-gfdleta-sw-v1",
    2: "wrf-v4.7.1-gsfc-shortwave-v2",
    0: "disabled",
    1: "wrf-v4.6.1-dudhia-shortwave-v1",
    3: "wrf-v4.6.1-cam3-v1",
    4: "rte-rrtmgp-v1",
    90: "analytic-clear-sky-v1",
}
#: Above-model optical-column policy is separate from the gas/RTE algorithm
#: identity because changing the cap changes model-top fluxes while retaining
#: the same packaged coefficient tables and solver.
LONGWAVE_ABOVE_ATMOSPHERE_POLICIES = {
    7: "model-column-top-interface-no-added-cap",
    31: "not-applicable-idealized-column-forcing",
    5: "model-column-top-interface-zero-hpa",
    99: "wrf-v4.6.1-gfdl-model-column",
    0: "not-applicable-radiation-disabled",
    1: "wrf-v4.6.1-rrtm-deltap-4mb-buffer-layers",
    3: "wrf-v4.6.1-cam-model-column",
    4: "wrf-v4.6.1-lw-4hpa-sw-half-ptop-clear-cap-to-rte-floor-v1",
    90: "not-applicable-analytic-surface-flux-proxy",
}
SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES = {
    7: "model-column-top-interface-no-added-cap",
    5: "model-column-top-interface-zero-hpa",
    99: "wrf-v4.6.1-gfdl-model-column",
    2: "wrf-v4.7.1-gsfc-clear-top-layer-to-zero-pressure",
    0: "not-applicable-radiation-disabled",
    1: "not-applicable-dudhia-model-column-only",
    3: "wrf-v4.6.1-cam-model-column",
    4: "wrf-v4.6.1-lw-4hpa-sw-half-ptop-clear-cap-to-rte-floor-v1",
    90: "not-applicable-analytic-surface-flux-proxy",
}
# Backward-compatible names for the historical coupled selections.  New
# identity code records each component independently; these aliases keep
# readers/tests that inspect a 4/4 or 90/90 setup source-compatible.
RADIATION_ALGORITHM_IDENTITIES = {
    key: LONGWAVE_ALGORITHM_IDENTITIES[key] for key in (0, 4, 90)}
RADIATION_ABOVE_ATMOSPHERE_POLICIES = {
    key: LONGWAVE_ABOVE_ATMOSPHERE_POLICIES[key] for key in (0, 4, 90)}
#: The urban canopy models (``sf_urban_physics``).  Bound into the
#: checkpoint header only when one runs, so every existing header is
#: unchanged; the string names the WRF version whose routine each CUDA
#: column is graded against.
URBAN_ALGORITHM_IDENTITIES = {
    0: "disabled",
    1: "urban-slucm-wrf-v4.7.1-water-balance-v4.8.0-v1",
    2: "urban-bep-wrf-v4.7.1-v1",
    3: "urban-bep-bem-wrf-v4.7.1-v1",
}
URBAN_UCM_GENERATION_IDENTITIES = {
    "wrf_471": URBAN_ALGORITHM_IDENTITIES[1],
    "wrf_480": "urban-slucm-wrf-v4.8.0-defined-entry-v1",
}


def urban_algorithm_identity(cfg):
    option = int(cfg.sf_urban_physics)
    if option == 1:
        from woof.config import URBAN_WATER_BALANCE_GENERATIONS
        water = getattr(cfg, "urban_water_balance", "wrf_480")
        if getattr(cfg, "urban_ucm_generation", "wrf_471") == "wrf_471":
            return URBAN_WATER_BALANCE_GENERATIONS[water]["algorithm"]
        if water == "wrf_471":
            return URBAN_UCM_GENERATION_IDENTITIES["wrf_480"] + "-legacy-rain-clamps"
    return (URBAN_UCM_GENERATION_IDENTITIES[getattr(cfg, "urban_ucm_generation", "wrf_471")]
            if option == 1 else URBAN_ALGORITHM_IDENTITIES[option])

from woof.wrf_exact import ENABLED as _WRF_EXACT  # noqa: E402
from woof.numerics import is_strict as _strict_numerics  # noqa: E402
from woof.wrf_exact import referee_correctly_rounded_trig as _new_native_trig  # noqa: E402
from woof.numerics import mode as _numerics_mode  # noqa: E402

# The registry declares all generations, independently of the builder's
# environment. Checkpoints still bind the generation actually selected.
KSAS_ALGORITHM_IDENTITIES = {
    "fast": "kiaps-sas-wrf471-fast-v1",
    "strict-glibc239": "kiaps-sas-wrf471-strict-glibc239-v1",
    "strict-glibc241": "kiaps-sas-wrf471-strict-glibc241-v1",
}

SAS_ALGORITHM_IDENTITIES = {
    "fast": "gfs-sas-2015-arw-shallow-cuda-fast-v2",
    "strict": "gfs-sas-2015-arw-shallow-wrf461-v1",
}

MSKF_ALGORITHM_IDENTITIES = {
    "strict": "multiscale-kf-wrf461-defined-aerosol-reads-v2-fast-default",
    "fast": "multiscale-kf-wrf461-v3-cuda-float32-fast",
}

CUMULUS_ALGORITHM_IDENTITIES = {
    95: ("old-gfs-sas-wrf461-cu_osas-strict-ql-backout-v2" if _WRF_EXACT else
         "old-gfs-sas-wrf461-cu_osas-counter-rng-ql-backout-v2"),

    7: "camzm-wrf471-stock-r8-v1",
    11: MSKF_ALGORITHM_IDENTITIES[
        "strict" if _WRF_EXACT or _numerics_mode() == "strict" else "fast"],
    0: "disabled",
    # The WRF v4.6.1 KF_eta_PARA port (2026-10-07), graded at 0 ULP on
    # every KF_eta_CPS output by tools/kf_wrf461_oracle.  It replaced a
    # re-derivation (v3) whose layer masses, trigger, closure arithmetic
    # and shallow NCA all differed, so a v3 checkpoint must not resume
    # under it: the held tendencies and W0AVG it carries were made by a
    # different algorithm.
    1: "kain-fritsch-wrf461-kfeta-port-v1",
    10: "kfcup-wrf461-corrected-pdf-native-rain-clock-v1",
    # The corrected-k22 identity IS the shipped algorithm (owner ruling);
    # a restart written under it must never resume under a WRF-faithful
    # build, which would be a different identity string.
    # v2 also binds the bounded shallow closure and lower-face W seam.
    # A v1 checkpoint must not silently resume with these changed defaults.
    3: "grell-freitas-wrf461-gfdrv-corrected-k22-v2",
    # New Tiedtke, cu_ntiedtke/cumastrn as shipped in WRF v4.6.1.
    #
    # BUMP THE -v1 IF THE DRIVER SEAM MOVES, not only if the kernels do.
    # This port computes PRATEC at max_ulp == 0 and deliberately does not
    # hand it to the driver (docs/ntiedtke/PORT-RECORD.md section 38), so every
    # checkpoint written under this identity carries cu_pratec == 0 by
    # construction.  An implementation that delivered it would give the
    # same slot a different meaning, and that is exactly the kind of
    # cross-resume this string exists to refuse.
    6: "tiedtke-wrf461-tiecnv-v1",
    14: KSAS_ALGORITHM_IDENTITIES[
        ("strict-glibc241" if _new_native_trig() else "strict-glibc239")
        if _WRF_EXACT or _numerics_mode() == "strict" else "fast"],
    # v2: default arithmetic bounds the explicit rates so no layer loses more water than it
    # holds over the step (woof.core.ntiedtke.bound_donor_depletion); strict keeps v1.
    16: ("new-tiedtke-wrf461-cumastrn-v1" if _WRF_EXACT
         else "new-tiedtke-wrf461-cumastrn-donor-bound-v2"),
    # Grell 3D, G3DRV/CUP_enss_3d/conv_grell_spread3d as shipped in WRF
    # v4.6.1. Shallow diagnostics are zero when ishallow=0.
    # Closure member 16 keeps WRF's original copy order; diagnostics
    # and ensemble arrays are carried by the canonical fields inventory.
    5: "grell-3d-wrf461-g3drv-spread3d-v1",
    # New SAS + NSCV, cu_nsas as shipped in WRF v4.6.1, complete row limits
    # and persistent per-step rain. Equal to the scheme's
    # table row (woof.core.column_cumulus), held there by
    # tests/test_nsas_admission.py.
    94: SAS_ALGORITHM_IDENTITIES["strict" if _strict_numerics() else "fast"],
    96: "new-sas-nscv-wrf461-cu_nsas-v1",
    2: "betts-miller-janjic-wrf461-bmjdrv-v1",
    93: "grell-devenyi-wrf461-grelldrv-stock-v1",

    # The previous Kain-Fritsch, KFCPS/KFPARA as shipped in WRF v4.6.1
    # (woof/core/kernels/kf_old.cu, graded at 0 ULP by
    # tools/kf_old_wrf461_oracle).  Same adapter class as 1, different
    # scheme, so a cu=1 checkpoint must never resume as 99 or back.
    99: "kain-fritsch-cu99-wrf461-kfcps-v1",
}

#: The implementation of the dycore's once-per-step mixing: WRF's diff_opt
#: = 2 package (km_opt 1 to 4, woof/core/dycore.py) and the sixth-order
#: filter (kernels/diff6.cu).  Configuration proves which closure runs and
#: with which constants (``configuration_sha256``); it cannot prove how the
#: operators are transcribed, and 2.8.8 changed exactly that under the same
#: selectors: the default build now compiles them with WRF's operation
#: order and IEEE arithmetic (no FMA, no flush to zero, IEEE division and
#: square root, woof.core.kernels.DIFFUSION_OPTIONS), km_opt = 1 runs WRF's
#: isotropic_km through the shared package, a PBL-off open boundary keeps
#: its surface-flux ring, and the sixth-order filter accumulates in place
#: within WRF's loop bounds.  Millions of tendency words moved on WRF's
#: fixtures, so a checkpoint written before 2.8.8 under any of these would
#: continue a trajectory that is not the one it claims.
#:
#: Recorded only for a domain that runs one of these operators
#: (:func:`dycore_mixing_identity`), the way the urban row is recorded only
#: when an urban model runs: a run without mixing (the default km_opt = 1
#: with khdif = kvdif = 0 under a PBL scheme, and no sixth order) writes
#: exactly the header it wrote before 2.8.8 and resumes across the
#: release, because nothing it integrates moved.  A further change to the
#: mixing operators advances this tag.
DYCORE_MIXING_ALGORITHM_IDENTITY = (
    "wrf-v4.6.1-diff-opt2-mixing-and-sixth-order-v1-wrf-order-ieee")


def dycore_mixing_identity(cfg) -> str | None:
    """The mixing identity this domain's checkpoint records, or ``None``.

    ``None`` exactly when the dycore builds no mixing tendency at all:
    :func:`woof.config.wrf_mixing_package_active` is false and the
    sixth-order filter is off -- the two predicates
    ``woof.core.dycore.prepare_fixed_tendencies`` reads as
    ``include_smag`` and ``include_diff6``.
    """
    from woof.config import wrf_mixing_package_active

    if wrf_mixing_package_active(cfg) or int(
            getattr(cfg, "diff_6th_opt", 0)) > 0:
        return DYCORE_MIXING_ALGORITHM_IDENTITY
    return None


#: The implementation of the implicit upper damping layer (damp_opt = 3,
#: inside WRF's advance_w).  2.8.8 forms its strength ``dampmag =
#: dts*dampcoef`` as WRF's one float32 product in the default build
#: (woof.core.acoustic.damp_magnitude); 2.8.7 formed it as a double product
#: rounded once, which is a different word for sound steps such as 4.5 s
#: with dampcoef 0.2, and then every damped w and geopotential word moved
#: by up to 2,401 ULP against WRF.  The selectors did not change, so a
#: 2.8.7 checkpoint of a damp_opt = 3 run would continue a trajectory that
#: is not the one it claims.
#:
#: Recorded only for a domain that runs the damper
#: (:func:`upper_damping_identity`), the way the mixing row is: a run with
#: damp_opt = 0 (the default) writes the header it wrote before 2.8.8 and
#: resumes across the release.  A further change to the damper advances
#: this tag.
UPPER_DAMPING_ALGORITHM_IDENTITY = (
    "wrf-v4.6.1-damp-opt3-advance-w-float32-dampmag-v1")


def upper_damping_identity(cfg) -> str | None:
    """The upper-damping identity this domain's checkpoint records, or
    ``None`` when the implicit damper is off (damp_opt != 3), the one
    predicate :func:`woof.core.acoustic.damp_magnitude` reads."""
    if int(getattr(cfg, "damp_opt", 0)) == 3:
        return UPPER_DAMPING_ALGORITHM_IDENTITY
    return None


#: PLAN REVIEW: every identity table the checkpoint writer resolves from
#: CONFIGURATION alone, as (table name, config attribute, label) rows.
#: These are the lookups :func:`physics_setup_identity` and the helpers it
#: calls perform on the way to a checkpoint whose only inputs are config
#: values -- which is exactly why the answer can be demanded before the
#: run starts instead of an hour into it.  Tables are NAMED rather than
#: captured so a row resolves through the module (monkeypatchable, and
#: comparable against the names the writer's own source reads:
#: tests/test_checkpoint_scheme_identity_gate.py derives the writer's set
#: from source rather than restating this list).  Radiation is absent and
#: handled by :func:`_radiation_identity_table_rows`: its selector is a
#: resolved (lw, sw) PAIR whose branch decides which tables are read at
#: all.
_CONFIGURED_IDENTITY_TABLES = (
    ("MICROPHYSICS_ALGORITHM_IDENTITIES", "mp_physics", "microphysics"),
    ("SURFACE_LAYER_ALGORITHM_IDENTITIES", "sf_sfclay_physics",
     "surface layer"),
    ("LAND_SURFACE_ALGORITHM_IDENTITIES", "sf_surface_physics",
     "land surface"),
    # A land-surface scheme is named TWICE on the way to a checkpoint:
    # once for its algorithm, and once for the packed parameter bundle
    # _land_surface_parameters_identity resolves through
    # LAND_SURFACE_PARAMETER_SOURCES -- which raises the same
    # "has no parameter-bundle row" RestartManifestError, from the same
    # point in the run, as the microphysics lookup that lost a forecast.
    # A scheme added to LAND_SURFACE_ALGORITHM_IDENTITIES and not to
    # LAND_SURFACE_PARAMETER_SOURCES reproduces that defect exactly, so
    # both tables are asked about here rather than only the one the first
    # instance of the defect happened to be in.
    ("LAND_SURFACE_PARAMETER_SOURCES", "sf_surface_physics",
     "land-surface parameter bundle"),
    ("PBL_ALGORITHM_IDENTITIES", "bl_pbl_physics", "PBL"),
    ("CUMULUS_ALGORITHM_IDENTITIES", "cu_physics", "cumulus"),
    # woof/io/restart.py binds the urban canopy model a checkpoint was
    # written under; the table carries 0 ("disabled") too, so asking at
    # every value refuses nothing the writer would write.
    ("URBAN_ALGORITHM_IDENTITIES", "sf_urban_physics", "urban canopy"),
    # Under Noah-MP the writer names the land surface by its snow-aging
    # generation (woof/io/restart.py); both generations carry a row, so
    # asking at every value refuses nothing the writer would write.
    ("NOAHMP_SNOW_AGE_ALGORITHM_IDENTITIES", "noahmp_snow_age",
     "Noah-MP snow aging"),
)

#: Rows of :data:`_CONFIGURED_IDENTITY_TABLES` the writer reaches only for
#: SOME values of their selector, as (table name, attribute) -> predicate
#: on the resolved id.  Asking outside the predicate would refuse a run
#: the writer would have written without complaint, which is the one way
#: a plan-review gate can be worse than no gate.
_CONDITIONAL_IDENTITY_TABLES = {
    # No land surface runs and no parameter bundle is resolved:
    # physics_setup_identity's ``sf_surface_physics != 0`` arm.
    ("LAND_SURFACE_PARAMETER_SOURCES", "sf_surface_physics"):
        lambda key: key != 0,
}


def _identity_table(name: str) -> dict:
    """The named module-level identity table, resolved at CALL time.

    Late resolution is what lets the gate see a table a test replaced and
    what lets the rows above be plain names.
    """
    return globals()[name]


def _radiation_identity_table_rows(cfg):
    """The radiation identity tables THIS configuration is named through.

    Mirrors the branch in :func:`_radiation_setup_identity` exactly,
    because which tables a radiation setup is looked up in is decided by
    the setup: a coupled selection resolves through the RADIATION_* pair
    (algorithm AND above-atmosphere policy), a mixed one through the
    LONGWAVE_*/SHORTWAVE_* pairs, and legacy RRTMG through module
    constants -- no table, so nothing to ask about.  Returned as
    (table name, config attribute, scheme id, label) rows.
    """
    from woof.config import radiation_scheme_ids
    from woof.physics_compat import (RRTMG_VARIANT_LEGACY,
                                      rrtmg_variant)

    lw_id, sw_id = radiation_scheme_ids(cfg)
    if ((lw_id, sw_id) == (4, 4)
            and rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY):
        return ()
    if (lw_id == sw_id
            and lw_id in _identity_table("RADIATION_ALGORITHM_IDENTITIES")):
        return (("RADIATION_ALGORITHM_IDENTITIES", "ra_lw_physics", lw_id,
                 "radiation"),
                ("RADIATION_ABOVE_ATMOSPHERE_POLICIES", "ra_lw_physics",
                 lw_id, "radiation above-atmosphere policy"))
    return (("LONGWAVE_ALGORITHM_IDENTITIES", "ra_lw_physics", lw_id,
             "longwave radiation"),
            ("LONGWAVE_ABOVE_ATMOSPHERE_POLICIES", "ra_lw_physics", lw_id,
             "longwave above-atmosphere policy"),
            ("SHORTWAVE_ALGORITHM_IDENTITIES", "ra_sw_physics", sw_id,
             "shortwave radiation"),
            ("SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES", "ra_sw_physics", sw_id,
             "shortwave above-atmosphere policy"))


def unidentifiable_checkpoint_schemes(cfg) -> list[str]:
    """Which of this configuration's schemes the checkpoint writer cannot name.

    One entry per unnameable selection, in table order, each spelling the
    label, the id and the table that has no row for it.  Empty is the
    answer for every configuration whose checkpoints can be written.
    """
    gaps: list[str] = []

    def _report(message: str) -> None:
        if message not in gaps:
            gaps.append(message)

    for name, attribute, label in _CONFIGURED_IDENTITY_TABLES:
        value = getattr(cfg, attribute, None)
        table = _identity_table(name)
        if isinstance(value, str) and table and all(
                isinstance(row, str) for row in table):
            # A generation-keyed table (NOAHMP_SNOW_AGE_ALGORITHM_IDENTITIES)
            # is looked up by the generation name the writer resolves.
            if value not in table:
                _report(f"{label} generation {value!r} ({attribute}) has no "
                        f"row in the checkpoint identity table")
            continue
        try:
            key = int(value)
        except (TypeError, ValueError):
            # One attribute can feed two tables; say it once.
            _report(f"{label} {attribute}={value!r} is not a scheme id")
            continue
        predicate = _CONDITIONAL_IDENTITY_TABLES.get((name, attribute))
        if predicate is not None and not predicate(key):
            continue
        if key not in _identity_table(name):
            _report(
                f"{label} scheme {key} ({attribute}) has no row in the "
                f"checkpoint identity table")
    for name, attribute, key, label in _radiation_identity_table_rows(cfg):
        if key not in _identity_table(name):
            _report(
                f"{label} scheme {key} ({attribute}) has no row in the "
                f"checkpoint identity table")
    return gaps


def require_identifiable_checkpoint_schemes(cfg) -> None:
    """Refuse, AT PLAN REVIEW, a run whose checkpoints could not be written.

    THE CONCRETE BREAKAGE (measured 2026-09-10 07:39Z, single-domain ERA5
    forecast, mp_physics=9): every scheme a checkpoint names is looked up
    in one of the tables above, and a scheme with no row raises
    :class:`RestartManifestError` from ``_scheme_algorithm``.  That lookup
    happens inside ``write_restart``, which happens at the FIRST restart
    interval -- so a configuration the loader accepts and the forecast
    integrates dies at its first checkpoint, taking the 59 minutes of
    forecast it had already produced with it.  Nothing before that point
    can tell the operator, because nothing before that point asks the
    question.  This asks it while the answer is still free.

    Called from :func:`woof.config.validate_run_config`, which every door
    runs before any device work, so it needs no flag and no opt-in: a run
    that would not have been checkpointable is refused where a
    configuration error belongs, and a run that is checkpointable never
    sees it.

    ``ValueError``, deliberately, and not the ``RestartManifestError``
    the writer's own lookup raises: this is a CONFIGURATION refusal, and
    every door already renders a ValueError from the invariant battery as
    a user-facing refusal with exit code 2 (woof/cli.py) rather than a
    traceback.  The manifest error stays exactly where it was, as the
    backstop for a state that reaches a write unnameable.
    """
    gaps = unidentifiable_checkpoint_schemes(cfg)
    if not gaps:
        return
    detail = "; ".join(gaps)
    raise ValueError(
        f"this configuration cannot be checkpointed: {detail}. A restart "
        "file records the identity of every scheme that integrated it, so "
        "an unnamed scheme fails the write at the first restart interval "
        "and discards the forecast produced up to it. Add the identity "
        "row for the scheme in woof/checkpoint_identity.py -- do not "
        "relax the check, which is what makes an incompatible resume "
        "fail before it "
        "restores.")


# ---------------------------------------------------------------------------
# AGREEMENT WITH THE REGISTRY, AT IMPORT.  The tables above are the source
# tools/build_registry.py copies each option's restart_algorithm_identity
# row from; this holds the tables to the registry's implemented inventory so
# a scheme that lands in the registry without its identity row fails THIS
# import -- the suite, not a forecast at its first restart interval.  Only
# woof.physics_registry is reached, which imports nothing from woof and
# no device library, so the module stays importable wherever config.py is.
def _require_agreement_with_the_registry() -> None:
    import os

    from woof.physics_registry import (
        REGISTRY_REBUILD_ENV, component_options, require_registry_agreement)

    require_registry_agreement(
        "woof.checkpoint_identity.MICROPHYSICS_ALGORITHM_IDENTITIES",
        "microphysics", MICROPHYSICS_ALGORITHM_IDENTITIES)
    require_registry_agreement(
        "woof.checkpoint_identity.SURFACE_LAYER_ALGORITHM_IDENTITIES",
        "surface_layer", SURFACE_LAYER_ALGORITHM_IDENTITIES)
    require_registry_agreement(
        "woof.checkpoint_identity.LAND_SURFACE_ALGORITHM_IDENTITIES",
        "land_surface", LAND_SURFACE_ALGORITHM_IDENTITIES)
    require_registry_agreement(
        "woof.checkpoint_identity.LAND_SURFACE_PARAMETER_SOURCES",
        "land_surface", LAND_SURFACE_PARAMETER_SOURCES,
        cited_absences={0: (
            "no land surface runs and no parameter bundle is resolved: "
            "physics_setup_identity's sf_surface_physics != 0 arm")})
    require_registry_agreement(
        "woof.checkpoint_identity.PBL_ALGORITHM_IDENTITIES",
        "pbl", PBL_ALGORITHM_IDENTITIES)
    require_registry_agreement(
        "woof.checkpoint_identity.CUMULUS_ALGORITHM_IDENTITIES",
        "cumulus", CUMULUS_ALGORITHM_IDENTITIES)
    require_registry_agreement(
        "woof.checkpoint_identity.URBAN_ALGORITHM_IDENTITIES",
        "urban", URBAN_ALGORITHM_IDENTITIES)
    if os.environ.get(REGISTRY_REBUILD_ENV) == "1":
        return
    # Radiation selects on a (lw, sw) PAIR, so the id-set helper does not
    # apply; every implemented option's resolved pair must be nameable in
    # both spectra's tables.  The legacy aggregate selector (-1, -1)
    # resolves to 4/4 through ra_physics=4.
    for option_id, option in component_options("radiation").items():
        lw = int(option["selectors"]["ra_lw_physics"])
        sw = int(option["selectors"]["ra_sw_physics"])
        if (lw, sw) == (-1, -1):
            lw, sw = 4, 4
        for value, table, name in (
                (lw, LONGWAVE_ALGORITHM_IDENTITIES,
                 "LONGWAVE_ALGORITHM_IDENTITIES"),
                (lw, LONGWAVE_ABOVE_ATMOSPHERE_POLICIES,
                 "LONGWAVE_ABOVE_ATMOSPHERE_POLICIES"),
                (sw, SHORTWAVE_ALGORITHM_IDENTITIES,
                 "SHORTWAVE_ALGORITHM_IDENTITIES"),
                (sw, SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES,
                 "SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES")):
            if value not in table:
                raise RuntimeError(
                    f"radiation option {option_id!r} resolves to "
                    f"({lw}, {sw}) and woof.checkpoint_identity.{name} has "
                    f"no row {value}; a checkpoint of it would fail at the "
                    "first restart interval")


_require_agreement_with_the_registry()
