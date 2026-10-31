"""The device inventories a CUDA preparation is priced from.

The exact ``DomainState`` allocation list, one boundary interval's side
tables and the card's CUDA context: the three things
:mod:`woof.ingest.preparation_price` needs to price a preparation before
its first device allocation.  They lived in :mod:`woof.core.preflight`,
the forecast memory preflight, which the standalone RW-WPS wheel does not
stage; a price that imported them from there raised ModuleNotFoundError in
that wheel and stopped every GFS, ERA5 and mapped preparation, and the
wheel builder refused the staging on the three unresolved imports.  Here
they sit in a leaf both packages carry, and preflight re-exports every
name, so the forecast preflight and the preparation price read one
inventory.

Module scope is stdlib plus :mod:`woof.config`; the one other internal
lookup, :mod:`woof.boundary_fields`, is function-local.  No CuPy.
"""
from __future__ import annotations

from woof.core.column_schemes import COLUMN_SCHEMES

from dataclasses import dataclass

from woof.microphysics_schemes import scheme as _named_scheme

from woof.config import (CUMULUS_ADVECTIVE_FORCING_SCHEMES,
                          SASE_PBL_SCHEME, RunConfig)


#: What a CUDA context grows by once a forecast has loaded its kernel
#: modules, over the BARE context a fresh process stands up.
#:
#: MEASURED 2026-08-20 (task 206), two Linux cards, driver 13030.  The
#: bare context is read by ``tools/vram_reserve_probe.py`` in a process
#: that creates a context and allocates one byte; the run-time figure is
#: ``device footprint peak - pool-held peak`` from fifteen whole
#: forecasts' own 20 Hz :class:`~woof.core.gpu_mem_watch.
#: GpuPeakMemoryWatcher` receipts, minus the reservation law's backing
#: store for the frame those runs launched:
#:
#:   ==============  ==========  ============  ==========
#:   card            bare        at run time   growth
#:   ==============  ==========  ============  ==========
#:   RTX 5070 Ti      230.0 MiB   382.8 MiB     152.8 MiB
#:   RTX 5070 Ti      230.0 MiB   386.5 MiB     156.5 MiB
#:   RTX 5090         506.0 MiB   659.7 MiB     153.7 MiB
#:   RTX 5090         506.0 MiB   664.3 MiB     158.3 MiB
#:   ==============  ==========  ============  ==========
#:
#: The growth is a CONSTANT across a 2.4x span of card -- which is what
#: NVRTC module images and the driver's own working set should be, and
#: is not what a flat total context can be.  192 MiB rounds the worst
#: measurement up; an envelope must not round down.
#:
#: This is what RETIRES :data:`woof.core.preflight.CUDA_CONTEXT_BYTES`
#: as the charged term.  That constant is one 2026-07-26 reading of one
#: card, and applied everywhere it was wrong in BOTH directions at once:
#: 48 MiB high on a 5070 Ti and 215 MiB LOW on the very 5090 it was taken
#: from, once that card ran under Linux rather than WDDM.  A term that
#: under-charges is not conservative, it is an OOM waiting for a big
#: enough card.
CONTEXT_RUNTIME_GROWTH_BYTES = 192 * 1024 ** 2

#: Bare-context bytes per resident thread: what a card's CUDA context is
#: priced at from its shader census alone.
#:
#: Same campaign, the three cards' bare contexts over their
#: resident-thread capacities: 1,748 B (RTX 3080, WDDM, with a live
#: desktop sharing the card), 2,243 B (RTX 5070 Ti) and 2,032 B
#: (RTX 5090).  2,304 B is the figure rounded up above all three, so no
#: card is priced below any card that campaign measured.
#:
#: A MODELLED number, and it says so wherever it is printed
#: (:func:`non_pool_basis`).  It prices EVERY card this build reads, the
#: present one included, and that is deliberate.  A present card's bare
#: context used to be read off the card instead, as the NVML
#: ``memory.used`` delta either side of the probe's own context, and
#: that delta is a card-wide figure in whole MiB: on one idle RTX 4090
#: one probe printed 395 MiB (414,187,520 bytes) with every other field
#: of the probe identical across readings, and two earlier receipts of
#: the same plan sit exactly 3 and 4 MiB below it (bare contexts of 392
#: and 391 MiB by subtraction, never printed), so one plan was quoted
#: 1,162,304,164, 1,163,352,740 and 1,166,498,468 bytes by three
#: readings of one document.  An instrument that answers three
#: different numbers for one thing prices nothing; the census answers
#: one.  The run door's own machine (``tilestream.autoplan.Machine``)
#: has always priced the census, so this is also what makes the plan
#: review, ``woof check``, the wizard and the door agree on one card
#: to the byte.  A profile that arrives with a stated bare context (a
#: calibration receipt, a target-hardware sizing document) is priced
#: as stated; nothing in this build takes a fresh reading.
MODELLED_BARE_CONTEXT_BYTES_PER_RESIDENT_THREAD = 2304


@dataclass(frozen=True)
class DeviceLocalMemoryProfile:
    """The device constants the local-memory reservation law needs."""

    name: str
    multiprocessor_count: int
    max_threads_per_multiprocessor: int
    default_stack_limit_bytes: int = 1024
    #: Device bytes a bare CUDA context holds on this card, when a
    #: document STATED them (a calibration receipt, a target-hardware
    #: sizing file).  ``None`` for every card this build reads itself,
    #: present or absent, which is then priced from its census at
    #: :data:`MODELLED_BARE_CONTEXT_BYTES_PER_RESIDENT_THREAD`; the
    #: constant's note says why a fresh reading is never taken.
    bare_context_bytes: int | None = None
    #: The compile platform of THIS card as
    #: :func:`woof.certify.compile_platform.compile_platform_fingerprint`
    #: read it -- ``(device_compute_capability, nvrtc_build)`` -- when the
    #: profile was measured off a present card, and ``None`` for a card
    #: that is not in the machine or whose toolchain could not be
    #: resolved.  The per-thread frames of the Noah-MP composed units are
    #: readings of exactly this pair
    #: (:data:`woof.core.kernel_frame_recordings.NOAHMP_COMPOSED_FRAME_RECORDINGS`):
    #: a profile that carries a recorded pair prices ``sf_surface_physics
    #: = 4`` from that platform's own row, and a profile without one --
    #: or on a pair nobody has read -- prices it from the ceiling over the
    #: recorded rows with the basis stated
    #: (:func:`woof.core.noahmp_frame_provenance.frame_basis_for_profile`).
    compile_platform: tuple[str, str] | None = None

    @property
    def platform_is_read(self) -> bool:
        return self.compile_platform is not None

    @property
    def resident_thread_capacity(self) -> int:
        return (self.multiprocessor_count
                * self.max_threads_per_multiprocessor)

    @property
    def context_is_measured(self) -> bool:
        return self.bare_context_bytes is not None

    @property
    def cuda_context_bytes(self) -> int:
        """Device bytes this card's CUDA context holds during a run.

        The bare context as stated by a document that carries one
        (``bare_context_bytes``), else priced from the census at
        :data:`MODELLED_BARE_CONTEXT_BYTES_PER_RESIDENT_THREAD`, plus
        :data:`CONTEXT_RUNTIME_GROWTH_BYTES`, the module-load growth
        both instrumented Linux cards showed to within 5 MiB of each
        other.  No card this build reads is measured for it.
        """
        bare = self.bare_context_bytes
        if bare is None:
            bare = (MODELLED_BARE_CONTEXT_BYTES_PER_RESIDENT_THREAD
                    * self.resident_thread_capacity)
        return int(bare) + CONTEXT_RUNTIME_GROWTH_BYTES

    def reservation_bytes(self, max_local_size_bytes: int) -> int:
        """Device bytes the driver reserves for a launched kernel whose
        per-thread local frame is ``max_local_size_bytes``.  Zero when the
        frame fits the default stack, whose store the context already
        carries.

        VERIFIED EXACT 2026-08-20 on three cards and both driver models
        (``tools/vram_reserve_probe.py``, validated in both directions:
        frames at or under the default stack step exactly zero device
        bytes, frames above it step this product to the byte on WDDM and
        to within 1.5 MiB on Linux).  The law is not the defect; what was
        wrong was the profile it was evaluated on.
        """
        over = int(max_local_size_bytes) - self.default_stack_limit_bytes
        return 0 if over <= 0 else over * self.resident_thread_capacity


#: Measured 2026-07-26 from ``cudaGetDeviceProperties`` +
#: ``cudaDeviceGetLimit(cudaLimitStackSize)`` on the run host.
#:
#: ``bare_context_bytes`` is deliberately left unset even though this
#: card's bare context WAS measured (530,579,456 B, test host B,
#: 2026-08-20).  This profile is what an ABSENT card is priced against,
#: and the absent-card path may never be more optimistic than the
#: present-card one -- the 2026-08-03 lesson that retired
#: :data:`woof.core.preflight.CARD_CLASS_MULTIPROCESSORS`.  A card in
#: the machine is read for its own census
#: (:func:`woof.core.preflight.local_memory_profile_from_device`) and
#: priced from that census at the same per-thread rate.
MEASURED_LOCAL_MEMORY_PROFILE = DeviceLocalMemoryProfile(
    name="NVIDIA GeForce RTX 5090",
    multiprocessor_count=170,
    max_threads_per_multiprocessor=1536,
    default_stack_limit_bytes=1024,
)


def state_array_shapes(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Exact ``DomainState`` allocation list (state.py:52-237 transcribed).

    Every array ``DomainState.__init__`` allocates, keyed by attribute
    name, under the same conditionals (``cfg.moist``, microphysics scheme,
    ``cfg.terrain_opt``).  Cross-checked against the restart
    manifest's attribute classification by test (a state.py field added
    without updating BOTH manifests fails the suite).
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    m = (nz, ny, nx)
    xs = (nz, ny, nx + 1)
    ys = (nz, ny + 1, nx)
    fl = (nz + 1, ny, nx)
    s2 = (ny, nx)
    shapes: dict[str, tuple[int, ...]] = {
        # Prognostics + EOS diagnostics.
        "u": xs, "v": ys, "w": fl, "thp": m, "php": fl, "mup": s2,
        "p": m, "al": m, "alt": m,
        # RK time-t copies.
        "u0": xs, "v0": ys, "w0": fl, "thp0": m, "php0": fl, "mup0": s2,
        # Slow-tendency slots.
        "ru_t": xs, "rv_t": ys, "rw_t": fl, "rth_t": m, "rph_t": fl,
        "rmu_t": s2,
        # Acoustic-substep perturbations.
        "u_pp": xs, "v_pp": ys, "w_pp": fl, "th_pp": m, "ph_pp": fl,
        "mu_pp": s2, "p_pp": m, "p_pp_old": m, "ww_pp": fl, "al_pp": m,
        # General-form plumbing + map factors / rotation.
        "mub2d": s2, "ht": s2,
        "c1h": (nz,), "c2h": (nz,), "c1f": (nz + 1,), "c2f": (nz + 1,),
        "c3h": (nz,), "c4h": (nz,), "c3f": (nz + 1,), "c4f": (nz + 1,),
        # Float64-differenced full-level coefficient drops, read only by
        # the opt-2 EOS branch but allocated unconditionally beside c3f.
        "dc3f": (nz,), "dc4f": (nz,),
        "msft": s2, "msfu": (ny, nx + 1), "msfv": (ny + 1, nx),
        "msfv_inv": (ny + 1, nx),
        "f": s2, "e": s2, "sina": s2, "cosa": s2,
        # Vertical-coordinate arrays.
        "dnw": (nz,), "rdnw": (nz,), "dn": (nz,), "rdn": (nz,),
        "fnp": (nz,), "fnm": (nz,), "znu": (nz,), "znw": (nz + 1,),
    }
    from woof.wrf_exact import DIAGNOSTICS_ENABLED
    if DIAGNOSTICS_ENABLED:
        shapes["p_perturbation"] = m
    if cfg.terrain_opt == 0:
        shapes.update(thb=(nz,), pb=(nz,), alb=(nz,), phb=(nz + 1,),
                      dphb_resid=(nz,))
    else:
        # dphb_resid follows the base profiles, one HALF level shorter
        # than phb: it is the per-layer correction the EOS adds to the
        # float32 phb difference, so with terrain it costs one more
        # (nz, ny, nx) field -- 46 MiB at 400x400x76.
        shapes.update(thb=m, pb=m, alb=m, phb=fl, dphb_resid=m)
    if cfg.moist:
        for name in ("qv", "qc", "qr", "qv0", "qc0", "qr0", "h_diabatic"):
            shapes[name] = m
        if cfg.cu_physics in CUMULUS_ADVECTIVE_FORCING_SCHEMES:
            # WRF RTHFTEN/RQVFTEN, allocated by the same table predicate
            # woof/core/state.py uses.  Two persistent mass-point rates
            # priced in the VRAM projection for the schemes that read them
            # and for nobody else.
            shapes["rthften"] = m
            shapes["rqvften"] = m
        if cfg.mp_physics == 5:
            # Ferrier-Aligo (Registry.EM_COMMON:3019, and the mp==5 arm of
            # woof/core/state.py): total ice, the qt carrier, the three
            # WRF state fields, and qi's RK time-t copy.
            for name in ("qi", "qt", "f_ice", "f_rain", "f_rimef", "qi0"):
                shapes[name] = m
        elif cfg.mp_physics == 50:
            # P3 one-category (Registry.EM_COMMON:3038, and the mp==50 arm
            # of woof/core/state.py): ONE ice mass with rime mass and rime
            # volume, no qs/qg/effs, plus the two cross-step supersaturation
            # carriers p3_main writes at the end of every call.  Every
            # transported field carries its RK time-t copy.
            for name in ("qi", "ni", "nr", "qir", "qib", "effc", "effi",
                         "th_old", "qv_old",
                         "qi0", "ni0", "nr0", "qir0", "qib0"):
                shapes[name] = m
        elif cfg.mp_physics == 2:
            for name in ("qi","qs","qg","qi0","qs0","qg0"):
                shapes[name] = m
        elif cfg.mp_physics == 3:
            # WSM3 (Registry.EM_COMMON:3017, and the mp==3 arm of
            # woof/core/state.py): moist:qv,qc,qr plus the three radiation
            # radii; no qi/qs/qg and no RK copies beyond the qv/qc/qr ones
            # every moist state carries.
            for name in ("effc", "effi", "effs"):
                shapes[name] = m
        elif cfg.mp_physics in COLUMN_SCHEMES:
            for name in COLUMN_SCHEMES[cfg.mp_physics].species:
                if name not in ("qv", "qc", "qr"):
                    shapes[name] = shapes[name + "0"] = m
            for name in ("effc", "effi", "effs"):
                shapes[name] = m
        elif cfg.mp_physics in (6, 8, 9, 10, 16, 18, 24, 28, 106):
            # WRF's SIX-MASS moist package, transcribed.  This tuple is
            # not "the schemes with ice" -- it is the schemes whose
            # Registry package is moist:qv,qc,qr,qi,qs,qg
            # (Registry.EM_COMMON:3021 WSM6, :3024 Thompson, :3025
            # Milbrandt-Yau, :3026 Morrison, :3031 WDM6, :3033 NSSL,
            # :3036 Thompson aerosol-aware), which is what makes qs/qg
            # and the third effective radius allocatable at all.
            #
            # mp=50 IS DELIBERATELY OUT, and the ``elif`` is the
            # structural half of that decision: woof/core/state.py's
            # allocator spells the same split the same way, so this
            # manifest stays a transcription of it and no later edit can
            # hand P3 both packages.  P3's Registry row is
            # moist:qv,qc,qr,qi with NO qs and NO qg, and
            # state:re_cloud,re_ice with NO re_snow
            # (Registry.EM_COMMON:3038); WRF's driver binds it with
            # N_ICECAT=1 and no QS/QG dummy in the argument list at all
            # (module_microphysics_driver.F:1569-1602, diag_effc_3d and
            # diag_effi_3d and no snow radius).  Its package is priced by
            # the mp==50 arm above.
            #
            # Two things break if 50 joins this tuple, both measured:
            # (1) the manifest declares qs/qg/effs/qs0/qg0 that the state
            #     builder never allocates, so the shared dycore-state
            #     workspace is sized for five phantom fields -- the
            #     equality in tests/test_mp_accepted_builds.py::
            #     test_accepted_mp_builds_its_real_case_workspace fails on
            #     exactly those five names; and
            # (2) scratch_slot_registry's absent-mass predicate below is
            #     "qi declared and qs NOT declared", so a declared qs
            #     silently drops ``moist_absent_mass`` -- the shared zero
            #     plane woof/core/moist.py really allocates for P3's
            #     calc_cq and slow_buoyancy -- out of the VRAM
            #     projection, leaving the arena one (nz, ny, nx) FP32
            #     plane short.
            for name in ("qi", "qs", "qg", "qi0", "qs0", "qg0",
                         "effc", "effi", "effs"):
                shapes[name] = m
        elif cfg.mp_physics == 30:
            # FAST_SBM (Registry/registry.sbm:318, and the mp==30 arm of
            # woof/core/state.py): the six-mass moist package, the 142
            # spectral bins and six number moments as scalars, the two
            # cross-step carriers and three effective radii, with every
            # transported field's RK time-t copy.  Chained after the
            # six-mass arm so that arm stays the mp==50 arm's only sibling
            # (tests/test_preflight.py pins the chain).
            from woof.core.fast_sbm_contract import BIN_SPECIES,EXTRA_SPECIES,NUMBER_SPECIES
            for name in (*BIN_SPECIES,"qi","qs","qg",*NUMBER_SPECIES,
                         "th_old","qv_old","effc","effi","effs"):
                shapes[name]=m
            for name in EXTRA_SPECIES:
                shapes[name+"0"]=m
        if cfg.mp_physics == 24:
            # WSM7: the hail mass beside the six (Registry.EM_COMMON:3034)
            # with its RK time-t copy, as woof/core/state.py allocates.
            for name in ("qh", "qh0"):
                shapes[name] = m
        if cfg.mp_physics == 9:
            # Milbrandt-Yau two-moment: hail mass beside graupel plus a
            # number moment for EVERY one of the six hydrometeors
            # (woof/core/moist.py::MY2_SPECIES), each transported field
            # with its RK time-t copy (woof/core/state.py, the mp==9
            # arms).  This block was missing at 1.9.0: the state builder
            # requested rebuilt("qi0"...) views from a shared workspace
            # this manifest had never priced, so an ACCEPTED mp=9 config
            # could not build its real-case workspace (1.9.1 D1).
            for name in ("qh", "nc", "nr", "ni", "ns", "ng", "nh",
                         "qh0", "nc0", "nr0", "ni0", "ns0", "ng0", "nh0"):
                shapes[name] = m
        if cfg.mp_physics == 16:
            for name in ("nn", "nc", "nr", "nn0", "nc0", "nr0"):
                shapes[name] = m
        if cfg.mp_physics == 8:
            for name in ("nr", "ni", "nr0", "ni0"):
                shapes[name] = m
        if cfg.mp_physics == 10:
            for name in ("nc", "nr", "ni", "ns", "ng", "nr0", "ni0",
                         "ns0", "ng0", "effr"):
                shapes[name] = m
        if cfg.mp_physics == 18:
            for name in (
                    "qh", "qndrop", "qnr", "qni", "qns", "qng", "qnh",
                    "qnn", "qvolg", "qvolh", "qh0", "qndrop0", "qnr0",
                    "qni0", "qns0", "qng0", "qnh0", "qnn0", "qvolg0",
                    "qvolh0"):
                shapes[name] = m
        if cfg.mp_physics == 28:
            # Thompson aerosol-aware: prognostic droplet number plus the two
            # aerosol number tracers, each with its RK time-t copy
            # (woof/core/state.py, the mp==28 arms).
            for name in ("nc", "nr", "ni", "nwfa", "nifa",
                         "nc0", "nr0", "ni0", "nwfa0", "nifa0"):
                shapes[name] = m
            # QNWFA2D / QNIFA2D surface emission tendencies, # kg-1 s-1.
            # Cross-step constants, allocated once per domain.
            for name in ("nwfa2d", "nifa2d"):
                shapes[name] = s2
        named = _named_scheme(cfg.mp_physics)
        if named is not None:
            # A NAMED scheme: the capability row is the manifest, the same
            # row woof/core/state.py allocates from.
            species = (named.ice_mass_species + named.moment_species
                       + named.aerosol_species)
            for name in species + tuple(n + "0" for n in species) + ("effc", "effi", "effs"):
                shapes[name] = m
            for name in named.surface_fields:
                shapes[name] = s2
    if cfg.bl_pbl_physics == 5 and cfg.bl_mynn_tkeadvect:
        shapes["qke_adv"] = m
        shapes["qke_adv0"] = m
    from woof.core.physics_inventory import PBL_ADVECTED_SCALARS
    for name, _ in PBL_ADVECTED_SCALARS.get(cfg.bl_pbl_physics, ()):
        shapes[name] = m
        shapes[name + "0"] = m
    if cfg.km_opt == 2:
        # WRF's two-time-level prognostic TKE (Registry.EM_COMMON:312):
        # the SERIALIZED carrier plus its REBUILT time-t copy.
        shapes["tke"] = m
        shapes["tke0"] = m
    if cfg.bl_pbl_physics in (SASE_PBL_SCHEME, 11):
        # The published subgrid energy (state.py allocates it under the
        # same two-scheme condition): SASE's prognostic closure energy,
        # or Shin-Hong's per-step TKE diagnostic -- the D1 gray-zone
        # instrument reads state.e_sgs whichever closure produced it.
        shapes["e_sgs"] = m
    shapes.update(chem_state_array_shapes(cfg))
    return shapes


def chem_state_array_shapes(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """The chem arrays a chem domain allocates, as float32-word shapes.

    Every transported row's field and RK time-t copy and every prescribed
    row's field, each (nz, ny, nx) float32 (views into one arena, so the
    sum is the arena), plus the mass ledger's (nrows,) float64 vectors,
    priced here as 2 * nrows float32 words each
    (woof/core/chem_state.py:attach_chem_state).  When any process runs,
    also every array a process declares (``ALLOCATES``, a float64 one at
    twice its words), the chem_prep met fields
    (woof/core/chem_prep.py:ChemPrep.refresh) and the per-row deposition
    velocity plane (woof/core/chem_driver.py:_context), each allocated
    once per chem domain.  Empty while chem is off.
    """
    if not getattr(cfg, "chem_sets", ""):
        return {}
    from woof.chem_table import load
    from woof.core.chem_context import (
        CHEM_PREP_FIELDS, CHEM_PREP_W_FIELDS, LEDGER_BUCKETS,
        allocation_shape, chem_processes,
    )
    from woof.core.chem_state import LEDGER_TOTALS, ledger_attr, process_attr

    table = load(cfg)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    m = (nz, ny, nx)
    shapes: dict[str, tuple[int, ...]] = {}
    for row in table.transported:
        shapes[row.state_attr] = m
        shapes[row.time_attr] = m
    for row in table.prescribed:
        shapes[row.state_attr] = m
    n = len(table.transported)
    for name in (*LEDGER_TOTALS, *LEDGER_BUCKETS):
        shapes[ledger_attr(name)] = (2 * n,)
    shapes[ledger_attr("started")] = (1,)
    processes = chem_processes(table) if table.processes else []
    for _key, module in processes:
        rows = module.rows(table)
        for alloc in getattr(module, "ALLOCATES", ()):
            shape = allocation_shape(alloc, len(rows), nz, ny, nx)
            if str(alloc.dtype).endswith("64"):
                shape = (2, *shape)
            shapes[process_attr(alloc)] = shape
    if processes:
        for name in CHEM_PREP_FIELDS:
            shapes["chemprep_" + name] = (
                (nz + 1, ny, nx) if name in CHEM_PREP_W_FIELDS else m)
        shapes["chemprep_ddvel"] = (len(table.rows), ny, nx)
    return shapes


def chem_process_memory_shapes(cfg):
    """Additional live process buffers, in float32 words, including peaks."""
    if not getattr(cfg, "chem_sets", ""):
        return {}, {}
    from woof.chem_table import load
    from woof.core.chem_context import chem_processes
    table = load(cfg)
    held, transient = {}, {}
    for key, module in chem_processes(table):
        price = getattr(module, "memory_shapes", None)
        if price is not None:
            keep, work = price(cfg, len(module.rows(table)))
            held.update({"chem/" + key + "/" + name: shape for name, shape in keep.items()})
            transient.update({"chem/" + key + "/" + name: shape for name, shape in work.items()})
    return held, transient


#: d01 external-LBC field inventory (state boundaries built by
#: build_state_lateral_boundaries: u/v/theta/phi/mu + selected scalars) with
#: each field's (levels, ny-extent, nx-extent) source dims.
#: ``boundary_species`` is the source's published hydrometeor inventory
#: (woof.boundary_fields.source_boundary_species): the analysed masses and
#: their seeded numbers the root's boundary tables then carry.
def _lbc_field_dims(cfg: RunConfig, *, boundary_species=()
                    ) -> dict[str, tuple[int, int, int]]:
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    dims = {"u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx),
            "theta": (nz, ny, nx), "phi": (nz + 1, ny, nx),
            "mu": (1, ny, nx)}
    from woof.boundary_fields import potential_external_scalar_fields
    for name in potential_external_scalar_fields(
            cfg, boundary_species=boundary_species):
        dims[name] = (nz, ny, nx)
    if getattr(cfg, "chem_sets", "") and cfg.specified:
        # The chem rows a data-store source fills ride the same tables
        # (woof.chem_source_init.chem_boundary_fields).
        from woof.chem_source_init import chem_boundary_fields
        from woof.chem_table import load as load_chem_table
        for name in chem_boundary_fields(load_chem_table(cfg), cfg):
            dims[name] = (nz, ny, nx)
    return dims


def lbc_interval_values(cfg: RunConfig, *, boundary_species=()) -> int:
    """FP32 values in ONE interval's side tables (value + tendency), per
    ``_field_boundary`` (lateral_bc.py:149-167): west/east
    ``(lev, ny, W)`` + south/north ``(lev, W, nx)``, each twice."""
    width = cfg.spec_bdy_width
    total = 0
    for lev, ny, nx in _lbc_field_dims(
            cfg, boundary_species=boundary_species).values():
        total += 2 * (2 * lev * ny * width + 2 * lev * width * nx)
    return total


__all__ = [
    "CONTEXT_RUNTIME_GROWTH_BYTES", "DeviceLocalMemoryProfile",
    "MEASURED_LOCAL_MEMORY_PROFILE",
    "MODELLED_BARE_CONTEXT_BYTES_PER_RESIDENT_THREAD",
    "chem_state_array_shapes", "lbc_interval_values", "state_array_shapes",
]
