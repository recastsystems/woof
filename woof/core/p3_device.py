"""Device-resident P3 (``mp_physics`` 50/51/52/53): compilation, residency, launch.

``n_cat`` is the authority's number of free ice categories: 1 for
mp_physics = 50, 2 for mp_physics = 52 (module_microphysics_driver.F:1654,
``N_ICECAT=2``).  The kernels are templates on it (woof/core/kernels/p3.cu)
and each extern "C" entry point exists twice, the mp=50 name and the same
name with ``_c2``; a 2-category launch passes four more field pointers
(:data:`CATEGORY2_FIELD_SLOTS`), three more diagnostics
(:data:`CATEGORY2_DIAG_SLOTS`) and the ice-ice collection table
(:data:`CATEGORY_TABLE_SLOTS`).  Nothing about a 1-category launch changes.

This module is the CUDA half of the P3 port.  ``woof/core/p3.py`` keeps the
CPU float32 transcription as the explicit reference/debug path; everything
here runs on the card and leaves the prognostic state THERE.

WHAT RUNS WHERE
---------------
On the device, every timestep: the whole of ``p3_main`` and the WRF
wrapper's precipitation conversions, over prognostic fields that never
leave the card, against lookup tables uploaded once per process.
On the host, once per process: the Rust rw-p3 bridge validates and parses
the table and generates rain tables in parallel. Python owns the ABI and
allocations only.
On the host, per step: nothing but kernel launches.  There is no
host round trip of any prognostic field, no per-column Python loop and no
transpose, because the kernels address woof's native ``(nz, ny, nx)``
storage directly as ``(nk, ncol)``.

LAYOUT, AND WHY
---------------
The reference uses one thread per column. The level arms divide the
independent process updates among four threads per column, with adjacent
x threads still reading adjacent columns. Column flags are combined with
integer OR. The sedimentation level arm also divides each substep among
four threads: it reduces the Courant maximum, completes all fluxes, then
updates levels. Floating point sums retain their original order.

Fields stay LEVEL-MAJOR -- element (k, i) at ``k * ncol + i`` -- so a warp
reading level k touches 32 consecutive floats.  That is fully coalesced AND
it is already how a woof ``DomainState`` stores a 3-D field, so the port
adds no transpose anywhere.  The alternative (column-contiguous, which is
the shape the old host path built) gives every thread its own cache line
per load and costs a full-domain transpose twice per step.

SCRATCH ACCOUNTING
------------------
Eighteen ``(nk, ncol)`` float32 companions: twelve carrying values between
kernels (:data:`SCRATCH_SLOTS`) and six of sedimentation workspace
(:data:`SEDW_SLOTS`), which the three sedimentation kernels share because
they run in sequence.  That is 72 bytes per grid cell, allocated once
through ``DomainState.scratch`` and reused for the life of the run.  Six
diagnostics and seven surface fields are additional and are outputs, not
scratch.  Four candidates were deliberately NOT given arrays -- ``inv_dzq``,
``t_old``, ``ze_ice`` and ``ze_rain`` -- because each is exactly
reproducible from a field that is stored, so recomputing them is bit-exact
and 4 * nk * ncol bytes cheaper each.

ARMS
----
``unfused`` is the reference: nine launches, one per step of the authority,
and the arm every agreement number in the receipts was measured on.
``fused`` composes the same device step functions into three launches and
additionally merges the homogeneous-freezing and final-diagnostics k-loops.
Fusing reorders nothing here, but "reorders nothing" is checked by a byte
gate (``tests/test_p3_cuda.py``) rather than asserted, because this program
has already shipped a vectorisation that was wrong in 39 elements out of
37.8 million and only a byte gate caught it.
``levels`` uses four launches with level-parallel preparation, process
rates and final diagnostics. ``sedlevels`` also parallelises sedimentation
substeps and is selected by ``cuda``. The sequential arms remain available
as references; ``tests/test_p3_speed_levels.py`` compares their buffers.

CONTRACTION AND SUBNORMALS
--------------------------
The strict module uses ``-fmad=false`` on every arm. The Fortran reference arm was built with
``-ffp-contract=off``, so a contracted ``a*b+c`` on the device is a
different number.  With contraction off, plain infix operators in the .cu
are the ``__fmul_rn``/``__fadd_rn`` intrinsics; the kernel file carries a
two-kernel probe and the test compiles the module both ways to show the
equality holds under ``false`` and breaks under ``true``.

``--ftz=false`` likewise, and it is why the unit compiles through the direct
NVRTC route (:func:`woof.nvrtc_ptx_cache.compile_using_nvrtc`) and not
``cupy.RawModule``: CuPy appends ``-ftz=true`` after the caller's options
and NVRTC honours the last one, so through RawModule every float32 result
that should be subnormal came back zero.  gfortran on x86 keeps them.
MEASURED by the column oracle against WRF 4.6.1 (tools/p3_wrf461_oracle):
with the flush, one snow column of the 30-step regime set left WRF at step
4 and reached 19 ULP in the ice diagnostics by step 28; without it every
field of every column is bit-identical to the Fortran in the replay module,
as it is under GPUWM_WRF_EXACT=1. Two-category default fast launches use
CUDA REAL32 transcendental leaves instead of the replay leaves' binary64
internals. Explicit native double expressions and subnormal controls stay
unchanged; the normal numerics boundary controls contraction in both modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from woof.core.kernels import _ENCODING, _preamble

_KDIR = Path(__file__).resolve().parent / "kernels"

#: WOOF's own float32 routines (``r_exp``/``r_log``/``r_pow``), one copy
#: in this tree.  P3 borrows it exactly the way ``noahmp_driver_gpu`` does --
#: by compiling after it -- rather than carrying a second copy, which is
#: the rule ``woof/core/noahmp_kernel_sources.py`` states and the reason
#: those routines have that name.
_LIBM_SOURCE = _KDIR / "noahmp_leaves.cu"
_P3_SOURCE = _KDIR / "p3.cu"

#: NVRTC options before the common numerics boundary. Strict mode retains
#: ``-fmad=false`` to match the Fortran reference's statement order; normal
#: fast mode lets the boundary remove it, as for the other physics modules.
DEFAULT_OPTIONS: tuple[str, ...] = ("-std=c++17", "-fmad=false",
                                     "--ftz=false")

#: Prognostic + input field slots.  MUST match the F_* defines in p3.cu;
#: tests/test_p3_cuda.py parses the .cu and pins the two lists together.
#: ``qzi`` is the three-moment ice moment in its ADVECTED form (N*Z)**0.5,
#: the WRF Registry's QZI (mp_physics=53 only; the kernels convert it to Z
#: and back exactly where the wrapper does, module_mp_p3.F:826-838 and
#: :912-924).  A two-moment launch passes no array for it.
FIELD_SLOTS: tuple[str, ...] = (
    "qc", "nc", "qr", "nr", "qi", "qir", "ni", "qib",
    "th", "qv", "th_old", "qv_old", "ssat", "pres", "dz", "qzi",
)
#: Field slots a launch may leave without an array: the kernels read them
#: only under the flag named here.
OPTIONAL_FIELD_SLOTS: dict[str, str] = {"qzi": "log_3momentIce"}
#: Values carried between kernels.  MUST match the S_* defines in p3.cu.
SCRATCH_SLOTS: tuple[str, ...] = (
    "rho", "inv_rho", "qvs", "qvi", "sup", "supi",
    "rhofacr", "rhofaci", "acn", "t", "tmparr1", "qv_cld",
)
#: Sedimentation workspace, shared by the three sedimentation steps.
SEDW_SLOTS: tuple[str, ...] = ("v_q", "v_n", "flux_q", "flux_n",
                               "flux_qir", "flux_bir")
#: Three-moment ice adds Z's reflectivity-weighted fall speed and flux
#: (module_mp_p3.F:4419, :4444), slots W_VZ/W_FZ after the six above.
#: Allocated only for a three-moment workspace, so a two-moment domain's
#: footprint does not move.
SEDW_SLOTS_3MOM: tuple[str, ...] = ("v_z", "flux_z")
#: THE FULL DIAGNOSTIC SET, decided once, here.  Adding a seventh later
#: moves the history stream and invalidates receipts registered against it,
#: so the Registry's P3 package names all six and all six are emitted on
#: every call: refl_10cm (as zdbz), re_cloud, re_ice, vmi3d, di3d, rhopo3d.
DIAG_SLOTS: tuple[str, ...] = ("zdbz", "effc", "effi", "vmi", "di", "rhopo")
#: Surface fields: the two p3_main precipitation RATES plus the five WRF
#: accumulators the wrapper converts them into (:892-898).
SURF_SLOTS: tuple[str, ...] = ("prt_liq", "prt_sol", "rainnc", "rainncv",
                               "sr", "snownc", "snowncv")
#: p3_init products, in the order p3.cu's T_* defines expect.  A
#: two-moment p3_init fills itab/itabcoll, a three-moment one
#: itab_3mom/itabcoll_3mom (module_mp_p3.F:343-472 reads exactly one of
#: the two table-1 variants); the slots the variant leaves empty are
#: uploaded as null pointers, which no kernel path of that variant reads.
TABLE_SLOTS: tuple[str, ...] = ("itab", "itabcoll", "vn_table", "vm_table",
                                "revap_table", "itab_3mom", "itabcoll_3mom")

#: The second ice category's four prognostic fields (mp_physics = 52),
#: appended after :data:`FIELD_SLOTS` in the F_QI2.. order of p3.cu.
CATEGORY2_FIELD_SLOTS: tuple[str, ...] = ("qi2", "qir2", "ni2", "qib2")
#: mp_p3_wrapper_wrf_2cat's category-2 outputs diag_vmi2_3d, diag_di2_3d and
#: diag_rhopo2_3d (module_mp_p3.F:1137-1139; the Registry's vmi3d_2, di3d_2,
#: rhopo3d_2), appended after :data:`DIAG_SLOTS`.
CATEGORY2_DIAG_SLOTS: tuple[str, ...] = ("vmi2", "di2", "rhopo2")
#: The nCat > 1 table (p3_lookupTable_2), appended after :data:`TABLE_SLOTS`.
CATEGORY_TABLE_SLOTS: tuple[str, ...] = ("itabcolli1", "itabcolli2")

#: The ice category counts the kernels are instantiated for.
N_CAT_SUPPORTED: tuple[int, ...] = (1, 2)


def _check_n_cat(n_cat: int) -> int:
    n_cat = int(n_cat)
    if n_cat not in N_CAT_SUPPORTED:
        raise ValueError(
            f"P3 n_cat={n_cat}: the device kernels are instantiated for "
            f"{N_CAT_SUPPORTED} ice categories (mp_physics 50/51 and 52); "
            "a launch for any other count would index fields and tables "
            "that were never passed.")
    return n_cat


def field_slots(n_cat: int = 1) -> tuple[str, ...]:
    """The field pointer list a launch for ``n_cat`` categories passes."""
    return FIELD_SLOTS + (CATEGORY2_FIELD_SLOTS if _check_n_cat(n_cat) > 1
                          else ())


def diag_slots(n_cat: int = 1) -> tuple[str, ...]:
    """The diagnostic outputs a launch for ``n_cat`` categories writes."""
    return DIAG_SLOTS + (CATEGORY2_DIAG_SLOTS if _check_n_cat(n_cat) > 1
                         else ())


def table_slots(n_cat: int = 1) -> tuple[str, ...]:
    """The p3_init products a launch for ``n_cat`` categories reads."""
    return TABLE_SLOTS + (CATEGORY_TABLE_SLOTS if _check_n_cat(n_cat) > 1
                          else ())


def kernel_name(name: str, n_cat: int = 1) -> str:
    """The extern "C" entry point of ``name`` for ``n_cat`` categories."""
    return name if _check_n_cat(n_cat) == 1 else f"{name}_c{n_cat}"

#: Kernel names per arm, in launch order.
ARM_KERNELS: dict[str, tuple[str, ...]] = {
    "unfused": ("p3k_prep", "p3k_kloop1", "p3k_kloopmain",
                "p3k_sed_cloud", "p3k_sed_rain", "p3k_sed_ice",
                "p3k_homofreeze", "p3k_final", "p3k_saveold_precip"),
    "fused": ("p3k_fused_process", "p3k_fused_sed", "p3k_fused_finish"),
    "sedlevels": ("p3k_levels_prepare", "p3k_levels_process",
                  "p3k_levels_sed", "p3k_levels_finish"),
    "levels": ("p3k_levels_prepare", "p3k_levels_process",
               "p3k_fused_sed", "p3k_levels_finish"),
}

#: ``run.p3_backend`` -> the arm it selects.  The config spelling is the
#: user-facing one ("cuda" is what a run asks for); the arm name is the
#: verification one. Historical authority receipts name "unfused";
#: default runs select "sedlevels" and explicit "fused" keeps its meaning.
CONFIG_ARM: dict[str, str] = {"cuda": "sedlevels", "fused": "fused"}

#: Launch shape (columns per block at most, level groups) of each level-group
#: kernel.  The group counts and column caps are the shared-array shapes in
#: p3.cu (flags[8][16], flags[16][8], co[16][4]); a column's levels are dealt
#: to its groups round-robin, and only flags and the Courant maximum are
#: combined across groups, so the shape moves speed, never bits.
LEVEL_GROUP_SHAPE: dict[str, tuple[int, int]] = {
    "p3k_levels_prepare": (16, 8),
    "p3k_levels_process": (8, 16),
    "p3k_levels_sed": (4, 16),
}

#: Threads per block.  One thread is one column, so this is a pure
#: occupancy knob and changes no number; the byte gate covers it.
DEFAULT_BLOCK = 128


def p3_source(*, kernel_dir: Path = _KDIR, native_gnu239: bool = False) -> str:
    """The exact translation unit nvrtc is handed.

    Assembled the way ``noahmp_driver_gpu.driver_source`` assembles its
    own: preamble, then the shared libm unit, then this scheme's source.
    ``kernel_dir`` composes it from another tree's kernel files (the A146
    census gates the tree it scans, A193).
    """
    kdir = Path(kernel_dir)
    return (_preamble(kdir)
            + (kdir / _LIBM_SOURCE.name).read_text(encoding=_ENCODING)
            + ((kdir / "p3_gnu239.cuh").read_text(encoding=_ENCODING)
               if native_gnu239 else "")
            + (kdir / _P3_SOURCE.name).read_text(encoding=_ENCODING))


class _Image:
    """The compiled image, offered to the kernel manifest's recorder."""

    def __init__(self, ptx):
        self.ptx = ptx


def _load_image(ptx):
    import cupy as cp

    module = cp.cuda.function.Module()
    module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    return module


def p3_module(options: tuple[str, ...] = DEFAULT_OPTIONS,
              source: str | None = None, *, native_gnu239: bool = False):
    """The P3 translation unit, compiled once per option set and device.

    ``source`` exists so a gate can compile a deliberately perturbed copy
    and show the gate can fail; leave it ``None`` for the real thing.  A
    loaded image belongs to one device's context, so the cache is keyed by
    the current device as well (RawModule did that internally).
    """
    import cupy as cp

    if native_gnu239:
        from woof import numerics
        if not numerics.is_strict():
            raise ValueError("the pinned native P3 math requires strict numerics")
        options = (*options, "-DP3_NATIVE_GNU239=1")
    return _p3_module(int(cp.cuda.Device().id), tuple(options), source)


@lru_cache(maxsize=None)
def _p3_module(device: int, options: tuple[str, ...], source: str | None):
    # The direct NVRTC route, so --ftz=false survives (see CONTRACTION).
    from woof import nvrtc_ptx_cache as compiler

    code = source if source is not None else p3_source(
        native_gnu239="-DP3_NATIVE_GNU239=1" in options)
    ptx, _mapping = compiler.compile_using_nvrtc(code, options, None, "p3.cu")
    module = _load_image(ptx)
    try:
        from woof.certify.kernel_manifest import record_module
    except Exception:                                   # pragma: no cover
        pass
    else:
        record_module("woof.core.p3_device:p3"
                      + ("" if source is None else "(substituted-source)"),
                      source=code, options=options, module=_Image(ptx))
    return module


@dataclass(frozen=True)
class P3DeviceTables:
    """``p3_init``'s products, resident on the card for the whole run.

    Uploaded once per process per table root.  They are 320 KiB in total
    (itab 56 KiB, itabcoll 240 KiB, three rain tables 12 KiB each), which
    is small enough to stay hot in L2 and is why no texture or constant
    binding is used.  A 2-category set adds the two ice-ice collection
    tables, 977 KiB each (25x4x5x25x4x5 float32), read only by the
    inter-category collection.
    """

    itab: object
    itabcoll: object
    vn_table: object
    vm_table: object
    revap_table: object
    pointers: object          # uint64[7], device
    nbytes: int
    itab_3mom: object = None
    itabcoll_3mom: object = None

    itabcolli1: object = None
    itabcolli2: object = None
    n_cat: int = 1

    @property
    def three_moment(self) -> bool:
        return self.itab_3mom is not None

    @property
    def arrays(self) -> tuple:
        return tuple(a for a in (self.itab, self.itabcoll, self.vn_table,
                                 self.vm_table, self.revap_table,
                                 self.itab_3mom, self.itabcoll_3mom,
                                 self.itabcolli1, self.itabcolli2)
                     if a is not None)


_TABLE_CACHE: dict[tuple[int, str, bool], P3DeviceTables] = {}


def device_tables(runtime=None, root: str | None = None, *,
                  three_moment: bool | None = None, n_cat: int = 1,
                  native_gnu239: bool | None = None) -> P3DeviceTables:
    """Upload ``p3_init``'s tables and keep them resident.

    Rust rw-p3 parses the pinned tables and generates rain tables in
    parallel before any state is touched. Python owns allocation and
    the ABI; p3_tables retains the host reference and asset identities.  ``three_moment`` selects p3_init's ``trplMomI``
    (module_mp_p3.F:158-162); left ``None`` it follows ``runtime``.
    """
    import cupy as cp

    from woof.core.p3_tables import p3_table_root

    n_cat = _check_n_cat(n_cat)
    from woof import numerics
    if three_moment is None:
        three_moment = bool(runtime is not None and runtime.three_moment)
    if n_cat > 1 and three_moment:
        raise ValueError("P3 has no two-category three-moment package")
    if root is None:
        root = p3_table_root()
    if native_gnu239 is None:
        native_gnu239 = numerics.is_strict() and (n_cat > 1 or three_moment)
    key = (int(cp.cuda.Device().id), root, n_cat, bool(three_moment), native_gnu239)

    def upload():
        if runtime is None:
            from types import SimpleNamespace
            from woof.core.p3_rust import init_two_moment, init_three_moment, init_table2
            init = init_three_moment if three_moment else init_two_moment
            values = init(root)
            source = SimpleNamespace(
                **dict(zip(("itab_3mom", "itabcoll_3mom") if three_moment
                           else ("itab", "itabcoll"), values[:2])),
                **dict(zip(("vn_table", "vm_table", "revap_table"), values[2:])),
                three_moment=bool(three_moment))
            if n_cat > 1:
                source.itabcolli1, source.itabcolli2 = init_table2(root)
        else:
            source = runtime
        if bool(source.three_moment) != bool(three_moment):
            raise ValueError(
                "P3 device tables were asked for "
                f"{'three' if three_moment else 'two'}-moment ice but the "
                "p3_init runtime handed over carries the other table-1 "
                "variant; the kernels would read a table that is not there")
        arrays, ptr_values = {}, []
        native_rain = {}
        if native_gnu239:
            from woof.core.p3_rust import pinned_native_rain
            native_rain = dict(zip(("vn_table", "vm_table", "revap_table"),
                                   pinned_native_rain()))
        for name in table_slots(n_cat):
            value = native_rain.get(name, getattr(source, name, None))
            if value is None and name in CATEGORY_TABLE_SLOTS:
                raise ValueError(f"P3 runtime has no {name}: two-category launch needs table 2")
            if value is None:
                arrays[name] = None
                ptr_values.append(0)
                continue
            host = np.ascontiguousarray(value, dtype=np.float32)
            arrays[name] = cp.asarray(host)
            ptr_values.append(int(arrays[name].data.ptr))
        ptrs = cp.asarray(np.array(ptr_values, dtype=np.uint64))
        return P3DeviceTables(
            **arrays, pointers=ptrs, n_cat=n_cat,
            nbytes=int(sum(a.nbytes for a in arrays.values()
                           if a is not None)))

    # Published with its upload event: every slab on the card reads these
    # tables from its own stream, and an upload still in flight on the first
    # slab's stream is not a table the others may read.
    from woof.core.device_cache import cached_ready
    return cached_ready(cp, _TABLE_CACHE, key, upload)


def _pointer_array(arrays):
    import cupy as cp
    return cp.asarray(np.array([int(a.data.ptr) if a is not None else 0 for a in arrays],
                               dtype=np.uint64))


def scratch_bytes_per_cell(three_moment: bool = False) -> int:
    """Measured, not estimated: bytes of P3 scratch per grid cell."""
    return 4 * (len(SCRATCH_SLOTS) + len(SEDW_SLOTS)
                + (len(SEDW_SLOTS_3MOM) if three_moment else 0))


@dataclass
class P3DeviceWorkspace:
    """The per-domain device buffers, allocated once and reused.

    Held by the caller (the ``DomainState`` adapter caches it on the state)
    so that a normal timestep allocates nothing.
    """

    ncol: int
    nk: int
    carriers: dict
    sedw: dict
    flags: object
    carrier_ptrs: object
    sedw_ptrs: object
    binding_cache: dict = field(default_factory=dict, repr=False)
    launch_binding: object = field(default=None, repr=False)

    def bound_pointers(self, fields, diag, surf, n_cat=1):
        """Upload immutable pointer tables once per resident array binding.

        Retaining the arrays prevents an address from being recycled while
        its table is cached. The upload event orders another stream without
        a host synchronization, including graph-capture reuse.
        """
        import cupy as cp
        from woof.core.device_cache import cached_ready

        groups = tuple(tuple(mapping.get(name) for name in names) for mapping, names
                       in ((fields, field_slots(n_cat)), (diag, diag_slots(n_cat)),
                           (surf, SURF_SLOTS)))
        key = (int(cp.cuda.Device().id), int(n_cat),
               tuple(int(array.data.ptr) if array is not None else 0 for group in groups for array in group))

        def upload():
            return tuple(_pointer_array(group) for group in groups), groups

        pointers, _owners = cached_ready(cp, self.binding_cache, key, upload)
        self.launch_binding = pointers
        return pointers

    @property
    def nbytes(self) -> int:
        total = sum(a.nbytes for a in self.carriers.values())
        total += sum(a.nbytes for a in self.sedw.values())

        return int(total + self.flags.nbytes)

    @property
    def three_moment(self) -> bool:
        return all(n in self.sedw for n in SEDW_SLOTS_3MOM)


def make_workspace(ncol: int, nk: int, *, allocate=None,
                   three_moment: bool = False) -> P3DeviceWorkspace:
    """Allocate (or fetch) the scratch companions for one domain.

    ``allocate(slot, nlev)`` must return a contiguous float32 array of
    exactly ``nlev * ncol`` elements; its shape is the caller's business,
    which is what lets a ``DomainState`` hand back its own ``(nz, ny, nx)``
    scratch slots and keep every P3 allocation inside the allocation gate.
    """
    import cupy as cp

    def default_alloc(slot, nlev):
        return cp.zeros((nlev, ncol), dtype=cp.float32)

    alloc = allocate or default_alloc

    def take(slot, nlev):
        buf = alloc(slot, nlev)
        if buf.size != nlev * ncol:
            raise ValueError(f"P3 workspace slot {slot!r} has {buf.size} "
                             f"elements, expected {nlev * ncol}")
        if buf.dtype != np.float32:
            raise TypeError(f"P3 workspace slot {slot!r} is {buf.dtype}, "
                            "expected float32")
        return buf

    carriers = {n: take("p3_" + n, nk) for n in SCRATCH_SLOTS}
    sed_names = SEDW_SLOTS + (SEDW_SLOTS_3MOM if three_moment else ())
    sedw = {n: take("p3_sed_" + n, nk) for n in sed_names}
    # The two column-scope logical flags, one pair per column, float so the
    # allocation gate prices them with everything else.
    flags = take("p3_flags", 2)
    # A two-moment workspace hands the kernels null W_VZ/W_FZ pointers:
    # only log_3momentIce paths touch them.
    sed_ptrs = [int(sedw[n].data.ptr) if n in sedw else 0
                for n in SEDW_SLOTS + SEDW_SLOTS_3MOM]
    return P3DeviceWorkspace(
        ncol=ncol, nk=nk, carriers=carriers, sedw=sedw, flags=flags,
        carrier_ptrs=_pointer_array([carriers[n] for n in SCRATCH_SLOTS]),
        sedw_ptrs=cp.asarray(np.array(sed_ptrs, dtype=np.uint64)))


def run_p3_device(fields: dict, diag: dict, surf: dict, *,
                  workspace: P3DeviceWorkspace,
                  tables: P3DeviceTables | None = None,
                  dt: float, it: int,
                  log_predictNc: bool = False,
                  log_3momentIce: bool = False,
                  clbfact_dep: float = 1.0, clbfact_sub: float = 1.0,
                  arm: str = "unfused", block: int = DEFAULT_BLOCK,
                  options: tuple[str, ...] = DEFAULT_OPTIONS,
                  module=None, n_cat: int = 1) -> None:
    """Run one P3 step in place on device arrays.

    ``fields``/``diag``/``surf`` map the slot names above to cupy arrays
    that are ``(nk, ncol)`` (2-D) or ``(nk, ny, nx)`` (3-D with ny*nx =
    ncol); surface arrays are ``(ncol,)`` or ``(ny, nx)``.  Nothing is
    copied to the host.  ``n_cat = 2`` (mp_physics = 52) also needs the
    :data:`CATEGORY2_FIELD_SLOTS` fields, the :data:`CATEGORY2_DIAG_SLOTS`
    outputs and a 2-category table set.
    copied to the host.

    ``log_3momentIce`` is the authority's ``present(qzi1_3d)``
    (module_mp_p3.F:822-823): it needs ``fields["qzi"]`` (the advected
    (N*Z)**0.5), three-moment tables and a three-moment workspace, and each
    missing piece is refused here by name rather than read as a null
    pointer on the card.
    """
    if arm not in ARM_KERNELS:
        raise ValueError(
            f"unknown P3 arm {arm!r}: expected one of "
            f"{tuple(ARM_KERNELS)}")
    n_cat = _check_n_cat(n_cat)
    if n_cat > 1 and log_3momentIce:
        raise ValueError("P3 has no two-category three-moment package")
    ncol, nk = workspace.ncol, workspace.nk
    if tables is None:
        tables = device_tables(n_cat=n_cat, three_moment=log_3momentIce)
    if tables.n_cat != n_cat:
        raise ValueError(
            f"P3 launch for {n_cat} ice categories was handed the "
            f"{tables.n_cat}-category table set; the ice-ice collection "
            "pointers the 2-category kernels read would not be there")
    if module is None:
        from woof import numerics

        # mp52's default fast path uses REAL32 transcendental leaves.
        # Keep the replay module and every other P3 variant unchanged.
        if n_cat == 2 and not numerics.is_strict():
            options = (*options, "-DP3_FAST_2CAT=1")
        if numerics.is_strict() and (n_cat > 1 or log_3momentIce):
            options = (*options, "-DP3_NATIVE_GNU239=1")
        module = p3_module(options)

    def flat(arr, want_len):
        v = arr.reshape(-1)
        if v.size != want_len:
            raise ValueError(f"P3 device array has {v.size} elements, "
                             f"expected {want_len}")
        if v.dtype != np.float32:
            raise TypeError("P3 device arrays must be float32, got "
                            f"{v.dtype}")
        return arr

    if log_3momentIce:
        if not tables.three_moment:
            raise ValueError(
                "P3 three-moment launch was handed two-moment tables: the "
                "mu_i closure would read the 3momI table through a null "
                "pointer (upload them with device_tables(three_moment=True))")
        if not workspace.three_moment:
            raise ValueError(
                "P3 three-moment launch was handed a two-moment workspace: "
                "Z sedimentation needs the v_z/flux_z slots "
                "(make_workspace(..., three_moment=True))")
    elif tables.itab is None:
        raise ValueError(
            "P3 two-moment launch was handed three-moment tables: the "
            "kernels would read the 2momI table through a null pointer")
    fslots = field_slots(n_cat)
    dslots = diag_slots(n_cat)
    for name in fslots:
        if name in OPTIONAL_FIELD_SLOTS and not log_3momentIce:
            continue
        if name not in fields:
            raise ValueError(f"P3 device launch is missing field {name!r}")
        flat(fields[name], nk * ncol)
    for name in dslots:
        if name not in diag:
            raise ValueError(
                f"P3 device launch is missing diagnostic {name!r}")
        flat(diag[name], nk * ncol)
    for name in SURF_SLOTS:
        flat(surf[name], ncol)

    f_ptr, d_ptr, p_ptr = workspace.bound_pointers(fields, diag, surf, n_cat=n_cat)

    args = (f_ptr, workspace.carrier_ptrs, workspace.sedw_ptrs, d_ptr, p_ptr,
            tables.pointers, workspace.flags,
            np.int32(ncol), np.int32(nk), np.float32(dt), np.int32(it),
            np.int32(1 if log_predictNc else 0),
            np.int32(1 if log_3momentIce else 0),
            np.float32(clbfact_dep), np.float32(clbfact_sub))
    grid = ((ncol + block - 1) // block,)
    for name in ARM_KERNELS[arm]:
        entry = module.get_function(kernel_name(name, n_cat))
        if name == "p3k_levels_finish":
            entry((grid[0], nk), (block,), args)
        elif name in LEVEL_GROUP_SHAPE:
            columns, groups = LEVEL_GROUP_SHAPE[name]
            level_block = min(block, columns)
            entry(((ncol + level_block - 1) // level_block,),
                  (level_block, groups), args)
        else:
            entry(grid, (block,), args)


# ---------------------------------------------------------------------------
# The radar observation operator H_Z(x): P3's reflectivity as a PURE function
# of the state.  woof/core/kernels/p3_zdiag.cu carries the statement-level
# account; this is its compile site and launcher, plus the host replay of
# the same statements on the CPU authority's own helpers.
# ---------------------------------------------------------------------------

_ZDIAG_SOURCE = _KDIR / "p3_zdiag.cu"

#: Fields the operator reads, in kernel argument order.  ``th`` is the FULL
#: potential temperature and ``pres`` the full pressure, as the step reads
#: them (woof/core/p3.py apply: ``mp_th = thb + thp``, ``pres = state.p``).
REFLECTIVITY_FIELDS: tuple[str, ...] = (
    "qr", "nr", "qi", "qir", "ni", "qib", "th", "pres")


def _clear_air_dbz() -> float:
    """``10*log10((1e-22 + 1e-22)*1e18)`` in the kernel's float32 order.

    The two accumulator seeds of p3_final_level (WRF module_mp_p3.F
    :2286-2287) with no species added: the scheme's own reflectivity of a
    cell with no rain and no ice.  The device evaluates ``log10`` in double
    and rounds once to float32 (``p3_log10``), which is what this does.
    """
    seed = np.float32(1.0e-22)
    total = np.float32(np.float32(seed + seed) * np.float32(1.0e18))
    return float(np.float32(np.float32(10.0)
                            * np.float32(np.log10(np.float64(total)))))


#: The operator's ONE clear-air value, dBZ: what every cell with no rain and
#: no ice above QSMALL reads (-36.9897).  ``woof.da.obsop`` records it as
#: P3's clear-air floor; tests/test_p3_reflectivity_gpu.py holds the kernel
#: to it bit for bit.
CLEAR_AIR_DBZ = _clear_air_dbz()


def p3_reflectivity_source(*, kernel_dir: Path = _KDIR) -> str:
    """The forecast unit (:func:`p3_source`) with the operator appended.

    The forecast unit's bytes are unchanged, so the forecast module and its
    receipts do not move; the operator compiles as its own module.
    """
    kdir = Path(kernel_dir)
    return (p3_source(kernel_dir=kdir)
            + (kdir / _ZDIAG_SOURCE.name).read_text(encoding=_ENCODING))


def p3_reflectivity_module(options: tuple[str, ...] = DEFAULT_OPTIONS):
    """The operator's translation unit, once per option set and device."""
    import cupy as cp

    return _p3_reflectivity_module(int(cp.cuda.Device().id), tuple(options))


@lru_cache(maxsize=None)
def _p3_reflectivity_module(device: int, options: tuple[str, ...]):
    from woof import nvrtc_ptx_cache as compiler

    code = p3_reflectivity_source()
    ptx, _mapping = compiler.compile_using_nvrtc(code, options, None,
                                                 "p3_reflectivity.cu")
    module = _load_image(ptx)
    try:
        from woof.certify.kernel_manifest import record_module
    except Exception:                                   # pragma: no cover
        pass
    else:
        record_module("woof.core.p3_device:p3_reflectivity",
                      source=code, options=options, module=_Image(ptx))
    return module


def run_p3_reflectivity(fields: dict, out, *,
                        tables: P3DeviceTables | None = None,
                        block: int = 256,
                        options: tuple[str, ...] = DEFAULT_OPTIONS):
    """Write P3's reflectivity of ``fields`` into ``out`` (dBZ, float32).

    ``fields`` maps :data:`REFLECTIVITY_FIELDS` to float32 cupy arrays of
    one common size; ``out`` is a contiguous float32 cupy array of that
    size.  The inputs are read only.
    """
    import cupy as cp

    n_cat = 2 if "qi2" in fields else 1
    three_moment = "qzi" in fields
    if tables is None:
        tables = device_tables(three_moment=three_moment, n_cat=n_cat)
    if tables.three_moment != three_moment:
        raise ValueError("P3 reflectivity fields and ice table moment count differ")
    size = int(out.size)
    arrays = []
    for name in REFLECTIVITY_FIELDS + (("qzi",) if three_moment else ()) + (CATEGORY2_FIELD_SLOTS if n_cat > 1 else ()):
        if name not in fields:
            raise ValueError(
                f"P3 reflectivity needs field {name!r}; it reads "
                f"{REFLECTIVITY_FIELDS}")
        value = fields[name]
        if int(value.size) != size:
            raise ValueError(
                f"P3 reflectivity field {name!r} has {value.size} elements, "
                f"the output has {size}")
        if value.dtype != np.float32:
            raise TypeError(
                f"P3 reflectivity field {name!r} is {value.dtype}; the "
                "operator runs the scheme's float32 statements")
        arrays.append(cp.ascontiguousarray(value).reshape(-1))
    if out.dtype != np.float32 or not out.flags.c_contiguous:
        raise TypeError("P3 reflectivity output must be contiguous float32")
    module = p3_reflectivity_module(options)
    module.get_function("p3k_reflectivity_3mom" if three_moment else kernel_name("p3k_reflectivity", n_cat))(
        ((size + block - 1) // block,), (block,),
        (*arrays, tables.itab_3mom if three_moment else tables.itab,
         out.reshape(-1), np.int32(size)))
    return out


def p3_reflectivity_host(fields: dict) -> np.ndarray:
    """The same statements on the host, on the CPU authority's helpers.

    ``woof.core.p3`` is the float32 transcription the device port is
    measured against; this replays p3_final_level's Z half per cell with
    its ``get_rain_dsd2``, ``impose_max_total_Ni``, ``calc_bulkRhoRime``,
    ``find_lookupTable_indices_1a`` and ``access_lookup_table`` on local
    values.  One Python iteration per cell holding rain or ice: a
    reference and a small-domain path, not a production one.
    """
    from woof.core import p3 as P

    n_cat = 2 if "qi2" in fields else 1
    slots = REFLECTIVITY_FIELDS + (CATEGORY2_FIELD_SLOTS if n_cat > 1 else ())
    f32 = np.float32
    host = {name: np.asarray(fields[name], dtype=np.float32)
            for name in slots}
    shape = host["qr"].shape
    out = np.full(shape, f32(CLEAR_AIR_DBZ), dtype=np.float32)
    active = (host["qr"] >= P.QSMALL) | (host["qi"] >= P.QSMALL)
    if n_cat > 1:
        active |= host["qi2"] >= P.QSMALL
    if not active.any():
        return out
    itab = P.p3_init().itab
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for index in zip(*np.nonzero(active)):
            pres = host["pres"][index]
            tm = f32((pres * f32(1.0e-5)) ** (P.RD * P.INV_CP))
            t = f32(host["th"][index] * tm)
            rho = f32(pres / (P.RD * t))
            inv_rho = f32(f32(1.0) / rho)
            ze_rain = ze_ice = f32(1.0e-22)
            qr = host["qr"][index]
            if qr >= P.QSMALL:
                nr, mu_r, lamr, _c, _l = P.get_rain_dsd2(
                    qr, host["nr"][index], f32(1.0))
                lam2 = f32(lamr * lamr)
                lam6 = f32(f32(lam2 * lam2) * lam2)
                ze_rain = f32(rho * nr * (mu_r + f32(6.0))
                              * (mu_r + f32(5.0)) * (mu_r + f32(4.0))
                              * (mu_r + f32(3.0)) * (mu_r + f32(2.0))
                              * (mu_r + f32(1.0)) / lam6)
                ze_rain = max(ze_rain, f32(1.0e-22))
            numbers = [host["ni"][index]]
            if n_cat == 1:
                numbers[0] = P.impose_max_total_Ni(numbers[0], inv_rho)
            else:
                numbers.append(host["ni2"][index])
                total = f32(f32(f32(0.) + numbers[0]) + numbers[1])
                if f32(total * rho) >= P.MAX_TOTAL_NI:
                    scale = f32(P.MAX_TOTAL_NI * inv_rho / total)
                    numbers = [f32(n * min(scale, f32(1.))) for n in numbers]
            for cat in range(n_cat):
                suffix = "" if cat == 0 else "2"
                ni = numbers[cat]
                qi = host["qi" + suffix][index]
                if qi >= P.QSMALL:
                    ni = max(ni, P.NSMALL)
                    qir, _bir, rhop = P.calc_bulkRhoRime(
                        qi, host["qir" + suffix][index], host["qib" + suffix][index])
                    dumi, dumjj, dumii, d1, d4, d5 = \
                        P.find_lookupTable_indices_1a(qi, ni, qir, rhop)
                    f1pr09 = P.access_lookup_table(itab, dumjj, dumii, dumi, 7,
                                                   d1, d4, d5)
                    f1pr10 = P.access_lookup_table(itab, dumjj, dumii, dumi, 8,
                                                   d1, d4, d5)
                    f1pr13 = P.access_lookup_table(itab, dumjj, dumii, dumi, 9,
                                                   d1, d4, d5)
                    ni = min(ni, f32(f1pr09 * qi))
                    ni = max(ni, f32(f1pr10 * qi))
                    ze_ice = f32(ze_ice + f32(0.1892) * f1pr13 * ni * rho)
                    ze_ice = max(ze_ice, f32(1.0e-22))
            out[index] = f32(f32(10.0) * f32(np.log10(
                np.float64(f32((ze_rain + ze_ice) * f32(1.0e18))))))
    return out


def reflectivity(state, *, temperature=None, pressure=None):
    """H_Z(x) for a P3 state: ``(nz, ny, nx)`` dBZ, the state untouched.

    The state's own fields: ``qr``, ``nr``, ``qi``, ``qir``, ``ni``,
    ``qib``, full theta ``thb + thp`` and full pressure ``p``, exactly the
    arrays the forecast step reads (woof/core/p3.py ``apply``).  The step
    diagnoses no t1d/p1d pair of its own, so ``temperature``/``pressure``
    are refused rather than half honoured.  On a CuPy state the result is
    the state's ``refl_10cm`` scratch slot (copy it to keep it, as for the
    other device routes); on a NumPy state a fresh float32 array.
    """
    if temperature is not None or pressure is not None:
        raise ValueError(
            "the P3 reflectivity operator forms its temperature from the "
            "state's full theta and pressure, as the P3 step does; it takes "
            "no temperature/pressure pair")
    missing = [name for name in ("qr", "nr", "qi", "qir", "ni", "qib")
               if getattr(state, name, None) is None]
    if missing:
        raise ValueError(
            "the P3 reflectivity operator needs the scheme's prognostic "
            f"fields; this state carries no {missing}")
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    shape = tuple(state.p.shape)
    species = {name: getattr(state, name) for name in REFLECTIVITY_FIELDS
               if name not in ("th", "pres")}
    if getattr(state, "qi2", None) is not None:
        species.update({name: getattr(state, name) for name in CATEGORY2_FIELD_SLOTS})
    if getattr(state, "qzi", None) is not None:
        species["qzi"] = state.qzi
    if not hasattr(state.p, "get"):
        species["th"] = np.broadcast_to(
            np.asarray(thb, np.float32) + np.asarray(state.thp, np.float32),
            shape)
        if getattr(state, "use_theta_m", 0):
            from woof.core.constants import RVOVRD
            species["th"] = species["th"] / (np.float32(1.0) + np.float32(RVOVRD) * state.qv)
        species["pres"] = state.p
        if "qzi" not in species:
            return p3_reflectivity_host(species)
        # An explicit host-state diagnostic still uses the scheme's CUDA
        # closure. Forecast states are resident and never take this branch.
        import cupy as cp
        out = cp.empty(shape, dtype=cp.float32)
        run_p3_reflectivity({n: cp.asarray(v) for n, v in species.items()}, out)
        return cp.asnumpy(out)
    import cupy as cp

    theta = cp.empty(shape, dtype=cp.float32)
    theta[...] = (state.total_theta() if getattr(state, "use_theta_m", 0)
                  else thb + state.thp)
    species["th"] = theta
    species["pres"] = state.p
    out = state.scratch(shape, "refl_10cm")
    run_p3_reflectivity(species, out)
    del theta
    return out
