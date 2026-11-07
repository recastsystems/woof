"""SSiB land surface, WRF ``sf_surface_physics = 8``, on CUDA.

The kernel (``woof/core/kernels/ssib.cu``) transcribes WRF v4.6.1
``phys/module_sf_ssib.F`` and the ``CASE (SSIBSCHEME)`` arm of
``phys/module_surface_driver.F`` one thread per column.  This module owns the
record layout the kernel reads, the vegetation tables it is given, and the
launch.

Records are field-major ``(n_fields, ncol)`` float32 arrays:

* ``STATIC``  -- per-column constants,
* ``FORCING`` -- per-step inputs, followed by ``nz`` CLDFRA levels,
* ``STATE``   -- carried state, updated in place.

The same three lists define the column oracle's file layout
(``tools/ssib_wrf461_oracle``), so the replay test and the forecast drive
one kernel entry through one layout.
"""

from __future__ import annotations

import hashlib
import json
from woof.core.device_cache import cuda_cache
from pathlib import Path

import numpy as np

#: Soil-geometry WRF resolves for SSIB (share/module_check_a_mundo.F:3571-
#: 3572); the three "layers" are canopy, ground surface and deep soil
#: temperatures and the three SSiB soil-moisture zones.
NUM_SOIL_LAYERS = 3
#: init_soil_depth_8 (share/module_soil_pre.F:1269-1285).
SOIL_LAYER_DEPTHS_M = (0.00, 0.05, 1.05)
SOIL_LAYER_THICKNESS_M = (0.05, 1.00, 0.00)

#: mminlu codes the kernel reads (module_sf_ssib.F:1009-1019).
MMINLU_CODES = {"SSIB": 1, "USGS": 2, "MODIS": 3,
                "MODIFIED_IGBP_MODIS_NOAH": 4}

STATIC = ("xland", "ivgtyp", "xlat", "mminlu", "ra_sw", "dtbl")
FORCING = ("rainncv", "raincv", "glw", "coszen", "swdown",
           "swvisdir", "swvisdif", "swnirdir", "swnirdif",
           "dz8w1", "u1", "v1", "qv1", "t1", "p1", "psfc",
           "hfx_sea", "qfx_sea", "lh_sea", "tsk_sea",
           "xice", "xice_save", "julian", "t2_sfclay")
SNOW_LAYER_VARS = ("dzo", "wo", "tssn", "tssno", "bwo", "bto", "cto",
                   "fio", "flo", "bio", "blo", "ho")
STATE = (("smois1", "smois2", "smois3", "tslb1", "tslb2", "tslb3",
          "snow", "canwat", "sfcrunoff", "sfcevp",
          "tsk", "t2", "q2", "albedo", "ust", "hfx", "lh", "grdflx", "qfx",
          "znt", "gz1oz0", "br", "u10", "v10", "snowh", "tsk_save",
          "ssib_br", "ssib_fm", "ssib_fh", "ssib_cm", "ssibxdd",
          "ssib_lhf", "ssib_shf", "ssib_ghf", "ssib_egs", "ssib_eci",
          "ssib_ect", "ssib_egi", "ssib_egt", "ssib_sdn", "ssib_sup",
          "ssib_ldn", "ssib_lup", "ssib_wat", "ssib_shc", "ssib_shg",
          "ssib_lai", "ssib_vcf", "ssib_z00", "ssib_veg",
          "alswvisdir", "alswvisdif", "alswnirdir", "alswnirdif",
          "swvisdir", "swvisdif", "swnirdir", "swnirdif",
          "isnow", "swe", "snowden", "snowdepth", "tkair")
         + tuple(f"{v}{k}" for k in range(1, 5) for v in SNOW_LAYER_VARS))

#: The WRF Registry's SSiB state (Registry/registry.ssib) that is not also
#: an ordinary land-surface field: what a restart must carry.
SSIB_REGISTRY_STATE = (
    ("ssib_fm", "ssib_fh", "ssib_cm", "ssibxdd", "ssib_br",
     "ssib_lhf", "ssib_shf", "ssib_ghf", "ssib_egs", "ssib_eci", "ssib_ect",
     "ssib_egi", "ssib_egt", "ssib_sdn", "ssib_sup", "ssib_ldn", "ssib_lup",
     "ssib_wat", "ssib_shc", "ssib_shg", "ssib_lai", "ssib_vcf", "ssib_z00",
     "ssib_veg", "isnow", "swe", "snowden", "snowdepth", "tkair")
    + tuple(f"{v}{k}" for k in range(1, 5) for v in SNOW_LAYER_VARS)
    + ("alswvisdir", "alswvisdif", "alswnirdir", "alswnirdif",
       "swvisdir", "swvisdif", "swnirdir", "swnirdif"))

#: Error bits (kernels/ssib.cu SSIB_ERR_*) and the WRF breakage each names.
ERRORS = {
    1: ("a land column's vegetation category maps to no SSiB type "
        "(module_sf_ssib.F:1022-1026 'ERROR in vegetation/landuse map')"),
    2: ("the land-use map is not one SSIB accepts: SSIB, USGS, MODIS or "
        "MODIFIED_IGBP_MODIS_NOAH (module_sf_ssib.F:1009-1019)"),
    4: ("the multi-layer snow pack's pore volume went negative "
        "(TEMRS2 'udum0 is WRONG', module_sf_ssib.F:5218-5221, a STOP)"),
}
#: Recorded, not fatal: TEMRS2 leaves ICASE unset when the top layer's
#: ice fraction exceeds 1 (module_sf_ssib.F:5413-5430).
ERR_ICASE_UNSET = 8

#: Default-arm fixes (kernels/ssib.cu SSIB_FIX_*); strict runs none.
FIX_SEAICE_SNOW_UNITS = 1
FIX_T2_CANOPY_AIR = 2
FIXES_ALL = FIX_SEAICE_SNOW_UNITS
# TA is canopy air, not a measured 2 m temperature. Keep WRF's T2 writer.

_DATA = Path(__file__).resolve().parents[1] / "data" / "ssib"
TABLE_PATH = _DATA / "ssib_veg_tables.json"
#: SHA-256 of TABLE_PATH, extracted from WRF v4.6.1 module_sf_ssib.F by
#: tools/ssib_wrf461_oracle/gen_tables.py and checked there against the bits
#: gfortran compiled.  A table whose bytes differ is refused at load: a
#: silently edited vegetation constant moves every SSiB column.
SSIB_TABLE_SHA256 = "2f1022209b782534eddf5d9a969e8e8169f867cde49a3145ec06eb2db991ca84"
#: Words the kernel's ``ssib_tab`` holds (NTAB_SSIB).
NTAB = 2670


def fixes_for_run() -> int:
    """All default-arm fixes, or none under strict WRF arithmetic."""
    from woof import wrf_exact
    return 0 if wrf_exact.ENABLED else FIXES_ALL


def load_tables(path: Path = TABLE_PATH, *, verify: bool = True) -> np.ndarray:
    """The ``ssib_tab`` words, in kernel order, after the hash pin."""
    raw = Path(path).read_bytes()
    if verify:
        manifest = json.loads((_DATA / "MANIFEST.json").read_text())
        from woof.core.noah import TBL_DIR
        landuse = Path(TBL_DIR) / "LANDUSE.TBL"
        expected = manifest["files"]["../noah_tables/LANDUSE.TBL"]
        if hashlib.sha256(landuse.read_bytes()).hexdigest() != expected:
            raise ValueError("the SSiB LANDUSE.TBL manifest hash differs; the cold surface optics would change")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != SSIB_TABLE_SHA256:
            raise ValueError(
                f"{path} has SHA-256 {digest}, not the pinned "
                f"{SSIB_TABLE_SHA256}: the SSiB vegetation tables differ from "
                "the WRF v4.6.1 transcription, and every SSiB column would "
                "integrate with them")
    doc = json.loads(raw)
    parts = []
    for name in doc["layout"]:
        words = [int(h, 16) for h in doc["tables"][name]["f32_hex"]]
        parts.append(np.array(words, "<u4").view("<f4"))
    tab = np.concatenate(parts).astype(np.float32)
    if tab.size != NTAB:
        raise ValueError(f"{path}: {tab.size} table words, kernel holds {NTAB}")
    return tab


@cuda_cache(maxsize=None, ready=True)
def _module_with_tables(tab_bytes: bytes, resident: bool = False,
                        direct_forcing: bool = False):
    """The compiled module with ``ssib_tab`` set (one per table content)."""
    import cupy as cp
    from woof.core.kernels import load_module_int_defines

    # Each table content owns a CUDA module and its constant memory. The
    # generic loader otherwise shares one module: uploading a mutation there
    # poisons a later cached launch of the original tables. All 256 digest
    # bits enter the module key; each integer is positive for the loader.
    digest = hashlib.sha256(tab_bytes).digest()
    defines = tuple((f"SSIB_TABLE_HASH_{i}",
                     int.from_bytes(digest[4*i:4*i+4], "big") + 1)
                    for i in range(8))
    if resident:
        defines += (("SSIB_DIRECT_FIELDS", 1),)
    if direct_forcing:
        defines += (("SSIB_DIRECT_FORCING", 1),)
    module = load_module_int_defines("ssib", defines)
    table = np.frombuffer(tab_bytes, np.float32)
    _table_view(module).set(table)
    return module


def _table_view(module):
    """View constant storage owned by either CUDA module interface."""
    import cupy as cp

    if hasattr(module, "get_global"):
        pointer = module.get_global("ssib_tab")
        return cp.ndarray((NTAB,), cp.float32, memptr=pointer)
    address = module.get_global_var("ssib_tab")
    memory = cp.cuda.UnownedMemory(address, NTAB * np.dtype(np.float32).itemsize,
                                  module)
    pointer = cp.cuda.MemoryPointer(memory, 0)
    return cp.ndarray((NTAB,), cp.float32, memptr=pointer)


def kernel(tables: np.ndarray | None = None, *, resident: bool = False,
           direct_forcing: bool = False):
    if direct_forcing and not resident:
        raise ValueError("direct SSiB forcing needs resident fields")
    tab = load_tables() if tables is None else np.ascontiguousarray(
        tables, np.float32)
    return _module_with_tables(tab.tobytes(), resident, direct_forcing).get_function("ssib_surface_step")


def step_fields(owner, records, *, nz: int, itimestep: int, status):
    """Update the bound device fields, without carried-state staging."""
    if isinstance(itimestep, bool) or int(itimestep) < 1:
        raise ValueError("SSiB itimestep must be a positive integer")
    ncol = records.ncol
    if records.direct_forcing:
        values, cloud, cloud_present, julian, dt = records.inputs
        # RawKernel's native Function is already compiled and bound here.
        # The kernel-pack Function uses the same launch interface.
        function = getattr(owner.function, 'kernel', owner.function)
        function(records.grid, records.block,
            (owner.static, records.descriptors, cloud, records.ncol_arg,
             records.nz_arg, np.int32(itimestep), records.fixes_arg,
             status, records.nx_arg, *values,
             records.cloud_flags[cloud_present], julian, dt,
             records.commit_descriptors, records.finish_arg))
        return
    owner.function(((ncol + 63) // 64,), (64,),
        (owner.static, records.descriptors, owner.forcing, np.int32(ncol),
         np.int32(nz), np.int32(itimestep), np.int32(fixes_for_run()),
         status, np.int32(records.nx)))


def step(static, state, forcing, *, nz: int, itimestep: int,
         fixes: int | None = None, tables: np.ndarray | None = None,
         function=None, status=None, defer_status_mask=False) -> int | None:
    """Launch SSiB, with synchronous error readback only for the oracle.

    ``static`` (len(STATIC), ncol), ``state`` (len(STATE), ncol) updated in
    place, ``forcing`` (len(FORCING) + nz, ncol): float32 CuPy arrays.
    Production supplies a prebound ``function`` and persistent int32
    ``status``. Fatal bits accumulate there for the model's existing health
    boundary, without a device read or table-file access in this call.
    """
    import cupy as cp

    ncol = int(state.shape[1])
    if static.shape != (len(STATIC), ncol) or state.shape != (len(STATE), ncol) \
            or forcing.shape != (len(FORCING) + nz, ncol):
        raise ValueError("SSiB record shapes do not match the layout")
    for a in (static, state, forcing):
        if a.dtype != cp.float32 or not a.flags.c_contiguous:
            raise ValueError("SSiB records must be C-contiguous float32")
    if isinstance(itimestep, bool) or int(itimestep) < 1:
        raise ValueError("SSiB itimestep must be a positive integer")
    asynchronous = status is not None
    err = status if asynchronous else cp.zeros(1, cp.int32)
    if err.shape != (1,) or err.dtype != cp.int32:
        raise ValueError("SSiB status must be one int32 device word")
    fn = kernel(tables) if function is None else function
    block = 64
    fn(((ncol + block - 1) // block,), (block,),
       (static, state, forcing, np.int32(ncol), np.int32(nz),
        np.int32(itimestep), np.int32(fixes_for_run() if fixes is None
                                      else fixes), err))
    if asynchronous:
        # ICASE-unset is a witnessed native undefined local, not a fatal
        # error. The diagnostic replay retains bit 8; health sees only the
        # three conditions for which native WRF aborts.
        if not defer_status_mask:
            cp.bitwise_and(err, np.int32(7), out=err)
        return None
    return int(err.get()[0])


def refuse_errors(bits: int) -> None:
    """Raise for every fatal error bit, naming WRF's breakage."""
    fatal = [text for bit, text in ERRORS.items() if bits & bit]
    if fatal:
        raise RuntimeError("SSiB refused the step: " + "; ".join(fatal))


def replay_columns(columns_bin: Path, *, fixes: int = 0,
                   tables: np.ndarray | None = None):
    """Run a column-oracle input file; returns (nstep, ncol, len(STATE))."""
    import cupy as cp

    raw = Path(columns_bin).read_bytes()
    ncol, nstep, nz = (int(v) for v in np.frombuffer(raw[:12], "<i4"))
    off = 12
    nc, ns, nf = len(STATIC), len(STATE), len(FORCING) + nz

    def take(n):
        nonlocal off
        a = np.frombuffer(raw, "<f4", n, off)
        off += 4 * n
        return a

    static = cp.asarray(np.ascontiguousarray(
        take(ncol * nc).reshape(ncol, nc).T), cp.float32)
    state = cp.asarray(np.ascontiguousarray(
        take(ncol * ns).reshape(ncol, ns).T), cp.float32)
    forcing = take(nstep * ncol * nf).reshape(nstep, ncol, nf)
    out = np.empty((nstep, ncol, ns), np.float32)
    errors = 0
    for s in range(nstep):
        frc = cp.asarray(np.ascontiguousarray(forcing[s].T), cp.float32)
        errors |= step(static, state, frc, nz=nz, itimestep=s + 1,
                       fixes=fixes, tables=tables)
        out[s] = cp.asnumpy(state).T
    return out, errors
