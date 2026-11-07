"""NTU multi-moment microphysics, WRF mp_physics=56.

The column transcription follows the pinned public WRF 4.7.1 module without
its recovered prototype's F1-F6 physics changes. Native INIT_AEROSOL's
unfilled top density is completed analytically in the cold-start driver.
A real native capture witnesses LARGE_DT's undefined DNGRM read; see the
source-pinned oracle receipts before interpreting that wet-graupel residue.

CUDA runs each column, including native substeps, activation, all moments,
ice shape/density, and sedimentation. The default uses CUDA transcendental
math; strict numerics selects the host-word proof vocabulary. Fixed native
AERO_CONST tables are uploaded once, with source/licence and byte hashes.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

#: Registry.EM_COMMON ntu package, scalar order after the moist masses.
#: (WRF name, engine attribute).  ``as`` is a Python keyword, so the snow
#: second moment lives at ``state.as_``.
NTU_SCALARS = (
    ("qnc", "nc"), ("qnr", "nr"), ("qni", "ni"), ("qns", "ns"),
    ("qng", "ng"), ("qnh", "nh"), ("qdcn", "qdcn"), ("qtcn", "qtcn"),
    ("qccn", "qccn"), ("qrcn", "qrcn"), ("qnin", "qnin"), ("fi", "fi"),
    ("fs", "fs"), ("vi", "vi"), ("vs", "vs"), ("vg", "vg"), ("ai", "ai"),
    ("as", "as_"), ("ag", "ag"), ("ah", "ah"), ("i3m", "i3m"),
)
NTU_MASS_SPECIES = ("qi", "qs", "qg", "qh")
NTU_SCALAR_ATTRS = tuple(attr for _, attr in NTU_SCALARS)
#: Every transported species beyond qv/qc/qr, in Registry order.
NTU_SPECIES = NTU_MASS_SPECIES + NTU_SCALAR_ATTRS
#: The five aerosol tracers NTU_INIT fills on a cold start.
NTU_AEROSOL = ("qdcn", "qtcn", "qccn", "qrcn", "qnin")

#: Kernel field-table order (ntu.cu NtuField3); W is last.
_FIELD3 = ("theta", "p", "dz", "pii", "qv", "qc", "qr", "qi", "qs", "qg",
           "qh", "nc", "nr", "ni", "ns", "ng", "nh", "qdcn", "qtcn", "qccn",
           "qrcn", "qnin", "fi", "fs", "vi", "vs", "vg", "ai", "as_", "ag",
           "ah", "i3m", "w")
_FIELD2 = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
           "graupelncv", "hailnc", "hailncv", "sr")

#: Column array tiers: the kernel's per-column work arrays are sized at
#: compile time.  A deeper column would overrun them, so it is refused.
KMAX_TIERS = (64, 128, 256)
VERTICAL_LEVEL_BOUNDS = (2, KMAX_TIERS[-1])
#: WRF Registry rconfig CCNTY default (Registry.EM_COMMON:2429).
CCNTY_DEFAULT = 2
CCNTY_CHOICES = (1, 2, 3, 4)
# Full optimization is the fast default. The recovered fast-compile mode
# increased resident stage time substantially; strict proof controls contraction
# independently of optimization.
MODULE_OPTIONS = ("-std=c++17", "--ftz=false")
_TPB = 64


def kmax_for(nz: int) -> int:
    for tier in KMAX_TIERS:
        if nz <= tier:
            return tier
    raise ValueError(
        f"mp_physics=56: nz={nz} exceeds the NTU column arrays "
        f"({KMAX_TIERS[-1]} levels); a deeper column would overrun them")


def module_source(kmax: int, kernel_dir=None) -> str:
    from woof.core.kernels import module_source_int_defines
    from pathlib import Path
    kw = {} if kernel_dir is None else {"kernel_dir": kernel_dir}
    source = module_source_int_defines("ntu", (("NTU_KMAX", int(kmax)),), **kw)
    directory = Path(kernel_dir) if kernel_dir is not None else Path(__file__).with_name("kernels")
    source += "\n" + (directory / "ntu_parallel.cuh").read_text(encoding="utf-8")
    # This is an NVRTC-only source. Remove its dead host include so the
    # direct image cache can key a self-contained compilation.
    source = source.replace("#if !defined(__CUDACC_RTC__)\n#include <stdint.h>\n#else\n", "").replace(
        "typedef int int32_t;\n#endif", "typedef int int32_t;")
    from woof import numerics
    if numerics.mode() != "strict":
        source = source.replace("#define NTU_EXPF(x)  gfk_exp(x)", "#define NTU_EXPF(x) expf(x)")
        source = source.replace("#define NTU_LOGF(x)  gfk_log(x)", "#define NTU_LOGF(x) logf(x)")
        source = source.replace("#define NTU_POWF(x, y) gfk_pow((x), (y))", "#define NTU_POWF(x, y) powf((x), (y))")
        source = source.replace("#define NTU_EXP(x)   glibc_exp(x)", "#define NTU_EXP(x) exp(x)")
        source = source.replace("#define NTU_LOG(x)   glibc_log(x)", "#define NTU_LOG(x) log(x)")
        source = source.replace("#define NTU_ERFF(x)   ntu_erff(x)", "#define NTU_ERFF(x) erff(x)")
        source = source.replace("#define NTU_ERF(x)    ntu_erf(x)", "#define NTU_ERF(x) erf(x)")
        source = source.replace("#define NTU_LOG10F(x) ntu_log10(x)", "#define NTU_LOG10F(x) log10f(x)")
        source = source.replace("#define NTU_COSF(x)   glibc_cosf(x)", "#define NTU_COSF(x) cosf(x)")
        source = source.replace("#define NTU_SINF(x)   glibc_sinf(x)", "#define NTU_SINF(x) sinf(x)")
    return source


def compile_options() -> tuple[str, ...]:
    """The NVRTC options of the active arm.

    The strict arm's options are spelled out here, not only applied by the
    GPUWM_WRF_EXACT hook at the NVRTC boundary, because the direct-route
    image cache keys on the options it is handed: without them a strict
    image and a default image of the same source would share one key.
    """
    from woof import wrf_exact
    if wrf_exact.ENABLED:
        return wrf_exact.effective_options(MODULE_OPTIONS)
    return MODULE_OPTIONS


def compile_image(kmax: int, arch=None):
    """Compile (or fetch from the direct-route cache) one KMAX tier."""
    from woof.nvrtc_ptx_cache import compile_using_nvrtc
    image, _ = compile_using_nvrtc(module_source(kmax), compile_options(),
                                   arch, "ntu.cu")
    from woof.certify.kernel_manifest import record_module
    record_module(f"woof.core.ntu:ntu[NTU_KMAX={kmax}]",
                  source=module_source(kmax), options=compile_options(), module=None)
    return image


@lru_cache(maxsize=None)
def _module(device: int, kmax: int):
    import cupy as cp
    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile
    key = f"woof.core.ntu:ntu[NTU_KMAX={kmax}]"
    with cp.cuda.Device(device), observe_module_compile(key):
        image = compile_image(kmax)
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _kernel(name: str, kmax: int):
    import cupy as cp
    return _module(cp.cuda.Device().id, kmax).get_function(name)


@lru_cache(maxsize=None)
def aero_constants(device: int, ccnty: int):
    """Upload the pinned native AERO_CONST state once for this device/CCNTY."""
    import cupy as cp
    if ccnty not in CCNTY_CHOICES:
        raise ValueError(f"ccnty={ccnty} names no NTU aerosol background "
                         f"(module_mp_ntu.F AERO_CONST takes {CCNTY_CHOICES})")
    import hashlib
    import json
    from pathlib import Path
    bank = json.loads((Path(__file__).parents[1] / "data" / "ntu_aero_v471.json").read_text(encoding="utf-8"))
    row = bank["ccnty"][str(ccnty)]
    raw = bytes.fromhex(row["bytes_hex"])
    if len(raw) != 800 or hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise RuntimeError("NTU native aerosol constants have invalid bytes or digest")
    with cp.cuda.Device(device):
        buf = cp.asarray(np.frombuffer(raw, dtype=np.uint8))
    return buf


@lru_cache(maxsize=256)
def _pointer_table(device, pointers):
    import cupy as cp
    with cp.cuda.Device(device):
        return cp.asarray(np.array(pointers, dtype=np.uint64))


def _ptr_table(arrays):
    import cupy as cp
    # A prepared state keeps its pointers. Upload its table once, rather
    # than putting pageable host-to-device pointer copies in every step.
    return _pointer_table(cp.cuda.Device().id, tuple(a.data.ptr for a in arrays))


def _require(name, a, shape):
    if a.shape != shape or a.dtype != np.float32 or not a.flags.c_contiguous:
        raise ValueError(f"mp_physics=56 field {name} must be contiguous "
                         f"float32 {shape}, got {a.dtype} {a.shape}")


def launch_init(phb, ph, p, pb, alt, qv, aerosol, *, ccnty: int,
                restart: bool) -> None:
    """NTU_INIT over every column: fills the five aerosol tracers in place
    (left unchanged on a restart, as WRF).  ``phb``/``ph`` are (nz+1, ny, nx)
    geopotential on w levels; ``p``/``pb`` are the pressure pair WRF sums;
    ``alt`` is the inverse dry density."""
    import cupy as cp
    nz, ny, nx = p.shape
    kmax = kmax_for(nz)
    for name, a in (("p", p), ("pb", pb), ("alt", alt), ("qv", qv)):
        _require(name, a, (nz, ny, nx))
    for name, a in (("phb", phb), ("ph", ph)):
        _require(name, a, (nz + 1, ny, nx))
    if len(aerosol) != 5:
        raise ValueError("NTU_INIT takes qdcn, qtcn, qccn, qrcn, qnin")
    for name, a in zip(NTU_AEROSOL, aerosol):
        _require(name, a, (nz, ny, nx))
    ac = aero_constants(cp.cuda.Device().id, int(ccnty))
    status = cp.zeros(1, dtype=cp.uint32)
    table = _ptr_table(aerosol)
    ncol = ny * nx
    _kernel("ntu_init_kernel", kmax)(
        ((ncol + _TPB - 1) // _TPB,), (_TPB,),
        (ac, phb, ph, p, pb, alt, qv, table, np.int64(ncol), np.int32(nz),
         np.int32(1 if restart else 0), status))


def launch(fields: dict, surface: dict, dt: float, *, ccnty: int, workspace=None) -> None:
    """One MP_NTU call over every column, fields updated in place.

    ``fields`` maps the names in ``_FIELD3`` to (nz, ny, nx) arrays (``w``
    is (nz+1, ny, nx)); ``surface`` maps ``_FIELD2`` to (ny, nx) arrays.
    ``theta`` and every moment are INOUT; p, dz, pii and w are read-only.
    """
    import cupy as cp
    nz, ny, nx = fields["theta"].shape
    kmax = kmax_for(nz)
    if nz < VERTICAL_LEVEL_BOUNDS[0]:
        raise ValueError("mp_physics=56 needs at least two levels")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"mp_physics=56 dt must be positive, got {dt}")
    arrays = []
    for name in _FIELD3:
        a = fields[name]
        _require(name, a, (nz + 1 if name == "w" else nz, ny, nx))
        arrays.append(a)
    sarrays = []
    for name in _FIELD2:
        a = surface[name]
        _require(name, a, (ny, nx))
        sarrays.append(a)
    ac = aero_constants(cp.cuda.Device().id, int(ccnty))
    t3, t2 = _ptr_table(arrays), _ptr_table(sarrays)
    ncol = ny * nx
    if workspace is None:
        workspace = cp.empty((WORK_VOLUMES, nz, ny, nx), dtype=cp.float32)
    _require("workspace", workspace, (WORK_VOLUMES, nz, ny, nx))
    launch_stages(_module(cp.cuda.Device().id, kmax), ac, t3, t2,
                  workspace, ncol, nz, dt)


WORK_VOLUMES = 28


def launch_stages(module, constants, fields, surface, workspace, ncol, nz, dt):
    """The production layout; also used by the native word replay."""
    cell_args = (constants, fields, workspace, np.int64(ncol), np.int32(nz), np.float32(dt))
    cell_grid = ((ncol*nz+127)//128,)
    module.get_function("ntu_level_pre_kernel")(cell_grid, (128,), cell_args)
    module.get_function("ntu_column_flux_kernel")(
        ((ncol+127)//128,), (128,),
        (fields, surface, workspace, np.int64(ncol), np.int32(nz), np.float32(dt)))
    module.get_function("ntu_level_post_kernel")(cell_grid, (128,), cell_args)


#: The precipitation accumulators WRF's NTU driver arm binds
#: (module_microphysics_driver.F:1301-1316), as woof's canonical slots.
PRECIPITATION_SLOTS = (
    "mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
    "mp_graupelnc", "mp_graupelncv", "mp_hailnc", "mp_hailncv", "mp_sr",
)
#: The adapter's per-call preparation volumes (write-before-read).
PREP_SLOTS = ("ntu_theta", "ntu_pii", "ntu_dz", "ntu_z8w", "ntu_workspace")


def ccnty_of(cfg) -> int:
    return int(getattr(cfg, "ntu_ccnty", CCNTY_DEFAULT))


def apply(state, cfg, dt: float, *, refl_10cm_due: bool = False):
    """Prepare WRF's driver inputs and apply NTU to ``state`` in place.

    module_microphysics_driver.F:1301-1316 hands MP_NTU the full potential
    temperature, the full pressure, dz8w, the staggered w, pi_phy and every
    moment; the scheme returns theta and the moments and accumulates the
    rain, snow, graupel and hail totals.  WRF's NTU arm writes no
    REFL_10CM; the reflectivity diagnostic (:func:`reflectivity`) is
    WOOF's, from the scheme's own size distributions.
    """
    import cupy as cp
    from woof.core import constants as c
    from woof.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         save_pre_mp_theta)
    from woof.core.state import DTYPE
    nz, ny, nx = state.p.shape
    missing = [name for name in NTU_SPECIES
               if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=56 state lacks NTU fields: "
                         + ", ".join(missing))
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    theta = state.scratch((nz, ny, nx), "ntu_theta")
    pii = state.scratch((nz, ny, nx), "ntu_pii")
    dz = state.scratch((nz, ny, nx), "ntu_dz")
    z8w = state.scratch((nz + 1, ny, nx), "ntu_z8w")
    theta[...] = thb + state.thp
    pii[...] = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    # phy_prep: z_at_w = (phb+ph)/g, dz8w = z_at_w(k+1) - z_at_w(k).
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    cp.subtract(z8w[1:], z8w[:-1], out=dz)
    surface = (ny, nx)
    slots = {name: state.scratch(surface, name) for name in PRECIPITATION_SLOTS}
    fields = {"theta": theta, "p": state.p, "dz": dz, "pii": pii,
              "w": state.w}
    for name in ("qv", "qc", "qr") + NTU_SPECIES:
        fields[name] = getattr(state, name)
    save_pre_mp_theta(state)
    launch(fields,
           {"rainnc": slots["mp_rainnc"], "rainncv": slots["mp_rainncv"],
            "snownc": slots["mp_snownc"], "snowncv": slots["mp_snowncv"],
            "graupelnc": slots["mp_graupelnc"],
            "graupelncv": slots["mp_graupelncv"],
            "hailnc": slots["mp_hailnc"], "hailncv": slots["mp_hailncv"],
            "sr": slots["mp_sr"]},
           dt, ccnty=ccnty_of(cfg),
           workspace=state.scratch((WORK_VOLUMES, nz, ny, nx), "ntu_workspace"))
    if refl_10cm_due:
        from woof.core.refl import stash_refl_10cm
        refl = reflectivity(state, theta=theta, pii=pii, cfg=cfg)
        stash_refl_10cm(state, refl)
    moist_physics_finish(state, cfg, theta, dt)
    return MicrophysicsDiagnostics(
        rainnc=slots["mp_rainnc"], rainncv=slots["mp_rainncv"],
        sr=slots["mp_sr"], snownc=slots["mp_snownc"],
        snowncv=slots["mp_snowncv"], graupelnc=slots["mp_graupelnc"],
        graupelncv=slots["mp_graupelncv"], hailnc=slots["mp_hailnc"],
        hailncv=slots["mp_hailncv"])


def reflectivity(state, *, temperature=None, pressure=None, theta=None,
                 pii=None, cfg=None):
    """10 cm reflectivity (dBZ) from NTU's own predicted distributions.

    WOOF's diagnostic: WRF's NTU arm writes no REFL_10CM.  Each category's
    gamma distribution is rebuilt by the scheme's own SOLVE_AFA* routines on
    copies of the moments and contributes its Rayleigh sixth moment; frozen
    categories carry the ice dielectric factor and their predicted density.
    Returns the state's ``refl_10cm`` scratch slot.
    """
    import cupy as cp
    from woof.core import constants as c
    from woof.core.state import DTYPE
    nz, ny, nx = state.p.shape
    if theta is None or pii is None:
        thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
        theta = state.scratch((nz, ny, nx), "ntu_theta")
        pii = state.scratch((nz, ny, nx), "ntu_pii")
        theta[...] = thb + state.thp
        pii[...] = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    refl = state.scratch((nz, ny, nx), "refl_10cm")
    arrays = {"theta": theta, "p": state.p, "dz": theta, "pii": pii,
              "w": state.w}
    for name in ("qv", "qc", "qr") + NTU_SPECIES:
        arrays[name] = getattr(state, name)
    table = _ptr_table([arrays[name] for name in _FIELD3])
    ccnty = ccnty_of(cfg) if cfg is not None else CCNTY_DEFAULT
    ac = aero_constants(cp.cuda.Device().id, ccnty)
    ncell = nz * ny * nx
    _kernel("ntu_refl_kernel", kmax_for(nz))(
        ((ncell + 127) // 128,), (128,), (ac, table, np.int64(ncell), refl))
    return refl


def init_state(state, cfg, *, restart: bool) -> dict:
    """NTU_INIT for a domain: the scheme's own aerosol climatology
    (AERO_CONST + INIT_AEROSOL) on a cold start, nothing on a restart."""
    import cupy as cp
    if restart:
        return {"ntu_aerosol": "restart: carried fields kept (WRF skips "
                               "INIT_AEROSOL when restart)"}
    nz, ny, nx = state.p.shape
    phb = state.phb if state.phb.ndim == 3 else cp.broadcast_to(
        state.phb[:, None, None], (nz + 1, ny, nx))
    phb = cp.ascontiguousarray(phb)
    alt = state.alt
    zeros = cp.zeros((nz, ny, nx), dtype=cp.float32)
    launch_init(phb, cp.ascontiguousarray(state.php), state.p, zeros,
                alt, state.qv, [getattr(state, n) for n in NTU_AEROSOL],
                ccnty=ccnty_of(cfg), restart=False)
    return {"ntu_aerosol": f"INIT_AEROSOL climatology, ccnty={ccnty_of(cfg)}"}


def frame_module():
    """Measure the widest admitted depth image, including the staged driver.

    Production selects 64, 128 or 256 levels. The current-platform depth
    sweep reads 2344, 4648 and 9256 bytes in both modes on sm_120. Measuring
    the deepest tier avoids recording only the inactive column wrapper.
    """
    import cupy as cp
    kmax = KMAX_TIERS[-1]
    return _module(cp.cuda.Device().id, kmax), module_source(kmax)


__all__ = ["NTU_SCALARS", "NTU_SPECIES", "NTU_MASS_SPECIES",
           "NTU_SCALAR_ATTRS", "NTU_AEROSOL", "KMAX_TIERS",
           "VERTICAL_LEVEL_BOUNDS", "CCNTY_DEFAULT", "kmax_for",
           "launch", "launch_init", "aero_constants", "module_source",
           "apply", "init_state", "reflectivity", "PRECIPITATION_SLOTS",
           "PREP_SLOTS"]
