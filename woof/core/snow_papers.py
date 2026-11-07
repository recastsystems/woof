"""Snow albedo written from papers, not bitwise to WRF.

Host code orchestrates only. Rust validates and decodes the generated Mie
asset; CUDA advances the grain state and solves spectral transfer.
"""
from functools import lru_cache
from pathlib import Path
import ctypes
import numpy as np

TABLE_SHA256 = "0e0a377955e78f6762d6d1a3318d9715fc8fffbd213b9645be88523e1fa2bce8"
TABLE_WORDS = 283703
STATE_FIELD = "snicar_statexy"
STATE_WORDS = 21
LABEL = "written from papers, not bitwise to WRF"


def identity():
    return {"algorithm": "snow-papers-layered-ad-shapes-ice-v2", "status": LABEL,
            "table_sha256": TABLE_SHA256, "state": "three layers: mass, Re, liquid mass, BC external/internal, dust external/internal",
            "spectrum": "NLR ASTM G173 direct/diffuse; 480 equal 0.01um bins, 0.2-5um",
            "habits": ["sphere", "spheroid", "hexagonal plate", "Koch snowflake"],
            "impurity": "Bond/Bergstrom BC and Zhang 2024 coarse-mode mineral RI; He coalbedo fits",
            "aging": "regenerated Flanner/Zender microphysics plus Brun wet growth"}


def settings(cfg):
    return {name: getattr(cfg, name) for name in
            ("snicar_snowshape_opt", "snicar_snowbc_intmix", "snicar_snowdust_intmix",
             "snicar_bc_ppb", "snicar_dust_ppm", "snicar_ice_density", "snicar_bubble_radius_um")}


@lru_cache(maxsize=1)
def _host_table():
    from woof.static.rust_bridge import load, _check
    library = load()
    try:
        reader = library.gpuwm_snow_optics_load_v1
    except AttributeError:
        raise RuntimeError("snow albedo requires rebuilt Rust static-fields bridge") from None
    values = np.empty(TABLE_WORDS, dtype=np.float32)
    ptr = ctypes.POINTER(ctypes.c_float)
    reader.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ptr, ctypes.c_size_t]
    reader.restype = ctypes.c_int32
    path = Path(__file__).resolve().parents[1]/"data"/"snow_papers"/"optics.bin"
    code = reader(str(path).encode(), TABLE_SHA256.encode(), values.ctypes.data_as(ptr), values.size)
    _check(library, code, "generated snow optics")
    values.flags.writeable = False
    return values


@lru_cache(maxsize=8)
def _device_table(device):
    import cupy as cp
    with cp.cuda.Device(device):
        return cp.asarray(_host_table())


@lru_cache(maxsize=8)
def _module(device):
    import cupy as cp
    from woof.core.noahmp_kernel_sources import _compile_loaded_module
    source = (Path(__file__).with_name("kernels")/"snow_papers.cu").read_text()
    with cp.cuda.Device(device):
        return _compile_loaded_module(source, ("-std=c++17", "--ftz=false"), "snow_papers")


def evaluate(inputs, ist, *, carbon=None, layers=None, state=None, options=None,
             glacier=None, ice_layers=None, air_temperature=None,
             return_state=False, return_fluxes=False):
    import cupy as cp
    n = len(inputs)
    if inputs.dtype != cp.float32 or inputs.shape != (n, 56) or not inputs.flags.c_contiguous:
        raise ValueError("snow albedo requires contiguous f32[n,56]")
    if ist.dtype != cp.int32 or ist.shape != (n,) or not ist.flags.c_contiguous:
        raise ValueError("snow albedo requires contiguous i32[n] surface classes")
    opt = dict(snicar_snowshape_opt=3, snicar_snowbc_intmix=True,
               snicar_snowdust_intmix=False, snicar_bc_ppb=0., snicar_dust_ppm=0.,
               snicar_ice_density=900., snicar_bubble_radius_um=1000.)
    opt.update(options or {})
    shape = opt["snicar_snowshape_opt"]
    if type(shape) is not int or not 1 <= shape <= 4:
        raise ValueError("snicar_snowshape_opt requires 1 sphere, 2 spheroid, 3 hexagonal plate, 4 Koch")
    if opt["snicar_snowbc_intmix"] and opt["snicar_snowdust_intmix"]:
        raise ValueError("He fits do not support simultaneous BC and dust internal mixing")
    if layers is None:
        layers = cp.zeros((n, 3, 4), dtype=cp.float32)
        layers[:, 2, 0] = inputs[:, 34]
        layers[:, 2, 2] = inputs[:, 30]
        layers[:, :, 3] = inputs[:, 28, None]
    if state is None:
        state = cp.zeros((n, STATE_WORDS), dtype=cp.float32)
    for value, expected in ((layers,(n,3,4)), (state,(n,STATE_WORDS))):
        if value.dtype != cp.float32 or value.shape != expected:
            raise ValueError("SNICAR layer/state ABI differs")
    forcing = cp.empty((n,6),dtype=cp.float32)
    bc = opt["snicar_bc_ppb"]
    dust = opt["snicar_dust_ppm"]
    if not np.isfinite(bc) or not 0 <= bc <= 10000 or not np.isfinite(dust) or not 0 <= dust <= 1000:
        raise ValueError("He concentration fits require BC 0–10000 ppb, dust 0–1000 ppm")
    if carbon is not None:
        if carbon.shape != (n,) or carbon.dtype != cp.float32:
            raise ValueError("snow impurity requires f32[n] kg/kg")
    build_ice = ice_layers is None
    rho, radius = opt["snicar_ice_density"], opt["snicar_bubble_radius_um"]
    if ice_layers is None:
        if not np.isfinite(rho) or not 650 <= rho <= 916.999 or not np.isfinite(radius) or not 30 <= radius <= 20000:
            raise ValueError("glacier optics require density 650–916.999 kg/m3 and bubble radius 30–20000 um")
        ice_layers = cp.empty((n,4,3),dtype=cp.float32)
    if ice_layers.shape != (n,4,3) or ice_layers.dtype != cp.float32:
        raise ValueError("glacier layer ABI requires f32[n,4,3]: depth, density, bubble radius")
    out = cp.empty((n,6),dtype=cp.float32)
    carried = cp.empty((n,STATE_WORDS),dtype=cp.float32)
    absorption = cp.empty((n,7,4),dtype=cp.float32)
    if n:
        device=cp.cuda.Device().id
        module = _module(device)
        # Assemble independent packet words together; the transfer solver and
        # its strict operation order are unchanged. No per-call host readback.
        pointer = lambda value: cp.ascontiguousarray(value, dtype=cp.float32) if value is not None else np.uint64(0)
        if air_temperature is not None:
            air_temperature = cp.broadcast_to(cp.asarray(air_temperature, dtype=cp.float32), (n,))
        module.get_function("snow_forcing_pack")(((n+255)//256,),(256,),
            (inputs, pointer(air_temperature), pointer(carbon),
             pointer(glacier.reshape(n) if glacier is not None else None),
             forcing, ice_layers if build_ice else np.uint64(0), np.int32(n),
             np.int32(bool(opt["snicar_snowbc_intmix"])),
             np.int32(2+bool(opt["snicar_snowdust_intmix"])),
             np.float32(bc*1e-9), np.float32(dust*1e-6),
             np.float32(rho if build_ice else 0), np.float32(radius if build_ice else 0)))
        module.get_function("snow_papers_albedo")((n,),(64,),
            (inputs,ist,cp.ascontiguousarray(layers),cp.ascontiguousarray(state),forcing,
             _device_table(device),cp.ascontiguousarray(ice_layers),out,carried,
             absorption,np.int32(n),np.int32(shape-1)))
    if return_fluxes:
        return out,carried,absorption
    return (out,carried) if return_state else out


def refreeze(carried, ice_before, ice_after, liquid_after):
    """Resolved phase-change increment; Lin 2025 section 2b2, 1000 um ice."""
    import cupy as cp
    s=carried.reshape(-1,3,7)
    frozen=cp.maximum(ice_after-ice_before,cp.float32(0))
    old=cp.maximum(ice_before,cp.float32(0))
    s[:,:,1]=cp.where(old+frozen>0,
        (old*s[:,:,1]+cp.float32(1000)*frozen)/cp.maximum(old+frozen,cp.float32(1e-20)),s[:,:,1])
    s[:,:,2]=liquid_after
