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
# Ice refractive-index sets (snicar_ice_ri_opt) and dust optical sets
# (snicar_dust_optics_opt): re-implemented from published descriptions,
# not bitwise to upstream. docs/SNOW-PAPERS.md names each paper and table.
ICE_TABLES = {1: ("optics-ice-warren1984.bin", "3b2addc46200f28e73fd8c8a7bdb6661633dfb51fc7c883cbcad64c1354e07da"),
              2: ("optics.bin", TABLE_SHA256),
              3: ("optics-ice-picard2016.bin", "03467d49b19c79cc9341daef96a6d5dd6daaa11dbdaeeb71fe52fc566d783c01")}
ICE_SETS = {1: "Warren 1984", 2: "Warren and Brandt 2008",
            3: "Picard 2016 absorption on Warren and Brandt 2008"}
DUST_ROWS_FILE, DUST_ROWS_SHA256 = "dust-rows.bin", "6b9b928a8f3ba719d5694976c1cc3e020a77c99a09fb734ceb6a95fdad2df792"
DUST_SETS = {0: "Zhang 2024 coarse-mode mean", 1: "Saharan, Balkanski 2007 central hematite",
             2: "Colorado, Skiles 2017", 3: "Greenland, Polashenski 2015 central"}
DUST_WORDS = 4*480*3
# The in-table dust row: OP + (4 habits + bubbles + BC)*NR... see snow_papers.cu.
DUST_ROW_START = 1504 + (5*32+1)*480*3
DEFAULT_ICE_RI, DEFAULT_DUST_OPTICS = 2, 0
# clean-snicar-layers solver choice (re-implemented from published
# descriptions, not bitwise to upstream). The default keeps adding-doubling;
# the band count is spectra's snicar_bandnumber_opt (1 five bands, 2 480).
SOLVERS = {1: "adding-doubling (Dang et al. 2019; Flanner et al. 2021 App. A)",
           2: "Toon et al. 1989 two-stream (Eddington VIS, hemispheric mean NIR)"}
LAYERS_LABEL = "re-implemented from published descriptions, not bitwise to upstream"


def identity(options=None):
    base = _identity()
    selected = _layer_choices(options)
    if selected:
        base = dict(base, **selected, layers_status=LAYERS_LABEL)
    return base


def _layer_choices(options):
    """Only non-default choices, so default identities keep their bytes."""
    options = options or {}
    out = {}
    solver = options.get("snicar_solver_opt", 1)
    if solver != 1:
        out["solver"] = SOLVERS.get(solver, solver)
    if out:
        out["glacier_columns"] = "480-bin adding-doubling (Fresnel ice interface)"
    return out


def _identity():
    return {"algorithm": "snow-papers-layered-ad-shapes-ice-v2", "status": LABEL,
            "table_sha256": TABLE_SHA256, "state": "three layers: mass, Re, liquid mass, BC external/internal, dust external/internal",
            "spectrum": "NLR ASTM G173 direct/diffuse; 480 equal 0.01um bins, 0.2-5um",
            "habits": ["sphere", "spheroid", "hexagonal plate", "Koch snowflake"],
            "impurity": "Bond/Bergstrom BC and Zhang 2024 coarse-mode mineral RI; He coalbedo fits",
            "aging": "regenerated Flanner/Zender microphysics plus Brun wet growth "
                     "(CLM5 tech note Eq. 2.3.75 volume form, 2026-11-09)",
            "fresh_grain": "1: 54.526-204.526 um; 2: Lin 2025 MODSCAG 33-91 um, "
                           "re-implemented from published descriptions, not bitwise to upstream"}


#: Incident-spectrum and band-count options (Lin et al. 2025 JHM; Lin, RAL
#: talk 2024, namelist slide). Re-implemented from published descriptions,
#: not bitwise to upstream. Spectrum 0 with 480 bands is this port's
#: packaged ASTM G173 path; those are the defaults and are left out of every
#: identity so default runs keep their bytes.
SPECTRUM_LABEL = "re-implemented from published descriptions, not bitwise to upstream"
SPECTRA = (None, "mid-latitude winter", "mid-latitude summer", "sub-Arctic winter",
           "sub-Arctic summer", "Summit Greenland summer", "high mountain summer")
SPECTRA_SHA256 = "2051bfa2da01c889cfe5b83ea723a8e265698a74699cf4c3a66093da7e2f5e0c"
SPECTRUM_DEFAULTS = {"snicar_solarspec_opt": 0, "snicar_bandnumber_opt": 2}
#: Five-band edges in micrometres (Flanner et al. 2007 JGR s2; CLM5
#: technical note): 0.3-0.7, 0.7-1.0, 1.0-1.2, 1.2-1.5, 1.5-5.0. Our grid
#: starts at 0.2 um, so band 1 also holds the weightless 0.2-0.3 um bins.
FIVE_BAND_EDGES_UM = (0.2, 0.7, 1.0, 1.2, 1.5, 5.0)
PACKET_STRIDE = 1 + 480*2


def settings(cfg):
    out = {name: getattr(cfg, name) for name in
           ("snicar_snowshape_opt", "snicar_snowbc_intmix", "snicar_snowdust_intmix",
            "snicar_bc_ppb", "snicar_dust_ppm", "snicar_ice_density", "snicar_bubble_radius_um")}
    # The fresh-grain, optical-set and spectrum controls enter only when
    # selected, so default option-3 settings, receipts and identities keep
    # their earlier bytes.
    for name, default in (("snicar_fresh_grain_opt", 1), ("snicar_ice_ri_opt", DEFAULT_ICE_RI),
                          ("snicar_dust_optics_opt", DEFAULT_DUST_OPTICS)):
        value = getattr(cfg, name, default)
        if value != default:
            out[name] = value
    for name, default in SPECTRUM_DEFAULTS.items():
        value = getattr(cfg, name, default)
        if value != default:
            out[name] = value
    # Non-default only: default opt_alb=3 runs keep their identity bytes.
    if getattr(cfg, "snicar_solver_opt", 1) != 1:
        out["snicar_solver_opt"] = cfg.snicar_solver_opt
    return out


@lru_cache(maxsize=1)
def _host_spectra():
    """(mu nodes f32[nmu], spectra f32[6,nmu,480,2]) from the packaged asset."""
    import hashlib
    path = Path(__file__).resolve().parents[1]/"data"/"snow_papers"/"solar_spectra.bin"
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != SPECTRA_SHA256:
        raise RuntimeError(f"{path.name}: digest differs from the packaged incident spectra")
    if blob[:8] != b"GPWMSPC1":
        raise RuntimeError(f"{path.name}: not an incident-spectrum table")
    ncase, nmu, nw = (int(v) for v in np.frombuffer(blob, "<i4", 3, 8))
    words = np.frombuffer(blob, "<f4", offset=20)
    if (ncase, nw) != (6, 480) or words.size != nmu + ncase*nmu*nw*2:
        raise RuntimeError(f"{path.name}: layout differs from 6 spectra x 480 bins")
    mu = words[:nmu].copy()
    spectra = words[nmu:].reshape(ncase, nmu, nw, 2).copy()
    mu.flags.writeable = False
    spectra.flags.writeable = False
    return mu, spectra


def _band_index():
    centres = 0.205 + 0.01*np.arange(480)
    return np.searchsorted(np.asarray(FIVE_BAND_EDGES_UM[1:-1]), centres, side="right")


@lru_cache(maxsize=32)
def _device_spectrum(device, option, bands, ice=DEFAULT_ICE_RI, dust=DEFAULT_DUST_OPTICS):
    """Device packets for the kernel's ``spec`` argument.

    480 bands: one packet, mu nodes then [nmu][480][direct,diffuse].
    5 bands: (direct table, direct packet, diffuse table, diffuse packet).
    Each table holds flux-weighted band optics in bins 0-4 (weights at
    mu=0.5 of the chosen spectrum); each packet holds that beam's band
    totals per mu with the other beam zero. Built on the device once.
    """
    import cupy as cp
    with cp.cuda.Device(device):
        if option:
            mu, spectra = _host_spectra()
            mu = cp.asarray(mu, dtype=cp.float32)
            spec = cp.asarray(spectra[option-1], dtype=cp.float32)
        else:
            tab = _device_table(device, ice, dust)
            mu = cp.asarray([0.5], dtype=cp.float32)
            spec = cp.stack((tab[2*32+480:2*32+960], tab[2*32+960:2*32+1440]), axis=-1)[None]
        if bands == 2:
            return cp.ascontiguousarray(cp.concatenate((mu, spec.reshape(-1))))
        return _five_band(device, mu, spec, ice, dust)


def _five_band(device, mu, spec, ice=DEFAULT_ICE_RI, dust=DEFAULT_DUST_OPTICS):
    import cupy as cp
    NR, NW, OP = 32, 480, 1504
    FR = OP + (5*NR+2)*NW*3
    MX = FR + NW*4
    AG = MX + 32*NW*3
    band = cp.asarray(_band_index())
    onehot = (band[:, None] == cp.arange(5)[None]).astype(cp.float64)
    ref = int(cp.argmin(cp.abs(mu - 0.5)).item()) if mu.size > 1 else 0
    base = _device_table(device, ice, dust).astype(cp.float64)
    lam = base[2*NR:2*NR+NW].copy()
    centre = cp.asarray([0.5*(a+b) for a, b in zip(FIVE_BAND_EDGES_UM[:-1], FIVE_BAND_EDGES_UM[1:])])
    out = []
    for beam in (0, 1):
        w = spec[ref, :, beam].astype(cp.float64)
        mw = onehot*w[:, None]
        total = cp.maximum(mw.sum(axis=0), 1e-30)
        t = base.copy()
        o = t[OP:FR].reshape(5*NR+2, NW, 3)
        ext, absorb, g = o[..., 0].copy(), o[..., 1].copy(), o[..., 2].copy()
        sca = ext - absorb
        e5 = ext @ mw / total
        s5 = sca @ mw / total
        g5 = (sca*g) @ mw / cp.maximum(sca @ mw, 1e-30)
        o[:, :5, 0] = e5
        o[:, :5, 1] = e5 - s5
        o[:, :5, 2] = g5
        fr = t[FR:MX].reshape(NW, 4)
        k_over_lam = (fr[:, 1]/lam) @ mw / total
        fr5 = (fr.T @ mw / total).T
        fr[:5] = fr5
        fr[:5, 1] = k_over_lam*centre      # band mean of 4*pi*k/lambda
        mx = t[MX:AG].reshape(32, NW, 3)
        for s in range(3):
            mx[:, :5, s] = (mx[:, :, s] @ mw) / total[None]
        t[2*NR:2*NR+5] = centre
        packet = cp.zeros((mu.size, NW, 2), dtype=cp.float64)
        packet[:, :5, beam] = spec[:, :, beam].astype(cp.float64) @ onehot
        out += [cp.ascontiguousarray(t.astype(cp.float32)),
                cp.ascontiguousarray(cp.concatenate((mu, packet.astype(cp.float32).reshape(-1))))]
    return tuple(out)


@lru_cache(maxsize=32)
def _lap_five_band(device, option, ice=DEFAULT_ICE_RI, dust=DEFAULT_DUST_OPTICS):
    """LAP optics rows (OC, five dust bins) averaged over the five bands with
    the same direct and diffuse weights _five_band gives the snow table:
    flux-weighted extinction and scattering, scattering-weighted asymmetry.
    Returns (direct rows, diffuse rows), each f32[6, 480, 3] with bins 0-4
    holding the band means (the kernel reads only those bins in five-band
    mode).  Re-implemented from published descriptions, not bitwise to upstream."""
    import cupy as cp
    with cp.cuda.Device(device):
        if option:
            mu, spectra = _host_spectra()
            mu = cp.asarray(mu, dtype=cp.float32)
            spec = cp.asarray(spectra[option-1], dtype=cp.float32)
        else:
            tab = _device_table(device, ice, dust)
            mu = cp.asarray([0.5], dtype=cp.float32)
            spec = cp.stack((tab[2*32+480:2*32+960], tab[2*32+960:2*32+1440]), axis=-1)[None]
        band = cp.asarray(_band_index())
        onehot = (band[:, None] == cp.arange(5)[None]).astype(cp.float64)
        ref = int(cp.argmin(cp.abs(mu - 0.5)).item()) if mu.size > 1 else 0
        lapt, _scav = _lap_device(device)
        base = lapt.astype(cp.float64).reshape(-1, 480, 3)
        out = []
        for beam in (0, 1):
            w = spec[ref, :, beam].astype(cp.float64)
            mw = onehot*w[:, None]
            total = cp.maximum(mw.sum(axis=0), 1e-30)
            rows = base.copy()
            ext, absorb, g = rows[..., 0].copy(), rows[..., 1].copy(), rows[..., 2].copy()
            sca = ext - absorb
            e5 = ext @ mw / total
            s5 = sca @ mw / total
            g5 = (sca*g) @ mw / cp.maximum(sca @ mw, 1e-30)
            rows[:, :5, 0] = e5
            rows[:, :5, 1] = e5 - s5
            rows[:, :5, 2] = g5
            out.append(cp.ascontiguousarray(rows.astype(cp.float32).reshape(lapt.shape)))
        return tuple(out)


def _packet_nodes(spec):
    """Number of mu nodes in a ``spec`` packet: nmu + nmu*480*2 words."""
    nmu, rest = divmod(int(spec.size), PACKET_STRIDE)
    if rest or nmu < 1:
        raise ValueError("incident-spectrum packet must hold nmu + nmu*480*2 words")
    return nmu


@lru_cache(maxsize=3)
def _host_table(ice=DEFAULT_ICE_RI):
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
    name, digest = ICE_TABLES[ice]
    path = Path(__file__).resolve().parents[1]/"data"/"snow_papers"/name
    code = reader(str(path).encode(), digest.encode(), values.ctypes.data_as(ptr), values.size)
    _check(library, code, "generated snow optics")
    values.flags.writeable = False
    return values


@lru_cache(maxsize=1)
def _host_dust_rows():
    from woof.static.rust_bridge import load, _check
    library = load()
    try:
        reader = library.gpuwm_snow_dust_rows_load_v1
    except AttributeError:
        raise RuntimeError("snicar_dust_optics_opt requires rebuilt Rust static-fields bridge") from None
    values = np.empty(DUST_WORDS, dtype=np.float32)
    ptr = ctypes.POINTER(ctypes.c_float)
    reader.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ptr, ctypes.c_size_t]
    reader.restype = ctypes.c_int32
    path = Path(__file__).resolve().parents[1]/"data"/"snow_papers"/DUST_ROWS_FILE
    code = reader(str(path).encode(), DUST_ROWS_SHA256.encode(), values.ctypes.data_as(ptr), values.size)
    _check(library, code, "generated snow dust rows")
    values.flags.writeable = False
    return values


@lru_cache(maxsize=8)
def _device_table(device, ice=DEFAULT_ICE_RI, dust=DEFAULT_DUST_OPTICS):
    """One device table per (ice set, dust set), built once at first use.
    A non-default dust set replaces only the in-table dust row; the kernel
    and its layout are unchanged."""
    import cupy as cp
    host = _host_table(ice)
    if dust != DEFAULT_DUST_OPTICS:
        host = host.copy()
        width = 480*3
        host[DUST_ROW_START:DUST_ROW_START+width] = _host_dust_rows()[dust*width:(dust+1)*width]
    with cp.cuda.Device(device):
        return cp.asarray(host)


@lru_cache(maxsize=8)
def _module(device):
    import cupy as cp
    from woof.core.noahmp_kernel_sources import _compile_loaded_module
    source = (Path(__file__).with_name("kernels")/"snow_papers.cu").read_text()
    with cp.cuda.Device(device):
        return _compile_loaded_module(source, ("-std=c++17", "--ftz=false"), "snow_papers")


def evaluate(inputs, ist, *, carbon=None, layers=None, state=None, options=None,
             glacier=None, ice_layers=None, air_temperature=None,
             return_state=False, return_fluxes=False, lap=None, deposition=None):
    import cupy as cp
    n = len(inputs)
    if inputs.dtype != cp.float32 or inputs.shape != (n, 56) or not inputs.flags.c_contiguous:
        raise ValueError("snow albedo requires contiguous f32[n,56]")
    if ist.dtype != cp.int32 or ist.shape != (n,) or not ist.flags.c_contiguous:
        raise ValueError("snow albedo requires contiguous i32[n] surface classes")
    opt = dict(snicar_snowshape_opt=3, snicar_snowbc_intmix=True,
               snicar_snowdust_intmix=False, snicar_bc_ppb=0., snicar_dust_ppm=0.,
               snicar_ice_density=900., snicar_bubble_radius_um=1000.,
               snicar_fresh_grain_opt=1,
               snicar_ice_ri_opt=DEFAULT_ICE_RI, snicar_dust_optics_opt=DEFAULT_DUST_OPTICS, **SPECTRUM_DEFAULTS,
               snicar_solver_opt=1)
    opt.update(options or {})
    ice_set, dust_set = opt["snicar_ice_ri_opt"], opt["snicar_dust_optics_opt"]
    if type(ice_set) is not int or ice_set not in ICE_TABLES:
        raise ValueError("snicar_ice_ri_opt requires 1 Warren 1984, 2 Warren and Brandt 2008, 3 Picard 2016")
    if type(dust_set) is not int or dust_set not in DUST_SETS:
        raise ValueError("snicar_dust_optics_opt requires 0 Zhang 2024, 1 Saharan, 2 Colorado, 3 Greenland")
    opt.update(options or {})
    spectrum, bands = opt["snicar_solarspec_opt"], opt["snicar_bandnumber_opt"]
    if type(spectrum) is not int or not 0 <= spectrum <= 6:
        raise ValueError("snicar_solarspec_opt requires 0 (packaged G173) or 1-6 "
                         "(mid-lat winter/summer, sub-Arctic winter/summer, Summit, high mountain)")
    if type(bands) is not int or bands not in (1, 2):
        raise ValueError("snicar_bandnumber_opt requires 1 (5 bands) or 2 (480 bands)")
    solver = opt["snicar_solver_opt"]
    if type(solver) is not int or solver not in SOLVERS:
        raise ValueError("snicar_solver_opt requires 1 adding-doubling or 2 Toon two-stream")
    shape = opt["snicar_snowshape_opt"]
    if type(shape) is not int or not 1 <= shape <= 4:
        raise ValueError("snicar_snowshape_opt requires 1 sphere, 2 spheroid, 3 hexagonal plate, 4 Koch")
    fresh_opt = opt["snicar_fresh_grain_opt"]
    if type(fresh_opt) is not int or fresh_opt not in (1, 2):
        raise ValueError("snicar_fresh_grain_opt requires 1 (54.526-204.526 um) or 2 (Lin 2025 MODSCAG 33-91 um)")
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
        lap_new = None
        if lap is not None:
            # snicar_lap_opt > 0: the same shared spectral solver (snicar_spectral)
            # with the particle optics, so every choice below reaches it too.
            if lap.shape != (n, LAP_WORDS) or lap.dtype != cp.float32:
                raise ValueError("LAP state ABI requires f32[n,47]")
            if deposition.shape != (n, LAP_DEPOSITION_WORDS) or deposition.dtype != cp.float32:
                raise ValueError("LAP deposition ABI requires f32[n,27]: 9 dry, 9 wet (kg m-2 s-1), 9 initial (kg m-2)")
            lap_rows, scav = _lap_device(device)
            lap_module = _lap_module(device)
            lap_new = cp.empty((n,LAP_WORDS),dtype=cp.float32)
            lap_in, dep_in = cp.ascontiguousarray(lap), cp.ascontiguousarray(deposition)
        def launch(table, spec, o, a, lap_table=None):
            spec_arg = spec if spec is not None else np.uint64(0)
            nodes = np.int32(0 if spec is None else _packet_nodes(spec))
            if lap is not None:
                lap_module.get_function("snow_lap_albedo")((n,),(64,),
                    (inputs,ist,cp.ascontiguousarray(layers),cp.ascontiguousarray(state),forcing,
                     table,cp.ascontiguousarray(ice_layers),lap_table,lap_in,dep_in,scav,
                     o,carried,a,lap_new,np.int32(n),np.int32(shape-1),
                     np.int32(bool(opt["snicar_snowbc_intmix"])),np.int32(bool(opt["snicar_snowdust_intmix"])),
                     np.int32(lap_init_top(opt)),np.int32(fresh_opt-1),spec_arg,nodes,np.int32(solver)))
                return
            args = (inputs,ist,cp.ascontiguousarray(layers),cp.ascontiguousarray(state),forcing,
                    table,cp.ascontiguousarray(ice_layers),o,carried,
                    a,np.int32(n),np.int32(shape-1),np.int32(fresh_opt-1),spec_arg,nodes)
            if solver == 1:
                module.get_function("snow_papers_albedo")((n,),(64,),args)
            else:
                # Separate image (Toon solver): the default kernel keeps its own code.
                module.get_function("snow_papers_albedo_layers")((n,),(64,),args + (np.int32(solver),))
        if spectrum == 0 and bands == 2:
            launch(_device_table(device, ice_set, dust_set), None, out, absorption,
                   lap_rows if lap is not None else None)
        elif bands == 2:
            launch(_device_table(device, ice_set, dust_set), _device_spectrum(device, spectrum, 2, ice_set, dust_set),
                   out, absorption, lap_rows if lap is not None else None)
        else:
            # Five bands: the direct beam uses direct-weighted band optics and
            # the diffuse beam diffuse-weighted ones (Flanner 2007 s2): two
            # launches over five bins each, merged on the device.  The LAP rows
            # are band-averaged with the same weights (_lap_five_band).
            t_dir, s_dir, t_dif, s_dif = _device_spectrum(device, spectrum, 1, ice_set, dust_set)
            l_dir, l_dif = _lap_five_band(device, spectrum, ice_set, dust_set) if lap is not None else (None, None)
            launch(t_dir, s_dir, out, absorption, l_dir)
            out_dif = cp.empty_like(out)
            abs_dif = cp.empty_like(absorption)
            launch(t_dif, s_dif, out_dif, abs_dif, l_dif)
            out[:, 2:4] = out_dif[:, 2:4]
            absorption[:, :, 2:4] = abs_dif[:, :, 2:4]
            flux = inputs[:, 52:56].sum(axis=1)
            mixed = (inputs[:, 52]*out[:, 0] + inputs[:, 53]*out[:, 1]
                     + inputs[:, 54]*out[:, 2] + inputs[:, 55]*out[:, 3])/cp.where(flux > 0, flux, 1)
            out[:, 5] = cp.where(flux > 0, mixed, inputs[:, 41])
        if lap is not None:
            return (out, carried, absorption, lap_new) if return_fluxes else (out, carried, lap_new)
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


# ---------------------------------------------------------------------------
# clean-snicar-lap: light-absorbing particles (LAP) in snow, snicar_lap_opt.
# Re-implemented from published descriptions, not bitwise to upstream Noah-MP
# (provenance: docs/PROVENANCE-clean-snicar-lap.md). Off (0) is the default;
# off leaves every array, kernel module and identity above unchanged.
# 1: deposition fluxes from a table (the analogue of upstream's read-table
#    switch); 2: a deposition series produced by `python -m
#    woof.snow_lap_deposition` (Rust, all cores) from GEOS-FP or MERRA-2.
LAP_LABEL = "re-implemented from published descriptions, not bitwise to upstream"
LAP_FIELD = "snicar_lapxy"
LAP_WORDS = 47
LAP_DEPOSITION_FIELD = "snicar_lapdepxy"
LAP_DEPOSITION_WORDS = 27
LAP_SPECIES = ("bc_hydrophilic", "bc_hydrophobic", "oc_hydrophilic", "oc_hydrophobic",
               "dust1", "dust2", "dust3", "dust4", "dust5")
LAP_TABLE_SHA256_FILE = "lap_optics.sha256"
LAP_TABLE_WORDS = 6 * 480 * 3
LAP_CONTROLS = ("snicar_lap_opt", "snicar_lap_table", "snicar_lap_series")


def lap_settings(cfg):
    """The LAP controls, refused by name where they cannot run."""
    opt = int(getattr(cfg, "snicar_lap_opt", 0) or 0)
    if opt not in (0, 1, 2):
        raise ValueError("snicar_lap_opt must be 0 (off), 1 (table fluxes) or 2 (deposition series)")
    if opt and (int(cfg.sf_surface_physics) != 4 or int(cfg.opt_alb) != 3):
        raise ValueError("snicar_lap_opt needs Noah-MP (sf_surface_physics=4) with opt_alb=3: "
                         "the particles darken the opt_alb=3 snow solver only")
    if opt == 2 and not getattr(cfg, "snicar_lap_series", ""):
        raise ValueError("snicar_lap_opt=2 needs snicar_lap_series, the receipt.json the "
                         "deposition producer wrote")
    return {name: getattr(cfg, name, "" if name != "snicar_lap_opt" else 0) for name in LAP_CONTROLS}


def _lap_data():
    return Path(__file__).resolve().parents[1]/"data"/"snow_papers"


@lru_cache(maxsize=1)
def lap_species_table():
    """Per-species meltwater scavenging factors and their sources."""
    import json
    table = json.loads((_lap_data()/"lap_species.json").read_text())
    rows = {row["key"]: row for row in table["species"]}
    if tuple(rows) != LAP_SPECIES:
        raise ValueError("lap_species.json must list the nine LAP species in kernel order")
    return table


@lru_cache(maxsize=1)
def _lap_host_table():
    from woof.static.rust_bridge import load, _check
    library = load()
    try:
        reader = library.gpuwm_snow_lap_optics_load_v1
    except AttributeError:
        raise RuntimeError("snicar_lap_opt requires a static-fields bridge built with "
                           "gpuwm_snow_lap_optics_load_v1 (rebuild tools/rustwx)") from None
    digest = (_lap_data()/LAP_TABLE_SHA256_FILE).read_text().split()[0]
    values = np.empty(LAP_TABLE_WORDS, dtype=np.float32)
    ptr = ctypes.POINTER(ctypes.c_float)
    reader.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ptr, ctypes.c_size_t]
    reader.restype = ctypes.c_int32
    code = reader(str(_lap_data()/"lap_optics.bin").encode(), digest.encode(),
                  values.ctypes.data_as(ptr), values.size)
    _check(library, code, "generated snow LAP optics")
    values.flags.writeable = False
    return values


@lru_cache(maxsize=8)
def _lap_device(device):
    import cupy as cp
    with cp.cuda.Device(device):
        scav = np.asarray([row["meltwater_scavenging"] for row in lap_species_table()["species"]],
                          dtype=np.float32)
        return cp.asarray(_lap_host_table()), cp.asarray(scav)


@lru_cache(maxsize=8)
def _lap_module(device):
    import cupy as cp
    from woof.core.noahmp_kernel_sources import _compile_loaded_module
    kernels = Path(__file__).with_name("kernels")
    # snow_papers.cu is reused verbatim; snow_lap.cu adds the LAP kernels.
    source = (kernels/"snow_papers.cu").read_text() + "\n" + (kernels/"snow_lap.cu").read_text()
    with cp.cuda.Device(device):
        return _compile_loaded_module(source, ("-std=c++17", "--ftz=false"), "snow_lap")


def lap_outflow(lap, qsnbot, dt):
    """Record QSNBOT*dt (kg/m2) for the next step's meltwater flushing."""
    import cupy as cp
    n = len(lap)
    if n:
        _lap_module(cp.cuda.Device().id).get_function("snow_lap_outflow")(
            ((n+255)//256,), (256,),
            (lap, cp.ascontiguousarray(qsnbot, dtype=cp.float32),
             cp.ascontiguousarray(cp.broadcast_to(cp.asarray(dt, dtype=cp.float32), (n,))), np.int32(n)))


def lap_initial_key(table):
    """Table initial loads: column kg/m2 mixed by mass, or a top-layer kg/kg ratio."""
    mode = table.get("initial_mode", "column_kg_m2")
    if mode not in ("column_kg_m2", "top_layer_mass_ratio"):
        raise ValueError("LAP table initial_mode must be column_kg_m2 or top_layer_mass_ratio")
    return "initial_kg_m2" if mode == "column_kg_m2" else "initial_top_layer_kg_kg"


@lru_cache(maxsize=16)
def _table_mode(path):
    import json
    return json.loads(Path(path).read_text()).get("initial_mode", "column_kg_m2")


def lap_init_top(opt):
    """1 when a table run gives its initial loads as a top-layer mass ratio."""
    if int(opt.get("snicar_lap_opt", 0) or 0) != 1 or not opt.get("snicar_lap_table"):
        return 0
    return int(_table_mode(str(opt["snicar_lap_table"])) == "top_layer_mass_ratio")


_LAP_PROVIDERS = {}


def lap_provider(state, settings, shape, start_time):
    """One device-resident forcing per domain state, built on first use."""
    key = (id(state), tuple(sorted((k, str(v)) for k, v in settings.items() if k in LAP_CONTROLS)))
    held = _LAP_PROVIDERS.get(key)
    if held is None or held[0] is not state:
        held = (state, LapDeposition(settings, shape, start_time))
        _LAP_PROVIDERS[key] = held
    return held[1]


class LapDeposition:
    """Device-resident deposition forcing for one domain (27 words per column).

    Table (opt 1): the same fluxes everywhere, from woof/data/snow_papers/
    lap_table_*.json or a user table. Series (opt 2): the producer's
    [time][18][column] fluxes; each stored value is the mean over its
    averaging window, so forcing is held piecewise constant over that window
    (the deposited mass equals the source's). The series' part before the
    forecast start is summed into the 9 initial loads (spin-up).
    """
    def __init__(self, settings, shape, start_time):
        import cupy as cp, json, hashlib
        from datetime import datetime, timedelta
        self.shape = tuple(shape)
        ncol = int(np.prod(self.shape))
        opt = int(settings["snicar_lap_opt"])
        self.buffer = cp.zeros((LAP_DEPOSITION_WORDS, ncol), dtype=cp.float32)
        self.series = None
        if opt == 1:
            path = settings.get("snicar_lap_table") or str(_lap_data()/"lap_table_zero.json")
            table = json.loads(Path(path).read_text())
            if table.get("units") != "kg m-2 s-1":
                raise ValueError(f"{path}: LAP table units must be kg m-2 s-1")
            for k, name in enumerate(LAP_SPECIES):
                row = table["species"][name]
                self.buffer[k] = np.float32(row["dry"]); self.buffer[9+k] = np.float32(row["wet"])
                self.buffer[18+k] = np.float32(row.get(lap_initial_key(table), 0.0))
            self.describe = {"mode": "table", "path": str(path), "initial": table.get("initial_mode", "column_kg_m2")}
            return
        receipt_path = Path(settings["snicar_lap_series"])
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("schema") != "gpuwm-snow-lap-deposition-series-v1":
            raise ValueError(f"{receipt_path}: not a LAP deposition series receipt")
        if receipt["columns"] != ncol:
            raise ValueError(f"{receipt_path}: series has {receipt['columns']} columns, domain has {ncol}")
        expected = [f"{s}_{kind}" for kind in ("dry", "wet") for s in LAP_SPECIES]
        if receipt["channels"] != expected:
            raise ValueError(f"{receipt_path}: channels must be {expected}")
        raw = (receipt_path.parent/receipt["series"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt["series_sha256"]:
            raise ValueError(f"{receipt_path}: series SHA-256 mismatch")
        times = [datetime.fromisoformat(t.replace("Z", "")) for t in receipt["times"]]
        nt = len(times)
        host = np.frombuffer(raw, dtype="<f4").reshape(nt, 18, ncol)
        half = timedelta(seconds=float(receipt["window_seconds"]) / 2)
        self.starts = [t - half for t in times]; self.ends = [t + half for t in times]
        start = start_time.replace(tzinfo=None) if start_time.tzinfo else start_time
        # Spin-up: deposition between the first window and the forecast start.
        init = np.zeros((9, ncol), dtype=np.float64)
        for k in range(nt):
            overlap = (min(self.ends[k], start) - self.starts[k]).total_seconds()
            if overlap > 0:
                init += (host[k, :9] + host[k, 9:]).astype(np.float64) * overlap
        self.buffer[18:] = cp.asarray(init.astype(np.float32))
        self.series = cp.asarray(host)
        self.current = -2
        self.describe = {"mode": "series", "path": str(receipt_path), "source": receipt["source"],
                         "times": nt, "window_seconds": receipt["window_seconds"],
                         "spinup_hours": max(0.0, (start - self.starts[0]).total_seconds() / 3600.0)}

    def at(self, valid_time):
        """The (27, ny, nx) forcing for a step valid at ``valid_time``."""
        if self.series is not None:
            t = valid_time.replace(tzinfo=None) if valid_time.tzinfo else valid_time
            k = next((i for i in range(len(self.starts)) if self.starts[i] <= t < self.ends[i]), -1)
            if k != self.current:
                if k < 0:
                    raise ValueError(f"snicar_lap_series does not cover {t.isoformat()} "
                                     f"({self.starts[0].isoformat()} to {self.ends[-1].isoformat()})")
                self.buffer[:18] = self.series[k]
                self.current = k
        return self.buffer.reshape(LAP_DEPOSITION_WORDS, *self.shape)
