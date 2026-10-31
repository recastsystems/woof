"""TEMPO's own 10 cm reflectivity of a given model state, outside a microphysics step.

The RRFS suites (hrrrv5 and the others on mp_tempo) form ``refl10cm`` only inside the TEMPO column
step, from the state that step leaves (module_mp_tempo_main.F90: final update, then the rain/snow/graupel
check-and-update, the snow moments and reflectivity_10cm).  A state the model never stepped to has no
reflectivity of its own:

* the start frame of a forecast (f000), written before any physics has run, carries the init's
  ``refl10cm`` (zero on every start, including an IAU start from a background full of rain);
* an analysis (DA posterior, recentred or clear-air-cleaned member init) carries the BACKGROUND's
  stored ``refl10cm``, since no analysis updates that diagnostic.

THE BREAKAGE THIS ENDS.  The hrrrv5 DA arms (Vermont 2023-07-10, V8X6000) could not score or draw the
10:15 analysis frame (the IAU start's f000 refl10cm was 0 everywhere), and the posterior O-A for radar
reflectivity was measured with WSM6's operator on a TEMPO suite, an operator residue of 13 dB at p99.9
against the stored background (refl_rediag.json), so "Z 3.93 -> 3.64 dBZ" was part operator.

This module runs exactly that tail of tempo_main, the same device functions the model's column step calls
(this file appends one kernel to tempo_v303's strict column source), on the given state with zero
tendencies.  The TEMPO parameters come from tempo_init's own route without a native library
(framework_init_rrfs: the pinned PARAMETER words, then tempo_table_init); the large collection and
lookup tables are not read by the reflectivity and are held as zeros.

On a stored background the result reproduces the model's ``refl10cm`` except for what the model's
step changed between its thermodynamic inputs and its final update (the step's temperature and vapour
change feed rho, the melting level and the snow moments): see ``residue``.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Mapping

import numpy as np

#: MPAS constants the state's temperature is formed with (mpas_constants.F: rgas, cp, p0).
RGAS, CP_DRY, P0 = 287.0, 1004.5, 1.0e5

#: Columns per launch: the per-column work area is ~16 KB, so a chunk is ~1 GB at most.
CHUNK_COLUMNS = 65536

#: Columns per host-to-card slab of the f000 start-frame reflectivity (refl10cm_staged): 1,344 MiB on the card at
#: 59 levels with hail-aware graupel (measured, RTX 5090), mostly the column work area, whatever the mesh size.
STAGE_COLUMNS = 32768

REFL_SOURCE = r"""
// The tail of tempo_main (module_mp_tempo_main.F90 final update -> rain/snow/graupel check_and_update ->
// snow moments -> reflectivity_10cm, tempo_v303 _COLUMN) on a given state with zero tendencies.
struct TempoReflWork {
  float temp[TEMPO_NZP], pres[TEMPO_NZP], qv[TEMPO_NZP], rho[TEMPO_NZP];
  float qr1[TEMPO_NZP], nr1[TEMPO_NZP], qs1[TEMPO_NZP], qg1[TEMPO_NZP], ng1[TEMPO_NZP], qb1[TEMPO_NZP];
  float rr[TEMPO_NZP], nr[TEMPO_NZP], rs[TEMPO_NZP], rg[TEMPO_NZP], ng[TEMPO_NZP], rb[TEMPO_NZP];
  float qrten[TEMPO_NZP], nrten[TEMPO_NZP], qsten[TEMPO_NZP], qgten[TEMPO_NZP], ngten[TEMPO_NZP], qbten[TEMPO_NZP];
  float mvd_r[TEMPO_NZP], mvd_g[TEMPO_NZP], refl10cm[TEMPO_NZP];
  double ilamr[TEMPO_NZP], ilamg[TEMPO_NZP], smob[TEMPO_NZP], smoc[TEMPO_NZP], smo2[TEMPO_NZP], smoz[TEMPO_NZP];
  bool l_qr[TEMPO_NZP], l_qs[TEMPO_NZP], l_qg[TEMPO_NZP];
  int idx_bg[TEMPO_NZP];
};
extern "C" __global__ void tempo_refl_sizes(unsigned long long* out) {
  out[0] = sizeof(TempoReflWork); out[1] = sizeof(TempoBlob);
}
extern "C" __global__ void tempo_refl_state(int nz, int stride, int first, int count,
    const float* t, const float* p, const float* qv, const float* qr, const float* nr, const float* qs,
    const float* qg, const float* ng, const float* qb, const TempoBlob* blob, int hail, int melting,
    TempoReflWork* work, float* dbz) {
  int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= count) return;
  long long g = (long long)first + c;
  TempoParams P = {blob};
  // global_dt 1 (the tendencies are zero, so it never scales a word), refl10cm_flag on, no radii.
  TempoCtx C = {1.0f, 1.0f, 0, hail, 0, melting, 0, 1, 1, 0, 0, 0, 0, 0, 0};
  TempoReflWork& W = work[c];
  const bool graupel_moments = ng != nullptr && qb != nullptr;
  for (int k = 1; k <= nz; ++k) {
    long long x = (long long)(k - 1) * stride + g;
    W.temp[k] = t[x]; W.pres[k] = p[x];
    W.qv[k] = tmax(PS(min_qv), qv[x]);
    W.rho[k] = PS(roverrv) * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + PS(roverrv)));
    W.qr1[k] = qr[x]; W.nr1[k] = nr[x]; W.qs1[k] = qs[x]; W.qg1[k] = qg[x];
    W.ng1[k] = graupel_moments ? ng[x] : 0.0f; W.qb1[k] = graupel_moments ? qb[x] : 0.0f;
    W.qrten[k] = 0.0f; W.nrten[k] = 0.0f; W.qsten[k] = 0.0f; W.qgten[k] = 0.0f; W.ngten[k] = 0.0f;
    W.qbten[k] = 0.0f;
    W.smob[k] = 0.0; W.smoc[k] = 0.0; W.smo2[k] = 0.0; W.smoz[k] = 0.0;
  }
  rain_check_and_update(P, C, nz, W.rho, W.l_qr, W.qr1, W.nr1, W.rr, W.nr, W.qrten, W.nrten, W.ilamr, W.mvd_r);
  snow_check_and_update(P, C, nz, W.rho, W.l_qs, W.qs1, W.rs, W.qsten);
  const float t0 = PS(t0);
  for (int k = 1; k <= nz; ++k) {
    if (W.l_qs[k]) {
      float tc0 = tmin(-0.1f, W.temp[k] - t0);
      snow_moments(P, W.rs[k], tc0, &W.smob[k], &W.smoc[k], nullptr, nullptr, nullptr,
                   &W.smo2[k], nullptr, nullptr, nullptr, &W.smoz[k]);
    }
  }
  graupel_check_and_update(P, C, nz, W.rho, W.l_qg, W.qg1, graupel_moments ? W.ng1 : nullptr,
                           graupel_moments ? W.qb1 : nullptr, W.rg, W.ng, W.rb, W.idx_bg,
                           W.qgten, W.ngten, W.qbten, W.ilamg, W.mvd_g);
  reflectivity_10cm(P, C, nz, melting, W.temp, W.l_qr, W.rr, W.nr, W.ilamr, W.l_qs, W.rs, W.smoc,
                    W.smob, W.smoz, W.l_qg, W.rg, W.ng, W.idx_bg, W.ilamg, W.refl10cm);
  for (int k = 1; k <= nz; ++k) dbz[(long long)(k - 1) * stride + g] = W.refl10cm[k];
}
"""


def _pin_metadata(pin_set: str) -> Path:
    """The framework metadata (TEMPO PARAMETER words) of the RRFS fork pin this suite runs."""
    from importlib import resources
    root = Path(str(resources.files('woof.hex'))) / 'data'
    name = {'rrfs': 'mpasphys-framework-rrfs', 'rrfs-1d6e370e': 'mpasphys-framework-rrfs',
            'rrfs-44c0849': 'mpasphys-framework-rrfs-44c0849'}.get(pin_set, f'mpasphys-framework-{pin_set}')
    path = root / name / 'metadata.json'
    if not path.is_file():
        raise FileNotFoundError(f'no TEMPO parameter words for pin set {pin_set!r} ({path})')
    return path


@lru_cache(maxsize=None)
def parameter_blob(pin_set: str, hailaware: bool, melting: bool) -> np.ndarray:
    """The TEMPO parameter blob (tempo_v303.TABLE_LAYOUT, binary64, Fortran order per table), host side.

    tempo_init's route without a native library (FrameworkInitializationRRFS.initialize_tempo_tables steps 1-2):
    the pinned PARAMETER words, then tempo_table_init (Rust) for these hail-aware and melting flags.  The
    collection, evaporation and lookup tables (steps 3-4) are not read by the reflectivity and stay zero.
    """
    import subprocess
    from woof.hex.mpasphys.rrfs.framework_init_rrfs import (TEMPO_CONSTANT_ARRAYS, TEMPO_CONSTANT_SCALARS,
                                                           TEMPO_DERIVED, TEMPO_HAIL_BINS, TEMPO_RADAR_BINS)
    from woof.hex.mpasphys.rrfs.tempo_v303 import TABLE_LAYOUT
    from woof.hex.mpasphys.resources_build import default_cache, locate_decoders
    meta = json.loads(_pin_metadata(pin_set).read_text())
    held: dict[str, np.ndarray] = {}
    for name, row in meta['tempo_constants'].items():
        words = np.array(row['words'], np.uint64 if row['dtype'] == 'float64' else np.uint32)
        held[name] = words.view(np.dtype(row['dtype']).newbyteorder('<')).astype(row['dtype']).reshape(row['shape'] or ())
    parts = [np.array([bool(hailaware), bool(melting), True], '<i4').tobytes()]
    parts += [np.float32(held[name]).tobytes() for name in TEMPO_CONSTANT_SCALARS]
    parts += [np.asarray(held[name], np.float32).tobytes(order='F') for name in TEMPO_CONSTANT_ARRAYS]
    tool = locate_decoders(default_cache())['tempo_table_init']
    raw = subprocess.run([str(tool)], input=b''.join(parts), check=True, capture_output=True).stdout
    offset = 0
    for name, dtype, shape in (*TEMPO_DERIVED, *TEMPO_HAIL_BINS, *(TEMPO_RADAR_BINS if melting else ())):
        n = math.prod(shape) if shape else 1
        value = np.frombuffer(raw, dtype=dtype, count=n, offset=offset)
        offset += n * value.itemsize
        held[name] = value.reshape(shape, order='F') if shape else value.reshape(()).copy()
    if offset != len(raw):
        raise ValueError('tempo_table_init wrote words the tables do not hold')
    blob = np.zeros(sum(math.prod(shape) for _, shape in TABLE_LAYOUT), np.float64)
    at = 0
    for name, shape in TABLE_LAYOUT:
        n = math.prod(shape)
        if name in held:
            value = np.asarray(held[name])
            if value.size != n:
                raise ValueError(f'TEMPO parameter {name}: {value.size} words, layout holds {n}')
            blob[at:at + n] = value.reshape(-1, order='F').astype(np.float64)
        at += n
    return blob


_MODULES: dict[tuple[int, int], Any] = {}
_MODULE_LOCK = threading.Lock()


def _module(device: int, nz: int):
    """tempo_v303's strict column source plus the state-reflectivity kernel, compiled once per card and depth.
    A second caller (the f000 thread behind the writer's early warm-up) waits for the first compile, never repeats it."""
    key = (int(device), int(nz))
    with _MODULE_LOCK:
        if key not in _MODULES:
            _MODULES[key] = _build_module(*key)
        return _MODULES[key]


def module_source(nz: int) -> str:
    """The module's text: tempo_v303's strict column routines (both numerics: the reflectivity reads the f64
    parameter blob) at ``nz`` levels, its MPAS slice kernels, then the state-reflectivity kernel.  Not the
    column driver (tempo_columns, the whole microphysics step): with it the module took 97.9 s of NVRTC on one
    core (an RTX 5090 host, lam7k, 2026-10-29) and step 1 waited 53 s for it.  The AOT inventory declares the same text
    (cuda_backend.aot_build.physics_inventory, family tempo_rrfs_refl; the slice kernels pair it with the
    fast TEMPO unit)."""
    from woof.hex.mpasphys.resources_build import _vendored_maths, tempo_maths
    from woof.hex.mpasphys.rrfs import tempo_v303 as tempo
    maths = tempo_maths(_vendored_maths())
    return (f"#define TEMPO_NZ_MAX {max(int(nz), 1)}\n" + tempo._shared_pow_define(maths) + maths
            + tempo._column_source('strict') + tempo._mpas_source('strict') + REFL_SOURCE)


def module_options() -> tuple[str, ...]:
    """Strict TEMPO compile options in both numerics (see :func:`module_source`)."""
    from woof.hex import numerics as hex_numerics
    from woof.hex.mpasphys.rrfs import tempo_v303 as tempo
    return tuple(hex_numerics.options(tempo.COMPILE_OPTIONS, numerics='strict'))


#: Seconds each (card, depth) module took to compile or load from the kernel pack.
MODULE_SECONDS: dict[tuple[int, int], float] = {}


def _build_module(device: int, nz: int):
    import cupy as cp
    from cupy.cuda import compiler, function
    started = time.time()
    source = module_source(nz)
    with cp.cuda.Device(device):
        # Through the engine's direct-NVRTC bridge: served from the AOT pack when it holds this text.
        image, _ = compiler.compile_using_nvrtc(source, options=module_options(), cache_in_memory=True)
        module = function.Module()
        module.load(image)
        sizes = cp.empty(2, dtype=cp.uint64)
        module.get_function('tempo_refl_sizes')((1,), (1,), (sizes,))
        result = module, tuple(int(v) for v in sizes.get())
    MODULE_SECONDS[(int(device), int(nz))] = round(time.time() - started, 2)
    return result


def refl10cm(fields: Mapping[str, Any], *, pin_set: str = 'rrfs-44c0849', hailaware: bool = True,
             melting: bool = False, cp: Any = None):
    """TEMPO 10 cm reflectivity (dBZ, ``(nz, ncells)`` float32 on the device) of one state.

    ``fields`` (each ``(nz, ncells)``): ``t`` (K) and ``p`` (Pa), or ``theta`` with ``p`` (``exner`` optional:
    else (p / p0) ** (rgas / cp)); ``qv``, ``qr``, ``nr``, ``qs``, ``qg`` (kg/kg, /kg); ``ng`` and ``volg``
    (hail-aware graupel number and volume) when the state carries them.
    """
    if cp is None:
        import cupy as cp
    f = {k: cp.ascontiguousarray(cp.asarray(v, dtype=cp.float32)) for k, v in fields.items() if v is not None}
    if 't' not in f:
        exner = f.get('exner')
        if exner is None:
            exner = ((f['p'].astype(cp.float64) / P0) ** (RGAS / CP_DRY)).astype(cp.float32)
        f['t'] = f['theta'] * exner
    nz, ncells = f['qv'].shape
    for name in ('t', 'p', 'qv', 'qr', 'nr', 'qs', 'qg'):
        if f[name].shape != (nz, ncells):
            raise ValueError(f'TEMPO reflectivity input {name} is {f[name].shape}, not {(nz, ncells)}')
    device = f['qv'].device.id
    module, (work_bytes, blob_bytes) = _module(device, nz)
    blob = cp.asarray(parameter_blob(pin_set, bool(hailaware), bool(melting)))
    if blob.nbytes != blob_bytes:
        raise ValueError(f'TEMPO parameter blob is {blob.nbytes} bytes, the module reads {blob_bytes}')
    ng, qb = f.get('ng'), f.get('volg')
    out = cp.empty((nz, ncells), cp.float32)
    chunk = max(1, min(ncells, CHUNK_COLUMNS))
    work = cp.empty(chunk * work_bytes, cp.uint8)
    kernel = module.get_function('tempo_refl_state')
    null = np.uint64(0)
    for first in range(0, ncells, chunk):
        count = min(chunk, ncells - first)
        kernel(((count + 127) // 128,), (128,), (
            np.int32(nz), np.int32(ncells), np.int32(first), np.int32(count),
            f['t'], f['p'], f['qv'], f['qr'], f['nr'], f['qs'], f['qg'],
            ng if ng is not None else null, qb if qb is not None else null,
            blob, np.int32(bool(hailaware)), np.int32(bool(melting)), work, out))
    return out


def column_radar(refl, zgrid, *, cp: Any = None):
    """``refl10cm_max`` and ``refl10cm_1km`` of a ``(nz, ncells)`` profile as the fork's tempo_radar forms them
    (mpas_atmphys_driver_microphysics.F:1383-1435): the column maximum floored at -35 dBZ, and the profile
    interpolated to 1 km above ground between the last layer centre below it and the next.  ``zgrid`` is
    ``(nz + 1, ncells)``."""
    if cp is None:
        import cupy as cp
    r = cp.maximum(cp.asarray(refl, cp.float32), cp.float32(-35.0))
    z = cp.asarray(zgrid, cp.float32)
    nz = r.shape[0]
    dz = z[1:nz + 1] - z[:nz]
    zp = z[:nz] - z[0:1] + cp.float32(0.5) * dz
    below = zp < cp.float32(1000.0)
    kp = cp.where(below.any(axis=0), nz - 1 - cp.argmax(below[::-1], axis=0), 0)
    kp = cp.minimum(kp, nz - 2)
    cols = cp.arange(r.shape[1])
    z1, z2 = zp[kp, cols], zp[kp + 1, cols]
    w1 = (z2 - cp.float32(1000.0)) / (z2 - z1)
    return r.max(axis=0), w1 * r[kp, cols] + (cp.float32(1.0) - w1) * r[kp + 1, cols]


def refl10cm_staged(state: Mapping[str, np.ndarray], zgrid: np.ndarray | None, *, pin_set: str = 'rrfs-44c0849',
                    hailaware: bool = True, melting: bool = False, cp: Any = None, stream: Any = None,
                    columns: int = STAGE_COLUMNS) -> dict[str, np.ndarray]:
    """``refl10cm`` (``(nz, ncells)``), ``refl10cm_max`` and, with ``zgrid``, ``refl10cm_1km`` of a HOST state, formed
    one slab of ``columns`` columns at a time; host float32 results, the same words as :func:`refl10cm` and
    :func:`column_radar` on the whole state (every column is formed from its own words only).

    THE BREAKAGE THIS PREVENTS: the f000 start frame of a same-host multi-card run is the GLOBAL frame, held on
    rank 0 (samehost_v842 GatheredPhysicsStreams), and its reflectivity was formed by copying the whole global
    state to rank 0's card while step 1 runs there.  On NOAA's conus3km over 4 x 96 GB cards (2,341,818 cells,
    rank 0 already at 91.2-91.6 GB of ~97.9 GB in its HB15 runs) that is 7,998 MiB more on one card at its
    step-1 peak (measured at 59 levels: inputs, the float64 exner, output and the work area, all at full mesh
    size), past the card.  Slabs keep the card's share at 1,344 MiB at any mesh size, with the same words; the
    side thread takes longer (2.6 -> 12.7 s at that size), still inside conus3km's 33-41 s first step.
    """
    if cp is None:
        import cupy as cp
    nz, ncells = state['qv'].shape
    out = {'refl10cm': np.empty((nz, ncells), np.float32), 'refl10cm_max': np.empty(ncells, np.float32)}
    if zgrid is not None:
        out['refl10cm_1km'] = np.empty(ncells, np.float32)
    step = max(1, int(columns))
    for first in range(0, ncells, step):
        cols = slice(first, min(ncells, first + step))
        slab = {k: np.ascontiguousarray(v[:, cols]) for k, v in state.items() if v is not None}
        z = refl10cm(slab, pin_set=pin_set, hailaware=hailaware, melting=melting, cp=cp)
        if zgrid is not None:
            peak, at_1km = column_radar(z, np.ascontiguousarray(zgrid[:, cols]), cp=cp)
            out['refl10cm_1km'][cols] = cp.asnumpy(at_1km, stream=stream)
        else:
            peak = cp.maximum(z, cp.float32(-35.0)).max(axis=0)
        out['refl10cm'][:, cols] = cp.asnumpy(z, stream=stream)
        out['refl10cm_max'][cols] = cp.asnumpy(peak, stream=stream)
        if stream is not None:
            stream.synchronize()
        del z, peak, slab
    return out


def read_state(path, fallback=None):
    """The reflectivity inputs of an MPAS state file (time level 0), host ``(nz, ncells)`` float32, and the
    file's stored refl10cm when it has one.  pressure_p/pressure_base (and ng/volg) missing from ``path`` come
    from ``fallback`` (one file or a list, in order: the member's background), as fromjedi rebuilds them."""
    from netCDF4 import Dataset

    def read(ds, name):
        v = ds[name]
        a = np.asarray(v[0] if v.dimensions[0] == 'Time' else v[:], np.float32)
        return np.ascontiguousarray(a.T) if a.ndim == 2 else a

    names = ('theta', 'pressure', 'pressure_p', 'pressure_base', 'qv', 'qr', 'nr', 'qs', 'qg', 'ng', 'volg',
             'refl10cm', 'zgrid')
    out, borrowed = {}, []
    with Dataset(str(path)) as d:
        d.set_auto_maskandscale(False)
        for name in names:
            if name in d.variables:
                out[name] = read(d, name)
    fallbacks = [fallback] if isinstance(fallback, (str, Path)) else list(fallback or ())
    for other in fallbacks:
        if other is None or not Path(other).exists():
            continue
        with Dataset(str(other)) as d:
            d.set_auto_maskandscale(False)
            for name in ('pressure_p', 'pressure_base', 'ng', 'volg', 'zgrid'):
                if name not in out and name in d.variables:
                    out[name] = read(d, name)
                    borrowed.append(f'{name} from {other}')
    # A history frame carries the full ``pressure``; an init or JEDI copy carries pressure_p (+ pressure_base).
    missing = [n for n in ('theta', 'qv', 'qr', 'nr', 'qs', 'qg') if n not in out]
    if 'pressure' not in out and 'pressure_p' not in out:
        missing.append('pressure or pressure_p')
    if missing:
        raise KeyError(f'{path}: no {missing} (TEMPO reflectivity cannot be formed)')
    base = out.get('pressure_base')
    if 'pressure' in out:
        p = out['pressure']
    else:
        p = out['pressure_p'] if base is None or not np.any(base) else (out['pressure_p'] + base).astype(np.float32)
    if not np.all(p > 0):
        raise ValueError(f'{path}: {int((p <= 0).sum())} points with non-positive pressure')
    state = {'theta': out['theta'], 'p': p, **{n: out[n] for n in ('qv', 'qr', 'nr', 'qs', 'qg', 'ng', 'volg')
                                               if n in out}}
    return state, out.get('refl10cm'), out.get('zgrid'), borrowed


def residue(formed, stored, floor: float = 0.0) -> dict[str, float]:
    """|formed - stored| over every word, both floored at ``floor`` (the JEDI copy's clear-air floor)."""
    a = np.maximum(np.asarray(formed, np.float32), np.float32(floor))
    b = np.maximum(np.asarray(stored, np.float32), np.float32(floor))
    d = np.abs(a - b)
    return dict(max_abs_db=float(d.max()), p999_abs_db=float(np.percentile(d, 99.9)),
                p99_abs_db=float(np.percentile(d, 99)), mean_abs_db=float(d.mean()),
                words_over_1db=int((d > 1.0).sum()), words=int(d.size),
                formed_ge_20=int((a >= 20).sum()), stored_ge_20=int((b >= 20).sum()))


def rediagnose_file(path, *, fallback=None, pin_set='rrfs-44c0849', hailaware=True, melting=False,
                    floor_dbz=0.0, write=True, cp=None) -> dict[str, Any]:
    """Rewrite ``refl10cm`` of an MPAS state file (a DA analysis or observer copy) with TEMPO's own reflectivity
    of the state's hydrometeors, floored at ``floor_dbz`` like the JEDI copy of the background.  Returns the
    record (the old field's change, and the operator residue when ``path`` still holds the background)."""
    if cp is None:
        import cupy as cp
    from netCDF4 import Dataset
    started = time.time()
    state, stored, _, borrowed = read_state(path, fallback)
    z = cp.asnumpy(refl10cm(state, pin_set=pin_set, hailaware=hailaware, melting=melting, cp=cp))
    z = np.maximum(z, np.float32(floor_dbz)).astype(np.float32)
    record = dict(file=str(path), operator='TEMPO reflectivity_10cm of the state (hexcore tempo_refl_v303)',
                  pin_set=pin_set, hailaware=bool(hailaware), melting=bool(melting), floor_dbz=float(floor_dbz),
                  from_fallback=borrowed, cells_ge_20_after=int((z >= 20).sum()))
    if stored is not None:
        record['change_from_stored'] = residue(z, stored, floor_dbz)
    if write:
        with Dataset(str(path), 'r+') as d:
            d.set_auto_maskandscale(False)
            v = d['refl10cm']
            if v.dimensions[0] == 'Time':
                v[0] = z.T
            else:
                v[:] = z.T
            d.setncattr('hexcore_refl10cm', 'TEMPO reflectivity_10cm of this state\'s hydrometeors '
                        f'(hexcore tempo_refl_v303, pin {pin_set}, hail-aware {bool(hailaware)}, melting '
                        f'{bool(melting)}), floored at {float(floor_dbz)} dBZ')
    record['seconds'] = round(time.time() - started, 2)
    return record


#: The start frame's reflectivity fields this completes (diag stream), when they hold a placeholder (one word everywhere).
START_TARGETS = ('refl10cm', 'refl10cm_max', 'refl10cm_1km')


def _logical(item) -> tuple[np.ndarray, tuple[str, ...]]:
    """A HistoryField's values as ``(vertical, nCells)`` (or ``(nCells,)``), and the axis order to restore."""
    values = np.asarray(item.values)
    dims = tuple(item.dimensions or ())
    if len(dims) != values.ndim:
        raise ValueError(f'history field dims {dims} do not match shape {values.shape}')
    if 'Time' in dims:
        values = np.take(values, 0, axis=dims.index('Time'))
        dims = tuple(d for d in dims if d != 'Time')
    if values.ndim == 1:
        return values, dims
    vert = next(i for i, d in enumerate(dims) if d.startswith('nVertLevels'))
    cell = dims.index('nCells')
    return np.ascontiguousarray(np.transpose(values, (vert, cell))), dims


def _restore(logical: np.ndarray, item) -> np.ndarray:
    """``logical`` back in ``item``'s own axis order and dtype."""
    values = np.asarray(item.values)
    dims = tuple(item.dimensions or ())
    flat = tuple(d for d in dims if d != 'Time')
    if logical.ndim == 2:
        vert = next(i for i, d in enumerate(flat) if d.startswith('nVertLevels'))
        order = [0, 1] if vert < flat.index('nCells') else [1, 0]
        logical = np.transpose(logical, order)
    if 'Time' in dims:
        logical = np.expand_dims(logical, dims.index('Time'))
    return np.ascontiguousarray(logical.reshape(values.shape), dtype=values.dtype)


def _placeholder(item) -> bool:
    """Every word of the field is the same word: the start frame's placeholder (zero, or the pool's initial
    value), never a diagnosed reflectivity unless the whole domain is echo-free, where forming it changes nothing."""
    values = np.ascontiguousarray(np.asarray(item.values))
    if values.size == 0:
        return False
    words = values.reshape(-1).view(np.uint32 if values.dtype.itemsize == 4 else np.uint64)
    return bool(np.all(words == words[0]))


class StartFrameReflectivity:
    """The start frame's (f000) ``refl10cm``, ``refl10cm_max`` and ``refl10cm_1km`` of the start state.

    THE BREAKAGE THIS ENDS: an MPAS-suite start frame is written before any microphysics has run, so its
    reflectivity is the init's placeholder, zero everywhere, even when the start state is full of rain:
    the hrrrv5 DA arms' IAU analysis-mean forecast from 10:15 (Vermont 2023-07-10) published an empty
    analysis-time map and its 10:15 frame could not be scored.  The state's own TEMPO reflectivity
    (``refl10cm`` above) is formed from the frame's captured state on a side thread and card stream while
    step 1 runs, and written into the held frame only where a field holds its placeholder (one word in every cell).
    The model state is untouched; only the f000 diagnostics frame changes.  The MPAS suite route holds
    the frame in correct mode only (streams_output_v842 first_frame_from_first_call), so exact mode
    writes native MPAS-A's f000 bit for bit.
    """

    def __init__(self, snapshot: Mapping[str, Any], *, hailaware: bool, melting: bool, pin_set: str,
                 device: int) -> None:
        import threading
        self.targets = [n for n in START_TARGETS if n in snapshot and _placeholder(snapshot[n])]
        self.record: dict[str, Any] = dict(operator='TEMPO reflectivity_10cm of the start state '
                                           '(hexcore tempo_refl_v303)', hailaware=bool(hailaware),
                                           melting=bool(melting), fields={})
        self.result: dict[str, np.ndarray] | None = None
        self.error: BaseException | None = None
        self.thread = None
        if not self.targets:
            self.record['applied'] = False
            self.record['reason'] = 'the frame holds no reflectivity placeholder (every field already varies)'
            return
        pressure = 'pressure' if 'pressure' in snapshot else None
        need = ['theta', 'qv', 'qr', 'nr', 'qs', 'qg']
        missing = [n for n in need if n not in snapshot]
        if pressure is None and not all(n in snapshot for n in ('pressure_p', 'pressure_base')):
            missing.append('pressure')
        if 'refl10cm_1km' in self.targets and 'zgrid' not in snapshot:
            self.targets.remove('refl10cm_1km')
            self.record['refl10cm_1km'] = 'not formed: the frame carries no zgrid'
        if missing:
            self.targets = []
            self.record['applied'] = False
            self.record['reason'] = f'the start frame carries no {missing}'
            return
        state = {n: _logical(snapshot[n])[0] for n in need + [n for n in ('ng', 'volg') if n in snapshot]}
        if pressure is not None:
            state['p'] = _logical(snapshot['pressure'])[0]
        else:
            state['p'] = (_logical(snapshot['pressure_p'])[0] + _logical(snapshot['pressure_base'])[0]).astype(np.float32)
        zgrid = _logical(snapshot['zgrid'])[0] if 'zgrid' in snapshot else None
        self._args = (state, zgrid, bool(hailaware), bool(melting), pin_set, int(device))
        self.thread = threading.Thread(target=self._run, name='f000-reflectivity', daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            import cupy as cp
            state, zgrid, hail, melt, pin_set, device = self._args
            started = time.time()
            with cp.cuda.Device(device), cp.cuda.Stream(non_blocking=True) as stream:
                # In slabs: on a same-host multi-card run this is the global frame on rank 0's card, beside
                # step 1 (refl10cm_staged names the breakage).
                host = refl10cm_staged(state, zgrid, pin_set=pin_set, hailaware=hail, melting=melt, cp=cp,
                                       stream=stream)
            self.result = host
            self.record['seconds'] = round(time.time() - started, 2)
            self.record['card_slab_columns'] = STAGE_COLUMNS
        except BaseException as error:  # reported in the receipt; never fails the forecast
            self.error = error

    def apply(self, snapshot: dict[str, Any], field_type: Any) -> dict[str, Any]:
        """Join the side thread and write the formed fields into the held frame; return the record."""
        if self.thread is None:
            return self.record
        waited = time.time()
        self.thread.join()
        # How long the forecast thread waited at step 1 for the side thread (0 when it was already done).
        self.record['step1_wait_seconds'] = round(time.time() - waited, 3)
        self.record['module_seconds'] = dict((f'card {k[0]}, {k[1]} levels', v) for k, v in MODULE_SECONDS.items())
        if self.error is not None or self.result is None:
            self.record['applied'] = False
            self.record['reason'] = f'TEMPO start reflectivity failed: {self.error!r}'
            return self.record
        for name in self.targets:
            item = snapshot[name]
            value = self.result[name]
            snapshot[name] = field_type(_restore(value, item), item.dimensions, item.attrs)
            self.record['fields'][name] = dict(minimum=float(value.min()), maximum=float(value.max()),
                                               cells_ge_20=int((value >= 20).sum()))
        self.record['applied'] = True
        return self.record


def warm_start_reflectivity(configs: Mapping[str, Any], nz: int, *, device: int) -> threading.Thread | None:
    """Compile the f000 reflectivity module and build the parameter words on a side thread, from the moment the
    writer knows it will hold the start frame (None: not a TEMPO run).  Breakage it prevents: compiled at the
    f000 write instead, the strict TEMPO column source (one NVRTC call on one core, ~25 s; tempo_v303
    _compiled_warp measured 27-29 s on an RTX 5090 host) outlasts step 1, and the forecast thread waits for it there."""
    if str(configs.get('config_microp_scheme', '')).strip() != 'mp_tempo':
        return None
    hail = bool(configs.get('config_tempo_hailaware', True))
    melt = bool(configs.get('config_tempo_refl10cm_from_melting', False))

    def run() -> None:
        try:
            import cupy as cp
            with cp.cuda.Device(device):
                _module(device, int(nz))
            parameter_blob('rrfs-44c0849', hail, melt)
        except BaseException:   # the f000 thread repeats the work and names any failure in the receipt
            pass

    thread = threading.Thread(target=run, name='f000-reflectivity-warm', daemon=True)
    thread.start()
    return thread


def start_frame_reflectivity(configs: Mapping[str, Any], snapshot: Mapping[str, Any], *,
                             device: int) -> StartFrameReflectivity | None:
    """The f000 reflectivity completion for this run's microphysics (None: not a TEMPO run)."""
    if str(configs.get('config_microp_scheme', '')).strip() != 'mp_tempo':
        return None
    # The TEMPO PARAMETER words are the same at both fork pins (metadata tempo_constants, 1d6e370e = 44c0849).
    return StartFrameReflectivity(snapshot, hailaware=bool(configs.get('config_tempo_hailaware', True)),
                                  melting=bool(configs.get('config_tempo_refl10cm_from_melting', False)),
                                  pin_set='rrfs-44c0849', device=device)


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('files', nargs='+', help='MPAS state files whose refl10cm is rewritten')
    parser.add_argument('--fallback', action='append', default=[],
                        help='per file, the background holding pressure/ng/volg the file lacks (repeat in order)')
    parser.add_argument('--pin-set', default='rrfs-44c0849')
    parser.add_argument('--hailaware', type=int, default=1)
    parser.add_argument('--melting', type=int, default=0)
    parser.add_argument('--floor-dbz', type=float, default=0.0)
    parser.add_argument('--check', action='store_true', help='measure the residue against the stored field only')
    parser.add_argument('--receipt')
    a = parser.parse_args(argv)
    rows = []
    for i, path in enumerate(a.files):
        fallback = a.fallback[i] if i < len(a.fallback) else None
        rows.append(rediagnose_file(path, fallback=fallback, pin_set=a.pin_set, hailaware=bool(a.hailaware),
                                    melting=bool(a.melting), floor_dbz=a.floor_dbz, write=not a.check))
    text = json.dumps(dict(members=rows), indent=2)
    if a.receipt:
        Path(a.receipt).write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
