"""WSM6's own 10 cm reflectivity of the start state, for the held f000 frame (mesoscale_reference and every mp_wsm6 run).

MPAS-A forms ``refl10cm``, ``refl10cm_max``, ``refl10cm_1km`` and ``refl10cm_1km_max`` only inside a microphysics
diagnostics step (mpas_atmphys_driver_microphysics.F:690-740: refl10cm_wsm6 on the state the step leaves), so the
start frame (f000), written before any physics, carries the init's placeholder: zero everywhere, even when the init
holds the first guess's own rain, snow and graupel (rw_mpas_init ``--first-guess-hydrometeors``).

THE BREAKAGE THIS ENDS (hex-meshes-sources review, 2026-10-29): the f000 fix (thompson_refl_start_v842, woof hex
ee94097c) formed the start state's echo for TEMPO and Thompson only, so a mesoscale_reference (WSM6) run from GFS with
GFS's own condensate still showed "No values to plot" at f000: a fix that reached one route and not the other.

This runs WSM6's refl10cm_wsm6 (wsm6_v842.CUDA_MODULES, the audited mp_wsm6.F90:2275-2442 text, with the
state_parameters / mp_wsm6_init / radar_init tables it reads) on the frame's captured start state, then the column
maximum and the 1 km interpolation as wsm6_radar_to_mpas does, on a side thread and card stream while step 1 runs.
Temperature is theta * (p / p0) ** (rgas / cp), as the Thompson start frame forms it.  Its own small compile unit
(the scheme's step module fuses radar into the microphysics column, so it has no radar-only entry): only the code
reachable from the radar entry is compiled, and the module is warmed at setup (``warm``) so step 1 does not wait.
Only the f000 diagnostics frame changes; the model state is untouched, and exact mode keeps native MPAS-A's zero
(the frame is held in correct mode only, streams_output_v842 first_frame_from_first_call).
"""
from __future__ import annotations

import threading
import time
from functools import lru_cache
from typing import Any, Mapping

import numpy as np

from . import thompson_refl_start_v842 as _thompson

RGAS, CP_DRY, P0 = _thompson.RGAS, _thompson.CP_DRY, _thompson.P0

#: The microphysics whose CUDA slot is wsm6_v842 (MPAS-A mesoscale_reference and the RRFS WSM6 suites).
SCHEMES = ('mp_wsm6',)

START_TARGETS = _thompson.START_TARGETS

NEED = ('qv', 'qr', 'qs', 'qg', 'theta')

#: mp_wsm6_init's hail switch: the native module parameter (mpas_atmphys_driver_microphysics.F:91), as bind_native.
HAIL_OPT = 0

#: Card bytes of one slab's column workspaces (refl10cm_wsm6's automatic arrays and one Wsm6State per column).
#: THE BREAKAGE THIS PREVENTS: one Wsm6Workspace is ~45 KB at 55 levels, so a whole 149k-cell frame would take
#: ~6.7 GB of the card beside step 1 (and a same-host multi-card f000 frame is the global frame on rank 0).
SLAB_WORKSPACE_MIB = 256

START_RADAR_SOURCE = r'''
// MPAS-A v8.4.2 mp_wsm6.F90:73-189 (state_parameters, mp_wsm6_init, radar_init) once per card: the state every
// column copies (as CUDA_COOP_COLUMNS wsm6_state_template).
extern "C" __global__ void wsm6_start_template(const float *c,int hail,Wsm6State *out,int *status) {
 if(blockIdx.x || threadIdx.x)return;
 Wsm6State S{};state_parameters(S);int err=0;
 mp_wsm6_init(S,c[15],c[16],100.f,c[17],c[2],hail,err);
 S.workspace=nullptr;*out=S;*status=err;
}
extern "C" __global__ void wsm6_start_sizes(unsigned long long *out) {
 if(blockIdx.x || threadIdx.x)return;out[0]=sizeof(Wsm6Workspace);out[1]=sizeof(Wsm6State);
}
// The f000 frame's WSM6 10 cm reflectivity of the start state: refl10cm_wsm6 (mp_wsm6.F90:2275-2442) on
// t, qv, qr, qs, qg and p, as mpas_atmphys_driver_microphysics.F:690-710 calls it after a step; the column maximum
// and the 1 km interpolation as wsm6_radar_to_mpas (:711-735). Arrays are [level, column] of ns columns.
extern "C" __global__ void wsm6_start_radar(int nz,int ns,const Wsm6State *tmpl,
 const float *t,const float *qv,const float *qr,const float *qs,const float *qg,const float *p,const float *zgrid,
 float *refl,float *refl_max,float *refl_1km,Wsm6Workspace *workspace) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ns)return;
 Wsm6Workspace &B=workspace[i];
 Wsm6State &S=B.state;
 for(int w=0;w<(int)(sizeof(Wsm6State)/sizeof(unsigned long long));++w)
   ((unsigned long long*)&S)[w]=((const unsigned long long*)tmpl)[w];
 S.workspace=&B;
 float *local=B.column_local,*radar=B.column_radar;
 for(int k=0;k<nz;++k) {
  const int x=k*ns+i;
  local[k]=t[x];local[nz+k]=qv[x];local[2*nz+k]=qr[x];local[3*nz+k]=qs[x];local[4*nz+k]=qg[x];local[5*nz+k]=p[x];
 }
 refl10cm_wsm6(S,local+nz,local+2*nz,local+3*nz,local+4*nz,local,local+5*nz,radar,1,nz);
 int kp=0;float maximum=-35.f;
 for(int k=0;k<nz;++k) {
  float dbz=mp_max(radar[k],-35.f);refl[k*ns+i]=dbz;maximum=mp_max(maximum,dbz);radar[k]=dbz;
  float zp=add(sub(zgrid[k*ns+i],zgrid[i]),mul(.5f,sub(zgrid[(k+1)*ns+i],zgrid[k*ns+i])));
  if(zp<1000.f)kp=k;
 }
 refl_max[i]=maximum;
 if(kp+1<nz) {
  float z1=add(sub(zgrid[kp*ns+i],zgrid[i]),mul(.5f,sub(zgrid[(kp+1)*ns+i],zgrid[kp*ns+i])));
  float z2=add(sub(zgrid[(kp+1)*ns+i],zgrid[i]),mul(.5f,sub(zgrid[(kp+2)*ns+i],zgrid[(kp+1)*ns+i])));
  float w1=div(sub(z2,1000.f),sub(z2,z1)),w2=sub(1.f,w1);
  refl_1km[i]=add(mul(w1,radar[kp]),mul(w2,radar[kp+1]));
 } else {
  refl_1km[i]=radar[kp];  // every level centre below 1 km: native reads an unassigned kp+1 word here (:711-727)
 }
}
'''


def source(device_math_header: str, nz: int, *, numerics: str | None = None) -> str:
    """The start-radar compile unit: WSM6's arithmetic, the shared device maths and the audited modules, plus the
    three entries above.  Only code reachable from them is compiled (no microphysics column, no driver)."""
    from . import wsm6_v842 as wsm6
    from .. import numerics as _hex_numerics
    arithmetic = wsm6.CUDA_ARITHMETIC.replace('#define WSM6_KMAX 256', f'#define WSM6_KMAX {max(2, int(nz))}')
    mode = _hex_numerics.mode() if numerics is None else _hex_numerics.normalise(numerics)
    header = device_math_header
    if mode == 'fast':
        arithmetic = wsm6._fast_arithmetic(arithmetic)
        header = wsm6._fast_maths(header)
    return arithmetic.replace('/* WSM6_SHARED_DEVICE_MATH */', header) + wsm6.CUDA_MODULES + START_RADAR_SOURCE


def device_math_header() -> str:
    from .. import rkind_libm
    return rkind_libm.wsm6_device_math_header()


@lru_cache(maxsize=None)
def _module(device: int, nz: int, mode: str):
    """(template, sizes, radar) entries on ``device`` (one module per card, level count and numerics)."""
    import cupy as cp
    from . import wsm6_v842 as wsm6
    with cp.cuda.Device(device):
        return wsm6._compile_cuda(source(device_math_header(), nz, numerics=mode),
                                  ('wsm6_start_template', 'wsm6_start_sizes', 'wsm6_start_radar'))


def _current_mode() -> str:
    from .. import numerics as _hex_numerics
    return _hex_numerics.mode()


@lru_cache(maxsize=None)
def _sizes(device: int, nz: int, mode: str) -> tuple[int, int]:
    import cupy as cp
    _, sizes, _ = _module(device, nz, mode)
    with cp.cuda.Device(device):
        out = cp.zeros(2, cp.uint64)
        sizes((1,), (1,), (out,))
        workspace, state = (int(v) for v in out.get())
    return workspace, state


_TEMPLATES: dict = {}


def _template(device: int, nz: int, mode: str, constants: np.ndarray):
    """The initialised Wsm6State on ``device`` (state_parameters + mp_wsm6_init + radar_init, once)."""
    import cupy as cp
    host = np.asarray(constants, dtype=np.float32)
    key = (device, nz, mode, host.tobytes())
    if key not in _TEMPLATES:
        template, _, _ = _module(device, nz, mode)
        _, state_bytes = _sizes(device, nz, mode)
        with cp.cuda.Device(device):
            out = cp.empty(state_bytes, cp.uint8)
            err = cp.zeros(1, cp.int32)
            template((1,), (1,), (cp.asarray(host), np.int32(HAIL_OPT), out, err))
            value = int(err.get()[0])
        if value:
            raise RuntimeError(f'MPAS WSM6 init status {value}')
        _TEMPLATES[key] = out
    return _TEMPLATES[key]


def host_constants(constants: Mapping[str, Any]) -> np.ndarray:
    """The 26 WSM6 constant words, read from the native constants as wsm6_v842.bind_native reads them."""
    from .wsm6_v842 import NATIVE_CONSTANT_NAMES
    return np.asarray([constants[name] if name is not None else 0 for name in NATIVE_CONSTANT_NAMES], dtype=np.float32)


def warm_start_reflectivity(configs: Mapping[str, Any], nz: int, *, device: int) -> threading.Thread | None:
    """Compile the start-radar unit on a side thread from the moment the writer knows it will hold the start frame
    (None: not a WSM6 run).  Breakage it prevents: compiled at the f000 write, the compile outlasts step 1 and the
    forecast thread waits for it there (as tempo_refl_v303.warm_start_reflectivity)."""
    if str(configs.get('config_microp_scheme', '')).strip() not in SCHEMES:
        return None
    mode = _current_mode()

    def run() -> None:
        try:
            import cupy as cp
            with cp.cuda.Device(device):
                _module(int(device), int(nz), mode)
                _sizes(int(device), int(nz), mode)
        except BaseException:   # the f000 thread repeats the work and names any failure in the receipt
            pass

    thread = threading.Thread(target=run, name='f000-reflectivity-warm-wsm6', daemon=True)
    thread.start()
    return thread


def radar(state: Mapping[str, Any], zgrid: Any, *, constants: np.ndarray, cp: Any,
          device: int | None = None) -> dict[str, Any]:
    """``refl10cm`` ``(nz, ncells)``, ``refl10cm_max``, ``refl10cm_1km`` and ``refl10cm_1km_max`` (dBZ, device
    float32) of one state.  ``state``: ``qv qr qs qg theta p`` ``(nz, ncells)``; ``zgrid`` ``(nz + 1, ncells)``."""
    device = int(cp.cuda.Device().id if device is None else device)
    f = {k: cp.ascontiguousarray(cp.asarray(state[k], dtype=cp.float32)) for k in (*NEED, 'p')}
    nz, ncells = f['qv'].shape
    for name, value in f.items():
        if value.shape != (nz, ncells):
            raise ValueError(f'WSM6 start reflectivity input {name} is {value.shape}, not {(nz, ncells)}')
    z = cp.ascontiguousarray(cp.asarray(zgrid, dtype=cp.float32))
    if z.shape != (nz + 1, ncells):
        raise ValueError(f'zgrid is {z.shape}, not {(nz + 1, ncells)}')
    mode = _current_mode()
    _, _, kernel = _module(device, nz, mode)
    workspace_bytes, _ = _sizes(device, nz, mode)
    pii = ((f['p'].astype(cp.float64) / P0) ** (RGAS / CP_DRY)).astype(cp.float32)
    t = cp.ascontiguousarray(f['theta'] * pii)
    out = {'refl10cm': cp.empty((nz, ncells), cp.float32), 'refl10cm_max': cp.empty((ncells,), cp.float32),
           'refl10cm_1km': cp.empty((ncells,), cp.float32)}
    workspace = cp.empty(ncells * workspace_bytes, cp.uint8)
    tmpl = _template(device, nz, mode, constants)
    kernel(((ncells + 63) // 64,), (64,),
           (np.int32(nz), np.int32(ncells), tmpl, t, f['qv'], f['qr'], f['qs'], f['qg'], f['p'], z,
            out['refl10cm'], out['refl10cm_max'], out['refl10cm_1km'], workspace))
    # The running 1 km maximum from the floor is this state's own 1 km value.
    out['refl10cm_1km_max'] = out['refl10cm_1km'].copy()
    del workspace
    return out


def slab_columns(nz: int, *, device: int) -> int:
    workspace_bytes, _ = _sizes(int(device), int(nz), _current_mode())
    return max(64, (SLAB_WORKSPACE_MIB << 20) // workspace_bytes // 64 * 64)


def radar_staged(state: Mapping[str, np.ndarray], zgrid: np.ndarray, *, constants: np.ndarray, cp: Any,
                 stream: Any = None, columns: int | None = None) -> dict[str, np.ndarray]:
    """:func:`radar` of a HOST state one slab at a time (SLAB_WORKSPACE_MIB); host float32 results, the same words as
    one whole-state call (every column is formed from its own words only)."""
    nz, ncells = state['qv'].shape
    device = int(cp.cuda.Device().id)
    step = max(1, int(columns or slab_columns(nz, device=device)))
    out = {'refl10cm': np.empty((nz, ncells), np.float32)}
    out.update({k: np.empty(ncells, np.float32) for k in START_TARGETS[1:]})
    for first in range(0, ncells, step):
        cols = slice(first, min(ncells, first + step))
        slab = {k: np.ascontiguousarray(v[:, cols]) for k, v in state.items()}
        part = radar(slab, np.ascontiguousarray(zgrid[:, cols]), constants=constants, cp=cp, device=device)
        for name, value in part.items():
            out[name][..., cols] = cp.asnumpy(value, stream=stream)
        if stream is not None:
            stream.synchronize()
        del part, slab
    return out


class StartFrameReflectivity(_thompson.StartFrameReflectivity):
    """The held f000 frame's WSM6 reflectivity fields, formed from its own start state beside step 1."""

    NEED = NEED
    THREAD = 'f000-reflectivity-wsm6'
    NEEDS_ZGRID = True

    def __init__(self, snapshot: Mapping[str, Any], *, scheme: str, device: int, constants: np.ndarray) -> None:
        self.constants = np.asarray(constants, dtype=np.float32)
        super().__init__(snapshot, scheme=scheme, device=device)
        self.record['operator'] = (f'{scheme} refl10cm_wsm6 of the start state (wsm6_refl_start_v842 '
                                   'wsm6_start_radar: mp_wsm6.F90:2275-2442 + the wsm6_radar_to_mpas maximum and 1 km)')

    def _form(self, state, zgrid, cp, stream) -> dict[str, np.ndarray]:
        self.record['card_slab_columns'] = slab_columns(state['qv'].shape[0], device=int(cp.cuda.Device().id))
        return radar_staged(state, zgrid, constants=self.constants, cp=cp, stream=stream)


def start_frame_reflectivity(configs: Mapping[str, Any], snapshot: Mapping[str, Any], *, device: int,
                             constants: Mapping[str, Any] | None = None) -> StartFrameReflectivity | None:
    """The f000 reflectivity completion for this run's microphysics (None: not a WSM6 run)."""
    scheme = str(configs.get('config_microp_scheme', '')).strip()
    if scheme not in SCHEMES:
        return None
    if constants is None:
        raise RuntimeError('the WSM6 start reflectivity needs the native constants (rho_a, rho_w, cliq, cpv)')
    return StartFrameReflectivity(snapshot, scheme=scheme, device=device, constants=host_constants(constants))
