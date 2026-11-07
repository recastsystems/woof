"""GPU TEMPO microphysics runtime: blob loading, NVRTC module, 3-D launch.

TEMPO (NCAR/TEMPO 9adf9ef, tempo_v3.0.3, the RRFS v2 pin; Apache-2.0, see
kernels/tempo/NOTICE-TEMPO.txt) runs column by column.  This module owns
the device side an engine adapter needs:

* ``TempoBlob`` layout parsed from ``kernels/tempo/tempo_common.cuh`` (the
  single source of the layout) and filled from a named-record blob file
  written by ``tools/tempo_ref/tempo_ref_driver.F90`` (TEMPO's own
  ``tempo_init`` state);
* one NVRTC module with a pack kernel ((nz, ny, nx) fields -> per-column
  float64 state), the TEMPO column kernel and an unpack kernel;
* ``TempoRuntime.step``: one microphysics call on whole 3-D fields, in
  column batches sized so the per-column work area fits ``max_work_bytes``.

The float32 model adapter is :mod:`woof.core.tempo` (mp_physics=900).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cupy as cp
import numpy as np

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "tempo"


def _source_fingerprint(directory=KERNEL_DIR):
    """Bind the compiled-cache key to every included TEMPO header."""
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.cuh")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update((directory.parent / "glibc_flt64.cuh").read_bytes())
    return digest.hexdigest()[:16]

# Column state order inside the kernel's ColState (matches the harness).
STATE_FIELDS = ("t", "p", "w", "dz", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr",
                "nc", "nwfa", "nifa", "ng", "qb")
# Fields the scheme updates (copied back by unpack).
UPDATED_FIELDS = ("t", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa",
                  "nifa", "ng", "qb")
SURFACE_DIAGS = ("rain_precip", "ice_liquid_equiv_precip", "snow_liquid_equiv_precip",
                 "graupel_liquid_equiv_precip", "frz_rain_precip", "frozen_fraction",
                 "max_hail_diameter_sfc", "max_hail_diameter_column")
COLUMN_DIAGS = ("refl10cm", "re_cloud", "re_ice", "re_snow")

CTX_DTYPE = np.dtype([
    ("global_dt", "<f8"), ("global_inverse_dt", "<f8"), ("aerosolaware_flag", "<i4"),
    ("hailaware_flag", "<i4"), ("semi_sedi_flag", "<i4"),
    ("refl10cm_from_melting_flag", "<i4"), ("turn_off_micro_flag", "<i4"),
    ("cloud_condensation_flag", "<i4"), ("refl10cm_flag", "<i4"), ("re_cloud_flag", "<i4"),
    ("re_ice_flag", "<i4"), ("re_snow_flag", "<i4"), ("max_hail_diameter_flag", "<i4"),
    ("first_call_main", "<i4")])

_KERNEL = r"""
#include "tempo_column.cuh"
#include "tempo_cloud_ml.cuh"
#include "../glibc_flt32.cuh"
#define NFLD 17
struct ColState { double f[NFLD][TEMPO_NZP]; };

// fields: NFLD device pointers to (nz, ny, nx) arrays of T, column = j*nx+i
template <typename T>
__device__ void pack_impl(const T* const* fields, int nz, int ncol_total, int col0, int ncol,
                          ColState* S) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncol) return;
  long long g = col0 + c;
  for (int j = 0; j < NFLD; ++j) {
    const T* a = fields[j];
    S[c].f[j][0] = 0.0;
    for (int k = 1; k <= nz; ++k) S[c].f[j][k] = (double)a[(long long)(k - 1) * ncol_total + g];
    for (int k = nz + 1; k < TEMPO_NZP; ++k) S[c].f[j][k] = 0.0;
  }
}
template <typename T>
__device__ void unpack_impl(T* const* fields, const int* which, int nwhich, int nz,
                            int ncol_total, int col0, int ncol, const ColState* S,
                            const TempoWork* W, double* sfc, double* coldiag) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncol) return;
  long long g = col0 + c;
  for (int m = 0; m < nwhich; ++m) {
    T* a = fields[m];
    int j = which[m];
    for (int k = 1; k <= nz; ++k) a[(long long)(k - 1) * ncol_total + g] = (T)S[c].f[j][k];
  }
  const TempoWork& w = W[c];
  double mh = w.max_hail_diameter[1];
  for (int k = 2; k <= nz; ++k) mh = w.max_hail_diameter[k] > mh ? w.max_hail_diameter[k] : mh;
  double s[8] = {w.rain_precip, w.ice_liquid_equiv_precip, w.snow_liquid_equiv_precip,
                 w.graupel_liquid_equiv_precip, w.frz_rain_precip, w.frozen_fraction,
                 w.max_hail_diameter[1], mh};
  for (int m = 0; m < 8; ++m) sfc[(long long)m * ncol_total + g] = s[m];
  for (int k = 1; k <= nz; ++k) {
    long long o = (long long)(k - 1) * ncol_total + g;
    long long plane = (long long)nz * ncol_total;
    coldiag[o] = w.refl10cm[k];
    coldiag[plane + o] = w.re_cloud[k];
    coldiag[2 * plane + o] = w.re_ice[k];
    coldiag[3 * plane + o] = w.re_snow[k];
  }
}
extern "C" __global__ void tempo_pack_f32(const float* const* f, int nz, int nt, int c0, int n, ColState* S) { pack_impl<float>(f, nz, nt, c0, n, S); }
extern "C" __global__ void tempo_pack_f64(const double* const* f, int nz, int nt, int c0, int n, ColState* S) { pack_impl<double>(f, nz, nt, c0, n, S); }
extern "C" __global__ void tempo_unpack_f32(float* const* f, const int* wh, int nw, int nz, int nt, int c0, int n, const ColState* S, const TempoWork* W, double* sfc, double* cd) { unpack_impl<float>(f, wh, nw, nz, nt, c0, n, S, W, sfc, cd); }
extern "C" __global__ void tempo_unpack_f64(double* const* f, const int* wh, int nw, int nz, int nt, int c0, int n, const ColState* S, const TempoWork* W, double* sfc, double* cd) { unpack_impl<double>(f, wh, nw, nz, nt, c0, n, S, W, sfc, cd); }

// Host float32 theta/Exner and m3/kg volume, TEMPO float64 column arithmetic.
// Keep the temperature and volume intermediates in the column, as the
// public tempo_run(theta, pii) interface does. Round only on host writeback.
extern "C" __global__ void tempo_pack_adapter(const float* const* f, const float* pii,
        const float* emission, double dt,
        int nz, int nt, int c0, int n, ColState* S) {
  pack_impl<float>(f, nz, nt, c0, n, S);
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  for (int k = 1; k <= nz; ++k) {
    long long o = (long long)(k - 1) * nt + c0 + c;
    S[c].f[0][k] *= (double)pii[o];
    S[c].f[16][k] *= 1000.0;
  }
  if (emission) S[c].f[13][1] = __dadd_rn(S[c].f[13][1],
                          __dmul_rn((double)emission[c0+c], dt));
}
extern "C" __global__ void tempo_unpack_adapter(float* const* f, const int* wh, int nw,
        const float* pii, const float* emission, double dt,
        int nz, int nt, int c0, int n, ColState* S, const TempoWork* W,
        double* sfc, double* cd) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  // tempo_aerosol_surface_emissions after tempo_run, before float32 storage.
  if (emission) S[c].f[13][1] = __dadd_rn(S[c].f[13][1],
                          __dmul_rn((double)emission[c0+c], dt));
  for (int k = 1; k <= nz; ++k) {
    long long o = (long long)(k - 1) * nt + c0 + c;
    S[c].f[0][k] /= (double)pii[o];
    S[c].f[16][k] *= 1.0e-3;
  }
  unpack_impl<float>(f, wh, nw, nz, nt, c0, n, S, W, sfc, cd);
}
extern "C" __global__ void tempo_adapter_exner(const float* p, float* pii, int n,
                                               float p0, float rcp) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) pii[i] = gfk_pow(__fdiv_rn(p[i], p0), rcp);
}

extern "C" __global__ void tempo_columns(const double* blob, TempoCtx C, int ncol, int nz,
                                         int first_global_col, ColState* S, TempoWork* W,
                                         const float* qc_bl, const float* fraction_bl, int nt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ncol) return;
  TempoParams P{reinterpret_cast<const TempoBlob*>(blob)};
  TempoCtx c = C;
  if (first_global_col + i != 0) c.first_call_main = 0;  // Fortran SAVE flag: first column only
  ColState& s = S[i];
  tempo_main_column(P, c, nz, W[i], s.f[4], s.f[5], s.f[7], s.f[6], s.f[8], s.f[9], s.f[16],
                    s.f[10], s.f[11], s.f[12], s.f[15], s.f[13], s.f[14], s.f[0], s.f[1],
                    s.f[2], s.f[3]);
  if (qc_bl && fraction_bl)
    tempo_subgrid_cloud_radii(P, c, nz, W[i], s.f[5], s.f[6], s.f[7], s.f[8],
                             s.f[12], s.f[2], qc_bl, fraction_bl, nt, first_global_col+i);
}
// Initial-state radar diagnosis: prepare only the native PSDs, then call
// the same reflectivity_10cm used by tempo_main_column. Packed fields are
// private copies; no tendencies, sedimentation or model writes occur.
extern "C" __global__ void tempo_initial_refl(const double* blob, TempoCtx C,
    int n, int nz, int c0, int nt, ColState* S, TempoWork* work, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  TempoParams P{reinterpret_cast<const TempoBlob*>(blob)};
  ColState& s = S[i];
  TempoWork& W = work[i];
  for (int k = 1; k <= nz; ++k) {
    W.temp[k] = s.f[0][k];
    W.qv[k] = tmax(PS(min_qv), s.f[4][k]);
    W.pres[k] = s.f[1][k];
    W.rho[k] = PS(roverrv) * W.pres[k] /
      (PS(rdry) * W.temp[k] * (W.qv[k] + PS(roverrv)));
    W.refl10cm[k] = -35.0;
  }
  rain_check_and_update(P, C, nz, W.rho, W.l_qr, s.f[6], s.f[11],
    W.rr, W.nr, W.qrten, W.nrten, W.ilamr, W.mvd_r);
  snow_check_and_update(P, C, nz, W.rho, W.l_qs, s.f[8], W.rs, W.qsten);
  double sq = 0.0, sn = 0.0, sb = 0.0;
  for (int k = 1; k <= nz; ++k) {
    sq += s.f[9][k]; sn += s.f[15][k]; sb += s.f[16][k];
    if (W.l_qs[k]) {
      double tc = tmin(-0.1, W.temp[k] - PS(t0));
      snow_moments(P, W.rs[k], tc, &W.smob[k], &W.smoc[k], nullptr,
        nullptr, nullptr, &W.smo2[k], nullptr, nullptr, nullptr, &W.smoz[k]);
    }
  }
  if (sq > PS(r1) && sn < PS(eps) && sb < PS(eps))
    graupel_init(P, nz, W.rho, s.f[9], s.f[15], s.f[16]);
  graupel_check_and_update(P, C, nz, W.rho, W.l_qg, s.f[9], s.f[15], s.f[16],
    W.rg, W.ng, W.rb, W.idx_bg, W.qgten, W.ngten, W.qbten, W.ilamg, W.mvd_g);
  reflectivity_10cm(P, C, nz, C.refl10cm_from_melting_flag, W.temp,
    W.l_qr, W.rr, W.nr, W.ilamr, W.l_qs, W.rs, W.smoc, W.smob, W.smoz,
    W.l_qg, W.rg, W.ng, W.idx_bg, W.ilamg, W.refl10cm);
  for (int k = 1; k <= nz; ++k)
    out[(long long)(k-1)*nt+c0+i] = (float)W.refl10cm[k];
}
extern "C" __global__ void tempo_sizes(long long* out) {
  out[0] = sizeof(TempoWork); out[1] = sizeof(ColState); out[2] = sizeof(TempoCtx);
  out[3] = TEMPO_NZ_MAX;
}
// NOAA MPAS 1d6e370e mpas_atmphys_init_tempo.F:172-178. Input climatology
// is retained; bounds apply to its emissions only. Host storage is float32.
extern "C" __global__ void tempo_mpas_emission_init(const double* blob,
    const float* nwfa, const double* depth, const double* area, float* emission, int n) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= n) return;
  TempoParams P{reinterpret_cast<const TempoBlob*>(blob)};
  double dz = tmax(9.0, (double)depth[i]);
  emission[i] = (float)(tmax(PS(nwfa_default), tmin(PS(aero_max), (double)nwfa[i]))
                      * 0.000196 * (5.0/dz) * (area[i]/9.0e6));
}
"""


def blob_layout(header_text: str) -> list[tuple[str, int]]:
    """(name, count) in TEMPO_FIELDS order from tempo_common.cuh."""
    body = header_text.split("#define TEMPO_FIELDS(TX)", 1)[1].split("struct TempoBlob", 1)[0]
    out = []
    for m in re.finditer(r"TX\((\w+),(\d+),(\d+),(\d+),(\d+),(\d+)\)", body.replace(" ", "")):
        n = 1
        for d in m.groups()[1:]:
            n *= int(d)
        out.append((m.group(1), n))
    if not out:
        raise RuntimeError("tempo_common.cuh: TEMPO_FIELDS not found")
    return out


def load_blob(path: str | Path, layout: list[tuple[str, int]]) -> np.ndarray:
    """Read a named-record blob (48-byte name, int64 count, float64 data)."""
    offs, total = {}, 0
    for name, n in layout:
        offs[name] = (total, n)
        total += n
    blob = np.zeros(total, dtype=np.float64)
    seen = set()
    with open(path, "rb") as f:
        while True:
            nm = f.read(48)
            if len(nm) < 48:
                break
            name = nm.decode().strip()
            n = int(np.frombuffer(f.read(8), dtype="<i8")[0])
            if name not in offs:
                raise ValueError(f"TEMPO blob {path}: unknown field {name}")
            o, cnt = offs[name]
            if cnt != n:
                raise ValueError(f"TEMPO blob {path}: {name} has {n} values, layout says {cnt}")
            blob[o:o + n] = np.frombuffer(f.read(8 * n), dtype="<f8")
            seen.add(name)
    missing = [n for n, _ in layout if n not in seen]
    if missing:
        raise ValueError(f"TEMPO blob {path}: missing fields {missing}")
    return blob


def build_blob(table_dir: str | Path, *, hailaware: bool = True,
               refl10cm_from_melting: bool = False,
               max_hail_diameter: bool = True) -> np.ndarray:
    """TEMPO's tempo_init state from the Rust port (woof.core.tempo_rust),
    in TEMPO_FIELDS order; bitwise TEMPO's own Fortran init
    (tests/test_tempo_rust_init.py)."""
    from woof.core import tempo_rust

    blob = tempo_rust.build_blob(table_dir, hailaware=hailaware,
                                 refl10cm_from_melting=refl10cm_from_melting,
                                 max_hail_diameter=max_hail_diameter)
    if tempo_rust.layout() != blob_layout((KERNEL_DIR / "tempo_common.cuh").read_text()):
        raise RuntimeError("rw-tempo blob layout differs from tempo_common.cuh; rebuild rw-tempo")
    return blob


@dataclass(frozen=True)
class TempoConfig:
    """ty_tempo_cfgs flags.  Defaults are TEMPO's (and the RRFS v2 profile)."""
    aerosolaware: bool = True
    hailaware: bool = True
    semi_sedi: bool = False
    refl10cm_from_melting: bool = False
    cloud_condensation: bool = True
    refl10cm: bool = True
    re_cloud: bool = True
    re_ice: bool = True
    re_snow: bool = True
    max_hail_diameter: bool = True
    ml_for_bl_nc: bool = False
    surface_emission_before: bool = False


class TempoRuntime:
    """Device blob plus compiled module; one ``step`` per microphysics call."""

    def __init__(self, blob_path: str | Path | None = None, *, table_dir: str | Path | None = None,
                 hailaware: bool = True, nz_max: int = 100,
                 max_work_bytes: int = 4 << 30, fmad: bool = False):
        """``table_dir`` (the four TEMPO table files) builds the blob through
        the Rust tempo_init (:func:`build_blob`); ``blob_path`` reads a
        named-record blob file instead (the Fortran reference dump)."""
        header = (KERNEL_DIR / "tempo_common.cuh").read_text()
        self.layout = blob_layout(header)
        if (blob_path is None) == (table_dir is None):
            raise ValueError("TempoRuntime needs exactly one of blob_path or table_dir")
        if table_dir is not None:
            self.blob = cp.asarray(build_blob(table_dir, hailaware=hailaware))
        else:
            self.blob = cp.asarray(load_blob(blob_path, self.layout))
        opts = ["-std=c++17", f"-I{KERNEL_DIR}", f"-DTEMPO_NZ_MAX={int(nz_max)}"]
        # NVRTC/CuPy keys the cache on code and options, not included bytes.
        # A corrected header must never replay an older compiled kernel.
        opts.append(f"-DTEMPO_SOURCE_FINGERPRINT=0x{_source_fingerprint()}")
        if not fmad:
            opts.append("--fmad=false")
        self.module = cp.RawModule(code=_KERNEL, options=tuple(opts))
        sizes = cp.zeros(4, dtype=cp.int64)
        self.module.get_function("tempo_sizes")((1,), (1,), (sizes,))
        self.work_bytes, self.col_bytes, ctx_bytes, self.nz_max = (int(x) for x in sizes.get())
        if ctx_bytes != CTX_DTYPE.itemsize:
            raise RuntimeError(f"TempoCtx size {ctx_bytes} != host {CTX_DTYPE.itemsize}")
        per_col = self.work_bytes + self.col_bytes
        self.batch = max(1, int(max_work_bytes // per_col))
        self.first_call = True
        self._work = None
        self._cols = None

    def _buffers(self, n):
        if self._work is None or self._work.size * 8 < n * self.work_bytes:
            self._work = cp.zeros(n * self.work_bytes // 8, dtype=cp.float64)
            self._cols = cp.zeros(n * self.col_bytes // 8, dtype=cp.float64)
        return self._cols, self._work

    def exner(self, pressure, output):
        """Float32 host Exner, using WOOF's own power routine in both modes."""
        from woof.core import constants as c
        if (pressure.dtype != np.float32 or output.dtype != np.float32
                or pressure.shape != output.shape or not pressure.flags.c_contiguous
                or not output.flags.c_contiguous):
            raise ValueError("TEMPO Exner requires matching contiguous float32 arrays to prevent incorrect addressing")
        n = pressure.size
        self.module.get_function("tempo_adapter_exner")(((n + 127) // 128,), (128,),
            (pressure, output, np.int32(n), np.float32(c.P0), np.float32(c.RCP)))

    def reflectivity(self, state, cfg):
        """Current-state native radar kernel, with bounded column work storage."""
        nz, ny, nx = state.p.shape
        thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
        theta = cp.ascontiguousarray(state.total_theta() if getattr(state, "use_theta_m", 0)
                              else thb + state.thp)
        exner = cp.empty_like(state.p)
        self.exner(state.p, exner)
        # Only temperature, pressure, qv and precipitating moments are read.
        fields = {"t": theta, "p": state.p, "qv": state.qv,
                  "qr": state.qr, "nr": state.nr, "qs": state.qs,
                  "qg": state.qg, "ng": state.ng, "qb": state.volg}
        ptrs = cp.asarray(np.array([fields.get(n, state.qv).data.ptr
                                    for n in STATE_FIELDS], dtype=np.uint64))
        ctx = np.zeros(1, dtype=CTX_DTYPE)
        ctx["global_dt"] = float(cfg.dt)
        ctx["global_inverse_dt"] = 1 / float(cfg.dt)
        ctx["hailaware_flag"] = 1
        ctx["refl10cm_from_melting_flag"] = int(
            getattr(state, "refl10cm_from_melting", False))
        output = state.scratch((nz, ny, nx), "refl_10cm")
        nt = ny * nx
        pack = self.module.get_function("tempo_pack_adapter")
        kernel = self.module.get_function("tempo_initial_refl")
        for c0 in range(0, nt, self.batch):
            n = min(self.batch, nt - c0)
            cols, work = self._buffers(n)
            work.fill(0)
            grid = ((n + 63) // 64,)
            dims = (np.int32(nz), np.int32(nt), np.int32(c0), np.int32(n))
            pack(grid, (64,), (ptrs, exner, np.uint64(0), np.float64(cfg.dt),
                              *dims, cols))
            kernel(grid, (64,), (self.blob, ctx, np.int32(n), np.int32(nz),
                                np.int32(c0), np.int32(nt), cols, work, output))
        return output

    def initialize_mpas_surface_emission(self, nwfa, depth, area, output):
        depth = cp.ascontiguousarray(depth, dtype=cp.float64)
        if (nwfa.dtype != np.float32 or output.dtype != np.float32
                or area.dtype != np.float64 or nwfa.size != area.size
                or depth.size != area.size or output.size != area.size):
            raise ValueError("TEMPO surface emission needs matching aerosol, depth and cell-area columns")
        n = area.size
        self.module.get_function("tempo_mpas_emission_init")(((n+127)//128,), (128,),
            (self.blob, nwfa, depth, area, output, np.int32(n)))

    def step(self, fields: dict, dt: float, cfg: TempoConfig = TempoConfig(), *,
             exner=None, surface_emission=None, qc_bl=None, fraction_bl=None) -> dict:
        """Run TEMPO on (nz, ny, nx) fields in place.

        ``fields`` maps every name in STATE_FIELDS to a C-contiguous cupy
        array (float32 or float64, all the same dtype): temperature in K,
        pressure in Pa, w and dz on mass levels, mixing ratios in kg/kg,
        numbers per kg, qb in L/kg.  The UPDATED_FIELDS are written back.
        Returns surface diagnostics (ny, nx) and column diagnostics
        (nz, ny, nx) as float64 cupy arrays.

        With ``exner`` present, float32 host fields instead hold theta and
        graupel volume in m3/kg. Conversion takes place inside the float64
        column. ``surface_emission`` is nwfa2d; the public TEMPO emission
        call runs after physics and before rounding nwfa to host storage.
        """
        arrs = [fields[n] for n in STATE_FIELDS]
        dtype = arrs[0].dtype
        if any(a.dtype != dtype or not a.flags.c_contiguous for a in arrs):
            raise ValueError("TEMPO fields must share one dtype and be C-contiguous")
        nz, ny, nx = arrs[0].shape
        adapter = exner is not None
        if adapter and (dtype != np.float32 or exner.dtype != dtype
                        or exner.shape != arrs[0].shape or not exner.flags.c_contiguous):
            raise ValueError("TEMPO theta interface requires contiguous float32 Exner matching host fields")
        if surface_emission is not None and (not adapter or surface_emission.dtype != dtype
                or surface_emission.shape != (ny, nx) or not surface_emission.flags.c_contiguous):
            raise ValueError("TEMPO emission requires the float32 theta interface and matching surface shape")
        if cfg.ml_for_bl_nc:
            for array in (qc_bl, fraction_bl):
                if (not adapter or array is None or array.dtype != np.float32
                        or array.shape != arrs[0].shape or not array.flags.c_contiguous):
                    raise ValueError("TEMPO ML cloud radii require supplied contiguous float32 PBL cloud mass and fraction matching the host column; omitting them drops the requested cloud effect")
        if nz > self.nz_max:
            raise ValueError(f"nz={nz} exceeds TEMPO_NZ_MAX={self.nz_max}")
        ncol = ny * nx
        suffix = "f32" if dtype == np.float32 else "f64"
        pack = self.module.get_function("tempo_pack_adapter" if adapter else f"tempo_pack_{suffix}")
        unpack = self.module.get_function("tempo_unpack_adapter" if adapter else f"tempo_unpack_{suffix}")
        kern = self.module.get_function("tempo_columns")
        ptrs = cp.asarray(np.array([a.data.ptr for a in arrs], dtype=np.uint64))
        which = [STATE_FIELDS.index(n) for n in UPDATED_FIELDS]
        out_ptrs = cp.asarray(np.array([fields[n].data.ptr for n in UPDATED_FIELDS], dtype=np.uint64))
        which_d = cp.asarray(np.array(which, dtype=np.int32))
        sfc = cp.zeros((len(SURFACE_DIAGS), ny, nx), dtype=cp.float64)
        coldiag = cp.zeros((len(COLUMN_DIAGS), nz, ny, nx), dtype=cp.float64)

        ctx = np.zeros(1, dtype=CTX_DTYPE)
        ctx["global_dt"] = dt
        ctx["global_inverse_dt"] = 1.0 / dt
        ctx["aerosolaware_flag"] = int(cfg.aerosolaware)
        ctx["hailaware_flag"] = int(cfg.hailaware)
        ctx["semi_sedi_flag"] = int(cfg.semi_sedi)
        ctx["refl10cm_from_melting_flag"] = int(cfg.refl10cm_from_melting)
        ctx["cloud_condensation_flag"] = int(cfg.cloud_condensation)
        ctx["refl10cm_flag"] = int(cfg.refl10cm)
        ctx["re_cloud_flag"] = int(cfg.re_cloud)
        ctx["re_ice_flag"] = int(cfg.re_ice)
        ctx["re_snow_flag"] = int(cfg.re_snow)
        ctx["max_hail_diameter_flag"] = int(cfg.max_hail_diameter)
        ctx["first_call_main"] = int(self.first_call)

        tpb = 64
        for c0 in range(0, ncol, self.batch):
            n = min(self.batch, ncol - c0)
            cols, work = self._buffers(n)
            grid = ((n + tpb - 1) // tpb,)
            dims = (np.int32(nz), np.int32(ncol), np.int32(c0), np.int32(n))
            before = surface_emission if cfg.surface_emission_before else None
            after = None if cfg.surface_emission_before else surface_emission
            pack_args = (ptrs, exner, before if before is not None else np.uint64(0),
                         np.float64(dt), *dims, cols) if adapter else (ptrs, *dims, cols)
            pack(grid, (tpb,), pack_args)
            kern(grid, (tpb,), (self.blob, ctx, np.int32(n), np.int32(nz), np.int32(c0), cols, work,
                qc_bl if cfg.ml_for_bl_nc else np.uint64(0),
                fraction_bl if cfg.ml_for_bl_nc else np.uint64(0), np.int32(ncol)))
            unpack_args = (out_ptrs, which_d, np.int32(len(which)))
            if adapter:
                unpack_args += (exner, after if after is not None else np.uint64(0),
                                np.float64(dt))
            unpack(grid, (tpb,), (*unpack_args, *dims, cols, work, sfc, coldiag))
        self.first_call = False
        return {"surface": dict(zip(SURFACE_DIAGS, sfc)), "column": dict(zip(COLUMN_DIAGS, coldiag))}


#: One quoted ``#include "..."`` line.  System includes (``<cmath>``) are left
#: for NVRTC, exactly as the ``-I`` compile leaves them.
_QUOTED_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]+"([^"]+)"[ \t]*$', re.M)


def composed_source(code: str = _KERNEL, kernel_dir: Path = KERNEL_DIR) -> str:
    """``code`` with every quoted ``#include`` replaced by the header's text.

    Each TEMPO header is ``#pragma once``, so expanding a header at its first
    include and dropping its later includes is what the preprocessor does with
    ``-I{kernel_dir}``.  Quoted paths resolve against ``kernel_dir``, as every
    TEMPO include is written (``"tempo_common.cuh"``, ``"../glibc_flt64.cuh"``).

    The runtime compiles ``_KERNEL`` through ``-I`` and binds its compile cache
    to the header bytes with ``TEMPO_SOURCE_FINGERPRINT``.  This one-string
    form is for the checks that compile a TEMPO probe without an include path
    (``tests/test_tempo_device_libm.py``,
    ``tools/tempo_ref/oracle/refute/libm_edge_probe.py``);
    ``tests/test_tempo_runtime_gpu.py`` proves it compiles to the same PTX as
    the ``-I`` build, so what those checks measure is the code the runtime runs.
    """

    seen: set[str] = set()

    def expand(source: str) -> str:
        def one(match: re.Match) -> str:
            name = match.group(1)
            if name in seen:
                return ""
            seen.add(name)
            return expand((kernel_dir / name).read_text(encoding="utf-8"))
        return _QUOTED_INCLUDE.sub(one, source)

    return expand(code)
