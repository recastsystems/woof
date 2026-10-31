"""TEMPO sedimentation with fall speeds fixed for the call, for fast numerics only (tempo-yolo sedimentation).

The source owner applies :func:`fast_sed_source` to the column support before the fast-dp and fast-maths
transforms, never in strict mode, so the strict source stays byte for byte the one every identity proof
measured (TEMPO module replay 0 words, x1.2562 6 h strict 0 of 29,644,818 against native).

What changes in fast mode, warp shape only (the serial column shape keeps the native text):

* each falling species computes its fall speeds, substep count and ktop ONCE per call, then runs every
  substep's flux update with those speeds (Thompson's shape). Native TEMPO repeats check_and_update and the
  fall speeds (pows over every level) after every substep; with hail-aware graupel over thin near-ground
  layers that was 19-24 % of TEMPO at 3 km. The final check_and_update after sedimentation still enforces
  every number and size limit, as before;
* the flux loop keeps the species' fields, speeds and accumulated net flux in registers (level k on lane
  (k-1)%32, row (k-1)/32); the flux from level k+1 arrives by warp shuffle, so a substep has no shared-array
  round trip and no __syncwarp. Tendencies and the surface precipitation are added once at the end;
* a species with no mass in the column is skipped, as before (tempo_any), so columns with nothing falling
  do no sedimentation work at all.

Unchanged: the order of species (rain, graupel, snow, ice, cloud: snow's melting blend reads the rain left
by rain's sedimentation), each species' own substep count and ktop, native ice's single pass of its
steps-divided flux, cloud's 500 m ktop, the precipitation threshold and limits.

THE BREAKAGE THIS ENDS: at 3 km sedimentation was 31-38 % of TEMPO's warp cycles (graupel alone 19-24 %),
almost all of it the per-substep recompute. Fast words change by design; the fast referee is the forecast
by eye against native and MRMS plus domain stats, not words.
"""
from __future__ import annotations

_START = "  // sedimentation: rain\n"
_END = "  if (!C.turn_off_micro_flag) {\n    freeze_cloud_melt_ice("
_HELPER_AT = "TEMPO_HD float tempo_cloud_number(float,float,float,float,float,float,float);\n"
_TAG = "tempo_fastsed_v303"

_HELPER = r"""#if TEMPO_WARP
// Fast numerics (tempo_fastsed_v303): one species' flux sedimentation with its fall speeds fixed for the
// call.  Level k lives on lane (k-1)%32, row (k-1)/32, in registers; the flux of level k+1 arrives by
// shuffle.  passes = substep iterations; steps = the substep divisor (native ice: one pass, steps-divided).
// Levels k <= ktop take inflow and outflow, level nz outflow only, the rest nothing (native sedimentation).
template<int NF>
TEMPO_HD void tempo_fsed(const TempoParams& P, const TempoCtx& C, int nz, const float* dz1d,
    const float* rho, int steps, int passes, int ktop, float* const (&xs)[NF], float* const (&tens)[NF],
    const float* const (&vts)[NF], const float (&lims)[NF], float* precip) {
  const int lane = TEMPO_LANE;
  const float ostep = 1.0f / (float)steps;
  const float cstep = C.global_dt * ostep;
  float x[NF][TEMPO_ROWS], v[NF][TEMPO_ROWS], acc[NF][TEMPO_ROWS], cf[TEMPO_ROWS];
  bool gin[TEMPO_ROWS], gout[TEMPO_ROWS];
  TEMPO_SYNC();
#pragma unroll
  for (int r = 0; r < TEMPO_ROWS; ++r) {
    const int k = (r << 5) + lane + 1;
    const bool live = k <= nz;
    gin[r] = live && k <= ktop;
    gout[r] = gin[r] || k == nz;
    cf[r] = live ? cstep / dz1d[k] : 0.0f;
#pragma unroll
    for (int j = 0; j < NF; ++j) {
      x[j][r] = live ? xs[j][k] : 0.0f;
      v[j][r] = live ? vts[j][k] : 0.0f;
      acc[j][r] = 0.0f;
    }
  }
  const float lowp = PS(low_limit_mass_for_precip);
  float pr = 0.0f;
  for (int n = 0; n < passes; ++n) {
    float f1 = 0.0f;
#pragma unroll
    for (int j = 0; j < NF; ++j) {
      float f[TEMPO_ROWS];
#pragma unroll
      for (int r = 0; r < TEMPO_ROWS; ++r) f[r] = v[j][r] * x[j][r];
      if (j == 0) f1 = f[0];
#pragma unroll
      for (int r = 0; r < TEMPO_ROWS; ++r) {
        const float dn = __shfl_down_sync(TEMPO_FULL, f[r], 1);
        const float wr = __shfl_sync(TEMPO_FULL, (r + 1 < TEMPO_ROWS) ? f[(r + 1 < TEMPO_ROWS) ? r + 1 : r] : 0.0f, 0);
        const float fin = gin[r] ? (lane < 31 ? dn : wr) : 0.0f;
        if (gout[r]) {
          const float d = fin - f[r];
          x[j][r] = tmax(lims[j], x[j][r] + d * cf[r]);
          acc[j][r] += d;
        }
      }
    }
    if (precip != nullptr && lane == 0 && x[0][0] > lowp) pr += f1;
  }
#pragma unroll
  for (int r = 0; r < TEMPO_ROWS; ++r) {
    const int k = (r << 5) + lane + 1;
    if (gout[r]) {
      const float b = ostep / (dz1d[k] * rho[k]);
#pragma unroll
      for (int j = 0; j < NF; ++j) {
        tens[j][k] = tens[j][k] + acc[j][r] * b;
        xs[j][k] = x[j][r];
      }
    }
  }
  if (precip != nullptr && lane == 0) *precip = *precip + pr * cstep;
  TEMPO_SYNC();
}
#endif
"""

_FAST_BLOCK = r"""#if TEMPO_WARP
  // Fast numerics (tempo_fastsed_v303): fall speeds once per call, register flux substeps (tempo_fsed).
  {
    const bool has_qb1d = (qb1d != nullptr);
    const float rb_limit = PS(meters3_to_liters) * PS(r1) / PA1(rho_g, PI(nrhg));
    // rain
    if (tempo_any(W.l_qr, nz)) {
      ktop_sedi = 1;
      substeps_sedi = 1;
      rain_fallspeed(P, C, nz, W.rhof, W.l_qr, W.rr, W.ilamr, dz1d, W.vtrr, W.vtnr, &substeps_sedi, &ktop_sedi);
      float* const xs[2] = {W.rr, W.nr};
      float* const ts[2] = {W.qrten, W.nrten};
      const float* const vs[2] = {W.vtrr, W.vtnr};
      const float ls[2] = {PS(r1), PS(r2)};
      tempo_fsed<2>(P, C, nz, dz1d, W.rho, substeps_sedi, substeps_sedi, ktop_sedi, xs, ts, vs, ls, &W.rain_precip);
    }
    // graupel
    if (tempo_any(W.l_qg, nz)) {
      ktop_sedi = 1;
      substeps_sedi = 1;
      graupel_fallspeed(P, C, nz, W.rhof, W.rho, W.visco, W.l_qg, W.rg, W.rb, has_qb1d, W.idx_bg,
                        W.ilamg, dz1d, W.vtrg, W.vtng, &substeps_sedi, &ktop_sedi);
      float* const xs[3] = {W.rg, W.ng, W.rb};
      float* const ts[3] = {W.qgten, W.ngten, W.qbten};
      const float* const vs[3] = {W.vtrg, W.vtng, W.vtrg};
      const float ls[3] = {PS(r1), PS(r2), rb_limit};
      tempo_fsed<3>(P, C, nz, dz1d, W.rho, substeps_sedi, substeps_sedi, ktop_sedi, xs, ts, vs, ls,
                    &W.graupel_liquid_equiv_precip);
    }
    // snow
    if (tempo_any(W.l_qs, nz)) {
      ktop_sedi = 1;
      substeps_sedi = 1;
      snow_fallspeed(P, C, nz, W.rhof, W.l_qs, W.rs, T.prr_sml, W.smob, W.smoc, W.rr, W.vtrr, dz1d,
                     W.vtrs, W.vtboost, &substeps_sedi, &ktop_sedi);
      float* const xs[1] = {W.rs};
      float* const ts[1] = {W.qsten};
      const float* const vs[1] = {W.vtrs};
      const float ls[1] = {PS(r1)};
      tempo_fsed<1>(P, C, nz, dz1d, W.rho, substeps_sedi, substeps_sedi, ktop_sedi, xs, ts, vs, ls,
                    &W.snow_liquid_equiv_precip);
    }
    // ice: native applies ONE pass of its steps-divided flux
    if (tempo_any(W.l_qi, nz)) {
      ktop_sedi = 1;
      substeps_sedi = 1;
      ice_fallspeed(P, C, nz, W.rhof, W.l_qi, W.ri, W.ilami, dz1d, W.vtri, W.vtni, &substeps_sedi, &ktop_sedi);
      float* const xs[2] = {W.ri, W.ni};
      float* const ts[2] = {W.qiten, W.niten};
      const float* const vs[2] = {W.vtri, W.vtni};
      const float ls[2] = {PS(r1), PS(r2)};
      tempo_fsed<2>(P, C, nz, dz1d, W.rho, substeps_sedi, 1, ktop_sedi, xs, ts, vs, ls, &W.ice_liquid_equiv_precip);
    }
    // cloud: one step, ktop within the lowest 500 m
    if (tempo_any(W.l_qc, nz)) {
      ktop_sedi = 1;
      cloud_fallspeed(P, C, nz, W.rhof, w1d, W.l_qc, W.rc, W.nc, W.ilamc, dz1d, W.vtrc, W.vtnc, &ktop_sedi);
      float* const xs[2] = {W.rc, W.nc};
      float* const ts[2] = {W.qcten, W.ncten};
      const float* const vs[2] = {W.vtrc, W.vtnc};
      const float ls[2] = {PS(r1), PS(r2)};
      tempo_fsed<2>(P, C, nz, dz1d, W.rho, 1, 1, ktop_sedi, xs, ts, vs, ls, &W.cloud_precip);
    }
  }
#else
"""


def fast_sed_source(source: str) -> str:
    """Swap TEMPO's per-substep sedimentation for the fixed-speed register flux (warp shape, fast only).

    Accepts the column support alone or a module that contains it once. Fails on source drift (a marker
    missing or duplicated) or when applied twice. No environment or numerics side effects: the caller
    selects fast mode.
    """
    if _TAG in source:
        raise ValueError("TEMPO fast sedimentation: already applied")
    for marker in (_START, _END, _HELPER_AT):
        count = source.count(marker)
        if count != 1:
            raise ValueError(f"TEMPO fast sedimentation: expected one {marker.splitlines()[0]!r}, found {count}")
    a, b = source.index(_START), source.index(_END)
    if not a < b:
        raise ValueError("TEMPO fast sedimentation: block markers out of order")
    native = source[a:b]
    source = source[:a] + _FAST_BLOCK + native + "#endif\n\n" + source[b:]
    return source.replace(_HELPER_AT, _HELPER + _HELPER_AT)
