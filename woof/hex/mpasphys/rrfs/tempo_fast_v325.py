"""Fast numerics for the TEMPO v3.2.5 column (tempo_v325), the tempo_v303 fast layers applied to the new column.

Strict never imports this module: tempo_v325 builds its strict source from ``_COLUMN`` byte for byte. Fast
words change by design (the referee is the forecast by eye against native and observations, plus domain
statistics); the layers are tempo_v303's, unchanged, in its order:

1. register flux sedimentation (tempo_fastsed_v303's ``tempo_fsed`` helper, warp shape only), with this
   column's species and rules: v3.2.5 itself fixes the rain and graupel fall speeds through the substeps
   (main:672-745), so those match the strict column's order; hail (main:745-776) falls like graupel's mass;
   ice and ice number run every substep (main:797-802; v3.0.3's single steps-divided pass is gone);
2. TEMPO's own binary64 in binary32 (tempo_fastdp_v303.fast_dp_source);
3. no binary64 at all: binary32 blob, constant-bank parameters, CUDA binary32 intrinsics
   (tempo_allfloat_v303.all_float_source);
4. graupel traces below 1e-9 kg/kg absent (tempo_tinymass_v303.tiny_mass_source, WOOF_HEX_TEMPO_TINY_MASS).

The (level, cell) phase kernels (tempo_phase_v325, the v3.0.3 split carried over) run this fast column by
default; WOOF_HEX_TEMPO_PHASE=0 keeps the warp kernel.
"""
from __future__ import annotations

from functools import lru_cache

from woof.hex.mpasphys.rrfs import tempo_fastsed_v303 as _sed

_START = "  // sedimentation: rain (main:644-683; the substeps no longer recompute the fall speeds)\n"
_END = "  if (!C.turn_off_micro_flag) {\n    freeze_cloud_melt_ice("
_TAG = "tempo_fast_v325"

_FAST_BLOCK = r"""#if TEMPO_WARP
  // Fast numerics (tempo_fast_v325): fall speeds once per call, register flux substeps (tempo_fsed).
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
    // hail (mass only)
    if (tempo_any(W.l_qh, nz)) {
      ktop_sedi = 1;
      substeps_sedi = 1;
      hail_fallspeed(P, C, nz, W.rhof, W.l_qh, W.rh, W.ilamh, dz1d, W.vtrh, &substeps_sedi, &ktop_sedi);
      float* const xs[1] = {W.rh};
      float* const ts[1] = {W.qhten};
      const float* const vs[1] = {W.vtrh};
      const float ls[1] = {PS(r1)};
      tempo_fsed<1>(P, C, nz, dz1d, W.rho, substeps_sedi, substeps_sedi, ktop_sedi, xs, ts, vs, ls,
                    &W.hail_liquid_equiv_precip);
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
    // ice: every substep (v3.2.5 main:797-802)
    if (tempo_any(W.l_qi, nz)) {
      ktop_sedi = 1;
      substeps_sedi = 1;
      ice_fallspeed(P, C, nz, W.rhof, W.l_qi, W.ri, W.ilami, dz1d, W.vtri, W.vtni, &substeps_sedi, &ktop_sedi);
      float* const xs[2] = {W.ri, W.ni};
      float* const ts[2] = {W.qiten, W.niten};
      const float* const vs[2] = {W.vtri, W.vtni};
      const float ls[2] = {PS(r1), PS(r2)};
      tempo_fsed<2>(P, C, nz, dz1d, W.rho, substeps_sedi, substeps_sedi, ktop_sedi, xs, ts, vs, ls,
                    &W.ice_liquid_equiv_precip);
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
    """The v3.2.5 column's sedimentation as register flux substeps (warp shape, fast only)."""
    if _TAG in source or _sed._TAG in source:
        raise ValueError("TEMPO v3.2.5 fast sedimentation: already applied")
    for marker in (_START, _END, _sed._HELPER_AT):
        count = source.count(marker)
        if count != 1:
            raise ValueError(f"TEMPO v3.2.5 fast sedimentation: expected one {marker.splitlines()[0]!r}, found {count}")
    a, b = source.index(_START), source.index(_END)
    if not a < b:
        raise ValueError("TEMPO v3.2.5 fast sedimentation: block markers out of order")
    source = source[:a] + _FAST_BLOCK + source[a:b] + "#endif\n\n" + source[b:]
    return source.replace(_sed._HELPER_AT, _sed._HELPER + _sed._HELPER_AT)


@lru_cache(maxsize=8)
def _fast_column(column: str, tiny: str) -> str:
    from woof.hex.mpasphys.rrfs.tempo_allfloat_v303 import all_float_source
    from woof.hex.mpasphys.rrfs.tempo_fastdp_v303 import fast_dp_source
    from woof.hex.mpasphys.rrfs.tempo_tinymass_v303 import tiny_mass_source
    return tiny_mass_source(all_float_source(fast_dp_source(fast_sed_source(column))), tiny)


def fast_column(column: str) -> str:
    """The fast v3.2.5 column: register sedimentation, binary32 arithmetic, all-float layer, tiny graupel."""
    import os
    return _fast_column(column, os.environ.get("WOOF_HEX_TEMPO_TINY_MASS", "graupel").strip().lower())
