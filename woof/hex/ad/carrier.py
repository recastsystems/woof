# SPDX-License-Identifier: Apache-2.0
"""The parameter carrier: makes the constants of :mod:`woof.hex.ad.params` differentiable inputs of the derivative modules.

Two source generators, both text rewrites that refuse unless every pattern matches its expected count, so a scheme
source that changed upstream can never be carried silently at the shipped constant:

* :func:`carry_p4` post-processes a generated P4 derivative source (MYNN surface layer, GSL drag; the output of the P4
  generator, one ``.p4.cu`` per family). The verbatim forward reads ``p4_parf(k, literal)``
  (``p4_par_on[k] ? p4_par_val[k] : literal``; module globals, zero-initialised, so by default it computes exactly
  the literal: the same float operation on the same value). The tangent copy reads ``ADF(value, p4_par_t[k])``; the
  reverse copy pushes a tape leaf with array id ``8 + k`` whose adjoint the sweep adds to
  ``p4_par_bar[k * 4096 + column]`` (one thread per column, fixed sweep order).
* :func:`carry_p3` rewrites the MYNN-EDMF whole-launch derivative source and its captured forward for the B1 slot: a
  ``MynnColumn`` member that is the tangent (mode 0) or a tape leaf (modes 1 and 2), set per column before the driver
  call; its bar is added to ``mw_par_bar[column]`` after the reverse. The forward reads ``mw_parf(0, b1)`` at the same
  occurrence, so finite differences and the derivative see the same parameter.

Command line::

    python -m woof.hex.ad.carrier p4 FAMILY IN.p4.cu OUT.cu [--constants a,b]
    python -m woof.hex.ad.carrier p3 DERIV_IN.cu FORWARD_IN.cu DERIV_OUT.cu FORWARD_OUT.cu

Each prints a JSON receipt (input and output SHA-256, the carried constants).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Sequence

from .params import CONSTANTS, Constant, select


class CarrierError(RuntimeError):
    """A pattern did not match its expected count: the source is not the one the carrier was written for."""


def _sub1(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        raise CarrierError(f"{what}: matched {n}, expected 1")
    return text.replace(old, new, 1)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


P4_MARK = "// ===== p4: derivative types ====="
P4_ENTRIES = "// ===== p4: entries part 1"
P4_HEAD = """// ===== rung3: parameter carrier (ad/rung3-params) =====
#define P4_NPAR 8
#define P4_PAR_COLS 4096
__device__ int p4_par_on[P4_NPAR];
__device__ float p4_par_val[P4_NPAR];
__device__ __forceinline__ float p4_parf(int k, float lit) { return p4_par_on[k] ? p4_par_val[k] : lit; }
"""
P4_ADHEAD = """
// ===== rung3: parameter carrier, derivative side =====
#if defined(P4_MODE_TL)
__device__ double p4_par_t[P4_NPAR];
__device__ __forceinline__ ADF p4_parA(int k, float lit) { return ADF(p4_parf(k, lit), p4_par_t[k]); }
#else
__device__ double p4_par_bar[P4_NPAR * P4_PAR_COLS];
__device__ int p4_par_cols_overflow;
__device__ __forceinline__ ADF p4_parA(int k, float lit) { return p4_leaf(p4_parf(k, lit), 8 + k, 0); }
#endif
"""


def p4_constants(family: str, names: Sequence[str] | None = None) -> list[Constant]:
    chosen = [c for c in (select(names) if names else CONSTANTS) if c.family == family]
    if not chosen:
        raise CarrierError(f"no carried constants in family {family}")
    return chosen


def carry_p4(source: str, family: str, names: Sequence[str] | None = None) -> str:
    """The carrier P4 source for ``family`` ("mynnsfc" or "ugwp") from the generated P4 source."""
    if family not in ("mynnsfc", "ugwp"):
        raise CarrierError(f"P4 family {family!r}: mynnsfc or ugwp")
    if source.count(P4_MARK) != 1:
        raise CarrierError("derivative-types marker")
    verb, deriv = source.split(P4_MARK)
    for c in p4_constants(family, names):
        for i, site in enumerate(c.sites):
            verb = _sub1(verb, site, site.replace(c.literal, f"p4_parf({c.slot}, {c.literal})"), f"{c.name} site {i} forward")
            dsite = c.derivative_prefix + site
            deriv = _sub1(deriv, dsite, dsite.replace(c.literal, f"p4_parA({c.slot}, {c.literal})"),
                          f"{c.name} site {i} derivative")
    deriv = _sub1(deriv, P4_ENTRIES, P4_ADHEAD + P4_ENTRIES, "entries part 1 marker")
    leaf = ("if (p.x <= -10) { if ({col} < P4_PAR_COLS) p4_par_bar[(long long)(-10 - p.x) * P4_PAR_COLS + {col}] += a;"
            " else p4_par_cols_overflow = 1;")
    if family == "ugwp":
        deriv = _sub1(deriv, "if (p.x <= -2) { TT *b = bin[-2 - p.x];",
                      "if (p.x <= -2) { " + leaf.replace("{col}", "i") + " continue; } TT *b = bin[-2 - p.x];", "ugwp sweep")
    else:
        deriv = _sub1(deriv, "__device__ __forceinline__ void p4_sweep(const int2 *ab, const double2 *w, double *adj, int cnt, long long stride, int t) {",
                      "__device__ __forceinline__ void p4_sweep(const int2 *ab, const double2 *w, double *adj, int cnt, long long stride, int t, int c) {",
                      "sweep signature")
        deriv = _sub1(deriv, "        if (p.x <= -2) continue;\n",
                      "        if (p.x <= -2) { " + leaf.replace("{col}", "c") + " } continue; }\n", "sweep leaf")
        deriv = _sub1(deriv, "p4_sweep(ab, w, adj, cnt, stride, t);", "p4_sweep(ab, w, adj, cnt, stride, t, c);", "sweep call")
    return P4_HEAD + verb + P4_MARK + deriv


P3_GLOBALS = """
// ===== rung3: parameter carrier (ad/rung3-params) =====
#define MW_NPAR 8
#define MW_PAR_COLS 4096
__device__ int mw_par_on[MW_NPAR];
__device__ float mw_par_val[MW_NPAR];
__device__ double mw_par_t[MW_NPAR];
__device__ double mw_par_bar[MW_NPAR * MW_PAR_COLS];
__device__ int mw_par_cols_overflow;
__device__ __forceinline__ float mw_parf(int k, float lit) { return mw_par_on[k] ? mw_par_val[k] : lit; }
"""
P3_B1_DECL = "static constexpr float b1 = 24.0f;"


def carry_p3(deriv_source: str, forward_source: str) -> tuple[str, str]:
    """(derivative, forward) carrier sources for the MYNN-EDMF B1 slot."""
    b1 = next(c for c in CONSTANTS if c.family == "mynnedmf" and c.slot == 0)
    occ = b1.sites[0]
    if deriv_source.count(P3_B1_DECL) != 1 or forward_source.count(P3_B1_DECL) != 1:
        raise CarrierError("b1 = 24 declaration")
    g = _sub1(deriv_source, '\n#include "mw_types.cuh"\n', '\n#include "mw_types.cuh"\n' + P3_GLOBALS, "include")
    g = _sub1(g, "int lane_sub = 0; unsigned lane_mask = 0u;", "int lane_sub = 0; unsigned lane_mask = 0u;\nMR mw_p_b1;", "member")
    g = _sub1(g, occ, occ.replace("r6_mul(b1, 0.25f)", "r6_mul(mw_p_b1, 0.25f)"), "derivative occurrence")
    g = _sub1(g, "  column.active_i = active_i; column.active_j = active_j;\n",
              "  column.active_i = active_i; column.active_j = active_j;\n"
              "  int mw_pid = -1;\n"
              f"  {{ const float pv = mw_parf(0, {b1.literal});\n"
              "    if constexpr(MODE==0) { column.mw_p_b1 = mw_par_t[0] != 0.0 ? F(pv, D(mw_par_t[0], &ctx)) : F(pv); }\n"
              "    else { mw_pid = ctx.leaf(); column.mw_p_b1 = F(pv, D(mw_pid, &ctx)); } }\n", "column init")
    g = _sub1(g, "    ctx.reverse();\n",
              "    ctx.reverse();\n"
              "    if (mw_pid >= 0) { if (index < MW_PAR_COLS) mw_par_bar[index] += ctx.bars[mw_pid]; else mw_par_cols_overflow = 1; }\n",
              "reverse collect")
    fwd_globals = (P3_GLOBALS.replace("__device__ double mw_par_t[MW_NPAR];\n", "")
                   .replace("__device__ double mw_par_bar[MW_NPAR * MW_PAR_COLS];\n", "")
                   .replace("__device__ int mw_par_cols_overflow;\n", ""))
    f = _sub1(fwd_globals + forward_source, occ, occ.replace("r6_mul(b1, 0.25f)", "r6_mul(mw_parf(0, b1), 0.25f)"),
              "forward occurrence")
    return g, f


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m woof.hex.ad.carrier", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="unit", required=True)
    p4 = sub.add_parser("p4", help="carry constants in a generated P4 source (mynnsfc, ugwp)")
    p4.add_argument("family", choices=("mynnsfc", "ugwp"))
    p4.add_argument("source")
    p4.add_argument("out")
    p4.add_argument("--constants", help="comma-separated constant names (default: every constant of the family)")
    p3 = sub.add_parser("p3", help="carry MYNN-EDMF B1 in the whole-launch derivative source and its captured forward")
    p3.add_argument("deriv_source")
    p3.add_argument("forward_source")
    p3.add_argument("deriv_out")
    p3.add_argument("forward_out")
    a = ap.parse_args(argv)
    try:
        if a.unit == "p4":
            src = open(a.source, encoding="utf-8").read()
            names = [n for n in a.constants.split(",")] if a.constants else None
            out = carry_p4(src, a.family, names)
            open(a.out, "w", encoding="utf-8", newline="").write(out)
            print(json.dumps({"unit": "p4", "family": a.family, "in_sha256": _sha(src), "out_sha256": _sha(out),
                              "constants": [{"name": c.name, "slot": c.slot, "literal": c.literal, "sites": len(c.sites)}
                                            for c in p4_constants(a.family, names)]}))
        else:
            g0 = open(a.deriv_source, encoding="utf-8").read()
            f0 = open(a.forward_source, encoding="utf-8").read()
            g, f = carry_p3(g0, f0)
            open(a.deriv_out, "w", encoding="utf-8", newline="").write(g)
            open(a.forward_out, "w", encoding="utf-8", newline="").write(f)
            print(json.dumps({"unit": "p3", "deriv_in": _sha(g0), "deriv_out": _sha(g), "forward_in": _sha(f0),
                              "forward_out": _sha(f), "constants": [{"name": "mynn_qke_dissipation_b1", "slot": 0}]}))
    except CarrierError as e:
        raise SystemExit(f"carrier: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
