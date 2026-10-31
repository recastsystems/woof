"""TEMPO elementary functions for fast numerics only.

The source owner applies :func:`fast_source` to the column support in both
serial and warp builders, never in strict mode. Like Thompson's fast support,
CUDA binary32 functions replace the binary64 glibc cores. Wide normal doubles
use a binary64 exponent and a binary32 mantissa. Exceptional operands and
results outside the normal range retain the supplied exact replica.

This deliberately changes fast words. It does not change the shared maths
header, gamma helpers, integer powers, reductions, or strict source.
"""

# The moderate power exponent bound avoids magnifying binary32 log error for
# arbitrarily large exponents (notably bases near one). Larger powers use libm.
_CORE = r"""
TEMPO_HD bool tempo_fast_normal(double x) {
  return x >= 0x1p-1022 && x <= 0x1.fffffffffffffp1023;
}
TEMPO_HD double tempo_fast_log2(double x) {
  int e;
  const double m = frexp(x, &e);
  return (double)e + (double)log2f((float)m);
}
TEMPO_HD bool tempo_fast_exponent(double y) {
  return y >= -1021.0 && y <= 1023.0;
}
TEMPO_HD double tempo_fast_exp2(double y) {
  const double e = floor(y);
  return ldexp((double)exp2f((float)(y - e)), (int)e);
}
TEMPO_HD bool tempo_fast_narrow(float r) {
  return fabsf(r) > 1.2e-38f && fabsf(r) < 3.0e38f;
}
"""

_SHARED_OLD = r"""struct TempoPowBase { double x, hi, lo; bool fast; };
#if defined(TEMPO_SHARED_POW)
TEMPO_HD TempoPowBase tempo_pow_base(double x) {
  TempoPowBase b; b.x = x; b.hi = 0.0; b.lo = 0.0;
  const uint32_t topx = g64_pow::top12(x);
  b.fast = !(topx - 0x001 >= 0x7ff - 0x001);
  if (b.fast) b.hi = g64_pow::log_inline(g64_pow::asuint64(x), &b.lo);
  return b;
}
TEMPO_HD double tempo_pow_with(const TempoPowBase& b, double y) {
  const uint32_t topy = g64_pow::top12(y);
  if (!b.fast || (topy & 0x7ff) - 0x3be >= 0x43e - 0x3be) return mpas_pow(b.x, y);
  const double ehi = __dmul_rn(y, b.hi);
  const double elo = __fma_rn(y, b.lo, __fma_rn(y, b.hi, -ehi));
  return g64_pow::exp_inline(ehi, elo, 0);
}
#else
TEMPO_HD TempoPowBase tempo_pow_base(double x) {TempoPowBase b; b.x = x; b.hi = 0.0; b.lo = 0.0; b.fast = false; return b;}
TEMPO_HD double tempo_pow_with(const TempoPowBase& b, double y) {return mpas_pow(b.x, y);}
#endif"""

_SHARED_NEW = r"""// Fast numerics: share binary32 log2, preserving the wide binary64 exponent.
struct TempoPowBase { double x, hi, lo; bool fast; };
TEMPO_HD TempoPowBase tempo_pow_base(double x) {
  TempoPowBase b; b.x = x; b.hi = 0.0; b.lo = 0.0;
  b.fast = tempo_fast_normal(x);
  if (b.fast) b.hi = tempo_fast_log2(x);
  return b;
}
TEMPO_HD double tempo_pow_with(const TempoPowBase& b, double y) {
  if (b.fast && isfinite(y) && fabs(y) <= 128.0) {
    const double z = y * b.hi;
    if (tempo_fast_exponent(z)) return tempo_fast_exp2(z);
  }
  return mpas_pow(b.x, y);
}"""


def _replacements():
    for scalar, suffix in (("float", "f"), ("double", "")):
        for name in ("exp", "log", "log10", "pow"):
            args = f"{scalar} x,{scalar} y" if name == "pow" else f"{scalar} x"
            call = "x,y" if name == "pow" else "x"
            old = f"TEMPO_HD {scalar} tempo_{name}({args}) {{return mpas_{name}{suffix}({call});}}"
            fallback = f"return mpas_{name}{suffix}({call});"
            if name == "exp":
                body = ("if (isfinite(x)) { const float r = expf((float)x); "
                        "if (tempo_fast_narrow(r)) return (" + scalar + ")r; ")
                if scalar == "double":
                    body += ("const double z = x * 1.4426950408889634; "
                             "if (tempo_fast_exponent(z)) return tempo_fast_exp2(z); ")
                body += "} "
            elif name in ("log", "log10"):
                body = (f"if (x > 1.2e-38 && x < 3.0e38) return ({scalar}){name}f((float)x); ")
                if scalar == "double":
                    factor = "0.6931471805599453" if name == "log" else "0.3010299956639812"
                    body += f"if (tempo_fast_normal(x)) return tempo_fast_log2(x) * {factor}; "
            else:
                body = ("if (x > 1.2e-38 && x < 3.0e38 && isfinite(y) && fabs((double)y) <= 128.0) { "
                        "const float r = powf((float)x,(float)y); "
                        f"if (tempo_fast_narrow(r)) return ({scalar})r; }} ")
                if scalar == "double":
                    body += ("if (tempo_fast_normal(x) && isfinite(y) && fabs(y) <= 128.0) { "
                             "const double z = y * tempo_fast_log2(x); "
                             "if (tempo_fast_exponent(z)) return tempo_fast_exp2(z); } ")
            new = f"TEMPO_HD {scalar} tempo_{name}({args}) {{" + body + fallback + "}"
            if scalar == "float" and name == "exp":
                new = _CORE + new
            yield old, new
    yield _SHARED_OLD, _SHARED_NEW


def fast_source(source: str) -> str:
    """Replace exactly one copy of each TEMPO helper, or fail on source drift.

    Accepts either ``tempo_v303._COLUMN`` or a complete CUDA module. The caller
    selects fast mode; this function has no environment or numerics side effects.
    Applying it twice is an error, as is a missing or duplicated helper.
    """
    for old, new in _replacements():
        count = source.count(old)
        if count != 1:
            raise ValueError(f"TEMPO fast maths: expected one {old.splitlines()[0]}, found {count}")
        source = source.replace(old, new)
    return source
