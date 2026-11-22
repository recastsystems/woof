"""Carry the tridiagonal sweeps' previous element in a register.

``tridiag2_`` (the solver every 8f8e8b59 MYNN tendency and the TKE/variance
prediction call) reads, in each iteration, the element its previous
iteration just stored to the column arena (``cp1(k-1)``, ``dp1(k-1)``
forward; ``x(k+1)`` backward). In native code that is a register; through
the escaping arena it is a store-to-load round trip per level. The stores
stay (the backward sweep reads ``cp1``/``dp1``); only the reload of the
value just computed becomes the register that holds it. The arrays are
distinct at every call site (asserted), so every word is unchanged.
"""
import re

FORWARD = """    cp1[0] = c__[1] / b[1];
    dp1[0] = d__[1] / b[1];
    i__1 = *kte;

#pragma unroll 1
for (k = 2; k <= i__1; ++k) {
	m = b[k] - cp1[k - 2] * a[k];
	cp1[k - 1] = c__[k] / m;
	dp1[k - 1] = (d__[k] - dp1[k - 2] * a[k]) / m;
/* L9004: */
    }
/* L9005: */
    x[*kte] = dp1[*kte - 1];

#pragma unroll 1
for (k = *kte - 1; k >= 1; --k) {
	x[k] = dp1[k - 1] - cp1[k - 1] * x[k + 1];
/* L9007: */
    }"""

CARRIED = """    real m480_cp = c__[1] / b[1];
    real m480_dp = d__[1] / b[1];
    cp1[0] = m480_cp;
    dp1[0] = m480_dp;
    i__1 = *kte;

#pragma unroll 1
for (k = 2; k <= i__1; ++k) {
	m = b[k] - m480_cp * a[k];
	m480_cp = c__[k] / m;
	cp1[k - 1] = m480_cp;
	m480_dp = (d__[k] - m480_dp * a[k]) / m;
	dp1[k - 1] = m480_dp;
/* L9004: */
    }
/* L9005: */
    real m480_x = dp1[*kte - 1];
    x[*kte] = m480_x;

#pragma unroll 1
for (k = *kte - 1; k >= 1; --k) {
	m480_x = dp1[k - 1] - cp1[k - 1] * m480_x;
	x[k] = m480_x;
/* L9007: */
    }"""


def recurrence(text):
    start = text.index('/* Subroutine */ int tridiag2_(', text.index('/* Subroutine */ int tridiag2_(') + 10)
    end = text.index('} /* tridiag2_ */', start)
    body = text[start:end]
    pattern = re.compile(r'\s*'.join(map(re.escape, FORWARD.split())))
    found = list(pattern.finditer(body))
    assert len(found) == 1
    for call in re.findall(r'tridiag2_\(m471_work,m471_n,([^;]*)\);', text):
        args = [a.strip() for a in re.sub(r'\s+', ' ', call).split(',')]
        assert len(set(args[1:])) == len(args) - 1, args       # no aliased arrays
    body = body[:found[0].start()] + CARRIED + body[found[0].end():]
    return text[:start] + body + text[end:]
