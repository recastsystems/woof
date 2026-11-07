"""Mechanical grading of every returned float32 word.

Strict requires identical words, including signed zero. Default allows a
relative residual of 1e-2 plus the stated absolute floor for each output.
These are port residual limits on the column fixtures, not forecast skill.
"""
from __future__ import annotations

import numpy as np

from woof.core.fp32_ulp import fp32_ulp_distance

DEFAULT_RTOL = 1e-2


def absolute_tolerance(name):
    if name == 'auto_rate':
        return 1e-18
    if name == 'qv':
        return 2e-7
    if name.startswith('q'):
        return 1e-7
    if name.startswith('prec'):
        return 1e-6
    if name.startswith('re_'):
        return 0.2
    if name == 'refl_10cm':
        return 0.2
    if name == 'th':
        return 1e-3
    if name.startswith('acphys'):
        return 5e-4
    if name.startswith('phys') or name == 'sr':
        return 1e-5
    return 1e-4


def relative_tolerance(name):
    if name in ('th', 'qv', 'refl_10cm', 'sr') or name.startswith(('phys', 'acphys')):
        return 0.0
    return DEFAULT_RTOL


def compare_words(name, actual, expected, *, strict):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.dtype != np.float32 or b.dtype != np.float32 or a.shape != b.shape:
        raise ValueError(f'{name}: both outputs must have the same float32 shape')
    x, y = a.view(np.uint32), b.view(np.uint32)
    bad = x != y
    finite = np.isfinite(a) & np.isfinite(b)
    signed_zero = bad & (((x | y) & 0x7fffffff) == 0)
    with np.errstate(invalid="ignore"):
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    tolerance = absolute_tolerance(name) + relative_tolerance(name) * np.abs(b.astype(np.float64))
    rejected = ~finite | (bad if strict else delta > tolerance)
    distance = fp32_ulp_distance(a, b)
    first = None
    if rejected.any():
        idx = tuple(np.argwhere(rejected)[0])
        first = {'index': [int(i) for i in idx],
                 'got_bits': f'{int(x[idx]):08x}', 'want_bits': f'{int(y[idx]):08x}'}
    return {'words': a.size, 'differ': int(bad.sum()),
            'signed_zero': int(signed_zero.sum()),
            'nonfinite': int((~finite).sum()),
            'rejected': int(rejected.sum()),
            'max_ulp': int(distance[finite].max(initial=0)),
            'max_abs': float(delta[finite].max(initial=0.0)),
            'first': first}
