"""Which C library produced a float32 artifact, measured rather than declared.

THE BREAKAGE THIS PREVENTS, measured 2026-08-27 (the libm-identity finding).  The registered
vertical artifact carries ``dcEdge**2/12`` as numpy's SCALAR power, which calls
the platform ``powf``.  ``powf`` is not required to be correctly rounded, and
the two C libraries this program runs on do not round it the same way.  On the
real 112,676-cell parent's own float32 ``dcEdge`` -- 338,022 values, input
SHA-256/16 ``689494c58c8c0c47`` on both boxes -- the two answers differ:

    box                                     dc2/12 vs correctly-rounded
    an RTX 5090, glibc 2.43              207 of 338,022
    this Windows desktop, MSVC CRT           14 of 338,022
    the two boxes against EACH OTHER        215 of 338,022, all 1 ULP

``np.square`` -- one correctly-rounded multiply -- gives the SAME digest on
both boxes.  So the artifact of record is byte-reproducible on one libm, and a
lane that rebuilds it elsewhere gets different bytes from identical source.

That has already nearly cost this program a day: a verification lane rebuilt
from pristine source on a second box, got a digest that did not match a frozen
one, and had to run a bisect to find out that nothing was broken.  A digest
that does not say which libm it belongs to invites exactly that hunt.

WHY A MEASUREMENT AND NOT A VERSION STRING.  ``platform.libc_ver()`` reports
nothing at all on Windows, reports the build-time glibc rather than the runtime
one in some builds, and says nothing about whether numpy dispatched ``powf``
through the C library or through its own SIMD loop.  A version string is a
claim about the environment; the fingerprint below is the environment's own
answer to the question that matters.  Two boxes that agree here produce the
same artifact bytes whatever their version strings say, and two that disagree
produce different ones however identical those strings look.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from typing import Any

import numpy as np

__all__ = [
    "SCALAR_POW_PROBE_SIZE",
    "KNOWN_SCALAR_POW_LIBMS",
    "ARRAY_POW_LEAVES_SCALAR_PATH",
    "scalar_pow_probe_inputs",
    "scalar_pow_fingerprint",
    "correctly_rounded_square_fingerprint",
    "describe_float_environment",
    "refuse_unless_recorded_libm_matches",
]

#: Probe width.  Large enough that two libms cannot agree by luck: the two
#: measured here disagree on about 0.065 % of float32 arguments, so this many
#: samples separates them by roughly 680 differing values.
SCALAR_POW_PROBE_SIZE = 1 << 20

_GOLDEN = np.uint32(2654435761)


def scalar_pow_probe_inputs(n: int = SCALAR_POW_PROBE_SIZE) -> np.ndarray:
    """Deterministic float32 probe values in a real mesh's ``dcEdge`` range.

    Built arithmetically rather than from ``numpy.random`` on purpose: a
    Generator's bit stream is not guaranteed identical across numpy versions,
    and a fingerprint whose INPUT depends on the numpy version cannot separate
    a libm difference from a numpy difference.  These bytes are the same on
    every numpy that exists.
    """

    i = np.arange(n, dtype=np.uint32)
    bits = (i * _GOLDEN) ^ ((i * _GOLDEN) >> np.uint32(13))
    frac = (bits & np.uint32(0x00FFFFFF)).astype(np.float64) / float(1 << 24)
    # 1 km to 1,000 km: the magnitudes a real mesh's edge lengths carry.
    return (1.0e3 + frac * 9.99e5).astype(np.float32)


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def scalar_pow_fingerprint(n: int = SCALAR_POW_PROBE_SIZE) -> str:
    """This box's ``powf`` answer, as a 16-hex fingerprint.

    Uses the two-array ``np.power`` form, which is the ``powf`` path and is
    what :func:`woof.hex.vertical.edge_dc_squared_over_twelve` uses to
    reproduce the artifact of record.  Measured equal to the element-wise
    scalar expression on both boxes, 0 of 338,022 and 0 of 200,000 differing.
    """

    x = scalar_pow_probe_inputs(n)
    twos = np.full(x.shape, 2.0, dtype=x.dtype)
    return _digest(np.power(x, twos))


def correctly_rounded_square_fingerprint(n: int = SCALAR_POW_PROBE_SIZE) -> str:
    """The same values through ``np.square``, one correctly-rounded multiply.

    This is the control.  It was measured IDENTICAL on both boxes, which is
    what proves the scalar-power split is the libm and not the inputs, not
    numpy's version, and not the machine's floating-point mode.  If this ever
    differs between two boxes, the diagnosis in this module is wrong and the
    cause is upstream of ``powf``.
    """

    return _digest(np.square(scalar_pow_probe_inputs(n)))


#: The libms measured so far, keyed by their own answer.
#:
#: ADDING A ROW IS ADDITIVE.  A third C library gets its own row, measured on
#: that box; no existing row is ever edited to make a machine agree, because an
#: edited row silently retires the artifacts minted against it.
KNOWN_SCALAR_POW_LIBMS: dict[str, dict[str, str]] = {
    "e8f06d91c151bfd8": {
        "libm": "GNU C Library (glibc) 2.43",
        "measured_on": "an RTX 5090, Ubuntu, Linux 7.0.0-29-generic",
        "numpy": "2.3.5",
        "measured_date": "2026-08-27",
        "note": (
            "the library the registered vertical artifact of record was minted "
            "on: evidence/parent-regen-20260827/, run s01, "
            "s01.vertical.nc.receipt.json"
        ),
    },
    "b713699c69b141dc": {
        "libm": "Microsoft Visual C++ runtime (MSVC CRT)",
        "measured_on": "the reference Windows 11 desktop (RTX 3080, sm_86)",
        "numpy": "2.2.6",
        "measured_date": "2026-08-27",
        "note": (
            "closer to correctly rounded than glibc on this probe, but NOT "
            "correctly rounded: 14 of the real parent's 338,022 dc2/12 values "
            "still differ from np.square here"
        ),
    },
    "ded6cf5be374b61d": {
        "libm": (
            "numpy's own CPU-dispatched SIMD pow kernel (engaged on AVX-512 "
            "silicon, over an unchanged glibc scalar powf)"
        ),
        "measured_on": (
            "the GitHub ubuntu-24.04 runner (first CI run to execute this "
            "gate, 33405889844) and WSL Ubuntu 24.04 glibc 2.39-0ubuntu8.7 "
            "on an AMD Ryzen 9 9950X3D"
        ),
        "numpy": "2.3.5 / 2.4.3 / 2.4.6 / 2.5.2 -- digest identical on all four",
        "measured_date": "2026-08-31",
        "note": (
            "NOT a third C library.  On the same box, the same probe through "
            "the element-wise SCALAR expression still digests e8f06d91c151bfd8 "
            "-- the registered glibc row -- so the platform's scalar powf is "
            "byte-identical from glibc 2.39 through 2.43 and what moved is "
            "numpy's ARRAY dispatch: np.power(array, array) takes numpy's "
            "vendored SIMD kernel on CPUs with the AVX-512 feature set and "
            "answers differently on 220,001 of the 1,048,576 probe values "
            "(float32; the float64 kernel diverges too, 8,926 of 200,000 on "
            "the equivalence test's own values).  Cross-checked on a "
            "no-AVX-512 Linux box the same day (an RTX 5090, Intel Core "
            "Ultra 7 270K, glibc 2.43, numpy 2.5.2): array == scalar there, "
            "both e8f06d91.  The correctly-rounded square control is "
            "d026ac1b6fd1ea85 on every box named here, which is what pins the "
            "split to the pow path.  A receipt carrying this fingerprint "
            "names bytes minted by numpy's kernel, and "
            "refuse_unless_recorded_libm_matches keeps them apart from glibc "
            "and MSVC artifacts exactly as it keeps those apart from each "
            "other.  See ARRAY_POW_LEAVES_SCALAR_PATH below for the one "
            "consequence the equivalence tests must honour."
        ),
    },
}


#: Fingerprints whose ARRAY ``np.power`` is MEASURED to leave the scalar
#: libm path on the box that produces them.
#:
#: On every platform measured before 2026-08-31, ``np.power(a, b)`` with two
#: arrays and the element-wise scalar expression answered identical bytes,
#: and the vectorized-equivalence tests assert exactly that.  On AVX-512
#: silicon numpy dispatches the array form to its own SIMD pow kernel, so on
#: those boxes the assertion is a statement about numpy's dispatch table,
#: not about this repository's loops -- the vectorized form CANNOT reproduce
#: the scalar transcription there, in either precision (measured: 36,855 of
#: 200,000 float32 and 8,926 of 200,000 float64 on the equivalence test's
#: own values).  ``tests/test_vertical_vectorized_equivalence.py`` skips its
#: pow-path bitwise legs BY NAME on these fingerprints -- a skip that names
#: the measured figures, never a widened tolerance -- and every other leg
#: still runs.  What governs ARTIFACTS on such a box is unchanged: the
#: receipt names this fingerprint and
#: :func:`refuse_unless_recorded_libm_matches` refuses cross-libm reads by
#: name.  Membership is additive and measured, like the table above; a row
#: enters here only with the same-box scalar digest recorded in its table
#: note, because that is what separates "numpy dispatched away from the
#: libm" from "the libm itself moved".
ARRAY_POW_LEAVES_SCALAR_PATH: frozenset[str] = frozenset({"ded6cf5be374b61d"})


def describe_float_environment() -> dict[str, Any]:
    """The block a receipt should carry so its digests name their libm."""

    fingerprint = scalar_pow_fingerprint()
    known = KNOWN_SCALAR_POW_LIBMS.get(fingerprint)
    return {
        "scalar_pow_fingerprint": fingerprint,
        "correctly_rounded_square_fingerprint": correctly_rounded_square_fingerprint(),
        "identified_libm": known["libm"] if known else None,
        "identified_from": known["measured_on"] if known else None,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "why": (
            "float32 arrays in this artifact are produced through numpy's "
            "scalar power, which calls the platform powf; powf is not required "
            "to be correctly rounded and the libraries measured so far disagree "
            "for about 0.065 % of float32 arguments, so a digest over these "
            "bytes belongs to this fingerprint and not to the source alone"
        ),
    }


def refuse_unless_recorded_libm_matches(
    recorded_fingerprint: str | None,
    *,
    what: str,
) -> None:
    """Refuse to read a byte-identity failure as a defect on a different libm.

    THE BREAKAGE THIS PREVENTS: a lane rebuilds a registered float32 artifact
    on a box whose ``powf`` is not the one the artifact was minted on, gets a
    different SHA-256 from identical source, and spends a day bisecting a
    defect that does not exist.  That is the shape of the four reds the
    parent-regeneration verification lane hit on 2026-08-27 and had to bisect
    by hand.

    ``recorded_fingerprint`` of ``None`` means the artifact predates this block
    and cannot say which libm minted it.  That is refused too, and named as
    what it is -- an artifact that cannot be compared byte-for-byte anywhere,
    including on the box that made it -- rather than passed as if the question
    had been asked and answered.
    """

    local = scalar_pow_fingerprint()
    if recorded_fingerprint == local:
        return

    known_local = KNOWN_SCALAR_POW_LIBMS.get(local)
    local_name = known_local["libm"] if known_local else f"an unmeasured libm ({local})"
    if recorded_fingerprint is None:
        raise ValueError(
            f"{what} carries no libm fingerprint, so a byte comparison against it "
            f"cannot be told apart from a real defect. This box is {local_name}. "
            f"float32 `dcEdge**2` goes through the platform `powf`, which the "
            f"libraries measured so far round differently for about 0.065 % of "
            f"arguments -- 215 of one real mesh's 338,022 edges -- so the same "
            f"source produces different artifact bytes on different libms. Re-mint "
            f"the artifact with `woof.hex.libm_identity.describe_float_environment()` "
            f"in its receipt, or compare against a copy minted on this box."
        )

    known_recorded = KNOWN_SCALAR_POW_LIBMS.get(recorded_fingerprint)
    recorded_name = (
        known_recorded["libm"]
        if known_recorded
        else f"an unmeasured libm ({recorded_fingerprint})"
    )
    raise ValueError(
        f"{what} was minted on {recorded_name} and this box is {local_name}, so its "
        f"float32 bytes CANNOT match here and a mismatch is not evidence of a defect. "
        f"Measured 2026-08-27: on the real 112,676-cell parent the two libms' "
        f"`dcEdge**2/12` disagree for 215 of 338,022 edges, every one of them by a "
        f"single ULP, and that moved `zb` and `zb3` in the registered vertical "
        f"artifact while 137 of 139 variables stayed bitwise equal. Compare against a "
        f"copy minted on this box, or run the comparison on the box named in the "
        f"artifact's receipt. Do NOT widen a tolerance to make this pass: the "
        f"difference is real, it is one ULP, and a tolerance that hides it also hides "
        f"a genuine one-ULP defect."
    )
