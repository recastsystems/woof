"""cumastrn:996-1016 graded at ``max_ulp == 0`` -- the momentum rescale.

THE ONLY CONSUMER OF ``zcons`` IN THE ENTIRE SCHEME. Every other mass-flux
cap uses ``zcons2`` = 3/(g*dt); this one uses ``zcons`` = 1/(g*dt), one
character away, three times tighter. Getting it wrong makes the momentum
rescale three times too permissive, and the result is finite, plausible and
off by a fixed ratio.

It produces ``zmfuus``/``zmfdus``, and it is THOSE, not ``pmfu``/``pmfd``,
that cududvn consumes -- so cududvn's captured entry state is this block's
exit, and no new output capture was needed.

BOTH OF THIS PHASE'S ERRORS LANDED ON THE MOMENTUM PATH (observed by
review, review): :833-919 misattributed as producing zmfuus/zmfdus, and
the zcons comment wrong about which constants differ -- whose sole consumer
is this block. That path is where the port has no precedent to lean on: GF
and KF both omit momentum coupling and there is no WOOF-ArWen analogue to
pattern-match against. Hence the extra checks below.
"""
from __future__ import annotations

import numpy as np
import pytest

from woof.verify.ntiedtke_oracle import NT_NZ, load_csv, word
from woof.verify.ntiedtke_ref import np_ntiedtke_momentum_rescale

_ZTMST = np.float32(60.0)


def _load():
    cols, exp = {}, {}
    for r in load_csv("nt-cududvn-in-surface.csv"):
        cols[(int(r["case"]), float(r["dx"]))] = {
            "ldcum": int(r["ldcum"]), "paph_sfc": word(r["paph_sfc"])}
    for r in load_csv("nt-cuflxn-in-surface.csv"):
        cols[(int(r["case"]), float(r["dx"]))]["kctop"] = int(r["kctop"])
    for r in load_csv("nt-mrescale-in-levels.csv"):
        s = cols[(int(r["case"]), float(r["dx"]))]
        k = int(r["k"]) - 1
        for f in ("pmfu", "pmfd", "paph"):
            s.setdefault(f, np.zeros(NT_NZ, dtype=np.float32))[k] = word(r[f])
    # cududvn's captured entry IS this block's exit: it takes the SCALED
    # pair under the names pmfu/pmfd.
    for r in load_csv("nt-cududvn-in-levels.csv"):
        s = exp.setdefault((int(r["case"]), float(r["dx"])), {})
        k = int(r["k"]) - 1
        s.setdefault("zmfuus", np.zeros(NT_NZ, dtype=np.float32))[k] = \
            word(r["pmfu"])
        s.setdefault("zmfdus", np.zeros(NT_NZ, dtype=np.float32))[k] = \
            word(r["pmfd"])
    return cols, exp


_COLS, _EXP = _load()
_KEYS = sorted(_COLS)


def _paph(c):
    out = np.zeros(NT_NZ + 1, dtype=np.float32)
    out[:NT_NZ] = c["paph"]
    out[NT_NZ] = c["paph_sfc"]
    return out


_GOT = {k: np_ntiedtke_momentum_rescale(
    ldcum=bool(_COLS[k]["ldcum"]), kctop=_COLS[k]["kctop"],
    paph=_paph(_COLS[k]), pmfu=_COLS[k]["pmfu"], pmfd=_COLS[k]["pmfd"],
    ztmst=_ZTMST) for k in _KEYS}


@pytest.mark.parametrize("field", ("zmfuus", "zmfdus"))
def test_outputs_are_bitwise(field):
    bad = []
    for key in _KEYS:
        g, e = _GOT[key][field], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key, d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"{field}: {bad[:3]}"


def test_the_momentum_CAP_NEVER_BINDS_and_that_is_the_worst_gap_here():
    """THE CONSTANT THIS PORT IS MOST EXPOSED ON IS UNTESTED. Measured.

        largest pmfu/zmfmax anywhere in the fixture   0.5076
        the cap binds when that exceeds               1.0

    So zmfs is 1.0 on every one of the 108 columns, zmfuus == pmfu
    identically, and the block is graded ONLY in the sense that its guards
    are evaluated. Substituting zcons2 for zcons -- a 3x looser cap --
    changes nothing, because neither binds.

    WHY THIS IS WORSE THAN THE OTHER NAMED GAPS. zcons has exactly one
    consumer in the entire scheme and it is here; every other mass-flux
    cap uses zcons2, one character away. A wrong choice is finite,
    plausible, and off by a FIXED RATIO -- it would reach f012 looking
    like a physics result. And this is on the momentum path, where the
    port has no precedent: GF and KF both omit momentum coupling, so
    there is no WOOF-ArWen analogue to be wrong against. Both of this phase's
    errors landed on that path.

    THE GOOD NEWS IS THE FACTOR. At 0.5076 the fixture is short by 2x,
    not by orders of magnitude, so a modestly stronger updraft binds it.
    That makes this a case-table item like the :566 demotion rather than
    an unreachable branch.

    Written toward the gap: this FAILS when a case binds the cap, and
    that failure is the signal to require it -- and to re-enable the
    zcons2-substitution check below, which is the one that proves the
    constant choice is observable at all.
    """
    bound = [k for k in _KEYS if float(_GOT[k]["zmfs"]) < 1.0]
    assert not bound, (
        f"{len(bound)} columns now bind the momentum cap. That is a "
        "coverage GAIN on the port's most exposed constant: invert this "
        "assertion, re-enable the zcons2 substitution test, and remove "
        "the gap from docs/ntiedtke/PORT-RECORD.md.")


def test_the_headroom_to_binding_is_measured_not_guessed():
    """How far the fixture is from exercising it, so the case-table work
    is sized rather than open-ended.

    If this ratio ever drops, the fixture has moved AWAY from covering
    the constant and someone should know.
    """
    g, dt = np.float32(9.81), _ZTMST
    zcons = np.float32(np.float32(1.0) / np.float32(g * dt))
    worst = 0.0
    for key in _KEYS:
        c = _COLS[key]
        if not c["ldcum"]:
            continue
        paph, pmfu, ktop = _paph(c), c["pmfu"], c["kctop"]
        for jk in range(max(2, ktop), NT_NZ + 1):
            zmfmax = np.float32((paph[jk - 1] - paph[jk - 2]) * zcons)
            if zmfmax > 0:
                worst = max(worst, float(pmfu[jk - 1]) / float(zmfmax))
    assert 0.45 <= worst < 1.0, (
        f"the fixture's closest approach to the momentum cap is {worst:.4f}; "
        "it was 0.5076 when this was written. Above 1.0 the cap binds and "
        "the test above should have caught it; well below 0.45 the "
        "case-table work just got harder and the gap should be re-sized.")


# ===========================================================================
# The kernel
# ===========================================================================

@pytest.fixture(scope="module")
def cuda_mrescale():
    cp = pytest.importorskip("cupy")
    from woof.core.ntiedtke import NtLaunchGeometry, NtStages

    keys = _KEYS
    ncol, n1 = len(keys), NT_NZ + 2

    def pack(name, sfc=None):
        a = np.zeros((n1, ncol), dtype=np.float32)
        for c, k in enumerate(keys):
            a[1:1 + NT_NZ, c] = _COLS[k][name]
            if sfc:
                a[1 + NT_NZ, c] = _COLS[k][sfc]
        return cp.asarray(a)

    def ivec(name):
        return cp.asarray(np.array([int(_COLS[k][name]) for k in keys],
                                   dtype=np.int32))

    # Seeded with a sentinel so "written" is distinguishable from
    # "untouched": the second loop assigns EVERY level, so a surviving
    # sentinel means the kernel skipped work it must always do.
    out = {n: cp.full((n1, ncol), np.float32(-7.0), dtype=np.float32)
           for n in ("zmfuus", "zmfdus")}

    stages = NtStages(NtLaunchGeometry(ncol=ncol, nz=NT_NZ))
    stages.launch("ntiedtke_momentum_rescale", (
        ivec("ldcum"), ivec("kctop"), pack("paph", sfc="paph_sfc"),
        pack("pmfu"), pack("pmfd"), out["zmfuus"], out["zmfdus"],
        np.int32(ncol), np.int32(NT_NZ), _ZTMST,
        np.float32(1004.5), np.float32(287.0), np.float32(461.6),
        np.float32(2.5e6), np.float32(3.50e5), np.float32(9.81)))
    cp.cuda.Stream.null.synchronize()
    stages.check_geometry()
    return keys, {n: cp.asnumpy(v)[1:1 + NT_NZ, :] for n, v in out.items()}


@pytest.mark.parametrize("field", ("zmfuus", "zmfdus"))
def test_kernel_outputs_are_bitwise(cuda_mrescale, field):
    """Graded against WRF, never against the mirror."""
    keys, got = cuda_mrescale
    bad = []
    for c, key in enumerate(keys):
        g, e = got[field][:, c], _EXP[key][field]
        d = np.nonzero(g.view(np.uint32) != e.view(np.uint32))[0]
        if d.size:
            bad.append((key, d.tolist()[:4], g[d[0]], e[d[0]]))
    assert not bad, f"{field}: {bad[:3]}"


def test_the_kernel_writes_every_level(cuda_mrescale):
    """The -7.0 sentinel must be gone everywhere.

    :1007-1008 assigns zmfuus/zmfdus at EVERY level before the
    conditional scaling, so a level outside the cloud carries pmfu/pmfd
    through unchanged rather than becoming zero. A kernel that only wrote
    inside the cloud would look correct against an oracle whose
    out-of-cloud values happen to match its input buffer.
    """
    keys, got = cuda_mrescale
    for f in ("zmfuus", "zmfdus"):
        assert not np.any(got[f] == np.float32(-7.0)), \
            f"{f} kept its sentinel; the kernel did not write every level"


def test_kernel_holds_no_local_frame():
    pytest.importorskip("cupy")
    from woof.core.kernels import load_module
    a = load_module("ntiedtke").get_function(
        "ntiedtke_momentum_rescale").attributes
    assert a["local_size_bytes"] == 0, a["local_size_bytes"]
