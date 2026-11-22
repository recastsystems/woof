"""No workspace slot may be written by a stage and read by none.

WHAT THIS CATCHES, and why it exists. ``ntiedtke_momentum_rescale``
(cumastrn:996-1016) applies a per-column limiter to the mass flux and
writes the result to ``zmfuus``/``zmfdus``. cumastrn:1026 hands THOSE to
cududvn. WOOF-ArWen's ``ntiedtke_cududvn`` declares its parameters ``pmfu`` and
``pmfd`` -- the reference's own dummy names -- and the assembler binds
arguments by name, so it found the slots literally called ``pmfu`` and
``pmfd`` and passed the UNSCALED flux. The rescale's output was written and
read by nothing; the whole stage was dead.

The kernel's own header says "pmfu/pmfd MUST BE THE SCALED PAIR ... the
unscaled pair would be wrong on exactly the columns the rescaling touched".
The requirement was understood, written at the call site, and agreed with
nothing. That is this campaign's recurring failure -- resolution by
apparent identity -- and a comment cannot be the check.

Measured consequence before the fix: on a live 4.5 km column the limiter
binds at zmfs = 0.33333334 and the convective momentum came out exactly
1/zmfs = 3x WRF's. The 18-column analytic fixture never sees it: the cap
does not bind anywhere in it -- the closest column reaches 0.5076 -- so
every kernel still graded at max_ulp == 0.

WHAT THIS DOES NOT CATCH, stated here rather than left to be discovered.
This gate is necessary and NOT sufficient. It fires only because nothing
read ``zmfuus``. The property that actually failed is narrower: a parameter
named ``pmfu`` was bound to the slot named ``pmfu`` while its own contract
required the scaled pair. Had any other stage read ``zmfuus`` for any
reason, this gate would pass and cududvn would still be misbound. The
complement -- asserting each stage's consumers against the reference call
graph -- is not written. Do not read a green run here as "every binding is
right"; read it as "no stage's output is unreachable".
"""
from __future__ import annotations

import re
from pathlib import Path

from woof.core.ntiedtke import (NT_CALL_ORDER, NT_STAGE_ALIASES,
                                 NT_STAGE_SIGNATURE, nt_resolve)

_CU = (Path(__file__).resolve().parents[1] / "woof" / "core" / "kernels"
       / "ntiedtke.cu")

#: Written by the pipeline and read by the ADAPTER rather than by a stage.
#: These are the words that cross the driver boundary, plus the three
#: launch-geometry reports every kernel writes and the host inspects.
_ADAPTER_OUTPUTS = frozenset({
    "rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten",
    "raincv", "pratec",
    "geom_report", "order_report", "ticket",
})

#: Scratch a single kernel both writes and reads within one launch, so it
#: is non-const at its only appearance and can never be "read" by another
#: stage. cududvn's four zmf* arrays are the port's only such case: the
#: below-cloud taper reads zmf*[kcbot] at every jk > kcbot and the tendency
#: loop then reads jk+1 after the taper has rewritten it, so the whole
#: column has to be materialised (see the kernel header). Each entry names
#: the owning stage, so an array cannot be parked here by a later change
#: without saying which kernel owns it.
_STAGE_LOCAL_SCRATCH = {
    "zmfuu": "ntiedtke_cududvn",
    "zmfuv": "ntiedtke_cududvn",
    "zmfdu": "ntiedtke_cududvn",
    "zmfdv": "ntiedtke_cududvn",
    # cumastrn:1037-1052's own local: the KE-dissipation kernel forms
    # zuv2, sums it into zsum22, and spends it on ztdis in the next loop.
    "zuv2": "ntiedtke_ke_dissipation",
    # cutypen's parcel scratch.
    "scr": "ntiedtke_cutypen",
    "scr_i": "ntiedtke_cutypen",
}

#: Materialised so the DECOMPOSITION HARNESSES can grade cumastrn's locals
#: at max_ulp == 0. None of them feeds a later stage, and each row says why
#: that is correct rather than a missing consumer -- which is the whole
#: question this file exists to ask, so an entry here is a claim about the
#: reference and not a way to quiet the gate.
#:
#: THE TWO THAT LOOK LIKE DEFECTS AND ARE NOT. cuinin produces pqsenh and
#: klwmin, and cumastrn:492/:556/:1759 pass both onward to cutypen and
#: cuascn -- so "no ArWen stage reads them" reads as a gap. It is not: in
#: cu_ntiedtke.F90 they appear in cutypen's DECLARATIONS (:1841, :1859) and
#: nowhere in its body, and cuascn never names pqsenh at all. They are dead
#: arguments in the reference, and not passing them is faithful. Checked by
#: reading the bodies, because the argument list alone says the opposite.
_GRADED_DIAGNOSTICS = {
    "pqsenh": "ntiedtke_cuinin",
    "klwmin": "ntiedtke_cuinin",
    # cumastrn's own intent(inout) output (:349, :429): the updraft
    # precipitation total, accumulated at :2977 and consumed by nothing
    # inside the scheme.
    "prain": "ntiedtke_cuflxn",
    # prep's echo of the scheme timestep, graded by run_nt_prep.
    "delt_out": "ntiedtke_prep",
    # the closure's diagnostics -- run_cu_ntiedtke.F90's header names
    # ztauc, ztau, zcape1, zcape2 and zheat as exactly the cumastrn locals
    # that never leave the driver, which is why the port surfaces them.
    "zcape": "ntiedtke_closure",
    "zcape1": "ntiedtke_closure",
    "zcape2": "ntiedtke_closure",
    "zheat": "ntiedtke_closure",
    "ztau_o": "ntiedtke_closure",
    "ztaubl": "ntiedtke_closure",
    "ztauc": "ntiedtke_closure",
}


def _parse_constness():
    """{kernel: {param: is_const}} straight from the kernel source.

    A pointer parameter declared ``const`` is a READ; one declared without
    it is a WRITE (the port has no write-only-looking const parameters --
    CUDA would reject the store). Scalars are neither and are dropped.
    """
    src = _CU.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\(', src):
        i = m.end() - 1
        depth, j = 0, i
        while True:
            depth += 1 if src[j] == "(" else -1 if src[j] == ")" else 0
            if depth == 0 and src[j] == ")":
                break
            j += 1
        params = re.sub(r"/\*.*?\*/", "", src[i + 1:j], flags=re.S)
        params = re.sub(r"//[^\n]*", "", params)
        fields = {}
        for p in (x.strip() for x in params.split(",")):
            if not p or "*" not in p:
                continue                      # scalar, not a buffer
            name = re.sub(r"\[\s*\]$", "", p.split()[-1]).lstrip("*")
            fields[name] = bool(re.match(r"\bconst\b", p))
        out[m.group(1)] = fields
    return out


def _slot(stage: str, param: str) -> str:
    """The workspace array a stage's parameter actually binds to.

    Uses the assembler's OWN resolver rather than a second copy of the
    alias walk: NT_SEEDS carries alias rows as prose ("alias of X") that
    NT_ALIASES does not, and a gate that resolved only half of them would
    report a dozen live arrays as dead.
    """
    return nt_resolve(NT_STAGE_ALIASES.get((stage, param), param))


def _appearances(stage_alias=True):
    """{slot: {stage: is_const}} over every buffer parameter in the walk.

    CONST-NESS IS NOT USED AS THE TEST, and that is deliberate. Most of
    this scheme's arrays are intent(inout) at their consumers, so they are
    non-const at every appearance and a "written but never read as const"
    rule flags forty of them. The port already records this hole -- the
    class-1 const scan is silent about in/out parameters, which is how
    pmfub and idtop both got through.

    The property used instead is stronger where it matters and has no
    false positives here: an array that ONE kernel names and no other
    kernel mentions at all cannot be feeding anything downstream.
    """
    const = _parse_constness()
    out = {}
    for stage in NT_CALL_ORDER:
        for param, is_const in const.get(stage, {}).items():
            slot = _slot(stage, param) if stage_alias else (
                nt_resolve(param))
            out.setdefault(slot, {})[stage] = is_const
    return out


def test_the_scan_sees_the_kernels():
    """A gate that scans nothing passes vacuously.

    Four pattern-driven gates in this tree have passed while matching no
    corpus at all, so the patterns are guarded and not just the inputs.
    """
    const = _parse_constness()
    assert len(const) >= 21, f"only {len(const)} kernels parsed"
    assert set(NT_CALL_ORDER) <= set(const), (
        "walk stages the parser did not see: "
        f"{sorted(set(NT_CALL_ORDER) - set(const))}")
    seen = _appearances()
    assert len(seen) > 60, f"only {len(seen)} slots found"
    assert sum(len(v) for v in seen.values()) > 300, "the scan is short"
    # Multi-stage sharing must actually be observed, or "named by exactly
    # one stage" is trivially false for the wrong reason.
    assert sum(1 for v in seen.values() if len(v) > 1) > 30


def _dead_slots(stage_alias=True):
    dead = {}
    for slot, stages in _appearances(stage_alias).items():
        if len(stages) != 1 or slot in _ADAPTER_OUTPUTS:
            continue
        (stage,), = (tuple(stages),)
        if stages[stage]:
            continue                    # const-only: consumed, not produced
        if _STAGE_LOCAL_SCRATCH.get(slot) == stage:
            continue
        if _GRADED_DIAGNOSTICS.get(slot) == stage:
            continue
        dead[slot] = stage
    return dead


def test_every_allowlisted_slot_is_really_produced_there():
    """The allowlists cannot name something that has since moved.

    An entry that no longer matches any stage is a stale excuse, and a
    stale excuse is indistinguishable from cover for a real dead write --
    which is the shape of four gates in this tree that went quiet.
    """
    seen = _appearances()
    for table, label in ((_STAGE_LOCAL_SCRATCH, "scratch"),
                         (_GRADED_DIAGNOSTICS, "diagnostic")):
        for slot, stage in table.items():
            assert slot in seen, f"{label} {slot!r} is no longer a bound slot"
            assert stage in seen[slot], (
                f"{label} {slot!r} is excused for {stage!r} but is bound by "
                f"{sorted(seen[slot])}")
            assert not seen[slot][stage], (
                f"{label} {slot!r} is const at {stage!r}, so it is consumed "
                f"there and needs no excuse")


def test_no_stage_output_is_unreachable():
    dead = _dead_slots()
    assert not dead, (
        "these workspace slots are written by one stage and named by no "
        "other, so the work that produced them cannot reach the forecast:\n"
        + "\n".join(f"  {s} <- {w}" for s, w in sorted(dead.items()))
        + "\nEither a consumer is bound to the wrong slot (see "
          "NT_STAGE_ALIASES) or the producing stage is doing nothing.")


def test_cududvn_consumes_the_rescaled_mass_flux():
    """The specific binding, pinned so a refactor cannot quietly undo it.

    Belt and braces with the structural gate above: that one fails if
    NOTHING reads zmfuus/zmfdus, this one fails if cududvn in particular
    stops reading them, which is the case that actually cost 3x.
    """
    assert _slot("ntiedtke_cududvn", "pmfu") == "zmfuus"
    assert _slot("ntiedtke_cududvn", "pmfd") == "zmfdus"
    # and the reference's own names still arrive as parameters, so this
    # test is about the BINDING and not about a rename
    assert "pmfu" in NT_STAGE_SIGNATURE["ntiedtke_cududvn"]
    assert "pmfd" in NT_STAGE_SIGNATURE["ntiedtke_cududvn"]
    # every OTHER stage taking pmfu still means the unscaled array
    others = [s for s in NT_CALL_ORDER
              if s != "ntiedtke_cududvn"
              and "pmfu" in NT_STAGE_SIGNATURE.get(s, ())]
    assert others, "no other stage takes pmfu; the alias is untested"
    for stage in others:
        assert _slot(stage, "pmfu") == "pmfu", stage


def test_the_gate_would_have_caught_the_defect():
    """The negative control: without the alias, the gate must FAIL.

    A gate that has never been shown to fire on the thing it was written
    for is not evidence. This reconstructs the pre-fix binding and asserts
    zmfuus/zmfdus come out dead.
    """
    dead = _dead_slots(stage_alias=False)      # the pre-fix binding
    for slot in ("zmfuus", "zmfdus"):
        assert slot in dead, (
            f"{slot} is not flagged without the stage alias, so the "
            f"dead-write gate would NOT have caught this defect and the "
            f"claim in this file's docstring is wrong")
        assert dead[slot] == "ntiedtke_momentum_rescale"
    # and with the alias in place they are clean, so the gate distinguishes
    # the two trees rather than failing or passing regardless
    assert not (set(_dead_slots()) & {"zmfuus", "zmfdus"})
