"""Architecture admission, and the record of which architectures hold an anchor.

Every NVIDIA architecture that can build and launch this port's kernels runs
it, by default, the way every other model runs on whatever card it is given.
Nothing in this module refuses a card for lacking an anchor.

WHAT AN ANCHOR IS.  The port's numerical contract (the FTZ/subnormal
behaviour of the kernels' arithmetic, the NVRTC contraction pin and the
frozen authority digests) was first measured on sm_120, compute capability
12.0 (:data:`PROVEN_COMPUTE`).  An anchor is that contract measured again on
real hardware of another architecture:

* a **contract receipt**: the measured subnormal/FTZ behaviour of the
  kernels' actual arithmetic and the contraction pin under the port's own
  NVRTC flags, on a card of that architecture; and
* an **authority anchor**: the frozen-authority verification run on that
  architecture, recording per item either byte-identity with the sm_120
  masked digests or a stable, twice-reproduced per-architecture digest set.

:data:`ADMITTED_BELOW_FLOOR` is that record.  Its name is older than the open
admission, from when a row was also a permission; it is now evidence only.
A card of an anchored architecture runs on a contract somebody measured on
that architecture, a card of any other architecture runs too, and every
receipt says which of the two it was (:func:`architecture_status`).  A
receipt never claims an anchor its architecture does not hold.

Each entry pins the exact evidence it rests on: ``evidence_sha256`` is
:func:`evidence_digest` over the evidence directory its contract receipt sits
in.  A tree that carries the receipts checks the pin against the files (and
opens them: the verdict, the card, the authority digests and the timing
readings each have to say what the entry says); a tree that ships without
receipts still names, byte for byte, the one record its entry was admitted
on.

WHAT STILL REFUSES, EACH NAMING ITS BREAKAGE (the gate law).

* **A card NVRTC cannot compile for** (:func:`nvrtc_refusal`).  Every kernel
  this port launches is compiled by NVRTC when the run starts, for the card's
  own ``compute_XY`` target.  The kernels ask for nothing an older card
  lacks: 32-bit integer ``atomicExch``, static shared memory under 8 KiB a
  block, IEEE FP64 arithmetic and float/integer bit reinterpretation; no warp
  intrinsics, no half or bfloat16 types, no inline PTX and no cooperative
  groups (a census of every CUDA source string in this package, 2026-09-29).
  So the floor is the toolchain's, not the kernels': the port runs on a
  CUDA 12 or CUDA 13 runtime (``require_cuda``'s ``min_runtime_version`` is
  12000), and the cards it can build for are the ones that toolkit's NVRTC
  targets (CUDA 13: compute_75, Turing, and newer; CUDA 12.8 and newer also
  reach compute_120).  The check asks NVRTC itself
  (:func:`nvrtc_target_archs`), so a toolkit that adds or drops a target
  moves the answer without an edit here.
* **A card whose predicted memory does not fit**, in
  :mod:`woof.hex.device_admission`, unchanged.
* **A card whose ordinary arithmetic is not IEEE** (:func:`numeric_route_refusal`),
  from :func:`measure_numeric_route`, which the forecast door runs on an
  unanchored card.

HOW THE RUN'S OWN GATE READS THIS MODULE.  ``cuda_backend.runtime.require_cuda``
asks :func:`admitted_architecture` about a card below its ``min_compute`` and
refuses with :func:`below_floor_refusal` when the answer is ``None``.  Both
now answer the NVRTC question and nothing else.  ``runtime.py`` itself is
left as it is: it is one of the fourteen sources the regional kernel-set
digest covers (``regional_admission.REGIONAL_KERNEL_SOURCES``), so moving one
of its bytes would lapse every minted limited-area class.  Its comment at
that call still describes the per-architecture gate this module retired.
Its ``required_compute`` pin is read only for a card at or above
``min_compute``, so a ``required_compute`` of ``(12, 0)`` now refuses only
a part newer than sm_120.  The forecast door, the driver, the proof harness and
every tool whose bytes nothing pins no longer pass it; the few call sites
left are the ones a recorded byte pin holds
(``tests/test_arch_admission.py``, ``PINNED_TO_SM120``).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


#: The architecture the port's numerical contract was originally proven on.
#: Everything below this consults :data:`ADMITTED_BELOW_FLOOR`.
PROVEN_COMPUTE: tuple[int, int] = (12, 0)


@dataclass(frozen=True, slots=True)
class ArchAnchor:
    """The evidence that admits one architecture below the proven floor."""

    compute: tuple[int, int]
    card: str
    admitted_on: str
    contract_receipt: str
    authority_anchor: str
    basis: str
    #: :func:`evidence_digest` of :attr:`evidence_directory`, as committed.
    #: Empty only on a campaign's in-process CANDIDATE row, which rests on no
    #: evidence yet; every registered entry carries one (the tests hold it).
    evidence_sha256: str = ""

    @property
    def sm(self) -> str:
        return f"sm_{self.compute[0]}{self.compute[1]}"

    @property
    def evidence_directory(self) -> str:
        """The directory the contract receipt sits in, relative to the tree."""

        return Path(self.contract_receipt).parent.as_posix()

    def as_dict(self) -> dict[str, object]:
        return {
            "compute_capability": f"{self.compute[0]}.{self.compute[1]}",
            "sm": self.sm,
            "card": self.card,
            "admitted_on": self.admitted_on,
            "contract_receipt": self.contract_receipt,
            "authority_anchor": self.authority_anchor,
            "basis": self.basis,
            "evidence_sha256": self.evidence_sha256,
        }


def evidence_digest(directory: str | Path) -> str:
    """One SHA-256 over every file of an anchor's evidence directory.

    Files are taken in POSIX relative-path order (bytecode caches skipped),
    one ``<relative path>\\0<sha256 of the file's bytes>\\n`` line each, and
    the digest is the SHA-256 of those lines.  The repository checks text
    out as LF on every platform (``.gitattributes``), so the digest is the
    same on every checkout of one commit.  A file added, removed or changed
    anywhere in the directory moves it.
    """

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"no evidence directory at {root}")
    entries = []
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        entries.append(
            (
                path.relative_to(root).as_posix(),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    lines = "".join(f"{relative}\0{digest}\n" for relative, digest in sorted(entries))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


#: Architectures below :data:`PROVEN_COMPUTE` holding a verified anchor.
#: An entry lands only together with the receipt and authority evidence it
#: names, minted on real hardware of that architecture.
ADMITTED_BELOW_FLOOR: Mapping[tuple[int, int], ArchAnchor] = MappingProxyType(
    {
        (8, 6): ArchAnchor(
            compute=(8, 6),
            card=(
                "NVIDIA GeForce RTX 3080 (10,240 MiB, 68 SM, WDDM), "
                "driver 610.74"
            ),
            admitted_on="2026-08-25",
            contract_receipt="evidence/sm86-tier-20260825/RECEIPT.md",
            authority_anchor="evidence/sm86-tier-20260825/authority",
            basis=(
                "sm86-tier campaign: FTZ/contraction contract measured on "
                "this card by the port's own decks and the engine probe "
                "arms; determinism and authority anchoring per the campaign "
                "receipt"
            ),
            evidence_sha256=(
                "75437c663c9894dc38e6a1b1517709906e08a14c9a56468016aec09a8fa93155"
            ),
        ),
        (8, 9): ArchAnchor(
            compute=(8, 9),
            card="NVIDIA GeForce RTX 4090 (24,564 MiB, 128 SM), driver 610.57.04",
            admitted_on="2026-09-24",
            contract_receipt="evidence/sm89-tier-20260924/RECEIPT.md",
            authority_anchor="evidence/sm89-tier-20260924/authority",
            # The campaign runner is cited where the pinned evidence carries
            # it (instruments/, beside its SHA256SUMS).  The campaign proposed
            # the row naming the path it ran from, tools/, which this tree
            # does not carry; the evidence tests hold that one difference.
            basis=(
                "sm89-tier campaign (evidence/sm89-tier-20260924/instruments/"
                "arch_anchor_campaign.py): FTZ route "
                "grid, contraction pin and the port's contract decks measured "
                "on this card, dual-run; determinism and authority anchoring "
                "per the campaign receipt"
            ),
            evidence_sha256=(
                "15357eb4d576853f49dba0aacfb7f7deaee36c2bdbda6aa667b8f55e4d32161b"
            ),
        ),
    }
)


#: Per-architecture ceilings for the FTZ guarded-fallback timing control
#: (``cuda_ftz.run_normalized_fallback_performance_control``), beside the
#: admission registry because the two answer the same question -- what is
#: proven on THIS architecture -- and drift apart when kept in different
#: files.  Made per-architecture 2026-08-25 (stale-guard audit,
#: finding 8): the single global 1.25 was calibrated when sm_120 was the
#: only architecture and hard-refused the newly admitted sm_86 tier on a
#: timing-only deviation while bitwise identity held.  Timings are
#: SECONDARY evidence everywhere this ceiling is consulted; correctness is
#: bitwise enabled/disabled identity, which no ceiling relaxes.
PERFORMANCE_RATIO_CEILINGS: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        "sm_120": MappingProxyType(
            {
                "ceiling": 1.25,
                "basis": (
                    "the original sm_120 calibration: the five named "
                    "normalized-kernel microbenchmarks each measured below "
                    "1.25x median enabled/disabled on the proven-floor "
                    "card, and every archived sm_120 binding declares this "
                    "ceiling -- unchanged, so those receipts stay valid"
                ),
            }
        ),
        "sm_86": MappingProxyType(
            {
                "ceiling": 1.75,
                "basis": (
                    "set from the RECORDED sm_86 deviation, not a fresh "
                    "calibration: the desktop RTX 3080 measured 1.471975x "
                    "and 1.565028x at transport.transport_edge_values "
                    "(bitwise identity held; evidence/sm86-tier-20260825/"
                    "contract/perf-control-stability-sm86.json, STATE.md "
                    "section 8) while the four other benchmarks passed at "
                    "1.25.  1.75 covers the top recorded reading with the "
                    "recorded run-to-run spread (~0.093) as headroom; a "
                    "fresh multi-run calibration on the 3080 is the named "
                    "follow-up that replaces this basis"
                ),
            }
        ),
        "sm_89": MappingProxyType(
            {
                "ceiling": 1.55,
                "basis": (
                    "a cross-process calibration on the RTX 4090 (the "
                    "calibration receipt perf-calibration-sm89.json in a "
                    "calibration/ folder of the evidence this row pins): "
                    "200 readings of the "
                    "guard-cost control in 10 separate processes, each "
                    "compiling into its own empty kernel cache and naming its "
                    "card, bitwise enabled/disabled identity held in every "
                    "one: median 1.174x, top 1.392857x; 52 readings breached "
                    "1.25x (49 at transport.transport_edge_values, 3 at "
                    "recovery.recover_edge_velocity_f32), 5 breached 1.35x "
                    "and none 1.40x.  1.55 is the top reading plus the spread "
                    "of the breaching readings (~0.143), rounded up to 0.05.  "
                    "It replaces 1.35, which was set by the same rule from 24 "
                    "readings in one warm process "
                    "(evidence/sm89-tier-20260924/contract/"
                    "perf-control-stability-sm89.json and "
                    "evidence/sm89-tier-20260924/contract/"
                    "perf-calibration-sm89.json, top 1.285714x) and which 5 "
                    "of the 200 exceed"
                ),
            }
        ),
    }
)


def performance_ratio_ceiling(sm: str) -> float:
    """The FTZ guard-cost timing ceiling for one architecture, by ``sm_NN``.

    Refuses an unregistered architecture by name: a ceiling nobody measured
    or recorded for this silicon would turn the timing control into a
    number invented at the call site.
    """

    row = PERFORMANCE_RATIO_CEILINGS.get(sm)
    if row is None:
        roster = ", ".join(sorted(PERFORMANCE_RATIO_CEILINGS))
        raise LookupError(
            f"no FTZ performance-ratio ceiling is registered for {sm}: the "
            f"guard-cost timing control has no measured or recorded bound "
            f"on this architecture, so a verdict from it would be an "
            f"invented number; registered architectures: {roster}"
        )
    return float(row["ceiling"])  # type: ignore[arg-type]


def architecture_anchor(compute: tuple[int, int]) -> ArchAnchor | None:
    """The registered anchor row of ``compute``, or ``None``.

    A lookup in the evidence record and nothing more: ``None`` means no
    anchor, never a refusal.
    """

    return ADMITTED_BELOW_FLOOR.get((int(compute[0]), int(compute[1])))


@dataclass(frozen=True, slots=True)
class ArchitectureAdmission:
    """One card's architecture: admitted, and whether it holds an anchor."""

    compute: tuple[int, int]
    #: The registry row, when this architecture holds one.
    anchor: ArchAnchor | None = None

    @property
    def sm(self) -> str:
        return f"sm_{self.compute[0]}{self.compute[1]}"

    @property
    def proven_floor(self) -> bool:
        """This is the architecture the contract was first measured on."""

        return tuple(self.compute) == PROVEN_COMPUTE

    @property
    def anchored(self) -> bool:
        return self.proven_floor or self.anchor is not None

    @property
    def status(self) -> str:
        return "anchored" if self.anchored else "unanchored"

    def as_dict(self) -> dict[str, object]:
        """The receipt row.  ``evidence_sha256`` is only ever a registered
        row's own pin; the proven floor and an unanchored card carry None."""

        if self.anchor is not None:
            anchor = f"{self.anchor.sm} anchor of {self.anchor.admitted_on}"
            basis = self.anchor.basis
        elif self.proven_floor:
            anchor = "the proven contract floor"
            basis = (
                "the architecture the numerical contract and the frozen "
                "authority digests were first measured on"
            )
        else:
            anchor = None
            basis = (
                "no anchor: the numerical contract has not been measured on "
                "this architecture, so nothing this card computes has been "
                "compared with the anchored architectures' results"
            )
        return {
            "compute_capability": f"{self.compute[0]}.{self.compute[1]}",
            "sm": self.sm,
            "anchor_status": self.status,
            "anchor": anchor,
            "evidence_sha256": (
                None if self.anchor is None else (self.anchor.evidence_sha256 or None)
            ),
            "contract_receipt": (
                None if self.anchor is None else self.anchor.contract_receipt
            ),
            "basis": basis,
        }

    def describe(self) -> str:
        """The door's plain sentence: anchored or not, and on what."""

        if self.anchor is not None:
            pin = self.anchor.evidence_sha256
            pinned = f", evidence pin {pin[:16]}..." if pin else ""
            return (
                f"anchored: the {self.anchor.admitted_on} anchor "
                f"({self.anchor.contract_receipt}{pinned})"
            )
        if self.proven_floor:
            return (
                "anchored: the proven contract floor, where the numerical "
                "contract was first measured"
            )
        return (
            "unanchored: no per-architecture anchor holds a measured "
            "numerical contract for this architecture; the run goes ahead and "
            "its receipts record it as unanchored"
        )


def architecture_status(compute: tuple[int, int]) -> ArchitectureAdmission:
    """What a receipt records about ``compute``.  Asks nothing of NVRTC."""

    compute = (int(compute[0]), int(compute[1]))
    return ArchitectureAdmission(compute=compute, anchor=architecture_anchor(compute))


def nvrtc_target_archs() -> tuple[int, ...]:
    """The ``compute_XY`` targets this install's NVRTC generates code for.

    Read from NVRTC right now (``nvrtcGetSupportedArchs``), as integers
    ``10 * major + minor``.  Raises whatever loading or asking NVRTC raised.
    """

    from cupy.cuda import nvrtc

    return tuple(sorted(int(value) for value in nvrtc.getSupportedArchs()))


def _target(compute: tuple[int, int]) -> int:
    return int(compute[0]) * 10 + int(compute[1])


def nvrtc_refusal(compute: tuple[int, int], targets: tuple[int, ...]) -> str:
    """The named refusal for a card NVRTC cannot compile for."""

    listed = ", ".join(f"compute_{value}" for value in targets) or "nothing"
    return (
        f"cuda.compute_capability={compute[0]}.{compute[1]} "
        f"(sm_{compute[0]}{compute[1]}): the NVRTC in this CUDA install "
        f"generates code for {listed}, and not for compute_{_target(compute)}.  "
        f"Every kernel this port launches is compiled by NVRTC for the card "
        f"when the run starts, so on this card none of them could be built "
        f"and the run would stop at its first kernel.  The kernels "
        f"themselves need nothing newer than the oldest of those targets; "
        f"the floor is the installed CUDA toolkit's NVRTC, so a toolkit "
        f"whose NVRTC targets this card (for compute_120, CUDA 12.8 or "
        f"newer) is the remedy"
    )


def admitted_architecture(
    compute: tuple[int, int], *, targets: tuple[int, ...] | None = None
) -> ArchitectureAdmission | None:
    """Admit ``compute``, or ``None`` when NVRTC cannot compile for it.

    ``None`` is the only refusal, and ``require_cuda`` turns it into
    :func:`below_floor_refusal`.  ``targets`` defaults to asking NVRTC; when
    NVRTC cannot be asked at all there is nothing here to refuse on, and
    the run's own ``from cupy.cuda import nvrtc`` names that failure.
    """

    compute = (int(compute[0]), int(compute[1]))
    if targets is None:
        try:
            targets = nvrtc_target_archs()
        except Exception:
            targets = None
    if targets is not None and _target(compute) not in targets:
        return None
    return architecture_status(compute)


def admitted_summary() -> str:
    """The architectures holding an anchor below the floor, for messages."""

    anchors = sorted(ADMITTED_BELOW_FLOOR.values(), key=lambda a: a.compute)
    if not anchors:
        return "none"
    return ", ".join(anchor.sm for anchor in anchors)


def below_floor_refusal(
    compute: tuple[int, int], floor: tuple[int, int]
) -> str:
    """The refusal ``require_cuda`` raises when :func:`admitted_architecture`
    answers ``None``: the card is one NVRTC cannot compile for.

    ``floor`` is the caller's ``min_compute``, kept for that call's
    signature; it no longer decides anything.
    """

    try:
        targets = nvrtc_target_archs()
    except Exception as error:  # pragma: no cover - require_cuda asked first
        return (
            f"cuda.compute_capability={compute[0]}.{compute[1]} "
            f"(sm_{compute[0]}{compute[1]}): NVRTC could not be asked which "
            f"targets it compiles for ({error}), and every kernel this port "
            f"launches is compiled by NVRTC when the run starts"
        )
    return nvrtc_refusal(compute, targets)


# ---------------------------------------------------------------------------
# the numeric route, measured on the card in hand
# ---------------------------------------------------------------------------
NUMERIC_ROUTE_SCHEMA = "gpuwm-hex.numeric-route/v1"

#: The compile options every port kernel is built with under strict numerics
#: (``KernelCache``'s ``base_options``); CuPy appends ``-ftz=true`` itself.  The probe
#: compiles with the run's own mode (:func:`woof.hex.numerics.kernel_base`), fast by default.
NUMERIC_ROUTE_OPTIONS: tuple[str, ...] = ("--std=c++17", "--fmad=false")

#: An FP32 subnormal (the transport deck's own witness value, 0x000116c2).
_SUBNORMAL_BITS = 0x000116C2

_PROBE_KERNEL = r"""
extern "C" __global__ void hex_numeric_route_probe(
    const unsigned int* f32_in, const unsigned long long* f64_in,
    unsigned int* f32_out, unsigned long long* f64_out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const float sub = __uint_as_float(f32_in[0]);
    const float one = __uint_as_float(f32_in[1]);
    const float zero = __uint_as_float(f32_in[2]);
    const float a = __uint_as_float(f32_in[3]);
    const float c = __uint_as_float(f32_in[4]);
    const float x = __uint_as_float(f32_in[5]);
    const float y = __uint_as_float(f32_in[6]);
    f32_out[0] = __float_as_uint(sub * one);
    f32_out[1] = __float_as_uint(sub + zero);
    f32_out[2] = __float_as_uint(mpas_mul(sub, one));
    f32_out[3] = __float_as_uint(mpas_add(sub, zero));
    f32_out[4] = __float_as_uint(a * a + c);
    f32_out[5] = __float_as_uint(x * y);
    f32_out[6] = __float_as_uint(x + y);
    const double dx = __longlong_as_double((long long)f64_in[0]);
    const double dy = __longlong_as_double((long long)f64_in[1]);
    f64_out[0] = (unsigned long long)__double_as_longlong(dx * dy);
    f64_out[1] = (unsigned long long)__double_as_longlong(dx + dy);
    f64_out[2] = (unsigned long long)__double_as_longlong(
        mpas_f32_to_f64_ieee(sub) * 1.0);
}
"""


def _probe_inputs():
    import numpy as np

    wide = np.float32(1.0) + np.float32(2.0**-23)
    f32 = np.array(
        [
            np.uint32(_SUBNORMAL_BITS).view(np.float32),
            1.0,
            0.0,
            wide,
            -(np.float32(1.0) + np.float32(2.0**-22)),
            1.1,
            3.3,
        ],
        dtype=np.float32,
    )
    f64 = np.array([1.1, 3.3], dtype=np.float64)
    return f32, f64


def _expected_bits() -> dict[str, int]:
    """The IEEE answers, from the host's own binary32/binary64 arithmetic."""

    import numpy as np

    f32, f64 = _probe_inputs()
    bits32 = lambda value: int(np.float32(value).view(np.uint32))  # noqa: E731
    bits64 = lambda value: int(np.float64(value).view(np.uint64))  # noqa: E731
    with np.errstate(all="ignore"):
        return {
            "subnormal": _SUBNORMAL_BITS,
            "separately_rounded": bits32(np.float32(f32[3] * f32[3]) + f32[4]),
            "x_times_y": bits32(f32[5] * f32[6]),
            "x_plus_y": bits32(f32[5] + f32[6]),
            "dx_times_dy": bits64(f64[0] * f64[1]),
            "dx_plus_dy": bits64(f64[0] + f64[1]),
            "subnormal_in_fp64": bits64(np.float64(f32[0])),
        }


def classify_numeric_route(
    f32_out: tuple[int, ...], f64_out: tuple[int, ...], numerics: str = "strict"
) -> dict[str, object]:
    """Turn the probe's raw output bits into the route's named behaviours."""

    expected = _expected_bits()
    subnormal = expected["subnormal"]
    plain = (int(f32_out[0]) & 0x7FFFFFFF, int(f32_out[1]) & 0x7FFFFFFF)
    if plain == (0, 0):
        ftz_route = "flush-to-zero"
    elif plain == (subnormal, subnormal):
        ftz_route = "ieee"
    else:
        ftz_route = "mixed"
    guarded = (
        "ieee"
        if (int(f32_out[2]), int(f32_out[3])) == (subnormal, subnormal)
        else "differs"
    )
    contraction = (
        "separately-rounded"
        if int(f32_out[4]) == expected["separately_rounded"]
        else "not-separately-rounded"
    )
    normal_range = (
        "ieee"
        if (
            int(f32_out[5]) == expected["x_times_y"]
            and int(f32_out[6]) == expected["x_plus_y"]
            and int(f64_out[0]) == expected["dx_times_dy"]
            and int(f64_out[1]) == expected["dx_plus_dy"]
        )
        else "differs"
    )
    fp64_keeps = int(f64_out[2]) == expected["subnormal_in_fp64"]
    return {
        "ftz_route": ftz_route,
        "guarded_fallback": guarded,
        "contraction": contraction,
        "normal_range": normal_range,
        "fp64_keeps_binary32_subnormals": fp64_keeps,
        # Strict: the behaviour every anchored architecture measured (sm_120,
        # sm_89, sm_86): the route flushes, the guarded fallback restores the
        # IEEE answer, --fmad=false keeps a*b+c separately rounded.  Fast: the
        # route flushes and ordinary arithmetic is IEEE; contraction is on and
        # the guarded fallback is compiled out by design (woof.hex.numerics).
        "numerics": numerics,
        "matches_anchored_route": (
            ftz_route == "flush-to-zero"
            and guarded == "ieee"
            and contraction == "separately-rounded"
            and normal_range == "ieee"
            and fp64_keeps
        ) if numerics == "strict" else (
            ftz_route == "flush-to-zero"
            and normal_range == "ieee"
            and fp64_keeps
        ),
        "f32_bits": [f"0x{int(value):08x}" for value in f32_out],
        "f64_bits": [f"0x{int(value):016x}" for value in f64_out],
    }


def measure_numeric_route(cupy_module=None) -> dict[str, object]:
    """Compile one ten-line kernel the way every port kernel is compiled
    and launch it on this card, twice.

    It measures the four behaviours the numerical contract is made of: what
    plain FP32 arithmetic does with a subnormal operand on this route, what
    the port's guarded fallback (``cuda_fp32.CUDA_FTZ_HELPERS``) returns for
    the same operand, whether ``--fmad=false`` keeps ``a*b+c`` separately
    rounded, and whether ordinary FP32/FP64 arithmetic agrees with IEEE.
    One NVRTC compile and two single-thread launches: well under a second
    after the CUDA context exists.  The compile cache goes to a temporary
    directory that is removed afterwards.
    """

    import os
    import tempfile
    import time

    import numpy as np

    from ..cuda_fp32 import CUDA_FTZ_HELPERS
    from .. import numerics as _numerics

    route_options = _numerics.kernel_base()
    if cupy_module is None:
        import cupy as cupy_module
    cp = cupy_module
    if getattr(cupy_module, "__name__", None) == "cupy":
        from .aot import install
        install(cp)
    started = time.perf_counter()
    f32_in, f64_in = _probe_inputs()
    previous = os.environ.get("CUPY_CACHE_DIR")
    from .cupy_cache import bound
    with tempfile.TemporaryDirectory(
        prefix="hex-numeric-route-", ignore_cleanup_errors=True
    ) as cache, bound(cache):
        try:
            module = cp.RawModule(
                code=CUDA_FTZ_HELPERS + _PROBE_KERNEL,
                options=route_options,
                backend="nvrtc",
                enable_cooperative_groups=False,
            )
            kernel = module.get_function("hex_numeric_route_probe")
            runs = []
            for _ in range(2):
                f32_out = cp.zeros(7, dtype=cp.uint32)
                f64_out = cp.zeros(3, dtype=cp.uint64)
                kernel(
                    (1,),
                    (1,),
                    (
                        cp.asarray(f32_in.view(np.uint32)),
                        cp.asarray(f64_in.view(np.uint64)),
                        f32_out,
                        f64_out,
                    ),
                )
                cp.cuda.runtime.deviceSynchronize()
                runs.append(
                    (
                        tuple(int(v) for v in f32_out.get()),
                        tuple(int(v) for v in f64_out.get()),
                    )
                )
        finally:
            if previous is None:
                os.environ.pop("CUPY_CACHE_DIR", None)
            else:
                os.environ["CUPY_CACHE_DIR"] = previous
    record = classify_numeric_route(*runs[0], numerics=_numerics.mode())
    record.update(
        schema=NUMERIC_ROUTE_SCHEMA,
        options=list(route_options),
        dual_run_identical=runs[0] == runs[1],
        seconds=round(time.perf_counter() - started, 3),
    )
    return record


def numeric_route_refusal(measurement: dict[str, object]) -> str | None:
    """The named refusal a numeric-route measurement calls for, or ``None``.

    Only ordinary arithmetic disagreeing with IEEE refuses: every kernel of
    the dycore is ordinary FP32/FP64 arithmetic, so on such a card every
    number the run produced would be wrong, not merely unanchored.  The
    other behaviours are recorded and do not refuse.  A route that does not
    flush, a guarded fallback that does not restore the IEEE answer, or a
    contracted ``a*b+c`` each move results in their last bits (at subnormal
    magnitudes, below 1.2e-38, or by one rounding), which is exactly what an
    unanchored receipt already says it has not ruled out.
    """

    if measurement.get("normal_range") == "ieee":
        return None
    return (
        "ordinary arithmetic on this card does not agree with IEEE 754: the "
        "numeric-route probe computed 1.1*3.3 and 1.1+3.3 in binary32 and "
        "binary64 and got "
        f"{measurement.get('f32_bits')} / {measurement.get('f64_bits')}, "
        "which is not what the host's IEEE arithmetic gives.  Every kernel of "
        "the dycore is that arithmetic, so every number the run produced "
        "would be wrong.  Check the driver and the card before anything else"
    )


__all__ = [
    "ADMITTED_BELOW_FLOOR",
    "ArchAnchor",
    "ArchitectureAdmission",
    "NUMERIC_ROUTE_OPTIONS",
    "NUMERIC_ROUTE_SCHEMA",
    "PERFORMANCE_RATIO_CEILINGS",
    "PROVEN_COMPUTE",
    "admitted_architecture",
    "admitted_summary",
    "architecture_anchor",
    "architecture_status",
    "below_floor_refusal",
    "classify_numeric_route",
    "evidence_digest",
    "measure_numeric_route",
    "numeric_route_refusal",
    "nvrtc_refusal",
    "nvrtc_target_archs",
    "performance_ratio_ceiling",
]
