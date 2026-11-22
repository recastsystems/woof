"""The dual-run byte comparison, on a no-ECC card.

The 5090 has no ECC, so a bit that flips in VRAM produces a plausible
forecast and nothing says otherwise. Running the identical configuration
twice and comparing bytes is the standing corruption detector; a
mismatch is a corruption finding that stops the claim, full stop.

The screen this reproduces is the one the shipped 250 m demonstration
committed (`docs/internal/receipts/les/inflow-fetch/dual_run_screen.txt`)
and reports the same lines. That script lived in a measurement-wave
scratch directory and was rewritten from scratch each campaign, which is
how the same misclassification could be re-made twice; this is that
re-implementation, in the tree, with a test.

Three comparisons, because they fail differently:

* **wrfout bytes.** The forecast itself. Byte-for-byte over every frame.
* **final-state digests.** The run receipts' own state digests, which
  cover arrays the history files do not carry.
* **non-environmental receipt scalars.** The receipt blocks with
  `ENVIRONMENTAL_FIELDS` excluded -- wall times, device names, host
  paths and the like legitimately differ between two runs and are not
  evidence of anything. The partition is the one
  `woof.verify.cases.convective_boundary_layer` already publishes, so
  the exclusion list is not invented here.

An exclusion is never a silent drop. Every excluded key whose two arms
DISAGREE is printed, with the reason it was excluded, above the verdict
line -- so a reader sees exactly what the screen chose not to count and
can overrule it. An exclusion nobody can see is an exclusion nobody can
audit, and the difference between that and a hidden corruption is a
reader's trust.

**And a screen that compared nothing REFUSES.** Because the comparison
was total over whatever it happened to find, it was also vacuous on
nothing, and that was a hole wide enough to drive the whole instrument
through: seven separate arrangements -- two runs with no receipts, two
with receipts that would not parse, a receipt only one arm wrote,
receipts saved under different names, zero-byte history files, a receipt
carrying no digest, and the same directory handed in as both arms --
each printed ``DUAL-RUN SCREEN: PASS (bit-identical)`` beside
``final-state digests compared: 0``. On a card with no ECC that is not
a weak detector, it is a false one, and a false detector is worse than
none because the run stops being watched.

So ``PASS`` is no longer a flag any branch sets. It is derived from
:class:`Evidence`, the counters of what was actually compared, and it is
unreachable while any of them sits below its floor
(:data:`MINIMUM_COMPARED_FRAMES`, :data:`MINIMUM_HASHED_FRAME_BYTES`,
:data:`MINIMUM_COMPARED_RECEIPT_DIGESTS`). A passing screen states how
much it compared to get there, so "identical" is a claim with a size
attached to it. The sibling comparator ``woof.certify.dualrun`` closed
the same hole the same way, and the floors here follow its rule: one,
not a larger number nobody could derive from the artifact.

Exit 0 PASS, 1 MISMATCH, 2 the screen could not be evaluated. The third
was previously unreachable -- "there was nothing to compare" left
through the same exit as "the bytes differ" and so presented an
unevaluated screen as a corruption finding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: The three outcomes, and the exit status each carries.
PASS = "PASS"
MISMATCH = "MISMATCH"
REFUSED = "REFUSED"

EXIT_STATUS = {PASS: 0, MISMATCH: 1, REFUSED: 2}

#: How much the screen must have compared before ``PASS`` means anything.
#:
#: One of each, not some larger number.  A threshold above "something" is
#: a figure nobody can derive from the artifact, and inventing one is how
#: a screen starts refusing legitimate work -- the failure this file's
#: other half already had to be rescued from.  The other half of the
#: closure is the count printed in the verdict line: a reader told the
#: screen cleared the floor with exactly ``1 final-state digest`` can see
#: it was thin without anyone having guessed a floor for them.
#:
#: ``MINIMUM_HASHED_FRAME_BYTES`` exists because two zero-byte files are
#: byte-identical.  Counting the FILES is not enough; a pair of arms that
#: each wrote an empty history file cleared a file-count floor while
#: hashing no forecast whatsoever.
MINIMUM_COMPARED_FRAMES = 1
MINIMUM_HASHED_FRAME_BYTES = 1
MINIMUM_COMPARED_RECEIPT_DIGESTS = 1


class ScreenRefusal(RuntimeError):
    """The screen cannot be set up at all, so there is no verdict.

    Distinct from a refusal recorded in :class:`Evidence`: that one is a
    comparison that ran and found itself standing on nothing, this one is
    a comparison that could not be configured.  Both leave through exit
    status 2 and neither is ever reported as a corruption finding.
    """


@dataclass(frozen=True)
class Evidence:
    """What the screen actually compared, and the verdict that follows.

    The whole point of this type is that :attr:`verdict` is a FUNCTION of
    the counters.  There is no ``ok`` flag to leave true because the loop
    that would have cleared it never ran over anything -- which is
    exactly how a screen that compared zero digests reported a
    bit-identical run.  To report ``PASS`` a caller has to have counted,
    and the only way to count is to have compared.

    ``mismatches`` outranks every refusal on purpose.  An observed byte
    difference is a positive finding; downgrading it to "could not
    evaluate" because some other leg of the screen was thin would throw
    away the one thing the screen exists to report.
    """

    frames_compared: int = 0
    frame_bytes_hashed: int = 0
    receipt_digests_compared: int = 0
    receipt_scalars_compared: int = 0
    mismatches: tuple[str, ...] = field(default=())
    refusals: tuple[str, ...] = field(default=())

    @property
    def insufficiencies(self) -> tuple[str, ...]:
        """Every floor this evidence failed to clear, each named."""

        missing: list[str] = []
        if self.frames_compared < MINIMUM_COMPARED_FRAMES:
            missing.append(
                f"compared {self.frames_compared} history files, needs at "
                f"least {MINIMUM_COMPARED_FRAMES}")
        if self.frame_bytes_hashed < MINIMUM_HASHED_FRAME_BYTES:
            missing.append(
                f"hashed {self.frame_bytes_hashed} history bytes, needs at "
                f"least {MINIMUM_HASHED_FRAME_BYTES} (two empty files are "
                f"byte-identical and prove nothing)")
        if self.receipt_digests_compared < MINIMUM_COMPARED_RECEIPT_DIGESTS:
            missing.append(
                f"compared {self.receipt_digests_compared} final-state "
                f"digests, needs at least "
                f"{MINIMUM_COMPARED_RECEIPT_DIGESTS} (digests cover the "
                f"arrays the history files do not carry)")
        return tuple(missing)

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        """Why this screen may not report ``PASS``.

        A structural refusal is reported alone: when the two arms are the
        same directory there is no useful second sentence about how few
        digests that directory agreed with itself on.
        """

        return self.refusals or self.insufficiencies

    @property
    def verdict(self) -> str:
        if self.mismatches:
            return MISMATCH
        return REFUSED if self.blocking_reasons else PASS

    def describe_scope(self) -> str:
        """The size to attach to an ``identical`` claim."""

        frames = _plural(self.frames_compared, "history file")
        digests = _plural(self.receipt_digests_compared, "final-state digest")
        scalars = _plural(self.receipt_scalars_compared,
                          "non-environmental scalar")
        return (f"{frames} ({self.frame_bytes_hashed} bytes), "
                f"{digests}, {scalars}")


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _digest(path: Path) -> tuple[str, int]:
    """The SHA-256 of a file, and how many bytes went into it.

    The byte count is returned rather than stat()ed so it is a property
    of what was actually READ: a file that shrank between the listing and
    the hash cannot report the larger figure.
    """

    h = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def _wrfouts(root: Path) -> dict[str, Path]:
    return {p.relative_to(root).as_posix(): p
            for p in sorted(root.rglob("wrfout_d*"))
            if p.is_file()}


def _receipts(root: Path) -> tuple[dict[str, dict], list[str]]:
    """Every parseable receipt, AND the names of the ones that were not.

    The unreadable ones used to be skipped silently, which inverted the
    screen's own logic: a truncated or corrupted receipt is one of the
    things a bad card produces, and this reader responded by removing it
    from the comparison and passing on what was left.
    """

    out: dict[str, dict] = {}
    unreadable: list[str] = []
    for p in sorted(root.rglob("*receipt*.json")):
        name = p.relative_to(root).as_posix()
        try:
            out[name] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            unreadable.append(f"{name} ({type(exc).__name__})")
    return out, unreadable


def _arm_setup_refusals(run_a: Path, run_b: Path) -> list[str]:
    """Ways the two arms are not two arms at all.

    Both are exactly the A/B law: an exactly-zero delta is evidence the
    experiment never ran, not evidence the card is healthy.  One
    directory handed in twice agrees with itself by construction, and an
    arm nested inside the other has ``rglob`` walking B's files as though
    they were A's, so both arms report the same union.
    """

    reasons: list[str] = []
    try:
        same = os.path.samefile(run_a, run_b)
    except OSError:
        same = False
    if same:
        reasons.append(
            f"both arms name the same directory ({run_a}); a run compared "
            f"with itself agrees with itself and screens nothing")
        return reasons

    a_res, b_res = run_a.resolve(), run_b.resolve()
    if a_res in b_res.parents or b_res in a_res.parents:
        reasons.append(
            f"one arm is nested inside the other (A={a_res}, B={b_res}); "
            f"each arm's file walk would collect the other's output")
    return reasons


def _environmental(repo: Path) -> frozenset[str]:
    sys.path.insert(0, str(repo))
    try:
        from woof.verify.cases.convective_boundary_layer import (
            ENVIRONMENTAL_FIELDS)
        return frozenset(ENVIRONMENTAL_FIELDS)
    except Exception as exc:                             # noqa: BLE001
        # Fail LOUD, not open: an empty exclusion set would make every
        # wall-clock difference look like corruption, and a guessed one
        # would hide a real difference behind a name nobody registered.
        raise ScreenRefusal(
            "could not import ENVIRONMENTAL_FIELDS from the tree; the "
            f"screen refuses to invent its own exclusion list ({exc})")


#: Run-local metadata, not science: absolute paths differ by run
#: directory, checkpoint bookkeeping names a run's own files, and the
#: background poller's sample COUNT differs by wall-clock jitter.
#: Content identity is carried by the digest keys, compared separately.
_RUN_LOCAL_LEAVES = frozenset({"path", "samples", "last_checkpoint"})

#: The block of a run receipt that holds memory telemetry.  Scoping the
#: rule below to this block is the point: an ``_observed`` suffix or a
#: ``peak_bytes`` leaf anywhere else in a receipt keeps being screened.
_MEMORY_BLOCK = "memory"

#: Leaf names inside a ``memory`` block that record an OBSERVATION of
#: the host's allocator rather than a property of the forecast.
#:
#: ``peak_bytes`` is a maximum over an asynchronously sampled series --
#: a background thread polls the allocator every 50 ms -- so its value
#: depends on where the poller's samples landed relative to transient
#: allocations, not only on which allocations happened.  Two runs whose
#: allocation trajectories are identical can and do report different
#: high-water marks; the tree already classifies the device-wide figure
#: as environmental for the neighbouring reason (co-tenant processes),
#: and the sampled peak of even the in-process pool carries the same
#: defect from the sampling itself.
#:
#: What is NOT here matters as much.  ``preflight_alloc_estimate_bytes``
#: is computed from the configuration, ``interval_seconds`` and
#: ``mechanism`` and ``scope`` are constants, and an instantaneous pool
#: reading is a property of the run -- all four stay screened, so the
#: memory block does not fall out of the comparison wholesale.
_OBSERVED_MEMORY_LEAF_SUFFIX = "_observed"
_OBSERVED_MEMORY_LEAVES = frozenset({"peak_bytes"})


def exclusion_reason(key: str, environmental: frozenset[str]) -> str | None:
    """Why this flattened receipt key is not compared, or ``None``.

    ``None`` means the key is on the determinism-comparison surface and
    a disagreement there is a corruption finding.  Every other return is
    a reason a reader can weigh, and the caller prints it beside the two
    values whenever an excluded key actually disagrees.
    """

    parts = key.split(".")
    leaf = parts[-1].split("[")[0]
    if leaf in environmental:
        return "registered ENVIRONMENTAL_FIELDS"
    if leaf in _RUN_LOCAL_LEAVES:
        return "run-local metadata"
    if "timing_seconds" in key:
        # Wall-clock timings differ between arms whenever the host is
        # under different load; they carry no state.
        return "wall-clock timing"
    if _MEMORY_BLOCK in parts[:-1] and (
            leaf.endswith(_OBSERVED_MEMORY_LEAF_SUFFIX)
            or leaf in _OBSERVED_MEMORY_LEAVES):
        return "sampled memory high-water"
    return None


def _flatten(doc, prefix="", into=None):
    into = {} if into is None else into
    if isinstance(doc, dict):
        for key, value in doc.items():
            _flatten(value, f"{prefix}.{key}" if prefix else str(key), into)
    elif isinstance(doc, list):
        for index, value in enumerate(doc):
            _flatten(value, f"{prefix}[{index}]", into)
    else:
        into[prefix] = doc
    return into


#: The verdict line, one per outcome.  ``PASS`` keeps its historical
#: wording so the greps and receipts written against it still read, and
#: gains the scope of the claim after it.
_VERDICT_TEXT = {
    PASS: "PASS (bit-identical)",
    MISMATCH: "MISMATCH -- CORRUPTION FINDING",
    REFUSED: "REFUSED -- NOT EVALUATED",
}


def _finish(evidence: Evidence, lines: list[str]) -> tuple[str, list[str]]:
    """Attach the reasons and the verdict, and report what was decided.

    Every caller leaves through here, so there is exactly one place in
    the file where a verdict line is produced and it is always the one
    :class:`Evidence` derived.
    """

    verdict = evidence.verdict
    if verdict == REFUSED:
        for reason in evidence.blocking_reasons:
            lines.append(f"  REFUSED: {reason}")
        lines.append("  a screen that compared nothing is not a screen; "
                     "this run is UNSCREENED, not clean")
    text = _VERDICT_TEXT[verdict]
    if verdict == PASS:
        text = f"{text} -- compared {evidence.describe_scope()}"
    lines.append(f"DUAL-RUN SCREEN: {text}")
    return verdict, lines


def screen(run_a: Path, run_b: Path, repo: Path) -> tuple[str, list[str]]:
    """Compare two run directories.  Returns the verdict and its report.

    The verdict is one of :data:`PASS`, :data:`MISMATCH`,
    :data:`REFUSED` -- never a bare boolean, because the two ways of not
    passing mean opposite things to whoever is reading the chain.
    """

    environmental = _environmental(repo)
    lines: list[str] = []
    refusals: list[str] = _arm_setup_refusals(run_a, run_b)
    if refusals:
        return _finish(Evidence(refusals=tuple(refusals)), lines)

    mismatches: list[str] = []
    a_files, b_files = _wrfouts(run_a), _wrfouts(run_b)
    lines.append(f"frames: A={len(a_files)} B={len(b_files)}")

    # An arm with no history at all did not run.  That is not the card
    # corrupting memory -- reporting it as a corruption finding, which is
    # what this path used to do, stops the chain with the wrong reason
    # and sends whoever reads it hunting a hardware fault.
    for arm, files in (("A", a_files), ("B", b_files)):
        if not files:
            refusals.append(f"arm {arm} has no wrfout files; there is "
                            f"nothing to compare and no verdict to give")
    if refusals:
        return _finish(Evidence(refusals=tuple(refusals)), lines)

    if set(a_files) != set(b_files):
        mismatches.append("wrfout file sets differ")
        lines.append(f"file sets differ: only in A "
                     f"{sorted(set(a_files) - set(b_files))}, only in B "
                     f"{sorted(set(b_files) - set(a_files))}")

    shared = sorted(set(a_files) & set(b_files))
    differing: list[str] = []
    frame_bytes = 0
    for name in shared:
        digest_a, size_a = _digest(a_files[name])
        digest_b, _ = _digest(b_files[name])
        frame_bytes += size_a
        if digest_a != digest_b:
            differing.append(name)
    lines.append(f"wrfout files compared: {len(shared)}; byte-identical: "
                 f"{len(shared) - len(differing)}; differing: "
                 f"{len(differing)}")
    lines.append(f"history bytes hashed: {frame_bytes}")
    if differing:
        mismatches.extend(differing)
        for name in differing[:10]:
            lines.append(f"  DIFFERS: {name}")

    a_receipts, a_unreadable = _receipts(run_a)
    b_receipts, b_unreadable = _receipts(run_b)
    for arm, unreadable in (("A", a_unreadable), ("B", b_unreadable)):
        for name in unreadable:
            refusals.append(f"receipt in arm {arm} could not be read: {name}")

    only_a = sorted(set(a_receipts) - set(b_receipts))
    only_b = sorted(set(b_receipts) - set(a_receipts))
    if only_a or only_b:
        # Unpaired receipts are not a corruption finding, they are an
        # incomplete pair: one arm did not write what the other did, so
        # whatever the paired remainder agreed on is not a statement
        # about this dual run.
        refusals.append(f"receipt sets differ, so they cannot be paired: "
                        f"only in A {only_a}, only in B {only_b}")

    shared_receipts = sorted(set(a_receipts) & set(b_receipts))

    digest_keys = 0
    digest_diff = 0
    scalar_keys = 0
    scalar_diff = 0
    excluded: list[str] = []
    reason_counts: dict[str, int] = {}
    for name in shared_receipts:
        flat_a = _flatten(a_receipts[name])
        flat_b = _flatten(b_receipts[name])
        for key in sorted(set(flat_a) & set(flat_b)):
            reason = exclusion_reason(key, environmental)
            if reason is not None:
                # Excluded, and SHOWN when it disagrees. A screen that
                # silently drops a differing key is indistinguishable
                # from one that never looked at it.
                if flat_a[key] != flat_b[key]:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    excluded.append(f"  EXCLUDED [{reason}]: {name}:{key} "
                                    f"{flat_a[key]!r} != {flat_b[key]!r}")
                continue
            if "digest" in key or "sha256" in key:
                digest_keys += 1
                if flat_a[key] != flat_b[key]:
                    digest_diff += 1
                    mismatches.append(f"{name}:{key}")
                    lines.append(f"  DIGEST DIFFERS: {name}:{key}")
            else:
                scalar_keys += 1
                if flat_a[key] != flat_b[key]:
                    scalar_diff += 1
                    mismatches.append(f"{name}:{key}")
                    if scalar_diff <= 10:
                        lines.append(f"  SCALAR DIFFERS: {name}:{key} "
                                     f"{flat_a[key]!r} != {flat_b[key]!r}")

    lines.append(f"final-state digests compared: {digest_keys}; differing: "
                 f"{digest_diff}")
    lines.append(f"non-environmental scalar blocks differing: {scalar_diff}")
    lines.append(f"non-environmental scalars compared: {scalar_keys}")
    lines.append(f"excluded as environmental, and differing: {len(excluded)}")
    if reason_counts:
        lines.append("excluded by reason: " + ", ".join(
            f"{reason} {count}"
            for reason, count in sorted(reason_counts.items())))
    lines.extend(excluded)

    return _finish(Evidence(frames_compared=len(shared),
                            frame_bytes_hashed=frame_bytes,
                            receipt_digests_compared=digest_keys,
                            receipt_scalars_compared=scalar_keys,
                            mismatches=tuple(mismatches),
                            refusals=tuple(refusals)), lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-a", type=Path, required=True)
    ap.add_argument("--run-b", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    for root in (args.run_a, args.run_b):
        if not root.is_dir():
            print(f"no such run directory: {root}", file=sys.stderr)
            return EXIT_STATUS[REFUSED]

    try:
        verdict, lines = screen(args.run_a, args.run_b, args.repo)
    except ScreenRefusal as exc:
        print(f"DUAL-RUN SCREEN: {_VERDICT_TEXT[REFUSED]} -- {exc}",
              file=sys.stderr)
        return EXIT_STATUS[REFUSED]

    text = "\n".join(lines) + "\n"
    print(text, end="")
    # Written on every outcome, refusals included.  A receipt file that
    # only appears when the screen passed is how a refusal becomes
    # invisible to everyone reading the campaign directory afterwards.
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
    return EXIT_STATUS[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
