"""Pin the shared FP32 total-order metric, and stop it being re-derived.

Sixteen modules in this repository measured oracle parity in ULP, each with
its own copy of the same two-line bit trick, and thirteen of those copies had
the same sign error.  Two tests here: one that the shared implementation is
right at the cases the broken copies got wrong, and one that no module
quietly grows a seventeenth copy.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest

from woof.core.fp32_ulp import fp32_ulp_distance, monotone_fp32_key


_REPO_ROOT = Path(__file__).resolve().parents[1]

#: The only modules allowed to build a float32 total ordering themselves.
#: Anything else must import from ``woof.core.fp32_ulp``.
_TOTAL_ORDER_OWNERS = {
    "woof/core/fp32_ulp.py": "the shared implementation itself",
    "woof/verify/state_equiv.py": (
        "a separate uint32 formulation whose subtraction must keep the"
        " registered candidate-minus-reference sign, which an unsigned"
        " magnitude distance cannot express"
    ),
    "woof/ingest/horiz.py": (
        "the uint32 ordered key a CUDA atomicMin/atomicMax reduces over"
        " (08e32e923's parabolic undershoot floor) and its exact inverse;"
        " the shared int64 key can be neither an atomic operand nor decoded"
        " back to the float it ordered"
    ),
}

#: Third-party sources are not ours to police.  ``work/`` holds frozen
#: release-snapshot trees -- historical bytes of already-shipped versions,
#: which no edit here can retroactively fix and which are not importable
#: live source.  ``evidence/`` is the same class (added 2026-08-30): capsule
#: trees that freeze the probe scripts a campaign actually ran on the rented
#: nodes -- the P3 oracle and ship waves (13ca00491, 86d44160a) landed
#: harness/receipt scripts carrying their own ULP helpers, and rewriting a
#: capsule to import this repo's owner would falsify the record of what ran.
#: Their sign-error copies only ever OVER-report (see woof/core/fp32_ulp.py),
#: so the bit-exact conclusions those receipts state could not have been
#: bought by the bug.  New LIVE probes still must import the owner.
#: ``.worktrees/`` holds OTHER CHECKOUTS (the 2026-07 Codex era left ~130
#: of them) and their runtime extractions -- not this tree's estate: a copy
#: of the bit trick inside a different checkout is that checkout's defect,
#: caught when this gate runs there.  The concrete breakage: a stale
#: ``.rw-wps-runtime.partial-*`` extraction under it sits past the Windows
#: MAX_PATH ceiling, where rglob lists a ``.py`` file that ``open()`` then
#: cannot reach, and the scan died on FileNotFoundError before reading a
#: single hash (measured 2026-08-31 at 114c920b5).
#: Nested worktrees are the same class, machine-made: the agent
#: harness checks out a full worktree per subagent lane, so each one
#: carries the two OWNER files at a relative path the allow-list cannot
#: name -- with any worktree present this gate could never be green, and
#: 31 offender rows measured at 40b9878c0 were exactly those checkouts
#: (plus one lane venv's vendored pip).  A worktree's own defects are
#: caught when this gate runs inside it, which every lane's pytest does.
#: ``.venv/`` is the environment ``install.sh`` creates inside the
#: checkout: third-party packages (Pillow, SciPy, urllib3, CuPy) are not
#: this estate, and with them in scope the gate was red on every checkout
#: installed the documented way.
_VENDORED = ("tools/grib1_bridge/vendor", "work/", "evidence/",
             ".worktrees/", ".venv/")

#: A NESTED CHECKOUT is its own estate wherever it sits, not only under
#: the two prefixes above: stray worktrees land at arbitrary root-level
#: names (``wt-self-nesting/`` carried three copies of the owner files,
#: measured 2026-08-31 at d0b421c22), and a prefix list only ever names
#: a checkout after it has already broken the gate.  Any directory that
#: carries a ``.git`` entry is a checkout; every file below it belongs
#: to THAT tree's copy of this gate, which runs when its lane runs
#: pytest.
_NESTED_CHECKOUT_CACHE: dict = {}


def _inside_nested_checkout(rel_dir: str) -> bool:
    """True when rel_dir (posix, relative to the repo root, "" for the
    root itself) sits at or below a directory that is its own git
    checkout."""
    if not rel_dir or rel_dir == ".":
        return False
    cached = _NESTED_CHECKOUT_CACHE.get(rel_dir)
    if cached is not None:
        return cached
    parent = rel_dir.rsplit("/", 1)[0] if "/" in rel_dir else ""
    verdict = (_inside_nested_checkout(parent)
               or (_REPO_ROOT / rel_dir / ".git").exists())
    _NESTED_CHECKOUT_CACHE[rel_dir] = verdict
    return verdict

#: Trees that a packaging run fills with copies of the source tree.  Both are
#: gitignored (``.gitignore:4-5``) and neither is a second implementation:
#: ``build/lib/woof/core/fp32_ulp.py`` is byte-identical to the file this scan
#: already reads.  A file under one of these is skipped *only* when its bytes
#: match a file that was scanned, so dropping a new copy of the bit trick into
#: ``build/`` does not get it past the gate.
_BUILD_OUTPUT = ("build/", "dist/")


def test_signed_zero_is_one_point_on_the_line():
    """The exact case every broken copy got wrong.

    ``-0.0`` and ``+0.0`` are numerically the same value in two encodings.
    Subtracting the negative branch from ``+2**31`` instead of ``INT32_MIN``
    placed them 2**32 apart, which is what made a bitwise-identical kernel
    read as catastrophically diverged.  MYNN's ``gh`` really is ``-0.0`` on
    neutral-shear pairs, so this is a live encoding, not a corner case.
    """
    got = np.array([0.0, -0.0], dtype=np.float32)
    want = np.array([-0.0, 0.0], dtype=np.float32)
    assert fp32_ulp_distance(got, want).tolist() == [0, 0]


def test_the_smallest_step_across_zero_is_two_ulp():
    """+1.4e-45 and -1.4e-45 are two representable steps apart, not 2**32.

    The line runs -denormal, -0.0/+0.0, +denormal.  The broken form reported
    4294967294 here.
    """
    tiny = np.array([np.float32(1.4e-45)], dtype=np.float32)
    assert fp32_ulp_distance(tiny, -tiny).tolist() == [2]


@pytest.mark.parametrize(
    "value",
    [0.0, 1.0, -1.0, 1.0e-30, -1.0e-30, 3.4e38, -3.4e38],
)
def test_adjacent_floats_are_one_ulp_apart(value):
    """Monotonicity: nextafter is exactly one key away, on both sides."""
    here = np.array([value], dtype=np.float32)
    above = np.nextafter(here, np.float32(np.inf))
    below = np.nextafter(here, np.float32(-np.inf))
    assert fp32_ulp_distance(here, above).tolist() == [1]
    assert fp32_ulp_distance(here, below).tolist() == [1]


def test_keys_increase_with_the_value_they_encode():
    """A total order, not a magnitude: sorting by key sorts by value."""
    values = np.array(
        [-np.inf, -3.4e38, -1.0, -1.4e-45, -0.0, 0.0, 1.4e-45, 1.0, 3.4e38],
        dtype=np.float32,
    )
    keys = monotone_fp32_key(values)
    assert np.all(np.diff(keys) >= 0)
    # -0.0 and +0.0 are the one place the order is flat rather than strict.
    assert int(np.count_nonzero(np.diff(keys) == 0)) == 1


def test_a_nan_never_reads_as_agreement():
    """Bit patterns, not arithmetic: NaN == NaN is false but must not pass."""
    nan = np.array([np.nan], dtype=np.float32)
    finite = np.array([1.0], dtype=np.float32)
    assert fp32_ulp_distance(nan, finite).tolist()[0] > 0


def test_mismatched_shapes_are_an_error_not_a_broadcast():
    """A gate comparing different shapes is measuring the wrong thing."""
    with pytest.raises(ValueError):
        fp32_ulp_distance(
            np.zeros(3, dtype=np.float32), np.zeros(4, dtype=np.float32)
        )


#: The float32 sign bit, as a number.  Earlier revisions of this gate matched
#: the string ``"2147483648"`` against ``ast.unparse`` output, which sees only
#: the spellings that *render* the constant: ``0x80000000`` and ``-0x80000000``
#: both unparse to decimal and were caught, but ``1 << 31``, ``2 ** 31`` and a
#: module-level ``SIGN_BIT`` name never produce those digits at all.  The
#: operand is folded to an integer instead, so every spelling that computes the
#: constant is the same spelling to this gate.
_SIGN_BIT = 1 << 31

#: Refuse to fold anything wider than this.  Folding is applied to arbitrary
#: repository source, and ``1 << 10**9`` must not be evaluated to find out that
#: it is not the sign bit.
_FOLD_LIMIT = 1 << 128

_FOLDABLE_BINARY = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.BitOr: lambda left, right: left | right,
    ast.BitAnd: lambda left, right: left & right,
    ast.BitXor: lambda left, right: left ^ right,
}


def _folded_int(node: ast.AST, constants: dict[str, int]) -> int | None:
    """Evaluate a constant integer expression, or return ``None``.

    ``ast.literal_eval`` covers the literals and their unary minus, but not
    the two forms that actually matter here: the width casts every copy of
    this trick wraps the constant in (``np.int64(-0x80000000)``,
    ``np.uint32(0x80000000)``) and the shift that spells it without writing
    it (``1 << 31``).  Both fold to an integer; neither is a literal.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            return None
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.UnaryOp):
        operand = _folded_int(node.operand, constants)
        if operand is None:
            return None
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.Invert):
            return ~operand
        return None
    if isinstance(node, ast.Call):
        # A width cast is not a value change at these magnitudes, and the
        # constant is always wrapped in one: ``np.int64(-0x80000000)``.
        if len(node.args) == 1 and not node.keywords:
            return _folded_int(node.args[0], constants)
        return None
    if isinstance(node, ast.BinOp):
        left = _folded_int(node.left, constants)
        right = _folded_int(node.right, constants)
        if left is None or right is None:
            return None
        if isinstance(node.op, (ast.LShift, ast.RShift, ast.Pow)):
            if not 0 <= right <= 128:
                return None
            if isinstance(node.op, ast.LShift):
                value = left << right
            elif isinstance(node.op, ast.RShift):
                value = left >> right
            else:
                value = left**right
        else:
            fold = _FOLDABLE_BINARY.get(type(node.op))
            if fold is None:
                return None
            value = fold(left, right)
        return value if abs(value) < _FOLD_LIMIT else None
    return None


def _constant_bindings(
    statements, outer: dict[str, int] | None = None
) -> dict[str, int]:
    """``NAME = <constant expression>`` bindings among these statements.

    Naming the constant is the cheapest way to hide it from a gate that reads
    expressions, so the name is resolved rather than trusted -- at module
    level, where the sign bit is usually parked, and inside the function,
    where a one-line local would otherwise be a free pass.
    """
    constants: dict[str, int] = dict(outer or {})
    for node in statements:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = (node.target,), node.value
        else:
            continue
        folded = _folded_int(value, constants)
        if folded is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = folded
    return constants


def _is_sign_bit(node: ast.AST, constants: dict[str, int]) -> bool:
    """True if this operand *is* the float32 sign bit, in any spelling.

    Both signs count: ``INT32_MIN - bits`` is the correct reflection and
    ``+2**31 - bits`` is the sign error this module exists to prevent, and a
    gate that saw only one of them would let the other back in.
    """
    folded = _folded_int(node, constants)
    return folded is not None and abs(folded) == _SIGN_BIT


def _rederivations_in_source(source: str) -> list[int]:
    """Line numbers of functions that build their own float32 total ordering.

    Two idioms produce one, and a rule shaped like either one alone misses
    the other -- a ``numpy.where`` rule let a ``struct``-based copy through
    on the first run of this gate:

    * reflection -- ``INT32_MIN - bits`` for the negative half, i.e. a
      subtraction whose *left* side is the sign-bit constant;
    * complement -- ``~bits`` for the negative half paired with
      ``bits ^ SIGN_BIT`` for the non-negative half.

    Matching on the operators rather than on the constant alone is what
    keeps the FP32 transcendental shims out of the result: those mask with
    ``&`` to read a sign, which is not an ordering.  Neither idiom is
    recognised by how the constant is written, only by what it evaluates to.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    constants = _constant_bindings(tree.body)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        scope = _constant_bindings(ast.walk(node), constants)
        reflects = xors = inverts = False
        for inner in ast.walk(node):
            if isinstance(inner, ast.UnaryOp) and isinstance(
                inner.op, ast.Invert
            ):
                inverts = True
            if not isinstance(inner, ast.BinOp):
                continue
            if isinstance(inner.op, ast.Sub):
                # A reflection subtracts a VALUE from the sign bit.  With a
                # constant on the right it is constant arithmetic -- the
                # int32 limits ``2**31 - 1`` and ``-(1 << 31) - 1`` that
                # bound a count or a CUDA stream priority -- and four such
                # limits (e13abf285, 39a8130ce, 1d5d57d09, the seed-range
                # test) read as orderings until this clause.
                reflects = reflects or (
                    _is_sign_bit(inner.left, scope)
                    and _folded_int(inner.right, scope) is None)
            elif isinstance(inner.op, ast.BitXor):
                xors = xors or _is_sign_bit(inner.left, scope) or (
                    _is_sign_bit(inner.right, scope)
                )
        if reflects or (xors and inverts):
            found.append(node.lineno)
    return found


def _rederivations(path: Path) -> list[int]:
    """``_rederivations_in_source`` for a file on disk."""
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    return _rederivations_in_source(source)


def _estate_sources() -> tuple[dict[str, Path], dict[str, Path]]:
    """Every ``.py`` this checkout owns, split into source and build output.

    THE ESTATE, in one place.  Two gates below read the same file set --
    the total-order scan and the compile sweep -- and a scan whose scope
    is written twice drifts into two different claims about what "the
    whole tree" means.
    """
    source, generated = {}, {}
    for path in sorted(_REPO_ROOT.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if any(rel.startswith(prefix) for prefix in _VENDORED):
            continue
        if _inside_nested_checkout(
                rel.rsplit("/", 1)[0] if "/" in rel else ""):
            continue
        target = (
            generated
            if any(rel.startswith(prefix) for prefix in _BUILD_OUTPUT)
            else source
        )
        target[rel] = path
    return source, generated


def test_every_source_file_in_the_estate_compiles():
    """THE BREAKAGE THIS GATE PREVENTS: a Python file that cannot be
    parsed shipping in the tree, and every whole-tree scan reporting
    green because it silently skipped the file it could not read.

    Found 2026-09-01 on the release line:
    ``tools/rustwx/crates/rw-mpas/tests/goldens/make_mesh_goldens.py``
    and ``probe_rules.py`` both carried an unterminated module docstring
    -- a ``_mesh()`` helper had been pasted INTO the docstring instead of
    after it -- so neither script would run at all, and the goldens the
    rw-mpas Rust tests consume could not be regenerated.  They sat that
    way through a release.

    Nothing caught it because nothing had to.  ``_rederivations`` below
    answers ``SyntaxError`` with ``return []``, which reads as "this file
    has no re-derivations" when the truth is "this file was never read".
    The scan directly above walks the entire estate and would have
    touched both files; an unparseable file was simply exempt from it.
    So the sweep is widened here rather than given its own scope: one
    enumeration, and a file that cannot be parsed is now a RED instead of
    a silent pass in every gate that shares it.
    """
    source, generated = _estate_sources()
    offenders = {}
    for rel, path in (*source.items(), *generated.items()):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as error:
            offenders[rel] = f"line {error.lineno}: {error.msg}"
        except UnicodeDecodeError as error:
            offenders[rel] = f"undecodable as utf-8: {error}"
    assert offenders == {}, (
        "these files do not compile, so they cannot run and every"
        " whole-tree scan skipped them rather than reading them:"
        f" {offenders}"
    )


def test_nothing_re_derives_the_total_order():
    """One implementation, imported everywhere.

    This is the gate the bug needed and did not have: the sign error was
    never a hard mistake to make, it was a mistake made thirteen times
    because each gate wrote its own copy.
    """
    source, generated = _estate_sources()

    scanned = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.values()
    }
    offenders = {}
    for rel, path in (*source.items(), *generated.items()):
        if rel in _TOTAL_ORDER_OWNERS:
            continue
        if rel in generated and (
            hashlib.sha256(path.read_bytes()).hexdigest() in scanned
        ):
            # A copy of a file this scan already read, under a path the
            # allow-list cannot name.  Reading it again would report the
            # owners twice; skipping it hides nothing.
            continue
        lines = _rederivations(path)
        if lines:
            offenders[rel] = lines
    assert offenders == {}, (
        "these modules build their own float32 total ordering instead of"
        f" importing woof.core.fp32_ulp: {offenders}"
    )


def test_the_owners_still_own_it():
    """The allow-list must describe reality, or it is just a mute button."""
    for rel in _TOTAL_ORDER_OWNERS:
        path = _REPO_ROOT / rel
        assert path.exists(), f"{rel} is allow-listed but does not exist"
        assert _rederivations(path), (
            f"{rel} no longer derives a total ordering; drop it from the"
            " allow-list rather than leaving a stale exemption"
        )


#: The gate above only means something if it bites, so every spelling of the
#: sign bit that a seventeenth copy could plausibly use is pinned here, along
#: with the shapes that must stay quiet.  ``1 << 31`` and the named constant
#: are the two that a string match against ``ast.unparse`` output let through.
_FLAGGED_SPELLINGS = {
    "shift_reflect": """
def key(values):
    bits = values.view(np.int32).astype(np.int64)
    return np.where(bits < 0, np.int64(-(1 << 31)) - bits, bits)
""",
    "shift_complement": """
def key(values):
    bits = values.view(np.uint32)
    return np.where(bits >> 31 != 0, ~bits, bits ^ (1 << 31))
""",
    "power_reflect": """
def key(values):
    bits = values.view(np.int32).astype(np.int64)
    return np.where(bits < 0, -(2 ** 31) - bits, bits)
""",
    "hex_reflect": """
def key(values):
    bits = values.view(np.int32).astype(np.int64)
    return np.where(bits < 0, np.int64(-0x80000000) - bits, bits)
""",
    "hex_complement": """
def key(values):
    bits = values.view(np.uint32)
    return np.where(bits >> 31 != 0, ~bits, bits ^ np.uint32(0x80000000))
""",
    "decimal_reflect": """
def key(values):
    bits = values.view(np.int32).astype(np.int64)
    return np.where(bits < 0, -2147483648 - bits, bits)
""",
    "decimal_complement": """
def key(values):
    bits = values.view(np.uint32)
    return np.where(bits >> 31 != 0, ~bits, bits ^ 2147483648)
""",
    "wrong_sign_reflect": """
def key(values):
    bits = values.view(np.int32).astype(np.int64)
    return np.where(bits < 0, np.int64(0x80000000) - bits, bits)
""",
    "named_constant": """
SIGN_BIT = 1 << 31


def key(values):
    bits = values.view(np.uint32)
    return np.where(bits >= SIGN_BIT, ~bits, bits ^ SIGN_BIT)
""",
    "local_constant_reflect": """
def key(values):
    sign = 1 << 31
    bits = values.view(np.int32).astype(np.int64)
    return np.where(bits < 0, -sign - bits, bits)
""",
    "local_constant_complement": """
def key(values):
    mask = 0x80000000
    bits = values.view(np.uint32)
    return np.where(bits >= mask, ~bits, bits ^ mask)
""",
    "struct_scalar": """
def ulp(got, want):
    left = struct.unpack("<i", struct.pack("<f", got))[0]
    right = struct.unpack("<i", struct.pack("<f", want))[0]
    if left < 0:
        left = -0x80000000 - left
    if right < 0:
        right = -0x80000000 - right
    return abs(left - right)
""",
}

_QUIET_SHAPES = {
    # The FP32 transcendental shims read a sign with a mask.  Masking is not
    # an ordering, and flagging it would push them onto an allow-list that
    # would then hide a real copy.
    "sign_mask": """
def copysign(magnitude, source):
    word = magnitude.view(np.uint32)
    other = source.view(np.uint32)
    return ((word & 0x7FFFFFFF) | (other & 0x80000000)).view(np.float32)
""",
    # Pinned bit patterns are data.  The sign bit appearing among them is a
    # coincidence of the values, not a derivation.
    "module_data_tuple": """
PINNED_BITS = (1000593162, 2147483648, 0x80000000, 1060320052)
SIGN_BIT = 0x80000000


def pinned():
    return np.asarray(PINNED_BITS, dtype=np.uint32)
""",
    # The int32 limits are the sign bit minus a constant: a bound on a
    # count, a CUDA grid or a stream priority, not a reflection of a value.
    "int32_limits": """
def bound(nobs, members, gx):
    dtype = np.int32 if nobs < 2**31 - 1 else np.int64
    assert gx * members <= (1 << 31) - 1
    return dtype, -(1 << 31) - 1
""",
    # A 2**32 unsigned wrap subtracts a different constant on the other side
    # of the operator; neither makes it a reflection.
    "unsigned_wrap": """
def signed(bits):
    return np.where(bits >= 0x80000000, bits - 0x100000000, bits)
""",
    # The compliant module: it imports the shared implementation.
    "imports_the_owner": """
from woof.core.fp32_ulp import fp32_ulp_distance, monotone_fp32_key


def gate(got, want, budget):
    assert np.all(np.diff(np.sort(monotone_fp32_key(got))) >= 0)
    return int(np.max(fp32_ulp_distance(got, want))) <= budget
""",
}


@pytest.mark.parametrize("label", sorted(_FLAGGED_SPELLINGS))
def test_every_spelling_of_the_sign_bit_is_flagged(label):
    """Computing the constant must not be a way around the gate."""
    assert _rederivations_in_source(_FLAGGED_SPELLINGS[label]), (
        f"{label} builds a float32 total ordering and escaped the gate"
    )


@pytest.mark.parametrize("label", sorted(_QUIET_SHAPES))
def test_bit_twiddling_that_is_not_an_ordering_stays_quiet(label):
    """False positives would force allow-list entries that hide real copies."""
    assert _rederivations_in_source(_QUIET_SHAPES[label]) == [], (
        f"{label} is not a total ordering and must not be flagged"
    )
