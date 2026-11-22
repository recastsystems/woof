"""The projected HRRR horizontal operator: native entry versus NumPy.

The NumPy mirror in ``_ProjectedCpuPlan._apply_numpy`` is the pinned
authority for this route -- it is what every nested HRRR preparation has
run -- so the native entry that replaces it has to reproduce it BIT FOR
BIT, not to a tolerance.  Every comparison here is on the raw uint32
view, which is why a signed zero or a NaN sign bit cannot hide inside an
``assert_allclose``.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from woof import explain
from woof.ingest.cpu_backend import CpuPreprocessBackend
from woof.ingest.hrrr import (
    HrrrNativeSnapshot,
    PROJECTED_OPERATOR_NUMPY,
    PROJECTED_OPERATOR_RUST,
    _ProjectedCpuPlan,
)
import woof.ingest.hrrr as hrrr_module
from woof.ingest.preprocess_backend import ParallelCpuPreprocessBackend
from woof.static.lambert import LambertGrid

#: The HRRR CONUS window every projected plan is expressed against.
SOURCE_NY = 1059
SOURCE_NX = 1799

#: docs/internal/receipts/les/les.km3.toml, the committed 3-domain
#: nest this operator's cost was measured on.  A real projection, so the
#: donors and fractions under test are the ones production produces.
NEST_REF_LAT = 46.35
NEST_REF_LON = -120.0
NEST_TRUELAT1 = 36.35
NEST_TRUELAT2 = 56.35
NEST_STAND_LON = -120.0


def _backend(workers: int | None = None) -> ParallelCpuPreprocessBackend:
    try:
        native = CpuPreprocessBackend()
    except (FileNotFoundError, OSError) as error:
        pytest.skip(f"native CPU bridge is not built: {error}")
    if not native.indexed_donor_interp:
        pytest.skip(
            "the built CPU bridge predates gpuwm_indexed_interp_f32")
    return ParallelCpuPreprocessBackend(workers=workers)


def _plan(backend, *, nx: int = 40, ny: int = 32, dx: float = 750.0):
    """A plan on real HRRR projection geometry and a real nest projection."""

    grid = LambertGrid(
        NEST_REF_LAT, NEST_REF_LON, NEST_TRUELAT1, NEST_TRUELAT2,
        NEST_STAND_LON, dx, dx, nx + 1, ny + 1)
    snapshot = HrrrNativeSnapshot(
        valid_time=datetime(2026, 8, 1, 8), forecast_hour=0,
        i_start=0, j_start=0, ny=SOURCE_NY, nx=SOURCE_NX, fields={})
    latitude, longitude = grid.latlon_mass()
    return _ProjectedCpuPlan(snapshot, latitude, longitude, backend)


def _rebind(plan, backend, donor_y, donor_x, fraction_y, fraction_x):
    """Point a real plan at crafted geometry, both operators together."""

    plan.iy = np.ascontiguousarray(donor_y, dtype=np.int32)
    plan.ix = np.ascontiguousarray(donor_x, dtype=np.int32)
    plan.fy = np.ascontiguousarray(fraction_y, dtype=np.float32)
    plan.fx = np.ascontiguousarray(fraction_x, dtype=np.float32)
    plan.target_shape = tuple(map(int, plan.iy.shape))
    plan._native = backend.indexed_donor_plan(
        plan.source_shape, plan.iy, plan.ix, plan.fy, plan.fx)
    return plan


def _bits(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float32).reshape(-1).view(
        np.uint32)


def _mirror(plan, field, method):
    """The reference, with IEEE flag noise from the inf arms silenced.

    ``errstate`` changes no value: the mirror evaluates its polynomial
    over the WHOLE array before selecting a branch, so an adversarial
    stencil legitimately raises invalid-operation on lanes whose result
    is then discarded.  Silencing the report keeps the suite's output
    about the comparison.
    """

    with np.errstate(all="ignore"):
        return plan._apply_numpy(field, method=method)


def _differing(reference, actual) -> tuple[int, int]:
    """(elements differing in value, elements differing only as NaN)."""

    reference = np.ascontiguousarray(reference, dtype=np.float32)
    actual = np.ascontiguousarray(actual, dtype=np.float32)
    assert reference.shape == actual.shape
    mask = _bits(reference) != _bits(actual)
    nan_only = (mask & np.isnan(reference).reshape(-1)
                & np.isnan(actual).reshape(-1))
    return int(np.count_nonzero(mask & ~nan_only)), int(
        np.count_nonzero(nan_only))


def _adversarial(shape, seed: int, *, mixed_infinity: bool) -> np.ndarray:
    """Plausible values plus every class that reaches an awkward branch."""

    rng = np.random.default_rng(seed)
    field = (rng.standard_normal(shape) * 40.0 + 280.0).astype(np.float32)
    flat = field.reshape(-1)
    classes = [
        np.float32(0.0),        # zero -> the tiny sentinel
        np.float32(-0.0),       # negative zero, same branch
        np.float32(1.0e-20),    # the sentinel value itself
        np.float32(1.0e-40),    # subnormal operand: the predicate corner
        np.float32(1.0e-30),    # subnormal product of two normals
        np.float32(np.nan),     # masked / missing donor
        np.float32(3.0e38),     # a product that overflows to infinity
    ]
    if mixed_infinity:
        classes += [np.float32(np.inf), np.float32(-np.inf)]
    pick = rng.choice(flat.size, size=flat.size // 4, replace=False)
    for slot, value in zip(np.array_split(pick, len(classes)), classes):
        flat[slot] = value
    return field


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_a_built_bridge_selects_the_native_operator_without_being_asked():
    plan = _plan(_backend())
    assert plan.operator == PROJECTED_OPERATOR_RUST
    assert plan._native is not None


def test_a_backend_without_the_entry_keeps_the_mirror_and_says_so(
        monkeypatch, capsys):
    monkeypatch.setattr(
        hrrr_module, "_PROJECTED_FALLBACK_ANNOUNCED", False, raising=False)
    records: list[dict] = []
    explain.add_warning_observer(records.append)
    try:
        plan = _plan(object())
    finally:
        explain.remove_warning_observer(records.append)
    assert plan.operator == PROJECTED_OPERATOR_NUMPY
    assert plan._native is None
    assert len(records) == 1
    action = records[0]["action"]
    assert "NumPy mirror" in action
    assert "cargo build --release --locked --offline" in action
    assert "warning:" in capsys.readouterr().err


#: The CPU preprocessing bridge's checkout build as each shell must
#: receive it, written out rather than derived so a generator that loses
#: its shell rule cannot also rewrite what it is judged against.  Windows
#: PowerShell 5.1 rejects `&&` with a parser error.
CPU_BRIDGE_BUILD_FOR_SHELL = {
    False: "cd tools/grib1_bridge && cargo build --release --locked "
           "--offline && cd ../..",
    True: "cd tools\\grib1_bridge; cargo build --release --locked "
          "--offline; cd ..\\..",
}


@pytest.mark.parametrize("windows", (False, True))
def test_the_fallback_sentence_spells_the_rebuild_for_the_shell(
        monkeypatch, windows):
    """The rebuild in this warning is one a Windows user can paste.

    It printed `cd tools/grib1_bridge && cargo build ...` on every OS,
    which Windows PowerShell 5.1 cannot parse.
    """

    from woof import bridges

    monkeypatch.setattr(bridges, "WINDOWS_SHELL", windows)
    monkeypatch.setattr(
        hrrr_module, "_PROJECTED_FALLBACK_ANNOUNCED", False, raising=False)
    records: list[dict] = []
    explain.add_warning_observer(records.append)
    try:
        _plan(object())
    finally:
        explain.remove_warning_observer(records.append)
    assert len(records) == 1
    action = records[0]["action"]
    assert CPU_BRIDGE_BUILD_FOR_SHELL[windows] in action, action
    if windows:
        assert "&&" not in action, (
            f"Windows PowerShell 5.1 cannot parse '&&': {action}")


def test_the_fallback_sentence_is_said_once_per_process(monkeypatch):
    monkeypatch.setattr(
        hrrr_module, "_PROJECTED_FALLBACK_ANNOUNCED", False, raising=False)
    records: list[dict] = []
    explain.add_warning_observer(records.append)
    try:
        for _ in range(3):
            _plan(object())
    finally:
        explain.remove_warning_observer(records.append)
    assert len(records) == 1


def test_the_mirror_stays_reachable_and_is_what_the_gate_compares():
    plan = _plan(_backend())
    field = _adversarial(
        (2, SOURCE_NY, SOURCE_NX), 11, mixed_infinity=False)
    mirror = _mirror(plan, field, "parabolic")
    assert mirror.shape == (2, *plan.target_shape)


# ---------------------------------------------------------------------------
# Bit identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["parabolic", "bilinear", "nearest"])
@pytest.mark.parametrize("levels", [None, 3])
def test_the_native_entry_reproduces_the_mirror_bit_for_bit(method, levels):
    backend = _backend()
    plan = _plan(backend)
    shape = ((SOURCE_NY, SOURCE_NX) if levels is None
             else (levels, SOURCE_NY, SOURCE_NX))
    field = _adversarial(shape, 20260812, mixed_infinity=False)
    value, nan_only = _differing(
        _mirror(plan, field, method),
        plan.apply(field, method=method))
    assert (value, nan_only) == (0, 0)


def test_a_stencil_holding_both_infinities_agrees_on_nan_not_on_its_sign():
    # The ONE documented corner.  An invalid operation raises the x86
    # default NaN with its sign bit set; a NaN that meets a differently
    # signed NaN propagates whichever operand the compiler put first, and
    # IEEE-754 specifies neither the choice nor the sign.  Both sides say
    # "not a number" and every finite-value guard downstream refuses both
    # the same way -- so this asserts the agreement that exists rather
    # than pretending to one that cannot.
    backend = _backend()
    plan = _plan(backend)
    field = _adversarial(
        (2, SOURCE_NY, SOURCE_NX), 31, mixed_infinity=True)
    mirror = _mirror(plan, field, "parabolic")
    native = plan.apply(field, method="parabolic")
    value, nan_only = _differing(mirror, native)
    assert value == 0
    # A compiler may propagate identical NaN sign bits. Exercise the NaN
    # arm without requiring a disagreement that IEEE-754 does not promise.
    assert np.any(np.isnan(mirror)), "the arm did not reach a NaN"
    np.testing.assert_array_equal(np.isnan(mirror), np.isnan(native))
    differing = _bits(mirror) != _bits(native)
    assert np.all(np.isnan(mirror).reshape(-1)[differing])
    assert np.all(np.isnan(native).reshape(-1)[differing])


def test_the_missing_value_predicate_follows_numpy_not_cuda():
    # A subnormal operand whose product with its partner is a NORMAL
    # number.  NumPy flushes only the RESULT of the product (it compares
    # it against FLT_MIN) and so takes the four-point branch; CUDA
    # flushes the OPERAND and so takes the missing-value branch.  The
    # answers are different numbers, not different roundings.
    #
    # This is the test that goes red if the native entry is ever pointed
    # at the CUDA predicate that lives beside it in the same library.
    backend = _backend()
    plan = _plan(backend)
    _rebind(plan, backend, [[5]], [[5]], [[np.float32(0.25)]],
            [[np.float32(0.25)]])
    field = np.zeros((SOURCE_NY, SOURCE_NX), dtype=np.float32)
    field[3:9, 3:9] = np.float32(7.5)
    # The (jy, jx) = (5, 5) donor, i.e. `b` in the x sweep of row 5.
    field[5, 5] = np.float32(1.0e-40)
    field[5, 6] = np.float32(1.0e10)
    mirror = _mirror(plan, field, "parabolic")
    native = plan.apply(field, method="parabolic")
    assert _differing(mirror, native) == (0, 0)
    assert np.isfinite(mirror).all()
    assert float(mirror[0, 0]) != 0.0


def test_the_zero_sentinel_round_trip_survives_the_port():
    # An all-zero stencil becomes an all-`tiny` stencil, and the answer
    # has to come back out as an exact zero -- the branch the CUDA plan's
    # own regression test pins on the regular-grid side.
    backend = _backend()
    plan = _plan(backend)
    _rebind(plan, backend, [[40]], [[40]], [[np.float32(0.5)]],
            [[np.float32(0.5)]])
    field = np.full((SOURCE_NY, SOURCE_NX), 3.25, dtype=np.float32)
    field[38:44, 38:44] = np.float32(0.0)
    mirror = _mirror(plan, field, "parabolic")
    native = plan.apply(field, method="parabolic")
    assert _differing(mirror, native) == (0, 0)
    assert _bits(native)[0] == _bits(np.float32(0.0))[0]


@pytest.mark.parametrize("fraction", [0.0, 1.0, 0.5, np.nextafter(1.0, 0.0)])
def test_the_fraction_end_points_take_the_same_branch_on_both_sides(fraction):
    # x == 0 and x == 1 are their own branches in `oned`, and a fraction
    # one ULP below 1 must NOT take them.
    backend = _backend()
    plan = _plan(backend)
    _rebind(plan, backend, [[12, 12]], [[12, 13]],
            [[np.float32(fraction)] * 2], [[np.float32(fraction)] * 2])
    field = _adversarial((SOURCE_NY, SOURCE_NX), 5, mixed_infinity=False)
    assert _differing(
        _mirror(plan, field, "parabolic"),
        plan.apply(field, method="parabolic")) == (0, 0)


def test_a_donor_on_the_window_edge_clamps_the_same_way():
    # np.clip on the offsets, at both ends of both axes.
    backend = _backend()
    plan = _plan(backend)
    donor_y = [[0, 0, SOURCE_NY - 1, SOURCE_NY - 1]]
    donor_x = [[0, SOURCE_NX - 1, 0, SOURCE_NX - 1]]
    _rebind(plan, backend, donor_y, donor_x,
            [[np.float32(0.3)] * 4], [[np.float32(0.7)] * 4])
    field = _adversarial((SOURCE_NY, SOURCE_NX), 9, mixed_infinity=False)
    assert _differing(
        _mirror(plan, field, "parabolic"),
        plan.apply(field, method="parabolic")) == (0, 0)


# ---------------------------------------------------------------------------
# Determinism and refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workers", [1, 2, 7, 64])
def test_worker_count_cannot_move_a_bit(workers):
    reference_plan = _plan(_backend(workers=1))
    field = _adversarial(
        (3, SOURCE_NY, SOURCE_NX), 4242, mixed_infinity=False)
    serial = reference_plan.apply(field, method="parabolic")
    parallel = _plan(_backend(workers=workers)).apply(
        field, method="parabolic")
    assert _differing(serial, parallel) == (0, 0)


def test_a_field_that_does_not_match_the_window_is_refused():
    plan = _plan(_backend())
    with pytest.raises(ValueError, match="trailing dimensions"):
        plan.apply(np.zeros((8, 8), dtype=np.float32))


def test_an_unknown_method_is_refused_by_name():
    plan = _plan(_backend())
    with pytest.raises(ValueError, match="nearest"):
        plan.apply(
            np.zeros((SOURCE_NY, SOURCE_NX), dtype=np.float32),
            method="sixteen_pt")


def test_the_native_plan_refuses_nearest_because_it_owns_no_donor_for_it():
    backend = _backend()
    plan = _plan(backend)
    with pytest.raises(ValueError, match="exact gather"):
        plan._native.apply(
            np.zeros((SOURCE_NY, SOURCE_NX), dtype=np.float32),
            method="nearest")


def test_doctor_reports_whether_the_estate_has_the_fast_operator():
    from woof import doctor

    _backend()  # skip unless a current library is actually resolvable
    check = doctor._cpu_library_check()
    assert check.status == "verified"
    assert "indexed-donor horizontal entry present" in check.detail


def test_a_bilinear_donor_outside_the_halo_is_refused_not_wrapped():
    # NumPy would wrap a negative index into the far edge of the window
    # and quietly interpolate the wrong four cells.  The geometry
    # constructor already rejects a window without the halo; the native
    # entry refuses the same thing at its own boundary rather than
    # reproducing the wrap.
    backend = _backend()
    plan = _plan(backend)
    _rebind(plan, backend, [[-1]], [[4]], [[np.float32(0.5)]],
            [[np.float32(0.5)]])
    with pytest.raises(ValueError, match="invalid dimensions"):
        plan.apply(
            np.zeros((SOURCE_NY, SOURCE_NX), dtype=np.float32),
            method="bilinear")
