"""``woof doctor``'s verdict on the eigensolver the matrix analysis needs.

This line exists because of a specific field failure, twice: the analysis died
inside cuSOLVER while every other piece of evidence said the GPU was healthy.
It was healthy -- cupy's core, its elementwise kernels and its reductions all
load independently of cuBLAS/cuSOLVER, so an install can multiply arrays all
day and still be unable to factor one.  A check that only proves "cupy
imports" reproduces the masquerade instead of catching it.

Every branch is forced here rather than left to whatever this box happens to
have, because on a working box only one of the four is reachable and the other
three are exactly the ones a reader in trouble will see.
"""

from __future__ import annotations

import pytest

from woof import doctor


def _check(monkeypatch, payload, *, box_major=12):
    """The check with both of its inputs forced.

    ``box_major`` is forced because the remedy's install line is derived
    from it: left to the host, these assertions would pass or fail on
    which card the test box happens to have.
    """
    monkeypatch.setattr(doctor, "_eigensolver_probe", lambda: payload)
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: box_major)
    return doctor._matrix_eigensolver_check()


def test_the_probe_runs_the_operation_rather_than_testing_for_a_name():
    """The probe must SOLVE, not import.  Nothing cheaper detects this fault."""
    source = doctor._EIGENSOLVER_PROBE
    assert "batched_eigh" in source
    assert "cupy.linalg.eigh" in source
    # And it must check the answer, not merely that the call returned.
    assert "residual" in source


def test_both_solvers_working_is_verified_and_prints_no_remedy(monkeypatch):
    check = _check(monkeypatch, {"jacobi": "ok", "library": "ok"})
    assert check.status == "verified"
    assert check.remedy is None and check.action is None
    assert not check.blocking


def test_no_cusolver_is_still_verified_and_says_the_default_does_not_need_it(
        monkeypatch):
    """THE branch this check was written for.

    A box with no cuSOLVER runs data assimilation perfectly well now, so the
    accurate verdict is "verified", with the absence reported as news and the
    install printed for the cases that still want it.  Calling it a failure
    would train the reader to ignore the line.
    """
    check = _check(monkeypatch, {
        "jacobi": "ok",
        "library": "RuntimeError: CUSOLVER_STATUS_NOT_INITIALIZED"})
    assert check.status == "verified"
    assert not check.blocking
    assert "cuSOLVER unavailable" in check.detail
    assert "the default analysis does not need" in check.detail
    assert check.remedy and "nvidia-cusolver" in check.remedy


def test_the_bundled_kernel_failing_is_reported_even_when_cusolver_works(
        monkeypatch):
    """The reverse fault must not hide behind a working fallback."""
    check = _check(monkeypatch, {
        "jacobi": "CompileException: nvrtc said no", "library": "ok"})
    assert check.status == "present"
    assert "cuSOLVER works" in check.detail
    assert "eigensolver='library'" in check.remedy
    assert not check.blocking


def test_neither_solver_working_names_both_failures_and_spares_forecasts(
        monkeypatch):
    check = _check(monkeypatch, {"jacobi": "boom", "library": "bang"})
    assert check.status == "missing"
    assert "boom" in check.detail and "bang" in check.detail
    # A forecast-only install must not read this as its problem.
    assert "forecasts are unaffected" in check.detail
    assert not check.blocking, (
        "an install that never assimilates radar must not exit nonzero")


def test_without_cupy_the_line_declines_to_judge(monkeypatch):
    check = _check(monkeypatch, {"cupy": "not installed"})
    assert check.status == "info"
    assert check.remedy is None
    assert "not judged" in check.detail


def test_a_probe_that_cannot_run_is_reported_as_a_gap_not_as_a_pass(
        monkeypatch):
    check = _check(monkeypatch, {"probe": "did not run: timeout"})
    assert check.status == "missing"
    assert "could not be run" in check.detail


def test_the_check_is_in_the_default_estate(monkeypatch):
    """A check behind a flag is a check nobody runs."""
    monkeypatch.setattr(doctor, "_eigensolver_probe",
                        lambda: {"jacobi": "ok", "library": "ok"})
    names = [c.name for c in doctor.collect_checks(sources=())]
    assert "matrix eigensolver" in names
    # In the GPU cluster, reading down the order the three fail in:
    # the wheel, then compiling with it, then factoring with it.
    assert names.index("cupy (GPU runtime)") < names.index(
        "CUDA kernel headers") < names.index("matrix eigensolver")


@pytest.mark.parametrize("payload", [
    {"jacobi": "ok", "library": "nope"},
    {"jacobi": "nope", "library": "ok"},
    {"jacobi": "nope", "library": "nope"},
    {"probe": "did not run"},
])
def test_every_branch_that_prints_a_remedy_prints_a_runnable_one(
        monkeypatch, payload):
    """The closing line of the report claims this of every remedy it prints."""
    check = _check(monkeypatch, payload)
    if not check.remedy:
        return
    for line in check.remedy.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert stripped.split()[0] in {
            "pip", "python", "woof", "set", "nvidia-smi"}, line


# ---------------------------------------------------------------------------
# The install line is the BOX's, not a constant (#76-class).
#
# `nvidia-cusolver-cu12` on a CUDA-13 box is the NVRTC shadow trap: the
# suffixed packages exist as deprecation tombstones, so the advice installs
# cleanly, reports success, and leaves the box exactly as broken.  NVIDIA
# has now deprecated the cu12 suffix too, so BOTH spellings are tombstones
# and the major rides in the version pin instead of the package name.
# ---------------------------------------------------------------------------

_TOMBSTONES = ("nvidia-cusolver-cu13", "nvidia-cublas-cu13",
               "nvidia-cusparse-cu13", "nvidia-cusolver-cu12",
               "nvidia-cublas-cu12", "nvidia-cusparse-cu12")


@pytest.mark.parametrize("box_major", (12, 13))
@pytest.mark.parametrize("payload", [
    {"jacobi": "ok", "library": "nope"},
    {"jacobi": "nope", "library": "nope"},
    {"probe": "did not run"},
])
def test_no_box_is_ever_told_to_install_a_tombstone_wheel(
        monkeypatch, payload, box_major):
    """THE regression this pair of tests exists for."""
    check = _check(monkeypatch, payload, box_major=box_major)
    assert check.remedy
    for tombstone in _TOMBSTONES:
        assert tombstone not in check.remedy, tombstone
        assert tombstone not in (check.action or "")
    # And it names the spelling that does work: unsuffixed names, with the
    # box's major in the version pin.
    assert 'pip install --no-deps "nvidia-cusolver=={}.*"'.format(
        box_major) in check.remedy


@pytest.mark.parametrize("payload", [
    {"jacobi": "ok", "library": "nope"},
    {"jacobi": "nope", "library": "nope"},
    {"probe": "did not run"},
])
def test_the_pin_is_the_boxs_major_and_never_the_other_one(
        monkeypatch, payload):
    """A CUDA-12 box must not be pinned to CUDA 13 libraries, or the reverse."""
    check = _check(monkeypatch, payload, box_major=12)
    assert check.remedy
    expected = ('"nvidia-cusolver==12.*" "nvidia-cublas==12.*" '
                '"nvidia-cusparse==12.*"')
    assert expected in check.remedy
    assert "==13.*" not in check.remedy


def test_an_unreadable_major_names_both_pins_and_defaults_to_neither(
        monkeypatch):
    """A silent default is how a CUDA-13 box ends up installing cu12."""
    check = _check(monkeypatch, {"jacobi": "nope", "library": "nope"},
                   box_major=None)
    assert check.remedy
    assert '"nvidia-cusolver==12.*"' in check.remedy
    assert '"nvidia-cusolver==13.*"' in check.remedy
    # The one command it leads with is the lookup, not an install.
    assert check.action.startswith("nvidia-smi")


def test_a_full_card_is_not_judged_as_a_missing_solver(monkeypatch):
    """Named breakage: the V12 box's card 0 was full and both solvers read
    as absent, with a cuSOLVER install line (2026-10-09)."""
    check = _check(monkeypatch, {
        "jacobi": "OutOfMemoryError: Out of memory allocating 2,560 bytes",
        "library": "OutOfMemoryError: Out of memory allocating 2,560 bytes"})
    assert check.status == "info"
    assert "not judged" in check.detail
    assert "cusolver" not in (check.remedy or "").lower()
