"""Layout/fusion candidates must match every scalar kernel output word."""
from __future__ import annotations

import re

import numpy as np
import pytest

from conftest import requires_gpu

MEMBERS = (1, 4, 10, 20, 40)


def _require_cuda_factories():
    # The batched flux sources read the dycore's own CuPy ElementwiseKernel
    # factories; without a loadable CuPy (fix/ds-cpu, 2.9.11) there are none.
    from woof.core.state import cp
    if cp is None:
        pytest.skip("the dycore's CuPy kernel factories need CuPy")


def _words(cp, actual, expected):
    assert cp.asnumpy(actual).view(np.uint32).tobytes() == cp.asnumpy(expected).view(np.uint32).tobytes()


def test_source_n1_is_exact_and_all_candidate_abis_audit():
    _require_cuda_factories()
    from woof.core.kernels import module_source
    from woof.ensemble.batch_kernel import _entry_parts, generate_batch_source
    from woof.ensemble.batch_operators import _FLUX_SPECS
    from woof.ensemble.batch_fluxes import _raw_source
    from woof.ensemble.batch_layout_trials import omega_trial_source, flux_trial_source
    source = module_source("advection")
    original_omega = _raw_source("omega", True)[0]
    for members in MEMBERS:
        for layout in ("outermost", "innermost"):
            omega, spec, _ = omega_trial_source(50, members, has_msf=True,
                                               layout=layout, cache_shared=True)
            _entry_parts(omega, spec)
            if members == 1:
                assert omega == original_omega
            elif layout == "outermost":
                generate_batch_source(omega, spec, members)
            for spec in _FLUX_SPECS.values():
                candidate = flux_trial_source(source, spec, members, layout=layout,
                                               zero_tendency=True)
                _entry_parts(candidate, spec)
                if members == 1:
                    assert candidate == source
                elif layout == "outermost":
                    generate_batch_source(candidate, spec, members)


def test_source_preserves_omega_float_intrinsics_and_fold_order():
    _require_cuda_factories()
    from woof.ensemble.batch_fluxes import _checked_operation
    from woof.ensemble.batch_layout_trials import _local_divergence_operation
    for has_msf in (False, True):
        operation = _checked_operation("omega", has_msf)
        candidate = _local_divergence_operation(operation, 50)
        original_math = [line for line in operation.splitlines() if "__f" in line]
        candidate_math = [line for line in candidate.splitlines() if "__f" in line]
        assert candidate_math == original_math
        assert "__trial_divv[k - 1]" in candidate
        assert "ww[(k + 1) * ncol + col] = divv;" not in candidate


def test_source_rejects_field_pointer_escape():
    from woof.core.kernels import module_source
    from woof.ensemble.batch_kernel import BatchKernelUnsupported
    from woof.ensemble.batch_layout_trials import flux_trial_source
    from woof.ensemble.batch_operators import FLUX_DIV_SCALAR_SPEC
    source = module_source("advection").replace("xface_cell_open(q,", "xface_cell_open(q + 1,", 1)
    with pytest.raises(BatchKernelUnsupported, match="direct member argument"):
        flux_trial_source(source, FLUX_DIV_SCALAR_SPEC, 4, layout="innermost")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("members", MEMBERS)
@pytest.mark.parametrize("layout", ("outermost", "innermost"))
@pytest.mark.parametrize("has_msf", (False, True))
@pytest.mark.parametrize("cache_shared", (False, True))
def test_omega_local_and_inner_words_match_scalar(members, layout, has_msf, cache_shared):
    import cupy as cp
    from woof.ensemble.batch_fluxes import prepare_omega_columns
    from woof.ensemble.batch_layout_trials import (
        prepare_omega_trial, pack_member_innermost, unpack_member_innermost,
    )
    nz, ny, nx = 7, 3, 137
    rng = np.random.default_rng(6501)
    ru = cp.asarray(rng.uniform(-5e4, 8e4, (members, nz, ny, nx + 1)).astype(np.float32))
    rv = cp.asarray(rng.uniform(-7e4, 9e4, (members, nz, ny + 1, nx)).astype(np.float32))
    dnw = cp.asarray(np.linspace(-0.3, -0.02, nz, dtype=np.float32))
    c1h = cp.asarray(np.linspace(1, 0.125, nz, dtype=np.float32))
    msft = cp.asarray(rng.uniform(0.7, 1.6, (ny, nx)).astype(np.float32))
    reference = cp.empty((members, nz + 1, ny, nx), cp.float32)
    trial_outer = cp.empty_like(reference)
    packed = tuple(pack_member_innermost(value) for value in (ru, rv, trial_outer)) if layout == "innermost" else (ru, rv, trial_outer)
    immutable = tuple(value.copy() for value in (ru, rv, dnw, c1h, msft))
    launch = prepare_omega_trial(*packed, dnw, c1h, dx=950, dy=1375, has_msf=has_msf,
                                 msft=msft, layout=layout, cache_shared=cache_shared)
    originals = tuple(prepare_omega_columns(
        ru[m:m + 1], rv[m:m + 1], reference[m:m + 1], dnw, c1h,
        dx=950, dy=1375, has_msf=has_msf, msft=msft) for m in range(members))
    for _ in range(2):
        launch()
        for original in originals:
            original()
        if layout == "innermost":
            unpack_member_innermost(packed[-1], out=trial_outer)
        _words(cp, trial_outer, reference)
    actual = cp.asnumpy(trial_outer).view(np.uint32)
    assert not actual[:, 0].any()
    assert not actual[:, -1].any()
    for actual, expected in zip((ru, rv, dnw, c1h, msft), immutable):
        _words(cp, actual, expected)
    if members == 1:
        assert launch.metadata["n1_original_factory"]


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("members", MEMBERS)
@pytest.mark.parametrize("layout", ("outermost", "innermost"))
@pytest.mark.parametrize("stagger", ("", "x", "y", "z"))
@pytest.mark.parametrize("boundary", ((False, False, False), (True, False, False), (True, True, True)))
def test_stencil_and_zero_fusion_words_match_scalar(members, layout, stagger, boundary):
    import cupy as cp
    from woof.ensemble.batch_operators import prepare_flux_div
    from woof.ensemble.batch_layout_trials import (
        prepare_flux_div_trial, pack_member_innermost, unpack_member_innermost,
    )
    nz, ny, nx = 5, 9, 11
    nlev, nys, nxs = nz + (stagger == "z"), ny + (stagger == "y"), nx + (stagger == "x")
    rng = np.random.default_rng(9011)
    shape = (members, nlev, nys, nxs)
    field = cp.asarray(rng.uniform(-20, 35, shape).astype(np.float32))
    ru = cp.asarray(rng.uniform(-60000, 80000, (members, nz, ny, nx + 1)).astype(np.float32))
    rv = cp.asarray(rng.uniform(-70000, 90000, (members, nz, ny + 1, nx)).astype(np.float32))
    rw = cp.asarray(rng.uniform(-3000, 4000, (members, nz + 1, ny, nx)).astype(np.float32))
    spacing = cp.asarray(np.linspace(-5, -1, nz, dtype=np.float32))
    fnm = cp.asarray(np.linspace(0.2, 0.8, nz, dtype=np.float32))
    fnp = cp.asarray(np.float32(1) - cp.asnumpy(fnm))
    msf = cp.asarray(rng.uniform(0.7, 1.5, (nys, nxs)).astype(np.float32))
    open_x, open_y, specified = boundary
    has_msf = open_x or stagger in ("x", "z")
    options = dict(dx=950, dy=1375, stagger=stagger, open_x=open_x, open_y=open_y,
                   spec=specified, has_msf=has_msf)
    inputs = (field, ru, rv, rw, spacing, fnm, fnp, msf)
    unchanged = tuple(value.copy() for value in inputs)
    for zero in (False, True):
        initial = cp.asarray(rng.uniform(-2, 3, shape).astype(np.float32))
        # Signed-zero and nonzero sentinels catch a skipped blanket-zero slot.
        initial[:, 0, 0, :2] = cp.asarray(np.array([-0.0, 27.25], np.float32))
        expected = initial.copy()
        actual = initial.copy()
        resident = tuple(pack_member_innermost(value) for value in (field, ru, rv, rw, actual)) if layout == "innermost" else (field, ru, rv, rw, actual)
        launch = prepare_flux_div_trial(*resident, spacing, fnm, fnp, msf,
                                        layout=layout, zero_tendency=zero, **options)
        scalar = tuple(prepare_flux_div(
            field[m:m + 1], ru[m:m + 1], rv[m:m + 1], rw[m:m + 1], expected[m:m + 1],
            spacing, fnm, fnp, msf, **options) for m in range(members))
        for _ in range(2):
            if zero:
                expected.fill(0)
            for original in scalar:
                original()
            launch()
            if layout == "innermost":
                unpack_member_innermost(resident[-1], out=actual)
            _words(cp, actual, expected)
        for value, saved in zip(inputs, unchanged):
            _words(cp, value, saved)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("members", MEMBERS)
def test_fused_one_level_w_zeros_every_word(members):
    import cupy as cp
    from woof.ensemble.batch_layout_trials import prepare_flux_div_trial
    nz, ny, nx = 1, 3, 137
    w = cp.ones((members, 2, ny, nx), cp.float32)
    ru = cp.ones((members, 1, ny, nx + 1), cp.float32)
    rv = cp.ones((members, 1, ny + 1, nx), cp.float32)
    rw = cp.ones_like(w)
    tend = cp.full_like(w, -19.5)
    one = cp.ones(1, cp.float32)
    msf = cp.ones((ny, nx), cp.float32)
    prepare_flux_div_trial(w, ru, rv, rw, tend, one, one, one, msf,
                           dx=1000, dy=1000, stagger="z", zero_tendency=True)()
    assert not cp.asnumpy(tend).view(np.uint32).any()
