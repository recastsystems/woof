"""Independent scalar-kernel word comparisons for batched weather operators."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]

MEMBER_COUNTS = (1, 4, 10, 20, 40)
SPATIAL_CORES = ((5, 3, 137), (7, 5, 11))
ADVECTION_FIELDS = (("scalar", "", "q", "flux_div_scalar"),
                    ("u", "x", "u", "flux_div_u"),
                    ("v", "y", "v", "flux_div_v"),
                    ("w", "z", "w", "flux_div_w"))
ADVECTION_FLAGS = ((False, False, False, False),
                   (False, False, True, False),
                   (True, False, True, False),
                   (False, True, True, True),
                   (True, True, False, False),
                   (True, True, True, True))


def _shape(core, stagger):
    nz, ny, nx = core
    return nz + int(stagger == "z"), ny + int(stagger == "y"), nx + int(stagger == "x")


def _words_equal(cp, observed, expected):
    assert cp.asnumpy(observed).view(np.uint32).tobytes() == np.asarray(
        expected).view(np.uint32).tobytes()


@pytest.mark.parametrize("members", MEMBER_COUNTS)
@pytest.mark.parametrize("stagger", ("", "x", "y", "z"))
@pytest.mark.parametrize("core", SPATIAL_CORES)
def test_diffusion_words_equal_independent_scalar_launches(members, stagger, core):
    import cupy as cp
    from woof.core.diffusion import _dz_spacings
    from woof.core.kernels import get_kernel
    from woof.ensemble.batch_operators import prepare_add_diff2

    shape = _shape(core, stagger)
    rng = np.random.default_rng(3941)
    host_f = rng.uniform(-9.0, 12.0, (members,) + shape).astype(np.float32)
    # Nonzero, distinct initial tendencies prove the additive operation too.
    host_tend = rng.uniform(-0.04, 0.07, host_f.shape).astype(np.float32)
    host_f += np.arange(members, dtype=np.float32)[:, None, None, None] * np.float32(0.25)
    host_f.reshape(members, -1)[:, :3] = np.array([-0.0, 0.0, 1.0e-38], np.float32)
    f, tend = cp.asarray(host_f), cp.asarray(host_tend)
    reference_f, reference_tend = cp.asarray(host_f), cp.asarray(host_tend)
    zf = np.concatenate(([0.0], np.cumsum(np.linspace(30.0, 180.0, core[0]))))
    rdzf_h, rdzc_h = (np.asarray(value, np.float32)
                      for value in _dz_spacings(zf, stagger))
    rdzf, rdzc = cp.asarray(rdzf_h), cp.asarray(rdzc_h)
    ref_rdzf, ref_rdzc = cp.asarray(rdzf_h), cp.asarray(rdzc_h)
    kh, kv, dx, dy = 17.25, 9.5, 750.0, 1100.0
    nlev, nys, nxs = shape
    nx = nxs - int(stagger == "x")
    ny = nys - int(stagger == "y")
    grid = ((nxs + 127) // 128, nys, nlev)
    scalars = (np.float32(kh), np.float32(kv),
               np.float32(1.0 / dx ** 2), np.float32(1.0 / dy ** 2))
    dimensions = (np.int32(nlev), np.int32(ny), np.int32(nys),
                  np.int32(nx), np.int32(nxs), np.int32(stagger == "z"))
    scalar = get_kernel("diffusion", "add_diff2")
    batch = prepare_add_diff2(f, tend, rdzf, rdzc,
                             kh=kh, kv=kv, dx=dx, dy=dy, stagger=stagger)
    # Reusing a bound operation must retain and update the same member arrays.
    for _ in range(2):
        batch()
        for member in range(members):
            scalar(grid, (128, 1, 1),
                   (reference_f[member], reference_tend[member]) + scalars
                   + (ref_rdzf, ref_rdzc) + dimensions)
    cp.cuda.get_current_stream().synchronize()
    _words_equal(cp, tend, cp.asnumpy(reference_tend))
    _words_equal(cp, f, host_f)
    _words_equal(cp, rdzf, rdzf_h)
    _words_equal(cp, rdzc, rdzc_h)
    if stagger == "z":
        _words_equal(cp, tend[:, 0], host_tend[:, 0])
        _words_equal(cp, tend[:, -1], host_tend[:, -1])


@pytest.mark.parametrize("members", MEMBER_COUNTS)
@pytest.mark.parametrize("stagger", ("", "x", "y", "z"))
@pytest.mark.parametrize("core", SPATIAL_CORES)
def test_rayleigh_words_equal_independent_scalar_launches(members, stagger, core):
    import cupy as cp
    from woof.core.kernels import get_kernel
    from woof.ensemble.batch_operators import prepare_rayleigh_damp

    shape = _shape(core, stagger)
    rng = np.random.default_rng(849)
    host = rng.uniform(-7.0, 10.0, (members,) + shape).astype(np.float32)
    host += np.arange(members, dtype=np.float32)[:, None, None, None] * np.float32(0.125)
    host.reshape(members, -1)[:, :3] = np.array([-0.0, 0.0, 1.0e-38], np.float32)
    field, reference = cp.asarray(host), cp.asarray(host)
    factors_h = np.linspace(1.0, 0.625, shape[0], dtype=np.float32)
    factors, ref_factors = cp.asarray(factors_h), cp.asarray(factors_h)
    nlev, nys, nxs = shape
    plane = nys * nxs
    grid = ((nlev * plane + 127) // 128, 1, 1)
    scalar = get_kernel("diffusion", "rayleigh_damp")
    batch = prepare_rayleigh_damp(field, factors)
    for _ in range(2):
        batch()
        for member in range(members):
            scalar(grid, (128, 1, 1),
                   (reference[member], ref_factors, np.int32(nlev), np.int32(plane)))
    cp.cuda.get_current_stream().synchronize()
    _words_equal(cp, field, cp.asnumpy(reference))
    _words_equal(cp, factors, factors_h)
    # The scalar kernel may flush subnormal input even for a factor of one.
    # Its full word comparison above remains the authority for those values.
    # Normal values and signed zeros must retain their original words below
    # the damping layer.
    below = host[:, 0]
    normal_or_zero = (np.abs(below) >= np.finfo(np.float32).tiny) | (below == 0.0)
    observed_below = cp.asnumpy(field[:, 0])
    assert observed_below.view(np.uint32)[normal_or_zero].tobytes() == below.view(
        np.uint32)[normal_or_zero].tobytes()


@pytest.mark.parametrize("members", MEMBER_COUNTS)
def test_one_launch_binds_complete_4d_backings_and_exact_byte_strides(monkeypatch, members):
    import cupy as cp
    from woof.ensemble import batch_operators as operators

    observed = []
    real_adapter = operators.prepare_batch_kernel_launch

    def capture(spec, count, grid, block, args, **kwargs):
        adapter = real_adapter(spec, count, grid, block, args, **kwargs)

        def launch():
            observed.append((spec, count, grid, block, args, kwargs))
            return adapter()

        return launch

    monkeypatch.setattr(operators, "prepare_batch_kernel_launch", capture)
    field = cp.arange(members * 3 * 5 * 139, dtype=cp.float32).reshape(members, 3, 5, 139)
    tend = cp.zeros_like(field)
    rdzf, rdzc = cp.ones(2, cp.float32), cp.ones(3, cp.float32)
    operators.prepare_add_diff2(field, tend, rdzf, rdzc,
                                kh=2.0, kv=1.0, dx=1000.0, dy=1000.0)()
    assert len(observed) == 1
    assert observed[0][4][0] is field and observed[0][4][1] is tend
    assert field.ndim == tend.ndim == 4
    assert observed[0][5]["pointer_strides"] == {
        "f": field.strides[0], "tend": tend.strides[0], "rdzf": 0, "rdzc": 0}
    observed.clear()
    operators.prepare_rayleigh_damp(field, rdzc)()
    assert len(observed) == 1
    assert observed[0][4][0] is field and observed[0][4][1] is rdzc
    assert observed[0][5]["pointer_strides"] == {"f": field.strides[0], "rdamp": 0}
    cp.cuda.get_current_stream().synchronize()


def test_member_tail_guards_and_one_level_diffusion_are_preserved():
    import cupy as cp
    from woof.core.kernels import get_kernel
    from woof.ensemble.batch_operators import prepare_add_diff2, prepare_rayleigh_damp

    members, shape = 4, (1, 3, 139)
    sentinel = np.float32(-811.5)
    guarded_f = cp.full((members + 2,) + shape, sentinel, cp.float32)
    guarded_tend = cp.full_like(guarded_f, sentinel)
    field, tend = guarded_f[1:-1], guarded_tend[1:-1]
    host = np.arange(members * np.prod(shape), dtype=np.float32).reshape((members,) + shape)
    field[...] = cp.asarray(host)
    tend.fill(np.float32(0.125))
    reference = cp.full_like(field, np.float32(0.125))
    # A one-level field never reads its dummy face spacing.
    rdzf = cp.asarray(np.array([1234.0], np.float32))
    rdzc = cp.asarray(np.array([0.5], np.float32))
    prepare_add_diff2(field, tend, rdzf, rdzc,
                      kh=2.0, kv=13.0, dx=900.0, dy=1200.0)()
    scalar = get_kernel("diffusion", "add_diff2")
    for member in range(members):
        scalar((2, 3, 1), (128, 1, 1),
               (field[member], reference[member], np.float32(2.0), np.float32(13.0),
                np.float32(1.0 / 900.0 ** 2), np.float32(1.0 / 1200.0 ** 2),
                rdzf, rdzc, np.int32(1), np.int32(3), np.int32(3),
                np.int32(139), np.int32(139), np.int32(0)))
    prepare_rayleigh_damp(field, cp.ones(1, cp.float32))()
    cp.cuda.get_current_stream().synchronize()
    _words_equal(cp, tend, cp.asnumpy(reference))
    _words_equal(cp, field, host)
    guards = np.full((1,) + shape, sentinel, np.float32)
    for backing in (guarded_f, guarded_tend):
        _words_equal(cp, backing[:1], guards)
        _words_equal(cp, backing[-1:], guards)


def test_binding_refuses_views_overlap_and_invalid_shared_profile_shapes():
    import cupy as cp
    from woof.ensemble.batch_operators import prepare_add_diff2, prepare_rayleigh_damp

    field = cp.ones((4, 3, 5, 11), cp.float32)
    tend = cp.zeros_like(field)
    rdzf, rdzc = cp.ones(2, cp.float32), cp.ones(3, cp.float32)
    kwargs = {"kh": 2.0, "kv": 1.0, "dx": 1000.0, "dy": 1000.0}
    with pytest.raises(ValueError, match="complete"):
        prepare_add_diff2(field[0], tend[0], rdzf, rdzc, **kwargs)
    with pytest.raises(ValueError, match="overlaps"):
        prepare_add_diff2(field, field, rdzf, rdzc, **kwargs)
    with pytest.raises(ValueError, match="rdzf must have shape"):
        prepare_add_diff2(field, tend, rdzc, rdzc, **kwargs)
    with pytest.raises(ValueError, match="rdamp must have shape"):
        prepare_rayleigh_damp(field, rdzf)
    with pytest.raises(ValueError, match="overlaps"):
        prepare_rayleigh_damp(field, field.reshape(-1)[:3])


def test_fixed_step_sanity_does_not_select_a_member_minimum():
    from woof.ensemble.batch_operators import fixed_step_seconds

    assert fixed_step_seconds(6.0) == 6.0
    assert fixed_step_seconds(np.float32(2.5)) == 2.5
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            fixed_step_seconds(value)
    for value in (True, [3.0, 6.0], np.array([3.0, 6.0])):
        with pytest.raises(TypeError, match="one real scalar"):
            fixed_step_seconds(value)


@pytest.mark.parametrize("members", MEMBER_COUNTS)
@pytest.mark.parametrize("field_kind,stagger,pointer_name,entry", ADVECTION_FIELDS)
@pytest.mark.parametrize("core", ((5, 9, 137), (7, 11, 19)))
@pytest.mark.parametrize("open_x,open_y,has_msf,specified", ADVECTION_FLAGS)
@pytest.mark.parametrize("vorder", (3, 5))
def test_advection_words_equal_independent_scalar_launches(
        members, field_kind, stagger, pointer_name, entry, core,
        open_x, open_y, has_msf, specified, vorder):
    import cupy as cp
    from woof.core.kernels import get_kernel
    from woof.ensemble import batch_operators as operators

    nz, ny, nx = core
    target_shape = _shape(core, stagger)
    rng = np.random.default_rng(14573)
    shapes = {"field": (members,) + target_shape,
              "ru": (members, nz, ny, nx + 1),
              "rv": (members, nz, ny + 1, nx),
              "rw": (members, nz + 1, ny, nx)}
    host = {name: rng.uniform(-7.0, 11.0, shape).astype(np.float32)
            for name, shape in shapes.items()}
    for value in host.values():
        value += np.arange(members, dtype=np.float32)[:, None, None, None] * np.float32(0.125)
    if not open_x:
        host["ru"][..., -1] = host["ru"][..., 0]
    if not open_y:
        host["rv"][:, :, -1] = host["rv"][:, :, 0]
    host["rw"][:, 0] = 0.0
    host["rw"][:, -1] = 0.0
    initial_tend = rng.uniform(-0.02, 0.04, shapes["field"]).astype(np.float32)
    fields = {name: cp.asarray(value) for name, value in host.items()}
    reference_fields = {name: cp.asarray(value) for name, value in host.items()}
    tend, reference_tend = cp.asarray(initial_tend), cp.asarray(initial_tend)
    profile_host = {"spacing": np.linspace(-15.0, -4.0, nz, dtype=np.float32),
                    "fnm": np.linspace(0.2, 0.75, nz, dtype=np.float32)}
    profile_host["fnp"] = np.float32(1.0) - profile_host["fnm"]
    profile_host["msf"] = rng.uniform(0.9, 1.4, target_shape[1:]).astype(np.float32)
    profiles = {name: cp.asarray(value) for name, value in profile_host.items()}
    reference_profiles = {name: cp.asarray(value) for name, value in profile_host.items()}
    dx, dy = 950.0, 1300.0
    batch = operators.prepare_flux_div(
        fields["field"], fields["ru"], fields["rv"], fields["rw"], tend,
        profiles["spacing"], profiles["fnm"], profiles["fnp"], profiles["msf"],
        dx=dx, dy=dy, stagger=stagger, open_x=open_x, open_y=open_y,
        has_msf=has_msf, spec=specified, vorder=vorder)
    nlev, nys, nxs = target_shape
    grid = ((nxs + 127) // 128, nys, nlev)
    tail = (reference_profiles["spacing"], reference_profiles["fnm"],
            reference_profiles["fnp"], reference_profiles["msf"],
            np.float32(1.0 / dx), np.float32(1.0 / dy),
            np.int32(nz), np.int32(ny), np.int32(nx), np.int32(open_x),
            np.int32(open_y), np.int32(has_msf), np.int32(specified),
            np.int32(vorder))
    # Compare the production scalar launcher, which specializes these dimensions.
    scalar = get_kernel("advection", entry, (nz, ny, nx))
    for _ in range(2):
        batch()
        for member in range(members):
            scalar(grid, (128, 1, 1),
                   tuple(reference_fields[name][member]
                         for name in ("field", "ru", "rv", "rw"))
                   + (reference_tend[member],) + tail)
    cp.cuda.get_current_stream().synchronize()
    _words_equal(cp, tend, cp.asnumpy(reference_tend))
    for name, value in fields.items():
        _words_equal(cp, value, host[name])
    for name, value in profiles.items():
        _words_equal(cp, value, profile_host[name])
    if stagger == "z":
        _words_equal(cp, tend[:, 0], initial_tend[:, 0])


@pytest.mark.parametrize("members", MEMBER_COUNTS)
@pytest.mark.parametrize("field_kind,stagger,pointer_name,entry", ADVECTION_FIELDS)
def test_advection_launch_has_all_member_backings_and_shared_zero_strides(
        monkeypatch, members, field_kind, stagger, pointer_name, entry):
    import cupy as cp
    from woof.ensemble import batch_operators as operators

    observed = []
    real_adapter = operators.prepare_batch_kernel_launch

    def capture(spec, count, grid, block, args, **kwargs):
        adapter = real_adapter(spec, count, grid, block, args, **kwargs)

        def launch():
            observed.append((spec, count, args, kwargs))
            return adapter()

        return launch

    monkeypatch.setattr(operators, "prepare_batch_kernel_launch", capture)
    nz, ny, nx = 5, 9, 137
    shape = _shape((nz, ny, nx), stagger)
    field = cp.ones((members,) + shape, cp.float32)
    ru = cp.ones((members, nz, ny, nx + 1), cp.float32)
    rv = cp.ones((members, nz, ny + 1, nx), cp.float32)
    rw = cp.zeros((members, nz + 1, ny, nx), cp.float32)
    tend = cp.zeros_like(field)
    spacing = cp.full(nz, -1.0, cp.float32)
    fnm = cp.full(nz, 0.5, cp.float32)
    fnp = cp.full(nz, 0.5, cp.float32)
    msf = cp.ones(shape[1:], cp.float32)
    operators.prepare_flux_div(
        field, ru, rv, rw, tend, spacing, fnm, fnp, msf,
        dx=1000.0, dy=1000.0, stagger=stagger)()
    assert len(observed) == 1
    spec, count, args, kwargs = observed[0]
    assert spec.entry == entry and count == members
    assert all(argument.ndim == 4 for argument in args[:5])
    assert all(actual is expected for actual, expected in zip(args[:5], (field, ru, rv, rw, tend)))
    spacing_name = "rdn" if stagger == "z" else "rdnw"
    assert kwargs["pointer_strides"] == {
        pointer_name: field.strides[0], "ru": ru.strides[0], "rv": rv.strides[0],
        "rw": rw.strides[0], "tend_out": tend.strides[0],
        spacing_name: 0, "fnm": 0, "fnp": 0, "msf": 0}
    cp.cuda.get_current_stream().synchronize()


def test_advection_binding_refuses_member_shape_alias_and_overlapping_open_stencils():
    import cupy as cp
    from woof.ensemble.batch_operators import prepare_flux_div_scalar

    members, nz, ny, nx = 4, 3, 5, 11
    field = cp.ones((members, nz, ny, nx), cp.float32)
    ru = cp.ones((members, nz, ny, nx + 1), cp.float32)
    rv = cp.ones((members, nz, ny + 1, nx), cp.float32)
    rw = cp.zeros((members, nz + 1, ny, nx), cp.float32)
    tend = cp.zeros_like(field)
    spacing, fnm, fnp = (cp.ones(nz, cp.float32) for _ in range(3))
    msf = cp.ones((ny, nx), cp.float32)
    args = (field, ru, rv, rw, tend, spacing, fnm, fnp, msf)
    kwargs = {"dx": 1000.0, "dy": 1000.0}
    with pytest.raises(ValueError, match="complete"):
        prepare_flux_div_scalar(field[0], ru, rv, rw, tend, spacing, fnm, fnp, msf, **kwargs)
    with pytest.raises(ValueError, match="ru must have member/C-grid shape"):
        prepare_flux_div_scalar(field, field, rv, rw, tend, spacing, fnm, fnp, msf, **kwargs)
    with pytest.raises(ValueError, match="overlaps"):
        prepare_flux_div_scalar(field, ru, rv, rw, field, spacing, fnm, fnp, msf, **kwargs)
    with pytest.raises(ValueError, match="open_y advection needs"):
        prepare_flux_div_scalar(*args, open_y=True, **kwargs)
