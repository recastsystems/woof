import math
from pathlib import Path
from types import SimpleNamespace
from _run_config_fake import run_config_fake

import numpy as np
import pytest


def _capture_kernels(monkeypatch, owner, module, *, cache=False, requests=None):
    calls, kernels = [], {}

    def get_kernel(requested_module, function, shape=()):
        assert requested_module == module
        if requests is not None:
            requests.append((function, shape))

        def launch(grid, block, args):
            calls.append((function, grid, block, args))

        return kernels.setdefault(function, launch) if cache else launch

    monkeypatch.setattr(owner, "get_kernel", get_kernel)
    return calls


class _LaunchState:
    def __init__(self, has_msf=True, *, column_thb=False, h_diabatic=None):
        self.p = np.empty((3, 2, 4), dtype=np.float32)
        self.thb = np.empty(3 if column_thb else (3, 2, 4), dtype=np.float32)
        self.has_msf = has_msf
        self.h_diabatic = h_diabatic
        self._values, self._scratch = {}, {}

    def __getattr__(self, name):
        return self._values.setdefault(name, object())

    def scratch(self, _shape, slot):
        return self._scratch.setdefault(slot, object())


def test_load_base_installs_copy_owned_dz_min_cache(monkeypatch):
    from woof.core import constants as c
    from woof.core import state as state_module
    from woof.core.state import DomainState

    phb = np.array([
        0.0, 100.0, 280.0, 540.0,
    ], dtype=np.float64)
    original_phb = phb.copy()
    expected_height = 0.5 * (phb[:-1] + phb[1:]) / c.G
    expected_dz_min = float(np.diff(expected_height, axis=0).min())

    state = DomainState.__new__(DomainState)
    # The hypsometric_opt lane extended load_base's copied coefficient set
    # with the hybrid c3h/c4h/c3f/c4f columns; the stub must carry them.
    coord_names = ("dnw", "rdnw", "dn", "rdn", "fnp", "fnm", "znu",
                   "znw", "c1h", "c2h", "c1f", "c2f",
                   "c3h", "c4h", "c3f", "c4f")
    coord = SimpleNamespace()
    for name in coord_names:
        size = 4 if name in ("znw", "c1f", "c2f", "c3f", "c4f") else 3
        values = np.linspace(1.0, 2.0, size, dtype=np.float64)
        setattr(coord, name, values)
        setattr(state, name, np.empty(size, dtype=np.float32))
    base = SimpleNamespace(
        thb=np.arange(3.0), pb=np.arange(3.0), alb=np.arange(3.0), phb=phb,
        p_top=5000.0, mub=90000.0, terrain_z=None)
    for name in ("thb", "pb", "alb"):
        setattr(state, name, np.empty(3, dtype=np.float32))
    state.phb = np.empty(4, dtype=np.float32)
    # The EOS-spelling lane added three float64-derived setup arrays that
    # load_base fills beside the profiles: the base-thickness correction
    # (one per LAYER, so nz = 3 against phb's nz + 1 = 4) and the two
    # full-level coefficient drops.  Same contract as the c3h/c4h note
    # above -- the stub carries what load_base writes, or it is testing a
    # load_base the model does not have.
    state.dphb_resid = np.empty(3, dtype=np.float32)
    state.dc3f = np.empty(3, dtype=np.float32)
    state.dc4f = np.empty(3, dtype=np.float32)
    state.mub2d = np.empty((1, 1), dtype=np.float32)
    state.ht = np.empty((1, 1), dtype=np.float32)
    state._phb_host = None
    state._dz_min = None
    state.cf1 = state.cf2 = state.cf3 = np.float32(0.0)
    state.cfn = state.cfn1 = np.float32(0.0)
    # Replace the array module itself: the no-CuPy import deliberately leaves
    # cp=None, and this test uses NumPy buffers rather than device arrays.
    monkeypatch.setattr(state_module, "cp", SimpleNamespace(asarray=np.asarray))

    state.load_base(coord, base)

    # A mutable BaseState must not invalidate only half of the cache pair.
    base.phb[...] += 10000.0

    np.testing.assert_array_equal(state.height_half(), expected_height)
    assert state.dz_min == expected_dz_min
    np.testing.assert_array_equal(state._phb_host, original_phb)


def test_cached_dz_min_preserves_height_half_preload_error():
    from woof.core.state import DomainState

    state = DomainState.__new__(DomainState)
    state._phb_host = None
    state._dz_min = None

    with pytest.raises(RuntimeError, match=r"height_half\(\) called before"):
        _ = state.dz_min


def test_prepare_acoustic_coefficients_launches_one_stage_kernel(monkeypatch):
    from woof.core import acoustic

    class FakeState:
        def __init__(self):
            for name in ("p", "alt", "mup", "rdn", "rdnw", "c1h", "c2h",
                         "c1f", "c2f", "mub2d"):
                setattr(self, name, object())
            self.requests = []

        def scratch(self, shape, slot):
            self.requests.append((tuple(shape), slot))
            return slot

    launches = []

    def fake_get_kernel(module, function):
        assert (module, function) == ("acoustic", "calc_coefs")

        def launch(grid, block, args):
            launches.append((grid, block, args))

        return launch

    monkeypatch.setattr(acoustic, "get_kernel", fake_get_kernel)
    state = FakeState()
    cfg = SimpleNamespace(nz=3, ny=2, nx=4, epssm=0.5, top_lid=False)

    coefficients = acoustic.prepare_acoustic_coefficients(state, cfg, 0.75)

    assert coefficients == ("acoustic_c2a", "acoustic_a",
                            "acoustic_alpha", "acoustic_gamma")
    assert len(launches) == 1
    assert [slot for _shape, slot in state.requests] == list(coefficients)


def test_acoustic_noop_and_level_guards_precede_device_work(monkeypatch):
    from woof.core import acoustic

    class NoWorkState:
        def scratch(self, *_args):
            raise AssertionError("scratch allocation occurred before guard")

    # One past the top WPHI_MAX_LEV tier.  This was 129 until the tier
    # ladder admitted deeper columns; the assertion is unchanged -- the
    # level guard still fires before a single scratch slot is requested --
    # only the depth that trips it moved.  That 129 is now ADMITTED is
    # asserted in tests/test_acoustic_nz_tiers.py.
    oversized = SimpleNamespace(
        nz=acoustic.MAX_ACOUSTIC_LEVELS + 1, ny=1, nx=1, epssm=0.5,
        top_lid=False)
    with pytest.raises(ValueError, match=r"nz=257 exceeds"):
        acoustic.prepare_acoustic_coefficients(NoWorkState(), oversized, 1.0)

    monkeypatch.setattr(
        acoustic, "prepare_acoustic_coefficients",
        lambda *_args: pytest.fail("zero-step run prepared coefficients"))
    acoustic.run_acoustic_only(NoWorkState(), oversized, 1.0, 0)
    acoustic.run_acoustic_only(NoWorkState(), oversized, 1.0, -2)


def test_dycore_prepares_acoustic_coefficients_outside_substep_loop():
    import inspect
    from woof.core import dycore

    source = inspect.getsource(dycore.step)
    stage = source.index("for istage, (nsub, dtau) in enumerate(stages):")
    prepare = source.index("prepare_acoustic_coefficients(", stage)
    bind = source.index("prepare_acoustic_substep_launch(", prepare)
    substeps = source.index("for i in range(nsub):", prepare)
    advance = source.index("launch_acoustic_substep(", substeps)

    assert stage < prepare < bind < substeps < advance


def test_prepared_acoustic_substep_reuses_launch_containers(monkeypatch):
    from woof.core import acoustic


    kernel_requests = []

    launches = _capture_kernels(monkeypatch, acoustic, "acoustic", cache=True, requests=kernel_requests)
    state = _LaunchState()
    cfg = SimpleNamespace(
        nz=3, ny=2, nx=4, dx=1000.0, dy=1200.0, epssm=0.1,
        smdiv=0.25, damp_opt=3, dampcoef=0.2, zdamp=5000.0,
        open_x=False, open_y=False, specified=False, nested=True,
        spec_zone=1, top_lid=False,
    )
    coefficients = tuple(object() for _ in range(4))

    launch_substep = acoustic.prepare_acoustic_substep_launch(
        state, cfg, 0.75, coefficients)
    launch_substep(first=True)
    launch_substep(first=False)
    launch_substep(first=False)

    expected = [
        "advance_uv", "advance_mu_th_msf", "advance_w_phi_msf",
        "advance_nested_phi_w",
    ]
    assert kernel_requests == [
        (name, () if name == "advance_uv" else (3, 2, 4))
        for name in expected
    ]
    assert [name for name, *_rest in launches] == expected * 3
    by_name = {
        name: [entry for entry in launches if entry[0] == name]
        for name in expected
    }
    for name, calls in by_name.items():
        assert len({id(grid) for _name, grid, _block, _args in calls}) == 1
        assert len({id(block) for _name, _grid, block, _args in calls}) == 1
        expected_arg_count = 2 if name == "advance_uv" else 1
        assert len({id(args) for _name, _grid, _block, args in calls}) \
            == expected_arg_count

    uv_args = [args for name, _grid, _block, args in launches
               if name == "advance_uv"]
    assert uv_args[0][-6] == np.float32(0.0)
    assert uv_args[1][-6] == np.float32(cfg.smdiv)
    assert uv_args[1] is uv_args[2]
    assert set(state._scratch) == {
        "acoustic_mu_pp_old", "acoustic_th_pp_old",
    }


@pytest.mark.parametrize("use_cq", [False, True])
@pytest.mark.parametrize("top_lid", [False, True])
@pytest.mark.parametrize("has_msf", [False, True])
@pytest.mark.parametrize("first", [False, True])
def test_prebound_acoustic_args_match_plain_path(
        monkeypatch, use_cq, top_lid, has_msf, first):
    """Every prebound launch is structurally identical to the plain path."""
    from woof.core import acoustic


    kernel_requests = []

    calls = _capture_kernels(monkeypatch, acoustic, "acoustic", requests=kernel_requests)
    state = _LaunchState(has_msf)
    cfg = SimpleNamespace(
        nz=3, ny=2, nx=4, dx=1000.0, dy=1200.0, epssm=0.1,
        smdiv=0.25, damp_opt=3, dampcoef=0.2, zdamp=5000.0,
        open_x=False, open_y=False, specified=False, nested=False,
        spec_zone=0, top_lid=top_lid,
    )
    base = tuple(object() for _ in range(4))
    cq = tuple(object() for _ in range(3)) + (True,)
    coefficients = base + cq if use_cq else base

    acoustic.acoustic_substep(
        state, cfg, 0.75, first=first, coefficients=coefficients)
    plain_calls = list(calls)
    mu_name = "advance_mu_th_msf" if has_msf else "advance_mu_th"
    w_name = "advance_w_phi_msf" if has_msf else "advance_w_phi"
    assert kernel_requests == [
        ("advance_uv", ()),
        (mu_name, (3, 2, 4) if has_msf else ()),
        (w_name, (3, 2, 4)),
    ]
    calls.clear()
    kernel_requests.clear()

    launch_substep = acoustic.prepare_acoustic_substep_launch(
        state, cfg, 0.75, coefficients)
    launch_substep(first=first)
    prebound_calls = list(calls)
    assert kernel_requests == [
        ("advance_uv", ()), (mu_name, (3, 2, 4)), (w_name, (3, 2, 4)),
    ]

    assert [call[0] for call in prebound_calls] == [
        "advance_uv",
        "advance_mu_th_msf" if has_msf else "advance_mu_th",
        "advance_w_phi_msf" if has_msf else "advance_w_phi",
    ]
    assert len(prebound_calls) == len(plain_calls)
    for prebound, plain in zip(prebound_calls, plain_calls, strict=True):
        pre_name, pre_grid, pre_block, pre_args = prebound
        name, grid, block, args = plain
        assert (pre_name, pre_grid, pre_block) == (name, grid, block)
        assert len(pre_args) == len(args)
        for position, (pre_arg, arg) in enumerate(
                zip(pre_args, args, strict=True)):
            if pre_arg is arg:
                continue
            assert type(pre_arg) is type(arg), position
            assert pre_arg == arg, position


def test_slow_tendency_hot_path_uses_raw_fused_kernels():
    import inspect
    from woof.core import dycore

    source = inspect.getsource(dycore._add_slow_tendencies)
    assert "_launch_slow_pgf(state, cfg, cq=cq)" in source
    assert "_launch_slow_buoyancy(state, cfg)" in source
    assert "_launch_slow_geopotential(state, cfg, ww" in source
    assert "cp.roll" not in source
    assert '"rk_dpn"' not in source


def test_add_rhs_ph_hadv_uses_supplied_face_masses_numerically(monkeypatch):
    from woof.core import dycore

    state = SimpleNamespace(
        p=np.zeros((2, 1, 3), dtype=np.float32),
        rph_t=np.zeros((3, 1, 3), dtype=np.float32),
        u=np.ones((2, 1, 4), dtype=np.float32),
        v=np.zeros((2, 2, 3), dtype=np.float32),
        php=np.zeros((3, 1, 3), dtype=np.float32),
        phb=np.zeros(3, dtype=np.float32),
            c1f=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            c2f=np.zeros(3, dtype=np.float32),
            cfn=np.float32(1.0),
            cfn1=np.float32(0.0),
        msft=np.ones((1, 3), dtype=np.float32),
        msfu=np.ones((1, 4), dtype=np.float32),
        msfv=np.ones((2, 3), dtype=np.float32),
        has_msf=False,
    )
    state.php[1, 0] = [0.0, 2.0, 5.0]
    mux = np.array([[1.0, 2.0, 4.0, 1.0]], dtype=np.float32)
    muy = np.ones((2, 3), dtype=np.float32)
    cfg = SimpleNamespace(dx=1.0, dy=1.0, h_sca_adv_order=2,
                          open_x=False, open_y=False, specified=False)

    def fake_get_kernel(module, function):
        assert (module, function) == ("dycore", "slow_geopotential_faces")

        def launch(_grid, _block, args):
            assert args[5] is mux and args[6] is muy
            rph_t, u, _v, php, phb, face_x = args[:6]
            c1f, c2f = args[7:9]
            phi = php[1, 0] + phb[1]
            ua = u[1, 0, :3] + u[0, 0, :3]
            dphi = phi - np.roll(phi, 1)
            flux = (c1f[1] * face_x[0, :3] + c2f[1]) * ua * dphi
            rph_t[1, 0] -= float(args[14]) * (np.roll(flux, -1) + flux)

        return launch

    monkeypatch.setattr(dycore, "get_kernel", fake_get_kernel)
    dycore.add_rhs_ph_hadv(state, cfg, mux, muy)

    # Hand calculation: dphi=(-5,2,3), flux=(-10,8,24), then -1/4(F+F+1).
    np.testing.assert_array_equal(
        state.rph_t[1, 0], np.array([0.5, -8.0, -3.5], np.float32))


def test_invalid_geopotential_order_applies_vertical_terms_before_raise(
        monkeypatch):
    from woof.core import dycore

    state = SimpleNamespace(
        p=np.zeros((2, 1, 1), dtype=np.float32),
        rph_t=np.zeros((3, 1, 1), dtype=np.float32),
        u=np.empty(0), v=np.empty(0), w=np.empty(0),
        ru_t=np.empty(0), rv_t=np.empty(0), rw_t=np.empty(0),
        rth_t=np.empty(0), msft=np.empty(0), msfu=np.empty(0),
        msfv=np.empty(0), has_msf=False,
        rotational=False,
        total_theta=lambda: np.zeros((2, 1, 1), dtype=np.float32),
    )
    cfg = run_config_fake(dx=1.0, dy=1.0, h_sca_adv_order=3,
                          open_x=False, open_y=False, specified=False)
    for name in ("launch_flux_div_scalar", "launch_flux_div_u",
                 "launch_flux_div_v", "launch_flux_div_w",
                 "_launch_slow_pgf", "_launch_slow_buoyancy"):
        monkeypatch.setattr(dycore, name, lambda *_args, **_kwargs: None)

    def vertical_only(target, _ww):
        target.rph_t[1] += np.float32(7.25)

    monkeypatch.setattr(dycore, "_launch_slow_geopotential_vertical",
                        vertical_only)
    with pytest.raises(ValueError, match=r"must be 2 or 5"):
        dycore._add_slow_tendencies(
            state, cfg, np.empty(0), np.empty(0), np.empty(0))
    assert state.rph_t[1, 0, 0] == np.float32(7.25)


def test_fused_dycore_kernel_pins_each_former_cupy_rounding_boundary():
    kernel = (Path(__file__).parents[1] / "woof" / "core" / "kernels" /
              "dycore.cu").read_text(encoding="utf-8")

    for instruction in ("add.rn.f32", "sub.rn.f32", "mul.rn.f32",
                        "div.rn.f32"):
        assert instruction in kernel
    assert "fmaf(" not in kernel
    assert "slow_pgf" in kernel
    assert "slow_buoyancy" in kernel
    assert "slow_geopotential" in kernel
    assert "small_step_init_uv" in kernel
    assert "small_step_init_column" in kernel
    assert "small_step_finish_uv" in kernel
    assert "small_step_finish_column" in kernel
    assert "rn_div(-rn_add(al1, al2), cht)" in kernel
    assert "rn_sub(0.0f, rn_add(al1, al2))" not in kernel
    supplied = kernel[kernel.index("void slow_geopotential_faces"):
                      kernel.index("// ---- RK small-step preparation")]
    assert "supplied_fcx_value" in supplied
    assert "supplied_fcy_value" in supplied
    assert "mup" not in supplied and "mub2d" not in supplied


def test_small_step_prep_and_finish_each_use_two_raw_launches(monkeypatch):
    from woof.core import dycore



    class FakeState:
        def __init__(self):
            self.p = np.empty((3, 2, 4), dtype=np.float32)
            self.thb = np.empty(3, dtype=np.float32)
            self.has_msf = True
            self.h_diabatic = None
            names = (
                "u_pp", "v_pp", "u0", "v0", "u", "v", "mup0", "mup",
                "mub2d", "c1h", "c2h", "msfu", "msfv", "w_pp",
                "th_pp", "ph_pp", "mu_pp", "al_pp", "p_pp",
                "p_pp_old", "w0", "w", "thp0", "thp", "php0", "php",
                "alt", "c1f", "c2f", "rdnw", "msft",
            )
            for name in names:
                if not hasattr(self, name):
                    setattr(self, name, object())

    launches = _capture_kernels(monkeypatch, dycore, "dycore")
    state = FakeState()
    cfg = SimpleNamespace(open_x=False, open_y=False, specified=False,
                          nested=False)
    dycore._init_small_steps(state, cfg)
    dycore._finish_small_steps(state, cfg)

    assert [name for name, *_ in launches] == [
        "small_step_init_uv", "small_step_init_column",
        "small_step_finish_uv", "small_step_finish_column",
    ]
    # The uv kernels' boundary_x/boundary_y flags sit between has_msf and
    # nz/ny/nx; a periodic config must send 0/0 -- the unconditional wrap
    # this pair carried before the flags existed.
    nz, ny, nx = state.p.shape
    for name, _grid, _block, args in launches:
        if name.endswith("_uv"):
            assert args[-5:] == (np.int32(0), np.int32(0), np.int32(nz),
                                 np.int32(ny), np.int32(nx))
    assert all(block == (256,) for _name, _grid, block, _args in launches)


def test_prepared_small_step_boundaries_reuse_launch_containers(monkeypatch):
    from woof.core import dycore



    launches = _capture_kernels(monkeypatch, dycore, "dycore", cache=True)
    state = _LaunchState(column_thb=True, h_diabatic=object())
    cfg = SimpleNamespace(open_x=False, open_y=False, specified=False,
                          nested=False)
    launch_init = dycore._prepare_small_step_init_launch(state, cfg)
    launch_finish_zero = dycore._prepare_small_step_finish_launch(
        state, cfg, 0.0)
    launch_finish_final = dycore._prepare_small_step_finish_launch(
        state, cfg, 60.0)

    for launch in (launch_init, launch_finish_zero, launch_finish_final):
        launch()
        launch()

    expected_pair = ["small_step_init_uv", "small_step_init_column"]
    expected_finish = ["small_step_finish_uv", "small_step_finish_column"]
    assert [name for name, *_rest in launches] == (
        expected_pair * 2 + expected_finish * 4)
    for offset in (0, 2, 4, 6, 8, 10):
        first, second = launches[offset:offset + 2]
        assert first[2] is second[2]
    for name in set(expected_pair + expected_finish):
        calls = [entry for entry in launches if entry[0] == name]
        grouped = {}
        for _function, grid, block, args in calls:
            grouped.setdefault(id(args), []).append((grid, block, args))
        assert all(len(group) == 2 for group in grouped.values())
        for group in grouped.values():
            assert group[0][0] is group[1][0]
            assert group[0][1] is group[1][1]
            assert group[0][2] is group[1][2]

    finish_columns = [args for name, _grid, _block, args in launches
                      if name == "small_step_finish_column"]
    assert finish_columns[0][15] is state.p
    assert finish_columns[0][-7] == np.float32(0.0)
    assert finish_columns[-1][15] is state.h_diabatic
    assert finish_columns[-1][-7] == np.float32(60.0)


def test_acoustic_bookkeeping_uses_batched_raw_launches(monkeypatch):
    import inspect
    from woof.core import dycore

    class FakeState:
        def __init__(self):
            self.p = np.empty((3, 2, 4), dtype=np.float32)
            self.mup = np.empty((2, 4), dtype=np.float32)
            self.has_msf = True
            for name in ("u_pp", "v_pp", "mu_pp", "c1h", "msfu", "msfv"):
                setattr(self, name, object())

    cfg = SimpleNamespace(
        emdiv=0.01, dx=1000.0, dy=1200.0, open_x=False, open_y=False,
        specified=True, spec_zone=1,
    )

    launches = _capture_kernels(monkeypatch, dycore, "acoustic")
    state = FakeState()
    mudf, mu_prev = object(), object()
    dycore.apply_emdiv_filter(state, cfg, mudf, mu_prev)
    dycore._update_emdiv_mudf(state, cfg, mudf, mu_prev, 0.75)

    class FakeArray:
        def __init__(self, size):
            self.size = size

    targets = tuple(FakeArray(n) for n in (24, 30, 32))
    sources = tuple(FakeArray(n) for n in (24, 30, 32))
    dycore._sumflux_launch("zero_sumflux", targets)
    dycore._sumflux_launch("accumulate_sumflux", targets, sources)
    dycore._sumflux_launch("finish_sumflux", targets, sources, 4)

    assert [name for name, *_ in launches] == [
        "apply_emdiv", "update_mudf", "zero_sumflux",
        "accumulate_sumflux", "finish_sumflux",
    ]
    assert "cp.roll" not in inspect.getsource(dycore.apply_emdiv_filter)


def test_prepared_acoustic_bookkeeping_reuses_launch_containers(monkeypatch):
    import inspect
    from woof.core import dycore

    class FakeArray:
        def __init__(self, size, shape=None):
            self.size = size
            self.shape = (size,) if shape is None else shape

    state = SimpleNamespace(
        p=FakeArray(24, (3, 2, 4)), mup=FakeArray(8, (2, 4)),
        u_pp=object(), v_pp=object(), mu_pp=object(), c1h=object(),
        msfu=object(), msfv=object(), has_msf=True,
    )
    cfg = SimpleNamespace(
        emdiv=0.01, dx=1000.0, dy=1200.0, open_x=False, open_y=False,
        specified=False, nested=True, spec_zone=1,
    )
    mudf, mu_prev = object(), object()
    targets = tuple(FakeArray(n) for n in (24, 30, 32))
    sources = tuple(FakeArray(n) for n in (24, 30, 32))

    launches = _capture_kernels(monkeypatch, dycore, "acoustic", cache=True)

    launch_emdiv = dycore._prepare_emdiv_filter_launch(state, cfg, mudf)
    launch_sumflux = dycore._prepare_sumflux_launch(
        "accumulate_sumflux", targets, sources)
    for launch in (launch_emdiv, launch_sumflux):
        launch()
        launch()

    assert [name for name, *_rest in launches] == [
        "apply_emdiv", "apply_emdiv",
        "accumulate_sumflux", "accumulate_sumflux",
    ]
    for name in ("apply_emdiv", "accumulate_sumflux"):
        calls = [entry for entry in launches if entry[0] == name]
        assert calls[0][1] is calls[1][1]
        assert calls[0][2] is calls[1][2]
        assert calls[0][3] is calls[1][3]

    source = inspect.getsource(dycore.step)
    substeps = source.index("for i in range(nsub):")
    finish = source.index("small_step_finishes[istage]()", substeps)
    hot_loop = source[substeps:finish]
    assert source.index("_prepare_emdiv_filter_launch", 0, substeps) >= 0
    assert "_prepare_emdiv_mudf_launch" not in source
    assert source.index("_prepare_sumflux_launch", 0, substeps) >= 0
    assert hot_loop.index("launch_emdiv_filter()") \
        < hot_loop.index("launch_acoustic_substep(") \
        < hot_loop.index("launch_sumflux_accumulation()")


def test_acoustic_fusions_pin_former_eager_rounding_boundaries():
    import inspect
    from woof.core import acoustic

    kernel = (Path(__file__).parents[1] / "woof" / "core" / "kernels" /
              "acoustic.cu").read_text(encoding="utf-8")

    for instruction in ("add.rn.f32", "sub.rn.f32", "mul.rn.f32",
                        "div.rn.f32"):
        assert instruction in kernel
    for symbol in ("advance_specified_phi_w", "apply_emdiv", "update_mudf",
                   "zero_sumflux", "accumulate_sumflux", "finish_sumflux"):
        assert f"void {symbol}" in kernel
    assert "diagnose_p_column" in kernel
    assert "void calc_p_pp" not in kernel
    assert 'get_kernel("acoustic", "calc_p_pp")' not in inspect.getsource(
        acoustic.acoustic_substep)


def test_health_report_has_one_host_readback_and_no_cupy_reductions():
    import inspect
    from woof import runtime
    from woof.core import dycore

    report_source = inspect.getsource(dycore.stability_report)
    assert report_source.count("cp.asnumpy") == 1
    assert "cp.abs" not in report_source
    assert ".max()" not in report_source

    # The monitored integration loop moved to woof/runtime.py (Phase 5,
    # Task 2 extraction); the frozen real74 profile delegates to it, so
    # the no-cupy-reduction pin now inspects the one shared loop.
    integration_source = inspect.getsource(
        runtime.integrate_prepared_case)
    monitored = integration_source[
        integration_source.index("report = stability_report"):]
    assert "cp.abs(state.w)" not in monitored
    assert "cp.argmax" not in monitored
    assert "cp.concatenate" not in monitored
    assert 'report["boundary_w_max"]' in monitored
    assert 'report["interior_w_max"]' in monitored


def test_health_report_rejects_empty_arrays_and_regions_before_work():
    from woof.core import dycore

    class NoWorkState:
        def __init__(self, u, w, thp):
            self.u, self.w, self.thp = u, w, thp

        def scratch(self, *_args):
            raise AssertionError("device work occurred before validation")

    valid = np.zeros((2, 5, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="zero-size array"):
        dycore.stability_report(
            NoWorkState(np.empty(0, np.float32), valid, valid))
    with pytest.raises(ValueError, match="must be positive"):
        dycore.stability_report(
            NoWorkState(valid, valid, valid), boundary_width=0)
    with pytest.raises(ValueError, match="empty w interior"):
        dycore.stability_report(
            NoWorkState(valid, valid, valid), boundary_width=3)


def test_health_report_decodes_full_64_bit_argmax(monkeypatch):
    from woof.core import dycore

    class FakeState:
        def __init__(self):
            self.u = np.zeros((1, 3, 3), dtype=np.float32)
            self.w = np.zeros((1, 3, 3), dtype=np.float32)
            self.thp = np.zeros((1, 3, 3), dtype=np.float32)
            self.buffers = {}

        def scratch(self, shape, slot):
            return self.buffers.setdefault(
                slot, np.zeros(shape, dtype=np.float32))

    wanted_index = (1 << 32) + 7

    def fake_get_kernel(module, function):
        assert module == "health"

        def launch(_grid, _block, args):
            if function == "health_final":
                result = args[1]
                result[:5] = [4.0, 7.0, 3.0, 7.0, 5.0]
                words = np.array(
                    [wanted_index & 0xffffffff, wanted_index >> 32],
                    dtype=np.uint32)
                result[5] = 0.0
                result[6:8] = words.view(np.float32)

        return launch

    monkeypatch.setattr(dycore, "get_kernel", fake_get_kernel)
    monkeypatch.setattr(dycore.cp, "asnumpy", lambda value: value)
    report = dycore.stability_report(FakeState(), boundary_width=1)

    assert report == {
        "u_max": 4.0, "w_max": 7.0, "th_max": 3.0,
        "cfl": None, "horizontal_cfl": None, "vertical_cfl": None,
        "nan": False,
        "boundary_w_max": 7.0, "interior_w_max": 5.0,
        "w_argmax": wanted_index,
    }


def test_co_located_cfl_passes_aloft_updraft_but_catches_surface_and_threshold(
        monkeypatch):
    """The safety gate pairs each upper-face w with its own cell's dz."""

    from woof.core import constants as c
    from woof.core import dycore

    class FakeState:
        def __init__(self, w, z):
            nz = len(z) - 1
            self.u = np.zeros((nz, 1, 2), dtype=np.float32)
            self.w = np.asarray(w, dtype=np.float32).reshape(nz + 1, 1, 1)
            self.thp = np.zeros((nz, 1, 1), dtype=np.float32)
            self.php = (
                np.asarray(z, dtype=np.float32) * np.float32(c.G)
            ).reshape(nz + 1, 1, 1)
            self.phb = np.zeros(nz + 1, dtype=np.float32)
            self.buffers = {}

        def scratch(self, shape, slot):
            return self.buffers.setdefault(
                slot, np.zeros(shape, dtype=np.float32))

    def fake_get_kernel(module, function):
        assert module == "health"

        def launch(_grid, _block, args):
            if function == "health_partial":
                (u, w, thp, ph, phb, partial, _nu, _nw, _nth, ncells,
                 phb_full, _width, ny, nx, gravity) = args
                plane = int(ny) * int(nx)
                count = int(ncells)
                rates = []
                for index in range(count):
                    level = index // plane
                    upper = index + plane
                    lower_base = phb[index] if int(phb_full) else phb[level]
                    upper_base = (
                        phb[upper] if int(phb_full) else phb[level + 1])
                    dz = ((ph.reshape(-1)[upper] + upper_base)
                          - (ph.reshape(-1)[index] + lower_base)) / gravity
                    rates.append(abs(w.reshape(-1)[upper]) / dz)
                partial.fill(0.0)
                partial[0, :6] = [
                    np.max(np.abs(u)), np.max(np.abs(w)),
                    np.max(np.abs(thp)), 0.0, 0.0, max(rates),
                ]
                partial[0, 6:8] = np.array(
                    [int(np.argmax(np.abs(w))), 0],
                    dtype=np.uint32).view(np.float32)
            else:
                partial, result, _nblocks = args
                result[:6] = partial[0, :6]
                result[6:8] = partial[0, 6:8]

        return launch

    monkeypatch.setattr(dycore, "get_kernel", fake_get_kernel)
    monkeypatch.setattr(dycore.cp, "asnumpy", lambda value: value)
    run = SimpleNamespace(dt=10.0, dx=1000.0)

    # 100 m/s belongs to the 1000 m upper layer.  The old global pairing
    # fabricated CFL=100 by combining it with the unrelated 10 m layer;
    # the co-located result is 1 and remains below the real gate.
    aloft = dycore.stability_report(
        FakeState([0.0, 0.0, 100.0], [0.0, 10.0, 1010.0]), run)
    assert aloft["vertical_cfl"] == pytest.approx(1.0)
    assert not dycore.stability_gate_failed(
        aloft, max_cfl=10.0, max_w_ms=150.0)

    # The same gate still catches an actual thin first-layer violation.
    surface = dycore.stability_report(
        FakeState([0.0, 11.0, 0.0], [0.0, 10.0, 1010.0]), run)
    assert surface["vertical_cfl"] == pytest.approx(11.0)
    assert dycore.stability_gate_failed(
        surface, max_cfl=10.0, max_w_ms=150.0)

    # Threshold continuity: equality passes; the next FP32 value fails.
    at_limit = dycore.stability_report(
        FakeState([0.0, 10.0, 0.0], [0.0, 10.0, 1010.0]), run)
    assert at_limit["cfl"] == pytest.approx(10.0)
    assert not dycore.stability_gate_failed(
        at_limit, max_cfl=10.0, max_w_ms=150.0)
    just_above = np.nextafter(
        np.float32(10.0), np.float32(np.inf), dtype=np.float32)
    above = dycore.stability_report(
        FakeState([0.0, just_above, 0.0], [0.0, 10.0, 1010.0]), run)
    assert above["cfl"] > 10.0
    assert dycore.stability_gate_failed(
        above, max_cfl=10.0, max_w_ms=150.0)


def test_bad_layer_geometry_reaches_the_cfl_the_safety_gate_reads():
    """Mask bit 32 has ONE channel, and ``max()`` used to throw it away.

    ``health.cu:86-91`` classifies a mass cell whose live layer thickness
    is non-positive or non-finite -- a collapsed or folded geopotential
    column, the classic precursor of an ARW vertical blow-up -- by setting
    mask bit 32, and ``health_final`` spends that bit as
    ``result[5] = nanf("")``.  That word is the vertical CFL rate, and this
    decoder is its only expression.  ``max(horizontal, nan)`` returns
    ``horizontal`` in CPython -- the comparison ``nan > x`` is False -- so
    the signal was dropped in the one line that carried it: ``u``, ``w``
    and ``thp`` are all still finite, ``nan`` is False, and
    ``stability_gate_failed``, which tests exactly ``cfl``'s finiteness,
    passed over a domain with an inverted layer.  ``result[5]`` is a
    WHOLE-DOMAIN reduction, so one such cell also replaced the real
    vertical maximum: the gate went blind precisely as the failure it
    exists to catch developed.
    """

    from woof.core import dycore

    run = SimpleNamespace(dt=10.0, dx=1000.0)
    # ``health_final``'s eight-word record: u_max, w_max, th_max, the
    # boundary and free-interior w maxima, the vertical rate, and two
    # index words.  Every field maximum finite and unremarkable; only the
    # vertical rate carries the mask, which is the whole point.
    host = np.array([5.0, 3.0, 1.0, 0.0, 0.0, np.nan, 0.0, 0.0],
                    dtype=np.float32)
    report = dycore.decode_stability_record(host, run)

    assert report["nan"] is False, "no field maximum can see bad geometry"
    assert not math.isfinite(report["cfl"])
    assert dycore.stability_gate_failed(report, max_cfl=10.0, max_w_ms=150.0)

    # A healthy record still reports the larger of the two terms, so the
    # guard is a NaN guard and not a blanket refusal.
    host[5] = 0.4
    healthy = dycore.decode_stability_record(host, run)
    assert healthy["horizontal_cfl"] == pytest.approx(0.05)
    assert healthy["vertical_cfl"] == pytest.approx(4.0)
    assert healthy["cfl"] == pytest.approx(4.0)
    assert not dycore.stability_gate_failed(
        healthy, max_cfl=10.0, max_w_ms=150.0)


def test_the_streamed_decoder_keeps_the_nan_the_resident_one_keeps():
    """The two paths are deliberately identical, so they failed together.

    ``tilestream.health_fold.TileHealthFold._report`` reproduces
    ``decode_stability_record``'s arithmetic rather than calling it, and
    its own docstring says the duplication exists so the two paths cannot
    drift.  Both therefore carried this defect, and under ``[tiles]`` with
    a host store the fold is the ONLY armed observer -- ``require_healthy``
    is skipped entirely there -- so the streamed decoder is the one that
    most needs the signal.  Driven rather than read: ``_report`` needs
    nothing off the instance but ``cfg``, ``width`` and the swdown flag,
    so it runs on a stand-in with no device.
    """

    from woof.core import dycore
    from tilestream.health_fold import TileHealthFold

    run = SimpleNamespace(dt=10.0, dx=1000.0)
    fold = SimpleNamespace(cfg=run, width=0, _have_swdown=False)
    host = np.array([5.0, 3.0, 1.0, 0.0, 0.0, np.nan, 0.0, 0.0],
                    dtype=np.float32)

    streamed = TileHealthFold._report(fold, host)
    assert streamed["nan"] is False
    assert not math.isfinite(streamed["cfl"])
    assert dycore.stability_gate_failed(
        streamed, max_cfl=10.0, max_w_ms=150.0)

    host[5] = 0.4
    healthy = TileHealthFold._report(fold, host)
    assert healthy["cfl"] == pytest.approx(4.0)

    # And the two decoders still agree word for word, which is the claim
    # the duplication rests on.
    assert healthy == dycore.decode_stability_record(host, run)


def test_the_run_loop_refuses_a_non_finite_cfl_and_not_only_a_non_finite_field():
    """The signal now has a channel AND an observer on the run route.

    ``integrate_prepared_case`` used to read nothing off the stability
    record but ``report["nan"]``, which is decided by ``u_max``,
    ``w_max`` and ``th_max`` alone -- geopotential is not among them, so
    a collapsed or folded layer passed every substep.  Pinned by
    inspection for the same reason the no-cupy-reduction pin above is:
    this loop needs a card and a prepared case to run, and the property
    being asserted is which fields of the report it consults.
    """

    import inspect

    from woof import runtime

    source = inspect.getsource(runtime.integrate_prepared_case)
    monitored = source[source.index("report = stability_report"):]
    assert 'report["cfl"]' in monitored
    refusal = monitored[monitored.index('report["cfl"]'):]
    assert "math.isfinite" in refusal
    assert "raise RuntimeError" in refusal


def test_health_kernel_keeps_first_index_tie_and_nan_bits():
    kernel = (Path(__file__).parents[1] / "woof" / "core" / "kernels" /
              "health.cu").read_text(encoding="utf-8")

    assert "index < current_index" in kernel
    assert "mask |= 1u" in kernel
    assert "mask |= 2u" in kernel
    assert "mask |= 4u" in kernel
    assert "health_partial" in kernel and "health_final" in kernel
    assert "unsigned long long& current_index" in kernel
    assert "indices[0] >> 32" in kernel
    assert "fabsf(w[upper])" in kernel
    assert "speed / dz" in kernel
