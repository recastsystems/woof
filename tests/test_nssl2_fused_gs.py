"""Contract gates for the independent fused NSSL GS entry point."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.core import nssl2_fused_gs as fused


def _workspace(shape=(4, 2, 3)):
    state = np.zeros((16, *shape), dtype=np.float32)
    export = np.zeros((5, *shape[1:]), dtype=np.float32)
    return fused.NSSL2DriverWorkspace(state, export, shape)


def _environment(shape=(4, 2, 3)):
    cells = [np.ones(shape, dtype=np.float32) for _ in range(7)]
    w = np.ones((shape[0] + 1, *shape[1:]), dtype=np.float32)
    # theta, rho, pressure, exner, interface w, t0 scratch, t7 scratch, dz
    return (*cells[:4], w, *cells[4:])


@pytest.mark.parametrize("strict", [False, True])
def test_launcher_passes_one_workspace_and_exact_environment(monkeypatch, strict):
    import woof.wrf_exact as exact
    monkeypatch.setattr(exact, "ENABLED", strict)
    workspace = _workspace()
    environment = _environment()
    calls = []

    class Kernel:
        def __call__(self, grid, block, args):
            calls.append((grid, block, args))

    def get_kernel(module, symbol, shapes=(0.,1.)):
        assert module == "nssl2_fused_gs"
        assert symbol in ("nssl2_prepare_fused_gs", "nssl2_fused_gs")
        return Kernel()

    monkeypatch.setattr(fused, "shape_kernel", get_kernel)
    # The once-per-device init-table build needs a card; this gate is the
    # launch contract, so it is stubbed here.
    monkeypatch.setattr(fused, "_ensure_init_tables", lambda shapes=(0.,1.): None)
    fused.launch_fused_gs(workspace, *environment, 12.5)

    assert len(calls) == 2
    grid, block, args = calls[0]
    assert grid == (1,)
    assert block == (128,)
    theta, rho, pressure, exner, w, temperature, target, dz = environment
    # Without an explicit loaded vapour the prepass reads the workspace qv.
    expected_prepass = (
        temperature, target, workspace.field("qv") if strict else workspace.state, theta, rho, pressure,
        exner,
    )
    assert all(
        actual is expected or (
            getattr(actual, "base", None) is not None
            and np.shares_memory(actual, expected)
            and actual.shape == expected.shape)
        for actual, expected in zip(args[:7], expected_prepass)
    )
    assert args[7] == np.int32(24)

    # The production adapter hands the loaded (Registry) vapour instead:
    # WRF's t7 is diagnosed before the first-step calcnfromq moves
    # sub-threshold condensate into vapour.
    calls.clear()
    loaded = np.full_like(theta, 1.0e-3)
    fused.launch_fused_gs(workspace, *environment, 12.5, loaded_vapor=loaded)
    assert calls[0][2][2] is (loaded if strict else workspace.state)

    grid, block, args = calls[1]
    assert grid == (1,)
    assert block == (128,)
    expected_fused = (
        workspace.state, theta, rho, pressure, exner,
        temperature, w, target, dz,
    )
    assert all(
        actual is expected
        for actual, expected in zip(args[:9], expected_fused)
    )
    assert args[9] == np.float32(12.5)
    # nz, ncol, n, then the hail switch: the default lane runs with
    # WRF's hail category present (nssl_hail_on resolving to 1).
    assert args[10:] == (
        np.int32(4), np.int32(6), np.int32(24), np.int32(1))


def test_callback_adapter_preserves_the_narrow_hook(monkeypatch):
    workspace = _workspace()
    environment = _environment()
    calls = []
    monkeypatch.setattr(
        fused,
        "launch_fused_gs",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    theta, rho, pressure, exner, w, temperature, target, dz = environment
    loaded = np.zeros_like(theta)
    fields = SimpleNamespace(
        theta=theta, rho=rho, pressure=pressure, pii=exner, w=w, dz=dz,
        qv=loaded,
    )
    callback = fused.NSSL2FusedGS(temperature, target, 30.0)
    callback(workspace, fields)
    assert calls == [((
        workspace, theta, rho, pressure, exner, w,
        temperature, target, dz, 30.0,
    ), {"hail_on": True, "loaded_vapor": loaded, "shapes": (0.,1.)})]

    # The variant's hail switch is carried by the adapter, not smuggled in
    # from module state: an adapter built hail-off must forward hail-off.
    calls.clear()
    fused.NSSL2FusedGS(temperature, target, 30.0, hail_on=False)(
        workspace, fields)
    assert calls[0][1]["hail_on"] is False
    assert calls[0][1]["loaded_vapor"] is loaded


@pytest.mark.parametrize("dt_s", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_launcher_rejects_invalid_step_before_compilation(dt_s):
    with pytest.raises(ValueError, match="positive finite"):
        fused.launch_fused_gs(_workspace(), *_environment(), dt_s)


def test_launcher_rejects_non_scalar_step_before_compilation():
    with pytest.raises(TypeError, match="positive finite"):
        fused.launch_fused_gs(_workspace(), *_environment(), object())


def test_launcher_validates_workspace_and_environment_before_compilation():
    environment = list(_environment())
    with pytest.raises(TypeError, match="NSSL2DriverWorkspace"):
        fused.launch_fused_gs(object(), *environment, 1.0)

    workspace = _workspace()
    bad_state = fused.NSSL2DriverWorkspace(
        np.zeros((15, *workspace.shape), dtype=np.float32),
        workspace.category_surface_export,
        workspace.shape,
    )
    with pytest.raises(ValueError, match="workspace state"):
        fused.launch_fused_gs(bad_state, *environment, 1.0)

    bad_dtype = list(environment)
    bad_dtype[2] = bad_dtype[2].astype(np.float64)
    with pytest.raises(TypeError, match="pressure_pa must be float32"):
        fused.launch_fused_gs(workspace, *bad_dtype, 1.0)

    bad_shape = list(environment)
    bad_shape[5] = np.zeros((24,), dtype=np.float32)
    with pytest.raises(ValueError, match="temperature_k must have shape"):
        fused.launch_fused_gs(workspace, *bad_shape, 1.0)

    bad_velocity = list(environment)
    bad_velocity[4] = np.zeros((5, 2, 6), dtype=np.float32)[:, :, ::2]
    assert not bad_velocity[4].flags.c_contiguous
    with pytest.raises(ValueError, match="vertical_velocity must be C-contiguous"):
        fused.launch_fused_gs(workspace, *bad_velocity, 1.0)

    cell_velocity = list(environment)
    cell_velocity[4] = np.zeros(workspace.shape, dtype=np.float32)
    with pytest.raises(ValueError, match="interface field"):
        fused.launch_fused_gs(workspace, *cell_velocity, 1.0)

    noncontiguous = list(environment)
    noncontiguous[7] = np.zeros((4, 2, 6), dtype=np.float32)[:, :, ::2]
    assert not noncontiguous[7].flags.c_contiguous
    with pytest.raises(ValueError, match="dz must be C-contiguous"):
        fused.launch_fused_gs(workspace, *noncontiguous, 1.0)


def test_fused_launcher_does_not_import_or_call_isolated_process_surface():
    source = inspect.getsource(fused)
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "woof.core.nssl2" not in imported_modules


def _fused_gs_cuda_source() -> str:
    return (
        Path(__file__).parents[1]
        / "woof" / "core" / "kernels" / "nssl2_fused_gs.cu"
    ).read_text(encoding="utf-8")


def test_cuda_centres_interface_w_onto_mass_levels_exactly_once():
    """``module_mp_nssl_2mom.F:14174-14176`` is ONE average, not two.

    ``dyn_em/solve_em.F`` hands ``microphysics_driver`` the staggered
    ``grid%w_2``; ``module_microphysics_driver.F`` forwards it unchanged;
    and ``module_mp_nssl_2mom.F:2827`` copies it into the gather-scatter
    slab as ``wn(ix,1,kz) = w(ix,kz,jy)`` with no de-staggering.  So
    ``wvel = 0.5*(w(kp1) + w(kgs))`` **is** the interface-to-mass average
    and there is no second one to reproduce.  Averaging twice delivers
    ``0.25*w[k] + 0.5*w[k+1] + 0.25*w[k+2]`` -- a half-level upward shift
    plus 1-2-1 smoothing -- into ``diagnose_primary_ice``.
    """
    source = _fused_gs_cuda_source()
    assert ("    in.wk = vertical_velocity[k0 * ncol + col];\n"
            "    in.wkp1 = vertical_velocity[min(nz - 1, k0 + 1) * ncol + col];\n"
            ) in source
    assert "    const float wvel = 0.5f * (in.wkp1 + in.wk);\n" in source
    assert source.count("in.wkp1 + in.wk") == 1
    # The second average is gone, along with the false premise that put it
    # there ("WRF's microphysics driver supplies a mass-level W field").
    assert "w_mass_kp" not in source
    assert "mass-level W field" not in source


def test_official_wrf_oracle_pins_the_single_average_and_its_top_clamp():
    """The centering rule, checked against WRF's own instrumented output.

    ``woof/data/nssl2/fused-gs-oracle/fused-gs.csv`` is emitted by
    ``tools/nssl2_wrf461_fused_gs_oracle/fused_gs_oracle.F90``, which records
    ``w_lower_m_s = velocity(level)``, ``w_upper_m_s =
    velocity(min(level+1, nz))`` and the ``w_center_m_s`` that
    ``nssl_2mom_gs`` actually used.  Every row must satisfy the transcription
    this kernel now carries, and must NOT satisfy the two-stage one.

    This is a CHARACTERIZATION test: it pins which rule WRF follows, and it
    passes whatever the kernel does.  What ties it to the kernel is the
    source pin in the test above; what prices the difference is
    ``test_official_wrf_oracle_rows_reach_qiint_with_a_shifted_w_center``.
    """
    import csv

    path = (Path(__file__).parents[1] / "woof" / "data" / "nssl2"
            / "fused-gs-oracle" / "fused-gs.csv")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 240

    columns: dict[tuple[str, int], dict[int, tuple[float, float, float]]] = {}
    for row in rows:
        lower = float(row["w_lower_m_s"])
        upper = float(row["w_upper_m_s"])
        centre = float(row["w_center_m_s"])
        # WRF's statement, level by level.
        assert centre == 0.5 * (lower + upper)
        columns.setdefault(
            (row["case"], int(row["repetition"])), {}
        )[int(row["k"])] = (lower, upper, centre)

    # The oracle's four levels are one interface stack -- w_upper(k) is
    # w_lower(k+1) -- and the top level is clamped onto its own lower face,
    # which is WRF's kp1 = Min(nz, kgs+1) with no nz+1 entry to read.
    top = max(level for level in next(iter(columns.values())))
    disagreements = 0
    for levels in columns.values():
        for level, (lower, upper, _centre) in levels.items():
            if level < top:
                assert upper == levels[level + 1][0]
            else:
                assert upper == lower
        # The two-stage rule this kernel used to carry, on the same stack:
        # 0.5*(w_mass[k+1] + w_mass[k]) = 0.25 w[k] + 0.5 w[k+1] + 0.25 w[k+2].
        # It disagrees with WRF at EVERY level where it is defined.
        for level in sorted(levels):
            if level + 1 not in levels:
                continue
            two_stage = 0.5 * (levels[level + 1][2] + levels[level][2])
            assert two_stage != levels[level][2]
            disagreements += 1
    assert disagreements == 180        # 30 cases x 2 repetitions x 3 levels

    # A worked row, so the size of the error is on the record: case
    # "zero_clear_dt0p1" has interfaces -3.75, 0.25, 8.25, 18.25 m/s.
    first = columns[("zero_clear_dt0p1", 0)]
    assert [first[level][2] for level in (1, 2, 3, 4)] == [
        -1.75, 4.25, 13.25, 18.25]
    assert 0.5 * (first[2][2] + first[1][2]) == 1.25   # the old kernel
    assert first[1][2] == -1.75                        # WRF


def test_official_wrf_oracle_rows_reach_qiint_with_a_shifted_w_center():
    """The size of the two-stage error, measured on WRF's own rows.

    ``w_center`` reaches exactly one consumer, ``diagnose_primary_ice``, and
    WRF's ``icenucopt=1`` source is linear in it
    (``module_mp_nssl_2mom.F:20733-20738``,
    ``qiint = idqis*il5*(cmassin/rho0)*max(0,wvel)*...``).  Twelve rows of the
    official fused-GS oracle clear every gate that path applies, and on those
    rows the two-stage average this kernel used to carry overstates ``wvel``
    by 18% to 106% -- so it overstates primary ice nucleation by the same
    factor.  That is the failure this fix removes, priced in WRF's numbers.
    """
    import csv
    import math

    path = (Path(__file__).parents[1] / "woof" / "data" / "nssl2"
            / "fused-gs-oracle" / "fused-gs.csv")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    columns: dict[tuple[str, int], dict[int, dict[str, str]]] = {}
    for row in rows:
        columns.setdefault(
            (row["case"], int(row["repetition"])), {}
        )[int(row["k"])] = row

    def ice_saturation(temperature: float, pressure: float) -> float:
        index = min(1000001, max(
            1, int((temperature - 163.15) / 0.002 + 1.5)))
        table = 163.15 + (index - 1) * 0.002
        return (380.0 / pressure) * math.exp(
            21.87455 * (table - 273.15) / (table - 7.66))

    inflated: list[float] = []
    for levels in columns.values():
        order = sorted(levels)
        nz = len(order)
        target = [float(levels[key]["primary_ice_target_m3"]) for key in order]
        centre = [float(levels[key]["w_center_m_s"]) for key in order]
        for k in range(nz):
            row = levels[order[k]]
            temperature = float(row["temperature_before_k"])
            # The kernel's own gate, term for term.
            target_km = max(k - 1, 0)
            target_kp = min(k + 1, nz - 2)
            span = max(target_kp - k, 0) + max(k - target_km, 0)
            if not (temperature < 268.15
                    and float(row["qni_before"]) < 1.0e6
                    and centre[k] > 0.0
                    and float(row["dz_m"]) > 0.0
                    and span > 0
                    and target[target_kp] - target[target_km] > 0.0
                    and float(row["qv_before"]) / ice_saturation(
                        temperature, float(row["pressure_pa"])) > 1.0):
                continue
            two_stage = 0.5 * (centre[min(k + 1, nz - 1)] + centre[k])
            inflated.append(two_stage / centre[k])

    assert len(inflated) == 12
    assert min(inflated) > 1.18
    assert max(inflated) > 2.0
    # The worst row on the record: combined_bigg_qiacr, oracle level 2.
    worst = columns[("combined_bigg_qiacr", 0)]
    assert float(worst[2]["w_center_m_s"]) == 4.25
    assert 0.5 * (float(worst[3]["w_center_m_s"])
                  + float(worst[2]["w_center_m_s"])) == 8.75


def test_table_initialization_orders_independent_member_streams(monkeypatch):
    import sys
    from types import SimpleNamespace
    from woof.core import nssl2_fused_gs as fused

    events = []
    class Stream:
        def is_capturing(self): return False
        def wait_event(self, event): events.append(('wait', self, event))
    class Event:
        def __init__(self, **kw): assert kw == {'disable_timing': True}
        def record(self, stream): events.append(('record', stream, self))
    first, second = Stream(), Stream()
    active = [first]
    cp = SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(getDevice=lambda: 0), Event=Event,
        get_current_stream=lambda: active[0]))
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    monkeypatch.setattr(fused, '_TABLES_READY', {})
    monkeypatch.setattr(fused, 'shape_kernel',
        lambda *args: lambda *launch: events.append(('initialize', active[0])))
    fused._ensure_init_tables((0., 1., 7, 2))
    active[0] = second
    fused._ensure_init_tables((0., 1., 7, 2))
    assert [e[0] for e in events] == ['initialize', 'record', 'wait', 'wait']
    assert events[2][2] is events[3][2]
    assert events[3][1] is second
