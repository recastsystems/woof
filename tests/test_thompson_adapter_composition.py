"""Composition gates for the guarded classic-Thompson adapter.

The older Thompson GPU tests deliberately admit individual CUDA slices.  The
tests in this module protect the production call graph: warm and cold source
networks must be disjoint, classic graupel number must exist on every call,
and the same number scratch must flow through source, fallout, final cleanup,
and (when requested) reflectivity.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu


_ORACLE = Path(__file__).parents[1] / "woof" / "data" / "thompson" / "oracle"


class _HostAdapterState:
    """Small NumPy-backed state for a launcher-only adapter call-graph gate."""

    def __init__(self) -> None:
        shape = (3, 1, 1)
        interface_shape = (4, 1, 1)
        self.p = np.full(shape, 80000.0, dtype=np.float32)
        self.thb = np.full((3,), 300.0, dtype=np.float32)
        self.thp = np.zeros(shape, dtype=np.float32)
        self.phb = np.asarray(
            [0.0, 9810.0, 19620.0, 29430.0], dtype=np.float32)
        self.php = np.zeros(interface_shape, dtype=np.float32)
        self.w = np.zeros(interface_shape, dtype=np.float32)
        self.qv = np.full(shape, 0.005, dtype=np.float32)
        self.qc = np.full(shape, 2.0e-4, dtype=np.float32)
        self.qr = np.full(shape, 1.0e-4, dtype=np.float32)
        self.nr = np.full(shape, 2.0e5, dtype=np.float32)
        self.qi = np.full(shape, 1.0e-6, dtype=np.float32)
        self.ni = np.full(shape, 1.0e4, dtype=np.float32)
        self.qs = np.full(shape, 2.0e-5, dtype=np.float32)
        self.qg = np.full(shape, 3.0e-5, dtype=np.float32)
        self.effc = np.zeros(shape, dtype=np.float32)
        self.effi = np.zeros(shape, dtype=np.float32)
        self.effs = np.zeros(shape, dtype=np.float32)
        self.h_diabatic = np.zeros(shape, dtype=np.float32)
        self._scratch: dict[str, np.ndarray] = {}

    def scratch(self, shape, name, dtype=None):
        value = self._scratch.get(name)
        if value is None:
            value = np.zeros(shape, dtype=np.float32 if dtype is None
                             else dtype)
            self._scratch[name] = value
        else:
            assert value.shape == tuple(shape)
        return value


def _record_adapter_call(monkeypatch, *, refl_due: bool):
    """Run the real mp=8 adapter while replacing every GPU launcher with a spy.

    mp=8 runs classic Thompson (woof/core/thompson_classic.py); every
    launcher in its ``LAUNCHERS`` and its reflectivity launcher are spied.
    No real kernel may be reached from this call: every woof binding of the
    kernel loader is replaced by one that fails the gate by name.  The
    breakage that prevents: a launch joining the adapter without joining
    the spy list, so on the host backend the real kernel ran and the gate
    stayed green without recording the launch it exists to see (THOMPSON-
    CARD 4.1, RTX 4090, b2e0bc80).
    """
    import woof.core.kernels as kernel_loader
    import woof.core.microphysics as microphysics
    import woof.core.refl as refl
    import woof.core.thompson_classic as classic
    import woof.core.thompson_runtime as runtime

    calls: list[tuple[str, tuple, dict]] = []

    def spy(name):
        def launch(*args, **kwargs):
            calls.append((name, args, kwargs))
        return launch

    for name in classic.LAUNCHERS:
        monkeypatch.setattr(classic, name, spy(name))

    def unspied(loader_name):
        def refuse(module_name, func, *args, **kwargs):
            raise AssertionError(
                f"_apply_thompson reached the real kernel {module_name}:"
                f"{func} through {loader_name}, from a launcher this gate "
                "does not spy; add the launcher to thompson_classic.LAUNCHERS "
                "and pin its place in the call graph")
        return refuse

    for loader_name in ("get_kernel", "get_kernel_int_defines"):
        loader = getattr(kernel_loader, loader_name)
        for module in list(sys.modules.values()):
            if (getattr(module, "__name__", "").startswith("woof.")
                    and getattr(module, loader_name, None) is loader):
                monkeypatch.setattr(module, loader_name, unspied(loader_name))

    table_owner = object()
    monkeypatch.setattr(
        runtime, "load_classic_device_tables", lambda _root: table_owner)
    monkeypatch.setattr(refl, "stash_refl_10cm", spy("stash"))
    monkeypatch.setenv("WOOF_THOMPSON_TABLE_ROOT", "host-call-graph-fixture")

    state = _HostAdapterState()
    diagnostics = microphysics._apply_thompson_strict(
        state, SimpleNamespace(), 10.0, refl_10cm_due=refl_due)
    assert diagnostics.rainnc is state._scratch["mp_rainnc"]
    return calls, state, table_owner


def _named_call(calls, name):
    matches = [call for call in calls if call[0] == name]
    assert len(matches) == 1, (name, [call[0] for call in calls])
    return matches[0]


def test_rain_evaporation_rejects_density_marker_identity_alias():
    """The held prr_gml marker and RHOF output have overlapping lifetimes."""
    from woof.core.thompson import launch_rain_evaporation

    shape = (2, 1, 1)
    fields = [np.zeros(shape, dtype=np.float32) for _ in range(5)]
    shared = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="must not alias reference_density"):
        launch_rain_evaporation(
            *fields, 10.0,
            reference_density=shared,
            graupel_melt_marker=shared)


def test_adapter_composes_all_thompson_phases_independently_of_output_cadence(
        monkeypatch):
    """mp=8's classic call graph: one launch per phase, in WRF's order.

    Asking for an output may add only the diagnostic itself, and the one
    private graupel number (WRF's ng1d) flows from the entry through the
    sources and the column stage into the reflectivity.
    """
    calls_off, state_off, owner_off = _record_adapter_call(
        monkeypatch, refl_due=False)
    calls_on, state_on, owner_on = _record_adapter_call(
        monkeypatch, refl_due=True)
    names_off = [name for name, _args, _kwargs in calls_off]
    names_on = [name for name, _args, _kwargs in calls_on]

    reflectivity = ("launch_refl10cm_classic", "stash")
    assert names_off == [name for name in names_on
                         if name not in reflectivity]
    assert names_off == ["launch_classic_entry", "launch_classic_sources",
                         "launch_classic_column",
                         "launch_classic_effective_radius",
                         "launch_classic_vapour_fill",
                         "launch_classic_finish"]
    assert names_on == ["launch_classic_entry", "launch_classic_sources",
                        "launch_classic_column", "launch_refl10cm_classic",
                        "stash", "launch_classic_effective_radius",
                        "launch_classic_vapour_fill",
                        "launch_classic_finish"]

    for calls, state, owner in ((calls_off, state_off, owner_off),
                                (calls_on, state_on, owner_on)):
        shadow = state._scratch["mp_thompson_graupel_number_shadow"]
        _, entry_args, _ = _named_call(calls, "launch_classic_entry")
        _, sources_args, _ = _named_call(calls, "launch_classic_sources")
        _, column_args, _ = _named_call(calls, "launch_classic_column")
        assert entry_args[5] is shadow
        assert sources_args[2] is shadow
        assert sources_args[5] is owner
        assert column_args[4] is shadow
        if "launch_refl10cm_classic" in [call[0] for call in calls]:
            _, refl_args, _ = _named_call(calls, "launch_refl10cm_classic")
            assert refl_args[5] is shadow
            _, stash_args, _ = _named_call(calls, "stash")
            assert stash_args[1] is state._scratch["refl_10cm"]


def _device_adapter_state():
    """Small guarded-mp8 device state with every hydrometeor group active."""
    import cupy as cp

    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_theta_perturbation

    cfg = RunConfig(
        nx=16, ny=12, nz=24, dx=1000.0, dy=1000.0, ztop=12000.0,
        dt=2.0, run_seconds=40.0, time_step_sound=4,
        moist=True, mp_physics=8)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.003 * np.asarray(z, dtype=float),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_theta_perturbation(
        cfg, coord, base,
        lambda x, z: 0.20 * np.exp(-(np.asarray(z) / 4500.0) ** 2)[
            :, None, None])

    with (_ORACLE / "mixed-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["phase"] == "before"]

    def profile(name):
        return np.asarray([float(row[name]) for row in rows], np.float32)

    for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
        values = profile(name)[:, None, None]
        getattr(state, name)[...] = cp.asarray(
            np.broadcast_to(values, (cfg.nz, cfg.ny, cfg.nx)))
    state.nr[...] = cp.asarray(np.broadcast_to(
        profile("nr_per_kg")[:, None, None], state.nr.shape))
    state.ni[...] = cp.asarray(np.broadcast_to(
        profile("ni_per_kg")[:, None, None], state.ni.shape))

    # A fresh init_theta_perturbation state has no EOS diagnostics yet
    # (dycore.step computes p/al/alt before calling microphysics); the
    # adapter consumes state.p directly, so compute them here exactly as
    # the model epilogue would.
    from woof.core.diagnostics import update_diagnostics

    update_diagnostics(state, cfg.hypsometric_opt)

    # The oracle qv column pairs with the oracle's own thermodynamics; that
    # qv can be subsaturated against this synthetic column, in which case the
    # scheme's saturation adjustment would evaporate every cloud cell before
    # the effective-radius kernel runs, leaving only background radii.  Carry
    # the lowest levels to slight liquid supersaturation against the state's
    # OWN temperature/pressure so the adjustment condenses cloud instead and
    # the composed gate exercises diagnosed (off-background) cloud radii.
    from woof.core import constants as gc

    thb = cp.asnumpy(state.thb)
    if thb.ndim == 1:
        thb = thb[:, None, None]
    theta = thb + cp.asnumpy(state.thp)
    pressure = np.asarray(cp.asnumpy(state.p), dtype=np.float64)
    temperature = theta * (pressure / gc.P0) ** gc.RCP
    saturation_vapor_pa = 611.2 * np.exp(
        17.67 * (temperature - 273.15) / (temperature - 29.65))
    saturation_mixing = (0.622 * saturation_vapor_pa
                         / (pressure - saturation_vapor_pa))
    state.qv[:6] = cp.asarray(
        np.asarray(1.10 * saturation_mixing[:6], dtype=np.float32))
    return state, cfg


@requires_gpu
@pytest.mark.gpu
def test_apply_thompson_publishes_kernel_micron_radii_exactly_once(
        monkeypatch):
    """Composed end-to-end radius gate: exactly ONE metre->micron conversion.

    The effective-radius kernel writer (kernels/thompson.cu) applies WRF's
    own driver-side metre->micron conversion after its clamps, so the state
    contract fields ``effc``/``effi``/``effs`` must hold the kernel's device
    output with not one changed bit.  Two earlier gates each proved half of
    this seam in isolation -- the kernel writes microns; a standalone helper
    scaled metres -- and no test composed them, which is exactly how a second
    adapter-side multiply (a ~1e6-fold magnification of every published
    radius) escaped.  This gate drives the real
    ``_apply_thompson`` path and pins both halves together:

    (a) ``state.effc/effi/effs`` are BIT-IDENTICAL to the kernel's micron
        output, captured on device at the launch seam (no tolerance); and
    (b) every published radius sits inside the kernel's own final clamp
        bands converted by its own fl(1e6) factor -- fl(2.49e-6)*fl(1e6) ..
        fl(50.0e-6)*fl(1e6) um cloud, fl(4.99e-6)*fl(1e6) ..
        fl(125.0e-6)*fl(1e6) um ice, fl(9.99e-6)*fl(1e6) ..
        fl(999.0e-6)*fl(1e6) um snow (thompson.cu final writer clamps).
    """
    import cupy as cp

    import woof.core.microphysics as microphysics
    import woof.core.thompson as thompson

    table_root = os.environ.get("WOOF_THOMPSON_TABLE_ROOT")
    if not table_root or not Path(table_root).is_dir():
        pytest.skip("WOOF_THOMPSON_TABLE_ROOT with the validated classic "
                    "tables is required for the composed mp8 radius gate")

    state, cfg = _device_adapter_state()

    real_launch = thompson.launch_effective_radius
    kernel_output: dict[str, object] = {}

    def capture_effective_radius(temperature, pressure, qv, qc, qi, ni, qs,
                                 effc, effi, effs):
        real_launch(temperature, pressure, qv, qc, qi, ni, qs,
                    effc, effi, effs)
        cp.cuda.Stream.null.synchronize()
        kernel_output["effc"] = effc.copy()
        kernel_output["effi"] = effi.copy()
        kernel_output["effs"] = effs.copy()

    monkeypatch.setattr(
        thompson, "launch_effective_radius", capture_effective_radius)

    microphysics._apply_thompson_strict(state, cfg, float(cfg.dt))
    cp.cuda.Stream.null.synchronize()
    assert set(kernel_output) == {"effc", "effi", "effs"}

    def micron_band(low_metres, high_metres):
        scale = np.float32(1.0e6)
        return (np.float32(np.float32(low_metres) * scale),
                np.float32(np.float32(high_metres) * scale))

    bands = {
        "effc": micron_band(2.49e-6, 50.0e-6),
        "effi": micron_band(4.99e-6, 125.0e-6),
        "effs": micron_band(9.99e-6, 999.0e-6),
    }
    for name in ("effc", "effi", "effs"):
        published = cp.asnumpy(getattr(state, name))
        from_kernel = cp.asnumpy(kernel_output[name])
        # (a) Exactly one conversion: the published state field is the
        # kernel's micron output, bit for bit.
        np.testing.assert_array_equal(
            published, from_kernel,
            err_msg=f"state.{name} is not the kernel's own micron output "
                    "(a second conversion ran between kernel and state)")
        assert published.tobytes() == from_kernel.tobytes(), name
        # (b) Physical magnitude: inside the kernel's own clamp band, in
        # microns (order 1e0..1e3, never 1e6-fold beyond it).
        low, high = bands[name]
        assert float(published.min()) >= float(low), (
            f"state.{name} min {float(published.min())!r} um below the "
            f"kernel clamp floor {float(low)!r} um")
        assert float(published.max()) <= float(high), (
            f"state.{name} max {float(published.max())!r} um above the "
            f"kernel clamp ceiling {float(high)!r} um")
        # The mixed column carries active cloud/ice/snow, so the composed
        # path must publish at least one diagnosed (off-background) radius.
        assert bool((published != low).any()), (
            f"state.{name} never left the clear-sky background -- the "
            "composed gate did not exercise a diagnosed radius")


def _oracle_case(name):
    with (_ORACLE / f"{name}-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    with (_ORACLE / f"{name}-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    assert len(rows) == 48
    return rows[:24], rows[24:], surface


def _host(rows, name):
    return np.asarray([float(row[name]) for row in rows], dtype=np.float32)


def _warm_efficiency_table(cp):
    """Sparse canonical t_Efrw rows touched by warm-overlap."""
    table = np.zeros((100, 100), dtype=np.float64, order="F")
    for cloud_bin, value in {
            2: 0.0,
            4: 0.09128311276435852,
            9: 0.5825913548469543,
            15: 0.8137867450714111,
            22: 0.898698091506958,
            23: 0.9053093791007996,
            27: 0.9252923727035522,
            28: 0.9291030168533325,
            30: 0.9357151985168457,
            }.items():
        table[49, cloud_bin - 1] = value
    return cp.asarray(table, order="F")


def _warm_owner(table):
    import cupy as cp

    from woof.core.thompson_runtime import (
        RAIN_FREEZING_TABLE_NAMES,
        RAIN_GRAUPEL_TABLE_NAMES,
        RAIN_SNOW_TABLE_NAMES,
        DeviceClassicTableSet,
    )

    # The verified owner is the production coefficient seam.  Its cold-table
    # view names every group even though this warm launcher dereferences only
    # t_Efrw, t_Efsw, and the two collision groups.  Alias one zero array per
    # required shape so this test remains small without weakening shape checks.
    rain_snow = cp.zeros((37, 9, 37, 37), dtype=cp.float64, order="F")
    rain_graupel = cp.zeros(
        (37, 37, 1, 37, 37), dtype=cp.float64, order="F")
    arrays = {
        "t_Efrw": table,
        "t_Efsw": cp.zeros((100, 100), dtype=cp.float64, order="F"),
        "tpi_ide": table,
        "tps_iaus": table,
        "tni_iaus": table,
        "tpi_qcfz": table,
        "tni_qcfz": table,
    }
    arrays.update({name: rain_snow for name in RAIN_SNOW_TABLE_NAMES})
    arrays.update({name: rain_graupel for name in RAIN_GRAUPEL_TABLE_NAMES})
    arrays.update({name: table for name in RAIN_FREEZING_TABLE_NAMES})

    return DeviceClassicTableSet(
        root=Path("warm-source-test-owner"),
        arrays=arrays,
        payload_bytes=int(
            table.nbytes + arrays["t_Efsw"].nbytes
            + rain_snow.nbytes + rain_graupel.nbytes),
        array_sha256={},
        identity_json="{}",
        identity_sha256="test-only-warm-owner",
        device_id=0,
        upload_seconds=0.0,
        verification_seconds=0.0,
        roundtrip_verified=True,
    )


@requires_gpu
@pytest.mark.gpu
def test_production_warm_source_matches_wrf_and_is_strictly_warm_only():
    import cupy as cp

    from woof.core.thompson import (
        launch_rain_sedimentation,
        launch_warm_frozen_source_network_from_owner,
    )

    before, after, surface = _oracle_case("warm-overlap")

    def volume(name):
        return cp.asarray(_host(before, name)[:, None, None])

    owner = _warm_owner(_warm_efficiency_table(cp))
    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qc = volume("qc")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qs = cp.zeros_like(qc)
    qg = cp.zeros_like(qc)
    ng = cp.zeros_like(qc)
    entry_warm = cp.ones_like(qc)
    snow_melt_marker = cp.empty_like(qc)
    dz = volume("dz_m")
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)

    launch_warm_frozen_source_network_from_owner(
        qc, qr, nr, qs, qg, ng, entry_warm, snow_melt_marker,
        temperature, pressure, qv, owner, 10.0)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, 10.0)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 0.0, 3.0e-5),
            (qv, "qv", 0.0, 2.0e-10),
            (qc, "qc", 1.0e-5, 3.0e-10),
            (qr, "qr", 1.0e-5, 3.0e-11),
            (nr, "nr_per_kg", 1.0e-5, 3.0e-2)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), _host(after, name),
            rtol=rtol, atol=atol)
    expected_rain = np.float32(float(surface["rainncv_mm"]))
    np.testing.assert_allclose(
        cp.asnumpy(rainncv), expected_rain, rtol=1.0e-5, atol=3.0e-9)

    # The cold fused network already owns these cells.  A global invocation of
    # the warm source would silently apply autoconversion/accretion twice.
    # The second cell emulates a cold-entry cell heated across freezing by the
    # preceding cold source.  The held entry_warm marker, not the mutated
    # temperature, must own branch selection for the complete source call.
    cold_temperature = cp.asarray(
        [268.15, 274.15], dtype=cp.float32)[:, None, None]
    cold_entry_warm = cp.zeros_like(cold_temperature)
    cold_snow_melt_marker = cp.full_like(cold_temperature, 1.0)
    cold_pressure = cp.full_like(cold_temperature, 80000.0)
    cold_qv = cp.full_like(cold_temperature, 0.004)
    cold_qc = cp.full_like(cold_temperature, 8.0e-4)
    cold_qr = cp.full_like(cold_temperature, 3.0e-4)
    cold_nr = cp.full_like(cold_temperature, 2.0e5)
    cold_qs = cp.full_like(cold_temperature, 2.0e-4)
    cold_qg = cp.full_like(cold_temperature, 2.0e-4)
    cold_ng = cp.full_like(cold_temperature, 8.0e3)
    mutable = (
        cold_temperature, cold_qc, cold_qr, cold_nr,
        cold_qs, cold_qg, cold_ng)
    snapshots = tuple(field.copy() for field in mutable)
    launch_warm_frozen_source_network_from_owner(
        cold_qc, cold_qr, cold_nr, cold_qs, cold_qg, cold_ng,
        cold_entry_warm, cold_snow_melt_marker,
        cold_temperature, cold_pressure, cold_qv, owner, 10.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(mutable, snapshots, strict=True):
        np.testing.assert_array_equal(cp.asnumpy(actual), cp.asnumpy(expected))
    np.testing.assert_array_equal(cp.asnumpy(cold_entry_warm), 0.0)
    np.testing.assert_array_equal(cp.asnumpy(cold_snow_melt_marker), 0.0)


@requires_gpu
@pytest.mark.gpu
def test_production_snow_melt_and_combined_fallout_match_wrf():
    import cupy as cp

    from woof.core.thompson import (
        launch_classic_graupel_number_init,
        launch_cloud_saturation_adjust,
        launch_final_phase_cleanup,
        launch_rain_evaporation,
        launch_rain_sedimentation,
        launch_snow_sedimentation,
        launch_warm_frozen_source_network_from_owner,
    )

    before, after, surface = _oracle_case("snow-melt")
    dt = float(surface["dt_s"])

    def volume(name):
        return cp.asarray(_host(before, name)[:, None, None])

    owner = _warm_owner(cp.asarray(
        np.zeros((100, 100), dtype=np.float64, order="F"), order="F"))
    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qc = volume("qc")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    qs = volume("qs")
    qg = volume("qg")
    dz = volume("dz_m")
    ng = cp.empty_like(qg)
    entry_warm = cp.ones_like(qg)
    snow_melt_marker = cp.empty_like(qg)
    frozen_density = cp.empty_like(qg)
    frozen_temperature = cp.empty_like(qg)
    rain_density = cp.empty_like(qg)
    snow_velocity_boost = cp.ones_like(qg)
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)

    launch_classic_graupel_number_init(
        qg, temperature, pressure, qv, ng)
    launch_warm_frozen_source_network_from_owner(
        qc, qr, nr, qs, qg, ng, entry_warm, snow_melt_marker,
        temperature, pressure, qv, owner, dt)
    launch_cloud_saturation_adjust(
        temperature, pressure, qv, qc,
        reference_density=frozen_density,
        reference_temperature=frozen_temperature)
    launch_rain_evaporation(
        qr, nr, temperature, pressure, qv, dt,
        reference_density=rain_density,
        graupel_melt_marker=entry_warm)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, dt,
        snow_melt_marker=snow_melt_marker,
        melt_rain_qr=qr,
        melt_rain_nr=nr,
        reference_density=frozen_density,
        reference_temperature=frozen_temperature,
        velocity_boost=snow_velocity_boost,
        accumulate_surface=True)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, dt,
        reference_density=rain_density,
        accumulate_surface=True)
    launch_final_phase_cleanup(qc, qi, ni, temperature, pressure, qv)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 3.0e-6, 3.0e-5),
            (qv, "qv", 5.0e-6, 3.0e-10),
            (qc, "qc", 8.0e-6, 1.0e-9),
            (qr, "qr", 1.2e-5, 2.0e-11),
            (nr, "nr_per_kg", 1.2e-5, 3.0e-2),
            (qs, "qs", 1.2e-5, 2.0e-11)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), _host(after, name),
            rtol=rtol, atol=atol)
    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"),
            (snowncv, "snowncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.2e-5, atol=3.0e-10)


@requires_gpu
@pytest.mark.gpu
def test_production_graupel_melt_fallout_number_and_refl_match_wrf():
    import cupy as cp

    from woof.core.refl import launch_refl10cm_thompson
    from woof.core.thompson import (
        launch_classic_graupel_number_finalize,
        launch_classic_graupel_number_init,
        launch_cloud_saturation_adjust,
        launch_final_phase_cleanup,
        launch_graupel_fallout_column_mask,
        launch_graupel_sedimentation,
        launch_rain_evaporation,
        launch_rain_sedimentation,
        launch_warm_frozen_source_network_from_owner,
    )

    before, after, surface = _oracle_case("graupel-melt")
    dt = float(surface["dt_s"])

    def volume(name):
        return cp.asarray(_host(before, name)[:, None, None])

    owner = _warm_owner(cp.asarray(
        np.zeros((100, 100), dtype=np.float64, order="F"), order="F"))
    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qc = volume("qc")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    qs = volume("qs")
    qg = volume("qg")
    dz = volume("dz_m")
    entry_active = (qg > cp.float32(1.0e-12)).astype(cp.float32)
    active_columns = cp.empty((1, 1), dtype=cp.float32)
    ng = cp.empty_like(qg)
    entry_warm = cp.ones_like(qg)
    snow_melt_marker = cp.empty_like(qg)
    frozen_density = cp.empty_like(qg)
    frozen_temperature = cp.empty_like(qg)
    rain_density = cp.empty_like(qg)
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    graupelnc = cp.zeros_like(rainnc)
    graupelncv = cp.zeros_like(rainnc)

    launch_classic_graupel_number_init(
        qg, temperature, pressure, qv, ng)
    launch_warm_frozen_source_network_from_owner(
        qc, qr, nr, qs, qg, ng, entry_warm, snow_melt_marker,
        temperature, pressure, qv, owner, dt)
    launch_graupel_fallout_column_mask(
        entry_active, qg, active_columns)
    launch_cloud_saturation_adjust(
        temperature, pressure, qv, qc,
        reference_density=frozen_density,
        reference_temperature=frozen_temperature)
    launch_rain_evaporation(
        qr, nr, temperature, pressure, qv, dt,
        reference_density=rain_density,
        graupel_melt_marker=entry_warm)
    launch_graupel_sedimentation(
        qg, temperature, pressure, qv, dz,
        rainnc, rainncv, graupelnc, graupelncv, dt,
        reference_density=frozen_density,
        active_columns=active_columns,
        graupel_number_shadow=ng)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, dt,
        reference_density=rain_density,
        accumulate_surface=True)
    launch_final_phase_cleanup(qc, qi, ni, temperature, pressure, qv)
    launch_classic_graupel_number_finalize(
        qg, temperature, pressure, qv, ng)
    refl = cp.empty_like(qg)
    launch_refl10cm_thompson(
        qv, qr, nr, qs, qg, ng, temperature, pressure, refl)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 3.0e-6, 3.0e-5),
            (qv, "qv", 5.0e-6, 3.0e-10),
            (qc, "qc", 8.0e-6, 1.0e-9),
            (qr, "qr", 1.2e-5, 2.0e-11),
            (nr, "nr_per_kg", 1.2e-5, 3.0e-2),
            (qg, "qg", 1.2e-5, 5.0e-10),
            (refl, "refl_dbz", 0.0, 5.0e-2)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), _host(after, name),
            rtol=rtol, atol=atol)

    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (graupelnc, "graupelnc_mm"),
            (graupelncv, "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.2e-5, atol=3.0e-10)


@requires_gpu
@pytest.mark.gpu
def test_production_warm_subsaturated_snow_graupel_path_matches_wrf():
    """Pin the coupled warm frozen path, including classic-ng continuity."""
    import cupy as cp

    from woof.core.refl import launch_refl10cm_thompson
    from woof.core.thompson import (
        launch_classic_graupel_number_finalize,
        launch_classic_graupel_number_init,
        launch_cloud_saturation_adjust,
        launch_final_phase_cleanup,
        launch_graupel_fallout_column_mask,
        launch_graupel_sedimentation,
        launch_rain_evaporation,
        launch_rain_sedimentation,
        launch_snow_sedimentation,
        launch_warm_frozen_source_network_from_owner,
    )

    before, after, surface = _oracle_case("warm-frozen-subsat")
    dt = float(surface["dt_s"])

    def volume(name):
        return cp.asarray(_host(before, name)[:, None, None])

    owner = _warm_owner(cp.asarray(
        np.zeros((100, 100), dtype=np.float64, order="F"), order="F"))
    temperature = volume("temp_k")
    pressure = volume("p_pa")
    qv = volume("qv")
    qc = volume("qc")
    qr = volume("qr")
    nr = volume("nr_per_kg")
    qi = volume("qi")
    ni = volume("ni_per_kg")
    qs = volume("qs")
    qg = volume("qg")
    dz = volume("dz_m")
    entry_qg = (qg > cp.float32(1.0e-12)).astype(cp.float32)
    entry_warm = cp.ones_like(qg)
    snow_melt_marker = cp.empty_like(qg)
    active_columns = cp.empty((1, 1), dtype=cp.float32)
    ng = cp.empty_like(qg)
    frozen_density = cp.empty_like(qg)
    frozen_temperature = cp.empty_like(qg)
    rain_density = cp.empty_like(qg)
    snow_velocity_boost = cp.ones_like(qg)
    rainnc = cp.zeros((1, 1), dtype=cp.float32)
    rainncv = cp.zeros_like(rainnc)
    snownc = cp.zeros_like(rainnc)
    snowncv = cp.zeros_like(rainnc)
    graupelnc = cp.zeros_like(rainnc)
    graupelncv = cp.zeros_like(rainnc)

    launch_classic_graupel_number_init(
        qg, temperature, pressure, qv, ng)
    launch_warm_frozen_source_network_from_owner(
        qc, qr, nr, qs, qg, ng, entry_warm, snow_melt_marker,
        temperature, pressure, qv, owner, dt)

    # WRF carries the source-updated classic ng into fallout.  Re-diagnosing
    # it from post-source qg here would erase the independent number tendency
    # and can materially alter both number fallout and reflectivity.
    source_ng = ng.copy()
    rediagnosed_ng = cp.empty_like(ng)
    launch_classic_graupel_number_init(
        qg, temperature, pressure, qv, rediagnosed_ng)
    launch_graupel_fallout_column_mask(
        entry_qg, qg, active_columns)
    launch_cloud_saturation_adjust(
        temperature, pressure, qv, qc,
        reference_density=frozen_density,
        reference_temperature=frozen_temperature)
    launch_rain_evaporation(
        qr, nr, temperature, pressure, qv, dt,
        reference_density=rain_density,
        graupel_melt_marker=entry_warm)
    launch_snow_sedimentation(
        qs, temperature, pressure, qv, dz,
        rainnc, rainncv, snownc, snowncv, dt,
        snow_melt_marker=snow_melt_marker,
        melt_rain_qr=qr,
        melt_rain_nr=nr,
        reference_density=frozen_density,
        reference_temperature=frozen_temperature,
        velocity_boost=snow_velocity_boost,
        accumulate_surface=True)
    launch_graupel_sedimentation(
        qg, temperature, pressure, qv, dz,
        rainnc, rainncv, graupelnc, graupelncv, dt,
        reference_density=frozen_density,
        active_columns=active_columns,
        graupel_number_shadow=ng,
        accumulate_surface=True)
    launch_rain_sedimentation(
        qr, nr, temperature, pressure, qv, dz,
        rainnc, rainncv, dt,
        reference_density=rain_density,
        accumulate_surface=True)
    launch_final_phase_cleanup(qc, qi, ni, temperature, pressure, qv)
    launch_classic_graupel_number_finalize(
        qg, temperature, pressure, qv, ng)
    refl = cp.empty_like(qg)
    launch_refl10cm_thompson(
        qv, qr, nr, qs, qg, ng, temperature, pressure, refl)
    cp.cuda.Stream.null.synchronize()

    for actual, name, rtol, atol in (
            (temperature, "temp_k", 3.0e-6, 3.0e-5),
            (qv, "qv", 5.0e-6, 3.0e-10),
            (qc, "qc", 8.0e-6, 1.0e-9),
            (qr, "qr", 1.5e-5, 3.0e-10),
            (nr, "nr_per_kg", 1.5e-5, 5.0e-2),
            (qs, "qs", 1.5e-5, 3.0e-10),
            (qg, "qg", 1.5e-5, 3.0e-10),
            (refl, "refl_dbz", 0.0, 5.0e-2)):
        np.testing.assert_allclose(
            cp.asnumpy(actual[:, 0, 0]), _host(after, name),
            rtol=rtol, atol=atol)

    for actual, name in (
            (rainnc, "rainnc_mm"),
            (rainncv, "rainncv_mm"),
            (snownc, "snownc_mm"),
            (snowncv, "snowncv_mm"),
            (graupelnc, "graupelnc_mm"),
            (graupelncv, "graupelncv_mm")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), np.float32(float(surface[name])),
            rtol=1.5e-5, atol=3.0e-9)

    initial_qs = _host(before, "qs")
    initial_qg = _host(before, "qg")
    active = (initial_qs > 0.0) & (initial_qg > 0.0)
    actual_temperature = cp.asnumpy(temperature[:, 0, 0])
    actual_qv = cp.asnumpy(qv[:, 0, 0])
    actual_qr = cp.asnumpy(qr[:, 0, 0])
    actual_qs = cp.asnumpy(qs[:, 0, 0])
    actual_qg = cp.asnumpy(qg[:, 0, 0])
    assert np.any(actual_temperature[active] < _host(before, "temp_k")[active])
    assert np.any(actual_qv[active] > _host(before, "qv")[active])
    assert np.any(actual_qr[active] > 0.0)
    assert np.all(actual_qs[active] < initial_qs[active])
    assert np.all(actual_qg[active] < initial_qg[active])
    assert not np.allclose(
        cp.asnumpy(source_ng[:, 0, 0]),
        cp.asnumpy(rediagnosed_ng[:, 0, 0]),
        rtol=1.0e-5, atol=1.0e-2)
