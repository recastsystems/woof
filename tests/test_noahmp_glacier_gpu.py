"""Device behaviour of the NOAHMP_GLACIER port.

Three claims that need a card:

* the CUDA column (woof/core/kernels/noahmp_glacier.cu) is BITWISE
  equal -- every output word, every column, every step -- to the CPython
  transcription over a multi-regime battery chained through 12 steps;
* the MPAS seam runs the full phase-1 chain with glacier columns
  present: the collaborator signature (landmask=1, IVGTYP=15, xice=0)
  admits and advances, the native-XLAND contract column (landmask=0,
  native xland=1, fractional ice below the 0.02 threshold) dispatches to
  the ice surface and never the water path, the classification receipt
  counts every class, the read-only guarantee holds, and the census
  names the glacier execution path;
* restart: an exported seam with glacier columns restores and continues
  bit-identically, and the glacier dispatch is red-on-revert (disabling
  the path refuses instead of silently running SFLX).
"""

import datetime

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof.core import mpas_column_batch as mcb  # noqa: E402
from woof.core.noahmp_glacier_gpu import (  # noqa: E402
    GLACIER_OUTPUT_ARRAYS, GLACIER_OUTPUT_SCALARS,
    evaluate_glacier_columns, evaluate_glacier_columns_on_host)

NZ, NCOL = 20, 8
DT = 120.0
START = datetime.datetime(2021, 6, 1, 18, 0)
ISICE = 15
ZSOIL = np.array([-0.1, -0.4, -1.0, -2.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# the battery: CUDA vs CPython, bit for bit, chained
# ---------------------------------------------------------------------------

def _battery(count=24, seed=7):
    """Multi-regime glacier columns: bare cold/warm, thin and deep
    snowpacks, melting packs with liquid, heavy snowfall, sublimation."""
    rng = np.random.default_rng(seed)
    f32 = np.float32
    inputs = {
        "cosz": rng.uniform(-0.2, 0.9, count).astype(f32),
        "sfctmp": rng.uniform(248.0, 278.0, count).astype(f32),
        "sfcprs": rng.uniform(70000.0, 101000.0, count).astype(f32),
        "uu": rng.uniform(-12.0, 12.0, count).astype(f32),
        "vv": rng.uniform(-12.0, 12.0, count).astype(f32),
        "q2": rng.uniform(1e-4, 6e-3, count).astype(f32),
        "soldn": rng.uniform(0.0, 700.0, count).astype(f32),
        "prcp": rng.uniform(0.0, 4e-4, count).astype(f32),
        "lwdn": rng.uniform(150.0, 340.0, count).astype(f32),
        "tbot": np.minimum(rng.uniform(255.0, 270.0, count),
                           263.15).astype(f32),
        "zlvl": np.full(count, 25.0, dtype=f32),
        "qsnow": rng.uniform(0.0, 2e-4, count).astype(f32),
        "sneqvo": np.zeros(count, dtype=f32),
        "albold": rng.uniform(0.55, 0.8, count).astype(f32),
        "cm": np.full(count, 0.01, dtype=f32),
        "ch": np.full(count, 0.01, dtype=f32),
        "tg": rng.uniform(255.0, 274.0, count).astype(f32),
        "tauss": rng.uniform(0.0, 2.0, count).astype(f32),
        "qsfc": rng.uniform(2e-4, 3e-3, count).astype(f32),
        "zsoil": ZSOIL,
    }
    isnow = np.zeros(count, dtype=np.int32)
    sneqv = np.zeros(count, dtype=np.float32)
    snowh = np.zeros(count, dtype=np.float32)
    snice = np.zeros((count, 3), dtype=np.float32)
    snliq = np.zeros((count, 3), dtype=np.float32)
    zsnso = np.tile(np.concatenate((np.zeros(3, dtype=f32), ZSOIL)),
                    (count, 1))
    stc = np.tile(np.concatenate((np.zeros(3, dtype=f32),
                                  np.array([262.0, 262.5, 263.0, 263.0],
                                           dtype=f32))), (count, 1))
    for i in range(count):
        layers = int(rng.integers(0, 4))       # 0..3 snow layers
        if layers == 0:
            if rng.random() < 0.5:             # shallow snow, no layer
                sneqv[i] = f32(rng.uniform(0.1, 4.0))
                snowh[i] = f32(sneqv[i] / 200.0)
            continue
        isnow[i] = -layers
        dz = rng.uniform(0.03, 0.3, layers).astype(f32)
        ice = (dz * rng.uniform(80.0, 300.0, layers)).astype(f32)
        liq = (ice * rng.uniform(0.0, 0.08, layers)).astype(f32)
        snice[i, 3 - layers:] = ice
        snliq[i, 3 - layers:] = liq
        depth = f32(0.0)
        for k in range(layers):
            slot = 3 - layers + k
            depth = f32(depth + dz[k])
            zsnso[i, slot] = -depth
            stc[i, slot] = f32(rng.uniform(258.0, 273.1))
        # shift soil interfaces below the pack
        zsnso[i, 3:] = ZSOIL - depth
        snowh[i] = depth
        sneqv[i] = f32(snice[i].sum() + snliq[i].sum())
    # ficeold consistent with the stack
    ficeold = np.zeros((count, 3), dtype=np.float32)
    live = np.arange(3)[None, :] >= (isnow[:, None] + 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(snice, snice + snliq, out=ficeold, where=live)
    inputs.update({
        "isnow": isnow, "sneqv": sneqv, "snowh": snowh,
        "snice": snice, "snliq": snliq, "zsnso": zsnso, "stc": stc,
        "smc": np.ones((count, 4), dtype=np.float32),
        "sh2o": np.zeros((count, 4), dtype=np.float32),
        "ficeold": ficeold,
    })
    return inputs, count


def _chain(inputs, outputs):
    """Feed step outputs back as the next step's state."""
    nxt = dict(inputs)
    for name in ("tg", "tauss", "qsfc", "qsnow", "sneqvo", "albold",
                 "cm", "ch", "sneqv", "snowh", "smc", "sh2o", "stc",
                 "zsnso", "snice", "snliq", "isnow"):
        nxt[name] = outputs[name]
    # ficeold for the next step, from the advanced stack
    live = np.arange(3)[None, :] >= (outputs["isnow"][:, None] + 3)
    ficeold = np.zeros_like(outputs["snice"])
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(outputs["snice"], outputs["snice"] + outputs["snliq"],
                  out=ficeold, where=live)
    nxt["ficeold"] = ficeold
    return nxt


def test_cuda_and_cpython_glacier_columns_are_bitwise_twins_over_12_steps():
    inputs, count = _battery()
    dev_in = inputs
    host_in = inputs
    for step in range(12):
        dev = evaluate_glacier_columns(dev_in, count, DT)
        host = evaluate_glacier_columns_on_host(host_in, count, DT)
        for name in GLACIER_OUTPUT_SCALARS:
            np.testing.assert_array_equal(
                dev[name].view(np.uint32), host[name].view(np.uint32),
                err_msg=f"step {step} field {name}")
        for name, _w in GLACIER_OUTPUT_ARRAYS:
            np.testing.assert_array_equal(
                dev[name].view(np.uint32), host[name].view(np.uint32),
                err_msg=f"step {step} field {name}")
        np.testing.assert_array_equal(dev["isnow"], host["isnow"],
                                      err_msg=f"step {step} isnow")
        dev_in = _chain(dev_in, dev)
        host_in = _chain(host_in, host)
    # the battery exercised layered packs, not just bare ice
    assert (host["isnow"] < 0).any()
    assert (host["sneqv"] > 0).any()


# ---------------------------------------------------------------------------
# the seam: glacier columns through the full phase-1 chain
# ---------------------------------------------------------------------------

def _profile(seed=0):
    rng = np.random.default_rng(seed)
    z_iface = np.linspace(0.0, 16000.0, NZ + 1)
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    p_iface = 101325.0 * np.exp(-z_iface / 8000.0)
    p_mid = 101325.0 * np.exp(-z_mid / 8000.0)
    theta = 300.0 + 25.0 * z_mid / 16000.0
    exner = (p_mid / 1.0e5) ** (287.0 / 1004.0)
    temp = theta * exner
    rho_dry = p_mid / (287.0 * temp)
    qv = 0.014 * np.exp(-z_mid / 3000.0)
    qc = np.where(z_mid < 4000.0, 8.0e-4, 0.0)

    def cols(profile, jitter=1.0e-3):
        base = np.repeat(profile[:, None], NCOL, axis=1)
        base *= 1.0 + jitter * rng.standard_normal(base.shape)
        return np.ascontiguousarray(base, dtype=np.float32)

    fields = {
        "u": cols(np.full(NZ, 5.0)),
        "v": cols(np.full(NZ, -3.0)),
        "theta": cols(theta),
        "pressure": cols(p_mid, jitter=0.0),
        "pressure_interface": cols(p_iface, jitter=0.0),
        "z_interface": np.ascontiguousarray(
            np.repeat(z_iface[:, None], NCOL, axis=1), dtype=np.float32),
        "w": cols(0.05 * np.sin(np.linspace(0, np.pi, NZ + 1)),
                  jitter=0.0),
        "rho_dry": cols(rho_dry, jitter=0.0),
        "qv": cols(qv),
        "qc": cols(qc),
        "qr": cols(np.full(NZ, 1.0e-5)),
        "qi": np.zeros((NZ, NCOL), dtype=np.float32),
        "qs": np.zeros((NZ, NCOL), dtype=np.float32),
        "qg": np.zeros((NZ, NCOL), dtype=np.float32),
    }
    return {name: cp.asarray(value) for name, value in fields.items()}


#: The eight-column surface: [0] land grass, [1] open water, [2] the
#: collaborator glacier signature, [3] the native-XLAND contract column,
#: [4] fractional sea ice at/above 0.02, [5] glacier under snow,
#: [6]-[7] land grass.
SURFACE = dict(
    landmask=np.array([1, 0, 1, 0, 0, 1, 1, 1], dtype=np.float32),
    xland=np.array([1, 2, 1, 1, 1, 1, 1, 1], dtype=np.float32),
    ivgtyp=np.array([10, 17, ISICE, ISICE, ISICE, ISICE, 10, 10],
                    dtype=np.int32),
    xice=np.array([0.0, 0.0, 0.0, 0.01, 0.05, 0.0, 0.0, 0.0],
                  dtype=np.float32),
    snow=np.array([0, 0, 0, 0, 0, 100.0, 0, 0], dtype=np.float32),
    snow_depth=np.array([0, 0, 0, 0, 0, 0.5, 0, 0], dtype=np.float32),
    tsk=np.array([300, 300, 263, 265, 265, 262, 300, 300],
                 dtype=np.float32),
    tmn=np.array([285, 285, 262, 264, 264, 261, 285, 285],
                 dtype=np.float32),
)


def _seam(**overrides):
    kwargs = dict(
        n_levels=NZ, n_columns=NCOL, dt=DT,
        radiation_seconds=600.0, surface_pbl_seconds=120.0,
        cumulus_seconds=None, cumulus_scheme=None,
        start_time=START,
        latitude_deg=np.full(NCOL, 62.0),
        longitude_deg=np.full(NCOL, -45.0),
        terrain_height_m=np.zeros(NCOL),
        z_interface_nominal_m=np.linspace(0.0, 16000.0, NZ + 1),
        p_top_pa=float(101325.0 * np.exp(-2.0)), dx_m=15000.0,
        xice_threshold=0.02, **SURFACE)
    kwargs.update(overrides)
    return mcb.run_mpas_column_batch(**kwargs)


def _phase1(seam, fields):
    return seam.run_phase1(
        dt=DT, u=fields["u"], v=fields["v"], theta=fields["theta"],
        pressure=fields["pressure"],
        pressure_interface=fields["pressure_interface"],
        z_interface=fields["z_interface"], w=fields["w"],
        rho_dry=fields["rho_dry"], qv=fields["qv"], qc=fields["qc"],
        qr=fields["qr"], qi=fields["qi"], qs=fields["qs"],
        qg=fields["qg"])


def _phase2(seam, fields):
    return seam.run_phase2(
        theta=fields["theta"], qv=fields["qv"], qc=fields["qc"],
        qr=fields["qr"], qi=fields["qi"], qs=fields["qs"],
        qg=fields["qg"], pressure=fields["pressure"],
        rho_dry=fields["rho_dry"], z_interface=fields["z_interface"])


def test_the_seam_admits_glacier_columns_and_names_every_class(monkeypatch):
    from test_noahmp_runtime import _host_layout_route
    _host_layout_route(monkeypatch)
    seam = _seam()
    receipt = seam.surface_classification
    assert receipt["xland_source"] == "native"
    assert receipt["xice_threshold"] == 0.02
    assert receipt["glacier_columns"] == 3      # columns 2, 3, 5
    assert receipt["sea_ice_columns"] == 1      # column 4
    assert receipt["open_water_columns"] == 1   # column 1
    assert receipt["sflx_land_columns"] == 3    # columns 0, 6, 7
    # NATIVE WINS VERBATIM: the field carries the caller's bytes
    np.testing.assert_array_equal(
        cp.asnumpy(seam._driver.fields["xland"]).reshape(-1),
        SURFACE["xland"])

    fields = _profile()
    before = {name: value.copy() for name, value in fields.items()}
    out = _phase1(seam, fields)
    # read-only guarantee, glacier columns included
    for name, value in before.items():
        assert bool((fields[name] == value).all()), name
    for name in ("du", "dv", "dtheta", "dqv"):
        assert bool(cp.isfinite(getattr(out, name)).all()), name
    census = seam.last_noahmp_census
    assert census["glacier"] == 3
    assert census["land"] == 3
    assert census["sea_ice"] == 1
    assert census["water"] == 1
    assert "noahmp-glacier/cuda" in census["glacier_path"]
    # the glacier arm's own write-back reached the glacier columns
    z0 = cp.asnumpy(seam._driver.fields["z0"]).reshape(-1)
    assert z0[2] == np.float32(0.002) and z0[5] == np.float32(0.002)
    snowc = cp.asnumpy(seam._driver.fields["snowc"]).reshape(-1)
    assert snowc[2] == 1.0 and snowc[3] == 1.0 and snowc[5] == 1.0
    tsk = cp.asnumpy(seam._driver.fields["tsk"]).reshape(-1)
    assert np.isfinite(tsk).all()
    _phase2(seam, fields)


def test_the_native_xland_contract_column_runs_ice_never_water():
    """Column 3: landmask=0, native xland=1, IVGTYP=15, xice=0.01.

    Under the native classification it dispatches to NOAHMP_GLACIER (the
    ice surface).  The refutation arm drops the native xland: the
    derivation from landmask=0 classifies the same column open water --
    no surface at all -- which is exactly the misroute the law forbids.
    Red when a rewrite lets the derivation override the native input.
    """
    seam = _seam()
    fields = _profile()
    _phase1(seam, fields)
    z0 = cp.asnumpy(seam._driver.fields["z0"]).reshape(-1)
    assert z0[3] == np.float32(0.002)           # glacier write-back ran
    assert seam.last_noahmp_census["glacier"] == 3

    derived = _seam(xland=None)                 # the fallback derivation
    assert derived.surface_classification["xland_source"] == \
        "derived-from-landmask"
    # landmask=0 -> xland=2 -> open water: one more water column, one
    # fewer glacier column.  The 0.02 threshold alone does NOT fix this.
    assert derived.surface_classification["glacier_columns"] == 2
    assert derived.surface_classification["open_water_columns"] == 2
    fields2 = _profile()
    _phase1(derived, fields2)
    z0d = cp.asnumpy(derived._driver.fields["z0"]).reshape(-1)
    assert z0d[3] != np.float32(0.002)          # no glacier write-back
    assert derived.last_noahmp_census["glacier"] == 2


def test_multi_hour_cadence_with_glacier_columns_stays_finite():
    seam = _seam()
    fields = _profile()
    for step in range(30):                      # 1 h at dt=120 s
        out = _phase1(seam, fields)
        assert bool(cp.isfinite(out.dtheta).all()), f"step {step}"
        _phase2(seam, fields)
    tracked = ("tsk", "tgxy", "snow", "snowh", "tslb", "tsnoxy",
               "smois", "sh2o", "snicexy", "snliqxy", "zsnsoxy")
    for name in tracked:
        assert bool(cp.isfinite(seam._driver.fields[name]).all()), name
    assert seam.last_noahmp_census["glacier"] == 3


def test_restart_round_trip_continues_bit_identically_with_glacier():
    seam = _seam()
    fields = _profile()
    for _ in range(4):
        _phase1(seam, fields)
        _phase2(seam, fields)
    payload = seam.export_state()

    fresh = _seam()
    fresh.restore_state(payload)
    fields_a = _profile(seed=3)
    fields_b = {name: value.copy() for name, value in fields_a.items()}
    for step in range(4):
        out_a = _phase1(seam, fields_a)
        out_b = _phase1(fresh, fields_b)
        for name in ("du", "dv", "dtheta", "dqv", "dqc", "h_diabatic"):
            np.testing.assert_array_equal(
                cp.asnumpy(getattr(out_a, name)),
                cp.asnumpy(getattr(out_b, name)),
                err_msg=f"step {step} {name}")
        _phase2(seam, fields_a)
        _phase2(fresh, fields_b)
    for name in ("tgxy", "snow", "snowh", "tsnoxy", "snicexy",
                 "snliqxy", "zsnsoxy", "tslb", "qsnowxy", "taussxy"):
        np.testing.assert_array_equal(
            cp.asnumpy(seam._driver.fields[name]),
            cp.asnumpy(fresh._driver.fields[name]), err_msg=name)


def test_a_mismatched_threshold_refuses_the_restart_payload():
    seam = _seam()
    fields = _profile()
    _phase1(seam, fields)
    _phase2(seam, fields)
    payload = seam.export_state()
    other = _seam(xice_threshold=0.5)
    with pytest.raises(ValueError, match="identity"):
        other.restore_state(payload)


def test_disabling_the_glacier_path_refuses_instead_of_running_sflx():
    from woof.core.noahmp_runtime import NoahmpGlacierColumnError

    seam = _seam()
    seam._driver.noahmp_params.glacier_path = False
    fields = _profile()
    with pytest.raises(NoahmpGlacierColumnError, match="glacier_path"):
        _phase1(seam, fields)


def test_glacier_writeback_reuses_indices_and_preserves_unmasked_words(monkeypatch):
    """Prevent 150 redundant index allocations/uploads per glacier step.

    (Host-classified layout, pinned: the resident layout runs the resident
    glacier driver, test_noahmp_glacier_slab.)

    Instrument only write-back, after the real evaluator returns. CuPy still
    allocates combined advanced-index arrays; the old host-index path adds
    two uploads for each of 76 accesses (228 index-sized allocations total).
    Reusing the two device indices leaves 78 on the measured CuPy backend.
    The upper bound allows backend scratch changes, but rejects that defect.
    """
    from woof.core import noahmp_runtime as runtime
    from test_noahmp_runtime import _host_layout_route
    _host_layout_route(monkeypatch)
    original = runtime._glacier_lsm_step
    observed = []

    def measured(fields, atmosphere, mask, **kwargs):
        host_mask = cp.asnumpy(mask).astype(bool)
        count = int(host_mask.sum())
        before = {k: cp.asnumpy(v)[..., ~host_mask].copy()
                  for k, v in fields.items()
                  if hasattr(v, 'shape') and v.shape[-2:] == host_mask.shape}
        active = False
        allocations = 0
        evaluator = kwargs['evaluator']

        def evaluate(*args, **kw):
            nonlocal active
            result = evaluator(*args, **kw)
            active = True
            return result

        class Hook(cp.cuda.memory_hook.MemoryHook):
            def malloc_preprocess(self, device_id, size, mem_size):
                nonlocal allocations
                if active and size == count * np.dtype(np.int64).itemsize:
                    allocations += 1

        kwargs['evaluator'] = evaluate
        with Hook():
            original(fields, atmosphere, mask, **kwargs)
        for name, value in before.items():
            after = cp.asnumpy(fields[name])[..., ~host_mask]
            np.testing.assert_array_equal(np.ascontiguousarray(after).view(np.uint8),
                                          np.ascontiguousarray(value).view(np.uint8),
                                          err_msg=name)
        observed.append((count, allocations))

    monkeypatch.setattr(runtime, '_glacier_lsm_step', measured)
    seam = _seam()
    fields = _profile()
    _phase1(seam, fields)
    assert observed and all(count == 3 for count, _ in observed)
    assert all(2 <= allocations <= 90 for _, allocations in observed), observed


@pytest.mark.parametrize('ncells,count,gathered', [
    (39, 17, False), (86337, 4096, False), (86338, 4096, True),
])
def test_glacier_sparse_input_copy_budget(monkeypatch, ncells, count, gathered):
    """Large sparse domains must not copy every non-glacier input to host.

    Stop at the evaluator boundary to measure marshalling alone. The one-cell
    cutoff pair also prevents applying the slower gather path to small calls.
    Full evolution and bitwise outputs are covered by the CUDA battery.
    """
    from woof.core import noahmp_runtime as runtime
    values = np.linspace(1., 2., ncells, dtype=np.float32).reshape(1, -1)
    device = cp.asarray(values)
    scalar_names = ('rainbl tmn swdown glw qsnowxy sneqvoxy alboldxy cmxy '
                    'chxy snow snowh tgxy taussxy qsfc').split()
    fields = {name: device for name in scalar_names}
    for name, levels in dict(smois=4, sh2o=4, tslb=4, tsnoxy=3,
                             snicexy=3, snliqxy=3, zsnsoxy=7).items():
        fields[name] = cp.broadcast_to(device, (levels, 1, ncells))
    fields['isnowxy'] = cp.zeros((1, ncells), dtype=cp.int32)
    atmosphere = {name: device[None] for name in
                  ('temperature', 'qv', 'u', 'v', 'dz')}
    atmosphere['p_interface'] = cp.broadcast_to(device, (2, 1, ncells))
    mask = np.zeros((1, ncells), dtype=bool)
    mask[0, np.linspace(0, ncells - 1, count, dtype=np.int64)] = True
    device_mask = cp.asarray(mask)
    selected = values[mask]
    sizes = []
    asnumpy = cp.asnumpy

    def measured(value, *args, **kwargs):
        sizes.append(value.nbytes)
        return asnumpy(value, *args, **kwargs)

    class InputsChecked(Exception):
        pass

    def evaluate(inputs, n, dt):
        assert n == count and dt == 120.
        for name in ('cosz', 'sfctmp', 'uu', 'vv', 'soldn', 'lwdn',
                     'qsnow', 'sneqvo', 'albold', 'cm', 'ch', 'sneqv',
                     'snowh', 'tg', 'tauss', 'qsfc'):
            np.testing.assert_array_equal(inputs[name].view(np.uint32),
                                          selected.view(np.uint32))
        np.testing.assert_array_equal(inputs['smc'].view(np.uint32),
                                      np.broadcast_to(selected[:, None],
                                                      (count, 4)).view(np.uint32))
        expected = ncells + 51 * 4 * (count if gathered else ncells)
        assert sum(sizes) == expected, (sum(sizes), expected)
        raise InputsChecked

    monkeypatch.setattr(cp, 'asnumpy', measured)
    with pytest.raises(InputsChecked):
        runtime._glacier_lsm_step(fields, atmosphere, device_mask, params=None,
                                 coszen=device, dt=120.,
                                 dzs=np.array([.1, .3, .6, 1.], dtype=np.float32),
                                 evaluator=evaluate)
