"""Dual-run uint32 gates for device legacy prep, with no tolerance path."""

import os
from itertools import product
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof.core import rrtmg_legacy_device as dev
from woof.core import rrtmg_legacy_prep as ref
from woof.core import rrtmg_mcica as mcica

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def host_oracle(monkeypatch):
    # The reference has an optional device shortcut for O3/VARINT. Force
    # the NumPy oracle for every comparison, without editing the reference.
    monkeypatch.setattr(ref, "_PERFWAVE_DEVICE_XP", None)


def device_kwargs(kw):
    return {k: device_kwargs(v) if isinstance(v, dict) else
            cp.asarray(v) if isinstance(v, np.ndarray) and v.ndim else v
            for k, v in kw.items()}


def assert_bits(got, want, path="result"):
    if isinstance(want, dict):
        assert set(got) == set(want), f"{path} key set"
        for k in want:
            assert_bits(got[k], want[k], path+"/"+k)
    elif isinstance(want, np.ndarray):
        assert isinstance(got, cp.ndarray), path+" must remain on device"
        assert got.dtype == want.dtype == np.float32, path+" dtype"
        assert got.shape == want.shape, path+" shape"
        a, b = cp.asnumpy(got).view(np.uint32), want.view(np.uint32)
        if not np.array_equal(a, b):
            idx = tuple(np.argwhere(a != b)[0])
            pytest.fail(f"{path}: {np.count_nonzero(a != b)} uint32 mismatches; "
                        f"first {idx}: got {a[idx]:08x} want {b[idx]:08x}")
    elif isinstance(want, (int, np.integer)):
        assert type(got) is int and got == want, path
    else:
        assert np.asarray(got, np.float32).view(np.uint32) == np.asarray(want, np.float32).view(np.uint32), path


def dual(kw, sw):
    host = ref.swrad_prep_batch if sw else ref.lwrad_prep_batch
    device = dev.swrad_prep_batch_device if sw else dev.lwrad_prep_batch_device
    expected = host(**kw)
    names = ("cldfmcl", "taucmcl", "ciwpmcl", "clwpmcl", "cswpmcl")
    if sw:
        names += ("ssacmcl", "asmcmcl", "fsfcmcl")
    for name in names:
        expected[name] = expected[name].transpose(1, 2, 0)
    dkw = device_kwargs(kw)
    for run in range(2):
        actual = device(**dkw)
        assert_bits(actual, expected, f"{'SW' if sw else 'LW'} run{run+1}")
        del actual


def synthetic(ncol, nz=59, seed=7103):
    rng = np.random.default_rng(seed)
    shape = (ncol, nz)
    interfaces = np.linspace(101300, 5000, nz+1, dtype=np.float32)
    pw = np.broadcast_to(interfaces, (ncol, nz+1)).copy()
    # A shallow top exercises the per-column Cavallo positivity clamp.
    pw[::3, -1] = np.float32(100)
    p = ((pw[:, :-1] + pw[:, 1:])*np.float32(.5)).astype(np.float32)
    kw = dict(p3d=p, p8w=pw,
              t3d=rng.uniform(180, 300, shape).astype(np.float32),
              t8w=rng.uniform(180, 300, (ncol, nz+1)).astype(np.float32),
              dz8w=rng.uniform(50, 1000, shape).astype(np.float32))
    for name in ("qv3d", "qc3d", "qr3d", "qi3d", "qs3d", "qg3d"):
        kw[name] = rng.uniform(0, .002, shape).astype(np.float32)
        kw[name].ravel()[::11] = np.float32(1.e-39)
        kw[name].ravel()[1::11] = np.float32(-0.0)
        kw[name].ravel()[2::11] = np.float32(-1.e-39)
    kw["cldfra3d"] = rng.uniform(0, 1, shape).astype(np.float32)
    kw["cldfra3d"].ravel()[::7] = np.float32(0)
    kw["cldfra3d"].ravel()[1::7] = np.float32(.01)
    kw["o33d"] = rng.uniform(1.e-8, 1.e-6, shape).astype(np.float32)
    for name, lo, hi in (("re_cloud", 2.5e-6, 30.e-6),
                         ("re_ice", 5.e-6, 150.e-6),
                         ("re_snow", 10.e-6, 200.e-6)):
        kw[name] = rng.uniform(lo, hi, shape).astype(np.float32)
        kw[name].ravel()[::5] = np.float32(lo)
        kw[name].ravel()[1::5] = np.float32(0)
    kw["re_snow"].ravel()[::5] = np.float32(130.e-6)
    kw["t3d"].ravel()[:min(8,ncol*nz)] = np.array(
        [179, 180, 200, 273, 273.15, 273.16, 274, 250], np.float32)[:min(8,ncol*nz)]
    for name, value in (("tsk", 290), ("xice", .2), ("snow", 30), ("xlat", 35),
                        ("xcoszen", .6), ("solcon", 1361), ("obscur", .1),
                        ("emiss", .98), ("albedo", .2)):
        kw[name] = np.full(ncol, value, np.float32)
    kw["xland"] = np.resize(np.array([1.5, 1, 2], np.float32), ncol)
    kw.update(icloud=1, warm_rain=False, cldovrlp=2, idcor=0, o3input=0,
              has_reqc=1, has_reqi=1, has_reqs=1, yr=2026, julian=183.75)
    return kw


def side_kwargs(kw, sw):
    kw = kw.copy()
    if sw:
        kw.pop("emiss")
    else:
        for n in ("albedo", "xcoszen", "solcon", "obscur"):
            kw.pop(n)
        kw["nlayers"] = ref.compute_lw_nlayers(kw["p3d"].shape[1]+1, 5000.0)
    return kw


VARIANTS = [dict(o3input=0), dict(o3input=2, warm_rain=True),
            dict(o3input=0, f_qi=False, f_qs=False, f_qg=False),
            dict(o3input=2, f_qi=False, f_qs=False, f_qg=False, warm_rain=True),
            dict(o3input=2, f_qc=False, f_qr=False, f_qg=False),
            dict(o3input=0, icloud=0)]


@pytest.mark.parametrize("ncol", [5000])
@pytest.mark.parametrize("radii", [(1,1,1), (0,0,0), (1,1,0)])
@pytest.mark.parametrize("variant", VARIANTS)
def test_synthetic_dual(ncol, radii, variant):
    kw = synthetic(ncol)
    kw.update(zip(("has_reqc", "has_reqi", "has_reqs"), radii))
    kw.update(variant)
    for sw in (False, True):
        dual(side_kwargs(kw, sw), sw)


@pytest.mark.parametrize("radii", list(product((0,1), repeat=3)))
def test_all_radii_routes(radii):
    kw = synthetic(7)
    kw.update(zip(("has_reqc", "has_reqi", "has_reqs"), radii))
    for sw in (False, True):
        dual(side_kwargs(kw, sw), sw)


def test_sw_fixture_day_groups_dual():
    import test_rrtmg_legacy_prep as deck
    fixtures = deck._sw_fixtures()
    for columns in deck._sw_day_groups().values():
        kw = deck._sw_batch_kwargs(fixtures, columns)
        dual(kw, True)


#: Real forecast decks: npz files of the exact prep keyword arguments one
#: product-suite radiation chunk received.  They are not committed (tens of
#: MB); point WOOF_RRTMG_LEGACY_DECKS at a folder of them.
DECKS = sorted(Path(os.environ.get("WOOF_RRTMG_LEGACY_DECKS",
                                   str(ROOT / "decks"))).glob("*.npz"))


@pytest.mark.parametrize("path", DECKS or [None], ids=lambda p: p.stem if p else "absent")
def test_forecast_deck_dual(path):
    if path is None:
        pytest.skip("real forecast decks not supplied")
    with np.load(path, allow_pickle=False) as data:
        kw = {k: v.item() if v.ndim == 0 else v for k, v in data.items()}
    dual(kw, path.name.startswith("sw_"))


def test_preflight_and_numpy_minmax():
    # Verify both argument orders, NaNs, and ties in float32 arrays on-node.
    a = np.array([0., -0., np.nan, 1., 1.e-39], np.float32)
    b = np.array([-0., 0., 1., np.nan, 0.], np.float32)
    for operation, cmp in ((np.maximum, np.greater), (np.minimum, np.less)):
        actual = operation(a,b)
        expected = np.where(cmp(a,b), a,b)
        expected[np.isnan(a)] = a[np.isnan(a)]
        expected[np.isnan(b)] = b[np.isnan(b)]
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    for _ in range(2):
        dev.gpu_preflight(force=True)
        frames = dev.gpu_local_frame_bytes()
        print("prep local frames:", frames)
        assert max(frames.values()) <= 128


@pytest.mark.parametrize("p_top", [200.0, 1000.0, 5000.0, 10000.0])
def test_buffer_sizes_and_clamp(p_top):
    kw = side_kwargs(synthetic(7), False)
    kw["nlayers"] = ref.compute_lw_nlayers(60, p_top)
    dual(kw, False)


@pytest.mark.parametrize("warm,o3,fqi,fqs,fqg", list(product((False, True), (0,2), (False,True), (False,True), (False,True))))
def test_species_flag_cross_product(warm, o3, fqi, fqs, fqg):
    # Exhaust all species/warm-rain/ozone interactions separately from the
    # wide matrix, which stresses the same paths at every required width.
    for radii in ((1,1,1), (0,0,0), (1,1,0)):
        kw = synthetic(7)
        kw.update(zip(("has_reqc", "has_reqi", "has_reqs"), radii))
        kw.update(warm_rain=warm, o3input=o3, f_qi=fqi, f_qs=fqs, f_qg=fqg)
        for sw in (False, True):
            dual(side_kwargs(kw, sw), sw)


def test_missing_optional_arrays_and_scalar_surfaces():
    kw = synthetic(7)
    kw.update(has_reqc=0, has_reqi=0, has_reqs=0, f_qi=False,
              f_qs=False, f_qg=False)
    for name in ("qc3d", "qr3d", "qi3d", "qs3d", "qg3d", "cldfra3d",
                 "re_cloud", "re_ice", "re_snow", "o33d"):
        kw[name] = None
    for name in ("tsk", "emiss", "albedo", "snow", "xice", "xland", "xlat", "solcon", "xcoszen", "obscur"):
        kw[name] = float(kw[name][0])
    for sw in (False, True):
        dual(side_kwargs(kw, sw), sw)


def test_override_and_noncontiguous_inputs():
    kw = synthetic(7)
    for name, value in kw.items():
        if isinstance(value, np.ndarray):
            kw[name] = value[::-1]
    kw["trace_gas_overrides"] = dict(co2=420.e-6, ch4=1800.e-9)
    for sw in (False, True):
        dual(side_kwargs(kw, sw), sw)


def test_lw_prep_does_not_download_or_synchronize(monkeypatch):
    dev.gpu_preflight()
    host_kw = side_kwargs(synthetic(7), False)
    expected = ref.lwrad_prep_batch(**host_kw)
    captured = []
    def generator(*args, layout):
        assert layout == "column"
        n, nl = args[1:3]
        # Stand-in for the independently tested McICA stage: expose its
        # inputs without adding its permitted preflight/readback/sync.
        captured.append(args)
        slab = cp.zeros((n, nl, mcica.NGPTLW), np.float32)
        return dict(cldfmcl=slab, taucmcl=slab, ciwpmcl=slab,
                    clwpmcl=slab, cswpmcl=slab,
                    reicmcl=args[11], relqmcl=args[12], resnmcl=args[13])
    def forbidden(*args, **kwargs):
        raise AssertionError("prep attempted a host download or synchronization")
    monkeypatch.setattr(cp, "asnumpy", forbidden)
    monkeypatch.setattr(cp.cuda.runtime, "deviceSynchronize", forbidden)
    monkeypatch.setattr(ref, "_host_f32", forbidden)
    monkeypatch.setattr(ref, "generate_lw_subcolumns", forbidden)
    kw = device_kwargs(host_kw)
    kw["subcolumn_generator"] = generator
    results = []
    for _ in range(2):
        result = dev.lwrad_prep_batch_device(**kw)
        assert isinstance(result["plev"], cp.ndarray)
        results.append(result)
    assert len(captured) == 2
    monkeypatch.undo()
    # Downloads are permitted for the test comparison after prep returned.
    keys = ("plev", "play", "tlev", "tlay", "hgt", "pdel", "o31d", "cldfrac")
    for result in results:
        assert_bits({k:result[k] for k in keys}, {k:expected[k] for k in keys})


@pytest.mark.parametrize("sw", [False, True])
def test_generator_hook_and_late_resolution(monkeypatch, sw):
    name = "gpu_generate_sw_subcolumns" if sw else "gpu_generate_lw_subcolumns"
    original = getattr(mcica, name)
    calls = []
    def hook(*args, **kw):
        assert all(isinstance(v, cp.ndarray) for v in args[6:-3])
        assert isinstance(args[-1], cp.ndarray)
        calls.append(args[1:3])
        return original(*args, **kw)
    kw = side_kwargs(synthetic(7), sw)
    monkeypatch.setattr(mcica, name, hook)
    dual(kw, sw)
    assert len(calls) == 2
    kw["subcolumn_generator"] = hook
    # NumPy oracle uses its host generator, while device explicitly uses hook.
    expected = (ref.swrad_prep_batch if sw else ref.lwrad_prep_batch)(
        **{k:v for k,v in kw.items() if k != "subcolumn_generator"})
    names = ("cldfmcl", "taucmcl", "ciwpmcl", "clwpmcl", "cswpmcl")
    if sw:
        names += ("ssacmcl", "asmcmcl", "fsfcmcl")
    for name in names:
        expected[name] = expected[name].transpose(1, 2, 0)
    for _ in range(2):
        got = (dev.swrad_prep_batch_device if sw else dev.lwrad_prep_batch_device)(**device_kwargs(kw))
        assert_bits(got, expected)
    assert len(calls) == 4


def test_day_and_shape_refusals():
    kw = side_kwargs(synthetic(7), True)
    kw["xcoszen"][3] = np.float32(0)
    for _ in range(2):
        with pytest.raises(ValueError, match="night column"):
            dev.swrad_prep_batch_device(**device_kwargs(kw))
    kw = side_kwargs(synthetic(7), False)
    kw["p8w"] = kw["p8w"][:, :-1]
    for _ in range(2):
        with pytest.raises(ValueError, match="p8w must have shape"):
            dev.lwrad_prep_batch_device(**device_kwargs(kw))


def test_constant_bands_per_call_and_positive_tiny_coszen():
    kw = side_kwargs(synthetic(7), True)
    kw["xcoszen"][0] = np.float32(1.e-39)
    dual(kw, True)
    # The band inputs are built per call: a shape-keyed cache would keep a
    # device slab for every last-chunk width a forecast ever produced.
    first = dev._bands(True, 7, 60)
    second = dev._bands(True, 7, 60)
    assert first[0] is not second[0]
    for bands in (first, second):
        for band, value in zip(bands, (0,1,0,0)):
            assert_bits(band, np.full(band.shape, value, np.float32))
    for band in dev._bands(False, 7, 73):
        assert band.shape == (mcica.NBNDLW, 7, 73)
        assert_bits(band, np.zeros(band.shape, np.float32))

@pytest.mark.parametrize('icloud', [0, 1])
def test_ferrier_hires_wrapper_remaps_total_ice_on_device(icloud):
    kw = synthetic(3)
    kw.update(mp_physics=5, icloud=icloud, f_qi=True, f_qs=False,
              has_reqc=0, has_reqi=0, has_reqs=0)
    for sw in (False, True):
        dual(side_kwargs(kw, sw), sw)


@pytest.mark.parametrize("sw", [False, True])
@pytest.mark.parametrize("snow_radius", [0, 1])
def test_canonical_microns_keep_stock_cloud_prep_bits(sw, snow_radius):
    # Stock metre radii are converted once inside the wrapper. Forecast
    # state already holds that rounded micron result. Its metre roundtrip
    # changes some words, observed at the second whole-model call.
    kw = side_kwargs(synthetic(7), sw)
    kw["has_reqs"] = snow_radius
    host = ref.swrad_prep_batch if sw else ref.lwrad_prep_batch
    device = dev.swrad_prep_batch_device if sw else dev.lwrad_prep_batch_device
    expected = host(**kw)
    names = ("cldfmcl", "taucmcl", "ciwpmcl", "clwpmcl", "cswpmcl")
    if sw:
        names += ("ssacmcl", "asmcmcl", "fsfcmcl")
    for name in names:
        expected[name] = expected[name].transpose(1, 2, 0)
    dkw = device_kwargs(kw)
    changed = 0
    for name in ("re_cloud", "re_ice", "re_snow"):
        microns = (kw[name] * np.float32(1.e6)).astype(np.float32)
        roundtrip = microns * np.float32(1.e-6) * np.float32(1.e6)
        changed += np.count_nonzero(microns.view(np.uint32) != roundtrip.view(np.uint32))
        dkw[name] = cp.asarray(microns)
    assert changed > 0, "fixture must exercise the lost radius bits"
    assert_bits(device(**dkw, radii_units="microns"), expected)


def test_unknown_radius_units_refused():
    with pytest.raises(ValueError, match="radii_units must be meters or microns"):
        dev.lwrad_prep_batch_device(radii_units="unknown")

@pytest.mark.parametrize("selector", [5, 15])
@pytest.mark.parametrize("icloud", [0, 1])
@pytest.mark.parametrize("sw", [False, True])
def test_ferrier_remap_matches_the_reference_outside_icloud(selector, icloud, sw):
    kw = synthetic(7)
    kw.update(mp_physics=selector, icloud=icloud,
              has_reqc=0, has_reqi=0, has_reqs=0,
              f_qc=True, f_qr=True, f_qi=True, f_qs=False, f_qg=False)
    dual(side_kwargs(kw, sw), sw)

@pytest.mark.gpu
def test_ssib_four_albedos_reach_the_device_engine_bitwise():
    kw = side_kwargs(synthetic(5, nz=12), True)
    kw['sf_surface_physics'] = 8
    kw['spectral_albedo'] = {name: np.full(5, np.float32(value), np.float32)
        for name, value in [('asdir',.1),('asdif',.2),('aldir',.3),('aldif',.4)]}
    dual(kw, True)
    actual = dev.swrad_prep_batch_device(**device_kwargs(kw))
    for name, value in kw['spectral_albedo'].items():
        assert_bits(actual[name], value, name)
    for bad in (None, {'asdir': np.zeros(5,np.float32)}):
        broken = device_kwargs(kw); broken['spectral_albedo'] = bad
        with pytest.raises(ValueError,match='four spectral albedos'):
            dev.swrad_prep_batch_device(**broken)
