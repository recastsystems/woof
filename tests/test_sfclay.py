"""WRF v4.6.1 MM5 surface-layer options 91 (classic) and 1 (revised).

The transcription authorities are ``phys/module_sf_sfclay.F``
(``SFCLAY1D``) and ``phys/physics_mmm/sf_sfclayrev.F90``
(``sf_sfclayrev_run``; ``module_sf_sfclayrev.F`` is its WRF wrapper).
The CUDA implementation is one thread per horizontal column/surface point.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import RunConfig, load_config


_BASE_CFG = dict(nx=8, ny=4, nz=6, dx=1000.0, dy=1000.0,
                 ztop=6000.0, dt=5.0, run_seconds=0.0)


def _surface_cases(seed=14, ny=6, nx=9):
    """Plausible randomized surface points with every stability branch.

    Input arrays are deliberately float32: the float64 mirror sees exactly
    the values supplied to the FP32 kernel, isolating arithmetic differences.
    Equal positive qv/qsfc makes virtual-temperature stability depend only on
    the imposed ground/air temperature difference; every third point is
    exactly neutral.
    """
    rng = np.random.default_rng(seed)
    shape = (ny, nx)
    t = rng.uniform(285.0, 305.0, shape)
    p = rng.uniform(88000.0, 102000.0, shape)
    qv = rng.uniform(0.002, 0.016, shape)
    delta = np.resize(np.array([-7.0, 0.0, 7.0]), shape)
    # Make the neutral subset exactly neutral in FP32 too.  Unit Exner and a
    # positive moisture value whose virtual correction rounds to one ensure
    # equality-driven WRF regime 3 is exercised without a tolerance rule.
    neutral = delta == 0.0
    p[neutral] = 100000.0
    qv[neutral] = 1.0e-8
    water = (np.arange(ny * nx).reshape(shape) % 5 == 4) & ~neutral
    znt = rng.uniform(0.015, 0.25, shape)
    znt[water] = 1.0e-4
    xland = np.where(water, 2.0, 1.0)
    data = {
        "u": rng.uniform(2.0, 14.0, shape),
        "v": rng.uniform(-5.0, 5.0, shape),
        "t": t,
        "qv": qv,
        "p": p,
        "dz8w": rng.uniform(25.0, 90.0, shape),
        "psfc": p.copy(),
        "tsk": t + delta,
        "znt": znt,
        "pblh": rng.uniform(300.0, 1400.0, shape),
        "mavail": rng.uniform(0.2, 1.0, shape),
        "xland": xland,
        "qsfc": qv.copy(),
        "ust": rng.uniform(0.12, 0.55, shape),
        # Classic SFCLAY diagnoses unstable z/L from the previous T* (MOL)
        # when old u* >= 0.01; a negative previous value is the normal
        # time-stepping state and ensures its unstable psi-table path runs.
        "mol": np.where(delta > 0.0, -0.2,
                        np.where(delta < 0.0, 0.2, 0.0)),
        "hfx": np.zeros(shape),
        "qfx": np.zeros(shape),
        "lakemask": np.zeros(shape),
    }
    return {name: np.ascontiguousarray(value, dtype=np.float32)
            for name, value in data.items()}


def _run_mirror(data, option):
    from woof.verify.npref import np_sfclay

    return np_sfclay(
        data["u"], data["v"], data["t"], data["qv"], data["p"],
        data["dz8w"], data["psfc"], data["tsk"], data["znt"],
        data["pblh"], data["mavail"], data["xland"], option=option,
        qsfc=data["qsfc"], ust=data["ust"], mol=data["mol"],
        hfx=data["hfx"], qfx=data["qfx"], lakemask=data["lakemask"],
        dx=1000.0,
    )


def test_sfclay_config_default_roundtrip_and_validation(tmp_path):
    assert RunConfig(**_BASE_CFG).sf_sfclay_physics == 0

    def write(value):
        path = tmp_path / f"sfclay-{value}.toml"
        path.write_text(textwrap.dedent(f"""
            [grid]
            nx = 8
            ny = 4
            nz = 6
            dx = 1000.0
            dy = 1000.0
            ztop = 6000.0
            [dynamics]
            dt = 5.0
            sf_sfclay_physics = {value}
            [run]
            run_seconds = 0.0
        """))
        return path

    for option in (0, 1, 91):
        assert load_config(write(option)).sf_sfclay_physics == option
    with pytest.raises(ValueError, match="sf_sfclay_physics"):
        load_config(write(2))


@pytest.mark.parametrize("option", [91, 1])
def test_mirror_exercises_stable_neutral_unstable_and_mo_consistency(option):
    from woof.core import constants as c

    data = _surface_cases(ny=1, nx=3)
    old_ust = data["ust"].astype(np.float64)
    out = _run_mirror(data, option)

    # Ground temperatures are air-7 K, air, air+7 K respectively.
    assert out["br"][0, 0] > 0.0
    assert out["br"][0, 1] == 0.0
    assert out["br"][0, 2] < 0.0
    assert int(out["regime"][0, 0]) in ((1, 2) if option == 91 else (1,))
    assert int(out["regime"][0, 1]) == 3
    assert int(out["regime"][0, 2]) == 4

    # WRF's integrated stability corrections: stable <= 0, neutral == 0,
    # unstable >= 0.  Both files constrain z/L to the tabulated/solved range.
    assert out["psim"][0, 0] <= 0.0 and out["psih"][0, 0] <= 0.0
    assert out["psim"][0, 1] == 0.0 and out["psih"][0, 1] == 0.0
    assert out["psim"][0, 2] >= 0.0 and out["psih"][0, 2] >= 0.0
    assert np.max(np.abs(out["zol"])) <= 10.0
    za = 0.5 * data["dz8w"].astype(np.float64)
    consistent = np.ones(out["zol"].shape, dtype=bool)
    if option == 91:
        # WRF's classic regime 1 updates RMOL but leaves inout ZOL untouched.
        consistent = out["regime"] != 1.0
    np.testing.assert_allclose((out["rmol"] * za)[consistent],
                               out["zol"][consistent], rtol=2e-14,
                               atol=2e-14)

    # SFCLAY damps u* by averaging new similarity-theory u* with its old
    # value.  Undo that average and recover k U / (ln(z/z0)-psi_m).
    diagnosed = 2.0 * out["ust"] - old_ust
    np.testing.assert_allclose(diagnosed, 0.4 * out["wspd"] / out["fm"],
                               rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(out["mol"],
                               0.4 * (out["theta_air"]
                                      - out["theta_ground"]) / out["fh"],
                               rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(out["lh"], c.XLV * out["qfx"], rtol=2e-14)
    np.testing.assert_allclose(out["hfx"],
                               out["flhc"] * (out["theta_ground"]
                                              - out["theta_air"]),
                               rtol=2e-14, atol=2e-14)


def test_revised_and_classic_are_distinct_transcriptions():
    data = _surface_cases(ny=2, nx=6)
    classic = _run_mirror(data, 91)
    revised = _run_mirror(data, 1)
    unstable = classic["br"] < 0.0
    stable = classic["br"] > 0.0
    assert np.max(np.abs(classic["psim"][unstable]
                         - revised["psim"][unstable])) > 1.0e-3
    assert np.max(np.abs(classic["psih"][stable]
                         - revised["psih"][stable])) > 1.0e-3


@pytest.mark.parametrize("option", [91, 1])
def test_one_fp32_ulp_temperature_difference_is_not_neutral(option):
    """WRF enters regime 3 only when its directly computed BR is zero."""
    from woof.verify.npref import np_sfclay

    shape = (1, 1)
    full = lambda value: np.full(shape, value, np.float32)
    tsk = np.nextafter(np.float32(300.0), np.float32(np.inf))
    out = np_sfclay(
        full(5.0), full(0.0), full(300.0), full(0.01), full(100000.0),
        full(40.0), full(100000.0), full(tsk), full(0.1), full(800.0),
        full(1.0), full(1.0), option=option, qsfc=full(0.01),
        ust=full(0.2), mol=full(0.0), hfx=full(0.0), qfx=full(0.0),
    )

    assert float(tsk - np.float32(300.0)) == 3.0517578125e-05
    assert out["theta_ground"][0, 0] > out["theta_air"][0, 0]
    assert out["br"][0, 0] < 0.0
    assert int(out["regime"][0, 0]) == 4


def test_classic_strong_stable_preserves_incoming_zol_and_updates_rmol():
    """Option 91 regime 1 updates RMOL but leaves WRF's inout ZOL alone."""
    from woof.core import constants as c
    from woof.verify.npref import np_sfclay

    shape = (1, 1)
    full = lambda value: np.full(shape, value, np.float32)
    incoming_zol = 7.25
    old_ust = 0.2
    old_mol = 0.4
    za = 40.0
    out = np_sfclay(
        full(0.1), full(0.0), full(300.0), full(0.01), full(100000.0),
        full(2.0 * za), full(100000.0), full(290.0), full(0.1),
        full(800.0), full(1.0), full(1.0), option=91, qsfc=full(0.01),
        ust=full(old_ust), mol=full(old_mol), zol=full(incoming_zol),
        hfx=full(0.0), qfx=full(0.0),
    )

    assert out["br"][0, 0] >= 0.2
    assert int(out["regime"][0, 0]) == 1
    assert out["zol"][0, 0] == incoming_zol
    za_over_l = min(0.4 * c.G / 300.0 * za * old_mol / old_ust ** 2,
                    9.999)
    assert out["rmol"][0, 0] == pytest.approx(za_over_l / za)
    assert out["rmol"][0, 0] * za != out["zol"][0, 0]


@pytest.mark.parametrize("option", [91, 1])
def test_coare_style_marine_flux_sanity(option):
    """Canonical 10 m/s, warm/moist sea case stays in published bulk-flux
    scales (COARE-style sanity, not a claim of bitwise COARE equivalence).

    The broad physical gates cover the usual order of magnitude: drag near
    1e-3, u* a few tenths m/s, sensible heat tens W/m2 or less, and latent
    heat tens-to-hundreds W/m2.
    """
    shape = (1, 1)
    full = lambda value: np.full(shape, value, np.float32)
    data = {
        "u": full(10.0), "v": full(0.0), "t": full(300.0),
        "qv": full(0.018), "p": full(101325.0), "dz8w": full(40.0),
        "psfc": full(101325.0), "tsk": full(301.0), "znt": full(1.0e-4),
        "pblh": full(800.0), "mavail": full(1.0), "xland": full(2.0),
        "qsfc": full(0.0), "ust": full(0.35), "mol": full(0.0),
        "hfx": full(0.0), "qfx": full(0.0), "lakemask": full(0.0),
    }
    out = _run_mirror(data, option)
    scalar = lambda name: float(out[name][0, 0])
    assert 0.2 < scalar("ust") < 0.7
    assert 5.0e-4 < scalar("cd") < 3.5e-3
    assert 0.0 < scalar("hfx") < 100.0
    assert 1.0e-5 < scalar("qfx") < 2.0e-4
    assert 25.0 < scalar("lh") < 500.0
    assert 6.0 < scalar("u10") < 12.0


@pytest.mark.parametrize("option", [91, 1])
def test_ustm_relaxes_on_the_uncorrected_wind_speed(option):
    """WRF's second friction velocity (``module_sf_sfclay.F:799-804``,
    ``physics_mmm/sf_sfclayrev.F90:757-763``).

    ``UST`` relaxes toward ``karman*WSPD/PSIX``, and ``WSPD`` carries the
    Beljaars/Mahrt-Sun ``vconv``/``vsgd`` additions plus a 0.1 m/s floor
    (:502,:528-529).  ``USTM`` repeats the relaxation on ``WSPDI``, the
    raw ``sqrt(ux*ux+vx*vx)`` (:801).  It is not a diagnostic copy of
    ``UST``: ``vertical_diffusion_2`` and ``tke_rhs`` are handed
    ``grid%ustm`` (``dyn_em/module_first_rk_step_part2.F:914,:1066``), so
    an unwritten or aliased USTM is a wrong surface momentum source.
    """
    from woof.verify.npref import np_sfclay

    shape = (1, 1)
    full = lambda value: np.full(shape, value, np.float32)
    karman, u, v = 0.4, 5.0, 0.0
    old_ust, old_ustm = 0.25, 0.0625        # both exact in FP32
    out = np_sfclay(
        full(u), full(v), full(300.0), full(0.01), full(100000.0),
        full(40.0), full(100000.0), full(305.0), full(0.1), full(800.0),
        full(1.0), full(1.0), option=option, qsfc=full(0.01),
        ust=full(old_ust), ustm=full(old_ustm), mol=full(0.0),
        hfx=full(200.0), qfx=full(1.0e-4), dx=1000.0,
    )

    psix = float(out["fm"][0, 0])
    wspd = float(out["wspd"][0, 0])
    wspdi = np.sqrt(u * u + v * v)
    # A convective land column, so WSPD really did pick up vconv and the
    # two wind speeds are different numbers.
    assert wspd > wspdi
    assert float(out["ust"][0, 0]) == (0.5 * old_ust
                                       + 0.5 * karman * wspd / psix)
    assert float(out["ustm"][0, 0]) == (0.5 * old_ustm
                                        + 0.5 * karman * wspdi / psix)
    assert float(out["ustm"][0, 0]) != float(out["ust"][0, 0])


@pytest.mark.parametrize("option", [91, 1])
def test_ustm_takes_neither_the_wind_floor_nor_the_land_floor(option):
    """A windless land column separates the two friction velocities.

    ``WSPD`` cannot fall below 0.1 m/s (``module_sf_sfclay.F:529``) and
    ``UST`` is then floored again over land (:817-819, 0.1 classic /
    0.001 revised).  ``WSPDI`` has neither guard, so USTM is pure
    relaxation: exactly half its incoming value here.
    """
    from woof.verify.npref import np_sfclay

    shape = (1, 1)
    full = lambda value: np.full(shape, value, np.float32)
    old_ustm = 0.0625
    out = np_sfclay(
        full(0.0), full(0.0), full(300.0), full(0.01), full(100000.0),
        full(40.0), full(100000.0), full(300.0), full(0.1), full(800.0),
        full(1.0), full(1.0), option=option, qsfc=full(0.01),
        ust=full(0.0), ustm=full(old_ustm), mol=full(0.0),
        hfx=full(0.0), qfx=full(0.0), dx=1000.0,
    )

    assert float(out["wspd"][0, 0]) == 0.1
    assert float(out["ustm"][0, 0]) == 0.5 * old_ustm
    if option == 91:
        # UST sits exactly on its land floor while USTM sits below it,
        # which is only possible because WRF floors UST alone.
        assert float(out["ust"][0, 0]) == 0.1
        assert float(out["ustm"][0, 0]) < 0.1


def test_sfclay_publishes_ustm_as_unconditional_registry_state():
    """The producer half of the USTM path.

    ``Registry.EM_COMMON:1954`` declares USTM ``ij misc 1 - r`` -- plain,
    unconditional, restart-carried state -- and ``module_surface_driver.F``
    passes it positionally to both MM5 schemes (:2073, :2127), so
    ``PRESENT(USTM)`` is never false in an EM build.  WOOF-ArWen's consumers read
    ``fields["ustm"]`` (``dycore.py:970``, ``:1164``); publishing it from the
    scheme's own output set is what makes that read a computed value instead
    of an array nobody writes.
    """
    from woof.core.physics_inventory import SFCLAY_OUTPUTS
    from woof.core.preflight import physics_field_names_2d

    assert "ustm" in SFCLAY_OUTPUTS
    # No km_opt / bl_pbl_physics predicate gates it.
    assert "ustm" in physics_field_names_2d()
    assert "ustm" in physics_field_names_2d(
        RunConfig(**_BASE_CFG, sf_sfclay_physics=1, km_opt=1,
                  bl_pbl_physics=1))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("option", [91, 1])
def test_sfclay_kernel_matches_float64_mirror_all_regimes(option):
    import cupy as cp

    from woof.core.sfclay import SFCLAY_OUTPUTS, sfclay

    data = _surface_cases(ny=7, nx=11)
    dev = {name: cp.asarray(value) for name, value in data.items()}
    got = sfclay(
        dev["u"], dev["v"], dev["t"], dev["qv"], dev["p"],
        dev["dz8w"], dev["psfc"], dev["tsk"], dev["znt"],
        dev["pblh"], dev["mavail"], dev["xland"], option=option,
        qsfc=dev["qsfc"], ust=dev["ust"], mol=dev["mol"],
        hfx=dev["hfx"], qfx=dev["qfx"], lakemask=dev["lakemask"],
        dx=1000.0,
    )
    ref = _run_mirror(data, option)

    regimes = set(cp.asnumpy(got.regime).astype(int).ravel())
    assert {3, 4}.issubset(regimes)
    assert regimes.intersection({1, 2})
    assert got.ust.dtype == cp.float32
    for name in SFCLAY_OUTPUTS:
        actual = cp.asnumpy(getattr(got, name))
        if name == "regime":
            np.testing.assert_array_equal(actual, ref[name], err_msg=name)
        else:
            scale = max(float(np.max(np.abs(ref[name]))), 1.0e-7)
            np.testing.assert_allclose(actual, ref[name], rtol=5.0e-4,
                                       atol=5.0e-5 * scale, err_msg=name)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("option", [91, 1])
def test_sfclay_kernel_writes_ustm_and_not_a_copy_of_ust(option):
    """The kernel half of ``module_sf_sfclay.F:799-804``.

    ``_allocate_result`` seeds ``result.ustm`` from the incoming inout
    array, so a kernel that never wrote USTM would hand back the zeros it
    was given -- which is exactly what ``vertical_diffusion_2`` and
    ``tke_rhs`` read before this line existed.  The column is convective
    land, so ``vconv`` separates the two friction velocities.
    """
    import cupy as cp

    from woof.core.sfclay import sfclay
    from woof.verify.npref import np_sfclay

    shape = (1, 1)
    host = lambda value: np.full(shape, value, np.float32)
    args = dict(
        u=host(5.0), v=host(0.0), t=host(300.0), qv=host(0.01),
        p=host(100000.0), dz8w=host(40.0), psfc=host(100000.0),
        tsk=host(305.0), znt=host(0.1), pblh=host(800.0),
        mavail=host(1.0), xland=host(1.0))
    inout = dict(qsfc=host(0.01), ust=host(0.25), ustm=host(0.0),
                 mol=host(0.0), hfx=host(200.0), qfx=host(1.0e-4))

    got = sfclay(*(cp.asarray(v) for v in args.values()), option=option,
                 dx=1000.0,
                 **{k: cp.asarray(v) for k, v in inout.items()})
    ref = np_sfclay(*args.values(), option=option, dx=1000.0, **inout)

    ustm = float(got.ustm.get()[0, 0])
    assert ustm > 0.0                       # written, not the seeded zero
    assert ustm != float(got.ust.get()[0, 0])
    assert ustm == pytest.approx(float(ref["ustm"][0, 0]), rel=5.0e-4)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("option", [91, 1])
def test_sfclay_kernel_one_fp32_ulp_temperature_difference_is_not_neutral(option):
    import cupy as cp

    from woof.core.sfclay import sfclay

    shape = (1, 1)
    full = lambda value: cp.full(shape, value, cp.float32)
    tsk = np.nextafter(np.float32(300.0), np.float32(np.inf))
    got = sfclay(
        full(5.0), full(0.0), full(300.0), full(0.01), full(100000.0),
        full(40.0), full(100000.0), full(tsk), full(0.1), full(800.0),
        full(1.0), full(1.0), option=option, qsfc=full(0.01),
        ust=full(0.2), mol=full(0.0), hfx=full(0.0), qfx=full(0.0),
    )

    assert float(got.br.get()[0, 0]) < 0.0
    assert int(got.regime.get()[0, 0]) == 4


@pytest.mark.gpu
@requires_gpu
def test_sfclay_kernel_classic_strong_stable_preserves_incoming_zol():
    import cupy as cp

    from woof.core.sfclay import sfclay

    shape = (1, 1)
    full = lambda value: cp.full(shape, value, cp.float32)
    incoming_zol = np.float32(7.25)
    got = sfclay(
        full(0.1), full(0.0), full(300.0), full(0.01), full(100000.0),
        full(80.0), full(100000.0), full(290.0), full(0.1), full(800.0),
        full(1.0), full(1.0), option=91, qsfc=full(0.01),
        ust=full(0.2), mol=full(0.4), zol=full(incoming_zol),
        hfx=full(0.0), qfx=full(0.0),
    )

    assert float(got.br.get()[0, 0]) >= 0.2
    assert int(got.regime.get()[0, 0]) == 1
    assert got.zol.get()[0, 0] == incoming_zol
    assert got.rmol.get()[0, 0] * np.float32(40.0) != incoming_zol


@pytest.mark.gpu
@requires_gpu
def test_sfclay_public_validation():
    import cupy as cp

    from woof.core.sfclay import sfclay

    data = _surface_cases(ny=2, nx=2)
    args = [cp.asarray(data[name]) for name in
            ("u", "v", "t", "qv", "p", "dz8w", "psfc", "tsk", "znt",
             "pblh", "mavail", "xland")]
    with pytest.raises(ValueError, match="option"):
        sfclay(*args, option=2)
    with pytest.raises(ValueError, match="isftcflx"):
        sfclay(*args, option=91, isftcflx=3)
    with pytest.raises(ValueError, match="iz0tlnd"):
        sfclay(*args, option=91, iz0tlnd=3)


def _degenerate_roughness_case(z0):
    """One land column with a degenerate z0; host-built float32 arrays.

    The subnormal must be materialized on the host: a device fill kernel's
    store would flush it to zero under this compile route, and the point is
    that the *stored field* can carry a subnormal (e.g. ingested static
    data), not only an exact zero.
    """
    shape = (1, 1)
    full = lambda v: np.full(shape, v, np.float32)
    return {
        "u": full(5.0), "v": full(0.0), "t": full(300.0), "qv": full(0.01),
        "p": full(100000.0), "dz8w": full(40.0), "psfc": full(100000.0),
        "tsk": full(303.0), "znt": full(z0), "pblh": full(800.0),
        "mavail": full(1.0), "xland": full(1.0), "qsfc": full(0.01),
        "ust": full(0.25), "mol": full(-0.2), "hfx": full(0.0),
        "qfx": full(0.0), "lakemask": full(0.0),
    }


@pytest.mark.parametrize("z0", [1.0e-38, 0.0])
@pytest.mark.parametrize("option", [91, 1])
def test_mirror_degenerate_roughness_is_defined(option, z0):
    """The float64 authority is finite and pins the defined semantics.

    WRF v4.6.1 has no guard here: module_sf_sfclay.F:494
    ``GZ1OZ0 = ALOG(ZA/ZNT)`` (and sf_sfclayrev.F90:318) at ``ZNT=0``
    propagates Inf into ``U10 = Inf/Inf = NaN``.  woof diverges by
    definition: ``z0 <= 0`` floors the logarithm's roughness at FLT_MIN,
    so zero roughness is a huge-but-finite contrast, not Inf.
    """
    data = _degenerate_roughness_case(z0)
    out = _run_mirror(data, option)
    for name, value in out.items():
        assert np.isfinite(value).all(), name
    za = 20.0
    z0_eff = float(np.float64(np.float32(z0))) if z0 > 0.0 \
        else 1.1754943508222875e-38
    num = za if option == 91 else za + z0_eff
    assert out["gz1oz0"][0, 0] == pytest.approx(np.log(num / z0_eff),
                                                rel=1e-12)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("z0", [1.0e-38, 0.0])
@pytest.mark.parametrize("option", [91, 1])
def test_kernel_degenerate_roughness_matches_mirror(option, z0):
    """z0=1e-38 and z0=0 through the real kernel on the device.

    Pre-guard, ``za/z0`` overflowed FP32 before the log could act (and the
    f32 division DAZes a subnormal z0 outright on sm_120), so
    ``gz1oz0 = logf(Inf) = Inf`` poisoned every downstream similarity
    quantity.  The kernel now rescues the ratio in float64 (bit-decoded
    widening, since the f32->f64 cvt also DAZes) and must land on the
    float64 mirror's defined values.
    """
    import cupy as cp

    from woof.core.sfclay import SFCLAY_OUTPUTS, sfclay

    data = _degenerate_roughness_case(z0)
    dev = {name: cp.asarray(value) for name, value in data.items()}
    got = sfclay(
        dev["u"], dev["v"], dev["t"], dev["qv"], dev["p"],
        dev["dz8w"], dev["psfc"], dev["tsk"], dev["znt"],
        dev["pblh"], dev["mavail"], dev["xland"], option=option,
        qsfc=dev["qsfc"], ust=dev["ust"], mol=dev["mol"],
        hfx=dev["hfx"], qfx=dev["qfx"], lakemask=dev["lakemask"],
        dx=1000.0,
    )
    # The device really received the subnormal, not a flushed zero.
    assert cp.asnumpy(dev["znt"]).view(np.uint32)[0, 0] == \
        np.asarray(data["znt"]).view(np.uint32)[0, 0]
    ref = _run_mirror(data, option)
    for name in SFCLAY_OUTPUTS:
        actual = cp.asnumpy(getattr(got, name))
        assert np.isfinite(actual).all(), name
        if name == "regime":
            np.testing.assert_array_equal(actual, ref[name], err_msg=name)
        else:
            scale = max(float(np.max(np.abs(ref[name]))), 1.0e-7)
            np.testing.assert_allclose(actual, ref[name], rtol=5.0e-4,
                                       atol=5.0e-5 * scale, err_msg=name)
    # And the two degenerate inputs stay distinguishable: the subnormal's
    # own logarithm, not one shared floor value.
    za = 20.0
    z0_eff = float(np.float64(np.float32(z0))) if z0 > 0.0 \
        else 1.1754943508222875e-38
    num = za if option == 91 else za + z0_eff
    assert float(got.gz1oz0.get()[0, 0]) == pytest.approx(
        np.log(num / z0_eff), rel=1e-6)

def test_mirror_nan_roughness_stays_nan():
    """A NaN znt must not come back as a plausible number.

    ``_sf_log_zratio``'s degenerate-roughness floor is written
    ``not z0 > 0.0``, and that predicate is true for NaN as well as for
    zero.  Without a NaN guard ahead of it, a NaN roughness leaves as
    ``log(num / FLT_MIN)`` -- roughly 90, a perfectly ordinary contrast
    that every downstream finiteness check accepts.  The corrupted state
    would then be invisible: the column would report finite fluxes
    computed from a roughness nobody has.

    Zero roughness is a physical input this port deliberately defines
    (see ``test_mirror_degenerate_roughness_is_defined``).  NaN roughness
    is evidence of a defect upstream, so it must survive as one.

    Pinned on the function rather than through a whole column because the
    defect and its guard both live here; a column carries unrelated
    downstream behaviour on NaN input.
    """
    from woof.verify.npref import _sf_log_zratio

    assert np.isnan(_sf_log_zratio(20.0, float("nan")))
    assert np.isnan(_sf_log_zratio(float("nan"), 0.1))
    # The defined zero-roughness case is untouched by the guard.
    assert np.isfinite(_sf_log_zratio(20.0, 0.0))


@pytest.mark.gpu
@requires_gpu
def test_kernel_nan_roughness_matches_mirror():
    """The device half of the same guard.

    ``sf_log_zratio``'s float64 rescue floors with ``!(zd > 0.0)``, true
    for NaN, so pre-guard the kernel returned ``log(num) - log(FLT_MIN)``
    for a NaN ``znt`` exactly as the mirror did.  Both now propagate.
    """
    import cupy as cp

    from woof.core.sfclay import sfclay

    data = _degenerate_roughness_case(float("nan"))
    dev = {name: cp.asarray(value) for name, value in data.items()}
    got = sfclay(
        dev["u"], dev["v"], dev["t"], dev["qv"], dev["p"],
        dev["dz8w"], dev["psfc"], dev["tsk"], dev["znt"],
        dev["pblh"], dev["mavail"], dev["xland"], option=91,
        qsfc=dev["qsfc"], ust=dev["ust"], mol=dev["mol"],
        hfx=dev["hfx"], qfx=dev["qfx"], lakemask=dev["lakemask"],
        dx=1000.0,
    )
    assert np.isnan(float(got.gz1oz0.get()[0, 0])), (
        "NaN roughness was laundered into a finite contrast on the device")
