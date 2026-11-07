"""Kain--Fritsch cumulus physics (WRF ``cu_physics=1``).

The numerical authority is WRF v4.6.1 itself: ``kernels/kf.cu`` is graded
word for word against the unmodified Fortran by
``tests/test_kf_wrf461_oracle.py``.  These tests pin the KF lookup table
(110,420 cells against the gfortran/glibc build of ``KF_LUTAB``), the
W0AVG weights, the F_QR mapping, the driver binding (NCA hold, trigger
history cadence) and the batched output validation.

The float64 mirror (``np_kf_column``) and every test graded against it were
retired on 2026-10-07 with the kernel they described: it re-derived the
scheme instead of porting it, and the oracle measured it off WRF on every
convecting column.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from types import SimpleNamespace
from _run_config_fake import run_config_fake

import numpy as np
import pytest

from conftest import requires_gpu


def test_kf_lookup_table_is_shipped_data_with_provenance():
    from woof.core.kf import load_kf_table

    table = load_kf_table()
    assert table.temperature.shape == (250, 220)
    assert table.qsat.shape == (250, 220)
    assert table.thetae_base.shape == (220,)
    assert table.log_ratio.shape == (200,)
    assert table.pressure_top == 5000.0
    assert table.pressure_reciprocal > 0.0
    assert table.thetae_reciprocal == 1.0
    assert np.isfinite(table.temperature).all()
    assert np.isfinite(table.qsat).all()
    assert np.all(np.diff(table.log_ratio) > 0.0)
    provenance = (Path(__file__).parents[1] / "woof" / "data" /
                  "kf_lutab" / "PROVENANCE.md")
    text = provenance.read_text(encoding="utf-8")
    assert "module_cu_kfeta.F:3174-3301" in text
    assert "https://github.com/wrf-model/WRF.git" in text
    assert "v4.6.1" in text
    assert "d66e442fccc04111067e29274c9f9eaccc3cef28" in text
    # The parent-file pin must be the LF digest -- the bytes a Linux checkout
    # and the gfortran oracle build see -- because that is where the pin gets
    # verified.  The CRLF rendering stays recorded so a Windows checkout that
    # computes it does not read a mismatch as a moved source, but it is
    # labelled, not offered as the authority.
    lf_digest = "e6376c2d85c45470f49d545b25d513b5ec111bf36b87beebc740bf42825c6e5f"
    crlf_digest = "b2ee225b2148d54afa464f941d967cb197a8a73e23d6fd3450086c6f3a705895"
    lowered = text.lower()
    assert lf_digest in lowered
    assert crlf_digest in lowered
    assert lowered.index(lf_digest) < lowered.index(crlf_digest)
    assert "CRLF" in text
    # The FABE erratum: the published relative move was the absolute one.
    assert "4.46e-5" in text
    assert "1.32e-6" in text
    assert "default-`REAL` arithmetic" in text
    # The runtime column is a port graded against WRF, and the only
    # remaining differences are on paths where WRF is undefined or aborts.
    assert "no deviations from WRF v4.6.1 on any defined path" in text
    assert "Deliberate runtime deviations" not in text
    for documented in ("`KLCL = 1`", "`wrf_error_fatal`",
                       "Fortran column-major order", "zero `TMB`",
                       "tools/kf_wrf461_oracle"):
        assert documented in text
    assert "SHA-256" in text
    license_file = provenance.with_name("LICENSE.txt")
    license_text = license_file.read_text(encoding="utf-8")
    assert "National Center for Atmospheric Research" in license_text
    assert "public domain" in license_text
    assert (hashlib.sha256((provenance.parent / "kf_lutab.npz").read_bytes())
            .hexdigest()) in text


#: SHA-256 of the raw binary32 streams `tools/kf_lutab_oracle.F90` writes --
#: WRF v4.6.1 `module_cu_kfeta.F:3174-3301` verbatim, gfortran 13.3.0 against
#: glibc 2.39 on x86-64, identical at `-O0 -ffp-contract=off` and at
#: `-O2 -ftree-vectorize`.  Recorded in `woof/data/kf_lutab/PROVENANCE.md`.
#: Fortran writes column-major, hence `order="F"` on the two 2-D members.
_KF_LUTAB_ORACLE_SHA256 = {
    "ttab.bin":
        "04c8e3cd4138e440d486b5899575063fde81b4beba5d82c8ffe6a2d766fd5ce0",
    "qstab.bin":
        "39ae8b543154f035cc47512929335c15c7880c30f33f41cd8a4518c3e8975d7a",
    "the0k.bin":
        "a6e207a3f0f9b3631aea74ce89b070b57eed6c298fb40f2488dca4627391874d",
    "alu.bin":
        "6836d80708f02642d1dc53b547817dbab623545cfee778671290d7d825eeaf4d",
    "scalars.bin":
        "206e7022af0e9ef62d5203083a83efac175607e8263c873b19b7708d6bbe352c",
}

#: NumPy ufuncs the generator must never reach for.  WRF's `KF_LUTAB` is
#: default `REAL` -- binary32 -- and gfortran lowers its `exp`, `**` and
#: `alog` to the scalar glibc `expf`, `powf` and `logf`.  NumPy's float32
#: transcendentals are a different function *and* move between releases, so
#: one appearing here is both a fidelity bug and a reproducibility bug.
_FORBIDDEN_NUMPY_TRANSCENDENTALS = (
    "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "power",
    "float_power", "sqrt", "cbrt",
)


def _oracle_stream_digests(table):
    """Digest a table the way `kf_lutab_oracle.F90` writes it out."""
    def as_le(array):
        return np.asarray(array, dtype="<f4")

    return {
        "ttab.bin": hashlib.sha256(
            as_le(table["temperature"]).tobytes(order="F")).hexdigest(),
        "qstab.bin": hashlib.sha256(
            as_le(table["qsat"]).tobytes(order="F")).hexdigest(),
        "the0k.bin": hashlib.sha256(
            as_le(table["thetae_base"]).tobytes()).hexdigest(),
        "alu.bin": hashlib.sha256(
            as_le(table["log_ratio"]).tobytes()).hexdigest(),
        "scalars.bin": hashlib.sha256(as_le([
            table["pressure_reciprocal"], table["thetae_reciprocal"],
            table["pressure_top"]]).tobytes()).hexdigest(),
    }


def test_kf_lutab_generator_preserves_wrf_single_precision_semantics():
    """Reproduce `KF_LUTAB` bit-for-bit, against WRF and not against NumPy.

    `module_cu_kfeta.F:3174-3301` is default `REAL`, and WRF v4.6.1 does not
    promote it (`arch/preamble:63` sets `NATIVE_RWORDSIZE = 4`; every
    `PROMOTION` line in `arch/configure.defaults` leaves `-fdefault-real-8`
    commented out).  So "WRF single precision semantics" means binary32
    arithmetic *plus* the single-precision libm gfortran calls -- glibc's
    `expf`, `powf`, `logf`.  This pins all 110,420 cells against the recorded
    output of that Fortran, and guards the generator against drifting back
    onto a NumPy transcendental, which is neither WRF's function nor stable
    between NumPy releases.
    """
    from woof.core import noahmp_libm
    from woof.core.kf import load_kf_table
    from tools import generate_kf_lutab

    # The libm the pinned digests were produced against.  A host that moves
    # off glibc 2.39 has not invalidated them -- the transcriptions are
    # self-contained -- but the claim they reproduce is version-specific.
    assert noahmp_libm.GLIBC_VERSION == "2.39"

    generated = generate_kf_lutab.generate()
    assert all(value.dtype == np.float32 for value in generated.values())

    # All 110,420 cells, against gfortran 13.3.0 + glibc 2.39.
    assert _oracle_stream_digests(generated) == _KF_LUTAB_ORACLE_SHA256

    # Anchors, so a digest mismatch reports a number and not just a hash.
    # Every value below is what the Fortran prints, not what NumPy computes;
    # (88, 211) and (148, 204) are the worst cells of the superseded
    # NumPy-built table (65 and 452 ULP off WRF respectively).
    anchors = ((0, 0), (10, 50), (100, 100), (249, 219), (88, 211),
               (148, 204))
    np.testing.assert_array_equal(
        [generated["temperature"][index] for index in anchors],
        np.asarray([150.0, 157.02183532714844, 232.8066864013672,
                    307.37762451171875, 238.96221923828125,
                    281.1182556152344], dtype=np.float32))
    np.testing.assert_array_equal(
        [generated["qsat"][index] for index in anchors],
        np.asarray([1.0677771999922925e-9, 1.3225719408538339e-9,
                    2.1497125271707773e-4, 3.207545354962349e-2,
                    1.9985366088803858e-4, 6.541135720908642e-3],
                   dtype=np.float32))
    np.testing.assert_array_equal(
        generated["thetae_base"][[0, 50, 100, 219]],
        np.asarray([352.6997985839844, 213.61912536621094,
                    179.8502655029297, 145.97471618652344],
                   dtype=np.float32))
    np.testing.assert_array_equal(
        generated["log_ratio"][[0, 1, 50, 100, 199]],
        np.asarray([-6.9077534675598145, -2.577021837234497,
                    1.3220229148864746, 2.015035390853882,
                    2.7031028270721436], dtype=np.float32))

    shipped = load_kf_table()
    for name in ("temperature", "qsat", "thetae_base", "log_ratio"):
        np.testing.assert_array_equal(generated[name], getattr(shipped, name))
    for name in ("pressure_top", "pressure_reciprocal",
                 "thetae_reciprocal"):
        assert float(generated[name]) == getattr(shipped, name)


def test_kf_lutab_generator_calls_no_numpy_transcendental(monkeypatch):
    """Negative control for the table's reproducibility.

    The digests above cannot tell you *why* the table reproduces, so this
    breaks every NumPy elementary function under the generator and reruns it.
    If a future edit reaches for `np.exp` instead of
    `woof.core.noahmp_libm.expf`, this fails immediately rather than five
    NumPy releases later.  `+ - * /` are left alone: those are correctly
    rounded in binary32 and are the same function everywhere.
    """
    from tools import generate_kf_lutab

    called: list[str] = []

    def forbid(name):
        def trap(*args, **kwargs):
            called.append(name)
            raise AssertionError(
                f"generate_kf_lutab called np.{name}; WRF's KF_LUTAB is "
                f"binary32 and gfortran lowers it to glibc's expf/powf/logf. "
                f"Use woof.core.noahmp_libm.")
        return trap

    for name in _FORBIDDEN_NUMPY_TRANSCENDENTALS:
        monkeypatch.setattr(np, name, forbid(name), raising=True)

    # The trap is live: prove it fires before trusting that it did not.
    with pytest.raises(AssertionError, match=r"called np\.exp"):
        np.exp(np.float32(1.0))
    called.clear()

    table = generate_kf_lutab.generate()
    assert called == []
    assert _oracle_stream_digests(table) == _KF_LUTAB_ORACLE_SHA256


def test_w0avg_weights_are_kf_eta_cps_single_precision_forms():
    """module_cu_kfeta.F:209-230, each operation in binary32."""
    from woof.core.kf import w0avg_factors

    f32 = np.float32
    assert w0avg_factors(cudt_minutes=5.0, dt=60.0, stepcu=5,
                         adaptive=False) == (f32(9.0), f32(1.0), f32(10.0))
    assert w0avg_factors(cudt_minutes=0.0, dt=60.0, stepcu=1,
                         adaptive=False) == (f32(1.0), f32(1.0), f32(2.0))
    # ADAPT_STEP_FLAG: W0AVGfctr = 2*MAX(CUDT*60,DT) - DT, W0fctr = DT,
    # W0den = 2*MAX(CUDT*60,DT).
    fctr, w0fctr, den = w0avg_factors(cudt_minutes=5.0, dt=50.0, stepcu=6,
                                      adaptive=True)
    assert (fctr, w0fctr, den) == (f32(550.0), f32(50.0), f32(600.0))
    fctr, w0fctr, den = w0avg_factors(cudt_minutes=0.0, dt=37.5, stepcu=1,
                                      adaptive=True)
    assert (fctr, w0fctr, den) == (f32(37.5), f32(37.5), f32(75.0))
    assert all(type(x) is np.float32 for x in (fctr, w0fctr, den))


def test_f_qr_follows_the_admitted_microphysics():
    """KF_eta_CPS folds DQRDT into RQCCUTEN without QR (:492-505)."""
    from woof.core.kf import kf_has_rain_for_microphysics

    assert kf_has_rain_for_microphysics(0) is False
    # 2, 3, 4, 5, 24 and 106 joined with golden/ports-mp (2.9.0); each carries
    # a rain species (WSM3 holds rain in qr above freezing).
    # 7 (Goddard 4-ice) joined with its port; it carries qr.
    # 95 (the old Eta Ferrier scheme) joined with its 3.0 port; it carries qr.
    for mp in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 18, 24, 28, 50, 95, 106):
        assert kf_has_rain_for_microphysics(mp) is True
    # A selector no port carries is still refused.
    with pytest.raises(ValueError):
        kf_has_rain_for_microphysics(12345)


@pytest.mark.gpu
@requires_gpu
def test_kf_trigger_history_advances_per_due_call_and_scheme_consumes_it(
        monkeypatch):
    """Pin W0AVG cadence: one recursive-mean sample per STEPCU due event.

    WRF's cumulus driver early-returns on non-due steps
    (module_cumulus_driver.F:830-864, RETURN at :863), so the W0AVG loop
    (module_cu_kfeta.F:218-251) runs once per due call with weight 1/TST,
    TST = 2*STEPCU on the model clock.  The update is the KF unit's own
    no-FMA kernel, so the expected mean is formed in binary32 on the host,
    operation for operation.
    """
    import cupy as cp

    from woof.core.kf import KFPhaseMode, KainFritsch

    shape = (8, 1, 1)
    atmosphere = {
        name: cp.ones(shape, dtype=cp.float32)
        for name in ("u", "v", "temperature", "qv", "qc", "pressure",
                     "rho", "exner", "dz")
    }
    state = SimpleNamespace(
        w=cp.arange(9, dtype=cp.float32).reshape(9, 1, 1),
        p=cp.ones(shape, dtype=cp.float32),
    )

    def scratch(requested_shape, _name):
        return cp.zeros(requested_shape, dtype=cp.float32)

    state.scratch = scratch
    captured = {}

    def fake_launch(*args, **kwargs):
        captured["w0avg"] = args[8].copy()
        captured["rho"] = args[5]
        captured.update(kwargs)
        zeros = cp.zeros(shape, dtype=cp.float32)
        flat = cp.zeros((1, 1), dtype=cp.float32)
        return {
            "rthcuten": zeros, "rqvcuten": zeros.copy(),
            "rqccuten": zeros.copy(), "rqicuten": zeros.copy(),
            "rqrcuten": zeros.copy(), "rqscuten": zeros.copy(),
            "rainc": flat, "nca_seconds": flat.copy(),
            "pratec": flat.copy(),
        }

    monkeypatch.setattr("woof.core.kf.launch_kf", fake_launch)
    cfg = run_config_fake(cudt_minutes=5.0, dt=60.0, dx=12000.0,
                          mp_physics=10)
    scheme = KainFritsch()
    f32 = np.float32
    expected = np.zeros(shape, dtype=np.float32)
    for event, offset in enumerate((0.0, 2.0, -1.0)):
        state.elapsed_seconds = event * 300.0
        w = np.arange(9, dtype=np.float32).reshape(9, 1, 1) + f32(offset)
        state.w[...] = cp.asarray(w)
        w0 = f32(0.5) * (w[:-1] + w[1:])
        expected = (expected * f32(9.0) + w0 * f32(1.0)) / f32(10.0)
        scheme.update_trigger_history(state=state, cfg=cfg)
        np.testing.assert_array_equal(cp.asnumpy(scheme.w0avg), expected)

    scheme(atmosphere=atmosphere, fields={}, state=state, cfg=cfg)

    # The scheme call at the same elapsed time consumes the already-updated
    # mean without double-counting the sample, and hands KF_eta_PARA the
    # driver's RHO and the namelist CUDT in minutes.
    np.testing.assert_array_equal(cp.asnumpy(captured["w0avg"]), expected)
    assert captured["rho"] is atmosphere["rho"]
    assert captured["cudt_minutes"] == 5.0
    assert captured["dt"] == 60.0
    assert captured["phase_mode"] == KFPhaseMode.SEPARATE_ICE_SNOW
    assert captured["f_qr"] is True

    # Under the compatibility substep integration (clock_dt > dt) a due
    # call advances the mean exactly once with the CLOCK-derived
    # TST = 2*NINT(cudt*60/clock_dt) = 10, and the scheme receives the
    # model-clock DT (module_cumulus_driver.F:1028).
    sub_cfg = run_config_fake(cudt_minutes=5.0, dt=7.5, clock_dt=60.0,
                              dx=12000.0, mp_physics=10)
    sub_scheme = KainFritsch()
    w = np.arange(9, dtype=np.float32).reshape(9, 1, 1)
    state.w[...] = cp.asarray(w)
    w0 = f32(0.5) * (w[:-1] + w[1:])
    state.elapsed_seconds = 292.5   # substep-calendar due instant
    sub_scheme.update_trigger_history(state=state, cfg=sub_cfg)
    first = (np.zeros(shape, np.float32) * f32(9.0)
             + w0 * f32(1.0)) / f32(10.0)
    np.testing.assert_array_equal(cp.asnumpy(sub_scheme.w0avg), first)
    sub_scheme(atmosphere=atmosphere, fields={}, state=state, cfg=sub_cfg)
    np.testing.assert_array_equal(cp.asnumpy(captured["w0avg"]), first)
    assert captured["dt"] == 60.0
    assert captured["cudt_minutes"] == 5.0

    # ADAPT_STEP_FLAG: the DT-weighted mean over 2*MAX(CUDT*60, DT).
    ada_cfg = run_config_fake(cudt_minutes=5.0, dt=50.0, dx=12000.0,
                              mp_physics=10, use_adaptive_time_step=True)
    ada_scheme = KainFritsch()
    state.elapsed_seconds = 0.0
    ada_scheme.update_trigger_history(state=state, cfg=ada_cfg)
    adaptive = (np.zeros(shape, np.float32) * f32(550.0)
                + w0 * f32(50.0)) / f32(600.0)
    np.testing.assert_array_equal(cp.asnumpy(ada_scheme.w0avg), adaptive)


#: The seven deep columns whose capped downdraft took all of the updraft's
#: precipitation on a global 60 km hex forecast (d547dc8c3; GFS 2026-10-10
#: 00Z, 163,842 cells, 55 levels, dt 360 s), written as KF_eta_CPS inputs by
#: tools/kf_wrf461_oracle/capped_columns.py, and WRF v4.6.1's own outputs for
#: them from the byte-unmodified module_cu_kfeta.F at -O0 (build.sh; the
#: SNaN build wrote the same bytes).  WRF returns PRATEC = 0 exactly on all
#: seven: the minus-a-few-ulp rain the pre-port kernel returned there was
#: its own arithmetic, not WRF's.
CAPPED = Path(__file__).parent / "data" / "oracles" / "kf"
CAPPED_SHA256 = {
    "columns-capped.bin":
        "ee88310e4a93064ed8ce6e0909b38c18700c38f7890088b5bad24cdde03193ed",
    "columns-capped.json":
        "b399e0f8758e5c5cf0f5f86603b49a8db73daa6f1249d8a95906f795a6e4df07",
    "oracle-capped-O0.out":
        "a155dd9933116ffffd911d6fe28a49531ce5f289e9cd89183ade97bdf7cb256f",
}
#: kernels/kf.cu's floor on PPTFLX (declared divergence, d547dc8c3): the
#: one statement where the port leaves WRF's arithmetic on purpose.
PPTFLX_FLOOR = "pptflx = fmaxf(trppt - tder, 0.0f);"


def test_the_capped_downdraft_fixture_is_wrfs_recorded_answer():
    from tools.kf_wrf461_oracle import kf_columns

    for name, digest in CAPPED_SHA256.items():
        assert hashlib.sha256((CAPPED / name).read_bytes()).hexdigest()             == digest, name
    out = kf_columns.read_outputs(str(CAPPED / "oracle-capped-O0.out"))
    assert len(out) == 7
    assert not any(column["fatal"] for column in out)
    # Deep convection on every column, and WRF's rain is exactly zero.
    assert [float(column["shall"]) for column in out] == [0.0] * 7
    assert [float(column["pratec"]) for column in out] == [0.0] * 7
    assert [float(column["raincv"]) for column in out] == [0.0] * 7


def test_the_rain_flux_floor_is_the_one_declared_statement():
    source = (Path(__file__).resolve().parents[1] / "woof" / "core"
              / "kernels" / "kf.cu").read_text(encoding="utf-8")
    assert source.count(PPTFLX_FLOOR) == 1
    assert source.count("pptflx = trppt - tder;") == 0
    assert "d547dc8c3" in source


@pytest.mark.gpu
@requires_gpu
def test_kf_rain_is_wrfs_zero_when_the_downdraft_evaporates_it_all():
    """The hex columns of d547dc8c3, graded word for word against WRF.

    Before the WRF port the kernel returned a rain increment of minus a few
    ulp on these columns and RAINC, which accumulates PRATEC*DT signed, went
    negative: a global 60 km forecast was refused at its first history frame
    with "negative accumulated precipitation in rainc".  The port returns
    WRF's own words on every output of all seven, rain exactly zero, and
    the floor kept on PPTFLX can never make the answer negative.
    """
    from tools.kf_wrf461_oracle import compare_woof

    report = compare_woof.run(str(CAPPED / "columns-capped.bin"),
                              str(CAPPED / "oracle-capped-O0.out"),
                              label="capped",
                              meta_path=str(CAPPED / "columns-capped.json"))
    assert report["graded"] == 7
    assert report["mismatched_words"] == 0, report["per_field"]
    assert report["exact_columns"] == 7


@pytest.mark.gpu
@requires_gpu
def test_native_kf_validation_batches_fields_and_resets_status():
    import cupy as cp

    from woof.core.kf import validate_kf_outputs

    shape = (8, 2, 3)
    surface = shape[1:]
    values = tuple(
        cp.full(shape if index < 6 else surface, index + 1, cp.float32)
        for index in range(8))
    before = tuple(value.copy() for value in values)
    status = cp.full((1,), cp.uint32(0xFFFFFFFF), cp.uint32)

    assert validate_kf_outputs(values, 0xFF, status) == 0
    assert int(status[0].item()) == 0
    for got, expected in zip(values, before):
        cp.testing.assert_array_equal(got, expected)

    for index, value in enumerate(values):
        value.flat[0] = cp.nan
        assert validate_kf_outputs(values, 0xFF, status) == 1 << index
        value.flat[0] = cp.float32(index + 1)

    values[-1].flat[0] = cp.inf
    values[0].flat[0] = cp.nan
    invalid = validate_kf_outputs(values, 0xFF, status)
    assert invalid & 1
    assert invalid & (1 << 7)
    values[-1].flat[0] = cp.float32(8.0)
    values[0].flat[0] = cp.float32(1.0)

    # Warm/no-separate-snow modes pass valid placeholder pointers for the
    # two inactive frozen categories; their payloads must not be read.
    values[3].flat[0] = cp.nan
    values[5].flat[0] = cp.inf
    assert validate_kf_outputs(values, 0xD7, status) == 0


@pytest.mark.gpu
@requires_gpu
def test_kf_production_binding_nca_hold_and_domain_flag():
    """Task 6b NCA persistence against the float64 driver mirror.

    Finalizes the interim Task-1/Task-4 every-call-recompute pin: a
    column with time left on its NCA timer holds its stored tendencies,
    PRATEC, RAINCV, and NCA across STEPCU events
    (module_cu_kfeta.F:410-412), RAINC accumulates PRATEC*DT on every
    dynamics step, and the stored rates are zeroed on the WRF
    NINT(NCA/DT) <= 1 boundary while the rain rate survives expiry
    (module_physics_addtendc.F:2139-2231; solve_em.F:3558-3571).
    """
    import cupy as cp

    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.kf import KainFritsch
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import (
        CumulusResult, _prepare_atmosphere, initialize_physics,
    )
    from woof.verify.npref import np_cumulus_nca_driver_step

    def state_for(cfg):
        coord = make_vertical_coord(cfg.nz)
        base = make_base_state(
            coord, lambda z: 300.0 + 0.003 * np.asarray(z),
            p_surf=cfg.p_surf, ztop=cfg.ztop)
        return init_moist_balanced(
            cfg, coord, base,
            lambda z: 0.010 * np.exp(-np.asarray(z) / 2200.0))

    # Morrison explicitly carries all four KF hydrometeor rates through the
    # NCA hold.  The separate CPU seam regression covers WRF's latent-energy
    # warm-rain and no-separate-snow closures.
    cfg = RunConfig(
        nx=3, ny=1, nz=16, dx=12000.0, dy=12000.0, ztop=9000.0,
        dt=60.0, run_seconds=0.0, moist=True, mp_physics=10, cu_physics=1,
        cudt_minutes=5.0)
    state = state_for(cfg)
    # cu_physics=1 auto-binds the production Kain-Fritsch adapter.
    driver = initialize_physics(state, cfg)
    assert isinstance(driver.cumulus_callable, KainFritsch)
    production_result = driver.cumulus_callable(
        atmosphere=_prepare_atmosphere(state), fields=driver.fields,
        state=state, cfg=cfg)
    for value in (production_result.rthcuten, production_result.rqvcuten,
                   production_result.rqccuten, production_result.rqicuten,
                   production_result.rqrcuten, production_result.rqscuten,
                   production_result.rainc, production_result.nca_seconds,
                   production_result.pratec):
        assert bool(cp.isfinite(value).all())
    # kf_init seeds the hold timer at -100 s (module_cu_kfeta.F:3154).
    cp.testing.assert_array_equal(driver.cu_nca, -100.0)

    shape = tuple(state.p.shape)
    surface = tuple(state.mup.shape)

    # Deterministic per-column scheme: column 0 fires a deep 480 s hold
    # (NIC=8 at dt=60), column 1 never triggers, and column 2 fires a
    # rain-free shallow hold of one cudt (module_cu_kfeta.F:2570-2573).
    # Outputs change per call, so a held column visibly ignores them.
    def scheme_fields(call_index):
        scale = np.float32(1.0 + 0.5 * call_index)
        fields = {}
        for name, base in (("rthcuten", 1.0e-5), ("rqvcuten", -2.0e-8),
                           ("rqccuten", 3.0e-9), ("rqicuten", 3.5e-9),
                           ("rqrcuten", 4.0e-9), ("rqscuten", 4.5e-9)):
            field = np.zeros(shape, np.float32)
            field[:, 0, 0] = np.float32(base) * scale
            field[:, 0, 2] = np.float32(2.0 * base) * scale
            fields[name] = field
        pratec = np.zeros(surface, np.float32)
        pratec[0, 0] = np.float32(2.5e-4) * scale
        nca = np.zeros(surface, np.float32)
        nca[0, 0] = 480.0
        nca[0, 2] = 300.0
        fields.update(pratec=pratec, nca_seconds=nca)
        return fields

    calls = []

    def deterministic_cumulus(**kwargs):
        fields = scheme_fields(len(calls))
        calls.append(float(kwargs["state"].elapsed_seconds))
        return CumulusResult(
            cp.asarray(fields["rthcuten"]), cp.asarray(fields["rqvcuten"]),
            rqccuten=cp.asarray(fields["rqccuten"]),
            rqicuten=cp.asarray(fields["rqicuten"]),
            rqrcuten=cp.asarray(fields["rqrcuten"]),
            rqscuten=cp.asarray(fields["rqscuten"]),
            rainc=cp.zeros(surface, cp.float32),
            nca_seconds=cp.asarray(fields["nca_seconds"]),
            pratec=cp.asarray(fields["pratec"]))

    history_calls = []
    deterministic_cumulus.update_trigger_history = lambda **kwargs: (
        history_calls.append(float(kwargs["state"].elapsed_seconds)))
    driver.cumulus_callable = deterministic_cumulus

    mirror = {
        "nca": np.full(surface, -100.0), "pratec": np.zeros(surface),
        "raincv": np.zeros(surface), "rainc": np.zeros(surface),
        **{name: np.zeros(shape) for name in
           ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
            "rqrcuten", "rqscuten")},
    }
    chm = cp.asnumpy(state.c1h[:, None, None] * state.total_mu()[None]
                     + state.c2h[:, None, None]).astype(np.float64)
    due_calls = {0.0: 0, 240.0: 1, 540.0: 2}   # ITIMESTEP 1, 5, 10
    for step_index in range(11):
        now = step_index * cfg.dt
        state.elapsed_seconds = now
        held = driver.compute(state, cfg)
        scheme = (scheme_fields(due_calls[now]) if now in due_calls
                  else None)
        mirror, applied = np_cumulus_nca_driver_step(
            mirror, scheme, dt=cfg.dt)
        # This step's RK forcing couples the pre-expiry applied rates.
        np.testing.assert_allclose(
            cp.asnumpy(held.rtheta),
            (chm * applied["rthcuten"]).astype(np.float32),
            rtol=2.0e-6, atol=0.0)
        # The expiry transition is complete before compute returns: the
        # current RK target above still contains the pre-expiry forcing, but
        # Morrison's later raw-rate inputs and the next held compose already
        # match advance_ppt's cleared state.
        #
        # The held producer rates then carry their mass-coupled words through
        # transport and are decoupled at completed MUTS by the step
        # epilogue's finish_history_mass, before microphysics and the next
        # held call (native calculate_phy_tend / phy_prep_part2; fold
        # 1eb3e8076a, tests/test_kf_held_mass.py).  The mirror takes the
        # same two FP32 mass rounds, so the held words stay word-for-word.
        weight = cp.asnumpy(state.c1h[:, None, None] * state.total_mu()[None]
                            + state.c2h[:, None, None])
        coupled = {}
        for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                     "rqrcuten", "rqscuten"):
            coupled[name] = mirror[name].astype(np.float32) * weight
            np.testing.assert_array_equal(
                cp.asnumpy(driver.cu_rates[name]), coupled[name],
                err_msg=name)
        if now == 420.0:
            assert bool(cp.any(held.rqr[:, 0, 0] != 0.0))
            for name in ("rqrcuten", "rqicuten", "rqscuten"):
                cp.testing.assert_array_equal(
                    driver.cu_rates[name][:, 0, 0], 0.0,
                    err_msg=f"expiry-step Morrison input {name}")
        np.testing.assert_array_equal(
            cp.asnumpy(driver.cu_nca), mirror["nca"])
        np.testing.assert_array_equal(
            cp.asnumpy(driver.cu_pratec), mirror["pratec"])
        np.testing.assert_allclose(
            cp.asnumpy(driver.cu_raincv), mirror["raincv"], rtol=1.0e-6)
        np.testing.assert_allclose(
            cp.asnumpy(driver.rainc), mirror["rainc"], rtol=1.0e-5,
            atol=1.0e-9)
        # The step epilogue's native transition (dycore _step_epilogue), on
        # this static column mass.
        driver.finish_history_mass(state, cfg)
        for name, words in coupled.items():
            decoupled = words / weight
            np.testing.assert_array_equal(
                cp.asnumpy(driver.cu_rates[name]), decoupled,
                err_msg=f"decoupled {name}")
            mirror[name] = decoupled.astype(np.float64)

    # The scheme runs on every STEPCU event; holding is per-column.  The
    # trigger-history hook follows the same due calendar: WRF's cumulus
    # driver early-returns on non-due steps (module_cumulus_driver.F:
    # 830-864, RETURN at :863), so W0AVG advances once per due event.
    assert calls == [0.0, 240.0, 540.0]
    assert history_calls == [0.0, 240.0, 540.0]
    rth0 = cp.asnumpy(driver.cu_rates["rthcuten"])
    # Column 0 was held straight through the 240 s STEPCU event (its call-1
    # doubled rates were discarded), expired after the 420 s step, and was
    # readmitted at 540 s with the call-2 rates.
    np.testing.assert_allclose(rth0[:, 0, 0], np.float32(2.0e-5),
                               rtol=0.0, atol=0.0)
    # Column 2's shallow hold expired after the 240 s step and its 540 s
    # recompute reported no new trigger history here beyond the call-2
    # values; column 1 never held anything.
    np.testing.assert_allclose(rth0[:, 0, 2], np.float32(4.0e-5),
                               rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(rth0[:, 0, 1], 0.0)
    # RAINC kept accumulating at the stored rate after column 0's expiry
    # (WRF leaves PRATEC alone at expiry: addtendc.F:2216-2217 commented
    # out): nine steps at the call-0 rate plus two at the call-2 rate.
    expected_rainc = (9 * 60.0 * float(np.float32(2.5e-4))
                      + 2 * 60.0 * float(np.float32(2.5e-4) * np.float32(2.0)))
    np.testing.assert_allclose(
        float(driver.rainc[0, 0]), expected_rainc, rtol=1.0e-5)
    assert float(driver.rainc[0, 1]) == float(driver.rainc[0, 2]) == 0.0

    # A nested-domain-style independent config with cu_physics=0 does not
    # inherit d01's scheme; SFCLAY merely supplies another enabled slot so a
    # second driver can be constructed for the assertion.
    cfg_off = dataclasses.replace(cfg, cu_physics=0, sf_sfclay_physics=1)
    off_driver = initialize_physics(state_for(cfg_off), cfg_off)
    assert off_driver.cu_physics == 0
    assert off_driver.cumulus_callable is None
