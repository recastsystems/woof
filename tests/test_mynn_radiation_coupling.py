"""WRF v4.6.1 MYNN-cloud/radiation and NSSL legacy-RRTMG gates."""

from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core.mynn_radiation import merge_mynn_bl_clouds
from woof.core.dudhia import DudhiaShortwaveRadiation
from woof.core.rrtmg_legacy import (
    _MP_DECLARES_RADII,
    RRTMGLegacyRadiation,
    legacy_cloud_fraction_flags,
    legacy_ice_active,
    legacy_radius_meters,
)
from woof.core.rrtmgp import RRTMGPRadiation


def _digest(*arrays) -> str:
    value = hashlib.sha256()
    for array in arrays:
        value.update(np.asarray(array).tobytes(order="C"))
    return value.hexdigest()


def _wrf_cal_cldfra1_frozen_oracle(qv, qc, qi, qs, temperature, pressure):
    """Source transcription of module_radiation_driver.F:3859-3979.

    This deliberately does not import woof's NumPy mirror.  It is the
    ``F_QC .and. F_QI .and. F_QS`` arm used by NSSL in WRF v4.6.1.
    """

    qv, qc, qi, qs, temperature, pressure = (
        np.asarray(value, dtype=np.float64)
        for value in (qv, qc, qi, qs, temperature, pressure)
    )
    tc = temperature - 273.15
    esw = 1000.0 * 0.61078 * np.exp(
        17.2693882 * tc / (temperature - 35.86))
    esi = 1000.0 * 0.61078 * np.exp(
        21.8745584 * tc / (temperature - 7.66))
    qvsw = (287.0 / 461.6) * esw / (pressure - esw)
    qvsi = (287.0 / 461.6) * esi / (pressure - esi)
    qcld = qi + qc + qs
    weight = np.where(
        qcld < 1.0e-12, 0.0,
        (qi + qs) / np.maximum(qcld, 1.0e-12),
    )
    qvs_weight = (1.0 - weight) * qvsw + weight * qvsi
    relative_humidity = qv / qvs_weight
    subsaturation = np.maximum(1.0e-10, qvs_weight - qv)
    argument = np.maximum(
        -6.9, -100.0 * qcld / subsaturation ** 0.49)
    fraction = np.maximum(1.0e-10, relative_humidity) ** 0.25 \
        * (1.0 - np.exp(argument))
    fraction = np.where(fraction < 0.01, 0.0, fraction)
    return np.where(
        qcld < 1.0e-12, 0.0,
        np.where(relative_humidity >= 1.0, 1.0, fraction),
    )


@pytest.mark.parametrize(
    "adapter",
    (DudhiaShortwaveRadiation, RRTMGPRadiation, RRTMGLegacyRadiation),
)
@pytest.mark.parametrize(
    ("bl_pbl_physics", "icloud_bl"), ((0, 1), (1, 1), (5, 0)),
)
def test_every_non_mynn_radiation_path_is_byte_identical(
        adapter, bl_pbl_physics, icloud_bl):
    """The shared seam returns before reading or writing a MYNN field."""

    # Tie the identity assertion to each production adapter, rather than
    # merely exercising a stand-alone helper under three descriptive labels.
    source = inspect.getsource(adapter.__call__)
    if adapter is RRTMGLegacyRadiation:
        # The legacy adapter runs this seam on the device, behind the same
        # gate: MYNN fields are read only when mynn_bl_cloud_active holds,
        # and the merge is the rla_mynn kernel, which
        # tests/test_rrtmg_legacy_device_glue.py holds bitwise to
        # merge_mynn_bl_clouds + mynn_bl_cloud_supplied + unsized_mynn_radii.
        # golden/ports-rad gates the merge on icloud = 1 too: with icloud = 2
        # the cloud fraction is WRF's binary cal_cldfra2, which the MYNN
        # subgrid merge does not touch.
        assert "active_bl = icloud == 1 and mynn_bl_cloud_active(" in source
        assert '"rla_mynn"' in source
        assert "if active_bl:" in source
    else:
        assert "merge_mynn_bl_clouds(" in source
    shape = (2, 3)
    qc = np.arange(6, dtype=np.float32).reshape(shape) * np.float32(1.0e-7)
    qi = qc * np.float32(1.0e-2)
    cldfra = np.linspace(0.0, 1.0, 6, dtype=np.float32).reshape(shape)
    objects = (qc, qi, cldfra)
    before = _digest(*objects)

    out = merge_mynn_bl_clouds(
        qc, qi, cldfra, bl_pbl_physics=bl_pbl_physics, icloud_bl=icloud_bl,
        itimestep=2,
    )

    assert all(got is want for got, want in zip(out, objects))
    assert _digest(*objects) == before


@pytest.mark.parametrize("itimestep", (1, 2))
def test_mynn_cloud_merge_is_exactly_the_wrf_source_transcription(itimestep):
    """Pin thresholds, first-step ordering, mass addition, and fraction copy.

    The expected values are transcribed from
    ``module_radiation_driver.F:1403-1429``, not calculated by another WOOF-ArWen
    implementation.  Equality at each threshold proves WRF's strict ``<`` and
    ``>`` comparisons.
    """

    qc = np.asarray([[0.9e-6, 1.0e-6, 0.0]], dtype=np.float32)
    qi = np.asarray([[0.9e-8, 1.0e-8, 0.0]], dtype=np.float32)
    diagnosed = np.asarray([[0.2, 0.3, 0.4]], dtype=np.float32)
    qc_bl = np.asarray([[4.0e-7, 5.0e-7, 6.0e-7]], dtype=np.float32)
    qi_bl = np.asarray([[4.0e-9, 5.0e-9, 6.0e-9]], dtype=np.float32)
    cldfra_bl = np.asarray([[0.0011, 0.8, 0.001]], dtype=np.float32)

    got_qc, got_qi, got_fraction = merge_mynn_bl_clouds(
        qc, qi, diagnosed, qc_bl=qc_bl, qi_bl=qi_bl,
        cldfra_bl=cldfra_bl, bl_pbl_physics=5, icloud_bl=1,
        itimestep=itimestep,
    )

    np.testing.assert_array_equal(
        got_qc, np.asarray([[1.3e-6, 1.0e-6, 0.0]], dtype=np.float32))
    np.testing.assert_array_equal(
        got_qi, np.asarray([[
            np.float32(0.9e-8) + np.float32(4.0e-9), 1.0e-8, 0.0,
        ]], dtype=np.float32))
    expected_fraction = (
        np.asarray([[0.2, 0.3, 0.4]], dtype=np.float32)
        if itimestep == 1 else cldfra_bl)
    np.testing.assert_array_equal(got_fraction, expected_fraction)


@pytest.mark.parametrize(
    ("mp_physics", "expected"),
    ((0, False), (1, False), (6, True), (8, True), (10, True), (18, True)),
)
def test_legacy_ice_activation_matches_wrf_registry_membership(
        mp_physics, expected):
    """Exercise multiple MP values on both sides of the F_QI/F_QS boundary."""

    assert legacy_ice_active(mp_physics) is expected


@requires_gpu
def test_nssl_legacy_mass_radius_and_cloud_fraction_match_wrf():
    """NSSL MP18 uses frozen mass, declared radii, and the WRF cloud oracle."""

    import cupy as cp

    from woof.core.rrtmgp import cal_cldfra1

    adapter_source = inspect.getsource(RRTMGLegacyRadiation.__call__)
    assert ("f_qi, f_qs = legacy_cloud_fraction_flags(mp_physics)"
            in adapter_source)
    # 2.9.1 exact chain: a default process converts the declared radii to
    # metres (a strict prep takes the microphysics microns directly).
    assert "radii = _radius_meters_blocks(radii)" in adapter_source
    assert "f_qi=f_qi, f_qs=f_qs" in adapter_source
    # NSSL declares both species, so the pair is (True, True) and this
    # oracle comparison is unchanged by the P3 split.
    assert legacy_cloud_fraction_flags(18) == (True, True)
    assert legacy_ice_active(18) is True
    assert _MP_DECLARES_RADII[18] is True
    effective_um = np.asarray([[12.0, 35.0, 90.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        legacy_radius_meters(effective_um),
        np.asarray([[12.0e-6, 35.0e-6, 90.0e-6]], dtype=np.float32),
    )

    from woof.core.rrtmg_legacy import _radius_meters_blocks
    converted = _radius_meters_blocks({"re_cloud": cp.asarray(effective_um)})
    np.testing.assert_array_equal(
        cp.asnumpy(converted["re_cloud"]).view(np.uint32),
        legacy_radius_meters(effective_um).view(np.uint32),
    )

    qv = np.asarray([[0.006, 0.004, 0.002]], dtype=np.float32)
    qc = np.zeros_like(qv)
    qi = np.asarray([[0.0, 2.0e-5, 0.0]], dtype=np.float32)
    qs = np.asarray([[1.0e-5, 3.0e-5, 4.0e-5]], dtype=np.float32)
    temperature = np.asarray([[270.0, 258.0, 245.0]], dtype=np.float32)
    pressure = np.asarray([[80000.0, 60000.0, 40000.0]], dtype=np.float32)
    oracle = _wrf_cal_cldfra1_frozen_oracle(
        qv, qc, qi, qs, temperature, pressure)
    got = cp.asnumpy(cal_cldfra1(
        cp.asarray(qv), cp.asarray(qc), cp.asarray(qi), cp.asarray(qs),
        cp.asarray(temperature), cp.asarray(pressure),
        f_qc=True, f_qi=legacy_cloud_fraction_flags(18)[0],
        f_qs=legacy_cloud_fraction_flags(18)[1],
    ))
    assert np.max(oracle) > 0.0
    np.testing.assert_allclose(got, oracle, rtol=2.0e-6, atol=2.0e-7)
