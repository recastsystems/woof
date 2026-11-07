"""The slab land surface (sf_surface_physics=1) against WRF's own Fortran.

``tools/slab_wrf461_oracle`` runs the byte-unmodified WRF v4.6.1
``phys/module_sf_slab.F`` (sha256 c44f2677...) at gfortran 13.3.0 -O0 /
glibc 2.39 over 96 columns for 36 consecutive surface calls each: day and
night, wet and dry, snow under IFSNOW 0/1/2, land and water side by side,
real.exe's water-to-ice columns and skins crossing the 273.16 K cap, soil
sub-step counts NSOIL 1, 2, 4, 5 and 8 (DELTSM 1 s to 300 s), and surface
pressure from 60 to 104 kPa.  The packaged copy is ``tests/data/oracles/slab``.

The bar is EVERY output word, every step, bitwise -- max ULP 0 and zero
differing bit patterns -- under ``GPUWM_WRF_EXACT=1``.  Default differences from the snow-cap fix are reported per field.

The exact arm covers IFSNOW=2 as well. The default caps its snowy top soil
layer, correcting WRF's different top-layer and skin snow-switch conditions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core import slab
from woof.verify import slab_oracle

ORACLE = slab_oracle.ORACLE_DIR
TOOL = Path(__file__).resolve().parents[1] / "tools/slab_wrf461_oracle"

#: Per-field differing-word counts of the DEFAULT arithmetic against the
#: oracle, all 96 cases, all 36 steps. Measured on G2
#: (RTX PRO 6000 Blackwell, NVRTC 12.9.86) on 2026-10-08. Equality, not a
#: bound: a default that drifted either way is a change someone must see.
DEFAULT_DIFFERING = dict(hfx=178, qfx=178, lh=178, capg=0, tsk=0,
                         qsfc=178, chklowq=0, tslb1=179, tslb2=178,
                         tslb3=178, tslb4=173, tslb5=0, flhc=0, flqc=0)


def _oracle():
    return slab_oracle.load()


def _admitted(oracle):
    return np.ones(len(oracle.cases), dtype=bool)


def test_fixture_is_the_recorded_build_output():
    provenance = (ORACLE / "provenance.txt").read_text(encoding="utf-8")
    for name in ("slab-input.bin", "slab-output.bin"):
        digest = hashlib.sha256((ORACLE / name).read_bytes()).hexdigest()
        assert f"{name} sha256 {digest}" in provenance
    assert ("module_sf_slab.F sha256 c44f26777b163234da249373b09a02fa92cb9532"
            "09bd35a6fe3c30c2158200cb") in provenance


def test_case_generator_reproduces_the_packaged_input(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "slab_generate_cases", TOOL / "generate_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases, zs, dzs = module.make_cases()
    module.write_input(tmp_path / "in.bin", cases, zs, dzs)
    assert ((tmp_path / "in.bin").read_bytes()
            == (ORACLE / "slab-input.bin").read_bytes())


def test_native_capture_accepts_both_fortran_stream_orders(tmp_path):
    expected = _oracle()
    for name in ("input", "output"):
        raw = (ORACLE / f"slab-{name}.bin").read_bytes()
        swapped = np.frombuffer(raw, dtype="u4").byteswap().tobytes()
        (tmp_path / f"native-{name}.bin").write_bytes(swapped)
    actual = slab_oracle.load_native_capture(
        tmp_path / "native-input.bin", tmp_path / "native-output.bin")
    for name in ("columns", "tslb", "zs", "dzs", "output"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.constants == expected.constants


def test_case_coverage_meets_the_check():
    oracle = _oracle()
    admitted = _admitted(oracle)
    columns = oracle.columns
    field = {name: k for k, name in enumerate(slab_oracle.COLUMN_FIELDS)}
    land = columns[:, field["xland"]] < 1.5
    assert admitted.sum() >= 64
    assert (land & admitted).sum() >= 48 and (~land & admitted).sum() >= 4
    assert (admitted & (columns[:, field["gsw"]] > 0)).any()
    assert (admitted & (columns[:, field["gsw"]] == 0)).any()
    assert (admitted & (columns[:, field["snowc"]] > 0)).sum() >= 10
    assert (admitted & (columns[:, field["flqc"]] == 0)).any()
    ice = land & (columns[:, field["tmn"]] == np.float32(271.4))
    assert ice.sum() >= 6
    # Threshold crossings the 36 steps actually reach: a snow column whose
    # top layer sits on the 273.16 K cap in some steps and below it in
    # others, and an ice column warming through 273.16 K.
    tslb1 = oracle.output[..., slab_oracle.OUTPUT_FIELDS.index("tslb1")]
    capped = (tslb1 == np.float32(273.16)).any(axis=1)
    below = (tslb1 < np.float32(273.16)).any(axis=1)
    assert (admitted & capped & below).any()
    assert {slab_nsoil(case["deltsm"], oracle.dzs[0])
            for case in oracle.cases} >= {1, 2, 4, 5, 8}


def slab_nsoil(deltsm, dzs1):
    f32 = np.float32
    return 1 + int(f32(f32(f32(f32(f32(1.25) * f32(4)) * f32(5.e-7)) / dzs1)
                       * f32(deltsm)) / dzs1)


def test_constants_are_wrfs():
    oracle = _oracle()
    for name, value in oracle.constants.items():
        if name == "karman" or name == "eomeg":
            # Arguments of SLAB that SLAB1D's five-layer arm never reads
            # (only the one-layer HM restoring term, unreachable in WRF).
            continue
        assert np.float32(slab.WRF_CONSTANTS[name]).view(np.uint32) \
            == np.float32(value).view(np.uint32), name


def test_soil_geometry_is_init_soil_depth_1():
    oracle = _oracle()
    zs, dzs = slab.soil_depths()
    np.testing.assert_array_equal(zs.view(np.uint32),
                                  oracle.zs.view(np.uint32))
    np.testing.assert_array_equal(dzs.view(np.uint32),
                                  oracle.dzs.view(np.uint32))


def test_wrf_snow_switch_defect_is_present_in_the_reference():
    oracle = _oracle()
    assert slab.require_admitted_ifsnow(2) == 2
    tsk = oracle.output[..., slab_oracle.OUTPUT_FIELDS.index("tsk")]
    tslb1 = oracle.output[..., slab_oracle.OUTPUT_FIELDS.index("tslb1")]
    split = np.array([np.abs(tsk[i] - tslb1[i]).max()
                      if case["ifsnow"] == 2 and "snow" in case["label"]
                      else 0.0 for i, case in enumerate(oracle.cases)])
    assert split.max() > 15.0


def test_comparator_catches_a_planted_one_ulp_change():
    oracle = _oracle()
    planted = oracle.output.copy()
    k = slab_oracle.OUTPUT_FIELDS.index("tslb2")
    planted[17, 20, k] = np.nextafter(planted[17, 20, k], np.float32(np.inf))
    report = slab_oracle.compare(planted, oracle.output)
    assert report["tslb2"] == {"words": report["tslb2"]["words"],
                               "differing": 1, "max_ulp": 1}
    assert all(row["differing"] == 0 for name, row in report.items()
               if name != "tslb2")


def _admitted_report(actual, oracle):
    keep = _admitted(oracle)
    return slab_oracle.compare(actual[keep], oracle.output[keep])


@pytest.mark.gpu
@requires_gpu
def test_cuda_column_matches_wrf_every_word_every_step():
    from woof.wrf_exact import ENABLED
    from woof.numerics import is_strict

    oracle = _oracle()
    keep = _admitted(oracle)
    trimmed = slab_oracle.SlabOracle(
        [c for c, k in zip(oracle.cases, keep) if k], oracle.columns[keep],
        oracle.tslb[keep], oracle.zs, oracle.dzs, oracle.nstep,
        oracle.constants, oracle.output[keep])
    actual = slab_oracle.run_cuda_columns(trimmed)
    report = slab_oracle.compare(actual, trimmed.output)
    print(json.dumps({"wrf_exact": ENABLED, "report": report}, indent=1))
    if ENABLED:
        assert all(row["differing"] == 0 and row["max_ulp"] == 0
                   for row in report.values()), report
    else:
        unchanged = np.array([c["ifsnow"] in (0, 1) for c in trimmed.cases])
        clean = slab_oracle.compare(actual[unchanged], trimmed.output[unchanged])
        if is_strict():
            assert all(r["differing"] == 0 for r in clean.values()), clean
            assert {name: row["differing"] for name, row in report.items()} == DEFAULT_DIFFERING, report
        else:
            # Fast arithmetic allows FMA and CUDA transcendentals. Keep
            # the native comparison with a physical error bound across all
            # 36 calls, while the opt-in strict run proves every word.
            a, e = actual[unchanged], trimmed.output[unchanged]
            np.testing.assert_allclose(a, e, rtol=5.e-4, atol=0.02)
            for k in (1, 5):  # moisture flux and saturation mixing ratio
                np.testing.assert_allclose(a[..., k], e[..., k], rtol=5.e-4, atol=1.e-7)
        snow = np.array([c["ifsnow"] == 2 and "snow" in c["label"]
                         for c in trimmed.cases])
        assert np.max(actual[snow, :, 7]) <= np.float32(273.16)
        assert report["tslb5"]["differing"] == 0, report


@pytest.mark.gpu
@requires_gpu
def test_a_planted_one_ulp_constant_is_caught_by_the_oracle():
    oracle = _oracle()
    keep = _admitted(oracle)
    trimmed = slab_oracle.SlabOracle(
        [c for c, k in zip(oracle.cases, keep) if k], oracle.columns[keep],
        oracle.tslb[keep], oracle.zs, oracle.dzs, oracle.nstep,
        oracle.constants, oracle.output[keep])
    planted = dict(slab.WRF_CONSTANTS)
    planted["xlv"] = np.nextafter(np.float32(planted["xlv"]),
                                  np.float32(np.inf))
    actual = slab_oracle.run_cuda_columns(trimmed, constants=planted)
    report = slab_oracle.compare(actual, trimmed.output)
    assert report["lh"]["differing"] > 0
    assert report["tsk"]["differing"] > 0
