"""The GFS surface layer (sf_sfclay_physics=3) against WRF v4.6.1, CPU tier.

The fixture of record is tools/sfgfs_wrf461_oracle's output, kept in
tests/data/oracles/sfgfs (test-only, in no wheel): 66 named columns (unstable and stable land, open
sea, calm, rough, very unstable, cold and dry, partial and full sea ice)
over six steps that carry ZNT, UST, SST and the *_SEA carriers, in four arms
(direct call; the FRACTIONAL_SEAICE wrapper as shipped and as repaired;
ISFFLX = 0), plus the FPVS table GFUNCPHYS built and an FPVS sweep.

The CPU reference (woof.verify.sfgfs_ref) is graded word for word here
where the host C library is glibc, the oracle's libm.  The product kernel
is graded on the card by tests/test_sfgfs_launcher_wrf461_parity.py.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from woof.verify import sfgfs_ref as ref
from woof.verify.sfgfs_oracle import (FIXTURE_DIR, FPVS_TABLE_SHA256,
                                       declared_divergence_columns,
                                       load_sfgfs_oracle, reference_arm,
                                       table_digest, ulp_distance)

#: SHA-256 of every fixture file, as build.sh wrote them on the sprint box
#: (tools/sfgfs_wrf461_oracle, WRF v4.6.1 d66e442, gfortran 13.3.0,
#: glibc 2.39).  A regenerated fixture changes these on purpose or not at
#: all.
FIXTURE_SHA256 = {
    "armA.bin": "b3d48bc7dd38435d5a875643e14b30759a8bb9af59115b3455da72ae5f631553",
    "armB.bin": "21b7357c868f0508f9f0751851f6fa454818eb9271eee4c257d84ee9a6d1269d",
    "armC.bin": "ba1c39b0365d2af7d9c9f996659bee27d6f9c09f47689d438f1fc22d82fa4635",
    "armD.bin": "0c8180af974859fd68b5800f23b9b20ab60af2eeb0c6bb02015413fda0900185",
    "cases.bin": "0b1d1bb794870fa9572e8b1bbeefe37fea22132df181e24da9596a415c353859",
    "fields.txt": "f741cd4d7aea80f59cc748ec878fba6c60a60885f50597e0627b0bf5718d2a5a",
    "fpvs_sweep.bin": "38d6d1edf60629c36048b035db7ee9e7361a526619e7b3f87e5677d7b5c23b05",
    "fpvs_table.bin": "bef4a54b636c57f74951560d3f5f0a65e12337cbedeae1ac17be1376f66e2235",
}

needs_glibc = pytest.mark.skipif(
    not ref.host_libm_is_oracle_libm(),
    reason="the CPU reference calls the host libm; only glibc's exp/log/pow "
           "are the oracle's words (the card kernel carries its own)")


def test_the_fixture_is_the_one_the_oracle_wrote():
    for name, digest in FIXTURE_SHA256.items():
        got = hashlib.sha256((FIXTURE_DIR / name).read_bytes()).hexdigest()
        assert got == digest, name


def test_the_fixture_covers_the_regimes_the_brief_names():
    fx = load_sfgfs_oracle()
    assert fx.ncol >= 64 and fx.nstep >= 2
    xland = fx.per_step[0, 7]
    xice = fx.per_step[0, 8]
    rb = fx.field("A", "br")
    land = (xland < 1.5) & (xice == 0)
    sea = (xland > 1.5) & (xice == 0)
    assert np.any(land & (rb[0] < 0)) and np.any(land & (rb[0] > 0))
    assert np.any(sea & (rb[0] < 0)) and np.any(sea & (rb[0] > 0))
    assert np.any((xice > 0.02) & (xice < 1.0))      # partial ice
    assert np.any(fx.field("A", "wspd")[0] == np.float32(1.0))   # calm
    assert np.any(fx.initial[0] >= 2.0)               # rough
    # ZNT really carries: some sea column's roughness moves between steps.
    znt = fx.field("A", "znt")
    assert np.any(sea & (znt[0] != znt[-1]))


@needs_glibc
def test_the_fpvs_table_is_gfuncphys_words():
    fx = load_sfgfs_oracle()
    assert table_digest(fx.fpvs_table) == FPVS_TABLE_SHA256
    built = ref.gpvs()
    assert np.array_equal(built.view(np.uint64),
                          fx.fpvs_table.view(np.uint64))


@needs_glibc
def test_fpvs_interpolation_is_wrfs_on_the_sweep():
    fx = load_sfgfs_oracle()
    got = np.asarray([ref.fpvs(fx.fpvs_table, float(t)) for t in fx.sweep_t])
    assert np.array_equal(got.view(np.uint64), fx.sweep_fpvs.view(np.uint64))


@needs_glibc
@pytest.mark.parametrize("arm", ["A", "C", "D"])
def test_every_output_word_of_every_step_is_wrfs(arm):
    fx = load_sfgfs_oracle()
    got = reference_arm(fx, arm, fx.fpvs_table)
    want = fx.arms[arm]
    bad = {name: int(ulp_distance(got[:, f], want[:, f]).max())
           for f, name in enumerate(fx.fields(arm))
           if not np.array_equal(got[:, f].view(np.uint32),
                                 want[:, f].view(np.uint32))}
    assert not bad, f"arm {arm}: max ULP by field {bad}"


@needs_glibc
def test_the_shipped_wrapper_differs_only_on_the_declared_columns():
    """WRF's open-water call passes XLAND, not XLAND_SEA (declared)."""
    fx = load_sfgfs_oracle()
    shipped = reference_arm(fx, "B", fx.fpvs_table)
    assert np.array_equal(shipped.view(np.uint32),
                          fx.arms["B"].view(np.uint32))
    differs = np.nonzero(np.any(fx.arms["B"] != fx.arms["C"], axis=(0, 1)))[0]
    declared = declared_divergence_columns(fx)
    assert set(differs.tolist()) <= set(declared.tolist())
    assert differs.size > 0


@needs_glibc
def test_a_planted_one_bit_change_in_a_constant_is_caught(monkeypatch):
    fx = load_sfgfs_oracle()
    flipped = np.frombuffer(
        (np.float32(0.014).view(np.uint32) ^ np.uint32(1)).tobytes(),
        dtype=np.float32)[0]
    real_F = ref.F

    def planted(x):
        if np.float32(x) == np.float32(0.014):
            return float(flipped)
        return real_F(x)

    monkeypatch.setattr(ref, "F", planted)
    got = reference_arm(fx, "A", fx.fpvs_table)
    assert not np.array_equal(got.view(np.uint32),
                              fx.arms["A"].view(np.uint32))
