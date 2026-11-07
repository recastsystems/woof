"""The GFS surface layer's PRODUCT launcher against WRF v4.6.1, bit for bit.

``woof.core.sf_gfs.launch_gfs_sfclay`` is what ``PhysicsDriver.
_run_gfs_sfclay`` calls: the unit as the loader assembles it
(``_EXTRA_HEADERS["sf_gfs"]``), the manifest-checked FPVS table uploaded to the card from
GFUNCPHYS's words, and both entry points.  This module
drives it on every column of every step of the oracle's fixture of record
(tests/test_sfgfs_wrf461_parity.py pins it) and compares every float32
output word for EQUALITY: arm A (direct), arm C (the FRACTIONAL_SEAICE
wrapper with the XLAND_SEA repair WOOF ships) and arm D (ISFFLX = 0).
Arm B, WRF's shipped wrapper, is the declared divergence: the launcher
must differ from it only on the sea-ice cells real.exe marked XLAND = 1.

Split from the CPU tier because cupy marks a whole module ``gpu``.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.verify.sfgfs_oracle import (FPVS_TABLE_SHA256,
                                       WRAPPER_XICE_THRESHOLD,
                                       declared_divergence_columns,
                                       load_sfgfs_oracle, table_digest,
                                       ulp_distance)


def _run_arm(fx, arm, *, module=None):
    import cupy as cp

    from woof.core import sf_gfs

    names = fx.fields(arm)
    out = np.zeros((fx.nstep, len(names), fx.ncol), dtype=np.float32)
    shape = (1, fx.ncol)

    def dev(a):
        return cp.asarray(np.ascontiguousarray(a, dtype=np.float32)
                          .reshape(shape))

    state = {"znt": dev(fx.initial[0]), "ust": dev(fx.initial[1])}
    outputs = {name: cp.zeros(shape, dtype=cp.float32)
               for name in sf_gfs.GFS_SFCLAY_OUTPUTS}
    seaice = None
    if arm in ("B", "C"):
        seaice = {name: cp.zeros(shape, dtype=cp.float32)
                  for name in sf_gfs.GFS_SEA_CARRIERS}
        seaice["sst"] = dev(fx.initial[2])
        seaice["xice_threshold"] = WRAPPER_XICE_THRESHOLD
    for step in range(fx.nstep):
        raw = fx.step_inputs(step)
        inputs = {name: dev(raw[name]) for name in sf_gfs.GFS_SFCLAY_INPUTS}
        if seaice is not None:
            seaice["xice"] = dev(raw["xice"])
            seaice["itimestep"] = step + 1
        sf_gfs.launch_gfs_sfclay(inputs, state, outputs,
                                 isfflx=0 if arm == "D" else 1,
                                 seaice=seaice)
        row = dict(outputs, **state)
        if seaice is not None:
            row.update(seaice)
            row["tsk_sea"] = None
        for f, name in enumerate(names):
            if name == "tsk_sea":
                # The repaired wrapper's open-water skin temperature: SST on
                # sea-ice cells (published to the carrier), TSK elsewhere.
                ice = ((raw["xice"] >= WRAPPER_XICE_THRESHOLD)
                       & (raw["xice"] <= np.float32(1.0)))
                out[step, f] = np.where(ice, cp.asnumpy(seaice["sst"])[0],
                                        raw["tsk"])
            else:
                out[step, f] = cp.asnumpy(row[name])[0]
    return out


def _mismatch(got, want, names):
    return {name: int(ulp_distance(got[:, f], want[:, f]).max())
            for f, name in enumerate(names)
            if not np.array_equal(got[:, f].view(np.uint32),
                                  want[:, f].view(np.uint32))}


@pytest.mark.gpu
def test_the_packaged_gfuncphys_table_is_word_for_word_on_card():
    import cupy as cp

    from woof.core import sf_gfs

    fx = load_sfgfs_oracle()
    table = cp.asnumpy(sf_gfs.fpvs_table())
    assert table_digest(table) == FPVS_TABLE_SHA256 == sf_gfs.FPVS_TABLE_SHA256
    assert np.array_equal(table.view(np.uint64), fx.fpvs_table.view(np.uint64))


@pytest.mark.gpu
@pytest.mark.parametrize("arm", ["A", "C", "D"])
def test_every_output_word_of_every_step_is_wrfs(arm):
    import cupy  # noqa: F401

    fx = load_sfgfs_oracle()
    got = _run_arm(fx, arm)
    bad = _mismatch(got, fx.arms[arm], fx.fields(arm))
    assert not bad, f"arm {arm}: max ULP by field {bad}"


@pytest.mark.gpu
def test_the_shipped_wrapper_is_the_declared_divergence_and_nothing_else():
    import cupy  # noqa: F401

    fx = load_sfgfs_oracle()
    got = _run_arm(fx, "C")
    differs = np.nonzero(np.any(
        got.view(np.uint32) != fx.arms["B"].view(np.uint32), axis=(0, 1)))[0]
    assert differs.size > 0
    assert set(differs.tolist()) <= set(
        declared_divergence_columns(fx).tolist())


@pytest.mark.gpu
def test_the_launch_constants_are_the_references():
    from woof.core import sf_gfs
    from woof.verify import sfgfs_ref

    k = sfgfs_ref.wrf_constants()
    for name, value in sf_gfs.wrf_surface_constants().items():
        assert np.float32(getattr(k, name)).view(np.uint32) == \
            np.float32(value).view(np.uint32), name


@pytest.mark.gpu
def test_a_planted_one_bit_change_in_one_kernel_constant_is_caught():
    """Flip the last bit of the Charnock constant in the unit's source."""
    import cupy as cp

    from woof.core import sf_gfs
    from woof.core.kernels import module_options, module_source

    src = module_source("sf_gfs")
    original = "charnock = F(.014f)"
    assert src.count(original) == 1
    bits = int(np.float32(0.014).view(np.uint32)) ^ 1
    planted = float(np.uint32(bits).view(np.float32))
    mutated = src.replace(original, f"charnock = F({planted!r}f)")
    module = cp.RawModule(code=mutated, options=module_options("sf_gfs"))
    fx = load_sfgfs_oracle()
    shape = (1, fx.ncol)
    raw = fx.step_inputs(0)

    def dev(a):
        return cp.asarray(np.ascontiguousarray(a, dtype=np.float32)
                          .reshape(shape))

    znt, ust = dev(fx.initial[0]), dev(fx.initial[1])
    outs = [cp.zeros(shape, dtype=cp.float32)
            for _ in sf_gfs.GFS_SFCLAY_OUTPUTS]
    k = sf_gfs.wrf_surface_constants()
    kernel = module.get_function("gfs_sfclay")
    kernel((1,), (128,),
           (np.int32(fx.ncol),
            *(dev(raw[name]) for name in sf_gfs.GFS_SFCLAY_INPUTS),
            znt, ust, *outs, sf_gfs.fpvs_table(), k["cp"], k["rovcp"],
            k["r"], k["xlv"], k["ep1"], k["ep2"], k["karman"], np.int32(1)))
    # Charnock reaches step one through the sea columns' roughness update
    # and the Zeng thermal roughness (CHS2); UST itself is rebuilt from CM,
    # which the INPUT roughness sets, so it is not the field to watch.
    got = dict(zip(sf_gfs.GFS_SFCLAY_OUTPUTS, (cp.asnumpy(o)[0] for o in outs)))
    got["znt"] = cp.asnumpy(znt)[0]
    moved = [name for name, value in got.items()
             if not np.array_equal(value.view(np.uint32),
                                   fx.field("A", name)[0].view(np.uint32))]
    assert "znt" in moved and "chs2" in moved, moved
    # ... and only the sea columns move: the constant is the Charnock one.
    sea = fx.per_step[0, 7] > np.float32(1.5)
    assert np.array_equal(got["znt"][~sea].view(np.uint32),
                          fx.field("A", "znt")[0][~sea].view(np.uint32))


@pytest.mark.gpu
def test_a_sliced_launch_equals_a_whole_one():
    """Column independence: two half-domain launches give the same words."""
    import cupy as cp

    from woof.core import sf_gfs

    fx = load_sfgfs_oracle()
    raw = fx.step_inputs(0)
    half = fx.ncol // 2

    def run(sl):
        shape = (1, sl.stop - sl.start)

        def dev(a):
            return cp.asarray(np.ascontiguousarray(a[sl], dtype=np.float32)
                              .reshape(shape))
        state = {"znt": dev(fx.initial[0]), "ust": dev(fx.initial[1])}
        outputs = {name: cp.zeros(shape, dtype=cp.float32)
                   for name in sf_gfs.GFS_SFCLAY_OUTPUTS}
        sf_gfs.launch_gfs_sfclay(
            {name: dev(raw[name]) for name in sf_gfs.GFS_SFCLAY_INPUTS},
            state, outputs, isfflx=1)
        return {name: cp.asnumpy(v)[0] for name, v in
                dict(outputs, **state).items()}

    whole = run(slice(0, fx.ncol))
    first, second = run(slice(0, half)), run(slice(half, fx.ncol))
    for name, value in whole.items():
        joined = np.concatenate([first[name], second[name]])
        assert np.array_equal(value.view(np.uint32), joined.view(np.uint32))
