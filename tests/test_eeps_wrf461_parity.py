"""The EEPS PBL (bl_pbl_physics=16) on the card, graded word for word.

Fixtures: ``tests/data/oracles/eeps`` (PROVENANCE.md there), WRF v4.6.1's
own ``module_bl_eepsilon.F`` driven by ``tools/eeps_wrf461_oracle`` at the
scalar WRF build's flags: 72 columns in eight regimes, three column sets
(dt 60 s with every moisture argument, dt 180 s without QS/QG, and a
restart-shaped set entered at itimestep 7 from carried TKE/epsilon), every
call's inputs recorded beside its outputs.

The replay carries its OWN state: after the first call of a set, PEK_PBL,
PEP_PBL, PEK_ADV and PEP_ADV come from the kernel's previous outputs, with
the oracle's fixed stand-in for advection applied to the advected pair in
float32 (``woof.verify.eeps_oracle.blend_carried``).  Only the
atmosphere is taken from the recording.  So a word lost anywhere in the
PEK/PEP handoff surfaces in the next call, not just in the call that lost
it.

Every output word must match: the six tendencies, PEK/PEP, PEK_ADV/PEP_ADV,
EXCH_H/EXCH_M on WRF's kts..kte+1 levels, PBLH and KPBL. Strict numerics
(GPUWM_WRF_EXACT=1) is required for the bitwise grade. These fixtures do
not qualify the default fast numerics or a real forecast trajectory.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof.core import eeps_pbl  # noqa: E402
from woof.verify.eeps_oracle import (blend_carried, read_trace,  # noqa: E402
                                      ulp_distance)

ROOT = Path(__file__).resolve().parents[1]
ORACLE = Path(os.environ.get("EEPS_ORACLE_DIR")
              or ROOT / "tests/data/oracles/eeps")
SETS = ("A", "B", "C")
COMPARED = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
            "rqiblten", "pek", "pep", "pek_adv", "pep_adv", "exch_h",
            "exch_m", "pblh", "kpbl")


def _trace(name):
    for candidate in (ORACLE / f"trace-{name}.bin.gz",
                      ORACLE / f"trace-{name}.bin"):
        if candidate.exists():
            return read_trace(candidate)
    pytest.skip(f"EEPS oracle trace {name} is absent under {ORACLE}")


def _dev(a):
    return cp.ascontiguousarray(cp.asarray(a, dtype=cp.float32)[:, None, :]
                                if a.ndim == 2 else
                                cp.asarray(a, dtype=cp.float32)[None, :])


def replay(trace, kernel=None):
    """Run every call of one recorded set; return per-step mismatch counts.

    The result maps step index -> {field: (words differing, max ULP)}.
    """
    steps = trace["steps"]
    first = steps[0]["inputs"]
    carried = {"pek_pbl": _dev(first["pek"]), "pep_pbl": _dev(first["pep"]),
               "pek_adv": _dev(first["pek_adv"]),
               "pep_adv": _dev(first["pep_adv"])}
    report = {}
    for index, step in enumerate(steps):
        inp, want = step["inputs"], step["outputs"]
        columns = {name: _dev(inp[key]) for name, key in (
            ("u", "u"), ("v", "v"), ("t", "t"), ("qv", "qv"), ("qc", "qc"),
            ("qi", "qi"), ("qr", "qr"), ("qs", "qs"), ("qg", "qg"),
            ("p", "p"), ("pi", "pi"), ("rho", "rho"),
            ("rthraten", "rthraten"), ("dz8w", "dz8w"))}
        surface = {name: _dev(inp[name]).reshape(1, -1)
                   for name in eeps_pbl.SURFACE_INPUTS}
        out = eeps_pbl.eeps_step(columns, surface, carried,
                                 dt=trace["header"]["dt"],
                                 first_step=(step["itimestep"] == 1),
                                 kernel=kernel)
        got = {name: cp.asnumpy(out[name])[:, 0, :]
               for name in eeps_pbl.TENDENCY_OUTPUTS}
        for name in eeps_pbl.FULL_LEVEL_OUTPUTS:
            got[name] = cp.asnumpy(out[name])[:, 0, :]
        got["pek"] = cp.asnumpy(carried["pek_pbl"])[:, 0, :]
        got["pep"] = cp.asnumpy(carried["pep_pbl"])[:, 0, :]
        got["pek_adv"] = cp.asnumpy(carried["pek_adv"])[:, 0, :]
        got["pep_adv"] = cp.asnumpy(carried["pep_adv"])[:, 0, :]
        got["pblh"] = cp.asnumpy(out["pblh"])[0]
        got["kpbl"] = cp.asnumpy(out["kpbl"])[0]
        fields = {}
        for name in COMPARED:
            if name == "kpbl":
                diff = got[name] != want[name]
                fields[name] = (int(diff.sum()),
                                int(np.abs(got[name] - want[name]).max()))
            else:
                ulp = ulp_distance(got[name], want[name])
                # A word grade includes signed zero. ULP distance alone
                # maps +0 and -0 to the same value and would miss a bit.
                diff = got[name].view(np.uint32) != want[name].view(np.uint32)
                fields[name] = (int(diff.sum()), int(ulp.max()))
        report[index] = fields
        # The handoff the dycore performs between calls, from OUR outputs.
        carried["pek_adv"] = _dev(blend_carried(got["pek_adv"]))
        carried["pep_adv"] = _dev(blend_carried(got["pep_adv"]))
    return report


@pytest.mark.parametrize("name", SETS)
@pytest.mark.skipif(os.environ.get("GPUWM_WRF_EXACT") != "1",
                    reason="native word grade requires GPUWM_WRF_EXACT=1")
def test_every_call_of_every_column_is_wrfs_words(name):
    report = replay(_trace(name))
    bad = {(step, field): counts for step, fields in report.items()
           for field, counts in fields.items() if counts[0]}
    assert not bad, f"set {name}: (step, field) -> (words, max ULP) {bad}"


def test_the_handoff_reproduces_the_oracles_next_inputs():
    """Our carried pair after the stand-in equals what WRF's next call read."""
    trace = _trace("A")
    steps = trace["steps"]
    for prev, nxt in zip(steps, steps[1:]):
        for name in ("pek_adv", "pep_adv"):
            ours = blend_carried(prev["outputs"][name])
            assert np.array_equal(ours.view(np.uint32),
                                  nxt["inputs"][name].view(np.uint32)), name


def test_a_planted_one_bit_constant_is_caught():
    """Flip the last bit of c2 (0.09) in the column and the replay sees it."""
    from woof import nvrtc_ptx_cache as compiler

    trace = _trace("A")
    nz = trace["header"]["nz"]
    source = eeps_pbl.module_source(nz)
    planted_c2 = np.float32(0.09).view(np.uint32) ^ np.uint32(1)
    line = "#define EEPS_C2 0.09f"
    assert source.count(line) == 1
    source = source.replace(
        line, f"#define EEPS_C2 __int_as_float(0x{int(planted_c2):08X})")
    from woof.wrf_exact import effective_options
    ptx, _ = compiler.compile_using_nvrtc(
        source, effective_options(eeps_pbl.MODULE_OPTIONS), None, "eeps_planted.cu")
    module = cp.cuda.function.Module()
    module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    report = replay(trace, kernel=module.get_function("eeps_column"))
    caught = sum(counts[0] for fields in report.values()
                 for counts in fields.values())
    assert caught > 0


def test_a_column_split_across_launch_shapes_is_the_same_words():
    """Columns are independent: a (ny, nx) launch equals the flat one."""
    trace = _trace("B")
    step = trace["steps"][0]
    inp = step["inputs"]
    nz, ncol = inp["u"].shape
    shape2 = (8, ncol // 8)

    def run(shape):
        cols = {name: cp.ascontiguousarray(
            cp.asarray(inp[key], dtype=cp.float32).reshape(nz, *shape))
            for name, key in zip(eeps_pbl.COLUMN_INPUTS, (
                "u", "v", "t", "qv", "qc", "qi", "qr", "qs", "qg", "p", "pi",
                "rho", "rthraten", "dz8w"))}
        sfc = {name: cp.ascontiguousarray(
            cp.asarray(inp[name], dtype=cp.float32).reshape(shape))
            for name in eeps_pbl.SURFACE_INPUTS}
        car = {"pek_pbl": inp["pek"], "pep_pbl": inp["pep"],
               "pek_adv": inp["pek_adv"], "pep_adv": inp["pep_adv"]}
        car = {k: cp.ascontiguousarray(cp.asarray(v, dtype=cp.float32)
                                       .reshape(nz, *shape))
               for k, v in car.items()}
        out = eeps_pbl.eeps_step(cols, sfc, car, dt=trace["header"]["dt"],
                                 first_step=True)
        return {k: cp.asnumpy(v).reshape(v.shape[0] if v.ndim == 3 else 1,
                                         -1)
                for k, v in out.items()}

    flat = run((1, ncol))
    split = run(shape2)
    for name in flat:
        assert np.array_equal(flat[name].view(np.uint32),
                              split[name].view(np.uint32)), name


@pytest.mark.skipif(os.environ.get("GPUWM_WRF_EXACT") != "1",
                    reason="native word grade requires GPUWM_WRF_EXACT=1")
def test_workspace_reuse_preserves_native_words_across_partial_tiles(monkeypatch):
    """A last partial block must not read or overwrite another tile's columns."""
    trace = _trace("A")
    trace = {**trace, "steps": [trace["steps"][0]]}
    step = trace["steps"][0]
    trace["steps"] = [{**step, **{section: {
        name: np.concatenate([array, array, array], axis=-1)
        for name, array in step[section].items()}
        for section in ("inputs", "outputs")}}]
    monkeypatch.setattr(eeps_pbl, "tile_columns", lambda fn, n: 128)
    report = replay(trace)
    assert all(counts[0] == 0 for fields in report.values()
               for counts in fields.values()), report
