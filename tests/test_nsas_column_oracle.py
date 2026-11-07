"""New SAS (cu_physics = 96) against WRF v4.6.1, word for word.

The referee is tools/nsas_wrf461_oracle: a harness linked against the
module_cu_nsas.o of the scalar WRF 4.6.1 build (the object inside the 0 ULP
referee wrf.exe), cross-checked byte for byte by two independent compiles of
the same source.  Its outputs ship as tests/data/oracles/nsas/*.npz:
672 columns at nz = 40 and 672 at nz = 60 -- deep, shallow-only, stable,
saturated, dry and edge families plus a seeded random sweep, each at
nsas_dx_factor 0 and 1, mixed microphysics classes, cudt steps 1/2/5.

What is graded, per column, in each arithmetic arm:

* the whole cu_nsas call (nsas_column): RTHCUTEN, RQVCUTEN, RQCCUTEN
  (where WRF forms it), RQICUTEN (likewise), RUCUTEN, RVCUTEN, RAINCV,
  PRATEC, HBOT (cloud base) and HTOP (cloud top), and the column prep;
* nsas2d alone and nscv2d alone, each from the oracle's own inputs to it.

The strict arm (``GPUWM_WRF_EXACT=1``) must be 0 ULP on every word.  The
default arm uses the same shared no-FTZ, no-contraction compiler route;
it is held to the measured bound below.  Each arm runs in its own process
because the arm is chosen before the first kernel compiles.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from _run_config_fake import run_config_fake
from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "data" / "oracles" / "nsas"

#: Default-arm bound, MEASURED on an RTX PRO 6000 Blackwell (sm_120,
#: NVRTC 12.9.86): the largest distance any default-arm word showed from
#: the WRF word across both fixture sets.  Both arms preserve subnormals; see REPORT in the port record.
DEFAULT_MAX_ULP = 0


def _compare(*extra, strict: bool) -> dict:
    env = dict(os.environ)
    env.pop("GPUWM_WRF_EXACT", None)
    if strict:
        env["GPUWM_WRF_EXACT"] = "1"
    out = subprocess.run(
        [sys.executable, "-m", "woof.verify.nsas_compare", *extra],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _misses(doc):
    return {(run["nz"], run["row_width"], name): metric
            for run in doc["runs"] for name, metric in run["fields"].items()
            if metric["different_words"]}


def test_fixture_digests_match():
    lines = (FIXTURES / "oracle-sha256sums.txt").read_text(
        encoding="utf-8").splitlines()
    seen = 0
    for line in lines:
        digest, name = line.split(None, 1)
        if name.startswith("raw/"):
            continue
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() \
            == digest, name
        seen += 1
    assert seen == 2


def test_workspace_slot_count_matches_the_table():
    from woof.core.column_cumulus import COLUMN_CUMULUS_SCHEMES
    src = (ROOT / "woof" / "core" / "kernels" / "nsas.cu").read_text(
        encoding="utf-8")
    declared = int(re.search(r"#define NSAS_WS_SLOTS (\d+)", src).group(1))
    assert declared == COLUMN_CUMULUS_SCHEMES[96].workspace_slots


def test_fixtures_exercise_every_branch_family():
    """The cases are worth grading only if both routines actually fire."""
    import numpy as np
    from woof.verify.nsas_compare import load
    for nz in (40, 60):
        fx = load(nz)
        single = fx["hdr_i"][:, 2] == 1
        deep = fx["out.deep.icps"][single, 0] == 1
        shallow = (fx["out.shallow.ktop"][single, 0] > 0) & ~deep
        dxf1 = fx["hdr_i"][single, 5] == 1
        assert deep.sum() > 300 and shallow.sum() > 60
        assert (deep & dxf1).sum() > 100 and (deep & ~dxf1).sum() > 100
        assert (fx["out.shallow.rain"][single, 0] > 0).sum() > 30
        assert np.isfinite(fx["out.driver.rthcuten"]).all()


def test_row_coupling_referee_exercises_neighbour_dependence():
    """WRF's nsas2d takes kbmax/kbm/kmax over the whole i-row.

    The port reduces the same limits over each complete row; this pins
    that WRF's own answer for a column really does move when a neighbour
    with a different surface pressure shares its row, so a launcher that omits the reduction cannot pass the row oracle.
    """
    from woof.verify.nsas_compare import row_coupling
    moved = [row_coupling(nz) for nz in (40, 60)]
    assert all(m["row_columns"] == 24 for m in moved)
    assert sum(m["columns_moved_by_row"] for m in moved) > 0


@requires_gpu
def test_strict_arm_is_bitwise_on_every_word():
    doc = _compare(strict=True)
    assert doc["strict"] is True
    assert sum(run["columns"] for run in doc["runs"]) == 2 * 672
    assert not _misses(doc), _misses(doc)


@requires_gpu
def test_default_arm_within_measured_bound():
    doc = _compare(strict=False)
    assert doc["strict"] is False
    worst = max(m["max_ulp"] for run in doc["runs"]
                for m in run["fields"].values())
    assert worst <= DEFAULT_MAX_ULP, _misses(doc)


#: One-bit plants: the last mantissa bit of a closure or an entrainment
#: constant, in the deep and the shallow routine.
_PLANTS = {
    "deep closure (mbdt)": "NSAS_MBDT=0x41200001",
    "deep entrainment (clam)": "NSAS_CLAM=0x3DCCCCCC",
    "deep organized entrainment (cxlamu)": "NSAS_CXLAMU=0x38D1B716",
    "shallow closure (betaw)": "NSCV_BETAW=0x3CF5C28E",
    "shallow entrainment (clam)": "NSCV_CLAM=0x3E99999B",
}


@requires_gpu
@pytest.mark.parametrize("label", sorted(_PLANTS))
def test_planted_one_bit_change_is_caught(label):
    doc = _compare("--plant", _PLANTS[label], strict=True)
    assert _misses(doc), f"{label}: a one-bit plant left every word equal"



@requires_gpu
def test_production_launcher_returns_wrfs_held_rate():
    import numpy as np
    import cupy as cp
    from types import SimpleNamespace
    from woof.core.column_cumulus import ColumnCumulus
    from woof.verify.nsas_compare import load
    fx = load(40)
    i = int(np.nonzero((fx["hdr_i"][:, 2] == 1)
                      & (fx["out.driver.pratec"][:, 0] > 0)
                      & (fx["hdr_i"][:, 4] == 6))[0][0])
    shape = (40, 1, 1)
    def array(name, levels=40):
        return cp.asarray(fx[f"in.{name}"][i].reshape(levels, 1, 1))
    atmosphere = dict(zip(("temperature", "qv", "qc", "qi", "pressure",
        "exner", "rho", "dz", "u", "v"),
        [array(n) for n in ("t", "qv", "qc", "qi", "p", "pi", "rho",
                            "dz8w", "u", "v")]))
    atmosphere["p_interface"] = array("p8w", 41)
    state = SimpleNamespace(p=atmosphere["pressure"], w=array("w", 41),
                            qc=atmosphere["qc"], qi=atmosphere["qi"])
    h = fx["hdr_f"][i]
    fields = {n: cp.full((1, 1), h[j], cp.float32)
              for n, j in (("xland", 3), ("pblh", 4), ("hfx", 5), ("qfx", 6))}
    cfg = run_config_fake(dx=float(h[1]), dt=float(h[0]), clock_dt=0.,
                          sas_pgcon=float(h[2]), mp_physics=6,
                          nsas_dx_factor=int(fx["hdr_i"][i, 5]))
    launcher = ColumnCumulus(96)
    launcher.bind_driver(SimpleNamespace(stepcu=int(fx["hdr_i"][i, 6])))
    result = launcher(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    assert result.persistent_pratec and result.rainc is None
    for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten"):
        cp.testing.assert_array_equal(getattr(result, name),
              cp.asarray(fx[f"out.driver.{name}"][i].reshape(shape)))
    cp.testing.assert_array_equal(result.pratec,
              cp.asarray(fx["out.driver.pratec"][i].reshape(1, 1)))


@requires_gpu
def test_persistent_rate_accumulates_each_step_and_keeps_momentum():
    import cupy as cp
    from test_physics_driver import _full_state
    from woof.core.physics import CumulusResult
    calls = []
    def cumulus(**kwargs):
        nz, ny, nx = kwargs["state"].p.shape
        z = cp.zeros((nz, ny, nx), cp.float32)
        calls.append(1)
        return CumulusResult(z, z, rucuten=cp.full_like(z, 1.e-5),
            rvcuten=cp.full_like(z, -1.e-5),
            pratec=cp.full((ny, nx), 0.125, cp.float32), persistent_pratec=True)
    state, cfg, driver = _full_state(nz=40, dt=60., cu_physics=96,
        cudt_minutes=5., sf_sfclay_physics=0, sf_surface_physics=0,
        bl_pbl_physics=0, cumulus=cumulus)
    cp.testing.assert_array_equal(driver.cu_pratec, 0.)
    for rate in driver.cu_rates.values():
        cp.testing.assert_array_equal(rate, 0.)
    got = driver.compute(state, cfg)
    assert bool(cp.all(got.ru > 0)) and bool(cp.all(got.rv < 0))
    cp.testing.assert_array_equal(driver.rainc, 7.5)
    state.elapsed_seconds = 60.
    again = driver.compute(state, cfg)
    cp.testing.assert_array_equal(again.ru, got.ru)
    cp.testing.assert_array_equal(driver.rainc, 15.)
    assert len(calls) == 1



def test_stage_preparation_has_level_major_kernel_layout():
    import numpy as np
    from woof.verify.nsas_compare import load, stage_preparation
    from woof.verify.nsas_oracle import PREP_LEV
    fx = load(40)
    idx = np.nonzero(fx["hdr_i"][:, 2] == 1)[0]
    packed = stage_preparation(fx, idx)
    assert packed.flags.c_contiguous
    for j, name in enumerate(PREP_LEV):
        np.testing.assert_array_equal(packed[j*40:(j+1)*40].T,
                                      fx[f"out.prep.{name}"][idx])


@requires_gpu
def test_shared_no_ftz_compiler_preserves_subnormal_product():
    import numpy as np
    import cupy as cp
    from woof.core.kernels import compile_module_no_ftz, module_options
    source = 'extern "C" __global__ void probe(const float* a, float* b) { b[0] = a[0]*a[1]; }'
    mod = compile_module_no_ftz("nsas_probe", source, module_options("nsas"))
    a = cp.asarray(np.array([1.e-20, 1.e-20], np.float32))
    b = cp.empty(1, cp.float32)
    mod.get_function("probe")((1,), (1,), (a, b))
    cp.testing.assert_array_equal(b, np.array([np.float32(1.e-20)*np.float32(1.e-20)], np.float32))



def test_mutation_constants_are_exactly_one_bit_flips():
    import numpy as np
    from woof.core.constants import CUDA_UNIT_DEFINES
    for spec in _PLANTS.values():
        name, bits = spec.split("=")
        base = int(np.float32(CUDA_UNIT_DEFINES["nsas"][name]).view(np.uint32))
        assert (base ^ int(bits, 16)).bit_count() == 1, spec


def test_constants_and_closure_tables_are_seeded_from_the_shared_table_path():
    from woof.core.constants import CUDA_UNIT_DEFINES, CUDA_UNIT_TABLES
    from woof.core.kernels import module_source
    src = module_source("nsas")
    for name in CUDA_UNIT_DEFINES["nsas"]:
        assert f"#define {name} " in src
    for name, values in CUDA_UNIT_TABLES["nsas"].items():
        assert len(values) == 15
        assert f"__constant__ float {name}[]" in src
