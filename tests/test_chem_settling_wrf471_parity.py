"""GOCART gravitational settling against WRF v4.7.1, bit for bit.

``tools/chem_wrf471_oracle/gocart/build.sh`` compiles the byte-unmodified
``chem/module_gocart_settling.F`` (gfortran 15.2.0 -O0, glibc 2.43) and
``run_gocart_settling.F90`` drives ``gocart_settling_driver`` for both
``chem_opt`` arms (300 GOCART_SIMPLE, 401 DUST), two growth groups (the five
dust bins with no growth, the four sea-salt bins with Gerber growth), three
time steps (non-integer ``dt`` included, so ``INT(dt)`` truncates) and three
consecutive steps carried on the same fields: 36 cases.  The port is the
table-driven ``chem_settling_gocart`` kernel handed the rows' radius, density
and growth arm.

Measured on the test host 1 RTX 4090 (sm_89), NVRTC 13: every species value, the
GRASET accumulators, the SETVEL velocities and the sub-step counts equal the
Fortran bit for bit (93,312 species values, max 0 ULP).  The fixtures were
rebuilt independently from the pinned sources and matched every recorded
sha256.  Made to fire before commit: capping the sub-steps at 11 instead of
WRF's 12 fails 19 of the 47 settling and drydep tests.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify.chem_oracle import load, load_case, ulp_table

ROOT = Path(__file__).resolve().parents[1] / "tests/data/oracles/chem/gocart/settling"
BASELINE = ROOT / "ULP.json"


def check_pins(root):
    lines = (root / "oracle-sha256sums.txt").read_text().splitlines()
    assert lines
    for line in lines:
        digest, path = line.split(maxsplit=1)
        assert hashlib.sha256((root / path.strip()).read_bytes()).hexdigest() == digest
    manifests = {str(p.relative_to(root)).replace("\\", "/") for p in root.glob("*/MANIFEST.txt")}
    assert manifests == {line.split(maxsplit=1)[1].strip().removeprefix("./") for line in lines if line.endswith("MANIFEST.txt")}


def to3(a):
    import cupy as cp
    return cp.asarray(np.ascontiguousarray(a.transpose(1,2,0)))


def to2(a):
    import cupy as cp
    return cp.asarray(np.ascontiguousarray(a.T))


def bits64(a):
    return np.ascontiguousarray(a.T).view("<f8").ravel()


def run_case(f, carry=None):
    import cupy as cp
    from woof.core.chem_settling import launch_settling, pack_settling_rows
    rows = carry["rows"] if carry else [to3(f["chem_in"][:,:,:,r]) for r in range(10,19)]
    accum = carry["accum"] if carry else [to2(f["accum_in"][:,:,r]) for r in range(5)]
    vel = [cp.empty_like(accum[0]) for _ in range(5)]
    rad,den = bits64(f["radius_bits"]),bits64(f["density_bits"])
    met = {n:to3(f[n]) for n in ("temp","pressure","dz","rho","qv")}
    steps=[]
    for start,end,arm in ((0,5,"none"),(5,9,"gerber")):
        params=pack_settling_rows([{"radius_m":rad[r],"density_kg_m3":den[r],"growth_arm":arm}
                                   for r in range(start,end)])
        out=launch_settling(rows[start:end],params,**met,dt=f["dt"],gravity=f["gravity"],
            dyn_visc=f["dyn_visc"],accumulators=accum if start==0 else None,
            velocities=vel if start==0 else None)
        steps.extend(out["nsteps"].get().transpose(0,2,1))
    # Restore the WRF index order in every compared output.
    field=np.stack([a.get().transpose(2,0,1) for a in rows],axis=3)
    ac=np.stack([a.get().T for a in accum],axis=2)
    vv=np.stack([a.get().T for a in vel],axis=2)
    ns=np.stack(steps,axis=2)
    if carry is not None:
        carry.update(rows=rows,accum=accum)
    return {"chem_out":ulp_table(field,f["chem_out"][:,:,:,10:19]),
            "accum_out":ulp_table(ac,f["accum_out"]),
            "velocity":ulp_table(vv,f["velocity"]),
            "nsteps":ulp_table(ns,f["nsteps"])}


def test_fixture_pins_and_coverage():
    check_pins(ROOT)
    cases=load(ROOT)
    assert len(cases)==36
    assert set(json.loads(BASELINE.read_text()))==set(cases)
    assert all(t["max_ulp"]==0 and t["n_nonzero"]==0
               for outputs in json.loads(BASELINE.read_text()).values() for t in outputs.values())
    assert {int(f["chem_opt"]) for f in cases.values()}=={300,401}
    assert {float(f["dt"]) for f in cases.values()}=={7.5,36.,60.}
    assert any(np.any(f["rh_requested"]<0.1) for f in cases.values())
    assert any(np.any(f["rh_requested"]>0.95) for f in cases.values())
    assert max(f["nsteps"].max() for f in cases.values())==12
    assert any(np.any(f["chem_in"]<0) for f in cases.values())
    for name,f in cases.items():
        if name.endswith("step2") or name.endswith("step3"):
            previous=cases[name[:-1]+str(int(name[-1])-1)]
            np.testing.assert_array_equal(f["chem_in"],previous["chem_out"])
            np.testing.assert_array_equal(f["accum_in"],previous["accum_out"])
    regular=cases["arm300_mode0_dt3_step1"]
    thin=cases["arm300_mode1_dt3_step1"]
    assert not np.array_equal(regular["chem_out"],thin["chem_out"])


@requires_gpu
@pytest.mark.parametrize("case", sorted(p.parent.name for p in ROOT.glob("*/MANIFEST.txt")))
def test_settling_parity(case):
    assert run_case(load_case(ROOT/case)) == json.loads(BASELINE.read_text())[case]


@requires_gpu
def test_three_steps_on_the_same_device_fields():
    baseline=json.loads(BASELINE.read_text())
    for arm in (300,401):
        for mode in (0,1):
            for dt in (1,2,3):
                carry={}
                for step in (1,2,3):
                    case=f"arm{arm}_mode{mode}_dt{dt}_step{step}"
                    assert run_case(load_case(ROOT/case),carry)==baseline[case]


@requires_gpu
def test_group_validation_and_arbitrary_rows():
    import cupy as cp
    from woof.core.chem_settling import pack_settling_rows, launch_settling
    with pytest.raises(ValueError,match="share"):
        pack_settling_rows([{"radius_m":1.e-6,"density_kg_m3":2000.,"growth_arm":"none"},
                            {"radius_m":2.e-6,"density_kg_m3":2000.,"growth_arm":"gerber"}])
    f=load_case(ROOT/"arm300_mode0_dt1_step1")
    original=to3(f["chem_in"][:,:,:,10])
    a,b=original.copy(),original.copy()
    params=pack_settling_rows([{"radius_m":0.73e-6,"density_kg_m3":2500.,"growth_arm":"none"}]*2)
    launch_settling([a,b],params,**{n:to3(f[n]) for n in ("temp","pressure","dz","rho","qv")},dt=7.5)
    cp.testing.assert_array_equal(a,b)
    assert not cp.array_equal(a,original)
