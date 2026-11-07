"""GOCART aerosol dry deposition against WRF v4.7.1, bit for bit.

``run_gocart_drydep.F90`` drives the byte-unmodified ``gocart_drydep_driver``
(chem/module_gocart_drydep.F) for both resistance arms: ``chem_opt=300``,
where the aerodynamic resistance is Wesely's ``aer_res_def`` (WRF's own
``depvel``, chem/module_dep_simple.F:1520-1616, extracted by line range), and
``chem_opt=401``, where it is the local stability formula; stable, unstable
and neutral columns (``rmol`` 0 and under 1e-6), small and large ``ust``,
``pblh/obk < -30``, and the domain's first row and column, which WRF leaves at
zero velocity; three consecutive steps for the DRYDEP accumulators.

WRF-Chem zeroes ``|rmol| < 1e-6`` IN THE MODEL'S OWN ``rmol`` (the INTENT(INOUT)
chain from wesely_driver) before GOCART reads it.  The port applies the same
zeroing to a local copy for both uses and never writes the model field; the
test asserts the input ``rmol`` is unchanged.  That is the one WRF-Chem side
effect the port does not reproduce, and it only touches columns WRF treats as
neutral anyway.

Measured on the test host 1 RTX 4090 (sm_89): ddvel, aer_res, the rmol used and the
DRYDEP accumulators equal the Fortran bit for bit.  Made to fire before
commit: changing the neutral-limit ``vds = 0.002*ustar`` to 0.003 fails 7
tests.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify.chem_oracle import load, load_case, ulp_table
from test_chem_settling_wrf471_parity import check_pins, to2, to3

ROOT=Path(__file__).resolve().parents[1]/"tests/data/oracles/chem/gocart/drydep"
BASELINE=ROOT/"ULP.json"


def run_case(f, carry=None):
    import cupy as cp
    from woof.core.chem_drydep_aerosol import launch_drydep, pack_drydep_rows
    # Include one arbitrary non-dust row, with a sentinel accumulator.
    rows=[to3(f["chem"][:,:,:,r]) for r in range(10,15)]+[to3(f["chem"][:,:,:,6])]
    accum=carry["accum"] if carry else [to2(f["accum_in"][:,:,r]) for r in range(5)]+[cp.full((6,6),123.,dtype=cp.float32)]
    rmol=to2(f["rmol_in"])
    out=launch_drydep(rows,pack_drydep_rows([{"is_dust":True}]*5+[{"is_dust":False}]),
        rho=to3(f["rho"]),rmol=rmol,**{n:to2(f[n]) for n in ("ust","znt","pbl")},
        dt=f["dt"],chem_opt=int(f["chem_opt"]),domain_extent=(1,6,1,6),tile_origin=(1,1),accumulators=accum)
    np.testing.assert_array_equal(rmol.get().T,f["rmol_in"])
    np.testing.assert_array_equal(accum[-1].get(),np.full((6,6),123.,np.float32))
    velocity=out["ddvel"].get().T
    for edge in (velocity[0,:],velocity[-1,:],velocity[:,0],velocity[:,-1]):
        assert np.all(edge==0)
    # All aerosol rows share this value; sulfate and msa included separately.
    target=f["ddvel"][:,:,10]
    for r in [2,4,*range(5,20)]:
        np.testing.assert_array_equal(f["ddvel"][:,:,r],target)
    ac=np.stack([a.get().T for a in accum[:5]],axis=2)
    if carry is not None:
        carry.update(accum=accum)
    return {"ddvel":ulp_table(velocity,target),"accum_out":ulp_table(ac,f["accum_out"]),
            "aer_res":ulp_table(out["aer_res"].get().T,f["aer_res"]),
            "rmol_used":ulp_table(out["rmol_used"].get().T,f["rmol_used"])}


def test_fixture_pins_and_coverage():
    check_pins(ROOT)
    cases=load(ROOT)
    assert len(cases)==6
    assert set(json.loads(BASELINE.read_text()))==set(cases)
    assert all(t["max_ulp"]==0 and t["n_nonzero"]==0
               for outputs in json.loads(BASELINE.read_text()).values() for t in outputs.values())
    a,b=cases["arm300_step1"],cases["arm401_step1"]
    assert int(a["numgas"])==5 and int(b["numgas"])==0
    assert not np.array_equal(a["ddvel"],b["ddvel"])
    assert np.any(a["rmol_in"]==0)
    assert np.any((abs(a["rmol_in"])<1.e-6)&(a["rmol_in"]!=0))
    assert np.any(a["rmol_in"]>0) and np.any(a["rmol_in"]<0)
    assert np.any(a["ust"]<0.1) and np.any(a["ust"]>0.1)
    assert {1.,2.}==set(a["xland"].ravel())
    assert np.any(a["hfx"]<1.e-5) and np.any(a["hfx"]>100)
    assert np.any(a["pbl"]*a["rmol_used"] < -30)
    for arm in (300,401):
        for step in (2,3):
            np.testing.assert_array_equal(cases[f"arm{arm}_step{step}"]["accum_in"],
                                          cases[f"arm{arm}_step{step-1}"]["accum_out"])


@requires_gpu
@pytest.mark.parametrize("case", sorted(p.parent.name for p in ROOT.glob("*/MANIFEST.txt")))
def test_drydep_parity(case):
    assert run_case(load_case(ROOT/case))==json.loads(BASELINE.read_text())[case]


@requires_gpu
def test_three_steps_on_the_same_device_accumulators():
    baseline=json.loads(BASELINE.read_text())
    for arm in (300,401):
        carry={}
        for step in (1,2,3):
            case=f"arm{arm}_step{step}"
            assert run_case(load_case(ROOT/case),carry)==baseline[case]
