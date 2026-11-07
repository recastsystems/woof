"""The GOCART inventory emission add against WRF v4.7.1, bit for bit.

``tools/chem_wrf471_oracle/gocart/build_inventory.sh`` compiles WRF's own
``emiss_opt == 6`` block (chem/emissions_driver.F:1597-1619) at -O0 and runs
it for kemit 1, 3, 9 and 12 with four time steps.  The port is the
table-driven kernel ``chem_inventory_add``: the six ``gocart_ecptec``
emissions become six terms whose arithmetic path comes from the target row's
units (ppmv -> the mol km-2 hr-1 ``conv`` path; ug kg-1 -> ``alt*dt/dz8w``).

Measured on the test host 1 RTX 4090 (sm_89): every species of every case,
including the untouched ones and the levels above kemit, bit-identical.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core import chem_inventory as inv
from woof.verify.chem_oracle import ORACLE_ROOT, load

FIXTURE = ORACLE_ROOT / "gocart" / "inventory"

#: registry.chem:4022 chem indices (1-based, dummy at 1) and
#: registry.chem:4058 emis_ant indices, as the harness stub declares them.
#: (emis index, chem index, target units)
EMISSIONS = ((2, 2, "ppmv"),      # e_so2 -> so2
             (3, 3, "ppmv"),      # e_sulf -> sulf
             (4, 7, "ug kg-1"),   # e_bc -> bc1
             (5, 9, "ug kg-1"),   # e_oc -> oc1
             (6, 6, "ug kg-1"),   # e_pm_25 -> p25
             (7, 20, "ug kg-1"))  # e_pm_10 -> p10


def test_fixture_is_the_pinned_one():
    for line in (FIXTURE / "oracle-sha256sums.txt").read_text().splitlines():
        digest, rel = line.split()
        assert hashlib.sha256((FIXTURE / rel).read_bytes()).hexdigest() == digest, rel


def test_mode_comes_from_units_not_names():
    rows = [{"name": "anything", "units": "ppmv",
             "emissions": [{"source": "s", "field": "f", "weight": 1.0,
                            "vertical": "kemit", "provenance": "x"},
                           {"source": "r", "field": "g", "weight": 1.0,
                            "vertical": "plumerise", "provenance": "x"}]}]
    terms = inv.pack_terms(rows)
    assert [(t.field, t.mode) for t in terms] == [("f", 1)]
    with pytest.raises(ValueError):
        inv.arithmetic_mode("mol mol-1")


def _gpu(a):
    import cupy as cp
    return cp.asarray(np.ascontiguousarray(a))


@requires_gpu
def test_every_case_is_bitwise_wrf():
    """Made to fire before commit, on the test host 1 RTX 4090: writing conv as
    ``4.828e-4/(rho*(dz*60))*dt`` instead of WRF's left-to-right order moves
    SO2 and sulfate in every case (1 to 31 of 768 cells per species; the
    kemit=1 case is the weakest, one SO2 and two sulfate cells)."""
    import cupy as cp

    cases = load(FIXTURE)
    assert len(cases) == 4
    for name, case in cases.items():
        kemit = int(case["kemit"])
        dt = float(case["dtstep"])
        # WRF (i, k, j[, n]) -> woof (k, j, i)
        tr = lambda a: np.transpose(a, (1, 2, 0))
        rho, alt, dz = (_gpu(tr(case[n])) for n in ("rho_phy", "alt", "dz8w"))
        chem = {n: _gpu(tr(case["chem_in"][..., n - 1]))
                for n in range(1, case["chem_in"].shape[-1] + 1)}
        fields, frames, terms = [], [], []
        for e, n, units in EMISSIONS:
            fields.append(chem[n])
            frames.append(_gpu(tr(case["emis_ant"][..., e - 1])))
            terms.append(inv.InventoryTerm(row=str(n), source="oracle",
                                           field=str(e), vertical="kemit",
                                           weight=1.0,
                                           mode=inv.arithmetic_mode(units)))
        inv.launch_inventory_add(rho, alt, dz, fields, frames, terms,
                                 kemit=kemit, dtstep=dt)
        for n, got in chem.items():
            want = tr(case["chem_out"][..., n - 1])
            np.testing.assert_array_equal(
                cp.asnumpy(got).view(np.uint32), want.view(np.uint32),
                err_msg=f"{name} chem index {n}")
