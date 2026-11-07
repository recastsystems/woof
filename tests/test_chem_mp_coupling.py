"""The aerosol-aware Thompson coupling against NOAA GSL's own ``get_niwfa``.

``tools/chem_wrf471_oracle/gocart/build_niwfa.sh`` compiles
``mp_thompson.F90:1025-1068`` (ccpp-physics 3e6660c6, kind_phys = 8, gfortran
15.2.0 -O0) and runs it on six columns spanning twelve orders of magnitude of
aerosol mass.  The port is the table-driven kernel in
``woof/core/kernels/chem_mp_coupling.cu`` handed GSL's twelve bins as rows.

Measured on the test host 1 RTX 4090 (sm_89): nifa and nwfa bit-identical to the
Fortran, both as the float64 before rounding and as the float32 stored.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core import chem_mp_coupling as mpc
from woof.verify.chem_oracle import ORACLE_ROOT, load_case

FIXTURE = ORACLE_ROOT / "gocart" / "niwfa"

#: get_niwfa's twelve bins as table terms (mp_thompson.F90:1062-1066):
#: (bin, target, group, group_factor, unit_mass literal)
GSL_BINS = (
    (1, "nifa", 0, 1.0, 4.0737762), (2, "nifa", 0, 1.0, 30.459203),
    (3, "nifa", 0, 1.0, 153.45048), (4, "nifa", 0, 1.0, 1011.5142),
    (5, "nifa", 0, 1.0, 5683.3501),
    (6, "nwfa", 0, 9.0, 0.0045435214), (7, "nwfa", 0, 9.0, 0.2907854),
    (8, "nwfa", 0, 9.0, 12.91224), (9, "nwfa", 0, 9.0, 206.2216),
    (10, "nwfa", 0, 9.0, 4326.23),
    (11, "nwfa", 1, 5.0, 0.3053104),
    (15, "nwfa", 2, 8.0, 0.3232698),
)


def _terms():
    return [mpc.CouplingTerm(name=f"bin{b}", target=mpc.TARGETS.index(t),
                             group=g, order=b, group_factor=f,
                             denominator=float(np.float32(d)),
                             to_kg_per_kg=1.0)
            for b, t, g, f, d in GSL_BINS]


def test_fixture_is_the_pinned_one():
    sums = (FIXTURE / "oracle-sha256sums.txt").read_text().splitlines()
    for line in sums:
        digest, rel = line.split()
        assert hashlib.sha256((FIXTURE / rel).read_bytes()).hexdigest() == digest, rel


def test_pack_terms_sorts_and_converts():
    rows = [
        {"name": "b", "units": "ppmv", "molar_mass_g_mol": 96.0576,
         "mp_coupling": {"target": "nwfa", "group": 1, "order": 0,
                         "group_factor": 5, "unit_mass": 0.3053104}},
        {"name": "a", "units": "ug kg-1",
         "mp_coupling": {"target": "nwfa", "group": 0, "order": 1,
                         "group_factor": 9.0, "unit_mass": 0.2907854}},
    ]
    terms = mpc.pack_terms(rows)
    assert [t.name for t in terms] == ["a", "b"]
    assert terms[0].to_kg_per_kg == 1.0e-9
    assert terms[1].to_kg_per_kg == 1.0e-6 * 96.0576 / 28.966
    assert terms[0].denominator == float(np.float32(0.2907854))


def test_pack_terms_refuses_ambiguous_slots_and_bad_rows():
    spec = {"target": "nwfa", "group": 0, "order": 1, "group_factor": 9.0,
            "unit_mass": 1.0}
    with pytest.raises(ValueError, match="both claim"):
        mpc.pack_terms([{"name": "a", "units": "ug kg-1", "mp_coupling": spec},
                        {"name": "b", "units": "ug kg-1", "mp_coupling": spec}])
    with pytest.raises(ValueError, match="not one of"):
        mpc.pack_terms([{"name": "a", "units": "ug kg-1",
                         "mp_coupling": dict(spec, target="nc")}])
    with pytest.raises(ValueError, match="molar_mass_g_mol"):
        mpc.pack_terms([{"name": "a", "units": "ppmv", "mp_coupling": spec}])


@requires_gpu
def test_niwfa_is_bitwise_gsl():
    """Made to fire before commit, on the test host 1 RTX 4090: taking GSL's
    denominators as float64 literals (``denominator=d``) instead of the
    float32 literals the Fortran widens moves the float64 result in 25 of
    the 30 nifa cells."""
    import cupy as cp

    case = load_case(FIXTURE / "niwfa")
    aerfld = case["aerfld"]                       # (ncol, nlev, 15), kg/kg
    fields = {f"bin{b}": cp.asarray(np.ascontiguousarray(aerfld[:, :, b - 1]))
              for b, *_ in GSL_BINS}
    shape = aerfld.shape[:2]
    nifa = cp.zeros(shape, cp.float32)
    nwfa = cp.zeros(shape, cp.float32)
    nifa64 = cp.zeros(shape, cp.float64)
    nwfa64 = cp.zeros(shape, cp.float64)
    fields = {k: cp.ascontiguousarray(v) for k, v in fields.items()}
    mpc.launch_niwfa(fields, _terms(), nifa, nwfa,
                     nifa_f64=nifa64, nwfa_f64=nwfa64)
    for name, got32, got64 in (("nifa", nifa, nifa64), ("nwfa", nwfa, nwfa64)):
        want32 = case[name]
        bits = case[f"{name}_f64_bits"].astype(np.uint32)   # (2, ncol, nlev)
        want64 = (bits[0].astype(np.uint64)
                  | (bits[1].astype(np.uint64) << np.uint64(32))).view(np.float64)
        np.testing.assert_array_equal(cp.asnumpy(got64).view(np.uint64),
                                      want64.view(np.uint64), err_msg=name)
        np.testing.assert_array_equal(cp.asnumpy(got32).view(np.uint32),
                                      want32.view(np.uint32), err_msg=name)


def test_host_niwfa_is_bitwise_gsl():
    """Grade the executable CPU preparation path against the pinned routine."""
    from woof.ingest.cpu_backend import CpuPreprocessBackend
    backend = CpuPreprocessBackend()
    case = load_case(FIXTURE / "niwfa")
    aerfld = case["aerfld"]
    fields = {f"bin{b}": np.ascontiguousarray(aerfld[:, :, b - 1]) for b, *_ in GSL_BINS}
    shape = aerfld.shape[:2]
    nifa, nwfa = (np.zeros(shape, np.float32) for _ in range(2))
    nifa64, nwfa64 = (np.zeros(shape, np.float64) for _ in range(2))
    backend.aerosol_numbers(fields, _terms(), nifa, nwfa, scale=mpc.SCALE,
                            nifa_f64=nifa64, nwfa_f64=nwfa64)
    for name, got32, got64 in (("nifa", nifa, nifa64), ("nwfa", nwfa, nwfa64)):
        bits = case[f"{name}_f64_bits"].astype(np.uint32)
        want64 = (bits[0].astype(np.uint64) | (bits[1].astype(np.uint64) << np.uint64(32)))
        np.testing.assert_array_equal(got64.view(np.uint64), want64)
        np.testing.assert_array_equal(got32.view(np.uint32), case[name].view(np.uint32))
