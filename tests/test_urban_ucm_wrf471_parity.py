"""The single-layer UCM against WRF v4.7.1's own ``urban``, bit for bit.

``tools/urban_wrf471_oracle/ucm_column_oracle.F90`` calls ``urban`` in the
byte-unmodified ``phys/module_sf_urban.F`` (gfortran 15.2 -O0, glibc 2.43)
over the 3 NLCD and 11 LCZ categories of WRF's own URBPARM tables, nine
scenarios each (day/night, dry/light/heavy rain, stable/unstable, a windy
day with canopy wind over 5 m/s, a January cold day, a humid night that puts
dew on the green roof, and a first level just above ZDC+Z0C+2 m that takes
the reduced canopy-wind branch) and six consecutive steps per column, under
twelve switch variants: on URBPARM.TBL the table defaults, CH_SCHEME 1
(mos), TS_SCHEME 2 (force-restore), both, AHOPTION+ALHOPTION, IMP_SCHEME 2
(water retention), BOUNDR/B/G 2, and GROPTION 1 (green roof) with and
without IRI_SCHEME 1; on URBPARM_LCZ.TBL the defaults, IMP_SCHEME 2 with
AHOPTION+ALHOPTION, and GROPTION 1.  3,240 column steps in all (commit
ce7b5f970's message says 4,584, a miscount; 3,240 is the fixture's row
count).  Measured bitwise on the test host 1 RTX 4090
(sm_89) and the test host 2 RTX 5090 (sm_120), NVRTC 13.4.92 both.

This fixture explicitly selects the older ``wrf_471`` water balance. The
production default is the scoped WRF 4.8 store correction. The shared
source-marked no-FTZ loader retains positive subnormal rainfall in both
arithmetic modes; ``test_urban_water_balance_gpu.py`` checks it directly.

**Measured: every output and every state field of every row is
bit-identical to WRF** -- 19 outputs (TS QS SH LH LH_KINEMATIC SW ALB LW G RN
PSIM PSIH GZ1OZ0 U10 V10 TH2 Q2 UST ZNT) and 44 state words (roof/wall/road
and canopy temperatures and humidity, the 3x4 layer temperatures, the four
M-O lengths, the six SFCDIF exchange coefficients, the green-roof canopy
water, skin and layer temperature and moisture, the three retention depths
and three evaporation fluxes).  Both from each row's own recorded state and
with the port carrying its own state across all six steps.

This gate has been observed to fail: see the mutation note in
:func:`test_every_row_is_bitwise_wrf`.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core.fp32_ulp import fp32_ulp_distance
from woof.verify.urban_ucm_oracle import (
    OUTPUT_COLUMNS,
    STATE_COLUMNS,
    UCM_ORACLE_DIR,
    UCM_VARIANTS,
    carried_replay,
    load_variant,
    replay,
)


def _report(port, ref) -> str:
    lines = []
    for key, want in ref.items():
        got = np.asarray(port[key], np.float32)
        want = np.asarray(want, np.float32)
        bad = np.nonzero(got.view(np.uint32) != want.view(np.uint32))[0]
        if bad.size:
            ulp = int(np.max(fp32_ulp_distance(got, want)))
            lines.append(f"{key}: {bad.size} rows, max {ulp} ULP, first row "
                         f"{bad[0]}: port {got[bad[0]]!r} wrf {want[bad[0]]!r}")
    return "\n".join(lines)


def test_fixture_is_the_pinned_one():
    sums = (UCM_ORACLE_DIR / "oracle-sha256sums.txt").read_text().split("\n")
    pinned = dict(reversed(line.split()) for line in sums if line.strip())
    for name in UCM_VARIANTS:
        for suffix in (".csv.gz", "-table.csv", "-switches.csv"):
            path = UCM_ORACLE_DIR / f"{name}{suffix}"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == pinned[path.name]


def test_every_switch_variant_moves_the_answer():
    """A switch the fixture cannot discriminate is a switch this gate cannot
    check (the Noah harness's rule): each variant's outputs must differ from
    its table's default fixture."""
    base = {}
    for table in ("nlcd", "lcz"):
        rows, _, _ = load_variant(f"ucm-{table}-default")
        base[table] = np.stack([rows[c] for c in OUTPUT_COLUMNS])
    for name in UCM_VARIANTS:
        table = name.split("-")[1]
        if name.endswith("-default"):
            continue
        rows, _, _ = load_variant(name)
        out = np.stack([rows[c] for c in OUTPUT_COLUMNS])
        assert out.shape == base[table].shape
        assert not np.array_equal(out.view(np.uint32), base[table].view(np.uint32)), name


def test_fixture_reaches_the_branches_it_claims():
    rows, table, _ = load_variant("ucm-lcz-default")
    # scenario 9 puts ZA below ZR+2 for the tall classes: UC = UA/2 (:900-902)
    low = (rows["scenario"] == 9)
    zr = table["zr"][rows["utype"] - 1]
    reduced = low & (zr + np.float32(2.0) >= rows["za"])
    assert np.any(reduced)
    assert np.all(rows["uc"][reduced] == rows["ua"][reduced] / np.float32(2.0))
    # canopy wind over 5 m/s takes CH_SCHEME 2's second arm (:1266)
    assert np.any(rows["uc"] > 5.0)
    # both stability arms of the grid diagnostics (:1607)
    assert np.any(rows["psim"] < 0) and np.any(rows["psim"] > 0)
    # rain over 1 mm/h (IMP_SCHEME 1 BETR = 0.7) and none
    assert np.any(rows["rain"] > 1.0) and np.any(rows["rain"] == 0.0)


@requires_gpu
@pytest.mark.parametrize("name", UCM_VARIANTS)
def test_every_row_is_bitwise_wrf(name):
    """Each row replayed from its recorded state.

    Made to fire before commit, on the test host 1 RTX 4090: respelling
    ``QS0R=0.622*ES/(PS-0.378*ES)`` in the kernel's roof Newton iteration as
    the algebraically identical ``0.622*(ES/(PS-0.378*ES))`` fails 10 of the
    12 variants -- every one whose roof takes that iteration; the two
    TS_SCHEME 2 variants never reach the line -- by up to 192 ULP in the
    roof evaporation flux; moving the folded ``2.5*10.**6./461.51`` by one
    ULP fails all twelve.
    """
    port, ref, codes = replay(name)
    assert not np.any(codes), np.unique(codes)
    assert set(ref) == set(OUTPUT_COLUMNS) | {f"after:{c}" for c in STATE_COLUMNS}
    report = _report(port, ref)
    assert not report, f"{name} differs from WRF:\n{report}"


@requires_gpu
@pytest.mark.parametrize("name", UCM_VARIANTS)
def test_carried_state_stays_on_wrf(name):
    """The port carries its own prognostic state across six steps."""
    port, ref = carried_replay(name)
    report = _report(port, ref)
    assert not report, f"{name} (carried) differs from WRF:\n{report}"


@requires_gpu
def test_wrfs_first_level_fatal_is_a_refusal():
    """module_sf_urban.F:825 stops WRF when ZDC+Z0C+2 >= ZA; the kernel
    returns code 1 for the column instead of integrating a log-profile that
    is undefined there."""
    from woof.core.urban_ucm import run_columns
    from woof.verify.urban_ucm_oracle import _inputs, _state_from, port_params

    rows, table, switches = load_variant("ucm-lcz-default")
    params = port_params(table, switches)
    sel = np.nonzero(rows["utype"] == 1)[0][:2]
    inputs = _inputs(rows, sel)
    limit = table["zdc"][0] + table["z0c"][0] + np.float32(2.0)
    inputs["za"] = np.asarray([limit, np.nextafter(limit, np.float32(1e9))],
                              dtype=np.float32)
    _, _, codes = run_columns(params, inputs, _state_from(rows, sel, "_in"))
    assert codes.tolist() == [1, 0]
