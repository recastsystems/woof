"""The Pleim-Xiu LSM port against WRF v4.6.1 as built, word for word.

The reference is tools/pxlsm_wrf461_oracle: the byte-unmodified
module_sf_pxlsm.F compiled with the scalar wrf.exe's flags and driven
through ``pxlsm`` itself on 13 call geometries x 96 columns x 6 consecutive
calls (every land-use table, both pxlsm_smois_init values, soil nudging on
and off, MODIS vegetation on and off, one/two/three PX sub-steps, day and
night, bare soil, crops, forest, urban, wetland, partial water, open water,
cold water turned to ice, analysed sea ice, snow, dry and wet soil).

STRICT (``GPUWM_WRF_EXACT=1``): every output word of every column of every
call equals WRF's -- TSK, HFX, QFX, LH, GRDFLX, both soil temperatures and
moistures, SNOW/SNOWH/SNOWC, CANWAT, and the rest of pxlsm's INOUT and OUT
set -- except the declared defect columns of the modis21-lake-defect case,
where WRF's RSTMIN is 0/0 and the port's is defined (and those columns are
asserted to be exactly that: NaN in WRF, finite here).

DEFAULT (no flag): the same kernel in explicit rounding with WOOF float32
math; the distance per field is measured and held to the tolerances in
DEFAULT_TOLERANCE_ULP, which are zero for both modes.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof import wrf_exact  # noqa: E402

from pxlsm_oracle_support import (OUTPUT_FIELDS, as_float,  # noqa: E402
                                  cases_index, read_case, run_port,
                                  ulp_distance, _load_tool)

try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # pragma: no cover - CPU-only host
    pytest.skip("no CUDA device", allow_module_level=True)

CASES = [entry["name"] for entry in cases_index()]
DEFECT_COLUMNS = _load_tool("gen_cases").DEFECT_COLUMNS


def _mask(case, field, shape):
    """Columns the comparison covers: all but the declared defect ones."""
    keep = np.ones(shape, dtype=bool)
    if case.name.endswith("-defect"):
        for j, i in DEFECT_COLUMNS:
            if len(shape) == 3:
                keep[j, :, i] = False
            else:
                keep[j, i] = False
    return keep



def _record_step(case, step, got, want):
    """Small evidence only, never raw field arrays."""
    import hashlib
    import json
    import os
    from pathlib import Path
    path = os.environ.get("WOOF_PXLSM_RECEIPT")
    if not path:
        return
    for field in OUTPUT_FIELDS:
        keep = _mask(case, field, want[field].shape)
        a, b = got[field][keep], want[field][keep]
        entry = dict(case=case.name, step=step + 1, field=field,
                     strict=bool(wrf_exact.ENABLED), words=int(a.size),
                     excluded_words=int(keep.size-keep.sum()),
                     differing_words=int(np.count_nonzero(a != b)),
                     port_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                     wrf_sha256=hashlib.sha256(b.tobytes()).hexdigest())
        if field != "isltyp":
            entry["max_ulp"] = int(ulp_distance(a, b).max(initial=0))
        with Path(path).open("a") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")


def test_the_fixture_covers_what_the_brief_names():
    """At least 64 columns, every switch value, nudged and unnudged columns."""
    seen = set()
    for name in CASES:
        case = read_case(name)
        assert case.nx * case.ny >= 64
        assert case.nsteps >= 2
        seen.add(("nudge", case.soil_nudge))
        seen.add(("init", case.smois_init))
        seen.add(("modis", case.modis_veg))
        seen.add(("nlcat", case.nlcat))
        seen.add(("substeps", int(case.dt / 40.0 + 1.0)))
    for item in [("nudge", 0), ("nudge", 1), ("init", 0), ("init", 1),
                 ("modis", 0), ("modis", 1), ("substeps", 1),
                 ("substeps", 2), ("substeps", 3)]:
        assert item in seen, item
    for nlcat in (20, 21, 24, 28, 40, 50, 61):
        assert ("nlcat", nlcat) in seen, nlcat


@pytest.mark.skipif(not wrf_exact.ENABLED,
                    reason="strict arithmetic needs GPUWM_WRF_EXACT=1")
@pytest.mark.parametrize("name", CASES)
def test_every_word_of_every_call_is_wrfs(name):
    case = read_case(name)
    failures = []
    for step, got in enumerate(run_port(case)):
        want = case.expected[step]
        _record_step(case, step, got, want)
        for field in OUTPUT_FIELDS:
            keep = _mask(case, field, want[field].shape)
            bad = (got[field] != want[field]) & keep
            if bad.any():
                where = np.argwhere(bad)[0].tolist()
                failures.append(
                    f"step {step + 1} {field}: {int(bad.sum())} words, first at "
                    f"{where} port={as_float(got[field])[tuple(where)]!r} "
                    f"wrf={as_float(want[field])[tuple(where)]!r} "
                    f"ulp={int(ulp_distance(got[field], want[field])[tuple(where)])}")
    assert not failures, "\n".join(failures[:40])


@pytest.mark.skipif(not wrf_exact.ENABLED,
                    reason="strict arithmetic needs GPUWM_WRF_EXACT=1")
def test_the_declared_rstmin_defect_is_exactly_what_it_says():
    """WRF goes NaN on the no-leaf-area columns; the port stays finite there
    and equals WRF on every other column (asserted above)."""
    case = read_case("modis21-lake-defect")
    for step, got in enumerate(run_port(case)):
        want = case.expected[step]
        for j, i in DEFECT_COLUMNS:
            assert np.isnan(as_float(want["tsk"])[j, i])
            assert np.isfinite(as_float(got["tsk"])[j, i])
            assert np.isfinite(as_float(got["rs"])[j, i])


#: The default arm's measured distance from WRF, per field, in ULP, with
#: no headroom. Both modes keep WRF's scalar order; every field is held to 0.
DEFAULT_TOLERANCE_ULP: dict[str, int] = {}


@pytest.mark.skipif(wrf_exact.ENABLED,
                    reason="the default arm is measured without the flag")
def test_the_default_arm_stays_within_its_measured_tolerance(capsys):
    worst: dict[str, int] = {}
    for name in CASES:
        case = read_case(name)
        for step, got in enumerate(run_port(case)):
            want = case.expected[step]
            _record_step(case, step, got, want)
            for field in OUTPUT_FIELDS:
                if field == "isltyp":
                    assert np.array_equal(got[field], want[field])
                    continue
                keep = _mask(case, field, want[field].shape)
                assert np.array_equal(got[field][keep], want[field][keep]), field
                d = ulp_distance(got[field], want[field])[keep]
                worst[field] = max(worst.get(field, 0), int(d.max(initial=0)))
    with capsys.disabled():
        print("\nPX default-arm max ULP per field:",
              {k: v for k, v in sorted(worst.items())})
    over = {f: (v, DEFAULT_TOLERANCE_ULP.get(f, 0)) for f, v in worst.items()
            if v > DEFAULT_TOLERANCE_ULP.get(f, 0)}
    assert not over, over

@pytest.mark.skipif(not wrf_exact.ENABLED, reason="strict mutation control")
def test_one_table_bit_moves_the_oracle_result():
    from woof.core.pxlsm import px_device_table
    case = read_case("usgs-day-init")
    table = px_device_table(case.nlcat).copy()
    table.view(np.uint32)[8, 15] ^= np.uint32(1)
    changed = next(run_port(case, table_override=table))
    assert np.any(changed["snoalb"] != case.expected[0]["snoalb"]), (
        "the field comparison failed to catch a planted snow-albedo table bit")

def test_tiles_use_domain_edges_for_nudging():
    from woof.core import pxlsm
    from pxlsm_oracle_support import device_state, port_words
    case = read_case("usgs-day-nudge")
    full = list(run_port(case))
    initial, atmosphere, lu, soil = device_state(case)
    tiled = []
    for j0, j1 in ((0, 4), (4, 8)):
        for i0, i1 in ((0, 6), (6, 12)):
            take = lambda a: cp.ascontiguousarray(a[..., j0:j1, i0:i1])
            tiled.append((j0, j1, i0, i1,
                          {k: take(a) for k, a in initial.items()},
                          {k: take(a) for k, a in atmosphere.items()},
                          take(lu), take(soil)))
    curr = np.float32(case.curr_secs0)
    for step in range(case.nsteps):
        records = []
        for j0, j1, i0, i1, fields, atm, lu_tile, soil_tile in tiled:
            pxlsm.launch_pxlsm(
                fields, atm, landusef=lu_tile, soilcbot=soil_tile,
                dt=case.dt, dtbl=case.dtbl, curr_secs=float(curr),
                itimestep=case.itimestep0 + step,
                smois_init=case.smois_init, soil_nudge=case.soil_nudge,
                modis_veg=case.modis_veg, anal_interval=case.anal_interval,
                domain_shape=(case.ny, case.nx), tile_origin=(j0, i0),
                dzs=case.dzs)
            records.append((j0, j1, i0, i1, port_words(fields)))
        for field in OUTPUT_FIELDS:
            assembled = np.empty_like(full[step][field])
            for j0, j1, i0, i1, rec in records:
                assembled[j0:j1, ..., i0:i1] = rec[field]
            np.testing.assert_array_equal(assembled, full[step][field],
                                          err_msg=f"step {step+1} {field}")
        curr = np.float32(curr + np.float32(case.dt))
