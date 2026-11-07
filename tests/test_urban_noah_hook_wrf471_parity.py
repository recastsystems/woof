"""Public WRF v4.7.1 lsm outputs for the urban natural-surface hand-over.

The CPU fixture checks are qualified. The GPU baseline is deliberately empty
until an authorized GPU run measures it. A GPU run prints the complete table
in the assertion, rather than accepting an invented tolerance or self-pinning.
Q1 is not exposed by the public WRF driver and is not recovered from QSFC.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core.urban_tables import urban_category_set
from woof.verify.urban_oracle import ORACLE_ROOT, load, ulp_table

DIRECTORY = ORACLE_ROOT / "infra" / "noah_hook"
CASES = (
    "off_reference", "slucm_frc0", "slucm_lcz_frc0", "slucm_frc",
    "slucm_frc0_monalb", "slucm_frc_monalb", "bep_frc", "bep_frc_monalb",
)
UNBLENDED = (
    "smois", "tslb", "sh2o", "smcrel", "canwat", "snow", "snowh", "snowc",
    "albbck", "z0", "znt", "lai", "smstav", "smstot", "sfcrunoff",
    "udrunoff", "acsnom", "acsnow", "snopcx", "potevp", "noahres", "chklowq",
)
SURFACE = ("tsk", "hfx", "qfx", "lh", "grdflx", "albedo", "qsfc", "emiss", "ust")

# Architecture -> case -> field -> measured maximum ULP against WRF v4.7.1.
# MEASURED 2026-10-07 on test host (RTX PRO 6000, sm_120, NVRTC 12.9.86) at
# development branch, under GPUWM_WRF_EXACT=1 and under default arithmetic
# alike: ZERO in every field of every case and of both column groups.  The
# rows this table held before (sm_89 from test host 1, 2026-09-30; sm_120 from
# test host 2 and test host 4, re-recorded 2026-10-01; in git history) were the plain
# Noah column's own classes -- the HFX/GRDFLX flux family, the runoff and
# snowmelt accumulators and the float64 NOAHRES residual -- and that lane
# closed them (WOOF's own float32 libm words, integer powers, NOAHRES summed
# in float32, no FMA contraction, SNOPCX through FDTLIW); this independent
# WRF v4.7.1 oracle now reads 0 with them.  sm_89 has not been re-read since:
# its row is retired rather than left asserting numbers the kernel no longer
# produces, and the first sm_89 run prints its table for pinning.
_FIELDS_BY_CASE: dict[str, tuple[str, ...]] = {
    'bep_frc': ('acsnom', 'acsnow', 'albbck', 'canwat', 'chklowq', 'lai', 'noahres', 'potevp', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk_rural_bep', 'tslb', 'udrunoff', 'z0', 'znt',),
    'bep_frc/nonurban': ('acsnom', 'acsnow', 'albbck', 'canwat', 'chklowq', 'lai', 'noahres', 'potevp', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk_rural_bep', 'tslb', 'udrunoff', 'z0', 'znt',),
    'bep_frc/urban': ('acsnom', 'acsnow', 'albbck', 'canwat', 'chklowq', 'lai', 'noahres', 'potevp', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk_rural_bep', 'tslb', 'udrunoff', 'z0', 'znt',),
    'bep_frc_monalb': ('acsnom', 'acsnow', 'albbck', 'canwat', 'chklowq', 'lai', 'noahres', 'potevp', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk_rural_bep', 'tslb', 'udrunoff', 'z0', 'znt',),
    'bep_frc_monalb/nonurban': ('acsnom', 'acsnow', 'albbck', 'canwat', 'chklowq', 'lai', 'noahres', 'potevp', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk_rural_bep', 'tslb', 'udrunoff', 'z0', 'znt',),
    'bep_frc_monalb/urban': ('acsnom', 'acsnow', 'albbck', 'canwat', 'chklowq', 'lai', 'noahres', 'potevp', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk_rural_bep', 'tslb', 'udrunoff', 'z0', 'znt',),
    'off_reference': ('acsnom', 'acsnow', 'albbck', 'albedo', 'canwat', 'chklowq', 'emiss', 'grdflx', 'hfx', 'lai', 'lh', 'noahres', 'potevp', 'qfx', 'qsfc', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk', 'tslb', 'udrunoff', 'ust', 'z0', 'znt',),
    'off_reference/nonurban': ('acsnom', 'acsnow', 'albbck', 'albedo', 'canwat', 'chklowq', 'emiss', 'grdflx', 'hfx', 'lai', 'lh', 'noahres', 'potevp', 'qfx', 'qsfc', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk', 'tslb', 'udrunoff', 'ust', 'z0', 'znt',),
    'off_reference/urban': ('acsnom', 'acsnow', 'albbck', 'albedo', 'canwat', 'chklowq', 'emiss', 'grdflx', 'hfx', 'lai', 'lh', 'noahres', 'potevp', 'qfx', 'qsfc', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk', 'tslb', 'udrunoff', 'ust', 'z0', 'znt',),
    'slucm_frc': ('acsnom', 'acsnow', 'albbck', 'canwat', 'chklowq', 'lai', 'noahres', 'potevp', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tslb', 'udrunoff', 'z0', 'znt',),
    'slucm_frc/nonurban': ('acsnom', 'acsnow', 'albbck', 'canwat', 'chklowq', 'lai', 'noahres', 'potevp', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tslb', 'udrunoff', 'z0', 'znt',),
    'slucm_frc/urban': ('acsnom', 'acsnow', 'albbck', 'canwat', 'chklowq', 'lai', 'noahres', 'potevp', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tslb', 'udrunoff', 'z0', 'znt',),
    'slucm_frc0': ('acsnom', 'acsnow', 'albbck', 'albedo', 'canwat', 'chklowq', 'emiss', 'grdflx', 'hfx', 'lai', 'lh', 'noahres', 'potevp', 'qfx', 'qsfc', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk', 'tslb', 'udrunoff', 'ust', 'z0', 'znt',),
    'slucm_frc0/nonurban': ('acsnom', 'acsnow', 'albbck', 'albedo', 'canwat', 'chklowq', 'emiss', 'grdflx', 'hfx', 'lai', 'lh', 'noahres', 'potevp', 'qfx', 'qsfc', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk', 'tslb', 'udrunoff', 'ust', 'z0', 'znt',),
    'slucm_frc0/urban': ('acsnom', 'acsnow', 'albbck', 'albedo', 'canwat', 'chklowq', 'emiss', 'grdflx', 'hfx', 'lai', 'lh', 'noahres', 'potevp', 'qfx', 'qsfc', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk', 'tslb', 'udrunoff', 'ust', 'z0', 'znt',),
    'slucm_frc0_monalb': ('acsnom', 'acsnow', 'albbck', 'albedo', 'canwat', 'chklowq', 'emiss', 'grdflx', 'hfx', 'lai', 'lh', 'noahres', 'potevp', 'qfx', 'qsfc', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk', 'tslb', 'udrunoff', 'ust', 'z0', 'znt',),
    'slucm_frc0_monalb/nonurban': ('acsnom', 'acsnow', 'albbck', 'albedo', 'canwat', 'chklowq', 'emiss', 'grdflx', 'hfx', 'lai', 'lh', 'noahres', 'potevp', 'qfx', 'qsfc', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk', 'tslb', 'udrunoff', 'ust', 'z0', 'znt',),
    'slucm_frc0_monalb/urban': ('acsnom', 'acsnow', 'albbck', 'albedo', 'canwat', 'chklowq', 'emiss', 'grdflx', 'hfx', 'lai', 'lh', 'noahres', 'potevp', 'qfx', 'qsfc', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk', 'tslb', 'udrunoff', 'ust', 'z0', 'znt',),
    'slucm_frc_monalb': ('acsnom', 'acsnow', 'albbck', 'canwat', 'chklowq', 'lai', 'noahres', 'potevp', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tslb', 'udrunoff', 'z0', 'znt',),
    'slucm_frc_monalb/nonurban': ('acsnom', 'acsnow', 'albbck', 'canwat', 'chklowq', 'lai', 'noahres', 'potevp', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tslb', 'udrunoff', 'z0', 'znt',),
    'slucm_frc_monalb/urban': ('acsnom', 'acsnow', 'albbck', 'canwat', 'chklowq', 'lai', 'noahres', 'potevp', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tslb', 'udrunoff', 'z0', 'znt',),
    'slucm_lcz_frc0': ('acsnom', 'acsnow', 'albbck', 'albedo', 'canwat', 'chklowq', 'emiss', 'grdflx', 'hfx', 'lai', 'lh', 'noahres', 'potevp', 'qfx', 'qsfc', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk', 'tslb', 'udrunoff', 'ust', 'z0', 'znt',),
    'slucm_lcz_frc0/nonurban': ('acsnom', 'acsnow', 'albbck', 'albedo', 'canwat', 'chklowq', 'emiss', 'grdflx', 'hfx', 'lai', 'lh', 'noahres', 'potevp', 'qfx', 'qsfc', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk', 'tslb', 'udrunoff', 'ust', 'z0', 'znt',),
    'slucm_lcz_frc0/urban': ('acsnom', 'acsnow', 'albbck', 'albedo', 'canwat', 'chklowq', 'emiss', 'grdflx', 'hfx', 'lai', 'lh', 'noahres', 'potevp', 'qfx', 'qsfc', 'sfcrunoff', 'sh2o', 'smcrel', 'smois', 'smstav', 'smstot', 'snopcx', 'snow', 'snowc', 'snowh', 'tsk', 'tslb', 'udrunoff', 'ust', 'z0', 'znt',),
}
MEASURED_MAX_ULP_BY_ARCHITECTURE: dict[str, dict[str, dict[str, int]]] = {
    "120": {case: {field: 0 for field in fields}
            for case, fields in _FIELDS_BY_CASE.items()},
}


def _layout(array):
    """WRF (i,j)/(i,k,j) -> kernel (j,i)/(k,j,i)."""
    a = np.asarray(array)
    axes = (1, 0) if a.ndim == 2 else (1, 2, 0)
    return np.ascontiguousarray(a.transpose(axes))


def test_fixture_inputs_outputs_and_nonurban_identity():
    fixtures = load(DIRECTORY)
    assert set(fixtures) == set(CASES)
    categories = urban_category_set(isurban=13)
    for name, case in fixtures.items():
        assert int(case["natural"]) == categories.natural
        np.testing.assert_array_equal(case["lcz_categories"], categories.lcz)
        assert case["dz8w_in"].shape == (16, 21, 1)
        for field in UNBLENDED + SURFACE:
            assert np.isfinite(case[f"{field}_out"]).all(), (name, field)
        for field in ("smois", "tslb", "sh2o"):
            assert case[f"{field}_in"].shape == (16, 4, 1)
        assert {3, 4, 8} == set(case["isltyp_in"].ravel())
        for field in ("swdown", "rainbl", "snow"):
            a = case[f"{field}_in"]
            assert (a == 0).any() and (a > 0).any(), (name, field)
        delta = case["tsk_in"] - case["t3d_in"][:, 0, :]
        assert (delta > 0).any() and (delta < 0).any()

    zero, off = fixtures["slucm_frc0"], fixtures["off_reference"]
    keep = zero["ivgtyp_in"][:, 0] == 10
    assert keep.any() and (~keep).any()
    assert np.all(zero["frc_urb2d_in"] == 0)
    # Both calls receive identical initialized state. This includes all array
    # outputs, not only the subset measured on the CUDA kernel.
    for field, a in zero.items():
        if field.endswith(("_in", "_out")) and a.ndim and a.shape[0] == 16:
            assert a[keep].tobytes() == off[field][keep].tobytes(), field

    lcz = fixtures["slucm_lcz_frc0"]
    assert int(lcz["use_wudapt_lcz"]) == 1
    assert set(categories.urban) <= set(lcz["ivgtyp_in"].ravel())
    assert (lcz["ivgtyp_in"] == 10).any()
    assert np.all(lcz["frc_urb2d_in"] == 0)
    for name in ("slucm_frc", "slucm_frc_monalb", "bep_frc", "bep_frc_monalb"):
        case = fixtures[name]
        urban = np.isin(case["ivgtyp_in"], categories.urban)
        fraction = case["frc_urb2d_in"][urban]
        assert ((fraction > 0) & (fraction < .99)).any()
        assert (fraction >= .99).any()
        assert np.all(case["ts_urb2d_in"][urban] != case["tsk_in"][urban])
        if int(case["option"]) == 2:
            before, after = case["tsk_rural_bep_in"], case["tsk_rural_bep_out"]
            assert np.all(before[urban] != case["tsk_in"][urban])
            assert np.any(before[urban].view(np.uint32) != after[urban].view(np.uint32))
            assert np.any(case["trb_urb4d_in"] != case["trb_urb4d_out"])

    # The monthly albedo switch must discriminate the remap's ALBBRD arm.
    assert fixtures["slucm_frc0"]["albbck_out"].tobytes() != fixtures[
        "slucm_frc0_monalb"]["albbck_out"].tobytes()


def test_fixture_receipts_match_published_bytes():
    sums = DIRECTORY / "FIXTURES.sha256"
    listed = set()
    for line in sums.read_text(encoding="ascii").splitlines():
        digest, relative = line.split(None, 1)
        path = DIRECTORY / relative
        assert path.is_relative_to(DIRECTORY)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative
        listed.add(relative)
    actual = {p.relative_to(DIRECTORY).as_posix() for p in DIRECTORY.rglob("*")
              if p.is_file() and (p.name == "MANIFEST.txt" or p.suffix == ".bin")}
    assert listed == actual


def _run_port(case, cp):
    from woof.core.noah import _F2D, _F3D, launch_noah, load_tables, pack_params

    shape = _layout(case["ivgtyp_in"]).shape
    device = {k: cp.asarray(_layout(case[f"{k}_in"]).copy())
              for k in ("ivgtyp", "isltyp")}
    derived = {
        "psfc": case["p8w3d_in"][:, 0, :],
        "sfcprs": (case["p8w3d_in"][:, 1, :] + case["p8w3d_in"][:, 0, :]) * np.float32(.5),
        "sfctmp": case["t3d_in"][:, 0, :],
        "qv1": case["qv3d_in"][:, 0, :],
        "dz8w1": case["dz8w_in"][:, 0, :],
    }
    for field in _F2D:
        if field in derived:
            value = _layout(derived[field])
        elif f"{field}_in" in case:
            value = _layout(case[f"{field}_in"])
        else:
            assert field == "reslin", field
            value = np.zeros(shape, np.float32)
        device[field] = cp.asarray(value.copy())
    for field in _F3D:
        device[field] = cp.asarray(_layout(case[f"{field}_in"]).copy())
    device["ebal"] = cp.zeros(shape, cp.int32)
    # UST is unchanged by the Noah kernel. Keeping it as a pass-through field
    # checks the FRC=0 WRF blend's identity without claiming Noah computes UST.
    device["ust"] = cp.asarray(_layout(case["ust_in"]).copy())
    categories = urban_category_set(isurban=int(case["isurban"]))
    mask = np.zeros(max(categories.urban) + 1, np.int32)
    mask[list(categories.urban)] = 1
    urban = dict(option=int(case["option"]), category_mask=cp.asarray(mask),
                 natural=categories.natural)
    for field in ("frc_urb2d", "ts_urb2d", "tsk_rural_bep"):
        urban[field] = cp.asarray(_layout(case[f"{field}_in"]).copy())
    for field in ("rural_q1", "rural_q2k", "rural_zlvl"):
        urban[field] = cp.zeros(shape, cp.float32)
    launch_noah(device, pack_params(load_tables()), float(case["dt"]),
                case["dzs_in"], isurban=int(case["isurban"]),
                isice=int(case["isice"]), itimestep=int(case["itimestep"]),
                xice_threshold=float(case["xice_threshold"]),
                frpcpn=bool(case["frpcpn"]), rdlai2d=bool(case["rdlai2d"]),
                opt_thcnd=int(case["opt_thcnd"]),
                usemonalb=bool(case["usemonalb"]),
                urban=urban if int(case["option"]) else None)
    outputs = {k: cp.asnumpy(v) for k, v in device.items()}
    outputs["tsk_rural_bep"] = cp.asnumpy(urban["tsk_rural_bep"])
    return outputs


@pytest.mark.gpu
@requires_gpu
def test_noah_hook_holds_measured_wrf_distance():
    import cupy as cp

    measured = {}
    for name, case in load(DIRECTORY).items():
        outputs = _run_port(case, cp)
        fields = list(UNBLENDED)
        if int(case["option"]) == 0 or "frc0" in name:
            fields.extend(SURFACE)
        elif int(case["option"]) == 2:
            fields.append("tsk_rural_bep")
        measured[name] = {field: ulp_table(outputs[field], _layout(case[f"{field}_out"]))[
            "max_ulp"] for field in fields}
        categories = urban_category_set(isurban=int(case["isurban"]))
        urban_columns = np.isin(case["ivgtyp_in"][:, 0], categories.urban)
        # Keep the same field table separately for urban and ordinary columns
        # so a new urban gap cannot be hidden by a non-urban maximum.
        for group, keep in (("urban", urban_columns), ("nonurban", ~urban_columns)):
            measured[f"{name}/{group}"] = {
                field: ulp_table(outputs[field][..., keep],
                                 _layout(case[f"{field}_out"])[..., keep])["max_ulp"]
                for field in fields}
    architecture = cp.cuda.Device().compute_capability
    recorded = MEASURED_MAX_ULP_BY_ARCHITECTURE.get(architecture)
    assert recorded is not None, (
        f"No authorized GPU measurement is recorded for architecture {architecture}.\n"
        "Review these distances against the existing Noah baseline, then pin this table:\n"
        + json.dumps(measured, indent=2, sort_keys=True))
    assert measured == recorded, (
        "Urban hand-over distance from WRF changed.\nMeasured:\n"
        + json.dumps(measured, indent=2, sort_keys=True)
        + "\nRecorded:\n" + json.dumps(recorded, indent=2, sort_keys=True))
