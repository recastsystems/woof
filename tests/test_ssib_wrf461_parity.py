"""SSiB (sf_surface_physics=8) against the WRF v4.6.1 column oracle.

The fixture is the gfortran build of the byte-unmodified
phys/module_sf_ssib.F driven through WRF's own CASE (SSIBSCHEME) driver arm
(tools/ssib_wrf461_oracle).  Strict arithmetic (GPUWM_WRF_EXACT=1) must
reproduce every carried state word at every step of every column bit for
bit; the default arm is graded against stated, measured tolerances.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from woof.core import ssib
from woof.core.fp32_ulp import monotone_fp32_key

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "oracles" / "ssib" / "ssib-wrf461-columns.npz"
KERNEL = ROOT / "woof" / "core" / "kernels" / "ssib.cu"


def _fixture():
    z = np.load(FIXTURE)
    meta = json.loads(bytes(z["meta"]).decode())
    return bytes(z["columns"]), bytes(z["out_ref"]), meta


def _enum(text: str, first: str) -> list[str]:
    body = re.search(r"enum\s*\{\s*(" + first + r".*?)\};", text, re.S).group(1)
    names = []
    for tok in body.replace("\n", " ").split(","):
        tok = tok.split("//")[0].strip()
        if not tok:
            continue
        names.append(tok.split("=")[0].strip())
    return names


def test_kernel_enums_are_the_python_layout():
    text = KERNEL.read_text(encoding="utf-8")
    c = _enum(text, "C_XLAND")
    f = _enum(text, "F_RAINNCV")
    s = _enum(text, "S_SMOIS1")
    assert c[:-1] == ["C_" + n.upper() for n in ssib.STATIC]
    assert c[-1] == "NC_SSIB"
    assert f[:-1] == ["F_" + n.upper() for n in ssib.FORCING]
    assert f[-1] == "NF_SSIB"
    head = [n for n in ssib.STATE if not n[:-1] in ssib.SNOW_LAYER_VARS]
    assert s[:len(head)] == ["S_" + n.upper() for n in head]
    assert s[len(head)] == "S_LAYER1"
    assert len(ssib.STATE) == len(head) + 48


def test_kernel_table_offsets_follow_the_table_file():
    text = KERNEL.read_text(encoding="utf-8")
    offsets = {m.group(1): int(m.group(2)) for m in
               re.finditer(r"#define T_(\w+)\s+(\d+)", text)}
    doc = json.loads(ssib.TABLE_PATH.read_text())
    at = 0
    for name in doc["layout"]:
        key = {"td_depth": "TDDEPTH"}.get(name, name.upper())
        assert offsets[key] == at, name
        at += len(doc["tables"][name]["f32_hex"])
    assert at == ssib.NTAB == int(re.search(r"#define NTAB_SSIB\s+(\d+)", text).group(1))


def test_table_file_is_pinned():
    digest = hashlib.sha256(ssib.TABLE_PATH.read_bytes()).hexdigest()
    assert digest == ssib.SSIB_TABLE_SHA256
    assert ssib.load_tables().size == ssib.NTAB


def test_one_bit_in_one_table_value_is_refused(tmp_path):
    doc = json.loads(ssib.TABLE_PATH.read_text())
    word = int(doc["tables"]["poros0"]["f32_hex"][6], 16) ^ 1
    doc["tables"]["poros0"]["f32_hex"][6] = f"{word:08x}"
    planted = tmp_path / "ssib_veg_tables.json"
    planted.write_text(json.dumps(doc, indent=1) + "\n")
    with pytest.raises(ValueError, match="pinned"):
        ssib.load_tables(planted)


def test_fixture_is_the_one_the_oracle_wrote():
    cols, ref, meta = _fixture()
    assert hashlib.sha256(cols).hexdigest() == meta["columns_sha256"]
    assert hashlib.sha256(ref).hexdigest() == meta["out_ref_sha256"]
    assert meta["ncol"] >= 64


def _ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in float32 units in the last place; +0 and -0 are equal."""
    return np.abs(monotone_fp32_key(a) - monotone_fp32_key(b))


def _replay(tmp_path, *, fixes, tables=None):
    cols, ref, meta = _fixture()
    path = tmp_path / "columns.bin"
    path.write_bytes(cols)
    out, errors = ssib.replay_columns(path, fixes=fixes, tables=tables)
    oracle = np.frombuffer(ref, np.float32).reshape(out.shape)
    return out, oracle, errors, meta


@pytest.mark.gpu
def test_strict_arm_is_zero_ulp_on_every_field_step_and_column(tmp_path):
    from woof import wrf_exact
    if not wrf_exact.ENABLED:
        pytest.skip("the 0 ULP proof runs under GPUWM_WRF_EXACT=1")
    out, oracle, errors, meta = _replay(tmp_path, fixes=0)
    assert errors & ~ssib.ERR_ICASE_UNSET == 0
    d = _ulps(out, oracle)
    bad = np.argwhere(out.view(np.uint32) != oracle.view(np.uint32))
    if bad.size:
        s, c, k = bad[0]
        raise AssertionError(
            f"{len(bad)} words differ; first at step {s + 1}, column "
            f"{meta['names'][c]}, field {ssib.STATE[k]}: gpu {out[s, c, k]!r} "
            f"oracle {oracle[s, c, k]!r}")


@pytest.mark.gpu
def test_planted_table_bit_is_caught_by_the_oracle(tmp_path):
    from woof import wrf_exact
    if not wrf_exact.ENABLED:
        pytest.skip("graded under GPUWM_WRF_EXACT=1")
    tab = ssib.load_tables().copy()
    doc = json.loads(ssib.TABLE_PATH.read_text())
    at = 0
    for name in doc["layout"]:
        if name == "poros0":
            break
        at += len(doc["tables"][name]["f32_hex"])
    # One bit of the porosity of SSiB type 7 (ground cover), which every
    # veg7 column reads every step.
    tab.view(np.uint32)[at + 6] ^= 1
    out, oracle, _, _ = _replay(tmp_path, fixes=0, tables=tab)
    assert (_ulps(out, oracle) > 0).any()


#: Measured float exp/log/pow and FMA arithmetic plus the sea-ice unit repair, with a
#: 1 percent ceiling margin on the 78-column fixture, as max |gpu - oracle| over
#: every step and column.  A field absent here must match exactly.
DEFAULT_TOLERANCE = {'smois1': 0.0046347308158874514, 'smois2': 0.003751044869422913, 'smois3': 9.03010368347168e-07, 'tslb1': 2.664411315917969, 'tslb2': 4.838740844726563, 'tslb3': 0.11632507324218751, 'snow': 21.75841204403201, 'canwat': 0.00043716613203287124, 'sfcrunoff': 0.006525216884911061, 'sfcevp': 0.28994365602731703, 'tsk': 4.838710021972656, 't2': 4.838710021972656, 'q2': 0.0004888973385095596, 'albedo': 0.47974910601973536, 'ust': 0.04318130865693092, 'hfx': 92.1603963470459, 'lh': 32.864799880981444, 'grdflx': 137.1807194519043, 'qfx': 1.3041586516919778e-05, 'br': 0.4640210282802582, 'u10': 0.10086023807525635, 'v10': 0.04741659283638001, 'snowh': 9.519234299659729e-07, 'tsk_save': 4.838710021972656, 'ssib_br': 0.4640210282802582, 'ssib_fm': 1.7067863988876344, 'ssib_fh': 1.8146111345291138, 'ssib_cm': 0.0008808315871283412, 'ssib_lhf': 32.864799880981444, 'ssib_shf': 92.1603963470459, 'ssib_ghf': 137.1807194519043, 'ssib_egs': 6.343785095214844, 'ssib_eci': 4.714679260253907, 'ssib_ect': 7.454128570556641, 'ssib_egi': 0.21468433380126953, 'ssib_egt': 1.289108295168262e-06, 'ssib_sdn': 0.000123291015625, 'ssib_sup': 202.99058166503906, 'ssib_lup': 21.403104553222658, 'ssib_wat': 0.002992817163467407, 'ssib_shc': 4.3720207786560055, 'ssib_shg': 0.9641280364990235, 'ssib_lai': 9.632110595703125e-07, 'alswvisdir': 0.7070005071908235, 'alswvisdif': 0.7070005673915147, 'alswnirdir': 0.2525022876262665, 'alswnirdif': 0.2525023177266121, 'swe': 1.8248334527015686e-07, 'snowden': 1.5411376953125e-05, 'snowdepth': 9.519234299659729e-07, 'dzo1': 7.863715291023255e-07, 'wo1': 1.552049070596695e-07, 'tssn1': 0.00018493652343750002, 'tssno1': 0.00018493652343750002, 'bwo1': 0.00114044189453125, 'bto1': 0.00114044189453125, 'cto1': 2.39875, 'fio1': 7.82608985900879e-07, 'flo1': 7.82608985900879e-07, 'bio1': 0.001109619140625, 'blo1': 0.000123291015625, 'ho1': 49.995, 'dzo2': 2.709031105041504e-07, 'wo2': 5.267560482025147e-08, 'tssn2': 0.00027740478515625, 'tssno2': 0.00027740478515625, 'bwo2': 0.00064727783203125, 'bto2': 0.00064727783203125, 'cto2': 1.325625, 'fio2': 1.9866228103637694e-06, 'flo2': 1.9866228103637694e-06, 'bio2': 0.00058563232421875, 'blo2': 0.00021479606628417968, 'ho2': 29.29, 'dzo3': 1.3168901205062867e-08, 'wo3': 3.127614036202431e-08, 'tssn3': 0.00225006103515625, 'tssno3': 0.0022808837890625, 'bwo3': 0.0015642547607421875, 'bto3': 0.0015642547607421875, 'cto3': 3.219375, 'fio3': 1.583278179168701e-05, 'flo3': 1.583278179168701e-05, 'bio3': 0.0005548095703125, 'blo3': 0.0016080808639526367, 'ho3': 16.4125, 'gz1oz0': 4.816055297851563e-07}


@pytest.mark.gpu
def test_default_arm_within_measured_tolerance(tmp_path):
    from woof import wrf_exact
    if wrf_exact.ENABLED:
        pytest.skip("the default arm is graded without GPUWM_WRF_EXACT")
    out, oracle, errors, meta = _replay(tmp_path, fixes=None)
    assert np.isfinite(out).all()
    assert np.isfinite(oracle).all()
    assert errors & ~ssib.ERR_ICASE_UNSET == 0
    for k, name in enumerate(ssib.STATE):
        err = float(np.max(np.abs(out[..., k].astype(np.float64)
                                  - oracle[..., k].astype(np.float64))))
        assert err <= DEFAULT_TOLERANCE.get(name, 0.0), (name, err)

@pytest.mark.gpu
def test_table_mutation_cannot_poison_a_later_default_launch():
    tab = ssib.load_tables()
    planted = tab.copy()
    planted.view(np.uint32)[0] ^= 1
    ssib.kernel(tab)
    ssib.kernel(planted)
    ssib.kernel(tab)
    mod = ssib._module_with_tables(tab.tobytes())
    actual = ssib._table_view(mod).get()
    np.testing.assert_array_equal(actual.view(np.uint32), tab.view(np.uint32))



@pytest.mark.gpu
def test_seaice_unit_repair_matches_explicitly_corrected_fortran(tmp_path):
    from woof import wrf_exact
    if not wrf_exact.ENABLED:
        pytest.skip('the operation-order proof runs under strict arithmetic')
    cols, ref, meta = _fixture()
    path = tmp_path / 'columns.bin'
    path.write_bytes(cols)
    actual, errors = ssib.replay_columns(path, fixes=ssib.FIX_SEAICE_SNOW_UNITS)
    with np.load(FIXTURE) as z:
        ref = bytes(z['out_default_units'])
    assert hashlib.sha256(ref).hexdigest() == meta['out_default_units_sha256']
    expected = np.frombuffer(ref,np.float32).reshape(actual.shape)
    assert errors == 0
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.gpu
def test_default_arm_reproduces_the_retained_field_receipt(tmp_path):
    from woof import wrf_exact
    if wrf_exact.ENABLED:
        pytest.skip("the retained default receipt uses ordinary arithmetic")
    from tools.ssib_wrf461_oracle.compare import grade
    expected = json.loads(FIXTURE.with_name("default-field-receipt.json").read_text())
    out, oracle, errors, meta = _replay(tmp_path, fixes=None)
    assert errors == 0
    actual = grade(out, oracle)
    assert sum(row["different_bits"] for row in actual.values()) == expected["different_bits"]
    for name, row in expected["fields"].items():
        assert {key: actual[name][key] for key in row} == row, name


def test_only_strict_compilation_disables_contraction():
    from woof.core.kernels import module_options
    from woof import wrf_exact
    expected = ("-std=c++17", "--fmad=false") if wrf_exact.ENABLED else ("-std=c++17",)
    assert module_options("ssib") == expected


@pytest.mark.gpu
def test_table_views_accept_both_cuda_module_interfaces():
    tab = ssib.load_tables()
    module = ssib._module_with_tables(tab.tobytes())
    raw_view = ssib._table_view(module).get()
    native_view = ssib._table_view(module.module).get()
    np.testing.assert_array_equal(raw_view.view(np.uint32), tab.view(np.uint32))
    np.testing.assert_array_equal(native_view.view(np.uint32), tab.view(np.uint32))
