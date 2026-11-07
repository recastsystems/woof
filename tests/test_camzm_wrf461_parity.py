"""The CAM Zhang-McFarlane (cu_physics = 7) oracle fixtures: integrity,
coverage, constants and the measured size of the declared divergence.

The fixtures are WRF v4.6.1 camzm_driver's own words
(tools/camzm_wrf461_oracle, README there).  The port is graded on them in
tests/test_camzm_launcher_wrf461_parity.py; this module needs no GPU.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from woof.verify.camzm_oracle import (FIXTURE_DIR, GROUPS, dims, fixture,
                                       step_arrays)


@pytest.fixture(autouse=True)
def _generated_streams_required(request):
    if request.node.name == "test_camzm_default_compilation_preserves_the_strict_operation_order":
        return
    if not (FIXTURE_DIR / "zm-uw12km-stock.bin").is_file():
        pytest.skip("raw CU7 streams were disposed; set CAMZM_ORACLE_DIR")


def test_every_committed_fixture_is_the_one_the_oracle_wrote():
    sums = (FIXTURE_DIR / "SHA256SUMS").read_text().split("\n")
    seen = 0
    for line in sums:
        if not line.strip():
            continue
        digest, name = line.split()
        data = (FIXTURE_DIR / name.removeprefix("./")).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name
        seen += 1
    # per tile: record and stock (.bin + .manifest each) and the case file
    assert seen == 5 * len(GROUPS)


def test_the_oracle_covers_the_regimes_the_brief_names():
    index = json.loads((FIXTURE_DIR / "cases-index.json").read_text())
    families = [f for g in GROUPS for f in index[f"cases-{g}"]["families"]]
    assert len(families) >= 64
    for family in ("tropical", "severe", "elevated", "marginal", "dry",
                   "terrain", "cold", "stable", "deeppbl", "toptrop",
                   "moistsat", "pbltrap"):
        assert family in families, family
    seen = {"no_trigger": 0, "convects": 0, "capelmt_edge": 0,
            "top_of_model": 0, "no_deep_pbl": 0, "snow": 0, "held": 0,
            "multi_step": 0, "myj_arm": 0, "uw_arm": 0, "ocean": 0}
    for g in GROUPS:
        fx = fixture(g)
        ncol, nk, nsteps = dims(fx)
        computed = 0
        arm = int(fx["const/bl_pbl_physics"][0])
        seen["myj_arm" if arm == 2 else "uw_arm"] += ncol
        for s in range(1, nsteps + 1):
            a = step_arrays(fx, s)
            if not a["ran"]:
                seen["held"] += 1
                continue
            computed += 1
            o = a["out"]
            conv = o["lengath2d"] == 1
            seen["no_trigger"] += int(np.sum(~conv))
            seen["convects"] += int(np.sum(conv))
            seen["capelmt_edge"] += int(np.sum((o["cape"] > 60) & (o["cape"] < 80)))
            seen["top_of_model"] += int(np.sum(conv & (o["jt2d"] == 3)))
            dead = conv & np.all(o["zmmu"] == 0, axis=0)
            seen["no_deep_pbl"] += int(np.sum(dead))
            seen["snow"] += int(np.sum(np.any(o["zmflxsnw"] > 0, axis=0)))
            seen["ocean"] += int(np.sum(conv & (a["xland"] == 2)))
        seen["multi_step"] += int(computed >= 2)
    for key, count in seen.items():
        assert count > 0, (key, seen)
    assert seen["multi_step"] == len(GROUPS)


def test_the_constants_are_the_shared_cam5_layers_words():
    from woof.core import cam5_support as cam
    fx = fixture("uw12km")
    for name, value in (("cpair", cam.CPAIR), ("gravit", cam.GRAVIT),
                        ("rair", cam.RAIR), ("latvap", cam.LATVAP),
                        ("latice", cam.LATICE), ("epsilo", cam.EPSILO),
                        ("rh2o", cam.RH2O), ("tmelt", cam.TMELT),
                        ("cpwv", cam.CPWV), ("cpliq", cam.CPLIQ)):
        assert float(fx[f"const/{name}"][0]).hex() == value.hex(), name


def test_the_saturation_table_is_the_shared_layers_table():
    """estblf at the table nodes, recomputed from the shared ESTBL words
    with wv_saturation's own interpolation, equals WRF v4.6.1's lookup."""
    from woof.core.cam5_support import ESTBL
    fx = fixture("uw12km")
    probe = fx["const/estblf_probe"]
    table = np.asarray(ESTBL, dtype=np.float64)
    for n in range(203):
        td = 173.16 + float(n)
        e = max(min(td, 375.16), 173.16)
        i = int(e - 173.16) + 1
        ai = float(np.trunc(e - 173.16))
        want = ((173.16 + ai - e + 1.0) * table[i - 1]
                - (173.16 + ai - e) * table[i])
        assert want.hex() == float(probe[n]).hex(), n


@pytest.mark.parametrize("group", GROUPS)
def test_the_declared_divergence_is_measured_not_assumed(group):
    """Stock WRF and the fixture of record agree wherever ZM did not
    convect at step 1 (the inputs are identical there), and differ on the
    convecting columns: the landfrac fix moves exactly those.  The
    oracle build additionally proves the fix is precisely the inversion of
    XLAND's one test (README: exchanged-XLAND identity)."""
    fx = fixture(group, stock=False)
    stock = fixture(group, stock=True)
    a = step_arrays(fx, 1)
    b = step_arrays(stock, 1)
    conv = a["out"]["lengath2d"] == 1
    moved = np.zeros(conv.shape, dtype=bool)
    for name, value in a["out"].items():
        w = np.asarray(value)
        other = np.asarray(b["out"][name])
        if w.dtype == np.float32:
            d = w.view(np.uint32) != other.view(np.uint32)
        else:
            d = w != other
        d = d.reshape(-1, conv.size).any(axis=0) if d.ndim > 1 else d
        moved |= d
    assert not np.any(moved & ~conv), "a non-convecting column moved"
    live = conv & (a["out"]["pratec"] > 0)
    assert np.all(moved[live]), "a precipitating column did not move"


@pytest.mark.parametrize("group", GROUPS)
def test_the_momtran_first_downdraft_level_is_unreachable(group):
    """momtran's k = 2 downdraft branch (fixed parenthesis) needs a
    downdraft at CAM interface 2, i.e. WRF level nk-1; cldprp never makes
    one (jd >= 5), and no fixture has one."""
    fx = fixture(group)
    _, nk, nsteps = dims(fx)
    for s in range(1, nsteps + 1):
        md = step_arrays(fx, s)["out"]["md3d"]
        assert np.all(md[nk - 2] == 0) and np.all(md[nk - 1] == 0)


def test_camzm_default_compilation_preserves_the_strict_operation_order():
    from woof.core.kernels import module_options
    from woof.numerics import is_strict
    expected = (("-std=c++17", "--fmad=false", "--ftz=false") if is_strict()
                else ("-std=c++17", "--ftz=false", "-DGPUWM_CAM_FAST=1"))
    assert module_options("camzm") == expected
    assert module_options("lake") == ("-std=c++17", "--fmad=false")
    assert module_options("uwpbl") == ("-std=c++17",)
