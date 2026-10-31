"""LES wherever the spacing asks for it, on every door that makes a grid.

THE BREAKAGES PINNED HERE (gate law: each test names what it prevents).

1. ``woof downscale`` (WOOF nest-downs and moving step-downs) gave every
   child finer than 4 km the HRRR-shape suite's MYNN 1-D PBL at ANY
   spacing and never wrote the LES closure.  A child at LES spacing on a
   1-D PBL grew w_max from 10.7 to 23 m/s and went non-finite (2026-09-19,
   fead4fee9a); the 2026-10-23 200 m nest-down ran MYNN at 200 m.
2. ``woof domain`` skipped index 0 when writing the closure, so a
   real-data top grid at LES spacing (the hex add-nest's single WOOF
   grid, an API single grid) kept the suite's 1-D PBL.
3. The door's gray-zone advisory was computed before the closure was
   written and told a 3 -> 1 -> 0.333 -> 0.111 km tree that both
   sub-kilometre nests ran ``bl_pbl_physics = 5`` while d04 was PBL off.
4. No door turned the inflow generator on, so an LES nest under a PBL
   parent took smooth inflow (the 250 m child's spin-up fetch exceeded
   most of its domain; seeding reached the plateau in 4 of 4 frames
   against 1 of 4, for wall time inside run-to-run noise).

CPU only, no card, no data.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import tomllib

import pytest

from woof.cli import main as cli_main
from woof.downscale import child_physics_config
from woof.domain_wizard import (
    DEFAULT_LES_BELOW_M,
    gray_zone_advisory,
    gray_zone_headline,
    with_les_closure,
)
from woof.hrrr_route_inputs import ROUTE_DEFAULT_PHYSICS_PROFILE

from test_downscale_cli import (_SURFACE_PARENT_CONFIG,
                                _give_the_parent_a_real_projection,
                                _restart_evidence)
from test_offline_child import _history


_PARENT = {"mp_physics": 10, "bl_pbl_physics": 1, "sf_surface_physics": 2,
           "num_soil_layers": 4, "cu_physics": 1, "epssm": 0.2, "dt": 18.0,
           "km_opt": 4}


# --- 1. nest-down children ---------------------------------------------


@pytest.mark.parametrize("dx", [250.0, 200.0, 111.1, 37.0, 10.0])
def test_a_nest_down_child_at_les_spacing_runs_pbl_off_with_a_3d_closure(dx):
    cfg, profile, basis = child_physics_config(
        _PARENT, child_dx_m=dx, requested=None,
        les_below_m=DEFAULT_LES_BELOW_M)
    assert profile == ROUTE_DEFAULT_PHYSICS_PROFILE
    assert (cfg["bl_pbl_physics"], cfg["km_opt"], cfg["mix_isotropic"],
            cfg["isfflx"]) == (0, 3, 1, 1)
    assert "LES closure" in basis
    # The suite's microphysics and land surface stay: only turbulence moves.
    assert (cfg["mp_physics"], cfg["sf_surface_physics"]) == (28, 3)


def test_the_tke_closure_and_the_parent_suite_both_take_the_les_rule():
    cfg, _, _ = child_physics_config(_PARENT, child_dx_m=100.0,
                                     requested="parent", les_below_m=250.0,
                                     les_closure="tke")
    assert (cfg["mp_physics"], cfg["bl_pbl_physics"], cfg["km_opt"]) == (10, 0, 2)


def test_a_coarser_child_and_an_explicit_zero_keep_the_1d_pbl():
    cfg, _, basis = child_physics_config(_PARENT, child_dx_m=333.3,
                                         requested=None, les_below_m=250.0)
    assert cfg["bl_pbl_physics"] == 5 and "LES" not in basis
    cfg, _, _ = child_physics_config(_PARENT, child_dx_m=100.0,
                                     requested=None, les_below_m=0)
    assert cfg["bl_pbl_physics"] == 5


def test_the_downscale_door_writes_the_closure_into_the_derived_child(tmp_path,
                                                                    capsys):
    """THE REPRODUCTION, at the CLI: a 1 km parent stepped down x5 to 200 m."""

    start = datetime(1974, 4, 3, 12)
    for index in range(3):
        frame = tmp_path / f"wrfout_d01_1974-04-03_{12 + index:02d}_00_00"
        _history(frame, start + timedelta(hours=index), ny=18, nx=20)
        _give_the_parent_a_real_projection(frame, ny=18, nx=20)
    restart = _restart_evidence(
        tmp_path / "gpuwmrst_d01_1974-04-03_12_00_00.npz",
        dict(_SURFACE_PARENT_CONFIG, nx=20, ny=18, nz=2, grid_id=1, dt=3.0,
             run_seconds=7200.0, nested=False, specified=False))
    out = tmp_path / "child-run"
    rc = cli_main([
        "downscale", str(tmp_path), "--parent-restart", str(restart),
        "--point", "39.5,-84.0", "--physics-profile", "parent",
        "--ratio", "5", "--child-size", "20,20", "--child-levels", "4,2.5",
        "--hours", "0.25", "--output-interval-seconds", "900",
        "--parent-terrain", "--accept-parent-cadence",
        "--out", str(out), "--dry-run"])
    captured = capsys.readouterr()
    child = tmp_path / "child-run.child.toml"
    assert child.exists(), captured.err[-2000:]
    text = child.read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    flat = {**raw, **raw.get("run", {})}
    assert flat["bl_pbl_physics"] == 0 and flat["km_opt"] == 3, text[:400]
    assert "LES closure" in captured.out
    # The door no longer has a 1-D PBL child to warn about.
    assert "runs a 1-D boundary-layer scheme" not in captured.err
    assert rc == 0, captured.err[-2000:]


# --- 2. the top grid of a run -------------------------------------------


_ROOT_ONLY = """[shared]
bl_pbl_physics = 5
km_opt = 4

[[domain]]
grid_id = 1
parent_id = 0
dx = 200.0
specified = true
nested = false
"""


def test_a_real_data_top_grid_at_les_spacing_takes_the_closure():
    raw = tomllib.loads(with_les_closure(_ROOT_ONLY, 250.0))
    root = raw["domain"][0]
    assert (root["bl_pbl_physics"], root["km_opt"]) == (0, 3)
    # Nothing to seed it from: a top grid has no parent node.
    assert "inflow_perturbation" not in root


def test_an_idealised_top_grid_is_still_left_alone():
    ideal = _ROOT_ONLY.replace("specified = true", "specified = false")
    assert with_les_closure(ideal, 250.0) == ideal


# --- 3 + 4. the advisory and the seeding --------------------------------


_TREE = """[shared]
bl_pbl_physics = 5
km_opt = 4

[[domain]]
grid_id = 1
parent_id = 0
dx = 3000.0
specified = true

[[domain]]
grid_id = 2
parent_id = 1
parent_grid_ratio = 3

[[domain]]
grid_id = 3
parent_id = 2
parent_grid_ratio = 3

[[domain]]
grid_id = 4
parent_id = 3
parent_grid_ratio = 3

[[domain]]
grid_id = 5
parent_id = 4
parent_grid_ratio = 3
"""


def test_the_first_les_nest_under_a_pbl_parent_is_seeded_and_the_next_is_not():
    domains = tomllib.loads(with_les_closure(_TREE, 250.0))["domain"]
    assert [d.get("bl_pbl_physics") for d in domains] == [None, None, None, 0, 0]
    # d04 (111 m) sits under d03's 1-D PBL; d05 (37 m) under LES d04.
    assert [d.get("inflow_perturbation") for d in domains] == [
        None, None, None, True, None]
    off = tomllib.loads(with_les_closure(_TREE, 250.0, "tke",
                                         inflow_seeding=False))["domain"]
    assert off[3]["km_opt"] == 2 and "inflow_perturbation" not in off[3]


def test_a_streamed_tree_is_not_seeded_until_seeding_is_measured_there():
    streamed = _TREE + '\n[tiles]\nmode = "auto"\n'
    domains = tomllib.loads(with_les_closure(streamed, 250.0))["domain"]
    assert domains[3]["bl_pbl_physics"] == 0
    assert "inflow_perturbation" not in domains[3]


def test_the_advisory_names_les_domains_apart_from_gray_zone_ones():
    chain = [3.0, 1.0, 1 / 3, 1 / 9]
    shared = {"bl_pbl_physics": 5}
    lines = gray_zone_advisory(chain, shared, 250.0)
    assert lines[0].startswith("GRAY ZONE: 1 domain(s)")
    assert "d03 333 m" in lines[0] and "d04" not in lines[0]
    assert lines[1].startswith("LES: 1 domain(s) at or below 250 m (d04 111 m)")
    heads = gray_zone_headline(chain, shared, 250.0)
    assert len(heads) == 2
    # Without the threshold the advisory reads exactly as before.
    assert gray_zone_advisory(chain, shared)[0].startswith(
        "GRAY ZONE: 2 domain(s) refine below 1 km (finest 111 m) with")


def test_a_tree_entirely_at_les_spacing_has_no_gray_zone_line():
    lines = gray_zone_advisory([0.2, 0.04], {"bl_pbl_physics": 5}, 250.0)
    assert len(lines) == 1 and lines[0].startswith("LES: 2 domain(s)")
    assert "coarser domains" not in lines[0]


def test_the_seeded_tree_loads_through_the_shared_config_gate(tmp_path):
    """The written keys are ones the experiment loader admits as a tree."""
    from woof.domain_wizard import _domain_spacings_m
    text = with_les_closure(_TREE, 250.0)
    assert [round(v, 1) for v in _domain_spacings_m(text)] == [
        3000.0, 1000.0, 333.3, 111.1, 37.0]
    from woof.core.inflow_perturbation import parent_pbl_refusal
    domains = tomllib.loads(text)["domain"]
    parent_pbl = [d.get("bl_pbl_physics", 5) for d in domains]
    for index, domain in enumerate(domains):
        if domain.get("inflow_perturbation"):
            assert parent_pbl_refusal(parent_pbl[index - 1]) is None
