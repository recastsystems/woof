"""F12: the 3 km storm row runs the transport limiter live (its exact derivative) and holds it past fct_live_max_s.

The held limiter put the 3 km adjoint-test's bump 1.1 per cent off the real model at every step size (dist49 at3: 1.0110
to 1.0142, outside the 1 per cent bar); the limiter's exact derivative grows on windows past about 30 minutes at 3 km.
"""
import json
from pathlib import Path

import woof.hex.ad_doors as d

STORM = "hrrrv5-q3.96.38278.n40.60w91.30"


def _words(argv, flag):
    return argv[argv.index(flag, 2) + 1]


def _toy(tmp_path: Path, live_max):
    (tmp_path / "k").write_text("x")
    row = {"asset_args": {"--kernels": "kernels"}, "asset_env": {}, "window_args": ["--fct-frozen", "0", "--anchors", "2"],
           "dt_s": 15.0}
    if live_max is not None:
        row["fct_live_max_s"] = live_max
    table = {"doors": {"adjoint-test": {"window_args": []}}}
    return table, row, {"kernels": tmp_path / "k"}


def test_live_inside_the_limit_held_past_it(tmp_path):
    table, row, assets = _toy(tmp_path, 1800)
    for steps, want in ((59, "0"), (120, "0"), (121, "1"), (239, "1")):
        argv, _ = d.window_command(table, row, assets, "adjoint-test", Path("/bin/true"), steps, tmp_path / "o", [])
        assert _words(argv, "--fct-frozen") == want, steps


def test_no_limit_leaves_the_row_word(tmp_path):
    table, row, assets = _toy(tmp_path, None)
    argv, _ = d.window_command(table, row, assets, "adjoint-test", Path("/bin/true"), 239, tmp_path / "o", [])
    assert _words(argv, "--fct-frozen") == "0"


def test_the_storm_row_runs_the_limiter_live():
    table = d.load_table()
    row = table["rows"][STORM]
    w = row["window_args"]
    assert w[w.index("--fct-frozen") + 1] == "0"
    assert row["fct_live_max_s"] == 1800 and row["dt_s"] == 15.0
    assert "FCT limiter coefficients held" not in row["what"]


def test_made_limited_area_classes_run_the_limiter_live():
    table = d.load_table()
    assert table["rows"]["hrrrv5-any"]["fct_live_max_s"] == 1800
    for cls in table["suites"]["hrrrv5"]["window_classes"]:
        w = cls["window_args"]
        if cls.get("whole") and cls.get("mesh_kind") == "limited-area":
            assert w[w.index("--fct-frozen") + 1] == "0", cls.get("from_recipe")


def test_made_row_recorded_before_the_rule_runs_it(tmp_path):
    # a made limited-area row cached (and a plan recorded with it) before 0.6.2: no fct_live_max_s, --fct-frozen 1
    table, row, assets = _toy(tmp_path, None)
    row["window_args"] = ["--fct-frozen", "1", "--anchors", "2"]
    row["dt_s"] = 150.0
    row["generated"] = {"from_row": "hrrrv5-any", "window_class": {"mesh_kind": "limited-area"}}
    for steps, want in ((11, "0"), (12, "0"), (13, "1"), (24, "1")):
        argv, _ = d.window_command(table, row, assets, "adjoint-test", Path("/bin/true"), steps, tmp_path / "o", [])
        assert _words(argv, "--fct-frozen") == want, steps
    row["generated"]["window_class"]["mesh_kind"] = "global"
    argv, _ = d.window_command(table, row, assets, "adjoint-test", Path("/bin/true"), 11, tmp_path / "o", [])
    assert _words(argv, "--fct-frozen") == "1"


def test_made_limited_area_rows_default_to_the_guarded_boundary_layer():
    table = d.load_table()
    assert d.boundary_layer_route(table["rows"]["hrrrv5-any"], None)[0] == "taped"   # the global classes
    for recipe, want in (("3km", "guarded"), ("3km-bounded", "guarded"), ("120", "taped")):
        made = {"generated": {"from_row": "hrrrv5-any", "suite": "hrrrv5", "window_class": {"from_recipe": recipe}}}
        name, route, is_default = d.boundary_layer_route(made, None)
        assert (name, is_default) == (want, True), recipe
    name, route, _ = d.boundary_layer_route(made, "guarded")
    assert route["window_env"]["AD_P3_HOLD_OUT"] == "rncblten,rniblten,rnifablten,rnwfablten"


def test_made_limited_area_rows_hold_the_boundary_layer_past_30_minutes(capsys):
    import argparse
    assert d.load_table()["rows"]["hrrrv5-any"]["boundary_layer"]["default_live_max_s"] == 1800
    made = {"dt_s": 150.0, "generated": {"from_row": "hrrrv5-any", "suite": "hrrrv5",
                                         "window_class": {"from_recipe": "3km", "mesh_kind": "limited-area"}}}
    for steps, want in ((11, None), (12, None), (13, "taped"), (24, "taped")):
        args = argparse.Namespace(boundary_layer=None)
        d.boundary_layer_for_window(args, made, steps, 150.0, "adjoint-test")
        assert args.boundary_layer == want, steps
    assert "boundary layer taped for this window (1 h" in capsys.readouterr().out
    args = argparse.Namespace(boundary_layer="guarded")   # a route the command names runs on any window
    d.boundary_layer_for_window(args, made, 24, 150.0, "sensitivity")
    assert args.boundary_layer == "guarded"
    glob = {"dt_s": 150.0, "generated": {"from_row": "hrrrv5-any", "suite": "hrrrv5",
                                         "window_class": {"from_recipe": "120"}}}
    args = argparse.Namespace(boundary_layer=None)   # the global classes' default is taped already
    d.boundary_layer_for_window(args, glob, 24, 150.0, "adjoint-test")
    assert args.boundary_layer is None


def test_a_proven_row_prints_the_species_reading_beside_its_one_unit_verdict(capsys):
    check = {"how": "own", "value": 1.786e-07, "roundings": -1.50, "bar": 4.77e-07, "bar_roundings": 4, "pass": True,
             "allocations": 22, "units": "fields", "species_units": False, "one_unit": None,
             "species_beside": {"value": 5.116e-07, "roundings": 4.29, "pass": False}}
    d.print_strict(check)
    out = capsys.readouterr().out
    assert "-1.50 roundings" in out and "PASS" in out
    assert "this proven row's verdict" in out and "+4.29 roundings" in out


def test_the_species_reading_says_its_conditioning_from_the_real_3km_windows():
    """A 3 km run (19 Nov 2026, merged build): the species-unit window read +4.29 roundings of the dot, FAIL,
    and the one-unit window -1.50, PASS. Their reverse sides' absolute terms were 128.58 and 34.11 against dots of 22
    and 12: the misses are +0.73 and -0.53 roundings of those terms. The door prints that beside the reading and does
    not judge on it."""
    from woof.hex import ad_doors as d
    species = {"y_Mx": 22.000000004930783, "x_MTy": 22.000011261233528,
               "conditioning": {"abs_terms_ad": 128.581547902415, "abs_terms_tl": 22.000000004930783}}
    one = {"y_Mx": 12.000000019532198, "x_MTy": 11.999997876830875,
           "conditioning": {"abs_terms_ad": 34.110355592351986, "abs_terms_tl": 12.000000019532198}}
    ts, to = d.dot_terms(species), d.dot_terms(one)
    assert abs(ts["ratio"] - 5.845) < 1e-3 and abs(ts["roundings_of_terms"] - 0.7344) < 1e-3
    assert abs(to["ratio"] - 2.843) < 1e-3 and abs(to["roundings_of_terms"] + 0.5270) < 1e-3
    assert "5.8 times the dot" in d.terms_words(ts) and "+0.73 roundings of those terms" in d.terms_words(ts)
    assert d.dot_terms({"y_Mx": 1.0, "x_MTy": 1.0}) is None and d.terms_words(None) == ""
    bar = d._dot_bar.bar_of(d.load_table(), "roundings")
    if True:
        doc = {"self_dot": {"units": "fields", "allocations": 22, "words": 1},
               "dots": [{**species, "rel_aligned": 5.116498628728264e-07}]}
        c = d.self_dot_check(doc, bar)
        assert c["pass"] is False and c["terms"] == ts          # the verdict stays on the dot's own scale


def test_species_units_need_the_window_end_boundary_in_the_field_map(tmp_path):
    """A 12 km run (19 Nov 2026): the 12 km README example's map holds boundaries 1-15; a 15-step window ends on 16,
    so the species-unit window stopped after 0 s. Without that boundary the door does not ask for species units."""
    import json as _json
    from woof.hex import ad_doors as d
    m = {"fields": {"scalars": {"shape": [13, 59, 2890], "strides": [682040, 11560, 4]}},
         "boundaries": {str(b): {} for b in range(1, 16)}}
    (tmp_path / "alloc-map.json").write_text(_json.dumps(m))
    assert d.self_dot_species_planes(tmp_path) is True
    assert d.self_dot_species_planes(tmp_path, 14) is True
    assert d.self_dot_species_planes(tmp_path, 15) is False
