"""dist44 door facts (processors only): a route's own kernel set and images stand in the row's place, the radiation
step's images are laid over a copy of the row's scheme bundle, the fast-against-exact figures are printed only for
the plan and window they were measured on, the host need is the one the command will really use and the refusal names
it, and the cloud scheme's gated derivative is a named option that is off by default."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex import ad_doors as d
from woof.hex.ad import assets

STORM = "hrrrv5-q3.96.38278.n40.60w91.30"
KS = "storm-bs5-lean2-astra2-dynE5-sp74"
KSD = KS + "-ddr"   # 0.6.0: the default route's set takes three dynamics derivative images in double arithmetic


def _row():
    return d.load_table()["rows"][STORM]


def _ns(**kw):
    base = dict(plan=None, init=None, boundary_layer=None, cloud_derivative=None, checked_forward=False, row=STORM,
                mesh=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_default_route_runs_its_own_kernel_set_and_the_exact_route_keeps_the_rows():
    row = _row()
    fast, taken = d.route_row(row, _ns())
    assert fast["assets"]["kernels"] == f"latest/kset/{KSD}/kernels.json"
    assert fast["assets"]["cubins"] == f"latest/kset/{KSD}/cubins"
    lay = [t for t in taken if t["option"] == "--boundary-layer"]   # dist46: the cloud default's images beside them
    assert {t["asset"] for t in lay} == {"kernels", "cubins"} and all(t["route"] == "tape-free" for t in lay)
    assert row["assets"]["kernels"].endswith("dyn6/kernels.json")          # the table's row is not changed
    exact, taken = d.route_row(row, _ns(boundary_layer="taped"))    # 0.6.0: the exact route keeps the single-arithmetic set
    assert exact["assets"]["kernels"] == f"latest/kset/{KS}/kernels.json"   # its exact reference bytes were read on it
    assert all(t["route"] == "taped" for t in taken if t["option"] == "--boundary-layer")
    assert "30 minutes" in row["gradient_line"]["boundary_layer"]["routes"]["taped"]["window_limit"]


def test_every_file_a_route_names_is_in_a_pinned_download():
    row, pins = _row(), assets.load_pins()
    holds = assets.pinned_holds()
    gl = row["gradient_line"]
    for routes in (gl["boundary_layer"]["routes"], gl["cloud_derivative"]["routes"]):
        for route in routes.values():
            items = list((route.get("assets") or {}).values()) + list(
                ((route.get("scheme_bundle") or {}).get("images") or {}).values())
            for item in items:
                assert any(item["path"].startswith(h) for h in holds), item["path"]
                assert item["arch"] == "sm_120"
            for key in ("assets_pinned_in",):
                if route.get(key):
                    assert route[key] in pins["assets"]
    want = assets.wanted(pins, d.example_reads(row, _ns()))
    assert "ad-aux-rad-step-sm120-v1.tar.gz" in want and f"ad-kset-{KSD}-v1.tar.gz" in want
    assert "ad-aux-cloud-number-sm120-v1.tar.gz" in want               # 0.6.0: the default on this row
    assert "ad-aux-cloud-gated-sm120-v2.tar.gz" not in want            # gated is a named word now
    assert "ad-aux-cloud-gated-sm120-v2.tar.gz" in assets.wanted(
        pins, d.example_reads(row, _ns(cloud_derivative="gated")))
    assert "ad-aux-cloud-gated-sm120-v2.tar.gz" not in assets.wanted(
        pins, d.example_reads(row, _ns(cloud_derivative="shipped")))
    # a window with the gradient line off (--checked-forward, another door) fetches no route file
    off = assets.wanted(pins, d.example_reads(row, _ns(), False))
    assert "ad-aux-rad-step-sm120-v1.tar.gz" not in off and f"ad-kset-{KSD}-v1.tar.gz" not in off


def test_the_cloud_derivative_is_a_named_option_and_off_by_default():
    row, table = _row(), d.load_table()
    cd = row["gradient_line"]["cloud_derivative"]
    assert cd["default"] == "number" and sorted(cd["routes"]) == ["gated", "number", "shipped"]   # 0.6.0: number
    name, route, default = d.cloud_derivative_route(row, "shipped")
    assert (name, default) == ("shipped", False) and not route.get("assets")
    fast, taken = d.route_row(row, _ns(cloud_derivative="gated"))
    assert {t["asset"] for t in taken} == {"kernels", "cubins", "tempo_block_cubin", "tempo_tlf_cubin", "tempo_cubin"}
    assert fast["assets"]["tempo_block_cubin"].endswith("cloud-gated/c2chain-0xc6-rd3600n100.cubin")
    assert fast["assets"]["tempo_cubin"].endswith("cloud-gated/split2-0xc6-rd3600n100.cubin")
    exact, taken = d.route_row(row, _ns(cloud_derivative="gated", boundary_layer="taped"))   # both routes take it
    assert {t["asset"] for t in taken} == {"kernels", "cubins", "tempo_block_cubin", "tempo_tlf_cubin", "tempo_cubin"}
    d.refuse_speed_words(_ns(cloud_derivative="gated"), table)
    with pytest.raises(d.AdDoorError, match="--cloud-derivative gated: row hrrrv5-x1.40962 offers no cloud-derivative"):
        d.refuse_speed_words(_ns(cloud_derivative="gated", row="hrrrv5-x1.40962"), table)
    with pytest.raises(d.AdDoorError, match="--checked-forward turns the gradient line off"):
        d.refuse_speed_words(_ns(cloud_derivative="gated", checked_forward=True), table)
    with pytest.raises(d.AdDoorError, match="--cloud-derivative gated runs files built for sm_120 cards only"):
        d.refuse_routes_card_family(STORM, {**row, "gradient_line": {**row["gradient_line"], "boundary_layer": {}}},
                                    _ns(cloud_derivative="gated", boundary_layer="taped"), "sm_89", "0")
    with pytest.raises(d.AdDoorError, match=r"--boundary-layer taped runs files built for sm_120 cards only.*"
                                            r"--checked-forward \(the row's own kernel set"):    # dist45: the same set
        d.refuse_routes_card_family(STORM, row, _ns(boundary_layer="taped"), "sm_89", "0")
    # six of seven measured directions within 3 per cent with the option; the near-ground wind one is not
    gated = cd["routes"]["gated"]["measured"]["directions"]
    near = [k for k, (_, after) in gated.items() if abs(after - 1.0) <= 0.0301]
    assert len(gated) == 7 and len(near) == 6 and abs(gated["box-wide low-level wind"][1] - 1.367) < 1e-9


def test_the_fast_against_exact_figures_are_printed_only_where_they_were_measured():
    route = _row()["gradient_line"]["boundary_layer"]["routes"]["tape-free"]
    assert "1.2e-6" not in json.dumps(route) and "7.8e-6" not in json.dumps(route)
    plan = f"the row's example plan ({_row()['example_plan']})"
    # dist48: the default route runs another form; the figures were measured with the form before it, so none is printed
    for ctx in (SimpleNamespace(plan_how=plan, steps=59), SimpleNamespace(plan_how=plan, steps=12),
                SimpleNamespace(plan_how="given (--plan)", steps=59)):
        words, measured = d.route_error_words(ctx, route)
        assert not measured and "e-06" not in words and "e-05" not in words
    fig = route["error_measured_before"]["figures"]
    assert max(fig["dJ_dtheta"]) < 3.1e-06 and max(fig["dJ_dpressure_p_stored"]) <= 9.8e-05


def test_the_host_need_is_the_one_the_command_will_use_and_the_refusal_names_it(tmp_path, monkeypatch):
    row = _row()
    with_check = d.store_need(row, 59, 400_000, line_on=True)
    alone = d.store_need(row, 59, 400_000, line_on=True, check_runs=False)
    assert with_check["host_mib"] == 258_057 and alone["host_mib"] == 158_617
    assert "answer check" in with_check["host_for"] and "alone" in alone["host_for"]
    assert alone["host_check_mib"] == 258_057 and with_check["host_gradient_mib"] == 158_617
    with pytest.raises(d.HostMemoryError, match=r"needs about 258,057 MiB \(the gradient with its answer check\).*"
                                                r"The gradient alone needs about 158,617 MiB: --no-self-dot"):
        d.refuse_host_too_small(STORM, 258_057, 200_000, what="the gradient with its answer check",
                                smaller="The gradient alone needs about 158,617 MiB: --no-self-dot runs it without "
                                        "its answer check (the receipt then reads not checked).")
    d.refuse_host_too_small(STORM, 158_617, 200_000, what="the gradient alone")
    # judged before any download: the command's words, then a passing verdict kept here by this door program
    monkeypatch.setenv(d.CHECKS_ENV, str(tmp_path / "checks"))
    plain = argparse.Namespace(no_self_dot=False, strict_from=None, self_dot=False)
    assert d.answer_check_expected(STORM, 59, plain, "sensitivity")[0] is True
    assert d.answer_check_expected(STORM, 59, argparse.Namespace(no_self_dot=True), "sensitivity")[0] is False
    (tmp_path / "checks").mkdir()
    key = {"row": STORM, "steps": 59, "door_code": {"ad_doors_sha256": hashlib.sha256(Path(d.__file__).read_bytes()).hexdigest()}}
    (tmp_path / "checks" / "a.json").write_text(json.dumps({"pass": True, "key": key}))
    assert d.answer_check_expected(STORM, 59, plain, "sensitivity")[0] is False
    assert d.answer_check_expected(STORM, 12, plain, "sensitivity")[0] is True         # another window length
    assert d.answer_check_expected(STORM, 59, argparse.Namespace(self_dot=True), "sensitivity")[0] is True
    (tmp_path / "checks" / "a.json").write_text(json.dumps({"pass": False, "key": key}))
    assert d.answer_check_expected(STORM, 59, plain, "sensitivity")[0] is True         # a failed verdict stands for none


def test_the_radiation_images_are_laid_over_a_copy_of_the_rows_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("WOOF_HEX_CACHE_ROOT", str(tmp_path / "cache"))
    row = _row()
    route = json.loads(json.dumps(row["gradient_line"]["boundary_layer"]["routes"]["tape-free"]))
    # dist48: the route also lays its boundary-layer form's p3 images (test_ad_door_dist48_bl.py); the radiation pair here
    route["scheme_bundle"]["images"] = {r: i for r, i in route["scheme_bundle"]["images"].items() if r.startswith("p7/")}
    root = tmp_path / "assets"
    base = root / "latest/bundle/gen-B9"
    (base / "p7").mkdir(parents=True)
    files = {"schemes.json": b"{}", "p7/rsw_taumol.sm_120.cubin": b"old-sw", "p7/taumol.sm_120.cubin": b"old-lw",
             "p8-sm_120.cubin": b"p8"}
    for rel, b in files.items():
        (base / rel).write_bytes(b)
    (base / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(b).hexdigest()}  {rel}\n" for rel, b in sorted(files.items())))
    new = {"p7/rsw_taumol.sm_120.cubin": b"new-sw", "p7/taumol.sm_120.cubin": b"new-lw"}
    for rel, item in route["scheme_bundle"]["images"].items():
        p = root / item["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(new[rel])
        item["sha256"] = hashlib.sha256(new[rel]).hexdigest()
    ctx = SimpleNamespace(root=root, assets={"bundle": base}, row_name=STORM)
    out = d.route_scheme_bundle(ctx, "--boundary-layer", "tape-free", route)
    assert out != base and (out / "p7/taumol.sm_120.cubin").read_bytes() == b"new-lw"
    assert (out / "p8-sm_120.cubin").read_bytes() == b"p8" and (base / "p7/taumol.sm_120.cubin").read_bytes() == b"old-lw"
    for line in (out / "SHA256SUMS").read_text().splitlines():        # every line of the copy's listing holds
        want, _, rel = line.partition("  ")
        assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == want
    assert d.route_scheme_bundle(ctx, "--boundary-layer", "tape-free", route) == out      # made once
    # an image that is not the pinned build is refused by name, with the exact route's word
    (root / route["scheme_bundle"]["images"]["p7/taumol.sm_120.cubin"]["path"]).write_bytes(b"other")
    with pytest.raises(d.AdDoorError, match="is not the build the table pins for --boundary-layer tape-free.*"
                                            "--boundary-layer taped runs without it"):
        d.route_scheme_bundle(ctx, "--boundary-layer", "tape-free", route)
    assert d.route_scheme_bundle(ctx, "--boundary-layer", "taped", {}) is None


def test_the_parser_takes_the_cloud_option():
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="door"))
    args = parser.parse_args(["sensitivity", "--out", "x", "--cloud-derivative", "gated"])
    assert args.cloud_derivative == "gated"
    assert parser.parse_args(["sensitivity", "--out", "x"]).cloud_derivative is None
    with pytest.raises(SystemExit):
        parser.parse_args(["sensitivity", "--out", "x", "--cloud-derivative", "other"])


def test_a_window_after_the_sessions_program_is_ended_waits_for_the_card(monkeypatch):
    """The card job's defect: the program a command ended gives the card back seconds later; the window that
    follows is admitted with a wait, where a command with no session is refused at once as before."""
    free = iter([13210, 13210, 97000])
    monkeypatch.setattr(d, "free_mib", lambda card: next(free))
    monkeypatch.setattr(d, "refuse_card_too_small", lambda card, need: None)
    monkeypatch.setattr(d.time, "sleep", lambda s: None)
    got = d.admit_card("0", 96000, d.SESSION_RELEASE_WAIT_S)
    assert got["free_mib"] == 97000
    monkeypatch.setattr(d, "free_mib", lambda card: 13210)
    with pytest.raises(d.AdDoorError, match="card 0 has 13210 MiB free"):
        d.admit_card("0", 96000, 0.0)
    import inspect
    src = inspect.getsource(d.Context.window)
    assert "release_wait = SESSION_RELEASE_WAIT_S" in src and "max(float(self.args.wait_card), release_wait)" in src
