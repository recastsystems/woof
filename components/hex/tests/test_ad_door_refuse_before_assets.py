# SPDX-License-Identifier: Apache-2.0
"""A differentiable-engine door that cannot run refuses before it fetches the row and plan (about 940 MB on first use).

THE BREAKAGES (a fresh install with only the two pinned files, 2026-11-11):
* ``--mesh x1.40962`` (the 120 km row) fetched both files and then refused: "ships no example plan" with no plan,
  a missing ``latest/...`` file with ``--plan``; with ``--init`` it went on to the recorder first.
* On a card of compute capability 8.9 the quick-start command fetched both files and then refused on the example
  plan's card class.
* ``--init`` on the 480 km row asked the recorder for "1.33333" hours, which is not a whole number of 1200 s steps.
* An asset root made by hand without the row's refuse list ran its window with nothing held by name.
"""
import argparse
import json
import types

import pytest

from woof.hex import ad_doors as d
from woof.hex.ad import assets, rows


def _parse(argv):
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


@pytest.fixture
def no_fetch(monkeypatch, tmp_path):
    """A card that fits (class 12.0, 97 GB), a host that fits, a runner, an empty row cache, and an asset root that must not be resolved."""
    for name in d.NO_LOCAL_GPU_ENV:
        monkeypatch.delenv(name, raising=False)
    fetched = []

    def asset_root(cache=None):
        fetched.append("asset_root")
        raise AssertionError("the asset root was resolved (a first use downloads ~940 MB)")

    monkeypatch.setattr(assets, "asset_root", asset_root)
    monkeypatch.setattr(assets, "_fetch", lambda *a, **k: fetched.append("_fetch"))
    for name in ("WOOF_HEX_AD_ASSETS", rows.IMAGE_BUILDER_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    monkeypatch.setattr(d, "resolve_runner", lambda explicit: tmp_path / "runner")
    monkeypatch.setattr(d, "compute_capability", lambda card: "12.0")
    monkeypatch.setattr(d, "total_mib", lambda card: 97887)
    monkeypatch.setattr(d, "free_mib", lambda card: 97000)
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 500000)   # a host that fits the 120 km class
    monkeypatch.setattr(d, "refuse_plan_libraries_missing", lambda *a, **k: None)   # an install that holds the plan's libraries
    return fetched


@pytest.fixture
def mesh120(monkeypatch, tmp_path):
    """The 120 km mesh as two empty files; its facts are given, so no file is read."""
    for name in ("x1.40962.grid.nc", "x1.40962.ugwp_oro_data.nc", "init.nc"):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(rows, "mesh_facts", lambda grid, init=None: {"cells": 40962, "edges": 122880, "levels": 59,
                                                                     "nominal_dc_m": 120000.0})
    return ["--mesh", "x1.40962", "--grid", str(tmp_path / "x1.40962.grid.nc"),
            "--oro", str(tmp_path / "x1.40962.ugwp_oro_data.nc"), "--suite", "hrrrv5"]


def test_the_shipped_pins_say_what_they_hold():
    holds = assets.pinned_holds()
    assert set(holds) == {"meshes/x1.40962.grid.nc", "meshes/x1.40962.ugwp_oro_data.nc", "plans/hrrrv5-x1.40962-20230331T12-e4/",
                          "latest/", "rows/hrrrv5-any/", "meshes/q3.96.38278.n40.60w91.30.grid.nc", "meshes/q3.96.38278.n40.60w91.30.ugwp_oro_data.nc",
                          "plans/hrrrv5-q3.96.38278.n40.60w91.30-20230331T18-e240/", "cases/q3.96.38278.n40.60w91.30-20230331T18/",
                          "gradient-meshes/q3.96.38278.n40.60w91.30/6km/",
                          # the storm row's optional drag image (its notice and licence beside it) and its next kernel set
                          "latest/aux/ugwp.dad.sm_120.cubin", "latest/aux/ugwp.dad.sm_120.NOTICE.txt",
                          "latest/aux/LICENSE-UGWP-Apache-2.0.txt", "latest/kset/storm-bs5-lean2-astra2-dyn6/",
                          # 0.6.0: the default route's 3 km set with three double-arithmetic dynamics images
                          "latest/kset/storm-bs5-lean2-astra2-dynE5-sp74-ddr/",
                          # the five images of the storm row's tape-free boundary layer (the closure-live form,
                          # dist48), notice and licence beside them
                          "latest/aux/bl-closure/direct120.cubin", "latest/aux/bl-closure/cap120.cubin",
                          "latest/aux/bl-closure/rel.0.cubin", "latest/aux/bl-closure/rel.1.cubin",
                          "latest/aux/bl-closure/rel.2.cubin", "latest/aux/bl-closure/NOTICE.txt",
                          "latest/aux/bl-closure/LICENSE-MYNN-EDMF-Apache-2.0.txt",
                          # (F10-full) the full-physics rows' floored boundary layer, notice and licence beside them
                          "latest/aux/bl-q1floor/rel.0.cubin", "latest/aux/bl-q1floor/rel.1.cubin",
                          "latest/aux/bl-q1floor/rel.2.cubin", "latest/aux/bl-q1floor/NOTICE.txt",
                          "latest/aux/bl-q1floor/LICENSE-MYNN-EDMF-Apache-2.0.txt",
                          # the three images of the storm row's guarded boundary layer, notice and licence beside them
                          "latest/aux/bl-guarded/rel.0.cubin", "latest/aux/bl-guarded/rel.1.cubin",
                          "latest/aux/bl-guarded/rel.2.cubin", "latest/aux/bl-guarded/NOTICE.txt",
                          "latest/aux/bl-guarded/LICENSE-MYNN-EDMF-Apache-2.0.txt",
                          # dist44: the default route's kernel set, the radiation step's two rewritten images and
                          # the cloud scheme's gated derivative images, each with its notice and licence
                          "latest/kset/storm-bs5-lean2-astra2-dynE5-sp74/",
                          "latest/aux/rad-step/rsw_taumol.sm_120.cubin", "latest/aux/rad-step/taumol.sm_120.cubin",
                          "latest/aux/rad-step/NOTICE.txt", "latest/aux/rad-step/LICENSE-AER-RRTMG-BSD-3-Clause.txt",
                          "latest/aux/rad-step/NOTICE-AER-RRTMG-as-distributed-with-MPAS.txt",
                          "latest/aux/cloud-gated/c2chain-0xc6-rd3600n100.cubin",
                          "latest/aux/cloud-gated/tlfd-0xc6-rd3600n100.cubin",
                          "latest/aux/cloud-gated/tlfd64-0xc6-rd3600n100.cubin",
                          "latest/aux/cloud-gated/split2-0xc6-rd3600n100.cubin", "latest/aux/cloud-gated/NOTICE.txt",
                          "latest/aux/cloud-gated/LICENSE-Apache-2.0.txt",
                          # 0.6.0: the default number-following cloud images, notice and licence beside them
                          "latest/aux/cloud-number/c2chain-0x06-rd3600n100-nfr-nfc-ts-dg200.cubin",
                          "latest/aux/cloud-number/tlfd-0x06-rd3600n100-nfr-nfc-ts-dg200.cubin",
                          "latest/aux/cloud-number/tlfd64-0x06-rd3600n100-nfr-nfc-ts-dg200.cubin",
                          "latest/aux/cloud-number/split2-0x06-rd3600n100-nfr-nfc-ts-dg200.cubin",
                          "latest/aux/cloud-number/NOTICE.txt", "latest/aux/cloud-number/LICENSE-Apache-2.0.txt",
                          # dist48: the 12 km example, its mesh file and its plan
                          "meshes/q12.96.2890.n40.60w91.30.grid.nc",
                          "plans/hrrrv5-q12.96.2890.n40.60w91.30-20240521T15-e15/"}
    for name in ("hrrrv5-x1.40962", "hrrrv5-full-x1.40962", "hrrrv5-q3.96.38278.n40.60w91.30",
                 "hrrrv5-q12.96.2890.n40.60w91.30"):   # the shipped rows
        row = d.load_table()["rows"][name]
        held = [rel for rel in row["assets"].values() if any(rel.startswith(h) for h in holds)]
        assert sorted(held) == sorted(row["assets"].values())        # every file of the shipped row
        assert any((row["example_plan"] + "/").startswith(h) for h in holds)
    assert assets.pinned_holds({"assets": {"a.tar.gz": {"bytes": 1, "sha256": "0" * 64}}}) is None   # a pin that does not say


@pytest.mark.parametrize("more", [[], ["--plan", "PLAN"], ["--init", "INIT"]])
def test_a_made_row_the_pins_do_not_hold_is_refused_before_any_download(no_fetch, mesh120, tmp_path, more, monkeypatch):
    monkeypatch.setattr(assets, "pinned_holds", lambda pins=None: ["plans/another-plan/"])   # pins that hold no row file
    more = [str(tmp_path / "init.nc") if x == "INIT" else str(tmp_path / "plan") if x == "PLAN" else x for x in more]
    with pytest.raises(d.AdDoorError, match=r"row hrrrv5-x1\.40962 reads latest/.* and \d+ more from the asset root, "
                                            r"and the files this install downloads do not hold them .*nothing was "
                                            r"downloaded"):
        d.Context(_parse(["param-grad", *mesh120, *more, "--out", str(tmp_path / "o")]), "param-grad")
    assert no_fetch == [] and not (tmp_path / "o").exists()


def test_a_made_row_with_no_plan_is_refused_before_any_download(no_fetch, mesh120, tmp_path, monkeypatch):
    """With pins that do hold the made rows' files, a row with no example plan still fetches nothing."""
    loaded = d.load_table

    def no_shipped_120(path=None):   # the mesh's row is made, not the shipped 120 km row with its example plan
        t = loaded(path)
        t["rows"].pop("hrrrv5-x1.40962", None)
        return t

    monkeypatch.setattr(d, "load_table", no_shipped_120)
    monkeypatch.setattr(assets, "pinned_holds", lambda pins=None: ["latest/", "rows/hrrrv5-any/", "rows/hrrrv5-x1.2562/"])
    with pytest.raises(d.AdDoorError, match="row hrrrv5-x1.40962 ships no example plan: pass --init INIT"):
        d.Context(_parse(["tangent", *mesh120, "--out", str(tmp_path / "o")]), "tangent")
    assert no_fetch == []
    with pytest.raises(AssertionError, match="asset root was resolved"):      # with a plan it goes on to the assets
        d.Context(_parse(["tangent", *mesh120, "--plan", str(tmp_path / "plan"), "--out", str(tmp_path / "o")]),
                  "tangent")
    assert no_fetch == ["asset_root"]


def test_an_asset_root_the_user_names_is_not_judged_by_the_pins(no_fetch, mesh120, tmp_path, monkeypatch):
    loaded = d.load_table

    def no_shipped_120(path=None):   # the mesh's row is made, not the shipped 120 km row with its example plan
        t = loaded(path)
        t["rows"].pop("hrrrv5-x1.40962", None)
        return t

    monkeypatch.setattr(d, "load_table", no_shipped_120)
    (tmp_path / "root").mkdir()
    with pytest.raises(d.AdDoorError, match="ships no example plan"):
        d.Context(_parse(["tangent", *mesh120, "--assets", str(tmp_path / "root"), "--out", str(tmp_path / "o")]),
                  "tangent")
    assert no_fetch == []


def test_another_card_class_is_refused_before_any_download(no_fetch, tmp_path, monkeypatch):
    monkeypatch.setattr(d, "compute_capability", lambda card: "8.9")
    with pytest.raises(d.AdDoorError, match=r"row hrrrv5-x1\.40962 runs on NVIDIA Blackwell cards of compute "
                                            r"capability 12\.0 .*an Ada Lovelace card .*compiled for sm_120, and "
                                            r"the row names no image recipe for sm_89"):
        d.Context(_parse(["adjoint-test", "--out", str(tmp_path / "o")]), "adjoint-test")
    assert no_fetch == [] and not (tmp_path / "o").exists()
    # the 120 km rows name no image recipe, so an image builder cannot help: still refused, still nothing fetched
    monkeypatch.setenv(rows.IMAGE_BUILDER_ENV, "builder {arch} {out}")
    with pytest.raises(d.AdDoorError, match="names no image recipe for sm_89"):
        d.Context(_parse(["adjoint-test", "--out", str(tmp_path / "o")]), "adjoint-test")
    assert no_fetch == []


def test_built_images_of_the_card_class_need_no_builder(tmp_path, monkeypatch):
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    monkeypatch.delenv(rows.IMAGE_BUILDER_ENV, raising=False)
    row = d.load_table()["rows"]["hrrrv5-x1.40962"]
    rows.refuse_arch_without_images("hrrrv5-x1.40962", row, "sm_120")         # the shipped class: nothing to refuse
    with pytest.raises(rows.AdRowError, match="names no image recipe for sm_89"):
        rows.refuse_arch_without_images("hrrrv5-x1.40962", row, "sm_89")
    # a row with an image recipe and that card class's images already built needs no builder
    recipe = dict(row, arch_images={"image_set": "test-set"})
    built = tmp_path / "images" / "test-set" / "sm_89" / "built"
    built.mkdir(parents=True)
    (built / ".complete.json").write_text("{}")
    rows.refuse_arch_without_images("hrrrv5-x1.40962", recipe, "sm_89")
    made = dict(d.load_table()["rows"]["hrrrv5-any"], mesh="x1.40962")
    with pytest.raises(rows.AdRowError, match="names no image recipe for sm_89"):
        rows.refuse_arch_without_images("hrrrv5-x1.40962", made, "sm_89")


@pytest.mark.parametrize("dt,steps", [(1200, 3), (1200, 1), (720, 3), (450, 3), (18, 3), (20, 7), (112.5, 3)])
def test_the_recorder_gets_a_whole_number_of_steps(tmp_path, monkeypatch, dt, steps):
    """The recorder's own rule (woof.hex.ad.record.chain.record): |hours * 3600 / dt - round(...)| <= 1e-9."""
    for name in ("grid.nc", "oro.nc", "init.nc"):
        (tmp_path / name).write_bytes(b"x")
    seen = {}

    def run(argv, stdout=None, stderr=None, env=None, **kw):
        seen["argv"] = argv
        return types.SimpleNamespace(returncode=7)

    monkeypatch.setattr(d.subprocess, "run", run)
    monkeypatch.setenv(d.RECORDER_ENV, "recorder --hours {hours} --steps {steps}")
    row = dict(d.load_table()["rows"]["hrrrv5-any"], mesh="x", dt_s=dt)
    args = types.SimpleNamespace(grid=str(tmp_path / "grid.nc"), oro=str(tmp_path / "oro.nc"), steps=steps,
                                 out=str(tmp_path / "out"), init=str(tmp_path / "init.nc"), card="0", cards=None,
                                 runner=None)
    with pytest.raises(d.AdDoorError, match="recording from"):
        d.record_plan(args, tmp_path, row)
    hours, recorded = float(seen["argv"][2]), int(seen["argv"][4])
    assert recorded == steps + 1
    assert abs(hours * 3600.0 / dt - recorded) <= 1e-9, seen["argv"]


def _window_row(tmp_path, refuse):
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    row = {"asset_args": {"--kernels": "kernels"}, "asset_env": {}, "window_args": ["--tempo", "all"], "window_env": {}}
    (root / "kernels.json").write_text("{}")
    paths = {"kernels": root / "kernels.json"}
    if refuse is not None:
        paths["refuse"] = root / "refuse.txt"
        if refuse is not False:
            (root / "refuse.txt").write_text(refuse)
    return row, paths


def test_a_row_that_names_a_refuse_list_needs_the_file(tmp_path):
    table = d.load_table()
    runner, out = tmp_path / "runner", tmp_path / "o"
    row, paths = _window_row(tmp_path, False)                    # named, and the file is not there
    with pytest.raises(d.AdDoorError, match=r"row asset refuse \(.*refuse\.txt\) is missing from the asset root"):
        d.window_command(table, row, paths, "tangent", runner, 3, out, [])
    row, paths = _window_row(tmp_path, "rlw_inatm,solar\n")      # named and there: handed to the runner
    argv, _ = d.window_command(table, row, paths, "tangent", runner, 3, out, [])
    assert argv[argv.index("--refuse") + 1] == "rlw_inatm,solar"
    row, paths = _window_row(tmp_path, "\n")                     # named and empty (the 3 km class): nothing is held
    argv, _ = d.window_command(table, row, paths, "tangent", runner, 3, out, [])
    assert "--refuse" not in argv
    row, paths = _window_row(tmp_path, None)                     # a row that names no list
    argv, _ = d.window_command(table, row, paths, "tangent", runner, 3, out, [])
    assert "--refuse" not in argv


def test_why_the_missing_list_matters_a_window_with_no_refuse_reads_valid(tmp_path):
    """The reading behind the refusal: with no --refuse and no refused kind reported, the window reads valid, and
    nothing in the receipt says the row's held kinds ran live."""
    (tmp_path / "stdout").write_text(json.dumps({"window_tangent_step": 1, "reference_check": True,
                                                 "boundary_hashes": "equal"}) + "\n")
    said = d.window_validity(tmp_path, {"refused": {}}, ["runner", "window", "--steps", "3"])
    assert said["valid"] is True and said["refused_kinds"] == 0 and said["refused_beyond_row_list"] == []
