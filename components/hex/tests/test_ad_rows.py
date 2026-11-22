"""Operator rows on demand (any mesh, any card class), CPU part.

THE BREAKAGES THESE HOLD SHUT: a door refusing every mesh but the one shipped row ("that mesh cannot run"); a door
refusing a card the row was not built for instead of building its images; a made row that keeps the template mesh's
sizes, timestep or kernel set (the runner would read x1.2562 topology tables on another mesh); a bundle for another
card that still names an sm_120 image (CUDA status 209 at load) or silently keeps one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from woof.hex import ad_doors as d
from woof.hex.ad import rows


def test_arch_of_names_the_card_class():
    assert rows.arch_of("8.9") == "sm_89" and rows.arch_of("12.0") == "sm_120" and rows.arch_of("9") == "sm_90"
    with pytest.raises(rows.AdRowError):
        rows.arch_of("x")


def test_table_suites_name_a_template_row_and_a_memory_rule():
    table = d.load_table()
    for name, suite in table["suites"].items():
        assert suite["template_row"] in table["rows"], name
        assert suite["memory"]["fixed_mib"] > 0 and suite["memory"]["mib_per_column_level"] > 0
    for name, row in table["rows"].items():
        spec = row.get("arch_images")
        if spec:
            for key in spec["bundles"] + list(spec["files"]):
                assert key in row["assets"], (name, key)


def _mesh(tmp_path: Path, name: str) -> Path:
    root = tmp_path / "meshes" / name
    root.mkdir(parents=True)
    (root / f"{name}.grid.nc").write_bytes(b"x")
    (root / f"{name}.ugwp_oro_data.nc").write_bytes(b"x")
    return root


def test_find_mesh_by_name_path_and_directory(tmp_path, monkeypatch):
    root = _mesh(tmp_path, "x1.10242")
    monkeypatch.setenv(rows.MESH_PATH_ENV, str(tmp_path / "meshes"))
    by_name = rows.find_mesh("x1.10242")
    assert by_name["mesh"] == "x1.10242" and by_name["oro"].name == "x1.10242.ugwp_oro_data.nc"
    assert rows.find_mesh(str(root / "x1.10242.grid.nc"))["mesh"] == "x1.10242"
    assert rows.find_mesh(str(root))["grid"] == by_name["grid"]
    with pytest.raises(rows.AdRowError, match="no grid file"):
        rows.find_mesh("x1.999")


def test_made_row_carries_the_mesh_and_is_cached(tmp_path, monkeypatch):
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    monkeypatch.setenv(rows.MESH_PATH_ENV, str(tmp_path / "meshes"))
    _mesh(tmp_path, "x1.10242")
    monkeypatch.setattr(rows, "mesh_facts", lambda grid, init=None: {
        "cells": 10242, "edges": 30720, "levels": 59, "nominal_dc_m": 240000.0})
    table = d.load_table()
    name, row = rows.make_row(table, "hrrrv5", rows.find_mesh("x1.10242"))
    template = table["rows"][table["suites"]["hrrrv5"]["template_row"]]
    assert name == "hrrrv5-x1.10242" and row["mesh"] == "x1.10242"
    assert (row["cells"], row["edges"], row["levels"]) == (10242, 30720, 59)
    assert 1800 % row["dt_s"] == 0 and row["dt_s"] <= 6 * 240   # hex's rule: 6 s/km, dividing the radiation step
    assert row["kset_mesh"] == "any" and "--local-topology" in row["window_args"]   # one kernel set, any mesh
    assert rows.ensure_kset(table, name, row, tmp_path, tmp_path) == {}
    with pytest.raises(d.AdDoorError, match="template"):
        d._row(table, "hrrrv5-any")
    assert "example_plan" not in row and Path(row["assets"]["grid"]).is_absolute()
    assert row["need_free_mib"] > template["need_free_mib"]
    cls = rows.window_class(table["suites"]["hrrrv5"], 10242)
    assert row["window_args"] == template["window_args"] + cls["window_args"] and row["doors"] == template["doors"]
    assert row["generated"]["window_class"]["from_recipe"] == cls["from_recipe"]
    assert rows.cached_rows()[name]["cells"] == 10242            # answers --row next time
    table["rows"]["hrrrv5-any"] = dict(template, record_env={"X": "1"})   # the template changed: the row is remade
    assert rows.make_row(table, "hrrrv5", rows.find_mesh("x1.10242"))[1]["record_env"] == {"X": "1"}
    assert rows.make_row(table, "hrrrv5", rows.find_mesh("x1.10242"), dt=600.0)[1]["dt_s"] == 600.0


def test_window_class_follows_the_mesh_size_and_refuses_above_one_card(tmp_path, monkeypatch):
    """THE BREAKAGE: a 120 km mesh run with the 480 km arena sizes (the reverse scratch overflows), or a 60 km mesh
    sent to one card (its step tape does not fit) instead of a refusal that names the multi-card recipe."""
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    table = d.load_table()
    for suite_name in ("hrrrv5", "hrrrv5-exact"):
        suite = table["suites"][suite_name]
        small, big = rows.window_class(suite, 2562), rows.window_class(suite, 40962)
        assert small is rows.window_class(suite, 10242) and small is not big
        assert big["window_args"] != small["window_args"]
        sixty = [c for c in suite["window_classes"] if int(c["max_cells"]) == 200000]
        if sixty:   # the published set carries the 60 km recipe (LATEST 20261110T2334Z-9c554a09): its class
            assert rows.window_class(suite, 163842) is sixty[0]
            with pytest.raises(rows.AdRowError, match="one rank per card"):
                rows.window_class(suite, 655362)
        else:
            with pytest.raises(rows.AdRowError, match="multi-card"):
                rows.window_class(suite, 163842)
        template = table["rows"][suite["template_row"]]
        for cls in suite["window_classes"]:   # every runner file a class or the template names has an asset
            keys = set(template["assets"]) | set(cls["assets"])
            assert set(template["asset_env"].values()) | set(template["asset_args"].values()) <= keys
            assert set((cls.get("asset_env") or {}).values()) <= keys
    assert "tempo_block_cubin" not in table["rows"]["hrrrv5-any-exact"]["assets"]   # exact: the taped TEMPO adjoint
    assert rows.window_class({}, 5) is None
    _mesh(tmp_path, "x1.40962")
    monkeypatch.setenv(rows.MESH_PATH_ENV, str(tmp_path / "meshes"))
    monkeypatch.setattr(rows, "mesh_facts", lambda grid, init=None: {
        "cells": 40962, "edges": 122880, "levels": 55, "nominal_dc_m": 120000.0})
    name, row = rows.make_row(table, "hrrrv5-exact", rows.find_mesh("x1.40962"))
    assert name == "hrrrv5-exact-x1.40962" and row["suite"] == "hrrrv5"
    assert row["window_args"][-8:] == rows.window_class(table["suites"]["hrrrv5-exact"], 40962)["window_args"]
    assert row["window_env"]["AD_TEMPO_ARENA_GIB"] == "8" and row["assets"]["bundle"].endswith("gen-A")
    with pytest.raises(d.AdDoorError, match="parameter-carrier"):
        d._row({**table, "rows": {**table["rows"], name: row}}, name, "param-grad")


def test_a_made_row_refuses_a_window_until_its_kernel_set_is_repinned(tmp_path, monkeypatch):
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    monkeypatch.delenv(rows.KSET_REPIN_ENV, raising=False)
    row = {"mesh": "x1.10242", "kset_mesh": "x1.2562", "suite": "hrrrv5", "assets": {"kernels": "k/kernels.json"}}
    with pytest.raises(rows.AdRowError, match="re-pin"):
        rows.ensure_kset({"suites": {}}, "hrrrv5-x1.10242", row, tmp_path / "plan", tmp_path)
    assert rows.ensure_kset({}, "r", dict(row, kset_mesh="x1.10242"), tmp_path, tmp_path) == {}


def _bundle(root: Path) -> Path:
    b = root / "bundle"
    (b / "p3").mkdir(parents=True)
    (b / "p4").mkdir()
    for rel in ("p8-sm_120.cubin", "p3/rel.0.cubin", "p4/ugwp.tl.sm_120.cubin", "p7.sm_120.cubin", "rl.ad.fatbin"):
        (b / rel).write_bytes(b"sm_120 " + rel.encode())
    (b / "schemes.json").write_text(json.dumps({"images": {
        "p8": "p8-sm_120.cubin", "p3_tl": "p3/rel.0.cubin", "p4_ugwp_tl": "p4/ugwp.tl.sm_120.cubin",
        "p7_taumol": "p7.sm_120.cubin", "ruc_land": "rl.ad.fatbin"}, "rows": []}))
    return b


def test_assemble_bundle_replaces_every_image_or_refuses(tmp_path):
    src = _bundle(tmp_path)
    built = tmp_path / "built"
    (built / "p3").mkdir(parents=True)
    (built / "p4").mkdir()
    (built / "p8-sm_89.cubin").write_bytes(b"89 p8")
    (built / "p3" / "rel.0.cubin").write_bytes(b"89 p3")
    (built / "p4" / "ugwp.tl.sm_89.cubin").write_bytes(b"89 p4")
    out = tmp_path / "out"
    report = rows.assemble_bundle(src, built, "sm_89", out, drop=("p7_taumol",))
    images = json.loads((out / "schemes.json").read_text())["images"]
    assert images == {"p8": "p8-sm_89.cubin", "p3_tl": "p3/rel.0.cubin", "p4_ugwp_tl": "p4/ugwp.tl.sm_89.cubin",
                      "ruc_land": "rl.ad.fatbin"}
    assert (out / "p3" / "rel.0.cubin").read_bytes() == b"89 p3" and not (out / "p8-sm_120.cubin").exists()
    assert report["dropped"] == ["p7_taumol"] and report["kept_fatbins"] == {"ruc_land": "rl.ad.fatbin"}
    assert "sm_120" not in (out / "SHA256SUMS").read_text().replace("p7.sm_120", "")
    (built / "p4" / "ugwp.tl.sm_89.cubin").unlink()
    with pytest.raises(rows.AdRowError, match="p4_ugwp_tl"):
        rows.assemble_bundle(src, built, "sm_89", out, drop=("p7_taumol",))


def test_card_images_built_once_by_the_builder_then_cached(tmp_path, monkeypatch):
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "cache" / "rows"))
    src = _bundle(tmp_path)
    builder = tmp_path / "build.sh"
    builder.write_text("#!/bin/sh\nset -e\nmkdir -p $2/p3 $2/p4 $2/tempo\n"
                       "for f in p8-$1.cubin p3/rel.0.cubin p4/ugwp.tl.$1.cubin tempo/a.$1.cubin; do echo $1 > $2/$f; done\n"
                       "echo run >> $2/../runs\n")
    builder.chmod(0o755)
    monkeypatch.setenv(rows.IMAGE_BUILDER_ENV, f"{builder} {{arch}} {{out}}")
    row = {"image_archs": ["sm_120"], "assets": {"bundle": str(src), "tempo_cubin": "x"},
           "arch_images": {"image_set": "t", "bundles": ["bundle"], "files": {"tempo_cubin": "tempo/a.{arch}.cubin"},
                           "drop": ["p7_taumol"]}}
    assert rows.ensure_arch_images({}, "r", row, "sm_120", tmp_path) == {}
    got = rows.ensure_arch_images({}, "r", row, "sm_89", tmp_path, needed={"bundle", "tempo_cubin"})
    assert got["tempo_cubin"].read_text().strip() == "sm_89"
    assert json.loads((got["bundle"] / "schemes.json").read_text())["images"]["p8"] == "p8-sm_89.cubin"
    again = rows.ensure_arch_images({}, "r", row, "sm_89", tmp_path, needed={"bundle", "tempo_cubin"})
    assert again == got
    assert (tmp_path / "cache" / "images" / "t" / "sm_89" / "runs").read_text().count("run") == 1
    with_own = dict(row, assets=dict(row["assets"], bundle_params=str(src)),
                    arch_images=dict(row["arch_images"], bundles=["bundle", "bundle_params"], own_images=["bundle_params"]))
    with pytest.raises(rows.AdRowError, match="own images"):
        rows.ensure_arch_images({}, "r", with_own, "sm_89", tmp_path, needed={"bundle_params"})
    no_recipe = dict(row, arch_images=None)
    with pytest.raises(rows.AdRowError, match="no image recipe"):
        rows.ensure_arch_images({}, "r", no_recipe, "sm_89", tmp_path)


def test_door_cli_takes_a_mesh(tmp_path):
    from woof.hex import cli
    args = cli.build_parser("adjoint-test").parse_args(
        ["adjoint-test", "--out", "x", "--mesh", "x1.10242", "--init", "i.nc", "--dt", "900", "--need-mib", "30000"])
    assert args.mesh == "x1.10242" and args.suite == "hrrrv5" and args.dt == 900.0 and args.need_mib == 30000


def test_cubin_arch_reads_both_elf_abis():
    import struct

    def elf(osabi, flags):
        b = bytearray(64)
        b[0:4] = bytes([0x7F]) + b"ELF"
        b[7] = osabi
        struct.pack_into("<H", b, 18, 190)
        struct.pack_into("<I", b, 48, flags)
        return bytes(b)
    assert rows.cubin_arch(elf(0x41, 0x6007802)) == "sm_120"   # measured: a plan module recorded on an RTX 5090
    assert rows.cubin_arch(elf(0x41, 0x6005904)) == "sm_89"    # measured: an image built for the RTX 4090
    assert rows.cubin_arch(elf(0x33, 0x590059)) == "sm_89"
    assert rows.cubin_arch(b"// ptx .version 8.0") is None


def test_a_multi_card_recipe_in_the_published_set_becomes_a_class_with_its_cards(tmp_path, monkeypatch):
    """THE BREAKAGE: the 60 km recipe lands in the published set and the door still refuses 60 km, or sends a 60 km
    window to one card (its step tape does not fit), or runs it without the shard and streamed-tape settings."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "tools" / "ad_doors_from_latest.py"
    spec = importlib.util.spec_from_file_location("ad_doors_from_latest", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    def recipe(bundle, env=None, args=(), cards=1):
        return {"env": {"AD_SCHEMES": "{L}/bundle/" + bundle, "AD_TAPE_PACK": "1", **(env or {})},
                "args": [["kernels", "{L}/kset/k/kernels.json"], ["cubins", "{L}/kset/k/cubins"],
                         ["schemes", "{L}/aux/s.json"], ["tempo-cubin", "{L}/aux/t.cubin"], ["fct-frozen", "1"],
                         ["refuse", "a,b"], *args],
                "needs": {"cards": cards}}

    manifest = {"id": "T", "recipes": {
        "480": recipe("gen-A"), "480-best": recipe("gen-B9", args=[["tape-gib", "2"]]),
        "120": recipe("gen-B9", {"AD_TEMPO_ARENA_GIB": "8"}, [["tape-gib", "6"]]),
        "60": None}}
    table = gen.build(manifest, d.load_table())["table"]
    with pytest.raises(rows.AdRowError, match="multi-card"):          # no 60 km recipe yet: refused with the reason
        rows.window_class(table["suites"]["hrrrv5"], 163842)
    manifest["recipes"]["60"] = recipe("gen-B9", {"AD_TAPE_STREAM": "1", "AD_MYNN_SHARD_DEVICES": "0,1,2,3"},
                                       [["tape-gib", "7"], ["host-budget-gb", "600"]], cards=4)
    table = gen.build(manifest, d.load_table())["table"]
    suite = table["suites"]["hrrrv5"]
    big = rows.window_class(suite, 163842)
    assert big["cards"] == 4 and big["window_env"] == {"AD_MYNN_SHARD_DEVICES": "0,1,2,3", "AD_TAPE_STREAM": "1"}
    assert big["window_args"] == ["--tape-gib", "7", "--host-budget-gb", "600"]
    assert rows.window_class(suite, 40962)["cards"] == 1
    with pytest.raises(rows.AdRowError, match="one rank per card"):
        rows.window_class(suite, 655362)
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    monkeypatch.setenv(rows.MESH_PATH_ENV, str(tmp_path / "meshes"))
    _mesh(tmp_path, "x1.163842")
    monkeypatch.setattr(rows, "mesh_facts", lambda grid, init=None: {
        "cells": 163842, "edges": 491520, "levels": 59, "nominal_dc_m": 60000.0})
    name, row = rows.make_row(table, "hrrrv5", rows.find_mesh("x1.163842"))
    assert row["cards"] == 4 and row["window_env"]["AD_TAPE_STREAM"] == "1"
    args = d.argparse.Namespace(cards=None, card=2)
    assert d.cards_of(args) == ["2"]                                   # one card given: the door refuses by name
    assert d.cards_of(d.argparse.Namespace(cards="4,5,6,7", card=None)) == ["4", "5", "6", "7"]


def test_admission_asks_at_least_the_measured_free_card_peak_of_each_class():
    """THE BREAKAGE: the made 240 km row asked 37,524 MiB and the window held 55,008 to 58,954 MiB on a free card
    (2026-11-09); a window admitted on less runs in a memory layout nobody measured (one such 480 km leg on a nearly
    full card returned a wrong adjoint with rc 0)."""
    suite = d.load_table()["suites"]["hrrrv5"]
    # 480 km class: the peak on the smallest card measured (x1.2562 on a 32 GB RTX 5090, the sensitivity door receipt);
    # 120 km class: the door peaks of each suite on a free card (box-d d120-e: fast 71,156, exact 84,864 MiB; the rule
    # asked 78,838 for both and the exact window held 6,026 MiB more than it was admitted on)
    exact = d.load_table()["suites"]["hrrrv5-exact"]
    measured = [(suite, 2562, 59, 24360), (suite, 40962, 59, 71156), (exact, 40962, 59, 84864)]
    for which, cells, levels, peak in measured:
        rule = rows.window_class(which, cells)["memory"]
        asked = rows.need_mib(rule, cells, levels)
        assert peak <= asked <= peak * 1.10, (cells, asked, peak)
    assert rows.need_mib(rows.window_class(suite, 40962)["memory"], 40962, 59) == 71156 + 2000
    assert rows.need_mib(rows.window_class(exact, 40962)["memory"], 40962, 59) == 84864 + 2000
    # the larger 480 km class meshes, measured only on big free cards: at least that peak less the growth into free
    # memory measured at x1.2562 (36,348 on a larger test card less 24,360 on the 5090 = 11,988 MiB)
    for cells, levels, peak in [(4905, 59, 40940), (10242, 59, 58954)]:
        asked = rows.need_mib(rows.window_class(suite, cells)["memory"], cells, levels)
        assert peak - 11988 <= asked <= peak, (cells, asked, peak)


def test_the_60_km_class_asks_its_measured_one_card_peak_not_the_provisional_fit(monkeypatch):
    """THE BREAKAGE: the 60 km window class carried no memory rule, so a plain 60 km door (x1.163842 x 59 levels)
    fell back to the suite's provisional fit and asked 300,334 MiB, more than the whole 96 GB card: every plain
    60 km door was refused though the 60 km recipe's window holds 82,958 MiB on one card (doors6003, 2026-11-11)."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "tools" / "ad_doors_from_latest.py"
    spec = importlib.util.spec_from_file_location("ad_doors_from_latest", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    table = d.load_table()
    for name, key in (("hrrrv5", "memory"), ("hrrrv5-exact", "exact_memory")):
        cls = rows.window_class(table["suites"][name], 163842)
        assert cls["memory"] == gen.MULTI[key], name          # the table is what the generator writes
        asked = rows.need_mib(cls["memory"], 163842, 59)
        peak = {"hrrrv5": 82958, "hrrrv5-exact": 88642}[name]   # recipe 60 dd60-one-self; door d60-p0001 exact
        assert peak <= asked <= peak * 1.10, (name, asked)
    monkeypatch.setattr(d, "total_mib", lambda card: 97887)   # one RTX PRO 6000, free
    monkeypatch.setattr(d, "free_mib", lambda card: 97250)
    assert d.admit_card("0", asked, 0.0)["free_mib"] == 97250


def test_the_60_km_fast_door_asks_at_least_the_card_memory_its_own_run_held():
    """THE BREAKAGE: the 60 km fast rule (84,958 MiB, the recipe's window peak + 2,000) is below what the fast
    adjoint-test door itself held on one card (88,660 MiB, 2026-11-11): the door is admitted on a
    card with 85 GB free and then runs out of card memory when another job holds the rest."""
    table = d.load_table()
    cls = rows.window_class(table["suites"]["hrrrv5"], 163842)
    asked = rows.need_mib(cls["memory"], 163842, 59)
    assert 88660 + 2000 <= asked <= 97887, asked   # the door's measured peak plus the stated margin, on one 96 GB card


def test_the_480_km_door_fits_a_32_gb_card_and_a_need_above_the_whole_card_says_so(monkeypatch):
    """THE BREAKAGE: every 480 km door on a 32,607 MiB RTX 5090 was refused 'card 0 has 32149 MiB free; this row's
    window needs 35860 MiB. Free the card or pass --wait-card SECONDS' (2026-11-11), though its windows held at
    most 24,360 MiB there; and a need above the whole card told the user to wait for memory that can never be free."""
    suite = d.load_table()["suites"]["hrrrv5"]
    asked = rows.need_mib(rows.window_class(suite, 2562)["memory"], 2562, 59)
    assert asked == 26360 and asked < 32149
    monkeypatch.setattr(d, "total_mib", lambda card: 32607)
    monkeypatch.setattr(d, "free_mib", lambda card: 32149)
    assert d.admit_card("0", asked, 0.0)["free_mib"] == 32149
    with pytest.raises(d.AdDoorError) as refused:
        d.admit_card("0", 49922, 0.0)
    assert "more than the whole card" in str(refused.value) and "--wait-card" not in str(refused.value)
    monkeypatch.setattr(d, "free_mib", lambda card: 20000)   # fits the card, not free now: the wait hint stays
    with pytest.raises(d.AdDoorError, match="--wait-card"):
        d.admit_card("0", asked, 0.0)


def test_a_limited_area_mesh_takes_the_whole_limited_area_class_never_the_size_class(tmp_path, monkeypatch):
    """THE BREAKAGE: the 38,278-cell 3 km storm box sent through the 120 km globe's class by its size (kernel set
    without the limited-area gather tables, radiation refused by name), or the storm box's recipe given to a globe."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "tools" / "ad_doors_from_latest.py"
    spec = importlib.util.spec_from_file_location("ad_doors_from_latest", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    def recipe(bundle, kset="k", refuse="a,b", env=None, args=()):
        return {"env": {"AD_SCHEMES": "{L}/bundle/" + bundle, "AD_TAPE_PACK": "1", **(env or {})},
                "args": [["kernels", "{L}/kset/" + kset + "/kernels.json"], ["cubins", "{L}/kset/" + kset + "/cubins"],
                         ["schemes", "{L}/aux/s.json"], ["tempo-cubin", "{L}/aux/t.cubin"], ["fct-frozen", "1"],
                         *([["refuse", refuse]] if refuse else []), *args]}

    manifest = {"id": "T", "recipes": {
        "480": recipe("gen-A"), "480-best": recipe("gen-B9", args=[["tape-gib", "2"]]),
        "120": recipe("gen-B9", env={"AD_TEMPO_ARENA_GIB": "8"}, args=[["tape-gib", "6"]])}}
    table = gen.build(manifest, d.load_table())["table"]
    with pytest.raises(rows.AdRowError, match="limited-area"):       # no 3 km recipe yet: refused with the reason
        rows.window_class(table["suites"]["hrrrv5"], 38278, "limited-area")
    manifest["recipes"]["3km"] = recipe("gen-B9", kset="storm-f", refuse="",
                                        env={"AD_DOT_SELF": "1", "AD_DOT_ALLOC": "1",
                                             "AD_BINDER_REFERENCE_PAYLOADS": "recording/payloads"},
                                        args=[["local-topology", "1"], ["mesh-constants", "1"]])
    manifest["recipes"]["3km"]["what"] = "storm box recipe: radiation and land live, NO refuse list"
    built = gen.build(manifest, d.load_table())
    table = built["table"]
    suite = table["suites"]["hrrrv5"]
    lam = rows.window_class(suite, 38278, "limited-area")
    glob = rows.window_class(suite, 38278)
    assert lam["whole"] and lam["from_recipe"] == "3km" and glob["from_recipe"] == "120"
    assert lam["assets"]["kernels"] == "latest/kset/storm-f/kernels.json" and lam["refuse_list"] == ""
    assert not {"AD_DOT_SELF", "AD_DOT_ALLOC", "AD_BINDER_REFERENCE_PAYLOADS"} & set(lam["window_env"])
    assert built["class_refuse"] == {"rows/hrrrv5-any/refuse-3km.txt": ""}
    with pytest.raises(rows.AdRowError, match="limited-area"):       # exact suite: no 3km-exact, refused, no fast images
        rows.window_class(table["suites"]["hrrrv5-exact"], 38278, "limited-area")
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    monkeypatch.setenv(rows.MESH_PATH_ENV, str(tmp_path / "meshes"))
    _mesh(tmp_path, "r300")
    monkeypatch.setattr(rows, "mesh_facts", lambda grid, init=None: {
        "cells": 38278, "edges": 115121, "levels": 59, "nominal_dc_m": 3000.0, "limited_area": True})
    name, row = rows.make_row(table, "hrrrv5", rows.find_mesh("r300"), dt=15.0)
    assert row["mesh_kind"] == "limited-area" and row["assets"]["kernels"].endswith("storm-f/kernels.json")
    assert "--refuse" not in row["window_args"] and row["window_args"].count("--local-topology") == 1
    assert "AD_TEMPO_ARENA_GIB" not in row["window_env"] and row["need_free_mib"] == 93000
    # its receipt said "radiation is held (its launches are refused by name)" over a run that refused nothing
    assert "radiation is held" not in row["what"] and row["what"].endswith("radiation and land live, NO refuse list")
    assert set(row["asset_env"].values()) | set(row["asset_args"].values()) <= set(row["assets"])
    root = tmp_path / "assets"
    gen.asset_root(root, tmp_path, "a,b", None, built["class_refuse"])
    assert (root / "rows/hrrrv5-any/refuse-3km.txt").read_text() == "\n"


def test_rows_take_the_newest_mesh_generic_kernel_set_the_published_recipes_name():
    """THE BREAKAGE: the door rows keep naming kset-d7 after the published set moved its recipes to kset-d9 and then
    dropped d7 from the tree: every door window dies on a missing kernel set."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "tools" / "ad_doors_from_latest.py"
    spec = importlib.util.spec_from_file_location("ad_doors_from_latest", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    def recipe(kset):
        return {"env": {"AD_SCHEMES": "{L}/bundle/gen-B9"},
                "args": [["kernels", "{L}/kset/" + kset + "/kernels.json"], ["cubins", "{L}/kset/" + kset + "/cubins"],
                         ["schemes", "{L}/aux/s.json"], ["tempo-cubin", "{L}/aux/t.cubin"], ["refuse", "a"]]}

    for named, want in (("kset-d9", "kset-d9"), ("kset-d7", "kset-d7"), ("kset-twh", "kset-d7"),
                        ("kset-d10", "kset-d10"), ("kset-d12", "kset-d12")):
        manifest = {"id": "T", "recipes": {"480": recipe(named), "480-best": recipe(named), "120": recipe(named)}}
        assert gen.generic_kset(manifest) == want
        row = gen.build(manifest, d.load_table())["table"]["rows"]["hrrrv5-any"]
        assert row["assets"]["kernels"] == f"latest/kset/{want}/kernels.json"


def test_a_users_own_x1_40962_case_takes_the_published_set_not_the_pinned_row(tmp_path, monkeypatch):
    """--mesh x1.40962 with the user's own --init or --plan: the row is made from the template that follows the
    published set; with no case of their own (the pinned row's example plan), or for a door the template does not run,
    the pinned row answers."""
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    monkeypatch.setenv(rows.MESH_PATH_ENV, str(tmp_path / "meshes"))
    _mesh(tmp_path, "x1.40962")
    monkeypatch.setattr(rows, "mesh_facts", lambda grid, init=None: {
        "cells": 40962, "edges": 122880, "levels": 59, "nominal_dc_m": 120000.0})
    table = d.load_table()
    template = table["rows"][table["suites"]["hrrrv5"]["template_row"]]
    table["rows"]["hrrrv5-any"] = dict(template, latest_id="L1")
    pinned = table["rows"]["hrrrv5-x1.40962"]
    assert not pinned.get("latest_id") and pinned.get("example_plan")
    mesh = rows.find_mesh("x1.40962")
    assert rows.make_row(table, "hrrrv5", mesh)[1] is pinned                                   # no case of their own
    # param-grad: the template has a parameter-carrier bundle since 2026-11-11 and runs the door, so a case the user
    # brings takes the made row for this door too; a template that does not run the door leaves it to the pinned row
    assert rows.make_row(table, "hrrrv5", mesh, own_case=True, door="param-grad")[1] is not pinned
    table["rows"]["hrrrv5-any"] = dict(template, latest_id="L1", doors=["tangent", "adjoint-test", "sensitivity"])
    assert rows.make_row(table, "hrrrv5", mesh, own_case=True, door="param-grad")[1] is pinned
    table["rows"]["hrrrv5-any"] = dict(template, latest_id="L1")
    name, made = rows.make_row(table, "hrrrv5", mesh, own_case=True, door="adjoint-test")
    assert name == "hrrrv5-x1.40962" and made is not pinned and made["generated"] and made["latest_id"] == "L1"
    assert "example_plan" not in made and made["cells"] == 40962
    table["rows"]["hrrrv5-any"] = dict(template, latest_id=None)                               # nothing newer
    assert rows.make_row(table, "hrrrv5", mesh, own_case=True, door="adjoint-test")[1] is pinned


DERIVATIVE_IMAGES = {"bundle", "tempo_tlf_cubin", "tempo_block_cubin", "tempo_serial_cubin", "tempo_cubin", "cubins",
                     "kernels", "schemes"}


def test_a_size_class_that_swaps_the_templates_images_says_so_in_the_row_label(tmp_path, monkeypatch):
    """THE BREAKAGE: the 60 km fast row's label (receipt row_what) said bundle B9 with the plume frozen and a float32
    TEMPO tangent, while its window class ran the exact row's bundle gen-A and tangent image ti2.tlf.cubin (round-8
    receipts d60 and d60b, 2026-11-11). A size class that swaps an image the template's words name must carry its own
    words, and the made row's label must be those words."""
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    monkeypatch.setenv(rows.MESH_PATH_ENV, str(tmp_path / "meshes"))
    table = d.load_table()
    for suite_name, suite in table["suites"].items():
        template = table["rows"][suite["template_row"]]
        for cls in suite["window_classes"]:
            if cls.get("whole"):
                continue
            # the derivative images the template's words describe (tape_unread is a memory list, not an image)
            swapped = {k: v for k, v in (cls.get("assets") or {}).items()
                       if k in DERIVATIVE_IMAGES and template["assets"].get(k) != v}
            if swapped:
                assert cls.get("what"), (suite_name, cls["from_recipe"], sorted(swapped))
                for v in swapped.values():
                    assert Path(v).name in cls["what"], (suite_name, cls["from_recipe"], v)
    _mesh(tmp_path, "x1.163842")
    monkeypatch.setattr(rows, "mesh_facts", lambda grid, init=None: {
        "cells": 163842, "edges": 491520, "levels": 59, "nominal_dc_m": 60000.0})
    name, row = rows.make_row(table, "hrrrv5", rows.find_mesh("x1.163842"))
    assert row["assets"]["bundle"].endswith("genA-gfrz") and row["assets"]["tempo_tlf_cubin"].endswith("ti2.tlf.cubin")
    assert "genA-gfrz" in row["what"] and "mountain-drag closure held" in row["what"] and "ti2.tlf.cubin" in row["what"] and "plume is differentiated" in row["what"]
    assert "frozen at the trajectory (bundle B9)" not in row["what"] and "runs in float32" not in row["what"]
    assert row["what"].startswith("hrrrv5 suite on x1.163842 (163,842 cells, 59 levels), dt ")
    assert "with the window class of recipe 60:" in row["what"]
    _mesh(tmp_path, "x1.40962")   # the 120 km class swaps no image: the template's words stay
    monkeypatch.setattr(rows, "mesh_facts", lambda grid, init=None: {
        "cells": 40962, "edges": 122880, "levels": 59, "nominal_dc_m": 120000.0})
    _, row120 = rows.make_row(table, "hrrrv5", rows.find_mesh("x1.40962"))
    assert "frozen at the trajectory (bundle B9)" in row120["what"] and "made on demand from row hrrrv5-any:" in row120["what"]


def test_a_limited_area_mesh_above_the_storm_box_takes_the_class_sized_to_the_mesh_and_the_card(tmp_path, monkeypatch):
    """THE BREAKAGE: the 135,318-cell Warn-on-Forecast-sized 3 km box refused at the door ("above this suite's largest
    one-card window class (50,000 cells)"), with --cards 0,1 as well (wofs-1, 2026-11-13); and a limited-area mesh on
    a card under 93,000 MiB refused as larger than the whole card, though the runner's out-of-core budget runs it.
    The storm box on a 96 GB card keeps its own class, words and kernel set (its answer bytes)."""
    table = d.load_table()
    # the fast suites 3 km class runs the 3 km storm rows default-route set from 0.6.2; the exact suite keeps storm-h
    for suite_name, recipe, kset in (("hrrrv5", "3km", "storm-bs5-lean2-astra2-dynE5-sp74-ddr"),
                                     ("hrrrv5-exact", "3km-exact", "storm-h")):
        suite = table["suites"][suite_name]
        storm = rows.window_class(suite, 38278, "limited-area")
        assert storm["from_recipe"] == recipe and storm["assets"]["kernels"] == f"latest/kset/{kset}/kernels.json"
        assert storm is rows.window_class(suite, 38278, "limited-area", card_mib=97887, levels=55)
        assert "card_words" not in storm and "--vram-budget-gb" not in storm["window_args"]
        big = rows.window_class(suite, 135318, "limited-area", card_mib=97887, levels=55)
        assert big["from_recipe"] == recipe + "-bounded" and big is rows.window_class(suite, 135318, "limited-area")
        assert big is rows.window_class(suite, 38278, "limited-area", card_mib=32607, levels=55)   # a home card
        assert big["assets"]["kernels"] == "latest/kset/storm-h-ws/kernels.json"
        assert big["assets"]["bundle"] == storm["assets"]["bundle"]   # the recipe's own images, fast or exact
        assert big["window_env"]["AD_TAPE_STREAM"] == "1" and big["cards_max"] == 2
        assert {k for k in storm["window_env"]} <= set(big["window_env"])
        for flag in ("--fct-frozen", "--local-topology", "--mesh-constants", "--native-schemes", "--tempo"):
            assert big["window_args"][big["window_args"].index(flag) + 1] == \
                storm["window_args"][storm["window_args"].index(flag) + 1]
        with pytest.raises(rows.AdRowError, match="one rank per card"):
            rows.window_class(suite, 250000, "limited-area")
        assert rows.window_class(suite, 163842)["from_recipe"].endswith("60")   # a globe never takes it
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    monkeypatch.setenv(rows.MESH_PATH_ENV, str(tmp_path / "meshes"))
    _mesh(tmp_path, "w900")
    monkeypatch.setattr(rows, "mesh_facts", lambda grid, init=None: {
        "cells": 135318, "edges": 406450, "levels": 55, "nominal_dc_m": 3000.0, "limited_area": True})
    name, row = rows.make_row(table, "hrrrv5", rows.find_mesh("w900"), dt=15.0, card_mib=97887)
    args = row["window_args"]
    assert row["mesh_kind"] == "limited-area" and row["generated"]["window_class"]["from_recipe"] == "3km-bounded"
    assert args[args.index("--scratch-gib") + 1] == "12"   # 1,576 B x 135,318 x 55 = 10.9 GiB, + 1
    assert "{vram_gb}" in args and "{host_gb}" in args and row["need_free_mib"] == 16384
    assert row["cards_max"] == 2 and row["shard_env"]["AD_MYNN_SHARD_DEVICES"] == "{ordinals}"
    filled, words = rows.fill_card_words(args, row["card_words"], 95000, 500000)
    assert filled[filled.index("--vram-budget-gb") + 1] == "89" and filled[filled.index("--host-budget-gb") + 1] == "390"
    assert words == {"{vram_gb}": "89", "{host_gb}": "390"} and not any("{" in a for a in filled)
    # a card or host that gives less than a word's smallest budget is refused, never rounded up to it
    with pytest.raises(rows.AdRowError) as short:
        rows.fill_card_words(args, row["card_words"], 4000, 500000, card="0", label="3km-bounded")
    assert str(short.value).startswith("card 0 has 4,000 MiB free; class 3km-bounded needs at least 11,264 MiB free")
    assert "nothing was run" in str(short.value)
    with pytest.raises(rows.AdRowError, match="this host has 8,000 MiB available; class 3km-bounded needs at least "
                                              "20,480 MiB available"):
        rows.fill_card_words(args, row["card_words"], 95000, 8000, card="0", label="3km-bounded")
    assert rows.fill_card_words(args, row["card_words"], 11264, 20480)[1] == {"{vram_gb}": "8", "{host_gb}": "16"}
    with pytest.raises(rows.AdRowError, match="does not report"):
        rows.fill_card_words(args, row["card_words"], 95000, None)
    # the storm box made on a 96 GB card, then on a 32 GB card: the cached row is not reused across classes
    monkeypatch.setattr(rows, "mesh_facts", lambda grid, init=None: {
        "cells": 38278, "edges": 115121, "levels": 55, "nominal_dc_m": 3000.0, "limited_area": True})
    _mesh(tmp_path, "r300")
    _, on96 = rows.make_row(table, "hrrrv5", rows.find_mesh("r300"), dt=15.0, card_mib=97887)
    assert on96["generated"]["window_class"]["from_recipe"] == "3km" and on96["need_free_mib"] == 93000
    assert "card_words" not in on96 and on96["assets"]["kernels"].endswith("storm-bs5-lean2-astra2-dynE5-sp74-ddr/kernels.json")
    _, on32 = rows.make_row(table, "hrrrv5", rows.find_mesh("r300"), dt=15.0, card_mib=32607)
    assert on32["generated"]["window_class"]["from_recipe"] == "3km-bounded"
    assert on32["window_args"][on32["window_args"].index("--scratch-gib") + 1] == "5"   # 3.1 GiB, + 1
    _, again = rows.make_row(table, "hrrrv5", rows.find_mesh("r300"), dt=15.0, card_mib=97887)
    assert again["window_args"] == on96["window_args"] and again["assets"] == on96["assets"]


def test_a_cached_row_planted_with_too_small_a_memory_need_asks_its_class_floor(tmp_path, monkeypatch):
    """THE BREAKAGE: w900's cached row.json edited to need_free_mib 1000 was admitted on a card with 4,000 MiB free and
    went on to the plan with --vram-budget-gb 8 (a budget the card does not have); --need-mib 2000 did the same
    (door-row check F1, 2026-11-13). A row loaded from the cache asks at least its class's floor, and a card budget
    the card cannot give is refused."""
    table = d.load_table()
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    monkeypatch.setenv(rows.MESH_PATH_ENV, str(tmp_path / "meshes"))
    _mesh(tmp_path, "w900")
    monkeypatch.setattr(rows, "mesh_facts", lambda grid, init=None: {
        "cells": 135318, "edges": 406450, "levels": 55, "nominal_dc_m": 3000.0, "limited_area": True})
    name, made = rows.make_row(table, "hrrrv5", rows.find_mesh("w900"), dt=15.0, card_mib=97887)
    assert made["need_free_mib"] == 16384 and rows.class_floor_mib(table, made) == 16384
    path = rows.rows_cache() / name / "row.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["row"]["need_free_mib"] = 1000   # the planted row
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert rows.cached_rows()[name]["need_free_mib"] == 1000
    # --mesh: the cached row is reused (same template, class and dt) and asks the class floor
    _, loaded = rows.make_row(table, "hrrrv5", rows.find_mesh("w900"), dt=15.0, card_mib=97887)
    assert loaded["need_free_mib"] == 16384 and loaded["window_args"] == made["window_args"]
    # --row NAME: the door's own load path floors the cached row the same way
    assert rows.with_class_floor(table, rows.cached_rows()[name])["need_free_mib"] == 16384
    monkeypatch.setattr(d, "total_mib", lambda card: 97887)
    monkeypatch.setattr(d, "free_mib", lambda card: 4000)
    with pytest.raises(d.AdDoorError, match="card 0 has 4000 MiB free; this row's window needs 16384 MiB"):
        d.admit_card("0", loaded["need_free_mib"], 0.0)
    # --need-mib 2000 passes the admission on 4,000 MiB free; the card word then refuses
    assert d.admit_card("0", 2000, 0.0)["free_mib"] == 4000
    with pytest.raises(rows.AdRowError, match="card 0 has 4,000 MiB free; class 3km-bounded needs at least 11,264"):
        rows.fill_card_words(loaded["window_args"], loaded["card_words"], 4000, 500000, card="0",
                             label="3km-bounded")
    # a row asking more than its floor keeps its own need; a shipped row (no generated) is left as it is
    assert rows.with_class_floor(table, {**made, "need_free_mib": 20000})["need_free_mib"] == 20000
    shipped = table["rows"][table["default_row"]]
    assert rows.with_class_floor(table, shipped) is shipped


def test_the_bounded_class_label_names_its_own_window_and_kernel_set():
    """THE BREAKAGE: the bounded class's words began with the whole 3 km class's ("3 km storm box, 4-step window on
    ONE card ... kernel set storm-h"), a window and kernel set it does not run (door-row check F2, 2026-11-13)."""
    table = d.load_table()
    for suite_name, recipe in (("hrrrv5", "3km-bounded"), ("hrrrv5-exact", "3km-exact-bounded")):
        cls = [c for c in table["suites"][suite_name]["window_classes"] if c["from_recipe"] == recipe]
        assert len(cls) == 1
        for key in ("what", "why"):
            text = cls[0][key]
            assert "storm box" not in text and "storm-box" not in text, (recipe, key)
            assert "storm-h" not in text.replace("storm-h-ws", ""), (recipe, key)
            assert "ONE card" not in text, (recipe, key)
        assert "kernel set storm-h-ws" in cls[0]["what"] and "200,000 cells" in cls[0]["what"]


def test_a_cached_made_row_is_checked_against_its_template_not_taken_as_shipped(monkeypatch, tmp_path):
    # the door puts the cached rows into table["rows"] before make_row; a stale made row must be made again
    import pytest as _pytest
    from woof.hex import ad_doors as _d
    from woof.hex.ad import rows as _rows
    table = _d.load_table()
    name = _rows.row_name("hrrrv5", "m-stale")
    stale = {"generated": {"template_sha256": "stale"}, "dt_s": 150.0, "cells": 10}
    monkeypatch.setattr(_rows, "cached_rows", lambda: {name: stale})
    table["rows"] = {name: stale, **table["rows"]}

    def remade(grid, init):
        raise RuntimeError("made again")
    monkeypatch.setattr(_rows, "mesh_facts", remade)
    with _pytest.raises(RuntimeError, match="made again"):
        _rows.make_row(table, "hrrrv5", {"mesh": "m-stale", "grid": tmp_path / "g.grid.nc", "oro": None})
