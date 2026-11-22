# SPDX-License-Identifier: Apache-2.0
"""Three first-contact refusals that came after the downloads, now before them. No card.

THE BREAKAGES (the dist48 checker's fresh installs of the sealed wheel, 2026-11-15):
* without the ``ad`` extra, or on Python 3.12, the README's 12 km command fetched all four files (1.19 GB) and started
  the runner; the 120 km example was refused at once, naming the missing library and the fix. The table named the
  libraries of the 120 km and 3 km example plans only;
* ``--at 10,-150`` on the 12 km cut fetched the example and then said "the perturbation is zero on every cell"; the
  2 MB mesh file answers it;
* on an H100 the command said "its derivative images are built for sm_120 and the row names no image recipe for
  sm_90": neither which cards run it nor what to do.
"""
import argparse
import hashlib
import re

import numpy as np
import pytest

from woof.hex import ad_doors as d
from woof.hex import ad_session as s
from woof.hex.ad import assets, rows

CUT = "hrrrv5-q12.96.2890.n40.60w91.30"
STORM = "hrrrv5-q3.96.38278.n40.60w91.30"
README_12 = ["adjoint-test", "--row", CUT, "--steps", "15", "--at", "40.6,-91.3", "--radius-km", "150"]


def _parse(argv):
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def _cut_geometry(cells=2890):
    """Cells on a cap of 405 km around 40.6N 91.3W, as the 12 km cut's (a stand-in: the real grid is a download)."""
    rng = np.random.Generator(np.random.PCG64(7))
    r = 405.0 * np.sqrt(rng.random(cells)) / d.EARTH_RADIUS_KM
    az = 2 * np.pi * rng.random(cells)
    la0, lo0 = np.radians(40.6), np.radians(-91.3)
    lat = np.arcsin(np.sin(la0) * np.cos(r) + np.cos(la0) * np.sin(r) * np.cos(az))
    lon = lo0 + np.arctan2(np.sin(az) * np.sin(r) * np.cos(la0), np.cos(r) - np.sin(la0) * np.sin(lat))
    return {"latCell": np.degrees(lat), "lonCell": np.degrees(lon), "areaCell": np.ones(cells)}


@pytest.fixture
def user(monkeypatch, tmp_path):
    """A 32 GB card of the right family, a host and disk that fit, a runner, a sessions folder of the test's own; the
    asset module fetches the mesh file alone (recorded) and refuses any other fetch."""
    for name in d.NO_LOCAL_GPU_ENV + s.SCHEDULER_ENV:
        monkeypatch.delenv(name, raising=False)
    for name in ("WOOF_HEX_AD_ASSETS", s.MODE_ENV, rows.IMAGE_BUILDER_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(s.SESSIONS_ENV, str(tmp_path / "sessions"))
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    fetched = []
    root = tmp_path / "root"

    def asset_root(cache=None, want=None, rows=True):
        if rows is False:
            fetched.append(("mesh only", list(want)))
            for rel in want:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_bytes(b"grid")
            return root
        fetched.append(("asset_root", want))
        raise AssertionError("the asset root was resolved (a first use downloads the example)")

    monkeypatch.setattr(assets, "asset_root", asset_root)
    monkeypatch.setattr(assets, "_fetch", lambda *a, **k: fetched.append("_fetch"))
    monkeypatch.setattr(d, "mesh_geometry", lambda grid, cells, edges: _cut_geometry(cells))
    monkeypatch.setattr(d, "resolve_runner", lambda explicit: tmp_path / "runner")
    monkeypatch.setattr(d, "compute_capability", lambda card: "12.0")
    monkeypatch.setattr(d, "total_mib", lambda card: 32607)
    monkeypatch.setattr(d, "free_mib", lambda card: 32000)
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 900_000)
    monkeypatch.setattr(d, "host_available_settled", lambda *a, **k: 900_000)
    monkeypatch.setattr(d, "_free_disk_mib", lambda folder: 9_000_000)
    monkeypatch.setattr(d, "install_site_packages", lambda: tmp_path / "site")   # an install without the ad extra
    return fetched


# ---- 1. the install check -----------------------------------------------------------------------------------------
def test_every_example_plan_names_the_libraries_it_replays():
    # a row whose example plan has no list is not judged at all: that is how the 12 km cut lost its install check
    table = d.load_table()
    for name, row in table["rows"].items():
        if not row.get("example_plan"):
            continue
        libraries = d.plan_libraries(table, row)
        assert [x["path"].split("/")[0] for x in libraries] == ["nvidia", "cupy", "cupy"], name
        assert all(len(x["sha256"]) == 64 and x["from"] for x in libraries), name
        assert "Python 3.14" in table["example_plan_install"][row["example_plan"]], name


def test_the_12km_cut_replays_the_same_library_files_as_the_other_examples():
    # the 12 km plan.json names these three files with these sums (read from the pinned plan, 2026-11-15)
    table = d.load_table()
    cut = d.plan_libraries(table, table["rows"][CUT])
    assert cut == d.plan_libraries(table, table["rows"][STORM])
    assert {x["path"]: x["sha256"][:12] for x in cut} == {
        "nvidia/cu13/lib/libcublas.so.13": "59dd485de665",
        "cupy/cuda/cub.cpython-314-x86_64-linux-gnu.so": "71e19ac3ad7a",
        "cupy/cuda/thrust.cpython-314-x86_64-linux-gnu.so": "84f5044ce641"}


def test_the_readme_12km_command_without_the_ad_extra_is_refused_before_any_download(user, tmp_path, capsys):
    with pytest.raises(d.AdDoorError, match=r"row hrrrv5-q12\.96\.2890\.n40\.60w91\.30: its example plan replays "
                                            r"nvidia/cu13/lib/libcublas\.so\.13 .*is not in this install.*"
                                            r"Python 3\.14 environment.*recast-woof\[ad\].*Nothing was downloaded"):
        d.Context(_parse(README_12 + ["--out", str(tmp_path / "o")]), "adjoint-test")
    assert user == [] and "STARTED FRESH" not in capsys.readouterr().out


def test_python_312_with_the_ad_extra_is_refused_before_any_download(user, tmp_path):
    # pip installs [ad] on 3.12 without complaint: cuBLAS is there, CuPy's modules are the 3.12 build
    site = tmp_path / "site"
    (site / "nvidia/cu13/lib").mkdir(parents=True)
    (site / "cupy/cuda").mkdir(parents=True)
    (site / "cupy/cuda/cub.cpython-312-x86_64-linux-gnu.so").write_bytes(b"3.12 build")
    table = d.load_table()
    lib = d.plan_libraries(table, table["rows"][CUT])[0]
    blob = b"the recorded cuBLAS"
    (site / lib["path"]).write_bytes(blob)
    real = dict(lib, sha256=hashlib.sha256(blob).hexdigest())
    table["example_plan_libraries"][table["rows"][CUT]["example_plan"]][0] = real
    with pytest.raises(d.AdDoorError, match=r"cub\.cpython-314-x86_64-linux-gnu\.so \(the CUB module of cupy-cuda13x "
                                            r"14\.2\.0 for CPython 3\.14\).*is not in this install.*Nothing was "
                                            r"downloaded"):
        d.refuse_plan_libraries_missing(table, CUT, table["rows"][CUT], site)


# ---- 2. a point off the cut ---------------------------------------------------------------------------------------
@pytest.fixture
def installed(user, monkeypatch):
    monkeypatch.setattr(d, "refuse_plan_libraries_missing", lambda *a, **k: None)   # an install that holds them
    return user


def test_a_bump_off_the_cut_is_refused_on_the_mesh_file_alone(installed, tmp_path, capsys):
    argv = ["adjoint-test", "--row", CUT, "--steps", "15", "--at", "10,-150", "--radius-km", "150"]
    with pytest.raises(d.AdDoorError, match=r"--at 10,-150: the bump \(Gaussian R 150 km, reaching about 455 km\) "
                                            r"touches no cell of row hrrrv5-q12\.96\.2890\.n40\.60w91\.30's mesh .*"
                                            r"nearest cell is [0-9,]+ km .*--at [0-9.]+,-[0-9.]+ is its centre.*"
                                            r"Nothing but the row's mesh file was downloaded"):
        d.Context(_parse(argv + ["--out", str(tmp_path / "o")]), "adjoint-test")
    assert installed == [("mesh only", ["meshes/q12.96.2890.n40.60w91.30.grid.nc"])]
    assert "STARTED FRESH" not in capsys.readouterr().out
    # the tangent door takes the same bump
    installed.clear()
    with pytest.raises(d.AdDoorError, match="touches no cell"):
        d.Context(_parse(["tangent"] + argv[1:] + ["--out", str(tmp_path / "t")]), "tangent")
    assert installed == [("mesh only", ["meshes/q12.96.2890.n40.60w91.30.grid.nc"])]


def test_the_readme_point_goes_on_to_the_downloads(installed, tmp_path):
    with pytest.raises(AssertionError, match="asset root was resolved"):
        d.Context(_parse(README_12 + ["--out", str(tmp_path / "o")]), "adjoint-test")
    assert [x[0] for x in installed] == ["mesh only", "asset_root"]


def test_a_metric_region_off_the_cut_is_refused_on_the_mesh_file_alone(installed, tmp_path):
    with pytest.raises(d.AdDoorError, match=r"the metric region \(cells within 150 km of 10\.0N -150\.0E\) holds no "
                                            r"cell of this mesh \(mesh q12\.96\.2890\.n40\.60w91\.30: cells at .*\)\. "
                                            r"Nothing but the row's mesh file was downloaded"):
        d.Context(_parse(["sensitivity", "--row", CUT, "--steps", "15", "--circle", "10,-150,150",
                          "--out", str(tmp_path / "o")]), "sensitivity")
    assert [x[0] for x in installed] == ["mesh only"]
    with pytest.raises(d.AdDoorError, match=r"box 0\.0\.\.5\.0 N.*holds no cell"):
        d.Context(_parse(README_12 + ["--box", "0,5,0,5", "--out", str(tmp_path / "b")]), "adjoint-test")


def test_level_and_size_words_are_judged_before_the_downloads(installed, tmp_path):
    with pytest.raises(d.AdDoorError, match=r"--level 60: row hrrrv5-q12\.96\.2890\.n40\.60w91\.30 has model levels "
                                            r"1\.\.59"):   # the plan's 59 levels (sensitivity.nc nVertLevels)
        d.Context(_parse(README_12 + ["--level", "60", "--out", str(tmp_path / "o")]), "adjoint-test")
    with pytest.raises(d.AdDoorError, match="--amplitude 0 --radius-km 150: the bump is zero on every cell"):
        d.Context(_parse(README_12 + ["--amplitude", "0", "--out", str(tmp_path / "o")]), "adjoint-test")
    assert {x[0] for x in installed} == {"mesh only"}


def test_adjoint_test_takes_a_box_without_the_default_circle():
    # the default circle went in beside --box, and select_cells refuses the two together
    args = _parse(README_12 + ["--box", "40,41,-92,-91", "--out", "o"])
    assert d.adjoint_test_region(args, "40.6,-91.3") == ("40,41,-92,-91", None)
    args = _parse(README_12 + ["--out", "o"])
    assert d.adjoint_test_region(args, "40.6,-91.3") == (None, "40.6,-91.3,187.5")
    cells, where = d.select_cells(_cut_geometry(), *d.adjoint_test_region(
        _parse(README_12 + ["--box", "40,41,-92,-91", "--out", "o"]), "40.6,-91.3"))
    assert cells.size > 0 and where.startswith("box")


def test_the_early_bump_rule_is_the_directions_rule():
    # a cell is on the bump when its Gaussian factor is above BUMP_KEEP: about 3.03 radii, as theta_direction keeps
    geometry = _cut_geometry()
    dist = d.great_circle_km(40.6, -91.3, geometry["latCell"], geometry["lonCell"])
    assert dist.max() < 410
    near = 40.6 + 1.01 * (405 + 3.0349 * 100) / 111.2   # just beyond the cap's edge plus the reach of R 100 km
    d.refuse_off_mesh(d.load_table(), CUT, d.load_table()["rows"][CUT],
                      _parse(README_12 + ["--at", "40.6,-91.3", "--out", "o"]), "adjoint-test", geometry)
    with pytest.raises(d.AdDoorError, match="touches no cell"):
        d.refuse_off_mesh(d.load_table(), CUT, d.load_table()["rows"][CUT],
                          _parse(README_12 + ["--at", f"{near},-91.3", "--radius-km", "100", "--out", "o"]),
                          "adjoint-test", geometry)


# ---- 3. an older card family ---------------------------------------------------------------------------------------
def test_an_h100_is_refused_in_plain_words(installed, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(d, "compute_capability", lambda card: "9.0")
    monkeypatch.setattr(d, "total_mib", lambda card: 81559)
    monkeypatch.setattr(d, "free_mib", lambda card: 81000)
    with pytest.raises(d.AdDoorError) as caught:
        d.Context(_parse(README_12 + ["--out", str(tmp_path / "o")]), "adjoint-test")
    words = re.sub(r"and card \S+ is", "and card N is", str(caught.value))   # N: the card the door was given
    need = int(re.search(r"with at least ([0-9,]+) MiB free", words).group(1).replace(",", ""))
    assert 25_000 <= need <= 32_000   # the door's own need for this command (the row's, or its route's)
    words = words.replace(f"{need:,} MiB free", "NEED MiB free")
    assert words.startswith("row hrrrv5-q12.96.2890.n40.60w91.30 runs on NVIDIA Blackwell cards of compute "
                            "capability 12.0 (GeForce RTX 50 series, RTX PRO Blackwell) only, and card N is a Hopper "
                            "card (compute capability 9.0, as H100, H200, GH200). To run it, use a compute "
                            "capability 12.0 card with at least NEED MiB free, for example GeForce RTX 5090 (32 GB) "
                            "or RTX PRO 6000 Blackwell (96 GB)."), words
    assert words.endswith("Nothing was downloaded") and installed == []
    assert "STARTED FRESH" not in capsys.readouterr().out


def test_the_storm_row_names_only_the_card_that_holds_it(installed, tmp_path, monkeypatch):
    monkeypatch.setattr(d, "compute_capability", lambda card: "8.9")
    monkeypatch.setattr(d, "total_mib", lambda card: 97887)   # a 96 GB card of another family, so only the family is wrong
    monkeypatch.setattr(d, "free_mib", lambda card: 97000)
    with pytest.raises(d.AdDoorError, match=r"card \S+ is an Ada Lovelace card \(compute capability 8\.9, as GeForce RTX "
                                            r"40 series.*at least 96,600 MiB free, for example RTX PRO 6000 Blackwell "
                                            r"\(96 GB\)\. \(Its derivative images are compiled for sm_120"):
        d.Context(_parse(["sensitivity", "--row", STORM, "--steps", "59", "--circle", "40.6,-91.3,150",
                          "--out", str(tmp_path / "o")]), "sensitivity")
    assert installed == []


def test_a_24gb_card_of_an_older_family_hears_its_family_before_its_size(installed, tmp_path, monkeypatch):
    # an RTX 4090 (8.9, 24,564 MiB) is too small for the cut and of the wrong family: a bigger card of its family would
    # not run it either, so the family is what it is told
    monkeypatch.setattr(d, "compute_capability", lambda card: "8.9")
    monkeypatch.setattr(d, "total_mib", lambda card: 24564)
    monkeypatch.setattr(d, "free_mib", lambda card: 24000)
    with pytest.raises(d.AdDoorError, match=r"is an Ada Lovelace card .*use a compute capability 12\.0 card with at "
                                            r"least [0-9]{2},[0-9]{3} MiB free"):
        d.Context(_parse(README_12 + ["--out", str(tmp_path / "o")]), "adjoint-test")
    assert installed == []


def test_card_family_words():
    assert rows.capability_of("sm_120") == "12.0" and rows.capability_of("sm_90") == "9.0"
    assert rows.capability_of("sm_100") == "10.0" and rows.capability_of("sm_90a") == "9.0"
    assert rows.family_words("sm_80") == "an Ampere card (compute capability 8.0, as A100, A30)"
    assert rows.family_words("sm_53") == "a card of compute capability 5.3"
    for cc in ("7.0", "7.5", "8.0", "8.6", "8.9", "9.0", "10.0", "12.0"):
        assert rows.arch_of(cc) in rows.CARD_FAMILIES


# ---- 4. the order: what the command asked comes first, the machine's state last -------------------------------------
# THE BREAKAGE (the 0.6.0 candidate's checker on a shared card and a busy host, 2026-11-15): a Python 3.12 install, an
# install without the ad extra and a bump off the cut were each first told to free the card (exit 2) or that the host
# had 26,828 MiB where the row needs 46,080 (exit 3). The user frees memory, runs again, and only then hears the
# install or the command was wrong.
@pytest.fixture
def busy(monkeypatch):
    """The card holds another job (25 GB free of the row's 28,700 MiB) and the host is busy (27 GB available)."""
    def make():
        monkeypatch.setattr(d, "free_mib", lambda card: 25000)
        monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 27000)
        monkeypatch.setattr(d, "host_available_settled", lambda *a, **k: 27000)
    return make


def test_on_a_busy_machine_a_missing_extra_is_said_before_the_memory(user, busy, tmp_path, capsys):
    busy()
    with pytest.raises(d.AdDoorError, match=r"is not in this install.*recast-woof\[ad\].*Nothing was downloaded") as e:
        d.Context(_parse(README_12 + ["--out", str(tmp_path / "o")]), "adjoint-test")
    assert not isinstance(e.value, d.HostMemoryError) and "MiB free" not in str(e.value)
    assert user == [] and "STARTED FRESH" not in capsys.readouterr().out


def test_on_a_busy_machine_a_bump_off_the_cut_is_said_before_the_memory(installed, busy, tmp_path):
    busy()
    argv = ["adjoint-test", "--row", CUT, "--steps", "15", "--at", "10,-150", "--radius-km", "150"]
    with pytest.raises(d.AdDoorError, match="touches no cell") as e:
        d.Context(_parse(argv + ["--out", str(tmp_path / "o")]), "adjoint-test")
    assert not isinstance(e.value, d.HostMemoryError)
    for more, words in ((["--level", "60"], "has model levels"), (["--steps", "16"], "1 to 15 steps")):
        with pytest.raises(d.AdDoorError, match=words):
            d.Context(_parse(README_12 + more + ["--out", str(tmp_path / "o")]), "adjoint-test")
    with pytest.raises(d.AdDoorError, match="offers guarded, tape-free, taped"):   # the command's handler judges it first of all
        d.refuse_speed_words(_parse(README_12 + ["--boundary-layer", "floored", "--out", "o"]), d.load_table())
    with pytest.raises(d.AdDoorError, match="holds no cell"):
        d.Context(_parse(["sensitivity", "--row", CUT, "--steps", "15", "--circle", "10,-150,150",
                          "--out", str(tmp_path / "o")]), "sensitivity")


def test_on_a_busy_machine_a_right_command_hears_the_card_then_the_host(installed, busy, tmp_path, monkeypatch):
    busy()
    with pytest.raises(d.AdDoorError, match=r"has 25000 MiB free; this row's window needs 28700 MiB"):
        d.Context(_parse(README_12 + ["--out", str(tmp_path / "o")]), "adjoint-test")
    monkeypatch.setattr(d, "free_mib", lambda card: 32000)   # the card is free now; the host is still busy
    with pytest.raises(d.HostMemoryError, match=r"this host has 27,000 MiB of memory available .* needs about "
                                                r"[0-9,]+ MiB.*Nothing but the row's mesh file was downloaded"):
        d.Context(_parse(README_12 + ["--out", str(tmp_path / "o")]), "adjoint-test")
    assert [x[0] for x in installed] == ["mesh only", "mesh only"]   # the 2 MB mesh, never the example


def test_a_card_too_small_in_all_is_said_before_the_install(user, busy, tmp_path):
    # no waiting and no install changes a card's whole size: the 3 km row on a 32 GB card, on an install without the extra
    busy()
    with pytest.raises(d.AdDoorError, match="more than the whole card"):
        d.Context(_parse(["adjoint-test", "--row", STORM, "--steps", "59", "--at", "40.6,-91.3", "--radius-km", "150",
                          "--out", str(tmp_path / "o")]), "adjoint-test")
    assert user == []


def test_a_command_that_is_refused_for_its_words_stops_no_loaded_program(installed, busy, tmp_path, monkeypatch):
    # the session is opened (and a program found loaded on the card stopped) only after the command's own words pass
    opened = []
    real = s.open_for
    monkeypatch.setattr(s, "open_for", lambda ctx, defer=False: opened.append(1) or real(ctx, defer=defer))
    with pytest.raises(d.AdDoorError, match="touches no cell"):
        d.Context(_parse(["adjoint-test", "--row", CUT, "--steps", "15", "--at", "10,-150", "--radius-km", "150",
                          "--out", str(tmp_path / "o")]), "adjoint-test")
    assert opened == []
