"""``rw_ensbatch`` / ``rw_obsgrid``: the seam, the contract, the artifacts.

The two binaries that close the render law's remaining holes -- ensemble
products (``woof/da/enprod.py`` said outright there was "no second engine
to switch to") and gridded radar observations.

Like ``tests/test_render_rust.py``, nothing here mocks an engine: the
end-to-end tests drive the real executables this checkout built, and skip
accurately when it has not built them.  What IS stubbed is the subprocess
layer of the resolver's decision table, because a stale-build refusal
cannot be produced without shipping a stale build.
"""

from __future__ import annotations

import datetime
import inspect
import re
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof import rustwx_lanes

_NZ, _NY, _NX = 3, 16, 20
_STAMP = "2026-07-28_20:00:00"


def _built(name: str) -> Path | None:
    from woof.bridges import executable_name

    path = (rustwx_lanes.crate_dir() / "target" / "release"
            / executable_name(name))
    return path if path.is_file() else None


_ENSEMBLE = _built(rustwx_lanes.ENSEMBLE_NAME)
_OBSGRID = _built(rustwx_lanes.OBSGRID_NAME)
_COMPARE = _built(rustwx_lanes.COMPARE_NAME)
_BUILD_HINT = ("this checkout has not built the rustwx workspace (cd "
               "tools/rustwx && cargo build --release --locked --offline)")

needs_ensemble = pytest.mark.skipif(_ENSEMBLE is None, reason=_BUILD_HINT)
needs_obsgrid = pytest.mark.skipif(_OBSGRID is None, reason=_BUILD_HINT)
needs_compare = pytest.mark.skipif(_COMPARE is None, reason=_BUILD_HINT)


# ---------------------------------------------------------------------------
# The contract markers, against the real binaries
# ---------------------------------------------------------------------------

@needs_ensemble
def test_the_ensemble_marker_is_the_built_binary_s_own_answer():
    """The Python constant against the Rust writer, not against itself.

    A marker both halves read from one Python string proves nothing.  The
    renderer incident (task #106) was exactly this: two builds with
    different md5s both reported ``verified`` because nothing asked them
    what grammar they spoke.
    """

    probe = subprocess.run([str(_ENSEMBLE), "--abi"], capture_output=True,
                           text=True, errors="replace", timeout=60)
    assert probe.returncode == 0, probe.stderr
    # Since 356c94560 the binary names the revision it was built from on a
    # second line.  The contract is the first line alone; comparing the
    # whole answer refused this checkout's own build, so `woof enprod
    # --engine rust` refused to draw.  The probe the doors use must accept
    # the build this checkout made.
    lines = probe.stdout.strip().splitlines()
    assert lines[0] == rustwx_lanes.ENSEMBLE_ABI_MARKER
    assert all(line.startswith("GPUWM_BRIDGE_SOURCE_REV=")
               for line in lines[1:]) and len(lines) <= 2, lines
    usable, evidence = rustwx_lanes.probe_ensemble_bin(_ENSEMBLE)
    assert usable, evidence


@needs_obsgrid
def test_the_obsgrid_marker_is_the_built_binary_s_own_answer():
    probe = subprocess.run([str(_OBSGRID), "--abi"], capture_output=True,
                           text=True, errors="replace", timeout=60)
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == rustwx_lanes.OBSGRID_ABI_MARKER


@needs_compare
def test_the_compare_marker_is_the_built_binary_s_own_answer():
    probe = subprocess.run([str(_COMPARE), "--abi"], capture_output=True,
                           text=True, errors="replace", timeout=60)
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == rustwx_lanes.COMPARE_ABI_MARKER


def test_no_two_bundled_binaries_pin_the_same_marker():
    """The renderer was the odd one out once; nothing may be again.

    Six binaries now answer ``--abi``.  If two of them pinned the same
    line, one would verify against the other's contract and a bundle
    could ship the wrong pair with every probe green.
    """

    from woof import rustwx, rustwx_fetch
    from woof.obs import nexrad

    markers = {
        "rw_fetch": rustwx_fetch.FETCH_ABI_MARKER,
        "rw_nexrad": nexrad.NEXRAD_ABI_MARKER,
        "rw_wrfbatch": rustwx.RENDERER_ABI_MARKER,
        "rw_ensbatch": rustwx_lanes.ENSEMBLE_ABI_MARKER,
        "rw_obsgrid": rustwx_lanes.OBSGRID_ABI_MARKER,
        "rw_compare": rustwx_lanes.COMPARE_ABI_MARKER,
    }
    for name, marker in markers.items():
        assert isinstance(marker, str) and marker.strip(), name
        assert "\t" in marker, f"{name}: not a tab-separated contract line"
    assert len(set(markers.values())) == len(markers), markers


def test_a_binary_that_predates_the_handshake_is_refused(monkeypatch,
                                                         tmp_path):
    """Exit 2 with no ``--abi`` line: every build older than the contract."""

    from woof import bridges

    monkeypatch.setattr(bridges, "launchable", lambda path: (True, "ok"))
    monkeypatch.setattr(
        rustwx_lanes.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=2, stdout="", stderr=""))
    ok, evidence = rustwx_lanes.probe_ensemble_bin(tmp_path / "rw_ensbatch")
    assert ok is False
    assert "--abi does not match the contract" in evidence
    assert "cargo build" in evidence


def test_compare_rejects_a_binary_that_cannot_apply_the_selected_theme(monkeypatch, tmp_path):
    from woof import bridges

    monkeypatch.setattr(bridges, "launchable", lambda path: (True, "ok"))
    old_marker = rustwx_lanes.COMPARE_ABI_MARKER.split(
        "\tgpuwm-rw-compare-presentation-v1", 1)[0]
    monkeypatch.setattr(rustwx_lanes.subprocess, "run", lambda *a, **k:
        SimpleNamespace(returncode=0, stdout=old_marker + "\n", stderr=""))
    ok, evidence = rustwx_lanes.probe_compare_bin(tmp_path / "rw_compare")
    assert ok is False
    assert "--abi does not match the contract" in evidence


def test_a_binary_answering_the_right_contract_is_accepted(monkeypatch,
                                                           tmp_path):
    """The negative control: the guard does not fire on the right answer."""

    from woof import bridges

    monkeypatch.setattr(bridges, "launchable", lambda path: (True, "ok"))
    monkeypatch.setattr(
        rustwx_lanes.subprocess, "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=rustwx_lanes.OBSGRID_ABI_MARKER + "\n",
            stderr=""))
    ok, evidence = rustwx_lanes.probe_obsgrid_bin(tmp_path / "rw_obsgrid")
    assert ok is True, evidence


_HEAD = "a" * 40
_OTHER = "b" * 40


def _stamped_ensemble(monkeypatch, tmp_path, *, built_from, printed=None,
                      declare=False):
    """An ``rw_ensbatch`` outside this tree, answering ``--abi`` stamped.

    The bytes carry the stamp the way a built binary does, because the
    shared check (:func:`woof.provenance_gate.bridge_tree_match`) reads
    it there; the checkout is a stand-in at ``_HEAD``.
    """

    from woof import bridges, provenance_gate

    binary = tmp_path / "elsewhere" / "rw_ensbatch"
    binary.parent.mkdir()
    binary.write_bytes(b"\x7fELF" + provenance_gate.ABI_SOURCE_REV_PREFIX
                       .encode("ascii") + built_from.encode("ascii"))
    tree = tmp_path / "checkout"
    tree.mkdir()
    monkeypatch.setattr(provenance_gate, "resolve", lambda: SimpleNamespace(
        install_kind="editable", source_root=str(tree),
        package_path=str(tree / "woof"), git={"commit_full": _HEAD}))
    if declare:
        monkeypatch.setenv(rustwx_lanes.ENSEMBLE_ENV, str(binary))
    else:
        monkeypatch.delenv(rustwx_lanes.ENSEMBLE_ENV, raising=False)
    monkeypatch.setattr(bridges, "launchable", lambda path: (True, "ok"))
    answer = (f"{rustwx_lanes.ENSEMBLE_ABI_MARKER}\n"
              f"GPUWM_BRIDGE_SOURCE_REV={printed or built_from}\n")
    monkeypatch.setattr(
        rustwx_lanes.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=answer,
                                        stderr=""))
    return binary


def test_a_stamped_answer_from_this_checkout_s_revision_is_accepted(
        monkeypatch, tmp_path):
    """The revision line is not part of the contract (2.8.7 enprod refusal)."""

    binary = _stamped_ensemble(monkeypatch, tmp_path, built_from=_HEAD)
    usable, evidence = rustwx_lanes.probe_ensemble_bin(binary)
    assert usable, evidence


def test_a_stamped_answer_from_another_checkout_is_refused_naming_both(
        monkeypatch, tmp_path):
    """Splitting the stamp off must not open the door the compare guarded."""

    binary = _stamped_ensemble(monkeypatch, tmp_path, built_from=_OTHER)
    usable, evidence = rustwx_lanes.probe_ensemble_bin(binary)
    assert usable is False
    assert _OTHER in evidence and _HEAD in evidence, evidence
    assert "another checkout" in evidence
    assert "cargo build" in evidence


def test_a_declared_stamped_binary_is_the_caller_s_choice(monkeypatch,
                                                          tmp_path):
    """Named through the override, as the renderer gate treats one."""

    binary = _stamped_ensemble(monkeypatch, tmp_path, built_from=_OTHER,
                               declare=True)
    usable, evidence = rustwx_lanes.probe_ensemble_bin(binary)
    assert usable, evidence


def test_only_the_stamp_line_is_split_off_the_contract(monkeypatch,
                                                       tmp_path):
    """Any other extra line still fails the exact comparison."""

    from woof import bridges

    monkeypatch.setattr(bridges, "launchable", lambda path: (True, "ok"))
    for answer in (
            f"{rustwx_lanes.ENSEMBLE_ABI_MARKER}\nsomething-else\n",
            f"GPUWM_BRIDGE_SOURCE_REV={_HEAD}\n"
            f"{rustwx_lanes.ENSEMBLE_ABI_MARKER}\n",
            f"{rustwx_lanes.ENSEMBLE_ABI_MARKER}\n"
            f"GPUWM_BRIDGE_SOURCE_REV={_HEAD}\n"
            f"GPUWM_BRIDGE_SOURCE_REV={_HEAD}\n"):
        monkeypatch.setattr(
            rustwx_lanes.subprocess, "run",
            lambda *a, _answer=answer, **k: SimpleNamespace(
                returncode=0, stdout=_answer, stderr=""))
        usable, evidence = rustwx_lanes.probe_ensemble_bin(
            tmp_path / "rw_ensbatch")
        assert usable is False, answer
        assert "--abi does not match the contract" in evidence


def test_an_override_naming_a_missing_file_is_a_hard_error(monkeypatch,
                                                           tmp_path):
    """Explicit configuration fails loudly, never falls through.

    Falling through would put the panels on matplotlib while the operator
    believed they were on the Rust engine, which is the render law
    violated silently.
    """

    missing = tmp_path / "nope.exe"
    monkeypatch.setenv(rustwx_lanes.ENSEMBLE_ENV, str(missing))
    with pytest.raises(FileNotFoundError, match="names a missing file"):
        rustwx_lanes.find_ensemble_bin()


# ---------------------------------------------------------------------------
# The engine resolver `woof enprod --engine` uses
# ---------------------------------------------------------------------------

def test_a_stale_binary_refuses_on_auto_as_well_as_on_rust(monkeypatch):
    """``auto`` has two answers, rust or a refusal; it does not degrade.

    This replaces the pair that pinned the degrade.  The render law
    permits ONE fallback and this is not it: an auto that quietly drew
    the same weather fields with matplotlib exited 0, so nothing in the
    run said which engine made the pictures.  The fallback stays
    reachable BY NAME, and is still never probed.
    """

    from woof.da.enprod import resolve_enprod_engine

    monkeypatch.setattr(rustwx_lanes, "find_ensemble_bin",
                        lambda: Path("staged-rw_ensbatch"))
    monkeypatch.setattr(rustwx_lanes, "probe_ensemble_bin",
                        lambda path: (False, "abi mismatch"))
    for request in ("auto", "rust"):
        with pytest.raises(RuntimeError, match="abi mismatch") as excinfo:
            resolve_enprod_engine(request)
        assert "--engine matplotlib" in str(excinfo.value)

    # ... and the fallback is never probed at all.
    def explode(path):
        raise AssertionError("--engine matplotlib must not probe the engine")

    monkeypatch.setattr(rustwx_lanes, "probe_ensemble_bin", explode)
    assert resolve_enprod_engine("matplotlib")[0] == "matplotlib"


def test_an_unbuilt_engine_names_the_build_line_and_the_way_out(monkeypatch):
    from woof.da.enprod import resolve_enprod_engine

    monkeypatch.setattr(rustwx_lanes, "find_ensemble_bin", lambda: None)
    with pytest.raises(RuntimeError) as excinfo:
        resolve_enprod_engine("auto")
    message = str(excinfo.value)
    assert "rw_ensbatch" in message
    assert "cargo build" in message
    assert "--engine matplotlib" in message


def test_both_lanes_resolve_their_engine_through_one_function(monkeypatch):
    """Lane rule 2: one question, one answer, two lanes."""

    from woof.da import enprod

    seen = []

    def record(request, **kwargs):
        seen.append((request, kwargs["name"]))
        return "rust", "recorded"

    monkeypatch.setattr(rustwx_lanes, "resolve_lane_engine", record)
    assert enprod.resolve_enprod_engine("auto") == ("rust", "recorded")
    assert rustwx_lanes.resolve_obsgrid_engine("auto") == ("rust", "recorded")
    assert seen == [("auto", rustwx_lanes.ENSEMBLE_NAME),
                    ("auto", rustwx_lanes.OBSGRID_NAME)]


def test_the_observation_grid_lane_refuses_by_name(monkeypatch):
    monkeypatch.setattr(rustwx_lanes, "find_obsgrid_bin", lambda: None)
    with pytest.raises(RuntimeError) as excinfo:
        rustwx_lanes.resolve_obsgrid_engine("rust")
    message = str(excinfo.value)
    assert rustwx_lanes.OBSGRID_NAME in message
    assert "cargo build" in message
    assert rustwx_lanes.resolve_obsgrid_engine("matplotlib")[0] == "matplotlib"


def test_the_ensemble_vocabulary_names_max_on_both_halves():
    """The ensemble max map is in the contract line, the product list
    and the default request, and enprod can name it."""

    from woof.da import enprod

    products_field = rustwx_lanes.ENSEMBLE_ABI_MARKER.split(
        "gpuwm-rw-ensbatch-fields-v1")[0].split("\t")
    assert products_field[0] == "gpuwm-rw-ensbatch-products-v1"
    assert [p for p in products_field[1:] if p] == list(
        rustwx_lanes.ENSEMBLE_PRODUCTS)
    assert "max" in rustwx_lanes.ENSEMBLE_PRODUCTS
    default = inspect.signature(
        rustwx_lanes.run_ensemble_renderer).parameters["products"].default
    assert default.split(",") == list(rustwx_lanes.ENSEMBLE_PRODUCTS)
    assert set(rustwx_lanes.ENSEMBLE_PRODUCTS) == set(enprod.PRODUCTS)


def test_the_matplotlib_ensemble_route_reports_itself_as_a_workaround():
    notice = rustwx_lanes.ensemble_workaround_notice()
    assert notice.startswith("WORKAROUND:")
    for field in rustwx_lanes.ENSEMBLE_FIELDS:
        assert field in notice
    for product in rustwx_lanes.ENSEMBLE_PRODUCTS:
        assert product in notice


def test_the_ensemble_engine_is_told_which_nest_to_reduce(monkeypatch, tmp_path):
    """``--domain dNN`` is the disambiguation the engine's refusal asks for."""

    seen = {}

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def record(command, **kwargs):
        seen["command"] = command
        return Result()

    monkeypatch.setattr(rustwx_lanes.subprocess, "run", record)
    rustwx_lanes.run_ensemble_renderer(
        tmp_path / "rw_ensbatch", tmp_path / "manifest.json",
        store_root=tmp_path, out_dir=tmp_path, domain="d02")
    assert "--domain" in seen["command"]
    assert seen["command"][seen["command"].index("--domain") + 1] == "d02"
    rustwx_lanes.run_ensemble_renderer(
        tmp_path / "rw_ensbatch", tmp_path / "manifest.json",
        store_root=tmp_path, out_dir=tmp_path)
    assert "--domain" not in seen["command"]


# ---------------------------------------------------------------------------
# The two halves of one command line
#
# The python half builds the argv and the RUST half parses it, and the
# only thing that ever made them agree was that somebody had run the
# pair.  A flag python emits that the engine has no arm for is not a
# refusal at plan review: it is `unknown option --domain` from the
# engine, raised after the manifest was read and the scratch store was
# opened.  These two read the engine's own grammar out of its source, so
# they answer with no build.
# ---------------------------------------------------------------------------

_ENSBATCH_SOURCE = (Path(__file__).resolve().parents[1] / "tools" / "rustwx"
                    / "crates" / "rw-wrfbatch" / "src" / "bin" / "ensbatch.rs")

_ARM = re.compile(
    r'^\s*("(?:--?[A-Za-z0-9-]+)"(?:\s*\|\s*"(?:--?[A-Za-z0-9-]+)")*)\s*=>',
    re.MULTILINE)


def _engine_option_arms() -> set[str]:
    """Every option literal ``rw_ensbatch``'s ``parse_args`` matches on.

    Read from the match arms and not from the usage line, because the
    usage line is prose and the arms are what actually accept a flag.
    """

    source = _ENSBATCH_SOURCE.read_text(encoding="utf-8")
    start = source.index("fn parse_args()")
    end = source.index("\nfn ", start + 1)
    arms: set[str] = set()
    for match in _ARM.finditer(source[start:end]):
        for literal in match.group(1).split("|"):
            arms.add(literal.strip().strip('"'))
    return arms


def test_the_engine_accepts_the_nest_choice_it_is_sent():
    """``--domain dNN`` is a flag the engine parses, not one it rejects."""

    arms = _engine_option_arms()
    assert "--store-root" in arms, (
        f"the arm scan found nothing useful: {sorted(arms)}")
    assert "--domain" in arms, (
        "woof/rustwx_lanes.py emits --domain to rw_ensbatch, so the "
        "engine has to have an arm for it; without one the enprod door "
        "passes plan review and the engine says 'unknown option "
        "--domain' after the roster is read")
    usage = _ENSBATCH_SOURCE.read_text(encoding="utf-8")
    assert "[--domain dNN]" in usage, "the usage line names the flag too"


def test_every_flag_the_ensemble_lane_emits_has_an_engine_arm(
        monkeypatch, tmp_path):
    """One command line, two halves, one vocabulary."""

    seen: dict[str, list[str]] = {}

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def record(command, **kwargs):
        seen["command"] = list(command)
        return Result()

    monkeypatch.setattr(rustwx_lanes.subprocess, "run", record)
    rustwx_lanes.run_ensemble_renderer(
        tmp_path / "rw_ensbatch", tmp_path / "manifest.json",
        store_root=tmp_path, out_dir=tmp_path, field="refl",
        products="mean,spread", threshold=35.0, neighborhood_km=40.0,
        frames=3, nan_policy="mask", pmm_tie_rule="flat-index",
        accept_status="DONE", source_label="WOOF", domain="d02",
        width=1200, height=900, members={0: tmp_path / "m0"})
    arms = _engine_option_arms()
    emitted = {token for token in seen["command"] if token.startswith("--")}
    assert emitted, "the builder emitted no flags at all"
    assert emitted <= arms, (
        f"rw_ensbatch has no arm for {sorted(emitted - arms)}; the engine "
        f"would answer 'unknown option' after the roster was read")


# ---------------------------------------------------------------------------
# End to end, against the real binaries
# ---------------------------------------------------------------------------

def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    assert data[12:16] == b"IHDR"
    return struct.unpack(">II", data[16:24])


@pytest.fixture(scope="module")
def ensemble_root(tmp_path_factory) -> Path:
    """The project's OWN synthetic ensemble writer, not a local fixture.

    Cross-lane rule: the reader is exercised against the other lane's
    real writer.  ``write_synthetic_ensemble`` is what ``woof enprod
    --make-fixture`` produces, so a fixture that reads here is a fixture
    an operator can produce.
    """

    from woof.da.enprod import write_synthetic_ensemble

    root = tmp_path_factory.mktemp("ens") / "ensemble"
    write_synthetic_ensemble(root, n_members=4)
    return root


@needs_ensemble
def test_the_ensemble_engine_draws_every_product(ensemble_root, tmp_path):
    from woof.da.enprod import MANIFEST_FILENAME

    out = tmp_path / "png"
    written, failures, skipped, report = rustwx_lanes.run_ensemble_renderer(
        _ENSEMBLE, ensemble_root / MANIFEST_FILENAME,
        store_root=tmp_path / "store", out_dir=out, field="refl",
        products=",".join(rustwx_lanes.ENSEMBLE_PRODUCTS), threshold=35.0,
        neighborhood_km=9.0, width=640, height=480)
    assert failures == [], failures
    assert len(written) == len(rustwx_lanes.ENSEMBLE_PRODUCTS), written
    assert skipped == []
    for path in written:
        assert path.is_file(), path
        assert path.stat().st_size > 5_000, path
        assert _png_size(path) == (640, 480), path

    # The panel stamps its own denominator; the report is that stamp, read
    # off the engine rather than recomputed, so a caption and a
    # provenance record cannot disagree.
    assert "members" in report and "n=4" in report["members"]
    assert "coverage" in report and "coverage=1.0000" in report["coverage"]
    assert "pmm_ties" in report
    # One stable colour per member NUMBER, which is the only label a
    # paintball plot has.
    legend = report["paintball_legend"]
    assert legend.split()[0].startswith("0=#")
    assert len(set(legend.split())) == 4


@needs_ensemble
def test_two_neighborhood_radii_are_two_files(ensemble_root, tmp_path):
    """A neighborhood is part of the product, so it is part of the name.

    MEASURED before this test existed: ``woof enprod --threshold 25
    --neighborhood-km 0,9,27`` reported ``3 panel(s)`` and left ONE file
    on disk.  The CLI takes a LIST of radii and drives the engine once
    per radius, and the engine named all of them
    ``ens_prob_refl_p25dbz`` -- so the 9 km field overwrote the point
    probability and the 27 km field overwrote that.  The matplotlib
    engine has always carried the ``r5km`` token; this is the Rust route
    catching up to it.
    """

    from woof.da.enprod import MANIFEST_FILENAME, radius_slug

    manifest = ensemble_root / MANIFEST_FILENAME
    produced: list[Path] = []
    for radius_km in (0.0, 9.0):
        written, failures, _skipped, _report = (
            rustwx_lanes.run_ensemble_renderer(
                _ENSEMBLE, manifest, store_root=tmp_path / "store",
                out_dir=tmp_path / "png", field="refl", products="prob",
                threshold=35.0, neighborhood_km=radius_km,
                width=320, height=240))
        assert failures == [], failures
        assert len(written) == 1, written
        produced.append(written[0])

    assert produced[0] != produced[1], produced
    assert produced[0].is_file() and produced[1].is_file()
    # And the token is the one the other engine writes, so the two
    # engines' filenames stay readable as the same grammar.
    assert radius_slug(9.0) in produced[1].name, produced[1].name
    assert radius_slug(9.0) not in produced[0].name, produced[0].name


@needs_ensemble
def test_the_render_layout_ruling_is_applied_at_write_time(ensemble_root,
                                                           tmp_path):
    """Never flat, at either end of the seam.

    The engine files under its OWN segments, which is why this call --
    which drives the engine directly, with no delivery -- still asserts
    them.  ``woof enprod`` points the engine at scratch and files the
    results itself, because the engine's first segment is a member count
    and its third is the literal word ``ensemble``, so two nests of one
    ensemble under one ``--out`` shared a folder.  That delivered tree is
    asserted in ``tests/test_enprod.py``.
    """

    from woof.da.enprod import MANIFEST_FILENAME

    out = tmp_path / "png"
    written, failures, _skipped, _report = (
        rustwx_lanes.run_ensemble_renderer(
            _ENSEMBLE, ensemble_root / MANIFEST_FILENAME,
            store_root=tmp_path / "store", out_dir=out, field="refl",
            products="mean", width=400, height=300))
    assert failures == [], failures
    assert not list(out.glob("*.png")), "the layout ruling was inverted flat"
    relative = written[0].relative_to(out).parts
    assert len(relative) == 4, relative
    assert relative[0].startswith("ens")
    assert relative[1] == "ens_mean_refl"


@needs_ensemble
def test_an_unknown_field_is_refused_with_the_vocabulary(ensemble_root,
                                                         tmp_path):
    from woof.da.enprod import MANIFEST_FILENAME

    result = subprocess.run(
        [str(_ENSEMBLE), "--store-root", str(tmp_path / "store"),
         "--out-dir", str(tmp_path / "png"), "--manifest",
         str(ensemble_root / MANIFEST_FILENAME), "--field", "not_a_field"],
        capture_output=True, text=True, errors="replace", timeout=120)
    assert result.returncode == 2, result.stdout
    assert "not_a_field" in result.stderr
    assert "--list-fields" in result.stderr


@pytest.fixture(scope="module")
def radar_grid(tmp_path_factory) -> Path:
    """One real ``gpuwm-obs.radar-grid`` file from the real writer.

    It goes through ``merge_contributions``, which always attaches reach
    windows, so this is a v2 file -- which is the point: the engine is
    asked to read what the pipeline actually produces.
    """

    pytest.importorskip("netCDF4", reason="the obs writer needs netCDF4")
    from woof.obs.radar_grid import write_radar_grid
    from woof.obs.sweeps import Moment, RadarSite, RadarVolume, Sweep
    from woof.obs.superob import (SuperobParams, merge_contributions,
                                   superob_volume)
    from woof.obs.target_grid import TargetGrid
    from woof.static.lambert import LambertGrid

    projection = LambertGrid(
        ref_lat=35.3331, ref_lon=-97.2778, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.2778, dx=3000.0, dy=3000.0, e_we=25, e_sn=21)
    grid = TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, 10000.0, 9), name="lanes-test")
    gates = 40
    azimuths = np.arange(0.0, 360.0, 4.0, dtype=np.float32)
    sweep = Sweep(
        sweep_index=0, elevation_number=1, elevation_angle_deg=0.5,
        nyquist_velocity_ms=32.0, start_status=3, end_status=2,
        cut_sector=0, complete=True, azimuth_deg=azimuths,
        elevation_deg=np.full(azimuths.size, 0.5, dtype=np.float32),
        moments={
            "REF": Moment("REF", "dBZ", gates, 2125.0, 250.0, np.tile(
                np.linspace(10.0, 55.0, gates, dtype=np.float32)[None, :],
                (azimuths.size, 1))),
            "VEL": Moment("VEL", "m/s", gates, 2125.0, 250.0, np.tile(
                np.linspace(-20.0, 20.0, gates, dtype=np.float32)[None, :],
                (azimuths.size, 1))),
        })
    centre_j, centre_i = grid.ny // 2, grid.nx // 2
    volume = RadarVolume(
        site=RadarSite(id="KTLX", name="test",
                       lat_deg=float(grid.lat[centre_j, centre_i]),
                       lon_deg=float(grid.lon[centre_j, centre_i]),
                       alt_m=380.0, source="test"),
        valid_time="2026-07-28T20:03:16Z", station_id="KTLX",
        volume_file="KTLX20260728_200316_V06", volume_sha256="0" * 64,
        volume_bytes=1, pack_path=Path("test.rdrpack"), pack_sha256="1" * 64,
        params={"moments": ["REF", "VEL"], "max_range_km": 250.0},
        framing={"magic": "AR2V0006", "block_count": 1}, sweeps=(sweep,))
    params = SuperobParams()
    observations = merge_contributions(
        [superob_volume(volume, grid, params=params)], grid, params=params,
        z_reduce="max")
    path = tmp_path_factory.mktemp("obs") / "radar-grid.nc"
    write_radar_grid(path, observations, grid,
                     valid_time="2026-07-28T20:03:16Z", params=params,
                     overwrite=True)
    return path


@needs_obsgrid
def test_the_obs_engine_reads_the_schema_natively(radar_grid, tmp_path):
    """No fake wrfout in the middle.

    Wrapping observations in a wrfout would make the renderer's
    ``TitleProvenance::LocalImport`` state that a model produced them,
    which is a lie on every panel.  This asserts the file is read as what
    it is: four gridded products and the site roster, off one NetCDF.
    """

    out = tmp_path / "png"
    written, failures, skipped, sites = rustwx_lanes.run_obsgrid_renderer(
        _OBSGRID, radar_grid, out_dir=out, products="all", sites=True,
        rings_km="50,100", width=640, height=480)
    assert failures == [], failures
    assert len(written) == 4, (written, skipped)
    for path in written:
        assert path.is_file() and path.stat().st_size > 5_000, path
        assert _png_size(path) == (640, 480), path
    # The site identifier is the file's own, resolved and position-checked
    # against `radar_lat`/`radar_lon` -- not a positional placeholder.
    assert sites == [{"index": 0, "id": "KTLX",
                      "lat": pytest.approx(35.34, abs=0.05),
                      "lon": pytest.approx(-97.27, abs=0.05)}]


@needs_obsgrid
def test_a_file_that_declares_another_schema_is_refused_by_name(tmp_path):
    """An undeclared or foreign layout is refused, never read hopefully.

    Every variable this reader addresses is a fixed name in a fixed
    dimension order, and a file with the right names in another order has
    the right shapes and the wrong field.
    """

    netCDF4 = pytest.importorskip("netCDF4")

    path = tmp_path / "not-a-radar-grid.nc"
    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.setncattr("schema", "some-other-thing.v9")
        dataset.createDimension("south_north", 2)
        dataset.createDimension("west_east", 2)
    result = subprocess.run(
        [str(_OBSGRID), "--obs", str(path), "--out-dir", str(tmp_path / "o")],
        capture_output=True, text=True, errors="replace", timeout=120)
    assert result.returncode != 0
    assert "gpuwm-obs.radar-grid.v1" in result.stderr
    assert "some-other-thing.v9" in result.stderr


# ---------------------------------------------------------------------------
# The overlay/annotation seam on the EXISTING renderer
# ---------------------------------------------------------------------------

def _fixture_wrfout(path: Path) -> Path:
    from woof.io.wrfout import WrfoutWriter, wrf_global_attrs

    rng = np.random.default_rng(11)
    lat = np.tile(np.linspace(38.0, 40.0, _NY)[:, None], (1, _NX))
    lon = np.tile(np.linspace(-98.0, -95.0, _NX)[None, :], (_NY, 1))
    grid = SimpleNamespace(truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
                           ref_lat=39.0, ref_lon=-96.5)
    attrs = wrf_global_attrs(
        grid, datetime.datetime(2026, 7, 28, 20), grid_id=2, parent_id=1,
        i_parent_start=5, j_parent_start=5, parent_grid_ratio=3, dt=6.0)
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0, dy=1000.0,
                      global_attrs=attrs) as writer:
        writer.write_frame(_STAMP, {
            "T": np.zeros((_NZ, _NY, _NX), np.float32),
            "MU": np.zeros((_NY, _NX), np.float32),
            "REFL_10CM": rng.uniform(-20.0, 65.0,
                                     (_NZ, _NY, _NX)).astype(np.float32),
            "T2": rng.uniform(280.0, 300.0, (_NY, _NX)).astype(np.float32),
            "XLAT": lat.astype(np.float32),
            "XLONG": lon.astype(np.float32),
            "HGT": np.zeros((_NY, _NX), np.float32),
            "SINALPHA": np.zeros((_NY, _NX), np.float32),
            "COSALPHA": np.ones((_NY, _NX), np.float32),
        })
    return path


@pytest.mark.skipif(_built("rw_wrfbatch") is None, reason=_BUILD_HINT)
def test_overlays_change_the_pixels_and_their_absence_does_not(tmp_path):
    """Both directions, which is what makes this a gate and not a demo.

    * WITH ``--overlays``, the PNG differs -- the flag is not accepted and
      ignored, which is exactly how an opt-in overlay would silently do
      nothing;
    * WITHOUT it, the PNG is byte-identical to one drawn with no flag at
      all, which is the standing pixel-identity claim for every product
      ``rw_wrfbatch`` already draws.
    """

    import hashlib
    import json

    renderer = _built("rw_wrfbatch")
    wrfout = _fixture_wrfout(tmp_path / "wrfout_d02_2026-07-28_20-00-00.nc")
    overlays = tmp_path / "overlays.json"
    overlays.write_text(json.dumps({
        "lines": [{"points": [[38.3, -97.6], [38.3, -95.6],
                              [39.7, -95.6], [39.7, -97.6]],
                   "closed": True, "color": "#c02020", "width": 3}],
        "points": [{"lat": 39.2, "lon": -96.9, "shape": "circle",
                    "radius_px": 8}],
        "rings": [{"lat": 39.0, "lon": -96.5, "radii_km": [40]}],
    }), encoding="utf-8")

    def render(tag: str, extra: list[str]) -> Path:
        out = tmp_path / f"out_{tag}"
        command = [
            str(renderer), "--store-root", str(tmp_path / f"store_{tag}"),
            "--out-dir", str(out), "--products", "composite_reflectivity",
            "--frames", "0", "--width", "500", "--height", "400",
            "--source-label", "WOOF", *extra, str(wrfout)]
        result = subprocess.run(command, capture_output=True, text=True,
                                errors="replace", timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr
        pngs = sorted(out.rglob("*.png"))
        assert len(pngs) == 1, pngs
        return pngs[0]

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    plain = digest(render("plain", []))
    again = digest(render("again", []))
    marked = digest(render("marked", ["--overlays", str(overlays)]))

    assert plain == again, (
        "two flagless renders of one file differ; the pixel-identity "
        "claim is not even reproducible")
    assert marked != plain, (
        "--overlays was accepted and drew nothing, which is the failure "
        "mode an opt-in overlay flag has")


@pytest.mark.skipif(_built("rw_wrfbatch") is None, reason=_BUILD_HINT)
def test_the_driver_forwards_the_overlay_flags(monkeypatch, tmp_path):
    """``woof.rustwx.run_renderer`` reaches the new surface."""

    from woof import rustwx

    seen: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(rustwx.subprocess, "run",
                        lambda command, **kwargs: (
                            seen.append([str(part) for part in command])
                            or Result()))
    rustwx.run_renderer(
        tmp_path / "rw_wrfbatch", tmp_path / "wrfout_d02_x.nc",
        store_root=tmp_path / "store", out_dir=tmp_path / "png",
        products="all", frames="all", width=800, height=600,
        overlays=tmp_path / "overlays.json",
        annotate=tmp_path / "annotate.json")
    command = seen[-1]
    assert command[command.index("--overlays") + 1] == str(
        tmp_path / "overlays.json")
    assert command[command.index("--annotate") + 1] == str(
        tmp_path / "annotate.json")

    # ...and omits them entirely when they are not asked for, because a
    # flag that is always present is a flag that always runs its code.
    seen.clear()
    rustwx.run_renderer(
        tmp_path / "rw_wrfbatch", tmp_path / "wrfout_d02_x.nc",
        store_root=tmp_path / "store", out_dir=tmp_path / "png",
        products="all", frames="all", width=800, height=600)
    assert "--overlays" not in seen[-1]
    assert "--annotate" not in seen[-1]


# ---------------------------------------------------------------------------
# P0: the two npz lanes' sibling wrfouts
# ---------------------------------------------------------------------------

def test_a_snapshot_without_coordinates_is_refused_by_name(tmp_path):
    """No guessing a grid from an array shape."""

    from woof.io.surface_wrfout import (SurfaceSnapshotRefusal,
                                         write_surface_wrfout)

    with pytest.raises(SurfaceSnapshotRefusal, match="no latitude/longitude"):
        write_surface_wrfout(tmp_path / "wrfout_x.nc",
                             {"T2": np.zeros((4, 5), np.float32)},
                             time_str=_STAMP, dx=1000.0)


def test_a_field_on_a_different_grid_than_its_coordinates_is_refused(
        tmp_path):
    from woof.io.surface_wrfout import (SurfaceSnapshotRefusal,
                                         write_surface_wrfout)

    snapshot = {
        "XLAT": np.zeros((4, 5), np.float32),
        "XLONG": np.zeros((4, 5), np.float32),
        "T2": np.zeros((5, 4), np.float32),
    }
    with pytest.raises(SurfaceSnapshotRefusal, match=r"T2 is \(5, 4\)"):
        write_surface_wrfout(tmp_path / "wrfout_x.nc", snapshot,
                             time_str=_STAMP, dx=1000.0)


@pytest.mark.skipif(_built("rw_wrfbatch") is None, reason=_BUILD_HINT)
def test_a_surface_snapshot_wrfout_renders_through_the_real_renderer(
        tmp_path):
    """The whole point of P0, proven against the artifact.

    Two lanes published ``.npz`` and nothing else, so the production
    renderer could not draw them at all.  This writes the same kind of
    snapshot ``tilestream.bigdomain.snapshot`` produces and puts it
    through the real ``rw_wrfbatch``.
    """

    from woof.io.surface_wrfout import (SURFACE_SNAPSHOT, VERTICAL_ATTR,
                                         write_surface_wrfout)
    from woof.io.wrfout import wrf_global_attrs

    ny, nx = 24, 30
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    core = 62.0 * np.exp(-(((yy - 12) / 4.0) ** 2
                           + ((xx - 15) / 5.0) ** 2)) - 12.0
    snapshot = {
        "LAT": np.tile(np.linspace(34.4, 36.3, ny)[:, None],
                       (1, nx)).astype(np.float32),
        "LON": np.tile(np.linspace(-98.4, -96.1, nx)[None, :],
                       (ny, 1)).astype(np.float32),
        "HT": np.full((ny, nx), 350.0, np.float32),
        "T2": (288.0 + 6.0 * core / 60.0).astype(np.float32),
        "U10": (8.0 * np.sin(xx / 7.0)).astype(np.float32),
        "V10": (8.0 * np.cos(yy / 7.0)).astype(np.float32),
        "PSFC": np.full((ny, nx), 97000.0, np.float32),
        "RAINNC": np.clip(core * 0.6, 0.0, None).astype(np.float32),
        "RAINC": np.zeros((ny, nx), np.float32),
        "REFL_COMPOSITE": core.astype(np.float32),
    }
    grid = SimpleNamespace(truelat1=33.0, truelat2=37.0, stand_lon=-97.2778,
                           ref_lat=35.3331, ref_lon=-97.2778)
    attrs = wrf_global_attrs(
        grid, datetime.datetime(2026, 7, 28, 20), grid_id=1, parent_id=1,
        i_parent_start=1, j_parent_start=1, parent_grid_ratio=1, dt=6.0)
    path = write_surface_wrfout(
        tmp_path / "wrfout_d01_bigdom_f000.nc", snapshot, time_str=_STAMP,
        dx=2000.0, global_attrs=attrs, grid_id=1)

    # The file says what its vertical axis is, so a reader looking for a
    # sounding finds the answer in the file rather than in a wrong plot.
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(path) as dataset:
        assert getattr(dataset, VERTICAL_ATTR) == SURFACE_SNAPSHOT
        assert dataset.dimensions["bottom_top"].size == 1

    out = tmp_path / "png"
    result = subprocess.run(
        [str(_built("rw_wrfbatch")), "--store-root", str(tmp_path / "store"),
         "--out-dir", str(out), "--products", "all", "--frames", "all",
         "--width", "500", "--height", "400", str(path)],
        capture_output=True, text=True, errors="replace", timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    produced = sorted(p.name for p in out.rglob("*.png"))
    assert any("composite_reflectivity" in name for name in produced), produced
    assert any("2m_temperature" in name for name in produced), produced
    assert any("10m_wind_speed" in name for name in produced), produced


# ---------------------------------------------------------------------------
# The run origin: what the CALLER's own invocation writes
# ---------------------------------------------------------------------------
#
# The test above supplies ``wrf_global_attrs``, so it never asked what the
# two shipping callers actually pass.  ``tilestream/run_bigdomain.py``
# passed no global attributes at all, and the frames it wrote were refused
# by the very renderer this writer exists to feed:
#
#     wrfout_bigdom_256_f020.nc has no sound WRF run origin:
#     START_DATE unavailable; SIMULATION_START_DATE unavailable;
#     XTIME fallback failed: XTIME is unavailable
#
# MEASURED on a real 256x256x33 tile-streamed forecast, 2026-08-17.

def test_a_snapshot_with_no_run_origin_is_refused_before_it_is_written(
        tmp_path):
    """A file the renderer will refuse is not written and called a success.

    Concrete breakage: ``rw_wrfbatch`` computes every product's lead time
    from ``START_DATE``/``SIMULATION_START_DATE`` and falls back to
    ``XTIME``.  A surface snapshot carries none of the three, so without
    an origin the writer produces a file that renders NOTHING -- which is
    what the tile-streamed lane shipped.
    """

    from woof.io.surface_wrfout import (SurfaceSnapshotRefusal,
                                         write_surface_wrfout)

    snapshot = {
        "XLAT": np.zeros((4, 5), np.float32),
        "XLONG": np.zeros((4, 5), np.float32),
    }
    target = tmp_path / "wrfout_x.nc"
    with pytest.raises(SurfaceSnapshotRefusal, match="run origin"):
        write_surface_wrfout(target, snapshot, time_str=_STAMP, dx=1000.0)
    assert not target.exists(), "the refused file must not be left on disk"


def test_a_start_time_becomes_the_run_origin_the_renderer_reads(tmp_path):
    from woof.io.surface_wrfout import write_surface_wrfout

    netCDF4 = pytest.importorskip("netCDF4")
    snapshot = {
        "XLAT": np.zeros((4, 5), np.float32),
        "XLONG": np.zeros((4, 5), np.float32),
    }
    path = write_surface_wrfout(
        tmp_path / "wrfout_x.nc", snapshot, time_str="1970-01-01_00:20:00",
        dx=1000.0, start_time=datetime.datetime(1970, 1, 1))
    with netCDF4.Dataset(path) as dataset:
        assert dataset.START_DATE == "1970-01-01_00:00:00"
        assert dataset.SIMULATION_START_DATE == "1970-01-01_00:00:00"


@pytest.mark.skipif(_built("rw_wrfbatch") is None, reason=_BUILD_HINT)
def test_the_tile_streamed_lane_s_own_call_renders(tmp_path):
    """The CALLER's invocation, not a fixture's.

    ``tilestream.run_bigdomain._write_snapshot_wrfout`` is driven exactly
    as ``stage_forecast`` drives it, and the frame it writes goes through
    the real ``rw_wrfbatch``.  Before this test the lane's own frames
    carried ``MAP_PROJ=0`` and no run origin and drew zero products.
    """

    pytest.importorskip("netCDF4")
    from woof.static.lambert import LambertGrid
    from tilestream.run_bigdomain import _write_snapshot_wrfout

    ny, nx = 24, 30
    grid = LambertGrid(e_we=nx + 1, e_sn=ny + 1, dx=3000.0, dy=3000.0,
                       ref_lat=35.3331, ref_lon=-97.2778, truelat1=33.0,
                       truelat2=37.0, stand_lon=-97.2778)
    lat, lon = grid.latlon_mass()
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    core = 62.0 * np.exp(-(((yy - 12) / 4.0) ** 2
                           + ((xx - 15) / 5.0) ** 2)) - 12.0
    snapshot = {
        "LAT": np.asarray(lat, np.float32),
        "LON": np.asarray(lon, np.float32),
        "HT": np.full((ny, nx), 350.0, np.float32),
        "T2": (288.0 + 6.0 * core / 60.0).astype(np.float32),
        "U10": (8.0 * np.sin(xx / 7.0)).astype(np.float32),
        "V10": (8.0 * np.cos(yy / 7.0)).astype(np.float32),
        "PSFC": np.full((ny, nx), 97000.0, np.float32),
        "RAINNC": np.clip(core * 0.6, 0.0, None).astype(np.float32),
        "RAINC": np.zeros((ny, nx), np.float32),
        "REFL_COMPOSITE": core.astype(np.float32),
        "elapsed_s": 1200.0,
    }
    cfg = SimpleNamespace(dx=3000.0, dy=3000.0, dt=15.0)
    written = _write_snapshot_wrfout(tmp_path / "bigdom_30_f020.npz",
                                     snapshot, cfg, "f020", grid=grid)
    assert written is not None and written.is_file()

    out = tmp_path / "png"
    result = subprocess.run(
        [str(_built("rw_wrfbatch")), "--store-root", str(tmp_path / "store"),
         "--out-dir", str(out), "--products", "all", "--frames", "all",
         "--width", "500", "--height", "400", str(written)],
        capture_output=True, text=True, errors="replace", timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    produced = sorted(p.name for p in out.rglob("*.png"))
    assert any("composite_reflectivity" in name for name in produced), produced


@pytest.mark.skipif(_built("rw_wrfbatch") is None, reason=_BUILD_HINT)
def test_the_da_cycle_lane_s_own_call_renders(tmp_path):
    """The other npz lane's invocation, on the same terms.

    ``tools/da_cycle_prepared._write_composite_wrfout`` writes the
    sibling wrfout for a DA leg's composite.  It supplies
    ``wrf_global_attrs``, so unlike the tile-streamed lane it always
    carried a run origin -- this pins that, because the two lanes share a
    writer and a change made for one must not silently take the origin
    away from the other.
    """

    pytest.importorskip("netCDF4")
    from woof.static.lambert import LambertGrid
    from tools.da_cycle_prepared import _write_composite_wrfout

    ny, nx = 20, 26
    grid = LambertGrid(e_we=nx + 1, e_sn=ny + 1, dx=3000.0, dy=3000.0,
                       ref_lat=39.79, ref_lon=-104.55, truelat1=30.0,
                       truelat2=60.0, stand_lon=-104.55)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    composite = (58.0 * np.exp(-(((yy - 10) / 3.5) ** 2
                                 + ((xx - 13) / 4.5) ** 2)) - 10.0)
    cfg = SimpleNamespace(dx=3000.0, dy=3000.0, dt=6.0)
    experiment = SimpleNamespace(
        start_time=datetime.datetime(2021, 12, 30, 17, 0, 0))
    written = _write_composite_wrfout(
        tmp_path / "leg01_composite.npz", composite, grid, cfg, 900.0,
        experiment, label="leg01")
    assert written is not None and written.is_file()

    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(written) as dataset:
        assert dataset.START_DATE == "2021-12-30_17:00:00"

    out = tmp_path / "png"
    result = subprocess.run(
        [str(_built("rw_wrfbatch")), "--store-root", str(tmp_path / "store"),
         "--out-dir", str(out), "--products", "all", "--frames", "all",
         "--width", "500", "--height", "400", str(written)],
        capture_output=True, text=True, errors="replace", timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    produced = sorted(p.name for p in out.rglob("*.png"))
    assert any("composite_reflectivity" in name for name in produced), produced


@pytest.mark.skipif(_built("rw_wrfbatch") is None, reason=_BUILD_HINT)
def test_the_streamed_lane_s_updraft_reaches_a_production_panel(tmp_path):
    """The column extremes draw, under WRF's own names for them.

    ``tilestream/bigdomain_render.py``'s ``wmax`` panel was the last
    weather field of that lane with no production renderer: a surface
    snapshot has no profile, so the column maximum updraft is the only
    way the storm's engine reaches a picture at all.

    The snapshot carries ``WMAX``/``WMIN``; the writer publishes them as
    ``W_UP_MAX``/``W_DN_MAX`` -- the Registry's names, so a reader who
    knows wrfout knows them without being told -- and the importer's
    raw-extras TABLE is what puts them in the store.  That table already
    carries ``WSPD10MAX`` and ``UP_HELI_MAX``, the same family of
    column/period maxima, so this is a row, not a code path.

    The divergence from WRF is the averaging WINDOW, and the file states
    it rather than leaving a reader to assume: WRF's are running maxima
    between history writes; these are the frame's own instant.
    """

    netCDF4 = pytest.importorskip("netCDF4")
    from woof.io.surface_wrfout import W_EXTREME_ATTR
    from woof.static.lambert import LambertGrid
    from tilestream.run_bigdomain import _write_snapshot_wrfout

    ny, nx = 24, 30
    grid = LambertGrid(e_we=nx + 1, e_sn=ny + 1, dx=3000.0, dy=3000.0,
                       ref_lat=35.3331, ref_lon=-97.2778, truelat1=33.0,
                       truelat2=37.0, stand_lon=-97.2778)
    lat, lon = grid.latlon_mass()
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    bell = np.exp(-(((yy - 12) / 4.0) ** 2 + ((xx - 15) / 5.0) ** 2))
    snapshot = {
        "LAT": np.asarray(lat, np.float32),
        "LON": np.asarray(lon, np.float32),
        "REFL_COMPOSITE": (62.0 * bell - 12.0).astype(np.float32),
        "WMAX": (24.0 * bell).astype(np.float32),
        "WMIN": (-9.0 * bell).astype(np.float32),
        "elapsed_s": 1200.0,
    }
    cfg = SimpleNamespace(dx=3000.0, dy=3000.0, dt=15.0)
    written = _write_snapshot_wrfout(tmp_path / "bigdom_30_f020.npz",
                                     snapshot, cfg, "f020", grid=grid)
    assert written is not None and written.is_file()

    with netCDF4.Dataset(written) as dataset:
        assert "W_UP_MAX" in dataset.variables
        assert "W_DN_MAX" in dataset.variables
        # The window divergence is IN the file, not only in a docstring.
        assert "INSTANTANEOUS" in getattr(dataset, W_EXTREME_ATTR)

    # The lane's own request: `all` is the named products, and the column
    # extremes are stored variables no named product draws, so they come
    # from `variables` beside it.
    from tilestream.bigdomain_render import RUST_PRODUCTS
    assert "variables" in RUST_PRODUCTS.split(",")
    out = tmp_path / "png"
    result = subprocess.run(
        [str(_built("rw_wrfbatch")), "--store-root", str(tmp_path / "store"),
         "--out-dir", str(out), "--products", RUST_PRODUCTS, "--frames", "all",
         "--width", "500", "--height", "400", str(written)],
        capture_output=True, text=True, errors="replace", timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    produced = sorted(p.name for p in out.rglob("*.png"))
    assert any("w_up_max" in name for name in produced), produced
    assert any("w_dn_max" in name for name in produced), produced


# ---------------------------------------------------------------------------
# The schema, not a table of names: what a snapshot carries is what the
# wrfout gets
# ---------------------------------------------------------------------------
#
# The writer used to publish a curated list of surface names and drop
# everything else without a word.  Nothing needed the curation --
# ``WrfoutWriter`` types an unknown ``(ny, nx)`` field as ``f4`` on its
# own and the render door draws any 2-D plane a wrfout carries -- so the
# list was doing only one thing: losing fields.  It drifted in both
# directions, carrying a ``Q2`` row neither lane produces while
# ``tilestream.bigdomain``'s ``COSZEN`` carrier was dropped from every
# tile-streamed frame ever written.
#
# These two are CPU-only on purpose: the loss happens in the writer, so
# it is provable without a renderer binary.

def test_every_2d_snapshot_field_reaches_the_wrfout(tmp_path, monkeypatch):
    """A 2-D grid-shaped carrier is published under its own name.

    ``COSZEN`` is the real case: a ``tilestream.bigdomain`` carrier with
    no row in the writer's old table and no Registry metadata either.  It
    has to reach the file on its SHAPE, or adding a field to a lane means
    editing a table in another package.

    And what does not reach the file is named rather than dropped: the
    ``*_units`` siblings and the bookkeeping scalars are not fields and
    are not reported, but a 3-D array in a file that declares itself
    surface-only is a real loss and the caller is told.
    """

    netCDF4 = pytest.importorskip("netCDF4")
    from woof.io.surface_wrfout import write_surface_wrfout
    from woof.io.wrfout import REGISTRY_VAR_META, WRFOUT_WRITER_ENV

    # The tape container is not what is under test here and the default
    # Rust writer needs a built cdylib, so this asks for the documented
    # netCDF4 engine and stays CPU-only.  What IS under test -- which
    # snapshot entries reach the file -- is decided before either engine
    # sees a byte.
    monkeypatch.setenv(WRFOUT_WRITER_ENV, "python")

    # The premise, stated rather than assumed: this field is unknown to
    # the writer's metadata, so nothing but its shape can carry it.
    assert "COSZEN" not in REGISTRY_VAR_META

    ny, nx = 6, 7
    snapshot = {
        "XLAT": np.full((ny, nx), 35.0, np.float32),
        "XLONG": np.full((ny, nx), -97.0, np.float32),
        "T2": np.full((ny, nx), 288.0, np.float32),
        "COSZEN": np.linspace(0.0, 1.0, ny * nx,
                              dtype=np.float32).reshape(ny, nx),
        "COSZEN_units": "1",
        "W_PROFILE": np.zeros((3, ny, nx), np.float32),
        "elapsed_s": 1200.0,
        "nx": nx, "ny": ny, "nz": 3,
        "dx": 3000.0, "dt": 15.0,
    }
    report = write_surface_wrfout(
        tmp_path / "wrfout_d01_x.nc", snapshot, time_str=_STAMP, dx=3000.0,
        start_time=datetime.datetime(2026, 7, 28, 20))

    with netCDF4.Dataset(report) as dataset:
        assert "COSZEN" in dataset.variables, sorted(dataset.variables)
        assert np.allclose(dataset.variables["COSZEN"][0],
                           snapshot["COSZEN"])
        # A colour-bar label and the bookkeeping scalars are not fields.
        for name in ("COSZEN_units", "elapsed_s", "nx", "ny", "nz", "dt"):
            assert name not in dataset.variables, name

    assert "COSZEN" in report.passed_through, report.passed_through
    assert "T2" in report.passed_through, report.passed_through
    assert "W_PROFILE" in report.skipped, report.skipped
    assert "surface-only" in report.skipped["W_PROFILE"]
    # The loss is what the lanes print, and only the loss.
    assert "W_PROFILE" in report.skipped_report()
    for quiet in ("COSZEN_units", "elapsed_s", "dx", "XLAT"):
        assert quiet not in report.skipped, report.skipped

    # The report IS the file, so a caller that wants only the path is not
    # forced to move when the writer starts reporting more.  These are
    # exactly the operations the producing lanes do with the return value:
    # name it, size it, interpolate it into a render command.
    assert isinstance(report, Path)
    assert report == tmp_path / "wrfout_d01_x.nc"
    assert report.name == "wrfout_d01_x.nc"
    assert report.stat().st_size > 0
    assert f"{report}".endswith("wrfout_d01_x.nc")
    assert report.path == tmp_path / "wrfout_d01_x.nc"
    assert type(report.path) is type(Path("."))


def test_a_structural_placeholder_survives_a_same_named_surface_field(
        tmp_path, monkeypatch):
    """A 2-D ``T`` does not become the file's mass coordinate.

    Publishing by shape means a snapshot key reaches the file under its
    own name, and ``T`` is the name where that would collide with
    STRUCTURE rather than with a placeholder: the renderer reads ``T`` as
    a ``(1, ny, nx)`` mass coordinate, and a 2-D surface plane put there
    would be read as a profile it is not.  So it is placed before the
    shape rule and a snapshot entry of the same name is reported as
    already written rather than silently winning.

    ``SINALPHA``/``COSALPHA`` are the opposite case and stay after: a lane
    that knows its grid rotation overrides the identity pair.
    """

    netCDF4 = pytest.importorskip("netCDF4")
    from woof.io.surface_wrfout import write_surface_wrfout
    from woof.io.wrfout import WRFOUT_WRITER_ENV

    monkeypatch.setenv(WRFOUT_WRITER_ENV, "python")

    ny, nx = 4, 5
    snapshot = {
        "XLAT": np.full((ny, nx), 35.0, np.float32),
        "XLONG": np.full((ny, nx), -97.0, np.float32),
        "T": np.full((ny, nx), 301.0, np.float32),
        "SINALPHA": np.full((ny, nx), 0.25, np.float32),
    }
    report = write_surface_wrfout(
        tmp_path / "wrfout_d01_t.nc", snapshot, time_str=_STAMP, dx=3000.0,
        start_time=datetime.datetime(2026, 7, 28, 20))

    with netCDF4.Dataset(report) as dataset:
        # Structure held: Time, bottom_top, south_north, west_east.
        assert dataset.variables["T"].shape == (1, 1, ny, nx)
        assert np.allclose(dataset.variables["T"][:], 0.0)
        # A stand-in yielded to the lane's own measurement.
        assert np.allclose(dataset.variables["SINALPHA"][0], 0.25)

    assert "T" in report.skipped, report.skipped
    assert "already written" in report.skipped["T"]
    assert "SINALPHA" in report.passed_through, report.passed_through


def test_a_snapshot_with_no_terrain_gets_no_invented_terrain(tmp_path,
                                                            monkeypatch):
    """No zero ``HGT`` or ``MU`` stands in for a field the lane lacks.

    Fails before this fix: the writer added both as zero planes, and the
    renderer draws every plane a wrfout carries, so a frame drawn with
    the default product set published a Terrain Height map reading 0 m
    and flat-zero ``wrf_hgt``, ``wrf_terrain`` and ``wrf_mu`` pictures
    (four of the eleven pictures of a ``woof cycle`` boundary).  A lane
    that has its terrain still publishes it.
    """

    netCDF4 = pytest.importorskip("netCDF4")
    from woof.io.surface_wrfout import write_surface_wrfout
    from woof.io.wrfout import WRFOUT_WRITER_ENV

    monkeypatch.setenv(WRFOUT_WRITER_ENV, "python")

    ny, nx = 4, 5
    snapshot = {
        "XLAT": np.full((ny, nx), 35.0, np.float32),
        "XLONG": np.full((ny, nx), -97.0, np.float32),
        "REFL_COMPOSITE": np.full((ny, nx), 30.0, np.float32),
    }
    bare = write_surface_wrfout(
        tmp_path / "wrfout_d01_bare.nc", snapshot, time_str=_STAMP,
        dx=3000.0, start_time=datetime.datetime(2026, 7, 28, 20))
    measured = write_surface_wrfout(
        tmp_path / "wrfout_d01_terrain.nc",
        {**snapshot, "HT": np.full((ny, nx), 1650.0, np.float32)},
        time_str=_STAMP, dx=3000.0,
        start_time=datetime.datetime(2026, 7, 28, 20))

    with netCDF4.Dataset(bare) as dataset:
        names = set(dataset.variables)
        assert "HGT" not in names and "MU" not in names, sorted(names)
        # What the file needs to be read at all is still there.
        assert dataset.variables["T"].shape == (1, 1, ny, nx)
        assert {"REFL_10CM", "SINALPHA", "COSALPHA"} <= names
    with netCDF4.Dataset(measured) as dataset:
        assert np.allclose(dataset.variables["HGT"][0], 1650.0)


def test_the_two_tilestream_carrier_tables_reach_the_writer(tmp_path,
                                                            monkeypatch):
    """Every ``SURFACE_CARRIERS`` label survives a write, by name.

    The two tables are in different packages -- ``tilestream.bigdomain``
    decides what a frame carries, ``woof.io.surface_wrfout`` decides what
    a wrfout gets -- and nothing made them agree, so ``COSZEN`` was
    carried and dropped.  This is the join, and it is a shape rule now
    rather than a second list to keep in step.
    """

    netCDF4 = pytest.importorskip("netCDF4")
    from woof.io.surface_wrfout import write_surface_wrfout
    from woof.io.wrfout import WRFOUT_WRITER_ENV
    from tilestream.bigdomain import SURFACE_CARRIERS, snapshot as build

    monkeypatch.setenv(WRFOUT_WRITER_ENV, "python")

    ny, nx, nz = 6, 7, 4
    store = {key: np.full((ny, nx), 1.5, np.float32)
             for _, (key, _units) in SURFACE_CARRIERS.items()}
    store["state/w"] = np.zeros((nz, ny, nx), np.float32)
    geo_store = {
        "setup/ht": np.full((ny, nx), 350.0, np.float32),
        "radiation/latitude_deg": np.full((ny, nx), 35.0, np.float32),
        "radiation/longitude_deg": np.full((ny, nx), -97.0, np.float32),
    }
    cfg = SimpleNamespace(nx=nx, ny=ny, nz=nz, dx=3000.0, dt=15.0)
    snap = build(store, geo_store, cfg, elapsed_s=1200.0, refl=False)

    report = write_surface_wrfout(
        tmp_path / "wrfout_d01_bigdom.nc", snap, time_str=_STAMP,
        dx=float(snap["dx"]), grid_id=1,
        start_time=datetime.datetime(1970, 1, 1))

    with netCDF4.Dataset(report) as dataset:
        missing = sorted(label for label in SURFACE_CARRIERS
                         if label not in dataset.variables)
        assert not missing, missing
    assert not report.skipped, report.skipped
