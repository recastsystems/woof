"""The renderer's map assets must arrive with the renderer that reads them.

``rw_wrfbatch`` draws coastlines, national and state borders, lakes and
counties from the Natural Earth + US Census shapefiles in
``tools/rustwx/assets/basemap``.  Those shapefiles reached a pip install
by no mechanism at all: the wheel declared ``tools =
["prepare_hrrr_*.sh"]`` and nothing else, and the bundle carried eight
binaries and no data.  The result was not an error -- it was a plot.  A
tropical cyclone rendered over a blank white rectangle, with nothing on
the image to say where on Earth it was, produced by a lane running the
published wheel exactly as documented.

The bug survived because the machine it was developed on hides it:
``rustwx-render`` falls back to a **cartopy** Natural Earth cache under
``$HOME/.local/share/cartopy``, and a workstation that has ever run
cartopy has one.  Every test in this file therefore either avoids the
renderer's own fallbacks entirely or redirects ``HOME``/``USERPROFILE``
away from them, and the end-to-end proof runs against an installed
wheel with the repository nowhere on the path.

Why the bundle and not the wheel
--------------------------------
Measured, not assumed: the wheel is 74.6 MiB compressed against PyPI's
100 MB per-file cap, and the asset tree deflates to 20.2 MiB.  Shipping
the shapefiles in the wheel leaves roughly half a megabyte of headroom
before an upload starts being rejected, which is not a margin a release
can be run on -- ``MANIFEST.in`` already externalizes the two largest
Thompson tables for exactly this reason.  The bundle has room, and it is
where the consuming binary already is: staged under
``<dest>/assets/basemap``, the shapefiles sit on the resolution ladder
``rw_wrfbatch`` already walks (``assets/basemap`` under the first eight
ancestors of its own directory), so the binary finds them with no
environment variable set and no cooperation from the Python half.

And why the companion as well
-----------------------------
The bundle is ``woof fetch-bridges``'s, and once the platform wheels
carried the renderer themselves nothing ran that command: neither the
README's Linux steps nor the Linux package's INSTALL.md.  Every picture
of a wheel install came out with no coastlines, borders or state lines,
and the one warning went to a stderr the run captured and dropped.  The
``recast-woof-data`` companion is a hard dependency every install pulls and is
not size-bound the way the ``woof`` wheel is, so since 2.8.0 it carries
the renderer's three layer directories and ``woof.rustwx`` hands them to
the renderer; a run whose renderer still has none says so in a
``render_basemap_missing`` event.  The tests at the end of this file pin
both halves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from woof import bridge_assets, rustwx

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_TOOL = REPO_ROOT / "tools" / "build_bridge_bundle.py"
SOURCE_ASSETS = REPO_ROOT / "tools" / "rustwx" / "assets"
#: The companion's copy of the renderer's layer directories.
COMPANION_BASEMAP = (REPO_ROOT / "recast-woof-data" / "woof_data" / "data"
                     / "basemap")

#: The source revision these tests "release".  Pin verifies every
#: binary's embedded GPUWM_BRIDGE_SOURCE_REV stamp against it, so the
#: stubs embed it; these tests are about the asset half and must not
#: fail (or pass) on the staleness half.
SOURCE_REV = "5eed" * 10


def _require_source_assets() -> None:
    if not SOURCE_ASSETS.is_dir():
        pytest.skip("the asset contract needs the source tree, not an install")


def _bundle_filename(release: str, platform: str) -> str:
    """The release tool's own naming, imported rather than repeated."""

    sys.path.insert(0, str(REPO_ROOT))
    try:
        import importlib

        packer = importlib.import_module("tools.build_bridge_bundle")
        return packer.bundle_filename(release, platform)
    finally:
        sys.path.remove(str(REPO_ROOT))


def _stub_payload(artifact) -> bytes:
    """The bytes a placeholder binary must carry to be pinnable.

    Both proofs the release tool asks for, on every stub: the declared
    contract marker (what a VENDORED artifact is held to, since its
    source does not move with this checkout) and the
    ``GPUWM_BRIDGE_SOURCE_REV`` stamp (what every other artifact is held
    to).  Carrying both means these tests reach the asset question with
    the binary question genuinely answered -- and it means a stub that
    goes stale against either check reds here rather than silently
    short-circuiting the check a test downstream is actually about.
    """

    from woof import bridges

    marker = bridges.BRIDGE_ABI_MARKERS.get(artifact.name, b"")
    stamp = bridge_assets.SOURCE_REV_MARKER + SOURCE_REV.encode()
    return f"stub::{artifact.name}::".encode() + marker + b"::" + stamp


def _stub_artifacts(directory: Path, platform: str) -> Path:
    """One placeholder file per bundled artifact, named for ``platform``.

    The placeholders embed each bridge's declared contract marker, so
    staging runs its real three-way check (size, SHA-256, ABI marker)
    instead of a weakened one.  These tests are about the asset half;
    they must not reach it by making the binary half easier.
    """

    directory.mkdir(parents=True, exist_ok=True)
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        name = bridge_assets.artifact_filename(artifact, platform)
        (directory / name).write_bytes(_stub_payload(artifact))
    return directory


def _pack(tmp_path: Path, *, platform: str = "linux-x86_64",
          release: str = "v0-assets") -> Path:
    """Pack a real bundle with the real release tool."""

    search = _stub_artifacts(tmp_path / "artifacts", platform)
    out = tmp_path / "bundles"
    result = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pack", "--release", release,
         "--platform", platform, "--search", str(search), "--out", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    return out / _bundle_filename(release, platform)


# ---------------------------------------------------------------------------
# The declaration cannot go stale
# ---------------------------------------------------------------------------

def test_the_bundle_carries_every_map_asset_file_in_the_tree(packed_basemap_bundle):
    """Walked from the tree, never enumerated.

    This is the gate the wheel's ``tools = ["prepare_hrrr_*.sh"]`` line
    did not have.  A new shapefile -- a new resolution, a new layer, a
    replacement of the whole ``basemap`` directory -- is carried because
    the packer walks what is there, and this test fails the moment a
    file on disk is not in the archive.
    """

    archive, _pins = packed_basemap_bundle
    import zipfile

    with zipfile.ZipFile(archive) as zf:
        members = set(zf.namelist())

    on_disk = set()
    for subdir in bridge_assets.REQUIRED_ASSET_SUBDIRS:
        root = SOURCE_ASSETS / subdir
        for path in root.rglob("*"):
            if path.is_file():
                on_disk.add("/".join((
                    bridge_assets.ASSET_ROOT, subdir,
                    path.relative_to(root).as_posix())))
    assert on_disk, "found no map assets on disk -- the walk is broken"
    missing = sorted(on_disk - members)
    assert not missing, (
        f"{len(missing)} map asset file(s) exist in the tree but would not "
        "reach a bundle, so an installed renderer would draw plots without "
        "them:\n  " + "\n  ".join(missing))


def test_the_shapefile_layers_the_renderer_reads_are_all_carried(packed_basemap_bundle):
    """Name the layers by hand, once, where a human will read the failure.

    The walk above proves the archive matches the tree; this proves the
    tree still holds what ``rustwx-render`` actually opens.  A layer
    deleted from the repository passes the walk (nothing on disk is
    missing from the archive) and silently stops being drawn, which is
    the same class of silent loss in a different disguise.
    """

    archive, _pins = packed_basemap_bundle
    import zipfile

    with zipfile.ZipFile(archive) as zf:
        members = set(zf.namelist())

    required = [
        "assets/basemap/natural_earth_10m/ne_10m_coastline.shp",
        "assets/basemap/natural_earth_10m/ne_10m_land.shp",
        "assets/basemap/natural_earth_10m/ne_10m_ocean.shp",
        "assets/basemap/natural_earth_10m/ne_10m_lakes.shp",
        "assets/basemap/natural_earth_10m/"
        "ne_10m_admin_0_boundary_lines_land.shp",
        "assets/basemap/natural_earth_10m/"
        "ne_10m_admin_1_states_provinces_lines.shp",
        "assets/basemap/natural_earth_110m/ne_110m_coastline.shp",
        "assets/basemap/us_counties_5m/cb_2023_us_county_5m.shp",
    ]
    absent = [name for name in required if name not in members]
    assert not absent, (
        "the renderer opens these layers and the bundle does not carry "
        "them:\n  " + "\n  ".join(absent))
    # A .shp without its .shx is an unreadable shapefile, not a partial one.
    for name in required:
        index = name[:-len(".shp")] + ".shx"
        assert index in members, f"{name} ships without its index {index}"


def test_pack_refuses_when_a_required_asset_directory_is_absent(tmp_path,
                                                               monkeypatch):
    """A missing asset tree stops the release, it does not thin the bundle."""

    _require_source_assets()
    empty = tmp_path / "empty-assets"
    (empty / "unrelated").mkdir(parents=True)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import importlib

        packer = importlib.import_module("tools.build_bridge_bundle")
        with pytest.raises(SystemExit) as refusal:
            packer.collect_assets(empty)
    finally:
        sys.path.remove(str(REPO_ROOT))
    assert "coastlines" in str(refusal.value)


def test_pin_refuses_a_bundle_that_carries_no_map_assets(tmp_path):
    """The regression gate: an asset-less bundle cannot be pinned.

    This is precisely the bundle 1.4.0 published -- eight binaries and
    nothing else -- and the release tool must now refuse to write pins
    for it rather than produce a wheel that stages geography-less plots.

    The binaries here are the SAME pinnable stubs the good bundle uses
    (:func:`_stub_payload`: contract marker plus source-rev stamp), so
    the only thing wrong with this archive is the missing ``assets/``
    tree.  That matters because pin checks the binaries first: a stub
    that carried the stamp but not the vendored artifacts' contract
    marker earned the staleness refusal instead, and the asset refusal
    this test names was never reached -- the test passed on the wrong
    sentence.  Both refusals stay live; each is asserted where it is
    the one thing wrong.
    """

    import zipfile

    platform = "linux-x86_64"
    release = "v0-assetless"
    archive = tmp_path / _bundle_filename(release, platform)
    with zipfile.ZipFile(archive, "w") as zf:
        for artifact in bridge_assets.BUNDLED_ARTIFACTS:
            zf.writestr(bridge_assets.artifact_filename(artifact, platform),
                        _stub_payload(artifact))
    result = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pin", "--release", release,
         "--source-rev", SOURCE_REV,
         "--bundle", str(archive), "--out", str(tmp_path / "pins.json")],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0
    assert "no assets/ members" in (result.stdout + result.stderr)
    assert not (tmp_path / "pins.json").exists()


# ---------------------------------------------------------------------------
# Staging puts them where the renderer looks
# ---------------------------------------------------------------------------

def _pin_document(tmp_path: Path, archive: Path, release: str) -> dict:
    out = tmp_path / "pins.json"
    result = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pin", "--release", release,
         "--source-rev", SOURCE_REV,
         "--bundle", str(archive), "--out", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def packed_basemap_bundle(tmp_path_factory):
    """One read-only packed archive; staging destinations remain test-local."""
    _require_source_assets()
    directory = tmp_path_factory.mktemp("basemap-bundle")
    archive = _pack(directory)
    pins = bridge_assets.parse_pins(_pin_document(directory, archive, "v0-assets"))
    return archive, pins


def test_staging_writes_the_assets_where_the_renderer_already_looks(
        tmp_path, monkeypatch, packed_basemap_bundle):
    """The staged path must be on ``rw_wrfbatch``'s own resolution ladder.

    Not "somewhere woof can find and pass along": the binary resolves
    ``assets/basemap`` under its own ancestors, so a bundle staged into
    ``~/.woof/bridges`` is found by a renderer nobody configured.  That
    is what makes the two halves inseparable in practice rather than
    only by intention.
    """

    archive, pins = packed_basemap_bundle
    bundle = pins.platforms["linux-x86_64"]
    assert bundle.assets, "the pinned bundle declares no assets"

    dest = tmp_path / "bridges"
    bridge_assets.stage_from_bundle(archive, bundle, dest,
                                    progress=lambda _msg: None)

    staged_root = dest / bridge_assets.ASSET_ROOT / "basemap"
    assert staged_root.is_dir()
    assert (staged_root / "natural_earth_10m" / "ne_10m_coastline.shp"
            ).is_file()

    # Every pinned asset landed, byte for byte.
    for pin in bundle.assets:
        path = dest / pin.path
        assert path.is_file(), f"{pin.path} was not staged"
        assert path.stat().st_size == pin.bytes
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin.sha256

    # And the renderer's ladder resolves it, with no environment help.
    monkeypatch.delenv("RUSTWX_BASEMAP_DIR", raising=False)
    monkeypatch.delenv("RUSTWX_ASSETS_DIR", raising=False)
    renderer = dest / "rw_wrfbatch"
    assert staged_root in rustwx.basemap_candidates(renderer)
    assert rustwx.resolve_basemap_dir(renderer) == staged_root


def test_a_corrupt_asset_is_refused_rather_than_staged(tmp_path, packed_basemap_bundle):
    """Asset bytes are verified exactly as binary bytes are."""

    archive, pins = packed_basemap_bundle
    bundle = pins.platforms["linux-x86_64"]

    victim = bundle.assets[0]
    tampered = bridge_assets.BundlePin(
        platform=bundle.platform, filename=bundle.filename,
        bytes=bundle.bytes, sha256=bundle.sha256, binaries=bundle.binaries,
        assets=(bridge_assets.AssetPin(
            path=victim.path, bytes=victim.bytes,
            sha256="0" * 64),) + bundle.assets[1:])
    dest = tmp_path / "bridges"
    with pytest.raises(bridge_assets.BridgeAssetError, match="SHA-256"):
        bridge_assets.stage_from_bundle(archive, tampered, dest,
                                        progress=lambda _msg: None)
    assert not (dest / victim.path).exists()


@pytest.mark.parametrize("escape", [
    "../outside.shp",
    "/etc/passwd",
    "C:/windows/system32/evil.dll",
    "assets\\basemap\\evil.shp",
    "basemap/no-asset-root.shp",
])
def test_a_pinned_asset_path_cannot_escape_the_destination(escape):
    """The pins document decides where staging writes, so it is checked."""

    payload = {
        "schema": bridge_assets.PINS_SCHEMA, "release": "v0",
        "platforms": {"linux-x86_64": {
            "bundle": {"filename": "b.zip", "bytes": 1, "sha256": "a" * 64},
            "binaries": [{"artifact": "grib1_bridge",
                          "filename": "grib1_bridge",
                          "bytes": 1, "sha256": "b" * 64}],
            "assets": [{"path": escape, "bytes": 1, "sha256": "c" * 64}]}},
    }
    with pytest.raises(bridge_assets.BridgeAssetError, match="asset path"):
        bridge_assets.parse_pins(payload)


def test_a_bundle_missing_a_pinned_asset_is_refused(tmp_path, packed_basemap_bundle):
    """A half-built bundle must not stage its binaries and shrug."""

    archive, pins = packed_basemap_bundle
    bundle = pins.platforms["linux-x86_64"]
    invented = bridge_assets.BundlePin(
        platform=bundle.platform, filename=bundle.filename,
        bytes=bundle.bytes, sha256=bundle.sha256, binaries=bundle.binaries,
        assets=bundle.assets + (bridge_assets.AssetPin(
            path="assets/basemap/never_packed.shp", bytes=1,
            sha256="d" * 64),))
    with pytest.raises(bridge_assets.BridgeAssetError,
                       match="never_packed.shp"):
        bridge_assets.stage_from_bundle(archive, invented,
                                        tmp_path / "bridges",
                                        progress=lambda _msg: None)


def test_current_binaries_with_missing_assets_are_reported_not_skipped(
        tmp_path, monkeypatch, capsys, packed_basemap_bundle):
    """The upgrade path out of 1.4.0, in the words a user needs.

    Someone who already ran ``fetch-bridges`` has eight current binaries
    and no assets.  ``nothing to fetch`` would leave them rendering
    blank maps forever, so the asset gap alone must drive a re-stage and
    say why.
    """

    archive, pins = packed_basemap_bundle
    bundle = pins.platforms["linux-x86_64"]

    dest = tmp_path / "bridges"
    bridge_assets.stage_from_bundle(archive, bundle, dest,
                                    progress=lambda _msg: None)
    # Exactly the 1.4.0 end state: binaries staged, assets absent.
    import shutil

    shutil.rmtree(dest / bridge_assets.ASSET_ROOT)
    staged, stale, absent = bridge_assets.classify_destination(dest, bundle)
    assert not stale and not absent, "the binaries should still be current"
    held, _stale_assets, absent_assets = bridge_assets.classify_assets(
        dest, bundle)
    assert not held and absent_assets

    monkeypatch.setattr(bridge_assets, "load_pins", lambda path=None: pins)
    monkeypatch.setattr(bridge_assets, "host_platform",
                        lambda: "linux-x86_64")
    import argparse

    args = argparse.Namespace(from_dir=str(archive.parent), dest=str(dest),
                              keep_bundle=False, list=False)
    assert bridge_assets.fetch_bridges_main(args) == 0
    out = capsys.readouterr().out
    assert "map asset" in out
    assert "no coastlines" in out
    for pin in bundle.assets:
        assert bridge_assets.matches_pin(dest / pin.path, pin)


# ---------------------------------------------------------------------------
# The silent state must stop being silent
# ---------------------------------------------------------------------------

def test_a_renderer_with_no_basemaps_warns_before_it_draws(tmp_path,
                                                           monkeypatch):
    """The other way to ship a plot believing it is something else.

    Delivery is fixed, but an install that staged its binaries under
    1.4.0 and never re-ran ``fetch-bridges`` still has eight executables
    and no assets.  That state renders successfully, exits zero, and
    produces blank geography -- so it must say so.
    """

    from woof import render

    monkeypatch.delenv("RUSTWX_BASEMAP_DIR", raising=False)
    monkeypatch.delenv("RUSTWX_ASSETS_DIR", raising=False)
    # A renderer with nothing on its ladder: no assets/basemap under any
    # ancestor of the binary, and a working directory equally bare.
    bridges = tmp_path / "deep" / "bridges"
    bridges.mkdir(parents=True)
    renderer = bridges / "rw_wrfbatch"
    renderer.write_bytes(b"stub")
    workdir = tmp_path / "deep" / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    # The checkout fallback (basemap_dir) is the last rung; point it away.
    monkeypatch.setattr(rustwx, "basemap_dir", lambda: tmp_path / "nowhere")
    # The companion carries them on every install since 2.8.0, so the
    # silent state is now an install whose companion lost them.
    monkeypatch.setattr(rustwx, "companion_basemap_dir", lambda: None)
    # ... and away from the cartopy cache this workstation may well have,
    # which the renderer would otherwise draw from.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))

    notice = render.missing_basemap_notice(renderer)
    assert notice is not None
    assert "no coastlines" in notice
    # Actionable: one sentence, one remedy, no multi-line bootstrap.
    assert notice.count("\n") == 0
    # The remedy is the package that carries them, reinstalled in place.
    assert "pip install --force-reinstall recast-woof-data" in notice

    # And silence once the assets are where the renderer looks.
    staged = bridges / "assets" / "basemap"
    staged.mkdir(parents=True)
    assert render.missing_basemap_notice(renderer) is None
    # A renderer that does not exist at all is the fallback notice's
    # business, not this one's.
    assert render.missing_basemap_notice(None) is None


def test_platform_renderer_receives_separately_staged_basemaps(tmp_path, monkeypatch):
    """A wheel's libexec binary must use assets fetched into the user's profile."""
    from woof import render

    for name in ("RUSTWX_BASEMAP_DIR", "RUSTWX_ASSETS_DIR"):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "profile with spaces"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(home)
    monkeypatch.setattr(rustwx, "basemap_dir", lambda: tmp_path / "no-checkout-assets")
    # An install whose companion carries no map assets (one from before
    # 2.8.0, or edited): the copy fetch-bridges staged is still used.
    monkeypatch.setattr(rustwx, "companion_basemap_dir", lambda: None)
    staged = home / ".woof" / "bridges" / "assets" / "basemap"
    staged.mkdir(parents=True)
    packaged = tmp_path / "venv" / "site-packages" / "woof" / "libexec" / "bridges" / "rw_wrfbatch"
    packaged.parent.mkdir(parents=True)
    packaged.write_bytes(b"unexecuted-artifact-location")
    assert staged not in rustwx.basemap_candidates(packaged)
    assert rustwx.resolve_basemap_dir(packaged) == staged
    assert render.missing_basemap_notice(packaged) is None
    # Observe the environment in a real child process, including paths with
    # spaces, rather than only asserting the dictionary constructed here.
    child = subprocess.run([sys.executable, "-I", "-c",
                            "import os; print(os.environ['RUSTWX_BASEMAP_DIR'])"],
                           env=rustwx.renderer_env(), capture_output=True, text=True, check=True)
    assert child.stdout.strip() == str(staged)


@pytest.mark.parametrize("override", ["RUSTWX_BASEMAP_DIR", "RUSTWX_ASSETS_DIR"])
def test_staged_basemaps_do_not_override_explicit_asset_configuration(tmp_path, monkeypatch, override):
    for name in ("RUSTWX_BASEMAP_DIR", "RUSTWX_ASSETS_DIR"):
        monkeypatch.delenv(name, raising=False)
    staged = tmp_path / "staged" / "assets" / "basemap"
    staged.mkdir(parents=True)
    monkeypatch.setattr(rustwx, "default_bridge_dir", lambda: tmp_path / "staged")
    monkeypatch.setattr(rustwx, "basemap_dir", lambda: tmp_path / "no-checkout-assets")
    monkeypatch.setenv(override, str(tmp_path / "explicit map assets"))
    env = rustwx.renderer_env()
    assert env[override] == str(tmp_path / "explicit map assets")
    assert env.get("RUSTWX_BASEMAP_DIR") != str(staged)


# ---------------------------------------------------------------------------
# Discoverability: the catalog without a file
# ---------------------------------------------------------------------------

def test_the_product_catalog_is_listable_without_a_wrfout(capsys, monkeypatch):
    """"What may I put in --products?" is a question about the build.

    It used to be answerable only by reading the source or by already
    having a wrfout: ``woof render --list-products`` alone refused with
    "at least one WRFOUT file is required".  A forecaster asking which
    products exist is precisely someone who has not run anything yet.
    """

    from woof import render

    args = argparse.Namespace(
        wrfout=[], list_products=True, pair=None, engine="matplotlib",
        products="all", timeidx="all", out=Path("out"), dpi=150,
        size="1200x900", heavy=False, source_label="WOOF-ArWen", explain=False)
    assert render.render_main(args) == 0
    out = capsys.readouterr().out
    assert "product catalog" in out
    # The matplotlib engine's own products, from its own declaration.
    for product in render.PRODUCTS:
        assert product in out


def test_the_rust_catalog_comes_from_the_renderer_not_a_copy(monkeypatch):
    """A second copy of the catalog in Python is drift waiting to happen.

    The rust catalog belongs to the renderer, which already answers
    ``--list-products`` with no inputs.  This asserts the Python half
    asks it rather than keeping a list -- the same discipline the
    ``--products`` parser already follows by passing unknown slugs
    straight through for the renderer's strict validation.
    """

    import inspect

    from woof import render

    source = inspect.getsource(render.catalog_main)
    assert "--list-products" in source
    assert "find_renderer" in source
    # No literal rust slug may appear in this module's catalog path.
    for slug in ("sbcape", "srh_0_1km", "composite_reflectivity"):
        assert slug not in source


def test_the_cartopy_cache_counts_as_geography_and_silences_the_warning(
        tmp_path, monkeypatch):
    """The fallback that hid this bug for a release must not be ignored.

    ``rustwx-render`` falls back to $HOME/.local/share/cartopy, so a
    workstation that has ever run cartopy draws perfectly good
    coastlines.  Warning there would be a false alarm on every developer
    machine, and a notice that cries wolf is a notice nobody reads.
    """

    from woof import render

    monkeypatch.delenv("RUSTWX_BASEMAP_DIR", raising=False)
    monkeypatch.delenv("RUSTWX_ASSETS_DIR", raising=False)
    bridges = tmp_path / "deep" / "bridges"
    bridges.mkdir(parents=True)
    renderer = bridges / "rw_wrfbatch"
    renderer.write_bytes(b"stub")
    workdir = tmp_path / "deep" / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(rustwx, "basemap_dir", lambda: tmp_path / "nowhere")
    monkeypatch.setattr(rustwx, "companion_basemap_dir", lambda: None)

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    assert rustwx.cartopy_natural_earth_root() is None
    assert render.missing_basemap_notice(renderer) is not None

    cartopy = home / ".local" / "share" / "cartopy" / "shapefiles" / "natural_earth"
    cartopy.mkdir(parents=True)
    assert rustwx.cartopy_natural_earth_root() is not None
    # A cache is geography only for the layers it holds.  The one on the
    # 5070 Ti host held a coastline and no state lines, so its pictures
    # had no borders; an empty or partial cache still warns.
    assert render.missing_basemap_notice(renderer) is not None
    (cartopy / "physical").mkdir()
    (cartopy / "physical" / "ne_10m_coastline.shp").write_bytes(b"shp")
    assert render.missing_basemap_notice(renderer) is not None
    (cartopy / "cultural").mkdir()
    for name in ("ne_10m_admin_0_boundary_lines_land.shp",
                 "ne_50m_admin_1_states_provinces_lines.shp"):
        (cartopy / "cultural" / name).write_bytes(b"shp")
    assert render.missing_basemap_notice(renderer) is None


# ---------------------------------------------------------------------------
# A wheel install draws its maps with no step after pip install
# ---------------------------------------------------------------------------

#: The files a picture's coastline, national borders, state lines and
#: counties are drawn from, relative to a basemap root.
DRAWN_LAYERS = (
    "natural_earth_10m/ne_10m_coastline.shp",
    "natural_earth_10m/ne_10m_admin_0_boundary_lines_land.shp",
    "natural_earth_10m/ne_10m_admin_1_states_provinces_lines.shp",
    "us_counties_5m/cb_2023_us_county_5m.shp",
)


def wheel_install(tmp_path, monkeypatch) -> Path:
    """A venv-shaped install with no staged bridge estate.

    The platform wheel's renderer sits in
    ``site-packages/woof/libexec/bridges`` with no ``assets/`` beside it
    or above it; ``woof fetch-bridges`` never ran, so
    ``~/.woof/bridges`` is empty; there is no checkout to fall back to,
    no cartopy cache and no ``RUSTWX_*`` override.  Exactly the install
    the README's Linux steps leave.  Returns the renderer's path.
    """

    from woof import bridges

    for name in ("RUSTWX_BASEMAP_DIR", "RUSTWX_ASSETS_DIR",
                 rustwx.RENDERER_ENV):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    site = tmp_path / "venv" / "lib" / "python3" / "site-packages"
    renderer = (site / "woof" / "libexec" / "bridges"
                / bridges.executable_name(rustwx.RENDERER_NAME))
    renderer.parent.mkdir(parents=True)
    renderer.write_bytes(b"unexecuted-artifact-location")
    renderer.chmod(0o755)
    monkeypatch.setattr(rustwx, "renderer_candidates", lambda: (renderer,))
    monkeypatch.setattr(rustwx, "basemap_dir",
                        lambda: tmp_path / "no-checkout" / "basemap")
    monkeypatch.setattr(rustwx, "default_bridge_dir",
                        lambda: home / ".woof" / "bridges")
    return renderer


def wheel_with_companion(tmp_path, monkeypatch, *, maps: bool = True):
    """:func:`wheel_install` beside a ``recast-woof-data`` that has its maps, or lost them.

    Returns the companion's map directory, or None when ``maps`` is
    false.  For the tests of callers that start the renderer themselves
    rather than through ``woof render``.
    """

    wheel_install(tmp_path, monkeypatch)
    companion = None
    if maps:
        companion = (tmp_path / "venv" / "lib" / "python3" / "site-packages"
                     / "woof_data" / "data" / "basemap")
        companion.mkdir(parents=True)
    monkeypatch.setattr(rustwx, "companion_basemap_dir", lambda: companion)
    return companion


def test_a_wheel_install_with_no_staged_estate_draws_its_maps(tmp_path,
                                                              monkeypatch):
    """THE DEFECT: every picture of a pip install had no geography.

    Measured on the 5070 Ti host: a fresh venv, ``pip install
    'recast-woof[all-cu12]'``, ``fetch-tables``, ``doctor``, then a forecast --
    4,002 pictures of an HRRR run and 24,203 of two ERA5 runs, every one
    with no coastline, border or state line.  The renderer resolved from
    ``libexec/bridges`` with nothing beside it and ``resolve_basemap_dir``
    answered None.  The companion every install pulls carries the layers
    now, and the renderer is handed them.
    """

    renderer = wheel_install(tmp_path, monkeypatch)
    found = rustwx.find_renderer()
    assert found == renderer.resolve()
    assert not (found.parent / "assets").exists()

    resolved = rustwx.resolve_basemap_dir(found)
    assert resolved is not None, (
        "a wheel install with no staged bridge estate resolves no map "
        "assets, so every picture is drawn with no coastlines, borders or "
        "state lines")
    for layer in DRAWN_LAYERS:
        assert (resolved / layer).is_file(), f"{layer} missing from {resolved}"
        assert (resolved / layer).with_suffix(".shx").is_file(), layer

    # What the renderer is actually handed, observed in a real child.
    child = subprocess.run(
        [sys.executable, "-I", "-c",
         "import os; print(os.environ['RUSTWX_BASEMAP_DIR'])"],
        env=rustwx.renderer_env(), capture_output=True, text=True, check=True)
    assert Path(child.stdout.strip()) == resolved

    from woof import render

    assert render.missing_basemap_notice(found) is None


def test_an_explicit_override_still_outranks_the_companion(tmp_path,
                                                           monkeypatch):
    wheel_install(tmp_path, monkeypatch)
    mine = tmp_path / "my maps"
    mine.mkdir()
    monkeypatch.setenv("RUSTWX_BASEMAP_DIR", str(mine))
    assert rustwx.renderer_env()["RUSTWX_BASEMAP_DIR"] == str(mine)
    assert rustwx.resolve_basemap_dir(rustwx.find_renderer()) == mine


def test_the_companion_copy_is_the_renderers_own_byte_for_byte():
    """Two copies of one tree, held to one.

    ``tools/rustwx/assets/basemap`` is the copy of record: the renderer's
    own build and the bridge bundle read it.  The companion carries its
    layer directories so a wheel install draws maps.  A layer changed or
    added on one side only would draw different geography on a checkout
    and on an install, or none at all on an install.
    """

    _require_source_assets()
    if not COMPANION_BASEMAP.is_dir():
        pytest.skip("the companion's copy needs the source tree")
    source = SOURCE_ASSETS / "basemap"
    layers = sorted(p.name for p in source.iterdir() if p.is_dir())
    carried = sorted(p.name for p in COMPANION_BASEMAP.iterdir()
                     if p.is_dir())
    remedy = ("copy each layer directory of tools/rustwx/assets/basemap "
              "over recast-woof-data/woof_data/data/basemap, e.g. python -c "
              "\"import shutil; [shutil.copytree(f'tools/rustwx/assets/"
              "basemap/{d}', f'recast-woof-data/woof_data/data/basemap/{d}', "
              "dirs_exist_ok=True) for d in " + repr(layers) + "]\"")
    assert carried == layers, (
        f"the renderer's layers {layers} and the companion's {carried} "
        f"differ; {remedy}")
    for layer in layers:
        mine = {p.relative_to(source / layer).as_posix(): p
                for p in (source / layer).rglob("*") if p.is_file()}
        theirs = {p.relative_to(COMPANION_BASEMAP / layer).as_posix(): p
                  for p in (COMPANION_BASEMAP / layer).rglob("*")
                  if p.is_file()}
        assert sorted(mine) == sorted(theirs), (
            f"{layer}: files differ; {remedy}")
        differ = [name for name in mine
                  if mine[name].read_bytes() != theirs[name].read_bytes()]
        assert not differ, f"{layer}: {differ} differ by bytes; {remedy}"
    # Beside the layers, only the note that says where they came from.
    loose = sorted(p.name for p in COMPANION_BASEMAP.iterdir() if p.is_file())
    assert loose == ["PROVENANCE.md"]


def test_the_companion_resolver_never_raises_on_a_missing_companion(
        monkeypatch):
    from woof import data_assets

    def gone():
        raise ModuleNotFoundError("No module named 'woof_data'",
                                  name="woof_data")

    monkeypatch.setattr(data_assets, "companion_root", gone)
    assert data_assets.companion_basemap_dir() is None

    def skewed():
        raise ImportError("mismatched companion")

    monkeypatch.setattr(data_assets, "companion_root", skewed)
    assert data_assets.companion_basemap_dir() is None
    assert data_assets.companion_reinstall_command().startswith(
        "pip install --force-reinstall recast-woof-data")


# ---------------------------------------------------------------------------
# A missing basemap is never silent again
# ---------------------------------------------------------------------------

def test_a_run_is_told_once_while_drawing_and_once_at_finalize(tmp_path,
                                                               monkeypatch):
    from woof import render

    wheel_install(tmp_path, monkeypatch)
    monkeypatch.setattr(rustwx, "companion_basemap_dir", lambda: None)
    told = []

    def warn(code, message, **fields):
        told.append((code, message, fields))

    folder = tmp_path / "png"
    assert render.announce_missing_basemap(warn, folder, stage="as-drawn")
    # every later frame of the same run returns at once
    assert not render.announce_missing_basemap(warn, folder, stage="as-drawn")
    assert render.announce_missing_basemap(warn, folder, stage="finalize")
    assert not render.announce_missing_basemap(warn, folder,
                                               stage="finalize")
    assert [code for code, _, _ in told] == [render.BASEMAP_MISSING_CODE] * 2
    code, message, fields = told[0]
    assert "no coastlines, borders or state lines" in message
    assert fields["remedy"].startswith("pip install --force-reinstall "
                                       "recast-woof-data")
    assert fields["remedy"] in message
    assert fields["render_stage"] == "as-drawn"
    assert told[1][2]["render_stage"] == "finalize"

    # An install that draws its maps is told nothing.
    monkeypatch.setattr(rustwx, "companion_basemap_dir",
                        lambda: COMPANION_BASEMAP)
    quiet = []
    assert not render.announce_missing_basemap(
        lambda *a, **k: quiet.append(a), tmp_path / "other",
        stage="as-drawn")
    assert quiet == []


def test_a_warning_that_raises_never_stops_the_picture(tmp_path,
                                                       monkeypatch):
    from woof import render

    wheel_install(tmp_path, monkeypatch)
    monkeypatch.setattr(rustwx, "companion_basemap_dir", lambda: None)

    def broken(*_args, **_fields):
        raise RuntimeError("observer gone")

    assert render.announce_missing_basemap(
        broken, tmp_path / "png", stage="as-drawn") is False


def test_the_finalize_render_tells_the_run_before_it_draws(tmp_path,
                                                          monkeypatch,
                                                          capsys):
    """The end-of-run render, on the path ``woof go`` and run-plan take."""

    from woof import go_cli, render

    wheel_install(tmp_path, monkeypatch)
    monkeypatch.setattr(rustwx, "companion_basemap_dir", lambda: None)
    frames = tmp_path / "run" / "wrfout"
    frames.mkdir(parents=True)
    (frames / "wrfout_d01_2026-09-26_06_00_00").write_bytes(b"frame")
    drawn = []
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(go_cli, "_run_stage",
                        lambda label, command, **kw: drawn.append(command))

    class Observer:
        def __init__(self):
            self.warnings = []

        def warn(self, code, message, **fields):
            self.warnings.append((code, message, fields))

    observer = Observer()
    plan = {"run": tmp_path / "run", "render": tmp_path / "png",
            "wrfout_dir": frames, "render_products": None}
    assert go_cli._render_stage(plan, explain=False, observer=observer)
    assert drawn, "the render stage drew nothing"
    assert [w[0] for w in observer.warnings] == [render.BASEMAP_MISSING_CODE]
    assert observer.warnings[0][2]["render_stage"] == "finalize"

    # A stage with no event stream belongs to a terminal command: stderr.
    plan = {**plan, "render": tmp_path / "png-2"}
    assert go_cli._render_stage(plan, explain=False, observer=None)
    err = capsys.readouterr().err
    assert "render: warning: no map assets resolve" in err
    assert "pip install --force-reinstall recast-woof-data" in err


def test_every_reader_keys_on_the_engines_code():
    """The web page, a remote machine's status and the terminal workspace
    each spell the code to stay off the render stack; one code, held here."""

    from woof import remote_artifacts, render, runplan
    from woof.gui import runs

    code = render.BASEMAP_MISSING_CODE
    assert code in runplan.WARNING_CODES
    assert runs.BASEMAP_MISSING == code
    assert remote_artifacts.RENDER_BASEMAP_MISSING == code
    tui = REPO_ROOT / "tools" / "arwen-tui" / "src" / "local_progress.rs"
    if tui.is_file():
        assert (f'RENDER_BASEMAP_MISSING:&str="{code}"'
                in tui.read_text(encoding="utf-8"))


def test_the_web_page_knows_the_pictures_have_no_maps(tmp_path):
    from woof.gui import runs
    from woof.render import BASEMAP_MISSING_CODE

    run = tmp_path / "run"
    run.mkdir()
    records = [
        {"schema_version": "gpuwm.run-plan.event.v1", "sequence": 1,
         "event": "stage_started", "stage": "forecast"},
        {"schema_version": "gpuwm.run-plan.event.v1", "sequence": 2,
         "event": "warning", "code": BASEMAP_MISSING_CODE,
         "message": "no map assets resolve for the renderer",
         "remedy": "pip install --force-reinstall recast-woof-data==2.8.0"},
    ]
    events = run / runs.EVENTS
    events.write_text("".join(json.dumps(r) + "\n" for r in records[:1]),
                      encoding="utf-8")
    assert runs.status(run)["basemap_missing"] is False
    events.write_text("".join(json.dumps(r) + "\n" for r in records),
                      encoding="utf-8")
    assert runs.status(run)["basemap_missing"] is True
    copy = json.loads((REPO_ROOT / "woof" / "gui" / "copy" / "screens.json")
                      .read_text(encoding="utf-8"))
    assert "recast-woof-data" in copy["mapviewer"]["no_basemap"]
    script = (REPO_ROOT / "woof" / "gui" / "static" / "js"
              / "mapviewer.js").read_text(encoding="utf-8")
    assert "V.no_basemap" in script and "st.basemap_missing" in script
    assert f'data.code !== "{BASEMAP_MISSING_CODE}"' in script
