"""The README (the PyPI long description) and the repository front page state this release's real facts.

THE BREAKAGE (woof hex 0.4.12 on PyPI, 2026-11-05): the repository README still said 0.4.10 and
``gpuwm>=2.9.7``, the long description cited ``kernel-packs-v0.4.9`` and NVRTC 12.9/13.0/13.4 only, and its
35 relative links resolved against pypi.org, where every one of them is a 404.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from woof.hex import assets

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
FRONT = ROOT.parent / "README.md"
# (Recast WOOF) these facts are gpuwm-hex's own PyPI page and declaration; woof.hex ships inside recast-woof,
# whose README and pyproject say its own release and engine, so this tree carries no components/hex/pyproject.toml
if not (ROOT / "pyproject.toml").is_file():
    pytest.skip("gpuwm-hex's own PyPI description and declaration; woof.hex ships inside recast-woof",
                allow_module_level=True)
DECLARATION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
VERSION = DECLARATION["project"]["version"]
ENGINE = next(r for r in DECLARATION["project"]["dependencies"] if r.startswith("woof>="))


def _packs():
    pins = assets.load_pins()
    default_tag = pins["release_assets"]["tag"]
    return [(tuple(row["compiler"]["nvrtc"]), row.get("release_tag", default_tag), row)
            for row in pins["assets"].values() if row.get("kind") == "kernel-pack"]


def test_the_long_description_names_this_version_and_engine_range():
    text = README.read_text(encoding="utf-8")
    assert f"woof hex {VERSION} requires woof" in text
    assert f"(`{ENGINE}`)" in text


def test_the_long_description_names_every_pinned_pack_and_release():
    text = " ".join(README.read_text(encoding="utf-8").split())
    section = text[text.index("### Kernel packs"):text.index("### The import namespace")]
    for version, tag, _ in _packs():
        assert ".".join(map(str, version)) in section, version
        assert f"`{tag}`" in section, tag
    named = set(re.findall(r"kernel-packs-v[0-9.]+[0-9]", section))
    assert named == {tag for _, tag, _ in _packs()}, named


def test_the_long_description_has_no_relative_links():
    text = README.read_text(encoding="utf-8")
    relative = [t for t in re.findall(r"\]\(([^)\s]+)\)", text) if not re.match(r"(https?:|mailto:|#)", t)]
    assert not relative, f"PyPI resolves these against pypi.org: {relative[:5]}"
    # The repository moved to recastsystems (2026-11-10); a link left on the old owner works only through
    # GitHub's redirect, which stops if a repo of the old name is ever created again.
    pinned = set(re.findall(r"github\.com/([^/]+)/gpuwm-hex/blob/(v[^/]+)/", text))
    assert pinned == {("recastsystems", f"v{VERSION}")}, pinned


def test_the_cu13_extra_cannot_resolve_an_nvrtc_without_a_pack():
    """A pack is bound to its NVRTC library bytes, so the ceiling is the newest pack's exact version: a
    ``<13.(N+1)`` cap would let a later 13.N patch release resolve and miss the pack (0.4.12's 13.5.37)."""
    extra = DECLARATION["project"]["optional-dependencies"]["gpu-cu13"]
    caps = [re.search(r"<=(13\.\d+\.\d+)", r) for r in extra if r.startswith("nvidia-cuda-nvrtc")]
    assert caps and caps[0], f"gpu-cu13 must cap NVRTC at the newest pinned CUDA 13 pack: {extra}"
    newest = max((row for version, _, row in _packs() if version[0] == 13), key=lambda row: tuple(row["compiler"]["nvrtc"]))
    assert newest["compiler"]["nvrtc_package"] == f"nvidia-cuda-nvrtc=={caps[0].group(1)}", (newest["compiler"], extra)


def test_each_gpu_extra_pulls_the_engine_extra_of_the_same_name():
    """THE BREAKAGE (pair line, 2026-11-10): hex[gpu-cu13] required plain ``woof``, so the engine's own NVRTC cap
    (<13.5 under its gpu-cu13) did not apply and a fresh install resolved NVRTC 13.5.37, a stack the engine excludes
    and the gate never ran (hrrrv5 differed in 8,152,918 of 37,778,553 words).  Pulling ``woof[<same extra>]``
    makes pip resolve one CUDA stack for the pair."""
    extras = DECLARATION["project"]["optional-dependencies"]
    for name in ("gpu-cu12", "gpu-cu13"):
        assert f"woof[{name}]" in extras[name], (name, extras[name])


def test_the_repository_front_page_states_this_version():
    if not FRONT.is_file():
        pytest.skip("the repository front page is not part of an installed or sdist tree")
    text = FRONT.read_text(encoding="utf-8")
    assert f"woof hex {VERSION} ports the whole model" in text
    assert f"## What {VERSION} claims" in text
    assert f"(`{ENGINE}`)" in text
    assert not re.search(r"woof hex 0\.\d+\.\d+ ports", text.replace(f"woof hex {VERSION} ports", ""))


@pytest.mark.parametrize("page", ["README.md", "docs/global-door.md", "docs/manual/10-whats-new-0.4.md"])
def test_no_page_offers_the_retired_spectral_package(page):
    assert "woof global" not in (ROOT / page).read_text(encoding="utf-8")


def _figures(text):
    """``3.0e-06`` and ``3.0e-6`` alike, as numbers, in the order written."""
    return [float(token) for token in re.findall(r"\d(?:\.\d+)?e-\d+", text)]


def test_the_fast_against_exact_figures_are_the_ones_the_command_prints():
    """The README states no fast-against-exact figure for the 3 km example's default route, and the command prints none.

    THE BREAKAGE (the candidate's card job, 2026-11-14): the README and the command both said moisture 1.1e-06 and
    stored pressure 9.6e-05 where the card read 1.315e-06 and 9.778e-05. From dist48 the default route runs another
    linear model (the closure-live form), which differs from the exact route by design; the figures were measured with
    the form before it, and a page that kept them would state a difference no run of the shipped route measured."""
    from types import SimpleNamespace

    from woof.hex import ad_doors

    text = README.read_text(encoding="utf-8")
    row_name = re.search(r"woof hex sensitivity --row (\S+) --steps (\d+) ", text)
    assert row_name, "the README's 3 km example command"
    row = ad_doors.load_table()["rows"][row_name.group(1)]
    route = row["gradient_line"]["boundary_layer"]["routes"]["tape-free"]
    context = SimpleNamespace(plan_how=f"the row's example plan ({row['example_plan']})", steps=int(row_name.group(2)))
    words, measured = ad_doors.route_error_words(context, route)
    assert not measured and words == route["error"]
    start = text.index("**The default route is fast")
    stated = _figures(text[start:text.index("\n\n", start)])
    before = _figures(" ".join(f"{v:.1e}" for v in (3.0e-06, 1.4e-06, 6.5e-06, 9.8e-05)))
    assert not set(stated) & set(before), stated


def test_the_pages_name_every_test_direction_adjoint_test_runs():
    """The README and the differentiable notes name every test direction `adjoint-test` runs, the word that picks
    them, and the word it prints for a direction that does not count.

    THE BREAKAGE (2026-11-14): `adjoint-test` went from one test direction to several, with a verdict over them,
    while both pages still described one bump and "PASS or FAIL against fixed bars": a user who read FAIL along a
    direction, or UNRESOLVED, found no word for it on either page."""
    from woof.hex import ad_doors

    directions = (ad_doors.load_table()["doors"]["adjoint-test"].get("directions") or {}).get("order")
    if not directions:
        pytest.skip("the table of this tree names no test directions: its adjoint-test runs the bump alone")
    for page in (README, ROOT / "docs" / "differentiable" / "README.md"):
        text = page.read_text(encoding="utf-8")
        missing = [name for name in directions if f"`{name}`" not in text]
        assert not missing, (page.name, missing)
        assert "`--directions`" in text and "UNRESOLVED" in text, page.name
        assert "PASS or FAIL against fixed bars" not in text, page.name


#: The release whose derivative engine library and pinned downloads this one carries unchanged.
ENGINE_RELEASE = "0.6.0"


def test_every_text_that_names_this_release_names_the_declared_version():
    """THE BREAKAGE THIS PREVENTS (a sealed candidate, 2026-11-14): a candidate was stamped with the patch number
    after its release's own, and the package, the engine crate, the pin tag, the changelog heading and forty README
    lines all had to move back.  One number is declared in pyproject.toml; every text that names this release names
    that one, and the changelog has one entry for it."""
    readme = README.read_text(encoding="utf-8")
    assert f'pip install "recast-woof[ad]=={VERSION}"' in readme
    assert f"**New in {VERSION}:**" in readme
    assert set(re.findall(r"/gpuwm-hex/blob/v(\d+\.\d+\.\d+)/", readme)) == {VERSION}
    assert f'pip install "recast-woof[ad]=={VERSION}"' in (ROOT / "docs/differentiable/README.md").read_text(encoding="utf-8")
    changelog = ROOT / "CHANGELOG.md"
    if changelog.is_file():  # the repository carries it; the source package does not
        headings = re.findall(r"^## (\d+\.\d+\.\d+)$", changelog.read_text(encoding="utf-8"), re.M)
        assert headings[0] == VERSION and len(set(headings)) == len(headings), headings[:3]
        numbers = [tuple(map(int, h.split("."))) for h in headings]
        assert numbers == sorted(numbers, reverse=True), "newest entry first, one entry a release"
    # 0.6.1 and 0.6.2 are defect releases: they keep the files 0.6.0 published (the pinned downloads) and the derivative engine
    # library 0.6.0 built, so the pin tag and the engine crate stay at ENGINE_RELEASE while the package moves.
    pins = (ROOT / "../../woof/hex/data/ad-assets.json").read_text(encoding="utf-8")
    assert f'"tag": "ad-assets-v{ENGINE_RELEASE}"' in pins
    crate = ROOT / "rust/hexad/Cargo.toml"
    if crate.is_file():
        assert tomllib.loads(crate.read_text(encoding="utf-8"))["package"]["version"] == ENGINE_RELEASE
        lock = (ROOT / "rust/hexad/Cargo.lock").read_text(encoding="utf-8")
        assert f'name = "hexad-phys"\nversion = "{ENGINE_RELEASE}"' in lock
    helpers = ROOT / "licenses/THIRD-PARTY-LICENSES-helper-binaries.txt"
    assert f"  hexad-phys {ENGINE_RELEASE}\n" in helpers.read_text(encoding="utf-8")


def test_the_source_package_carries_the_changelog_this_file_reads():
    """THE BREAKAGE (the 0.6.0 candidate's source package, 2026-11-15): the test above reads CHANGELOG.md, and from
    an unpacked source package it failed with FileNotFoundError: MANIFEST.in did not include the file."""
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    assert "include CHANGELOG.md" in manifest
    assert (ROOT / "CHANGELOG.md").is_file()
