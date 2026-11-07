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
    pinned = set(re.findall(r"github\.com/FahrenheitResearch/gpuwm-hex/blob/(v[^/]+)/", text))
    assert pinned == {f"v{VERSION}"}, pinned


def test_the_cu13_extra_cannot_resolve_an_nvrtc_without_a_pack():
    """A pack is bound to its NVRTC library bytes, so the ceiling is the newest pack's exact version: a
    ``<13.(N+1)`` cap would let a later 13.N patch release resolve and miss the pack (0.4.12's 13.5.37)."""
    extra = DECLARATION["project"]["optional-dependencies"]["gpu-cu13"]
    caps = [re.search(r"<=(13\.\d+\.\d+)", r) for r in extra if r.startswith("nvidia-cuda-nvrtc")]
    assert caps and caps[0], f"gpu-cu13 must cap NVRTC at the newest pinned CUDA 13 pack: {extra}"
    newest = max((row for version, _, row in _packs() if version[0] == 13), key=lambda row: tuple(row["compiler"]["nvrtc"]))
    assert newest["compiler"]["nvrtc_package"] == f"nvidia-cuda-nvrtc=={caps[0].group(1)}", (newest["compiler"], extra)


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
