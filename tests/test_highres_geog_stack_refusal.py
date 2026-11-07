"""The high-resolution terrain path refuses BEFORE it downloads.

2.3.2's failure was not that a dependency could be missing.  It was the
ORDER: the only check for rasterio lived inside the mosaic step, which
runs after the tile fetch, so a missing library cost the user 160.7 MiB
of Copernicus downloads and then produced a bare ``RuntimeError``
traceback at exit 1 with no remedy in it.

Three properties are pinned here, and each one failed in 2.3.2:

- the refusal fires with the network provably untouched;
- it is a named :class:`HighresRefusal` carrying the exact install
  command, not a raw exception class;
- even the retired ``on_refuse = "fallback-30s"`` spelling cannot
  swallow it when an older caller supplies that value directly. A broken
  install must not hand back baseline terrain because a library is missing.

A fourth is pinned since the warp substrate moved to Rust, and it is the
opposite direction: an environment carrying ONLY the shipped default --
the ``static-fields`` library, no rasterio, no pyproj -- must NOT be
refused.  The old spelling of this gate asked "is rasterio importable?",
which after the flip refuses a perfectly good wheel install for missing
a library nothing on the default path reads.  A gate that fires on a
working configuration is not a stricter gate, it is a broken one.
"""
from __future__ import annotations

from datetime import date

import pytest

from woof.static import geog_stack, rust_bridge
from woof.static.highres_production import (HighresRefusal,
                                             apply_highres_statics,
                                             require_geography_stack)


class _Config:
    """Minimal stand-in for a validated [static.highres] block."""

    def __init__(self, on_refuse="error"):
        self.enabled = True
        self.cache_root = "unused-because-we-refuse-first"
        self.on_refuse = on_refuse
        self.terrain_source = "auto"
        self.fields = "auto"

    def echo(self):
        return {}


def _hide_python_stack(monkeypatch):
    """Make the pure-Python fallback's libraries look absent."""
    monkeypatch.setattr(geog_stack, "find_spec", lambda name: None)


def _hide_rust_engine(monkeypatch):
    """Make the default engine look unloadable."""
    monkeypatch.setattr(rust_bridge, "unavailable_reason",
                        lambda: "FileNotFoundError: staged for this test")


def _blind(monkeypatch):
    """No engine at all: neither the default nor its fallback can run."""
    _hide_rust_engine(monkeypatch)
    _hide_python_stack(monkeypatch)


def _exploding_urlopen(*args, **kwargs):
    raise AssertionError(
        "the network was touched before the dependency was checked")


@pytest.mark.parametrize("on_refuse", ["error", "fallback-30s"])
def test_no_engine_at_all_refuses_before_any_fetch(monkeypatch, on_refuse):
    _blind(monkeypatch)
    with pytest.raises(HighresRefusal) as caught:
        apply_highres_statics(
            {"HGT_M": None}, object(), config=_Config(on_refuse),
            domain_id=1, case_date=date(2024, 6, 1), landuse_attrs={},
            urlopen=_exploding_urlopen)
    assert caught.value.reason == "geography-stack-missing"


def test_the_selected_fallback_with_no_libraries_refuses(monkeypatch):
    """``WOOF_STATIC_PYTHON=1`` on a box with no rasterio is the same
    breakage wearing the other hat: the engine the caller ASKED for
    cannot run, and finding that out after the fetch is the 2.3.2 bug."""
    _hide_python_stack(monkeypatch)
    monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
    with pytest.raises(HighresRefusal) as caught:
        require_geography_stack()
    assert caught.value.reason == "geography-stack-missing"
    assert rust_bridge.STATIC_PYTHON_ENV in str(caught.value)


def test_the_shipped_default_alone_is_not_refused(monkeypatch):
    """The reachability half: a bare wheel install builds terrain.

    The default engine is the Rust static-fields library, staged by
    `woof fetch-bridges` into every wheel install.  rasterio and pyproj
    are the parity fallback and live in the [geog] extra.  If this gate
    refused that configuration, `pip install recast-woof` could not build
    high-resolution terrain -- which is exactly the 2.3.2 outcome this
    file exists to prevent, arrived at from the other side.
    """
    if rust_bridge.unavailable_reason() is not None:
        pytest.fail(
            "the Rust static-fields bridge is not loadable here, so this "
            "test cannot tell a working default from a broken one")
    _hide_python_stack(monkeypatch)
    require_geography_stack()  # must not raise


def test_the_refusal_names_the_exact_command_to_run(monkeypatch):
    _blind(monkeypatch)
    with pytest.raises(HighresRefusal) as caught:
        require_geography_stack()
    message = str(caught.value)
    # The remedy has to be pasteable, and it has to name BOTH ways out:
    # restore the default engine, or install the fallback's libraries.
    assert "woof fetch-bridges" in message
    assert "pip install --upgrade 'recast-woof[geog]'" in message
    assert "rasterio" in message and "pyproj" in message
    assert "woof doctor" in message


def test_fallback_30s_does_not_swallow_a_broken_install(monkeypatch):
    """A coverage policy must not silently answer an install question."""
    _blind(monkeypatch)
    with pytest.raises(HighresRefusal):
        apply_highres_statics(
            {"HGT_M": None}, object(), config=_Config("fallback-30s"),
            domain_id=1, case_date=date(2024, 6, 1), landuse_attrs={},
            urlopen=_exploding_urlopen)


def test_a_disabled_block_is_still_the_identity(monkeypatch):
    """Nothing is checked, fetched or printed when the block is off."""
    _blind(monkeypatch)

    class Off:
        enabled = False

    baseline = {"HGT_M": None}
    fields, receipt = apply_highres_statics(
        baseline, object(), config=Off(), domain_id=1,
        case_date=date(2024, 6, 1), landuse_attrs={},
        urlopen=_exploding_urlopen)
    assert fields is baseline
    assert receipt is None


def test_the_probe_reports_the_truth_on_this_box():
    """Instrument check: this checkout's dev environment has both.

    rasterio and pyproj are the [geog] extra now, so this is a statement
    about the development environment (which installs them to run the
    parity comparisons), not about what a user install must carry.  A
    venv built from `[dev]` alone does not carry the extra, and there
    the statement is about the box, not the probe: it skips naming the
    modules the probe found missing, and the probe's own mechanism is
    held by the vacuity test beside this one (proof/node-reds-276).
    """
    missing = geog_stack.missing_geog_modules()
    if missing:
        pytest.skip(
            f"the [geog] extra is not installed here: {', '.join(missing)} "
            "missing (pip install 'recast-woof[geog]'); the probe reported that "
            "truth and the vacuity test below holds its mechanism")
    assert missing == ()


def test_the_probe_is_not_vacuously_green(monkeypatch):
    """Both directions: it must also be able to report absence."""
    _hide_python_stack(monkeypatch)
    assert set(geog_stack.missing_geog_modules()) == {"rasterio", "pyproj"}
