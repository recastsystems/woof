"""Dealiasing is what a bare radar ingest does, and off is what gets spelled.

The four risk masks find SIGNATURES of aliasing, not aliasing.  A spatially
coherent fold covering a whole region has a present and plausible Nyquist,
speeds well inside the reject fraction, no in-cell spread and no gate-to-gate
jump in its interior: it passes every one of them and reaches the filter as a
smooth, plausible, wrong wind field.  The unfolder that excludes that case was
implemented, vendored and named the shipped default engine, and then left
behind a ``store_true`` flag on every door.  "Off is not a decision at all"
was the documented claim; it was a decision, made by omission, for every run
that did not know to ask.

Everything here is on parameter objects and parsers: no scipy, no Rust
bridge, no volume.
"""

from __future__ import annotations

import pytest


def _an_engine_can_run_here() -> bool:
    """Whether this install can dealias at all, asked the shipped way."""

    from woof.obs.dealias import ENGINES, engine_unavailable_reason

    return any(engine_unavailable_reason(engine) is None
               for engine in ENGINES)


def test_a_bare_superob_dealiases():
    """The default is the capability, resolved against this install."""

    from woof.obs.superob import SuperobParams

    params = SuperobParams()
    if not _an_engine_can_run_here():
        # Neither engine can run here: masking only, reached by descent
        # rather than by omission, and the file's own statement says so.
        assert params.dealias is None
        return
    assert params.dealias is not None

    from woof.obs.dealias import (DEFAULT_ENGINE_CHAIN, ENGINE_REGION_GLOBAL,
                                   ENGINE_VAD_REGION, first_available_engine)

    # The chain is stated, and its head is the shipped engine.
    assert DEFAULT_ENGINE_CHAIN == (ENGINE_REGION_GLOBAL, ENGINE_VAD_REGION)
    assert params.dealias.engine == first_available_engine()
    # Off is a thing a caller states now.
    assert SuperobParams(dealias=None).dealias is None


def test_the_default_resolves_down_a_stated_chain_and_never_dies_on_a_door():
    """A missing optional library is a descent, not a dead front door."""

    from woof.obs.dealias import (DealiasParams, ENGINE_REGION_GLOBAL,
                                   ENGINE_VAD_REGION,
                                   resolve_default_dealias)

    def nothing_available(engine):
        return f"{engine} is not staged in this test"

    def only_vad(engine):
        return None if engine == ENGINE_VAD_REGION else "no shared library"

    default = DealiasParams()
    assert default.engine == ENGINE_REGION_GLOBAL

    stepped = resolve_default_dealias(default, unavailable_reason=only_vad)
    assert stepped is not None and stepped.engine == ENGINE_VAD_REGION
    # The refinement default belongs to the engine, and the engine changed.
    assert stepped.refinement is False

    assert resolve_default_dealias(
        default, unavailable_reason=nothing_available) is None

    # A NAMED engine is honoured exactly, never resolved away.
    named = DealiasParams(engine=ENGINE_VAD_REGION)
    assert resolve_default_dealias(
        named, unavailable_reason=nothing_available) is named


def test_every_radar_door_defaults_dealias_on_and_spells_the_off_switch():
    import tools.obs_radar_grid_build as build
    import tools.obs_radar_grid_from_pack as from_pack

    minimal = {
        build: ["--site", "QQQQ", "--valid-time", "2026-08-05T04:00:00Z",
                "--grid-wrfout", "g", "--out", "o", "--work-dir", "w"],
        from_pack: ["--pack", "p.rdrpack", "--grid-wrfout", "g",
                    "--out", "o", "--max-range-km", "250",
                    "--max-elevation-deg", "20"],
    }
    for module, argv in minimal.items():
        parser = module.build_parser()
        assert parser.parse_args(argv).dealias is True, module.__name__
        assert parser.parse_args(argv + ["--no-dealias"]).dealias is False, \
            module.__name__


def test_both_doors_resolve_dealias_through_one_function():
    """One function, every door, or two doors resolve one install twice."""

    import woof.obs.cli as obs_cli
    import woof.obs.dealias as dealias
    import tools.obs_radar_grid_build as build
    import tools.obs_radar_grid_from_pack as from_pack

    assert build.dealias_params_from_args is dealias.dealias_params_from_args
    assert (from_pack.dealias_params_from_args
            is dealias.dealias_params_from_args)
    # The woof door reaches the same function rather than inlining its own
    # DealiasParams behind a scipy check that tested the wrong engine's
    # prerequisite.
    source = obs_cli._radar_grid.__doc__ or ""
    del source
    import inspect

    body = inspect.getsource(obs_cli._radar_grid)
    assert "dealias_params_from_args" in body
    assert "scipy_available" not in body
    assert "SCIPY_REMEDY" not in body


def test_a_named_engine_that_cannot_run_is_still_refused_by_name():
    from woof.obs.dealias import (DealiasParams, ENGINE_VAD_REGION,
                                   dealias_params_from_args)

    class _Args:
        dealias = True
        dealias_engine = ENGINE_VAD_REGION
        dealias_refinement = None

    with pytest.raises(SystemExit) as error:
        dealias_params_from_args(
            _Args(), DealiasParams,
            lambda engine: "that engine is not staged in this test")
    assert ENGINE_VAD_REGION in str(error.value)
    assert "not staged" in str(error.value)

