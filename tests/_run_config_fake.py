"""A RunConfig stand-in that carries every RunConfig default.

Unit tests drive one driver method with a ``SimpleNamespace`` instead of a
whole validated ``RunConfig``.  A namespace that names only the fields the
test cares about breaks every time production starts reading one more
field (``use_theta_m``, ``sst_update``, ``ra_lw_physics``, ``cudt_minutes``
...), and the red says nothing about the behaviour under test.  This
namespace starts from ``RunConfig``'s own field defaults, so an unnamed
field reads exactly what a default production run reads, and the test's
keywords override them.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace


def run_config_defaults() -> dict:
    """Every RunConfig field that has a default, at that default."""

    from woof.config import RunConfig

    values = {}
    for field in dataclasses.fields(RunConfig):
        if field.default is not dataclasses.MISSING:
            values[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            values[field.name] = field.default_factory()
    return values


def run_config_fake(**values) -> SimpleNamespace:
    """A namespace with RunConfig's defaults, overridden by ``values``."""

    return SimpleNamespace(**{**run_config_defaults(), **values})
