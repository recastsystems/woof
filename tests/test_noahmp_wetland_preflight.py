"""Wetland stores must be admitted before either nested domain allocates."""
from dataclasses import replace
from math import prod

import pytest

from woof.config import RunConfig
from woof.core.preflight import physics_array_shapes, physics_field_names_2d


@pytest.mark.parametrize("nx,ny", [(14, 14), (18, 18)])
def test_wetland_prices_all_five_resident_planes_only_when_on(nx, ny):
    cfg = RunConfig(nx=nx, ny=ny, nz=10, dx=3000., dy=3000.,
                    ztop=10000., dt=12., run_seconds=120., sf_surface_physics=4)
    assert cfg.noahmp_wetland == 0
    off = physics_array_shapes(cfg)
    on_cfg = replace(cfg, noahmp_wetland=1)
    on = physics_array_shapes(on_cfg)
    names = {"wetstorexy", "wetfsatmx", "wetcap", "wetfsatxy", "wetlhxy"}
    expected = {f"fields/{name}": (ny, nx) for name in names}
    assert set(on) - set(off) == set(expected)
    assert all(on[name] == shape for name, shape in off.items())
    assert {name: on[name] for name in expected} == expected
    assert names <= set(physics_field_names_2d(on_cfg))
    assert names.isdisjoint(physics_field_names_2d(cfg))
    assert sum(4 * prod(shape) for name, shape in on.items() if name not in off) == 20 * nx * ny
