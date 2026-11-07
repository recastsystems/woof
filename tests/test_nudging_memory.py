"""Admission must include the graph and both analysis times before allocation."""
from dataclasses import replace
from math import prod
from datetime import datetime

import pytest

from woof.core import preflight as pf
from woof.experiment import experiment_from_run_config
from woof.spectral_nudging.config import NudgingConfig, NudgingOptions
from woof.spectral_nudging.memory import memory_inventory
from woof.config import RunConfig


def experiment(*, use_theta_m=0):
    run = RunConfig(nx=96, ny=80, nz=16, dx=9000., dy=9000., ztop=16000.,
                    dt=6., run_seconds=12., specified=True,
                    open_x=False, open_y=False, moist=True, mp_physics=0,
                    use_theta_m=use_theta_m,
                    bl_pbl_physics=0, sf_sfclay_physics=0, sf_surface_physics=0,
                    ra_lw_physics=0, ra_sw_physics=0, cu_physics=0)
    return experiment_from_run_config(run, datetime(2026, 1, 1))


def enabled(exp, **kwargs):
    return replace(exp, nudging=NudgingOptions(((1, NudgingConfig(enabled=True, **kwargs)),)))


@pytest.mark.parametrize('mode,convention', [('spectral', 'regional'), ('spectral', 'wrf'), ('grid', 'wrf')])
@pytest.mark.parametrize('use_theta_m', [0, 1])
def test_selected_domain_prices_graph_endpoints_and_refresh(mode, convention, use_theta_m):
    exp = experiment(use_theta_m=use_theta_m)
    controls = NudgingConfig(enabled=True, mode=mode, convention=convention, gq=.001, fgdt=1.)
    bare = pf.estimate_experiment(exp)
    priced = pf.estimate_experiment(replace(exp, nudging=NudgingOptions(((1, controls),))))
    rows = list(memory_inventory(exp.root.run, controls))
    items = {item.name: item for item in priced.domains[0].items if item.name.startswith('nudging/')}
    assert items.keys() == {row[0] for row in rows}
    assert items['nudging/analysis_start/u'].shape == (16, 80, 97)
    assert items['nudging/analysis_end/v'].shape == (16, 81, 96)
    assert items['nudging/nudging_tendency_qv'].shape == (16, 80, 96)
    assert 'nudging/height/qv' not in items  # alias of temperature
    assert ('nudging/previous/qv' in items) == bool(use_theta_m)
    active = 5 if mode == 'spectral' else 4  # grid has no geopotential relaxation
    work = [item for name, item in items.items() if '/fft_workspace_bound/' in name]
    assert len(work) == (active if mode == 'spectral' else 0)
    assert priced.subtotal_bytes - bare.subtotal_bytes == sum(prod(shape) * size for _, _, shape, size, _ in rows)
    assert priced.alloc_estimate_bytes > bare.alloc_estimate_bytes


def test_off_and_unselected_domains_have_no_nudging_charge():
    exp = experiment()
    assert list(memory_inventory(exp.root.run, NudgingConfig())) == []
    assert pf.estimate_experiment(replace(exp, nudging=NudgingOptions(((2, NudgingConfig(enabled=True)),)))) == pf.estimate_experiment(exp)


def test_card_budget_that_used_to_pass_is_refused_before_allocation():
    exp = experiment()
    old = pf.estimate_experiment(exp).alloc_estimate_bytes
    nudged = enabled(exp)
    new = pf.estimate_experiment(nudged).alloc_estimate_bytes
    budget = (old + new) // 2
    # This is the production first lever. Even the smallest radiation chunk
    # cannot admit the omitted graph on this card; the caller must refuse.
    assert pf.recommend_column_chunk(exp, budget) is not None
    assert pf.recommend_column_chunk(nudged, budget) is None
