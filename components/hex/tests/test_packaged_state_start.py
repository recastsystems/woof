"""Refuse unavailable state before work, and preserve native producer soils.

Breakage: packaged dict rows used to fall back to four layers for RUC, and a
limited source could build a mesh whose cells have no meteorology.
"""
import pytest
from woof.hex.init_sources import load_table,resolve_source,admit_cycle,Cycle,InitSourceRefusal
from woof.hex.hrrr_intermediate import soil_set_for_suite
from woof.hex.packaged_intermediate import coverage_refusal


def test_packaged_row_soil_and_cycle_contract():
    row=resolve_source(load_table(),'rrfs-v1',coverage='limited_area')
    assert soil_set_for_suite(row,'hrrrv5') == 'nodes'
    assert soil_set_for_suite(row,None) == 'layers'
    for hour in (0,3,6,9,12,15,18,21):
        admit_cycle(row,Cycle.parse(f'2026-10-14T{hour:02}'))
    with pytest.raises(InitSourceRefusal):
        admit_cycle(row,Cycle.parse('2026-10-14T01'))
    with pytest.raises(InitSourceRefusal):
        admit_cycle(row,Cycle.parse('2026-08-11T00'))


def test_coverage_uses_the_row_before_mesh_or_download():
    row=resolve_source(load_table(),'rrfs',coverage='limited_area')
    assert coverage_refusal(row,(37,-96.5),300) is None
    refusal=coverage_refusal(row,(49,-125),300)
    assert refusal.startswith("This region reaches past RRFS v1's CONUS map by about ")
    assert refusal.endswith('Move it inside the map or start from GFS.')
    synthetic={**row,'id':'other','coverage_label':'the supplied map'}
    assert 'the supplied map' in coverage_refusal(synthetic,(49,-125),300)
