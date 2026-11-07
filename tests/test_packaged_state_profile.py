"""A state-only profile must carry echoes without enlarging the forecast profile.

Breakage: silently reusing the pressure-only profile initializes a clear model,
while modifying that shared profile expands every WOOF decode and fetch.
"""
import json
from woof.source_authorities import packaged_authorities
from woof.fetch import mapped_record_levels
from woof import fetch_routes
from datetime import datetime


def test_state_fields_are_opt_in_and_match_the_published_index_names():
    base = packaged_authorities('rrfs-prslev-2dfld-grib2-v1')
    state = packaged_authorities('rrfs-prslev-2dfld-state-grib2-v1')
    before = json.loads(base['mapping'].read_text())
    after = json.loads(state['mapping'].read_text())
    fields = {'CLMR': ('cloud_water_mixing_ratio',22),
              'ICMR': ('cloud_ice_mixing_ratio',23),
              'RWMR': ('rain_water_mixing_ratio',24),
              'SNMR': ('snow_mixing_ratio',25),
              'GRLE': ('graupel_or_hail_mixing_ratio',32)}
    original = mapped_record_levels('rrfs')
    selected = mapped_record_levels('rrfs', profile='rrfs-prslev-2dfld-state-grib2-v1')
    for idx_name,(name,param) in fields.items():
        assert name not in before['fields']
        selector = after['fields'][name]['selectors'][0]
        assert (selector['discipline'],selector['category'],selector['parameter'],selector['level_type']) == (0,1,param,100)
        assert idx_name not in original
        assert len(selected[idx_name]) == 45
    assert selected['RH'] == {'2 m above ground'}
    assert 'RH' not in original


def test_off_synoptic_cycles_fetch_the_published_pair_to_f018():
    # Breakage: excluding the four short cycles refuses a valid state start.
    for hour in (3,9,15,21):
        plan = fetch_routes.resolve_request('rrfs', cycle=datetime(2026,10,14,hour), hours=18)
        assert {obj.role for obj in plan.objects} == {'prslev','2dfld'}
        assert {obj.lead for obj in plan.objects} == set(range(19))
