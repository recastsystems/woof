"""GF-Li on the lane layout (woof.hex.mpasphys.rrfs.cu_gfl_bf6a4b0): placement only, sized from the source's own allocations.

The breakage these guard: 25.3 GB against 15.5 GB and 1.59x MPAS physics per forecast hour on the 96 -> 3 km zoom globe
(RTX 5090, 2026-10-31), from a 311 KB scratch share per column and a second column-major copy of every field.
"""
import re

import pytest

from woof.hex.mpasphys.rrfs import cu_gfl_bf6a4b0 as gfl


def test_a_column_needs_what_the_real_runs_measured():
    """The real hrrrv5 zoom-globe runs (IFS and GFS, 59 levels) reported a deepest column of 45,136 bytes."""
    assert gfl.deepest(gfl._SCHEME_SOURCE, 59, kme=60) == 45136


def test_every_allocation_in_the_source_has_a_known_size():
    """A size the reader cannot evaluate would size the scratch short, and the allocator would trap at the first deep column."""
    sizes, calls = gfl.frames(gfl._SCHEME_SOURCE, 59, kme=60)
    allocations = len(re.findall(r'workspace\.allocate \(', gfl._SCHEME_SOURCE))
    parsed = len(gfl._ALLOCATE.findall(gfl._SCHEME_SOURCE))
    assert parsed == allocations
    assert sizes['cu_grell_freitas_li'] > 0 and sizes['cu_gfl_deep_run'] > sizes['cu_gfl_sh_run'] > 0
    assert 'cu_gfl_deep_run' in calls['cu_grell_freitas_li'] and 'cu_gfl_sh_run' in calls['cu_grell_freitas_li']


@pytest.mark.parametrize("levels", [30, 50, 59, 65, 90])
def test_the_bound_grows_with_levels_and_chemistry(levels):
    plain = gfl.deepest(gfl._SCHEME_SOURCE, levels)
    assert plain % 16 == 0
    assert gfl.deepest(gfl._SCHEME_SOURCE, levels + 5) > plain
    assert gfl.deepest(gfl._SCHEME_SOURCE, levels, 6) > plain


def lowered():
    return gfl.lower(gfl._SUPPORT, gfl._SCHEME_SOURCE, gfl._lane_entry())


def test_every_scheme_pointer_is_placed():
    source = lowered()
    scheme = source[source.index('GFS '):]
    assert not re.search(r'\bGFReal \* \w', source[source.index('void calc_cldfraction_monan'):source.index('gfl_column')])
    assert not re.search(r'\(char \*\) \w+ \+ \(unsigned long long\)', scheme)
    assert source.count('workspace.allocate_bytes (') == 4
    assert 'GFRawP zuh2' in source


def test_every_field_reaches_the_scheme_in_its_own_place():
    entry = gfl._lane_entry()
    call = entry[entry.index('cu_grell_freitas_li(workspace'):]
    call = call[:call.index('\n')]
    for index, name in enumerate(gfl.FIELDS):
        if name in gfl._SCALARS:
            assert f'&scalar_{name}' in call
        elif name in gfl.LEVEL_ARGUMENTS:
            assert f'GFS((GFReal*)(ptrs[{index}]+column*strides[{index}]),kstrides[{index}])' in call
        else:
            assert f'*)(ptrs[{index}]+column*strides[{index}])' in call
    # The same argument order as the source's own entry.
    original = gfl._ENTRY[gfl._ENTRY.index('cu_grell_freitas_li(workspace'):]
    original = original[:original.index('\n')]
    assert re.findall(r'ptrs\[(\d+)\]', re.sub(r'kstrides\[\d+\]', '', call.replace('&scalar_', ''))) != []
    assert call.endswith(original[original.index('&ims'):]) or original[original.index('&ims'):].rstrip(';') in call


def test_level_arguments_are_the_scheme_fields_not_on_the_surface():
    assert set(gfl.LEVEL_ARGUMENTS) <= set(gfl.FIELDS)
    assert not set(gfl.LEVEL_ARGUMENTS) & set(gfl.SURFACE_FIELDS)
    assert not set(gfl.LEVEL_ARGUMENTS) & set(gfl.SCALAR_FIELDS)
    assert set(gfl.FIELDS) == set(gfl.LEVEL_ARGUMENTS) | set(gfl.SURFACE_FIELDS) | set(gfl.SCALAR_FIELDS)
