"""Phase ordering and workspace bounds without acquiring a device."""
import sys
from types import SimpleNamespace

import pytest

from woof.hex import numerics
from woof.hex.mpasphys import thompson_v842 as port
from woof.hex.mpasphys.thompson_phases_v842 import (CARRY_BYTES_PER_COLUMN, CELL_PHASES, KERNELS, METHOD_CELL_PHASES,
                                                    PHASE_GROUPS, TAIL_PHASE, WARP_GROUPS, WARP_PHASE)


@pytest.mark.parametrize('mode,workspace,expected', [
    ('fast', True, len(KERNELS)), ('strict', True, 1), ('fast', False, 1)])
def test_ordered_tiles_use_one_allocation_and_no_host_reads(monkeypatch, mode, workspace, expected):
    calls, allocations = [], []
    def allocate(size, *, dtype):
        allocations.append(size)
        return object()
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(empty=allocate, uint8='uint8'))
    monkeypatch.setattr(numerics, 'mode', lambda: mode)
    monkeypatch.setattr(port, 'workspace_tile_columns', lambda: 64)
    monkeypatch.setattr(port, 'resident_columns', lambda kernel, block_threads=32: 64)
    def kernel(name):
        def launch(grid, block, args):
            calls.append((name, grid, block, args))
        return launch
    module = SimpleNamespace(get_function=kernel)
    port._launch_columns('mpas_thompson_step', module, ('state',), 55, 97, workspace)
    assert len(allocations) == int(workspace)
    assert len(calls) == expected * (2 if workspace else 1)
    if not workspace:
        assert calls == [('mpas_thompson_step', (4,), (32,), ('state',))]
        return
    assert allocations == [port.workspace_bytes(port.workspace_layout(), 'mpas_thompson_step', 55, 64)]
    for tile, (start, count) in enumerate(((0, 64), (64, 33))):
        for phase in range(expected):
            name, grid, block, args = calls[tile*expected+phase]
            threads = 128 if expected > 1 else 32
            cells = count*55 if expected > 1 and phase in CELL_PHASES else count
            cells = count*32 if expected > 1 and phase in WARP_GROUPS else cells
            assert name == KERNELS[phase]
            assert grid == ((cells+threads-1)//threads,)
            assert block == (threads,)
            assert args[0] == 'state'
            assert tuple(map(int, args[-2:])) == (start, count)
            assert args[1] is calls[0][3][1]


def test_carry_size_includes_partial_warp(monkeypatch):
    monkeypatch.setattr(numerics, 'mode', lambda: 'strict')
    original = port.workspace_bytes(port.workspace_layout(), 'mpas_thompson_step', 55, 33)
    monkeypatch.setattr(numerics, 'mode', lambda: 'fast')
    assert port.workspace_bytes(port.workspace_layout(), 'mpas_thompson_step', 55, 33) == 2*original+128*CARRY_BYTES_PER_COLUMN


def test_aerosol_keeps_column_method(monkeypatch):
    from woof.hex.mpasphys import thompson_aero_v842
    monkeypatch.setattr(numerics, 'mode', lambda: 'fast')
    source = thompson_aero_v842.cuda_source(device_math_source='// test maths', max_levels=55)
    assert 'void mp_thompson(' in source
    assert 'mp_thompson_phase' not in source


def test_aerosol_takes_the_shared_workspace_and_phases(monkeypatch):
    """Thompson-aerosol runs mp_thompson's workspace launch (no 46 KB frame, no 128-column batches)."""
    from woof.hex.mpasphys import thompson_aero_v842
    monkeypatch.setattr(numerics, 'mode', lambda: 'fast')
    source = thompson_aero_v842.cuda_source(device_math_source='// test maths', max_levels=55, workspace=True)
    assert 'static constexpr bool is_aerosol_aware = true;' in source
    assert 'THOMPSON_COLUMN_KMAX]' not in source
    for name in KERNELS:
        assert source.count('void '+name+'(') == 1
    # the aerosol carriers and surface emission ride the driver commit phase
    tail = source[source.index('void '+KERNELS[-1]+'('):]
    assert 'nwfa2d[col]*dt' in tail and 'nifa[idx]=nifa1d[k]' in tail
    assert thompson_aero_v842.workspace_layout() == port.workspace_layout()
    monkeypatch.setattr(numerics, 'mode', lambda: 'strict')
    strict = thompson_aero_v842.cuda_source(device_math_source='// test maths', max_levels=55, workspace=True)
    assert 'mp_thompson_phase' not in strict and 'THOMPSON_COLUMN_KMAX]' not in strict


def test_phase_boundaries_do_not_retain_column_arrays():
    source = port.cuda_source(device_math_source='// test maths', max_levels=55,
                              workspace=True, numerics='fast')
    for name in KERNELS:
        assert source.count('void '+name+'(') == 1
    assert 'float* carry;' in source
    assert 'THOMPSON_COLUMN_KMAX]' not in source
    assert 'ws.carry[0]=no_micro?1.0f:0.0f;' in source


def test_serial_scans_remain_ordered_and_initialization_is_cell_local():
    source = port.cuda_source(device_math_source='// test maths', max_levels=55,
                              workspace=True, numerics='fast')
    head = source.index('void mp_thompson_phase(')
    brace = source.index('{', head)
    method = source[brace:port._body_end(source, brace)]
    for phase in METHOD_CELL_PHASES - {5, 6, 7, 8, 9}:
        start = method.index(f'constexpr(Phase=={phase})')
        brace = method.index('{', start)
        body = method[brace:port._body_end(method, brace)]
        assert 'for(k=kts;' not in body
        assert 'for(k=kte;' not in body
    for phase in (3, 10):
        # the graupel intercept pass: k_0 and the n0_min running minimum taken across the warp
        start = method.index(f'constexpr(Phase=={phase})')
        brace = method.index('{', start)
        body = method[brace:port._body_end(method, brace)]
        assert 'for(k=' not in body
        assert 'k_0=tc_warp_max(k_0);' in body
        assert body.count('tc_warp_running_min(tcs_n0,n0_min,tcs_lane,kte-kts+1);') == 1
        assert 'n0_min = tc_min(n0_exp,n0_min);' not in body and 'n0_exp = n0_min;' in body
    head = source.index('void mpas_thompson_step(')
    brace = source.index('{', head)
    kernel = source[brace:port._body_end(source, brace)]
    assert 'for(int k=tc_k-1;k<tc_k;k++)' in kernel
    assert 'for(int k=0;k<nz;k++)' not in kernel


def test_fused_slab_keeps_process_order_and_serial_barriers():
    source = port.cuda_source(device_math_source='// test maths', max_levels=55,
                              workspace=True, numerics='fast')
    assert tuple(p for group in PHASE_GROUPS for p in group) == tuple(range(TAIL_PHASE+1))
    for index, (name, group) in enumerate(zip(KERNELS, PHASE_GROUPS)):
        head = source.index('void '+name+'(')
        brace = source.index('{', head)
        kernel = source[brace:port._body_end(source, brace)]
        if group == (TAIL_PHASE,):
            continue  # Driver commit calls no process method.
        import re
        calls = tuple(map(int, re.findall(r'm\.mp_thompson_phase<(\d+)>', kernel)))
        assert calls == ((4,) if group == (4, 5, 6, 7, 8, 9) else group)
        assert index in CELL_PHASES or len(group) == 1
    assert source.count('__launch_bounds__(128, 4)') == len(KERNELS)-1-len(WARP_GROUPS)
    assert source.count('__launch_bounds__(128, 5)') == 1
    assert f'__launch_bounds__(128, 5) void {KERNELS[4]}(' in source
    for index in WARP_GROUPS:
        assert f'__launch_bounds__(128, 12) void {KERNELS[index]}(' in source


def test_fast_workspace_uses_block_wide_level_major_planes():
    fast = port.cuda_source(device_math_source='// test maths', max_levels=55,
                            workspace=True, numerics='fast')
    strict = port.cuda_source(device_math_source='// test maths', max_levels=55,
                              workspace=True, numerics='strict')
    assert '#define TC_WS_LANES 128' in fast
    assert 'p[k*TC_WS_LANES]' in fast
    for name, offset in (('rain', 128), ('snow', 256), ('graupel', 384), ('ice', 512)):
        assert f'm.ws.carry[{offset}]={name};' in fast
        assert f'{name}=m.ws.carry[{offset}]' in fast
    assert 'int stride;' not in fast
    assert '__launch_bounds__(128, 4)' not in strict
    assert 'p[k*TC_WS_LANES]' in strict
    assert '(local/TC_WS_LANES)' in strict


def test_conservation_rates_are_typed_cell_scalars():
    import re
    source = port.cuda_source(device_math_source='// test maths', max_levels=55,
                              workspace=True, numerics='fast')
    rates = re.findall(r'double tc_cell_(\w+)=0.f;', source)
    assert len(rates) == len(set(rates)) == 23
    start = source.index('constexpr(Phase==4)')
    brace = source.index('{', start)
    body = source[brace:port._body_end(source, brace)]
    assert all('tc_cell_' + name in body for name in rates)
    assert all(not re.search(r'\b' + name + r'\[', source) for name in rates)
    markers = (1940, 1951, 2022, 2572, 2674, 2931)
    positions = [body.index('// MPAS module_mp_thompson.F:' + str(n) + '\n')
                 for n in markers]
    assert positions == sorted(positions)


@pytest.mark.parametrize('field', ['prr_rcs', 'pnr_wau'])
def test_conservation_rate_view_cannot_escape_cell(field):
    from woof.hex.mpasphys.thompson_phases_v842 import split_source
    base = port.cuda_source(device_math_source='// test maths', max_levels=55,
                            workspace=True, numerics='strict')
    before = field+'[0+tc_sub(k,(kts))*(1)] = tc_'
    assert before in base
    changed = base.replace(before, field+'[k+1] = tc_', 1)
    with pytest.raises(ValueError, match='nonlocal index: '+field):
        split_source(changed, port._body_end)


def test_sedimentation_runs_one_warp_per_column():
    source = port.cuda_source(device_math_source='// test maths', max_levels=55,
                              workspace=True, numerics='fast')
    start = source.index(f'constexpr(Phase=={WARP_PHASE})')
    brace = source.index('{', start)
    body = source[brace:port._body_end(source, brace)]
    # every species of the native sub-step text, none of its serial level loops
    for call in ('tc_sed_warp<true>(vtrk,', 'tc_sed_cloud_warp(vtck,', 'tc_sed_warp<true>(vtik,',
                 'tc_sed_warp<false>(vtsk,', 'tc_sed_warp<false>(vtgk,'):
        assert body.count(call) == 1
    assert 'for(n=1; n<=nstep; n+=1)' not in body
    assert 'for(k=kte; k>=kts; k+=(-1))' not in body
    # the fall-speed chains are resolved across the warp, and only the 500 m search stays serial
    for name in ('vtrk', 'vtnrk', 'vtik', 'vtnik', 'vtsk', 'vtgk'):
        assert body.count(f'tc_sed_chain_copy({name},') == 1
    assert body.count('nstep=tc_warp_max(nstep);') == 4
    assert body.count('for(k=') == 1 and 'for(k=kts; k<=tc_sub(kte,1); k+=1) {' in body
    assert sorted(PHASE_GROUPS[i] for i in WARP_GROUPS) == [(3,), (10,), (WARP_PHASE,)]
    index = PHASE_GROUPS.index((WARP_PHASE,))
    head = source.index('void '+KERNELS[index]+'(')
    kernel = source[source.index('{', head):port._body_end(source, source.index('{', head))]
    assert 'int tc_local=tc_index>>5;if(tc_local>=tc_n)return;' in kernel
    assert 'if((threadIdx.x&31)==0){m.ws.carry[128]=rain;' in kernel


def test_changed_sedimentation_transcription_is_refused():
    from woof.hex.mpasphys.thompson_phases_v842 import split_source
    base = port.cuda_source(device_math_source='// test maths', max_levels=55,
                            workspace=True, numerics='strict')
    before = 'sed_g[0+tc_sub(k,(kts))*(1)] = tc_mul(vtgk[0+tc_sub(k,(kts))*(1)],rg[0+tc_sub(k,(kts))*(1)]);'
    assert base.count(before) == 1
    changed = base.replace(before, before.replace('tc_mul(vtgk', 'tc_mul(1.01f*vtgk'), 1)
    with pytest.raises(ValueError, match='sedimentation transcription changed'):
        split_source(changed, port._body_end)


def test_auto_tile_is_capped_by_the_workspace_budget(monkeypatch):
    """A 164k-column rank asked for a 5.4 GB tile and ran a 32 GB card out of memory (2026-10-27)."""
    allocations, starts = [], []
    def allocate(size, *, dtype):
        allocations.append(size)
        return object()
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(empty=allocate, uint8='uint8'))
    monkeypatch.setattr(numerics, 'mode', lambda: 'fast')
    monkeypatch.setattr(port, 'workspace_tile_columns', lambda: None)
    monkeypatch.setattr(port, 'resident_columns', lambda kernel, block_threads=32: 1 << 20)
    monkeypatch.setenv('WOOF_HEX_THOMPSON_WORKSPACE_MIB', '256')
    def kernel(name):
        def launch(grid, block, args):
            if name == KERNELS[0]:
                starts.append(int(args[-2]))
        return launch
    columns = 164_000
    port._launch_columns('mpas_thompson_step', SimpleNamespace(get_function=kernel), ('state',), 55, columns, True)
    assert len(allocations) == 1 and allocations[0] <= 256 << 20
    stride = port.budget_tile_columns('mpas_thompson_step', 55)
    assert stride % 128 == 0 and starts == list(range(0, columns, stride))
