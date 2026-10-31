"""SMOKE at the shipping RRFSx 44c0849 pin (5d2c71e2) and the zero-copy pool views.

Breakage these prevent: (1) a default-pin smoke run binding the 9652bbf wrapper ABI (uthres_sg_in/albedo_drag/feff/
sep_in, no nwfa2d/nifa2d/config_mp_aero_emission) against 44c0849 pools; (2) a pool view passed with contiguous
Fortran strides when it is the provider's (k,i) words, which would read and write the wrong elements in place.
The bitwise proof is the strict replay of real native 44c0849 fire calls on the GPU
(tools/rrfs_smoke_replay/external_replay.py --pin 5d2c71e2).
"""
import shutil
import subprocess

import pytest


def test_default_pin_selects_the_5d2c71e2_graph():
    from woof.hex.physics_suites import load_table
    from woof.hex.mpasphys.rrfs import smoke_resident

    class Config:
        table_key = 'mpasphys_rrfs_44c0849'
    row = load_table()['mpasphys_rrfs_44c0849']
    assert row['slots']['smoke']['cuda_entries']['on']['module'] == 'rrfs.smoke_resident'
    assert tuple(row['outer_scalar_groups']) == ('chemistry',)
    smoke = smoke_resident.pinned(Config())
    assert smoke.SMOKE_PIN == row['submodule_pins']['SMOKE'] == '5d2c71e2cf4083b6beeedbfa4d2091674b1556ea'
    Config.table_key = 'mpasphys_rrfs'
    assert smoke_resident.pinned(Config()).SMOKE_PIN.startswith('9652bbf')


def test_5d2c71e2_wrapper_abi_and_lowering():
    from woof.hex.mpasphys.rrfs import smoke_9652bbf as old, smoke_5d2c71e2 as new
    args = new.ROUTINES['mpas_smoke_driver']['args']
    for name in ('uthres_sg_in', 'albedo_drag_in', 'feff_in', 'sep_in'):
        assert name not in args
    at = args.index('uthres_in')
    assert args[at:at + 3] == ['uthres_in', 'rdrag_in', 'ssm_in']
    at = args.index('ktop_deep')
    assert args[at:at + 4] == ['ktop_deep', 'nwfa2d', 'nifa2d', 'config_mp_aero_emission']
    # The generated declarations reproduce the 9652bbf graph exactly before they are swapped.
    for name in ('mpas_smoke_driver', 'gocart_dust_fengsha_driver', 'smoke_from_mpas_fields', 'smoke_to_mpas_fields'):
        assert old.CUDA_SOURCE.count(new.signature(name, old.ROUTINES[name]) + ' {') == 1
        assert new.CUDA_SOURCE.count(new.signature(name, new.ROUTINES[name]) + ' {') == 1
        assert old.CUDA_SOURCE.count(new.entry(name, old.ROUTINES[name])) == 1
    text = new.cuda_source('', columns=True)
    # Every phase carries the new automatic arrays in the same arena order; seasalt runs in inject, the TEMPO
    # aerosol emission in diagnostics.
    assert text.count('column_alloc<float>(ims,ime,kms,kme,jms,jme,1,num_e_ss_out)') == 7
    assert '__uint_as_float(0x57007A45U)' in text and '__uint_as_float(0x5006B7F3U)' in text
    assert new.DRIVER_BINDINGS['wrapper_bindings']['rdrag_in'] == 'rdrag_p'
    assert 'feff_in' not in new.DRIVER_BINDINGS['wrapper_bindings']


def test_pool_view_descriptor_addresses_provider_words(tmp_path):
    from woof.hex.mpasphys.rrfs import smoke_9652bbf as smoke
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    source = smoke.CUDA_SOURCE
    head = source[:source.index('struct FStr {')].replace('#include <cuda/barrier>\n', '')
    start = source.index('struct Argument {')
    arguments = source[start:source.index('// MPAS-Model-rrfs/src/core_atmosphere', start)]
    body = '''
    // chem(n,k,i) as the provider's C-order (species, level, cell) words: element strides (L*C, C, 1).
    const int S=3,L=4,C=5; float words[S*L*C]; for(int i=0;i<S*L*C;i++) words[i]=float(i);
    Argument a{}; a.data=words; for(int d=0;d<3;d++){a.lo[d]=1;} a.hi[0]=S;a.hi[1]=L;a.hi[2]=C;
    a.st[0]=L*C;a.st[1]=C;a.st[2]=1;
    auto chem=sd_arg_float(a,3);
    for(int n=1;n<=S;n++)for(int k=1;k<=L;k++)for(int i=1;i<=C;i++) assert(chem(n,k,i)==float((n-1)*L*C+(k-1)*C+(i-1)));
    Argument f{}; f.data=words; f.lo[0]=f.lo[1]=1; f.hi[0]=L; f.hi[1]=C;
    auto fortran=sd_arg_float(f,2);
    for(int k=1;k<=L;k++)for(int i=1;i<=C;i++) assert(fortran(k,i)==float((k-1)+(i-1)*L));
    '''
    path = tmp_path / 'views.cpp'
    path.write_text('#include <cassert>\n#define SDDEV\n' + head + arguments + '\nint main(){\n' + body + '\n}\n')
    binary = tmp_path / 'views'
    subprocess.run([compiler, '-std=c++17', '-O2', str(path), '-o', str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=5)
    assert smoke._ARGUMENT_DTYPE.itemsize == 72


def test_column_slot_storage_is_sized_from_the_graph(monkeypatch):
    """Breakage: column slots held a fixed 16,384-word SAVE arena (49 KiB of it unused since the plume vectors
    moved into the Plume struct) and tiles sized from free memory alone, ~16 GiB of smoke storage at 3 km."""
    import sys
    import types
    from woof.hex.mpasphys.rrfs import smoke_9652bbf as old, smoke_5d2c71e2 as new
    for module in (old, new.graph):
        # initialize(): 5 sea-salt (5 bins), 2 radiation and 2 deposition (1000), 5 dust (5), 1 FENGSHA (13),
        # 4 salt (9) and 1 soil (30) tables = 4,129 words, rounded up to 64.
        assert module.column_persistent_words(module.cuda_source('', columns=True)) == 4160

    class Runtime:
        @staticmethod
        def memGetInfo():
            return 60 << 30, 96 << 30
    monkeypatch.delenv('WOOF_HEX_SMOKE_TILE_MIB', raising=False)
    monkeypatch.setitem(sys.modules, 'cupy', types.SimpleNamespace(cuda=types.SimpleNamespace(runtime=Runtime)))
    # mv3km (206,726 columns): four equal tiles of whole 128-column blocks, not three wide ones and a sliver.
    assert old._column_tile(206_726, 90_832) == 51_712
    assert old._column_tile(2_562, 90_832) == 2_562
    monkeypatch.setenv('WOOF_HEX_SMOKE_TILE_MIB', '4096')
    assert old._column_tile(206_726, 90_832) == 41_472
