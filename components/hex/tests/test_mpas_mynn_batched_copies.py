"""Host-execute the CUDA indexing and real PBL commit map, including halos.

This checks raw copy words and field selection. It does not measure CUDA or
establish the strict forecast gate.
"""
import ctypes
import os
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys import bl_mynn_v842 as pbl
from woof.hex.mpasphys.mynn_copies_v842 import MynnCopies, kernel_source


@pytest.fixture
def copies(monkeypatch, tmp_path):
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('g++ required for host indexing witness')
    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: 0)),
                         empty=np.empty, empty_like=np.empty_like, zeros=np.zeros, ones=np.ones,
                         asarray=np.asarray, asfortranarray=np.asfortranarray, ndarray=np.ndarray,
                         float32=np.float32, int32=np.int32, uint8=np.uint8)
    copier = MynnCopies(cp)
    launches = []

    def compile_host(source):
        count = source.count('const unsigned int* s') - 1
        args, types = [], []
        for j in range(count):
            args.extend((f's{j}', f'd{j}', *(f'{v}{j}' for v in ('nr', 'nc', 'sr', 'sc', 'dr', 'dc')), f'fill{j}'))
            types.extend((ctypes.c_void_p, ctypes.c_void_p, *([ctypes.c_int] * 6), ctypes.c_uint))
        signature = source.split('mynn_copy_fields(', 1)[1].split(') {', 1)[0]
        host = '''#define __global__
struct Dim {int x,y;};
Dim blockIdx,blockDim,threadIdx;
''' + source + '\nextern "C" void witness(int blocks, int fields, ' + signature + ''') {
    blockDim={128,1};
    for(int f=0;f<fields;++f) for(int b=0;b<blocks;++b) for(int t=0;t<128;++t) {
        blockIdx={b,f};threadIdx={t,0}; mynn_copy_fields(''' + ','.join(args) + ''');
    }
}'''
        src, libpath = tmp_path / f'copy-{count}.cpp', tmp_path / f'copy-{count}.so'
        src.write_text(host)
        subprocess.run([compiler, '-std=c++17', '-O1', '-shared', '-fPIC', str(src),
                        '-o', str(libpath)], capture_output=True, text=True, check=True, timeout=30)
        lib = ctypes.CDLL(str(libpath))
        lib.witness.argtypes = [ctypes.c_int, ctypes.c_int, *types]

        def launch(grid, block, arguments):
            launches.append((grid, block))
            args = [a.ctypes.data if isinstance(a, np.ndarray) else int(a) for a in arguments]
            lib.witness(*grid, *args)
        return SimpleNamespace(get_function=lambda name: launch)

    monkeypatch.setattr(pbl, '_compile_device_module', compile_host)
    return copier, launches


def _words(shape, offset=0):
    bits = np.array([0, 0x80000000, 1, 0x7fc12345, 0x7f812345,
                     0xff800000, 0x3f800000, 0xdeadbeef], dtype=np.uint32)
    return np.array(np.resize(np.roll(bits, offset), shape).view(np.float32), order='F')


def test_copy_strided_fields_and_partial_blocks(copies):
    copier, launches = copies
    source = _words((7, 71))[:4, :67].T
    storage = np.full((7, 71), -123., dtype=np.float32, order='F')
    other = np.full(75, -1, dtype=np.int32)
    integer = np.arange(67, dtype=np.int32)[::-1]
    copier.copy_many(((storage[:4, :67].T, source), (other[2:69], integer)))
    np.testing.assert_array_equal(storage[:4, :67].T.view(np.uint32), source.view(np.uint32))
    np.testing.assert_array_equal(other[2:69], integer)
    assert (storage[4:] == -123).all() and (storage[:, 67:] == -123).all()
    assert (other[:2] == -1).all() and (other[69:] == -1).all()
    # One thread per destination unit-stride row (67), each walking the other axis.
    assert launches == [((1, 2), (128,))]


def test_input_diagnostic_pack_preserves_every_word(copies):
    copier, launches = copies
    adapter = object.__new__(pbl.MynnDriverAdapter)
    adapter.cp, adapter._copies = copier.cp, copier
    adapter._launch = lambda *args: None  # dx arithmetic is outside this copy witness.
    diagnostics = {n: _words((6, 71), j)[:4, :67] for j, n in enumerate(pbl._DIAGNOSTICS)}
    surface = {n: np.ones(67, np.float32) for n in ('ch', 'qsfc', 'rmol', 'skintemp')}
    result = adapter.from_mpas(diagnostics, surface, np.ones(67, np.float32), 480000.)
    for source, target in pbl._DIAGNOSTICS.items():
        np.testing.assert_array_equal(result[target].view(np.uint32), diagnostics[source].T.view(np.uint32))
    for name in (*pbl._BUDGETS, *pbl._ZERO_TENDENCIES, 'pattern_spp_pbl',
                 'maxwidth', 'maxmf', 'ztop_plume', 'ktop_plume'):
        assert not result[name].view(np.uint32).any()
    for source, target in (('ch','ch'), ('qsfc','qsfc'), ('rmol','rmol'), ('skintemp','ts')):
        np.testing.assert_array_equal(result[target], surface[source])
    assert launches == [((1, 41), (128,))]


def test_batched_zero_fill_including_empty_field(copies):
    copier, launches = copies
    storage = np.full((7, 71), -123., np.float32, order='F')
    integer = np.full(67, -1, np.int32)
    copier.copy_many(((storage[:4, :67], None), (integer, None),
                      (np.empty((0, 5), np.float32), None)))
    assert not storage[:4, :67].view(np.uint32).any()
    assert not integer.any()
    assert (storage[4:] == -123).all() and (storage[:, 67:] == -123).all()
    assert launches == [((1, 3), (128,))]


def test_copy_batch_raw_fill_and_legacy_parameter_limit(copies):
    copier, launches = copies
    # 85 fields fit the legacy 4096-byte parameter limit, including pointer
    # alignment. A larger independent batch needs two launches.
    fields = [np.empty(7, np.float32) for _ in range(86)]
    copier.copy_many([(field, np.uint32(0x7f812345)) for field in fields])
    for field in fields:
        np.testing.assert_array_equal(field.view(np.uint32), np.full(7, 0x7f812345, np.uint32))
    assert launches == [((1, 85), (128,)), ((1, 1), (128,))]
    integer = np.empty(7, np.int32)
    copier.copy_many([(integer[::-1], -1)])
    assert (integer == -1).all()


def test_copy_batch_rejects_unrepresentable_word_stride(copies):
    copier, launches = copies
    source = np.lib.stride_tricks.as_strided(np.zeros(1, np.float32), shape=(1,), strides=(1 << 33,))
    with pytest.raises(ValueError, match='signed 32-bit'):
        copier.copy_many([(np.empty(1, np.float32), source)])
    assert not launches


@pytest.mark.parametrize('prepared_order', ['C', 'F'])
@pytest.mark.parametrize('cloud_coupling', [False, True])
def test_native_binder_includes_private_surface_fields_in_input_batch(copies, prepared_order, cloud_coupling):
    copier, launches = copies
    copier.cp.asfortranarray = copier.cp.ones = lambda *a, **k: pytest.fail('individual layout/fill launch')
    ncol, nz = 67, 4
    adapter = object.__new__(pbl.MynnDriverAdapter)
    adapter.cp, adapter._copies = copier.cp, copier
    adapter._launch = lambda *args: None  # Mesh-dx arithmetic is separate.
    radiation_calls = []

    def radiation(lw, sw):
        radiation_calls.append((lw, sw))
        return np.asfortranarray((lw+sw).T)

    adapter.radiation = radiation
    prepared = {native: _words((nz+int(name == 'w'), ncol), j).copy(order=prepared_order)
                for j, (name, native) in enumerate(pbl._MYNN_PREPARED.items())}
    prepared['psfc_p'] = _words((ncol,))
    diag = {n: _words((nz+1, ncol), j) for j, n in enumerate(pbl._DIAGNOSTICS)}
    diag.update({n: _words((ncol,), j) for j, n in enumerate(
        ('ch','qsfc','rmol','hfx','qfx','ust','wspd','znt','hpbl'))})
    sfc = {n: _words((ncol,), j) for j, n in enumerate(('skintemp','xland'))}
    scalar = dict.fromkeys(pbl._MYNN_BOUNDS, 1)
    scalar.update(ite=ncol, kte=nz, kme=nz+1, dt_pbl=360.)
    scalar.update(dict.fromkeys(pbl._MYNN_FLAGS, True))
    # Padded and differently strided radiation inputs must reach the fused
    # layout kernel as views rather than being packed into temporary arrays.
    tendency = dict(rthratenlw=np.ones((nz+2,ncol+3),np.float32)[:, :ncol],
                    rthratensw=np.full((nz+2,ncol),2.,np.float32,order='F'))
    class Provider(dict):
        level = 1
        cp = copier.cp
        mynn_v842_resources = dict(device_math_source='approved source', driver=SimpleNamespace(adapter=adapter))
    provider = Provider(prepared=prepared, diag_physics=diag, sfc_input=sfc,
        scalar_state=scalar, tend_physics=tendency, constants=dict.fromkeys(pbl._MYNN_CONSTANTS,1.),
        configs=dict(config_do_restart=False, config_do_DAcycling=False, config_len_disp=480000.),
        mesh=dict(meshDensity=np.ones(ncol,np.float32)))
    call = SimpleNamespace(slot='pbl', scheme='bl_mynn', backend='hex-cuda', itimestep=2,
        options=dict(config_hex_icloud_bl=cloud_coupling, config_pbl_scheme='bl_mynn'))
    fields = pbl.bind_native_pools(provider, call)['mynn_v842']['arguments']
    assert fields['icloud_bl'] == int(cloud_coupling)
    packed = sum(value.ndim == 2 and not value.T.flags.f_contiguous for value in prepared.values())
    assert launches == [((1, 51+packed), (128,))]
    for name, native in pbl._MYNN_PREPARED.items():
        source = prepared[native]
        expected = source.T if source.ndim == 2 else source
        assert fields[name].flags.f_contiguous
        np.testing.assert_array_equal(fields[name].view(np.uint32), expected.view(np.uint32))
    for name, source in [(n,diag[n]) for n in ('hfx','qfx','ust','wspd','znt')] + [
            ('pblh',diag['hpbl']),('xland',sfc['xland'])]:
        np.testing.assert_array_equal(fields[name].view(np.uint32),source.view(np.uint32))
        assert not np.shares_memory(fields[name],source)
    assert not fields['uoce'].view(np.uint32).any() and not fields['voce'].view(np.uint32).any()
    assert (fields['kpbl'] == 1).all()
    assert len(radiation_calls) == 1
    for actual, name in zip(radiation_calls[0], ('rthratenlw','rthratensw')):
        assert np.shares_memory(actual,tendency[name]) and actual.shape == (nz,ncol)
    np.testing.assert_array_equal(fields['rthraten'],np.full((ncol,nz),3.,np.float32))


@pytest.fixture(scope='module')
def radiation_kernel(tmp_path_factory):
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('g++ required for radiation layout witness')

    def _body(source, name):
        start = source.index('{', source.index('extern "C" __global__ void '+name+'('))
        depth, end = 1, start+1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        return source[start+1:end-1]
    entries = []
    for name in ('mynn_radiation', 'mynn_layout', 'mynn_radiation_layout'):
        start = pbl.ADAPTER_CUDA_SOURCE.index('extern "C" __global__ void '+name+'(')
        signature = pbl.ADAPTER_CUDA_SOURCE[start:pbl.ADAPTER_CUDA_SOURCE.index('{',start)]
        entries.append(signature+'{'+_body(pbl.ADAPTER_CUDA_SOURCE,name)+'}')
    source = '''#define __global__
struct Dim {int x;};
Dim blockIdx,blockDim{128},threadIdx;
float mpas_add(float a,float b){volatile float v=a+b;return v;}
''' + '\n'.join(entries) + '''
extern "C" void original(int nc,int nk,const float* lw,const float* sw,float* tmp,float* out) {
    for(int n=0;n<nc*nk;++n) {blockIdx.x=n/128;threadIdx.x=n%128;
        mynn_radiation(nc*nk,lw,sw,tmp);
    }
    for(int n=0;n<nc*nk;++n) {blockIdx.x=n/128;threadIdx.x=n%128;
        mynn_layout(nc,nk,0,tmp,out);
    }
}
extern "C" void fused(int nc,int nk,const float* lw,const float* sw,int lr,int lc,int sr,int sc,float* out) {
    for(int n=0;n<((nc*nk+127)/128)*128;++n) {blockIdx.x=n/128;threadIdx.x=n%128;
        mynn_radiation_layout(nc,nk,lw,sw,lr,lc,sr,sc,out);
    }
}'''
    root = tmp_path_factory.mktemp('radiation-layout')
    path, library = root/'kernel.cpp', root/'kernel.so'
    path.write_text(source)
    subprocess.run([compiler,'-std=c++17','-O1','-shared','-fPIC','-ffp-contract=off',
                    '-fno-fast-math',str(path),'-o',str(library)],capture_output=True,text=True,check=True,timeout=30)
    lib = ctypes.CDLL(str(library))
    lib.original.argtypes = [ctypes.c_int]*2+[ctypes.c_void_p]*4
    lib.fused.argtypes = [ctypes.c_int]*2+[ctypes.c_void_p]*2+[ctypes.c_int]*4+[ctypes.c_void_p]
    return lib


@pytest.mark.parametrize('layout', ['C','F','strided'])
def test_radiation_sum_and_layout_match_original_words(radiation_kernel, layout):
    nz, ncol = 5, 67
    lw, sw = (_words((nz,ncol),j).copy(order='C' if layout == 'C' else 'F') for j in (1,3))
    if layout == 'strided':
        lw = _words((nz+2,ncol+3),1)[:nz,:ncol][:,::-1]
        sw = _words((nz+2,ncol+3),3)[:nz,:ncol]
    old = np.empty((ncol,nz),np.float32,order='F')
    packed_lw, packed_sw = np.asfortranarray(lw), np.asfortranarray(sw)
    temporary = np.empty_like(packed_lw,order='F')
    radiation_kernel.original(ncol,nz,*(a.ctypes.data for a in (packed_lw,packed_sw,temporary,old)))
    adapter = object.__new__(pbl.MynnDriverAdapter)
    adapter.cp = SimpleNamespace(empty=np.empty,float32=np.float32,ndarray=np.ndarray)
    launches = []
    def launch(name,count,args):
        launches.append(name)
        radiation_kernel.fused(*(a.ctypes.data if isinstance(a,np.ndarray) else a for a in args))
    adapter._launch = launch
    changed = adapter.radiation(lw,sw)
    np.testing.assert_array_equal(changed.view(np.uint32),old.view(np.uint32))
    assert launches == ['mynn_radiation_layout']


@pytest.mark.gpu
def test_cuda_radiation_sum_and_layout_words():
    if os.environ.get('MPAS_MYNN_FUSION_GPU') != '1':
        pytest.skip('requires approved GPU runner and MPAS_MYNN_FUSION_GPU=1')
    cp = pytest.importorskip('cupy')
    adapter = pbl.MynnDriverAdapter()
    nz, ncol = 5, 67
    for order in ('C','F'):
        lw = cp.array(_words((nz+2,ncol+3),1),order=order)[:nz,:ncol][:,::-1]
        sw = cp.array(_words((nz+2,ncol+3),3),order=order)[:nz,:ncol]
        packed_lw, packed_sw = cp.asfortranarray(lw), cp.asfortranarray(sw)
        temporary = cp.empty_like(packed_lw,order='F')
        adapter._launch('mynn_radiation',temporary.size,(temporary.size,packed_lw,packed_sw,temporary))
        original = adapter.layout(temporary)
        changed = adapter.radiation(lw,sw)
        np.testing.assert_array_equal(cp.asnumpy(changed).view(np.uint32),
                                      cp.asnumpy(original).view(np.uint32))


@pytest.mark.parametrize('flags', [(False, False, False, False),
                                  (True, True, False, True), (True, True, True, True)])
def test_complete_commit_matches_individual_native_copies(copies, flags):
    copier, launches = copies
    ncol, nz = 67, 4
    fields = dict(kts=1, kte=nz, **dict(zip(('f_ni', 'f_nc', 'f_nifa', 'f_nwfa'), flags)))
    diag_names = (*pbl._DIAGNOSTICS, *pbl._BUDGETS, 'kzh', 'kzm', 'kzq')
    tendencies = ('rqsblten', 'rniblten', 'rncblten', 'rnifablten', 'rnwfablten',
                  'rublten', 'rvblten', 'rthblten', 'rqvblten', 'rqcblten', 'rqiblten')
    diag = {n: np.full((nz+2, ncol+3), -123., np.float32, order='F') for n in diag_names}
    diag.update(hpbl=np.full(ncol+3, -123., np.float32), kpbl=np.full(ncol+3, -123, np.int32))
    tend = {n: np.full((nz+2, ncol+3), -123., np.float32, order='F') for n in tendencies}
    diag_view = {n: v[..., :ncol] for n, v in diag.items()}
    tend_view = {n: v[:, :ncol] for n, v in tend.items()}
    names = (*pbl._DIAGNOSTICS.values(), *pbl._BUDGETS, *tendencies, 'exch_h', 'exch_m')
    fields.update({n: _words((ncol, nz+1), j) for j, n in enumerate(names)})
    fields.update(pblh=_words((ncol,)), kpbl=np.arange(ncol, dtype=np.int32))
    expected_diag, expected_tend = ({n: v.copy() for n, v in d.items()} for d in (diag, tend))
    # Original adapter.to_mpas followed by commit's individual copy-outs.
    for native, field in (*pbl._DIAGNOSTICS.items(), *((n, n) for n in pbl._BUDGETS)):
        expected_diag[native][:nz, :ncol] = fields[field][:, :nz].T
    extra = ['rqsblten'] + (['rniblten'] if flags[0] else [])
    if all(flags[1:]):
        extra += ['rncblten', 'rnifablten', 'rnwfablten']
    for n in (*extra, *tendencies[5:]):
        expected_tend[n][:nz, :ncol] = fields[n][:, :nz].T
    for native, field in (('kzh', 'exch_h'), ('kzm', 'exch_m'), ('kzq', 'exch_h')):
        expected_diag[native][:nz, :ncol] = fields[field][:, :nz].T
    expected_diag['hpbl'][:ncol] = fields['pblh']
    expected_diag['kpbl'][:ncol] = fields['kpbl']
    provider = {'diag_physics': diag_view, 'tend_physics': tend_view}
    class Resident(dict):
        resident = True
    pbl.commit_native_outputs(Resident(provider), SimpleNamespace(adapter=copier), fields)
    for actual, expected in ((diag, expected_diag), (tend, expected_tend)):
        for name in actual:
            np.testing.assert_array_equal(actual[name].view(np.uint32), expected[name].view(np.uint32), err_msg=name)
    assert len(launches) == 1


@pytest.mark.gpu
def test_cuda_batched_strided_copies():
    if os.environ.get('MPAS_MYNN_FUSION_GPU') != '1':
        pytest.skip('requires approved GPU runner and MPAS_MYNN_FUSION_GPU=1')
    cp = pytest.importorskip('cupy')
    source = cp.array(_words((7, 71)), order='F')[:4, :67].T
    storage = cp.full((7, 71), -123., dtype=cp.float32, order='F')
    integer = cp.arange(67, dtype=cp.int32)[::-1]
    other = cp.full(75, -1, cp.int32)
    zero = cp.full((67, 4), -1., cp.float32, order='F')
    MynnCopies(cp).copy_many(((storage[:4, :67].T, source), (other[2:69], integer), (zero, None)))
    np.testing.assert_array_equal(cp.asnumpy(storage[:4, :67].T).view(np.uint32),
                                  cp.asnumpy(source).view(np.uint32))
    np.testing.assert_array_equal(cp.asnumpy(other[2:69]), np.arange(67, dtype=np.int32)[::-1])
    assert (cp.asnumpy(storage[4:]) == -123).all()
    assert (cp.asnumpy(storage[:, 67:]) == -123).all()
    assert (cp.asnumpy(other[:2]) == -1).all() and (cp.asnumpy(other[69:]) == -1).all()
    assert not cp.asnumpy(zero).view(np.uint32).any()
    # Exercise the largest legacy launch descriptor and a second batch on
    # CUDA, where the parameter ABI differs from the host indexing witness.
    fields = [cp.empty(7, cp.float32) for _ in range(86)]
    MynnCopies(cp).copy_many([(field, np.uint32(0x7f812345)) for field in fields])
    for field in fields:
        np.testing.assert_array_equal(cp.asnumpy(field).view(np.uint32),
                                      np.full(7, 0x7f812345, np.uint32))
    MynnCopies(cp).copy_many(((other[::-1], -1),))
    assert (cp.asnumpy(other) == -1).all()


@pytest.mark.parametrize('enabled,budgets_complete', [(False, True), (True, False), (True, True)])
def test_wrapper_batches_preserve_copy_and_optional_output_contract(copies, enabled, budgets_complete):
    copier, launches = copies
    copier.cp.asfortranarray = lambda *a, **k: pytest.fail('individual layout launch')
    ncol, nz = 67, 4
    arguments = dict(its=1, ite=ncol, kts=1, kte=nz, kme=nz+1,
                     spp_pbl=int(enabled), delt=360., errmsg=np.zeros(512, np.uint8))
    for name, spec in pbl.MYNN_ARGUMENTS.items():
        source = pbl.WRAPPER_TO_MMM.get(name, name)
        if spec['dimensions']:
            dtype = np.int32 if spec['kind'] == 'int' else np.float32
            shape = (ncol,) if len(spec['dimensions']) == 1 else (ncol, nz+1)
            arguments.setdefault(source, np.full(shape, -123, dtype=dtype, order='F'))
        elif source not in arguments:
            arguments[source] = 0
    for name in pbl._MYNN_FLAGS:
        arguments[name] = enabled
    for _, (source, _) in pbl._OPTIONAL_SPECIES.items():
        arguments[source] = np.full((ncol, nz+1), .125, np.float32, order='C')
    arguments['qv'] = np.full((ncol, nz+1), .25, np.float32, order='C')
    arguments['pattern_spp'] = np.full((ncol, nz+1), .5, np.float32, order='F')
    for name in arguments:
        if name.startswith('edmf_') and name.endswith('_dd'):
            arguments[name] = None
    if not budgets_complete:
        arguments[pbl._BUDGETS[0]] = None
    before = {n: a.copy() for n, a in arguments.items() if isinstance(a, np.ndarray)}
    expected = {}
    in_place = set()

    def pre(mixing, specific, flags, **kw):
        for name, flag in zip(('qc', 'qi', 'qs'), flags):
            assert (mixing[name] == (.125 if flag else 0)).all()
        for name in specific:
            specific[name].fill(.0625)

    def numerical(values, constants):
        for name in pbl._OPTIONAL_TENDENCIES:
            assert not values[name].view(np.uint32).any()
        assert (values['pattern_spp_pbl'] == (.5 if enabled else 0)).all()
        for name, (source, flag) in pbl._OPTIONAL_SPECIES.items():
            if not name.startswith('sq'):
                assert (values[name] == (.125 if arguments[flag] else 0)).all()
        for j, (name, spec) in enumerate(pbl.MYNN_ARGUMENTS.items()):
            if not spec['dimensions'] or spec['intent'] not in ('out','inout') or values.get(name) is None:
                continue
            source = pbl.WRAPPER_TO_MMM.get(name, name)
            if source in before and np.shares_memory(values[name], arguments[source]):
                # In place only on the caller's own Fortran tile (native passes it by reference).
                assert values[name].flags.f_contiguous and values[name].shape[0] == ncol
                in_place.add(source)
            values[name].fill(j+1)
            expected[source] = j+1
        return np.zeros(ncol, np.int32)

    def post(mixing, tendencies, flags, delt, **kw):
        for j, (name, field) in enumerate(tendencies.items()):
            field.fill(-300-j)
            expected[{'dqv':'rqvblten', 'dqc':'rqcblten', 'dqi':'rqiblten', 'dqs':'rqsblten'}[name]] = -300-j

    driver = object.__new__(pbl.MynnBlDriver)
    driver.adapter = SimpleNamespace(cp=copier.cp, copy_many=copier.copy_many, pre=pre, post=post)
    driver.numerical = SimpleNamespace(run=numerical)
    errors = driver.run(arguments, None, options=pbl.MynnOptions())
    assert not errors.any()
    assert len(launches) == 2
    # The F-ordered caller tiles go to MMM in place; the C-ordered species keep private copies.
    assert in_place and not in_place & {'qv', 'qc', 'qi', 'qs'}
    for name, array in before.items():
        target = arguments[name]
        if name == 'errmsg':
            assert (target == 32).all()
            continue
        mmm = next((n for n in pbl.MYNN_ARGUMENTS if pbl.WRAPPER_TO_MMM.get(n,n) == name), name)
        selected = name in expected
        if mmm in pbl._OPTIONAL_TENDENCIES and not enabled:
            selected = False
        if mmm in pbl._BUDGETS and not budgets_complete:
            selected = False
        if selected:
            if target.ndim == 2:
                assert (target[:, :nz] == expected[name]).all()
                np.testing.assert_array_equal(target[:, nz:].view(np.uint32), array[:, nz:].view(np.uint32))
            else:
                assert (target == expected[name]).all()
        else:
            np.testing.assert_array_equal(target.view(np.uint32), array.view(np.uint32), err_msg=name)


def test_input_diagnostic_pool_views_are_updated_in_place(copies):
    """Column-fastest (level, column) pool arrays reach MMM as views and the commit skips them."""
    copier, launches = copies
    adapter = object.__new__(pbl.MynnDriverAdapter)
    adapter.cp, adapter._copies = copier.cp, copier
    adapter._launch = lambda *args: None
    ncol, nz = 67, 4
    diagnostics = {n: np.ascontiguousarray(_words((nz, ncol), j)) for j, n in enumerate(pbl._DIAGNOSTICS)}
    surface = {n: np.ones(ncol, np.float32) for n in ('ch', 'qsfc', 'rmol', 'skintemp')}
    result = adapter.from_mpas(diagnostics, surface, np.ones(ncol, np.float32), 480000.)
    for source, target in pbl._DIAGNOSTICS.items():
        assert np.shares_memory(result[target], diagnostics[source])
        assert result[target].flags.f_contiguous and result[target].shape == (ncol, nz)
    assert launches == [((1, 41 - len(pbl._DIAGNOSTICS)), (128,))]
    # The commit has nothing to move for those fields: MMM already wrote the pool.
    launches.clear()
    fields = dict(kts=1, kte=nz, f_ni=False, f_nc=False, f_nifa=False, f_nwfa=False, **result)
    tendencies = ('rqsblten', 'rublten', 'rvblten', 'rthblten', 'rqvblten', 'rqcblten', 'rqiblten')
    for j, name in enumerate((*pbl._BUDGETS, *tendencies, 'exch_h', 'exch_m')):
        fields[name] = _words((ncol, nz), j)
    fields.update(pblh=_words((ncol,)), kpbl=np.arange(ncol, dtype=np.int32))
    diag = dict(diagnostics)
    diag.update({n: np.zeros((nz, ncol), np.float32) for n in (*pbl._BUDGETS, 'kzh', 'kzm', 'kzq')})
    diag.update(hpbl=np.zeros(ncol, np.float32), kpbl=np.zeros(ncol, np.int32))
    tend = {n: np.zeros((nz, ncol), np.float32) for n in tendencies}
    class Resident(dict):
        resident = True
    pbl.commit_native_outputs(Resident(diag_physics=diag, tend_physics=tend), SimpleNamespace(adapter=copier), fields)
    assert launches == [((1, len(pbl._BUDGETS) + len(tendencies) + 3 + 2), (128,))]
    for name in pbl._BUDGETS:
        np.testing.assert_array_equal(diag[name].view(np.uint32), fields[name].T.view(np.uint32))
